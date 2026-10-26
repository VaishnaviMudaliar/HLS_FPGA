#include "preprocessing_hls.hpp"
#include <hls_stream.h>
#include <cmath>
#include <hls_math.h>

constexpr float PI = 3.14159265358979323846f;

// Struct to hold all design parameters for dataflow streaming
struct config_t {
    int input_size;
    int window_size;
    int stride;
    int n_win;
    WindowType window_type;
    NormalizationType normalization;
};

// -------------------------------------------------------------------
// Pipelined Single-Window Normalization with Expanded Arithmetic
// -------------------------------------------------------------------
void normalize_single_window_hls(
    data_t window[MAX_WINDOW_SIZE],
    int window_size,
    NormalizationType normalization)
{
#pragma HLS INLINE

    if (normalization == NormalizationType::NONE)
        return;

    // Expanded Reciprocal and Difference Types to prevent range/wrapping issues
    typedef ap_fixed<25,9>  wide_diff_t; // Range [-256, 255.99] to handle max span (-128 to +128)
    typedef ap_fixed<32,12> inv_t;       // Supports multipliers up to 2047 (e.g., stddev = 0.0005)

    if (normalization == NormalizationType::MIN_MAX)
    {
        const data_t EPSILON = 0.001f;
        data_t min_val = window[0];
        data_t max_val = window[0];

        MIN_MAX_LOOP: for (int i = 1; i < MAX_WINDOW_SIZE; i++)
        {
#pragma HLS PIPELINE II=1
            if (i < window_size) {
                if (window[i] < min_val) min_val = window[i];
                if (window[i] > max_val) max_val = window[i];
            }
        }

        // Widen range to ap_fixed<25,9> to allow max difference up to 256
        wide_diff_t range = (wide_diff_t)max_val - (wide_diff_t)min_val;
        if (range < (wide_diff_t)EPSILON) range = 1.0f;

        // Compute reciprocal using 32-bit wide type
        inv_t inv_range = (inv_t)1.0f / range;

        NORM_MINMAX_LOOP: for (int i = 0; i < MAX_WINDOW_SIZE; i++)
        {
#pragma HLS PIPELINE II=1
            if (i < window_size) {
                wide_diff_t diff = (wide_diff_t)window[i] - (wide_diff_t)min_val;
                // Compute normalized result with wide multiplier before casting back
                window[i] = (data_t)(diff * inv_range);
            }
        }
    }
    else if (normalization == NormalizationType::Z_SCORE)
    {
        const data_t EPSILON = 0.0001f;

        typedef ap_fixed<40,20> sum_t;
        sum_t sum = 0;

        MEAN_LOOP: for (int i = 0; i < MAX_WINDOW_SIZE; i++)
        {
#pragma HLS PIPELINE II=1
            if (i < window_size)
                sum += (sum_t)window[i];
        }

        data_t mean = (data_t)(sum / window_size);

        typedef ap_fixed<48,24> var_accum_t;
        var_accum_t var_sum = 0;

        VAR_LOOP: for (int i = 0; i < MAX_WINDOW_SIZE; i++)
        {
#pragma HLS PIPELINE II=1
            if (i < window_size) {
                wide_diff_t diff = (wide_diff_t)window[i] - (wide_diff_t)mean;
                var_sum += (var_accum_t)(diff * diff);
            }
        }

        var_accum_t variance = var_sum / window_size;
        data_t stddev = (data_t)hls::sqrt(variance);

        if (stddev < EPSILON) stddev = 1.0f;

        // Use wider inv_t type (ap_fixed<32,12>) so stddev = 0.005 (inv = 200) does not wrap ap_fixed<24,8>
        inv_t inv_stddev = (inv_t)1.0f / (inv_t)stddev;

        NORM_ZSCORE_LOOP: for (int i = 0; i < MAX_WINDOW_SIZE; i++)
        {
#pragma HLS PIPELINE II=1
            if (i < window_size) {
                wide_diff_t diff = (wide_diff_t)window[i] - (wide_diff_t)mean;
                window[i] = (data_t)(diff * inv_stddev);
            }
        }
    }
}
//==============================================================
// Dataflow Task 1: Read Memory
//==============================================================

static void read_input_task(
    const data_t input[MAX_INPUT_SIZE],
    hls::stream<data_t>& in_fifo,
    hls::stream<config_t>& cfg_fifo_out,
    config_t cfg)
{
#pragma HLS INLINE off

    // 1. Stream configuration down the DATAFLOW pipeline first
    cfg_fifo_out.write(cfg);

    // 2. Bound dynamic iteration count to avoid unnecessary loop overhead
    int read_count = (cfg.input_size > MAX_INPUT_SIZE) ? MAX_INPUT_SIZE : cfg.input_size;

    // 3. Single-pass pipelined memory read loop (II=1)
    READ_LOOP: for (int i = 0; i < read_count; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=4 max=1024 avg=256
        in_fifo.write(input[i]);
    }
}
//==============================================================
// Dataflow Task 2: Window Generator
//==============================================================

void window_gen_task(
    hls::stream<data_t>& in_stream,
    hls::stream<data_t>& window_out_stream,
    hls::stream<config_t>& config_in_stream,
    hls::stream<config_t>& config_out_stream)
{
#pragma HLS INLINE off

    config_t cfg = config_in_stream.read();
    config_out_stream.write(cfg);

    int input_size  = cfg.input_size;
    int window_size = cfg.window_size;
    int stride      = cfg.stride;

    if (input_size < window_size || stride <= 0) return;

    data_t line_buffer[MAX_WINDOW_SIZE];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete

    WINDOW_GEN_LOOP: for (int i = 0; i < input_size; i++)
    {
#pragma HLS PIPELINE II=1
        data_t sample = in_stream.read();

        SHIFT_LOOP: for (int j = 0; j < MAX_WINDOW_SIZE - 1; j++) {
#pragma HLS UNROLL
            line_buffer[j] = line_buffer[j + 1];
        }
        line_buffer[MAX_WINDOW_SIZE - 1] = sample;

        if (i >= window_size - 1 && ((i - (window_size - 1)) % stride == 0)) {
            STREAM_WINDOW: for (int w = 0; w < window_size; w++) {
#pragma HLS PIPELINE II=1
                window_out_stream.write(line_buffer[MAX_WINDOW_SIZE - window_size + w]);
            }
        }
    }
}
//==============================================================
// Dataflow Task 3: Apply Window Function
//==============================================================

static void apply_window_task(
    hls::stream<data_t>& window_fifo,
    hls::stream<data_t>& coeff_fifo,
    hls::stream<config_t>& cfg_fifo_in,
    hls::stream<config_t>& cfg_fifo_out)
{
#pragma HLS INLINE off

    config_t cfg = cfg_fifo_in.read();
    cfg_fifo_out.write(cfg);

    data_t coeff[MAX_WINDOW_SIZE];
#pragma HLS ARRAY_PARTITION variable=coeff complete

    // Inline window coefficient generation
    if (cfg.window_size > 0) {
        if (cfg.window_size == 1) {
            coeff[0] = 1.0f;
        } else {
            GEN_COEFF_LOOP: for (int i = 0; i < MAX_WINDOW_SIZE; i++) {
#pragma HLS PIPELINE II=1
                if (i < cfg.window_size) {
                    float angle = (2.0f * PI * static_cast<float>(i)) /
                                  static_cast<float>(cfg.window_size - 1);

                    switch (cfg.window_type) {
                        case WindowType::RECTANGULAR:
                            coeff[i] = 1.0f;
                            break;
                        case WindowType::HAMMING:
                            coeff[i] = 0.54f - 0.46f * hls::cos(angle);
                            break;
                        case WindowType::HANN:
                            coeff[i] = 0.5f * (1.0f - hls::cos(angle));
                            break;
                    }
                }
            }
        }
    }

    APPLY_OUTER: for (int w = 0; w < MAX_WINDOWS; w++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_WINDOWS
        if (w < cfg.n_win) {
            APPLY_INNER: for (int i = 0; i < MAX_WINDOW_SIZE; i++) {
#pragma HLS PIPELINE II=1
                if (i < cfg.window_size) {
                    data_t sample = window_fifo.read();
                    coeff_fifo.write(sample * coeff[i]);
                }
            }
        }
    }
}

//==============================================================
// Dataflow Task 4: Normalize & Write Back
//==============================================================

static void normalize_store_task(
    hls::stream<data_t>& coeff_fifo,
    data_t output[MAX_WINDOWS][MAX_WINDOW_SIZE],
    hls::stream<config_t>& cfg_fifo_in)
{
#pragma HLS INLINE off

    config_t cfg = cfg_fifo_in.read();

    data_t window_buf[MAX_WINDOW_SIZE];
#pragma HLS ARRAY_PARTITION variable=window_buf complete

    NORM_OUTER: for (int w = 0; w < MAX_WINDOWS; w++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_WINDOWS
        if (w < cfg.n_win) {
            NORM_READ: for (int i = 0; i < MAX_WINDOW_SIZE; i++) {
#pragma HLS PIPELINE II=1
                if (i < cfg.window_size) {
                    window_buf[i] = coeff_fifo.read();
                }
            }

            normalize_single_window_hls(window_buf, cfg.window_size, cfg.normalization);

            NORM_WRITE: for (int i = 0; i < MAX_WINDOW_SIZE; i++) {
#pragma HLS PIPELINE II=1
                if (i < cfg.window_size) {
                    output[w][i] = window_buf[i];
                }
            }
        }
    }
}

//==============================================================
// CSIM Compatibility Stubs
//==============================================================

void generate_window_coefficients(
    data_t coeff[MAX_WINDOW_SIZE],
    int window_size,
    WindowType window_type)
{
    if (window_size <= 0) return;
    if (window_size == 1) { coeff[0] = 1.0f; return; }

    for (int i = 0; i < window_size; i++) {
        float angle = (2.0f * PI * static_cast<float>(i)) /
                      static_cast<float>(window_size - 1);
        switch (window_type) {
            case WindowType::RECTANGULAR: coeff[i] = 1.0f; break;
            case WindowType::HAMMING:     coeff[i] = 0.54f - 0.46f * hls::cos(angle); break;
            case WindowType::HANN:        coeff[i] = 0.5f * (1.0f - hls::cos(angle)); break;
        }
    }
}

void sliding_window_hls(
    const data_t input[MAX_INPUT_SIZE],
    data_t windows[MAX_WINDOWS][MAX_WINDOW_SIZE],
    int input_size, int window_size, int stride, int &num_windows)
{
    num_windows = 0;
    if (window_size <= 0 || stride <= 0 || input_size < window_size) return;
    for (int start = 0; start + window_size <= input_size; start += stride) {
        for (int i = 0; i < window_size; i++) windows[num_windows][i] = input[start + i];
        num_windows++;
    }
}

void apply_window_hls(
    data_t windows[MAX_WINDOWS][MAX_WINDOW_SIZE],
    int num_windows, int window_size, WindowType window_type)
{
    data_t coeff[MAX_WINDOW_SIZE];
    generate_window_coefficients(coeff, window_size, window_type);
    for (int w = 0; w < num_windows; w++) {
        for (int i = 0; i < window_size; i++) windows[w][i] *= coeff[i];
    }
}

void min_max_normalize_hls(data_t windows[MAX_WINDOWS][MAX_WINDOW_SIZE], int num_windows, int window_size) {
    for (int w = 0; w < num_windows; w++) normalize_single_window_hls(windows[w], window_size, NormalizationType::MIN_MAX);
}

void zscore_normalize_hls(data_t windows[MAX_WINDOWS][MAX_WINDOW_SIZE], int num_windows, int window_size) {
    for (int w = 0; w < num_windows; w++) normalize_single_window_hls(windows[w], window_size, NormalizationType::Z_SCORE);
}

void normalize_windows_hls(data_t windows[MAX_WINDOWS][MAX_WINDOW_SIZE], int num_windows, int window_size, NormalizationType normalization) {
    for (int w = 0; w < num_windows; w++) normalize_single_window_hls(windows[w], window_size, normalization);
}

//==============================================================
// Top-Level Data flow Kernel
//==============================================================

void ml_preprocessing_hls(
    const data_t input[MAX_INPUT_SIZE],
    data_t output[MAX_WINDOWS][MAX_WINDOW_SIZE],
    int input_size,
    int window_size,
    int stride,
    WindowType window_type,
    NormalizationType normalization,
    int &num_windows)
{
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem depth=1024 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem depth=65536 max_write_burst_length=64

#pragma HLS INTERFACE s_axilite port=input_size bundle=control
#pragma HLS INTERFACE s_axilite port=window_size bundle=control
#pragma HLS INTERFACE s_axilite port=stride bundle=control
#pragma HLS INTERFACE s_axilite port=window_type bundle=control
#pragma HLS INTERFACE s_axilite port=normalization bundle=control
#pragma HLS INTERFACE s_axilite port=num_windows bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    if (input_size > MAX_INPUT_SIZE || input_size <= 0 ||
        window_size <= 0 || window_size > MAX_WINDOW_SIZE ||
        stride <= 0 || input_size < window_size)
    {
        num_windows = 0;
        return;
    }

    int n_win = ((input_size - window_size) / stride) + 1;
    num_windows = n_win;

    config_t cfg;
    cfg.input_size   = input_size;
    cfg.window_size  = window_size;
    cfg.stride       = stride;
    cfg.n_win        = n_win;
    cfg.window_type  = window_type;
    cfg.normalization = normalization;

    // Data FIFOs - Sized to prevent HLS 214-114 task stalls on large window counts
    hls::stream<data_t> in_fifo("in_fifo");
    hls::stream<data_t> window_fifo("window_fifo");
    hls::stream<data_t> coeff_fifo("coeff_fifo");

    // Configuration FIFOs (Point-to-Point Chain across DATAFLOW stages)
    hls::stream<config_t> cfg_c1("cfg_c1");
    hls::stream<config_t> cfg_c2("cfg_c2");
    hls::stream<config_t> cfg_c3("cfg_c3");

#pragma HLS STREAM variable=in_fifo depth=1024
#pragma HLS STREAM variable=window_fifo depth=4096
#pragma HLS STREAM variable=coeff_fifo depth=4096

#pragma HLS STREAM variable=cfg_c1 depth=4
#pragma HLS STREAM variable=cfg_c2 depth=4
#pragma HLS STREAM variable=cfg_c3 depth=4

#pragma HLS DATAFLOW

    // Stage 1: Read input array into stream and pass configuration struct down pipeline
    read_input_task(input, in_fifo, cfg_c1, cfg);

    // Stage 2: Generate sliding windows with zero-overhead counter (II=1)
    window_gen_task(in_fifo, window_fifo, cfg_c1, cfg_c2);

    // Stage 3: Apply Hamming / Hann / Rectangular window function
    apply_window_task(window_fifo, coeff_fifo, cfg_c2, cfg_c3);

    // Stage 4: Perform Min-Max / Z-Score normalization and store to m_axi interface
    normalize_store_task(coeff_fifo, output, cfg_c3);
}
