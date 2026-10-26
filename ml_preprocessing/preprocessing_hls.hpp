#ifndef PREPROCESSING_HLS_HPP
#define PREPROCESSING_HLS_HPP

#include <ap_fixed.h>
#include <ap_int.h>

//==============================================================
// Configuration Parameters
//==============================================================

constexpr int MAX_INPUT_SIZE  = 1024;
constexpr int MAX_WINDOW_SIZE = 64;
// Guarantees allocation headroom for worst-case window size (e.g. min window size = 1 or 4)
constexpr int MAX_WINDOWS     = MAX_INPUT_SIZE;
//==============================================================
// Data Type
//==============================================================

// Base data type (8 integer bits, range -128 to +127.99)
typedef ap_fixed<24, 8> data_t;

// Expanded intermediate span types
typedef ap_fixed<25, 9> span_t;

// Expanded reciprocal multiplier type (15 integer bits, range -16384 to +16383.99)
// Supports small standard deviations down to EPSILON (0.0001 -> scale = 10,000)
typedef ap_fixed<32, 15> inv_t;

//==============================================================
// Supported Window Functions
//==============================================================

enum class WindowType
{
    RECTANGULAR,
    HAMMING,
    HANN
};

//==============================================================
// Supported Normalization Methods
//==============================================================

enum class NormalizationType
{
    NONE,
    MIN_MAX,
    Z_SCORE
};

//==============================================================
// Sliding Window Generator
//==============================================================

void sliding_window_hls(
    const data_t input[MAX_INPUT_SIZE],
    data_t windows[MAX_WINDOWS][MAX_WINDOW_SIZE],
    int input_size,
    int window_size,
    int stride,
    int &num_windows);

//==============================================================
// Window Function
//==============================================================

void apply_window_hls(
    data_t windows[MAX_WINDOWS][MAX_WINDOW_SIZE],
    int num_windows,
    int window_size,
    WindowType window_type);

//==============================================================
// Normalization
//==============================================================

void normalize_windows_hls(
    data_t windows[MAX_WINDOWS][MAX_WINDOW_SIZE],
    int num_windows,
    int window_size,
    NormalizationType normalization);

//==============================================================
// Min-Max Normalization
//==============================================================

void min_max_normalize_hls(
    data_t windows[MAX_WINDOWS][MAX_WINDOW_SIZE],
    int num_windows,
    int window_size);

//==============================================================
// Z-Score Normalization
//==============================================================

void zscore_normalize_hls(
    data_t windows[MAX_WINDOWS][MAX_WINDOW_SIZE],
    int num_windows,
    int window_size);

//==============================================================
// Top-Level ML Preprocessing Kernel
//==============================================================

void ml_preprocessing_hls(
    const data_t input[MAX_INPUT_SIZE],
    data_t output[MAX_WINDOWS][MAX_WINDOW_SIZE],
    int input_size,
    int window_size,
    int stride,
    WindowType window_type,
    NormalizationType normalization,
    int &num_windows);

#endif
