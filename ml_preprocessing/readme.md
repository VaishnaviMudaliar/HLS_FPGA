
# HLS ML Preprocessing Block

Hardware-accelerated machine learning preprocessing kernel implemented using **Vitis HLS**.

This module provides an efficient, hardware-friendly implementation of common preprocessing operations used in time-series, audio, and sensor-based ML pipelines. The design emphasizes fixed-point arithmetic, single-pass streaming architecture, low-resource utilization, and synthesizability on Xilinx FPGAs.

---

## Motivation

Many FPGA-based machine learning systems focus heavily on neural network inference acceleration while preprocessing stages (sliding window extraction, windowing functions, feature normalization) remain CPU-bound. Offloading these preprocessing steps to software creates bandwidth bottlenecks and CPU overhead.

This project implements an end-to-end, single-pass streaming preprocessing pipeline in hardware to enable fully integrated FPGA ML accelerators.

---

## Architecture Overview

The pipeline uses a **Single-Pass Streaming DATAFLOW Architecture**. Intermediate multi-pass external memory round-trips have been eliminated by streaming data concurrently between tasks using point-to-point `hls::stream<>` FIFOs and configuration channels.


```

[ Input Data (m_axi / Stream) ]
            │
            ▼
┌───────────────────────┐
│ Read & Stream Task    │
└───────────┬───────────┘
            │ hls::stream
            ▼
┌───────────────────────┐
│ Window Generator Task │   <-- Buffers only 1 window (MAX_WINDOW_SIZE)
└───────────┬───────────┘
            │ hls::stream
            ▼
┌───────────────────────┐
│ Apply Window Task     │   <-- Dynamic Hamming / Hann / Rectangular
└───────────┬───────────┘
            │ hls::stream
            ▼
┌───────────────────────┐
│ Normalization Task    │   <-- Pipelined Min-Max / Z-Score (II=1)
└───────────┬───────────┘
            │
            ▼
[ Preprocessed Output (m_axi) ]

```

---

## Implemented Features

### 1. Sliding Window Generation
Generates overlapping or contiguous windows from time-series input sequences.
- **Configurable Parameters**: Input size (up to 1024), Window size (up to 64), Stride.
- **Memory Efficiency**: Local storage is restricted to **1 single window buffer** (64 samples) rather than storing the full matrix in external DDR.
- **Pipeline Optimization**: Unrolled parallel register shifting and indexing eliminates memory dependency violations to achieve **Initiation Interval (II) = 1**.

### 2. Window Functions
Dynamic coefficient generation supporting window sizes up to `MAX_WINDOW_SIZE = 64`:
- **Rectangular Window**
- **Hamming Window**
- **Hann Window**

### 3. Feature Normalization
- **Min-Max Normalization**: Scales values into the range $[0, 1]$.
  $$\hat{x} = \frac{x - \min(x)}{\max(x) - \min(x)}$$
- **Z-Score Normalization**: Standardizes features to zero mean and unit variance.
  $$\hat{x} = \frac{x - \mu}{\sigma}$$
- **Hardware Optimizations**: 
  - Division operations inside normalization loops are converted to single pre-calculated reciprocal multiplications ($1 / \text{range}$ or $1 / \sigma$), achieving **Initiation Interval (II) = 1**.
  - Replaced square-root divider dependencies with optimized fixed-point arithmetic.

---

## Hardware Implementation Details

### Fixed-Point Precision & Numerics
- **Base Data Type**: `data_t` = `ap_fixed<24,8>` (8 integer bits, 16 fractional bits, range $[-128, 127.99]$).
- **Intermediate Span Precision**:
  - Difference (`diff`) and range (`range`) widened to `ap_fixed<25,9>` (range $[-256, 255.99]$) to prevent intermediate integer wrapping on wide spans (e.g., $-100$ to $+100$).
  - Reciprocal scale factors (`inv_range` and `inv_stddev`) expanded to `inv_t` = `ap_fixed<32,12>` (12 integer bits), supporting scale multipliers up to 2047 without wrapping on fine-grained signals with small standard deviations ($\sigma = 0.005$).
- **Expanded Accumulator Headroom**: 
  - Mean accumulator widened to `ap_fixed<40,20>` (20 integer bits) to prevent summation overflow across 64 samples.
  - Variance sum accumulator (`var_sum`) expanded to `ap_fixed<48,24>` (24 integer bits / max range $\approx 8.38 \times 10^6$), preventing intermediate accumulation wrapping on high standard deviation windows ($\text{stddev} > 22.6$).
- **Precision Preservation**: Standard deviation calculation applies `hls::sqrt()` directly on wide variance arithmetic before casting down to `data_t` resolution, preventing wrapping on large variances ($\text{variance} > 128$).

### HLS Optimizations Applied
- `#pragma HLS DATAFLOW`: Task-level concurrency across memory read, windowing, coefficient applying, and normalization stages.
- `#pragma HLS PIPELINE II=1`: Applied across all inner sample-processing loops.
- **Reciprocal Pre-calculation**: Avoids logic-heavy division hardware inside loop pipelines.
- `#pragma HLS ARRAY_PARTITION complete`: Applied to local window buffers to resolve inter-iteration memory dependencies and enable parallel register access.

### Top-Level Interface
```cpp
void ml_preprocessing_hls(
    const data_t input[MAX_INPUT_SIZE],
    data_t output[MAX_WINDOWS][MAX_WINDOW_SIZE],
    int input_size,
    int window_size,
    int stride,
    WindowType window_type,
    NormalizationType normalization,
    int &num_windows
);

```

* **Data Ports**: Configured with `m_axi` (offset=slave, bundle=gmem) with 64-word burst length support.
* **Control Interface**: Configured with `s_axilite` for register control and parameter pass-through.

---

## Synthesis Results

### Configuration & Target Hardware

* **Tool**: Vitis HLS 2023.2
* **Target Device**: Zynq UltraScale+ MPSoC ZCU102 (`xczu9eg-ffvb1156-2-e`)
* **Clock Target**: 10.0 ns (100 MHz) | **Estimated**: ~7.3 ns

### Resource Comparison Across Iterations

| Resource | Initial Version | Intermediate DATAFLOW | Final Version (All Edge Fixes & Clean II=1) | Total Reduction vs. Initial |
| --- | --- | --- | --- | --- |
| **LUTs** | 178,103 (65.0%) | 22,079 (8.05%) | **32,132 (11.6%)** | **~82.0% Reduction** |
| **FFs** | 147,658 (27.0%) | 18,763 (3.42%) | **25,000 (4.58%)** | **~83.1% Reduction** |
| **BRAMs** | 260 (28.5%) | 12 (1.31%) | **10 (1.09%)** | **~96.1% Reduction** |
| **DSPs** | 10 (0.4%) | 34 (1.35%) | **35 (1.39%)** | **+25 (Mapped logic to DSPs)** |
| **Memory Dependency Violations** | Memory Dependency | II Violation (II=64) | **0 (Resolved to II=1)** | **Clean Pipeline Execution** |
| **Memory Passes** | 3 DDR Round-trips | 1 Pass (Streaming) | **1 Pass (Streaming)** | **Eliminated DDR bottlenecks** |

### Per-Task Sub-Block Breakdown

* **`read_input_task`**: 325 LUTs, 124 FFs, 0 DSPs, 0 BRAMs
* **`window_gen_task`**: 2,691 LUTs, 4,139 FFs, 0 DSPs, 0 BRAMs
* **`apply_window_task`**: 6,807 LUTs, 5,719 FFs, 28 DSPs, 0 BRAMs
* **`normalize_store_task`**: 19,626 LUTs, 12,469 FFs, 7 DSPs, 0 BRAMs

---

## Verification & Test Coverage

Self-checking testbench (`tb.cpp`) with non-zero return codes on failure. Verified through **Vitis HLS C Simulation (CSIM)**.

| Test Case | Test Description / Parameter Scope | Status |
| --- | --- | --- |
| **Sliding Window Generation** | Window count and value extraction (Sizes 4, 8, 16) | ✅ PASS |
| **Rectangular Window** | Window coefficient verification (Sizes 4, 8, 16) | ✅ PASS |
| **Hamming Window** | Window coefficient verification (Sizes 4, 8, 16) | ✅ PASS |
| **Hann Window** | Window coefficient verification (Sizes 4, 8, 16) | ✅ PASS |
| **Min-Max Normalization** | Single-window range scaling $[0, 1]$ | ✅ PASS |
| **Z-Score Normalization** | Zero mean & unit variance standardization | ✅ PASS |
| **Complete Processing Pipeline** | End-to-end kernel execution across all stages | ✅ PASS |
| **Z-Score Regression 1** | Alternating $\pm 12$ window ($\text{Variance} = 144 > 128$) | ✅ PASS |
| **Z-Score Regression 2** | High StdDev window (`var_sum` > 32,768, StdDev > 22.6) | ✅ PASS |
| **Min-Max Wide Span** | Wide range $-100$ to $+100$ ($\text{range} = 200$), tests `ap_fixed<25,9>` | ✅ PASS |
| **Z-Score Small StdDev** | Small standard deviation ($\sigma = 0.005$, $\text{multiplier} = 200$), tests `ap_fixed<32,12>` | ✅ PASS |
| **Sliding Window Bounds** | 1024 samples, window size 4, stride 1 (generates exactly 1021 windows) | ✅ PASS |

---

## Project File Structure

```
├── preprocessing_hls.hpp   # Configuration constants, enums, and function declarations
├── preprocessing_hls.cpp   # DATAFLOW top-level kernel & streaming task implementations
└── tb.cpp                  # Automated self-checking testbench with multi-size & regression coverage

```

---

## License

Distributed under the MIT License.

