# YAML Configuration Design for Whisper Harness
# ==============================================
# 
# IMPLEMENTATION STATUS: ✅ COMPLETE & TESTED
# All features from this design have been implemented in benchmark.py
# Tests available in test_harness.py (run with: python -m unittest test_harness.py -v)

## Overview

This document describes the YAML configuration system for the Whisper Harness benchmarking tool.
The system supports flexible configuration formats, multiple frameworks, and comprehensive testing options.

## Key Features

1. **Singular/Plural Flexibility**: Both `model: small` and `models: [small, medium]` work seamlessly
2. **Shortcut Format**: Simple configs for common frameworks (faster-whisper, whisper.cpp)
3. **Full Explicit Format**: Complete control with offload configurations
4. **Test Dataset Integration**: Audio files with ground truth for WER/CER calculation
5. **Framework Auto-Detection**: Quantizations automatically detected when not specified
6. **GPU Offloading Support**: Advanced layer-based offloading for transcribe.cpp

---

## Complete YAML Syntax Reference

### Global Settings (Optional)

```yaml
settings:
  output_dir: "benchmark_results"
  mode: "quick"  # Options: quick, full, compare
  default_language: "ru"
  default_threads: 2
```

### Test Datasets

Define audio files with optional ground truth references:

```yaml
test_datasets:
  - name: "russian_speech_sample"
    audio_path: "/path/to/audio.wav"
    reference_text: "/path/to/ground_truth.txt"
    # Or inline reference text:
    # reference_text: "Привет мир это тестовая транскрипция"
    language: "ru"
    
  - name: "english_podcast"
    audio_path: "podcast_ep01.wav"
    reference_text: "podcast_transcript.txt"
    language: "en"
    
  - name: "multilingual_mix"
    audio_path: "mixed_audio.wav"
    # No reference = only timing metrics (no WER/CER)
    languages: ["ru", "en"]
```

### Benchmark Configurations

#### 1. Shortcut Format - faster-whisper (plural form)

```yaml
benchmarks:
  - framework: faster-whisper
    models: [medium, small]  # plural: tests both models
    quantizations: [int8_float32, float16]
    devices: [cuda]
    beam_sizes: [1, 3]
```

#### 2. Shortcut Format - whisper.cpp (singular form)

```yaml
  - framework: whisper.cpp
    model: ggerganov/whisper.cpp  # singular form
    quantization: [q5_0, q4_0]   # can still be a list
    devices: [cpu]
    threads: 2
```

#### 3. HuggingFace Models - Auto-detect Quantizations

```yaml
  - framework: huggingface
    models: 
      - openai/whisper-small-ru
      - sberbank-ai/whisper-medium-ru
    devices: [cuda]
    load_in_8bit: false
    # torch_dtype auto-detected from model
```

#### 4. Sber GigaAM with ONNX (Recommended for Low-RAM Systems)

```yaml
  - framework: sber
    models: [ctc, rnnt]  # model types: ctc or rnnt
    use_onnx: true
    devices: [cpu, cuda]
```

#### 5. ONNX ASR Models (GigaAM v3 via onnx-asr)

```yaml
  - framework: onnx-asr
    models: [gigaam-v3-e2e-ctc, gigaam-v3-e2e-rnnt]
    devices: [cuda, cpu]
    execution_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

#### 6. Full Explicit Format - transcribe.cpp with GPU Offloading

Ideal for GGUF models like GigaAM v3:

```yaml
  - framework: transcribe.cpp
    models: [handy-computer/gigaam-v3-e2e-rnnt-gguf]
    quantizations: [Q6_K, Q8_0, Q5_K_M]
    devices: [cuda, cpu, offload]
    threads: 4
    # Offload configurations - tests different GPU/CPU splits
    offload_configs:
      # Option 1: Split by layers - first 10 layers on GPU
      - layer_count: 10
        split_mode: "layer"
      # Option 2: All on CPU (no offloading)
      - layer_count: 0
        split_mode: "none"
      # Option 3: All on GPU (if VRAM allows)
      - layer_count: 999
        split_mode: "layer"
      # Option 4: Dynamic based on available VRAM
      - split_mode: "auto"
        max_vram_gb: 2.0
```

#### 7. Explicit Per-Config Override - Full Control

```yaml
  - framework: faster-whisper
    model: large-v3
    quantizations: [float16]
    devices: [cuda]
    beam_sizes: [5]
    language: "ru"
    compute_type: "float16"
    gpu_id: 0
    threads: 4
```

#### 8. Mixed Device Testing - One Model on CPU and GPU

```yaml
  - framework: faster-whisper
    model: small
    quantizations: [int8_float32]
    devices: [cuda, cpu]  # Tests both devices
    beam_sizes: [1]
```

#### 9. Minimal Config - All Defaults

```yaml
  - framework: faster-whisper
    # model defaults to "small"
    # quantizations auto-detected as [int8_float32]
    # devices auto-detected as [cuda] or [cpu] based on system
```

---

## Offload Configuration Options

For frameworks supporting GPU offloading (transcribe.cpp, llama.cpp-based):

```yaml
offload_configs:
  # Option 1: Split by layers
  - layer_count: <int>      # Number of layers to offload to GPU
    split_mode: "layer"     # Layer-by-layer split
    
  # Option 2: Split by rows/blocks  
  - block_count: <int>
    split_mode: "row"
    
  # Option 3: No offloading (CPU only)
  - split_mode: "none"
  
  # Option 4: Full GPU (if memory allows)
  - split_mode: "full_gpu"
  
  # Option 5: Dynamic based on available VRAM
  - split_mode: "auto"
    max_vram_gb: 2.0        # Don't exceed this VRAM usage
```

---

## Framework Support Matrix

| Framework | Shortcut | Plural/Singular | Offload Support | Auto Quantizations |
|-----------|----------|-----------------|-----------------|-------------------|
| faster-whisper | ✅ | ✅ | ❌ | ✅ |
| whisper.cpp | ✅ | ✅ | ⚠️ (via main.cpp params) | ✅ |
| huggingface | ✅ | ✅ | ❌ (uses device_map) | ✅ |
| sber | ✅ | ✅ | ❌ | ✅ |
| onnx-asr | ✅ | ✅ | ✅ (execution providers) | ✅ |
| transcribe.cpp | ✅ | ✅ | ✅ (layer offloading) | ✅ |

---

## Key Features Explained

### 1. Flexible Input Forms

Both singular and plural forms are supported:
- `model: small` ≡ `models: [small]`
- `quantization: [q5_0]` ≡ `quantizations: [q5_0]`
- `device: cuda` ≡ `devices: [cuda]`

### 2. Smart Defaults

If quantizations are not specified, they are auto-detected based on the framework:
- **faster-whisper**: `[int8_float32]`
- **whisper.cpp**: `[q5_0]`
- **huggingface**: Auto-detected from model config
- **sber**: `[float32]` or `[int8]` if ONNX
- **onnx-asr**: Framework defaults
- **transcribe.cpp**: `[Q5_K_M]`

### 3. Test Datasets

Define multiple audio files with ground truth once, reuse across all benchmarks:
- Audio paths can be absolute or relative
- Reference text can be a file path or inline string
- Multiple datasets can be defined in a single config
- WER/CER only calculated when reference text is provided

### 4. Offload Control

Fine-grained GPU offloading for memory-constrained systems:
- Layer-based splitting for precise control
- Dynamic VRAM-based allocation
- CPU-only fallback options
- Automatic detection of optimal settings

### 5. Framework Agnostic

Same YAML structure works across all supported frameworks:
- Consistent field names
- Unified configuration format
- Easy to switch between frameworks
- Mix multiple frameworks in one config

---

## Usage Examples

### Quick Benchmark with Default Config

```bash
python benchmark.py --audio test.wav --mode quick
```

### Full Benchmark with YAML Config

```bash
python benchmark.py --audio test.wav --config benchmark_config.yaml --mode full
```

### With Ground Truth for Accuracy Metrics

```bash
python benchmark.py --audio test.wav --reference ground_truth.txt
```

### Generate Sample Config

```bash
python benchmark.py --generate-sample-config
```

### Run Tests

```bash
python -m unittest test_harness.py -v
```

Tests cover:
- OffloadConfig parsing and defaults
- TestDataset configuration
- BenchmarkConfig generation
- Singular/plural form handling (normalize_to_list)
- parse_offload_configs with various input types
- YAML config generation from shortcut and explicit formats
- Auto-quantization detection
- GPU availability checks

---

## Complete Example Configuration

See `sample_benchmark_config.yaml` for a complete working example that includes:
- All framework types
- Multiple test datasets
- Various offload configurations
- Both singular and plural forms
- Inline and file-based references

---

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure model ID is correct and internet connection is available
2. **OOM errors**: Reduce beam_size, use int8 quantization, or switch to CPU
3. **Slow performance**: Use smaller models, reduce beam_size, enable GPU
4. **WER not calculated**: Ensure reference_text is provided in test_datasets

### Getting Help

- Check README.md for installation instructions
- Review CUDA_CACHE_FREE.md for VRAM management options
- Run tests to verify your setup: `python -m unittest test_harness.py -v`
