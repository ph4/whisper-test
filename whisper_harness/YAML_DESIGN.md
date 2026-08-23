# YAML Configuration Design for Whisper Harness
# ==============================================
# 
# IMPLEMENTATION STATUS: ✅ COMPLETE
# All features from this design have been implemented in benchmark.py
# Tests available in test_harness.py (TestBenchmarkConfig, TestYAMLConfigGeneration)

## Final YAML Syntax Design

The configuration supports:
1. **Singular and plural forms** (model/models, quantization/quantizations, device/devices)
2. **Shortcut format** for common frameworks (faster-whisper, whisper.cpp)
3. **Full explicit format** for fine-grained control
4. **Offload configurations** for transcribe.cpp and similar frameworks
5. **Test audio files and ground truth references** in the config
6. **WER/CER evaluation data** embedded or referenced

---

## Example YAML Structure

```yaml
# Global settings (optional)
settings:
  output_dir: "benchmark_results"
  mode: "quick"  # quick, full, compare
  default_language: "ru"
  default_threads: 2
  
# Test datasets - audio files with optional ground truth
test_datasets:
  - name: "russian_speech_sample"
    audio_path: "/path/to/audio.wav"
    reference_text: "/path/to/ground_truth.txt"
    # Or inline reference:
    # reference_text: "Привет мир это тестовая транскрипция"
    language: "ru"
    
  - name: "english_podcast"
    audio_path: "podcast_ep01.wav"
    reference_text: "podcast_transcript.txt"
    language: "en"
    
  - name: "multilingual_mix"
    audio_path: "mixed_audio.wav"
    # No reference = no WER/CER calculation
    languages: ["ru", "en"]

# Benchmark configurations
benchmarks:
  # SHORTCUT FORMAT - faster-whisper with multiple models
  - framework: faster-whisper
    models: [medium, small]  # plural form
    quantizations: [int8_float32, float16]
    devices: [cuda]
    beam_sizes: [1, 3]
    
  # SHORTCUT FORMAT - singular form also works
  - framework: whisper.cpp
    model: ggerganov/whisper.cpp  # singular form
    quantization: [q5_0, q4_0]   # can still be list
    devices: [cpu]
    threads: 2
    
  # HuggingFace models - auto-detect quantizations
  - framework: huggingface
    models: 
      - openai/whisper-small-ru
      - sberbank-ai/whisper-medium-ru
    devices: [cuda]
    load_in_8bit: false
    
  # Sber GigaAM with ONNX
  - framework: sber
    models: [ctc, rnnt]  # model types
    use_onnx: true
    devices: [cpu, cuda]
    
  # FULL EXPLICIT FORMAT - transcribe.cpp with offloading
  - framework: transcribe.cpp
    models: [handy-computer/gigaam-v3-e2e-rnnt-gguf]
    quantizations: [Q6_K, Q8_0, Q5_K_M]
    devices: [cuda, cpu, offload]
    offload_configs:
      - layer_count: 10  # offload first 10 layers to GPU
        split_mode: "layer"
      - layer_count: 0   # all on CPU
        split_mode: "none"
      - layer_count: 999 # all on GPU (if fits)
        split_mode: "layer"
    threads: 4
    
  # ONNX ASR models
  - framework: onnx-asr
    models: [gigaam-v3-e2e-ctc, gigaam-v3-e2e-rnnt]
    devices: [cuda, cpu]
    execution_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
  # Explicit per-config override
  - framework: faster-whisper
    model: large-v3
    quantizations: [float16]
    devices: [cuda]
    beam_sizes: [5]
    language: "ru"
    compute_type: "float16"
    gpu_id: 0
```

---

## Implementation Details

### Data Classes (benchmark.py)

- `OffloadConfig`: layer_count, split_mode, block_count, max_vram_gb
- `TestDataset`: name, audio_path, reference_text, language, languages
- `BenchmarkConfig`: framework, model, quantization, device, beam_size, offload_config, etc.
- `BenchmarkResult`: timing metrics, memory metrics, WER/CER, transcription
- `SystemConfig`: CPU/GPU info, RAM/VRAM totals

### Key Functions

- `parse_yaml_config(config_path)`: Load YAML file
- `normalize_to_list(value)`: Handle singular/plural forms
- `parse_offload_configs(offload_data)`: Parse offload configurations
- `generate_test_configs(yaml_config, system_config, mode)`: Generate all benchmark configs

### Offload Configuration Options

For frameworks that support GPU offloading (transcribe.cpp, llama.cpp-based):

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

## Key Features

1. **Flexible Input**: Both `model: small` and `models: [small, medium]` work
2. **Smart Defaults**: If quantizations not specified, auto-detect based on framework
3. **Test Datasets**: Define multiple audio files with ground truth once, reuse across benchmarks
4. **Offload Control**: Fine-grained GPU offloading for memory-constrained systems
5. **Framework Agnostic**: Same YAML structure works across all supported frameworks

---

## Usage Examples

```bash
# Quick benchmark with default config
python benchmark.py --audio test.wav --mode quick

# Full benchmark with YAML config
python benchmark.py --audio test.wav --config benchmark_config.yaml --mode full

# With ground truth for accuracy metrics
python benchmark.py --audio test.wav --reference ground_truth.txt

# Generate sample config
python benchmark.py --generate-sample-config
```

---

## Testing

Run tests with:
```bash
python -m unittest test_harness.TestBenchmarkConfig -v
python -m unittest test_harness.TestYAMLConfigGeneration -v
```

Tests cover:
- OffloadConfig parsing
- TestDataset parsing
- BenchmarkConfig generation
- Singular/plural form handling
- Auto-quantization detection
- Offload configuration combinations
- GPU availability checks
