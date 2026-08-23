#!/usr/bin/env python3
"""Benchmark harness for Whisper ASR models.

Extends the transcription harness with comprehensive benchmarking capabilities:
- Multi-framework benchmarking (faster-whisper, whisper.cpp, HF)
- Automated test configuration generation
- Memory monitoring (RAM/VRAM)
- Accuracy metrics (WER/CER)
- Result export (CSV, JSON, Markdown)
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.memory_monitor import MemoryMonitor, get_system_info
from utils.metrics import calculate_wer, calculate_cer, calculate_rtf


@dataclass
class OffloadConfig:
    """Configuration for GPU offloading (layer splitting)."""
    
    layer_count: int = 0
    split_mode: str = "none"  # "none", "layer", "row", "full_gpu", "auto"
    block_count: Optional[int] = None
    max_vram_gb: Optional[float] = None


@dataclass
class TestDataset:
    """Test dataset configuration."""
    
    name: str = ""
    audio_path: str = ""
    reference_text: Optional[str] = None  # Path to ground truth file or inline text
    language: str = "ru"
    languages: Optional[List[str]] = None  # For multilingual audio


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    
    framework: str = "faster-whisper"
    model: str = "small"
    quantization: str = "int8_float32"
    device: str = "cuda"
    beam_size: int = 1
    language: str = "ru"
    threads: int = 2
    gpu_id: int = 0
    
    # Framework-specific parameters
    compute_type: Optional[str] = None  # for faster-whisper
    torch_dtype: Optional[str] = None  # for HF
    load_in_8bit: bool = False  # for HF
    use_onnx: bool = False  # for Sber
    
    # Offloading configuration (for transcribe.cpp etc.)
    offload_config: Optional[OffloadConfig] = None
    
    # Test dataset reference
    dataset_name: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    
    framework: str = ""
    model: str = ""
    quantization: str = ""
    device: str = ""
    beam_size: int = 0
    status: str = "PENDING"
    error_message: str = ""
    
    # Timing metrics
    load_time_sec: float = 0.0
    transcribe_time_sec: float = 0.0
    total_time_sec: float = 0.0
    rtf: float = 0.0
    audio_duration_sec: float = 0.0
    
    # Memory metrics
    ram_before_mb: float = 0.0
    ram_after_mb: float = 0.0
    ram_peak_mb: float = 0.0
    vram_before_mb: float = 0.0
    vram_after_mb: float = 0.0
    vram_peak_mb: float = 0.0
    
    # Accuracy metrics
    wer: Optional[float] = None
    cer: Optional[float] = None
    word_count: int = 0
    char_count: int = 0
    
    # Transcription result
    transcription: str = ""
    
    # Config snapshot
    config: Optional[BenchmarkConfig] = None
    
    # Dataset name (for multi-dataset benchmarks)
    dataset_name: str = "default"
    
    # Offload configuration info
    offload_split_mode: str = ""
    offload_layer_count: int = 0


@dataclass
class SystemConfig:
    """System configuration information."""
    
    cpu_model: str = ""
    cpu_cores: int = 0
    total_ram_gb: float = 0.0
    gpu_name: str = ""
    gpu_count: int = 0
    total_vram_gb: float = 0.0
    cuda_version: str = ""
    python_version: str = ""
    platform: str = ""


def parse_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Supports two formats:
    1. Explicit config with all parameters
    2. Shortcut format for faster-whisper and whisper.cpp
    
    Also supports:
    - Singular and plural forms (model/models, quantization/quantizations, etc.)
    - Offload configurations for transcribe.cpp
    - Test datasets with ground truth references
    
    Example shortcut format:
    ```yaml
    settings:
      output_dir: "benchmark_results"
      mode: "quick"
      default_language: "ru"
      
    test_datasets:
      - name: "russian_sample"
        audio_path: "/path/to/audio.wav"
        reference_text: "/path/to/ground_truth.txt"
        language: "ru"
        
    benchmarks:
      - framework: faster-whisper
        models: [medium, small]
        quantizations: [int8_float32, float16]
        devices: [cuda]
        
      - framework: whisper.cpp
        model: ggerganov/whisper.cpp
        quantizations: [q5_0, q4_0]
        
      - framework: transcribe.cpp
        models: [handy-computer/gigaam-v3-e2e-rnnt-gguf]
        quantizations: [Q6_K, Q8_0]
        devices: [cuda, cpu, offload]
        offload_configs:
          - layer_count: 10
            split_mode: "layer"
          - layer_count: 0
            split_mode: "none"
    ```
    """
    try:
        import yaml
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            
        return config
    except ImportError:
        print("Warning: PyYAML not installed. Install with: pip install pyyaml")
        return {}
    except FileNotFoundError:
        print(f"Warning: Config file not found: {config_path}")
        return {}


def normalize_to_list(value: Any) -> List[Any]:
    """Normalize a value to a list (handles singular/plural forms)."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def parse_offload_configs(offload_data: Any) -> List[OffloadConfig]:
    """Parse offload configuration from YAML data."""
    if not offload_data:
        return [OffloadConfig()]  # Return default config
    
    offload_configs = []
    offload_list = normalize_to_list(offload_data)
    
    for item in offload_list:
        if isinstance(item, dict):
            config = OffloadConfig(
                layer_count=item.get("layer_count", 0),
                split_mode=item.get("split_mode", "none"),
                block_count=item.get("block_count"),
                max_vram_gb=item.get("max_vram_gb"),
            )
            offload_configs.append(config)
        elif isinstance(item, str):
            # Simple string format like "layer:10" or "auto"
            config = OffloadConfig(split_mode=item)
            offload_configs.append(config)
    
    return offload_configs if offload_configs else [OffloadConfig()]


def generate_test_configs(
    yaml_config: Dict[str, Any],
    system_config: SystemConfig,
    mode: str = "quick",
) -> Tuple[List[BenchmarkConfig], List[TestDataset]]:
    """Generate test configurations from YAML config.
    
    Supports both singular and plural forms:
    - model/models
    - quantization/quantizations  
    - device/devices
    - beam_size/beam_sizes
    
    Args:
        yaml_config: Parsed YAML configuration
        system_config: Detected system configuration
        mode: Benchmark mode ('quick', 'full', 'compare')
        
    Returns:
        Tuple of (List of BenchmarkConfig objects, List of TestDataset objects)
    """
    configs = []
    benchmarks = yaml_config.get("benchmarks", [])
    
    # Parse test datasets
    test_datasets = []
    for dataset in yaml_config.get("test_datasets", []):
        test_dataset = TestDataset(
            name=dataset.get("name", "default"),
            audio_path=dataset.get("audio_path", ""),
            reference_text=dataset.get("reference_text"),
            language=dataset.get("language", "ru"),
            languages=dataset.get("languages"),
        )
        test_datasets.append(test_dataset)
    
    for bench in benchmarks:
        framework = bench.get("framework", "faster-whisper")
        
        # Support both singular and plural forms
        models = normalize_to_list(bench.get("models") or bench.get("model", "small"))
        
        # Get quantizations - can be list or single value
        quantizations = normalize_to_list(bench.get("quantizations") or bench.get("quantization", []))
        
        # If no quantizations specified, use defaults based on framework
        if not quantizations:
            if framework in ["faster-whisper", "fast_whisper"]:
                quantizations = ["int8_float32"]
            elif framework in ["whisper.cpp", "whisper_cpp"]:
                quantizations = ["q5_0"]
            elif framework in ["huggingface", "hf_whisper"]:
                quantizations = ["float16"]
            elif framework in ["sber", "sber_gigaam"]:
                quantizations = ["float32"]  # Sber models typically float32
            elif framework in ["onnx-asr", "onnx"]:
                quantizations = ["float32"]
            elif framework in ["transcribe.cpp"]:
                quantizations = ["Q5_K_M"]
        
        # Get devices - support both singular and plural
        devices = normalize_to_list(bench.get("devices") or bench.get("device", ["cuda"]))
        
        # Get beam sizes - support both singular and plural
        beam_sizes = normalize_to_list(bench.get("beam_sizes") or bench.get("beam_size", [1]))
        
        # Parse offload configurations
        offload_configs = parse_offload_configs(bench.get("offload_configs"))
        
        # Generate configs for all combinations
        for model in models:
            for quantization in quantizations:
                for device in devices:
                    # Skip CUDA if not available
                    if device == "cuda" and system_config.gpu_count == 0:
                        continue
                    
                    # Handle "offload" device specially
                    is_offload_device = device == "offload"
                    actual_device = "cpu" if is_offload_device else device
                    
                    # Skip large models if VRAM is limited
                    if "large" in model.lower() and system_config.total_vram_gb < 2.5:
                        if mode != "full":
                            print(f"Skipping {model} on {framework}: insufficient VRAM")
                            continue
                    
                    # Determine if we should use offload configs
                    offloads_to_use = offload_configs if is_offload_device or framework == "transcribe.cpp" else []
                    if not offloads_to_use:
                        offloads_to_use = [None]
                    
                    for offload_config in offloads_to_use:
                        for beam_size in beam_sizes:
                            config = BenchmarkConfig(
                                framework=framework,
                                model=model,
                                quantization=quantization,
                                device=actual_device,
                                beam_size=beam_size,
                                language=bench.get("language", "ru"),
                                threads=bench.get("threads", 2),
                                gpu_id=bench.get("gpu_id", 0),
                                compute_type=bench.get("compute_type"),
                                torch_dtype=bench.get("torch_dtype"),
                                load_in_8bit=bench.get("load_in_8bit", False),
                                use_onnx=bench.get("use_onnx", False),
                                offload_config=offload_config,
                            )
                            configs.append(config)
    
    return configs, test_datasets


class BenchmarkRunner:
    """Runs benchmarks for different frameworks."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.transcriber = None
        
    def load_transcriber(self):
        """Load transcriber based on framework."""
        from transcribers import (
            FasterWhisperTranscriber,
            WhisperCppTranscriber,
            HuggingFaceWhisperTranscriber,
            SberGigaAMTranscriber,
        )
        
        framework_map = {
            "faster-whisper": FasterWhisperTranscriber,
            "fast_whisper": FasterWhisperTranscriber,
            "whisper.cpp": WhisperCppTranscriber,
            "whisper_cpp": WhisperCppTranscriber,
            "huggingface": HuggingFaceWhisperTranscriber,
            "hf_whisper": HuggingFaceWhisperTranscriber,
            "sber": SberGigaAMTranscriber,
            "sber_gigaam_ctc": SberGigaAMTranscriber,
        }
        
        transcriber_class = framework_map.get(self.config.framework.lower())
        if transcriber_class is None:
            raise ValueError(f"Unknown framework: {self.config.framework}")
        
        # Prepare kwargs based on framework
        kwargs = {
            "model_id": self.config.model,
            "device": self.config.device,
        }
        
        if self.config.framework in ["faster-whisper", "fast_whisper"]:
            kwargs["compute_type"] = self.config.compute_type or self.config.quantization
            kwargs["beam_size"] = self.config.beam_size
        elif self.config.framework in ["whisper.cpp", "whisper_cpp"]:
            kwargs["quantization"] = self.config.quantization
            kwargs["n_threads"] = self.config.threads
        elif self.config.framework in ["huggingface", "hf_whisper"]:
            kwargs["torch_dtype"] = self.config.torch_dtype or self.config.quantization
            kwargs["load_in_8bit"] = self.config.load_in_8bit
        elif self.config.framework in ["sber", "sber_gigaam_ctc"]:
            kwargs["use_onnx"] = self.config.use_onnx
        
        self.transcriber = transcriber_class(**kwargs)
        
    def run(self, audio_path: str, ground_truth: Optional[str] = None) -> BenchmarkResult:
        """Run benchmark and return results."""
        result = BenchmarkResult(
            framework=self.config.framework,
            model=self.config.model,
            quantization=self.config.quantization,
            device=self.config.device,
            beam_size=self.config.beam_size,
            config=self.config,
        )
        
        # Initialize memory monitor
        monitor = MemoryMonitor(sampling_interval_ms=100)
        
        try:
            # Get initial memory
            monitor.start()
            time.sleep(0.5)  # Let memory stabilize
            initial_ram, initial_vram = monitor.get_latest_memory()
            result.ram_before_mb = initial_ram
            result.vram_before_mb = initial_vram
            
            # Load transcriber
            start_time = time.time()
            self.load_transcriber()
            # Trigger lazy loading
            self.transcriber._ensure_loaded()
            load_time = time.time() - start_time
            result.load_time_sec = load_time
            
            # Get memory after loading
            time.sleep(0.5)
            loaded_ram, loaded_vram = monitor.get_latest_memory()
            result.ram_after_mb = loaded_ram
            result.vram_after_mb = loaded_vram
            
            # Run transcription
            transcribe_start = time.time()
            transcribe_result = self.transcriber.transcribe(
                audio_path,
                language=self.config.language,
            )
            transcribe_time = time.time() - transcribe_start
            result.transcribe_time_sec = transcribe_time
            result.total_time_sec = load_time + transcribe_time
            
            # Get transcription text
            result.transcription = transcribe_result.get("text", "")
            result.audio_duration_sec = transcribe_result.get("duration", 0.0)
            
            # Calculate RTF
            if result.audio_duration_sec > 0:
                result.rtf = calculate_rtf(result.audio_duration_sec, transcribe_time)
            
            # Get peak memory
            monitor.stop()
            peak_ram, peak_vram = monitor.get_peak_memory()
            result.ram_peak_mb = max(peak_ram, loaded_ram)
            result.vram_peak_mb = max(peak_vram, loaded_vram) if peak_vram else loaded_vram
            
            # Calculate accuracy metrics if ground truth provided
            if ground_truth:
                result.wer = calculate_wer(ground_truth, result.transcription)
                result.cer = calculate_cer(ground_truth, result.transcription)
                result.word_count = len(result.transcription.split())
                result.char_count = len(result.transcription)
            
            result.status = "PASS"
            
        except Exception as e:
            monitor.stop()
            result.status = "FAIL"
            result.error_message = str(e)
            
        finally:
            # Cleanup
            self.transcriber = None
            from utils.memory_monitor import clear_gpu_cache
            clear_gpu_cache()
        
        return result


class BenchmarkHarness:
    """Main benchmark harness for running multiple configurations."""
    
    def __init__(
        self,
        yaml_config: Dict[str, Any],
        output_dir: str = "benchmark_results",
        mode: str = "quick",
    ):
        self.yaml_config = yaml_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        
        # Get system info
        sys_info = get_system_info()
        self.system_config = SystemConfig(
            cpu_model=sys_info.get("cpu_model", ""),
            cpu_cores=sys_info.get("cpu_cores", 0),
            total_ram_gb=sys_info.get("ram_total_gb", 0.0),
            gpu_name=sys_info.get("gpu_name", ""),
            gpu_count=sys_info.get("gpu_count", 0),
            total_vram_gb=sys_info.get("vram_total_gb", 0.0),
            python_version=sys_info.get("python_version", ""),
            platform=sys_info.get("platform", ""),
        )
        
        # Generate test configurations and datasets
        configs_result = generate_test_configs(
            yaml_config, self.system_config, mode
        )
        self.test_configs = configs_result[0] if isinstance(configs_result, tuple) else configs_result
        self.test_datasets = configs_result[1] if isinstance(configs_result, tuple) else []
        
        self.results: List[BenchmarkResult] = []
        
    def log(self, message: str):
        """Log message to console."""
        print(message)
        
    def run_all(self, audio_path: str, ground_truth: Optional[str] = None):
        """Run all benchmark configurations."""
        self.log("=" * 60)
        self.log("Whisper ASR Benchmark Harness")
        self.log("=" * 60)
        
        # If test datasets are defined, use them
        if self.test_datasets:
            self.log(f"Test datasets: {len(self.test_datasets)}")
            for dataset in self.test_datasets:
                self.log(f"  - {dataset.name}: {dataset.audio_path}")
        else:
            self.log(f"Audio: {audio_path}")
            
        self.log(f"System: {self.system_config.cpu_model} | {self.system_config.gpu_name}")
        self.log(f"VRAM: {self.system_config.total_vram_gb:.1f} GB | RAM: {self.system_config.total_ram_gb:.1f} GB")
        self.log(f"Configs to test: {len(self.test_configs)}")
        self.log("=" * 60)
        
        total_start = time.time()
        
        # Determine which audio files to test
        audio_files_to_test = []
        if self.test_datasets:
            for dataset in self.test_datasets:
                audio_files_to_test.append((dataset.audio_path, dataset.reference_text, dataset.name))
        else:
            audio_files_to_test.append((audio_path, ground_truth, "default"))
        
        # Run benchmarks for each audio file
        all_results = []
        for test_audio, test_ground_truth, dataset_name in audio_files_to_test:
            if not os.path.exists(test_audio):
                self.log(f"\n⚠️ Warning: Audio file not found: {test_audio}, skipping...")
                continue
                
            self.log(f"\n{'='*60}")
            self.log(f"Testing on dataset: {dataset_name}")
            self.log(f"Audio file: {test_audio}")
            self.log(f"{'='*60}")
            
            for i, config in enumerate(self.test_configs, 1):
                self.log(f"\n[{i}/{len(self.test_configs)}] {config.framework} | {config.model} | {config.quantization} | {config.device}")
                
                # Add offload info if present
                if config.offload_config:
                    self.log(f"    Offload: {config.offload_config.split_mode} (layers: {config.offload_config.layer_count})")
                
                try:
                    runner = BenchmarkRunner(config)
                    result = runner.run(test_audio, test_ground_truth)
                    result.dataset_name = dataset_name  # Tag result with dataset name
                    self.results.append(result)
                    
                    if result.status == "PASS":
                        self.log(f"  ✓ PASSED | RTF: {result.rtf:.3f} | VRAM: {result.vram_peak_mb:.0f}MB | RAM: {result.ram_peak_mb:.0f}MB")
                        if result.wer is not None:
                            self.log(f"           | WER: {result.wer:.2f}% | CER: {result.cer:.2f}%")
                    else:
                        self.log(f"  ✗ FAILED: {result.error_message}")
                        
                except Exception as e:
                    self.log(f"  ✗ ERROR: {e}")
                    
                # Clear GPU memory between tests
                from utils.memory_monitor import clear_gpu_cache
                clear_gpu_cache()
                
                time.sleep(1.0)  # Cooldown
        
        total_time = time.time() - total_start
        self.log(f"\n{'='*60}")
        self.log(f"Benchmark completed in {total_time:.2f}s ({total_time/60:.2f} min)")
        self.log(f"Results: {len([r for r in self.results if r.status == 'PASS'])} passed, "
                f"{len([r for r in self.results if r.status != 'PASS'])} failed")
        
        # Save results
        self.save_results()
        self.generate_report()
        
    def save_results(self):
        """Save results to CSV and JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_path = self.output_dir / f"results_{timestamp}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "framework", "model", "quantization", "device", "beam_size",
                "load_time_sec", "transcribe_time_sec", "rtf", "total_time_sec",
                "ram_before_mb", "ram_after_mb", "ram_peak_mb",
                "vram_before_mb", "vram_after_mb", "vram_peak_mb",
                "wer", "cer", "word_count", "char_count",
                "status", "error_message",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {k: getattr(result, k, "") for k in fieldnames}
                writer.writerow(row)
        
        # Save JSON
        json_path = self.output_dir / f"results_{timestamp}.json"
        json_data = {
            "system_config": asdict(self.system_config),
            "mode": self.mode,
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    **{k: getattr(r, k, "") for k in fieldnames},
                }
                for r in self.results
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        self.log(f"Results saved to: {csv_path}")
        self.log(f"JSON saved to: {json_path}")
        
    def generate_report(self):
        """Generate Markdown report."""
        report_path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        passed = [r for r in self.results if r.status == "PASS"]
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Whisper ASR Benchmark Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # System Info
            f.write("## System Configuration\n\n")
            f.write(f"- **CPU**: {self.system_config.cpu_model} ({self.system_config.cpu_cores} cores)\n")
            f.write(f"- **RAM**: {self.system_config.total_ram_gb:.1f} GB\n")
            f.write(f"- **GPU**: {self.system_config.gpu_name} ({self.system_config.gpu_count}x)\n")
            f.write(f"- **VRAM**: {self.system_config.total_vram_gb:.1f} GB\n")
            f.write(f"- **Python**: {self.system_config.python_version}\n")
            f.write(f"- **Platform**: {self.system_config.platform}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total Tests**: {len(self.results)}\n")
            f.write(f"- **Passed**: {len(passed)}\n")
            f.write(f"- **Failed**: {len(self.results) - len(passed)}\n\n")
            
            if not passed:
                f.write("⚠️ No successful benchmarks to report.\n")
                return
            
            # Best by RTF
            f.write("## 🏆 Best Performance (Lowest RTF)\n\n")
            best_rtf = sorted(passed, key=lambda x: x.rtf)[:5]
            f.write("| Rank | Framework | Model | Quantization | Device | RTF | Time (s) | VRAM (MB) |\n")
            f.write("|------|-----------|-------|--------------|--------|-----|----------|----------|\n")
            for i, r in enumerate(best_rtf, 1):
                f.write(f"| {i} | {r.framework} | {r.model} | {r.quantization} | {r.device} | "
                       f"{r.rtf:.3f} | {r.total_time_sec:.1f} | {r.vram_peak_mb:.0f} |\n")
            f.write("\n")
            
            # Best by VRAM
            f.write("## 💾 Most Memory Efficient (Lowest VRAM)\n\n")
            best_vram = sorted(passed, key=lambda x: x.vram_peak_mb if x.vram_peak_mb > 0 else float("inf"))[:5]
            f.write("| Rank | Framework | Model | Quantization | Device | VRAM (MB) | RTF | Time (s) |\n")
            f.write("|------|-----------|-------|--------------|--------|-----------|-----|----------|\n")
            for i, r in enumerate(best_vram, 1):
                f.write(f"| {i} | {r.framework} | {r.model} | {r.quantization} | {r.device} | "
                       f"{r.vram_peak_mb:.0f} | {r.rtf:.3f} | {r.total_time_sec:.1f} |\n")
            f.write("\n")
            
            # Best by WER
            wer_results = [r for r in passed if r.wer is not None]
            if wer_results:
                f.write("## 🎯 Most Accurate (Lowest WER)\n\n")
                best_wer = sorted(wer_results, key=lambda x: x.wer)[:5]
                f.write("| Rank | Framework | Model | Quantization | WER (%) | CER (%) | RTF | VRAM (MB) |\n")
                f.write("|------|-----------|-------|--------------|---------|---------|-----|----------|\n")
                for i, r in enumerate(best_wer, 1):
                    f.write(f"| {i} | {r.framework} | {r.model} | {r.quantization} | "
                           f"{r.wer:.2f} | {r.cer:.2f} | {r.rtf:.3f} | {r.vram_peak_mb:.0f} |\n")
                f.write("\n")
            
            # Detailed Results
            f.write("## Detailed Results\n\n")
            f.write("| Framework | Model | Quantization | Device | Beam | RTF | Time (s) | VRAM (MB) | RAM (MB) | WER (%) | Status |\n")
            f.write("|-----------|-------|--------------|--------|------|-----|----------|-----------|----------|---------|--------|\n")
            for r in sorted(passed, key=lambda x: (x.framework, x.model)):
                wer_str = f"{r.wer:.2f}" if r.wer else "N/A"
                f.write(f"| {r.framework} | {r.model} | {r.quantization} | {r.device} | {r.beam_size} | "
                       f"{r.rtf:.3f} | {r.total_time_sec:.1f} | {r.vram_peak_mb:.0f} | {r.ram_peak_mb:.0f} | {wer_str} | {r.status} |\n")
            f.write("\n")
            
            # Recommendations
            f.write("## 💡 Recommendations\n\n")
            
            if len(passed) > 0:
                # Find optimal configurations
                gpu_optimal = min([r for r in passed if r.device == "cuda"], key=lambda x: x.rtf, default=None)
                cpu_optimal = min([r for r in passed if r.device == "cpu"], key=lambda x: x.rtf, default=None)
                
                if gpu_optimal:
                    f.write(f"### Best GPU Performance\n")
                    f.write(f"- **Framework**: {gpu_optimal.framework}\n")
                    f.write(f"- **Model**: {gpu_optimal.model}\n")
                    f.write(f"- **Quantization**: {gpu_optimal.quantization}\n")
                    f.write(f"- **RTF**: {gpu_optimal.rtf:.3f}\n")
                    f.write(f"- **VRAM**: {gpu_optimal.vram_peak_mb:.0f} MB\n\n")
                
                if cpu_optimal:
                    f.write(f"### Best CPU Performance\n")
                    f.write(f"- **Framework**: {cpu_optimal.framework}\n")
                    f.write(f"- **Model**: {cpu_optimal.model}\n")
                    f.write(f"- **Quantization**: {cpu_optimal.quantization}\n")
                    f.write(f"- **RTF**: {cpu_optimal.rtf:.3f}\n")
                    f.write(f"- **RAM**: {cpu_optimal.ram_peak_mb:.0f} MB\n\n")
        
        self.log(f"Report generated: {report_path}")


def create_sample_config() -> str:
    """Create a sample YAML configuration string."""
    return """# Sample Whisper ASR Benchmark Configuration
# Usage: python benchmark.py --config this_file.yaml --audio test.wav

benchmarks:
  # Faster-Whisper with multiple quantizations (shortcut format)
  - framework: faster-whisper
    model: small
    quantizations: [int8_float32, float16]
    devices: [cuda]
    beam_sizes: [1, 3]
    
  # Whisper.cpp with different quantizations
  - framework: whisper.cpp
    model: ggerganov/whisper.cpp
    quantizations: [q5_0, q4_0]
    devices: [cpu]
    threads: 2
    
  # HuggingFace Whisper (Russian fine-tuned)
  - framework: huggingface
    model: sberbank-ai/whisper-small-ru
    quantizations: [float16]
    devices: [cuda]
    load_in_8bit: false
    
  # Full explicit configuration
  - framework: faster-whisper
    model: medium
    quantizations: [int8_float32]
    devices: [cuda, cpu]
    beam_sizes: [1]
    compute_type: int8_float32
    language: ru
"""


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Whisper ASR Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Run benchmarks from YAML config
  python benchmark.py --config benchmark_config.yaml --audio test.wav
  
  # Quick benchmark with default config
  python benchmark.py --audio test.wav --mode quick
  
  # Full benchmark suite
  python benchmark.py --audio test.wav --mode full --config full_config.yaml
  
  # With ground truth for accuracy metrics
  python benchmark.py --audio test.wav --reference ground_truth.txt
  
  # Generate sample config
  python benchmark.py --generate-sample-config
""",
    )
    
    parser.add_argument(
        "--audio",
        type=str,
        required=False,
        default=None,
        help="Path to audio file (WAV 16kHz mono recommended)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to ground truth text file for WER/CER calculation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "full", "compare"],
        help="Benchmark mode",
    )
    parser.add_argument(
        "--generate-sample-config",
        action="store_true",
        help="Generate sample YAML configuration file",
    )
    
    args = parser.parse_args()
    
    # Generate sample config if requested
    if args.generate_sample_config:
        sample_config = create_sample_config()
        output_file = "sample_benchmark_config.yaml"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(sample_config)
        print(f"Sample config written to: {output_file}")
        return 0
    
    # Validate audio file (required for actual benchmarking)
    if not args.audio:
        print("Error: --audio is required for benchmarking")
        return 1
        
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return 1
    
    # Load reference text if provided
    ground_truth = None
    if args.reference:
        if not os.path.exists(args.reference):
            print(f"Error: Reference file not found: {args.reference}")
            return 1
        with open(args.reference, "r", encoding="utf-8") as f:
            ground_truth = f.read().strip()
    
    # Load YAML config
    if args.config:
        yaml_config = parse_yaml_config(args.config)
    else:
        # Use default config if none provided
        yaml_config = {
            "benchmarks": [
                {
                    "framework": "faster-whisper",
                    "model": "small",
                    "quantizations": ["int8_float32"],
                    "devices": ["cuda"],
                    "beam_sizes": [1],
                }
            ]
        }
    
    # Check if we have benchmarks defined
    if not yaml_config.get("benchmarks"):
        print("Error: No benchmarks defined in configuration")
        return 1
    
    # Create and run harness
    harness = BenchmarkHarness(
        yaml_config=yaml_config,
        output_dir=args.output_dir,
        mode=args.mode,
    )
    
    harness.run_all(args.audio, ground_truth)
    
    print(f"\n{'='*60}")
    print("Benchmark completed!")
    print(f"Results saved to: {harness.output_dir}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
