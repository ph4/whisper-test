#!/usr/bin/env python3
"""
Whisper Model Benchmarking Harness
===================================
Comprehensive performance and accuracy testing for Whisper models
on systems with limited VRAM (2GB) and RAM (4-8GB).

Supports:
- faster-whisper framework
- whisper.cpp framework
- Multiple models and quantizations
- Russian language optimization
- Detailed memory monitoring (RAM/VRAM)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import threading
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Third-party imports with graceful handling
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory monitoring will be limited.")

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. VRAM monitoring will use nvidia-smi fallback.")

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("Warning: faster-whisper not available. Install with: pip install faster-whisper")

# Configuration constants
DEFAULT_RANDOM_SEED = 42
MEMORY_MONITOR_INTERVAL_MS = 100
WARMUP_RUNS = 1
MAX_AUDIO_DURATION_SEC = 300  # 5 minutes


@dataclass
class SystemConfig:
    """System configuration information"""
    cpu_model: str = ""
    cpu_cores: int = 0
    total_ram_gb: float = 0.0
    gpu_name: str = ""
    gpu_count: int = 0
    total_vram_gb: float = 0.0
    cuda_version: str = ""
    python_version: str = ""
    platform: str = ""


@dataclass
class MemorySnapshot:
    """Memory measurement snapshot"""
    timestamp: float = 0.0
    ram_mb: float = 0.0
    vram_mb: float = 0.0


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    framework: str = ""
    model: str = ""
    quantization: str = ""
    device: str = ""
    beam_size: int = 0
    load_time_sec: float = 0.0
    transcribe_time_sec: float = 0.0
    rtf: float = 0.0
    total_time_sec: float = 0.0
    ram_before_mb: float = 0.0
    ram_after_mb: float = 0.0
    ram_peak_mb: float = 0.0
    vram_before_mb: float = 0.0
    vram_after_mb: float = 0.0
    vram_peak_mb: float = 0.0
    wer: Optional[float] = None
    cer: Optional[float] = None
    word_count: int = 0
    char_count: int = 0
    status: str = "PENDING"
    error_message: str = ""
    transcription: str = ""
    audio_duration_sec: float = 0.0


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run"""
    framework: str = "faster-whisper"
    model: str = "medium"
    quantization: str = "int8_float32"
    device: str = "cuda"
    beam_size: int = 1
    language: str = "ru"
    threads: int = 2
    gpu_id: int = 0


class MemoryMonitor:
    """Monitors RAM and VRAM usage during benchmark"""
    
    def __init__(self, interval_ms: int = MEMORY_MONITOR_INTERVAL_MS):
        self.interval_ms = interval_ms
        self.snapshots: List[MemorySnapshot] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
    def start(self):
        """Start memory monitoring thread"""
        self.running = True
        self.snapshots = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop memory monitoring thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                snapshot = self._take_snapshot()
                with self.lock:
                    self.snapshots.append(snapshot)
            except Exception as e:
                pass  # Silently ignore monitoring errors
            time.sleep(self.interval_ms / 1000.0)
            
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a single memory snapshot"""
        snapshot = MemorySnapshot(timestamp=time.time())
        
        # Get RAM usage
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            snapshot.ram_mb = process.memory_info().rss / (1024 * 1024)
        else:
            # Fallback using /proc/self/status on Linux
            try:
                with open('/proc/self/status', 'r') as f:
                    for line in f:
                        if line.startswith('VmRSS:'):
                            snapshot.ram_mb = float(line.split()[1]) / 1024
                            break
            except:
                snapshot.ram_mb = 0.0
                
        # Get VRAM usage
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                snapshot.vram_mb = info.used / (1024 * 1024)
                pynvml.nvmlShutdown()
            except:
                snapshot.vram_mb = 0.0
        else:
            # Fallback using nvidia-smi
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        snapshot.vram_mb = float(lines[1])
            except:
                snapshot.vram_mb = 0.0
                
        return snapshot
    
    def get_peak_memory(self) -> Tuple[float, float]:
        """Get peak RAM and VRAM usage"""
        with self.lock:
            if not self.snapshots:
                return 0.0, 0.0
            peak_ram = max(s.ram_mb for s in self.snapshots)
            peak_vram = max(s.vram_mb for s in self.snapshots)
        return peak_ram, peak_vram
    
    def get_latest_memory(self) -> Tuple[float, float]:
        """Get latest RAM and VRAM usage"""
        with self.lock:
            if not self.snapshots:
                return 0.0, 0.0
            latest = self.snapshots[-1]
        return latest.ram_mb, latest.vram_mb


def get_system_config() -> SystemConfig:
    """Gather system configuration information"""
    config = SystemConfig()
    
    # CPU info
    if PSUTIL_AVAILABLE:
        config.cpu_cores = psutil.cpu_count(logical=False) or 1
        config.total_ram_gb = psutil.virtual_memory().total / (1024**3)
    
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('model name'):
                    config.cpu_model = line.split(':')[1].strip()
                    break
    except:
        config.cpu_model = "Unknown"
    
    # GPU info
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            config.gpu_count = pynvml.nvmlDeviceGetCount()
            if config.gpu_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                config.gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(config.gpu_name, bytes):
                    config.gpu_name = config.gpu_name.decode('utf-8')
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                config.total_vram_gb = info.total / (1024**3)
            pynvml.nvmlShutdown()
        except:
            pass
    else:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(', ')
                    config.gpu_name = parts[0]
                    if len(parts) > 1:
                        config.total_vram_gb = float(parts[1]) / 1024
                config.gpu_count = len(lines)
        except:
            pass
    
    # CUDA version
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    config.cuda_version = line.strip()
                    break
    except:
        config.cuda_version = "Unknown"
    
    # Python and platform
    import platform
    config.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    config.platform = platform.platform()
    
    return config


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate between reference and hypothesis"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if not ref_words:
        return 0.0 if not hyp_words else 100.0
    
    # Simple Levenshtein distance for WER
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    wer = (dp[m][n] / m) * 100
    return wer


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate between reference and hypothesis"""
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())
    
    if not ref_chars:
        return 0.0 if not hyp_chars else 100.0
    
    m, n = len(ref_chars), len(hyp_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    cer = (dp[m][n] / m) * 100
    return cer


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe or sox"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    
    # Fallback: assume 5 minutes if unable to determine
    return 300.0


def convert_audio_to_wav(input_path: str, output_path: str, sample_rate: int = 16000) -> bool:
    """Convert audio file to WAV format with specified sample rate"""
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-ar', str(sample_rate),
            '-ac', '1',
            '-f', 'wav',
            output_path
        ], check=True, capture_output=True, timeout=300)
        return True
    except Exception as e:
        print(f"Error converting audio: {e}")
        return False


class FasterWhisperRunner:
    """Runner for faster-whisper framework"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = None
        
    def load_model(self) -> Tuple[float, str]:
        """Load the Whisper model"""
        start_time = time.time()
        
        try:
            # Map quantization to compute_type
            compute_type_map = {
                'float32': 'float32',
                'float16': 'float16',
                'int8': 'int8',
                'int8_float16': 'int8_float16',
                'int8_float32': 'int8_float32',
            }
            compute_type = compute_type_map.get(self.config.quantization, 'int8_float32')
            
            # Handle special Russian models
            model_path = self.config.model
            if '/' in model_path:
                # HuggingFace model path
                model_path = model_path
            
            self.model = FasterWhisperModel(
                model_size_or_path=model_path,
                device=self.config.device,
                compute_type=compute_type,
                gpu_id=self.config.gpu_id,
                cpu_threads=self.config.threads if self.config.device == 'cpu' else None,
            )
            
            load_time = time.time() - start_time
            return load_time, ""
            
        except Exception as e:
            load_time = time.time() - start_time
            return load_time, str(e)
    
    def transcribe(self, audio_path: str) -> Tuple[float, str, str]:
        """Transcribe audio file"""
        start_time = time.time()
        
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=self.config.language,
                beam_size=self.config.beam_size,
                vad_filter=True,
            )
            
            transcription = " ".join([segment.text for segment in segments])
            transcribe_time = time.time() - start_time
            
            return transcribe_time, transcription, ""
            
        except Exception as e:
            transcribe_time = time.time() - start_time
            return transcribe_time, "", str(e)
    
    def unload_model(self):
        """Unload model and clear GPU memory"""
        self.model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass


class WhisperCppRunner:
    """Runner for whisper.cpp framework"""
    
    def __init__(self, config: BenchmarkConfig, whisper_cpp_path: str = "./whisper.cpp"):
        self.config = config
        self.whisper_cpp_path = whisper_cpp_path
        self.model_path = None
        
    def _get_model_filename(self) -> str:
        """Get the model filename for whisper.cpp"""
        model_map = {
            'tiny': 'ggml-tiny.bin',
            'tiny.en': 'ggml-tiny.en.bin',
            'base': 'ggml-base.bin',
            'base.en': 'ggml-base.en.bin',
            'small': 'ggml-small.bin',
            'small.en': 'ggml-small.en.bin',
            'medium': 'ggml-medium.bin',
            'medium.en': 'ggml-medium.en.bin',
            'large-v1': 'ggml-large-v1.bin',
            'large-v2': 'ggml-large-v2.bin',
            'large-v3': 'ggml-large-v3.bin',
        }
        
        # Add quantization suffix
        base_name = model_map.get(self.config.model, f'ggml-{self.config.model}.bin')
        quant_suffix = {
            'f32': '',
            'f16': '.q5_0',  # Common quantization
            'q8_0': '.q8_0',
            'q5_0': '.q5_0',
            'q4_0': '.q4_0',
        }
        suffix = quant_suffix.get(self.config.quantization, '')
        
        return base_name.replace('.bin', f'{suffix}.bin') if suffix else base_name
    
    def download_model_if_needed(self) -> bool:
        """Download whisper.cpp model if not present"""
        model_filename = self._get_model_filename()
        model_dir = os.path.join(self.whisper_cpp_path, 'models')
        model_path = os.path.join(model_dir, model_filename)
        
        if os.path.exists(model_path):
            self.model_path = model_path
            return True
        
        # Try to download using provided script
        try:
            os.makedirs(model_dir, exist_ok=True)
            download_script = os.path.join(self.whisper_cpp_path, 'models', 'download-ggml-model.sh')
            if os.path.exists(download_script):
                model_name = self.config.model.split('-')[0]  # Get base name
                subprocess.run([download_script, model_name], 
                             cwd=os.path.join(self.whisper_cpp_path, 'models'),
                             check=True, timeout=600)
                if os.path.exists(model_path):
                    self.model_path = model_path
                    return True
        except Exception as e:
            print(f"Error downloading model: {e}")
        
        return False
    
    def load_model(self) -> Tuple[float, str]:
        """Load the Whisper model (whisper.cpp loads on demand)"""
        start_time = time.time()
        
        if not self.download_model_if_needed():
            return 0.0, f"Model not found: {self._get_model_filename()}"
        
        # whisper.cpp doesn't have explicit load, it loads during inference
        load_time = time.time() - start_time
        return load_time, ""
    
    def transcribe(self, audio_path: str) -> Tuple[float, str, str]:
        """Transcribe audio file using whisper.cpp"""
        start_time = time.time()
        
        try:
            cmd = [
                os.path.join(self.whisper_cpp_path, 'main'),
                '-m', self.model_path,
                '-f', audio_path,
                '-l', self.config.language,
                '--threads', str(self.config.threads),
            ]
            
            # Add GPU acceleration if available and configured
            if self.config.device == 'cuda':
                cmd.extend(['-ngl', '99'])  # Use all GPU layers
            
            # Add beam size
            if self.config.beam_size > 1:
                cmd.extend(['--beam-size', str(self.config.beam_size)])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            transcribe_time = time.time() - start_time
            
            if result.returncode != 0:
                return transcribe_time, "", result.stderr
            
            # Parse output - whisper.cpp outputs to stdout
            transcription = result.stdout
            # Extract just the text, removing timestamps if present
            lines = transcription.split('\n')
            text_lines = [line for line in lines if not line.startswith('[')]
            transcription = '\n'.join(text_lines).strip()
            
            return transcribe_time, transcription, ""
            
        except subprocess.TimeoutExpired:
            return 300.0, "", "Transcription timed out"
        except Exception as e:
            transcribe_time = time.time() - start_time
            return transcribe_time, "", str(e)
    
    def unload_model(self):
        """No explicit unload needed for whisper.cpp"""
        pass


class BenchmarkHarness:
    """Main benchmark harness for Whisper models"""
    
    def __init__(self, args):
        self.args = args
        self.results: List[BenchmarkResult] = []
        self.system_config = get_system_config()
        self.memory_monitor = MemoryMonitor(interval_ms=args.monitor_memory_interval)
        
        # Output directory setup
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging setup
        self.log_file = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Define test configurations based on mode
        self.test_configs = self._generate_test_configs()
        
    def log(self, message: str):
        """Log message to console and file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def _generate_test_configs(self) -> List[BenchmarkConfig]:
        """Generate test configurations based on mode"""
        configs = []
        
        # Parse arguments
        frameworks = self.args.frameworks.split(',')
        models = self.args.models.split(',')
        quantizations = self.args.quantizations.split(',')
        beam_sizes = [int(x) for x in self.args.beam_sizes.split(',')]
        
        # Filter based on mode
        if self.args.mode == 'quick':
            models = ['small', 'medium']
            beam_sizes = [1, 3]
            if 'large' in str(models):
                models = [m for m in models if 'large' not in m]
        
        # Generate all combinations
        for framework in frameworks:
            for model in models:
                # Skip large models on quick mode or if VRAM is limited
                if self.args.mode == 'quick' and 'large' in model:
                    continue
                    
                # Check if model should be skipped due to VRAM limits
                if self.system_config.total_vram_gb < 2.5 and 'large' in model:
                    if self.args.mode != 'full':
                        self.log(f"Skipping {model} on {framework}: insufficient VRAM ({self.system_config.total_vram_gb:.1f}GB)")
                        continue
                
                for quantization in quantizations:
                    # Skip incompatible quantizations
                    if framework == 'whisper.cpp' and quantization not in ['f32', 'f16', 'q8_0', 'q5_0', 'q4_0']:
                        continue
                    
                    for device in ['cuda', 'cpu']:
                        # Skip CUDA if not available
                        if device == 'cuda' and self.system_config.gpu_count == 0:
                            continue
                        
                        for beam_size in beam_sizes:
                            config = BenchmarkConfig(
                                framework=framework,
                                model=model,
                                quantization=quantization,
                                device=device,
                                beam_size=beam_size,
                                language=self.args.language,
                                threads=self.args.threads,
                                gpu_id=self.args.gpu_id,
                            )
                            configs.append(config)
        
        return configs
    
    def _check_russian_support(self, model: str, framework: str) -> bool:
        """Check if model supports Russian language"""
        russian_models = [
            'bond005/whisper_large_v2_ru',
            ' whisper-large-v2-russian',
            'sber-whisper',
            'yandex-whisper',
        ]
        
        # Standard Whisper models support Russian
        standard_models = ['tiny', 'base', 'small', 'medium', 'large']
        
        if any(ru_model in model.lower() for ru_model in russian_models):
            return True
        
        if any(std in model.lower() for std in standard_models):
            # large-v2 and large-v3 have better Russian support
            if 'large-v2' in model or 'large-v3' in model or 'large' in model:
                return True
            return True  # Most standard models support Russian
        
        return False
    
    def run_single_benchmark(self, config: BenchmarkConfig, audio_path: str, 
                            ground_truth: Optional[str] = None) -> BenchmarkResult:
        """Run a single benchmark configuration"""
        result = BenchmarkResult(
            framework=config.framework,
            model=config.model,
            quantization=config.quantization,
            device=config.device,
            beam_size=config.beam_size,
            language=config.language,
        )
        
        self.log(f"\n{'='*60}")
        self.log(f"Testing: {config.framework} | {config.model} | {config.quantization} | "
                f"{config.device} | beam={config.beam_size}")
        self.log(f"{'='*60}")
        
        # Check Russian language support
        if not self._check_russian_support(config.model, config.framework):
            self.log(f"Warning: Model '{config.model}' may not have optimal Russian language support")
        
        # Initialize runner
        if config.framework == 'faster-whisper':
            if not FASTER_WHISPER_AVAILABLE:
                result.status = "SKIP"
                result.error_message = "faster-whisper not installed"
                return result
            runner = FasterWhisperRunner(config)
        elif config.framework == 'whisper.cpp':
            runner = WhisperCppRunner(config, self.args.whisper_cpp_path)
        else:
            result.status = "SKIP"
            result.error_message = f"Unknown framework: {config.framework}"
            return result
        
        # Get initial memory readings
        time.sleep(0.5)  # Allow memory to stabilize
        initial_ram, initial_vram = self.memory_monitor.get_latest_memory()
        result.ram_before_mb = initial_ram
        result.vram_before_mb = initial_vram
        
        # Load model
        self.log("Loading model...")
        load_time, load_error = runner.load_model()
        result.load_time_sec = load_time
        
        if load_error:
            if "OOM" in load_error.upper() or "memory" in load_error.lower():
                result.status = "OOM"
                result.error_message = load_error
                self.log(f"FAILED (OOM): {load_error}")
                return result
            else:
                result.status = "FAIL"
                result.error_message = load_error
                self.log(f"FAILED: {load_error}")
                return result
        
        # Get memory after loading
        time.sleep(0.5)
        loaded_ram, loaded_vram = self.memory_monitor.get_latest_memory()
        result.ram_after_mb = loaded_ram
        result.vram_after_mb = loaded_vram
        
        self.log(f"Model loaded in {load_time:.2f}s | RAM: {loaded_ram:.0f}MB (+{loaded_ram-initial_ram:.0f}MB) | "
                f"VRAM: {loaded_vram:.0f}MB (+{loaded_vram-initial_vram:.0f}MB)")
        
        # Start memory monitoring
        self.memory_monitor.start()
        
        # Warm-up run (not counted in metrics)
        if self.args.warmup_runs > 0:
            self.log("Running warm-up...")
            try:
                runner.transcribe(audio_path)
                time.sleep(1.0)  # Allow GPU to cool down
            except:
                pass
        
        # Actual transcription
        self.log("Transcribing...")
        transcribe_time, transcription, transcribe_error = runner.transcribe(audio_path)
        result.transcribe_time_sec = transcribe_time
        result.transcription = transcription
        
        # Stop memory monitoring
        self.memory_monitor.stop()
        
        # Get peak memory
        peak_ram, peak_vram = self.memory_monitor.get_peak_memory()
        result.ram_peak_mb = max(peak_ram, loaded_ram)
        result.vram_peak_mb = max(peak_vram, loaded_vram)
        
        # Unload model
        runner.unload_model()
        time.sleep(0.5)
        
        # Calculate metrics
        result.audio_duration_sec = get_audio_duration(audio_path)
        if result.audio_duration_sec > 0:
            result.rtf = transcribe_time / result.audio_duration_sec
        result.total_time_sec = load_time + transcribe_time
        
        # Check for errors
        if transcribe_error:
            if "OOM" in transcribe_error.upper() or "memory" in transcribe_error.lower():
                result.status = "OOM"
            else:
                result.status = "FAIL"
            result.error_message = transcribe_error
            self.log(f"FAILED ({result.status}): {transcribe_error}")
            return result
        
        # Calculate accuracy metrics if ground truth available
        if ground_truth:
            result.wer = calculate_wer(ground_truth, transcription)
            result.cer = calculate_cer(ground_truth, transcription)
            result.word_count = len(transcription.split())
            result.char_count = len(transcription)
        
        result.status = "PASS"
        
        self.log(f"PASSED | RTF: {result.rtf:.3f} | Total: {result.total_time_sec:.2f}s | "
                f"RAM Peak: {result.ram_peak_mb:.0f}MB | VRAM Peak: {result.vram_peak_mb:.0f}MB")
        
        if ground_truth:
            self.log(f"WER: {result.wer:.2f}% | CER: {result.cer:.2f}% | Words: {result.word_count}")
        
        return result
    
    def run_all_benchmarks(self, audio_path: str, ground_truth: Optional[str] = None):
        """Run all benchmark configurations"""
        self.log(f"\nStarting benchmark suite")
        self.log(f"Audio file: {audio_path}")
        self.log(f"Total configurations: {len(self.test_configs)}")
        self.log(f"Mode: {self.args.mode}")
        
        total_start = time.time()
        
        for i, config in enumerate(self.test_configs, 1):
            self.log(f"\n[{i}/{len(self.test_configs)}]")
            
            try:
                result = self.run_single_benchmark(config, audio_path, ground_truth)
                self.results.append(result)
                
                # Save intermediate results
                if i % 5 == 0:
                    self.save_results()
                    
            except Exception as e:
                self.log(f"ERROR running benchmark: {e}")
                traceback.print_exc()
                
                # Create error result
                error_result = BenchmarkResult(
                    framework=config.framework,
                    model=config.model,
                    quantization=config.quantization,
                    device=config.device,
                    beam_size=config.beam_size,
                    status="ERROR",
                    error_message=str(e),
                )
                self.results.append(error_result)
            
            # Clear GPU memory between tests
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
            
            time.sleep(2.0)  # Cooldown between tests
        
        total_time = time.time() - total_start
        self.log(f"\n{'='*60}")
        self.log(f"Benchmark suite completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        self.log(f"Results: {len([r for r in self.results if r.status == 'PASS'])} passed, "
                f"{len([r for r in self.results if r.status in ['FAIL', 'OOM', 'ERROR']])} failed")
        
        # Save final results
        self.save_results()
        self.generate_report()
    
    def save_results(self):
        """Save results to CSV and JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save CSV
        csv_path = self.output_dir / f"results_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                'framework', 'model', 'quantization', 'device', 'beam_size',
                'load_time_sec', 'transcribe_time_sec', 'rtf', 'total_time_sec',
                'ram_before_mb', 'ram_after_mb', 'ram_peak_mb',
                'vram_before_mb', 'vram_after_mb', 'vram_peak_mb',
                'wer', 'cer', 'word_count', 'char_count',
                'status', 'error_message'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {k: getattr(result, k, '') for k in fieldnames}
                writer.writerow(row)
        
        # Save JSON
        json_path = self.output_dir / f"results_{timestamp}.json"
        json_data = {
            'system_config': asdict(self.system_config),
            'benchmark_args': vars(self.args),
            'timestamp_start': datetime.now().isoformat(),
            'audio_path': self.args.audio,
            'results': [asdict(r) for r in self.results],
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        self.log(f"Results saved to: {csv_path}")
        self.log(f"JSON saved to: {json_path}")
    
    def generate_report(self):
        """Generate Markdown report"""
        report_path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # Filter successful results
        passed = [r for r in self.results if r.status == 'PASS']
        
        with open(report_path, 'w') as f:
            f.write("# Whisper Model Benchmark Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # System Info
            f.write("## System Configuration\n\n")
            f.write(f"- **CPU**: {self.system_config.cpu_model}\n")
            f.write(f"- **CPU Cores**: {self.system_config.cpu_cores}\n")
            f.write(f"- **RAM**: {self.system_config.total_ram_gb:.1f} GB\n")
            f.write(f"- **GPU**: {self.system_config.gpu_name}\n")
            f.write(f"- **GPU Count**: {self.system_config.gpu_count}\n")
            f.write(f"- **VRAM**: {self.system_config.total_vram_gb:.1f} GB\n")
            f.write(f"- **CUDA**: {self.system_config.cuda_version}\n")
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
            f.write("| Rank | Framework | Model | Quantization | Device | Beam | RTF | Time (s) | VRAM (MB) |\n")
            f.write("|------|-----------|-------|--------------|--------|------|-----|----------|----------|\n")
            for i, r in enumerate(best_rtf, 1):
                f.write(f"| {i} | {r.framework} | {r.model} | {r.quantization} | {r.device} | {r.beam_size} | "
                       f"{r.rtf:.3f} | {r.total_time_sec:.1f} | {r.vram_peak_mb:.0f} |\n")
            f.write("\n")
            
            # Best by VRAM
            f.write("## 💾 Most Memory Efficient (Lowest VRAM)\n\n")
            best_vram = sorted(passed, key=lambda x: x.vram_peak_mb if x.vram_peak_mb > 0 else float('inf'))[:5]
            f.write("| Rank | Framework | Model | Quantization | Device | Beam | VRAM (MB) | RTF | Time (s) |\n")
            f.write("|------|-----------|-------|--------------|--------|------|-----------|-----|----------|\n")
            for i, r in enumerate(best_vram, 1):
                f.write(f"| {i} | {r.framework} | {r.model} | {r.quantization} | {r.device} | {r.beam_size} | "
                       f"{r.vram_peak_mb:.0f} | {r.rtf:.3f} | {r.total_time_sec:.1f} |\n")
            f.write("\n")
            
            # Best by WER (if available)
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
            
            # Detailed Results by Framework
            for framework in ['faster-whisper', 'whisper.cpp']:
                fw_results = [r for r in passed if r.framework == framework]
                if fw_results:
                    f.write(f"## {framework} Results\n\n")
                    f.write("| Model | Quantization | Device | Beam | RTF | Time (s) | VRAM (MB) | RAM (MB) | Status |\n")
                    f.write("|-------|--------------|--------|------|-----|----------|-----------|----------|--------|\n")
                    for r in sorted(fw_results, key=lambda x: (x.model, x.quantization, x.device, x.beam_size)):
                        f.write(f"| {r.model} | {r.quantization} | {r.device} | {r.beam_size} | "
                               f"{r.rtf:.3f} | {r.total_time_sec:.1f} | {r.vram_peak_mb:.0f} | "
                               f"{r.ram_peak_mb:.0f} | {r.status} |\n")
                    f.write("\n")
            
            # Recommendations
            f.write("## 💡 Recommendations\n\n")
            
            # Find optimal for different scenarios
            gpu_optimal = min([r for r in passed if r.device == 'cuda' and r.vram_peak_mb < 2000], 
                            key=lambda x: x.rtf, default=None)
            cpu_optimal = min([r for r in passed if r.device == 'cpu'], 
                            key=lambda x: x.rtf, default=None)
            balanced = min(passed, key=lambda x: x.rtf * x.vram_peak_mb / 1000, default=None)
            
            if gpu_optimal:
                f.write(f"### Best GPU Performance (< 2GB VRAM)\n")
                f.write(f"- **Model**: {gpu_optimal.model}\n")
                f.write(f"- **Framework**: {gpu_optimal.framework}\n")
                f.write(f"- **Quantization**: {gpu_optimal.quantization}\n")
                f.write(f"- **RTF**: {gpu_optimal.rtf:.3f}\n")
                f.write(f"- **VRAM**: {gpu_optimal.vram_peak_mb:.0f} MB\n\n")
            
            if cpu_optimal:
                f.write(f"### Best CPU Performance\n")
                f.write(f"- **Model**: {cpu_optimal.model}\n")
                f.write(f"- **Framework**: {cpu_optimal.framework}\n")
                f.write(f"- **Quantization**: {cpu_optimal.quantization}\n")
                f.write(f"- **RTF**: {cpu_optimal.rtf:.3f}\n")
                f.write(f"- **RAM**: {cpu_optimal.ram_peak_mb:.0f} MB\n\n")
            
            if balanced:
                f.write(f"### Best Balance (Speed × Memory)\n")
                f.write(f"- **Model**: {balanced.model}\n")
                f.write(f"- **Framework**: {balanced.framework}\n")
                f.write(f"- **Quantization**: {balanced.quantization}\n")
                f.write(f"- **RTF**: {balanced.rtf:.3f}\n")
                f.write(f"- **VRAM**: {balanced.vram_peak_mb:.0f} MB\n")
                f.write(f"- **Beam Size**: {balanced.beam_size}\n\n")
            
            # Russian-specific models tested
            russian_models = [r for r in passed if any(x in r.model.lower() 
                              for x in ['ru', 'russian', 'sber', 'yandex', 'bond'])]
            if russian_models:
                f.write("## 🇷🇺 Russian-Specific Models\n\n")
                f.write("| Model | Framework | RTF | WER (%) | VRAM (MB) |\n")
                f.write("|-------|-----------|-----|---------|----------|\n")
                for r in russian_models:
                    wer_str = f"{r.wer:.2f}" if r.wer else "N/A"
                    f.write(f"| {r.model} | {r.framework} | {r.rtf:.3f} | {wer_str} | {r.vram_peak_mb:.0f} |\n")
                f.write("\n")
        
        self.log(f"Report generated: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Whisper Model Benchmarking Harness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with common configurations
  python whisper_benchmark.py --audio test.wav --mode quick
  
  # Full benchmark with all configurations
  python whisper_benchmark.py --audio test.wav --mode full \\
      --frameworks faster-whisper,whisper.cpp \\
      --models tiny,base,small,medium,large-v3 \\
      --beam-sizes 1,3,5
  
  # Compare frameworks only
  python whisper_benchmark.py --audio test.wav --mode compare \\
      --models small,medium
  
  # With ground truth for accuracy metrics
  python whisper_benchmark.py --audio test.wav --ground-truth "Reference text" \\
      --language ru
        """
    )
    
    parser.add_argument('--audio', type=str, required=True,
                       help='Path to audio file (WAV 16kHz mono recommended)')
    parser.add_argument('--ground-truth', type=str, default=None,
                       help='Ground truth transcription for WER/CER calculation')
    parser.add_argument('--frameworks', type=str, default='faster-whisper',
                       help='Comma-separated list of frameworks (faster-whisper,whisper.cpp)')
    parser.add_argument('--models', type=str, 
                       default='tiny,base,small,medium,large-v2,large-v3',
                       help='Comma-separated list of models to test')
    parser.add_argument('--quantizations', type=str, 
                       default='int8,int8_float16,int8_float32,float16,float32',
                       help='Comma-separated list of quantizations')
    parser.add_argument('--beam-sizes', type=str, default='1,3,5',
                       help='Comma-separated list of beam sizes')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['quick', 'full', 'compare', 'optimal'],
                       help='Benchmark mode')
    parser.add_argument('--language', type=str, default='ru',
                       help='Language code for transcription')
    parser.add_argument('--threads', type=int, default=2,
                       help='Number of CPU threads for whisper.cpp')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU ID for multi-GPU systems')
    parser.add_argument('--whisper-cpp-path', type=str, default='./whisper.cpp',
                       help='Path to whisper.cpp installation')
    parser.add_argument('--monitor-memory-interval', type=int, default=100,
                       help='Memory monitoring interval in ms')
    parser.add_argument('--warmup-runs', type=int, default=1,
                       help='Number of warm-up runs before measurement')
    parser.add_argument('--convert-audio', action='store_true',
                       help='Convert audio to WAV 16kHz mono if needed')
    
    args = parser.parse_args()
    
    # Validate audio file
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
    
    # Convert audio if requested
    audio_path = args.audio
    if args.convert_audio:
        wav_path = os.path.splitext(args.audio)[0] + '_converted.wav'
        print(f"Converting audio to WAV 16kHz mono...")
        if convert_audio_to_wav(args.audio, wav_path):
            audio_path = wav_path
            print(f"Converted: {wav_path}")
        else:
            print("Warning: Audio conversion failed, using original file")
    
    # Create and run benchmark
    harness = BenchmarkHarness(args)
    harness.run_all_benchmarks(audio_path, args.ground_truth)
    
    print(f"\n{'='*60}")
    print(f"Benchmark completed!")
    print(f"Results saved to: {harness.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
