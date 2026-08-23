"""Memory monitoring utilities for RAM and VRAM."""

import os
import subprocess
import threading
import time
from typing import Any


class MemoryMonitor:
    """Monitor RAM and VRAM usage during transcription.
    
    Samples memory usage at regular intervals (default 100ms)
    to capture peak consumption during model loading and inference.
    
    Attributes:
        sampling_interval_ms: Interval between samples in milliseconds.
        peak_ram_mb: Peak RAM usage in MB.
        peak_vram_mb: Peak VRAM usage in MB (None if no GPU).
    """

    def __init__(self, sampling_interval_ms: int = 100) -> None:
        """Initialize memory monitor.
        
        Args:
            sampling_interval_ms: Sampling interval in milliseconds.
        """
        self.sampling_interval_ms = sampling_interval_ms
        self.peak_ram_mb: float = 0.0
        self.peak_vram_mb: float | None = None
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._ram_samples: list[float] = []
        self._vram_samples: list[float] = []
        self._has_gpu: bool = self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check if NVIDIA GPU is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _get_ram_usage_mb(self) -> float:
        """Get current RAM usage in MB using psutil or /proc/self/status."""
        try:
            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback to /proc/self/status on Linux
            try:
                with open("/proc/self/status", "r") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            # VmRSS is in kB
                            return int(line.split()[1]) / 1024
            except Exception:
                pass
        return 0.0

    def _get_vram_usage_mb(self) -> float | None:
        """Get current VRAM usage in MB using nvidia-smi."""
        if not self._has_gpu:
            return None

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse output (may have multiple GPUs)
                values = result.stdout.strip().split("\n")
                # Return max usage across all GPUs
                return max(float(v) for v in values if v.strip())
        except Exception:
            pass
        return None

    def _sample_loop(self) -> None:
        """Background thread loop for sampling memory."""
        while self._running:
            ram = self._get_ram_usage_mb()
            self._ram_samples.append(ram)
            self.peak_ram_mb = max(self.peak_ram_mb, ram)

            if self._has_gpu:
                vram = self._get_vram_usage_mb()
                if vram is not None:
                    self._vram_samples.append(vram)
                    if self.peak_vram_mb is None or vram > self.peak_vram_mb:
                        self.peak_vram_mb = vram

            time.sleep(self.sampling_interval_ms / 1000.0)

    def start(self) -> None:
        """Start memory monitoring."""
        self._running = True
        self._ram_samples = []
        self._vram_samples = []
        self.peak_ram_mb = 0.0
        self.peak_vram_mb = None

        # Take initial sample
        self.peak_ram_mb = self._get_ram_usage_mb()
        self._ram_samples.append(self.peak_ram_mb)

        if self._has_gpu:
            initial_vram = self._get_vram_usage_mb()
            if initial_vram is not None:
                self.peak_vram_mb = initial_vram
                self._vram_samples.append(initial_vram)

        # Start background thread
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop memory monitoring."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dictionary with peak RAM, peak VRAM, and sample counts.
        """
        return {
            "peak_ram_mb": self.peak_ram_mb,
            "peak_vram_mb": self.peak_vram_mb,
            "ram_samples": len(self._ram_samples),
            "vram_samples": len(self._vram_samples),
            "avg_ram_mb": sum(self._ram_samples) / len(self._ram_samples) if self._ram_samples else 0,
            "avg_vram_mb": sum(self._vram_samples) / len(self._vram_samples) if self._vram_samples else None,
        }

    def __enter__(self) -> "MemoryMonitor":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()


def clear_gpu_cache() -> None:
    """Clear GPU cache to free VRAM."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def get_system_info() -> dict[str, Any]:
    """Get system information for benchmarking reports.
    
    Returns:
        Dictionary with CPU, RAM, GPU, and CUDA info.
    """
    info: dict[str, Any] = {}

    # CPU info
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":")[1].strip()
                    break
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    # MemTotal is in kB
                    info["ram_total_gb"] = round(int(line.split()[1]) / (1024 * 1024), 2)
                    break
    except Exception:
        info["cpu_model"] = "Unknown"
        info["ram_total_gb"] = "Unknown"

    # GPU info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpus = result.stdout.strip().split("\n")
            info["gpus"] = []
            for gpu in gpus:
                parts = gpu.split(", ")
                info["gpus"].append({
                    "name": parts[0],
                    "vram_total_gb": round(int(parts[1].split()[0]) / 1024, 2),
                })
    except Exception:
        info["gpus"] = []

    # Python and library versions
    import sys
    info["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["torch_version"] = "Not installed"
        info["cuda_available"] = False

    return info
