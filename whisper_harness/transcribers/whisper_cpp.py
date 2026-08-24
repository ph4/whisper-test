"""Whisper.cpp transcriber implementation."""

import os
import subprocess
import tempfile
import time
from typing import Any

from transcribers.base import Transcriber


class WhisperCppTranscriber(Transcriber):
    """Transcriber using whisper.cpp via pywhispercpp or subprocess.
    
    Supports GGML/GGUF quantized models for efficient CPU/GPU inference.
    Automatically downloads models from HuggingFace if not present locally.
    
    Attributes:
        quantization: Quantization type (q5_0, q8_0, f16, f32).
        n_threads: Number of CPU threads (default 2 for A4-5300).
        use_gpu: Enable GPU acceleration via CUDA.
    """

    def __init__(
        self,
        model_id: str = "ggerganov/whisper.cpp",
        device: str = "cpu",
        quantization: str = "q5_0",
        n_threads: int = 2,
        use_gpu: bool = False,
        gpu_layers: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Whisper.cpp transcriber.
        
        Args:
            model_id: HuggingFace repo id or local path to model.
            device: Device ('cpu' or 'cuda').
            quantization: Quantization type for GGML model.
            n_threads: Number of CPU threads.
            use_gpu: Enable GPU acceleration.
            gpu_layers: Number of layers to offload to GPU (for partial/full offloading).
                       If None and use_gpu=True, attempts full offloading.
            **kwargs: Additional parameters.
        """
        super().__init__(model_id, device, **kwargs)
        self.quantization = quantization
        self.n_threads = n_threads
        self.use_gpu = use_gpu if device == "cuda" else False
        self.gpu_layers = gpu_layers
        self._model_path: str | None = None

    def _download_model(self) -> str:
        """Download model from HuggingFace if needed."""
        try:
            from huggingface_hub import hf_hub_download

            # Map quantization to filename pattern
            quant_map = {
                "q4_0": "q4_0",
                "q4_1": "q4_1",
                "q5_0": "q5_0",
                "q5_1": "q5_1",
                "q8_0": "q8_0",
                "f16": "f16",
                "f32": "f32",
            }

            quant_suffix = quant_map.get(self.quantization, self.quantization)

            # Determine model file based on model_id
            if self.model_id == "ggerganov/whisper.cpp":
                # Default whisper.cpp models
                model_file = f"ggml-{quant_suffix}.bin"
            else:
                # Custom models (e.g., Russian fine-tunes)
                model_file = f"ggml-model-{quant_suffix}.bin"

            model_path = hf_hub_download(
                repo_id=self.model_id,
                filename=model_file,
                cache_dir=os.path.expanduser("~/.cache/whisper.cpp"),
            )
            return model_path

        except ImportError:
            raise RuntimeError(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}") from e

    def _load_model(self) -> None:
        """Load Whisper.cpp model (download if needed)."""
        start_time = time.perf_counter()

        try:
            # Check if model_id is a local path
            if os.path.exists(self.model_id):
                self._model_path = self.model_id
            else:
                self._model_path = self._download_model()

            self._model = self._model_path
            self._load_time = time.perf_counter() - start_time

        except Exception as e:
            raise RuntimeError(f"Failed to load whisper.cpp model: {e}") from e

    def _transcribe_with_pywhispercpp(self, audio_path: str, language: str) -> dict[str, Any] | None:
        """Try transcription using pywhispercpp library with GPU offloading."""
        try:
            from pywhispercpp.model import Model

            # Configure GPU offloading
            model_kwargs = {
                "model": self._model_path,
                "language": language,
                "threads": self.n_threads,
            }
            
            # Add GPU offloading if enabled
            if self.use_gpu:
                model_kwargs["use_gpu"] = True
                if self.gpu_layers is not None:
                    model_kwargs["n_gpu_layers"] = self.gpu_layers
                else:
                    # Full offloading by default when GPU is enabled
                    model_kwargs["n_gpu_layers"] = 999

            model = Model(**model_kwargs)

            start_time = time.perf_counter()
            segments = model.transcribe(audio_path)
            transcribe_time = time.perf_counter() - start_time

            text = " ".join(seg.text for seg in segments)

            return {
                "text": text.strip(),
                "transcribe_time": transcribe_time,
            }

        except ImportError:
            return None
        except Exception:
            return None

    def _transcribe_with_subprocess(self, audio_path: str, language: str) -> dict[str, Any]:
        """Fallback transcription using whisper.cpp binary via subprocess with GPU offloading."""
        # Find whisper.cpp binary
        binary_names = ["whisper-cli", "main", "whisper-main"]
        binary_path: str | None = None

        for name in binary_names:
            # Check common locations
            candidates = [
                f"/usr/local/bin/{name}",
                f"/usr/bin/{name}",
                os.path.expanduser(f"~/whisper.cpp/{name}"),
                os.path.expanduser(f"~/.local/bin/{name}"),
            ]
            for candidate in candidates:
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    binary_path = candidate
                    break
            if binary_path:
                break

        if not binary_path:
            raise RuntimeError(
                "whisper.cpp binary not found. Install whisper.cpp or use pywhispercpp."
            )

        start_time = time.perf_counter()

        cmd = [
            binary_path,
            "-m", self._model_path,  # type: ignore[list-item]
            "-f", audio_path,
            "-l", language,
            "-t", str(self.n_threads),
            "--no-timestamps",
        ]

        # Configure GPU offloading
        if self.use_gpu:
            if self.gpu_layers is not None:
                # Partial offloading: specific number of layers
                cmd.extend(["-ngl", str(self.gpu_layers)])
            else:
                # Full offloading: all layers to GPU
                cmd.extend(["-ngl", "999"])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        transcribe_time = time.perf_counter() - start_time

        if result.returncode != 0:
            raise RuntimeError(f"whisper.cpp failed: {result.stderr}")

        return {
            "text": result.stdout.strip(),
            "transcribe_time": transcribe_time,
        }

    def transcribe(
        self, audio_path: str, language: str = "ru"
    ) -> dict[str, Any]:
        """Transcribe audio using whisper.cpp.
        
        Args:
            audio_path: Path to audio file.
            language: Language code ('ru' for Russian).
            
        Returns:
            Dictionary with transcription results and metrics.
            
        Raises:
            FileNotFoundError: If audio file doesn't exist.
            RuntimeError: If transcription fails.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Lazy loading
        self._ensure_loaded()

        if self._model_path is None:
            raise RuntimeError("Model not loaded")

        # Import memory monitor
        from utils.memory_monitor import MemoryMonitor

        monitor = MemoryMonitor()
        monitor.start()

        # Try pywhispercpp first, fallback to subprocess
        result = self._transcribe_with_pywhispercpp(audio_path, language)

        if result is None:
            result = self._transcribe_with_subprocess(audio_path, language)

        monitor.stop()

        # Get audio duration using ffprobe or simple estimation
        duration = self._get_audio_duration(audio_path)

        return {
            "text": result["text"],
            "duration": duration,
            "transcribe_time": result["transcribe_time"],
            "load_time": self._load_time,
            "model_name": f"whisper.cpp-{self.model_id}-{self.quantization}",
            "memory_peak_mb": monitor.peak_ram_mb,
            "vram_peak_mb": monitor.peak_vram_mb if monitor.peak_vram_mb else None,
            "framework": "whisper.cpp",
            "device": "gpu" if self.use_gpu else "cpu",
            "threads": self.n_threads,
            "quantization": self.quantization,
            "gpu_layers": self.gpu_layers if self.use_gpu else 0,
        }

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return float(result.stdout.strip())
        except Exception:
            # Fallback: assume 1 second per MB (rough estimate)
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            return max(file_size_mb, 1.0)
