"""Transcribe.cpp transcriber for GGUF quantized models."""

import os
import time
from typing import Any

from .base import Transcriber


class TranscribeCppTranscriber(Transcriber):
    """Transcriber using whisper.cpp/transcribe.cpp for GGUF models.
    
    Supports running GGUF quantized models from HuggingFace or local paths.
    Ideal for memory-constrained systems with various quantization levels.
    
    Attributes:
        quantization: GGUF quantization type (Q4_0, Q5_K_M, Q6_K, Q8_0, etc.).
        n_threads: Number of CPU threads for inference.
        use_gpu: Enable GPU acceleration if available.
        gpu_layers: Number of layers to offload to GPU (for partial/full offloading).
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        quantization: str = "Q5_K_M",
        n_threads: int = 4,
        use_gpu: bool = False,
        gpu_layers: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Transcribe.cpp transcriber.
        
        Args:
            model_id: Model identifier (HuggingFace repo/model or local path to .gguf file).
                     Examples: "handy-computer/gigaam-v3-e2e-rnnt-gguf", "/path/to/model.gguf"
            device: Device ('cpu' or 'cuda').
            quantization: GGUF quantization type (Q4_0, Q4_K_S, Q5_K_M, Q6_K, Q8_0, F16, F32).
            n_threads: Number of CPU threads for inference.
            use_gpu: Enable GPU acceleration (requires CUDA).
            gpu_layers: Number of transformer layers to offload to GPU.
                       If None and use_gpu=True, attempts full offloading (999 layers).
                       For partial offloading, set to specific number (e.g., 10, 20).
            **kwargs: Additional parameters passed to whisper_main.
        """
        super().__init__(model_id, device, **kwargs)
        self.quantization = quantization
        self.n_threads = n_threads
        self.use_gpu = use_gpu if device == "cuda" else False
        self.gpu_layers = gpu_layers
        self._model_path: str | None = None
        self._downloaded_model: bool = False

    def _download_gguf_model(self) -> str:
        """Download GGUF model from HuggingFace if needed."""
        from huggingface_hub import hf_hub_download, list_repo_files
        
        # Check if model_id is a local path
        if os.path.isfile(self.model_id):
            return self.model_id
        
        # Try to find GGUF files in the repo
        try:
            repo_files = list_repo_files(self.model_id)
            gguf_files = [f for f in repo_files if f.endswith('.gguf')]
            
            if not gguf_files:
                raise RuntimeError(f"No GGUF files found in repo: {self.model_id}")
            
            # Select appropriate quantization or first available
            target_file = None
            quant_lower = self.quantization.lower()
            
            # Try exact match first
            for f in gguf_files:
                if quant_lower in f.lower():
                    target_file = f
                    break
            
            # Fallback to first available if no match
            if target_file is None:
                target_file = gguf_files[0]
                print(f"⚠️  Quantization '{self.quantization}' not found. Using: {target_file}")
            
            # Download the model
            model_path = hf_hub_download(
                repo_id=self.model_id,
                filename=target_file,
                cache_dir=os.path.expanduser("~/.cache/whisper-cpp"),
            )
            
            self._downloaded_model = True
            return model_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to download GGUF model: {e}") from e

    def _load_model(self) -> None:
        """Load GGUF model path (lazy loading - actual loading happens during transcription)."""
        start_time = time.perf_counter()
        
        try:
            # Get the model path (download if needed)
            self._model_path = self._download_gguf_model()
            
            # Verify file exists
            if not os.path.exists(self._model_path):
                raise FileNotFoundError(f"GGUF model not found: {self._model_path}")
            
            # Import pywhispercpp or whisper.cpp Python bindings
            try:
                import pywhispercpp
                self._backend = "pywhispercpp"
            except ImportError:
                try:
                    import whisper_cpp
                    self._backend = "whisper_cpp"
                except ImportError:
                    raise RuntimeError(
                        "Neither pywhispercpp nor whisper.cpp Python bindings installed.\n"
                        "Install with: pip install pywhispercpp\n"
                        "or: pip install whisper-cpp-python"
                    )
            
            self._load_time = time.perf_counter() - start_time
            
        except Exception as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                raise RuntimeError(
                    f"OOM: Cannot load GGUF model {self.model_id}. "
                    f"Try a more quantized version (Q4_0, Q4_K_S)."
                ) from e
            raise RuntimeError(f"Failed to load GGUF model: {e}") from e

    def transcribe(
        self, audio_path: str, language: str = "ru"
    ) -> dict[str, Any]:
        """Transcribe audio using whisper.cpp with GGUF model.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.).
            language: Language code ('ru' for Russian, 'en' for English, etc.).
            
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
            raise RuntimeError("Model path not set")

        # Import memory monitor
        from utils.memory_monitor import MemoryMonitor

        monitor = MemoryMonitor()
        monitor.start()

        start_time = time.perf_counter()
        
        try:
            if self._backend == "pywhispercpp":
                text = self._transcribe_pywhispercpp(audio_path, language)
            elif self._backend == "whisper_cpp":
                text = self._transcribe_whisper_cpp(audio_path, language)
            else:
                raise RuntimeError("Unknown backend")
                
            transcribe_time = time.perf_counter() - start_time
            
        except Exception as e:
            monitor.stop()
            raise RuntimeError(f"Transcription failed: {e}") from e

        monitor.stop()
        
        # Get audio duration
        duration = self._get_audio_duration(audio_path)

        return {
            "text": text.strip(),
            "duration": duration,
            "transcribe_time": transcribe_time,
            "load_time": self._load_time,
            "model_name": f"transcribe-cpp-{os.path.basename(self._model_path)}",
            "memory_peak_mb": monitor.peak_ram_mb,
            "vram_peak_mb": monitor.peak_vram_mb if monitor.peak_vram_mb else None,
            "framework": "transcribe.cpp",
            "device": "gpu" if self.use_gpu else "cpu",
            "quantization": self.quantization,
            "threads": self.n_threads,
            "gpu_layers": self.gpu_layers if self.use_gpu else 0,
        }

    def _transcribe_pywhispercpp(self, audio_path: str, language: str) -> str:
        """Transcribe using pywhispercpp library with GPU offloading."""
        import pywhispercpp
        import numpy as np
        from scipy.io import wavfile
        
        # Load and preprocess audio
        sample_rate, audio = wavfile.read(audio_path)
        audio = audio.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            from scipy.signal import resample
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = resample(audio, num_samples)
        
        # Configure model parameters with GPU offloading
        model_kwargs = {
            "model_path": self._model_path,
            "n_threads": self.n_threads,
            "language": language,
        }
        
        # Add GPU offloading if enabled
        if self.use_gpu:
            model_kwargs["use_gpu"] = True
            if self.gpu_layers is not None:
                model_kwargs["n_gpu_layers"] = self.gpu_layers
            else:
                # Full offloading by default
                model_kwargs["n_gpu_layers"] = 999
        
        # Initialize model
        model = pywhispercpp.Model(**model_kwargs)
        
        # Run transcription
        segments = model.transcribe(audio)
        text = " ".join([seg.text for seg in segments])
        
        return text

    def _transcribe_whisper_cpp(self, audio_path: str, language: str) -> str:
        """Transcribe using whisper_cpp library with GPU offloading."""
        import whisper_cpp
        import numpy as np
        from scipy.io import wavfile
        
        # Load and preprocess audio
        sample_rate, audio = wavfile.read(audio_path)
        audio = audio.astype(np.float32) / 32768.0
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            from scipy.signal import resample
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = resample(audio, num_samples)
        
        # Configure context parameters with GPU offloading
        ctx_params = whisper_cpp.whisper_context_default_params()
        ctx_params.use_gpu = self.use_gpu
        
        if self.use_gpu:
            if self.gpu_layers is not None:
                ctx_params.n_gpu_layers = self.gpu_layers
            else:
                # Full offloading by default
                ctx_params.n_gpu_layers = 999
        
        # Initialize context with parameters
        ctx = whisper_cpp.whisper_init_from_file_with_params(
            self._model_path.encode(),
            ctx_params
        )
        
        if ctx is None:
            raise RuntimeError("Failed to initialize whisper.cpp context")
        
        try:
            # Set parameters
            params = whisper_cpp.whisper_full_default_params(
                whisper_cpp.WHISPER_SAMPLING_GREEDY
            )
            params.language = language.encode('utf-8')
            params.n_threads = self.n_threads
            
            # Run transcription
            result = whisper_cpp.whisper_full(
                ctx,
                params,
                audio.ctypes.data_as(whisper_cpp.ctypes.POINTER(whisper_cpp.c_float)),
                len(audio),
            )
            
            if result != 0:
                raise RuntimeError(f"whisper_full failed with code {result}")
            
            # Extract text
            text_parts = []
            n_segments = whisper_cpp.whisper_full_n_segments(ctx)
            for i in range(n_segments):
                segment_text = whisper_cpp.whisper_full_get_segment_text(ctx, i)
                text_parts.append(segment_text.decode('utf-8'))
            
            text = " ".join(text_parts)
            
        finally:
            whisper_cpp.whisper_free(ctx)
        
        return text

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            import subprocess

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
            import os
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            return max(file_size_mb, 1.0)
