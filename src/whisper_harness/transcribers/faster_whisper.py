"""Faster-Whisper transcriber implementation."""

import time
from typing import Any

from .base import Transcriber


class FasterWhisperTranscriber(Transcriber):
    """Transcriber using faster-whisper library.
    
    Optimized for GPU inference with support for various quantization types.
    Automatically clears CUDA cache on initialization to free VRAM.
    
    Attributes:
        compute_type: Quantization type (int8_float16, int8_float32, float16, float32).
        beam_size: Beam search size for decoding.
        gpu_id: GPU device ID for multi-GPU systems.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        compute_type: str = "int8_float32",
        beam_size: int = 1,
        gpu_id: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize Faster-Whisper transcriber.
        
        Args:
            model_id: Model size ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3').
            device: Device ('cuda' or 'cpu').
            compute_type: Quantization type for inference.
            beam_size: Number of beams for beam search.
            gpu_id: GPU device ID.
            **kwargs: Additional parameters passed to WhisperModel.
        """
        super().__init__(model_id, device, **kwargs)
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.gpu_id = gpu_id
        self._model_instance: Any = None

    def _load_model(self) -> None:
        """Load Faster-Whisper model with specified quantization."""
        start_time = time.perf_counter()

        try:
            from faster_whisper import WhisperModel

            # Clear CUDA cache before loading to free VRAM
            if self.device == "cuda":
                from utils.memory_monitor import clear_gpu_cache
                clear_gpu_cache()

            # Map model_id to faster-whisper format
            model_path = self.model_id
            if self.model_id in ["tiny", "base", "small", "medium", "large-v2", "large-v3"]:
                model_path = self.model_id

            self._model_instance = WhisperModel(
                model_size_or_path=model_path,
                device=self.device,
                device_index=self.gpu_id,
                compute_type=self.compute_type,
            )
            self._model = self._model_instance  # Set base class attribute

        except ImportError as e:
            raise RuntimeError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            ) from e
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                raise RuntimeError(
                    f"OOM: Model {self.model_id} ({self.compute_type}) doesn't fit in VRAM. "
                    f"Try smaller model or different quantization."
                ) from e
            raise RuntimeError(f"Failed to load model: {e}") from e

        self._load_time = time.perf_counter() - start_time

    def transcribe(
        self, audio_path: str, language: str = "ru"
    ) -> dict[str, Any]:
        """Transcribe audio using Faster-Whisper.
        
        Args:
            audio_path: Path to audio file.
            language: Language code ('ru' for Russian).
            
        Returns:
            Dictionary with transcription results and metrics.
            
        Raises:
            FileNotFoundError: If audio file doesn't exist.
            RuntimeError: If transcription fails.
        """
        import os

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Lazy loading
        self._ensure_loaded()

        if self._model_instance is None:
            raise RuntimeError("Model not loaded")

        # Import memory monitor
        from utils.memory_monitor import MemoryMonitor

        monitor = MemoryMonitor()
        monitor.start()

        start_time = time.perf_counter()

        try:
            segments, info = self._model_instance.transcribe(
                audio_path,
                language=language,
                beam_size=self.beam_size,
                vad_filter=False,  # Disable VAD to prevent OOM on memory-constrained systems
            )

            text = " ".join(segment.text for segment in segments)
            transcribe_time = time.perf_counter() - start_time

        except Exception as e:
            monitor.stop()
            raise RuntimeError(f"Transcription failed: {e}") from e

        monitor.stop()

        return {
            "text": text.strip(),
            "duration": info.duration,
            "transcribe_time": transcribe_time,
            "load_time": self._load_time,
            "model_name": f"faster-whisper-{self.model_id}-{self.compute_type}",
            "memory_peak_mb": monitor.peak_ram_mb,
            "vram_peak_mb": monitor.peak_vram_mb if monitor.peak_vram_mb else None,
            "framework": "faster-whisper",
            "device": self.device,
            "beam_size": self.beam_size,
            "compute_type": self.compute_type,
        }
