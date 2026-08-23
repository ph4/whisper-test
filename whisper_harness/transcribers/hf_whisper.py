"""Hugging Face Whisper transcriber implementation."""

import os
import time
from typing import Any

from transcribers.base import Transcriber


class HuggingFaceWhisperTranscriber(Transcriber):
    """Transcriber using Hugging Face transformers for Whisper models.
    
    Supports fine-tuned Russian models from HuggingFace Hub.
    Uses explicit model.generate() instead of pipeline for better memory control.
    
    Attributes:
        torch_dtype: Data type for model weights.
        load_in_8bit: Enable 8-bit quantization (requires bitsandbytes).
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-medium",
        device: str = "cuda",
        torch_dtype: str = "float16",
        load_in_8bit: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize Hugging Face Whisper transcriber.
        
        Args:
            model_id: HuggingFace model repo id.
            device: Device ('cuda' or 'cpu').
            torch_dtype: Data type ('float16', 'float32', 'bfloat16').
            load_in_8bit: Enable 8-bit quantization.
            **kwargs: Additional parameters.
        """
        super().__init__(model_id, device, **kwargs)
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit
        self._processor: Any = None

    def _load_model(self) -> None:
        """Load Whisper model from HuggingFace."""
        start_time = time.perf_counter()

        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            # Map dtype string to torch dtype
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            torch_dtype = dtype_map.get(self.torch_dtype, torch.float32)

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=os.path.expanduser("~/.cache/huggingface"),
            )

            # Prepare model kwargs
            model_kwargs = {
                "cache_dir": os.path.expanduser("~/.cache/huggingface"),
                "low_cpu_mem_usage": True,
            }

            if self.device == "cuda":
                model_kwargs["torch_dtype"] = torch_dtype
                if self.load_in_8bit:
                    try:
                        model_kwargs["load_in_8bit"] = True
                        model_kwargs["device_map"] = "auto"
                    except Exception:
                        print("bitsandbytes not available, using standard loading")

            # Load model
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                **model_kwargs,
            )

            if self.device == "cuda" and not self.load_in_8bit:
                self._model.to("cuda")
                from utils.memory_monitor import clear_gpu_cache
                clear_gpu_cache()

            self._load_time = time.perf_counter() - start_time

        except ImportError as e:
            raise RuntimeError(
                "transformers not installed. Run: pip install transformers"
            ) from e
        except Exception as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                raise RuntimeError(
                    f"OOM: Model {self.model_id} doesn't fit in memory. "
                    f"Try smaller model or enable 8-bit quantization."
                ) from e
            raise RuntimeError(f"Failed to load HF Whisper model: {e}") from e

    def transcribe(
        self, audio_path: str, language: str = "ru"
    ) -> dict[str, Any]:
        """Transcribe audio using Hugging Face Whisper.
        
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

        if self._model is None or self._processor is None:
            raise RuntimeError("Model or processor not loaded")

        # Import memory monitor
        from utils.memory_monitor import MemoryMonitor

        monitor = MemoryMonitor()
        monitor.start()

        try:
            import torch
            from scipy.io import wavfile

            # Load audio
            sample_rate, audio = wavfile.read(audio_path)
            audio = audio.astype(np.float32) / 32768.0

            # Resample if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                from scipy.signal import resample
                num_samples = int(len(audio) * 16000 / sample_rate)
                audio = resample(audio, num_samples)

            # Prepare input
            inputs = self._processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )

            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Get language and task tokens
            lang_token = "<|ru|>" if language == "ru" else "<|en|>"
            task_token = "<|transcribe|>"

            # Run inference
            start_time = time.perf_counter()
            with torch.no_grad():
                predicted_ids = self._model.generate(
                    **inputs,
                    language=language,
                    task="transcribe",
                    max_length=448,
                )
            transcribe_time = time.perf_counter() - start_time

            # Decode output
            text = self._processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

        except ImportError as e:
            monitor.stop()
            raise RuntimeError(f"Missing dependency: {e}") from e
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
            "model_name": f"hf-whisper-{self.model_id}",
            "memory_peak_mb": monitor.peak_ram_mb,
            "vram_peak_mb": monitor.peak_vram_mb if monitor.peak_vram_mb else None,
            "framework": "huggingface-transformers",
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "load_in_8bit": self.load_in_8bit,
        }

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
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            return max(file_size_mb, 1.0)


# Fix missing numpy import in transcribe method
import numpy as np  # noqa: E402
