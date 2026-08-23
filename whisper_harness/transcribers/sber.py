"""Sber GigaAM transcriber implementation (CTC and RNNT)."""

import os
import time
from typing import Any

from transcribers.base import Transcriber


class SberGigaAMTranscriber(Transcriber):
    """Transcriber for Sberbank GigaAM models.
    
    Supports both CTC and RNNT architectures.
    Prefers ONNX runtime for memory efficiency on constrained systems.
    Falls back to PyTorch/transformers if ONNX not available.
    
    Attributes:
        model_type: Architecture type ('ctc' or 'rnnt').
        use_onnx: Prefer ONNX runtime for inference.
    """

    # Model mappings
    MODELS = {
        "ctc": "sberbank-ai/gigaam-v2-ctc",
        "rnnt": "sberbank-ai/gigaam-v2-rnnt",
    }

    def __init__(
        self,
        model_id: str = "ctc",
        device: str = "cpu",
        use_onnx: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize Sber GigaAM transcriber.
        
        Args:
            model_id: Model type ('ctc', 'rnnt') or HuggingFace repo id.
            device: Device ('cpu' or 'cuda').
            use_onnx: Prefer ONNX runtime for memory efficiency.
            **kwargs: Additional parameters.
        """
        super().__init__(model_id, device, **kwargs)
        self.use_onnx = use_onnx
        self._processor: Any = None
        self._onnx_session: Any = None

    def _load_model_onnx(self) -> None:
        """Load model using ONNX runtime (memory efficient)."""
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            # Determine model repo
            if self.model_id in ["ctc", "rnnt"]:
                repo_id = self.MODELS[self.model_id]
            else:
                repo_id = self.model_id

            # Try to find ONNX model
            try:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="model.onnx",
                    cache_dir=os.path.expanduser("~/.cache/sber"),
                )
            except Exception:
                # Fallback to exported ONNX from community
                raise RuntimeError("ONNX model not available for this repo")

            # Create session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2  # A4-5300 has 2 cores

            providers = ["CPUExecutionProvider"]
            if self.device == "cuda":
                providers.insert(0, "CUDAExecutionProvider")

            self._onnx_session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers,
            )
            self._model = self._onnx_session

        except ImportError:
            raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e

    def _load_model_transformers(self) -> None:
        """Load model using transformers library."""
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            # Determine model repo
            if self.model_id in ["ctc", "rnnt"]:
                repo_id = self.MODELS[self.model_id]
            else:
                repo_id = self.model_id

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                repo_id,
                cache_dir=os.path.expanduser("~/.cache/sber"),
            )

            # Load model
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                repo_id,
                cache_dir=os.path.expanduser("~/.cache/sber"),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )

            if self.device == "cuda":
                self._model.to("cuda")
                torch.cuda.empty_cache()

        except ImportError:
            raise RuntimeError("transformers not installed. Run: pip install transformers")
        except Exception as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                raise RuntimeError(
                    f"OOM: Sber model doesn't fit in memory. Try ONNX mode or smaller model."
                ) from e
            raise RuntimeError(f"Failed to load Sber model: {e}") from e

    def _load_model(self) -> None:
        """Load Sber GigaAM model (ONNX preferred, fallback to transformers)."""
        start_time = time.perf_counter()

        if self.use_onnx:
            try:
                self._load_model_onnx()
                self._load_time = time.perf_counter() - start_time
                return
            except Exception as e:
                print(f"ONNX loading failed: {e}. Falling back to transformers...")

        # Fallback to transformers
        self._load_model_transformers()
        self._load_time = time.perf_counter() - start_time

    def _transcribe_onnx(self, audio_path: str) -> dict[str, Any]:
        """Transcribe using ONNX runtime."""
        import numpy as np
        from scipy.io import wavfile

        # Load audio
        sample_rate, audio = wavfile.read(audio_path)
        audio = audio.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

        # Resample if needed (GigaAM expects 16kHz)
        if sample_rate != 16000:
            from scipy.signal import resample
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = resample(audio, num_samples)

        # Prepare input
        inputs = self._processor(
            audio,
            sampling_rate=16000,
            return_tensors="np",
            padding=True,
        )

        # Run inference
        start_time = time.perf_counter()
        outputs = self._onnx_session.run(None, {"input_values": inputs["input_values"]})
        transcribe_time = time.perf_counter() - start_time

        # Decode output
        predicted_ids = np.argmax(outputs[0], axis=-1)
        text = self._processor.batch_decode(predicted_ids)[0]

        return {
            "text": text.strip(),
            "transcribe_time": transcribe_time,
        }

    def _transcribe_transformers(self, audio_path: str) -> dict[str, Any]:
        """Transcribe using transformers."""
        import torch
        from scipy.io import wavfile

        # Load audio
        sample_rate, audio = wavfile.read(audio_path)
        audio = audio.astype(np.float32) / 32768.0

        # Resample if needed
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

        # Run inference
        start_time = time.perf_counter()
        with torch.no_grad():
            predicted_ids = self._model.generate(**inputs)
        transcribe_time = time.perf_counter() - start_time

        # Decode output
        text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return {
            "text": text.strip(),
            "transcribe_time": transcribe_time,
        }

    def transcribe(
        self, audio_path: str, language: str = "ru"
    ) -> dict[str, Any]:
        """Transcribe audio using Sber GigaAM.
        
        Args:
            audio_path: Path to audio file.
            language: Language code (ignored, GigaAM is Russian-only).
            
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

        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Import memory monitor
        from utils.memory_monitor import MemoryMonitor

        monitor = MemoryMonitor()
        monitor.start()

        # Choose transcription method
        if self._onnx_session is not None:
            result = self._transcribe_onnx(audio_path)
        else:
            result = self._transcribe_transformers(audio_path)

        monitor.stop()

        # Get audio duration
        duration = self._get_audio_duration(audio_path)

        return {
            "text": result["text"],
            "duration": duration,
            "transcribe_time": result["transcribe_time"],
            "load_time": self._load_time,
            "model_name": f"sber-gigaam-{self.model_id}",
            "memory_peak_mb": monitor.peak_ram_mb,
            "vram_peak_mb": monitor.peak_vram_mb if monitor.peak_vram_mb else None,
            "framework": "sber-gigaam",
            "device": self.device,
            "use_onnx": self._onnx_session is not None,
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
            import os
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            return max(file_size_mb, 1.0)
