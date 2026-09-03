"""Base transcriber interface for ASR models."""

from abc import ABC, abstractmethod
from typing import Any


class Transcriber(ABC):
    """Abstract base class for all ASR transcribers.
    
    Implements lazy loading pattern to minimize memory usage on resource-constrained systems.
    All concrete implementations must override _load_model() and transcribe() methods.
    
    Attributes:
        model_id: Identifier for the model (size name or HuggingFace repo id).
        device: Device to run inference on ('cpu' or 'cuda').
        _model: Loaded model instance (None until first transcribe call).
        _load_time: Time taken to load the model in seconds.
    """

    def __init__(self, model_id: str, device: str = "cpu", **kwargs: Any) -> None:
        """Initialize transcriber with model configuration.
        
        Args:
            model_id: Model identifier (e.g., 'medium', 'large-v3', 'sberbank-ai/whisper-small-ru').
            device: Target device for inference ('cpu' or 'cuda').
            **kwargs: Additional model-specific parameters.
        """
        self.model_id = model_id
        self.device = device
        self._model: Any = None  # Lazy loading!
        self._load_time: float = 0.0
        self._extra_kwargs: dict[str, Any] = kwargs

    @abstractmethod
    def _load_model(self) -> None:
        """Load model into memory.
        
        This method is called once during the first transcribe() call.
        Implementations should handle device placement and any necessary preprocessing.
        Must set self._model and self._load_time.
        
        Raises:
            RuntimeError: If model loading fails (OOM, network error, etc.).
        """
        pass

    @abstractmethod
    def transcribe(
        self, audio_path: str, language: str = "ru"
    ) -> dict[str, Any]:
        """Transcribe audio file to text.
        
        Implements lazy loading: loads model on first call, reuses for subsequent calls.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.).
            language: Language code for transcription (default: 'ru').
            
        Returns:
            Dictionary containing:
                - text: Transcribed text (str).
                - duration: Audio duration in seconds (float).
                - transcribe_time: Inference time in seconds (float).
                - load_time: Model load time in seconds (0 if already loaded) (float).
                - model_name: Name/identifier of the used model (str).
                - memory_peak_mb: Peak memory consumption in MB (float, optional).
                
        Raises:
            FileNotFoundError: If audio file doesn't exist.
            RuntimeError: If transcription fails.
        """
        pass

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded (lazy loading helper)."""
        if self._model is None:
            self._load_model()

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self._model is not None

    @property
    def load_time(self) -> float:
        """Get model load time in seconds."""
        return self._load_time
