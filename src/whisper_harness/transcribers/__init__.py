"""Transcribers package for ASR models."""

from .base import Transcriber  # noqa: E402  (requires parent on path when not installed)
from .faster_whisper import FasterWhisperTranscriber  # noqa: E402
from .whisper_cpp import WhisperCppTranscriber  # noqa: E402
from .hf_whisper import HuggingFaceWhisperTranscriber  # noqa: E402
from .transcribe_cpp import TranscribeCppTranscriber  # noqa: E402
from .sber import SberGigaAMTranscriber  # noqa: E402

__all__ = [
    "Transcriber",
    "FasterWhisperTranscriber",
    "WhisperCppTranscriber",
    "HuggingFaceWhisperTranscriber",
    "TranscribeCppTranscriber",
    "SberGigaAMTranscriber",
]


def run_self_test(quick: bool = False, framework: str = "all") -> None:
    """Run self-test for all transcribers.
    
    Args:
        quick: If True, only check imports without loading models.
        framework: Specific framework to test or "all".
    """
    from .self_test import main as self_test_main
    import sys as _sys

    args = ["self_test"]
    if quick:
        args.append("--quick")
    if framework != "all":
        args.extend(["--framework", framework])

    _sys.argv = args
    self_test_main()
