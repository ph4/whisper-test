"""Transcribers package for ASR models."""

from transcribers.base import Transcriber
from transcribers.faster_whisper import FasterWhisperTranscriber
from transcribers.whisper_cpp import WhisperCppTranscriber
from transcribers.hf_whisper import HuggingFaceWhisperTranscriber
from transcribers.transcribe_cpp import TranscribeCppTranscriber

__all__ = [
    "Transcriber",
    "FasterWhisperTranscriber",
    "WhisperCppTranscriber",
    "HuggingFaceWhisperTranscriber",
    "TranscribeCppTranscriber",
]


def run_self_test(quick: bool = False, framework: str = "all") -> None:
    """Run self-test for all transcribers.
    
    Args:
        quick: If True, only check imports without loading models.
        framework: Specific framework to test or "all".
    """
    from transcribers.self_test import main as self_test_main
    import sys
    
    args = ["self_test"]
    if quick:
        args.append("--quick")
    if framework != "all":
        args.extend(["--framework", framework])
    
    sys.argv = args
    self_test_main()
