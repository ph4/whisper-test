"""Pytest configuration: add src/ to path so tests can import whisper_harness."""

import sys
from pathlib import Path

# Ensure whisper_harness is importable when running pytest from repo root
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
