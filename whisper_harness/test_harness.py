#!/usr/bin/env python3
"""Unit tests for whisper_harness components."""

import os
import sys
import tempfile
import unittest
from pathlib import Path


class TestMetrics(unittest.TestCase):
    """Tests for utils.metrics module."""

    def test_calculate_wer_perfect_match(self):
        """WER should be 0.0 for identical texts."""
        from utils.metrics import calculate_wer
        ref = "привет мир это тест"
        hyp = "привет мир это тест"
        self.assertEqual(calculate_wer(ref, hyp), 0.0)

    def test_calculate_wer_one_word_different(self):
        """WER should reflect word differences."""
        from utils.metrics import calculate_wer
        ref = "привет мир это тест"
        hyp = "привет мир это другой"
        self.assertEqual(calculate_wer(ref, hyp), 0.25)  # 1/4 words wrong

    def test_calculate_wer_empty_reference(self):
        """WER should handle empty reference."""
        from utils.metrics import calculate_wer
        ref = ""
        hyp = "привет мир"
        # Empty reference with non-empty hypothesis = 1.0 (all wrong)
        self.assertEqual(calculate_wer(ref, hyp), 1.0)

    def test_calculate_cer_perfect_match(self):
        """CER should be 0.0 for identical texts."""
        from utils.metrics import calculate_cer
        ref = "привет мир"
        hyp = "привет мир"
        self.assertEqual(calculate_cer(ref, hyp), 0.0)

    def test_calculate_rtf(self):
        """RTF calculation."""
        from utils.metrics import calculate_rtf
        rtf = calculate_rtf(5.0, 10.0)
        self.assertEqual(rtf, 0.5)

    def test_normalize_text(self):
        """Text normalization."""
        from utils.metrics import normalize_text
        text = "Привет, Мир! Это  ТЕСТ."
        normalized = normalize_text(text)
        self.assertEqual(normalized, "привет мир это тест")


class TestMemoryMonitor(unittest.TestCase):
    """Tests for utils.memory_monitor module."""

    def test_memory_monitor_basic(self):
        """Basic memory monitor functionality."""
        from utils.memory_monitor import MemoryMonitor
        monitor = MemoryMonitor(sampling_interval_ms=50)
        monitor.start()
        import time
        time.sleep(0.15)
        monitor.stop()
        
        self.assertGreater(monitor.peak_ram_mb, 0)
        self.assertIsInstance(monitor.get_stats(), dict)

    def test_memory_monitor_context_manager(self):
        """Memory monitor as context manager."""
        from utils.memory_monitor import MemoryMonitor
        with MemoryMonitor() as monitor:
            import time
            time.sleep(0.1)
        
        self.assertGreater(monitor.peak_ram_mb, 0)

    def test_get_system_info(self):
        """System info retrieval."""
        from utils.memory_monitor import get_system_info
        info = get_system_info()
        
        self.assertIn("python_version", info)
        self.assertIn("cpu_model", info)
        self.assertIn("ram_total_gb", info)


class TestTranscriberBase(unittest.TestCase):
    """Tests for transcribers.base module."""

    def test_base_class_abstract(self):
        """Transcriber is abstract and cannot be instantiated."""
        from transcribers.base import Transcriber
        
        with self.assertRaises(TypeError):
            Transcriber(model_id="test")

    def test_concrete_implementation_required(self):
        """Concrete implementation must override methods."""
        from transcribers.base import Transcriber
        
        class TestTranscriber(Transcriber):
            def _load_model(self) -> None:
                self._model = "loaded"
            
            def transcribe(self, audio_path: str, language: str = "ru") -> dict:
                # Call _ensure_loaded to trigger lazy loading
                self._ensure_loaded()
                return {"text": "test", "duration": 1.0, "transcribe_time": 0.1}
        
        transcriber = TestTranscriber(model_id="test")
        self.assertFalse(transcriber.is_loaded)
        
        # After calling transcribe, model should be loaded (lazy loading)
        result = transcriber.transcribe("dummy.wav")
        self.assertTrue(transcriber.is_loaded)
        self.assertEqual(result["text"], "test")


class TestCLI(unittest.TestCase):
    """Tests for cli.py module."""

    def test_cli_help(self):
        """CLI should show help without errors."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Whisper ASR Transcription Harness", result.stdout)

    def test_cli_missing_audio(self):
        """CLI should error on missing audio file."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "cli.py", "--audio", "nonexistent.wav", "--model-type", "fast_whisper"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("not found", result.stderr.lower())


class TestConfigLoading(unittest.TestCase):
    """Tests for YAML config loading."""

    def test_config_load_nonexistent(self):
        """Config loading should handle missing files gracefully."""
        from cli import load_config
        config = load_config("nonexistent.yaml")
        self.assertEqual(config, {})


def create_test_audio(duration_sec: float = 1.0, sample_rate: int = 16000) -> str:
    """Create a temporary WAV file for testing."""
    import numpy as np
    from scipy.io import wavfile
    
    duration = duration_sec
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate simple sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio = (audio * 32767).astype(np.int16)
    
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()
    wavfile.write(temp_file.name, sample_rate, audio)
    
    return temp_file.name


if __name__ == "__main__":
    unittest.main(verbosity=2)
