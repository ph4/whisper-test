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


class TestBenchmarkConfig(unittest.TestCase):
    """Tests for benchmark.py configuration and parsing."""

    def test_offload_config_default(self):
        """OffloadConfig should have sensible defaults."""
        from benchmark import OffloadConfig
        config = OffloadConfig()
        self.assertEqual(config.layer_count, 0)
        self.assertEqual(config.split_mode, "none")
        self.assertIsNone(config.block_count)
        self.assertIsNone(config.max_vram_gb)

    def test_offload_config_custom(self):
        """OffloadConfig should accept custom values."""
        from benchmark import OffloadConfig
        config = OffloadConfig(
            layer_count=10,
            split_mode="layer",
            block_count=5,
            max_vram_gb=2.0
        )
        self.assertEqual(config.layer_count, 10)
        self.assertEqual(config.split_mode, "layer")
        self.assertEqual(config.block_count, 5)
        self.assertEqual(config.max_vram_gb, 2.0)

    def test_test_dataset_default(self):
        """TestDataset should have sensible defaults."""
        from benchmark import TestDataset
        dataset = TestDataset()
        self.assertEqual(dataset.name, "")
        self.assertEqual(dataset.audio_path, "")
        self.assertIsNone(dataset.reference_text)
        self.assertEqual(dataset.language, "ru")
        self.assertIsNone(dataset.languages)

    def test_benchmark_config_default(self):
        """BenchmarkConfig should have sensible defaults."""
        from benchmark import BenchmarkConfig
        config = BenchmarkConfig()
        self.assertEqual(config.framework, "faster-whisper")
        self.assertEqual(config.model, "small")
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.beam_size, 1)
        self.assertIsNone(config.offload_config)

    def test_normalize_to_list_none(self):
        """normalize_to_list should handle None."""
        from benchmark import normalize_to_list
        result = normalize_to_list(None)
        self.assertEqual(result, [])

    def test_normalize_to_list_single_value(self):
        """normalize_to_list should wrap single values in list."""
        from benchmark import normalize_to_list
        result = normalize_to_list("small")
        self.assertEqual(result, ["small"])

    def test_normalize_to_list_already_list(self):
        """normalize_to_list should return list unchanged."""
        from benchmark import normalize_to_list
        input_list = ["small", "medium"]
        result = normalize_to_list(input_list)
        self.assertEqual(result, input_list)
        self.assertIs(result, input_list)  # Returns same reference

    def test_parse_offload_configs_empty(self):
        """parse_offload_configs should return default config for empty input."""
        from benchmark import parse_offload_configs
        configs = parse_offload_configs(None)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].split_mode, "none")

    def test_parse_offload_configs_dict(self):
        """parse_offload_configs should handle dict input."""
        from benchmark import parse_offload_configs
        data = {"layer_count": 10, "split_mode": "layer"}
        configs = parse_offload_configs(data)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].layer_count, 10)
        self.assertEqual(configs[0].split_mode, "layer")

    def test_parse_offload_configs_list_of_dicts(self):
        """parse_offload_configs should handle list of dicts."""
        from benchmark import parse_offload_configs
        data = [
            {"layer_count": 10, "split_mode": "layer"},
            {"layer_count": 0, "split_mode": "none"}
        ]
        configs = parse_offload_configs(data)
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0].layer_count, 10)
        self.assertEqual(configs[1].layer_count, 0)

    def test_parse_offload_configs_string(self):
        """parse_offload_configs should handle string input."""
        from benchmark import parse_offload_configs
        data = "auto"
        configs = parse_offload_configs(data)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].split_mode, "auto")


class TestYAMLConfigGeneration(unittest.TestCase):
    """Tests for YAML config generation from parsed data."""

    def setUp(self):
        """Set up test fixtures."""
        from benchmark import SystemConfig
        self.system_config = SystemConfig(
            cpu_model="Test CPU",
            cpu_cores=4,
            total_ram_gb=8.0,
            gpu_name="Test GPU",
            gpu_count=1,
            total_vram_gb=2.0,
        )

    def test_generate_configs_shortcut_format(self):
        """Should generate configs from shortcut YAML format."""
        from benchmark import generate_test_configs
        yaml_config = {
            "benchmarks": [
                {
                    "framework": "faster-whisper",
                    "models": ["small", "medium"],
                    "quantizations": ["int8_float32"],
                    "devices": ["cuda"],
                }
            ]
        }
        configs, datasets = generate_test_configs(yaml_config, self.system_config)
        # Should generate 2 configs (small + medium)
        self.assertEqual(len(configs), 2)
        frameworks = [c.framework for c in configs]
        self.assertTrue(all(f == "faster-whisper" for f in frameworks))

    def test_generate_configs_singular_form(self):
        """Should handle singular form (model instead of models)."""
        from benchmark import generate_test_configs
        yaml_config = {
            "benchmarks": [
                {
                    "framework": "whisper.cpp",
                    "model": "ggerganov/whisper.cpp",
                    "quantization": ["q5_0"],
                    "devices": ["cpu"],
                }
            ]
        }
        configs, datasets = generate_test_configs(yaml_config, self.system_config)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].model, "ggerganov/whisper.cpp")

    def test_generate_configs_auto_quantizations(self):
        """Should auto-detect quantizations when not specified."""
        from benchmark import generate_test_configs
        yaml_config = {
            "benchmarks": [
                {
                    "framework": "faster-whisper",
                    "model": "small",
                    # No quantizations specified
                }
            ]
        }
        configs, datasets = generate_test_configs(yaml_config, self.system_config)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].quantization, "int8_float32")

    def test_generate_configs_with_offload(self):
        """Should handle offload configurations."""
        from benchmark import generate_test_configs
        yaml_config = {
            "benchmarks": [
                {
                    "framework": "transcribe.cpp",
                    "models": ["test-model"],
                    "devices": ["offload"],
                    "offload_configs": [
                        {"layer_count": 10, "split_mode": "layer"},
                        {"layer_count": 0, "split_mode": "none"}
                    ]
                }
            ]
        }
        configs, datasets = generate_test_configs(yaml_config, self.system_config)
        # Should generate 2 configs (one per offload_config)
        self.assertEqual(len(configs), 2)
        self.assertIsNotNone(configs[0].offload_config)
        self.assertEqual(configs[0].offload_config.layer_count, 10)

    def test_generate_configs_test_datasets(self):
        """Should parse test datasets from YAML."""
        from benchmark import generate_test_configs
        yaml_config = {
            "test_datasets": [
                {
                    "name": "test_dataset",
                    "audio_path": "/path/to/audio.wav",
                    "reference_text": "/path/to/ground_truth.txt",
                    "language": "ru"
                }
            ],
            "benchmarks": [
                {
                    "framework": "faster-whisper",
                    "model": "small"
                }
            ]
        }
        configs, datasets = generate_test_configs(yaml_config, self.system_config)
        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0].name, "test_dataset")
        self.assertEqual(datasets[0].audio_path, "/path/to/audio.wav")

    def test_generate_configs_skip_cuda_no_gpu(self):
        """Should skip CUDA configs when no GPU available."""
        from benchmark import generate_test_configs, SystemConfig
        system_config = SystemConfig(
            gpu_count=0,  # No GPU
        )
        yaml_config = {
            "benchmarks": [
                {
                    "framework": "faster-whisper",
                    "model": "small",
                    "devices": ["cuda", "cpu"],
                }
            ]
        }
        configs, datasets = generate_test_configs(yaml_config, system_config)
        # Should only have CPU config
        devices = [c.device for c in configs]
        self.assertNotIn("cuda", devices)
        self.assertIn("cpu", devices)


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
