#!/usr/bin/env python3
"""Unit tests for whisper_harness components."""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wavfile
from scipy.io import wavfile as scipy_wavfile

# Import modules under test
from whisper_harness.benchmark import (
    BenchmarkConfig,
    BenchmarkHarness,
    OffloadConfig,
    TestDataset as BenchmarkTestDataset,  # aliased to avoid pytest "Test*" collection conflict
    generate_test_configs,
    normalize_to_list,
    parse_offload_configs,
    SystemConfig,
)
from whisper_harness.utils.metrics import calculate_wer, calculate_cer, calculate_rtf, normalize_text
from whisper_harness.utils.memory_monitor import MemoryMonitor, get_system_info


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_audio_file():
    """Create a short test WAV file."""
    sample_rate = 16000
    duration = 0.5  # 0.5 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wavfile.write(fd.name, sample_rate, (audio * 32767).astype(np.int16))
    yield fd.name
    os.unlink(fd.name)


@pytest.fixture
def system_config():
    """Standard test system config."""
    return SystemConfig(
        cpu_model="Test CPU",
        cpu_cores=4,
        total_ram_gb=8.0,
        gpu_name="Test GPU",
        gpu_count=1,
        total_vram_gb=2.0,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_wer_perfect_match(self):
        ref = "привет мир это тест"
        hyp = "привет мир это тест"
        assert calculate_wer(ref, hyp) == 0.0

    def test_wer_one_word_different(self):
        ref = "привет мир это тест"
        hyp = "привет мир это другой"
        assert calculate_wer(ref, hyp) == 0.25

    def test_wer_empty_reference(self):
        ref = ""
        hyp = "привет мир"
        # All wrong = 1.0
        assert calculate_wer(ref, hyp) == 1.0

    def test_wer_empty_both(self):
        assert calculate_wer("", "") == 0.0

    def test_cer_perfect_match(self):
        ref = "привет мир"
        hyp = "привет мир"
        assert calculate_cer(ref, hyp) == 0.0

    def test_cer_one_char_different(self):
        ref = "привет"
        hyp = "привит"
        # 1 char wrong out of 6
        assert calculate_cer(ref, hyp) == pytest.approx(1 / 6, rel=0.01)

    def test_rtf_calculation(self):
        assert calculate_rtf(5.0, 10.0) == 0.5

    def test_rtf_faster_than_realtime(self):
        # 3 seconds to transcribe 10 seconds of audio = RTF 0.3
        assert calculate_rtf(3.0, 10.0) == 0.3

    def test_normalize_text(self):
        text = "Привет, Мир! Это  ТЕСТ."
        normalized = normalize_text(text)
        assert normalized == "привет мир это тест"

    def test_normalize_text_empty(self):
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""


# ---------------------------------------------------------------------------
# Memory Monitor
# ---------------------------------------------------------------------------

class TestMemoryMonitor:
    def test_monitor_basic(self):
        monitor = MemoryMonitor(sampling_interval_ms=50)
        monitor.start()
        time.sleep(0.15)
        monitor.stop()
        assert monitor.peak_ram_mb > 0
        assert isinstance(monitor.get_stats(), dict)

    def test_monitor_stats(self):
        monitor = MemoryMonitor(sampling_interval_ms=50)
        monitor.start()
        time.sleep(0.1)
        monitor.stop()
        stats = monitor.get_stats()
        assert "peak_ram_mb" in stats
        assert "avg_ram_mb" in stats
        assert stats["peak_ram_mb"] > 0

    def test_system_info(self):
        info = get_system_info()
        assert "python_version" in info
        assert "cpu_model" in info
        assert "ram_total_gb" in info


# ---------------------------------------------------------------------------
# OffloadConfig
# ---------------------------------------------------------------------------

class TestOffloadConfig:
    def test_defaults(self):
        cfg = OffloadConfig()
        assert cfg.layer_count == 0
        assert cfg.split_mode == "none"
        assert cfg.block_count is None
        assert cfg.max_vram_gb is None

    def test_custom_values(self):
        cfg = OffloadConfig(
            layer_count=10,
            split_mode="layer",
            block_count=5,
            max_vram_gb=2.0,
        )
        assert cfg.layer_count == 10
        assert cfg.split_mode == "layer"
        assert cfg.block_count == 5
        assert cfg.max_vram_gb == 2.0


# ---------------------------------------------------------------------------
# TestDataset
# ---------------------------------------------------------------------------

class TestDatasetUnit:
    """Unit tests for benchmark.TestDataset."""

    def test_defaults(self):
        ds = BenchmarkTestDataset()
        assert ds.name == ""
        assert ds.audio_path == ""
        assert ds.reference_text is None
        assert ds.language == "ru"
        assert ds.languages is None


# ---------------------------------------------------------------------------
# BenchmarkConfig
# ---------------------------------------------------------------------------

class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.framework == "faster-whisper"
        assert cfg.model == "small"
        assert cfg.device == "cuda"
        assert cfg.beam_size == 1
        assert cfg.offload_config is None


# ---------------------------------------------------------------------------
# normalize_to_list
# ---------------------------------------------------------------------------

class TestNormalizeToList:
    def test_none(self):
        assert normalize_to_list(None) == []

    def test_single_string(self):
        assert normalize_to_list("small") == ["small"]

    def test_already_list(self):
        lst = ["small", "medium"]
        result = normalize_to_list(lst)
        assert result == ["small", "medium"]
        # Should return the same reference
        assert result is lst


# ---------------------------------------------------------------------------
# parse_offload_configs
# ---------------------------------------------------------------------------

class TestParseOffloadConfigs:
    def test_none(self):
        configs = parse_offload_configs(None)
        assert len(configs) == 1
        assert configs[0].split_mode == "none"

    def test_dict(self):
        configs = parse_offload_configs({"layer_count": 10, "split_mode": "layer"})
        assert len(configs) == 1
        assert configs[0].layer_count == 10
        assert configs[0].split_mode == "layer"

    def test_list_of_dicts(self):
        configs = parse_offload_configs([
            {"layer_count": 10, "split_mode": "layer"},
            {"layer_count": 0, "split_mode": "none"},
        ])
        assert len(configs) == 2
        assert configs[0].layer_count == 10
        assert configs[1].layer_count == 0

    def test_string(self):
        configs = parse_offload_configs("auto")
        assert len(configs) == 1
        assert configs[0].split_mode == "auto"


# ---------------------------------------------------------------------------
# generate_test_configs
# ---------------------------------------------------------------------------

class TestGenerateTestConfigs:
    def test_shortcut_plural(self, system_config):
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
        configs, datasets = generate_test_configs(yaml_config, system_config)
        assert len(configs) == 2
        assert all(c.framework == "faster-whisper" for c in configs)
        models = [c.model for c in configs]
        assert "small" in models
        assert "medium" in models

    def test_singular_form(self, system_config):
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
        configs, datasets = generate_test_configs(yaml_config, system_config)
        assert len(configs) == 1
        assert configs[0].model == "ggerganov/whisper.cpp"

    def test_auto_quantizations(self, system_config):
        yaml_config = {
            "benchmarks": [
                {
                    "framework": "faster-whisper",
                    "model": "small",
                    # No quantizations specified
                }
            ]
        }
        configs, datasets = generate_test_configs(yaml_config, system_config)
        assert len(configs) == 1
        assert configs[0].quantization == "int8_float32"

    def test_with_offload(self, system_config):
        yaml_config = {
            "benchmarks": [
                {
                    "framework": "transcribe.cpp",
                    "models": ["test-model"],
                    "devices": ["offload"],
                    "offload_configs": [
                        {"layer_count": 10, "split_mode": "layer"},
                        {"layer_count": 0, "split_mode": "none"},
                    ],
                }
            ]
        }
        configs, datasets = generate_test_configs(yaml_config, system_config)
        assert len(configs) == 2
        assert configs[0].offload_config is not None
        assert configs[0].offload_config.layer_count == 10
        assert configs[1].offload_config.layer_count == 0

    def test_test_datasets(self, system_config):
        yaml_config = {
            "test_datasets": [
                {
                    "name": "test_dataset",
                    "audio_path": "/path/to/audio.wav",
                    "reference_text": "/path/to/ground_truth.txt",
                    "language": "ru",
                }
            ],
            "benchmarks": [
                {
                    "framework": "faster-whisper",
                    "model": "small",
                }
            ],
        }
        configs, datasets = generate_test_configs(yaml_config, system_config)
        assert len(datasets) == 1
        assert datasets[0].name == "test_dataset"
        assert datasets[0].audio_path == "/path/to/audio.wav"

    def test_skip_cuda_when_no_gpu(self):
        system_config = SystemConfig(gpu_count=0)
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
        devices = [c.device for c in configs]
        assert "cuda" not in devices
        assert "cpu" in devices

    def test_minimal_config(self, system_config):
        yaml_config = {
            "benchmarks": [
                {
                    "framework": "faster-whisper",
                    # Only framework specified, everything else defaults
                }
            ]
        }
        configs, datasets = generate_test_configs(yaml_config, system_config)
        assert len(configs) == 1
        assert configs[0].model == "small"
        assert configs[0].quantization == "int8_float32"


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

class TestCLI:
    def test_help(self):
        import subprocess
        repo_root = Path(__file__).parent.parent
        env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
        result = subprocess.run(
            [sys.executable, str(repo_root / "src" / "whisper_harness" / "cli.py"), "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0

    def test_missing_audio_error(self):
        import subprocess
        repo_root = Path(__file__).parent.parent
        env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "src" / "whisper_harness" / "cli.py"),
                "--audio", "nonexistent.wav",
                "--model-type", "fast_whisper",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 1


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_load_nonexistent(self):
        from whisper_harness.cli import load_config
        config = load_config("nonexistent.yaml")
        assert config == {}

    def test_load_valid_yaml(self, tmp_path):
        from whisper_harness.cli import load_config
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "benchmarks:\n  - framework: faster-whisper\n    model: small\n"
        )
        config = load_config(str(yaml_file))
        assert "benchmarks" in config
        assert len(config["benchmarks"]) == 1
