#!/usr/bin/env python3
"""
Test script to verify the benchmark harness structure without requiring actual models.
This validates the code logic, data structures, and output formats.
"""

import sys
import os
sys.path.insert(0, '/workspace')

from whisper_benchmark import (
    SystemConfig, MemorySnapshot, BenchmarkResult, BenchmarkConfig,
    MemoryMonitor, calculate_wer, calculate_cer, get_system_config
)


def test_data_structures():
    """Test that all data structures work correctly"""
    print("Testing data structures...")
    
    # Test SystemConfig
    config = SystemConfig()
    assert hasattr(config, 'cpu_model')
    assert hasattr(config, 'total_ram_gb')
    assert hasattr(config, 'gpu_name')
    print("  ✓ SystemConfig works")
    
    # Test MemorySnapshot
    snapshot = MemorySnapshot(timestamp=1.0, ram_mb=500.0, vram_mb=800.0)
    assert snapshot.timestamp == 1.0
    assert snapshot.ram_mb == 500.0
    assert snapshot.vram_mb == 800.0
    print("  ✓ MemorySnapshot works")
    
    # Test BenchmarkResult
    result = BenchmarkResult(
        framework='faster-whisper',
        model='medium',
        quantization='int8_float32',
        device='cuda',
        beam_size=5,
        status='PASS'
    )
    assert result.framework == 'faster-whisper'
    assert result.model == 'medium'
    assert result.status == 'PASS'
    print("  ✓ BenchmarkResult works")
    
    # Test BenchmarkConfig
    config = BenchmarkConfig(
        framework='faster-whisper',
        model='small',
        quantization='int8',
        device='cuda',
        beam_size=3,
        language='ru'
    )
    assert config.framework == 'faster-whisper'
    assert config.language == 'ru'
    print("  ✓ BenchmarkConfig works")


def test_wer_cer():
    """Test WER and CER calculations"""
    print("\nTesting WER/CER calculations...")
    
    # Perfect match
    wer = calculate_wer("hello world", "hello world")
    assert wer == 0.0, f"Expected 0.0, got {wer}"
    print("  ✓ WER perfect match: 0%")
    
    # Complete mismatch
    wer = calculate_wer("hello world", "goodbye universe")
    assert wer > 0, f"Expected > 0, got {wer}"
    print(f"  ✓ WER complete mismatch: {wer:.1f}%")
    
    # Partial match
    wer = calculate_wer("the quick brown fox", "the quick red fox")
    assert 0 < wer < 100, f"Expected 0-100, got {wer}"
    print(f"  ✓ WER partial match: {wer:.1f}%")
    
    # CER tests
    cer = calculate_cer("hello", "hello")
    assert cer == 0.0, f"Expected 0.0, got {cer}"
    print(f"  ✓ CER perfect match: {cer}%")
    
    cer = calculate_cer("hello", "hallo")
    assert cer == 20.0, f"Expected 20.0, got {cer}"  # 1 char change out of 5
    print(f"  ✓ CER one char change: {cer}%")


def test_system_config():
    """Test system configuration detection"""
    print("\nTesting system configuration detection...")
    
    config = get_system_config()
    
    print(f"  CPU Model: {config.cpu_model}")
    print(f"  CPU Cores: {config.cpu_cores}")
    print(f"  Total RAM: {config.total_ram_gb:.1f} GB")
    print(f"  GPU Name: {config.gpu_name or 'Not detected'}")
    print(f"  GPU Count: {config.gpu_count}")
    print(f"  Total VRAM: {config.total_vram_gb:.1f} GB")
    print(f"  CUDA Version: {config.cuda_version}")
    print(f"  Python Version: {config.python_version}")
    print(f"  Platform: {config.platform}")
    
    assert config.cpu_cores > 0, "CPU cores should be detected"
    assert config.total_ram_gb > 0, "RAM should be detected"
    assert config.python_version != "", "Python version should be detected"
    print("  ✓ System config detection works")


def test_memory_monitor():
    """Test memory monitoring functionality"""
    print("\nTesting memory monitor...")
    
    monitor = MemoryMonitor(interval_ms=50)
    
    # Start monitoring
    monitor.start()
    import time
    time.sleep(0.2)  # Let it collect some snapshots
    
    # Get readings
    latest_ram, latest_vram = monitor.get_latest_memory()
    peak_ram, peak_vram = monitor.get_peak_memory()
    
    print(f"  Latest RAM: {latest_ram:.0f} MB")
    print(f"  Latest VRAM: {latest_vram:.0f} MB")
    print(f"  Peak RAM: {peak_ram:.0f} MB")
    print(f"  Peak VRAM: {peak_vram:.0f} MB")
    
    # Stop monitoring
    monitor.stop()
    
    # Should have collected snapshots
    assert len(monitor.snapshots) > 0, "Should have collected snapshots"
    print(f"  ✓ Collected {len(monitor.snapshots)} snapshots")
    print("  ✓ Memory monitor works")


def test_benchmark_result_serialization():
    """Test that benchmark results can be serialized"""
    print("\nTesting result serialization...")
    
    from dataclasses import asdict
    import json
    
    result = BenchmarkResult(
        framework='faster-whisper',
        model='medium',
        quantization='int8_float32',
        device='cuda',
        beam_size=5,
        load_time_sec=2.5,
        transcribe_time_sec=15.3,
        rtf=0.085,
        total_time_sec=17.8,
        ram_before_mb=450.0,
        ram_after_mb=1200.0,
        ram_peak_mb=1350.0,
        vram_before_mb=500.0,
        vram_after_mb=850.0,
        vram_peak_mb=920.0,
        wer=5.2,
        cer=2.1,
        word_count=150,
        char_count=800,
        status='PASS',
        error_message='',
        transcription='Test transcription',
        audio_duration_sec=180.0
    )
    
    # Test asdict
    result_dict = asdict(result)
    assert 'framework' in result_dict
    assert 'vram_peak_mb' in result_dict
    print("  ✓ asdict() works")
    
    # Test JSON serialization
    json_str = json.dumps(result_dict, ensure_ascii=False)
    assert len(json_str) > 0
    print("  ✓ JSON serialization works")
    
    # Test deserialization
    loaded = json.loads(json_str)
    assert loaded['framework'] == 'faster-whisper'
    assert loaded['vram_peak_mb'] == 920.0
    print("  ✓ JSON deserialization works")


def main():
    print("="*60)
    print("Whisper Benchmark Harness - Unit Tests")
    print("="*60)
    
    try:
        test_data_structures()
        test_wer_cer()
        test_system_config()
        test_memory_monitor()
        test_benchmark_result_serialization()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
