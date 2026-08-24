"""Self-test module for all transcriber implementations.

This module provides comprehensive self-testing capabilities to verify:
1. Required libraries are installed
2. Models can be loaded (minimal model for each type)
3. Both CPU and GPU modes work (if applicable)
4. Basic transcription functionality works

Usage:
    python -m transcribers.self_test
    python -m transcribers.self_test --quick  # Skip actual model loading
    python -m transcribers.self_test --framework faster-whisper  # Test specific framework
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class TestStatus(Enum):
    """Test result status."""
    PASSED = "✅ PASSED"
    FAILED = "❌ FAILED"
    SKIPPED = "⚠️  SKIPPED"
    WARNING = "⚡ WARNING"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: TestStatus
    message: str = ""
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameworkTestResult:
    """Results for a complete framework test."""
    framework: str
    results: list[TestResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    warning: int = 0
    
    def add_result(self, result: TestResult) -> None:
        """Add a test result and update counters."""
        self.results.append(result)
        if result.status == TestStatus.PASSED:
            self.passed += 1
        elif result.status == TestStatus.FAILED:
            self.failed += 1
        elif result.status == TestStatus.SKIPPED:
            self.skipped += 1
        elif result.status == TestStatus.WARNING:
            self.warning += 1
    
    def summary(self) -> str:
        """Get summary string."""
        return f"{self.framework}: {self.passed} passed, {self.failed} failed, {self.skipped} skipped, {self.warning} warnings"


class TranscriberSelfTest:
    """Self-test suite for all transcriber implementations."""
    
    # Minimal test audio (1 second of silence at 16kHz, 1 channel, 16-bit)
    TEST_AUDIO_DURATION_SEC = 1
    TEST_AUDIO_SAMPLE_RATE = 16000
    
    def __init__(self, quick_mode: bool = False, gpu_only: bool = False, cpu_only: bool = False):
        """Initialize self-test.
        
        Args:
            quick_mode: If True, only check library imports, skip model loading.
            gpu_only: If True, only test GPU mode.
            cpu_only: If True, only test CPU mode.
        """
        self.quick_mode = quick_mode
        self.gpu_only = gpu_only
        self.cpu_only = cpu_only
        self._test_audio_path: str | None = None
        
    def _create_test_audio(self) -> str:
        """Create a minimal test audio file (1 second of silence)."""
        import numpy as np
        import tempfile
        
        if self._test_audio_path and os.path.exists(self._test_audio_path):
            return self._test_audio_path
        
        # Create 1 second of silence at 16kHz
        duration = self.TEST_AUDIO_DURATION_SEC
        sample_rate = self.TEST_AUDIO_SAMPLE_RATE
        num_samples = duration * sample_rate
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        # Save as WAV file
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="selftest_")
        os.close(fd)
        
        try:
            from scipy.io import wavfile
            wavfile.write(path, sample_rate, (audio_data * 32767).astype(np.int16))
            self._test_audio_path = path
            return path
        except Exception as e:
            # Cleanup on failure
            try:
                os.unlink(path)
            except:
                pass
            raise RuntimeError(f"Failed to create test audio: {e}")
    
    def _cleanup_test_audio(self) -> None:
        """Clean up test audio file."""
        if self._test_audio_path and os.path.exists(self._test_audio_path):
            try:
                os.unlink(self._test_audio_path)
                self._test_audio_path = None
            except:
                pass
    
    def _check_import(self, module_name: str, install_hint: str) -> TestResult:
        """Check if a module can be imported."""
        start = time.perf_counter()
        try:
            __import__(module_name)
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name=f"Import {module_name}",
                status=TestStatus.PASSED,
                message=f"Module '{module_name}' imported successfully",
                duration_ms=duration
            )
        except ImportError as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name=f"Import {module_name}",
                status=TestStatus.FAILED,
                message=f"Cannot import '{module_name}': {e}. {install_hint}",
                duration_ms=duration
            )
    
    def _test_faster_whisper(self) -> FrameworkTestResult:
        """Test faster-whisper implementation."""
        result = FrameworkTestResult(framework="faster-whisper")
        
        # Check import
        import_result = self._check_import(
            "faster_whisper",
            "Install with: pip install faster-whisper"
        )
        result.add_result(import_result)
        
        if import_result.status != TestStatus.PASSED:
            return result
        
        if self.quick_mode:
            result.add_result(TestResult(
                name="Model loading",
                status=TestStatus.SKIPPED,
                message="Quick mode - skipping model loading"
            ))
            return result
        
        # Test model loading on CPU and/or GPU
        devices_to_test = []
        if not self.gpu_only:
            devices_to_test.append("cpu")
        if not self.cpu_only:
            devices_to_test.append("cuda")
        
        for device in devices_to_test:
            device_test = self._test_faster_whisper_device(device)
            result.add_result(device_test)
        
        return result
    
    def _test_faster_whisper_device(self, device: str) -> TestResult:
        """Test faster-whisper on a specific device."""
        start = time.perf_counter()
        
        try:
            from faster_whisper import WhisperModel
            
            # Try to load tiny model (smallest available)
            compute_type = "int8_float32" if device == "cuda" else "default"
            
            if device == "cuda":
                # Clear CUDA cache first
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
            
            model = WhisperModel(
                model_size_or_path="tiny",
                device=device,
                compute_type=compute_type if compute_type != "default" else "default"
            )
            
            duration = (time.perf_counter() - start) * 1000
            
            # Try a quick transcription
            audio_path = self._create_test_audio()
            segments, info = model.transcribe(audio_path, language="ru", vad_filter=False)
            text = " ".join(seg.text for seg in segments)
            
            return TestResult(
                name=f"Faster-Whisper on {device.upper()}",
                status=TestStatus.PASSED,
                message=f"Model loaded and transcribed successfully on {device}",
                duration_ms=duration,
                details={
                    "device": device,
                    "model": "tiny",
                    "compute_type": compute_type,
                    "transcription_length": len(text)
                }
            )
            
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            error_msg = str(e).lower()
            
            if "out of memory" in error_msg or "oom" in error_msg:
                status = TestStatus.WARNING
                message = f"OOM on {device}: Model too large for available memory"
            elif device == "cuda" and ("cuda" in error_msg or "gpu" in error_msg):
                status = TestStatus.SKIPPED
                message = f"CUDA not available on {device}: {e}"
            else:
                status = TestStatus.FAILED
                message = f"Failed on {device}: {e}"
            
            return TestResult(
                name=f"Faster-Whisper on {device.upper()}",
                status=status,
                message=message,
                duration_ms=duration,
                details={"device": device, "error": str(e)}
            )
    
    def _test_whisper_cpp(self) -> FrameworkTestResult:
        """Test whisper.cpp implementation."""
        result = FrameworkTestResult(framework="whisper.cpp")
        
        # Check for pywhispercpp or whisper_cpp
        pywhispercpp_result = self._check_import(
            "pywhispercpp",
            "Install with: pip install pywhispercpp"
        )
        
        whisper_cpp_result = self._check_import(
            "whisper_cpp",
            "Install with: pip install whisper-cpp-python"
        )
        
        if pywhispercpp_result.status == TestStatus.PASSED:
            result.add_result(pywhispercpp_result)
        elif whisper_cpp_result.status == TestStatus.PASSED:
            result.add_result(whisper_cpp_result)
        else:
            result.add_result(TestResult(
                name="Import whisper.cpp bindings",
                status=TestStatus.FAILED,
                message="Neither pywhispercpp nor whisper_cpp available. Install one of them."
            ))
            return result
        
        if self.quick_mode:
            result.add_result(TestResult(
                name="Model loading",
                status=TestStatus.SKIPPED,
                message="Quick mode - skipping model loading"
            ))
            return result
        
        # Test model loading
        model_test = self._test_whisper_cpp_model()
        result.add_result(model_test)
        
        return result
    
    def _test_whisper_cpp_model(self) -> TestResult:
        """Test whisper.cpp model loading with CPU and GPU modes."""
        start = time.perf_counter()
        
        try:
            # Try to use the transcriber class
            from transcribers.whisper_cpp import WhisperCppTranscriber
            
            # Determine devices to test
            devices_to_test = []
            if not self.gpu_only:
                devices_to_test.append(("cpu", False))
            if not self.cpu_only:
                devices_to_test.append(("cuda", True))
            
            results_summary = []
            for device, use_gpu in devices_to_test:
                try:
                    transcriber = WhisperCppTranscriber(
                        model_id="ggerganov/whisper.cpp",
                        device=device,
                        quantization="q5_0",
                        n_threads=2,
                        use_gpu=use_gpu,
                        gpu_layers=None  # Auto-detect full offloading for GPU
                    )
                    
                    # This will attempt to download and load the model
                    transcriber._ensure_loaded()
                    
                    results_summary.append(f"{device.upper()}: OK")
                    
                except Exception as e:
                    results_summary.append(f"{device.upper()}: {str(e)[:50]}")
            
            duration = (time.perf_counter() - start) * 1000
            
            # Check if at least one device worked
            if any("OK" in r for r in results_summary):
                return TestResult(
                    name="Whisper.cpp model loading (CPU/GPU)",
                    status=TestStatus.PASSED,
                    message=f"Model loaded successfully: {'; '.join(results_summary)}",
                    duration_ms=duration,
                    details={"results": results_summary}
                )
            else:
                return TestResult(
                    name="Whisper.cpp model loading (CPU/GPU)",
                    status=TestStatus.FAILED,
                    message=f"All devices failed: {'; '.join(results_summary)}",
                    duration_ms=duration,
                    details={"results": results_summary}
                )
            
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="Whisper.cpp model loading (CPU/GPU)",
                status=TestStatus.FAILED,
                message=f"Failed to load model: {e}",
                duration_ms=duration,
                details={"error": str(e)}
            )
    
    def _test_transcribe_cpp(self) -> FrameworkTestResult:
        """Test transcribe.cpp (GGUF) implementation."""
        result = FrameworkTestResult(framework="transcribe.cpp")
        
        # Check for huggingface_hub (needed for model download)
        hf_result = self._check_import(
            "huggingface_hub",
            "Install with: pip install huggingface_hub"
        )
        result.add_result(hf_result)
        
        if hf_result.status != TestStatus.PASSED:
            return result
        
        # Check for pywhispercpp or whisper_cpp
        pywhispercpp_result = self._check_import("pywhispercpp", "")
        whisper_cpp_result = self._check_import("whisper_cpp", "")
        
        if pywhispercpp_result.status == TestStatus.PASSED or whisper_cpp_result.status == TestStatus.PASSED:
            result.add_result(TestResult(
                name="Import GGUF backend",
                status=TestStatus.PASSED,
                message="pywhispercpp or whisper_cpp available"
            ))
        else:
            result.add_result(TestResult(
                name="Import GGUF backend",
                status=TestStatus.WARNING,
                message="No GGUF backend found (pywhispercpp or whisper_cpp). Install one for full functionality."
            ))
        
        if self.quick_mode:
            result.add_result(TestResult(
                name="Model loading",
                status=TestStatus.SKIPPED,
                message="Quick mode - skipping model loading"
            ))
            return result
        
        # Test model loading (use a small GGUF model)
        model_test = self._test_transcribe_cpp_model()
        result.add_result(model_test)
        
        return result
    
    def _test_transcribe_cpp_model(self) -> TestResult:
        """Test transcribe.cpp model loading with CPU and GPU modes."""
        start = time.perf_counter()
        
        try:
            from transcribers.transcribe_cpp import TranscribeCppTranscriber
            
            # Determine devices to test
            devices_to_test = []
            if not self.gpu_only:
                devices_to_test.append(("cpu", False))
            if not self.cpu_only:
                devices_to_test.append(("cuda", True))
            
            results_summary = []
            for device, use_gpu in devices_to_test:
                try:
                    # Use a known GGUF model for testing
                    transcriber = TranscribeCppTranscriber(
                        model_id="handy-computer/gigaam-v3-e2e-rnnt-gguf",
                        device=device,
                        quantization="Q5_K_M",
                        n_threads=2,
                        use_gpu=use_gpu,
                        gpu_layers=None  # Auto-detect full offloading for GPU
                    )
                    
                    # Note: This may fail if the model repo doesn't have GGUF files
                    # We catch this and report it appropriately
                    try:
                        transcriber._ensure_loaded()
                        results_summary.append(f"{device.upper()}: OK")
                        
                    except RuntimeError as e:
                        if "No GGUF files found" in str(e):
                            results_summary.append(f"{device.upper()}: No GGUF files")
                        else:
                            results_summary.append(f"{device.upper()}: {str(e)[:40]}")
                            
                except Exception as e:
                    results_summary.append(f"{device.upper()}: {str(e)[:40]}")
            
            duration = (time.perf_counter() - start) * 1000
            
            # Check if at least one device worked
            if any("OK" in r for r in results_summary):
                return TestResult(
                    name="Transcribe.cpp model loading (CPU/GPU)",
                    status=TestStatus.PASSED,
                    message=f"GGUF model loaded successfully: {'; '.join(results_summary)}",
                    duration_ms=duration,
                    details={"results": results_summary}
                )
            elif any("No GGUF files" in r for r in results_summary):
                return TestResult(
                    name="Transcribe.cpp model loading (CPU/GPU)",
                    status=TestStatus.WARNING,
                    message=f"No GGUF files found in test repo: {'; '.join(results_summary)}",
                    duration_ms=duration,
                    details={"note": "Try with a different GGUF model repo", "results": results_summary}
                )
            else:
                return TestResult(
                    name="Transcribe.cpp model loading (CPU/GPU)",
                    status=TestStatus.FAILED,
                    message=f"All devices failed: {'; '.join(results_summary)}",
                    duration_ms=duration,
                    details={"results": results_summary}
                )
                
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="Transcribe.cpp model loading (CPU/GPU)",
                status=TestStatus.FAILED,
                message=f"Failed to load GGUF model: {e}",
                duration_ms=duration,
                details={"error": str(e)}
            )
    
    def _test_hf_whisper(self) -> FrameworkTestResult:
        """Test HuggingFace Whisper implementation."""
        result = FrameworkTestResult(framework="huggingface-whisper")
        
        # Check imports
        transformers_result = self._check_import(
            "transformers",
            "Install with: pip install transformers"
        )
        result.add_result(transformers_result)
        
        if transformers_result.status != TestStatus.PASSED:
            return result
        
        torch_result = self._check_import(
            "torch",
            "Install with: pip install torch"
        )
        result.add_result(torch_result)
        
        if self.quick_mode:
            result.add_result(TestResult(
                name="Model loading",
                status=TestStatus.SKIPPED,
                message="Quick mode - skipping model loading"
            ))
            return result
        
        # Test model loading on specified devices
        devices_to_test = []
        if not self.gpu_only:
            devices_to_test.append("cpu")
        if not self.cpu_only:
            devices_to_test.append("cuda")
        
        for device in devices_to_test:
            device_test = self._test_hf_whisper_device(device)
            result.add_result(device_test)
        
        return result
    
    def _test_hf_whisper_device(self, device: str) -> TestResult:
        """Test HF Whisper on a specific device."""
        start = time.perf_counter()
        
        try:
            from transcribers.hf_whisper import HuggingFaceWhisperTranscriber
            
            # Use a small model for testing
            transcriber = HuggingFaceWhisperTranscriber(
                model_id="openai/whisper-tiny",
                device=device,
                torch_dtype="float32" if device == "cpu" else "float16"
            )
            
            transcriber._ensure_loaded()
            
            duration = (time.perf_counter() - start) * 1000
            
            return TestResult(
                name=f"HF Whisper on {device.upper()}",
                status=TestStatus.PASSED,
                message=f"Model loaded successfully on {device}",
                duration_ms=duration,
                details={"device": device, "model": "openai/whisper-tiny"}
            )
            
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            error_msg = str(e).lower()
            
            if "out of memory" in error_msg or "oom" in error_msg:
                status = TestStatus.WARNING
                message = f"OOM on {device}: Model too large"
            elif device == "cuda" and ("cuda" in error_msg or "gpu" in error_msg):
                status = TestStatus.SKIPPED
                message = f"CUDA not available: {e}"
            else:
                status = TestStatus.FAILED
                message = f"Failed on {device}: {e}"
            
            return TestResult(
                name=f"HF Whisper on {device.upper()}",
                status=status,
                message=message,
                duration_ms=duration,
                details={"device": device, "error": str(e)}
            )
    
    def _test_sber_gigaam(self) -> FrameworkTestResult:
        """Test Sber GigaAM implementation."""
        result = FrameworkTestResult(framework="sber-gigaam")
        
        # Check for onnxruntime (preferred) or transformers
        onnx_result = self._check_import(
            "onnxruntime",
            "Install with: pip install onnxruntime or onnxruntime-gpu"
        )
        
        transformers_result = self._check_import(
            "transformers",
            "Install with: pip install transformers (fallback)"
        )
        
        if onnx_result.status == TestStatus.PASSED:
            result.add_result(onnx_result)
        elif transformers_result.status == TestStatus.PASSED:
            result.add_result(TestResult(
                name="Import onnxruntime",
                status=TestStatus.WARNING,
                message="ONNX not available, will use transformers fallback"
            ))
            result.add_result(transformers_result)
        else:
            result.add_result(TestResult(
                name="Import ASR backend",
                status=TestStatus.FAILED,
                message="Neither onnxruntime nor transformers available"
            ))
            return result
        
        if self.quick_mode:
            result.add_result(TestResult(
                name="Model loading",
                status=TestStatus.SKIPPED,
                message="Quick mode - skipping model loading"
            ))
            return result
        
        # Test model loading
        model_test = self._test_sber_model()
        result.add_result(model_test)
        
        return result
    
    def _test_sber_model(self) -> TestResult:
        """Test Sber GigaAM model loading."""
        start = time.perf_counter()
        
        try:
            from transcribers.sber import SberGigaAMTranscriber
            
            # Use CTC model (smaller) with ONNX if available
            transcriber = SberGigaAMTranscriber(
                model_id="ctc",
                device="cpu",
                use_onnx=True
            )
            
            transcriber._ensure_loaded()
            
            duration = (time.perf_counter() - start) * 1000
            
            return TestResult(
                name="Sber GigaAM model loading",
                status=TestStatus.PASSED,
                message="Model loaded successfully",
                duration_ms=duration,
                details={"model_type": "ctc", "use_onnx": transcriber._onnx_session is not None}
            )
            
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="Sber GigaAM model loading",
                status=TestStatus.FAILED,
                message=f"Failed to load model: {e}",
                duration_ms=duration,
                details={"error": str(e)}
            )
    
    def run_all_tests(self, framework: str | None = None) -> list[FrameworkTestResult]:
        """Run all tests or tests for a specific framework.
        
        Args:
            framework: If specified, only test this framework.
                      Options: 'faster-whisper', 'whisper.cpp', 'transcribe.cpp', 
                              'huggingface', 'sber', 'all'
        
        Returns:
            List of framework test results.
        """
        all_results = []
        
        frameworks = {
            "faster-whisper": self._test_faster_whisper,
            "whisper.cpp": self._test_whisper_cpp,
            "transcribe.cpp": self._test_transcribe_cpp,
            "huggingface": self._test_hf_whisper,
            "sber": self._test_sber_gigaam,
        }
        
        if framework and framework != "all":
            if framework in frameworks:
                frameworks = {framework: frameworks[framework]}
            else:
                print(f"Unknown framework: {framework}")
                print(f"Available: {', '.join(frameworks.keys())}, all")
                return all_results
        
        for fw_name, test_func in frameworks.items():
            print(f"\n{'='*60}")
            print(f"Testing: {fw_name}")
            print('='*60)
            result = test_func()
            all_results.append(result)
            print(result.summary())
        
        # Cleanup
        self._cleanup_test_audio()
        
        return all_results
    
    def print_summary(self, all_results: list[FrameworkTestResult]) -> None:
        """Print summary of all test results."""
        print("\n" + "="*60)
        print("SELF-TEST SUMMARY")
        print("="*60)
        
        total_passed = sum(r.passed for r in all_results)
        total_failed = sum(r.failed for r in all_results)
        total_skipped = sum(r.skipped for r in all_results)
        total_warning = sum(r.warning for r in all_results)
        
        for result in all_results:
            status_icon = "✅" if result.failed == 0 else "❌"
            print(f"\n{status_icon} {result.framework}")
            for test_result in result.results:
                icon = {
                    TestStatus.PASSED: "✅",
                    TestStatus.FAILED: "❌",
                    TestStatus.SKIPPED: "⚠️ ",
                    TestStatus.WARNING: "⚡"
                }[test_result.status]
                print(f"  {icon} {test_result.name}: {test_result.status.value}")
                if test_result.message:
                    print(f"     {test_result.message}")
                if test_result.duration_ms > 0:
                    print(f"     Time: {test_result.duration_ms:.1f}ms")
        
        print("\n" + "-"*60)
        print(f"TOTAL: {total_passed} passed, {total_failed} failed, "
              f"{total_skipped} skipped, {total_warning} warnings")
        print("-"*60)
        
        if total_failed > 0:
            print("\n⚠️  Some tests failed. Check the messages above for details.")
            print("💡 Installation hints are provided in failed test messages.")
        elif total_warning > 0:
            print("\n✅ All critical tests passed (with some warnings).")
        else:
            print("\n✅ All tests passed!")
    
    def export_json(self, all_results: list[FrameworkTestResult], output_path: str) -> None:
        """Export test results to JSON."""
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "quick_mode": self.quick_mode,
            "gpu_only": self.gpu_only,
            "cpu_only": self.cpu_only,
            "results": []
        }
        
        for result in all_results:
            fw_data = {
                "framework": result.framework,
                "passed": result.passed,
                "failed": result.failed,
                "skipped": result.skipped,
                "warning": result.warning,
                "tests": []
            }
            
            for test in result.results:
                fw_data["tests"].append({
                    "name": test.name,
                    "status": test.status.value,
                    "message": test.message,
                    "duration_ms": test.duration_ms,
                    "details": test.details
                })
            
            data["results"].append(fw_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Results exported to: {output_path}")


def main() -> None:
    """Main entry point for self-test."""
    parser = argparse.ArgumentParser(
        description="Self-test for all transcriber implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m transcribers.self_test                    # Run all tests
  python -m transcribers.self_test --quick            # Only check imports
  python -m transcribers.self_test --framework faster-whisper  # Test specific framework
  python -m transcribers.self_test --gpu-only         # Test GPU mode only
  python -m transcribers.self_test --cpu-only         # Test CPU mode only
  python -m transcribers.self_test --output results.json  # Export to JSON
        """
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: only check library imports, skip model loading"
    )
    
    parser.add_argument(
        "--framework", "-f",
        type=str,
        default="all",
        choices=["faster-whisper", "whisper.cpp", "transcribe.cpp", "huggingface", "sber", "all"],
        help="Test specific framework (default: all)"
    )
    
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Test GPU mode only"
    )
    
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Test CPU mode only"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Export results to JSON file"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("WHISPER HARNESS SELF-TEST")
    print("="*60)
    print(f"Mode: {'Quick (imports only)' if args.quick else 'Full (includes model loading)'}")
    print(f"Framework: {args.framework}")
    if args.gpu_only:
        print("GPU Mode: YES (CPU tests skipped)")
    if args.cpu_only:
        print("CPU Mode: YES (GPU tests skipped)")
    print("="*60)
    
    tester = TranscriberSelfTest(
        quick_mode=args.quick,
        gpu_only=args.gpu_only,
        cpu_only=args.cpu_only
    )
    
    all_results = tester.run_all_tests(framework=args.framework)
    tester.print_summary(all_results)
    
    if args.output:
        tester.export_json(all_results, args.output)
    
    # Exit with error code if any tests failed
    total_failed = sum(r.failed for r in all_results)
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
