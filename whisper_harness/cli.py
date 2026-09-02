#!/usr/bin/env python3
"""CLI entry point for Whisper transcription harness.

Supports multiple ASR backends: faster-whisper, whisper.cpp, transcribe.cpp (GGUF), and HuggingFace.
Optimized for resource-constrained systems (GTX 1050 2GB, 4-8GB RAM).

Usage:
    python cli.py --audio test.wav --model-type fast_whisper --model-id medium --compute-type int8_float32
    python cli.py --audio test.wav --model-type whisper_cpp --model-id ggerganov/whisper.cpp --quantization q5_0
    python cli.py --audio test.wav --model-type transcribe_cpp --model-id handy-computer/gigaam-v3-e2e-rnnt-gguf --quantization Q5_K_M
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print("Warning: PyYAML not installed. Using default config.")
        return {}
    except FileNotFoundError:
        print(f"Warning: Config file not found: {config_path}")
        return {}


def create_transcriber(
    model_type: str,
    model_id: str,
    device: str,
    **kwargs: Any,
):
    """Factory function to create transcriber instances."""
    from transcribers import (
        FasterWhisperTranscriber,
        WhisperCppTranscriber,
        HuggingFaceWhisperTranscriber,
        TranscribeCppTranscriber,
    )
    from transcribers.sber import SberGigaAMTranscriber

    transcriber_map = {
        "fast_whisper": FasterWhisperTranscriber,
        "faster_whisper": FasterWhisperTranscriber,
        "whisper_cpp": WhisperCppTranscriber,
        "whisper.cpp": WhisperCppTranscriber,
        "transcribe_cpp": TranscribeCppTranscriber,
        "transcribe.cpp": TranscribeCppTranscriber,
        "hf_whisper": HuggingFaceWhisperTranscriber,
        "huggingface": HuggingFaceWhisperTranscriber,
    }

    transcriber_class = transcriber_map.get(model_type.lower())
    if transcriber_class is None:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {', '.join(transcriber_map.keys())}"
        )

    # Remove special handling for Sber model type since we removed it
    return transcriber_class(model_id=model_id, device=device, **kwargs)


def generate_reference_text(audio_path: str) -> str:
    """Generate reference text using TTS (for demonstration)."""
    # Simple placeholder - in real scenario would use gTTS or similar
    print("⚠️  Warning: --generate-reference creates placeholder text.")
    print("   For accurate WER, provide actual ground truth transcript.")
    return "Это тестовая транскрипция для демонстрации работы метрики WER."


def save_results(
    result: dict[str, Any],
    output_path: str,
    reference: str | None = None,
) -> None:
    """Save results to JSON file."""
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "audio_file": os.path.abspath(result.get("audio_path", "")),
        "reference_text": reference,
        "result": result,
    }

    # Add WER/CER if reference provided
    if reference:
        from utils.metrics import calculate_wer, calculate_cer

        output_data["wer"] = calculate_wer(reference, result.get("text", ""))
        output_data["cer"] = calculate_cer(reference, result.get("text", ""))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"💾 Results saved to: {output_path}")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Whisper ASR Transcription Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Faster-Whisper with int8 quantization (optimal for GTX 1050 2GB)
  python cli.py --audio test.wav --model-type fast_whisper --model-id medium --compute-type int8_float32

  # Whisper.cpp with q5_0 quantization
  python cli.py --audio test.wav --model-type whisper_cpp --quantization q5_0

  # Transcribe.cpp with GGUF quantized GigaAM model
  python cli.py --audio test.wav --model-type transcribe_cpp --model-id handy-computer/gigaam-v3-e2e-rnnt-gguf --quantization Q5_K_M

  # HuggingFace Whisper with reference for WER calculation
  python cli.py --audio test.wav --model-type hf_whisper --model-id sberbank-ai/whisper-small-ru --reference ground_truth.txt

  # Load config from YAML
  python cli.py --audio test.wav --config benchmark_config.yaml
        """,
    )

    # Required arguments
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file (WAV, MP3, etc.)",
    )

    # Model selection
    parser.add_argument(
        "--model-type",
        type=str,
        required="--config" not in sys.argv,
        choices=[
            "fast_whisper", "whisper_cpp",
            "transcribe_cpp",
            "hf_whisper",
        ],
        help="Transcriber backend to use",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model identifier (size name or HuggingFace repo)",
    )

    # Model-specific parameters
    parser.add_argument(
        "--quantization",
        type=str,
        default="q5_0",
        help="Quantization type for whisper.cpp (q4_0, q5_0, q8_0, f16, f32)",
    )

    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8_float32",
        help="Compute type for faster-whisper (int8_float16, int8_float32, float16, float32)",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Beam search size (default: 1 for speed)",
    )

    # Device and threads
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for inference",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Number of CPU threads (default: 2 for A4-5300)",
    )

    # Language and reference
    parser.add_argument(
        "--language",
        type=str,
        default="ru",
        help="Language code (default: ru)",
    )

    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to ground truth text file for WER/CER calculation",
    )

    parser.add_argument(
        "--generate-reference",
        action="store_true",
        help="Generate placeholder reference text (for demo)",
    )

    # Output and config
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    # Additional options
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Enable 8-bit quantization for HF models",
    )

    args = parser.parse_args()

    # Load config from YAML if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)

    # Validate audio file
    if not os.path.exists(args.audio):
        print(f"❌ Error: Audio file not found: {args.audio}", file=sys.stderr)
        return 1

    # Set default model_id based on model_type
    if args.model_id is None:
        defaults = {
            "fast_whisper": "medium",
            "whisper_cpp": "ggerganov/whisper.cpp",
            "transcribe_cpp": "handy-computer/gigaam-v3-e2e-rnnt-gguf",
            "hf_whisper": "openai/whisper-medium",
        }
        args.model_id = defaults.get(args.model_type, "medium")

    # Load reference text
    reference_text: str | None = None
    if args.reference:
        if not os.path.exists(args.reference):
            print(f"❌ Error: Reference file not found: {args.reference}", file=sys.stderr)
            return 1
        with open(args.reference, "r", encoding="utf-8") as f:
            reference_text = f.read().strip()
    elif args.generate_reference:
        reference_text = generate_reference_text(args.audio)

    # Prepare transcriber kwargs
    kwargs = {
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "quantization": args.quantization,
        "n_threads": args.threads,
        "load_in_8bit": args.load_in_8bit,
    }

    # Filter out None/default values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    print("=" * 60)
    print("🎙️  Whisper Transcription Harness")
    print("=" * 60)
    print(f"📁 Audio: {args.audio}")
    print(f"🤖 Model Type: {args.model_type}")
    print(f"🧠 Model ID: {args.model_id}")
    print(f"🌐 Language: {args.language}")
    print(f"💻 Device: {args.device}")
    if reference_text:
        print(f"📝 Reference: {args.reference or 'generated'}")
    print("=" * 60)

    try:
        # Create transcriber
        transcriber = create_transcriber(
            model_type=args.model_type,
            model_id=args.model_id,
            device=args.device,
            **kwargs,
        )

        print(f"\n⏳ Loading model...")
        
        # Run transcription
        result = transcriber.transcribe(args.audio, language=args.language)
        result["audio_path"] = args.audio

        # Print formatted results
        from utils.metrics import format_metrics

        print(format_metrics(result, reference=reference_text))

        # Save results if output path specified
        if args.output:
            save_results(result, args.output, reference=reference_text)

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"\n❌ Runtime Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
