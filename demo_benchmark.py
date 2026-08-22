#!/usr/bin/env python3
"""
Demo script that shows the benchmark harness output format
without requiring actual Whisper models to be installed.
This creates sample results to demonstrate the output formats.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from dataclasses import asdict


def create_demo_results(output_dir: str = "demo_results"):
    """Create demonstration benchmark results"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Sample system config (simulating target system)
    system_config = {
        "cpu_model": "AMD A4-5300 APU with Radeon HD Graphics",
        "cpu_cores": 2,
        "total_ram_gb": 6.0,
        "gpu_name": "NVIDIA GeForce GTX 1050",
        "gpu_count": 1,
        "total_vram_gb": 2.0,
        "cuda_version": "release 12.0, V12.0.82",
        "python_version": "3.10.12",
        "platform": "Linux-5.15.0-x86_64-with-glibc2.35"
    }
    
    # Sample benchmark results (simulated for typical GTX 1050 2GB system)
    demo_results = [
        # faster-whisper GPU results
        {
            "framework": "faster-whisper", "model": "tiny", "quantization": "int8_float32",
            "device": "cuda", "beam_size": 1, "load_time_sec": 1.2, "transcribe_time_sec": 8.5,
            "rtf": 0.028, "total_time_sec": 9.7, "ram_before_mb": 350, "ram_after_mb": 420,
            "ram_peak_mb": 480, "vram_before_mb": 450, "vram_after_mb": 520, "vram_peak_mb": 580,
            "wer": 8.5, "cer": 3.2, "word_count": 450, "char_count": 2100, "status": "PASS", "error_message": ""
        },
        {
            "framework": "faster-whisper", "model": "tiny", "quantization": "int8_float32",
            "device": "cuda", "beam_size": 3, "load_time_sec": 1.2, "transcribe_time_sec": 12.3,
            "rtf": 0.041, "total_time_sec": 13.5, "ram_before_mb": 350, "ram_after_mb": 420,
            "ram_peak_mb": 490, "vram_before_mb": 450, "vram_after_mb": 530, "vram_peak_mb": 600,
            "wer": 7.2, "cer": 2.8, "word_count": 455, "char_count": 2120, "status": "PASS", "error_message": ""
        },
        {
            "framework": "faster-whisper", "model": "base", "quantization": "int8_float32",
            "device": "cuda", "beam_size": 1, "load_time_sec": 1.5, "transcribe_time_sec": 12.5,
            "rtf": 0.042, "total_time_sec": 14.0, "ram_before_mb": 380, "ram_after_mb": 520,
            "ram_peak_mb": 620, "vram_before_mb": 450, "vram_after_mb": 620, "vram_peak_mb": 680,
            "wer": 6.8, "cer": 2.5, "word_count": 460, "char_count": 2150, "status": "PASS", "error_message": ""
        },
        {
            "framework": "faster-whisper", "model": "small", "quantization": "int8_float32",
            "device": "cuda", "beam_size": 1, "load_time_sec": 2.1, "transcribe_time_sec": 22.5,
            "rtf": 0.075, "total_time_sec": 24.6, "ram_before_mb": 420, "ram_after_mb": 720,
            "ram_peak_mb": 850, "vram_before_mb": 450, "vram_after_mb": 780, "vram_peak_mb": 850,
            "wer": 5.2, "cer": 1.8, "word_count": 465, "char_count": 2180, "status": "PASS", "error_message": ""
        },
        {
            "framework": "faster-whisper", "model": "small", "quantization": "int8_float32",
            "device": "cuda", "beam_size": 3, "load_time_sec": 2.1, "transcribe_time_sec": 35.0,
            "rtf": 0.117, "total_time_sec": 37.1, "ram_before_mb": 420, "ram_after_mb": 720,
            "ram_peak_mb": 880, "vram_before_mb": 450, "vram_after_mb": 800, "vram_peak_mb": 870,
            "wer": 4.5, "cer": 1.5, "word_count": 468, "char_count": 2190, "status": "PASS", "error_message": ""
        },
        {
            "framework": "faster-whisper", "model": "medium", "quantization": "int8_float32",
            "device": "cuda", "beam_size": 1, "load_time_sec": 3.5, "transcribe_time_sec": 45.0,
            "rtf": 0.150, "total_time_sec": 48.5, "ram_before_mb": 500, "ram_after_mb": 950,
            "ram_peak_mb": 1150, "vram_before_mb": 450, "vram_after_mb": 820, "vram_peak_mb": 890,
            "wer": 4.2, "cer": 1.4, "word_count": 470, "char_count": 2200, "status": "PASS", "error_message": ""
        },
        {
            "framework": "faster-whisper", "model": "medium", "quantization": "int8_float32",
            "device": "cuda", "beam_size": 3, "load_time_sec": 3.5, "transcribe_time_sec": 68.0,
            "rtf": 0.227, "total_time_sec": 71.5, "ram_before_mb": 500, "ram_after_mb": 950,
            "ram_peak_mb": 1200, "vram_before_mb": 450, "vram_after_mb": 850, "vram_peak_mb": 920,
            "wer": 3.8, "cer": 1.2, "word_count": 472, "char_count": 2210, "status": "PASS", "error_message": ""
        },
        {
            "framework": "faster-whisper", "model": "medium", "quantization": "int8_float32",
            "device": "cuda", "beam_size": 5, "load_time_sec": 3.5, "transcribe_time_sec": 95.0,
            "rtf": 0.317, "total_time_sec": 98.5, "ram_before_mb": 500, "ram_after_mb": 950,
            "ram_peak_mb": 1250, "vram_before_mb": 450, "vram_after_mb": 870, "vram_peak_mb": 950,
            "wer": 3.5, "cer": 1.1, "word_count": 475, "char_count": 2220, "status": "PASS", "error_message": ""
        },
        # OOM case for large model
        {
            "framework": "faster-whisper", "model": "large-v3", "quantization": "int8_float32",
            "device": "cuda", "beam_size": 1, "load_time_sec": 5.2, "transcribe_time_sec": 0,
            "rtf": 0, "total_time_sec": 5.2, "ram_before_mb": 550, "ram_after_mb": 1200,
            "ram_peak_mb": 1400, "vram_before_mb": 450, "vram_after_mb": 1850, "vram_peak_mb": 1980,
            "wer": None, "cer": None, "word_count": 0, "char_count": 0, 
            "status": "OOM", "error_message": "CUDA out of memory. Tried to allocate 250MB but only 150MB available"
        },
        # CPU results
        {
            "framework": "faster-whisper", "model": "small", "quantization": "int8_float32",
            "device": "cpu", "beam_size": 1, "load_time_sec": 2.5, "transcribe_time_sec": 125.0,
            "rtf": 0.417, "total_time_sec": 127.5, "ram_before_mb": 350, "ram_after_mb": 850,
            "ram_peak_mb": 1050, "vram_before_mb": 0, "vram_after_mb": 0, "vram_peak_mb": 0,
            "wer": 5.5, "cer": 2.0, "word_count": 462, "char_count": 2170, "status": "PASS", "error_message": ""
        },
        {
            "framework": "faster-whisper", "model": "medium", "quantization": "int8_float32",
            "device": "cpu", "beam_size": 1, "load_time_sec": 3.8, "transcribe_time_sec": 245.0,
            "rtf": 0.817, "total_time_sec": 248.8, "ram_before_mb": 420, "ram_after_mb": 1350,
            "ram_peak_mb": 1650, "vram_before_mb": 0, "vram_after_mb": 0, "vram_peak_mb": 0,
            "wer": 4.3, "cer": 1.5, "word_count": 468, "char_count": 2195, "status": "PASS", "error_message": ""
        },
        # whisper.cpp results (simulated)
        {
            "framework": "whisper.cpp", "model": "small", "quantization": "q8_0",
            "device": "cuda", "beam_size": 1, "load_time_sec": 0.8, "transcribe_time_sec": 28.0,
            "rtf": 0.093, "total_time_sec": 28.8, "ram_before_mb": 280, "ram_after_mb": 520,
            "ram_peak_mb": 620, "vram_before_mb": 450, "vram_after_mb": 680, "vram_peak_mb": 750,
            "wer": 5.8, "cer": 2.1, "word_count": 458, "char_count": 2150, "status": "PASS", "error_message": ""
        },
        {
            "framework": "whisper.cpp", "model": "medium", "quantization": "q8_0",
            "device": "cuda", "beam_size": 1, "load_time_sec": 1.2, "transcribe_time_sec": 52.0,
            "rtf": 0.173, "total_time_sec": 53.2, "ram_before_mb": 320, "ram_after_mb": 780,
            "ram_peak_mb": 920, "vram_before_mb": 450, "vram_after_mb": 850, "vram_peak_mb": 920,
            "wer": 4.5, "cer": 1.6, "word_count": 465, "char_count": 2180, "status": "PASS", "error_message": ""
        },
        {
            "framework": "whisper.cpp", "model": "medium", "quantization": "q5_0",
            "device": "cuda", "beam_size": 1, "load_time_sec": 1.1, "transcribe_time_sec": 48.0,
            "rtf": 0.160, "total_time_sec": 49.1, "ram_before_mb": 310, "ram_after_mb": 680,
            "ram_peak_mb": 820, "vram_before_mb": 450, "vram_after_mb": 720, "vram_peak_mb": 780,
            "wer": 4.8, "cer": 1.8, "word_count": 462, "char_count": 2170, "status": "PASS", "error_message": ""
        },
        # Russian-specific model
        {
            "framework": "faster-whisper", "model": "bond005/whisper_large_v2_ru", "quantization": "int8_float32",
            "device": "cuda", "beam_size": 1, "load_time_sec": 3.8, "transcribe_time_sec": 48.0,
            "rtf": 0.160, "total_time_sec": 51.8, "ram_before_mb": 520, "ram_after_mb": 980,
            "ram_peak_mb": 1180, "vram_before_mb": 450, "vram_after_mb": 840, "vram_peak_mb": 910,
            "wer": 3.2, "cer": 1.0, "word_count": 478, "char_count": 2230, "status": "PASS", "error_message": ""
        },
    ]
    
    # Save CSV
    csv_path = output_path / f"demo_results_{timestamp}.csv"
    fieldnames = [
        'framework', 'model', 'quantization', 'device', 'beam_size',
        'load_time_sec', 'transcribe_time_sec', 'rtf', 'total_time_sec',
        'ram_before_mb', 'ram_after_mb', 'ram_peak_mb',
        'vram_before_mb', 'vram_after_mb', 'vram_peak_mb',
        'wer', 'cer', 'word_count', 'char_count',
        'status', 'error_message'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in demo_results:
            writer.writerow(result)
    
    # Save JSON
    json_path = output_path / f"demo_results_{timestamp}.json"
    json_data = {
        'system_config': system_config,
        'benchmark_args': {
            'audio': 'demo_audio.wav',
            'frameworks': 'faster-whisper,whisper.cpp',
            'models': 'tiny,base,small,medium,large-v3,bond005/whisper_large_v2_ru',
            'mode': 'full',
            'language': 'ru'
        },
        'timestamp_start': datetime.now().isoformat(),
        'audio_duration_sec': 300.0,
        'results': demo_results,
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # Generate Markdown report
    report_path = output_path / f"demo_report_{timestamp}.md"
    
    passed = [r for r in demo_results if r['status'] == 'PASS']
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Whisper Model Benchmark Report (Demo)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("> ⚠️ Это демонстрационный отчет с симулированными данными.\n")
        f.write("> Реальные результаты зависят от вашей системы.\n\n")
        
        # System Info
        f.write("## System Configuration\n\n")
        f.write(f"- **CPU**: {system_config['cpu_model']}\n")
        f.write(f"- **CPU Cores**: {system_config['cpu_cores']}\n")
        f.write(f"- **RAM**: {system_config['total_ram_gb']:.1f} GB\n")
        f.write(f"- **GPU**: {system_config['gpu_name']}\n")
        f.write(f"- **VRAM**: {system_config['total_vram_gb']:.1f} GB\n")
        f.write(f"- **CUDA**: {system_config['cuda_version']}\n")
        f.write(f"- **Python**: {system_config['python_version']}\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Total Tests**: {len(demo_results)}\n")
        f.write(f"- **Passed**: {len(passed)}\n")
        f.write(f"- **Failed (OOM)**: {len(demo_results) - len(passed)}\n\n")
        
        # Best by RTF
        f.write("## 🏆 Best Performance (Lowest RTF)\n\n")
        best_rtf = sorted(passed, key=lambda x: x['rtf'])[:5]
        f.write("| Rank | Framework | Model | Quantization | Device | Beam | RTF | Time (s) | VRAM (MB) |\n")
        f.write("|------|-----------|-------|--------------|--------|------|-----|----------|----------|\n")
        for i, r in enumerate(best_rtf, 1):
            f.write(f"| {i} | {r['framework']} | {r['model']} | {r['quantization']} | {r['device']} | {r['beam_size']} | "
                   f"{r['rtf']:.3f} | {r['total_time_sec']:.1f} | {r['vram_peak_mb']:.0f} |\n")
        f.write("\n")
        
        # Best by VRAM
        f.write("## 💾 Most Memory Efficient (Lowest VRAM)\n\n")
        best_vram = sorted(passed, key=lambda x: x['vram_peak_mb'] if x['vram_peak_mb'] > 0 else float('inf'))[:5]
        f.write("| Rank | Framework | Model | Quantization | Device | Beam | VRAM (MB) | RTF | Time (s) |\n")
        f.write("|------|-----------|-------|--------------|--------|------|-----------|-----|----------|\n")
        for i, r in enumerate(best_vram, 1):
            f.write(f"| {i} | {r['framework']} | {r['model']} | {r['quantization']} | {r['device']} | {r['beam_size']} | "
                   f"{r['vram_peak_mb']:.0f} | {r['rtf']:.3f} | {r['total_time_sec']:.1f} |\n")
        f.write("\n")
        
        # Best by WER
        wer_results = [r for r in passed if r['wer'] is not None]
        if wer_results:
            f.write("## 🎯 Most Accurate (Lowest WER)\n\n")
            best_wer = sorted(wer_results, key=lambda x: x['wer'])[:5]
            f.write("| Rank | Framework | Model | Quantization | WER (%) | CER (%) | RTF | VRAM (MB) |\n")
            f.write("|------|-----------|-------|--------------|---------|---------|-----|----------|\n")
            for i, r in enumerate(best_wer, 1):
                f.write(f"| {i} | {r['framework']} | {r['model']} | {r['quantization']} | "
                       f"{r['wer']:.2f} | {r['cer']:.2f} | {r['rtf']:.3f} | {r['vram_peak_mb']:.0f} |\n")
            f.write("\n")
        
        # Detailed Results
        f.write("## Detailed Results\n\n")
        f.write("| Framework | Model | Quantization | Device | Beam | RTF | Time (s) | VRAM (MB) | RAM (MB) | WER (%) | Status |\n")
        f.write("|-----------|-------|--------------|--------|------|-----|----------|-----------|----------|---------|--------|\n")
        for r in sorted(demo_results, key=lambda x: (x['framework'], x['model'], x['device'])):
            wer_str = f"{r['wer']:.2f}" if r['wer'] else "N/A"
            vram_str = f"{r['vram_peak_mb']:.0f}" if r['vram_peak_mb'] > 0 else "N/A"
            f.write(f"| {r['framework']} | {r['model']} | {r['quantization']} | {r['device']} | {r['beam_size']} | "
                   f"{r['rtf']:.3f} | {r['total_time_sec']:.1f} | {vram_str} | {r['ram_peak_mb']:.0f} | {wer_str} | {r['status']} |\n")
        f.write("\n")
        
        # Recommendations
        f.write("## 💡 Recommendations\n\n")
        
        gpu_optimal = min([r for r in passed if r['device'] == 'cuda' and r['vram_peak_mb'] < 2000], 
                         key=lambda x: x['rtf'], default=None)
        cpu_optimal = min([r for r in passed if r['device'] == 'cpu'], 
                         key=lambda x: x['rtf'], default=None)
        accuracy_optimal = min([r for r in passed if r['wer'] is not None], 
                              key=lambda x: x['wer'], default=None)
        
        if gpu_optimal:
            f.write(f"### Best GPU Performance (< 2GB VRAM)\n")
            f.write(f"- **Model**: {gpu_optimal['model']}\n")
            f.write(f"- **Framework**: {gpu_optimal['framework']}\n")
            f.write(f"- **Quantization**: {gpu_optimal['quantization']}\n")
            f.write(f"- **Beam Size**: {gpu_optimal['beam_size']}\n")
            f.write(f"- **RTF**: {gpu_optimal['rtf']:.3f} ({300 * gpu_optimal['rtf']:.1f}s для 5 мин аудио)\n")
            f.write(f"- **VRAM**: {gpu_optimal['vram_peak_mb']:.0f} MB\n")
            f.write(f"- **WER**: {gpu_optimal['wer']:.2f}%\n\n")
        
        if cpu_optimal:
            f.write(f"### Best CPU Performance (GPU не требуется)\n")
            f.write(f"- **Model**: {cpu_optimal['model']}\n")
            f.write(f"- **Framework**: {cpu_optimal['framework']}\n")
            f.write(f"- **Quantization**: {cpu_optimal['quantization']}\n")
            f.write(f"- **RTF**: {cpu_optimal['rtf']:.3f} ({300 * cpu_optimal['rtf']:.1f}s для 5 мин аудио)\n")
            f.write(f"- **RAM**: {cpu_optimal['ram_peak_mb']:.0f} MB\n")
            f.write(f"- **WER**: {cpu_optimal['wer']:.2f}%\n\n")
        
        if accuracy_optimal:
            f.write(f"### Лучшая точность (минимальный WER)\n")
            f.write(f"- **Model**: {accuracy_optimal['model']}\n")
            f.write(f"- **Framework**: {accuracy_optimal['framework']}\n")
            f.write(f"- **Quantization**: {accuracy_optimal['quantization']}\n")
            f.write(f"- **WER**: {accuracy_optimal['wer']:.2f}%\n")
            f.write(f"- **CER**: {accuracy_optimal['cer']:.2f}%\n")
            f.write(f"- **VRAM**: {accuracy_optimal['vram_peak_mb']:.0f} MB\n")
            f.write(f"- **RTF**: {accuracy_optimal['rtf']:.3f}\n\n")
        
        # Key findings
        f.write("## 🔍 Key Findings\n\n")
        f.write("1. **medium int8_float32 на GPU** потребляет ~820-920 MB VRAM (в пределах 2GB)\n")
        f.write("2. **large-v3** вызывает OOM на 2GB VRAM при загрузке модели\n")
        f.write("3. **whisper.cpp** использует меньше RAM чем faster-whisper на CPU\n")
        f.write("4. **beam_size=1** оптимален для скорости, beam_size=5 дает лучшую точность\n")
        f.write("5. **Русскоязычные модели** (bond005/whisper_large_v2_ru) показывают лучшую точность\n")
        f.write("6. **int8_float32** квантизация дает лучший баланс точности и памяти\n\n")
        
        # Production recommendations
        f.write("## 🚀 Production Recommendations\n\n")
        f.write("### Для транскрипции голосовых сообщений до 5 минут:\n\n")
        f.write("**Оптимальная конфигурация (GPU)**:\n")
        f.write("- Модель: `small` или `medium`\n")
        f.write("- Фреймворк: `faster-whisper`\n")
        f.write("- Квантизация: `int8_float32`\n")
        f.write("- Beam size: `1` (скорость) или `3` (баланс)\n")
        f.write("- Устройство: `cuda`\n\n")
        
        f.write("**Ожидаемые метрики**:\n")
        f.write("- small: RTF ~0.075 (22.5s для 5 мин), VRAM ~850MB, WER ~5%\n")
        f.write("- medium: RTF ~0.15 (45s для 5 мин), VRAM ~890MB, WER ~4%\n\n")
        
        f.write("### Если GPU недоступен:\n")
        f.write("- Модель: `small`\n")
        f.write("- Фреймворк: `faster-whisper` или `whisper.cpp`\n")
        f.write("- Квантизация: `int8_float32` или `q8_0`\n")
        f.write("- Потоки: `2` (для AMD A4-5300)\n\n")
        
        f.write("---\n")
        f.write("*Report generated by Whisper Benchmark Harness*\n")
    
    print(f"Demo results created in {output_path}/")
    print(f"  - CSV: {csv_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - Report: {report_path}")
    
    return output_path


if __name__ == '__main__':
    print("="*60)
    print("Whisper Benchmark Harness - Demo Results Generator")
    print("="*60)
    print()
    print("Creating demonstration benchmark results...")
    print("(Simulated data for AMD A4-5300 + GTX 1050 2GB system)")
    print()
    
    output_path = create_demo_results()
    
    print()
    print("="*60)
    print("✅ Demo results created successfully!")
    print("="*60)
    print()
    print("To view the report:")
    print(f"  cat {output_path / list(output_path.glob('*.md'))[0]}")
    print()
    print("To run real benchmarks:")
    print("  pip install -r requirements.txt")
    print("  python whisper_benchmark.py --audio your_audio.wav --mode quick")
