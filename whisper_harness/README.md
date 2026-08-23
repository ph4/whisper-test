# Whisper Transcription Harness

Модульный харнесс для бенчмаркинга и тестирования ASR-моделей (Whisper, Sber GigaAM) на ресурсоограниченных системах.

## 🎯 Целевое железо

- **CPU**: AMD A4-5300 (2 ядра, 3.4 ГГц)
- **RAM**: 4-8 ГБ DDR3
- **GPU**: NVIDIA GTX 1050 2GB
- **Python**: 3.12+

## 📦 Установка

### Базовая установка

```bash
# Создать venv с uv
uv venv
source .venv/bin/activate

# Установить зависимости
pip install -r requirements.txt
```

### Опциональные зависимости

```bash
# Для whisper.cpp (рекомендуется q5_0 для баланса скорость/качество)
pip install pywhispercpp

# Для Sber GigaAM через ONNX (экономит RAM)
pip install onnxruntime

# Для точного расчета WER
pip install jiwer

# Для загрузки конфигов из YAML
pip install pyyaml
```

## 📁 Структура проекта

```
whisper_harness/
├── transcribers/
│   ├── __init__.py
│   ├── base.py              # Базовый класс Transcriber
│   ├── faster_whisper.py    # Адаптер для faster-whisper
│   ├── whisper_cpp.py       # Адаптер для whisper.cpp
│   ├── sber.py              # Адаптер для Sber GigaAM
│   └── hf_whisper.py        # Адаптер для HuggingFace Whisper
├── utils/
│   ├── __init__.py
│   ├── metrics.py           # Расчет WER, CER, RTF
│   └── memory_monitor.py    # Мониторинг RAM/VRAM
├── cli.py                   # Точка входа CLI
├── requirements.txt         # Зависимости
└── README.md                # Документация
```

## 💡 Примеры использования (CLI)

### 1. Быстрый тест Faster-Whisper (оптимально для 2GB VRAM)

```bash
python cli.py --audio test.wav --model-type fast_whisper --model-id medium --compute-type int8_float32 --language ru
```

### 2. Тест Whisper.cpp с автозагрузкой квантованной модели

```bash
python cli.py --audio test.wav --model-type whisper_cpp --model-id ggerganov/whisper.cpp --quantization q5_0 --language ru
```

### 3. Тест Sber GigaAM через ONNX (экономия RAM)

```bash
python cli.py --audio test.wav --model-type sber_gigaam_ctc --use-onnx
```

### 4. Тест с оценкой точности (WER/CER)

```bash
python cli.py --audio test.wav --model-type hf_whisper --model-id sberbank-ai/whisper-small-ru --reference ground_truth.txt --output result.json
```

### 5. Загрузка конфигурации из YAML

```bash
python cli.py --audio test.wav --config benchmark_config.yaml
```

## 📊 Benchmark Harness (Updated!)

Запуск автоматизированного бенчмарка с несколькими конфигурациями:

```bash
# Быстрый бенчмарк с конфиг по умолчанию
python benchmark.py --audio test.wav --mode quick

# Полный бенчмарк с YAML конфигом
python benchmark.py --audio test.wav --config benchmark_config.yaml --mode full

# С ground truth для расчета метрик точности
python benchmark.py --audio test.wav --reference ground_truth.txt

# Генерация примера конфига
python benchmark.py --generate-sample-config
```

### Формат YAML конфигурации

Поддерживаются оба формата - **singular** и **plural**:

**1. Shortcut формат (рекомендуется)** - просто укажите framework, model/models и quantizations:

```yaml
benchmarks:
  # Faster-Whisper с несколькими моделями (plural form)
  - framework: faster-whisper
    models: [medium, small]
    quantizations: [int8_float32, float16]
    devices: [cuda]
    beam_sizes: [1, 3]
    
  # Whisper.cpp - singular form тоже работает
  - framework: whisper.cpp
    model: ggerganov/whisper.cpp
    quantization: [q5_0, q4_0]
    devices: [cpu]
    
  # HuggingFace Whisper - квантизации авто-детектятся
  - framework: huggingface
    models: 
      - openai/whisper-small-ru
      - sberbank-ai/whisper-medium-ru
    devices: [cuda]
```

**2. Полный явный формат с offloading** - для transcribe.cpp и подобных:

```yaml
benchmarks:
  - framework: transcribe.cpp
    models: [handy-computer/gigaam-v3-e2e-rnnt-gguf]
    quantizations: [Q6_K, Q8_0, Q5_K_M]
    devices: [cuda, cpu, offload]
    offload_configs:
      - layer_count: 10  # offload first 10 layers to GPU
        split_mode: "layer"
      - layer_count: 0   # all on CPU
        split_mode: "none"
      - layer_count: 999 # all on GPU (if fits)
        split_mode: "layer"
    threads: 4
```

**3. Test datasets с ground truth** - для автоматического WER/CER расчета:

```yaml
test_datasets:
  - name: "russian_speech_sample"
    audio_path: "/path/to/audio.wav"
    reference_text: "/path/to/ground_truth.txt"
    # Или inline текст:
    # reference_text: "Привет мир это тестовая транскрипция"
    language: "ru"
    
  - name: "english_podcast"
    audio_path: "podcast_ep01.wav"
    reference_text: "podcast_transcript.txt"
    language: "en"
```

При наличии `test_datasets` бенчмарк запустится на каждом аудиофайле из списка.

### Поддерживаемые фреймворки

| Framework | Shortcut | Plural/Singular | Offload Support | Auto Quantizations |
|-----------|----------|-----------------|-----------------|-------------------|
| faster-whisper | ✅ | ✅ | ❌ | ✅ |
| whisper.cpp | ✅ | ✅ | ⚠️ | ✅ |
| huggingface | ✅ | ✅ | ❌ | ✅ |
| sber | ✅ | ✅ | ❌ | ✅ |
| onnx-asr | ✅ | ✅ | ✅ | ✅ |
| transcribe.cpp | ✅ | ✅ | ✅ | ✅ |

## 📊 Интерпретация метрик

| Метрика | Описание | Оптимальное значение |
|---------|----------|---------------------|
| **RTF** (Real Time Factor) | `время_транскрипции / длительность_аудио` | < 1.0 (быстрее реального времени) |
| **WER** (Word Error Rate) | Процент ошибок распознавания слов | < 10% (отлично для русского) |
| **CER** (Character Error Rate) | Процент ошибок распознавания символов | < 5% |
| **RAM peak** | Пиковое потребление оперативной памяти | < 4000 MB для 4GB систем |
| **VRAM peak** | Пиковое потребление видеопамяти | < 1800 MB для GTX 1050 2GB |

## 🛠️ Как добавить новую модель

### Шаг 1: Создать файл адаптера

Создайте новый файл в `transcribers/`, например `yandex.py`:

```python
from transcribers.base import Transcriber
from typing import Any

class YandexWhisperTranscriber(Transcriber):
    def _load_model(self) -> None:
        # Логика загрузки модели
        pass
    
    def transcribe(self, audio_path: str, language: str = "ru") -> dict[str, Any]:
        # Логика транскрипции
        pass
```

### Шаг 2: Добавить импорт в `__init__.py`

```python
# transcribers/__init__.py
from transcribers.yandex import YandexWhisperTranscriber

__all__ = [..., "YandexWhisperTranscriber"]
```

### Шаг 3: Добавить в фабрику CLI

```python
# cli.py
transcriber_map = {
    ...
    "yandex": YandexWhisperTranscriber,
}
```

## ⚠️ Известные ограничения и Troubleshooting

### OOM на GTX 1050 2GB

**Проблема**: Модель не помещается в VRAM.

**Решения**:
1. Уменьшите `beam_size` до 1
2. Используйте `int8_float32` вместо `float16`
3. Переключитесь на `whisper.cpp` с квантованием `q4_0`/`q5_0`
4. Используйте CPU для больших моделей

```bash
# Пример: smaller model with int8
python cli.py --audio test.wav --model-type fast_whisper --model-id small --compute-type int8_float32
```

### Нехватка RAM для Sber GigaAM

**Проблема**: PyTorch-версия потребляет >4GB RAM.

**Решение**: Используйте ONNX-версию

```bash
python cli.py --audio test.wav --model-type sber_gigaam_ctc --use-onnx
```

### whisper.cpp бинарь не найден

**Проблема**: `pywhispercpp` не установлен, бинарный файл не найден.

**Решения**:
1. Установите `pywhispercpp`: `pip install pywhispercpp`
2. Или скачайте whisper.cpp вручную:
   ```bash
   git clone https://github.com/ggerganov/whisper.cpp
   cd whisper.cpp && make
   cp main /usr/local/bin/whisper-cli
   ```

## 🏆 Рекомендации для GTX 1050 2GB

| Модель | Квантизация | VRAM | RTF | Рекомендация |
|--------|-------------|------|-----|--------------|
| tiny | int8_float32 | ~200MB | 0.1x | ✅ Отлично для быстрых задач |
| base | int8_float32 | ~350MB | 0.15x | ✅ Хороший баланс |
| small | int8_float32 | ~500MB | 0.3x | ✅ Рекомендуется для продакшена |
| medium | int8_float32 | ~850MB | 0.5x | ✅ Работает, но медленно |
| large-v3 | int8_float32 | ~2.5GB | ❌ OOM | Не рекомендуется |

**Оптимальная конфигурация для продакшена**:
```bash
python cli.py --audio test.wav \
    --model-type fast_whisper \
    --model-id small \
    --compute-type int8_float32 \
    --beam-size 1 \
    --language ru
```

## 📝 Лицензия

MIT License
