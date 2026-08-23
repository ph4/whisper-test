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

Файл `requirements.txt` содержит все зависимости с комментариями. Для включения опциональных функций раскомментируйте соответствующие строки:

```bash
# Отредактировать requirements.txt и раскомментировать нужные строки, затем:
pip install -r requirements.txt
```

**Опционально:**

- **PyTorch** (для улучшенного управления VRAM): Если не установлен, используется ctypes + CUDA driver API
- **whisper.cpp** (pywhispercpp): Для квантованных моделей
- **ONNX Runtime** (onnxruntime-gpu): Для Sber GigaAM и других ONNX моделей (рекомендуется для систем с малым RAM)
- **Инструменты разработки** (pytest, black, flake8): Для тестирования и разработки

См. раздел "Platform-Specific Notes" в `requirements.txt` для деталей установки.

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
├── benchmark.py             # Бенчмарк харнесс с YAML поддержкой
├── test_harness.py          # Тесты для бенчмарк харнесса
├── requirements.txt         # Все зависимости (обязательные и опциональные)
├── sample_benchmark_config.yaml  # Пример YAML конфигурации
├── README.md                # Документация
└── YAML_DESIGN.md           # Полная спецификация YAML конфигурации
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

### 📝 Полная спецификация YAML конфигурации

Конфигурация поддерживает все возможные опции из DESIGN документа:

#### Глобальные настройки (опционально)

```yaml
settings:
  output_dir: "benchmark_results"
  mode: "quick"  # quick, full, compare
  default_language: "ru"
  default_threads: 2
```

#### Test Datasets с ground truth

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
    
  - name: "multilingual_mix"
    audio_path: "mixed_audio.wav"
    # Без reference = только timing метрики (без WER/CER)
    languages: ["ru", "en"]
```

#### Benchmark Configurations

**1. SHORTCUT формат - faster-whisper (plural form)**

```yaml
benchmarks:
  - framework: faster-whisper
    models: [medium, small]  # plural: тестирует обе модели
    quantizations: [int8_float32, float16]
    devices: [cuda]
    beam_sizes: [1, 3]
```

**2. SHORTCUT формат - whisper.cpp (singular form)**

```yaml
  - framework: whisper.cpp
    model: ggerganov/whisper.cpp  # singular form
    quantization: [q5_0, q4_0]   # может быть списком
    devices: [cpu]
    threads: 2
```

**3. HuggingFace модели - авто-детект квантизаций**

```yaml
  - framework: huggingface
    models: 
      - openai/whisper-small-ru
      - sberbank-ai/whisper-medium-ru
    devices: [cuda]
    load_in_8bit: false
    # torch_dtype авто-детектится из модели
```

**4. Sber GigaAM с ONNX (рекомендуется для систем с малым RAM)**

```yaml
  - framework: sber
    models: [ctc, rnnt]  # типы моделей: ctc или rnnt
    use_onnx: true
    devices: [cpu, cuda]
```

**5. ONNX ASR модели (GigaAM v3 через onnx-asr)**

```yaml
  - framework: onnx-asr
    models: [gigaam-v3-e2e-ctc, gigaam-v3-e2e-rnnt]
    devices: [cuda, cpu]
    execution_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

**6. FULL EXPLICIT формат - transcribe.cpp с GPU offloading**

Идеально для GGUF моделей типа GigaAM v3:

```yaml
  - framework: transcribe.cpp
    models: [handy-computer/gigaam-v3-e2e-rnnt-gguf]
    quantizations: [Q6_K, Q8_0, Q5_K_M]
    devices: [cuda, cpu, offload]
    threads: 4
    # Конфигурации offload - тестирует разные GPU/CPU split
    offload_configs:
      # Опция 1: Split по слоям - первые 10 слоев на GPU
      - layer_count: 10
        split_mode: "layer"
      # Опция 2: Все на CPU (без offloading)
      - layer_count: 0
        split_mode: "none"
      # Опция 3: Все на GPU (если VRAM позволяет)
      - layer_count: 999
        split_mode: "layer"
      # Опция 4: Динамически на основе доступного VRAM
      - split_mode: "auto"
        max_vram_gb: 2.0
```

**7. Explicit per-config override - полный контроль**

```yaml
  - framework: faster-whisper
    model: large-v3
    quantizations: [float16]
    devices: [cuda]
    beam_sizes: [5]
    language: "ru"
    compute_type: "float16"
    gpu_id: 0
    threads: 4
```

**8. Mixed device testing - одна модель на CPU и GPU**

```yaml
  - framework: faster-whisper
    model: small
    quantizations: [int8_float32]
    devices: [cuda, cpu]  # Тестирует оба устройства
    beam_sizes: [1]
```

**9. Minimal config - все значения по умолчанию**

```yaml
  - framework: faster-whisper
    # model defaults to "small"
    # quantizations auto-detected as [int8_float32]
    # devices auto-detected как [cuda] или [cpu] в зависимости от системы
```

### 🧩 Offload Configuration Options

Для фреймворков с поддержкой GPU offloading (transcribe.cpp, llama.cpp-based):

```yaml
offload_configs:
  # Option 1: Split по слоям
  - layer_count: <int>      # Количество слоев для GPU
    split_mode: "layer"     # Послойный split
    
  # Option 2: Split по rows/blocks  
  - block_count: <int>
    split_mode: "row"
    
  # Option 3: Без offloading (только CPU)
  - split_mode: "none"
  
  # Option 4: Полный GPU (если память позволяет)
  - split_mode: "full_gpu"
  
  # Option 5: Динамически на основе доступного VRAM
  - split_mode: "auto"
    max_vram_gb: 2.0        # Не превышать этот лимит VRAM
```

### 📋 Поддерживаемые фреймворки

| Framework | Shortcut | Plural/Singular | Offload Support | Auto Quantizations |
|-----------|----------|-----------------|-----------------|-------------------|
| faster-whisper | ✅ | ✅ | ❌ | ✅ |
| whisper.cpp | ✅ | ✅ | ⚠️ (via main.cpp params) | ✅ |
| huggingface | ✅ | ✅ | ❌ (uses device_map) | ✅ |
| sber | ✅ | ✅ | ❌ | ✅ |
| onnx-asr | ✅ | ✅ | ✅ (execution providers) | ✅ |
| transcribe.cpp | ✅ | ✅ | ✅ (layer offloading) | ✅ |

### 🔑 Ключевые особенности

1. **Гибкий ввод**: Оба формата `model: small` и `models: [small, medium]` работают
2. **Умные дефолты**: Если quantizations не указаны, авто-детект на основе фреймворка
3. **Test Datasets**: Определите несколько аудиофайлов с ground truth один раз, используйте во всех бенчмарках
4. **Offload Control**: Детальный контроль GPU offloading для систем с ограниченной памятью
5. **Framework Agnostic**: Одна структура YAML работает со всеми поддерживаемыми фреймворками

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
