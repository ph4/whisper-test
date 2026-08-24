# Whisper Benchmark Harness

Комплексный testing harness для сравнительного анализа производительности и точности моделей Whisper.

## Возможности

- **Поддержка фреймворков**: faster-whisper, whisper.cpp, transcribe.cpp (GGUF), HuggingFace transformers
- **Модели**: tiny, base, small, medium, large-v2, large-v3 + русскоязычные модели
- **Квантизации**: float32, float16, int8, int8_float16, int8_float32, GGUF (Q4_0, Q5_K_M, Q8_0, etc.)
- **Мониторинг памяти**: RAM и VRAM с интервалом 100мс
- **Метрики**: RTF, WER, CER, время загрузки/транскрипции
- **Режимы**: quick, full, compare, optimal
- **Отчеты**: CSV, JSON, Markdown
- **Self-Test**: Автоматическая проверка установки библиотек и загрузки моделей (CPU/GPU)

## Установка

```bash
# Установить зависимости
pip install -r requirements.txt

# Для whisper.cpp (опционально)
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make
# Для GPU поддержки:
# make LLAMA_CUDA=1
```

## Быстрый старт

### Базовое тестирование

```bash
python whisper_benchmark.py --audio test_audio.wav --mode quick
```

### Полное тестирование

```bash
python whisper_benchmark.py \
    --audio test_audio.wav \
    --frameworks faster-whisper,whisper.cpp \
    --models tiny,base,small,medium,large-v3 \
    --quantizations int8,int8_float16,int8_float32 \
    --beam-sizes 1,3,5 \
    --output-dir results/ \
    --mode full \
    --language ru \
    --monitor-memory-interval 100
```

### С ground truth для оценки точности

```bash
python whisper_benchmark.py \
    --audio test_audio.wav \
    --ground-truth "Эталонная транскрипция на русском языке" \
    --language ru \
    --mode quick
```

### Конвертация аудио в нужный формат

```bash
python whisper_benchmark.py \
    --audio input.mp3 \
    --convert-audio \
    --mode quick
```

## Аргументы командной строки

| Аргумент | Описание | По умолчанию |
|----------|----------|--------------|
| `--audio` | Путь к аудиофайлу | (обязательно) |
| `--ground-truth` | Эталонная транскрипция для WER/CER | None |
| `--frameworks` | Фреймворки через запятую (faster-whisper, whisper.cpp, transcribe.cpp, huggingface) | faster-whisper |
| `--models` | Модели через запятую | tiny,base,small,medium,large-v2,large-v3 |
| `--quantizations` | Квантизации через запятую | int8,int8_float16,int8_float32,float16,float32 |
| `--beam-sizes` | Размеры луча через запятую | 1,3,5 |
| `--output-dir` | Директория для результатов | benchmark_results |
| `--mode` | Режим тестирования | full |
| `--language` | Язык транскрипции | ru |
| `--threads` | Количество CPU потоков | 2 |
| `--gpu-id` | ID GPU для multi-GPU | 0 |
| `--whisper-cpp-path` | Путь к whisper.cpp | ./whisper.cpp |
| `--monitor-memory-interval` | Интервал мониторинга (мс) | 100 |
| `--warmup-runs` | Прогревочные запуски | 1 |
| `--convert-audio` | Конвертировать аудио в WAV | False |

## 🔧 Self-Test / Диагностика

**Проверка установки и работоспособности всех фреймворков:**

```bash
# Перейти в директорию whisper_harness
cd whisper_harness

# Быстрая проверка импортов библиотек
python -m transcribers.self_test --quick

# Полное тестирование с загрузкой минимальных моделей (tiny) на CPU и GPU
python -m transcribers.self_test

# Тест конкретного фреймворка
python -m transcribers.self_test --framework faster-whisper
python -m transcribers.self_test --framework whisper.cpp
python -m transcribers.self_test --framework transcribe.cpp
python -m transcribers.self_test --framework huggingface

# Только GPU или только CPU тесты
python -m transcribers.self_test --gpu-only
python -m transcribers.self_test --cpu-only

# Экспорт результатов в JSON
python -m transcribers.self_test --output selftest_results.json
```

**Что проверяется:**
1. ✅ Наличие установленных библиотек (faster_whisper, pywhispercpp, transformers, onnxruntime, etc.)
2. ✅ Возможность загрузки минимальной модели (tiny) для каждого фреймворка
3. ✅ Работа на CPU и GPU (если доступно)
4. ✅ Базовая транскрипция тестового аудио (1 секунда тишины)

**Интерпретация результатов:**
- ✅ **PASSED** - Библиотека установлена и модель загружается успешно
- ❌ **FAILED** - Ошибка: требуется установка библиотеки или исправление конфигурации
- ⚠️ **SKIPPED** - Пропущено (например, CUDA недоступен или quick mode)
- ⚡ **WARNING** - Работает, но есть ограничения (например, нет GPU или используется fallback)

## Режимы работы

### quick
Быстрое тестирование основных конфигураций (small, medium). Пропускает заведомо неподходящие модели для 2GB VRAM.

### full
Полное тестирование всех комбинаций моделей, квантизаций и beam sizes. Может занять несколько часов.

### compare
Сравнение фреймворков faster-whisper vs whisper.cpp на одинаковых моделях.

### optimal
Автоматический поиск оптимальной конфигурации по балансу скорость/память.

## Выходные файлы

### CSV (`results_YYYYMMDD_HHMMSS.csv`)
Детальные метрики для каждой конфигурации:
- framework, model, quantization, device, beam_size
- load_time_sec, transcribe_time_sec, rtf, total_time_sec
- ram_before_mb, ram_after_mb, ram_peak_mb
- vram_before_mb, vram_after_mb, vram_peak_mb
- wer, cer, word_count, char_count
- status, error_message

### JSON (`results_YYYYMMDD_HHMMSS.json`)
Полная информация включая:
- Конфигурация системы (CPU, RAM, GPU, CUDA version)
- Версии библиотек
- Timestamp начала/окончания
- Все метрики из CSV

### Markdown отчет (`report_YYYYMMDD_HHMMSS.md`)
- Сводная таблица с лучшими результатами
- Топ-5 по производительности (RTF)
- Топ-5 по эффективности памяти (VRAM)
- Топ-5 по точности (WER)
- Детальные результаты по фреймворкам
- Рекомендации по оптимальным конфигурациям

## Пример отчета

```markdown
## 🏆 Best Performance (Lowest RTF)

| Rank | Framework | Model | Quantization | Device | Beam | RTF | Time (s) | VRAM (MB) |
|------|-----------|-------|--------------|--------|------|-----|----------|-----------|
| 1 | faster-whisper | small | int8_float32 | cuda | 1 | 0.125 | 37.5 | 650 |
| 2 | faster-whisper | medium | int8_float32 | cuda | 1 | 0.245 | 73.5 | 820 |
...

## 💡 Recommendations

### Best GPU Performance (< 2GB VRAM)
- **Model**: small
- **Framework**: faster-whisper
- **Quantization**: int8_float32
- **RTF**: 0.125
- **VRAM**: 650 MB
```

## Русскоязычные модели

Harness автоматически проверяет поддержку русского языка. Рекомендуемые модели:

### Стандартные Whisper
- `large-v3` - лучшая поддержка русского
- `large-v2` - отличная поддержка
- `medium`, `small`, `base`, `tiny` - хорошая поддержка

### Специализированные (HuggingFace)
- `bond005/whisper_large_v2_ru` - fine-tuned для русского
- Другие модели с суффиксом `_ru` или `russian`

## Мониторинг памяти

Harness измеряет:
- **RAM**: через psutil (или /proc/self/status на Linux)
- **VRAM**: через pynvml (или nvidia-smi fallback)
- **Частота**: каждые 100мс во время транскрипции
- **Метрики**: до загрузки, после загрузки, пиковое потребление

## Обработка ошибок

- **OOM (Out of Memory)**: graceful handling, пропуск конфигурации
- **Таймауты**: 10 минут на транскрипцию
- **Логирование**: все ошибки детально логируются
- **Продолжение**: тестирование продолжается после ошибок

## Воспроизводимость

- Фиксированный random seed
- Очистка GPU памяти между тестами (`torch.cuda.empty_cache()`)
- Warm-up run перед измерением
- Перезапуск процесса для каждого теста (опционально)

## Системные требования

### Минимальные
- CPU: 2 ядра
- RAM: 4 GB
- GPU: NVIDIA с 2GB VRAM (опционально)

### Рекомендуемые
- CPU: 4+ ядра
- RAM: 8+ GB
- GPU: NVIDIA GTX 1050 Ti или лучше с 4GB+ VRAM

## Лицензия

MIT License
