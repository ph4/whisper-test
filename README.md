# Whisper Harness

Модульный харнесс для бенчмаркинга и тестирования ASR-моделей (Whisper, Sber GigaAM) на ресурсоограниченных системах.

> **Перенесён в `src/whisper_harness/`.** Старая одномодульная версия — в `legacy/`.

## 🎯 Целевое железо

- **CPU**: AMD A4-5300 (2 ядра, 3.4 ГГц)
- **RAM**: 4–8 ГБ DDR3
- **GPU**: NVIDIA GTX 1050 2 ГБ
- **Python**: 3.10+

## Возможности

- **Фреймворки**: faster-whisper, whisper.cpp, transcribe.cpp (GGUF), HuggingFace transformers, Sber GigaAM (ONNX)
- **Модели**: tiny, base, small, medium, large-v2, large-v3 + русскоязычные модели
- **Квантизации**: float32, float16, int8, int8_float16, int8_float32, GGUF (Q4_0, Q5_K_M, Q8_0 и др.)
- **GPU offloading**: послойный offload для систем с ≤2 ГБ VRAM
- **Мониторинг памяти**: RAM и VRAM с интервалом 100 мс
- **Метрики**: RTF, WER, CER, время загрузки/транскрипции
- **Режимы**: quick, full, compare
- **Отчёты**: CSV, JSON, Markdown
- **Self-test**: автоматическая проверка установки библиотек и загрузки моделей (CPU/GPU)

## Установка

```bash
# Создать venv и установить зависимости
uv venv
source .venv/bin/activate
pip install -r requirements.txt

# Опционально: PyTorch для улучшенного управления VRAM
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Опционально: whisper.cpp
pip install pywhispercpp

# Опционально: ONNX Runtime для Sber GigaAM
pip install onnxruntime-gpu
```

Или установить как пакет:

```bash
pip install -e .

# После чего доступны команды:
whisper-bench --help
whisper-cli --help
```

## Структура проекта

```
whisper-test/
├── src/whisper_harness/          # Основной код
│   ├── cli.py                    # CLI для одиночного запуска
│   ├── benchmark.py              # Бенчмарк харнесс
│   ├── transcribers/             # Адаптеры фреймворков
│   │   ├── base.py               # Базовый класс Transcriber
│   │   ├── faster_whisper.py
│   │   ├── whisper_cpp.py
│   │   ├── hf_whisper.py
│   │   ├── transcribe_cpp.py
│   │   ├── sber.py
│   │   └── self_test.py         # Диагностика установки
│   └── utils/
│       ├── metrics.py            # WER, CER, RTF
│       └── memory_monitor.py     # RAM/VRAM мониторинг
├── configs/                      # YAML-конфигурации
│   └── sample_benchmark_config.yaml
├── tests/                        # pytest тесты
│   └── test_harness.py
├── docs/
│   └── YAML_DESIGN.md            # Полная спецификация YAML
├── examples/                     # Демо-скрипты и примеры
│   ├── demo_benchmark.py
│   └── demo_results/
└── legacy/                       # Старая одномодульная версия
```

## Быстрый старт

### CLI — одиночная транскрипция

```bash
# Faster-Whisper (рекомендуется, оптимально для 2 ГБ VRAM)
python -m whisper_harness.cli \
    --audio test.wav \
    --model-type fast_whisper \
    --model-id medium \
    --compute-type int8_float32 \
    --language ru

# Whisper.cpp с квантованной моделью
python -m whisper_harness.cli \
    --audio test.wav \
    --model-type whisper_cpp \
    --model-id ggerganov/whisper.cpp \
    --quantization q5_0

# Transcribe.cpp + GGUF GigaAM (рекомендуется для малой памяти)
python -m whisper_harness.cli \
    --audio test.wav \
    --model-type transcribe_cpp \
    --model-id handy-computer/gigaam-v3-e2e-rnnt-gguf \
    --quantization Q5_K_M
```

### Бенчмарк — массовое тестирование

```bash
# Быстрый бенчмарк
python -m whisper_harness.benchmark --audio test.wav --mode quick

# Полный бенчмарк с YAML
python -m whisper_harness.benchmark \
    --audio test.wav \
    --config configs/sample_benchmark_config.yaml \
    --mode full

# С ground truth для WER/CER
python -m whisper_harness.benchmark \
    --audio test.wav \
    --reference ground_truth.txt \
    --mode quick
```

### YAML-конфигурация

См. `configs/sample_benchmark_config.yaml` и `docs/YAML_DESIGN.md`.

## 🔧 Self-Test / Диагностика

Проверка установки всех библиотек и возможности загрузки моделей.

```bash
# Полное тестирование всех фреймворков (CPU + GPU)
python -m whisper_harness.transcribers.self_test

# Быстрая проверка (только импорты, без загрузки моделей)
python -m whisper_harness.transcribers.self_test --quick

# Тестирование конкретного фреймворка
python -m whisper_harness.transcribers.self_test --framework faster-whisper
python -m whisper_harness.transcribers.self_test --framework whisper.cpp
python -m whisper_harness.transcribers.self_test --framework transcribe.cpp
python -m whisper_harness.transcribers.self_test --framework huggingface

# Только GPU или только CPU
python -m whisper_harness.transcribers.self_test --gpu-only
python -m whisper_harness.transcribers.self_test --cpu-only

# Экспорт результатов в JSON
python -m whisper_harness.transcribers.self_test --output selftest_results.json
```

**Что проверяется:**
1. ✅ Наличие установленных библиотек (faster_whisper, pywhispercpp, transformers, onnxruntime и т.д.)
2. ✅ Возможность загрузки минимальной модели (tiny) для каждого фреймворка
3. ✅ Работа на CPU и GPU (если доступно)
4. ✅ Поддержка GPU offloading (полный и частичный через `--gpu-layers`)
5. ✅ Базовая транскрипция тестового аудио

**Интерпретация результатов:**
- ✅ **PASSED** — библиотека установлена и модель загружается успешно
- ❌ **FAILED** — ошибка: требуется установка или исправление конфигурации
- ⚠️ **SKIPPED** — пропущено (CUDA недоступен, quick mode и т.д.)
- ⚡ **WARNING** — работает, но есть ограничения (нет GPU, используется fallback)

**GPU Offloading:**
- whisper.cpp и transcribe.cpp поддерживают partial/full offloading через параметр `gpu_layers`
- `gpu_layers=None` + `use_gpu=True` → полный offloading (все слои на GPU)
- `gpu_layers=N` → частичный offloading (N слоёв на GPU, остальные на CPU)
- Self-test автоматически проверяет оба режима при наличии GPU

## Запуск тестов

```bash
pytest
```

## Рекомендации для GTX 1050 2 ГБ

| Модель   | Квантизация  | VRAM   | RTF  | Рекомендация |
|----------|--------------|--------|------|--------------|
| tiny     | int8_float32 | ~200 МБ | 0.1x | ✅ Отлично для быстрых задач |
| base     | int8_float32 | ~350 МБ | 0.15x | ✅ Хороший баланс |
| small    | int8_float32 | ~500 МБ | 0.3x | ✅ Рекомендуется для продакшена |
| medium   | int8_float32 | ~850 МБ | 0.5x | ✅ Работает, но медленно |
| large-v3 | int8_float32 | ~2.5 ГБ | ❌ OOM | ❌ Не рекомендуется |

**Оптимальная конфигурация для продакшена:**

```bash
python -m whisper_harness.cli \
    --audio test.wav \
    --model-type fast_whisper \
    --model-id small \
    --compute-type int8_float32 \
    --beam-size 1 \
    --language ru
```

## Лицензия

MIT License
