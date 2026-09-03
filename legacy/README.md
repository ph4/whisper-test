# Legacy v1

Старая одномодульная версия проекта, которая жила в корне репозитория.

## Файлы

- `whisper_benchmark.py` — монолитный бенчмарк-харнесс (1123 LOC, всё в одном файле)
- `test_harness.py` — старые тесты (с хардкоженным `/workspace` в импортах)
- `demo_benchmark.py` — скрипт для генерации демо-результатов

## Что заменили

- `whisper_benchmark.py` → `src/whisper_harness/benchmark.py` + `src/whisper_harness/transcribers/` (модульная архитектура)
- `test_harness.py` → `tests/test_harness.py` (pytest, без хардкоженых путей)
- `demo_benchmark.py` → `examples/demo_benchmark.py`

## Зачем храним

- Референс старого API
- Сравнение с новой архитектурой
- Если кому-то нужен старый одномодульный скрипт для быстрого запуска

## Как запустить (если очень хочется)

```bash
cd legacy
python whisper_benchmark.py --audio test.wav --mode quick
```

Эти файлы больше не поддерживаются. Используйте новый пакет:

```bash
pip install -e .
whisper-bench --help
```
