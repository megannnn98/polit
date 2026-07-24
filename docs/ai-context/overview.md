# polit — Анализ политического содержания комментариев

## Назначение

Анонимный анализ политических позиций, выраженных в текстах Telegram-каналов.
Модель: Qwen3-4B-Instruct (4-битная квантизация через BitsAndBytes).
Целевая платформа: Kaggle GPU (T4/P100).

## Архитектура — 8 шагов пайплайна

```
app.db (SQLite, исходная БД Telegram)
  │
  ├─[1] db_explorer     → автоопределение структуры (таблицы users/messages)
  ├─[2] anonymizer      → user_000001..N, mapping.json
  ├─[3] preprocessor    → фильтрация: длина, дубликаты, не-политические, цитаты
  ├─[4] model_loader    → Qwen3-4B, 4-bit, CUDA
  ├─[5] analyzer        → промпт → инференс → парсинг JSON + merge блоков
  ├─[6] validator       → Pydantic-валидация + авто-коррекция сумм шкал
  ├─[7] storage         → SQLite (WAL) + JSONL + CSV
  └─[8] pipeline        → оркестрация: resume, retry, чекпоинты
       ↓
  output_dir/
    analysis.db        (SQLite: analysis_runs, anonymous_users, political_analysis, evidence, errors)
    results.jsonl      (построчные результаты)
    errors.jsonl       (ошибки)
    summary.csv        (сводка)
    anonymization_mapping.json
```

## Ключевые инварианты

- **Анонимность:** mapping-файл не содержит имён — только real_id → user_NNNNNN
- **Шкалы:** left+right=100, authoritarian+libertarian=100, scientific+anti_scientific=100
- **Исходная БД:** только чтение, не изменяется
- **Resume:** через `is_user_processed(anon_id, run_id)` в SQLite

## ТЗ — обязательные поля результата

- `content_analysis.axes.{economic,authority,science}` — пары значений + confidence
- `content_analysis.ideological_similarity` — 8 независимых шкал 0-100
- `content_analysis.protest_rhetoric` — score + confidence
- `evidence` — подтверждающие цитаты с comment_id
- `contradictions` — внутренние противоречия
- `unknown_topics` — темы с недостатком данных
- `overall_confidence` — 0.0-1.0
- `insufficient_data` — boolean
- `disclaimer` — в JSONL-записи
