# Модули

## db_explorer.py — автоопределение структуры SQLite

**Ключевые типы:**
- `DBStructure` — результат: tables[], users_table, messages_table, колонки
- `TableInfo`, `Column` — метаданные таблиц

**Ключевые функции:**
- `explore_database(path)` — главная точка входа, возвращает DBStructure
- `_classify_tables(structure)` — эвристика поиска таблиц users/messages
- `_detect_sqlite(path)` — проверка заголовка SQLite

**Эвристика users:** есть колонка `tg_id`/`user_id`/`id` + `username`/`name`
**Эвристика messages:** есть колонка `text`/`comment`/`message` + `user`/`user_id`/`author`

**Исправлено по ревью:** удалён `col_types` (dead code, строка 111).

---

## anonymizer.py — анонимизация user_XXXXXX

**Ключевой тип:** `Anonymizer` (dataclass)

**Методы:**
- `anonymize(real_id) → str` — user_000001, идемпотентный
- `deanonymize(anon_id) → int|None`
- `build_from_database(db_path, structure)` — читает users_table
- `save_mapping(dir) / load_mapping(dir)` — JSON persistence
- `delete_mapping(dir)` — очистка

**Исправлено по ревью:**
- `build_from_database`: добавлен `ORDER BY` для детерминизма (строка 108-109)
- `load_mapping`: counter = `data.get("counter", 0)` — просто и надёжно (строка 75)

---

## preprocessor.py — фильтрация и предобработка

**Ключевые типы:** `CommentRecord`, `UserComments`

**Ключевые функции:**
- `load_user_comments(db_path, structure, anonymizer, min_comments, min_comment_length) → list[UserComments]`
- `limit_comments(comments, max_comments, max_input_tokens)` — отсечение по кол-ву и токенам
- `_is_quote(text) → bool` — детекция цитат (>, «», @reply)
- `_extract_own_text(text) → str` — отделение своего текста от цитируемого
- `_is_political(text) → bool` — поиск 34 политических ключевых слов

**Порядок фильтрации:**
1. Удаление пустых сообщений
2. Дедупликация по тексту (case-insensitive)
3. Удаление коротких (< min_comment_length)
4. Пользователи с < min_comments отсеиваются
5. Выделение своего текста из цитат
6. Фильтр не-политических

**Исправлено по ревью:**
- `_extract_own_text` применяется только если `_is_quote=True` (строка 194-197)
- `skipped_political` теперь инкрементируется + `continue` при пустом political_comments (строка 201-203)

---

## model_loader.py — загрузка Qwen3-4B

**Ключевые типы:** `ModelConfig` (dataclass)

**Ключевые функции:**
- `load_model(config) → (model, tokenizer)`
  - 4-bit через `BitsAndBytesConfig` (nf4, float16 compute, double quant)
  - `trust_remote_code=True`
  - Требует CUDA

---

## analyzer.py — промпт, инференс, парсинг

**Константы:** `PROMPT_VERSION = "2.0"`, `SYSTEM_PROMPT`, `_build_analysis_prompt()`

**Ключевые типы:** `AnalysisResult` (raw_response, parsed_json, is_valid, error)

**Ключевые функции:**
- `run_inference(model, tokenizer, comments, max_new_tokens, seed) → AnalysisResult`
- `_extract_json(text) → dict|None` — 3 стратегии: прямой парсинг, очистка markdown, поиск по скобкам
- `split_into_blocks(comments, max_comments_per_block=20, max_chars_per_block=8000) → list[list]`
- `merge_block_results(block_results) → dict` — взвешенное усреднение + дедупликация evidence

**Исправлено по ревью:**
- `_extract_json`: добавлен `logger.debug()` при неудаче (строка 199)
- `merge_block_results`: дедупликация evidence по (comment_id, quote) (строка 255-268)

---

## validator.py — Pydantic-валидация ответа модели

**Pydantic-модели:**
- `EconomicAxis`, `AuthorityAxis`, `ScienceAxis` — пары шкал + confidence
  - Каждая имеет `@model_validator(mode="after") check_sum()` — авто-коррекция до 100
- `Axes`, `IdeologicalSimilarity`, `ProtestRhetoric`, `EvidenceItem`, `Contradiction`
- `ContentAnalysis`, `AnalysisResponse` — верхнеуровневые

**Ключевые функции:**
- `validate_analysis_response(raw_json, comment_ids) → (AnalysisResponse|None, errors)`
- `build_correction_prompt(original_prompt, raw_response, errors) → str`

**Исправлено по ревью:**
- `AxesScores` dead code — удалён
- Silent auto-correction → `logger.warning()` во всех трёх model_validator (строки 31, 56, 80)
- Дублирующая проверка сумм удалена из `validate_analysis_response` (строки 155-164)

---

## storage.py — SQLite (WAL) + JSONL + CSV

**Ключевой тип:** `Storage`

**Таблицы SQLite:**
- `analysis_runs` — метаданные прогонов
- `anonymous_users` — anon_id → real_id + статистика
- `user_statistics` — дубликат статистики
- `political_analysis` — оси (score_a, score_b, confidence)
- `evidence` — цитаты по категориям
- `processing_errors` — ошибки с raw_response

**Методы:**
- `start_run(config) → run_id`, `complete_run(run_id, stats)`
- `save_user_result(...)` — SQLite + JSONL
- `save_error(...)` — SQLite + JSONL
- `is_user_processed(anon_id, run_id) → bool` — для resume
- `save_summary(results) → CSV`
- `close()` — закрытие соединения

**Исправлено по ревью:**
- Хрупкое извлечение score_a/score_b → явный `axis_key_pairs` (строки 198-206)
- CSV: все 8 идеологий вместо 3 (строки 310-317)
- Документирована непотокобезопасность (строки 20-21)

---

## pipeline.py — главный оркестратор

**Ключевые типы:** `PipelineConfig`, `PipelineState`, `Pipeline`

**Шаги `Pipeline.run()`:**
1. `explore_database` → DBStructure
2. `Anonymizer().build_from_database` + `save_mapping`
3. `load_user_comments` → список UserComments
4. `load_model` → model, tokenizer
5. `storage.start_run`
6. Цикл по пользователям: `_process_user` → `_analyze_block` (с retry)
7. `_save_checkpoint` каждые 10 пользователей
8. `storage.complete_run` + `save_summary`

**Retry-логика в `_analyze_block`:**
- `max_retries + 1` попыток
- При ошибке валидации: correction prompt → повторный инференс
- Все попытки исчерпаны → `save_error`

**Поля конфига:** min_comments=20, max_users=None, max_comments_per_user=300,
  max_input_tokens=24000, max_new_tokens=900, seed=42, resume=True,
  max_retries=1, max_comments_per_block=20
