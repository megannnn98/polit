# Итоги code review (2026-07-24)

Все 14 находок исправлены.

## Критические (2)

| # | Файл | Описание | Статус |
|---|------|----------|--------|
| 1 | anonymizer.py:108 | Недетерминированная анонимизация: нет ORDER BY | ✅ Добавлен `ORDER BY` |
| 2 | storage.py:198 | Хрупкое извлечение score_a/score_b через keys()[0] | ✅ Явный `axis_key_pairs` |

## Высокие (5)

| # | Файл | Описание | Статус |
|---|------|----------|--------|
| 3 | validator.py:31-82 | Silent auto-correction сумм шкал | ✅ `logger.warning()` |
| 4 | preprocessor.py:194 | `_extract_own_text` на не-цитатах | ✅ Только при `_is_quote=True` |
| 5 | analyzer.py:255 | Дубликаты evidence при merge блоков | ✅ Дедупликация |
| 6 | preprocessor.py:201 | `skipped_political` всегда 0 | ✅ `continue` при пустом |
| 7 | validator.py:155-164 | Дублирующая проверка сумм | ✅ Удалена |

## Средние (5)

| # | Файл | Описание | Статус |
|---|------|----------|--------|
| 8 | validator.py:14-17 | `AxesScores` dead code | ✅ Удалён |
| 9 | db_explorer.py:111 | `col_types` не используется | ✅ Удалена строка |
| 10 | analyzer.py:199 | `_extract_json` без логгирования | ✅ `logger.debug()` |
| 11 | anonymizer.py:75 | Хрупкое восстановление counter | ✅ `data.get("counter", 0)` |
| 12 | notebook.ipynb:222 | `isInternetEnabled: false` | ✅ `true` |

## Низкие (2)

| # | Файл | Описание | Статус |
|---|------|----------|--------|
| 13 | storage.py:20 | Непотокобезопасность не документирована | ✅ Docstring |
| 14 | storage.py:310 | CSV: только 3/8 идеологий | ✅ Все 8 |

## Новые тесты

- `test_analyzer.py` — 7 тестов (`_extract_json`, `split_into_blocks`, `merge_block_results`)
- `test_preprocessor.py` — +2 теста (`test_extract_own_text_only_for_quotes`, `test_is_political_excludes_non_political`)

## Остаточный риск

`test_load_user_comments_filters` (test_preprocessor.py:152) — тело пустое, нет assert'ов. Не блокирует.
