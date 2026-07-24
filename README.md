# Анализ политического содержания комментариев

Анонимный анализ политических позиций, выраженных в текстах Telegram-каналов, с использованием языковой модели Qwen3-4B-Instruct.

## Дисклеймер

**Результат описывает политическое содержание предоставленных сообщений и не является достоверным определением личных убеждений автора.**

## Возможности

- Автоматическое определение структуры SQLite-базы данных
- Анонимизация пользователей (user_000001, user_000002, ...)
- Предварительная обработка: удаление дубликатов, коротких и не-политических сообщений
- Анализ с использованием Qwen3-4B-Instruct в 4-битной квантизации
- Проверка результатов через Pydantic
- Возобновление обработки после перезапуска
- Экспорт в SQLite, JSONL и CSV

## Установка

```bash
pip install transformers accelerate bitsandbytes torch pydantic
```

## Запуск

### Kaggle Notebook

1. Загрузите базу данных `app.db` как Kaggle Dataset
2. Откройте `notebook.ipynb`
3. В ячейке конфигурации укажите путь к базе данных
4. Запустите все ячейки

### Локально

```bash
# Через модуль
python -m political_analysis.pipeline --database app.db --output-dir ./results

# После установки
pip install -e .
political-analysis --database app.db --output-dir ./results

# Список аргументов
python -m political_analysis.pipeline --help
```

## Конфигурация

| Параметр | По умолчанию | Описание |
|---|---|---|
| `database_path` | `app.db` | Путь к базе данных SQLite |
| `model_id` | `Qwen/Qwen3-4B-Instruct-2507` | ID модели HuggingFace |
| `min_comments` | `20` | Минимальное число комментариев для анализа |
| `min_comment_length` | `20` | Минимальная длина комментария в символах |
| `max_users` | `null` | Ограничение на N самых активных пользователей |
| `max_comments_per_user` | `300` | Максимальное число комментариев на пользователя |
| `max_input_tokens` | `24000` | Максимальное число токенов на вход |
| `max_new_tokens` | `900` | Максимальная длина ответа модели |
| `seed` | `42` | Seed для воспроизводимости |
| `resume` | `true` | Возобновление после перезапуска |

## Структура результатов

### analysis.db

- `analysis_runs` — информация о каждом прогона
- `anonymous_users` — анонимные пользователи
- `user_statistics` — статистика комментариев
- `political_analysis` — результаты по осям
- `evidence` — подтверждающие цитаты
- `processing_errors` — ошибки обработки

### results.jsonl

JSON-записи с полным результатом анализа для каждого пользователя.

### summary.csv

Сводная таблица: анонимный ID, количество комментариев, оценки.

### run_config.json

Конфигурация прогона.

## Шкалы анализа

### Оси

- **Экономическая:** left ↔ right (сумма = 100)
- **Власть:** authoritarian ↔ libertarian (сумма = 100)
- **Наука:** scientific ↔ anti_scientific (сумма = 100)

### Идеологические сходства (независимые, 0-100)

- liberalism
- conservatism
- social_democracy
- marxism
- anarchism
- nationalism
- fascism_nazism
- political_indifference

## Тесты

```bash
cd /home/b/Documents/polit
python -m pytest tests/ -v
```

## Конфиденциальность

- Имена пользователей заменяются на `user_XXXXXX`
- Модель не получает идентификаторы
- Файл маппинга сохраняется отдельно и может быть удалён

## Структура проекта

```
polit/
├── app.db                          # Исходная база данных
├── analysis.db                     # База результатов
├── results.jsonl                   # Результаты в JSONL
├── errors.jsonl                    # Ошибки
├── run_config.json                 # Конфигурация прогона
├── summary.csv                     # Сводная таблица
├── README.md                       # Этот файл
├── notebook.ipynb                  # Kaggle Notebook
├── political_analysis/             # Основной модуль
│   ├── __init__.py
│   ├── db_explorer.py              # Исследование структуры БД
│   ├── anonymizer.py               # Анонимизация
│   ├── preprocessor.py             # Предобработка
│   ├── model_loader.py             # Загрузка модели
│   ├── analyzer.py                 # Анализ
│   ├── validator.py                # Валидация
│   ├── storage.py                  # Хранение
│   └── pipeline.py                 # Главный конвейер
└── tests/                          # Тесты
    ├── __init__.py
    ├── test_db_explorer.py
    ├── test_anonymizer.py
    ├── test_preprocessor.py
    ├── test_validator.py
    └── test_pipeline.py
```
