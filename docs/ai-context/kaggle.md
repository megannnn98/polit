# Kaggle-деплой

## Структура

Проект использует **notebook-подход** (`.ipynb`), не script-подход как в jobb.

Файл: `notebook.ipynb` — 9 ячеек.

## Ключевые поля метаданных

```json
"kaggle": {
  "accelerator": "gpu",
  "dataSources": [
    {
      "sourceType": "kaggleDataset",
      "sourceId": "telegram-political-comments"  // или <owner>/telegram-political-comments
    }
  ],
  "isGpuEnabled": true,
  "isInternetEnabled": true,
  "language": "python"
}
```

## Типичная ошибка

Если `dataSources` пуст или неправильный → Kaggle не монтирует датасет в `/kaggle/input/` → `FileNotFoundError`.

## Жизненный цикл ячейки

1. `pip install transformers accelerate bitsandbytes torch pydantic`
2. `PipelineConfig` — database_path, model_id, seed, output_dir
3. `explore_database` — проверка структуры
4. `Anonymizer.build_from_database` + `load_user_comments`
5. `load_model` (Qwen3-4B, 4-bit)
6. Тест на одном пользователе: `run_inference` + `validate_analysis_response`
7. Пакетная: `Pipeline.run()` — с предварительной инъекцией model/tokenizer/anonymizer
8. Экспорт: список файлов в output_dir
9. Сводка: pandas.read_csv(summary.csv) + статистика

## Важно

- `isInternetEnabled: true` — требуется для pip install в ячейке 1
- `isGpuEnabled: true` — модель требует CUDA для 4-bit квантизации
- Датасет должен быть прикреплён как Kaggle Input (Data → Add Data)
- `resume=True` + `is_user_processed` — защита от перезаписи при перезапуске

## Отличия от jobb-подхода

| | polit | jobb |
|---|---|---|
| Тип kernel | notebook (.ipynb) | script (.py) |
| Конфиг | внутри .ipynb | kernel-metadata.json |
| Запуск | через mimo kaggle-kernel skill | `kaggle kernels push -p kaggle/<name>` |
| Вывод модели | Kaggle Output | скачивание через `kaggle kernels output` |
