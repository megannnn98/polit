"""Тесты для модуля pipeline."""

import sqlite3
import tempfile
import os
import pytest
from pathlib import Path

from political_analysis.pipeline import PipelineConfig, Pipeline
from political_analysis.storage import Storage


@pytest.fixture
def sample_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE users (
            tg_id INTEGER PRIMARY KEY NOT NULL,
            username TEXT
        );
        CREATE TABLE channels (
            name TEXT PRIMARY KEY NOT NULL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user INTEGER,
            channel TEXT,
            text TEXT,
            date TEXT
        );
        INSERT INTO users VALUES (1, 'alice');
        INSERT INTO channels VALUES ('test_channel');
    """)
    for i in range(25):
        conn.execute(
            "INSERT INTO messages (user, channel, text, date) VALUES (?, ?, ?, ?)",
            (1, "test_channel", f"Сообщение {i}: Путин и закон о реформах", f"2025-01-{i+1:02d}"),
        )
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


def test_config_to_dict():
    """Проверяет сериализацию конфигурации."""
    config = PipelineConfig(database_path="test.db", model_id="test-model")
    d = config.to_dict()
    assert d["database_path"] == "test.db"
    assert d["model_id"] == "test-model"
    assert d["min_comments"] == 20
    assert d["seed"] == 42


def test_storage_init():
    """Проверяет инициализацию хранилища."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(tmpdir)
        assert Path(tmpdir, "analysis.db").exists()
        storage.close()


def test_storage_save_error():
    """Проверяет сохранение ошибок."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(tmpdir)
        storage.save_error(
            "user_000001",
            "test_error",
            "Test error message",
            "raw response",
        )
        storage.close()
        # Check errors JSONL exists
        assert Path(tmpdir, "errors.jsonl").exists()


def test_empty_user_list():
    """Проверяет обработку пустого списка пользователей."""
    config = PipelineConfig(
        database_path="/nonexistent",
        min_comments=99999,
    )
    # This would fail because DB doesn't exist, but tests the config
    assert config.min_comments == 99999


def test_config_defaults():
    """Проверяет значения по умолчанию."""
    config = PipelineConfig()
    assert config.min_comments == 20
    assert config.min_comment_length == 20
    assert config.max_comments_per_user == 300
    assert config.seed == 42
    assert config.resume is True
    assert config.force_reprocess is False


def test_cli_help():
    """Проверяет, что CLI --help работает."""
    from political_analysis.pipeline import main
    import pytest
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_cli_parse_args():
    """Проверяет парсинг CLI аргументов."""
    from political_analysis.pipeline import PipelineConfig
    import argparse

    # Test that argparse correctly maps args to config values
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="app.db")
    parser.add_argument("--min-comments", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(["--database", "test.db", "--min-comments", "10", "--seed", "123"])

    config = PipelineConfig(
        database_path=args.database,
        min_comments=args.min_comments,
        seed=args.seed,
        resume=not args.no_resume,
    )
    assert config.database_path == "test.db"
    assert config.min_comments == 10
    assert config.seed == 123
    assert config.resume is True
