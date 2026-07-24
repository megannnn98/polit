"""Тесты для модуля db_explorer."""

import sqlite3
import tempfile
import os
import pytest
from pathlib import Path

from political_analysis.db_explorer import explore_database, DBStructure


@pytest.fixture
def sample_db():
    """Создаёт временную SQLite-базу для тестов."""
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
            date TEXT,
            FOREIGN KEY(user) REFERENCES users(tg_id),
            FOREIGN KEY(channel) REFERENCES channels(name)
        );
        INSERT INTO users VALUES (1, 'alice');
        INSERT INTO users VALUES (2, 'bob');
        INSERT INTO channels VALUES ('test_channel');
        INSERT INTO messages VALUES (1, 1, 'test_channel', 'Hello world test', '2025-01-01');
        INSERT INTO messages VALUES (2, 2, 'test_channel', 'Another message here', '2025-01-02');
    """)
    conn.close()
    yield path
    os.unlink(path)


def test_explore_database_structure(sample_db):
    """Проверяет, что explore_database корректно определяет структуру."""
    structure = explore_database(sample_db)
    assert isinstance(structure, DBStructure)
    assert len(structure.tables) >= 3
    table_names = {t.name for t in structure.tables}
    assert "users" in table_names
    assert "messages" in table_names
    assert "channels" in table_names


def test_explore_database_users_table(sample_db):
    """Проверяет определение таблицы пользователей."""
    structure = explore_database(sample_db)
    assert structure.users_table == "users"
    assert structure.users_id_column == "tg_id"
    assert structure.users_name_column == "username"


def test_explore_database_messages_table(sample_db):
    """Проверяет определение таблицы сообщений."""
    structure = explore_database(sample_db)
    assert structure.messages_table == "messages"
    assert structure.messages_text_column == "text"
    assert structure.messages_user_column == "user"
    assert structure.messages_date_column == "date"


def test_explore_database_not_found():
    """Проверяет ошибку при отсутствии файла."""
    with pytest.raises(FileNotFoundError):
        explore_database("/nonexistent/path/db.sqlite")


def test_explore_database_not_sqlite():
    """Проверяет ошибку для не-SQLite файла."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    with open(path, "w") as f:
        f.write("not a database")
    with pytest.raises(ValueError):
        explore_database(path)
    os.unlink(path)


def test_explore_database_summary(sample_db):
    """Проверяет генерацию сводки."""
    structure = explore_database(sample_db)
    summary = structure.summary()
    assert "users" in summary
    assert "messages" in summary
    assert "Channels" in summary or "channels" in summary.lower()
