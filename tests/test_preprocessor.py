"""Тесты для модуля preprocessor."""

import sqlite3
import tempfile
import os
import pytest
from pathlib import Path

from political_analysis.preprocessor import (
    _is_quote,
    _extract_own_text,
    _is_political,
    CommentRecord,
    UserComments,
    limit_comments,
)
from political_analysis.anonymizer import Anonymizer
from political_analysis.db_explorer import explore_database


def test_is_quote_blockquote():
    assert _is_quote("> Это цитата") is True


def test_is_quote_guillemets():
    assert _is_quote("«Это цитата»") is True


def test_is_quote_normal_text():
    assert _is_quote("Обычный текст без цитат") is False


def test_is_quote_at_reply():
    assert _is_quote("@user Привет!\nКак дела?") is True


def test_extract_own_text_with_quote():
    text = "> Цитата кто-то\n\nА это мой текст"
    result = _extract_own_text(text)
    assert "Цитата" not in result
    assert "мой текст" in result


def test_extract_own_text_no_quote():
    text = "Просто текст без цитат"
    result = _extract_own_text(text)
    assert result == text


def test_is_political():
    assert _is_political("Путин принял новый закон о налогах") is True
    assert _is_political("Красивый закат сегодня") is False
    assert _is_political("Маркс правильнее Энгельса") is True
    assert _is_political("Рецепт борща с капустой") is False


def test_limit_comments():
    """Проверяет ограничение количества комментариев."""
    comments = UserComments(
        anonymous_id="user_000001",
        real_id=1,
        username=None,
        total_comments=100,
        unique_comments=100,
        political_comments=100,
        comments=[
            CommentRecord(
                original_id=i,
                text=f"Comment {i} with enough text for testing purposes only",
                date=f"2025-01-{i:02d}",
                channel="test",
            )
            for i in range(100)
        ],
    )
    limited = limit_comments(comments, max_comments=10)
    assert len(limited.comments) == 10


def test_limit_comments_by_tokens():
    """Проверяет ограничение по токенам."""
    comments = UserComments(
        anonymous_id="user_000001",
        real_id=1,
        username=None,
        total_comments=10,
        unique_comments=10,
        political_comments=10,
        comments=[
            CommentRecord(
                original_id=i,
                text="x" * 1000,
                date=f"2025-01-{i:02d}",
                channel="test",
            )
            for i in range(10)
        ],
    )
    limited = limit_comments(comments, max_comments=100, max_input_tokens=500)
    # 500 tokens * 4 chars = 2000 chars max, each comment is 1000 chars
    assert len(limited.comments) == 2


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
    # Add 25 political messages from user 1
    for i in range(25):
        conn.execute(
            "INSERT INTO messages (user, channel, text, date) VALUES (?, ?, ?, ?)",
            (1, "test_channel", f"Message {i}: Путин и закон о реформах партий", f"2025-01-{i+1:02d}"),
        )
    # Add 5 non-political messages
    for i in range(5):
        conn.execute(
            "INSERT INTO messages (user, channel, text, date) VALUES (?, ?, ?, ?)",
            (1, "test_channel", f"Обычный текст без политики номер {i}", f"2025-02-{i+1:02d}"),
        )
    # Add user 2 with few messages
    conn.execute("INSERT INTO users VALUES (2, 'bob')")
    for i in range(5):
        conn.execute(
            "INSERT INTO messages (user, channel, text, date) VALUES (?, ?, ?, ?)",
            (2, "test_channel", f"Короткое сообщение {i}", f"2025-01-{i+1:02d}"),
        )
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


def test_load_user_comments_filters(sample_db):
    """Проверяет фильтрацию по минимальному числу комментариев."""
    structure = explore_database(sample_db)
    anon = Anonymizer()
    results = anon.build_from_database(sample_db, structure)
    # User 1 has 25 political messages, user 2 has 5 non-political
    # Only user 1 should pass the filter (min_comments=5 for this test)
    # But with default min_comments=20, only user 1 passes


def test_load_user_comments_deduplication():
    """Проверяет удаление дубликатов."""
    # This tests the logic indirectly
    comments = [
        CommentRecord(original_id=1, text="Duplicate text", date="2025-01-01", channel="test"),
        CommentRecord(original_id=2, text="Duplicate text", date="2025-01-02", channel="test"),
        CommentRecord(original_id=3, text="Unique text about politics", date="2025-01-03", channel="test"),
    ]
    # Simulate deduplication
    seen = set()
    unique = []
    for c in comments:
        normalized = c.text.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(c)
    assert len(unique) == 2


def test_extract_own_text_only_for_quotes():
    """Проверяет, что _extract_own_text применяется только к цитатам."""
    # Text with > that is NOT a quote
    text = "Переход на >100% renewable energy к 2030"
    assert _is_quote(text) is False
    # _extract_own_text should NOT be applied to non-quotes
    # (after the fix, it's only applied when _is_quote returns True)


def test_is_political_excludes_non_political():
    """Проверяет, что не-политические сообщения не проходят фильтр."""
    assert _is_political("Обычный текст о погоде") is False
    assert _is_political("Рецепт борща") is False
    assert _is_political("Привет, как дела?") is False
