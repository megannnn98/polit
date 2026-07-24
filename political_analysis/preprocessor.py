"""Предварительная обработка комментариев перед анализом."""

from __future__ import annotations

import re
import sqlite3
import logging
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

logger = logging.getLogger(__name__)

QUOTE_PATTERNS = [
    re.compile(r"^>.*$", re.MULTILINE),
    re.compile(r"^«.*»$", re.MULTILINE),
    re.compile(r'""".*?"', re.MULTILINE | re.DOTALL),
]

POLITICAL_KEYWORDS = [
    "политик", "власть", "государств", "правительств", "парти",
    "выбор", "демократ", "свобод", "равенств", "справедлив",
    "капитал", "социализ", "коммуниз", "либерал", "консерв",
    "реформ", "революц", "протест", "митинг", "закон",
    "налог", "бюджет", "инфляц", "безработиц", "профсоюз",
    "олигарх", "бюрократ", "империализ", "интернационал",
    "класс", "буржу", "пролетари", "эксплуатац", "собственность",
    "privatiz", "nationaliz", "милитариз", "санкц", "эмиграц",
    "цензур", "репресси", "диктатур", "тоталитар", "фашизм",
    "нацизм", "антифашизм", "анархи", "феминизм", "гendер",
    "environment", "климат", "экологи", "мигрант",
    "Маркс", "Ленин", "Энгельс", "Троцкий", "Сталин",
    "Путин", "Зеленский", "США", "Европ", "Росси",
    "Украин", "НАТО", "СССР", "советск",
]


@dataclass
class CommentRecord:
    """Один обработанный комментарий."""
    original_id: int
    text: str
    date: str
    channel: str
    is_quote: bool = False
    quote_author: str | None = None
    original_length: int = 0


@dataclass
class UserComments:
    """Коллекция комментариев одного пользователя."""
    anonymous_id: str
    real_id: int
    username: str | None = None
    total_comments: int = 0
    unique_comments: int = 0
    political_comments: int = 0
    comments: list[CommentRecord] = field(default_factory=list)


def _is_quote(text: str) -> bool:
    """Определяет, является ли сообщение цитатой."""
    stripped = text.strip()
    if stripped.startswith(">"):
        return True
    if stripped.startswith("«") and stripped.endswith("»"):
        return True
    if len(stripped) > 4 and stripped.startswith('"') and stripped.endswith('"'):
        return True
    if stripped.startswith("@") and "\n" in stripped:
        return True
    return False


def _extract_own_text(text: str) -> str:
    """Пытается отделить собственный текст от цитат."""
    lines = text.split("\n")
    own_lines = []
    in_quote = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">") or stripped.startswith("«"):
            in_quote = True
            continue
        if in_quote and not stripped:
            in_quote = False
            continue
        if not in_quote:
            own_lines.append(line)
    result = "\n".join(own_lines).strip()
    if not result:
        return text
    return result


def _is_political(text: str) -> bool:
    """Проверяет, содержит ли сообщение политическое содержание."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in POLITICAL_KEYWORDS)


def load_user_comments(
    db_path: str | Path,
    structure: "DBStructure",  # noqa: F821
    anonymizer: "Anonymizer",  # noqa: F821
    min_comments: int = 20,
    min_comment_length: int = 20,
) -> list[UserComments]:
    """Загружает и предварительно обрабатывает комментарии пользователей.

    Args:
        db_path: путь к базе данных
        structure: структура БД
        anonymizer: анонимизатор
        min_comments: минимальное количество комментариев
        min_comment_length: минимальная длина комментария

    Returns:
        список UserComments для пользователей с достаточным числом комментариев
    """
    if not structure.messages_table or not structure.users_table:
        raise ValueError("Messages or users table not identified")

    conn = sqlite3.connect(
        f"file:{quote(str(Path(db_path).resolve()))}?mode=ro", uri=True
    )
    try:
        query = f"""
            SELECT m.{structure.messages_id_column or 'rowid'},
                   m.{structure.messages_user_column},
                   m.{structure.messages_text_column},
                   COALESCE(m.{structure.messages_date_column}, ''),
                   m.channel
            FROM {structure.messages_table} m
            ORDER BY m.{structure.messages_user_column},
                     m.{structure.messages_date_column}
        """
        cur = conn.execute(query)
        rows = cur.fetchall()
    finally:
        conn.close()

    logger.info("Loaded %d total messages from database", len(rows))

    user_data: dict[int, list[CommentRecord]] = {}
    for row in rows:
        msg_id, user_id, text, date, channel = row
        if text is None:
            continue
        if not isinstance(text, str):
            text = str(text)
        user_data.setdefault(user_id, []).append(
            CommentRecord(
                original_id=msg_id,
                text=text,
                date=date or "",
                channel=channel or "",
                original_length=len(text),
            )
        )

    logger.info("Found %d unique users", len(user_data))

    results: list[UserComments] = []
    skipped_short = 0
    skipped_political = 0

    for real_id, raw_comments in user_data.items():
        # Remove empty messages
        comments = [c for c in raw_comments if c.text.strip()]
        if not comments:
            skipped_short += 1
            continue

        # Deduplicate by text
        seen_texts: set[str] = set()
        unique_comments: list[CommentRecord] = []
        for c in comments:
            normalized = c.text.strip().lower()
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_comments.append(c)
        comments = unique_comments

        # Remove too short
        comments = [c for c in comments if len(c.text.strip()) >= min_comment_length]
        if not comments:
            skipped_short += 1
            continue

        # Classify quotes and extract own text
        for c in comments:
            if _is_quote(c.text):
                c.is_quote = True
                c.text = _extract_own_text(c.text)

        # Filter non-political
        political_comments = [c for c in comments if _is_political(c.text)]
        if len(political_comments) == 0:
            skipped_political += 1
            continue

        # Check min_comments AFTER political filter
        if len(political_comments) < min_comments:
            skipped_short += 1
            continue

        anon_id = anonymizer.anonymize(real_id)
        uc = UserComments(
            anonymous_id=anon_id,
            real_id=real_id,
            total_comments=len(raw_comments),
            unique_comments=len(comments),
            political_comments=len(political_comments),
            comments=political_comments,
        )
        results.append(uc)

    logger.info(
        "Processed users: %d with sufficient comments, "
        "%d skipped (short/no political content)",
        len(results),
        skipped_short + skipped_political,
    )
    return results


def limit_comments(
    comments: UserComments,
    max_comments: int = 300,
    max_input_tokens: int = 24000,
) -> UserComments:
    """Ограничивает количество комментариев для пользователя.

    Args:
        comments: комментарии пользователя
        max_comments: максимальное число комментариев
        max_input_tokens: максимальное число токенов на вход (~4 символа/токен)

    Returns:
        UserComments с ограниченным списком
    """
    max_chars = max_input_tokens * 4

    # Sort by date (newest first)
    comments.comments.sort(key=lambda c: c.date, reverse=True)

    limited: list[CommentRecord] = []
    total_chars = 0
    for c in comments.comments:
        if len(limited) >= max_comments:
            break
        if total_chars + len(c.text) > max_chars:
            break
        limited.append(c)
        total_chars += len(c.text)

    comments.comments = limited
    return comments
