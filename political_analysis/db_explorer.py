"""Автоматическое определение структуры базы данных SQLite."""

from __future__ import annotations

import sqlite3
import logging
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass
class Column:
    name: str
    type: str
    notnull: bool
    pk: bool
    default: str | None = None


@dataclass
class TableInfo:
    name: str
    columns: list[Column] = field(default_factory=list)
    row_count: int = 0
    foreign_keys: list[dict] = field(default_factory=list)
    indexes: list[str] = field(default_factory=list)


@dataclass
class DBStructure:
    path: str
    tables: list[TableInfo] = field(default_factory=list)
    users_table: str | None = None
    messages_table: str | None = None
    users_id_column: str | None = None
    users_name_column: str | None = None
    messages_user_column: str | None = None
    messages_text_column: str | None = None
    messages_date_column: str | None = None
    messages_id_column: str | None = None
    user_fk_column: str | None = None
    user_fk_target: str | None = None

    def summary(self) -> str:
        lines = [f"Database: {self.path}"]
        for t in self.tables:
            lines.append(f"  {t.name} ({t.row_count} rows):")
            for c in t.columns:
                pk = " PK" if c.pk else ""
                nn = " NOT NULL" if c.notnull else ""
                lines.append(f"    {c.name}: {c.type}{pk}{nn}")
            if t.foreign_keys:
                for fk in t.foreign_keys:
                    lines.append(
                        f"    FK: {fk['from']} -> {fk['table']}.{fk['to']}"
                    )
            if t.indexes:
                lines.append(f"    Indexes: {', '.join(t.indexes)}")
        if self.users_table:
            lines.append(f"  >> Users table: {self.users_table}")
        if self.messages_table:
            lines.append(f"  >> Messages table: {self.messages_table}")
        return "\n".join(lines)


def _detect_sqlite(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(16) == b"SQLite format 3\x00"
    except OSError:
        return False


def _get_table_info(conn: sqlite3.Connection, table_name: str) -> TableInfo:
    info = TableInfo(name=table_name)

    cur = conn.execute(f"PRAGMA table_info({table_name})")
    for row in cur.fetchall():
        info.columns.append(
            Column(
                name=row[1],
                type=row[2],
                notnull=bool(row[3]),
                default=row[4],
                pk=bool(row[5]),
            )
        )

    cur = conn.execute(f"SELECT count(*) FROM {table_name}")
    info.row_count = cur.fetchone()[0]

    cur = conn.execute(f"PRAGMA foreign_key_list({table_name})")
    for row in cur.fetchall():
        info.foreign_keys.append(
            {"id": row[0], "table": row[2], "from": row[3], "to": row[4]}
        )

    cur = conn.execute(f"PRAGMA index_list({table_name})")
    for row in cur.fetchall():
        info.indexes.append(row[1])

    return info


def _classify_tables(structure: DBStructure) -> None:
    """Определяет таблицы пользователей и сообщений по эвристике."""
    for table in structure.tables:
        col_names = {c.name.lower() for c in table.columns}

        is_users = False
        if "tg_id" in col_names or "user_id" in col_names or "id" in col_names:
            if "username" in col_names or "name" in col_names:
                is_users = True
        if table.row_count > 0 and table.row_count < 50000:
            has_text = "text" in col_names or "comment" in col_names or "message" in col_names
            if not has_text and is_users:
                # Small table with id + username = likely users
                pass
            elif not is_users:
                continue

        if is_users:
            structure.users_table = table.name
            for c in table.columns:
                cn = c.name.lower()
                if cn in ("tg_id", "user_id", "id") and c.pk:
                    structure.users_id_column = c.name
                elif cn in ("username", "user_name", "name", "login"):
                    structure.users_name_column = c.name

    for table in structure.tables:
        if table.name == structure.users_table:
            continue
        col_names = {c.name.lower() for c in table.columns}
        has_text = "text" in col_names or "comment" in col_names or "message" in col_names
        has_user_ref = any(
            c.name.lower() in ("user", "user_id", "author", "from_user")
            for c in table.columns
        )
        if has_text and has_user_ref:
            structure.messages_table = table.name
            for c in table.columns:
                cn = c.name.lower()
                if cn in ("text", "comment", "message", "content"):
                    structure.messages_text_column = c.name
                elif cn in ("user", "user_id", "author", "from_user"):
                    structure.messages_user_column = c.name
                    structure.user_fk_column = c.name
                elif cn in ("date", "created_at", "timestamp", "time"):
                    structure.messages_date_column = c.name
                elif cn == "id":
                    structure.messages_id_column = c.name
            break

    # Resolve FK target
    if structure.messages_table and structure.users_table and structure.user_fk_column:
        for table in structure.tables:
            if table.name == structure.messages_table:
                for fk in table.foreign_keys:
                    if fk["from"] == structure.user_fk_column:
                        structure.user_fk_target = fk["to"]
                        break
        if not structure.user_fk_target:
            structure.user_fk_target = structure.users_id_column


def explore_database(db_path: str | Path) -> DBStructure:
    """Изучает структуру SQLite-базы данных.

    Определяет тип БД, список таблиц, связи между ними,
    автоматически находит таблицы пользователей и сообщений.
    Не изменяет исходную базу.

    Args:
        db_path: путь к файлу базы данных

    Returns:
        DBStructure с полной информацией о структуре

    Raises:
        FileNotFoundError: если файл не найден
        ValueError: если файл не является SQLite-базой
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    if not _detect_sqlite(path):
        raise ValueError(f"Not a SQLite database: {path}")

    structure = DBStructure(path=str(path))

    # Read-only: source DB may live on a read-only mount (e.g. Kaggle input),
    # where sqlite3's default rwc open mode fails with "unable to open database file"
    conn = sqlite3.connect(f"file:{quote(str(path.resolve()))}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT name, type FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        for row in cur.fetchall():
            table_info = _get_table_info(conn, row[0])
            structure.tables.append(table_info)
            logger.info("Found table: %s (%d rows)", row[0], table_info.row_count)

        _classify_tables(structure)
        logger.info("Users table: %s", structure.users_table)
        logger.info("Messages table: %s", structure.messages_table)
    finally:
        conn.close()

    return structure
