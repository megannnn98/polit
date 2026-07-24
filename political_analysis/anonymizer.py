"""Анонимизация пользователей: замена реальных ID на user_XXXXXX."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Anonymizer:
    """Хранит соответствие реальных ID анонимным.

    Attributes:
        mapping: dict[реальный_id] = анонимный_id
        reverse: dict[анонимный_id] = реальный_id
        counter: текущий счётчик для генерации ID
    """

    mapping: dict[int, str] = field(default_factory=dict)
    reverse: dict[str, int] = field(default_factory=dict)
    counter: int = 0

    def anonymize(self, real_id: int) -> str:
        """Возвращает анонимный ID для пользователя.

        Если пользователь уже был анонимизирован, возвращает
        предыдущий анонимный ID.
        """
        if real_id in self.mapping:
            return self.mapping[real_id]
        self.counter += 1
        anon_id = f"user_{self.counter:06d}"
        self.mapping[real_id] = anon_id
        self.reverse[anon_id] = real_id
        return anon_id

    def deanonymize(self, anon_id: str) -> int | None:
        """Возвращает реальный ID по анонимному или None."""
        return self.reverse.get(anon_id)

    def get_mapping_path(self, base_dir: str | Path) -> Path:
        """Возвращает путь к файлу соответствия."""
        return Path(base_dir) / "anonymization_mapping.json"

    def save_mapping(self, base_dir: str | Path) -> Path:
        """Сохраняет соответствие в JSON-файл."""
        path = self.get_mapping_path(base_dir)
        data = {
            "mapping": {str(k): v for k, v in self.mapping.items()},
            "reverse": {v: k for k, v in self.mapping.items()},
            "counter": self.counter,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Saved anonymization mapping to %s (%d users)", path, self.counter)
        return path

    @classmethod
    def load_mapping(cls, base_dir: str | Path) -> Anonymizer:
        """Загружает соответствие из JSON-файла."""
        path = cls().get_mapping_path(base_dir)
        if not path.exists():
            logger.warning("No mapping file found at %s", path)
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        anon = cls()
        anon.mapping = {int(k): v for k, v in data["mapping"].items()}
        anon.reverse = {k: int(v) for k, v in data["reverse"].items()}
        anon.counter = data.get("counter", 0)
        logger.info("Loaded anonymization mapping: %d users", len(anon.mapping))
        return anon

    def delete_mapping(self, base_dir: str | Path) -> bool:
        """Удаляет файл соответствия. Возвращает True если удалён."""
        path = self.get_mapping_path(base_dir)
        if path.exists():
            path.unlink()
            logger.info("Deleted mapping file: %s", path)
            return True
        return False

    def build_from_database(
        self,
        db_path: str | Path,
        structure: "DBStructure",  # noqa: F821
    ) -> Anonymizer:
        """Создаёт маппинг из существующей базы данных.

        Args:
            db_path: путь к базе данных
            structure: структура БД (из db_explorer)

        Returns:
            self (для цепочки вызовов)
        """
        if not structure.users_table or not structure.users_id_column:
            raise ValueError("Cannot build mapping: users table not identified")

        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.execute(
                f"SELECT {structure.users_id_column} FROM {structure.users_table} "
                f"ORDER BY {structure.users_id_column}"
            )
            for row in cur.fetchall():
                self.anonymize(row[0])
        finally:
            conn.close()

        logger.info(
            "Built anonymization mapping for %d users from %s",
            len(self.mapping),
            db_path,
        )
        return self
