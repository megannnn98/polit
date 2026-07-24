"""Хранение результатов анализа в SQLite и JSONL."""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Storage:
    """Управляет хранением результатов анализа.

    NOT thread-safe: uses a single SQLite connection.
    Designed for single-threaded pipeline usage.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / "analysis.db"
        self.jsonl_path = self.output_dir / "results.jsonl"
        self.errors_path = self.output_dir / "errors.jsonl"
        self.config_path = self.output_dir / "run_config.json"
        self.summary_path = self.output_dir / "summary.csv"
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                model_id TEXT,
                prompt_version TEXT,
                seed INTEGER,
                total_users INTEGER DEFAULT 0,
                processed_users INTEGER DEFAULT 0,
                skipped_users INTEGER DEFAULT 0,
                error_users INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS anonymous_users (
                anon_id TEXT PRIMARY KEY,
                real_id INTEGER UNIQUE,
                total_comments INTEGER DEFAULT 0,
                unique_comments INTEGER DEFAULT 0,
                political_comments INTEGER DEFAULT 0,
                analyzed_comments INTEGER DEFAULT 0,
                processed_at TEXT,
                run_id TEXT
            );

            CREATE TABLE IF NOT EXISTS user_statistics (
                anon_id TEXT PRIMARY KEY,
                total_comments INTEGER DEFAULT 0,
                unique_comments INTEGER DEFAULT 0,
                political_comments INTEGER DEFAULT 0,
                analyzed_comments INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS political_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anon_id TEXT NOT NULL,
                run_id TEXT,
                axis_name TEXT NOT NULL,
                score_a INTEGER,
                score_b INTEGER,
                confidence REAL,
                FOREIGN KEY (anon_id) REFERENCES anonymous_users(anon_id)
            );

            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anon_id TEXT NOT NULL,
                run_id TEXT,
                category TEXT NOT NULL,
                comment_id INTEGER,
                quote TEXT,
                FOREIGN KEY (anon_id) REFERENCES anonymous_users(anon_id)
            );

            CREATE TABLE IF NOT EXISTS processing_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anon_id TEXT,
                error_type TEXT,
                error_message TEXT,
                raw_response TEXT,
                attempt INTEGER DEFAULT 1,
                run_id TEXT,
                created_at TEXT
            );
        """)
        conn.commit()

    def start_run(self, config: dict) -> str:
        """Начинает новый прогон анализа."""
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO analysis_runs (run_id, started_at, model_id, prompt_version, seed)
               VALUES (?, ?, ?, ?, ?)""",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                config.get("model_id"),
                config.get("prompt_version"),
                config.get("seed"),
            ),
        )
        conn.commit()
        logger.info("Started analysis run: %s", run_id)
        return run_id

    def complete_run(self, run_id: str, stats: dict) -> None:
        """Завершает прогон анализа."""
        conn = self._get_conn()
        conn.execute(
            """UPDATE analysis_runs
               SET completed_at = ?, total_users = ?, processed_users = ?,
                   skipped_users = ?, error_users = ?
               WHERE run_id = ?""",
            (
                datetime.now(timezone.utc).isoformat(),
                stats.get("total_users", 0),
                stats.get("processed_users", 0),
                stats.get("skipped_users", 0),
                stats.get("error_users", 0),
                run_id,
            ),
        )
        conn.commit()

    def save_user_result(
        self,
        anon_id: str,
        real_id: int,
        statistics: dict,
        analysis: dict,
        evidence: dict,
        contradictions: list,
        unknown_topics: list,
        overall_confidence: float,
        insufficient_data: bool,
        run_id: str,
    ) -> None:
        """Сохраняет результат анализа пользователя."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()

        # Anonymous user record
        conn.execute(
            """INSERT OR REPLACE INTO anonymous_users
               (anon_id, real_id, total_comments, unique_comments,
                political_comments, analyzed_comments, processed_at, run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                anon_id,
                real_id,
                statistics.get("total_comments", 0),
                statistics.get("unique_comments", 0),
                statistics.get("political_comments", 0),
                statistics.get("analyzed_comments", 0),
                now,
                run_id,
            ),
        )

        # User statistics
        conn.execute(
            """INSERT OR REPLACE INTO user_statistics
               (anon_id, total_comments, unique_comments, political_comments, analyzed_comments)
               VALUES (?, ?, ?, ?, ?)""",
            (
                anon_id,
                statistics.get("total_comments", 0),
                statistics.get("unique_comments", 0),
                statistics.get("political_comments", 0),
                statistics.get("analyzed_comments", 0),
            ),
        )

        # Political analysis axes
        axes = analysis.get("content_analysis", {}).get("axes", {})
        axis_key_pairs = {
            "economic": ("left", "right"),
            "authority": ("authoritarian", "libertarian"),
            "science": ("scientific", "anti_scientific"),
        }
        for axis_name, (key_a, key_b) in axis_key_pairs.items():
            axis_data = axes.get(axis_name, {})
            if axis_data:
                score_a = axis_data.get(key_a)
                score_b = axis_data.get(key_b)
                confidence = axis_data.get("confidence")
                conn.execute(
                    """INSERT INTO political_analysis
                       (anon_id, run_id, axis_name, score_a, score_b, confidence)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (anon_id, run_id, axis_name, score_a, score_b, confidence),
                )

        # Evidence
        for category, items in evidence.items():
            for item in items:
                conn.execute(
                    """INSERT INTO evidence (anon_id, run_id, category, comment_id, quote)
                       VALUES (?, ?, ?, ?, ?)""",
                    (anon_id, run_id, category, item.get("comment_id"), item.get("quote", "")),
                )

        conn.commit()

        # JSONL
        record = {
            "anonymous_user_id": anon_id,
            "statistics": statistics,
            "content_analysis": analysis.get("content_analysis"),
            "evidence": evidence,
            "contradictions": contradictions,
            "unknown_topics": unknown_topics,
            "overall_confidence": overall_confidence,
            "insufficient_data": insufficient_data,
            "partial_response": analysis.get("partial_response", False),
            "disclaimer": "Результат описывает политическое содержание предоставленных сообщений и не является достоверным определением личных убеждений автора.",
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_error(
        self,
        anon_id: str | None,
        error_type: str,
        error_message: str,
        raw_response: str = "",
        attempt: int = 1,
        run_id: str = "",
    ) -> None:
        """Сохраняет ошибку обработки."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO processing_errors
               (anon_id, error_type, error_message, raw_response, attempt, run_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                anon_id,
                error_type,
                error_message,
                raw_response[:5000],
                attempt,
                run_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

        # Errors JSONL
        record = {
            "anonymous_user_id": anon_id,
            "error_type": error_type,
            "error_message": error_message,
            "raw_response": raw_response[:2000],
            "attempt": attempt,
        }
        with open(self.errors_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def is_user_processed(self, anon_id: str, run_id: str) -> bool:
        """Проверяет, был ли пользователь уже обработан."""
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT 1 FROM anonymous_users WHERE anon_id = ? AND run_id = ?",
            (anon_id, run_id),
        )
        return cur.fetchone() is not None

    def is_user_processed_in_any_run(self, anon_id: str) -> bool:
        """Проверяет, был ли пользователь обработан в любом предыдущем прогона."""
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT 1 FROM anonymous_users WHERE anon_id = ?",
            (anon_id,),
        )
        return cur.fetchone() is not None

    def get_last_run_id(self) -> str | None:
        """Возвращает run_id последнего прогона."""
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT run_id FROM analysis_runs ORDER BY started_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        return row[0] if row else None

    def save_config(self, config: dict) -> None:
        """Сохраняет конфигурацию прогона."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def save_summary(self, results: list[dict]) -> None:
        """Сохраняет CSV-сводку."""
        with open(self.summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "anonymous_user_id",
                "total_comments",
                "political_comments",
                "insufficient_data",
                "overall_confidence",
                "economic_left",
                "economic_right",
                "authority_authoritarian",
                "authority_libertarian",
                "science_scientific",
                "science_anti_scientific",
                "liberalism",
                "conservatism",
                "social_democracy",
                "marxism",
                "anarchism",
                "nationalism",
                "fascism_nazism",
                "political_indifference",
            ])
            for r in results:
                stats = r.get("statistics", {})
                analysis = r.get("content_analysis", {})
                axes = analysis.get("axes", {})
                econ = axes.get("economic", {})
                auth = axes.get("authority", {})
                sci = axes.get("science", {})
                ideol = analysis.get("ideological_similarity", {})
                writer.writerow([
                    r.get("anonymous_user_id", ""),
                    stats.get("total_comments", 0),
                    stats.get("political_comments", 0),
                    r.get("insufficient_data", False),
                    r.get("overall_confidence", 0),
                    econ.get("left"),
                    econ.get("right"),
                    auth.get("authoritarian"),
                    auth.get("libertarian"),
                    sci.get("scientific"),
                    sci.get("anti_scientific"),
                    ideol.get("liberalism"),
                    ideol.get("conservatism"),
                    ideol.get("social_democracy"),
                    ideol.get("marxism"),
                    ideol.get("anarchism"),
                    ideol.get("nationalism"),
                    ideol.get("fascism_nazism"),
                    ideol.get("political_indifference"),
                ])

    def close(self) -> None:
        """Закрывает соединение с БД."""
        if self._conn:
            self._conn.close()
            self._conn = None
