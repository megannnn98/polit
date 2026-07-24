"""Главный цикл обработки: анонимизация, анализ, хранение."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .db_explorer import explore_database
from .anonymizer import Anonymizer
from .preprocessor import load_user_comments, limit_comments, UserComments
from .analyzer import (
    run_inference,
    split_into_blocks,
    merge_block_results,
    AnalysisResult,
)
from .validator import validate_analysis_response, build_correction_prompt
from .storage import Storage

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    database_path: str = "app.db"
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    min_comments: int = 20
    min_comment_length: int = 20
    max_comments_per_user: int = 300
    batch_size: int = 1
    max_input_tokens: int = 24000
    max_new_tokens: int = 1500
    seed: int = 42
    resume: bool = True
    force_reprocess: bool = False
    output_dir: str = "."
    max_retries: int = 1
    max_comments_per_block: int = 50
    stability_runs: int = 1
    stability_threshold: float = 15.0

    def to_dict(self) -> dict:
        return {
            "database_path": self.database_path,
            "model_id": self.model_id,
            "min_comments": self.min_comments,
            "min_comment_length": self.min_comment_length,
            "max_comments_per_user": self.max_comments_per_user,
            "batch_size": self.batch_size,
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "resume": self.resume,
            "force_reprocess": self.force_reprocess,
            "output_dir": self.output_dir,
            "max_retries": self.max_retries,
            "max_comments_per_block": self.max_comments_per_block,
            "stability_runs": self.stability_runs,
            "stability_threshold": self.stability_threshold,
        }


@dataclass
class PipelineState:
    total_users: int = 0
    processed_users: int = 0
    skipped_users: int = 0
    error_users: int = 0
    start_time: float = 0.0
    results: list[dict] = field(default_factory=list)


class Pipeline:
    """Главный конвейер анализа."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.storage = Storage(config.output_dir)
        self.anonymizer: Anonymizer | None = None
        self.model = None
        self.tokenizer = None
        self.run_id: str = ""
        self.state = PipelineState()

    def run(self) -> PipelineState:
        """Запускает полный конвейер анализа."""
        self.state.start_time = time.time()

        # Log configuration
        logger.info("=" * 60)
        logger.info("Political Analysis Pipeline")
        logger.info("Model: %s", self.config.model_id)
        logger.info("Seed: %d", self.config.seed)
        logger.info("Min comments: %d", self.config.min_comments)
        logger.info("=" * 60)

        # Step 1: Explore database
        logger.info("Step 1: Exploring database structure...")
        structure = explore_database(self.config.database_path)
        logger.info("\n%s", structure.summary())

        # Step 2: Build anonymizer
        logger.info("Step 2: Building anonymizer...")
        self.anonymizer = Anonymizer()
        self.anonymizer.build_from_database(self.config.database_path, structure)
        mapping_path = self.anonymizer.save_mapping(self.config.output_dir)
        logger.info("Mapping saved to %s", mapping_path)

        # Step 3: Load comments
        logger.info("Step 3: Loading and preprocessing comments...")
        user_comments = load_user_comments(
            self.config.database_path,
            structure,
            self.anonymizer,
            min_comments=self.config.min_comments,
            min_comment_length=self.config.min_comment_length,
        )
        self.state.total_users = len(user_comments)
        logger.info("Found %d users with sufficient comments", self.state.total_users)

        # Step 4: Load model (skip if caller already preloaded it)
        if self.model is not None and self.tokenizer is not None:
            logger.info("Step 4: Using preloaded model")
        else:
            logger.info("Step 4: Loading model...")
            from .model_loader import load_model, ModelConfig
            model_config = ModelConfig(
                model_id=self.config.model_id,
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.0,
                do_sample=False,
                seed=self.config.seed,
            )
            self.model, self.tokenizer = load_model(model_config)

        # Step 5: Start run
        self.run_id = self.storage.start_run(self.config.to_dict())

        # Step 6: Process users
        logger.info("Step 5: Processing users...")
        for i, uc in enumerate(user_comments):
            anon_id = uc.anonymous_id

            # Check resume — look for user in ANY previous run
            if self.config.resume and not self.config.force_reprocess:
                if self.storage.is_user_processed_in_any_run(anon_id):
                    logger.info(
                        "[%d/%d] Skipping %s (already processed)",
                        i + 1,
                        self.state.total_users,
                        anon_id,
                    )
                    self.state.skipped_users += 1
                    continue

            logger.info(
                "[%d/%d] Processing %s (%d political comments)",
                i + 1,
                self.state.total_users,
                anon_id,
                len(uc.comments),
            )

            try:
                result = self._process_user(uc)
                self.state.results.append(result)
                self.state.processed_users += 1
            except Exception as e:
                logger.error("Error processing %s: %s", anon_id, e)
                self.storage.save_error(
                    anon_id,
                    "processing_error",
                    str(e),
                    run_id=self.run_id,
                )
                self.state.error_users += 1

            # Save checkpoint every 10 users
            if (i + 1) % 10 == 0:
                self._save_checkpoint()

        # Step 7: Save final results
        self._save_checkpoint()
        self.storage.complete_run(self.run_id, {
            "total_users": self.state.total_users,
            "processed_users": self.state.processed_users,
            "skipped_users": self.state.skipped_users,
            "error_users": self.state.error_users,
        })

        # Step 8: Save summary
        self.storage.save_summary(self.state.results)

        elapsed = time.time() - self.state.start_time
        logger.info("=" * 60)
        logger.info("Pipeline completed in %.1f seconds", elapsed)
        logger.info("Total users: %d", self.state.total_users)
        logger.info("Processed: %d", self.state.processed_users)
        logger.info("Skipped: %d", self.state.skipped_users)
        logger.info("Errors: %d", self.state.error_users)
        logger.info("=" * 60)

        return self.state

    def _process_user(self, uc: UserComments) -> dict:
        """Обрабатывает одного пользователя."""
        # Limit comments
        uc = limit_comments(uc, self.config.max_comments_per_user, self.config.max_input_tokens)

        # Prepare comments for model
        comments_for_model = [
            {
                "id": c.original_id,
                "text": c.text,
                "date": c.date,
                "channel": c.channel,
            }
            for c in uc.comments
        ]

        # Split into blocks if too many
        blocks = split_into_blocks(comments_for_model, self.config.max_comments_per_block)

        block_results: list[dict] = []
        for block_idx, block in enumerate(blocks):
            logger.debug(
                "  Block %d/%d: %d comments", block_idx + 1, len(blocks), len(block)
            )

            result = self._analyze_block(block)
            if result.parsed_json:
                result.parsed_json["analyzed_comments"] = len(block)
                block_results.append(result.parsed_json)
            else:
                logger.warning("  Block %d failed: %s", block_idx + 1, result.error)

        if not block_results:
            raise ValueError("All blocks failed analysis")

        # Merge block results
        merged = merge_block_results(block_results)
        merged["analyzed_comments"] = sum(
            r.get("analyzed_comments", 0) for r in block_results
        )

        # Build final result
        evidence = merged.get("evidence", {})
        stat = {
            "total_comments": uc.total_comments,
            "unique_comments": uc.unique_comments,
            "political_comments": uc.political_comments,
            "analyzed_comments": merged.get("analyzed_comments", len(uc.comments)),
        }

        self.storage.save_user_result(
            anon_id=uc.anonymous_id,
            real_id=uc.real_id,
            statistics=stat,
            analysis=merged,
            evidence=evidence,
            contradictions=merged.get("contradictions", []),
            unknown_topics=merged.get("unknown_topics", []),
            overall_confidence=merged.get("overall_confidence", 0.0),
            insufficient_data=merged.get("insufficient_data", False),
            run_id=self.run_id,
        )

        # Build JSONL record
        record = {
            "anonymous_user_id": uc.anonymous_id,
            "statistics": stat,
            "content_analysis": merged.get("content_analysis"),
            "evidence": evidence,
            "contradictions": merged.get("contradictions", []),
            "unknown_topics": merged.get("unknown_topics", []),
            "overall_confidence": merged.get("overall_confidence", 0.0),
            "insufficient_data": merged.get("insufficient_data", False),
            "disclaimer": "Результат описывает политическое содержание предоставленных сообщений и не является достоверным определением личных убеждений автора.",
        }
        return record

    def _analyze_block(self, comments: list[dict]) -> AnalysisResult:
        """Анализирует один блок комментариев с retry."""
        attempt = 0
        max_attempts = self.config.max_retries + 1

        while attempt < max_attempts:
            attempt += 1
            result = run_inference(
                self.model,
                self.tokenizer,
                comments,
                max_new_tokens=self.config.max_new_tokens,
                seed=self.config.seed,
            )

            if not result.is_valid:
                logger.warning("  Attempt %d: invalid JSON", attempt)
                continue

            # Validate
            comment_ids = [c["id"] for c in comments]
            validated, errors = validate_analysis_response(
                result.parsed_json, comment_ids
            )

            if not errors:
                result.parsed_json = validated.model_dump()
                return result

            logger.warning("  Attempt %d: validation errors: %s", attempt, errors)

            # Try correction
            if attempt < max_attempts:
                correction_prompt = build_correction_prompt(
                    "", result.raw_response, errors
                )
                # Re-run with correction
                correction_comments = comments + [
                    {"id": -1, "text": correction_prompt, "date": "", "channel": ""}
                ]
                result = run_inference(
                    self.model,
                    self.tokenizer,
                    correction_comments,
                    max_new_tokens=self.config.max_new_tokens,
                    seed=self.config.seed,
                )
                if result.is_valid:
                    validated, errors = validate_analysis_response(
                        result.parsed_json, comment_ids
                    )
                    if not errors:
                        result.parsed_json = validated.model_dump()
                        return result

        # All attempts failed
        self.storage.save_error(
            None,
            "validation_error",
            f"All {max_attempts} attempts failed",
            result.raw_response if result else "",
            attempt,
            self.run_id,
        )
        return result

    def _save_checkpoint(self) -> None:
        """Сохраняет контрольную точку."""
        elapsed = time.time() - self.state.start_time
        logger.info(
            "Checkpoint: %d/%d processed, %.1fs elapsed",
            self.state.processed_users,
            self.state.total_users,
            elapsed,
        )


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint для запуска пайплайна."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Anonymous political content analysis pipeline"
    )
    parser.add_argument(
        "--database", default="app.db",
        help="Path to SQLite database (default: app.db)"
    )
    parser.add_argument(
        "--model-id", default="Qwen/Qwen3-4B-Instruct-2507",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--min-comments", type=int, default=20,
        help="Minimum comments per user (default: 20)"
    )
    parser.add_argument(
        "--min-comment-length", type=int, default=20,
        help="Minimum comment length in chars (default: 20)"
    )
    parser.add_argument(
        "--max-comments-per-user", type=int, default=300,
        help="Max comments per user (default: 300)"
    )
    parser.add_argument(
        "--max-input-tokens", type=int, default=24000,
        help="Max input tokens (default: 24000)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=1500,
        help="Max new tokens for generation (default: 1500)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Output directory (default: current dir)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Disable resume (reprocess all users)"
    )
    parser.add_argument(
        "--force-reprocess", action="store_true",
        help="Force reprocessing already completed users"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = PipelineConfig(
        database_path=args.database,
        model_id=args.model_id,
        min_comments=args.min_comments,
        min_comment_length=args.min_comment_length,
        max_comments_per_user=args.max_comments_per_user,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        force_reprocess=args.force_reprocess,
    )

    pipeline = Pipeline(config)
    state = pipeline.run()
    sys.exit(0 if state.error_users == 0 else 1)


if __name__ == "__main__":
    main()
