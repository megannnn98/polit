"""Валидация JSON-ответа модели через Pydantic."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class EconomicAxis(BaseModel):
    left: int | None = Field(None, ge=0, le=100)
    right: int | None = Field(None, ge=0, le=100)
    confidence: float | None = Field(None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_sum(self):
        if self.left is not None and self.right is not None:
            total = self.left + self.right
            if total != 100:
                orig_left, orig_right = self.left, self.right
                if total > 0:
                    self.left = round(self.left * 100 / total)
                    self.right = 100 - self.left
                else:
                    self.left = 50
                    self.right = 50
                logger.warning(
                    "Economic axis sum auto-corrected: %d+%d=%d → %d+%d=100",
                    orig_left, orig_right, total, self.left, self.right,
                )
        return self


class AuthorityAxis(BaseModel):
    authoritarian: int | None = Field(None, ge=0, le=100)
    libertarian: int | None = Field(None, ge=0, le=100)
    confidence: float | None = Field(None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_sum(self):
        if self.authoritarian is not None and self.libertarian is not None:
            total = self.authoritarian + self.libertarian
            if total != 100:
                orig_auth, orig_lib = self.authoritarian, self.libertarian
                if total > 0:
                    self.authoritarian = round(self.authoritarian * 100 / total)
                    self.libertarian = 100 - self.authoritarian
                else:
                    self.authoritarian = 50
                    self.libertarian = 50
                logger.warning(
                    "Authority axis sum auto-corrected: %d+%d=%d → %d+%d=100",
                    orig_auth, orig_lib, total, self.authoritarian, self.libertarian,
                )
        return self


class ScienceAxis(BaseModel):
    scientific: int | None = Field(None, ge=0, le=100)
    anti_scientific: int | None = Field(None, ge=0, le=100)
    confidence: float | None = Field(None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_sum(self):
        if self.scientific is not None and self.anti_scientific is not None:
            total = self.scientific + self.anti_scientific
            if total != 100:
                orig_sci, orig_anti = self.scientific, self.anti_scientific
                if total > 0:
                    self.scientific = round(self.scientific * 100 / total)
                    self.anti_scientific = 100 - self.scientific
                else:
                    self.scientific = 50
                    self.anti_scientific = 50
                logger.warning(
                    "Science axis sum auto-corrected: %d+%d=%d → %d+%d=100",
                    orig_sci, orig_anti, total, self.scientific, self.anti_scientific,
                )
        return self


class Axes(BaseModel):
    economic: EconomicAxis | None = None
    authority: AuthorityAxis | None = None
    science: ScienceAxis | None = None


class IdeologicalSimilarity(BaseModel):
    liberalism: int | None = Field(None, ge=0, le=100)
    conservatism: int | None = Field(None, ge=0, le=100)
    social_democracy: int | None = Field(None, ge=0, le=100)
    marxism: int | None = Field(None, ge=0, le=100)
    anarchism: int | None = Field(None, ge=0, le=100)
    nationalism: int | None = Field(None, ge=0, le=100)
    fascism_nazism: int | None = Field(None, ge=0, le=100)
    political_indifference: int | None = Field(None, ge=0, le=100)


class ProtestRhetoric(BaseModel):
    score: int | None = Field(None, ge=0, le=100)
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class EvidenceItem(BaseModel):
    comment_id: int | str
    quote: str = Field(min_length=1, max_length=500)


class ContentAnalysis(BaseModel):
    axes: Axes | None = None
    ideological_similarity: IdeologicalSimilarity | None = None
    protest_rhetoric: ProtestRhetoric | None = None


class Contradiction(BaseModel):
    topic: str
    comments: list[int | str] = Field(default_factory=list)
    description: str = ""


class AnalysisResponse(BaseModel):
    content_analysis: ContentAnalysis | None = None
    evidence: dict[str, list[EvidenceItem]] = Field(default_factory=dict)
    contradictions: list[Contradiction] = Field(default_factory=list)
    unknown_topics: list[str] = Field(default_factory=list)
    overall_confidence: float | None = Field(None, ge=0.0, le=1.0)
    insufficient_data: bool = False


def validate_analysis_response(
    raw_json: dict[str, Any],
    comment_ids: list[int | str] | None = None,
) -> tuple[AnalysisResponse | None, list[str]]:
    """Валидирует JSON-ответ модели.

    Args:
        raw_json: распарсенный JSON из модели
        comment_ids: допустимые ID комментариев для проверки цитат

    Returns:
        tuple (валидный ответ, список ошибок)
    """
    errors: list[str] = []

    try:
        response = AnalysisResponse(**raw_json)
    except Exception as e:
        errors.append(f"Pydantic validation error: {e}")
        return None, errors

    # Check evidence quotes exist in source comments
    if comment_ids:
        for category, items in response.evidence.items():
            for item in items:
                if isinstance(item.comment_id, int):
                    if item.comment_id not in comment_ids:
                        errors.append(
                            f"Evidence quote references non-existent comment_id={item.comment_id}"
                        )

    return response, errors


def build_correction_prompt(
    original_prompt: str,
    raw_response: str,
    errors: list[str],
) -> str:
    """Строит промпт для исправления ошибок.

    Args:
        original_prompt: исходный промпт
        raw_response: ответ модели с ошибками
        errors: список найденных ошибок

    Returns:
        новый промпт с запросом исправления
    """
    error_list = "\n".join(f"- {e}" for e in errors)
    return f"""Ваш предыдущий ответ содержит ошибки:
{error_list}

Исходный ответ:
{raw_response}

Пожалуйста, исправьте ошибки и верните исправленный JSON.
Убедитесь что:
1. Все суммы шкал (left+right, authoritarian+libertarian, scientific+anti_scientific) = 100
2. Все значения в допустимых диапазонах
3. Цитаты указывают на реальные ID комментариев
4. JSON валиден и содержит все обязательные поля"""
