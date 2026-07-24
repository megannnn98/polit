"""Тесты для модуля validator."""

import pytest

from political_analysis.validator import (
    validate_analysis_response,
    EconomicAxis,
    AuthorityAxis,
    ScienceAxis,
    AnalysisResponse,
    build_correction_prompt,
)


def test_valid_response():
    """Проверяет валидный JSON-ответ."""
    raw = {
        "content_analysis": {
            "axes": {
                "economic": {"left": 65, "right": 35, "confidence": 0.76},
                "authority": {"authoritarian": 30, "libertarian": 70, "confidence": 0.71},
                "science": {"scientific": 80, "anti_scientific": 20, "confidence": 0.68},
            },
            "ideological_similarity": {
                "liberalism": 35,
                "conservatism": 12,
                "social_democracy": 58,
                "marxism": 66,
                "anarchism": 42,
                "nationalism": 8,
                "fascism_nazism": 2,
                "political_indifference": 5,
            },
            "protest_rhetoric": {"score": 45, "confidence": 0.61},
        },
        "evidence": {"economic": [{"comment_id": 1, "quote": "Тестовая цитата"}]},
        "contradictions": [],
        "unknown_topics": ["миграция"],
        "overall_confidence": 0.7,
        "insufficient_data": False,
    }
    response, errors = validate_analysis_response(raw)
    assert response is not None
    assert len(errors) == 0


def test_economic_axis_auto_correct():
    """Проверяет автокоррекцию суммы шкалы."""
    axis = EconomicAxis(left=70, right=40)
    # 70 + 40 = 110, should auto-correct to 100
    assert axis.left + axis.right == 100


def test_authority_axis_auto_correct():
    """Проверяет автокоррекцию шкалы authority."""
    axis = AuthorityAxis(authoritarian=60, libertarian=60)
    assert axis.authoritarian + axis.libertarian == 100


def test_science_axis_auto_correct():
    """Проверяет автокоррекцию шкалы science."""
    axis = ScienceAxis(scientific=30, anti_scientific=70)
    assert axis.scientific + axis.anti_scientific == 100


def test_evidence_nonexistent_comment():
    """Проверяет ошибку при цитате из несуществующего комментария."""
    raw = {
        "content_analysis": {
            "axes": {
                "economic": {"left": 50, "right": 50, "confidence": 0.5},
                "authority": {"authoritarian": 50, "libertarian": 50, "confidence": 0.5},
                "science": {"scientific": 50, "anti_scientific": 50, "confidence": 0.5},
            },
            "ideological_similarity": {},
            "protest_rhetoric": {"score": 0, "confidence": 0.5},
        },
        "evidence": {"economic": [{"comment_id": 999, "quote": "Цитата"}]},
        "contradictions": [],
        "unknown_topics": [],
        "overall_confidence": 0.5,
        "insufficient_data": False,
    }
    response, errors = validate_analysis_response(raw, comment_ids=[1, 2, 3])
    assert response is not None
    assert len(errors) > 0
    assert any("999" in e for e in errors)


def test_empty_response():
    """Проверяет обработку пустого ответа."""
    raw = {}
    response, errors = validate_analysis_response(raw)
    # Empty response might still parse (all fields optional)
    # But should have no errors for empty valid structure
    assert response is not None or len(errors) > 0


def test_insufficient_data():
    """Проверяет insufficient_data=true."""
    raw = {
        "content_analysis": None,
        "evidence": {},
        "contradictions": [],
        "unknown_topics": ["все темы"],
        "overall_confidence": 0.1,
        "insufficient_data": True,
    }
    response, errors = validate_analysis_response(raw)
    assert response is not None
    assert response.insufficient_data is True


def test_build_correction_prompt():
    """Проверяет генерацию промпта исправления."""
    prompt = build_correction_prompt(
        "Original prompt",
        '{"left": 70, "right": 40}',
        ["Economic axis sum: 70 + 40 != 100"],
    )
    assert "70 + 40 != 100" in prompt
    assert "исправьте" in prompt.lower() or "correct" in prompt.lower()


def test_range_violation():
    """Проверяет обнаружение значений вне диапазона."""
    raw = {
        "content_analysis": {
            "axes": {
                "economic": {"left": 150, "right": -50, "confidence": 0.5},
                "authority": {"authoritarian": 50, "libertarian": 50, "confidence": 0.5},
                "science": {"scientific": 50, "anti_scientific": 50, "confidence": 0.5},
            },
            "ideological_similarity": {},
            "protest_rhetoric": {"score": 0, "confidence": 0.5},
        },
        "evidence": {},
        "contradictions": [],
        "unknown_topics": [],
        "overall_confidence": 0.5,
        "insufficient_data": False,
    }
    # Pydantic should catch ge=0, le=100 violations
    response, errors = validate_analysis_response(raw)
    # Either response is None or errors contain the range violation
    assert response is None or len(errors) > 0
