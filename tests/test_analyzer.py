"""Тесты для модуля analyzer."""

import pytest

from political_analysis.analyzer import (
    _extract_json,
    split_into_blocks,
    merge_block_results,
)


def test_extract_json_direct():
    """Проверяет прямой парсинг JSON."""
    text = '{"left": 50, "right": 50}'
    result = _extract_json(text)
    assert result == {"left": 50, "right": 50}


def test_extract_json_markdown_block():
    """Проверяет извлечение JSON из markdown-блока."""
    text = '```json\n{"left": 50, "right": 50}\n```'
    result = _extract_json(text)
    assert result == {"left": 50, "right": 50}


def test_extract_json_with_surrounding_text():
    """Проверяет извлечение JSON из текста с обрамлением."""
    text = 'Вот результат:\n{"left": 50, "right": 50}\nКонец.'
    result = _extract_json(text)
    assert result == {"left": 50, "right": 50}


def test_extract_json_invalid():
    """Проверяет обработку невалидного JSON."""
    text = "This is not JSON at all"
    result = _extract_json(text)
    assert result is None


def test_split_into_blocks():
    """Проверяет разбиение на блоки."""
    items = list(range(10))
    blocks = split_into_blocks(items, max_comments_per_block=3)
    assert len(blocks) == 4
    assert blocks[0] == [0, 1, 2]
    assert blocks[-1] == [9]


def test_merge_block_results_deduplicates_evidence():
    """Проверяет дедупликацию evidence при merge блоков."""
    block1 = {
        "content_analysis": {"axes": {"economic": {"left": 60, "right": 40, "confidence": 0.7}}},
        "evidence": {"economic": [{"comment_id": 1, "quote": "Quote A"}]},
        "contradictions": [],
        "unknown_topics": [],
        "overall_confidence": 0.7,
        "analyzed_comments": 5,
    }
    block2 = {
        "content_analysis": {"axes": {"economic": {"left": 70, "right": 30, "confidence": 0.8}}},
        "evidence": {"economic": [{"comment_id": 1, "quote": "Quote A"}]},  # Duplicate
        "contradictions": [],
        "unknown_topics": [],
        "overall_confidence": 0.8,
        "analyzed_comments": 5,
    }
    merged = merge_block_results([block1, block2])
    evidence = merged.get("evidence", {}).get("economic", [])
    assert len(evidence) == 1  # Should be deduplicated


def test_merge_block_results_combines_unique_evidence():
    """Проверяет объединение уникальных evidence."""
    block1 = {
        "content_analysis": {"axes": {"economic": {"left": 60, "right": 40, "confidence": 0.7}}},
        "evidence": {"economic": [{"comment_id": 1, "quote": "Quote A"}]},
        "contradictions": [],
        "unknown_topics": [],
        "overall_confidence": 0.7,
        "analyzed_comments": 5,
    }
    block2 = {
        "content_analysis": {"axes": {"economic": {"left": 70, "right": 30, "confidence": 0.8}}},
        "evidence": {"economic": [{"comment_id": 2, "quote": "Quote B"}]},
        "contradictions": [],
        "unknown_topics": [],
        "overall_confidence": 0.8,
        "analyzed_comments": 5,
    }
    merged = merge_block_results([block1, block2])
    evidence = merged.get("evidence", {}).get("economic", [])
    assert len(evidence) == 2


def test_merge_block_results_ideological_similarity():
    """Проверяет weighted merge ideological_similarity."""
    block1 = {
        "content_analysis": {
            "axes": {"economic": {"left": 60, "right": 40, "confidence": 0.7}},
            "ideological_similarity": {"marxism": 80, "anarchism": 20},
            "protest_rhetoric": {"score": 30, "confidence": 0.6},
        },
        "evidence": {},
        "contradictions": [],
        "unknown_topics": [],
        "overall_confidence": 0.7,
        "analyzed_comments": 10,
    }
    block2 = {
        "content_analysis": {
            "axes": {"economic": {"left": 70, "right": 30, "confidence": 0.8}},
            "ideological_similarity": {"marxism": 40, "anarchism": 60},
            "protest_rhetoric": {"score": 70, "confidence": 0.9},
        },
        "evidence": {},
        "contradictions": [],
        "unknown_topics": [],
        "overall_confidence": 0.8,
        "analyzed_comments": 20,
    }
    merged = merge_block_results([block1, block2])
    ideol = merged["content_analysis"]["ideological_similarity"]
    # block2 has 2x weight: marxism = (80*10 + 40*20) / 30 = 53
    assert ideol["marxism"] == 53
    # anarchism = (20*10 + 60*20) / 30 = 47
    assert ideol["anarchism"] == 47
    protest = merged["content_analysis"]["protest_rhetoric"]
    assert protest["score"] == 57  # (30*10 + 70*20) / 30
    assert protest["confidence"] == round((0.6 * 10 + 0.9 * 20) / 30, 2)  # 0.8


def test_merge_block_results_insufficient_data():
    """Проверяет insufficient_data: True если ХОТЯ БЫ ОДИН блок insufficient."""
    block1 = {
        "content_analysis": {"axes": {}, "ideological_similarity": {}, "protest_rhetoric": {}},
        "evidence": {},
        "contradictions": [],
        "unknown_topics": [],
        "overall_confidence": 0.7,
        "analyzed_comments": 5,
        "insufficient_data": False,
    }
    block2 = {
        "content_analysis": {"axes": {}, "ideological_similarity": {}, "protest_rhetoric": {}},
        "evidence": {},
        "contradictions": [],
        "unknown_topics": [],
        "overall_confidence": 0.3,
        "analyzed_comments": 2,
        "insufficient_data": True,
    }
    merged = merge_block_results([block1, block2])
    assert merged["insufficient_data"] is True


def test_merge_block_results_axes_confidence():
    """Проверяет усреднение confidence по осям."""
    block1 = {
        "content_analysis": {
            "axes": {"economic": {"left": 60, "right": 40, "confidence": 0.6}},
            "ideological_similarity": {},
            "protest_rhetoric": {},
        },
        "evidence": {},
        "contradictions": [],
        "unknown_topics": [],
        "overall_confidence": 0.7,
        "analyzed_comments": 10,
    }
    block2 = {
        "content_analysis": {
            "axes": {"economic": {"left": 70, "right": 30, "confidence": 0.9}},
            "ideological_similarity": {},
            "protest_rhetoric": {},
        },
        "evidence": {},
        "contradictions": [],
        "unknown_topics": [],
        "overall_confidence": 0.8,
        "analyzed_comments": 10,
    }
    merged = merge_block_results([block1, block2])
    assert merged["content_analysis"]["axes"]["economic"]["confidence"] == 0.75
