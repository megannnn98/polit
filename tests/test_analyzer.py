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
