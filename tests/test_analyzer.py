"""Тесты для модуля analyzer."""

import pytest

from political_analysis.analyzer import (
    _BalancedJsonStoppingCriteria,
    _extract_json,
    _json_object_end,
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


def test_extract_json_ignores_braces_inside_strings():
    """Проверяет, что фигурные скобки в строках не ломают поиск JSON."""
    text = 'prefix {"quote": "текст с {примером}", "score": 50} suffix'
    result = _extract_json(text)
    assert result == {"quote": "текст с {примером}", "score": 50}


def test_extract_json_skips_invalid_object_before_valid_json():
    """Проверяет, что невалидный объект перед JSON не блокирует парсинг."""
    text = "prefix {broken} middle {\"score\": 50}"
    result = _extract_json(text)
    assert result == {"score": 50}


def test_extract_json_invalid():
    """Проверяет обработку невалидного JSON."""
    text = "This is not JSON at all"
    result = _extract_json(text)
    assert result is None


def test_extract_json_recovers_truncated_after_content_analysis():
    """Проверяет fallback для ответа, оборванного после content_analysis."""
    text = """{
  "content_analysis": {
    "axes": {
      "economic": {"left": 30, "right": 70, "confidence": 0.75},
      "authority": {"authoritarian": 85, "libertarian": 15, "confidence": 0.8},
      "science": {"scientific": 40, "anti_scientific": 60, "confidence": 0.6}
    },
    "ideological_similarity": {
      "liberalism": 15,
      "conservatism": 70,
      "social_democracy": 10,
      "marxism": 5,
      "anarchism": 2,
      "nationalism": 80,
      "fascism_nazism": 45,
      "political_indifference": 5
    },
    "protest_rhetoric": {"score": 60, "confidence": 0.7}
  },
  "ev"""
    result = _extract_json(text)
    assert result is not None
    assert result["partial_response"] is True
    assert result["content_analysis"]["axes"]["economic"]["right"] == 70
    assert result["content_analysis"]["ideological_similarity"]["nationalism"] == 80
    assert result["evidence"] == {}
    assert result["overall_confidence"] == 0.71


def test_split_into_blocks():
    """Проверяет разбиение на блоки."""
    items = [{"text": str(i)} for i in range(10)]
    blocks = split_into_blocks(items, max_comments_per_block=3)
    assert len(blocks) == 4
    assert blocks[0] == [{"text": "0"}, {"text": "1"}, {"text": "2"}]
    assert blocks[-1] == [{"text": "9"}]


def test_split_into_blocks_respects_char_limit():
    """Проверяет разбиение по суммарной длине текста."""
    items = [{"text": "aaaa"}, {"text": "bbbb"}, {"text": "cccc"}]
    blocks = split_into_blocks(
        items,
        max_comments_per_block=10,
        max_chars_per_block=8,
    )
    assert blocks == [[items[0], items[1]], [items[2]]]


def test_json_object_end_waits_for_complete_json():
    """Проверяет детекцию завершённого верхнеуровневого JSON."""
    assert _json_object_end('{"a": {"b": 1}} trailing') == len('{"a": {"b": 1}}')
    assert _json_object_end('{"a": 1') is None


def test_balanced_json_stopping_criteria():
    """Проверяет остановку после валидного JSON-ответа."""
    class FakeTokenizer:
        def decode(self, generated_ids, skip_special_tokens=True):
            del generated_ids, skip_special_tokens
            return '{"a": 1}'

    stopper = _BalancedJsonStoppingCriteria(FakeTokenizer(), prompt_token_count=2)
    assert stopper([[100, 101, 1, 2, 3]], None) is True


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
