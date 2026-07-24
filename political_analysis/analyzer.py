"""Генерация промпта, инференс модели и парсинг ответа."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PROMPT_VERSION = "2.0"

SYSTEM_PROMPT = """Вы — аналитик политических текстов. Анализируете ТОЛЬКО явно выраженные позиции в сообщениях.

ПРАВИЛА:
1. Анализируйте только то, что ЧЁТКО написано в тексте.
2. Не делайте выводов по имени, нику, полу, возрасту или месту проживания.
3. Пересказ чужой позиции — это НЕ мнение автора.
4. Цитата — это НЕ поддержка процитированной позиции.
5. Нельзя делать вывод по одному сообщению.
6. Не заполняйте отсутствующие данные догадками.
7. Протестный потенциал ≠ склонность к преступлению/насилию.
8. Если данных недостаточно — честно укажите insufficient_data=true.

ШКАЛЫ:
- economic: left (0-100) + right (0-100) = 100
- authority: authoritarian (0-100) + libertarian (0-100) = 100
- science: scientific (0-100) + anti_scientific (0-100) = 100

ИДЕОЛОГИЧЕСКИЕ СХОДСТВА (независимые, 0-100 каждое):
- liberalism, conservatism, social_democracy, marxism,
- anarchism, nationalism, fascism_nazism, political_indifference

ОБЯЗАТЕЛЬНЫЕ ПОЛЯ:
- confidence (0.0-1.0)
- insufficient_data (boolean)
- evidence (подтверждающие цитаты)
- contradictions (противоречия)
- unknown_topics (недостаточно данных темы)

Отвечайте ТОЛЬКО валидным JSON без markdown-обёрток."""


def _build_analysis_prompt(comments: list[dict]) -> str:
    """Строит промпт для анализа комментариев."""
    comments_text = "\n---\n".join(
        f"[{c['date']}] #{c['id']}: {c['text']}" for c in comments
    )
    return f"""Проанализируйте политическое содержание следующих сообщений.

Сообщения ({len(comments)} штук):
{comments_text}

Верните JSON со следующей структурой:
{{
  "content_analysis": {{
    "axes": {{
      "economic": {{"left": <0-100>, "right": <0-100>, "confidence": <0.0-1.0>}},
      "authority": {{"authoritarian": <0-100>, "libertarian": <0-100>, "confidence": <0.0-1.0>}},
      "science": {{"scientific": <0-100>, "anti_scientific": <0-100>, "confidence": <0.0-1.0>}}
    }},
    "ideological_similarity": {{
      "liberalism": <0-100>,
      "conservatism": <0-100>,
      "social_democracy": <0-100>,
      "marxism": <0-100>,
      "anarchism": <0-100>,
      "nationalism": <0-100>,
      "fascism_nazism": <0-100>,
      "political_indifference": <0-100>
    }},
    "protest_rhetoric": {{"score": <0-100>, "confidence": <0.0-1.0>}}
  }},
  "evidence": {{
    "economic": [{{"comment_id": <id>, "quote": "<цитата>"}}],
    "authority": [],
    "science": [],
    "ideologies": [],
    "protest_rhetoric": []
  }},
  "contradictions": [{{"topic": "<тема>", "comments": [<id>], "description": "<описание>"}}],
  "unknown_topics": ["<тема>"],
  "overall_confidence": <0.0-1.0>,
  "insufficient_data": <true/false>
}}"""


@dataclass
class AnalysisResult:
    """Результат анализа одного пользователя."""
    raw_response: str = ""
    parsed_json: dict | None = None
    is_valid: bool = False
    error: str | None = None


def run_inference(
    model,
    tokenizer,
    comments: list[dict],
    max_new_tokens: int = 1500,
    temperature: float = 0.0,
    do_sample: bool = False,
    seed: int = 42,
) -> AnalysisResult:
    """Запускает инференс модели на комментариях.

    Args:
        model: загруженная модель
        tokenizer: токенизатор
        comments: список комментариев [{id, text, date, channel}]
        max_new_tokens: максимальная длина ответа
        temperature: температура генерации
        do_sample: сэмплирование
        seed: seed для воспроизводимости

    Returns:
        AnalysisResult с сырым ответом и распарсенным JSON
    """
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt = _build_analysis_prompt(comments)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    result = AnalysisResult(raw_response=raw_response)
    result.parsed_json = _extract_json(raw_response)
    if result.parsed_json is not None:
        result.is_valid = True
    else:
        result.error = "Failed to extract valid JSON from model response"

    return result


def _extract_json(text: str) -> dict | None:
    """Извлекает JSON из ответа модели.

    Пробует несколько стратегий:
    1. Прямой парсинг
    2. Поиск JSON в markdown-блоке
    3. Поиск по фигурным скобкам
    """
    text = text.strip()

    # Remove markdown code block wrappers
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1

    logger.debug("Failed to extract JSON from: %.200s...", text)
    return None


def split_into_blocks(
    comments: list[dict],
    max_comments_per_block: int = 50,
) -> list[list[dict]]:
    """Разбивает комментарии на блоки для обработки.

    Args:
        comments: все комментарии
        max_comments_per_block: макс. комментариев в блоке

    Returns:
        список блоков
    """
    blocks = []
    for i in range(0, len(comments), max_comments_per_block):
        blocks.append(comments[i : i + max_comments_per_block])
    return blocks


def merge_block_results(block_results: list[dict]) -> dict:
    """Объединяет результаты нескольких блоков.

    Усредняет числовые показатели, взвешивая по количеству
    комментариев в блоке.
    """
    if not block_results:
        return {}
    if len(block_results) == 1:
        return block_results[0]

    merged = json.loads(json.dumps(block_results[0]))
    total_comments = sum(r.get("analyzed_comments", 1) for r in block_results)

    # Average axes
    for axis in ["economic", "authority", "science"]:
        if axis not in merged.get("content_analysis", {}).get("axes", {}):
            continue
        for key in merged["content_analysis"]["axes"][axis]:
            if key == "confidence":
                continue
            values = []
            weights = []
            for r in block_results:
                if r.get("content_analysis", {}).get("axes", {}).get(axis, {}).get(key) is not None:
                    w = r.get("analyzed_comments", 1)
                    values.append(r["content_analysis"]["axes"][axis][key])
                    weights.append(w)
            if values:
                merged["content_analysis"]["axes"][axis][key] = round(
                    sum(v * w for v, w in zip(values, weights)) / sum(weights)
                )

    # Merge evidence (deduplicate by comment_id + quote)
    all_evidence: dict[str, list] = {}
    for r in block_results:
        for key, items in r.get("evidence", {}).items():
            existing = {
                (e.get("comment_id"), e.get("quote"))
                for e in all_evidence.get(key, [])
            }
            for item in items:
                pair = (item.get("comment_id"), item.get("quote"))
                if pair not in existing:
                    all_evidence.setdefault(key, []).append(item)
                    existing.add(pair)
    merged["evidence"] = all_evidence

    # Merge ideological_similarity (weighted average of numeric fields)
    if merged.get("content_analysis", {}).get("ideological_similarity"):
        ideol_keys = list(merged["content_analysis"]["ideological_similarity"].keys())
        for key in ideol_keys:
            values = []
            weights = []
            for r in block_results:
                val = r.get("content_analysis", {}).get("ideological_similarity", {}).get(key)
                if val is not None:
                    w = r.get("analyzed_comments", 1)
                    values.append(val)
                    weights.append(w)
            if values:
                merged["content_analysis"]["ideological_similarity"][key] = round(
                    sum(v * w for v, w in zip(values, weights)) / sum(weights)
                )

    # Merge protest_rhetoric (weighted average)
    if merged.get("content_analysis", {}).get("protest_rhetoric"):
        for key in ["score", "confidence"]:
            values = []
            weights = []
            for r in block_results:
                val = r.get("content_analysis", {}).get("protest_rhetoric", {}).get(key)
                if val is not None:
                    w = r.get("analyzed_comments", 1)
                    values.append(val)
                    weights.append(w)
            if values:
                weighted = sum(v * w for v, w in zip(values, weights)) / sum(weights)
                merged["content_analysis"]["protest_rhetoric"][key] = (
                    round(weighted, 2) if key == "confidence" else round(weighted)
                )

    # Merge axes confidence (average)
    for axis in ["economic", "authority", "science"]:
        confs = []
        for r in block_results:
            c = r.get("content_analysis", {}).get("axes", {}).get(axis, {}).get("confidence")
            if c is not None:
                confs.append(c)
        if confs and merged.get("content_analysis", {}).get("axes", {}).get(axis):
            merged["content_analysis"]["axes"][axis]["confidence"] = round(
                sum(confs) / len(confs), 2
            )

    # Merge contradictions and unknown_topics
    all_contradictions = []
    all_unknown = []
    for r in block_results:
        all_contradictions.extend(r.get("contradictions", []))
        all_unknown.extend(r.get("unknown_topics", []))
    merged["contradictions"] = all_contradictions
    merged["unknown_topics"] = list(set(all_unknown))

    # Overall confidence
    confs = [
        r.get("overall_confidence", 0.5)
        for r in block_results
        if r.get("overall_confidence") is not None
    ]
    merged["overall_confidence"] = round(sum(confs) / len(confs), 2) if confs else 0.0

    # insufficient_data: True if ANY block has insufficient data
    merged["insufficient_data"] = any(
        r.get("insufficient_data", False) for r in block_results
    )

    return merged
