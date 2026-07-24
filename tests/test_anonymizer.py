"""Тесты для модуля anonymizer."""

import json
import os
import tempfile
import pytest
from pathlib import Path

from political_analysis.anonymizer import Anonymizer


def test_anonymize_creates_id():
    """Проверяет генерацию анонимного ID."""
    anon = Anonymizer()
    result = anon.anonymize(12345)
    assert result == "user_000001"
    assert anon.counter == 1


def test_anonymize_deterministic():
    """Проверяет, что повторный вызов возвращает тот же ID."""
    anon = Anonymizer()
    id1 = anon.anonymize(12345)
    id2 = anon.anonymize(12345)
    assert id1 == id2
    assert anon.counter == 1


def test_anonymize_sequential():
    """Проверяет последовательную генерацию ID."""
    anon = Anonymizer()
    id1 = anon.anonymize(100)
    id2 = anon.anonymize(200)
    id3 = anon.anonymize(300)
    assert id1 == "user_000001"
    assert id2 == "user_000002"
    assert id3 == "user_000003"


def test_deanonymize():
    """Проверяет обратное преобразование."""
    anon = Anonymizer()
    anon.anonymize(12345)
    assert anon.deanonymize("user_000001") == 12345
    assert anon.deanonymize("nonexistent") is None


def test_save_and_load_mapping():
    """Проверяет сохранение и загрузку маппинга."""
    anon = Anonymizer()
    anon.anonymize(100)
    anon.anonymize(200)
    anon.anonymize(300)

    with tempfile.TemporaryDirectory() as tmpdir:
        anon.save_mapping(tmpdir)
        loaded = Anonymizer.load_mapping(tmpdir)
        assert loaded.anonymize(100) == "user_000001"
        assert loaded.anonymize(200) == "user_000002"
        assert loaded.deanonymize("user_000001") == 100


def test_delete_mapping():
    """Проверяет удаление файла маппинга."""
    anon = Anonymizer()
    anon.anonymize(100)
    with tempfile.TemporaryDirectory() as tmpdir:
        anon.save_mapping(tmpdir)
        assert anon.delete_mapping(tmpdir) is True
        assert not Path(tmpdir, "anonymization_mapping.json").exists()
        assert anon.delete_mapping(tmpdir) is False


def test_mapping_no_names():
    """Проверяет, что в маппинге нет имён пользователей."""
    anon = Anonymizer()
    anon.anonymize(12345)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = anon.save_mapping(tmpdir)
        with open(path) as f:
            data = json.load(f)
        # Only IDs and anon IDs, no names
        for key in data["mapping"]:
            assert not any(c.isalpha() for c in key if not key.startswith("user_"))
