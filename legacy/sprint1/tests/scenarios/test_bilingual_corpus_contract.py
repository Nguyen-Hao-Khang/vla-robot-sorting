"""Offline contract checks for the Sprint 1 EN/VI semantic-command corpus."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "experiments" / "corpora" / "bilingual_commands.jsonl"
SCHEMA_PATH = ROOT / "experiments" / "corpora" / "bilingual_commands.schema.json"


def _load_jsonl() -> list[dict[str, Any]]:
    assert CORPUS_PATH.is_file(), f"missing corpus: {CORPUS_PATH}"
    records: list[dict[str, Any]] = []
    lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            record = json.loads(line)
            assert isinstance(record, dict), f"line {line_number}: record must be an object"
            records.append(record)
    return records


def _assert_schema(value: Any, schema: dict[str, Any], path: str) -> None:
    if "anyOf" in schema:
        errors: list[AssertionError] = []
        for option in schema["anyOf"]:
            try:
                _assert_schema(value, option, path)
                return
            except AssertionError as error:
                errors.append(error)
        raise AssertionError(f"{path}: value matches no anyOf option: {errors[0]}")

    expected_type = schema.get("type")
    expected_python_type = {
        "array": list,
        "boolean": bool,
        "null": type(None),
        "object": dict,
        "string": str,
    }.get(expected_type)
    if expected_python_type is not None:
        assert isinstance(value, expected_python_type), f"{path}: expected {expected_type}"

    if "enum" in schema:
        assert value in schema["enum"], f"{path}: {value!r} not in enum"
    if "const" in schema:
        assert value == schema["const"], f"{path}: expected {schema['const']!r}"
    if isinstance(value, str):
        if "minLength" in schema:
            assert len(value) >= schema["minLength"], f"{path}: shorter than minLength"
        if "pattern" in schema:
            assert re.fullmatch(schema["pattern"], value), f"{path}: pattern mismatch"
    if isinstance(value, list):
        if "minItems" in schema:
            assert len(value) >= schema["minItems"], f"{path}: fewer than minItems"
        if "items" in schema:
            for index, item in enumerate(value):
                _assert_schema(item, schema["items"], f"{path}[{index}]")
    if isinstance(value, dict):
        required = schema.get("required", [])
        for field in required:
            assert field in value, f"{path}: missing required field {field!r}"
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unexpected = set(value) - set(properties)
            assert not unexpected, f"{path}: unexpected fields {sorted(unexpected)!r}"
        for field, child_schema in properties.items():
            if field in value:
                _assert_schema(value[field], child_schema, f"{path}.{field}")


def test_every_jsonl_record_matches_the_project_schema() -> None:
    assert SCHEMA_PATH.is_file(), f"missing schema: {SCHEMA_PATH}"
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["type"] == "object"
    records = _load_jsonl()
    assert len(records) >= 64, "regression corpus must retain at least 64 cases"
    for record in records:
        _assert_schema(record, schema, record["case_id"])


def test_case_ids_languages_and_difficulty_coverage_are_stable() -> None:
    records = _load_jsonl()
    case_ids = [record["case_id"] for record in records]
    assert len(case_ids) == len(set(case_ids))

    languages = Counter(record["language"] for record in records)
    assert languages["en"] >= 20
    assert languages["vi"] >= 20
    assert languages["code_switch"] >= 6
    assert {record["difficulty"] for record in records} == {
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "L6",
        "L7",
        "L8",
    }


def test_behavior_specific_metadata_prevents_silent_guessing() -> None:
    records = _load_jsonl()
    for record in records:
        behavior = record["expected_behavior"]
        if behavior == "structured_success":
            assert record["operation"] == "pick_and_place"
            assert record["target"]["reference"]
            assert record["destination"]["reference"]
        elif behavior == "clarify":
            assert record["ambiguity_reason"]
        elif behavior == "reobserve":
            assert record["observation_reason"]
        elif behavior == "abort":
            assert record["safety_reason"]


def test_pairs_and_equivalence_groups_are_semantically_consistent() -> None:
    records = _load_jsonl()
    by_id = {record["case_id"]: record for record in records}
    equivalence_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        paired_case_id = record["paired_case_id"]
        if paired_case_id is not None:
            assert paired_case_id in by_id
            paired = by_id[paired_case_id]
            assert paired["paired_case_id"] == record["case_id"]
            assert paired["semantic_equivalence_id"] == record["semantic_equivalence_id"]
            assert paired["expected_behavior"] == record["expected_behavior"]
        if record["semantic_equivalence_id"] is not None:
            equivalence_groups[record["semantic_equivalence_id"]].append(record)

    for equivalence_id, group in equivalence_groups.items():
        normalized_intents = {record["normalized_intent_id"] for record in group}
        behaviors = {record["expected_behavior"] for record in group}
        assert len(normalized_intents) == 1, equivalence_id
        assert len(behaviors) == 1, equivalence_id
        if len(group) > 1:
            assert {"en", "vi"}.issubset({record["language"] for record in group})


def test_ambiguity_reobservation_abort_and_stress_categories_exist() -> None:
    records = _load_jsonl()
    behaviors = Counter(record["expected_behavior"] for record in records)
    assert behaviors["clarify"] >= 8
    assert behaviors["reobserve"] >= 4
    assert behaviors["abort"] >= 6

    tags = {tag for record in records for tag in record["tags"]}
    assert {
        "multiple_matching_targets",
        "multiple_matching_destinations",
        "missing_relation_reference",
        "vague_pronoun",
        "stale_reference",
        "conflicting_attributes",
        "unsafe_request",
        "unseen_wording",
        "unseen_category_wording",
    }.issubset(tags)
