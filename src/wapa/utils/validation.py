"""Validate identifiers and keep evaluator labels out of model inputs."""

from __future__ import annotations
from collections import defaultdict

INPUT_FIELDS = (
    "cell_id",
    "group_id",
    "original_persona_id",
    "prompt_id",
    "profile_prompt_id",
    "pipeline_split",
    "user_profile",
    "prompt",
    "response_a",
    "response_b",
)
FORBIDDEN = ("label", "preferred", "rejected", "chosen", "winner", "is_queried")


def index_rows(rows):
    result = {}
    for row in rows:
        key = str(row["cell_id"])
        if key in result:
            raise ValueError(f"duplicate cell: {key}")
        result[key] = row
    if not result:
        raise ValueError("empty cell universe")
    return result


def validate_inputs(rows, *, split=None, rectangular=False):
    if rows and rows[0].get("task_type") == "categorical_choice":
        from .categorical import validate

        return validate(rows, split=split, rectangular=rectangular)
    index_rows(rows)
    for row in rows:
        if any(any(s in k.lower() for s in FORBIDDEN) for k in row):
            raise ValueError("label-bearing field in model inputs")
        for field in INPUT_FIELDS:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"missing/non-string {field}")
        if row["response_a"] == row["response_b"]:
            raise ValueError("identical pair")
        if split and row["pipeline_split"] != split:
            raise ValueError("wrong split")
    if rectangular:
        groups = defaultdict(set)
        for row in rows:
            groups[row["original_persona_id"]].add(row["prompt_id"])
        if len({frozenset(x) for x in groups.values()}) != 1:
            raise ValueError("expected a complete persona × prompt rectangle")
        if sum(map(len, groups.values())) != len(rows):
            raise ValueError("duplicate persona/prompt")
    return {
        "cells": len(rows),
        "personas": len({r["original_persona_id"] for r in rows}),
        "prompts": len({r["prompt_id"] for r in rows}),
    }


def validate_labels(rows, expected=None):
    index = index_rows(rows)
    if expected is not None and set(index) != set(expected):
        raise ValueError("label coverage mismatch")
    for row in rows:
        if type(row.get("preference_label_y")) is not int or row[
            "preference_label_y"
        ] not in (0, 1):
            raise ValueError("labels must be binary integers")
    return {k: r["preference_label_y"] for k, r in index.items()}
