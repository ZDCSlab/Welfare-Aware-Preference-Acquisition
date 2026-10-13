"""Label-free categorical contracts; option IDs retain their source order."""

import math

from .validation import FORBIDDEN, index_rows

FIELDS = (
    "cell_id",
    "group_id",
    "original_persona_id",
    "prompt_id",
    "profile_prompt_id",
    "pipeline_split",
    "user_profile",
    "prompt",
    "benchmark_id",
    "task_type",
    "option_ids",
    "option_texts",
)


def validate(rows, *, split=None, rectangular=False):
    index_rows(rows)
    users = {}
    for row in rows:
        if set(row) != set(FIELDS):
            raise ValueError(
                "categorical inputs require exactly the label-free contract fields"
            )
        if any(any(s in k.lower() for s in FORBIDDEN) for k in row):
            raise ValueError("label-bearing input")
        for key in FIELDS[:-2]:
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"invalid categorical {key}")
        if row["task_type"] != "categorical_choice":
            raise ValueError("wrong task type")
        if split and row["pipeline_split"] != split:
            raise ValueError("wrong split")
        ids, texts = row["option_ids"], row["option_texts"]
        if (
            not isinstance(ids, list)
            or not isinstance(texts, list)
            or not 2 <= len(ids) <= 5
            or len(ids) != len(texts)
            or ids != list("ABCDE"[: len(ids)])
            or any(not isinstance(t, str) or not t.strip() for t in texts)
            or len(set(texts)) != len(texts)
        ):
            raise ValueError("invalid ordered options")
        seen = users.setdefault(row["original_persona_id"], set())
        if row["prompt_id"] in seen:
            raise ValueError("duplicate user/question")
        seen.add(row["prompt_id"])
    if rectangular and len({frozenset(v) for v in users.values()}) != 1:
        raise ValueError("expected complete user/question rectangle")
    return {
        "cells": len(rows),
        "personas": len(users),
        "prompts": len({r["prompt_id"] for r in rows}),
    }


def probability_vector(values, option_ids):
    if (
        not isinstance(values, (list, tuple))
        or len(values) != len(option_ids)
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 <= v <= 1
            for v in values
        )
    ):
        raise ValueError("invalid categorical probability vector")
    total = math.fsum(values)
    if not math.isclose(total, 1, rel_tol=0, abs_tol=1e-6 + 1e-12):
        raise ValueError("categorical probabilities must sum to one")
    # Remove only accepted serialization roundoff; raw output remains in replay.
    return [float(v / total) for v in values]


def labels(rows, inputs):
    index = index_rows(inputs)
    result = index_rows(rows)
    if set(index) != set(result):
        raise ValueError("categorical label coverage mismatch")
    for key, row in result.items():
        if (
            set(row) != {"cell_id", "selected_option_id"}
            or row["selected_option_id"] not in index[key]["option_ids"]
        ):
            raise ValueError("invalid categorical label")
    return {key: row["selected_option_id"] for key, row in result.items()}
