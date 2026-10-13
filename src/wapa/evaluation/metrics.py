"""Observed Brier utility, fractional lower tail and evaluator-only oracle."""

import math
from collections import defaultdict

from ..acquisition.core import allocate, validate_evidence
from ..utils.categorical import probability_vector
from ..utils.validation import index_rows


def label_indices(rows, labels):
    truth = index_rows(labels)
    if set(truth) != {r["cell_id"] for r in rows}:
        raise ValueError("labels must cover exactly the evaluated pool")
    result = {}
    for row in rows:
        label = truth[row["cell_id"]]
        if "selected_option_id" in label:
            result[row["cell_id"]] = row["option_ids"].index(
                label["selected_option_id"]
            )
        else:
            y = label["preference_label_y"]
            if type(y) is not int or y not in (0, 1) or row["option_ids"] != ["A", "B"]:
                raise ValueError("invalid binary label")
            result[row["cell_id"]] = 0 if y == 1 else 1
    return result


def brier(p, y):
    return math.fsum((v - (k == y)) ** 2 for k, v in enumerate(p)) / 2


def tail_mean(values, fraction=0.1):
    ordered = sorted(values)
    mass = len(ordered) * fraction
    n = int(mass)
    return (
        math.fsum(ordered[:n]) + (mass - n) * ordered[n]
        if n < len(ordered)
        else math.fsum(ordered)
    ) / mass


def evaluate(rows, labels, selected_ids=(), target_group=()):
    index_rows(rows)
    ys = label_indices(rows, labels)
    selected_list = list(selected_ids)
    selected = set(selected_list)
    if len(selected_list) != len(selected) or not selected <= set(ys):
        raise ValueError("invalid selected IDs")
    utilities, accuracy = defaultdict(list), []
    for row in rows:
        p = probability_vector(row["target_probabilities"], row["option_ids"])
        y = ys[row["cell_id"]]
        queried = row["cell_id"] in selected
        utilities[row["original_persona_id"]].append(1 if queried else 1 - brier(p, y))
        accuracy.append(
            1 if queried else int(max(range(len(p)), key=p.__getitem__) == y)
        )
    u = {k: math.fsum(v) / len(v) for k, v in utilities.items()}
    target = list(target_group)
    if len(set(target)) != len(target) or set(target) - set(u):
        raise ValueError("duplicate or unknown target personas")
    return dict(
        mean=math.fsum(u.values()) / len(u),
        tail=tail_mean(u.values()),
        target=math.fsum(u[k] for k in target) / len(target) if target else None,
        accuracy=math.fsum(accuracy) / len(accuracy),
        persona_utilities=u,
        cells=len(rows),
        queried=len(selected),
    )


def oracle(rows, labels, budget, **parameters):
    """Full-information welfare optimum; never called by deployable selection."""
    validate_evidence(rows)
    ys = label_indices(rows, labels)
    gains = [
        brier(
            probability_vector(r["target_probabilities"], r["option_ids"]),
            ys[r["cell_id"]],
        )
        for r in rows
    ]
    return dict(
        method="welfare_optimal_oracle",
        budget=budget,
        uses_ground_truth=True,
        **allocate(rows, gains, budget, **parameters),
    )
