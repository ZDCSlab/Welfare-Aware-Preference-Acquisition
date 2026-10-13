"""Exact marginal allocation for frozen, label-free preference probabilities."""

import heapq
import math
import random
from collections import defaultdict

from .welfare import resolve_parameters, social_weight
from ..utils.categorical import probability_vector
from ..utils.validation import index_rows, FORBIDDEN

METHODS = ("wapa", "random", "entropy", "reference_disagreement")


def validate_evidence(rows):
    index_rows(rows)
    prompts = defaultdict(set)
    for row in rows:
        if any(any(word in key.lower() for word in FORBIDDEN) for key in row):
            raise ValueError("acquisition evidence must not contain labels")
        for name in ("cell_id", "original_persona_id", "prompt_id"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"missing {name}")
        ids = row["option_ids"]
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("distinct options required")
        probability_vector(row["target_probabilities"], ids)
        refs = row["reference_probabilities"]
        if not refs:
            raise ValueError("at least one reference required")
        for q in refs:
            probability_vector(q, ids)
        user, prompt = row["original_persona_id"], row["prompt_id"]
        if prompt in prompts[user]:
            raise ValueError("duplicate persona/prompt")
        prompts[user].add(prompt)
    if len({frozenset(p) for p in prompts.values()}) != 1:
        raise ValueError("acquisition requires a complete persona × prompt rectangle")


def expected_brier(p, q):
    """Expected normalized multiclass Brier; binary reduces to q(1-p)^2+(1-q)p^2."""
    return max(
        0.0,
        min(
            1.0,
            (
                1
                + math.fsum(x * x for x in p)
                - 2 * math.fsum(x * y for x, y in zip(p, q, strict=True))
            )
            / 2,
        ),
    )


def cell_score(row, method):
    ids = row["option_ids"]
    p = probability_vector(row["target_probabilities"], ids)
    refs = [probability_vector(q, ids) for q in row["reference_probabilities"]]
    q = [math.fsum(v[k] for v in refs) / len(refs) for k in range(len(p))]
    if method == "wapa":
        return expected_brier(p, q)
    if method == "entropy":
        return min(1.0, -math.fsum(x * math.log(x) for x in p if x) / math.log(len(p)))
    if method == "reference_disagreement":
        # Mean squared distance to ensemble mean, normalized by simplex maximum.
        variance = math.fsum(
            math.fsum((x - y) ** 2 for x, y in zip(v, q)) for v in refs
        ) / len(refs)
        return min(1.0, variance / (1 - 1 / len(p)))
    raise ValueError(f"unknown scoring method: {method}")


def exact_gain(x, delta, rho, weight):
    if rho == 1:
        return weight * delta
    ratio = math.log1p(delta / x)
    return (
        weight * ratio if rho == 0 else weight * x**rho * math.expm1(rho * ratio) / rho
    )


def allocate(rows, gains, budget, *, first_order=False, **parameters):
    """Merge decreasing per-persona marginal sequences using a heap."""
    if type(budget) is not int or not 0 <= budget <= len(rows):
        raise ValueError("budget must be an integer between 0 and the pool size")
    if len(gains) != len(rows) or any(
        not math.isfinite(v) or not 0 <= v <= 1 for v in gains
    ):
        raise ValueError("one bounded gain per cell required")
    grouped = defaultdict(list)
    for row, gain in zip(rows, gains, strict=True):
        grouped[row["original_persona_id"]].append((row, gain))
    parameters = resolve_parameters(grouped, **parameters)
    sizes = {len(v) for v in grouped.values()}
    if len(sizes) != 1:
        raise ValueError("equal per-persona pool sizes required")
    m = sizes.pop()
    ordered = {
        u: sorted(v, key=lambda x: (-x[1], x[0]["cell_id"])) for u, v in grouped.items()
    }
    utility = {u: 1 - math.fsum(g for _, g in v) / m for u, v in ordered.items()}
    initial = dict(utility)
    pointers = dict.fromkeys(ordered, 0)
    heap = []

    def push(u):
        row, gain = ordered[u][pointers[u]]
        x = utility[u] + parameters["epsilon"]
        weight = social_weight(u, parameters)
        priority = (
            weight * x ** (parameters["rho"] - 1) * gain / m
            if first_order
            else exact_gain(x, gain / m, parameters["rho"], weight)
        )
        heapq.heappush(heap, (-priority, row["cell_id"], u))

    for u in ordered:
        push(u)
    selected, trace = [], []
    for _ in range(budget):
        neg, cell, u = heapq.heappop(heap)
        _, gain = ordered[u][pointers[u]]
        selected.append(cell)
        trace.append(
            {
                "cell_id": cell,
                "persona_id": u,
                "priority": -neg,
                "utility_before": utility[u],
            }
        )
        utility[u] = min(1.0, utility[u] + gain / m)
        pointers[u] += 1
        if pointers[u] < m:
            push(u)
    return dict(
        selected_cell_ids=selected,
        allocator="first_order" if first_order else "exact_marginal",
        parameters=parameters,
        trace=trace,
        initial_utilities=initial,
        final_utilities=utility,
    )


def select(rows, method="wapa", budget=0, seed=42, **parameters):
    validate_evidence(rows)
    if method not in METHODS:
        raise ValueError(f"choose one of {METHODS}")
    if type(budget) is not int or not 0 <= budget <= len(rows):
        raise ValueError("invalid budget")
    if method == "random":
        resolve_parameters({r["original_persona_id"] for r in rows}, **parameters)
        selected = random.Random(seed).sample(
            sorted(r["cell_id"] for r in rows), budget
        )
        result = dict(selected_cell_ids=selected, parameters={}, trace=[])
    else:
        result = allocate(
            rows,
            [cell_score(r, method) for r in rows],
            budget,
            first_order=method != "wapa",
            **parameters,
        )
    return dict(method=method, seed=seed, budget=budget, **result)
