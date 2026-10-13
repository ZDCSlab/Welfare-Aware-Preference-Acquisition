import itertools
import math
import random

import pytest

from wapa.acquisition.core import select, cell_score, expected_brier, METHODS
from wapa.evaluation.metrics import oracle, evaluate, tail_mean, brier
from wapa.labelling.collect import aggregate, parse_response
from wapa.labelling.preferences import build
from wapa.training.dpo import prepare
from wapa.evaluation.policy import implicit_probability
from wapa.demo import run_demo


def fixture():
    inputs, evidence, labels = [], [], []
    rng = random.Random(9)
    for i in range(3):
        for j in range(3):
            key = f"{i}:{j}"
            p, q = rng.random(), rng.random()
            inputs.append(
                dict(
                    cell_id=key,
                    group_id="test",
                    original_persona_id=str(i),
                    prompt_id=str(j),
                    profile_prompt_id=key,
                    pipeline_split="acquisition",
                    user_profile="Prefers quiet places",
                    prompt="Where to go?",
                    response_a="A response",
                    response_b="B response",
                )
            )
            evidence.append(
                dict(
                    cell_id=key,
                    original_persona_id=str(i),
                    prompt_id=str(j),
                    option_ids=["A", "B"],
                    target_probabilities=[p, 1 - p],
                    reference_probabilities=[[q, 1 - q], [0.7, 0.3]],
                )
            )
            labels.append(dict(cell_id=key, preference_label_y=(i + j) % 2))
    return inputs, evidence, labels


def objective(rows, gains, chosen, rho, gamma):
    users = sorted({r["original_persona_id"] for r in rows})
    total = 0
    for u in users:
        remaining = [
            g
            for r, g in zip(rows, gains)
            if r["original_persona_id"] == u and r["cell_id"] not in chosen
        ]
        x = 1 - sum(remaining) / 3 + 0.5
        h = math.log(x) if rho == 0 else x**rho / rho
        total += (gamma if u == "0" else 1) * h
    return total


@pytest.mark.parametrize("rho", [-5, 0, 0.5, 1])
@pytest.mark.parametrize("gamma", [1, 2])
def test_wapa_and_oracle_match_exhaustive_optimum(rho, gamma):
    _, rows, labels = fixture()
    parameters = dict(rho=rho, gamma=gamma, target_group=["0"])
    for result, gains in [
        (select(rows, "wapa", 4, **parameters), [cell_score(r, "wapa") for r in rows]),
        (
            oracle(rows, labels, 4, **parameters),
            [
                brier(r["target_probabilities"], 0 if y["preference_label_y"] else 1)
                for r, y in zip(rows, labels)
            ],
        ),
    ]:
        actual = objective(rows, gains, set(result["selected_cell_ids"]), rho, gamma)
        best = max(
            objective(rows, gains, {rows[i]["cell_id"] for i in indices}, rho, gamma)
            for indices in itertools.combinations(range(len(rows)), 4)
        )
        assert actual == pytest.approx(best, abs=1e-12)


def test_binary_and_multiclass_brier():
    p, q = 0.3, 0.8
    assert expected_brier([p, 1 - p], [q, 1 - q]) == pytest.approx(
        q * (1 - p) ** 2 + (1 - q) * p * p
    )
    p, q = [0.1, 0.2, 0.7], [0.5, 0.2, 0.3]
    assert expected_brier(p, q) == pytest.approx(
        sum(q[k] * brier(p, k) for k in range(3))
    )


def test_multiclass_methods_and_seed():
    _, rows, _ = fixture()
    for r in rows:
        r.update(
            option_ids=["A", "B", "C"],
            target_probabilities=[0.1, 0.3, 0.6],
            reference_probabilities=[[0.2, 0.5, 0.3], [0.8, 0.1, 0.1]],
        )
    for method in METHODS:
        result = select(rows, method, 4, seed=123, rho=-5)
        assert len(set(result["selected_cell_ids"])) == 4
        assert result == select(list(reversed(rows)), method, 4, seed=123, rho=-5)
    assert select(rows, "random", 4, seed=123) != select(rows, "random", 4, seed=42)


def test_no_label_leakage_and_invalid_pool():
    _, rows, _ = fixture()
    rows[0]["preference_label_y"] = 1
    with pytest.raises(ValueError, match="labels"):
        select(rows, "wapa", 2)
    del rows[0]["preference_label_y"]
    with pytest.raises(ValueError, match="rectangle"):
        select(rows[:-1], "wapa", 2)
    with pytest.raises(ValueError):
        select(rows, "wapa", 2, gamma=2)


def test_soft_label_orientation_and_isolation():
    inputs, rows, labels = fixture()
    rows[0]["target_probabilities"] = [0.2, 0.8]
    rows[1]["target_probabilities"] = [0.8, 0.2]
    result, audit = build(inputs, rows, [rows[2]["cell_id"]], labels)
    assert result[0]["chosen"] == "B response"
    assert result[0]["preference_probability"] == 0.8
    assert result[1]["chosen"] == "A response"
    assert result[1]["preference_probability"] == 0.8
    assert result[2]["preference_probability"] == 1
    changed = [
        dict(r, preference_label_y=1 - r["preference_label_y"])
        if r["cell_id"] != rows[2]["cell_id"]
        else r
        for r in labels
    ]
    assert build(inputs, rows, [rows[2]["cell_id"]], changed) == (result, audit)
    assert result[0]["prompt"] == "Where to go?"
    assert "quiet places" in result[0]["system"]


def test_fractional_tail_and_full_query():
    assert tail_mean([0.2, 0.4, 0.6], 0.5) == pytest.approx((0.2 + 0.5 * 0.4) / 1.5)
    _, rows, labels = fixture()
    report = evaluate(rows, labels, [r["cell_id"] for r in rows], ["0"])
    assert (
        report["mean"] == report["tail"] == report["target"] == report["accuracy"] == 1
    )


def test_aggregation_order_and_completeness():
    inputs, _, _ = fixture()
    records = []
    for row in inputs:
        for model in ("target", "ref"):
            for order, p in [("original", 0.8), ("swapped", 0.4)]:
                records.append(
                    dict(
                        cell_id=row["cell_id"],
                        model_id=model,
                        order=order,
                        option_ids=["A", "B"],
                        probabilities=[p, 1 - p],
                    )
                )
    rows = aggregate(inputs, records, "target", ["ref"])
    assert rows[0]["target_probabilities"] == pytest.approx([0.7, 0.3])
    with pytest.raises(ValueError, match="incomplete"):
        aggregate(inputs, records[:-1], "target", ["ref"])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate(inputs, records + [records[0]], "target", ["ref"])


def test_reference_parser():
    assert parse_response('{"choice":"B","probability_a":0.2}', ["A", "B"], False) == [
        0.2,
        0.8,
    ]
    with pytest.raises(ValueError):
        parse_response('{"choice":"A","probability_a":0.2}', ["A", "B"], False)
    with pytest.raises(ValueError):
        parse_response('{"choice":"A","probability_a":true}', ["A", "B"], False)
    assert parse_response(
        '{"choice":"C","probabilities":{"A":0.1,"B":0.2,"C":0.7}}',
        ["A", "B", "C"],
        True,
    ) == [0.1, 0.2, 0.7]


def test_baselines_use_own_normalized_state():
    _, rows, _ = fixture()
    for method in ("entropy", "reference_disagreement"):
        result = select(rows, method, 1, rho=-5, gamma=2, target_group=["0"])
        scores = {r["cell_id"]: cell_score(r, method) for r in rows}
        risks = {
            u: sum(scores[r["cell_id"]] for r in rows if r["original_persona_id"] == u)
            / 3
            for u in ("0", "1", "2")
        }
        priorities = {
            r["cell_id"]: (2 if r["original_persona_id"] == "0" else 1)
            * (1 - risks[r["original_persona_id"]] + 0.5) ** -6
            * scores[r["cell_id"]]
            / 3
            for r in rows
        }
        assert result["selected_cell_ids"] == [max(priorities, key=priorities.get)]


def test_implicit_score():
    assert implicit_probability(-2, -4, -3, -3) == pytest.approx(
        1 / (1 + math.exp(-0.2))
    )
    assert implicit_probability(-3, -4, -3, -4) == 0.5


def test_demo_and_training_config(tmp_path):
    import yaml

    root = tmp_path / "demo"
    run_demo(root)
    result = prepare(
        dict(
            dataset_dir=str(root / "preferences"),
            base_model="base/model",
            checkpoint_output=str(tmp_path / "checkpoints"),
            output=str(tmp_path / "training"),
        )
    )
    cfg = yaml.safe_load(open(result["training_config"]))
    assert cfg["dpo_soft_labels"] is True
    assert cfg["model_name_or_path"] == "base/model"
    assert cfg["template"] == "qwen3_nothink"
