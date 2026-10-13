"""CPU-only synthetic example; no API requests or model training."""

from pathlib import Path
from .acquisition.core import select, METHODS
from .evaluation.metrics import evaluate, oracle
from .labelling.preferences import save
from .utils.io import write_json, write_rows


def run_demo(output):
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("choose an empty demo directory")
    inputs, evidence, labels = [], [], []
    for i in range(4):
        for j in range(5):
            key = f"u{i}:q{j}"
            p = 0.15 + 0.13 * i + 0.07 * j
            q = 0.8 if (i + j) % 2 else 0.2
            inputs.append(
                dict(
                    cell_id=key,
                    group_id="demo",
                    original_persona_id=f"u{i}",
                    prompt_id=f"q{j}",
                    profile_prompt_id=key,
                    pipeline_split="acquisition",
                    user_profile=f"Persona {i}",
                    prompt=f"Question {j}",
                    response_a="Answer A",
                    response_b="Answer B",
                )
            )
            evidence.append(
                dict(
                    cell_id=key,
                    original_persona_id=f"u{i}",
                    prompt_id=f"q{j}",
                    option_ids=["A", "B"],
                    target_probabilities=[p, 1 - p],
                    reference_probabilities=[[q, 1 - q], [1 - q, q], [0.7, 0.3]],
                )
            )
            labels.append(dict(cell_id=key, preference_label_y=int((i + j) % 2)))
    write_rows(root / "inputs.json", inputs)
    write_rows(root / "evidence.json", evidence)
    write_rows(root / "labels.json", labels)
    parameters = dict(rho=-5, gamma=2, epsilon=0.5, target_group=["u0"])
    report = {"no_queries": evaluate(evidence, labels, target_group=["u0"])}
    # Freeze all deployable selections before invoking the label-reading evaluator.
    selections = {m: select(evidence, m, 4, **parameters) for m in METHODS}
    for method, selection in selections.items():
        write_json(root / f"{method}.json", selection)
        report[method] = evaluate(
            evidence, labels, selection["selected_cell_ids"], ["u0"]
        )
    optimum = oracle(evidence, labels, 4, **parameters)
    write_json(root / "oracle.json", optimum)
    report["oracle"] = evaluate(evidence, labels, optimum["selected_cell_ids"], ["u0"])
    save(
        root / "preferences",
        inputs,
        evidence,
        selections["wapa"]["selected_cell_ids"],
        labels,
    )
    save(root / "all_pseudo_labels", inputs, evidence)
    save(
        root / "all_ground_truth_labels",
        inputs,
        evidence,
        [r["cell_id"] for r in inputs],
        labels,
    )
    write_json(root / "metrics.json", report)
    return dict(output=str(root), methods=list(selections), cells=len(inputs), budget=4)
