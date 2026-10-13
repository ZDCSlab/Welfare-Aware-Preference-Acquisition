"""Combine queried human labels with unqueried base-model probabilities."""

from pathlib import Path

from ..utils.io import write_rows, write_json
from ..utils.validation import validate_inputs, index_rows
from ..utils.categorical import probability_vector
from ..utils.prompts import generation_messages


def build(inputs, evidence, selected_ids=(), query_labels=()):
    validate_inputs(inputs, split="acquisition", rectangular=True)
    if any(r.get("task_type") == "categorical_choice" for r in inputs):
        raise ValueError("soft DPO requires Roleplay response pairs")
    ei = index_rows(evidence)
    if set(ei) != {r["cell_id"] for r in inputs}:
        raise ValueError("evidence must cover training inputs exactly")
    selected_ids = list(selected_ids)
    selected = set(selected_ids)
    if len(selected) != len(selected_ids) or not selected <= set(ei):
        raise ValueError("invalid selected cells")
    # Only selected labels enter construction; additional evaluator labels are ignored.
    queried = [r for r in query_labels if r["cell_id"] in selected]
    labels = index_rows(queried) if queried else {}
    if set(labels) != selected:
        raise ValueError("missing queried labels")
    result, audit = [], []
    for row in inputs:
        key = row["cell_id"]
        erow = ei[key]
        if any(erow.get(k) != row[k] for k in ("original_persona_id", "prompt_id")):
            raise ValueError("evidence persona/prompt does not match input")
        if erow["option_ids"] != ["A", "B"]:
            raise ValueError("pairwise probability coordinates must be A, B")
        p = probability_vector(erow["target_probabilities"], ["A", "B"])[0]
        if key in selected:
            y = labels[key]["preference_label_y"]
            if type(y) is not int or y not in (0, 1):
                raise ValueError("queried labels must be binary")
            probability = 1.0
        else:
            y = int(p >= 0.5)
            probability = p if y else 1 - p
        result.append(
            dict(
                cell_id=key,
                system=generation_messages(row)[0]["content"],
                prompt=row["prompt"],
                chosen=row["response_a"] if y else row["response_b"],
                rejected=row["response_b"] if y else row["response_a"],
                preference_probability=probability,
            )
        )
        audit.append(
            dict(
                cell_id=key,
                source="human" if key in selected else "pseudo",
                chosen_is_a=bool(y),
                preference_probability=probability,
            )
        )
    return result, audit


def save(output, inputs, evidence, selected_ids=(), query_labels=()):
    rows, audit = build(inputs, evidence, selected_ids, query_labels)
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("choose an empty preference output directory")
    write_rows(root / "train.json", rows)
    write_rows(root / "label_sources.jsonl", audit)
    write_json(
        root / "dataset_info.json",
        {
            "preferences": dict(
                file_name="train.json",
                formatting="alpaca",
                ranking=True,
                columns={
                    k: k
                    for k in (
                        "system",
                        "prompt",
                        "chosen",
                        "rejected",
                        "preference_probability",
                    )
                },
            )
        },
    )
    return dict(
        cells=len(rows),
        queried=sum(r["source"] == "human" for r in audit),
        output=str(root),
    )
