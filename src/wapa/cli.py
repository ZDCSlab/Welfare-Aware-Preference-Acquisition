"""Independent stages; acquisition commands never open evaluator labels."""

import argparse
import json
from pathlib import Path

from .utils.io import read_config, read_rows, write_rows, write_json


def run(command, config):
    if command == "collect":
        from .labelling.collect import collect

        return collect(config)
    if command == "aggregate":
        from .labelling.collect import aggregate

        rows = aggregate(
            read_rows(config["inputs"]),
            [r for p in config["predictions"] for r in read_rows(p)],
            config["target"],
            config["references"],
        )
        write_rows(config["output"], rows)
        return dict(cells=len(rows), output=config["output"])
    if command == "acquire":
        from .acquisition.core import select

        if "labels" in config:
            raise ValueError("acquisition config must not contain a label path")
        result = select(
            read_rows(config["evidence"]),
            config.get("method", "wapa"),
            config["budget"],
            config.get("seed", 42),
            **config.get("welfare", {}),
        )
        write_json(config["output"], result)
        return dict(selected=len(result["selected_cell_ids"]), output=config["output"])
    if command == "preferences":
        from .labelling.preferences import save

        inputs, evidence = read_rows(config["inputs"]), read_rows(config["evidence"])
        scheme = config.get("scheme", "selected")
        if scheme == "all_pseudo_labels":
            selected, labels = [], []
        elif scheme == "all_ground_truth_labels":
            selected, labels = (
                [r["cell_id"] for r in inputs],
                read_rows(config["labels"]),
            )
        elif scheme == "selected":
            selection = json.loads(Path(config["selection"]).read_text())
            if selection.get("uses_ground_truth"):
                raise ValueError("oracle is evaluation-only, not a training selector")
            selected = selection["selected_cell_ids"]
            labels = read_rows(config["labels"])
        else:
            raise ValueError("unknown training scheme")
        return save(config["output"], inputs, evidence, selected, labels)
    if command in ("train-prepare", "train"):
        from .training.dpo import prepare, launch

        return (prepare if command == "train-prepare" else launch)(config)
    if command == "generate":
        from .generation.generate import generate

        return generate(config)
    if command == "score-policy":
        from .evaluation.policy import score_policy

        return score_policy(config)
    if command in ("evaluate", "oracle"):
        from .evaluation.metrics import evaluate, oracle

        rows, labels = read_rows(config["evidence"]), read_rows(config["labels"])
        if command == "oracle":
            result = oracle(rows, labels, config["budget"], **config.get("welfare", {}))
        else:
            selected = (
                json.loads(Path(config["selection"]).read_text())["selected_cell_ids"]
                if config.get("selection")
                else []
            )
            result = evaluate(rows, labels, selected, config.get("target_group", []))
        write_json(config["output"], result)
        return result
    raise ValueError(f"unknown command: {command}")


def main():
    parser = argparse.ArgumentParser(prog="wapa", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "collect",
        "aggregate",
        "acquire",
        "preferences",
        "train-prepare",
        "train",
        "generate",
        "score-policy",
        "evaluate",
        "oracle",
    ):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import run_demo

        result = run_demo(args.output)
    else:
        result = run(args.command, read_config(args.config))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
