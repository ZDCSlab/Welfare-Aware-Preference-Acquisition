"""Resolve a fractional budget and fixed Target into runnable stage configurations."""

import argparse
import json
import math
from pathlib import Path

import yaml

from wapa.acquisition.core import METHODS, validate_evidence
from wapa.acquisition.welfare import resolve_parameters
from wapa.utils.io import read_config, read_rows, write_text


def prepare(config):
    rows = read_rows(config["evidence"])
    validate_evidence(rows)
    fraction = config["budget_fraction"]
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("budget_fraction must be between zero and one")
    budget = math.floor(len(rows) * fraction)
    target = json.loads(Path(config["target_file"]).read_text())["target_group"]
    welfare = resolve_parameters(
        {r["original_persona_id"] for r in rows},
        target_group=target,
        **config["welfare"],
    )
    root = Path(config["output"])
    configs = root / "configs"

    def save(name, value):
        write_text(configs / f"{name}.yaml", yaml.safe_dump(value, sort_keys=False))

    common = dict(
        evidence=config["evidence"], labels=config["labels"], target_group=target
    )
    for method in METHODS:
        selection = str(root / method / "selection.json")
        save(
            f"acquire_{method}",
            dict(
                evidence=config["evidence"],
                method=method,
                budget=budget,
                seed=config.get("seed", 42),
                welfare=welfare,
                output=selection,
            ),
        )
        save(
            f"evaluate_{method}",
            dict(
                **common,
                selection=selection,
                output=str(root / method / "metrics.json"),
            ),
        )
        if config.get("preferences", False):
            save(
                f"preferences_{method}",
                dict(
                    inputs=config["inputs"],
                    evidence=config["evidence"],
                    labels=config["labels"],
                    selection=selection,
                    scheme="selected",
                    output=str(root / method / "preferences"),
                ),
            )
    save(
        "oracle",
        dict(
            evidence=config["evidence"],
            labels=config["labels"],
            budget=budget,
            welfare=welfare,
            output=str(root / "oracle/selection.json"),
        ),
    )
    save(
        "evaluate_oracle",
        dict(
            **common,
            selection=str(root / "oracle/selection.json"),
            output=str(root / "oracle/metrics.json"),
        ),
    )
    save(
        "evaluate_no_queries",
        dict(**common, output=str(root / "no_queries/metrics.json")),
    )
    if config.get("preferences", False):
        for scheme in ("all_pseudo_labels", "all_ground_truth_labels"):
            save(
                f"preferences_{scheme}",
                dict(
                    inputs=config["inputs"],
                    evidence=config["evidence"],
                    labels=config["labels"],
                    scheme=scheme,
                    output=str(root / scheme / "preferences"),
                ),
            )
    return dict(
        cells=len(rows),
        budget=budget,
        target_personas=len(target),
        configs=str(configs),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(read_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
