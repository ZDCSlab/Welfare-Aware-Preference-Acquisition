"""Prepare the pinned, upstream-imputed OpinionQA acquisition benchmark."""

import csv
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from wapa.utils.io import write_text, write_json, write_rows, read_config
from wapa.utils.categorical import labels, validate

REVISION = "643d40f4ad71ee01b8023417e2d8396eab105f54"
DEMOGRAPHICS = (
    "F_AGECAT",
    "F_SEX",
    "F_EDUCCAT2",
    "F_PARTY_FINAL",
    "F_IDEO",
    "F_INCOME",
    "F_RELIG",
    "F_RACETHNMOD",
    "F_CREGION",
)


def prepare(config):
    seed = config.get("seed", 42)
    if (
        config["source_revision"] != REVISION
        or config["source_repository"] != "ZDCSlab/Group-Adaptive-Elicitation"
        or config["option_order"] != "source"
    ):
        raise ValueError("unsupported OpinionQA source or option-order contract")
    raw = Path(config["raw_root"])
    source = f"https://raw.githubusercontent.com/{config['source_repository']}/{REVISION}/dataset/opinionqa/"
    for relative in ("data/opinionqa_responses.csv", "codebook.jsonl"):
        path = raw / Path(relative).name
        if not path.exists():
            with urlopen(source + relative, timeout=60) as response:
                content = response.read().decode("utf-8")
            write_text(path, content)
    write_json(
        raw / "manifest.json",
        {
            "source": source,
            "revision": REVISION,
            "label_provenance": "upstream_processed_with_demographic_imputation",
            "imputation_fraction": None,
        },
    )
    codebook = {
        r["id"]: r
        for r in map(json.loads, (raw / "codebook.jsonl").read_text().splitlines())
    }
    questions = [key for key in codebook if key not in DEMOGRAPHICS]
    if len(questions) != config["questions"]:
        raise ValueError("question count changed")
    with (raw / "opinionqa_responses.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))

    def user_id(row):
        value = row["caseid"]
        # Source uses integral IDs with a literal .0 suffix; retain the original in users.json.
        result = value.removesuffix(".0")
        if not result.isdigit():
            raise ValueError("nonintegral source user ID")
        return result

    if len({user_id(r) for r in records}) != len(records):
        raise ValueError("duplicate source user ID")
    if any(r[key] not in codebook[key]["options"] for r in records for key in codebook):
        raise ValueError("source has missing/unknown answers or demographics")
    summaries = {}
    for region, expected in config["source_users"].items():
        group = sorted(
            [
                r
                for r in records
                if codebook["F_CREGION"]["options"][r["F_CREGION"]].lower() == region
            ],
            key=user_id,
        )
        if len(group) != expected:
            raise ValueError(f"source user count changed: {region}")
        n = config["users"][region]
        if n != (expected // 2 if region == "south" else expected):
            raise ValueError("only South half-user subsampling is supported")
        chosen = (
            set(
                np.random.default_rng(seed).choice(
                    [user_id(r) for r in group], n, replace=False
                )
            )
            if region == "south"
            else {user_id(r) for r in group}
        )
        group = [r for r in group if user_id(r) in chosen]
        inputs, evaluator = [], []
        for record in group:
            uid = user_id(record)
            profile = "\n".join(
                f"{codebook[k]['question'].capitalize()}: {codebook[k]['options'][record[k]]}"
                for k in DEMOGRAPHICS
            )
            for qid in questions:
                cell = f"opinionqa:{region}:{uid}:{qid}"
                options = codebook[qid]["options"]
                inputs.append(
                    {
                        "cell_id": cell,
                        "group_id": region,
                        "original_persona_id": uid,
                        "prompt_id": qid,
                        "profile_prompt_id": cell,
                        "pipeline_split": "acquisition",
                        "user_profile": profile,
                        "prompt": codebook[qid]["question"],
                        "benchmark_id": "opinionqa",
                        "task_type": "categorical_choice",
                        "option_ids": list(options),
                        "option_texts": list(options.values()),
                    }
                )
                evaluator.append({"cell_id": cell, "selected_option_id": record[qid]})
        counts = validate(inputs, split="acquisition", rectangular=True)
        labels(evaluator, inputs)
        root = Path(config["output"]) / region
        write_rows(root / "acquisition/inputs.parquet", inputs)
        write_rows(root / "labels/acquisition.parquet", evaluator)
        target = sorted(
            np.random.default_rng(seed)
            .choice(sorted(chosen), max(1, n // 10), replace=False)
            .tolist()
        )
        write_json(
            root / "users.json",
            {
                "seed": seed,
                "sampling": "sorted IDs; numpy.default_rng(seed).choice without replacement",
                "source_users": expected,
                "users": [
                    {"user_id": user_id(r), "source_caseid": r["caseid"]} for r in group
                ],
                "random10_target": target,
            },
        )
        write_json(
            root / "manifest.json",
            {
                "benchmark_id": "opinionqa",
                "task_type": "categorical_choice",
                "seed": seed,
                "source_revision": REVISION,
                "counts": counts,
                "option_order": "source",
                "label_provenance": "upstream_processed_with_demographic_imputation",
                "imputation_fraction": None,
                "supported_stages": ["dataset", "evidence", "acquisition"],
            },
        )
        summaries[region] = counts
    return summaries


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset/opinionqa.yaml")
    args = parser.parse_args()
    print(prepare(read_config(args.config)))


if __name__ == "__main__":
    main()
