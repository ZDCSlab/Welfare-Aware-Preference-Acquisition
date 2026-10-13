"""Pinned Roleplay source download and deterministic label-separated construction."""

from pathlib import Path
from collections import defaultdict
import json
import hashlib
import pyarrow.parquet as pq
from wapa.utils.io import read_rows, write_rows, write_json
from wapa.utils.validation import validate_inputs, validate_labels

SOURCE = "sher222/persona-iterative-responses"
REVISION = "731c2ebfbc2562b7bf206b089c03f4bc2c93f828"
CONTINENTS = {
    "Africa": "africa",
    "Asia": "asia",
    "Europe": "europe",
    "North America": "north_america",
    "South America": "south_america",
}


def download(output):
    from huggingface_hub import snapshot_download

    result = snapshot_download(
        repo_id=SOURCE,
        repo_type="dataset",
        revision=REVISION,
        local_dir=output,
        allow_patterns=["README.md", "data/*.parquet"],
    )

    return result


def group_for(traits):
    if traits.get("age") not in ("young adult", "middle aged", "elderly") or traits.get(
        "gender"
    ) not in ("male", "female"):
        return "heterogeneous"
    return CONTINENTS.get(traits.get("continent"), "heterogeneous")


def source_identity(source_split, persona_id, prompt_id):
    """Preserve the frozen source IDs and A/B orientation, independent of split names."""

    def digest(*parts):
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    key = digest(REVISION, source_split, persona_id, prompt_id)
    orientation = digest("roleplay_six_group_welfare_pipeline_v1", "ab-order", key)
    return key, int(orientation[:16], 16) % 2 == 0


def prepare(raw_root, split_plan, output, group="north_america", seed=42):
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    plan = json.loads(Path(split_plan).read_text())[group]
    lookup = {
        (source, prompt): split
        for split, spec in plan.items()
        for source in [spec["source_split"]]
        for prompt in spec["prompt_ids"]
    }
    result = defaultdict(list)
    labels = defaultdict(list)
    paths = sorted(Path(raw_root).glob("data/*.parquet"))
    if not paths:
        raise FileNotFoundError("raw_root must contain data/*.parquet")
    for path in paths:
        source = "train" if path.name.startswith("train-") else "test"
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                persona = row["score_persona"]
                question = row["question"]
                traits = persona.get("traits") or {}
                if (
                    group_for(traits) != group
                    or question["question_type"]["type"] != "global"
                ):
                    continue
                prompt_id = question["question_uuid"]
                split = lookup.get((source, prompt_id))
                if split is None:
                    continue
                user = persona["persona_uuid"]
                if user not in plan[split]["persona_ids"]:
                    raise ValueError("unexpected persona in fixed split plan")
                if row["x"] != question["question"] or row["level"] != user:
                    raise ValueError("source identity mismatch")
                key, preferred_a = source_identity(source, user, prompt_id)
                result[split].append(
                    {
                        "cell_id": key,
                        "profile_prompt_id": key,
                        "group_id": group,
                        "original_persona_id": user,
                        "prompt_id": prompt_id,
                        "pipeline_split": split,
                        "user_profile": persona["persona_description"],
                        "prompt": row["x"],
                        "response_a": row["yw"] if preferred_a else row["yl"],
                        "response_b": row["yl"] if preferred_a else row["yw"],
                    }
                )
                labels[split].append(
                    {"cell_id": key, "preference_label_y": int(preferred_a)}
                )
    summary = {}
    for split, spec in plan.items():
        inputs = sorted(
            result[split], key=lambda r: (r["original_persona_id"], r["prompt_id"])
        )
        summary[split] = validate_inputs(inputs, split=split, rectangular=True)
        if summary[split]["cells"] != len(spec["persona_ids"]) * len(
            spec["prompt_ids"]
        ):
            raise ValueError("source coverage mismatch")
        validate_labels(labels[split], {r["cell_id"] for r in inputs})
        write_rows(root / split / "inputs.parquet", inputs)
        write_rows(
            root / "labels" / (split + ".parquet"),
            sorted(labels[split], key=lambda r: r["cell_id"]),
        )
    write_rows(
        root / "heldout.parquet",
        read_rows(root / "test_seen_personas/inputs.parquet")
        + read_rows(root / "test_heldout_personas/inputs.parquet"),
    )
    write_rows(
        root / "labels/heldout.parquet",
        read_rows(root / "labels/test_seen_personas.parquet")
        + read_rows(root / "labels/test_heldout_personas.parquet"),
    )
    train_users = set(plan["acquisition"]["persona_ids"])
    if train_users != set(
        plan["test_seen_personas"]["persona_ids"]
    ) or train_users & set(plan["test_heldout_personas"]["persona_ids"]):
        raise ValueError("invalid persona split")
    write_json(
        root / "manifest.json",
        dict(
            source_dataset=SOURCE,
            source_revision=REVISION,
            seed=seed,
            group=group,
            counts=summary,
            identity="source IDs and fixed A/B orientation",
        ),
    )
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/roleplay")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--group",
        default="north_america",
        choices=[*CONTINENTS.values(), "heterogeneous"],
    )
    parser.add_argument("--split-plan", default="configs/dataset/roleplay_splits.json")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download(args.raw)
    print(prepare(args.raw, args.split_plan, args.output, args.group))


if __name__ == "__main__":
    main()
