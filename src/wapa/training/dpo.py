"""Prepare and launch soft-label DPO through a patched LlamaFactory installation."""

import json
import math
import subprocess
from pathlib import Path

from ..utils.io import read_rows, write_text

REVISION = "7fcf5b3b130e5713b52415bb7404c476fada9c8c"


def prepare(config):
    import yaml

    root = Path(config["dataset_dir"]).resolve()
    rows = read_rows(root / "train.json")
    if not rows or len({r["cell_id"] for r in rows}) != len(rows):
        raise ValueError("nonempty, unique training comparisons required")
    for row in rows:
        q = row["preference_probability"]
        if (
            isinstance(q, bool)
            or not isinstance(q, (int, float))
            or not math.isfinite(q)
            or not 0 <= q <= 1
        ):
            raise ValueError("invalid soft label")
        if (
            not row["chosen"].strip()
            or not row["rejected"].strip()
            or row["chosen"] == row["rejected"]
        ):
            raise ValueError("invalid response pair")
    columns = json.loads((root / "dataset_info.json").read_text())["preferences"][
        "columns"
    ]
    if any(
        columns.get(k) != k
        for k in ("system", "prompt", "chosen", "rejected", "preference_probability")
    ):
        raise ValueError("training columns do not match prepared preferences")
    settings = dict(config.get("parameters", {}))
    forbidden = (
        "stage",
        "do_train",
        "dataset",
        "dataset_dir",
        "model_name_or_path",
        "output_dir",
        "dpo_soft_labels",
        "pref_loss",
    )
    if any(k in settings for k in forbidden):
        raise ValueError(
            "set model, dataset and output in top-level training configuration"
        )
    if (
        any(
            settings.get(k, 0) != 0
            for k in ("dpo_label_smoothing", "pref_ftx", "pref_bco_weight", "val_size")
        )
        or settings.get("ld_alpha") is not None
        or settings.get("max_samples") is not None
    ):
        raise ValueError("soft DPO requires full-support plain sigmoid training")
    seed = config.get("seed", 42)
    settings.update(
        stage="dpo",
        do_train=True,
        dataset="preferences",
        dataset_dir=str(root),
        model_name_or_path=config["base_model"],
        output_dir=str(Path(config["checkpoint_output"]).resolve()),
        dpo_soft_labels=True,
        pref_loss="sigmoid",
        seed=seed,
        data_seed=seed,
    )
    settings.setdefault("template", "qwen3_nothink")
    settings.setdefault("finetuning_type", "lora")
    settings.setdefault("pref_beta", 0.1)
    settings.setdefault("num_train_epochs", 3)
    settings.setdefault("overwrite_output_dir", False)
    output = Path(config["output"]) / "train.yaml"
    write_text(output, yaml.safe_dump(settings, sort_keys=False))
    return dict(training_config=str(output), cells=len(rows))


def launch(config):
    # Feature detection works with an externally installed, patched checkout.
    from llamafactory.hparams import FinetuningArguments

    if "dpo_soft_labels" not in FinetuningArguments.__dataclass_fields__:
        raise RuntimeError(
            "apply patches/llamafactory-soft-label-dpo.patch before training"
        )
    checkpoint = Path(config["checkpoint_output"])
    if (
        checkpoint.exists()
        and any(checkpoint.iterdir())
        and not config.get("parameters", {}).get("resume_from_checkpoint")
    ):
        raise FileExistsError(
            "nonempty checkpoint directory; configure resume_from_checkpoint or choose a new output"
        )
    result = prepare(config)
    subprocess.run(
        ["llamafactory-cli", "train", str(Path(result["training_config"]).resolve())],
        check=True,
    )
    return result
