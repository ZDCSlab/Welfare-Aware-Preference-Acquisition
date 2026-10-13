"""DPO-implied probabilities from full response likelihoods under policy and base."""

import math

from ..utils.io import Cache, read_rows, write_rows
from ..utils.prompts import generation_messages
from ..utils.validation import validate_inputs


def implicit_probability(policy_a, policy_b, reference_a, reference_b, beta=0.1):
    values = (policy_a, policy_b, reference_a, reference_b, beta)
    if not all(math.isfinite(v) for v in values) or beta <= 0:
        raise ValueError("finite log likelihoods and positive beta required")
    z = beta * ((policy_a - reference_a) - (policy_b - reference_b))
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def score_policy(config):
    rows = read_rows(config["inputs"])
    validate_inputs(rows)
    if any(r.get("task_type") == "categorical_choice" for r in rows):
        raise ValueError("DPO evaluation requires response pairs")
    specs = [config["reference"], config["policy"]]
    cache = Cache(config["output"] + ".cache.jsonl")
    likelihoods = []
    # Sequential model loading also supports two full checkpoints on one device.
    for spec in specs:
        backend = None
        values = {}
        for row in rows:
            messages = generation_messages(row)
            pair = []
            for key in ("response_a", "response_b"):
                request = dict(model=spec, messages=messages, response=row[key])

                def run():
                    nonlocal backend
                    from ..utils.inference import load_backend

                    if backend is None:
                        backend = load_backend(spec)
                    return backend.sequence_logp(messages, row[key])

                pair.append(cache.get_or_run(request, run))
            values[row["cell_id"]] = pair
        likelihoods.append(values)
        del backend
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = []
    for row in rows:
        key = row["cell_id"]
        p = implicit_probability(
            *likelihoods[1][key], *likelihoods[0][key], config.get("beta", 0.1)
        )
        result.append(
            {k: row[k] for k in ("cell_id", "original_persona_id", "prompt_id")}
            | dict(option_ids=["A", "B"], target_probabilities=[p, 1 - p])
        )
    write_rows(config["output"], result)
    return dict(cells=len(result), output=config["output"])
