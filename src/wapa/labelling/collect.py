"""Per-model collection with local logits or an OpenAI-compatible JSON endpoint."""

import json
import math
import os
import time

from ..utils.io import Cache, read_rows, read_config, write_rows
from ..utils.prompts import pair_messages, categorical_messages
from ..utils.validation import validate_inputs, index_rows
from ..utils.categorical import probability_vector


def response_schema(options, categorical):
    props = {"choice": {"type": "string", "enum": options}}
    probability = {"type": "number", "minimum": 0, "maximum": 1}
    if categorical:
        props["probabilities"] = {
            "type": "object",
            "properties": {k: probability for k in options},
            "required": options,
            "additionalProperties": False,
        }
    else:
        props["probability_a"] = probability
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def parse_response(content, options, categorical):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    value = json.loads(content, object_pairs_hook=unique)
    expected = {"choice", "probabilities" if categorical else "probability_a"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("unexpected response fields")
    if categorical:
        probs = value["probabilities"]
        if not isinstance(probs, dict) or set(probs) != set(options):
            raise ValueError("option support mismatch")
        vector = probability_vector([probs[k] for k in options], options)
    else:
        p = value["probability_a"]
        if (
            isinstance(p, bool)
            or not isinstance(p, (int, float))
            or not math.isfinite(p)
            or not 0 <= p <= 1
        ):
            raise ValueError("invalid probability_a")
        vector = [p, 1 - p]
    if value["choice"] != options[max(range(len(vector)), key=vector.__getitem__)]:
        raise ValueError("choice and probabilities disagree")
    return vector


def collect(config):
    rows = read_rows(config["inputs"])
    validate_inputs(rows, split="acquisition", rectangular=True)
    spec = config["model"]
    template = read_config(config["prompt_template"])
    cache = Cache(config["output"] + ".cache.jsonl")
    backend = spec.get("backend", "transformers")
    model = None
    client = None
    results = []
    if backend not in ("transformers", "openai"):
        raise ValueError("backend must be transformers or openai")
    for row in rows:
        categorical = row.get("task_type") == "categorical_choice"
        options = row["option_ids"] if categorical else ["A", "B"]
        orders = ["original"] if categorical else ["original", "swapped"]
        for order in orders:
            messages = (
                categorical_messages(row, template)
                if categorical
                else pair_messages(
                    row, template, order, preserve_prediction_template=True
                )
            )
            request = dict(
                cell_id=row["cell_id"],
                model=spec,
                messages=messages,
                order=order,
                request=config.get("request", {}),
                options=options,
            )

            def run():
                nonlocal model, client
                if backend == "transformers":
                    if model is None:
                        from ..utils.inference import load_backend

                        model = load_backend(spec)
                    logits = model.option_logits(messages, options)
                    maximum = max(logits)
                    exps = [math.exp(v - maximum) for v in logits]
                    return probability_vector([v / sum(exps) for v in exps], options)
                if client is None:
                    from openai import OpenAI

                    client = OpenAI(
                        base_url=spec.get("base_url"),
                        api_key=os.environ[spec["api_key_env"]],
                        timeout=spec.get("timeout", 120),
                        max_retries=0,
                    )
                settings = dict(config.get("request", {}))
                settings.setdefault("temperature", 0)
                interface = spec.get("interface", "chat_completions")
                schema = {
                    "name": "preference",
                    "strict": True,
                    "schema": response_schema(options, categorical),
                }
                if interface == "responses":
                    settings.setdefault("max_output_tokens", 256 if categorical else 64)
                    settings["text"] = {"format": {"type": "json_schema", **schema}}
                elif interface == "chat_completions":
                    settings.setdefault(
                        "max_completion_tokens", 256 if categorical else 64
                    )
                    settings["response_format"] = {
                        "type": "json_schema",
                        "json_schema": schema,
                    }
                else:
                    raise ValueError("interface must be chat_completions or responses")
                for attempt in range(config.get("retries", 3) + 1):
                    try:
                        if interface == "responses":
                            response = client.responses.create(
                                model=spec["model_name"], input=messages, **settings
                            )
                            if getattr(response, "status", "completed") != "completed":
                                raise ValueError("incomplete reference response")
                            content = response.output_text
                        else:
                            response = client.chat.completions.create(
                                model=spec["model_name"], messages=messages, **settings
                            )
                            content = response.choices[0].message.content
                        return parse_response(content, options, categorical)
                    except Exception as exc:
                        if getattr(exc, "status_code", None) in (
                            401,
                            402,
                            403,
                        ) or attempt == config.get("retries", 3):
                            raise
                        time.sleep(min(2**attempt, 8))

            probabilities = cache.get_or_run(request, run)
            results.append(
                dict(
                    cell_id=row["cell_id"],
                    model_id=spec["id"],
                    order=order,
                    option_ids=options,
                    probabilities=probabilities,
                )
            )
    write_rows(config["output"], results)
    return dict(predictions=len(results), output=config["output"])


def aggregate(inputs, predictions, target, references):
    validate_inputs(inputs, split="acquisition", rectangular=True)
    if not references or len(set([target, *references])) != len(references) + 1:
        raise ValueError("unique target and reference model IDs required")
    ii = index_rows(inputs)
    grouped = {}
    for record in predictions:
        key = (record["cell_id"], record["model_id"], record["order"])
        if key in grouped or key[0] not in ii or key[1] not in [target, *references]:
            raise ValueError("duplicate or unexpected prediction")
        row = ii[key[0]]
        options = row.get("option_ids", ["A", "B"])
        orders = (
            ["original"]
            if row.get("task_type") == "categorical_choice"
            else ["original", "swapped"]
        )
        if record["option_ids"] != options or key[2] not in orders:
            raise ValueError("unexpected prediction coordinates")
        grouped[key] = probability_vector(record["probabilities"], options)
    result = []
    for row in inputs:
        options = row.get("option_ids", ["A", "B"])
        categorical = row.get("task_type") == "categorical_choice"
        probs = {}
        for model in [target, *references]:
            try:
                original = grouped[row["cell_id"], model, "original"]
                swapped = (
                    None if categorical else grouped[row["cell_id"], model, "swapped"]
                )
            except KeyError as exc:
                raise ValueError("incomplete model/order coverage") from exc
            probs[model] = (
                original
                if categorical
                else [(original[k] + swapped[1 - k]) / 2 for k in range(2)]
            )
        result.append(
            {k: row[k] for k in ("cell_id", "original_persona_id", "prompt_id")}
            | dict(
                option_ids=options,
                target_probabilities=probs[target],
                reference_model_ids=references,
                reference_probabilities=[probs[m] for m in references],
            )
        )
    return result
