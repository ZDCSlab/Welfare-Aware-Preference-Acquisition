"""Generate free responses from profile and prompt only."""

from ..utils.io import Cache, digest, read_rows, write_rows
from ..utils.prompts import generation_messages
from ..utils.validation import index_rows


def generate(config):
    rows = read_rows(config["inputs"])
    index_rows(rows)
    spec, settings = config["model"], config.get("decoding", {})
    seed = config.get("seed", 42)
    cache = Cache(config["output"] + ".cache.jsonl")
    model = None
    output = []
    for row in rows:
        messages = generation_messages(row)
        record_seed = int(digest([seed, row["cell_id"]])[:8], 16)
        request = dict(
            model=spec, messages=messages, settings=settings, seed=record_seed
        )

        def run():
            nonlocal model
            import torch
            from ..utils.inference import load_backend

            if model is None:
                model = load_backend(spec)
            torch.manual_seed(record_seed)
            return model.generate(messages, settings)

        value = cache.get_or_run(request, run)
        output.append(
            dict(
                cell_id=row["cell_id"],
                original_persona_id=row["original_persona_id"],
                prompt_id=row["prompt_id"],
                **value,
            )
        )
    write_rows(config["output"], output)
    return dict(cells=len(output), output=config["output"])
