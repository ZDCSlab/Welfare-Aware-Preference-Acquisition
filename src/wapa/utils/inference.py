"""Lazy local inference backends. No model is loaded by importing WAPA."""

from .prompts import render_qwen_nothink, render_llama3


class TransformersBackend:
    def __init__(self, spec):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.spec = dict(spec)
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec["model_path"], trust_remote_code=False
        )
        dtype = getattr(torch, spec.get("dtype", "bfloat16"))
        self.model = AutoModelForCausalLM.from_pretrained(
            spec["model_path"],
            torch_dtype=dtype,
            device_map=spec.get("device_map", "auto"),
            trust_remote_code=False,
        )
        if spec.get("adapter_path"):
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, spec["adapter_path"])
        self.model.eval()
        self.max_length = int(spec.get("max_length", 32768))
        self.template = spec.get("template", "qwen_nothink")

    def render(self, messages):
        if self.template == "qwen_nothink":
            return render_qwen_nothink(messages)
        if self.template == "llama3":
            return render_llama3(messages)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _encode(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.max_length:
            raise ValueError("context too long; refusing silent truncation")
        return self.torch.tensor([ids], device=self.model.device)

    def logits(self, messages):
        inputs = self._encode(self.render(messages))
        a = self.tokenizer.encode("A", add_special_tokens=False)
        b = self.tokenizer.encode("B", add_special_tokens=False)
        if len(a) != 1 or len(b) != 1 or a == b:
            raise ValueError("A/B must be distinct single tokens")
        with self.torch.inference_mode():
            output = self.model(input_ids=inputs).logits[0, -1]
        return float(output[a[0]]), float(output[b[0]])

    def option_logits(self, messages, option_ids):
        inputs = self._encode(self.render(messages))
        ids = [self.tokenizer.encode(k, add_special_tokens=False) for k in option_ids]
        if any(len(v) != 1 for v in ids) or len({v[0] for v in ids}) != len(ids):
            raise ValueError("option letters must be distinct single tokens")
        with self.torch.inference_mode():
            output = self.model(input_ids=inputs).logits[0, -1]
        return [float(output[v[0]]) for v in ids]

    def sequence_logp(self, messages, response):
        prefix, suffix = sequence_token_ids(
            self.tokenizer, self.template, messages, response, self.max_length
        )
        ids = self.torch.tensor([prefix + suffix], device=self.model.device)
        with self.torch.inference_mode():
            logits = self.model(input_ids=ids).logits[0, len(prefix) - 1 : -1].float()
            targets = ids[0, len(prefix) :]
            return float(logits.log_softmax(-1).gather(1, targets[:, None]).sum())

    def generate(self, messages, settings):
        return self.generate_batch([messages], settings)[0]

    def generate_batch(self, messages_batch, settings):
        """Left-pad requests and remove per-row EOS padding from returned outputs."""
        if not messages_batch:
            return []
        if len(messages_batch) > 1 and settings.get("temperature", 0) > 0:
            raise ValueError(
                "batched generation requires temperature 0 to preserve per-record RNG streams"
            )
        encoded = [
            self.tokenizer.encode(self.render(m), add_special_tokens=False)
            for m in messages_batch
        ]
        width = max(map(len, encoded))
        max_new = int(settings.get("max_new_tokens", 8192))
        if max_new < 1 or not all(encoded):
            raise ValueError("positive generation budget and nonempty prompts required")
        if width + max_new > self.max_length:
            raise ValueError("prompt plus generation budget exceeds context")
        options = {
            "max_new_tokens": max_new,
            "do_sample": settings.get("temperature", 0) > 0,
        }
        if options["do_sample"]:
            options.update(
                temperature=settings["temperature"], top_p=settings.get("top_p", 1.0)
            )
        eos = (
            self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if self.template == "qwen_nothink"
            else self.tokenizer.eos_token_id
        )
        if eos is None:
            raise ValueError("generation requires an EOS token")
        inputs = self.torch.tensor(
            [[eos] * (width - len(ids)) + ids for ids in encoded],
            device=self.model.device,
        )
        mask = self.torch.tensor(
            [[0] * (width - len(ids)) + [1] * len(ids) for ids in encoded],
            device=self.model.device,
        )
        with self.torch.inference_mode():
            out = self.model.generate(
                input_ids=inputs,
                attention_mask=mask,
                eos_token_id=eos,
                pad_token_id=eos,
                **options,
            )
        results = []
        for sequence in out:
            tokens = sequence[width:].tolist()
            stopped = eos in tokens
            if stopped:
                tokens = tokens[: tokens.index(eos) + 1]
            results.append(
                {
                    "text": self.tokenizer.decode(tokens, skip_special_tokens=True),
                    "output_tokens": len(tokens),
                    "finish_reason": "stop" if stopped else "length",
                }
            )
        return results


def load_backend(spec):
    if spec.get("backend", "transformers") != "transformers":
        raise ValueError("unsupported local backend")
    return TransformersBackend(spec)


def sequence_token_ids(tokenizer, template, messages, response, max_length):
    """Full response plus training EOS, excluding prompt tokens from log likelihood."""
    renderers = {
        "qwen_nothink": (render_qwen_nothink, "<|im_end|>\n"),
        "llama3": (render_llama3, "<|eot_id|>"),
    }
    if template not in renderers:
        raise ValueError(
            "Sequence scoring requires an explicitly matched training template"
        )
    renderer, end = renderers[template]
    prefix = tokenizer.encode(renderer(messages), add_special_tokens=False)
    suffix = tokenizer.encode(response + end, add_special_tokens=False)
    if not prefix or not suffix or len(prefix) + len(suffix) > max_length:
        raise ValueError("source response too long or empty; no truncation")
    return prefix, suffix
