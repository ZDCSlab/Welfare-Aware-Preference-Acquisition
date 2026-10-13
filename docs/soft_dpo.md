# Soft-label DPO

The patch targets LlamaFactory revision `7fcf5b3b130e5713b52415bb7404c476fada9c8c` and extends its dataset column mapping, conversion, pairwise processing, collator and existing DPO trainer.

With `dpo_soft_labels: true`, a per-pair `preference_probability` q defines

```text
loss = -q * logsigmoid(z) - (1-q) * logsigmoid(-z)
z = beta * [(log pi(chosen) - log pi_ref(chosen))
            - (log pi(rejected) - log pi_ref(rejected))].
```

Soft labels require plain sigmoid DPO with label smoothing, FTX, BCO and LD-DPO disabled.

The WAPA data builder orients queried pairs using their actual label and sets q=1. It orients unqueried pairs using the base-model prediction and stores the corresponding probability of the chosen response. Every acquisition comparison remains in the training dataset with equal example weight.

The default context is Qwen's no-thinking format. Persona background belongs to system; the original request belongs to user. Chosen and rejected are assistant completions. Use the same chat template for training and scoring. Policy scoring includes its response-ending tokens.

`setup_llamafactory.py` creates a separate pinned checkout and applies the patch idempotently. For an existing checkout, use the revision listed above. Install that checkout in the training environment, then use `wapa train-prepare` and `wapa train`.
