# LlamaFactory soft-label DPO patch

`llamafactory-soft-label-dpo.patch` extends the existing trainer at revision
`7fcf5b3b130e5713b52415bb7404c476fada9c8c`.

Prepare a separate checkout using `python scripts/setup_llamafactory.py --directory ../LlamaFactory-wapa`, then install it in the training environment. See [soft DPO](../docs/soft_dpo.md) for the probability convention and loss. Upstream LlamaFactory licensing applies to the framework; its source is not bundled here.
