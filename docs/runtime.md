# Runtime notes

## Credentials and API requests

`.env` is a local shell environment file and is ignored by Git. Fill in the two variables in `.env.example`, then load `.env` using `set -a; source .env; set +a` before collection. The collector reads the exported environment variables. `api_key_env` selects the variable for each model.

The reference examples retain the paper's model IDs and provider settings. Qwen3.8-Max and GLM-5.2 use DashScope Chat Completions with thinking disabled. DeepSeek-V4-Flash uses the DeepSeek Responses endpoint with `reasoning.effort: none`. Change `base_url`, `model_name`, `interface`, and `request` for another provider.

Both interfaces request structured JSON output. Roleplay requests contain the profile, prompt and displayed candidate responses; no ground-truth labels enter the request. OpinionQA uses its own categorical prompt templates and option schema.

## Caching and outputs

Each collection command handles one model and caches successful calls in an adjacent `.cache.jsonl`. Failed requests are retried up to the configured limit, then collection stops. Authentication and billing errors stop collection immediately. Use one writer per cache file.

Caches identify requests by their input text, model specification and settings. Use a new output/cache path after updating model or adapter contents. Generation and policy scoring also cache successful calls.

Small result files may be replaced on rerun. Choose an empty output directory for preference dataset construction. For training, use a new checkpoint directory or set `resume_from_checkpoint` to continue a run.

## Study configurations and seeds

The study configs resolve fractional budgets and a fixed Target group into stage-specific YAML files. Acquisition uses model predictions; evaluation and preference construction read the queried labels. The Oracle uses full labels to optimize the configured scalar welfare objective.

Default seeds are 42 but can be changed. The supplied Roleplay split and source A/B orientations are fixed. Weighted selection ties use cell IDs; Random uses seeded sampling.
