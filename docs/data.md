# Data formats

Use JSON lists, JSONL records or Parquet tables. All stages join by unique `cell_id`. Acquisition operates on a complete persona × prompt rectangle, with unique persona IDs across collections in the pool.

## Model inputs

Roleplay records contain string fields:

```text
cell_id, group_id, original_persona_id, prompt_id, profile_prompt_id,
pipeline_split, user_profile, prompt, response_a, response_b
```

The split is `acquisition`, `test_seen_personas` or `test_heldout_personas`. Ground-truth labels are stored in separate files. The Roleplay script retains the supplied fixed split plan and deterministic source A/B orientation.

OpinionQA replaces `response_a` and `response_b` with `option_ids` (ordered letters) and `option_texts`, and adds `benchmark_id: opinionqa` and `task_type: categorical_choice`. The supplied preparation uses 3–5 options per question. The implementation accepts 2–5.

## Labels

Stored separately:

```json
{"cell_id": "example", "preference_label_y": 1}
```

For Roleplay, `1` means original A is preferred and `0` means B. For OpinionQA:

```json
{"cell_id": "example", "selected_option_id": "C"}
```

Supply labels for every evaluated comparison when running evaluation, and for the selected comparison IDs when constructing training data. Real annotations can use the same format.

## Model predictions and aggregate evidence

Per-model prediction:

```json
{"cell_id":"example","model_id":"target","order":"original",
 "option_ids":["A","B"],"probabilities":[0.7,0.3]}
```

Probabilities use displayed coordinates, including swapped Roleplay requests. Aggregation checks complete model/order coverage and writes original coordinates:

```json
{"cell_id":"example","original_persona_id":"person-1","prompt_id":"question-1",
 "option_ids":["A","B"],"target_probabilities":[0.7,0.3],
 "reference_model_ids":["reference-1","reference-2"],
 "reference_probabilities":[[0.9,0.1],[0.8,0.2]]}
```

`reference_probabilities` contains one vector per reference model. The selector computes their equal-weight mean. Probabilities must be finite, nonnegative, and sum to one (serialization tolerance `1e-6`).

Policy scoring emits target-probability records for evaluation. Acquisition additionally requires the reference vectors.

## Preferences and outputs

A preference directory contains `train.json`, `dataset_info.json` and `label_sources.jsonl`. Training records contain `cell_id`, `system`, `prompt`, `chosen`, `rejected`, and `preference_probability`. The audit records which labels were queried and how A/B was oriented.

Selections record the method, budget, selected IDs and welfare parameters. Weighted selections also include the allocation trace and initial/final estimated utilities. Oracle selections are explicitly marked `uses_ground_truth: true`.

Evaluation emits summary metrics and individual utilities as JSON. Generation emits one record per input with text, output-token count and finish reason. Successful model calls are cached in adjacent `.cache.jsonl` files. Assign a separate cache to each collection process.
