"""Exercise stage boundaries with tiny local data and mocked model calls."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from wapa.cli import run
from wapa.utils.io import write_rows, write_json, read_rows
from test_core import fixture

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collection_training_and_generation_boundaries(tmp_path, monkeypatch):
    import wapa.utils.inference

    inputs, _, labels = fixture()
    input_path = tmp_path / "inputs.json"
    labels_path = tmp_path / "labels.json"
    write_rows(input_path, inputs)
    write_rows(labels_path, labels)
    calls = []

    class Backend:
        def option_logits(self, messages, options):
            calls.append(messages)
            assert len(options) == 2
            return [1.0, 0.0]

        def sequence_logp(self, messages, response):
            assert messages[-1]["content"] == "Where to go?"
            return -1.0 if response == "A response" else -2.0

        def generate(self, messages, settings):
            assert messages[-1]["content"] == "Where to go?"
            assert "A response" not in str(messages)
            return dict(text="A quiet park.", output_tokens=4, finish_reason="stop")

    monkeypatch.setattr(wapa.utils.inference, "load_backend", lambda spec: Backend())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            manual_seed=lambda seed: None,
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    )
    prediction_paths = []
    for model in ("target", "ref"):
        output = tmp_path / f"{model}.json"
        cfg = dict(
            inputs=str(input_path),
            output=str(output),
            model=dict(id=model, model_path="mock"),
            prompt_template=str(ROOT / "configs/prompts/target_pairwise.yaml"),
        )
        run("collect", cfg)
        count = len(calls)
        run("collect", cfg)
        assert len(calls) == count
        prediction_paths.append(str(output))
    evidence = tmp_path / "evidence.parquet"
    run(
        "aggregate",
        dict(
            inputs=str(input_path),
            predictions=prediction_paths,
            target="target",
            references=["ref"],
            output=str(evidence),
        ),
    )
    selection = tmp_path / "selection.json"
    run(
        "acquire",
        dict(evidence=str(evidence), method="wapa", budget=3, output=str(selection)),
    )
    preferences = tmp_path / "preferences"
    run(
        "preferences",
        dict(
            inputs=str(input_path),
            evidence=str(evidence),
            labels=str(labels_path),
            selection=str(selection),
            output=str(preferences),
        ),
    )
    assert len(read_rows(preferences / "train.json")) == len(inputs)
    run(
        "train-prepare",
        dict(
            dataset_dir=str(preferences),
            base_model="mock",
            checkpoint_output=str(tmp_path / "checkpoint"),
            output=str(tmp_path / "train"),
        ),
    )
    scores = tmp_path / "scores.json"
    run(
        "score-policy",
        dict(
            inputs=str(input_path),
            policy=dict(model_path="mock"),
            reference=dict(model_path="mock"),
            output=str(scores),
        ),
    )
    assert all(r["target_probabilities"] == [0.5, 0.5] for r in read_rows(scores))
    run(
        "evaluate",
        dict(
            evidence=str(scores),
            labels=str(labels_path),
            output=str(tmp_path / "metrics.json"),
        ),
    )
    generation = tmp_path / "generated.json"
    run(
        "generate",
        dict(
            inputs=str(input_path),
            model=dict(model_path="mock"),
            output=str(generation),
        ),
    )
    assert len(read_rows(generation)) == len(inputs)


def test_roleplay_preparation(tmp_path):
    module = load_script("prepare_roleplay")
    plan = {
        "north_america": {
            "acquisition": dict(
                source_split="train", prompt_ids=["q1"], persona_ids=["u1"]
            ),
            "test_seen_personas": dict(
                source_split="train", prompt_ids=["q2"], persona_ids=["u1"]
            ),
            "test_heldout_personas": dict(
                source_split="test", prompt_ids=["q2"], persona_ids=["u2"]
            ),
        }
    }
    split = tmp_path / "splits.json"
    write_json(split, plan)

    def row(user, prompt):
        return dict(
            score_persona=dict(
                persona_uuid=user,
                persona_description="Profile",
                traits=dict(
                    age="young adult", gender="female", continent="North America"
                ),
            ),
            question=dict(
                question_type=dict(type="global"),
                question_uuid=prompt,
                question="Question",
            ),
            x="Question",
            level=user,
            yw="Preferred",
            yl="Other",
        )

    raw = tmp_path / "raw"
    write_rows(raw / "data/train-0.parquet", [row("u1", "q1"), row("u1", "q2")])
    write_rows(raw / "data/test-0.parquet", [row("u2", "q2")])
    output = tmp_path / "prepared"
    result = module.prepare(raw, split, output, "north_america")
    assert result["acquisition"]["cells"] == 1
    assert len(read_rows(output / "heldout.parquet")) == 2
    record = read_rows(output / "acquisition/inputs.parquet")[0]
    label = read_rows(output / "labels/acquisition.parquet")[0]
    assert (
        record["response_a" if label["preference_label_y"] else "response_b"]
        == "Preferred"
    )


def test_opinionqa_preparation_from_cached_source(tmp_path):
    import csv

    module = load_script("prepare_opinionqa")
    raw = tmp_path / "raw"
    raw.mkdir()
    codebook = [
        dict(
            id=k,
            question=k,
            options={"1": "west"} if k == "F_CREGION" else {"1": "description"},
        )
        for k in module.DEMOGRAPHICS
    ]
    codebook.append(
        dict(
            id="Q1",
            question="Question?",
            options={"A": "One", "B": "Two", "C": "Three"},
        )
    )
    write_rows(raw / "codebook.jsonl", codebook)
    with (raw / "opinionqa_responses.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["caseid", *module.DEMOGRAPHICS, "Q1"]
        )
        writer.writeheader()
        for i in range(2):
            writer.writerow(
                dict(caseid=f"{i}.0", Q1="C", **{k: "1" for k in module.DEMOGRAPHICS})
            )
    output = tmp_path / "processed"
    result = module.prepare(
        dict(
            source_revision=module.REVISION,
            source_repository="ZDCSlab/Group-Adaptive-Elicitation",
            option_order="source",
            raw_root=str(raw),
            output=str(output),
            questions=1,
            source_users={"west": 2},
            users={"west": 2},
            seed=13,
        )
    )
    assert result["west"]["cells"] == 2
    records = read_rows(output / "west/acquisition/inputs.parquet")
    assert records[0]["option_ids"] == ["A", "B", "C"]
    assert (
        read_rows(output / "west/labels/acquisition.parquet")[0]["selected_option_id"]
        == "C"
    )


def test_generated_study_configs(tmp_path):
    from wapa.utils.io import read_config

    module = load_script("prepare_acquisition")
    inputs, evidence, labels = fixture()
    write_rows(tmp_path / "inputs.json", inputs)
    write_rows(tmp_path / "evidence.json", evidence)
    write_json(tmp_path / "target.json", {"target_group": ["0"]})
    config = dict(
        inputs=str(tmp_path / "inputs.json"),
        evidence=str(tmp_path / "evidence.json"),
        labels=str(tmp_path / "labels.json"),
        target_file=str(tmp_path / "target.json"),
        budget_fraction=0.2,
        welfare=dict(gamma=2, rho=-5, epsilon=0.5),
        preferences=True,
        output=str(tmp_path / "study"),
    )
    # Preparing configs must work before any label file exists.
    result = module.prepare(config)
    assert result["budget"] == 1
    write_rows(tmp_path / "labels.json", labels)
    directory = Path(result["configs"])
    for method in ("wapa", "random", "entropy", "reference_disagreement"):
        acquisition = read_config(directory / f"acquire_{method}.yaml")
        assert acquisition["welfare"]["target_group"] == ["0"]
        run("acquire", acquisition)
        metrics = run("evaluate", read_config(directory / f"evaluate_{method}.yaml"))
        assert metrics["queried"] == 1 and metrics["target"] is not None
        run("preferences", read_config(directory / f"preferences_{method}.yaml"))
    for name in ("all_pseudo_labels", "all_ground_truth_labels"):
        run("preferences", read_config(directory / f"preferences_{name}.yaml"))
    run("oracle", read_config(directory / "oracle.yaml"))
    run("evaluate", read_config(directory / "evaluate_oracle.yaml"))
    assert (
        run("evaluate", read_config(directory / "evaluate_no_queries.yaml"))["queried"]
        == 0
    )


def test_chat_and_responses_request_formats(tmp_path, monkeypatch):
    inputs, _, _ = fixture()
    write_rows(tmp_path / "inputs.json", inputs)
    monkeypatch.setenv("TEST_REFERENCE_KEY", "dummy-key")
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        assert kwargs["response_format"]["json_schema"]["strict"]
        assert kwargs["extra_body"] == {"enable_thinking": False}
        assert "input" not in kwargs and "max_output_tokens" not in kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"choice":"A","probability_a":0.8}'
                    )
                )
            ]
        )

    def responses(**kwargs):
        calls.append(kwargs)
        assert kwargs["text"]["format"]["strict"]
        assert kwargs["reasoning"] == {"effort": "none"}
        assert "messages" not in kwargs and "max_completion_tokens" not in kwargs
        return SimpleNamespace(
            status="completed", output_text='{"choice":"A","probability_a":0.8}'
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat)),
        responses=SimpleNamespace(create=responses),
    )
    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: client)
    )
    for interface, request in [
        ("chat_completions", dict(extra_body={"enable_thinking": False})),
        ("responses", dict(reasoning={"effort": "none"}, tools=[])),
    ]:
        config = dict(
            inputs=str(tmp_path / "inputs.json"),
            output=str(tmp_path / f"{interface}.json"),
            prompt_template=str(ROOT / "configs/prompts/reference_pairwise.yaml"),
            request=request,
            model=dict(
                id="ref",
                backend="openai",
                model_name="mock",
                api_key_env="TEST_REFERENCE_KEY",
                interface=interface,
            ),
        )
        run("collect", config)
        assert len(read_rows(config["output"])) == 2 * len(inputs)
    assert len(calls) == 4 * len(inputs)
