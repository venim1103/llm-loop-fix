import json
from dataclasses import FrozenInstanceError, replace

import pytest

from llm_loop_fix.records import (
    ChatMessage,
    DatasetManifest,
    EvaluationItem,
    EvaluationProfile,
    GenerationRecord,
    SampleKey,
    SOURCE_MODEL_REPO,
    SOURCE_MODEL_REVISION,
    QualityObservation,
    audit_dataset_overlaps,
    load_evaluation_items,
    pair_generation_records,
    record_from_json,
    record_to_json,
)


def test_item_loader_preserves_identity_and_metadata(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text(
        json.dumps({"id": "math_1", "prompt": "Solve 2+2", "reference_answer": "4"}) + "\n",
        encoding="utf-8",
    )
    items = load_evaluation_items(
        path, source="controlled-fixture", source_revision="fixture-v1", role="development"
    )
    assert items == (
        EvaluationItem("math_1", "controlled-fixture", "fixture-v1", "development", "Solve 2+2", "4"),
    )
    assert items[0].schema_version == 1
    with pytest.raises(FrozenInstanceError):
        items[0].role = "confirmation"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "\n  \n",
        "plain prompt without stable identity\n",
        '{"id":"a","prompt":""}\n',
        '{"prompt":"no id"}\n',
        '{"id":"a","prompt":42}\n',
        '{"id":"a","id":"b","prompt":"duplicate key"}\n',
        '{"id":"a","prompt":"first"}\n{"id":"a","prompt":"second"}\n',
        '{"id":"a","prompt":"hello","role":"confirmation"}\n',
    ],
)
def test_item_loader_rejects_invalid_or_ambiguous_inputs(tmp_path, text):
    path = tmp_path / "items.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_evaluation_items(
            path, source="controlled-fixture", source_revision="fixture-v1", role="diagnostic"
        )


@pytest.mark.parametrize("role", ["", "test", "train"])
def test_item_rejects_unknown_role(role):
    with pytest.raises(ValueError, match="dataset role"):
        EvaluationItem("a", "controlled-fixture", "fixture-v1", role, "prompt")


@pytest.fixture
def generations():
    baseline = tuple(
        GenerationRecord("experiment-v1", "base", "profile-v1", SampleKey("a", index, 0),
                         10, (1, 2), "output", "stop")
        for index in range(2)
    )
    candidate = tuple(replace(record, candidate_id="edit") for record in baseline)
    return baseline, candidate, tuple(record.key for record in baseline)


def test_pairing_uses_keys_not_list_positions(generations):
    baseline, candidate, expected = generations
    pairs = pair_generation_records(reversed(baseline), candidate, expected_keys=reversed(expected))
    assert pairs == tuple(zip(baseline, candidate))


@pytest.mark.parametrize("side", ["baseline", "candidate", "both"])
def test_pairing_rejects_missing_even_when_both_sides_omit(generations, side):
    baseline, candidate, expected = generations
    if side in {"baseline", "both"}:
        baseline = baseline[:1]
    if side in {"candidate", "both"}:
        candidate = candidate[:1]
    with pytest.raises(ValueError, match="missing"):
        pair_generation_records(baseline, candidate, expected_keys=expected)


@pytest.mark.parametrize("side", ["baseline", "candidate", "expected"])
def test_pairing_rejects_duplicate_keys(generations, side):
    baseline, candidate, expected = generations
    if side == "baseline":
        baseline = (baseline[0], baseline[0])
    elif side == "candidate":
        candidate = (candidate[0], candidate[0])
    else:
        expected = (expected[0], expected[0])
    with pytest.raises(ValueError, match="duplicate"):
        pair_generation_records(baseline, candidate, expected_keys=expected)


@pytest.mark.parametrize(
    "changes",
    [{"profile_id": "other"}, {"experiment_id": "other"}, {"candidate_id": "other"},
     {"prompt_tokens": 11}, {"key": SampleKey("unexpected", 0, 0)},
     {"key": SampleKey("a", 0, 1)}],
)
def test_pairing_rejects_identity_or_prompt_mismatch(generations, changes):
    baseline, candidate, expected = generations
    candidate = (replace(candidate[0], **changes), candidate[1])
    with pytest.raises(ValueError):
        pair_generation_records(baseline, candidate, expected_keys=expected)


def test_empty_pairing_is_not_success():
    with pytest.raises(ValueError, match="nonempty"):
        pair_generation_records((), (), expected_keys=())


@pytest.mark.parametrize(
    "changes", [{"prompt_tokens": True}, {"generated_token_ids": [1]},
                {"generated_token_ids": (-1,)}, {"finish_reason": "inferred_cap"}],
)
def test_generation_rejects_invalid_payloads(generations, changes):
    baseline, _, _ = generations
    with pytest.raises(ValueError):
        replace(baseline[0], **changes)


def test_empty_output_is_retained_not_dropped(generations):
    baseline, candidate, expected = generations
    candidate = tuple(replace(record, generated_text="", generated_token_ids=(), finish_reason="error")
                      for record in candidate)
    pairs = pair_generation_records(baseline, candidate, expected_keys=expected)
    assert len(pairs) == 2
    assert pairs[0][1].finish_reason == "error"


def test_json_round_trip_restores_immutable_records(generations):
    baseline, _, _ = generations
    item = EvaluationItem("a", "fixture", "v1", "benchmark", messages=(ChatMessage("user", "question"),))
    for record in (item, *baseline):
        assert record_from_json(record_to_json(record)) == record
        assert record_to_json(record_from_json(record_to_json(record))) == record_to_json(record)


@pytest.mark.parametrize(
    "changes", [{"schema_version": True}, {"schema_version": 2}, {"record_type": "unknown"},
                {"generated_token_ids": {"0": 1}}, {"extra": "silently ignored"}],
)
def test_record_parser_rejects_invalid_schema(generations, changes):
    baseline, _, _ = generations
    payload = json.loads(record_to_json(baseline[0]))
    payload.update(changes)
    with pytest.raises(ValueError):
        record_from_json(json.dumps(payload))


@pytest.mark.parametrize("text", ['{"schema_version":1,"schema_version":1}', '{"schema_version":NaN}'])
def test_record_parser_rejects_duplicate_or_nonfinite_json(text):
    with pytest.raises(ValueError):
        record_from_json(text)


def test_chat_loader_preserves_message_structure(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text('{"id":"a","messages":[{"role":"user","content":"question"}]}\n', encoding="utf-8")
    items = load_evaluation_items(path, source="fixture", source_revision="v1", role="diagnostic")
    assert items[0].prompt is None
    assert items[0].messages == (ChatMessage("user", "question"),)


def test_dataset_digest_is_order_independent_and_role_sensitive():
    first = EvaluationItem("a", "fixture", "v1", "development", "first")
    second = replace(first, item_id="b", prompt="second")
    assert DatasetManifest((first, second)).sha256 == DatasetManifest((second, first)).sha256
    assert DatasetManifest((first,)).sha256 != DatasetManifest((replace(first, role="confirmation"),)).sha256
    with pytest.raises(ValueError, match="overlap"):
        DatasetManifest((first, replace(first, item_id="b", role="confirmation")))
    with pytest.raises(ValueError, match="duplicate"):
        DatasetManifest((first, first))


@pytest.fixture
def source_profile():
    return EvaluationProfile(
        track="source_reproduction", task="math500", model_repo=SOURCE_MODEL_REPO,
        model_revision=SOURCE_MODEL_REVISION, tokenizer_repo=SOURCE_MODEL_REPO,
        tokenizer_revision=SOURCE_MODEL_REVISION,
        chat_template_sha256="b" * 64, backend="offline-fixture", backend_version="fixture-v1",
        grader="math-verify", grader_version="0.9.0", dependency_lock_sha256="c" * 64,
        eos_token_ids=(2,),
    )


def test_profile_round_trip_and_track_identity(source_profile):
    assert record_from_json(record_to_json(source_profile)) == source_profile
    assert replace(source_profile, track="validated_repair").sha256 != source_profile.sha256
    assert replace(source_profile, backend_version="fixture-v2").sha256 != source_profile.sha256
    assert replace(source_profile, chat_template_sha256="d" * 64).sha256 != source_profile.sha256


@pytest.mark.parametrize(
    "changes", [{"temperature": 0.7}, {"max_new_tokens": 8192}, {"thinking": False},
                {"samples_per_item": 1}, {"model_revision": "main"}, {"tokenizer_revision": "latest"},
                {"temperature": float("nan")}, {"top_p": True}, {"eos_token_ids": [2]},
                {"eos_token_ids": (2, 2)}, {"dependency_lock_sha256": "unpinned"},
                {"stop_sequences": ("extra-stop",)}, {"seed_policy": "independent_per_sample"},
                {"model_repo": "other/model"}, {"tokenizer_revision": "e" * 40},
                {"grader_version": "0.8.0"}, {"grader": "exact-string"}],
)
def test_profile_rejects_unpinned_or_changed_source_protocol(source_profile, changes):
    with pytest.raises(ValueError):
        replace(source_profile, **changes)


def test_source_task_samples_and_prompt_templates(source_profile):
    with pytest.raises(ValueError):
        replace(source_profile, task="gsm8k")
    gsm = replace(source_profile, task="gsm8k", samples_per_item=2)
    ifeval = replace(source_profile, task="ifeval", samples_per_item=2,
                     prompt_template="ifeval_user_v1", grader="ifeval-strict", grader_version="fixture-v1")
    assert gsm.sha256 != ifeval.sha256
    assert record_from_json(record_to_json(ifeval)) == ifeval


def test_overlap_audit_normalizes_prompts_and_chat_without_answers():
    first = EvaluationItem("a", "fixture", "v1", "development", "Solve  2 + 2", reference_answer="4")
    second = EvaluationItem("b", "fixture", "v1", "confirmation",
                            messages=(ChatMessage("user", "  SOLVE 2 + 2  "),), reference_answer="wrong")
    dataset = DatasetManifest((first, second))
    audit = audit_dataset_overlaps(dataset)
    assert audit.review_required
    assert audit.dataset_sha256 == dataset.sha256
    assert audit.compared_pairs == 1
    assert audit.findings[0].kind == "normalized_exact"
    assert audit.findings[0].similarity == 1
    changed_answer = DatasetManifest((replace(first, reference_answer="other"), second))
    assert audit_dataset_overlaps(changed_answer).findings == audit.findings


def test_near_overlap_is_review_candidate_not_semantic_equivalence():
    first = EvaluationItem("a", "fixture", "v1", "development", "Calculate the sum of 100 and 200.")
    second = replace(first, item_id="b", role="benchmark", prompt="Calculate the sum of 100 and 201.")
    audit = audit_dataset_overlaps(DatasetManifest((first, second)))
    assert audit.review_required
    assert audit.findings[0].kind == "near_duplicate"
    assert audit.findings[0].similarity < 1


def test_overlap_audit_order_is_deterministic_and_only_cross_role():
    first = EvaluationItem("a", "fixture", "v1", "diagnostic", "First question")
    second = replace(first, item_id="b", role="development", prompt="first  question")
    third = replace(first, item_id="c", prompt="completely unrelated")
    dataset = DatasetManifest((first, second, third))
    audit = audit_dataset_overlaps(dataset)
    assert audit == audit_dataset_overlaps(DatasetManifest(tuple(reversed(dataset.items))))
    assert audit.compared_pairs == 2
    assert len(audit.findings) == 1
    assert audit_dataset_overlaps(DatasetManifest((first, third)), max_comparisons=0).compared_pairs == 0


@pytest.mark.parametrize("threshold", [0, 1.1, float("nan"), True])
def test_overlap_audit_rejects_invalid_threshold(threshold):
    item = EvaluationItem("a", "fixture", "v1", "diagnostic", "question")
    with pytest.raises(ValueError):
        audit_dataset_overlaps(DatasetManifest((item,)), near_duplicate_threshold=threshold)


def test_overlap_audit_refuses_exceeded_budgets_without_partial_results():
    first = EvaluationItem("a", "fixture", "v1", "diagnostic", "question one")
    second = replace(first, item_id="b", role="confirmation", prompt="question two")
    dataset = DatasetManifest((first, second))
    with pytest.raises(ValueError, match="comparison budget"):
        audit_dataset_overlaps(dataset, max_comparisons=0)
    with pytest.raises(ValueError, match="character budget"):
        audit_dataset_overlaps(dataset, max_prompt_characters=3)


def test_overlap_audit_handles_unicode_normalization_and_expansion_budget():
    first = EvaluationItem("a", "fixture", "v1", "development", "solve")
    second = replace(first, item_id="b", role="confirmation", prompt="\uff33\uff2f\uff2c\uff36\uff25")
    assert audit_dataset_overlaps(DatasetManifest((first, second))).findings[0].kind == "normalized_exact"
    expanding = replace(first, prompt="\ufdfa")
    with pytest.raises(ValueError, match="normalized prompt character budget"):
        audit_dataset_overlaps(DatasetManifest((expanding,)), max_prompt_characters=4)


@pytest.mark.parametrize(
    "changes", [{"correctness": None}, {"correctness": 1}, {"usability": "empty"},
                {"grader_status": "error"}, {"reason": 42}],
)
def test_quality_observation_rejects_contradictory_or_mutable_schema(changes):
    quality = QualityObservation("source-v1", "math-verify@0.9.0", "4", "usable", True, "graded")
    with pytest.raises(ValueError):
        replace(quality, **changes)