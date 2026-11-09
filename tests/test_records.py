import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from llm_loop_fix.records import (
    ChatMessage,
    DatasetManifest,
    EvaluationItem,
    EvaluationProfile,
    EvaluationRunIdentity,
    GenerationRecord,
    SampleKey,
    SOURCE_MODEL_REPO,
    SOURCE_MODEL_REVISION,
    QualityObservation,
    GenerationHealth,
    RepetitionObservation,
    RepetitionSpan,
    ThinkingObservation,
    TokenTailObservation,
    InstructionSpec,
    audit_dataset_overlaps,
    load_evaluation_items,
    pair_generation_records,
    record_from_json,
    record_to_json,
    record_vllm_generations,
)
from llm_loop_fix.detector import RepetitionSpec, TokenTailSpec, diagnose_generation_batch, diagnose_generation_records, observe_generation_health, observe_token_repetition, observe_token_tail
from llm_loop_fix.evaluation import evaluate_generation_batch


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


def test_pairing_checks_prompt_tokens_not_only_equal_lengths(generations):
    baseline, candidate, expected = generations
    baseline = tuple(replace(record, prompt_token_ids=tuple(range(10))) for record in baseline)
    candidate = tuple(replace(record, prompt_token_ids=(99,) + tuple(range(1, 10))) for record in candidate)
    with pytest.raises(ValueError, match="prompt mismatch"):
        pair_generation_records(baseline, candidate, expected_keys=expected)


@pytest.fixture
def native_vllm_output():
    profile = EvaluationProfile(
        "source_reproduction", "math500", SOURCE_MODEL_REPO, SOURCE_MODEL_REVISION,
        SOURCE_MODEL_REPO, SOURCE_MODEL_REVISION, "a" * 64, "vllm", "controlled-api-fixture",
        "math-verify", "0.9.0", "b" * 64, (2,),
    )
    item = EvaluationItem("a", "fixture", "v1", "development", "question", "4")
    output = SimpleNamespace(request_id="native-1", prompt_token_ids=list(range(10)), finished=True,
                             outputs=[SimpleNamespace(index=index, text="output", token_ids=[1, 2],
                                                      finish_reason="stop", stop_reason=None)
                                      for index in range(4)])
    arguments = dict(experiment_id="experiment-v1", candidate_id="base", expected_request_id="native-1",
                     expected_prompt_token_ids=tuple(range(10)), thinking_open_from_prompt=True,
                     backend_version=profile.backend_version)
    return item, output, profile, arguments


def test_native_adapter_snapshots_complete_reordered_outputs(native_vllm_output):
    item, output, profile, arguments = native_vllm_output
    output.outputs.reverse()
    records = record_vllm_generations(item, output, profile, **arguments)
    assert tuple(record.key.sample_index for record in records) == (0, 1, 2, 3)
    assert all(record.key.seed == profile.sampling_seed for record in records)
    assert all(record.prompt_token_ids == tuple(range(10)) for record in records)
    assert all(record.backend_request_id == "native-1" for record in records)
    assert all(record_from_json(record_to_json(record)) == record for record in records)
    output.prompt_token_ids[0] = 99
    output.outputs[0].token_ids[0] = 99
    assert records[0].prompt_token_ids[0] == 0
    assert records[3].generated_token_ids == (1, 2)


def test_run_bound_native_output_preserves_planned_identity(native_vllm_output):
    item, output, profile, arguments = native_vllm_output
    run_identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    arguments = dict(arguments, experiment_id=run_identity.sha256, run_identity=run_identity)
    records = record_vllm_generations(item, output, profile, **arguments)
    assert tuple(record.key for record in records) == run_identity.expected_keys
    assert all(record.experiment_id == run_identity.sha256 for record in records)
    assert all(record_from_json(record_to_json(record)) == record for record in records)


@pytest.mark.parametrize("changed", ["item", "experiment"])
def test_run_bound_native_output_refuses_wrong_declaration(native_vllm_output, changed):
    item, output, profile, arguments = native_vllm_output
    run_identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    arguments = dict(arguments, experiment_id=run_identity.sha256, run_identity=run_identity)
    output.outputs = None
    if changed == "item":
        item = replace(item, prompt="different question")
    else:
        arguments["experiment_id"] = "other-experiment"
    with pytest.raises(ValueError, match="frozen evaluation run/profile|frozen run dataset"):
        record_vllm_generations(item, output, profile, **arguments)


@pytest.mark.parametrize("native,recorded", [("stop", "stop"), ("length", "length"),
                                            ("abort", "cancelled"), ("error", "error")])
def test_native_adapter_preserves_finish_metadata(native_vllm_output, native, recorded):
    item, output, profile, arguments = native_vllm_output
    for completion in output.outputs:
        completion.finish_reason = native
        completion.token_ids = []
        completion.text = ""
    records = record_vllm_generations(item, output, profile, **arguments)
    assert all(record.finish_reason == recorded and record.backend_finish_reason == native for record in records)
    assert all(record.generated_token_ids == () and record.generated_text == "" for record in records)
    assert observe_generation_health(records[0], thinking_required=True).cap_hit is (native == "length")


@pytest.mark.parametrize("change", ["unfinished", "missing", "duplicate", "unknown_finish", "prompt",
                                  "request", "version", "stop", "boolean_prompt"])
def test_native_adapter_refuses_unplanned_or_incomplete_outputs(native_vllm_output, change):
    item, output, profile, arguments = native_vllm_output
    if change == "unfinished":
        output.finished = False
    elif change == "missing":
        output.outputs.pop()
    elif change == "duplicate":
        output.outputs[1].index = 0
    elif change == "unknown_finish":
        output.outputs[0].finish_reason = "repetition"
    elif change == "prompt":
        output.prompt_token_ids[0] = 99
    elif change == "request":
        output.request_id = "other"
    elif change == "version":
        arguments["backend_version"] = "other"
    elif change == "stop":
        output.outputs[0].stop_reason = "unplanned stop"
    else:
        output.prompt_token_ids[1] = True
    with pytest.raises(ValueError):
        record_vllm_generations(item, output, profile, **arguments)


def test_native_adapter_preserves_declared_eos_stop_detail(native_vllm_output):
    item, output, profile, arguments = native_vllm_output
    output.outputs[0].stop_reason = 2
    records = record_vllm_generations(item, output, profile, **arguments)
    assert records[0].backend_finish_reason == "stop"
    assert records[0].backend_stop_reason == 2
    assert record_from_json(record_to_json(records[0])) == records[0]


@pytest.mark.parametrize("failure", ["prefix", "seed_policy", "prompt_limit", "generation_limit"])
def test_native_adapter_refuses_unsupported_context_and_limits(native_vllm_output, failure):
    item, output, profile, arguments = native_vllm_output
    if failure == "prefix":
        arguments["thinking_open_from_prompt"] = None
    elif failure == "seed_policy":
        profile = replace(profile, track="validated_repair", seed_policy="per_sample_fixture")
    elif failure == "prompt_limit":
        profile = replace(profile, track="validated_repair", max_new_tokens=2, max_model_len=5)
    else:
        profile = replace(profile, track="validated_repair", max_new_tokens=1)
    with pytest.raises(ValueError):
        record_vllm_generations(item, output, profile, **arguments)


def test_native_request_ids_may_differ_between_matched_candidates(native_vllm_output):
    item, output, profile, arguments = native_vllm_output
    baseline = record_vllm_generations(item, output, profile, **arguments)
    output.request_id = "native-2"
    candidate = record_vllm_generations(item, output, profile, **dict(arguments, candidate_id="edit", expected_request_id="native-2"))
    pairs = pair_generation_records(baseline, candidate, expected_keys=tuple(record.key for record in baseline))
    assert len(pairs) == 4
    assert pairs[0][0].backend_request_id != pairs[0][1].backend_request_id


def test_legacy_generation_json_has_unavailable_native_snapshot(generations):
    baseline, _, _ = generations
    payload = json.loads(record_to_json(baseline[0]))
    for field in ("prompt_token_ids", "backend_request_id", "backend_finish_reason", "backend_stop_reason"):
        payload.pop(field)
    restored = record_from_json(json.dumps(payload))
    assert restored == baseline[0]
    payload["prompt_token_ids"] = "not token IDs"
    with pytest.raises(ValueError, match="JSON array"):
        record_from_json(json.dumps(payload))


@pytest.mark.parametrize(
    "changes", [{"prompt_tokens": True}, {"generated_token_ids": [1]},
                {"generated_token_ids": (-1,)}, {"finish_reason": "inferred_cap"},
                {"prompt_token_ids": list(range(10))}, {"prompt_token_ids": (1,)},
                {"prompt_token_ids": (True,) + tuple(range(9))},
                {"backend_request_id": ""}, {"backend_finish_reason": []},
                {"backend_finish_reason": "length"}, {"backend_stop_reason": True},
                {"backend_stop_reason": 2}],
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


@pytest.fixture
def diagnostic_generations(source_profile):
    dataset = DatasetManifest((EvaluationItem("a", "fixture", "v1", "diagnostic", "prompt", "4"),))
    first = GenerationRecord("experiment-v1", "base", source_profile.sha256, SampleKey("a", 0, 0),
                             10, (1, 3, 4, 5), r"reasoning</think>\boxed{4}", "stop", thinking_open_from_prompt=True)
    health = observe_generation_health(
        first, thinking_required=True, profile=source_profile, repetition_spec=RepetitionSpec(max_tokens=64),
        tail_spec=TokenTailSpec(source_profile.tokenizer_repo, source_profile.tokenizer_revision, (9,), (2,),
                               min_run_tokens=3, max_tokens=64),
    )
    quality = QualityObservation(source_profile.extraction_version, "math-verify@0.9.0", "4", "usable", True, "graded")
    records = tuple(replace(first, key=SampleKey("a", index, 0), health=health, quality=quality)
                    for index in range(source_profile.samples_per_item))
    return dataset, records, source_profile


@pytest.mark.parametrize("cap", [False, True])
def test_no_edit_diagnosis_does_not_force_localization_or_classify_caps(diagnostic_generations, cap):
    dataset, records, profile = diagnostic_generations
    if cap:
        records = tuple(replace(record, finish_reason="length", health=replace(record.health, cap_hit=True)) for record in records)
    result = diagnose_generation_records(records, dataset=dataset, profile=profile)
    assert result.outcome == "no_patch"
    assert result.cap_hits == (4 if cap else 0)
    assert result.sample_keys == tuple(record.key for record in records)
    assert result.dataset_sha256 == dataset.sha256 and result.profile_sha256 == profile.sha256
    assert result.scope == "recorded_evidence_only"
    assert not result.localization_authorized
    assert result == diagnose_generation_records(tuple(reversed(records)), dataset=dataset, profile=profile)
    with pytest.raises(FrozenInstanceError):
        result.outcome = "patched"


@pytest.mark.parametrize("missing", ["health", "quality", "tail", "repetition"])
def test_no_edit_diagnosis_preserves_unavailable_evidence(diagnostic_generations, missing):
    dataset, records, profile = diagnostic_generations
    if missing in ("health", "quality"):
        records = tuple(replace(record, **{missing: None}) for record in records)
    elif missing == "tail":
        records = tuple(replace(record, health=replace(record.health, token_tail=None, observation_version="finish-thinking-repetition-v2")) for record in records)
    else:
        records = tuple(replace(record, health=replace(record.health, repetition=None, repetition_detected=None)) for record in records)
    result = diagnose_generation_records(records, dataset=dataset, profile=profile)
    assert result.outcome == "insufficient_evidence"
    assert not result.localization_authorized


@pytest.mark.parametrize("pattern", ["repetition", "tail"])
def test_no_edit_diagnosis_literal_patterns_are_not_repair_authority(diagnostic_generations, pattern):
    dataset, records, profile = diagnostic_generations
    tokens = (1, 3, 4) * 3 if pattern == "repetition" else (1, 3, 4) + (9,) * 3
    first = replace(records[0], health=None, generated_token_ids=tokens)
    health = observe_generation_health(
        first, thinking_required=True, profile=profile, repetition_spec=RepetitionSpec(max_tokens=64),
        tail_spec=TokenTailSpec(profile.tokenizer_repo, profile.tokenizer_revision, (9,), (2,), min_run_tokens=3, max_tokens=64),
    )
    records = tuple(replace(record, generated_token_ids=tokens, health=health) for record in records)
    result = diagnose_generation_records(records, dataset=dataset, profile=profile)
    assert result.outcome == "insufficient_evidence"
    assert ("literal_repetition_observed" if pattern == "repetition" else "declared_tail_observed") in result.reasons
    assert not result.localization_authorized


@pytest.mark.parametrize("role", ["confirmation", "benchmark", "clean_reference"])
def test_no_edit_diagnosis_refuses_protected_role_use(diagnostic_generations, role):
    dataset, records, profile = diagnostic_generations
    dataset = DatasetManifest((replace(dataset.items[0], role=role),))
    with pytest.raises(ValueError, match="protected dataset roles"):
        diagnose_generation_records(records, dataset=dataset, profile=profile)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "profile", "quality", "framing"])
def test_no_edit_diagnosis_refuses_incomplete_identity_or_stale_framing(diagnostic_generations, failure):
    dataset, records, profile = diagnostic_generations
    if failure == "missing":
        records = records[:-1]
    elif failure == "duplicate":
        records = records[:-1] + (records[0],)
    elif failure == "profile":
        profile = replace(profile, backend_version="different-version")
    elif failure == "quality":
        records = tuple(replace(record, quality=replace(record.quality, grader_id="other-grader")) for record in records)
    else:
        records = tuple(replace(record, generated_text="missing closure") for record in records)
    with pytest.raises(ValueError):
        diagnose_generation_records(records, dataset=dataset, profile=profile)


@pytest.mark.parametrize("failure", ["outcome", "count", "keys"])
def test_no_edit_diagnosis_cannot_be_forged_as_repair_report(diagnostic_generations, failure):
    dataset, records, profile = diagnostic_generations
    result = diagnose_generation_records(records, dataset=dataset, profile=profile)
    with pytest.raises(ValueError):
        replace(result, **{"outcome": "patched"} if failure == "outcome" else {"cap_hits": True} if failure == "count" else {"sample_keys": list(result.sample_keys)})


@pytest.mark.parametrize("status", ["incorrect", "unavailable", "unusable"])
def test_no_edit_diagnosis_preserves_independent_task_quality(diagnostic_generations, status):
    dataset, records, profile = diagnostic_generations
    quality = records[0].quality
    if status == "incorrect":
        quality = replace(quality, correctness=False)
    elif status == "unavailable":
        quality = replace(quality, correctness=None, grader_status="unavailable", reason="fixture_unavailable")
    else:
        quality = QualityObservation(profile.extraction_version, "math-verify@0.9.0", None, "empty", False, "graded")
    records = tuple(replace(record, quality=quality) for record in records)
    result = diagnose_generation_records(records, dataset=dataset, profile=profile)
    assert result.outcome == "insufficient_evidence"
    assert ("quality_unavailable" if status == "unavailable" else "recorded_task_quality_failure") in result.reasons
    assert not result.localization_authorized


def test_no_edit_diagnosis_retains_backend_failure(diagnostic_generations):
    dataset, records, profile = diagnostic_generations
    first = replace(records[0], health=None, finish_reason="error")
    health = observe_generation_health(
        first, thinking_required=True, profile=profile, repetition_spec=RepetitionSpec(max_tokens=64),
        tail_spec=TokenTailSpec(profile.tokenizer_repo, profile.tokenizer_revision, (9,), (2,), min_run_tokens=3, max_tokens=64),
    )
    records = tuple(replace(record, finish_reason="error", health=health) for record in records)
    result = diagnose_generation_records(records, dataset=dataset, profile=profile)
    assert result.outcome == "insufficient_evidence"
    assert "backend_failure" in result.reasons
    assert not result.localization_authorized


@pytest.mark.parametrize("limit", ["records", "text"])
def test_no_edit_diagnosis_refuses_oversized_inputs(diagnostic_generations, limit):
    dataset, records, profile = diagnostic_generations
    records = records * 33 if limit == "records" else tuple(replace(record, generated_text="x" * 262145) for record in records)
    with pytest.raises(ValueError, match="bounded record|token/text envelope"):
        diagnose_generation_records(records, dataset=dataset, profile=profile)


def test_no_edit_diagnosis_refuses_noncanonical_report_digest(diagnostic_generations):
    dataset, records, profile = diagnostic_generations
    result = diagnose_generation_records(records, dataset=dataset, profile=profile)
    with pytest.raises(ValueError, match="canonical identity digests"):
        replace(result, evidence_sha256="not-a-digest")


def test_diagnosis_batch_preserves_strict_canonical_readonly_abstention(diagnostic_generations):
    dataset, records, profile = diagnostic_generations
    identity = EvaluationRunIdentity(dataset, profile, "e" * 40)
    identity = EvaluationRunIdentity.from_json(identity.to_json(), expected_sha256=identity.sha256)
    records = tuple(replace(record, experiment_id=identity.sha256, prompt_token_ids=tuple(range(10))) for record in records)
    before = tuple(record_to_json(record) for record in records)
    result = diagnose_generation_batch(identity, list(reversed(records)), candidate_id="base")
    assert result.outcome == "no_patch" and result.scope == "recorded_evidence_only"
    assert not result.localization_authorized
    assert result.sample_keys == identity.expected_keys
    assert result.dataset_sha256 == identity.dataset.sha256 and result.profile_sha256 == profile.sha256
    assert tuple(record_to_json(record) for record in records) == before
    assert result == diagnose_generation_records(records, dataset=dataset, profile=profile, run_identity=identity)


@pytest.fixture
def complete_diagnosis_batch(diagnostic_generations):
    dataset, records, profile = diagnostic_generations
    identity = EvaluationRunIdentity(dataset, profile, "e" * 40)
    records = tuple(replace(record, experiment_id=identity.sha256, prompt_token_ids=tuple(range(10))) for record in records)
    return identity, records


@pytest.mark.parametrize("role", ["confirmation", "benchmark", "clean_reference"])
def test_diagnosis_batch_refuses_protected_roles(complete_diagnosis_batch, role):
    identity, records = complete_diagnosis_batch
    identity = replace(identity, dataset=DatasetManifest((replace(identity.dataset.items[0], role=role),)))
    records = tuple(replace(record, experiment_id=identity.sha256) for record in records)
    with pytest.raises(ValueError, match="protected dataset roles"):
        diagnose_generation_batch(identity, records, candidate_id="base")


@pytest.mark.parametrize("changed", ["missing_key", "prompt_missing", "prompt_drift", "context_missing", "candidate",
                                     "experiment", "quality_grader", "generation_limit"])
def test_diagnosis_batch_refuses_incomplete_or_stale_evidence(complete_diagnosis_batch, changed):
    identity, records = complete_diagnosis_batch
    if changed == "missing_key":
        records = records[:-1]
    elif changed == "prompt_missing":
        records = tuple(replace(record, prompt_token_ids=None) for record in records)
    elif changed == "prompt_drift":
        records = (replace(records[0], prompt_token_ids=(99,) + records[0].prompt_token_ids[1:]),) + records[1:]
    elif changed == "context_missing":
        records = tuple(replace(record, health=None, thinking_open_from_prompt=None) for record in records)
    elif changed in ("candidate", "experiment"):
        records = tuple(replace(record, **{{"candidate": "candidate_id", "experiment": "experiment_id"}[changed]: "other"}) for record in records)
    elif changed == "quality_grader":
        records = tuple(replace(record, quality=replace(record.quality, grader_id="other")) for record in records)
    else:
        records = (replace(records[0], generated_token_ids=(1,) * 32769, health=None),) + records[1:]
    with pytest.raises(ValueError):
        diagnose_generation_batch(identity, records, candidate_id="base")


@pytest.mark.parametrize("missing", ["quality", "health"])
def test_diagnosis_batch_keeps_missing_observations_insufficient(complete_diagnosis_batch, missing):
    identity, records = complete_diagnosis_batch
    records = tuple(replace(record, **{missing: None}) for record in records)
    result = diagnose_generation_batch(identity, records, candidate_id="base")
    assert result.outcome == "insufficient_evidence" and not result.localization_authorized
    assert missing + "_unavailable" in result.reasons


@pytest.mark.parametrize("cap", [False, True])
def test_diagnosis_batch_does_not_promote_caps_to_pathology(complete_diagnosis_batch, cap):
    identity, records = complete_diagnosis_batch
    if cap:
        records = tuple(replace(record, finish_reason="length", health=replace(record.health, cap_hit=True)) for record in records)
    result = diagnose_generation_batch(identity, records, candidate_id="base")
    assert result.outcome == "no_patch" and not result.localization_authorized
    assert result.cap_hits == (len(records) if cap else 0)


def test_diagnosis_batch_accepts_controlled_grading_without_repair_authority(complete_diagnosis_batch):
    identity, records = complete_diagnosis_batch
    records = tuple(replace(record, quality=None) for record in records)
    graded = evaluate_generation_batch(identity, records, candidate_id="base")
    result = diagnose_generation_batch(identity, graded, candidate_id="base")
    assert result.outcome == "no_patch" and result.scope == "recorded_evidence_only"
    assert not result.localization_authorized
    assert all(record.quality is None for record in records)
    assert all(record.health is original.health for record, original in zip(graded, records))


@pytest.mark.parametrize("changed", ["run", "count", "generator"])
def test_diagnosis_batch_refuses_invalid_admission(complete_diagnosis_batch, changed):
    identity, records = complete_diagnosis_batch
    if changed == "run":
        identity = None
    elif changed == "count":
        records = records[:1] * 129
    else:
        def forbidden_generator():
            raise AssertionError("unbounded input must not be consumed")
            yield None
        records = forbidden_generator()
    with pytest.raises(ValueError):
        diagnose_generation_batch(identity, records, candidate_id="base")


def test_diagnosis_batch_retains_literal_pattern_insufficiency(complete_diagnosis_batch):
    identity, records = complete_diagnosis_batch
    first = replace(records[0], generated_token_ids=(1, 3, 4) * 3 + (9,) * 3 + (2,), health=None)
    profile = identity.profile
    health = observe_generation_health(first, thinking_required=True, profile=profile, repetition_spec=RepetitionSpec(max_tokens=64),
                                       tail_spec=TokenTailSpec(profile.tokenizer_repo, profile.tokenizer_revision, (9,), (2,), min_run_tokens=3, max_tokens=64))
    records = tuple(replace(first, key=record.key, health=health) for record in records)
    result = diagnose_generation_batch(identity, records, candidate_id="base")
    assert result.outcome == "insufficient_evidence" and not result.localization_authorized
    assert "literal_repetition_observed" in result.reasons and "declared_tail_observed" in result.reasons


def test_run_bound_no_edit_diagnosis_preserves_matching_data(diagnostic_generations):
    dataset, records, profile = diagnostic_generations
    run_identity = EvaluationRunIdentity(dataset, profile, "e" * 40)
    records = tuple(replace(record, experiment_id=run_identity.sha256) for record in records)
    result = diagnose_generation_records(records, dataset=dataset, profile=profile, run_identity=run_identity)
    assert result.outcome == "no_patch"
    assert not result.localization_authorized
    changed = DatasetManifest((replace(dataset.items[0], reference_answer="7"),))
    with pytest.raises(ValueError, match="frozen evaluation run"):
        diagnose_generation_records(records, dataset=changed, profile=profile, run_identity=run_identity)


def test_run_bound_diagnosis_uses_canonical_dataset_identity(diagnostic_generations):
    dataset, records, profile = diagnostic_generations
    second = replace(dataset.items[0], item_id="b", prompt="other prompt")
    dataset = DatasetManifest((dataset.items[0], second))
    run_identity = EvaluationRunIdentity(dataset, profile, "e" * 40)
    records = records + tuple(replace(record, key=replace(record.key, item_id="b")) for record in records)
    records = tuple(replace(record, experiment_id=run_identity.sha256) for record in records)
    result = diagnose_generation_records(records, dataset=dataset, profile=profile, run_identity=run_identity)
    assert result == diagnose_generation_records(records, dataset=DatasetManifest(tuple(reversed(dataset.items))),
                                                  profile=profile, run_identity=run_identity)


def test_run_persistence_restores_canonical_frozen_declarations(source_profile):
    first = EvaluationItem("a", "fixture", "v1", "development", reference_answer="4",
                           messages=(ChatMessage("user", "first"),),
                           instructions=(InstructionSpec("fixture:constraint", '{"expected":"ok"}'),))
    second = replace(first, item_id="b", messages=(ChatMessage("user", "second"),))
    identity = EvaluationRunIdentity(DatasetManifest((second, first)), source_profile, "e" * 40)
    serialized = identity.to_json()
    restored = EvaluationRunIdentity.from_json(serialized, expected_sha256=identity.sha256)
    assert restored.sha256 == identity.sha256
    assert restored.profile == identity.profile
    assert restored.dataset.items == (first, second)
    assert restored.expected_keys == identity.expected_keys
    assert restored.to_json() == serialized
    assert restored.scope == "declared_data_profile_code_only"
    assert serialized == replace(identity, dataset=DatasetManifest((first, second))).to_json()
    with pytest.raises(FrozenInstanceError):
        restored.code_revision = "f" * 40


@pytest.mark.parametrize("changed", ["dataset_hash", "profile_hash", "run_hash", "code", "reference", "role",
                                     "profile", "scope", "schema", "boolean_schema", "type", "extra", "missing",
                                     "items", "profile_type", "nested_type", "nested_extra", "duplicate_items"])
def test_run_persistence_refuses_changed_or_malformed_declarations(source_profile, changed):
    item = EvaluationItem("a", "fixture", "v1", "development", "question", "4")
    identity = EvaluationRunIdentity(DatasetManifest((item,)), source_profile, "e" * 40)
    payload = json.loads(identity.to_json())
    if changed in ("dataset_hash", "profile_hash", "run_hash"):
        payload[{"dataset_hash": "dataset_sha256", "profile_hash": "profile_sha256", "run_hash": "sha256"}[changed]] = "f" * 64
    elif changed == "code":
        payload["code_revision"] = "f" * 40
    elif changed == "reference":
        payload["items"][0]["reference_answer"] = "7"
    elif changed == "role":
        payload["items"][0]["role"] = "confirmation"
    elif changed == "profile":
        payload["profile"]["chat_template_sha256"] = "d" * 64
    elif changed == "scope":
        payload["scope"] = "authenticated_execution"
    elif changed in ("schema", "boolean_schema"):
        payload["schema_version"] = 2 if changed == "schema" else True
    elif changed == "type":
        payload["record_type"] = "generation"
    elif changed == "extra":
        payload["accepted_repair"] = True
    elif changed == "missing":
        del payload["scope"]
    elif changed == "items":
        payload["items"] = []
    elif changed == "profile_type":
        payload["profile"] = []
    elif changed == "nested_type":
        payload["items"][0] = payload["profile"]
    elif changed == "nested_extra":
        payload["items"][0]["accepted_repair"] = True
    else:
        payload["items"].append(payload["items"][0])
    with pytest.raises(ValueError):
        EvaluationRunIdentity.from_json(json.dumps(payload))


@pytest.mark.parametrize("text", ["[]", "{}", '{"schema_version":1,"schema_version":1}', "NaN", '{"'])
def test_run_persistence_refuses_malformed_json(text):
    with pytest.raises(ValueError):
        EvaluationRunIdentity.from_json(text)


@pytest.mark.parametrize("expected", ["main", "A" * 64, "", True])
def test_run_persistence_requires_canonical_expected_identity(expected):
    with pytest.raises(ValueError, match="canonical SHA-256"):
        EvaluationRunIdentity.from_json("{}", expected_sha256=expected)


def test_run_persistence_refuses_consistent_but_unexpected_declaration(source_profile):
    item = EvaluationItem("a", "fixture", "v1", "development", "question", "4")
    identity = EvaluationRunIdentity(DatasetManifest((item,)), source_profile, "e" * 40)
    changed = replace(identity, code_revision="f" * 40)
    text = changed.to_json()
    assert EvaluationRunIdentity.from_json(text).sha256 == changed.sha256
    with pytest.raises(ValueError, match="identity mismatch"):
        EvaluationRunIdentity.from_json(text, expected_sha256=identity.sha256)


@pytest.mark.parametrize("text", [" " * 1048577, "\u00e9" * 600000])
def test_run_persistence_refuses_oversized_input_before_parsing(text):
    with pytest.raises(ValueError, match="UTF-8 budget"):
        EvaluationRunIdentity.from_json(text)


def test_run_persistence_refuses_oversized_output(source_profile):
    item = EvaluationItem("a", "fixture", "v1", "development", "x" * 1048576, "4")
    identity = EvaluationRunIdentity(DatasetManifest((item,)), source_profile, "e" * 40)
    with pytest.raises(ValueError, match="UTF-8 budget"):
        identity.to_json()


def test_run_batch_restores_and_hands_off_complete_canonical_records(native_vllm_output):
    item, output, profile, arguments = native_vllm_output
    second = replace(item, item_id="b", prompt="second question")
    identity = EvaluationRunIdentity(DatasetManifest((second, item)), profile, "e" * 40)
    records = record_vllm_generations(item, output, profile, **dict(arguments, experiment_id=identity.sha256), run_identity=identity)
    records += tuple(replace(record, key=replace(record.key, item_id="b"),
                             prompt_token_ids=(99,) + record.prompt_token_ids[1:]) for record in records)
    restored = EvaluationRunIdentity.from_json(identity.to_json(), expected_sha256=identity.sha256)
    supplied = list(reversed(records))
    result = restored.validate_generations(supplied, candidate_id="base")
    assert result == records
    assert tuple(record.key for record in result) == restored.expected_keys
    assert all(record is original for record, original in zip(result, records))
    supplied.clear()
    assert len(result) == 8


@pytest.mark.parametrize("changed", ["missing", "duplicate", "unexpected", "mixed_candidate", "candidate", "experiment",
                                     "profile", "seed", "prompt_missing", "context_missing", "prompt_drift", "context_drift",
                                     "prompt_limit", "generation_limit", "aggregate_tokens", "ascii_text", "utf8_text",
                                     "native_stop_token", "native_stop_string"])
def test_run_batch_refuses_incomplete_stale_or_overbudget_records(native_vllm_output, changed):
    item, output, profile, arguments = native_vllm_output
    if changed in ("prompt_limit", "generation_limit"):
        profile = replace(profile, track="validated_repair", max_new_tokens=2, max_model_len=12)
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    records = record_vllm_generations(item, output, profile, **dict(arguments, experiment_id=identity.sha256), run_identity=identity)
    if changed == "missing":
        records = records[:-1]
    elif changed == "duplicate":
        records += records[:1]
    elif changed == "unexpected":
        records = (replace(records[0], key=replace(records[0].key, item_id="other")),) + records[1:]
    elif changed == "mixed_candidate":
        records = (replace(records[0], candidate_id="edit"),) + records[1:]
    elif changed in ("candidate", "experiment", "profile"):
        records = tuple(replace(record, **{{"candidate": "candidate_id", "experiment": "experiment_id", "profile": "profile_id"}[changed]: "other"}) for record in records)
    elif changed == "seed":
        records = (replace(records[0], key=replace(records[0].key, seed=1)),) + records[1:]
    elif changed in ("prompt_missing", "context_missing"):
        records = tuple(replace(record, **{{"prompt_missing": "prompt_token_ids", "context_missing": "thinking_open_from_prompt"}[changed]: None}) for record in records)
    elif changed == "prompt_drift":
        records = (replace(records[0], prompt_token_ids=(99,) + records[0].prompt_token_ids[1:]),) + records[1:]
    elif changed == "context_drift":
        records = (replace(records[0], thinking_open_from_prompt=False),) + records[1:]
    elif changed == "prompt_limit":
        records = tuple(replace(record, prompt_tokens=11, prompt_token_ids=tuple(range(11))) for record in records)
    elif changed == "generation_limit":
        records = tuple(replace(record, generated_token_ids=(1, 1, 1)) for record in records)
    elif changed == "aggregate_tokens":
        records = (replace(records[0], generated_token_ids=(1,) * (262144 - 46 + 1)),) + records[1:]
    elif changed in ("ascii_text", "utf8_text"):
        text = "x" * (1048576 - 18 + 1) if changed == "ascii_text" else "\u00e9" * 524279 + "x"
        records = (replace(records[0], generated_text=text),) + records[1:]
    else:
        records = tuple(replace(record, backend_stop_reason=999 if changed == "native_stop_token" else "undeclared") for record in records)
    with pytest.raises(ValueError):
        identity.validate_generations(records, candidate_id="base")


@pytest.mark.parametrize("changed", ["empty", "over_count", "invalid_record", "generator", "subclass"])
def test_run_batch_refuses_unbounded_or_nonplain_collections(native_vllm_output, changed):
    item, _, profile, _ = native_vllm_output
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)

    def forbidden_generator():
        raise AssertionError("unbounded inputs must not be consumed")
        yield None

    class ForbiddenList(list):
        def __iter__(self):
            raise AssertionError("custom inputs must not be consumed")

    supplied = {"empty": (), "over_count": (None,) * 4097, "invalid_record": (None,),
                "generator": forbidden_generator(), "subclass": ForbiddenList([None])}[changed]
    with pytest.raises(ValueError, match="bounded plain record"):
        identity.validate_generations(supplied, candidate_id="base")


@pytest.mark.parametrize("native,recorded", [("stop", "stop"), ("length", "length"), ("abort", "cancelled"), ("error", "error")])
def test_run_batch_preserves_backend_outcomes_and_attached_health(native_vllm_output, native, recorded):
    item, output, profile, arguments = native_vllm_output
    for completion in output.outputs:
        completion.finish_reason = native
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    records = record_vllm_generations(item, output, profile, **dict(arguments, experiment_id=identity.sha256), run_identity=identity)
    records = tuple(replace(record, health=observe_generation_health(record, thinking_required=profile.thinking, profile=profile)) for record in records)
    result = identity.validate_generations(records, candidate_id="base")
    assert all(record is original for record, original in zip(result, records))
    assert all(record.finish_reason == recorded and record.backend_finish_reason == native for record in result)
    assert all(record.health.cap_hit is (native == "length") for record in result)
    assert all(record.quality is None for record in result)


def test_run_batch_refuses_attached_health_policy_drift(native_vllm_output):
    item, output, profile, arguments = native_vllm_output
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    records = record_vllm_generations(item, output, profile, **dict(arguments, experiment_id=identity.sha256), run_identity=identity)
    records = tuple(replace(record, health=observe_generation_health(record, thinking_required=False)) for record in records)
    with pytest.raises(ValueError, match="health thinking policy"):
        identity.validate_generations(records, candidate_id="base")


def test_run_batch_preserves_exact_quality_and_token_bound_health(diagnostic_generations):
    dataset, records, profile = diagnostic_generations
    identity = EvaluationRunIdentity(dataset, profile, "e" * 40)
    records = tuple(replace(record, experiment_id=identity.sha256, prompt_token_ids=tuple(range(10))) for record in records)
    result = identity.validate_generations(records, candidate_id="base")
    assert all(record is original and record.health is original.health and record.quality is original.quality
               for record, original in zip(result, records))
    assert all(record_from_json(record_to_json(record)) == record for record in result)
    assert all(record.health.repetition.token_ids_sha256 is not None and record.health.token_tail.profile_sha256 == profile.sha256
               for record in result)


def test_run_batch_refuses_mixed_observation_contracts(diagnostic_generations):
    dataset, records, profile = diagnostic_generations
    identity = EvaluationRunIdentity(dataset, profile, "e" * 40)
    records = tuple(replace(record, experiment_id=identity.sha256, prompt_token_ids=tuple(range(10))) for record in records)
    records = (replace(records[0], health=None),) + records[1:]
    with pytest.raises(ValueError, match="health identities"):
        identity.validate_generations(records, candidate_id="base")


def test_evaluation_run_identity_hashes_frozen_data_profile_code_and_schedule(source_profile):
    first = EvaluationItem("a", "fixture", "v1", "development", "first", "4")
    second = replace(first, item_id="b", prompt="second")
    dataset = DatasetManifest((second, first))
    identity = EvaluationRunIdentity(dataset, source_profile, "e" * 40)
    assert identity.sha256 == replace(identity, dataset=DatasetManifest((first, second))).sha256
    assert identity.expected_keys == tuple(SampleKey(item_id, index, 0) for item_id in ("a", "b") for index in range(4))
    assert identity.sha256 != replace(identity, code_revision="f" * 40).sha256
    assert identity.sha256 != replace(identity, profile=replace(source_profile, chat_template_sha256="d" * 64)).sha256
    assert identity.sha256 != replace(identity, dataset=DatasetManifest((replace(first, reference_answer="7"), second))).sha256
    assert identity.sha256 != replace(identity, dataset=DatasetManifest((replace(first, role="confirmation"), second))).sha256
    with pytest.raises(FrozenInstanceError):
        identity.code_revision = "f" * 40


@pytest.mark.parametrize("revision", ["main", "A" * 40, "", True])
def test_evaluation_run_identity_requires_exact_code_pin(source_profile, revision):
    dataset = DatasetManifest((EvaluationItem("a", "fixture", "v1", "development", "prompt"),))
    with pytest.raises(ValueError, match="exact code revision"):
        EvaluationRunIdentity(dataset, source_profile, revision)


@pytest.mark.parametrize("failure", ["samples", "seed_policy"])
def test_evaluation_run_identity_refuses_unsupported_schedule(source_profile, failure):
    dataset = DatasetManifest((EvaluationItem("a", "fixture", "v1", "development", "prompt"),))
    profile = replace(source_profile, track="validated_repair", **({"samples_per_item": 4097} if failure == "samples" else {"seed_policy": "other-policy"}))
    with pytest.raises(ValueError, match="bounded declared sampling schedule"):
        EvaluationRunIdentity(dataset, profile, "e" * 40)


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


@pytest.mark.parametrize("task,samples", [("math500", 4), ("gsm8k", 2), ("ifeval", 2)])
def test_vllm_sampling_kwargs_bind_declared_profile(source_profile, task, samples):
    profile = replace(source_profile, backend="vllm", task=task, samples_per_item=samples,
                      prompt_template="ifeval_user_v1" if task == "ifeval" else "math_qwen_boxed_v1",
                      grader="ifeval-strict" if task == "ifeval" else "math-verify")
    before = record_to_json(profile)
    arguments = profile.vllm_sampling_kwargs(backend_version=profile.backend_version)
    assert arguments == {
        "n": samples, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "presence_penalty": 1.5, "max_tokens": 32768, "seed": 0,
        "stop": [], "stop_token_ids": [2], "ignore_eos": False,
    }
    arguments["stop_token_ids"].append(7)
    arguments["stop"].append("unexpected")
    arguments["n"] = 99
    assert record_to_json(profile) == before
    assert profile.vllm_sampling_kwargs(backend_version=profile.backend_version)["stop_token_ids"] == [2]
    assert profile.vllm_sampling_kwargs(backend_version=profile.backend_version)["stop"] == []


@pytest.mark.parametrize("failure", ["backend", "version", "seed_policy"])
def test_vllm_sampling_kwargs_refuse_incompatible_declarations(source_profile, failure):
    profile = source_profile if failure == "backend" else replace(source_profile, backend="vllm")
    if failure == "seed_policy":
        profile = replace(profile, track="validated_repair", seed_policy="per_sample_fixture")
    with pytest.raises(ValueError):
        profile.vllm_sampling_kwargs(backend_version="other-version" if failure == "version" else profile.backend_version)


def test_vllm_sampling_kwargs_translate_repair_profile_values(source_profile):
    profile = replace(source_profile, track="validated_repair", backend="vllm",
                      temperature=0.8, top_p=0.9, top_k=5, presence_penalty=0.2,
                      max_new_tokens=64, max_model_len=96, samples_per_item=3,
                      engine_seed=7, sampling_seed=11, eos_token_ids=(8, 2), stop_sequences=("end", "STOP"))
    assert profile.vllm_sampling_kwargs(backend_version=profile.backend_version) == {
        "n": 3, "temperature": 0.8, "top_p": 0.9, "top_k": 5,
        "presence_penalty": 0.2, "max_tokens": 64, "seed": 11,
        "stop": ["end", "STOP"], "stop_token_ids": [8, 2], "ignore_eos": False,
    }


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
                {"grader_status": "error"}, {"reason": 42},
                {"instruction_following": [True]}, {"instruction_following": (1,)},
                {"instruction_following": (False,)}],
)
def test_quality_observation_rejects_contradictory_or_mutable_schema(changes):
    quality = QualityObservation("source-v1", "math-verify@0.9.0", "4", "usable", True, "graded")
    with pytest.raises(ValueError):
        replace(quality, **changes)


def test_token_repetition_span_stops_before_continuation(generations):
    baseline, _, _ = generations
    generation = replace(baseline[0], generated_token_ids=(1, 2, 3) + (7, 8, 9) * 3 + (10, 11, 12))
    spec = RepetitionSpec()
    observation = observe_token_repetition(generation, spec)
    assert observation.status == "complete"
    assert observation.token_count == 15
    assert observation.spec_sha256 == spec.sha256
    assert [(span.start, span.end, span.period, span.repeats) for span in observation.spans] == [(3, 12, 3, 3)]
    with pytest.raises(AttributeError):
        observation.spans[0].end = 15


def test_token_repetition_observation_does_not_infer_loop_from_cap(generations):
    baseline, _, _ = generations
    generation = replace(baseline[0], finish_reason="length", generated_token_ids=tuple(range(12)))
    observation = observe_token_repetition(generation, RepetitionSpec())
    assert observation.status == "complete"
    assert observation.spans == ()
    assert observe_generation_health(generation, thinking_required=True).cap_hit
    assert generation.quality is None


@pytest.mark.parametrize("tokens,expected", [((), []), ((1, 2, 3), []), ((1, 2) * 2, []), ((1, 2) * 3, [(0, 6, 2, 3)]), ((5,) + (1, 2) * 3 + (6,) + (3, 4) * 3 + (7,), [(1, 7, 2, 3), (8, 14, 2, 3)]), ((1, 2) * 3 + (1,), [(0, 6, 2, 3)]), ((9,) * 12, [(0, 12, 1, 12)])])
def test_token_repetition_golden_exact_nonoverlapping_spans(generations, tokens, expected):
    baseline, _, _ = generations
    generation = replace(baseline[0], generated_token_ids=tokens)
    observation = observe_token_repetition(generation, RepetitionSpec(min_period=1, max_period=6, max_tokens=128))
    assert observation.status == "complete"
    assert [(span.start, span.end, span.period, span.repeats) for span in observation.spans] == expected


@pytest.mark.parametrize("finish,tokens,reason", [("cancelled", (1, 2) * 3, "backend_failure"), ("error", (1, 2) * 3, "backend_failure"), ("stop", tuple(range(10)), "token_budget_exceeded"), ("length", (1, 2) * 5, "token_budget_exceeded")])
def test_token_repetition_unavailable_is_not_clean(generations, finish, tokens, reason):
    baseline, _, _ = generations
    generation = replace(baseline[0], finish_reason=finish, generated_token_ids=tokens)
    observation = observe_token_repetition(generation, RepetitionSpec(max_tokens=9))
    assert observation.status == "unavailable"
    assert observation.reason == reason
    assert observation.spans == ()
    assert observation.token_count == len(tokens)


@pytest.mark.parametrize("change", [{"min_period": 0}, {"min_period": True}, {"min_period": 49}, {"max_period": 49}, {"min_repeats": 1}, {"max_tokens": 8}, {"max_tokens": 32769}, {"algorithm": "legacy-tail-flood"}])
def test_token_repetition_spec_is_strict_frozen_and_fingerprinted(change):
    spec = RepetitionSpec()
    with pytest.raises(ValueError):
        replace(spec, **change)
    with pytest.raises(AttributeError):
        spec.max_tokens = 128
    assert replace(spec, max_tokens=128).sha256 != spec.sha256


def test_token_repetition_schema_refuses_inconsistent_or_mutable_spans():
    span = RepetitionSpan(1, 7, 2, 3)
    for change in ({"start": True}, {"end": 8}, {"period": 0}, {"repeats": 1}):
        with pytest.raises(ValueError):
            replace(span, **change)
    observation = RepetitionObservation(RepetitionSpec().sha256, 9, "complete", (span,))
    for change in ({"spans": [span]}, {"spans": (span, span)}, {"token_count": 6}, {"status": "unavailable"}, {"reason": "unexpected"}, {"spec_sha256": "bad"}):
        with pytest.raises(ValueError):
            replace(observation, **change)


@pytest.mark.parametrize("finish,tokens,detected", [("stop", (1, 2, 3) * 3, True), ("length", tuple(range(12)), False), ("error", (1, 2, 3) * 3, None)])
def test_repetition_health_roundtrip_is_token_bound_and_independent(generations, finish, tokens, detected):
    baseline, _, _ = generations
    generation = replace(baseline[0], finish_reason=finish, generated_token_ids=tokens)
    health = observe_generation_health(generation, thinking_required=True, repetition_spec=RepetitionSpec(max_tokens=128))
    assert health.repetition_detected is detected
    assert health.cap_hit is (finish == "length")
    assert health.tail_degeneration_detected is None
    assert health.observation_version == "finish-thinking-repetition-v2"
    generation = replace(generation, health=health)
    restored = record_from_json(record_to_json(generation))
    assert restored == generation
    assert isinstance(restored.health.repetition.spans, tuple)
    with pytest.raises(ValueError, match="exact generated tokens"):
        replace(generation, generated_token_ids=tuple(token + 100 for token in tokens))


@pytest.mark.parametrize("kind", ["spec", "thinking", "missing", "version", "mixed_side"])
def test_repetition_health_pairing_refuses_different_contracts(generations, kind):
    baseline, candidate, expected = generations
    spec = RepetitionSpec(max_tokens=128)
    base_health = observe_generation_health(baseline[0], thinking_required=True, repetition_spec=spec)
    baseline = tuple(replace(record, health=base_health) for record in baseline)
    candidate_spec = replace(spec, max_tokens=64) if kind == "spec" else spec
    candidate_health = observe_generation_health(candidate[0], thinking_required=True, repetition_spec=candidate_spec) if kind == "spec" else base_health
    if kind == "thinking":
        candidate_health = replace(candidate_health, thinking=replace(candidate_health.thinking, required=False))
    candidate = tuple(replace(record, health=candidate_health) for record in candidate)
    if kind == "missing":
        candidate = tuple(replace(record, health=None) for record in candidate)
    elif kind == "version":
        candidate = tuple(replace(record, health=observe_generation_health(record, thinking_required=True)) for record in candidate)
    elif kind == "mixed_side":
        candidate = (replace(candidate[0], health=observe_generation_health(candidate[0], thinking_required=True, repetition_spec=replace(spec, max_tokens=64))), candidate[1])
    with pytest.raises(ValueError, match="health|mixes"):
        pair_generation_records(baseline, candidate, expected_keys=expected)


def test_repetition_health_pairing_allows_different_observed_patterns(generations):
    baseline, candidate, expected = generations
    spec = RepetitionSpec(max_tokens=128)
    baseline = tuple(replace(record, generated_token_ids=(1, 2, 3) * 3) for record in baseline)
    candidate = tuple(replace(record, generated_token_ids=tuple(range(12))) for record in candidate)
    base_health = observe_generation_health(baseline[0], thinking_required=True, repetition_spec=spec)
    candidate_health = observe_generation_health(candidate[0], thinking_required=True, repetition_spec=spec)
    baseline = tuple(replace(record, health=base_health) for record in baseline)
    candidate = tuple(replace(record, health=candidate_health) for record in candidate)
    pairs = pair_generation_records(baseline, candidate, expected_keys=expected)
    assert len(pairs) == 2
    assert pairs[0][0].health.repetition_detected is True
    assert pairs[0][1].health.repetition_detected is False


@pytest.mark.parametrize("kind", ["missing", "span_object", "span_bounds", "token_digest", "version"])
def test_repetition_health_json_refuses_malformed_nested_observation(generations, kind):
    baseline, _, _ = generations
    generation = replace(baseline[0], generated_token_ids=(1, 2, 3) * 3)
    generation = replace(generation, health=observe_generation_health(generation, thinking_required=True, repetition_spec=RepetitionSpec(max_tokens=128)))
    payload = json.loads(record_to_json(generation))
    if kind == "missing":
        del payload["health"]["repetition"]
    elif kind == "span_object":
        payload["health"]["repetition"]["spans"] = {}
    elif kind == "span_bounds":
        payload["health"]["repetition"]["spans"][0]["end"] += 1
    elif kind == "token_digest":
        payload["health"]["repetition"]["token_ids_sha256"] = "0" * 64
    else:
        payload["health"]["observation_version"] = "finish-thinking-v1"
    with pytest.raises(ValueError):
        record_from_json(json.dumps(payload))


def test_repetition_health_budget_unavailable_roundtrip_and_legacy_json(generations):
    baseline, _, _ = generations
    generation = replace(baseline[0], generated_token_ids=tuple(range(12)))
    health = observe_generation_health(generation, thinking_required=True, repetition_spec=RepetitionSpec(max_tokens=9))
    assert health.repetition_detected is None
    assert health.repetition.reason == "token_budget_exceeded"
    generation = replace(generation, health=health)
    assert record_from_json(record_to_json(generation)) == generation
    legacy = replace(baseline[0], health=observe_generation_health(baseline[0], thinking_required=True))
    payload = json.loads(record_to_json(legacy))
    del payload["health"]["repetition"]
    assert record_from_json(json.dumps(payload)) == legacy


@pytest.mark.parametrize("tokens,start,end,meets", [((7,) + (0, 1) * 8 + (99,), 1, 17, True), ((99,), 0, 0, False)])
def test_token_tail_observes_whitespace_before_terminal_eos_without_quality_claim(generations, tokens, start, end, meets):
    baseline, _, _ = generations
    generation = replace(baseline[0], generated_token_ids=tokens)
    spec = TokenTailSpec("fixture/tokenizer", "a" * 40, (0, 1), (99,), max_tokens=128)
    observation = observe_token_tail(generation, spec)
    assert observation.status == "complete"
    assert (observation.start, observation.end) == (start, end)
    assert observation.run_length == end - start
    assert observation.meets_threshold is meets
    assert observation.terminal_eos_excluded is True
    assert observation.spec_sha256 == spec.sha256
    assert generation.quality is None
    assert observe_generation_health(generation, thinking_required=True).tail_degeneration_detected is None


@pytest.mark.parametrize("finish,tokens,start,end,meets", [("stop", (), 0, 0, False), ("stop", (7,) + (0,) * 15, 1, 16, False), ("stop", (7,) + (0,) * 16, 1, 17, True), ("stop", (0,) * 16 + (7,), 17, 17, False), ("stop", (0,) * 16 + (100,), 17, 17, False), ("length", (7,) + (0, 1) * 8, 1, 17, True), ("length", (7,) + (0, 1) * 8 + (99,), 18, 18, False)])
def test_token_tail_boundaries_do_not_override_native_finish(generations, finish, tokens, start, end, meets):
    baseline, _, _ = generations
    generation = replace(baseline[0], finish_reason=finish, generated_token_ids=tokens)
    spec = TokenTailSpec("fixture/tokenizer", "a" * 40, (0, 1), (99,), max_tokens=128)
    observation = observe_token_tail(generation, spec)
    assert (observation.start, observation.end, observation.meets_threshold) == (start, end, meets)
    assert observation.terminal_eos_excluded is False
    assert observe_generation_health(generation, thinking_required=True).cap_hit is (finish == "length")


@pytest.mark.parametrize("finish,reason", [("cancelled", "backend_failure"), ("error", "backend_failure"), ("stop", "token_budget_exceeded")])
def test_token_tail_unavailable_has_no_clean_or_threshold_claim(generations, finish, reason):
    baseline, _, _ = generations
    generation = replace(baseline[0], finish_reason=finish, generated_token_ids=(7,) + (0,) * 16)
    observation = observe_token_tail(generation, TokenTailSpec("fixture/tokenizer", "a" * 40, (0, 1), (99,), max_tokens=16))
    assert observation.status == "unavailable"
    assert observation.reason == reason
    assert observation.start is observation.end is observation.meets_threshold is observation.terminal_eos_excluded is None
    assert observation.run_length is None
    assert observation.token_count == 17


@pytest.mark.parametrize("change", [{"tokenizer_repo": ""}, {"tokenizer_revision": "main"}, {"tokenizer_revision": "A" * 40}, {"whitespace_token_ids": []}, {"whitespace_token_ids": ()}, {"whitespace_token_ids": (0, 0)}, {"whitespace_token_ids": (True,)}, {"eos_token_ids": (0,)}, {"min_run_tokens": 0}, {"min_run_tokens": True}, {"max_tokens": 15}, {"max_tokens": 32769}, {"algorithm": "legacy"}])
def test_token_tail_spec_is_frozen_strict_and_tokenizer_bound(change):
    spec = TokenTailSpec("fixture/tokenizer", "a" * 40, (0, 1), (99,))
    with pytest.raises(ValueError):
        replace(spec, **change)
    with pytest.raises(AttributeError):
        spec.whitespace_token_ids = (2,)
    assert replace(spec, tokenizer_revision="b" * 40).sha256 != spec.sha256
    assert replace(spec, whitespace_token_ids=(1, 0)).sha256 == spec.sha256


def test_token_tail_schema_refuses_impossible_and_mutable_claims():
    observation = TokenTailObservation("0" * 64, "1" * 64, 3, "complete", 1, 3, True, False)
    for change in ({"start": True}, {"end": 2}, {"start": 3}, {"terminal_eos_excluded": True}, {"status": "unavailable"}, {"reason": "unexpected"}, {"token_ids_sha256": "bad"}, {"meets_threshold": 1}):
        with pytest.raises(ValueError):
            replace(observation, **change)
    with pytest.raises(AttributeError):
        observation.start = 0


@pytest.mark.parametrize("finish", ["stop", "length", "error"])
def test_token_tail_health_profile_roundtrip_preserves_independent_observations(generations, source_profile, finish):
    baseline, _, _ = generations
    generation = replace(baseline[0], profile_id=source_profile.sha256, generated_token_ids=(7,) + (0, 1) * 8 + (2,), finish_reason=finish)
    spec = TokenTailSpec(source_profile.tokenizer_repo, source_profile.tokenizer_revision, (0, 1), source_profile.eos_token_ids, max_tokens=128)
    health = observe_generation_health(generation, thinking_required=True, repetition_spec=RepetitionSpec(max_tokens=128), tail_spec=spec, profile=source_profile)
    generation = replace(generation, health=health)
    restored = record_from_json(record_to_json(generation))
    assert restored == generation
    assert isinstance(restored.health.token_tail, TokenTailObservation)
    assert restored.health.token_tail.profile_sha256 == source_profile.sha256
    assert restored.health.token_tail.spec_sha256 == spec.sha256
    assert restored.health.cap_hit is (finish == "length")
    assert restored.health.tail_degeneration_detected is None
    assert restored.quality is None
    with pytest.raises(ValueError, match="exact generation/profile|exact generated tokens"):
        replace(generation, generated_token_ids=tuple(token + 100 for token in generation.generated_token_ids))


@pytest.mark.parametrize("kind", ["missing_profile", "record_profile", "tokenizer_repo", "tokenizer_revision", "eos", "thinking", "envelope"])
def test_token_tail_health_refuses_incompatible_profile_before_observing(generations, source_profile, monkeypatch, kind):
    baseline, _, _ = generations
    profile = replace(source_profile, track="validated_repair", max_new_tokens=4) if kind == "envelope" else source_profile
    generation = replace(baseline[0], profile_id=profile.sha256, generated_token_ids=(7,) + (0, 1) * 8 + (2,))
    spec = TokenTailSpec(profile.tokenizer_repo, profile.tokenizer_revision, (0, 1), profile.eos_token_ids, max_tokens=128)
    if kind == "record_profile":
        generation = replace(generation, profile_id="wrong-profile")
    elif kind == "tokenizer_repo":
        spec = replace(spec, tokenizer_repo="wrong/tokenizer")
    elif kind == "tokenizer_revision":
        spec = replace(spec, tokenizer_revision="a" * 40)
    elif kind == "eos":
        spec = replace(spec, eos_token_ids=(99,))

    def refuse_observer(*args, **kwargs):
        pytest.fail("Observer ran before incompatible profile rejection")

    monkeypatch.setattr("llm_loop_fix.detector.observe_token_tail", refuse_observer)
    monkeypatch.setattr("llm_loop_fix.detector.observe_token_repetition", refuse_observer)
    with pytest.raises(ValueError, match="profile|tokenizer|envelope"):
        observe_generation_health(generation, thinking_required=kind != "thinking", repetition_spec=RepetitionSpec(max_tokens=128), tail_spec=spec, profile=None if kind == "missing_profile" else profile)


@pytest.mark.parametrize("kind", ["spec", "mixed_side", "missing"])
def test_token_tail_health_pairing_rejects_different_contracts(generations, source_profile, kind):
    baseline, candidate, expected = generations
    baseline = tuple(replace(record, profile_id=source_profile.sha256) for record in baseline)
    candidate = tuple(replace(record, profile_id=source_profile.sha256) for record in candidate)
    spec = TokenTailSpec(source_profile.tokenizer_repo, source_profile.tokenizer_revision, (0, 1), source_profile.eos_token_ids, max_tokens=128)
    base_health = observe_generation_health(baseline[0], thinking_required=True, tail_spec=spec, profile=source_profile)
    baseline = tuple(replace(record, health=base_health) for record in baseline)
    alternate_health = observe_generation_health(candidate[0], thinking_required=True, tail_spec=replace(spec, min_run_tokens=15), profile=source_profile)
    if kind == "mixed_side":
        candidate = (replace(candidate[0], health=alternate_health), replace(candidate[1], health=base_health))
    elif kind == "missing":
        candidate = tuple(replace(record, health=None) for record in candidate)
    else:
        candidate = tuple(replace(record, health=alternate_health) for record in candidate)
    with pytest.raises(ValueError, match="health|mixes"):
        pair_generation_records(baseline, candidate, expected_keys=expected)


def test_token_tail_health_pairing_preserves_different_suffixes_without_quality(generations, source_profile):
    baseline, candidate, expected = generations
    baseline = tuple(replace(record, profile_id=source_profile.sha256, generated_token_ids=(7,) + (0,) * 16 + (2,)) for record in baseline)
    candidate = tuple(replace(record, profile_id=source_profile.sha256, generated_token_ids=(2,)) for record in candidate)
    spec = TokenTailSpec(source_profile.tokenizer_repo, source_profile.tokenizer_revision, (0, 1), source_profile.eos_token_ids, max_tokens=128)
    base_health = observe_generation_health(baseline[0], thinking_required=True, tail_spec=spec, profile=source_profile)
    candidate_health = observe_generation_health(candidate[0], thinking_required=True, tail_spec=spec, profile=source_profile)
    baseline = tuple(replace(record, health=base_health) for record in baseline)
    candidate = tuple(replace(record, health=candidate_health) for record in candidate)
    pairs = pair_generation_records(baseline, candidate, expected_keys=expected)
    assert pairs[0][0].health.token_tail.meets_threshold is True
    assert pairs[0][1].health.token_tail.meets_threshold is False
    assert pairs[0][1].quality is None
    assert pairs[0][1].health.tail_degeneration_detected is None


@pytest.mark.parametrize("kind", ["missing", "profile", "token", "degeneration", "finish"])
def test_token_tail_health_json_refuses_inconsistent_attachment(generations, source_profile, kind):
    baseline, _, _ = generations
    generation = replace(baseline[0], profile_id=source_profile.sha256, generated_token_ids=(7,) + (0,) * 16 + (2,))
    spec = TokenTailSpec(source_profile.tokenizer_repo, source_profile.tokenizer_revision, (0, 1), source_profile.eos_token_ids, max_tokens=128)
    generation = replace(generation, health=observe_generation_health(generation, thinking_required=True, tail_spec=spec, profile=source_profile))
    payload = json.loads(record_to_json(generation))
    if kind == "missing":
        del payload["health"]["token_tail"]
    elif kind == "profile":
        payload["health"]["token_tail"]["profile_sha256"] = "0" * 64
    elif kind == "token":
        payload["health"]["token_tail"]["token_ids_sha256"] = "0" * 64
    elif kind == "degeneration":
        payload["health"]["tail_degeneration_detected"] = True
    else:
        payload["finish_reason"] = "length"
        payload["health"]["cap_hit"] = True
    with pytest.raises(ValueError):
        record_from_json(json.dumps(payload))


def test_token_tail_health_budget_unavailable_and_legacy_defaults_roundtrip(generations, source_profile):
    baseline, _, _ = generations
    generation = replace(baseline[0], profile_id=source_profile.sha256, generated_token_ids=(7,) + (0,) * 16 + (2,))
    spec = TokenTailSpec(source_profile.tokenizer_repo, source_profile.tokenizer_revision, (0, 1), source_profile.eos_token_ids, max_tokens=16)
    generation = replace(generation, health=observe_generation_health(generation, thinking_required=True, tail_spec=spec, profile=source_profile))
    assert generation.health.token_tail.status == "unavailable"
    assert generation.health.token_tail.reason == "token_budget_exceeded"
    assert generation.health.tail_degeneration_detected is None
    assert record_from_json(record_to_json(generation)) == generation
    legacy = replace(baseline[0], health=observe_generation_health(baseline[0], thinking_required=True))
    payload = json.loads(record_to_json(legacy))
    del payload["health"]["token_tail"]
    assert record_from_json(json.dumps(payload)) == legacy


@pytest.mark.parametrize("finish,cap", [("stop", False), ("length", True), ("error", False)])
def test_health_cap_uses_backend_finish_not_sequence_length(generations, finish, cap):
    baseline, _, _ = generations
    generation = replace(baseline[0], finish_reason=finish, generated_token_ids=(1,))
    health = observe_generation_health(generation, thinking_required=True)
    assert health.cap_hit is cap
    assert health.repetition_detected is None
    assert health.tail_degeneration_detected is None


@pytest.mark.parametrize(
    "text,prompt_open,complete,malformed",
    [("reasoning</think>answer", True, True, False),
     ("reasoning</think>answer", False, False, True),
     ("reasoning</think>answer", None, None, None),
     ("<think>reasoning</think>answer", False, True, False),
     ("<think>reasoning</think>answer", True, False, True),
     ("reasoning without closure", True, False, False),
     ("<think>reasoning</think><think>continued", False, False, False)],
)
def test_thinking_health_requires_explicit_prompt_context(generations, text, prompt_open, complete, malformed):
    baseline, _, _ = generations
    generation = replace(baseline[0], generated_text=text, thinking_open_from_prompt=prompt_open)
    health = observe_generation_health(generation, thinking_required=True)
    assert health.thinking.closure_complete is complete
    assert health.thinking.malformed is malformed
    generation = replace(generation, health=health)
    assert record_from_json(record_to_json(generation)) == generation


def test_health_schema_rejects_contradictions_and_pairing_context_mismatch(generations):
    baseline, candidate, expected = generations
    with pytest.raises(ValueError):
        ThinkingObservation(True, None, False, True, True, False)
    health = GenerationHealth(True, ThinkingObservation(True, None, False, False, None, None))
    with pytest.raises(ValueError, match="finish/prompt"):
        replace(baseline[0], health=health)
    candidate = (replace(candidate[0], thinking_open_from_prompt=True), candidate[1])
    with pytest.raises(ValueError, match="mismatch"):
        pair_generation_records(baseline, candidate, expected_keys=expected)


def test_instruction_metadata_is_canonical_immutable_and_returns_fresh_kwargs():
    instruction = InstructionSpec("fixture:keywords", '{"nested":{"words":["alpha","beta"]},"count":2}')
    assert instruction == InstructionSpec("fixture:keywords", '{"count":2,"nested":{"words":["alpha","beta"]}}')
    kwargs = instruction.kwargs
    kwargs["nested"]["words"].append("mutated")
    assert instruction.kwargs["nested"]["words"] == ["alpha", "beta"]
    assert InstructionSpec("fixture:noargs", "null").kwargs == {}
    assert InstructionSpec("fixture:noargs", "null") != InstructionSpec("fixture:noargs", "{}")


def test_instruction_metadata_loader_and_record_json_preserve_order(tmp_path):
    path = tmp_path / "ifeval.jsonl"
    path.write_text(json.dumps({"id":"a", "prompt":"Write two words", "evaluator_id":"ifeval-strict",
                               "instructions":[{"instruction_id":"fixture:word-count", "kwargs":{"count":2}},
                                               {"instruction_id":"fixture:noargs", "kwargs":None}]})+"\n", encoding="utf-8")
    items = load_evaluation_items(path, source="fixture", source_revision="v1", role="benchmark")
    assert tuple(instruction.instruction_id for instruction in items[0].instructions) == (
        "fixture:word-count", "fixture:noargs"
    )
    text = record_to_json(items[0])
    assert json.loads(text)["instructions"][1]["kwargs"] is None
    assert record_from_json(text) == items[0]
    assert DatasetManifest(items).sha256 != DatasetManifest((replace(items[0], instructions=()),)).sha256


@pytest.mark.parametrize("kwargs", ["[]", "true", '{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}',
                                     '{"text":"'+("x"*65536)+'"}'])
def test_instruction_metadata_rejects_invalid_or_over_budget_arguments(kwargs):
    with pytest.raises(ValueError):
        InstructionSpec("fixture:instruction", kwargs)