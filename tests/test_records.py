import json
from dataclasses import FrozenInstanceError, replace

import pytest

from llm_loop_fix.records import (
    ChatMessage,
    DatasetManifest,
    EvaluationItem,
    GenerationRecord,
    SampleKey,
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