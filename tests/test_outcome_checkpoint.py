import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def checkpoint(monkeypatch, tmp_path):
    pytest.importorskip("fcntl")
    script = Path(__file__).resolve().parents[1] / ".github/hooks/outcome_checkpoint.py"
    spec = importlib.util.spec_from_file_location("outcome_checkpoint", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)

    def invoke(event, data, session=None):
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(data)))
        return module.main(event, session)

    return invoke


def test_harmless_window_and_resume(checkpoint):
    event = {"sessionId": "fixture-a", "cwd": "/repo", "timestamp": 1}
    start = checkpoint("sessionStart", {**event, "source": "startup", "initialPrompt": "not retained"})
    assert "reconstruct" in start["additionalContext"]
    tool = {**event, "toolName": "bash", "toolArgs": {}, "toolResult": {}}
    for _ in range(7):
        assert checkpoint("postToolUse", tool) == {}
    assert "Review:" in checkpoint("postToolUse", tool)["additionalContext"]
    resume = checkpoint("sessionStart", {**event, "source": "resume"})
    assert "not retained" not in resume["additionalContext"]
    assert "reconstruct" in checkpoint(
        "sessionStart", {**event, "sessionId": "fixture-b", "source": "new"}
    )["additionalContext"]


def test_route_change_does_not_reset_count_or_block_handoff(checkpoint):
    session = "fixture"
    item = {
        "outcome": "Offline reversible trial", "gap": "Missing production call",
        "work": "Static inventory", "evidence": "integration test", "downstream": "trial",
        "misses": 0,
    }
    review = {"evidence": "tests/test_pipeline.py", "why": "dependency still absent",
              "next": "Wire and run the actual trial", "advanced": False}
    for _ in range(2):
        checkpoint("begin", item, session)
        checkpoint("review", {**review, "route": "continue"}, session)
    resume = checkpoint("sessionStart", {"sessionId": session, "cwd": "/repo", "timestamp": 2, "source": "resume"})
    assert '"misses": 2' in resume["additionalContext"]
    with pytest.raises(ValueError, match="two non-advancing"):
        checkpoint("begin", {**item, "work": "Renamed inventory"}, session)
    assert "stop with the blocker" in checkpoint("review", {**review, "route": "blocked"}, session)["additionalContext"]
    checkpoint("review", {**review, "route": "redirect"}, session)
    checkpoint("begin", {**item, "work": review["next"]}, session)
    checkpoint("review", {**review, "route": "continue", "advanced": True}, session)
    assert '"misses": 0' in checkpoint("sessionStart", {"sessionId": session, "source": "resume"})["additionalContext"]


def test_corrupt_state_not_silently_reset(checkpoint, tmp_path):
    event = {"sessionId": "fixture", "source": "resume"}
    checkpoint("sessionStart", event)
    state = tmp_path / ".runs/outcome-checkpoint/fixture.json"
    state.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        checkpoint("sessionStart", event)
    assert state.read_text(encoding="utf-8") == "{"
