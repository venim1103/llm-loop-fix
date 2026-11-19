"""Advisory Copilot checkpoint pilot. Never decides permissions or blocks a turn."""

import fcntl
import json
import os
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[2]
WINDOW = 8
FIELDS = ("outcome", "gap", "work", "evidence", "downstream")
ROUTES = {"continue", "redirect", "replan", "blocked"}
REVIEW = "Review: what gap changed (evidence pointer), why it matters, and the best executable next action."


def read_object(stream, limit):
    payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("checkpoint input exceeds size limit")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def texts(value, fields):
    return isinstance(value, dict) and all(
        isinstance(value.get(key), str)
        and len(value[key]) <= 280 and value[key].strip() and value[key].isprintable()
        for key in fields
    )


def load_state(path):
    if not path.exists():
        return {
            "item": None, "misses": None, "tools": 0, "started": False, "review": None,
        }, "No checkpoint: reconstruct the five fields and unresolved count from the live queue/session once."
    try:
        with path.open() as stream:
            state = read_object(stream, 8192)
        if (
            state["misses"] not in (None, 0, 1, 2)
            or type(state["tools"]) is not int
            or not 0 <= state["tools"] < WINDOW
            or type(state["started"]) is not bool
            or (state["item"] is not None and not texts(state["item"], FIELDS))
            or (state["review"] is not None and not texts(state["review"], ("route", "why", "next", "evidence")))
        ):
            raise ValueError("invalid state")
    except (ValueError, KeyError) as exc:
        raise ValueError("corrupt checkpoint; retain it and recover from the live queue under instruction-level controls") from exc
    return state, ""


def update(state, event, data):
    if event == "begin":
        if state["started"]:
            raise ValueError("finish the active item with review; a rename does not start a new one")
        if not texts(data, FIELDS):
            raise ValueError("begin requires five short fields")
        if state["misses"] is None:
            if type(data.get("misses")) is not int or data["misses"] not in (0, 1, 2):
                raise ValueError("restore the unresolved count (0, 1, or 2) from the live queue/session")
            state["misses"] = data["misses"]
            if state["misses"] == 2:
                return "Two unresolved items restored. Record a route-changing review before starting."
        if state["misses"] == 2:
            review = state["review"]
            if (
                state["item"] is not None or review is None
                or review["route"] not in {"redirect", "replan"} or data["work"] != review["next"]
            ):
                raise ValueError("two non-advancing items: begin only the changed work in a redirect or replan review")
        state["item"] = {key: data[key] for key in FIELDS}
        state["started"] = True
        state["tools"] = 0
        return REVIEW

    if event == "review":
        fields = ("evidence", "why", "next", "route")
        if not texts(data, fields) or data["route"] not in ROUTES:
            raise ValueError("review requires evidence pointer, why, next, and a valid route")
        advanced = data.get("advanced")
        if type(advanced) is not bool:
            raise ValueError("review requires advanced: true or false")
        misses = state["misses"]
        if state["started"]:
            misses = 0 if advanced else min(2, misses + 1)
        elif advanced or misses != 2 or data["route"] == "continue":
            raise ValueError("no open item or pending route change to review")
        state["review"] = {key: data[key] for key in fields}
        state["started"] = False
        state["misses"] = misses
        if misses == 2:
            if data["route"] in {"redirect", "replan"}:
                # Permit a changed route, not a reset of the unresolved count.
                state["item"] = None
            else:
                return "Do not start a third item: record a changed route or stop with the blocker."
        return "Review recorded. Pause, cleanup, and blocked handoff are always allowed."

    if event == "sessionStart":
        return f"Outcome checkpoint: {json.dumps(state, ensure_ascii=True)}. {REVIEW}"
    state["tools"] = (state["tools"] + 1) % WINDOW
    return REVIEW if state["tools"] == 0 else ""


def main(event, session_id=None):
    manual = event in {"begin", "review"}
    if not manual and event not in {"sessionStart", "postToolUse"}:
        raise ValueError("unsupported event")
    data = read_object(sys.stdin, 8192 if manual else 1024 * 1024)
    if not manual:
        if data.get("agentId"):
            return {}
        session_id = data.get("sessionId")
        if event == "postToolUse" and not isinstance(data.get("toolName"), str):
            raise ValueError("expected camelCase postToolUse event")
    if not isinstance(session_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", session_id):
        raise ValueError("invalid session ID")
    store = ROOT / ".runs" / "outcome-checkpoint"
    store.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = store / f"{session_id}.json"
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state, notice = load_state(path)
        message = update(state, event, data)
        with NamedTemporaryFile(mode="w", dir=store, delete=False) as stream:
            temp = Path(stream.name)
            try:
                json.dump(state, stream, ensure_ascii=False)
                stream.close()
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)
    context = " ".join(filter(None, (notice, message)))
    return {"additionalContext": f"Checkpoint session {session_id}. {context}"} if context else {}


if __name__ == "__main__":
    try:
        print(json.dumps(main(*sys.argv[1:])))
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"Outcome checkpoint degraded: {exc}", file=sys.stderr)
        sys.exit(2)
