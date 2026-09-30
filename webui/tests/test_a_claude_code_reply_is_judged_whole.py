"""A Claude Code run's solver reply is judged from the whole reply, inside its wrapper.

Measured 2026-09-29: Claude Code hands an MCP tool's text back as {"result": "<the reply>"}. The
interface judged the shortened copy, found no status in it (its quotes are escaped), and told a
run that had finished with status completed_unverified that it "computed nothing".
"""
import json
from pathlib import Path

from webui.outcome import classify_solver_result, verdict_reason

INNER = {"job_id": "30ac3f80", "solver": "fenics", "status": "completed_unverified",
         "trustworthy_result": False,
         "verification": "NOT VERIFIED -- the process exited cleanly but produced NO output files"}


def test_the_wrapped_reply_is_read_inside_its_wrapper():
    wrapped = json.dumps({"result": json.dumps(INNER)})
    assert classify_solver_result(wrapped) == "unverified"
    assert verdict_reason(wrapped).startswith("NOT VERIFIED")
    ok = json.dumps({"result": json.dumps({"status": "completed", "trustworthy_result": True})})
    assert classify_solver_result(ok) == "verified"


def test_a_shortened_copy_with_escaped_quotes_is_not_called_a_failure():
    wrapped = json.dumps({"result": json.dumps(INNER)})
    shortened = wrapped[:140] + " ...[4000 characters left out]... "
    assert classify_solver_result(shortened) == "unverified"


def test_the_adapter_stamps_the_verdict_from_the_whole_reply():
    src = (Path(__file__).resolve().parents[1] / "claude_code.py").read_text()
    assert 'event["verdict"] = classify_solver_result(str(body or ""))' in src
    assert '"result": shorten(body)' in src
