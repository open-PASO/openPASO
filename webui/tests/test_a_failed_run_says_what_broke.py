"""A failed solver run is explained by what broke, not by the sentence every failure shares.

Measured 2026-09-29 in the web interface: a run met nine failed run_simulation calls. Each of
those steps, and each turn's closing line, said "NOT VERIFIED -- the solver run errored, so no
number is backed by a valid run ...", because the reason was read from the verification
sentence first. None of them said the error ("NameError: name 'SplineGeometry' is not defined"), or the
fix openPASO had named beside it.
"""
import json

from webui.outcome import classify_solver_result, verdict_reason

NOT_VERIFIED = ("NOT VERIFIED -- the solver run errored, so no number is backed by a valid run. "
                "Per openPASO attestation this claim must NOT be reported as a result; revise "
                "the setup and re-run.")
TRACEBACK = ('Traceback (most recent call last):\n  File "solve.py", line 33, in <module>\n'
             "    geo = SplineGeometry()\n          ^^^^^^^^^^^^^^\n"
             "NameError: name 'SplineGeometry' is not defined\n")


def _reply(**extra) -> str:
    return json.dumps({"job_id": "x", "solver": "ngsolve", "status": "failed",
                       "error": TRACEBACK, "trustworthy_result": False,
                       "verification": NOT_VERIFIED, **extra})


def test_the_reason_is_the_line_that_says_what_went_wrong():
    raw = _reply()
    assert classify_solver_result(raw) == "failed"
    why = verdict_reason(raw)
    assert why.startswith("NameError: name 'SplineGeometry' is not defined"), why
    assert "NOT VERIFIED" not in why and "Traceback" not in why, why


def test_the_fix_openpaso_named_rides_with_it():
    why = verdict_reason(_reply(what_the_error_is="SplineGeometry lives in netgen.geom2d"))
    assert why == ("NameError: name 'SplineGeometry' is not defined -- "
                   "SplineGeometry lives in netgen.geom2d"), why


def test_an_error_that_names_no_exception_is_given_whole():
    raw = json.dumps({"status": "failed", "error": "the drag became non-finite at t = 0.31",
                      "verification": NOT_VERIFIED})
    assert verdict_reason(raw) == "the drag became non-finite at t = 0.31"


def test_a_run_that_ran_keeps_its_verification_sentence():
    """Only a failure is explained by its error: an unverified run says what was not verified."""
    raw = json.dumps({"status": "completed_unverified", "trustworthy_result": False,
                      "error": "a warning the solver printed",
                      "verification": "NOT VERIFIED -- the process produced NO output files"})
    assert classify_solver_result(raw) == "unverified"
    assert verdict_reason(raw).startswith("NOT VERIFIED -- the process produced NO output files")


def test_the_page_reads_it_the_same_way():
    """The page mirrors the rule for records the server did not stamp."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "ui" / "src" / "components"
           / "Transcript.tsx").read_text()
    assert "(err ? (troubleOf(err) || err) + (named ? ` -- ${named}` : '') : '') || note" in src
