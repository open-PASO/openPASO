"""Smoke tests for the WebUI.

Run from repo root: .venv-lg/bin/pytest webui/tests -v

These tests use FastAPI's TestClient (HTTP) and websockets for the
streamed channel. No vLLM, no GPU; the runner uses the mock LLM so
the end-to-end spawn_subagent chain is exercised in <1 s.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pytest
from fastapi.testclient import TestClient

from webui import app as webui_app
from webui import catalog, config, files, runs, sessions


@pytest.fixture(autouse=True)
def allow_the_test_model(monkeypatch):
    """The fake model is refused unless the server was started for testing
    (OPENPASO_TEST_MODEL). These tests are that case, and say so."""
    monkeypatch.setattr(config, "ALLOW_TEST_MODEL", True)


@pytest.fixture
def client():
    return TestClient(webui_app.app)


# ───────────────────────────────────────────────────────────────────
# Config endpoints
# ───────────────────────────────────────────────────────────────────
def _all_models(client):
    return [m for g in client.get("/api/models").json()["groups"] for m in g["models"]]


def test_models_say_where_they_run_and_whether_they_work(client):
    """The picker used to be one flat list: local models with no server behind
    them looked exactly like hosted ones that worked."""
    r = client.get("/api/models").json()
    kinds = {g["kind"] for g in r["groups"]}
    assert {"openrouter", "local"} <= kinds
    for g in r["groups"]:
        assert g["title"] and g["note"], "every group must say where data goes"
        for m in g["models"]:
            assert isinstance(m["available"], bool) and m["status"]
    assert any(m["id"].startswith("qwen2.5-") for m in _all_models(client))


def test_a_fresh_install_has_no_default_model_rather_than_a_server_it_lacks(monkeypatch):
    from webui import claude_code
    monkeypatch.setattr(claude_code, "available", lambda: False)
    monkeypatch.setattr(config, "openrouter_key", lambda: None)
    assert config.default_model() is None


def test_the_fake_model_needs_the_server_to_be_started_for_testing(client, monkeypatch):
    """Asking for it over the API is not enough: this API has no authentication,
    and the fake model fabricates answers."""
    monkeypatch.setattr(config, "ALLOW_TEST_MODEL", False)
    r = client.post("/api/sessions", json={"model": "mock", "test": True})
    assert r.status_code == 400 and "mock" in r.json()["detail"]


def test_an_approval_that_arrives_first_is_not_lost():
    """The step is announced before the gate opens, so a fast client can answer
    before there is anything to answer, and the run waited for ever."""
    import asyncio as aio
    from webui.runner import ApprovalGate

    async def main():
        gate = ApprovalGate()
        gate.resolve("tc_1", True)            # the answer, before the question
        decision = await aio.wait_for(gate.open("tc_1"), timeout=1)
        assert decision["approved"] is True

    aio.run(main())


def test_switching_to_run_without_asking_releases_a_waiting_step():
    import asyncio as aio
    from webui.runner import ApprovalGate

    async def main():
        gate = ApprovalGate()
        fut = gate.open("tc_9")
        assert gate.open_all(True) == 1
        assert (await aio.wait_for(fut, timeout=1))["approved"] is True

    aio.run(main())


def test_the_fake_model_is_never_offered_to_a_person(client):
    """The mock answers one canned turn and runs no solver. It was once the
    default, and a fabricated run looked exactly like real work."""
    assert "mock" not in [m["id"] for m in _all_models(client)]
    r = client.post("/api/sessions", json={"model": "mock", "mode": "accept"})
    assert r.status_code == 400, "a person must not be able to create a fake-model run"


def test_the_solver_count_is_measured_not_asserted(client):
    """The first screen used to state nine solvers from a hardcoded array.

    It must come from the registry, and when the check cannot run it must say
    so rather than guess."""
    r = client.get("/api/solvers").json()
    assert "ok" in r and "solvers" in r
    if r["ok"]:
        assert all("status" in s and "name" in s for s in r["solvers"])


def test_mcp_servers(client):
    r = client.get("/api/mcp_servers").json()
    assert any(s["id"] == "openpaso" for s in r["servers"])


def test_modes_are_only_the_ones_that_do_something(client):
    """"autonomous" was offered and read by nothing: it behaved exactly like accept."""
    r = client.get("/api/config").json()
    assert [m["id"] for m in r["modes"]] == ["plan", "accept"]
    assert all(m["label"] and m["detail"] for m in r["modes"])


def test_claude_code_cannot_be_started_in_ask_before_each_step(client):
    """It runs headless and cannot stop to ask; it used to accept the mode and
    silently ignore it."""
    r = client.post("/api/sessions", json={"model": "claude-code", "mode": "plan"})
    assert r.status_code == 400


def test_files_are_served_from_run_folders_only(client):
    for rel in ("../../etc/passwd", "campaign/cell/out.txt", "webui_abc123/../../x", ""):
        for url in ("/api/file", "/api/viz"):
            assert client.get(url, params={"rel": rel}).status_code == 403, (url, rel)
    assert client.get("/sandbox-file/campaign/cell/out.txt").status_code == 403
    # the old sandbox-wide listing and the parameter extractor are gone
    assert client.get("/api/files").status_code in (404, 405)
    assert client.post("/api/extract_params", json={"source": "N = 3"}).status_code in (404, 405)


def test_stop_pressed_straight_after_send_still_stops_the_run():
    """The flag that says a turn is over was cleared inside the turn's own task,
    so a Stop in the same breath as Send was told nothing was running."""
    import asyncio as aio
    from webui import runs as runs_mod
    run = runs_mod.Run.__new__(runs_mod.Run)
    run.state = {"events": [], "mode": "accept"}
    run._turn_ended = True
    run.turn_task = None            # `running` is read from this
    run.steers = []
    run.push_snapshot = lambda: aio.sleep(0)
    run.emit = lambda e: aio.sleep(0)

    async def never_ending(*a, **k):
        await aio.sleep(30)

    run._turn = never_ending

    async def main():
        await runs_mod.Run.prompt(run, "do something")
        # what stop() looks at, before the turn task has had any time to run
        assert run._turn_ended is False
        assert run.turn_task is not None and not run.turn_task.done()
        run.turn_task.cancel()

    aio.run(main())


def test_an_xdmf_file_is_not_read_as_hdf5(tmp_path):
    """h5py cannot read XDMF, and the failure was swallowed: the file was
    labelled HDF5 with no contents, and the .h5 holding the numbers never shown."""
    from webui import viz
    p = config.SANDBOX_ROOT / "webui_abc123def" / "out.xdmf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('<Xdmf><Domain><Grid><Geometry><DataItem Format="HDF">mesh.h5:/points'
                 '</DataItem></Geometry><Attribute><DataItem Format="HDF">field.h5:/u'
                 '</DataItem></Attribute></Grid></Domain></Xdmf>')
    try:
        out = viz._hdf(p)
        assert out["kind"] == "xdmf"
        assert out["data_files"] == ["mesh.h5", "field.h5"]
        assert "<Xdmf>" in out["text"]
    finally:
        p.unlink()


def test_a_guess_at_a_path_is_not_exported_as_a_setting(monkeypatch, tmp_path):
    """An exported FOURC_BINARY is read by the backend as an explicit override
    and stops its own search, so guessing a path that does not exist told a
    person with 4C installed elsewhere that they do not have 4C."""
    monkeypatch.delenv("FOURC_BINARY", raising=False)
    monkeypatch.delenv("FOURC_ROOT", raising=False)
    monkeypatch.setattr(config.Path, "home", staticmethod(lambda: tmp_path))
    assert config._fourc_env() == {}, "nothing is claimed about a 4C that is not there"
    (tmp_path / "4C" / "build").mkdir(parents=True)
    (tmp_path / "4C" / "build" / "4C").write_text("#!/bin/sh\n")
    found = config._fourc_env()
    assert found["FOURC_BINARY"].endswith("4C/build/4C")
    monkeypatch.setenv("FOURC_BINARY", "/opt/4C/bin/4C")
    assert config._fourc_env()["FOURC_BINARY"] == "/opt/4C/bin/4C", "what is set by hand is kept"


def test_a_path_in_a_key_is_scrubbed_like_one_in_a_value():
    from webui.privacy import scrub
    out = scrub({f"{Path.home()}/private/run.json": {"note": f"{Path.home()}/x"}})
    text = json.dumps(out)
    assert str(Path.home()) not in text and "~/private/run.json" in text


def test_a_run_cannot_be_started_on_a_model_that_cannot_run(client, monkeypatch):
    async def nothing_available():
        return {"groups": [{"kind": "openrouter", "title": "t", "note": "", "models": [
            {"id": "deepseek/deepseek-v4.1-flash", "label": "DeepSeek v4.1 Flash",
             "kind": "openrouter", "available": False, "status": "no OpenRouter key"}]}],
            "default": None}
    monkeypatch.setattr(catalog, "models", nothing_available)
    r = client.post("/api/sessions", json={"model": "deepseek/deepseek-v4.1-flash", "mode": "accept"})
    assert r.status_code == 409 and "no OpenRouter key" in r.json()["detail"]
    assert "not sent" in r.json()["detail"]


def test_a_long_result_keeps_the_verdict_at_its_end():
    """openPASO stamps its verification verdict at the END of a report. Cutting
    a long result to its first 8000 characters threw that away, and a verified
    run then showed as unverified."""
    from webui.outcome import RESULT_LIMIT, classify_solver_result, shorten
    body = json.dumps({"status": "completed", "log": "x" * 40000, "trustworthy_result": True})
    raw = "[{'type': 'text', 'text': '" + body + "'}]"
    assert len(raw) > RESULT_LIMIT
    assert classify_solver_result(shorten(raw)) == "verified"
    assert len(shorten(raw)) <= RESULT_LIMIT + 200


def test_claude_code_reporting_an_error_is_not_a_turn_that_merely_had_no_result():
    """Its result message can say the run failed while the command exits zero.
    Taking any text as a finished turn recorded a failure as an empty success."""
    import asyncio as aio
    from webui import claude_code

    class _Proc:
        returncode = 0

        class _Out:
            def __init__(self):
                self.lines = [
                    json.dumps({"type": "result", "subtype": "error_during_execution",
                                "is_error": True, "result": "tool use failed"}).encode() + b"\n",
                    b"",
                ]

            async def readline(self):
                return self.lines.pop(0) if self.lines else b""

        def __init__(self):
            self.stdout = self._Out()
            self.stderr = None

        async def wait(self):
            return 0

    async def emit(_e):
        return None

    try:
        aio.run(claude_code._consume(_Proc(), emit, None, []))
        raise AssertionError("a reported error must not pass as a finished turn")
    except RuntimeError as exc:
        assert "stopped with an error" in str(exc) and "tool use failed" in str(exc)


def test_claude_code_writes_its_results_into_the_run(tmp_path):
    """Its openPASO server took the shared install directories, so a Claude Code
    run's output landed outside the run and two runs could collide."""
    from webui import claude_code
    cfg = claude_code._mcp_config(["openpaso"], tmp_path)
    env = cfg["mcpServers"]["openpaso"]["env"]
    for key in ("OPENPASO_CELL_WORKDIR", "OPENPASO_OUTPUT_DIR", "OPENPASO_COUPLING_DIR",
                "OPENPASO_MESH_DIR", "OPENPASO_BENCHMARK_DIR"):
        assert env[key].startswith(str(tmp_path.resolve())), (key, env[key])


def test_every_text_file_is_scrubbed_however_it_is_named(client):
    """Deciding by suffix let a solver's own formats through unscrubbed."""
    work = config.SANDBOX_ROOT / "webui_abc123def" / "work"
    work.mkdir(parents=True, exist_ok=True)
    deck = work / "case.4c"
    deck.write_text(f"MESHFILE: {Path.home()}/meshes/part.msh\n")
    binary = work / "field.bin"
    binary.write_bytes(b"\x00\x01\x02" + str(Path.home()).encode())
    try:
        text = client.get("/sandbox-file/webui_abc123def/work/case.4c")
        assert str(Path.home()) not in text.text and "~/meshes/part.msh" in text.text
        # a binary file is served as it is; scrubbing it would corrupt it
        assert client.get("/sandbox-file/webui_abc123def/work/field.bin").status_code == 200
    finally:
        deck.unlink(); binary.unlink()


def test_a_download_is_scrubbed_like_everything_else(client, tmp_path, monkeypatch):
    """The download route hands over whole files; it used to be the one way a
    home path could reach the browser."""
    run = config.SANDBOX_ROOT / "webui_abc123def"
    (run / "work").mkdir(parents=True, exist_ok=True)
    log = run / "work" / "solver.log"
    log.write_text(f"reading mesh from {Path.home()}/runs/mesh.msh\n")
    try:
        r = client.get("/sandbox-file/webui_abc123def/work/solver.log")
        assert r.status_code == 200
        assert str(Path.home()) not in r.text and "~/runs/mesh.msh" in r.text
    finally:
        log.unlink()


def test_stop_leaves_a_process_that_was_already_in_the_folder(tmp_path):
    """A terminal someone opened in the run folder is not the run's work."""
    import subprocess, time
    from webui import proctree
    proc = subprocess.Popen(["bash", "-c", "sleep 60 & wait"], cwd=tmp_path, start_new_session=True)
    try:
        time.sleep(0.5)
        started_later = time.time() + 5
        assert proc.pid in proctree.run_processes(tmp_path)
        assert proc.pid not in proctree.run_processes(tmp_path, since=started_later)
        assert proctree.run_processes(tmp_path, since=time.time() - 60)
    finally:
        proc.kill(); proc.wait(timeout=10)


def test_a_table_preview_reads_only_what_it_shows(tmp_path):
    """A solver log can be hundreds of megabytes; the preview used to read all
    of it to show the first 999 rows."""
    import time
    from webui import viz
    big = config.SANDBOX_ROOT / "webui_abc123def" / "big.csv"
    big.parent.mkdir(parents=True, exist_ok=True)
    with big.open("w") as f:
        f.write("x,y\n")
        for i in range(400_000):          # about 5 MB
            f.write(f"{i},{i * 2}\n")
    try:
        t0 = time.monotonic()
        out = viz._csv(big)
        took = time.monotonic() - t0
        assert out["kind"] == "table" and out["truncated"] is True
        assert len(out["rows"]) == 999, len(out["rows"])
        assert took < 0.25, f"read the whole file: {took:.2f}s"
    finally:
        big.unlink()


def test_a_tab_separated_table_is_read_with_tabs(tmp_path):
    from webui import viz
    p = config.SANDBOX_ROOT / "webui_abc123def" / "t.tsv"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x\ty\n1\t2\n")
    try:
        out = viz._csv(p)
        assert out["header"] == ["x", "y"] and out["rows"] == [["1", "2"]]
    finally:
        p.unlink()


def test_a_link_out_of_a_run_does_not_open_another_run(client):
    """A run can write a symlink. The name then reads as a path inside that run
    while pointing into a different one."""
    a = config.SANDBOX_ROOT / "webui_aaa111" / "work"
    b = config.SANDBOX_ROOT / "webui_bbb222" / "work"
    a.mkdir(parents=True, exist_ok=True)
    b.mkdir(parents=True, exist_ok=True)
    (b / "secret.txt").write_text("another run's numbers\n")
    link = a / "elsewhere"
    link.unlink(missing_ok=True)
    link.symlink_to(b, target_is_directory=True)
    try:
        for url in ("/api/file", "/api/viz"):
            r = client.get(url, params={"rel": "webui_aaa111/work/elsewhere/secret.txt"})
            assert r.status_code == 403, (url, r.status_code)
        assert client.get("/sandbox-file/webui_aaa111/work/elsewhere/secret.txt").status_code == 403
    finally:
        link.unlink(missing_ok=True)
        (b / "secret.txt").unlink(missing_ok=True)


def test_a_listing_cannot_step_into_a_sibling_folder(client):
    """"webui_x/work2" starts with "webui_x/work": a prefix is not a boundary."""
    sid = client.post("/api/sessions", json={"model": "mock", "test": True}).json()["id"]
    try:
        run = runs.get(sid)
        (run.workdir.parent / "work2").mkdir(parents=True, exist_ok=True)
        (run.workdir.parent / "work2" / "not_yours.txt").write_text("x")
        r = client.get(f"/api/sessions/{sid}/files", params={"sub": "../work2"})
        assert r.status_code in (403, 404), r.status_code
        if r.status_code == 200:                      # never, but be explicit
            assert "not_yours.txt" not in r.text
    finally:
        runs.RUNS.pop(sid, None)
        client.delete(f"/api/sessions/{sid}")


def test_private_paths_never_reach_the_browser():
    from webui.privacy import scrub
    home = str(Path.home())
    user = Path.home().name
    removable = f"/media/{user}/disk"          # a mounted drive of the same person
    out = scrub({"a": f"{home}/x/run.py", "b": [f"/home/{user}/y"], "c": f"{removable}/z"})
    text = json.dumps(out)
    assert home not in text and removable not in text
    assert "~/x/run.py" in text


def test_classify():
    p = config.SANDBOX_ROOT / "x" / "out.vtu"
    assert files.classify(p) == "vtk"
    p2 = config.SANDBOX_ROOT / "x" / "data.csv"
    assert files.classify(p2) == "table"


# ───────────────────────────────────────────────────────────────────
# Sessions round-trip
# ───────────────────────────────────────────────────────────────────
def test_session_lifecycle(client):
    n = client.post("/api/sessions", json={"model": "mock", "test": True}).json()
    sid = n["id"]
    got = client.get(f"/api/sessions/{sid}").json()
    assert got["id"] == sid and got["model"] == "mock"
    # a run nobody prompted, and a fake-model run, are not listed as work
    listed = client.get("/api/sessions").json()["sessions"]
    assert not any(s["id"] == sid for s in listed)
    assert any(s["id"] == sid for s in client.get("/api/sessions?all=true").json()["sessions"])
    assert client.delete(f"/api/sessions/{sid}").json()["deleted"]


def test_the_first_prompt_reaches_the_server_without_the_tab(client, monkeypatch):
    """It used to be held in the browser until the socket said hello: closing
    the tab in that moment lost the message and left a run nobody could see."""
    sent = {}

    async def record(self, text, attachments=None):
        sent["run"], sent["text"], sent["files"] = self.sid, text, attachments

    monkeypatch.setattr(runs.Run, "prompt", record)
    sid = client.post("/api/sessions", json={"model": "mock", "test": True}).json()["id"]
    try:
        assert client.post(f"/api/sessions/{sid}/prompt", json={"text": " "}).status_code == 400
        assert client.post("/api/sessions/nosuchrun00/prompt", json={"text": "hi"}).status_code == 404
        r = client.post(f"/api/sessions/{sid}/prompt",
                        json={"text": "hello there", "attachments": ["part.msh"]})
        assert r.status_code == 200 and r.json()["sent"]
        assert sent == {"run": sid, "text": "hello there", "files": ["part.msh"]}
    finally:
        runs.RUNS.pop(sid, None)
        client.delete(f"/api/sessions/{sid}")


def test_the_prompt_is_on_disk_as_soon_as_it_is_sent(client):
    """Only every tenth event was written, so a restart during the first turn
    left a record with no prompt — and a run with no prompt is hidden."""
    import asyncio as aio
    sid = client.post("/api/sessions", json={"model": "mock", "test": True}).json()["id"]
    try:
        run = runs.get(sid)
        aio.run(run.emit({"type": "turn_start"}))
        aio.run(run.emit({"type": "user_msg", "text": "solve the plate"}))
        saved = json.loads((config.SESSION_DIR / f"{sid}.json").read_text())
        assert [e["type"] for e in saved["events"]] == ["turn_start", "user_msg"]
        assert saved["events"][1]["text"] == "solve the plate"
    finally:
        runs.RUNS.pop(sid, None)
        client.delete(f"/api/sessions/{sid}")


def test_deleting_refuses_a_made_up_id_before_touching_anything(client):
    for bad in ("../webui_other", "..%2Fx", "not-hex-id", ""):
        assert client.delete(f"/api/sessions/{bad}").status_code in (400, 404, 405)


def test_ending_one_step_leaves_a_step_that_started_earlier(tmp_path):
    """Both carry the run's marker; only the one being ended may be ended."""
    import subprocess, time
    from webui import proctree
    env = {**os.environ, "OPENPASO_CELL_WORKDIR": str(tmp_path.resolve())}
    earlier = subprocess.Popen(["bash", "-c", "sleep 60 & wait"], cwd="/tmp", env=env, start_new_session=True)
    time.sleep(0.6)
    step_started = time.time()
    time.sleep(0.6)
    later = subprocess.Popen(["bash", "-c", "sleep 60 & wait"], cwd="/tmp", env=env, start_new_session=True)
    try:
        time.sleep(0.6)
        claimed = proctree.run_processes(tmp_path, step_started, since_covers_marker=True)
        assert later.pid in claimed, claimed
        assert earlier.pid not in claimed, claimed
    finally:
        for p in (earlier, later):
            p.kill(); p.wait(timeout=10)


def test_two_prompts_at_once_cannot_both_pass_the_run_limit(monkeypatch):
    """Counting and starting were separate, so two prompts arriving together
    both saw room and both started."""
    import asyncio as aio
    from webui import runs as runs_mod

    started = []

    class FakeRun:
        running = False
        async def prompt(self, text, attachments=None):
            started.append(text)
            await aio.sleep(0)

    monkeypatch.setattr(config, "MAX_RUNNING", 2)
    monkeypatch.setattr(runs_mod, "running_count", lambda: len(started))

    async def main():
        runs = [FakeRun() for _ in range(5)]
        answers = await aio.gather(*[runs_mod.start_turn(r, f"go {i}") for i, r in enumerate(runs)])
        return answers

    answers = aio.run(main())
    assert len(started) == 2, started
    refused = [a for a in answers if a]
    assert len(refused) == 3 and "not sent" in refused[0]


def test_scrubbing_leaves_a_field_file_s_numbers_alone():
    """Frames are megabytes of base64, in which a path can appear by chance.
    Replacing it changed the numbers a run had computed."""
    import base64, os
    from webui.privacy import scrub_text
    frames = base64.b64encode(os.urandom(300_000)).decode()
    doc = json.dumps({"kind": "field_series", "frames": frames,
                      "provenance": {"source": f"{Path.home()}/run/out.vtu"}})
    out = scrub_text(doc)
    assert json.loads(out)["frames"] == frames, "the data must come back exactly"
    assert str(Path.home()) not in out and "~/run/out.vtu" in out
    # and a path written where a path really occurs is still found
    assert "~" in scrub_text(f"--prefix={Path.home()}/opt")


def test_an_svg_a_run_wrote_cannot_run_as_code(client):
    """A model can write an SVG with a script in it. Drawn in a page it never
    runs; opened at its own address it would run in this interface's origin."""
    work = config.SANDBOX_ROOT / "webui_abc123def" / "work"
    work.mkdir(parents=True, exist_ok=True)
    art = work / "plot.svg"
    art.write_text('<svg xmlns="http://www.w3.org/2000/svg"><script>fetch("/api/sessions")</script></svg>')
    try:
        r = client.get("/sandbox-file/webui_abc123def/work/plot.svg")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/svg+xml")
        assert "default-src 'none'" in r.headers.get("content-security-policy", "")
        assert r.headers.get("x-content-type-options") == "nosniff"
    finally:
        art.unlink()


def test_a_pdf_is_not_mistaken_for_text_and_rewritten(client):
    """A PDF opens with an ASCII header and turns binary later, so a sniff of
    the first kilobytes called it text and it was changed on the way out."""
    work = config.SANDBOX_ROOT / "webui_abc123def" / "work"
    work.mkdir(parents=True, exist_ok=True)
    doc = work / "report.pdf"
    body = b"%PDF-1.7\n" + b"a" * 9000 + b"\x00\x01\x02binary tail" + str(Path.home()).encode()
    doc.write_bytes(body)
    try:
        r = client.get("/sandbox-file/webui_abc123def/work/report.pdf")
        assert r.status_code == 200
        assert r.content == body, "served exactly as the run wrote it"
    finally:
        doc.unlink()


def test_a_file_calling_itself_a_field_must_carry_a_field(tmp_path):
    """Adopted on its word, the page asked for frames that were not there."""
    from webui import viz
    p = config.SANDBOX_ROOT / "webui_abc123def" / "claims.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"kind": "field_series", "note": "written by hand"}))
    try:
        assert viz._json(p)["kind"] != "field_series"
    finally:
        p.unlink()


def test_a_data_field_holding_prose_is_still_scrubbed_when_streamed(client):
    """The key is not enough: a "data" field can hold text, and passing it
    through unread because of its name would carry a home path out."""
    work = config.SANDBOX_ROOT / "webui_abc123def" / "work"
    work.mkdir(parents=True, exist_ok=True)
    doc = work / "notes.json"
    filler = "the run wrote a long note here. " * 400_000          # over the stream limit
    doc.write_text(json.dumps({"data": f"{filler} written in {Path.home()}/run/x", "n": 1}))
    try:
        r = client.get("/sandbox-file/webui_abc123def/work/notes.json")
        assert r.status_code == 200 and len(r.text) > 8 * 1024 * 1024
        assert str(Path.home()) not in r.text, "prose under any key is scrubbed"
        assert "~/run/x" in r.text
    finally:
        doc.unlink()


def test_a_streamed_field_file_keeps_its_frames_whole(client):
    """Above the whole-file limit the download is streamed, and a chunk
    boundary inside the base64 used to put the payload back in the scrubber's
    way — the same corruption, one path further on."""
    import base64
    work = config.SANDBOX_ROOT / "webui_abc123def" / "work"
    work.mkdir(parents=True, exist_ok=True)
    frames = base64.b64encode(os.urandom(9_000_000)).decode()      # over the 8 MB limit
    big = work / "field.json"
    big.write_text(json.dumps({"kind": "field_series", "note": f"written in {Path.home()}/run",
                               "frames": frames}))
    try:
        r = client.get("/sandbox-file/webui_abc123def/work/field.json")
        assert r.status_code == 200
        assert len(r.text) > 8 * 1024 * 1024, "this is the streamed path"
        back = json.loads(r.text)
        assert back["frames"] == frames, "the frames must survive byte for byte"
        assert str(Path.home()) not in r.text and "~/run" in back["note"]
    finally:
        big.unlink()


def test_encoded_data_survives_every_substitution_not_only_the_newest():
    """A path pattern keeps out of the middle of base64 by requiring a boundary,
    but "=" must stay a boundary so that --prefix=/home/... is caught — and "="
    is base64's padding, so concatenated frames contained an equals sign
    followed by a home path and were rewritten. The split now covers every
    substitution, not only the one fixed last."""
    import base64
    from webui.privacy import scrub_text
    planted = "/home/" + "someone/private"      # built, so this file holds no path
    frames = (base64.b64encode(os.urandom(3000)).decode() + "="
              + planted + base64.b64encode(os.urandom(3000)).decode())
    doc = json.dumps({"kind": "field_series", "frames": frames,
                      "note": f"wrote {Path.home()}/run/out.vtu"})
    out = scrub_text(doc)
    assert json.loads(out)["frames"] == frames, "the frames must come back byte for byte"
    assert str(Path.home()) not in out and "~/run/out.vtu" in out


def test_the_encoded_guard_cannot_hide_a_real_path():
    """Protecting "any long run of base64 characters" also protects a long
    enough home path, and the guard would then hide exactly what the scrubbing
    exists to remove. Only the payloads that carry encoded data are protected."""
    from webui.privacy import scrub_text
    long_path = f"{Path.home()}/" + "a" * 60
    out = scrub_text(f"the run wrote to {long_path} and stopped")
    assert str(Path.home()) not in out, out[:120]


def test_a_name_inside_encoded_data_is_left_alone():
    """The frames of a field file are base64, and a user name occurs in one by
    chance: "...+alexander/..." has exactly the characters the name pattern
    treats as a boundary. Rewriting it decodes to a field nobody computed."""
    import base64
    import getpass
    from webui.privacy import scrub_text
    user = getpass.getuser()
    if len(user) < 3:
        return
    frames = base64.b64encode(os.urandom(4000)).decode()
    planted = frames[:200] + "+" + user + "/" + frames[200:]
    doc = json.dumps({"kind": "field_series", "frames": planted,
                      "provenance": {"source": f"/home/{user}/run/out.vtu"}})
    out = scrub_text(doc)
    assert json.loads(out)["frames"] == planted, "encoded data must come back unchanged"
    assert f"/home/{user}" not in out and "~/run/out.vtu" in out
    # and in prose the name is still removed
    assert user not in scrub_text(f"the run was started by {user} at noon")


def test_the_machine_name_is_not_handed_to_the_browser(tmp_path):
    from webui import viz
    p = config.SANDBOX_ROOT / "webui_abc123def" / "f.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}")
    try:
        out = viz._field_series(p, {"field": "u", "provenance": {
            "host": "hereon-workstation-3", "solver": "FEniCSx", "true_max": 1.5}})
        assert out["provenance"] == {"solver": "FEniCSx", "true_max": 1.5}
    finally:
        p.unlink()


def test_a_run_over_the_limit_is_refused_before_it_is_created(client, monkeypatch):
    from webui import runs
    monkeypatch.setattr(runs, "running_count", lambda: config.MAX_RUNNING)
    before = len(client.get("/api/sessions?all=true").json()["sessions"])
    r = client.post("/api/sessions", json={"model": "mock", "test": True})
    assert r.status_code == 409 and "not sent" in r.json()["detail"]
    assert len(client.get("/api/sessions?all=true").json()["sessions"]) == before


def test_two_uploads_that_sanitise_alike_are_both_kept(client):
    """"a b.msh" and "a_b.msh" become the same name; the second used to replace
    the first while the answer said both were saved."""
    sid = client.post("/api/sessions", json={"model": "mock", "test": True}).json()["id"]
    try:
        first = client.post(f"/api/sessions/{sid}/upload",
                            files={"files": ("a b.msh", b"$MeshFormat\nfirst\n", "application/octet-stream")})
        second = client.post(f"/api/sessions/{sid}/upload",
                             files={"files": ("a_b.msh", b"$MeshFormat\nsecond\n", "application/octet-stream")})
        assert first.status_code == 200 and second.status_code == 200
        names = sorted(e["name"] for e in
                       client.get(f"/api/sessions/{sid}/files", params={"sub": "uploads"}).json()["entries"])
        assert names == ["a_b-2.msh", "a_b.msh"], names
        assert second.json()["saved"][0]["name"] == "a_b-2.msh"
    finally:
        runs.RUNS.pop(sid, None)
        client.delete(f"/api/sessions/{sid}")


def test_a_record_downloaded_during_a_run_holds_what_has_happened(client):
    """On disk it is a checkpoint; the manifest promised every event."""
    sid = client.post("/api/sessions", json={"model": "mock", "test": True}).json()["id"]
    try:
        import asyncio as aio
        run = runs.get(sid)
        aio.run(run.emit({"type": "turn_start"}))
        aio.run(run.emit({"type": "user_msg", "text": "solve it"}))
        for i in range(4):                      # below the checkpoint interval
            aio.run(run.emit({"type": "agent_msg", "text": f"thinking {i}"}))
        got = client.get(f"/api/sessions/{sid}/manifest").json()
        texts = [e.get("text") for e in got["events"] if e["type"] == "agent_msg"]
        assert texts == ["thinking 0", "thinking 1", "thinking 2", "thinking 3"], texts
    finally:
        runs.RUNS.pop(sid, None)
        client.delete(f"/api/sessions/{sid}")


def test_a_seeded_conversation_does_not_open_with_an_empty_message():
    """The turn has already emitted turn_start and user_msg when the history is
    built, and dropping only the last event left a turn with no message in it."""
    from webui.runs import _history
    prior = [{"type": "turn_start"}, {"type": "user_msg", "text": "solve the plate"},
             {"type": "agent_msg", "text": "done"}, {"type": "done", "outcome": "no_result"}]
    current = prior + [{"type": "turn_start"}, {"type": "user_msg", "text": "now refine it"}]
    for i in range(len(current) - 1, -1, -1):
        if current[i]["type"] == "turn_start":
            cut = current[:i]
            break
    h = _history(cut)
    assert ("user", "") not in h and all(text.strip() for role, text in h if role == "user")
    assert h[0] == ("user", "solve the plate")


def test_upload_is_bound_to_a_run_and_to_simulation_file_types(client):
    sid = client.post("/api/sessions", json={"model": "mock", "test": True}).json()["id"]
    try:
        ok = client.post(f"/api/sessions/{sid}/upload",
                         files={"files": ("my part.msh", b"$MeshFormat\n", "application/octet-stream")})
        assert ok.status_code == 200 and ok.json()["saved"][0]["name"] == "my_part.msh"
        bad = client.post(f"/api/sessions/{sid}/upload",
                          files={"files": ("run.sh", b"rm -rf /", "text/plain")})
        assert bad.status_code == 415
        listing = client.get(f"/api/sessions/{sid}/files", params={"sub": "uploads"}).json()
        assert [e["name"] for e in listing["entries"]] == ["my_part.msh"]
        assert "/home/" not in json.dumps(listing), "absolute paths must not reach the browser"
        assert client.get(f"/api/sessions/{sid}/files", params={"sub": "../../"}).status_code in (403, 404)
    finally:
        client.delete(f"/api/sessions/{sid}")


def _solver_result(status, trustworthy=None):
    body = {"status": status}
    if trustworthy is not None:
        body["trustworthy_result"] = trustworthy
    return "[{'type': 'text', 'text': '" + json.dumps(body) + "'}]"


def test_a_solver_result_counts_only_when_openpaso_verified_it():
    from webui.outcome import classify_solver_result as c
    assert c(_solver_result("completed", True)) == "verified"
    assert c(_solver_result("completed", False)) == "unverified"
    assert c(_solver_result("completed")) == "unverified"
    assert c(_solver_result("completed_with_warnings", True)) == "unverified"
    assert c(_solver_result("failed")) == "failed"
    assert c("[{'type': 'text', 'text': 'Unknown solver: abaqus'}]") == "failed"
    assert c("") == "failed"


def _coupling_result(**fields):
    return "[{'type': 'text', 'text': '" + json.dumps(fields) + "'}]"


def test_a_ladder_answers_for_the_ladder_not_for_its_best_level():
    """couple_levels will carry a verdict for the whole ladder. Levels 1 and 2
    verified with level 3 a null exchange is the normal shape of a failed
    ladder, and taking the best evidence in the payload would call it
    finished — the claim this interface exists to refuse."""
    from webui.outcome import classify_solver_result as c
    bad = _coupling_result(all_levels_converged=False, trustworthy_result=False,
                           verification="level 3 is not a coupled result: residual 0.0 at the first step",
                           levels=[{"level": 1, "converged": True, "trustworthy_result": True},
                                   {"level": 2, "converged": True, "trustworthy_result": True},
                                   {"level": 3, "converged": True, "trustworthy_result": False,
                                    "coupled_evidence": "null exchange"}])
    good = _coupling_result(all_levels_converged=True, trustworthy_result=True,
                            verification="verified",
                            levels=[{"level": 1, "trustworthy_result": True},
                                    {"level": 2, "trustworthy_result": True}])
    assert c(bad) == "unverified"
    assert c(good) == "verified"
    # until the tool carries a verdict at all, a ladder stays "ran, not verified"
    assert c(_coupling_result(all_levels_converged=True,
                              levels=[{"level": 1, "converged": True}])) == "unverified"


def test_the_legacy_coupled_solve_report_is_read_as_a_run_that_happened():
    """coupled_solve answers in prose, not JSON: a convergence report and
    openPASO's verification note. Read as JSON it became "failed", and a
    coupling that really ran was reported as having computed nothing."""
    from webui.outcome import classify_solver_result as c
    ran = ("Coupling converged in 12 iterations (residual 4.1e-07)\n\n"
           "[openPASO verification: NOT VERIFIED — openPASO's independent critic has not "
           "reviewed this setup...]")
    reviewed = ("Coupling converged in 9 iterations\n\n"
                "[openPASO verification: LEGACY coupled_solve — critic-reviewed. Trust is "
                "governed by the convergence report above...]")
    assert c(ran) == "unverified"
    assert c(reviewed) == "unverified", "prose carries no machine-readable verdict"
    assert c("Backend not found: fenicsx or dealii") == "failed"
    assert c("Unknown problem: thermoelastic. Available: ['fsi']") == "failed"


def test_stop_counts_the_processes_that_really_ended(monkeypatch):
    from webui import proctree
    alive = {41, 42}
    monkeypatch.setattr(proctree, "_alive", lambda pid: pid in alive)
    monkeypatch.setattr(proctree.os, "kill", lambda pid, sig: alive.discard(pid) if pid == 41 else None)
    # 41 dies, 42 will not: the count is what ended, not what was asked to end
    assert proctree._end([41, 42], grace=0.2) == 1


def test_a_correction_sent_late_becomes_a_message_the_record_keeps():
    """It is run as a follow-up turn. Without a user message in the log, the
    history a restarted run is given lost it, though the transcript showed it."""
    from webui.runs import _history
    events = [
        {"type": "turn_start"}, {"type": "user_msg", "text": "Solve it."},
        {"type": "done", "outcome": "no_result"},
        {"type": "turn_start"}, {"type": "user_msg", "text": "Use a finer mesh."},
        {"type": "agent_msg", "text": "Refining."}, {"type": "done", "outcome": "no_result"},
    ]
    h = _history(events)
    assert ("user", "Use a finer mesh.") in h


def test_a_coupling_is_judged_before_its_record_is_shortened():
    """A coupling reply carries no "status" and routinely runs past the length
    at which a result is shortened for the record. Judged from that shortened
    copy it read as a call that computed nothing, and a failed one whose first
    participant looked good read as verified — the interface asserting the two
    things it exists to prevent."""
    from webui.outcome import classify_solver_result as c, shorten

    verified = _wrap({"converged": True, "iterations": 7, "trustworthy_result": True,
                      "verification": "VERIFIED - evidence and a critic review on record",
                      "history": [[i, 1.0 / (i + 1)] for i in range(300)],
                      "participant_output_logs": {"fluid": "x" * 4000, "solid": "y" * 4000}})
    failed = _wrap({"participants": [{"name": "fluid", "status": "completed", "trustworthy_result": True},
                                     {"name": "solid", "status": "failed", "trustworthy_result": False}],
                    "pad": "z" * 9000, "converged": False, "trustworthy_result": False})
    assert len(verified) > 8000 and len(failed) > 8000, "these are the everyday sizes"
    # the invariant: shortening a record for storage never changes the verdict
    assert c(verified) == "verified" and c(shorten(verified)) == "verified"
    assert c(failed) == c(shorten(failed)), "the same evidence, the same answer"
    assert c(shorten(failed)) != "verified", "a mixture is never a verified result"


def test_the_three_ladder_shapes_as_the_tool_really_answers_them():
    """The field names and shapes couple_levels actually lands with: a verdict
    for the ladder at the top, and one per level inside.

    The third is the one that catches a reader who rebuilds the verdict from
    the parts: every level says it is trustworthy, all_levels_converged is
    true, and the ladder is still not verified — the mesh never refined
    between levels, which no single level can see."""
    from webui.outcome import classify_solver_result as c, shorten

    verified = _wrap({"all_levels_converged": True, "trustworthy_result": True,
                      "verification": "VERIFIED - every level converged, exchanged a real "
                                      "flux, refined, and the review is on record",
                      "levels": [{"level": 1, "trustworthy_result": True},
                                 {"level": 2, "trustworthy_result": True}]})
    null_level = _wrap({"all_levels_converged": False, "trustworthy_result": False,
                        "verification": "level 2 is not a coupled result",
                        "levels": [{"level": 1, "trustworthy_result": True},
                                   {"level": 2, "trustworthy_result": False,
                                    "coupled_evidence": "residual 0.0 at the first step"}]})
    never_refined = _wrap({"all_levels_converged": True, "trustworthy_result": False,
                           "verification": "the mesh did not refine between levels",
                           "levels": [{"level": 1, "trustworthy_result": True},
                                      {"level": 2, "trustworthy_result": True}]})

    for name, reply, want in (("verified ladder", verified, "verified"),
                              ("null level", null_level, "unverified"),
                              ("never refined", never_refined, "unverified")):
        assert c(reply) == want, (name, c(reply))
        padded = reply.replace('"levels"', '"logs": "' + "x" * 9000 + '", "levels"')
        assert c(shorten(padded)) == want, (name, "shortened", c(shorten(padded)))


def test_every_reader_of_a_run_gives_the_same_verdict():
    """The transcript read the verdict recorded on the event; the run list and
    the downloaded record re-read the shortened text beside it. A coupling's
    trustworthy_result sits in the middle of a long reply, which is the part
    that is cut, so one run was Finished on its own page and "computed nothing"
    in the list — one fact, two readers, two answers."""
    from webui.outcome import classify_solver_result as c, fold, shorten

    reply = _wrap({"head": "a" * 6000, "converged": True, "trustworthy_result": True,
                   "verification": "VERIFIED", "tail": "b" * 6000})
    recorded = shorten(reply)
    assert c(reply) == "verified"
    assert c(recorded) != "verified", "the flag really is in the part that gets cut"

    events = [{"type": "turn_start"}, {"type": "user_msg", "text": "couple them"},
              {"type": "tool_result", "tool": "couple", "result": recorded,
               "verdict": c(reply)},
              {"type": "done"}]
    assert fold(events) == "completed", "the recorded verdict is what counts"


def test_a_report_with_no_verdict_of_its_own_is_not_verified_by_a_part_of_it():
    """The docstring's ladder: levels 1 and 2 verified, level 3 a null
    exchange. Taking the best evidence anywhere reported it as finished."""
    from webui.outcome import classify_solver_result as c
    mixed = _wrap({"levels": [{"level": 1, "trustworthy_result": True},
                              {"level": 2, "trustworthy_result": False,
                               "coupled_evidence": "null exchange"}]})
    allgood = _wrap({"levels": [{"trustworthy_result": True}, {"trustworthy_result": True}]})
    assert c(mixed) == "unverified"
    assert c(allgood) == "verified"


def _wrap(obj):
    return "[{'type': 'text', 'text': '" + json.dumps(obj) + "'}]"


def test_a_coupling_reports_its_verdict_in_its_own_shape():
    """couple and couple_precice carry the verification gate's verdict with no
    `status` field at all. Reading only the run shape called every verified
    coupling a failure, so a real coupled result could never be finished."""
    from webui.outcome import SOLVER_TOOLS, classify_solver_result as c
    assert c(_coupling_result(converged=True, iterations=7, trustworthy_result=True)) == "verified"
    assert c(_coupling_result(converged=True, trustworthy_result=False)) == "unverified"
    assert c(_coupling_result(converged=False, error="coupling driver failed")) == "failed"
    # a ladder answers for the ladder: one verified level inside it is not a
    # verdict about the whole, and the top of that reply carries none today
    nested = _coupling_result(all_levels_converged=True, levels_run=2,
                              levels=[{"converged": True, "trustworthy_result": False},
                                      {"converged": True, "trustworthy_result": True}])
    assert c(nested) == "unverified"
    assert {"couple", "couple_levels", "couple_precice", "verify_mesh_independence"} <= SOLVER_TOOLS


def test_the_run_list_is_scrubbed_like_every_other_answer(client):
    """A prompt can name the folder someone worked in."""
    sid = client.post("/api/sessions", json={"model": "mock", "test": True}).json()["id"]
    try:
        run = runs.get(sid)
        run.state["events"].append({"type": "user_msg", "seq": 1,
                                    "text": f"use the mesh in {Path.home()}/meshes/part.msh"})
        run.save()
        body = client.get("/api/sessions?all=true").text
        assert str(Path.home()) not in body and "~/meshes/part.msh" in body
    finally:
        client.delete(f"/api/sessions/{sid}")


def test_the_log_holds_the_result_the_model_was_given():
    """The correction was added to what the model read after the event had been
    recorded, so rebuilding the conversation from the log dropped a message the
    model had already acted on."""
    import asyncio as aio
    from langchain_core.tools import StructuredTool
    from webui.runner import ApprovalGate, _wrap_tool
    events = []

    async def emitter(e):
        events.append(e)

    tool = StructuredTool.from_function(func=lambda x: f"ran {x}", name="t", description="d")
    wrapped = _wrap_tool(tool, emitter=emitter, get_mode=lambda: "accept", gate=ApprovalGate(),
                         take_steers=lambda: [{"id": "st_1", "text": "Use a finer mesh."}])
    given = aio.run(wrapped.ainvoke({"x": "1"}))
    recorded = next(e for e in events if e["type"] == "tool_result")["result"]
    assert "Use a finer mesh." in given
    assert "Use a finer mesh." in recorded, "the log must match what the model read"


def test_a_correction_reaches_a_critic_while_it_works_and_the_main_agent_after():
    """A critic can run for many minutes. A correction queued until it finished
    arrived after the thing it was meant to prevent."""
    from webui.runs import Run
    run = Run.__new__(Run)
    run.steers = [{"id": "st_1", "text": "Stop running MPI tests."}]
    run.emit = lambda e: asyncio.sleep(0)
    run.push_snapshot = lambda: asyncio.sleep(0)
    first = run._take_steers("sa_1111")
    assert [s["text"] for s in first] == ["Stop running MPI tests."]
    assert run._take_steers("sa_1111") == [], "the same sub-agent is not told twice"
    # a second critic in the same run is a second worker and hears it too
    assert [s["text"] for s in run._take_steers("sa_2222")] == ["Stop running MPI tests."]
    assert [s["text"] for s in run._take_steers("main")] == ["Stop running MPI tests."]
    assert run.steers == []


def test_a_big_file_that_merely_mentions_a_field_series_is_not_read_as_one(tmp_path):
    """The head was searched for the words, so an ordinary large JSON whose
    notes mention a field series came back as a broken field descriptor and the
    page tried to draw it."""
    from webui import viz
    p = config.SANDBOX_ROOT / "webui_abc123def" / "notes.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"kind": "notes",
                             "text": "compared against a field_series written earlier",
                             "pad": "x" * 5_000_000}))
    try:
        assert viz._json(p)["kind"] != "field_series"
    finally:
        p.unlink()


def test_the_model_list_needs_no_network_without_a_key(monkeypatch):
    """A cold model list reached OpenRouter even with no key, which put a
    network call in front of a page (and a test suite) that needs none."""
    import asyncio as aio
    asked = []

    async def record(self, url, *a, **k):
        asked.append(str(url))
        raise OSError("no network in this test")

    monkeypatch.setattr(catalog, "_PRICES", {}, raising=False)
    monkeypatch.setattr(catalog, "_PRICES_AT", 0, raising=False)
    monkeypatch.setattr("httpx.AsyncClient.get", record)
    monkeypatch.setattr(config, "openrouter_key_source", lambda: (None, None))
    groups = aio.run(catalog.models())
    # the local probes still happen — they ask this machine whether a model
    # server is listening — but nothing leaves it
    assert not [u for u in asked if u.startswith(config.OPENROUTER_URL)], asked
    assert any(m["status"].startswith("no OpenRouter key")
               for g in groups["groups"] for m in g["models"] if g["kind"] == "openrouter")


def test_a_large_field_file_is_described_without_being_parsed(tmp_path):
    from webui import viz
    p = config.SANDBOX_ROOT / "webui_abc123def" / "field.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    frames = [[[0.123456] * 100 for _ in range(100)] for _ in range(60)]
    p.write_text(json.dumps({"kind": "field_series", "field": "vorticity", "unit": "1/s",
                             "nx": 100, "ny": 100, "vmin": -1.0, "vmax": 1.0,
                             "times": [i * 0.1 for i in range(40)], "frames": frames}))
    try:
        assert p.stat().st_size > 4 * 1024 * 1024, p.stat().st_size
        out = viz._json(p)
        assert out["kind"] == "field_series"
        assert out["field"] == "vorticity" and out["nx"] == 100 and out["n_frames"] == 40
        assert out["vmin"] == -1.0 and out["vmax"] == 1.0
    finally:
        p.unlink()


def test_a_file_name_with_a_question_mark_still_downloads(client):
    run = config.SANDBOX_ROOT / "webui_abc123def" / "work"
    run.mkdir(parents=True, exist_ok=True)
    odd = run / "sweep?a=1.txt"
    odd.write_text("value 3\n")
    try:
        from urllib.parse import quote
        url = "/sandbox-file/" + "/".join(quote(s, safe="") for s in
                                          ("webui_abc123def", "work", "sweep?a=1.txt"))
        r = client.get(url)
        assert r.status_code == 200 and "value 3" in r.text
    finally:
        odd.unlink()


def test_a_run_cut_off_by_a_restart_says_so_rather_than_guessing():
    """A run lives in the server process: stopping the server ends it wherever
    it was. The record used to stop mid-sentence and the page had to guess."""
    from webui.outcome import UNFINISHED, fold
    ev = [{"type": "turn_start"}, {"type": "user_msg", "text": "flow past a cylinder"},
          {"type": "tool_result", "tool": "run_bash", "result": "ok"},
          {"type": "error", "outcome": UNFINISHED,
           "message": "The server this run was working in was stopped..."},
          {"type": "done", "outcome": UNFINISHED}]
    assert fold(ev) == UNFINISHED
    # and without that record, a log that simply stops is still not a result
    assert fold(ev[:3]) == "running"


def test_outcome_is_judged_per_turn_and_needs_a_verified_solver_result():
    from webui.outcome import fold
    ev = lambda *xs: [x if isinstance(x, dict) else {"type": x} for x in xs]
    tool = lambda raw: {"type": "tool_result", "tool": "run_simulation", "result": raw}
    solved = tool(_solver_result("completed", True))
    shaky = tool(_solver_result("completed", False))
    unknown = tool("[{'type': 'text', 'text': 'Unknown solver: abaqus'}]")
    assert fold(ev("turn_start", "user_msg", "done")) == "no_result"
    assert fold(ev("turn_start", "user_msg", solved, "done")) == "completed"
    assert fold(ev("turn_start", "user_msg", shaky, "done")) == "unverified"
    assert fold(ev("turn_start", "user_msg", unknown, "done")) == "no_result"
    # an old record's "done: completed" is not taken on trust
    assert fold(ev("turn_start", "user_msg", {"type": "done", "outcome": "completed"})) == "no_result"
    assert fold(ev("turn_start", "user_msg", {"type": "error", "outcome": "failed"}, "done")) == "failed"
    # an error without an outcome of its own is a failure
    assert fold(ev("turn_start", "user_msg", solved, {"type": "error"}, "done")) == "failed"
    # a follow-up still working is running, whatever the turn before did
    assert fold(ev("turn_start", "user_msg", solved, "done", "turn_start", "user_msg")) == "running"
    # a follow-up that ran no solver has no result even if an earlier turn did
    assert fold(ev("turn_start", "user_msg", solved, "done", "turn_start", "user_msg", "done")) == "no_result"


def test_stop_ends_processes_started_in_the_run_folder():
    """Stop used to cancel the waiting coroutine and leave the command and its
    solver running while the screen said the run had stopped."""
    import subprocess, tempfile, time
    from webui import proctree
    with tempfile.TemporaryDirectory() as d:
        proc = subprocess.Popen(["bash", "-c", "sleep 120 & wait"], cwd=d, start_new_session=True)
        time.sleep(0.5)
        assert proc.pid in proctree.run_processes(Path(d))
        assert proctree.end_run_processes(Path(d)) >= 1
        proc.wait(timeout=10)
        assert proctree.run_processes(Path(d)) == []


def test_a_correction_is_handed_to_the_model_with_the_next_tool_result():
    from langchain_core.tools import StructuredTool
    from webui.runner import ApprovalGate, _wrap_tool
    pending = [{"id": "st_1", "text": "Use a finer mesh."}]
    def take():
        out = list(pending); pending.clear(); return out
    async def emitter(_e):
        return None
    tool = StructuredTool.from_function(func=lambda x: f"ran {x}", name="t", description="d")
    wrapped = _wrap_tool(tool, emitter=emitter, get_mode=lambda: "accept",
                         gate=ApprovalGate(), take_steers=take)
    result = asyncio.run(wrapped.ainvoke({"x": "1"}))
    assert "ran 1" in result and "Use a finer mesh." in result and "MESSAGE FROM THE USER" in result
    assert pending == []


def test_after_a_stop_the_model_is_given_the_steps_it_really_took():
    """Re-seeding used to hand back prompts and messages only, so after a Stop
    the model said nothing had been computed while the page showed the solves."""
    from webui.runs import _history
    events = [
        {"type": "turn_start"}, {"type": "user_msg", "text": "Solve the cantilever."},
        {"type": "agent_msg", "text": "I will run three meshes."},
        {"type": "tool_call_pending", "call_id": "a", "tool": "run_bash", "agent": "main",
         "args": {"command": "python cantilever.py 40"}},
        {"type": "tool_result", "call_id": "a", "tool": "run_bash", "result": "tip deflection -1.8e-3 relative error"},
        {"type": "tool_call_pending", "call_id": "b", "tool": "knowledge", "agent": "main", "args": {"query": "x"}},
        {"type": "tool_call_rejected", "call_id": "b"},
        {"type": "error", "outcome": "interrupted", "message": "Run stopped."},
        {"type": "done", "outcome": "interrupted"},
    ]
    h = _history(events)
    assert h[0] == ("user", "Solve the cantilever.")
    text = h[1][1]
    assert h[1][0] == "assistant"
    assert "python cantilever.py 40" in text and "-1.8e-3" in text
    assert "skipped this step" in text and "stopped the run" in text


def test_one_hung_step_can_be_ended_and_the_run_carries_on():
    """A correction waits for the current step; a step that hangs used to leave
    Stop, which ends the whole run, as the only way out."""
    import asyncio as aio, subprocess, time
    from langchain_core.tools import StructuredTool
    from webui import proctree
    from webui.runner import ApprovalGate, StepControl, _wrap_tool
    with tempfile.TemporaryDirectory() as d:
        def hang(x: str) -> str:
            r = subprocess.run(["bash", "-c", "sleep 300"], cwd=d, capture_output=True, text=True)
            return f"exit {r.returncode}"
        server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)", "-m", "server"], cwd=d)
        try:
            steps = StepControl()
            events = []
            async def emitter(e):
                events.append(e)
            tool = StructuredTool.from_function(func=hang, name="run_bash", description="d")
            wrapped = _wrap_tool(tool, emitter=emitter, get_mode=lambda: "accept", gate=ApprovalGate(),
                                 steps=steps, take_steers=lambda: [{"id": "s", "text": "Skip the MPI test."}])
            async def main():
                call = aio.create_task(wrapped.ainvoke({"x": "1"}))
                for _ in range(50):
                    await aio.sleep(0.1)
                    if steps.running():
                        break
                await aio.sleep(0.5)
                cid = steps.running()[0]
                steps.ended.add(cid)
                await aio.to_thread(proctree.end_step_processes, Path(d))
                return await aio.wait_for(call, timeout=15)
            t0 = time.monotonic()
            result = aio.run(main())
            assert time.monotonic() - t0 < 15
            assert "The user ended this step" in result and "Skip the MPI test." in result
            assert server.poll() is None, "the openPASO server of the run must survive"
            assert steps.tasks == {} and steps.ended == set()
        finally:
            server.kill()


# ───────────────────────────────────────────────────────────────────
# Runner end-to-end with mock LLM.
#
# We exercise the agent flow directly via runner.build_agent_for_session
# + runner.stream_turn rather than through TestClient's WebSocket. The
# TestClient's anyio-driven WS loop deadlocks against asyncio.to_thread
# inside the gated spawn_subagent tool; the runner itself works fine
# under plain asyncio, which is what the live server uses.
# ───────────────────────────────────────────────────────────────────
import asyncio
import tempfile
from contextlib import asynccontextmanager

from webui.runner import (ApprovalGate, build_agent_for_session,
                          open_agent_for_session, stream_turn)


def _run_turn(*, mode, prompt, approve_first=False):
    seen, events = [], []

    async def main():
        async def emitter(e):
            seen.append(e["type"])
            events.append(e)

        gate = ApprovalGate()

        def get_mode():
            return mode

        with tempfile.TemporaryDirectory() as d:
            agent = build_agent_for_session(
                model="mock", mcp_on=False, workdir=Path(d),
                emitter=emitter, get_mode=get_mode, gate=gate)

            turn = asyncio.create_task(stream_turn(
                agent=agent, user_text=prompt, emitter=emitter, emit_done=True))

            if approve_first:
                # Wait until the first pending call appears, then approve.
                while turn.done() is False:
                    pending = next((e for e in events
                                    if e["type"] == "tool_call_pending"), None)
                    if pending:
                        gate.resolve(pending["call_id"], True)
                        break
                    await asyncio.sleep(0.05)
            return await turn

    final = asyncio.run(main())
    return final, seen, events


def test_runner_mock_run_accept():
    final, seen, _ = _run_turn(
        mode="accept", prompt="Plan a Poisson MMS demo. Use the critic.")
    assert "subagent_spawned" in seen, (
        f"expected spawn_subagent in {set(seen)}")
    assert "subagent_returned" in seen
    assert "done" in seen
    assert "critic approved" in final.lower()


def test_runner_mock_plan_mode_pending_then_approve():
    final, seen, events = _run_turn(
        mode="plan", prompt="Begin.", approve_first=True)
    pending = [e for e in events if e["type"] == "tool_call_pending"]
    assert pending, "no tool_call_pending event in plan mode"
    # After approval the call must have executed (tool_result) and the
    # turn must have terminated (done).
    assert any(e["type"] == "tool_result" for e in events)
    assert "done" in seen, f"missing done in {set(seen)}"


def test_webui_keeps_mcp_context_open_for_agent_lifetime(monkeypatch):
    import agent as langgraph_agent

    lifecycle = []

    @asynccontextmanager
    async def fake_mcp_session(_workdir, **kwargs):
        # the product gets every tool and no sandbox jail; the campaign's
        # defaults (a fixed subset, bubblewrap) are for measurements only
        assert kwargs == {"surface": "all", "isolate": False}, kwargs
        lifecycle.append("opened")
        try:
            yield []
        finally:
            lifecycle.append("closed")

    monkeypatch.setattr(
        langgraph_agent, "openpaso_mcp_tools_session", fake_mcp_session)

    async def main():
        async def emitter(_event):
            return None

        with tempfile.TemporaryDirectory() as directory:
            async with open_agent_for_session(
                    model="mock", mcp_on=True, workdir=Path(directory),
                    emitter=emitter, get_mode=lambda: "accept",
                    gate=ApprovalGate()) as built:
                assert built is not None
                assert lifecycle == ["opened"]
            assert lifecycle == ["opened", "closed"]

    asyncio.run(main())


def test_a_run_does_not_belong_to_the_browser_tab():
    """Closing the socket used to cancel the run. The run now lives on the
    server; a mock turn completes with nobody subscribed."""
    from webui import runs
    s = sessions.new_session(model="mock", mode="accept", mcp_servers=[])
    sessions.save(s)
    try:
        async def main():
            run = runs.get(s["id"])
            assert not run.subscribers
            await run.prompt("Plan a Poisson demo. Use the critic.")
            await asyncio.wait_for(run.turn_task, timeout=60)
            await asyncio.sleep(0.2)
            types = [e["type"] for e in run.state["events"]]
            assert types[0] == "turn_start" and "user_msg" in types
            done = [e for e in run.state["events"] if e["type"] == "done"]
            assert len(done) == 1 and done[0]["outcome"] == "no_result", done
            # the tool connection is opened and closed by one task, whichever task asks
            closer = asyncio.create_task(run.close_agent())
            await closer
            assert run.agent is None and run._keeper is None
        asyncio.run(main())
    finally:
        runs.RUNS.pop(s["id"], None)
        sessions.delete(s["id"])
