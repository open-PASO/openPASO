"""FastAPI app for the openPASO WebUI.

Start with::

    .venv-lg/bin/uvicorn webui.app:app --reload --port 8080

Endpoints:

* ``GET  /`` — single-page UI (HTML)
* ``GET  /api/models`` / ``/api/mcp_servers`` / ``/api/modes``
* ``GET  /api/file?rel=…`` — file text content (read only)
* ``GET  /api/viz?rel=…`` — visualization payload
* ``GET  /sandbox-file/{path:path}`` — raw file bytes for vtk.js etc.
* ``GET  /api/sessions`` / ``POST /api/sessions`` / ``DELETE``
* ``WS   /ws/{session_id}`` — streamed run channel

The WebSocket protocol is symmetric JSON:
``{"type": "...", ...}`` either direction. See :mod:`webui.runner` for
outbound event types and :func:`_handle_inbound` for the inbound set.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
import logging
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import catalog, config, files, runs, sessions, viz
from .outcome import fold as outcome_fold
from .privacy import scrub, scrub_text
from .runner import _session_workdir

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("openpaso.webui")

app = FastAPI(title="openPASO WebUI", version="0.1.0")


@app.on_event("shutdown")
async def _say_what_happened_to_live_runs():
    """Write the truth into every run that was working when the server went down.

    A run lives in this process. Stopping the server — a restart, a crash, a
    machine going to sleep — ends it wherever it was, and the run's record used
    to simply stop mid-sentence. The page then had to guess: "usually because
    the server was restarted". It is not a guess from in here."""
    for run in list(runs.RUNS.values()):
        if not run.running:
            continue
        with contextlib.suppress(Exception):
            if run.turn_task and not run.turn_task.done():
                run.turn_task.cancel()
            # and end what it had started: a solver does not stop because the
            # server that asked for it went away, and the record would then say
            # the run had stopped while it kept every core it was given
            from . import proctree
            ended = proctree.end_run_processes(run.workdir, 1.5, run.state.get("created_at"))
            note = ("The server this run was working in was stopped, so the run stopped with it. "
                    "What it had already done is in the record below; send a follow-up to carry on.")
            if ended:
                note += (f" {ended} process{'es' if ended != 1 else ''} it had started "
                         f"{'were' if ended != 1 else 'was'} ended.")
            # through emit, so these carry the sequence numbers and the server
            # marker every other ending has; a page that reconnects reads them
            # exactly as it reads a normal end
            await run.emit({"type": "error", "outcome": runs.UNFINISHED,
                            "message": note, "processes_ended": ended})
            await run.emit({"type": "done", "outcome": runs.UNFINISHED})
            run.save()
# The interface is served by this app, so cross-origin access is only ever
# wanted from the Vite dev server. A wildcard let any page a researcher had
# open read this sandbox and, while POST /api/file existed, write to it.
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:5173",
                                  "http://127.0.0.1:5173"],
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])

STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text()


# ───────────────────────────────────────────────────────────────────
# Config endpoints
# ───────────────────────────────────────────────────────────────────
@app.get("/api/models")
async def get_models():
    """Models grouped by where they run, each with whether it works right now,
    what it costs and where the data goes. The mock is never offered."""
    out = scrub(await catalog.models())
    # what this person's own runs with each model have cost: a measured range,
    # not an estimate, and only where there is one
    costs: dict[str, list[float]] = {}
    for p in config.SESSION_DIR.glob("*.json"):
        row = _summary(p)
        if row and row.get("prompt") and row.get("cost_usd") is not None and not row.get("running"):
            costs.setdefault(row["model"], []).append(float(row["cost_usd"]))
    for g in out["groups"]:
        for m in g["models"]:
            c = costs.get(m["id"])
            if c:
                m.update(past_runs=len(c), past_cost_low=min(c), past_cost_high=max(c))
    return out


def _served_build() -> str | None:
    """The bundle this server hands out, so a page can notice it is an old one.

    A tab left open across an update keeps running the code it was loaded with.
    Someone then uses a control that was fixed an hour ago and watches it not
    work, which is indistinguishable from a broken product."""
    try:
        html = (config.REPO / "webui" / "static" / "index.html").read_text()
        import re as _re
        m = _re.search(r"assets/(index-[A-Za-z0-9_-]+\.js)", html)
        return m.group(1) if m else None
    except OSError:
        return None


@app.get("/api/config")
async def get_config():
    return {"modes": [{"id": m, **config.MODE_INFO[m]} for m in config.MODES],
            "default_mode": config.DEFAULT_MODE,
            "docs_url": config.DOCS_URL,
            "max_running": config.MAX_RUNNING,
            "build": _served_build()}


def _artefacts(work: Path) -> list[dict]:
    """Every file a run wrote, with its size and checksum."""
    import hashlib
    out: list[dict] = []
    if not work.is_dir():
        return out
    root = work.resolve()
    for f in sorted(work.rglob("*")):
        # a link the run made can point anywhere; hashing what it points at
        # would put a file from outside the run into its record
        if f.is_symlink() or not f.is_file():
            continue
        if not f.resolve().is_relative_to(root):
            continue
        digest, size = hashlib.sha256(), 0
        with f.open("rb") as fh:                 # a run may write gigabytes
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        out.append({
            "path": str(f.relative_to(work)),
            "bytes": size,
            "sha256": digest.hexdigest(),
            "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(f.stat().st_mtime)),
        })
    return out


@app.get("/api/sessions/{sid}/manifest")
async def get_manifest(sid: str):
    """Everything needed to check or reproduce one run, in one file.

    A result that cannot be traced back to what produced it is not a result. The
    interface shows a summary; this is the record behind it: every event of the
    run in order, and a hash for every artefact it wrote.

    One thing here is not verbatim. A tool result longer than 8000 characters
    is shortened when it is recorded — both ends kept, and the middle marked
    with how much was left out. The model was given the whole of it; the
    verdict on a solver result was read from the whole of it before shortening
    and is recorded beside the text. The files themselves are on disk,
    unshortened, with their checksums below."""
    # a live run's record on disk is a checkpoint, not the present: turn
    # boundaries and endings are written at once and the rest every tenth event,
    # so a download during a run would miss the newest steps it promises
    live = runs.live(sid)
    if live is not None:
        state = live.state
    else:
        try:
            state = sessions.load(sid)
        except Exception:
            return JSONResponse({"error": f"no such run: {sid}"}, status_code=404)

    events = state.get("events") or []
    outcome = outcome_fold(events)

    work = config.SANDBOX_ROOT / f"webui_{sid}"
    # off the event loop: a run's files can be gigabytes, and hashing them here
    # held up every other run's events, and Stop with them
    artefacts = await asyncio.to_thread(_artefacts, work)

    check = await catalog.solvers()
    solvers = {r["name"]: r["version"] for r in check.get("solvers", [])
               if r["status"] == "available"}

    prompt = next((e.get("text") for e in events if e.get("type") == "user_msg"), None)
    return scrub({
        "run": sid,
        "outcome": outcome,
        "prompt": prompt,
        "model": state.get("model"),
        "mode": state.get("mode"),
        "mcp_servers": state.get("mcp_servers"),
        "tokens": {"in": state.get("tokens_in"), "out": state.get("tokens_out")},
        "solver_versions": solvers,
        "working_directory": f"eval_interactive/webui_{sid}",
        "artefacts": artefacts,
        "events": events,
        "openpaso_commit": _commit(),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })


def _commit() -> str | None:
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(config.REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


@app.get("/api/solvers")
async def get_solvers(refresh: bool = False):
    """What is installed, asked of openPASO's own Python in a subprocess.

    It used to run in the web server's Python, which is not the one openPASO
    uses (it lacked scikit-fem, so the page said 8 of 9 while runs had all nine),
    and inside the event loop, freezing every other request while it ran."""
    # the error of a failed check quotes the command, which names machine paths
    return scrub(await catalog.solvers(refresh=refresh))


@app.get("/api/mcp_servers")
async def get_mcp():
    return {"servers": [
        {"id": k, "label": v["label"], "default_on": v["default_on"]}
        for k, v in config.MCP_SERVERS.items()
    ]}


# ───────────────────────────────────────────────────────────────────
# Files & viz
# ───────────────────────────────────────────────────────────────────
_RUN_REL = re.compile(r"^webui_[0-9a-f]{6,32}/")


def _run_path(rel: str) -> str:
    """Files are served from run folders only. The sandbox also holds other
    working directories (evaluation campaigns, scratch) that are nobody's run.

    The name is checked, and then where it actually leads: a run can write a
    symlink, and "webui_A/work/elsewhere -> ../../webui_B/work" reads as a path
    inside run A while pointing into run B."""
    rel = (rel or "").lstrip("/")
    m = _RUN_REL.match(rel)
    if not m or ".." in Path(rel).parts:
        raise HTTPException(403, "Only files inside a run's folder can be opened here.")
    run_root = (config.SANDBOX_ROOT / m.group(0).rstrip("/")).resolve()
    target = (config.SANDBOX_ROOT / rel).resolve()
    if target != run_root and run_root not in target.parents:
        raise HTTPException(403, "That file is outside the run it is asked for.")
    return rel


@app.get("/api/file")
async def api_file(rel: str):
    try:
        return scrub(files.read_text(_run_path(rel)))
    except PermissionError:
        raise HTTPException(403, "That path is outside the run folders.")


# There is no POST /api/file: it once wrote anywhere under the sandbox with no
# authentication. Files reach a run only through its upload endpoint.


@app.get("/api/viz")
async def api_viz(rel: str):
    try:
        return scrub(viz.visualize(_run_path(rel)))
    except PermissionError:
        raise HTTPException(403, "That path is outside the run folders.")


# Text a run wrote can hold the home directory it worked in, and this is the
# path a Download link uses. Everything else the browser is shown is scrubbed;
# without this the one route that hands over whole files was the exception.
# What counts as text is decided by looking, not by a list of suffixes: a
# solver writes .4c, .i, .msh and .vtk files that are plain text, and an
# allowlist quietly let those through unscrubbed.
_SNIFF_BYTES = 8192
_SCRUB_WHOLE = 8 * 1024 * 1024        # bigger than this is streamed in pieces
_CARRY = 4096                          # so a path split across two pieces still matches


# Formats that are binary whatever their first bytes look like. A PDF opens
# with an ASCII header and turns binary later, so a sniff of the first kilobytes
# called it text, scrubbed it and served it as text: the file was changed on the
# way out and no longer opened.
_ALWAYS_BINARY = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".zip", ".gz",
                  ".bz2", ".xz", ".tar", ".h5", ".hdf5", ".npy", ".npz", ".vtu",
                  ".vtk", ".vtp", ".pvtu", ".exo", ".e", ".med", ".msh", ".stl",
                  ".step", ".stp", ".iges", ".igs", ".brep", ".woff", ".woff2", ".ttf"}


def _is_text(p: Path) -> bool:
    if p.suffix.lower() in _ALWAYS_BINARY:
        return False
    try:
        with p.open("rb") as fh:
            head = fh.read(_SNIFF_BYTES)
    except OSError:
        return False
    if b"\0" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        # a multi-byte character may straddle the end of the sample
        try:
            head[:-4].decode("utf-8")
        except UnicodeDecodeError:
            return False
    return True


_ENCODED_KEY = re.compile(r'"(?:frames|mask|data|image)"\s*:\s*"')
_LOOKS_ENCODED = re.compile(r"[A-Za-z0-9+/=\s]{64}")
_PEEK = 64          # how much of a value is examined before trusting the key


def _scrubbed_stream(p: Path):
    """The file, scrubbed, in pieces, so a huge log need not be held in memory.

    Each piece keeps the tail of the one before it in view, so a path across a
    boundary is still found — and the encoded payloads are passed through as
    they are, which the whole-file path does by looking at the value entire.
    A field file is tens of megabytes, so its frames cannot be held to be
    examined: this follows the key that opens the value and copies everything
    until the quote that closes it, however many pieces that takes. Without it
    a chunk boundary inside the base64 put the payload back in the scrubber's
    way, and the numbers a solver produced could be rewritten again."""
    carry, inside = "", False
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            text = carry + chunk
            carry = ""
            while text:
                if inside:                      # copying an encoded value
                    end = text.find('"')
                    if end < 0:
                        yield text.encode("utf-8")
                        text = ""
                        break
                    yield text[:end + 1].encode("utf-8")
                    text, inside = text[end + 1:], False
                    continue
                m = _ENCODED_KEY.search(text)
                if m:
                    # the key is not enough: a "data" field can hold prose, and
                    # passing that through unread would carry a home path out
                    value = text[m.end():]
                    if len(value) < _PEEK and len(text) < 1024 * 1024:
                        carry = text          # decide once the value is in view
                        text = ""
                        break
                    if _LOOKS_ENCODED.match(value[:_PEEK]):
                        yield scrub_text(text[:m.end()]).encode("utf-8")
                        text, inside = text[m.end():], True
                        continue
                    yield scrub_text(text[:m.end()]).encode("utf-8")
                    text = text[m.end():]     # ordinary text: scrub it as prose
                    continue
                # no key in view: scrub all but a tail, which may hold half of
                # a path or half of a key and is judged with the next piece
                if len(text) > _CARRY:
                    yield scrub_text(text[:-_CARRY]).encode("utf-8")
                    carry = text[-_CARRY:]
                else:
                    carry = text
                text = ""
    if carry:
        yield (carry if inside else scrub_text(carry)).encode("utf-8")


@app.get("/sandbox-file/{rel:path}")
async def sandbox_file(rel: str):
    try:
        p = files._safe(_run_path(rel))
    except PermissionError:
        raise HTTPException(403, "That path is outside the run folders.")
    if not p.is_file():
        raise HTTPException(404, "not a file")
    if not _is_text(p):
        return FileResponse(p)
    disposition = {"Content-Disposition": f'inline; filename="{p.name}"'}
    # an SVG is text and is scrubbed like text, but it is a picture: served as
    # plain text a browser shows its source instead of drawing it
    kind = ("image/svg+xml" if p.suffix.lower() == ".svg" else "text/plain") + "; charset=utf-8"
    if p.suffix.lower() == ".svg":
        # An SVG is a document: a model can write one containing a script, and
        # opening its address in a tab would run that script in this
        # interface's own origin, where the run records live. Drawn in a page
        # (<img>) it never runs; served directly it is locked down instead of
        # being served as something it is not.
        disposition["Content-Security-Policy"] = "default-src 'none'; style-src 'unsafe-inline'; sandbox"
        disposition["X-Content-Type-Options"] = "nosniff"
    if p.stat().st_size > _SCRUB_WHOLE:
        return StreamingResponse(_scrubbed_stream(p), media_type=kind, headers=disposition)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        raise HTTPException(404, "could not read that file")
    return Response(scrub_text(text), media_type=kind, headers=disposition)


# ───────────────────────────────────────────────────────────────────
# Sessions
# ───────────────────────────────────────────────────────────────────
_SUMMARY_CACHE: dict[str, tuple[float, dict]] = {}


def _summary(path: Path) -> dict | None:
    """One row of the run list, or None when the record is gone — another tab
    may delete a run between the listing and this."""
    sid = path.stem
    if not path.exists():
        return None
    live = runs.live(sid)
    mtime = None
    if live is not None:
        st, running, outcome = live.state, live.running, live.outcome()
        waiting = live.waiting
        with contextlib.suppress(OSError):
            mtime = path.stat().st_mtime
    else:
        try:
            mtime = path.stat().st_mtime
        except OSError:      # another tab deleted the run while this was reading
            return None
        hit = _SUMMARY_CACHE.get(sid)
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            st = json.loads(path.read_text())
        except Exception:
            return None
        running, waiting, outcome = False, False, outcome_fold(st.get("events") or [])
        if outcome == "running":
            outcome = runs.UNFINISHED
    events = st.get("events") or []
    prompt = next((e.get("text") for e in events if e.get("type") == "user_msg"), None)
    label, kind = catalog.model_label(st.get("model", ""))
    out = {
        "id": st.get("id"), "created_at": st.get("created_at"),
        "updated_at": mtime if mtime is not None else st.get("created_at"),
        "model": st.get("model"), "model_label": label, "model_kind": kind,
        "model_detail": st.get("model_detail") or st.get("claude_model"),
        "mode": st.get("mode"), "prompt": prompt, "outcome": outcome,
        "running": running,
        "waiting": waiting,
        "steps": sum(1 for e in events if e.get("type") == "tool_call_pending"),
        "cost_usd": st.get("cost_usd"),
    }
    if live is None and mtime is not None:
        _SUMMARY_CACHE[sid] = (mtime, out)
    return out


@app.get("/api/sessions")
async def list_sessions(all: bool = False):
    """Runs a person has actually started, newest first. Records with no prompt
    (a page that was opened and left) and test-model runs are not work, so they
    are left out unless asked for."""
    rows = []
    # the mtime is read once, here, and a record that disappears between the
    # glob and this is simply not in the list rather than a failed request
    dated = []
    for path in config.SESSION_DIR.glob("*.json"):
        try:
            dated.append((path.stat().st_mtime, path))
        except OSError:
            continue
    paths = [path for _, path in sorted(dated, key=lambda x: x[0], reverse=True)]
    for path in paths:
        # a live run is read on the event loop: its events and its pending
        # approvals change there, and reading them from a worker thread could
        # catch a half-written list. Only records on disk are worth offloading.
        row = (_summary(path) if runs.live(path.stem)
               else await asyncio.to_thread(_summary, path))
        if not row:
            continue
        if not all and (not row["prompt"] or row["model_kind"] == "test"):
            continue
        rows.append(row)
    # a prompt can name the folder someone worked in; every other route scrubs
    return scrub({"sessions": rows, "running": runs.running_count()})


@app.post("/api/sessions")
async def new_session(body: dict | None = None):
    body = body or {}
    model = body.get("model")
    mode = body.get("mode") or config.DEFAULT_MODE
    # the test model fabricates answers: it exists for the test suites, which
    # set OPENPASO_TEST_MODEL. A request asking for it is not enough, or any
    # caller of this unauthenticated local API could make one.
    allow_mock = bool(body.get("test")) and config.ALLOW_TEST_MODEL
    known = (set(config.OPENROUTER_MODELS) | {config.CLAUDE_CODE_ID}
             | {k for k in config.MODELS if k != "mock" or allow_mock})
    if model not in known:
        raise HTTPException(400, f"unknown model: {model}")
    if mode not in config.MODES:
        raise HTTPException(400, f"unknown mode: {mode}")
    if not allow_mock:
        # the same answer the picker gives, so a run cannot be started against a
        # model server that is not running or a key that is not there
        offered = {m["id"]: m for g in (await catalog.models())["groups"] for m in g["models"]}
        info = offered.get(model)
        if info is None and model == config.CLAUDE_CODE_ID:
            # not in the catalogue at all: Claude Code is not installed here, so
            # the run would be created and fail at its first step instead
            raise HTTPException(409, "Claude Code is not installed on this machine. "
                                     "Your prompt was not sent.")
        if info and not info["available"]:
            raise HTTPException(409, f"{info['label']} cannot run right now: {info['status']}. "
                                     "Your prompt was not sent.")
    if runs.running_count() >= config.MAX_RUNNING:
        raise HTTPException(409, f"{config.MAX_RUNNING} runs are already working on this machine. "
                                 "Your prompt was not sent. Wait for one to finish or stop one, then press Run again.")
    if model == config.CLAUDE_CODE_ID and mode == "plan":
        raise HTTPException(400, "Claude Code cannot stop to ask before each step. "
                                 "Choose 'Run without asking', or a different model.")
    s = sessions.new_session(model=model, mode=mode,
                             mcp_servers=body.get("mcp_servers"))
    sessions.save(s)
    return s


@app.post("/api/sessions/{sid}/prompt")
async def start_run(sid: str, body: dict | None = None):
    """Send a run its message over HTTP.

    The first prompt used to be held in the browser and sent once the socket
    said hello. Closing the tab in that moment left a run with no prompt: it was
    on the server, hidden from the list because nothing had been asked of it,
    and the message was gone."""
    body = body or {}
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "nothing to send")
    try:
        run = runs.get(sid)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "no such run")
    refused = await runs.start_turn(run, text, body.get("attachments") or [])
    if refused:
        raise HTTPException(409, refused)
    return {"sent": True}


@app.get("/api/sessions/{sid}")
async def get_session(sid: str):
    try:
        run = runs.get(sid)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "no such run")
    return scrub({**run.snapshot(), "events": run.state["events"]})


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    live = runs.live(sid)
    if live is not None and live.running:
        raise HTTPException(409, "This run is still working. Stop it first.")
    if live is not None:
        await live.close_agent()
        runs.RUNS.pop(sid, None)
    import shutil
    live = runs.live(sid)
    if live is not None and live.running:
        raise HTTPException(409, "This run is still working. Stop it first.")
    if not re.fullmatch(r"[0-9a-f]{6,32}", sid or ""):
        # before anything is built from it: "../other_run" would otherwise name
        # a folder whose processes were ended before the id was refused
        raise HTTPException(400, "bad run id")
    folder = config.SANDBOX_ROOT / f"webui_{sid}"
    # A run the server no longer holds (it was restarted under it) can still
    # have a solver of its own running. Deleting the folder under it left it
    # computing against files that were gone, with nothing on screen to say so.
    if folder.is_dir():
        from . import proctree
        try:
            since = sessions.load(sid).get("created_at")
        except Exception:
            since = None
        work = folder / "work"
        # a run can replace its own work directory with a link; ending "its"
        # processes would then mean ending whatever lives where that points
        if work.exists() and work.resolve().is_relative_to(folder.resolve()):
            await asyncio.to_thread(proctree.end_run_processes, work, 3.0, since)
    try:
        deleted = sessions.delete(sid)
    except ValueError:
        raise HTTPException(400, "bad run id")
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)
    _SUMMARY_CACHE.pop(sid, None)
    if live is not None:
        # a tab still holding this run could otherwise prompt it or change its
        # mode, and Run.save would write the record back after the delete
        live.deleted = True
        for ws in list(live.subscribers):
            with contextlib.suppress(Exception):
                await ws.close()
        live.subscribers.clear()
    return {"deleted": deleted}


@app.get("/api/sessions/{sid}/files")
async def run_files(sid: str, sub: str = ""):
    """This run's own folder only. The file panel used to open at the root of
    the sandbox and list every run's directory side by side."""
    try:
        runs.get(sid)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "no such run")
    base = f"webui_{sid}/work"
    rel = f"{base}/{sub}".rstrip("/") if sub else base
    try:
        listing = files.list_dir(rel)
    except PermissionError:
        raise HTTPException(403, "That path is outside this run's folder.")
    # a prefix is not a boundary: "webui_x/work2" starts with "webui_x/work"
    got = listing.get("rel") or ""
    if got != base and not got.startswith(base + "/"):
        raise HTTPException(403, "outside this run")
    for e in listing.get("entries", []):
        e.pop("abs_path", None)
        e["sub"] = e["rel_path"][len(base) + 1:]
    listing.pop("path", None)
    listing["sub"] = sub
    return listing


@app.post("/api/sessions/{sid}/upload")
async def upload(sid: str, files_in: list[UploadFile] = File(..., alias="files")):
    """Put a person's own geometry, mesh or input into a run's folder.

    Bound to one run and to a list of file types a simulation can use. There was
    no way to bring your own geometry in at all, so the tool could only solve
    problems it could generate from a sentence."""
    try:
        run = runs.get(sid)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "no such run")
    import re as _re
    dest = run.workdir / "uploads"
    dest.mkdir(parents=True, exist_ok=True)
    if not dest.resolve().is_relative_to(run.workdir.resolve()):
        # the run made "uploads" a link somewhere else; a person's file would
        # then be written outside the run that was promised to hold it
        raise HTTPException(409, "This run's uploads folder does not point inside the run. "
                                 "Nothing was written.")
    saved = []
    for up in files_in:
        name = _re.sub(r"[^A-Za-z0-9._-]", "_", Path(up.filename or "file").name)[:120]
        if Path(name).suffix.lower() not in runs.UPLOAD_SUFFIXES:
            raise HTTPException(415, f"{name}: this file type is not accepted. Accepted: "
                                     + " ".join(sorted(runs.UPLOAD_SUFFIXES)))
        # two different names can come out of that substitution the same way
        # ("a b.msh" and "a_b.msh"); the second write would replace the first
        # while the answer said both were saved
        target = dest / name
        if target.exists():
            stem, suffix, n = Path(name).stem, Path(name).suffix, 2
            while target.exists():
                name = f"{stem}-{n}{suffix}"
                target = dest / name
                n += 1
        size = 0
        with target.open("wb") as fh:
            while chunk := await up.read(1 << 20):
                size += len(chunk)
                if size > runs.UPLOAD_MAX_BYTES:
                    fh.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(413, f"{name} is larger than 500 MB.")
                fh.write(chunk)
        saved.append({"name": name, "bytes": size})
        await run.emit({"type": "file_uploaded", "name": name, "bytes": size})
    run.save()
    return {"saved": saved}


# ───────────────────────────────────────────────────────────────────
# WebSocket: a tab subscribes to a run; the run does not belong to the tab
# ───────────────────────────────────────────────────────────────────
@app.websocket("/ws/{sid}")
async def ws_endpoint(ws: WebSocket, sid: str):
    await ws.accept()
    try:
        run = runs.get(sid)
    except (FileNotFoundError, ValueError):
        await ws.send_text(json.dumps({"type": "error", "message": f"no such run: {sid}"}))
        await ws.close()
        return
    try:
        # The live record, not a copy read from disk: a second tab used to load
        # its own stale copy and write it back over the running one on close.
        # Greeting and subscribing happen under the run's send lock: an event
        # appended after the snapshot but delivered before the greeting was
        # dropped by the page, which clears what it holds when the greeting
        # arrives, so a step could be missing from that tab for good.
        async with run.send_lock:
            await ws.send_text(json.dumps(scrub({"type": "hello", "session": run.snapshot(),
                                                 "events": run.state["events"]}), default=str))
            run.subscribers.add(ws)
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            await _handle_inbound(run, msg, ws)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("ws error on run %s", sid)
    finally:
        run.subscribers.discard(ws)
        # Leaving a page no longer stops the work. An idle run with nobody
        # watching gives back its openPASO server process until it is needed.
        if not run.running and not run.subscribers:
            await run.close_agent()


async def _tell(ws: WebSocket, message: str):
    with contextlib.suppress(Exception):
        await ws.send_text(json.dumps({"type": "notice", "message": message}))


async def _handle_inbound(run: "runs.Run", msg: dict, ws: WebSocket):
    t = msg.get("type")
    if t == "prompt":
        refused = await runs.start_turn(run, msg.get("text", ""), msg.get("attachments"))
        if refused:
            await _tell(ws, refused)
    elif t == "steer":
        if run.running:
            await run.steer(msg.get("text", "").strip())
        else:
            await run.prompt(msg.get("text", ""))
    elif t == "stop":
        ended = await run.stop()
        if ended < 0:
            await _tell(ws, "Nothing is running in this run.")
    elif t == "end_step":
        why = await run.end_step(msg.get("call_id", ""))
        if why:
            await _tell(ws, why)
    elif t == "approve":
        run.gate.resolve(msg.get("call_id", ""), True)
    elif t == "reject":
        run.gate.resolve(msg.get("call_id", ""), False, msg.get("reason", ""))
    elif t == "set_mode":
        mode = msg.get("mode")
        if mode not in config.MODES:
            return
        if mode == "plan" and run.state.get("model") == config.CLAUDE_CODE_ID:
            await _tell(ws, "Claude Code cannot stop to ask before each step.")
            return
        run.state["mode"] = mode
        run.save()
        released = run.gate.open_all(True) if mode == "accept" else 0
        await run.emit({"type": "mode_changed", "mode": mode})
        if released:
            # the wrapper reads the mode before it waits, so a step already
            # waiting stayed waiting and the run looked stuck on "Waiting for you"
            await _tell(ws, f"{released} step{'s' if released != 1 else ''} that "
                            f"{'were' if released != 1 else 'was'} waiting for you now run.")
        await run.push_snapshot()
    elif t == "set_model":
        if any(e.get("type") == "user_msg" for e in run.state["events"]):
            # what produced a result must stay what the record says produced it
            await _tell(ws, "The model of a run cannot change after it has started. Start a new run.")
            return
        wanted = msg.get("model")
        groups = await catalog.models()
        offered = {m["id"]: m for g in groups["groups"] for m in g["models"]}
        if wanted not in offered:
            # the same rule as POST /api/sessions: the test model and anything
            # unknown are not choices a person can make
            await _tell(ws, f"{wanted} is not a model you can choose here.")
            return
        if not offered[wanted]["available"]:
            await _tell(ws, f"{offered[wanted]['label']} cannot run right now: {offered[wanted]['status']}.")
            return
        if wanted == config.CLAUDE_CODE_ID and run.mode() == "plan":
            await _tell(ws, "Claude Code cannot stop to ask before each step. "
                            "Switch steps to \"Run without asking\" first.")
            return
        run.state["model"] = wanted
        await run.close_agent()
        run.save()
        await run.push_snapshot()
