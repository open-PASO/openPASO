"""Run a turn through the Claude Code CLI instead of an API model.

Most people who would try openPASO already have Claude Code and do not have an
API key, so this is the path that needs no key and no GPU: the CLI is driven
headless, openPASO is handed to it as an MCP server, and its streamed JSON is
translated into the same events the rest of the WebUI already speaks.

Nothing here knows about the agent loop. Claude Code runs its own.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import AsyncIterator, Callable

from . import config

BINARY = "claude"


def available() -> bool:
    return shutil.which(BINARY) is not None


def signed_in() -> bool | None:
    """Whether a Claude login is on this machine. None when it cannot be told.

    Being on PATH is not being signed in: an installed but unauthenticated CLI
    was offered as a ready model and failed on the first step. A login may also
    live in the system keyring, which cannot be read from here, so this says
    "unknown" rather than claiming the model is unusable."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return True
    if (Path.home() / ".claude" / ".credentials.json").is_file():
        return True
    return None


def _mcp_config(servers: list[str], workdir: Path | None = None) -> dict:
    """The --mcp-config payload, built from the same entries the WebUI shows.

    The openPASO server writes simulation output, coupling files and meshes to
    directories it takes from the environment, defaulting to shared ones next to
    the install. The LangGraph path points them at the run's own folder; this
    one did not, so a Claude Code run's results landed outside the run, missing
    from its Files view and its record, and two runs at once could overwrite
    each other."""
    out: dict[str, dict] = {}
    for sid in servers:
        spec = config.MCP_SERVERS.get(sid)
        if not spec:
            continue
        env = dict(spec.get("env_extra") or {})
        if workdir is not None:
            work = Path(workdir).resolve()
            env.update({
                "OPENPASO_CELL_WORKDIR": str(work),
                "OPENPASO_OUTPUT_DIR": str(work / "simulation_outputs"),
                "OPENPASO_COUPLING_DIR": str(work / "coupling"),
                "OPENPASO_MESH_DIR": str(work / "meshes"),
                "OPENPASO_BENCHMARK_DIR": str(work / "benchmark_results"),
            })
        out[sid] = {
            "command": spec["command"],
            "args": list(spec.get("args") or []),
            "env": env,
        }
    return {"mcpServers": out}


def _hook_settings(servers: dict) -> dict:
    """Claude Code settings for one run: the hook that refuses the main conversation's review
    filing (webui/critic_hook.py), for every openPASO server the run is given."""
    import shlex
    import sys
    hook = str(Path(__file__).resolve().parent / "critic_hook.py")
    return {"hooks": {"PreToolUse": [
        {"matcher": f"mcp__{sid}__submit_critic_review",
         # quoted: a path with a space or a shell character would not start the hook, and a hook
         # that does not start lets the call through (Copilot on the org PR)
         "hooks": [{"type": "command", "command": shlex.join([sys.executable, hook])}]}
        for sid in servers]}}


# The work an agent has to do here is write an input deck and run a solver, so
# it needs the file and shell tools as well as openPASO's own. This list says
# WHICH KINDS of thing a run may do; it is not a boundary on WHERE. These are
# Claude Code's own tools, and they take absolute paths: the run starts in its
# own folder but can read and write anywhere this account can, exactly as
# run_bash can on the other path. The interface says so where the mode is
# chosen, and a run's own Files view still shows only the run's folder.
WORK_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch"]


async def stream_turn(
    task: str,
    *,
    workdir: Path,
    servers: list[str],
    model: str | None = None,
    mode: str = "accept",
    emit: Callable[[dict], "asyncio.Future | None"],
    state: dict | None = None,
) -> str:
    """Run one task and emit WebUI events. Returns the final text."""
    if not available():
        raise RuntimeError(
            "Claude Code is not on PATH. Install it, or pick an API model instead.")
    if mode == "plan":
        raise RuntimeError(
            "Claude Code runs headless here, so it cannot stop and ask you to "
            "approve each step. Choose \"Run without asking\" for it, or pick "
            "another model to keep \"Ask before each step\".")

    cfg = _mcp_config(servers, workdir)
    tmp = Path(tempfile.mkdtemp(prefix="openpaso-cc-"))
    cfg_path = tmp / "mcp.json"
    cfg_path.write_text(json.dumps(cfg))
    settings_path = tmp / "settings.json"
    settings_path.write_text(json.dumps(_hook_settings(cfg["mcpServers"])))

    allowed = list(WORK_TOOLS) + [f"mcp__{sid}" for sid in cfg["mcpServers"]]
    cmd = [
        BINARY, "--print", task,
        "--output-format", "stream-json",
        "--verbose",
        "--permission-mode", "acceptEdits",
        "--allowedTools", *allowed,
        "--mcp-config", str(cfg_path),
        # only a sub-agent files a review (the hook refuses the main conversation's filing)
        "--settings", str(settings_path),
    ]
    if model:
        cmd += ["--model", model]
    # A follow-up continues the same Claude conversation instead of starting blind.
    if state and state.get("claude_session_id"):
        cmd += ["--resume", state["claude_session_id"]]

    workdir.mkdir(parents=True, exist_ok=True)
    root = str(workdir.resolve())
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=root,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        # one stream-json line can hold a whole file Claude read; the default
        # 64 KB line limit crashed runs with "chunk is longer than limit"
        limit=64 * 1024 * 1024,
        # its own process group, and the run's marker, so Stop can end it and
        # everything it launched (it was left running before)
        env={**os.environ, "PYVISTA_OFF_SCREEN": "true", "OPENPASO_CELL_WORKDIR": root},
        start_new_session=True,
    )
    # Its diagnostics go to stderr, which nothing read until stdout had ended.
    # A chatty run filled the pipe, and the child then blocked before writing
    # its result: the run hung with no way to tell why.
    errors: list[str] = []

    async def drain() -> None:
        while True:
            line = await proc.stderr.readline()
            if not line:
                return
            if len(errors) < 200:
                errors.append(line.decode("utf-8", "replace").rstrip())

    draining = asyncio.create_task(drain())
    try:
        return await _consume(proc, emit, state, errors)
    except asyncio.CancelledError:
        _kill(proc)
        raise
    except RuntimeError:
        raise                 # already carries what Claude Code said
    finally:
        draining.cancel()
        # one of these per turn, left behind on a long-lived server
        shutil.rmtree(tmp, ignore_errors=True)


def _kill(proc) -> None:
    import signal
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            return


async def _consume(proc, emit, state, errors: list[str] | None = None) -> str:

    final = ""
    calls: dict[str, str] = {}
    assert proc.stdout is not None
    async for line in _lines(proc.stdout):
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        kind = msg.get("type")
        if kind == "system" and msg.get("subtype") == "init":
            # which Claude model actually ran: the record used to say only
            # "claude-code", so nobody could tell what produced a number
            if state is not None:
                state["claude_session_id"] = msg.get("session_id") or state.get("claude_session_id")
                if msg.get("model"):
                    state["claude_model"] = msg["model"]
            if msg.get("model"):
                await _send(emit, {"type": "model_info", "model": msg["model"]})
        elif kind == "assistant":
            for block in (msg.get("message") or {}).get("content") or []:
                if block.get("type") == "text" and block.get("text", "").strip():
                    await _send(emit, {"type": "agent_msg", "text": block["text"]})
                elif block.get("type") == "tool_use":
                    cid = block.get("id") or uuid.uuid4().hex[:8]
                    # Strip the MCP prefix so the step reads as the tool's name.
                    name = str(block.get("name", "")).split("__")[-1]
                    calls[cid] = name
                    await _send(emit, {"type": "tool_call_pending", "call_id": cid,
                                       "tool": name, "args": block.get("input") or {}})
                    await _send(emit, {"type": "tool_call_executing",
                                       "call_id": cid, "tool": name})
        elif kind == "user":
            for block in (msg.get("message") or {}).get("content") or []:
                if block.get("type") != "tool_result":
                    continue
                cid = block.get("tool_use_id", "")
                body = block.get("content")
                if isinstance(body, list):
                    body = " ".join(b.get("text", "") for b in body
                                    if isinstance(b, dict))
                from .outcome import SOLVER_TOOLS, classify_solver_result, shorten, verdict_reason
                tool = calls.get(cid, "")
                event = {"type": "tool_result", "call_id": cid, "tool": tool,
                         "result": shorten(body)}
                if tool in SOLVER_TOOLS:
                    # judged from the WHOLE reply, as the LangGraph path does: the record
                    # keeps a shortened copy, and a Claude Code run's reply arrives wrapped
                    # as {"result": "<escaped JSON>"}. Judged from the copy, a run that
                    # finished was read as one that "computed nothing" (measured
                    # 2026-09-29: its status was completed_unverified).
                    event["verdict"] = classify_solver_result(str(body or ""))
                    event["why"] = verdict_reason(str(body or ""))
                await _send(emit, event)
        elif kind == "result":
            final = msg.get("result") or final
            # Claude Code can report a failure in this message and still exit
            # zero: is_error, or a subtype that names one. Treating any text as
            # a finished turn recorded such a run as a turn that merely
            # produced no result, which is a different thing entirely.
            if msg.get("is_error") or str(msg.get("subtype", "")).startswith("error"):
                failed = str(final or msg.get("subtype") or "it reported an error").strip()
                raise RuntimeError("Claude Code stopped with an error: " + failed[:400])
            if state is not None and msg.get("session_id"):
                state["claude_session_id"] = msg["session_id"]
            cost = msg.get("total_cost_usd")
            if cost is not None:
                await _send(emit, {"type": "cost", "usd": cost})
            usage = msg.get("usage") or {}
            if usage:
                await _send(emit, {"type": "token_count",
                                   "input": usage.get("input_tokens", 0),
                                   "output": usage.get("output_tokens", 0)})

    await proc.wait()
    if proc.returncode != 0:
        # A non-zero exit is a failure even when something was printed first.
        # Treating a final message as proof of success let a crashed command be
        # recorded as a turn that merely produced no result.
        said = "\n".join(errors[-20:]).strip() if errors else (str(final)[-400:] if final else "")
        raise RuntimeError(f"Claude Code exited {proc.returncode}: {said[:400]}")
    if not str(final).strip():
        # exit 0 and nothing said is not a turn that produced no result: it is a
        # turn that produced nothing at all, and reporting it as the former
        # would put "the reply above" under an empty space
        said = "\n".join(errors[-5:]).strip() if errors else ""
        raise RuntimeError("Claude Code finished without saying anything"
                           + (f". It printed: {said[:300]}" if said else "."))
    return final


async def _lines(stream: asyncio.StreamReader) -> AsyncIterator[str]:
    while True:
        raw = await stream.readline()
        if not raw:
            return
        text = raw.decode("utf-8", "replace").strip()
        if text:
            yield text


async def _send(emit, event: dict) -> None:
    res = emit(event)
    if asyncio.isfuture(res) or asyncio.iscoroutine(res):
        await res
