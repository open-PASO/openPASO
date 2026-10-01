"""LangGraph agents for the open-weight ablation.

Two agent constructors:

* :func:`build_bare_agent`  — host-side toolset only (bash + web search +
  spawn_subagent). No openPASO MCP. Mirrors what Claude has in v1 BARE.
* :func:`build_mcp_agent`   — async context manager yielding the same host-side
    toolset plus the campaign openPASO MCP tools. Keep the context open for the full
    run so server-side state survives between calls.

Both conditions get parity with what Claude Code offers natively:

| Host-side tool       | Why both conditions need it                        |
|---------------------|----------------------------------------------------|
| ``run_bash``        | Equivalent of Claude's Bash; runs scripts/solvers  |
| ``read_file``       | Equivalent of Claude's Read                        |
| ``write_file``      | Equivalent of Claude's Write                       |
| ``web_search``      | Equivalent of Claude's WebSearch (literature/benchmarks) |
| ``spawn_subagent``  | Equivalent of Claude's Agent tool; needed so the   |
|                     | model can fulfil the MANDATORY CRITIC protocol the |
|                     | openPASO server prompts it to follow                  |

MCP_FULL exposes the explicit ``CAMPAIGN_MCP_TOOL_ALLOWLIST`` below. Deprecated
coupling shims and environment-mutating setup/reload tools are deliberately not
part of the experimental surface; adding a tool is a reviewed contract change,
not an automatic side effect of server registration.

All LLM calls go through ``langchain_openai.ChatOpenAI`` pointed at a local
vLLM server, which surfaces Qwen2.5's native tool-call format through the
OpenAI schema.
"""
from __future__ import annotations

import atexit
import contextvars
import asyncio
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Sequence

from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

REPO = Path(__file__).resolve().parents[1]

PORTS = {"7b": 8000, "14b": 8001, "32b": 8002}

# WHAT OPENPASO RECOMMENDS AND WHAT THE AGENT CAN CALL MUST BE THE SAME SET.
# This list was frozen on 2026-09-01. `couple_levels` was written on 2026-09-11
# for the failure it is named after -- "six proven couplings never reached level
# 3 when every level cost ten calls" -- and, because nothing re-checked the list,
# stayed unreachable: 3 invocations in 2095 trajectories against 1076 of
# `couple`, while openPASO's text recommending it appears in 260 of them.
# `check_input` is what the coupled ladder's own step-1 brief tells the worker to
# run. tests/test_a_tool_openpaso_tells_you_to_call_can_be_called.py is what keeps
# the two sides together from here on.
#
# tests/test_what_openpaso_recommends_is_reachable.py holds the same line from the served-text side.
# coupled_solve stays off deliberately: the text names it in order to say
# DEPRECATED, and the test reads that sentence rather than the token.
# WHAT A SILENT SUB-AGENT HANDS BACK. One sentence, shared word for word with
# the browser interface's own handler (webui/runner.py), so a person reading a
# transcript and a person reading a campaign trajectory see the same words.
# `{who}` is the role that was silent -- critic, verifier or researcher -- or
# "sub-agent". This branch keeps the old product name until the freeze; the
# rename script maps it.

# A SUB-AGENT'S REPLY IS BOUNDED BEFORE THE PARENT CARRIES IT. Measured: a critic
# sub-agent's free-text verdict ran to 121,752 characters (it was cut only by the
# model's output cap) and went into the parent's context whole, where every later
# call re-read it. Both arms; the start and the end are kept, and the cut says so.
SUBAGENT_REPORT_CAP = 24_000


def _bounded_report(text: str, cap: int = SUBAGENT_REPORT_CAP) -> str:
    if len(text) <= cap:
        return text
    head, tail = text[: cap * 2 // 3], text[-(cap // 3):]
    return (head + f"\n\n[... {len(text) - len(head) - len(tail):,} characters of this "
            f"sub-agent's reply left out here: it ran to {len(text):,} characters; its start "
            f"and its end are kept ...]\n\n" + tail)


SILENT_SUBAGENT_REPORT = (
    "[the {who} returned no text. This is NOT approval and NOT a review: it "
    "produced nothing. Treat the step as not done. Do not write its answer for "
    "it \u2014 run it again, or say that it produced nothing. openPASO's "
    "verification gate holds no review for this setup.]")

CAMPAIGN_MCP_TOOL_ALLOWLIST = frozenset({
    "audit_results",
    "check_input",
    "couple",
    "couple_levels",
    "couple_precice",
    "developer",
    "discover",
    "examples",
    "generate_mesh",
    "knowledge",
    "prepare_simulation",
    "run_simulation",
    "run_with_generator",
    "session_insights",
    "submit_critic_review",
    "verify_interface_flux",
    "verify_mesh_independence",
    "verify_pde_consistency",
    "visualize",
    # writes the served participant contract (solve elided) to a path through
    # the same door the knowledge reply uses; recommended by the worker brief
    "write_participant_contract",
})

_CRITIC_BLOCK = (
    "MANDATORY CRITIC: For every major step (problem setup, parameter "
    "choices, mesh/discretisation, BCs, solver choice, result "
    "interpretation), call `spawn_subagent` with role=\"critic\" and a "
    "ruthlessly skeptical task description. Pass the current state to the "
    "critic. Only proceed once the critic returns an explicit \"approved\" "
    "verdict. Do not approve your own work.\n"
)

BARE_SYSTEM = (
    "You are a finite-element simulation assistant. You will be given a "
    "problem statement and a writable result file path. You have host-side "
    "tools only — no FEM-aware MCP layer. Solve the problem from first "
    "principles using whatever solvers are installed on the system. Write "
    "scripts (Python / 4C YAML / etc.), run them with `run_bash`, and "
    "produce the requested RESULT lines in the result file.\n\n"
    "Tools available: run_bash, read_file, write_file, web_search, "
    "spawn_subagent.\n\n"
)
# NOTE: the mandatory-critic instruction is deliberately NOT part of the
# baseline. The critic is one of the things openPASO provides, so giving it to the
# unequipped arm hands the control group an openPASO method and understates the
# measured difference. The two arms are: host tools only (BARE) vs the openPASO
# tool layer, which includes the mandatory critic (MCP).

# THIS STRING IS THE ONLY ONE THE MODEL EVER READS.
#
# src/server.py builds an 8,919-character `instructions` block — workflow,
# mandatory critic, mesh independence, coupling. The client throws it away:
# `grep -rn "instructions"` over the installed langchain_mcp_adapters package
# returns ZERO hits, because get_tools() returns tool schemas and nothing else.
# An edit to server.py's instructions is decoration; an edit here reaches the
# agent. That was learned the hard way — audit_results was added to server.py's
# workflow as step 5 and went on being called by 1 run in 325.
#
# It also named the wrong coupling tools. `coupled_solve` and `transfer_field`
# are marked DEPRECATED in server.py and were called ZERO times in 325 MCP
# runs, while `couple` — which this prompt never mentioned — was used in 70.
def _mcp_system_prompt() -> str:
    """The openPASO arm's system text is openPASO's OWN instructions string -- the
    one server.py hands to FastMCP -- read from the product (core.instructions).
    MCP clients fold or drop a server's instructions, so the harness puts the
    same bytes in front of the model; it adds only its own tool wiring (the
    host-side tools, and how to run the critic openPASO demands with this
    harness's spawn_subagent). No knowledge, no file names, no coaching lives
    here: whatever openPASO should say about itself is said in openPASO."""
    from core.instructions import INSTRUCTIONS    # src/ is put on sys.path below
    return (INSTRUCTIONS
            + "\n\nHost-side tools (also available): run_bash, read_file, "
              "write_file, web_search, spawn_subagent.\n\n"
            + _CRITIC_BLOCK)


# ────────────────────────────────────────────────────────────────────
# LLM factory
# ────────────────────────────────────────────────────────────────────
def _llm(size: str, *, temperature: float, seed: int) -> ChatOpenAI:
    port = PORTS[size]
    return ChatOpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="not-used-by-vllm",
        model=f"qwen2.5-{size}",
        temperature=temperature,
        seed=seed,
        timeout=600,
    )


# ────────────────────────────────────────────────────────────────────
# Host-side tools (parity with Claude Code's native surface)
# ────────────────────────────────────────────────────────────────────
# THE AGENT CANNOT SEE A CLOCK, AND IT GUESSES BADLY.
#
# The prompt states the wall-clock budget once, at the start, and nothing ever
# tells the agent how much is left. Measured on C9, 27B, seeds 22 and 23: both
# wrote COULD_NOT_COMPLETE blaming the budget — "Time constraints (45 minutes)
# prevented completion", "Insufficient time ... within the 45-minute budget" —
# after using 1093 s and 1110 s of 2700. They stopped at eighteen minutes
# believing they were out of forty-five, and threw away 59% of the run.
#
# So the harness stamps the remaining time on every command result. It is set
# by the runner, it carries no domain content, and BOTH ARMS get it: the bare
# arm builds its shell tool from this same function, and a clock is not an
# openPASO capability.
_DEADLINE = None

_SENSITIVE_ENV_MARKERS = (
    "API_KEY", "TOKEN", "SECRET", "PASSWORD", "PASSPHRASE",
    "CREDENTIAL", "AUTH", "COOKIE", "ASKPASS",
)
_OWNED_SCRATCH: set[Path] = set()


@atexit.register
def _cleanup_owned_scratch() -> None:
    for scratch in tuple(_OWNED_SCRATCH):
        shutil.rmtree(scratch, ignore_errors=True)


def _clean_subprocess_env() -> dict[str, str]:
    """Return runtime environment without credentials or key-vault paths."""
    clean = {
        key: value for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in _SENSITIVE_ENV_MARKERS)
    }
    clean.pop("OPENPASO_BLIND_KEYS", None)
    clean.pop("SSH_AUTH_SOCK", None)
    # THE OPERATOR'S LANGUAGE IS NOT THE AGENT'S, AND IT IS NOT THE
    # CATALOGUE'S.
    #
    # This function copies the operator's environment, LANG included. On this
    # machine that is de_DE.UTF-8, so the kernel and the shell answered the
    # agent in German: `Zugriff auf 'exports.json' nicht moeglich: Datei oder
    # Verzeichnis nicht gefunden`, fifteen times across three of the nine most
    # recent coupled trajectories. Three costs, in order of how much they hurt:
    #
    #  * the signal catalogue is keyed on ENGLISH error text --
    #    ModuleNotFoundError, DIVERGED_PC_FAILED, FACTOR_NUMERIC_ZEROPIVOT. A
    #    German message matches nothing, so the failure-mode lookup is
    #    unreachable for exactly the failures it exists to explain (found by
    #    the interface session, 2026-09-19).
    #  * LC_NUMERIC comes with it, and under de_DE bash's own
    #    `printf '%.1f' 1.5` REFUSES its argument and prints `1,0`. Every
    #    float() in this repository raises on a decimal comma.
    #  * a model prompted in English has to guess at the message.
    #
    # Set HERE rather than on the jail's --setenv list, because the product
    # path builds its shell from this same function with no jail at all: a
    # person running one simulation was getting the German errors too.
    clean["LC_ALL"] = "C.UTF-8"
    clean["LANG"] = "C.UTF-8"
    clean["LANGUAGE"] = ""
    return clean


def _pin_backend_runtime_env(env: dict[str, str], *,
                             home: Path | None = None,
                             workspace: Path | None = None) -> None:
    """Expose mounted solver runtimes after the sandbox replaces ``HOME``."""
    host_home = (home or Path.home()).resolve()
    host_workspace = (workspace or REPO.parent).resolve()
    defaults = (
        ("FENICS_PYTHON",
         host_home / "miniconda3/envs/fenics/bin/python", "file"),
        ("DUNE_PYTHON",
         host_home / "miniconda3/envs/dune-py313/bin/python", "file"),
        ("FEBIO_BINARY", host_home / "FEBio/bin/febio4", "file"),
        ("DEAL_II_DIR", host_home / "dealii/build", "dir"),
        ("SPARTA_BINARY",
         host_workspace / "sparta/src/spa_serial", "file"),
        ("SPARTA_ROOT", host_workspace / "sparta", "dir"),
        ("SPARTA_DATA_DIR", host_workspace / "sparta/data", "dir"),
    )
    for key, path, kind in defaults:
        exists = path.is_file() if kind == "file" else path.is_dir()
        if not env.get(key) and exists:
            env[key] = str(path)


_ACTION_BUDGET = None        # (limit, counter) set by the runner
_ACTIONS_USED = 0


def note_action() -> None:
    """Count one tool call, so the agent can be told what it has left."""
    global _ACTIONS_USED
    _ACTIONS_USED += 1


def _time_left_note() -> str:
    """What the agent has left -- in ACTIONS as well as minutes.

    THE CLOCK WAS NEVER THE BINDING CONSTRAINT. Measured from file mtimes over
    six coupled runs: five wrote their submission at 93-99% of their whole
    file-activity span, with 8 to 115 seconds of activity after it, and the
    only run that reached a gradeable order with both prescribed codes proven
    submitted at 68%. They run out of ACTIONS, and a note in minutes cannot
    tell a model that it has eight tool calls left of fifty.

    BATS (arXiv 2511.17006) measured the bare budget tracker at 40.4% fewer
    searches and 31.3% lower cost at a budget of ten, with ReAct going
    12.6% -> 24.6% on BrowseComp once budget awareness was added. This is the
    cheapest of the four interventions that measurement pointed at.

    Both halves are omitted when unset, so nothing is invented: an unset
    deadline or budget prints nothing rather than a guess.
    """
    parts = []
    if _DEADLINE is not None:
        import time as _t
        left = _DEADLINE[0] - _t.time()
        total = _DEADLINE[1]
        parts.append("clock: budget spent" if left <= 0 else
                     f"clock: {int(left // 60)} min left of {int(total // 60)}")
    # NO INVENTED ACTION LIMIT. The graph ceiling is RECURSION_LIMIT = 1000 and
    # run_blind.py's own comment says it "sits above what the clock can" reach,
    # so there is no action budget to count down to. Reporting "465 of 500
    # left" would tell the agent it has room while the clock is what ends the
    # run. What is true and useful is how many actions it has SPENT, and how
    # much clock remains -- so the urgency trigger is keyed on the clock.
    if _ACTIONS_USED:
        parts.append(f"actions spent: {_ACTIONS_USED}")
    # NO PACING OR COACHING TEXT HERE: the harness states the clock and the
    # actions spent, nothing else. A sentence telling the agent when to write
    # or what a partial result is worth is coaching, verifies nothing, and
    # does not belong in either layer.
    return ("\n[" + " | ".join(parts) + "]") if parts else ""


def _format_audit_reply(findings) -> str:
    """The audit's wording, in ONE place, for every route that submits.

    Extracted when the shell route was added: the same submission must get the
    same message whether it was written with write_file or a heredoc, and two
    copies of this wording would drift the moment either is edited.
    """
    if findings == "NOEVIDENCE":
        return ("\n[auto-audit: no per-level result files were found "
                "to check, so nothing here was verified.]")
    if findings:
        return ("\n\nAUTO-AUDIT of your result files (your own files "
                "only, no reference solution):\n" + findings)
    if findings == "":
        return ("\n[auto-audit: no findings -- self-consistent, which "
                "is necessary but not sufficient for correct]")
    return ""


def _add_mount_dirs(argv: list[str], root: Path, target: Path,
                    made: set[Path]) -> None:
    """Create bubblewrap mount points below a tmpfs-covered directory."""
    relative = target.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if current not in made:
            argv.extend(("--dir", str(current)))
            made.add(current)


def _sandbox_scratch_path(workdir: Path) -> Path:
    work = workdir.resolve()
    digest = hashlib.sha256(str(work).encode()).hexdigest()[:24]
    return Path("/tmp/openpaso-cell-scratch") / digest


def _relocate_dune_cache(dune_cache: Path) -> None:
    """Point copied DUNE compiler commands at the cache's sandbox mount."""
    script = dune_cache / "python" / "dune" / "generated" / "buildScript.sh"
    if not script.is_file():
        raise RuntimeError(
            f"neutral DUNE baseline has no generated compiler script: {script}")
    text = script.read_text()
    prefix_line = next(
        (line.strip() for line in text.splitlines()
         if line.strip().startswith("DUNE_CXX_COMPILER_LAUNCHER=")
         and line.strip().endswith("/compiler_launcher.sh")),
        None)
    if prefix_line is None:
        raise RuntimeError(
            "neutral DUNE baseline compiler script has no cache-root marker")
    old_root = prefix_line.split("=", 1)[1].removesuffix(
        "/compiler_launcher.sh")
    sandbox_root = "/tmp/dune-cache/dune-py"
    launchers = (script, dune_cache / "compiler_launcher.sh")
    for launcher in launchers:
        if not launcher.is_file():
            raise RuntimeError(
                f"neutral DUNE baseline is missing launcher: {launcher}")
        content = launcher.read_text()
        launcher.write_text(content.replace(old_root, sandbox_root))
        if old_root in launcher.read_text():
            raise RuntimeError(
                f"DUNE launcher still references baseline path: {launcher}")


def sandbox_scratch_for(workdir: Path) -> Path:
    """Return the host scratch visible as ``/tmp`` to exactly one cell."""
    root = Path("/tmp/openpaso-cell-scratch")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    scratch = _sandbox_scratch_path(workdir)
    scratch.mkdir(mode=0o700, exist_ok=True)
    _OWNED_SCRATCH.add(scratch)
    baseline = os.environ.get("OPENPASO_DUNE_CACHE_BASELINE")
    dune_cache = scratch / "dune-cache" / "dune-py"
    if baseline and not dune_cache.exists():
        dune_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(baseline), dune_cache, symlinks=True)
        _relocate_dune_cache(dune_cache)
    return scratch


def cleanup_sandbox_scratch(workdir: Path) -> None:
    """Remove disposable caches after the cell's processes have exited."""
    scratch = _sandbox_scratch_path(workdir)
    shutil.rmtree(scratch, ignore_errors=True)
    _OWNED_SCRATCH.discard(scratch)



# HOW MANY STEPS A SUB-AGENT GETS BEFORE THE HARNESS ENDS IT MID-TASK.
#
# This was a bare literal, 40, which is about twenty tool calls. Measured on
# C2 iteration 2 (2026-09-23): the cap ended a sub-agent with "Sorry, need more
# steps to process this request." in 3 of 5 cells, against 1 of 5 in the
# previous round. In one of the three it cut the side-A worker one call after
# it had run `rm -f config.json imports.json ... && rm -rf out-vtk-files`,
# leaving only the script; the orchestrator then wrote both participants
# itself, and that cell is the one whose Kratos side built a clockwise
# triangle and graded unphysical.
#
# A worker that writes and debugs a participant needs more than twenty calls.
# Doubled, named, and recorded in every ledger next to the provider pin, so a
# later reader can tell which rounds ran under which cap without consulting a
# commit. Both arms share spawn_subagent, so this is a harness setting and not
# a capability of either side. The wall (45 min) and the parent's call budget
# are untouched.
WORKER_RECURSION_LIMIT = 80

def _runtime_mount_roots() -> tuple[Path, ...]:
    """The solver runtimes rebound into the cell after the home tmpfs.

    Lifted out of the jail builder so the shell's path rule can say what
    is legitimate using the same list the mount uses, rather than a second
    copy that drifts from it.
    """
    workspace = REPO.parent.resolve()
    home = Path.home().resolve()
    return (
        REPO / ".venv",
        Path(sys.prefix),
        Path(sys.base_prefix),
        workspace / "open-fem-agent/.venv",
        workspace / "febio-src/cbuild",
        workspace / "sparta/src",
        workspace / "sparta/data",
        workspace / "sparta/examples",
        home / "miniconda3",
        home / "4C",
        home / "FEBio",
        home / "dealii",
        home / ".local/share/python",
        home / ".local/include",
        home / ".local/lib",
    )


def _paths_outside_the_cell(command: str, workdir: Path) -> list[str]:
    """Absolute paths in `command` that sit under the host home but outside the cell.

    Narrow on purpose. The mounted solver runtimes are absolute paths under the
    same home -- every coupled run names one to invoke its interpreter -- and a
    rule that fired on those would fire on the one command the run cannot avoid.
    So the list is: under the host's home, not inside the cell, not inside a
    mounted runtime root, and not the campaign tree itself, which the agent
    never sees and whose paths only appear here as the cell's own prefix.
    """
    home = Path.home().resolve()
    work = workdir.resolve()
    allowed = [work, *(r.resolve() for r in _runtime_mount_roots() if r.is_dir())]
    found: list[str] = []
    for raw in re.findall(r"(?<![\w/])" + re.escape(str(home)) + r"[\w./+-]*", command):
        candidate = Path(raw.rstrip("/.,;:'\""))
        if candidate == home or any(candidate.is_relative_to(a) for a in allowed):
            continue
        if raw not in found:
            found.append(raw)
    return found


def _private_dirs_in(root: Path) -> list[Path]:
    """Directories directly inside a mounted runtime root that a cell must not read.

    A `.claude` folder (Claude Code settings; its allowlisted commands can carry
    credentials) and any checkout of this project or its predecessors (a folder
    with src/tools/ beside a pyproject.toml or a CLAUDE.md). Top level only: a
    runtime root's own tree is the solver's, and that is what a cell may read.
    """
    out: list[Path] = []
    try:
        children = sorted(root.iterdir())
    except OSError:
        return out
    for child in children:
        try:
            if child.is_symlink() or not child.is_dir():
                continue
            if child.name == ".claude":
                out.append(child)
            elif (child / "src" / "tools").is_dir() and (
                    (child / "pyproject.toml").is_file() or (child / "CLAUDE.md").is_file()):
                out.append(child)
        except OSError:
            continue
    return out


def _sandboxed_process_argv(workdir: Path, process: list[str], *,
                            source_repo: Path | None = None,
                            isolate: bool = True) -> list[str]:
    """Run a process with only this cell and private scratch writable.

    `isolate=False` returns the bare argv with no bubblewrap at all, for the
    PRODUCT path (`run_agent.py`). The evaluation harness never passes it and
    keeps the hard requirement below, which is what
    tests/test_campaign_host_isolation.py pins.

    WHY THE PRODUCT PATH DOES NOT ISOLATE. bubblewrap is Linux-only and this
    jail additionally rebinds a fixed list of runtime roots -- ~/miniconda3,
    ~/4C, ~/FEBio, ~/dealii, a sibling checkout's venv -- which is this
    machine's layout, not a stranger's. Someone whose FEniCS lives in
    ~/mambaforge would get "solver not found" inside the sandbox while it works
    on the host. A blind evaluation needs that jail and pays for it with a
    fixed machine; a person running their own simulation does not, and a
    product that only works on one directory layout is worse than one that
    trusts the user's own permissions.

    The trade is stated rather than hidden: read_file and write_file confine
    themselves to `workdir`, but a SHELL cannot be confined by a path check,
    so `run_bash` on this path runs with the user's own permissions.
    `run_agent.py` says so on startup.
    """
    if not isolate:
        return list(process)

    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError(
            "bubblewrap is required for blind-run shell isolation")

    work = workdir.resolve()
    private_tmp = sandbox_scratch_for(work)
    home = Path.home().resolve()
    workspace = REPO.parent.resolve()
    # THE INTERPRETER RUNNING THIS MUST SURVIVE THE tmpfs OVER $HOME, and it is
    # named here by where it actually is rather than by a checkout name.
    #
    # The list below used to start at `workspace / "open-fem-agent/.venv"`,
    # a sibling directory that happened to be this machine's older checkout.
    # openPASO's own `.venv` was not in it. A clone anywhere under $HOME --
    # which is what the README tells a reader to make -- therefore had its
    # virtual environment hidden by the tmpfs, and `run_agent.py` died before
    # the first model call with `McpError: Connection closed` above a bare
    # `bwrap: execvp .../.venv/bin/python: No such file or directory`. Nothing
    # in that pair names the cause. It was invisible here because a clone
    # OUTSIDE $HOME (in /tmp, say) is not hidden and works fine.
    #
    # `sys.prefix` and `sys.base_prefix` cover a venv layered on conda or on a
    # system Python, where the executable is a symlink out of the venv; REPO's
    # own `.venv` covers the symlink path itself, which must exist to be
    # followed.
    runtime_paths = _runtime_mount_roots()

    argv = [
        bwrap,
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-ipc",
        "--ro-bind", "/", "/",
        "--dev-bind", "/dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/dev/shm",
        # Hide user credentials, every other checkout, and historical runs.
        # Only the explicit solver runtime roots below are rebound.
        "--tmpfs", str(home),
    ]
    made: set[Path] = set()
    for runtime in runtime_paths:
        if runtime.is_dir():
            _add_mount_dirs(argv, home, runtime, made)
            argv.extend(("--ro-bind", str(runtime), str(runtime)))
    # A SOLVER'S ROOT CAN HOLD WHAT NO CELL MAY READ, so what it holds of ours
    # is hidden again after the root is mounted. Measured 2026-09-24: the 4C
    # root is mounted because the binary lives in it, and it also held an
    # untracked copy of this project's predecessor (its knowledge, templates and
    # run records; 19 comparison-arm cells of older rounds opened files in it)
    # and a Claude Code settings file whose allowlisted commands carry an
    # administrator password. Found by shape, in every mounted root, so the
    # next copy dropped into a solver tree is hidden without a new line here.
    for runtime in runtime_paths:
        for hidden in _private_dirs_in(runtime):
            argv.extend(("--tmpfs", str(hidden)))
    dune_host_cache = home / "miniconda3/envs/dune-py313/.cache"
    if dune_host_cache.is_dir():
        argv.extend(("--tmpfs", str(dune_host_cache)))

    # THE CELL SPOKE GERMAN TO AN ENGLISH-PROMPTED MODEL, AND GOT ITS DECIMALS
    # WRONG.
    #
    # The operator's environment is passed through, this machine runs
    # LANG=de_DE.UTF-8, and the kernel and shell messages the agent reads came
    # back as `Zugriff auf 'exports.json' nicht möglich: Datei oder Verzeichnis
    # nicht gefunden`. Fifteen of them were served into three of the nine most
    # recent C3 trajectories. The same locale sets LC_NUMERIC, under which
    # bash's own `printf '%.1f' 1.5` REFUSES the argument and prints `1,0` --
    # a wrong number, silently, in any shell arithmetic the agent does, and a
    # decimal comma that every float() in this repository raises on.
    #
    # C.UTF-8 is POSIX numbers and English messages with UTF-8 text intact.
    argv.extend(("--setenv", "LC_ALL", "C.UTF-8",
                 "--setenv", "LANG", "C.UTF-8",
                 "--setenv", "LANGUAGE", ""))

    # `/tmp` persists between this cell's shell calls, but maps back inside the
    # cell rather than to the host's shared scratch tree.
    argv.extend(("--bind", str(private_tmp), "/tmp"))

    for hidden_root in (workspace, Path("/tmp")):
        if work.is_relative_to(hidden_root):
            _add_mount_dirs(argv, hidden_root, work, made)
            break
    argv.extend(("--bind", str(work), str(work)))

    cwd = work
    if source_repo is not None:
        source = source_repo.resolve()
        source_mount = Path("/tmp/openpaso-source")
        _add_mount_dirs(argv, Path("/tmp"), source_mount, made)
        argv.extend(("--ro-bind", str(source), str(source_mount)))
        sessions = source / "data" / "sessions"
        if sessions.is_dir():
            session_output = work / ".openpaso_sessions"
            session_output.mkdir(parents=True, exist_ok=True)
            argv.extend(("--bind", str(session_output),
                         str(source_mount / "data" / "sessions")))
        cwd = source_mount / "src"

    # THE TMPFS OVER $HOME WAS WRITABLE, AND EVERY CALL GOT A FRESH ONE.
    #
    # So a shell command that wrote outside the cell -- into a sibling of the
    # host home it had inferred from its own working-directory path --
    # SUCCEEDED, printed nothing unusual, and lost the file the moment the call
    # returned. Measured on 2026-09-19: `mkdir -p ~alex/openpaso_scratch && echo hi
    # > .../note.txt && cat .../note.txt` returns rc=0 and prints `hi`; the next
    # call cannot find the file. 155 coupled runs referred to a path outside
    # their cell and three of the five C3 runs of 2026-09-18 spent six to ten
    # minutes of a forty-five minute wall on it.
    #
    # Read-only turns that into an error in the same call, where the agent is
    # still looking. It is applied AFTER the runtime binds, because those must
    # be mounted into the tmpfs first; `--remount-ro` changes this one mount
    # point and not what is mounted beneath it, so the DUNE JIT cache tmpfs
    # above stays writable. The cell's own home is not this path: HOME is set
    # to the cell below.
    argv.extend(("--remount-ro", str(home)))

    argv.extend((
        "--setenv", "HOME", str(work),
        "--setenv", "TMPDIR", "/tmp",
        "--setenv", "XDG_CACHE_HOME", "/tmp/.cache",
        "--setenv", "PYTHONPYCACHEPREFIX", "/tmp/pycache",
        "--setenv", "DUNE_PY_DIR", "/tmp/dune-cache",
        "--chdir", str(cwd),
    ))
    argv.extend(process)
    return argv


def _sandboxed_bash_argv(workdir: Path, command: str, *,
                         isolate: bool = True) -> list[str]:
    """Build the argv for one host shell call, isolated unless told otherwise."""
    return _sandboxed_process_argv(
        workdir, ["/bin/bash", "-lc", command], isolate=isolate)


def _bash_tool_for(workdir: Path, *, audit_on_submit: bool = False,
                   isolate: bool = True, budget_note: bool = True,
                   advice: bool = False):
    """A `run_bash` tool bound to `workdir`.

    `isolate=False` and `budget_note=False` are the PRODUCT path's settings
    (`run_agent.py`): no bubblewrap, and no `[actions spent: N]` suffix. That
    suffix is the harness's wall-clock stamp and a person running one
    simulation has no campaign clock to be reminded of. Both default to the
    harness's behaviour so nothing there changes.
    """

    def _note() -> str:
        return _time_left_note() if budget_note else ""

    # THE AUTO-AUDIT WAS ATTACHED TO write_file ONLY, AND AGENTS SUBMIT WITH A
    # HEREDOC.
    #
    # Its docstring says the write of RESULT.txt is "the only moment that
    # reaches 100% of submitters". Measured over the openPASO-arm runs whose
    # trajectory records the write at all: 57% wrote RESULT.txt by SHELL only,
    # 29% by both, 14% by write_file only — and an auto-audit reply appears in
    # 29% of them. So the hook reached about a quarter of submitters, not all.
    #
    # C1_27b_MCP_seed84 is the case in full. It wrote RESULT.txt with
    # write_file into a nested work/work/ directory, where the audit fired and
    # correctly reported "found NO per-level result files to check"; then it
    # ran `rm -rf work` and wrote the real submission with a run_bash heredoc,
    # which no audit watches. Its residual history is 0.5*0.5^k, bit-identical
    # at all three levels, and the audit refuses exactly that when it is given
    # the chance — every check fires when run by hand.
    #
    # So the same audit now also runs after a shell command that TOUCHED
    # RESULT.txt. Nothing is forced and nothing is blocked: the findings are
    # appended to the reply the agent is already reading, at the moment the
    # submission exists.
    def _audit_after_shell(before: float | None) -> str:
        if not audit_on_submit:
            return ""
        try:
            rt = next(iter(sorted(workdir.rglob("RESULT.txt"))), None)
            if rt is None:
                return ""
            if before is not None and rt.stat().st_mtime <= before:
                return ""            # untouched by this command
            body = rt.read_text(errors="replace")
            # THE GIVE-UP CHECK BELONGS ON THIS PATH TOO, and its absence here
            # cost a whole run. C2_27b_MCP_seed1202 finished the work -- six
            # solution files, six interface files, three residual histories and
            # six run logs, the complete deliverable set -- and then wrote
            # COULD_NOT_COMPLETE. The check built for exactly that fired zero
            # times, because it hung on write_file alone while this same file
            # already records that 57% of submitters write RESULT.txt by shell
            # only. Nineteenth instance of the shape: the mechanism existed,
            # was instrumented, and could not see the case it was built for.
            head = ""
            if "COULD_NOT_COMPLETE" in body.upper():
                try:
                    head = _work_on_disk_contradicting_a_give_up(workdir)
                except Exception:                      # noqa: BLE001
                    head = ""
                if head:
                    head = "\n\n" + head
            findings = _audit_submission(rt, body)
        except Exception as exc:                       # noqa: BLE001
            return f"\n[auto-audit unavailable: {type(exc).__name__}]"
        return head + _format_audit_reply(findings)

    # THE PER-LEVEL ARTEFACTS ARE WRITTEN BY THE AGENT'S OWN SOLVER SCRIPTS,
    # WHICH NO write_file HOOK CAN SEE.
    #
    # The early artefact check was attached to write_file and reached NONE of
    # round 7's three openPASO runs, while the submission audit reached all three.
    # Measured in their work dirs: 6, 3 and 11 Python scripts producing 12, 5
    # and 15 per-level CSVs. The agent writes a program with write_file and the
    # PROGRAM writes the deliverables, so the only channel that sees them is
    # the shell command that ran it. This is the same lesson this file already
    # records for RESULT.txt -- 57% of submitters wrote it by shell only -- and
    # the fix there was never extended to the artefacts.
    # GENERIC: any level-indexed artefact (<kind>_level<k>[_<side>].csv/.log).
    # The kinds are taken from the files' own names below, so the hook knows
    # no task's naming scheme.
    _ART = ("*_level*.csv", "*_level*.log")
    # A PARTICIPANT SCRIPT WRITTEN BY HEREDOC IS STILL A PARTICIPANT SCRIPT.
    # 18 of 18 runs on the coupled cell set FACE_HEAT_FLUX with no condition;
    # catching that only on write_file would miss every agent that uses a
    # heredoc, which this file already measured at 57% for RESULT.txt.
    # A 4C DECK IS A SCRIPT FOR THIS PURPOSE. _fourc_deck_write_check judges
    # .yaml/.yml/.dat and returns "" for anything else, so a route watching only
    # *.py could never hand it a deck. Measured over the live trajectories:
    # 229 of 486 decks (47%) reach disk by shell heredoc, against 452 of 7339
    # participant scripts (6%).
    _SCRIPTS = ("*.py", "*.yaml", "*.yml", "*.dat")

    def _artefact_mtimes() -> dict:
        out = {}
        for pat in _ART:
            for f in workdir.rglob(pat):
                try:
                    out[f] = f.stat().st_mtime
                except OSError:
                    pass
        return out

    def _script_mtimes() -> dict:
        out = {}
        for pat in _SCRIPTS:
            for f in workdir.rglob(pat):
                try:
                    out[f] = f.stat().st_mtime
                except OSError:
                    pass
        return out

    def _script_check_after_shell(before: dict) -> str:
        # ONE FLAG WAS DOING TWO JOBS. `audit_on_submit` switches on the
        # campaign's GRADING hooks -- RESULT.txt, COULD_NOT_COMPLETE, the
        # *_level*.csv deliverable family -- and it also switched on the script
        # LINTS, which are ordinary product capability: they read the script the
        # user's agent just wrote and name calls known to stop the run on this
        # install. openPASO's own rule is that every check in workspace_advisor
        # is product code, not evaluation code.
        #
        # Measured: in Option B nothing linted a written script at all. The
        # NGSolve `sym`/`Div` rules added this morning, and the
        # constant-deliverable check added this afternoon, reached the
        # evaluation arm and no product user. A deal.II step trial made it
        # concrete -- 52 of its 60 calls were shell, its files arriving by
        # `cat >` heredoc, which is the route that was hardest gated off.
        if not (audit_on_submit or advice):
            return ""
        try:
            now = _script_mtimes()
            touched = [f for f, t in now.items()
                       if before.get(f) is None or t > before[f]]
            for f in sorted(touched, key=lambda x: now[x], reverse=True):
                try:
                    # A solver writes .yaml output too (4C leaves a monitor file
                    # per Dirichlet condition). Both content checks read what
                    # they are given, so cap the read rather than the glob: a
                    # deck is kilobytes, and looks_like_deck() refuses the rest.
                    if f.stat().st_size > 2_000_000:
                        continue
                    _txt = f.read_text(errors="replace")
                    got = (_constant_deliverable_check(f, _txt)
                           + _script_noop_check(f, _txt)
                           + _registry_attribute_check(f, _txt)
                           + _extra_script_checks(f, _txt)
                           + _fourc_deck_write_check(f, _txt)
                           + _participant_write_check(f, _txt)
                           + _config_write_check(f, _txt))
                except OSError:
                    continue
                if got:
                    return got            # one finding per command, not per file
        except Exception:                 # noqa: BLE001
            return ""
        return ""

    def _artefact_check_after_shell(before: dict) -> str:
        if not audit_on_submit:
            return ""
        try:
            now = _artefact_mtimes()
            touched = [f for f, t in now.items()
                       if before.get(f) is None or t > before[f]]
            if not touched:
                return ""
            # ONE CHECK PER KIND, not one check overall.
            #
            # The first version took the single newest touched file. A solver
            # script writes every artefact in one go, so "newest" is arbitrary
            # within the batch: measured on a nine-file write it picked
            # residual_level1.csv, whose history was fine, and the INTERFACE
            # SIGN -- the finding that decided the round -- was never looked
            # at. Grouping by kind bounds the output at two blocks while
            # making sure each check that the batch made possible is run.
            # THE PATTERNS OVERLAP AND MTIMES TIE. "residual_level*.csv" is
            # also matched by "*_level*.csv", and files written by one `cp`
            # can share an mtime to float precision, so max() picked an
            # arbitrary representative per kind and the same file — hence the
            # same sentence — could be checked under two kinds in one reply.
            # Caught by this hook's own test, which counted a finding twice.
            # Deterministic tie-break, one check per FILE, one copy per BLOCK.
            blocks, chosen = [], set()
            kinds: dict = {}
            for f in touched:
                stem = f.name.lower().split("_level", 1)[0]
                kinds.setdefault((stem, f.suffix.lower()), []).append(f)
            for _kind in sorted(kinds):
                same = kinds[_kind]
                newest = max(same, key=lambda f: (now[f], f.name))
                if newest in chosen:
                    continue
                chosen.add(newest)
                got = (_level_index_check(workdir, newest)
                       + _identical_levels_check(workdir, newest)
                       + _wrong_level_run_log_check(workdir, newest)
                       + _discarded_proof_check(
                           newest, newest.read_text(errors="replace"))
                       + _early_artefact_check(workdir, newest))
                if got and got not in blocks:
                    blocks.append(got)
            return "".join(blocks)
        except Exception:                              # noqa: BLE001
            return ""

    def _outside_path_note(command: str) -> str:
        """The rule `write_file` states, stated for the shell as well.

        NOT behind `audit_on_submit`: the openPASO arm's auto-audit is a capability
        of the product and belongs there, but where a tool may write is the
        contract of the tool, and the bare arm runs in the same jail. Giving
        one arm the sentence and the other the bare errno would be a harness
        difference dressed as a product difference.

        The jail now refuses the write, so this says where to go instead; the
        record shows agents reading the kernel's message as "make the parent
        directory first" and trying again in the same place.
        """
        try:
            outside = _paths_outside_the_cell(command, workdir)
        except Exception:                              # noqa: BLE001
            return ""
        if not outside:
            # /tmp IS THIS SHELL'S PRIVATE SCRATCH, AND THE FILE TOOLS CANNOT REACH IT.
            # Measured: a shell made a folder under /tmp, write_file then refused
            # every path in it, and the run lost three to four minutes before
            # writing inside its working directory. Said at the command that
            # creates or writes there, in both arms.
            try:
                made = re.findall(r"(?:\bmkdir|\btouch|\bcp|\bmv|\btee|>>?)\s+(?:-\S+\s+)*(/tmp/[\w./+-]*)",
                                  command)
            except Exception:                          # noqa: BLE001
                made = []
            if not made:
                return ""
            return ("\n[" + made[0] + " is this shell's private scratch: it lasts between your shell "
                    "commands, but read_file, write_file and edit_file cannot reach it and nothing "
                    f"there is part of your result. Write scripts and every deliverable inside "
                    f"your working directory {workdir}; use a relative path.]")
        return ("\n[" + ", ".join(outside[:3]) + " is outside your working "
                f"directory {workdir}, which is read-only from here. All files "
                "\u2014 scripts, logs, and every required deliverable \u2014 must "
                "be written inside it; use a relative path.]")

    def _result_mtime() -> float | None:
        try:
            rt = next(iter(sorted(workdir.rglob("RESULT.txt"))), None)
            return rt.stat().st_mtime if rt else None
        except Exception:                              # noqa: BLE001
            return None

    @tool
    def run_bash(command: str) -> str:
        """Run a shell command inside the cell's sandbox dir. Returns stdout+stderr (truncated to 12 KB)."""
        note_action()
        _before = _result_mtime()
        _before_art = _artefact_mtimes()
        _before_scr = _script_mtimes()
        _started_at = time.time()
        # THE WHOLE PROCESS GROUP DIES ON TIMEOUT, NOT JUST THE SHELL.
        #
        # This was subprocess.run(..., timeout=900). On timeout Python kills
        # its direct child — which is `bash` — and every GRANDCHILD survives,
        # reparented to init. A solver launched from that shell then runs
        # forever: three DUNE processes from round-1 timeouts were found
        # 23-24 hours later, still at 100% CPU, having burned 71 CPU-hours
        # between them. They also contend for the machine with whatever runs
        # next, which silently slows and can time out later cells.
        #
        # start_new_session puts the shell and everything it spawns in one
        # process group; on timeout that group is signalled, TERM then KILL.
        proc = None
        try:
            proc = subprocess.Popen(
                _sandboxed_bash_argv(workdir, command, isolate=isolate),
                cwd=workdir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=_clean_subprocess_env(),
                start_new_session=True,
            )
            out_s, err_s = proc.communicate(timeout=900)
            out = (out_s or "") + (("\n[stderr]\n" + err_s) if err_s else "")
            out = out[-12000:] if len(out) > 12000 else out
            # A submission written by heredoc is still a submission.
            # THE REGISTRY CHECK READS THE OUTPUT, NOT THE DISK: the run that
            # this check exists for left no attribute-constructor call in any
            # file, only a traceback it misread.
            return (out
                    + (_registry_error_check(out) + _eaten_error_check(out)
                       + _env_after_wrapper_check(command)
                       + _fourc_run_check(command, out, workdir)
                       + _participant_run_check(out, command)
                       + _participant_command_check(command, workdir, out)
                       + _fourc_after_shell_check(workdir, _started_at, command)
                       if audit_on_submit else "")
                    + _script_check_after_shell(_before_scr)
                    + _artefact_check_after_shell(_before_art)
                    + _audit_after_shell(_before)
                    + _outside_path_note(command) + _note())
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            return ("[timeout after 900s; the command and everything it "
                    "spawned were terminated]" + _note())
        except (OSError, UnicodeError, ValueError) as e:
            _kill_group(proc)
            return f"[command failed to launch: {type(e).__name__}: {e}]"
    return run_bash


def _kill_group(proc) -> None:
    """TERM then KILL the process group `proc` leads. Never raises."""
    if proc is None or proc.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def _keep_version(keep: Path | None, workdir: Path, p: Path, content: str) -> None:
    """Keep a full copy of what was written, outside the agent's folder.

    THE TRANSCRIPT CUTS A WRITTEN FILE AT 600 CHARS, so a check that fired on a
    write cannot be judged after the run: across five runs four firings of one
    check were recorded and three of them could not be read back, because each
    fix overwrote the version the check had seen. Both arms, every write
    (edit_file writes through write_file); never raises."""
    if keep is None:
        return
    try:
        keep.mkdir(parents=True, exist_ok=True)
        n = sum(1 for _ in keep.iterdir())
        rel = str(p.resolve().relative_to(workdir.resolve())).replace("/", "__")
        (keep / f"{n:04d}__{rel}").write_text(content)
    except Exception:                                   # noqa: BLE001
        pass


def _read_write_tools_for(workdir: Path, *, audit_on_submit: bool = False,
                          advice: bool = False, budget_note: bool = True,
                          keep_versions: Path | None = None):
    # THE CLOCK RIDES ON EVERY TOOL REPLY, NOT ONLY ON THE SHELL'S. The wall
    # clock stamp and the action count were appended to run_bash replies alone,
    # so a run that spent its minutes in write_file calls -- measured: seven
    # full re-writes of a 47k-character participant, 25 of 45 minutes -- read
    # "actions spent: 9" with 29 minutes gone, and died at the wall with a
    # converged level-1 result set on disk and nothing handed in. Same stamp,
    # same text, both arms; the product path passes budget_note=False as it
    # does for the shell tool.
    @tool
    def read_file(path: str, max_bytes: int = 200_000) -> str:
        """Read a file inside the cell sandbox."""
        try:
            p = Path(path)
            if not p.is_absolute():
                p = workdir / p
            wd = workdir.resolve()
            resolved = p.resolve()
            if not resolved.is_relative_to(wd):
                return (f"[read refused: {p} is outside your working "
                        f"directory {wd}]")
            data = resolved.read_bytes()[:max_bytes]
            return data.decode("utf-8", errors="replace")
        except FileNotFoundError:
            return f"[file not found: {path}]"
        except (OSError, ValueError) as e:
            return f"[read failed: {type(e).__name__}: {e}]"

    @tool
    def write_file(path: str, content: str) -> str:
        """Write `content` to `path` (relative paths resolve inside the cell sandbox)."""
        if budget_note:
            note_action()
        try:
            p = Path(path)
            if not p.is_absolute():
                p = workdir / p
            wd = workdir.resolve()
            if not p.resolve().is_relative_to(wd):
                return (f"[write refused: {p} is outside your working "
                        f"directory {wd}. All files — scripts, logs, and "
                        f"every required deliverable — must be written "
                        f"inside it; use a relative path.]")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            _keep_version(keep_versions, workdir, p, content)
            reply = f"wrote {len(content)} chars to {p}"
            # ROUND-8 MECHANISM, openPASO ARM ONLY: the submission is audited the
            # moment it is written, and the findings are placed in the reply
            # the agent is already reading. Round 7 measured why voluntary
            # does not work: a calibrated audit tool plus the instruction to
            # run it, sitting in the one channel read by every run, was used
            # by 1 of 51 agents — while the tool's checks catch 15 of the 18
            # submitted-and-wrong runs of earlier rounds. Participation is now
            # by default. The bare arm is untouched: this flag is set only by
            # build_mcp_agent, because the audit is openPASO's capability.
            # FIRE THE COUPLED CHECKS ON THE ARTEFACT WRITE, not only on
            # submission. See _early_artefact_check for the mtime measurement
            # that forced this.
            if audit_on_submit and p.name != "RESULT.txt":
                # THE CAMPAIGN'S DELIVERABLE FAMILY: these read a grading
                # contract (level indices, run-log naming, discarded proof)
                # that a product user does not have.
                reply += _level_index_check(workdir, p)
                reply += _identical_levels_check(workdir, p)
                reply += _wrong_level_run_log_check(workdir, p)
                reply += _discarded_proof_check(p, content)
            if (audit_on_submit or advice) and p.name != "RESULT.txt":
                # THE LINT FAMILY: product capability. Every one of these reads
                # the script itself and names something known to stop the run
                # on this install, or a number that was invented.
                reply += _constant_deliverable_check(p, content)
                reply += _script_noop_check(p, content)
                reply += _registry_attribute_check(p, content)
                reply += _extra_script_checks(p, content)
                reply += _fourc_deck_write_check(p, content)
                reply += _participant_write_check(p, content)
                reply += _config_write_check(p, content)
                reply += _early_artefact_check(workdir, p)
            if audit_on_submit and p.name == "RESULT.txt":
                # A GIVE-UP FILED OVER FINISHED WORK, caught structurally.
                # Checked before the numeric audit because it is the more
                # basic error: the audit asks whether the numbers are
                # self-consistent, this asks whether numbers were submitted
                # at all when they existed.
                if "COULD_NOT_COMPLETE" in content.upper():
                    try:
                        _contra = _work_on_disk_contradicting_a_give_up(workdir)
                    except Exception:                # noqa: BLE001
                        _contra = ""
                    if _contra:
                        reply += "\n\n" + _contra
                try:
                    findings = _audit_submission(p, content)
                except Exception as e:               # noqa: BLE001
                    findings = None
                    reply += f"\n[auto-audit unavailable: {type(e).__name__}]"
                # ORDER MATTERS: "NOEVIDENCE" IS A TRUTHY STRING.
                #
                # This chain tested `if findings:` first, so the sentinel took
                # the findings branch and the agent received the bare token
                # NOEVIDENCE under the heading "AUTO-AUDIT of your submission",
                # followed by "check the named place, fix if real". The
                # paragraph below — written precisely because the old code told
                # 134 of 137 HONEST_INCOMPLETE runs "clean" — could never
                # print. HONEST_INCOMPLETE is the largest single bucket in the
                # campaign at 159 rows, so the message that never printed was
                # the one aimed at the most common outcome.
                # NEVER an all-clear on nothing, and never two copies of the
                # wording: _format_audit_reply is shared with the shell route.
                reply += _format_audit_reply(findings)
            if budget_note:
                reply += _time_left_note()
            return reply
        except (OSError, UnicodeError, ValueError) as e:
            return (f"[write failed: {type(e).__name__}: {e} — "
                    "use a relative path inside the sandbox]")

    @tool
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace ONE exact occurrence of `old` in `path` with `new` (relative paths resolve inside the cell sandbox). Use it to fill a hole or fix a line in a large file instead of re-typing the whole file."""
        # AN EDIT IS NOT A RE-TYPE. Measured over fifteen coupled cells of one
        # problem: 2,029,176 characters re-emitted into the participant
        # scripts, a mean of 135k per cell -- three full copies of a 36k
        # served contract before any physics -- because every correction had
        # to re-send the whole file. The one correct cell wrote its file once.
        # This tool changes one exact passage and hands the result to
        # write_file, so the same checks, the same refusals and the same
        # clock stamp apply; it counts as one action, not two.
        try:
            p = Path(path)
            if not p.is_absolute():
                p = workdir / p
            wd = workdir.resolve()
            if not p.resolve().is_relative_to(wd):
                return (f"[edit refused: {p} is outside your working directory {wd}]")
            if not p.is_file():
                return f"[edit refused: {p} does not exist; write_file creates a file]"
            text = p.read_text()
        except (OSError, UnicodeError, ValueError) as e:
            return f"[edit failed: {type(e).__name__}: {e}]"
        if not old:
            return "[edit refused: `old` is empty; give the exact text to replace]"
        n = text.count(old)
        if n == 0:
            return (f"[edit refused: the text to replace was not found in {p.name} "
                    f"({len(text)} chars). read_file it and copy the exact passage, "
                    f"whitespace included.]")
        if n > 1:
            return (f"[edit refused: the text occurs {n} times in {p.name}; include "
                    f"more surrounding lines so it occurs once.]")
        content = text.replace(old, new, 1)
        reply = write_file.invoke({"path": str(p), "content": content})
        return (f"edited {p.name}: replaced {len(old)} chars with {len(new)} "
                f"({len(content)} chars now). " + reply)

    return [read_file, write_file, edit_file]


# What has already been asked in this process. Bounded: the web interface keeps
# a server up for days, and an unbounded dict keyed by whatever anyone searched
# for is a slow leak. Oldest out first; 256 is far more than one run asks.
_BLOCKED_MESSAGE = (
    "[the search returned nothing after three attempts on all backends. "
    "A search engine answers an empty list when it is throttling a machine, which is "
    "the usual reason for this, so treat it as 'could not search', NOT as 'the "
    "web has nothing on this'. Do not conclude anything from it: wait and try "
    "once more, ask a shorter query, or use openPASO's own knowledge and "
    "examples tools.]")

_SEARCH_CACHE: dict[tuple[str, str, int], str] = {}
_SEARCH_CACHE_MAX = 256
_SEARCH_BLOCKED: dict[tuple[str, str, int], float] = {}
_SEARCH_BLOCKED_FOR = 60.0        # seconds a refusal is remembered

# Whose search this is. One command-line run is one process, but the web
# interface serves many runs for days from a single one: without this, a run
# could be handed snippets another run fetched, and its transcript would show
# results it never asked for. The interface sets it per run; anything that does
# not is one scope, exactly as before.
SEARCH_SCOPE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "openpaso_search_scope", default="")


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web. Returns up to max_results result snippets.

    DuckDuckGo throttles repeated searches from one machine, and when it does it
    answers with an EMPTY LIST rather than an error. The old code read that as
    "no results" and said so: an agent was told the web knows nothing about the
    Schaefer-Turek benchmark, and went on to work from memory. Measured on a
    throttled machine, five searches in a row returned nothing on all three
    backends while the same queries returned hits seconds later.

    So: each backend is tried more than once with a pause between attempts, an
    answer already fetched in this process is reused rather than asked for
    again, and an empty answer is reported as what it almost always is — a
    block, not an empty web — so nobody mistakes it for evidence of absence.
    """
    # THE MAINTAINED CLIENT FIRST. duckduckgo_search was renamed ddgs and the old
    # name asks DuckDuckGo alone; ddgs asks several engines. Measured 2026-09-28 on
    # this machine: duckduckgo_search 8.1.1 answered 0 hits for "Schafer Turek
    # benchmark cylinder" on all three of its backends, and furniture-shop and
    # language-course pages for two other queries of one web-interface run, while
    # ddgs 9.16 answered all four queries with the benchmark paper, the FEniCSx
    # tutorial and the dolfinx forum.
    try:
        from ddgs import DDGS
        backends = ("auto",)               # ddgs picks among its engines itself
    except ImportError:
        try:
            from duckduckgo_search import DDGS   # the old name: DuckDuckGo only
            backends = ("auto", "html", "lite")
        except ImportError:
            return ("[web_search unavailable: install the search client to enable it — "
                    "`pip install ddgs` (see langgraph_eval/requirements-langgraph.txt)]")

    # the query that is remembered is the query that is sent: keying on a
    # lowercased form while searching the original would let one spelling
    # answer for another, and a search engine's results are not case-blind
    query = query.strip()
    key = (SEARCH_SCOPE.get(""), query, max_results)
    if key in _SEARCH_CACHE:
        return _SEARCH_CACHE[key]
    # A query that was just refused is refused again: retrying it immediately
    # costs nine requests and five seconds of pauses, and repetition is what
    # causes the throttling in the first place. Remembered briefly, so a
    # provider that recovers is not locked out.
    blocked_at = _SEARCH_BLOCKED.get(key)
    if blocked_at is not None and time.time() - blocked_at < _SEARCH_BLOCKED_FOR:
        return _BLOCKED_MESSAGE

    last_err = None
    empties = 0                      # answers that came back with nothing in them
    for pause in (0.0, 1.5, 4.0):
        if pause:
            time.sleep(pause)
        for backend in backends:
            try:
                with DDGS() as ddgs:
                    hits = list(ddgs.text(query, max_results=max_results,
                                          backend=backend))
                if hits:
                    out = "\n\n".join(
                        f"{h.get('title')}\n{h.get('href')}\n{h.get('body')}"
                        for h in hits)
                    if len(_SEARCH_CACHE) >= _SEARCH_CACHE_MAX:
                        _SEARCH_CACHE.pop(next(iter(_SEARCH_CACHE)))
                    _SEARCH_CACHE[key] = out
                    return out
                empties += 1
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                continue

    if empties == 0 and last_err:
        # Nothing ever answered: a broken connection or a refused request, not
        # a provider saying "nothing". Reported as itself and NOT remembered as
        # a refusal, because the next attempt may get through.
        #
        # Decided on whether any answer came back at all, not on whether an
        # error was seen anywhere: one transient error followed by empty
        # answers IS the throttle, and reading the presence of an error as
        # "unreachable" named the wrong failure and skipped the memory that
        # keeps a stuck run from paying nine requests again.
        return (f"[the search could not be made: {last_err}. This is a failure to reach the "
                "search provider, NOT an answer about the web. Do not conclude anything from "
                "it: try again, or use openPASO's own knowledge and examples tools.]")
    _SEARCH_BLOCKED[key] = time.time()
    if len(_SEARCH_BLOCKED) > _SEARCH_CACHE_MAX:
        _SEARCH_BLOCKED.pop(next(iter(_SEARCH_BLOCKED)))
    if last_err:                     # some answered empty, one could not be reached
        return _BLOCKED_MESSAGE + f" (one attempt also failed: {last_err})"
    return _BLOCKED_MESSAGE


# ────────────────────────────────────────────────────────────────────
# spawn_subagent — sibling LangGraph agent on the same vLLM endpoint
# ────────────────────────────────────────────────────────────────────
def _make_spawn_subagent_tool(
    *, size: str, seed: int, workdir: Path,
    parent_tools: Sequence[BaseTool], depth: int,
):
    """Returns a tool that spawns a depth-limited sibling agent.

    The sub-agent reuses the same vLLM server (cheap on memory) with a
    slightly higher temperature and a derived seed. It gets the same
    workdir-bound bash/read/write/web_search and the parent's openPASO
    tools, but its own ``spawn_subagent`` is *not* re-installed beyond
    depth 1 to prevent runaway recursion.
    """

    @tool
    async def spawn_subagent(role: str, task: str, context: str = "") -> str:
        """Spawn a sub-agent. role∈{critic, researcher, verifier, worker}; task = what it should do; context = facts to pass in.

        The critic role should ruthlessly challenge the parent's setup; the
        verifier should re-derive numbers independently; the researcher
        should look things up via web_search and the openPASO knowledge tool.
        Returns the sub-agent's final message text.
        """
        # ASYNC, AND ainvoke BELOW, BECAUSE THE MCP TOOLS ARE ASYNC-ONLY.
        # This was a sync `def` calling `sub_agent.invoke`. The sub-agent
        # inherits the parent's openPASO tools, which langchain_mcp_adapters
        # returns as coroutine-only StructuredTools, so the first time a
        # critic reached for `knowledge` or `discover` it raised
        # "NotImplementedError: StructuredTool does not support sync
        # invocation" — caught by the except below and returned to the model
        # as a string, so it looked like a critic verdict rather than a dead
        # mechanism. Round 1: this fired in 14 of 14 coupled and 16 of 18
        # single-code openPASO runs, i.e. the MANDATORY critic the server
        # instructions demand never ran once in the entire campaign.
        if depth >= 2:
            return "[spawn_subagent denied: max depth 2 to prevent recursion]"
        sub_tools = list(parent_tools)
        # Give the sub-agent the same host tools, but no further nesting:
        sub_tools = [t for t in sub_tools if t.name != "spawn_subagent"]
        if role == "critic":
            sys = (
                "You are a ruthlessly critical reviewer. Challenge every "
                "parameter choice, check units, look for sign errors, "
                "verify BCs, and validate against the literature via "
                "web_search. Respond with one of: APPROVED: <reason> | "
                "REJECTED: <issue and required fix>."
            )
        elif role == "verifier":
            sys = (
                "You are an independent verifier. Re-derive the requested "
                "quantity from first principles or by an alternative "
                "method/solver, then compare with the parent's number."
            )
        elif role == "worker":
            # A bounded step worker: the same tools, one step, one check.
            # Plumbing only -- the step and its check come from the parent's
            # task text (which openPASO's ladder writes); nothing here knows
            # any task.
            sys = (
                "You are the worker for exactly ONE step of a larger job. Do "
                "only what the task text asks, in the working directory it "
                "names, using the tools; the step ends when the CHECK stated "
                "in the task passes on disk. Report either DONE with the "
                "files you produced, or the exact error text you could not "
                "get past. Do not judge or attempt the rest of the job."
            )
        else:
            sys = (
                "You are a research assistant. Look up authoritative "
                "sources for the requested information and summarise."
            )
        sub_llm = _llm(size, temperature=0.3, seed=seed + 100 + depth)
        sub_agent = create_react_agent(
            sub_llm, tools=sub_tools, prompt=sys,
        )
        msg = f"Task: {task}\n\nContext provided by parent:\n{context}"
        # THE SUBMISSION'S WRITE CHECK REACHES THE PARENT EVEN WHEN A WORKER WROTE IT. Measured
        # (round 46, C2 7211): a worker wrote RESULT.txt, the audit named the missing level-1
        # run logs in the worker's write reply, the worker's report to the parent carried none
        # of it, and the parent stopped at 21 min left. Plumbing only: the finding text is the
        # same openPASO audit the write hook calls.
        _rt = None
        try:
            _rt = next(iter(sorted(workdir.rglob("RESULT.txt"))), None)
        except Exception:                              # noqa: BLE001
            _rt = None
        _rt_before = _rt.stat().st_mtime if _rt is not None and _rt.exists() else None
        _t0 = time.time()
        try:
            out = await sub_agent.ainvoke(
                {"messages": [("user", msg)]},
                config={"recursion_limit": WORKER_RECURSION_LIMIT},
            )
            report = out["messages"][-1].content
            # THE STEP CAP HAS TWO EXITS AND ONLY ONE WAS COVERED. LangGraph's
            # prebuilt agent returns "Sorry, need more steps to process this
            # request." as an ordinary final message when the recursion limit
            # is hit gracefully; only the raising exit reached the report
            # below. Measured: a worker died at the cap with a 45 kB filled
            # participant on disk, the parent read the bare sentence,
            # overwrote that file with the pristine contract and gave up with
            # 24 minutes left. Both exits now hand the parent the same report.
            report = _report_if_worker_out_of_steps(report, workdir, _t0, WORKER_RECURSION_LIMIT)
            # A SILENT WORKER THAT WROTE FILES DID NOT "PRODUCE NOTHING": its files are
            # named, newest first, as for a worker out of steps (measured: that sentence
            # was handed back for a worker whose files were on disk).
            if not str(report).strip() and role not in ("critic", "verifier", "researcher"):
                _l = _worker_failure_report(RuntimeError(""), workdir, _t0, WORKER_RECURSION_LIMIT)
                if " Everything it wrote" in _l:
                    report = ("[the sub-agent returned no text; this is NOT approval and NOT a "
                              "verdict on its work." + _l[_l.find(" Everything it wrote"):])
            # SILENCE IS NOT ASSENT, AND IT USED TO BE HANDED BACK AS "".
            #
            # MEASURED on a live run: the model spawned a critic, the critic ran
            # for 2m16s and returned a result of length ZERO, and the model then
            # composed "CRITIC REVIEW ... VERDICT: APPROVED with minor notes.
            # Ready to run.", filed it through submit_critic_review, took the
            # token and ran. An empty string handed to a model is the absence of
            # a complaint, which is what assent looks like -- and a sub-agent
            # produces nothing when it runs out of steps or its tool call died
            # just as readily as when it has nothing to say. The exception path
            # here was already loud; silence was the half that was not, which is
            # the worse half, because a crash is visible and a blank is not.
            #
            # One sentence, shared word for word with the browser interface's
            # handler, so a person reading a transcript and a person reading a
            # campaign trajectory see the same words. (This branch keeps the old
            # product name until the freeze; the rename script maps it.)
            if not str(report).strip():
                who = role if role in ("critic", "verifier", "researcher") else "sub-agent"
                report = SILENT_SUBAGENT_REPORT.format(who=who)
            # A WORKER THAT WROTE DELIVERABLES TOOK EVERY WRITE-TIME FINDING
            # WITH IT. The body lives in openPASO; this only calls it.
            try:
                _dw = _deliverable_findings_after_worker(workdir)
                if _dw:
                    report = str(report) + str(_dw)
            except Exception:                              # noqa: BLE001
                pass
            try:
                _rt2 = next(iter(sorted(workdir.rglob("RESULT.txt"))), None)
                if _rt2 is not None and _rt2.exists() and (_rt_before is None or _rt2.stat().st_mtime > _rt_before):
                    _f = _audit_submission(_rt2, _rt2.read_text(errors="replace"))
                    if _f:
                        report = (str(report) + "\n\n[the worker wrote RESULT.txt; the write check on it reported:]\n"
                                  + str(_f))
            except Exception:                          # noqa: BLE001
                pass
            return _bounded_report(str(report))
        except Exception as e:
            # Still returned as text so one bad sub-agent cannot kill the run,
            # but marked loudly enough that a transcript sweep finds it: a
            # broken mechanism must not read like a verdict -- and the files
            # the worker left are named, because a worker that ran out of
            # steps handed the parent LangGraph's bare "need more steps" and
            # the parent rebuilt by hand, at a quarter of the served
            # contract's fidelity, what was sitting finished on disk.
            return _worker_failure_report(e, workdir, _t0, WORKER_RECURSION_LIMIT)

    return spawn_subagent


def _report_if_worker_out_of_steps(report, workdir: Path, started_at: float, limit: int):
    """LangGraph's graceful step-cap message, turned into the same out-of-steps
    report the raising exit produces; any other report passes through."""
    text = str(report or "")
    if "need more steps" in text.lower() and len(text) < 400:
        return _worker_failure_report(RuntimeError(text), workdir, started_at, limit)
    return report


def _worker_failure_report(exc: BaseException, workdir: Path, started_at: float,
                           limit: int) -> str:
    """What the parent reads when a worker dies: not a verdict, and the files
    the worker left on disk, newest first, so the work continues from them."""
    files = []
    try:
        for q in Path(workdir).rglob("*"):
            try:
                if (q.is_file() and q.stat().st_mtime >= started_at
                        and "trajectory" not in q.name):
                    files.append(q)
            except OSError:
                continue
    except OSError:
        files = []
    files.sort(key=lambda q: q.stat().st_mtime, reverse=True)
    shown = ", ".join(f"{q.relative_to(workdir)} ({q.stat().st_size} B)" for q in files[:12])
    out_of_steps = ("recursion" in type(exc).__name__.lower()
                    or "need more steps" in str(exc).lower())
    if out_of_steps:
        head = (f"[WORKER OUT OF STEPS: the harness allows a worker {limit} graph "
                f"steps (about {limit // 2} tool calls) and this one used them all. "
                f"This is NOT a verdict on its work.")
    else:
        head = (f"[SUBAGENT FAILED — this is NOT a review verdict — "
                f"{type(exc).__name__}: {str(exc)[:300]}")
    if files:
        tail = (f" Everything it wrote is on disk, newest first: {shown}. Continue "
                f"FROM those files -- read them and change them in place -- rather "
                f"than rebuilding them; a new worker can be handed the file names "
                f"and the one thing left to do.]")
    else:
        tail = " It left no new file on disk.]"
    return head + tail


# ────────────────────────────────────────────────────────────────────
# openPASO MCP tool loader (langchain-mcp-adapters)
# ────────────────────────────────────────────────────────────────────
def _openpaso_mcp_client(workdir: Path | None = None, *,
                         isolate: bool = True):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    env = _clean_subprocess_env()
    # SAY WHERE THE JOURNAL GOES; DO NOT INFER IT FROM A MOUNT. The server used
    # to write into <source>/data/sessions and the jail bound the cell's own
    # directory over that path. That works only while the default happens to be
    # the install tree -- which is the defect openPASO just fixed -- so the
    # campaign now names the destination, and the bind below is belt to its
    # braces. `participant_lint` reads this same directory.
    # `workdir` is optional here -- the product path calls this with none, and
    # there the product's own default (a state directory outside the install)
    # is the right answer.
    if workdir is not None:
        env["OPENPASO_JOURNAL_LIVE_DIR"] = str(Path(workdir) / ".openpaso_sessions")
    env.pop("OFA_DISABLE_CRITIC", None)
    env.pop("OFA_DISABLE_PITFALLS", None)
    env["FOURC_ROOT"] = env.get("FOURC_ROOT", str(Path.home() / "4C"))
    env["FOURC_BINARY"] = env.get(
        "FOURC_BINARY", str(Path.home() / "4C/build/4C"))
    # LD_LIBRARY_PATH CARRIES MORE THAN ONE SOLVER, AND `get(default)` DROPS
    # THE REST.
    #
    # This was env.get("LD_LIBRARY_PATH", "/opt/4C-dependencies/lib"): if the
    # outer environment had the variable set to anything at all, 4C's
    # dependency path was silently discarded, and preCICE's was never added
    # under any circumstance. Measured on this host: `import precice` fails
    # with "libprecice.so.3: cannot open shared object file" although
    # /opt/precice/lib/libprecice.so.3 -> libprecice.so.3.1.2 is present and
    # the Python binding is installed, and it succeeds the moment
    # /opt/precice/lib is on the path (returns 3.1.2;v3.1.2). 103 of the C2
    # run directories mention preCICE, so agents do reach for that path and
    # it could not load for any of them.
    #
    # Composed rather than defaulted: every required entry, then whatever was
    # inherited, order preserved, duplicates dropped, and entries that do not
    # exist on this host left out so a stale path cannot mask a real one.
    _lib_dirs = ["/opt/4C-dependencies/lib", "/opt/precice/lib"]
    _seen, _parts = set(), []
    for _d in _lib_dirs + [
            x for x in env.get("LD_LIBRARY_PATH", "").split(":") if x]:
        if _d in _seen:
            continue
        _seen.add(_d)
        if Path(_d).is_dir():
            _parts.append(_d)
    env["LD_LIBRARY_PATH"] = ":".join(_parts)
    _pin_backend_runtime_env(env)
    source_repo = Path(os.environ.get(
        "OPENPASO_SOURCE_SNAPSHOT", str(REPO))).resolve()
    env["PYTHONPATH"] = str(source_repo / "src")
    # THE TOOLS MUST WRITE INTO THIS CELL'S SANDBOX.
    #
    # run_simulation and friends wrote to <repo>/simulation_outputs, a single
    # directory shared by every caller. An agent that followed the documented
    # openPASO workflow therefore produced its solution files where nothing
    # downstream looks, and where another cell could overwrite or read them.
    # It is the openPASO-arm tools that do this, so the cost fell entirely on the
    # arm under test: 8 of 14 coupled openPASO runs in round 1 went through it.
    if workdir is not None:
        cell_work = Path(workdir).resolve()
        env["OPENPASO_CELL_WORKDIR"] = str(cell_work)
        env["OPENPASO_OUTPUT_DIR"] = str(cell_work / "simulation_outputs")
        env["OPENPASO_COUPLING_DIR"] = str(cell_work / "coupling")
        env["OPENPASO_MESH_DIR"] = str(cell_work / "meshes")
        env["OPENPASO_BENCHMARK_DIR"] = str(cell_work / "benchmark_results")
        # ONE JIT CACHE PER CELL, ON BOTH ROUTES. The shell sandbox already
        # binds this cell's private scratch at /tmp and points XDG_CACHE_HOME
        # into it; the participants the server launches (couple, couple_levels)
        # inherited the SERVER's environment and compiled into the host's
        # shared ~/.cache, so five concurrent cells contended for one FEniCSx
        # form cache. Measured: "JIT compilation timed out" cost one cell its
        # first give-up. The same host directory serves both routes now.
        env["XDG_CACHE_HOME"] = str(sandbox_scratch_for(cell_work) / ".cache")

    # THE SERVER INTERPRETER, RESOLVED — NOT ASSUMED.
    #
    # This was REPO/".venv/bin/python". A git worktree has no .venv, so on any
    # worktree checkout every MCP-arm run crashed at agent construction with
    # FileNotFoundError before the model was ever called — while the BARE arm
    # ran fine. A fleet launched that way silently becomes one-armed, which is
    # the worst possible shape for an A/B campaign. Third instance of this
    # exact defect class (backend_imports.json, the fixture runner) — same
    # fix: explicit env var first, then the repo venv, then the primary
    # checkout's venv, and REFUSE loudly rather than launch a crippled arm.
    _cands = [os.environ.get("OPENPASO_PYTHON"),
              str(REPO / ".venv/bin/python")]
    _server_py = next((c for c in _cands if c and Path(c).is_file()), None)
    if _server_py is None:
        raise RuntimeError(
            "no interpreter found for the openPASO MCP server; set OPENPASO_PYTHON. "
            "Refusing to build a silently crippled MCP arm.")
    command = _server_py
    args = ["-m", "server"]
    cwd = source_repo / "src"
    if workdir is not None and isolate:
        wrapped = _sandboxed_process_argv(
            Path(workdir), [command, *args], source_repo=source_repo)
        command, args = wrapped[0], wrapped[1:]
        cwd = Path(workdir)
        env["PYTHONPATH"] = "/tmp/openpaso-source/src"
    elif workdir is not None:
        # PRODUCT PATH: the server runs unwrapped, so it reads the repo where
        # it actually is rather than through the sandbox's /tmp bind, and the
        # user's own solver installs stay visible wherever they live.
        cwd = Path(workdir)
        env["PYTHONPATH"] = str(source_repo / "src")
    client = MultiServerMCPClient({
        "openpaso": {
            "command": command,
            "args": args,
            "cwd": str(cwd),
            "env": env,
            "transport": "stdio",
        }
    })
    return client


def _load_openpaso_mcp_tools(workdir: Path | None = None) -> list[BaseTool]:
    """List tools for discovery-only callers.

    Agent runs must use :func:`openpaso_mcp_tools_session`; tools returned here
    create a new server for every call and therefore cannot carry server-side
    state such as critic reviews.
    """
    client = _openpaso_mcp_client(workdir)
    return asyncio.run(client.get_tools())


@asynccontextmanager
async def openpaso_mcp_tools_session(workdir: Path | None = None, *,
                                     surface: str = "campaign",
                                     isolate: bool = True):
    """Yield tools bound to one openPASO process for an entire agent run.

    ``surface="campaign"`` yields exactly CAMPAIGN_MCP_TOOL_ALLOWLIST and
    refuses if the server cannot supply all of it, because a measurement is
    only comparable against a fixed tool contract.

    ``surface="all"`` yields every tool the server registers. That is what an
    ordinary user of the product should get -- setup_backend and the catalog
    tools are useful to a person and are simply not part of the experiment's
    fixed surface. Nothing but the tool list differs between the two.
    """
    from langchain_mcp_adapters.tools import load_mcp_tools

    if surface not in ("campaign", "all"):
        raise ValueError(f"surface must be 'campaign' or 'all', not {surface!r}")

    client = _openpaso_mcp_client(workdir, isolate=isolate)
    async with client.session("openpaso") as session:
        available = await load_mcp_tools(session, server_name="openpaso")
        if surface == "all":
            yield available
            return
        names = {tool.name for tool in available}
        missing = CAMPAIGN_MCP_TOOL_ALLOWLIST - names
        if missing:
            raise RuntimeError(
                "openPASO campaign tool contract is incomplete; missing: "
                + ", ".join(sorted(missing)))
        yield [tool for tool in available
               if tool.name in CAMPAIGN_MCP_TOOL_ALLOWLIST]


# ────────────────────────────────────────────────────────────────────
# Public factories
# ────────────────────────────────────────────────────────────────────


# sys.path is extended ONCE here, not on every RESULT.txt write —
# the per-call insert accumulated 27 duplicate entries in 25 writes
# and kept putting openPASO's src ahead of the venv for every import.
import sys as _sys_for_path
_sys_for_path.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src"))


# ─────────────────────────────────────────────────────────────────────
# THE WORKSPACE CHECKS ARE openPASO's, NOT THE HARNESS'S.
#
# They were accreting here, one plausible check at a time, until the harness
# was a second knowledge system keyed to this campaign's contract -- which
# breaks the attribution (an uplift produced by the runner is not openPASO's),
# cannot ship, and is invisible to a fresh draw. Boundary set explicitly on
# 2026-09-03: the harness owns plumbing and hook POINTS only; every check
# body lives in openPASO (tools/workspace_advisor.py, sibling of
# tools/result_audit.py) and is imported here like the audit already was.
# Two pacing checks that coached the agent about its clock rather than
# verifying anything were deleted outright, not moved.
from tools.workspace_advisor import (          # noqa: E402
    _discarded_proof_check, _early_artefact_check, _eaten_error_check,
    _constant_deliverable_check, _env_after_wrapper_check,
    _extra_script_checks, _flat,
    _identical_levels_check, _level_index_check, _registry_attribute_check,
    _looks_like_captured_output, _registry_error_check, _script_noop_check,
    _work_on_disk_contradicting_a_give_up,
    deliverable_findings_after_worker as _deliverable_findings_after_worker,
    _wrong_level_run_log_check, _fourc_deck_write_check, _fourc_run_check,
    _fourc_after_shell_check, _participant_write_check,
    _config_write_check, _participant_run_check,
    _participant_command_check)



def _audit_submission(result_path: Path, content: str):
    """Run the openPASO result audit in-process on the submission's directory.

    Returns a findings string, "" for clean, and raises only when the audit
    module itself is unavailable. claimed order is parsed from the submission
    text so ORDER MISMATCH can fire; absent an order claim, the zero-field,
    floor and non-monotone checks still run.
    """
    import re as _re
    from tools.result_audit import audit as _audit
    # LAST match, line-anchored, sign/exponent allowed, and labels that only
    # LOOK like an order excluded. The old regex took the FIRST match of a
    # loose pattern: "ORDER_OF_MAGNITUDE = 5" became a claim of order 5, and
    # an ELEMENT_ORDER line ahead of the real one won. It also missed
    # ORDER_L2 (a digit ends [A-Z_]*), lowercase, and negatives.
    claimed = None
    for mm in _re.finditer(
            r"^\s*(?!.*OF_MAGNITUDE)([A-Za-z_0-9]*ORDER[A-Za-z_0-9]*)\s*=\s*"
            r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$",
            content, _re.M):
        lab = mm.group(1).upper()
        if any(k in lab for k in ("ELEMENT", "POLYNOMIAL", "DEGREE", "MESH")):
            continue                      # describes the discretisation
        try:
            claimed = float(mm.group(2))
        except ValueError:
            claimed = None
    r = _audit(str(result_path.parent), claimed_order=claimed,
               summary_path=str(result_path))
    if r.get("sequences_found", 0) == 0 and r.get("clean"):
        return "NOEVIDENCE"
    if r.get("clean"):
        # WHAT DID NOT RUN is printed as the audit states it: a clean audit
        # whose checks were skipped must not read like a fully checked one.
        gaps = r.get("not_run") or []
        if gaps:
            return ("no finding among the checks that ran; these did NOT run, so "
                    "what they check is unverified:\n"
                    + "\n".join(f"  * {g}" for g in gaps))
        return ""
    # LEAD WITH THE SINGLE NEXT FIX, then the full findings. The prioritisation
    # body lives in openPASO (tools/result_audit.what_to_fix_next); the harness
    # only prints what it returns. Advisory only — the gate never blocks.
    lead = r.get("what_to_fix_next") or ""
    body = "\n".join(f"  * {f['sequence']}: {f['finding']}"
                     for f in r.get("findings", []))
    return f"{lead}\n\n{body}" if lead else body

def _host_tools(workdir: Path, *, size: str, seed: int,
                parent_tools: list[BaseTool], depth: int,
                audit_on_submit: bool = False,
                keep_versions: Path | None = None) -> list[BaseTool]:
    tools: list[BaseTool] = []
    tools.append(_bash_tool_for(workdir, audit_on_submit=audit_on_submit))
    tools.extend(_read_write_tools_for(workdir, audit_on_submit=audit_on_submit,
                                       keep_versions=keep_versions))
    tools.append(web_search)
    spawn = _make_spawn_subagent_tool(
        size=size, seed=seed, workdir=workdir,
        parent_tools=parent_tools + tools, depth=depth,
    )
    tools.append(spawn)
    return tools


def build_bare_agent(*, size: str, seed: int, workdir: Path, depth: int = 0,
                     keep_versions: Path | None = None):
    tools = _host_tools(workdir, size=size, seed=seed,
                        parent_tools=[], depth=depth, keep_versions=keep_versions)
    llm = _llm(size, temperature=0.2, seed=seed)
    return create_react_agent(llm, tools=tools, prompt=BARE_SYSTEM)


# FILES WRITTEN DURING AN MCP TOOL CALL GET THE SAME CHECKS AS FILES WRITTEN
# BY THE SHELL. MEASURED, C2_27b_MCP_seed1701: it drove its coupling through
# couple() -- exactly what the must-read asks -- and its participant scripts
# wrote run_level1_B.log (179 bytes of the agent's own prose) DURING the MCP
# call. The discarded-proof check, wired to run_bash and write_file, never saw
# it: the artefacts appeared between hook points. Twenty-first instance of the
# theme, on the newest mechanism. The hook below is plumbing only -- every
# check body stays in openPASO (tools/workspace_advisor).
_MCP_HOOK_ART = ("*_level*.csv", "*_level*.log")


def _mcp_artefact_mtimes(workdir: Path) -> dict:
    out = {}
    for pat in _MCP_HOOK_ART:
        for f in workdir.rglob(pat):
            try:
                out[f] = f.stat().st_mtime
            except OSError:
                pass
    return out


def _post_mcp_artefact_check(workdir: Path, before: dict) -> str:
    """One check per file kind over what the MCP call caused to appear."""
    try:
        import fnmatch
        now = _mcp_artefact_mtimes(workdir)
        touched = [f for f, t in now.items()
                   if before.get(f) is None or t > before[f]]
        if not touched:
            return ""
        blocks, chosen = [], set()
        for pat in _MCP_HOOK_ART:
            same = [f for f in touched if fnmatch.fnmatch(f.name, pat)]
            if not same:
                continue
            newest = max(same, key=lambda f: (now[f], f.name))
            if newest in chosen:
                continue
            chosen.add(newest)
            got = (_level_index_check(workdir, newest)
                   + _identical_levels_check(workdir, newest)
                   + _discarded_proof_check(
                       newest, newest.read_text(errors="replace"))
                   + _early_artefact_check(workdir, newest))
            if got and got not in blocks:
                blocks.append(got)
        return "".join(blocks)
    except Exception:                                  # noqa: BLE001
        return ""


def _wrap_mcp_tool_with_artefact_hook(tool: BaseTool, workdir: Path):
    """Append openPASO's write-time findings to the reply of any MCP call that
    left new deliverable files behind. String replies only; structured
    replies pass through untouched."""
    inner = tool.coroutine
    if inner is None:
        return tool

    async def hooked(*a, **kw):
        before = _mcp_artefact_mtimes(workdir)
        res = await inner(*a, **kw)
        got = _post_mcp_artefact_check(workdir, before)
        if got and isinstance(res, str):
            return res + got
        return res

    try:
        tool.coroutine = hooked
    except Exception:                                  # noqa: BLE001
        return tool                # unwrappable tool shape: leave it alone
    return tool


@asynccontextmanager
async def build_mcp_agent(*, size: str, seed: int, workdir: Path,
                          depth: int = 0, keep_versions: Path | None = None):
    """Yield an MCP agent whose tools share one live openPASO server."""
    async with openpaso_mcp_tools_session(workdir) as mcp_tools:
        mcp_tools = [_wrap_mcp_tool_with_artefact_hook(t, workdir)
                     for t in mcp_tools]
        host = _host_tools(workdir, size=size, seed=seed,
                           parent_tools=mcp_tools, depth=depth,
                           audit_on_submit=True, keep_versions=keep_versions)
        llm = _llm(size, temperature=0.2, seed=seed)
        yield create_react_agent(llm, tools=mcp_tools + host,
                                 prompt=_mcp_system_prompt())


__all__ = ["build_bare_agent", "build_mcp_agent",
           "openpaso_mcp_tools_session", "CAMPAIGN_MCP_TOOL_ALLOWLIST",
           "sandbox_scratch_for", "cleanup_sandbox_scratch"]
