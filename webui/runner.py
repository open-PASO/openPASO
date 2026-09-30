"""Agent runner that streams structured events to a single WebSocket.

Wraps the existing ``langgraph_eval.agent`` factories and surfaces every
intermediate step the user wants to see in the UI:

* user_msg / agent_msg / agent_chunk
* tool_call_pending  → in plan mode, blocks waiting for an approve/reject
                       event from the client before executing
* tool_call_executing / tool_result
* subagent_spawned / subagent_returned (via a callback we wire into
  spawn_subagent so its children are visible)
* token_count after each LLM call
* error / done

The runner is mode-aware: plan mode gates every tool call; accept mode
auto-approves; autonomous is identical to accept but the UI is told not
to interrupt.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from . import config

REPO = config.REPO
sys.path.insert(0, str(REPO / "langgraph_eval"))


# ───────────────────────────────────────────────────────────────────
# Mock LLM (no GPU) — drives a deterministic plan→spawn_critic→done
# chain so the UI is usable without vLLM. Real models go through
# ChatOpenAI as configured in langgraph_eval/agent.py.
#
# Implementation reuses the langgraph_eval/_mock_openai_server.py
# fake OpenAI server (proven by the langgraph_eval smoke tests). We
# start one thread per app process on first mock use; subsequent
# sessions all point at the same loopback endpoint.
# ───────────────────────────────────────────────────────────────────
_MOCK_PORT = 18234
_MOCK_STARTED = False


def _ensure_mock_server():
    global _MOCK_STARTED
    if _MOCK_STARTED:
        return
    import _mock_openai_server as mock_srv  # from langgraph_eval/
    try:
        mock_srv.start(port=_MOCK_PORT)
    except OSError:
        # another openPASO process on this machine already serves the same
        # fake model on this port; use it rather than fail
        import socket
        socket.create_connection(("127.0.0.1", _MOCK_PORT), timeout=3).close()
    _MOCK_STARTED = True


def _mock_chat_model():
    _ensure_mock_server()
    from langchain_openai import ChatOpenAI
    # disable_streaming forces astream_events to go through the
    # non-streaming agenerate path, so the mock server's plain JSON
    # response works (no SSE needed).
    return ChatOpenAI(
        base_url=f"http://127.0.0.1:{_MOCK_PORT}/v1",
        api_key="not-used", model="mock-qwen",
        temperature=0.0, timeout=30,
        disable_streaming=True,
    )


# ───────────────────────────────────────────────────────────────────
# Per-session workdir
# ───────────────────────────────────────────────────────────────────
def _session_workdir(session_id: str) -> Path:
    p = config.SANDBOX_ROOT / f"webui_{session_id}" / "work"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ───────────────────────────────────────────────────────────────────
# Approval gate (plan mode)
# ───────────────────────────────────────────────────────────────────
class ApprovalGate:
    """A future-per-pending-call gate. The UI signals approve/reject via
    ``resolve`` from a separate code path."""

    def __init__(self):
        self._pending: dict[str, asyncio.Future] = {}
        self._early: dict[str, dict] = {}

    def open(self, call_id: str) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        early = self._early.pop(call_id, None)
        if early is not None:
            fut.set_result(early)     # the answer arrived before we asked
        self._pending[call_id] = fut
        return fut

    def resolve(self, call_id: str, approved: bool, reason: str = ""):
        fut = self._pending.pop(call_id, None)
        if fut is None:
            # the step is announced before the gate is open, so a fast client
            # can answer first; hold the decision rather than dropping it and
            # leaving the run waiting for an answer that already came
            self._early[call_id] = {"approved": approved, "reason": reason}
            return
        if not fut.done():
            fut.set_result({"approved": approved, "reason": reason})

    def open_all(self, approved: bool = True, reason: str = "") -> int:
        """Answer every waiting step at once, for a switch to "run without
        asking" while one is waiting."""
        waiting = list(self._pending)
        for call_id in waiting:
            self.resolve(call_id, approved, reason)
        return len(waiting)


# ───────────────────────────────────────────────────────────────────
# Tool wrapping for mode gating + event emission
# ───────────────────────────────────────────────────────────────────
class StepControl:
    """The steps of a run that are executing now, so a person can end one that
    hangs without stopping the whole run."""

    def __init__(self):
        self.tasks: dict[str, asyncio.Future] = {}
        self.ended: set[str] = set()
        self.started: dict[str, float] = {}

    def running(self) -> list[str]:
        return [c for c, t in self.tasks.items() if not t.done()]


def set_search_scope(name: str) -> None:
    """Say whose searches these are.

    One server process serves many runs for days, and the search tool keeps what
    it has already fetched. Without a scope a run could be handed snippets
    another run fetched, and its transcript would show results it never asked
    for. Harmless where the tool has no such scope (an older agent module)."""
    try:
        import agent as la
        scope = getattr(la, "SEARCH_SCOPE", None)
        if scope is not None:
            scope.set(name)
    except Exception:          # the interface must not fail over a cache key
        pass


def _wrap_tool(tool, *, emitter, get_mode, gate, agent_label="agent", take_steers=None,
               steps: StepControl | None = None):
    """Return a copy of ``tool`` whose invoke emits events and (when in
    plan mode) waits for explicit approval before running.

    The original tool's func is preserved; we only intercept invocation.
    """
    from langchain_core.tools import BaseTool

    class Gated(BaseTool):
        name: str = tool.name
        description: str = tool.description
        args_schema: Any = getattr(tool, "args_schema", None)

        async def _arun(self, *args, **kwargs):
            call_id = f"tc_{uuid.uuid4().hex[:8]}"
            await emitter({"type": "tool_call_pending",
                           "call_id": call_id, "tool": tool.name,
                           "args": kwargs, "agent": agent_label})
            mode = get_mode()
            if mode == "plan":
                fut = gate.open(call_id)
                try:
                    decision = await fut
                finally:
                    # a run stopped while this waited would leave the future here
                    gate._pending.pop(call_id, None)
                if not decision["approved"]:
                    await emitter({"type": "tool_call_rejected",
                                   "call_id": call_id,
                                   "reason": decision.get("reason", "")})
                    return f"[rejected by user: {decision.get('reason','')}]"
            await emitter({"type": "tool_call_executing",
                           "call_id": call_id, "tool": tool.name,
                           # the record says who let it run
                           "approved_by": "user" if mode == "plan" else "mode"})
            started = time.monotonic()
            inner = asyncio.ensure_future(
                tool.ainvoke(kwargs) if hasattr(tool, "ainvoke")
                else asyncio.to_thread(tool.invoke, kwargs))
            if steps is not None:
                steps.tasks[call_id] = inner
                steps.started[call_id] = time.time()
            ended_by_user = lambda: steps is not None and call_id in steps.ended  # noqa: E731
            try:
                result = await inner
            except asyncio.CancelledError:
                # ended with "End this step", not the whole run being stopped
                me = asyncio.current_task()
                if not ended_by_user() or (me is not None and me.cancelling()):
                    raise
                result = ""
            except Exception as e:
                if not ended_by_user():
                    await emitter({"type": "tool_error",
                                   "call_id": call_id,
                                   "error": f"{type(e).__name__}: {e}"})
                    raise
                result = f"{type(e).__name__}: {e}"
            finally:
                if steps is not None:
                    steps.tasks.pop(call_id, None)
                    steps.started.pop(call_id, None)
            if ended_by_user():
                steps.ended.discard(call_id)
                secs = time.monotonic() - started
                result = (f"[The user ended this step after {secs:.0f} s. The processes it had "
                          f"started were ended, so it has no usable result.]"
                          + (f"\n\nOutput before it was ended:\n{str(result)[:4000]}" if str(result).strip() else ""))
            # A correction the user sent while this step ran. A ReAct agent reads
            # the tool result next, so that is where it is handed over: at most
            # one step late, and never by interrupting a solver mid-calculation.
            # It is added BEFORE the event is recorded, so the log holds what the
            # model was actually given: rebuilding the conversation from the log
            # after a stop used to drop a correction the model had already read.
            steers = take_steers() if take_steers else []   # main agent: takes them
            if steers:
                note = "\n\n".join(x["text"] for x in steers)
                result = (f"{result}\n\n[MESSAGE FROM THE USER, sent while this step was "
                          f"running. Read it and adjust what you do next:]\n{note}")
            from .outcome import SOLVER_TOOLS, classify_solver_result, shorten
            # The verdict is read from the WHOLE result and carried, because
            # what is recorded is a shortened copy and a shortened copy is not
            # valid JSON. Re-deriving from it fell to a regex that needs a
            # "status" key — which a coupling reply does not have — so a
            # coupling the gate had certified was read as "computed nothing",
            # and a failed one whose first participant looked good was read as
            # verified. The evidence must be judged before it is cut.
            event = {"type": "tool_result", "call_id": call_id,
                     "tool": tool.name, "result": shorten(result)}
            if tool.name in SOLVER_TOOLS:
                from .outcome import verdict_reason
                event["verdict"] = classify_solver_result(str(result))
                # and why, in the solver's own words, read from the whole reply
                # before it is cut: the closing line says this rather than a
                # sentence of ours that fits every case and explains none
                event["why"] = verdict_reason(str(result))
            await emitter(event)
            return result

        def _run(self, *args, **kwargs):
            # asyncio.get_event_loop() is deprecated and, from Python 3.12, raises
            # "There is no current event loop" once anything in the process has
            # already run asyncio.run() and closed the loop behind itself. The
            # failure therefore depends on what ran before, which is the worst
            # kind. asyncio.run() makes and closes its own loop, and refuses just
            # the same as the old code did if one is already running.
            return asyncio.run(self._arun(*args, **kwargs))

    return Gated()


# ───────────────────────────────────────────────────────────────────
# Agent factories with WebUI hooks
# ───────────────────────────────────────────────────────────────────
def build_agent_for_session(*, model: str, mcp_on: bool,
                            workdir: Path,
                            emitter,
                            get_mode,
                            gate: ApprovalGate,
                            checkpointer=None,
                            take_steers=None,
                            steps: StepControl | None = None,
                            _mcp_tools=None):
    """Build a LangGraph ReAct agent with all WebUI hooks wired in.

    * ``model`` is a key from ``config.MODELS``. ``mock`` skips vLLM.
    * ``mcp_on`` attaches openPASO via langchain-mcp-adapters when True.
    * ``emitter`` is an async function ``(event_dict) -> None`` used to
      stream events back over the WebSocket.
    * ``get_mode`` is a callable returning the current mode string.
    * ``gate`` is the per-session ApprovalGate.
    """
    import agent as la

    # ── LLM
    if model == "mock":
        llm = _mock_chat_model()
    elif model in config.OPENROUTER_MODELS:
        key = config.openrouter_key()
        if not key:
            raise ValueError(
                "No OpenRouter key. Copy .env.example to .env and paste your "
                "key after OPENROUTER_API_KEY=, then pick this model again.")
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            base_url=config.OPENROUTER_URL, api_key=key, model=model,
            temperature=0.2, timeout=600,
        )
    else:
        if model not in config.MODELS:
            raise ValueError(f"unknown model: {model}")
        port = config.MODELS[model]["port"]
        from langchain_openai import ChatOpenAI
        # disable_streaming forces a single JSON POST to /v1/chat/
        # completions instead of an SSE stream. The user's local
        # transformers_openai_server.py returns plain JSON, which
        # langchain-openai's streaming parser refuses ("No generations
        # found in stream"); vLLM works in either mode. So we always
        # disable streaming for predictability.
        llm = ChatOpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="not-used", model=model,
            temperature=0.2, timeout=900,
            disable_streaming=True,
        )

    # Stateful MCP tools must be supplied by open_agent_for_session(), whose
    # context stays alive until the WebSocket session closes.
    if mcp_on and _mcp_tools is None:
        raise RuntimeError(
            "MCP-enabled WebUI agents must be built with "
            "open_agent_for_session() so server state persists across calls")
    mcp_tools = list(_mcp_tools or [])

    # ── Host tools (bash/read/write/web_search/spawn_subagent)
    host = []
    # The product path's settings, which _bash_tool_for documents as such.
    # Taking the bare defaults gave a researcher's own run the evaluation
    # harness's bubblewrap jail and its "[actions spent: N]" stamp, which told
    # the model it was in a graded campaign and confounded anything measured
    # through this interface. The jail also explains solvers probing as
    # available on the host and then being unfindable inside the run.
    bash = la._bash_tool_for(workdir, isolate=False, budget_note=False)
    # Every command carries the run's marker, so Stop can find what it started
    # even after `cd /tmp`: a critic's checks run from /tmp were invisible to it.
    # The transcript still shows the command exactly as the model wrote it.
    import shlex
    from langchain_core.tools import StructuredTool as _ST
    _marker = f"export OPENPASO_CELL_WORKDIR={shlex.quote(str(Path(workdir).resolve()))}\n"

    def run_bash(command: str) -> str:
        return bash.invoke({"command": _marker + command})

    host.append(_ST.from_function(func=run_bash, name=bash.name,
                                  description=bash.description))
    host.extend(la._read_write_tools_for(workdir))
    host.append(la.web_search)

    # spawn_subagent: build the sub-agent INLINE so it uses the parent's
    # active model (mock vs vLLM). Using langgraph_eval's factory
    # hard-codes a vLLM endpoint by size, which breaks the mock path.
    from langgraph.prebuilt import create_react_agent

    def _sub_llm():
        """The sub-agent's model.

        This resolved its endpoint through config.MODELS[model]["port"], which
        raises KeyError for every OpenRouter id, so on the backend actually in
        use the critic did not run at all: it returned "[sub-agent error:
        KeyError]" and the interface, which renders neither subagent event,
        showed nothing. Every recorded spawn came from a mock session.

        It still uses the parent's model, so this is a second opinion from the
        same model and not an independent one. That is a real limit and the
        interface must not describe its output as verification.
        """
        if model == "mock":
            return _mock_chat_model()
        from langchain_openai import ChatOpenAI
        if model in config.OPENROUTER_MODELS:
            key = config.openrouter_key()
            if not key:
                raise ValueError("No OpenRouter key; the sub-agent cannot run.")
            return ChatOpenAI(
                base_url=config.OPENROUTER_URL, api_key=key, model=model,
                temperature=0.3, timeout=600,
            )
        if model not in config.MODELS:
            raise ValueError(f"unknown model for sub-agent: {model}")
        port = config.MODELS[model]["port"]
        return ChatOpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="not-used", model=model,
            temperature=0.3, timeout=600,
            disable_streaming=True,
        )

    _SUB_PROMPTS = {
        "critic": ("You are a ruthlessly critical reviewer. Challenge "
                   "every parameter choice, check units, look for sign "
                   "errors, verify BCs and validate against literature "
                   "via web_search if available. Respond with one of: "
                   "APPROVED: <reason> | REJECTED: <issue and fix>."),
        "verifier": ("You are an independent verifier. Re-derive the "
                     "requested quantity from first principles or by "
                     "an alternative method, then compare numerically."),
        "researcher": ("You are a research assistant. Look up "
                       "authoritative sources for the requested "
                       "information and summarise."),
        # openPASO's own instructions require a coupled problem's participant
        # scripts to be written by a sub-agent with this role, one ladder step
        # each. Without it the brief arrived at a research assistant, which
        # answered with a summary instead of writing the participant.
        "worker": ("You do the work you are given, in full, in this run's own "
                   "directory. Write the files the brief asks for, run what it "
                   "says to run, and report what you actually did and what the "
                   "output was. Do not summarise instead of doing it, and do "
                   "not hand the work back unfinished without saying so."),
    }

    async def spawn_subagent_emitting(role: str, task: str,
                                      context: str = "") -> str:
        sa_id = f"sa_{uuid.uuid4().hex[:8]}"
        await emitter({"type": "subagent_spawned", "sa_id": sa_id,
                       "role": role, "task": task, "context": context})
        # The critic's own commands used to run unseen: its tools were the raw
        # ones, not the wrapped ones. They are shown now, labelled with its role,
        # and gated like the main agent's: "Ask before each step" means every
        # step, including the ones a critic takes (it ran a solver unasked).
        sub_tools = [_wrap_tool(t, emitter=emitter, get_mode=get_mode,
                                gate=gate, agent_label=role, steps=steps,
                                take_steers=(lambda: take_steers(sa_id)) if take_steers else None)
                     for t in (mcp_tools + host) if t.name != "spawn_subagent"]
        sys = _SUB_PROMPTS.get(role, _SUB_PROMPTS["researcher"])
        sub_agent = create_react_agent(_sub_llm(), tools=sub_tools,
                                       prompt=sys)
        msg = f"Task: {task}\n\nContext provided by parent:\n{context}"
        try:
            # ainvoke avoids the inner asyncio.run() that the sync .invoke
            # would require, and keeps us on the caller's event loop.
            out = await sub_agent.ainvoke(
                {"messages": [("user", msg)]},
                config={"recursion_limit": 40})
            res = out["messages"][-1].content
            if not str(res).strip():
                # It happens: a sub-agent spends its steps and ends with nothing
                # to say. An empty string handed back to the model reads as
                # assent — "the critic had no objection" — when in fact nobody
                # reviewed anything, and openPASO's gate will hold no review
                # either. Say it, so the model cannot mistake silence for a
                # verdict.
                # One sentence, shared word for word with langgraph_eval's own
                # spawn_subagent, so a transcript here and a campaign trajectory
                # say the same thing. The campaign branch carries "OASiS" until
                # its freeze and the rename maps it to "openPASO" on the way in.
                who = role if role in ("critic", "verifier", "researcher") else "sub-agent"
                res = (f"[the {who} returned no text. This is NOT approval and NOT a review: "
                       "it produced nothing. Treat the step as not done. Do not write its "
                       "answer for it — run it again, or say that it produced nothing. "
                       "openPASO's verification gate holds no review for this setup.]")
        except Exception as e:
            res = f"[sub-agent error: {type(e).__name__}: {e}]"
        from .outcome import shorten
        # shorten, not slice: a verdict cut without a mark reads as a verdict
        # that ended there, and someone concludes the critic never raised what
        # it raised in the part that was dropped
        await emitter({"type": "subagent_returned",
                       "sa_id": sa_id, "result": shorten(str(res), 6000)})
        return res

    from langchain_core.tools import StructuredTool
    spawn_wrapped = StructuredTool.from_function(
        coroutine=spawn_subagent_emitting,
        name="spawn_subagent",
        description=("Spawn a sub-agent. role∈{worker, critic, verifier, "
                     "researcher}. task = what it should do. context = "
                     "facts to pass in. Returns its final message."),
    )
    host.append(spawn_wrapped)

    # ── Gate every tool through the wrapper for mode gating
    gated = [_wrap_tool(t, emitter=emitter, get_mode=get_mode, gate=gate,
                        agent_label="main", take_steers=take_steers, steps=steps)
             for t in mcp_tools + host]

    from langgraph.prebuilt import create_react_agent
    # Use the EXACT same system prompts as the langgraph_eval driver, including
    # the MANDATORY CRITIC paragraph. Softening them in the WebUI would change
    # the agent's behaviour relative to the paper claim, and any failure mode
    # the strict prompt causes on small models is a real finding, not a bug to
    # paper over.
    #
    # The openPASO arm's text became a function when the server's own
    # instructions became its source, so it is built per call rather than read
    # from a constant. Calling the private name is deliberate: the guarantee
    # above is that these are the same bytes the driver uses, and a local copy
    # would quietly stop being that.
    prompt = (la._mcp_system_prompt() if mcp_on else la.BARE_SYSTEM)
    return create_react_agent(llm, tools=gated, prompt=prompt,
                              checkpointer=checkpointer)


@asynccontextmanager
async def open_agent_for_session(**kwargs):
    """Yield one WebUI agent and keep its MCP process alive across turns."""
    import agent as la

    workdir = kwargs["workdir"]
    try:
        if kwargs.get("mcp_on"):
            # The defaults are the evaluation campaign's: a fixed subset of
            # tools and a bubblewrap jail. The function's own docstring says
            # surface="all" is what an ordinary user of the product should get,
            # and the jail made solvers that probe as installed on this machine
            # unfindable inside a run.
            async with la.openpaso_mcp_tools_session(
                    workdir, surface="all", isolate=False) as mcp_tools:
                yield build_agent_for_session(
                    **kwargs, _mcp_tools=mcp_tools)
        else:
            yield build_agent_for_session(**kwargs)
    finally:
        la.cleanup_sandbox_scratch(workdir)


# ───────────────────────────────────────────────────────────────────
# Streamed turn
# ───────────────────────────────────────────────────────────────────
async def stream_turn(*, agent, user_text: str, emitter, emit_done: bool = False,
                      thread_id: str | None = None,
                      history: list[tuple[str, str]] | None = None):
    """Run one user turn. Streams chunks/events via ``emitter`` and
    returns the final message text. Emits a 'thinking' status as soon
    as we start so the user sees activity even before the first model
    response, and an 'error' event with a clear message on any failure
    (rather than dying silently)."""
    final_text = ""
    # Each turn used to send only its own message to an agent with no memory, so
    # a follow-up such as "now refine the mesh" started from nothing. The agent
    # keeps the conversation per run now; `history` re-seeds it when that memory
    # was lost (server restart, or a stop that left a call unanswered).
    inputs = {"messages": list(history or []) + [("user", user_text)]}
    await emitter({"type": "status", "message": "thinking…"})
    try:
        # LangGraph's default ceiling is 25 steps, which a simulation task
        # passes while it is still reading documentation: a real run died at it
        # having written its solver but never run it. openPASO's own rule is
        # that there is one budget and it is the clock, so the step ceiling sits
        # well above anything the wall time can reach.
        async for event in agent.astream_events(
                inputs, version="v2",
                config={"recursion_limit": 400,
                        "configurable": {"thread_id": thread_id or uuid.uuid4().hex}}):
            kind = event.get("event")
            name = event.get("name")
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and getattr(chunk, "content", None):
                    await emitter({"type": "agent_chunk",
                                   "text": chunk.content})
            elif kind == "on_chat_model_end":
                gen = event["data"].get("output")
                if gen is not None and hasattr(gen, "usage_metadata"):
                    um = gen.usage_metadata or {}
                    await emitter({"type": "token_count",
                                   "input": um.get("input_tokens"),
                                   "output": um.get("output_tokens")})
                if gen is not None and getattr(gen, "content", ""):
                    final_text = gen.content
                    # Non-streaming models (disable_streaming=True) won't
                    # emit on_chat_model_stream chunks, so surface the
                    # full content here as an agent_msg.
                    await emitter({"type": "agent_msg",
                                   "text": gen.content})
            elif kind == "on_chain_end" and name == "LangGraph":
                msgs = event["data"].get("output", {}).get("messages") or []
                if msgs:
                    last = msgs[-1]
                    final_text = (getattr(last, "content", "")
                                  or final_text)
    except Exception:
        # The caller decides the terminal state and reports it. Emitting a
        # "done" here as well is what let a crashed run read as "finished".
        raise
    if emit_done:
        # Only for standalone callers. The web UI's run decides the outcome
        # itself, because "no exception" is not "a solver ran"; this default
        # used to be on and announced "completed" for turns that computed
        # nothing, before the run could say otherwise.
        await emitter({"type": "done", "final_text": final_text})
    return final_text
