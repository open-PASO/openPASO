"""A run lives on the server, not in a browser tab.

Before this, everything about a run (its agent, its turn, its state) was held by
the WebSocket handler of the tab that started it. Closing that socket cancelled
the run: going back, reloading, opening another run, or a network blip all
killed work that could have been running for a quarter of an hour. Two tabs on
one run each loaded their own copy of the record, and the tab that closed last
wrote its stale copy over the live one.

Now there is one `Run` per session in this process. Browser tabs subscribe to
it and receive its events; they can come and go. The run keeps going, and only
the run writes its record.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import traceback
import uuid
from pathlib import Path

from fastapi import WebSocket

from . import config, proctree, sessions
from .outcome import (RUNNING, SOLVER_TOOLS, classify_solver_result,
                      clean_outcome, fold)

log = logging.getLogger("openpaso.webui.runs")

# What a person may upload into a run: geometry, meshes, solver input, data.
UPLOAD_SUFFIXES = {
    ".step", ".stp", ".iges", ".igs", ".stl", ".brep", ".geo",
    ".msh", ".vtk", ".vtu", ".xdmf", ".h5", ".hdf5", ".exo", ".e", ".med",
    ".inp", ".dat", ".yaml", ".yml", ".json", ".xml", ".feb", ".py",
    ".csv", ".txt", ".md", ".npy", ".npz", ".png", ".jpg", ".jpeg", ".pdf",
}
UPLOAD_MAX_BYTES = 500 * 1024 * 1024

UNFINISHED = "unfinished"   # the log has no end and nothing is running it


class Run:
    # a default on the class, so a Run built in a test or by a path that does
    # not run __init__ still answers this question
    deleted = False

    def __init__(self, sid: str):
        self.sid = sid
        self.state = sessions.load(sid)
        if self.state.get("mode") not in config.MODES:
            self.state["mode"] = config.DEFAULT_MODE
        self.workdir = _workdir(sid)
        from .runner import ApprovalGate, StepControl
        self.gate = ApprovalGate()
        self.steps = StepControl()
        self.agent = None
        self.agent_context = None
        self.turn_task: asyncio.Task | None = None
        self.subscribers: set[WebSocket] = set()
        # held while an event goes out and while a tab is being greeted, so the
        # two cannot interleave and leave a tab's first view missing an event
        self.send_lock = asyncio.Lock()
        self.deleted = False
        self.steers: list[dict] = []           # corrections not yet delivered
        # conversation memory survives the agent being rebuilt; a new thread is
        # started (and re-seeded from the log) only when the old one cannot be
        # trusted, e.g. after a stop left a tool call without its answer
        from langgraph.checkpoint.memory import MemorySaver
        self.saver = MemorySaver()
        self.thread = 0
        self.thread_seeded = False
        self._seq = max((e.get("seq", 0) for e in self.state["events"]), default=0)
        self._agent_lock = asyncio.Lock()
        self._keeper: asyncio.Task | None = None
        self._agent_stop: asyncio.Event | None = None
        self._turn_ended = True

    # ── state ───────────────────────────────────────────────────────────
    @property
    def running(self) -> bool:
        return self.turn_task is not None and not self.turn_task.done()

    @property
    def waiting(self) -> bool:
        """A step is waiting for the person to approve it."""
        return self.running and any(not f.done() for f in self.gate._pending.values())

    def outcome(self) -> str:
        # something is working on it: that is the state, whatever the log's last
        # turn says (a follow-up's first event may not be written yet)
        if self.running:
            return RUNNING
        o = fold(self.state["events"])
        if o == RUNNING and not self.running:
            return UNFINISHED
        return o

    def snapshot(self) -> dict:
        snap = {k: v for k, v in self.state.items() if k != "events"}
        snap["running"] = self.running
        snap["outcome"] = self.outcome()
        snap["pending_steers"] = len(self.steers)
        snap["waiting"] = self.waiting
        return snap

    def mode(self) -> str:
        return self.state.get("mode", config.DEFAULT_MODE)

    # ── events ──────────────────────────────────────────────────────────
    async def emit(self, event: dict):
        event = dict(event)
        event.setdefault("t", int(time.time() * 1000))
        # The verdict travels with the event instead of being worked out again
        # on the other side. Two readings of the same result, one here and one
        # in the browser, will differ eventually — and they did: the page said
        # a result was verified while this rule called it unverified. What the
        # server decided is what the page shows; the page keeps its own reading
        # only for records written before this stamp existed.
        if (event.get("type") == "tool_result" and event.get("tool") in SOLVER_TOOLS
                and "verdict" not in event):
            # only where the tool layer did not judge the full result already:
            # this text may be a shortened copy, and that is not the evidence
            event["verdict"] = classify_solver_result(event.get("result") or "")
        if event.get("type") == "done" and event.get("outcome"):
            event["by"] = "server"
        if event.get("type") != "status":
            self._seq += 1
            event["seq"] = self._seq
            self.state["events"].append({k: v for k, v in event.items() if k != "session"})
            # what a turn is and how it ended is written at once; the rest is
            # checkpointed. A restart during an early turn used to leave a
            # record with no prompt in it, which the run list then hides.
            if event.get("type") in ("turn_start", "user_msg", "done", "error",
                                     "file_uploaded", "mode_changed") or self._seq % 10 == 0:
                self.save()
        if event.get("type") == "cost":
            self.state["cost_usd"] = round(self.state.get("cost_usd", 0.0) + (event.get("usd") or 0.0), 6)
        if event.get("type") == "model_info":
            self.state["model_detail"] = event.get("model")
        if event.get("type") == "token_count":
            ti, to = event.get("input") or 0, event.get("output") or 0
            self.state["tokens_in"] = self.state.get("tokens_in", 0) + ti
            self.state["tokens_out"] = self.state.get("tokens_out", 0) + to
            if self.state.get("model") in config.OPENROUTER_MODELS:
                from .catalog import cost_of
                c = cost_of(self.state["model"], ti, to)
                if c is not None:
                    self.state["cost_usd"] = round(self.state.get("cost_usd", 0.0) + c, 6)
        if event.get("type") in ("cost", "token_count") and "cost_usd" in self.state:
            # an open page shows the running total without waiting for a reload
            event["cost_usd_total"] = self.state["cost_usd"]
        await self.broadcast(event)

    async def broadcast(self, event: dict):
        """Send one event to every tab watching.

        Sent to all of them at once, with a deadline: awaiting each in turn let
        a tab in a background window, whose socket has stopped draining, hold up
        the run itself and every other tab with it. A socket that cannot take an
        event within the deadline is dropped; the run is the record, and that tab
        reconnects and is replayed from it.

        Under the same lock the handshake uses, so an event cannot slip between
        a new tab's snapshot and its hello and be lost from that tab's view."""
        from .privacy import scrub
        text = json.dumps(scrub(event), default=str)
        async with self.send_lock:
            watchers = list(self.subscribers)
            if not watchers:
                return

            async def to(ws):
                try:
                    await asyncio.wait_for(ws.send_text(text), timeout=10)
                except Exception:
                    self.subscribers.discard(ws)

            await asyncio.gather(*(to(ws) for ws in watchers))

    async def push_snapshot(self):
        await self.broadcast({"type": "session", "session": self.snapshot()})

    def save(self):
        if self.deleted:                  # never write a deleted run back
            return
        try:
            sessions.save(self.state)
        except Exception:
            log.warning("could not save run %s", self.sid)

    # ── agent ───────────────────────────────────────────────────────────
    async def ensure_agent(self):
        """The agent, with its openPASO tool server.

        One dedicated task opens that connection and the same task closes it.
        The connection lives in anyio task groups, which must be exited by the
        task that entered them: closing it from whichever task noticed the run
        was idle (a closing browser tab, the end of a turn) broke a second run's
        connection as it was being opened."""
        async with self._agent_lock:
            if self.agent is not None and self._keeper is not None and not self._keeper.done():
                return self.agent
            loop = asyncio.get_running_loop()
            ready: asyncio.Future = loop.create_future()
            self._agent_stop = asyncio.Event()
            self._keeper = asyncio.create_task(self._keep(ready, self._agent_stop))
            self.agent = await ready
            return self.agent

    async def _keep(self, ready: asyncio.Future, stop: asyncio.Event):
        from .runner import open_agent_for_session
        try:
            async with open_agent_for_session(
                    model=self.state["model"],
                    mcp_on="openpaso" in self.state.get("mcp_servers", []),
                    workdir=self.workdir, emitter=self.emit, get_mode=self.mode,
                    gate=self.gate, checkpointer=self.saver,
                    take_steers=self._take_steers, steps=self.steps) as agent:
                if not ready.done():
                    ready.set_result(agent)
                await stop.wait()
        except BaseException as exc:          # noqa: BLE001 - handed to the waiter
            if not ready.done():
                ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                log.warning("openPASO connection for run %s ended: %s", self.sid, _leaf(exc))
        finally:
            if self.agent is not None and self._keeper is asyncio.current_task():
                self.agent = None

    async def close_agent(self):
        keeper, self._keeper = self._keeper, None
        self.agent = None
        if keeper is None:
            return
        if self._agent_stop is not None:
            self._agent_stop.set()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(asyncio.shield(keeper), timeout=20)
        if not keeper.done():
            keeper.cancel()

    # ── turns ───────────────────────────────────────────────────────────
    async def prompt(self, text: str, attachments: list[str] | None = None):
        text = (text or "").strip()
        if not text or self.deleted:      # the run and its folder are gone
            return
        if self.running:
            # a message sent while the run works is a correction, not a new turn
            await self.steer(text)
            return
        attachments = [a for a in (attachments or []) if a]
        agent_text = text
        if attachments:
            agent_text += ("\n\n(The user uploaded these files into your working "
                           "directory: " + ", ".join(f"uploads/{a}" for a in attachments) + ")")
        # before the task exists: Stop pressed in the same breath as Send used to
        # read the previous turn's flag and answer "Nothing is running in this run"
        self._turn_ended = False
        self.turn_task = asyncio.create_task(
            self._turn(agent_text, shown={"text": text, "attachments": attachments}))
        await self.push_snapshot()

    async def _turn(self, text: str, shown: dict | None = None):
        """One turn. It ends in exactly one terminal state, and subscribers are told
        which. `completed` is a claim about the physics: it needs a solver to
        have run, otherwise the run ended without a result."""
        from .runner import set_search_scope
        set_search_scope(self.sid)        # this run's searches are its own
        outcome = "completed"
        use_claude = self.state["model"] == config.CLAUDE_CODE_ID
        start = len(self.state["events"])
        self._turn_ended = False
        await self.emit({"type": "turn_start"})
        if shown is not None:
            await self.emit({"type": "user_msg", **shown})
        try:
            if use_claude:
                from . import claude_code
                await claude_code.stream_turn(
                    text, workdir=self.workdir,
                    servers=list(self.state.get("mcp_servers") or []),
                    mode=self.mode(), emit=self.emit, state=self.state)
            else:
                from .runner import stream_turn
                agent = await self.ensure_agent()
                history = None
                if not self.thread_seeded:
                    # everything before this turn: it has already emitted its
                    # turn_start and its user_msg, and dropping only the last
                    # left an empty turn marker, so a seeded conversation began
                    # with a blank message from the user
                    prior = self.state["events"]
                    for i in range(len(prior) - 1, -1, -1):
                        if prior[i].get("type") == "turn_start":
                            prior = prior[:i]
                            break
                    else:
                        prior = prior[:-1]
                    history = _history(prior)
                await stream_turn(agent=agent, user_text=text, emitter=self.emit,
                                  thread_id=f"{self.sid}:{self.thread}",
                                  history=history)
                # only now: a turn that failed before the model saw the history
                # would otherwise leave the next follow-up with no past at all
                self.thread_seeded = True
        except asyncio.CancelledError:
            outcome = "interrupted"
            raise
        except Exception as exc:   # noqa: BLE001 - reported to the person
            outcome = "failed"
            log.exception("turn failed for run %s", self.sid)
            leaf = _leaf(exc)
            message = f"{type(leaf).__name__}: {leaf}".rstrip(": ")
            if type(leaf).__name__ in ("BrokenResourceError", "ClosedResourceError", "EndOfStream"):
                message = ("The connection to openPASO's tool server broke. Send a follow-up to "
                           f"start it again. ({type(leaf).__name__})")
            elif "chunk is longer than limit" in str(leaf) or "Separator is found" in str(leaf):
                message = ("The interface could not read a very long message from Claude Code "
                           "(a limitation of this interface, not of your input). Send a follow-up to continue.")
            elif type(leaf).__name__ == "GraphRecursionError":
                message = ("The run used all the steps it is allowed in one turn without finishing. "
                           "Send a follow-up asking for a smaller piece of the problem.")
            await self.close_agent()
            # closing the connection does not end a solver the run had started:
            # it would have kept every core it was given while the page said the
            # run had failed
            ended = await asyncio.to_thread(proctree.end_run_processes, self.workdir, 3.0,
                                            self.state.get("created_at"))
            if ended:
                message += (f" {ended} process{'es' if ended != 1 else ''} it had started "
                            f"{'were' if ended != 1 else 'was'} ended.")
            await self.emit({"type": "error", "outcome": "failed", "message": message,
                             "traceback": _tail(traceback.format_exception(exc), 4000),
                             "processes_ended": ended})
        finally:
            if outcome == "completed":
                outcome = clean_outcome(self.state["events"][start:])
            if outcome != "interrupted":
                self._turn_ended = True
                await self.emit({"type": "done", "outcome": outcome})
            self.save()
            if outcome != "interrupted":
                asyncio.get_running_loop().call_soon(
                    lambda: asyncio.ensure_future(self._after_turn()))

    async def _after_turn(self):
        if self.deleted:
            return
        await self.push_snapshot()
        # corrections that arrived after the last step finished are not dropped:
        # they become the next turn, and the log says so
        if self.steers and not self.running:
            pending, self.steers = self.steers, []
            for s in pending:
                await self.emit({"type": "steer_state", "id": s["id"], "state": "sent_as_followup"})
            joined = "\n\n".join(s["text"] for s in pending)
            self._turn_ended = False
            # shown, so the log holds it as a message from the user: the history
            # a restarted run is given is rebuilt from those, and without it the
            # model lost a follow-up the transcript said had been sent
            self.turn_task = asyncio.create_task(self._turn(
                "While you were working I sent the following. Act on it now:\n\n" + joined,
                shown={"text": joined, "attachments": []}))
            await self.push_snapshot()
            return
        if self.deleted:            # the run and its folder are gone
            return
        # a follow-up may have arrived between the end of that turn and this:
        # closing the connection under it would break the turn now running
        if not self.subscribers and not self.running:
            await self.close_agent()

    # ── corrections ─────────────────────────────────────────────────────
    async def steer(self, text: str):
        sid = f"st_{uuid.uuid4().hex[:8]}"
        self.steers.append({"id": sid, "text": text})
        # Claude Code runs as one command that cannot be spoken to while it
        # works: there is no tool result of ours to attach the message to, so it
        # is sent as a follow-up the moment the turn ends. Saying "it will reach
        # the model at the next step" would be a promise this path cannot keep.
        state = ("queued_until_turn_ends"
                 if self.state.get("model") == config.CLAUDE_CODE_ID else "queued")
        await self.emit({"type": "user_steer", "id": sid, "text": text, "state": state})
        await self.push_snapshot()

    def _take_steers(self, agent: str = "main") -> list[dict]:
        """Called by the tool wrapper when a tool returns: the correction is
        handed to the model with that result, which it reads next.

        A sub-agent gets a copy rather than taking it away. A critic can run for
        many minutes, and a correction sent during its work ("stop running MPI
        tests") is for whoever is working; queueing it until the critic finished
        meant it arrived after the thing it was meant to prevent. The main agent
        still receives it, so the run as a whole is not steered behind its
        back."""
        if agent != "main":
            # keyed by the sub-agent itself: two critics in one run are two
            # workers, and the second must hear the correction as well
            fresh = [s for s in self.steers if agent not in s.setdefault("seen_by", set())]
            for s in fresh:
                s["seen_by"].add(agent)
            return [{"id": s["id"], "text": s["text"]} for s in fresh]
        taken, self.steers = self.steers, []
        # this is called from a tool's thread of work, so the events go out as
        # tasks on the loop that is running it; asyncio.ensure_future needs one
        # and raises where there is none
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            for s in taken:
                loop.create_task(self.emit(
                    {"type": "steer_state", "id": s["id"], "state": "delivered"}))
            if taken:
                loop.create_task(self.push_snapshot())
        return taken

    # ── end one step ────────────────────────────────────────────────────
    async def end_step(self, call_id: str) -> str | None:
        """End one executing step and let the run carry on. Returns why it
        could not, or None."""
        task = self.steps.tasks.get(call_id)
        if task is None or task.done():
            return "That step is no longer running."
        self.steps.ended.add(call_id)
        await self.emit({"type": "step_ending", "call_id": call_id})
        # the step's processes first: a solver killed underneath its tool makes
        # the tool return by itself, which leaves the openPASO connection intact.
        # Only what began with this step: a run can have another step working at
        # the same time, and that one is not the one being ended.
        since = self.steps.started.get(call_id) or self.state.get("created_at")
        await asyncio.to_thread(proctree.end_step_processes, self.workdir, 3.0, since)
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(asyncio.shield(task), timeout=6)
        if not task.done():
            task.cancel()
        return None

    # ── stop ────────────────────────────────────────────────────────────
    async def stop(self) -> int:
        task = self.turn_task
        if task is None or task.done() or self._turn_ended:
            return -1
        # undelivered corrections belong to the stopped work
        for s in self.steers:
            await self.emit({"type": "steer_state", "id": s["id"], "state": "not_delivered"})
        self.steers = []
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        await self.close_agent()
        since = self.state.get("created_at")
        ended = await asyncio.to_thread(proctree.end_run_processes, self.workdir, 3.0, since)
        left = await asyncio.to_thread(proctree.run_processes, self.workdir, since)
        # the conversation may now hold a tool call with no answer; continue in a
        # fresh thread seeded from the log rather than on a broken one
        self.thread += 1
        self.thread_seeded = False
        msg = ("Run stopped. " + (f"{ended} process{'es' if ended != 1 else ''} it had started "
               f"{'were' if ended != 1 else 'was'} ended." if ended else "No leftover processes were found to end."))
        if left:
            msg += f" {len(left)} could not be ended."
        self._turn_ended = True
        await self.emit({"type": "error", "outcome": "interrupted", "message": msg,
                         "processes_ended": ended, "processes_left": len(left)})
        await self.emit({"type": "done", "outcome": "interrupted"})
        self.save()
        await self.push_snapshot()
        return ended


def _tail(lines, limit: int) -> str:
    """The end of a traceback, saying so when the start was dropped: the frames
    that matter are the last ones, and a cut with no mark reads as the whole."""
    text = "".join(lines)
    if len(text) <= limit:
        return text
    return (f"[… {len(text) - limit} characters of earlier frames left out …]\n"
            + text[-limit:])


def _leaf(exc: BaseException) -> BaseException:
    """The first real exception inside nested exception groups."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _workdir(sid: str) -> Path:
    from .runner import _session_workdir
    return _session_workdir(sid)


_HISTORY_LIMIT = 60_000   # characters of earlier work handed back to the model


def _clip(text: str, limit: int) -> str:
    """Shorten, and say so. A conclusion cut mid-sentence at two thousand
    characters — "…the mesh is adequate for" — reads as approval to whoever is
    handed it next."""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f" […{len(text) - limit} more characters]"


def _history(events: list[dict]) -> list[tuple[str, str]]:
    """Earlier turns of this run, for a thread that has lost its memory (server
    restart, or a stop).

    Only the prompts and the model's messages used to be handed back. After a
    Stop the model then read its own plan with none of the steps it had taken,
    and told the person nothing had been computed while the page showed three
    solved meshes. Each turn now carries a record of its steps: what was run,
    what came back (shortened), and how the turn ended."""
    turns: list[tuple[str, list[str]]] = []
    calls: dict[str, dict] = {}
    for e in events:
        t = e.get("type")
        if t == "user_msg":
            turns.append((e.get("text", ""), []))
            continue
        if not turns:
            continue
        work = turns[-1][1]
        if t == "agent_msg" and (e.get("text") or "").strip():
            work.append(e["text"].strip())
        elif t == "tool_call_pending":
            calls[e.get("call_id", "")] = e
        elif t in ("tool_result", "tool_error", "tool_call_rejected"):
            call = calls.get(e.get("call_id", ""), {})
            who = "" if call.get("agent") in (None, "main") else f"{call.get('agent')} "
            args = json.dumps(call.get("args") or {}, default=str)
            if len(args) > 1500:
                args = args[:1500] + " …"
            if t == "tool_call_rejected":
                got = "[the user skipped this step; it did not run]"
            else:
                got = str(e.get("result") if t == "tool_result" else e.get("error"))
                if len(got) > 2500:
                    got = got[:1200] + "\n … \n" + got[-1200:]
            work.append(f"[{who}step] {call.get('tool', e.get('tool', '?'))} {args}\n→ {got}")
        elif t == "subagent_returned" and (e.get("result") or "").strip():
            work.append("[critic/helper conclusion] " + _clip(e["result"], 2000))
        elif t == "error" and e.get("outcome") == "interrupted":
            unfinished = [f"{c.get('tool')} {json.dumps(c.get('args') or {}, default=str)[:300]}"
                          for cid, c in calls.items()
                          if not any(x.get("call_id") == cid and x.get("type") in
                                     ("tool_result", "tool_error", "tool_call_rejected")
                                     for x in events)]
            work.append("[the user stopped the run here"
                        + (f"; this step was running and has no result: {unfinished[-1]}"
                           if unfinished else "; a step that was running has no result")
                        + "]")
        elif t == "error":
            work.append(f"[the turn failed: {_clip(str(e.get('message')), 500)}]")
    out: list[tuple[str, str]] = []
    for prompt, work in turns:
        out.append(("user", prompt))
        if work:
            out.append(("assistant", "[Record of what I did for this message, rebuilt from the run log "
                                     "because the conversation was restarted. These steps really ran.]\n\n"
                                     + "\n\n".join(work)))
    # keep the most recent work when a long run does not fit
    total = 0
    for i in range(len(out) - 1, -1, -1):
        total += len(out[i][1])
        if total > _HISTORY_LIMIT:
            role, text = out[i]
            out[i] = (role, "[… earlier part shortened …]\n" + text[-(_HISTORY_LIMIT // 4):])
            out = out[i:] if role == "user" else [("user", "(earlier messages shortened)")] + out[i:]
            break
    return out


# ── registry ────────────────────────────────────────────────────────────
RUNS: dict[str, Run] = {}


def get(sid: str) -> Run:
    run = RUNS.get(sid)
    if run is None:
        run = Run(sid)          # raises FileNotFoundError / ValueError
        RUNS[sid] = run
    return run


def live(sid: str) -> Run | None:
    return RUNS.get(sid)


def running_count() -> int:
    return sum(1 for r in RUNS.values() if r.running)


_ADMISSION = asyncio.Lock()


async def start_turn(run: "Run", text: str, attachments: list[str] | None = None) -> str | None:
    """Start a turn if the machine has room for it. Returns why it could not.

    Counting and starting have to happen together: two prompts arriving at the
    same moment both saw a count below the limit and both started, so the limit
    the interface advertises was a suggestion under exactly the conditions it
    exists for."""
    async with _ADMISSION:
        if not run.running and running_count() >= config.MAX_RUNNING:
            return (f"{config.MAX_RUNNING} runs are already working on this machine. "
                    "Your message was not sent. Wait for one to finish or stop one, then send it again.")
        await run.prompt(text, attachments)
        return None
