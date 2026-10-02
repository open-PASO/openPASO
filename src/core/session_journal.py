"""Session journal — records structured events during MCP tool calls.

Minimal, append-only, zero-friction for tool authors.
Each MCP session (one server process) gets one journal.
Events are recorded in-memory and persisted to JSON on shutdown.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


# Valid event types
EVENT_TYPES = frozenset({
    "tool_call",          # any tool invocation
    "tool_success",       # tool completed successfully
    "tool_error",         # tool returned an error
    "knowledge_lookup",   # knowledge/prepare_simulation called
    "source_read",        # developer(action='files') called
    "parameter_override", # agent used non-default parameters
    "convergence_issue",  # coupling took many iterations
    "critic_review",      # submit_critic_review called
})


@dataclass
class JournalEvent:
    """A single event in the session journal."""

    timestamp: float
    event_type: str
    tool_name: str
    solver: str = ""
    physics: str = ""
    details: dict = field(default_factory=dict)
    error_message: str = ""
    notes: str = ""
    input_snapshot: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.event_type not in EVENT_TYPES:
            raise ValueError(
                f"Unknown event_type: {self.event_type!r}. "
                f"Valid: {sorted(EVENT_TYPES)}"
            )
        # Sanitise: truncate long error messages to prevent bloat
        if len(self.error_message) > 500:
            self.error_message = self.error_message[:497] + "..."
        if len(self.notes) > 500:
            self.notes = self.notes[:497] + "..."
        # Ensure details and input_snapshot are serialisable
        if not isinstance(self.details, dict):
            self.details = {}
        if not isinstance(self.input_snapshot, dict):
            self.input_snapshot = {}


def state_dir(name: str) -> Path:
    """A writable directory for openPASO's runtime state, NEVER the install tree.

    A pip-installed package lives in site-packages, which is shared and often read-only, so
    anything openPASO writes while it runs -- build logs, the source-discovery cache, staged
    community knowledge -- must go to the user's state area, not beside the code. This is the one
    answer to "where does runtime state go", the same resolution the journal uses:

      OPENPASO_STATE_HOME override  ->  $XDG_STATE_HOME/openpaso/<name>  ->  ~/.local/state/openpaso/<name>

    Writing into the install also makes the code directory stateful (version control is for not
    doing that) and invalidates anything that hashes an installation by the act of using it -- the
    same defect the journal had (see live_session_dir)."""
    import os as _os
    override = _os.environ.get("OPENPASO_STATE_HOME")
    if override:
        return Path(override) / name
    xdg = _os.environ.get("XDG_STATE_HOME")
    root = Path(xdg) if xdg else Path.home() / ".local" / "state"
    return root / "openpaso" / name


def live_session_dir() -> Path:
    """Where a running session's journal is written. NEVER the install tree.

    IT USED TO BE `<install>/data/sessions`, so the FIRST RECORDED EVENT -- in
    practice the first tool call -- wrote into the directory openPASO was
    imported from. Measured on 2026-09-19: importing alone changes nothing, and
    one `record()` adds a file. Three consequences, and the first is the one
    that cost a round:

      * anything that hashes an installation is invalidated by the act of
        using it. A five-run coupled round was voided as an infrastructure failure because a
        reply-size measurement, run from inside the frozen source snapshot
        those cells ran against, journalled four knowledge lookups into it. Not
        one byte of code differed; the tree hash moved anyway, and the guard
        was right to say so.
      * a read-only install -- a container layer, /opt, a system package, a tree
        `pip install`ed as root and run as a user -- cannot write it, and the
        writer is best-effort and swallows the exception, so nobody learns.
      * two checkouts of openPASO share nothing, but a journal written beside
        the code makes the code directory stateful, which is the thing version
        control is for not doing.

    Order: the explicit override, then XDG's state directory, then its default
    location. An evaluation harness sets the override to each run's own directory, so
    a run's journal stays with the run. The server's shutdown save uses the same
    function, so there is exactly one answer to "where does the journal go".
    """
    import os as _os
    base = _os.environ.get("OPENPASO_JOURNAL_LIVE_DIR")
    if base:                       # a campaign points this directly at each cell's own dir
        return Path(base)
    return state_dir("sessions")   # otherwise the shared state resolution


@dataclass
class SessionJournal:
    """Collects events for one user session."""

    session_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    started_at: float = field(default_factory=time.time)
    events: list[JournalEvent] = field(default_factory=list)
    _solvers_used: set = field(default_factory=set, repr=False)
    _physics_used: set = field(default_factory=set, repr=False)

    # ── Recording ────────────────────────────────────────────

    def record(
        self,
        event_type: str,
        tool_name: str,
        *,
        solver: str = "",
        physics: str = "",
        details: dict | None = None,
        error_message: str = "",
        notes: str = "",
        input_snapshot: dict | None = None,
    ) -> JournalEvent:
        """Record an event. Returns the created event.

        This is the only method tool authors need to call.
        Designed to never raise — silently drops malformed events
        so it cannot break the tool it instruments.
        """
        try:
            evt = JournalEvent(
                timestamp=time.time(),
                event_type=event_type,
                tool_name=tool_name,
                solver=solver,
                physics=physics,
                details=details or {},
                input_snapshot=input_snapshot or {},
                error_message=str(error_message),
                notes=str(notes),
            )
        except (ValueError, TypeError):
            # Malformed event — drop silently rather than crash the tool
            return None  # type: ignore[return-value]

        self.events.append(evt)
        if solver:
            self._solvers_used.add(solver)
        if physics:
            self._physics_used.add(physics)
        self._persist_live(evt)
        return evt

    # THE JOURNAL IS WRITTEN WHERE IT CAN BE READ WHILE THE SESSION RUNS.
    #
    # It used to reach disk only when the server shut down. The server is its own
    # process -- the agent reaches it over stdio -- so anything outside it that
    # asked "has this run fetched that contract?" read an empty journal. Measured:
    # the write-time check `contract_never_fetched` fired in 19 recorded runs and
    # was wrong in all 19, including one that graded correct. One line per event,
    # appended as it is recorded, into the same session directory the shutdown
    # save uses (an evaluation harness mounts it at the run's work/.openpaso_sessions).
    # Best-effort and silent on failure: it must never break the tool it records.
    def _persist_live(self, evt: JournalEvent) -> None:
        try:
            directory = live_session_dir()
            directory.mkdir(parents=True, exist_ok=True)
            with open(directory / f"session_{self.session_id}.jsonl", "a") as fh:
                fh.write(json.dumps(_event_to_dict(evt), default=str) + "\n")
        except Exception:                                   # noqa: BLE001
            pass

    # ── Queries ──────────────────────────────────────────────

    @property
    def solvers_used(self) -> set[str]:
        return set(self._solvers_used)

    @property
    def physics_used(self) -> set[str]:
        return set(self._physics_used)

    @property
    def error_count(self) -> int:
        return sum(1 for e in self.events if e.event_type == "tool_error")

    @property
    def duration_seconds(self) -> float:
        if not self.events:
            return 0.0
        return self.events[-1].timestamp - self.started_at

    def events_by_type(self, event_type: str) -> list[JournalEvent]:
        return [e for e in self.events if e.event_type == event_type]

    # ── Serialisation ────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": time.time(),
            "duration_seconds": round(self.duration_seconds, 1),
            "solvers_used": sorted(self._solvers_used),
            "physics_used": sorted(self._physics_used),
            "event_count": len(self.events),
            "error_count": self.error_count,
            "events": [_event_to_dict(e) for e in self.events],
        }

    def save(self, directory: str | Path) -> Path:
        """Persist journal to disk as JSON. Creates directory if needed.

        Returns the path to the saved file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"session_{self.session_id}.json"
        data = self.to_dict()
        # Atomic write: write to tmp then rename
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2, default=str))
            tmp.rename(path)
        except Exception:
            # If rename fails (Windows), fall back to direct write
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            path.write_text(json.dumps(data, indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SessionJournal":
        """Load a journal from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        journal = cls(
            session_id=data["session_id"],
            started_at=data["started_at"],
        )
        for evt_data in data.get("events", []):
            journal.record(
                event_type=evt_data["event_type"],
                tool_name=evt_data["tool_name"],
                solver=evt_data.get("solver", ""),
                physics=evt_data.get("physics", ""),
                details=evt_data.get("details", {}),
                error_message=evt_data.get("error_message", ""),
                notes=evt_data.get("notes", ""),
            )
        return journal

    def reset(self):
        """Clear all events. Useful for testing."""
        self.events.clear()
        self._solvers_used.clear()
        self._physics_used.clear()


def _event_to_dict(evt: JournalEvent) -> dict[str, Any]:
    """Convert event to dict, dropping empty strings and empty dicts to save space.

    Preserves numeric zeros and False values (only drops '' and {}).
    """
    d = asdict(evt)
    return {k: v for k, v in d.items() if v != "" and v != {}}


# ── Singleton ────────────────────────────────────────────────

_journal: Optional[SessionJournal] = None


def get_journal() -> SessionJournal:
    """Get the session-global journal (singleton per process)."""
    global _journal
    if _journal is None:
        _journal = SessionJournal()
    return _journal


def reset_journal() -> SessionJournal:
    """Reset the global journal. For testing only."""
    global _journal
    _journal = SessionJournal()
    return _journal
