"""Enforced critic review.

THE HOLE THIS CLOSES
--------------------
openPASO's run tools took ``critic_approved: bool`` — a value the AGENT passes.
The critic sub-agent lives in the client harness, and the server never observed
it: there is no spawn record, no review log, and nothing in ``src/`` references
``spawn_subagent`` at all. So an agent could call

    run_simulation(..., critic_approved=True)

having never spawned a critic, and the verification gate would stamp the result
trustworthy. The "MANDATORY CRITIC" in the server instructions was a request,
not an enforcement — and a request is exactly what an unreliable model ignores.

WHAT IS ENFORCED NOW
--------------------
A review must EXIST, be BOUND to the exact setup being run, and be ON RECORD:

  * the critic's findings are submitted to the server, which digests the setup
    they reviewed and issues a token;
  * a run tool accepts a token only if it is known to the server, unexpired,
    unused, and its digest matches the deck actually being executed;
  * changing so much as a character of the deck after review invalidates the
    token, so "review a clean setup, then run a different one" is blocked;
  * every review is retained for audit with its findings.

WHAT THIS DOES NOT DO
---------------------
The server cannot judge whether the critique was any *good* — it is not an
oracle for review quality. It enforces that a substantive review of THIS setup
happened and is auditable. That converts an unverifiable self-report into a
checkable fact, which is the part that can be enforced in software.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

# A review must say something. An empty or one-word "looks fine" is refused,
# because it is indistinguishable from no review at all.
MIN_FINDINGS_CHARS = 80
DEFAULT_TTL_S = 3600.0


class CriticGateError(Exception):
    """Raised when a run is attempted without a valid, matching review."""


def setup_digest(*parts: str) -> str:
    """Stable digest of everything that defines the run being reviewed."""
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


@dataclass
class ReviewRecord:
    token: str
    digest: str
    solver: str
    findings: str
    created: float
    ttl_s: float = DEFAULT_TTL_S
    # THE TEXT THE DIGEST WAS TAKEN OF, so a run whose digest does not match
    # can say WHICH PART of the setup changed instead of only that something
    # did. Measured 2026-09-19: ten of ten coupled runs were told "a critic
    # review exists for this solver but NOT for this setup" with no way for
    # the caller -- or for us reading the record afterwards -- to see what
    # differed. What a mismatch reply shows of it: for a coupling setup, the
    # changed argument's name and its short reviewed and current values (scalars
    # such as tol or max_iter; a long value is only named); for a script or deck,
    # the NUMBER of the first differing line and the caller's current line, never
    # a stored line, which may hold private literals. The audit log keeps its
    # length, not its text.
    setup_text: str = ""
    consumed_by: str | None = None
    consumed_at: float | None = None
    # A REJECTION IS KEPT, NOT DROPPED. Refusing a rejected review left no trace of it, so a
    # working agent could ask one critic after another until one approved the same text, and
    # the record showed only the approval (both peer sessions named this, 2026-10-01). A
    # rejection issues no token; it blocks that exact text until the text changes.
    rejected: bool = False
    # the file the review named, when it named one: rejections of earlier versions of the
    # same file are counted beside a later approval
    source_name: str = ""

    def expired(self, now: float | None = None) -> bool:
        return (now or time.time()) > self.created + self.ttl_s


class CriticRegistry:
    """Server-side record of critic reviews. The agent cannot forge entries
    because it never holds the store; it can only ask for a token."""

    def __init__(self, audit_path: Path | None = None):
        self._reviews: dict[str, ReviewRecord] = {}
        self._audit = Path(audit_path) if audit_path else None

    # ── issuing ──────────────────────────────────────────────────────────
    def submit_review(self, *, solver: str, findings: str,
                      digest: str, ttl_s: float = DEFAULT_TTL_S,
                      setup_text: str = "", rejected: bool = False,
                      source_name: str = "") -> ReviewRecord:
        text = (findings or "").strip()
        if len(text) < MIN_FINDINGS_CHARS:
            raise CriticGateError(
                f"review refused: findings are {len(text)} characters; a review "
                f"must state what was actually checked (>= {MIN_FINDINGS_CHARS}). "
                f"An empty approval is indistinguishable from no review.")
        rec = ReviewRecord(token=secrets.token_urlsafe(24), digest=digest,
                           solver=solver, findings=text, created=time.time(),
                           ttl_s=ttl_s, setup_text=setup_text or "",
                           rejected=bool(rejected), source_name=source_name or "")
        self._reviews[rec.token] = rec
        self._append_audit("review_submitted", rec)
        return rec

    # ── redeeming ────────────────────────────────────────────────────────
    def consume(self, token: str | None, *, digest: str, solver: str,
                job_id: str = "") -> ReviewRecord:
        """Validate a token against the run actually being executed.

        Raises CriticGateError with a specific reason on any mismatch; the
        caller turns that into a NOT VERIFIED verdict.
        """
        if not token:
            raise CriticGateError(
                "no critic review: openPASO treats no result as trustworthy until "
                "an independent critic has reviewed this setup. Submit the "
                "critic's findings to obtain a review token, then re-run.")
        rec = self._reviews.get(token)
        if rec is None:
            raise CriticGateError(
                "critic review token is not known to this server; it cannot be "
                "self-issued.")
        if rec.rejected:
            raise CriticGateError("that review REJECTED the setup; it issues no approval.")
        if rec.expired():
            raise CriticGateError("critic review has expired; review the setup again.")
        if rec.consumed_by is not None:
            raise CriticGateError(
                f"critic review was already used for job {rec.consumed_by}; "
                f"each run needs its own review.")
        if rec.solver != solver:
            raise CriticGateError(
                f"critic review was for solver '{rec.solver}', not '{solver}'.")
        if rec.digest != digest:
            raise CriticGateError(
                "critic review does not match the setup being run: the input "
                "changed after it was reviewed. Review the setup you intend to "
                "run.")
        rec.consumed_by = job_id or "unnamed-job"
        rec.consumed_at = time.time()
        self._append_audit("review_consumed", rec)
        return rec

    # ── audit ────────────────────────────────────────────────────────────
    def _append_audit(self, event: str, rec: ReviewRecord) -> None:
        if self._audit is None:
            return
        try:
            self._audit.parent.mkdir(parents=True, exist_ok=True)
            with open(self._audit, "a", encoding="utf-8") as fh:
                payload = asdict(rec)
                # the setup's own text stays out of the log (a script, a deck: whatever the
                # caller ran); its length says enough to tell two reviews apart (Copilot on the
                # org PR: the field was called server-side only and was written out whole)
                payload["setup_chars"] = len(payload.pop("setup_text", "") or "")
                payload["event"] = event
                payload["at"] = time.time()
                fh.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception:
            pass          # auditing must never break a run

    def records(self) -> list[ReviewRecord]:
        return list(self._reviews.values())
