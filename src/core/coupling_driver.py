"""General, physics-agnostic partitioned coupling driver.

Replaces the overfit `problem=`-enum coupled_solve. The driver owns ONLY the
iteration math (data exchange + relaxation + convergence). It knows nothing about
heat/elasticity/flux, no geometry, no benchmark answer. Physics lives entirely in
the participant scripts (which the agent writes) and the data currency is
InterfaceData JSON (file handshake).

Contract for a PARTICIPANT (any solver, any code, any physics):
  It is a runnable command. Each iteration the driver:
    1. writes <work_dir>/imports.json  = the InterfaceData this participant must
       consume this iteration (boundary values from its coupling partners), or
       an empty file on iteration 0.
    2. runs the participant command in <work_dir>.
    3. reads <work_dir>/exports.json   = the InterfaceData the participant produced
       on the shared interface (whatever quantities it exports — opaque to driver).
  The participant decides HOW to apply imports (Dirichlet, Neumann, Robin, traction,
  flux, concentration, ...) and WHAT to export. The driver treats both as opaque
  numbers on coordinates -> works for ANY coupling.

Convergence is on the stacked export-vector change between iterations. If it does
not converge within max_iter, the driver returns success=False LOUDLY (the most
general silent-wrong guard: never frame a non-converged run as a result).

WHAT A PARTITIONED COUPLING GETS WRONG QUIETLY, and what this driver records so
the validators can catch it (each of these was demonstrated against the previous
version of this file, which reported several of them as a clean convergence):

  * a participant that exits NON-ZERO but leaves an exports.json behind — the
    iteration used to continue on the output of a crashed solve;
  * a participant that never reads imports.json (or reads a stale copy of its own
    last answer) — its export never moves, the residual is zero at iteration 2 and
    the coupling "converges" instantly to the initial condition;
  * a partner name in `imports_from` that matches no participant (a typo) — the
    edge used to be dropped silently, turning a two-way coupling into a one-way
    one with no trace in the output;
  * an export whose length changes between iterations — the relaxation reshape
    either broadcast silently or raised out of the driver;
  * a residual computed as ONE relative norm over every participant and every
    block at once, so a large, quickly-settled block (forces, fluxes) masks a
    small one (displacements, temperatures) that is still moving.

None of these is decided here: the driver records evidence
(`returncodes`, `responsiveness`, `block_residuals`, `graph`, `theta`) and
core.quality_checks turns it into verdict-bearing findings. Enforcement is always
by verdict, never by exception.

STOCHASTIC PARTICIPANTS HAVE A RESIDUAL FLOOR, and the driver knows about it.
A Monte-Carlo participant (DSMC, any sampled estimator) returns a slightly
different export every time it is asked the SAME question. The residual is the
change in the export vector, so it cannot fall below the size of that sampling
scatter no matter how well the physics has settled — and a `tol` underneath the
floor therefore ends every run as "did not converge", on a coupling that is
right. That verdict is honest but useless, and it is the reason a stochastic
coupling could not be assessed on convergence at all.

`noise_replicates` / `noise_floor` fix it without ever softening the guard:

  * the floor is MEASURED, not assumed. With `noise_replicates=N` the driver
    runs each participant N times on the SAME imports and evaluates its OWN
    residual expression across independent replicates. That number IS the
    residual a perfectly converged run would still report. Use N >= 4: the
    floor is itself an estimate and three samples is a bad one;
  * convergence is then declared against `max(tol, floor)` — never below the
    floor, never above the tolerance the caller asked for;
  * the stopping statistic becomes a BLOCK MEAN of the last `noise_block`
    residuals once a floor is in play, so a single lucky dip into the noise
    does not end the run;
  * `noise_floor` on the result is what any check of it must use: a tolerance
    tighter than the floor is measuring the sampler, not the coupling;
  * a floor measured as exactly zero is REPORTED, because for a Monte-Carlo
    participant that means a fixed seed, and a residual that falls under a
    fixed seed proves only that the same draw was repeated.

The feature is inert for deterministic participants: their replicates are
bit-identical, the floor is 0, and `max(tol, 0) == tol`.
"""
from __future__ import annotations
import hashlib
import json
import re
import os
import tempfile
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# The exports.json each participant held when a coupling this process ran CONVERGED, by
# path, with its digest. A later coupling warm-starts only from these: a file written by a
# standalone run of the participant (its fallback imports, another level's constants) is
# not an interface state of this coupling.
_COUPLED_EXPORTS: dict = {}
# ... and the n_steps that coupling's level stated for that participant (None: not stated),
# so a later level can tell a time axis in the export's width from vector components.
_COUPLED_STEPS: dict = {}


def _stated_n_steps(p) -> Optional[int]:
    """The n_steps a participant's level states: its ./config.json with the level's
    OPENPASO_CONFIG_JSON merged over it, as the served transient contracts read them."""
    cfg: dict = {}
    try:
        c = Path(p.work_dir) / "config.json"
        if c.is_file():
            d = json.loads(c.read_text() or "{}")
            if isinstance(d, dict):
                cfg.update(d)
    except (OSError, ValueError):
        pass
    try:
        d = json.loads((getattr(p, "env", None) or {}).get("OPENPASO_CONFIG_JSON") or "{}")
        if isinstance(d, dict):
            cfg.update(d)
    except (TypeError, ValueError):
        pass
    try:
        return int(cfg["n_steps"]) if cfg.get("n_steps") is not None else None
    except (TypeError, ValueError):
        return None


def _seed_misfit(seed, stated, old_steps) -> str:
    """'' when a stored export fits this level; else what does not fit, in a few words.

    A time-history export holds one column per step of the window it was run with. When
    this level states another n_steps, the partner's trace-length guard stops the run at
    iteration 1 (measured: three level runs of one ladder whose n_steps doubled with the
    mesh). A width that did not track the old n_steps (vector components) is left alone."""
    widths = set()
    for arr in (seed.values, seed.normal_fluxes):
        if arr is None:
            continue
        a = np.asarray(arr, float)
        if a.size:
            widths.add(int(a.shape[1]) if a.ndim >= 2 else 1)
    for m in sorted(widths):
        for n in sorted(stated):
            if m != n and (old_steps is None or m == old_steps):
                return f"{m} column(s) per point; this level's n_steps is {n}"
    return ""


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""

import numpy as np

from core.field_transfer import InterfaceData

# How many per-iteration non-finite warnings to keep before collapsing the rest
# into a count. An unbounded list turned a single NaN into ~80 near-identical
# lines, which buries every other finding in the validation block.
_MAX_NONFINITE_WARNINGS = 4
# Non-finite exports of one participant, in a row, that end the coupling. One is enough:
# the residual and the relaxation carry a NaN into every later iteration (measured: a
# single non-finite first pass, with finite exports ever after, still ends 150 iterations
# later in "did not converge ... last residual nan").
_NONFINITE_STOP = 1
_MAX_PARTICIPANT_STREAM_BYTES = 3_500_000


@dataclass
class Participant:
    """One coupled solver. `command` reads imports.json / writes exports.json in work_dir."""
    name: str
    command: list[str]        # e.g. ["python", "subdomain_A.py"] or ["/path/4C", "deckB.yaml", "out"]
    work_dir: Path
    # which partner-export this participant imports (edge): partner_name -> None (take its export)
    imports_from: list[str] = field(default_factory=list)
    timeout: int = 3600
    # Files the solver needs in work_dir (species/surface/mesh/config data).
    # Staged (copied in) once before the iteration loop. A missing file is a
    # LOUD setup error — the alternative is the solver dying mid-iteration
    # with an opaque 'Cannot open ...' (the SPARTA failure mode).
    data_files: list[str] = field(default_factory=list)
    # Extra process environment for THIS participant, merged over the server's.
    # Data plumbing only (the multi-level call passes each level's mesh keys this
    # way); openPASO writes no file of the agent's to do it.
    env: dict | None = None


@dataclass
class CouplingResult:
    converged: bool
    iterations: int
    residual: float
    exports: dict[str, dict]          # name -> InterfaceData.to_dict()
    history: list[float]
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    # ── evidence the validators consume (never interpreted here) ──────────
    # participant name -> exit code of its LAST run
    returncodes: dict[str, int] = field(default_factory=dict)
    # "<participant>.values" / "<participant>.normal_fluxes" -> relative change
    # of that block alone on the final iteration. A global norm hides a small
    # block behind a large one; these do not.
    block_residuals: dict[str, float] = field(default_factory=dict)
    # the same blocks' fixed-point residuals: raw output against the relaxed input
    block_fixed_point: dict[str, float] = field(default_factory=dict)
    # and each block's estimated distance to its fixed point, from its last two steps
    block_distance: dict[str, float] = field(default_factory=dict)
    # each block's last change against its own largest value, and its tiny entries' against
    # theirs (see _scaled_change)
    block_scale_change: dict[str, float] = field(default_factory=dict)
    block_tiny_change: dict[str, float] = field(default_factory=dict)
    # participant -> "responsive" | "unresponsive" | "imports never changed"
    #              | "no imports declared"
    responsiveness: dict[str, str] = field(default_factory=dict)
    # for an "unresponsive" participant: {"changed": [...], "unchanged": [...]}, the
    # imported columns ("<partner>.<key>") that moved, and those that never moved, on
    # the iterations where its imports changed and its export did not
    responsiveness_detail: dict = field(default_factory=dict)
    # the coupling graph as the driver resolved it (declared vs. actually wired)
    graph: dict = field(default_factory=dict)
    # relaxation actually applied: {"mode": "aitken"|"constant", "theta0": float,
    #                               "applied": float}
    theta: dict = field(default_factory=dict)
    # participant -> finite-difference interface sensitivity (see
    # probe_interface_sensitivity), or None where it could not be measured
    sensitivity: dict = field(default_factory=dict)
    # ── the stochastic branch ─────────────────────────────────────────────
    # Stochastic branch. `noise_floor` is None when no floor was measured or
    # declared; 0.0 means it WAS established and came out zero, which is a
    # different statement and the reason these are not collapsed.
    noise_floor: Optional[float] = None
    tol_effective: Optional[float] = None
    stopped_at_noise_floor: bool = False
    # PROVENANCE, not warnings. How the floor was measured, and the fixed-seed
    # caveat, belong next to the number rather than in `validation` — the tool
    # copies `warnings` into `validation`, and an agent is told that an empty
    # validation block is what a correct coupling looks like. Putting a routine
    # measurement note there would make every stochastic-aware run look flagged
    # and every deterministic one that merely asked for a floor look flagged
    # too.
    notes: list[str] = field(default_factory=list)
    # THE CRITERION ACTUALLY APPLIED — a third channel, and it exists because
    # the other two are both wrong for it. "This run was judged at the measured
    # noise floor rather than at your tol" is not provenance (an agent MUST see
    # it, and must not apply a tighter tolerance) and it is not a finding
    # either (the coupling is correct). It sat in `warnings` when this branch
    # was written, which was harmless there because the tool then decided
    # trustworthiness with a keyword filter that these words happened to miss.
    # The tool now takes any finding at all as untrustworthy — deliberately, so
    # that no check can be lost by rewording — and under that rule a warning
    # here stamps NOT VERIFIED on every correct stochastic coupling, which is
    # the exact verdict this whole branch exists to stop being unavoidable.
    # So it gets its own list, and the tool reports it in the coverage channel
    # that is always printed and never flips the verdict.
    criterion_notes: list[str] = field(default_factory=list)


def _stack(ifd: InterfaceData) -> np.ndarray:
    v = np.asarray(ifd.values, float).ravel()
    if ifd.normal_fluxes is not None:
        v = np.concatenate([v, np.asarray(ifd.normal_fluxes, float).ravel()])
    return v


def _participant_env(p: 'Participant'):
    """The subprocess environment for one participant: the server's plus its own `env`."""
    if not getattr(p, 'env', None):
        return None
    merged = dict(os.environ)
    merged.update({str(k): str(v) for k, v in p.env.items()})
    return merged


def _relax(prev: np.ndarray, new: np.ndarray, theta: float) -> np.ndarray:
    return (1 - theta) * prev + theta * new


def _aitken(prev_relaxed, new_raw, res_prev, theta_prev, lo=0.05, hi=1.0):
    """Aitken dynamic relaxation on the GLOBAL interface residual r_k = G(x_k) - x_k.

    theta_k = -theta_{k-1} * (r_{k-1} . (r_k - r_{k-1})) / ||r_k - r_{k-1}||^2

    Two things this function used to get wrong, both of which cost convergence on
    correct setups (a false 'NOT CONVERGED' on a good coupling is as corrosive as
    a missed silent-wrong):

      * `res_prev` MUST be the previous RESIDUAL r_{k-1}. It used to be handed the
        previous RAW EXPORT G(x_{k-1}), which makes theta an arbitrary number in
        [lo, hi] with no relation to the iteration.
      * there must be ONE theta for the whole interface state. Aitken's derivation
        is for a single scalar sequence extrapolated from the composite fixed-point
        map; giving each participant its own theta relaxes the two halves of one
        coupled system by different amounts, which on a Dirichlet-Neumann split
        drove the two thetas apart (to the clamp at either end) and made the
        iteration diverge where constant relaxation converged.

    Returns (theta, r_k) — the caller must store r_k and hand it back next time.
    """
    r_new = new_raw - prev_relaxed
    if res_prev is None or res_prev.shape != r_new.shape:
        return min(max(theta_prev, lo), hi), r_new
    dr = r_new - res_prev
    denom = float(np.dot(dr, dr))
    if not np.isfinite(denom) or denom < 1e-30:
        return min(max(theta_prev, lo), hi), r_new
    theta = -theta_prev * float(np.dot(res_prev, dr)) / denom
    if not np.isfinite(theta):
        return min(max(theta_prev, lo), hi), r_new
    return min(max(theta, lo), hi), r_new


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def _bounded_stream(text: str) -> str:
    """Keep both ends of large solver output below the 8 MB cap on captured
    output."""
    raw = (text or "").encode("utf-8", errors="replace")
    if len(raw) <= _MAX_PARTICIPANT_STREAM_BYTES:
        return raw.decode("utf-8", errors="replace")
    half = _MAX_PARTICIPANT_STREAM_BYTES // 2
    omitted = len(raw) - 2 * half
    return (raw[:half].decode("utf-8", errors="replace")
            + f"\n[... {omitted} bytes omitted ...]\n"
            + raw[-half:].decode("utf-8", errors="replace"))


# A STOP KEEPS ITS NAME. The tail was the last 300 characters of a participant's stderr, and a
# served check's message runs longer: measured, couple()'s lead quoted a solve self-check from
# mid-word ("ale ...") without the check's name, and the run read it as a check of ours that was
# "too strict". The tail now starts at the last line that names a stop (a capitalised label and a
# colon, or a Python exception line) when that line starts within the last `most` characters.
_STOP_LINE = re.compile(r"^(?!NOTE\b|WARNING\b)(?:[A-Z][A-Z0-9_ ()/'-]{2,60}:|\w*(?:Error|Exception):|"
                        r"SPARTA failed)", re.M)


def _stderr_tail(text: str, n: int = 300, most: int = 1500) -> str:
    """The end of a participant's stderr: from the last line that names a stop, when that line
    starts within the last `most` characters, else the last `n` characters."""
    text = text or ""
    hits = [m for m in _STOP_LINE.finditer(text) if m.start() >= len(text) - most]
    return text[hits[-1].start():] if hits else text[-n:]


def _output_tail(r) -> str:
    """The stderr tail, and the stdout tail when stderr carried nothing. Measured on a coupled
    elastic round: a participant that printed the solver's console (FEBio's error box) to stdout
    and exited 1 was reported twice as "stderr tail: ''", the stop it printed nowhere in sight."""
    err = _stderr_tail(r.stderr)
    if (r.stderr or "").strip() or not (r.stdout or "").strip():
        return err
    return f"{err} -- stderr was empty; stdout tail: {_stderr_tail(r.stdout)}"


def _persist_participant_output(p: Participant, result, iteration: int) -> None:
    """Atomically retain the latest native process output for attribution."""
    log = p.work_dir / "participant_output.log"
    tmp = p.work_dir / ".participant_output.log.tmp"
    payload = (
        f"iteration: {iteration}\n"
        f"command: {json.dumps(p.command)}\n"
        f"returncode: {result.returncode}\n"
        "--- stdout ---\n"
        f"{_bounded_stream(result.stdout)}\n"
        "--- stderr ---\n"
        f"{_bounded_stream(result.stderr)}\n"
    )
    tmp.write_text(payload, encoding="utf-8", errors="replace")
    tmp.replace(log)


def _rel_change(new: np.ndarray, prev: np.ndarray) -> float:
    """Relative change of one block, taken ENTRY BY ENTRY.

    A block norm is itself a scale-masking device: an interface vector that
    happens to hold a huge entry and a small one in the same flat array (mixed
    quantities in one `values` list — very common) has its norm set entirely by
    the huge entry, so the small entry can oscillate by 100% of itself with the
    block residual reading 1e-12. Demonstrated: a two-entry export whose second
    component's true fixed point was 100 converged at iteration 2 reporting 1.5,
    with every check silent.

    So: the reported number is the WORST entry-wise relative change, not the
    norm ratio. Entries that are zero to within the block's own dynamic range
    (below 1e-13 of the largest entry) are skipped — their relative change is
    numerical noise, not information. Returns NaN if it cannot be formed.
    """
    if new.shape != prev.shape or new.size == 0:
        return float("nan")
    scale = max(float(np.max(np.abs(new))) if new.size else 0.0,
                float(np.max(np.abs(prev))) if prev.size else 0.0)
    if scale <= 0:
        return 0.0
    mag = np.maximum(np.abs(new), np.abs(prev))
    live = mag > 1e-13 * scale
    if not np.any(live):
        return 0.0
    return float(np.max(np.abs(new[live] - prev[live]) / mag[live]))


def _scaled_change(new: np.ndarray, prev: np.ndarray, own_below: float = 1e-6) -> tuple:
    """(the largest change of a block's entries against the block's largest value, the largest
    change of its TINY entries against their own values); NaN where it cannot be formed.

    THE WORST ENTRY AGAINST ITSELF IS NOT THE BLOCK'S PRECISION. An entry of one field that
    holds a small share of its block's largest value -- an interface end, a flux passing through
    zero -- reads a large relative change for an absolute one at the iteration's tolerance, and a
    finer mesh puts an entry nearer zero. Measured on a right steady ladder: 1.1e-6, 3.7e-6 and
    1.74e-4 at levels 1-3 from an entry holding 0.9 % of its block's largest value, and the ladder
    read NOT VERIFIED three rounds running. So each entry is measured against the block's largest
    value -- except an entry more than six orders below it, which is taken as another quantity in
    the same array (a flat list of mixed quantities, the case _rel_change's entry-by-entry measure
    exists for) and measured against itself. Entries zero to within the block's own range (below
    1e-13 of the largest) are skipped, as there."""
    if new.shape != prev.shape or new.size == 0:
        return float("nan"), float("nan")
    scale = max(float(np.max(np.abs(new))), float(np.max(np.abs(prev))))
    if scale <= 0:
        return 0.0, 0.0
    mag = np.maximum(np.abs(new), np.abs(prev))
    d = np.abs(new - prev)
    tiny = (mag < own_below * scale) & (mag > 1e-13 * scale)
    on_scale = float(np.max(d[~tiny])) / scale if np.any(~tiny) else 0.0
    own = float(np.max(d[tiny] / mag[tiny])) if np.any(tiny) else 0.0
    return on_scale, own


def _blocks(ifd: InterfaceData) -> dict[str, np.ndarray]:
    """Split an export into the pieces that must each converge on their own.

    Components matter, not just arrays: a TSI or FSI interface carries
    temperature and displacement (or force and displacement) inside ONE `values`
    array of shape (N, n_comp), on scales that differ by many orders of
    magnitude. Lumping them into a single norm is exactly how the small one stops
    being visible.
    """
    b: dict[str, np.ndarray] = {}
    for label, arr in (("values", ifd.values),
                       ("normal_fluxes", ifd.normal_fluxes)):
        if arr is None:
            continue
        a = np.asarray(arr, float)
        if a.ndim >= 2 and a.shape[-1] > 1:
            for c in range(a.shape[-1]):
                b[f"{label}[{c}]"] = a[..., c].ravel()
        else:
            b[label] = a.ravel()
    return b


def _invoke(p: Participant, imp: dict) -> tuple[Optional[InterfaceData], Optional[str]]:
    """Write imports, run the participant once, read its export back.

    Returns (InterfaceData, None) or (None, error). This is the NOISE-FLOOR
    MEASUREMENT's way of driving a participant, and it deliberately reproduces
    the iteration loop's own path byte for byte where that is observable from
    inside the participant: the same `sort_keys=True` serialisation of
    imports.json, the same deletion of any stale exports.json, the same refusal
    of a non-zero exit code. A floor measured by a different mechanism than the
    residual it is compared against would not be a floor for that residual.

    It is NOT used by the loop itself, which carries evidence collection
    (returncodes, digests, block residuals, length and emptiness refusals) that
    a replicate run has no use for and that would only be recorded twice.
    """
    (p.work_dir / "imports.json").write_text(
        json.dumps(imp, indent=2, sort_keys=True))
    ep = p.work_dir / "exports.json"
    if ep.exists():
        ep.unlink()
    try:
        r = subprocess.run(p.command, cwd=str(p.work_dir), env=_participant_env(p), capture_output=True,
                           text=True, timeout=p.timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, f"participant {p.name} timed out"
    if not ep.exists():
        return None, (f"participant {p.name} wrote no exports.json "
                      f"(rc={r.returncode}). stderr tail: {_output_tail(r)}")
    # Same rule as the loop: a solver that writes its last iterate and then
    # aborts has not answered the question, and a floor measured across crashed
    # runs is a floor on the crash, not on the sampling.
    if r.returncode != 0:
        return None, (f"participant {p.name} exited with code {r.returncode} "
                      f"during a replicate run; its exports.json is the output "
                      f"of a FAILED run and cannot define a noise floor. "
                      f"stderr tail: {_output_tail(r)}")
    try:
        return InterfaceData.from_json(ep), None
    except Exception as e:
        return None, f"participant {p.name} bad exports.json: {e}"


def _residual_of(new: dict[str, np.ndarray],
                 prev: dict[str, np.ndarray]) -> float:
    """The driver's own residual expression, on two sets of export vectors.

    Kept as one function so the convergence test and the noise-floor
    measurement cannot drift apart.
    """
    num = 0.0
    ref = 0.0
    for n, v in new.items():
        num += float(np.sum((v - prev[n]) ** 2))
        ref += float(np.sum(v ** 2)) + 1e-30
    return float(np.sqrt(num / ref))


def _repeat_probe(participants: list[Participant], last_imports: dict[str, str],
                  raw_last: dict[str, np.ndarray]) -> list[str]:
    """One more run of each participant on the imports its last iteration read,
    compared with the export that iteration produced. Returns a finding per
    participant whose answer changed; [] when every one repeats (or cannot be
    re-run -- a failed probe says nothing)."""
    out: list[str] = []
    for p in participants:
        ref = raw_last.get(p.name)
        text = last_imports.get(p.name)
        if ref is None or text is None:
            continue
        try:
            imp = json.loads(text)
        except (TypeError, ValueError):
            continue
        ifd, err = _invoke(p, imp)
        if err or ifd is None:
            continue
        got = _stack(ifd)
        if got.size != ref.size:
            out.append(f"PARTICIPANT {p.name} IS NOT A FUNCTION OF ITS IMPORTS: run again on "
                       f"the imports its last iteration read, it exported {got.size} numbers "
                       f"where that iteration exported {ref.size}.")
            continue
        scale = max(float(np.max(np.abs(ref))) if ref.size else 0.0, 1e-300)
        dev = float(np.max(np.abs(got - ref))) / scale if ref.size else 0.0
        if dev > 1e-9:
            out.append(
                f"PARTICIPANT {p.name} IS NOT A FUNCTION OF ITS IMPORTS: run again on the "
                f"imports its last iteration read, it returned an export {dev:.2%} off that "
                f"iteration's (largest difference over its largest value), and no coupling "
                f"converges below that scatter, whatever the relaxation. If it samples by "
                f"design (a Monte-Carlo or particle method), re-run with noise_replicates>=2 "
                f"so the driver measures that floor and judges against it. A finite-element "
                f"solve repeats bit for bit, so there the scatter is a defect: look for state "
                f"its inputs do not set -- memory allocated but never initialised (a mask or "
                f"vector built at a size and only partly written), a list where a full-length "
                f"array belongs, an unseeded random draw, a file left by an earlier run.")
    return out


def _measure_noise_floor(participants: list[Participant], replicates: int,
                         imports_for: dict[str, dict], where: str,
                         against: Optional[dict] = None
                         ) -> tuple[Optional[float], Optional[str], list[str]]:
    """Run every participant `replicates` times on the SAME imports and report
    the residual the driver would still see between independent answers.

    Returns (floor, error, notes). `floor` is None only when the measurement
    could not be made at all.

    WHY REPLICATES AND NOT A STANDARD DEVIATION. The quantity that has to be
    beaten is the residual, and the residual is a normalised difference of two
    export vectors. So the floor is that same expression, evaluated on
    independent answers to one question — no distributional assumption, no
    conversion factor, and directly comparable to the number the loop reports.

    `against` decides WHICH comparison, and the two are genuinely different:

      * `against=<the loop's relaxed_prev>` is the faithful one. Each replicate
        is evaluated against the very vector the loop compares to, so the value
        IS the residual the loop reports, measured several times.
      * `against=None` (before the loop, where no relaxed_prev exists)
        evaluates replicates against EACH OTHER. That is a LOWER BOUND, not the same
        number. It was first written down here as conservative on the argument
        that the relaxed blend averages noise down — and measuring it showed the
        opposite: the relaxed vector is a lagged average carrying its own
        accumulated noise, and noise that has propagated through a partner over
        earlier iterations is missing from the pairwise figure entirely. On one
        two-participant case the pairwise estimate came out around a third of
        the faithful one. It is used to avoid a pointless long run, never as the
        final word.
    """
    notes: list[str] = []
    draws: list[dict[str, np.ndarray]] = []
    for k in range(replicates):
        one: dict[str, np.ndarray] = {}
        for p in participants:
            ifd, err = _invoke(p, imports_for.get(p.name, {}))
            if err:
                return None, f"noise-floor measurement ({where}): {err}", notes
            one[p.name] = _stack(ifd)
        if draws and any(one[n].size != draws[0][n].size for n in one):
            return None, (f"noise-floor measurement ({where}): a participant "
                          f"changed its export size between replicate runs, so "
                          f"no floor can be defined for it"), notes
        draws.append(one)
    if against is not None:
        # THE FAITHFUL MEASUREMENT, available only once the loop has a previous
        # relaxed vector to compare against: repeat the iteration N times from
        # the same state and take the residual it ACTUALLY REPORTS each time.
        # Nothing is modelled — this is the same expression on the same two
        # operands the loop uses.
        vals = [_residual_of(d, against) for d in draws]
        n_ind = len(vals)
    else:
        # PRE-LOOP there is no previous relaxed vector, so the only thing
        # available is the scatter BETWEEN independent answers. That is a LOWER
        # BOUND on the loop's floor and not the same number: in the loop the
        # comparison is against a lagged relaxed average which carries its own
        # accumulated noise, and noise that has propagated through a partner
        # over earlier iterations is not in this at all. It is used to avoid a
        # pointless long run; the verdict is re-judged against the faithful
        # measurement if the loop ends un-converged.
        pairs = [(i, j) for i in range(replicates)
                 for j in range(i + 1, replicates)]
        vals = [_residual_of(draws[i], draws[j]) for i, j in pairs]
        n_ind = len(pairs)
    floor = float(sum(vals) / len(vals))
    notes.append(f"noise floor {floor:.3e} from {replicates} replicate runs "
                 f"per participant ({n_ind} samples, {where})")
    # THE FLOOR IS ITSELF AN ESTIMATE, and at three replicates it is a bad one.
    # Measured here: the same coupling gave 1.2e-03 from three replicates and
    # 9.6e-03 from five — a factor of eight, from estimator scatter alone, on a
    # quantity the convergence verdict is compared against. Three replicates
    # give only three pairs and they are not independent. Say so rather than let
    # a lucky low draw make the criterion look tighter than it is.
    if n_ind < 6:
        notes.append(
            f"this floor rests on only {n_ind} replicate samples and is "
            f"itself a noisy estimate — raise noise_replicates to 4 or more "
            f"(6+ pairs) before relying on the number, especially before using "
            f"it as an acceptance tolerance")
    return floor, None, notes


class _Anderson:
    """Anderson mixing (type II, window m) on the whole interface state -- the interface
    quasi-Newton family partitioned-coupling libraries use. With x the relaxed state handed
    to the participants and g = G(x) their new exports, the residual is f = g - x; the next
    state is g_k - dG gamma with gamma the least-squares solution of dF gamma = f_k over the
    last m differences. Falls back to constant relaxation until two residuals exist. Measured
    reason for existing: a three-component thermo-elastic exchange took 28-60 Aitken iterations
    per level (two 4C runs each); the wall clock, not the physics, ended those runs.
    """

    def __init__(self, m: int = 5, beta: float = 0.5):
        self.m, self.beta = int(m), float(beta)
        self.xs: list = []
        self.gs: list = []

    def step(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        self.xs.append(np.asarray(x, float).copy())
        self.gs.append(np.asarray(g, float).copy())
        if len(self.xs) > self.m + 1:
            self.xs.pop(0)
            self.gs.pop(0)
        f = [gi - xi for xi, gi in zip(self.xs, self.gs)]
        if len(f) < 2:
            return (1.0 - self.beta) * x + self.beta * g
        dF = np.column_stack([f[i + 1] - f[i] for i in range(len(f) - 1)])
        dG = np.column_stack([self.gs[i + 1] - self.gs[i] for i in range(len(f) - 1)])
        try:
            gamma, *_ = np.linalg.lstsq(dF, f[-1], rcond=None)
        except np.linalg.LinAlgError:
            return (1.0 - self.beta) * x + self.beta * g
        x_new = g - dG @ gamma
        if not np.all(np.isfinite(x_new)):
            return (1.0 - self.beta) * x + self.beta * g
        return x_new


def run_coupling(participants: list[Participant], max_iter: int = 50,
                 tol: float = 1e-6, accelerator: str = "auto",
                 theta0: float = 0.5, probe: bool = True,
                 noise_floor: Optional[float] = None,
                 noise_replicates: int = 0,
                 noise_block: int = 3) -> CouplingResult:
    """Run a general fixed-point partitioned coupling. Physics-agnostic.

    Each iteration: every participant consumes its partners' latest exports (relaxed),
    runs, and produces new exports. Converges when the relaxed export-vector stops
    changing. Returns success=False if not converged within max_iter.

    probe: after the iteration settles, spend ONE extra solve per participant
        measuring whether its answer actually depends on its imports (see
        probe_interface_sensitivity). Turn it off only if that solve is
        genuinely unaffordable — the result then says the question was not asked.

    accelerator: "auto" (default: Aitken for a single-field exchange, Anderson for a
        multi-field one, resolved from the first exports), "aitken" (ONE dynamic theta for the whole interface state,
        recomputed each iteration from the previous residual, starting from
        theta0 — see _aitken), "anderson" (Anderson mixing / interface quasi-Newton
        on the whole interface state, window 5, theta0 as the mixing weight until
        two residuals exist — see _Anderson) or "constant" (theta fixed at theta0
        for the whole run). theta0 is the ONLY relaxation knob; there is no
        separate per-field or per-participant theta.

    STOCHASTIC PARTICIPANTS (see the module docstring):
      * `noise_replicates >= 2` measures the residual noise floor by running
        every participant that many times on the same imports, before the loop.
      * `noise_floor` declares one instead (or raises a measured one), for a
        caller who established it independently.
      * whichever is larger of `tol` and the floor becomes the convergence
        criterion, and once a non-zero floor is in play the stopping statistic
        is the mean of the last `noise_block` residuals rather than a single
        one, so one lucky dip into the noise cannot end the run.
      * if the loop still ends un-converged, the floor is RE-MEASURED at the
        final state — the pre-loop estimate is taken with the participants in
        their iteration-1 fallback, which need not carry the same relative
        scatter as the settled state — and the verdict is re-judged against it.
        That second measurement is paid only on failure.
    """
    names = [p.name for p in participants]
    # ── coupling graph: an unknown partner name is a SETUP error, not a no-op ──
    # `imports_from` used to be filtered with `if src in exports`, so a typo
    # ("Bee" for "B") silently deleted that edge and the run became a one-way
    # coupling that converged in a few iterations and reported nothing unusual.
    for p in participants:
        unknown = [s for s in p.imports_from if s not in names]
        if unknown:
            return CouplingResult(
                False, 0, float("nan"), {}, [],
                error=(f"participant {p.name}: imports_from names no such "
                       f"participant: {unknown} (participants are {names}). "
                       "Refusing to run: dropping the edge would silently turn "
                       "this into a one-way coupling."),
                graph={"participants": names,
                       "declared_edges": {q.name: list(q.imports_from)
                                          for q in participants}})
        if p.name in p.imports_from:
            return CouplingResult(
                False, 0, float("nan"), {}, [],
                error=f"participant {p.name}: imports_from includes itself.",
                graph={"participants": names,
                       "declared_edges": {q.name: list(q.imports_from)
                                          for q in participants}})
    graph = {"participants": names,
             "declared_edges": {p.name: list(p.imports_from) for p in participants}}

    # ── stage participant data files BEFORE the loop (loud on missing) ──
    for p in participants:
        p.work_dir.mkdir(parents=True, exist_ok=True)
        for df in p.data_files:
            src = Path(df).expanduser()
            if not src.is_file():
                return CouplingResult(
                    False, 0, float("nan"), {}, [],
                    error=f"participant {p.name}: data file not found: {df}",
                    graph=graph)
            dest = p.work_dir / src.name
            if dest.resolve() != src.resolve():
                shutil.copy(src, dest)

    exports: dict[str, InterfaceData] = {}      # latest relaxed exports per participant
    res_prev: dict[str, np.ndarray] = {}        # previous Aitken residual r_{k-1}
    relaxed_prev: dict[str, np.ndarray] = {}
    prev_blocks: dict[str, dict[str, np.ndarray]] = {}
    theta_global: float = theta0
    anderson = _Anderson(m=5, beta=theta0)
    # "auto": Aitken for a single-field exchange (measured on every scalar heat coupling that
    # reached CORRECT: 9-50 iterations), Anderson for a multi-field one (measured on the
    # thermo-elastic pair: 25 iterations against Aitken's 55). Resolved from the first exports.
    accelerator_requested = accelerator
    history: list[float] = []
    warnings: list[str] = []
    returncodes: dict[str, int] = {}
    block_residuals: dict[str, float] = {}
    block_fixed_point: dict[str, float] = {}
    block_distance: dict[str, float] = {}          # estimated distance to the fixed point
    block_scale_change: dict[str, float] = {}      # the last change on the block's own scale
    block_tiny_change: dict[str, float] = {}       # ... and of its tiny entries on their own
    step_hist: dict[str, list] = {}               # each block's last four steps
    # participant -> list of (imports digest, exports digest) per iteration
    trace: dict[str, list[tuple[str, str]]] = {p.name: [] for p in participants}
    last_imports: dict[str, str] = {}
    nonfinite_hits = 0
    nonfinite_run: dict[str, int] = {}          # participant -> consecutive non-finite exports
    # ── the stochastic branch's state, bound BEFORE _finish so every exit
    # carries it. `floor` and `tol_eff` are rebound below once the floor is
    # measured; _finish reads them at call time, so a late measurement is
    # reported by an early-written return path without any threading.
    notes: list[str] = []
    criterion_notes: list[str] = []
    floor: Optional[float] = None if noise_floor is None else float(noise_floor)
    tol_eff: float = tol
    block: int = 1
    at_floor: bool = False

    warm_seeded: list = []                  # filled at the warm start below; _finish reads it

    def _finish(**kw) -> CouplingResult:
        if kw.get("converged"):
            for _p in participants:
                _ep = _p.work_dir / "exports.json"
                if _ep.is_file():
                    _COUPLED_EXPORTS[str(_ep.resolve())] = _file_digest(_ep)
                    _COUPLED_STEPS[str(_ep.resolve())] = _stated_n_steps(_p)
        if warm_seeded and kw.get("iterations") == 1 and kw.get("error"):
            kw["error"] = (str(kw["error"]) + f" (iteration 1's imports were a WARM START seeded from the "
                           f"exports.json a previous coupling left in the work directories of "
                           f"{', '.join(warm_seeded)}, not this run's partner output)")
        kw.setdefault("warnings", warnings)
        kw.setdefault("returncodes", returncodes)
        kw.setdefault("block_residuals", block_residuals)
        kw.setdefault("block_fixed_point", block_fixed_point)
        kw.setdefault("block_distance", block_distance)
        kw.setdefault("block_scale_change", block_scale_change)
        kw.setdefault("block_tiny_change", block_tiny_change)
        kw.setdefault("responsiveness", _responsiveness(trace, participants))
        kw.setdefault("responsiveness_detail", _responsiveness_detail(trace, participants))
        kw.setdefault("graph", graph)
        rp = res_prev.get("*")
        kw.setdefault("theta", {"mode": accelerator, "theta0": theta0,
                                "applied": (theta_global if accelerator == "aitken"
                                            else theta0),
                                # Norm of the residual Aitken was last given. At
                                # convergence this is small BY DEFINITION — it is
                                # the fixed-point residual. Reporting it is what
                                # makes the bookkeeping checkable from outside:
                                # handing the formula the raw export instead (the
                                # bug this replaced) leaves a number of the size
                                # of the SOLUTION here, not of the residual.
                                "residual_norm": (None if rp is None
                                                  else float(np.linalg.norm(rp)))})
        kw.setdefault("notes", notes)
        kw.setdefault("criterion_notes", criterion_notes)
        kw.setdefault("noise_floor", floor)
        # Only reported once a floor is in play. Naming an effective tolerance
        # on a run that had none would invite anyone checking it to use it.
        kw.setdefault("tol_effective", None if floor is None else tol_eff)
        kw.setdefault("stopped_at_noise_floor", at_floor)
        return CouplingResult(**kw)

    # ── the noise floor, before anything iterates ──────────────────────────
    if noise_replicates and noise_replicates >= 2:
        measured, err, got = _measure_noise_floor(
            participants, int(noise_replicates),
            {p.name: {} for p in participants}, "iteration-1 state")
        if err:
            return _finish(converged=False, iterations=0, residual=float("nan"),
                           exports={}, history=history, error=err)
        notes.extend(got)
        floor = measured if floor is None else max(floor, measured)
        if measured == 0.0:
            notes.append(
                f"noise floor measured as EXACTLY 0 from {noise_replicates} "
                f"replicate runs: every participant returned a bit-identical "
                f"export. For a deterministic solver that is what should "
                f"happen. For a Monte-Carlo participant it means the SEED IS "
                f"FIXED — the residual then measures the repeatability of one "
                f"random draw, not whether the physics settled, and a run that "
                f"meets tol under a fixed seed is not evidence it would meet it "
                f"under another. Vary the seed between replicates to measure a "
                f"real floor.")
    tol_eff = tol if not floor else max(tol, float(floor))
    block = max(1, int(noise_block)) if floor else 1
    if floor and tol_eff > tol:
        criterion_notes.append(
            f"CONVERGENCE IS AT THE NOISE FLOOR, NOT AT tol: the requested "
            f"tol={tol:.1e} is below the residual noise floor {floor:.3e}, "
            f"which no amount of iterating can cross. The run is judged against "
            f"{tol_eff:.3e} instead, over a block mean of the last {block} "
            f"residuals. ANY TOLERANCE APPLIED TO THIS RESULT — including an "
            f"acceptance tolerance — MUST BE AT LEAST {floor:.3e} RELATIVE.")

    # WARM START FROM A PREVIOUS RUN'S EXPORTS. A level k+1 coupling run in the
    # same work directories finds level k's converged exports.json there; seeding
    # iteration 1's IMPORTS with it starts the fixed-point iteration next to its
    # answer instead of at nothing (each participant maps the coarse samples onto
    # its own nodes, as the served contracts do). Only the imports are seeded: the
    # relaxation state still starts at iteration 1 from the new exports, so a
    # changed point count between levels is never relaxed against. Measured on a
    # manufactured 4C+FEniCSx thermo-elastic pair: 55 iterations cold at every
    # level; recorded runs reached level 2 or 3 and ran out of wall clock.
    warm_seeded = []
    stale_skipped = []
    shape_skipped = []
    for p in participants:
        ep = p.work_dir / "exports.json"
        if not ep.is_file():
            continue
        # A STANDALONE RUN'S EXPORT IS NOT A PREVIOUS LEVEL'S STATE. Measured: a ladder was
        # seeded from a side's standalone exports.json (another time-window length), failed at
        # iteration 1 on the partner's length guard, and the reply named the partner.
        if _COUPLED_EXPORTS.get(str(ep.resolve())) != _file_digest(ep):
            try:                          # named only where the old rule would have seeded from it
                _sv = _stack(InterfaceData.from_json(ep))
                if _sv.size and np.all(np.isfinite(_sv)):
                    stale_skipped.append(p.name)
            except Exception:             # noqa: BLE001 -- an unreadable file seeds nothing either way
                pass
            continue
        try:
            seed = InterfaceData.from_json(ep)
            sv = _stack(seed)
            if sv.size and np.all(np.isfinite(sv)):
                # THE SEED MUST FIT THIS LEVEL: the stated n_steps of the side that wrote it
                # and of every side that reads it.
                _stated = {n for n in [_stated_n_steps(p)] + [_stated_n_steps(q) for q in participants
                                                               if p.name in (q.imports_from or [])]
                           if n is not None}
                _why = _seed_misfit(seed, _stated, _COUPLED_STEPS.get(str(ep.resolve())))
                if _why:
                    shape_skipped.append(f"{p.name} ({_why})")
                    continue
                exports[p.name] = seed
                warm_seeded.append(p.name)
        except Exception:                             # noqa: BLE001 -- a stale file is not an error
            continue
    if warm_seeded:
        notes.append(f"warm start: iteration 1 imports were seeded from the exports.json already in the "
                     f"work directories of {', '.join(warm_seeded)} (the interface state a converged "
                     f"coupling left there, typically the previous mesh level's); the relaxation starts "
                     f"fresh at iteration 1")
    if stale_skipped:
        notes.append(f"no warm start from {', '.join(stale_skipped)}: the exports.json there was not left by "
                     f"a converged coupling (a standalone run writes one too), so iteration 1 starts cold")
    if shape_skipped:
        notes.append(f"no warm start from {'; '.join(shape_skipped)}: the previous level's export does not fit "
                     f"this level's window, so the sides that read it start cold at iteration 1")

    stalled_at = None
    raw_last: dict[str, np.ndarray] = {}
    for it in range(1, max_iter + 1):
        new_exports: dict[str, InterfaceData] = {}
        for p in participants:
            # assemble imports = latest exports of the partners this participant reads
            imp = {src: exports[src].to_dict() for src in p.imports_from if src in exports}
            imp_text = json.dumps(imp, indent=2, sort_keys=True)
            (p.work_dir / "imports.json").write_text(imp_text)
            last_imports[p.name] = imp_text
            ep = p.work_dir / "exports.json"
            if ep.exists():
                ep.unlink()
            try:
                r = subprocess.run(p.command, cwd=str(p.work_dir), env=_participant_env(p), capture_output=True,
                                   text=True, timeout=p.timeout, stdin=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                return _finish(converged=False, iterations=it, residual=float("nan"),
                               exports={}, history=history,
                               error=(f"participant {p.name} timed out after "
                                      f"{p.timeout}s at iteration {it} — the "
                                      "coupling was killed, no result"))
            except (OSError, ValueError) as e:
                return _finish(converged=False, iterations=it, residual=float("nan"),
                               exports={}, history=history,
                               error=f"participant {p.name} could not be launched: {e}")
            _persist_participant_output(p, r, it)
            returncodes[p.name] = int(r.returncode)
            if not ep.exists():
                return _finish(converged=False, iterations=it, residual=float("nan"),
                               exports={}, history=history,
                               error=f"participant {p.name} wrote no exports.json "
                                     f"(rc={r.returncode}). stderr tail: {_output_tail(r)}")
            # A NON-ZERO exit code is a failed solve even when exports.json is
            # present: a solver that diverges often writes its last iterate and
            # then aborts. Continuing on that output produced a converged-looking
            # coupling built on a crashed participant.
            if r.returncode != 0:
                return _finish(converged=False, iterations=it, residual=float("nan"),
                               exports={n: e.to_dict() for n, e in exports.items()},
                               history=history,
                               error=(f"participant {p.name} exited with code "
                                      f"{r.returncode} at iteration {it}; its "
                                      "exports.json is the output of a FAILED run "
                                      "and must not be coupled on. stderr tail: "
                                      f"{_output_tail(r)}"))
            try:
                new_exports[p.name] = InterfaceData.from_json(ep)
            except Exception as e:
                return _finish(converged=False, iterations=it, residual=float("nan"),
                               exports={}, history=history,
                               error=f"participant {p.name} bad exports.json: {e}")
            trace[p.name].append((_digest(imp_text), _digest(ep.read_text(errors="replace")),
                                  _import_columns(imp_text)))
            ifd = new_exports[p.name]
            v = _stack(ifd)
            # An EMPTY export is not a converged one. With nothing in the stacked
            # vector the residual is 0/1e-30 = 0 at iteration 2, so a participant
            # that writes a well-formed but empty interface converges instantly
            # and every value-based check has nothing to look at and says nothing.
            if v.size == 0:
                return _finish(converged=False, iterations=it, residual=float("nan"),
                               exports={}, history=history,
                               error=(f"participant {p.name} exported an EMPTY "
                                      "interface (no values) at iteration "
                                      f"{it} — there is nothing to couple, and an "
                                      "empty exchange would otherwise report a "
                                      "residual of zero"))
            coords = np.asarray(ifd.coordinates, float)
            bad = (not np.all(np.isfinite(v))) or (
                coords.size and not np.all(np.isfinite(coords)))
            if bad:
                nonfinite_hits += 1
                if nonfinite_hits <= _MAX_NONFINITE_WARNINGS:
                    where = "values/fluxes" if not np.all(np.isfinite(v)) else "coordinates"
                    warnings.append(
                        f"{p.name}: non-finite export {where} at iter {it}")
                # A NON-FINITE EXPORT ENDS THE COUPLING, BY NAME. Measured: a run whose side
                # exported NaN flux iterated 150 times (two minutes) to "did not converge ...
                # last residual nan", and a NaN first pass never recovers (see _NONFINITE_STOP).
                nonfinite_run[p.name] = nonfinite_run.get(p.name, 0) + 1
                if nonfinite_run[p.name] >= _NONFINITE_STOP:
                    cols = [name for name, arr in (("values", getattr(ifd, "values", None)),
                                                   ("normal_fluxes", getattr(ifd, "normal_fluxes", None)),
                                                   ("coordinates", ifd.coordinates))
                            if arr is not None and np.asarray(arr, float).size
                            and not np.all(np.isfinite(np.asarray(arr, float)))]
                    return _finish(converged=False, iterations=it, residual=float("nan"),
                                   exports={}, history=history,
                                   error=(f"participant {p.name} exported non-finite "
                                          f"{', '.join(cols) or 'data'} at iteration {it}: the "
                                          "residual and the relaxation carry a non-finite number "
                                          "into every later iteration, so the coupling stops here. "
                                          "Find where that side's own computation produces it -- "
                                          "a division by a zero weight or norm, a solve that "
                                          "failed -- and run that side standalone first."))
            else:
                nonfinite_run[p.name] = 0
            # An export whose length changes between iterations breaks relaxation:
            # numpy either broadcasts a length-1 block up to the new length or
            # raises out of the driver. Neither is a result.
            if p.name in relaxed_prev and v.shape != relaxed_prev[p.name].shape:
                return _finish(converged=False, iterations=it, residual=float("nan"),
                               exports={}, history=history,
                               error=(f"participant {p.name} changed its export "
                                      f"length from {relaxed_prev[p.name].shape[0]} to "
                                      f"{v.shape[0]} at iteration {it} — the exported "
                                      "interface must have the same layout every "
                                      "iteration or relaxation is meaningless"))

        # relaxation + residual on the concatenated export vector
        if it == 1:
            for n, ifd in new_exports.items():
                exports[n] = ifd
                relaxed_prev[n] = _stack(ifd)
                prev_blocks[n] = _blocks(ifd)
            history.append(float("nan"))
            continue

        # A participant that changes its export LENGTH between iterations used
        # to reach _relax and raise a bare numpy broadcast ValueError straight
        # out of run_coupling — or, when the new length was 1, broadcast
        # silently and be misreported as a non-convergence with no clue why.
        # Both are the same setup error, so name it here.
        for p in participants:
            m, k = _stack(new_exports[p.name]).size, relaxed_prev[p.name].size
            if m != k:
                return CouplingResult(
                    False, it, float("nan"), {}, history,
                    error=(f"participant {p.name} changed its export size from "
                           f"{k} to {m} at iteration {it}. Every participant must "
                           f"export the SAME number of points, in the same order, "
                           f"every iteration (and keep normal_fluxes present or "
                           f"absent consistently) — the driver relaxes export "
                           f"vectors element by element."),
                    warnings=warnings, notes=notes,
                    criterion_notes=criterion_notes)

        raw_last = {n: _stack(e).copy() for n, e in new_exports.items()}   # before relaxation
        total_res = 0.0
        total_ref = 0.0
        # ONE theta for the whole interface state (see _aitken): Aitken is applied
        # to the composite fixed-point map, not to each participant separately.
        if accelerator == "auto":
            multi = any(getattr(new_exports[p.name].values, "ndim", 1) == 2
                        and new_exports[p.name].values.shape[1] >= 2 for p in participants)
            accelerator = "anderson" if multi else "aitken"
            notes.append(f"accelerator 'auto' resolved to '{accelerator}' "
                         f"({'a multi-field' if multi else 'a single-field'} interface exchange)")
        relaxed_all = None
        if accelerator == "aitken":
            raw_all = np.concatenate([_stack(new_exports[p.name]) for p in participants])
            prev_all = np.concatenate([relaxed_prev[p.name] for p in participants])
            th, r_k = _aitken(prev_all, raw_all, res_prev.get("*"), theta_global)
            theta_global = th
            res_prev["*"] = r_k
        elif accelerator == "anderson":
            raw_all = np.concatenate([_stack(new_exports[p.name]) for p in participants])
            prev_all = np.concatenate([relaxed_prev[p.name] for p in participants])
            relaxed_all = anderson.step(prev_all, raw_all)
            th = theta0
        else:
            th = theta0
        offset = 0
        for p in participants:
            n = p.name
            raw_new = _stack(new_exports[n])
            prev = relaxed_prev[n]
            if relaxed_all is not None:
                relaxed = relaxed_all[offset:offset + raw_new.size]
                offset += raw_new.size
            else:
                relaxed = _relax(prev, raw_new, th)
            total_res += float(np.sum((raw_new - prev) ** 2))
            total_ref += float(np.sum(raw_new ** 2)) + 1e-30
            # per-block relative change, so a large settled block cannot mask a
            # small moving one in the single global norm below; and beside it the
            # block's OWN FIXED-POINT RESIDUAL (its raw output against the relaxed input
            # that produced it). The change alone is, under an accelerator, the size of
            # the last step, not the distance left: measured, a block that moved 5.0e-4
            # of itself on its last Aitken step had landed within 4.8e-7 of an
            # independently converged run, and a right ladder read NOT VERIFIED on it.
            nb = _blocks(new_exports[n])
            prelax = None
            if prev is not None and np.asarray(prev).size == raw_new.size:
                _vshape = np.asarray(new_exports[n].values).shape
                _nv = int(np.prod(_vshape)) if _vshape else 1
                _fl = new_exports[n].normal_fluxes
                _carrier = InterfaceData(
                    coordinates=new_exports[n].coordinates,
                    field_name=new_exports[n].field_name,
                    values=np.asarray(prev[:_nv], float).reshape(_vshape),
                    normal_fluxes=(np.asarray(prev[_nv:], float).reshape(np.asarray(_fl).shape)
                                   if _fl is not None else None))
                prelax = _blocks(_carrier)
            for bname, arr in nb.items():
                pb = prev_blocks.get(n, {}).get(bname)
                block_residuals[f"{n}.{bname}"] = (
                    _rel_change(arr, pb) if pb is not None else float("nan"))
                (block_scale_change[f"{n}.{bname}"],
                 block_tiny_change[f"{n}.{bname}"]) = (
                    _scaled_change(arr, pb) if pb is not None else (float("nan"), float("nan")))
                fp = (prelax or {}).get(bname)
                block_fixed_point[f"{n}.{bname}"] = (
                    _rel_change(arr, fp) if fp is not None else float("nan"))
                # THE DISTANCE LEFT, FROM HOW FAST THE STEPS SHRINK. Without relaxation
                # the fixed-point residual IS the last step, so it could not clear a
                # block that had landed (measured: named 6e-8 .. 1.4e-7 from its own
                # fixed point, a 1e-5 limit, seven false caveats and a right ladder
                # denied). The ratio is taken over TWO steps, because every side reads
                # the partner's previous output: a two-sided exchange alternates, and
                # the step can hold still for one iteration and fall 4x on the next
                # (measured on a linear pair: one-step ratios 1, 0.25, 1, 0.25, ...,
                # where the last-step estimate is 5x short). With steps s1..s4 and
                # q = s4/s2 < 1, the fixed point lies within q (s3 + s4) / (1 - q) --
                # exact for a steady two-step contraction, and s4 r / (1 - r) again
                # when every step shrinks by the same r.
                _key = f"{n}.{bname}"
                _hs = step_hist.setdefault(_key, [])
                if block_residuals[_key] == block_residuals[_key]:
                    _hs.append(block_residuals[_key])
                    del _hs[:-4]
                else:
                    _hs.clear()
                # THE LATEST TWO-STEP RATIO ALONE: max(s4/s2, s3/s1) let an earlier slow
                # pair override the final accelerated jump and named a block 5.4e-7 from its
                # fixed point (measured). s4/s2 alone is still exact for a steady two-step
                # contraction, in both phases of it.
                if len(_hs) == 4 and _hs[1] > 0:
                    _q = _hs[3] / _hs[1]
                    block_distance[_key] = (_q * (_hs[2] + _hs[3]) / (1.0 - _q)
                                            if _q < 0.95 else float("inf"))
            prev_blocks[n] = nb
            # write relaxed values back into the InterfaceData carrier
            ifd = new_exports[n]
            ncomp = ifd.values.size
            ifd.values = relaxed[:ncomp].reshape(ifd.values.shape)
            if ifd.normal_fluxes is not None:
                ifd.normal_fluxes = relaxed[ncomp:].reshape(ifd.normal_fluxes.shape)
            exports[n] = ifd
            relaxed_prev[n] = relaxed

        res = float(np.sqrt(total_res / total_ref))
        history.append(res)
        # `_stat`/`tol_eff` are the plain last-residual-against-tol test unless a
        # noise floor is in play: with floor=None, block is 1 and tol_eff is tol,
        # so this is `res < tol` exactly.
        if _stat(history, block) < tol_eff:
            at_floor = bool(floor and tol_eff > tol)
            if nonfinite_hits > _MAX_NONFINITE_WARNINGS:
                warnings.append(f"... {nonfinite_hits - _MAX_NONFINITE_WARNINGS} "
                                "further non-finite export warnings suppressed")
            # Only on the success path: a run that already failed has a finding,
            # and the probe would cost a solve to say something already known.
            sens = (probe_interface_sensitivity(participants, last_imports)
                    if probe else {})
            return _finish(converged=True, iterations=it, residual=res,
                           exports={n: e.to_dict() for n, e in exports.items()},
                           history=history, sensitivity=sens)
        # A RESIDUAL THAT HAS STOPPED FALLING STOPS THE LOOP. Measured: a
        # coupling whose outer boundary condition was never applied iterated 150
        # times at a residual of 3e-2 -- about twelve minutes of a forty-five
        # minute run -- and ended where it stood after twenty. A deterministic
        # fixed-point iteration whose residual no longer falls does not converge
        # by iterating on. Not with a noise floor in play: a sampled participant
        # plateaus by design, and that route re-measures its floor instead.
        # A RESIDUAL THAT STILL FALLS IS NOT A STALL: it goes on while its rate reaches
        # tol within max_iter, and stops, said as what it is, when it cannot.
        if (floor is None and not noise_replicates and it >= _STALL_MIN_ITERS
                and it < max_iter):
            verdict = _stop_verdict(history, tol_eff, max_iter - it)
            if verdict is not None:
                stalled_at = (it, verdict)
                break

    if nonfinite_hits > _MAX_NONFINITE_WARNINGS:
        warnings.append(f"... {nonfinite_hits - _MAX_NONFINITE_WARNINGS} further "
                        "non-finite export warnings suppressed")

    # ── did not converge. If a floor was in play, the pre-loop estimate was
    # taken with the participants in their iteration-1 fallback, which need not
    # carry the same relative scatter as the settled state. Re-measure THERE
    # before calling a stochastic coupling a failure. Paid only on failure.
    last = history[-1] if history else float("nan")
    if noise_replicates and noise_replicates >= 2:
        # The sensitivity probe FIRST, while the work directories still hold
        # what the iteration left: _measure_noise_floor overwrites imports.json
        # and exports.json with replicate content and does not restore them, so
        # a probe run afterwards would compare the participant against a
        # replicate rather than against its own converged answer.
        sens = (probe_interface_sensitivity(participants, last_imports)
                if probe else {})
        imports_now = {p.name: {s: exports[s].to_dict()
                                for s in p.imports_from if s in exports}
                       for p in participants}
        measured, err, got = _measure_noise_floor(
            participants, int(noise_replicates), imports_now,
            "final state, against the residual the loop reports",
            against=relaxed_prev)
        if not err:
            notes.extend(got)
            if measured is not None and measured > (floor or 0.0):
                floor = measured
                tol_eff = max(tol, float(floor))
            if _stat(history, block) < tol_eff:
                at_floor = True
                criterion_notes.append(
                    f"CONVERGED AT THE RE-MEASURED NOISE FLOOR. The iteration did "
                    f"not reach tol={tol:.1e}, but the residual noise floor "
                    f"measured with the participants in their FINAL state is "
                    f"{floor:.3e}, and the block mean of the last {block} "
                    f"residuals is below it. The residual has stopped measuring "
                    f"the coupling and started measuring the sampler. Any "
                    f"tolerance applied to this result must be at least "
                    f"{floor:.3e} relative.")
                return _finish(
                    converged=True, iterations=max_iter, residual=last,
                    exports={n: e.to_dict() for n, e in exports.items()},
                    history=history, sensitivity=sens)

    # ONE MORE RUN OF EACH SIDE ON THE IMPORTS ITS LAST ITERATION READ. A finite-
    # element solve is a function of its inputs and returns the same export bit for
    # bit; one that does not puts a floor under the residual that no iteration can
    # beat. Measured on one coupled round: every coupling that stalled had a side
    # whose free-dof mask was uninitialised memory -- six runs on one imports.json,
    # six exports 11-92 % apart -- while the pair itself converges in 5-8 iterations
    # when that side is repeatable, and nothing served named it.
    repeat_notes: list[str] = []
    probe_ran = False
    if not noise_replicates and floor is None and raw_last:
        repeat_notes = _repeat_probe(participants, last_imports, raw_last)
        probe_ran = True
    stall = stalled_at is not None and stalled_at[1]["kind"] == "stall"
    if stalled_at is not None and not stall:
        _it, _v = stalled_at
        err_msg = (f"did not converge to tol={tol_eff:g}: STOPPED at iteration {_it} of "
                   f"{max_iter}, because at the rate its residual still falls it cannot reach "
                   f"tol within max_iter. It falls steadily, about {_v['rate']:.4f} per iteration "
                   f"over the last {3 * _STALL_WINDOW} iterations (last residual {last:.2e}); at "
                   f"that rate tol is about {_v['needed']:,} iterations away, and max_iter left "
                   f"{max_iter - _it}. This is a slow contraction, not a stall: at this rate it "
                   f"needs a max_iter of about {_it + _v['needed']:,} — result is NOT trustworthy")
    elif stall:
        _it, _v = stalled_at
        _early, _late = _v["early"], _v["late"]
        err_msg = (f"did not converge to tol={tol_eff:g}: STOPPED at iteration {_it} of "
                   f"{max_iter}, because the residual stopped falling (median "
                   f"{_late:.2e} over the last {_STALL_WINDOW} iterations against "
                   f"{_early:.2e} over the {_STALL_WINDOW} before; last residual "
                   f"{last:.2e}). A fixed-point iteration whose residual no longer "
                   f"falls does not converge by iterating on — result is NOT trustworthy")
    else:
        err_msg = (f"did not converge to tol={tol_eff:g} in {max_iter} iters "
                   f"(last residual {last:.2e}) — result is NOT trustworthy")
    if repeat_notes:
        err_msg += "; " + " ".join(repeat_notes)
    elif probe_ran and stall:
        # MEASURED, NOT GUESSED: every side was run again on its last imports and
        # returned the same export, so no side is noisy and the sampled-estimator
        # route below would only relax the criterion onto a real defect.
        err_msg += ("; every participant returned the same export when run again on "
                    "the imports of its last iteration, so the stall is not scatter in "
                    "one side: look at the exchange itself -- which side imposes what, "
                    "the sign of each exported flux, the order of the interface points")
    if floor is None and _stalled(history) and not repeat_notes and not probe_ran:
        # The residual stopped falling rather than never having fallen. That is
        # what a sampling floor looks like from outside, and it is also what a
        # theta above the stability limit looks like; the driver cannot tell
        # them apart without a measurement, so it names the measurement.
        err_msg += ("; the residual STOPPED FALLING rather than falling too "
                    "slowly — if any participant is a Monte-Carlo / sampled "
                    "estimator, re-run with noise_replicates>=2 so the driver "
                    "can measure its residual floor and judge against it "
                    "instead of against an unreachable tol")
    return _finish(converged=False,
                   iterations=(stalled_at[0] if stalled_at is not None else max_iter),
                   residual=last,
                   exports={n: e.to_dict() for n, e in exports.items()},
                   history=history, error=err_msg)


def probe_interface_sensitivity(participants: list[Participant],
                                last_imports: dict[str, str],
                                delta: float = 1e-3) -> dict[str, dict]:
    """Does each participant's answer ACTUALLY depend on what it is handed?

    Watching the iteration cannot establish this. A participant that never opens
    imports.json but advances some internal state produces a different export
    every iteration, converges, and reads as fully responsive — it was stamped
    trustworthy. One that adds a token 1e-16 multiple of its import to defeat a
    byte-identity test does the same.

    So ask the question directly, twice, after the coupling has settled:

      NOISE   re-run the participant on EXACTLY the imports it last had. A solver
              that is a function of its boundary data returns the same answer. A
              non-zero value here means hidden state or nondeterminism, and a
              fixed-point iteration over such a participant has not converged to
              anything, whatever the residual says. This is the run that catches
              a participant which ignores imports.json while advancing a counter
              — the single-run probe cannot, because its output moves anyway.

      SIGNAL  re-run it again with every exchanged number nudged by `delta`
              relative, and measure how far the answer moves from the NOISE run
              (one further step of whatever internal drift exists, so the two are
              directly comparable).

    S = signal / delta is a finite-difference interface sensitivity; S = 0 means
    the output is not a function of the input at all. Two extra solves per
    participant for the whole coupling.

    Runs in place and restores imports.json and exports.json, so the work
    directory is left as the coupling left it. Returns per participant
    {"noise", "signal", "S"} with None where a quantity could not be measured —
    never a number that would read as a pass.
    """
    out: dict[str, dict] = {}

    def _blank(reason):
        return {"noise": None, "signal": None, "S": None, "blocks": {},
                "detail": reason}

    for p in participants:
        if not p.imports_from:
            out[p.name] = _blank("participant declares no imports")
            continue
        ip, ep = p.work_dir / "imports.json", p.work_dir / "exports.json"
        if not ep.exists():
            out[p.name] = _blank("no exports.json left from the run")
            continue
        snapshot = None
        try:
            base_vec = _stack(InterfaceData.from_json(ep))
            saved_exports = ep.read_text()
            saved_imports = last_imports.get(
                p.name, ip.read_text() if ip.exists() else "{}")
            imp = json.loads(saved_imports or "{}")
            # taken before the first perturbed re-run, restored in `finally`
            snapshot = _snapshot_tree(p.work_dir)
        except Exception as e:
            out[p.name] = _blank(f"could not read the run state: {e}")
            continue

        def _run(text) -> Optional[InterfaceData]:
            try:
                ip.write_text(text)
                if ep.exists():
                    ep.unlink()
                r = subprocess.run(p.command, cwd=str(p.work_dir), env=_participant_env(p),
                                   capture_output=True, text=True, timeout=p.timeout, stdin=subprocess.DEVNULL)
                if r.returncode != 0 or not ep.exists():
                    return None
                return InterfaceData.from_json(ep)
            except Exception:
                return None

        try:
            repeat = _run(saved_imports)
            if repeat is None:
                out[p.name] = _blank("the participant did not re-run cleanly")
                continue
            noise = _rel_change(_stack(repeat), base_vec)

            touched = False
            for d in imp.values():
                if not isinstance(d, dict):
                    continue
                try:
                    _tv = np.asarray(d.get("values") or [], float)
                    _tscale = float(np.max(np.abs(_tv[np.isfinite(_tv)]))) if _tv.size else 0.0
                except (TypeError, ValueError):
                    _tscale = 0.0
                for key in ("values", "normal_fluxes"):
                    if d.get(key) is None:
                        continue
                    a = np.asarray(d[key], float)
                    if a.size == 0 or not np.all(np.isfinite(a)):
                        continue
                    scale = float(np.max(np.abs(a))) or 1.0
                    # A FLUX AT ROUND-OFF IS NUDGED ON THE TRACE'S SCALE. Nudged relative to
                    # itself it never really moved, and a side that ignores it read as
                    # responsive or unresponsive by chance (measured: 1e-17 fluxes).
                    if key == "normal_fluxes" and _tscale > 0 and scale <= 1e-12 * _tscale:
                        d[key] = (a + delta * _tscale).tolist()
                        touched = True
                        continue
                    # relative nudge, with an absolute floor so exactly-zero
                    # entries still move (a zero import is still an import)
                    d[key] = (a + delta * np.where(np.abs(a) > 0,
                                                   np.abs(a), scale)).tolist()
                    touched = True
            if not touched:
                out[p.name] = {"noise": _f(noise), "signal": None, "S": None,
                               "blocks": {},
                               "detail": "its imports carry no numbers to perturb"}
                continue
            nudged = _run(json.dumps(imp, indent=2, sort_keys=True))
            if nudged is None:
                out[p.name] = {"noise": _f(noise), "signal": None, "S": None,
                               "blocks": {},
                               "detail": "the perturbed run did not complete"}
                continue
            signal = _rel_change(_stack(nudged), _stack(repeat))
            # PER BLOCK as well as in total. A participant can hold its physics
            # frozen at a stale value while echoing an imported quantity back in
            # another block: the stacked vector then responds fully and the
            # frozen half is invisible. Demonstrated — the values block was
            # pinned to a constant while normal_fluxes passed the import
            # straight through, and the coupling was stamped trustworthy.
            nb, rb = _blocks(nudged), _blocks(repeat)
            blocks = {k: _f(_rel_change(v, rb[k]) if k in rb and v.shape == rb[k].shape
                            else float("nan"))
                      for k, v in nb.items()}
            out[p.name] = {"noise": _f(noise), "signal": _f(signal),
                           "S": (None if signal != signal else float(signal) / delta),
                           "blocks": {k: (None if v is None else v / delta)
                                      for k, v in blocks.items()},
                           "detail": ""}
        finally:
            # RESTORE THE WHOLE WORK DIRECTORY, NOT JUST THE TWO JSON FILES.
            #
            # This probe re-runs the participant TWICE — once on its own saved
            # imports, once on imports nudged by `delta` — and used to restore
            # only imports.json and exports.json. Every OTHER file the
            # participant writes was therefore left in the state produced by
            # the PERTURBED run: solution CSVs, VTU dumps, logs, RESULT files.
            # Anything downstream that reads those files reads a solve of a
            # deliberately wrong problem, off by a relative 1e-3, with nothing
            # in the output saying so. Found when a convergence study measured
            # orders 1.98, 0.32, -0.12 off a perturbed dump.
            #
            # The snapshot is taken lazily, only for participants that are
            # actually probed, and only of regular files.
            try:
                ip.write_text(saved_imports)
                ep.write_text(saved_exports)
            except OSError:
                pass
            if snapshot is not None:
                _restore_tree(p.work_dir, snapshot)
                shutil.rmtree(snapshot, ignore_errors=True)
    return out


def _snapshot_tree(work_dir: Path) -> Optional[Path]:
    """Copy every regular file under `work_dir` into a temp tree."""
    try:
        dest = Path(tempfile.mkdtemp(prefix="openpaso_probe_snap_"))
        for src in work_dir.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(work_dir)
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / rel)
        return dest
    except OSError:
        return None


def _restore_tree(work_dir: Path, snapshot: Path) -> None:
    """Put back everything the snapshot holds; delete what it does not.

    Files the probe CREATED are removed too — a stray output from a perturbed
    solve is as misleading as a modified one.
    """
    try:
        kept = set()
        for src in snapshot.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(snapshot)
            kept.add(rel)
            dst = work_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        for cur in list(work_dir.rglob("*")):
            if cur.is_file() and cur.relative_to(work_dir) not in kept:
                cur.unlink(missing_ok=True)
    except OSError:
        pass


def _f(x) -> Optional[float]:
    return None if x is None or x != x else float(x)


def _import_columns(imp_text: str) -> dict:
    """{"<partner>.<key>": array} for every numeric array in an imports.json text."""
    try:
        data = json.loads(imp_text or "{}")
    except Exception:                                        # noqa: BLE001
        return {}
    out: dict = {}
    if isinstance(data, dict):
        for partner, block in data.items():
            if isinstance(block, dict):
                for key, val in block.items():
                    if isinstance(val, list):
                        try:
                            out[f"{partner}.{key}"] = np.asarray(val, dtype=float).ravel()
                        except (TypeError, ValueError):
                            continue
    return out


def _column_moved(a, b) -> bool:
    """A column moved when it changed by more than round-off of its own scale:
    relaxing a constant column shifts it by a bit, which is no change at all."""
    if a is None or b is None or a.shape != b.shape:
        return True
    scale = max(float(np.abs(a).max(initial=0.0)), float(np.abs(b).max(initial=0.0)), 1e-300)
    return bool(np.abs(a - b).max(initial=0.0) > 1e-9 * scale)


def _responsiveness_detail(trace: dict, participants: list[Participant]) -> dict:
    """WHICH IMPORTED COLUMNS MOVED. A side's export can stay byte-identical while its
    imports change in a column it does not apply: measured, a Neumann side whose partner
    exported the same placeholder flux (1e-10 at every point) every iteration saw only the
    partner's values move under relaxation, and was told its answer did not depend on its
    imports and that it never read imports.json -- it did, and applied the flux."""
    out: dict = {}
    for name, status in _responsiveness(trace, participants).items():
        if status != "unresponsive":
            continue
        hist = trace.get(name) or []
        changed: set = set()
        seen: set = set()
        for a, b in zip(hist, hist[1:]):
            if len(a) < 3 or len(b) < 3 or a[0] == b[0] or a[1] != b[1]:
                continue
            both = set(a[2]) & set(b[2])                     # a column that appears is no change
            seen |= both
            changed |= {k for k in both if _column_moved(a[2][k], b[2][k])}
        # the points themselves never move; naming them as "unchanged" says nothing
        static = {k for k in seen if k.endswith(".coordinates")}
        if seen:
            out[name] = {"changed": sorted(changed), "unchanged": sorted(seen - changed - static)}
    return out


def _responsiveness(trace: dict[str, list[tuple[str, str]]],
                    participants: list[Participant]) -> dict[str, str]:
    """Did each participant's export ever MOVE when its imports moved?

    The discriminator between a real converged solve and a participant that
    exits 0 having done nothing (or that re-serves a cached answer, or that
    never opened imports.json): compare the byte digest of what it was handed
    with the byte digest of what it produced. A solver whose input changed and
    whose output is byte-identical is not a function of its input.

    A genuinely converged participant does not trip this: its export becomes
    byte-identical only once the imports it is given have also stopped changing,
    and the check only looks at iteration pairs where the imports DID change.
    """
    declared = {p.name: list(p.imports_from) for p in participants}
    out: dict[str, str] = {}
    for name, hist in trace.items():
        if not declared.get(name):
            out[name] = "no imports declared"
            continue
        moved_in = moved_out = 0
        for h0, h1 in zip(hist, hist[1:]):
            i0, e0, i1, e1 = h0[0], h0[1], h1[0], h1[1]
            if i0 != i1:
                moved_in += 1
                if e0 != e1:
                    moved_out += 1
        if moved_in == 0:
            out[name] = "imports never changed"
        elif moved_out == 0:
            out[name] = "unresponsive"
        else:
            out[name] = "responsive"
    return out


def _stat(history: list[float], block: int) -> float:
    """The stopping statistic: the last residual, or the mean of the last
    `block` of them once a noise floor is in play.

    Block-averaging is the whole discipline for a stochastic coupling. A single
    residual that dips under the floor says nothing — the noise puts it there
    roughly half the time — so with a floor active the run only stops when a
    WHOLE BLOCK of consecutive residuals averages below it.
    """
    vals = [v for v in history[-block:] if np.isfinite(v)]
    if not vals:
        return float("inf")
    if len(vals) < block:
        # Not enough post-NaN history yet to fill the block; refuse to stop.
        return float("inf")
    return float(sum(vals) / len(vals))


# A RESIDUAL THAT HAS STOPPED FALLING STOPS THE LOOP (see run_coupling). Not before
# this many iterations, over two windows of this many, and "stopped" means the
# later window's median is at least this fraction of the earlier one's. A coupling
# that still contracts at 0.98 per iteration falls to 0.82 over a window and is
# never looked at. One that contracts at 0.99 reads 0.904 and is looked at, and it
# is not a stall: it reaches 1e-6 from 1 in about 1,400 iterations. _stop_verdict
# tells the two apart.
_STALL_MIN_ITERS = 30
_STALL_WINDOW = 10
_STALL_RATIO = 0.9


def _plateaued(history: list[float], window: int = _STALL_WINDOW,
               ratio: float = _STALL_RATIO):
    """(early, late) medians when the last `window` residuals sit at `ratio` or
    more of the `window` before them, else None. Finite residuals only."""
    vals = [v for v in history if np.isfinite(v)]
    if len(vals) < 2 * window:
        return None
    early = float(np.median(vals[-2 * window:-window]))
    late = float(np.median(vals[-window:]))
    return (early, late) if early > 0 and late >= ratio * early else None


def _reaches_tol(history: list[float], tol: float, remaining: int,
                 window: int = _STALL_WINDOW) -> bool:
    """Does the rate measured over the last 2*window residuals reach `tol` within
    `remaining` iterations? A slow contraction is not a plateau: at 0.99 per
    iteration the median ratio over ten is 0.904, above the plateau bar, and the
    run was stopped at iteration 30 whatever max_iter said. A residual that is
    flat or rising never gets there, and still stops."""
    vals = [v for v in history if np.isfinite(v) and v > 0]
    if len(vals) < 2 * window or remaining <= 0:
        return False
    early = float(np.median(vals[-2 * window:-window]))
    late = float(np.median(vals[-window:]))
    if not (0.0 < late < early):
        return False
    rate = (late / early) ** (1.0 / window)
    last = vals[-1]
    if last <= tol:
        return True
    return float(np.log(tol / last) / np.log(rate)) <= remaining


def _steady_fall(history: list[float], window: int = _STALL_WINDOW):
    """The rate per iteration at which the residual still falls, or None.

    It falls when, over the last three windows of `window` residuals, the LARGEST and
    the SMALLEST residual of each window both lie below those of the window before,
    and each one's later fall is at least half its earlier fall (in log): a residual
    settling onto a floor slows down and is no fall. The extremes and not the
    medians: a map that contracts 1 % per iteration while its residual swings by a
    factor of three inside every four iterations (measured on a coupled pair) has
    window medians that alternate up and down, while both extremes fall every
    window. A residual driven by a forcing that never dies out, or cycling at a
    unit rate, has extremes that stand still or jump about. The rate is read off
    both extremes over the last two windows."""
    import math
    vals = [v for v in history if np.isfinite(v) and v > 0]
    if len(vals) < 3 * window:
        return None
    wins = [vals[-3 * window:-2 * window], vals[-2 * window:-window], vals[-window:]]
    logs = []
    for pick in (max, min):
        s = [float(pick(w)) for w in wins]
        if not s[2] < s[1] < s[0]:
            return None
        d1, d2 = math.log(s[1] / s[0]), math.log(s[2] / s[1])
        if d2 > 0.5 * d1:                   # d1, d2 < 0: the later fall is less than half
            return None
        logs.append(math.log(s[2] / s[0]))
    return math.exp(sum(logs) / (2 * 2 * window))


def _stop_verdict(history: list[float], tol: float, remaining: int,
                  window: int = _STALL_WINDOW):
    """None to go on; otherwise why to stop now, as a dict.

    {"kind": "stall", "early", "late"} -- the residual has stopped falling: the
    window medians read `ratio` or more (_plateaued), their rate does not reach tol
    in the iterations left (_reaches_tol), and it does not fall steadily either
    (_steady_fall).
    {"kind": "slow", "early", "late", "rate", "needed"} -- it still falls steadily at
    `rate` per iteration, and at that rate tol is `needed` iterations away, more than
    `remaining`. Measured on two coupled pairs that contract 1 % per iteration: both
    were stopped as stalls at iteration 30, and one of them at iteration 40 however
    large max_iter was."""
    import math
    if sum(1 for v in history if np.isfinite(v) and v > 0) < 3 * window:
        return None                        # a stall cannot be told from a slow fall yet
    plateau = _plateaued(history, window)
    if plateau is None or _reaches_tol(history, tol, remaining, window):
        return None
    early, late = plateau
    rate = _steady_fall(history, window)
    if rate is None:
        return {"kind": "stall", "early": early, "late": late}
    top = max(v for v in history[-window:] if np.isfinite(v) and v > 0)
    needed = 0 if top <= tol else int(math.ceil(math.log(tol / top) / math.log(rate)))
    if needed <= remaining:
        return None
    return {"kind": "slow", "early": early, "late": late, "rate": rate, "needed": needed}


def _stalled(history: list[float], window: int = 12, floor_window: int = 6) -> bool:
    """Has the residual stopped falling? Compares the last half of a window
    against the first half. Used only to make a failure message name the right
    next step, never to declare convergence.

    MEDIANS, AND AS LONG A WINDOW AS THE HISTORY ALLOWS.

    This used to average three residuals against the previous three. On the
    plateau this exists to detect — a stochastic participant whose residual has
    bottomed out in its own sampling noise — three-sample means scatter so much
    that `late > 0.5 * early` failed by chance roughly one run in three. The
    symptom was a genuinely stalled coupling whose failure message did NOT name
    the noise_replicates route, so the user was left halving theta against a
    floor no theta can reach. The hint appeared or not depending on the noise
    draw, which makes it useless as a hint.

    Two changes, both aimed at estimator scatter rather than at the threshold:
    a longer window, and the median instead of the mean, so one lucky spike in
    six cannot move the comparison. The window shrinks to what is available,
    down to `floor_window`, because refusing to answer on a short history is
    what a run with a small max_iter would get otherwise.

    The threshold is unchanged. This makes the SAME question answerable, it
    does not make the answer easier to get.
    """
    vals = [v for v in history if np.isfinite(v)]
    if len(vals) < floor_window:
        return False
    w = min(window, len(vals))
    w -= w % 2                       # even, so the halves are the same size
    half = w // 2
    early = float(np.median(vals[-w:-half]))
    late = float(np.median(vals[-half:]))
    return late > 0.5 * early
