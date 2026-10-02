"""Interface grading quantities — the ones that move when the coupling breaks.

Why the field alone is not enough
---------------------------------
The primary, key-free grade is the observed order from mesh halving.  It is
reference-free, which is its virtue, and it is blind to the single most
important coupled failure mode, which is its limit: **a partitioned scheme that
iterates the wrong transmission condition converges cleanly to the wrong fixed
point, at order two.**  Measured on a real Dirichlet-Neumann solve
(``scripts/mutate_coupling.py``):

    variant                       order_u (mesh halving)   true order   jump_q
    correct                                1.838              1.884     1.6e-14
    wrong transmitted quantity             1.865              0.112     3.0
    reversed interface mapping             1.847              0.405     0.75
    receiver's coefficient 25% wrong       1.847              0.002     1.6e-14

Every mutation self-converges at order ~1.85.  The key-free order cannot
separate any of them from the correct run.  The two-sided flux jump separates
two of the three with the keys still sealed, and the true error separates all
three once the key is opened.  Hence: order from halving stays the primary
grade, the interface quantities are added, and neither is described as
sufficient alone.

What is graded, and against what
--------------------------------
``jump_u``    ``|u_A - u_B|`` at fixed interface probes. Reference-free.
``jump_q``    ``|q_A + q_B| / |q_A|`` with OUTWARD normals, so continuity means
              the sum vanishes. Reference-free, and its correct limit is known
              to be zero without any solution — which is exactly what the
              mesh-halving order cannot give.
``q_profile`` the whole flux profile, not one scalar. A scalar summary of the
              interface is invariant under an interface-mapping reversal — the
              mean interface value and the net flux were measured UNCHANGED to
              six digits under ``MUT_MAP`` — while the profile moves. A single
              number cannot grade a coupling.
``q_true``    flux error against the sealed exact interface flux (phase 2).

Fabrication
-----------
The flux is agent-reported, so an agent could report ``q_B = -q_A`` by
construction.  :func:`recover_flux_from_field` therefore recomputes the
interface flux from the agent's own submitted FIELD values by one-sided
quadratic extrapolation on the probe columns, and :func:`flux_consistency`
compares the two.  Faking the flux then requires faking a consistent field.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path


def _rms(v) -> float:
    v = [float(a) for a in v]
    return math.sqrt(sum(a * a for a in v) / len(v)) if v else 0.0


def read_interface_csv(path: Path, ncoord: int, nval: int, nflux: int):
    """``coords..., values..., fluxes...`` — any bad row invalidates the file."""
    pts, vals, flux = [], [], []
    with open(path, newline="", errors="ignore") as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    if not rows:
        return None, "empty file"
    start = 0
    try:
        float(rows[0][0])
    except (ValueError, IndexError):
        start = 1
    need = ncoord + nval + nflux
    for r in rows[start:]:
        if len(r) < need:
            return None, f"row has {len(r)} columns, need {need}"
        try:
            nums = [float(c) for c in r[:need]]
        except ValueError:
            return None, "unparsable numeric value"
        if not all(math.isfinite(v) for v in nums):
            return None, "non-finite value"
        pts.append(tuple(nums[:ncoord]))
        vals.append(tuple(nums[ncoord:ncoord + nval]))
        flux.append(tuple(nums[ncoord + nval:]))
    return (pts, vals, flux), "ok"


@dataclass
class InterfaceReport:
    levels: list = field(default_factory=list)
    jump_u_rel: list = field(default_factory=list)
    jump_q_rel: list = field(default_factory=list)
    q_profile_order: float | None = None
    consistency: list = field(default_factory=list)
    verdict: str = "NOT_ASSESSED"
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.verdict}: jump_q_rel={self.jump_q_rel}, "
                f"jump_u_rel={self.jump_u_rel}")


def two_sided_jumps(side_a, side_b, coord_tol: float = 1e-6):
    """``jump_u`` and ``jump_q`` at matched interface probe points.

    The two subdomains report at the SAME physical points with OPPOSITE outward
    normals, so a correct coupling has ``u_A - u_B = 0`` and ``q_A + q_B = 0``.
    Points are matched by coordinate, not by row order, so an agent that writes
    one side reversed is detected rather than silently graded against itself.
    """
    (pa, va, qa), (pb, vb, qb) = side_a, side_b
    if len(pa) != len(pb):
        return None, f"sides report {len(pa)} and {len(pb)} interface points"
    idx = {tuple(round(c, 9) for c in p): i for i, p in enumerate(pb)}
    du, dq, scale_u, scale_q = [], [], [], []
    for i, p in enumerate(pa):
        key = tuple(round(c, 9) for c in p)
        j = idx.get(key)
        if j is None:
            best, bd = None, coord_tol
            for k, jj in idx.items():
                d = max(abs(a - b) for a, b in zip(k, p))
                if d < bd:
                    best, bd = jj, d
            j = best
        if j is None:
            return None, f"subdomain B reports no point at {p}"
        du += [a - b for a, b in zip(va[i], vb[j])]
        dq += [a + b for a, b in zip(qa[i], qb[j])]
        scale_u += list(va[i])
        scale_q += list(qa[i])
        scale_u += list(vb[j])
        scale_q += list(qb[j])
    # TWO DEAD SIDES MUST NOT CERTIFY EACH OTHER.
    #
    # The jumps are relative to a scale, and both scales are floored at 1e-30
    # so the division is safe. That made the DEGENERATE case read as the
    # PERFECT one: two sides exporting identically zero traces and fluxes give
    # du = dq = 0, both ratios come out 0.0, and assess() then stamps
    # INTERFACE_SATISFIED because 0.0 < 5e-3. Measured on a live coupled run
    # whose three levels carried 132 and 88 interface points of literal 0.0 and
    # which reported an interface residual of 6.8e-11 as evidence of success.
    #
    # A ratio whose numerator AND denominator are both zero is not a small
    # number, it is an undefined one, and reporting 0.0 for it is the single
    # most flattering thing this module could do. The same lesson is already
    # written twice elsewhere in this tree: "d == 0 IS THE DEFECT AT ITS MOST
    # BLATANT, NOT A REASON TO SKIP" (result_audit), and the near-zero test
    # that must "INCLUDE exact zero" because three runs delivered literal 0.0
    # everywhere.
    #
    # The scale now includes BOTH sides, too. It was built from side A alone,
    # so the same physical defect scored thirty orders of magnitude apart
    # depending on which file happened to be named first.
    # PER CHANNEL, because a dead channel hides under a live one.
    #
    # The first version of this refusal required BOTH scales to be zero, and
    # re-testing it against the shape actually measured showed that is not
    # enough: a run whose displacements were real (max|u| = 4.87e-07) and whose
    # TRACTIONS were ~1e-18 and cancelling gave jump_q_rel = 0.0 and read as a
    # satisfied interface. That is the signature of a pair that exchanged
    # nothing while still solving -- each side returning the answer it would
    # have returned alone.
    #
    # A channel whose own scale is at round-off has no jump to measure: the
    # ratio is round-off over round-off, and reporting 0.0 for it is the most
    # flattering thing this function could do. So each channel is judged on its
    # own scale and refused on its own.
    #
    # This never speaks on a correctly coupled result: such a result has a real
    # jump on every channel it exchanges, so no channel sits at round-off. It
    # speaks only where a whole channel is dead across the pair.
    #
    # NOT THE SAME QUESTION AS "did this side export anything". A Dirichlet side
    # legitimately holds the seam at zero and exports a real recovered flux -- so
    # a check reading ONE side's exports.json must leave values-zero-and-flux-real
    # alone. This reads BOTH sides' interface files for a level and asks whether a
    # whole channel is dead across the pair, which is a different condition.
    _ROUNDOFF = 1e-14
    su, sq = _rms(scale_u), _rms(scale_q)
    dead = [name for name, scale in (("value", su), ("flux", sq))
            if scale <= _ROUNDOFF]
    if dead:
        both = len(dead) == 2
        return None, (
            f"the interface {' and '.join(dead)} channel"
            f"{'s are' if both else ' is'} identically zero on both sides "
            f"(scale at round-off), so there is no jump to measure there and "
            f"nothing here says the transmission condition holds"
            + ("" if both else
               f"; the other channel carries data, which is what a pair that "
               f"solved but exchanged nothing looks like"))
    return {"jump_u": _rms(du), "jump_q": _rms(dq),
            "jump_u_rel": _rms(du) / su,
            "jump_q_rel": _rms(dq) / sq}, "ok"


def recover_flux_from_field(field_pts, field_vals, iface_pts, k_normal,
                            normal_axis: int, iface_coord: float,
                            outward_sign: float):
    """Independent flux estimate from the agent's own FIELD submission.

    One-sided quadratic extrapolation of the normal derivative onto the
    interface, using the three probe columns nearest to it.  Accurate to
    ``O(h_probe^2)`` — far short of grading an order, and ample to catch an
    ``O(1)`` fabricated flux, which is all it is for.

    ``outward_sign`` is ``+1`` when the subdomain's outward normal points along
    ``+e_axis`` and ``-1`` otherwise.
    """
    rows: dict = {}
    for p, v in zip(field_pts, field_vals):
        key = tuple(round(c, 9) for i, c in enumerate(p) if i != normal_axis)
        rows.setdefault(key, []).append((p[normal_axis], v))
    out = []
    for p in iface_pts:
        key = tuple(round(c, 9) for i, c in enumerate(p) if i != normal_axis)
        col = sorted(rows.get(key, []), key=lambda a: abs(a[0] - iface_coord))
        if len(col) < 3:
            out.append(None)
            continue
        (x0, v0), (x1, v1), (x2, v2) = col[0], col[1], col[2]
        ncomp = len(v0)
        est = []
        for c in range(ncomp):
            y0, y1, y2 = v0[c], v1[c], v2[c]
            # derivative at iface_coord of the quadratic through the 3 points
            d01 = (y1 - y0) / (x1 - x0)
            d12 = (y2 - y1) / (x2 - x1)
            d012 = (d12 - d01) / (x2 - x0)
            dv = d01 + d012 * (2 * iface_coord - x0 - x1)
            est.append(-k_normal * dv * outward_sign)
        out.append(tuple(est))
    return out


def _lagrange3(xs, ys, x: float) -> float:
    """Three-point Lagrange value at x. The nodes need not be equispaced."""
    tot = 0.0
    for i in range(3):
        term = ys[i]
        for j in range(3):
            if i != j:
                term *= (x - xs[j]) / (xs[i] - xs[j])
        tot += term
    return tot


def _interp_tangential(column: dict, target: tuple, ncomp: int):
    """Value at an arbitrary tangential point inside one normal column.

    ``column`` maps a tangential key (tuple of the non-normal coordinates) to a
    value tuple.  Interpolation is tensor-product three-point Lagrange, applied
    one tangential axis at a time, so it works in 2-D (one tangential axis) and
    3-D (two) without special-casing either.
    """
    if not column:
        return None
    if not target:                                  # nothing left to reduce
        return next(iter(column.values()))
    axis_vals = sorted({k[0] for k in column})
    if len(axis_vals) < 3:
        return None
    near = sorted(axis_vals, key=lambda a: abs(a - target[0]))[:3]
    near.sort()
    sub = []
    for a in near:
        red = {k[1:]: v for k, v in column.items() if k[0] == a}
        got = _interp_tangential(red, target[1:], ncomp)
        if got is None:
            return None
        sub.append(got)
    return tuple(_lagrange3(near, [s[c] for s in sub], target[0])
                 for c in range(ncomp))


def recover_normal_derivative(field_pts, field_vals, iface_pts,
                              normal_axis: int, iface_coord: float,
                              outward_sign: float):
    """``-du/dn`` at the interface probes, from the agent's own FIELD file.

    WHY THIS EXISTS AND `recover_flux_from_field` DOES NOT SUFFICE. That
    function keys the field columns on the EXACT tangential coordinate of each
    interface probe, so it only works when the interface probes sit on the
    field grid's tangential lines.  They do not, and by construction: measured
    on a recorded coupled submission, the field grid is a midpoint rule on the
    subdomain box while the interface probes are a midpoint rule on a
    sub-interval of the interface at twice the density — **no interface probe
    shares a tangential coordinate with the field grid.**  So the recovery returned
    ``None`` for every point of every coupled submission ever graded, and the
    consistency check downstream could only ever answer NOT_ASSESSED.

    That is the sixth time in this campaign a mechanism was built, documented
    with measured numbers, and could not reach the case it was built for.  It
    is also the real reason issue #78 sat open behind "needs a key-schema
    field": the blocker was geometric, not schematic.

    METHOD. The field probes form a tensor grid, so for each of the three
    normal columns nearest the interface the value at the probe's tangential
    position is obtained by three-point Lagrange interpolation (error
    ``O(h_t^3)``), and the normal derivative at the interface is then the
    derivative of the quadratic through those three columns (``O(h_n^2)``,
    one-sided).  Ample for its purpose, which is catching an ``O(1)``
    invention, and far short of grading an order.

    Returns one value per interface point: ``-du/dn`` in the OUTWARD direction
    of this subdomain, or ``None`` where three columns were not available.
    """
    if not field_pts or not iface_pts:
        return [None] * len(iface_pts)
    ncomp = len(field_vals[0])
    cols: dict = {}
    for p, v in zip(field_pts, field_vals):
        n = round(p[normal_axis], 9)
        t = tuple(round(c, 9) for i, c in enumerate(p) if i != normal_axis)
        cols.setdefault(n, {})[t] = v
    normals = sorted(cols)
    if len(normals) < 3:
        return [None] * len(iface_pts)
    near = sorted(normals, key=lambda a: abs(a - iface_coord))[:3]
    near.sort()
    out = []
    for p in iface_pts:
        t = tuple(round(c, 9) for i, c in enumerate(p) if i != normal_axis)
        vals = [_interp_tangential(cols[a], t, ncomp) for a in near]
        if any(v is None for v in vals):
            out.append(None)
            continue
        est = []
        for c in range(ncomp):
            y0, y1, y2 = (v[c] for v in vals)
            x0, x1, x2 = near
            d01 = (y1 - y0) / (x1 - x0)
            d12 = (y2 - y1) / (x2 - x1)
            d012 = (d12 - d01) / (x2 - x0)
            dv = d01 + d012 * (2 * iface_coord - x0 - x1)
            est.append(-dv * outward_sign)          # = -du/dn, coefficient-free
        out.append(tuple(est))
    return out


def scalar_flux_components(spec: dict) -> tuple:
    """Which transmitted components ARE a scalar conduction flux ``-k du/dn``?

    The ratio test below assumes the reported flux is a CONSTANT times the
    normal derivative of the reported field. That is true for an isotropic
    scalar conduction flux and false for three families that this campaign
    contains, so applying it everywhere would invent defects:

    * **elasticity / thermoelastic traction** — ``t = sigma . n`` carries
      tangential derivatives (``t_x = (lam+2mu) du_x/dx + lam du_y/dy``), so
      ``t_x / (du_x/dx)`` is not constant even for a perfect solve. Measured on
      the runs on disk, C7/C9/C11/C12 produce 71 INCONSISTENT triples under the
      unrestricted test; the physics, not the agent, makes the ratio vary.
    * **anisotropic conduction** — ``q_n = -(K_xx du/dx + K_xy du/dy)`` has the
      same problem for any off-diagonal ``K`` (C6, D1).
    * **DSMC and FSI interfaces** (C13, C14) — the transmitted quantity is not
      a gradient of the reported field at all.

    Returns ``(components, reason)``. An UNKNOWN physics family returns no
    components and says so: a grader may abstain, never guess.
    """
    fam = (spec.get("physics_family") or "").strip()
    coeff = str(spec.get("coefficients") or "") + str(spec.get("subdomain_a") or "")
    anisotropic = "[[" in coeff
    scalar_families = {"diffusion", "reaction_diffusion", "transient_diffusion"}
    if fam in scalar_families and not anisotropic:
        return (0,), f"{fam} with an isotropic scalar conductivity"
    if fam in scalar_families and anisotropic:
        return (), (f"{fam} but the conductivity is a full tensor, so the "
                    f"normal flux mixes in the tangential derivative")
    if fam == "conjugate_heat_transfer":
        return ((), "anisotropic conjugate heat transfer: the normal flux "
                    "mixes in the tangential derivative") if anisotropic else \
               ((0,), "conjugate heat transfer with an isotropic conductivity")
    if fam == "thermo_mechanical":
        # component 0 is the temperature, whose flux IS -k dT/dn; the traction
        # components that follow are not
        return ((0,), "thermoelasticity: the temperature flux only, not the "
                      "traction") if not anisotropic else \
               ((), "anisotropic thermoelasticity")
    if fam in ("elasticity", "fem_dsmc", "fluid_structure_interaction"):
        return (), f"{fam}: the transmitted quantity is not -k du/dn"
    return (), (f"unknown physics family {fam!r}: the check abstains rather "
                f"than assume the transmitted flux is a scalar conduction flux")



def _nonfinite_note(pairs, c):
    """A NOT_ASSESSED entry for component `c` when the reported flux or the field's normal
    derivative holds a value that is not finite, else None. Every comparison with a NaN is
    False, so a NaN fell through to CONSISTENT (measured: one NaN among eight points of a flux
    twice the derivative read CONSISTENT here, and an inf read CONSISTENT in the multiple)."""
    nq = sum(1 for r, _g in pairs if not math.isfinite(float(r[c])))
    ng = sum(1 for _r, g in pairs if not math.isfinite(float(g[c])))
    if not (nq or ng):
        return None
    what = " and ".join(w for w in (
        f"the reported flux holds {nq} NaN or infinite value{'s' if nq != 1 else ''}" if nq else "",
        f"the field's normal derivative holds {ng} NaN or infinite value{'s' if ng != 1 else ''}"
        if ng else "") if w)
    return {"component": c, "verdict": "NOT_ASSESSED",
            "detail": f"{what}: nothing is checked and nothing passes"}


def _not_assessed_detail(per_comp) -> str:
    """The overall NOT_ASSESSED detail: a component's own reason when it is the one about a
    non-finite value, so the verdict says why nothing passed."""
    for d in per_comp:
        if "NaN or infinite" in d.get("detail", ""):
            return d["detail"]
    return "no component could be assessed"

def flux_ratio_consistency(reported, dudn, spread_tol: float = 0.30,
                           floor_frac: float = 0.05, components=None):
    """Is the reported flux the agent's own field times a CONSTANT?

    WHY NOT COMPARE AGAINST ``k * du/dn`` DIRECTLY. That needs the per-side
    conductivity as structured data, which the specs carry only as prose
    ("thermal conductivity k = 1 in subdomain A"), and prose parsing in a
    grader is how a grader starts inventing.  It is also unnecessary: for a
    flux the agent really computed from its own solution,

        q_n(x) / (-du/dn)(x)  ==  k   for every interface point,

    so the RATIO IS CONSTANT ALONG THE INTERFACE whatever k is.  A reported
    profile that was not computed from the submitted field has no reason to
    hold that ratio, and the test needs no coefficient, no key, and no
    reference solution — it runs with the answers sealed.

    Points where ``|du/dn|`` is below ``floor_frac`` of its own RMS are
    dropped: there the ratio is a small number over a smaller one and says
    nothing.  A NEGATIVE implied coefficient is reported separately — that is
    not invention but a violated sign convention, which the task states and
    the gate is entitled to grade.
    """
    pairs = [(r, g) for r, g in zip(reported, dudn) if g is not None]
    if not pairs:
        return {"verdict": "NOT_ASSESSED",
                "detail": "the field could not be interpolated to the "
                          "interface probes (fewer than three probe columns, "
                          "or the field probes are not a tensor grid)"}
    ncomp = min(len(pairs[0][0]), len(pairs[0][1]))
    if components is not None:
        want = [c for c in components if 0 <= c < ncomp]
        if not want:
            return {"verdict": "NOT_APPLICABLE",
                    "detail": "no transmitted component of this problem is a "
                              "scalar conduction flux, so a constant ratio to "
                              "the normal derivative is not expected and "
                              "nothing is asserted"}
    else:
        want = list(range(ncomp))
    rms = [_rms([abs(g[c]) for _r, g in pairs]) for c in range(ncomp)]
    rep_rms = [_rms([abs(r[c]) for r, _g in pairs]) for c in range(ncomp)]
    per_comp = []
    for c in want:
        _nf = _nonfinite_note(pairs, c)
        if _nf:
            per_comp.append(_nf)
            continue
        # A REPORTED FLUX OF ZERO IS NOT A CONSTANT RATIO. Measured on real
        # submissions: C10_27b_BARE_seed4 and C1_27b_BARE_seed4 export fluxes
        # that are identically zero, so every ratio is 0/x = 0, the spread is
        # exactly 0.0, and the first version of this function called them
        # CONSISTENT — blessing the one case the coupled grader elsewhere
        # refuses outright. C3_27b_BARE_seed6's side B does the same and came
        # out INCONSISTENT with a spread of 1e299, which accuses the agent of
        # invention when the honest statement is that it reported nothing.
        if rep_rms[c] <= 0.0:
            per_comp.append({
                "component": c, "verdict": "NOT_ASSESSED",
                "detail": "the reported flux is identically zero, so there is "
                          "no profile to compare with the field: nothing is "
                          "checked and nothing passes"})
            continue
        keep = [(r[c], g[c]) for r, g in pairs
                if abs(g[c]) >= floor_frac * max(rms[c], 1e-300)]
        if len(keep) < 3:
            # A submitted field with no gradient at the interface is not an
            # abstention for want of geometry — it is a statement about the
            # field, and the grader should say which of the two it means.
            flat = rms[c] <= 0.0 or all(abs(g[c]) <= 0.0 for _r, g in pairs)
            per_comp.append({"component": c, "verdict": "NOT_ASSESSED",
                             "detail": ("the submitted field is flat at the "
                                        "interface: its normal derivative is "
                                        "identically zero, so there is nothing "
                                        "for the reported flux to follow from"
                                        if flat else
                                        "fewer than three interface points "
                                        "carry a normal derivative above 5% of "
                                        "its own RMS")})
            continue
        ratios = sorted(q / d for q, d in keep)
        n = len(ratios)
        med = ratios[n // 2]
        lo = ratios[max(0, int(0.10 * n))]
        hi = ratios[min(n - 1, int(0.90 * n))]
        spread = (hi - lo) / max(abs(med), 1e-300)
        per_comp.append({
            "component": c, "implied_coefficient": med, "spread": spread,
            "n_points": n,
            "verdict": ("SIGN_CONVENTION" if med < 0 else
                        "CONSISTENT" if spread <= spread_tol else
                        "INCONSISTENT")})
    # WORST ACROSS COMPONENTS, never best. Each transmitted component is a
    # separate physical condition: C1 transmits a temperature and two traction
    # components. The first version reported CONSISTENT whenever ANY component
    # was consistent, so a component with a reversed sign was absorbed by a
    # healthy neighbour — 262 triples were booked CONSISTENT and their spreads
    # went up to 2.54, which is only possible if a flipped component was hiding
    # inside a passing verdict.
    bad = [d for d in per_comp if d["verdict"] == "INCONSISTENT"]
    flipped = [d for d in per_comp if d["verdict"] == "SIGN_CONVENTION"]
    ok = [d for d in per_comp if d["verdict"] == "CONSISTENT"]
    if bad:
        verdict, detail = "INCONSISTENT", (
            "the reported interface flux is not a constant multiple of the "
            "normal derivative of the submitted field, so the two were not "
            "computed from each other: implied coefficient varies by "
            + ", ".join(f"{d['spread']:.0%} (component {d['component']})"
                        for d in bad))
    elif flipped:
        verdict, detail = "SIGN_CONVENTION", (
            "the reported flux is proportional to the field's normal "
            "derivative but with the OPPOSITE sign on component(s) "
            + ", ".join(str(d["component"]) for d in flipped)
            + ": the outward-normal convention the task states was not "
              "followed")
    elif ok:
        verdict, detail = "CONSISTENT", (
            "the reported flux is the submitted field's normal derivative "
            "times a constant, as it must be: implied coefficient "
            + ", ".join(f"{d['implied_coefficient']:.4g} (+/-{d['spread']:.1%})"
                        for d in ok))
    else:
        verdict, detail = "NOT_ASSESSED", _not_assessed_detail(per_comp)
    return {"verdict": verdict, "detail": detail, "per_component": per_comp,
            "spread_tolerance": spread_tol}


def flux_multiple_consistency(reported, dudn, rtol: float = 0.30,
                              floor_frac: float = 0.05, components=None):
    """Is the reported flux a CONSTANT multiple of the field's normal derivative?

    Read as the best constant multiple, ``k = sum(q g) / sum(g g)`` over the
    interface points, and how far the flux lies from it: ``misfit = |q - k g| /
    |q|`` (root mean square). Same verdicts as :func:`flux_ratio_consistency`:
    CONSISTENT (misfit within ``rtol``, ``k > 0``), SIGN_CONVENTION (a clean
    multiple with ``k < 0``), INCONSISTENT, NOT_ASSESSED.

    WHY NOT THE SPREAD OF q / g. The ratio is ill-conditioned wherever the normal
    derivative passes through zero and next to a corner of the interface, where
    a one-sided difference is least accurate. Measured on right fields read leg by
    leg across a bent interface: the 10-90 % spread of q / g read 130 % -> 204 %
    -> 43 % over three levels, while the misfit of the best constant multiple read
    36 % -> 28 % -> 11 %; on a straight one 31 % -> 12 % -> 8 % against 12 % -> 5 %
    -> 3 %. Each point weighs by its own gradient, so a point where both pass
    through zero moves nothing. A scaled flux keeps a small misfit: its ``k`` is
    what shows it, against the stated coefficient.
    """
    pairs = [(r, g) for r, g in zip(reported, dudn) if g is not None]
    if not pairs:
        return {"verdict": "NOT_ASSESSED",
                "detail": "the field could not be interpolated to the "
                          "interface probes (fewer than three probe columns, "
                          "or the field probes are not a tensor grid)"}
    ncomp = min(len(pairs[0][0]), len(pairs[0][1]))
    want = ([c for c in components if 0 <= c < ncomp] if components is not None
            else list(range(ncomp)))
    per_comp = []
    for c in want:
        _nf = _nonfinite_note(pairs, c)
        if _nf:
            per_comp.append(_nf)
            continue
        q = [float(r[c]) for r, _g in pairs]
        g = [float(d[c]) for _r, d in pairs]
        qq, gg = sum(v * v for v in q), sum(v * v for v in g)
        if qq <= 0.0:
            per_comp.append({"component": c, "verdict": "NOT_ASSESSED",
                             "detail": "the reported flux is identically zero, so there is "
                                       "no profile to compare with the field"})
            continue
        grms = math.sqrt(gg / len(g))
        if gg <= 0.0 or sum(1 for v in g if abs(v) >= floor_frac * grms) < 3:
            per_comp.append({"component": c, "verdict": "NOT_ASSESSED",
                             "detail": ("the submitted field is flat at the interface: its "
                                        "normal derivative is identically zero" if gg <= 0.0 else
                                        "fewer than three interface points carry a normal "
                                        "derivative above 5% of its own RMS")})
            continue
        k = sum(a * b for a, b in zip(q, g)) / gg
        misfit = math.sqrt(sum((a - k * b) ** 2 for a, b in zip(q, g)) / qq)
        per_comp.append({"component": c, "implied_coefficient": k, "misfit": misfit,
                         "n_points": len(q),
                         "verdict": ("INCONSISTENT" if misfit > rtol else
                                     "SIGN_CONVENTION" if k < 0 else "CONSISTENT")})
    bad = [d for d in per_comp if d["verdict"] == "INCONSISTENT"]
    flipped = [d for d in per_comp if d["verdict"] == "SIGN_CONVENTION"]
    ok = [d for d in per_comp if d["verdict"] == "CONSISTENT"]
    if bad:
        verdict, detail = "INCONSISTENT", (
            "the reported interface flux is not a constant multiple of the normal "
            "derivative of the submitted field: it differs from the best such multiple by "
            + ", ".join(f"{d['misfit']:.0%} (component {d['component']})" for d in bad))
    elif flipped:
        verdict, detail = "SIGN_CONVENTION", (
            "the reported flux is a constant multiple of the field's normal derivative, "
            "with the OPPOSITE sign on component(s) "
            + ", ".join(str(d["component"]) for d in flipped)
            + ": the outward-normal convention the task states was not followed")
    elif ok:
        verdict, detail = "CONSISTENT", (
            "the reported flux is the submitted field's normal derivative times a "
            "constant, as it must be: implied coefficient "
            + ", ".join(f"{d['implied_coefficient']:.4g} (misfit {d['misfit']:.1%})" for d in ok))
    else:
        verdict, detail = "NOT_ASSESSED", _not_assessed_detail(per_comp)
    return {"verdict": verdict, "detail": detail, "per_component": per_comp,
            "misfit_tolerance": rtol}


def flux_consistency(reported, recovered, rtol: float = 0.25):
    """Does the reported interface flux match one recomputed from the field?"""
    pairs = [(r, g) for r, g in zip(reported, recovered) if g is not None]
    if not pairs:
        return {"verdict": "NOT_ASSESSED",
                "detail": "no probe column had three points to extrapolate from"}
    num = _rms([a - b for r, g in pairs for a, b in zip(r, g)])
    den = max(_rms([a for r, _ in pairs for a in r]), 1e-30)
    rel = num / den
    return {"verdict": "CONSISTENT" if rel <= rtol else "INCONSISTENT",
            "relative_difference": rel, "tolerance": rtol,
            "detail": ("the reported interface flux does not follow from the "
                       "submitted field: one of the two was not computed from "
                       "the other" if rel > rtol else
                       "reported flux agrees with the field-derived estimate")}


def order_from_halving(seq) -> float | None:
    s = [float(v) for v in seq]
    if len(s) < 2 or any(v <= 0 for v in s):
        return None
    return sum(math.log2(s[i] / s[i + 1]) for i in range(len(s) - 1)) / (len(s) - 1)


def assess(per_level: list, jump_tol: float = 5e-3) -> InterfaceReport:
    """``per_level`` is coarse-to-fine, each entry the output of two_sided_jumps."""
    rep = InterfaceReport()
    rep.levels = list(range(1, len(per_level) + 1))
    rep.jump_u_rel = [d["jump_u_rel"] for d in per_level]
    rep.jump_q_rel = [d["jump_q_rel"] for d in per_level]
    if not per_level:
        rep.verdict = "NOT_ASSESSED"
        return rep
    worst_q = max(rep.jump_q_rel)
    worst_u = max(rep.jump_u_rel)
    if worst_q > jump_tol or worst_u > jump_tol:
        rep.verdict = "INTERFACE_NOT_SATISFIED"
        rep.notes.append(
            f"the two subdomains disagree at the interface: max relative "
            f"field jump {worst_u:.3e}, max relative FLUX jump {worst_q:.3e}, "
            f"tolerance {jump_tol:g}. A flux jump that does not fall under "
            f"refinement means the scheme converged to a fixed point of the "
            f"wrong transmission condition — which the observed order cannot "
            f"see, because convergence to a wrong answer is still convergence.")
    else:
        rep.verdict = "INTERFACE_SATISFIED"
    return rep
