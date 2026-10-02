"""Does the delivered field actually satisfy the equation the task stated?

WHY THIS EXISTS. Measured over 464 single-code runs, among
those with a complete level set the SELF-convergence order — computed
from the agent's own numbers, no reference — has a median of 1.96 (bare) and
1.99 (openPASO). The discretisations converge cleanly. So an order near zero
against an independent reference is almost never the finite element method
failing to converge; it is a field converging beautifully TO THE WRONG
FUNCTION (7% of all runs), or a field whose overall size is wrong by orders
of magnitude (11-30% of runs).

A refinement study cannot see either one, and neither can the agent's own
verdict: MESH_INDEPENDENCE = NOT_CONVERGED catches about three quarters of the
wrong runs but also fires on HALF the correct ones, so on its own it is close to
uninformative.

WHAT THIS CHECKS, AND WHY IT LEAKS NO REFERENCE SOLUTION. For any smooth test
function v that vanishes on the boundary, a field u solving

    L u = -div(K grad u) = f          with u prescribed on the boundary

satisfies the weak identity

    integral of u * (L* v)  ==  integral of f * v

where L* is the adjoint, and L* = L for symmetric constant K. Every ingredient
is PUBLIC: the operator and the source come from the task text, and the values
come from the agent's own run. No exact solution is used, none is
revealed, and nothing here can be run backwards to obtain one — a single
scalar identity per test function cannot reconstruct a field.

MEASURED SEPARATION, by execution on 26 anisotropic-Poisson runs, whose probe grid
is a 44x44 midpoint rule so the quadrature is exact to O(h^2):

    solves the stated PDE   2.32e-02 -> 5.27e-03 -> 1.23e-03 -> 2.56e-04
    does not                6.25e+00 -> 6.37e+00 -> 6.40e+00 -> 6.40e+00
    does not (other cause)  3.13e+00 -> 3.01e+00 -> 2.96e+00 -> 2.93e+00

Four orders of magnitude apart at the finest level, and the failing ones are
flat, which is the signature: a wrong operator, a wrong source, or a source
evaluated in the wrong coordinate frame leaves a residual that refinement
cannot remove.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class LevelResult:
    level: int
    n_points: int
    residual: float
    detail: str = ""
    umax: float = 0.0           # the field's largest magnitude at this level
    scale: float = float("nan")     # the discrete check: A u as a multiple of the load (least squares)
    left: float = float("nan")      # ... and what is left of the load after that factor


@dataclass
class ConsistencyResult:
    levels: list = field(default_factory=list)
    rate: float | None = None
    verdict: str = "UNKNOWN"
    explanation: str = ""

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "explanation": self.explanation,
            "observed_rate": self.rate,
            "levels": [{"level": r.level, "points": r.n_points,
                        "relative_weak_residual": r.residual,
                        "detail": r.detail} for r in self.levels],
        }


def _detect_midpoint_grid(coords: list, box: list | None = None) -> tuple:
    """Is this a tensor grid of cell midpoints, and what is the cell volume?

    The quadrature below is a midpoint rule, which is second-order accurate on
    exactly this arrangement and meaningless on a scatter. Returning the weight
    ONLY when the arrangement is right is what keeps a number from being
    reported that the method cannot support.

    UNIFORM WAS NOT ENOUGH, AND THE GAP BLESSED A WRONG FIELD. Given the box,
    the points must also be the CENTRES of its cells: the outermost one sits
    half a step inside each face. A VERTEX grid -- the corners of the same
    tiling, which is what a participant's own mesh dump is -- is just as
    uniform, and under the midpoint weight its boundary row carries a whole
    cell where it should carry half. Measured on a coupled run that was correct, over its own mesh dumps: the delivered field read 0.370, 0.200,
    0.104 and the same field scaled by 1.20 read 0.245, 0.040, 0.076 -- the
    20 % error scoring BETTER than the truth, both CONSISTENT. On that run's
    probe-grid deliverables, true cell midpoints, the same check separates them
    (1.79e-02 -> 1.63e-03 against a flat 1.9e-01).

    `box` stays optional so a caller that has none keeps the old behaviour.
    """
    axes = []
    for d in range(len(coords[0])):
        vals = sorted({round(p[d], 12) for p in coords})
        if len(vals) < 2:
            return None, "an axis carries a single distinct coordinate"
        steps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        h = sum(steps) / len(steps)
        if h <= 0 or max(abs(s - h) for s in steps) > 1e-9 * max(h, 1e-30):
            return None, "the points are not uniformly spaced along every axis"
        axes.append((vals, h))
    if len(coords) != math.prod(len(v) for v, _ in axes):
        return None, "the points do not fill a full tensor grid"
    if box is not None:
        for d, (vals, h) in enumerate(axes):
            try:
                lo, hi = float(box[d][0]), float(box[d][1])
            except (IndexError, TypeError, ValueError):
                break
            near_lo, near_hi = vals[0] - lo, hi - vals[-1]
            if (abs(near_lo - h / 2) > 0.05 * h) or (abs(near_hi - h / 2) > 0.05 * h):
                shape = ("the corners of the cells (a vertex grid)"
                         if min(abs(near_lo), abs(near_hi)) < 0.05 * h
                         else f"offset {near_lo:.4g} and {near_hi:.4g} from the faces")
                return None, (
                    f"along axis {d} the points are not the CENTRES of the box's "
                    f"cells but {shape}, and this check integrates with a midpoint "
                    f"rule: every point carries one whole cell, so a row sitting on "
                    f"a face carries twice its share. Hand in the field at the "
                    f"points your task prescribes -- a probe grid is cell midpoints "
                    f"-- rather than your own mesh nodes, or the residual measures "
                    f"the quadrature instead of the field")
    weight = 1.0
    for _, h in axes:
        weight *= h
    return weight, ""


def _adjoint_of_v_flat(coeff, pts, box):
    """L* v and v for a v whose VALUE AND SLOPE both vanish on the box.

    WHY THIS v AND NOT THE OBVIOUS ONE. Integrating the identity by parts twice
    leaves two boundary terms:

        boundary integral of  -K grad(u).n v   -> zero because v = 0
        boundary integral of   u K grad(v).n   -> survives unless u = 0 there

    The natural choice, a product of half-period sines, is zero on every face
    but its slope is not, so the second term only vanishes when the FIELD is
    zero on the boundary -- which one side of a coupled problem never is,
    because the interface carries the partner's data. That version refused
    nearly every coupled side: measured across 94 graded coupled cells, side B
    came back NOT_APPLICABLE 92 times.

    v = prod sin^2 has value and normal derivative both zero on every face, so
    BOTH terms vanish for any u at all, and it is used for every case.

    Measured on a manufactured case deliberately not zero on the boundary
    (u = sin(pi x) sin(pi y) + x, so -lap u = 2 pi^2 sin sin), levels 16 to 128:

        v                 correct field          wrong field (0.5x amplitude)
        sines             8.13e-01 flat          3.13e-01 flat
        sin^2 (here)      4.87e-03 -> 7.53e-05   5.02e-01 flat
                          falls 64.7x

    The sines cannot tell them apart at all, and rank the correct field WORSE.
    sin^2 separates them, and the fall is the documented factor of four per
    refinement. It also holds at k = 200 on an off-unit box, catches a 3 %
    amplitude error (3.47e-02 flat), and catches the near-zero field shape that
    C9 keeps producing. Where u IS zero on the boundary the sines are exact
    (residual 1.8e-16) and this one merely converges -- that is the whole cost,
    and the verdict reads the fall rather than an absolute floor.

    Replayed over the 94 graded coupled cells of the accepted operator
    (benchmarks/pde_check_calibration):

        v                          false alarms   wrong caught   no verdict
        sines                          1 of 14        6 of 17           78
        sin^2 behind a size test       1 of 14       16 of 17           35
        sin^2 always (this)            0 of 14       14 of 17           35
    """
    import numpy as np
    dim = len(box)
    xs = [np.asarray([p[d] for p in pts], dtype=float) for d in range(dim)]
    Ls = [float(hi - lo) for lo, hi in box]
    args = [math.pi * (xs[d] - box[d][0]) / Ls[d] for d in range(dim)]
    sin2 = [np.sin(a) ** 2 for a in args]

    v = np.ones_like(xs[0])
    for s in sin2:
        v = v * s

    # d2/dx_i^2 of sin^2(a_i) = 2 (pi/L_i)^2 cos(2 a_i); the cross terms
    # d2/dx_i dx_j carry sin(2a_i) sin(2a_j) (pi/L_i)(pi/L_j).
    def d2(i, j):
        out = np.ones_like(xs[0])
        for d in range(dim):
            if i == j:
                out = out * (2.0 * (math.pi / Ls[d]) ** 2 * np.cos(2 * args[d])
                             if d == i else sin2[d])
            elif d == i or d == j:
                out = out * (math.pi / Ls[d]) * np.sin(2 * args[d])
            else:
                out = out * sin2[d]
        return out

    K = _as_tensor(coeff, dim)
    Lv = np.zeros_like(xs[0])
    for i in range(dim):
        for j in range(dim):
            if K[i][j] != 0.0:
                Lv = Lv - K[i][j] * d2(i, j)
    return v, Lv


def _as_tensor(coeff, dim: int) -> list:
    if isinstance(coeff, (int, float)):
        return [[float(coeff) if i == j else 0.0 for j in range(dim)]
                for i in range(dim)]
    rows = [[float(c) for c in row] for row in coeff]
    if len(rows) != dim or any(len(r) != dim for r in rows):
        raise ValueError(f"coefficient tensor is not {dim}x{dim}")
    for i in range(dim):
        for j in range(i + 1, dim):
            if abs(rows[i][j] - rows[j][i]) > 1e-12 * max(1.0, abs(rows[i][j])):
                raise ValueError(
                    "the coefficient tensor is not symmetric; this check "
                    "assumes a self-adjoint operator, so a non-symmetric K "
                    "needs the true adjoint and is refused rather than "
                    "answered wrongly")
    return rows


def _eval_source(expr: str, pts, dim: int):
    """Evaluate the task's source expression at the points, in PYTHON syntax."""
    import numpy as np
    names = ["x", "y", "z"][:dim]
    env = {n: np.asarray([p[d] for p in pts], dtype=float)
           for d, n in enumerate(names)}
    env.update({"np": np, "pi": math.pi, "sin": np.sin, "cos": np.cos,
                "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                "tan": np.tan, "tanh": np.tanh, "sinh": np.sinh,
                "cosh": np.cosh, "abs": np.abs, "Abs": np.abs})
    out = eval(expr, {"__builtins__": {}}, env)              # noqa: S307
    return np.asarray(out, dtype=float) * np.ones_like(env[names[0]])


def check_levels(levels: dict, source_expr: str, coefficient,
                 box: list, reaction: float = 0.0) -> ConsistencyResult:
    """levels maps a level number to a list of (coords..., u) rows.

    `reaction` is the constant c of -div(K grad u) + c u = f. The operator stays
    self-adjoint, so the adjoint of the test function gains exactly + c v and
    the identity is otherwise unchanged; c = 0 reproduces the pure-diffusion
    behaviour bit for bit.

    WHY IT IS HERE. Without it the check judges a reacting field against a
    different equation than the one that produced it. Measured on two
    coupled runs that were correct, over their own delivered probe
    files: side A (c = 10) reads 1.79e-02 -> 5.18e-03 -> 1.63e-03 CONSISTENT
    with the term and 3.61e-01 -> 3.54e-01 -> 3.52e-01 INCONSISTENT without it.
    A flat 35 % residual on the one field that is right is not a weak check, it
    is a wrong one, and every reacting side of every coupled problem would get
    it.
    """
    import numpy as np
    res = ConsistencyResult()
    for lvl in sorted(levels):
        rows = levels[lvl]
        if not rows:
            res.levels.append(LevelResult(lvl, 0, float("nan"),
                                          "no rows"))
            continue
        dim = len(rows[0]) - 1
        pts = [r[:dim] for r in rows]
        u = np.asarray([r[dim] for r in rows], dtype=float)
        if not np.all(np.isfinite(u)):
            res.levels.append(LevelResult(
                lvl, len(rows), float("nan"),
                "the delivered field carries a non-finite value"))
            continue
        weight, why = _detect_midpoint_grid(pts, box)
        if weight is None:
            res.levels.append(LevelResult(lvl, len(rows), float("nan"), why))
            continue
        # THE IDENTITY NEEDS u = 0 ON THE BOUNDARY, NOT ONLY v = 0.
        #
        # Integrating by parts twice leaves
        #     int u (L* v) - int (L u) v = - closed_int u k dv/dn + closed_int v k du/dn
        # and v vanishing on the box kills only the SECOND term. The first
        # survives unless u vanishes there too. So on a subdomain with a
        # non-zero trace on any face -- which is EVERY side of a partitioned
        # coupling, since the interface carries data -- the identity fails for
        # a perfectly correct field.
        #
        # Measured on one coupled problem, whose side B has an interface trace
        # of about -3.7e-03: this check called side B INCONSISTENT for all four
        # runs examined, INCLUDING one verified correct against an independent
        # reference with order 1.95. A false accusation against the one right
        # answer, and the coupling payload was telling agents to run it on
        # each side.
        #
        # Detected from the delivered field alone: on a grid of cell midpoints
        # a field vanishing on the boundary has an outermost layer of size
        # O(h) relative to its own scale, while a face carrying data does not.
        # A FIELD CARRYING DATA ON ITS BOUNDARY IS THE NORMAL COUPLED CASE, AND
        # IT USED TO BE REFUSED. The sines below vanish on every face but their
        # slope does not, so the identity needs u = 0 there too. Measured across
        # 94 graded coupled cells, that refused side B 92 times -- i.e. this
        # check could not speak about one side of almost any coupled problem,
        # which is exactly where it was about to be run by default.
        #
        # `_adjoint_of_v_flat` uses a v whose value AND slope vanish, so both
        # boundary terms go for any u, and it is used UNCONDITIONALLY.
        #
        # IT WAS A FALLBACK BEHIND A SIZE TEST ON THAT OUTERMOST LAYER, AND THE
        # SIZE TEST CANNOT BE MADE TO WORK. On a midpoint grid the outermost
        # probe sits h/2 inside the face, so a field that IS zero on the
        # boundary still reads |grad u| h/2 there -- percent-level, the same
        # order as a face genuinely carrying a partner's data. The two cases
        # are not separable by magnitude. Measured on a coupled run that was
        # CORRECT, has a layer of 0.067 of its own scale, went to the sines,
        # and was called INCONSISTENT -- 1.32e-02 -> 5.79e-03 -> 1.06e-02, flat
        # and non-monotone. The same three files under the v below fall
        # 1.25e-02 -> 3.76e-03 -> 1.59e-03, monotone, and read CONSISTENT. The
        # field was right the whole time; the routing was wrong.
        #
        # The sines are exact (1.8e-16) where u really does vanish and this v
        # merely converges there, which is the only thing given up, and the
        # verdict reads the FALL not an absolute floor. Convergence is enough.
        v, Lv = _adjoint_of_v_flat(coefficient, pts, box)
        if reaction:
            Lv = Lv + float(reaction) * v
        f = _eval_source(source_expr, pts, dim)
        lhs = float(np.sum(u * Lv) * weight)
        rhs = float(np.sum(f * v) * weight)
        denom = max(abs(rhs), 1e-300)
        res.levels.append(LevelResult(lvl, len(rows), abs(lhs - rhs) / denom,
                                      f"lhs={lhs:.6e} rhs={rhs:.6e}",
                                      umax=float(np.abs(u).max())))
    return _decide(res)


def tensor_grid(rows, tol: float = 1e-7):
    """(xs, ys, U) when the (x, y, u) rows are the nodes of one tensor grid, each once;
    None otherwise (a scattered, crossed or repeated node set)."""
    import numpy as np
    a = np.asarray(rows, dtype=float)
    if a.ndim != 2 or a.shape[1] < 3 or len(a) < 9 or not np.isfinite(a).all():
        return None
    unit = tol * max(float(np.ptp(a[:, 0])), float(np.ptp(a[:, 1])), 1e-300)
    qx, ix = np.unique(np.round(a[:, 0] / unit), return_inverse=True)
    qy, iy = np.unique(np.round(a[:, 1] / unit), return_inverse=True)
    if len(qx) < 3 or len(qy) < 3 or len(qx) * len(qy) != len(a):
        return None
    U = np.full((len(qx), len(qy)), np.nan)
    U[ix, iy] = a[:, 2]
    if np.isnan(U).any():
        return None                                  # a node listed twice, another missing
    xs = np.bincount(ix, weights=a[:, 0]) / np.bincount(ix)
    ys = np.bincount(iy, weights=a[:, 1]) / np.bincount(iy)
    return xs, ys, U


# The 7-point, degree-5 rule on the reference triangle (barycentric l1, l2; weights sum to 1).
_TRI7 = ((1 / 3, 1 / 3, 0.225),
         (0.059715871789770, 0.470142064105115, 0.132394152788506),
         (0.470142064105115, 0.059715871789770, 0.132394152788506),
         (0.470142064105115, 0.470142064105115, 0.132394152788506),
         (0.797426985353087, 0.101286507323456, 0.125939180544827),
         (0.101286507323456, 0.797426985353087, 0.125939180544827),
         (0.101286507323456, 0.101286507323456, 0.125939180544827))
_GAUSS3 = ((-0.774596669241483, 5 / 9), (0.0, 8 / 9), (0.774596669241483, 5 / 9))


def check_levels_on_grid(raw_levels: dict, source_expr: str, coefficient, box: list,
                         reaction: float = 0.0, rule: str = "anti") -> ConsistencyResult:
    """The identity of check_levels, integrated EXACTLY over the field a tensor grid's
    nodes carry under one reconstruction: P1 with every cell cut along its main
    diagonal ('main'), along the other one ('anti'), or bilinear cells ('q1').

    WHY. check_levels reads each dump through a Delaunay rebuild, and on a tensor grid
    every cell's four corners are cocircular: the rebuild picks either diagonal, cell
    by cell (measured: 31 / 33, 128 / 128, 52 / 48 of each). Its value error is O(h^2),
    the size of the residual itself, and on one side that solved its own P1 system to
    1e-12 the two nearly cancelled: 5.990e-04 -> 5.826e-04 -> 2.177e-04 and DOES NOT
    SATISFY, where the same field on its own triangles falls 9.84e-4 -> 3.51e-4 ->
    9.39e-5. A structured solver's field is one of these reconstructions exactly."""
    import numpy as np
    res = ConsistencyResult()
    for lvl in sorted(raw_levels):
        g = tensor_grid(raw_levels[lvl])
        if g is None:
            res.levels.append(LevelResult(lvl, len(raw_levels[lvl]), float("nan"),
                                          "not a tensor grid"))
            continue
        xs, ys, U = g
        X0, X1 = np.meshgrid(xs[:-1], ys[:-1], indexing="ij")
        HX = np.diff(xs)[:, None] * np.ones((1, len(ys) - 1))
        HY = np.ones((len(xs) - 1, 1)) * np.diff(ys)[None, :]
        u00, u10, u01, u11 = U[:-1, :-1], U[1:, :-1], U[:-1, 1:], U[1:, 1:]
        pts, wts, uq = [], [], []
        if rule == "q1":
            for s, ws in _GAUSS3:
                for t, wt in _GAUSS3:
                    a, b = 0.5 * (1 + s), 0.5 * (1 + t)
                    pts.append(np.stack([X0 + a * HX, X1 + b * HY], -1).reshape(-1, 2))
                    wts.append((0.25 * ws * wt * HX * HY).ravel())
                    uq.append(((1 - a) * (1 - b) * u00 + a * (1 - b) * u10
                               + (1 - a) * b * u01 + a * b * u11).ravel())
        else:
            # corners as (a, b) in the cell's unit square, with their values
            c = {(0, 0): u00, (1, 0): u10, (0, 1): u01, (1, 1): u11}
            tris = (((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (0, 1))) if rule == "main" else \
                   (((0, 0), (1, 0), (0, 1)), ((1, 0), (1, 1), (0, 1)))
            for p0, p1, p2 in tris:
                for l1, l2, w in _TRI7:
                    l0 = 1.0 - l1 - l2
                    a = l0 * p0[0] + l1 * p1[0] + l2 * p2[0]
                    b = l0 * p0[1] + l1 * p1[1] + l2 * p2[1]
                    pts.append(np.stack([X0 + a * HX, X1 + b * HY], -1).reshape(-1, 2))
                    wts.append((w * 0.5 * HX * HY).ravel())
                    uq.append((l0 * c[p0] + l1 * c[p1] + l2 * c[p2]).ravel())
        P = np.concatenate(pts)
        W = np.concatenate(wts)
        u = np.concatenate(uq)
        v, Lv = _adjoint_of_v_flat(coefficient, P, box)
        if reaction:
            Lv = Lv + float(reaction) * v
        f = _eval_source(source_expr, P, 2)
        lhs = float(np.sum(W * u * Lv))
        rhs = float(np.sum(W * f * v))
        res.levels.append(LevelResult(lvl, int(U.size), abs(lhs - rhs) / max(abs(rhs), 1e-300),
                                      f"lhs={lhs:.6e} rhs={rhs:.6e} ({rule})",
                                      umax=float(np.abs(U).max())))
    return _decide(res)



# ── THE DISCRETE RESIDUAL, ON THE SIDE'S OWN NODES ───────────────────────────
#
# WHY. The weak identity above integrates against a smooth test function, so it reads the
# FALL of a residual that is O(h^2) for every right field; a wrong field is told apart only
# once three levels show it flat. A field that IS the solution of its own discrete system is
# told apart at once: rebuild the system on the nodes the side dumped and apply it to the
# dumped values, and at every node inside the box the residual is round-off. Measured on a
# coupled round with a 2x2 conductivity, on the sides' own dumps: the right sides read
# 7e-12 .. 2e-9 of the load at every level (deal.II on its bilinear cells, NGSolve on its
# nodes' triangles, which here are netgen's own), the same fields judged with the scalar
# K[0][0] read 0.45 .. 1.0, and a side that had solved with K[0][0] read 0.59 .. 0.65.
#
# IT CONFIRMS, AND CONDEMNS ONLY A LOAD OFF BY ONE FACTOR. The rebuild is exact only when the
# side's mesh, element and load are the ones rebuilt here (P1 on the nodes' Delaunay triangles, or
# on a tensor grid P1 with either diagonal or bilinear cells; the load the source's nodal
# interpolant), so a residual above round-off alone condemns nothing and the weak identity decides.
# A RESIDUAL THAT ONE FACTOR EXPLAINS DOES CONDEMN, at any level and at the first. Measured on a
# coupled round: a side whose load left out its test function read A u = 3.87, 3.97, 3.99 times the
# load of f at its three levels (after that factor 7.6e-2, 3.2e-2, 1.0e-2 of the load were left,
# against 2.8, 3.0, 3.0 before it), and one that added its load inside the trial-function loop 4.00
# times it (1.3e-7 left); both converged cleanly to the wrong field, and the first read VERIFIED at
# level 1. Over 785 levels of
# 329 coupled cells of earlier rounds (their own dumps and configs), the rule below -- the residual
# above 0.1 of the load, the factor off one by more than 0.1, and at most a quarter of the residual
# left after it -- held on none of the 188 sides (542 levels) the three-level identity confirms (the
# largest factor there, 0.76 on a rebuild that does not match its mesh, left 0.91 of its residual);
# it held on three sides whose A u was about zero times the load, two of them already condemned by
# the identity and one it had not judged.
_DISCRETE_ROUNDOFF = 1e-6
_SCALE_RESIDUAL, _SCALE_OFF, _SCALE_LEFT = 0.1, 0.1, 0.25


def _one_factor_condemns(r) -> bool:
    """True when this level's A u is one factor other than one times the load (see above)."""
    import math
    return (math.isfinite(r.residual) and math.isfinite(r.scale) and math.isfinite(r.left)
            and r.residual > _SCALE_RESIDUAL and abs(r.scale - 1.0) > _SCALE_OFF
            and r.left <= _SCALE_LEFT * r.residual)
_GAUSS2 = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))


def _k_matrix(coefficient):
    """A coefficient as a 2x2 matrix: k times the identity for one number."""
    import numpy as np
    if isinstance(coefficient, (int, float)):
        return float(coefficient) * np.eye(2)
    K = np.asarray(coefficient, dtype=float)
    if K.size != 4:
        raise ValueError("the coefficient is neither one number nor a 2x2 matrix")
    return K.reshape(2, 2)


def _p1_system(P, T, K, c):
    """(A, M) of P1 triangles T on nodes P: A carries (K grad u) . grad v + c u v, M the mass."""
    import numpy as np
    import scipy.sparse as sp
    X = P[T]
    e1, e2 = X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    span = max(float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1])), 1e-300)
    ok = np.abs(det) > 1e-14 * span * span                       # a sliver of collinear nodes
    T, e1, e2, det = T[ok], e1[ok], e2[ok], det[ok]
    area = 0.5 * np.abs(det)
    g1 = np.stack([e2[:, 1], -e2[:, 0]], 1) / det[:, None]
    g2 = np.stack([-e1[:, 1], e1[:, 0]], 1) / det[:, None]
    G = np.stack([-(g1 + g2), g1, g2], 1)                        # the three hats' gradients
    Ke = area[:, None, None] * np.einsum("mai,ij,mbj->mab", G, K, G)
    Me = area[:, None, None] * (np.ones((3, 3)) + np.eye(3))[None] / 12.0
    r, q, n = np.repeat(T, 3, axis=1).ravel(), np.tile(T, (1, 3)).ravel(), len(P)
    return (sp.csr_matrix(((Ke + c * Me).ravel(), (r, q)), (n, n)),
            sp.csr_matrix((Me.ravel(), (r, q)), (n, n)))


def _q1_system(xs, ys, K, c):
    """(A, M) of bilinear cells on the tensor grid xs x ys, node (i, j) numbered i*(ny+1)+j; two
    Gauss points a direction, exact for a constant K."""
    import numpy as np
    import scipy.sparse as sp
    nx, ny = len(xs) - 1, len(ys) - 1
    I, J = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij"))
    hx, hy = np.diff(xs)[I], np.diff(ys)[J]
    ids = np.stack([I * (ny + 1) + J, (I + 1) * (ny + 1) + J, I * (ny + 1) + J + 1,
                    (I + 1) * (ny + 1) + J + 1], 1)
    Ke, Me = np.zeros((len(I), 4, 4)), np.zeros((len(I), 4, 4))
    for s in _GAUSS2:
        for t in _GAUSS2:
            a, b = 0.5 * (s + 1.0), 0.5 * (t + 1.0)
            N = np.array([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b])
            dN = np.stack([np.stack([-(1 - b) / hx, -(1 - a) / hy], 1),
                           np.stack([(1 - b) / hx, -a / hy], 1),
                           np.stack([-b / hx, (1 - a) / hy], 1),
                           np.stack([b / hx, a / hy], 1)], 1)
            w = 0.25 * hx * hy
            Ke += w[:, None, None] * np.einsum("mai,ij,mbj->mab", dN, K, dN)
            Me += w[:, None, None] * np.outer(N, N)[None]
    r, q, n = np.repeat(ids, 4, axis=1).ravel(), np.tile(ids, (1, 4)).ravel(), (nx + 1) * (ny + 1)
    return (sp.csr_matrix(((Ke + c * Me).ravel(), (r, q)), (n, n)),
            sp.csr_matrix((Me.ravel(), (r, q)), (n, n)))


def check_levels_discrete(raw_levels: dict, source_expr: str, coefficient, box: list,
                          reaction: float = 0.0) -> ConsistencyResult:
    """Does each level's dumped field solve the discrete system of -div(K grad u) + c u = f
    rebuilt on its own nodes? `raw_levels` maps a level to (x, y, u) rows at the side's mesh
    nodes; `coefficient` is one number or a 2x2 matrix, used as given (inside the box only its
    symmetric part acts). The residual is taken at the nodes strictly inside the box -- free
    whatever the boundary conditions -- relative to the load there; on a tensor grid the
    smallest of the three rebuilds (bilinear cells, triangles cut along either diagonal)
    counts. CONSISTENT when every level is at round-off (see _DISCRETE_ROUNDOFF); INCONSISTENT
    when at some level, the first included, one factor other than one explains the residual
    (see _one_factor_condemns); otherwise NOT_APPLICABLE with the residuals and the factor
    measured, because only an exact rebuild can condemn anything else."""
    import numpy as np
    from scipy.spatial import Delaunay
    K = _k_matrix(coefficient)
    c = float(reaction or 0.0)
    (x0, x1), (y0, y1) = box
    tol = 1e-7 * max(x1 - x0, y1 - y0)
    res = ConsistencyResult()
    hows = set()
    for lvl in sorted(raw_levels):
        a = np.asarray(raw_levels[lvl], dtype=float)
        if a.ndim != 2 or a.shape[0] < 9 or a.shape[1] < 3 or not np.isfinite(a[:, :3]).all():
            res.levels.append(LevelResult(lvl, len(a), float("nan"), "no finite rows of x, y and the field"))
            continue
        g = tensor_grid(a[:, :3])
        if g is not None:
            xs, ys, U = g
            P = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
            u = U.ravel()
            ny = len(ys) - 1
            I, J = (q.ravel() for q in np.meshgrid(np.arange(len(xs) - 1), np.arange(ny), indexing="ij"))
            p00, p10 = I * (ny + 1) + J, (I + 1) * (ny + 1) + J
            p01, p11 = p00 + 1, p10 + 1
            cands = {"bilinear cells": lambda: _q1_system(xs, ys, K, c),
                     "triangles cut along one diagonal": lambda: _p1_system(
                         P, np.concatenate([np.stack([p00, p10, p11], 1), np.stack([p00, p11, p01], 1)]), K, c),
                     "triangles cut along the other diagonal": lambda: _p1_system(
                         P, np.concatenate([np.stack([p00, p10, p01], 1), np.stack([p10, p11, p01], 1)]), K, c)}
        else:
            _, first = np.unique(np.round(a[:, :2], 12), axis=0, return_index=True)
            a = a[np.sort(first)]
            P, u = a[:, :2], a[:, 2]
            cands = {"P1 triangles of its nodes' Delaunay triangulation":
                     lambda: _p1_system(P, Delaunay(P).simplices, K, c)}
        inner = (P[:, 0] > x0 + tol) & (P[:, 0] < x1 - tol) & (P[:, 1] > y0 + tol) & (P[:, 1] < y1 - tol)
        if inner.sum() < 4:
            res.levels.append(LevelResult(lvl, len(P), float("nan"), "fewer than four nodes inside the box"))
            continue
        f = np.asarray(_eval_source(source_expr, P.tolist(), 2), dtype=float)
        best = None
        for how, build in cands.items():
            try:
                A, M = build()
            except Exception:                                   # noqa: BLE001 -- not rebuildable
                continue
            b = M @ f
            scale = float(np.abs(b[inner]).max())
            if not scale > 0:
                continue
            Au, bi = (A @ u)[inner], b[inner]
            rel = float(np.abs(Au - bi).max()) / scale
            if best is None or rel < best[0]:
                s = float(Au @ bi) / float(bi @ bi)             # A u as a multiple of the load
                best = (rel, how, s, float(np.abs(Au - s * bi).max()) / scale)
        if best is None:
            res.levels.append(LevelResult(lvl, len(P), float("nan"),
                                          "no rebuild with a nonzero load inside the box"))
            continue
        hows.add(best[1])
        res.levels.append(LevelResult(lvl, len(P), best[0], best[1], umax=float(np.abs(u).max()),
                                      scale=best[2], left=best[3]))
    good = [r for r in res.levels if math.isfinite(r.residual)]
    seq = ", ".join(f"{r.residual:.1e}" for r in good)
    how = " / ".join(sorted(hows))
    factor = [r for r in good if _one_factor_condemns(r)]
    if good and all(r.residual <= _DISCRETE_ROUNDOFF for r in good):
        res.verdict = "CONSISTENT"
        res.explanation = (
            f"rebuilt on its own nodes ({how}) with the full coefficient and the source at the "
            f"nodes, the discrete system of this equation leaves round-off at every node inside "
            f"the box, at every level ({seq} of the load): the field is that system's solution. "
            f"That says nothing about the values it holds on its boundary and interface, which "
            f"the interface and outer-boundary checks judge.")
    elif factor:
        res.verdict = "INCONSISTENT"
        _lv = ", ".join(str(r.level) for r in factor)
        res.explanation = (
            f"rebuilt on its own nodes ({how}) with the full coefficient and the source at the "
            f"nodes, the field's A u at the nodes inside the box is "
            f"{', '.join(f'{r.scale:.3g}' for r in factor)} times the load of f at level"
            f"{'s' if len(factor) > 1 else ''} {_lv}, and after that factor "
            f"{', '.join(f'{r.left:.1e}' for r in factor)} of the load is left (against "
            f"{', '.join(f'{r.residual:.1e}' for r in factor)} before it): "
            + ("the field solves this equation with no load: none of the load of f is in it."
               if all(abs(r.scale) < _SCALE_OFF for r in factor) else
               "the field solves this equation with its load scaled by that factor, so its load or its "
               "stiffness is off by it."
               + (" A load summed without its test function is about the dofs of one cell times the "
                  "right one." if any(r.scale > 1.5 for r in factor) else "")))
    else:
        res.verdict = "NOT_APPLICABLE"
        res.explanation = (
            (f"rebuilt on its own nodes ({how}) with the full coefficient and the source at the "
             f"nodes, the discrete system of this equation leaves {seq} of the load at the nodes "
             f"inside the box, level by level, and A u there is "
             f"{', '.join(f'{r.scale:.3g}' for r in good)} times the load with "
             f"{', '.join(f'{r.left:.1e}' for r in good)} of it left after that factor. Round-off "
             f"would confirm that the field solves this system, and a residual that one factor "
             f"other than one explains would condemn it; this one is neither, so this check does not "
             f"judge it." if good else
             "no level could be rebuilt: " + "; ".join(r.detail for r in res.levels if r.detail)[:300]))
    return res


def _decide(res: ConsistencyResult) -> ConsistencyResult:
    """Turn per-level residuals into a verdict.

    Shared by the scalar and the elasticity path so one operator cannot drift
    into a different rule than the other: the verdict reads the FALL, and the
    round-off, meaningless-magnitude and too-few-levels branches below are the
    ones both operators need.
    """
    import math
    # A RESIDUAL OF 1e+297 IS NOT A VERDICT. Measured on real runs
    # outside this check's operator, the relative residual came back as
    # 1.197e+297 and 5.190e+293 — the ratio of two quantities that have nothing
    # to do with each other. The magnitudes were reported with a confident
    # INCONSISTENT and CONSISTENT respectively. A number that large means the
    # comparison is meaningless, not that the field is very wrong.
    for r in res.levels:
        if math.isfinite(r.residual) and abs(r.residual) > 1e6:
            r.detail = (r.detail + " | REFUSED: relative residual "
                        f"{r.residual:.3e} is far outside anything a "
                        "discretisation produces, so the two sides of the "
                        "identity are not comparable — check that the source "
                        "term, the coefficient and the field are the ones the "
                        "task states, and that this is a scalar "
                        "-div(K grad u) = f problem at all")
            r.residual = float("nan")
    good = [r for r in res.levels if math.isfinite(r.residual)]
    if len(good) < 2:
        res.verdict = "NOT_APPLICABLE"
        res.explanation = (
            "fewer than two levels could be checked. This test needs the "
            "prescribed probe points, which form a uniform tensor grid of cell "
            "midpoints; on a scatter the midpoint quadrature has no accuracy "
            "and no number is reported rather than a misleading one."
            + (" " + good[0].detail if good else "")
            + " " + "; ".join(r.detail for r in res.levels if r.detail)[:300])
        return res
    first, last = good[0].residual, good[-1].residual
    if last > 0 and first > 0 and len(good) >= 2:
        span = len(good) - 1
        res.rate = math.log(first / last) / (span * math.log(2.0)) if last else None
    # A RESIDUAL AT ROUND-OFF IS THE IDENTITY HOLDING, NOT A FLAT FAILURE.
    #
    # The decay test asks whether the residual FALLS, which is the right
    # question for a discretisation whose error shrinks with h. It has no
    # answer when the residual is already zero: a field that satisfies the weak
    # identity exactly gives 0.000e+00 at every level, `last < first/3` is
    # false, and the verdict came out INCONSISTENT with the explanation "the
    # weak residual is FLAT: 0.000e+00 -> 0.000e+00" — condemning the one
    # field that could not be more right. Found by writing the test that asks
    # whether a genuine scalar-diffusion case still passes.
    if max(first, last) <= 1e-12:
        res.verdict = "CONSISTENT"
        res.explanation = (
            f"the weak residual is at round-off ({first:.3e} -> {last:.3e}): "
            f"the identity holds as exactly as double precision allows, so "
            f"your field satisfies the equation you were given. Note that this "
            f"is what an ANALYTIC or interpolated exact field also gives — it "
            f"says the operator and source match, not that a solver ran.")
    elif last > 1.0:
        # A FALL FROM 64 TO 4.7 IS NOT A FIELD SATISFYING ITS EQUATION. The
        # decay test alone read a uniform field of 7.5e12 (a scalar-transport
        # deck with no Dirichlet condition) as CONSISTENT because its relative
        # residual fell 64 -> 34 -> 4.7. A residual above the field's own
        # scale at the finest level means the field is not a solution at any
        # level, whatever the trend.
        res.verdict = "INCONSISTENT"
        res.explanation = (
            f"the weak residual is still {last:.3e} at the finest level -- "
            f"larger than the field's own scale -- after {first:.3e} -> "
            f"{last:.3e}. A field that solves the stated equation has a "
            f"residual well below one there however coarse the mesh; a number "
            f"above one at every level is a field that is not a solution, and "
            f"the fall does not change that. Look first at the field itself (a "
            f"uniform or astronomical column is no solution), then at the "
            f"source term, then the coefficient, then an element-local "
            f"assembly defect (quadrature, a wrong map).")
    elif len(good) < 3:
        # TWO LEVELS CANNOT TELL A FALL FROM A TURN. The old rule judged two
        # levels by one ratio, and both directions were measured wrong: a
        # correct coupled field judged on its last two levels alone (5.18e-03 ->
        # 1.87e-03, the numbers every correct run of its problem reads there)
        # was called wrong, and the same kind of field made one percent wrong
        # fell TENFOLD between its first two levels before rising at the third.
        res.verdict = "NOT_APPLICABLE"
        res.explanation = (
            f"only two levels could be checked ({first:.3e} -> {last:.3e}). Two "
            f"levels cannot tell a residual that keeps falling from one that "
            f"turns: a field one percent off its equation was measured falling "
            f"tenfold between its first two levels and rising at the third. The "
            f"check needs a third level.")
    elif (lambda z: len(z) >= 2 and max(z) > 2.0 * min(z))([r.umax for r in good if r.umax > 0]):
        # A FIELD THAT CHANGES SIZE BETWEEN LEVELS IS A DIFFERENT PROBLEM AT EACH.
        # Measured on a coupled run: a Neumann side fed a partner flux that grew
        # about fourfold per level read a flat residual and was called wrong; held
        # at one level's data, its residual fell at every step. This side cannot be
        # judged until its data settles -- which is not a clean bill either.
        # READ BEFORE THE FALL, NOT AFTER IT: placed after the branch below, this was
        # reached only when the residual did not fall. Measured: the solution scaled
        # by 3.0, 1.5 and 1.1 on three levels -- right at no level -- fell 1.94 ->
        # 0.49 -> 0.099 and was called CONSISTENT.
        sizes = " -> ".join(f"{r.umax:.3g}" for r in good if r.umax > 0)
        seq = " -> ".join(f"{r.residual:.3e}" for r in good)
        falls = all(b.residual < 0.8 * a.residual for a, b in zip(good, good[1:]))
        res.verdict = "UNSETTLED"
        res.explanation = (
            f"the field itself changes size across the levels (largest value "
            f"{sizes}) and the weak residual "
            + (f"falls ({seq}), but between fields of different size a fall is no "
               f"refinement study of one problem. "
               if falls else f"does not fall ({seq}). ")
            + "A side handed different data at each level solves a different "
              "problem at each, so this check cannot say whether it solves its "
              "equation: judge the data it imports first -- a partner whose exports "
              "grow or shrink like that is the likelier defect -- then this side.")
    elif all(b.residual < 0.8 * a.residual for a, b in zip(good, good[1:])):
        # EVERY STEP FALLS, not only the first against the last. Measured on 73
        # correct coupled sides (their own mesh dumps, three levels): every step
        # fell by a fifth or more. The same fields scaled by 1.01 -- a field
        # solving a source one percent off -- level off at the size of the
        # difference or rise toward it, and every one of them fails this, where
        # the old first-against-last rule (last < first/3) passed 5 % of them and
        # called one correct side wrong (2.21e-03 -> 1.31e-03 -> 7.79e-04).
        # THE OTHER DIRECTION IS WEAKER, measured 2026-09-25 over 184 recorded sides
        # on tensor grids: scaled by 0.99 about half still pass (99 of 184), where the
        # discretisation residual's sign adds to the defect and three levels cannot
        # separate the two. A pass here is weaker evidence than a failure.
        res.verdict = "CONSISTENT"
        seq = " -> ".join(f"{r.residual:.3e}" for r in good)
        res.explanation = (
            f"the weak residual falls at every refinement ({seq}; rate "
            f"{res.rate:.2f} per refinement overall). The field solves the "
            f"equation you state inside its subdomain. That says nothing about "
            f"the values it holds on its boundary and interface -- a field "
            f"solving the right equation with the wrong boundary data passes "
            f"here -- which the interface and outer-boundary checks judge.")
    else:
        res.verdict = "INCONSISTENT"
        seq = " -> ".join(f"{r.residual:.3e}" for r in good)
        res.explanation = (
            f"the weak residual does not fall at every refinement: {seq}. A "
            f"field that solves the equation you state shows a residual that "
            f"falls with each refinement, about fourfold per halving of h for a "
            f"P1 field; a field solving a different equation levels off at the "
            f"size of the difference or rises toward it. Look at the source term "
            f"first -- a sign, a missing term, or an "
            f"expression evaluated in element-local instead of global "
            f"coordinates -- then the coefficient. This check does not see the "
            f"boundary conditions.")
    return res


def _adjoint_elastic_flat(lam: float, mu: float, pts, box, direction: int):
    """v and L*v for LINEAR ELASTICITY, v with value and slope zero on the box.

    For constant lambda and mu the elasticity operator is self-adjoint, exactly
    as constant-K diffusion is, so the same weak identity holds componentwise:

        integral u . (L* v)  =  integral f . v ,
        L* v = -div(sigma(v)) = -[ mu lap(v) + (mu + lambda) grad(div v) ]

    and with a v whose value AND normal slope vanish on every face, both
    boundary terms go for any u -- including a coupled side whose interface
    carries the partner's displacement.

    `direction` picks v = (W, 0) or (0, W) with W = prod sin^2, so the two
    calls together test both momentum equations rather than a single mixture
    that a sign error in one component could hide.

    WHY THIS EXISTS. The scalar check REFUSES elasticity, deliberately: handed
    an elasticity result set it once reported the field converging to the wrong
    solution, which was meaningless. That refusal covered the coupled
    elasticity problems, whose equation is exactly this one -- so the one
    coupled family with a recent correct run was the one nothing could check.

    Calibrated on a manufactured case deliberately not zero on the boundary
    (u = (sin(pi x) sin(pi y) + x, (sin(pi x) sin(pi y))/2 + y/2), f computed
    from it symbolically, lambda and mu of order 1e3), levels 16 to 128, worst of
    the two directions:

        field                      residual                     fall
        exact                      5.578e-03 -> 8.611e-05       64.8x
        3 % low on ux              3.501e-02 -> 3.008e-02        1.2x
        3 % low on uy              3.541e-02 -> 3.008e-02        1.2x
        half amplitude             5.028e-01 -> 5.000e-01        1.0x
        uy dropped entirely        1.000e+00 -> 1.000e+00        1.0x

    The exact field falls at the documented factor of four per refinement and
    every wrong one stays flat, including a 3 % amplitude error in a single
    component.
    """
    import numpy as np

    if len(box) != 2:
        raise ValueError("the elasticity identity here is written for 2D")
    xs = [np.asarray([p[d] for p in pts], dtype=float) for d in range(2)]
    Ls = [float(hi - lo) for lo, hi in box]
    a, b = math.pi / Ls[0], math.pi / Ls[1]
    ax = a * (xs[0] - box[0][0])
    by = b * (xs[1] - box[1][0])
    S, T = np.sin(ax) ** 2, np.sin(by) ** 2
    Sp, Tp = a * np.sin(2 * ax), b * np.sin(2 * by)
    Spp, Tpp = 2 * a * a * np.cos(2 * ax), 2 * b * b * np.cos(2 * by)

    W = S * T
    lap = Spp * T + S * Tpp
    Wxx, Wxy, Wyy = Spp * T, Sp * Tp, S * Tpp

    if direction == 0:                       # v = (W, 0)
        Lvx = -(mu * lap + (mu + lam) * Wxx)
        Lvy = -((mu + lam) * Wxy)
        return (W, np.zeros_like(W)), (Lvx, Lvy)
    Lvx = -((mu + lam) * Wxy)                # v = (0, W)
    Lvy = -(mu * lap + (mu + lam) * Wyy)
    return (np.zeros_like(W), W), (Lvx, Lvy)


def _flat_test_function(pts, box):
    """W = prod sin^2 on the box and its two first derivatives.

    Value AND normal slope vanish on every face, so both boundary terms of the
    weak identity go for any field -- the same v `_adjoint_elastic_flat` builds,
    returned with its gradient because the thermal term below needs div v.
    """
    import numpy as np
    xs = [np.asarray([p[d] for p in pts], dtype=float) for d in range(2)]
    Ls = [float(hi - lo) for lo, hi in box]
    a, b = math.pi / Ls[0], math.pi / Ls[1]
    ax = a * (xs[0] - box[0][0])
    by = b * (xs[1] - box[1][0])
    S, T = np.sin(ax) ** 2, np.sin(by) ** 2
    Sp, Tp = a * np.sin(2 * ax), b * np.sin(2 * by)
    return S * T, Sp * T, S * Tp


def check_levels_thermoelastic(levels: dict, source_x: str, source_y: str,
                               lam: float, mu: float, beta: float,
                               box: list) -> ConsistencyResult:
    """Does a DISPLACEMENT field satisfy -div(sigma_mech(u)) = f - beta grad T?

    `levels` maps a level number to rows of (x, y, T, ux, uy): the delivered
    temperature rides along because the momentum equation of a thermo-elastic
    problem carries it. With sigma = 2 mu eps(u) + lambda tr(eps) I - beta T I,
    equilibrium div(sigma) + f = 0 reads -div(sigma_mech(u)) = f - beta grad T,
    and against a v whose value and slope vanish on every face

        integral u . (L* v)  =  integral f . v  +  beta * integral T div v ,

    the thermal term integrated by parts once (v = 0 on the boundary). Both
    momentum directions are tested and the WORST decides the level, as in the
    isothermal check; beta = 0 reproduces it bit for bit.

    WHY THIS EXISTS. The temperature of a thermo-elastic side was judged
    against its own heat equation while the displacement of the same side
    was judged against nothing: a run whose config stated NO body force
    (source_ux = source_uy = 0) solved a different problem, its displacement
    came out 400 times too small on both sides and converged cleanly, its
    temperature was right, and every self-consistency check passed. This is
    the check that separates a displacement solving the STATED momentum
    equation from one solving another -- a dropped body force in the deck, a
    missing thermal term, a wrong lambda or mu -- from the agent's own files
    and its own config, with no reference.
    """
    import numpy as np

    res = ConsistencyResult()
    for lvl in sorted(levels):
        rows = levels[lvl]
        if not rows:
            res.levels.append(LevelResult(lvl, 0, float("nan"), "no rows"))
            continue
        pts = [r[:2] for r in rows]
        Tf = np.asarray([r[2] for r in rows], dtype=float)
        ux = np.asarray([r[3] for r in rows], dtype=float)
        uy = np.asarray([r[4] for r in rows], dtype=float)
        if not (np.all(np.isfinite(ux)) and np.all(np.isfinite(uy)) and np.all(np.isfinite(Tf))):
            res.levels.append(LevelResult(
                lvl, len(rows), float("nan"),
                "the delivered field carries a non-finite value"))
            continue
        weight, why = _detect_midpoint_grid(pts, box)
        if weight is None:
            res.levels.append(LevelResult(lvl, len(rows), float("nan"), why))
            continue
        fx = _eval_source(source_x, pts, 2)
        fy = _eval_source(source_y, pts, 2)
        W, Wx, Wy = _flat_test_function(pts, box)
        sides = []
        for direction, f_here, dW in ((0, fx, Wx), (1, fy, Wy)):
            (_vx, _vy), (Lvx, Lvy) = _adjoint_elastic_flat(lam, mu, pts, box,
                                                           direction)
            lhs = float(np.sum(ux * Lvx + uy * Lvy) * weight)
            rhs = float((np.sum(f_here * W) + float(beta) * np.sum(Tf * dW))
                        * weight)
            sides.append((direction, lhs, rhs))
        # A DIRECTION WHOSE TEST INTEGRAL VANISHES CARRIES NO INFORMATION. A
        # field symmetric about the box's mid-line makes one direction's lhs
        # and rhs both round-off (measured: 1.2e-15 against 0.0), and their
        # ratio read as a 100 % residual -- a right field judged INCONSISTENT.
        # Such a direction is skipped; the other one judges the level.
        big = max((max(abs(l), abs(r)) for _, l, r in sides), default=0.0)
        worst, detail = -1.0, ""
        for direction, lhs, rhs in sides:
            if max(abs(lhs), abs(rhs)) <= 1e-9 * big:
                continue
            # RELATIVE TO THE LARGER SIDE, not to the stated load alone: a
            # config that states no body force under a loaded field makes the
            # stated side tiny, and a ratio against it was refused as "not
            # comparable" instead of read as the order-one residual it is.
            rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
            if rel > worst:
                worst, detail = rel, (f"worst component is the "
                                      f"{'x' if direction == 0 else 'y'} "
                                      f"momentum equation: "
                                      f"lhs={lhs:.6e} rhs={rhs:.6e}")
        if worst < 0:
            res.levels.append(LevelResult(lvl, len(rows), float("nan"),
                                          "both momentum test integrals vanish: "
                                          "no information at this level"))
            continue
        res.levels.append(LevelResult(lvl, len(rows), worst, detail))
    return _decide(res)


def check_levels_elastic(levels: dict, source_x: str, source_y: str,
                         lam: float, mu: float, box: list) -> ConsistencyResult:
    """Does a DISPLACEMENT field satisfy -div(sigma(u)) = f on this box?

    `levels` maps a level number to rows of (x, y, ux, uy). Both momentum
    equations are tested and the WORST of the two decides the level, so a
    component that is right cannot cover for one that is not.
    """
    import numpy as np

    res = ConsistencyResult()
    for lvl in sorted(levels):
        rows = levels[lvl]
        if not rows:
            res.levels.append(LevelResult(lvl, 0, float("nan"), "no rows"))
            continue
        pts = [r[:2] for r in rows]
        ux = np.asarray([r[2] for r in rows], dtype=float)
        uy = np.asarray([r[3] for r in rows], dtype=float)
        if not (np.all(np.isfinite(ux)) and np.all(np.isfinite(uy))):
            res.levels.append(LevelResult(
                lvl, len(rows), float("nan"),
                "the delivered field carries a non-finite value"))
            continue
        weight, why = _detect_midpoint_grid(pts, box)
        if weight is None:
            res.levels.append(LevelResult(lvl, len(rows), float("nan"), why))
            continue
        fx = _eval_source(source_x, pts, 2)
        fy = _eval_source(source_y, pts, 2)
        worst, detail = -1.0, ""
        for direction, f_here in ((0, fx), (1, fy)):
            (vx, vy), (Lvx, Lvy) = _adjoint_elastic_flat(lam, mu, pts, box,
                                                         direction)
            lhs = float(np.sum(ux * Lvx + uy * Lvy) * weight)
            rhs = float(np.sum(f_here * (vx if direction == 0 else vy))
                        * weight)
            rel = abs(lhs - rhs) / max(abs(rhs), 1e-300)
            if rel > worst:
                worst, detail = rel, (f"worst component is the "
                                      f"{'x' if direction == 0 else 'y'} "
                                      f"momentum equation: "
                                      f"lhs={lhs:.6e} rhs={rhs:.6e}")
        res.levels.append(LevelResult(lvl, len(rows), worst, detail))
    return _decide(res)
