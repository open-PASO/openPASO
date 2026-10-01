"""Kratos Multiphysics participant for the openPASO `couple` driver (NEUMANN side).

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1, so an
iteration-1 fallback is mandatory), writes exports.json LAST and exits 0.
Needs KratosMultiphysics + ConvectionDiffusionApplication importable in the
interpreter named in `command`: take it from `discover(query='list')`, never a
guessed system python3 (it exists, so nothing looks wrong, and raises
ModuleNotFoundError because the Kratos build targets another Python version).

THE OTHER HALF OF participant_kratos.py. That file is the DIRICHLET side: it
imports the partner's `values` (interface temperature), fixes them as nodal
TEMPERATURE, and exports the consistent REACTION_FLUX. This file is the NEUMANN
side: it imports the partner's `normal_fluxes` and applies them as the natural
boundary condition, then exports the interface TEMPERATURE its solve produced,
which is what a Dirichlet partner consumes.

Physics: steady conduction  -div(K grad T) = f  on one rectangular subdomain of
a domain split by a straight interface at x = IFACE_X. The non-interface
x-boundary carries a Dirichlet value T_OUTER; whether the top and bottom edges
are held too is FULL_OUTER_DIRICHLET's, from your problem.

  x = OUTER_X : Dirichlet T = T_OUTER
  x = IFACE_X : INTERFACE, natural BC = the partner's imported flux
  top/bottom  : held at T_OUTER or natural -- FULL_OUTER_DIRICHLET, from your problem

THE SIGN, WHICH IS THE ONLY THING A NEUMANN PARTICIPANT CAN GET SILENTLY WRONG.
Every participant exports its outward normal flux density with respect to ITS
OWN outward normal,
      q = -(K grad T) . n_own,
so on a shared interface the two sides carry OPPOSITE signs. Integrating the
weak form on THIS subdomain,
      int_Omega K grad T . grad v  =  int_Omega f v  +  int_dOmega (K grad T . n) v ds,
and on the interface n = n_own = -n_partner, so
      K grad T . n_own = -q_own = +q_partner.
The partner's number is therefore applied UNCHANGED — no minus sign anywhere.
Kratos's FluxCondition2D2N enforces exactly  K grad T . n = FACE_HEAT_FLUX
(verified by a patch test: T fixed to 0 at x=0, FACE_HEAT_FLUX=1 on x=1, K=1,
no source -> T(1)=+1.000000, i.e. dT/dx=+1 not -1), so

      FACE_HEAT_FLUX  :=  the imported `normal_fluxes`, verbatim.

If you ever flip that sign to "make the temperatures look right", you have
built a coupling that drives heat the wrong way across the interface and still
converges. The self-check printed at the end of this script is the guard: it
evaluates the discrete divergence theorem on this subdomain and must come out
at round-off.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER   = "left"        # the partner's `name` in your couple(...) call
X0, X1    = 0.6, 1.0      # this subdomain's x-extent
Y0, Y1    = 0.0, 0.4      # this subdomain's y-extent
IFACE_AXIS = "x"          # the line x = IFACE_X; this contract has no other
IFACE_X   = 0.6           # the shared interface: X0 or X1
K         = 1.6           # conductivity of THIS subdomain


def F_SRC(x, y):
    """Volumetric source f in  -div(K grad T) = f, as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above.
    A CONSTANT CANNOT REPRESENT A POLYNOMIAL SOURCE: if your problem states
    one, or you derived it from a manufactured solution, a single number here
    silently solves a different problem. With the whole outer boundary
    prescribed and no source, the answer degenerates to the profile between
    the outer values.

    Unlike its FEniCSx and DUNE siblings, this one is called ONCE PER NODE
    with SCALAR coordinates, so write it with plain math or NumPy scalars —
    do not assume arrays:

        return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    """
    return 0.0 * x

T_OUTER   = 300.0         # Dirichlet value on the NON-interface x-boundary
# WHICH NON-INTERFACE EDGES ARE HELD IS YOUR PROBLEM'S TO SAY, NOT THIS FILE'S:
# a default here once chose a boundary condition for problems it never saw, and
# a field obeying the wrong condition converges cleanly with nothing to show it.
# True if y = Y0 and y = Y1 are held at T_OUTER, False if they are natural (zero
# flux); the script refuses to run until you set it.
FULL_OUTER_DIRICHLET = None  # <-- True or False, FROM YOUR PROBLEM STATEMENT
if FULL_OUTER_DIRICHLET is None:
    sys.exit("FULL_OUTER_DIRICHLET is unset: say whether the non-interface y-edges "
             "are held at T_OUTER (True) or natural (False), from your problem "
             "statement. A default here would be choosing your boundary "
             "condition for you.")
NX, NY    = 20, 16        # this subdomain's OWN mesh; need not match the partner
Q_INIT    = 0.0           # iteration-1 fallback interface flux density


def source(x, y):
    """Volumetric source at the nodes — the interpolation point for F_SRC.

    Sampling at the nodes is the P1 interpolant of the source, an O(h^2) load
    error, the same order as the discretization error, so it does not touch
    the second-order rate.
    """
    return F_SRC(x, y)
# ─────────────────────────────────────────────────────────────────────────

# ── THE PER-LEVEL RULE (served). ./config.json names the level and overrides the
#    mesh, the box, K, F_SRC and T_OUTER; the dumps below carry the level. Write:
#      {"level": k, "nx": .., "ny": .., "x0": .., "x1": .., "y0": .., "y1": ..,
#       "iface": "left|right", "k": <diffusivity>,
#       "source_expr": "<f(x, y), or 0.0>", "outer": <the non-interface value, if any>}
#    openPASO judges this side's field against them; without them it abstains.
LEVEL, _REACTION = 1, 0.0
if Path("config.json").is_file() or os.environ.get("OPENPASO_CONFIG_JSON"):
    try:
        _cfg = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
        _cfg.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))   # a multi-level call's level keys
        LEVEL = int(_cfg.get("level", LEVEL))
        NX = int(_cfg.get("nx", NX))
        NY = int(_cfg.get("ny", NY))
        X0 = float(_cfg.get("x0", X0)); X1 = float(_cfg.get("x1", X1))
        Y0 = float(_cfg.get("y0", Y0)); Y1 = float(_cfg.get("y1", Y1))
        K = float(_cfg.get("k", K)); T_OUTER = float(_cfg.get("outer", T_OUTER))
        if _cfg.get("source_expr") is not None:
            _src = compile(str(_cfg["source_expr"]).replace("^", "**"), "<source_expr>", "eval")
            F_SRC = lambda x, y, _c=_src: eval(_c, {"__builtins__": {}}, dict(  # noqa: E731
                x=x, y=y, pi=np.pi, sin=np.sin, cos=np.cos, exp=np.exp, sqrt=np.sqrt)) + 0.0 * x
        _ifc = str(_cfg.get("iface", "")).lower()
        IFACE_X = {"left": X0, "right": X1}.get(_ifc, IFACE_X)
        IFACE_AXIS = "y" if _ifc in ("bottom", "top") else IFACE_AXIS
        _REACTION = float(_cfg.get("reaction") or 0.0)
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

AX = 0 if IFACE_AXIS == "x" else 1         # the coordinate the interface FIXES
AL = 1 - AX                                # the coordinate that RUNS ALONG it
LO, HI = (X0, X1) if AX == 0 else (Y0, Y1)         # this subdomain, across the interface
ALO, AHI = (Y0, Y1) if AX == 0 else (X0, X1)       # this subdomain, along it
ON_MAX_X = abs(IFACE_X - HI) < abs(IFACE_X - LO)   # interface at this side's MAX of that axis?
OUTER_X = LO if ON_MAX_X else HI
S = 1.0 if ON_MAX_X else -1.0          # outward normal at the interface = S * e_x
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)



def _partner_block(imp):
    """The partner's block from imports.json, by the name the driver actually used."""
    if not isinstance(imp, dict) or not imp:
        return None
    if PARTNER in imp:
        return imp[PARTNER] or None
    others = [k for k in imp if isinstance(imp.get(k), dict)]
    if len(others) == 1:     # named differently in couple(...) than here: read it, say so
        import sys as _sys
        print(f"NOTE: PARTNER is {PARTNER!r} but imports.json is keyed {others[0]!r} "
              f"-- reading that block; align PARTNER with the name in your "
              f"couple(...) call.", file=_sys.stderr)
        return imp[others[0]] or None
    if others:
        raise SystemExit(f"imports.json holds blocks named {others} and none is "
                         f"PARTNER={PARTNER!r}: set PARTNER to the partner's name "
                         f"in your couple(...) call.")
    return None


def read_imports():
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1,
    so the caller must fall back to an initial guess."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return _partner_block(json.loads(p.read_text() or "{}"))
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, y):
    """Map the partner's samples onto THIS participant's interface points.
    The driver does no interpolation — non-matching meshes are handled here."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(y), float(fallback))
    ys = np.array([c[AL] for c in imp["coordinates"]], float)   # the coordinate ALONG the interface
    # `or []` and not `.get(key, [])`: a partner that writes the key with an
    # explicit null gets [] here instead of a TypeError out of np.asarray, and
    # falls through to the fallback like any other unusable import.
    vs = np.asarray(imp.get(key) or [], float).ravel()
    if vs.size != ys.size:
        return np.full(len(y), float(fallback))
    o = np.argsort(ys)
    return np.interp(y, ys[o], vs[o])


def build_model():
    """Structured triangulation of [X0,X1] x [Y0,Y1]; returns (mp, nid)."""
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    model = KM.Model()
    mp = model.CreateModelPart("thermal")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    settings = KM.ConvectionDiffusionSettings()
    settings.SetUnknownVariable(KM.TEMPERATURE)
    settings.SetDiffusionVariable(KM.CONDUCTIVITY)
    settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
    # FluxCondition2D2N reads its load from the SURFACE source variable named
    # here; leave it FACE_HEAT_FLUX or the condition assembles nothing and the
    # interface silently becomes insulated.
    settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)
    for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX,
              KM.REACTION_FLUX):
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(1)

    props = mp.CreateNewProperties(1)
    nid, cnt = {}, 1
    for j in range(NY + 1):
        for i in range(NX + 1):
            mp.CreateNewNode(cnt, X0 + (X1 - X0) * i / NX,
                             Y0 + (Y1 - Y0) * j / NY, 0.0)
            nid[(i, j)] = cnt
            cnt += 1
    eid = 1
    for j in range(NY):
        for i in range(NX):
            a, b, c, d = nid[(i, j)], nid[(i+1, j)], nid[(i+1, j+1)], nid[(i, j+1)]
            mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props)
            eid += 1
            mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props)
            eid += 1
    return mp, nid
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


def main():
    """Solve once and write exports.json. Returns (model_part, node_index_map)
    so a verification script can `import` this file, call main(), and integrate
    the volume field against a manufactured solution — the coupling itself only
    ever needs the file handshake."""
    if IFACE_AXIS != "x" or _REACTION or min(abs(IFACE_X - X0), abs(IFACE_X - X1)) > TOL:
        sys.exit(f"this side solves -div(K grad T) = f (no reaction) across x = IFACE_X, X0 or X1; "
                 f"got reaction {_REACTION:g}, axis {IFACE_AXIS}, IFACE_X {IFACE_X} on [{X0},{X1}]")
    # NX < 1 would put the interface column and the outer Dirichlet column on
    # the SAME nodes; the Dirichlet condition wins, the imported flux is
    # discarded, and the run still exits 0 with a plausible-looking export.
    if NX < 1 or NY < 1:
        sys.exit(f"NX,NY = {NX},{NY}: need at least one element in each "
                 "direction, and NX >= 1 so the interface and the outer "
                 "Dirichlet boundary do not land on the same nodes")

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    mp, nid = build_model()
    i_if = NX if ON_MAX_X else 0            # column index of the interface
    i_out = 0 if ON_MAX_X else NX           # column index of x = OUTER_X
    # The two column indices above are the ONLY place the geometry meets the
    # node numbering. If an edit to build_model ever reorders the grid they
    # would silently point at interior columns, the interface flux would be
    # applied inside the domain and the run would still exit 0. Check them
    # against the coordinates they are supposed to name.
    for idx, want, what in ((i_if, IFACE_X, "interface"),
                            (i_out, OUTER_X, "outer Dirichlet")):
        got = mp.Nodes[nid[(idx, 0)]].X
        if abs(got - want) > TOL:
            sys.exit(f"internal: the {what} column sits at x={got}, not "
                     f"x={want} — the mesh and the column indices disagree")
    y_if = np.array([Y0 + (Y1 - Y0) * j / NY for j in range(NY + 1)])
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    q_in = sample(read_imports(), "normal_fluxes", Q_INIT, y_if)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    for n in mp.Nodes:
        n.SetSolutionStepValue(KM.CONDUCTIVITY, K)
        n.SetSolutionStepValue(KM.HEAT_FLUX, float(source(n.X, n.Y)))
        n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)

    for j in range(NY + 1):                 # outer x Dirichlet boundary
        n = mp.Nodes[nid[(i_out, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, float(T_OUTER))
        n.Fix(KM.TEMPERATURE)
    if FULL_OUTER_DIRICHLET:
        # The interface corners lie on y=Y0,Y1, so they remain outer-boundary
        # Dirichlet nodes even though adjacent interface facets carry flux.
        for i in range(NX + 1):
            for j in (0, NY):
                n = mp.Nodes[nid[(i, j)]]
                n.SetSolutionStepValue(KM.TEMPERATURE, float(T_OUTER))
                n.Fix(KM.TEMPERATURE)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    # ── the interface: the partner's flux, applied UNCHANGED (see the header) ──
    for j in range(NY + 1):
        # THE NODAL VALUE IS THE LOAD. FluxCondition2D2N integrates FACE_HEAT_FLUX
        # from its NODES; a condition created without this SetSolutionStepValue
        # assembles zero, exits 0 and is reported unresponsive (measured).
        mp.Nodes[nid[(i_if, j)]].SetSolutionStepValue(
            KM.FACE_HEAT_FLUX, float(q_in[j]))
    props = mp.GetProperties()[1]
    for j in range(NY):
        mp.CreateNewCondition("FluxCondition2D2N", j + 1,
                              [nid[(i_if, j)], nid[(i_if, j + 1)]], props)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    # AddDof with a REACTION variable: without the second argument the fixed
    # dofs have nowhere to store their reaction and it is silently discarded.
    # This side does not export the reaction, but the conservation self-check
    # below is built from it.
    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
    # arg 4 is CalculateReactionsFlag: True, or the self-check has no reactions.
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder,
                                              True, False, False, False)
    strategy.Initialize()
    strategy.Solve()
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
    # TWO THINGS YOUR SOLVE ABOVE MUST DO, or the self-check below reads zeros:
    #   * AddDof(TEMPERATURE, REACTION_FLUX, mp) -- the SECOND argument gives
    #     every fixed dof a place to store its reaction; without it the
    #     reaction is silently discarded.
    #   * run the strategy with CalculateReactionsFlag=True (the 4th positional
    #     argument of ResidualBasedLinearStrategy); the conservation self-check
    #     is built from REACTION_FLUX.

    # ── HELD EDGES, END TO END (served) ─ keep. Measured: fills that skipped the
    #    interface column on the held edges left its two end nodes free.
    _loose = [_n for _n in mp.Nodes if not _n.IsFixed(KM.TEMPERATURE) and (
        abs((_n.X, _n.Y)[AX] - OUTER_X) < TOL or (bool(FULL_OUTER_DIRICHLET) and (
            abs((_n.X, _n.Y)[AL] - ALO) < TOL or abs((_n.X, _n.Y)[AL] - AHI) < TOL)))]
    if _loose:
        sys.exit(f"OUTER BOUNDARY: {len(_loose)} node(s) on the edges this side holds are free "
                 f"(the first at ({_loose[0].X:g}, {_loose[0].Y:g})); every node of them, "
                 f"{'the interface end nodes included, ' if FULL_OUTER_DIRICHLET else ''}"
                 f"must be fixed before the solve. Nothing was exported.")

    # ── served: the trace is read from the nodes AT x = IFACE_X, checked by
    #    coordinate before anything is exported. Measured: two of three worker
    #    fills exported the OUTER column's temperature under the interface's
    #    coordinates; a wrong column exits 0 and couples on, and this stops it.
    _if_nodes = [mp.Nodes[nid[(i_if, j)]] for j in range(NY + 1)]
    _off = [n for n in _if_nodes if abs(n.X - IFACE_X) > TOL]
    if _off:
        sys.exit(f"EXPORT SELF-CHECK: {len(_off)} of {len(_if_nodes)} nodes read as the "
                 f"interface column sit at x={_off[0].X:.6g}, not x={IFACE_X}: i_if names "
                 f"the wrong column. The interface column is the one whose x equals "
                 f"IFACE_X: ON_MAX_X = abs(IFACE_X - X1) < abs(IFACE_X - X0), then "
                 f"i_if = NX if ON_MAX_X else 0 and i_out = 0 if ON_MAX_X else NX")
    T = np.array([n.GetSolutionStepValue(KM.TEMPERATURE) for n in _if_nodes])

    # ── this side's own outward normal flux, from what the CONDITIONS assembled
    #
    # NOT A PROJECTION OF THE GRADIENT. Kratos stores REACTION_FLUX on FIXED dofs
    # only, so the reaction formula exports zero on these free interface dofs,
    # and an L2 projection of the elementwise P1 gradient is O(h) where the field
    # is O(h^2). MEASURED on the 3-D sibling, imposing q(y,z) = 2 + 3 sin(4y)
    # cos(3z) and asking for it back:
    #     gradient averaging  1.27e+00, 8.37e-01, 3.95e-01   order 0.60, 1.08
    #     assembled conditions 5.16e-01, 1.77e-01, 4.77e-02  order 1.55, 1.89
    # NOT AN ECHO OF q_in EITHER, though algebraically exact here: an echo never
    # passes through the discretisation, so applied on the wrong facets or with
    # the wrong sign it reads the same. Summing each condition's own right-hand
    # side MEASURES what entered the linear system: int_Gamma g phi_i ds when the
    # interface is built correctly, something else when not. On those rows
    # r_i = +(assembled interface load), so the outward density is -r_i/w_i --
    # the same expression the Dirichlet side uses.
    info = mp.ProcessInfo
    rhs_i = np.zeros(len(mp.Nodes) + 1)
    w_i = np.zeros(len(mp.Nodes) + 1)
    for cond in mp.Conditions:
        vec = KM.Vector()
        cond.CalculateRightHandSide(vec, info)
        nds = cond.GetNodes()
        # w_i = int_Gamma phi_i ds: in 2-D the interface facets are segments,
        # so a node's weight is half of each segment it belongs to.
        p0, p1 = nds[0], nds[1]
        seg = ((p1.X - p0.X) ** 2 + (p1.Y - p0.Y) ** 2) ** 0.5
        for k, nd in enumerate(nds):
            rhs_i[nd.Id] += float(vec[k])
            w_i[nd.Id] += 0.5 * seg
    ids_if = [nid[(i_if, j)] for j in range(NY + 1)]
    r_if = np.array([rhs_i[i] for i in ids_if])
    wq = np.array([w_i[i] for i in ids_if])
    Q = np.where(np.abs(wq) > 1e-14, -r_if / np.maximum(np.abs(wq), 1e-300)
                 * np.sign(np.where(wq == 0, 1.0, wq)), 0.0)
    if FULL_OUTER_DIRICHLET:
        # Corner reactions also contain the perpendicular outer-boundary flux
        # and cannot be separated into one interface contribution, so the
        # recovered interface flux at an interface/outer corner is not a clean
        # datum: retain the corner points for exchange, but zero this mixed
        # reaction rather than report it as interface flux.
        Q[[0, -1]] = 0.0

    # ── CONSERVATION SELF-CHECK: the discrete divergence theorem ──────────────
    # Summing the unconstrained residual r = A u - b over ALL nodes gives
    # sum_i r_i = -sum_i b_i, because sum_i A_ij = int K grad(sum_i phi_i).grad
    # phi_j = 0 (the phi_i are a partition of unity). r vanishes on free rows,
    # so over the fixed rows alone
    #       sum_{fixed} r_i  +  int_Omega f dOmega  +  int_Gamma q_applied ds  =  0
    # exactly, at round-off, for ANY mesh. It is not a discretisation check: it
    # fails only if the flux was applied with the wrong sign or magnitude, or
    # not applied at all — which is precisely the failure mode this side has.
    hy = (Y1 - Y0) / NY
    load_iface = float(hy * (0.5 * q_in[0] + q_in[1:-1].sum() + 0.5 * q_in[-1]))
    load_vol = 0.0
    _cw = 0                                           # triangles whose nodes run clockwise
    for el in mp.Elements:
        nds = el.GetNodes()
        x = [n.X for n in nds]
        y = [n.Y for n in nds]
        det = (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])
        _cw += det < 0
        load_vol += (0.5 * abs(det)) * sum(
            n.GetSolutionStepValue(KM.HEAT_FLUX) for n in nds) / 3.0
    if _cw:
        raise SystemExit(f"MESH: {_cw} of {len(mp.Elements)} triangles run clockwise; order "
                         f"each element's nodes counter-clockwise.")
    # Over EVERY fixed row: held y-edges' reactions belong to the sum too.
    react = sum(n.GetSolutionStepValue(KM.REACTION_FLUX) for n in mp.Nodes
                if n.IsFixed(KM.TEMPERATURE))
    imb_abs = abs(react + load_vol + load_iface)
    scale = max(abs(react), abs(load_vol), abs(load_iface))
    # No heat flow yet (iteration 1, no source): a ratio of round-offs means nothing.
    if scale <= 1e-10 * K * max(1.0, abs(T_OUTER)) * (Y1 - Y0):
        bal = f"balance trivial (no heat flow yet, |imbalance|={imb_abs:.3e})"
    else:
        bal = (f"balance |sum(reactions)+vol+iface| = {imb_abs:.3e} abs / "
               f"{imb_abs / scale:.3e} rel")
        if imb_abs / scale > 1e-3:        # exact on a right solve (measured 1e-16..4e-15)
            raise SystemExit(f"CONSERVATION SELF-CHECK: imbalance {imb_abs / scale:.3g} of scale: "
                             f"the partner's flux entered with the wrong sign or size, or not at "
                             f"all. Nothing was exported.")

    print(f"[kratos neumann] interface n={len(T)} "
          f"q_applied=[{q_in.min():.6g},{q_in.max():.6g}] "
          f"T=[{T.min():.6g},{T.max():.6g}] {bal}")
    print(f"\nNDOF = {len(mp.Nodes)}")

    # ── EXPORT SELF-CHECK (served) ─ keep this block. A singular system leaves NaN
    #    behind ("Error zero sum"), and nothing else would stop its export.
    if not (np.isfinite(np.asarray(T, float)).all() and np.isfinite(np.asarray(Q, float)).all()):
        raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes -- the solve "
                         "produced no usable field, so nothing was exported. When Kratos printed "
                         "'Error zero sum' or 'Error zero in diagonal', the assembled system is "
                         "singular: an element whose nodes run clockwise (negative area), a node "
                         "that belongs to no element, or no held value anywhere.")

    # PER-LEVEL PERSISTENCE: this level's whole field and its interface trace
    # and flux, named by LEVEL, never overwritten by the next level (exports.json
    # is): interpolate THESE onto the probe points your task names.
    # A DUMP DEFECT MUST NOT COST YOU THE SOLVE: exports.json, the driver's proof
    # that this participant succeeded, is written after these files.
    try:
        with open(f"field_level{LEVEL}.csv", "w") as _f:
            _f.write("x,y,u\n")
            for n in mp.Nodes:
                _f.write(f"{float(n.X):.11e},{float(n.Y):.11e},{float(n.GetSolutionStepValue(KM.TEMPERATURE)):.11e}\n")
        with open(f"interface_level{LEVEL}.csv", "w") as _f:
            _f.write("x,y,u,qn\n")
            for n, t, q in zip(_if_nodes, T, Q):
                _f.write(f"{float(n.X):.11e},{float(n.Y):.11e},{float(t):.11e},{float(q):.11e}\n")
    except Exception as _dump_exc:
        # AND LEAVE NO HALF-WRITTEN FILE BEHIND. `open(..., "w")` truncates
        # before it fails, so a dump that died mid-way leaves a header-only
        # CSV -- a file that looks like a submission and carries no rows.
        for _partial in (f"field_level{LEVEL}.csv", f"interface_level{LEVEL}.csv"):
            try:
                if Path(_partial).is_file() and len(
                        Path(_partial).read_text().splitlines()) <= 1:
                    Path(_partial).unlink()
            except OSError:
                pass
        print(f"[kratos_neumann per-level dump] level {LEVEL} dump failed: "
              f"{_dump_exc!r}. exports.json is still written, so the coupling\n"
              f"continues, but this level has no field file to hand in. Fix the\n"
              f"names the dump reads and run this level again.")

    # exports.json LAST: the driver takes its existence as proof of success.
    Path("exports.json").write_text(json.dumps({
        "field_name": "temperature",
        "n_points": int(len(T)),
        # the coordinates of the SAME node objects the values were read from
        "coordinates": [[float(n.X), float(n.Y)] for n in _if_nodes],
        "values": [float(t) for t in T],
        "normal_fluxes": [float(q) for q in Q],
    }, indent=2))
    return mp, nid


if __name__ == "__main__":
    main()
