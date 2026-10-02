"""Kratos Multiphysics participant for the openPASO `couple` driver (DIRICHLET side).

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST. Needs KratosMultiphysics + ConvectionDiffusionApplication
importable in the interpreter named in `command`.

This is the DIRICHLET side: it imports the partner's `values` (interface temperature), applies them as fixed nodal
TEMPERATURE on the interface, exports temperature + its own outward normal flux.

Slab [X0, X1] x [0, H], conductivity K.
  x = X0 : Dirichlet T = T_OUTER (hot side)
  x = X1 : INTERFACE, Dirichlet T = imported partner values (interpolated over y)
  top/bot: insulated (natural)

The exported outward normal flux density is the CONSISTENT (reaction) flux, not
a difference quotient of the solution — see the block above the export below.
"""
import json
import os
from pathlib import Path

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER = "right"
X0, X1 = 0.0, 0.5
H = 1.0
K = 1.0
T_OUTER = 115.0
T_INIT = 62.0            # iteration-1 fallback interface temperature
nx, ny = 32, 32
# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..}
#    next to this script overrides nx, ny and names the level; the per-level
#    dumps below carry that level so the coarse levels survive the fine ones.
LEVEL = 1
_cfg = {}
for _src, _txt in (("config.json", Path("config.json").read_text() if Path("config.json").is_file() else ""),
                   ("OPENPASO_CONFIG_JSON", os.environ.get("OPENPASO_CONFIG_JSON", ""))):
    try:
        _cfg.update(**json.loads(_txt or "{}"))
        LEVEL = int(_cfg.get("level", LEVEL))
        nx = int(_cfg.get("nx", nx))
        ny = int(_cfg.get("ny", ny))
    except (ValueError, TypeError) as _e:
        raise SystemExit(f"{_src} could not be read ({_e}); nothing was solved")
# ─────────────────────────────────────────────────────────────────────────



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


def imported_T(y_coords: np.ndarray) -> np.ndarray:
    p = Path("imports.json")
    imp = json.loads(p.read_text() or "{}") if p.is_file() else {}
    d = _partner_block(imp)
    if not d or "values" not in d:
        return np.full_like(y_coords, float(T_INIT))
    src_y = np.asarray(d["coordinates"], float)[:, 1]
    src_v = np.asarray(d["values"], float).ravel()
    o = np.argsort(src_y)
    return np.interp(y_coords, src_y[o], src_v[o])


def solve(T_if_in: np.ndarray):
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    model = KM.Model()
    mp = model.CreateModelPart("thermal")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    settings = KM.ConvectionDiffusionSettings()
    settings.SetUnknownVariable(KM.TEMPERATURE)
    settings.SetDiffusionVariable(KM.CONDUCTIVITY)
    settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
    settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)
    for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX,
              KM.REACTION_FLUX):   # REACTION_FLUX carries the interface reaction
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(1)

    props = mp.CreateNewProperties(1)
    nid, cnt = {}, 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            mp.CreateNewNode(cnt, X0 + (X1 - X0) * i / nx, H * j / ny, 0.0)
            nid[(i, j)] = cnt
            cnt += 1
    eid = 1
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = nid[(i, j)], nid[(i+1, j)], nid[(i+1, j+1)], nid[(i, j+1)]
            mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props); eid += 1
            mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props); eid += 1

    for node in mp.Nodes:
        node.SetSolutionStepValue(KM.CONDUCTIVITY, K)
        node.SetSolutionStepValue(KM.HEAT_FLUX, 0.0)
        node.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)
    for j in range(ny + 1):                       # outer hot side
        n = mp.Nodes[nid[(0, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, T_OUTER)
        n.Fix(KM.TEMPERATURE)
    for j in range(ny + 1):                       # INTERFACE Dirichlet = imported
        n = mp.Nodes[nid[(nx, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, float(T_if_in[j]))
        n.Fix(KM.TEMPERATURE)

    # AddDof with a REACTION variable: without the second argument the fixed
    # dofs have nowhere to store their reaction and it is silently discarded.
    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
    # arg 4 is CalculateReactionsFlag: it must be True, the interface flux below
    # is read out of the reactions.
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder,
                                              True, False, False, False)
    strategy.Initialize()
    strategy.Solve()
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
    # TWO THINGS YOUR SOLVE ABOVE MUST DO, or the recovery below reads zeros:
    #   * AddDof(TEMPERATURE, REACTION_FLUX, mp) -- the SECOND argument gives
    #     every fixed dof a place to store its reaction; without it the
    #     reaction is silently discarded.
    #   * run the strategy with CalculateReactionsFlag=True (the 4th positional
    #     argument of ResidualBasedLinearStrategy) -- the interface flux is
    #     read out of REACTION_FLUX, and a solve that never computed reactions
    #     exports a flux of exactly zero.
    return mp, nid


def main() -> None:
    y_if = np.array([H * j / ny for j in range(ny + 1)])
    T_in = imported_T(y_if)
    mp, nid = solve(T_in)
    # EVERY TRIANGLE COUNTER-CLOCKWISE: measured, a clockwise mesh leaves the field the
    # same to the bit and flips the sign of the reaction read below, with no error.
    _cw = sum((p[1].X - p[0].X) * (p[2].Y - p[0].Y) < (p[2].X - p[0].X) * (p[1].Y - p[0].Y)
              for p in (e.GetNodes() for e in mp.Elements))
    if _cw:
        raise SystemExit(f"MESH: {_cw} of {len(mp.Elements)} triangles run clockwise; order "
                         f"each element's nodes counter-clockwise.")

    T_if = np.array([mp.Nodes[nid[(nx, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                     for j in range(ny + 1)])

    # Outward normal flux density q = -(K grad T).n on the interface.
    #
    # NOT A DIFFERENCE QUOTIENT OR A PROJECTED GRADIENT: the gradient of a P1
    # solution is only O(h) accurate ON the boundary, which is exactly what the
    # coupling reads. Measured against a manufactured solution with a known
    # exact interface flux: order ~1 for those, ~2 for the consistent flux below.
    #
    # THE CONSISTENT (REACTION) FLUX. From
    #     a(u,v) - (f,v) = int_dOmega (K grad u . n) v ds = -int_Gamma qn v ds
    # it follows that for every basis function phi_i on the interface
    #     int_Gamma qn phi_i ds = -r_i,   r = A u_h - b
    # with r the UNCONSTRAINED residual on the constrained rows — which is
    # precisely what Kratos calls the REACTION: its builder recomputes the RHS
    # with NO Dirichlet condition applied and stores -b_i = (A u_h - b)_i at
    # every fixed dof. Dividing by w_i = int_Gamma phi_i ds turns the functional
    # into a density the partner can interpolate pointwise.
    #
    # w_i for the P1 trace on this interface: the interface is the straight
    # segment x = X1 cut into ny equal pieces by THIS script's own node layout,
    # so int_Gamma phi_i ds is hy for an interior node and hy/2 at the two ends,
    # exactly. Regenerate this if the mesh above is ever made non-uniform.
    hy = H / ny
    w = np.full(ny + 1, hy)
    w[0] = w[-1] = 0.5 * hy

    r = np.array([mp.Nodes[nid[(nx, j)]].GetSolutionStepValue(KM.REACTION_FLUX)
                  for j in range(ny + 1)])
    q_out = -r / w

    # An interface node that ALSO lies on the outer Dirichlet boundary carries
    # the OUTER reaction as well, so its residual is not this interface's flux.
    # With nx >= 1 the two x-faces share no node, but the guard is kept because
    # an edited geometry can make them coincide.
    outer_ids = {nid[(0, j)] for j in range(ny + 1)}
    suspect = np.array([nid[(nx, j)] in outer_ids for j in range(ny + 1)])
    good = np.where(~suspect)[0]
    if len(good):
        for i in np.where(suspect)[0]:
            q_out[i] = q_out[good[np.argmin(np.abs(good - i))]]

    # ── EXPORT SELF-CHECK ─ keep this block. It stops two exports that look fine
    #    and are worthless: a non-finite field, and a flux that is the partner's
    #    array negated instead of a recovery from THIS side's own system.
    _chk_vals = np.asarray(T_if, float).ravel()
    _chk_flux = np.asarray(q_out, float).ravel()
    if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
        raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes; "
                         "the solve did not produce a usable field, so nothing was "
                         "exported")
    _chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
                if Path("imports.json").is_file() else {})
    _chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                                for _d in _chk_imp.values()])
                if _chk_imp else np.zeros(0))
    # (A Dirichlet side's check: a Neumann side's consistent recovery of a CONSTANT
    #  applied flux can legitimately reproduce it to the last bit.)
    if _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
            and np.array_equal(_chk_flux, -_chk_qin):
        raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's "
                         "array negated, bit for bit: a copy, not a recovery from "
                         "this side's own assembled system")

    # THE RUN-LOG CONTRACT LINE: `NDOF = <integer>` on a line of its OWN.
    # The audit reads that exact shape, and they read it PER
    # LEVEL: it is how anyone checking the result tells a refined mesh from the same mesh run
    # three times. A number inside a prose sentence does not count, and a
    # wrong number is worse than none -- one coupled run that was right in
    # every other respect reported NDOF = 1 at all three levels, and its
    # refined mesh could not be told from an unrefined one.
    try:
        print(f"\nNDOF = {int(len(mp.Nodes))}")
    except Exception as _ndof_exc:
        print(f"[kratos] could not report NDOF: {_ndof_exc!r}. Your task's"
              f" execution log needs `NDOF = <integer>` on a line of its own,"
              f" so print your own degree-of-freedom count here.")

    # PER-LEVEL PERSISTENCE: this level's field and interface data, named by
    # LEVEL; exports.json is overwritten by the next level, these are not.
    # Interpolate THESE onto the probe points your task names. A file the next
    # level overwrites cannot carry a mesh study.
    # A DUMP DEFECT MUST NOT COST YOU THE SOLVE. exports.json is the driver's
    # proof that this participant succeeded, and it is written after these files,
    # so an exception here would throw away a coupling iteration that worked.
    try:
        with open(f"field_level{LEVEL}.csv", "w") as _f:
            _f.write("x,y,u\n")
            for n in mp.Nodes:
                _f.write(f"{float(n.X):.11e},{float(n.Y):.11e},{float(n.GetSolutionStepValue(KM.TEMPERATURE)):.11e}\n")
        with open(f"interface_level{LEVEL}.csv", "w") as _f:
            _f.write("x,y,u,qn\n")
            for y, t, q in zip(y_if, T_if, q_out):
                _f.write(f"{float(X1):.11e},{float(y):.11e},{float(t):.11e},{float(q):.11e}\n")
    except Exception as _dump_exc:
        # AND KEEP BOTH FILES OR NEITHER. A dump that failed part-way can leave a
        # truncated file, a whole field file with no interface file, or a file an
        # earlier run wrote, and any of them could be read as this level's result.
        # So both of this level's files go, whatever they hold.
        for _partial in (f"field_level{LEVEL}.csv", f"interface_level{LEVEL}.csv"):
            try:
                Path(_partial).unlink(missing_ok=True)
            except OSError:
                pass
        print(f"[kratos per-level dump] level {LEVEL} dump failed: "
              f"{_dump_exc!r}. exports.json is still written, so the coupling\n"
              f"continues, but this level has no field file to hand in. Fix the\n"
              f"names the dump reads and run this level again.")
    Path("exports.json").write_text(json.dumps({
        "field_name": "temperature",
        "n_points": len(T_if),
        "coordinates": [[X1, float(y)] for y in y_if],
        "values": [float(v) for v in T_if],
        "normal_fluxes": [float(v) for v in q_out],
    }, indent=2))
    print("kratos DIRICHLET side: T_if(applied)=%.6f q_out(computed)=%.6f"
          % (float(T_if.mean()), float(q_out.mean())))


if __name__ == "__main__":
    main()
