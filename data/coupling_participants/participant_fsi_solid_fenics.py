"""FEniCSx (dolfinx) STRUCTURE participant for the openPASO `couple` driver — FSI.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

Physics: plane-strain linear elasticity on a rectangular wall clamped at both
ends.  The bottom edge is the FSI interface and carries the traction handed in
by the fluid participant as a NEUMANN load.  What is exported back is the
interface displacement.

SIGN CONVENTION — the imported traction is already the load ON THIS BODY
(t = sigma_f . n_s, with n_s the structure's outward normal on the interface).
It is applied DIRECTLY, with NO further sign change.  See the fluid
participant's docstring for the derivation.

This is the SAME physics as participant_fsi_solid_skfem.py in a different code.
Running the FSI pair once with each is the same-code / cross-code comparison:
the fluid is FEniCSx either way, so any difference in the converged interface
displacement is the structure discretisation, not the coupling.
"""
import dolfinx
import json
import sys
from pathlib import Path

import numpy as np
# MAKE THIS CODE SPEAK, BEFORE THE SOLVE RUNS. It is silent by default, and a
# per-level run log carrying no line the solver itself emitted cannot
# establish which code ran on this side, however right its numbers are.
# It sits HERE, beside the level rule, and not up with the imports:
# measured over agent-written participants, a line placed in the import
# block survived in about half of them because that block gets rewritten,
# while everything beside the level rule survived in all of them.
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER    = "fluid"     # the fluid participant's `name` in your couple(...) call
LX         = 1.2         # wall length
Y0         = 0.18        # the FSI interface (this body's LOWER edge)
HS         = 0.04        # wall thickness
NXS, NYS   = 46, 3       # this body's OWN mesh; need not match the fluid's
E_MOD      = 1.5e6       # Young's modulus
NU         = 0.35       # Poisson ratio
CLAMP_X    = (0.0, 1.2)  # x positions of the clamped ends
T_INIT     = 0.0         # iteration-1 fallback interface traction (both comps)
FEEDBACK   = True        # SET False ONLY to suppress the fluid->structure
                         # direction (freezes the load at T_INIT). A real FSI
                         # run keeps this True.
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


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return _partner_block(json.loads(p.read_text()))
    except json.JSONDecodeError:
        return None


def sample_vec(imp, x_targets, fallback, ncomp=2):
    """Map the partner's samples onto THIS participant's interface points."""
    x_targets = np.asarray(x_targets, float)
    if not imp or not imp.get("coordinates"):
        return np.full(x_targets.shape + (ncomp,), float(fallback))
    xs = np.asarray(imp["coordinates"], float)[:, 0]
    vals = np.asarray(imp["values"], float).reshape(len(xs), -1)
    order = np.argsort(xs)
    out = np.zeros(x_targets.shape + (ncomp,))
    for c in range(min(ncomp, vals.shape[1])):
        out[..., c] = np.interp(x_targets, xs[order], vals[order, c])
    return out


def main():
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    comm = MPI.COMM_SELF
    msh = dmesh.create_rectangle(
        comm, [np.array([0.0, Y0]), np.array([LX, Y0 + HS])], [NXS, NYS],
        dmesh.CellType.triangle)
    gdim = msh.geometry.dim

    lam = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))       # plane strain
    mu = E_MOD / (2.0 * (1.0 + NU))

    V = fem.functionspace(msh, ("Lagrange", 2, (gdim,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sig(w):
        return 2.0 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(gdim)

    fdim = msh.topology.dim - 1
    iface = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], Y0))
    ft = dmesh.meshtags(msh, fdim, np.sort(iface),
                        np.full(len(iface), 4, dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    # traction as a P1 vector field on the interface, sampled from the import
    imp = read_imports() if FEEDBACK else None
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    V1 = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
    t_fn = fem.Function(V1)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
    xd = V1.tabulate_dof_coordinates()[:, :gdim]
    t_fn.x.array.reshape(-1, gdim)[:] = sample_vec(imp, xd[:, 0], T_INIT, gdim)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    a = ufl.inner(sig(u), eps(v)) * ufl.dx
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
    L = ufl.inner(t_fn, v) * ds(4)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    clamp_facets = dmesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], CLAMP_X[0]) | np.isclose(x[0], CLAMP_X[1]))
    zero = fem.Function(V)
    bcs = [fem.dirichletbc(zero, fem.locate_dofs_topological(V, fdim, clamp_facets))]

    d = fem.Function(V, name="displacement")
    LinearProblem(a, L, bcs=bcs, u=d, petsc_options_prefix="solid_",
                  petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                 "pc_factor_mat_solver_type": "mumps"}).solve()
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    # ── interface displacement at the interface NODES ──────────────────────
    d1 = fem.Function(V1)
    d1.interpolate(d)
    inode = np.where(np.abs(xd[:, 1] - Y0) < 1e-9)[0]
    inode = inode[np.argsort(xd[inode, 0])]
    x_if = xd[inode, 0]
    disp = d1.x.array.reshape(-1, gdim)[inode]
    t_applied = sample_vec(imp, x_if, T_INIT, gdim)
    ref_coords = np.column_stack([x_if, np.full_like(x_if, Y0)])

    one = fem.Constant(msh, 1.0)
    fx = fem.assemble_scalar(fem.form(t_fn[0] * one * ds(4)))
    fy = fem.assemble_scalar(fem.form(t_fn[1] * one * ds(4)))

    # ── EXPORT SELF-CHECK ─ keep this block. TWO of the three checks, and the
    #    third DELIBERATELY LEFT OUT: in FSI the two sides' tractions are
    #    anti-parallel BY CONVENTION and must sum to zero, so the scalar
    #    contracts' "exported flux is the partner's array negated" check would
    #    fault a correct export here.
    _chk_d = np.asarray(disp, float).ravel()
    _chk_t = np.asarray(t_applied, float).ravel()
    if not (np.isfinite(_chk_d).all() and np.isfinite(_chk_t).all()):
        raise SystemExit("EXPORT SELF-CHECK: non-finite interface displacement "
                         "or traction; the solve did not produce a usable "
                         "field, so nothing was exported")
    _chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
                if Path("imports.json").is_file() else {})
    _chk_tin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                                for _d in _chk_imp.values()])
                if _chk_imp else np.zeros(0))
    if _chk_tin.size and np.abs(_chk_tin).max() > 0 \
            and np.abs(_chk_d).max() < 1e-12 * np.abs(_chk_tin).max():
        raise SystemExit("EXPORT SELF-CHECK: the structure returns a ~0 "
                         "displacement against a nonzero imported traction: "
                         "the fluid load never entered the assembled system "
                         "(the Neumann condition that applies it is missing or "
                         "on the wrong surface). Fix the application; do not "
                         "couple on")

    out = {
        "field_name": "interface_displacement",
        "n_points": int(len(x_if)),
        "coordinates": ref_coords.tolist(),
        "values": disp.tolist(),
        # w.r.t. THIS body's own outward normal n_s, so this and the fluid's
        # normal_fluxes (w.r.t. n_f = -n_s) must SUM to zero.
        "normal_fluxes": t_applied.tolist(),
        "meta": {
            "net_force_received": [float(fx), float(fy)],
            "feedback": bool(FEEDBACK),
            "max_abs_disp": [float(np.max(np.abs(disp[:, 0]))),
                             float(np.max(np.abs(disp[:, 1])))],
            "n_dofs": int(V.dofmap.index_map.size_global * gdim),
        },
    }
    Path("exports.json").write_text(json.dumps(out, indent=2))
    print(f"[solid-fenics] recv_force=({fx:.6e},{fy:.6e}) "
          f"max|dy|={np.max(np.abs(disp[:, 1])):.6e}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
