"""FEniCSx (dolfinx) FLUID participant for the openPASO `couple` driver — FSI.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

Physics: incompressible Navier-Stokes (Taylor-Hood P2/P1) on a rectangular
channel whose ONE moving boundary is the FSI interface.  The interface
displacement handed in by the structure participant is extended into the fluid
domain by a harmonic (Laplace) ALE lift and the mesh is moved by it before the
flow is solved.  What is exported back is the traction the fluid exerts ON THE
STRUCTURE.

SIGN CONVENTION — the single most common FSI wiring error, so it is written
down here and nowhere else is allowed to re-derive it:

    exported traction  t = sigma_f . n_s        with n_s = -n_f

    n_f  outward unit normal of the FLUID domain on the interface
    n_s  outward unit normal of the STRUCTURE on the same surface = -n_f

  Cauchy's t(n) = sigma.n is the traction exerted BY the material n points INTO,
  ON the material n points OUT OF.  So sigma_f . n_f is what the STRUCTURE does
  to the FLUID; the load on the structure is its negative.  Check: static fluid
  at pressure p>0 under a wall gives sigma_f = -p I, n_f = +e_y, and the export
  t = -sigma_f . n_f = +p e_y  — the fluid pushes the wall AWAY from itself.

  The structure participant applies the imported t DIRECTLY as a Neumann load
  (no further sign change).  Flipping this sign is a physics mutation: it turns
  a bulging wall into a collapsing one.

The traction is not evaluated pointwise (sigma.n of a Taylor-Hood solution is
discontinuous and inaccurate at nodes).  It is the VARIATIONALLY CONSISTENT
traction: the L2(Gamma) projection of sigma.n onto the interface trace of the
continuous P1 vector space, i.e. the consistent nodal forces divided through by
the interface mass matrix.  That is the field whose integral over Gamma is the
exact discrete interface force, which is what the equilibrium check needs.

Interface parametrisation is LAGRANGIAN: exported/imported coordinates are the
REFERENCE (undeformed) positions of the interface nodes, on both sides.  The
interface is a material surface, so this is the stable parametrisation; using
deformed coordinates makes the exchange chase its own tail.
"""
import dolfinx
import json
import os
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
import basix.ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER    = "solid"     # the structure participant's `name` in your couple(...) call
LX         = 1.2         # channel length
HY         = 0.18        # channel height (undeformed)
IFACE_SIDE = "top"       # which fluid boundary is the FSI interface: "top" | "bottom"
NX, NY     = 46, 6       # fluid mesh; config.json's nx, ny set it per level (below)
MU         = 0.8         # dynamic viscosity
RHO_F      = 1.2         # fluid density
U_MEAN     = 0.75        # mean inflow speed (parabolic profile)
DT         = 0.0         # 0.0 -> STEADY. >0 -> ONE backward-Euler step from rest:
                         # the unsteady term rho_f/dt * (u - 0) enters, and the
                         # interface is a MOVING wall with u = d/dt rather than a
                         # stationary one. This is the setting in which the
                         # added-mass instability of partitioned FSI exists at all;
                         # a steady coupling has no added mass.
D_INIT     = 0.0         # iteration-1 fallback interface displacement (both comps)
MOVE_MESH  = True        # SET False ONLY to suppress the structure->fluid direction
                         # (the one-way control). A real FSI run keeps this True.
# ─────────────────────────────────────────────────────────────────────────


# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..} next to this
#    script, and the keys a multi-level couple call hands in OPENPASO_CONFIG_JSON (they win), set
#    NX, NY and name the level; the per-level dumps below carry that level, so the coarse levels
#    survive the fine ones. A config that cannot be read stops the run: a level run on the edit
#    block's mesh would hand in the wrong level.
LEVEL = 1
try:
    _cfg = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
    _cfg.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))
    LEVEL = int(_cfg.get("level", LEVEL))
    NX, NY = int(_cfg.get("nx", NX)), int(_cfg.get("ny", NY))
except (ValueError, TypeError, AttributeError) as _cfg_exc:
    raise SystemExit(f"config.json / OPENPASO_CONFIG_JSON: the level and the mesh keys nx, ny could "
                     f"not be read ({_cfg_exc!r}); fix them, nothing was solved")

IFACE_Y = HY if IFACE_SIDE == "top" else 0.0
# outward normal of the FLUID on the interface, as a sign on e_y
NF_SIGN = 1.0 if IFACE_SIDE == "top" else -1.0



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
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return _partner_block(json.loads(p.read_text()))
    except json.JSONDecodeError:
        return None


def sample_vec(imp, x_targets, fallback, ncomp=2):
    """Map the partner's samples onto THIS participant's interface points.

    The driver does no interpolation — non-matching interface meshes are handled
    here, component by component, by 1-D interpolation along the interface
    parameter (the x coordinate).  np.interp clamps outside the partner's range,
    which is what we want at a clamped end.
    """
    if not imp or not imp.get("coordinates"):
        return np.full((len(x_targets), ncomp), float(fallback))
    xs = np.asarray(imp["coordinates"], float)[:, 0]
    vals = np.asarray(imp["values"], float).reshape(len(xs), -1)
    order = np.argsort(xs)
    out = np.zeros((len(x_targets), ncomp))
    for c in range(min(ncomp, vals.shape[1])):
        out[:, c] = np.interp(x_targets, xs[order], vals[order, c])
    return out


# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
def _signed_areas(msh) -> np.ndarray:
    """Twice the signed area of every (P1, triangular) cell, from the geometry
    as it stands. Sign is per-cell and arbitrary — dolfinx does not orient all
    cells the same way — so it is only ever compared with the SAME cell's value
    on the undeformed mesh."""
    cells = np.asarray(msh.geometry.dofmap).reshape(-1, 3)
    px = msh.geometry.x[:, :2]
    v0 = px[cells[:, 1]] - px[cells[:, 0]]
    v1 = px[cells[:, 2]] - px[cells[:, 0]]
    return v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end


# ── HOLE NAMES IN WORDS ─ begin
#   PETSc  the PETSc module of petsc4py
#   W      the function space the flow is solved on, velocity and pressure
#          together; its dof count is printed as NDOF
#   dofc   the dof coordinates of the continuous P1 VECTOR space that tt and
#          vv belong to (its tabulate_dof_coordinates(), first gdim columns),
#          taken BEFORE the mesh moves: one row per mesh node, in that space's
#          dof order. The served lines pick the interface rows out of it and
#          index the projected traction, one row per node of that space, with
#          them. Not W's: a mixed space cannot tabulate
#   ds     ufl.Measure('ds', domain=msh, subdomain_data=<facet tags>), the
#          interface facets tagged 4 and the inlet facets tagged 1
#   fem    the dolfinx.fem module
#   gdim   the geometric dimension, msh.geometry.dim
#   msh    the dolfinx mesh, moved by the imported interface displacement
#          before the flow is solved on it
#   n      ufl.FacetNormal(msh), the fluid's outward unit normal
#   nit    the Newton iteration count of the flow solve, an int
#   ph     the solved pressure, a fem.Function (the solution's pressure
#          sub-function) or a UFL expression of it: the served lines
#          integrate it over ds(1) and write it at the mesh nodes
#   sigma  the fluid's Cauchy stress of the solved flow, a gdim x gdim UFL
#          expression; the served lines project -dot(sigma, n) over ds(4)
#   tt     ufl.TrialFunction of that same P1 vector space
#   uh     the solved velocity, a fem.Function (the solution's velocity
#          sub-function) or a UFL expression of it, written at the mesh
#          nodes by the per-level dump
#   ufl    the ufl module
#   vv     ufl.TestFunction of that same P1 vector space
# ── HOLE NAMES IN WORDS ─ end


def main():
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    comm = MPI.COMM_SELF
    msh = dmesh.create_rectangle(
        comm, [np.array([0.0, 0.0]), np.array([LX, HY])], [NX, NY],
        dmesh.CellType.triangle)
    gdim = msh.geometry.dim

    # ── P1 vector space used for BOTH the ALE lift and the traction projection.
    #    Its dof ordering must coincide with the geometry node ordering for the
    #    in-place mesh move below; assert it rather than trust it.
    V1 = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
    dofc = V1.tabulate_dof_coordinates()[:, :gdim]
    if dofc.shape[0] != msh.geometry.x.shape[0] or not np.allclose(
            dofc, msh.geometry.x[:, :gdim], atol=1e-12):
        raise RuntimeError(
            "P1 dof ordering does not match the geometry node ordering; the "
            "in-place mesh move would scramble the mesh")
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    # REFERENCE interface node coordinates (Lagrangian parametrisation)
    tol = 1e-9
    iface_nodes = np.where(np.abs(dofc[:, 1] - IFACE_Y) < tol)[0]
    x_iface = dofc[iface_nodes, 0]
    order = np.argsort(x_iface)
    iface_nodes = iface_nodes[order]
    x_iface = x_iface[order]
    ref_coords = np.column_stack([x_iface, np.full_like(x_iface, IFACE_Y)])

    # ── imported interface displacement ────────────────────────────────────
    imp = read_imports()
    d_iface = sample_vec(imp, x_iface, D_INIT, ncomp=2)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    # ── facet tags: 1 inflow, 2 outflow, 3 fixed wall, 4 FSI interface ─────
    def _inflow(x):
        return np.isclose(x[0], 0.0)

    def _outflow(x):
        return np.isclose(x[0], LX)

    def _iface(x):
        return np.isclose(x[1], IFACE_Y)

    def _wall(x):
        return np.isclose(x[1], HY - IFACE_Y)   # the other horizontal boundary

    fdim = msh.topology.dim - 1
    marks, facets = [], []
    for tag, fn in ((1, _inflow), (2, _outflow), (3, _wall), (4, _iface)):
        f = dmesh.locate_entities_boundary(msh, fdim, fn)
        facets.append(f)
        marks.append(np.full(len(f), tag, dtype=np.int32))
    facets = np.concatenate(facets)
    marks = np.concatenate(marks)
    srt = np.argsort(facets)
    ft = dmesh.meshtags(msh, fdim, facets[srt], marks[srt])

    # ── ALE lift: harmonic extension of the interface displacement ─────────
    area0 = _signed_areas(msh)          # undeformed reference for the guard
    d_ale = fem.Function(V1, name="ale_displacement")
    if MOVE_MESH and np.any(np.abs(d_iface) > 0):
        u_, v_ = ufl.TrialFunction(V1), ufl.TestFunction(V1)
        # JACOBIAN STIFFENING: weight the extension by 1/|cell| raised to
        # ALE_STIFF, so SMALL cells are stiff and take less of the deformation
        # than large ones. That is what keeps a graded mesh from inverting near
        # the interface first, where the cells are smallest and the motion is
        # largest. ALE_STIFF = 0 is the plain harmonic extension.
        #
        # On a UNIFORM mesh every cell has the same volume, so the weight is a
        # single constant multiplying a bilinear form whose right-hand side is
        # zero — it cannot change the answer there, by construction. This knob
        # earns its place only on a graded mesh, and it is written this way
        # rather than as a bare `ALE_STIFF *` factor because a bare factor is a
        # NO-OP EVERYWHERE and would have read as a stiffening that was never
        # applied.
        ALE_STIFF = 1.0          # Jacobian stiffening exponent; 0.0 = plain harmonic
        a = ((1.0 / ufl.CellVolume(msh)) ** ALE_STIFF
             * ufl.inner(ufl.grad(u_), ufl.grad(v_)) * ufl.dx)
        L = ufl.inner(fem.Constant(msh, np.zeros(gdim)), v_) * ufl.dx
        g_iface = fem.Function(V1)
        g_iface.x.array.reshape(-1, gdim)[iface_nodes] = d_iface
        zero = fem.Function(V1)
        bcs = [fem.dirichletbc(g_iface, fem.locate_dofs_topological(
                   V1, fdim, ft.find(4))),
               fem.dirichletbc(zero, fem.locate_dofs_topological(
                   V1, fdim, np.concatenate([ft.find(1), ft.find(2), ft.find(3)])))]
        pr = LinearProblem(a, L, bcs=bcs, u=d_ale,
                           petsc_options_prefix="ale_",
                           petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        pr.solve()
        msh.geometry.x[:, :gdim] += d_ale.x.array.reshape(-1, gdim)

        # ── REFUSE AN INVERTED MESH. A partitioned FSI that is diverging hands
        # over a displacement larger than the domain within a few iterations;
        # the ALE lift then folds cells over and the flow solve that follows is
        # on a mesh with negative volumes. What that produced before this guard
        # was not an error but a HANG: the Newton solve on a folded mesh neither
        # converges nor fails, the participant sits inside its timeout, and the
        # coupling looks like a slow run rather than a divergence. Exiting here
        # turns it into the finding it is.
        moved = _signed_areas(msh)
        # Each cell against ITS OWN undeformed area, not against a global sign:
        # dolfinx does not give every triangle the same node orientation, so
        # half the cells have negative signed area on a perfectly good mesh.
        # The first version of this guard compared against the median sign and
        # reported exactly half the cells as inverted on an undeformed mesh —
        # a guard that fires on every correct run is worse than none.
        bad = int(np.sum(moved * area0 <= 0.0))
        shrunk = float(np.min(np.abs(moved) / np.maximum(np.abs(area0), 1e-300)))
        if bad or shrunk < 1e-6:
            raise RuntimeError(
                f"ALE mesh INVERTED or DEGENERATE: {bad} of {len(moved)} cells "
                f"changed orientation and the smallest cell retained "
                f"{shrunk:.2e} of its undeformed area, after applying an "
                f"interface displacement of max |d| = "
                f"{np.max(np.abs(d_iface)):.3e} against a domain height of "
                f"{HY:.3e}. The coupling iteration is DIVERGING (added mass, "
                f"too large a theta, or a sign error) — it is not converging "
                f"slowly.")

    # ── Navier-Stokes, Taylor-Hood P2/P1, monolithic Newton ────────────────
    Ve = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(gdim,))
    Qe = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = fem.functionspace(msh, basix.ufl.mixed_element([Ve, Qe]))
    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    n = ufl.FacetNormal(msh)

    F = (RHO_F * ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
         + MU * ufl.inner(ufl.grad(u) + ufl.grad(u).T, ufl.grad(v)) * ufl.dx
         - ufl.inner(p, ufl.div(v)) * ufl.dx
         + ufl.inner(ufl.div(u), q) * ufl.dx)
    if DT > 0.0:
        # backward Euler, ONE step from rest: (u - u_old)/dt with u_old = 0
        F += (RHO_F / DT) * ufl.inner(u, v) * ufl.dx

    W0 = W.sub(0)
    Vsub, _ = W0.collapse()

    def _parabolic(x):
        vals = np.zeros((gdim, x.shape[1]))
        vals[0] = 6.0 * U_MEAN * x[1] * (HY - x[1]) / HY**2
        return vals

    u_in = fem.Function(Vsub)
    u_in.interpolate(_parabolic)
    u_zero = fem.Function(Vsub)
    # THE INTERFACE VELOCITY. Steady: the interface is a stationary no-slip wall
    # in its DEFORMED position, so u = 0 there and the mesh carries the whole
    # coupling. Transient: the wall is MOVING, u = d/dt, and the kinematic
    # condition is what makes the fluid feel the structure's inertia — this term
    # is the added mass.
    if DT > 0.0:
        u_iface = fem.Function(Vsub)
        u_iface.interpolate(d_ale)
        u_iface.x.array[:] /= DT
    else:
        u_iface = u_zero
    bcs = [
        fem.dirichletbc(u_in, fem.locate_dofs_topological(
            (W0, Vsub), fdim, ft.find(1)), W0),
        fem.dirichletbc(u_zero, fem.locate_dofs_topological(
            (W0, Vsub), fdim, ft.find(3)), W0),
        fem.dirichletbc(u_iface, fem.locate_dofs_topological(
            (W0, Vsub), fdim, ft.find(4)), W0),
    ]

    problem = NonlinearProblem(
        F, w, bcs=bcs, petsc_options_prefix="ns_",
        petsc_options={"snes_type": "newtonls", "snes_rtol": 1e-11,
                       "snes_atol": 1e-12, "snes_max_it": 40,
                       "ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps",
                       "snes_error_if_not_converged": True,
                       "ksp_error_if_not_converged": True})
    problem.solve()
    reason = problem.solver.getConvergedReason()
    nit = problem.solver.getIterationNumber()
    if reason <= 0:
        raise RuntimeError(f"fluid Newton did not converge (reason={reason}, "
                           f"{nit} iterations)")
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    # ── variationally consistent traction on the interface ─────────────────
    #   t = sigma_f . n_s = -sigma_f . n_f    (see the module docstring)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    uh, ph = w.sub(0), w.sub(1)
    sigma = -ph * ufl.Identity(gdim) + MU * (ufl.grad(uh) + ufl.grad(uh).T)
    tt, vv = ufl.TrialFunction(V1), ufl.TestFunction(V1)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end
    a_m = ufl.inner(tt, vv) * ds(4)
    L_t = ufl.inner(-ufl.dot(sigma, n), vv) * ds(4)

    from dolfinx.fem.petsc import assemble_matrix, assemble_vector
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    A = assemble_matrix(fem.form(a_m))
    A.assemble()
    ai, aj, av = A.getValuesCSR()
    M = sp.csr_matrix((av, aj, ai), shape=A.getSize())
    b = assemble_vector(fem.form(L_t))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bvec = b.array.copy()
    # the mass matrix is singular off the interface: put 1 on those diagonals
    diag = M.diagonal().copy()
    dead = np.where(np.abs(diag) < 1e-14)[0]
    Mfix = M.tolil()
    for i in dead:
        Mfix[i, i] = 1.0
    bvec[dead] = 0.0
    t_all = spla.spsolve(Mfix.tocsc(), bvec).reshape(-1, gdim)
    traction = t_all[iface_nodes]

    # net interface force, for the equilibrium check on the other side
    fx = fem.assemble_scalar(fem.form(-ufl.dot(sigma, n)[0] * ds(4)))
    fy = fem.assemble_scalar(fem.form(-ufl.dot(sigma, n)[1] * ds(4)))

    # THE RUN-LOG CONTRACT LINE: `NDOF = <integer>` on a line of its OWN, printed
    # PER LEVEL. It is how anyone checking the result tells a refined mesh from the same mesh run
    # three times. The LEADING NEWLINE is deliberate: a program that writes without
    # a trailing newline glues its text onto the front of the next line.
    try:
        print(f"\nNDOF = {int(W.dofmap.index_map.size_global * W.dofmap.index_map_bs)}")
    except Exception as _ndof_exc:
        print(f"[fsi-fluid] could not report NDOF: {_ndof_exc!r}. Your task's execution"
              f" log needs `NDOF = <integer>` on a line of its own, so print your"
              f" own degree-of-freedom count here.")

    # ── EXPORT SELF-CHECK ─ keep this block. TWO of the three checks, and the
    #    third DELIBERATELY LEFT OUT. A non-finite export is worthless, and so
    #    is a fluid that reports ~0 traction on a nonzero interface: the load
    #    path is not there, and it still couples and still converges. The
    #    "exported flux is the partner's array negated" check that the scalar
    #    contracts carry does NOT belong here: in FSI the two sides' tractions
    #    are anti-parallel BY CONVENTION and must sum to zero, so that check
    #    would fault a correct export.
    _chk_t = np.asarray(traction, float).ravel()
    _chk_d = np.asarray(d_iface, float).ravel()
    if not (np.isfinite(_chk_t).all() and np.isfinite(_chk_d).all()):
        raise SystemExit("EXPORT SELF-CHECK: non-finite interface traction or "
                         "imposed displacement; the solve did not produce a "
                         "usable field, so nothing was exported")
    if _chk_t.size and np.abs(_chk_t).max() == 0.0:
        raise SystemExit("EXPORT SELF-CHECK: the exported traction is "
                         "identically zero on every interface point. A viscous "
                         "channel flow exerts a load on its wall; a zero here "
                         "means the stress was never evaluated on the interface "
                         "facets. Fix the recovery; do not couple on")


    # PER-LEVEL PERSISTENCE: this level's whole flow field, at the nodes of the moved mesh, and its
    # interface data -- the reference coordinates, the displacement imposed there, the traction
    # exported -- named by LEVEL and never overwritten by the next level (exports.json is).
    # Interpolate THESE onto the points your task names. A DUMP DEFECT MUST NOT COST THE SOLVE:
    # exports.json is written after them either way, and a half-written dump is removed.
    _dumps = (f"field_level{LEVEL}.csv", f"interface_level{LEVEL}.csv")
    try:
        def _at_nodes(value, space):
            _f = fem.Function(space)
            if isinstance(value, fem.Function):
                _f.interpolate(value)
            else:                                   # a UFL expression of the solution
                _ip = space.element.interpolation_points
                _f.interpolate(fem.Expression(value, _ip() if callable(_ip) else _ip))
            return _f.x.array.reshape(-1, space.dofmap.index_map_bs)
        _Vn = tt.ufl_function_space()               # the P1 vector space of the projection
        _Pn = fem.functionspace(msh, ("Lagrange", 1))
        _xv = _Vn.tabulate_dof_coordinates()[:, :gdim]
        _xp = _Pn.tabulate_dof_coordinates()[:, :gdim]
        _iv = np.lexsort((np.round(_xv[:, 1], 12), np.round(_xv[:, 0], 12)))
        _ip = np.lexsort((np.round(_xp[:, 1], 12), np.round(_xp[:, 0], 12)))
        _vel, _pre = _at_nodes(uh, _Vn)[_iv], _at_nodes(ph, _Pn)[_ip, 0]
        with open(_dumps[0], "w") as _f:
            _f.write("x,y,vx,vy,p\n")
            for (_px, _py), (_vx, _vy), _pp in zip(_xv[_iv], _vel, _pre):
                _f.write(f"{_px:.11e},{_py:.11e},{_vx:.11e},{_vy:.11e},{_pp:.11e}\n")
        with open(_dumps[1], "w") as _f:
            _f.write("x,y,dx,dy,tx,ty\n")
            for (_px, _py), (_dx, _dy), (_tx, _ty) in zip(ref_coords, d_iface, traction):
                _f.write(f"{_px:.11e},{_py:.11e},{_dx:.11e},{_dy:.11e},{_tx:.11e},{_ty:.11e}\n")
    except Exception as _dump_exc:
        for _partial in _dumps:
            try:
                if Path(_partial).is_file() and len(Path(_partial).read_text().splitlines()) <= 1:
                    Path(_partial).unlink()
            except OSError:
                pass
        print(f"[fsi-fluid per-level dump] level {LEVEL} dump failed: {_dump_exc!r}. exports.json is "
              f"still written, so the coupling continues, but this level has no field file to hand "
              f"in. Fix the names the dump reads and run this level again.")

    out = {
        "field_name": "traction_on_structure",
        "n_points": int(len(x_iface)),
        "coordinates": ref_coords.tolist(),
        "values": traction.tolist(),
        # `normal_fluxes` is the SAME traction expressed w.r.t. THIS
        # participant's own outward normal, which is the convention the driver's
        # conservation check requires: the two participants' normals are
        # anti-parallel, so the two exported flux fields must SUM to zero.
        "normal_fluxes": (-traction).tolist(),
        "meta": {
            "sign_convention": "values = sigma_f . n_s (load ON the structure); "
                               "normal_fluxes = sigma_f . n_f (own outward normal)",
            "net_force": [float(fx), float(fy)],
            "mesh_moved": bool(MOVE_MESH),
            "dt": float(DT),
            "newton_iterations": int(nit),
            # what this participant ACTUALLY imposed on the interface, at its own
            # interface nodes — the kinematic-continuity check compares this
            # against the structure's displacement and so measures the transfer
            # in the structure->fluid direction rather than assuming it
            "displacement_imposed": d_iface.tolist(),
            "p_inlet_mean": float(
                fem.assemble_scalar(fem.form(ph * ds(1)))
                / fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ds(1)))),
        },
    }
    Path("exports.json").write_text(json.dumps(out, indent=2))
    print(f"[fluid] newton={nit} net_force=({fx:.6e},{fy:.6e}) "
          f"|d_iface|max={np.max(np.abs(d_iface)):.4e}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
