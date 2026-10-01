"""scikit-fem STRUCTURE participant for the openPASO `couple` driver — FSI.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

Physics: plane-strain linear elasticity on a rectangular wall clamped at both
ends.  The bottom edge is the FSI interface and carries the traction handed in
by the fluid participant as a NEUMANN load.  What is exported back is the
interface displacement.

SIGN CONVENTION — the imported traction is already the load ON THIS BODY
(t = sigma_f . n_s, n_s the structure's outward normal on the interface).  It is
applied DIRECTLY, with NO further sign change.  See the fluid participant's
docstring for the derivation; re-deriving it on this side is how the sign gets
flipped twice.

The interface parametrisation is LAGRANGIAN on both sides: the exported
coordinates are the REFERENCE (undeformed) interface node positions.
"""
import logging
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
logging.basicConfig(level=logging.INFO)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
from skfem import (Basis, FacetBasis, ElementVector, ElementTriP2, MeshTri,
                   asm, condense, solve, BilinearForm, LinearForm)
from skfem.helpers import dot
from skfem.models.elasticity import linear_elasticity
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
RHO_S      = 0.0         # structure density; only used when DT > 0
DT         = 0.0         # 0.0 -> STATIC. >0 -> ONE backward-Euler step from rest,
                         # which adds rho_s/dt^2 * M to the stiffness. Pair it with
                         # the same DT in the fluid participant.
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


def make_sampler(imp, fallback, ncomp=2):
    """Return f(x_array) -> (len(x), ncomp) traction, interpolated along the
    interface parameter x.  The driver does no interpolation: non-matching
    interface meshes are handled here."""
    if not imp or not imp.get("coordinates"):
        return lambda xs: np.full(np.shape(xs) + (ncomp,), float(fallback))
    xs_src = np.asarray(imp["coordinates"], float)[:, 0]
    vals = np.asarray(imp["values"], float).reshape(len(xs_src), -1)
    order = np.argsort(xs_src)
    xs_src, vals = xs_src[order], vals[order]

    def f(xq):
        xq = np.asarray(xq, float)
        out = np.zeros(xq.shape + (ncomp,))
        for c in range(min(ncomp, vals.shape[1])):
            out[..., c] = np.interp(xq, xs_src, vals[:, c])
        return out
    return f


def main():
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    lam = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))       # plane strain
    mu = E_MOD / (2.0 * (1.0 + NU))

    m = MeshTri().init_tensor(np.linspace(0.0, LX, NXS + 1),
                              np.linspace(Y0, Y0 + HS, NYS + 1))
    e = ElementVector(ElementTriP2())
    basis = Basis(m, e)

    K = asm(linear_elasticity(lam, mu), basis)
    if DT > 0.0 and RHO_S > 0.0:
        # backward Euler from rest: d_tt ~ d/dt^2, so the effective operator is
        # K + rho_s/dt^2 * M. Lowering rho_s (or dt) is what makes the structure
        # light against the fluid it has to push, which is the added-mass regime.
        @BilinearForm
        def _mass(u, v, w):
            return dot(u, v)
        K = K + (RHO_S / DT**2) * asm(_mass, basis)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    imp = read_imports() if FEEDBACK else None
    sampler = make_sampler(imp, T_INIT, ncomp=2)

    iface_facets = m.facets_satisfying(lambda x: np.isclose(x[1], Y0))
    fb = FacetBasis(m, e, facets=iface_facets)
    xq = fb.global_coordinates().value          # (dim, nfacets, nqp)
    tq = sampler(xq[0])                         # (nfacets, nqp, 2)

    @LinearForm
    def neumann(v, w):
        return w["tx"] * v[0] + w["ty"] * v[1]

    # skfem passes extra kwargs through as quadrature-point arrays.
    f = asm(neumann, fb, tx=tq[..., 0], ty=tq[..., 1])

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    # clamped ends: every dof on x = CLAMP_X
    clamped = basis.get_dofs(
        lambda x: np.isclose(x[0], CLAMP_X[0]) | np.isclose(x[0], CLAMP_X[1]))
    d = solve(*condense(K, f, D=clamped))
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    # ── interface displacement at the interface NODES (P1 subset of P2) ─────
    nodal = basis.nodal_dofs                      # (ncomp, nvertices)
    inode = np.where(np.isclose(m.p[1], Y0))[0]
    order = np.argsort(m.p[0, inode])
    inode = inode[order]
    x_if = m.p[0, inode]
    disp = np.column_stack([d[nodal[0, inode]], d[nodal[1, inode]]])
    ref_coords = np.column_stack([x_if, np.full_like(x_if, Y0)])

    # net force actually received on the interface, for the equilibrium check
    fx = float(np.sum(asm(neumann, fb, tx=tq[..., 0], ty=0.0 * tq[..., 1])))
    fy = float(np.sum(asm(neumann, fb, tx=0.0 * tq[..., 0], ty=tq[..., 1])))

    # ── what this body ACTUALLY received on the interface ───────────────────
    # This is the traction after THIS participant's own interpolation from the
    # fluid's (different) interface points, sampled at this body's nodes. It is
    # what `normal_fluxes` carries, and what the driver's conservation check
    # therefore measures is THE FORCE TRANSFER: whether the mapping between two
    # non-matching interface discretisations conserved the interface force.
    # That is the conservation question a partitioned FSI actually has to
    # answer, and a lossy or partially-covering mapping shows up here.
    #
    # It is NOT an independent statement about the structure's stress state,
    # and it therefore cannot catch a sign convention that is wrong on BOTH
    # sides — only a reference solve can. Do not read a clean balance as more
    # than it is.
    #
    # WHY NOT THE TRACTION RECOVERED FROM THE STRUCTURE'S OWN STRESS FIELD,
    # which would be independent: on a bending structure with clamped ends it
    # does not converge, so it cannot be used as a check.
    #
    # The mechanism: where the clamped Dirichlet boundary meets the loaded face
    # there is a genuine stress singularity. The pointwise traction there
    # DIVERGES under refinement, and a nodal quadrature of it converges to the
    # wrong number.
    #
    # What that looks like if you try it, on a four-level refinement of any
    # clamped bending wall: the applied net force settles immediately and agrees
    # with the fluid's to four digits, while the recovered net climbs level by
    # level and is still less than half the applied one at the finest mesh, and
    # the recovered POINTWISE maximum grows monotonically — ending orders of
    # magnitude above the applied maximum rather than approaching it. Both
    # behaviours are correct; they are what a singularity does. A conservation
    # check built on the recovered traction would therefore fault a coupling
    # that is right, which is why the check above uses the APPLIED traction.
    t_applied = sampler(x_if)

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
        # w.r.t. THIS body's own outward normal n_s, so that this and the
        # fluid's `normal_fluxes` (w.r.t. n_f = -n_s) must SUM to zero.
        "normal_fluxes": t_applied.tolist(),
        "meta": {
            "net_force_received": [fx, fy],
            "feedback": bool(FEEDBACK),
            "max_abs_disp": [float(np.max(np.abs(disp[:, 0]))),
                             float(np.max(np.abs(disp[:, 1])))],
            "n_dofs": int(K.shape[0]),
            "dt": float(DT), "rho_s": float(RHO_S),
        },
    }
    Path("exports.json").write_text(json.dumps(out, indent=2))
    print(f"[solid] recv_force=({fx:.6e},{fy:.6e}) "
          f"max|dy|={np.max(np.abs(disp[:, 1])):.6e}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
