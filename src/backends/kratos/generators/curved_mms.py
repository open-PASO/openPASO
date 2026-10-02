"""Kratos curved-geometry MMS generators and knowledge.

Family: steady diffusion (Poisson) on a 2D annulus meshed with Gmsh,
with a smooth polar-friendly manufactured solution

    u*(r, theta) = A + B*ln(r) + C*r^m*cos(m*theta) + D*r^p

on r_inner <= r <= r_outer (r = 0 excluded, so ln(r) and r^p are smooth).
The first three terms are harmonic; only the D*r^p term sources the PDE:

    -kappa * lap(u*) = -kappa * D * p^2 * r^(p-2) =: f      (sympy-verified)

Dirichlet u = u* is imposed on both boundary circles, so the exact
solution of the BVP is u* itself and the P1 L2 error is expected to
converge at the theoretical order 2.

The generated script genuinely exercises Kratos: it builds a ModelPart
from the Gmsh mesh and solves with the ConvectionDiffusionApplication
LaplacianElement2D3N via ResidualBasedLinearStrategy.
"""

_DEFAULTS = {
    "r_inner": 0.5,        # inner radius (> 0)
    "r_outer": 1.0,        # outer radius (> r_inner)
    "coeff_a": 1.0,        # A  (constant term)
    "coeff_b": 1.0,        # B  (ln(r) term, harmonic)
    "coeff_c": 1.0,        # C  (r^m cos(m*theta) term, harmonic)
    "coeff_d": 1.0,        # D  (r^p term -> sources f)
    "mode_m": 2,           # angular mode number (integer >= 1)
    "radial_p": 3.0,       # radial exponent p (any real; p=2 gives constant f)
    "conductivity": 1.0,   # kappa (> 0)
    "mesh_size": 0.1,      # Gmsh target element size h (> 0)
    "msh_file": "",        # optional pre-built Gmsh .msh path ("" = mesh in-process)
}

_COEFF_KEYS = ("coeff_a", "coeff_b", "coeff_c", "coeff_d")


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_parameters(params: dict) -> list[str]:
    """Validate curved-MMS parameters; return a list of error strings ([] = OK)."""
    errors: list[str] = []
    p = dict(_DEFAULTS)
    p.update(params or {})

    for key in ("r_inner", "r_outer", "radial_p", "conductivity", "mesh_size") + _COEFF_KEYS:
        if not _is_num(p[key]):
            errors.append(f"{key} must be numeric, got {p[key]!r}")
    if errors:
        return errors  # cannot do range checks on non-numeric values

    if p["r_inner"] <= 0.0:
        errors.append(f"r_inner must be > 0 (r=0 is excluded from the annulus), got {p['r_inner']}")
    if p["r_outer"] <= p["r_inner"]:
        errors.append(f"r_outer must be > r_inner, got r_outer={p['r_outer']} <= r_inner={p['r_inner']}")
    m = p["mode_m"]
    if not isinstance(m, int) or isinstance(m, bool) or m < 1:
        errors.append(f"mode_m must be an integer >= 1, got {m!r}")
    if p["conductivity"] <= 0.0:
        errors.append(f"conductivity must be > 0, got {p['conductivity']}")
    if p["mesh_size"] <= 0.0:
        errors.append(f"mesh_size must be > 0, got {p['mesh_size']}")
    if not isinstance(p["msh_file"], str):
        errors.append(f"msh_file must be a string path (or '' to mesh in-process), got {p['msh_file']!r}")
    return errors


def _curved_mms_annulus_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Steady diffusion MMS on a Gmsh-meshed annulus — Kratos ConvectionDiffusionApplication.

    MESHING IS AGENT-DRIVEN: the eval-time agent controls the discretization.
    Either (a) set mesh_size h and let the script mesh the annulus with the
    Gmsh Python API in-process (it also writes annulus_mms.msh for reuse), or
    (b) pass msh_file pointing to a .msh the agent built itself (any conforming
    P1 triangle mesh of the same annulus); the script then gmsh.open()s it and
    uses the identical extraction path. Run a sequence of halved mesh_size
    values and fit the L2_ERROR lines to measure the convergence order
    (theoretical L2 order for P1 triangles: 2).

    NOTE: run the generated script with an interpreter that imports Kratos.
    RE-MEASURED 2026-08-18: the repo venv
    ({PYTHON}, 3.12) now
    imports Kratos 10.3 with ConvectionDiffusion, StructuralMechanics,
    FluidDynamics AND gmsh; /mnt/kratos-tier2/kv/bin/python (3.12) imports
    Kratos 10.4.3 with the same applications but NO gmsh; /usr/bin/python3
    (3.8) does NOT import Kratos at all. Always probe rather than trust a
    recorded path — see KNOWLEDGE['curved_mms'].
    """
    errs = validate_parameters(params)
    if errs:
        raise ValueError("Invalid curved_mms parameters: " + "; ".join(errs))
    p = dict(_DEFAULTS)
    p.update(params or {})
    r_i = float(p["r_inner"])
    r_o = float(p["r_outer"])
    A = float(p["coeff_a"])
    B = float(p["coeff_b"])
    C = float(p["coeff_c"])
    D = float(p["coeff_d"])
    m = int(p["mode_m"])
    pexp = float(p["radial_p"])
    kappa = float(p["conductivity"])
    h = float(p["mesh_size"])
    msh_file = p["msh_file"]
    return f'''\
"""Curved-geometry MMS: -kappa*lap(u) = f on annulus {r_i} <= r <= {r_o} — Kratos ConvectionDiffusion.

Manufactured solution u*(r,theta) = A + B*ln(r) + C*r^m*cos(m*theta) + D*r^p
(first three terms harmonic); exact source f = -kappa*D*p^2*r^(p-2), sympy-verified:
    lap(u*) = u*_rr + u*_r/r + u*_tt/r^2 = D*p^2*r^(p-2).
Dirichlet u = u* on both circles => exact BVP solution is u*.
Expected P1 L2 convergence order: 2. Meshing is agent-driven (mesh_size / MSH_FILE).
"""
import json
import math
from collections import Counter

import numpy as np
import gmsh
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401 — registers LaplacianElement2D3N

R_INNER = {r_i}
R_OUTER = {r_o}
COEFF_A = {A}
COEFF_B = {B}
COEFF_C = {C}
COEFF_D = {D}
MODE_M = {m}
RADIAL_P = {pexp}
KAPPA = {kappa}
MESH_SIZE = {h}
MSH_FILE = r"{msh_file}"  # "" -> mesh the annulus in-process with Gmsh
import os
MUTATE = os.environ.get("SURVEY_MUTATE", "") == "1"   # planted failure: a source 10% off


def u_exact(x, y):
    r = math.hypot(x, y)
    th = math.atan2(y, x)
    return (COEFF_A + COEFF_B * math.log(r)
            + COEFF_C * r**MODE_M * math.cos(MODE_M * th)
            + COEFF_D * r**RADIAL_P)


def f_source(x, y):
    r = math.hypot(x, y)
    return -KAPPA * COEFF_D * RADIAL_P**2 * r**(RADIAL_P - 2.0)


def solve_at(MESH_SIZE, MSH_FILE):
    """Mesh the annulus (or open MSH_FILE), solve with Kratos, return
    (n_nodes, n_elements, l2_error, u, xy, conn). Called twice: the verdict
    needs an order, and an order needs two meshes."""
    # ---------------- Gmsh mesh (agent-driven) ----------------
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if MSH_FILE:
        gmsh.open(MSH_FILE)
    else:
        gmsh.model.add("annulus_mms")
        disk_outer = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, R_OUTER, R_OUTER)
        disk_inner = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, R_INNER, R_INNER)
        gmsh.model.occ.cut([(2, disk_outer)], [(2, disk_inner)])
        gmsh.model.occ.synchronize()
        surfs = [t for _d, t in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, surfs, name="domain")
        inner_curves, outer_curves = [], []
        for _d, t in gmsh.model.getEntities(1):
            xmin, ymin, _z0, xmax, ymax, _z1 = gmsh.model.getBoundingBox(1, t)
            rad = 0.5 * max(xmax - xmin, ymax - ymin)
            (outer_curves if abs(rad - R_OUTER) < abs(rad - R_INNER) else inner_curves).append(t)
        gmsh.model.addPhysicalGroup(1, inner_curves, name="inner")
        gmsh.model.addPhysicalGroup(1, outer_curves, name="outer")
        gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE)
        gmsh.model.mesh.generate(2)
        gmsh.write("annulus_mms.msh")

    node_tags, coords_flat, _pc = gmsh.model.mesh.getNodes()
    coords_flat = np.asarray(coords_flat, dtype=float).reshape(-1, 3)
    tag2row = {{int(t): i for i, t in enumerate(node_tags)}}
    tris = None
    for etype, conn_flat in zip(*gmsh.model.mesh.getElements(2)[::2]):
        if int(etype) == 2:  # 3-node triangles
            tris = np.asarray(conn_flat, dtype=int).reshape(-1, 3)
    gmsh.finalize()
    if tris is None:
        raise RuntimeError("Gmsh mesh contains no 3-node triangles")

    # Renumber to contiguous 1..N (Gmsh tags are not contiguous) and enforce CCW
    used = sorted({{int(t) for tri in tris for t in tri}})
    gid2kid = {{g: i + 1 for i, g in enumerate(used)}}
    xy = np.array([coords_flat[tag2row[g]][:2] for g in used])
    conn = []
    for tri in tris:
        n = [gid2kid[int(t)] for t in tri]
        pts = xy[[i - 1 for i in n]]
        area2 = ((pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1])
                 - (pts[2, 0] - pts[0, 0]) * (pts[1, 1] - pts[0, 1]))
        if area2 < 0.0:
            n = [n[0], n[2], n[1]]
        conn.append(n)

    # Tolerance-free boundary detection: edges adjacent to exactly one triangle
    edge_count = Counter()
    for n in conn:
        for a, b in ((n[0], n[1]), (n[1], n[2]), (n[2], n[0])):
            edge_count[(min(a, b), max(a, b))] += 1
    boundary_nodes = sorted({{v for e, c in edge_count.items() if c == 1 for v in e}})

    # ---------------- Kratos ModelPart + real CDA solve ----------------
    model = KM.Model()
    mp = model.CreateModelPart("Thermal")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    mp.AddNodalSolutionStepVariable(KM.TEMPERATURE)   # before CreateNewNode!
    mp.AddNodalSolutionStepVariable(KM.HEAT_FLUX)
    mp.AddNodalSolutionStepVariable(KM.CONDUCTIVITY)
    mp.AddNodalSolutionStepVariable(KM.REACTION_FLUX)
    mp.SetBufferSize(1)

    settings = KM.ConvectionDiffusionSettings()
    settings.SetUnknownVariable(KM.TEMPERATURE)
    settings.SetDiffusionVariable(KM.CONDUCTIVITY)
    settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
    settings.SetReactionVariable(KM.REACTION_FLUX)
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)

    props = mp.CreateNewProperties(1)
    props.SetValue(KM.CONDUCTIVITY, KAPPA)  # LaplacianElement reads the NODAL value; kept for tooling

    for i, (x, y) in enumerate(xy, start=1):
        mp.CreateNewNode(i, float(x), float(y), 0.0)
    for eid, n in enumerate(conn, start=1):
        mp.CreateNewElement("LaplacianElement2D3N", eid, n, props)  # string factory only

    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)

    for node in mp.Nodes:
        node.SetSolutionStepValue(KM.CONDUCTIVITY, KAPPA)  # element reads diffusivity nodally
        node.SetSolutionStepValue(KM.HEAT_FLUX, f_source(node.X, node.Y) * (1.10 if MUTATE else 1.0))
    for i in boundary_nodes:
        node = mp.GetNode(i)
        node.SetSolutionStepValue(KM.TEMPERATURE, u_exact(node.X, node.Y))
        node.Fix(KM.TEMPERATURE)

    from KratosMultiphysics import python_linear_solver_factory
    try:
        lin_solver = python_linear_solver_factory.ConstructSolver(
            KM.Parameters('{{"solver_type": "sparse_lu"}}'))          # LinearSolversApplication
    except Exception as _e:
        # Both are direct LU solves, so the ANSWER should not change -- but say
        # that the substitution happened. A fallback the reader cannot see is
        # indistinguishable from the thing it replaced, and a silent swap of a
        # solver, a mesh or a domain is how a different problem gets reported as
        # this one. If this line prints, LinearSolversApplication is not
        # available on your install.
        print(f"NOTE: sparse_lu unavailable ({{type(_e).__name__}}); "
              f"falling back to the core skyline_lu_factorization. "
              f"Both are direct solves; the result should be unaffected.")
        lin_solver = python_linear_solver_factory.ConstructSolver(
            KM.Parameters('{{"solver_type": "skyline_lu_factorization"}}'))  # core fallback
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, lin_solver, False, False, False, False)
    strategy.SetEchoLevel(0)
    strategy.Initialize()
    strategy.Solve()

    n_nodes = len(used)
    u = np.array([mp.GetNode(i).GetSolutionStepValue(KM.TEMPERATURE)
                  for i in range(1, n_nodes + 1)])

    # ---------------- L2 error (mid-edge quadrature, exact for quadratics) ----------------
    err2 = 0.0
    for n in conn:
        pts = xy[[i - 1 for i in n]]
        uh = u[[i - 1 for i in n]]
        area = 0.5 * abs((pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1])
                         - (pts[2, 0] - pts[0, 0]) * (pts[1, 1] - pts[0, 1]))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            mx, my = 0.5 * (pts[a] + pts[b])
            uh_mid = 0.5 * (uh[a] + uh[b])
            err2 += area / 3.0 * (uh_mid - u_exact(mx, my))**2
    l2_error = math.sqrt(err2)
    return len(used), len(conn), l2_error, u, xy, conn



n_nodes, n_elements, l2_error, u, xy, conn = solve_at(MESH_SIZE, MSH_FILE)
print(f"MESH_SIZE = {{MESH_SIZE}}")
print(f"N_NODES = {{n_nodes}}")
print(f"N_ELEMENTS = {{n_elements}}")
print(f"L2_ERROR = {{l2_error:.12e}}")
# ---------------- the ladder: the same annulus at half the mesh size ----------------
# The L2 order between the two levels is judged against the P1 prediction, 2, as a
# VERDICT line in the coverage harness's grammar (scripts/coverage_harness/definitions.py).
# An agent-supplied MSH_FILE has no in-process half-size sibling, so it gets no verdict.
if MSH_FILE:
    order = float("nan")
    print("ORDER = nan  (an agent-supplied mesh has no half-size sibling here; no verdict)")
else:
    _, _, l2_error_half, _, _, _ = solve_at(0.5 * MESH_SIZE, "")
    order = (math.log(l2_error / l2_error_half) / math.log(2.0)
             if l2_error > 0 and l2_error_half > 0 else float("nan"))
    print(f"L2_ERROR_HALF = {{l2_error_half:.12e}}")
    print(f"ORDER = {{order:.4f}}  (P1 triangles: 2 expected)")
    print(f"VERDICT kratos curved_mms mms_order ref=2 got={{order:.6e}} tol=1.500000e-01 "
          + ("PASS" if abs(order - 2.0) <= 0.15 else "FAIL"))
if MUTATE:
    print("[MUTATED: the source term is 10% off -- FAIL on the order verdict is the control working]")

import meshio
pts3 = np.hstack([xy, np.zeros((n_nodes, 1))])
cells = np.array([[i - 1 for i in n] for n in conn])
meshio.Mesh(pts3, [("triangle", cells)],
            point_data={{"temperature": u,
                         "u_exact": np.array([u_exact(x, y) for x, y in xy])}}
            ).write("result.vtu")

summary = {{"l2_error": float(l2_error), "mesh_size": float(MESH_SIZE),
            "measured_l2_order": float(order),
            "n_nodes": int(n_nodes), "n_elements": int(n_elements),
            "expected_l2_order": 2.0, "element_type": "LaplacianElement2D3N (P1 tri)",
            "r_inner": float(R_INNER), "r_outer": float(R_OUTER)}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Kratos curved-MMS annulus solve complete.")
'''


KNOWLEDGE = {
    "curved_mms": {
        "description": ("Curved-geometry manufactured-solution family: steady diffusion on a "
                        "Gmsh-meshed 2D annulus, solved with the real ConvectionDiffusionApplication "
                        "stationary path (LaplacianElement2D3N + ResidualBasedLinearStrategy). "
                        "u*(r,theta) = A + B*ln(r) + C*r^m*cos(m*theta) + D*r^p; the ln and modal "
                        "terms are harmonic, so f = -kappa*D*p^2*r^(p-2) exactly (closed form "
                        "cross-checked with sympy). Theoretical L2 order for P1 triangles: 2."),
        "application": "ConvectionDiffusionApplication (+ LinearSolversApplication for sparse_lu)",
        "elements": ["LaplacianElement2D3N (string factory only, see poisson pitfalls)"],
        "variables": {
            "unknown": "TEMPERATURE",
            "diffusion": "CONDUCTIVITY (read NODALLY by LaplacianElement — see pitfalls)",
            "source": "HEAT_FLUX (nodal)",
            "reaction": "REACTION_FLUX",
        },
        "theoretical_l2_order": 2.0,
        "how_to_grade_convergence": (
            "Run the emitted script at a sequence of mesh sizes (e.g. h, h/2, h/4) "
            "and fit log(L2 error) against log(h); the slope is the observed order. "
            "Both the errors and the order are OUTPUTS of your run — measure them, "
            "do not assume them. Compare the slope against theoretical_l2_order to "
            "decide whether the deck is correct."),
        # NO MEASURED RESULT HERE. This field held our own error table and the
        # observed orders from a recorded run. That is the answer to a
        # convergence study, sitting inside the tool the study is meant to
        # evaluate — an agent could read the result instead of computing it,
        # and convergence order is exactly what such a study measures. Removed
        # 2026-08-06 by the contamination merge gate.
        #
        # What survives is the falsifiable statement: the discretisation is
        # expected to reach its theoretical order once the mesh resolves the
        # geometry. Whether it does on any given draw is for the run to show.
        "verified_convergence": (
            "Exercised on this install (Kratos 10.4 under the system python3, "
            "gmsh OCC annulus) over a refinement sequence; the L2 order "
            "approaches the theoretical value above once the isoparametric "
            "elements resolve the curved boundary. Measure it on your own "
            "sequence — a plateau below the theoretical order is the signal "
            "that the geometry, not the solver, is limiting you. "
            "REPRODUCED 2026-08-03 on Kratos 10.4.0, and again 2026-08-09: "
            "this family is DETERMINISTIC — re-running the same parameter "
            "draw at the same mesh_size returns an identical L2_ERROR to "
            "every printed digit (checked by repeating a run in place and by "
            "repeating it in a fresh working directory), because the Gmsh OCC "
            "geometry is rebuilt from the same construction each time. So a "
            "series that differs between two of your runs means the draw, the "
            "mesh size or the interpreter changed — it is never run-to-run "
            "noise, and there is nothing to gain by averaging repeats. "
            "The SAME check is how the draw trap was found, and it is the "
            "reason no error table is quoted above: the parameter draw used "
            "while this family was developed (r_inner=0.6, r_outer=1.7, "
            "coeff_a=0.7, coeff_b=-1.3, coeff_c=0.8, coeff_d=0.35, mode_m=3, "
            "radial_p=3, conductivity=2.5) is NOT the template default draw "
            "(r_inner=0.5, r_outer=1.0, coeff_a..d=1, mode_m=2, radial_p=3, "
            "conductivity=1), and the two produce visibly different error "
            "CONSTANTS while sharing the same theoretical order. Reference "
            "errors quoted for one draw therefore say nothing about the "
            "other: re-derive your own reference for the draw you actually "
            "run, and never compare a run against a number that came from a "
            "different parameter set."),
        "pitfalls": [
            '[Environment] RE-MEASURED 2026-08-18, and the earlier entry is now WRONG — '
            'the host changed under it. The LESSON is per-host, so do not carry any '
            'interpreter path from this text: on one machine the project venv '
            'imported Kratos 10.3 with ConvectionDiffusion, StructuralMechanics, '
            'FluidDynamics and gmsh, a second interpreter imported Kratos 10.4.3 '
            'with those applications but WITHOUT gmsh, and the system python3 did '
            'not import Kratos at all. Take the interpreter from '
            'discover(query=\'list\'). ALWAYS PROBE '
            'the interpreter you are about to use with `import KratosMultiphysics` '
            'rather than trusting any path recorded here, including this one: an '
            'entry about the installed environment is true on the day it is measured '
            'and silently rots afterwards. The historical GLIBC failure, kept because '
            'its signature is distinctive: from a venv whose wheel is incompatible the '
            'import dies inside the DYNAMIC LINKER, before any Kratos code runs, so no '
            'Kratos diagnostic is produced at all: an ImportError naming '
            '/lib/x86_64-linux-gnu/libc.so.6 and a GLIBC_2.32 version that is not found, '
            'required by the wheel\'s Kratos.cpython-312-x86_64-linux-gnu.so. Kratos\'s '
            'own except-branch then prints \'Unable to find KratosCore.\' followed by a '
            'sentence naming the library-path variable, which is substituted at runtime. '
            'Run generated Kratos scripts with one of the two working interpreters, and '
            'mind that they do not carry the same extras: gmsh imports under '
            '/usr/bin/python3 and in the repo .venv, but is NOT installed in the Tier-2 '
            'environment (checked 2026-08-09). '
            'Application availability under /usr/bin/python3 is NOT fixed — every '
            'Kratos application is a separate pip wheel, so ALWAYS probe with '
            '`import KratosMultiphysics.X` instead of trusting any list. Measured on this '
            'host 2026-08-03: 26 applications importable (Core, LinearSolvers, '
            'StructuralMechanics, ContactStructuralMechanics, ConstitutiveLaws, '
            'ConvectionDiffusion, FluidDynamics, CoSimulation, FSI, DEM, MPM, GeoMechanics, '
            'CompressiblePotentialFlow, RANS, Rom, Iga, Poromechanics, ShallowWater, Dam, '
            'DemStructuresCoupling, CableNet, Optimization, ShapeOptimization, MeshMoving, '
            'Meshing, Mapping) after pip-installing the matching 10.4.0 wheels; before that '
            'only four were present. Two wheels install under /usr/bin/python3 but do NOT '
            'import: SwimmingDEM and Chimera. Both are stopped by the dynamic linker, '
            'which reports that libKratosSwimmingDEMCore.so and '
            'libKratosChimeraApplicationCore.so respectively cannot be opened because no '
            'such file or directory exists — the per-application Core library is simply '
            'not in the wheel, so the text comes from the loader and not from Kratos. '
            '(Verified empirically 2026-08-01; list corrected and re-measured '
            '2026-08-03 — the previous four-app list was already incomplete when written, '
            'ContactStructuralMechanicsApplication was importable too. Both import '
            'failures re-run 2026-08-09 under /usr/bin/python3; under '
            '/mnt/kratos-tier2/kv/bin/python SwimmingDEM is not installed at all and '
            'raises ModuleNotFoundError: No module named '
            '\'KratosMultiphysics.SwimmingDEMApplication\', while Chimera fails through '
            'the loader exactly as it does under 3.8.)',
            '[Numerical] LaplacianElement2D3N reads the diffusivity NODALLY via '
            'ConvectionDiffusionSettings.GetDiffusionVariable() — the CONDUCTIVITY value '
            'on the Properties object is IGNORED by this element. Confirmed by a swap '
            'test: corrupting the Properties CONDUCTIVITY while the nodal values stay '
            'correct leaves the solution unchanged, whereas corrupting the nodal '
            'CONDUCTIVITY destroys it. Signal: solution scales with the nodal value, '
            'not the property; forgetting SetSolutionStepValue(CONDUCTIVITY, ...) on '
            'nodes gives a singular or zero-diffusivity system.',
            '[Numerical] Curved boundaries with straight P1 edges: Gmsh places boundary '
            'NODES exactly on the circles, but element edges are straight chords, so the '
            'computed domain is a polygon inscribed in the annulus. The geometric '
            '(domain-approximation) error is O(h^2) — the same order as the P1 '
            'interpolation error — so the L2 convergence order 2 is PRESERVED without '
            'curved (isoparametric) elements. Signal: the measured L2 order holds at '
            'the theoretical value for the element as the mesh is refined; a plateau '
            'well below it (around order 1.5 for P1) is NOT the straight-chord geometry '
            'and points instead at misclassified boundary nodes or an inconsistent '
            'source term. (Verified 2026-08-01.)',
            '[Integration] msh -> ModelPart conversion pitfalls (all hit or guarded '
            'while building this family): (1) Gmsh node tags are NOT contiguous after '
            'OCC boolean cuts — '
            'renumber to 1..N before CreateNewNode or Kratos raises on missing node ids; '
            '(2) triangle orientation from Gmsh is not guaranteed CCW — check the signed '
            'area and flip, else element Jacobians go negative; (3) '
            'AddNodalSolutionStepVariable(TEMPERATURE/HEAT_FLUX/CONDUCTIVITY/'
            'REACTION_FLUX) must happen BEFORE the first CreateNewNode, and '
            'CONVECTION_DIFFUSION_SETTINGS must be on ProcessInfo before the solve; '
            '(4) prefer tolerance-free topological boundary detection (edges adjacent to '
            'exactly one triangle) over radius comparisons — it cannot misclassify nodes '
            'on curved boundaries; Gmsh physical groups ("inner"/"outer") remain useful '
            'when the two circles need DIFFERENT boundary conditions. Signal: (1) raises '
            '"Error: Node #<id> does not exist" (or KeyError) at CreateNewElement; (2) '
            'raises "Element found with negative Jacobian" (or NaN/absurd L2 error) at '
            'solve; (3) does NOT raise at all — an earlier wording claimed a "missing '
            'variable" or "CONVECTION_DIFFUSION_SETTINGS not defined" error at AddDof or '
            'solver init and neither string exists in Kratos. Measured 2026-08-13 on '
            'Kratos 10.4.3: with the nodal variables added but '
            'CONVECTION_DIFFUSION_SETTINGS never put on ProcessInfo, CreateNewElement '
            'and AddDof both succeed and the process SEGFAULTS inside '
            'CalculateLocalSystem — exit 139, empty stderr, no Kratos exception and no '
            'message of any kind. The identical script with the settings object on '
            'ProcessInfo exits 0. So the signal is the crash, not a diagnostic: check '
            'ProcessInfo.Has(CONVECTION_DIFFUSION_SETTINGS) yourself before solving; '
            '(4) shows up as a convergence-order drop '
            'below 2 with errors concentrated at near-boundary nodes.',
            '[Workflow] Meshing is agent-driven: the template accepts mesh_size (in-process '
            'Gmsh OCC meshing, also writes annulus_mms.msh) or msh_file (gmsh.open() an '
            'agent-built .msh; identical extraction path). Measure convergence by rerunning '
            'with halved mesh_size and fitting the machine-readable "L2_ERROR = ..." lines. '
            'Signal: a run whose stdout contains no "L2_ERROR = " line did not reach the '
            'error computation — treat it as failed, never fit an order to it.',
        ],
    },
}

GENERATORS = {
    "curved_mms_annulus_2d": _curved_mms_annulus_2d,
}
