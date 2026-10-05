"""Kratos MPM (Material Point Method) generators and knowledge.

THE GENERATOR IN THIS FILE DRIVES MPMApplication.

It did not until 2026-08-09. `_mpm_2d_kratos` used to emit a standalone
explicit MPM written in numpy: its own background grid, its own shape
functions, its own USL update loop. The string "KratosMultiphysics" did not
occur anywhere in what it wrote. It ran and it computed something — 256
material points, 5000 steps, ~3 minutes, a result.vtu and a
results_summary.json reporting max|v| = 3.04 m/s for a body falling under
gravity — but nothing in that path was Kratos, and the KNOWLEDGE below IS about
MPMApplication. An agent reading knowledge('mpm') and an agent running the
template were being told about two different codes.

The rewrite drives MPMApplication directly: it writes the two mdpa files, the
materials json and the ProjectParameters, then calls
`MpmAnalysis(model, parameters).Run()` in-process and reads MP_DISPLACEMENT,
MP_VELOCITY and MP_MASS back off the material points. Every requirement below
was already documented with its failure signature in KNOWLEDGE["mpm"]
["pitfalls"], so the rewrite is that specification executed:

    two mdpa files, the grid under its own import key ..... pitfall 1
    MATERIAL_POINTS_PER_ELEMENT in the MATERIALS json ..... pitfall 2
    a count from the GRID geometry's allowed set ......... pitfall 3
    materials addressed via Initial_MPM_Material ......... pitfall 5
    Dirichlet data on Background_Grid, not the body ...... pitfall 6
    solver_type a short label, not a class name .......... pitfall 7
    scheme_type partitioned by solver ................... pitfall 8
    the fully-qualified constitutive-law name ........... pitfall 9
    a grid that encloses the whole trajectory ........... pitfall 10
    gravity as an opt-in process ........................ pitfall 11

Executed 2026-08-09 on the 28-application build at /mnt/kratos-tier2/kv against
MPMApplication 10.4.3: rc=0 in 29 s, 80 material points, total material-point
mass 138.888888927 exactly equal to the seeded mass (nothing erased), peak
MP_DISPLACEMENT 0.120 m on a 0.4 m column — 30% strain, which is the
large-deformation regime the template claims.

WHY NOTHING CAUGHT THE OLD ONE, measured 2026-08-07:

  * KratosBackend.validate_input carries an "honesty guard" whose closing check
    is 'neither uses KratosMultiphysics nor runs a solve'. The template passed
    it on the strength of ONE line: `from scipy.sparse.linalg import spsolve`.
    spsolve was never called, and neither was lil_matrix. The guard was being
    held green by an unused import — a guard satisfied by a symbol's PRESENCE
    rather than its USE, which is the same defect shape as an expectation
    satisfied by a word the fixture prints itself. Fixed in the same pass: the
    guard now strips import lines and looks for a CALL. Measured over all 21
    Kratos templates, mpm was the only one whose marker was not a call.

  * Seven of the twenty-one Kratos templates are standalone in the sense of not
    importing Kratos (cosimulation, heat, heat_transient, linear_elasticity,
    mpm, shape_optimization, structural_dynamics). For six of them the guard's
    own comment says this is deliberate — the "scipy/numpy assemble-and-solve
    pattern" — and those six do call a real solve (spsolve or factorized).
    mpm was the only one of the seven that called neither. It is no longer one
    of them: it imports KratosMultiphysics and runs MPMApplication.
"""

# Material-point counts MPMApplication 10.4.2 and 10.4.3 accept, keyed by the
# geometry of the BACKGROUND GRID — not of the body. On those releases anything
# else is rejected by name (see pitfall 3). Kratos 10.3.0 and 10.4.0 do not use
# this table: they accept 1, 4, 9 and 16 per quadrilateral and only WARN on any
# other count, seeding the default 4 per element instead. Measured on 10.3.0:
# 25 per quadrilateral ran rc 0 with 'Available options are: 1, 4, 9, 16.' and
# 'The default number of material points: 4 is currently assumed.', 80 material
# points on 20 body elements. The generator builds a quadrilateral grid, so it
# validates against the quadrilateral counts BOTH release lines accept.
_MATERIAL_POINTS_PER_ELEMENT = {
    "Triangular": (1, 3, 4, 6, 12),
    "Quadrilateral": (1, 4, 9, 16, 25),
    "Tetrahedral": (1, 4, 8, 14, 24),
    "Hexahedral": (1, 8, 27, 64, 125),
}
_QUADRILATERAL_COUNTS_BEFORE_10_4_2 = (1, 4, 9, 16)


def _mpm_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable program. All parameter defaults are placeholders.

    Material Point Method for large-deformation solid mechanics, on
    KratosMultiphysics.MPMApplication.

    The defaults are a soft neo-Hookean column settling under its own weight on
    a fixed floor: 100 implicit Newmark steps, ~30 s, ~30% peak strain. They are
    placeholders like every other template's — the point is that the deck they
    produce is one MPMApplication accepts and runs.
    """
    n_cells_x = params.get("n_cells_x", 12)
    n_cells_y = params.get("n_cells_y", 12)
    # The Kratos key is MATERIAL_POINTS_PER_ELEMENT. `particles_per_cell` is
    # accepted as an alias because that is what this generator used to call it,
    # and PARTICLES_PER_ELEMENT is still Kratos's own deprecated spelling.
    mppe = int(params.get("material_points_per_element",
                          params.get("particles_per_cell", 4)))
    allowed = tuple(n for n in _MATERIAL_POINTS_PER_ELEMENT["Quadrilateral"]
                    if n in _QUADRILATERAL_COUNTS_BEFORE_10_4_2)
    if mppe not in allowed:
        raise ValueError(
            f"material_points_per_element={mppe} is not available for "
            f"Quadrilateral elements on every Kratos release, and the "
            f"background grid this template builds is quadrilateral. "
            f"Available options are: "
            f"{', '.join(str(a) for a in allowed[:-1])} and {allowed[-1]}. "
            f"Kratos 10.4.2 and 10.4.3 also accept 25 and refuse any other "
            f"count at solver Initialize; 10.3.0 and 10.4.0 only warn and "
            f"seed 4 per element instead. Refusing here means the deck is "
            f"never written.")
    E = params.get("E", 1.0e4)
    nu = params.get("nu", 0.3)
    density = params.get("density", 1000.0)
    gravity = params.get("gravity", -9.81)
    dt = params.get("dt", 2e-3)
    T_end = params.get("T_end", 0.2)
    domain_x = params.get("domain_x", 1.0)
    domain_y = params.get("domain_y", 1.0)
    body_x0 = params.get("body_x0", 0.3)
    body_x1 = params.get("body_x1", 0.7)
    body_y0 = params.get("body_y0", 0.0)
    body_y1 = params.get("body_y1", 0.4)
    # Fully-qualified registered name. The short family label is the single
    # most common MPM setup error (pitfall 9).
    law = params.get("constitutive_law", "HyperElasticNeoHookeanPlaneStrain2DLaw")

    header = f'''\
"""Material Point Method — large-deformation solid — Kratos MPMApplication"""
import json

import KratosMultiphysics as KM
import KratosMultiphysics.MPMApplication as KratosMPM
from KratosMultiphysics.MPMApplication.mpm_analysis import MpmAnalysis

# Grid parameters — set for your problem.
#
# The BACKGROUND GRID must enclose the whole TRAJECTORY of the body, not just
# its starting position. Material points that leave the grid are ERASED and the
# mass they carry leaves the simulation with them; the receipt is two log lines
# ("Search Element for Material Point: N is failed" then "MaterialPointErase-
# Process: N particle elements have been erased"), not an error, so a partially
# escaped body silently loses mass. The mass check at the bottom of this script
# is what turns that into something you can see.
n_cells_x, n_cells_y = {n_cells_x}, {n_cells_y}
domain_x, domain_y = {domain_x}, {domain_y}
body_x0, body_x1 = {body_x0}, {body_x1}
body_y0, body_y1 = {body_y0}, {body_y1}

# Material parameters — set for your problem
E, nu, density = {E}, {nu}, {density}
gravity = {gravity}
dt, T_end = {dt}, {T_end}

# Drawn from the GRID geometry's allowed set: this grid is QUADRILATERAL.
# Kratos 10.4.2 and 10.4.3 accept 1, 4, 9, 16 and 25 and reject any other
# value at solver Initialize with a message that names the geometry. Kratos
# 10.3.0 and 10.4.0 accept 1, 4, 9 and 16 and only WARN on any other value,
# seeding 4 per element instead.
material_points_per_element = {mppe}

# The fully-qualified registered law name. "LinearElasticPlaneStrain2DLaw" and
# other short family labels are not registered components.
constitutive_law = "{law}"
'''
    return header + _MPM_BODY


# Everything below is parameter-free, so it is kept out of the f-string above:
# an MPM deck is mostly nested JSON and brace-doubling it would make the one
# artefact a reader needs to check unreadable.
_MPM_BODY = '''
dx = domain_x / n_cells_x
dy = domain_y / n_cells_y
nx = n_cells_x + 1


def node_id(i, j):
    return j * nx + i + 1


def _nodes_block(ids):
    out = []
    for k in sorted(ids):
        j, i = divmod(k - 1, nx)
        out.append(f"{k} {i * dx:.10g} {j * dy:.10g} 0.0")
    return "\\n".join(out)


# ── background grid ────────────────────────────────────────────────────────
# Plain FEM quads over the whole domain. The grid mdpa uses ORDINARY element
# names (Element2D4N); only the body mdpa carries MPM* names.
grid_elements, floor_nodes, grid_nodes = [], set(), set()
eid = 0
for cj in range(n_cells_y):
    for ci in range(n_cells_x):
        eid += 1
        n1, n2 = node_id(ci, cj), node_id(ci + 1, cj)
        n3, n4 = node_id(ci + 1, cj + 1), node_id(ci, cj + 1)
        grid_elements.append(f"{eid} 0 {n1} {n2} {n3} {n4}")
        grid_nodes.update((n1, n2, n3, n4))
        if cj == 0:
            floor_nodes.update((n1, n2))

grid_mdpa = (
    "Begin Properties 0\\nEnd Properties\\n"
    "Begin Nodes\\n" + _nodes_block(grid_nodes) + "\\nEnd Nodes\\n"
    "Begin Elements Element2D4N\\n" + "\\n".join(grid_elements)
    + "\\nEnd Elements\\n"
    "Begin SubModelPart Parts_Grid\\n  Begin SubModelPartNodes\\n"
    + "\\n".join(str(n) for n in sorted(grid_nodes))
    + "\\n  End SubModelPartNodes\\n  Begin SubModelPartElements\\n"
    + "\\n".join(str(e + 1) for e in range(len(grid_elements)))
    + "\\n  End SubModelPartElements\\nEnd SubModelPart\\n"
    "Begin SubModelPart DISPLACEMENT_floor\\n  Begin SubModelPartNodes\\n"
    + "\\n".join(str(n) for n in sorted(floor_nodes))
    + "\\n  End SubModelPartNodes\\nEnd SubModelPart\\n")

# ── body ───────────────────────────────────────────────────────────────────
# MPM elements on the cells whose centre lies inside the body region. They
# share the grid's nodes; the runtime seeds material points inside them.
body_elements, body_nodes = [], set()
bid = 1000
for cj in range(n_cells_y):
    for ci in range(n_cells_x):
        cx, cy = (ci + 0.5) * dx, (cj + 0.5) * dy
        if not (body_x0 <= cx <= body_x1 and body_y0 <= cy <= body_y1):
            continue
        bid += 1
        n1, n2 = node_id(ci, cj), node_id(ci + 1, cj)
        n3, n4 = node_id(ci + 1, cj + 1), node_id(ci, cj + 1)
        body_elements.append(f"{bid} 0 {n1} {n2} {n3} {n4}")
        body_nodes.update((n1, n2, n3, n4))
if not body_elements:
    raise SystemExit(
        "no grid cell centre lies inside the body region — the body would be "
        "empty and the run would abort with 'No degrees of freedom in model "
        "part: MPM_Material'. Widen the body box or refine the grid.")

body_mdpa = (
    "Begin Properties 0\\nEnd Properties\\n"
    "Begin Nodes\\n" + _nodes_block(body_nodes) + "\\nEnd Nodes\\n"
    "Begin Elements MPMUpdatedLagrangian2D4N\\n" + "\\n".join(body_elements)
    + "\\nEnd Elements\\n"
    "Begin SubModelPart Parts_Body\\n  Begin SubModelPartNodes\\n"
    + "\\n".join(str(n) for n in sorted(body_nodes))
    + "\\n  End SubModelPartNodes\\n  Begin SubModelPartElements\\n"
    + "\\n".join(e.split()[0] for e in body_elements)
    + "\\n  End SubModelPartElements\\nEnd SubModelPart\\n")

with open("grid.mdpa", "w") as _f:
    _f.write(grid_mdpa)
with open("body.mdpa", "w") as _f:
    _f.write(body_mdpa)

# MATERIAL_POINTS_PER_ELEMENT lives HERE, in the materials json under
# properties[i].Material.Variables — not in ProjectParameters. On Kratos
# 10.4.2 and 10.4.3 its absence is a hard error; 10.3.0 and 10.4.0 only warn
# and assume 1 material point per element.
#
# The body is addressed through Initial_MPM_Material.<SubModelPart>: at
# materials-reading time the body sub model parts only exist under Initial_.
with open("ParticleMaterials.json", "w") as _f:
    json.dump({"properties": [{
        "model_part_name": "Initial_MPM_Material.Parts_Body",
        "properties_id": 1,
        "Material": {
            "constitutive_law": {"name": constitutive_law},
            "Variables": {
                "THICKNESS": 1.0,
                "DENSITY": density,
                "YOUNG_MODULUS": E,
                "POISSON_RATIO": nu,
                "MATERIAL_POINTS_PER_ELEMENT": material_points_per_element,
            },
            "Tables": {},
        }}]}, _f, indent=2)

n_steps = max(1, int(round(T_end / dt)))
parameters = {
    "problem_data": {"problem_name": "mpm_2d", "parallel_type": "OpenMP",
                     "start_time": 0.0, "end_time": T_end, "echo_level": 1},
    "solver_settings": {
        # solver_type is a SHORT LABEL, not a solver class name. Accepted:
        # static / quasi_static / dynamic (any capitalisation shown in the
        # knowledge). "MPMImplicitDynamicSolver" is rejected.
        "solver_type": "Dynamic",
        "model_part_name": "MPM_Material",
        "domain_size": 2,
        "echo_level": 0,
        "analysis_type": "non_linear",
        "time_integration_method": "implicit",
        # scheme_type is partitioned by solver: implicit takes newmark or
        # bossak only; central_difference and forward_euler are explicit-only.
        "scheme_type": "newmark",
        # The BODY mdpa.
        "model_import_settings": {"input_type": "mdpa",
                                  "input_filename": "body"},
        # The GRID has its OWN import key. Omitting this block is not a
        # missing-key error — Kratos falls back to a default name and the
        # failure surfaces as 'Error opening mdpa file : "unknown_name_Grid.mdpa"'.
        "grid_model_import_settings": {"input_type": "mdpa",
                                       "input_filename": "grid"},
        "material_import_settings": {
            "materials_filename": "ParticleMaterials.json"},
        "time_stepping": {"time_step": dt},
        "convergence_criterion": "residual_criterion",
        "displacement_relative_tolerance": 1e-4,
        "residual_relative_tolerance": 1e-4,
        "max_iteration": 20,
        "problem_domain_sub_model_part_list": ["Parts_Grid", "Parts_Body"],
        "processes_sub_model_part_list": ["DISPLACEMENT_floor"],
        # The body mdpa's element names are ignored: the material-point element
        # is chosen from the GRID geometry plus these flags. Writing a UP
        # element name without pressure_dofs silently yields displacement
        # elements and volumetric locking, with no message.
        "pressure_dofs": False,
    },
    "processes": {
        # Dirichlet data attaches to a sub model part of Background_Grid, never
        # of MPM_Material: the material points move, the grid does not, so the
        # constrained set has to be a grid region.
        "constraints_process_list": [{
            "python_module": "assign_vector_variable_process",
            "kratos_module": "KratosMultiphysics",
            "Parameters": {
                "model_part_name": "Background_Grid.DISPLACEMENT_floor",
                "variable_name": "DISPLACEMENT",
                "constrained": [True, True, True],
                "value": [0.0, 0.0, 0.0],
                "interval": [0.0, "End"]}}],
        # Gravity is OPT-IN. Without this block MP_VOLUME_ACCELERATION stays
        # zero and the body simply does not move: the run is successful,
        # converged and wrong, with no warning of any kind.
        "loads_process_list": [{
            "python_module": "assign_gravity_to_material_point_process",
            "kratos_module": "KratosMultiphysics.MPMApplication",
            "Parameters": {"model_part_name": "MPM_Material",
                           "modulus": abs(gravity),
                           "direction": [0.0, -1.0 if gravity < 0 else 1.0,
                                         0.0]}}],
    },
    "output_processes": {"vtk_output": [{
        "python_module": "mpm_vtk_output_process",
        "kratos_module": "KratosMultiphysics.MPMApplication",
        "Parameters": {
            "model_part_name": "MPM_Material",
            "output_control_type": "step",
            "output_interval": max(1, n_steps // 20),
            "file_format": "ascii",
            "output_path": "vtk_output",
            "gauss_point_variables_in_elements": [
                "MP_DISPLACEMENT", "MP_VELOCITY", "MP_CAUCHY_STRESS_VECTOR"],
        }}]},
}

model = KM.Model()
MpmAnalysis(model, KM.Parameters(json.dumps(parameters))).Run()

# Read the answer off the material points. MP_* variables live in the
# MPMApplication namespace, not in the KratosMultiphysics core one.
mp = model["MPM_Material"]
seeded = len(body_elements) * material_points_per_element
max_disp = max_vel = total_mass = 0.0
for el in mp.Elements:
    d = el.CalculateOnIntegrationPoints(KratosMPM.MP_DISPLACEMENT,
                                        mp.ProcessInfo)[0]
    v = el.CalculateOnIntegrationPoints(KratosMPM.MP_VELOCITY,
                                        mp.ProcessInfo)[0]
    m = el.CalculateOnIntegrationPoints(KratosMPM.MP_MASS, mp.ProcessInfo)[0]
    max_disp = max(max_disp, (d[0] ** 2 + d[1] ** 2) ** 0.5)
    max_vel = max(max_vel, (v[0] ** 2 + v[1] ** 2) ** 0.5)
    total_mass += m

summary = {
    "solver": "KratosMultiphysics.MPMApplication",
    "n_material_points": mp.NumberOfElements(),
    "n_material_points_seeded": seeded,
    # Non-zero means points left the grid and took their mass with them. The
    # log says so in a WARNING; this line says so in the result file.
    "material_points_erased": seeded - mp.NumberOfElements(),
    "n_steps": n_steps,
    "dt": dt,
    "max_MP_DISPLACEMENT": max_disp,
    "max_MP_VELOCITY": max_vel,
    "total_material_point_mass": total_mass,
    "grid": f"{n_cells_x}x{n_cells_y}",
}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("MPM complete:", json.dumps(summary))
if summary["material_points_erased"]:
    print(f"WARNING: {summary['material_points_erased']} material points left "
          f"the background grid and their mass is no longer in the "
          f"simulation. Enlarge the grid so it encloses the trajectory.")
'''


KNOWLEDGE = {
    "mpm": {
        "description": "Material Point Method via Kratos MPMApplication",
        "application": "MPMApplication (pip install KratosMPMApplication)",
        "elements": {
            "2D": [
                "MPMUpdatedLagrangian2D3N",
                "MPMUpdatedLagrangian2D4N",
            ],
            "3D": [
                "MPMUpdatedLagrangian3D4N",
                "MPMUpdatedLagrangian3D8N",
            ],
            "axisymmetric": [
                "MPMUpdatedLagrangianAxisymmetry2D3N",
                "MPMUpdatedLagrangianAxisymmetry2D4N",
            ],
            "PQ_variant": [
                "MPMUpdatedLagrangianPQ",
            ],
            "UP_variant": [
                "MPMUpdatedLagrangianUP",
                "MPMUpdatedLagrangianUP2D3N",
            ],
        },
        # Registered names, checked with KratosGlobals.HasConstitutiveLaw against the
        # installed MPMApplication 10.4.0 wheel (2026-08-03). The previous entries were
        # family labels ("LinearElastic", "NeoHookean", "HenckyMC", "HenckyBorjaCamClay",
        # "Johnson-Cook") — none of those five strings resolves to a law, so a
        # Materials.json built from them fails with "not registered".
        "constitutive_laws": [
            "LinearElasticIsotropic3DLaw / LinearElasticIsotropicPlaneStrain2DLaw / "
            "LinearElasticIsotropicPlaneStress2DLaw / LinearElasticIsotropicAxisym2DLaw "
            "(small strain)",
            "HyperElasticNeoHookean3DLaw / HyperElasticNeoHookeanPlaneStrain2DLaw / "
            "HyperElasticNeoHookeanAxisym2DLaw / HyperElasticNeoHookeanUP3DLaw "
            "(finite strain)",
            "HenckyMCPlastic3DLaw / HenckyMCPlasticPlaneStrain2DLaw / "
            "HenckyMCPlasticAxisym2DLaw (Mohr-Coulomb with Hencky strain; "
            "HenckyMCStrainSofteningPlastic* for softening)",
            "HenckyBorjaCamClayPlastic3DLaw / HenckyBorjaCamClayPlasticPlaneStrain2DLaw "
            "(critical state)",
            "JohnsonCookThermalPlastic3DLaw / JohnsonCookThermalPlastic2DPlaneStrainLaw / "
            "JohnsonCookThermalPlastic2DAxisymLaw (rate-dependent plasticity)",
        ],
        # solver_type is a SHORT LABEL, not a class name. The accepted strings are
        # "static"/"Static", "quasi_static"/"Quasi-static" and "dynamic"/"Dynamic";
        # for "dynamic" a second key "time_integration_method" selects
        # "implicit" or "explicit". The class names MPMStaticSolver /
        # MPMImplicitDynamicSolver / MPMExplicitSolver are NOT accepted values.
        "solver_types": [
            "static (or Static)",
            "quasi_static (or Quasi-static)",
            "dynamic (or Dynamic) + time_integration_method: implicit | explicit",
        ],
        "scheme_types": {
            "implicit dynamic / quasi-static": ["newmark", "bossak (default)"],
            "explicit dynamic": ["central_difference (default)", "forward_euler"],
            "static": "no scheme_type key exists; supplying one fails validation",
        },
        # USL / USF / MUSL are values of the "stress_update" key and apply ONLY to
        # an explicit solver running scheme_type "forward_euler"; with
        # "central_difference" the option is forced to 0 and stress_update is
        # ignored entirely.
        "stress_update": ["usf (default)", "usl", "musl"],
        "model_parts": (
            "MPM needs TWO mdpa files and creates three model parts. "
            "solver_settings.grid_model_import_settings.input_filename names the "
            "background GRID mdpa (meshed with plain FEM element names such as "
            "Element2D4N); solver_settings.model_import_settings.input_filename names "
            "the BODY mdpa (meshed with MPM* element names). The model parts are "
            "'Background_Grid', 'MPM_Material' and 'Initial_MPM_Material' — those names "
            "are fixed, there is no key to rename the grid. Boundary conditions attach "
            "to sub model parts of Background_Grid; materials attach to sub model parts "
            "of Initial_MPM_Material."
        ),
        "pitfalls": [
            "[API] Kratos MPM element names ALL start with the literal prefix \"MPM\": MPMUpdatedLagrangian2D4N, MPMUpdatedLagrangian3D8N, MPMUpdatedLagrangianAxisymmetry2D4N, MPMUpdatedLagrangianPQ, MPMUpdatedLagrangianUP, etc. The prior catalog listed UpdatedLagrangianPQ2D / UpdatedLagrangianAxisym (without the MPM prefix) — none of those are registered. Signal: model_part.CreateNewElement(\"UpdatedLagrangian2D3N\", ...) raises 'is not registered!' and lists the registered elements; the full line is Error: The Element \"UpdatedLagrangian2D3N\" is not registered! — Error:, the word Element and the name are all inserted at runtime around the literal, which is the trailing clause. Prepending MPM makes the identical call succeed. Beware grepping for the bare name — it matches as a substring of the MPM-prefixed one, so only element creation settles it. (Verified by execution 2026-08-07.)",
            "[Input] MPM reads TWO mdpa files, and the background grid has its own key: solver_settings.grid_model_import_settings.input_filename. Omitting that block does not raise a missing-key error — the default filename is used and the failure surfaces as a missing file. Signal: RuntimeError 'Error opening mdpa file : \"unknown_name_Grid.mdpa\"' — the literal string unknown_name_Grid is the giveaway that the grid import block is absent rather than the file being misnamed. (Verified by execution 2026-08-07.)",
            "[Input] MATERIAL_POINTS_PER_ELEMENT lives in the MATERIALS json, under properties[i].Material.Variables — not in ProjectParameters. On Kratos 10.4.2 and 10.4.3 its absence is a hard error; on 10.3.0 and 10.4.0 it is only a warning, and 1 material point per element is assumed. Signal: on 10.4.2/10.4.3 RuntimeError '\"MATERIAL_POINTS_PER_ELEMENT\" is not specified in Properties' raised from MaterialPointGeneratorUtility during solver Initialize; on 10.3.0 the run completes with the warning 'MATERIAL_POINTS_PER_ELEMENT is not specified in Properties, ' followed by '1 material point per element is assumed.' (Verified by execution 2026-08-07 on 10.4.3; the 10.3.0 warning re-measured by execution, the 10.4.0 and 10.4.2 behaviour read from their sources.)",
            "[Input] The number of material points per element is drawn from a fixed set that depends on the GRID element geometry, and the sets are NOT the same across geometries: Triangular 1/3/4/6/12, Quadrilateral 1/4/9/16/25, Tetrahedral 1/4/8/14/24, Hexahedral 1/8/27/64/125 on Kratos 10.4.2 and 10.4.3, where anything else is rejected. Signal: RuntimeError 'The input number of MATERIAL_POINTS_PER_ELEMENT (5) is not available for Quadrilateral elements' followed by 'Available options are: 1, 4, 9, 16 and 25.' — the message names the GRID geometry, so it is also how you discover your background grid is quads when you assumed triangles. (Verified by execution 2026-08-07 on 10.4.3; the allowed sets were read back from the installed libKratosMPMCore. Kratos 10.3.0 and 10.4.0 only warn and fall back to the geometry default, 4 for quadrilaterals ('The default number of material points: 4 is currently assumed.') and 3 for triangles, and their sets differ (quadrilateral 1, 4, 9, 16), so on those releases the same mistake yields a different material-point count instead of an error; the 10.3.0 quadrilateral case was re-measured by execution.)",
            "[Input] The legacy spelling PARTICLES_PER_ELEMENT still works, and the solver REWRITES YOUR MATERIALS FILE ON DISK to the new name as a side effect of running. Signal: the run prints '\\'PARTICLES_PER_ELEMENT\\' is deprecated; use \\'MATERIAL_POINTS_PER_ELEMENT\\' instead.' and completes normally, after which the materials json in the working directory no longer contains the string PARTICLES_PER_ELEMENT — a version-controlled input file is modified by a simulation run. (Verified by execution 2026-08-07.)",
            "[Input] Materials entries address the BODY through 'Initial_MPM_Material.<SubModelPart>'. Using the MPM_Material root instead fails, because at materials-reading time the body sub model parts only exist under Initial_. Signal: RuntimeError 'There is no sub model part with name \"Parts_Parts_Auto1\" in model part \"MPM_Material\"' followed by the list of sub model parts that DO exist. (Verified by execution 2026-08-07.)",
            "[BC] Boundary conditions attach to sub model parts of Background_Grid, never of MPM_Material — the material points move, the grid does not, so the constrained set has to be a grid region. Signal: pointing a constraints_process_list entry at 'MPM_Material.<name>' raises RuntimeError 'There is no sub model part with name \"DISPLACEMENT_Displacement_Auto1\" in model part \"MPM_Material\"'; the same block with 'Background_Grid.<name>' runs. (Verified by execution 2026-08-07.)",
            "[API] solver_settings.solver_type takes a short label, not a solver class name. Accepted: 'static'/'Static', 'quasi_static'/'Quasi-static', 'dynamic'/'Dynamic' (which then requires time_integration_method 'implicit' or 'explicit'). Signal: 'MPMImplicitDynamicSolver' raises Exception 'The requested solver type \"MPMImplicitDynamicSolver\" is not in the python solvers wrapper' + 'Available options are: \"static\", \"dynamic\", \"quasi_static\"'. (Verified by execution 2026-08-07 — the class-name spellings were previously served as the solver_types list.)",
            "[API] scheme_type is partitioned by solver: an implicit run takes only newmark or bossak, an explicit run only central_difference or forward_euler, and a static run has no scheme_type key at all. Signal: scheme_type 'central_difference' on an implicit dynamic solver raises Exception 'The requested scheme type \"central_difference\" is not available!' + 'Available options are: \"newmark\", \"bossak\"'. (Verified by execution 2026-08-07.)",
            "[Input] Constitutive law names are the fully-qualified registered strings; the short family label is the single most common MPM setup error. Signal: 'LinearElasticPlaneStrain2DLaw' raises RuntimeError 'Kratos components missing \"LinearElasticPlaneStrain2DLaw\"' — the fix is LinearElasticIsotropicPlaneStrain2DLaw, i.e. the 'Isotropic' the short name drops. (Verified by execution 2026-08-07.)",
            "[Numerical] Material points that leave the background grid are DELETED, and the mass they carry leaves the simulation with them. The receipt is two log lines, not an error, so a partially-escaped body silently loses mass; only when the last point is gone does the run stop. Signal: 'Search Element for Material Point: 26 is failed. Geometry is cleared.' then 'MaterialPointEraseProcess: 1 particle elements have been erased.', and once the body is entirely outside, RuntimeError 'No degrees of freedom in model part: MPM_Material'. (Verified by execution 2026-08-07 by letting a body free-fall out of its grid.)",
            "[Physics] Gravity is opt-in. Without an assign_gravity_to_material_point_process block, MP_VOLUME_ACCELERATION stays zero and the body simply does not fall — the run is successful, converged and wrong. Signal: an otherwise identical deck with the gravity process removed completes with exit code 0, unchanged total material-point mass, and MP_DISPLACEMENT exactly 0.0 where the reference gives -0.04905; no warning is emitted. (Verified by execution 2026-08-07.)",
            "[Input] The BODY mdpa carries MPM* element names but the runtime ignores them: the material-point element type is chosen from the GRID geometry plus the ProjectParameters flags pressure_dofs and is_pqmpm. Writing MPMUpdatedLagrangianUP2D3N in the body mdpa without \"pressure_dofs\": true silently yields plain displacement elements. Signal: no message at all — the mixed formulation is simply absent, so a nearly-incompressible run shows volumetric locking rather than reporting a configuration error. (Verified from Kratos source 10.4.3, MaterialPointGeneratorUtility hard-codes the element stem; not separately executed.)",
        ],
        "guidance": [
            "[Numerical] The background grid must enclose the whole TRAJECTORY of the body, not just its initial position — points that exit are erased (see pitfalls).",
            "[Numerical] Penalty Dirichlet conditions take penalty_coefficient on Kratos 10.4.2 and 10.4.3 (the older name penalty_factor is auto-renamed there); 10.3.0 and 10.4.0 know only penalty_factor and reject penalty_coefficient with 'is present in this Parameters but NOT in the default values'. It defaults to 0, which silently disables the constraint; shipped tests use 1e10 to 1e12, i.e. two to three orders above YOUNG_MODULUS.",
            "[Numerical] Cell-crossing instability: Kratos MPM does NOT implement GIMP or CPDI — both strings appear in zero files of MPMApplication, so advice to 'use GIMP or CPDI shape functions' cannot be acted on here. The mitigation Kratos does provide is PQMPM (partitioned-quadrature MPM), switched on with \"is_pqmpm\": true in solver_settings, which makes the generator build MPMUpdatedLagrangianPQ elements instead.",
            "[Numerical] Material points per cell: 4-16 typical, but only from the geometry's allowed set (see pitfalls).",
            "[Numerical] Time step: dt < h/c where h=cell size, c=wave speed.",
            "[Numerical] Zero-energy modes possible with linear elements: use stabilization.",
            "[Numerical] An inverted deformation gradient is the canonical MPM blow-up and it does raise: 'MPM UPDATED LAGRANGIAN DISPLACEMENT ELEMENT INVERTED: |F|<0 detF = <value>'. It usually means the time step is too large or there are too few material points per element.",
        ]
    },
}

GENERATORS = {
    "mpm_2d": _mpm_2d_kratos,
}
