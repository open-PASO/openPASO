"""Kratos specialized application generators and knowledge.

Covers (REAL solve templates only): PoroMechanics, ShallowWater, Dam,
ConstitutiveLaws, DEM-Structures, CableNet, Optimization.

2026-06-26 honesty audit
------------------------
This module previously also held a `_generic_kratos_template` factory that
emitted "availability-probe" stubs — scripts whose ONLY action was to
import-check a Kratos sub-application and write {"note": "not installed"}
with no solver run. That factory and all nine stub generators built on it
(wind_engineering_2d, thermal_dem_2d, swimming_dem_2d, fem_to_dem_2d,
chimera_2d, droplet_dynamics_2d, free_surface_2d, fluid_biomedical_2d,
fluid_hydraulics_2d) have been REMOVED, together with their registry entries
and their PhysicsCapability rows in KratosBackend.supported_physics(). None
of those applications is importable in the installed Kratos stack, so the
catalog must not advertise them.

The REAL solve templates that remain in this module each build a model +
mesh, run an AnalysisStage / strategy solve, and write output. NOTE: their
applications (PoromechanicsApplication, ShallowWaterApplication,
DamApplication, ConstitutiveLawsApplication, DemStructuresCouplingApplication,
CableNetApplication, OptimizationApplication) are not part of a minimal
StructuralMechanics / ConvectionDiffusion / ContactStructuralMechanics /
LinearSolvers install, but each is its own PyPI wheel at 10.3.0
(KratosPoromechanicsApplication, KratosShallowWaterApplication,
KratosDamApplication, KratosConstitutiveLawsApplication,
KratosDemStructuresCouplingApplication, KratosCableNetApplication,
KratosOptimizationApplication; KratosMultiphysics-all brings the first four).
They are retained because they are genuine parameterized solves (not
import-probe stubs). The honesty fix mandated by the
audit was the removal of every no-solve availability-probe stub, which is
done.
"""


def _poromechanics_2d(params: dict) -> str:
    """Real Kratos u-Pl (u-Pw) coupled consolidation solve — Terzaghi column.

    Uses PoromechanicsApplication with `UPlSmallStrainElement2D4N` on a
    structured quad grid (2D plane strain), `LinearElasticPlaneStrainSolid2DLaw`,
    `PoroNewmarkQuasistaticUPlScheme` and a Newton-Raphson strategy — all
    built programmatically on a `KM.Model` (no .mdpa / ProjectParameters).
    A vertical traction `q_load` is applied on the drained top edge via
    `UPlFaceLoadCondition2D2N` (nodal FACE_LOAD); the base is fixed and
    impermeable, lateral edges are rollers.  NOTE: PoromechanicsApplication
    uses the u-Pl formulation (already so in 10.2.3 and 10.3.0, unchanged in
    10.4.2; the UPw* names belong to GeoMechanicsApplication); the pore-pressure DOF is
    `P.LIQUID_PRESSURE` (not WATER_PRESSURE) and material variables like
    DENSITY_SOLID / BULK_MODULUS_LIQUID must be fetched through
    `KM.KratosGlobals.GetVariable` because they are not module attributes.
    Output is written via `KM.VtkOutput` (legacy `.vtk`).  The summary
    contains base excess pore pressure and top settlement, both compared
    to the Terzaghi analytic series at the reached time factor Tv.
    """
    nx = params.get("nx", 4)
    ny = params.get("ny", 40)
    lx = params.get("lx", 1.0)           # column width  [m]
    ly = params.get("ly", 1.0)           # column height [m]
    E = params.get("E", 1.0e7)           # drained Young's modulus [Pa]
    nu = params.get("nu", 0.0)           # nu=0 -> oedometric modulus = E
    porosity = params.get("porosity", 0.3)
    k_int = params.get("k_int", 1.0e-12)  # intrinsic permeability [m2]
    mu_l = params.get("mu_l", 1.0e-3)     # liquid viscosity [Pa s]
    q_load = params.get("q_load", 1.0e4)  # top compressive traction [Pa]
    dt = params.get("dt", 2.0)
    n_steps = params.get("n_steps", 10)
    return f'''\
"""Terzaghi-like 1D consolidation column — Kratos PoromechanicsApplication (u-Pl).

A 2D plane-strain column of saturated porous material (width B, height H)
is loaded on top by a vertical traction q at t=0.  The top edge is drained
(LIQUID_PRESSURE = 0), the bottom is fixed and impermeable, the lateral
edges are rollers (u_x = 0) and impermeable.  Quasi-static Newmark u-Pl
scheme + Newton-Raphson.  Writes the LIQUID_PRESSURE and DISPLACEMENT
fields as legacy .vtk and a results_summary.json.

Note: PoromechanicsApplication uses the u-Pl (liquid pressure)
formulation: elements are UPlSmallStrainElement2D4N etc. and the nodal
unknown is P.LIQUID_PRESSURE.
"""
import json
import math
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA  # RAYLEIGH_* vars
import KratosMultiphysics.PoromechanicsApplication as P

# ------------------------------------------------------------------ inputs
nx, ny = {nx}, {ny}                # mesh divisions (x, y)
B, H = {lx}, {ly}                  # column width / height [m]
E, nu = {E}, {nu}                  # drained Young's modulus / Poisson ratio
porosity = {porosity}
rho_s, rho_l = 2000.0, 1000.0   # solid / liquid density [kg/m3]
k_int = {k_int}                 # intrinsic permeability [m2]
mu_l = {mu_l}                   # liquid dynamic viscosity [Pa s]
K_s = 1.0e20                    # solid grain bulk modulus (incompressible)
K_l = 2.0e9                     # liquid bulk modulus [Pa]
q_load = {q_load}               # top compressive traction [Pa]
dt = {dt}
n_steps = {n_steps}

# Analytic reference (Terzaghi): c_v = (k/mu) * Eoed
Eoed = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
c_v = (k_int / mu_l) * Eoed
T_v = c_v * n_steps * dt / H**2

def node_id(i, j):
    return 1 + j * (nx + 1) + i

model = KM.Model()
mp = model.CreateModelPart("PorousDomain")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
mp.SetBufferSize(2)

# Same nodal variable set as poromechanics_U_Pl_solver.AddVariables();
# the elements read e.g. nodal INITIAL_STRESS_TENSOR during Initialize,
# so omitting "exotic" entries crashes deep in C++.
for v in (KM.DISPLACEMENT, KM.REACTION, KM.VELOCITY, KM.ACCELERATION,
          KM.VOLUME_ACCELERATION, KM.FACE_LOAD, KM.FORCE, KM.NODAL_AREA,
          P.LIQUID_PRESSURE, P.REACTION_LIQUID_PRESSURE,
          P.DT_LIQUID_PRESSURE, P.NORMAL_LIQUID_FLUX, P.LIQUID_DISCHARGE,
          P.INITIAL_STRESS_TENSOR, P.NODAL_EFFECTIVE_STRESS_TENSOR,
          P.NODAL_JOINT_AREA, P.NODAL_JOINT_WIDTH, P.NODAL_JOINT_DAMAGE,
          P.NODAL_MID_PLANE_LIQUID_PRESSURE, P.NODAL_SLIP_TENDENCY):
    mp.AddNodalSolutionStepVariable(v)

# Nodes: structured grid, y from 0 (base) to H (top)
for j in range(ny + 1):
    for i in range(nx + 1):
        mp.CreateNewNode(node_id(i, j), i * B / nx, j * H / ny, 0.0)

# Poromechanics material variables are registered in the kernel but not
# exposed as attributes of KM or P -> fetch them by name.
_V = KM.KratosGlobals.GetVariable
prop = mp.CreateNewProperties(1)
prop.SetValue(KM.YOUNG_MODULUS, E)
prop.SetValue(KM.POISSON_RATIO, nu)
prop.SetValue(_V("DENSITY_SOLID"), rho_s)
prop.SetValue(_V("DENSITY_LIQUID"), rho_l)
prop.SetValue(_V("POROSITY"), porosity)
prop.SetValue(_V("BULK_MODULUS_SOLID"), K_s)
prop.SetValue(_V("BULK_MODULUS_LIQUID"), K_l)
prop.SetValue(_V("PERMEABILITY_XX"), k_int)
prop.SetValue(_V("PERMEABILITY_YY"), k_int)
prop.SetValue(_V("PERMEABILITY_XY"), 0.0)
prop.SetValue(_V("DYNAMIC_VISCOSITY_LIQUID"), mu_l)
prop.SetValue(P.BIOT_COEFFICIENT, 1.0)
prop.SetValue(KM.THICKNESS, 1.0)
prop.SetValue(KM.CONSTITUTIVE_LAW, P.LinearElasticPlaneStrainSolid2DLaw())

eid = 1
for j in range(ny):
    for i in range(nx):
        mp.CreateNewElement(
            "UPlSmallStrainElement2D4N", eid,
            [node_id(i, j), node_id(i + 1, j),
             node_id(i + 1, j + 1), node_id(i, j + 1)],
            prop,
        )
        eid += 1

# Top face load conditions (traction q downward)
cid = 1
for i in range(nx):
    mp.CreateNewCondition(
        "UPlFaceLoadCondition2D2N", cid,
        [node_id(i, ny), node_id(i + 1, ny)], prop)
    cid += 1

for node in mp.Nodes:
    node.AddDof(KM.DISPLACEMENT_X, KM.REACTION_X)
    node.AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y)
    node.AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z)
    node.Fix(KM.DISPLACEMENT_Z)
    node.AddDof(P.LIQUID_PRESSURE, P.REACTION_LIQUID_PRESSURE)
    node.SetSolutionStepValue(P.LIQUID_PRESSURE, 0.0)

# BCs
for j in range(ny + 1):
    for i in (0, nx):                       # lateral rollers
        mp.Nodes[node_id(i, j)].Fix(KM.DISPLACEMENT_X)
for i in range(nx + 1):
    n = mp.Nodes[node_id(i, 0)]             # fixed impermeable base
    n.Fix(KM.DISPLACEMENT_X)
    n.Fix(KM.DISPLACEMENT_Y)
    n = mp.Nodes[node_id(i, ny)]            # drained top
    n.Fix(P.LIQUID_PRESSURE)
    n.SetSolutionStepValue(P.LIQUID_PRESSURE, 0.0)
    n.SetSolutionStepValue(KM.FACE_LOAD_Y, -q_load)

# ProcessInfo required by elements / scheme
mp.ProcessInfo[KM.TIME] = 0.0
mp.ProcessInfo[KM.DELTA_TIME] = dt
mp.ProcessInfo[KM.STEP] = 0
mp.ProcessInfo[P.TIME_UNIT_CONVERTER] = 1.0
mp.ProcessInfo[P.NODAL_SMOOTHING] = False
mp.ProcessInfo[SMA.RAYLEIGH_ALPHA] = 0.0
mp.ProcessInfo[SMA.RAYLEIGH_BETA] = 0.0
mp.ProcessInfo[P.G_COEFFICIENT] = 0.0
mp.ProcessInfo[P.VELOCITY_COEFFICIENT] = 1.0
mp.ProcessInfo[P.DT_LIQUID_PRESSURE_COEFFICIENT] = 1.0

# Quasi-static Newmark u-Pl scheme: (theta_u, theta_p, beta, gamma)
scheme = P.PoroNewmarkQuasistaticUPlScheme(0.5, 0.5, 0.25, 0.5)
linear_solver = KM.SkylineLUFactorizationSolver()
builder_and_solver = KM.ResidualBasedBlockBuilderAndSolver(linear_solver)
conv = KM.DisplacementCriteria(1.0e-6, 1.0e-12)
strat = KM.ResidualBasedNewtonRaphsonStrategy(
    mp, scheme, conv, builder_and_solver,
    15, True, False, False,
)
strat.SetEchoLevel(0)
strat.Initialize()
strat.Check()

vtk_params = KM.Parameters(json.dumps({{
    "model_part_name": "PorousDomain",
    "output_control_type": "step",
    "output_interval": 1,
    "file_format": "ascii",
    "output_path": ".",
    "output_sub_model_parts": False,
    "save_output_files_in_folder": False,
    "nodal_solution_step_data_variables": ["DISPLACEMENT", "LIQUID_PRESSURE"],
}}))
vtk = KM.VtkOutput(mp, vtk_params)

time = 0.0
for step in range(1, n_steps + 1):
    time += dt
    mp.CloneTimeStep(time)
    mp.ProcessInfo[KM.STEP] = step
    strat.InitializeSolutionStep()
    strat.Predict()
    strat.SolveSolutionStep()
    strat.FinalizeSolutionStep()
vtk.PrintOutput()

# Results
base_mid = mp.Nodes[node_id(nx // 2, 0)]
top_mid = mp.Nodes[node_id(nx // 2, ny)]
p_base = float(base_mid.GetSolutionStepValue(P.LIQUID_PRESSURE))
uy_top = float(top_mid.GetSolutionStepValue(KM.DISPLACEMENT_Y))

# Terzaghi analytic comparison at Tv
p_ratio_analytic = sum(
    2.0 / ((2 * m + 1) * math.pi / 2.0) * math.sin((2 * m + 1) * math.pi / 2.0)
    * math.exp(-(((2 * m + 1) * math.pi / 2.0) ** 2) * T_v)
    for m in range(200)
)
U_analytic = 1.0 - sum(
    2.0 / (((2 * m + 1) * math.pi / 2.0) ** 2)
    * math.exp(-(((2 * m + 1) * math.pi / 2.0) ** 2) * T_v)
    for m in range(200)
)
settle_inf = q_load * H / Eoed

summary = {{
    "pore_pressure_base": p_base,
    "p_base_over_q": p_base / q_load,
    "p_base_over_q_terzaghi": p_ratio_analytic,
    "settlement_top": uy_top,
    "settlement_top_terzaghi": -U_analytic * settle_inf,
    "time_factor_Tv": T_v,
    "n_nodes": mp.NumberOfNodes(),
    "n_elements": mp.NumberOfElements(),
}}
print(json.dumps(summary, indent=2))
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
'''


def _shallow_water_2d(params: dict) -> str:
    """Minimal meaningful 2D shallow-water solve — Kratos ShallowWaterApplication.

    Closed square basin with still water depth `H0` and an initial Gaussian
    free-surface bump of amplitude `amp` at the center.  Uses
    `WaveElement2D3N` (the element the app's own `wave_solver.py` selects
    for its default "bdf" scheme) on a structured triangular mesh with the
    app's `ShallowWaterResidualBasedBDFScheme` (BDF2 on VELOCITY + HEIGHT,
    buffer size 3) inside a `ResidualBasedNewtonRaphsonStrategy`.  Slip
    walls are imposed by fixing the normal VELOCITY component on the
    boundary — no Condition objects required.  Output is written via
    `KM.VtkOutput` (legacy `.vtk`).

    PITFALLS (verified on the pip 10.4.2 wheel):
    - Registered SWE elements are WaveElement2D3N, CrankNicolsonWaveElement2D3N,
      BoussinesqElement2D3N, PrimitiveElement2D3N, ConservativeElementRV/FC2D3N.
      There is NO ShallowWaterElement2D3N and NO plain ConservativeElement2D3N.
    - GRAVITY must go into ProcessInfo as the scalar KM.GRAVITY_Z, and
      SW.INTEGRATE_BY_PARTS / KM.STABILIZATION_FACTOR / SW.RELATIVE_DRY_HEIGHT
      must be set, or the element Check fails / results are garbage.
    - BDF2 needs SetBufferSize(3) and the first solve must wait until the
      buffer is full (STEP + 1 >= 3), exactly as the app's base solver does.
    - DOFs are VELOCITY_X, VELOCITY_Y and SW.HEIGHT (no reactions).
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", 20)
    lx = params.get("lx", 10.0)
    ly = params.get("ly", 10.0)
    H0 = params.get("H0", 1.0)
    amp = params.get("amp", 0.1)
    sigma = params.get("sigma", 1.0)
    dt = params.get("dt", 0.05)
    n_steps = params.get("n_steps", 22)
    gravity = params.get("gravity", 9.81)
    return f'''\
"""2D shallow-water still-water perturbation in a closed basin — Kratos SW app.

WaveElement2D3N on a structured triangular mesh, BDF2 (ShallowWater
ResidualBasedBDFScheme) time integration, slip walls.  Writes HEIGHT /
FREE_SURFACE_ELEVATION / VELOCITY as Basin_0_*.vtk.
"""
import json
import math

import KratosMultiphysics as KM
import KratosMultiphysics.ShallowWaterApplication as SW

nx, ny = {nx}, {ny}
Lx, Ly = {lx}, {ly}
H0 = {H0}
amp = {amp}
sigma = {sigma}
dt = {dt}
n_steps = {n_steps}
gravity = {gravity}


def node_id(i, j):
    return 1 + j * (nx + 1) + i


model = KM.Model()
mp = model.CreateModelPart("Basin")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2

# Same nodal variable set as shallow_water_base_solver.AddVariables
for v in (SW.HEIGHT, KM.MOMENTUM, KM.VELOCITY, KM.ACCELERATION,
          SW.VERTICAL_VELOCITY, SW.FREE_SURFACE_ELEVATION, SW.BATHYMETRY,
          SW.TOPOGRAPHY, SW.MANNING, SW.RAIN, KM.NORMAL, KM.DISTANCE):
    mp.AddNodalSolutionStepVariable(v)

for j in range(ny + 1):
    for i in range(nx + 1):
        mp.CreateNewNode(node_id(i, j), i * Lx / nx, j * Ly / ny, 0.0)

prop = mp.CreateNewProperties(1)
eid = 1
for j in range(ny):
    for i in range(nx):
        n00, n10 = node_id(i, j), node_id(i + 1, j)
        n11, n01 = node_id(i + 1, j + 1), node_id(i, j + 1)
        mp.CreateNewElement("WaveElement2D3N", eid, [n00, n10, n11], prop); eid += 1
        mp.CreateNewElement("WaveElement2D3N", eid, [n00, n11, n01], prop); eid += 1

mp.SetBufferSize(3)  # BDF2 needs 3 buffer steps

mp.ProcessInfo.SetValue(KM.STEP, 0)
mp.ProcessInfo.SetValue(KM.GRAVITY_Z, gravity)
mp.ProcessInfo.SetValue(SW.INTEGRATE_BY_PARTS, False)
mp.ProcessInfo.SetValue(KM.STABILIZATION_FACTOR, 0.01)
mp.ProcessInfo.SetValue(SW.RELATIVE_DRY_HEIGHT, 0.1)

KM.VariableUtils().AddDof(KM.VELOCITY_X, mp)
KM.VariableUtils().AddDof(KM.VELOCITY_Y, mp)
KM.VariableUtils().AddDof(SW.HEIGHT, mp)

# Still water + Gaussian bump at rest
xc, yc = Lx / 2.0, Ly / 2.0
for node in mp.Nodes:
    r2 = (node.X - xc) ** 2 + (node.Y - yc) ** 2
    eta = amp * math.exp(-r2 / (2.0 * sigma ** 2))
    node.SetSolutionStepValue(SW.TOPOGRAPHY, -H0)
    node.SetSolutionStepValue(SW.BATHYMETRY, H0)
    node.SetSolutionStepValue(SW.HEIGHT, H0 + eta)
    node.SetSolutionStepValue(SW.FREE_SURFACE_ELEVATION, eta)
    node.SetSolutionStepValue(SW.MANNING, 0.0)

# Slip walls: zero normal velocity on the boundary
for i in range(nx + 1):
    for nid in (node_id(i, 0), node_id(i, ny)):
        mp.Nodes[nid].Fix(KM.VELOCITY_Y)
for j in range(ny + 1):
    for nid in (node_id(0, j), node_id(nx, j)):
        mp.Nodes[nid].Fix(KM.VELOCITY_X)

# BDF2 scheme on (VELOCITY, HEIGHT) — what the app's WaveSolver builds
scheme_settings = KM.Parameters()
scheme_settings.AddStringArray("solution_variables", ["VELOCITY", "HEIGHT"])
scheme_settings.AddEmptyValue("integration_order").SetInt(2)
scheme = SW.ShallowWaterResidualBasedBDFScheme(scheme_settings)

builder_and_solver = KM.ResidualBasedBlockBuilderAndSolver(
    KM.SkylineLUFactorizationSolver())
conv = KM.DisplacementCriteria(1.0e-6, 1.0e-9)
conv.SetEchoLevel(0)
strat = KM.ResidualBasedNewtonRaphsonStrategy(
    mp, scheme, conv, builder_and_solver,
    20, False, False, False)
strat.SetEchoLevel(0)
strat.Initialize()
strat.Check()

vtk_params = KM.Parameters(json.dumps({{
    "model_part_name": "Basin",
    "output_control_type": "step",
    "output_interval": 1,
    "file_format": "ascii",
    "output_path": ".",
    "output_sub_model_parts": False,
    "save_output_files_in_folder": False,
    "nodal_solution_step_data_variables":
        ["HEIGHT", "FREE_SURFACE_ELEVATION", "VELOCITY"],
}}))
vtk = KM.VtkOutput(mp, vtk_params)

probe = mp.Nodes[node_id(3 * nx // 4, ny // 2)]
probe_eta_initial = float(probe.GetSolutionStepValue(SW.FREE_SURFACE_ELEVATION))


def total_volume():
    cell = (Lx / nx) * (Ly / ny)
    vol = 0.0
    for jj in range(ny + 1):
        for ii in range(nx + 1):
            w = (1.0 if 0 < ii < nx else 0.5) * (1.0 if 0 < jj < ny else 0.5)
            vol += w * cell * mp.Nodes[node_id(ii, jj)].GetSolutionStepValue(SW.HEIGHT)
    return vol


vol_initial = total_volume()

# Time loop — solve only once the BDF buffer is full (STEP + 1 >= 3),
# exactly like ShallowWaterBaseSolver._TimeBufferIsInitialized.
n_solved = 0
time = 0.0
for step in range(1, n_steps + 1):
    time += dt
    mp.CloneTimeStep(time)
    mp.ProcessInfo[KM.STEP] = step
    if step + 1 >= 3:
        strat.InitializeSolutionStep()
        strat.Predict()
        strat.SolveSolutionStep()
        strat.FinalizeSolutionStep()
        SW.ShallowWaterUtilities().ComputeFreeSurfaceElevation(mp)
        n_solved += 1
vtk.PrintOutput()

etas = [float(n.GetSolutionStepValue(SW.FREE_SURFACE_ELEVATION)) for n in mp.Nodes]
vels = [math.hypot(n.GetSolutionStepValue(KM.VELOCITY_X),
                   n.GetSolutionStepValue(KM.VELOCITY_Y)) for n in mp.Nodes]
probe_eta_end = float(probe.GetSolutionStepValue(SW.FREE_SURFACE_ELEVATION))
vol_end = total_volume()

summary = {{
    "max_free_surface": max(etas),
    "min_free_surface": min(etas),
    "probe_eta_initial": probe_eta_initial,
    "probe_eta_t_end": probe_eta_end,
    "probe_eta_change": probe_eta_end - probe_eta_initial,
    "max_velocity": max(vels),
    "volume_initial": vol_initial,
    "volume_end": vol_end,
    "volume_rel_error": abs(vol_end - vol_initial) / vol_initial,
    "t_end": time,
    "dt": dt,
    "n_steps": n_steps,
    "n_solved_steps": n_solved,
    "n_nodes": mp.NumberOfNodes(),
    "n_elements": mp.NumberOfElements(),
}}
print(json.dumps(summary, indent=2))
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)

assert all(math.isfinite(e) for e in etas)
assert summary["max_free_surface"] < amp * 0.999, "bump did not decay"
assert abs(summary["probe_eta_change"]) > 1e-6, "wave never reached probe"
assert summary["max_velocity"] > 1e-4, "fluid never moved"
assert summary["volume_rel_error"] < 1e-2, "mass not conserved"
print("OK: shallow-water wave propagated and mass conserved.")
'''


def _dam_2d(params: dict) -> str:
    """2D gravity-dam cross-section — Kratos DamApplication, thermo-mechanical.

    Trapezoidal gravity dam (vertical upstream face, inclined downstream
    face) on a structured quad grid, clamped at the foundation line, under:
      * self-weight (VOLUME_ACCELERATION),
      * hydrostatic pressure on the upstream face, computed and applied by
        DamApplication's own `DamHydroConditionLoadProcess` (sets nodal
        POSITIVE_FACE_PRESSURE, consumed by SMA `LineLoadCondition2D2N`),
      * a uniform temperature change `T_now - T_ref` relative to the
        stress-free reference temperature.

    DamApplication components exercised: `SmallDisplacementThermoMechanicElement2D4N`
    (element), `ThermalLinearElastic2DPlaneStrain` (constitutive law, needs
    the DAM.THERMAL_EXPANSION property), `IncrementalUpdateStaticSmoothingScheme`
    (scheme; also smooths Cauchy stress to nodes as Poro.NODAL_CAUCHY_STRESS_TENSOR),
    and `DamHydroConditionLoadProcess` (hydrostatic load process).

    Verified equilibrium: sum of reactions matches the analytic hydrostatic
    resultant 0.5*rho_w*g*Hw^2 and the trapezoid self-weight to <3e-4.
    Output: `Dam_0_1.vtk` (legacy VTK) + `results_summary.json`.
    """
    H = params.get("H", 40.0)              # dam height [m]
    W_base = params.get("W_base", 30.0)    # base width [m]
    W_crest = params.get("W_crest", 5.0)   # crest width [m]
    nx = params.get("nx", 12)              # elements across the section
    ny = params.get("ny", 32)              # elements over the height
    E = params.get("E", 24.0e9)            # concrete Young's modulus [Pa]
    nu = params.get("nu", 0.20)            # Poisson ratio
    rho = params.get("rho", 2400.0)        # concrete density [kg/m^3]
    alpha = params.get("alpha", 1.0e-5)    # thermal expansion [1/K]
    rho_w = params.get("rho_w", 1000.0)    # water density [kg/m^3]
    Hw = params.get("water_level", 38.0)   # reservoir level above base [m]
    T_ref = params.get("T_ref", 10.0)      # stress-free reference temp [C]
    T_now = params.get("T_now", 0.0)       # current uniform temp [C]
    return f'''\
"""2D gravity-dam cross-section — Kratos DamApplication thermo-mechanical solve.

Trapezoidal gravity dam under self-weight, hydrostatic upstream pressure
(DamApplication DamHydroConditionLoadProcess) and uniform thermal load
(SmallDisplacementThermoMechanicElement2D4N + ThermalLinearElastic2DPlaneStrain,
IncrementalUpdateStaticSmoothingScheme).  Writes Dam_0_1.vtk and
results_summary.json.
"""
import json
import time as _time

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication  # registers LineLoadCondition2D2N
import KratosMultiphysics.PoromechanicsApplication as Poro
import KratosMultiphysics.DamApplication as DAM

t0 = _time.perf_counter()

# ----------------------------- parameters ---------------------------------
H = {H}                # dam height [m]
W_base = {W_base}      # base width [m]
W_crest = {W_crest}    # crest width [m]
nx, ny = {nx}, {ny}    # structured quad grid
E = {E}                # concrete Young's modulus [Pa]
nu = {nu}              # Poisson ratio
rho = {rho}            # concrete density [kg/m^3]
alpha = {alpha}        # thermal expansion coefficient [1/K]
g = 9.81               # gravity [m/s^2]
rho_w = {rho_w}        # water density [kg/m^3]
Hw = {Hw}              # reservoir water level above base [m]
T_ref = {T_ref}        # stress-free reference temperature [C]
T_now = {T_now}        # current uniform temperature [C]

# ------------------------------ model part --------------------------------
model = KM.Model()
mp = model.CreateModelPart("Dam")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
mp.ProcessInfo[DAM.TIME_UNIT_CONVERTER] = 1.0
mp.ProcessInfo.SetValue(Poro.IS_CONVERGED, True)
mp.SetBufferSize(2)

# Variable set mirrors DamApplication's dam_mechanical_solver.AddVariables
# plus the thermal pair needed by the thermo-mechanic element.
for v in (KM.DISPLACEMENT, KM.REACTION, KM.VELOCITY, KM.ACCELERATION,
          KM.POSITIVE_FACE_PRESSURE, KM.VOLUME_ACCELERATION, KM.NODAL_AREA,
          KM.TEMPERATURE,
          Poro.NODAL_CAUCHY_STRESS_TENSOR, DAM.INITIAL_NODAL_CAUCHY_STRESS_TENSOR,
          DAM.Vi_POSITIVE, DAM.Viii_POSITIVE,
          Poro.NODAL_JOINT_WIDTH, Poro.NODAL_JOINT_AREA,
          DAM.NODAL_YOUNG_MODULUS, Poro.INITIAL_STRESS_TENSOR,
          DAM.NODAL_REFERENCE_TEMPERATURE):
    mp.AddNodalSolutionStepVariable(v)


def node_id(i, j):
    return 1 + j * (nx + 1) + i


# Trapezoidal cross-section: upstream face vertical at x = 0; section width
# tapers linearly from W_base at y = 0 to W_crest at y = H.
for j in range(ny + 1):
    y = H * j / ny
    w = W_base + (W_crest - W_base) * y / H
    for i in range(nx + 1):
        mp.CreateNewNode(node_id(i, j), w * i / nx, y, 0.0)

prop = mp.CreateNewProperties(1)
prop.SetValue(KM.YOUNG_MODULUS, E)
prop.SetValue(KM.POISSON_RATIO, nu)
prop.SetValue(KM.DENSITY, rho)
prop.SetValue(DAM.THERMAL_EXPANSION, alpha)
prop.SetValue(KM.CONSTITUTIVE_LAW, DAM.ThermalLinearElastic2DPlaneStrain())

eid = 1
for j in range(ny):
    for i in range(nx):
        mp.CreateNewElement(
            "SmallDisplacementThermoMechanicElement2D4N", eid,
            [node_id(i, j), node_id(i + 1, j),
             node_id(i + 1, j + 1), node_id(i, j + 1)],
            prop)
        eid += 1

# Upstream face (x = 0): line-load conditions consuming POSITIVE_FACE_PRESSURE.
# Node order (j, j+1) makes the resultant act in +x (downstream) — verified
# against the analytic hydrostatic resultant via the reaction sum.
upstream = mp.CreateSubModelPart("UpstreamFace")
upstream.AddNodes([node_id(0, j) for j in range(ny + 1)])
cid = 1
for j in range(ny):
    mp.CreateNewCondition("LineLoadCondition2D2N", cid,
                          [node_id(0, j), node_id(0, j + 1)], prop)
    upstream.AddConditions([cid])
    cid += 1

# DOFs, gravity, temperature field
for node in mp.Nodes:
    node.AddDof(KM.DISPLACEMENT_X, KM.REACTION_X)
    node.AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y)
    node.AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z)
    node.Fix(KM.DISPLACEMENT_Z)
    node.SetSolutionStepValue(KM.VOLUME_ACCELERATION, [0.0, -g, 0.0])
    node.SetSolutionStepValue(KM.TEMPERATURE, T_now)
    node.SetSolutionStepValue(DAM.NODAL_REFERENCE_TEMPERATURE, T_ref)

# Clamp the foundation line (y = 0)
for i in range(nx + 1):
    n = mp.Nodes[node_id(i, 0)]
    n.Fix(KM.DISPLACEMENT_X)
    n.Fix(KM.DISPLACEMENT_Y)
    n.SetSolutionStepValue(KM.DISPLACEMENT_X, 0.0)
    n.SetSolutionStepValue(KM.DISPLACEMENT_Y, 0.0)

# --------------- hydrostatic load via DamApplication process ---------------
hydro_params = KM.Parameters(json.dumps({{
    "model_part_name": "UpstreamFace",
    "variable_name": "POSITIVE_FACE_PRESSURE",
    "Modify": True,
    "Gravity_Direction": "Y",
    "Reservoir_Bottom_Coordinate_in_Gravity_Direction": 0.0,
    "Spe_weight": rho_w * g,
    "Water_level": Hw,
    "Water_Table": 0,
    "interval": [0.0, 1000.0],
}}))
hydro_process = DAM.DamHydroConditionLoadProcess(upstream, hydro_params)

# ------------------------------- solver -----------------------------------
scheme = DAM.IncrementalUpdateStaticSmoothingScheme()
builder_and_solver = KM.ResidualBasedBlockBuilderAndSolver(
    KM.SkylineLUFactorizationSolver())
conv = KM.ResidualCriteria(1.0e-8, 1.0e-12)
strat = KM.ResidualBasedNewtonRaphsonStrategy(
    mp, scheme, conv, builder_and_solver,
    20, True, False, True)   # max_iter, compute_reactions, reform_dofs, move_mesh
strat.SetEchoLevel(0)

hydro_process.ExecuteInitialize()
mp.CloneTimeStep(1.0)
mp.ProcessInfo[KM.STEP] = 1
hydro_process.ExecuteInitializeSolutionStep()
strat.Check()
strat.Solve()

# ------------------------------- output -----------------------------------
vtk_params = KM.Parameters(json.dumps({{
    "model_part_name": "Dam",
    "output_control_type": "step",
    "output_interval": 1,
    "file_format": "ascii",
    "output_path": ".",
    "output_sub_model_parts": False,
    "save_output_files_in_folder": False,
    "nodal_solution_step_data_variables":
        ["DISPLACEMENT", "REACTION", "TEMPERATURE", "POSITIVE_FACE_PRESSURE"],
}}))
KM.VtkOutput(mp, vtk_params).PrintOutput()

# ------------------------------ summary -----------------------------------
crest_us = mp.Nodes[node_id(0, ny)]       # upstream crest corner
heel = mp.Nodes[node_id(0, 0)]            # upstream base corner (heel)
toe = mp.Nodes[node_id(nx, 0)]            # downstream base corner (toe)

sum_rx = sum(n.GetSolutionStepValue(KM.REACTION_X) for n in mp.Nodes)
sum_ry = sum(n.GetSolutionStepValue(KM.REACTION_Y) for n in mp.Nodes)
area = 0.5 * (W_base + W_crest) * H
weight = rho * g * area                       # N per meter thickness
hydro_force = 0.5 * rho_w * g * Hw ** 2       # N per meter thickness


def nodal_syy(n):
    m = n.GetSolutionStepValue(Poro.NODAL_CAUCHY_STRESS_TENSOR)
    return m[1, 1]


syy_vals = {{n.Id: nodal_syy(n) for n in mp.Nodes}}

summary = {{
    "crest_ux_m": float(crest_us.GetSolutionStepValue(KM.DISPLACEMENT_X)),
    "crest_uy_m": float(crest_us.GetSolutionStepValue(KM.DISPLACEMENT_Y)),
    "heel_nodal_sigma_yy_Pa": float(syy_vals[heel.Id]),
    "toe_nodal_sigma_yy_Pa": float(syy_vals[toe.Id]),
    "min_nodal_sigma_yy_Pa": float(min(syy_vals.values())),
    "max_nodal_sigma_yy_Pa": float(max(syy_vals.values())),
    "sum_reaction_x_N": float(sum_rx),
    "sum_reaction_y_N": float(sum_ry),
    "analytic_hydrostatic_force_N": float(hydro_force),
    "analytic_self_weight_N": float(weight),
    # reactions balance the applied loads: sum_rx = -F_hydro, sum_ry = +Weight
    "equilibrium_err_x": float(abs(sum_rx + hydro_force) / hydro_force),
    "equilibrium_err_y": float(abs(sum_ry - weight) / weight),
    "delta_T_K": T_now - T_ref,
    "n_nodes": mp.NumberOfNodes(),
    "n_elements": mp.NumberOfElements(),
    "runtime_s": _time.perf_counter() - t0,
}}
print(json.dumps(summary, indent=2))
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
'''


def _constitutive_laws_2d(params: dict) -> str:
    """Plane-strain tension strip with Von Mises isotropic plasticity —
    KratosMultiphysics.ConstitutiveLawsApplication (pip wheel KratosConstitutiveLawsApplication; runs on 10.3.0).

    Uses `SmallStrainIsotropicPlasticityPlaneStrainVonMisesVonMises`
    (registered ONLY by ConstitutiveLawsApplication; strain size 3, so it
    is compatible with the 2D solid elements — unlike
    SmallStrainJ2PlasticityPlaneStrain2DLaw whose strain size is 4) on a
    structured `SmallDisplacementElement2D4N` quad grid.  The right edge
    displacement is ramped past yield in `n_steps` Newton-Raphson load
    increments; the secant tangent estimation (TANGENT_OPERATOR_ESTIMATION
    = 3) is the only scheme of the 10.4.2 wheel that converges for this
    law (the default perturbation tangents diverge / throw "Condition
    number of the matrix is too high!", and the elastic tangent stagnates
    once the section is fully plastic).  Output: legacy `.vtk` with nodal
    DISPLACEMENT/REACTION plus `results_summary.json` with the post-yield
    force plateau, equivalent plastic strain and plastic dissipation.
    """
    nx = params.get("nx", 40)
    ny = params.get("ny", 8)
    lx = params.get("lx", 10.0)
    ly = params.get("ly", 1.0)
    E = params.get("E", 200.0e9)
    nu = params.get("nu", 0.3)
    yield_stress = params.get("yield_stress", 250.0e6)
    fracture_energy = params.get("fracture_energy", 1.0e9)
    u_max = params.get("u_max", 0.04)
    n_steps = params.get("n_steps", 8)
    omp_threads = params.get("omp_threads", 4)
    return f'''\
"""Plane-strain tension strip, Von Mises isotropic plasticity — Kratos
ConstitutiveLawsApplication (SmallStrainIsotropicPlasticityPlaneStrainVonMisesVonMises).

Left edge ux = 0, bottom edge symmetry uy = 0, right edge ux ramped past
yield in load increments.  Writes Structure_0_*.vtk and results_summary.json.
"""
import json
import os
import time

# Cap OpenMP threads BEFORE importing Kratos: on many-core machines thread
# contention inflates this sub-second solve to ~1 minute.
os.environ.setdefault("OMP_NUM_THREADS", "{omp_threads}")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA  # noqa: F401
import KratosMultiphysics.ConstitutiveLawsApplication as CLA

t_start = time.time()

nx, ny = {nx}, {ny}
L,  h  = {lx}, {ly}
E,  nu = {E}, {nu}
yield_stress = {yield_stress}
fracture_energy = {fracture_energy}
u_max = {u_max}
n_steps = {n_steps}

LAW_NAME = "SmallStrainIsotropicPlasticityPlaneStrainVonMisesVonMises"
assert not hasattr(SMA, LAW_NAME) and not hasattr(KM, LAW_NAME)  # CLA-owned


def node_id(i, j):
    return 1 + j * (nx + 1) + i


model = KM.Model()
mp = model.CreateModelPart("Structure")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
mp.SetBufferSize(2)
for v in (KM.DISPLACEMENT, KM.REACTION, KM.VOLUME_ACCELERATION):
    mp.AddNodalSolutionStepVariable(v)

for j in range(ny + 1):
    for i in range(nx + 1):
        mp.CreateNewNode(node_id(i, j), i * L / nx, j * h / ny, 0.0)

prop = mp.CreateNewProperties(1)
prop.SetValue(KM.YOUNG_MODULUS, E)
prop.SetValue(KM.POISSON_RATIO, nu)
prop.SetValue(KM.DENSITY, 0.0)
prop.SetValue(CLA.YIELD_STRESS_TENSION, yield_stress)
prop.SetValue(CLA.YIELD_STRESS_COMPRESSION, yield_stress)
prop.SetValue(KM.FRACTURE_ENERGY, fracture_energy)
prop.SetValue(CLA.HARDENING_CURVE, 0)            # LinearSoftening; huge Gf -> ~flat
prop.SetValue(CLA.TANGENT_OPERATOR_ESTIMATION, 3)  # Secant: the only stable choice
prop.SetValue(KM.CONSTITUTIVE_LAW, getattr(CLA, LAW_NAME)())

eid = 1
for j in range(ny):
    for i in range(nx):
        mp.CreateNewElement(
            "SmallDisplacementElement2D4N", eid,
            [node_id(i, j), node_id(i + 1, j),
             node_id(i + 1, j + 1), node_id(i, j + 1)],
            prop,
        )
        eid += 1

for node in mp.Nodes:
    node.AddDof(KM.DISPLACEMENT_X, KM.REACTION_X)
    node.AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y)
    node.AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z)
    node.Fix(KM.DISPLACEMENT_Z)
    node.SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)

for j in range(ny + 1):
    n = mp.Nodes[node_id(0, j)]
    n.Fix(KM.DISPLACEMENT_X)
    n.SetSolutionStepValue(KM.DISPLACEMENT_X, 0.0)
for i in range(nx + 1):
    n = mp.Nodes[node_id(i, 0)]
    n.Fix(KM.DISPLACEMENT_Y)
    n.SetSolutionStepValue(KM.DISPLACEMENT_Y, 0.0)
right_nodes = [mp.Nodes[node_id(nx, j)] for j in range(ny + 1)]
for n in right_nodes:
    n.Fix(KM.DISPLACEMENT_X)

scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
builder_and_solver = KM.ResidualBasedBlockBuilderAndSolver(
    KM.SkylineLUFactorizationSolver()
)
conv = KM.ResidualCriteria(1.0e-7, 1.0e-10)
strat = KM.ResidualBasedNewtonRaphsonStrategy(
    mp, scheme, conv, builder_and_solver,
    50, True, False, True,
)
strat.SetEchoLevel(0)

vtk_params = KM.Parameters(json.dumps({{
    "model_part_name": "Structure",
    "output_control_type": "step",
    "output_interval": 1,
    "file_format": "ascii",
    "output_path": ".",
    "output_sub_model_parts": False,
    "save_output_files_in_folder": False,
    "nodal_solution_step_data_variables": ["DISPLACEMENT", "REACTION"],
}}))
vtk = KM.VtkOutput(mp, vtk_params)

strat.Check()
force_history = []
for step in range(1, n_steps + 1):
    mp.CloneTimeStep(float(step))
    mp.ProcessInfo[KM.STEP] = step
    u_step = u_max * step / n_steps
    for n in right_nodes:
        n.SetSolutionStepValue(KM.DISPLACEMENT_X, u_step)
    strat.Solve()
    fx = sum(n.GetSolutionStepValue(KM.REACTION_X) for n in right_nodes)
    force_history.append({{"u": u_step, "Fx": fx}})
vtk.PrintOutput()


def gp_max(var):
    return max(
        max(e.CalculateOnIntegrationPoints(var, mp.ProcessInfo))
        for e in mp.Elements
    )


max_eps_p = float(gp_max(CLA.EQUIVALENT_PLASTIC_STRAIN))
max_diss = float(gp_max(KM.PLASTIC_DISSIPATION))
max_uniax = float(gp_max(CLA.UNIAXIAL_STRESS))
n_yielded_gp, n_gp = 0, 0
for elem in mp.Elements:
    for v in elem.CalculateOnIntegrationPoints(
            KM.PLASTIC_DISSIPATION, mp.ProcessInfo):
        n_gp += 1
        if v > 1.0e-12:
            n_yielded_gp += 1

tip = mp.Nodes[node_id(nx, 0)]
F_final = force_history[-1]["Fx"]
summary = {{
    "law": LAW_NAME + " (ConstitutiveLawsApplication)",
    "element": "SmallDisplacementElement2D4N",
    "n_nodes": mp.NumberOfNodes(),
    "n_elements": mp.NumberOfElements(),
    "n_steps": n_steps,
    "applied_strain_x": u_max / L,
    "tip_ux": float(tip.GetSolutionStepValue(KM.DISPLACEMENT_X)),
    "axial_force_final": F_final,
    "axial_force_if_elastic": force_history[0]["Fx"] * n_steps,
    "axial_stress_final": F_final / h,
    "yield_stress": yield_stress,
    "max_equivalent_plastic_strain": max_eps_p,
    "max_plastic_dissipation": max_diss,
    "max_uniaxial_stress": max_uniax,
    "fraction_gauss_points_yielded": n_yielded_gp / n_gp,
    "force_history": force_history,
    "runtime_s": time.time() - t_start,
}}
print(f"plateau stress={{F_final / h:.5g}} Pa (sigma_y={{yield_stress:.5g}}), "
      f"max eq. plastic strain={{max_eps_p:.5f}}, "
      f"max plastic dissipation={{max_diss:.5g}}")
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)

assert max_diss > 0.0 and max_eps_p > 1.0e-4, "specimen did not yield"
'''


def _dem_structures_2d(params: dict) -> str:
    """Spherical DEM particle(s) dropped under gravity onto a clamped elastic
    FEM plate (SmallDisplacementElement3D4N), two-way coupled with
    KratosMultiphysics.DemStructuresCouplingApplication following the official
    dem_fem_coupling_algorithm pattern:

      - SkinDetectionProcess3D -> SurfaceLoadFromDEMCondition3D3N on the skin
      - DemStructuresCouplingUtilities().TransferStructuresSkinToDem
        (skin becomes a DEM rigid-face wall sharing the structural nodes)
      - InterpolateStructuralSolutionForDEM (structure motion -> DEM walls
        during explicit substepping)
      - ComputeDEMFaceLoadUtility (DEM contact tractions -> DEM_SURFACE_LOAD
        applied in the implicit Bossak structural solve)
      - SmoothLoadTrasferredToFem (temporal load relaxation)

    All mdpa / materials files are generated at runtime.  Writes Structure_*.vtk
    (DISPLACEMENT, VELOCITY, REACTION, DEM_SURFACE_LOAD), particles_final.vtk
    and results_summary.json, and hard-fails unless free fall, contact, load
    transfer, plate deflection and rebound are all physically consistent.
    Verified defaults: runtime about 5 s, single OpenMP thread.
    """
    particle_radius = params.get("particle_radius", 0.05)
    n_particles = params.get("n_particles", 1)
    drop_gap = params.get("drop_gap", 0.02)
    plate_lx = params.get("plate_lx", 0.5)
    plate_ly = params.get("plate_ly", 0.5)
    plate_t = params.get("plate_t", 0.05)
    nx = params.get("nx", 10)
    ny = params.get("ny", 10)
    nz = params.get("nz", 2)
    E_plate = params.get("E_plate", 1.0e8)
    nu_plate = params.get("nu_plate", 0.3)
    rho_plate = params.get("rho_plate", 1000.0)
    E_dem = params.get("E_dem", 1.0e7)
    rho_particle = params.get("rho_particle", 2500.0)
    restitution = params.get("restitution", 0.6)
    friction = params.get("friction", 0.3)
    dt_struct = params.get("dt_struct", 1.0e-3)
    dt_dem = params.get("dt_dem", 2.0e-5)
    t_end = params.get("t_end", 0.12)
    return f'''\
"""DEM-FEM coupled impact -- KratosMultiphysics.DemStructuresCouplingApplication.

Generated by _dem_structures_2d().  Sphere(s) of radius {particle_radius} m
dropped from a {drop_gap} m gap onto a clamped {plate_lx} x {plate_ly} x
{plate_t} m elastic plate; two-way DEM-FEM coupling.
"""
import json
import math
import os
import time as walltime
import weakref

# A single OpenMP thread is ~100x faster here: the DEM step on one particle is
# dominated by parallel-region spin overhead when 20 threads are used.
os.environ["OMP_NUM_THREADS"] = os.environ.get("KRATOS_DEMFEM_THREADS", "1")

import KratosMultiphysics as Kratos
import KratosMultiphysics.StructuralMechanicsApplication  # noqa: F401 (registers SMA)
import KratosMultiphysics.DEMApplication as Dem
# NEEDS AN APPLICATION THAT IS NOT IN EVERY KRATOS BUILD.
# Without this guard the script dies on a bare ImportError naming a
# module path, which reads as a broken template rather than a missing
# optional component -- and openPASO knows the difference, because it
# probes what is installed.
try:
    import KratosMultiphysics.DemStructuresCouplingApplication as DemFem
except ImportError as _kratos_app_missing:
    raise SystemExit(
        "This template needs KratosMultiphysics.DemStructuresCouplingApplication (DEM-to-structure coupling), which is NOT present in this Kratos install. It is a real Kratos capability, not a stub, but it is outside the minimal pip stack (StructuralMechanics, ConvectionDiffusion, ContactStructuralMechanics, LinearSolvers). It is published as its own wheel: pip install KratosDemStructuresCouplingApplication, or ask openPASO for a physics your installed applications can solve: discover(query='physics', solver='kratos')."
        + ' [' + str(_kratos_app_missing) + ']')
from KratosMultiphysics.DemStructuresCouplingApplication.dem_main_script_ready_for_coupling_with_fem import (
    StructuresCoupledDEMAnalysisStage,
)
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import (
    StructuralMechanicsAnalysis,
)

# ----------------------------------------------------------------- parameters
particle_radius = {particle_radius} # m
n_particles     = {n_particles} # spheres in a row over the plate centre
drop_gap        = {drop_gap} # m, initial clearance sphere surface <-> plate top
plate_lx        = {plate_lx} # m
plate_ly        = {plate_ly} # m
plate_t         = {plate_t} # m plate thickness
nx, ny, nz      = {nx}, {ny}, {nz} # structured mesh divisions
E_plate         = {E_plate} # Pa
nu_plate        = {nu_plate}
rho_plate       = {rho_plate} # kg/m3
E_dem           = {E_dem} # Pa (particle and wall contact stiffness)
rho_particle    = {rho_particle} # kg/m3
restitution     = {restitution}
friction        = {friction}
dt_struct       = {dt_struct} # s, implicit FEM step (= coupling step)
dt_dem          = {dt_dem} # s, explicit DEM substep
t_end           = {t_end} # s
g               = 9.81      # m/s2
vtk_every       = 20        # structural steps between VTK snapshots

work_dir = (os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
os.chdir(work_dir)

particle_mass = 4.0 / 3.0 * math.pi * particle_radius**3 * rho_particle
z_top = 0.0                          # plate occupies z in [-plate_t, 0]
z0 = z_top + particle_radius + drop_gap

# ------------------------------------------------------- generate input files
def write_structure_mdpa():
    nxn, nyn, nzn = nx + 1, ny + 1, nz + 1

    def nid(i, j, k):
        return 1 + i + j * nxn + k * nxn * nyn

    lines = ["Begin ModelPartData", "End ModelPartData", "",
             "Begin Properties 1", "End Properties", "", "Begin Nodes"]
    all_nodes = []
    for k in range(nzn):
        z = -plate_t + plate_t * k / nz
        for j in range(nyn):
            y = plate_ly * j / ny
            for i in range(nxn):
                x = plate_lx * i / nx
                node_id = nid(i, j, k)
                all_nodes.append(node_id)
                lines.append(f"  {{node_id}} {{x:.10g}} {{y:.10g}} {{z:.10g}}")
    lines += ["End Nodes", "", "Begin Elements SmallDisplacementElement3D4N"]
    eid = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v000, v100 = nid(i, j, k), nid(i + 1, j, k)
                v010, v110 = nid(i, j + 1, k), nid(i + 1, j + 1, k)
                v001, v101 = nid(i, j, k + 1), nid(i + 1, j, k + 1)
                v011, v111 = nid(i, j + 1, k + 1), nid(i + 1, j + 1, k + 1)
                # 6-tet Kuhn split along the v000-v111 diagonal (conforming,
                # positive jacobians)
                for conn in ((v000, v100, v110, v111), (v000, v110, v010, v111),
                             (v000, v010, v011, v111), (v000, v011, v001, v111),
                             (v000, v001, v101, v111), (v000, v101, v100, v111)):
                    eid += 1
                    lines.append("  %d 1 %d %d %d %d" % ((eid,) + conn))
    lines += ["End Elements", ""]
    fixed = [nid(i, j, k) for k in range(nzn) for j in range(nyn)
             for i in range(nxn) if i in (0, nx) or j in (0, ny)]
    lines += ["Begin SubModelPart Parts_plate",
              "Begin SubModelPartNodes"]
    lines += [f"  {{n}}" for n in all_nodes]
    lines += ["End SubModelPartNodes", "Begin SubModelPartElements"]
    lines += [f"  {{e}}" for e in range(1, eid + 1)]
    lines += ["End SubModelPartElements", "End SubModelPart", "",
              "Begin SubModelPart DISPLACEMENT_fixed",
              "Begin SubModelPartNodes"]
    lines += [f"  {{n}}" for n in fixed]
    lines += ["End SubModelPartNodes", "End SubModelPart", ""]
    with open("plate_structure.mdpa", "w") as f:
        f.write("\\n".join(lines))
    return fixed


def write_spheres_mdpa():
    xs = [plate_lx / 2.0 + (i - (n_particles - 1) / 2.0) * 2.5 * particle_radius
          for i in range(n_particles)]
    lines = ["Begin ModelPartData", "End ModelPartData", "",
             "Begin Properties 1", "End Properties", "", "Begin Nodes"]
    for i, x in enumerate(xs):
        lines.append(f"  {{i + 1}} {{x:.10g}} {{plate_ly / 2.0:.10g}} {{z0:.10g}}")
    lines += ["End Nodes", "", "Begin Elements SphericParticle3D"]
    for i in range(n_particles):
        lines.append(f"  {{i + 1}} 1 {{i + 1}}")
    lines += ["End Elements", "", "Begin NodalData RADIUS"]
    for i in range(n_particles):
        lines.append(f"  {{i + 1}} 0 {{particle_radius:.10g}}")
    lines += ["End NodalData", "", "Begin SubModelPart DEMParts_balls",
              "Begin SubModelPartNodes"]
    lines += [f"  {{i + 1}}" for i in range(n_particles)]
    lines += ["End SubModelPartNodes", "Begin SubModelPartElements"]
    lines += [f"  {{i + 1}}" for i in range(n_particles)]
    lines += ["End SubModelPartElements", "End SubModelPart", ""]
    with open("plate_dropDEM.mdpa", "w") as f:
        f.write("\\n".join(lines))


def write_dem_materials():
    relation_vars = {{
        "COEFFICIENT_OF_RESTITUTION": restitution,
        "STATIC_FRICTION": friction,
        "DYNAMIC_FRICTION": friction,
        "FRICTION_DECAY": 500.0,
        "ROLLING_FRICTION": 0.01,
        "ROLLING_FRICTION_WITH_WALLS": 0.01,
        "DEM_DISCONTINUUM_CONSTITUTIVE_LAW_NAME": "DEM_D_Hertz_viscous_Coulomb",
    }}
    mats = {{
        "materials": [
            {{"material_name": "DEM-Material", "material_id": 1,
             "Variables": {{"PARTICLE_DENSITY": rho_particle,
                           "YOUNG_MODULUS": E_dem, "POISSON_RATIO": 0.25}}}},
            {{"material_name": "Wall-Material", "material_id": 2,
             "Variables": {{"YOUNG_MODULUS": E_dem, "POISSON_RATIO": 0.25,
                           "COMPUTE_WEAR": False}}}},
        ],
        "material_relations": [
            {{"material_names_list": ["DEM-Material", "DEM-Material"],
             "material_ids_list": [1, 1], "Variables": relation_vars}},
            {{"material_names_list": ["DEM-Material", "Wall-Material"],
             "material_ids_list": [1, 2], "Variables": relation_vars}},
        ],
        "material_assignation_table": [["SpheresPart", "DEM-Material"]],
    }}
    with open("MaterialsDEM.json", "w") as f:
        json.dump(mats, f, indent=2)


def write_structural_materials():
    mats = {{"properties": [{{
        "model_part_name": "Structure.Parts_plate",
        "properties_id": 1,
        "Material": {{
            "constitutive_law": {{"name": "LinearElastic3DLaw"}},
            "Variables": {{"YOUNG_MODULUS": E_plate, "POISSON_RATIO": nu_plate,
                          "DENSITY": rho_plate}},
            "Tables": {{}},
        }},
    }}]}}
    with open("StructuralMaterials.json", "w") as f:
        json.dump(mats, f, indent=2)


fixed_node_ids = write_structure_mdpa()
write_spheres_mdpa()
write_dem_materials()
write_structural_materials()

dem_parameters = {{
    "Dimension": 3,
    "BoundingBoxOption": False,
    "AutomaticBoundingBoxOption": False,
    "dem_inlet_option": False,
    "GravityX": 0.0, "GravityY": 0.0, "GravityZ": -g,
    "RotationOption": True,
    "CleanIndentationsOption": False,
    "DeltaOption": "Absolute",
    "SearchTolerance": 0.001,
    "search_tolerance_against_walls": 0.001,
    "solver_settings": {{
        "strategy": "sphere_strategy",
        "material_import_settings": {{"materials_filename": "MaterialsDEM.json"}},
    }},
    "creator_destructor_settings": {{}},
    "do_print_results_option": False,
    "post_vtk_option": False,
    # PostElasticForces/... switch COMPUTE_FEM_RESULTS_OPTION on, which makes
    # the DEM strategy accumulate nodal ELASTIC_FORCES/DEM_PRESSURE on the
    # rigid-face (= structure skin) nodes -> needed by ComputeDEMFaceLoadUtility
    "PostElasticForces": True,
    "PostContactForces": True,
    "PostPressure": True,
    "PostNodalArea": True,
    "TranslationalIntegrationScheme": "Symplectic_Euler",
    "RotationalIntegrationScheme": "Direct_Integration",
    "AutomaticTimestep": False,
    "MaxTimeStep": dt_dem,
    "FinalTime": t_end,
    "NeighbourSearchFrequency": 20,
    "ElementType": "SphericPartDEMElement3D",
    "GraphExportFreq": 1.0e10,
    "VelTrapGraphExportFreq": 1.0e10,
    "OutputTimeStep": 1.0e10,
    "echo_level": 0,
    "problem_name": "plate_drop",
}}

structural_parameters = {{
    "problem_data": {{"problem_name": "plate", "parallel_type": "OpenMP",
                     "echo_level": 0, "start_time": 0.0, "end_time": t_end}},
    "solver_settings": {{
        "solver_type": "Dynamic",
        "model_part_name": "Structure",
        "domain_size": 3,
        "echo_level": 0,
        "analysis_type": "non_linear",
        "time_integration_method": "implicit",
        "scheme_type": "bossak",
        "model_import_settings": {{"input_type": "mdpa",
                                  "input_filename": "plate_structure"}},
        "material_import_settings": {{"materials_filename": "StructuralMaterials.json"}},
        "time_stepping": {{"time_step": dt_struct}},
        "rotation_dofs": False,
        "linear_solver_settings": {{"solver_type": "skyline_lu_factorization"}},
    }},
    "processes": {{
        "constraints_process_list": [{{
            "python_module": "assign_vector_variable_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "AssignVectorVariableProcess",
            "Parameters": {{"model_part_name": "Structure.DISPLACEMENT_fixed",
                           "variable_name": "DISPLACEMENT",
                           "constrained": [True, True, True],
                           "value": [0.0, 0.0, 0.0],
                           "interval": [0.0, "End"]}},
        }}],
        "loads_process_list": [],
    }},
    "output_processes": {{}},
}}


# ------------------------------------------------------------ coupled driver
class CoupledDropAlgorithm:
    """dem_fem_coupling_algorithm.Algorithm with VTK output + measurements."""

    def __init__(self):
        self.model = Kratos.Model()
        self.dem_solution = StructuresCoupledDEMAnalysisStage(
            self.model, Kratos.Parameters(json.dumps(dem_parameters)))
        self.dem_solution.coupling_analysis = weakref.proxy(self)
        self.structural_solution = StructuralMechanicsAnalysis(
            self.model, Kratos.Parameters(json.dumps(structural_parameters)))
        self._AddDEMVariablesToStructural()
        # measurement containers
        self.traj = []          # (t, z, vz, total_force_z) every DEM substep
        self.struct_hist = []   # per structural step dicts

    def _AddDEMVariablesToStructural(self):
        mp = self.structural_solution._GetSolver().main_model_part
        for var in (DemFem.DEM_SURFACE_LOAD,
                    DemFem.BACKUP_LAST_STRUCTURAL_VELOCITY,
                    DemFem.BACKUP_LAST_STRUCTURAL_DISPLACEMENT,
                    DemFem.SMOOTHED_STRUCTURAL_VELOCITY,
                    Dem.DELTA_DISPLACEMENT, Dem.DEM_PRESSURE,
                    Dem.DEM_NODAL_AREA, Dem.ELASTIC_FORCES,
                    Dem.CONTACT_FORCES, Dem.TANGENTIAL_ELASTIC_FORCES,
                    Dem.SHEAR_STRESS, Dem.NON_DIMENSIONAL_VOLUME_WEAR,
                    Dem.IMPACT_WEAR):
            mp.AddNodalSolutionStepVariable(var)

    def ReadDemModelParts(self, starting_node_Id=0, starting_elem_Id=0,
                          starting_cond_Id=0):
        creator_destructor = self.dem_solution.creator_destructor
        structures_mp = self.structural_solution._GetSolver().GetComputingModelPart()
        max_node_Id = creator_destructor.FindMaxNodeIdInModelPart(structures_mp)
        max_elem_Id = creator_destructor.FindMaxElementIdInModelPart(structures_mp)
        max_cond_Id = creator_destructor.FindMaxConditionIdInModelPart(structures_mp)
        self.dem_solution.BaseReadModelParts(max_node_Id, max_elem_Id, max_cond_Id)
        self.dem_solution.all_model_parts.MaxNodeId = max_node_Id

    def Initialize(self):
        self.structural_solution.Initialize()   # reads plate_structure.mdpa
        self.dem_solution.Initialize()          # reads plate_dropDEM.mdpa

        self._DetectStructuresSkin()
        self._TransferStructuresSkinToDem()
        self.dem_solution._GetSolver().Initialize()

        vtk_params = Kratos.Parameters(json.dumps({{
            "model_part_name": "Structure",
            "output_control_type": "step",
            "output_interval": 1,
            "file_format": "ascii",
            "output_path": ".",
            "output_sub_model_parts": False,
            "save_output_files_in_folder": False,
            "nodal_solution_step_data_variables":
                ["DISPLACEMENT", "VELOCITY", "REACTION", "DEM_SURFACE_LOAD"],
        }}))
        self.vtk_io = Kratos.VtkOutput(self.structural_mp, vtk_params)
        self.ball_nodes = list(self.dem_solution.spheres_model_part.Nodes)

    def _DetectStructuresSkin(self):
        skin_params = Kratos.Parameters("""{{
            "name_auxiliar_model_part": "DetectedByProcessSkinModelPart",
            "name_auxiliar_condition": "SurfaceLoadFromDEMCondition",
            "list_model_parts_to_assign_conditions": []
        }}""")
        self.structural_mp = self.structural_solution._GetSolver().GetComputingModelPart()
        Kratos.SkinDetectionProcess3D(self.structural_mp, skin_params).Execute()
        self.skin_mp = self.structural_mp.GetSubModelPart("DetectedByProcessSkinModelPart")

    def _TransferStructuresSkinToDem(self):
        dem_walls_mp = self.dem_solution.rigid_face_model_part.CreateSubModelPart(
            "SkinTransferredFromStructure")
        # id 2 matches the sphere<->wall material relation [1,2] in MaterialsDEM
        props = Kratos.Properties(2)
        props[Dem.STATIC_FRICTION] = friction
        props[Dem.DYNAMIC_FRICTION] = friction
        props[Dem.WALL_COHESION] = 0.0
        props[Dem.COMPUTE_WEAR] = False
        props[Dem.SEVERITY_OF_WEAR] = 0.001
        props[Dem.IMPACT_WEAR_SEVERITY] = 0.001
        props[Dem.BRINELL_HARDNESS] = 200.0
        props[Kratos.YOUNG_MODULUS] = E_dem
        props[Kratos.POISSON_RATIO] = 0.25
        dem_walls_mp.AddProperties(props)
        # DEM's InitializeFEMElements requires the wall submodelpart to carry
        # PROPERTIES_ID (normally set through the materials assignation table)
        dem_walls_mp.SetValue(Dem.PROPERTIES_ID, 2)
        DemFem.DemStructuresCouplingUtilities().TransferStructuresSkinToDem(
            self.skin_mp, dem_walls_mp, props)

    def yield_DEM_time(self, current_time, current_time_plus_increment, delta_time):
        current_time += delta_time
        tolerance = 0.0001
        while current_time < (current_time_plus_increment - tolerance * delta_time):
            yield current_time
            current_time += delta_time
        yield current_time_plus_increment

    def RunSolutionLoop(self):
        ds = self.dem_solution
        ss = self.structural_solution
        ds.step, ds.time, ds.time_old_print = 0, 0.0, 0.0
        Dt_struct = ss._GetSolver().settings["time_stepping"]["time_step"].GetDouble()
        struct_step = 0

        while ss.time < ss.end_time:
            DemFem.DemStructuresCouplingUtilities().SmoothLoadTrasferredToFem(
                ds.rigid_face_model_part, 0.4)

            ss.time = ss._GetSolver().AdvanceInTime(ss.time)
            ss.InitializeSolutionStep()
            ss._GetSolver().Predict()
            ss._GetSolver().SolveSolutionStep()
            ss.FinalizeSolutionStep()
            ss.OutputSolutionStep()
            struct_step += 1

            t_final = ss.time
            Dt_DEM = ds.spheres_model_part.ProcessInfo.GetValue(Kratos.DELTA_TIME)
            DemFem.InterpolateStructuralSolutionForDEM().SaveStructuralSolution(
                self.structural_mp)
            DemFem.ComputeDEMFaceLoadUtility().ClearDEMFaceLoads(self.skin_mp)

            for _ in self.yield_DEM_time(ds.time, t_final, Dt_DEM):
                ds.time = ds.time + ds._GetSolver().dt
                ds.step += 1
                # C++ fast path (the python DEMFEMProcedures variant is
                # deprecated and prints a warning every substep)
                ds._GetSolver()._UpdateTimeInModelParts(ds.time)
                ds.InitializeSolutionStep()
                ds._GetSolver().Predict()
                DemFem.InterpolateStructuralSolutionForDEM().InterpolateStructuralSolution(
                    self.structural_mp, Dt_struct, ss.time,
                    ds._GetSolver().dt, ds.time)
                ds.SolverSolve()
                ds.FinalizeSolutionStep()
                DemFem.ComputeDEMFaceLoadUtility().CalculateDEMFaceLoads(
                    self.skin_mp, ds._GetSolver().dt, Dt_struct)
                n = self.ball_nodes[0]
                self.traj.append((ds.time, n.Z,
                                  n.GetSolutionStepValue(Kratos.VELOCITY_Z),
                                  n.GetSolutionStepValue(Kratos.TOTAL_FORCES_Z)))

            DemFem.InterpolateStructuralSolutionForDEM().RestoreStructuralSolution(
                self.structural_mp)

            self._RecordStructuralStep(ss.time)
            if struct_step % vtk_every == 0:
                self.vtk_io.PrintOutput()

        self.vtk_io.PrintOutput()

    def _RecordStructuralStep(self, t):
        min_uz, sum_rz = 0.0, 0.0
        sum_dem_load_z, dem_force_z = 0.0, 0.0
        fixed = set(fixed_node_ids)
        for node in self.structural_mp.Nodes:
            uz = node.GetSolutionStepValue(Kratos.DISPLACEMENT_Z)
            min_uz = min(min_uz, uz)
            if node.Id in fixed:
                sum_rz += node.GetSolutionStepValue(Kratos.REACTION_Z)
        for node in self.skin_mp.Nodes:
            lz = node.GetSolutionStepValue(DemFem.DEM_SURFACE_LOAD_Z)
            sum_dem_load_z += lz
            # DEM_SURFACE_LOAD is a traction [N/m2]; integrate with the DEM
            # nodal area to recover the transferred force
            dem_force_z += lz * node.GetSolutionStepValue(Dem.DEM_NODAL_AREA)
        center = self._CenterTopNode()
        self.struct_hist.append({{
            "t": t,
            "min_uz": min_uz,
            "center_uz": center.GetSolutionStepValue(Kratos.DISPLACEMENT_Z),
            "sum_reaction_z": sum_rz,
            "sum_dem_surface_load_z": sum_dem_load_z,
            "dem_transferred_force_z": dem_force_z,
        }})

    def _CenterTopNode(self):
        if not hasattr(self, "_center_node"):
            cx, cy = plate_lx / 2.0, plate_ly / 2.0
            best, best_d = None, 1.0e30
            for node in self.structural_mp.Nodes:
                d = (node.X0 - cx)**2 + (node.Y0 - cy)**2 + (node.Z0 - z_top)**2
                if d < best_d:
                    best, best_d = node, d
            self._center_node = best
        return self._center_node

    def Finalize(self):
        self.dem_solution.Finalize()
        self.structural_solution.Finalize()


t_start_wall = walltime.time()
algo = CoupledDropAlgorithm()
algo.Initialize()
algo.RunSolutionLoop()
algo.Finalize()
runtime = walltime.time() - t_start_wall

# ------------------------------------------------------------------ analysis
traj = algo.traj
baseline_fz = traj[0][3]   # TOTAL_FORCES_Z during free fall (gravity only)
contact_idx = None
for i, (t, z, vz, fz) in enumerate(traj):
    if abs(fz - baseline_fz) > max(1.0e-3, 0.02 * abs(baseline_fz)):
        contact_idx = i
        break
if contact_idx is None:
    raise RuntimeError("FAILURE: particle never contacted the plate "
                       "(no contact force detected)")

t_c, z_c, v_imp, _ = traj[contact_idx - 1]
# free-fall verification at the last pre-contact sample
z_analytic = z0 - 0.5 * g * t_c**2
v_analytic = -g * t_c
ff_err = abs(z_c - z_analytic) / (z0 - z_analytic + 1.0e-30)
contact_force_peak = max(abs(fz - baseline_fz) for (_, _, _, fz) in traj)
min_z = min(z for (_, z, _, _) in traj)
max_z_after_contact = max(z for (t, z, _, _) in traj[contact_idx:])
rebound_height = max_z_after_contact - min_z
# outgoing velocity: first sample after contact force vanished again
v_out = 0.0
for (t, z, vz, fz) in traj[contact_idx:]:
    if abs(fz - baseline_fz) < max(1.0e-3, 0.02 * abs(baseline_fz)) and vz > 0.0:
        v_out = vz
        break

max_deflection = min(h["min_uz"] for h in algo.struct_hist)
peak_dem_load = min(h["sum_dem_surface_load_z"] for h in algo.struct_hist)
peak_dem_force = min(h["dem_transferred_force_z"] for h in algo.struct_hist)
peak_reaction = max(h["sum_reaction_z"] for h in algo.struct_hist)
final_center_uz = algo.struct_hist[-1]["center_uz"]
# reaction impulse over the whole run (trapezoid, uniform dt_struct)
reaction_impulse = sum(h["sum_reaction_z"] for h in algo.struct_hist) * dt_struct

# ------------------------------------------------------- honesty hard checks
checks = {{
    "free_fall_rel_err_below_2pct": ff_err < 0.02,
    "impact_velocity_matches_sqrt_2gh": abs(abs(v_imp) - math.sqrt(2 * g * drop_gap))
                                        / math.sqrt(2 * g * drop_gap) < 0.05,
    "contact_force_nonzero": contact_force_peak > 1.0,
    "plate_deflects_downward": max_deflection < -1.0e-7,
    "particle_rebounds": rebound_height > 1.0e-3 and v_out > 0.0,
    "dem_load_transferred_to_fem": peak_dem_load < -1.0,
}}
if not all(checks.values()):
    raise RuntimeError(f"FAILURE: physics sanity checks failed: {{checks}}")

# ------------------------------------------------------------ particle .vtk
with open("particles_final.vtk", "w") as f:
    n = len(algo.ball_nodes)
    f.write("# vtk DataFile Version 3.0\\nDEM particles (final state)\\nASCII\\n"
            "DATASET UNSTRUCTURED_GRID\\n")
    f.write(f"POINTS {{n}} float\\n")
    for node in algo.ball_nodes:
        f.write(f"{{node.X}} {{node.Y}} {{node.Z}}\\n")
    f.write(f"CELLS {{n}} {{2 * n}}\\n")
    for i in range(n):
        f.write(f"1 {{i}}\\n")
    f.write(f"CELL_TYPES {{n}}\\n" + "1\\n" * n)
    f.write(f"POINT_DATA {{n}}\\nSCALARS RADIUS float 1\\nLOOKUP_TABLE default\\n")
    for node in algo.ball_nodes:
        f.write(f"{{node.GetSolutionStepValue(Kratos.RADIUS)}}\\n")
    f.write("VECTORS VELOCITY float\\n")
    for node in algo.ball_nodes:
        v = node.GetSolutionStepValue(Kratos.VELOCITY)
        f.write(f"{{v[0]}} {{v[1]}} {{v[2]}}\\n")

summary = {{
    "problem": "DEM sphere dropped on clamped elastic FEM plate "
               "(two-way DemStructuresCoupling)",
    "parameters": {{
        "particle_radius_m": particle_radius, "n_particles": n_particles,
        "particle_mass_kg": particle_mass, "drop_gap_m": drop_gap,
        "plate_m": [plate_lx, plate_ly, plate_t],
        "E_plate_Pa": E_plate, "E_dem_Pa": E_dem,
        "rho_particle": rho_particle, "rho_plate": rho_plate,
        "restitution_input": restitution,
        "dt_struct_s": dt_struct, "dt_dem_s": dt_dem, "t_end_s": t_end,
    }},
    "free_fall_check": {{
        "t_last_precontact_s": t_c, "z_measured_m": z_c,
        "z_analytic_m": z_analytic, "rel_error": ff_err,
        "v_impact_measured_m_s": v_imp, "v_impact_analytic_m_s": v_analytic,
        "v_impact_sqrt_2gh_m_s": -math.sqrt(2 * g * drop_gap),
    }},
    "contact": {{
        "first_contact_time_s": traj[contact_idx][0],
        "peak_contact_force_on_particle_N": contact_force_peak,
        "peak_dem_surface_load_sum_z_N_per_m2": peak_dem_load,
        "peak_transferred_force_z_N": peak_dem_force,
        "static_particle_weight_N": particle_mass * g,
    }},
    "structure_response": {{
        "max_plate_deflection_m": max_deflection,
        "final_center_deflection_m": final_center_uz,
        "peak_sum_reaction_z_N": peak_reaction,
        "reaction_impulse_Ns": reaction_impulse,
        "particle_momentum_change_Ns": particle_mass * (abs(v_imp) + v_out),
    }},
    "rebound": {{
        "min_particle_center_z_m": min_z,
        "max_z_after_contact_m": max_z_after_contact,
        "rebound_height_m": rebound_height,
        "v_out_m_s": v_out,
        "apparent_restitution": v_out / abs(v_imp) if v_imp else None,
    }},
    "checks": checks,
    "n_dem_steps": len(traj),
    "n_struct_steps": len(algo.struct_hist),
    "n_structure_nodes": algo.structural_mp.NumberOfNodes(),
    "n_skin_conditions": algo.skin_mp.NumberOfConditions(),
    "runtime_s": runtime,
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("COUPLED DEM-FEM SUMMARY")
print(f"  free-fall rel. err          : {{ff_err:.3e}}")
print(f"  impact velocity             : {{v_imp:.4f}} (analytic "
      f"{{-math.sqrt(2 * g * drop_gap):.4f}}) m/s")
print(f"  peak contact force          : {{contact_force_peak:.2f}} N")
print(f"  peak transferred DEM force  : {{peak_dem_force:.2f}} N "
      f"(traction sum {{peak_dem_load:.1f}} N/m2)")
print(f"  max plate deflection        : {{max_deflection:.3e}} m")
print(f"  rebound height              : {{rebound_height:.4f}} m, "
      f"v_out {{v_out:.3f}} m/s")
print(f"  runtime                     : {{runtime:.1f}} s")
'''


def _cable_net_2d(params: dict) -> str:
    """Minimal 3D cable net — Kratos CableNetApplication (pip 10.4.2).

    Two `SlidingCableElement3D3N` cables cross at one shared frictionless
    slider node which is pulled down by a prescribed, ramped displacement;
    an `EmpiricalSpringElement3D2N` (linear polynomial, highest-degree-
    first) drags the slider off-center in x.  Both element types are
    registered by CableNetApplication itself and both carry load.

    Verified pitfalls baked in:
    - SlidingCableElement3D3N needs CONSTITUTIVE_LAW (TrussConstitutiveLaw),
      CROSS_AREA and DENSITY on the Properties.
    - strategy.Check() before strategy.Initialize() SEGFAULTS (the sliding
      cable's Check dereferences the constitutive law cloned in Initialize).
    - VtkOutput cannot handle the sliding cable's generic geometry; output
      goes through a visualization SubModelPart of 2-node line elements.
    - EmpiricalSpringElement3D2N has a uBLAS aliasing bug in its RHS
      (GlobalizeVector); the force is only exact when the element axis is
      global-x-aligned — guaranteed here at the converged state by placing
      the spring anchor at the slider's prescribed final height.
    - Keep the cables taut during the ramp (sag only increases), otherwise
      the element silently drops stiffness (mIsCompressed).
    """
    span_x = params.get("span_x", 3.0)          # cable X span
    half_span_z = params.get("half_span_z", 1.5)  # cable Z half span
    x0_s = params.get("x0_s", 1.0)              # initial slider x (off-center)
    y0_s = params.get("y0_s", -0.5)             # initial slider sag
    y_final = params.get("y_final", -0.8)       # prescribed final sag (load)
    E = params.get("E", 1.0e8)                  # Young's modulus
    A = params.get("A", 1.0e-4)                 # cable cross-section area
    rho = params.get("rho", 7850.0)
    k_spring = params.get("k_spring", 1.0e3)    # linear spring constant
    xG = params.get("xG", -1.0)                 # spring anchor x position
    n_steps = params.get("n_steps", 8)          # displacement ramp steps
    return f'''\
"""Minimal 3D cable net — Kratos CableNetApplication.

Two crossing SlidingCableElement3D3N cables share one frictionless slider
node S, pulled down by a ramped prescribed displacement; an
EmpiricalSpringElement3D2N drags the slider off-center in x.
Writes vis_0_{n_steps}.vtk (DISPLACEMENT, REACTION) and results_summary.json.
"""
import json
import math
import time
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
# NEEDS AN APPLICATION THAT IS NOT IN EVERY KRATOS BUILD.
# Without this guard the script dies on a bare ImportError naming a
# module path, which reads as a broken template rather than a missing
# optional component -- and openPASO knows the difference, because it
# probes what is installed.
try:
    import KratosMultiphysics.CableNetApplication as CNA
except ImportError as _kratos_app_missing:
    raise SystemExit(
        "This template needs KratosMultiphysics.CableNetApplication (cable-net elements), which is NOT present in this Kratos install. It is a real Kratos capability, not a stub, but it is outside the minimal pip stack (StructuralMechanics, ConvectionDiffusion, ContactStructuralMechanics, LinearSolvers). It is published as its own wheel: pip install KratosCableNetApplication, or ask openPASO for a physics your installed applications can solve: discover(query='physics', solver='kratos')."
        + ' [' + str(_kratos_app_missing) + ']')

t_start = time.time()

span_x = {span_x}
half_span_z = {half_span_z}
x0_s, y0_s = {x0_s}, {y0_s}
y_final = {y_final}
E, A, rho = {E}, {A}, {rho}
k_spring = {k_spring}
xG = {xG}
n_steps = {n_steps}

model = KM.Model()
mp = model.CreateModelPart("Structure")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
mp.SetBufferSize(2)
for v in (KM.DISPLACEMENT, KM.REACTION, KM.VOLUME_ACCELERATION):
    mp.AddNodalSolutionStepVariable(v)

nA = mp.CreateNewNode(1, 0.0, 0.0, 0.0)
nS = mp.CreateNewNode(2, x0_s, y0_s, 0.0)
nB = mp.CreateNewNode(3, span_x, 0.0, 0.0)
nC = mp.CreateNewNode(4, 0.5 * span_x, 0.0, -half_span_z)
nD = mp.CreateNewNode(5, 0.5 * span_x, 0.0, half_span_z)
nG = mp.CreateNewNode(6, xG, y_final, 0.0)

prop_cable = mp.CreateNewProperties(1)
prop_cable.SetValue(KM.YOUNG_MODULUS, E)
prop_cable.SetValue(KM.DENSITY, rho)
prop_cable.SetValue(SMA.CROSS_AREA, A)
prop_cable.SetValue(KM.CONSTITUTIVE_LAW, SMA.TrussConstitutiveLaw())

prop_spring = mp.CreateNewProperties(2)
prop_spring.SetValue(KM.DENSITY, rho)
prop_spring.SetValue(SMA.CROSS_AREA, A)
poly = KM.Vector(2)
poly[0] = k_spring
poly[1] = 0.0
prop_spring.SetValue(CNA.SPRING_DEFORMATION_EMPIRICAL_POLYNOMIAL, poly)

mp.CreateNewElement("SlidingCableElement3D3N", 1, [1, 2, 3], prop_cable)
mp.CreateNewElement("SlidingCableElement3D3N", 2, [4, 2, 5], prop_cable)
mp.CreateNewElement("EmpiricalSpringElement3D2N", 3, [2, 6], prop_spring)

for node in mp.Nodes:
    node.AddDof(KM.DISPLACEMENT_X, KM.REACTION_X)
    node.AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y)
    node.AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z)

for nid in (1, 3, 4, 5, 6):
    n = mp.Nodes[nid]
    for var in (KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y, KM.DISPLACEMENT_Z):
        n.Fix(var)
        n.SetSolutionStepValue(var, 0.0)
nS.Fix(KM.DISPLACEMENT_Y)

scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
builder_and_solver = KM.ResidualBasedBlockBuilderAndSolver(
    KM.SkylineLUFactorizationSolver())
conv = KM.ResidualCriteria(1.0e-10, 1.0e-12)
strat = KM.ResidualBasedNewtonRaphsonStrategy(
    mp, scheme, conv, builder_and_solver, 60, True, False, True)
strat.SetEchoLevel(0)
strat.Initialize()  # MUST precede Check(): sliding cable Check segfaults otherwise
strat.Check()

for step in range(1, n_steps + 1):
    mp.CloneTimeStep(float(step))
    mp.ProcessInfo[KM.STEP] = step
    nS.SetSolutionStepValue(
        KM.DISPLACEMENT_Y, (y_final - y0_s) * step / n_steps)
    strat.Solve()

# VtkOutput cannot handle the sliding cable's generic geometry ->
# visualization SubModelPart with plain 2-node lines.
vis = mp.CreateSubModelPart("vis")
vis.AddNodes([1, 2, 3, 4, 5, 6])
prop_vis = mp.CreateNewProperties(99)
for eid, conn in enumerate(([1, 2], [2, 3], [4, 2], [2, 5], [2, 6]), start=101):
    vis.CreateNewElement("Element3D2N", eid, conn, prop_vis)

vtk_params = KM.Parameters(json.dumps({{
    "model_part_name": "Structure.vis",
    "output_control_type": "step",
    "output_interval": 1,
    "file_format": "ascii",
    "output_path": ".",
    "output_sub_model_parts": False,
    "save_output_files_in_folder": False,
    "nodal_solution_step_data_variables": ["DISPLACEMENT", "REACTION"],
}}))
KM.VtkOutput(vis, vtk_params).PrintOutput()

ux = nS.GetSolutionStepValue(KM.DISPLACEMENT_X)
uz = nS.GetSolutionStepValue(KM.DISPLACEMENT_Z)
xs, ys, zs = x0_s + ux, y0_s + nS.GetSolutionStepValue(KM.DISPLACEMENT_Y), uz
ry_slider = nS.GetSolutionStepValue(KM.REACTION_Y)
RA = [nA.GetSolutionStepValue(v) for v in (KM.REACTION_X, KM.REACTION_Y, KM.REACTION_Z)]
RB = [nB.GetSolutionStepValue(v) for v in (KM.REACTION_X, KM.REACTION_Y, KM.REACTION_Z)]
RC = [nC.GetSolutionStepValue(v) for v in (KM.REACTION_X, KM.REACTION_Y, KM.REACTION_Z)]
RD = [nD.GetSolutionStepValue(v) for v in (KM.REACTION_X, KM.REACTION_Y, KM.REACTION_Z)]
RG = [nG.GetSolutionStepValue(v) for v in (KM.REACTION_X, KM.REACTION_Y, KM.REACTION_Z)]
norm = lambda v: math.sqrt(sum(c * c for c in v))

# analytic cross-check (frictionless sliding => equal segment tension,
# N = E*A*e_GL*L/L0 with Green-Lagrange strain of the TOTAL cable length)
L0x = math.hypot(x0_s, y0_s) + math.hypot(span_x - x0_s, y0_s)
L0z = 2.0 * math.sqrt((0.5 * span_x - x0_s) ** 2 + y0_s ** 2 + half_span_z ** 2)
l0s = math.hypot(x0_s - xG, y0_s - y_final)

def cable_force(L, L0):
    e_gl = (L * L - L0 * L0) / (2.0 * L0 * L0)
    return E * A * e_gl * L / L0

def slider_x_residual(x):
    h = -y_final
    l1 = math.hypot(x, h)
    l2 = math.hypot(span_x - x, h)
    Nx = cable_force(l1 + l2, L0x)
    lz = math.sqrt((0.5 * span_x - x) ** 2 + h * h + half_span_z ** 2)
    Nz = cable_force(2.0 * lz, L0z)
    Fs = k_spring * ((x - xG) - l0s)
    return Nx * (-x / l1 + (span_x - x) / l2) + Nz * 2.0 * (0.5 * span_x - x) / lz - Fs

a, b = xG + 1.0e-6, 0.5 * span_x
for _ in range(200):
    m = 0.5 * (a + b)
    if slider_x_residual(a) * slider_x_residual(m) <= 0.0:
        b = m
    else:
        a = m
x_an = 0.5 * (a + b)

h = -y_final
l1 = math.hypot(x_an, h)
l2 = math.hypot(span_x - x_an, h)
Nx_an = cable_force(l1 + l2, L0x)
lz = math.sqrt((0.5 * span_x - x_an) ** 2 + h * h + half_span_z ** 2)
Nz_an = cable_force(2.0 * lz, L0z)
Fs_an = k_spring * ((x_an - xG) - l0s)
Ry_up_an = Nx_an * (h / l1 + h / l2) + Nz_an * 2.0 * h / lz

Nx_fem_A, Nx_fem_B = norm(RA), norm(RB)
Nz_fem_C, Nz_fem_D = norm(RC), norm(RD)
Fs_fem = norm(RG)

runtime = time.time() - t_start
summary = {{
    "elements": ["SlidingCableElement3D3N x2", "EmpiricalSpringElement3D2N x1"],
    "slider_pos_fem": [xs, ys, zs],
    "slider_x_analytic": x_an,
    "slider_x_rel_err": abs(xs - x_an) / abs(x_an),
    "slider_reaction_y_fem": ry_slider,
    "slider_pulldown_force_analytic": Ry_up_an,
    "pulldown_rel_err": abs(abs(ry_slider) - Ry_up_an) / Ry_up_an,
    "cable_x_tension_analytic": Nx_an,
    "cable_x_tension_fem_anchorA": Nx_fem_A,
    "cable_x_tension_fem_anchorB": Nx_fem_B,
    "cable_z_tension_analytic": Nz_an,
    "cable_z_tension_fem_anchorC": Nz_fem_C,
    "cable_z_tension_fem_anchorD": Nz_fem_D,
    "spring_force_analytic": Fs_an,
    "spring_force_fem_anchorG": Fs_fem,
    "spring_elongation": (xs - xG) - l0s,
    "n_nodes": mp.NumberOfNodes(),
    "n_elements": mp.NumberOfElements(),
    "n_load_steps": n_steps,
    "runtime_s": runtime,
}}
print(f"slider: fem x={{xs:.6f}} (analytic {{x_an:.6f}}), y={{ys:.4f}}, z={{zs:.2e}}")
print(f"cable X tension: fem {{Nx_fem_A:.3f}}/{{Nx_fem_B:.3f}}  analytic {{Nx_an:.3f}}")
print(f"cable Z tension: fem {{Nz_fem_C:.3f}}/{{Nz_fem_D:.3f}}  analytic {{Nz_an:.3f}}")
print(f"spring force:    fem {{Fs_fem:.3f}}  analytic {{Fs_an:.3f}}")
print(f"pull-down force: fem {{abs(ry_slider):.3f}}  analytic {{Ry_up_an:.3f}}")
print(f"runtime: {{runtime:.2f}} s")

assert abs(xs - x_an) < 1e-4, "slider x mismatch"
assert abs(zs) < 1e-8, "slider z should remain 0 by symmetry"
assert abs(abs(ry_slider) - Ry_up_an) / Ry_up_an < 1e-4, "pull-down mismatch"
assert abs(Nx_fem_A - Nx_fem_B) / Nx_an < 1e-6, "unequal tension in cable X"
assert abs(Nx_fem_A - Nx_an) / Nx_an < 1e-4, "cable X tension mismatch"
assert abs(Nz_fem_C - Nz_an) / Nz_an < 1e-4, "cable Z tension mismatch"
assert abs(Fs_fem - Fs_an) / Fs_an < 1e-4, "spring force mismatch"

with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("OK")
'''


def _optimization_2d(params: dict) -> str:
    """Compliance (linear strain energy) optimization — Kratos OptimizationApplication.

    Plane-stress cantilever (SmallDisplacementElement2D4N quad grid,
    LinearElasticPlaneStress2DLaw, tip PointLoadCondition2D1N).  Uses the
    compiled OptimizationApplication module:
      * `KOA.OptimizationUtils.CreateEntitySpecificPropertiesForContainer`
        to give every element its own THICKNESS design variable,
      * `KOA.ResponseUtils.LinearStrainEnergyResponseUtils` for the
        compliance value and the semi-analytic d(compliance)/d(thickness),
      * `KOA.ResponseUtils.MassResponseUtils` for the mass constraint and
        its gradient,
      * `Kratos.TensorAdaptors` (VariableTensorAdaptor wrapped in a
        DoubleCombinedTensorAdaptor over the element container — same
        construction as MasterControl.GetPhysicalKratosVariableMap) to
        receive the gradients. This needs Kratos >= 10.4.2: TensorAdaptors
        does not exist in 10.3.x, and in 10.3.0 and 10.4.0 CalculateGradient
        takes list_of_container_expressions instead.

    The script verifies the gradient against a forward finite difference
    (asserts rel. error < 1e-4) and runs `n_iter` mass-constrained projected
    steepest-descent steps (asserts monotone objective decrease and constant
    mass).  Writes `Structure_0_*.vtk` (nodal DISPLACEMENT/REACTION, element
    THICKNESS and THICKNESS_SENSITIVITY) and `results_summary.json`.

    Pitfall: the Newton-Raphson strategy MUST be built with
    MoveMeshFlag=False, otherwise the deformed coordinates corrupt the
    geometric responses (mass evaluated on the deformed mesh).
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", 5)
    lx = params.get("lx", 4.0)
    ly = params.get("ly", 1.0)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    t0 = params.get("t0", 1.0)
    rho = params.get("rho", 1.0)
    load = params.get("load", -10.0)
    n_iter = params.get("n_iter", 5)
    step_size = params.get("step_size", 0.15)
    t_min = params.get("t_min", 0.2)
    t_max = params.get("t_max", 3.0)
    return f'''\
"""Compliance + thickness-sensitivity + projected descent — Kratos OptimizationApplication.

2D plane-stress cantilever; per-element THICKNESS is the design variable.
Verifies the KOA semi-analytic gradient against finite differences and runs a
mass-constrained projected steepest descent (objective decreases monotonically).
"""
import json
import time
import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
# NEEDS AN APPLICATION THAT IS NOT IN EVERY KRATOS BUILD.
# Without this guard the script dies on a bare ImportError naming a
# module path, which reads as a broken template rather than a missing
# optional component -- and openPASO knows the difference, because it
# probes what is installed.
try:
    import KratosMultiphysics.OptimizationApplication as KOA
except ImportError as _kratos_app_missing:
    raise SystemExit(
        "This template needs KratosMultiphysics.OptimizationApplication (the optimization driver), which is NOT present in this Kratos install. It is a real Kratos capability, not a stub, but it is outside the minimal pip stack (StructuralMechanics, ConvectionDiffusion, ContactStructuralMechanics, LinearSolvers). It is published as its own wheel: pip install KratosOptimizationApplication (this template uses the Kratos >= 10.4.2 gradient API), or ask openPASO for a physics your installed applications can solve: discover(query='physics', solver='kratos')."
        + ' [' + str(_kratos_app_missing) + ']')

t_start = time.perf_counter()

nx, ny = {nx}, {ny}
L, h = {lx}, {ly}
E, nu = {E}, {nu}
t0 = {t0}
rho = {rho}
P = {load}
n_iter = {n_iter}
step_size = {step_size}
t_min, t_max = {t_min}, {t_max}
fd_h = 1.0e-6


def node_id(i, j):
    return 1 + j * (nx + 1) + i


model = KM.Model()
mp = model.CreateModelPart("Structure")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
mp.SetBufferSize(2)
for v in (KM.DISPLACEMENT, KM.REACTION, KM.VOLUME_ACCELERATION, SMA.POINT_LOAD):
    mp.AddNodalSolutionStepVariable(v)

for j in range(ny + 1):
    for i in range(nx + 1):
        mp.CreateNewNode(node_id(i, j), i * L / nx, -h / 2.0 + j * h / ny, 0.0)

prop = mp.CreateNewProperties(1)
prop.SetValue(KM.YOUNG_MODULUS, E)
prop.SetValue(KM.POISSON_RATIO, nu)
prop.SetValue(KM.THICKNESS, t0)
prop.SetValue(KM.DENSITY, rho)
prop.SetValue(KM.CONSTITUTIVE_LAW, SMA.LinearElasticPlaneStress2DLaw())

eid = 1
for j in range(ny):
    for i in range(nx):
        mp.CreateNewElement(
            "SmallDisplacementElement2D4N", eid,
            [node_id(i, j), node_id(i + 1, j),
             node_id(i + 1, j + 1), node_id(i, j + 1)],
            prop)
        eid += 1

j_mid = ny // 2
tip = mp.Nodes[node_id(nx, j_mid)]
mp.CreateNewCondition("PointLoadCondition2D1N", 1, [tip.Id], mp.CreateNewProperties(2))
tip.SetSolutionStepValue(SMA.POINT_LOAD, [0.0, P, 0.0])

for node in mp.Nodes:
    node.AddDof(KM.DISPLACEMENT_X, KM.REACTION_X)
    node.AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y)
    node.AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z)
    node.Fix(KM.DISPLACEMENT_Z)
    node.SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)

for j in range(ny + 1):
    n = mp.Nodes[node_id(0, j)]
    n.Fix(KM.DISPLACEMENT_X)
    n.Fix(KM.DISPLACEMENT_Y)
    n.SetSolutionStepValue(KM.DISPLACEMENT_X, 0.0)
    n.SetSolutionStepValue(KM.DISPLACEMENT_Y, 0.0)

# every element gets its own Properties clone -> per-element THICKNESS design var
KOA.OptimizationUtils.CreateEntitySpecificPropertiesForContainer(mp, mp.Elements, False)

# MoveMeshFlag (last ctor argument) MUST be False: geometric responses such as
# mass are otherwise evaluated on the deformed configuration.
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
bs = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
conv = KM.ResidualCriteria(1.0e-8, 1.0e-12)
strat = KM.ResidualBasedNewtonRaphsonStrategy(mp, scheme, conv, bs, 20, True, False, False)
strat.SetEchoLevel(0)
strat.Check()

solve_counter = [0]


def solve_primal():
    solve_counter[0] += 1
    mp.CloneTimeStep(float(solve_counter[0]))
    mp.ProcessInfo[KM.STEP] = solve_counter[0]
    strat.Solve()


LSE = KOA.ResponseUtils.LinearStrainEnergyResponseUtils
MASS = KOA.ResponseUtils.MassResponseUtils
MASS.Check(mp)


def thickness_gradient(response_utils):
    ta = KM.TensorAdaptors.DoubleTensorAdaptor(
        KM.TensorAdaptors.VariableTensorAdaptor(mp.Elements, KM.THICKNESS), copy=False)
    cta = KM.TensorAdaptors.DoubleCombinedTensorAdaptor(
        [ta], perform_collect_data_recursively=False,
        perform_store_data_recursively=False)
    response_utils.CalculateGradient(KM.THICKNESS, mp, cta, 1e-8)
    return np.array(cta.data, copy=True)


solve_primal()
J0 = LSE.CalculateValue(mp)
M0 = MASS.CalculateValue(mp)
tip_uy0 = float(tip.GetSolutionStepValue(KM.DISPLACEMENT_Y))
g0 = thickness_gradient(LSE)

# verification 1: finite-difference check of dJ/dt on element 1
elem1 = mp.Elements[1]
elem1.Properties.SetValue(KM.THICKNESS, t0 + fd_h)
solve_primal()
J_pert = LSE.CalculateValue(mp)
elem1.Properties.SetValue(KM.THICKNESS, t0)
solve_primal()
fd_grad = (J_pert - J0) / fd_h
fd_rel_err = abs(fd_grad - g0[0]) / abs(fd_grad)
print(f"FD check (elem 1): analytic dJ/dt = {{g0[0]:.8f}}, FD = {{fd_grad:.8f}}, "
      f"rel. err = {{fd_rel_err:.3e}}")
assert fd_rel_err < 1e-4, "semi-analytic gradient disagrees with finite difference"

# verification 2: mass-constrained projected steepest descent
elems = list(mp.Elements)
t_field = np.full(len(elems), t0)
obj_history = [J0]
mass_history = [M0]
for it in range(n_iter):
    g = thickness_gradient(LSE)
    gm = thickness_gradient(MASS)
    d = -(g - (g @ gm) / (gm @ gm) * gm)
    d /= np.max(np.abs(d))
    t_field = np.clip(t_field + step_size * d, t_min, t_max)
    for e, te in zip(elems, t_field):
        e.Properties.SetValue(KM.THICKNESS, float(te))
    solve_primal()
    obj_history.append(LSE.CalculateValue(mp))
    mass_history.append(MASS.CalculateValue(mp))
    print(f"iter {{it + 1}}: compliance = {{obj_history[-1]:.6f}}, "
          f"mass = {{mass_history[-1]:.6f}}")

monotone = all(b < a for a, b in zip(obj_history, obj_history[1:]))
assert monotone, "objective did not decrease monotonically"

g_final = thickness_gradient(LSE)
for e, te, ge in zip(elems, t_field, g_final):
    e.SetValue(KM.THICKNESS, float(te))
    e.SetValue(SMA.THICKNESS_SENSITIVITY, float(ge))

vtk_params = KM.Parameters(json.dumps({{
    "model_part_name": "Structure",
    "output_control_type": "step",
    "output_interval": 1,
    "file_format": "ascii",
    "output_path": ".",
    "output_sub_model_parts": False,
    "save_output_files_in_folder": False,
    "nodal_solution_step_data_variables": ["DISPLACEMENT", "REACTION"],
    "element_data_value_variables": ["THICKNESS", "THICKNESS_SENSITIVITY"],
}}))
KM.VtkOutput(mp, vtk_params).PrintOutput()

runtime = time.perf_counter() - t_start
summary = {{
    "compliance_initial": float(J0),
    "compliance_final": float(obj_history[-1]),
    "objective_history": [float(v) for v in obj_history],
    "objective_monotone_decreasing": bool(monotone),
    "mass": float(M0),
    "mass_history": [float(v) for v in mass_history],
    "gradient_norm_initial": float(np.linalg.norm(g0)),
    "gradient_norm_final": float(np.linalg.norm(g_final)),
    "fd_check": {{
        "element_id": 1,
        "analytic_dJ_dt": float(g0[0]),
        "finite_difference_dJ_dt": float(fd_grad),
        "rel_error": float(fd_rel_err),
    }},
    "tip_uy_initial": tip_uy0,
    "tip_uy_final": float(tip.GetSolutionStepValue(KM.DISPLACEMENT_Y)),
    "thickness_min_final": float(t_field.min()),
    "thickness_max_final": float(t_field.max()),
    "n_nodes": mp.NumberOfNodes(),
    "n_elements": mp.NumberOfElements(),
    "n_primal_solves": solve_counter[0],
    "runtime_seconds": runtime,
}}
print(json.dumps(summary, indent=2))
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "poromechanics": {
        "description": "Poromechanics: consolidation, fracture in porous media, dam/tunnel engineering",
        "application": "PoromechanicsApplication",
        # Real registered names on the 10.4.x pip wheels — the u-Pw
        # formulation was renamed u-Pl (liquid pressure); the old
        # UPw* / WATER_PRESSURE names are NOT registered.  Verified
        # empirically 2026-06-12 while building the real solve template.
        "elements": [
            "UPlSmallStrainElement2D3N",
            "UPlSmallStrainElement2D4N",
            "UPlSmallStrainElement3D4N",
            "UPlSmallStrainElement3D8N",
            "SmallStrainUPlDiffOrderElement2D6N",
        ],
        "capabilities": ["u-pl coupling (formerly u-pw)", "fracture_propagation", "interface_elements"],
        "pitfalls": [
            "[Physics] Different from GeoMechanicsApplication \u2014 this focuses on fracture in porous media Signal: the two applications register different element stems for the same u-p formulation \u2014 PoromechanicsApplication uses UPl* with LIQUID_PRESSURE, GeoMechanicsApplication uses UPw* with WATER_PRESSURE \u2014 so an element name valid under one raises 'is not registered' under the other.",
            "[API] PoromechanicsApplication uses the u-Pl formulation (already so in 10.2.3 and 10.3.0, unchanged in 10.4.2; the UPw* names belong to GeoMechanicsApplication): registered elements are UPlSmallStrainElement2D3N/2D4N/3D4N/3D8N, SmallStrainUPlDiffOrderElement*, FIC and interface variants; conditions are UPlFaceLoadCondition2D2N / UPlNormalFaceLoadCondition2D2N / UPlNormalLiquidFluxCondition2D2N. The pressure DOF is LIQUID_PRESSURE (reaction REACTION_LIQUID_PRESSURE), not WATER_PRESSURE. Signal: CreateNewElement('UPwSmallStrainElement2D4N', ...) raises 'The Element UPwSmallStrainElement2D4N is not registered!'; the exception message of a bogus element name dumps the full registered list, which shows only UPl* names. (Verified empirically 2026-06-12.)",
            "[API] Poromechanics material variables (DENSITY_SOLID, DENSITY_LIQUID, POROSITY, BULK_MODULUS_SOLID, BULK_MODULUS_LIQUID, PERMEABILITY_XX/YY/XY, DYNAMIC_VISCOSITY_LIQUID) are registered in the C++ kernel only \u2014 they are attributes of neither KratosMultiphysics nor the Poromechanics module. Fetch them via KM.KratosGlobals.GetVariable(\"DENSITY_SOLID\"). Note the names are *_LIQUID, not *_FLUID / *_WATER. Signal: AttributeError: Module KratosMultiphysics has no attribute DENSITY_SOLID. (Verified empirically 2026-06-12.)",
            "[Input] BIOT_COEFFICIENT is a required Properties entry of the UPl elements and is NOT derived from the bulk moduli \u2014 set prop.SetValue(P.BIOT_COEFFICIENT, 1.0) explicitly. Signal: element Check() fails with 'Error: BIOT_COEFFICIENT has Key zero, is not defined or has an invalid value at element N'. (Verified empirically 2026-06-12.)",
            "[Input] UPl element Initialize reads exotic nodal historical variables; a minimal nodal-variable list crashes inside C++. Mirror the full poromechanics_U_Pl_solver.AddVariables() set: INITIAL_STRESS_TENSOR, NODAL_EFFECTIVE_STRESS_TENSOR, NODAL_AREA, NODAL_JOINT_AREA/WIDTH/DAMAGE, NODAL_MID_PLANE_LIQUID_PRESSURE, NODAL_SLIP_TENDENCY, VELOCITY, ACCELERATION, FACE_LOAD, FORCE, DT_LIQUID_PRESSURE, NORMAL_LIQUID_FLUX, LIQUID_DISCHARGE. Signal: 'Error: This container only can store the variables specified in its variables list. The variables list doesn't have this variable: INITIAL_STRESS_TENSOR'; multi-threaded the same defect surfaces as an uninformative 'terminate called recursively' core dump \u2014 rerun with OMP_NUM_THREADS=1 to see the real exception. (Verified empirically 2026-06-12.)",
            "[API] PoroNewmarkQuasistaticUPlScheme takes (theta_u, theta_p, beta, gamma) \u2014 thetas FIRST (defaults 0.5, 0.5, 0.25, 0.5); a wrong order still runs but mis-weights the time integration. The scheme also requires ProcessInfo VELOCITY_COEFFICIENT, DT_LIQUID_PRESSURE_COEFFICIENT, TIME_UNIT_CONVERTER, NODAL_SMOOTHING, G_COEFFICIENT and RAYLEIGH_ALPHA/RAYLEIGH_BETA \u2014 the Rayleigh variables only exist after importing StructuralMechanicsApplication. Signal: KM.KratosGlobals.GetVariable('RAYLEIGH_ALPHA') fails with 'Variable RAYLEIGH_ALPHA is unknown' unless StructuralMechanicsApplication is imported; scheme Check() names whichever ProcessInfo entry is missing. (Verified empirically 2026-06-12.)",
        ]    },
    "shallow_water": {
        "description": "Shallow water equations (Saint-Venant) for flood/dam-break/coastal simulation",
        "application": "ShallowWaterApplication",
        # Real registered names in KratosShallowWaterApplication
        # are BoussinesqElement2D{3,4}N — NOT ShallowWaterElement.
        # Catalog drift caught 2026-06-01 by kratos_eletype_
        # scanner; corrected per the rans_shallowwater_element_
        # naming Tier-2 fixture.
        "elements": [
            "BoussinesqElement2D3N",
            "BoussinesqElement2D4N",
            # Full registered set on the 10.4.2 wheel, probed
            # empirically 2026-06-12 via the bogus-CreateNewElement
            # trick while building the real solve template.
            # WaveElement2D3N is the minimal/safest choice (it is what
            # the app's default WaveSolver builds for the "bdf" scheme).
            "WaveElement2D3N",
            "CrankNicolsonWaveElement2D3N",
            "PrimitiveElement2D3N",
            "ConservativeElementRV2D3N",
            "ConservativeElementFC2D3N",
        ],
        "solver_types": ["explicit", "semi-implicit"],
        "pitfalls": [
            "[API] The KratosShallowWaterApplication element name is BoussinesqElement2D3N / BoussinesqElement2D4N, NOT ShallowWaterElement2D3N. The Application class is named \"ShallowWater\" but the underlying element registration uses the \"Boussinesq\" stem (after the Boussinesq equations underlying the depth-averaged formulation). Signal: calling model_part.CreateNewElement with the unregistered name ShallowWaterElement2D3N raises 'is not registered' from kratos/python/add_model_part_to_python.cpp:159 on 10.3.0 (:154 in 10.4.0 to 10.4.3); the same call with 'BoussinesqElement2D3N' succeeds. (Verified empirically 2026-06-01 \u2014 same Tier-2 fixture as the RANS naming entry, rans_shallowwater_element_naming in scripts/tier2_fixtures/kratos/.)",
            "[Numerical] 2D only (depth-averaged) Signal: enforced by the element registry rather than by a convergence failure: the 2D Boussinesq elements construct while every 3D spelling raises 'is not registered', so a 3D shallow-water model cannot be assembled at all.",
            "[Input] Gravity is the SCALAR ProcessInfo[KM.GRAVITY_Z] (not a GRAVITY vector, not a nodal variable) \u2014 the base shallow-water solver sets ProcessInfo[GRAVITY_Z] = 9.81 and the elements read only that. Signal: silent g=0 \u2014 the initial free-surface perturbation never moves, all velocities stay exactly 0, rc=0. (Verified empirically 2026-06-12.)",
            "[Numerical] The BDF2 wave scheme (ShallowWaterResidualBasedBDFScheme) needs SetBufferSize(3) and the time loop must NOT solve until the buffer is full \u2014 replicate _TimeBufferIsInitialized (STEP + 1 >= 3); the first CloneTimeStep only shifts the initial condition into history. Signal: solving at step 1 uses a zero-filled history slot and corrupts the BDF time derivative \u2014 first-step velocities are wildly wrong, then the run \"recovers\" with a polluted transient. (Verified empirically 2026-06-12.)",
            "[API] FREE_SURFACE_ELEVATION is a derived variable, not solved: it stays at its initial value unless SW.ShallowWaterUtilities().ComputeFreeSurfaceElevation(model_part) is called after each step (the WaveSolver does this in FinalizeSolutionStep). Sign conventions: TOPOGRAPHY = -still_depth, HEIGHT = still_depth + eta, FREE_SURFACE_ELEVATION = HEIGHT + TOPOGRAPHY. Signal: HEIGHT evolves but FREE_SURFACE_ELEVATION stays frozen at the initial field. (Verified empirically 2026-06-12.)",
        ],
        "guidance": [
            "[Numerical] Wetting/drying needs special treatment",
            "[Numerical] Friction: Manning formula with roughness coefficient",
        ]
    },
    "wind_engineering": {
        "description": "Wind engineering: atmospheric boundary layer, wind loading on structures",
        "application": "WindEngineeringApplication",
        "capabilities": ["ABL_inlet_generation", "wind_pressure_coefficients", "vortex_shedding"],
        "pitfalls": [
            "[Numerical] Requires FluidDynamicsApplication + RANSApplication Signal: FluidDynamicsApplication and RANSApplication both import while WindEngineeringApplication itself does not \u2014 the dependencies are satisfiable and the application is the missing piece, which is the opposite of where a convergence-shaped hint would send a reader.",
            "[Integration] There is no catalog template for this physics. The availability-probe stub generators were removed in the 2026-06-26 honesty audit, so generate_input raises rather than emitting a script that exits 0 without solving. No KratosWindEngineeringApplication wheel is published, so a real template is impossible on a pip stack; a source build with that application enabled is required. Signal: `import KratosMultiphysics.WindEngineeringApplication` raises ModuleNotFoundError while both its stated prerequisites import; the physics key is present in the knowledge catalog while the generator registry holds no entry for it.",
        ],
    },
    "dam": {
        "description": "Dam engineering: thermal-mechanical analysis, seepage, cracking",
        "application": "DamApplication",
        "capabilities": ["thermal_analysis", "mechanical_analysis", "thermo_mechanical_coupled",
                         "seepage_analysis", "joint_elements"],
        # 2026-06-12: stub replaced by a real thermo-mechanical solve
        # (KratosDamApplication 10.4.2 wheel pip-installed); template
        # verified against analytic reaction sums (hydrostatic resultant
        # to 1.3e-4, trapezoid weight to 4.7e-5).
        "elements": ["SmallDisplacementThermoMechanicElement2D4N"],
        "constitutive_laws": ["ThermalLinearElastic2DPlaneStrain (PYTHON CLASS name — the "
                              "Materials.json registry string is ThermalLinearPlaneStrain, "
                              "see pitfalls)"],
        "pitfalls": [
            '[API] DamApplication thermal laws have a PYTHON CLASS name that differs from '
            'the string under which they are registered for Materials.json. '
            'DAM.ThermalLinearElastic2DPlaneStrain() instantiates fine from Python, but '
            '{"constitutive_law": {"name": "ThermalLinearElastic2DPlaneStrain"}} fails — the '
            'registered strings are ThermalLinearPlaneStrain / ThermalLinearPlaneStress / '
            'ThermalElasticIsotropic3D. Pick the route first, then the name. '
            'Signal: hasattr(DAM, "ThermalLinearElastic2DPlaneStrain") is True while '
            'KratosGlobals.HasConstitutiveLaw("ThermalLinearElastic2DPlaneStrain") is False '
            'and HasConstitutiveLaw("ThermalLinearPlaneStrain") is True. (Verified by '
            'execution 2026-08-03 on Kratos 10.4.0 with the 10.4.0 Dam wheel.)',
            '[Input] DamHydroConditionLoadProcess (C++-only, defaults undocumented '
            'in the wheel) requires keys model_part_name, variable_name '
            '("POSITIVE_FACE_PRESSURE"), Modify, Gravity_Direction ("Y"), '
            'Reservoir_Bottom_Coordinate_in_Gravity_Direction, Spe_weight '
            '(a SPECIFIC WEIGHT rho_w*g in N/m3, not a density), Water_level, '
            'Water_Table (int table id, 0 = none) and interval. The default '
            'interval is [0.0, 0.0] — pass an interval containing your TIME or '
            'the process silently does nothing. '
            'Signal: rc=0 with zero hydrostatic load (zero POSITIVE_FACE_PRESSURE '
            'everywhere) when TIME falls outside interval; ValidateAndAssignDefaults '
            'errors on any wrong key name. (Verified empirically 2026-06-12.)',
            '[Input] DamHydroConditionLoadProcess only SETS nodal '
            'POSITIVE_FACE_PRESSURE; nothing applies it as a force unless load '
            'conditions exist. DamApplication registers no 2D pressure condition of '
            'its own — add SMA LineLoadCondition2D2N on the wet face (it picks up '
            'POSITIVE/NEGATIVE_FACE_PRESSURE from nodal data). Condition node '
            'ordering sets the load direction: bottom->top on an upstream face at '
            'x=0 pushes downstream (+x). '
            'Signal: rc=0 with zero displacement and zero reactions (silent '
            'wrongness) when the conditions are missing; sum(REACTION_X) flips '
            'sign vs the analytic -0.5*gamma_w*Hw^2 when the node order is '
            'reversed. (Verified empirically 2026-06-12.)',
            '[API] SmallDisplacementThermoMechanicElement2D4N requires nodal '
            'solution-step variables TEMPERATURE and DAM.NODAL_REFERENCE_TEMPERATURE '
            '(both set), and Properties needs DAM.THERMAL_EXPANSION (the Dam '
            'variable, not the SMA one). '
            "Signal: 'Missing variable ...' at strategy Check() or first "
            'GetSolutionStepValue. (Verified empirically 2026-06-12.)',
            '[API] IncrementalUpdateStaticSmoothingScheme smooths Cauchy stress to '
            'the nodes in FinalizeSolutionStep and therefore needs the full '
            'dam_mechanical_solver.AddVariables() nodal set — in particular '
            'KM.NODAL_AREA, Poro.NODAL_CAUCHY_STRESS_TENSOR, Poro.NODAL_JOINT_WIDTH, '
            'Poro.NODAL_JOINT_AREA — even for a purely mechanical static run. '
            '(Bonus: nodal-smoothed stress comes for free — read components from '
            'the per-node 3x3 NODAL_CAUCHY_STRESS_TENSOR Matrix.) '
            "Signal: 'This container only can store the variables specified in its "
            "variables list' naming NODAL_AREA / NODAL_CAUCHY_STRESS_TENSOR during "
            'FinalizeSolutionStep. (Verified empirically 2026-06-12.)',
        ],
    },
    "constitutive_laws": {
        "description": "Extended constitutive law library: hyperelastic, plasticity, damage, viscoplastic",
        "application": "ConstitutiveLawsApplication",
        # REGISTERED NAMES ONLY. Every string below was accepted by
        # KratosGlobals.HasConstitutiveLaw on the installed Kratos 10.4.0
        # (/usr/bin/python3, 2026-08-03). The previous version of this block
        # listed model *families* (Ogden, Yeoh, Arruda-Boyce, Blatz-Ko, Mazars,
        # Perzyna, ModifiedCamClay, CriticalStateLine, RankineFragile,
        # DruckerPragerViscoplastic) — NONE of those resolve to a law in 10.4.0,
        # neither as a registry string nor as a Python attribute of
        # ConstitutiveLawsApplication, so a Materials.json built from them dies
        # with "not registered". See pitfall on yield-surface tags below.
        "laws": {
            "hyperelastic": ["HyperElastic3DLaw", "HyperElasticPlaneStrain2DLaw",
                             "HyperElasticIsotropicOgden1D (1D truss/cable only)",
                             "HyperElasticIsotropicHenky1D (1D truss/cable only)"],
            "plasticity": ["SmallStrainIsotropicPlasticity3D<YS><PP>",
                           "SmallStrainIsotropicPlasticityPlaneStrain<YS><PP>",
                           "SmallStrainKinematicPlasticity3D<YS><PP>",
                           "FiniteStrainIsotropicPlasticity3D<YS><PP>",
                           "<YS>/<PP> in {VonMises, Tresca, DruckerPrager, MohrCoulomb, "
                           "ModifiedMohrCoulomb} — 23 of the 25 3D isotropic pairings exist "
                           "(MohrCoulombModifiedMohrCoulomb and ModifiedMohrCoulombMohrCoulomb "
                           "are NOT registered in 10.4.0)"],
            "damage": ["SmallStrainIsotropicDamage3D<YS> (YS also Rankine / SimoJu)",
                       "SmallStrainDplusDminusDamage<TensionYS><CompressionYS><2D|3D>",
                       "SmallStrainThermalIsotropicDamage3D<YS>",
                       "DamageDPlusDMinusMasonry3DLaw"],
            "viscoplastic": ["GenericSmallStrainViscoplasticity3D"],
            "viscoelastic": ["ViscousGeneralizedMaxwell3D", "ViscousGeneralizedKelvin3D"],
        },
        "pitfalls": [
            "[Numerical] These laws extend StructuralMechanicsApplication Signal: ConstitutiveLawsApplication's loader imports StructuralMechanicsApplication first; the composite law names resolve through KratosGlobals.HasConstitutiveLaw only once both applications are loaded, so a stack with one of them fails on the inner import.",
            "[API] Must be registered via constitutive_law.name in MaterialsDEM.json Signal: the law name is matched verbatim \u2014 the full registered name resolves as an attribute of DEMApplication while natural abbreviations of it raise AttributeError. Nothing normalises, completes or fuzzy-matches the string.",
            "[Numerical] SmallStrainJ2PlasticityPlaneStrain2DLaw has GetStrainSize() == 4 (xx, yy, zz, xy) and is silently incompatible with the 2D solid elements (SmallDisplacementElement2D4N builds a 3-component B-matrix). Use the generic CLA plane-strain plasticity laws with strain size 3 instead, e.g. SmallStrainIsotropicPlasticityPlaneStrainVonMisesVonMises. Signal: no exception \u2014 the solve \"converges\" to garbage: reactions ~20x the elastic force, accumulated plastic strain O(10). (Verified empirically 2026-06-12 on the 10.4.2 wheel.)",
            "[Numerical] TANGENT_OPERATOR_ESTIMATION for the generic isotropic plasticity laws: the default second-order perturbation tangent throws condition-number errors and diverges after first yield; 0 (elastic tangent) makes Newton stagnate on a flat post-yield branch; 3 (secant) is the robust choice \u2014 prop.SetValue(CLA.TANGENT_OPERATOR_ESTIMATION, 3). Signal: 'Error: Condition number of the matrix is too high!, cond_number = 1.4e16' sprayed from OpenMP threads (MathUtils::CheckConditionNumber via SmallDisplacement::CalculateAll) and the NR residual diverges to ~1e13 N. (Verified empirically 2026-06-12.)",
            "[Input] The generic isotropic plasticity laws Check() requires YIELD_STRESS_TENSION + YIELD_STRESS_COMPRESSION (CLA variables), FRACTURE_ENERGY (KM) and HARDENING_CURVE (CLA, int enum: 0 LinearSoftening, 1 ExponentialSoftening, 2 InitialHardeningExponentialSoftening, 3 PerfectPlasticity, 4 CurveFittingHardening, 5 LinearExponentialSoftening, 6 CurveDefinedByPoints). Plain KM.YIELD_STRESS is NOT accepted. Curve 3 makes the perturbation tangent exactly singular; curve 0 with a huge FRACTURE_ENERGY is the practical near-perfect-plasticity setting. Signal: 'Error: HARDENING_CURVE is not a defined value', then the same for FRACTURE_ENERGY / YIELD_STRESS_TENSION / YIELD_STRESS_COMPRESSION one by one at law Check(). (Verified empirically 2026-06-12.)",
            "[Numerical] SmallStrainKinematicPlasticityPlaneStrainVonMisesVonMises segfaults during the first FE solve even though its Check() passes (KINEMATIC_HARDENING_TYPE is a CLA variable, KINEMATIC_HARDENING_MODULUS is a KM variable). Avoid; use the isotropic family. Signal: SIGSEGV, rc=139, no Python traceback, during the first strategy.Solve(). (Verified empirically 2026-06-12 on 10.4.2; REPRODUCED by execution 2026-08-03 on 10.4.0 \u2014 single SmallDisplacementElement2D4N, plane-strain uniaxial stretch past yield, TANGENT_OPERATOR_ESTIMATION=3, law assigned as a Python instance: rc=139, while the identical case with SmallStrainIsotropicPlasticityPlaneStrainVonMisesVonMises exits 0.) THIS LAW IS ALSO THE ONE REGISTRY HOLE: of the 254 SmallStrain*/FiniteStrain* Python classes exported by ConstitutiveLawsApplication 10.4.0, exactly two are absent from the string registry that Materials.json uses \u2014 this law and FiniteStrainIsotropicPlasticityFactory. So the JSON route fails LOUDLY (HasConstitutiveLaw is False, \"not registered\") while the Python-instance route reaches the solver and dies with SIGSEGV. Two different failure modes for the same law, depending on how you assign it.",
            "[Input] HARDENING_CURVE is NOT range-checked by the law Check(): values 7 and 8 \u2014 outside the documented 0..6 enum \u2014 pass Check() without any complaint on SmallStrainIsotropicPlasticity3DVonMisesVonMises. Two curves need extra properties that Check() DOES demand: curve 2 (InitialHardeningExponentialSoftening) needs MAXIMUM_STRESS, curve 4 (CurveFittingHardening) needs CURVE_FITTING_PARAMETERS; curves 0, 1, 3, 5, 6 pass with only YIELD_STRESS_TENSION / YIELD_STRESS_COMPRESSION / FRACTURE_ENERGY set. Signal: 'Error: MAXIMUM_STRESS is not a defined value' (curve 2), 'Error: CURVE_FITTING_PARAMETERS is not a defined value' (curve 4); an out-of-range curve index produces NO error at all. (Verified by execution 2026-08-03 on Kratos 10.4.0.)",
            "[API] The yield-surface tags VonMises / Tresca / DruckerPrager / MohrCoulomb / ModifiedMohrCoulomb / Rankine / SimoJu are NOT laws on their own \u2014 they only exist as substrings inside a composite law name. Writing {\"constitutive_law\": {\"name\": \"VonMises\"}} in Materials.json fails. Signal: KratosGlobals.HasConstitutiveLaw(\"VonMises\") is False while HasConstitutiveLaw(\"SmallStrainIsotropicPlasticity3DVonMisesVonMises\") is True. (Verified by execution 2026-08-03 on Kratos 10.4.0.)",
            "[API] EQUIVALENT_PLASTIC_STRAIN / PLASTIC_DISSIPATION / UNIAXIAL_STRESS cannot be read from the law object \u2014 use element.CalculateOnIntegrationPoints(CLA.EQUIVALENT_PLASTIC_STRAIN, mp.ProcessInfo) (PLASTIC_DISSIPATION is a KM variable, the other two CLA). Signal: law.Has(CLA.EQUIVALENT_PLASTIC_STRAIN) returns False / GetValue returns 0.0 while the integration points have demonstrably yielded. (Verified empirically 2026-06-12.)",
            "[Performance] Cap OpenMP threads BEFORE importing Kratos for small plasticity models \u2014 os.environ.setdefault(\"OMP_NUM_THREADS\", \"4\") at the top of the script. Signal: identical model and results, 59 s wall at 20 threads vs 0.6 s at 4 threads (thread-spin overhead dominates the per-Gauss-point return mapping). (Measured 2026-06-12.)",
        ],
    },
    "thermal_dem": {
        "description": "Thermal DEM: heat transfer between particles (conduction, convection, radiation)",
        "application": "ThermalDEMApplication",
        "capabilities": ["particle_heat_conduction", "convection", "radiation", "sintering"],
        "pitfalls": [
            "[Numerical] Requires DEMApplication as base Signal: ThermalDEMApplication is built on DEMApplication: the application's Python loader imports its dependencies first, so a build missing one fails on that inner import line rather than on the user's own import statement.",
            "[Integration] There is no catalog template for this physics. The availability-probe stub generators were removed in the 2026-06-26 honesty audit, so generate_input raises rather than emitting a script that exits 0 without solving. No KratosThermalDEMApplication wheel is published either, so a real template is impossible on a pip stack; a source build with that application enabled (plus DEM) is required. Signal: `import KratosMultiphysics.ThermalDEMApplication` raises ModuleNotFoundError while its DEMApplication prerequisite imports, so the absence is specific to this wheel rather than to the stack.",
        ],
        "guidance": [
            "[Physics] Temperature DOF per particle",
        ]
    },
    "swimming_dem": {
        "description": "Swimming DEM: particles in fluid flow (CFD-DEM coupling)",
        "application": "SwimmingDEMApplication",
        "capabilities": ["particle_laden_flow", "fluidized_bed", "sedimentation",
                         "Schiller-Naumann_drag", "virtual_mass", "Basset_history"],
        "pitfalls": [
            "[Numerical] Requires FluidDynamicsApplication + DEMApplication Signal: SwimmingDEMApplication needs both FluidDynamicsApplication and DEMApplication: the application's Python loader imports its dependencies first, so a build missing one fails on that inner import line rather than on the user's own import statement.",
            "[Integration] There is no catalog template for this physics. The availability-probe stub generators were removed in the 2026-06-26 honesty audit, so generate_input raises rather than emitting a script that exits 0 without solving. Installing the application is also not a free action: the published KratosSwimmingDEMApplication wheel pins an OLDER kratosmultiphysics patch release, so installing it downgrades the core out from under every other Kratos wheel in the environment. Signal: after a naive install, pip check reports the other Kratos application wheels requiring the newer core that was just downgraded; without the install, `import KratosMultiphysics.SwimmingDEMApplication` raises ModuleNotFoundError while both its prerequisites import.",
        ],
        "guidance": [
            "[Numerical] Two-way coupling: particles affect fluid momentum",
        ]
    },
    "dem_structures_coupling": {
        "description": "DEM-FEM coupling: particle impact on deformable structures",
        "application": "DemStructuresCouplingApplication",
        "capabilities": ["impact_loading", "blast_on_structures", "wear"],
        "pitfalls": [
            "[Numerical] Requires DEMApplication + StructuralMechanicsApplication Signal: DemStructuresCouplingApplication needs both DEMApplication and StructuralMechanicsApplication: the application's Python loader imports its dependencies first, so a build missing one fails on that inner import line rather than on the user's own import statement.",
            "[Input] The DEM wall submodelpart created at runtime for DemStructuresCouplingUtilities().TransferStructuresSkinToDem must carry dem_walls_mp.SetValue(Dem.PROPERTIES_ID, <id>) where <id> appears in a material_relations entry of MaterialsDEM.json together with the sphere material (particle-wall contact law lookup is by Properties Id). Signal: RuntimeError assembled from two literals with the part name between them \u2014 'PROPERTIES_ID is not set for SubModelPart' and '. Make sure the Materials file contains material assignation for this SubModelPart' \u2014 so the line reads PROPERTIES_ID is not set for SubModelPart SkinTransferredFromStructure . Make sure ... ; raised from ExplicitSolverStrategy::InitializeFEMElements. (Verified empirically 2026-06-12.)",
            "[API] SurfaceLoadFromDEMCondition3D4N is NOT registered on the 10.4.2 wheel \u2014 only SurfaceLoadFromDEMCondition3D3N and LineLoadFromDEMCondition2D2N. Mesh the structure with TETRAHEDRA so SkinDetectionProcess3D creates triangle skin conditions; a hexahedral mesh (quad skin) cannot get the DEM load applied. Signal: condition creation fails with 'not registered' for SurfaceLoadFromDEMCondition3D4N during SkinDetectionProcess3D on a quad skin. (Verified empirically 2026-06-12.)",
            "[Input] Wall nodal forces stay identically zero unless COMPUTE_FEM_RESULTS_OPTION is enabled \u2014 DEM_procedures.DEMFEMProcedures only turns it on when one of PostElasticForces / PostContactForces / PostPressure / PostNodalArea is true in the DEM parameters; without it ComputeDEMFaceLoadUtility.CalculateDEMFaceLoads transfers nothing. Signal: rc=0 but DEM_SURFACE_LOAD is identically zero on every wall node while the particle visibly bounces (silent one-way decoupling). (Verified empirically 2026-06-12.)",
            "[Physics] DEM_SURFACE_LOAD is a TRACTION (N/m2), not a nodal force \u2014 cross-checking action=reaction requires integrating it with DEM_NODAL_AREA (needs PostNodalArea: true). Also: TOTAL_FORCES lives in core KratosMultiphysics, not the DEM module (unlike CONTACT_FORCES / ELASTIC_FORCES). Signal: action-reaction check off by the nodal-area factor; AttributeError: module 'KratosMultiphysics.DEMApplication' has no attribute 'TOTAL_FORCES_Z'. (Verified empirically 2026-06-12.)",
            "[Input] DEM nodal variables (DEM_SURFACE_LOAD, BACKUP_LAST_STRUCTURAL_*, SMOOTHED_STRUCTURAL_VELOCITY, DELTA_DISPLACEMENT, DEM_PRESSURE, DEM_NODAL_AREA, ELASTIC_FORCES, CONTACT_FORCES, TANGENTIAL_ELASTIC_FORCES, SHEAR_STRESS, NON_DIMENSIONAL_VOLUME_WEAR, IMPACT_WEAR) must be added to the STRUCTURAL model part before its Initialize() \u2014 the transferred wall conditions reuse the structural nodes and their variable list. Also set dem_inlet_option: false (the default true with empty dem_inlets_settings breaks FinalizeSolutionStep). Signal: 'This container only can store the variables specified in its variables list' on the first coupling step; NameError/AttributeError on an undefined DEM_inlet in FinalizeSolutionStep when dem_inlet_option is left true. (Verified empirically 2026-06-12.)",
            "[Performance] OpenMP thread-spin overhead dominates few-particle DEM: set OMP_NUM_THREADS=1 (or a small number) BEFORE importing Kratos. Signal: ~88 ms per DEM substep at 20 threads on a 1-particle problem (~400 s total) vs 4 s single-threaded, identical results. (Measured 2026-06-12.)",
        ],
    },
    "fem_to_dem": {
        "description": "FEM-to-DEM transition: continuum fracture → discrete particles",
        "application": "FemToDemApplication",
        "capabilities": ["progressive_fracture", "concrete_cracking", "rock_fragmentation"],
        "pitfalls": [
            "[Integration] There is no catalog template for this physics. The availability-probe stub generators were removed in the 2026-06-26 honesty audit, so generate_input raises rather than emitting a script that exits 0 without solving. The KratosFemToDemApplication wheel is also frozen at an older release with no wheel for current CPython, so it cannot be installed into a modern Kratos stack. A source build with applications/FemToDemApplication enabled is required. Signal: `import KratosMultiphysics.FemToDemApplication` raises ModuleNotFoundError while its prerequisites DEMApplication and StructuralMechanicsApplication both import; pip finds no installable distribution for the running interpreter.",
        ],
        "guidance": [
            "[Numerical] Mesh-dependent fracture \u2014 requires damage regularization",
        ]    },
    "cable_net": {
        "description": "Cable and net structures: cables, membranes, form-finding",
        "application": "CableNetApplication",
        "elements": [
            # Elements registered by CableNetApplication itself
            # (cable_net_application.cpp Register()):
            "WeakSlidingElement3D3N",       # 3-node triangle sliding contact
            "SlidingCableElement3D3N",      # 3-node sliding cable (Line3DN)
            "RingElement3D3N",              # 3-node ring (Line3DN)
            "RingElement3D4N",              # 4-node ring (Line3DN)
            "EmpiricalSpringElement3D2N",   # 2-node empirical spring
            # Elements inherited from StructuralMechanicsApplication (loaded
            # first as CableNetApplication's transitive dependency):
            "CableElement3D2N",             # SMA-owned 2-node cable
            "MembraneElement3D3N",          # SMA-owned 3-node membrane
            "MembraneElement3D4N",          # SMA-owned 4-node membrane
        ],
        "variables": ["SPRING_DEFORMATION_EMPIRICAL_POLYNOMIAL"],
        "capabilities": ["form_finding", "prestress", "wind_loading_on_cables"],
        # 2026-06-12: stub replaced by a real sliding-cable + empirical-
        # spring solve (KratosCableNetApplication 10.4.2 wheel pip-
        # installed); template verified against the analytic equal-
        # tension pulley equilibrium to rel. err < 3e-13.
        "pitfalls": [
            '[API] strategy.Check() before strategy.Initialize() SEGFAULTS with '
            'SlidingCableElement3D3N models: the element Check dereferences the '
            'constitutive-law pointer, which is only cloned in Initialize(). Call '
            'strat.Initialize() first (the solid-element reference pattern calls '
            'Check() directly, which is fine there but not here). '
            "Signal: 'Fatal Python error: Segmentation fault' (no Kratos exception) "
            'pointing at the Check() line. (Verified empirically 2026-06-12 on the '
            '10.4.2 wheel.)',
            '[API] KM.VtkOutput rejects the sliding-cable/ring generic Line3DN '
            'geometry. Workaround: after the solve, build a visualization '
            'SubModelPart holding the same nodes plus plain Element3D2N 2-node lines '
            'per segment and pass THAT to VtkOutput — the .vtk then carries the '
            'DISPLACEMENT/REACTION nodal fields. '
            "Signal: the two literals of kratos/input_output/vtk_output.cpp:451, "
            "'Modelpart contains elements or conditions with' and "
            "'geometries for which no VTK-output is implemented!', which the run "
            "joins and prefixes, so the line reads: Error: Modelpart contains "
            "elements or conditions with geometries for which no VTK-output is "
            "implemented! (Verified "
            'empirically 2026-06-12.)',
            '[Numerical] EmpiricalSpringElement3D2N has a uBLAS aliasing bug in its '
            'RHS (GlobalizeVector does noalias(rInputVector) = prod(Matrix(trans(T)), '
            'rInputVector) — an in-place sequential update of the vector being read; '
            'empirical_spring.cpp, 10.4.2 == current master). The nodal force is NOT '
            'F*ex but F*(ex_x, ex_x*ey_x, ex_x*(ez_x + ez_y*ey_x)): a spring whose '
            'current axis is global-y or global-z aligned produces EXACTLY ZERO '
            'force, silently; the force is exact only when the spring axis is '
            'global-x aligned. The LHS is unaffected (Matrix = prod(...) makes a '
            'temporary), so Newton still converges — to a wrong equilibrium. '
            'Signal: probe CalculateRightHandSide on a skewed spring: RHS/F matches '
            '(ex_x, ex_x*ey_x, ex_x*(ez_x+ez_y*ey_x)) instead of (ex_x, ey... ) — '
            'i.e. components vanish as the axis rotates away from global x; a pure '
            'x-stretch returns R = k*dl exactly. (Verified empirically 2026-06-12, '
            'predicted vs measured to 4 decimals.)',
            "[API] CableNetApplication's own Register() exposes "
            "5 elements with a Sliding/Ring/EmpiricalSpring "
            "vocabulary, NOT the simpler CableElement3D2N / "
            "MembraneElement3D3N / MembraneElement3D4N that the "
            "catalog historically grouped under cable_net. The "
            "simpler-named elements ARE usable from a CableNet "
            "ProjectParameters.json — they exist because "
            "CableNetApplication's Python loader imports "
            "StructuralMechanicsApplication FIRST (its "
            "transitive dependency), and Cable/Membrane "
            "elements live in SMA. Signal: source walk of "
            "applications/CableNetApplication/"
            "cable_net_application.cpp shows only the 5 "
            "Sliding/Ring/Empirical KRATOS_REGISTER_ELEMENT "
            "calls; CableElement3D2N + MembraneElement3D[34]N "
            "registrations live in applications/"
            "StructuralMechanicsApplication/"
            "structural_mechanics_application.cpp:578/619/620. "
            "If a user removes the SMA dependency or loads "
            "CableNetApplication in isolation, the simpler "
            "names fail with 'Element <X> is not registered'. "
            "(File walk 2026-06-02.)",
            "[Input] EmpiricalSpringElement3D2N's "
            "SPRING_DEFORMATION_EMPIRICAL_POLYNOMIAL Vector "
            "follows the highest-degree-first NumPy "
            "np.poly1d convention — coefficients[0] is the "
            "highest-degree term, coefficients[-1] is the "
            "constant. The source loop "
            "(empirical_spring.cpp:118) computes "
            "`current_int_force += poly[i] * pow(disp, "
            "size-1-i)` for i in [0,size). Entering the "
            "coefficients in physicist-style "
            "ascending-degree order (constant first) "
            "silently swaps high- and low-order terms — no "
            "exception, just wildly wrong forces (e.g. for "
            "a linear spring users pass [k_lin, 0] not [0, "
            "k_lin]). KRATOS_ERROR_IF at line 275 only "
            "rejects size < 2 (i.e. order-0 polynomial), "
            "NOT a wrong-order vector. Signal: integrated "
            "force at a known displacement is off by a "
            "factor of `disp^(degree)` from the expected "
            "value. (File walk empirical_spring.cpp "
            "2026-06-03.)",
            "[Numerical] EmpiricalSpringElement3D2N's "
            "CalculateMassMatrix (empirical_spring.cpp:443) "
            "UNCONDITIONALLY calls CalculateLumpedMassVector "
            "and fills only the diagonal — no consistent-"
            "mass path. Setting "
            "ProcessInfo[USE_CONSISTENT_MASS_MATRIX]=true "
            "or any equivalent ProjectParameters.json knob "
            "has zero effect for this element. Signal: in "
            "explicit-dynamics or modal analyses, the off-"
            "diagonal mass-coupling contribution is missing "
            "and the first few eigenfrequencies differ from "
            "the consistent-mass reference by O(10%); no "
            "warning, no error. Combine with the fact that "
            "DENSITY + CROSS_AREA are BOTH required "
            "(Check() at lines 256-266 KRATOS_ERROR if "
            "either is missing or <= eps) even for purely-"
            "static analyses where mass is unused — users "
            "running form-finding without dynamics still "
            "hit 'DENSITY not provided' / 'CROSS_AREA not "
            "provided' KRATOS_ERROR at element-Check time. "
            "(File walk empirical_spring.cpp 2026-06-03.)",
            "[Input]+[Numerical] SlidingCableElement3D3N "
            "(sliding_cable_element_3D.cpp) is the most "
            "feature-rich of the CableNet elements and has "
            "THREE knobs users often miss: "
            "(1) `CONSTITUTIVE_LAW` is REQUIRED on Properties. "
            "Init() (line 97-106) KRATOS_ERRORs with 'A "
            "constitutive law needs to be specified for the "
            "element with ID <X>' if missing. Use a 1D law "
            "such as TrussConstitutiveLaw for linear elastic "
            "or HyperElastic1D for nonlinear. Unique to "
            "SlidingCable among the 5 cable_net-registered "
            "elements (RingElement3D / EmpiricalSpringElement "
            "use direct properties, not a CL). Check() (line "
            "778-794) verifies CONSTITUTIVE_LAW presence + "
            "delegates to `mpConstitutiveLaw->Check(...)` — "
            "but does NOT validate CROSS_AREA / DENSITY which "
            "are still accessed downstream (same gap as "
            "RingElement). "
            "(2) `mIsCompressed` silent zero-stiffness: "
            "GetInternalForces (line 345-348) sets the flag "
            "when total_internal_force < 0 AND "
            "|current_length - ref_length| > numeric_eps. When "
            "set, CalculateLeftHandSide (line 545-547) AND "
            "the internal-force component of "
            "CalculateRightHandSide (line 563-565) become "
            "no-ops — the element drops out of the global "
            "system. Correct cable physics (can't push) but "
            "confusing on first encounter: 'Newton converges "
            "but the cable seems inactive'. Body forces "
            "(self-weight) ARE still applied. Signal: "
            "displacement diverges from expected analytical "
            "form at points where the segment goes into "
            "compression. "
            "(3) `FRICTION_COEFFICIENT` (Properties) activates "
            "Capstan-equation-style frictional sliding when "
            "> 0 (line 353-415). Internal normal force "
            "propagates across nodes as "
            "n_next = n_prev ± μ·deviation_force, with sign "
            "chosen by segment-strain ordering "
            "(el_i_1 < el_i_0 → minus). Default (omitted or "
            "0.0) is FRICTIONLESS sliding — the cable slides "
            "freely through intermediate nodes with no force "
            "redistribution. Users who assume a real cable "
            "must have friction need to set this explicitly. "
            "(File walk sliding_cable_element_3D.cpp "
            "2026-06-03.)",
            "[Input]+[Numerical] RingElement3D3N / "
            "RingElement3D4N (ring_element_3D.cpp) has TWO "
            "sharp edges users routinely miss: "
            "(1) Check() at lines 632-647 ONLY verifies "
            "(a) Id() >= 1, (b) GetCurrentLength() > 0, "
            "and (c) PointsNumber() ∈ {3, 4}. It does NOT "
            "verify CROSS_AREA / YOUNG_MODULUS / DENSITY are "
            "set on the element's Properties — these are "
            "accessed in LinearStiffness() (line 702-705: "
            "CROSS_AREA * YOUNG_MODULUS / GetRefLength()) and "
            "CalculateLumpedMassVector (line 482-484: A * L * "
            "rho). Missing CROSS_AREA or YOUNG_MODULUS gives "
            "k_0 = 0 → silent zero stiffness → Newton solver "
            "fails to converge with no actionable error. "
            "Missing DENSITY makes the lumped-mass diagonal "
            "zero → eigenvalue problem yields infinite "
            "frequencies. The element-level Check is WEAKER "
            "than EmpiricalSpringElement3D2N's. "
            "(2) CalculateLumpedMassVector (ring_element_3D."
            "cpp:488-493) assigns `total_mass = A * L_ref * "
            "rho` to EVERY local DOF — i.e. each of the 3*N "
            "translational DOFs gets the FULL ring mass, not "
            "total_mass / (3*N) per DOF as in a standard "
            "lumped-mass distribution. Comment on line 469-471 "
            "explicitly flags this: 'ATTENTION !!!! this "
            "function uses a fictitious mass for the sliding "
            "nodes — needs improvement !!!'. Explicit-dynamics "
            "or modal analyses with this element get "
            "artificially heavy rings — the first few "
            "frequencies are off by a factor of "
            "sqrt(3 * N_nodes) compared to standard FEA "
            "lumping. Plus: CalculateMassMatrix is the same "
            "unconditional-lumped pattern as "
            "EmpiricalSpringElement3D2N (lines 498-521) — "
            "USE_CONSISTENT_MASS_MATRIX is silently ignored. "
            "Signal: the Newton loop simply exhausts its "
            "iteration budget on a ring model with no actionable "
            "error. What you can grep for is the convergence "
            "criterion's own reporting, not a ring message: the "
            "'RESIDUAL CRITERION' line carries 'Initial residual "
            "norm = ' and a current-residual value that has gone "
            "to nan, and an adaptive Newton-Raphson strategy adds "
            "'ATTENTION: max iterations exceeded'. Check "
            "Properties has CROSS_AREA and YOUNG_MODULUS set, "
            "since the element Check skips that. Modal analysis returns "
            "infinity / NaN frequencies → DENSITY missing. "
            "First few non-rigid eigenfrequencies off by a "
            "factor of sqrt(3*N_nodes) from analytic — the "
            "lumped-mass artifact (every translational DOF gets "
            "full ring mass). "
            "(File walk ring_element_3D.cpp 2026-06-03.)",
            "[Input] EdgeCableElementProcess "
            "(custom_processes/edge_cable_element_process.h) "
            "creates ONE SlidingCable or Ring element from "
            "user-supplied node IDs at ExecuteInitialize time "
            "(not per-step like ApplyWeakSlidingProcess). 4 "
            "edges users routinely hit: "
            "(a) `element_type` is a 2-VALUE enum: 'cable' "
            "(→ SlidingCableElement3D3N, expects 3 nodes) or "
            "'ring' (→ RingElement3D4N, expects 4 nodes). Any "
            "other value KRATOS_ERRORs with 'element type : "
            "<X> not available for sliding process' (a "
            "MISLEADING message — this is the EdgeCable process, "
            "not the sliding process; the error text is "
            "copy-paste from ApplyWeakSliding). "
            "(b) `node_id_order` must match the element-type's "
            "node count: 3 entries for 'cable', 4 for 'ring'. "
            "The number_nodes consistency check at line 108 "
            "(`mrModelPart.Nodes().size() == number_nodes`) "
            "counts ALL nodes in the model part (NOT the "
            "sub-model-part) — so passing the full 'Structure' "
            "model part with 1000+ nodes plus "
            "`node_id_order=[1,2,3]` always fails with "
            "'numbers of nodes in submodel part not consistent "
            "with numbers of nodes in process properties' even "
            "though the user did supply 3 IDs. Workaround: pass "
            "a SUB-model-part that contains exactly the 3-or-4 "
            "edge nodes. "
            "(c) `edge_sub_model_part_name` is a PHANTOM "
            "parameter — declared in default_parameters (line "
            "64) and ValidateAndAssignDefaults-validated, but "
            "NEVER REFERENCED in CreateEdgeCableElement. Users "
            "setting this expecting the process to query a "
            "specific sub-model-part get no effect; mrModelPart "
            "is used directly throughout. Same pattern as "
            "ApplyWeakSlidingProcess's "
            "`computing_model_part_name` (edge a of pitfall 6). "
            "(d) Underlying geometry is Line3DN<NodeType> (the "
            "stub geometry documented in pitfall 8 from tick "
            "#116) — number_nodes comes from "
            "node_id_order.size() and is unchecked at "
            "construction (Line3DN's own PointsNumber check is "
            "commented out). Supplying 5 node IDs with "
            "element_type='cable' silently constructs a "
            "5-node Line3DN, then "
            "SlidingCableElement3D3N::Create receives a Line3DN "
            "with the wrong PointsNumber — crashes deep in the "
            "element ctor with no surface-level diagnostic. "
            "Signal: KRATOS_ERROR 'element type : <X> not "
            "available for sliding process' on any element_type "
            "other than 'cable'/'ring' (note: misleading text — "
            "this is the EdgeCable process, copy-pasted from "
            "ApplyWeakSliding); KRATOS_ERROR 'numbers of nodes "
            "in submodel part not consistent with numbers of "
            "nodes in process properties' when the model_part "
            "passed in is the global Structure (1000+ nodes) "
            "instead of a 3-node-or-4-node sub-model-part; "
            "silent edge_sub_model_part_name parameter (declared "
            "but never read); deep ctor crash inside "
            "SlidingCableElement3D3N::Create on a node_id_order "
            "length that disagrees with element_type. "
            "(File walk applications/CableNetApplication/"
            "custom_processes/edge_cable_element_process.h "
            "2026-06-03.)",
            "[Input]+[Performance] ApplyWeakSlidingProcess "
            "(custom_processes/apply_weak_sliding_process.h) is the "
            "auto-creator for WeakSlidingElement3D3N elements — it "
            "rebuilds the slave→master triangle elements EVERY time "
            "step (ExecuteInitializeSolutionStep creates, "
            "ExecuteFinalizeSolutionStep removes via "
            "RemoveElementFromAllLevels). Four edges users "
            "routinely hit: "
            "(a) ProcessParameters JSON accepts 6 keys (model_part_"
            "name_slave / model_part_name_master / computing_model_"
            "part_name / element_id / property_id / debug_info) but "
            "`computing_model_part_name` is a PHANTOM parameter — "
            "declared in default_parameters (line 66-74) and "
            "ValidateAndAssignDefaults-validated, but the only "
            "line that referenced it is COMMENTED OUT at "
            "apply_weak_sliding_process.h:117. Created elements "
            "are always added to mrModelPart directly (line 136: "
            "`mrModelPart.AddElement(pElem)`), bypassing whatever "
            "the user sets. Setting this parameter has zero "
            "effect. "
            "(b) `FindNearestNeighbours` is O(N·M) BRUTE FORCE — "
            "iterates over all master nodes TWICE (once for "
            "nearest, once for second-nearest) for each slave "
            "node. For N slave / M master nodes the cost is "
            "2·N·M per time step. The header declares "
            "Bucket+KDTree typedefs at lines 53-55 (Bucket<3, "
            "NodeType, NodeVector, NodeTypePointer, NodeIterator, "
            "DoubleVectorIterator> + Tree<KDTreePartition<"
            "BucketType>>) but they are NEVER USED in the "
            "implementation. Dead KD-tree scaffolding — the actual "
            "search is O(N·M). Users with large meshes (e.g. "
            "10^5 slave × 10^4 master) get unusable rebuild times. "
            "(c) `distance = 1e12;` MAGIC INFINITY at lines 148 + "
            "163 with explicit `// better: std::numeric_limits"
            "<double>::max()` comments — the author knows it's "
            "wrong but didn't fix it. If a slave node is genuinely "
            ">1e12 units from any master (rare but possible in "
            "geophysical / outsized-domain simulations), the "
            "initial `neighbour_ids = {0, 0}` is returned UNCHANGED, "
            "leading to `master_model_part.pGetNode(0)` which fails "
            "since Kratos node IDs are 1-based — but the failure "
            "comes deep inside Triangle3D3 construction with an "
            "opaque error. "
            "(d) NO Check() override is provided — missing sub-"
            "model-parts (typo in model_part_name_slave / _master) "
            "crash deep inside GetSubModelPart with the cryptic "
            "'There is no sub model part with name ...' KRATOS_ERROR "
            "at runtime, not at parameter-validation time. "
            "Slave-node-3 topology in CreateElements (line 127-129: "
            "[master_n0, master_n1, slave_node]) MATCHES the "
            "WeakSlidingElement3D3N hard-coded topology contract "
            "(edge 1 of pitfall 7 from tick #100) — confirming the "
            "two are co-designed. "
            "Signal: O(N·M) per-step rebuild dominates wall-time "
            "on N slave × M master meshes (the Bucket+KDTree "
            "typedefs at lines 53-55 are dead scaffolding — never "
            "used); silent computing_model_part_name parameter "
            "(declared, commented-out reference in source) means "
            "user choice is ignored; the node lookup failing when a "
            "slave node is genuinely > 1e12 units from any master "
            "(the 1e12 magic infinity leaves neighbour_ids={0,0} "
            "unchanged → master_part.pGetNode(0), and Kratos IDs are "
            "1-based) — the message for that is "
            "'RuntimeError: Error: Node index not found: 0.' from "
            "kratos/includes/mesh.h:288, NOT the "
            "'There is no node with id 0' / 'index out of bounds' "
            "previously quoted here, neither of which is emitted by "
            "any Kratos library in this build (measured directly: "
            "mp.GetNode(0) on a populated model part gives exactly "
            "the mesh.h text); KRATOS_ERROR 'There is no sub model "
            "part with name \"<X>\" in model part \"<Y>\"' deep in "
            "GetSubModelPart from a typo in model_part_name_slave/"
            "_master (no Check() override) — that one reproduces "
            "verbatim, with both names interpolated and in double "
            "quotes. (Node-lookup and sub-model-part messages "
            "verified by execution 2026-08-07, Kratos 10.4.3.) "
            "(File walk applications/"
            "CableNetApplication/custom_processes/"
            "apply_weak_sliding_process.h 2026-06-03.)",
            "[API]+[Reference] Line3DN<TPointType> "
            "(custom_geometries/line_3d_n.h) is the SHARED "
            "geometry class used by all four sliding-/cable-/ring-"
            "registered elements (SlidingCableElement3D3N, "
            "RingElement3D3N, RingElement3D4N — plus implicit "
            "use by WeakSlidingElement3D3N's 3-node topology). "
            "It is a STUB GEOMETRY with the following gaps: "
            "(1) Constructor's PointsNumber validation is "
            "COMMENTED OUT (line 173-175: `//if (BaseType::"
            "PointsNumber() != 3) KRATOS_ERROR << \"Invalid "
            "points number. Expected 3, given ...\"`) — Line3DN("
            "PointsArrayType) silently accepts ANY N, no count "
            "check. "
            "(2) GetGeometryFamily and GetGeometryType overrides "
            "are BOTH commented out (lines 206-214) — the class "
            "INHERITS whatever Geometry<TPointType> defaults to. "
            "mdpa parsers expecting Kratos_Line3DN may fall "
            "through to the generic family. "
            "(3) Class docstring (line 50): 'arbitrary node 3D "
            "line geometry with quadratic shape functions' — but "
            "ShapeFunctionValue (line 564-569) KRATOS_ERRORs "
            "'\\'ShapeFunctionValue\\' not available for "
            "arbitrarty noded line' (sic — typo 'arbitrarty' "
            "appears 18 times across most geometry methods: "
            "ShapeFunctionValue, ShapeFunctionsLocalGradients × 3 "
            "overloads, ShapeFunctionsGradients, "
            "PointsLocalCoordinates, LumpingFactors, DomainSize, "
            "IsInside, Jacobian × 4 overloads, plus internal "
            "CalculateShapeFunctionsIntegrationPointsValues). NO "
            "shape functions are implemented despite the "
            "docstring claim. "
            "(4) ShapeFunctionsIntegrationPointsGradients and "
            "InverseOfJacobian(rResult, rPoint) KRATOS_ERROR "
            "'Jacobian is not square' (it's a 1D-in-3D line, so "
            "the rectangular Jacobian doesn't admit an inverse — "
            "use DeterminantOfJacobian / Jacobian instead). "
            "(5) EdgesNumber() = 0 and FacesNumber() = 0 (correct "
            "for a 1D-in-3D geometry — but unusual; users probing "
            "geom.EdgesNumber() get 0, not 1). "
            "How cable_net elements work around it: "
            "SlidingCable/Ring/EmpiricalSpring/WeakSliding all "
            "compute their stiffness / internal-force kinematics "
            "DIRECTLY from nodal coordinates (X0/Y0/Z0 + "
            "DISPLACEMENT_X/Y/Z), bypassing the geometry's "
            "shape-function API. Users trying to evaluate post-"
            "processed scalars on a Line3DN geometry via "
            "geom.ShapeFunctionValue(...) hit the KRATOS_ERROR. "
            "Signal: KRATOS_ERROR \"'ShapeFunctionValue' not "
            "available for arbitrarty noded line\" (yes, typo "
            "'arbitrarty' in the source — appears 18 times across "
            "the geometry method stubs) when post-processing "
            "scalars via geom.ShapeFunctionValue / "
            "ShapeFunctionsLocalGradients / "
            "ShapeFunctionsGradients / PointsLocalCoordinates / "
            "LumpingFactors / DomainSize / IsInside / Jacobian on "
            "a cable_net element's geometry; KRATOS_ERROR "
            "'Jacobian is not square' from "
            "InverseOfJacobian(rResult, rPoint) — use "
            "DeterminantOfJacobian / Jacobian instead. Silent "
            "acceptance of any N at Line3DN construction "
            "(PointsNumber validation commented out, line 173-175) "
            "means a downstream Create() crashes with an "
            "element-specific topology error instead of a clean "
            "geometry-level rejection. EdgesNumber() returns 0 "
            "and FacesNumber() returns 0 (correct for 1D-in-3D, "
            "but unusual). "
            "(File walk applications/CableNetApplication/"
            "custom_geometries/line_3d_n.h 2026-06-03.)",
            "[Input]+[Numerical] WeakSlidingElement3D3N "
            "(weak_coupling_slide.cpp) has FOUR edges users "
            "routinely miss when wiring slave-node sliding "
            "contact: "
            "(1) Topology contract is HARD-CODED — nodes 1+2 "
            "are the LINE the slave runs on, node 3 IS the "
            "slave. Source header comment lines 11-12 spell "
            "this out but the .mdpa file format gives no hint. "
            "Swapping the slave-node position (e.g. listing the "
            "slave as node 1 and the rail endpoints as 2+3) "
            "silently builds a different, wrong constraint — "
            "no exception, just an unphysical residual. The "
            "Check() method (lines 340-370) only verifies "
            "nodecount == 3 and dimension == 3, NOT the role "
            "ordering. "
            "(2) The 'spring stiffness' α is read from "
            "Properties[YOUNG_MODULUS] (line 122 + line 310, "
            "with literal source comment 'simplified \"spring "
            "stiffness\"'). Common confusion: this is NOT the "
            "elastic modulus of any material — it's the penalty "
            "stiffness for the sliding constraint. Setting "
            "YOUNG_MODULUS to a steel-like 2.1e11 yields a "
            "near-infinite penalty that locks the slave to the "
            "rail with zero slack; users wanting a soft "
            "constraint must pick α deliberately based on the "
            "global stiffness scale. Check() (line 361-365) "
            "KRATOS_ERRORs only when YOUNG_MODULUS is missing "
            "or ≤ eps — any positive value passes silently. "
            "(3) The element contributes NO mass and NO damping "
            "in explicit dynamics. AddExplicitContribution with "
            "Variable<double>& destination (lines 399-409) is "
            "explicitly OVERRIDDEN to be a no-op with the "
            "source comment 'overwriting base class function to "
            "omit error msg / this element does not contribute "
            "any mass or damping'. Stable explicit time-step "
            "estimators that include this element's contribution "
            "via CalculateLumpedMassMatrix get an incorrect "
            "(infinite-period) result. "
            "(4) The stiffness-matrix entries are ANALYTIC "
            "expressions of the 9 nodal coordinates "
            "(X0/Y0/Z0 + displacement) with denominators "
            "proportional to "
            "pow((Xa-Xb-ua+ub)^2 + (Ya-Yb-va+vb)^2 + "
            "(Za-Zb-wa+wb)^2, 2) or pow(..., 3). When the "
            "rail-segment endpoints (nodes 1+2) DEGENERATE to "
            "the same point (zero-length rail), the matrix "
            "divides by zero — NaN propagates silently into "
            "the global stiffness. No guard exists. Users with "
            "geometric tolerance ≤ 1e-6 on a CAD-imported mesh "
            "can hit this if endpoints snap together. "
            "Signal: NaN propagating silently into global K and "
            "downstream into the residual norm at the first solve "
            "step when the rail-segment endpoints (nodes 1+2) "
            "have degenerated to the same point (zero-length "
            "rail) — the symptom is a nan current-residual value "
            "on the criterion's 'RESIDUAL CRITERION' line, with no "
            "exception, on a CAD-imported mesh with snap "
            "tolerance ≤ 1e-6; explicit-dynamics with this "
            "element gives infinite stable time step from the "
            "no-op CalculateLumpedMassMatrix; YOUNG_MODULUS set "
            "to a steel-like 2.1e11 locks the slave to the rail "
            "with no slack (unphysical penalty α). Check() "
            "accepts wrong role-ordering (slave listed at node "
            "position 1 instead of node position 3) — wrong "
            "constraint, no exception. "
            "(File walk weak_coupling_slide.cpp 2026-06-03.)",
            "[Validation] EmpiricalSpringElementProcess "
            "(custom_processes/empirical_spring_element_process.h, "
            "the C++ class wrapped by python_scripts/"
            "empirical_spring_element_process.py) has a MISLEADING "
            "ERROR MESSAGE on the displacement/force-data "
            "validation. Constructor lines 83-84: "
            "  KRATOS_ERROR_IF(mParameters[\"node_ids\"].size()!=2) "
            "    << \"exactly two nodes for each spring needed !\"; "
            "  KRATOS_ERROR_IF(mParameters[\"displacement_data\"]"
            "    .size()!=mParameters[\"force_data\"].size()) "
            "    << \"only two nodes for each spring allowed !\"; "
            "The line-84 check is about ARRAY LENGTH MISMATCH "
            "between displacement_data and force_data (the data "
            "points fed to numpy.polyfit), but the error message "
            "is a copy-paste of the line-83 NODE-COUNT message. "
            "User-visible effect: if a user passes "
            "displacement_data=[0.0, 1.0, 2.0] and "
            "force_data=[0.0, 1.0] (a common mistake when adding "
            "a data point to one array but forgetting the other), "
            "Kratos aborts with 'only two nodes for each spring "
            "allowed !' — pointing the user at node_ids (which is "
            "fine) instead of the actual array-length bug. The "
            "Python wrapper at python_scripts/"
            "empirical_spring_element_process.py:45 calls "
            "polyfit(displacement_data.GetVector(), "
            "force_data.GetVector(), polynomial_order.GetInt()) "
            "BEFORE the C++ constructor's check runs — so the "
            "actual failure path is: numpy.polyfit raises "
            "TypeError 'expected x and y to have same length' "
            "(or similar) on line 45 before the C++ check ever "
            "fires. If someone calls the C++ class directly (e.g. "
            "from a custom Python script bypassing the wrapper), "
            "the C++ check fires with the misleading message. The "
            "Python wrapper takes the same Parameters dict so the "
            "user can hit this either way depending on import "
            "path. "
            "Signal: from the Python wrapper, numpy.polyfit "
            "raises TypeError 'expected x and y to have same "
            "length' on line 45 BEFORE the C++ check ever fires; "
            "from a custom Python script calling the C++ class "
            "directly, KRATOS_ERROR 'only two nodes for each "
            "spring allowed !' fires on a displacement_data / "
            "force_data length mismatch (misleading — the error "
            "text points at node_ids, but the actual cause is "
            "the data-array mismatch). Fix is to make sure "
            "len(displacement_data) == len(force_data); "
            "node_ids is unrelated. "
            "(File walk empirical_spring_element_process.h "
            "2026-06-03.)",
            "[Validation]+[Integration] SlidingEdgeProcess "
            "(custom_processes/sliding_edge_process.h + the "
            "python_scripts/sliding_edge_process.py wrapper) is "
            "REGISTERED via pybind11 but is BROKEN through every "
            "user path: "
            "(1) Schema/code mismatch at the C++ level. The "
            "constructor's default_parameters JSON declares 8 "
            "keys (constraint_name, master_sub_model_part_name, "
            "slave_sub_model_part_name, variable_names, "
            "reform_every_step, debug_info, angled_initial_line, "
            "follow_line). The body actually reads 11 keys: of "
            "those 8 it reads 7 (constraint_name is in defaults "
            "but NEVER consulted by the code), and it ALSO reads "
            "4 keys that are NOT in defaults: bucket_size, "
            "neighbor_search_radius, must_find_neighbor, AND "
            "constraint_set_name (32 occurrences in the source "
            "vs the 1 occurrence of constraint_name in the JSON "
            "block). Effect: ValidateAndAssignDefaults at the "
            "ctor's line 76 either (a) accepts user input that "
            "ONLY uses the 8 default keys → runtime "
            "Kratos::Exception fires later when the code looks "
            "up mParameters[\"bucket_size\"] / "
            "[\"neighbor_search_radius\"] / "
            "[\"must_find_neighbor\"] / "
            "[\"constraint_set_name\"], or (b) rejects user "
            "input that adds the missing-from-defaults keys "
            "because ValidateAndAssignDefaults errors on extra "
            "keys. NO valid input configuration exists. "
            "(2) Python-wrapper is even more broken. "
            "python_scripts/sliding_edge_process.py:32 reads "
            "from `model_part_name.GetSubModelPart(...)` — but "
            "`model_part_name` is an UNDEFINED variable in that "
            "scope (NameError at instantiation). Line 34 then "
            "tries `settings[\"model_name\"]` — a key not in "
            "the Python defaults block either (which itself "
            "differs from the C++ defaults by including "
            "computing_model_part and excluding bucket_size / "
            "neighbor_search_radius / must_find_neighbor). The "
            "wrapper cannot be instantiated. "
            "(3) Even bypassing the wrapper to call the C++ "
            "binding directly hits the schema mismatch from (1). "
            "Effect on users: any attempt to use 'sliding_edge' "
            "as a process raises either NameError (wrapper "
            "path) or KratosError on missing key "
            "(direct-C++ path). Users wanting in-plane sliding-"
            "edge MPC behavior should fall back to "
            "ApplyWeakSlidingProcess (separate edge in this "
            "cable_net pitfall list) or hand-craft "
            "MasterSlaveConstraint objects via the core API. "
            "Signal: NOT the NameError, which is what this entry "
            "used to promise and is unreachable in practice. The "
            "undefined `model_part_name` at "
            "python_scripts/sliding_edge_process.py:32 is real as a "
            "static defect, but the wrapper never gets there: line 28 "
            "calls `default_settings.ValidateAndAssignDefaults("
            "settings)` with the arguments the wrong way round — the "
            "DEFAULTS object validated against the USER's Parameters "
            "— so a well-formed input dies first with "
            "'RuntimeError: Error: The item with name "
            "\"computing_model_part\" is present in this Parameters "
            "but NOT in the default values'. That is the message to "
            "match on. (Measured 2026-08-07 on Kratos 10.4.3, "
            "CableNetApplication: the wrapper Factory was called with "
            "all 8 documented keys and raised the Parameters error, "
            "never a NameError.) Beyond the wrapper: "
            "KratosError 'Getting a value that does not exist. "
            "entry string : bucket_size' (or "
            "neighbor_search_radius / must_find_neighbor / "
            "constraint_set_name) when bypassing the wrapper "
            "and calling the C++ binding with ONLY the 8 default "
            "keys; conversely KratosError on extra keys when "
            "trying to satisfy the code by adding the "
            "missing-from-defaults keys to the input — "
            "ValidateAndAssignDefaults rejects them. There is "
            "NO valid input configuration that runs end-to-end. "
            "Workaround: use ApplyWeakSlidingProcess instead, "
            "or hand-craft MasterSlaveConstraint via the "
            "ModelPart core API. "
            "(File walk sliding_edge_process.h + "
            "python_scripts/sliding_edge_process.py 2026-06-03.)",
        ],
    },
    "chimera": {
        "description": "Chimera/overset grid method for moving bodies in flow",
        "application": "ChimeraApplication",
        "capabilities": ["overset_grids", "moving_bodies", "interpolation_at_interfaces"],
        "pitfalls": [
            "[Numerical] Requires FluidDynamicsApplication Signal: ChimeraApplication cannot load without FluidDynamicsApplication: the application's Python loader imports its dependencies first, so a build missing one fails on that inner import line rather than on the user's own import statement.",
            "[Integration] Catalog template is an availability-probe STUB, not a solver \u2014 and the KratosChimeraApplication 10.4.2 wheel on PyPI is BROKEN: it ships the pybind module (KratosChimeraApplication.cpython-312-*.so) but omits libKratosChimeraApplicationCore.so it links against, so the import fails even after a successful pip install (verified 2026-06-12; the wheel RECORD lists no Core lib). Requires a Kratos source build with applications/ChimeraApplication enabled. Signal: ImportError: libKratosChimeraApplicationCore.so: cannot open shared object file: No such file or directory \u2014 immediately after 'pip install KratosChimeraApplication' succeeds.",
        ],
        "guidance": [
            "[Numerical] Hole-cutting algorithm needed",
            "[Numerical] Conservation at chimera boundaries is approximate",
        ]
    },
    "droplet_dynamics": {
        "description": "Droplet dynamics: impact, spreading, contact angles",
        "application": "DropletDynamicsApplication",
        "capabilities": ["droplet_impact", "contact_angle", "surface_tension", "two_phase"],
        "pitfalls": [
            "[Integration] There is no catalog template for this physics. The availability-probe stub generators were removed in the 2026-06-26 honesty audit, so generate_input raises rather than emitting a script that exits 0 without solving. Nor is a real template possible on a pip stack: no KratosDropletDynamicsApplication wheel is published. A Kratos source build with applications/DropletDynamicsApplication enabled is required. Signal: `import KratosMultiphysics.DropletDynamicsApplication` raises ModuleNotFoundError and the package index offers no distribution at any version; the physics key is present in the knowledge catalog while the generator registry holds no entry for it, so generate_input raises a ValueError naming the missing template and the physics key. That ValueError is openPASO's own, raised in src/backends/kratos/backend.py, not a Kratos diagnostic — nothing in Kratos is reached at all. (pip index versions KratosDropletDynamicsApplication re-run 2026-08-09: no matching distribution, confirming the wheel is still unpublished.)",
        ],
    },
    "free_surface": {
        "description": "Free-surface flow (Eulerian approach)",
        "application": "FreeSurfaceApplication",
        "capabilities": ["free_surface_tracking", "wave_propagation", "sloshing"],
        "pitfalls": [
            "[Integration] There is no catalog template for this physics. The availability-probe stub generators were removed in the 2026-06-26 honesty audit, so generate_input raises rather than emitting a script that exits 0 without solving. No KratosFreeSurfaceApplication wheel is published; a source build with that application enabled is required. Signal: `import KratosMultiphysics.FreeSurfaceApplication` raises ModuleNotFoundError and the package index offers no distribution at any version; the physics key is present in the knowledge catalog while the generator registry holds no entry for it.",
        ],
    },
    "fluid_biomedical": {
        "description": "Biomedical fluid dynamics: blood flow, hemodynamics",
        "application": "FluidDynamicsBiomedicalApplication",
        "capabilities": ["blood_flow", "WSS_computation", "aneurysm_risk", "stent_flow"],
        "pitfalls": [
            "[Integration] There is no catalog template for this physics. The availability-probe stub generators were removed in the 2026-06-26 honesty audit, so generate_input raises rather than emitting a script that exits 0 without solving. No KratosFluidDynamicsBiomedicalApplication wheel is published, so a real template is impossible on a pip stack; a source build with that application enabled (plus FluidDynamics) is required. Signal: `import KratosMultiphysics.FluidDynamicsBiomedicalApplication` raises ModuleNotFoundError while its FluidDynamicsApplication prerequisite imports; the physics key is present in the knowledge catalog while the generator registry holds no entry for it.",
        ],
        "guidance": [
            "[Numerical] Non-Newtonian blood models (Carreau-Yasuda)",
            "[Numerical] Patient-specific geometry from CT/MRI",
        ]
    },
    "fluid_hydraulics": {
        "description": "Hydraulic fluid dynamics: open channels, pipes, spillways",
        "application": "FluidDynamicsHydraulicsApplication",
        "capabilities": ["open_channel", "pipe_network", "spillway", "hydraulic_jump"],
        "pitfalls": [
            "[Integration] There is no catalog template for this physics. The availability-probe stub generators were removed in the 2026-06-26 honesty audit, so generate_input raises rather than emitting a script that exits 0 without solving. No KratosFluidDynamicsHydraulicsApplication wheel is published; a source build with that application enabled (plus FluidDynamics) is required. Signal: `import KratosMultiphysics.FluidDynamicsHydraulicsApplication` raises ModuleNotFoundError while its FluidDynamicsApplication prerequisite imports; the physics key is present in the knowledge catalog while the generator registry holds no entry for it.",
        ],
    },
    "optimization": {
        "description": "General optimization framework: gradient-based, adjoint, multi-objective",
        "application": "OptimizationApplication",
        "capabilities": ["gradient_based", "adjoint_sensitivity", "constraint_handling",
                         "multi_objective", "response_function_library"],
        "pitfalls": [
            "[Numerical] MoveMeshFlag=True (the last ResidualBasedNewtonRaphsonStrategy ctor argument, as in the solid reference template) silently corrupts every GEOMETRIC response: the strategy moves the node coordinates by the converged displacements, so mass/volume responses integrate on the deformed mesh. Pass False for response evaluation. Signal: MassResponseUtils.CalculateValue returns e.g. 5.94 instead of the exact rho*L*h*t = 4.0 and the mass gradient becomes non-uniform; the compliance FD-check still passes (consistent configuration), which masks the defect. (Verified empirically 2026-06-12.)",
            "[API] On Kratos 10.4.2 and 10.4.3, KOA.ResponseUtils.*.CalculateGradient wants a DoubleCombinedTensorAdaptor, not a numpy array or raw VariableTensorAdaptor: build it exactly like MasterControl.GetPhysicalKratosVariableMap does \u2014 inner DoubleTensorAdaptor(VariableTensorAdaptor(mp.Elements, KM.THICKNESS), copy=False), combined adaptor with perform_collect_data_recursively=False, perform_store_data_recursively=False; read cta.data (a numpy view \u2014 copy it before reuse). Per-element design variables also need per-element Properties first: KOA.OptimizationUtils.CreateEntitySpecificPropertiesForContainer(mp, mp.Elements, False). Signal: TypeError: incompatible function arguments on CalculateGradient with anything but the combined adaptor; with shared Properties, perturbing THICKNESS changes EVERY element at once \u2014 the FD check returns the SUM of all sensitivities instead of one. In 10.3.0 and 10.4.0 CalculateGradient takes list_of_container_expressions (ContainerExpressions such as Kratos.Expression.ElementExpression) instead, and KM.TensorAdaptors does not exist in 10.3.x. (Verified empirically 2026-06-12; the version split read from the 10.3.0, 10.4.0 and 10.4.2 sources.)",
            "[Physics] Plane-STRAIN constitutive laws ignore THICKNESS \u2014 for thickness-design optimization use LinearElasticPlaneStress2DLaw (the 2D solid element scales integration weights by the THICKNESS property); set both THICKNESS and DENSITY on the Properties. Signal: thickness gradient is meaningless/zero-effect under a plane-strain law; MassResponseUtils.Check(mp) raises if DENSITY (or THICKNESS in 2D) is missing. (Verified empirically 2026-06-12.)",
            "[API] POINT_LOAD lives in StructuralMechanicsApplication, not core (SMA.POINT_LOAD; KM.POINT_LOAD does not exist), must be added via AddNodalSolutionStepVariable BEFORE creating nodes, and needs a PointLoadCondition2D1N on the loaded node \u2014 the nodal value alone is never assembled. Same module split for SMA.THICKNESS_SENSITIVITY. Signal: AttributeError: Module KratosMultiphysics has no attribute POINT_LOAD; or rc=0 with exactly zero deflection when the condition is missing. (Verified empirically 2026-06-12.)",
            "[API] The python-layer LinearStrainEnergyResponseFunction class requires the full OptimizationProblem + ExecutionPolicyDecorator stack wrapping an AnalysisStage; for a programmatic ModelPart the compiled statics KOA.ResponseUtils.LinearStrainEnergyResponseUtils / MassResponseUtils.CalculateValue/CalculateGradient give the same values directly. Signal: RuntimeError/KeyError about a missing execution policy or optimization problem component when instantiating the python response class standalone. (Verified empirically 2026-06-12.)",
        ],
        "guidance": [
            "[Numerical] Adjoint requires application-specific adjoint solver support",
        ]
    },
}

# 2026-06-26 honesty audit: the nine availability-probe stub generators
# (wind_engineering_2d, thermal_dem_2d, swimming_dem_2d, fem_to_dem_2d,
# chimera_2d, droplet_dynamics_2d, free_surface_2d, fluid_biomedical_2d,
# fluid_hydraulics_2d) were removed — their apps are not importable and the
# stubs ran no solver. Only genuine parameterized solve generators remain.
GENERATORS = {
    "poromechanics_2d": _poromechanics_2d,
    "shallow_water_2d": _shallow_water_2d,
    "dam_2d": _dam_2d,
    "constitutive_laws_2d": _constitutive_laws_2d,
    "dem_structures_2d": _dem_structures_2d,
    # PhysicsCapability name is 'dem_structures_coupling' (see
    # backend.py), so generate_input() builds the key as
    # 'dem_structures_coupling_2d' — alias to the same template
    # to keep the dispatch consistent with the catalog name.
    "dem_structures_coupling_2d": _dem_structures_2d,
    "cable_net_2d": _cable_net_2d,
    "optimization_2d": _optimization_2d,
}
