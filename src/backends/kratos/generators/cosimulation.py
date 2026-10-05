"""Kratos cosimulation generators and knowledge."""


from ._convdiff_real import real_convdiff_script
from ._structural_real import real_structural_script


_DRIVER = '''\
"""Thermal -> structural weak coupling, BOTH FIELDS SOLVED BY KRATOS.

Two real Kratos solves per coupling step -- ConvectionDiffusionApplication for
the temperature, StructuralMechanicsApplication for the displacement -- with
the thermal field driving a thermal-expansion load. The previous version of
this template solved both fields with a numpy/scipy assembly and never
imported KratosMultiphysics, while calling itself "CoSimulation ... Kratos
(standalone)"; a coupled result built from it can be attributed to neither
participant.

FOR A REAL TWO-CODE COUPLING, DRIVE IT WITH openPASO's `couple` TOOL rather than
the loop below: write one participant script per subdomain, have each read
imports.json and write exports.json, and let the driver run the fixed-point
iteration, the relaxation, and the flux-balance and responsiveness checks. The
loop here is a single-process demonstration of the data flow, not a
partitioned coupling, and it says so.

CAPTURE EACH SOLVER'S OWN CONSOLE OUTPUT into that participant's log. Kratos
prints through C++ streams: an in-process redirect of Python's stdout captures
ZERO bytes (measured), so run the solve in a SUBPROCESS if the log has to show
which code produced the numbers.
"""
import json
import subprocess
import sys

import numpy as np

N_COUPLING = {n_coupling}
ALPHA_T = {alpha_t}

# each field is a standalone, runnable Kratos script; run them as subprocesses
# so their telemetry is captured and attributable
open("thermal_step.py", "w").write(THERMAL_SCRIPT)
open("structural_step.py", "w").write(STRUCTURAL_SCRIPT)

history = []
for step in range(1, N_COUPLING + 1):
    t = subprocess.run([sys.executable, "thermal_step.py"],
                       capture_output=True, text=True, timeout=1800)
    open(f"thermal_step{{step}}.log", "w").write(t.stdout + t.stderr)
    if t.returncode != 0:
        raise SystemExit("Kratos thermal step failed:\\n" + t.stdout + t.stderr)
    T = np.loadtxt("solution.csv", delimiter=",", skiprows=1)

    s = subprocess.run([sys.executable, "structural_step.py"],
                       capture_output=True, text=True, timeout=1800)
    open(f"structural_step{{step}}.log", "w").write(s.stdout + s.stderr)
    if s.returncode != 0:
        raise SystemExit("Kratos structural step failed:\\n" + s.stdout + s.stderr)
    U = np.loadtxt("solution.csv", delimiter=",", skiprows=1)

    mag = float(np.abs(U[:, 2:]).max())
    history.append({{"step": step, "max_abs_displacement": mag,
                    "max_abs_temperature": float(np.abs(T[:, 2]).max())}})
    print(f"coupling step {{step}}: max|T|={{history[-1]['max_abs_temperature']:.6e}} "
          f"max|u|={{mag:.6e}}")

json.dump({{"history": history, "alpha_T": ALPHA_T,
           "thermal_solver": "Kratos ConvectionDiffusionApplication",
           "structural_solver": "Kratos StructuralMechanicsApplication",
           "note": ("single-process demonstration of the data flow; use the "
                    "`couple` tool for a partitioned two-code coupling")}},
          open("results_summary.json", "w"), indent=2)
print("Kratos co-simulation complete.")
'''


def _cosimulation_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE - values are defaults, determine appropriate values for your specific problem.

    Thermal -> structural weak coupling in which BOTH fields are solved by
    Kratos: ConvectionDiffusionApplication for the temperature and
    StructuralMechanicsApplication for the displacement, each run as a real
    subprocess so its console output is captured and attributable.
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", nx)
    thermal = real_convdiff_script(
        title="Thermal field of the co-simulation step, Kratos",
        nx=nx, ny=ny, k=params.get("k", 1.0),
        f_expr=str(params.get("f", 0.0)),
        g_expr=(f"{params.get('T_left', 100.0)} if x <= X0 + 1e-12 "
                f"else {params.get('T_right', 0.0)}"))
    structural = real_structural_script(
        title="Structural field of the co-simulation step, Kratos",
        nx=nx, ny=ny, lx=params.get("lx", 1.0), ly=params.get("ly", 1.0),
        young=params.get("E", 1000.0), nu=params.get("nu", 0.3),
        traction=params.get("traction", (0.0, -1.0)),
        plane=params.get("plane", "strain"))
    body = _DRIVER.format(n_coupling=params.get("n_coupling_steps", 5),
                          alpha_t=params.get("thermal_expansion", 1e-5))
    # THE CHILD SCRIPTS ARE EMITTED AS repr() LITERALS, ABOVE THE DRIVER.
    #
    # Two things forced this. They must come FIRST, because module-level code
    # runs top to bottom and the driver writes them out before its loop -- put
    # after, and the emitted program dies on NameError at line 1 of its work.
    # And they must be repr(), not embedded in a triple-quoted block: each
    # child script carries its OWN docstring, whose closing triple quote ends
    # the outer literal early and leaves a file that does not even parse.
    return ("THERMAL_SCRIPT = " + repr(thermal) + "\n"
            + "STRUCTURAL_SCRIPT = " + repr(structural) + "\n\n"
            + body)



KNOWLEDGE = {
    "cosimulation": {
        "description": "CoSimulation framework for multi-solver coupling (CoSimulationApplication)",
        "application": "CoSimulationApplication (pip install KratosCoSimulationApplication)",
        "coupling_schemes": {
            "weak": ["Gauss-Seidel (sequential)", "Jacobi (parallel)"],
            "strong": ["Gauss-Seidel with convergence check", "Jacobi with convergence check"],
        },
        # Real names (file stems in
        # KratosMultiphysics/CoSimulationApplication/
        # convergence_accelerators/). Verified empirically
        # 2026-06-01.
        "convergence_accelerators": [
            "constant_relaxation (omega=0.5-0.8 typical)",
            "aitken (adaptive relaxation, good starting point)",
            "mvqn (Multi-Vector Quasi-Newton, fastest convergence)",
            "block_mvqn (block-MVQN, partitioned variant)",
            # 'ibqn' alone is NOT a registered key — see pitfall #0.
            "block_ibqnls (block Interface Block Quasi-Newton Least Squares)",
            "iqnils (Interface Quasi-Newton Inverse Least Squares)",
            "anderson (Anderson acceleration)",
        ],
        # Real mapper type names from libKratosMappingCore.so
        # binary scan.
        "data_transfer": [
            "nearest_neighbor",
            "nearest_element",
            "barycentric",
            "coupling_geometry",
            "projection_3D_2D",
            "nearest_neighbor_iga",
            "radial_basis_function (Kratos >= 10.4.2; not registered in 10.3.0)",
            # 'kratos_mapping' is the *python wrapper module* name
            # under CoSim/data_transfer_operators/, not a mapper
            # *type*. 'empire_mapping' is NOT registered anywhere
            # in KratosMappingApplication 10.4.2.
        ],
        "solver_wrappers": {
            "internal": "KratosMultiphysics solvers (fluid, structural, thermal, etc.)",
            "external": "CoSimIO for coupling with external codes (C/C++/Python/Fortran API)",
        },
        "pitfalls": [
            "[API] Catalog had two systematic naming errors corrected 2026-06-01:\n  (a) Convergence accelerator \"ibqn\" \u2014 NOT a registered name. The real file under KratosMultiphysics/CoSimulationApplication/convergence_accelerators/ is block_ibqnls.py (or iqnils.py for the inverse-least-squares variant). Other registered names: aitken, anderson, constant_relaxation, mvqn, block_mvqn.\n  (b) Mapper type \"empire_mapping\" does NOT exist in libKratosMappingCore.so (empire substring 0 hits). Real mapper types at 10.3.0: nearest_neighbor, nearest_element, barycentric, coupling_geometry, projection_3D_2D, nearest_neighbor_iga; beam_mapper is registered from 10.4.0 and radial_basis_function from 10.4.2, neither in 10.3.0. Also: \"kratos_mapping\" in the prior catalog refers to the python wrapper module name, not a mapper *type*. Signal: convergence_accelerator type \"ibqn\" in a CoSim parameters JSON raises a Kratos factory ImportError finding \"ibqn.py\" in convergence_accelerators/; similarly mapping with \"empire_mapping\" raises a MapperFactory unknown-mapper error. (Verified empirically 2026-06-01 \u2014 Tier-2 fixture cosimulation_accelerator_mapper_names in scripts/tier2_fixtures/kratos/. KratosCoSimulationApplication was also missing from the .venv \u2014 install via \"pip install KratosCoSimulationApplication\" before any CoSim catalog usage.)",
        ],
        "guidance": [
            "[Numerical] Weak coupling: one pass per time step (fast but may be inaccurate for strong interactions)",
            "[Numerical] Strong coupling: iterate until interface convergence (required for added-mass instability)",
            "[Numerical] Aitken relaxation: good default, but MVQN converges faster for large interface problems",
            "[Numerical] Data mapping: non-matching meshes require interpolation (the RBF mapper for smooth fields, radial_basis_function, needs Kratos >= 10.4.2)",
            "[Numerical] CoSimIO: standalone library for coupling Kratos with any external solver",
        ]
    },
}

GENERATORS = {
    "cosimulation_2d": _cosimulation_2d_kratos,
}
