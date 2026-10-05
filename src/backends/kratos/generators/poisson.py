"""Kratos Poisson equation generators and knowledge."""


from ._convdiff_real import CROSS_CHECK_NOTE, real_convdiff_script


def _poisson_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE - values are defaults, determine appropriate values for your specific problem.

    Poisson -div(k grad u) = f on a box, u = g on the boundary, solved BY
    KRATOS (ConvectionDiffusionApplication, LaplacianElement2D3N).

    This used to emit a numpy/scipy assembly with no `import
    KratosMultiphysics` in it. It was labelled "(manual assembly)", which was
    honest but still the wrong artefact: an agent told to solve with Kratos got
    a script that cannot run Kratos, and its output cannot be attributed to
    the code the task named. Measured consequence -- a coupled 4C+Kratos task
    got numerically right answers from a run whose two participants were both
    this template, having invoked neither code.
    """
    nx = params.get("nx", 32)
    ny = params.get("ny", nx)
    f_val = params.get("f", 1.0)
    return real_convdiff_script(
        title=f"Poisson -div(k grad u) = {f_val} on a box, u = 0 on the boundary",
        nx=nx, ny=ny, k=params.get("k", 1.0),
        f_expr=f"{f_val}", g_expr="0.0",
        x0=params.get("x0", 0.0), x1=params.get("x1", 1.0),
        y0=params.get("y0", 0.0), y1=params.get("y1", 1.0))


KNOWLEDGE = {
    "poisson": {
        "description": "Poisson/diffusion via Kratos ConvectionDiffusionApplication",
        "application": "ConvectionDiffusionApplication (pip install KratosConvectionDiffusionApplication)",
        "elements": ["LaplacianElement2D3N/3D4N (steady Laplace only)",
                     "EulerianConvDiff2D3N/3D4N (convection-diffusion, transient)"],
        "solver_types": ["stationary", "transient (theta scheme: 0=FE, 0.5=CN, 1=BE)"],
        "variables": {
            "unknown": "TEMPERATURE",
            "diffusion": "CONDUCTIVITY (property on Properties object)",
            "source": "HEAT_FLUX (nodal solution step variable)",
            "reaction": "REACTION_FLUX",
            "convection": "CONVECTION_VELOCITY (for transport problems)",
        },
        "settings_object": "ConvectionDiffusionSettings — must be set on ProcessInfo, maps variable names",
        "pitfalls": [
            "[API] ConvectionDiffusionApplication element names \u2014 LaplacianElement2D3N / LaplacianElement3D4N / EulerianConvDiff2D3N / EulerianConvDiff3D4N \u2014 are C++-registered Kratos elements accessible ONLY through the string-typed factory call: model_part.CreateNewElement(\"LaplacianElement2D3N\", id, node_id_list, properties). They are NOT exposed as Python attributes on KratosMultiphysics.ConvectionDiffusionApplication, so CDA.LaplacianElement2D3N raises AttributeError. Wrong-named strings (e.g. \"ConvDiff2D3N\" without the \"Eulerian\" prefix) are rejected at CreateNewElement with \"The Element 'X' is not registered!\". Also: in a fresh install of KratosMultiphysics 10.4.2 the CDA sub-application is NOT included by default \u2014 pip install KratosConvectionDiffusionApplication is the separate package needed before this element family is usable. Signal: hasattr(CDA, \"LaplacianElement2D3N\") is False; mp.CreateNewElement(\"LaplacianElement2D3N\", ...) returns a Kratos Element; mp.CreateNewElement(\"ConvDiff2D3N\", ...) raises with \"is not registered\". (Verified empirically 2026-06-01 \u2014 Tier-2 fixture poisson_cda_element_string_factory in scripts/tier2_fixtures/kratos/.)",
            "[Numerical] LaplacianElement2D3N / LaplacianElement3D4N DO assemble the HEAT_FLUX volumetric source term when the ConvectionDiffusionSettings on ProcessInfo declares HEAT_FLUX as the VolumeSourceVariable. On a P1 unit-right triangle, LaplacianElement2D3N.CalculateRightHandSide with HEAT_FLUX=10 set on all 3 nodes returns the consistent load RHS_i = 10 * area / 3 = 1.66667 on every node \u2014 classic linear-shape-function integration of a constant source. (Catalog falsification verified empirically 2026-06-01 \u2014 Tier-2 fixture poisson_laplacian_element_assembles_heat_flux. The prior catalog claim that this element \"does NOT assemble HEAT_FLUX\" was WRONG and has been corrected.) Signal: with ConvectionDiffusionSettings.SetVolumeSourceVariable(HEAT_FLUX) and HEAT_FLUX set on nodes, RHS node values equal source * triangle_area / 3 for LaplacianElement2D3N; with HEAT_FLUX=0 the RHS is exactly zero.",
            "[Numerical] LaplacianElement2D3N and EulerianConvDiff2D3N are NOT interchangeable, not even with zero velocity in a stationary solve \u2014 swapping them can silently destroy the solution. Both assemble the same HEAT_FLUX volume source, but EulerianConvDiff always adds the mass term M/dt and weights its diffusion stiffness by ProcessInfo[TIME_INTEGRATION_THETA]; if THETA is never set (a hand-built strategy) it reads as 0 and the LHS is M/dt with NO conductivity term. Measured element-level on a unit right triangle (nodal and Properties CONDUCTIVITY = 1, zero velocity, DELTA_TIME = 1): LaplacianElement2D3N LHS = [[1,-0.5,-0.5],[-0.5,0.5,0],[-0.5,0,0.5]] (the P1 stiffness matrix); EulerianConvDiff2D3N with THETA unset LHS = [[0.0833,0.0417,0.0417],[...]] = the consistent MASS matrix; with THETA = 1 (what Kratos' stationary solver sets) LHS = [[1.0833,-0.4583,-0.4583],[...]] = M/dt + K; all RHS = 1.6667. Consequence at system level with THETA unset, 12x12 P1 unit square, f = 2*pi^2*sin(pi x)sin(pi y), u = 0 on the boundary, ResidualBasedLinearStrategy: LaplacianElement2D3N gives max T = 0.9755 (exact 1.0), while EulerianConvDiff2D3N returns a field with NO diffusion in it. WHAT that wrong field looks like depends on DELTA_TIME, so do NOT use \"all zeros\" as the test: with DELTA_TIME = 1.0 the same case gives max T = 19.7392 = 2*pi^2, i.e. T reproduces the SOURCE field (LHS = M/dt, so T = M^-1 M f = f); with DELTA_TIME = 0 (easy to hit by setting ProcessInfo[TIME] and then calling CloneTimeStep with the same value) the system degenerates, Kratos prints \"[WARNING] ResidualBasedBlockBuilderAndSolver: ATTENTION! setting the RHS to zero!\" and T comes out 0.0 everywhere. With THETA = 1 one solve is a backward-Euler step of size dt from the initial field (its result changes with dt), not the steady solution. Use EulerianConvDiff inside a Kratos convection-diffusion solver, which sets TIME_INTEGRATION_THETA (the stationary solver 1.0, the transient solver transient_parameters.theta, default 0.5), or LaplacianElement for steady problems. Signal: check the ELEMENT, not the field \u2014 with TIME_INTEGRATION_THETA unset, CalculateLocalSystem on one triangle returns the consistent mass matrix (0.0833/0.0417 for a unit right triangle) instead of the P1 stiffness (1/-0.5), and the LHS scales with 1/dt instead of being dt-independent. (Verified by execution 2026-08-03 on Kratos 10.4.0; system-level wording re-derived 2026-08-03 (re-audit) \u2014 the prior catalog claim that the two elements \"differ by less than 1e-12 relative norm\" with zero convection was WRONG, and the 2026-08-03 replacement text \"TEMPERATURE == 0.0 at EVERY node \u2014 no exception, no warning\" was ALSO wrong: it reported a DELTA_TIME = 0 artifact of its own fixture, and a warning IS printed. The THETA dependence was measured on Kratos 10.3.0.)",
            "[Integration] ConvectionDiffusionSettings MUST be set on ProcessInfo before solve. Signal: Solve() without them crashes the process with SIGSEGV (exit 139, no Python message) at the first element assembly; strategy.Check() run first instead raises 'No CONVECTION_DIFFUSION_SETTINGS defined in ProcessInfo.'. Only the read is silent: on a ModelPart that never had the settings assigned, ProcessInfo[CONVECTION_DIFFUSION_SETTINGS] returns None instead of raising \u2014 and that read itself inserts the key, so a subsequent ProcessInfo.Has(...) reports True. Guarding with Has() after a read therefore passes on a model that has no settings, and it also defeats the element's own Has() check in Check(), which then segfaults too. (Measured on Kratos 10.3.0.)",
            "[Numerical] CONDUCTIVITY for LaplacianElement2D3N is read NODALLY (via ConvectionDiffusionSettings.GetDiffusionVariable), NOT from the Properties object. Swap test on a unit right triangle, reading LHS[0][0]: nodal 1 + Properties 1 -> 1.0; nodal 1 + Properties 999 -> 1.0 (Properties IGNORED); nodal 999 + Properties 1 -> 999.0. So SetSolutionStepValue(CONDUCTIVITY, k) on every node is mandatory; setting it only on Properties gives a zero-diffusivity (singular) system. DENSITY / SPECIFIC_HEAT follow the same settings-driven lookup. Signal: the solution scales with the nodal value and is unaffected by the Properties value. (Verified by execution 2026-08-03 on Kratos 10.4.0 \u2014 the prior catalog text \"Properties (CONDUCTIVITY, DENSITY, SPECIFIC_HEAT) go on Properties object, NOT on nodes\" had it exactly backwards for this element, and contradicted KNOWLEDGE['curved_mms'] pitfall #1, which was right.)",
            "[Integration] Material properties assigned via Begin Properties block in .mdpa OR via Materials.json Signal: which route is authoritative is ELEMENT-dependent, and the mismatch is silent: LaplacianElement2D3N ignores the CONDUCTIVITY on the Properties object entirely and reads the diffusion variable nominated by ConvectionDiffusionSettings off the NODES, so a Properties-only or Materials.json-only assignment yields a zero-diffusivity system with no error.",
            "[Integration] VTK output: add vtk_output_process to output_processes in ProjectParameters.json Signal: the output block resolves the core module KratosMultiphysics.vtk_output_process and its Factory, wrapping the core KM.VtkOutput class \u2014 neither is an attribute of ConvectionDiffusionApplication. A wrong module name in output_processes fails at process construction, before the solve.",
        ],
    },
}

GENERATORS = {
    "poisson_2d": _poisson_2d_kratos,
}
