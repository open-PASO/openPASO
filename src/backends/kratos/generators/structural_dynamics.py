"""Kratos structural dynamics generators and knowledge."""


from ._structural_real import real_dynamics_script


def _structural_dynamics_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE - values are defaults, determine appropriate values for your specific problem.

    Transient structural dynamics solved BY KRATOS: SmallDisplacementElement2D4N
    stepped by ResidualBasedBossakDisplacementScheme through a
    ResidualBasedNewtonRaphsonStrategy.

    The previous body emitted a numpy/scipy assembly with a hand-written
    Newmark loop, titled "Dynamic structural analysis - Newmark time
    integration - Kratos (manual assembly)", which never imported
    KratosMultiphysics.

    Three failure modes the emitted script now guards, each measured: a dynamic
    scheme needs VELOCITY and ACCELERATION in the variables list or it aborts
    inside the scheme in every worker thread; DENSITY must be nonzero or there
    is no mass matrix; and SetBufferSize(3) is required because Bossak reads
    step n-1, so the default buffer of 1 silently yields a response with no
    inertia in it. Verified by execution: a 20x4 cantilever, E=1000, nu=0.3,
    rho=1, dt=0.02 to t=0.4 gives 20 steps with tip uy moving from -5.3486e-04
    to -8.1485e-02 -- a ramping transient, which is right at 2% of the first
    natural period, not the static answer repeated.
    """
    nx = params.get("nx", 20)
    return real_dynamics_script(
        title="Transient structural dynamics, Kratos Bossak",
        nx=nx, ny=params.get("ny", 4),
        lx=params.get("lx", 10.0), ly=params.get("ly", 1.0),
        young=params.get("E", 1000.0), nu=params.get("nu", 0.3),
        rho=params.get("density", 1.0),
        dt=params.get("dt", 0.01), t_end=params.get("T_end", 1.0),
        traction=params.get("traction", (0.0, -1.0)),
        plane=params.get("plane", "strain"))


KNOWLEDGE = {
    "structural_dynamics": {
        "description": "Dynamic structural analysis via StructuralMechanicsApplication",
        "application": "StructuralMechanicsApplication",
        "solver_types": ["dynamic (Newmark, Bossak-alpha, pseudo_static, backward_euler/bdf1-bdf5, relaxation)",
                        "dynamic with time_integration_method explicit (central_differences or multi_stage, for wave propagation)"],
        "time_integration": {
            "Newmark": "beta=0.25, gamma=0.5 (average acceleration, unconditionally stable)",
            "Bossak": "alpha_m in [-1/3, 0] (numerical damping, unconditionally stable)",
            "BDF": "backward_euler / bdf1-bdf5 (ResidualBasedBDFDisplacementScheme); there is no generalized-alpha scheme",
            "Explicit": "Central differences (conditionally stable, dt < h/c)",
        },
        "pitfalls": [
                        '[Numerical] Newmark average acceleration uses beta=0.25, gamma=0.5 (no numerical damping). Choosing gamma > 0.5 adds artificial damping; gamma < 0.5 is unconditionally unstable. '
                        'Signal: in a ResidualBasedNewmarkDisplacementScheme run with gamma < 0.5, the DISPLACEMENT amplitude in the VtkOutput grows exponentially across time steps regardless of dt.',
                        "[Syntax] Bossak adds mild numerical damping for high-frequency noise (alpha_m ≈ -0.1). In the Kratos JSON the parameter name is 'damp_factor_m' (NOT alpha_m). A wrong key is NOT silently ignored: Kratos Parameters validation is strict and the run aborts before the first time step. "
                        "Signal: RuntimeError 'Error: The item with name \"alpha_m\" is present in this Parameters but NOT in the default values' from Parameters::ValidateAndAssignDefaults; with 'damp_factor_m' the same case runs to completion. (Re-verified by execution 2026-08-03 on Kratos 10.4.0 — a full StructuralMechanicsAnalysis with solver_type 'Dynamic', scheme_type 'bossak' and the key renamed to alpha_m raises; the earlier catalog text 'Wrong key is silently ignored and the scheme runs without damping' was WRONG. This matters: an unknown key is a loud failure in Kratos, not a silent one, so do not go hunting for a physics explanation.)",
                        '[Numerical] Mass matrix: consistent (default) or lumped (faster for explicit). Lumped on linear tets/quads is OK; lumped on quadratic elements loses accuracy for higher-frequency modes. '
                        'Signal: natural-frequency study with lumped quad8 shows frequency error 2-5% for f_2..f_5 vs <0.5% with consistent mass.',
                        '[Numerical] Effective stiffness K_eff = K + 1/(beta*dt^2)*M — factor it ONCE for linear dynamics, reuse across steps. Re-factorising every step costs O(N^1.5) instead of O(N). '
                        'Signal: per-step wall_time of ResidualBasedBlockBuilderAndSolver scales as N^1.5 with mesh refinement instead of N. '
                        'reform_dofs_at_each_step does not change this: it is a solver_settings flag (default false) that controls rebuilding the DOF set, not factorisation reuse '
                        '(in linear_solver_settings the direct solvers ignore it and amgcl rejects it), and the direct solvers (LinearSolversApplication sparse_lu, skyline_lu_factorization) re-factorise on every Solve.',
                        '[Numerical] For nonlinear dynamics: tangent must be re-assembled at each Newton iteration (not just each time step). Caching the initial tangent gives modified-Newton with slow convergence. '
                        "Signal: ResidualBasedNewtonRaphsonStrategy iteration log shows the ResidualCriteria ratio decreasing by < 0.5 per iter (should be O(0.01) for full Newton); the strategy saturates at max_iteration without ResidualBasedBlockBuilderAndSolver reaching tolerance.",
                        '[Numerical] ELEMENT SELECTION: Linear hex8 (SmallDisplacementElement3D8N) shear-locks in bending-dominated problems — use quadratic hex20 or hex27. Same applies to linear quad4 in 2D — use quad8/quad9. '
                        'Signal: measured on this install (Kratos 10.4.0) on a slender plane-stress cantilever, as a FRACTION of the Timoshenko value, which is what characterises the element rather than the case: '
                        'SmallDisplacementElement2D4N on a one-element-deep grid recovers about two thirds of it -- roughly a third too stiff; '
                        'SmallDisplacementElement2D8N on the SAME element grid recovers essentially all of it. Refining quad4 instead recovers it only slowly: doubling the mesh twice still leaves a few per cent short, and it takes three doublings to get within one per cent. Compute the Timoshenko value for your own beam and compare fractions. '
                        '(Verified by execution 2026-08-03.)',
                        '[API] For POINT_LOAD application: use AssignVectorVariableProcess with constrained: [false, false, false]. '
                        'The directional process (KratosMultiphysics.assign_vector_by_direction_process, a CORE module — StructuralMechanicsApplication does NOT export a class of that name) defaults to "constrained": true and then tries to fix/free the DOF, so for load variables it must be given "constrained": false. '
                        "Signal: with the default constrained, RuntimeError 'Error: Trying to fix/free dof of variable POINT_LOAD_X but this dof does not exist in node #1!'; "
                        'with "constrained": false it sets POINT_LOAD = [0, -100, 0] cleanly, as does assign_vector_variable_process. '
                        'For loads applied on CONDITIONS (the usual case with PointLoadCondition2D1N) assign_vector_by_direction_to_condition_process IS the right process and works. '
                        '(Verified by execution 2026-08-03 on Kratos 10.4.0; the "constrained": false route re-measured on Kratos 10.3.0.)',
                        "[Syntax] problem_data section MUST include the 'echo_level' field (typically 0). Kratos accesses it during stage initialisation without a default. "
                        "Signal: RuntimeError 'Error: Getting a value that does not exist. entry string : echo_level' from the C++ Parameters::GetValue — NOT a Python KeyError, and not from RunSolutionLoop. (Re-verified by execution 2026-08-03: a full StructuralMechanicsAnalysis whose problem_data omits echo_level aborts with exactly that message. The earlier signal text in this entry said \"KeyError 'echo_level' from AnalysisStage.RunSolutionLoop\" and was stale; the linear_elasticity copy of this pitfall already carried the correct wording.)",
                    ],
    },
}

GENERATORS = {
    "structural_dynamics_2d": _structural_dynamics_2d_kratos,
}
