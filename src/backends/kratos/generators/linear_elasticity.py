"""Kratos linear elasticity generators and knowledge."""


from ._structural_real import real_structural_script


def _elasticity_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE - values are defaults, determine appropriate values for your specific problem.

    Linear elasticity on a rectangle, clamped left edge, traction on the right,
    solved BY KRATOS (StructuralMechanicsApplication,
    SmallDisplacementElement2D4N + LinearElasticPlaneStrain/Stress2DLaw).

    The previous body emitted a numpy/scipy assembly whose own first line read
    "Linear elasticity: rectangular domain, fixed left - Kratos (manual
    assembly)" and which never imported KratosMultiphysics. A result from
    it cannot be attributed to Kratos, which fails any task that names the
    code -- and every coupled task does. See _structural_real for the API
    facts, each of which was a corrected first guess.

    Verified by execution on a slender cantilever at about 10:1: bilinear
    quads come in a couple of per cent UNDER the Euler-Bernoulli slender
    estimate P L^3/(3 E I), and further still under Timoshenko.

    THAT DEFICIT IS THE ELEMENT BEING TOO STIFF -- shear locking -- AND IT
    IS NOT SHEAR FLEXIBILITY. An earlier version of this entry said it was,
    which is backwards: shear flexibility ADDS deflection. Timoshenko is
    Euler-Bernoulli PLUS a shear term, so a result below Euler-Bernoulli is
    even further below Timoshenko, and nothing that adds compliance can
    explain it. Measured on the verified case: Euler-Bernoulli 4.0000e-02,
    Timoshenko 4.0312e-02 (shear adds 3.1e-04), quad4 result 3.9010e-02 --
    2.5% under the first and 3.2% under the second.

    The practical consequence is the opposite of the old advice: a few per
    cent under is the ordinary locking of a bilinear quad at this aspect
    ratio and is tolerable, but it is an ERROR SOURCE and it gets worse as
    the element gets thinner, not better. Refining in-plane barely helps;
    the cure is a quadratic element. structural_dynamics.py carries the
    measured recovery curve and reaches the same conclusion, which this
    entry used to contradict.

    Derive the slender estimate for your own L, E, I and P and compare, and
    compute Timoshenko too if the beam is not slender -- the gap between the
    two tells you whether shear matters at all for your geometry.
    """
    nx = params.get("nx", 40)
    return real_structural_script(
        title="Linear elasticity: rectangle, clamped left, traction right",
        nx=nx, ny=params.get("ny", max(1, nx // 10)),
        lx=params.get("lx", 10.0), ly=params.get("ly", 1.0),
        young=params.get("E", 1.0e5), nu=params.get("nu", 0.3),
        rho=params.get("rho", 0.0),
        traction=params.get("traction", (0.0, -1.0)),
        plane=params.get("plane", "strain"))


def _elasticity_nonlinear_kratos(params: dict) -> str:
    """NOT A RUNNABLE INPUT - this variant serves no template.

    This variant used to emit a hand-written Total-Lagrangian solver
    (numpy/scipy, no `import KratosMultiphysics`). The real Kratos route that
    replaced it was first measured NOT to converge; the cause is the strategy
    call, not Kratos and not the tangent. No template is served because the
    route's builder (_structural_real.real_nonlinear_script) still passes
    MoveMeshFlag False.

    What was measured on Kratos 10.3.0 (40 x 4 triangle mesh, L = 10, h = 1,
    E = 1e5, nu = 0.3, traction 1, 5 load steps, DisplacementCriteria(1e-9,
    1e-12), 30 max iterations):
      * TotalLagrangianElement2D3N + CLA.KirchhoffSaintVenantPlaneStrain2DLaw
        builds and runs. The St-Venant-Kirchhoff and HyperElastic laws are in
        ConstitutiveLawsApplication, NOT StructuralMechanicsApplication, which
        carries only the linear ones.
      * ResidualBasedNewtonRaphsonStrategy(mp, scheme, criteria, builder, 30,
        True, False, False), i.e. MoveMeshFlag (the 8th positional argument)
        False, hits MAX ITERATIONS on all five steps: the absolute displacement
        norm never changes, so the ratio only falls as 1/k (1, 0.5, 0.333, ...).
        TotalLagrangian builds F from the CURRENT node coordinates
        (total_lagrangian.cpp, GetGeometry().Jacobian), and those move only if
        the strategy moves the mesh. With MoveMeshFlag False, F stays the
        identity and the residual never changes. That is a missing mesh
        update, not a wrong tangent, and the returned iterate is no solution.
      * With MoveMeshFlag True, the StructuralMechanics solver default
        "move_mesh_flag": true, every load step converges in 4 iterations
        (ratio 1, 3e-4, 4e-7, 3e-13), and the tip deflection agrees with
        SmallDisplacementElement2D3N + LinearElasticPlaneStrain2DLaw on the
        same triangle mesh to 2e-5 relative, as it must at this small load.

    Use the LINEAR route, which is verified: `linear_elasticity/2d` emits
    SmallDisplacementElement2D4N with LinearElasticPlaneStrain2DLaw and
    reproduces a slender cantilever tip deflection to within a few per cent
    of the Euler-Bernoulli estimate. For genuinely large deformation, 4C's
    SOLID element with KINEM nonlinear is an alternative: prepare_simulation(
    'fourc', 'nonlinear elasticity') selects 4C's linear_elasticity/nonlinear_3d
    template, SOLID HEX8 with KINEM nonlinear, which ran rc 0 with a converged
    Newton step on the March 2026 development build and on 4C 2026.3.0. In 2-D,
    4C up to 2026.1.0 (and that March 2026 development build) uses the WALL
    element; the 2026.2.0 release removed WALL, and a 2-D SOLID with
    THICKNESS and PLANE_ASSUMPTION takes its place.
    """
    return (
        "# =====================================================\n"
        "# Kratos reference entry: linear_elasticity / 2d_nonlinear\n"
        "# NOT a runnable input - no template is served for this variant.\n"
        "# =====================================================\n"
        "# Measured on Kratos 10.3.0:\n"
        "#   TotalLagrangianElement2D3N with\n"
        "#   ConstitutiveLawsApplication.KirchhoffSaintVenantPlaneStrain2DLaw\n"
        "#   under ResidualBasedNewtonRaphsonStrategy converges (4 iterations\n"
        "#   per load step) only when MoveMeshFlag, the 8th positional\n"
        "#   argument, is True - the StructuralMechanics solver default\n"
        "#   \"move_mesh_flag\": true. TotalLagrangian computes F from the\n"
        "#   current node coordinates; with MoveMeshFlag False the residual\n"
        "#   never changes, the displacement ratio falls only as 1/k and\n"
        "#   every load step hits MAX ITERATIONS. That is a missing mesh\n"
        "#   update, not a wrong tangent.\n"
        "#\n"
        "# No template is served for this variant. Two routes that ARE verified:\n"
        "#   prepare_simulation('kratos', 'linear_elasticity')  -> small\n"
        "#     displacement, SmallDisplacementElement2D4N, cantilever tip\n"
        "#     within a few per cent of the Euler-Bernoulli estimate\n"
        "#   prepare_simulation('fourc', 'nonlinear elasticity') -> SOLID HEX8\n"
        "#     with KINEM nonlinear for genuine large deformation (a 2-D 4C\n"
        "#     solid is WALL up to 4C 2026.1.0; the 2026.2.0 release removed\n"
        "#     WALL, and a 2-D SOLID takes THICKNESS and PLANE_ASSUMPTION)\n"
        "# =====================================================\n")


KNOWLEDGE = {
    "linear_elasticity": {
        "description": "Structural mechanics via StructuralMechanicsApplication (SMA)",
        "application": "StructuralMechanicsApplication (pip install KratosStructuralMechanicsApplication)",
        "elements": {
            "2D": ["SmallDisplacementElement2D3N/4N/6N/8N/9N (linear, small strain)",
                   "TotalLagrangianElement2D3N/4N (nonlinear, large deformation)",
                   "UpdatedLagrangianElement2D3N/4N"],
            "3D": ["SmallDisplacementElement3D4N/8N/10N/20N/27N",
                   "TotalLagrangianElement3D4N/8N"],
            "shells": ["ShellThinElement3D3N (MITC, Kirchhoff-Love)",
                      "ShellThickElement3D4N (Reissner-Mindlin)"],
            "beams": ["CrBeamElement3D2N (co-rotational)", "CrLinearBeamElement3D2N"],
            "trusses": ["TrussElement3D2N", "TrussLinearElement3D2N"],
            "cables": ["CableElement3D2N"],
            "springs": ["SpringDamperElement3D2N", "NodalConcentratedElement2D1N/3D1N"],
        },
        # Every name below was checked against the INSTALLED Kratos with
        # KratosGlobals.HasConstitutiveLaw (10.4.0, /usr/bin/python3, 2026-08-03)
        # and RE-checked 2026-08-03 (re-audit) with ONLY StructuralMechanicsApplication +
        # ConstitutiveLawsApplication imported. That distinction matters: the
        # registry is filled per IMPORTED APPLICATION, so a name checked in a
        # process that also imported MPMApplication/DamApplication can be
        # "registered" and still be unusable in a structural job. Confirm with
        # KM.ReadMaterialsUtility, not just HasConstitutiveLaw.
        "constitutive_laws": {
            "linear": ["LinearElastic3DLaw", "LinearElasticPlaneStrain2DLaw",
                       "LinearElasticPlaneStress2DLaw", "LinearElasticAxisym2DLaw",
                       "TrussConstitutiveLaw", "BeamConstitutiveLaw"],
            "hyperelastic": ["KirchhoffSaintVenant3DLaw (Saint Venant-Kirchhoff; also "
                             "KirchhoffSaintVenantPlaneStrain2DLaw / ...PlaneStress2DLaw)",
                             "HyperElasticSimoTaylorNeoHookean3DLaw / "
                             "HyperElasticSimoTaylorNeoHookeanPlaneStrain2DLaw "
                             "(the Neo-Hookean family that IS available to a "
                             "StructuralMechanics job)",
                             "NOT HyperElasticNeoHookean3DLaw / "
                             "HyperElasticNeoHookeanPlaneStrain2DLaw / "
                             "HyperElasticNeoHookeanAxisym2DLaw — those are "
                             "MPMApplication laws. With only StructuralMechanics + "
                             "ConstitutiveLaws imported, HasConstitutiveLaw is False "
                             "and Materials.json dies with 'Error: Kratos components "
                             "missing \"HyperElasticNeoHookean3DLaw\"'. They appear "
                             "only after 'import KratosMultiphysics.MPMApplication'.",
                             "HyperElastic3DLaw (registered, but a DIFFERENT C++ class from "
                             "KirchhoffSaintVenant3DLaw — do not treat it as the SVK law)"],
            "plasticity": ("SmallStrainIsotropicPlasticity3D<YieldSurface><PlasticPotential> — "
                           "5 yield surfaces x 5 plastic potentials, but only 23 of the 25 "
                           "pairings are registered in 10.4.0 (MohrCoulombModifiedMohrCoulomb "
                           "and ModifiedMohrCoulombMohrCoulomb are absent). See KNOWLEDGE["
                           "'plasticity'] for the hardening-curve enum."),
            "damage": ["SmallStrainIsotropicDamageFactory (JSON factory)",
                       "SmallStrainIsotropicDamage3D<YieldSurface> (e.g. ...3DVonMises)",
                       "SmallStrainDplusDminusDamage<TensionYS><CompressionYS><2D|3D> "
                       "(tension/compression split)"],
            "viscoelastic": ["ViscousGeneralizedMaxwell3D (relaxation)",
                             "ViscousGeneralizedKelvin3D (creep)"],
        },
        "solver_types": ["static (Newton-Raphson)",
                        "dynamic (Newmark, Bossak, pseudo_static, backward_euler/bdf1-bdf5, relaxation)",
                        "dynamic with time_integration_method explicit (central_differences, multi_stage)",
                        "formfinding"],
        "pitfalls": [
            "[Syntax] Element names in the .mdpa MUST include the node-count suffix: SmallDisplacementElement2D3N, not SmallDisplacement2D. Kratos resolves element types via a registry keyed by the full name. Signal: ModelPart.CreateNewElement('SmallDisplacement2D', ...) raises RuntimeError 'is not registered!' — the whole line being Error: The Element \"SmallDisplacement2D\" is not registered!, then 'Maybe you need to import the application where it is defined?' and 'The following Elements are registered:' with the list — while the same call with SmallDisplacementElement2D3N succeeds. All three strings are literals in kratos/python/add_model_part_to_python.cpp (ModelPartCreateNewElement: line 159 in 10.3.0, 154 in 10.4.0 to 10.4.3); only the element name between them is interpolated. An earlier wording offered 'Trying to construct an element with a wrong name' as an alternative message: no such text exists anywhere in the Kratos source or in any installed library, and it is not what the call prints. (Re-measured by execution 2026-08-13 on Kratos 10.4.3, with and without StructuralMechanicsApplication imported — the wording is the same either way.)",
            "[Integration] Materials are defined in StructuralMaterials.json, referenced from the .mdpa by Properties ID. Defining material parameters inline in the .mdpa via 'Begin Properties N' works for simple cases but breaks for laws that need Tables (temperature-dependent E, hardening curves). Signal: Element.Initialize raises RuntimeError 'A constitutive law needs to be specified for the element with ID N' from applications/StructuralMechanicsApplication/custom_elements/solid_elements/base_solid_element.cpp when the Property has YOUNG_MODULUS / POISSON_RATIO set but no CONSTITUTIVE_LAW. (Verified empirically 2026-06-01 \u2014 prior catalog text said 'No constitutive law assigned to Property X' and pointed at AnalysisStage.Initialize; the real error message references the element ID, not the Property, and originates in base_solid_element.cpp:249.)",
            "[Syntax] SubModelPart names must match EXACTLY between .mdpa and ProjectParameters.json \u2014 Kratos is case-sensitive and does not strip whitespace. Signal: RuntimeError 'Error: There is no sub model part with name \"NAME\" in model part \"PARENT\"' from ModelPart::ErrorNonExistingSubModelPart in model_part.cpp, listed alongside the available SubModelPart names. (Verified empirically 2026-06-01 \u2014 prior wording 'SubModelPart ... does not exist' used CamelCase; the real error text is lowercase 'sub model part' with spaces.)",
            "[Numerical] For nonlinear analyses: increase the max_iteration argument passed to ResidualBasedNewtonRaphsonStrategy (it is a positional constructor argument on the Strategy, NOT a field on the ResidualCriteria). The Python solver wrappers typically pull it from the JSON solver_settings.max_iteration field; default 10 may not suffice for material nonlinearity or large deformation. Signal: max_iteration is a POSITIONAL argument of the ResidualBasedNewtonRaphsonStrategy constructor: omitting it raises a pybind11 TypeError whose only literal is 'arguments. The following argument types are supported:'; the head comes from the bound signature, so the line reads __init__(): incompatible constructor arguments. followed by the overload list. It cannot be defaulted away. It is not an attribute of ResidualCriteria, so setting it there has no effect at all.",
            "[API] DISPLACEMENT variable is the structural DOF; ROTATION is required additionally for beams and shells. Without ROTATION added to the ModelPart variables list, the beam element can be created and Initialize succeeds, but the failure surfaces when the solver/strategy tries to compute rotational DOFs (Solve / Check). Signal: the predictable Kratos pattern for missing-variable errors fires at the first GetSolutionStepValue(ROTATION_*) inside the strategy: RuntimeError 'This container only can store the variables specified in its variables list. The variables list doesn't have this variable: ROTATION_X/Y/Z' from variables_list_data_value_container. (Verified empirically 2026-06-01 \u2014 prior catalog text said the error fires at beam-element InitializeSolutionStep with 'not found in variables list' wording; reality is Initialize alone does NOT raise, and when it does fire later the wording matches the container error pattern, not the prior text.)",
            "[Numerical] SHEAR LOCKING: linear hex8 (3D8N) and quad4 (2D4N) elements lock in bending-dominated problems, producing overly stiff results and wrong frequencies. Use quadratic elements (3D20N, 3D27N, 2D8N, 2D9N) for any problem with significant bending. Signal: tip deflection on a cantilever beam meshed with 3D8N is 20-40% smaller than analytic; switching to 3D20N recovers it within 1-2%.",
            "[API] For POINT_LOAD on NODES: use AssignVectorVariableProcess with constrained: [false, false, false]. There is no AssignVectorByDirectionProcess class on StructuralMechanicsApplication, but there IS a core Python process module KratosMultiphysics.assign_vector_by_direction_process \u2014 its default \"constrained\": true makes it fix/free the DOF, so on a load variable it fails unless \"constrained\": false is given. Signal: two different failures depending on the route taken. (a) SMA.AssignVectorByDirectionProcess -> AttributeError (hasattr is False). (b) KratosMultiphysics.assign_vector_by_direction_process.Factory(...) with variable_name POINT_LOAD and the default constrained -> RuntimeError 'Error: Trying to fix/free dof of variable POINT_LOAD_X but this dof does not exist in node #1!'; with \"constrained\": false the same call sets POINT_LOAD = [0, -100, 0]. assign_vector_variable_process on the same model part sets POINT_LOAD = [0, -100, 0] cleanly. For loads carried by CONDITIONS use assign_vector_by_direction_to_condition_process, which works. (Verified by execution 2026-08-03 on Kratos 10.4.0 \u2014 supersedes the 2026-06-01 note, which took route (a) alone and concluded that the named class was simply not available to crash; route (b) does crash, and with the originally-documented message.)",
            "[Syntax] problem_data section MUST include the 'echo_level' field. Kratos accesses it during stage initialisation without a default. Signal: Parameters::GetValue raises RuntimeError 'Error: Getting a value that does not exist. entry string : echo_level' from kratos/sources/kratos_parameters.cpp when problem_data omits the field. (Verified empirically 2026-06-01 \u2014 prior catalog text said KeyError from RunSolutionLoop; Kratos uses RuntimeError, not Python KeyError, and the message originates in C++ GetValue, not RunSolutionLoop.)",
            "[API] ConstitutiveLaw assignment requires an INSTANCE, not the class \u2014 properties.SetValue(CONSTITUTIVE_LAW, LinearElastic3DLaw) fails because the class object is passed instead of LinearElastic3DLaw(). Signal: TypeError with text 'incompatible function arguments' from the SetValue binding; the .pyi shows the second arg type is the law instance.",
            "[API] Variables (DISPLACEMENT, REACTION, VELOCITY, POINT_LOAD, etc.) must be added to the ModelPart's nodal-variables list via ModelPart.AddNodalSolutionStepVariable BEFORE any Node, Element, or Condition is created. Adding the variable after CreateNewNode raises a runtime error and the existing nodes do not get the DOF. Signal: Two distinct failure modes (both verified empirically 2026-06-01): (a) If the variable is added AFTER any Node is created, ModelPart.AddNodalSolutionStepVariable raises RuntimeError 'Attempting to add the variable \"X\" to the model part with name \"Y\" which is not empty' from ModelPart::AddNodalSolutionStepVariable in kratos/includes/model_part.h (reported as line 534 on 10.3.0; the check sits at line 521 in 10.4.2 and 10.4.3) \u2014 Kratos refuses to extend the variables list once nodes exist. (b) If the variable was NEVER added, the first GetSolutionStepValue / SetSolutionStepValue on the Node raises RuntimeError 'This container only can store the variables specified in its variables list. The variables list doesn't have this variable: X' from variables_list_data_value_container. (Prior catalog text only described mode (b) as happening 'when a process tries to read the freshly-added variable' \u2014 that is wrong; the freshly-added case is mode (a) and fires at the add call.)",
        ],
    },
}

GENERATORS = {
    "linear_elasticity_2d": _elasticity_2d_kratos,
    "linear_elasticity_2d_nonlinear": _elasticity_nonlinear_kratos,
}
