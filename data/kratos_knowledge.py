"""
Comprehensive Kratos Multiphysics domain knowledge for openPASO.

This module encodes deep domain knowledge about ALL major Kratos applications,
their Python API patterns, element types, constitutive laws, solver configuration,
and known pitfalls.

Kratos uses a three-file system:
  - MainKratos.py: Python driver script
  - ProjectParameters.json: solver settings, BCs, materials, output
  - mesh.mdpa: mesh data (nodes, elements, conditions, sub-model-parts)
  - (optional) StructuralMaterials.json or ThermalMaterials.json

Install: pip install KratosMultiphysics-all (core + 16 applications, plus MeshMovingApplication
  as a dependency; it resolves to the 10.3.0 set. IgaApplication, RANSApplication,
  OptimizationApplication, CableNetApplication etc. are separate wheels)
Individual: pip install KratosStructuralMechanicsApplication, etc.
"""

# ═══════════════════════════════════════════════════════════════════════════
# COMPLETE LIST OF KRATOS APPLICATIONS (~40+ applications)
# ═══════════════════════════════════════════════════════════════════════════

KRATOS_APPLICATIONS = {
    # ── Core / Infrastructure ──
    "LinearSolversApplication": {
        "physics": "Linear algebra backends (Eigen, AMGCL)",
        "description": "Provides linear solver wrappers. Always needed.",
    },
    "TrilinosApplication": {
        "physics": "Distributed-memory parallel solvers (MPI)",
        "description": "MPI-parallel linear solvers via Trilinos (AztecOO, Amesos, ML) and AMGCL.",
    },
    "MetisApplication": {
        "physics": "Graph partitioning for MPI domain decomposition",
        "description": "Interface to METIS for mesh partitioning.",
    },
    "MappingApplication": {
        "physics": "Data transfer between non-matching grids",
        "description": "Nearest-neighbor, nearest-element, barycentric mapping. Shared & distributed memory.",
    },
    "MeshMovingApplication": {
        "physics": "ALE mesh deformation",
        "description": "Laplacian, structural similarity, and superposition mesh motion solvers.",
    },
    "MeshingApplication": {
        "physics": "Mesh generation and adaptive remeshing",
        "description": "Interfaces to Triangle, TetGen, MMG. Local refinement, metric-based remeshing.",
    },

    # ── Structural Mechanics ──
    "StructuralMechanicsApplication": {
        "physics": "Solid, shell, beam, membrane, truss, cable mechanics",
        "description": "Linear/nonlinear, static/dynamic structural analysis.",
    },
    "ContactStructuralMechanicsApplication": {
        "physics": "Contact mechanics (mortar-based, penalty, ALM)",
        "description": "Frictionless and frictional contact with mortar formulations.",
    },
    "ConstitutiveLawsApplication": {
        "physics": "Advanced material models",
        "description": "Plasticity, damage, viscoelasticity, viscoplasticity, composites, fatigue.",
    },

    # ── Fluid Dynamics ──
    "FluidDynamicsApplication": {
        "physics": "Incompressible / weakly-compressible Navier-Stokes",
        "description": "VMS, QSVMS, OSS, FIC stabilized elements. Monolithic and fractional step.",
    },
    "FluidDynamicsBiomedicalApplication": {
        "physics": "Biomedical fluid flows",
        "description": "Specialized blood flow and biomedical CFD models.",
    },
    "RANSApplication": {
        "physics": "Reynolds-Averaged Navier-Stokes turbulence",
        "description": "k-epsilon, k-omega SST turbulence models.",
    },
    "CompressiblePotentialFlowApplication": {
        "physics": "Compressible potential flow (aerodynamics)",
        "description": "Full potential and transonic small perturbation formulations.",
    },

    # ── Thermal / Convection-Diffusion ──
    "ConvectionDiffusionApplication": {
        "physics": "Scalar transport, heat conduction, Poisson equation",
        "description": "Steady/transient convection-diffusion-reaction. Eulerian stabilized elements.",
    },

    # ── Multiphysics Coupling ──
    "CoSimulationApplication": {
        "physics": "Multi-code / multi-physics coupling",
        "description": "Gauss-Seidel/Jacobi schemes, convergence accelerators, data transfer.",
    },
    "FSIApplication": {
        "physics": "Fluid-Structure Interaction",
        "description": "Partitioned FSI with Dirichlet-Neumann coupling, ALE mesh movement.",
    },

    # ── Particle Methods ──
    "DEMApplication": {
        "physics": "Discrete Element Method (granular mechanics)",
        "description": "Spherical/non-spherical particles, contact laws, inlets, FEM-DEM coupling.",
    },
    "SwimmingDEMApplication": {
        "physics": "Coupled DEM-fluid (particle-laden flows)",
        "description": "DEM particles in fluid flow with drag, buoyancy forces.",
    },
    "MPMApplication": {
        "physics": "Material Point Method (large deformation geomechanics)",
        "description": "Hybrid particle-grid for landslides, impact, penetration, free-surface flows.",
    },
    "PfemFluidDynamicsApplication": {
        "physics": "Particle Finite Element Method for fluids",
        "description": "Free-surface flows, wave breaking, fluid-structure with remeshing.",
    },

    # ── Geomechanics ──
    "GeoMechanicsApplication": {
        "physics": "Soil mechanics, consolidation, groundwater flow",
        "description": "Two-phase (soil-water) formulations, Biot consolidation, slope stability.",
    },

    # ── Optimization ──
    "ShapeOptimizationApplication": {
        "physics": "Shape optimization with Vertex Morphing",
        "description": "Node-based shape optimization, adjoint sensitivity, filtering.",
    },
    "OptimizationApplication": {
        "physics": "Topology and shape optimization",
        "description": "SIMP topology optimization, gradient-based and gradient-free methods.",
    },
    "TopologyOptimizationApplication": {
        "physics": "Topology optimization (SIMP method)",
        "description": "Density-based topology optimization for structural problems.",
    },

    # ── Reduced Order Modeling ──
    "RomApplication": {
        "physics": "Reduced Order Models (POD, HROM)",
        "description": "Proper Orthogonal Decomposition, Hyper-Reduced Order Models, EmpiricalCubature.",
    },

    # ── Other Physics ──
    "ChimeraApplication": {
        "physics": "Overlapping (Chimera) mesh method",
        "description": "Overset grids for moving body problems without remeshing.",
    },
    "ConstitutiveModelsApplication": {
        "physics": "Advanced constitutive model library",
        "description": "Additional material models beyond core structural mechanics.",
    },
    "DamApplication": {
        "physics": "Dam engineering (thermo-mechanical)",
        "description": "Concrete dam analysis including thermal effects and aging.",
    },
    "PoromechanicsApplication": {
        "physics": "Porous media mechanics",
        "description": "Coupled solid-fluid in porous media (Biot theory).",
    },
    "SolidMechanicsApplication": {
        "physics": "Legacy solid mechanics (deprecated, use StructuralMechanics)",
        "description": "Older solid mechanics; most functionality moved to StructuralMechanicsApplication.",
    },
    "WindEngineeringApplication": {
        "physics": "Wind engineering and atmospheric flows",
        "description": "Wind loads on structures, atmospheric boundary layer.",
    },
    "StatisticsApplication": {
        "physics": "Runtime statistics computation",
        "description": "Temporal and spatial statistics of simulation variables.",
    },
    "CableNetApplication": {
        "physics": "Cable and net structures",
        "description": "Specialized elements for cable nets and tensile structures.",
    },
    "IgaApplication": {
        "physics": "Isogeometric Analysis",
        "description": "NURBS-based elements for exact geometry representation.",
    },
    "MedApplication": {
        "physics": "MED file format I/O",
        "description": "Read/write MED mesh format (Salome/Code_Aster compatible).",
    },
    "HDF5Application": {
        "physics": "HDF5 I/O",
        "description": "Read/write simulation data in HDF5 format.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. STRUCTURAL MECHANICS APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

STRUCTURAL_MECHANICS = {
    "application": "StructuralMechanicsApplication",
    "analysis_class": "structural_mechanics_analysis.StructuralMechanicsAnalysis",
    "import": "from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis",

    # ── Element Types ──
    # Naming convention: <ElementType><Dim>D<NumNodes>N
    "elements": {
        # Solid continuum elements
        "SmallDisplacementElement2D3N": "Linear triangle, small strain (infinitesimal)",
        "SmallDisplacementElement2D4N": "Linear quad, small strain",
        "SmallDisplacementElement2D6N": "Quadratic triangle, small strain",
        "SmallDisplacementElement2D8N": "Quadratic quad, small strain",
        "SmallDisplacementElement2D9N": "Biquadratic quad, small strain",
        "SmallDisplacementElement3D4N": "Linear tetrahedron, small strain",
        "SmallDisplacementElement3D8N": "Linear hexahedron, small strain",
        "SmallDisplacementElement3D10N": "Quadratic tetrahedron, small strain",
        "SmallDisplacementElement3D20N": "Quadratic hexahedron, small strain",
        "SmallDisplacementElement3D27N": "Triquadratic hexahedron, small strain",

        "TotalLagrangianElement2D3N": "Triangle, total Lagrangian (large strain, ref config)",
        "TotalLagrangianElement2D4N": "Quad, total Lagrangian",
        "TotalLagrangianElement2D6N": "Quadratic triangle, total Lagrangian",
        "TotalLagrangianElement2D8N": "Quadratic quad, total Lagrangian",
        "TotalLagrangianElement3D4N": "Tetrahedron, total Lagrangian",
        "TotalLagrangianElement3D8N": "Hexahedron, total Lagrangian",
        "TotalLagrangianElement3D10N": "Quadratic tetrahedron, total Lagrangian",
        "TotalLagrangianElement3D20N": "Quadratic hexahedron, total Lagrangian",

        "UpdatedLagrangianElement2D3N": "Triangle, updated Lagrangian (large strain, current config)",
        "UpdatedLagrangianElement2D4N": "Quad, updated Lagrangian",
        "UpdatedLagrangianElement3D4N": "Tetrahedron, updated Lagrangian",
        "UpdatedLagrangianElement3D8N": "Hexahedron, updated Lagrangian",

        # Shell elements
        "ShellThinElement3D3N": "Thin shell triangle (Kirchhoff, no transverse shear)",
        "ShellThinElementCorotational3D4N": "Thin shell quad (corotational)",
        "ShellThickElementCorotational3D3N": "Thick shell triangle (corotational, with transverse shear)",
        "ShellThickElement3D4N": "Thick shell quad",

        # Beam elements
        "CrBeamElement3D2N": "Co-rotational 3D Euler-Bernoulli beam",
        "CrBeamElement2D2N": "Co-rotational 2D Euler-Bernoulli beam",
        "CrLinearBeamElement3D2N": "Linear co-rotational 3D beam",
        "CrLinearBeamElement2D2N": "Linear co-rotational 2D beam",

        # Membrane elements
        "MembraneElement3D3N": "Membrane triangle (no bending stiffness)",
        "MembraneElement3D4N": "Membrane quad",

        # Truss / cable / spring elements
        "TrussElement3D2N": "3D truss (tension and compression)",
        "TrussLinearElement3D2N": "Linear 3D truss",  # LinearTrussElement3D2N is also registered
        "CableElement3D2N": "Cable element (tension only, no compression)",
        "SpringDamperElement3D2N": "Point-to-point spring + damper",

        # Nodal concentrated elements
        "NodalConcentratedElement2D1N": "Nodal mass/stiffness/damping (2D)",
        "NodalConcentratedElement3D1N": "Nodal mass/stiffness/damping (3D)",
    },

    # ── Conditions (for applying loads/BCs) ──
    "conditions": {
        "PointLoadCondition2D1N": "Point load in 2D",
        "PointLoadCondition3D1N": "Point load in 3D",
        "LineLoadCondition2D2N": "Distributed line load (2D)",
        "LineLoadCondition3D2N": "Distributed line load (3D)",
        "SurfaceLoadCondition3D3N": "Surface pressure/traction (triangle)",
        "SurfaceLoadCondition3D4N": "Surface pressure/traction (quad)",
        "PointMomentCondition3D1N": "Concentrated moment",
        "LineLoadCondition2D3N": "Quadratic line load",
        "SurfaceLoadCondition3D6N": "Quadratic surface load (triangle)",
        "SurfaceLoadCondition3D8N": "Quadratic surface load (quad)",
    },

    # ── Solver Types ──
    "solver_types": {
        "Static": "structural_mechanics_static_solver (Newton-Raphson for nonlinear, direct for linear); 'static' is accepted too",
        "Dynamic": ("time_integration_method 'implicit' (the default) -> structural_mechanics_implicit_dynamic_solver "
                    "(scheme_type newmark, bossak (default), pseudo_static, backward_euler, bdf1-bdf5, relaxation); "
                    "'explicit' -> structural_mechanics_explicit_dynamic_solver (central_differences or multi_stage, "
                    "no system solve); 'dynamic' is accepted too"),
        "formfinding": ("structural_mechanics_formfinding_solver (for membrane/cable form-finding); lower-case only: "
                        "'Formfinding' and 'Explicit' are rejected as solver_type values"),
        "eigen_value": "structural_mechanics_eigensolver",
        "harmonic_analysis": "structural_mechanics_harmonic_analysis_solver",
        "adjoint_static": "structural_mechanics_adjoint_static_solver",
        "prebuckling": "structural_mechanics_prebuckling_solver",
    },

    # ── Analysis Types ──
    "analysis_types": {
        "linear": "Single-step linear solution",
        "non_linear": "Newton-Raphson iterations with convergence check",
    },

    # ── Convergence Criteria ──
    "convergence_criteria": [
        "displacement_criterion",          # ||du||/||u|| < tol
        "residual_criterion",              # ||R|| < tol
        "and_criterion",                   # Both displacement AND residual
        "or_criterion",                    # Either displacement OR residual
    ],  # with rotation_dofs / volumetric_strain_dofs / strain_dofs true, these names switch to MixedGenericCriteria / ResidualDisplacementAndOtherDoFCriteria

    # ── Dynamic Analysis Schemes ──
    "time_integration_schemes": {
        "newmark": {"beta": 0.25, "gamma": 0.5, "description": "Newmark-beta (unconditionally stable)"},
        "bossak": {"alpha": -0.3, "description": "Bossak (numerical damping via alpha)"},
        "pseudo_static": {"description": "ResidualBasedPseudoStaticDisplacementScheme"},
        "backward_euler": {"description": "BDF1 (ResidualBasedBDFDisplacementScheme); bdf2 ... bdf5 are accepted too"},
        "relaxation": {"description": "ResidualBasedRelaxationScheme"},
    },  # implicit scheme_type default is "bossak" (damp_factor_m -0.3); there is no generalized-alpha scheme; explicit: central_differences, multi_stage

    # ── ProjectParameters.json Template ──
    "project_parameters_template": {
        "problem_data": {
            "problem_name": "my_structure",
            "parallel_type": "OpenMP",
            "echo_level": 1,
            "start_time": 0.0,
            "end_time": 1.0,
        },
        "solver_settings": {
            "solver_type": "Static",
            "model_part_name": "Structure",
            "domain_size": 3,
            "echo_level": 1,
            "analysis_type": "non_linear",
            "model_import_settings": {
                "input_type": "mdpa",
                "input_filename": "my_structure",
            },
            "material_import_settings": {
                "materials_filename": "StructuralMaterials.json",
            },
            "time_stepping": {
                "time_step": 1.0,  # For static: pseudo-time step (load stepping)
            },
            "convergence_criterion": "residual_criterion",
            "displacement_relative_tolerance": 1e-6,
            "displacement_absolute_tolerance": 1e-9,
            "residual_relative_tolerance": 1e-6,
            "residual_absolute_tolerance": 1e-9,
            "max_iteration": 30,
            "rotation_dofs": False,  # True for shells/beams
            "volumetric_strain_dofs": False,  # True for mixed U-P formulations
            "linear_solver_settings": {
                "solver_type": "LinearSolversApplication.sparse_lu",  # or "amgcl"
            },
        },
        "processes": {
            "constraints_process_list": [
                # Dirichlet BCs (fixed displacement)
            ],
            "loads_process_list": [
                # Neumann BCs (forces, pressures)
            ],
            "list_other_processes": [],
        },
        "output_processes": {
            "vtk_output": [
                {
                    "python_module": "vtk_output_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "VtkOutputProcess",
                    "Parameters": {
                        "model_part_name": "Structure",
                        "output_control_type": "step",
                        "output_interval": 1,
                        "file_format": "ascii",  # or "binary"
                        "output_precision": 7,
                        "output_sub_model_parts": False,
                        "output_path": "vtk_output",
                        "nodal_solution_step_data_variables": ["DISPLACEMENT", "REACTION"],
                        "gauss_point_variables_in_elements": ["VON_MISES_STRESS"],
                    },
                },
            ],
        },
    },

    # ── Materials JSON Template ──
    "materials_template": {
        "properties": [
            {
                "model_part_name": "Structure.Parts_Body",
                "properties_id": 1,
                "Material": {
                    "constitutive_law": {
                        "name": "LinearElastic3DLaw",
                    },
                    "Variables": {
                        "DENSITY": 7850.0,
                        "YOUNG_MODULUS": 2.1e11,
                        "POISSON_RATIO": 0.3,
                    },
                    "Tables": {},
                },
            },
        ],
    },

    "pitfalls": [
        "Element naming: <Type><Dim>D<Nodes>N, e.g. SmallDisplacementElement3D4N",
        "For shells/beams, set rotation_dofs: true in solver_settings",
        "For large deformation, use TotalLagrangianElement or UpdatedLagrangianElement",
        "SmallDisplacementElement is for infinitesimal strain ONLY (< ~5% strain)",
        "Material is defined via StructuralMaterials.json, NOT in the .mdpa Properties block",
        "The constitutive_law name must match exactly (case-sensitive)",
        "For plane stress: use LinearElasticPlaneStress2DLaw, NOT LinearElastic3DLaw",
        "For plane strain: use LinearElasticPlaneStrain2DLaw",
        "For dynamic: set solver_type to Dynamic and configure time_integration_settings",
        "Convergence: and_criterion (both disp+residual) is most robust",
        "For nonlinear: use pseudo-time stepping (time_step < 1.0) to ramp loads",
        "VTK output: gauss_point_variables need elements that compute them",
        "DISPLACEMENT is the DOF, REACTION is the support reaction force",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. CONSTITUTIVE LAWS (from ConstitutiveLawsApplication + StructuralMechanics)
# ═══════════════════════════════════════════════════════════════════════════

CONSTITUTIVE_LAWS = {
    # ── Elasticity ──
    "linear_elastic": {
        "LinearElastic3DLaw": {
            "params": ["DENSITY", "YOUNG_MODULUS", "POISSON_RATIO"],
            "description": "3D isotropic linear elasticity",
        },
        "LinearElasticPlaneStrain2DLaw": {
            "params": ["DENSITY", "YOUNG_MODULUS", "POISSON_RATIO"],
            "description": "2D plane strain",
        },
        "LinearElasticPlaneStress2DLaw": {
            "params": ["DENSITY", "YOUNG_MODULUS", "POISSON_RATIO"],
            "description": "2D plane stress",
        },
        "LinearElasticAxisym2DLaw": {
            "params": ["DENSITY", "YOUNG_MODULUS", "POISSON_RATIO"],
            "description": "Axisymmetric linear elastic",
        },
        "TrussConstitutiveLaw": {
            "params": ["DENSITY", "YOUNG_MODULUS"],
            "description": "1D truss (bar) constitutive law",
        },
        "BeamConstitutiveLaw": {
            "params": ["DENSITY", "YOUNG_MODULUS", "POISSON_RATIO"],
            "description": "Beam constitutive law",
        },
    },

    # ── Hyperelasticity ──
    "hyperelastic": {
        "KirchhoffSaintVenant3DLaw": {
            "params": ["DENSITY", "YOUNG_MODULUS", "POISSON_RATIO"],
            "description": "Saint Venant-Kirchhoff (small strain, large rotation)",
        },
        "HyperElastic3DLaw": {
            "params": ["DENSITY", "YOUNG_MODULUS", "POISSON_RATIO"],
            "description": ("Neo-Hookean hyperelastic (large deformation); C++ class HyperElasticIsotropicNeoHookean3D, "
                            "ConstitutiveLawsApplication. The Simo-Taylor formulation is HyperElasticSimoTaylorNeoHookean3DLaw"),
        },
        "HyperElasticPlaneStrain2DLaw": {
            "params": ["DENSITY", "YOUNG_MODULUS", "POISSON_RATIO"],
            "description": "2D plane strain Neo-Hookean (C++ class HyperElasticIsotropicNeoHookeanPlaneStrain2D)",
        },
    },

    # ── Plasticity (Small Strain) ──
    # Factory pattern: SmallStrainIsotropicPlasticityFactory (optional "law_type": "3D" (default) or "PlaneStrain")
    # Combines yield_surface + plastic_potential
    "plasticity_small_strain": {
        "factory_name": "SmallStrainIsotropicPlasticityFactory",
        "yield_surfaces": [
            "VonMises",
            "Tresca",
            "ModifiedMohrCoulomb",
            "DruckerPrager",
            "MohrCoulomb",
        ],  # Rankine and SimoJu are rejected by the plasticity factory (damage laws only)
        "plastic_potentials": [
            "VonMises",
            "Tresca",
            "ModifiedMohrCoulomb",
            "DruckerPrager",
            "MohrCoulomb",
        ],
        "params": [
            "DENSITY", "YOUNG_MODULUS", "POISSON_RATIO",
            "YIELD_STRESS_TENSION", "YIELD_STRESS_COMPRESSION",
            "FRACTURE_ENERGY", "HARDENING_CURVE",
            "FRICTION_ANGLE", "DILATANCY_ANGLE",
            "MAXIMUM_STRESS", "MAXIMUM_STRESS_POSITION",
        ],
        "hardening_curves": {
            0: "Linear Softening",
            1: "Exponential Softening",
            2: "Initial Hardening + Exponential Softening (parabolic)",
            3: "Perfect Plasticity",
            4: "Curve Fitting Hardening (VonMises/Tresca only)",
            5: "Linear + Exponential Softening",
            6: "Curve Defined By Points",
        },
        "materials_json_example": {
            "constitutive_law": {
                "name": "SmallStrainIsotropicPlasticityFactory",
                "yield_surface": "VonMises",
                "plastic_potential": "VonMises",
            },
            "Variables": {
                "DENSITY": 7850.0,
                "YOUNG_MODULUS": 2.069e11,
                "POISSON_RATIO": 0.29,
                "YIELD_STRESS_TENSION": 275e6,
                "YIELD_STRESS_COMPRESSION": 275e6,
                "FRACTURE_ENERGY": 1e5,
                "HARDENING_CURVE": 1,
            },
        },
    },

    # ── Plasticity (Finite Strain) ──
    "plasticity_finite_strain": {
        "description": "Multiplicative decomposition F = Fe * Fp, exponential map integrator",
        "factory_name": None,  # no finite-strain factory is registered
        "note": ("Registered directly as FiniteStrainIsotropicPlasticity3D<YieldSurface><PlasticPotential> "
                 "(e.g. FiniteStrainIsotropicPlasticity3DVonMisesVonMises); SmallStrainIsotropicPlasticityFactory "
                 "builds only SmallStrainIsotropicPlasticity* laws. Uses TotalLagrangianElement or UpdatedLagrangianElement"),
    },

    # ── Damage ──
    "damage_isotropic": {
        "factory_name": "SmallStrainIsotropicDamageFactory",
        # the factory appends plastic_potential to the name, so pass "plastic_potential": ""
        "specific_name_pattern": "SmallStrainIsotropicDamage3D<YieldSurface>",  # e.g. SmallStrainIsotropicDamage3DVonMises
        "params": [
            "DENSITY", "YOUNG_MODULUS", "POISSON_RATIO",
            "YIELD_STRESS_TENSION", "YIELD_STRESS_COMPRESSION",
            "FRACTURE_ENERGY", "SOFTENING_TYPE",
            "FRICTION_ANGLE",
        ],
        "softening_types": {
            0: "Linear Softening",
            1: "Exponential Softening",
            2: "Hardening Damage (parabolic)",
        },
    },
    "damage_dplus_dminus": {
        "name_pattern": "SmallStrainDplusDminusDamage<TensionYS><CompressionYS>3D",
        "description": "Separate tension/compression damage via spectral decomposition",
        "example_name": "SmallStrainDplusDminusDamageModifiedMohrCoulombVonMises3D",
        "extra_params": ["FRACTURE_ENERGY_COMPRESSION"],
    },

    # ── Viscoelasticity ──
    "viscoelastic": {
        "ViscousGeneralizedMaxwell3D": {
            "params": ["DENSITY", "YOUNG_MODULUS", "POISSON_RATIO", "VISCOUS_PARAMETER", "DELAY_TIME"],
            "description": "Generalized Maxwell (stress relaxation)",
        },
        "ViscousGeneralizedKelvin3D": {
            "params": ["DENSITY", "YOUNG_MODULUS", "POISSON_RATIO", "DELAY_TIME"],
            "description": "Generalized Kelvin (creep)",
        },
    },

    # ── Composites ──
    "composites": {
        "ParallelRuleOfMixturesLaw3D": {
            "params": ["DENSITY", "LAYER_EULER_ANGLES", "combination_factors"],
            "description": "Iso-strain (Voigt) composite. Sub-properties define constituents.",
        },
        "SerialParallelRuleOfMixturesLaw3D": {  # 2D variant: SerialParallelRuleOfMixturesLaw2D
            "params": ["combination_factors", "parallel_behaviour_directions", "LAYER_EULER_ANGLES"],
            "description": "Serial-parallel composite for fiber-matrix materials.",
        },
    },

    # ── High Cycle Fatigue ──
    "fatigue": {
        "name_pattern": "SmallStrainHighCycleFatigue3DLaw<YieldSurface>",
        "params": ["HIGH_CYCLE_FATIGUE_COEFFICIENTS"],
        "description": "HCF with Wohler-curve-based damage accumulation. Coefficients: [Su, STHR1, STHR2, ALFAF, BETAF, AUXR1, AUXR2].",
    },

    # ── Tangent Operator Estimation ──
    "tangent_operator_types": {
        0: "Analytic",
        1: "FirstOrderPerturbation",
        2: "SecondOrderPerturbation (default)",
        3: "Secant",
        4: "SecondOrderPerturbationV2",
    },

    # ── Stress Measures ──
    "stress_measures": [
        "StressMeasure_PK1",      # First Piola-Kirchhoff (ref config, non-symmetric)
        "StressMeasure_PK2",      # Second Piola-Kirchhoff (ref config)
        "StressMeasure_Kirchhoff", # Kirchhoff (current config)
        "StressMeasure_Cauchy",    # Cauchy / true stress (current config)
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# 3. FLUID DYNAMICS APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

FLUID_DYNAMICS = {
    "application": "FluidDynamicsApplication",
    "analysis_class": "fluid_dynamics_analysis.FluidDynamicsAnalysis",
    "import": "from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis",

    # ── Solver Types ──
    "solver_types": {
        "monolithic": {
            "solver_type": "Monolithic",
            "description": "Coupled velocity-pressure solve. More robust, larger system.",
            "python_module": "navier_stokes_monolithic_solver",
        },
        "fractional_step": {
            "solver_type": "FractionalStep",
            "description": "Split velocity and pressure steps. Faster per step, needs smaller dt.",
            "python_module": "navier_stokes_solver_fractionalstep",
        },
    },

    # ── Element Types ──
    "elements": {
        # VMS (Variational MultiScale) - monolithic, simplex only (use QSVMS2D4N/QSVMS3D8N for quads/hexes)
        "VMS2D3N": "VMS stabilized triangle (2D, 3 nodes), monolithic",
        "VMS3D4N": "VMS stabilized tetrahedron, monolithic",

        # QSVMS (Quasi-Static VMS) - preferred for monolithic
        "QSVMS2D3N": "Quasi-static VMS triangle, dynamic subscales",
        "QSVMS2D4N": "Quasi-static VMS quad",
        "QSVMS3D4N": "Quasi-static VMS tetrahedron",
        "QSVMS3D8N": "Quasi-static VMS hexahedron",

        # Fractional step elements (simplex only)
        "FractionalStep2D3N": "Fractional step triangle",
        "FractionalStep3D4N": "Fractional step tetrahedron",

        # Two-fluid (free surface)
        "TwoFluidNavierStokes2D3N": "Two-fluid (level-set) triangle",
        "TwoFluidNavierStokes3D4N": "Two-fluid (level-set) tetrahedron",

        # Weakly compressible
        "WeaklyCompressibleNavierStokes2D3N": "Weakly compressible NS (low Mach)",
        "WeaklyCompressibleNavierStokes3D4N": "Weakly compressible NS (low Mach)",
    },

    # ── Conditions ──
    "conditions": {
        "MonolithicWallCondition2D2N": "Wall (no-slip) for monolithic, 2D",
        "MonolithicWallCondition3D3N": "Wall (no-slip) for monolithic, 3D triangle",
        "NavierStokesWallCondition2D2N": "Wall with wall law support",
        "NavierStokesWallCondition3D3N": "Wall with wall law support (3D)",
    },

    # ── Stabilization Settings ──
    "stabilization": {
        "formulation": {
            "element_type": "vms",  # or "qsvms", "dvms"; "fic" takes no use_orthogonal_subscales
            "dynamic_tau": 1.0,     # Stabilization parameter (0.0 = no dynamic tau); the vms default is 0.01
            "use_orthogonal_subscales": False,  # False = ASGS, True = OSS; sets ProcessInfo OSS_SWITCH
        },  # "oss_switch" (int) is only a solver_settings key of the fractional_step solver
        "methods": {
            "ASGS": "Algebraic Sub-Grid Scales (use_orthogonal_subscales=false, default)",
            "OSS": "Orthogonal Sub-Scales (use_orthogonal_subscales=true, more accurate but more expensive)",
            "VMS": "Variational MultiScale (classical)",
            "QSVMS": "Quasi-Static VMS with dynamic subscales (recommended)",
            "FIC": "Finite Increment Calculus",
        },
    },

    # ── Turbulence (via RANSApplication) ──
    "turbulence_models": {
        "k_epsilon": "Standard k-epsilon (RANSApplication required)",
        "k_omega_sst": "k-omega SST (RANSApplication required)",
        "note": "Turbulence models require RANSApplication, not part of base FluidDynamics",
    },

    # ── Wall Laws ──
    "wall_laws": {
        "no_slip": "Zero velocity at wall (resolved boundary layer)",
        "navier_slip": "Partial slip with slip coefficient",
        "wall_law": "Logarithmic wall law for under-resolved boundary layers",
    },

    # ── ProjectParameters.json Template ──
    "project_parameters_template": {
        "problem_data": {
            "problem_name": "my_fluid",
            "parallel_type": "OpenMP",
            "echo_level": 0,
            "start_time": 0.0,
            "end_time": 10.0,
        },
        "solver_settings": {
            "model_part_name": "FluidModelPart",
            "domain_size": 2,
            "solver_type": "Monolithic",
            "model_import_settings": {
                "input_type": "mdpa",
                "input_filename": "my_fluid",
            },
            "material_import_settings": {
                "materials_filename": "FluidMaterials.json",
            },
            "formulation": {
                "element_type": "vms",
                "dynamic_tau": 1.0,
            },
            "maximum_iterations": 10,
            "echo_level": 1,
            "relative_velocity_tolerance": 1e-5,
            "absolute_velocity_tolerance": 1e-7,
            "relative_pressure_tolerance": 1e-5,
            "absolute_pressure_tolerance": 1e-7,
            "volume_model_part_name": "FluidParts_Fluid",
            "skin_parts": ["AutomaticInlet2D_Inlet", "Outlet2D_Outlet", "NoSlip2D_Wall"],
            "no_skin_parts": [],
            "time_stepping": {
                "automatic_time_step": False,
                "time_step": 0.01,
            },
            "linear_solver_settings": {
                "solver_type": "amgcl",
                "max_iteration": 200,
                "tolerance": 1e-7,
                "scaling": False,
            },
        },
        "processes": {
            "initial_conditions_process_list": [],
            "boundary_conditions_process_list": [
                # Inlet velocity, outlet pressure, no-slip walls
            ],
            "gravity": [
                {
                    "python_module": "assign_vector_by_direction_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "AssignVectorByDirectionProcess",
                    "Parameters": {
                        "model_part_name": "FluidModelPart.FluidParts_Fluid",
                        "variable_name": "BODY_FORCE",
                        "modulus": 9.81,
                        "constrained": False,
                        "direction": [0.0, -1.0, 0.0],
                    },
                },
            ],
        },
        "output_processes": {
            "vtk_output": [],
        },
    },

    # ── Fluid Materials JSON ──
    "materials_template": {
        "properties": [
            {
                "model_part_name": "FluidModelPart.FluidParts_Fluid",
                "properties_id": 1,
                "Material": {
                    "constitutive_law": {
                        "name": "Newtonian2DLaw",  # or "Newtonian3DLaw"
                    },
                    "Variables": {
                        "DENSITY": 1000.0,
                        "DYNAMIC_VISCOSITY": 1e-3,
                    },
                    "Tables": {},
                },
            },
        ],
    },

    "pitfalls": [
        "Monolithic solver uses VMS/QSVMS elements; FractionalStep uses FractionalStep elements",
        "The solver auto-selects elements based on formulation.element_type and dimension",
        "For high Re: reduce time_step, increase mesh resolution near walls",
        "PRESSURE is the primary pressure variable; VELOCITY is the velocity vector",
        "Outlet: apply zero PRESSURE or use 'outlet' condition",
        "DYNAMIC_VISCOSITY (not kinematic!) is the material parameter",
        "For free surface: use TwoFluidNavierStokes elements with DISTANCE level-set",
        "skin_parts list must match SubModelPart names in .mdpa exactly",
        "BODY_FORCE (not GRAVITY) is the variable for gravity in fluid",
        "Reynolds number = DENSITY * U * L / DYNAMIC_VISCOSITY",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONVECTION-DIFFUSION APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

CONVECTION_DIFFUSION = {
    "application": "ConvectionDiffusionApplication",
    "analysis_class": "convection_diffusion_analysis.ConvectionDiffusionAnalysis",
    "import": "from KratosMultiphysics.ConvectionDiffusionApplication.convection_diffusion_analysis import ConvectionDiffusionAnalysis",

    # ── Variable Mapping (ConvectionDiffusionSettings) ──
    # The solver maps generic PDE variables to physical variables:
    "convection_diffusion_variables": {
        "unknown_variable": "TEMPERATURE",          # Primary unknown (DOF)
        "diffusion_variable": "CONDUCTIVITY",       # Diffusion coefficient
        "volume_source_variable": "HEAT_FLUX",      # Volumetric source term (f)
        "surface_source_variable": "FACE_HEAT_FLUX", # Neumann BC flux
        "density_variable": "DENSITY",
        "specific_heat_variable": "SPECIFIC_HEAT",
        "convection_variable": "CONVECTION_VELOCITY", # Advection velocity
        "velocity_variable": "VELOCITY",
        "mesh_velocity_variable": "MESH_VELOCITY",
        "projection_variable": "PROJECTED_SCALAR1",
        "reaction_variable": "REACTION_FLUX",        # Output reaction
    },

    # ── Solver Types ──
    "solver_types": {
        "Stationary": "Steady-state (no time derivatives)",
        "Transient": "Time-dependent with BDF or theta scheme",
    },

    # ── Element Types ──
    "elements": {
        # Eulerian convection-diffusion (auto-selected by solver)
        "EulerianConvDiff2D3N": "Eulerian conv-diff triangle (2D, 3 nodes) — DEFAULT",
        "EulerianConvDiff2D4N": "Eulerian conv-diff quad (2D, 4 nodes)",
        "EulerianConvDiff3D4N": "Eulerian conv-diff tetrahedron",
        "EulerianConvDiff3D8N": "Eulerian conv-diff hexahedron",

        # Laplacian elements (diffusion + volumetric source)
        "LaplacianElement2D3N": "Diffusion (Laplacian) triangle — assembles HEAT_FLUX volumetric source",
        "LaplacianElement3D4N": "Diffusion (Laplacian) tetrahedron — assembles HEAT_FLUX volumetric source",

        # NOTE: LaplacianElement DOES assemble HEAT_FLUX (verified in laplacian_element.cpp).
        # Use EulerianConvDiff if you also need convection or time-dependent dynamics.

        # Generic element names used for ReplaceElementsAndConditionsProcess
        "Element2D3N": "Generic 2D triangle (replaced at runtime)",
        "Element2D4N": "Generic 2D quad (replaced at runtime)",
        "Element3D4N": "Generic 3D tetrahedron (replaced at runtime)",
    },

    # ── Conditions ──
    "conditions": {
        "LineCondition2D2N": "2D boundary line (for FACE_HEAT_FLUX)",
        "SurfaceCondition3D3N": "3D boundary triangle surface",
        "SurfaceCondition3D4N": "3D boundary quad surface",
        "ThermalFace2D2N": "Thermal face condition (alternative)",
        "ThermalFace3D3N": "Thermal face condition 3D",
    },

    # ── Time Scheme Parameter ──
    "theta_scheme": {
        0.0: "Forward Euler (explicit, conditionally stable)",
        0.5: "Crank-Nicolson (second-order, oscillation-prone)",
        1.0: "Backward Euler (implicit, first-order, most stable)",
        "default": 0.5,
    },

    # ── ProjectParameters.json Template ──
    "project_parameters_template": {
        "problem_data": {
            "problem_name": "my_thermal",
            "parallel_type": "OpenMP",
            "echo_level": 0,
            "start_time": 0.0,
            "end_time": 1.0,
        },
        "solver_settings": {
            "model_part_name": "ThermalModelPart",
            "domain_size": 2,
            "solver_type": "Transient",  # or "Stationary"
            "analysis_type": "linear",
            "model_import_settings": {
                "input_type": "mdpa",
                "input_filename": "my_thermal",
            },
            "material_import_settings": {
                "materials_filename": "ThermalMaterials.json",
            },
            "convection_diffusion_variables": {
                "density_variable": "DENSITY",
                "diffusion_variable": "CONDUCTIVITY",
                "unknown_variable": "TEMPERATURE",
                "volume_source_variable": "HEAT_FLUX",
                "surface_source_variable": "FACE_HEAT_FLUX",
                "projection_variable": "PROJECTED_SCALAR1",
                "convection_variable": "CONVECTION_VELOCITY",
                "mesh_velocity_variable": "MESH_VELOCITY",
                "velocity_variable": "VELOCITY",
                "specific_heat_variable": "SPECIFIC_HEAT",
                "reaction_variable": "REACTION_FLUX",
            },
            "time_stepping": {
                "time_step": 0.1,
            },
            "linear_solver_settings": {
                "solver_type": "LinearSolversApplication.sparse_lu",
            },
        },
        "processes": {
            "constraints_process_list": [],
            "loads_process_list": [],
        },
        "output_processes": {
            "vtk_output": [],
        },
    },

    # ── Thermal Materials JSON ──
    "materials_template": {
        "properties": [
            {
                "model_part_name": "ThermalModelPart.Parts_Solid",
                "properties_id": 1,
                "Material": {
                    # no "constitutive_law" entry: ConvDiff uses none, and ReadMaterialsUtility
                    # errors on an unregistered law name
                    "Variables": {
                        "DENSITY": 1.0,
                        "CONDUCTIVITY": 1.0,
                        "SPECIFIC_HEAT": 1.0,
                    },
                    "Tables": {},
                },
            },
        ],
    },

    "pitfalls": [
        "LaplacianElement DOES assemble HEAT_FLUX volumetric source terms (verified against laplacian_element.cpp).",
        "Use EulerianConvDiff elements only if you also need convection or transient dynamics; LaplacianElement is sufficient for stationary Poisson with source.",
        "Properties must be assigned NODALLY (not on element) via assign_properties process",
        "The solver uses ReplaceElementsAndConditionsProcess to swap generic elements at runtime",
        "For pure Poisson: still use ConvectionDiffusionApplication with zero convection velocity",
        "For stationary: set solver_type to 'Stationary', not 'Transient' with 1 step",
        "TEMPERATURE is the DOF variable, not SCALAR or PHI",
        "HEAT_FLUX is volumetric source (W/m^3), FACE_HEAT_FLUX is surface flux (W/m^2)",
        "convection_diffusion_variables block MUST be present in solver_settings",
        "For convection-dominated: reduce time step or use stabilized elements (SUPG built-in)",
        "TetrahedralMeshOrientationCheck is run automatically; malformed elements will cause errors",
        "Material properties are applied to nodes, not elements (different from StructuralMechanics)",
    ],

    # ── Source Term Setup (the gotcha we encountered) ──
    "source_term_setup": {
        "correct_approach": {
            "description": "Use EulerianConvDiff elements + HEAT_FLUX nodal variable",
            "steps": [
                "1. Define EulerianConvDiff elements in .mdpa (or use generic Element2D3N)",
                "2. Set volume_source_variable: HEAT_FLUX in convection_diffusion_variables",
                "3. Apply source via process: assign_scalar_variable_process on HEAT_FLUX",
                "4. Make sure material has CONDUCTIVITY, DENSITY, SPECIFIC_HEAT defined",
            ],
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 5. CO-SIMULATION APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

COSIMULATION = {
    "application": "CoSimulationApplication",
    "analysis_class": "co_simulation_analysis.CoSimulationAnalysis",
    "import": "from KratosMultiphysics.CoSimulationApplication.co_simulation_analysis import CoSimulationAnalysis",

    # ── Coupling Schemes ──
    "coupling_schemes": {
        "gauss_seidel_weak": "Weak (explicit) Gauss-Seidel: one pass per time step",
        "gauss_seidel_strong": "Strong (implicit) Gauss-Seidel: iterate to convergence",
        "jacobi_weak": "Weak Jacobi: parallel solvers, one pass",
        "jacobi_strong": "Strong Jacobi: parallel solvers with convergence iterations",
    },

    # ── Convergence Accelerators ──
    "convergence_accelerators": {
        "constant_relaxation": {
            "description": "Fixed relaxation factor (simplest, slowest)",
            "params": {"alpha": 0.125},
        },
        "aitken": {
            "description": "Aitken adaptive relaxation (good default for FSI)",
            "params": {"init_alpha": 0.1, "init_alpha_max": 0.45},
        },
        "mvqn": {
            "description": "Multi-Vector Quasi-Newton (fast convergence, needs memory)",
            "params": {},
        },
        "block_ibqnls": {
            "description": "Interface Block Quasi-Newton Least Squares (block variant)",
            "params": {},
        },
        "block_mvqn": {
            "description": "Block Multi-Vector Quasi-Newton (block variant)",
            "params": {},
        },
        "anderson": {
            "description": "Anderson acceleration (related to DIIS)",
            "params": {},
        },
        "iqnils": {
            "description": "Interface Quasi-Newton with Inverse Least Squares",
            "params": {},
        },
    },

    # ── Convergence Criteria ──
    "convergence_criteria": [
        "relative_norm_initial_residual",
        "relative_norm_previous_residual",
        "absolute_norm_energy_conjugate",
    ],  # the last is energy-conjugate based (needs criteria_composition "energy_conjugate"); the relative_norm criteria also take abs_tolerance

    # ── Predictors ──
    "predictors": [
        "average_value_based",        # Use average of previous values
        "linear",                     # Linear extrapolation
        "linear_derivative_based",    # Based on derivative history
    ],

    # ── Data Transfer Operators ──
    "data_transfer_operators": [
        "copy",                       # Direct copy (matching meshes)
        "kratos_mapping",             # Using MappingApplication
        "sum_distributed_to_single",  # Collect distributed data
        "copy_single_to_distributed", # Distribute single data (deprecated alias of transfer_one_to_many)
    ],

    # ── Solver Wrapper Types ──
    "solver_wrappers": {
        "kratos": "Internal Kratos solver (any application)",
        "external": "External solver via CoSimIO",
        "sdof": "Single-degree-of-freedom model",
    },

    # ── ProjectParameters.json Template (Coupled Simulation) ──
    "project_parameters_template": {
        "problem_data": {
            "problem_name": "coupled_simulation",
            "start_time": 0.0,
            "end_time": 10.0,
            "echo_level": 0,
            "parallel_type": "OpenMP",
        },
        "solver_settings": {
            "type": "coupled_solvers.gauss_seidel_strong",
            "echo_level": 1,
            # no "coupling_scheme" or "max_iteration" key: the scheme is the "type" above, and the
            # iteration limit, accelerators and criteria sit at solver_settings level
            "num_coupling_iterations": 15,
            "convergence_accelerators": [
                {
                    "type": "aitken",
                    "solver": "fluid",
                    "data_name": "displacement",
                    "init_alpha": 0.25,
                },
            ],
            "convergence_criteria": [
                {
                    "type": "relative_norm_initial_residual",
                    "solver": "fluid",
                    "data_name": "displacement",
                    "abs_tolerance": 1e-7,
                    "rel_tolerance": 1e-5,
                },
            ],
            # required: the order the solvers run in and the data each one receives and sends
            "coupling_sequence": [
                {
                    "name": "fluid",
                    "input_data_list": [],
                    "output_data_list": [],
                },
                {
                    "name": "structure",
                    "input_data_list": [
                        {
                            "data": "force",
                            "from_solver": "fluid",
                            "from_solver_data": "force",
                            "data_transfer_operator": "mapper",
                            "data_transfer_operator_options": ["swap_sign"],
                        },
                    ],
                    "output_data_list": [
                        {
                            "data": "displacement",
                            "to_solver": "fluid",
                            "to_solver_data": "displacement",
                            "data_transfer_operator": "mapper",
                        },
                    ],
                },
            ],
            "solvers": {
                "fluid": {
                    "type": "solver_wrappers.kratos.fluid_dynamics_wrapper",
                    "solver_wrapper_settings": {
                        "input_file": "ProjectParametersFluid",
                    },
                    "data": {
                        "displacement": {
                            "model_part_name": "FluidModelPart.Interface",
                            "variable_name": "MESH_DISPLACEMENT",
                            "dimension": 2,
                        },
                        "force": {
                            "model_part_name": "FluidModelPart.Interface",
                            "variable_name": "REACTION",
                            "dimension": 2,
                        },
                    },
                },
                "structure": {
                    "type": "solver_wrappers.kratos.structural_mechanics_wrapper",
                    "solver_wrapper_settings": {
                        "input_file": "ProjectParametersStructure",
                    },
                    "data": {
                        "displacement": {
                            "model_part_name": "Structure.Interface",
                            "variable_name": "DISPLACEMENT",
                            "dimension": 2,
                        },
                        "force": {
                            "model_part_name": "Structure.Interface",
                            "variable_name": "POINT_LOAD",
                            "dimension": 2,
                        },
                    },
                },
            },
            "coupling_operations": {},
            "data_transfer_operators": {
                "mapper": {
                    "type": "kratos_mapping",
                    "mapper_settings": {
                        "mapper_type": "nearest_neighbor",
                    },
                },
            },
        },
    },

    "pitfalls": [
        "Each solver needs its OWN ProjectParameters.json (referenced by input_file)",
        "Data names used in coupling_sequence, convergence_accelerators and convergence_criteria must match the solvers' data blocks",
        "Interface SubModelParts must exist in BOTH solver meshes",
        "For FSI: fluid sends REACTION (force), structure sends DISPLACEMENT",
        "convergence_accelerator: aitken is safest default; mvqn is fastest but can diverge",
        "For weak coupling: no convergence_accelerator needed (single pass)",
        "CoSimIO is needed for coupling with EXTERNAL (non-Kratos) solvers",
        "Data transfer can fail silently if interface model parts don't match",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# 6. FSI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

FSI = {
    "application": "FSIApplication",
    "description": "Fluid-Structure Interaction with partitioned coupling",
    "dependencies": [
        "FluidDynamicsApplication",
        "StructuralMechanicsApplication",
        "MeshMovingApplication",
        "MappingApplication",
    ],

    "coupling_types": {
        "partitioned_dirichlet_neumann": {
            "description": "Most common: structure displacement -> fluid mesh, fluid force -> structure",
            "solver_type": "Partitioned",
            "coupling_scheme": "DirichletNeumann",
        },
    },

    # ── FSI ProjectParameters.json Template ──
    "project_parameters_template": {
        "problem_data": {
            "problem_name": "fsi_simulation",
            "parallel_type": "OpenMP",
            "start_time": 0.0,
            "end_time": 10.0,
        },
        "solver_settings": {
            "solver_type": "Partitioned",
            "coupling_scheme": "DirichletNeumann",
            "echo_level": 1,
            "structure_solver_settings": {
                "solver_type": "Dynamic",
                "_comment": "Full StructuralMechanics ProjectParameters here",
            },
            "fluid_solver_settings": {
                "solver_type": "Monolithic",
                "_comment": "Full FluidDynamics ProjectParameters here",
            },
            "mesh_solver_settings": {
                "solver_type": "structural_similarity",
                "echo_level": 0,
                "domain_size": 2,
                "model_part_name": "FluidModelPart",
                "reform_dofs_each_step": False,
            },
            "coupling_settings": {
                "nl_tol": 1e-6,
                "nl_max_it": 15,
                "solve_mesh_at_each_iteration": True,
                "mapper_settings": [
                    {
                        "mapper_face": "unique",
                        "fluid_interface_submodelpart_name": "FluidNoSlipInterface2D_InterfaceFluid",
                        "structure_interface_submodelpart_name": "StructureInterface2D_InterfaceStructure",
                    },
                ],
                "coupling_strategy_settings": {
                    "solver_type": "Relaxation",
                    "acceleration_type": "Aitken",
                    "w_0": 0.825,
                },
            },
        },
    },

    "pitfalls": [
        "Interface model parts must exist in BOTH fluid and structure meshes",
        "Mesh solver (ALE) deforms the fluid mesh to match structural displacement",
        "Start with loose coupling (weak) before attempting strong coupling",
        "Aitken relaxation is the safest starting choice for FSI convergence",
        "For 3D FSI: ensure normals are consistently oriented on interface",
        "ALE mesh motion can degrade mesh quality; use remeshing for large deformations",
        "Time step must be small enough for BOTH fluid CFL and structural dynamics",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# 7. CONTACT STRUCTURAL MECHANICS APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

CONTACT_STRUCTURAL_MECHANICS = {
    "application": "ContactStructuralMechanicsApplication",
    "description": "Contact mechanics with mortar-based formulations",
    "depends_on": "StructuralMechanicsApplication",

    "contact_formulations": {
        "ALM": {
            "description": "Augmented Lagrangian Method (most accurate, expensive)",
            "solver_type": "contact_structural_mechanics_static_solver",
        },
        "penalty": {
            "description": "Penalty method (simpler, needs tuning of penalty parameter)",
            "parameter": ("INITIAL_PENALTY (core variable in ProcessInfo): computed automatically as "
                          "stiffness_factor * E_mean / h_mean (times 1e4 in the penalty process) unless "
                          "advance_ALM_parameters.manual_ALM is true, then taken from advance_ALM_parameters.penalty. "
                          "PENALTY_PARAMETER belongs to the legacy ContactMechanicsApplication"),
        },
        "mortar_NTN": {
            "description": "Node-to-Node mortar (simplified)",
        },
        "mortar_NTS": {
            "description": "Node-to-Segment mortar (standard)",
        },
    },

    "contact_types": {
        "frictionless": "Normal contact only (no tangential)",
        "frictional": "Coulomb friction model",
    },

    "search_algorithms": {
        "octree": "OcTree-based contact search",
        "bins": "Bins-based spatial search (default)",
    },

    "conditions": {
        "ALMFrictionlessMortarContactCondition2D2N": "ALM frictionless 2D contact",
        "ALMFrictionlessMortarContactCondition3D3N": "ALM frictionless 3D (triangle)",
        "ALMFrictionlessMortarContactCondition3D4N": "ALM frictionless 3D (quad)",
        "ALMFrictionalMortarContactCondition3D3N": "ALM frictional 3D (triangle)",
        "PenaltyFrictionlessMortarContactCondition3D3N": "Penalty frictionless 3D",
        "PenaltyFrictionalMortarContactCondition3D3N": "Penalty frictional 3D",
        "MeshTyingMortarCondition2D2N": "Mesh tying (glued contact) 2D",
        "MeshTyingMortarCondition3D3N": "Mesh tying (glued contact) 3D",
    },

    "pitfalls": [
        "Contact surfaces need proper master/slave designation",
        "Penalty parameter: too small = penetration, too large = ill-conditioning",
        "ALM is more robust but adds Lagrange multiplier DOFs",
        "Contact search can be expensive; tune search parameters",
        "For self-contact: both surfaces are master and slave",
        "Frictional contact requires FRICTION_COEFFICIENT in material",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# 8. GEO-MECHANICS APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

GEOMECHANICS = {
    "application": "GeoMechanicsApplication",
    "description": "Soil mechanics, consolidation, groundwater flow",

    "formulations": {
        "U": "Displacement only (drained analysis)",
        "Pw": "Water pressure only (seepage analysis)",
        "U-Pw": "Coupled displacement + water pressure (Biot consolidation)",
    },

    "constitutive_laws": {
        "GeoLinearElasticPlaneStrain2DLaw": "Linear elastic, plane strain",
        "GeoIncrementalLinearElastic3DLaw": "Incremental linear elastic, 3D",
        "GeoMohrCoulombWithTensionCutOff2D": "Mohr-Coulomb with tension cut-off (2D)",
        "GeoMohrCoulombWithTensionCutOff3D": "Mohr-Coulomb with tension cut-off (3D)",
        "SmallStrainUDSM2DPlaneStrainLaw": "User-defined soil model from an external library (also SmallStrainUDSM3DLaw)",
        "SmallStrainUMAT2DPlaneStrainLaw": "UMAT from an external library (also SmallStrainUMAT3DLaw)",
    },  # GeoMechanicsApplication registers no Drucker-Prager or Cam-Clay law

    "element_types": {
        "UPwSmallStrainElement2D3N": "Coupled U-Pw triangle (small strain)",
        "UPwSmallStrainElement2D4N": "Coupled U-Pw quad (small strain)",
        "UPwSmallStrainElement3D4N": "Coupled U-Pw tetrahedron",
        "UPwSmallStrainElement3D8N": "Coupled U-Pw hexahedron",
        "SmallStrainUPwDiffOrderElement2D6N": "Different order (quad U, lin Pw) triangle",
        "SmallStrainUPwDiffOrderElement3D10N": "Different order tetrahedron",
        "SteadyStatePwElement2D3N": "Steady seepage triangle",
        "TransientPwElement2D3N": "Transient seepage triangle",
    },

    "pitfalls": [
        "WATER_PRESSURE is the pore water pressure DOF",
        "Effective stress = total stress - pore pressure (Terzaghi principle)",
        "For drained: use displacement-only elements (not U-Pw)",
        "For undrained: use U-Pw with very low permeability",
        "Gravity loading is critical: VOLUME_ACCELERATION = [0, -9.81, 0]",
        "Initial stress state (K0 procedure) often needed before loading",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# 9. DEM APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

DEM = {
    "application": "DEMApplication",
    "description": "Discrete Element Method for granular and particle mechanics",

    # Registered element names, confirmed by ModelPart.CreateNewElement against an
    # installed DEMApplication 10.4.3 (2026-08-07). The bare stems previously listed
    # here ("SphericParticle", "Cluster") are NOT registered names.
    "particle_types": {
        "SphericParticle3D": "Standard spherical particle (3D)",
        "SphericContinuumParticle3D": "Bonded spheres (cohesive materials, concrete, rock)",
        "CylinderParticle2D": "Genuine 2D particle (disk). Kratos DEM is not 3D-only",
        "CylinderContinuumParticle2D": "Bonded 2D disk",
        "Cluster3D": "Non-spherical particle (composed of multiple spheres)",
    },

    "contact_laws": {
        "Hertz": "Hertzian normal contact (nonlinear elastic)",
        "LinearElastic": "Linear spring-dashpot model",
        "CoulombFriction": "Tangential Coulomb friction",
        "BondedDEM": "Bonded contacts (for concrete/rock fracture simulation)",
    },

    "features": [
        "Particle inlets (inject particles with specified granulometry and mass flow)",
        "FEM walls (rigid wall boundaries from FEM mesh)",
        "Cohesive materials (bonded contacts for concrete, rock)",
        "Breakable clusters (fragmentation simulation)",
        "Parallel search (bins-based spatial search)",
        "DEM-FEM coupling (forces on FEM walls from particles)",
    ],

    "main_variables": {
        "RADIUS": "Particle radius. Core Kratos variable, per NODE, the one mandatory nodal input",
        "PARTICLE_DENSITY": "Density, per PROPERTIES. NOT 'DENSITY' — that spelling is accepted and never read",
        "PARTICLE_MATERIAL": "Material ID for contact law selection",
        "STATIC_FRICTION": "Friction, per material_relations pair. 'PARTICLE_FRICTION' is not a Kratos variable",
        "DYNAMIC_FRICTION": "Friction, per material_relations pair",
        "VELOCITY": "Particle velocity",
        "ANGULAR_VELOCITY": "Particle angular velocity",
        "TOTAL_FORCES": "Total force on particle",
    },

    "pitfalls": [
        "DEM uses explicit time integration (conditionally stable; dt ~ R/v_sound)",
        "Time step is VERY small compared to FEM (order of 1e-5 to 1e-7 seconds)",
        "MaxTimeStep is the only key that sets dt and it is used verbatim; Kratos DEM "
        "performs no stability check and prints no critical-time-step warning",
        "Particle overlap should be small (<5% of radius) for accurate results",
        "Contact stiffness must be calibrated against bulk material properties",
        "DEM is computationally expensive; use coarsest particles possible",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# 10. MPM APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

MPM_APPLICATION = {
    "application": "MPMApplication",
    "description": "Material Point Method for large deformation problems",
    "import": "from KratosMultiphysics.MPMApplication.mpm_analysis import MPMAnalysis",

    # solver_type takes a SHORT LABEL, not a solver class name. Confirmed by
    # execution against MPMApplication 10.4.3 (2026-08-07): the class-name
    # spellings previously listed here all raise
    # 'The requested solver type "..." is not in the python solvers wrapper'.
    "solver_types": {
        "static": "Static equilibrium (no inertia). Also accepted: 'Static'",
        "quasi_static": "Quasi-static (slow loading). Also accepted: 'Quasi-static'",
        "dynamic": ("Transient. Also accepted: 'Dynamic'. Requires a second key "
                    "time_integration_method: 'implicit' or 'explicit'"),
    },

    # Registered element names, confirmed by ModelPart.CreateNewElement against an
    # installed MPMApplication 10.4.3 (2026-08-07). Every name carries the literal
    # MPM prefix; the un-prefixed spellings previously listed here are NOT
    # registered and raise 'The Element "..." is not registered!'.
    "element_types": {
        "MPMUpdatedLagrangianUP2D3N": "Mixed U-P triangle (stabilized, incompressible)",
        "MPMUpdatedLagrangian2D3N": "Standard displacement triangle",
        "MPMUpdatedLagrangian2D4N": "Standard displacement quad",
        "MPMUpdatedLagrangian3D4N": "Standard displacement tetrahedron",
        "MPMUpdatedLagrangian3D8N": "Standard displacement hexahedron",
        "MPMUpdatedLagrangianAxisymmetry2D3N": "Axisymmetric triangle",
    },

    # The number of material points per element is an integer in the MATERIALS json
    # (properties[i].Material.Variables.MATERIAL_POINTS_PER_ELEMENT), drawn from a
    # fixed per-geometry set. There is no "quadrature_types" concept in Kratos MPM;
    # the labels "MP", "MP_16" and "Gauss_Legendre" previously listed here appear in
    # zero files of the installed distribution and have been removed.
    # Allowed sets read back from the installed libKratosMPMCore 10.4.3 and
    # confirmed by execution (2026-08-07). They differ per geometry.
    "material_points_per_element": {
        "key": "MATERIAL_POINTS_PER_ELEMENT (int, in the materials json)",
        "Triangular": [1, 3, 4, 6, 12],
        "Quadrilateral": [1, 4, 9, 16, 25],
        "Tetrahedral": [1, 4, 8, 14, 24],
        "Hexahedral": [1, 8, 27, 64, 125],
    },

    # Registered law names, confirmed with KratosGlobals.HasConstitutiveLaw against
    # an installed MPMApplication 10.4.3 (2026-08-07). The family labels previously
    # listed here ("LinearElastic", "NeoHookean", "MohrCoulomb", "DruckerPrager",
    # "Johnson-Cook") all return False and fail with 'Kratos components missing'.
    "constitutive_laws": [
        "LinearElasticIsotropic3DLaw / LinearElasticIsotropicPlaneStrain2DLaw / "
        "LinearElasticIsotropicPlaneStress2DLaw / LinearElasticIsotropicAxisym2DLaw",
        "HyperElasticNeoHookean3DLaw / HyperElasticNeoHookeanPlaneStrain2DLaw / "
        "HyperElasticNeoHookeanAxisym2DLaw / HyperElasticNeoHookeanUP3DLaw",
        "HenckyMCPlastic3DLaw / HenckyMCPlasticPlaneStrain2DLaw / "
        "HenckyMCPlasticAxisym2DLaw (Mohr-Coulomb; there is no DruckerPrager law "
        "registered by MPMApplication)",
        "HenckyBorjaCamClayPlastic3DLaw / HenckyBorjaCamClayPlasticPlaneStrain2DLaw",
        "JohnsonCookThermalPlastic3DLaw / JohnsonCookThermalPlastic2DPlaneStrainLaw / "
        "JohnsonCookThermalPlastic2DAxisymLaw",
    ],

    "applications": [
        "Landslides and slope failures",
        "Impact and penetration problems",
        "Free-surface flows (Eulerian mode)",
        "Soil-structure interaction",
        "Explosion and blast loading",
    ],

    "pitfalls": [
        "Background grid must cover the whole TRAJECTORY of the body, not just its "
        "initial position — material points that leave the grid are ERASED and the "
        "mass they carry leaves the simulation, logged only as "
        "'MaterialPointEraseProcess: N particle elements have been erased.'",
        "Material points carry all history; the grid is rebuilt each step",
        "Boundary conditions attach to sub model parts of 'Background_Grid'; materials "
        "attach to sub model parts of 'Initial_MPM_Material'. Both names are fixed",
        "Cell-crossing instability: Kratos MPM implements neither FLIP/PIC blending "
        "nor GIMP/CPDI (none of those strings occurs in MPMApplication). The "
        "mitigation it does offer is PQMPM, enabled with \"is_pqmpm\": true",
        "Explicit MPM: dt limited by CFL condition on background grid",
        "Gravity is opt-in: with no assign_gravity_to_material_point_process the body "
        "does not fall and the run still succeeds",
        "For geomechanics: initialize stress state before loading",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# 11. MESHING APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

MESHING = {
    "application": "MeshingApplication",
    "description": "Mesh generation, adaptation, and remeshing",

    "backends": {
        "Triangle": "2D Delaunay triangulation (constrained)",
        "TetGen": "3D tetrahedral meshing",
        "MMG": "Isotropic/anisotropic remeshing (2D, 3D, surfaces)",
    },

    "mmg_process": {
        "description": "Metric-based adaptive remeshing using MMG library",
        "capabilities": [
            "mmg2d: 2D triangle adaptation",
            "mmgs: Surface triangle adaptation",
            "mmg3d: 3D tetrahedral adaptation",
        ],
        "features": [
            "Preserves SubModelParts during remeshing",
            "Interpolates nodal values between old and new mesh",
            "Supports Hessian-based metric (solution-adaptive)",
            "Supports gradient-based metric",
            "Supports custom metric field",
        ],
    },

    "local_refinement": {
        "LocalRefineTriangleMesh": "Bisection-based 2D refinement",
        "LocalRefineTetrahedraMesh": "Bisection-based 3D refinement",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 12. SHAPE & TOPOLOGY OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

OPTIMIZATION = {
    "ShapeOptimizationApplication": {
        "method": "Vertex Morphing",
        "description": "Node-based shape optimization using sensitivity filtering",
        "features": [
            "Adjoint-based sensitivity analysis",
            "Vertex Morphing parameterization (1e4-1e6 design variables)",
            "Adaptive filter radii",
            "Constraint handling (volume, stress)",
        ],
    },
    "OptimizationApplication": {
        "description": "General optimization framework",
        "methods": [
            "SIMP topology optimization",
            "Gradient-based (adjoint)",
            "Gradient-free (evolutionary, genetic)",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 13. REDUCED ORDER MODELING (RomApplication)
# ═══════════════════════════════════════════════════════════════════════════

ROM = {
    "application": "RomApplication",
    "description": "Reduced Order Models for fast parametric studies",

    "methods": {
        "POD": "Proper Orthogonal Decomposition (snapshot-based basis)",
        "HROM": "Hyper-Reduced Order Model (empirical cubature for nonlinear)",
    },

    "workflow": [
        "1. Run full-order model snapshots for training parameters",
        "2. Compute POD basis (stored in RomParameters.json)",
        "3. Optionally train HROM (EmpiricalCubature -> ElementsAndWeights.json)",
        "4. Run ROM/HROM for new parameters (orders of magnitude faster)",
    ],

    "configuration_files": {
        "RomParameters.json": "POD basis vectors and ROM configuration",
        "ElementsAndWeights.json": "HROM element selection and integration weights",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# MDPA MESH FORMAT (complete specification)
# ═══════════════════════════════════════════════════════════════════════════

MDPA_FORMAT = {
    "description": "Kratos model part file format (.mdpa)",
    "encoding": "Free format (whitespace-insensitive)",

    "blocks": {
        "ModelPartData": {
            "syntax": "Begin ModelPartData\n  VARIABLE value\nEnd ModelPartData",
            "description": "Global model-part-level data",
        },
        "Properties": {
            "syntax": "Begin Properties <id>\n  VARIABLE value\nEnd Properties",
            "description": "Material property sets (referenced by elements)",
            "value_types": {
                "scalar": "THICKNESS 19.5",
                "vector": "VOLUME_ACCELERATION [3] (0.0, 0.0, -9.81)",
                "matrix": "LOCAL_INERTIA [3,3] ((0,0.27,0.27),(0.087,0,0.27),(0.075,0.23,0))",
            },
        },
        "Nodes": {
            "syntax": "Begin Nodes\n  <id> <x> <y> <z>\nEnd Nodes",
            "description": "Node coordinates (always 3D, z=0 for 2D)",
        },
        "Elements": {
            "syntax": "Begin Elements <element_name>\n  <id> <prop_id> <n1> <n2> ... <nN>\nEnd Elements",
            "description": "Element connectivity (element_name = registered element type)",
        },
        "Conditions": {
            "syntax": "Begin Conditions <condition_name>\n  <id> <prop_id> <n1> <n2> ... <nN>\nEnd Conditions",
            "description": "Boundary conditions and loads",
        },
        "NodalData": {
            "syntax": "Begin NodalData <VARIABLE>\n  <node_id> <is_fixed(0/1)> <value>\nEnd NodalData",
            "description": "Node-level initial/fixed values. Accessed via GetSolutionStepValue().",
        },
        "ElementalData": {
            "syntax": "Begin ElementalData <VARIABLE>\n  <elem_id> <value>\nEnd ElementalData",
            "description": "Element-level data. Accessed via GetValue().",
        },
        "ConditionalData": {
            "syntax": "Begin ConditionalData <VARIABLE>\n  <cond_id> <value>\nEnd ConditionalData",
            "description": "Condition-level data.",
        },
        "Table": {
            "syntax": "Begin Table <id> <var_x> <var_y>\n  <x> <y>\nEnd Table",
            "description": "Piecewise-linear lookup tables for time-varying properties.",
        },
        "SubModelPart": {
            "syntax": (
                "Begin SubModelPart <name>\n"
                "  Begin SubModelPartData\n    ...\n  End SubModelPartData\n"
                "  Begin SubModelPartTables\n    <table_id>\n  End SubModelPartTables\n"
                "  Begin SubModelPartNodes\n    <node_id>\n  End SubModelPartNodes\n"
                "  Begin SubModelPartElements\n    <elem_id>\n  End SubModelPartElements\n"
                "  Begin SubModelPartConditions\n    <cond_id>\n  End SubModelPartConditions\n"
                "  Begin SubModelPart <nested_name>\n    ...\n  End SubModelPart\n"
                "End SubModelPart"
            ),
            "description": "Hierarchical groups for BCs, loads, output. Can be nested.",
        },
    },

    "element_naming_convention": {
        "pattern": "<ElementType><Dim>D<NumNodes>N",
        "examples": {
            "SmallDisplacementElement2D3N": "Small disp, 2D, triangle (3 nodes)",
            "TotalLagrangianElement3D10N": "Total Lagrangian, 3D, quadratic tet (10 nodes)",
            "VMS2D3N": "VMS fluid, 2D, triangle",
            "EulerianConvDiff3D4N": "Convection-diffusion, 3D, tet",
            "CrBeamElement3D2N": "Co-rotational beam, 3D, 2 nodes",
            "ShellThinElement3D3N": "Thin shell, 3D, triangle",
        },
    },

    "condition_naming_convention": {
        "pattern": "<ConditionType><Dim>D<NumNodes>N",
        "examples": {
            "PointLoadCondition3D1N": "Point load, 3D, 1 node",
            "LineLoadCondition2D2N": "Line load, 2D, 2 nodes",
            "SurfaceLoadCondition3D3N": "Surface load, 3D, triangle",
            "MonolithicWallCondition3D3N": "Fluid wall, 3D, triangle",
            "LineCondition2D2N": "Generic line condition",
        },
    },

    "complete_mdpa_example": '''Begin ModelPartData
End ModelPartData

Begin Properties 1
End Properties

Begin Nodes
  1  0.0000000000  0.0000000000  0.0000000000
  2  1.0000000000  0.0000000000  0.0000000000
  3  1.0000000000  1.0000000000  0.0000000000
  4  0.0000000000  1.0000000000  0.0000000000
  5  0.5000000000  0.5000000000  0.0000000000
End Nodes

Begin Elements SmallDisplacementElement2D3N
  1  1  1  2  5
  2  1  2  3  5
  3  1  3  4  5
  4  1  4  1  5
End Elements

Begin Conditions LineLoadCondition2D2N
  1  1  2  3
End Conditions

Begin SubModelPart Parts_Body
  Begin SubModelPartNodes
    1
    2
    3
    4
    5
  End SubModelPartNodes
  Begin SubModelPartElements
    1
    2
    3
    4
  End SubModelPartElements
End SubModelPart

Begin SubModelPart FixedSupport
  Begin SubModelPartNodes
    1
    4
  End SubModelPartNodes
End SubModelPart

Begin SubModelPart LoadSurface
  Begin SubModelPartNodes
    2
    3
  End SubModelPartNodes
  Begin SubModelPartConditions
    1
  End SubModelPartConditions
End SubModelPart
''',
}


# ═══════════════════════════════════════════════════════════════════════════
# BOUNDARY CONDITION PROCESSES (common to all applications)
# ═══════════════════════════════════════════════════════════════════════════

BC_PROCESSES = {
    "AssignScalarVariableProcess": {
        "python_module": "assign_scalar_variable_process",
        "kratos_module": "KratosMultiphysics",
        "description": "Fix or set a scalar variable on a model part",
        "parameters": {
            "model_part_name": "MainModelPart.SubModelPartName",
            "variable_name": "TEMPERATURE",  # or DISPLACEMENT_X, PRESSURE, etc.
            "value": 100.0,       # Can be float or expression: "sqrt(x**2+y**2)*t"
            "constrained": True,  # True = Dirichlet BC, False = initial condition
            "interval": [0.0, "End"],  # Time interval for application
        },
    },
    "AssignVectorVariableProcess": {
        "python_module": "assign_vector_variable_process",
        "kratos_module": "KratosMultiphysics",
        "description": "Fix or set a vector variable (per-component control)",
        "parameters": {
            "model_part_name": "MainModelPart.SubModelPartName",
            "variable_name": "DISPLACEMENT",
            "value": [0.0, 0.0, None],  # None = ignore that component
            "constrained": [True, True, False],
            "interval": [0.0, "End"],
        },
    },
    "AssignVectorByDirectionProcess": {
        "python_module": "assign_vector_by_direction_process",
        "kratos_module": "KratosMultiphysics",
        "description": "Set a vector by direction + magnitude",
        "parameters": {
            "model_part_name": "MainModelPart.SubModelPartName",
            "variable_name": "VELOCITY",
            "modulus": 1.0,           # Can be expression
            "direction": [1.0, 0.0, 0.0],  # Normalized internally
            "constrained": True,
            "interval": [0.0, "End"],
        },
    },
    "ApplyConstantScalarValueProcess": {
        "python_module": "process_factory",
        "kratos_module": "KratosMultiphysics",
        "process_name": "ApplyConstantScalarValueProcess",
        "description": ("Legacy C++ process (upstream docs recommend assign_scalar_variable_process); "
                        "sets a constant scalar at ExecuteInitialize"),
        "parameters": {
            "model_part_name": "MainModelPart.SubModelPartName",
            "variable_name": "TEMPERATURE",
            "value": 0.0,
            "is_fixed": True,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# VTK OUTPUT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

VTK_OUTPUT = {
    "python_module": "vtk_output_process",
    "kratos_module": "KratosMultiphysics",
    "process_name": "VtkOutputProcess",
    "default_parameters": {
        "model_part_name": "PLEASE_SPECIFY_MODEL_PART_NAME",  # must be set
        "file_format": "binary",    # "ascii" or "binary"
        "output_precision": 7,
        "output_control_type": "step",  # "step" or "time"
        "output_interval": 1.0,     # Every N steps or dt seconds
        "output_sub_model_parts": False,
        "output_path": "VTK_Output",
        "save_output_files_in_folder": True,
        "nodal_solution_step_data_variables": [],  # e.g. ["DISPLACEMENT", "VELOCITY", "PRESSURE"]
        "nodal_data_value_variables": [],
        "element_data_value_variables": [],
        "gauss_point_variables_in_elements": [],   # e.g. ["VON_MISES_STRESS", "CAUCHY_STRESS_TENSOR"]
        "condition_data_value_variables": [],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN KRATOS.PY DRIVER TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

MAIN_KRATOS_TEMPLATES = {
    "structural_mechanics": '''import KratosMultiphysics
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

if __name__ == "__main__":
    with open("ProjectParameters.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    model = KratosMultiphysics.Model()
    simulation = StructuralMechanicsAnalysis(model, parameters)
    simulation.Run()
''',

    "fluid_dynamics": '''import KratosMultiphysics
from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis

if __name__ == "__main__":
    with open("ProjectParameters.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    model = KratosMultiphysics.Model()
    simulation = FluidDynamicsAnalysis(model, parameters)
    simulation.Run()
''',

    "convection_diffusion": '''import KratosMultiphysics
from KratosMultiphysics.ConvectionDiffusionApplication.convection_diffusion_analysis import ConvectionDiffusionAnalysis

if __name__ == "__main__":
    with open("ProjectParameters.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    model = KratosMultiphysics.Model()
    simulation = ConvectionDiffusionAnalysis(model, parameters)
    simulation.Run()
''',

    "co_simulation": '''import KratosMultiphysics
from KratosMultiphysics.CoSimulationApplication.co_simulation_analysis import CoSimulationAnalysis

if __name__ == "__main__":
    with open("ProjectParametersCoSim.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    simulation = CoSimulationAnalysis(parameters)  # signature: CoSimulationAnalysis(cosim_settings, models=None); no Model argument
    simulation.Run()
''',

    "mpm": '''import KratosMultiphysics
from KratosMultiphysics.MPMApplication.mpm_analysis import MPMAnalysis

if __name__ == "__main__":
    with open("ProjectParameters.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    model = KratosMultiphysics.Model()
    simulation = MPMAnalysis(model, parameters)
    simulation.Run()
''',

    "fluid_thermal_coupled": '''import KratosMultiphysics
from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis

# One-way coupling: fluid velocity drives thermal convection
# Both solvers share the same mesh via ConnectivityPreserveModeler

class FluidThermalAnalysis(FluidDynamicsAnalysis):
    """Custom analysis stage for fluid-thermal coupling."""

    def __init__(self, model, parameters):
        super().__init__(model, parameters)
        # Thermal solver initialized separately using shared model part

if __name__ == "__main__":
    with open("ProjectParameters.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    model = KratosMultiphysics.Model()
    simulation = FluidThermalAnalysis(model, parameters)
    simulation.Run()
''',
}


# ═══════════════════════════════════════════════════════════════════════════
# LINEAR SOLVER OPTIONS
# ═══════════════════════════════════════════════════════════════════════════

LINEAR_SOLVERS = {
    "direct": {
        "LinearSolversApplication.sparse_lu": {
            "description": "Eigen SparseLU (serial, robust, small-medium problems)",
            "max_recommended_dofs": 100000,
        },
        "skyline_lu_factorization": {
            "description": "Built-in skyline LU (serial, simple problems only)",
            "max_recommended_dofs": 50000,
        },
    },
    "iterative": {
        "amgcl": {
            "description": "AMGCL algebraic multigrid (default iterative, very fast)",
            "parameters": {
                "solver_type": "amgcl",
                "max_iteration": 200,
                "tolerance": 1e-7,
                "provide_coordinates": False,
                "smoother_type": "ilu0",  # or "damped_jacobi", "gauss_seidel"
                "krylov_type": "gmres",   # or "cg" for symmetric positive definite
                "coarsening_type": "aggregation",  # or "smoothed_aggregation"
                "scaling": False,
            },
        },
        "LinearSolversApplication.sparse_qr": {
            "description": "Eigen SparseQR (for non-square or poorly conditioned)",
        },
    },
    "trilinos_mpi": {
        # TrilinosApplication solver_type values; there is no MueLu solver. The Amesos2 types
        # (amesos2, klu2, basker, ...) exist only in builds with TRILINOS_EXCLUDE_AMESOS2_SOLVER=OFF (default ON)
        "aztec": "AztecOO Krylov solver (MPI; also cg, bicgstab, gmres)",
        "amesos": "Amesos direct solver (MPI; also klu, super_lu_dist, mumps)",
        "multi_level": "ML AMG preconditioned solver (MPI)",
        "amgcl": "AMGCL AMG solver (MPI)",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# KNOWN PITFALLS AND GOTCHAS (CROSS-APPLICATION)
# ═══════════════════════════════════════════════════════════════════════════

GLOBAL_PITFALLS = [
    # ── File System ──
    "Kratos ALWAYS needs 3 files: MainKratos.py + ProjectParameters.json + mesh.mdpa",
    "Materials are in a SEPARATE JSON file (StructuralMaterials.json, FluidMaterials.json, etc.)",
    "The materials_filename in ProjectParameters must match the actual file name exactly",

    # ── Element Naming ──
    "Element names are case-sensitive and follow pattern: <Type><Dim>D<Nodes>N",
    "The element name in .mdpa must match a REGISTERED element (typos give cryptic errors)",
    "Different applications register different elements; mixing is not allowed",

    # ── SubModelParts ──
    "SubModelPart names in processes MUST match those in .mdpa exactly (case-sensitive)",
    "Use dot notation for nested SubModelParts: 'Structure.Parts_Body.SubPart'",
    "model_part_name in processes: 'MainModelPartName.SubModelPartName'",

    # ── Convergence ──
    "Non-convergence: first try reducing time_step (load increment)",
    "For nonlinear: start with small loads and ramp up using pseudo-time stepping",
    "Check that BCs don't over-constrain the problem (causes singular matrix)",

    # ── Common Errors ──
    "ERROR: 'The variables list doesn't have this variable:' -> add variable via AddNodalSolutionStepVariable",
    "ERROR: 'Element not registered' -> wrong application imported, or typo in element name",
    "NO ERROR for missing BCs: Kratos has no 'Zero diagonal in system matrix' error; the block builder silently "
    "sets the diagonal of all-zero rows, and a singular system (no Dirichlet BCs) solves without an exception to "
    "a meaningless field. At most the linear solver warns (amgcl: 'Non converged linear solution'; "
    "skyline_lu_factorization can print 'Error zero sum') -> check BCs and zero-volume elements yourself",
    "ERROR: 'WARNING:: ELEMENT ID: N INVERTED. DETJ0: ...' (SmallDisplacement/TotalLagrangian/UpdatedLagrangian "
    "elements) -> element nodes in wrong order, or severely distorted mesh",

    # ── Performance ──
    "For > 100k DOFs: use AMGCL iterative solver, not direct LU",
    "For MPI: need TrilinosApplication + MetisApplication for partitioning",
    "VTK output: set output_interval > 1 to avoid I/O bottleneck",

    # ── ConvectionDiffusion Specific ──
    # NOTE 2026-06-01: prior text said "LaplacianElement does NOT
    # assemble HEAT_FLUX" — that was WRONG. Empirically verified
    # (Kratos 10.4.2 + KratosConvectionDiffusionApplication 10.4.2,
    # Tier-2 fixture
    # scripts/tier2_fixtures/kratos/poisson_laplacian_element_assembles_heat_flux/)
    # that LaplacianElement2D3N.CalculateRightHandSide returns the
    # consistent P1 load 10 * area / 3 = 1.66667 per node when
    # HEAT_FLUX=10 is set on all 3 nodes of a unit-right triangle.
    "LaplacianElement2D3N / LaplacianElement3D4N DO assemble the "
    "HEAT_FLUX volumetric source term — the prior catalog text was "
    "wrong; see Tier-2 fixture poisson_laplacian_element_assembles_"
    "heat_flux.",
    "Choose between LaplacianElement and EulerianConvDiff based "
    "on whether the problem has CONVECTION_VELOCITY or transient "
    "mass, NOT based on whether HEAT_FLUX source is supported "
    "(both support it).",
    "ConvectionDiffusion assigns material properties NODALLY, not element-wise",
]
