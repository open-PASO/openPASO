"""
MCP-side deep domain knowledge for SELECTED backends.

This file is NOT a complete brain — it carries prose-style knowledge for
only **4 of 8** registered backends, in two roles:

  1. **Canonical** for FEniCSx (`_FENICS_KNOWLEDGE`, 31 keys, ~91 KB):
     fenics/backend.py reads this FIRST and falls back to its own
     generator KNOWLEDGE only when a key is missing. Touch with care
     — adding/renaming keys here can shadow generator pitfalls.

  2. **Last-fallback supplement** for deal.II (`_DEALII_KNOWLEDGE`,
     14 keys, ~20 KB): dealii/backend.py reads
     data/dealii_knowledge.py first, then generator KNOWLEDGE, then
     this dict. Useful only for keys NOT covered upstream.

  3. **Workflows.py merge source** for 4C/fourc (`_4C_KNOWLEDGE`,
     9 keys, ~11 KB): the `module_knowledge` workflow overlays
     backend pitfalls onto this dict for prose like description /
     weak_form / problem_type / materials.

Cross-solver hints (`_CROSS_SOLVER_KNOWLEDGE`) supply small
benchmark-verification notes and live independently of any backend.

**NOT covered here** (their canonical knowledge lives elsewhere):
skfem, NGSolve, Kratos, DUNE, FEBio — these backends carry their
catalogs in `src/backends/<name>/generators/` plus
`data/<name>_knowledge.py` (where applicable). Do not add new
entries for those backends here; extend the per-backend module
instead.

(`_FEBIO_KNOWLEDGE` was removed 2026-06-02: it had only 4 keys
while the febio backend's own generators carry 16 physics with
matching `description` fields, and the workflows.py merge shape
allowed deep_knowledge to *shadow* the more-comprehensive backend
catalog. The backend is now the single source of truth for febio.)
"""

import json
from mcp.server.fastmcp import FastMCP
from core.registry import get_backend, available_backends


# ═══════════════════════════════════════════════════════════════════════════════
# 4C MULTIPHYSICS — COMPREHENSIVE DOMAIN KNOWLEDGE
# Ported from 4c-ai-interface generators (9 physics modules, 30+ material types)
# ═══════════════════════════════════════════════════════════════════════════════

_4C_KNOWLEDGE = {
    "scalar_transport": {
        "description": "Solves advection-diffusion equation for scalar transport. Special cases: Poisson (stationary, zero velocity), heat conduction, SUPG-stabilised advection.",
        "problem_type": "Scalar_Transport",
        "required_sections": ["PROBLEM TYPE", "SCALAR TRANSPORT DYNAMIC", "SOLVER 1", "MATERIALS", "TRANSPORT GEOMETRY"],
        "materials": {
            "MAT_scatra": {"DIFFUSIVITY": "Isotropic diffusion coefficient > 0 (typical 0.01-100)"},
            "MAT_Fourier": {"CAPA": "Volumetric heat capacity (rho*c_p) > 0", "CONDUCT": "Thermal conductivity (YAML: constant: [value]) > 0"},
        },
        "time_integration": {
            "TIMEINTEGR": "Stationary | One_Step_Theta (default) | BDF2 | Gen_Alpha",
            "SOLVERTYPE": "linear_full (linear) | nonlinear (nonlinear terms)",
            "VELOCITYFIELD": "zero (pure diffusion) | function (prescribed) | Navier_Stokes",
        },
        "solver": {"small": "UMFPACK (direct, ~50k DOFs)", "large": "Belos + MueLu (iterative, scalable)"},
        "pitfalls": [
            "Section name is 'SCALAR TRANSPORT DYNAMIC', NOT 'SCATRA DYNAMIC'",
            "VELOCITYFIELD defaults to 'zero'; omitting it is fine for pure diffusion",
            "VTK: scalar transport has no runtime-VTK section of its own (both 'SCALAR TRANSPORT DYNAMIC/RUNTIME VTK OUTPUT' and 'IO/RUNTIME VTK OUTPUT/SCATRA' abort with \"Section '...' is not a valid section name.\"); it writes <prefix>-vtk-files/scatra-*.vtu and <prefix>-scatra.pvd with no IO section, at every RESULTSEVERY and every RESTARTEVERY step (both default 1), in the data format of IO/RUNTIME VTK OUTPUT (e.g. OUTPUT_DATA_FORMAT; INTERVAL_STEPS is ignored)",
            "Geometry section: TRANSPORT GEOMETRY with TRANSP element category",
            "NUMDOF=1, all arrays (ONOFF/VAL/FUNCT) have exactly 1 entry",
        ],
        "variants": ["poisson_2d", "heat_transient_2d"],
    },
    "solid_mechanics": {
        "description": "Quasi-static structural problems. DYNAMICTYPE: Statics, small/large deformation, 2D (WALL before 4C 2026.2.0, SOLID with THICKNESS and PLANE_ASSUMPTION from 4C 2026.2.0 on) / 3D (SOLID).",
        "problem_type": "Structure",
        "required_sections": ["PROBLEM TYPE", "STRUCTURAL DYNAMIC", "SOLVER 1", "MATERIALS", "STRUCTURE GEOMETRY"],
        "materials": {
            "MAT_Struct_StVenantKirchhoff": {"YOUNG": "> 0 (steel 210e3 MPa)", "NUE": "0 < nu < 0.5", "DENS": "Optional for statics"},
            "MAT_ElastHyper + ELAST_CoupNeoHooke": {"NUMMAT": "1", "MATIDS": "[id]", "DENS": "> 0"},
            "MAT_Struct_PlasticNlnLogNeoHooke": {"YOUNG": "> 0", "NUE": "0 < nu < 0.5", "YIELD": "Initial yield > 0", "SATHARDENING": ">= 0", "HARDEXPO": "> 0"},
        },
        "time_integration": {
            "DYNAMICTYPE": "Statics (quasi-static, incremental loading)",
            "KINEM": "linear (small def) vs nonlinear (large def)",
            "MAXITER": "1 for linear, 20-50 for nonlinear",
            "TOLDISP": "1e-6 to 1e-10", "TOLRES": "1e-6 to 1e-10",
        },
        "solver": {"small": "UMFPACK (direct, ~50k DOFs)", "large": "Belos + MueLu (GMRES + AMG)"},
        "pitfalls": [
            "KINEM must match material: Neo-Hookean/plasticity REQUIRE nonlinear",
            "MAXITER=1 only for truly linear problems",
            "HEX8 suffers locking — use TECH: eas_full, fbar, or higher-order elements",
            "2D before 4C 2026.2.0 uses WALL (not SOLID), with THICK and STRESS_STRAIN; 4C 2026.2.0 and later have no WALL, their 2D SOLID takes THICKNESS and PLANE_ASSUMPTION",
            "Neumann BCs on SOLID/WALL need NUMDOF >= the dimension (3 is enough in 3D, 2 in 2D); NUMDOF: 6 is accepted, but the entries beyond the dimension are not moments and must stay 0: a switched-on one aborts on a 3-D SOLID surface load ('Number of Dimensions in Neumann_Evaluation is 3. Further DoFs are not considered.') and on the 2-D SOLID of 4C 2026.2.0 and later ('You have activated more than 2 dofs in your Neumann boundary condition. This is higher than the dimension of the element.'), while WALL (4C before 2026.2.0) ignores it",
        ],
        "variants": ["linear_2d", "nonlinear_3d"],
    },
    "fluid": {
        "description": "Incompressible Navier-Stokes with SUPG/PSPG stabilisation. Fixed Eulerian (NA: Euler) or ALE (for FSI).",
        "problem_type": "Fluid",
        "required_sections": ["PROBLEM TYPE", "PROBLEM SIZE", "FLUID DYNAMIC", "SOLVER 1", "MATERIALS", "FLUID GEOMETRY"],
        "materials": {
            "MAT_fluid": {"DYNVISCOSITY": "Dynamic viscosity [Pa*s] > 0 (water 1e-3, air 1.8e-5)", "DENSITY": "Fluid density [kg/m^3] > 0 (water 1000, air 1.2)"},
        },
        "time_integration": {
            "schemes": ["Np_Gen_Alpha (RECOMMENDED)", "BDF2", "OneStepTheta", "Stationary"],
            "TIMESTEP": "Time step size", "NUMSTEP": "Number of steps", "ITEMAX": "Max nonlinear iters (default 10)",
        },
        "solver": {"small_2d": "UMFPACK (< ~50k DOFs)", "large_or_3d": "Belos with block preconditioner"},
        "pitfalls": [
            "NUMDOF INCLUDES pressure: 3 in 2D (vx,vy,p), 4 in 3D (vx,vy,vz,p)",
            "Stabilisation (SUPG/PSPG) critical — without it, equal-order elements oscillate",
            "Fully Dirichlet velocity: pressure up to constant — PIN at one node",
            "FLUID GEOMETRY uses FLUID category (not SOLID)",
            "Use NA: Euler for pure fluid, NA: ALE only for FSI mesh motion",
        ],
        "variants": ["channel_2d", "cavity_2d"],
    },
    "fsi": {
        "description": "Monolithic/partitioned coupling of incompressible Navier-Stokes with geometrically nonlinear structures via ALE mesh motion. Most complex problem type in 4C.",
        "problem_type": "Fluid_Structure_Interaction",
        "required_sections": [
            "PROBLEM TYPE", "STRUCTURAL DYNAMIC", "STRUCTURAL DYNAMIC/GENALPHA",
            "FLUID DYNAMIC", "ALE DYNAMIC", "FSI DYNAMIC", "FSI DYNAMIC/MONOLITHIC SOLVER",
            "SOLVER 1, 2, 3", "MATERIALS", "STRUCTURE GEOMETRY", "FLUID GEOMETRY",
            "CLONING MATERIAL MAP", "DESIGN FSI COUPLING LINE CONDITIONS (2D) / DESIGN FSI COUPLING SURF CONDITIONS (3D)",
        ],
        "materials": {
            "MAT_fluid": "Newtonian (DYNVISCOSITY, DENSITY)",
            "MAT_ElastHyper": "Hyperelastic structure (Neo-Hooke)",
            "ALE clone": "Spring-based ALE via CLONING MATERIAL MAP",
        },
        "coupling": {
            "recommended": "iter_mortar_monolithicfluidsplit",
            "alternatives": ["iter_monolithicfluidsplit", "iter_stagg_AITKEN_rel_param"],
        },
        "pitfalls": [
            "Fluid MUST use NA: ALE (NOT Euler!) for FSI",
            "ALE Dirichlet BCs on ALL outer fluid boundaries (not FSI interface) — missing = mesh distortion",
            "CLONING MATERIAL MAP is MANDATORY (fluid mat → ALE pseudo-mat)",
            "SHAPEDERIVATIVES: true in MONOLITHIC SOLVER",
            "Each field (structure, fluid, ALE) needs own SOLVER N entry",
            "2D: DESIGN FSI COUPLING LINE CONDITIONS, 3D: SURF CONDITIONS",
            "Structure NUMDOF = dim, Fluid NUMDOF = dim+1 (includes pressure)",
        ],
        "variants": ["fsi_2d"],
    },
    "beams": {
        "description": "Geometrically exact beam elements: BEAM3R (Reissner, shear-deformable), BEAM3EB (Euler-Bernoulli), BEAM3K (Kirchhoff). CRITICAL: MUST use inline mesh (NODE COORDS + STRUCTURE ELEMENTS), NOT Exodus.",
        "problem_type": "Structure",
        "required_sections": ["PROBLEM TYPE", "STRUCTURAL DYNAMIC", "SOLVER 1", "MATERIALS", "NODE COORDS", "STRUCTURE ELEMENTS", "DNODE-NODE TOPOLOGY", "DLINE-NODE TOPOLOGY"],
        "beam_types": {
            "BEAM3R": {"name": "Reissner (shear-deformable)", "topologies": ["LINE2", "LINE3", "LINE4"], "dofs": "6 or 9 (HERMITE)"},
            "BEAM3EB": {"name": "Euler-Bernoulli (torsion-free)", "topologies": ["LINE2"], "dofs": "6"},
            "BEAM3K": {"name": "Kirchhoff (with torsion)", "topologies": ["LINE2", "LINE3"], "dofs": "6 or 7"},
        },
        "materials": {
            "MAT_BeamReissnerElastHyper": {"YOUNG": "> 0", "SHEARMOD": "G = E/(2(1+nu))", "CROSSAREA": "> 0", "MOMINPOL": "J", "MOMIN2": "I_yy", "MOMIN3": "I_zz", "SHEARCORR": "circle: 6/7, rect: 5/6"},
        },
        "pitfalls": [
            "Beams CANNOT use Exodus — must use inline NODE COORDS + STRUCTURE ELEMENTS",
            "TRIADS required for BEAM3R/K (initial orientation)",
            "LINE3: endpoint1-endpoint2-midpoint ordering (NOT sequential!)",
            "GenAlphaLieGroup REQUIRED for dynamics (not standard GenAlpha)",
            "MASSLIN: rotations required with GenAlphaLieGroup",
            "Cross-section properties must be mutually consistent",
        ],
        "variants": ["cantilever_static", "cantilever_dynamic"],
    },
    "contact": {
        "description": "Mortar-based contact between deformable bodies. Penalty / Uzawa / Nitsche. Adds CONTACT DYNAMIC + MORTAR COUPLING on top of structure.",
        "problem_type": "Structure",
        "required_sections": ["PROBLEM TYPE", "STRUCTURAL DYNAMIC", "MORTAR COUPLING", "CONTACT DYNAMIC", "SOLVER 1", "MATERIALS", "STRUCTURE GEOMETRY", "DESIGN SURF MORTAR CONTACT CONDITIONS 3D"],
        "materials": {
            "MAT_Struct_StVenantKirchhoff": {"YOUNG": "> 0", "NUE": "0 < nu < 0.5", "DENS": ">= 0 (0 for quasi-static)"},
            "MAT_ElastHyper": "For large-deformation contact",
        },
        "strategies": {
            "Penalty": "Stiff spring on penetration. PENALTYPARAM (1e2-1e5): too low=penetration, too high=ill-conditioning",
            "Uzawa": "Augmented Lagrangian. Accurate, expensive. Only with STRUCTURAL DYNAMIC INT_STRATEGY: Old: under Standard (the default after the 4C 2026.1.0 release, which defaults to Old) it aborts with 'This contact strategy is not yet considered!', and the Old integration has no runtime VTK output (IO/RUNTIME VTK OUTPUT/STRUCTURE OUTPUT_STRUCTURE: true aborts with 'Runtime output is not available in the old structure time integration! You need to take the new one, i.e. set `INT_STRATEGY: Standard`!').",
            "Nitsche": "Variationally consistent penalty. Accuracy + simplicity.",
        },
        "pitfalls": [
            "Both MORTAR COUPLING and CONTACT DYNAMIC required — missing either crashes/ignores",
            "Each interface needs BOTH Slave and Master with same InterfaceID",
            "PENALTYPARAM tuning critical: start 1e3, adjust by penetration depth",
            "Quasi-static MUST use load stepping — full load in 1 step → Newton divergence",
            "Slave surface = finer mesh or softer body",
            "KINEM must be nonlinear for correct gap computation",
            "Contact surfaces must NOT overlap initially",
        ],
        "variants": ["penalty_3d"],
    },
    "structural_dynamics": {
        "description": "Time-dependent structural: impact, vibration, wave propagation. GenAlpha (implicit, recommended) or ExplicitEuler.",
        "problem_type": "Structure",
        "required_sections": ["PROBLEM TYPE", "STRUCTURAL DYNAMIC", "SOLVER 1", "MATERIALS", "STRUCTURE GEOMETRY"],
        "materials": {
            "MAT_Struct_StVenantKirchhoff": {"YOUNG": "> 0", "NUE": "0 < nu < 0.5", "DENS": "MANDATORY > 0 for dynamics (zero = singular mass matrix!)"},
            "MAT_ElastHyper + ELAST_CoupNeoHooke": {"YOUNG": "> 0", "NUE": "0 < nu < 0.5", "DENS": "> 0 in wrapper"},
        },
        "time_integration": {
            "GenAlpha": "Implicit, 2nd order, RHO_INF [0,1]: 1=energy-conserving, 0=max damping (typical 0.8-0.9)",
            "GenAlphaLieGroup": "Lie-group variant for beams (rotational DOFs on SO(3))",
            "ExplicitEuler": "Explicit, CFL-constrained (dt < h/c where c=sqrt(E/rho))",
        },
        "damping": {"Rayleigh": "M_DAMP (low freq) + K_DAMP (high freq)", "None": "Numerical dissipation only"},
        "pitfalls": [
            "DENS MANDATORY and > 0 — zero/missing = zero mass matrix (singular)",
            "Time step must resolve highest frequency of interest",
            "Explicit: CFL violation = immediate divergence",
            "RHO_INF=1: energy-conserving but may show spurious ringing — reduce to 0.8",
        ],
        "variants": ["genalpha_2d"],
    },
    "particle_pd": {
        "description": "Bond-based peridynamics for fracture. Non-local integral equations. CRITICAL: SPH section MANDATORY even for pure PD (else 'pd_neighbor_pairs=0' crash).",
        "problem_type": "Particle",
        "required_sections": ["PROBLEM TYPE", "IO", "BINNING STRATEGY", "PARTICLE DYNAMIC", "PARTICLE DYNAMIC/SPH", "PARTICLE DYNAMIC/PD", "MATERIALS", "PARTICLES"],
        "materials": {
            "MAT_ParticlePD": {"INITRADIUS": "dx/2", "INITDENSITY": "e.g. 8e-3 g/mm^3 steel", "YOUNG": "e.g. 190e3 MPa steel", "CRITICAL_STRETCH": "Bond break 0.001-0.05"},
            "MAT_ParticleSPHBoundary": {"INITRADIUS": "Same as PD", "INITDENSITY": "Can be 1 (rigid)"},
        },
        "solver": "Explicit only (VelocityVerlet). dt < dx/sqrt(E/rho), safety factor 0.5.",
        "pre_cracks": "Visibility condition: bonds crossing line segments broken at init. Format: 'x1 y1 x2 y2 ; x3 y3 x4 y4'",
        "pitfalls": [
            "PARTICLE DYNAMIC/SPH section MANDATORY for PD — missing causes crash",
            "DOMAINBOUNDINGBOX must enclose ALL particles including moving impactor",
            "Use boundaryphase (NOT rigidphase) for rigid impactors",
            "Horizon ratio m=delta/dx >= 3 for convergence",
            "BIN_SIZE_LOWER_BOUND > horizon (else neighbors missed)",
            "Bond-based PD restricts Poisson's ratio: nu=0.25 (2D), nu=1/3 (3D)",
            "CFL violation = UNSTABLE",
        ],
        "unit_systems": {"mm_ms_g": "Length=mm, Time=ms, Mass=g, Stress=MPa", "SI": "Length=m, Time=s, Mass=kg, Stress=Pa"},
        "variants": ["plate_2d", "impact_2d"],
    },
    "particle_sph": {
        "description": "Smoothed Particle Hydrodynamics for free-surface flows (dam break, sloshing). Meshfree Lagrangian, kernel-weighted summation.",
        "problem_type": "Particle",
        "required_sections": ["PROBLEM TYPE", "IO", "BINNING STRATEGY", "PARTICLE DYNAMIC", "PARTICLE DYNAMIC/SPH", "MATERIALS", "PARTICLES"],
        "materials": {
            "MAT_ParticleSPHFluid": {
                "INITRADIUS": "Kernel support (3*dx for QuinticSpline)", "INITDENSITY": "Reference density",
                "BULK_MODULUS": "Artificial (>> rho*v_max^2)", "DYNAMIC_VISCOSITY": "Physical viscosity",
                "ARTIFICIAL_VISCOSITY": "Monaghan shock capturing (0-1, typical 0.1)",
                "BACKGROUNDPRESSURE": "> 0 for free-surface", "EXPONENT": "Tait EOS (1=linear, 7=water)",
            },
        },
        "solver": "Explicit (VelocityVerlet). CFL: dt < 0.25*h/c_s where c_s=sqrt(K/rho).",
        "pitfalls": [
            "KERNEL_SPACE_DIM MUST match physical dimension — mismatch = wrong normalization",
            "INITRADIUS is kernel support radius (3*dx for QuinticSpline), NOT half spacing",
            "DOMAINBOUNDINGBOX must accommodate fluid expansion/splashing",
            "BULK_MODULUS >= 100*rho*v_max^2 for <1% density variation",
            "Boundary particles MUST use same INITDENSITY as fluid (Adami formulation)",
            "BACKGROUNDPRESSURE > 0 for free-surface problems",
        ],
        "variants": ["poiseuille_2d", "dam_break_2d"],
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# FENICSX (DOLFINX) — COMPREHENSIVE DOMAIN KNOWLEDGE
# ═══════════════════════════════════════════════════════════════════════════════


def get_deep_fenics_knowledge(physics: str) -> dict:
    """Get deep FEniCSx knowledge for a specific physics type."""
    return _FENICS_KNOWLEDGE.get(physics, {})


_FENICS_KNOWLEDGE = {'element_catalog': {'description': 'Complete catalog of finite element families '
                                    'available in FEniCSx via Basix. Elements are '
                                    'created with basix.ufl.element() or '
                                    'basix.ufl.blocked_element().',
                     'basix_element_families': {'P (Lagrange)': {'basix_name': 'basix.ElementFamily.P',
                                                                 'ufl_name': "'Lagrange' "
                                                                             "or 'P'",
                                                                 'continuity': 'C0 '
                                                                               '(continuous '
                                                                               'across '
                                                                               'facets)',
                                                                 'orders': '1, 2, 3, '
                                                                           '... '
                                                                           '(arbitrary '
                                                                           'order); '
                                                                           'pyramid is '
                                                                           'capped at '
                                                                           'degree 2 with '
                                                                           'the default '
                                                                           'variant — '
                                                                           'degree 3 '
                                                                           'raises '
                                                                           'RuntimeError '
                                                                           "'Non-equispaced "
                                                                           'points on '
                                                                           'pyramids '
                                                                           'not '
                                                                           'supported '
                                                                           "yet.' unless "
                                                                           'lagrange_variant='
                                                                           'equispaced is '
                                                                           'passed (then '
                                                                           'degrees 3 and 4 '
                                                                           'build) '
                                                                           '(verified '
                                                                           'basix '
                                                                           '0.10.0)',
                                                                 'cell_types': 'interval, '
                                                                               'triangle, '
                                                                               'quadrilateral, '
                                                                               'tetrahedron, '
                                                                               'hexahedron, '
                                                                               'prism, '
                                                                               'pyramid',
                                                                 'api': "basix.ufl.element('Lagrange', "
                                                                        'cell, degree)',
                                                                 'variants': {'equispaced': 'basix.LagrangeVariant.equispaced '
                                                                                            '(equally '
                                                                                            'spaced '
                                                                                            'points; '
                                                                                            'not the default: '
                                                                                            'basix.ufl.element '
                                                                                            'uses gll_warped for '
                                                                                            'P/DG at every degree, '
                                                                                            'and its points coincide '
                                                                                            'with equispaced for '
                                                                                            'degree <= 2)',
                                                                              'gll_warped': 'basix.LagrangeVariant.gll_warped '
                                                                                            '(GLL '
                                                                                            'points, '
                                                                                            'lower '
                                                                                            'Lebesgue '
                                                                                            'constant '
                                                                                            'for '
                                                                                            'high '
                                                                                            'order)',
                                                                              'gll_isaac': 'basix.LagrangeVariant.gll_isaac '
                                                                                           '(GLL '
                                                                                           'with '
                                                                                           'Isaac '
                                                                                           'warp '
                                                                                           'on '
                                                                                           'simplices)',
                                                                              'gll_centroid': 'basix.LagrangeVariant.gll_centroid '
                                                                                              '(GLL '
                                                                                              'with '
                                                                                              'centroid '
                                                                                              'warp)',
                                                                              'chebyshev_warped': 'basix.LagrangeVariant.chebyshev_warped '
                                                                                                  '(Chebyshev '
                                                                                                  'points) '
                                                                                                  '— '
                                                                                                  'DISCONTINUOUS '
                                                                                                  'ONLY: '
                                                                                                  'on '
                                                                                                  'a '
                                                                                                  'continuous '
                                                                                                  'Lagrange '
                                                                                                  'element '
                                                                                                  'basix '
                                                                                                  '0.10 '
                                                                                                  'raises '
                                                                                                  'RuntimeError '
                                                                                                  "'This "
                                                                                                  'variant '
                                                                                                  'of '
                                                                                                  'Lagrange '
                                                                                                  'is '
                                                                                                  'only '
                                                                                                  'supported '
                                                                                                  'for '
                                                                                                  'discontinuous '
                                                                                                  "elements'. "
                                                                                                  'Use '
                                                                                                  'it '
                                                                                                  'on '
                                                                                                  "'DG'. "
                                                                                                  '(Verified '
                                                                                                  '2026-08-03.)',
                                                                              'chebyshev_isaac': 'basix.LagrangeVariant.chebyshev_isaac '
                                                                                                 '— '
                                                                                                 'discontinuous '
                                                                                                 'only '
                                                                                                 '(same '
                                                                                                 'RuntimeError '
                                                                                                 'on '
                                                                                                 'continuous '
                                                                                                 'Lagrange)',
                                                                              'chebyshev_centroid': 'basix.LagrangeVariant.chebyshev_centroid '
                                                                                                    '— '
                                                                                                    'discontinuous '
                                                                                                    'only '
                                                                                                    '(same '
                                                                                                    'RuntimeError '
                                                                                                    'on '
                                                                                                    'continuous '
                                                                                                    'Lagrange)'},
                                                                 'notes': 'Use '
                                                                          'gll_warped '
                                                                          'for degree '
                                                                          '>= 5 to '
                                                                          'avoid Runge '
                                                                          'phenomenon. '
                                                                          'DG variant: '
                                                                          "'DG' or "
                                                                          'basix.ElementFamily.P '
                                                                          'with '
                                                                          'discontinuous=True.'},
                                                'DG (Discontinuous Lagrange)': {'basix_name': 'basix.ElementFamily.P '
                                                                                              '(with '
                                                                                              'discontinuous=True)',
                                                                                'ufl_name': "'DG' "
                                                                                            'or '
                                                                                            "'Discontinuous "
                                                                                            "Lagrange'",
                                                                                'continuity': 'Discontinuous '
                                                                                              '(no '
                                                                                              'inter-element '
                                                                                              'continuity)',
                                                                                'orders': '0, '
                                                                                          '1, '
                                                                                          '2, '
                                                                                          '... '
                                                                                          '(arbitrary '
                                                                                          'order, '
                                                                                          'DG0 '
                                                                                          '= '
                                                                                          'piecewise '
                                                                                          'constant)',
                                                                                'api': "basix.ufl.element('DG', "
                                                                                       'cell, '
                                                                                       'degree)',
                                                                                'use_cases': 'Advection-dominated '
                                                                                             'problems, '
                                                                                             'conservation '
                                                                                             'laws, '
                                                                                             'DG '
                                                                                             'methods, '
                                                                                             'interior '
                                                                                             'penalty'},
                                                'RT (Raviart-Thomas)': {'basix_name': 'basix.ElementFamily.RT',
                                                                        'ufl_name': "'RT' "
                                                                                    'or '
                                                                                    "'Raviart-Thomas'",
                                                                        'continuity': 'H(div) '
                                                                                      '— '
                                                                                      'normal '
                                                                                      'component '
                                                                                      'continuous '
                                                                                      'across '
                                                                                      'facets',
                                                                        'orders': '1, '
                                                                                  '2, '
                                                                                  '3, '
                                                                                  '...',
                                                                        'cell_types': 'triangle, '
                                                                                      'quadrilateral, '
                                                                                      'tetrahedron, '
                                                                                      'hexahedron',
                                                                        'api': "basix.ufl.element('RT', "
                                                                               'cell, '
                                                                               'degree)',
                                                                        'use_cases': 'Mixed '
                                                                                     'Poisson '
                                                                                     '(Darcy '
                                                                                     'flow), '
                                                                                     'flux-conservative '
                                                                                     'methods',
                                                                        'notes': 'Pair '
                                                                                 'with '
                                                                                 'DG(k-1) '
                                                                                 'for '
                                                                                 'stable '
                                                                                 'mixed '
                                                                                 'Poisson. '
                                                                                 'Normal '
                                                                                 'component '
                                                                                 'preserved '
                                                                                 'by '
                                                                                 'contravariant '
                                                                                 'Piola '
                                                                                 'map.'},
                                                'BDM (Brezzi-Douglas-Marini)': {'basix_name': 'basix.ElementFamily.BDM',
                                                                                'ufl_name': "'BDM' "
                                                                                            'or '
                                                                                            "'Brezzi-Douglas-Marini'",
                                                                                'continuity': 'H(div) '
                                                                                              '— '
                                                                                              'normal '
                                                                                              'component '
                                                                                              'continuous',
                                                                                'orders': '1, '
                                                                                          '2, '
                                                                                          '3, '
                                                                                          '...',
                                                                                'cell_types': 'triangle, '
                                                                                              'quadrilateral, '
                                                                                              'tetrahedron, '
                                                                                              'hexahedron',
                                                                                'api': "basix.ufl.element('BDM', "
                                                                                       'cell, '
                                                                                       'degree)',
                                                                                'notes': 'Full '
                                                                                         'polynomial '
                                                                                         'space '
                                                                                         'on '
                                                                                         'each '
                                                                                         'cell '
                                                                                         '(more '
                                                                                         'DOFs '
                                                                                         'than '
                                                                                         'RT '
                                                                                         'but '
                                                                                         'better '
                                                                                         'approximation).'},
                                                'N1E (Nedelec 1st kind)': {'basix_name': 'basix.ElementFamily.N1E',
                                                                           'ufl_name': "'N1curl' "
                                                                                       'or '
                                                                                       "'Nedelec "
                                                                                       '1st '
                                                                                       'kind '
                                                                                       "H(curl)'",
                                                                           'continuity': 'H(curl) '
                                                                                         '— '
                                                                                         'tangential '
                                                                                         'component '
                                                                                         'continuous '
                                                                                         'across '
                                                                                         'facets',
                                                                           'orders': '1, '
                                                                                     '2, '
                                                                                     '3, '
                                                                                     '...',
                                                                           'cell_types': 'triangle, '
                                                                                         'quadrilateral, '
                                                                                         'tetrahedron, '
                                                                                         'hexahedron',
                                                                           'api': "basix.ufl.element('N1curl', "
                                                                                  'cell, '
                                                                                  'degree)',
                                                                           'use_cases': 'Maxwell '
                                                                                        'equations, '
                                                                                        'electromagnetic '
                                                                                        'wave '
                                                                                        'propagation, '
                                                                                        'curl-curl '
                                                                                        'problems',
                                                                           'notes': 'Tangential '
                                                                                    'component '
                                                                                    'preserved '
                                                                                    'by '
                                                                                    'covariant '
                                                                                    'Piola '
                                                                                    'map. '
                                                                                    'Essential '
                                                                                    'for '
                                                                                    'electromagnetics.'},
                                                'N2E (Nedelec 2nd kind)': {'basix_name': 'basix.ElementFamily.N2E',
                                                                           'ufl_name': "'N2curl' "
                                                                                       'or '
                                                                                       "'Nedelec "
                                                                                       '2nd '
                                                                                       'kind '
                                                                                       "H(curl)'",
                                                                           'continuity': 'H(curl)',
                                                                           'orders': '1, '
                                                                                     '2, '
                                                                                     '...',
                                                                           'cell_types': 'triangle, '
                                                                                         'quadrilateral, '
                                                                                         'tetrahedron, '
                                                                                         'hexahedron',
                                                                           'api': "basix.ufl.element('N2curl', "
                                                                                  'cell, '
                                                                                  'degree)',
                                                                           'notes': 'Full '
                                                                                    'polynomial '
                                                                                    'space '
                                                                                    '(more '
                                                                                    'DOFs '
                                                                                    'than '
                                                                                    'N1E, '
                                                                                    'better '
                                                                                    'approximation).'},
                                                'CR (Crouzeix-Raviart)': {'basix_name': 'basix.ElementFamily.CR',
                                                                          'ufl_name': "'CR' "
                                                                                      'or '
                                                                                      "'Crouzeix-Raviart'",
                                                                          'continuity': 'Nonconforming '
                                                                                        '— '
                                                                                        'continuous '
                                                                                        'at '
                                                                                        'facet '
                                                                                        'midpoints '
                                                                                        'only',
                                                                          'orders': '1 '
                                                                                    'only '
                                                                                    '(degree '
                                                                                    '2 '
                                                                                    'raises '
                                                                                    'RuntimeError '
                                                                                    "'Degree "
                                                                                    'must '
                                                                                    'be '
                                                                                    '1 '
                                                                                    'for '
                                                                                    "Crouzeix-Raviart')",
                                                                          'cell_types': 'triangle, '
                                                                                        'tetrahedron '
                                                                                        'ONLY '
                                                                                        '— '
                                                                                        'quadrilateral/hexahedron '
                                                                                        'raise '
                                                                                        'ValueError '
                                                                                        "'Unknown "
                                                                                        'element '
                                                                                        'family: '
                                                                                        'CR '
                                                                                        'with '
                                                                                        'cell '
                                                                                        'type '
                                                                                        "quadrilateral' "
                                                                                        '(verified '
                                                                                        '2026-08-03)',
                                                                          'api': "basix.ufl.element('CR', "
                                                                                 'cell, '
                                                                                 '1)',
                                                                          'use_cases': 'Stokes '
                                                                                       '(CR/DG0 '
                                                                                       'pair '
                                                                                       'is '
                                                                                       'inf-sup '
                                                                                       'stable), '
                                                                                       'nonconforming '
                                                                                       'methods'},
                                                'bubble': {'basix_name': 'basix.ElementFamily.bubble',
                                                           'ufl_name': "'Bubble'",
                                                           'continuity': 'Zero on '
                                                                         'element '
                                                                         'boundaries '
                                                                         '(vanishes on '
                                                                         'facets)',
                                                           'orders': 'Minimum degree '
                                                                     'per cell type: 2 '
                                                                     'for interval, 3 '
                                                                     'for triangle, 4 '
                                                                     'for tet, 2 for '
                                                                     'quad, 2 for hex '
                                                                     '(below that: '
                                                                     'RuntimeError '
                                                                     "'Bubble element "
                                                                     'on a <cell> must '
                                                                     'have degree at '
                                                                     "least N'). "
                                                                     'Verified '
                                                                     '2026-08-03; the '
                                                                     'interval minimum '
                                                                     'was added '
                                                                     '2026-08-03 after '
                                                                     'a full degree '
                                                                     'sweep — '
                                                                     'cell_types '
                                                                     'listed interval '
                                                                     'but its minimum '
                                                                     'was missing.',
                                                           'cell_types': 'interval, '
                                                                         'triangle, '
                                                                         'quadrilateral, '
                                                                         'tetrahedron, '
                                                                         'hexahedron',
                                                           'api': "basix.ufl.element('Bubble', "
                                                                  'cell, degree)',
                                                           'use_cases': 'MINI element '
                                                                        'for Stokes '
                                                                        '(Lagrange + '
                                                                        'Bubble '
                                                                        'enrichment), '
                                                                        'stabilization'},
                                                'Regge': {'basix_name': 'basix.ElementFamily.Regge',
                                                          'ufl_name': "'Regge'",
                                                          'continuity': 'Tangent-tangent '
                                                                        'component '
                                                                        'continuous',
                                                          'orders': '0, 1, 2, ...',
                                                          'cell_types': 'triangle, '
                                                                        'tetrahedron',
                                                          'api': "basix.ufl.element('Regge', "
                                                                 'cell, degree)',
                                                          'use_cases': 'Linearized '
                                                                       'general '
                                                                       'relativity, '
                                                                       'metric '
                                                                       'tensors, '
                                                                       'elasticity '
                                                                       'complexes'},
                                                'HHJ (Hellan-Herrmann-Johnson)': {'basix_name': 'basix.ElementFamily.HHJ',
                                                                                  'ufl_name': "'HHJ'",
                                                                                  'continuity': 'Normal-normal '
                                                                                                'component '
                                                                                                'continuous',
                                                                                  'orders': '0, '
                                                                                            '1, '
                                                                                            '2, '
                                                                                            '...',
                                                                                  'cell_types': 'triangle, '
                                                                                                'tetrahedron '
                                                                                                '(verified '
                                                                                                '2026-08-03)',
                                                                                  'api': "basix.ufl.element('HHJ', "
                                                                                         'cell, '
                                                                                         'degree)',
                                                                                  'use_cases': 'Kirchhoff '
                                                                                               'plates, '
                                                                                               'biharmonic '
                                                                                               'equation '
                                                                                               '(symmetric '
                                                                                               'tensor '
                                                                                               'field '
                                                                                               'for '
                                                                                               'moments)'},
                                                'serendipity': {'basix_name': 'basix.ElementFamily.serendipity',
                                                                'ufl_name': "'S' or "
                                                                            "'serendipity'",
                                                                'continuity': 'C0',
                                                                'orders': '1, 2, 3, '
                                                                          '...',
                                                                'cell_types': 'quadrilateral, '
                                                                              'hexahedron '
                                                                              '(interval '
                                                                              'also '
                                                                              'builds: '
                                                                              'dim 2 '
                                                                              'at '
                                                                              'degree '
                                                                              '1, 3 at '
                                                                              'degree '
                                                                              '2). '
                                                                              'Simplices/prism/pyramid '
                                                                              'raise '
                                                                              'ValueError '
                                                                              "'Unknown "
                                                                              'element '
                                                                              'family: '
                                                                              'serendipity '
                                                                              'with '
                                                                              'cell '
                                                                              'type '
                                                                              "triangle'. "
                                                                              'Cell '
                                                                              'types '
                                                                              'swept '
                                                                              'by '
                                                                              'execution '
                                                                              '2026-08-03.',
                                                                'api': "basix.ufl.element('S', "
                                                                       'cell, degree)',
                                                                'notes': 'Fewer DOFs '
                                                                         'than '
                                                                         'tensor-product '
                                                                         'Lagrange on '
                                                                         'quads/hexes. '
                                                                         'S2 has no '
                                                                         'interior '
                                                                         'node on '
                                                                         'quad.'},
                                                'DPC (Discontinuous Piecewise Complete)': {'basix_name': 'basix.ElementFamily.DPC',
                                                                                           'ufl_name': "'DPC'",
                                                                                           'continuity': 'Discontinuous',
                                                                                           'orders': '0, '
                                                                                                     '1, '
                                                                                                     '2, '
                                                                                                     '...',
                                                                                           'cell_types': 'quadrilateral, '
                                                                                                         'hexahedron '
                                                                                                         '(interval '
                                                                                                         'also '
                                                                                                         'builds: '
                                                                                                         'dim '
                                                                                                         '1 '
                                                                                                         'at '
                                                                                                         'degree '
                                                                                                         '0, '
                                                                                                         '2 '
                                                                                                         'at '
                                                                                                         'degree '
                                                                                                         '1). '
                                                                                                         'Simplices/prism/pyramid '
                                                                                                         'raise '
                                                                                                         'ValueError '
                                                                                                         "'Unknown "
                                                                                                         'element '
                                                                                                         'family: '
                                                                                                         'DPC '
                                                                                                         'with '
                                                                                                         'cell '
                                                                                                         'type '
                                                                                                         "triangle'. "
                                                                                                         'Cell '
                                                                                                         'types '
                                                                                                         'swept '
                                                                                                         'by '
                                                                                                         'execution '
                                                                                                         '2026-08-03.',
                                                                                           'api': "basix.ufl.element('DPC', "
                                                                                                  'cell, '
                                                                                                  'degree)',
                                                                                           'notes': 'Complete '
                                                                                                    'polynomial '
                                                                                                    'on '
                                                                                                    'quads/hexes '
                                                                                                    '(not '
                                                                                                    'tensor-product). '
                                                                                                    'Used '
                                                                                                    'in '
                                                                                                    'compatible '
                                                                                                    'DG '
                                                                                                    'schemes.'},
                                                'Hermite': {'basix_name': 'basix.ElementFamily.Hermite',
                                                            'ufl_name': "'Hermite'",
                                                            'continuity': 'C1 (value '
                                                                          'and '
                                                                          'gradient '
                                                                          'continuous '
                                                                          'at '
                                                                          'vertices)',
                                                            'orders': '3',
                                                            'cell_types': 'interval, '
                                                                          'triangle, '
                                                                          'tetrahedron',
                                                            'api': 'basix.ufl.element(basix.ElementFamily.Hermite, '
                                                                   'basix.CellType.triangle, '
                                                                   '3) — the ENUM is '
                                                                   'required; the '
                                                                   "string 'Hermite' "
                                                                   'raises ValueError '
                                                                   "'Unknown element "
                                                                   'family: Hermite '
                                                                   'with cell type '
                                                                   "triangle' in basix "
                                                                   '0.10 (verified '
                                                                   '2026-08-03)',
                                                            'use_cases': 'Beam/plate '
                                                                         'problems '
                                                                         'requiring C1 '
                                                                         'continuity, '
                                                                         'Kirchhoff '
                                                                         'theory'},
                                                'iso (isoparametric/macro)': {'basix_name': 'basix.ElementFamily.iso',
                                                                              'ufl_name': "'iso'",
                                                                              'continuity': 'C0 '
                                                                                            '(piecewise '
                                                                                            'on '
                                                                                            'sub-cells)',
                                                                              'orders': '1 and 2 without a '
                                                                                        'variant (tetrahedron: 1 '
                                                                                        'only); degree > 2 raises '
                                                                                        'RuntimeError '
                                                                                        "'Lagrange "
                                                                                        'elements '
                                                                                        'of '
                                                                                        'degree '
                                                                                        '> '
                                                                                        '2 '
                                                                                        'need '
                                                                                        'to '
                                                                                        'be '
                                                                                        'given '
                                                                                        'a '
                                                                                        "variant.' "
                                                                                        'unless '
                                                                                        'a '
                                                                                        'LagrangeVariant '
                                                                                        'is '
                                                                                        'passed, and even then '
                                                                                        'builds only on interval, '
                                                                                        'quadrilateral and '
                                                                                        'hexahedron (triangle '
                                                                                        "raises 'Only degree 0 to "
                                                                                        '2 macro polysets are '
                                                                                        'currently implemented on '
                                                                                        "a triangle.'). "
                                                                                        'Verified '
                                                                                        'on basix 0.10.0.',
                                                                              'cell_types': 'interval, '
                                                                                            'triangle, '
                                                                                            'quadrilateral, '
                                                                                            'hexahedron, '
                                                                                            'tetrahedron '
                                                                                            '(tetrahedron '
                                                                                            'at degree 1 '
                                                                                            'only: degree 2 '
                                                                                            'raises '
                                                                                            'RuntimeError '
                                                                                            "'Only "
                                                                                            'degree '
                                                                                            '0 '
                                                                                            'and '
                                                                                            '1 '
                                                                                            'macro '
                                                                                            'polysets '
                                                                                            'are '
                                                                                            'currently '
                                                                                            'implemented '
                                                                                            'on '
                                                                                            'a '
                                                                                            "tetrahedron.'). "
                                                                                            'Verified '
                                                                                            'on basix 0.10.0.',
                                                                              'api': "basix.ufl.element('iso', "
                                                                                     'cell, '
                                                                                     'degree)',
                                                                              'notes': 'Macro '
                                                                                       'element: '
                                                                                       'cell '
                                                                                       'is '
                                                                                       'split '
                                                                                       'into '
                                                                                       'sub-cells, '
                                                                                       'lower-order '
                                                                                       'polynomial '
                                                                                       'on '
                                                                                       'each. '
                                                                                       'Fewer '
                                                                                       'DOFs '
                                                                                       'than '
                                                                                       'standard '
                                                                                       'high-order.'}},
                     'compound_elements': {'blocked_element': {'api': 'basix.ufl.blocked_element(sub_element, '
                                                                      'shape=(gdim,))',
                                                               'use': 'Vector/tensor '
                                                                      'function spaces '
                                                                      'from scalar '
                                                                      'elements. E.g., '
                                                                      'vector Lagrange '
                                                                      'for elasticity.',
                                                               'example': 'Ve = '
                                                                          "basix.ufl.element('Lagrange', "
                                                                          'cell, 2); '
                                                                          'basix.ufl.blocked_element(Ve, '
                                                                          'shape=(3,))'},
                                           'mixed_element': {'api': 'basix.ufl.mixed_element([el1, '
                                                                    'el2, ...])',
                                                             'use': 'Combine different '
                                                                    'elements for '
                                                                    'mixed '
                                                                    'formulations '
                                                                    '(Taylor-Hood, '
                                                                    'Stokes, etc.)',
                                                             'example': 'P2 = '
                                                                        "basix.ufl.element('Lagrange', "
                                                                        'cell, 2, '
                                                                        'shape=(gdim,)); '
                                                                        'P1 = '
                                                                        "basix.ufl.element('Lagrange', "
                                                                        'cell, 1); ME '
                                                                        '= '
                                                                        'basix.ufl.mixed_element([P2, '
                                                                        'P1])'},
                                           'enriched_element': {'api': 'basix.ufl.enriched_element([el1, '
                                                                       'el2])',
                                                                'use': 'Combine '
                                                                       'elements to '
                                                                       'enrich '
                                                                       'approximation '
                                                                       'space. Used '
                                                                       'for MINI '
                                                                       'element.',
                                                                'example': 'P1 = '
                                                                           "basix.ufl.element('Lagrange', "
                                                                           'cell, 1, '
                                                                           'shape=(gdim,)); '
                                                                           'B = '
                                                                           "basix.ufl.element('Bubble', "
                                                                           'cell, 3, '
                                                                           'shape=(gdim,)); '
                                                                           'MINI = '
                                                                           'basix.ufl.enriched_element([P1, '
                                                                           'B])'}},
                     'cell_types': {'interval': '1D line segment',
                                    'triangle': '2D simplex (3 vertices)',
                                    'quadrilateral': '2D quad (4 vertices)',
                                    'tetrahedron': '3D simplex (4 vertices)',
                                    'hexahedron': '3D brick (8 vertices)',
                                    'prism': '3D triangular prism (6 vertices)',
                                    'pyramid': '3D pyramid (5 vertices)'},
                     'pitfalls': ['In dolfinx >= 0.8, use basix.ufl.element() NOT '
                                  'ufl.FiniteElement() — the legacy names are GONE, '
                                  'not merely deprecated: ufl.FiniteElement / '
                                  'ufl.VectorElement / ufl.MixedElement all raise '
                                  'AttributeError "module \'ufl\' has no attribute '
                                  '\'FiniteElement\'" on ufl 2025.2.1 (verified '
                                  '2026-08-03)',
                                  'For vector elements use blocked_element or shape= '
                                  'parameter, NOT VectorElement (removed)',
                                  'For mixed spaces use basix.ufl.mixed_element, NOT '
                                  'ufl.MixedElement (removed)',
                                  'Element variant matters for high order (>= 5): use '
                                  'gll_warped to avoid ill-conditioning. The '
                                  'chebyshev_* variants are DISCONTINUOUS-ONLY — '
                                  'asking for them on continuous Lagrange raises '
                                  "RuntimeError 'This variant of Lagrange is only "
                                  "supported for discontinuous elements' (verified "
                                  '2026-08-03)',
                                  'Not all element families support all cell types — '
                                  'check Basix docs for compatibility. Measured on '
                                  'basix 0.10: CR and Regge are simplex-only; iso is '
                                  'only implemented at degree 1 on tetrahedra; Lagrange '
                                  'on pyramid stops at degree 2 with the default variant '
                                  '(degree >= 3 needs lagrange_variant=equispaced)',
                                  'Bubble element minimum degree depends on cell type: '
                                  '3 for triangle, 4 for tet, 2 for quad, 2 for hex',
                                  'Serendipity and DPC elements are the tensor-product '
                                  'families: quadrilateral and hexahedron (plus the '
                                  'degenerate interval case, which also builds). Both '
                                  "raise ValueError 'Unknown element family: <fam> "
                                  "with cell type triangle' on simplices, prisms and "
                                  'pyramids. (Cell-type support swept by execution on '
                                  'basix 0.10.0, 2026-08-03 — the earlier wording '
                                  "'only available on quads/hexes' omitted interval.)",
                                  '[API] Some families are reachable ONLY through the '
                                  'basix.ElementFamily ENUM, not the family string. '
                                  'Hermite is the concrete case. Signal: '
                                  "basix.ufl.element('Hermite', 'triangle', 3) raises "
                                  "ValueError 'Unknown element family: Hermite with "
                                  "cell type triangle', while "
                                  'basix.ufl.element(basix.ElementFamily.Hermite, '
                                  'basix.CellType.triangle, 3) builds the 10-dof C1 '
                                  'element. (Verified empirically 2026-08-03, basix '
                                  '0.10.0.)',
                                  "[API] 'CG' still resolves. Signal: "
                                  "basix.ufl.element('CG', 'triangle', 1) returns a "
                                  'valid element and only emits a DeprecationWarning ( '
                                  '\'"CG" element name is deprecated. Consider using '
                                  '"Lagrange" or "P" instead\') — it is NOT rejected. '
                                  "'P' also resolves. The name that genuinely raises "
                                  "ValueError 'Unknown element family: P1 with cell "
                                  "type triangle' is the old DOLFIN degree-suffixed "
                                  "form 'P1'. (Verified empirically 2026-08-03 — "
                                  "corrects an older catalog claim that 'CG' "
                                  'raises.)']},
 'mesh_catalog': {'description': 'Complete mesh creation, import, and manipulation '
                                 'capabilities in DOLFINx.',
                  'built_in_meshes': {'create_unit_square': {'api': 'dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, '
                                                                    'nx, ny, '
                                                                    'cell_type=CellType.triangle)',
                                                             'geometry': '[0,1] x '
                                                                         '[0,1]',
                                                             'cell_types': 'CellType.triangle '
                                                                           '(default), '
                                                                           'CellType.quadrilateral'},
                                      'create_unit_cube': {'api': 'dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, '
                                                                  'nx, ny, nz, '
                                                                  'cell_type=CellType.tetrahedron)',
                                                           'geometry': '[0,1]^3',
                                                           'cell_types': 'CellType.tetrahedron '
                                                                         '(default), '
                                                                         'CellType.hexahedron'},
                                      'create_rectangle': {'api': 'dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, '
                                                                  '[p0, p1], [nx, ny], '
                                                                  'cell_type=...)',
                                                           'geometry': 'Arbitrary '
                                                                       'rectangle [p0, '
                                                                       'p1]',
                                                           'cell_types': 'CellType.triangle, '
                                                                         'CellType.quadrilateral'},
                                      'create_box': {'api': 'dolfinx.mesh.create_box(MPI.COMM_WORLD, '
                                                            '[p0, p1], [nx, ny, nz], '
                                                            'cell_type=...)',
                                                     'geometry': 'Arbitrary box [p0, '
                                                                 'p1]',
                                                     'cell_types': 'CellType.tetrahedron, '
                                                                   'CellType.hexahedron'},
                                      'create_unit_interval': {'api': 'dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, '
                                                                      'n)',
                                                               'geometry': '[0,1] '
                                                                           'interval'},
                                      'create_interval': {'api': 'dolfinx.mesh.create_interval(MPI.COMM_WORLD, '
                                                                 'n, [a, b])',
                                                          'geometry': '[a,b] '
                                                                      'interval'}},
                  'gmsh_integration': {'api_0_9': 'dolfinx.io.gmshio.model_to_mesh(gmsh.model, '
                                                  'MPI.COMM_WORLD, rank=0)',
                                       'api_0_10': 'dolfinx.io.gmsh.model_to_mesh(gmsh.model, '
                                                   'MPI.COMM_WORLD, rank=0) — returns '
                                                   'a MeshData object with '
                                                   '.mesh/.cell_tags/.facet_tags/.ridge_tags/.peak_tags/.physical_groups '
                                                   '(verified 2026-08-03)',
                                       'read_from_msh': "dolfinx.io.gmsh.read_from_msh('file.msh', "
                                                        'MPI.COMM_WORLD, rank=0) — '
                                                        'NOTE the module is `gmsh`, '
                                                        'not `gmshio`: `import '
                                                        'dolfinx.io.gmshio` raises '
                                                        'ModuleNotFoundError: No '
                                                        'module named '
                                                        "'dolfinx.io.gmshio' on 0.10 "
                                                        '(verified 2026-08-03)',
                                       'workflow': '1. Build geometry with gmsh Python '
                                                   'API, 2. Mesh with '
                                                   'gmsh.model.mesh.generate(dim), 3. '
                                                   'Convert with model_to_mesh()',
                                       'returns': 'MeshData with mesh, cell_tags '
                                                  '(codim 0), facet_tags (codim 1), '
                                                  'ridge/peak tags, physical group '
                                                  'lookup',
                                       'notes': 'Gmsh model processed on rank 0, '
                                                'DOLFINx mesh distributed across all '
                                                'ranks automatically.'},
                  'xdmf_import': {'read_mesh': 'with '
                                               'dolfinx.io.XDMFFile(MPI.COMM_WORLD, '
                                               "'mesh.xdmf', 'r') as f: mesh = "
                                               'f.read_mesh()',
                                  'read_tags': "f.read_meshtags(mesh, name='facets')",
                                  'notes': 'Good for pre-generated meshes. Geometry '
                                           'order <= 2 supported.'},
                  'vtkhdf_import': {'api': "dolfinx.io.vtkhdf.read_mesh(MPI.COMM_WORLD, "
                                           "'mesh.vtkhdf') — new in 0.10 (communicator "
                                           'first, then filename)',
                                    'notes': "Kitware's future-proof format. "
                                             'Transition from XDMF has started. '
                                             'Writing is present too in 0.10: '
                                             'dolfinx.io.vtkhdf exposes write_mesh, '
                                             'write_point_data, write_cell_data '
                                             '(verified 2026-08-03).'},
                  'mesh_refinement': {'PREREQUISITE': 'mesh.topology.create_entities(1) '
                                                      'MUST be called before either '
                                                      'refine entry point on a freshly '
                                                      'built mesh, otherwise '
                                                      "RuntimeError 'Missing entities "
                                                      'of dimension 1, need to call '
                                                      "create_entities(1)' "
                                                      "(uniform_refine) / 'Missing "
                                                      'IndexMap in Topology. Maybe you '
                                                      "need to create_entities(1).' "
                                                      '(refine). Verified 2026-08-03.',
                                      'uniform_refine': 'mesh.topology.create_entities(1); '
                                                        'm2 = '
                                                        'dolfinx.mesh.uniform_refine(mesh) '
                                                        '— refines all cells '
                                                        'uniformly, returns a Mesh (32 '
                                                        '-> 128 cells on a 4x4 unit '
                                                        'square)',
                                      'refine': 'mesh.topology.create_entities(1); m2, '
                                                'parent_cells, parent_facets = '
                                                'dolfinx.mesh.refine(mesh, edges=None) '
                                                '— returns a 3-TUPLE, not a bare Mesh. '
                                                'Measured element types on 0.10.0 '
                                                '(2026-08-03): (Mesh, ndarray, '
                                                'NoneType) for the default call — the '
                                                'third slot is None unless facet '
                                                'parents are requested, so do not '
                                                'assume it is an array',
                                      'partitioner': 'Optional custom partitioner for '
                                                     'distributing refined mesh'},
                  'mesh_operations': {'create_submesh': 'dolfinx.mesh.create_submesh(mesh, '
                                                        'dim, entities) — extract '
                                                        'subdomain mesh',
                                      'meshtags': 'dolfinx.mesh.meshtags(mesh, dim, '
                                                  'entities, values) — tag entities '
                                                  'with integer markers',
                                      'locate_entities': 'dolfinx.mesh.locate_entities(mesh, '
                                                         'dim, marker_fn) — find '
                                                         'entities satisfying '
                                                         'geometric condition',
                                      'locate_entities_boundary': 'dolfinx.mesh.locate_entities_boundary(mesh, '
                                                                  'dim, marker_fn) — '
                                                                  'boundary entities '
                                                                  'only',
                                      'exterior_facet_indices': 'dolfinx.mesh.exterior_facet_indices(mesh.topology) '
                                                                '— all exterior '
                                                                'facets'},
                  'pitfalls': ['MUST pass MPI.COMM_WORLD (or appropriate communicator) '
                               'to all mesh creation functions',
                               'Gmsh model_to_mesh: module renamed from gmshio to gmsh '
                               'in dolfinx 0.10',
                               'For parallel: gmsh model built on rank 0 only (if '
                               'gmsh.isInitialized())',
                               'Topology connectivity must be created before use: '
                               'mesh.topology.create_connectivity(dim1, dim2). '
                               'Measured scope on 0.10: locate_entities_boundary and '
                               'ds/dS assembly build facet->cell connectivity '
                               'themselves and work without the call. '
                               'locate_dofs_topological does NOT build it: it works '
                               'only if the connectivity already exists (e.g. an '
                               'earlier locate_entities_boundary call or gmsh '
                               'model_to_mesh created it), otherwise it raises '
                               "RuntimeError 'Entity-to-cell connectivity has not "
                               "been computed. Missing dims 1->2' (2-D mesh). "
                               'dolfinx.mesh.exterior_facet_indices(mesh.topology) '
                               "also needs it and raises RuntimeError 'Facet to cell "
                               "connectivity has not been computed.'",
                               'Both refine entry points need '
                               'mesh.topology.create_entities(1) first — see '
                               'mesh_refinement.PREREQUISITE',
                               'Branching meshes (T-joints, 3+ cells per facet) '
                               'supported since 0.10',
                               'create_unit_square default is triangles — use '
                               'CellType.quadrilateral explicitly for quads']},
 'solver_catalog': {'description': 'Complete PETSc/SLEPc solver and preconditioner '
                                   'catalog for DOLFINx.',
                    'linear_solvers': {'high_level_api': {'LinearProblem': {'api': 'dolfinx.fem.petsc.LinearProblem(a, '
                                                                                   'L, '
                                                                                   "petsc_options_prefix='myprob_', "
                                                                                   'bcs=bcs, '
                                                                                   'petsc_options={...})',
                                                                            'usage': 'Simplest '
                                                                                     'interface: '
                                                                                     'problem.solve() '
                                                                                     'returns '
                                                                                     'Function. '
                                                                                     'ALL '
                                                                                     'non-form '
                                                                                     'args '
                                                                                     'are '
                                                                                     'keyword-only '
                                                                                     'in '
                                                                                     'dolfinx '
                                                                                     '0.10; '
                                                                                     'petsc_options_prefix '
                                                                                     'is '
                                                                                     'REQUIRED '
                                                                                     '— '
                                                                                     'omitting '
                                                                                     'it '
                                                                                     'raises '
                                                                                     'TypeError '
                                                                                     "'missing "
                                                                                     '1 '
                                                                                     'required '
                                                                                     'keyword-only '
                                                                                     'argument: '
                                                                                     "petsc_options_prefix'.",
                                                                            'returns': 'problem.solve() '
                                                                                       'returns '
                                                                                       'a '
                                                                                       'dolfinx.fem.Function '
                                                                                       '(NOT '
                                                                                       'a '
                                                                                       'tuple) '
                                                                                       'on '
                                                                                       '0.10; '
                                                                                       'the '
                                                                                       'KSP '
                                                                                       'is '
                                                                                       'reachable '
                                                                                       'as '
                                                                                       'problem.solver '
                                                                                       'for '
                                                                                       'getConvergedReason() '
                                                                                       '/ '
                                                                                       'getIterationNumber(). '
                                                                                       'Same '
                                                                                       'for '
                                                                                       'NonlinearProblem.solve(). '
                                                                                       '(Verified '
                                                                                       '2026-08-03.)',
                                                                            '0_10_note': 'Now '
                                                                                         'supports '
                                                                                         'blocked '
                                                                                         'problems '
                                                                                         'via '
                                                                                         "kind='mpi' "
                                                                                         'or '
                                                                                         "kind='nest'"},
                                                          'DIAGNOSTIC_TRAP': '[API] '
                                                                             'When a '
                                                                             'LinearProblem '
                                                                             '/ '
                                                                             'NonlinearProblem '
                                                                             'CONSTRUCTOR '
                                                                             'raises, '
                                                                             'dolfinx '
                                                                             '0.10 '
                                                                             'immediately '
                                                                             'emits a '
                                                                             'SECOND, '
                                                                             'misleading '
                                                                             'traceback '
                                                                             'from '
                                                                             '__del__ '
                                                                             'on the '
                                                                             'half-built '
                                                                             'object: '
                                                                             '"Exception '
                                                                             'ignored '
                                                                             'in: '
                                                                             '<function '
                                                                             'LinearProblem.__del__ '
                                                                             '...> '
                                                                             'AttributeError: '
                                                                             "'LinearProblem' "
                                                                             'object '
                                                                             'has no '
                                                                             'attribute '
                                                                             '\'_solver\'" '
                                                                             '(and '
                                                                             "'_snes' "
                                                                             'for '
                                                                             'NonlinearProblem). '
                                                                             'The REAL '
                                                                             'error is '
                                                                             'the '
                                                                             'first '
                                                                             'one — '
                                                                             'e.g. '
                                                                             'TypeError '
                                                                             'missing '
                                                                             'petsc_options_prefix. '
                                                                             'Do not '
                                                                             'chase '
                                                                             'the '
                                                                             '_solver '
                                                                             '/ _snes '
                                                                             'AttributeError; '
                                                                             'it is '
                                                                             'garbage-collection '
                                                                             'noise '
                                                                             'and '
                                                                             'appears '
                                                                             'on '
                                                                             'stderr '
                                                                             'even '
                                                                             'when the '
                                                                             'real '
                                                                             'exception '
                                                                             'was '
                                                                             'caught '
                                                                             'and '
                                                                             'handled. '
                                                                             '(Verified '
                                                                             'empirically '
                                                                             '2026-08-03, '
                                                                             'dolfinx '
                                                                             '0.10.0.)'},
                                       'direct_solvers': {'mumps': {'options': {'ksp_type': 'preonly',
                                                                                'pc_type': 'lu',
                                                                                'pc_factor_mat_solver_type': 'mumps'},
                                                                    'use': 'General '
                                                                           'sparse, '
                                                                           'parallel, '
                                                                           'recommended '
                                                                           'default '
                                                                           'direct '
                                                                           'solver'},
                                                          'superlu_dist': {'options': {'ksp_type': 'preonly',
                                                                                       'pc_type': 'lu',
                                                                                       'pc_factor_mat_solver_type': 'superlu_dist'},
                                                                           'use': 'Alternative '
                                                                                  'parallel '
                                                                                  'direct '
                                                                                  'solver'},
                                                          'umfpack': {'options': {'ksp_type': 'preonly',
                                                                                  'pc_type': 'lu',
                                                                                  'pc_factor_mat_solver_type': 'umfpack'},
                                                                      'use': 'Sequential '
                                                                             'only, '
                                                                             'good for '
                                                                             'small '
                                                                             'problems'}},
                                       'iterative_solvers': {'CG': {'options': {'ksp_type': 'cg'},
                                                                    'use': 'Symmetric '
                                                                           'positive '
                                                                           'definite '
                                                                           '(Poisson, '
                                                                           'elasticity, '
                                                                           'heat)',
                                                                    'requires': 'SPD '
                                                                                'matrix '
                                                                                'and '
                                                                                'SPD '
                                                                                'preconditioner'},
                                                             'GMRES': {'options': {'ksp_type': 'gmres'},
                                                                       'use': 'Non-symmetric '
                                                                              'systems '
                                                                              '(advection, '
                                                                              'Navier-Stokes)',
                                                                       'notes': 'Restarted, '
                                                                                'set '
                                                                                'ksp_gmres_restart '
                                                                                'for '
                                                                                'large '
                                                                                'problems'},
                                                             'BiCGStab': {'options': {'ksp_type': 'bcgs'},
                                                                          'use': 'Non-symmetric '
                                                                                 'alternative '
                                                                                 'to '
                                                                                 'GMRES'},
                                                             'MinRes': {'options': {'ksp_type': 'minres'},
                                                                        'use': 'Symmetric '
                                                                               'indefinite '
                                                                               '(saddle-point: '
                                                                               'Stokes, '
                                                                               'mixed '
                                                                               'Poisson)'},
                                                             'Richardson': {'options': {'ksp_type': 'richardson'},
                                                                            'use': 'Simple '
                                                                                   'iteration, '
                                                                                   'often '
                                                                                   'as '
                                                                                   'smoother'}},
                                       'preconditioners': {'ILU': {'options': {'pc_type': 'ilu'},
                                                                   'use': 'General-purpose '
                                                                          'incomplete '
                                                                          'LU '
                                                                          '(sequential)'},
                                                           'ICC': {'options': {'pc_type': 'icc'},
                                                                   'use': 'Incomplete '
                                                                          'Cholesky '
                                                                          'for SPD '
                                                                          'systems '
                                                                          '(sequential)'},
                                                           'Jacobi': {'options': {'pc_type': 'jacobi'},
                                                                      'use': 'Diagonal '
                                                                             'scaling, '
                                                                             'cheap, '
                                                                             'for DG '
                                                                             'mass '
                                                                             'matrices'},
                                                           'SOR': {'options': {'pc_type': 'sor'},
                                                                   'use': 'Successive '
                                                                          'over-relaxation'},
                                                           'GAMG': {'options': {'pc_type': 'gamg'},
                                                                    'use': 'PETSc '
                                                                           'native '
                                                                           'smoothed '
                                                                           'aggregation '
                                                                           'AMG — good '
                                                                           'for '
                                                                           'Poisson, '
                                                                           'elasticity',
                                                                    'notes': 'Provide '
                                                                             'near-nullspace '
                                                                             '(rigid '
                                                                             'body '
                                                                             'modes) '
                                                                             'for '
                                                                             'elasticity'},
                                                           'hypre_boomeramg': {'options': {'pc_type': 'hypre',
                                                                                           'pc_hypre_type': 'boomeramg'},
                                                                               'use': 'Classical '
                                                                                      'AMG '
                                                                                      'via '
                                                                                      'hypre '
                                                                                      '— '
                                                                                      'excellent '
                                                                                      'for '
                                                                                      'Poisson, '
                                                                                      'good '
                                                                                      'for '
                                                                                      'elasticity',
                                                                               'tuning': {'pc_hypre_boomeramg_strong_threshold': '0.25 '
                                                                                                                                 '(2D) '
                                                                                                                                 'or '
                                                                                                                                 '0.5-0.7 '
                                                                                                                                 '(3D)',
                                                                                          'pc_hypre_boomeramg_agg_nl': '2-4 '
                                                                                                                       '(aggressive '
                                                                                                                       'coarsening '
                                                                                                                       'levels)'}},
                                                           'BDDC': {'options': {'pc_type': 'bddc'},
                                                                    'use': 'Balancing '
                                                                           'domain '
                                                                           'decomposition '
                                                                           'by '
                                                                           'constraints '
                                                                           '— scalable '
                                                                           'parallel',
                                                                    'caveat': 'NOT '
                                                                              'usable '
                                                                              'as a '
                                                                              'bare '
                                                                              'option '
                                                                              'dict: '
                                                                              'PCBDDC '
                                                                              'requires '
                                                                              'a '
                                                                              'MATIS-format '
                                                                              'operator, '
                                                                              'and a '
                                                                              'plain '
                                                                              'dolfinx-assembled '
                                                                              'AIJ '
                                                                              'matrix '
                                                                              'makes '
                                                                              'KSPSetUp '
                                                                              'abort '
                                                                              'with '
                                                                              'PETSc '
                                                                              'error '
                                                                              'code 62 '
                                                                              '(verified '
                                                                              '2026-08-03).'},
                                                           'fieldsplit': {'options': {'pc_type': 'fieldsplit'},
                                                                          'use': 'Block '
                                                                                 'preconditioner '
                                                                                 'for '
                                                                                 'saddle-point '
                                                                                 '(Stokes, '
                                                                                 'mixed)',
                                                                          'caveat': 'NOT '
                                                                                    'usable '
                                                                                    'as '
                                                                                    'a '
                                                                                    'bare '
                                                                                    'option '
                                                                                    'dict: '
                                                                                    'the '
                                                                                    'splits '
                                                                                    'must '
                                                                                    'be '
                                                                                    'defined '
                                                                                    '(IS '
                                                                                    'fields '
                                                                                    'via '
                                                                                    'pc.setFieldSplitIS '
                                                                                    '/ '
                                                                                    'a '
                                                                                    'blocked '
                                                                                    'LinearProblem '
                                                                                    'with '
                                                                                    "kind='nest'), "
                                                                                    'otherwise '
                                                                                    'KSPSetUp '
                                                                                    'aborts '
                                                                                    'with '
                                                                                    'PETSc '
                                                                                    'error '
                                                                                    'code '
                                                                                    '77 '
                                                                                    '(verified '
                                                                                    '2026-08-03).'}}},
                    'nonlinear_solvers': {'SNES_via_NonlinearProblem': {'api_0_9': 'problem '
                                                                                   '= '
                                                                                   'NonlinearProblem(F, '
                                                                                   'u, '
                                                                                   'bcs); '
                                                                                   'solver '
                                                                                   '= '
                                                                                   'NewtonSolver(MPI.COMM_WORLD, '
                                                                                   'problem)',
                                                                        'api_0_10': 'problem '
                                                                                    '= '
                                                                                    'dolfinx.fem.petsc.NonlinearProblem(F, '
                                                                                    'u, '
                                                                                    'bcs=bcs, '
                                                                                    "petsc_options_prefix='myprob_', "
                                                                                    'petsc_options={...}); '
                                                                                    'problem.solve()',
                                                                        '0_10_signature_pitfalls': 'ALL '
                                                                                                   'kwargs '
                                                                                                   'are '
                                                                                                   'keyword-only '
                                                                                                   '(after '
                                                                                                   'the '
                                                                                                   '* '
                                                                                                   'in '
                                                                                                   'the '
                                                                                                   'signature). '
                                                                                                   'NonlinearProblem(F, '
                                                                                                   'u, '
                                                                                                   'bcs) '
                                                                                                   'as '
                                                                                                   'positional '
                                                                                                   'fails '
                                                                                                   'with '
                                                                                                   'TypeError '
                                                                                                   "'takes "
                                                                                                   '3 '
                                                                                                   'positional '
                                                                                                   'arguments '
                                                                                                   'but '
                                                                                                   '4 '
                                                                                                   'were '
                                                                                                   "given'. "
                                                                                                   'Omitting '
                                                                                                   'petsc_options_prefix '
                                                                                                   'fails '
                                                                                                   'with '
                                                                                                   'TypeError '
                                                                                                   "'missing "
                                                                                                   '1 '
                                                                                                   'required '
                                                                                                   'keyword-only '
                                                                                                   'argument: '
                                                                                                   "petsc_options_prefix'. "
                                                                                                   '(Empirically '
                                                                                                   'verified '
                                                                                                   '2026-06-01 '
                                                                                                   '— '
                                                                                                   'Tier-2 '
                                                                                                   'fixture '
                                                                                                   'nonlinear_problem_signature_kwargs.)',
                                                                        'note': 'dolfinx.nls.petsc.NewtonSolver '
                                                                                'deprecated '
                                                                                'in '
                                                                                '0.10 '
                                                                                'in '
                                                                                'favor '
                                                                                'of '
                                                                                'NonlinearProblem '
                                                                                'wrapping '
                                                                                'SNES '
                                                                                'directly'},
                                          'snes_types': {'newtonls': {'options': {'snes_type': 'newtonls'},
                                                                      'description': 'Newton '
                                                                                     'with '
                                                                                     'line '
                                                                                     'search '
                                                                                     '(default, '
                                                                                     'most '
                                                                                     'common)'},
                                                         'newtontr': {'options': {'snes_type': 'newtontr'},
                                                                      'description': 'Newton '
                                                                                     'with '
                                                                                     'trust '
                                                                                     'region '
                                                                                     '(more '
                                                                                     'robust '
                                                                                     'for '
                                                                                     'difficult '
                                                                                     'problems)'},
                                                         'nrichardson': {'options': {'snes_type': 'nrichardson'},
                                                                         'description': 'Nonlinear '
                                                                                        'Richardson '
                                                                                        '(fixed-point)'},
                                                         'ngmres': {'options': {'snes_type': 'ngmres'},
                                                                    'description': 'Nonlinear '
                                                                                   'GMRES '
                                                                                   '(Anderson '
                                                                                   'acceleration)'}},
                                          'convergence': {'snes_atol': 'Absolute '
                                                                       'tolerance on '
                                                                       'residual norm '
                                                                       '(default '
                                                                       '1e-50, set to '
                                                                       '1e-8 or 1e-10)',
                                                          'snes_rtol': 'Relative '
                                                                       'tolerance '
                                                                       '(default 1e-8)',
                                                          'snes_stol': 'Step tolerance '
                                                                       'for '
                                                                       '||delta_x||/||x|| '
                                                                       '(default 1e-8)',
                                                          'snes_max_it': 'Maximum '
                                                                         'nonlinear '
                                                                         'iterations '
                                                                         '(default 50)',
                                                          'snes_monitor': 'Print '
                                                                          'convergence '
                                                                          'info (set '
                                                                          'to '
                                                                          'None/empty '
                                                                          'string)'},
                                          'custom_newton': {'description': 'Hand-written '
                                                                           'Newton '
                                                                           'loop for '
                                                                           'full '
                                                                           'control '
                                                                           '(jsdokken '
                                                                           'tutorial '
                                                                           'chapter 4)',
                                                            'approach': 'Assemble F '
                                                                        'and J '
                                                                        'manually, '
                                                                        'solve '
                                                                        'J*du=-F, '
                                                                        'update u, '
                                                                        'check '
                                                                        'convergence',
                                                            'api': 'dolfinx.fem.petsc.assemble_matrix(a), '
                                                                   'dolfinx.fem.petsc.assemble_vector(L), '
                                                                   'apply_lifting, '
                                                                   'set_bc',
                                                            'convergence_criterion': "'residual' "
                                                                                     '(default) '
                                                                                     'or '
                                                                                     "'incremental'"}},
                    'eigenvalue_solvers': {'SLEPc_EPS': {'api': 'from slepc4py import '
                                                                'SLEPc; eps = '
                                                                'SLEPc.EPS().create(MPI.COMM_WORLD)',
                                                         'use': 'Generalized '
                                                                'eigenvalue problem '
                                                                'A*x = lambda*B*x',
                                                         'methods': 'krylovschur '
                                                                    '(default, '
                                                                    'recommended), '
                                                                    'arnoldi, lanczos, '
                                                                    'power, jd '
                                                                    '(Jacobi-Davidson)',
                                                         'spectral_transform': 'ST for '
                                                                               'shift-and-invert '
                                                                               'to '
                                                                               'find '
                                                                               'eigenvalues '
                                                                               'near a '
                                                                               'target',
                                                         'demo': 'Electromagnetic '
                                                                 'modal analysis '
                                                                 '(waveguide demo)'}},
                    'block_solvers': {'description': 'For saddle-point problems '
                                                     '(Stokes, mixed Poisson)',
                                      'api_0_10': 'dolfinx.fem.petsc.assemble_matrix(a_block, '
                                                  "bcs=bcs, kind='mpi'|'nest') — the "
                                                  'separate assemble_matrix_block / '
                                                  'assemble_matrix_nest functions were '
                                                  'REMOVED in 0.10 and raise '
                                                  'AttributeError (verified '
                                                  '2026-08-03)',
                                      'high_level': 'dolfinx.fem.petsc.LinearProblem(a_block, '
                                                    "L_block, kind='nest'|'mpi', "
                                                    'petsc_options_prefix=...) handles '
                                                    'blocked problems directly in 0.10',
                                      'nullspace': 'Build nullspace for pressure '
                                                   '(constant) or rigid body modes '
                                                   '(elasticity), attach to matrix '
                                                   'with '
                                                   'A.setNullSpace(PETSc.NullSpace().create(constant=True)) '
                                                   '— note dolfinx.la has NO '
                                                   'create_petsc_nullspace_constants '
                                                   'helper (verified 2026-08-03)'},
                    'alternative_backends': {'pyamg': {'api': 'Convert DOLFINx matrix '
                                                              'to scipy sparse, use '
                                                              'pyamg.ruge_stuben_solver() '
                                                              'or '
                                                              'pyamg.smoothed_aggregation_solver()',
                                                       'note': 'Serial only (not '
                                                               'MPI-parallel), good '
                                                               'for rapid prototyping',
                                                       'demo': 'demo_pyamg.py'},
                                             'scipy': {'api': 'mat.to_scipy() to '
                                                              'convert DOLFINx matrix, '
                                                              'then use '
                                                              'scipy.sparse.linalg',
                                                       'note': 'Useful for interfacing '
                                                               'with optimization '
                                                               '(scipy.optimize)'}},
                    'pitfalls': ["Always set petsc_options as dict: {'ksp_type': 'cg', "
                                 "'pc_type': 'gamg'}",
                                 'For elasticity AMG: MUST provide near-nullspace (6 '
                                 'rigid body modes in 3D) via setNearNullSpace()',
                                 'For Stokes: pressure nullspace (constant) must be '
                                 'set via setNullSpace()',
                                 'GAMG/hypre strong_threshold: 0.25 for 2D, 0.5-0.7 '
                                 'for 3D (wrong value = poor convergence)',
                                 'Direct solvers fail silently for very large problems '
                                 '— check ksp_monitor for divergence',
                                 'NewtonSolver is DEAD for 0.10 problems — use '
                                 'dolfinx.fem.petsc.NonlinearProblem(F, u, bcs=..., '
                                 "petsc_options_prefix='x_') then problem.solve(). "
                                 'Constructing dolfinx.nls.petsc.NewtonSolver(comm, '
                                 'problem) around a 0.10 NonlinearProblem first emits '
                                 'DeprecationWarning: dolfinx.nls.petsc.NewtonSolver '
                                 'is deprecated. Use '
                                 'dolfinx.fem.petsc.NonlinearProblem, a high level '
                                 'interface to PETSc SNES. and then raises '
                                 "AttributeError: 'NonlinearProblem' object has no "
                                 "attribute 'a' (verified 2026-08-03). It is not a "
                                 'total removal: '
                                 'dolfinx.fem.petsc.NewtonSolverNonlinearProblem still '
                                 'exists and NewtonSolver(comm, '
                                 'NewtonSolverNonlinearProblem(F, u, bcs=...)) still '
                                 'solves (returning (iterations, converged)) — but the '
                                 'supported path is problem.solve(), with '
                                 'problem.solver a petsc4py PETSc.SNES. Signal: on a '
                                 'converged 0.10 run '
                                 'problem.solver.getConvergedReason() returns 3 (SNES '
                                 'CONVERGED_FNORM_RELATIVE) and getIterationNumber() '
                                 'the Newton count; on failure it returns a NEGATIVE '
                                 'code (-5 = DIVERGED_MAX_IT, -6 = '
                                 'DIVERGED_LINE_SEARCH) and NOTHING is raised, so you '
                                 'must assert it (or pass '
                                 "petsc_options={'snes_error_if_not_converged': "
                                 'True}). Any pitfall text below that names '
                                 'NewtonSolver.solve as its signal is describing a '
                                 '0.9-era code path.',
                                 'snes_atol default is 1e-50 (effectively disabled) — '
                                 'you MUST set it explicitly. Measured defaults on '
                                 'petsc4py 3.24.4: (rtol, atol, stol, max_it) = (1e-8, '
                                 '1e-50, 1e-8, 50) (verified 2026-08-03)',
                                 "[API] pc_type 'bddc' / 'fieldsplit' / hypre 'ams' "
                                 'cannot be used as bare option dicts — each needs '
                                 'extra setup (MATIS operator, field splits, '
                                 'discrete-gradient operator respectively). Signal: '
                                 "LinearProblem(..., petsc_options={'pc_type': "
                                 "'bddc'}) aborts inside KSPSetUp with PETSc error "
                                 "code 62; 'fieldsplit' with error 77; hypre 'ams' "
                                 'with error 83 (verified 2026-08-03)'],
                    'by_physics': {'poisson': 'CG + hypre/GAMG (or LU for small)',
                                   'elasticity': 'CG + GAMG with near-nullspace (or LU '
                                                 'for small)',
                                   'heat_transient': 'CG + hypre per time step',
                                   'stokes': 'MinRes + fieldsplit (AMG for velocity '
                                             'block, mass matrix for Schur complement)',
                                   'navier_stokes': 'SNES newtonls + GMRES + AMG (or '
                                                    'LU for small)',
                                   'helmholtz': 'GMRES + LU (complex-valued, direct '
                                                'often needed)',
                                   'maxwell': 'GMRES + AMS (from hypre) for H(curl) '
                                              'problems — AMS needs the discrete '
                                              'gradient + vertex coordinates attached '
                                              'to the PC; '
                                              "{'pc_type':'hypre','pc_hypre_type':'ams'} "
                                              'alone aborts with PETSc error 83 '
                                              '(verified 2026-08-03)',
                                   'cahn_hilliard': 'SNES + LU per time step'}},
 'boundary_conditions': {'description': 'Complete boundary condition types and API in '
                                        'DOLFINx.',
                         'dirichlet': {'api': 'dolfinx.fem.dirichletbc(value, dofs, '
                                              'V=None)',
                                       'locate_topological': 'dolfinx.fem.locate_dofs_topological(V, '
                                                             'entity_dim, entities)',
                                       'locate_geometrical': 'dolfinx.fem.locate_dofs_geometrical(V, '
                                                             'marker_fn)',
                                       'component_wise': 'V0, _ = V.sub(0).collapse(); '
                                                         'dofs = '
                                                         'locate_dofs_topological((V.sub(0), '
                                                         'V0), fdim, facets)',
                                       'enforcement': 'Strong enforcement via lifting '
                                                      '(modify RHS, zero rows/cols in '
                                                      'matrix)',
                                       'notes': 'DOLFINx uses the lifting approach '
                                                'internally, not identity rows'},
                         'neumann': {'api': 'L += g * v * ds(marker)',
                                     'description': 'Natural BC: specified flux, added '
                                                    'as surface integral in weak form',
                                     'notes': 'Zero Neumann (insulated/free) = do '
                                              'nothing (natural condition). Non-zero: '
                                              'integrate over ds with marker.'},
                         'robin': {'api': 'a += r * u * v * ds(marker); L += r * s * v '
                                          '* ds(marker)',
                                   'description': 'Mixed BC: -k*du/dn = r*(u - s) '
                                                  'where r=transfer coefficient, '
                                                  's=ambient value',
                                   'use_cases': 'Convective heat transfer, radiation, '
                                                'absorbing boundary'},
                         'periodic': {'library': 'dolfinx_mpc (extension by Jørgen S. '
                                                 'Dokken)',
                                      'api': 'mpc = '
                                             'dolfinx_mpc.MultiPointConstraint(V); '
                                             'mpc.create_periodic_constraint_geometrical(V, '
                                             'indicator, relation, bcs, scale)',
                                      'notes': 'NOT built into DOLFINx core — requires '
                                               'separate dolfinx_mpc package',
                                      'topological': 'mpc.create_periodic_constraint_topological(V, '
                                                     'meshtag, tag, relation, bcs, '
                                                     'scale)'},
                         'point_constraints': {'approach': 'Use '
                                                           'locate_dofs_geometrical '
                                                           'with a function checking '
                                                           'point proximity',
                                               'lagrange_multiplier': 'Possible via '
                                                                      'real-valued '
                                                                      'function space '
                                                                      '(workaround for '
                                                                      'integral '
                                                                      'constraints)'},
                         'outlet_do_nothing': {'description': 'Natural (do-nothing) BC '
                                                              'at outlet: zero stress '
                                                              'condition',
                                               'api': 'Simply do not specify any BC on '
                                                      'the outlet boundary — it is '
                                                      'naturally satisfied'},
                         'pitfalls': ['Connectivity: '
                                      'mesh.topology.create_connectivity(fdim, tdim) '
                                      'is not required before '
                                      'locate_entities_boundary or ds/dS assembly on '
                                      'dolfinx 0.10, because these build it themselves. '
                                      'locate_dofs_topological does NOT build it: it '
                                      'works only if facet->cell connectivity already '
                                      'exists (e.g. created by an earlier '
                                      'locate_entities_boundary call), otherwise '
                                      "RuntimeError 'Entity-to-cell connectivity has not "
                                      "been computed. Missing dims 1->2' (2-D mesh). It "
                                      'IS still required before '
                                      'dolfinx.mesh.exterior_facet_indices(mesh.topology), '
                                      "which raises RuntimeError 'Facet to cell "
                                      "connectivity has not been computed.' Calling it "
                                      'explicitly remains harmless and is the safer '
                                      'tutorial pattern. (Verified by execution on '
                                      'dolfinx 0.10.0.)',
                                      'For sub-space BCs: with a scalar or Constant '
                                      'value whose size equals the sub-space block '
                                      'size (e.g. V.sub(i) of a vector space, or a '
                                      'scalar field of a mixed space), '
                                      'locate_dofs_topological(V.sub(i), fdim, facets) '
                                      '+ dirichletbc(value, dofs, V.sub(i)) suffices. '
                                      'When the value is a Function on the collapsed '
                                      'sub-space, which is required for a vector '
                                      'sub-space of a mixed space such as Taylor-Hood '
                                      'W.sub(0), pass the (W.sub(i), W_i_collapsed) '
                                      'tuple to locate_dofs_topological',
                                      'Periodic BCs require dolfinx_mpc extension — '
                                      'not natively in DOLFINx (confirmed: no '
                                      'periodic-constraint API anywhere in dolfinx '
                                      '0.10)',
                                      'Dirichlet value type must match: '
                                      'np.array([0.0]*gdim, dtype=default_scalar_type) '
                                      'for a vector space, scalar for a scalar space. '
                                      'A scalar on a vector space raises RuntimeError '
                                      "'Rank mismatch between Constant and function "
                                      "space in DirichletBC' (verified 2026-08-03 — it "
                                      'is a dolfinx rank check, NOT a numpy broadcast '
                                      'error)',
                                      'For enclosed flows (all Dirichlet velocity): '
                                      'pin pressure at one DOF to remove nullspace',
                                      '[API] A strong DirichletBC on a DG '
                                      'FunctionSpace is a silent no-op — impose '
                                      'inflow/boundary data weakly through a ds '
                                      'integral instead. Signal: '
                                      'fem.locate_dofs_topological on a DG1 space with '
                                      'the x=0 boundary facets of an 8x8 unit square '
                                      'returns an EMPTY array (0 dofs), so the BC '
                                      'constrains nothing and no error is raised. '
                                      '(Verified empirically 2026-08-03.)']},
 'io_catalog': {'description': 'Complete I/O capabilities in DOLFINx for '
                               'visualization, checkpointing, and data exchange.',
                'vtx_writer': {'api': 'dolfinx.io.VTXWriter(MPI.COMM_WORLD, '
                                      "'output.bp', [u], engine='BP4')",
                               'write': 'writer.write(t)',
                               'close': 'writer.close()',
                               'features': 'Arbitrary-order Lagrange, time series, '
                                           'parallel',
                               'viewer': 'ParaView (open .bp directory)',
                               'notes': 'Requires ADIOS2. Best for Lagrange elements. '
                                        'VTXMeshPolicy controls mesh update '
                                        'frequency.'},
                'xdmf_file': {'api': 'dolfinx.io.XDMFFile(MPI.COMM_WORLD, '
                                     "'output.xdmf', 'w')",
                              'write_mesh': 'f.write_mesh(mesh)',
                              'write_function': 'f.write_function(u, t)',
                              'read_mesh': 'f.read_mesh()',
                              'features': 'XML+HDF5, parallel, read/write meshes and '
                                          'functions',
                              'notes': 'Geometry order <= 2 supported. Good for '
                                       'meshes. For functions, VTX preferred.'},
                'vtkhdf': {'api': "dolfinx.io.vtkhdf.read_mesh(comm, 'file.vtkhdf') — "
                                  'new in 0.10 (communicator first)',
                           'notes': "Kitware's future format. Reading AND writing are "
                                    'both available on 0.10: read_mesh, write_mesh, '
                                    'write_point_data, write_cell_data (verified '
                                    "2026-08-03 — the older 'writing in progress' note "
                                    'is stale).'},
                'checkpointing': {'library': 'adios4dolfinx (extension by Jørgen S. '
                                             'Dokken)',
                                  'api': 'adios4dolfinx.write_mesh(filename, mesh); '
                                         'adios4dolfinx.write_function(filename, u)',
                                  'read': 'adios4dolfinx.read_mesh(filename, comm); '
                                          'adios4dolfinx.read_function(filename, u) '
                                          '(u = dolfinx.fem.Function(V), filled in '
                                          'place)',
                                  'features': 'N-to-M checkpointing (write on N ranks, '
                                              'read on M ranks), function + mesh + '
                                              'meshtags',
                                  'notes': 'Requires ADIOS2. Essential for '
                                           'restart/continuation simulations.'},
                'function_evaluation': {'at_points': 'u.eval(points, cells) — evaluate '
                                                     'function at arbitrary points '
                                                     '(must find containing cells '
                                                     'first)',
                                        'find_cells': 'dolfinx.geometry.bb_tree + '
                                                      'compute_collisions_points + '
                                                      'compute_colliding_cells',
                                        'interpolation': 'u.interpolate(expr) — '
                                                         'interpolate expression or '
                                                         'function into FE space',
                                        'nonmatching': 'dolfinx.fem.Function.interpolate_nonmatching() '
                                                       '— interpolate between '
                                                       'different meshes'},
                'visualization': {'pyvista': {'api': 'grid = '
                                                     'pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V))',
                                              'scalar_warp': 'grid.warp_by_scalar()',
                                              'vector_glyphs': "grid.glyph(orient='vectors', "
                                                               'factor=0.1)',
                                              'streamlines': "grid.streamlines(vectors='vectors')"}},
                'pitfalls': ['VTXWriter requires ADIOS2 — check dolfinx.io.has_adios2',
                             'XDMFFile: only geometry order <= 2; for high-order '
                             'elements, use VTX. Concretely, XDMFFile.write_function '
                             'on a P2 Function over a P1 mesh raises RuntimeError '
                             "'Degree of output Function must be same as mesh degree. "
                             "Maybe the Function needs to be interpolated?' while "
                             'VTXWriter writes the same P2 Function fine (verified '
                             '2026-08-03)',
                             'VTXWriter only works with (discontinuous) Lagrange '
                             'elements — not RT, Nedelec, etc. Exact message: '
                             "RuntimeError 'Only (discontinuous) Lagrange functions "
                             "are supported. Interpolate Functions before output.' "
                             '(verified 2026-08-03)',
                             'Function eval requires finding containing cell first — '
                             'use dolfinx.geometry.bb_tree + compute_collisions_points '
                             '+ compute_colliding_cells',
                             'Checkpointing (restart) requires adios4dolfinx extension '
                             '— not built into DOLFINx',
                             'Close writers explicitly (writer.close()) to flush data '
                             'to disk']},
 'ufl_reference': {'description': 'Unified Form Language (UFL) reference for '
                                  'expressing variational forms in FEniCSx.',
                   'differential_operators': {'grad(f)': 'Gradient: scalar -> vector, '
                                                         'vector -> tensor',
                                              'div(v)': 'Divergence: vector -> scalar, '
                                                        'tensor -> vector',
                                              'curl(v)': 'Curl: vector -> vector (3D) '
                                                         'or scalar (2D)',
                                              'nabla_grad(f)': 'Same as grad but with '
                                                               'different index '
                                                               'convention for tensors',
                                              'nabla_div(v)': 'Same as div but with '
                                                              'different index '
                                                              'convention',
                                              'Dx(f, i)': 'Partial derivative df/dx_i'},
                   'algebraic_operators': {'inner(a, b)': 'Full contraction (all '
                                                          'indices). For vectors: dot '
                                                          'product. Complex: '
                                                          'conjugates 2nd arg.',
                                           'dot(a, b)': 'Contracts last index of a '
                                                        'with first of b',
                                           'outer(a, b)': 'Outer product (tensor '
                                                          'product)',
                                           'cross(a, b)': 'Cross product (3D vectors)',
                                           'det(A)': 'Determinant of matrix',
                                           'tr(A)': 'Trace of matrix',
                                           'sym(A)': 'Symmetric part: 0.5*(A + A^T)',
                                           'skew(A)': 'Skew part: 0.5*(A - A^T)',
                                           'dev(A)': 'Deviatoric part: A - tr(A)/dim * '
                                                     'I',
                                           'inv(A)': 'Matrix inverse (use cofac for '
                                                     'better numerical stability)',
                                           'cofac(A)': 'Cofactor matrix: det(A) * '
                                                       'inv(A)^T',
                                           'transpose(A)': 'Matrix transpose'},
                   'measures': {'dx': 'Volume (cell) integration',
                                'ds': 'Exterior facet (boundary) integration',
                                'dS': 'Interior facet integration (DG methods)',
                                'dx(marker)': 'Integration over subdomain with given '
                                              'marker',
                                'ds(marker)': 'Integration over boundary with given '
                                              'marker'},
                   'special_functions': {'ufl.variable(expr)': 'Declare expression as '
                                                               'differentiable '
                                                               'variable',
                                         'ufl.diff(f, var)': 'Differentiate f with '
                                                             'respect to variable var',
                                         'ufl.derivative(F, u, v)': 'Gateaux '
                                                                    'derivative of '
                                                                    'form F w.r.t. u '
                                                                    'in direction v '
                                                                    '(for Newton '
                                                                    'Jacobian)',
                                         'ufl.adjoint(a)': 'Adjoint of bilinear form '
                                                           '(swap trial/test)',
                                         'ufl.action(a, f)': 'Replace trial function '
                                                             'with coefficient f',
                                         'ufl.replace(form, {old: new})': 'Substitute '
                                                                          'expressions '
                                                                          'in form',
                                         'ufl.lhs(F)': 'Extract bilinear (left) part '
                                                       'from equation',
                                         'ufl.rhs(F)': 'Extract linear (right) part '
                                                       'from equation',
                                         'ufl.system(F)': 'Split into (lhs, rhs) pair'},
                   'dg_operators': {'jump(v)': "Jump across interior facet: v('+') - "
                                               "v('-')",
                                    'jump(v, n)': "Jump with normal: v('+')*n('+') + "
                                                  "v('-')*n('-')",
                                    'avg(v)': 'Average across interior facet: '
                                              "0.5*(v('+') + v('-'))",
                                    "v('+'), v('-')": 'Restriction to '
                                                      'positive/negative side of '
                                                      'interior facet'},
                   'form_compilation': {'form_compiler_options': 'Passed to FFCx: run '
                                                                 "'ffcx --help' for "
                                                                 'all options',
                                        'jit_options': 'Passed to CFFI JIT compilation '
                                                       'of generated C code',
                                        'quadrature_degree': 'Set via metadata: '
                                                             "dx(metadata={'quadrature_degree': "
                                                             'q})',
                                        'example': 'dolfinx.fem.form(a, '
                                                   "form_compiler_options={'table_rtol': "
                                                   "1e-8}, jit_options={'timeout': "
                                                   '120}) — FFCx 0.10 options: epsilon, '
                                                   'scalar_type, sum_factorization, '
                                                   'table_rtol, table_atol, verbosity, '
                                                   'part; unknown keys (e.g. '
                                                   "'optimize') are silently ignored, "
                                                   'and fem.form sets scalar_type from '
                                                   'its dtype argument'},
                   'automatic_differentiation': {'description': 'UFL supports symbolic '
                                                                'differentiation for '
                                                                'deriving Jacobians, '
                                                                'sensitivities, '
                                                                'adjoint operators',
                                                 'jacobian_example': 'F = '
                                                                     'inner(sigma(u), '
                                                                     'grad(v)) * dx; J '
                                                                     '= '
                                                                     'ufl.derivative(F, '
                                                                     'u, du) — '
                                                                     'auto-derive '
                                                                     'Newton Jacobian',
                                                 'material_tangent': 'c = '
                                                                     'ufl.variable(c); '
                                                                     'psi = f(c); '
                                                                     'dpsi_dc = '
                                                                     'ufl.diff(psi, c) '
                                                                     '— material law '
                                                                     'differentiation',
                                                 'adjoint_optimization': 'Use '
                                                                         'ufl.adjoint() '
                                                                         'and '
                                                                         'ufl.action() '
                                                                         'for '
                                                                         'PDE-constrained '
                                                                         'optimization'}},
 'poisson': {'description': 'Poisson equation -div(kappa * grad(u)) = f. Foundation of '
                            'all elliptic PDEs. Covers steady-state diffusion, '
                            'electrostatics, potential flow.',
             'weak_form': 'kappa * inner(grad(u), grad(v)) * dx = inner(f, v) * dx + '
                          'inner(g, v) * ds',
             'function_space': 'Lagrange order 1 or 2 (higher order for smooth '
                               'solutions)',
             'demo_url': 'https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_poisson.html',
             'code_skeleton': {'imports': 'from mpi4py import MPI; from dolfinx import '
                                          'fem, mesh, io; from dolfinx.fem.petsc '
                                          'import LinearProblem; import ufl; import '
                                          'numpy as np',
                               'mesh': 'domain = '
                                       'mesh.create_unit_square(MPI.COMM_WORLD, 32, '
                                       '32)',
                               'space': "V = fem.functionspace(domain, ('Lagrange', "
                                        '1))',
                               'bc': 'fdim = domain.topology.dim - 1; boundary_facets '
                                     '= mesh.locate_entities_boundary(domain, fdim, '
                                     'lambda x: np.full(x.shape[1], True)); bc = '
                                     'fem.dirichletbc(0.0, '
                                     'fem.locate_dofs_topological(V, fdim, '
                                     'boundary_facets), V)',
                               'forms': 'u, v = ufl.TrialFunction(V), '
                                        'ufl.TestFunction(V); a = inner(grad(u), '
                                        'grad(v)) * ufl.dx; L = f * v * ufl.dx',
                               'solve': 'problem = LinearProblem(a, L, bcs=[bc], '
                                        "petsc_options_prefix='solve', "
                                        "petsc_options={'ksp_type': 'cg', 'pc_type': "
                                        "'hypre'}); uh = problem.solve()"},
             'solver': {'direct': 'ksp_type: preonly, pc_type: lu, '
                                  'pc_factor_mat_solver_type: mumps',
                        'iterative': 'ksp_type: cg, pc_type: hypre (BoomerAMG)'},
             'mixed_formulation': {'description': 'Mixed Poisson: introduce flux sigma '
                                                  '= -grad(u), solve for (sigma, u) '
                                                  'simultaneously',
                                   'elements': 'Raviart-Thomas for sigma + DG(k-1) for '
                                               'u',
                                   'demo_url': 'https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_mixed-poisson.html',
                                   'block_preconditioner': 'Block-diagonal Riesz-map '
                                                           'preconditioner for the '
                                                           'saddle-point system'},
             'matrix_free': {'description': 'Matrix-free CG solver using action of '
                                            'bilinear form (no explicit matrix '
                                            'assembly)',
                             'demo_url': 'https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_poisson_matrix_free.html',
                             'notes': 'Computes matrix-vector product on-the-fly. '
                                      'Diagonal assembly available for Jacobi '
                                      'preconditioning.'},
             'pitfalls': ['[API] locate_entities_boundary builds facet->cell '
                          'connectivity itself, so it needs no prior '
                          'create_connectivity, and neither do ds and dS assembly. '
                          'locate_dofs_topological does NOT build it: it works after '
                          'locate_entities_boundary or with gmsh-imported facet tags '
                          '(model_to_mesh builds the connectivity), but not on facets '
                          'from locate_entities on a fresh mesh, so call '
                          'mesh.topology.create_connectivity(fdim, tdim) explicitly '
                          'before it. Signal: on dolfinx 0.10.0, '
                          'locate_dofs_topological without that connectivity raises '
                          "RuntimeError 'Entity-to-cell connectivity has not been "
                          "computed. Missing dims 1->2' (2-D mesh), and "
                          'dolfinx.mesh.exterior_facet_indices(mesh.topology) on a '
                          "fresh mesh raises RuntimeError 'Facet to cell connectivity "
                          "has not been computed.' — so the common 'grab all boundary "
                          "facets' idiom needs create_connectivity(fdim, tdim) too. "
                          '(Verified by execution on dolfinx 0.10.0.)',
                          '[API] dolfinx.default_scalar_type for Constants and '
                          'Function arrays so dtype matches the PETSc build (float64 '
                          'if PETSc is real, complex128 if PETSc is complex). Signal: '
                          '[re-measured 2026-08-03 on a REAL conda-forge build] '
                          'assembling a ufl form that carries an imaginary coefficient '
                          "raises ValueError 'Unexpected complex value in real "
                          "expression.' from fem.form, and writing a complex value "
                          'into a real Function array raises TypeError "float() '
                          'argument must be a string or a real number, not '
                          '\'complex\'". Note fem.Constant(mesh, 1+2j) itself does NOT '
                          'raise — the failure surfaces at form compilation / array '
                          'assignment, not at Constant construction.',
                          '[API] VTXWriter (ADIOS2 backend) supports only Lagrange / '
                          'DG element families. Mixed / Nedelec / BDM / RT Functions '
                          'cannot be written. Signal: [exact text, re-measured '
                          '2026-08-03 on 0.10.0] VTXWriter construction raises '
                          "RuntimeError 'Only (discontinuous) Lagrange functions are "
                          "supported. Interpolate Functions before output.' — the "
                          "older quoted strings 'Cannot interpolate function to the "
                          "VTX output basis' / 'ADIOS2 VTX only supports Lagrange "
                          "elements' do NOT appear in current dolfinx, and the error "
                          'fires at VTXWriter(...) construction, not at .write().',
                          '[Physics] Pure-Neumann Poisson admits the constant null '
                          'space — the solution is determined only up to a constant. '
                          'Either pin one DOF (DirichletBC on a single point) or add a '
                          'Lagrange multiplier enforcing mean(u) = 0. Signal: '
                          'LinearProblem.solve returns without raising, but CG with '
                          "pc_type='none' does NOT converge: "
                          'problem.solver.getConvergedReason() is -4 (DIVERGED_DTOL, '
                          '17 iterations), and the '
                          'resulting Function array has a HUGE additive offset '
                          'accommodating the null space — np.array shows max ≈ min ≈ '
                          'O(1e6) with tiny std (e.g. max=2.18e+06, std=112 on an 8x8 '
                          'unit square with f=1). No exception is raised unless '
                          "petsc_options sets 'ksp_error_if_not_converged': True, "
                          "which gives 'petsc4py.PETSc.Error: error code 91' with the "
                          "line 'KSPSolve() has not converged, reason DIVERGED_DTOL'; "
                          'without it you observe the bug only as the un-pinned '
                          'constant or the negative converged reason. (Verified by '
                          'execution on dolfinx 0.10.0.)',
                          '[Syntax] For non-unit kappa coefficients: define as '
                          'fem.Constant for spatially uniform, or fem.Function '
                          '(interpolated) for spatially varying. Plain Python floats '
                          'inside ufl forms work for unit coefficients but lose unit '
                          'metadata. Signal: ufl form runs but the assembled stiffness '
                          'scale disagrees with the analytic kappa-scaled stiffness by '
                          'exactly the kappa value (when float coefficient was '
                          'forgotten).'],
             'materials': {'kappa': {'range': [0.001, 1000000.0],
                                     'unit': 'W/(m*K) or dimensionless'}},
             'reference_solutions': {'unit_square_f1': 'max(u) ~ 0.0737 for '
                                                       '-laplacian(u)=1 on [0,1]^2, '
                                                       'u=0 on boundary (re-measured '
                                                       '2026-08-03 on dolfinx 0.10.0: '
                                                       '0.073657 with P1 on a 64x64 '
                                                       'mesh)',
                                     'mms_convergence': 'Manufactured u = sin(pi x) '
                                                        'sin(pi y), f = 2 pi^2 sin(pi '
                                                        'x) sin(pi y): observed L2 '
                                                        'rates on N = 8,16,32,64 are '
                                                        'P1 -> 1.97/1.99/2.00, P2 -> '
                                                        '3.00/3.00/3.00, P3 -> '
                                                        '4.06/4.03/4.01, i.e. the '
                                                        'textbook O(h^(k+1)) (verified '
                                                        '2026-08-03)'}},
 'linear_elasticity': {'description': 'Linear elasticity with Lame parameters. Small '
                                      'strain assumption. Plane strain, plane stress, '
                                      'or full 3D.',
                       'weak_form': 'inner(sigma(u), epsilon(v)) * dx = dot(f, v) * dx '
                                    '+ dot(t, v) * ds',
                       'function_space': "Vector Lagrange: element('Lagrange', cell, "
                                         '1, shape=(gdim,))',
                       'demo_url': 'https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity.html',
                       'constitutive': {'sigma(u)': 'lambda_ * nabla_div(u) * '
                                                    'Identity(d) + 2*mu * epsilon(u)',
                                        'epsilon(u)': 'ufl.sym(ufl.grad(u)) = '
                                                      '0.5*(grad(u) + grad(u)^T)',
                                        'mu': 'E / (2*(1+nu))',
                                        'lambda_': 'E*nu / ((1+nu)*(1-2*nu))',
                                        'plane_stress_lambda': '2*mu*lambda_ / (2*mu + '
                                                               'lambda_)'},
                       'code_skeleton': {'space': 'V = fem.functionspace(domain, '
                                                  "('Lagrange', 1, (gdim,)))",
                                         'sigma': 'def sigma(u): return lambda_ * '
                                                  'ufl.nabla_div(u) * '
                                                  'ufl.Identity(len(u)) + '
                                                  '2*mu*ufl.sym(ufl.grad(u))',
                                         'forms': 'a = ufl.inner(sigma(u), epsilon(v)) '
                                                  '* ufl.dx; L = ufl.dot(f, v) * '
                                                  'ufl.dx'},
                       'solver': {'recommended': 'CG + GAMG with near-nullspace (rigid '
                                                 'body modes)',
                                  'alternative': 'LU (MUMPS) for small problems',
                                  'near_nullspace': '6 modes in 3D: 3 translations + 3 '
                                                    'rotations. Set via '
                                                    'matrix.setNearNullSpace()',
                                  'demo_url': 'https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_elasticity.html'},
                       'static_condensation': {'description': 'Mixed '
                                                              'stress-displacement '
                                                              'formulation with '
                                                              'condensation of '
                                                              'internal stress DOFs',
                                               'demo_url': 'https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_static-condensation.html',
                                               'notes': 'Uses numba for efficient '
                                                        'condensation of block forms. '
                                                        "Cook's membrane benchmark."},
                       'pitfalls': ['[Syntax] Vector function space for elasticity in '
                                    "dolfinx is created with ('Lagrange', 1, (gdim,)) "
                                    '— the trailing shape tuple marks it '
                                    "vector-valued. Passing ('Lagrange', 1) gives a "
                                    'SCALAR space; the weak form fails at construction '
                                    'when ufl.sym(ufl.grad) is invoked on the scalar '
                                    'trial. Signal: ufl.sym raises ValueError '
                                    "'Symmetric part of tensor with rank != 2 is "
                                    "undefined.' inside the form definition (before "
                                    'assemble). (Verified empirically 2026-06-01 — '
                                    "prior wording 'Invalid ranks' / 'expected rank 1 "
                                    "trial' does not appear in current dolfinx.)",
                                    '[Syntax] Dirichlet BC value for a vector '
                                    'elasticity space must be np.array([0.0]*gdim, '
                                    'dtype=default_scalar_type) — not scalar 0. '
                                    'Signal: [[re-measured 2026-08-03 on dolfinx '
                                    '0.10.0] fem.dirichletbc(0.0, dofs, V) on a vector '
                                    "space raises RuntimeError 'Rank mismatch between "
                                    "Constant and function space in DirichletBC'. This "
                                    'is a dolfinx rank check — the older quoted numpy '
                                    "ValueError 'could not broadcast input array from "
                                    "shape () into shape (gdim,)' does NOT appear.",
                                    '[Physics] Plane strain vs plane stress: lambda '
                                    'must be adjusted. Plane stress uses lambda_star = '
                                    '2*lambda*mu/(lambda+2*mu). Signal: [MEASURED '
                                    '2026-08-03, cantilever 1.0 x 0.2, P2 vector '
                                    'Lagrange, 40x8 triangles, end traction] the '
                                    'plane-STRAIN tip deflection is a factor (1 - '
                                    'nu^2) of the plane-STRESS one — 0.90941 measured '
                                    'vs 0.9100 predicted at nu=0.3, and 0.79100 vs '
                                    '0.7975 at nu=0.45. So plane strain is only ~9% '
                                    'stiffer at nu=0.3, NOT ~30%, and the factor is '
                                    '(1-nu^2), NOT (1-nu). (Corrects the previous '
                                    "wording, which quoted both '~30%' and '(1-nu)' — "
                                    'neither reproduces.)',
                                    '[Numerical] Near-incompressible (nu > 0.49): a '
                                    'mixed (u, p) formulation is the robust choice, '
                                    'but the severity of locking depends strongly on '
                                    'ELEMENT ORDER and mesh, not just on nu. Signal: '
                                    '[MEASURED 2026-08-03] (same cantilever, tip '
                                    'deflection vs a P2/P1 Taylor-Hood reference): P1 '
                                    'triangles lock by 7.2x / 16.5x / 19.5x at nu = '
                                    '0.49 / 0.499 / 0.4999 on a coarse 10x2 mesh, '
                                    'improving to 1.2x / 2.4x / 9.2x on 80x16 — i.e. '
                                    'locking is real but mesh-dependent and is a '
                                    "factor of ~2-20, NOT 'orders of magnitude' and "
                                    "NOT the '~1e-3 of analytic' quoted previously. P2 "
                                    'triangles do NOT meaningfully lock at all here: '
                                    'TaylorHood/P2 = 1.00-1.06 across every nu and '
                                    'mesh tested. Recommend mixed (or P2+) for nu > '
                                    '0.49; do not expect the P1 catastrophe from P2.',
                                    '[Numerical] For GAMG/AMG: provide the '
                                    'near-nullspace (rigid body modes — 3 translations '
                                    '+ 3 rotations in 3D) via '
                                    'A.setNearNullSpace(PETSc.NullSpace().create(vectors=rbm)). '
                                    'Signal: [MEASURED 2026-08-03, dolfinx 0.10.0; P1 '
                                    'tetrahedral cantilever 2x1x1, CG + GAMG, rtol '
                                    '1e-8] WITHOUT the near-nullspace the solve still '
                                    'CONVERGES (reason 2) in 31 / 38 / 41 iterations '
                                    'at 1911 / 7623 / 19575 dofs; WITH the 6 '
                                    'rigid-body modes it takes 16 / 16 / 23. IMPORTANT '
                                    'CORRECTION: at these sizes the near-nullspace is '
                                    'a 1.8x-2.4x iteration-count win, NOT the 10x-50x '
                                    'previously claimed, and its absence does NOT '
                                    "produce 'KSP did not converge' / iteration count "
                                    '= max_it. The claim of outright failure was not '
                                    "reproduced up to ~20k dofs; treat 'MUST' as "
                                    "'strongly recommended, and increasingly so with "
                                    "problem size' and measure your own iteration "
                                    'counts before quoting a speedup.',
                                    "[Physics] There is no dolfinx 'default': "
                                    'whichever lambda you put in sigma() decides it, '
                                    'and the standard 3D Lame lambda = '
                                    'E*nu/((1+nu)(1-2nu)) written in a 2D form gives '
                                    'PLANE STRAIN. Forgetting this is a silent source '
                                    'of wrong answers for thin structures. Signal: a '
                                    '2D vector-Lagrange dolfinx Function plate deflection '
                                    'differs from the plane-stress reference by factor '
                                    '(1-nu^2) — measured 0.90941 at nu=0.3 against a '
                                    'predicted 0.9100 (verified empirically '
                                    '2026-08-03).',
                                    '[API] dolfinx.fem.functionspace rejects element '
                                    'family names that basix does not register. '
                                    'Signal: [[re-measured 2026-08-03 on dolfinx '
                                    '0.10.0 / basix 0.10.0] the DOLFIN degree-suffixed '
                                    "name 'P1' raises ValueError 'Unknown element "
                                    "family: P1 with cell type triangle'. 'CG' however "
                                    'STILL WORKS — it only emits a DeprecationWarning '
                                    '(\'"CG" element name is deprecated. Consider '
                                    'using "Lagrange" or "P" instead\') and builds the '
                                    "space. The previous wording, which claimed ('CG', "
                                    "1) raises ValueError 'Unknown element family CG', "
                                    'does not reproduce.',
                                    '[API] dolfinx XDMFFile.write_function requires '
                                    'the Function degree to match the mesh degree. P2 '
                                    'on a P1 mesh (the common case) is rejected — '
                                    'interpolate to a matching-degree space, or use '
                                    'VTKFile / VTXWriter. Signal: '
                                    'XDMFFile.write_function raises RuntimeError '
                                    "'Degree of output Function must be same as mesh "
                                    'degree. Maybe the Function needs to be '
                                    "interpolated?'. (Verified empirically 2026-06-01 "
                                    "— prior wording 'XDMF mesh must be P1' does not "
                                    'appear.)'],
                       'materials': {'E': {'range': [1.0, 1000000000000.0],
                                           'unit': 'Pa',
                                           'examples': {'steel': 210000000000.0,
                                                        'aluminum': 70000000000.0,
                                                        'rubber': 1000000.0}},
                                     'nu': {'range': [0.0, 0.499],
                                            'unit': 'dimensionless',
                                            'examples': {'steel': 0.3,
                                                         'rubber': 0.49}}}},
 'stokes': {'description': 'Stokes flow (Re -> 0). Linear saddle-point problem. Mixed '
                           'P2/P1 (Taylor-Hood) or MINI element.',
            'weak_form': 'nu*inner(grad(u),grad(v))*dx - p*div(v)*dx - q*div(u)*dx = '
                         'dot(f,v)*dx',
            'function_space': 'Mixed: Taylor-Hood P2/P1 (inf-sup stable). Alternative: '
                              'MINI (P1+Bubble/P1), CR/DG0.',
            'demo_url': 'https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_stokes.html',
            'element_construction': {'taylor_hood': 'P2v = '
                                                    "basix.ufl.element('Lagrange', "
                                                    'cell, 2, shape=(gdim,)); P1 = '
                                                    "basix.ufl.element('Lagrange', "
                                                    'cell, 1); TH = '
                                                    'basix.ufl.mixed_element([P2v, '
                                                    'P1])',
                                     'mini': "P1v = basix.ufl.element('Lagrange', "
                                             'cell, 1, shape=(gdim,)); B = '
                                             "basix.ufl.element('Bubble', cell, "
                                             'gdim+1, shape=(gdim,)); V_el = '
                                             'basix.ufl.enriched_element([P1v, B]); P1 '
                                             "= basix.ufl.element('Lagrange', cell, "
                                             '1); MINI = '
                                             'basix.ufl.mixed_element([V_el, P1])'},
            'solver': {'direct': 'LU (MUMPS) for small problems (linear system, no '
                                 'Newton)',
                       'iterative': 'MinRes + fieldsplit block preconditioner',
                       'block_precon': 'AMG for velocity block, pressure mass matrix '
                                       'for Schur complement approximation'},
            'pitfalls': ['[Numerical] MUST use an inf-sup stable velocity-pressure '
                         'pair. Taylor-Hood (P2v + P1) and MINI (P1v + Bubble enriched '
                         '+ P1) are stable; equal-order P1/P1 constructs a valid mixed '
                         'FunctionSpace but the discrete LBB condition is violated, so '
                         'the pressure field develops checkerboard oscillations in the '
                         'kernel direction. Signal: with the same 4x4 unit-square '
                         'triangulation in dolfinx 0.10, basix.ufl.mixed_element '
                         'returns FunctionSpaces with dim 187 (TH), 139 (MINI), 75 '
                         '(P1/P1); the P1/P1 system assembles but the pressure null '
                         'space has more vectors than just the constant pressure. '
                         '(Verified empirically 2026-06-01 — Tier-2 fixture '
                         'stokes_basix_element_construction in '
                         'scripts/tier2_fixtures/fenics/. Re-verified 2026-08-03 on '
                         'dolfinx 0.10.0: on the 4x4 mesh the three dims are still '
                         'exactly 187 / 139 / 75, and the instability is now measured '
                         'rather than advisory — SVD of the bc-applied Stokes matrix '
                         'on an 8x8 mesh gives numerical null dimension 1 for '
                         'Taylor-Hood (the constant pressure alone) versus 8 for '
                         'P1/P1. MIND THE MESH: the dims and the SVD were measured on '
                         'DIFFERENT meshes. On 8x8 the same three pairs give 659 / 499 '
                         '/ 243, not 187 / 139 / 75. Corrected 2026-08-03 — the '
                         "earlier wording said 'the same 8x8 mesh' for both, which "
                         'does not reproduce.)',
                         '[Numerical] Pressure for enclosed (all-Dirichlet on '
                         'velocity) flows is determined only up to an additive '
                         'constant. Pin one pressure DOF with a fem.dirichletbc at a '
                         'chosen vertex, or build the constant nullspace yourself with '
                         'PETSc.NullSpace().create(constant=True) and call '
                         'A.setNullSpace(ns) on the PETSc matrix. There is NO '
                         'dolfinx.la.create_petsc_nullspace_constants helper in '
                         'dolfinx 0.10 — dolfinx.la exposes only BlockMode / IndexMap '
                         '/ InsertMode / MatrixCSR / Norm / Vector / is_orthonormal / '
                         'matrix_csr / norm / orthonormalize / vector (plus the '
                         'lazily-imported petsc submodule). Signal: [MEASURED on '
                         'dolfinx 0.10.0, lid-driven cavity, Taylor-Hood, MUMPS] '
                         'skipping the pin does NOT make MUMPS complain — '
                         'problem.solver.getConvergedReason() returns 4 '
                         "(CONVERGED_ITS) and no 'INFOG(1)=-9' is emitted; instead the "
                         'pressure comes back with an arbitrary offset of '
                         'unpredictable magnitude: mean(p) varied over fifteen or '
                         'more orders of magnitude across five mesh resolutions of '
                         'the SAME problem, with no pattern, and it changes with the '
                         'OpenMP thread count. max|u| is no reliable check either: it '
                         'mostly stayed at the lid speed 1.0, but the resolution that '
                         'got the ~1e16-1e17 offset can come back with a polluted '
                         'velocity too (max|u| = 1.014 in a single-threaded run). '
                         'Diagnose from the pressure magnitude, not from the converged '
                         'reason.',
                         '[API] basix.ufl.element supports quadrilateral cells '
                         '(CellType.quadrilateral) for Taylor-Hood-style Q2/Q1: pass '
                         'cell=msh.basix_cell() from a create_unit_square(..., '
                         'cell_type=CellType.quadrilateral) mesh and the same '
                         "'Lagrange' family string + degree=2/1. Triangle-mesh helpers "
                         'like the default cell from create_unit_square use '
                         'CellType.triangle; the cell type must match. Signal: '
                         "msh.basix_cell() returns 'CellType.triangle' or "
                         "'CellType.quadrilateral' consistent with the mesh "
                         'constructor. (Catalog claim inherited; not separately Tier-2 '
                         'falsified this iteration.)',
                         '[Numerical] Block preconditioner is essential for iterative '
                         'MinRes / GMRES solves beyond ~100k dofs. Use fieldsplit with '
                         'PETSc PCFIELDSPLIT: type=Schur, with A^-1 on the velocity '
                         'block (AMG via PCHYPRE / GAMG) and a pressure mass matrix '
                         'M_p as the Schur-complement approximation. Without '
                         'fieldsplit the saddle-point spectrum forces MinRes iteration '
                         'counts to scale with mesh refinement. Signal: PETSc KSPSolve '
                         'iteration count grows like O(h^-2) without fieldsplit and '
                         'stays O(1) with the block preconditioner. (Catalog claim '
                         'inherited; not separately Tier-2 falsified this '
                         'iteration.)']},
 'navier_stokes': {'description': 'Incompressible Navier-Stokes: rho*(du/dt + '
                                  '(u.grad)u) = -grad p + mu*lap(u) + f, div(u) = 0, '
                                  'written in kinematic form with nu = mu/rho. Two '
                                  'standard dolfinx routes: (1) a monolithic mixed '
                                  'velocity-pressure formulation solved by Newton '
                                  '(steady or fully implicit), (2) IPCS '
                                  'fractional-step splitting for transient flow.',
                   'minimal_working_example': '# Steady monolithic Taylor-Hood '
                                              'Navier-Stokes, lid-driven cavity, '
                                              'dolfinx 0.10.\n'
                                              'from mpi4py import MPI\n'
                                              'import numpy as np\n'
                                              'import ufl\n'
                                              'import basix.ufl\n'
                                              'from dolfinx import mesh, fem\n'
                                              'from dolfinx.fem.petsc import '
                                              'NonlinearProblem\n'
                                              '\n'
                                              'msh = '
                                              'mesh.create_unit_square(MPI.COMM_WORLD, '
                                              '16, 16, mesh.CellType.triangle)\n'
                                              'gdim = msh.geometry.dim\n'
                                              'n = ufl.FacetNormal(msh)\n'
                                              '\n'
                                              'P2 = basix.ufl.element("Lagrange", '
                                              'msh.basix_cell(), 2, shape=(gdim,))\n'
                                              'P1 = basix.ufl.element("Lagrange", '
                                              'msh.basix_cell(), 1)\n'
                                              'W = fem.functionspace(msh, '
                                              'basix.ufl.mixed_element([P2, P1]))\n'
                                              'V, _ = W.sub(0).collapse()\n'
                                              'Q, _ = W.sub(1).collapse()\n'
                                              '\n'
                                              'w = fem.Function(W)\n'
                                              'u, p = ufl.split(w)\n'
                                              'v, q = ufl.TestFunctions(W)\n'
                                              'nu = fem.Constant(msh, 0.02)\n'
                                              '\n'
                                              'F = (nu * ufl.inner(ufl.grad(u), '
                                              'ufl.grad(v)) * ufl.dx\n'
                                              '     + ufl.inner(ufl.dot(u, '
                                              'ufl.nabla_grad(u)), v) * ufl.dx\n'
                                              '     - p * ufl.div(v) * ufl.dx\n'
                                              '     - q * ufl.div(u) * ufl.dx)\n'
                                              'J = ufl.derivative(F, w)\n'
                                              '\n'
                                              'msh.topology.create_connectivity(gdim - '
                                              '1, gdim)\n'
                                              'lid = '
                                              'mesh.locate_entities_boundary(msh, gdim '
                                              '- 1, lambda x: np.isclose(x[1], 1.0))\n'
                                              'walls = mesh.locate_entities_boundary(\n'
                                              '    msh, gdim - 1,\n'
                                              '    lambda x: np.isclose(x[0], 0.0) | '
                                              'np.isclose(x[0], 1.0) | '
                                              'np.isclose(x[1], 0.0))\n'
                                              '\n'
                                              'u_lid = fem.Function(V)\n'
                                              'u_lid.interpolate(lambda x: '
                                              'np.vstack((np.ones_like(x[0]), '
                                              'np.zeros_like(x[0]))))\n'
                                              'u_wall = fem.Function(V)\n'
                                              'u_wall.x.array[:] = 0.0\n'
                                              'p_ref = fem.Function(Q)\n'
                                              'p_ref.x.array[:] = 0.0\n'
                                              '\n'
                                              'bcs = [\n'
                                              '    fem.dirichletbc(u_lid,\n'
                                              '                    '
                                              'fem.locate_dofs_topological((W.sub(0), '
                                              'V), gdim - 1, lid), W.sub(0)),\n'
                                              '    fem.dirichletbc(u_wall,\n'
                                              '                    '
                                              'fem.locate_dofs_topological((W.sub(0), '
                                              'V), gdim - 1, walls), W.sub(0)),\n'
                                              '    fem.dirichletbc(p_ref, '
                                              'fem.locate_dofs_geometrical(\n'
                                              '        (W.sub(1), Q), lambda x: '
                                              'np.isclose(x[0], 0.0) & '
                                              'np.isclose(x[1], 0.0)), W.sub(1)),\n'
                                              ']\n'
                                              '\n'
                                              'problem = NonlinearProblem(\n'
                                              '    F, w, J=J, bcs=bcs, '
                                              'petsc_options_prefix="ns_",\n'
                                              '    petsc_options={"snes_type": '
                                              '"newtonls", "snes_rtol": 1e-9, '
                                              '"snes_atol": 1e-11,\n'
                                              '                   "snes_max_it": 30, '
                                              '"ksp_type": "preonly", "pc_type": '
                                              '"lu",\n'
                                              '                   '
                                              '"pc_factor_mat_solver_type": "mumps"})\n'
                                              'problem.solve()\n'
                                              'snes = problem.solver\n'
                                              'assert snes.getConvergedReason() > 0, '
                                              'f"Newton failed: '
                                              '{snes.getConvergedReason()}"\n'
                                              '\n'
                                              'uh = w.sub(0).collapse()\n'
                                              'flux = msh.comm.allreduce(\n'
                                              '    '
                                              'fem.assemble_scalar(fem.form(ufl.dot(u, '
                                              'n) * ufl.ds)), op=MPI.SUM)\n'
                                              'umax = '
                                              'msh.comm.allreduce(np.max(np.abs(uh.x.array)), '
                                              'op=MPI.MAX)\n'
                                              'prange = '
                                              'msh.comm.allreduce(float(np.ptp(w.sub(1).collapse().x.array)), '
                                              'op=MPI.MAX)\n'
                                              'print(f"SNES '
                                              'reason={snes.getConvergedReason()} '
                                              'its={snes.getIterationNumber()} "\n'
                                              '      '
                                              'f"|F|={snes.getFunctionNorm():.3e}")\n'
                                              'print(f"net boundary flux = '
                                              '{flux:.3e}   (mass conservation, must '
                                              'be ~0)")\n'
                                              'print(f"max|u| = {umax:.4f}   (must not '
                                              'exceed the lid speed 1.0)")\n'
                                              'print(f"pressure range = {prange:.4f}   '
                                              '(must be finite)")\n',
                   'function_space': {'REQUIRED': 'Monolithic route -- one mixed '
                                                  'space, velocity degree exactly one '
                                                  'higher than pressure degree:\n'
                                                  '  P2 = '
                                                  "basix.ufl.element('Lagrange', "
                                                  'msh.basix_cell(), 2, '
                                                  'shape=(msh.geometry.dim,))\n'
                                                  '  P1 = '
                                                  "basix.ufl.element('Lagrange', "
                                                  'msh.basix_cell(), 1)\n'
                                                  '  W  = '
                                                  'dolfinx.fem.functionspace(msh, '
                                                  'basix.ufl.mixed_element([P2, P1]))\n'
                                                  '  V, _ = W.sub(0).collapse()   # '
                                                  'collapsed velocity space (needed '
                                                  'for BC values)\n'
                                                  '  Q, _ = W.sub(1).collapse()   # '
                                                  'collapsed pressure space\n'
                                                  '  w = dolfinx.fem.Function(W); u, p '
                                                  '= ufl.split(w)\n'
                                                  '  v, q = ufl.TestFunctions(W)\n'
                                                  'IPCS route -- two SEPARATE (not '
                                                  'mixed) spaces:\n'
                                                  '  V = '
                                                  'dolfinx.fem.functionspace(msh, '
                                                  "('Lagrange', 2, "
                                                  '(msh.geometry.dim,)))\n'
                                                  '  Q = '
                                                  'dolfinx.fem.functionspace(msh, '
                                                  "('Lagrange', 1))",
                                      'OPTIONAL': 'The Taylor-Hood pair may be raised '
                                                  'to (velocity degree k, pressure '
                                                  'degree k-1) for any k >= 2; (3, 2) '
                                                  'is verified to work. Cell type may '
                                                  'be triangle or quadrilateral; both '
                                                  'work with the k/(k-1) pairing. In '
                                                  '3D use shape=(3,) and '
                                                  'mesh.create_unit_cube. For '
                                                  'post-processing, '
                                                  '`w.sub(0).collapse()` returns a '
                                                  'standalone velocity Function on V '
                                                  'and `w.sub(1).collapse()` a '
                                                  'standalone pressure Function on Q.',
                                      'explanation': 'Velocity and pressure are '
                                                     'coupled by the incompressibility '
                                                     'constraint, which makes the '
                                                     'discrete system a saddle-point '
                                                     'problem. Only an inf-sup (LBB) '
                                                     'stable pair gives a solvable, '
                                                     'oscillation-free system; '
                                                     'equal-order Lagrange pairs do '
                                                     'not.',
                                      'pitfalls': ['Never use an equal-order pair '
                                                   '(P1/P1, P2/P2). Signal: P1/P1 on '
                                                   'triangles fails in the first linear '
                                                   'solve: KSP getConvergedReason() == '
                                                   '-11 (with ksp_converged_reason set '
                                                   "PETSc prints 'Linear "
                                                   '<prefix> solve did not converge '
                                                   'due to DIVERGED_PC_FAILED '
                                                   "iterations 0' and 'PC failed due "
                                                   "to FACTOR_NUMERIC_ZEROPIVOT'), "
                                                   'SNES getConvergedReason() == -3, '
                                                   'and the returned solution is all '
                                                   'zero. P1/P1 on quadrilaterals and '
                                                   'P2/P2 on both cell types report KSP '
                                                   'CONVERGED_ITS on every step, but '
                                                   'Newton ends with SNES -6 '
                                                   '(DIVERGED_LINE_SEARCH), max|u| a '
                                                   'few times (2-9x) the imposed lid '
                                                   'speed and max|p| of order '
                                                   '1e16-1e18.']},
                   'weak_form': {'REQUIRED': 'Steady monolithic residual (kinematic '
                                             'form, nu = mu/rho), Newton on F == 0:\n'
                                             '  F = (nu*ufl.inner(ufl.grad(u), '
                                             'ufl.grad(v))*ufl.dx\n'
                                             '       + ufl.inner(ufl.dot(u, '
                                             'ufl.nabla_grad(u)), v)*ufl.dx\n'
                                             '       - p*ufl.div(v)*ufl.dx\n'
                                             '       - q*ufl.div(u)*ufl.dx)\n'
                                             'with `u, p = ufl.split(w)` where w is a '
                                             'Function on the mixed space (a '
                                             'TrialFunction will NOT work -- Newton '
                                             'needs the current iterate). A body force '
                                             'is added as an extra term `- '
                                             'ufl.inner(f, v)*ufl.dx` with `f = '
                                             'dolfinx.fem.Constant(msh, np.array([0.0, '
                                             '-9.81]))`; with no body force simply '
                                             'leave that term out, as the example '
                                             'does.',
                                 'OPTIONAL': 'J = ufl.derivative(F, w) may be passed '
                                             'as J= to NonlinearProblem; if omitted '
                                             'dolfinx forms the same Jacobian '
                                             'automatically. For a fully implicit '
                                             'transient run add ufl.inner((u - '
                                             'u_n)/dt, v)*ufl.dx with u_n a Function '
                                             'on the mixed space and step in time with '
                                             'the same NonlinearProblem. The '
                                             'convective term may be written '
                                             'ufl.dot(u, ufl.nabla_grad(u)) or '
                                             'ufl.grad(u)*u -- both are the same '
                                             'operator (u.grad)u; ufl.nabla_grad and '
                                             'ufl.grad are transposes of each other, '
                                             'so do not mix the two conventions inside '
                                             'one form. The sign convention -p*div(v) '
                                             '- q*div(u) keeps the block matrix '
                                             'symmetric in the Stokes limit; using '
                                             '+q*div(u) is also valid but changes the '
                                             'sign of the pressure block.',
                                 'explanation': 'The pressure enters only through '
                                                '-p*div(v), and the incompressibility '
                                                'constraint is imposed weakly by the '
                                                '-q*div(u) row. There is no separate '
                                                'pressure equation, which is why the '
                                                'pressure is only determined up to a '
                                                'constant when no traction (outflow) '
                                                'boundary is present.',
                                 'pitfalls': ['Use a Function (not a TrialFunction) as '
                                              'the unknown in the monolithic residual '
                                              'and pass it as the second argument to '
                                              'NonlinearProblem. Signal: with '
                                              'ufl.TrialFunctions(W) the convective '
                                              'term multiplies the trial function by '
                                              'itself and UFL refuses the form with '
                                              '"ufl.algorithms.check_arities.ArityMismatch: '
                                              'Multiplying expressions with '
                                              'overlapping form argument number 1, '
                                              'argument is v_1."']},
                   'boundary_conditions': {'REQUIRED': 'A Dirichlet BC on a sub-space '
                                                       'of a mixed space needs THREE '
                                                       'things: a Function living on '
                                                       'the COLLAPSED sub-space, a dof '
                                                       'pair produced by passing a '
                                                       '(sub-space, collapsed-space) '
                                                       'tuple, and the sub-space as '
                                                       'third argument:\n'
                                                       '  V, _ = W.sub(0).collapse()\n'
                                                       '  u_bc = '
                                                       'dolfinx.fem.Function(V)\n'
                                                       '  u_bc.interpolate(lambda x: '
                                                       'np.vstack((4.0*x[1]*(1.0-x[1]), '
                                                       'np.zeros_like(x[0]))))\n'
                                                       '  facets = '
                                                       'dolfinx.mesh.locate_entities_boundary(msh, '
                                                       'gdim-1, marker)\n'
                                                       '  dofs = '
                                                       'dolfinx.fem.locate_dofs_topological((W.sub(0), '
                                                       'V), gdim-1, facets)\n'
                                                       '  bc = '
                                                       'dolfinx.fem.dirichletbc(u_bc, '
                                                       'dofs, W.sub(0))\n'
                                                       'For an ENCLOSED domain (every '
                                                       'boundary facet carries a '
                                                       'velocity Dirichlet BC, e.g. a '
                                                       'lid-driven cavity) you MUST '
                                                       'also pin the pressure:\n'
                                                       '  Q, _ = W.sub(1).collapse()\n'
                                                       '  p_ref = '
                                                       'dolfinx.fem.Function(Q); '
                                                       'p_ref.x.array[:] = 0.0\n'
                                                       '  pdofs = '
                                                       'dolfinx.fem.locate_dofs_geometrical(\n'
                                                       '      (W.sub(1), Q), lambda x: '
                                                       'np.isclose(x[0], 0.0) & '
                                                       'np.isclose(x[1], 0.0))\n'
                                                       '  '
                                                       'bcs.append(dolfinx.fem.dirichletbc(p_ref, '
                                                       'pdofs, W.sub(1)))',
                                           'OPTIONAL': 'If the domain has an outflow '
                                                       'boundary you may leave it '
                                                       'free: omitting any term on '
                                                       'that part of the boundary '
                                                       "imposes the natural 'do "
                                                       "nothing' condition "
                                                       'nu*grad(u).n - p*n = 0, which '
                                                       'fixes the pressure level, so '
                                                       'NO pressure pinning is needed. '
                                                       'Alternatively impose p = 0 '
                                                       'there with a Dirichlet BC on '
                                                       'W.sub(1). A component-only BC '
                                                       'uses a nested sub-space, e.g. '
                                                       'W.sub(0).sub(1) with its own '
                                                       'collapse(). '
                                                       'locate_dofs_geometrical and '
                                                       'locate_dofs_topological are '
                                                       'interchangeable; both accept '
                                                       'the (sub-space, '
                                                       'collapsed-space) tuple form.',
                                           'explanation': 'dirichletbc needs to know '
                                                          'which dofs of the mixed '
                                                          'space to constrain AND '
                                                          'which entries of the value '
                                                          'Function supply the data; '
                                                          'the tuple form of '
                                                          'locate_dofs_* returns '
                                                          'exactly that pair of index '
                                                          'arrays.',
                                           'pitfalls': ['Never pass a raw numpy array '
                                                        'or a fem.Constant (with more '
                                                        'than one component), or a '
                                                        'collapsed-space Function with a '
                                                        'single (non-paired) dof array, '
                                                        'to dirichletbc on the velocity '
                                                        'sub-space W.sub(0). Signal: it '
                                                        'depends on the mistake. An '
                                                        'array or Constant with a single '
                                                        'dof array raises '
                                                        "'RuntimeError: Creating a "
                                                        'DirichletBC using a Constant is '
                                                        'not supported when the Constant '
                                                        'size is not equal to the block '
                                                        'size of the constrained '
                                                        '(sub-)space. Use a fem::Function '
                                                        "to create the fem::DirichletBC.'; "
                                                        'an array or Constant with the '
                                                        'paired dofs, or a Function with '
                                                        'a single dof array, raises '
                                                        "'TypeError: __init__(): "
                                                        'incompatible function '
                                                        'arguments. The following '
                                                        'argument types are '
                                                        "supported:' followed by four "
                                                        'overloads and a line '
                                                        "beginning 'Invoked with "
                                                        'types: '
                                                        "dolfinx.cpp.fem.DirichletBC_float64, ...' "
                                                        'that names the passed types '
                                                        "(e.g. 'ndarray, list' for an "
                                                        'array with paired dofs).',
                                                        'Pin the pressure whenever no '
                                                        'part of the boundary is left '
                                                        'traction-free. Signal: with '
                                                        'pc_type lu and '
                                                        'pc_factor_mat_solver_type '
                                                        'mumps the solve still '
                                                        "'converges' but the pressure "
                                                        'carries an arbitrary additive '
                                                        'constant that jumps '
                                                        'unpredictably between mesh '
                                                        "sizes; with PETSc's built-in "
                                                        'LU or ILU instead you get '
                                                        "'Linear <prefix> solve did "
                                                        'not converge due to '
                                                        'DIVERGED_PC_FAILED iterations '
                                                        "0', 'PC failed due to "
                                                        "FACTOR_NUMERIC_ZEROPIVOT', "
                                                        'SNES getConvergedReason() == '
                                                        '-3, and an all-zero '
                                                        'solution.']},
                   'solver': {'REQUIRED': 'Monolithic (steady or implicit transient) '
                                          '-- SNES via NonlinearProblem; '
                                          'petsc_options_prefix is a REQUIRED '
                                          'keyword-only argument on dolfinx 0.10:\n'
                                          '  from dolfinx.fem.petsc import '
                                          'NonlinearProblem\n'
                                          '  problem = NonlinearProblem(F, w, bcs=bcs, '
                                          "petsc_options_prefix='ns_',\n"
                                          "      petsc_options={'snes_type': "
                                          "'newtonls', 'snes_rtol': 1e-9,\n"
                                          "                     'snes_atol': 1e-11, "
                                          "'snes_max_it': 30,\n"
                                          "                     'ksp_type': 'preonly', "
                                          "'pc_type': 'lu',\n"
                                          '                     '
                                          "'pc_factor_mat_solver_type': 'mumps'})\n"
                                          '  problem.solve()                      # '
                                          'returns the solution Function\n'
                                          '  assert '
                                          'problem.solver.getConvergedReason() > 0   # '
                                          'REQUIRED, see pitfalls\n'
                                          'problem.solver is a petsc4py SNES: '
                                          'getConvergedReason(), getIterationNumber(), '
                                          'getFunctionNorm(), getKSP() are all '
                                          'available on it.\n'
                                          'IPCS -- three '
                                          'dolfinx.fem.petsc.LinearProblem objects, '
                                          'each with its own petsc_options_prefix and '
                                          'its own u= output Function; '
                                          'LinearProblem.solve() re-assembles matrix '
                                          'and rhs on every call, so time-dependent '
                                          'forms are correct.',
                              'OPTIONAL': 'On the monolithic Taylor-Hood saddle-point '
                                          'matrix only two linear-solver settings were '
                                          "found to work: ksp_type 'preonly' with "
                                          "pc_type 'lu' and pc_factor_mat_solver_type "
                                          "'mumps' (default choice), or the same with "
                                          "'superlu_dist'. snes_monitor, "
                                          'snes_converged_reason and '
                                          'ksp_converged_reason may be added as keys '
                                          "with value None to get PETSc's own progress "
                                          "text. 'snes_error_if_not_converged': True "
                                          'turns a diverged Newton into a raised '
                                          'exception instead of a silent bad answer. '
                                          "snes_linesearch_type may be 'bt' (default) "
                                          "or 'basic'; 'basic' removes the line search "
                                          'and is useful when doing continuation. The '
                                          'IPCS sub-problems are ordinary elliptic and '
                                          'mass systems, not saddle-point systems, and '
                                          'iterative solvers do work there: ksp_type '
                                          "'bcgs' + pc_type 'hypre' for the momentum "
                                          "step, 'cg' + 'hypre' for the pressure "
                                          "Poisson step and 'cg' + 'jacobi' for the "
                                          'mass-matrix projection step were executed '
                                          'and give the same answer as MUMPS LU.',
                              'explanation': 'dolfinx 0.10 drives all nonlinear solves '
                                             'through PETSc SNES; the old DOLFIN-era '
                                             'wrapper classes are either gone or need '
                                             'a different problem class.',
                              'pitfalls': ['Always check '
                                           'problem.solver.getConvergedReason() > 0 -- '
                                           'solve() does not raise on divergence. '
                                           'Signal: on a diverged Newton, '
                                           'problem.solve() returns a Function '
                                           'normally and only getConvergedReason() '
                                           'reveals the failure (e.g. -5 '
                                           'DIVERGED_MAX_IT, -6 DIVERGED_LINE_SEARCH, '
                                           '-9 DIVERGED_DTOL).',
                                           'Never wrap a '
                                           'dolfinx.fem.petsc.NonlinearProblem in '
                                           'dolfinx.nls.petsc.NewtonSolver. Signal: '
                                           '"AttributeError: \'NonlinearProblem\' '
                                           "object has no attribute 'a'. Did you mean: "
                                           '\'A\'?" raised from dolfinx/nls/petsc.py '
                                           'in NewtonSolver.__init__, followed by '
                                           '"Exception ignored ... AttributeError: '
                                           "'NewtonSolver' object has no attribute "
                                           '\'_A\'".',
                                           'Never name dolfinx.PETScKrylovSolver. '
                                           'Signal: "ImportError: cannot import name '
                                           '\'PETScKrylovSolver\' from \'dolfinx\'" '
                                           'and the same ImportError from '
                                           "'dolfinx.fem.petsc'; the class does not "
                                           'exist in any dolfinx module on 0.10.',
                                           'Do not hand the monolithic mixed matrix to '
                                           'a single-level preconditioner. Signal: '
                                           "pc_type 'lu' without "
                                           'pc_factor_mat_solver_type, and pc_type '
                                           "'ilu', both give KSP getConvergedReason() "
                                           '== -11 (DIVERGED_PCSETUP_FAILED) and an '
                                           'all-zero solution even when the pressure '
                                           "IS pinned; pc_type 'gamg' gives KSP "
                                           "DIVERGED_MAX_IT; pc_type 'hypre' lets the "
                                           'KSP converge but Newton then ends in SNES '
                                           '-6 DIVERGED_LINE_SEARCH; pc_type '
                                           "'fieldsplit' raises "
                                           '"petsc4py.PETSc.Error: error code 77 ... '
                                           'PCFieldSplitSetDefaults() ... Unhandled '
                                           'case, must have at least two fields, not '
                                           '1" because a monolithic mixed space is one '
                                           'field to PETSc.']},
                   'time_integration': {'REQUIRED': 'COMPLETE runnable IPCS '
                                                    '(incremental pressure correction) '
                                                    'transient solver -- run this '
                                                    'script unchanged:\n'
                                                    '----------------------------------------------------------------\n'
                                                    'from mpi4py import MPI\n'
                                                    'import numpy as np\n'
                                                    'import ufl\n'
                                                    'from dolfinx import mesh, fem\n'
                                                    'from dolfinx.fem.petsc import '
                                                    'LinearProblem\n'
                                                    '\n'
                                                    'msh = '
                                                    'mesh.create_rectangle(MPI.COMM_WORLD,\n'
                                                    '                            '
                                                    '[np.array([0.0, 0.0]), '
                                                    'np.array([2.0, 1.0])],\n'
                                                    '                            [32, '
                                                    '16], mesh.CellType.triangle)\n'
                                                    'gdim = msh.geometry.dim\n'
                                                    'V = fem.functionspace(msh, '
                                                    "('Lagrange', 2, (gdim,)))\n"
                                                    'Q = fem.functionspace(msh, '
                                                    "('Lagrange', 1))\n"
                                                    '\n'
                                                    'u_n, u_s, u_h = fem.Function(V), '
                                                    'fem.Function(V), fem.Function(V)\n'
                                                    'p_n, p_h = fem.Function(Q), '
                                                    'fem.Function(Q)\n'
                                                    'u, v = ufl.TrialFunction(V), '
                                                    'ufl.TestFunction(V)\n'
                                                    'p, q = ufl.TrialFunction(Q), '
                                                    'ufl.TestFunction(Q)\n'
                                                    'dt = fem.Constant(msh, 0.01)\n'
                                                    'nu = fem.Constant(msh, 0.01)\n'
                                                    'n = ufl.FacetNormal(msh)\n'
                                                    '\n'
                                                    'msh.topology.create_connectivity(gdim '
                                                    '- 1, gdim)\n'
                                                    'inlet = '
                                                    'mesh.locate_entities_boundary(msh, '
                                                    'gdim - 1,\n'
                                                    '                                      '
                                                    'lambda x: np.isclose(x[0], 0.0))\n'
                                                    'outlet = '
                                                    'mesh.locate_entities_boundary(msh, '
                                                    'gdim - 1,\n'
                                                    '                                       '
                                                    'lambda x: np.isclose(x[0], 2.0))\n'
                                                    'walls = '
                                                    'mesh.locate_entities_boundary(\n'
                                                    '    msh, gdim - 1, lambda x: '
                                                    'np.isclose(x[1], 0.0) | '
                                                    'np.isclose(x[1], 1.0))\n'
                                                    '\n'
                                                    'u_in = fem.Function(V)\n'
                                                    'u_in.interpolate(lambda x: '
                                                    'np.vstack((4.0 * x[1] * (1.0 - '
                                                    'x[1]),\n'
                                                    '                                      '
                                                    'np.zeros_like(x[0]))))\n'
                                                    'u_w = fem.Function(V)\n'
                                                    'u_w.x.array[:] = 0.0\n'
                                                    'bcu = [fem.dirichletbc(u_in, '
                                                    'fem.locate_dofs_topological(V, '
                                                    'gdim - 1, inlet)),\n'
                                                    '       fem.dirichletbc(u_w, '
                                                    'fem.locate_dofs_topological(V, '
                                                    'gdim - 1, walls))]\n'
                                                    'p_out = fem.Function(Q)\n'
                                                    'p_out.x.array[:] = 0.0\n'
                                                    'bcp = [fem.dirichletbc(p_out, '
                                                    'fem.locate_dofs_topological(Q, '
                                                    'gdim - 1, outlet))]\n'
                                                    '\n'
                                                    'F1 = (ufl.inner(u - u_n, v) / dt '
                                                    '* ufl.dx\n'
                                                    '      + ufl.inner(ufl.dot(u_n, '
                                                    'ufl.nabla_grad(u_n)), v) * '
                                                    'ufl.dx\n'
                                                    '      + nu * '
                                                    'ufl.inner(ufl.grad(u), '
                                                    'ufl.grad(v)) * ufl.dx\n'
                                                    '      - p_n * ufl.div(v) * '
                                                    'ufl.dx\n'
                                                    '      + p_n * ufl.dot(n, v) * '
                                                    'ufl.ds)\n'
                                                    'a1, L1 = ufl.lhs(F1), '
                                                    'ufl.rhs(F1)\n'
                                                    'a2 = ufl.inner(ufl.grad(p), '
                                                    'ufl.grad(q)) * ufl.dx\n'
                                                    'L2 = (ufl.inner(ufl.grad(p_n), '
                                                    'ufl.grad(q)) * ufl.dx\n'
                                                    '      - ufl.div(u_s) * q / dt * '
                                                    'ufl.dx)\n'
                                                    'a3 = ufl.inner(u, v) * ufl.dx\n'
                                                    'L3 = ufl.inner(u_s, v) * ufl.dx - '
                                                    'dt * ufl.inner(ufl.grad(p_h - '
                                                    'p_n), v) * ufl.dx\n'
                                                    '\n'
                                                    "lu = {'ksp_type': 'preonly', "
                                                    "'pc_type': 'lu',\n"
                                                    '      '
                                                    "'pc_factor_mat_solver_type': "
                                                    "'mumps'}\n"
                                                    's1 = LinearProblem(a1, L1, '
                                                    'bcs=bcu, u=u_s,\n'
                                                    '                   '
                                                    "petsc_options_prefix='ipcs1_', "
                                                    'petsc_options=lu)\n'
                                                    's2 = LinearProblem(a2, L2, '
                                                    'bcs=bcp, u=p_h,\n'
                                                    '                   '
                                                    "petsc_options_prefix='ipcs2_', "
                                                    'petsc_options=lu)\n'
                                                    's3 = LinearProblem(a3, L3, '
                                                    'bcs=[], u=u_h,\n'
                                                    '                   '
                                                    "petsc_options_prefix='ipcs3_', "
                                                    'petsc_options=lu)\n'
                                                    '\n'
                                                    'for step in range(200):\n'
                                                    '    s1.solve()\n'
                                                    '    s2.solve()\n'
                                                    '    s3.solve()\n'
                                                    '    u_n.x.array[:] = u_h.x.array\n'
                                                    '    p_n.x.array[:] = p_h.x.array\n'
                                                    '\n'
                                                    'flux = '
                                                    'msh.comm.allreduce(fem.assemble_scalar(\n'
                                                    '    fem.form(ufl.dot(u_h, n) * '
                                                    'ufl.ds(domain=msh))), '
                                                    'op=MPI.SUM)\n'
                                                    'divn = '
                                                    'np.sqrt(msh.comm.allreduce(fem.assemble_scalar(\n'
                                                    '    fem.form(ufl.div(u_h) ** 2 * '
                                                    'ufl.dx)), op=MPI.SUM))\n'
                                                    'umax = '
                                                    'msh.comm.allreduce(np.max(np.abs(u_h.x.array)), '
                                                    'op=MPI.MAX)\n'
                                                    "print(f'net boundary flux = "
                                                    '{flux:.3e}  (inflow == outflow, '
                                                    "must be ~0)')\n"
                                                    "print(f'||div u||_L2 = "
                                                    '{divn:.3e}   max|u| = '
                                                    '{umax:.4f}   '
                                                    "finite={np.all(np.isfinite(u_h.x.array))}')\n"
                                                    '----------------------------------------------------------------\n'
                                                    'Step 1 gives a tentative velocity '
                                                    'u_s, step 2 the pressure update, '
                                                    'step 3 the projected '
                                                    'divergence-free velocity u_h. The '
                                                    'three LinearProblem objects must '
                                                    'have DIFFERENT '
                                                    'petsc_options_prefix values.',
                                        'OPTIONAL': 'The convective term may be made '
                                                    'semi-implicit (ufl.dot(u_n, '
                                                    'ufl.nabla_grad(u))) to relax the '
                                                    'step-size restriction, at the '
                                                    'cost of re-factorising a '
                                                    'non-symmetric matrix every step. '
                                                    'theta-weighting of the viscous '
                                                    'term (Crank-Nicolson) raises the '
                                                    'temporal order of the viscous '
                                                    'part. For a fully implicit '
                                                    'alternative with no splitting '
                                                    'error, add ufl.inner((u - '
                                                    'u_n)/dt, v)*ufl.dx to the '
                                                    'monolithic residual in '
                                                    'minimal_working_example and '
                                                    're-solve the same '
                                                    'NonlinearProblem each step. As '
                                                    'written above the convective term '
                                                    'is an explicit Euler step, so the '
                                                    'temporal accuracy of the whole '
                                                    'scheme is first order in dt; the '
                                                    'monolithic implicit route has no '
                                                    'splitting error at all.',
                                        'explanation': 'IPCS decouples velocity and '
                                                       'pressure into three much '
                                                       'smaller and much better '
                                                       'conditioned systems, none of '
                                                       'which is a saddle-point '
                                                       'problem, at the price of a '
                                                       'splitting error and a '
                                                       'step-size restriction from the '
                                                       'explicit convection term.',
                                        'pitfalls': ['Keep the IPCS time step below '
                                                     'the convective CFL limit dt < '
                                                     'h/max|u|, and test stability '
                                                     'over MANY steps, not over a '
                                                     'short physical horizon. Signal: '
                                                     'with dt above the limit the '
                                                     'velocity grows from order 1 to '
                                                     'order 1e2-1e3 within the first '
                                                     '8-50 steps while every '
                                                     'LinearProblem still reports '
                                                     'CONVERGED_ITS -- there is no '
                                                     'solver error at all, only the '
                                                     'magnitude gives it away.']},
                   'materials': {'nu': {'range': [1e-06, 1.0],
                                        'unit': 'm^2/s',
                                        'description': 'Kinematic viscosity mu/rho; '
                                                       'for a unit-length, '
                                                       'unit-velocity domain nu = '
                                                       '1/Re'},
                                 'rho': {'range': [0.1, 2000.0],
                                         'unit': 'kg/m^3',
                                         'description': 'Density; absorbed into nu and '
                                                        'into the pressure in the '
                                                        'kinematic formulation above'},
                                 'Re': {'range': [1, 10000],
                                        'unit': 'dimensionless',
                                        'description': 'Reynolds number U*L/nu; steady '
                                                       'laminar 2D solutions exist '
                                                       'well past Re=1000 but Newton '
                                                       'needs continuation to reach '
                                                       'them'}},
                   'pitfalls': ['[API] Never wrap a dolfinx.fem.petsc.NonlinearProblem '
                                'in dolfinx.nls.petsc.NewtonSolver -- that pairing is '
                                'from dolfinx <= 0.8 and is broken on 0.10. Build '
                                'NonlinearProblem(F, w, bcs=bcs, '
                                "petsc_options_prefix='ns_', petsc_options={...}) and "
                                'call problem.solve() directly; problem.solver is the '
                                'petsc4py SNES. Signal: the constructor call '
                                'dolfinx.nls.petsc.NewtonSolver(comm, problem) raises '
                                '"AttributeError: \'NonlinearProblem\' object has no '
                                'attribute \'a\'. Did you mean: \'A\'?" from '
                                'dolfinx/nls/petsc.py inside NewtonSolver.__init__ '
                                '(line self._A = create_matrix(problem.a)), '
                                'immediately followed by "Exception ignored in: '
                                '<function NewtonSolver.__del__ ...> AttributeError: '
                                '\'NewtonSolver\' object has no attribute \'_A\'". '
                                'dolfinx.nls.petsc.NewtonSolver itself still exists '
                                'and still works, but only with the separate legacy '
                                'class '
                                'dolfinx.fem.petsc.NewtonSolverNonlinearProblem(F, u, '
                                'bcs=bcs) -- that pairing was executed and returns the '
                                'same solution as the SNES route. (Verified by '
                                'execution on dolfinx 0.10.0.)',
                                '[API] Never name dolfinx.PETScKrylovSolver (or '
                                'dolfinx.fem.petsc.PETScKrylovSolver). It is a legacy '
                                'DOLFIN name that exists in no dolfinx 0.10 module. To '
                                'reach the Krylov solver use '
                                'LinearProblem(...).solver, which IS a petsc4py KSP, '
                                'or problem.solver.getKSP() on a NonlinearProblem; '
                                'configure it through the petsc_options dict. Signal: '
                                '"ImportError: cannot import name '
                                '\'PETScKrylovSolver\' from \'dolfinx\'" and '
                                '"ImportError: cannot import name '
                                '\'PETScKrylovSolver\' from \'dolfinx.fem.petsc\'"; '
                                "hasattr(m, 'PETScKrylovSolver') is False for m in "
                                'dolfinx, dolfinx.fem, dolfinx.fem.petsc, dolfinx.nls, '
                                'dolfinx.nls.petsc, dolfinx.la, dolfinx.la.petsc, '
                                'dolfinx.cpp and dolfinx.cpp.fem. (Verified by '
                                'execution on dolfinx 0.10.0.)',
                                '[API] problem.solve() does NOT raise when Newton '
                                'fails -- it returns the solution Function whatever '
                                'happened, so an unchecked script silently reports '
                                'garbage as a result. Always assert '
                                'problem.solver.getConvergedReason() > 0 after every '
                                "solve, or pass 'snes_error_if_not_converged': True in "
                                'petsc_options. Signal: a lid-driven cavity that has '
                                'diverged still returns normally from problem.solve(); '
                                'getConvergedReason() is negative (-5 DIVERGED_MAX_IT, '
                                '-6 DIVERGED_LINE_SEARCH, -9 DIVERGED_DTOL) while the '
                                'returned velocity Function holds values ~1e3 times '
                                'the imposed lid speed. With '
                                "'snes_error_if_not_converged': True the same run "
                                'instead raises "petsc4py.PETSc.Error: error code 91 '
                                '... [0] SNESSolve() at .../snes/interface/snes.c ... '
                                '[0] SNESSolve has not converged". (Verified by '
                                'execution on dolfinx 0.10.0.)',
                                '[Performance] The monolithic Taylor-Hood matrix is a '
                                'saddle-point matrix with zeros on the pressure '
                                'diagonal, so a plain single-level preconditioner '
                                "cannot handle it. Use ksp_type 'preonly' with pc_type "
                                "'lu' and pc_factor_mat_solver_type 'mumps' (or "
                                "'superlu_dist'); those two were executed "
                                'successfully, everything else tried was not. Signal: '
                                'on an otherwise correct, pressure-pinned lid-driven '
                                "cavity, pc_type 'lu' WITHOUT "
                                "pc_factor_mat_solver_type and pc_type 'ilu' both give "
                                'KSP getConvergedReason() == -11 '
                                '(DIVERGED_PCSETUP_FAILED), SNES -3, and an all-zero '
                                "solution; pc_type 'gamg' gives KSP -3 "
                                "(DIVERGED_MAX_IT); pc_type 'hypre' converges the KSP "
                                'but Newton then stops at SNES -6 '
                                "(DIVERGED_LINE_SEARCH); and pc_type 'fieldsplit' "
                                'raises "petsc4py.PETSc.Error: error code 77 ... [0] '
                                'PCFieldSplitSetDefaults() ... [0] PETSc has generated '
                                'inconsistent data [0] Unhandled case, must have at '
                                'least two fields, not 1", because a single mixed '
                                'FunctionSpace looks like one field to PETSc -- '
                                'Schur-complement splitting needs block/nest assembly, '
                                'not the monolithic mixed space. The IPCS '
                                'sub-problems, by contrast, are ordinary elliptic and '
                                'mass systems and do accept iterative solvers. '
                                '(Verified by execution on dolfinx 0.10.0.)',
                                '[Numerical] Use an inf-sup (LBB) stable '
                                'velocity/pressure pair -- Taylor-Hood '
                                'basix.ufl.mixed_element([P_k_vector, P_(k-1)_scalar]) '
                                'with k >= 2. Equal-order pairs fail, and HOW they '
                                'fail depends on the cell type, so a clean KSP report '
                                'is not evidence that the pair is stable. Signal: '
                                'P1/P1 on TRIANGLES with pc_type lu + mumps fails in '
                                'the first linear solve: KSP getConvergedReason() == '
                                '-11 (with ksp_converged_reason set PETSc prints '
                                '"Linear <prefix> solve did not converge due to '
                                'DIVERGED_PC_FAILED iterations 0" and "PC failed due '
                                'to FACTOR_NUMERIC_ZEROPIVOT"), SNES '
                                'getConvergedReason() == -3, and the returned arrays '
                                'are all zero. P1/P1 on QUADRILATERALS, and P2/P2 on '
                                'both cell types, report KSP CONVERGED_ITS on every '
                                'step and print nothing by default, but Newton ends '
                                'with SNES getConvergedReason() == -6 '
                                '(DIVERGED_LINE_SEARCH), with max|u| 2-9x the imposed '
                                'lid speed (which is an upper bound for the true '
                                'solution) and max|p| of order 1e16-1e18. The stable '
                                'P2/P1 pair on the identical setup gives max|u| '
                                'exactly equal to the lid speed. Check the SNES '
                                'reason. (Verified by execution on dolfinx 0.10.0 with '
                                'PETSc 3.24.5, with the pairs (2,1), (1,1) and (2,2) '
                                'on triangles and on quadrilaterals, at two mesh '
                                'resolutions each.)',
                                '[Physics] An ENCLOSED domain -- every boundary facet '
                                'carrying a velocity Dirichlet BC, e.g. a lid-driven '
                                'cavity -- leaves the pressure determined only up to '
                                'an additive constant, because pressure enters the '
                                'weak form only through its gradient. Fix it by '
                                'pinning one pressure dof: Q, _ = W.sub(1).collapse(); '
                                'p_ref = fem.Function(Q); p_ref.x.array[:] = 0.0; '
                                'pdofs = fem.locate_dofs_geometrical((W.sub(1), Q), '
                                'lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], '
                                '0.0)); bcs.append(fem.dirichletbc(p_ref, pdofs, '
                                "W.sub(1))). Any domain with a traction-free ('do "
                                "nothing') outflow boundary does NOT need this. "
                                'Signal: the failure mode depends entirely on the '
                                'linear solver. With pc_type lu + '
                                'pc_factor_mat_solver_type mumps the singular system '
                                'is factorised anyway, SNES reports '
                                'CONVERGED_FNORM_ABS, the velocity is correct to '
                                'machine precision, and only the pressure is wrong -- '
                                'it differs from the pinned solution by a pure '
                                'additive constant whose value jumps unpredictably '
                                '(both signs, magnitudes spanning several orders) as '
                                'the mesh is refined or the element degree changed, so '
                                'nothing in the solver output flags the problem. With '
                                "PETSc's built-in LU or with pc_type ilu the same "
                                'system instead prints "Linear <prefix> solve did not '
                                'converge due to DIVERGED_PC_FAILED iterations 0" and '
                                '"PC failed due to FACTOR_NUMERIC_ZEROPIVOT", KSP '
                                'getConvergedReason() == -11, SNES '
                                'getConvergedReason() == -3, and the returned velocity '
                                'and pressure are all zero. (Verified by execution on '
                                'dolfinx 0.10.0; reproduced on triangles and '
                                'quadrilaterals at P2/P1 and P3/P2.)',
                                '[Numerical] Newton from a zero initial guess is not '
                                'robust at higher Re on a lid-driven cavity, and the '
                                'outcome depends on the line search and on the mesh -- '
                                'the cure is continuation: solve at a low Re, keep the '
                                'solution Function as the initial guess (do not zero '
                                'it), raise Re, solve again. Only nu.value has to '
                                'change between solves; the same NonlinearProblem '
                                "object can be reused. Signal: with the default 'bt' "
                                'line search a cold start at Re = 1000 is '
                                'mesh-dependent -- it converged on 16x16, 32x32 and '
                                '64x64 meshes but stopped with DIVERGED_LINE_SEARCH '
                                '(getConvergedReason() == -6) on 8x8 and 24x24, while '
                                'cold starts at Re = 500, 600 and 800 converged on '
                                "16x16 and 32x32. With snes_linesearch_type 'basic' "
                                'the cold start already fails at Re = 600 on 16x16, '
                                'and at Re = 1000 it stops with DIVERGED_DTOL '
                                '(getConvergedReason() == -9; with '
                                'snes_converged_reason set PETSc prints "Nonlinear '
                                '<prefix> solve did not converge due to DIVERGED_DTOL '
                                'iterations N") on every mesh tried, with max|u| of '
                                'order 1e3-1e4 against an imposed lid speed of 1; '
                                'refining does not rescue it. CORRECTION: an earlier '
                                'version of this entry said doubling the resolution '
                                'makes the blow-up LARGER; the magnitude is not '
                                'monotone in mesh size, so do not use it as the test. '
                                'The observable is the negative converged reason, not '
                                'how big max|u| got. Stepping Re through 100, 200, '
                                '400, 600, 800, 1000, 1500, 2000, 3000, 5000 with the '
                                'previous solution retained converged at every stage '
                                'in a handful of Newton iterations with max|u| staying '
                                'exactly at the lid speed on a 32x32 mesh; on 16x16 it '
                                'converged up to Re = 2000 and then stopped with '
                                'DIVERGED_LINE_SEARCH (-6) at Re = 3000 and 5000. '
                                '(Verified by execution on dolfinx 0.10.0 with PETSc '
                                '3.24.5.)',
                                '[API] A Dirichlet BC on a sub-space of a mixed space '
                                'needs a Function on the COLLAPSED sub-space plus the '
                                'tuple form of locate_dofs_*: V, _ = '
                                'W.sub(0).collapse(); u_bc = fem.Function(V); '
                                'u_bc.interpolate(...); dofs = '
                                'fem.locate_dofs_topological((W.sub(0), V), gdim-1, '
                                'facets); bc = fem.dirichletbc(u_bc, dofs, W.sub(0)). '
                                'Signal: passing a raw numpy array, a fem.Constant, or '
                                'a Function together with a single (non-tuple) dof '
                                'array all raise the same error, which begins '
                                '"TypeError: __init__(): incompatible function '
                                'arguments. The following argument types are '
                                'supported:", then lists four overloads, then ends '
                                'with a line beginning "Invoked with types: '
                                'dolfinx.cpp.fem.DirichletBC_float64,". A Function '
                                'built on the UNCOLLAPSED mixed space W does NOT raise '
                                'here, so the absence of a TypeError is not proof that '
                                'the BC is right. (Verified by execution on dolfinx '
                                '0.10.0.)',
                                '[Numerical] The IPCS scheme above treats convection '
                                'EXPLICITLY, so its time step obeys a convective CFL '
                                'condition -- dt must stay below h/max|u| (h = cell '
                                'size), and measurably further below it than that '
                                'formula suggests. Exceeding it makes the run blow up '
                                'with no solver complaint whatsoever, and -- the part '
                                'that catches people out -- a short test run will not '
                                'reveal it, because a large dt reaches the chosen end '
                                'time in only a handful of steps. Always judge '
                                'stability by a FIXED NUMBER OF STEPS, not by a fixed '
                                'final time. Signal: on a channel meshed at h = 0.125 '
                                'with peak inflow speed 1, running 120 steps at dt = '
                                '0.5 / 0.2 / 0.125 / 0.1 / 0.05 first drives max|u| '
                                'past 1e2 at (0-based) step 8 / 11 / 15 / 18 / 51 '
                                'respectively, with max|u| between 1.8e2 and 6.8e2 at '
                                'that step, and the field then overflows to inf/NaN '
                                'about eight steps later, long before step 120 -- '
                                'while all three LinearProblem.solver objects keep '
                                'reporting CONVERGED_ITS on every single step, also '
                                'after the field is NaN. On that same mesh dt = 0.02 '
                                'survives all 120 steps with max|u| staying at '
                                '1.000000, so the threshold there lies between 0.02 '
                                'and 0.05, i.e. well below h itself. The exact same dt '
                                'values look perfectly healthy (max|u| between 1.03 '
                                'and 1.57) when the loop is stopped at a fixed end '
                                'time of 0.6, because that is only 1 to 12 steps. The '
                                'cure that was executed is simply a smaller dt; the '
                                'standard alternatives are a semi-implicit convective '
                                'term (ufl.dot(u_n, ufl.nabla_grad(u)), which moves '
                                'the convection into the matrix) or the monolithic '
                                'implicit route, which has no explicit term to limit '
                                'at all. Refining dt below the limit keeps changing '
                                'the answer (it does not saturate), so the splitting '
                                'error is still the dominant error there. (Verified by '
                                'execution on dolfinx 0.10.0.)',
                                '[API] XDMFFile.write_function only accepts a Function '
                                'whose degree matches the mesh geometry degree, so a '
                                'P2 velocity cannot be written directly. Either '
                                'interpolate into a matching P1 space first (u1 = '
                                "fem.Function(fem.functionspace(msh, ('Lagrange', 1, "
                                '(gdim,)))); u1.interpolate(u2)), or use '
                                'dolfinx.io.VTXWriter or dolfinx.io.VTKFile, both of '
                                'which take the P2 Function unchanged. Signal: '
                                '"RuntimeError: Degree of output Function must be same '
                                'as mesh degree. Maybe the Function needs to be '
                                'interpolated?". (Verified by execution on dolfinx '
                                '0.10.0: the error fires for scalar and vector '
                                'Lagrange at degree 2 and degree 3 on an affine mesh, '
                                'and does not fire at degree 1; all three fixes were '
                                'executed and wrote successfully.)']},
 'heat': {'description': 'Heat conduction in dolfinx 0.10, steady or transient. '
                         'Fourier: rho*cp*dT/dt - div(k*grad(T)) = Q, with k the '
                         'conductivity [W/(m*K)] and rho*cp the volumetric heat '
                         'capacity [J/(m^3*K)]. The transient case is advanced with a '
                         'theta-method (theta=1 backward Euler, theta=0.5 '
                         'Crank-Nicolson); the stiffness+mass matrix does not change '
                         'in time, so it is assembled once and only the right-hand '
                         'side is rebuilt each step.',
          'minimal_working_example': '\n'
                                     '# Transient heat on the unit square with a '
                                     'theta-method.\n'
                                     '# Two-material conductivity, a time-dependent '
                                     'Dirichlet wall, a prescribed\n'
                                     '# inflow wall, a convective (Robin) wall and one '
                                     'insulated wall.\n'
                                     '# Runs unchanged; prints two self-checks that '
                                     'need no reference solution.\n'
                                     'from mpi4py import MPI\n'
                                     'import numpy as np\n'
                                     'import ufl\n'
                                     'from dolfinx import fem, la, mesh\n'
                                     'from dolfinx.fem import petsc as fp\n'
                                     'from petsc4py import PETSc\n'
                                     '\n'
                                     'msh = mesh.create_unit_square(MPI.COMM_WORLD, '
                                     '24, 24, mesh.CellType.triangle)\n'
                                     'V = fem.functionspace(msh, ("Lagrange", 1))\n'
                                     'u, v = ufl.TrialFunction(V), '
                                     'ufl.TestFunction(V)\n'
                                     'T_n, T_h = fem.Function(V, name="T"), '
                                     'fem.Function(V)\n'
                                     'T_n.x.array[:] = 0.0\n'
                                     '\n'
                                     'Vk = fem.functionspace(msh, ("DG", 0))\n'
                                     'k = fem.Function(Vk)\n'
                                     'k.interpolate(lambda x: np.where(x[0] < 0.5, '
                                     '1.0, 10.0))\n'
                                     'rho_cp, Q = fem.Constant(msh, 2.0), '
                                     'fem.Constant(msh, 5.0)\n'
                                     'q_in, h_c, T_inf = fem.Constant(msh, 2.0), '
                                     'fem.Constant(msh, 3.0), fem.Constant(msh, 0.0)\n'
                                     'dt_val = 0.01\n'
                                     'dt, theta = fem.Constant(msh, dt_val), '
                                     'fem.Constant(msh, 1.0)\n'
                                     '\n'
                                     'fdim = msh.topology.dim - 1\n'
                                     'where = {1: lambda x: np.isclose(x[0], 0.0), 2: '
                                     'lambda x: np.isclose(x[0], 1.0),\n'
                                     '         3: lambda x: np.isclose(x[1], 1.0)}\n'
                                     'ents = [mesh.locate_entities_boundary(msh, fdim, '
                                     'f) for f in where.values()]\n'
                                     'vals = [np.full(len(e), m, dtype=np.int32) for '
                                     'm, e in zip(where, ents)]\n'
                                     'ents, vals = np.concatenate(ents), '
                                     'np.concatenate(vals)\n'
                                     'srt = np.argsort(ents)\n'
                                     'ft = mesh.meshtags(msh, fdim, ents[srt], '
                                     'vals[srt])\n'
                                     'ds = ufl.Measure("ds", domain=msh, '
                                     'subdomain_data=ft)\n'
                                     '\n'
                                     '\n'
                                     'def spatial(T):\n'
                                     '    return (ufl.inner(k * ufl.grad(T), '
                                     'ufl.grad(v)) * ufl.dx\n'
                                     '            + h_c * (T - T_inf) * v * ds(2))\n'
                                     '\n'
                                     '\n'
                                     'F = ((rho_cp / dt) * (u - T_n) * v * ufl.dx\n'
                                     '     + theta * spatial(u) + (1.0 - theta) * '
                                     'spatial(T_n)\n'
                                     '     - Q * v * ufl.dx - q_in * v * ds(3))\n'
                                     'a_f, L_f = fem.form(ufl.lhs(F)), '
                                     'fem.form(ufl.rhs(F))\n'
                                     'res_f = fem.form(ufl.replace(F, {u: T_h}))\n'
                                     '\n'
                                     'g_D = fem.Constant(msh, 0.0)\n'
                                     'dofs_D = fem.locate_dofs_topological(V, fdim, '
                                     'ft.find(1))\n'
                                     'bc = fem.dirichletbc(g_D, dofs_D, V)\n'
                                     '\n'
                                     'A = fp.assemble_matrix(a_f, bcs=[bc])\n'
                                     'A.assemble()\n'
                                     'b = fp.create_vector(V)\n'
                                     'ksp = PETSc.KSP().create(msh.comm)\n'
                                     'ksp.setOperators(A)\n'
                                     'ksp.setType("preonly")\n'
                                     'ksp.getPC().setType("lu")\n'
                                     '\n'
                                     'nloc = V.dofmap.index_map.size_local * '
                                     'V.dofmap.index_map_bs\n'
                                     'free = np.setdiff1d(np.arange(nloc), '
                                     'dofs_D[dofs_D < nloc])\n'
                                     'U_f, Un_f = fem.form(rho_cp * T_h * ufl.dx), '
                                     'fem.form(rho_cp * T_n * ufl.dx)\n'
                                     'rob_f = fem.form(h_c * (theta * (T_h - T_inf) + '
                                     '(1.0 - theta) * (T_n - T_inf)) * ds(2))\n'
                                     'src = '
                                     'msh.comm.allreduce(fem.assemble_scalar(fem.form(Q '
                                     '* ufl.dx(domain=msh))), op=MPI.SUM)\n'
                                     'neu = '
                                     'msh.comm.allreduce(fem.assemble_scalar(fem.form(q_in '
                                     '* ds(3))), op=MPI.SUM)\n'
                                     '\n'
                                     't = 0.0\n'
                                     'for _ in range(40):\n'
                                     '    t += dt_val\n'
                                     '    g_D.value = 1.0 + 2.0 * t\n'
                                     '    with b.localForm() as loc:\n'
                                     '        loc.set(0.0)\n'
                                     '    fp.assemble_vector(b, L_f)\n'
                                     '    fp.apply_lifting(b, [a_f], bcs=[[bc]])\n'
                                     '    b.ghostUpdate(addv=PETSc.InsertMode.ADD, '
                                     'mode=PETSc.ScatterMode.REVERSE)\n'
                                     '    fp.set_bc(b, [bc])\n'
                                     '    ksp.solve(b, T_h.x.petsc_vec)\n'
                                     '    T_h.x.scatter_forward()\n'
                                     '    Un = '
                                     'msh.comm.allreduce(fem.assemble_scalar(Un_f), '
                                     'op=MPI.SUM)\n'
                                     '    U = '
                                     'msh.comm.allreduce(fem.assemble_scalar(U_f), '
                                     'op=MPI.SUM)\n'
                                     '    rob = '
                                     'msh.comm.allreduce(fem.assemble_scalar(rob_f), '
                                     'op=MPI.SUM)\n'
                                     '    r = fem.assemble_vector(res_f)\n'
                                     '    r.scatter_reverse(la.InsertMode.add)\n'
                                     '    free_res = '
                                     'msh.comm.allreduce(float(np.abs(r.array[free]).max()), '
                                     'op=MPI.MAX)\n'
                                     '    wall_in = '
                                     'msh.comm.allreduce(float(np.sum(r.array[:nloc])), '
                                     'op=MPI.SUM)\n'
                                     '    T_n.x.array[:] = T_h.x.array\n'
                                     '\n'
                                     'print(f"KSP converged reason = '
                                     '{ksp.getConvergedReason()} (>0 means '
                                     'converged)")\n'
                                     'print(f"t_end={t:.3f}  imposed wall temperature '
                                     'g_D={float(g_D.value):.4f}")\n'
                                     'print(f"T range=[{T_h.x.array.min():.6f}, '
                                     '{T_h.x.array.max():.6f}]  "\n'
                                     '      f"all '
                                     'finite={bool(np.all(np.isfinite(T_h.x.array)))}")\n'
                                     'print(f"CHECK 1  residual at the free dofs = '
                                     '{free_res:.3e}   (must be <= ~1e-12)")\n'
                                     'print(f"stored-energy rate dU/dt = {(U - Un) / '
                                     'dt_val:.8f}")\n'
                                     'print(f"source {src:.6f} + inflow {neu:.6f} + '
                                     'wall {wall_in:.6f} - convected {rob:.6f}")\n'
                                     'print(f"CHECK 2  energy imbalance = "\n'
                                     '      f"{(U - Un) / dt_val - (src + neu + '
                                     'wall_in - rob):.3e}   (must be <= ~1e-12)")\n',
          'function_space': {'REQUIRED': 'V = fem.functionspace(msh, ("Lagrange", '
                                         '1))            # temperature\n'
                                         'Vk = fem.functionspace(msh, ("DG", '
                                         '0))                 # conductivity field\n'
                                         'k = fem.Function(Vk)\n'
                                         'k.interpolate(lambda x: np.where(x[0] < 0.5, '
                                         '1.0, 10.0))',
                             'OPTIONAL': 'Degree is optional: ("Lagrange", 1) or '
                                         '("Lagrange", 2) both work; P_k converges at '
                                         'order k+1 in L2 for a smooth solution. Cell '
                                         'type is optional: mesh.CellType.triangle or '
                                         'mesh.CellType.quadrilateral in 2-D '
                                         '(create_unit_square), tetrahedron or '
                                         'hexahedron in 3-D (create_unit_cube); every '
                                         'statement in this entry was checked on '
                                         'triangles and on quadrilaterals at degree 1 '
                                         'and degree 2. The conductivity space is '
                                         'optional: ("DG", 0) for piecewise-constant '
                                         'materials, ("Lagrange", 1) for a smoothly '
                                         'varying field, or just fem.Constant(msh, '
                                         'value) if k is uniform.',
                             'explanation': 'The temperature is a scalar continuous '
                                            'field, so a Lagrange space is the right '
                                            'choice. A conductivity that varies in '
                                            'space is a Function on its own space, '
                                            'never a Constant: fem.Constant only holds '
                                            'a fixed number and rejects anything '
                                            'callable or symbolic.',
                             'pitfalls': ['Do not pass a callable or a UFL expression '
                                          'to fem.Constant for a spatially varying k; '
                                          "use fem.Function on a ('DG', 0) space and "
                                          '.interpolate. Signal: RuntimeError: '
                                          'Unsupported dtype',
                                          'Do not pass a Python int to fem.Constant. '
                                          'Signal: fem.Constant(msh, 1) raises '
                                          'RuntimeError: Unsupported dtype; '
                                          'fem.Constant(msh, 1.0) works.',
                                          'Do not use the legacy names '
                                          'fem.FunctionSpace / '
                                          'fem.VectorFunctionSpace. Signal: '
                                          "fem.FunctionSpace(msh, ('Lagrange', 1)) "
                                          'raises TypeError: FunctionSpace.__init__() '
                                          'missing 1 required positional argument: '
                                          "'cppV', and hasattr(fem, "
                                          "'VectorFunctionSpace') is False; "
                                          'fem.functionspace (lower case s) is the '
                                          'only spelling that works.']},
          'weak_form': {'REQUIRED': '# theta-method written as ONE residual, then '
                                    'split by UFL:\n'
                                    'F = ((rho_cp / dt) * (u - T_n) * v * ufl.dx\n'
                                    '     + theta * spatial(u) + (1.0 - theta) * '
                                    'spatial(T_n)\n'
                                    '     - Q * v * ufl.dx - q_in * v * ds(3))\n'
                                    'a_f, L_f = fem.form(ufl.lhs(F)), '
                                    'fem.form(ufl.rhs(F))\n'
                                    '# where spatial(T) = ufl.inner(k*ufl.grad(T), '
                                    'ufl.grad(v))*ufl.dx\n'
                                    '#                    + h_c*(T - T_inf)*v*ds(2)',
                        'OPTIONAL': 'theta is optional and must lie in [0, 1]: '
                                    'theta=1.0 is backward Euler (first order, '
                                    'monotone), theta=0.5 is Crank-Nicolson (second '
                                    'order, rings on sharp transients), theta=0.0 is '
                                    'forward Euler (conditionally stable, needs a very '
                                    'small dt). The source Q and the inflow q_in are '
                                    'optional (drop the terms if zero). For a STEADY '
                                    'problem drop the whole (rho_cp/dt)*(u - T_n) term '
                                    'and set theta=1: a = ufl.inner(k*ufl.grad(u), '
                                    'ufl.grad(v))*ufl.dx ; L = Q*v*ufl.dx + '
                                    'q_in*v*ds(3).',
                        'explanation': 'Writing the whole step as one residual F and '
                                       'letting ufl.lhs / ufl.rhs split it makes it '
                                       'impossible to lose a term or to get the sign '
                                       'of the old-time contribution wrong, and '
                                       'ufl.replace(F, {u: T_h}) then gives the '
                                       'matching residual form for free. The '
                                       'time-derivative term contributes a mass matrix '
                                       '(rho_cp/dt)*u*v*dx on the left and '
                                       '(rho_cp/dt)*T_n*v*dx on the right, with the '
                                       'SAME positive sign on both sides.',
                        'pitfalls': ['Do not divide a Form by a Constant. Signal: u * '
                                     'v * ufl.dx / dt raises TypeError: unsupported '
                                     "operand type(s) for /: 'Form' and 'Constant'; "
                                     'write (u / dt) * v * ufl.dx instead.',
                                     'Do not drop the mass term from the left-hand '
                                     'side. Signal: with only the stiffness matrix and '
                                     'no Dirichlet bc the matrix is singular, LU '
                                     'raises nothing, and max|T| is ~3e15 after one '
                                     'step, growing by a factor ~3e15 per step to nan.',
                                     'Do not flip the sign of the stiffness term. '
                                     'Signal: on an adiabatic cell with uniform T0=1 '
                                     'the field stays at 1.000000 for four steps and '
                                     'then grows geometrically by ~40-78x per step, '
                                     'reaching 1e17-1e23 by step 20.',
                                     'Crank-Nicolson (theta=0.5) rings in TIME on a '
                                     'sharp transient: the nodal value next to the '
                                     'transient alternates up-down every step. Signal: '
                                     '18 sign reversals in a 20-step run with a '
                                     'sawtooth amplitude 2-9% of the imposed jump, '
                                     'while backward Euler shows exactly 0 sign '
                                     'reversals.']},
          'boundary_conditions': {'REQUIRED': '# 1. tag the boundary pieces you need a '
                                              'ds() term on\n'
                                              'fdim = msh.topology.dim - 1\n'
                                              'ents = '
                                              'mesh.locate_entities_boundary(msh, '
                                              'fdim, lambda x: np.isclose(x[0], 0.0))\n'
                                              'ft = mesh.meshtags(msh, fdim, ents, '
                                              'np.full(len(ents), 1, dtype=np.int32))\n'
                                              'ds = ufl.Measure("ds", domain=msh, '
                                              'subdomain_data=ft)\n'
                                              '\n'
                                              '# 2. Dirichlet (prescribed '
                                              'temperature), possibly time dependent\n'
                                              'g_D = fem.Constant(msh, 0.0)\n'
                                              'bc = fem.dirichletbc(g_D, '
                                              'fem.locate_dofs_topological(V, fdim, '
                                              'ft.find(1)), V)\n'
                                              'g_D.value = 1.0 + 2.0 * t          # '
                                              'inside the time loop, EVERY step\n'
                                              '\n'
                                              '# 3. Neumann (prescribed inflow q_in, '
                                              'W/m^2 entering the domain)\n'
                                              '#    add   - q_in * v * ds(3)   to the '
                                              'residual F\n'
                                              '# 4. Robin / convection to an ambient '
                                              'T_inf with film coefficient h_c\n'
                                              '#    add   + h_c * (T - T_inf) * v * '
                                              'ds(2)   to the residual F\n'
                                              '# 5. Insulated / adiabatic / symmetry '
                                              'wall: write NOTHING for it',
                                  'OPTIONAL': 'Any subset of Dirichlet / Neumann / '
                                              'Robin / insulated may be used; every '
                                              'wall not named in a Dirichlet bc and '
                                              'not carrying a ds() term is '
                                              'automatically a zero-flux wall. The '
                                              'locator may also be '
                                              'mesh.exterior_facet_indices(msh.topology) '
                                              'for "the whole boundary", but that call '
                                              'needs '
                                              'msh.topology.create_connectivity(fdim, '
                                              'msh.topology.dim) first. The Dirichlet '
                                              'value may be a fem.Constant (uniform) '
                                              'or a fem.Function (spatially varying); '
                                              'a Function is updated with '
                                              'g.interpolate(...) instead of g.value = '
                                              '...',
                                  'explanation': 'The div(k*grad(T)) term integrates '
                                                 'by parts into a volume term plus a '
                                                 'boundary flux term, so a wall with '
                                                 'no boundary term already has zero '
                                                 'heat flux. Prescribing T=0 on a wall '
                                                 'that is meant to be insulated is '
                                                 'therefore not a harmless extra '
                                                 'constraint but a different physical '
                                                 'problem. A Dirichlet value that '
                                                 'changes in time must be written into '
                                                 'the SAME Constant object the bc was '
                                                 'built from.',
                                  'pitfalls': ['Do not put a Dirichlet bc on an '
                                               'insulated wall. Signal: left wall at '
                                               'T=1 with the other walls left alone '
                                               'gives max(T)=min(T)=1.000000 and a net '
                                               'flux through those walls of -1.3e-15; '
                                               'adding T=0 there gives min(T)=0.0, '
                                               'mean(T)=0.270 and 5.25 units of heat '
                                               'leaking out of a wall that should '
                                               'carry none.',
                                               'Write the new Dirichlet value into '
                                               'Constant.value; do not rebind the '
                                               'Python name. Signal: with g_D.value = '
                                               'g(t) the final max(T) is 2.000000 = '
                                               'g(t_end); with the update missing, or '
                                               'with g_D = fem.Constant(msh, g(t)) '
                                               'rebinding the name, max(T) is 1.000000 '
                                               '= g(t_start) and the two wrong '
                                               'variants are bit-identical.',
                                               'ds(marker) without subdomain_data '
                                               'silently integrates nothing. Signal: '
                                               'fem.assemble_scalar(fem.form(fem.Constant(msh,1.0)*ufl.ds(3))) '
                                               'returns 0.0 with no error; the same '
                                               "integrand against ufl.Measure('ds', "
                                               'domain=msh, subdomain_data=ft) returns '
                                               '1.0. A tag value absent from the '
                                               'meshtags also returns 0.0 silently.',
                                               'mesh.exterior_facet_indices needs the '
                                               'facet-cell connectivity first. Signal: '
                                               'RuntimeError: Facet to cell '
                                               'connectivity has not been computed.']},
          'solver': {'REQUIRED': '# LHS is time independent -> assemble ONCE, outside '
                                 'the loop\n'
                                 'A = fp.assemble_matrix(a_f, bcs=[bc])\n'
                                 'A.assemble()\n'
                                 'b = fp.create_vector(V)\n'
                                 'ksp = PETSc.KSP().create(msh.comm)\n'
                                 'ksp.setOperators(A)\n'
                                 'ksp.setType("preonly")\n'
                                 'ksp.getPC().setType("lu")\n'
                                 '\n'
                                 '# inside the loop: rebuild ONLY the right-hand side\n'
                                 'with b.localForm() as loc:\n'
                                 '    loc.set(0.0)\n'
                                 'fp.assemble_vector(b, L_f)\n'
                                 'fp.apply_lifting(b, [a_f], bcs=[[bc]])\n'
                                 'b.ghostUpdate(addv=PETSc.InsertMode.ADD, '
                                 'mode=PETSc.ScatterMode.REVERSE)\n'
                                 'fp.set_bc(b, [bc])\n'
                                 'ksp.solve(b, T_h.x.petsc_vec)\n'
                                 'T_h.x.scatter_forward()\n'
                                 'T_n.x.array[:] = T_h.x.array      # LAST line of the '
                                 'loop body',
                     'OPTIONAL': 'The steady problem, or a transient one where clarity '
                                 'beats speed, may instead use\n'
                                 '    from dolfinx.fem.petsc import LinearProblem\n'
                                 '    problem = LinearProblem(a, L, bcs=[bc], '
                                 'petsc_options_prefix="heat_",\n'
                                 '                            '
                                 'petsc_options={"ksp_type": "preonly", "pc_type": '
                                 '"lu"})\n'
                                 '    T_h = problem.solve()\n'
                                 'petsc_options_prefix is REQUIRED (keyword-only) and '
                                 'must end in "_". Solver choice is optional: ksp_type '
                                 '"preonly" + pc_type "lu" for small problems, '
                                 'ksp_type "cg" + pc_type "hypre" or "gamg" for large '
                                 'symmetric ones (the heat operator is symmetric '
                                 'positive definite). Check convergence with '
                                 'ksp.getConvergedReason() > 0 (or '
                                 'problem.solver.getConvergedReason()).',
                     'explanation': 'The mass+stiffness matrix does not depend on '
                                    'time, so assembling it once and reusing the '
                                    'factorisation is the whole point of the pattern; '
                                    'LinearProblem.solve() re-assembles and '
                                    're-factorises the matrix on every call. '
                                    'apply_lifting moves the known Dirichlet values to '
                                    'the right-hand side and set_bc writes them into '
                                    'the constrained rows; both are needed for the '
                                    'manual pattern.',
                     'pitfalls': ['Copy the new solution into the old one at the end '
                                  'of every step. Signal: with T_n.x.array[:] = '
                                  'T_h.x.array missing, int(T dx) is 0.100406 at steps '
                                  '1, 2, 3, 10 and 30 (the field is bit-identical '
                                  'every step) instead of 0.100406 / 0.150204 / '
                                  '0.187653 / 0.352449 / 0.609701; KSP still reports '
                                  'converged.',
                                  'Do not omit apply_lifting. Signal: the KSP still '
                                  'reports converged reason 4 and the field is finite, '
                                  'but the residual of the step form at the free dofs '
                                  'is 1.696 instead of 4.9e-14 and the temperature '
                                  'range is wrong.',
                                  'Do not rebuild LinearProblem, or call .solve() on '
                                  'one, inside the time loop. Signal: 100 steps on a '
                                  '64x64 P1 mesh take 0.151 s assembling once, 1.233 s '
                                  'with one LinearProblem re-solved each step (8.2x), '
                                  'and 3.138 s with a fresh LinearProblem each step '
                                  '(20.8x); all three give the identical answer, so '
                                  'only the wall-clock time exposes it.',
                                  'petsc_options_prefix is a required keyword '
                                  'argument. Signal: TypeError: '
                                  'LinearProblem.__init__() missing 1 required '
                                  "keyword-only argument: 'petsc_options_prefix'"]},
          'time_integration': {'REQUIRED': 'theta = fem.Constant(msh, 1.0)     # '
                                           'backward Euler\n'
                                           'dt = fem.Constant(msh, dt_val)     # the '
                                           'SAME object used in the form\n'
                                           '# and, at the end of each step:  '
                                           'T_n.x.array[:] = T_h.x.array',
                               'OPTIONAL': 'theta = 1.0  backward Euler: first order '
                                           'in time, unconditionally stable, monotone '
                                           '(never reverses direction), damps high '
                                           'frequencies strongly. The safe default.\n'
                                           'theta = 0.5  Crank-Nicolson: second order '
                                           'in time, unconditionally stable in the '
                                           'energy norm but NOT monotone; it rings on '
                                           'sharp transients unless dt is of order '
                                           'h^2.\n'
                                           'theta = 0.0  forward Euler: only '
                                           'conditionally stable, needs dt = O(h^2/k) '
                                           'and still requires a mass-matrix solve, so '
                                           'it buys nothing here.\n'
                                           'dt may be changed between steps by writing '
                                           'dt.value = new_dt, but then the LHS matrix '
                                           'must be re-assembled.',
                               'explanation': 'The theta-method amplification factor '
                                              'for a mode of the discrete Laplacian '
                                              'with eigenvalue lam is (1 - '
                                              '(1-theta)*dt*lam) / (1 + theta*dt*lam). '
                                              'For theta=1 it is positive and below 1 '
                                              'for every lam, so the solution is '
                                              'monotone. For theta=0.5 it becomes '
                                              'negative once dt*lam > 2, and since the '
                                              'largest lam scales like 1/h^2 that '
                                              'happens for any usable dt on any '
                                              'reasonably fine mesh, which is exactly '
                                              'the ringing.',
                               'pitfalls': ['Crank-Nicolson ringing does NOT go away '
                                            'by refining the mesh. Signal: at a fixed '
                                            'dt the nodal history next to a step '
                                            'Dirichlet reverses sign 18 times in 20 '
                                            'steps at every mesh from N=20 to N=160; '
                                            'it reaches 0 reversals only once dt is '
                                            'cut to about h^2.',
                                            'Reuse the same dt and theta Constant '
                                            'objects that were baked into the form; '
                                            'rebinding the Python name to a new '
                                            'fem.Constant has no effect on the '
                                            'compiled form. Signal: measured on the '
                                            'Dirichlet value, rebinding g_D = '
                                            'fem.Constant(msh, new_value) each step '
                                            'gives numbers bit-identical to never '
                                            'updating at all.']},
          'materials': {'REQUIRED': 'k = fem.Function(fem.functionspace(msh, ("DG", '
                                    '0)))\n'
                                    'k.interpolate(lambda x: np.where(x[0] < 0.5, 1.0, '
                                    '10.0))\n'
                                    'rho_cp = fem.Constant(msh, 2.0)',
                        'OPTIONAL': 'k may be a fem.Constant (uniform), a ("DG", 0) '
                                    'Function (piecewise-constant per cell, the right '
                                    'choice for distinct material blocks), or a '
                                    '("Lagrange", 1) Function (continuous field). All '
                                    'three were checked and all three give an exactly '
                                    'closed energy balance. A temperature-dependent '
                                    'k(T) is also legal, but then the problem is '
                                    'nonlinear and must go through '
                                    'dolfinx.fem.petsc.NonlinearProblem instead of a '
                                    'linear solve. Anisotropic conductivity: replace '
                                    'k*ufl.grad(T) by ufl.dot(K, ufl.grad(T)) with K a '
                                    'tensor-valued Constant or Function.\n'
                                    'Typical ranges: k about 0.02 (air) to 400 '
                                    '(copper) W/(m*K); rho*cp about 1e3 (gas) to 4e6 '
                                    '(water) J/(m^3*K).',
                        'explanation': 'Write the conductivity INSIDE the gradient '
                                       'product, ufl.inner(k*ufl.grad(T), '
                                       'ufl.grad(v))*ufl.dx, so that a discontinuous k '
                                       'is handled correctly cell by cell; a DG0 '
                                       'Function keeps the material jump exactly on '
                                       'the cell boundaries.',
                        'pitfalls': ['A spatially varying property cannot be a '
                                     'fem.Constant. Signal: fem.Constant(msh, lambda '
                                     'x: ...) and fem.Constant(msh, '
                                     'ufl.SpatialCoordinate(msh)[0]) both raise '
                                     'RuntimeError: Unsupported dtype.']},
          'steady_state': {'REQUIRED': 'a = ufl.inner(k * ufl.grad(u), ufl.grad(v)) * '
                                       'ufl.dx + h_c * u * v * ds(2)\n'
                                       'L = Q * v * ufl.dx + q_in * v * ds(3) + h_c * '
                                       'T_inf * v * ds(2)\n'
                                       'problem = LinearProblem(a, L, bcs=[bc], '
                                       'petsc_options_prefix="heat_",\n'
                                       '                        '
                                       'petsc_options={"ksp_type": "preonly", '
                                       '"pc_type": "lu"})\n'
                                       'T_h = problem.solve()',
                           'OPTIONAL': 'The Robin terms (ds(2)) and the source terms '
                                       'are optional. If there is NO Dirichlet bc '
                                       'anywhere and no Robin term, the pure Neumann '
                                       'problem is singular (temperature defined up to '
                                       'a constant) and needs either a nullspace or '
                                       'one pinned dof.',
                           'explanation': 'The steady form is the transient one with '
                                          'the time-derivative term deleted. A useful '
                                          'reference-free check for either: the net '
                                          'heat flux through a Dirichlet wall equals '
                                          'the sum over all dofs of the assembled '
                                          'residual of the UNCONSTRAINED form, and '
                                          'source + inflow + wall-flux - convected-out '
                                          'must vanish.',
                           'pitfalls': ['A steady problem with only Neumann/insulated '
                                        'walls has a singular matrix. Signal: LU '
                                        'returns garbage of order 1e15 (growing to nan '
                                        'if iterated) and raises nothing at all.']},
          'pitfalls': ['[API] An insulated / adiabatic / symmetry wall is a NATURAL '
                       'boundary condition: write nothing at all for it. Do NOT add a '
                       'Dirichlet bc with value 0 there -- that forces T=0 on the '
                       'wall, which is a completely different problem from zero heat '
                       'flux. Signal: [MEASURED, dolfinx 0.10.0, steady conduction on '
                       'the unit square, k=1, left wall T=1, top/bottom/right '
                       'physically insulated] with only the left bc the solution is '
                       'exactly uniform, max(T)=1.000000 and min(T)=1.000000, and the '
                       'net flux through the top+bottom walls is -1.33e-15 (machine '
                       'zero); adding T=0 on top+bottom gives min(T)=0.000000, '
                       'mean(T)=0.270218 and a net flux of +5.25 through walls that '
                       'must carry none. Identical conclusion on triangles and '
                       'quadrilaterals at P1 and P2.',
                       '[API] A time-dependent Dirichlet value must be written into '
                       'the SAME fem.Constant object the DirichletBC was built from, '
                       'via g_D.value = <new number>, once per time step. Rebinding '
                       'the Python name (g_D = fem.Constant(msh, new_value)) does '
                       'nothing, because the bc holds a reference to the original '
                       'object. Signal: [MEASURED, transient heat, T0=0, left wall '
                       'g(t)=1+2t, other walls insulated, backward Euler, 50 steps of '
                       'dt=0.01] with the update, max(T) at t=0.5 is 2.000000 = g(0.5) '
                       'and mean(T)=1.288479; with the update omitted, max(T) is '
                       '1.000000 = g(0) and mean(T)=0.760618, i.e. the field relaxes '
                       "toward the FIRST step's boundary value instead of evolving. "
                       'The name-rebinding variant produces bit-identical numbers to '
                       'doing nothing at all. No error, no warning; the KSP reports '
                       'converged. Same on triangles and quadrilaterals at P1 and P2.',
                       '[API] The last statement of the time-loop body must copy the '
                       'new solution into the old one: T_n.x.array[:] = T_h.x.array. '
                       'Signal: [MEASURED, 32x32 unit square, backward Euler, 30 steps '
                       'of dt=0.01] with the copy, int(T dx) is 0.100406 / 0.150204 / '
                       '0.187653 / 0.352449 / 0.609701 at steps 1 / 2 / 3 / 10 / 30; '
                       'without it the value is 0.100406 at EVERY step and the '
                       'solution vector is bit-identical at every step. Nothing is '
                       'raised and the KSP reports converged each time. Same on '
                       'triangles and quadrilaterals at P1 and P2.',
                       '[Numerical] The time-derivative term gives a mass matrix with '
                       'the SAME positive sign on both sides: (rho_cp/dt)*u*v*dx on '
                       'the left, (rho_cp/dt)*T_n*v*dx on the right. Signal: '
                       '[MEASURED, adiabatic cell test -- all walls natural, uniform '
                       'T0=1, no source, so the answer must stay T=1 for ever; 16x16 '
                       'unit square, 20 steps of dt=0.01] correct form: '
                       'max|T|=1.000000 and int(T dx)=1.000000 at every step. '
                       'IMPORTANT CORRECTION to the previously written claim: flipping '
                       "the SIGN of the old-time mass term does NOT produce 'magnitude "
                       "growing geometrically' -- max|T| stays exactly 1.000000 while "
                       'int(T dx) alternates -1.000000, +1.000000, -1.000000, ... The '
                       'error is a bounded sign flip every step, not a blow-up. Two '
                       'other placement errors do blow up, and they look different: '
                       'dropping the mass term from the LHS leaves the singular '
                       'pure-Neumann stiffness matrix, LU raises nothing and returns '
                       'max|T| ~ 3e15 after one step growing by a factor ~3e15 per '
                       'step to nan by step 20; flipping the sign of the STIFFNESS '
                       'term keeps max|T|=1.000000 for four steps and then grows '
                       'geometrically by a factor 40-78 per step to 1e17-1e23 by step '
                       '20. Dropping the old-time term from the RHS entirely gives '
                       'max|T| = 0.000000 at every step. All four variants verified on '
                       'triangles and quadrilaterals at P1 and P2. Use the '
                       'adiabatic-cell test (uniform initial temperature, no source, '
                       'no bc, temperature must not move) to tell them apart.',
                       '[Numerical] Crank-Nicolson (theta=0.5) on a sharp transient '
                       'produces a step-to-step oscillation IN TIME at the nodes next '
                       'to the transient; backward Euler (theta=1) does not. Signal: '
                       '[MEASURED, T0=0 with the left wall jumping to T=1 at t=0+, all '
                       'other walls insulated; the nodal history at the first free '
                       'node was scored with d_n = T^n - T^{n-1} and the alternation '
                       'amplitude max{min(|d_n|,|d_{n+1}|) : d_n*d_{n+1} < 0}, which '
                       'is exactly 0 for a monotone sequence] backward Euler gives 0 '
                       'sign reversals and amplitude exactly 0.0 in EVERY '
                       'configuration tested; Crank-Nicolson gives 18 sign reversals '
                       'in 20 steps with amplitude 2-9% of the imposed jump. TWO '
                       'CORRECTIONS to the previously written claim: (1) the artifact '
                       'is NOT a 10-30% over/undershoot of the field -- the '
                       'temperature never leaves the physical range, min(T) stays at '
                       'or above +1.4e-6 and max(T) is exactly 1.000000, so a min/max '
                       'check will not catch it; look at the time history of a node, '
                       'not at the field range. (2) the amplitude is 2-9%, not 10-30%. '
                       'The claim that it does not damp under refinement is only half '
                       'right: refining the MESH at fixed dt does not remove it (18 '
                       'reversals at every mesh from N=20 to N=160), and refining both '
                       'together with dt ~ h does not remove it either, but cutting dt '
                       'to about h^2 does (0 reversals once dt/h^2 <= about 1.3, still '
                       '6% at dt/h^2 = 3.2). So Crank-Nicolson is only safe here at a '
                       'step size that throws away its own advantage. Verified on a '
                       '1-D interval, on triangles and on quadrilaterals; P2 rings at '
                       'coarser dt/h^2 than P1 (P2 already rings at dt/h^2 = 1 where '
                       'P1 does not).',
                       '[API] A UFL Form cannot be divided by a fem.Constant; divide '
                       'the argument instead. Signal: u * v * ufl.dx / dt raises '
                       "TypeError: unsupported operand type(s) for /: 'Form' and "
                       "'Constant'. Write (u / dt) * v * ufl.dx or (1.0/dt) * u * v * "
                       'ufl.dx.',
                       '[API] A spatially varying material property must be a '
                       'fem.Function on its own space, not a fem.Constant, and '
                       'fem.Constant rejects Python ints. Signal: fem.Constant(msh, '
                       'lambda x: np.where(x[0] < 0.5, 1.0, 10.0)) raises '
                       'RuntimeError: Unsupported dtype; so does fem.Constant(msh, '
                       'ufl.SpatialCoordinate(msh)[0]); so does fem.Constant(msh, 1) '
                       'with a plain int. The working pattern is k = '
                       "fem.Function(fem.functionspace(msh, ('DG', 0))) followed by "
                       'k.interpolate(lambda x: ...), which was verified to give an '
                       'exactly closed energy balance (residual 1e-13 or smaller) on '
                       'triangles and quadrilaterals at P1 and P2, and also with a '
                       "('Lagrange', 1) conductivity field.",
                       '[Input] A ds(marker) term with no subdomain_data attached to '
                       'the measure integrates over NOTHING and returns zero silently. '
                       'Signal: fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * '
                       'ufl.ds(3))) returns 0.0 with no error and no warning, while '
                       'fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ufl.ds)) '
                       'returns 4.0 (the perimeter of the unit square). After building '
                       "ds = ufl.Measure('ds', domain=msh, subdomain_data=ft) the same "
                       'tagged form returns 1.0 (the length of the tagged edge). A '
                       'marker value that is not present in the meshtags also returns '
                       "0.0 silently, so a typo'd tag number removes a Neumann or "
                       'Robin condition without any diagnostic. Always assemble '
                       'fem.Constant(msh,1.0)*ds(m) once and check it equals the '
                       'expected boundary measure.',
                       '[API] mesh.exterior_facet_indices(msh.topology) needs the '
                       'facet-to-cell connectivity to exist first. Signal: '
                       'RuntimeError: Facet to cell connectivity has not been '
                       'computed. Call '
                       'msh.topology.create_connectivity(msh.topology.dim - 1, '
                       'msh.topology.dim) before it.',
                       '[Performance] The theta-method LHS does not depend on time: '
                       'assemble the matrix and set up the KSP once, outside the loop, '
                       'and rebuild only the RHS vector inside it. Signal: [MEASURED, '
                       '64x64 unit square, P1, 100 steps, LU] assemble-once with KSP '
                       'reuse takes 0.151 s; keeping one LinearProblem and calling '
                       '.solve() every step takes 1.233 s (8.2x slower, because '
                       'LinearProblem.solve() re-assembles and re-factorises the '
                       'matrix on every call); constructing a fresh LinearProblem '
                       'every step takes 3.138 s (20.8x). All three produce the '
                       'identical final int(T dx) = 0.7621753876, so the answer never '
                       'reveals the mistake -- only the wall-clock time does.',
                       '[API] In the manual assemble-once loop, apply_lifting and '
                       'set_bc are both required after assembling the RHS; omitting '
                       'apply_lifting silently produces a wrong answer that still '
                       'reports convergence. Signal: [MEASURED, transient heat with a '
                       'Dirichlet wall] with apply_lifting the residual of the step '
                       'form at the free dofs is 4.928e-14 and the temperature range '
                       'is [1.071425, 1.989625]; with apply_lifting removed the KSP '
                       'still reports converged reason 4, the field is finite, but the '
                       'free-dof residual is 1.696e+00 and the range is [0.180492, '
                       '1.800000]. The cheapest detector is to assemble ufl.replace(F, '
                       '{u: T_h}) and check that its entries at the non-Dirichlet dofs '
                       'are at machine zero.',
                       '[API] dolfinx.fem.petsc.LinearProblem and NonlinearProblem '
                       'take petsc_options_prefix as a REQUIRED keyword-only argument '
                       'in 0.10. Signal: TypeError: LinearProblem.__init__() missing 1 '
                       "required keyword-only argument: 'petsc_options_prefix' (and "
                       'the same message for NonlinearProblem). There is no '
                       'dolfinx.PETScKrylovSolver in any version of the package; the '
                       "low-level solver is petsc4py's PETSc.KSP().create(msh.comm).",
                       '[API] The space constructor is fem.functionspace with a '
                       "lower-case s. Signal: fem.FunctionSpace(msh, ('Lagrange', 1)) "
                       'raises TypeError: FunctionSpace.__init__() missing 1 required '
                       "positional argument: 'cppV' (fem.FunctionSpace exists but is "
                       'the internal class, not the factory), and '
                       'fem.VectorFunctionSpace does not exist at all -- hasattr(fem, '
                       "'VectorFunctionSpace') is False. For a vector field use "
                       "fem.functionspace(msh, ('Lagrange', 1, (dim,)))."]},
 'convection_diffusion': {'description': 'Advection-diffusion equation with SUPG '
                                         '(Streamline Upwind Petrov-Galerkin) '
                                         'stabilization for advection-dominated '
                                         'transport.',
                          'weak_form': 'inner(b, grad(u))*v*dx + kappa*inner(grad(u), '
                                       'grad(v))*dx = f*v*dx',
                          'supg_stabilization': {'description': 'Add stabilization '
                                                                'term: tau * inner(b, '
                                                                'grad(v)) * (inner(b, '
                                                                'grad(u)) + '
                                                                'kappa*div(grad(u)) - '
                                                                'f) * dx',
                                                 'tau': 'h / (2*|b|) * (coth(Pe_h) - '
                                                        '1/Pe_h) where Pe_h = '
                                                        '|b|*h/(2*kappa) is cell '
                                                        'Peclet number',
                                                 'implementation': 'Modify test '
                                                                   'function: v_stab = '
                                                                   'v + tau * inner(b, '
                                                                   'grad(v))'},
                          'alternative_stabilizations': {'DG': 'Discontinuous Galerkin '
                                                               'with upwind flux — '
                                                               'naturally handles '
                                                               'advection',
                                                         'GLS': 'Galerkin Least '
                                                                'Squares — similar to '
                                                                'SUPG but also '
                                                                'stabilizes reaction'},
                          'pitfalls': ['[Numerical] Without stabilization, the '
                                       'Galerkin method oscillates whenever the CELL '
                                       'Peclet number Pe_h = |b|*h/(2*kappa) exceeds '
                                       '~1. Signal: [MEASURED 2026-08-03, dolfinx '
                                       '0.10.0, P1 on the unit square, b=(1,0), '
                                       'kappa=1e-3, u=0 at x=0 and u=1 at x=1 — an '
                                       'exponential outflow layer] the Galerkin '
                                       'undershoot is -6.28 / -3.00 / -1.72 / -1.41 / '
                                       '-1.02 / -0.56 / -0.08 at N = 8 / 16 / 32 / 64 '
                                       '/ 128 / 256 / 512 (Pe_h = 62.5 down to 0.98). '
                                       'IMPORTANT CORRECTION: the oscillation DOES '
                                       'damp under refinement — it disappears once the '
                                       'mesh resolves the layer (Pe_h < 1). The '
                                       "previous wording ('oscillation amplitude does "
                                       "not damp with mesh refinement') is falsified. "
                                       'The real argument for stabilisation is COST: '
                                       'resolving the layer needs h < 2*kappa/|b|, '
                                       'which is unaffordable for small kappa. SUPG '
                                       'with the same meshes keeps the undershoot at '
                                       '-0.042 down to -0.000 throughout.',
                                       '[Numerical] SUPG tau parameter depends on mesh '
                                       'size h and velocity magnitude — must compute '
                                       'PER CELL via tau = h/(2*|b|) * (coth(Pe_h) - '
                                       '1/Pe_h) using ufl.CellDiameter inside the '
                                       'dolfinx fem.form. A single global tau Constant '
                                       'does not vanish as h -> 0 and leaves a fixed '
                                       'streamline-diffusion floor. Signal: [MEASURED '
                                       '2026-08-03; MMS u = a smooth manufactured '
                                       'solution, b=(1,1), kappa=0.01, N = 8..128] the '
                                       'per-cell CellDiameter tau keeps the L2 error '
                                       'converging at the theoretical O(h^(k+1)). With '
                                       'a FIXED tau Constant taken from the coarse '
                                       'mesh the error stops moving under refinement. '
                                       'IMPORTANT CORRECTION: a constant tau does not '
                                       'degrade the rate to ~O(h); it STALLS '
                                       'convergence entirely — the error plateaus at a '
                                       'fixed floor and further refinement buys '
                                       'nothing. Same behaviour on P2. Look for a flat '
                                       'error curve, not a halved slope.',
                                       '[Numerical] DG methods are a cleaner '
                                       'alternative for pure advection (no diffusion). '
                                       'Signal: for vanishing diffusion kappa -> 0, '
                                       "the SUPG dolfinx ufl form's tau degenerates "
                                       '(tau -> h/|b|, but stabilisation residual '
                                       'scales with kappa) and the LinearProblem '
                                       'solution oscillates between elements; an '
                                       'upwind DG basix.ufl element on the same mesh '
                                       'produces a smooth Function with no parameter '
                                       'tuning. (Audit 2026-06-02.)',
                                       '[Numerical] For time-dependent: SUPG in space '
                                       '+ implicit time stepping. Mixing SUPG with '
                                       'explicit Euler can break: SUPG injects '
                                       'time-derivative coupling via the residual, '
                                       'which needs implicit treatment. Signal: '
                                       'explicit dolfinx fem.assemble + SUPG diverges '
                                       'to NaN within a few steps even below the '
                                       'convective CFL; switching to implicit (theta=1 '
                                       'or BDF2) inside a NonlinearProblem restores '
                                       'stability. (Audit 2026-06-02.)']},
 'hyperelasticity': {'description': 'Nonlinear hyperelasticity with large '
                                    'deformations. Stored energy function approach.',
                     'weak_form': 'delta_Pi(u;v) = 0 where Pi = integral(psi(F) dx - '
                                  'T.u ds), solved as F(u,v) = dPi/du[v] = 0',
                     'function_space': 'Vector Lagrange order 1 or 2',
                     'demo_url': 'https://jsdokken.com/dolfinx-tutorial/chapter2/hyperelasticity.html',
                     'kinematics': {'F': 'ufl.variable(ufl.Identity(d) + ufl.grad(u)) '
                                         '— deformation gradient',
                                    'C': 'F.T * F — right Cauchy-Green tensor',
                                    'J': 'ufl.det(F) — volume ratio (J>0 required)',
                                    'I_C': 'ufl.tr(C) — first invariant',
                                    'I_Cbar': 'J^(-2/d) * I_C — isochoric first '
                                              'invariant'},
                     'material_models': {'neo_hookean': {'psi': '(mu/2)*(I_C - 3) - '
                                                                'mu*ln(J) + '
                                                                '(lambda_/2)*(ln(J))**2',
                                                         'parameters': 'mu = '
                                                                       'E/(2*(1+nu)), '
                                                                       'lambda_ = '
                                                                       'E*nu/((1+nu)*(1-2*nu))'},
                                         'mooney_rivlin': {'psi': 'c1*(I_C - 3) + '
                                                                  'c2*(II_C - 3) + '
                                                                  '(K/2)*(J-1)**2',
                                                           'parameters': 'c1, c2 '
                                                                         '(material '
                                                                         'constants), '
                                                                         'K (bulk '
                                                                         'modulus)',
                                                           'notes': 'II_C = '
                                                                    '0.5*(tr(C)^2 - '
                                                                    'tr(C^2)) is '
                                                                    'second '
                                                                    'invariant'}},
                     'code_skeleton': {'F': 'F = ufl.variable(ufl.Identity(d) + '
                                            'ufl.grad(u))',
                                       'psi': 'psi = (mu/2)*(ufl.tr(F.T*F) - 3) - '
                                              'mu*ufl.ln(ufl.det(F)) + '
                                              '(lmbda/2)*(ufl.ln(ufl.det(F)))**2',
                                       'P': 'P = ufl.diff(psi, F)  # First '
                                            'Piola-Kirchhoff stress via automatic '
                                            'differentiation',
                                       'F_form': 'F_form = ufl.inner(P, ufl.grad(v)) * '
                                                 'ufl.dx - ufl.dot(traction, v) * '
                                                 'ufl.ds'},
                     'solver': {'nonlinear': 'NonlinearProblem with SNES newtonls',
                                'petsc_options': {'snes_type': 'newtonls',
                                                  'ksp_type': 'preonly',
                                                  'pc_type': 'lu',
                                                  'pc_factor_mat_solver_type': 'mumps'},
                                'load_stepping': 'For large deformations: apply load '
                                                 'in increments, solving at each step'},
                     'pitfalls': ['[API] Several signals below name '
                                  'dolfinx.nls.petsc.NewtonSolver.solve, which is a '
                                  'dolfinx 0.9-era code path. Signal: on dolfinx '
                                  '0.10.0, NewtonSolver(MPI.COMM_WORLD, problem) '
                                  'around a 0.10 NonlinearProblem emits a '
                                  'DeprecationWarning and then raises AttributeError: '
                                  "'NonlinearProblem' object has no attribute 'a'. "
                                  '(Verified empirically 2026-08-03.) Read those '
                                  'signals as PETSc SNES signals instead: use '
                                  'NonlinearProblem(..., petsc_options_prefix=..., '
                                  "petsc_options={'snes_monitor': ''}) and inspect "
                                  'problem.solver.getConvergedReason() / '
                                  'getIterationNumber().',
                                  '[Numerical] Large load steps cause Newton '
                                  'divergence in hyperelasticity. Use incremental load '
                                  'stepping: ramp the dirichletbc value or body-force '
                                  'fem.Constant across N steps, solving at each level. '
                                  'Signal: the SNES converged reason goes negative '
                                  '(DIVERGED_LINE_SEARCH / DIVERGED_MAX_IT) with the '
                                  'residual at the last iter still O(1); reducing the '
                                  'per-step load increment by 2-4x recovers '
                                  'convergence. (Claim inherited; the '
                                  'NewtonSolver-specific wording was corrected '
                                  '2026-08-03 — see the version note above.)',
                                  '[Numerical] Near-incompressible regime (nu > 0.49) '
                                  'can make the pure-displacement formulation lock — '
                                  'use a dolfinx mixed (u, p) '
                                  'basix.ufl.mixed_element([P2-vector, P1]) '
                                  'FunctionSpace or the F-bar method (uniform-pressure '
                                  'projection). Signal: [CAUTION on magnitude, '
                                  '2026-08-03] the previously quoted signal '
                                  "('Cook-membrane tip deflection at nu = 0.4999 with "
                                  "pure P2 displacement is O(1e-3) of analytic') is "
                                  'not supported by measurement in the linear analogue '
                                  '— a P2 cantilever at nu=0.4999 came within 0-6% of '
                                  'the P2/P1 Taylor-Hood reference on every mesh '
                                  'tested, while P1 was 9x-20x too stiff. Expect the '
                                  'severe locking at P1, not at P2; measure the ratio '
                                  'for your own geometry rather than assuming orders '
                                  'of magnitude. (Linear-elasticity measurement '
                                  '2026-08-03; the hyperelastic Cook membrane itself '
                                  'was NOT re-run.)',
                                  '[Physics] Neo-Hookean / any compressible '
                                  'hyperelastic model requires J = det(F) > 0 '
                                  'everywhere. A locally inverted element gives J <= 0 '
                                  'and the ln(J) terms go non-finite. Signal: '
                                  '[MEASURED on dolfinx 0.10.0] the failure is SILENT '
                                  '— nothing is raised. With a deformation giving '
                                  'det(F) = -2, '
                                  'dolfinx.fem.assemble_scalar(dolfinx.fem.form(psi*ufl.dx)) '
                                  'returns nan and the assembled residual vector is '
                                  'all-nan (numpy.isnan(b.array).any() is True), with '
                                  'no RuntimeError and no FloatingPointError. Defend '
                                  'by monitoring J: assemble ufl.conditional(ufl.lt(J, '
                                  '0), 1.0, 0.0)*ufl.dx and require it to stay 0, and '
                                  'assert numpy.isfinite(u.x.array).all() after every '
                                  'load step. NewtonSolver.solve is a dolfinx 0.9-era '
                                  'call and cannot be constructed around a 0.10 '
                                  'NonlinearProblem.',
                                  '[API] ufl.variable() + ufl.diff() automate stress '
                                  'computation from a stored energy W. Wrap F in '
                                  'ufl.variable to mark it as the differentiation '
                                  'target, define W(F_var), then P = ufl.diff(W, '
                                  'F_var) yields the 1st Piola-Kirchhoff stress as a '
                                  'VariableDerivative expression (class '
                                  'ufl.classes.VariableDerivative; there is no '
                                  'top-level ufl.VariableDerivative) directly usable '
                                  'inside the residual ufl.inner(P, grad(v))*dx form. '
                                  'Signal: type(ufl.variable(F)) is '
                                  'ufl.classes.Variable; type(ufl.diff(W, '
                                  "F_var)).__name__ == 'VariableDerivative'. NOTE the "
                                  'spelling: the class lives in the ufl.variable '
                                  'MODULE, so repr(type(...)) prints "<class '
                                  '\'ufl.variable.Variable\'>", but the attribute '
                                  'ufl.variable is the FUNCTION variable() and shadows '
                                  'the submodule — writing `ufl.variable.Variable` '
                                  "raises AttributeError: 'function' object has no "
                                  "attribute 'Variable'. Use ufl.classes.Variable. "
                                  "Hand-coding the gradient bypasses ufl's analytic "
                                  'differentiation and is error-prone. (Verified '
                                  'empirically 2026-06-01; spelling re-checked by '
                                  'execution 2026-08-03 on ufl 2025.2.1.)',
                                  '[Numerical] Near-incompressibility split: decompose '
                                  'F = F_iso * F_vol where F_vol = (J^(1/3))*I (via '
                                  'ufl.det and ufl.Identity); then W = W_iso(F_iso) + '
                                  'U(J) with a quadratic-in-(J-1) volumetric penalty '
                                  'U(J) = kappa/2 * (J - 1)^2. Avoids volumetric '
                                  'locking in pure-displacement settings AND retains a '
                                  'well-conditioned tangent. Signal: what the split '
                                  'buys you is a clean, bounded volumetric pressure - '
                                  'dolfinx fem.assemble_scalar of the post-processed '
                                  'dU/dJ = kappa*(J - 1) stays bounded - and a tangent '
                                  'that Newton keeps converging on. IMPORTANT '
                                  'CORRECTION: it does NOT quieten the element-to-'
                                  'element pressure oscillation, and an earlier version '
                                  'of this entry claimed it did. Project the hydrostatic '
                                  'pressure -tr(sigma)/d into a DG0 space and compare '
                                  'the element-to-element spread three ways on the same '
                                  'mesh: pure displacement WITH the split and pure '
                                  'displacement WITHOUT it come out alike, while a mixed '
                                  '(u, p) space carrying the same isochoric energy is '
                                  'dramatically tighter. So if the oscillating pressure '
                                  'is what you are trying to fix, the observable will '
                                  'not move when you add the split - reach for the mixed '
                                  'formulation instead, and keep the split for the '
                                  'locking and conditioning it does address. (Verified by execution on dolfinx 0.10.0, 2026-08-06.)',
                                  '[API] PETSc SNES residual monitor: pass '
                                  "'snes_monitor': '' (or 'snes_monitor_short') in the "
                                  'petsc_options dict of '
                                  'dolfinx.fem.petsc.NonlinearProblem — NOT to '
                                  'dolfinx.nls.petsc.NewtonSolver, which cannot wrap a '
                                  '0.10 NonlinearProblem at all. Signal: [MEASURED on '
                                  'dolfinx 0.10.0 / petsc4py 3.24.4] the monitor '
                                  'writes to STDOUT, not stderr, one line per '
                                  "iteration in the exact form '  0 SNES Function norm "
                                  "1.093750000000e+00' / '  1 SNES Function norm "
                                  "2.011629733446e-01' / ... A healthy Newton drops "
                                  'the norm by orders of magnitude per line; a stalled '
                                  'one plateaus at a fixed O(1) value, at which point '
                                  'halve the load increment. Read the final state from '
                                  'problem.solver.getConvergedReason() and '
                                  'problem.solver.getIterationNumber().'],
                     'materials': {'E': {'range': [100.0, 1000000000000.0],
                                         'unit': 'Pa'},
                                   'nu': {'range': [0.0, 0.499],
                                          'unit': 'dimensionless'}}},
 'thermal_structural': {'description': 'One-way (sequential) thermo-elasticity: first '
                                       'solve steady heat conduction for a temperature '
                                       'field on a scalar space, then solve linear '
                                       'elasticity on a vector space in which that '
                                       'temperature Function enters the stress as a '
                                       'thermal strain. Two-way variants additionally '
                                       'feed the deformation back into the heat '
                                       'problem and are solved by Picard iteration. '
                                       'Both fields live on separate function spaces '
                                       'over the same mesh.',
                        'minimal_working_example': 'import numpy as np\n'
                                                   'import ufl\n'
                                                   'from mpi4py import MPI\n'
                                                   'from dolfinx import fem, mesh\n'
                                                   'from dolfinx.fem.petsc import '
                                                   'LinearProblem, assemble_vector, '
                                                   'set_bc\n'
                                                   '\n'
                                                   'msh = '
                                                   'mesh.create_rectangle(MPI.COMM_WORLD, '
                                                   '[np.array([0.0, 0.0]), '
                                                   'np.array([1.0, 0.2])],\n'
                                                   '                            [20, '
                                                   '4], mesh.CellType.triangle)\n'
                                                   'tdim = msh.topology.dim\n'
                                                   'E, nu, alpha = 210e9, 0.3, 12e-6\n'
                                                   'mu = fem.Constant(msh, E / (2.0 * '
                                                   '(1.0 + nu)))\n'
                                                   'lam = fem.Constant(msh, E * nu / '
                                                   '((1.0 + nu) * (1.0 - 2.0 * nu)))\n'
                                                   'alpha_c = fem.Constant(msh, '
                                                   'alpha)\n'
                                                   'kappa = fem.Constant(msh, 45.0)\n'
                                                   'T_ref = fem.Constant(msh, 293.15)\n'
                                                   'T_hot, T_cold = 393.15, 293.15\n'
                                                   '\n'
                                                   '# ---- step 1: thermal problem on '
                                                   'a SCALAR space ----\n'
                                                   'S = fem.functionspace(msh, '
                                                   '("Lagrange", 1))\n'
                                                   'Tt, s = ufl.TrialFunction(S), '
                                                   'ufl.TestFunction(S)\n'
                                                   'a_T = kappa * '
                                                   'ufl.inner(ufl.grad(Tt), '
                                                   'ufl.grad(s)) * ufl.dx\n'
                                                   'L_T = fem.Constant(msh, 0.0) * s * '
                                                   'ufl.dx\n'
                                                   'left = '
                                                   'mesh.locate_entities_boundary(msh, '
                                                   'tdim - 1, lambda x: '
                                                   'np.isclose(x[0], 0.0))\n'
                                                   'right = '
                                                   'mesh.locate_entities_boundary(msh, '
                                                   'tdim - 1, lambda x: '
                                                   'np.isclose(x[0], 1.0))\n'
                                                   'bc_T = '
                                                   '[fem.dirichletbc(fem.Constant(msh, '
                                                   'T_hot),\n'
                                                   '                        '
                                                   'fem.locate_dofs_topological(S, '
                                                   'tdim - 1, left), S),\n'
                                                   '        '
                                                   'fem.dirichletbc(fem.Constant(msh, '
                                                   'T_cold),\n'
                                                   '                        '
                                                   'fem.locate_dofs_topological(S, '
                                                   'tdim - 1, right), S)]\n'
                                                   'prob_T = LinearProblem(a_T, L_T, '
                                                   'bcs=bc_T, '
                                                   'petsc_options_prefix="therm_",\n'
                                                   '                       '
                                                   'petsc_options={"ksp_type": "cg", '
                                                   '"pc_type": "hypre",\n'
                                                   '                                      '
                                                   '"ksp_rtol": 1e-12})\n'
                                                   'T = prob_T.solve()\n'
                                                   'T.name = "temperature"\n'
                                                   'print("thermal KSP reason :", '
                                                   'prob_T.solver.getConvergedReason())\n'
                                                   '\n'
                                                   '# ---- step 2: elasticity on a '
                                                   'VECTOR space, T enters the form '
                                                   'directly ----\n'
                                                   'V = fem.functionspace(msh, '
                                                   '("Lagrange", 1, (tdim,)))\n'
                                                   'u, v = ufl.TrialFunction(V), '
                                                   'ufl.TestFunction(V)\n'
                                                   'Id = ufl.Identity(tdim)\n'
                                                   '\n'
                                                   '\n'
                                                   'def eps(w):\n'
                                                   '    return ufl.sym(ufl.grad(w))\n'
                                                   '\n'
                                                   '\n'
                                                   'def sigma_elastic(w):\n'
                                                   '    return 2.0 * mu * eps(w) + lam '
                                                   '* ufl.tr(eps(w)) * Id\n'
                                                   '\n'
                                                   '\n'
                                                   'beta = (3.0 * lam + 2.0 * mu) * '
                                                   'alpha_c          # thermal stress '
                                                   'modulus\n'
                                                   'a_u = ufl.inner(sigma_elastic(u), '
                                                   'eps(v)) * ufl.dx\n'
                                                   'L_u = beta * (T - T_ref) * '
                                                   'ufl.div(v) * ufl.dx   # thermal '
                                                   'term belongs on the RHS\n'
                                                   'bc_u = '
                                                   '[fem.dirichletbc(np.zeros(tdim, '
                                                   'dtype=np.float64),\n'
                                                   '                        '
                                                   'fem.locate_dofs_topological(V, '
                                                   'tdim - 1, left), V)]\n'
                                                   'prob_u = LinearProblem(a_u, L_u, '
                                                   'bcs=bc_u, '
                                                   'petsc_options_prefix="elast_",\n'
                                                   '                       '
                                                   'petsc_options={"ksp_type": "cg", '
                                                   '"pc_type": "gamg",\n'
                                                   '                                      '
                                                   '"ksp_rtol": 1e-10})\n'
                                                   'uh = prob_u.solve()\n'
                                                   'uh.name = "displacement"\n'
                                                   'print("elastic KSP reason :", '
                                                   'prob_u.solver.getConvergedReason())\n'
                                                   '\n'
                                                   '# ---- physical self-checks (no '
                                                   'reference solution needed) ----\n'
                                                   'Tv = T.x.array\n'
                                                   'print("T within BC range  :", '
                                                   'bool(Tv.min() >= T_cold - 1e-6 and '
                                                   'Tv.max() <= T_hot + 1e-6))\n'
                                                   'ux = uh.x.array.reshape(-1, '
                                                   'tdim)[:, 0]\n'
                                                   'print("heated bar extends :", '
                                                   'bool(ux.max() > 0.0))\n'
                                                   'print("all values finite  :", '
                                                   'bool(np.all(np.isfinite(uh.x.array))))\n'
                                                   'res = '
                                                   'assemble_vector(fem.form(ufl.action(a_u, '
                                                   'uh) - L_u))\n'
                                                   'res.ghostUpdate()\n'
                                                   'set_bc(res, bc_u)\n'
                                                   'rel = res.norm() / '
                                                   'assemble_vector(fem.form(L_u)).norm()\n'
                                                   'print("relative residual  : %.3e '
                                                   '(should be ~1e-9 or smaller)" % '
                                                   'rel)\n'
                                                   'sig = sigma_elastic(uh) - beta * '
                                                   '(T - T_ref) * Id\n'
                                                   'vol = '
                                                   'fem.assemble_scalar(fem.form(1.0 * '
                                                   'ufl.dx(domain=msh)))\n'
                                                   'print("mean sigma_xx (0 for a bar '
                                                   'free at one end): %.3e"\n'
                                                   '      % '
                                                   '(fem.assemble_scalar(fem.form(sig[0, '
                                                   '0] * ufl.dx)) / vol))\n',
                        'function_space': {'REQUIRED': 'Two separate spaces on the '
                                                       'SAME mesh:\n'
                                                       '    S = fem.functionspace(msh, '
                                                       '("Lagrange", '
                                                       '1))                     # '
                                                       'temperature\n'
                                                       '    V = fem.functionspace(msh, '
                                                       '("Lagrange", 1, '
                                                       '(msh.topology.dim,)))  # '
                                                       'displacement\n'
                                                       'The third entry of the vector '
                                                       'tuple is a SHAPE TUPLE, e.g. '
                                                       '(2,) or (3,); '
                                                       '`fem.functionspace` is spelled '
                                                       'with a lower-case s in dolfinx '
                                                       '0.10.',
                                           'OPTIONAL': 'Degrees 1, 2 or 3 on either '
                                                       'space, and the two degrees '
                                                       'need NOT match (a P1 '
                                                       'temperature Function can be '
                                                       'used inside a P2 displacement '
                                                       'form). Cell types triangle, '
                                                       'quadrilateral, tetrahedron and '
                                                       'hexahedron all work. No mixed '
                                                       'element is needed: this is a '
                                                       'sequential solve, not a '
                                                       'monolithic one.',
                                           'explanation': 'The temperature is a scalar '
                                                          'field and the displacement '
                                                          'a vector field, so they '
                                                          'need different spaces. '
                                                          'Because the coupling is '
                                                          'one-way, the temperature is '
                                                          'just a known coefficient '
                                                          'Function inside the '
                                                          'elasticity form.',
                                           'pitfalls': ['Do not build a single mixed '
                                                        'element for T and u unless '
                                                        'you actually solve them '
                                                        'monolithically. Signal: none '
                                                        '- it runs, but the extra '
                                                        'unknowns are wasted work.']},
                        'weak_form': {'REQUIRED': 'Step 1 (thermal):\n'
                                                  '    a_T = '
                                                  'kappa*ufl.inner(ufl.grad(Tt), '
                                                  'ufl.grad(s))*ufl.dx\n'
                                                  '    L_T = '
                                                  'Q*s*ufl.dx                    # Q = '
                                                  'fem.Constant(msh, 0.0) if no '
                                                  'source\n'
                                                  'Step 2 (elasticity), with T the '
                                                  'Function returned by step 1:\n'
                                                  '    Id  = '
                                                  'ufl.Identity(msh.topology.dim)\n'
                                                  '    eps = lambda w: '
                                                  'ufl.sym(ufl.grad(w))\n'
                                                  '    sigma_elastic = lambda w: '
                                                  '2*mu*eps(w) + '
                                                  'lam*ufl.tr(eps(w))*Id\n'
                                                  '    beta = (3*lam + '
                                                  '2*mu)*alpha         # thermal '
                                                  'stress modulus, 3D and plane '
                                                  'strain\n'
                                                  '    a_u  = '
                                                  'ufl.inner(sigma_elastic(u), '
                                                  'eps(v))*ufl.dx\n'
                                                  '    L_u  = beta*(T - '
                                                  'T_ref)*ufl.div(v)*ufl.dx\n'
                                                  'The full stress is sigma = '
                                                  'sigma_elastic(u) - beta*(T - '
                                                  'T_ref)*Id; because the thermal part '
                                                  'does not contain the trial function '
                                                  'it must be moved to L_u, and '
                                                  'ufl.inner(c*Id, eps(v)) equals '
                                                  'c*ufl.div(v).',
                                      'OPTIONAL': 'Add a body force `ufl.inner(f, '
                                                  'v)*ufl.dx` or a traction '
                                                  '`ufl.inner(t, v)*ufl.ds(tag)` to '
                                                  'L_u. Add a heat source or a Neumann '
                                                  'flux to L_T. A '
                                                  'temperature-dependent kappa(T) '
                                                  'makes step 1 nonlinear: then use '
                                                  'fem.petsc.NonlinearProblem with a '
                                                  'Function T and call '
                                                  'problem.solve().',
                                      'explanation': 'The two fields meet in exactly '
                                                     'one term: beta*(T - T_ref). beta '
                                                     'is the thermal stress modulus '
                                                     'and is NOT alpha alone - alpha '
                                                     'must be multiplied by (3*lam + '
                                                     '2*mu), which is E/(1 - 2*nu).',
                                      'pitfalls': ['Never leave the thermal term '
                                                   'inside the bilinear form a_u. '
                                                   'Signal: '
                                                   '`ufl.algorithms.check_arities.ArityMismatch: '
                                                   'Adding expressions with '
                                                   'non-matching form arguments () vs '
                                                   "('v_1',).`",
                                                   'After ANY form-compile failure, '
                                                   'delete the stale entry from the '
                                                   'FFCx cache before re-running. '
                                                   'Signal: the second run of the same '
                                                   'broken script hides the real error '
                                                   'behind `TimeoutError: JIT '
                                                   'compilation timed out, probably '
                                                   'due to a failed previous compile. '
                                                   'Try cleaning cache (e.g. remove '
                                                   '<cache_dir>/libffcx_forms_<hash>.c) '
                                                   'or increase timeout option.` - the '
                                                   'message names the exact file to '
                                                   'delete.',
                                                   'Never use alpha alone where (3*lam '
                                                   '+ 2*mu)*alpha is required. Signal: '
                                                   'no error; max(abs(uh.x.array)) '
                                                   'comes out around 1e-15 instead of '
                                                   'O(alpha*dT*L), i.e. numerically '
                                                   'indistinguishable from zero for SI '
                                                   'stiffnesses.',
                                                   'In 2D never write the thermal term '
                                                   'as C:(eps(u) - '
                                                   'alpha*dT*ufl.Identity(2)). Signal: '
                                                   'no error; the free expansion is '
                                                   'silently that of plane STRESS, so '
                                                   'the displacement is uniformly too '
                                                   'small by a factor that grows with '
                                                   'nu.']},
                        'boundary_conditions': {'REQUIRED': 'Thermal: at least one '
                                                            'Dirichlet condition, '
                                                            'otherwise the '
                                                            'pure-Neumann heat problem '
                                                            'is singular.\n'
                                                            '    '
                                                            'fem.dirichletbc(fem.Constant(msh, '
                                                            'T_hot),\n'
                                                            '                    '
                                                            'fem.locate_dofs_topological(S, '
                                                            'tdim-1, facets), S)\n'
                                                            'Mechanical: enough '
                                                            'Dirichlet constraints to '
                                                            'remove EVERY rigid body '
                                                            'mode - 3 scalar '
                                                            'constraints in 2D, 6 in '
                                                            '3D. Either clamp a whole '
                                                            'face,\n'
                                                            '    '
                                                            'fem.dirichletbc(np.zeros(tdim), '
                                                            'fem.locate_dofs_topological(V, '
                                                            'tdim-1, f), V)\n'
                                                            'or use the statically '
                                                            'determinate minimal set '
                                                            '(2D: u_x and u_y at one '
                                                            'corner, u_y at a second '
                                                            'corner):\n'
                                                            '    def point_bc(V, comp, '
                                                            'px, py):\n'
                                                            '        Vs = V.sub(comp)\n'
                                                            '        Vc, _ = '
                                                            'Vs.collapse()\n'
                                                            '        dofs = '
                                                            'fem.locate_dofs_geometrical(\n'
                                                            '            (Vs, Vc), '
                                                            'lambda x: '
                                                            'np.isclose(x[0], px) & '
                                                            'np.isclose(x[1], py))\n'
                                                            '        z = '
                                                            'fem.Function(Vc)          '
                                                            '# value MUST be a '
                                                            'Function on Vc\n'
                                                            '        z.x.array[:] = '
                                                            '0.0\n'
                                                            '        return '
                                                            'fem.dirichletbc(z, dofs, '
                                                            'Vs)\n'
                                                            '    bcs = [point_bc(V, 0, '
                                                            '0.0, 0.0), point_bc(V, 1, '
                                                            '0.0, 0.0),\n'
                                                            '           point_bc(V, 1, '
                                                            'L, 0.0)]',
                                                'OPTIONAL': 'Clamping a whole face is '
                                                            'legal and simpler, but it '
                                                            'is over-constrained: it '
                                                            'blocks free thermal '
                                                            'expansion of that face '
                                                            'and therefore creates '
                                                            'stress that the minimal '
                                                            'set does not. Choose '
                                                            'deliberately. Symmetry '
                                                            '(roller) conditions on '
                                                            'one component of a face '
                                                            'are the usual middle '
                                                            'ground. Thermal '
                                                            'Neumann/Robin conditions '
                                                            'are added as ufl.ds terms '
                                                            'in L_T instead of '
                                                            'Dirichlet.',
                                                'explanation': 'A thermal load is '
                                                               'self-equilibrated, so '
                                                               'the elasticity problem '
                                                               'is solvable only up to '
                                                               'a rigid body motion; '
                                                               'the Dirichlet set '
                                                               'exists to pin that '
                                                               'motion, not to carry '
                                                               'load. Any two minimal '
                                                               'sets give the same '
                                                               'stress and strain '
                                                               'energy but different '
                                                               'displacement fields.',
                                                'pitfalls': ['Never solve the '
                                                             'elasticity step with '
                                                             'bcs=[]. Signal: no '
                                                             'exception and no '
                                                             'zero-pivot message; with '
                                                             'pc_type lu '
                                                             '`problem.solver.getConvergedReason()` '
                                                             'returns 4 and the '
                                                             'displacement is silently '
                                                             'non-unique '
                                                             '(solver-dependent) or, '
                                                             'for a '
                                                             'non-self-equilibrated '
                                                             'load, of order 1e7 m '
                                                             'with true residual '
                                                             'norm(A*u - b)/norm(b) ~ '
                                                             '1.',
                                                             'For a single-component '
                                                             'point constraint, pass a '
                                                             'Function on the '
                                                             'COLLAPSED subspace. '
                                                             'Signal: passing a '
                                                             'fem.Constant or a numpy '
                                                             'array with a (sub, '
                                                             'collapsed) dof pair '
                                                             'gives `TypeError: '
                                                             '__init__(): incompatible '
                                                             'function arguments.`, '
                                                             'and calling '
                                                             'fem.locate_dofs_geometrical '
                                                             'on V.sub(i) alone gives '
                                                             '`RuntimeError: Cannot '
                                                             'tabulate coordinates for '
                                                             'a FunctionSpace that is '
                                                             'a subspace.`']},
                        'solver': {'REQUIRED': 'Both steps are LINEAR, so both are '
                                               'fem.petsc.LinearProblem, and '
                                               'petsc_options_prefix is a REQUIRED '
                                               'keyword argument in dolfinx 0.10:\n'
                                               '    prob = LinearProblem(a, L, '
                                               'bcs=bcs, '
                                               'petsc_options_prefix="therm_",\n'
                                               '                         '
                                               'petsc_options={"ksp_type": "cg", '
                                               '"pc_type": "hypre"})\n'
                                               '    uh = '
                                               'prob.solve()                       # '
                                               'returns the Function, not a tuple\n'
                                               '    assert '
                                               'prob.solver.getConvergedReason() > 0\n'
                                               'solve() does NOT raise on failure - '
                                               'you must check getConvergedReason() '
                                               'yourself, or pass '
                                               '"ksp_error_if_not_converged": True.',
                                   'OPTIONAL': 'Thermal: cg + hypre (SPD). Elasticity: '
                                               'cg + gamg, or {"ksp_type": "preonly", '
                                               '"pc_type": "lu", '
                                               '"pc_factor_mat_solver_type": "mumps"} '
                                               'for small problems. Tighten ksp_rtol '
                                               'to 1e-10/1e-12 if you intend to assert '
                                               'bounds on the temperature; the default '
                                               'rtol leaves errors far larger than '
                                               '1e-8.',
                                   'explanation': 'Both operators are symmetric '
                                                  'positive definite once the '
                                                  'Dirichlet sets are in place, so CG '
                                                  'is the natural Krylov method and '
                                                  'the only real question is the '
                                                  'preconditioner.',
                                   'pitfalls': ['Never trust a solve without reading '
                                                'getConvergedReason(). Signal: a singular '
                                                'elasticity system (bcs=[]) returns reason 4 '
                                                '(CONVERGED_ITS) from pc_type lu while the answer '
                                                'is not unique -- garbage of order 1e6-1e9 m for '
                                                'a load with a net resultant such as gravity; '
                                                'with plain cg (default preconditioner) the '
                                                'reason is -4 (DIVERGED_DTOL) for such a load but '
                                                '2 (converged) for a self-equilibrated thermal '
                                                'load, and cg + hypre or cg + gamg give -8 '
                                                '(DIVERGED_INDEFINITE_PC) or -10 '
                                                '(DIVERGED_INDEFINITE_MAT); none of these raises '
                                                'an exception.',
                                                'Do not assert tight bounds on the '
                                                'temperature at the default ksp_rtol. '
                                                'Signal: a check like `T.x.array.max() '
                                                '<= T_hot + 1e-8` fails on a correct '
                                                'model because CG stopped several '
                                                'orders of magnitude short of that.']},
                        'coupling': {'REQUIRED': 'One-way (the default, and what '
                                                 'almost every thermo-mechanical '
                                                 'problem needs):\n'
                                                 '  1. solve the thermal problem -> '
                                                 'Function T\n'
                                                 '  2. use T directly inside L_u; '
                                                 'nothing is iterated.\n'
                                                 'Two-way (only when the deformation '
                                                 'genuinely changes the heat problem, '
                                                 'e.g. conduction on the deformed '
                                                 'configuration or a strain-dependent '
                                                 'conductivity):\n'
                                                 '    F = ufl.Identity(tdim) + '
                                                 'ufl.grad(u_k)\n'
                                                 '    J = ufl.det(F)\n'
                                                 '    a_T = '
                                                 'kappa*J*ufl.inner(ufl.inv(F).T*ufl.grad(Tt),\n'
                                                 '                            '
                                                 'ufl.inv(F).T*ufl.grad(s))*ufl.dx\n'
                                                 '    for it in range(max_it):\n'
                                                 '        T_old.x.array[:] = '
                                                 'T.x.array\n'
                                                 '        prob_T.solve()      # built '
                                                 'with u=T so it updates T in place\n'
                                                 '        prob_u.solve()      # built '
                                                 'with u=u_k\n'
                                                 '        r = '
                                                 'sqrt(fem.assemble_scalar(fem.form(ufl.inner(T-T_old, '
                                                 'T-T_old)*ufl.dx))\n'
                                                 '                 / '
                                                 'fem.assemble_scalar(fem.form(ufl.inner(T, '
                                                 'T)*ufl.dx)))\n'
                                                 '        if r < 1e-8: break',
                                     'OPTIONAL': 'Pass u=T and u=u_k to the two '
                                                 'LinearProblem constructors so each '
                                                 'solve updates the existing Function '
                                                 'in place; that is what lets the '
                                                 'other form see the new iterate. A '
                                                 'relative Picard tolerance of 1e-6 to '
                                                 '1e-8 is reachable in a handful of '
                                                 'sweeps.',
                                     'explanation': 'The Picard residual is the '
                                                    'relative change of the field '
                                                    'between two sweeps, not a PDE '
                                                    'residual. It contracts '
                                                    'geometrically, and its first '
                                                    'value tells you at once whether '
                                                    'the two-way term matters at all '
                                                    'for your parameters.',
                                     'pitfalls': ['Do not add Picard iteration by '
                                                  'reflex. Signal: for metals the '
                                                  'relative change of the displacement '
                                                  'after the FIRST coupling sweep is '
                                                  'only about a tenth of the thermal '
                                                  'strain alpha*(T-T_ref), so the '
                                                  'one-shot answer is already correct '
                                                  'to several digits and the loop '
                                                  'exits immediately.']},
                        'materials': {'REQUIRED': 'E [Pa], nu [-], alpha [1/K], kappa '
                                                  '[W/(m K)], T_ref [K], all as '
                                                  'fem.Constant.\n'
                                                  '3D and PLANE STRAIN:\n'
                                                  '    mu  = E/(2*(1+nu))\n'
                                                  '    lam = E*nu/((1+nu)*(1-2*nu))\n'
                                                  '    beta = (3*lam + '
                                                  '2*mu)*alpha        # = '
                                                  'E*alpha/(1-2*nu)\n'
                                                  'PLANE STRESS (2D only) - BOTH '
                                                  'constants change:\n'
                                                  '    lam_ps = 2*lam*mu/(lam + 2*mu)\n'
                                                  '    beta_ps = (2*lam_ps + '
                                                  '2*mu)*alpha  # = E*alpha/(1-nu)\n'
                                                  'Never mix a plane-stress lambda '
                                                  'with a plane-strain beta.',
                                      'OPTIONAL': 'E in [1e3, 1e12] Pa; nu in [0.0, '
                                                  '0.499] (above ~0.45 use a mixed u-p '
                                                  'formulation instead); alpha in '
                                                  '[1e-7, 1e-4] 1/K - steel 12e-6, '
                                                  'aluminium 23e-6, concrete 10e-6; '
                                                  'kappa order 45 W/(m K) for steel, '
                                                  '200 for aluminium. Keep one '
                                                  'consistent unit system: SI gives '
                                                  'displacements of order 1e-3 m for a '
                                                  '1 m bar at 100 K.',
                                      'explanation': 'beta, not alpha, is what '
                                                     'multiplies (T - T_ref) in the '
                                                     'stress. Writing beta out as '
                                                     'E*alpha/(1-2*nu) (plane strain / '
                                                     '3D) or E*alpha/(1-nu) (plane '
                                                     'stress) is the quickest way to '
                                                     'check you have the right one.',
                                      'pitfalls': ['Plane strain and plane stress are '
                                                   'not interchangeable. Signal: no '
                                                   'error; at the same E, nu and alpha '
                                                   'the plane-strain model gives a '
                                                   'visibly larger in-plane expansion '
                                                   'and larger deviatoric stress than '
                                                   'plane stress, and the gap widens '
                                                   'as nu grows.',
                                                   'T_ref is part of the material '
                                                   'data, not a cosmetic offset. '
                                                   'Signal: no error; leaving T_ref at '
                                                   '0 for an SI problem near room '
                                                   'temperature multiplies the '
                                                   'displacement several-fold and, in '
                                                   'a constrained body, the deviatoric '
                                                   'stress too.']},
                        'pitfalls': ['[Numerical] The thermal term is the ONLY place '
                                     'the temperature enters the elasticity form, and '
                                     'it must carry the thermal stress modulus beta = '
                                     '(3*lam + 2*mu)*alpha, not alpha. Two distinct '
                                     'failures were reproduced. (a) Omitting the term '
                                     'entirely. Signal: no error message whatsoever; '
                                     'when there is no other load the solve returns a '
                                     'displacement that is BIT-EXACTLY zero everywhere '
                                     '- `np.all(uh.x.array == 0.0)` is True - not '
                                     'merely small, and not only at the unconstrained '
                                     'boundaries as sometimes claimed. If a body force '
                                     'or traction is also present the displacement is '
                                     'NOT zero, it is silently just the isothermal '
                                     "answer, so 'zero displacement' is only a "
                                     'reliable tell for a thermally-loaded-only model. '
                                     '(b) Using alpha where (3*lam + 2*mu)*alpha '
                                     'belongs. Signal: no error; max(abs(uh.x.array)) '
                                     'drops to about 1e-15 for SI steel constants, '
                                     'i.e. by the factor 3*lam + 2*mu, which reads as '
                                     'zero on any plot. Both were reproduced at '
                                     'degrees 1 and 2 on triangles, quadrilaterals, '
                                     'tetrahedra and hexahedra, in 2D and 3D. '
                                     '(Executed on dolfinx 0.10.0.)',
                                     '[API] The thermal term contains no trial '
                                     'function, so it is a linear functional and '
                                     'belongs in L, never in the bilinear form. '
                                     'Signal: writing `a = inner(sigma_elastic(u) - '
                                     'beta*(T-T_ref)*Id, eps(v))*dx` and handing it to '
                                     'fem.petsc.LinearProblem raises, at form-compile '
                                     'time, '
                                     '`ufl.algorithms.check_arities.ArityMismatch: '
                                     'Adding expressions with non-matching form '
                                     "arguments () vs ('v_1',).` The traceback ends in "
                                     'ufl/algorithms/check_arities.py. The fix is L = '
                                     'beta*(T - T_ref)*div(v)*dx, using '
                                     'inner(c*Identity(d), eps(v)) == c*div(v). '
                                     '(Executed on dolfinx 0.10.0.)',
                                     '[API] After any form-compile failure, the FFCx '
                                     'JIT cache is left holding a stale generated .c '
                                     'file, and running the SAME script again hides '
                                     'the real error. Signal: the second run does not '
                                     'repeat the ArityMismatch (or whatever the true '
                                     'error was); instead it raises, after a pause, '
                                     '`FileExistsError: [Errno 17] File exists: '
                                     "'<cache_dir>/libffcx_forms_<hash>.c'` chained "
                                     'into `TimeoutError: JIT compilation timed out, '
                                     'probably due to a failed previous compile. Try '
                                     'cleaning cache (e.g. remove '
                                     '<cache_dir>/libffcx_forms_<hash>.c) or increase '
                                     'timeout option.` (the real message prints the '
                                     'absolute path of that file). Delete the named '
                                     'file and re-run to see the actual error again. '
                                     'This bites every time a form is edited and '
                                     're-run after a compile failure, whatever the '
                                     'physics. (Executed on dolfinx 0.10.0.)',
                                     '[Physics] In 2D, the textbook shorthand sigma = '
                                     'C:(eps(u) - alpha*dT*I) is WRONG if you spell I '
                                     'as ufl.Identity(2) while using plane-strain Lame '
                                     'constants, because tr(Identity(2)) is 2 and not '
                                     '3: the thermal modulus silently becomes (2*lam + '
                                     '2*mu)*alpha instead of (3*lam + 2*mu)*alpha. '
                                     'Signal: no error, no warning; the free thermal '
                                     'expansion of an unconstrained body comes out as '
                                     'the plane-STRESS value while the stiffness is '
                                     'still plane strain, so the whole displacement '
                                     'field is uniformly too small by a fixed factor '
                                     'that grows with nu. This is 2D-ONLY: in 3D the '
                                     'same expression is exactly correct (the two '
                                     'forms agree to every digit), which is why the '
                                     'mistake survives a 3D test. Always write the '
                                     'thermal term explicitly as (3*lam + '
                                     '2*mu)*alpha*(T - T_ref). (Executed on dolfinx '
                                     '0.10.0.)',
                                     '[Physics] Plane strain and plane stress need '
                                     'DIFFERENT lambda AND a different thermal '
                                     'modulus. Plane strain / 3D: lam = '
                                     'E*nu/((1+nu)*(1-2*nu)) with beta = '
                                     '(3*lam+2*mu)*alpha = E*alpha/(1-2*nu). Plane '
                                     'stress: lam_ps = 2*lam*mu/(lam+2*mu) with '
                                     'beta_ps = (2*lam_ps+2*mu)*alpha = '
                                     'E*alpha/(1-nu). Signal: nothing is raised; the '
                                     'two models simply differ, plane strain giving '
                                     'the larger in-plane expansion and the larger '
                                     'deviatoric stress at the same E, nu, alpha, with '
                                     'the gap widening as nu grows. Two specific wrong '
                                     'recipes were checked: substituting lam_ps into '
                                     '(3*lam+2*mu)*alpha overshoots the expansion, and '
                                     "substituting E' = E/(1-nu**2) into the "
                                     'plane-strain lambda (a recipe that appears in '
                                     'older notes) does NOT produce a large error at '
                                     'nu = 0.3 - it lands within a couple of percent '
                                     'of plain plane strain, so it is a silent wrong '
                                     'model rather than an obvious one. Validate the '
                                     '2D choice against a 3D run with u_z constrained '
                                     '(plane strain) or with free z faces (plane '
                                     'stress). (Executed on dolfinx 0.10.0.)',
                                     '[Numerical] A thermal load is self-equilibrated, '
                                     'so with no displacement Dirichlet condition the '
                                     'stiffness matrix is singular: a dense eigenvalue '
                                     'check shows exactly 3 (2D) or 6 (3D) eigenvalues '
                                     'below 1e-8 times the largest, the rigid '
                                     'translations and rotations. What actually '
                                     'happens on solve is NOT a hang and NOT a '
                                     'zero-pivot report - that previously quoted '
                                     'behaviour does NOT reproduce. Signal: with '
                                     '{"ksp_type": "preonly", "pc_type": "lu"} '
                                     "(PETSc's own LU, MUMPS, or Cholesky) "
                                     '`problem.solver.getConvergedReason()` returns 4 '
                                     '(CONVERGED_ITS), no message of any kind is '
                                     'printed, and for a self-equilibrated thermal '
                                     'load you get a finite but solver-dependent '
                                     'displacement (different LU packages give '
                                     'different max(abs(u)) while the strain energy is '
                                     'identical), while for a load with a net '
                                     'resultant such as gravity you get max(abs(u)) of '
                                     'order 1e6-1e9 m with a true residual '
                                     'norm(A*u-b)/norm(b) between about 1 and 1000. '
                                     'With {"ksp_type": "cg"} a load with a net '
                                     'resultant gives reason -4 (DIVERGED_DTOL), and '
                                     'adding "ksp_error_if_not_converged": True raises '
                                     '`petsc4py.PETSc.Error: error code 91` with the '
                                     'line `KSPSolve() has not converged, reason '
                                     'DIVERGED_DTOL`; for the self-equilibrated '
                                     'thermal load cg instead converges (reason 2) to '
                                     'one member of the non-unique solution family and '
                                     'nothing is raised, so a missing Dirichlet set is '
                                     'not detected that way. Fix by adding a minimal '
                                     'constraint set (3 scalar constraints in 2D, 6 in '
                                     '3D); any two minimal sets give the same strain '
                                     'energy and the same stress. (Executed on dolfinx '
                                     '0.10.0.)',
                                     '[Input] The reference temperature is real '
                                     'physics: the thermal strain is alpha*(T - '
                                     'T_ref), so leaving T_ref at 0 in an SI model at '
                                     'room temperature adds a constant pre-strain of '
                                     'order alpha*300. Signal: no error and no warning '
                                     'of any kind. In a body held by a MINIMAL '
                                     'constraint set the deviatoric stress is the SAME '
                                     'TO ROUND-OFF with T_ref = 0 and with T_ref = '
                                     '300 K and only the displacement shifts (a '
                                     'uniform stress-free expansion); in a body with a '
                                     'clamped face the displacement grows several-fold '
                                     'AND the deviatoric stress grows by a similar '
                                     'factor. CORRECTION on how to test that first '
                                     'case: an earlier version of this entry said the '
                                     'deviatoric stress is BIT-IDENTICAL, and it is '
                                     'not - the two runs are separate linear solves and '
                                     'differ in their last digits, so np.array_equal '
                                     'returns False and that is not a bug. Compare with '
                                     'a relative norm and expect agreement at round-off, '
                                     'not equality. The often-repeated claim that this '
                                     'makes '
                                     "'the Newton iteration oscillate' does NOT "
                                     'reproduce and cannot: the one-way thermo-elastic '
                                     'step is linear, and solving it through '
                                     'fem.petsc.NonlinearProblem/SNES converges in '
                                     'exactly 1 Newton iteration for either T_ref, to '
                                     'a displacement identical to the LinearProblem '
                                     'answer to the last bit. (Executed on dolfinx '
                                     '0.10.0.)',
                                     '[Integration] Two-way thermo-mechanical coupling '
                                     'needs a real feedback term before Picard '
                                     'iteration means anything. A concrete one that '
                                     'works in dolfinx is conduction on the DEFORMED '
                                     'configuration pulled back to the reference mesh: '
                                     'F = Identity(d) + grad(u), J = det(F), a_T = '
                                     'kappa*J*inner(inv(F).T*grad(T), '
                                     'inv(F).T*grad(s))*dx. Alternating solves then '
                                     'converge geometrically in the relative change '
                                     'norm(T_new - T_old)/norm(T_new) computed with '
                                     'fem.assemble_scalar. Signal: no error either '
                                     'way; the size of the effect is what matters, and '
                                     'the relative one-shot error in the displacement '
                                     'is proportional to the thermal strain alpha*(T - '
                                     'T_ref) - roughly a tenth of it. For metals that '
                                     'strain is order 1e-3, so a single non-iterated '
                                     'pass is already accurate to several digits and '
                                     'the Picard loop exits on the first test; the '
                                     'loop only pays for itself when the thermal '
                                     'strain reaches a few percent. The older claim '
                                     "that the one-shot error is 'of order "
                                     "alpha*DeltaT*L' overstates it by roughly the "
                                     'inverse of the thermal strain. (Executed on '
                                     'dolfinx 0.10.0.)']},
 'biharmonic': {'description': 'Biharmonic equation (4th order): laplacian^2(u) = f. '
                               'Used for Kirchhoff plates, stream function '
                               'formulation. Requires DG or C1 elements.',
                'weak_form_ip': 'inner(div(grad(u)), div(grad(v)))*dx - '
                                'inner(avg(div(grad(u))), jump(grad(v),n))*dS - '
                                'inner(jump(grad(u),n), avg(div(grad(v))))*dS + '
                                'alpha/h*inner(jump(grad(u),n), jump(grad(v),n))*dS',
                'method': 'Interior Penalty (IP-DG): C0 elements with penalty on '
                          'gradient jumps',
                'function_space': 'Lagrange order 2 (with interior penalty for C0 '
                                  'elements)',
                'demo_url': 'https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_biharmonic.html',
                'alternative': 'Hermite elements (C1 conforming) — avoids DG penalty '
                               'terms but limited to simplices',
                'solver': 'LU (direct) for moderate sizes, GMRES for large',
                'pitfalls': ['[Numerical] Penalty parameter alpha must be large enough '
                             "for coercivity, but the standard 'alpha = 4*(k+1)^2' "
                             'rule of thumb is a STABILITY floor, not an accuracy '
                             'optimum. Signal: [MEASURED 2026-08-03, dolfinx 0.10.0; '
                             'a C0-IP manufactured solution on a simply-supported '
                             'square, L2 error over a refinement ladder] P2 at the '
                             'rule-of-thumb alpha = 4*(k+1)^2 and P2 at a much '
                             'smaller alpha converge at the SAME order, but the '
                             'small-alpha run carries an error constant several times '
                             'lower. Run your own ladder at two penalties and compare '
                             'the constants; the numbers are not reproduced here '
                             'because a convergence study is only evidence when it is '
                             'yours. IMPORTANT '
                             'CORRECTION: too-small alpha does NOT make the solution '
                             'norm diverge under mesh refinement. Measured the other '
                             'way round — a drastically under-penalised run gives a '
                             'HUGE coarse-mesh error that then converges FASTER than '
                             'the nominal rate as the mesh is refined. The '
                             'observable for an under-penalised C0-IP scheme is a '
                             'blown-up error CONSTANT on coarse meshes, not divergence '
                             'under refinement.',
                             '[API] h_E (cell-size measure for the penalty weight) '
                             'should use ufl.CellDiameter / ufl.FacetArea so the '
                             'penalty tracks the local element size on graded / '
                             'locally refined meshes. Signal: [MEASURED 2026-08-03] on '
                             'a UNIFORM refinement sequence, hard-coding h = 1/8 as a '
                             'fem.Constant while refining from N=16 to N=128 does NOT '
                             'break convergence — the measured P2 L2 rates are 2.65 / '
                             '2.59 / 2.43, at least as good as the CellDiameter form. '
                             'IMPORTANT CORRECTION: the previously quoted signal '
                             "('convergence rate degrades from O(h^2) to ~O(h) or "
                             "stagnates') is NOT reproducible on uniform meshes. Treat "
                             'this as a graded-mesh concern only, and diagnose it by '
                             'comparing local penalty magnitudes rather than by '
                             'watching a global rate.',
                             '[Performance] Interior penalty requires interior facet '
                             'integrals (dS) — more expensive than standard FEM (each '
                             'facet visited from both sides). Signal: time the C0-IP '
                             'assemble_matrix against a Poisson assemble_matrix on the '
                             'SAME space and mesh; the C0-IP form costs several times '
                             'the Poisson one, and the earlier wording of this entry '
                             "('5-10x') is a LOWER BOUND rather than a range — the "
                             'measured ratio can sit well above it, so do not treat a '
                             'larger factor as a sign that something is wrong. When '
                             'comparing against the mixed (u + auxiliary sigma) '
                             'alternative, which avoids dS at the cost of doubling the '
                             'DOF count, normalise PER DOF: otherwise the doubled space '
                             'is confounded with the facet-integral cost and the '
                             'comparison measures the wrong thing. (Verified by execution on dolfinx 0.10.0, 2026-08-06.)',
                             '[Numerical] Alternative: split into two 2nd-order '
                             'equations (mixed method with auxiliary variable). '
                             'Signal: [MEASURED 2026-08-03, dolfinx 0.10.0] writing '
                             'the naive single 4th-order form inner(div(grad(u)), '
                             'div(grad(v)))*dx on a C0 Lagrange space raises NOTHING — '
                             'it compiles and assembles cleanly (P2 on an 8x8 unit '
                             'square: 3073 nonzeros), and on P1 it assembles an '
                             'IDENTICALLY ZERO matrix (497 stored nonzeros, max '
                             '|entry| = 0.0) because div(grad(.)) of a P1 function '
                             'vanishes cell-wise. IMPORTANT CORRECTION: dolfinx does '
                             'NOT raise `NotImplementedError: H2 conformity required` '
                             '(no such error exists) and it does NOT silently '
                             'substitute the interior-penalty form. The failure is '
                             'silent and numerical: a singular/inconsistent operator. '
                             'You must write the dS interior-penalty terms yourself, '
                             'or use the mixed (u, sigma) split with sigma = '
                             'Laplacian(u) on P1 x P1.']},
 'helmholtz': {'description': 'Helmholtz equation: -laplacian(u) - k^2*u = f. '
                              'Acoustic/optical wave propagation. Can be '
                              'complex-valued.',
               'weak_form': 'inner(grad(u), grad(v))*dx - k**2 * inner(u, v)*dx = '
                            'inner(f, v)*dx',
               'function_space': 'Lagrange order 2+ (need ~10 points per wavelength '
                                 'for accuracy)',
               'demo_url': 'https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_helmholtz.html',
               'complex_valued': {'description': 'Helmholtz with complex '
                                                 'source/solution requires '
                                                 'complex-valued PETSc build',
                                  'scalar_type': 'np.complex128',
                                  'notes': 'DOLFINx supports float32, float64, '
                                           'complex64, complex128 scalar types'},
               'absorbing_bc': {'description': 'First-order absorbing BC: du/dn = '
                                               '-ik*u on artificial boundary',
                                'implementation': 'Add -1j*k*inner(u,v)*ds to bilinear '
                                                  'form'},
               'solver': 'GMRES + LU (direct) for moderate sizes. Indefinite system — '
                         'CG does NOT work.',
               'pitfalls': ["[Numerical] Need a fine mesh, and '~10 points per "
                            "wavelength' is a FLOOR that is far too loose for P1. "
                            'Signal: [MEASURED 2026-08-03, dolfinx 0.10.0; a '
                            'manufactured solution on the unit square with Dirichlet '
                            'data interpolated from it, P1, non-resonant k chosen so '
                            'k^2/pi^2 stays away from any m^2+n^2] IMPORTANT '
                            'CORRECTION: for k*h >~ 1 the scheme does not converge at '
                            '~O(h) — it does not converge AT ALL (relative error >= 1 '
                            'with NEGATIVE measured rates). Clean O(h^2) only returns '
                            'once k*h <~ 0.5. The pollution effect is visible as the '
                            'required points-per-wavelength for a fixed accuracy '
                            'growing with k (33 pts/wave gives 2.6% at k=12 but 29 '
                            'pts/wave still gives 14% at k=55). PRACTICAL WARNING for '
                            'MMS tests: pick k away from resonance — k^2 near '
                            'pi^2*(m^2+n^2) makes the discrete system near-singular '
                            'and the error study meaningless.',
                            '[Numerical] System is INDEFINITE — standard CG diverges. '
                            'Use GMRES or a direct solver. Signal: [re-verified on '
                            'dolfinx 0.10.0, k = 20 on a 32x32 unit square, P1] with '
                            "petsc_options={'ksp_type': 'cg', 'pc_type': 'icc'} PETSc "
                            'stops after 4 iterations and '
                            'problem.solver.getConvergedReason() returns -10, which is '
                            'DIVERGED_INDEFINITE_MAT in PETSc.KSP.ConvergedReason '
                            '(DIVERGED_INDEFINITE_PC is -8, a different code); '
                            "'pc_type' 'jacobi', 'none' and 'ilu' also give -10, after "
                            '2-4 iterations. problem.solve() itself does not raise. '
                            'For ~< 100k DOFs use LU; for larger meshes use GMRES + a '
                            'shifted-Laplacian preconditioner.',
                            '[Numerical] High wavenumber k: requires specialized '
                            'preconditioners (shifted Laplacian). Signal: with default '
                            'ILU, GMRES on a high-k problem does NOT settle at a small '
                            'stagnation residual - it runs out the iteration budget, '
                            '`getConvergedReason()` returns -3 (DIVERGED_MAX_IT), and '
                            'the true relative residual it stops at is LARGER than the '
                            'right-hand side, i.e. worse than the zero initial guess. '
                            'Read the reason code and the true relative residual; a '
                            'threshold on the residual alone will miss it. THREE '
                            'CORRECTIONS to the way this used to be written. (1) The '
                            "'stagnates at residual ~1e-2' number does not reproduce; "
                            'see above. (2) ILU and Jacobi do not behave alike - on a '
                            'small enough problem Jacobi converges, so do not treat '
                            "'ILU/Jacobi' as one failing pair; test the preconditioner "
                            'you actually intend to use. (3) The shifted-Laplacian '
                            'preconditioner (assemble the SAME form with a complex '
                            'shift on the k^2 term and apply its inverse as the PC) '
                            'does NOT restore a fixed handful of iterations. Its '
                            'iteration count GROWS with k on a fixed mesh, which is the '
                            'known behaviour of an O(1) shift, so measure it at two '
                            'wavenumbers on the same mesh and expect a rising count '
                            'rather than a flat one. All of this needs a complex PETSc '
                            'build; on a real build the form will not even compile. (Verified by execution on dolfinx 0.10.0, 2026-08-06.)',
                            '[API] Complex mode: PETSc must be built with '
                            '--with-scalar-type=complex; the default conda-forge '
                            'fenics-dolfinx build is REAL. Verify with '
                            'numpy.issubdtype(dolfinx.default_scalar_type, '
                            'numpy.complexfloating) before building the form. Signal: '
                            '[exact text measured on a REAL build, dolfinx 0.10.0 / '
                            'PETSc 3.24.4] dolfinx.fem.form on a form carrying an '
                            'imaginary coefficient raises ValueError: Unexpected '
                            'complex value in real expression., and writing a complex '
                            'number into a Function array raises TypeError: float() '
                            'argument must be a string or a real number, not '
                            "'complex'. Function.interpolate of a complex callable "
                            'silently drops the imaginary part with ComplexWarning: '
                            'Casting complex values to real discards the imaginary '
                            "part. The previously quoted 'TypeError: cannot convert "
                            "complex to real' does NOT appear anywhere."]},
 'maxwell': {'description': "Maxwell's equations for electromagnetic wave propagation. "
                            'Curl-curl formulation. Requires H(curl) (Nedelec) '
                            'elements.',
             'weak_form_curl_curl': 'inner(curl(E), curl(v))*dx - k0**2 * epsilon_r * '
                                    'inner(E, v)*dx = inner(J, v)*dx',
             'function_space': 'Nedelec 1st kind (N1curl) — H(curl) conforming, '
                               'tangential continuity',
             'demos': {'scattering_wire': 'https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_scattering_boundary_conditions.html',
                       'scattering_pml': 'https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_pml.html',
                       'waveguide_modes': 'https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_half_loaded_waveguide.html',
                       'axisymmetric_sphere': 'https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_axis.html'},
             'pml': {'description': 'Perfectly Matched Layer — artificial absorbing '
                                    'boundary layer',
                     'implementation': 'Complex-valued coordinate stretching '
                                       'transforms Maxwell equations in PML region'},
             'eigenvalue': {'description': 'Electromagnetic modal analysis — find '
                                           'waveguide modes using SLEPc EPS',
                            'elements': 'N1curl (Nedelec) for transverse + Lagrange '
                                        'for axial component on quads',
                            'solver': 'SLEPc Krylov-Schur with spectral transformation '
                                      '(shift-and-invert)'},
             'solver': 'GMRES + AMS (auxiliary-space Maxwell solver from hypre) for '
                       'curl-curl',
             'pitfalls': ['[Physics] MUST use H(curl) elements (Nedelec / N1curl) for '
                          'Maxwell — standard Lagrange spaces lack the tangential '
                          'continuity that the physical fields require. Signal: '
                          'dolfinx.fem.form does NOT fail at form construction '
                          '(ufl.curl is accepted on vector Lagrange and even on scalar '
                          'Lagrange in 2D), so the bug is silent at compile/assemble '
                          'time. The observable failure is numerical: the '
                          'post-processed B = curl(A) field has spurious normal jumps '
                          'at element interfaces, and convergence against an analytic '
                          'test (e.g., uniform B in a cavity) plateaus at ~10% error '
                          'regardless of refinement. (Verified empirically 2026-06-01 '
                          "— prior catalog wording 'violates physical constraints' "
                          'implied a syntactic/assembly-time rejection; in current '
                          'dolfinx the form compiles fine and the bug surfaces in the '
                          'field values.)',
                          '[Syntax] Complex-valued Maxwell: PETSc must be compiled '
                          'with --with-scalar-type=complex. Signal: [exact text '
                          're-measured 2026-08-03 on a REAL conda-forge build, dolfinx '
                          '0.10.0 / PETSc 3.24.4] building the form raises ValueError '
                          "'Unexpected complex value in real expression.' at "
                          'dolfinx.fem.form(...) — before assemble_vector is ever '
                          'reached. Writing a complex value into a real Function array '
                          'raises TypeError "float() argument must be a string or a '
                          'real number, not \'complex\'". The previously quoted '
                          "strings 'cannot convert complex to float' / 'imaginary part "
                          "discarded' do NOT appear. Check with "
                          'numpy.issubdtype(dolfinx.default_scalar_type, '
                          'numpy.complexfloating) before building the form.',
                          '[Numerical] PML (Perfectly Matched Layer): requires '
                          'coordinate stretching of the form x_i → x_i*(1 + '
                          'i*sigma(x_i)/omega) inside the PML region. A real-only '
                          'stretching (real sigma) gives a lossy real boundary, NOT a '
                          'radiating PML. Signal: a fem.Function evaluated in the PML '
                          'region decays by orders of magnitude only when the '
                          'coordinate-stretch coefficient is constructed with '
                          'numpy.complex128 ScalarType — with a real-only stretch the '
                          'dolfinx.fem.assemble_vector output shows a standing-wave '
                          'reflection back into the domain.',
                          '[Numerical] Low-frequency breakdown: curl-curl + '
                          'omega^2-mass formulation becomes ill-conditioned as omega → '
                          '0 because the gradient kernel of curl is no longer '
                          'regularised by the mass term. Use mixed (A, phi) '
                          'formulation with a Lagrange multiplier on the divergence. '
                          'Signal: KSP iteration count for GMRES + AMS preconditioner '
                          'explodes as omega is reduced below ~10^-3 of the lowest '
                          'cavity eigenvalue; condition number printed by PETSc grows '
                          'as 1/omega^2.',
                          "[API] Edge elements (basix.ElementFamily.N1E / 'Nedelec 1st "
                          "kind H(curl)') have DOF ordering by edge, not by node: a "
                          'degree of freedom is the MOMENT of the field along an edge, '
                          'not a component of the field at a point. So a tangential '
                          'boundary condition must be INTERPOLATED into the edge basis '
                          '- build a fem.Function on the same N1curl space, '
                          '`f.interpolate(<the intended field>)`, and pass that '
                          'Function to fem.dirichletbc. Signal: the loud half is a '
                          'vector CONSTANT, which dolfinx refuses outright rather than '
                          'accepting quietly - `RuntimeError: Creating a DirichletBC '
                          'using a Constant is not supported when the Constant size is '
                          'not equal to the block size of the constrained (sub-)space. '
                          'Use a fem::Function to create the fem::DirichletBC.` The '
                          'SILENT mistake is one level lower and is the one to guard '
                          'against: writing the intended component values straight '
                          'into the boundary entries of the dof array by hand. Nothing '
                          'raises, because those entries are edge moments and a '
                          "component value is simply a different number - the field's "
                          'tangential trace on the boundary then comes out O(1) wrong '
                          'while the solve reports success. Check it by measuring the '
                          'tangential trace against the field you meant to impose: via '
                          'the interpolated Function it is at round-off, via '
                          'hand-written component values it is not. IMPORTANT '
                          'CORRECTION: an earlier version of this entry said the '
                          'dirichletbc silently sets only the first component on each '
                          'edge - it does not, the Constant spelling is rejected with '
                          'the message above, and an agent waiting for a silent '
                          'first-component bug will wait for ever. (Verified by execution on dolfinx 0.10.0, 2026-08-06.)']},
 'cahn_hilliard': {'description': 'Cahn-Hilliard equation: a nonlinear, '
                                  'time-dependent, fourth-order PDE that models phase '
                                  'separation (spinodal decomposition) of a binary '
                                  'mixture. dc/dt = div(M*grad(mu)) with mu = df/dc - '
                                  'lmbda*laplacian(c) and a double-well bulk energy '
                                  'f(c). It is solved as a coupled system of two '
                                  'second-order equations for the concentration c and '
                                  'the chemical potential mu on one mixed function '
                                  'space.',
                   'minimal_working_example': '# Cahn-Hilliard spinodal decomposition, '
                                              'dolfinx 0.10.\n'
                                              'from mpi4py import MPI\n'
                                              'import numpy as np\n'
                                              'import ufl\n'
                                              'import basix.ufl\n'
                                              'from dolfinx import mesh, fem\n'
                                              'from dolfinx.fem.petsc import '
                                              'NonlinearProblem\n'
                                              '\n'
                                              'lmbda = 1.0e-2      # gradient-energy '
                                              '(interface) parameter\n'
                                              'dt_val = 5.0e-6     # time step\n'
                                              'theta = 0.5         # 0.5 = '
                                              'Crank-Nicolson, 1.0 = backward Euler\n'
                                              'M = 1.0             # mobility\n'
                                              '\n'
                                              'msh = '
                                              'mesh.create_unit_square(MPI.COMM_WORLD, '
                                              '48, 48, mesh.CellType.triangle)\n'
                                              'P1 = basix.ufl.element("Lagrange", '
                                              'msh.basix_cell(), 1)\n'
                                              'ME = fem.functionspace(msh, '
                                              'basix.ufl.mixed_element([P1, P1]))\n'
                                              '\n'
                                              'u = fem.Function(ME)      # current '
                                              'step  (c, mu)\n'
                                              'u0 = fem.Function(ME)     # previous '
                                              'step (c0, mu0)\n'
                                              'q, v = ufl.TestFunctions(ME)\n'
                                              'c, mu = ufl.split(u)\n'
                                              'c0, mu0 = ufl.split(u0)\n'
                                              '\n'
                                              'rng = np.random.default_rng(2)\n'
                                              'u.x.array[:] = 0.0\n'
                                              'u.sub(0).interpolate(lambda x: 0.63 + '
                                              '0.02 * (0.5 - rng.random(x.shape[1])))\n'
                                              'u.x.scatter_forward()\n'
                                              'u0.x.array[:] = u.x.array\n'
                                              '\n'
                                              'cv = ufl.variable(c)\n'
                                              'f_chem = 100.0 * cv**2 * (1.0 - cv) ** '
                                              '2\n'
                                              'dfdc = ufl.diff(f_chem, cv)\n'
                                              '\n'
                                              'mu_mid = (1.0 - theta) * mu0 + theta * '
                                              'mu\n'
                                              'dt = fem.Constant(msh, dt_val)\n'
                                              '\n'
                                              'F0 = ((ufl.inner(c, q) - ufl.inner(c0, '
                                              'q)) * ufl.dx\n'
                                              '      + dt * M * '
                                              'ufl.inner(ufl.grad(mu_mid), '
                                              'ufl.grad(q)) * ufl.dx)\n'
                                              'F1 = (ufl.inner(mu, v) * ufl.dx\n'
                                              '      - ufl.inner(dfdc, v) * ufl.dx\n'
                                              '      - lmbda * ufl.inner(ufl.grad(c), '
                                              'ufl.grad(v)) * ufl.dx)\n'
                                              'F = F0 + F1\n'
                                              '\n'
                                              'problem = NonlinearProblem(\n'
                                              '    F, u, petsc_options_prefix="ch_",\n'
                                              '    petsc_options={"snes_type": '
                                              '"newtonls", "snes_rtol": 1e-10, '
                                              '"snes_atol": 1e-12,\n'
                                              '                   "snes_max_it": 50, '
                                              '"ksp_type": "preonly", "pc_type": '
                                              '"lu",\n'
                                              '                   '
                                              '"pc_factor_mat_solver_type": "mumps"})\n'
                                              '\n'
                                              'mass_form = fem.form(c * ufl.dx)\n'
                                              'energy_form = fem.form((100.0 * c**2 * '
                                              '(1 - c) ** 2\n'
                                              '                        + 0.5 * lmbda * '
                                              'ufl.inner(ufl.grad(c), ufl.grad(c))) * '
                                              'ufl.dx)\n'
                                              'm_start = '
                                              'msh.comm.allreduce(fem.assemble_scalar(mass_form), '
                                              'op=MPI.SUM)\n'
                                              'e_prev = '
                                              'msh.comm.allreduce(fem.assemble_scalar(energy_form), '
                                              'op=MPI.SUM)\n'
                                              'e_start = e_prev\n'
                                              'monotone = True\n'
                                              '\n'
                                              'for step in range(50):\n'
                                              '    u0.x.array[:] = u.x.array\n'
                                              '    problem.solve()\n'
                                              '    assert '
                                              'problem.solver.getConvergedReason() > '
                                              '0, problem.solver.getConvergedReason()\n'
                                              '    e = '
                                              'msh.comm.allreduce(fem.assemble_scalar(energy_form), '
                                              'op=MPI.SUM)\n'
                                              '    if e > e_prev + 1e-10:\n'
                                              '        monotone = False\n'
                                              '    e_prev = e\n'
                                              '\n'
                                              'm_end = '
                                              'msh.comm.allreduce(fem.assemble_scalar(mass_form), '
                                              'op=MPI.SUM)\n'
                                              'ch = u.sub(0).collapse().x.array\n'
                                              'print(f"SNES '
                                              'reason={problem.solver.getConvergedReason()} '
                                              '"\n'
                                              '      '
                                              'f"its(last)={problem.solver.getIterationNumber()}")\n'
                                              'print(f"mass drift |m_end - m_start| = '
                                              '{abs(m_end - m_start):.3e}  (must be '
                                              '~0)")\n'
                                              'print(f"free energy {e_start:.6f} -> '
                                              '{e_prev:.6f}, monotone decreasing = '
                                              '{monotone}")\n'
                                              'print(f"c range [{ch.min():.4f}, '
                                              '{ch.max():.4f}]  (phases separated if '
                                              'it spans ~0 to ~1)")\n',
                   'runtime_note': 'The example above runs 50 Newton solves on a 48x48 '
                                   'mixed P1/P1 mesh and took a little over five '
                                   'minutes on the reference machine. It is '
                                   'deliberately long enough for the free energy to '
                                   'fall and the phases to separate; if you only need '
                                   'to confirm the setup compiles and the SNES '
                                   'converges, cut the step count and check the SNES '
                                   'reason and the mass drift, which are meaningful '
                                   'from the first step.',
                   'function_space': {'REQUIRED': 'One mixed space holding BOTH '
                                                  'unknowns, two identical scalar '
                                                  'Lagrange elements:\n'
                                                  '  P1 = '
                                                  "basix.ufl.element('Lagrange', "
                                                  'msh.basix_cell(), 1)\n'
                                                  '  ME = '
                                                  'dolfinx.fem.functionspace(msh, '
                                                  'basix.ufl.mixed_element([P1, P1]))\n'
                                                  '  u  = dolfinx.fem.Function(ME)   # '
                                                  'current step\n'
                                                  '  u0 = dolfinx.fem.Function(ME)   # '
                                                  'previous step\n'
                                                  '  c, mu   = ufl.split(u)\n'
                                                  '  c0, mu0 = ufl.split(u0)\n'
                                                  '  q, v = ufl.TestFunctions(ME)\n'
                                                  'Both components must live on the '
                                                  'SAME space object so that one '
                                                  'NonlinearProblem solves the coupled '
                                                  'system.',
                                      'OPTIONAL': 'Both components may be raised '
                                                  'together to degree 2; degree 1 and '
                                                  'degree 2 were both executed on '
                                                  'triangles and on quadrilaterals and '
                                                  'all four combinations conserve mass '
                                                  'to round-off and decrease the free '
                                                  'energy monotonically. Cell type may '
                                                  'be triangle or quadrilateral. There '
                                                  'is no inf-sup condition here, so '
                                                  'the two components do NOT need '
                                                  'different degrees -- unlike '
                                                  'Stokes/Navier-Stokes, equal order '
                                                  'is correct. The mesh must resolve '
                                                  'the interface: with gradient '
                                                  'parameter lmbda the interface '
                                                  'thickness is of order sqrt(lmbda), '
                                                  'so the cell size should be several '
                                                  'times smaller than that.',
                                      'explanation': 'The fourth-order operator is '
                                                     'split into two second-order '
                                                     'equations by introducing the '
                                                     'chemical potential mu as a '
                                                     'second unknown, which lets '
                                                     'ordinary C0 Lagrange elements be '
                                                     'used instead of C1 elements.',
                                      'pitfalls': ['Set the initial concentration on '
                                                   'the first sub-space AND give it a '
                                                   'random perturbation: '
                                                   'u.sub(0).interpolate(lambda x: '
                                                   '0.63 + 0.02*(0.5 - '
                                                   'rng.random(x.shape[1]))). Signal: '
                                                   'any UNIFORM initial concentration '
                                                   'leaves the field frozen at that '
                                                   'value for the whole run -- '
                                                   'standard deviation exactly 0.0 '
                                                   'starting from a uniform 0.5, and '
                                                   'at round-off (order 1e-17) '
                                                   'starting from a uniform 0.63.',
                                                   'Build the residual from '
                                                   'ufl.split(u), never from '
                                                   'u.sub(0)/u.sub(1). Signal: the '
                                                   'u.sub(i) version compiles and '
                                                   'builds a NonlinearProblem without '
                                                   'complaint, then fails on the very '
                                                   "FIRST solve with 'PC failed due to "
                                                   "FACTOR_OTHER' and 'Nonlinear "
                                                   '<prefix> solve did not converge '
                                                   'due to DIVERGED_LINEAR_SOLVE '
                                                   "iterations 0' (SNES reason -3, KSP "
                                                   'reason -11), leaving the '
                                                   'concentration exactly at its '
                                                   'initial values.']},
                   'weak_form': {'REQUIRED': 'Two residual blocks summed into ONE form '
                                             'F, with the chemical-potential '
                                             'derivative obtained by automatic '
                                             'differentiation:\n'
                                             '  cv = ufl.variable(c)\n'
                                             '  f_chem = 100.0*cv**2*(1.0 - '
                                             'cv)**2          # double-well bulk '
                                             'energy\n'
                                             '  dfdc = ufl.diff(f_chem, cv)\n'
                                             '  mu_mid = (1.0 - theta)*mu0 + theta*mu\n'
                                             '  dt = dolfinx.fem.Constant(msh, '
                                             'dt_value)\n'
                                             '  F0 = ((ufl.inner(c, q) - ufl.inner(c0, '
                                             'q))*ufl.dx\n'
                                             '        + '
                                             'dt*M*ufl.inner(ufl.grad(mu_mid), '
                                             'ufl.grad(q))*ufl.dx)\n'
                                             '  F1 = (ufl.inner(mu, v)*ufl.dx\n'
                                             '        - ufl.inner(dfdc, v)*ufl.dx\n'
                                             '        - lmbda*ufl.inner(ufl.grad(c), '
                                             'ufl.grad(v))*ufl.dx)\n'
                                             '  F = F0 + F1\n'
                                             'ufl.variable() is REQUIRED before '
                                             'ufl.diff(): ufl.diff can only '
                                             'differentiate with respect to a '
                                             'ufl.variable, not with respect to a '
                                             'plain split component.',
                                 'OPTIONAL': 'The bulk energy may be any double well; '
                                             '100*c^2*(1-c)^2 is the standard demo '
                                             'choice and puts the two stable phases at '
                                             'c = 0 and c = 1. Its prefactor sets the '
                                             'well depth and therefore the time scale. '
                                             'The mobility M may be a constant or '
                                             'degenerate (M = c*(1-c)); with a '
                                             'degenerate mobility the mu equation '
                                             'stays the same and only the '
                                             'dt*M*grad(mu) term changes. theta may be '
                                             '1.0 (backward Euler, more dissipative) '
                                             'or 0.5 (Crank-Nicolson). No boundary '
                                             'term appears because the natural BCs are '
                                             'the physical ones (see '
                                             'boundary_conditions).',
                                 'explanation': 'F0 is the mass-balance equation '
                                                'integrated by parts once, F1 defines '
                                                'the chemical potential. Summing them '
                                                'into one residual and handing it to a '
                                                'single NonlinearProblem solves the '
                                                'coupled 2x2 block system.',
                                 'pitfalls': ['Call ufl.variable(c) before ufl.diff. '
                                              'Signal: ufl.diff(100*c**2*(1-c)**2, c) '
                                              'with a plain split component c raises '
                                              '"ValueError: Expecting a Variable or '
                                              'SpatialCoordinate in diff."; wrapping '
                                              'it first (cv = ufl.variable(c)) returns '
                                              'a VariableDerivative and works.']},
                   'boundary_conditions': {'REQUIRED': 'None. Cahn-Hilliard with the '
                                                       'natural boundary conditions '
                                                       'needs NO '
                                                       'dolfinx.fem.dirichletbc at all '
                                                       '-- pass bcs=[] or omit the '
                                                       'argument:\n'
                                                       '  problem = '
                                                       'NonlinearProblem(F, u, '
                                                       "petsc_options_prefix='ch_', "
                                                       'petsc_options={...})\n'
                                                       'Integrating both blocks by '
                                                       'parts and dropping the '
                                                       'boundary integrals imposes '
                                                       'grad(mu).n = 0 (no mass flux '
                                                       'through the wall) and '
                                                       'grad(c).n = 0 (the interface '
                                                       'meets the wall at 90 degrees). '
                                                       'Those are the standard '
                                                       'physical conditions.',
                                           'OPTIONAL': 'A wetting boundary condition '
                                                       'replaces grad(c).n = 0 by an '
                                                       'extra surface-energy term '
                                                       'added to F1 as a ds integral. '
                                                       'A Dirichlet condition on c is '
                                                       'imposed on the first sub-space '
                                                       'with the collapsed-Function '
                                                       'pattern: V0, _ = '
                                                       'ME.sub(0).collapse(); g = '
                                                       'dolfinx.fem.Function(V0); dofs '
                                                       '= '
                                                       'dolfinx.fem.locate_dofs_topological((ME.sub(0), '
                                                       'V0), gdim-1, facets); bc = '
                                                       'dolfinx.fem.dirichletbc(g, '
                                                       'dofs, ME.sub(0)).',
                                           'explanation': 'Because there are no '
                                                          'essential boundary '
                                                          'conditions, total mass is '
                                                          'exactly conserved by the '
                                                          'discretisation, which gives '
                                                          'a free correctness check.',
                                           'pitfalls': ['Pass bcs=[] (or omit bcs) '
                                                        'unless the physics really '
                                                        'calls for a wall condition. '
                                                        'Signal: with the natural BCs '
                                                        'the assembled integral of c '
                                                        'stays at its initial value to '
                                                        'round-off (drift ~1e-15 over '
                                                        '25 steps); adding a Dirichlet '
                                                        'condition on c over the whole '
                                                        'boundary raises that drift to '
                                                        '~2e-2, thirteen orders of '
                                                        'magnitude larger, because the '
                                                        'BC lets mass in and out.']},
                   'solver': {'REQUIRED': 'One SNES solve per time step through '
                                          'dolfinx.fem.petsc.NonlinearProblem; '
                                          'petsc_options_prefix is a REQUIRED '
                                          'keyword-only argument on dolfinx 0.10:\n'
                                          '  from dolfinx.fem.petsc import '
                                          'NonlinearProblem\n'
                                          '  problem = NonlinearProblem(F, u, '
                                          "petsc_options_prefix='ch_',\n"
                                          "      petsc_options={'snes_type': "
                                          "'newtonls', 'snes_rtol': 1e-10,\n"
                                          "                     'snes_atol': 1e-12, "
                                          "'snes_max_it': 50,\n"
                                          "                     'ksp_type': 'preonly', "
                                          "'pc_type': 'lu',\n"
                                          '                     '
                                          "'pc_factor_mat_solver_type': 'mumps'})\n"
                                          'Time loop -- copy the current state into u0 '
                                          'BEFORE each solve, then check convergence:\n'
                                          '  for step in range(n_steps):\n'
                                          '      u0.x.array[:] = u.x.array\n'
                                          '      problem.solve()\n'
                                          '      assert '
                                          'problem.solver.getConvergedReason() > 0\n'
                                          'The same NonlinearProblem object is reused '
                                          'for every step; only u0 and, if wanted, '
                                          'dt.value change. problem.solver is a '
                                          'petsc4py SNES.',
                              'OPTIONAL': 'dt may be ramped between steps via dt.value '
                                          '= new_dt because dt is a '
                                          'dolfinx.fem.Constant, so no form has to be '
                                          "rebuilt. 'snes_monitor': None and "
                                          "'snes_converged_reason': None add PETSc's "
                                          'own progress text. '
                                          "'snes_error_if_not_converged': True turns a "
                                          'diverged step into a raised exception. The '
                                          'Jacobian may be supplied explicitly as '
                                          'J=ufl.derivative(F, u); dolfinx forms the '
                                          'same one automatically otherwise.',
                              'explanation': 'The Cahn-Hilliard residual is nonlinear '
                                             'through the double-well term, so every '
                                             'time step is a Newton solve; the 2x2 '
                                             'block matrix is small and indefinite, so '
                                             'a sparse direct factorisation is the '
                                             'robust default.',
                              'pitfalls': ['Check problem.solver.getConvergedReason() '
                                           '> 0 after every step. Signal: '
                                           'problem.solve() returns the solution '
                                           'Function without raising even when Newton '
                                           'failed, so a time loop happily marches on '
                                           'with a non-converged state.',
                                           'Copy u into u0 INSIDE the loop, before '
                                           'each solve. Signal: if u0 is never '
                                           'refreshed, every step re-solves the '
                                           'identical problem, so SNES reports 0 '
                                           'iterations from the second step onward and '
                                           'the concentration stops evolving after a '
                                           'single step (its spread stays at the '
                                           'initial noise level while a correct run '
                                           'has already separated).',
                                           'Set snes_max_it generously (50, not the '
                                           'PETSc default). Signal: the Newton '
                                           'iteration count is not constant -- it '
                                           'climbs steeply while the phases are '
                                           'separating (3, 4, 10, 15, 19 over the '
                                           'first five steps, and 21 at step 50 of the '
                                           'working example; about 17 on average at '
                                           'degree 1 and degree 2 alike, on triangles '
                                           'and on quadrilaterals) and only falls '
                                           'again afterwards, so a tight cap turns a '
                                           'perfectly healthy run into DIVERGED_MAX_IT '
                                           'part-way through.']},
                   'time_integration': {'REQUIRED': 'theta-weighted (Crank-Nicolson at '
                                                    'theta=0.5) treatment of the '
                                                    'chemical potential in the '
                                                    'mass-balance block:\n'
                                                    '  mu_mid = (1.0 - theta)*mu0 + '
                                                    'theta*mu\n'
                                                    'with theta = 0.5 and a SMALL time '
                                                    'step. For lmbda = 1e-2 and '
                                                    'mobility M = 1 on a unit square, '
                                                    'dt = 5e-6 was executed '
                                                    'successfully; dt = 1e-4 and dt = '
                                                    '1e-3 both fail in the first steps '
                                                    '(see pitfalls).',
                                        'OPTIONAL': 'theta = 1.0 gives backward Euler '
                                                    '(first order, more dissipative, '
                                                    'usually more robust for the first '
                                                    'few steps). dt may be ramped '
                                                    'upward once the interfaces have '
                                                    'formed and the dynamics has '
                                                    'slowed, by assigning to dt.value. '
                                                    'There is no adaptive time '
                                                    'stepping in '
                                                    'dolfinx.fem.petsc.NonlinearProblem '
                                                    '-- any dt control must be written '
                                                    'by hand in the Python time loop.',
                                        'explanation': 'The stiffness comes from the '
                                                       'fourth-order operator: the '
                                                       'spinodal instability amplifies '
                                                       'short-wavelength modes very '
                                                       'fast, so the admissible step '
                                                       'is tiny while the initial '
                                                       'perturbation is still being '
                                                       'amplified, and can be relaxed '
                                                       'once the interfaces have '
                                                       'formed and coarsening has '
                                                       'taken over.',
                                        'pitfalls': ['dolfinx does NOT do adaptive '
                                                     'time stepping here. Signal: the '
                                                     "string 'step rejected, reducing "
                                                     "dt' is emitted by nothing in "
                                                     'this stack -- it does not occur '
                                                     'in the PETSc library at all, and '
                                                     'DIVERGED_STEP_REJECTED is a '
                                                     "member of PETSc's TS (time "
                                                     'stepper) reasons, which '
                                                     "dolfinx's NonlinearProblem never "
                                                     'uses; problem.solver is a SNES '
                                                     'and PETSc.SNES.ConvergedReason '
                                                     'has no such member.']},
                   'materials': {'lmbda': {'range': [0.0001, 0.1],
                                           'unit': 'energy*length^2',
                                           'description': 'Gradient-energy '
                                                          'coefficient; interface '
                                                          'thickness scales like '
                                                          'sqrt(lmbda), so the mesh '
                                                          'must resolve that length'},
                                 'M': {'range': [0.001, 100.0],
                                       'unit': 'mobility',
                                       'description': 'Mobility in dc/dt = '
                                                      'div(M*grad(mu)); sets the '
                                                      'diffusive time scale. May be '
                                                      'constant or degenerate M = '
                                                      'c*(1-c)'},
                                 'well_depth': {'range': [1.0, 1000.0],
                                                'unit': 'energy density',
                                                'description': 'Prefactor of the '
                                                               'double well f = '
                                                               'A*c^2*(1-c)^2; A = 100 '
                                                               'is the standard demo '
                                                               'value'},
                                 'c': {'range': [0.0, 1.0],
                                       'unit': 'dimensionless',
                                       'description': 'Concentration/phase field; the '
                                                      'two stable phases of '
                                                      'A*c^2*(1-c)^2 sit at 0 and 1, '
                                                      'and the discrete solution '
                                                      'overshoots slightly past both'}},
                   'pitfalls': ['[Numerical] The system is stiff and the time step must be '
                                'small, but the failure at a wrong dt is NOT what is usually '
                                'claimed, and it depends on the mesh. Executed with the example '
                                'above (lmbda = 1e-2, M = 1, theta = 0.5, a 0.63 +/- 0.01 '
                                "random initial concentration from the example's seed), the "
                                "string 'DIVERGED_FNORM_NAN' never appeared at any dt. TOO "
                                'LARGE (dt >= 1e-2): on 64x64 and 96x96 meshes Newton converges '
                                'easily, 1-3 iterations per step, and there is no error of any '
                                'kind, but the scheme DAMPS the perturbation instead of '
                                'amplifying it -- the spread max(c) - min(c) falls from 2e-2 to '
                                'about 2e-6 in the first step at dt = 1, to about 2e-5 at dt = '
                                '0.1 and to about 2e-4 at dt = 1e-2, and stays below 1e-3 over '
                                'five steps. No phase separation occurs: a completely silent '
                                "physical failure. On the example's own 48x48 mesh the same dt "
                                'values behave differently: dt = 1 stops with DIVERGED_MAX_IT '
                                '(getConvergedReason() == -5) in the first step, dt = 0.1 with '
                                'DIVERGED_LINE_SEARCH (-6) at step 4, and at dt = 1e-2 the '
                                'spread first drops to 5e-4 and then regrows to 4e-2 by step 5. '
                                'INTERMEDIATE (dt = 1e-4 and 1e-3): Newton fails within the '
                                'first four steps on all three meshes, with '
                                'getConvergedReason() == -6 (DIVERGED_LINE_SEARCH) or -5 '
                                '(DIVERGED_MAX_IT) and the residual norm stalled between about '
                                "1e-5 and 1e-1; PETSc prints 'Nonlinear <prefix> solve did not "
                                "converge due to DIVERGED_LINE_SEARCH' (or '... due to "
                                "DIVERGED_MAX_IT') only when snes_converged_reason is set. dt = "
                                '1e-5 and dt = 5e-6 both converge and produce phase separation. '
                                'Signal: watch BOTH getConvergedReason() and the spread of c -- '
                                'a converged step whose min(c) and max(c) stay equal to the '
                                'initial mean means the step is too big, not that the model is '
                                'fine. (Verified by execution on dolfinx 0.10.0 with PETSc '
                                '3.24.5 at dt = 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5 on 48x48, 64x64 '
                                'and 96x96 meshes, and at dt = 5e-6 on 48x48.)',
                                '[Numerical] The initial condition must contain a '
                                "random perturbation. The commonly quoted rule that 'c "
                                '= 0.5 exactly gives no phase separation because it is '
                                "the unstable symmetric mean' gets the right answer "
                                'for the wrong reason, and the wrong reason will '
                                'mislead you: ANY uniform initial concentration gives '
                                'no phase separation, and 0.5 is in no way special. '
                                'Signal: starting from a uniform field '
                                '(interpolate(lambda x: np.full(x.shape[1], 0.5))) the '
                                'concentration stays at exactly that value for the '
                                'whole run with standard deviation 0.0 -- but a '
                                'uniform 0.63, the value usually recommended as the '
                                "'safe' one, is equally frozen (standard deviation "
                                'stuck at round-off, order 1e-17). Conversely 0.5 PLUS '
                                'noise separates perfectly well, and in fact faster '
                                'than 0.63 plus the same noise: over the same number '
                                'of steps the concentration spreads from about [0.49, '
                                '0.51] out to roughly [-0.08, 1.08]. The perturbation, '
                                'not the mean value, is what breaks the symmetry. Use '
                                'u.sub(0).interpolate(lambda x: 0.63 + 0.02*(0.5 - '
                                'rng.random(x.shape[1]))) with rng = '
                                'np.random.default_rng(seed). (Verified by execution '
                                'on dolfinx 0.10.0: uniform 0.5, uniform 0.63, 0.5 + '
                                'noise and 0.63 + noise all run side by side.)',
                                '[API] Wrap the concentration in ufl.variable() before '
                                'differentiating the double well with ufl.diff -- cv = '
                                'ufl.variable(c); f = 100*cv**2*(1-cv)**2; dfdc = '
                                'ufl.diff(f, cv). ufl.diff returns exactly the '
                                'analytic derivative: compared against a hand-coded '
                                '200*c*(1-c)*(1-2*c), the L2 norm of the difference is '
                                'at round-off level. The danger of hand-coding is NOT '
                                'a loss of Newton convergence -- dolfinx builds the '
                                'Jacobian from whatever residual you wrote, so the '
                                'solver is perfectly happy -- it is that you silently '
                                'solve a different physical problem, and the solver '
                                'output gives you no hint at all. Signal: the '
                                'frequently repeated expression 12*c*(c-1)*(2*c-1) is '
                                'NOT the derivative of 100*c^2*(1-c)^2; it is the '
                                'derivative of 6*c^2*(1-c)^2, i.e. exactly 0.06 times '
                                'too small (at c = 0.3 it gives 1.008 where the '
                                'correct value is 16.8). Running the identical time '
                                'loop with it substituted, Newton converges in exactly '
                                '2 iterations on every single step with zero failures '
                                '-- BETTER-looking than the correct run, which needs '
                                'about 16 -- while the concentration barely moves: '
                                'over the same 60 steps the correct residual spreads c '
                                'across roughly [-0.09, 1.06] and the corrupted one '
                                'leaves it inside [0.626, 0.634]. Low, flat Newton '
                                'counts on a Cahn-Hilliard run in the separation phase '
                                'are a warning sign, not a good sign. (Verified by '
                                'execution on dolfinx 0.10.0.)',
                                '[API] dolfinx.fem.petsc.NonlinearProblem performs NO '
                                'adaptive time stepping, so no dt-control message can '
                                'ever appear; if a step fails you must catch it and '
                                'shrink dt yourself in the Python loop. Signal: the '
                                "string 'step rejected, reducing dt' is emitted by "
                                'nothing in this stack. It does not occur anywhere in '
                                'the installed PETSc library (a case-sensitive search '
                                "of the shared object finds neither 'step rejected' "
                                "nor 'Step rejected' nor 'reducing dt'). The only "
                                'related PETSc symbol is DIVERGED_STEP_REJECTED, which '
                                "is a member of PETSc.TS.ConvergedReason -- PETSc's "
                                "time-stepper object, which dolfinx's NonlinearProblem "
                                'never creates. problem.solver is a SNES, and '
                                'PETSc.SNES.ConvergedReason has no '
                                'DIVERGED_STEP_REJECTED member at all. What a failing '
                                'step really reports is DIVERGED_LINE_SEARCH (-6) or '
                                'DIVERGED_MAX_IT (-5). (Verified by execution and by '
                                'searching the installed PETSc 3.24 library on dolfinx '
                                '0.10.0.)',
                                '[API] Build the residual from ufl.split(u) and '
                                'ufl.split(u0), NEVER from u.sub(0)/u.sub(1). The '
                                'u.sub(i) spelling is not caught by UFL and not caught '
                                'by the form compiler -- ufl.form() succeeds, '
                                'NonlinearProblem is constructed, and the failure only '
                                'shows up when you solve. Signal: on the identical '
                                'Cahn-Hilliard problem the ufl.split version converges '
                                'every step (mass drift ~1e-15, concentration '
                                'spreading out to roughly [-0.11, 1.07]) while the '
                                'u.sub version fails on the FIRST step, printing '
                                '"Linear <prefix> solve did not converge due to '
                                'DIVERGED_PC_FAILED iterations 0", "PC failed due to '
                                'FACTOR_OTHER" and "Nonlinear <prefix> solve did not '
                                'converge due to DIVERGED_LINEAR_SOLVE iterations 0" '
                                '-- SNES getConvergedReason() == -3, KSP '
                                'getConvergedReason() == -11 -- and leaves the '
                                'concentration array untouched at its initial values; '
                                'comparing the two fields afterwards gives a maximum '
                                'difference of about 0.73, i.e. the u.sub run simply '
                                'never moved. (Verified by execution on dolfinx '
                                '0.10.0.)',
                                '[Physics] Verify a Cahn-Hilliard run with two checks '
                                'that need no reference solution: total mass must be '
                                'conserved exactly, and the total free energy must '
                                'decrease monotonically. Assemble mass = '
                                'fem.assemble_scalar(fem.form(c*ufl.dx)) and energy = '
                                'fem.assemble_scalar(fem.form((100*c**2*(1-c)**2 + '
                                '0.5*lmbda*ufl.inner(ufl.grad(c), '
                                'ufl.grad(c)))*ufl.dx)) each step. Signal: with no '
                                'Dirichlet conditions the mass drift over a run is at '
                                'round-off level (order 1e-16 to 1e-15) and the free '
                                'energy is strictly non-increasing. Adding a Dirichlet '
                                'condition on c over the whole boundary of the same '
                                'problem raises the drift to about 2e-2 -- thirteen '
                                'orders of magnitude -- so a visible drift means a '
                                'boundary condition is leaking mass, not that the '
                                'solver is inaccurate. (Verified by execution on '
                                'dolfinx 0.10.0, with and without the Dirichlet BC.)',
                                '[API] Refresh the previous-step Function INSIDE the '
                                'time loop -- u0.x.array[:] = u.x.array must run '
                                'before every problem.solve(), and there is nothing in '
                                'dolfinx that will do it for you or warn you. Signal: '
                                'if u0 is left at the initial state, the first step '
                                'advances normally and then every later step re-solves '
                                'the identical problem, whose solution is already in '
                                'hand. SNES therefore reports 0 iterations from the '
                                'second step onward (mean iteration count over 25 '
                                'steps drops to 0.1, against about 17 for a correct '
                                'run), mass conservation still looks perfect, and the '
                                'concentration simply stops evolving: its spread stays '
                                'at the initial noise level while a correct run has '
                                'separated into the two phases. (Verified by execution '
                                'on dolfinx 0.10.0.)',
                                '[Numerical] The concentration legitimately overshoots '
                                'the two wells: the converged solution takes values '
                                'slightly below 0 and slightly above 1. Do not treat '
                                'that as a bug or clamp it. Signal: after phase '
                                'separation the min and max of the concentration array '
                                'sit a few percent outside [0, 1]; a solution pinned '
                                'exactly inside [0, 1] usually means the field has not '
                                'separated at all. (Verified by execution on dolfinx '
                                '0.10.0.)']},
 'eigenvalue': {'description': 'Eigenvalue problems A*x = lambda*B*x using SLEPc. '
                               'Vibration modes, buckling, electromagnetic modes.',
                'minimal_working_example': '# COMPLETE runnable script - Dirichlet '
                                           'Laplacian eigenmodes.\n'
                                           '# Executed on dolfinx 0.10.0 / slepc4py '
                                           '3.24.3.\n'
                                           'from mpi4py import MPI\n'
                                           'from dolfinx import mesh, fem, '
                                           'default_scalar_type\n'
                                           'from dolfinx.fem.petsc import '
                                           'assemble_matrix\n'
                                           'from slepc4py import SLEPc\n'
                                           'import ufl\n'
                                           'import numpy as np\n'
                                           '\n'
                                           'N_EIGS = 5\n'
                                           'domain = '
                                           'mesh.create_unit_square(MPI.COMM_WORLD, '
                                           '32, 32,\n'
                                           '                                 '
                                           'mesh.CellType.triangle)\n'
                                           'fdim = domain.topology.dim - 1\n'
                                           'domain.topology.create_connectivity(fdim, '
                                           'domain.topology.dim)\n'
                                           "V = fem.functionspace(domain, ('Lagrange', "
                                           '1))\n'
                                           '\n'
                                           'def boundary(x):\n'
                                           '    return (np.isclose(x[0], 0) | '
                                           'np.isclose(x[0], 1)\n'
                                           '            | np.isclose(x[1], 0) | '
                                           'np.isclose(x[1], 1))\n'
                                           'facets = '
                                           'mesh.locate_entities_boundary(domain, '
                                           'fdim, boundary)\n'
                                           'bc = '
                                           'fem.dirichletbc(default_scalar_type(0),\n'
                                           '                     '
                                           'fem.locate_dofs_topological(V, fdim, '
                                           'facets), V)\n'
                                           '\n'
                                           'u = ufl.TrialFunction(V)\n'
                                           'v = ufl.TestFunction(V)\n'
                                           'a = fem.form(ufl.dot(ufl.grad(u), '
                                           'ufl.grad(v)) * ufl.dx)\n'
                                           'm = fem.form(u * v * ufl.dx)\n'
                                           'A = assemble_matrix(a, '
                                           'bcs=[bc])            # diag defaults to 1\n'
                                           'A.assemble()\n'
                                           'M = assemble_matrix(m, bcs=[bc], '
                                           'diag=0.0)  # REQUIRED: 0.0 here\n'
                                           'M.assemble()\n'
                                           '\n'
                                           'eps = SLEPc.EPS().create(MPI.COMM_WORLD)\n'
                                           'eps.setOperators(A, M)\n'
                                           'eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)\n'
                                           'eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)\n'
                                           'eps.setDimensions(N_EIGS, max(4 * N_EIGS, '
                                           '20))\n'
                                           'eps.setTolerances(tol=1e-9, max_it=2000)\n'
                                           'eps.getST().setType(SLEPc.ST.Type.SINVERT)  '
                                           '# REQUIRED with diag=0.0\n'
                                           'eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)\n'
                                           'eps.setTarget(0.0)\n'
                                           'eps.solve()\n'
                                           '\n'
                                           'n_conv = eps.getConverged()\n'
                                           'if n_conv < N_EIGS:\n'
                                           "    raise RuntimeError(f'converged "
                                           "{n_conv}, reason '\n"
                                           '                       '
                                           "f'{eps.getConvergedReason()}')\n"
                                           'xr, xi = A.createVecs()\n'
                                           'Ax, Mx = A.createVecLeft(), '
                                           'A.createVecLeft()\n'
                                           'eigenvalues, residuals = [], []\n'
                                           'for i in range(N_EIGS):\n'
                                           '    lam = eps.getEigenvalue(i).real\n'
                                           '    eps.getEigenvector(i, xr, xi)\n'
                                           '    A.mult(xr, Ax)\n'
                                           '    M.mult(xr, Mx)\n'
                                           '    Ax.axpy(-lam, Mx)\n'
                                           '    eigenvalues.append(lam)\n'
                                           '    residuals.append(Ax.norm() / '
                                           'max(xr.norm(), 1e-300))\n'
                                           'if max(residuals) > 1e-6 or any(l <= 0 for '
                                           'l in eigenvalues):\n'
                                           "    raise RuntimeError(f'bad eigenpairs "
                                           "{eigenvalues} {residuals}')\n"
                                           "print('eigenvalues:', eigenvalues)\n"
                                           "print('max eigenpair residual:', "
                                           'max(residuals))\n',
                'function_space': {'REQUIRED': 'V = fem.functionspace(domain, '
                                               "('Lagrange', degree))",
                                   'OPTIONAL': "('Lagrange', degree) for scalar "
                                               'problems (degree 1/2/3 verified on '
                                               'triangles, degree 1 on '
                                               "quadrilaterals); ('N1curl', degree) "
                                               'for electromagnetic modes; '
                                               "('Lagrange', degree, (gdim,)) for "
                                               'structural modes.',
                                   'explanation': 'Both matrices must be assembled on '
                                                  'the SAME space.'},
                'weak_form': {'REQUIRED': 'a = fem.form(ufl.dot(ufl.grad(u), '
                                          'ufl.grad(v)) * ufl.dx)  # -> A\n'
                                          'm = fem.form(u * v * '
                                          'ufl.dx)                              # -> M',
                              'OPTIONAL': 'Weight with material coefficients: k * '
                                          'grad-grad for conductivity/stiffness, rho * '
                                          'u * v for density.',
                              'explanation': 'TWO separate forms are REQUIRED. '
                                             'eps.setOperators(A) alone solves the '
                                             'standard problem A x = lambda x, not the '
                                             'physical generalised one.'},
                'boundary_conditions': {'REQUIRED': 'A = assemble_matrix(a, '
                                                    'bcs=[bc])            # diag stays '
                                                    '1\n'
                                                    'M = assemble_matrix(m, bcs=[bc], '
                                                    'diag=0.0)  # diag MUST be 0.0',
                                        'OPTIONAL': 'Pure-Neumann problems: assemble '
                                                    'both with bcs=[] and expect a '
                                                    'zero eigenvalue with a constant '
                                                    'eigenvector.',
                                        'explanation': 'The constrained rows must not '
                                                       'carry the same diagonal in '
                                                       'both matrices. diag=0.0 on M '
                                                       'sends the constraint modes to '
                                                       'infinity; anything else leaves '
                                                       'them in the finite spectrum at '
                                                       'lambda = A_ii / M_ii.'},
                'solver': {'REQUIRED': 'eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)\n'
                                       'eps.getST().setType(SLEPc.ST.Type.SINVERT)\n'
                                       'eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)\n'
                                       'eps.setTarget(0.0)',
                           'OPTIONAL': 'eps.setTarget(sigma) anywhere in the spectrum '
                                       'for interior modes; eps.setDimensions(nev, '
                                       'ncv) with ncv >= 2*nev.',
                           'explanation': 'SINVERT is REQUIRED whenever M is assembled '
                                          'with diag=0.0, because the default '
                                          'transform factorises M directly.'},
                'verification': 'Verify each returned pair by its own residual, which '
                                'needs no reference solution: A.mult(x, Ax); M.mult(x, '
                                'Mx); Ax.axpy(-lam, Mx); require Ax.norm() / x.norm() '
                                'at the solver tolerance, and require lambda > 0 for a '
                                'Dirichlet Laplacian. The known failure here is a run '
                                'that exits 0 and reports spurious constraint modes as '
                                'physical eigenvalues.',
                'demo_url': 'https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_half_loaded_waveguide.html',
                'solver_types': {'krylovschur': 'Default, recommended for most '
                                                'problems',
                                 'arnoldi': 'Standard Arnoldi iteration',
                                 'lanczos': 'For symmetric (Hermitian) problems',
                                 'power': 'Power iteration (only for dominant '
                                          'eigenvalue)',
                                 'jd': 'Jacobi-Davidson (interior eigenvalues)'},
                'pitfalls': ['[Integration] Eigenvalue problems in dolfinx use SLEPc, '
                             'whose Python binding is slepc4py.SLEPc.EPS; slepc4py is '
                             'packaged SEPARATELY from petsc4py and can be missing. '
                             'Signal: `from slepc4py import SLEPc` resolves on a '
                             'complete install (reference install: slepc4py 3.24.3, '
                             "SLEPc.EPS is <class 'slepc4py.SLEPc.EPS'>). When the "
                             'package is absent the exact error on Python 3.12 is '
                             "ModuleNotFoundError: No module named 'slepc4py' — note "
                             'ModuleNotFoundError, a SUBCLASS of ImportError, and note '
                             'the quotes around the module name; catching ImportError '
                             'still works.',
                             '[Numerical] Shift-and-invert spectral transformation '
                             '(SINVERT) is essential for interior eigenvalues. '
                             'SLEPc.EPS().setST(...) with a SLEPc.ST configured to '
                             'SINVERT centers the spectrum on the target value. '
                             'Signal: searching for eigenvalues near k^2_estimate on '
                             'the dolfinx-assembled stiffness Matrix without SINVERT '
                             'returns extreme eigenvalues (highest or lowest) instead; '
                             'with SINVERT and target = k^2_estimate the returned '
                             'eigenvalues cluster near the target. (Claim inherited — '
                             'not yet empirically separated.)',
                             '[API] eps.setDimensions(nev, ncv) requests nev '
                             'eigenvalues with an ncv-dimensional search space '
                             "(SLEPc's own heuristic is ncv >= 2*nev). Signal: "
                             '[MEASURED on slepc4py 3.24.3] check the COUNT, not the '
                             'reason code — a SUCCESSFUL solve returns '
                             'eps.getConvergedReason() == 1 '
                             "(SLEPc.EPS.ConvergedReason.CONVERGED_TOL), so 'error "
                             "code != 0' is exactly backwards; failure codes are "
                             'NEGATIVE (-1 = DIVERGED_ITS, -2 = DIVERGED_BREAKDOWN, -3 '
                             '= DIVERGED_SYMMETRY_LOST). The real test is '
                             'eps.getConverged() >= nev; on a 16x16 P1 Dirichlet '
                             'Laplacian with shift-and-invert both (nev=4, ncv=8) and '
                             '(nev=4, ncv=5) returned nconv=4 with reason 1, so a '
                             'tight ncv did not fail here — if nconv falls short, '
                             'raise ncv and the iteration limit in '
                             'eps.setTolerances(tol, max_it).',
                             '[Numerical] For a generalised eigenvalue problem A*x = '
                             'lambda*B*x with Dirichlet BC, the Dirichlet rows of A '
                             'and of the mass matrix B must not both be given the SAME '
                             'diagonal value, or every constrained DOF contributes a '
                             'spurious eigenvalue at A_ii/B_ii. Signal: assembling A '
                             'with bcs (default diag=1) and B ALSO with bcs (default '
                             'diag=1) makes the lowest reported eigenvalues come back '
                             'as exactly 1.0 to within round-off (1.0000000000003, '
                             '1.0000000000031, ...), one per constrained DoF, with the '
                             'physical spectrum following them; the spurious modes sit '
                             'at lambda = 1, NOT at lambda = 0 as an earlier version '
                             'of this entry claimed. THE ONLY CLEAN RECIPE VERIFIED '
                             'HERE is: assemble A with bcs=[bc] (diag left at 1) and B '
                             'with bcs=[bc], diag=0.0, AND set the spectral transform '
                             'to shift-and-invert, '
                             'eps.getST().setType(SLEPc.ST.Type.SINVERT). That '
                             'reproduces the exactly-reduced interior-DoF eigenproblem '
                             'to machine precision and returns exactly as many finite '
                             'eigenvalues as there are free DoFs. Verified at Lagrange '
                             'degree 1/2/3 on triangles and degree 1 on '
                             'quadrilaterals. NOTE the kwarg is `diag`, not '
                             "`diagonal`: dolfinx 0.10's assemble_matrix signature is "
                             '(a, bcs=None, diag=1, constants=None, coeffs=None, '
                             'kind=None), and `diagonal=0.0` raises TypeError '
                             "'assemble_matrix() got an unexpected keyword argument'. "
                             '(Verified empirically 2026-08-03.)',
                             '[Numerical] TWO CORRECTIONS to the recipe above, both '
                             'falsified by execution on 2026-08-03 and both previously '
                             'asserted in this catalog. (1) `diag=0.0` on the mass '
                             'matrix does NOT work with the default spectral '
                             'transform: the default EPS transform is a plain shift, '
                             'which factorises B itself, and the zeroed constrained '
                             "rows give an exact zero pivot. Signal: '[0] Zero pivot "
                             'in LU factorization: '
                             "https://petsc.org/release/faq/#zeropivot' then '[0] Zero "
                             "pivot row 0 value 0. tolerance 2.22045e-14', through "
                             'STSetUp_Shift -> KSPSetUp -> PCSetUp_LU -> '
                             'MatLUFactorNumeric_SeqAIJ, reaching Python as '
                             "'SystemError: <cyfunction EPS.solve at 0x...> returned a "
                             "result with an exception set'. Adding SINVERT makes the "
                             'identical run succeed. (2) Assembling B with `bcs=[]` is '
                             'NOT a clean recipe: with a CONSISTENT mass matrix it '
                             'solves a DIFFERENT pencil, because the '
                             'boundary-to-interior blocks of B are not zero. Signal: '
                             'on a coarse mesh its eigenvalues differ from the '
                             'exactly-reduced interior-DoF problem in the second '
                             'significant digit; the gap shrinks under refinement (on '
                             'a fine mesh the two agree to several digits, which is '
                             'exactly why the error is easy to miss), whereas B with '
                             'diag=0.0 matches the reduced problem to machine '
                             'precision at EVERY mesh — checked against a dense '
                             'scipy.linalg.eigvals of the interior block. SCOPE, from '
                             'an adversarial re-check: if B is mass-LUMPED the '
                             'boundary-interior blocks vanish and bcs=[] does '
                             'reproduce the constrained spectrum exactly — but then '
                             'the constrained modes reappear at lambda = 1/B_ii and '
                             'INTERLEAVE with the physical ones, so the smallest-real '
                             'request returns a mixture. The diag=0.0 + SINVERT recipe '
                             'is the one that is correct in both cases. (Verified '
                             'empirically 2026-08-03.)',
                             '[Integration] Complex-valued eigenvalues require dolfinx '
                             '+ PETSc + SLEPc all compiled with '
                             '--with-scalar-type=complex. The default conda-forge '
                             'fenics-dolfinx build is REAL: '
                             'dolfinx.default_scalar_type is numpy.float64 (verified '
                             'empirically 2026-06-01). For complex Helmholtz / Maxwell '
                             'eigenproblems either rebuild with complex scalar OR '
                             'split into (re, im) real-pair formulation. Signal: '
                             'dolfinx.default_scalar_type returns numpy.float64 in a '
                             'real build; '
                             'numpy.issubdtype(dolfinx.default_scalar_type, '
                             'np.complexfloating) is False. A form with a complex '
                             'literal (e.g. 1j*v*ufl.dx) is then rejected by fem.form '
                             'with ValueError: Unexpected complex value in real '
                             'expression., and a complex fem.Constant or complex-dtype '
                             'Function in a float64 form raises TypeError: __init__(): '
                             'incompatible function arguments. Only '
                             'Function.interpolate of a complex Python callable drops '
                             'the imaginary part, with ComplexWarning: Casting complex '
                             'values to real discards the imaginary part. (Verified by '
                             'execution on the real dolfinx 0.10.0 build.)']},
 'reaction_diffusion': {'description': 'Systems of coupled reaction-diffusion '
                                       'equations in dolfinx 0.10: several chemical '
                                       'species that diffuse and react with each '
                                       'other. For species i: d(c_i)/dt = '
                                       'div(D_i*grad(c_i)) + R_i(c_1,...,c_n), with '
                                       'R_i nonlinear. All species live in ONE mixed '
                                       'function space so that the reaction coupling '
                                       'is solved implicitly and monolithically by a '
                                       'single Newton (PETSc SNES) solve per time '
                                       'step.',
                        'minimal_working_example': '\n'
                                                   '# Two coupled species on a mixed '
                                                   'element:  2A <-> B\n'
                                                   '#   dA/dt = D_A*lap(A) - '
                                                   '2*(k_f*A^2 - k_r*B)\n'
                                                   '#   dB/dt = D_B*lap(B) +   '
                                                   '(k_f*A^2 - k_r*B)\n'
                                                   '# All boundaries no-flux, so int(A '
                                                   '+ 2B) dx is conserved exactly -- '
                                                   'that is\n'
                                                   '# the reference-free self-check '
                                                   'printed at the end.\n'
                                                   'from mpi4py import MPI\n'
                                                   'import numpy as np\n'
                                                   'import basix.ufl\n'
                                                   'import ufl\n'
                                                   'from dolfinx import fem, mesh\n'
                                                   'from dolfinx.fem.petsc import '
                                                   'NonlinearProblem\n'
                                                   '\n'
                                                   'msh = '
                                                   'mesh.create_unit_square(MPI.COMM_WORLD, '
                                                   '32, 32, mesh.CellType.triangle)\n'
                                                   'P1 = basix.ufl.element("Lagrange", '
                                                   'msh.basix_cell(), 1)\n'
                                                   'W = fem.functionspace(msh, '
                                                   'basix.ufl.mixed_element([P1, '
                                                   'P1]))\n'
                                                   '\n'
                                                   'w = fem.Function(W)\n'
                                                   'w_n = '
                                                   'fem.Function(w.function_space)\n'
                                                   'a, b = ufl.split(w)\n'
                                                   'a_n, b_n = ufl.split(w_n)\n'
                                                   'va, vb = ufl.TestFunctions(W)\n'
                                                   '\n'
                                                   'D_a, D_b = fem.Constant(msh, '
                                                   '0.01), fem.Constant(msh, 0.005)\n'
                                                   'k_f, k_r = fem.Constant(msh, 5.0), '
                                                   'fem.Constant(msh, 1.0)\n'
                                                   'dt_val = 0.02\n'
                                                   'dt = fem.Constant(msh, dt_val)\n'
                                                   '\n'
                                                   'w_n.sub(0).interpolate(lambda x: '
                                                   '1.0 + 0.5 * np.cos(np.pi * x[0]) * '
                                                   'np.cos(np.pi * x[1]))\n'
                                                   'w_n.sub(1).interpolate(lambda x: '
                                                   '0.2 + 0.0 * x[0])\n'
                                                   'w_n.x.scatter_forward()\n'
                                                   'w.x.array[:] = w_n.x.array\n'
                                                   '\n'
                                                   'rate = k_f * a * a - k_r * b\n'
                                                   'F = ((a - a_n) / dt * va * ufl.dx\n'
                                                   '     + D_a * '
                                                   'ufl.inner(ufl.grad(a), '
                                                   'ufl.grad(va)) * ufl.dx\n'
                                                   '     + 2.0 * rate * va * ufl.dx\n'
                                                   '     + (b - b_n) / dt * vb * '
                                                   'ufl.dx\n'
                                                   '     + D_b * '
                                                   'ufl.inner(ufl.grad(b), '
                                                   'ufl.grad(vb)) * ufl.dx\n'
                                                   '     - rate * vb * ufl.dx)\n'
                                                   '\n'
                                                   'problem = NonlinearProblem(\n'
                                                   '    F, w, '
                                                   'petsc_options_prefix="rd_",\n'
                                                   '    petsc_options={"snes_type": '
                                                   '"newtonls", "snes_rtol": 1e-10, '
                                                   '"snes_atol": 1e-12,\n'
                                                   '                   "snes_max_it": '
                                                   '30, "ksp_type": "preonly", '
                                                   '"pc_type": "lu"})\n'
                                                   '\n'
                                                   'mass_f = fem.form((a + 2.0 * b) * '
                                                   'ufl.dx)\n'
                                                   'mass_n_f = fem.form((a_n + 2.0 * '
                                                   'b_n) * ufl.dx)\n'
                                                   'M0 = '
                                                   'msh.comm.allreduce(fem.assemble_scalar(mass_n_f), '
                                                   'op=MPI.SUM)\n'
                                                   '\n'
                                                   'its = []\n'
                                                   'for step in range(25):\n'
                                                   '    problem.solve()\n'
                                                   '    reason = '
                                                   'problem.solver.getConvergedReason()\n'
                                                   '    if reason <= 0:\n'
                                                   '        raise RuntimeError(f"SNES '
                                                   'failed at step {step} with reason '
                                                   '{reason}")\n'
                                                   '    '
                                                   'its.append(problem.solver.getIterationNumber())\n'
                                                   '    w_n.x.array[:] = w.x.array\n'
                                                   '\n'
                                                   'M1 = '
                                                   'msh.comm.allreduce(fem.assemble_scalar(mass_f), '
                                                   'op=MPI.SUM)\n'
                                                   'a_h = w.sub(0).collapse()\n'
                                                   'b_h = w.sub(1).collapse()\n'
                                                   'a_min = '
                                                   'msh.comm.allreduce(float(a_h.x.array.min()), '
                                                   'op=MPI.MIN)\n'
                                                   'b_min = '
                                                   'msh.comm.allreduce(float(b_h.x.array.min()), '
                                                   'op=MPI.MIN)\n'
                                                   'a_max = '
                                                   'msh.comm.allreduce(float(a_h.x.array.max()), '
                                                   'op=MPI.MAX)\n'
                                                   'b_max = '
                                                   'msh.comm.allreduce(float(b_h.x.array.max()), '
                                                   'op=MPI.MAX)\n'
                                                   '\n'
                                                   'print(f"SNES converged reason = '
                                                   '{problem.solver.getConvergedReason()} '
                                                   '(>0 means converged)")\n'
                                                   'print(f"Newton iterations per '
                                                   'step: min={min(its)} '
                                                   'max={max(its)} last={its[-1]}")\n'
                                                   'print(f"t_end = {25 * '
                                                   'dt_val:.3f}")\n'
                                                   'print(f"species A range = '
                                                   '[{a_min:.8f}, {a_max:.8f}]")\n'
                                                   'print(f"species B range = '
                                                   '[{b_min:.8f}, {b_max:.8f}]")\n'
                                                   'print(f"both species non-negative '
                                                   '= {bool(a_min >= 0.0 and b_min >= '
                                                   '0.0)}")\n'
                                                   'print(f"conserved quantity int(A + '
                                                   '2B) dx : start {M0:.12f}  end '
                                                   '{M1:.12f}")\n'
                                                   'print(f"CHECK relative drift = '
                                                   '{abs(M1 - M0) / abs(M0):.3e}   '
                                                   '(must be <= ~1e-10)")\n',
                        'function_space': {'REQUIRED': 'import basix.ufl\n'
                                                       'P1 = '
                                                       'basix.ufl.element("Lagrange", '
                                                       'msh.basix_cell(), 1)\n'
                                                       'W = fem.functionspace(msh, '
                                                       'basix.ufl.mixed_element([P1, '
                                                       'P1]))\n'
                                                       'w = fem.Function(W)            '
                                                       '# current step, ALL species\n'
                                                       'w_n = '
                                                       'fem.Function(w.function_space)   '
                                                       '# previous step',
                                           'OPTIONAL': 'The number of species is '
                                                       'optional: pass one element per '
                                                       'species, e.g. '
                                                       'basix.ufl.mixed_element([P1, '
                                                       'P1, P1]) for three. The '
                                                       'species need NOT share a '
                                                       'degree: '
                                                       'basix.ufl.mixed_element([P1, '
                                                       'P2]) with P2 = '
                                                       'basix.ufl.element("Lagrange", '
                                                       'msh.basix_cell(), 2) was '
                                                       'verified to build a valid '
                                                       '2-subspace mixed space. If '
                                                       'every species uses the SAME '
                                                       'element, the shorthand '
                                                       'fem.functionspace(msh, '
                                                       '("Lagrange", 1, (n_species,))) '
                                                       'gives an equivalent blocked '
                                                       'space with n_species subspaces '
                                                       'and the same ufl.split '
                                                       'behaviour. Cell type is '
                                                       'optional: triangle or '
                                                       'quadrilateral in 2-D, '
                                                       'tetrahedron or hexahedron in '
                                                       '3-D; every statement here was '
                                                       'checked on triangles and '
                                                       'quadrilaterals at degree 1 and '
                                                       '2.',
                                           'explanation': 'One mixed space means one '
                                                          'residual, one Jacobian and '
                                                          'one Newton solve per time '
                                                          'step, so the inter-species '
                                                          'coupling is treated fully '
                                                          'implicitly. Solving the '
                                                          'species one at a time in '
                                                          'separate spaces '
                                                          'reintroduces an operator '
                                                          'split whose error you then '
                                                          'have to control by '
                                                          'shrinking dt.',
                                           'pitfalls': ['basix.ufl.mixed_element takes '
                                                        'ONE list argument. Signal: '
                                                        'basix.ufl.mixed_element(P1, '
                                                        'P1) raises TypeError: '
                                                        'mixed_element() takes 1 '
                                                        'positional argument but 2 '
                                                        'were given.',
                                                        'There is no ufl.MixedElement '
                                                        'in this stack. Signal: '
                                                        "AttributeError: module 'ufl' "
                                                        'has no attribute '
                                                        "'MixedElement'."]},
                        'weak_form': {'REQUIRED': 'a, b = ufl.split(w)              # '
                                                  'trial-side species, from the SAME '
                                                  'w\n'
                                                  'a_n, b_n = ufl.split(w_n)        # '
                                                  'previous-step species\n'
                                                  'va, vb = ufl.TestFunctions(W)    # '
                                                  'one test function per species\n'
                                                  '\n'
                                                  'rate = k_f * a * a - k_r * b     # '
                                                  'the nonlinear reaction, written '
                                                  'once\n'
                                                  'F = ((a - a_n) / dt * va * ufl.dx\n'
                                                  '     + D_a * ufl.inner(ufl.grad(a), '
                                                  'ufl.grad(va)) * ufl.dx\n'
                                                  '     + 2.0 * rate * va * ufl.dx\n'
                                                  '     + (b - b_n) / dt * vb * '
                                                  'ufl.dx\n'
                                                  '     + D_b * ufl.inner(ufl.grad(b), '
                                                  'ufl.grad(vb)) * ufl.dx\n'
                                                  '     - rate * vb * ufl.dx)',
                                      'OPTIONAL': 'The reaction expression is '
                                                  'free-form UFL: products, powers, '
                                                  'ufl.exp for Arrhenius kinetics, '
                                                  'ufl.conditional for switches. The '
                                                  'time discretisation is optional; '
                                                  'backward Euler (everything '
                                                  'evaluated at the new step, as '
                                                  'written above) is the safe default. '
                                                  'A theta-method is written as '
                                                  'theta*op(a, b) + (1-theta)*op(a_n, '
                                                  'b_n) with op() containing the '
                                                  'diffusion and reaction terms; theta '
                                                  'must be at least 0.5 and theta = 1 '
                                                  'is strongly preferred for stiff '
                                                  'kinetics. Extra terms such as '
                                                  'advection (ufl.dot(vel, '
                                                  'ufl.grad(a))*va*dx) or a source (- '
                                                  's*va*dx) may simply be added to F.',
                                      'explanation': 'F is a single rank-one form over '
                                                     'the whole mixed space. Do NOT '
                                                     'build a bilinear/linear pair by '
                                                     'hand: dolfinx differentiates F '
                                                     'for you and the resulting '
                                                     'Jacobian contains every '
                                                     'cross-species partial '
                                                     'derivative, which is exactly '
                                                     'what makes Newton converge '
                                                     'quadratically. Note the '
                                                     'stoichiometric factors (here 2 '
                                                     'for A, 1 for B) carry opposite '
                                                     'signs on the two equations.',
                                      'pitfalls': ['Use ufl.split(w), never w.sub(i) '
                                                   'or w.split(), to get the species '
                                                   'out of the mixed Function for the '
                                                   'residual. Signal: the form still '
                                                   'compiles, but the solve dies with '
                                                   'Error: error code 73 ... [0] '
                                                   'Object is in wrong state / [0] '
                                                   'Matrix is missing diagonal entry '
                                                   '0, because the automatic Jacobian '
                                                   'is empty on those rows.',
                                                   'Do not hand-write the Jacobian. '
                                                   'Signal: a Jacobian missing one '
                                                   'cross-species partial makes SNES '
                                                   'take 8 iterations with a linear '
                                                   'residual ratio of about 0.13 '
                                                   'instead of 3 iterations with '
                                                   'residuals 2.37e-03, 6.23e-07, '
                                                   '4.33e-14; on a single-species '
                                                   'quadratic reaction it hits '
                                                   'snes_max_it and returns reason -5 '
                                                   '(DIVERGED_MAX_IT) with the '
                                                   'residual stuck at 8.0e-06.']},
                        'boundary_conditions': {'REQUIRED': '# No-flux (zero Neumann) '
                                                            'on every wall is the '
                                                            'DEFAULT for every\n'
                                                            '# species: write nothing '
                                                            'at all. That is what the '
                                                            'example uses.\n'
                                                            '\n'
                                                            '# To clamp ONE species on '
                                                            'part of the boundary:\n'
                                                            'fdim = msh.topology.dim - '
                                                            '1\n'
                                                            'facets = '
                                                            'mesh.locate_entities_boundary(msh, '
                                                            'fdim, lambda x: '
                                                            'np.isclose(x[0], 0.0))\n'
                                                            'V0, _ = '
                                                            'W.sub(0).collapse()                    '
                                                            '# collapsed sub-space\n'
                                                            'dofs = '
                                                            'fem.locate_dofs_topological((W.sub(0), '
                                                            'V0), fdim, facets)\n'
                                                            'g = fem.Function(V0)\n'
                                                            'g.x.array[:] = 1.5\n'
                                                            'bc = fem.dirichletbc(g, '
                                                            'dofs, W.sub(0))\n'
                                                            '# then pass bcs=[bc] to '
                                                            'NonlinearProblem',
                                                'OPTIONAL': 'A prescribed influx of a '
                                                            'species is a ds() term '
                                                            'added to F: - j_in * va * '
                                                            'ds(marker), with ds = '
                                                            'ufl.Measure("ds", '
                                                            'domain=msh, '
                                                            'subdomain_data=facet_tags). '
                                                            'A surface reaction is the '
                                                            'same thing with a '
                                                            'nonlinear integrand, e.g. '
                                                            '+ k_s * a * va * '
                                                            'ds(marker). Different '
                                                            'species may have '
                                                            'different boundary '
                                                            'conditions; build one '
                                                            'DirichletBC per (species, '
                                                            'boundary) pair and pass '
                                                            'the list.',
                                                'explanation': 'Concentrations are '
                                                               'usually confined '
                                                               '(no-flux) or fed '
                                                               'through a known flux, '
                                                               'so most '
                                                               'reaction-diffusion '
                                                               'problems need no '
                                                               'Dirichlet bc at all -- '
                                                               'and with no-flux '
                                                               'everywhere the total '
                                                               'amount of each '
                                                               'conserved element is '
                                                               'exactly preserved, '
                                                               'which is the best '
                                                               'available correctness '
                                                               'check. A Dirichlet bc '
                                                               'on one species of a '
                                                               'mixed space with a '
                                                               'spatially varying value '
                                                               'needs the '
                                                               'collapsed sub-space in '
                                                               'BOTH the dof lookup '
                                                               'and the value '
                                                               'Function; a constant '
                                                               'value can instead be a '
                                                               'fem.Constant with the '
                                                               'plain W.sub(i) dof '
                                                               'array.',
                                                'pitfalls': ['A Dirichlet bc on one species has two '
                                                             'valid spellings. (a) For a value that '
                                                             'varies in space, use a Function on the '
                                                             'collapsed space together with the '
                                                             '(sub-space, collapsed-space) PAIR: V0, '
                                                             '_ = W.sub(0).collapse(); dofs = '
                                                             'fem.locate_dofs_topological((W.sub(0), '
                                                             'V0), fdim, facets); fem.dirichletbc(g, '
                                                             'dofs, W.sub(0)) with g = '
                                                             'fem.Function(V0). (b) For a constant '
                                                             'value, use a fem.Constant (or a '
                                                             'scalar) with the plain dof array: '
                                                             'fem.dirichletbc(fem.Constant(msh, '
                                                             '1.5), '
                                                             'fem.locate_dofs_topological(W.sub(0), '
                                                             'fdim, facets), W.sub(0)); the '
                                                             "Constant's size must equal the "
                                                             'sub-space block size. Signal: mixing '
                                                             'the two (a Constant with the pair, a '
                                                             'Function with the plain array) or '
                                                             'omitting W.sub(0) raises TypeError: '
                                                             '__init__(): incompatible function '
                                                             'arguments. The following argument '
                                                             'types are supported: ... with a line '
                                                             'such as Invoked with types: '
                                                             'dolfinx.cpp.fem.DirichletBC_float64, '
                                                             'dolfinx.cpp.fem.Constant_float64, '
                                                             'list, '
                                                             'dolfinx.cpp.fem.FunctionSpace_float64.',
                                                             'fem.Function.collapse() '
                                                             'returns a SINGLE '
                                                             'Function; '
                                                             'fem.FunctionSpace.collapse() '
                                                             'returns a (space, '
                                                             'dofmap) pair. Signal: a, '
                                                             'b = w.sub(0).collapse() '
                                                             'raises '
                                                             'NotImplementedError: '
                                                             'Cannot take length of '
                                                             'non-vector expression.']},
                        'solver': {'REQUIRED': 'from dolfinx.fem.petsc import '
                                               'NonlinearProblem\n'
                                               'problem = NonlinearProblem(\n'
                                               '    F, w, petsc_options_prefix="rd_",\n'
                                               '    petsc_options={"snes_type": '
                                               '"newtonls", "snes_rtol": 1e-10,\n'
                                               '                   "snes_atol": 1e-12, '
                                               '"snes_max_it": 30,\n'
                                               '                   "ksp_type": '
                                               '"preonly", "pc_type": "lu"})\n'
                                               '\n'
                                               'for step in range(n_steps):\n'
                                               '    problem.solve()\n'
                                               '    if '
                                               'problem.solver.getConvergedReason() <= '
                                               '0:\n'
                                               '        raise RuntimeError("SNES '
                                               'failed")\n'
                                               '    w_n.x.array[:] = '
                                               'w.x.array          # LAST line of the '
                                               'loop body',
                                   'OPTIONAL': 'bcs=[...] is optional (omit it for an '
                                               'all-no-flux problem). J= is optional '
                                               'and should normally be OMITTED. '
                                               'petsc_options is optional; useful '
                                               'entries are "snes_monitor": None and '
                                               '"ksp_monitor": None for iteration '
                                               'printout, "snes_linesearch_type": "bt" '
                                               'or "basic", and '
                                               '"snes_error_if_not_converged": True to '
                                               'turn a silent failure into an '
                                               'exception. For large 3-D problems '
                                               'replace the LU preconditioner by '
                                               '"ksp_type": "gmres" with "pc_type": '
                                               '"hypre" or "gamg" (the coupled system '
                                               'is nonsymmetric, so cg is not safe). '
                                               'Useful SNES converged-reason values: 2 '
                                               '= CONVERGED_FNORM_ABS, 3 = '
                                               'CONVERGED_FNORM_RELATIVE, 4 = '
                                               'CONVERGED_SNORM_RELATIVE, -5 = '
                                               'DIVERGED_MAX_IT, -6 = '
                                               'DIVERGED_LINE_SEARCH.',
                                   'explanation': 'In dolfinx 0.10 NonlinearProblem '
                                                  'owns a PETSc SNES; you call '
                                                  'problem.solve() directly and '
                                                  'inspect problem.solver. When J is '
                                                  'not supplied, dolfinx builds it '
                                                  'with ufl.derivative(F, w, '
                                                  'TrialFunction(W)) -- the exact '
                                                  'Jacobian including every '
                                                  'cross-species term -- so there is '
                                                  'nothing to gain by writing one.',
                                   'pitfalls': ['petsc_options_prefix is a required '
                                                'keyword-only argument. Signal: '
                                                'TypeError: '
                                                'NonlinearProblem.__init__() missing 1 '
                                                'required keyword-only argument: '
                                                "'petsc_options_prefix'.",
                                                'Do not wrap a 0.10 NonlinearProblem '
                                                'in dolfinx.nls.petsc.NewtonSolver. '
                                                'Signal: AttributeError: '
                                                "'NonlinearProblem' object has no "
                                                "attribute 'a'.",
                                                'A converged SNES does not mean a '
                                                'sound solution: check the field too. '
                                                'Signal: with an explicit (theta=0) '
                                                'treatment of a stiff reaction the '
                                                'SNES reports converged reason 3 on '
                                                'the very steps where max|c| is racing '
                                                'past 1e6.',
                                                'Copy w into w_n at the end of every '
                                                'step, or nothing advances. Signal: '
                                                'with the copy missing the Newton '
                                                'count per step collapses to 3, 1, 0, '
                                                '0, 0, ... and min(w) is identical at '
                                                'every step; the SNES still reports '
                                                'converged reason 2.']},
                        'time_integration': {'REQUIRED': 'dt = fem.Constant(msh, '
                                                         'dt_val)\n'
                                                         '# backward Euler: every term '
                                                         'of F except (..._n) '
                                                         'evaluated at the new step\n'
                                                         '# and, at the end of each '
                                                         'step:  w_n.x.array[:] = '
                                                         'w.x.array',
                                             'OPTIONAL': 'theta-method: F = (w - '
                                                         'w_n)/dt * v * dx + '
                                                         'theta*op(new) + '
                                                         '(1-theta)*op(old). theta = 1 '
                                                         '(backward Euler) is the only '
                                                         'choice that survived every '
                                                         'stiffness tested here; theta '
                                                         '= 0.5 (Crank-Nicolson) still '
                                                         'runs but the Newton line '
                                                         'search starts failing on '
                                                         'stiff kinetics; theta = 0 '
                                                         '(explicit) is unusable for '
                                                         'stiff reactions. Adaptive '
                                                         'stepping: shrink dt and '
                                                         're-solve whenever '
                                                         'problem.solver.getConvergedReason() '
                                                         '<= 0 or the Newton iteration '
                                                         'count climbs.',
                                             'explanation': 'Reaction terms with a '
                                                            'large rate constant make '
                                                            'the system stiff: the '
                                                            'reaction time scale 1/k '
                                                            'is far shorter than the '
                                                            'diffusion time scale '
                                                            'L^2/D. An implicit '
                                                            'treatment removes the '
                                                            'step-size restriction '
                                                            'entirely, and it is cheap '
                                                            'here because the same '
                                                            'Newton solve already '
                                                            'handles the nonlinearity.',
                                             'pitfalls': ['An explicit (theta = 0) '
                                                          'reaction term blows up at '
                                                          'time steps that backward '
                                                          'Euler handles comfortably. '
                                                          'Signal: with a forward rate '
                                                          'of 100 the explicit run '
                                                          'passes 1e6 in max|c| at '
                                                          'step 3 for dt=0.05 and at '
                                                          'step 4 for dt=0.01, while '
                                                          'theta=1 runs every step to '
                                                          'completion with max|c| '
                                                          'below 1.0.',
                                                          'No external ODE package is '
                                                          'needed for stiff kinetics. '
                                                          'Signal: plain backward '
                                                          'Euler in dolfinx converged '
                                                          '(reason 2) at every step '
                                                          'for forward rate constants '
                                                          'from 1e2 up to 1e8 at dt = '
                                                          '0.05 and dt = 0.01, with '
                                                          'the conserved quantity '
                                                          'unchanged.']},
                        'conservation_check': {'REQUIRED': 'mass_f = fem.form((a + 2.0 '
                                                           '* b) * ufl.dx)     # the '
                                                           'conserved combination\n'
                                                           'M0 = '
                                                           'msh.comm.allreduce(fem.assemble_scalar(mass_n_f), '
                                                           'op=MPI.SUM)\n'
                                                           '# ... time loop ...\n'
                                                           'M1 = '
                                                           'msh.comm.allreduce(fem.assemble_scalar(mass_f), '
                                                           'op=MPI.SUM)\n'
                                                           'assert abs(M1 - M0) / '
                                                           'abs(M0) < 1e-10',
                                               'OPTIONAL': 'The conserved combination '
                                                           'follows from the '
                                                           'stoichiometry: for n*A <-> '
                                                           'B it is int(A + n*B) dx; '
                                                           'for a closed system with '
                                                           'several reactions it is '
                                                           'any atom balance. It is '
                                                           'conserved exactly only '
                                                           'when EVERY boundary is '
                                                           'no-flux; with an influx '
                                                           'term the same identity '
                                                           'becomes d/dt int(...) dx = '
                                                           'total influx.',
                                               'explanation': 'Because the constant '
                                                              'test function (1, n) '
                                                              'lies in the mixed '
                                                              'space, the reaction '
                                                              'contributions cancel '
                                                              'identically in the '
                                                              'discrete residual, so '
                                                              'this balance holds to '
                                                              'round-off for any mesh, '
                                                              'degree and step size. '
                                                              'It is the strongest '
                                                              'correctness check '
                                                              'available without a '
                                                              'reference solution, and '
                                                              'it catches a wrong '
                                                              'stoichiometric factor, '
                                                              'a wrong sign and a '
                                                              'missing term at once.',
                                               'pitfalls': ['A wrong stoichiometric '
                                                            'factor or sign shows up '
                                                            'immediately as a drift. '
                                                            'Signal: the correct '
                                                            '2A<->B system keeps int(A '
                                                            '+ 2B) dx at a relative '
                                                            'drift of 1e-15 or smaller '
                                                            'over 25 steps; writing '
                                                            'the factor as 1.0 instead '
                                                            'of 2.0 on the A equation '
                                                            'drifts by 4.04e-01, and '
                                                            'flipping its sign makes '
                                                            'the SNES fail with reason '
                                                            '-6 (DIVERGED_LINE_SEARCH) '
                                                            'and a nan.']},
                        'pitfalls': ['[API] Build the mixed space with '
                                     'basix.ufl.mixed_element and a LIST of elements, '
                                     'then fem.functionspace. Signal: '
                                     'basix.ufl.mixed_element(P1, P1) raises '
                                     'TypeError: mixed_element() takes 1 positional '
                                     'argument but 2 were given, and the legacy '
                                     'ufl.MixedElement([P1, P1]) raises '
                                     "AttributeError: module 'ufl' has no attribute "
                                     "'MixedElement'. The working call is "
                                     'basix.ufl.mixed_element([P1, P1]) with P1 = '
                                     "basix.ufl.element('Lagrange', msh.basix_cell(), "
                                     '1). Species may use different degrees: '
                                     'mixed_element([P1, P2]) builds a valid '
                                     '2-subspace space. If all species share one '
                                     "element, fem.functionspace(msh, ('Lagrange', 1, "
                                     '(n,))) is an equivalent blocked space.',
                                     '[API] Take the species out of the mixed Function '
                                     'with ufl.split(w) when building the residual. '
                                     'w.sub(i) and w.split() also COMPILE, which is '
                                     'what makes this dangerous, but the automatic '
                                     'Jacobian is then empty on those rows. Signal: '
                                     '[MEASURED, dolfinx 0.10.0, two-species mixed P1 '
                                     'system] with a, b = ufl.split(w) the SNES '
                                     'converges in 3 iterations (residuals 1.80e-03, '
                                     '4.73e-07, 3.28e-14); with a, b = w.sub(0), '
                                     'w.sub(1) or a, b = w.split() the same script '
                                     'raises Error: error code 73 with the stack '
                                     'ending in [0] PCSetUp_LU() ... [0] '
                                     'MatLUFactorSymbolic_SeqAIJ() ... [0] Object is '
                                     'in wrong state / [0] Matrix is missing diagonal '
                                     'entry 0. Use w.sub(i).collapse() only for '
                                     'POST-PROCESSING (extracting a species for output '
                                     'or for min/max), never inside the residual.',
                                     '[API] fem.Function.collapse() returns one '
                                     'Function; fem.FunctionSpace.collapse() returns a '
                                     '(FunctionSpace, dofmap) pair. Signal: a_h, dofs '
                                     '= w.sub(0).collapse() raises '
                                     'NotImplementedError: Cannot take length of '
                                     'non-vector expression., because Python tries to '
                                     'iterate the returned Function. The correct forms '
                                     'are a_h = w.sub(0).collapse() (a Function on the '
                                     'collapsed space, ready for .x.array.min()) and '
                                     'V0, sub_map = W.sub(0).collapse() (the space '
                                     'plus the index array that maps its dofs into the '
                                     'parent vector, so w.x.array[sub_map] is that '
                                     "species' block).",
                                     '[API] A Dirichlet condition on ONE species of a '
                                     'mixed space has two valid spellings. For a '
                                     'spatially varying value: V0, _ = '
                                     'W.sub(0).collapse(); dofs = '
                                     'fem.locate_dofs_topological((W.sub(0), V0), '
                                     'fdim, facets); g = fem.Function(V0); bc = '
                                     'fem.dirichletbc(g, dofs, W.sub(0)). For a '
                                     'constant value: bc = '
                                     'fem.dirichletbc(fem.Constant(msh, val), '
                                     'fem.locate_dofs_topological(W.sub(0), fdim, '
                                     "facets), W.sub(0)), where the Constant's size "
                                     'must equal the sub-space block size. Signal: '
                                     '[MEASURED] mixing them (a fem.Constant with the '
                                     '(W.sub(0), V0) pair, a Function on V0 with the '
                                     'plain W.sub(0) dof array) or omitting the '
                                     'trailing W.sub(0) all raise the same TypeError: '
                                     '__init__(): incompatible function arguments. The '
                                     'following argument types are supported: ... '
                                     'ending in a line such as Invoked with types: '
                                     'dolfinx.cpp.fem.DirichletBC_float64, '
                                     'dolfinx.cpp.fem.Constant_float64, list, '
                                     'dolfinx.cpp.fem.FunctionSpace_float64.',
                                     '[Numerical] Nonlinear reaction terms need a true '
                                     'Newton solve; lagging one factor of the '
                                     'nonlinearity (Picard) is far worse than it '
                                     'looks. Signal: [MEASURED, dolfinx 0.10.0, steady '
                                     '-0.01*lap(u) + 5*u^2 = 1 on a 24x24 unit square '
                                     'with u=1 on the whole boundary, initial guess '
                                     'u=1] Newton via NonlinearProblem reaches ||F|| '
                                     'below 1e-14 in 5 iterations with the sequence '
                                     '1.597e-01, 2.879e-02, 2.766e-03, 4.657e-05, '
                                     '1.657e-08, 2.460e-15 -- the exponents roughly '
                                     'double each step, which is quadratic '
                                     'convergence. The Picard iteration (solve '
                                     '-0.01*lap(u_new) + 5*u_old*u_new = 1 repeatedly '
                                     'with LinearProblem) is still at ||F|| = 2.97e-09 '
                                     'after 200 iterations. SCORE BOTH METHODS ON THE '
                                     'SAME QUANTITY, which is the whole trick here: the '
                                     'l2 norm of the TRUE nonlinear residual with the '
                                     'Dirichlet rows zeroed. Do that and the Picard '
                                     'ratio is STEADY - a near-constant factor per '
                                     'iteration, close enough to one that the iteration '
                                     'is still orders of magnitude above the Newton '
                                     'answer after forty times the work, while both are '
                                     'converging on the same solution. IMPORTANT '
                                     'CORRECTION: an earlier version of this entry said '
                                     'the ratio ALTERNATES between a small and a '
                                     'greater-than-one value; that alternation is an '
                                     'artefact of scoring the two methods on different '
                                     'residuals and does not reproduce when the same '
                                     'residual is used for both. If you see an '
                                     'alternating ratio, you are measuring two '
                                     'different quantities, not a property of Picard - '
                                     'and do not wait for the ratio to reach ~0.5 '
                                     'either, which the earlier claim before that one '
                                     'quoted. Same behaviour on triangles and '
                                     'quadrilaterals at P1 and P2. (Verified by execution on dolfinx 0.10.0, 2026-08-06.)',
                                     '[API] Do NOT supply J to NonlinearProblem and do '
                                     'NOT reach for ufl.variable / ufl.diff: dolfinx '
                                     'already differentiates the residual exactly. '
                                     'Signal: [MEASURED] '
                                     'dolfinx.fem.petsc.NonlinearProblem calls '
                                     "derivative_block(F, u), whose source says 'This "
                                     'is identical to calling ufl.derivative '
                                     "directly', and the assembled matrices agree "
                                     'exactly -- ||J_auto - '
                                     'J_from_ufl.derivative||_inf = 0.000e+00 against '
                                     '||J_auto||_inf = 1.834e-01. IMPORTANT CORRECTION '
                                     'to the previously written advice: ufl.diff '
                                     'cannot even be applied to a species pulled out '
                                     'of a mixed Function -- ufl.diff(k1*a*a*b - k2*b, '
                                     'a) raises ValueError: Expecting a Variable or '
                                     'SpatialCoordinate in diff. Wrapping the species '
                                     'in ufl.variable makes ufl.diff compile, but it '
                                     'yields only the scalar partial dR/dA, not a '
                                     'Jacobian form, so it still has to be assembled '
                                     'into a bilinear form by hand. The real risk is '
                                     'exactly that hand assembly: a Jacobian that '
                                     'drops one cross-species partial made SNES take 8 '
                                     'iterations with a linear residual ratio of about '
                                     '0.13 (2.37e-03, 3.09e-04, 4.05e-05, 5.30e-06, '
                                     '...) instead of 3 quadratic ones, and on a '
                                     'single-species quadratic reaction it hit '
                                     'snes_max_it and returned reason -5 '
                                     '(DIVERGED_MAX_IT) with the residual stuck at '
                                     '8.0e-06.',
                                     '[Numerical] Species concentrations really do go '
                                     'negative, and the cause is the step size '
                                     "relative to h^2, not merely 'steep gradients'. "
                                     'Signal: [MEASURED, dolfinx 0.10.0, pure '
                                     'diffusion (D=1) of a sharp blob of height 1.0 on '
                                     'the unit square, all boundaries no-flux, '
                                     'BACKWARD Euler -- so the time integrator is as '
                                     'stable as it gets, 5 steps, 32x32 mesh where '
                                     'h^2/(6D) = 1.63e-04] min(c) is exactly 0.0 for '
                                     'dt = 1e-2 and 1e-3, then -2.87e-03 at dt = 3e-4, '
                                     '-1.94e-02 at dt = 1e-4 and -2.75e-02 at dt = '
                                     '3e-5, i.e. an undershoot of up to 2.8% of the '
                                     'peak on P1 triangles; the worst values were '
                                     '-5.05e-02 (P2 triangles), -1.85e-02 (P1 '
                                     'quadrilaterals) and -3.45e-02 (P2 '
                                     'quadrilaterals). The total mass stays exactly '
                                     'conserved (0.06738281) throughout, so a mass '
                                     'check will NOT catch this. Two cures were '
                                     'verified: refine the mesh until h^2 is small '
                                     'compared with dt (at dt=1e-4 the undershoot '
                                     'falls from -2.82e-02 at N=16 to exactly 0.0 at '
                                     'N=128), or lump the mass matrix by integrating '
                                     'the time-derivative term with '
                                     "ufl.dx(metadata={'quadrature_rule': 'vertex', "
                                     "'quadrature_degree': 1}), which gave min(c) = "
                                     'exactly 0.0 at every dt tested. NOTE the lumping '
                                     'trick is only correct for degree 1 -- with P2 '
                                     'the vertex rule under-integrates and silently '
                                     'changes the answer (the total mass came out as '
                                     'the P1 value).',
                                     '[Numerical] Stiff kinetics need an implicit '
                                     'treatment; an explicit or theta<0.5 reaction '
                                     'term diverges at step sizes backward Euler '
                                     'handles easily. Signal: [MEASURED, two-species '
                                     '2A<->B on a 24x24 unit square, D_A=0.01, 40 '
                                     'steps, blow-up declared when max|c| exceeds 1e6] '
                                     'with forward rate 100 the theta=0 run passes 1e6 '
                                     'at step 3 for dt=0.05 and at step 4 for dt=0.01, '
                                     'while theta=1 completes every step with max|c| '
                                     'below 1.0; even at forward rate 1 the theta=0 '
                                     'run blows up at dt=0.05. theta=0 becomes stable '
                                     'again at dt = 2e-3 and below. Crank-Nicolson '
                                     '(theta=0.5) stays bounded, but at forward rate '
                                     '100 with dt=0.01 its Newton line search stalls '
                                     'in the first step (lambda shrinks to about 1e-6 '
                                     'while ||F|| stays near 0.575) and SNES returns a '
                                     'negative reason whose code depends on the '
                                     "element and on snes_max_it: with the example's "
                                     'snes_max_it=30 it is -5 (DIVERGED_MAX_IT) on P1 '
                                     'and P2 triangles and P1 quadrilaterals and -6 '
                                     '(DIVERGED_LINE_SEARCH) on P2 quadrilaterals; '
                                     'with snes_max_it=100, P1 triangles give -6 as '
                                     'well. Verified on triangles and quadrilaterals '
                                     'at P1 and P2. CRITICAL detail: on the diverging '
                                     'steps the SNES still reports converged reason 3 '
                                     '-- the nonlinear solver is solving each step '
                                     'correctly, the SCHEME is unstable -- so you must '
                                     'test the field (np.isfinite and a magnitude '
                                     'bound), not just the solver reason.',
                                     '[Integration] An external stiff-ODE package is '
                                     'NOT required for high Damkohler numbers in '
                                     "dolfinx; the previously written claim that 'for "
                                     'very stiff systems (Da > 1000) external SUNDIALS '
                                     "coupling is required' is FALSIFIED. Signal: "
                                     '[MEASURED] plain backward Euler through '
                                     'NonlinearProblem converged with SNES reason 2 at '
                                     'every one of 40 steps for forward rate constants '
                                     'of 1e2, 1e3, 1e4, 1e5, 1e6 and 1e8 (Damkohler = '
                                     'k*L^2/D from 1e4 to 1e10) at both dt = 0.05 and '
                                     'dt = 0.01, with the conserved quantity unchanged '
                                     'and max|c| bounded between 0.86 and 0.94. '
                                     'Nothing named sundials, scikits.odes, assimulo '
                                     'or cvode is importable in a standard conda-forge '
                                     'fenics environment, so an entry that recommends '
                                     'it sends the user to a package that is not '
                                     'there.',
                                     '[Numerical] Use the stoichiometric conservation '
                                     'law as the correctness check, because it holds '
                                     'to round-off independently of mesh, degree and '
                                     'step size. Signal: [MEASURED, the two-species '
                                     '2A<->B example, all boundaries no-flux, 25 '
                                     'backward-Euler steps] the conserved combination '
                                     'assembled with '
                                     'dolfinx.fem.assemble_scalar(dolfinx.fem.form((A_h '
                                     '+ 2*B_h)*ufl.dx)) starts at 1.400162760417 and '
                                     'ends at 1.400162760417, a relative drift of '
                                     '1.744e-15, while the individual species ranges '
                                     'have moved from A in [0.5, 1.5] and B = 0.2 to A '
                                     'in [0.271, 0.395] and B in [0.331, 0.739]. The '
                                     'constant test function (1, 2) lies in the mixed '
                                     'space, so the reaction terms cancel identically '
                                     'in the discrete residual. A wrong stoichiometric '
                                     'factor, a wrong sign on one of the two reaction '
                                     'terms or a missing term all show up as an '
                                     'immediate drift.',
                                     '[API] dolfinx.fem.petsc.NonlinearProblem '
                                     'requires petsc_options_prefix as a keyword-only '
                                     'argument and is driven by calling '
                                     'problem.solve() directly; its solver attribute '
                                     'is a petsc4py SNES. Signal: omitting the prefix '
                                     'gives TypeError: NonlinearProblem.__init__() '
                                     'missing 1 required keyword-only argument: '
                                     "'petsc_options_prefix', and wrapping the problem "
                                     'in the legacy dolfinx.nls.petsc.NewtonSolver '
                                     "gives AttributeError: 'NonlinearProblem' object "
                                     "has no attribute 'a'. Read convergence with "
                                     'problem.solver.getConvergedReason() (2 = '
                                     'CONVERGED_FNORM_ABS, 3 = '
                                     'CONVERGED_FNORM_RELATIVE, 4 = '
                                     'CONVERGED_SNORM_RELATIVE, -5 = DIVERGED_MAX_IT, '
                                     '-6 = DIVERGED_LINE_SEARCH) and '
                                     'problem.solver.getIterationNumber().',
                                     '[API] Set the initial condition per species with '
                                     'w_n.sub(i).interpolate(...), then '
                                     'w_n.x.scatter_forward(), then seed the Newton '
                                     'guess with w.x.array[:] = w_n.x.array, and end '
                                     'every step with w_n.x.array[:] = w.x.array. '
                                     'Signal: [MEASURED, 16x16 unit square, 10 '
                                     'backward-Euler steps of the two-species example] '
                                     'with the copy, min(w) walks 0.217637, 0.232551, '
                                     '0.245252, ... and the Newton count is 3 every '
                                     'step; with the copy missing, min(w) is 0.217637 '
                                     'at all ten steps and the Newton count collapses '
                                     'to its floor from the second step on -- the '
                                     'solver is re-converging on a problem it has '
                                     'already solved. Nothing is raised and the SNES '
                                     'reports a POSITIVE reason throughout. Two things '
                                     'to watch, and take them together: the iteration '
                                     'count collapsing, AND the converged reason '
                                     'switching to the small-update criterion '
                                     '(SNORM_RELATIVE, reason 4 -- "converged because '
                                     'the update was tiny"), which is exactly what '
                                     're-solving a solved step looks like. CORRECTION: '
                                     'an earlier version of this entry said the count '
                                     'drops to 0 and called that the only visible tell. '
                                     'The floor is not always 0 -- it can sit at 1 with '
                                     'reason 4 -- so a check that waits for a literal '
                                     'zero will miss the bug. The cheapest independent '
                                     'confirmation is that the state stops moving: '
                                     'min(w) must WALK from step to step, and a value '
                                     'repeated to full precision across every step '
                                     'means the copy is missing. Same on triangles and '
                                     'quadrilaterals at P1 and P2. (Verified by execution on dolfinx 0.10.0, 2026-08-06.)']},
 'nearly_incompressible_elasticity': {'description': 'Mixed methods for nearly '
                                                     'incompressible elasticity (nu -> '
                                                     '0.5) to avoid volumetric '
                                                     'locking.',
                                      'weak_form': 'For p = -lambda*div(u), use the '
                                                   'symmetric form '
                                                   '2*mu*inner(eps(u),eps(v))*dx - '
                                                   'p*div(v)*dx - q*div(u)*dx - '
                                                   '(1/lambda)*p*q*dx = dot(f,v)*dx. '
                                                   'Multiplying the entire pressure '
                                                   'constraint row by -1 changes no '
                                                   'solution; changing only one sign '
                                                   'changes the PDE.',
                                      'function_space': 'Taylor-Hood mixed element: '
                                                        'continuous vector Lagrange P2 '
                                                        'for displacement + continuous '
                                                        'scalar Lagrange P1 for pressure',
                                      'approach': {'displacement_pressure': 'u-p '
                                                                            'formulation: '
                                                                            'displacement '
                                                                            '(vector) '
                                                                            '+ '
                                                                            'pressure '
                                                                            '(scalar) '
                                                                            'as '
                                                                            'independent '
                                                                            'unknowns',
                                                   'three_field': 'u-p-theta: '
                                                                  'displacement + '
                                                                  'pressure + '
                                                                  'dilatation (for '
                                                                  'Neo-Hookean)'},
                                      'solver': 'Direct MUMPS for moderate problems; '
                                                'MINRES or GMRES with a block '
                                                'preconditioner at scale '
                                                '(saddle-point structure)',
                                      'pitfalls': ['[API] A Dirichlet condition on the '
                                                   'displacement leg of a mixed space '
                                                   'must retain the map to the PARENT '
                                                   'space. Use W0 = W.sub(0); V0, _ = '
                                                   'W0.collapse(); then '
                                                   'locate_dofs_topological((W0, V0), '
                                                   'fdim, facets) and '
                                                   'dirichletbc(u0, dofs, W0). Locating '
                                                   'dofs on V0 alone and passing a V0 '
                                                   'BC into a problem assembled on W '
                                                   'does not constrain the intended '
                                                   'parent unknowns. Signal: a fully '
                                                   'clamped problem solves to '
                                                   'displacements of order 1e12-1e13 '
                                                   'and changes wildly under refinement '
                                                   'although the load is bounded.',
                                                   '[Numerical] Low-order displacement '
                                                   'formulations LOCK as nu -> 0.5 — a '
                                                   'mixed (u, p) method is the robust '
                                                   'fix. Signal: [MEASURED 2026-08-03, '
                                                   'dolfinx 0.10.0; cantilever 1.0 x '
                                                   '0.2 with end traction, tip '
                                                   'deflection against a P2/P1 '
                                                   'Taylor-Hood reference] P1 '
                                                   'triangles are 7.2x / 3.2x / 1.6x / '
                                                   '1.2x too stiff at nu=0.49 on 10x2 '
                                                   '/ 20x4 / 40x8 / 80x16 meshes; '
                                                   '16.5x / 11.4x / 5.5x / 2.4x at '
                                                   'nu=0.499; 19.5x / 18.7x / 15.4x / '
                                                   '9.2x at nu=0.4999. P2 triangles '
                                                   'are within 0-6% of the mixed '
                                                   'reference at EVERY nu and mesh '
                                                   'tested. IMPORTANT CORRECTION: the '
                                                   'locking ratio is NOT ~1/(1-2nu) — '
                                                   'that formula predicts 500x at '
                                                   'nu=0.499 where 2.4x-16.5x is '
                                                   'measured, depending on mesh. '
                                                   'Locking magnitude depends jointly '
                                                   'on nu, element order and h; quote '
                                                   'a measured ratio, not the '
                                                   '1/(1-2nu) rule.',
                                                   '[Numerical] Inf-sup (LBB) '
                                                   'condition: pressure FunctionSpace '
                                                   'must be STRICTLY SMALLER than the '
                                                   'displacement FunctionSpace. '
                                                   'Signal: [measured on the Stokes '
                                                   'analogue 2026-08-03, dolfinx '
                                                   '0.10.0] SVD of the bc-applied '
                                                   'saddle-point matrix on an 8x8 unit '
                                                   'square gives numerical null '
                                                   'dimension 1 for P2/P1 Taylor-Hood '
                                                   '(the constant pressure alone) but '
                                                   '8 for equal-order P1/P1 — the '
                                                   'extra kernel vectors ARE the '
                                                   'checkerboard modes. The LBB '
                                                   'constant collapsing with h is the '
                                                   'diagnostic for inf-sup failure.',
                                                   '[Numerical] Taylor-Hood (P2/P1) or (P2/DG0) satisfy inf-sup; P1/P0 '
                                                   'does NOT. Signal: in the incompressible limit (no -(1/lambda)*p*q '
                                                   'term, clamped boundary) the bc-applied P1/DG0 saddle-point matrix is '
                                                   'genuinely singular: its numerical null dimension grows with '
                                                   'refinement (14 at 4x4, 30 at 8x8), and the MUMPS direct solve comes '
                                                   'back with a NEGATIVE converged reason (-11, DIVERGED_PCSETUP_FAILED) '
                                                   'and a non-finite field at every level. With the -(1/lambda)*p*q term '
                                                   'of this entry the matrix is nonsingular and the outcome depends on nu '
                                                   'and the mesh: at nu = 0.49 and 0.499, P1/DG0 solves (reason 4, finite '
                                                   'field) from 4x4 to 64x64 but is far too stiff on coarse meshes '
                                                   '(max|u| 0.0012 against 0.0192 for Taylor-Hood at 4x4, nu = 0.499), '
                                                   'and at nu = 0.4999 it solves at 4x4 and 8x8 while MUMPS fails (-11, '
                                                   'non-finite field) from 16x16 on. Taylor-Hood solves at every level. '
                                                   'Check `getConvergedReason() > 0` and `np.isfinite(...).all()` on the '
                                                   'result, and compare the displacement with a Taylor-Hood run on the '
                                                   'same mesh. (Verified by execution on dolfinx 0.10.0.)',
                                                   '[Numerical] Penalty method (large '
                                                   'kappa) is alternative but '
                                                   'introduces parameter sensitivity. '
                                                   'Signal: penalty too small -> '
                                                   'volumetric locking returns '
                                                   '(det(F)-1 deviates by > 1% from '
                                                   '0); penalty too large -> condition '
                                                   'number exceeds 1e14 and Newton '
                                                   'stalls. Mixed method is '
                                                   'parameter-free and preferred for '
                                                   'production runs. (Audit '
                                                   '2026-06-02.)']},
 'contact': {'description': 'Contact mechanics in FEniCSx: a body is prevented from '
                            'penetrating a rigid obstacle or another body. DOLFINx has '
                            'no contact solver of its own, so the constraint is '
                            'imposed by hand, most simply by adding a one-sided '
                            'penalty term built with ufl.max_value on the potentially '
                            'contacting surface and solving the resulting nonlinear '
                            'problem with dolfinx.fem.petsc.NonlinearProblem.',
             'minimal_working_example': 'import numpy as np\n'
                                        'import ufl\n'
                                        'from mpi4py import MPI\n'
                                        'from petsc4py import PETSc\n'
                                        'from dolfinx import mesh, fem, '
                                        'default_scalar_type\n'
                                        'from dolfinx.fem.petsc import '
                                        'NonlinearProblem, assemble_vector\n'
                                        '\n'
                                        'E, nu = 1000.0, 0.3\n'
                                        'mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + '
                                        'nu) * (1 - 2 * nu))\n'
                                        'g0 = 0.005          # initial gap between '
                                        'block bottom and the rigid plane\n'
                                        'delta = 0.02        # prescribed downward '
                                        'displacement of the top face\n'
                                        'nx, ny = 24, 12\n'
                                        'h = 0.5 / ny\n'
                                        'kappa_val = 1.0e2 * E / h    # penalty '
                                        'stiffness: 1e2 * E / h\n'
                                        '\n'
                                        'msh = mesh.create_rectangle(MPI.COMM_WORLD, '
                                        '[[0.0, 0.0], [1.0, 0.5]],\n'
                                        '                            [nx, ny], '
                                        'mesh.CellType.triangle)\n'
                                        'gdim = msh.geometry.dim\n'
                                        'V = fem.functionspace(msh, ("Lagrange", 1, '
                                        '(gdim,)))\n'
                                        'u = fem.Function(V, name="u")\n'
                                        'v = ufl.TestFunction(V)\n'
                                        '\n'
                                        '\n'
                                        'def sigma(w):\n'
                                        '    return 2 * mu * ufl.sym(ufl.grad(w)) + '
                                        'lam * ufl.div(w) * ufl.Identity(gdim)\n'
                                        '\n'
                                        '\n'
                                        'fdim = msh.topology.dim - 1\n'
                                        'bot = mesh.locate_entities_boundary(msh, '
                                        'fdim, lambda x: np.isclose(x[1], 0.0))\n'
                                        'top = mesh.locate_entities_boundary(msh, '
                                        'fdim, lambda x: np.isclose(x[1], 0.5))\n'
                                        'mt = mesh.meshtags(msh, fdim, np.sort(bot), '
                                        'np.full(len(bot), 1, dtype=np.int32))\n'
                                        'ds = ufl.Measure("ds", domain=msh, '
                                        'subdomain_data=mt)\n'
                                        '\n'
                                        'kappa = fem.Constant(msh, '
                                        'default_scalar_type(kappa_val))\n'
                                        'gap = u[1] + '
                                        'g0                                   # gap < '
                                        '0  =>  penetration\n'
                                        'pressure = kappa * ufl.max_value(-gap, '
                                        '0.0)       # one-sided penalty pressure\n'
                                        'F = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) '
                                        '* ufl.dx - pressure * v[1] * ds(1)\n'
                                        '\n'
                                        'uD = fem.Function(V)\n'
                                        'uD.sub(1).interpolate(lambda x: '
                                        'np.full_like(x[0], -delta))\n'
                                        'tdofs = fem.locate_dofs_topological(V, fdim, '
                                        'top)\n'
                                        'bc = fem.dirichletbc(uD, tdofs)\n'
                                        '\n'
                                        'problem = NonlinearProblem(\n'
                                        '    F, u, bcs=[bc], '
                                        'petsc_options_prefix="contact_",\n'
                                        '    petsc_options={"snes_type": "newtonls", '
                                        '"snes_linesearch_type": "l2",\n'
                                        '                   "snes_rtol": 1e-10, '
                                        '"snes_atol": 1e-12, "snes_max_it": 50,\n'
                                        '                   "ksp_type": "preonly", '
                                        '"pc_type": "lu"})\n'
                                        'problem.solve()\n'
                                        'reason = problem.solver.getConvergedReason()\n'
                                        'assert reason > 0, f"SNES did not converge, '
                                        'reason {reason}"\n'
                                        '\n'
                                        'comm = msh.comm\n'
                                        'Rc = '
                                        'comm.allreduce(fem.assemble_scalar(fem.form(pressure '
                                        '* ds(1))), MPI.SUM)\n'
                                        'b = assemble_vector(fem.form(F))\n'
                                        'b.ghostUpdate(addv=PETSc.InsertMode.ADD, '
                                        'mode=PETSc.ScatterMode.REVERSE)\n'
                                        'owned = tdofs[tdofs < '
                                        'V.dofmap.index_map.size_local]\n'
                                        'Rtop = comm.allreduce(float(b.array_r[owned * '
                                        'gdim + 1].sum()), MPI.SUM)\n'
                                        '\n'
                                        'bdofs = fem.locate_dofs_topological(V.sub(1), '
                                        'fdim, bot)\n'
                                        'uy_bot = '
                                        'comm.allreduce(float(u.x.array[bdofs].min()), '
                                        'MPI.MIN)\n'
                                        'pen = max(0.0, -(uy_bot + g0))\n'
                                        '\n'
                                        'print(f"SNES reason {reason}, iterations '
                                        '{problem.solver.getIterationNumber()}")\n'
                                        'print(f"penalty kappa            = '
                                        '{kappa_val:.4e}")\n'
                                        'print(f"lowest point of contact face = '
                                        '{uy_bot:.6e}   rigid plane at {-g0:.6e}")\n'
                                        'print(f"max penetration          = {pen:.4e}  '
                                        '({pen / h:.3e} of an element edge)")\n'
                                        'print(f"total contact force      = '
                                        '{Rc:.6e}")\n'
                                        'print(f"reaction on loaded face  = '
                                        '{Rtop:.6e}")\n'
                                        'print(f"vertical equilibrium |Rc+Rtop|/|Rc| = '
                                        '{abs(Rc + Rtop) / abs(Rc):.3e}")\n'
                                        'print(f"contact force is compressive (>0): '
                                        '{Rc > 0}")\n'
                                        'print(f"penetration under 1% of an element '
                                        'edge: {pen / h < 1e-2}")\n'
                                        'print(f"all finite: '
                                        '{bool(np.all(np.isfinite(u.x.array)))}")\n',
             'function_space': {'REQUIRED': 'Solid contact (vector displacement):\n'
                                            '    gdim = msh.geometry.dim\n'
                                            '    V = fem.functionspace(msh, '
                                            '("Lagrange", 1, (gdim,)))\n'
                                            'Scalar obstacle problem:\n'
                                            '    V = fem.functionspace(msh, '
                                            '("Lagrange", 1))\n'
                                            'The name is fem.functionspace, all lower '
                                            'case.',
                                'OPTIONAL': 'Degree 1 or 2 both work. Cell types '
                                            'triangle, quadrilateral, tetrahedron all '
                                            'work; the penalty behaviour is the same '
                                            'in 2D and 3D. There is no separate space '
                                            'for the contact pressure in the penalty '
                                            'method - the pressure is a UFL expression '
                                            'in u, not an unknown.',
                                'explanation': 'The penalty method adds no new '
                                               'unknowns, so the space is just the '
                                               'ordinary displacement (or scalar) '
                                               'space. A Lagrange-multiplier '
                                               'formulation would need a second space '
                                               'on the contact surface, which DOLFINx '
                                               'cannot assemble into a single problem '
                                               'without an extension package.',
                                'pitfalls': ['Write fem.functionspace, not '
                                             'fem.FunctionSpace. Signal: TypeError: '
                                             'FunctionSpace.__init__() missing 1 '
                                             "required positional argument: 'cppV'"]},
             'weak_form': {'REQUIRED': 'Define the signed gap so that a NEGATIVE gap '
                                       'means penetration, then add a one-sided '
                                       'penalty pressure and subtract its virtual work '
                                       'on the contact surface measure:\n'
                                       '    gap = u[1] + '
                                       'g0                                # rigid '
                                       'plane below the body\n'
                                       '    pressure = kappa * ufl.max_value(-gap, '
                                       '0.0)    # zero when not in contact\n'
                                       '    F = ufl.inner(sigma(u), '
                                       'ufl.sym(ufl.grad(v))) * ufl.dx \\\n'
                                       '        - pressure * v[1] * ds(1)\n'
                                       'For the scalar obstacle problem u >= phi:\n'
                                       '    F = ufl.inner(ufl.grad(u), ufl.grad(v)) * '
                                       'ufl.dx - f * v * ufl.dx \\\n'
                                       '        - kappa * ufl.max_value(phi - u, 0.0) '
                                       '* v * ufl.dx\n'
                                       'F is a residual form, not a bilinear/linear '
                                       'pair. Do NOT form the Jacobian by hand: '
                                       'NonlinearProblem calls ufl.derivative(F, u) '
                                       'for you, and it differentiates ufl.max_value '
                                       'correctly.',
                           'OPTIONAL': 'ufl.max_value(x, 0.0) and '
                                       'ufl.conditional(ufl.gt(x, 0), x, 0) are '
                                       'interchangeable here. The penalty may be '
                                       'integrated on a facet measure ds(tag) (surface '
                                       'contact, the usual case) or on ufl.dx (volume '
                                       'obstacle constraint). The second argument of '
                                       'max_value may be the int 0 or the float 0.0.',
                           'explanation': 'The penalty term is the derivative of the '
                                          'energy 0.5*kappa*max(0, -gap)**2, so it '
                                          'produces a contact pressure that is exactly '
                                          'zero while the surfaces are apart and grows '
                                          'linearly with the penetration once they '
                                          'touch. Because max_value is not smooth, the '
                                          'problem is nonlinear even for linear '
                                          'elasticity.',
                           'pitfalls': ['Subtract the penalty virtual work, do not add '
                                        'it. Signal: with the sign flipped SNES never '
                                        'converges - the monitor shows the function '
                                        'norm flipping between the same two values '
                                        "over and over until 'Nonlinear <prefix> solve "
                                        'did not converge due to DIVERGED_MAX_IT '
                                        "iterations 50'.",
                                        'Restrict the penalty to a tagged facet '
                                        'measure ds(1), not plain ufl.ds. Signal: with '
                                        'plain ufl.ds the solve still reports '
                                        'CONVERGED_FNORM_ABS, but the contact-force '
                                        'integral comes out three orders of magnitude '
                                        'too large and the largest displacement '
                                        'exceeds the displacement you prescribed.']},
             'boundary_conditions': {'REQUIRED': 'Tag the candidate contact surface '
                                                 'and build a measure for it:\n'
                                                 '    fdim = msh.topology.dim - 1\n'
                                                 '    bot = '
                                                 'mesh.locate_entities_boundary(msh, '
                                                 'fdim, lambda x: np.isclose(x[1], '
                                                 '0.0))\n'
                                                 '    mt = mesh.meshtags(msh, fdim, '
                                                 'np.sort(bot), np.full(len(bot), 1, '
                                                 'dtype=np.int32))\n'
                                                 '    ds = ufl.Measure("ds", '
                                                 'domain=msh, subdomain_data=mt)\n'
                                                 'Dirichlet data for the driving '
                                                 'face:\n'
                                                 '    uD = fem.Function(V)\n'
                                                 '    uD.sub(1).interpolate(lambda x: '
                                                 'np.full_like(x[0], -delta))\n'
                                                 '    bc = fem.dirichletbc(uD, '
                                                 'fem.locate_dofs_topological(V, fdim, '
                                                 'top))\n'
                                                 'At least one Dirichlet condition is '
                                                 'REQUIRED: the penalty term alone '
                                                 'does not remove rigid-body motion.',
                                     'OPTIONAL': 'The contact surface may be driven by '
                                                 'a prescribed displacement (as above) '
                                                 'or by a traction term on another '
                                                 'facet tag. A displacement-driven '
                                                 'setup is safer, because a body held '
                                                 'only by contact still has rigid-body '
                                                 'modes. mesh.meshtags requires the '
                                                 'entity array to be sorted and '
                                                 'duplicate-free (documented '
                                                 'precondition; only debug builds '
                                                 "check it and raise 'MeshTag data is "
                                                 "not sorted'). Release builds accept "
                                                 'an unsorted array silently, but '
                                                 'assembly assumes the order when it '
                                                 'strips ghost entities, so unsorted '
                                                 'tags that include ghosts give wrong '
                                                 'integrals in parallel. Always pass '
                                                 'np.sort(...) (or reorder the values '
                                                 'with np.argsort).',
                                     'explanation': 'The contact constraint itself is '
                                                    'NOT a Dirichlet condition - it '
                                                    'lives in the residual. Dirichlet '
                                                    'conditions are only used to load '
                                                    'the body and to remove rigid-body '
                                                    'motion.',
                                     'pitfalls': ['Constrain enough of the body to '
                                                  'remove rigid-body motion. Signal: '
                                                  'with the body held only by the '
                                                  'penalty term SNES reports '
                                                  'DIVERGED_MAX_IT and the '
                                                  'displacement array drifts off to '
                                                  'order 1e14 instead of staying the '
                                                  'order of the prescribed load.']},
             'solver': {'REQUIRED': '    from dolfinx.fem.petsc import '
                                    'NonlinearProblem\n'
                                    '    problem = NonlinearProblem(\n'
                                    '        F, u, bcs=[bc], '
                                    'petsc_options_prefix="contact_",\n'
                                    '        petsc_options={"snes_type": "newtonls",\n'
                                    '                       "snes_linesearch_type": '
                                    '"l2",\n'
                                    '                       "snes_rtol": 1e-10, '
                                    '"snes_atol": 1e-12,\n'
                                    '                       "snes_max_it": 50,\n'
                                    '                       "ksp_type": "preonly", '
                                    '"pc_type": "lu"})\n'
                                    '    problem.solve()\n'
                                    '    assert problem.solver.getConvergedReason() > '
                                    '0\n'
                                    'petsc_options_prefix is a REQUIRED keyword-only '
                                    'argument. problem.solve() returns the solution '
                                    'Function; it does NOT return a convergence flag, '
                                    'so you MUST query problem.solver (a petsc4py '
                                    'SNES) yourself.',
                        'OPTIONAL': 'snes_linesearch_type may be "l2" (most robust '
                                    'here), "basic" (fewest iterations at moderate '
                                    'penalty) or "bt". Do NOT use "cp". pc_type "lu" '
                                    'for small problems, or an iterative KSP for large '
                                    'ones. snes_divergence_tolerance may be raised '
                                    'when a very stiff penalty is unavoidable.',
                        'explanation': 'Even linear elasticity becomes nonlinear once '
                                       'the one-sided penalty is added, because the '
                                       'active contact set is unknown. Newton with a '
                                       'line search finds the active set; each '
                                       'iteration solves an ordinary linear elasticity '
                                       'system.',
                        'pitfalls': ['Always check problem.solver.getConvergedReason() '
                                     '> 0. Signal: problem.solve() returns a Function '
                                     'and raises nothing even when SNES reports '
                                     'DIVERGED_DTOL, so a silently wrong solution is '
                                     'returned.',
                                     'Do not use snes_linesearch_type "cp". Signal: '
                                     'with "cp" the obstacle problem stalls, SNES '
                                     'reports DIVERGED_MAX_IT (getConvergedReason() == '
                                     '-5) after hitting the iteration limit, and the '
                                     'computed minimum stops an order of magnitude '
                                     'short of the obstacle, so no contact is detected '
                                     'at all; the same script with "l2" or "basic" '
                                     'converges.',
                                     'Do not wrap this problem in '
                                     'dolfinx.nls.petsc.NewtonSolver. Signal: '
                                     "AttributeError: 'NonlinearProblem' object has no "
                                     "attribute 'a'"]},
             'scalar_obstacle_example': '# Second complete runnable example: the '
                                        'scalar obstacle problem.\n'
                                        '# Find u with -laplacian(u) = f subject to u '
                                        '>= phi, by penalty.\n'
                                        'import numpy as np\n'
                                        'import ufl\n'
                                        'from mpi4py import MPI\n'
                                        'from dolfinx import mesh, fem\n'
                                        'from dolfinx.fem.petsc import '
                                        'NonlinearProblem\n'
                                        '\n'
                                        'N = 32\n'
                                        'msh = mesh.create_unit_square(MPI.COMM_WORLD, '
                                        'N, N, mesh.CellType.triangle)\n'
                                        'V = fem.functionspace(msh, ("Lagrange", 1))\n'
                                        'u, v = fem.Function(V, name="u"), '
                                        'ufl.TestFunction(V)\n'
                                        '\n'
                                        'f = fem.Constant(msh, -10.0)          # '
                                        'downward load\n'
                                        'phi = fem.Constant(msh, -0.2)         # rigid '
                                        'obstacle: constraint u >= phi\n'
                                        'kappa = fem.Constant(msh, 1.0e5)      # '
                                        'penalty stiffness\n'
                                        'gap = phi - u                         # gap > '
                                        '0 means the constraint is violated\n'
                                        '\n'
                                        'F = (ufl.inner(ufl.grad(u), ufl.grad(v)) * '
                                        'ufl.dx - f * v * ufl.dx\n'
                                        '     - kappa * ufl.max_value(gap, 0.0) * v * '
                                        'ufl.dx)\n'
                                        '\n'
                                        'msh.topology.create_connectivity(msh.topology.dim '
                                        '- 1, msh.topology.dim)\n'
                                        'bdofs = fem.locate_dofs_topological(V, '
                                        'msh.topology.dim - 1,\n'
                                        '                                    '
                                        'mesh.exterior_facet_indices(msh.topology))\n'
                                        'bc = fem.dirichletbc(fem.Constant(msh, 0.0), '
                                        'bdofs, V)\n'
                                        '\n'
                                        'problem = NonlinearProblem(\n'
                                        '    F, u, bcs=[bc], '
                                        'petsc_options_prefix="obstacle_",\n'
                                        '    petsc_options={"snes_type": "newtonls", '
                                        '"snes_linesearch_type": "l2",\n'
                                        '                   "snes_rtol": 1e-10, '
                                        '"snes_atol": 1e-12, "snes_max_it": 60,\n'
                                        '                   "ksp_type": "preonly", '
                                        '"pc_type": "lu"})\n'
                                        'problem.solve()\n'
                                        'reason = problem.solver.getConvergedReason()\n'
                                        'assert reason > 0, f"SNES did not converge, '
                                        'reason {reason}"\n'
                                        '\n'
                                        'comm = msh.comm\n'
                                        'umin = comm.allreduce(float(u.x.array.min()), '
                                        'MPI.MIN)\n'
                                        'pen = max(0.0, float(phi.value) - umin)\n'
                                        'contact_force = comm.allreduce(\n'
                                        '    fem.assemble_scalar(fem.form(kappa * '
                                        'ufl.max_value(gap, 0.0) * ufl.dx)), MPI.SUM)\n'
                                        'load = '
                                        'comm.allreduce(fem.assemble_scalar(fem.form(-f '
                                        '* ufl.dx)), MPI.SUM)\n'
                                        '\n'
                                        'print(f"SNES reason {reason}, iterations '
                                        '{problem.solver.getIterationNumber()}")\n'
                                        'print(f"min(u)              = {umin:.6e}   '
                                        'obstacle at {float(phi.value):.6e}")\n'
                                        'print(f"max violation       = {pen:.4e}   '
                                        '({pen / (1.0 / N):.3e} of an element edge)")\n'
                                        'print(f"total contact force = '
                                        '{contact_force:.6e}  (must be >= 0)")\n'
                                        'print(f"applied load        = {load:.6e}")\n'
                                        'print(f"contact force does not exceed the '
                                        'applied load: {contact_force <= load}")\n'
                                        'print(f"constraint violated by less than 1% '
                                        'of |phi|: {pen / abs(float(phi.value)) < '
                                        '1e-2}")\n'
                                        'print(f"all finite: '
                                        '{bool(np.all(np.isfinite(u.x.array)))}")\n',
             'penalty_parameter': {'REQUIRED': 'For solid contact start from\n'
                                               '    kappa = 1.0e2 * E / h\n'
                                               'with E the Young modulus of the softer '
                                               'body and h the element size on the '
                                               'contact surface. This value was '
                                               'measured on the example above to leave '
                                               'a penetration far below one percent of '
                                               'an element edge while Newton still '
                                               'converges in a handful of iterations.',
                                   'OPTIONAL': 'Anything from about 10 * E / h to 1e6 '
                                               '* E / h works on the example above; '
                                               'larger penalty means smaller '
                                               'penetration and more Newton '
                                               'iterations, until Newton stops '
                                               'converging at all. Always report the '
                                               'penetration together with the result.',
                                   'explanation': 'The penalty replaces the exact '
                                                  'inequality constraint by a stiff '
                                                  'spring, so the answer always '
                                                  'penetrates a little; the '
                                                  'penetration falls roughly in '
                                                  'inverse proportion to kappa until '
                                                  'the nonlinear solver breaks down.',
                                   'pitfalls': ['A too-small penalty does NOT produce '
                                                'an error message - SNES reports '
                                                'success. Signal: SNES reports '
                                                'CONVERGED_FNORM_ABS while the contact '
                                                'surface sits well below the rigid '
                                                'plane.',
                                                'A too-large penalty makes SNES report '
                                                'DIVERGED_DTOL, not any '
                                                'condition-number message. Signal: '
                                                "'Nonlinear <prefix> solve did not "
                                                'converge due to DIVERGED_DTOL '
                                                "iterations N' (printed when "
                                                'snes_converged_reason is set), where '
                                                'N depends on the line search: 1 with '
                                                "snes_linesearch_type 'basic', 6 with "
                                                "the example's 'l2' (kappa = 1e8 * E / "
                                                'h).']},
             'contact_search': {'REQUIRED': 'For a rigid obstacle described '
                                            'analytically (a plane, a sphere) no '
                                            'search is needed at all: write the gap '
                                            'directly as a UFL expression in u and the '
                                            'spatial coordinate, as in the example '
                                            'above. Only body-to-body contact needs a '
                                            'geometric search.',
                                'OPTIONAL': 'The real DOLFINx 0.10 search API is:\n'
                                            '    tree  = dolfinx.geometry.bb_tree(msh, '
                                            'dim, padding=0.0, entities=None)\n'
                                            '    ftree = dolfinx.geometry.bb_tree(msh, '
                                            'msh.topology.dim - 1, entities=facets)\n'
                                            '    mtree = '
                                            'dolfinx.geometry.create_midpoint_tree(msh, '
                                            'msh.topology.dim - 1, facets)\n'
                                            '    near  = '
                                            'dolfinx.geometry.compute_closest_entity(ftree, '
                                            'mtree, msh, points)\n'
                                            '    cand  = '
                                            'dolfinx.geometry.compute_collisions_points(tree, '
                                            'points)\n'
                                            '    cells = '
                                            'dolfinx.geometry.compute_colliding_cells(msh, '
                                            'cand, points)\n'
                                            'compute_collisions_points and '
                                            'compute_colliding_cells return a '
                                            'dolfinx.graph.AdjacencyList; use '
                                            '.links(i) per point. '
                                            'dolfinx.geometry.squared_distance and '
                                            'compute_distance_gjk also exist.',
                                'explanation': 'bb_tree(msh, dim) builds a '
                                               'bounding-box hierarchy over mesh '
                                               'entities of that dimension; passing '
                                               'entities=facets restricts it to the '
                                               'contact surface, which is what a '
                                               'contact search wants.',
                                'pitfalls': ['The function is '
                                             'compute_collisions_points, not '
                                             'compute_collisions. Signal: '
                                             'AttributeError: module '
                                             "'dolfinx.geometry' has no attribute "
                                             "'compute_collisions'",
                                             'A bounding-box tree is not a free win at '
                                             'small surface sizes; measure before '
                                             'adopting it. Signal: for a few hundred '
                                             'surface facets a plain NumPy all-pairs '
                                             'distance computation is faster than the '
                                             'tree.']},
             'other_formulations': {'penalty': 'What the example above does. No extra '
                                               'unknowns, one parameter to choose, the '
                                               'constraint is only approximately '
                                               'satisfied. This is the only '
                                               'formulation verified to run on this '
                                               'installation.',
                                    'nitsche': 'Variationally consistent weak '
                                               'enforcement; needs the traction '
                                               'sigma(u).n on the contact surface plus '
                                               'a stabilisation term, so it is written '
                                               'the same way as the penalty form with '
                                               'extra terms. Not verified here.',
                                    'lagrange_multiplier': 'Introduces the contact '
                                                           'pressure as a real unknown '
                                                           'on the contact surface. '
                                                           'This needs a mixed space '
                                                           'whose second component '
                                                           'lives only on a surface, '
                                                           'which DOLFINx cannot '
                                                           'assemble in one block '
                                                           'without an extension. Not '
                                                           'verified here.',
                                    'extension_packages': 'A dolfinx_contact extension '
                                                          'package exists as a '
                                                          'separate project, but it is '
                                                          'NOT part of DOLFINx and '
                                                          'must not be assumed '
                                                          'present: import '
                                                          'dolfinx_contact raises '
                                                          'ModuleNotFoundError on a '
                                                          'standard DOLFINx '
                                                          'installation.'},
             'pitfalls': ['[API] There is no contact object of any kind in DOLFINx: no '
                          'solver, no contact boundary, no gap function. Build the '
                          'penalty residual by hand as in the example. Signal: '
                          'dolfinx.fem has no attribute named after contact - '
                          '`dolfinx.fem.ContactBoundary` raises "AttributeError: '
                          "module 'dolfinx.fem' has no attribute "
                          '\'ContactBoundary\'" and `dolfinx.fem.ContactProblem` '
                          'raises "AttributeError: module \'dolfinx.fem\' has no '
                          'attribute \'ContactProblem\'"; filtering dir(dolfinx), '
                          'dir(dolfinx.fem) and dir(dolfinx.fem.petsc) for names '
                          "containing 'ontact' returns the empty list [] in every "
                          'case. Do not fall back on the separate dolfinx_contact '
                          'package either unless you have checked it is installed: '
                          '`import dolfinx_contact` raises "ModuleNotFoundError: No '
                          'module named \'dolfinx_contact\'" on a standard DOLFINx '
                          'installation. Verified on DOLFINx 0.10.0.',
                          '[Numerical] A penalty stiffness that is too small gives a '
                          'WRONG answer with NO error message at all - this is the '
                          'most dangerous failure mode of penalty contact. Signal: '
                          "SNES reports success, printing 'Nonlinear <prefix> solve "
                          "converged due to CONVERGED_FNORM_ABS iterations 3' and "
                          'getConvergedReason() returns 2, while the body has moved '
                          'straight through the obstacle - on the scalar obstacle '
                          'problem with a badly undersized penalty the solution '
                          'minimum sits below the obstacle by more than ten element '
                          'edges, and on the elastic block the contact face ends up a '
                          'third of an element edge below the rigid plane. There is no '
                          'warning of any kind. You MUST compute the penetration '
                          'yourself after every solve and check it against the element '
                          'size; a converged reason is not evidence that the '
                          'constraint holds. Verified in 2D and 3D, on triangles, '
                          'quadrilaterals and tetrahedra, at degree 1 and degree 2.',
                          '[Numerical] A penalty stiffness that is too large makes '
                          'SNES diverge, and the reason it reports is a '
                          'divergence-tolerance trip, NOT any condition-number '
                          "message. The previously quoted signal 'PETSc "
                          "condition-number warning > 1e14' does NOT exist: PETSc "
                          'prints no condition-number warning by default, and with '
                          'every monitor enabled the only lines emitted are the SNES '
                          'function norms and the converged-reason line. Signal: what '
                          "actually appears is 'Nonlinear <prefix> solve did not "
                          "converge due to DIVERGED_DTOL iterations N' with "
                          'getConvergedReason() returning -9. N is 1 with '
                          "snes_linesearch_type 'basic', where the residual after the "
                          'first Newton step grows in direct proportion to the '
                          "penalty; with the 'l2' line search of the example the "
                          'first five iterations do not depend on the penalty and it '
                          'trips at iteration 6 (measured at 1e8, 1e10 and 1e12 * E / '
                          "h). With snes_linesearch_type 'bt' it is instead "
                          'DIVERGED_LINE_SEARCH with reason -6. The linear solve '
                          'inside is perfectly healthy - the KSP monitor reports '
                          "'Linear <prefix> solve converged due to CONVERGED_ITS "
                          "iterations 1' in the same run. With 'basic' and 'l2' the "
                          'residual crosses the default snes_divergence_tolerance of '
                          '10000.0 (relative to the initial residual) long before '
                          'conditioning is a real problem. Verified in 2D and 3D, on '
                          'triangles, quadrilaterals and tetrahedra, at degree 1 and '
                          'degree 2; the exact penalty value at which it trips '
                          'depends on the line search.',
                          '[Numerical] If a very stiff penalty is genuinely needed, do '
                          'not just raise it and hope - use penalty continuation, i.e. '
                          'solve with a modest penalty first and then re-solve with '
                          'the penalty increased decade by decade, reusing the '
                          'previous solution as the starting point (assign the new '
                          'value to a fem.Constant and call problem.solve() again). '
                          'Signal: cold-started at a very stiff penalty the same '
                          'problem reports DIVERGED_DTOL after a single iteration, '
                          'while continued from the previous decade every solve '
                          'reports CONVERGED_FNORM_RELATIVE or '
                          'CONVERGED_SNORM_RELATIVE in one or two Newton iterations '
                          'and the penetration keeps shrinking. Setting '
                          'snes_divergence_tolerance to a very large number is an '
                          'alternative that also works but needs many more Newton '
                          'iterations.',
                          '[API] The DOLFINx 0.10 geometry search functions are not '
                          'named the way older tutorials name them. '
                          'dolfinx.geometry.bb_tree DOES exist with exactly that name '
                          'and the signature bb_tree(mesh, dim, *, padding=0.0, '
                          'entities=None) returning a '
                          'dolfinx.geometry.BoundingBoxTree, and the entities keyword '
                          'is what you use to restrict the tree to contact facets. But '
                          'the collision query is compute_collisions_points, not '
                          'compute_collisions. Signal: "AttributeError: module '
                          "'dolfinx.geometry' has no attribute "
                          '\'compute_collisions\'". The working set of names on this '
                          'release is bb_tree, BoundingBoxTree, create_midpoint_tree, '
                          'compute_collisions_points, compute_collisions_trees, '
                          'compute_colliding_cells, compute_closest_entity, '
                          'squared_distance, compute_distance_gjk and '
                          'determine_point_ownership.',
                          '[Performance] A naive all-pairs gap search really is '
                          'quadratic, but a bounding-box tree is a much smaller win '
                          'than usually claimed, so measure instead of assuming. '
                          'Signal: doubling the number of contact-surface facets '
                          'multiplies the wall time of a NumPy all-pairs distance '
                          'computation by about four, as expected for quadratic cost - '
                          'but the same doubling multiplies '
                          'dolfinx.geometry.compute_closest_entity by very nearly as '
                          'much, and up to several thousand surface facets the tree '
                          'version is only about twice as fast; for a few hundred '
                          'facets the plain NumPy version is the faster of the two. '
                          'The claim that a bounding-box tree makes contact search '
                          'near-linear was NOT reproduced on this installation.',
                          '[Integration] problem.solve() on a NonlinearProblem never '
                          'raises on non-convergence, so an unchecked contact solve '
                          'silently returns a body that has passed through the '
                          'obstacle. Signal: the identical script prints a plausible '
                          'displacement field whether SNES returned 2 '
                          '(CONVERGED_FNORM_ABS) or -9 (DIVERGED_DTOL); only '
                          'problem.solver.getConvergedReason() distinguishes them. '
                          'Assert on it, and additionally check global equilibrium - '
                          'the integral of the penalty pressure over the contact '
                          'surface must balance the reaction on the loaded face, which '
                          'it does to machine precision for a good penalty and fails '
                          'by order one when the solve has diverged.']},
 'fracture': {'description': 'Phase-field (variational) fracture in FEniCSx. A crack '
                             'is represented by a smooth scalar damage field d in '
                             '[0,1] that degrades the elastic stiffness, so cracks '
                             'nucleate and grow without remeshing. DOLFINx has no '
                             'phase-field solver of its own: you build the two coupled '
                             'problems yourself and alternate between them (a '
                             'staggered scheme).',
              'minimal_working_example': 'import numpy as np\n'
                                         'import ufl\n'
                                         'from mpi4py import MPI\n'
                                         'from dolfinx import mesh, fem\n'
                                         'from dolfinx.fem.petsc import LinearProblem, '
                                         'NonlinearProblem\n'
                                         '\n'
                                         'E, nu = 210.0, 0.3\n'
                                         'mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + '
                                         'nu) * (1 - 2 * nu))\n'
                                         'Gc, l0, k_res = 2.7e-3, 0.04, 1e-6\n'
                                         'nx, nsteps, umax = 32, 8, 8.0e-3\n'
                                         '\n'
                                         'msh = '
                                         'mesh.create_unit_square(MPI.COMM_WORLD, nx, '
                                         'nx, mesh.CellType.triangle)\n'
                                         'tdim = msh.topology.dim\n'
                                         'V = fem.functionspace(msh, ("Lagrange", 1, '
                                         '(2,)))   # displacement\n'
                                         'W = fem.functionspace(msh, ("Lagrange", '
                                         '1))         # damage\n'
                                         'Q = fem.functionspace(msh, ("DG", '
                                         '0))               # history (cell-wise)\n'
                                         '\n'
                                         'u, v = fem.Function(V, name="u"), '
                                         'ufl.TestFunction(V)\n'
                                         'd, q = fem.Function(W, name="d"), '
                                         'ufl.TestFunction(W)\n'
                                         'dtr = ufl.TrialFunction(W)\n'
                                         'd_it, d_prev, H, Hn = (fem.Function(W), '
                                         'fem.Function(W),\n'
                                         '                       fem.Function(Q), '
                                         'fem.Function(Q))\n'
                                         '\n'
                                         'e = '
                                         'ufl.sym(ufl.grad(u))                            '
                                         '# spectral tension/compression split\n'
                                         'm = 0.5 * (e[0, 0] + e[1, 1])\n'
                                         'rad = ufl.sqrt(0.25 * (e[0, 0] - e[1, 1]) ** '
                                         '2 + e[0, 1] ** 2 + 1e-30)\n'
                                         'ev = [m + rad, m - rad]\n'
                                         'psi_p = 0.5 * lam * ufl.max_value(ufl.tr(e), '
                                         '0) ** 2 \\\n'
                                         '    + mu * sum(ufl.max_value(l, 0) ** 2 for '
                                         'l in ev)\n'
                                         'psi_m = 0.5 * lam * ufl.min_value(ufl.tr(e), '
                                         '0) ** 2 \\\n'
                                         '    + mu * sum(ufl.min_value(l, 0) ** 2 for '
                                         'l in ev)\n'
                                         'Fu = ufl.derivative((((1 - d) ** 2 + k_res) '
                                         '* psi_p + psi_m) * ufl.dx, u, v)\n'
                                         'a_d = ((2 * H + Gc / l0) * dtr * q\n'
                                         '       + Gc * l0 * ufl.inner(ufl.grad(dtr), '
                                         'ufl.grad(q))) * ufl.dx\n'
                                         'L_d = 2 * H * q * ufl.dx\n'
                                         '\n'
                                         'nc = msh.topology.index_map(tdim).size_local '
                                         '+ msh.topology.index_map(tdim).num_ghosts\n'
                                         'cid = np.arange(nc, dtype=np.int32)\n'
                                         'mp = mesh.compute_midpoints(msh, tdim, cid)\n'
                                         'msh.topology.create_connectivity(tdim, '
                                         'tdim)\n'
                                         'notch = cid[(mp[:, 0] <= 0.5) & '
                                         '(np.abs(mp[:, 1] - 0.5) < 0.5 / nx)]\n'
                                         'H.x.array[fem.locate_dofs_topological(Q, '
                                         'tdim, notch)] = 1.0e3   # pre-crack\n'
                                         '\n'
                                         'fdim = tdim - 1\n'
                                         'bot = mesh.locate_entities_boundary(msh, '
                                         'fdim, lambda x: np.isclose(x[1], 0.0))\n'
                                         'top = mesh.locate_entities_boundary(msh, '
                                         'fdim, lambda x: np.isclose(x[1], 1.0))\n'
                                         'zero, uD = fem.Function(V), fem.Function(V)\n'
                                         'bcs = [fem.dirichletbc(zero, '
                                         'fem.locate_dofs_topological(V, fdim, bot)),\n'
                                         '       fem.dirichletbc(uD, '
                                         'fem.locate_dofs_topological(V, fdim, top))]\n'
                                         '\n'
                                         'pu = NonlinearProblem(Fu, u, bcs=bcs, '
                                         'petsc_options_prefix="pfu_",\n'
                                         '                      '
                                         'petsc_options={"snes_type": "newtonls",\n'
                                         '                                     '
                                         '"snes_linesearch_type": "basic",\n'
                                         '                                     '
                                         '"snes_rtol": 1e-8, "snes_atol": 1e-12,\n'
                                         '                                     '
                                         '"snes_max_it": 60,\n'
                                         '                                     '
                                         '"ksp_type": "preonly", "pc_type": "lu"})\n'
                                         'pd = LinearProblem(a_d, L_d, u=d, '
                                         'petsc_options_prefix="pfd_",\n'
                                         '                   '
                                         'petsc_options={"ksp_type": "preonly", '
                                         '"pc_type": "lu"})\n'
                                         'psiE = fem.Expression(psi_p, '
                                         'Q.element.interpolation_points)\n'
                                         'Es_form = fem.form(Gc / (2 * l0) * (d ** 2 + '
                                         'l0 ** 2\n'
                                         '                                    * '
                                         'ufl.inner(ufl.grad(d), ufl.grad(d))) * '
                                         'ufl.dx)\n'
                                         'comm, worst_drop, dmax_hist, Es_hist = '
                                         'msh.comm, 0.0, [], []\n'
                                         '\n'
                                         'for step in range(1, nsteps + 1):\n'
                                         '    uD.x.array[:] = 0.0\n'
                                         '    uD.sub(1).interpolate(lambda x: '
                                         'np.full_like(x[0], umax * step / nsteps))\n'
                                         '    d_prev.x.array[:] = d.x.array\n'
                                         '    for it in range(250):\n'
                                         '        d_it.x.array[:] = d.x.array\n'
                                         '        pu.solve()\n'
                                         '        assert '
                                         'pu.solver.getConvergedReason() > 0, "u-solve '
                                         'diverged"\n'
                                         '        Hn.interpolate(psiE)\n'
                                         '        H.x.array[:] = np.maximum(H.x.array, '
                                         'Hn.x.array)    # irreversibility\n'
                                         '        pd.solve()\n'
                                         '        if '
                                         'comm.allreduce(np.max(np.abs(d.x.array - '
                                         'd_it.x.array)), MPI.MAX) < 1e-3:\n'
                                         '            break\n'
                                         '    worst_drop = min(worst_drop, '
                                         'comm.allreduce(\n'
                                         '        float(np.min(d.x.array - '
                                         'd_prev.x.array)), MPI.MIN))\n'
                                         '    '
                                         'dmax_hist.append(comm.allreduce(float(d.x.array.max()), '
                                         'MPI.MAX))\n'
                                         '    '
                                         'Es_hist.append(comm.allreduce(fem.assemble_scalar(Es_form), '
                                         'MPI.SUM))\n'
                                         '    print("step %2d  u_top=%.4e  '
                                         'staggered=%3d  max(d)=%.6f  surface '
                                         'energy=%.6e"\n'
                                         '          % (step, umax * step / nsteps, it '
                                         '+ 1, dmax_hist[-1], Es_hist[-1]))\n'
                                         '\n'
                                         'print("crack grew (surface energy up):", '
                                         'Es_hist[-1] > Es_hist[0])\n'
                                         'print("surface energy grew monotonically:", '
                                         'all(np.diff(Es_hist) > -1e-14))\n'
                                         'print("worst per-step damage decrease: %.3e '
                                         '(irreversible if > -1e-3)" % worst_drop)\n'
                                         'print("irreversibility satisfied:", '
                                         'worst_drop > -1e-3)\n'
                                         'print("damage non-negative:", '
                                         'comm.allreduce(float(d.x.array.min()), '
                                         'MPI.MIN) >= 0.0)\n'
                                         'print("final max(d) = %.6f" % '
                                         'dmax_hist[-1])\n'
                                         'print("all finite:", '
                                         'bool(np.all(np.isfinite(d.x.array)))\n'
                                         '      and '
                                         'bool(np.all(np.isfinite(u.x.array))))\n',
              'function_space': {'REQUIRED': 'THREE spaces on the SAME mesh:\n'
                                             '    V = fem.functionspace(msh, '
                                             '("Lagrange", 1, (2,)))  # displacement '
                                             '(vector)\n'
                                             '    W = fem.functionspace(msh, '
                                             '("Lagrange", 1))        # damage d '
                                             '(scalar)\n'
                                             '    Q = fem.functionspace(msh, ("DG", '
                                             '0))              # history field H\n'
                                             'Use (3,) instead of (2,) for the '
                                             'displacement block in 3D. The history '
                                             'field H is REQUIRED - it is what makes '
                                             'the crack irreversible - and a cell-wise '
                                             'DG 0 space is the simplest correct '
                                             'choice for it because the driving energy '
                                             'is discontinuous across elements.',
                                 'OPTIONAL': 'Degree 1 is the safe default and is what '
                                             'has been verified most widely. Degree 2 '
                                             'works for the displacement and damage, '
                                             'but only with a larger residual '
                                             'stiffness (see model_parameters). '
                                             'Triangles, quadrilaterals and tetrahedra '
                                             'all work. H may also be put in the same '
                                             'Lagrange space as d, at the price of an '
                                             'interpolation that is not uniquely '
                                             'defined at element boundaries.',
                                 'explanation': 'Displacement and damage are separate '
                                                'unknowns solved in alternation, so '
                                                'they get separate spaces rather than '
                                                'one mixed space. The history field is '
                                                'a state variable, not an unknown.',
                                 'pitfalls': ['Write fem.functionspace, all lower '
                                              'case. Signal: TypeError: '
                                              'FunctionSpace.__init__() missing 1 '
                                              "required positional argument: 'cppV'",
                                              'Q.element.interpolation_points is an '
                                              'attribute, not a method. Signal: '
                                              'writing '
                                              'Q.element.interpolation_points() raises '
                                              "TypeError: 'numpy.ndarray' object is "
                                              'not callable']},
              'weak_form': {'REQUIRED': 'Two problems. (1) Displacement, from the '
                                        'degraded energy - build the residual with '
                                        'ufl.derivative, do NOT hand-code the stress:\n'
                                        '    g = (1 - d) ** 2 + '
                                        'k_res                       # degradation '
                                        'function\n'
                                        '    Fu = ufl.derivative((g * psi_p + psi_m) * '
                                        'ufl.dx, u, v)\n'
                                        '(2) Damage, the AT2 equation, which is LINEAR '
                                        'in d once H is known:\n'
                                        '    a_d = ((2 * H + Gc / l0) * dtr * q\n'
                                        '           + Gc * l0 * '
                                        'ufl.inner(ufl.grad(dtr), ufl.grad(q))) * '
                                        'ufl.dx\n'
                                        '    L_d = 2 * H * q * ufl.dx\n'
                                        'psi_p is the tensile part of the elastic '
                                        'energy density and psi_m the compressive '
                                        'part; only psi_p is degraded and only psi_p '
                                        'drives H.',
                            'OPTIONAL': 'AT2 (shown) has the quadratic dissipation '
                                        'd**2 and no elastic threshold. AT1 replaces '
                                        'it with a linear term and needs an extra '
                                        'bound constraint d >= 0. The degradation '
                                        'function may be any monotone g with g(0)=1, '
                                        'g(1)=k_res; (1-d)**2 is the standard choice.',
                            'explanation': 'The displacement problem is nonlinear '
                                           'because the tension-compression split is '
                                           'not a smooth function of the strain, so it '
                                           'needs NonlinearProblem; the damage problem '
                                           'is a plain linear reaction-diffusion '
                                           'problem and should use LinearProblem.',
                            'pitfalls': ['Degrade only the tensile part psi_p, never '
                                         'the total energy. Signal: degrading the '
                                         'total energy makes the model blind to the '
                                         'sign of the load - a compression step and a '
                                         'tension step of the same magnitude produce '
                                         'damage fields that agree to every printed '
                                         'digit.',
                                         'Drive the history field from psi_p, not from '
                                         'the total energy or from d. Signal: driving '
                                         'H with the total energy reproduces the same '
                                         'sign-blind behaviour under compression.']},
              'boundary_conditions': {'REQUIRED': 'Dirichlet conditions on the '
                                                  'DISPLACEMENT only, applied through '
                                                  'a Function whose values you '
                                                  'overwrite each load step:\n'
                                                  '    uD = fem.Function(V)\n'
                                                  '    uD.x.array[:] = 0.0\n'
                                                  '    uD.sub(1).interpolate(lambda x: '
                                                  'np.full_like(x[0], u_top))\n'
                                                  '    bcs = [fem.dirichletbc(zero, '
                                                  'fem.locate_dofs_topological(V, '
                                                  'fdim, bot)),\n'
                                                  '           fem.dirichletbc(uD,   '
                                                  'fem.locate_dofs_topological(V, '
                                                  'fdim, top))]\n'
                                                  'The damage problem needs NO '
                                                  'boundary condition at all - the '
                                                  'gradient term makes it well posed '
                                                  'on its own.\n'
                                                  'An initial crack is imposed by '
                                                  'SEEDING THE HISTORY FIELD, not by a '
                                                  'boundary condition on d:\n'
                                                  '    mp = '
                                                  'mesh.compute_midpoints(msh, tdim, '
                                                  'cid)\n'
                                                  '    notch = cid[(mp[:, 0] <= 0.5) & '
                                                  '(np.abs(mp[:, 1] - 0.5) < 0.5 / '
                                                  'nx)]\n'
                                                  '    '
                                                  'msh.topology.create_connectivity(tdim, '
                                                  'tdim)\n'
                                                  '    '
                                                  'H.x.array[fem.locate_dofs_topological(Q, '
                                                  'tdim, notch)] = 1.0e3',
                                      'OPTIONAL': 'A pre-crack can instead be cut '
                                                  'geometrically into the mesh, or '
                                                  'imposed with a Dirichlet condition '
                                                  'd = 1 on the crack dofs. Seeding H '
                                                  'is the least intrusive and needs no '
                                                  'special mesh.',
                                      'explanation': 'Because H enters the damage '
                                                     'equation as both a reaction '
                                                     'coefficient and the right-hand '
                                                     'side, a large seeded H drives d '
                                                     'to very nearly 1 there, which is '
                                                     'exactly a pre-existing crack.',
                                      'pitfalls': ['Select the notch cells by element '
                                                   'MIDPOINT, not with '
                                                   'mesh.locate_entities and a thin '
                                                   'band. Signal: mesh.locate_entities '
                                                   'only selects a cell when ALL its '
                                                   'vertices satisfy the predicate, so '
                                                   'a band thinner than one element '
                                                   'selects nothing at all and the '
                                                   'notch silently does not exist - '
                                                   'the run then shows damage creeping '
                                                   'up from zero everywhere instead of '
                                                   'a crack starting at the notch tip.',
                                                   'Call '
                                                   'msh.topology.create_connectivity(tdim, '
                                                   'tdim) before '
                                                   'fem.locate_dofs_topological with '
                                                   'entity dimension tdim. Signal: '
                                                   'RuntimeError: Entity-to-cell '
                                                   'connectivity has not been '
                                                   'computed. Missing dims 2->2']},
              'solver': {'REQUIRED': 'A STAGGERED loop: inside each load step, '
                                     'alternate the two problems until d stops '
                                     'changing.\n'
                                     '    pu = NonlinearProblem(Fu, u, bcs=bcs, '
                                     'petsc_options_prefix="pfu_",\n'
                                     '                          petsc_options={...})\n'
                                     '    pd = LinearProblem(a_d, L_d, u=d, '
                                     'petsc_options_prefix="pfd_",\n'
                                     '                       '
                                     'petsc_options={"ksp_type": "preonly",\n'
                                     '                                      "pc_type": '
                                     '"lu"})\n'
                                     '    for it in range(250):\n'
                                     '        d_it.x.array[:] = d.x.array\n'
                                     '        pu.solve()\n'
                                     '        assert pu.solver.getConvergedReason() > '
                                     '0\n'
                                     '        Hn.interpolate(psiE)\n'
                                     '        H.x.array[:] = np.maximum(H.x.array, '
                                     'Hn.x.array)\n'
                                     '        pd.solve()\n'
                                     '        if '
                                     'comm.allreduce(np.max(np.abs(d.x.array - '
                                     'd_it.x.array)),\n'
                                     '                          MPI.MAX) < 1e-3:\n'
                                     '            break\n'
                                     'petsc_options_prefix is REQUIRED on both '
                                     'problems (distinct prefixes are conventional and '
                                     'keep the two option sets apart, though reusing '
                                     'one prefix does not raise). Build both problem '
                                     'objects ONCE, outside the load loop, and '
                                     're-solve them; they pick up the new values of uD '
                                     'and H automatically.',
                         'OPTIONAL': 'The staggered tolerance (1e-3 on max|d - d_prev| '
                                     'above) and the iteration cap may be varied. '
                                     'pc_type lu for small problems. A monolithic '
                                     'Newton on a mixed (u, d) space is the usual '
                                     'alternative; it was NOT verified here, so no '
                                     'claim is made about its iteration counts.',
                         'explanation': 'Each half of the staggered step is a '
                                        'well-behaved problem: given d, the '
                                        'displacement problem is an ordinary nonlinear '
                                        'elasticity solve; given H, the damage problem '
                                        'is linear. The alternation is what converges '
                                        'slowly.',
                         'pitfalls': ['Allow a generous staggered iteration cap. '
                                      'Signal: the count sits in the single digits '
                                      'while the crack is dormant and then jumps by '
                                      'one to two orders of magnitude in the single '
                                      'load step where the crack actually runs; a cap '
                                      'of a few tens silently truncates exactly that '
                                      'step.',
                                      'Create both problem objects outside the load '
                                      'loop. Signal: reconstructing NonlinearProblem '
                                      'before every solve makes a sequence of solves '
                                      'take orders of magnitude longer than reusing '
                                      'one object, because the object construction, '
                                      'not the solve, then dominates the run.']},
              'model_parameters': {'Gc': 'Critical energy release rate, energy per '
                                         'unit crack area.',
                                   'l0': 'Regularisation length. It is a MODEL '
                                         'parameter, not a numerical one: it sets the '
                                         'width of the diffuse crack and, with Gc, the '
                                         'effective strength.',
                                   'k_res': 'Residual stiffness in g(d) = (1-d)**2 + '
                                            'k_res, which keeps the fully broken '
                                            'material from having exactly zero '
                                            'stiffness. REQUIRED and strictly '
                                            'positive. Its usable size depends on the '
                                            'polynomial degree - see the pitfall '
                                            'below.',
                                   'mesh_size': 'The element size h must resolve l0. '
                                                'Use h <= l0/2 as a starting point and '
                                                'refine until the answer stops moving.',
                                   'pitfalls': ['Do not make k_res as small as you '
                                                'can. Signal: at degree 1 the '
                                                'staggered displacement solve is happy '
                                                'with a residual stiffness many orders '
                                                'of magnitude below one, but at degree '
                                                '2 the very same problem stops '
                                                'converging and SNES reports '
                                                'DIVERGED_MAX_IT (getConvergedReason() '
                                                '== -5) on the first solve of the '
                                                'first load step; raising k_res fixes '
                                                'it.']},
              'irreversibility': {'REQUIRED': 'A history field updated with an '
                                              'element-wise maximum after every '
                                              'displacement solve:\n'
                                              '    '
                                              'Hn.interpolate(psiE)                              '
                                              '# psiE = Expression(psi_p, ...)\n'
                                              '    H.x.array[:] = '
                                              'np.maximum(H.x.array, Hn.x.array)  # H '
                                              'can only grow\n'
                                              'Because H can only grow and d grows '
                                              'with H, the crack cannot heal.',
                                  'OPTIONAL': 'The alternative is to solve the damage '
                                              'problem as a bound-constrained problem '
                                              'with d >= d_previous, or to project d = '
                                              'max(d, d_previous) after each solve. '
                                              'The history field is by far the '
                                              'simplest and is what the example uses.',
                                  'explanation': 'Without it, the damage equation is '
                                                 'driven by the INSTANTANEOUS elastic '
                                                 'energy, which falls when the load '
                                                 'falls, so the damage falls with it '
                                                 'and the crack closes up again.',
                                  'pitfalls': ['Omitting the history update is not a '
                                               'small error. Signal: on a '
                                               'load-then-unload path, individual '
                                               'nodes fall from fully damaged back to '
                                               'nearly undamaged within one load step '
                                               'and the total damage integrated over '
                                               'the body drops by roughly forty '
                                               'percent by the time the load is back '
                                               'to zero; with the history field the '
                                               'same integral does not fall at all.']},
              'tension_compression_split': {'REQUIRED': 'Use the SPECTRAL (Miehe) '
                                                        'split, which decomposes the '
                                                        'strain by its eigenvalues. '
                                                        'Closed form in 2D:\n'
                                                        '    e = ufl.sym(ufl.grad(u))\n'
                                                        '    m = 0.5 * (e[0, 0] + e[1, '
                                                        '1])\n'
                                                        '    rad = ufl.sqrt(0.25 * '
                                                        '(e[0, 0] - e[1, 1]) ** 2 + '
                                                        'e[0, 1] ** 2 + 1e-30)\n'
                                                        '    ev = [m + rad, m - rad]\n'
                                                        '    psi_p = 0.5 * lam * '
                                                        'ufl.max_value(ufl.tr(e), 0) '
                                                        '** 2 \\\n'
                                                        '        + mu * '
                                                        'sum(ufl.max_value(l, 0) ** 2 '
                                                        'for l in ev)\n'
                                                        '    psi_m = 0.5 * lam * '
                                                        'ufl.min_value(ufl.tr(e), 0) '
                                                        '** 2 \\\n'
                                                        '        + mu * '
                                                        'sum(ufl.min_value(l, 0) ** 2 '
                                                        'for l in ev)\n'
                                                        'The small constant inside the '
                                                        'square root is REQUIRED: '
                                                        'without it the derivative is '
                                                        'undefined where the two '
                                                        'eigenvalues coincide.',
                                            'OPTIONAL': 'In 3D the same construction '
                                                        'works with the three '
                                                        'eigenvalues from the '
                                                        'closed-form (Cardano) '
                                                        'solution:\n'
                                                        '    qq = ufl.tr(e) / 3\n'
                                                        '    B = e - qq * '
                                                        'ufl.Identity(3)\n'
                                                        '    p = ufl.sqrt(ufl.inner(B, '
                                                        'B) / 6 + 1e-30)\n'
                                                        '    r = ufl.det(B / p) / 2\n'
                                                        '    phi = '
                                                        'ufl.acos(ufl.max_value(ufl.min_value(r, '
                                                        '1 - 1e-12), -1 + 1e-12)) / 3\n'
                                                        '    l1 = qq + 2 * p * '
                                                        'ufl.cos(phi)\n'
                                                        '    l3 = qq + 2 * p * '
                                                        'ufl.cos(phi + 2 * np.pi / 3)\n'
                                                        '    ev = [l1, 3 * qq - l1 - '
                                                        'l3, l3]\n'
                                                        'The volumetric-deviatoric '
                                                        '(Amor) split is simpler but '
                                                        'does NOT do the job - see the '
                                                        'pitfall.',
                                            'explanation': 'Without a split, the '
                                                           'energy that drives damage '
                                                           'does not know whether the '
                                                           'material is being pulled '
                                                           'apart or pressed together, '
                                                           'so cracks appear under '
                                                           'pure compression, which is '
                                                           'physically wrong.',
                                            'pitfalls': ['No split at all makes the '
                                                         'model completely sign-blind. '
                                                         'Signal: running the '
                                                         'identical specimen once in '
                                                         'tension and once in '
                                                         'compression gives damage '
                                                         'fields that agree to every '
                                                         'printed digit, and a crack '
                                                         'forms in both.',
                                                         'The volumetric-deviatoric '
                                                         '(Amor) split does NOT '
                                                         'suppress compressive damage '
                                                         'in a uniaxial test. Signal: '
                                                         'with the '
                                                         'volumetric-deviatoric split '
                                                         'a uniaxial compression step '
                                                         'still drives the maximum '
                                                         'damage all the way to one, '
                                                         'essentially as if there were '
                                                         'no split; only the spectral '
                                                         'split brings it down.',
                                                         'Expect the spectral split to '
                                                         'reduce compressive damage '
                                                         'strongly but not to exactly '
                                                         'zero. Signal: the maximum '
                                                         'damage under uniaxial '
                                                         'compression drops by about '
                                                         'an order of magnitude, to a '
                                                         'small fraction of one rather '
                                                         'than to zero, because the '
                                                         'lateral (Poisson) strains '
                                                         'are still tensile.']},
              'mesh_resolution': {'REQUIRED': 'Refine until the answer stops moving, '
                                              'and refine the region the crack will '
                                              'pass through. Report the dissipated '
                                              'energy against Gc times the crack area '
                                              'as your convergence check.',
                                  'OPTIONAL': 'Uniform refinement is fine for small '
                                              'problems; a locally refined band along '
                                              'the expected crack path is much '
                                              'cheaper.',
                                  'explanation': 'The regularised model only '
                                                 "reproduces Griffith's criterion in "
                                                 'the limit of a mesh fine compared '
                                                 'with l0. On a coarse mesh the '
                                                 'diffuse crack profile cannot be '
                                                 'represented, and the discrete model '
                                                 'behaves as if the material were '
                                                 'tougher than it is.',
                                  'pitfalls': ['A coarse mesh makes the specimen look '
                                               'STRONGER, not weaker. Signal: at fixed '
                                               'l0 and fixed Gc, coarsening the mesh '
                                               'raises both the energy dissipated in '
                                               'breaking the specimen and the '
                                               'displacement at which the crack '
                                               'finally runs; refining lowers both '
                                               'monotonically towards the Griffith '
                                               'value. If you are looking for a '
                                               'shortfall in fracture energy on a '
                                               'coarse mesh you will not find one - '
                                               'the error has the opposite sign.']},
              'extensions': {'note': 'Third-party phase-field frameworks built on '
                                     'DOLFINx exist as separate projects, but none is '
                                     'part of DOLFINx and none may be assumed present: '
                                     '`import phasefieldx` raises ModuleNotFoundError '
                                     'on a standard DOLFINx installation. Write the '
                                     'staggered scheme yourself, as in the example '
                                     'above.'},
              'pitfalls': ['[Numerical] Irreversibility is not optional and its '
                           'absence is not a small effect. Enforce it with a history '
                           'field updated by an element-wise maximum after every '
                           'displacement solve: `H.x.array[:] = np.maximum(H.x.array, '
                           'Hn.x.array)` where Hn holds the interpolated tensile '
                           'energy density. Signal: on a load-then-unload path WITHOUT '
                           'the history update, individual nodes fall from fully '
                           'damaged back to almost undamaged inside a single load step '
                           '- the worst per-step change in d is close to minus one - '
                           'and the damage integrated over the whole body has lost '
                           'roughly forty percent of its peak by the time the load is '
                           'back at zero, i.e. the crack visibly heals. WITH the '
                           'history update the same integral loses exactly nothing '
                           'during unloading. Verified at degree 1 and degree 2 in 2D '
                           'and at degree 1 in 3D; the effect is the same size in all '
                           'of them. Caveat: the history field guarantees the DRIVING '
                           'FORCE is monotone, not that every nodal value of d is; '
                           'small decreases still occur while the load is increasing '
                           'and the crack is redistributing, and those are a '
                           'discretisation artefact rather than healing.',
                           '[Numerical] A mesh that is coarse compared with l0 makes '
                           'the model TOUGHER, not weaker. The previously stated '
                           "signal - that the computed fracture energy 'under-shoots "
                           "Griffith's G_c * area by ~30-50% when h ~ l0' - is WRONG "
                           'IN SIGN and was not reproduced. Signal: running the same '
                           'specimen at a sequence of mesh sizes with l0 and Gc held '
                           'fixed, the energy dissipated in breaking it comes out '
                           'ABOVE Gc times the crack area at every resolution, the '
                           'excess grows steadily as the mesh is coarsened, and the '
                           'displacement at which the crack finally runs also rises '
                           'with coarsening - a coarse run reports a specimen that is '
                           'harder to break, not easier. Refining drives both '
                           'quantities down monotonically towards the Griffith value. '
                           'Use this as the convergence test: compute the dissipated '
                           'energy, divide by Gc times the crack area, and refine '
                           'until that ratio stops falling.',
                           '[Physics] Without a tension-compression split the model '
                           'cannot tell tension from compression at all, and the '
                           'standard volumetric-deviatoric split does not fix it. '
                           'Signal: read the damage extremum straight off the dolfinx '
                           'Function with d.x.array.max() at each load step: with no '
                           'split, the identical specimen loaded in tension and loaded '
                           'in compression produces damage fields that agree to every '
                           'printed digit, and a full crack forms under pure '
                           'compression. Switching to the volumetric-deviatoric (Amor) '
                           'split changes almost nothing - uniaxial compression still '
                           'drives the maximum damage to one. Only the SPECTRAL '
                           '(Miehe) eigenvalue split works: it brings the maximum '
                           'damage under uniaxial compression down by about an order '
                           'of magnitude while leaving the tension case cracking '
                           'normally. Note the spectral split does NOT give exactly '
                           'zero damage under compression - the lateral Poisson '
                           'strains are tensile - so a small nonzero maximum damage is '
                           'the correct result, not a bug. Verified in 2D on triangles '
                           'and quadrilaterals and in 3D on tetrahedra.',
                           '[Numerical] The staggered iteration count is not roughly '
                           'constant: it explodes in the one load step where the crack '
                           'propagates. Signal: the number of staggered sweeps needed '
                           'to reach a fixed tolerance on d sits in the single digits '
                           'while the crack is dormant, then jumps by one to two '
                           'orders of magnitude in the single step in which the '
                           'integrated damage takes its big jump, then drops straight '
                           'back to single digits afterwards. Set the staggered '
                           'iteration cap generously - a cap of a few tens truncates '
                           'exactly the step that matters, and it does so SILENTLY: '
                           'nothing is raised. The tell is the sweep count coming back '
                           'EQUAL TO THE CAP on one or two consecutive steps; record '
                           'the per-step sweep count and compare it against max_sweeps '
                           'rather than inspecting the crack. IMPORTANT CORRECTION: an '
                           'earlier version of this entry said the truncation leaves a '
                           'half-propagated crack. It does not - truncation DELAYS '
                           'propagation by roughly one load step and the regularised '
                           'surface energy has caught up again by the end of the ramp, '
                           'so a final-state check on the crack will show nothing '
                           'wrong and an agent waiting for a visibly half-formed crack '
                           'will never see one. Verified on 2D triangles, 2D '
                           'quadrilaterals and 3D tetrahedra, with the spike appearing '
                           'in each case at the step of largest damage growth. No '
                           'claim is made here about a monolithic scheme; that was not '
                           'run. (Verified by execution on dolfinx 0.10.0, 2026-08-06.)',
                           '[Numerical] The residual stiffness k_res in g(d) = '
                           '(1-d)**2 + k_res has a lower limit that depends on the '
                           'polynomial degree, so a value copied from a degree-1 '
                           'example can break a degree-2 run. Signal: the identical '
                           'notched specimen runs happily at degree 1 with a residual '
                           'stiffness far below one (k_res = 1e-6), but at degree 2 a '
                           'displacement solve inside the staggered loop fails with '
                           'SNES reporting DIVERGED_MAX_IT (getConvergedReason() == '
                           '-5). The very first solve, with d = 0, still converges. '
                           'With only the displacement space at degree 2 it fails on '
                           'the second staggered sweep of the first load step; with '
                           'displacement and damage both at degree 2 it fails at load '
                           'step 5 of the example. In the displacement-only variant it '
                           'keeps failing there with -5 when the Newton iteration '
                           'limit is raised tenfold, when the line search is changed '
                           "to 'bt' or 'l2', and when k_res is raised to 1e-4; k_res = "
                           '1e-2 makes both degree-2 variants run through all load '
                           'steps. The cause is the nearly-zero stiffness left in '
                           'fully broken elements, which higher-order elements feel '
                           'much more.',
                           '[API] There is no phase-field anything in DOLFINx, and the '
                           'third-party frameworks are not installed by default - '
                           'write the staggered scheme yourself. Signal: `import '
                           'phasefieldx` raises "ModuleNotFoundError: No module named '
                           '\'phasefieldx\'" on a standard DOLFINx installation. The '
                           'pieces you do get from DOLFINx are ordinary ones: '
                           'fem.functionspace for the three fields, NonlinearProblem '
                           'for the displacement step, LinearProblem for the damage '
                           'step, and fem.Expression + Function.interpolate to update '
                           'the history field.',
                           '[Numerical] Seeding a pre-crack by writing a large value '
                           'into the history field makes the damage overshoot one '
                           'slightly near the seed, which is harmless but will trip a '
                           'naive assertion that d <= 1. Signal: max(d) comes out a '
                           'fraction of a percent above one from the very first load '
                           'step, at nodes on the boundary between seeded and unseeded '
                           'cells; it is the AT2 gradient term smoothing a step change '
                           'in the driving force, not a solver failure. Check `min(d) '
                           '>= 0` and that the regularised surface energy increases, '
                           'rather than asserting a hard upper bound of one. The same '
                           'overshoot region is where the small non-monotone decreases '
                           'of d appear even with the history field enabled.',
                           '[Integration] Neither `problem.solve()` raises on failure, '
                           'so an unchecked staggered loop will happily iterate on a '
                           'diverged displacement field. Signal: the loop keeps '
                           'running and prints plausible damage values while SNES has '
                           'been returning a negative converged reason for every '
                           'sweep. Put `assert pu.solver.getConvergedReason() > 0` '
                           'immediately after the displacement solve, and separately '
                           'check that the regularised surface energy is '
                           'non-decreasing across load steps - it is a cheap, '
                           'reference-free indicator that the crack is growing rather '
                           'than the solution falling apart.']},
 'stokes_darcy': {'description': 'Coupled free-fluid / porous-medium flow: Stokes in '
                                 'the open region, Darcy in the porous region. The '
                                 'route verified here is the single-mesh BRINKMAN '
                                 'formulation - one Taylor-Hood velocity/pressure pair '
                                 'over the whole domain plus a Darcy drag term mu/K*u '
                                 'switched on only in the porous cells. It is NOT a '
                                 'true Stokes-Darcy interface coupling: the velocity '
                                 'is continuous across the interface by construction '
                                 'and the Beavers-Joseph-Saffman slip condition is not '
                                 'imposed anywhere in the form.',
                  'minimal_working_example': 'import numpy as np\n'
                                             'import basix.ufl\n'
                                             'import ufl\n'
                                             'from mpi4py import MPI\n'
                                             'from dolfinx import fem, mesh\n'
                                             'from dolfinx.fem.petsc import '
                                             'LinearProblem\n'
                                             '\n'
                                             'msh = '
                                             'mesh.create_rectangle(MPI.COMM_WORLD, '
                                             '[np.array([0.0, 0.0]), np.array([2.0, '
                                             '1.0])],\n'
                                             '                            [32, 16], '
                                             'mesh.CellType.triangle)\n'
                                             'gdim, tdim = msh.geometry.dim, '
                                             'msh.topology.dim\n'
                                             '\n'
                                             '# ---- cell markers: 1 = free fluid (y > '
                                             '0.5), 2 = porous bed (y < 0.5) ----\n'
                                             'por = mesh.locate_entities(msh, tdim, '
                                             'lambda x: x[1] <= 0.5 + 1e-12)\n'
                                             'ids = '
                                             'np.full(msh.topology.index_map(tdim).size_local, '
                                             '1, dtype=np.int32)\n'
                                             'ids[por] = 2\n'
                                             'ct = mesh.meshtags(msh, tdim, '
                                             'np.arange(ids.size, dtype=np.int32), '
                                             'ids)\n'
                                             'dx = ufl.Measure("dx", domain=msh, '
                                             'subdomain_data=ct)\n'
                                             '\n'
                                             '# ---- facet markers: 1 = inlet, 2 = '
                                             'outlet above, 3 = outlet below ----\n'
                                             'fs, ms = [], []\n'
                                             'for tag, fn in ((1, lambda x: '
                                             'np.isclose(x[0], 0.0)),\n'
                                             '                (2, lambda x: '
                                             'np.isclose(x[0], 2.0) & (x[1] >= 0.5 - '
                                             '1e-12)),\n'
                                             '                (3, lambda x: '
                                             'np.isclose(x[0], 2.0) & (x[1] <= 0.5 + '
                                             '1e-12))):\n'
                                             '    f = '
                                             'mesh.locate_entities_boundary(msh, tdim '
                                             '- 1, fn)\n'
                                             '    fs.append(f)\n'
                                             '    ms.append(np.full(f.size, tag, '
                                             'dtype=np.int32))\n'
                                             'fa, ma = np.concatenate(fs), '
                                             'np.concatenate(ms)\n'
                                             'srt = np.argsort(fa)\n'
                                             'ft = mesh.meshtags(msh, tdim - 1, '
                                             'fa[srt], ma[srt])\n'
                                             'ds = ufl.Measure("ds", domain=msh, '
                                             'subdomain_data=ft)\n'
                                             '\n'
                                             '# ---- Taylor-Hood velocity/pressure '
                                             'over the WHOLE domain ----\n'
                                             'Ve = basix.ufl.element("Lagrange", '
                                             'msh.basix_cell(), 2, shape=(gdim,))\n'
                                             'Qe = basix.ufl.element("Lagrange", '
                                             'msh.basix_cell(), 1)\n'
                                             'W = fem.functionspace(msh, '
                                             'basix.ufl.mixed_element([Ve, Qe]))\n'
                                             '(u, p), (v, q) = ufl.TrialFunctions(W), '
                                             'ufl.TestFunctions(W)\n'
                                             '\n'
                                             'mu = fem.Constant(msh, 1.0)          # '
                                             'dynamic viscosity\n'
                                             'K = fem.Constant(msh, 1e-4)          # '
                                             'permeability of the bed\n'
                                             'p_in = fem.Constant(msh, 1.0)\n'
                                             'n = ufl.FacetNormal(msh)\n'
                                             'a = (2.0 * mu * '
                                             'ufl.inner(ufl.sym(ufl.grad(u)), '
                                             'ufl.sym(ufl.grad(v))) * dx\n'
                                             '     - p * ufl.div(v) * dx\n'
                                             '     - q * ufl.div(u) * dx\n'
                                             '     + (mu / K) * ufl.inner(u, v) * '
                                             'dx(2))     # Darcy drag, porous cells '
                                             'only\n'
                                             'L = -p_in * ufl.dot(n, v) * '
                                             'ds(1)              # inlet pressure; '
                                             'outlet p = 0 naturally\n'
                                             '\n'
                                             'V0, _ = W.sub(0).collapse()\n'
                                             'u_wall = fem.Function(V0)\n'
                                             'walls = '
                                             'mesh.locate_entities_boundary(msh, tdim '
                                             '- 1,\n'
                                             '                                      '
                                             'lambda x: np.isclose(x[1], 0.0) | '
                                             'np.isclose(x[1], 1.0))\n'
                                             'bcs = [fem.dirichletbc(u_wall,\n'
                                             '                       '
                                             'fem.locate_dofs_topological((W.sub(0), '
                                             'V0), tdim - 1, walls), W.sub(0))]\n'
                                             '\n'
                                             'problem = LinearProblem(a, L, bcs=bcs, '
                                             'petsc_options_prefix="sd_",\n'
                                             '                        '
                                             'petsc_options={"ksp_type": "preonly", '
                                             '"pc_type": "lu",\n'
                                             '                                       '
                                             '"pc_factor_mat_solver_type": "mumps"})\n'
                                             'wh = problem.solve()\n'
                                             'uh, ph = wh.sub(0).collapse(), '
                                             'wh.sub(1).collapse()\n'
                                             'print("KSP converged reason :", '
                                             'problem.solver.getConvergedReason())\n'
                                             '\n'
                                             '# ---- physical self-checks (no '
                                             'reference solution needed) ----\n'
                                             'q_in = '
                                             '-fem.assemble_scalar(fem.form(ufl.dot(uh, '
                                             'n) * ds(1)))\n'
                                             'q_free = '
                                             'fem.assemble_scalar(fem.form(ufl.dot(uh, '
                                             'n) * ds(2)))\n'
                                             'q_por = '
                                             'fem.assemble_scalar(fem.form(ufl.dot(uh, '
                                             'n) * ds(3)))\n'
                                             'print("mass balance rel err : %.3e (must '
                                             'be ~1e-14)" % (abs(q_in - q_free - '
                                             'q_por) / abs(q_in)))\n'
                                             'a_free = '
                                             'fem.assemble_scalar(fem.form(1.0 * '
                                             'dx(1)))\n'
                                             'a_por = fem.assemble_scalar(fem.form(1.0 '
                                             '* dx(2)))\n'
                                             'm_free = '
                                             'fem.assemble_scalar(fem.form(uh[0] * '
                                             'dx(1))) / a_free\n'
                                             'm_por = '
                                             'fem.assemble_scalar(fem.form(uh[0] * '
                                             'dx(2))) / a_por\n'
                                             'print("mean u_x free / porous : %.6e  '
                                             '%.6e" % (m_free, m_por))\n'
                                             'print("bed is much slower     :", '
                                             'bool(abs(m_por) < 0.1 * abs(m_free)))\n'
                                             'print("flow goes inlet->outlet:", '
                                             'bool(q_in > 0.0 and m_free > 0.0))\n'
                                             'p0 = fem.assemble_scalar(fem.form(ph * '
                                             'ds(1))) / '
                                             'fem.assemble_scalar(fem.form(1.0 * '
                                             'ds(1)))\n'
                                             'print("recovered inlet pressure ~ '
                                             'imposed : %.4f" % p0)\n'
                                             'pcheck = fem.functionspace(msh, ("DG", '
                                             '0))\n'
                                             'pav = fem.Function(pcheck)\n'
                                             'pav.interpolate(fem.Expression(ph, '
                                             'pcheck.element.interpolation_points))\n'
                                             'osc = '
                                             'np.sqrt(fem.assemble_scalar(fem.form((ph '
                                             '- pav) ** 2 * ufl.dx)))\n'
                                             'sm = '
                                             'np.sqrt(fem.assemble_scalar(fem.form(ph '
                                             '** 2 * ufl.dx)))\n'
                                             'print("pressure oscillation indicator : '
                                             '%.3e (must stay well below 1)" % (osc / '
                                             'sm))\n'
                                             'print("all values finite    :", '
                                             'bool(np.all(np.isfinite(wh.x.array))))\n',
                  'function_space': {'REQUIRED': 'One inf-sup stable (LBB) '
                                                 'velocity/pressure pair over the '
                                                 'WHOLE domain, built with basix.ufl:\n'
                                                 '    import basix.ufl\n'
                                                 '    Ve = '
                                                 'basix.ufl.element("Lagrange", '
                                                 'msh.basix_cell(), 2, shape=(gdim,))\n'
                                                 '    Qe = '
                                                 'basix.ufl.element("Lagrange", '
                                                 'msh.basix_cell(), 1)\n'
                                                 '    W  = fem.functionspace(msh, '
                                                 'basix.ufl.mixed_element([Ve, Qe]))\n'
                                                 '    (u, p), (v, q) = '
                                                 'ufl.TrialFunctions(W), '
                                                 'ufl.TestFunctions(W)\n'
                                                 'Note the plural '
                                                 'TrialFunctionS/TestFunctionS on a '
                                                 'mixed space.',
                                     'OPTIONAL': 'Any stable pair: P2/P1 (Taylor-Hood) '
                                                 'and P3/P2 both work, on triangles, '
                                                 'quadrilaterals, tetrahedra and '
                                                 'hexahedra, in 2D and 3D. Equal order '
                                                 '(P1/P1) is NOT an option without a '
                                                 'stabilisation term. Recovering the '
                                                 'components afterwards: uh, ph = '
                                                 'wh.sub(0).collapse(), '
                                                 'wh.sub(1).collapse().',
                                     'explanation': 'The Brinkman form is a Stokes '
                                                    'saddle point everywhere; the '
                                                    'porous region only adds a '
                                                    'positive-definite mass-like term. '
                                                    'So the space requirement is '
                                                    'exactly the Stokes one, and a '
                                                    'single H1 velocity space covers '
                                                    'both regions.',
                                     'pitfalls': ['Never use an equal-order pair such '
                                                  'as P1/P1. Signal: no error, KSP '
                                                  'reason 4, and a checkerboard '
                                                  'pressure - ||p - '
                                                  'cellwise_average(p)||/||p|| stays '
                                                  'between about 0.7 and 0.95 on every '
                                                  'mesh, where a stable pair gives a '
                                                  'few percent and falling; max|p| is '
                                                  'many times the imposed pressure and '
                                                  'jumps around erratically from mesh '
                                                  'to mesh.',
                                                  'Do not pair P1 velocity with DG0 '
                                                  'pressure. Signal: '
                                                  'problem.solver.getConvergedReason() '
                                                  'returns -11 '
                                                  '(DIVERGED_PCSETUP_FAILED) and '
                                                  'max(abs(uh.x.array)) is inf.']},
                  'weak_form': {'REQUIRED': 'Brinkman: Stokes everywhere plus a Darcy '
                                            'drag restricted to the porous cells.\n'
                                            '    a = '
                                            '(2*mu*ufl.inner(ufl.sym(ufl.grad(u)), '
                                            'ufl.sym(ufl.grad(v)))*dx\n'
                                            '         - p*ufl.div(v)*dx\n'
                                            '         - q*ufl.div(u)*dx\n'
                                            '         + (mu/K)*ufl.inner(u, '
                                            'v)*dx(2))     # 2 = porous marker\n'
                                            '    L = ufl.inner(f, '
                                            'v)*dx                    # f = '
                                            'fem.Constant(msh, np.zeros(gdim))\n'
                                            '`dx` MUST be built as ufl.Measure("dx", '
                                            'domain=msh, subdomain_data=cell_tags) so '
                                            "that dx(2) means 'porous cells only'.",
                                'OPTIONAL': 'Use ufl.inner(ufl.grad(u), ufl.grad(v)) '
                                            'instead of the symmetric gradient if you '
                                            'want the vector-Laplacian form (then the '
                                            'natural outflow condition changes '
                                            'meaning). K may be a spatially varying '
                                            'fem.Function on a DG0 space instead of a '
                                            'Constant; then write (mu/K)*inner(u, '
                                            'v)*dx over the whole domain with K huge '
                                            'in the free region, or keep dx(2). A '
                                            'Forchheimer correction adds a nonlinear '
                                            '|u|-dependent drag and turns the problem '
                                            'into a NonlinearProblem.',
                                'explanation': 'As K decreases the drag term mu/K '
                                               'dominates and the momentum equation '
                                               "degenerates to Darcy's law u = "
                                               '-(K/mu)*grad(p) inside the marked '
                                               'cells, while the unmarked cells stay '
                                               'pure Stokes. One form, one space, no '
                                               'interface term.',
                                'pitfalls': ['Do not expect the Beavers-Joseph-Saffman '
                                             'condition from this form. Signal: no '
                                             'error; the velocity is a single H1 field '
                                             'so it is continuous across the interface '
                                             'and there is no slip jump at all - the '
                                             'only interface physics present is a '
                                             'Brinkman screening layer whose thickness '
                                             'scales like sqrt(K).',
                                             'Do not forget subdomain_data on the '
                                             'measure. Signal: a bare ufl.dx(2) raises '
                                             'nothing but integrates over an empty '
                                             'set, so the drag term silently '
                                             'disappears and the porous region flows '
                                             'like open fluid.']},
                  'boundary_conditions': {'REQUIRED': 'No-slip walls as a Dirichlet '
                                                      'condition on the velocity '
                                                      'SUBSPACE, using the collapsed '
                                                      'space for both the dof lookup '
                                                      'and the value:\n'
                                                      '    V0, _ = '
                                                      'W.sub(0).collapse()\n'
                                                      '    u_wall = '
                                                      'fem.Function(V0)                       '
                                                      '# zero by default\n'
                                                      '    dofs = '
                                                      'fem.locate_dofs_topological((W.sub(0), '
                                                      'V0), tdim-1, wall_facets)\n'
                                                      '    bcs = '
                                                      '[fem.dirichletbc(u_wall, dofs, '
                                                      'W.sub(0))]\n'
                                                      'Drive the flow with a pressure '
                                                      'difference imposed WEAKLY '
                                                      'through the natural term, which '
                                                      'needs no pressure Dirichlet '
                                                      'condition and leaves no '
                                                      'pressure nullspace:\n'
                                                      '    n = ufl.FacetNormal(msh)\n'
                                                      '    L = -p_in*ufl.dot(n, '
                                                      'v)*ds(1)                   # '
                                                      'outlet p = 0 by doing nothing',
                                          'OPTIONAL': 'A prescribed inflow profile '
                                                      'works too: interpolate it into '
                                                      'a Function on V0 and apply it '
                                                      'like the wall condition. If '
                                                      'every boundary carries a '
                                                      'velocity Dirichlet condition, '
                                                      'the pressure is determined only '
                                                      'up to a constant and you must '
                                                      'attach a nullspace or pin one '
                                                      'pressure dof.',
                                          'explanation': 'Every boundary facet you do '
                                                         'not constrain and do not put '
                                                         'in a ds term gets the '
                                                         'do-nothing condition sigma.n '
                                                         '= 0, which is a free '
                                                         'outflow. That is the correct '
                                                         'outlet condition and a '
                                                         'silent leak everywhere else.',
                                          'pitfalls': ['Mark interface-adjacent facets '
                                                       'with tolerant inequalities, '
                                                       'never strict ones. Signal: no '
                                                       'error; with x[1] > 0.5 and '
                                                       'x[1] < 0.5 the facets touching '
                                                       'y = 0.5 land in neither set, '
                                                       'the measured boundary length '
                                                       'is short by two facets, and '
                                                       'the mass-balance check is off '
                                                       'by several percent.',
                                                       'Constrain every wall, '
                                                       'including the ones you did not '
                                                       'think about. Signal: no error, '
                                                       'KSP reason 4, and a mass '
                                                       'balance error of order 1 '
                                                       '(about 99% in a 3D box whose '
                                                       'two z faces were left '
                                                       'unconstrained) because those '
                                                       'faces became outlets.']},
                  'solver': {'REQUIRED': 'A sparse direct solve is the reliable '
                                         'default for the coupled saddle-point '
                                         'system:\n'
                                         '    problem = LinearProblem(a, L, bcs=bcs, '
                                         'petsc_options_prefix="sd_",\n'
                                         '                            '
                                         'petsc_options={"ksp_type": "preonly",\n'
                                         '                                           '
                                         '"pc_type": "lu",\n'
                                         '                                           '
                                         '"pc_factor_mat_solver_type": "mumps"})\n'
                                         '    wh = problem.solve()\n'
                                         '    assert '
                                         'problem.solver.getConvergedReason() > 0\n'
                                         'petsc_options_prefix is a REQUIRED keyword '
                                         'argument in dolfinx 0.10, and solve() does '
                                         'not raise on failure.',
                             'OPTIONAL': 'MUMPS handles the whole permeability range '
                                         'down to K = 1e-12 (drag coefficient 1e12) on '
                                         'this build without pivoting trouble, so vary '
                                         'K freely before reaching for anything '
                                         'cleverer. For a stand-alone Darcy pressure '
                                         'block, CG with {"pc_type": "hypre", '
                                         '"pc_hypre_type": "boomeramg"} is the robust '
                                         'choice; gamg and bjacobi+ilu also converge '
                                         'but need more iterations when the '
                                         'permeability field is disconnected. For '
                                         'large 3D problems use a fieldsplit '
                                         'preconditioner rather than LU.',
                             'explanation': 'The system is symmetric but indefinite, '
                                            'so plain CG is not applicable to the '
                                            'coupled system; either factor it directly '
                                            'or use MINRES/fieldsplit with a proper '
                                            'block preconditioner.',
                             'pitfalls': ['Always read getConvergedReason(). Signal: '
                                          'an unstable element pair returns reason -11 '
                                          'with inf in the solution, and an '
                                          'equal-order pair returns reason 4 with a '
                                          'physically meaningless pressure - neither '
                                          'raises an exception.']},
                  'subdomains': {'REQUIRED': 'Cell tags are what make one mesh behave '
                                             'as two materials:\n'
                                             '    por = mesh.locate_entities(msh, '
                                             'tdim, lambda x: x[1] <= 0.5 + 1e-12)\n'
                                             '    ids = '
                                             'np.full(msh.topology.index_map(tdim).size_local, '
                                             '1, dtype=np.int32)\n'
                                             '    ids[por] = 2\n'
                                             '    ct = mesh.meshtags(msh, tdim, '
                                             'np.arange(ids.size, dtype=np.int32), '
                                             'ids)\n'
                                             '    dx = ufl.Measure("dx", domain=msh, '
                                             'subdomain_data=ct)\n'
                                             'The entity array and the value array '
                                             'must have the same length and correspond '
                                             'element by element. The entity array '
                                             'must also be sorted and duplicate-free '
                                             '(documented meshtags precondition), so '
                                             'sort a concatenated facet array with '
                                             'np.argsort before calling meshtags, as '
                                             'the example does. Release builds do not '
                                             'check this, but unsorted tags that '
                                             'include ghost entities integrate '
                                             'incorrectly in parallel.',
                                 'OPTIONAL': 'Cell tags can also come from a gmsh .msh '
                                             'file through dolfinx.io.gmsh.read_from_msh '
                                             '(the returned MeshData carries '
                                             '.cell_tags; dolfinx.io.gmshio no longer '
                                             'exists in 0.10), which is '
                                             'the practical route for a non-trivial '
                                             'porous geometry. Any integer tags work; '
                                             'the numbers only have to match the '
                                             'dx(tag) calls.',
                                 'explanation': 'locate_entities marks a cell when ALL '
                                                'of its vertices satisfy the '
                                                'predicate, so use a tolerance (1e-12) '
                                                'on interface-aligned predicates and '
                                                'make sure the interface lies on a '
                                                'mesh line.',
                                 'pitfalls': ['Give the interface predicate a '
                                              'tolerance and make the two sets '
                                              'exhaustive. Signal: no error; cells or '
                                              'facets on the dividing line are dropped '
                                              'from both tags, which shows up only as '
                                              'a percent-level defect in an integral, '
                                              'e.g. a marked outlet length of 0.875 '
                                              'where the geometry says 1.0.']},
                  'two_mesh_route': {'REQUIRED': 'If you really need different spaces '
                                                 'per region (H(div) Darcy next to H1 '
                                                 'Stokes), carve out a submesh. In '
                                                 'dolfinx 0.10 the signature and the '
                                                 'return are:\n'
                                                 '    create_submesh(msh: Mesh, dim: '
                                                 'int, entities: NDArray[np.int32])\n'
                                                 '        -> (Mesh, EntityMap, '
                                                 'EntityMap, NDArray[np.int32])\n'
                                                 '    submesh, cell_map, vertex_map, '
                                                 'node_map = mesh.create_submesh(msh, '
                                                 'tdim, cells)\n'
                                                 'FOUR objects, of which the two '
                                                 'middle ones are EntityMap objects '
                                                 '(methods: dim, topology, '
                                                 'sub_topology, '
                                                 'sub_topology_to_topology), not plain '
                                                 'arrays. Any form that mixes a '
                                                 'submesh function with a parent-mesh '
                                                 'measure needs the entity map passed '
                                                 'as a SEQUENCE:\n'
                                                 '    F = '
                                                 'fem.form(p_sub*v_parent*dx(2), '
                                                 'entity_maps=[cell_map])\n'
                                                 'The same kwarg exists on '
                                                 'fem.petsc.LinearProblem(..., '
                                                 'entity_maps=[...]), which is how a '
                                                 'genuine two-domain block system is '
                                                 'assembled.',
                                     'OPTIONAL': 'A trial function on the submesh and '
                                                 'a test function on the parent mesh '
                                                 'assemble into a rectangular coupling '
                                                 'block, so the block layout '
                                                 '[[A_stokes, C], [C_transpose, '
                                                 'A_darcy]] is buildable. This route '
                                                 'is more work and is only worth it '
                                                 'when the per-region spaces genuinely '
                                                 'differ; the Brinkman route needs '
                                                 'none of it.',
                                     'explanation': 'The EntityMap tells the assembler '
                                                    'which parent cell each submesh '
                                                    'cell corresponds to, so the two '
                                                    'meshes can appear in one '
                                                    'integral.',
                                     'pitfalls': ['Unpack four values, not three. '
                                                  'Signal: `ValueError: too many '
                                                  'values to unpack (expected 3)`.',
                                                  'Pass entity_maps as a list of '
                                                  'EntityMap objects. Signal: omitting '
                                                  'it gives `RuntimeError: '
                                                  'Incompatible mesh. argument '
                                                  'entity_maps must be provided.`, and '
                                                  'passing an older-style dict '
                                                  '{submesh: array} gives `TypeError: '
                                                  '__init__(): incompatible function '
                                                  'arguments.` listing entity_maps as '
                                                  'Sequence[dolfinx.cpp.mesh.EntityMap].',
                                                  'Do not integrate a submesh Function '
                                                  'over the parent domain and expect a '
                                                  'warning. Signal: none; cells '
                                                  'outside the submesh contribute '
                                                  'exactly zero, so the integral '
                                                  'silently returns the sub-region '
                                                  'value.',
                                                  'After a form-compile failure, clear '
                                                  'the stale FFCx cache entry before '
                                                  're-running. Signal: the second run '
                                                  'of the same broken script replaces '
                                                  'the real error with `TimeoutError: '
                                                  'JIT compilation timed out, probably '
                                                  'due to a failed previous compile. '
                                                  'Try cleaning cache (e.g. remove '
                                                  '<cache_dir>/libffcx_forms_<hash>.c) '
                                                  'or increase timeout option.`']},
                  'interface_physics': {'REQUIRED': 'State which model you are '
                                                    'running. The Brinkman form above '
                                                    'enforces continuity of velocity '
                                                    'and of normal stress across the '
                                                    'interface because there is only '
                                                    'one H1 velocity field; it does '
                                                    'NOT impose the '
                                                    'Beavers-Joseph-Saffman tangential '
                                                    'slip condition, and there is no '
                                                    'alpha_BJ parameter anywhere in '
                                                    'it. The interface behaviour it '
                                                    'does produce is a Brinkman '
                                                    'screening layer: below the '
                                                    'interface the tangential velocity '
                                                    'decays over a length that '
                                                    'approaches sqrt(K/mu_eff) as K '
                                                    'becomes small.',
                                        'OPTIONAL': 'A true Stokes-Darcy interface '
                                                    'coupling with '
                                                    'Beavers-Joseph-Saffman needs two '
                                                    'regions with their own spaces '
                                                    '(see two_mesh_route) and an '
                                                    'explicit facet integral carrying '
                                                    'the tangential slip term with a '
                                                    'Constant alpha_BJ of order 0.1 to '
                                                    '1. Nothing in dolfinx assembles '
                                                    'that for you.',
                                        'explanation': 'Mesh resolution near the '
                                                       'interface is set by the '
                                                       'screening length, not by the '
                                                       'geometry: cells there should '
                                                       'be no larger than sqrt(K).',
                                        'pitfalls': ['Resolve the screening layer. '
                                                     'Signal: no error; with cells '
                                                     'about ten times sqrt(K) the '
                                                     'interface velocity comes out '
                                                     'roughly a third too low and the '
                                                     'total flux a couple of percent '
                                                     'off, and both converge as the '
                                                     'cells shrink to sqrt(K).']},
                  'pitfalls': ['[API] There is no built-in Stokes-Darcy, Darcy or '
                               'Brinkman anything in dolfinx - the whole coupled form '
                               'has to be written by hand. Signal: `AttributeError: '
                               "module 'dolfinx.fem' has no attribute 'StokesDarcy'`, "
                               'and a name scan over dolfinx, dolfinx.fem, '
                               'dolfinx.fem.petsc, dolfinx.mesh, dolfinx.io, '
                               'dolfinx.nls and dolfinx.la for any name containing '
                               'darcy, stokes, brinkman, porous or biot returns an '
                               'empty list. The verified working route is the '
                               'single-mesh Brinkman formulation in '
                               'minimal_working_example: Taylor-Hood over the whole '
                               'domain plus (mu/K)*inner(u, v)*dx(porous_marker). '
                               '(Executed on dolfinx 0.10.0.)',
                               '[API] dolfinx.mesh.create_submesh has the signature '
                               'create_submesh(msh, dim, entities) and returns FOUR '
                               'objects in 0.10: (Mesh, EntityMap, EntityMap, numpy '
                               'int32 array) - the sub mesh, the entity map, the '
                               'vertex map and the geometry node map. Signal: the '
                               'three-value unpacking that older code and older '
                               'tutorials use raises `ValueError: too many values to '
                               'unpack (expected 3)`. The two middle returns are '
                               'dolfinx.mesh.EntityMap objects with methods dim, '
                               'topology, sub_topology and sub_topology_to_topology; '
                               'they are NOT index arrays, so indexing them fails. '
                               '(Executed on dolfinx 0.10.0.)',
                               '[API] A form that mixes a submesh Function with a '
                               'parent-mesh measure must be compiled with entity_maps, '
                               'and in 0.10 that argument is a SEQUENCE of EntityMap '
                               'objects. Signal: without it, '
                               '`fem.form(p_sub*v_parent*dx(2))` raises `RuntimeError: '
                               'Incompatible mesh. argument entity_maps must be '
                               'provided.`; with the older dict form '
                               'entity_maps={submesh: array} it raises `TypeError: '
                               '__init__(): incompatible function arguments.` whose '
                               'type list shows `entity_maps: '
                               'collections.abc.Sequence[dolfinx.cpp.mesh.EntityMap]`. '
                               'With entity_maps=[cell_map] the form compiles and '
                               'assembles, including rectangular coupling blocks '
                               'between a submesh trial function and a parent test '
                               'function. One silent trap remains: integrating such a '
                               'form over the WHOLE parent domain does not warn - '
                               'cells outside the submesh contribute exactly zero, so '
                               'the integral quietly equals the sub-region one. '
                               '(Executed on dolfinx 0.10.0.)',
                               '[Numerical] The Brinkman/Stokes system is a saddle '
                               'point, so the velocity and pressure spaces must '
                               'satisfy the LBB (inf-sup) condition; the Darcy drag '
                               'term does not rescue an unstable pair. Signal: with '
                               'equal-order P1/P1 nothing is raised, '
                               'getConvergedReason() returns 4, the mass balance is '
                               'still exact to about 1e-15, and only the pressure is '
                               'wrong - it oscillates cell to cell, with ||p - '
                               'cellwise_average(p)||/||p|| between about 0.7 and 0.95 '
                               'on every mesh tried while a stable pair gives a few '
                               'percent that shrinks under refinement, and with a peak '
                               'magnitude one to three orders above the imposed inlet '
                               'pressure. That peak is erratic rather than monotone in '
                               'the mesh size, so use the oscillation ratio and not '
                               'max|p| as the test. With P1 velocity and DG0 pressure '
                               'the failure is loud instead: getConvergedReason() '
                               'returns -11 (DIVERGED_PCSETUP_FAILED) and the solution '
                               'is inf. P2/P1 and P3/P2 were both clean on triangles '
                               'and quadrilaterals in 2D and on tetrahedra and '
                               'hexahedra in 3D. (Executed on dolfinx 0.10.0.)',
                               '[Input] Facet and cell predicates for the region '
                               'touching the porous interface must use tolerant '
                               'inequalities. dolfinx marks an entity only when ALL of '
                               'its vertices satisfy the predicate, so splitting an '
                               'outlet with x[1] > 0.5 and x[1] < 0.5 leaves the two '
                               'facets that touch y = 0.5 in neither set. Signal: no '
                               'error and no warning; the two tagged pieces of a '
                               'boundary of length 1.0 measure 0.875 in total, and a '
                               'mass-balance check that should read 1e-14 reads a few '
                               'percent instead. Use x[1] >= 0.5 - 1e-12 and x[1] <= '
                               '0.5 + 1e-12. (Executed on dolfinx 0.10.0.)',
                               '[Input] Every boundary facet that carries neither a '
                               'Dirichlet condition nor a ds term silently gets the '
                               'do-nothing condition sigma.n = 0, i.e. it becomes a '
                               'free outlet. Signal: no error, KSP reason 4, '
                               'plausible-looking velocity and pressure fields - but '
                               'the inflow/outflow balance is off by order 1 (about '
                               '99% in a 3D box in which the two z faces were left '
                               'unconstrained). This is why a flux balance over the '
                               'tagged inlet and outlet belongs in every run: it is '
                               'the one check that catches a forgotten wall. (Executed '
                               'on dolfinx 0.10.0.)',
                               '[Physics] The single-mesh Brinkman form does NOT '
                               'impose the Beavers-Joseph-Saffman interface condition. '
                               'There is one H1 velocity field over both regions, so '
                               'the velocity - including its tangential component - is '
                               'continuous across the interface by construction and no '
                               'slip jump and no alpha_BJ parameter exist anywhere in '
                               'the form. What the model does produce is a Brinkman '
                               'screening layer: below the interface the tangential '
                               'velocity decays exponentially, and the measured decay '
                               'length TENDS TO sqrt(K/mu_eff) as K decreases. Do not '
                               'test the DIRECTION of that approach: an earlier version '
                               'of this entry said the decay length comes down to '
                               'sqrt(K/mu_eff) from above, and on a mesh resolved enough '
                               'to measure it the ratio sits at or just below one, so '
                               'only the limit reproduces, not the direction. Signal: '
                               'numerical, not textual - with cells '
                               'about ten times sqrt(K) the interface velocity is '
                               'roughly a third below its resolved value and the total '
                               'flux a couple of percent off, with no error message; '
                               'refining until the cell size reaches sqrt(K) removes '
                               'both errors. Size the mesh near the interface from '
                               'sqrt(K), not from the geometry. Any claim about '
                               'agreement or disagreement with laboratory '
                               'Beavers-Joseph experiments is outside what this model '
                               'can be said to represent. (Executed on dolfinx '
                               '0.10.0.)',
                               '[Performance] A large permeability CONTRAST is not by '
                               'itself a preconditioner problem - the geometry of the '
                               'permeability field is. Signal: for a Darcy pressure '
                               'block with a single planar jump, taking the contrast '
                               'from 1 to 1e-9 leaves CG iteration counts essentially '
                               'flat for every preconditioner tried (jacobi, '
                               'bjacobi+ilu, icc, asm, gamg, hypre boomeramg): all '
                               'return getConvergedReason() = 2 with a true relative '
                               'residual at the requested tolerance, and the worst '
                               'degradation is about 20% more iterations. Nothing '
                               "stalls. The previously quoted 'block-Jacobi stalls "
                               "with residual ratio ~1' does NOT reproduce. Where the "
                               'contrast does bite is when the high-permeability '
                               'region is disconnected from the Dirichlet boundary or '
                               'randomly distributed cell by cell: there jacobi grows '
                               'by roughly an order of magnitude and gamg by about '
                               'five times, while hypre boomeramg stays at about ten '
                               'iterations throughout. Use hypre boomeramg for Darcy '
                               'pressure blocks and stop worrying about the contrast '
                               'number itself. (Executed on dolfinx 0.10.0, serial.)']},
 '_provenance': {'description': 'Record of what was actually EXECUTED against an '
                                'installed dolfinx to substantiate the entries in this '
                                'catalog. Entries not listed here were not re-run in '
                                'that pass and carry their older audit tag.',
                 '2026-08-03_adversarial_reverification': {'environment': 'dolfinx '
                                                                          '0.10.0, '
                                                                          'basix '
                                                                          '0.10.0, ufl '
                                                                          '2025.2.1, '
                                                                          'petsc4py '
                                                                          '3.24.4 '
                                                                          '(real '
                                                                          'scalars, '
                                                                          '32-bit '
                                                                          'indices, '
                                                                          'MUMPS + '
                                                                          'SuperLU_DIST '
                                                                          '+ UMFPACK '
                                                                          'all '
                                                                          'present), '
                                                                          'slepc4py '
                                                                          '3.24.3, '
                                                                          'mpi4py '
                                                                          '4.1.1, gmsh '
                                                                          '4.15.1, '
                                                                          'Python '
                                                                          '3.12.13 '
                                                                          'conda-forge, '
                                                                          'Linux '
                                                                          'x86_64. NOT '
                                                                          'installed '
                                                                          'in that '
                                                                          'env: '
                                                                          'dolfinx_mpc, '
                                                                          'adios4dolfinx, '
                                                                          'pyamg.',
                                                           'method': "Every 'works' "
                                                                     'claim was '
                                                                     'executed; every '
                                                                     "'fails' / "
                                                                     'pitfall claim '
                                                                     'was reproduced '
                                                                     'by running the '
                                                                     'WRONG variant '
                                                                     'and comparing '
                                                                     'the observed '
                                                                     'error text or '
                                                                     'numerical '
                                                                     'misbehaviour to '
                                                                     'the documented '
                                                                     'signal. '
                                                                     'Numerical claims '
                                                                     '(element orders, '
                                                                     'stabilisation '
                                                                     'requirements, '
                                                                     'locking, penalty '
                                                                     'scaling, '
                                                                     'pollution) were '
                                                                     'checked with '
                                                                     'manufactured-solution '
                                                                     'convergence '
                                                                     'studies: '
                                                                     'prescribe an '
                                                                     'exact u, derive '
                                                                     'f symbolically '
                                                                     '(sympy) or '
                                                                     'analytically, '
                                                                     'refine, and fit '
                                                                     'the observed L2 '
                                                                     'rate.',
                                                           'mms_studies_run': ['Poisson, '
                                                                               'smooth '
                                                                               'manufactured '
                                                                               'solution, '
                                                                               'P1/P2/P3, '
                                                                               'N=8..64 '
                                                                               '(orders '
                                                                               'matched '
                                                                               'theory)',
                                                                               'Convection-diffusion '
                                                                               'b=(1,1), '
                                                                               'kappa=0.01, '
                                                                               'P1 and '
                                                                               'P2, '
                                                                               'N=8..128: '
                                                                               'plain '
                                                                               'Galerkin, '
                                                                               'per-cell '
                                                                               'CellDiameter '
                                                                               'SUPG '
                                                                               'tau, '
                                                                               'and '
                                                                               'fixed '
                                                                               'global '
                                                                               'tau',
                                                                               'Advection '
                                                                               'boundary '
                                                                               'layer '
                                                                               'kappa=1e-3, '
                                                                               'N=8..512, '
                                                                               'Galerkin '
                                                                               'vs '
                                                                               'SUPG '
                                                                               'undershoot',
                                                                               'Biharmonic '
                                                                               'C0-IP '
                                                                               'u=sin(pi '
                                                                               'x)sin(pi '
                                                                               'y) '
                                                                               '(simply '
                                                                               'supported), '
                                                                               'P2/P3, '
                                                                               'alpha '
                                                                               'in '
                                                                               '{4(k+1)^2, '
                                                                               '8, 1, '
                                                                               '0.1, '
                                                                               '0.01, '
                                                                               '1e-4, '
                                                                               '1e-6}, '
                                                                               'plus '
                                                                               'hard-coded '
                                                                               'vs '
                                                                               'CellDiameter '
                                                                               'h, '
                                                                               'N=8..128',
                                                                               'Helmholtz '
                                                                               'u=sin(kx/sqrt2)sin(ky/sqrt2), '
                                                                               'P1, '
                                                                               'non-resonant '
                                                                               'k in '
                                                                               '{12.17, '
                                                                               '27.57, '
                                                                               '54.55}, '
                                                                               'N=16..256',
                                                                               'Elasticity '
                                                                               'cantilever: '
                                                                               'plane-strain '
                                                                               'vs '
                                                                               'plane-stress '
                                                                               'tip '
                                                                               'deflection '
                                                                               'at '
                                                                               'nu=0.3/0.45; '
                                                                               'volumetric '
                                                                               'locking '
                                                                               'P1 vs '
                                                                               'P2 vs '
                                                                               'P2/P1 '
                                                                               'Taylor-Hood '
                                                                               'at '
                                                                               'nu=0.49/0.499/0.4999 '
                                                                               'on '
                                                                               'four '
                                                                               'meshes',
                                                                               'Stokes: '
                                                                               'SVD '
                                                                               'null-space '
                                                                               'dimension '
                                                                               'of the '
                                                                               'bc-applied '
                                                                               'saddle-point '
                                                                               'matrix, '
                                                                               'Taylor-Hood '
                                                                               'vs '
                                                                               'equal-order '
                                                                               'P1/P1, '
                                                                               '8x8'],
                                                           'generator_execution': 'All '
                                                                                  '35 '
                                                                                  '(physics, '
                                                                                  'variant) '
                                                                                  'pairs '
                                                                                  'exposed '
                                                                                  'by '
                                                                                  'src/backends/fenics/generators '
                                                                                  'were '
                                                                                  'generated '
                                                                                  'and '
                                                                                  'RUN '
                                                                                  'on '
                                                                                  'the '
                                                                                  'installed '
                                                                                  'dolfinx; '
                                                                                  'all '
                                                                                  '35 '
                                                                                  'exit '
                                                                                  'with '
                                                                                  'return '
                                                                                  'code '
                                                                                  '0. '
                                                                                  'RETURN '
                                                                                  'CODE '
                                                                                  'WAS '
                                                                                  'THE '
                                                                                  'ONLY '
                                                                                  'CRITERION '
                                                                                  'APPLIED '
                                                                                  'IN '
                                                                                  'THAT '
                                                                                  'PASS '
                                                                                  '— '
                                                                                  'the '
                                                                                  'outputs '
                                                                                  'were '
                                                                                  'not '
                                                                                  'checked '
                                                                                  'for '
                                                                                  'physical '
                                                                                  'sanity, '
                                                                                  'and '
                                                                                  'a '
                                                                                  '2026-08-03 '
                                                                                  'audit '
                                                                                  're-run '
                                                                                  'found '
                                                                                  'THREE '
                                                                                  'templates '
                                                                                  'that '
                                                                                  'exit '
                                                                                  '0 '
                                                                                  'while '
                                                                                  'producing '
                                                                                  'wrong '
                                                                                  'numbers. '
                                                                                  'Do '
                                                                                  'not '
                                                                                  'treat '
                                                                                  "'rc=0' "
                                                                                  'from '
                                                                                  'this '
                                                                                  'catalog '
                                                                                  'as '
                                                                                  'evidence '
                                                                                  'that '
                                                                                  'a '
                                                                                  'template '
                                                                                  'is '
                                                                                  'correct:\n'
                                                                                  '  - '
                                                                                  'dg_methods/2d: '
                                                                                  'prints '
                                                                                  "'DG "
                                                                                  'advection-diffusion '
                                                                                  "solved' "
                                                                                  'and '
                                                                                  'u: '
                                                                                  'min=inf, '
                                                                                  'max=inf. '
                                                                                  'The '
                                                                                  'raw '
                                                                                  'PETSc '
                                                                                  'KSP '
                                                                                  'returns '
                                                                                  'converged '
                                                                                  'reason '
                                                                                  '-11 '
                                                                                  '(KSP_DIVERGED_PC_FAILED) '
                                                                                  'and '
                                                                                  'the '
                                                                                  'script '
                                                                                  'never '
                                                                                  'inspects '
                                                                                  'it.\n'
                                                                                  '  - '
                                                                                  'mixed_poisson/2d: '
                                                                                  'imposes '
                                                                                  'sigma.n '
                                                                                  '= 0 '
                                                                                  'on '
                                                                                  'the '
                                                                                  'ENTIRE '
                                                                                  'boundary, '
                                                                                  'so '
                                                                                  'the '
                                                                                  'pressure '
                                                                                  'is '
                                                                                  'determined '
                                                                                  'only '
                                                                                  'up '
                                                                                  'to '
                                                                                  'a '
                                                                                  'constant '
                                                                                  'and '
                                                                                  'no '
                                                                                  'nullspace '
                                                                                  'is '
                                                                                  'attached; '
                                                                                  'it '
                                                                                  'reports '
                                                                                  'min(p) '
                                                                                  '= '
                                                                                  'max(p) '
                                                                                  '= '
                                                                                  '-1.035309e+13 '
                                                                                  '(KSP '
                                                                                  'reason '
                                                                                  '4, '
                                                                                  'LU '
                                                                                  'on '
                                                                                  'a '
                                                                                  'singular '
                                                                                  'saddle-point '
                                                                                  'system).\n'
                                                                                  '  - '
                                                                                  'eigenvalue/2d: '
                                                                                  'assembles '
                                                                                  'BOTH '
                                                                                  'A '
                                                                                  'and '
                                                                                  'the '
                                                                                  'mass '
                                                                                  'matrix '
                                                                                  'M '
                                                                                  'with '
                                                                                  'the '
                                                                                  'same '
                                                                                  'Dirichlet '
                                                                                  'diagonal '
                                                                                  '(diag=1.0) '
                                                                                  'and '
                                                                                  'asks '
                                                                                  'for '
                                                                                  'SMALLEST_REAL, '
                                                                                  'so '
                                                                                  'the '
                                                                                  'two '
                                                                                  'lowest '
                                                                                  'reported '
                                                                                  'eigenvalues '
                                                                                  'are '
                                                                                  'the '
                                                                                  'spurious '
                                                                                  'constraint '
                                                                                  'modes '
                                                                                  '1.0000000000003 '
                                                                                  'and '
                                                                                  '1.0000000000031; '
                                                                                  'the '
                                                                                  'true '
                                                                                  'fundamental '
                                                                                  '19.79 '
                                                                                  'is '
                                                                                  'third. '
                                                                                  'This '
                                                                                  'is '
                                                                                  'exactly '
                                                                                  'the '
                                                                                  'failure '
                                                                                  'documented '
                                                                                  'in '
                                                                                  'the '
                                                                                  'eigenvalue '
                                                                                  'pitfall '
                                                                                  'of '
                                                                                  'this '
                                                                                  'catalog, '
                                                                                  'and '
                                                                                  'the '
                                                                                  "generator's "
                                                                                  'inline '
                                                                                  'comment '
                                                                                  'claiming '
                                                                                  'diag=0.0 '
                                                                                  'leaves '
                                                                                  'M '
                                                                                  'singular '
                                                                                  'is '
                                                                                  'wrong '
                                                                                  '— '
                                                                                  'assembling '
                                                                                  'M '
                                                                                  'with '
                                                                                  'diag=0.0, '
                                                                                  'or '
                                                                                  'with '
                                                                                  'bcs=[], '
                                                                                  'both '
                                                                                  'give '
                                                                                  'the '
                                                                                  'clean '
                                                                                  'spectrum '
                                                                                  '19.82 '
                                                                                  '/ '
                                                                                  '49.71 '
                                                                                  '/ '
                                                                                  '49.92 '
                                                                                  '/ '
                                                                                  '80.30.\n'
                                                                                  'Timing '
                                                                                  'note: '
                                                                                  'navier_stokes/3d '
                                                                                  'converges '
                                                                                  '(3 '
                                                                                  'Newton '
                                                                                  'iterations, '
                                                                                  '49072 '
                                                                                  'dofs). '
                                                                                  'The '
                                                                                  'earlier '
                                                                                  'claim '
                                                                                  'that '
                                                                                  'it '
                                                                                  "'needs "
                                                                                  'more '
                                                                                  'than '
                                                                                  '300 '
                                                                                  "s' "
                                                                                  'is '
                                                                                  'hardware-dependent '
                                                                                  'and '
                                                                                  'did '
                                                                                  'not '
                                                                                  'reproduce '
                                                                                  '— '
                                                                                  'it '
                                                                                  'completed '
                                                                                  'in '
                                                                                  '230 '
                                                                                  's '
                                                                                  'on '
                                                                                  'the '
                                                                                  'audit '
                                                                                  'machine. '
                                                                                  'Budget '
                                                                                  'by '
                                                                                  'measurement, '
                                                                                  'not '
                                                                                  'by '
                                                                                  'this '
                                                                                  'number.',
                                                           'corrections_landed': 'element_catalog '
                                                                                 '(CR '
                                                                                 'cells, '
                                                                                 'Hermite '
                                                                                 'api, '
                                                                                 'Bubble '
                                                                                 'hex '
                                                                                 'minimum, '
                                                                                 'chebyshev '
                                                                                 'variants, '
                                                                                 'iso '
                                                                                 'cells, '
                                                                                 'HHJ '
                                                                                 'cells, '
                                                                                 'pyramid '
                                                                                 'degree, '
                                                                                 "'CG' "
                                                                                 'vs '
                                                                                 "'P1'), "
                                                                                 'mesh_catalog '
                                                                                 '(gmshio '
                                                                                 'module '
                                                                                 'removed, '
                                                                                 'refine '
                                                                                 'prerequisites '
                                                                                 'and '
                                                                                 'return '
                                                                                 'shapes, '
                                                                                 'vtkhdf '
                                                                                 'writing), '
                                                                                 'solver_catalog '
                                                                                 '(assemble_matrix_block/nest '
                                                                                 'removed, '
                                                                                 'bddc/fieldsplit/ams '
                                                                                 'need '
                                                                                 'extra '
                                                                                 'setup, '
                                                                                 'NewtonSolver '
                                                                                 'hard '
                                                                                 'failure), '
                                                                                 'boundary_conditions '
                                                                                 '(connectivity '
                                                                                 'scope, '
                                                                                 'BC '
                                                                                 'rank-mismatch '
                                                                                 'message, '
                                                                                 'DG '
                                                                                 'no-op), '
                                                                                 'io_catalog '
                                                                                 '(VTX '
                                                                                 'exact '
                                                                                 'message), '
                                                                                 'poisson '
                                                                                 '(VTX '
                                                                                 'message, '
                                                                                 'complex '
                                                                                 'messages, '
                                                                                 'exterior_facet_indices '
                                                                                 'exception), '
                                                                                 'linear_elasticity '
                                                                                 '(plane-stress '
                                                                                 'factor, '
                                                                                 'locking '
                                                                                 'magnitude, '
                                                                                 "'CG' "
                                                                                 'family, '
                                                                                 'BC '
                                                                                 'message), '
                                                                                 'stokes '
                                                                                 '(nullspace '
                                                                                 'helper '
                                                                                 'does '
                                                                                 'not '
                                                                                 'exist), '
                                                                                 'convection_diffusion '
                                                                                 '(oscillation '
                                                                                 'damping, '
                                                                                 'constant-tau '
                                                                                 'stagnation), '
                                                                                 'biharmonic '
                                                                                 '(alpha '
                                                                                 'behaviour, '
                                                                                 'hard-coded '
                                                                                 'h, '
                                                                                 'no '
                                                                                 'H2 '
                                                                                 'NotImplementedError), '
                                                                                 'helmholtz '
                                                                                 '(no '
                                                                                 'convergence '
                                                                                 'at '
                                                                                 'all '
                                                                                 'for '
                                                                                 'k*h>1), '
                                                                                 'nearly_incompressible_elasticity '
                                                                                 '(1/(1-2nu) '
                                                                                 'rule), '
                                                                                 'hyperelasticity '
                                                                                 '(NewtonSolver-era '
                                                                                 'signals, '
                                                                                 'P2 '
                                                                                 'locking '
                                                                                 'magnitude), '
                                                                                 'maxwell '
                                                                                 '(complex '
                                                                                 'message), '
                                                                                 'parallel_computing '
                                                                                 '(assemble_scalar '
                                                                                 'is '
                                                                                 'not '
                                                                                 'collective), '
                                                                                 'plus '
                                                                                 'the '
                                                                                 'generator-level '
                                                                                 'catalogs '
                                                                                 'for '
                                                                                 'dg_methods, '
                                                                                 'mixed_poisson, '
                                                                                 'nonlinear_pde '
                                                                                 'and '
                                                                                 'magnetostatics.',
                                                           'not_re-run': 'Claims '
                                                                         'requiring '
                                                                         'hardware/scale '
                                                                         'or absent '
                                                                         'packages '
                                                                         'were left '
                                                                         'with their '
                                                                         'inherited '
                                                                         'tag and NOT '
                                                                         'upgraded: '
                                                                         'fieldsplit '
                                                                         'iteration-count '
                                                                         'scaling '
                                                                         'beyond ~100k '
                                                                         'dofs, GAMG '
                                                                         'near-nullspace '
                                                                         'benefit at '
                                                                         'large scale, '
                                                                         'complex-PETSc '
                                                                         'behaviour '
                                                                         '(this build '
                                                                         'is real), '
                                                                         'dolfinx_mpc '
                                                                         '/ '
                                                                         'adios4dolfinx '
                                                                         '/ pyamg '
                                                                         'behaviour '
                                                                         '(not '
                                                                         'installed), '
                                                                         'demo_catalog '
                                                                         'and '
                                                                         'tutorial_catalog '
                                                                         'URLs '
                                                                         '(documentation '
                                                                         'links, not '
                                                                         'executed), '
                                                                         'and the '
                                                                         'hyperelastic '
                                                                         'Cook-membrane '
                                                                         'locking '
                                                                         'figure (only '
                                                                         'its linear '
                                                                         'analogue was '
                                                                         'measured).'},
                 '2026-08-03_pass2_eight_untouched_topics': {'scope': 'The eight '
                                                                      'physics topics '
                                                                      'that the '
                                                                      '2026-08-03 '
                                                                      'adversarial '
                                                                      'audit found had '
                                                                      'ZERO of their '
                                                                      'pitfalls ever '
                                                                      'executed — '
                                                                      'heat, '
                                                                      'navier_stokes, '
                                                                      'cahn_hilliard, '
                                                                      'reaction_diffusion, '
                                                                      'contact, '
                                                                      'fracture, '
                                                                      'stokes_darcy, '
                                                                      'thermal_structural '
                                                                      '— plus the five '
                                                                      'generator-tier '
                                                                      'catalogs that '
                                                                      'actually reach '
                                                                      'an agent '
                                                                      '(matrix_free_poisson, '
                                                                      'multiphase, '
                                                                      'time_dependent_heat, '
                                                                      'nonlinear_pde, '
                                                                      'magnetostatics), '
                                                                      'plus a '
                                                                      'stale-API sweep '
                                                                      'of the rest of '
                                                                      'the catalog.',
                                                             'environment': 'dolfinx '
                                                                            '0.10.0, '
                                                                            'basix '
                                                                            '0.10.0, '
                                                                            'ufl '
                                                                            '2025.2.1, '
                                                                            'petsc4py '
                                                                            '3.24.4 '
                                                                            '(REAL '
                                                                            'scalars), '
                                                                            'slepc4py '
                                                                            '3.24.3, '
                                                                            'mpi4py '
                                                                            '4.1.1, '
                                                                            'Python '
                                                                            '3.12 '
                                                                            'conda-forge, '
                                                                            'Linux '
                                                                            'x86_64. '
                                                                            'NOT '
                                                                            'installed: '
                                                                            'dolfinx_mpc, '
                                                                            'adios4dolfinx, '
                                                                            'pyamg, '
                                                                            'dolfinx_contact, '
                                                                            'phasefieldx, '
                                                                            'SUNDIALS '
                                                                            'bindings '
                                                                            '— claims '
                                                                            'about '
                                                                            'those '
                                                                            'were '
                                                                            'deleted, '
                                                                            'not '
                                                                            'downgraded.',
                                                             'method': 'Every kept '
                                                                       'claim comes '
                                                                       'from a run '
                                                                       'executed in '
                                                                       'this '
                                                                       'environment. '
                                                                       'For each '
                                                                       'pitfall the '
                                                                       'WRONG variant '
                                                                       'was written '
                                                                       'and run, and '
                                                                       'the Signal '
                                                                       'quotes the '
                                                                       'text that '
                                                                       'variant '
                                                                       'actually '
                                                                       'emitted; where '
                                                                       'the previously '
                                                                       'catalogued '
                                                                       'text did not '
                                                                       'reproduce, the '
                                                                       'entry now says '
                                                                       'so and quotes '
                                                                       'the real text. '
                                                                       'Claims that '
                                                                       'could not be '
                                                                       'executed were '
                                                                       'DELETED rather '
                                                                       'than carried '
                                                                       'forward with '
                                                                       "an 'inherited' "
                                                                       'tag. Each '
                                                                       'topic gained a '
                                                                       'complete '
                                                                       'runnable '
                                                                       'minimal_working_example '
                                                                       'that was '
                                                                       'extracted back '
                                                                       'out of this '
                                                                       'catalog and '
                                                                       'run '
                                                                       'unmodified, '
                                                                       'and REQUIRED / '
                                                                       'OPTIONAL '
                                                                       'blocks for '
                                                                       'function '
                                                                       'space, weak '
                                                                       'form, boundary '
                                                                       'conditions and '
                                                                       'solver.',
                                                             'self_correction': 'An '
                                                                                'adversarial '
                                                                                'critic '
                                                                                'was '
                                                                                'run '
                                                                                'against '
                                                                                'the '
                                                                                'load-bearing '
                                                                                'claims '
                                                                                'and '
                                                                                'falsified '
                                                                                'several '
                                                                                'of '
                                                                                'them; '
                                                                                'the '
                                                                                'corrections '
                                                                                'are '
                                                                                'in '
                                                                                'the '
                                                                                'entries. '
                                                                                'Notably: '
                                                                                'the '
                                                                                'eigenvalue '
                                                                                'recipe '
                                                                                "'assemble "
                                                                                'the '
                                                                                'mass '
                                                                                'matrix '
                                                                                'with '
                                                                                "bcs=[]' "
                                                                                'is '
                                                                                'NOT '
                                                                                'clean '
                                                                                'for a '
                                                                                'consistent '
                                                                                'mass '
                                                                                'matrix '
                                                                                '(it '
                                                                                'solves '
                                                                                'a '
                                                                                'different '
                                                                                'pencil) '
                                                                                'and '
                                                                                'IS '
                                                                                'clean '
                                                                                'for a '
                                                                                'lumped '
                                                                                'one; '
                                                                                'the '
                                                                                'DG '
                                                                                'boundary '
                                                                                'defect '
                                                                                'is '
                                                                                'fixed '
                                                                                'structurally '
                                                                                'by '
                                                                                'the '
                                                                                'outflow '
                                                                                'restriction '
                                                                                'but '
                                                                                'is '
                                                                                'also '
                                                                                'masked '
                                                                                'by '
                                                                                'the '
                                                                                'Nitsche '
                                                                                'block '
                                                                                'whenever '
                                                                                'the '
                                                                                'diffusivity '
                                                                                'is '
                                                                                'non-zero, '
                                                                                'so a '
                                                                                'form '
                                                                                'carrying '
                                                                                'both '
                                                                                'changes '
                                                                                'cannot '
                                                                                'show '
                                                                                'which '
                                                                                'mattered; '
                                                                                'a '
                                                                                'fixed '
                                                                                'SIPG '
                                                                                'penalty '
                                                                                'of 10 '
                                                                                'loses '
                                                                                'coercivity '
                                                                                'at DG '
                                                                                'degree '
                                                                                '3 '
                                                                                'while '
                                                                                'the '
                                                                                'KSP '
                                                                                'still '
                                                                                'reports '
                                                                                'CONVERGED; '
                                                                                'and '
                                                                                'superlu_dist '
                                                                                'can '
                                                                                'return '
                                                                                'KSPConvergedReason '
                                                                                '4 on '
                                                                                'the '
                                                                                'mixed-Poisson '
                                                                                'saddle '
                                                                                'point '
                                                                                'together '
                                                                                'with '
                                                                                'a '
                                                                                'mass-balance '
                                                                                'residual '
                                                                                'of '
                                                                                'order '
                                                                                '1, so '
                                                                                'the '
                                                                                'converged '
                                                                                'reason '
                                                                                'alone '
                                                                                'is '
                                                                                'not a '
                                                                                'correctness '
                                                                                'check.',
                                                             'verification_style': 'Physical '
                                                                                   'self-checks '
                                                                                   'that '
                                                                                   'need '
                                                                                   'NO '
                                                                                   'reference '
                                                                                   'solution '
                                                                                   'were '
                                                                                   'preferred '
                                                                                   'and '
                                                                                   'are '
                                                                                   'built '
                                                                                   'into '
                                                                                   'the '
                                                                                   'shipped '
                                                                                   'templates: '
                                                                                   'eigenpair '
                                                                                   'residual '
                                                                                   '||A '
                                                                                   'x '
                                                                                   '- '
                                                                                   'lambda '
                                                                                   'M '
                                                                                   'x|| '
                                                                                   '/ '
                                                                                   '||x||, '
                                                                                   'cellwise '
                                                                                   'mass '
                                                                                   'balance '
                                                                                   '||div(sigma_h) '
                                                                                   '- '
                                                                                   'f|| '
                                                                                   '/ '
                                                                                   '||f||, '
                                                                                   'closed '
                                                                                   'energy '
                                                                                   'balance, '
                                                                                   'stoichiometric '
                                                                                   'conservation, '
                                                                                   'free-energy '
                                                                                   'monotonicity, '
                                                                                   'KSP/SNES '
                                                                                   'converged '
                                                                                   'reason, '
                                                                                   'finiteness '
                                                                                   'AND '
                                                                                   'magnitude. '
                                                                                   'Return '
                                                                                   'code '
                                                                                   '0 '
                                                                                   'is '
                                                                                   'treated '
                                                                                   'as '
                                                                                   'no '
                                                                                   'evidence '
                                                                                   'at '
                                                                                   'all.',
                                                             'not_verified': 'Anything '
                                                                             'needing '
                                                                             'hardware/scale '
                                                                             'or an '
                                                                             'absent '
                                                                             'package: '
                                                                             'fieldsplit '
                                                                             'iteration-count '
                                                                             'scaling '
                                                                             'beyond '
                                                                             '~100k '
                                                                             'dofs, '
                                                                             'GAMG '
                                                                             'near-nullspace '
                                                                             'benefit '
                                                                             'at large '
                                                                             'scale, '
                                                                             'complex-PETSc '
                                                                             'behaviour '
                                                                             '(this '
                                                                             'build is '
                                                                             'real), '
                                                                             'dolfinx_mpc '
                                                                             '/ '
                                                                             'adios4dolfinx '
                                                                             '/ pyamg '
                                                                             '/ '
                                                                             'dolfinx_contact '
                                                                             '/ '
                                                                             'phasefieldx '
                                                                             'behaviour, '
                                                                             'self-contact, '
                                                                             'monolithic '
                                                                             'phase-field '
                                                                             'Newton, '
                                                                             'and the '
                                                                             'demo/tutorial '
                                                                             'URL '
                                                                             'catalogs '
                                                                             '(documentation '
                                                                             'links, '
                                                                             'not '
                                                                             'executed).'}},
 'multiphysics_submeshes': {'description': 'Solving PDEs on subdomains with different '
                                           'physics using DOLFINx submeshes (0.10+ '
                                           'feature).',
                            'demo_url': 'https://jsdokken.com/FEniCS-workshop/src/multiphysics/submeshes.html',
                            'approach': {'create_submesh': 'Extract subdomain mesh '
                                                           'from parent mesh',
                                         'restriction': 'Integration over subdomains '
                                                        'using measures dx(marker)',
                                         'coupling': 'Transfer data between submeshes '
                                                     'via interpolation or shared '
                                                     'DOFs'},
                            'use_cases': 'Different materials, different physics '
                                         '(FSI), domain decomposition'},
 'optimal_control': {'description': 'PDE-constrained optimization and adjoint methods '
                                    'in FEniCSx.',
                     'demo_url': 'https://jsdokken.com/FEniCS-workshop/src/applications/optimal_control.html',
                     'approach': {'derive_adjoint': 'Use UFL adjoint() and action() to '
                                                    'derive adjoint PDE',
                                  'interface_scipy': 'Extract gradient via adjoint '
                                                     'solve, pass to scipy.optimize '
                                                     'for minimization',
                                  'dolfin_adjoint': 'Algorithmic differentiation tool '
                                                    '(github.com/dolfin-adjoint/dolfin-adjoint) '
                                                    '— automatic tape-based AD'},
                     'use_cases': 'Shape optimization, topology optimization, '
                                  'parameter estimation, inverse problems'},
 'complex_valued': {'description': 'Solving PDEs with complex-valued solutions in '
                                   'DOLFINx (Helmholtz, Maxwell, wave scattering).',
                    'demo_url': 'https://jsdokken.com/dolfinx-tutorial/chapter1/complex_mode.html',
                    'scalar_types': {'float32': 'Single precision real',
                                     'float64': 'Double precision real (default)',
                                     'complex64': 'Single precision complex',
                                     'complex128': 'Double precision complex'},
                    'api': 'dolfinx.default_scalar_type — check/switch between '
                           'real/complex builds',
                    'demo_types_url': 'https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_types.html',
                    'pitfalls': ['PETSc must be compiled with '
                                 '--with-scalar-type=complex for complex problems',
                                 'The PETSc scalar type is a build-time choice (a real '
                                 'petsc4py cannot assemble or solve complex forms '
                                 'through dolfinx.fem.petsc), but DOLFINx itself can '
                                 'compile and assemble float32/float64/complex64/'
                                 'complex128 forms and Functions in the same session '
                                 'via dtype=..., solving with scipy (see '
                                 'demo_types.py)',
                                 'Some solvers (CG) do not work with complex '
                                 'arithmetic — use GMRES',
                                 'inner(a,b) in UFL conjugates the second argument for '
                                 'complex-valued problems']},
 'parallel_computing': {'description': 'MPI-based parallel computing in DOLFINx. '
                                       'First-class parallel from ground up.',
                        'api': {'communicator': 'All mesh/solver creation takes '
                                                'MPI.COMM_WORLD (or sub-communicator)',
                                'run': 'mpirun -np N python script.py',
                                'partitioning': 'Automatic mesh partitioning on '
                                                'creation (configurable partitioner)',
                                'assembly': 'dolfinx.fem.assemble_scalar() returns the '
                                            'RANK-LOCAL contribution ONLY — it does '
                                            'NOT reduce across ranks. You MUST wrap '
                                            'it: '
                                            'comm.allreduce(fem.assemble_scalar(fem.form(M)), '
                                            'op=MPI.SUM). (Verified empirically '
                                            '2026-08-03 on 2 ranks: 0.505859 + '
                                            '0.494141 for a functional whose true '
                                            'value is 1.0. Any error norm / integral '
                                            'computed without the allreduce is '
                                            'silently wrong in parallel.)'},
                        'performance': {'scaling': 'Strong and weak scaling '
                                                   'demonstrated up to thousands of '
                                                   'cores',
                                        'mesh_partitioning': 'Graph-based (ParMETIS, '
                                                             'SCOTCH, or KaHIP) for '
                                                             'load balancing',
                                        'ghost_layer': 'DOLFINx manages ghost '
                                                       'cells/DOFs automatically',
                                        'neighbourhood_collectives': 'MPI '
                                                                     'Neighbourhood '
                                                                     'collectives for '
                                                                     'efficient halo '
                                                                     'exchange'},
                        'pitfalls': ['[Numerical] fem.assemble_scalar is NOT '
                                     "collective — it returns each rank's local piece, "
                                     'so forgetting comm.allreduce(..., op=MPI.SUM) '
                                     'makes every L2 error, energy and volume integral '
                                     'wrong in parallel while looking perfectly '
                                     'plausible in serial. Signal: running '
                                     'fem.assemble_scalar(fem.form(1.0*ufl.dx(domain=msh))) '
                                     'on a 16x16 unit square under mpirun -np 2 '
                                     'returns 0.505859 on rank 0 and 0.494141 on rank '
                                     "1 instead of 1.0 on both; dolfinx's own "
                                     "docstring states 'The returned value is local "
                                     "and not accumulated across processes.' (Verified "
                                     'empirically 2026-08-03.)',
                                     'MUST use MPI communicator consistently — do not '
                                     'mix serial and parallel operations',
                                     'Output: only rank 0 should print; use if '
                                     'MPI.COMM_WORLD.rank == 0:',
                                     'Some operations (e.g., Gmsh model creation) '
                                     'should be done on rank 0 only',
                                     'pyamg is serial-only — use PETSc AMG for '
                                     'parallel',
                                     'Function evaluation at points requires parallel '
                                     'geometric search (BoundingBoxTree)']},
 'api_changes': {'description': 'Critical API changes between DOLFINx versions. '
                                'Essential for writing version-portable code.',
                 '0_9_to_0_10': {'NewtonSolver_deprecated': 'dolfinx.nls.petsc.NewtonSolver '
                                                            'deprecated -> use '
                                                            'dolfinx.fem.petsc.NonlinearProblem '
                                                            'wrapping PETSc SNES '
                                                            'directly',
                                 'gmsh_module_renamed': 'dolfinx.io.gmshio -> '
                                                        'dolfinx.io.gmsh (module '
                                                        'rename)',
                                 'gmsh_returns_MeshData': 'model_to_mesh() returns '
                                                          'MeshData dataclass (with '
                                                          'cell_tags, facet_tags by '
                                                          'codimension) instead of '
                                                          'tuple',
                                 'LinearProblem_blocked': 'dolfinx.fem.petsc.LinearProblem '
                                                          'now supports blocked '
                                                          "problems (kind='mpi' or "
                                                          "kind='nest')",
                                 'ZeroBaseForm': 'ufl.ZeroBaseForm removes need for '
                                                 'dummy 0*v*dx to compile empty forms',
                                 'uniform_refine': 'dolfinx.mesh.uniform_refine() '
                                                   'added (2D/3D meshes only - intervals '
                                                   "raise 'Uniform refinement only for 2D "
                                                   "and 3D meshes'; triangle, "
                                                   'quadrilateral, tetrahedron, hexahedron '
                                                   'and prism work; pyramids need '
                                                   'tetrahedra in the same mesh; call '
                                                   'create_entities(1), plus '
                                                   'create_entities(2) for '
                                                   'hex/prism/pyramid meshes)',
                                 'vtkhdf_reader': 'dolfinx.io.vtkhdf.read_mesh() added '
                                                  "(Kitware's next-gen format)",
                                 'branching_meshes': 'T-joints (3+ cells per facet) '
                                                     'now supported as input meshes'},
                 '0_7_to_0_8': {'basix_ufl_element': 'Use basix.ufl.element() instead '
                                                     'of ufl.FiniteElement()',
                                'mixed_element': 'Use basix.ufl.mixed_element() '
                                                 'instead of ufl.MixedElement()',
                                'blocked_element': 'Use basix.ufl.blocked_element() '
                                                   'for vector/tensor elements',
                                'functionspace': 'fem.functionspace() (lowercase) '
                                                 'replaces fem.FunctionSpace()'},
                 '0_9_to_0_10_MEASURED_ADDITIONS': {'gmshio_module_removed': '`import '
                                                                             'dolfinx.io.gmshio` '
                                                                             'now '
                                                                             'raises '
                                                                             'ModuleNotFoundError '
                                                                             '— it is '
                                                                             'not an '
                                                                             'alias, '
                                                                             'the '
                                                                             'module '
                                                                             'is gone. '
                                                                             'Use '
                                                                             'dolfinx.io.gmsh.',
                                                    'assemble_matrix_block_nest_removed': 'dolfinx.fem.petsc.assemble_matrix_block '
                                                                                          '/ '
                                                                                          'assemble_matrix_nest '
                                                                                          'no '
                                                                                          'longer '
                                                                                          'exist; '
                                                                                          'use '
                                                                                          'assemble_matrix(..., '
                                                                                          "kind='mpi'|'nest') "
                                                                                          'or '
                                                                                          'LinearProblem(..., '
                                                                                          'kind=...).',
                                                    'NewtonSolver_hard_break': 'dolfinx.nls.petsc.NewtonSolver '
                                                                               'still '
                                                                               'imports '
                                                                               'and '
                                                                               'warns, '
                                                                               'but '
                                                                               'wrapping '
                                                                               'a 0.10 '
                                                                               'NonlinearProblem '
                                                                               'raises '
                                                                               'AttributeError: '
                                                                               "'NonlinearProblem' "
                                                                               'object '
                                                                               'has no '
                                                                               'attribute '
                                                                               "'a'. "
                                                                               'The '
                                                                               '0.9 '
                                                                               'two-step '
                                                                               'Newton '
                                                                               'pattern '
                                                                               'is '
                                                                               'dead '
                                                                               'code, '
                                                                               'not '
                                                                               'merely '
                                                                               'deprecated.',
                                                    'interpolation_points_removed': 'basix.ufl '
                                                                                    'elements '
                                                                                    'have '
                                                                                    'no '
                                                                                    '`interpolation_points` '
                                                                                    'attribute '
                                                                                    'at '
                                                                                    'all '
                                                                                    '(neither '
                                                                                    'property '
                                                                                    'nor '
                                                                                    'method) '
                                                                                    '— '
                                                                                    'AttributeError: '
                                                                                    "'_BasixElement' "
                                                                                    'object '
                                                                                    'has '
                                                                                    'no '
                                                                                    'attribute '
                                                                                    "'interpolation_points'. "
                                                                                    'The '
                                                                                    'points '
                                                                                    'are '
                                                                                    'element.basix_element.points.',
                                                    'refine_needs_entities': 'dolfinx.mesh.uniform_refine '
                                                                             '/ refine '
                                                                             'require '
                                                                             'mesh.topology.create_entities(1) '
                                                                             'first, '
                                                                             'and '
                                                                             'refine '
                                                                             'returns '
                                                                             'a '
                                                                             '3-tuple '
                                                                             '(Mesh, '
                                                                             'parent_cells, '
                                                                             'parent_facets).',
                                                    'create_form_signature': 'dolfinx.fem.create_form(form, '
                                                                             'function_spaces, '
                                                                             'msh, '
                                                                             'subdomains, '
                                                                             'coefficient_map, '
                                                                             'constant_map, '
                                                                             'entity_maps=None) '
                                                                             '— no '
                                                                             'parent_mesh= '
                                                                             '/ '
                                                                             'coefficients= '
                                                                             '/ '
                                                                             'constants= '
                                                                             'kwargs.',
                                                    'vtkhdf_writing': 'dolfinx.io.vtkhdf '
                                                                      'now exports '
                                                                      'write_mesh / '
                                                                      'write_point_data '
                                                                      '/ '
                                                                      'write_cell_data, '
                                                                      'not just '
                                                                      'read_mesh.'},
                 'pitfalls': ['Online tutorials may use old API (ufl.FiniteElement, '
                              'FunctionSpace) — translate to new API. '
                              'ufl.FiniteElement / VectorElement / MixedElement are '
                              'REMOVED, so old scripts fail with AttributeError on the '
                              'ufl module rather than a deprecation warning (verified '
                              '2026-08-03)',
                              'The jsdokken tutorial is updated for latest version — '
                              'use it as primary reference',
                              'DOLFINx version in Docker images may differ from pip '
                              'install — check dolfinx.__version__',
                              'There is no errornorm helper in dolfinx or ufl — '
                              'assemble inner(uh-uex, uh-uex)*dx yourself, and '
                              'remember to comm.allreduce the result in parallel '
                              '(verified 2026-08-03)']},
 'demo_catalog': {'description': 'Complete catalog of official DOLFINx demos '
                                 '(docs.fenicsproject.org/dolfinx/main/python/demos.html).',
                  'demos': {'demo_poisson': 'Poisson equation — fundamental elliptic '
                                            'PDE',
                            'demo_mixed-poisson': 'Mixed Poisson with Raviart-Thomas '
                                                  'elements and block preconditioner',
                            'demo_stokes': 'Stokes equations with Taylor-Hood elements',
                            'demo_navier-stokes': 'Divergence-conforming DG for '
                                                  'Navier-Stokes',
                            'demo_elasticity': 'Linear elasticity with algebraic '
                                               'multigrid (GAMG)',
                            'demo_static-condensation': 'Static condensation of mixed '
                                                        "elasticity (Cook's membrane)",
                            'demo_cahn-hilliard': 'Cahn-Hilliard phase-field equation '
                                                  '(spinodal decomposition)',
                            'demo_biharmonic': 'Biharmonic equation with interior '
                                               'penalty DG',
                            'demo_helmholtz': 'Helmholtz equation (complex-valued)',
                            'demo_scattering_boundary_conditions': 'EM scattering from '
                                                                   'wire (scattering '
                                                                   'BCs)',
                            'demo_pml': 'EM scattering from wire (perfectly matched '
                                        'layer)',
                            'demo_half_loaded_waveguide': 'Electromagnetic modal '
                                                          'analysis (SLEPc eigenvalue)',
                            'demo_axis': 'Axisymmetric EM scattering from sphere',
                            'demo_poisson_matrix_free': 'Matrix-free CG solver for '
                                                        'Poisson',
                            'demo_types': 'Solving PDEs with different scalar types '
                                          '(float32/64, complex64/128)',
                            'demo_lagrange_variants': 'Lagrange element variants '
                                                      '(equispaced, GLL, Chebyshev)',
                            'demo_gmsh': 'Mesh generation with Gmsh integration',
                            'demo_interpolation-io': 'Interpolation and I/O operations',
                            'demo_pyvista': 'Visualization with PyVista',
                            'demo_pyamg': 'Poisson and elasticity with pyamg (serial '
                                          'AMG)'}},
 'tutorial_catalog': {'description': 'Complete catalog of '
                                     'jsdokken.com/dolfinx-tutorial chapters.',
                      'chapter1_fundamentals': {'fundamentals': 'Solving the Poisson '
                                                                'equation — basic '
                                                                'FEniCSx workflow',
                                                'complex_mode': 'Poisson with complex '
                                                                'numbers'},
                      'chapter2_gallery': {'heat_equation': 'Transient heat equation '
                                                            '(backward Euler)',
                                           'diffusion_code': 'Diffusion of a Gaussian '
                                                             'function',
                                           'nonlinpoisson': 'Nonlinear Poisson (Newton '
                                                            'method)',
                                           'linearelasticity': 'Linear elasticity '
                                                               '(cantilever beam)',
                                           'hyperelasticity': 'Hyperelasticity '
                                                              '(Neo-Hookean beam '
                                                              'bending)',
                                           'navierstokes': 'Navier-Stokes theory (IPCS '
                                                           'splitting)',
                                           'ns_code1': 'Channel flow (Poiseuille, '
                                                       'IPCS)',
                                           'ns_code2': 'Flow past cylinder (DFG 2D-3 '
                                                       'benchmark)'},
                      'chapter3_bcs_subdomains': {'neumann_dirichlet': 'Combining '
                                                                       'Dirichlet and '
                                                                       'Neumann BCs',
                                                  'robin_neumann_dirichlet': 'Multiple '
                                                                             'Dirichlet, '
                                                                             'Neumann, '
                                                                             'and '
                                                                             'Robin '
                                                                             'conditions',
                                                  'multiple_dirichlet': 'Setting '
                                                                        'multiple '
                                                                        'Dirichlet '
                                                                        'conditions',
                                                  'component_bc': 'Component-wise '
                                                                  'Dirichlet BC '
                                                                  '(vector problems)',
                                                  'subdomains': 'Defining subdomains '
                                                                'for different '
                                                                'materials',
                                                  'em': 'Electromagnetics example '
                                                        '(curl-curl with subdomains)'},
                      'chapter4_advanced': {'solvers': 'Solver configuration (PETSc '
                                                       'options)',
                                            'newton_solver': 'Custom Newton solver '
                                                             'implementation',
                                            'compiler_parameters': 'JIT options and '
                                                                   'visualization '
                                                                   '(Pandas)',
                                            'convergence': 'Error control — computing '
                                                           'convergence rates'},
                      'fenics_workshop': {'url': 'https://jsdokken.com/FEniCS-workshop/',
                                          'topics': 'UFL elements, form compilation, '
                                                    'advanced elements (Nedelec, RT), '
                                                    'mixed problems, '
                                                    'restriction/submeshes, optimal '
                                                    'control, multiphysics'}}}

# ═══════════════════════════════════════════════════════════════════════════════
# DEAL.II — COMPREHENSIVE DOMAIN KNOWLEDGE
# ═══════════════════════════════════════════════════════════════════════════════

_DEALII_KNOWLEDGE = {
    "poisson": {
        "description": "Poisson equation solved with deal.II (step-3/4/5). Foundation of all elliptic PDEs.",
        "tutorial_steps": {"step-3": "Basic Poisson on hyper_cube", "step-4": "Dim-independent with non-constant coefficients", "step-5": "Variable coefficient + grid read from disk (GridIn) + global refinement", "step-6": "Higher-order elements + automatic adaptivity", "step-7": "Helmholtz + convergence tables"},
        "function_space": "FE_Q<dim>(degree) — tensor-product Lagrange on quads/hexes",
        "element_catalog": {
            "FE_Q(1)": "Bilinear (2D) / trilinear (3D), standard choice",
            "FE_Q(2)": "Biquadratic, better accuracy for smooth solutions",
            "FE_SimplexP(1)": "Linear on triangles/tets (for simplex meshes)",
            "FE_DGQ(p)": "Discontinuous Galerkin variant",
        },
        "solver": {
            "small": "SolverCG + PreconditionSSOR (or PreconditionIdentity for debugging)",
            "medium": "SolverCG + SparseMIC (incomplete Cholesky)",
            "large": "SolverCG + TrilinosWrappers::PreconditionAMG (algebraic multigrid)",
            "matrix_free": "SolverCG + PreconditionChebyshev (step-37 pattern, fastest)",
        },
        "grid_generators": {
            "hyper_cube": "[0,1]^dim, all boundary_id=0 (use colorize=true for distinct IDs)",
            "hyper_rectangle": "Box [p1,p2]; colorize=false (default): all boundary_id=0 in 2D/3D (1D: left=0, right=1 always); colorize=true: x-=0,x+=1,y-=2,y+=3,z-=4,z+=5",
            "subdivided_hyper_rectangle": "Box with per-axis subdivision control",
            "hyper_ball": "Circular disk / ball with SphericalManifold",
            "hyper_shell": "Annulus / spherical shell (inner/outer radius)",
            "hyper_L": "L-shaped domain — classic corner singularity benchmark",
            "plate_with_a_hole": "Rectangle with cylindrical hole — stress concentration",
            "channel_with_cylinder": "Flow channel with obstacle — DFG benchmark geometry",
            "cheese": "Rectangle with square holes",
            "hyper_cube_slit": "Square with slit for singularity testing",
        },
        "output": "DataOut → VTU (standard), also VTK, gnuplot, SVG",
        "pitfalls": [
            "Call triangulation.refine_global() BEFORE distributing DOFs",
            "Boundary IDs on hyper_cube: ALL faces = 0 by default; use colorize=true (hyper_rectangle likewise gives all 0 in 2D/3D unless colorize=true)",
            "hyper_rectangle colorized (colorize=true): x-=0, x+=1, y-=2, y+=3, z-=4, z+=5",
            "Use DynamicSparsityPattern → copy_from → SparsityPattern (two-step)",
            "QGauss degree should be fe.degree + 1 for optimal convergence",
            "For Neumann-only: solution up to constant — need mean-value constraint",
            "Hanging node constraints MUST be applied on adaptively refined meshes (AffineConstraints)",
            "Forgetting update_values|update_gradients|update_JxW_values in FEValues → silent wrong results",
            "DataOut: must call build_patches() before writing",
        ],
    },
    "linear_elasticity": {
        "description": "Linear elasticity (step-8/17). Vector-valued FESystem with Lamé parameters.",
        "tutorial_steps": {
            "step-8": "Elasticity with FESystem, body forces, component-wise assembly",
            "step-17": "Parallel elasticity with PETSc",
            "step-18": "Quasi-static large-deformation (incremental loading, Lagrangian mesh)",
            "step-44": "Nonlinear solid mechanics — compressible Neo-Hookean, three-field formulation",
        },
        "function_space": "FESystem<dim>(FE_Q<dim>(1), dim) — vector Lagrange",
        "constitutive": {
            "lame": "mu = E/(2(1+nu)), lambda = E*nu/((1+nu)(1-2*nu))",
            "plane_stress": "lambda_star = 2*mu*lambda/(2*mu + lambda)",
        },
        "solver": {
            "small": "SolverCG + PreconditionSSOR",
            "large": "SolverCG + TrilinosWrappers::PreconditionAMG (provide rigid body modes for near-nullspace!)",
        },
        "pitfalls": [
            "Use system_to_component_index() to map local DOF to physical component",
            "For plane stress: use modified lambda_star = 2*mu*lambda/(2*mu + lambda)",
            "Near-incompressible (nu→0.5): MUST use mixed methods to avoid volumetric locking",
            "Providing rigid body modes to AMG dramatically improves convergence for elasticity",
            "Component mask needed for applying BC to individual displacement components",
            "VectorTools::interpolate_boundary_values needs ZeroFunction<dim>(dim) for vector BC",
            "Boundary IDs depend on GridGenerator — check docs for each generator",
        ],
    },
    "heat": {
        "description": "Heat equation — transient diffusion (step-26). Adaptive mesh in time.",
        "tutorial_steps": {
            "step-26": "Transient heat with adaptive mesh refinement, solution interpolation between meshes",
            "step-86": "Heat equation with PETSc time-stepping (TS) framework",
        },
        "function_space": "FE_Q<dim>(1) or FE_Q<dim>(2) — scalar Lagrange",
        "time_integration": "Backward Euler (stable) or Crank-Nicolson (2nd order, theta=0.5)",
        "solver": "SolverCG + PreconditionSSOR per time step",
        "pitfalls": [
            "Mass matrix assembly needed for transient terms",
            "When using adaptive refinement in time: MUST interpolate solution from old to new mesh",
            "Lumped mass matrix can introduce oscillations near steep gradients",
            "Initial condition via VectorTools::interpolate or VectorTools::project",
        ],
    },
    "stokes": {
        "description": "Stokes flow (step-22). Mixed FE with Schur complement preconditioning.",
        "tutorial_steps": {
            "step-22": "Stokes with block preconditioner, Schur complement",
            "step-45": "Parallel Stokes with periodic BCs using Trilinos",
            "step-55": "Parallel Stokes with AMG for velocity block",
            "step-56": "Stokes with geometric multigrid",
        },
        "function_space": "Taylor-Hood: FESystem(FE_Q<dim>(2)^dim, FE_Q<dim>(1)) — Q2/Q1",
        "solver": {
            "recommended": "SolverGMRES or SolverMinRes with block preconditioner",
            "block_precon": "AMG for velocity block + pressure mass matrix for Schur complement",
            "alternative_elements": "FE_BernardiRaugel + FE_DGP(0) for low-order stable pair",
        },
        "pitfalls": [
            "MUST use inf-sup stable pair — Q1/Q1 (equal-order) is UNSTABLE",
            "Taylor-Hood Q2/Q1 is the standard stable pair",
            "Pressure unique only up to constant for enclosed flows — pin one pressure DOF",
            "Schur complement preconditioning essential for efficiency at scale",
            "Pressure mass matrix is a good Schur complement approximation",
        ],
    },
    "navier_stokes": {
        "description": "Navier-Stokes (step-57). Nonlinear extension of Stokes with Newton iteration.",
        "tutorial_steps": {
            "step-57": "Stationary incompressible NS, Newton + continuation in Reynolds number",
            "step-35": "NS via projection/pressure-correction method (time-dependent)",
        },
        "function_space": "Same as Stokes: Taylor-Hood Q2/Q1",
        "solver": "Newton outer loop + direct solve (UMFPACK) per Newton step for small problems",
        "pitfalls": [
            "Newton convergence depends critically on initial guess — use continuation in Re",
            "Start from Stokes solution (Re→0) and gradually increase Re",
            "For Re > ~500, need very fine mesh or stabilization",
        ],
    },
    "advection_dg": {
        "description": "Advection (step-9: SUPG on continuous FE_Q; step-12: DG). Discontinuous Galerkin for transport.",
        "tutorial_steps": {
            "step-9": "Advection with streamline-diffusion (SUPG) stabilization on continuous FE_Q + multithreading (WorkStream) + adaptive refinement",
            "step-12": "DG for linear advection with MeshWorker framework",
            "step-30": "Anisotropic mesh refinement for DG advection",
        },
        "function_space": "FE_DGQ<dim>(p) — discontinuous Lagrange, degree 1-3",
        "solver": "SolverGMRES + PreconditionBlockJacobi (ILU per block)",
        "pitfalls": [
            (
                "[API] Sparsity pattern must include face-coupling: "
                "DoFTools::make_flux_sparsity_pattern(). Signal: "
                "using the regular make_sparsity_pattern() on a DG "
                "discretization gives a matrix with missing off-"
                "diagonal entries for face-coupling DOFs; in a Debug "
                "build assembly then aborts at the first face "
                "contribution with ExcInvalidIndex: `You are trying "
                "to access the matrix entry with index <i,j>, but "
                "this entry does not exist in the sparsity pattern "
                "of this matrix.` The check is an Assert, so a "
                "Release build does not stop: the out-of-pattern "
                "face entries are dropped (SparseMatrix::add) or "
                "added into the next stored entry after the missing "
                "column (AffineConstraints::distribute_local_to_global), "
                "and the matrix is silently wrong. (Audit 2026-06-02; "
                "corrected against deal.II 9.8.0-pre.)"
            ),
            (
                "[Numerical] Interior penalty parameter (alpha "
                "fed to MeshWorker / FEInterfaceValues face "
                "integrators) must be large enough for stability "
                "(scales with p^2). Signal: alpha too small -> "
                "coercivity loss and the computed L^2 norm from "
                "VectorTools::integrate_difference diverges with "
                "mesh refinement; alpha too large -> condition "
                "number > 1e14 and SolverGMRES stagnates. Rule: "
                "alpha = 4 * (p+1)^2 for SIPG with FE_DGQ<dim>(p). "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Face integrals require careful "
                "normal orientation handling via FEValues / "
                "FEInterfaceValues. Signal: FEFaceValues / "
                "FEInterfaceValues evaluated on the (-) side "
                "gives normal pointing FROM (-) TO (+); "
                "swapping +/- in the jump integral gives a "
                "SIGN ERROR — the assembled SparseMatrix is "
                "the TRANSPOSE of what was intended, and "
                "convergence rate degrades from O(h^(p+1)) to "
                "O(1) (no convergence, diagnosable via "
                "KellyErrorEstimator). (Audit 2026-06-02.)"
            ),
            (
                "[Performance] Streamline ordering of DOFs can "
                "help GMRES convergence. Signal: default DoF "
                "renumbering (Cuthill-McKee) gives GMRES "
                "convergence in ~50-100 iters for advection-"
                "dominated flow; switching to "
                "DoFRenumbering::downstream(b) cuts it to ~10-20 "
                "iters because the upwind sweep matches the "
                "matrix sparsity structure. (Audit 2026-06-02.)"
            ),
        ],
    },
    "wave_equation": {
        "description": "Wave equation (step-23/24/25). Time-dependent hyperbolic PDE.",
        "tutorial_steps": {
            "step-23": "Wave equation in bounded domain",
            "step-24": "Thermoacoustic tomography with absorbing BCs",
            "step-25": "Nonlinear wave (sine-Gordon soliton)",
            "step-48": "Parallel wave equation, matrix-free",
            "step-62": "Elastic wave propagation in phononic crystals",
        },
        "function_space": "FE_Q<dim>(1) — scalar Lagrange per time step",
        "solver": "SolverCG + PreconditionJacobi per time step (mass matrix is SPD)",
    },
    "nonlinear_elasticity": {
        "description": "Nonlinear solid mechanics (step-44). Neo-Hookean, three-field formulation.",
        "tutorial_steps": {"step-44": "Compressible Neo-Hookean with quasi-incompressible three-field formulation"},
        "function_space": "FESystem for displacement + pressure + dilatation (3-field)",
        "solver": "Newton iteration with direct solver",
        "pitfalls": [
            (
                "[Numerical] Three-field formulation needed for "
                "quasi-incompressible materials. Signal: single-"
                "field displacement formulation locks for nu > "
                "0.49 — incompressible Neo-Hookean block under "
                "uniaxial extension shows displacement ~500x too "
                "small. step-44 uses (u, p_tilde, J_tilde) three-"
                "field FESystem to recover the correct response. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Newton convergence requires good "
                "initial guess and small load steps. Signal: a "
                "single load step from zero to full deformation "
                "diverges within 2-3 Newton iterations (visible "
                "in SolverControl::log_history) for stretch "
                "ratios > 1.1; subdividing into 10-20 load "
                "increments with the previous AffineConstraints-"
                "constrained solution as initial guess brings "
                "each step inside Newton's convergence basin. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Performance] Automatic differentiation "
                "(step-71/72) avoids hand-coding Jacobians. "
                "Signal: hand-coded tangent for Mooney-Rivlin or "
                "Holzapfel-Gasser-Ogden has dozens of lines of "
                "tensor index arithmetic — easy to miss a term "
                "and get linear-rate Newton convergence; "
                "Differentiation::AD::ResidualLinearization "
                "(Sacado backend) generates the exact tangent "
                "and restores quadratic convergence. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "compressible_euler": {
        "description": "Compressible Euler equations (step-33/67/69). Hyperbolic conservation laws.",
        "tutorial_steps": {
            "step-33": "Compressible Euler, basic conservation law framework",
            "step-67": "High-order DG + explicit time stepping + matrix-free (fastest)",
            "step-69": "Euler with first-order viscous stabilization",
            "step-76": "Cell-centric matrix-free with MPI-3.0 shared memory",
        },
        "function_space": "FE_DGQ<dim>(2-5) — high-order DG",
        "solver": "Explicit Runge-Kutta (no linear solve needed, matrix-free)",
        "pitfalls": [
            (
                "[Numerical] MUST use DG elements — continuous "
                "(CG / FE_Q) elements are unstable for Euler. "
                "Signal: a Sod shock-tube benchmark with FE_Q "
                "develops uncontrolled oscillations that propagate "
                "across the domain within a few time steps; the "
                "same setup with FE_DGQ produces sharp shocks. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Numerical flux choice: Lax-Friedrichs "
                "(simple), HLLC (better shock resolution). Signal: "
                "Lax-Friedrichs over-smears contact discontinuities "
                "in a Sod problem by ~30% of the analytical jump; "
                "HLLC resolves them to <5% smearing. Use LF for "
                "robustness, HLLC for accuracy. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] CFL condition mandatory for explicit "
                "time stepping. Signal: dt > h / (|u| + c) gives "
                "NaN within ~10 steps because the explicit "
                "stencil cannot propagate information faster than "
                "one element per step. CFL safety factor ~0.3 is "
                "typical for SSP-RK3. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Shock capturing / slope limiting "
                "needed for discontinuous solutions. Signal: "
                "high-order DG without limiting produces "
                "Gibbs-style oscillations around shocks (over/"
                "undershoot 5-20% of the jump); applying a "
                "TVB-Minmod limiter or entropy-viscosity "
                "stabilisation eliminates them at the cost of "
                "local accuracy reduction near the shock. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "contact": {
        "description": "Contact / variational inequalities (step-41/42). Active set strategy.",
        "tutorial_steps": {
            "step-41": "Obstacle problem (variational inequality)",
            "step-42": "3D elasto-plastic contact with isotropic hardening (parallel)",
        },
        "solver": "Projected CG with AMG preconditioner + active set iteration",
        "pitfalls": [
            (
                "[Numerical] Active set changes require iterating "
                "between constraint detection (AffineConstraints / "
                "PETScWrappers::MPI::Vector test) and SolverGMRES "
                "solve. Signal: a single-shot SolverCG / SolverGMRES "
                "call where the active set is predicted from the "
                "initial guess gives the wrong contact zone for "
                "typical Hertz benchmarks (~30-50% wrong contact "
                "radius); the outer loop should iterate until two "
                "consecutive active sets are identical, usually "
                "3-10 iterations. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Penalty parameter: too small = "
                "constraint violation, too large = ill-"
                "conditioning. Signal: too low -> max penetration "
                "> 5% element edge; too high -> SolverGMRES "
                "stagnates with condition number > 1e14. Rule of "
                "thumb: penalty = 1e3 * E / h for typical Hertz "
                "contact with elastic Young modulus E. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] Use AffineConstraints to enforce contact "
                "constraints. Signal: hand-modifying the system "
                "matrix (zeroing rows + setting diag = 1) for "
                "constrained nodes is brittle and breaks "
                "parallel assembly. AffineConstraints<double> "
                "with constraints.add_line + add_entries handles "
                "the matrix modifications consistently across "
                "MPI ranks. (Audit 2026-06-02.)"
            ),
        ],
    },
    "grid_generator_catalog": {
        "description": "Complete catalog of deal.II GridGenerator functions for mesh creation.",
        "generators": {
            "hyper_cube": {"geometry": "Unit cube [0,1]^dim", "dims": "1D,2D,3D", "boundary_ids": "All = 0 (colorize=true for distinct)"},
            "hyper_rectangle": {"geometry": "Axis-aligned box [p1,p2]", "dims": "1D,2D,3D", "boundary_ids": "2D/3D: all = 0 unless colorize=true (then x-=0,x+=1,y-=2,y+=3,z-=4,z+=5); 1D: x-=0,x+=1 always"},
            "subdivided_hyper_rectangle": {"geometry": "Box with per-axis subdivision control", "dims": "1D,2D,3D"},
            "hyper_ball": {"geometry": "Circular disk / ball", "dims": "2D,3D", "notes": "SphericalManifold attached"},
            "hyper_shell": {"geometry": "Annulus / spherical shell", "dims": "2D,3D", "notes": "Inner + outer radius"},
            "hyper_L": {"geometry": "L-shaped domain (3D: Fichera corner)", "dims": "2D,3D", "notes": "Classic corner singularity benchmark"},
            "plate_with_a_hole": {"geometry": "Rectangle with cylindrical hole", "dims": "2D", "notes": "Stress concentration factor"},
            "channel_with_cylinder": {"geometry": "Flow channel with obstacle", "dims": "2D,3D", "notes": "DFG benchmark (Schäfer-Turek)"},
            "cylinder": {"geometry": "Cylinder along x (circular cross-section in 3D; in 2D the rectangle [-half_length,half_length]x[-radius,radius])", "dims": "2D,3D"},
            "cylinder_shell": {"geometry": "Hollow cylinder (pipe wall)", "dims": "3D"},
            "truncated_cone": {"geometry": "Cone frustum", "dims": "3D"},
            "cheese": {"geometry": "Rectangle with square holes", "dims": "2D,3D"},
            "hyper_cross": {"geometry": "Cross/plus shape", "dims": "2D,3D"},
            "pipe_junction": {"geometry": "Pipe bifurcation", "dims": "3D"},
            "Airfoil::create_triangulation": {"geometry": "NACA/Joukowski airfoil", "dims": "2D"},
            "extrude_triangulation": {"geometry": "Extrude 2D → 3D", "notes": "Layered 3D from 2D base"},
            "merge_triangulations": {"geometry": "Union of two meshes", "notes": "Combine separate grids"},
        },
    },
    "solver_catalog": {
        "description": "Complete deal.II solver and preconditioner catalog.",
        "solvers": {
            "SolverCG": "Conjugate Gradient — SPD systems (Poisson, elasticity, heat)",
            "SolverGMRES": "Restarted GMRES — non-symmetric (advection, NS)",
            "SolverFGMRES": "Flexible GMRES — variable preconditioner per iteration",
            "SolverBicgstab": "BiCGStab — non-symmetric alternative",
            "SolverMinRes": "MinRes — symmetric indefinite (Stokes, saddle-point)",
            "SparseDirectUMFPACK": "Direct — small/medium, complex-valued, debugging",
        },
        "preconditioners": {
            "PreconditionIdentity": "None — debugging only",
            "PreconditionJacobi": "Diagonal scaling — DG mass matrices",
            "PreconditionSSOR": "Symmetric SOR — CG-compatible, general purpose",
            "PreconditionChebyshev": "Polynomial — matrix-free multigrid smoothers (step-37)",
            "SparseMIC": "Incomplete Cholesky — SPD systems",
            "SparseILU": "Incomplete LU — general non-symmetric",
            "TrilinosWrappers::PreconditionAMG": "Algebraic multigrid (ML/MueLu) — large elliptic/elasticity",
        },
        "by_physics": {
            "poisson": "CG + SSOR (small) or CG + AMG (large) or CG + Chebyshev+GMG (fastest)",
            "elasticity": "CG + AMG (provide rigid body modes for near-nullspace)",
            "heat_transient": "CG + SSOR per time step",
            "stokes": "GMRES/MinRes + block preconditioner (AMG for velocity, mass-matrix for Schur)",
            "navier_stokes": "GMRES + block precon, Newton outer loop",
            "advection_dg": "GMRES + ILU or block-Jacobi",
            "euler_dg": "Explicit RK (no linear solve) — matrix-free",
            "wave": "CG + Jacobi per time step (mass matrix is SPD)",
        },
    },
    "element_catalog": {
        "description": "Complete deal.II finite element catalog.",
        "elements": {
            "FE_Q(p)": {"type": "Lagrange Qp", "continuity": "C0", "use": "Poisson, heat, elasticity — standard choice"},
            "FE_DGQ(p)": {"type": "DG Lagrange", "continuity": "Discontinuous", "use": "Advection, Euler, transport"},
            "FESystem(FE_Q(p), dim)": {"type": "Vector Lagrange", "continuity": "C0", "use": "Elasticity, displacement"},
            "FE_RaviartThomas(p)": {"type": "H(div) conforming", "continuity": "Normal continuous", "use": "Darcy flow, mixed Poisson"},
            "FE_Nedelec(p)": {"type": "H(curl) conforming", "continuity": "Tangential continuous", "use": "Maxwell, electromagnetics"},
            "FE_SimplexP(p)": {"type": "Simplex Lagrange", "continuity": "C0", "use": "Triangle/tet meshes"},
            "FE_BernardiRaugel": {"type": "Enriched velocity", "continuity": "C0", "use": "Low-order inf-sup stable Stokes"},
            "FE_Bernstein(p)": {"type": "Bernstein polynomials", "continuity": "C0", "use": "Positivity-preserving"},
        },
    },
    "tutorial_catalog": {
        "description": "Complete deal.II tutorial step catalog — maps step numbers to physics types and key features.",
        "step-1": "Grid generation and output",
        "step-2": "DOF setup and sparsity patterns",
        "step-3": "Poisson equation (basic)",
        "step-4": "Non-constant coefficients (dim-independent)",
        "step-5": "Variable coefficients, reading a grid from disk (GridIn), successively (globally) refined grids",
        "step-6": "Higher order elements + automatic adaptivity",
        "step-7": "Helmholtz + Neumann BCs + convergence tables",
        "step-8": "Elasticity (vector FE, FESystem)",
        "step-9": "Advection with streamline-diffusion (SUPG) stabilization on continuous FE_Q + multithreading (WorkStream) + adaptive refinement",
        "step-12": "DG advection (MeshWorker framework)",
        "step-15": "Minimal surface (nonlinear, Newton)",
        "step-16": "Geometric multigrid for Laplace",
        "step-17": "Parallel elasticity with PETSc",
        "step-18": "Quasi-static large-deformation elasticity",
        "step-20": "Mixed Darcy flow (Raviart-Thomas)",
        "step-22": "Stokes flow (Schur complement preconditioning)",
        "step-23": "Wave equation (time-dependent hyperbolic)",
        "step-26": "Heat equation (transient, adaptive mesh in time)",
        "step-27": "hp-FEM (combined h- and p-refinement)",
        "step-29": "Complex Helmholtz / scattering",
        "step-31": "Boussinesq convection (2D)",
        "step-33": "Compressible Euler equations",
        "step-35": "Navier-Stokes (projection method)",
        "step-36": "Eigenvalue problems (SLEPc)",
        "step-37": "Matrix-free methods (Laplace, fastest pattern)",
        "step-40": "Parallel with Trilinos (distributed)",
        "step-41": "Obstacle / contact problem",
        "step-42": "3D elasto-plastic contact",
        "step-44": "Nonlinear solid mechanics (Neo-Hookean, 3-field)",
        "step-45": "Parallel Stokes with periodic BCs",
        "step-47": "Biharmonic / Kirchhoff plate (C0 interior penalty)",
        "step-49": "Complex mesh generation + external mesh import",
        "step-51": "HDG (hybridizable DG) for convection-diffusion",
        "step-55": "Parallel Stokes + AMG",
        "step-56": "Stokes with geometric multigrid",
        "step-57": "Navier-Stokes (stationary, Newton + continuation)",
        "step-59": "DG + matrix-free (interior penalty)",
        "step-62": "Elastic wave propagation (phononic crystals)",
        "step-67": "Compressible Euler (high-order DG, matrix-free, explicit RK)",
        "step-70": "Particle FSI (immersed boundary method)",
        "step-71": "Automatic differentiation (magneto-mechanical coupling)",
        "step-72": "AD for Jacobians (nonlinear PDEs)",
        "step-74": "SIPG DG for Poisson",
        "step-77": "SUNDIALS KINSOL nonlinear solver",
        "step-79": "Topology optimization (SIMP)",
        "step-81": "Time-harmonic Maxwell equations",
        "step-85": "CutFEM for Poisson on circular domain",
        "step-87": "Remote point evaluation on distributed meshes",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# FEBIO — REMOVED 2026-06-02
# The previous `_FEBIO_KNOWLEDGE` covered only 4 of febio's 16 live
# supported physics, while the febio backend's own generators carry
# matching `description` fields for all 16. workflows.py's merge
# (`dict(deep_knowledge); ... if k not in knowledge`) allowed the
# 4-key deep_knowledge entry to SHADOW the backend's larger catalog.
# The febio backend is now the single source of truth for febio
# knowledge — extend src/backends/febio/generators/ to grow it.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-SOLVER VALIDATION KNOWLEDGE
# Verified results from 10 benchmarks across FEniCS, deal.II, and 4C.
# This knowledge helps fresh agents set up correct simulations and verify results.
# ═══════════════════════════════════════════════════════════════════════════════

_CROSS_SOLVER_KNOWLEDGE = {
    "cross_validation_principles": {
        "description": (
            "Cross-solver validation means running the same problem on multiple independent "
            "solvers and checking that they produce consistent results. This is a powerful "
            "verification technique — if two solvers agree, it's strong evidence both are correct."
        ),
        "methodology": [
            "Define the problem precisely (domain, BCs, material, source term)",
            "Run on 2+ solvers with comparable discretizations",
            "Compare key output quantities (max field value, tip displacement, etc.)",
            "Expect small differences (1-3%) from different element types — this is normal",
            "Large differences (>5%) indicate a setup error in one of the solvers",
        ],
    },
    "element_type_effects": {
        "description": (
            "Different solvers use different default element types. P1 triangles and Q1 "
            "quadrilaterals give slightly different results on the same mesh density. Both "
            "converge to the same solution under refinement. Differences of 1-3% between "
            "tri and quad elements are expected and normal — not a sign of error."
        ),
    },
    "4c_inline_mesh_notes": {
        "description": (
            "4C inline mesh (NODE COORDS + ELEMENTS) creates self-contained input files "
            "without external Exodus mesh dependencies."
        ),
        "key_pitfalls": [
            "Elasticity in 2D has 2 DOFs per node and no z-dof (WALL before 4C 2026.2.0, 2-D SOLID from 4C 2026.2.0 on): "
            "DIRICH needs NUMDOF >= 2 (a third entry is ignored; NUMDOF 1 aborts with "
            "'1 DOFs given but 2 expected in Point Dirichlet boundary condition'); "
            "structural NEUMANN needs NUMDOF >= 2 with the entries beyond 2 left off",
            "Element ordering: node IDs counter-clockwise for QUAD4",
            "Structure .vtu for ParaView needs BOTH IO/RUNTIME VTK OUTPUT with INTERVAL_STEPS (default -1 = off) "
            "AND IO/RUNTIME VTK OUTPUT/STRUCTURE with OUTPUT_STRUCTURE: true plus at least one field such as "
            "DISPLACEMENT: true (OUTPUT_STRUCTURE alone aborts with 'No data was written or writer was already "
            "in final phase.'); either section alone writes no .vtu. Scalar transport writes .vtu with no IO "
            "section at all",
        ],
    },
}
