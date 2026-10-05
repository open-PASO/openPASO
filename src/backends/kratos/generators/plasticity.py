"""Kratos plasticity generators and knowledge.

Covers the ConstitutiveLawsApplication plasticity framework:
- Yield surfaces: VonMises, DruckerPrager, MohrCoulomb, ModifiedMohrCoulomb, Tresca (Rankine and SimoJu are damage-only)
- Plastic potentials: same set (non-associative flow supported)
- Hardening: perfect, linear, exponential, curve-fitting
- Strain formulations: small strain, finite strain
"""


def _plasticity_3d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE — uniaxial compression with Mohr-Coulomb plasticity.

    Single 3D hex element, displacement-controlled. Demonstrates the
    ConstitutiveLawsApplication plasticity framework."""
    E = params.get("E", 50e6)
    nu = params.get("nu", 0.3)
    c = params.get("c", 50e3)
    phi = params.get("phi", 30.0)
    psi = params.get("psi", 0.1)  # >0 to avoid singular denominator
    max_strain = params.get("max_strain", 0.005)
    n_steps = params.get("n_steps", 100)

    import math
    sin_phi = math.sin(math.radians(phi))
    cos_phi = math.cos(math.radians(phi))
    sigma_y = 2 * c * cos_phi / (1 - sin_phi)

    return f'''\
"""Mohr-Coulomb plasticity: uniaxial compression — Kratos ConstitutiveLawsApplication"""
import sys, json
import numpy as np

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.ConstitutiveLawsApplication as CLA

# Parameters
E = {E}
nu = {nu}
c = {c}
phi_deg = {phi}
psi_deg = {psi}
max_strain = {max_strain}
n_steps = {n_steps}

sin_phi = np.sin(np.radians(phi_deg))
cos_phi = np.cos(np.radians(phi_deg))
sigma_y = 2 * c * cos_phi / (1 - sin_phi)
print(f"Analytical yield stress: {{sigma_y/1e3:.1f}} kPa")

# Create model
model = KM.Model()
mp = model.CreateModelPart("Structure")
mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
mp.AddNodalSolutionStepVariable(KM.REACTION)
mp.AddNodalSolutionStepVariable(KM.VOLUME_ACCELERATION)
mp.SetBufferSize(2)

L = 1.0
mp.CreateNewNode(1, 0.0, 0.0, 0.0)
mp.CreateNewNode(2, L,   0.0, 0.0)
mp.CreateNewNode(3, L,   L,   0.0)
mp.CreateNewNode(4, 0.0, L,   0.0)
mp.CreateNewNode(5, 0.0, 0.0, L)
mp.CreateNewNode(6, L,   0.0, L)
mp.CreateNewNode(7, L,   L,   L)
mp.CreateNewNode(8, 0.0, L,   L)

# Material — use CLA variables, NOT KM
prop = mp.CreateNewProperties(1)
prop.SetValue(KM.YOUNG_MODULUS, E)
prop.SetValue(KM.POISSON_RATIO, nu)
prop.SetValue(KM.DENSITY, 0.0)
prop.SetValue(CLA.YIELD_STRESS_COMPRESSION, sigma_y)
prop.SetValue(CLA.YIELD_STRESS_TENSION, sigma_y)
prop.SetValue(CLA.FRICTION_ANGLE, phi_deg)
prop.SetValue(CLA.DILATANCY_ANGLE, psi_deg)
prop.SetValue(KM.FRACTURE_ENERGY, 1e10)
prop.SetValue(CLA.HARDENING_CURVE, 3)  # 3 = perfect plasticity

# Constitutive law — use specific class, NOT the factory with parameters
cl = CLA.SmallStrainIsotropicPlasticity3DModifiedMohrCoulombModifiedMohrCoulomb()
prop.SetValue(KM.CONSTITUTIVE_LAW, cl)

mp.CreateNewElement("SmallDisplacementElement3D8N", 1, [1,2,3,4,5,6,7,8], prop)

dt = 1.0 / n_steps
mp.ProcessInfo[KM.DELTA_TIME] = dt
mp.ProcessInfo[KM.DOMAIN_SIZE] = 3

for node in mp.Nodes:
    node.AddDof(KM.DISPLACEMENT_X, KM.REACTION_X)
    node.AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y)
    node.AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z)

# BCs: bottom fixed in z, symmetry planes, top prescribed
for nid in [1,2,3,4]:
    mp.Nodes[nid].Fix(KM.DISPLACEMENT_Z)
    mp.Nodes[nid].SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)
for nid in [1,4,5,8]:
    mp.Nodes[nid].Fix(KM.DISPLACEMENT_X)
    mp.Nodes[nid].SetSolutionStepValue(KM.DISPLACEMENT_X, 0.0)
for nid in [1,2,5,6]:
    mp.Nodes[nid].Fix(KM.DISPLACEMENT_Y)
    mp.Nodes[nid].SetSolutionStepValue(KM.DISPLACEMENT_Y, 0.0)
for nid in [5,6,7,8]:
    mp.Nodes[nid].Fix(KM.DISPLACEMENT_Z)

linear_solver = KM.SkylineLUFactorizationSolver()
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
convergence_criterion = KM.ResidualCriteria(1e-6, 1e-9)
builder_and_solver = KM.ResidualBasedBlockBuilderAndSolver(linear_solver)
strategy = KM.ResidualBasedNewtonRaphsonStrategy(
    mp, scheme, convergence_criterion, builder_and_solver, 30, True, False, True)
strategy.SetEchoLevel(0)
strategy.Check()

results = {{"steps": [], "strain_zz": [], "stress_zz_kPa": []}}
for step in range(1, n_steps + 1):
    mp.CloneTimeStep(step * dt)
    mp.ProcessInfo[KM.STEP] = step
    u_z = -max_strain * L * step / n_steps
    for nid in [5,6,7,8]:
        mp.Nodes[nid].SetSolutionStepValue(KM.DISPLACEMENT_Z, u_z)
    strategy.Solve()
    stress_vec = mp.Elements[1].CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, mp.ProcessInfo)
    s_avg = np.mean([np.array([s[i] for i in range(6)]) for s in stress_vec], axis=0)
    results["steps"].append(step)
    results["strain_zz"].append(u_z / L)
    results["stress_zz_kPa"].append(s_avg[2] / 1e3)

with open("results_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Peak stress: {{max(abs(s) for s in results['stress_zz_kPa']):.1f}} kPa (analytical: {{sigma_y/1e3:.1f}} kPa)")
'''


GENERATORS = {
    "plasticity_3d": _plasticity_3d_kratos,
}

KNOWLEDGE = {
    "plasticity": {
        "description": (
            "Elasto-plasticity via ConstitutiveLawsApplication: "
            "5 yield surfaces (plasticity), 5 plastic potentials, "
            "6 hardening curves. Rankine + SimoJu yield surfaces "
            "exist only in the *damage* family of constitutive "
            "laws, NOT in plasticity — see pitfall #0."
        ),
        "application": "ConstitutiveLawsApplication (pip install KratosConstitutiveLawsApplication)",
        "requires": "StructuralMechanicsApplication (for elements)",
        "yield_surfaces": [
            # PLASTICITY-COMPATIBLE (combine with
            # SmallStrainIsotropicPlasticity3D / Kinematic etc.):
            "VonMises — J2 metal plasticity",
            "Tresca — maximum shear stress",
            "DruckerPrager — smooth cone (soil/concrete)",
            "MohrCoulomb — hexagonal pyramid (classical soil plasticity)",
            "ModifiedMohrCoulomb — regularized MC with tension/compression asymmetry",
            # DAMAGE-ONLY (combine with SmallStrainDplus
            # DminusDamage* or SmallStrainIsotropicDamage*,
            # NOT SmallStrainIsotropicPlasticity*):
            "Rankine — DAMAGE only (maximum tensile stress, brittle)",
            "SimoJu — DAMAGE only (damage-type yield for quasi-brittle materials)",
        ],
        "plastic_potentials": [
            "VonMises, Tresca, DruckerPrager, MohrCoulomb, ModifiedMohrCoulomb",
            "Non-associative flow: yield surface and plastic potential can differ",
            "Example: MohrCoulomb yield + DruckerPrager potential",
        ],
        "hardening_curves": {
            "0": "LinearSoftening",
            "1": "ExponentialSoftening",
            "2": "InitialHardeningExponentialSoftening",
            "3": "PerfectPlasticity (constant threshold)",
            "4": "CurveFittingHardening (polynomial + exponential)",
            "5": "LinearExponentialSoftening",
            "6": "CurveDefinedByPoints (piecewise-linear)",
        },
        "constitutive_law_naming": {
            "pattern": "<StrainSize><HardeningType><Dimension><YieldSurface><PlasticPotential>",
            "example": "SmallStrainIsotropicPlasticity3DModifiedMohrCoulombModifiedMohrCoulomb",
            "strain_sizes": ["SmallStrain", "FiniteStrain"],
            "hardening_types": ["Isotropic", "Kinematic"],
            "note": "Use the specific class directly in Python API, NOT the factory with Parameters",
        },
        "python_api": {
            "variable_locations": {
                "CLA (ConstitutiveLawsApplication)": [
                    "YIELD_STRESS_COMPRESSION", "YIELD_STRESS_TENSION",
                    "FRICTION_ANGLE", "DILATANCY_ANGLE",
                    "HARDENING_CURVE", "COHESION",
                ],
                "KM (KratosMultiphysics)": [
                    "YOUNG_MODULUS", "POISSON_RATIO", "DENSITY",
                    "FRACTURE_ENERGY", "YIELD_STRESS",
                ],
            },
            "instantiation": (
                "cl = CLA.SmallStrainIsotropicPlasticity3DModifiedMohrCoulombModifiedMohrCoulomb(); "
                "prop.SetValue(KM.CONSTITUTIVE_LAW, cl)"
            ),
            "factory_warning": (
                "SmallStrainIsotropicPlasticityFactory() takes NO constructor arguments. "
                "It reads yield_surface/plastic_potential from Properties in JSON workflow only. "
                "For Python API, use the specific pre-combined class directly."
            ),
        },
        "parameter_translation": {
            "classical_MC_to_modified_MC": {
                "description": "Modified MC uses YIELD_STRESS_COMPRESSION/TENSION instead of cohesion+friction angle",
                "uniaxial_sigma3_eq_0": "YIELD_STRESS_COMPRESSION = 2*c*cos(phi)/(1-sin(phi))",
                "triaxial_with_confining": "YIELD_STRESS_COMPRESSION = 2*c*cos(phi)/(1-sin(phi)) + sigma_3*(1+sin(phi))/(1-sin(phi))",
                "note": "The threshold depends on the stress state — set it for the dominant loading condition",
            },
        },
        "benchmarks": {
            "uniaxial_compression": {
                "description": "Single element uniaxial compression — simplest MC test",
                "analytical_yield": "sigma_y = 2*c*cos(phi)/(1-sin(phi))",
                "example_params": "E=50 MPa, nu=0.3, c=50 kPa, phi=30 deg → sigma_y = 173.2 kPa",
                "reference": "de Souza Neto, Peric, Owen: Computational Methods of Plasticity",
            },
            "triaxial_compression": {
                "description": "Triaxial with confining pressure sigma_3",
                "analytical_yield": "sigma_1 = sigma_3*(1+sin(phi))/(1-sin(phi)) + 2*c*cos(phi)/(1-sin(phi))",
                "example_params": "sigma_3=100 kPa, c=50 kPa, phi=30 deg → sigma_1 = 473.2 kPa, q = 373.2 kPa",
                "elastic_offset": (
                    "With confining pressure, the deviatoric stress at zero axial strain is NOT zero. "
                    "q = E*eps_a - sigma_3*(1-2*nu). For sigma_3=100 kPa, nu=0.3: q_offset = -40 kPa. "
                    "The elastic line in a q-eps plot starts at -40 kPa, not at the origin."
                ),
                "boundary_conditions": (
                    "Use Neumann (pressure) BCs for confining on lateral faces + Dirichlet (displacement) "
                    "for axial compression on the top face. Do NOT use fully displacement-controlled BCs "
                    "for all faces — this bypasses global equilibrium iteration and the material tangent "
                    "is never tested, hiding convergence issues in the constitutive law."
                ),
                "reference": "DIANA FEA Mohr-Coulomb Model Verification; validated against PLAXIS",
            },
        },
        "return_mapping": {
            "description": (
                "MC return mapping formulas for solver developers. "
                "MC with Lode-angle smoothing admits a single-step closed-form cone return, "
                "analogous to Drucker-Prager (de Souza Neto Ch. 8-9)."
            ),
            "lode_angle_factor": (
                "K(theta, angle) = cos(theta) - sin(theta)*sin(angle)/sqrt(3), "
                "where theta is the Lode angle (clamped to +/-29 deg for smoothing). "
                "K_phi uses the friction angle, K_psi uses the dilatancy angle."
            ),
            "yield_function": "F = K_phi * sqrt(J2) + sin(phi) * p - cos(phi) * c",
            "cone_return": (
                "Dgamma = F_trial / (K_phi * K_psi * G + kappa * sin(phi) * sin(psi) + cos(phi)^2 * H). "
                "Stress update: devstress *= (1 - K_psi*G*Dgamma/sqrt(J2_trial)), "
                "p = p_trial - kappa*sin(psi)*Dgamma, "
                "strainbar_p += cos(phi)*Dgamma."
            ),
            "apex_return": (
                "When sqrt(J2) - K_psi*G*Dgamma < 0, return to apex: devstress = 0, "
                "dstrainv = (sin(phi)*p_trial - cos(phi)*c) / (kappa*sin(phi) + cos^2(phi)/sin(phi)*H)."
            ),
            "lode_smoothing_error": (
                "Clamping the Lode angle at +/-29 deg introduces ~3% error at the compression/extension "
                "meridians (exact theta = +/-30 deg). This is inherent to the smoothing approach and "
                "acceptable for engineering use. For exact MC, use a principal-stress-space return mapping "
                "(Sloan et al., IJNME 2001)."
            ),
        },
        "pitfalls": [
                        '[API] Rankine and SimoJu yield surfaces '
                        'are DAMAGE-only — they do NOT combine into '
                        'the SmallStrainIsotropicPlasticity factory. '
                        'The catalog yield_surfaces list previously '
                        'implied SmallStrainIsotropicPlasticity3D'
                        'RankineRankine / ...SimoJuSimoJu exist via '
                        'the documented <StrainSize><HardeningType>'
                        '<Dimension><YieldSurface><PlasticPotential>'
                        ' pattern — they do not. The only Rankine / '
                        'SimoJu instantiations registered in '
                        'KratosConstitutiveLawsApplication 10.4.2 '
                        'are damage models: '
                        'SmallStrainDplusDminusDamageRankineRankine3D, '
                        'SmallStrainDplusDminusDamageSimoJuSimoJu3D, '
                        'SmallStrainIsotropicDamage3DRankine, '
                        'SmallStrainIsotropicDamage3DSimoJu. The 5 '
                        'plasticity-compatible yield surfaces are '
                        'VonMises, Tresca, DruckerPrager, MohrCoulomb, '
                        'ModifiedMohrCoulomb. '
                        "Signal: hasattr(CLA, 'SmallStrainIsotropic"
                        "Plasticity3DRankineRankine') is False; "
                        "trying to construct it raises "
                        "AttributeError on the CLA module. "
                        "(Verified empirically 2026-06-01 — "
                        "Tier-2 fixture plasticity_rankine_simoju_"
                        "are_damage_only in scripts/tier2_fixtures/"
                        "kratos/.)",
                        '[Numerical] Mohr-Coulomb dilatancy angle psi=0 causes a singular plastic denominator (dF:C:dG = 0) at the MC compression meridian (Lode = -30 deg). Use psi >= 0.1 deg as workaround, or principal-stress-space return mapping (Sloan et al., 2001). '
                        "Signal: ResidualBasedNewtonRaphsonStrategy reports 'Convergence is not achieved' with the global residual stuck; local return-mapping in the ConstitutiveLaw emits 'division by zero' / NaN in PK2_STRESS_VECTOR returned by CalculateOnIntegrationPoints.",
                        '[Numerical] MC yield surface has 6 corners in the deviatoric plane; backward Euler needs Lode-angle smoothing (Drucker-Prager at |theta| >= 29 deg) or explicit corner return mapping. '
                        "Signal: ResidualCriteria reports the global residual ratio not decreasing across ResidualBasedNewtonRaphsonStrategy iterations; CalculateOnIntegrationPoints returns NaN entries in PK2_STRESS_VECTOR for stress states whose Lode angle is close to |30 deg|.",
                        '[Integration] Modified MC (Kratos) and Classical MC (textbook) use DIFFERENT parameterisations. Modified MC uses YIELD_STRESS_COMPRESSION/TENSION; Classical MC uses cohesion + friction angle. '
                        'Signal: yield surface intersects axes at wrong values — sigma_y in uniaxial compression disagrees with hand-calc by tan(phi)-related factor.',
                        '[Syntax] For perfect plasticity use HARDENING_CURVE=3 with large FRACTURE_ENERGY (e.g., 1e10). HARDENING_CURVE=0 (linear) still softens unless FRACTURE_ENERGY is very large. '
                        'Signal: stress-strain past yield droops with negative slope despite HARDENING_MODULUS=0; integrated fracture energy < 0.5 of analytical perfect-plastic value.',
                        '[API] Python API: constitutive-law variables are split across modules. FRICTION_ANGLE, DILATANCY_ANGLE, YIELD_STRESS_COMPRESSION live in ConstitutiveLawsApplication (CLA); FRACTURE_ENERGY, YOUNG_MODULUS in KratosMultiphysics (KM). '
                        "Signal: Attribute lookup raises AttributeError 'has no attribute' with the module and the name interpolated around it — the line reads Module KratosMultiphysics has no attribute FRICTION_ANGLE. — at the moment the wrong module is dotted into (e.g. KM.FRICTION_ANGLE), BEFORE properties.SetValue is even reached. The correct path is ConstitutiveLawsApplication.FRICTION_ANGLE (returns a DoubleVariable). (Verified empirically 2026-06-01 — prior catalog claim said the error fires 'from properties.SetValue'; reality is the AttributeError fires at attribute access, never reaching SetValue.)",
                        '[API] SmallStrainIsotropicPlasticityFactory() takes NO constructor arguments. Passing KM.Parameters raises TypeError. Use the specific pre-combined class (e.g. SmallStrainIsotropicPlasticity3DVonMisesVonMises). '
                        "Signal: TypeError '__init__(): incompatible constructor arguments. The following argument types are supported: 1. KratosConstitutiveLawsApplication.SmallStrainIsotropicPlasticityFactory()' when the factory is called with KM.Parameters. (Verified empirically 2026-06-01 after KratosConstitutiveLawsApplication was installed; prior text said 'incompatible function arguments' / 'from SetValue binding' — the actual message says 'constructor arguments' and originates from the factory __init__ binding, not SetValue.)",
                        '[Numerical] SHEAR LOCKING: linear hex8 (3D8N) locks in bending-dominated plasticity. Uniform-stress benchmarks (uniaxial, triaxial) are fine; gradient-stress problems need quadratic elements (3D20N, 3D27N). '
                        'Signal: bending-plasticity tip rotation 20-40% smaller than analytic with hex8; switching to hex20 recovers it.',
                        '[Numerical] Fully displacement-controlled single-element tests can pass spuriously — Newton converges in 1 iteration without exercising the material tangent. Always include at least one Neumann-loaded face (e.g., confining pressure in triaxial). '
                        "Signal: ResidualBasedNewtonRaphsonStrategy reports 1 iteration per CloneTimeStep for every step; ResidualCriteria initial-residual ratio is below tolerance from iteration 0 — the algorithmic tangent (CalculateOnIntegrationPoints DEFORMATION_GRADIENT path) is never tested.",
                        '[Physics] Triaxial elastic offset: with confining sigma_3, deviatoric q at zero axial strain is q = -sigma_3*(1-2*nu), NOT zero. For sigma_3=100 kPa, nu=0.3: q_0 = -40 kPa. '
                        "Signal: CalculateOnIntegrationPoints PK2_STRESS_VECTOR[2] at zero axial DISPLACEMENT_Z is non-zero and matches -sigma_3*(1-2*nu); the SmallDisplacementElement stress-strain curve starts on a parallel line offset from origin (the solver is correct).",
                    ],
        "elements": [
            "SmallDisplacementElement3D8N (linear hex, small strain)",
            "SmallDisplacementElement3D4N (linear tet)",
            "SmallDisplacementElement2D3N/4N (2D plane strain/stress)",
            "TotalLagrangianElement3D8N (finite strain — use with FiniteStrain* laws)",
        ],
    },
}
