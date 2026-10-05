"""Kratos RANS turbulence modeling generators and knowledge.

Covers k-epsilon, k-omega SST, and other RANS turbulence models.
Application: RANSApplication.
"""


# NOTE (2026-06-26 honesty audit): the previous _rans_2d generator was an
# availability-probe stub (import-check + {"note": "not installed"}, no
# solver run). RANSApplication is NOT importable in the installed Kratos
# stack, so 'rans' has been removed from the generator registry and from
# KratosBackend.supported_physics(). KNOWLEDGE retained for reference.


KNOWLEDGE = {
    "rans": {
        "description": "RANS turbulence modeling for incompressible flow",
        "application": "RANSApplication (pip install KratosRANSApplication)",
        "models": {
            "k_epsilon": "Standard k-epsilon with wall functions",
            "k_omega": "Wilcox k-omega model",
            "k_omega_sst": "Menter SST model (recommended for general use)",
            "spalart_allmaras": "One-equation SA model",
        },
        "elements": [
            # Real RANSApplication names: drop the 'Element'
            # suffix and add an AFC / CWD / RFC stabilization tag
            # in the middle (verified empirically 2026-06-01 via
            # kratos_eletype_scanner.py). The catalog previously
            # listed RansKEpsilonKElement2D3N etc. — none
            # registered.
            "RansKEpsilonK{AFC,CWD,RFC}2D3N",
            "RansKEpsilonK{AFC,CWD,RFC}3D4N",
            "RansKEpsilonEpsilon{AFC,CWD,RFC}2D3N",
            "RansKEpsilonEpsilon{AFC,CWD,RFC}3D4N",
            "RansKOmegaK{AFC,CWD,RFC}{2D3N,3D4N}",
            "RansKOmegaOmega{AFC,CWD,RFC}{2D3N,3D4N}",
            "RansKOmegaSSTK{AFC,CWD,RFC}{2D3N,3D4N}",
            "RansKOmegaSSTOmega{AFC,CWD,RFC}{2D3N,3D4N}",
            # Example expansions actually verified:
            "RansKEpsilonKAFC2D3N",
            "RansKEpsilonEpsilonCWD2D3N",
            "RansKOmegaSSTKRFC2D3N",
        ],
        "wall_treatment": ["wall_functions (log law)", "low_Re (resolve boundary layer)"],
        "pitfalls": [
            "[API] RANSApplication element names DROP the \"Element\" suffix and ADD an AFC / CWD / RFC stabilization-scheme tag in the middle. Examples: RansKEpsilonKAFC2D3N, RansKEpsilonEpsilonCWD2D3N, RansKOmegaSSTKRFC2D3N. The catalog previously listed RansKEpsilonKElement2D3N / RansKOmegaSSTKElement2D3N \u2014 none of those names are registered. Signal: calling model_part.CreateNewElement with the unregistered name RansKEpsilonKElement2D3N raises 'is not registered' from kratos/python/add_model_part_to_python.cpp:159 on 10.3.0 (:154 in 10.4.0 to 10.4.3); dropping the Element suffix and inserting AFC / CWD / RFC lets the call succeed. (Verified empirically 2026-06-01 \u2014 Tier-2 fixture rans_shallowwater_element_naming in scripts/tier2_fixtures/kratos/.)",
            "[Integration] Requires FluidDynamicsApplication as dependency Signal: RANSApplication needs FluidDynamicsApplication: the application's Python loader imports its dependencies first, so a build missing one fails on that inner import line rather than on the user's own import statement.",
        ],
        "guidance": [
            "[Numerical] Wall distance computation needed for SST model",
            "[Physics] Inlet turbulence: specify k and epsilon/omega from turbulence intensity",
            "[Numerical] y+ must be appropriate for chosen wall treatment (30-300 for wall functions)",
        ]
    },
}

# Empty: RANSApplication not installable in this Kratos stack; the prior
# generator was a no-solve probe stub (removed).
GENERATORS = {}
