"""Mixture/multiscale material generator for 4C.

Covers fiber-reinforced composites, biological tissue mixtures.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class MixtureGenerator(BaseGenerator):
    """Generator for mixture/composite material problems in 4C."""

    module_key = "mixture"
    display_name = "Mixture/Composite Materials"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Mixture theory for fiber-reinforced composites and biological tissues.  "
                "Multiple material constituents with individual constitutive laws "
                "combined via mixture rules."
            ),
            "materials": {
                "MAT_Mixture": "General mixture material with multiple constituents",
                "constituents": [
                    "MAT_ElastHyper (isotropic ground substance)",
                    "MAT_Muscle_Weickenmeier (skeletal muscle)",
                    "MAT_Muscle_Giantesio (active muscle)",
                    "Fiber families with anisotropic response",
                ],
            },
            "applications": ["arterial wall mechanics", "tendon/ligament",
                             "muscle tissue", "fiber-reinforced polymers",
                             "growth and remodeling"],
            "pitfalls": [
                (
                    "[Numerical] The mixture rule weights "
                    "constituent stresses by MASS fractions "
                    "(the input key is MASSFRAC), not volume "
                    "fractions, and they must sum to 1.  You "
                    "do NOT need to check this in "
                    "pre-processing -- the rule checks itself. "
                    "Signal: 'Mass fractions at element <n> "
                    "sum to <value> instead of 1.0, which is "
                    "unphysical.' from src/mixture/src/"
                    "4C_mixture_rule_simple.cpp at "
                    "SimpleMixtureRule::setup(), exit 1, in "
                    "both the under- and over-sum direction. "
                    "(Verified by execution 2026-08-06; no "
                    "stress is ever evaluated, so the "
                    "predicted artefact of a total exceeding "
                    "the sum of the constituents cannot "
                    "appear.)"
                ),
                (
                    "[Input] Fibre direction is chosen by the "
                    "integer INIT on the anisotropic summand "
                    "(e.g. ELAST_CoupAnisoExpo): INIT 0 takes "
                    "the direction from the GAMMA angle, INIT "
                    "1 takes it PER ELEMENT from a FIBER1 "
                    "vector on the element line.  There is no "
                    "FIBER_VEC key in 4C. Signal: writing "
                    "FIBER_VEC is a parse abort, 'Could not "
                    "match this input' from global_data/"
                    "4C_global_data_read.cpp; INIT 1 with no "
                    "element data gives 'Could not find "
                    "element coordinate system or element "
                    "fibers!' from mat/"
                    "4C_mat_anisotropy_extension_default.cpp. "
                    "Supplying FIBER1 per element changes the "
                    "answer, which is the point on a curved "
                    "geometry. (Verified by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Numerical] Growth is a property of the "
                    "MIXTURE RULE, not a flag you add to a "
                    "steady one.  Use "
                    "MIX_GrowthRemodelMixtureRule with a "
                    "GROWTH_STRATEGY and a constituent that "
                    "carries the time scale (e.g. "
                    "MIX_Constituent_FullConstrainedMixtureFiber "
                    "with GROWTH_CONSTANT and DECAY_TIME); "
                    "GROWTH_CONSTANT is the driver. Signal: "
                    "there is no RHO_GROWTH key -- adding one "
                    "is a parse abort, 'Could not match this "
                    "input' from global_data/"
                    "4C_global_data_read.cpp -- and attaching "
                    "GROWTH_STRATEGY to the steady "
                    "MIX_Rule_Simple is rejected the same way, "
                    "so the rule that a steady solve cannot "
                    "grow is enforced by the input spec rather than "
                    "showing up as a quiet pure-elastic "
                    "answer. (Verified by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Numerical] A near-incompressible mixture "
                    "on plain displacement-based hexes LOCKS "
                    "volumetrically, and the fix in 4C is an "
                    "ELEMENT TECHNOLOGY on the element line -- "
                    "add `TECH fbar` -- not a mixed (u, p) or "
                    "augmented-Lagrangian reformulation of the "
                    "material (the volumetric penalty itself "
                    "lives in the elastic summand, e.g. "
                    "ELAST_VolSussmanBathe). Signal: push the "
                    "Poisson ratio towards 0.5 on a "
                    "displacement-constrained specimen and the "
                    "deflection collapses by an order of "
                    "magnitude or more against the same mesh "
                    "with TECH fbar; compare the two before "
                    "trusting any near-incompressible mixture "
                    "result. (Verified by execution "
                    "2026-08-06.)"
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "mixture_3d", "description": "Mixture material under loading"}]

    def get_template(self, variant: str = "mixture_3d") -> str:
        if variant not in ("mixture_3d", "default"):
            raise ValueError(f"Unknown variant {variant!r}")
        from ..backend import _find_fourc_binary
        from ..grammar_dialect import dialect_of
        from ..inline_mesh import matched_mixture_3d_input
        return matched_mixture_3d_input(dialect=dialect_of(_find_fourc_binary()))

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
