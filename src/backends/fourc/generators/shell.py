"""Shell element generator for 4C.

Covers Kirchhoff-Love and Reissner-Mindlin shell elements.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class ShellGenerator(BaseGenerator):
    """Generator for shell structure problems in 4C."""

    module_key = "shell"
    display_name = "Shell Elements"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Thin and thick shell elements for plates, curved shells, "
                "and general 3D surface structures.  Includes Kirchhoff-Love "
                "(thin, no transverse shear) and Reissner-Mindlin (thick, "
                "transverse shear) formulations."
            ),
            "elements": {
                # Verified against `4C --parameters` and by execution.  There is
                # no "SHELL KIRCHHOFF"/"SHELL REISSNER"/"SOLIDSHELL" element
                # type: 4C registers exactly these.
                "Reissner-Mindlin (7-parameter)": [
                    "SHELL7P QUAD4 <n..> MAT m THICK t EAS e1 e2 e3 e4 e5 "
                    "SDC s [USE_ANS true]",
                    "SHELL7P QUAD8 / QUAD9 / TRI3 / TRI6",
                    "SHELL7PSCATRA (same, scalar-transport coupled)",
                ],
                "Kirchhoff-Love": [
                    "SHELL_KIRCHHOFF_LOVE_NURBS NURBS9 <n..> MAT m GP a b "
                    "(NURBS9 only; needs PROBLEM TYPE.SHAPEFCT: Nurbs)",
                ],
                "solid-shell": [
                    "SOLID HEX8 ... TECH: shell_ans | shell_eas | shell_eas_ans "
                    "(the continuum-shell route is a TECH flag on SOLID, not a "
                    "separate element type)",
                ],
            },
            "materials": [
                "MAT_Struct_StVenantKirchhoff / MAT_ElastHyper (SHELL7P)",
                "MAT_Kirchhoff_Love_shell with YOUNG_MODULUS, POISSON_RATIO, "
                "THICKNESS (SHELL_KIRCHHOFF_LOVE_NURBS only — note the "
                "thickness is a MATERIAL parameter here, not an element one)",
            ],
            "pitfalls": [
                (
                    "[API] Kirchhoff-Love needs C1 continuity, and in 4C the "
                    "ONLY element that provides it is "
                    "SHELL_KIRCHHOFF_LOVE_NURBS on the NURBS9 cell, with "
                    "MAT_Kirchhoff_Love_shell and PROBLEM TYPE.SHAPEFCT: "
                    "Nurbs. There is no DKT element and no 'SHELL_KL_NURBS' "
                    "short name. Signal: every wrong spelling fails at the "
                    "element factory with \"Unknown type '<what you wrote>' of "
                    "finite element\" from core/comm/src/"
                    "4C_comm_parobjectfactory.cpp — including 'DKT' and "
                    "'SHELL_KL_NURBS'. Writing SHELL KIRCHHOFF QUAD4 is read "
                    "as element type SHELL with cell type KIRCHHOFF and dies "
                    "on 'Unknown celltype KIRCHHOFF'. Worst of the set: "
                    "pairing the CORRECT type name with a non-NURBS9 cell "
                    "reports the TYPE as unknown and never echoes the cell "
                    "type, so the message points away from the actual "
                    "mistake. (Falsified and corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Numerical] SHELL7P does lock for thin shells, but the "
                    "cure is NOT reduced integration: the element has no "
                    "integration-rule key at all. The two knobs are EAS (a "
                    "five-slot vector, e.g. N_4 N_4 N_4 none none) and "
                    "USE_ANS: true, the assumed-natural-strain treatment of "
                    "transverse shear. Signal: with a real EAS vector but "
                    "USE_ANS left at its default of false, a thin plate comes "
                    "out orders of magnitude too stiff, and switching USE_ANS "
                    "on is what recovers it. A WALL-style 'GP 2 2' is rejected "
                    "with \"After parsing, the line still contains 'GP 2 2'.\" "
                    "followed by the element's full accepted key list. Third "
                    "trap, and the dangerous one: setting every EAS slot to "
                    "'none' makes reference BLAS abort with ' ** On entry to "
                    "DGEMM parameter number 10 had an illegal value', and "
                    "because XERBLA terminates through Fortran STOP the "
                    "process EXITS 0 having run no time step and no result "
                    "test — check for a 'Finalised step' line before trusting "
                    "a zero exit code. (Falsified and corrected by execution "
                    "2026-08-06.)"
                ),
                (
                    "[Input] THICK is the SHELL7P thickness and it is "
                    "REQUIRED — there is no default to fall back on, so this "
                    "cannot go wrong silently. Signal: omitting it aborts at "
                    "parse with \"Required value 'THICK' not found in input "
                    "line\" from core/io/src/4C_io_input_spec_builders.cpp and "
                    "runs no step. Do not reach for the Kirchhoff-Love "
                    "spelling: THICKNESS belongs to MAT_Kirchhoff_Love_shell "
                    "in the MATERIALS section, and writing it on a SHELL7P "
                    "element line lets the parser consume the THICK prefix and "
                    "then complain about the NEXT key instead "
                    "(\"Required value 'SDC' not found in input line\"), which "
                    "names a key that was never the problem. (Falsified and "
                    "corrected by execution 2026-08-06.)"
                ),
                (
                    "[Input] The SHELL7P director is computed by the element "
                    "and cannot be given in the input: there is no director "
                    "key. Do not spend time looking for one. Signal: any "
                    "attempt (e.g. 'DIR 0 0 1' on the element line) is "
                    "rejected with \"After parsing, the line still contains "
                    "'DIR 0 0 1'.\" from core/io/src/4C_io_input_spec.cpp, and "
                    "the same message prints the element's entire accepted key "
                    "set — AXI, CIR, EAS, FIBER1..3, MAT, RAD, SDC, THICK, "
                    "USE_ANS — which is the fastest way to see that no "
                    "director entry exists. The vector keys that DO exist are "
                    "material-orientation data, not directors: adding FIBER1 "
                    "to an isotropic shell is accepted and leaves the answer "
                    "bit-identical. If you need a smoother director field, "
                    "refine or re-mesh; it is not an input option. (Falsified "
                    "and corrected by execution 2026-08-06.)"
                ),
                (
                    "[Output] 4C writes NO shell stress resultants. There is "
                    "no N_xx / M_xx output and nothing to switch on that would "
                    "produce one — the through-thickness integration is the "
                    "post-processor's job. What you can ask for is IO."
                    "STRUCT_STRESS: Cauchy plus IO.STRUCT_STRAIN: GL together "
                    "with IO/RUNTIME VTK OUTPUT/STRUCTURE.STRESS_STRAIN: true "
                    "(a BOOL in that section — not the element key of the same "
                    "name, which belongs to MEMBRANE (and, before 4C 2026.2.0, "
                    "to WALL) and is rejected "
                    "on a SHELL7P line with \"After parsing, the line still "
                    "contains 'STRESS_STRAIN plane_stress'.\"). Signal: the "
                    ".vtu then carries element_cauchy_stresses_xyz, "
                    "nodal_cauchy_stresses_xyz, element_GL_strains_xyz and "
                    "nodal_GL_strains_xyz — global-frame tensors, no "
                    "resultants. The only shell-specific extra field is "
                    "OPTIONAL_QUANTITY: shell7pthickness | "
                    "shell7pthicknessdirector. (Falsified and corrected by "
                    "execution 2026-08-06.)"
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "shell_3d", "description": "Shell structure under loading"}]

    def get_template(self, variant: str = "shell_3d") -> str:
        if variant not in ("shell_3d", "default"):
            raise ValueError(f"Unknown variant {variant!r}")
        from ..inline_mesh import matched_shell_3d_input
        return matched_shell_3d_input()

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
