# `verify_interface_flux`

**For two coupled solvers: checks that what leaves one side arrives at the other.**

Group: Check the answer.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `interface_files` | string | yes |  |
| `solution_files` | string | no | `''` |
| `interface_axis` | integer | no | `0` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    On a COUPLED task: is your interface flux the right sign, and do
    your two sides actually agree there? Runs on YOUR OWN files, with no
    reference solution.
    
    THIS EXISTED ONLY INSIDE AN INDEPENDENT CHECK UNTIL NOW, which is why
    it is here. The check below is the one that decided one coupled
    recorded run: a result set whose two codes both genuinely ran,
    whose coupling genuinely iterated over three mesh levels, and whose
    interface FIELD matched to 0.000e+00 across the seam, was still
    complete but unphysical — because one side reported its flux with the
    INWARD normal. The relative flux jump came out 8.139e-01, 9.066e-01,
    9.530e-01 over the three levels: not shrinking, and growing. The
    agent had no way to see that before handing in. Now it does.
    
    WHAT IT CHECKS, all of it key-free:
    
    1. SIGN AND SELF-CONSISTENCY, per side. For a flux you really computed
       from your own solution,
    
           q_n(x) / (-du/dn)(x)  ==  k   at every interface point,
    
       so the flux is a CONSTANT multiple of -du/dn whatever k is — and
       POSITIVE. A NEGATIVE multiple means your normal points the
       wrong way: the task defines q_n = -(K grad u) . n_out with n_out
       pointing OUT of the subdomain. A flux far from every constant
       multiple, level after level, does not follow from the field you
       delivered; one that comes closer at every level is discretisation.
    
       The trap this catches most often: on the NEUMANN side the flux you
       IMPORT and the flux you REPORT have OPPOSITE signs. Kratos's
       FACE_HEAT_FLUX is the INWARD normal flux (measured against a closed
       form: u came out +0.875 where the inward reading predicts +0.875),
       so the number you write into each per-level interface file is the
       NEGATIVE of the one you applied.
    
    2. THE TWO-SIDED JUMP, and its refinement trend. The field trace must
       be continuous and the two fluxes must cancel, because the normals
       are anti-parallel. A jump that stays O(1) as h halves means the
       iteration converged to a fixed point of the WRONG transmission
       condition — and your observed order cannot see that, because
       convergence to a wrong answer is still convergence.
    
    Args:
        interface_files: comma-separated per-level interface files, one
            per side, in the shape `x, y, u, qn`; each name must carry
            `level<k>_<side>`, which is how level and side are read. Give
            BOTH sides and all levels; the trend is the informative part.
        solution_files: comma-separated per-level field files, one per
            side, `x, y, u`, named the same way. Needed for check 1 —
            without them the sign cannot be tested, only the jump.
        interface_axis: a fallback only. The normal is read from the files:
            the coordinate a straight interface holds constant, and on a bent
            interface each leg's own.
    
    Returns: per-side sign verdicts (leg by leg on a bent interface), per-level
        jumps, the refinement trend of each, and NOT_ASSESSED wherever a check
        could not look at anything — a check that could not run never reports
        success.
    ```
