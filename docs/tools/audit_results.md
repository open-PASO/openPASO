# `audit_results`

**Reads the result files and names anything missing, inconsistent or suspicious.**

Group: Check the answer.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `work_dir` | string | yes |  |
| `claimed_order` | number | no | `0.0` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Check your OWN result files for the failures that most often sink a
    result set — BEFORE you hand it in. Uses only files you produced; no
    reference solution is involved, so a clean audit means self-consistent,
    not correct.
    
    What it catches, measured on 94 independently-checked correct
    result sets (no false alarm on any) and 102 complete-but-wrong ones
    (39 caught, about four in ten — more when the result set states its
    claimed convergence order, which the order check needs):
    
      * NEAR-ZERO FIELD - your finest solution peaks below 1e-8. On a
        driven problem check that the source/load reaches the solve (a
        defined function no condition references, a load curve never
        activated, boundary values all zero) before reading it as small.
      * FLOOR - successive refinement levels within 5% of each other:
        whatever limits your number, it is not the mesh. Usual cause is a
        solver tolerance (nonlinear/iterative defaults stop near 1e-6).
      * ORDER MISMATCH - your levels improve at a measurably lower rate
        than the order you are about to claim. Usual causes: element
        degree below what the task states, volumetric locking, a
        first-order integrator behind a spatial study.
      * NON-MONOTONE - a refinement made the answer worse.
    
    Call it on the directory holding your per-level outputs (it reads
    your summary file and your per-level field files), pass the
    convergence order you intend to claim, and treat any finding as a
    reason to look BEFORE handing in - each one names where to look.
    
    Args:
        work_dir: directory containing your results (searched recursively)
        claimed_order: the convergence order your result set will claim
            (0 = no order claim, order checks are skipped)
    ```
