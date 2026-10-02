# `verify_pde_consistency`

**Checks that the computed field actually satisfies the equation it claims to solve.**

Group: Check the answer.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `solution_files` | string | yes |  |
| `source_term` | string | yes |  |
| `coefficient` | string | no | `'1.0'` |
| `domain` | string | no | `'[[0,1],[0,1]]'` |
| `equation` | string | no | `''` |
| `reaction` | string | no | `'0.0'` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Does your field actually satisfy the equation the task stated?
    
    The body is shared with couple(), which runs this same check on
    every converged level when it is handed the four arguments. Two
    wordings of the invitation to call this tool were measured and both
    failed -- 2 calls against 14 asks, then 0 against 7 -- so the check
    stopped being something to remember and became something couple()
    does with what the agent already gave it.
    ```
