# `couple_levels`

**Does the same on several mesh sizes, so the accuracy of the coupled answer can be measured.**

Group: Two solvers on one problem.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `participants` | string or array | yes |  |
| `levels` | string or array | yes |  |
| `critic_approved` | boolean | no | `False` |
| `max_iter` | integer | no | `150` |
| `tol` | number | no | `1e-06` |
| `accelerator` | string | no | `'auto'` |
| `theta` | number | no | `0.5` |
| `probe` | boolean | no | `True` |
| `history_dir` | string | no | `''` |
| `history_pattern` | string | no | `'coupling_history_level{k}.csv'` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    EVERY PRESCRIBED MESH LEVEL IN ONE CALL -- the same partitioned coupling as
    `couple`, run once per level of a task's mesh sequence.
    
    Measured over three rounds of recorded runs: six couplings that were proven at
    level 1 (both codes ran, the iteration converged) never reached level 3,
    because every level cost the agent ten more tool calls -- edit both
    config.json files, call couple, save the history, write the deliverables --
    and the wall clock ran out. This call does the per-level plumbing itself:
    
      * `levels` is a JSON list, one entry per level, with each side's mesh
        keys under that side's NAME, e.g. (your task's own numbers go here)
            [{"level": 1, "A": {"nx": <nA>, "ny": <mA>},  "B": {"nx": <nB>, "ny": <mB>}},
             {"level": 2, "A": {"nx": <2*nA>, "ny": <2*mA>}, "B": {"nx": <2*nB>, "ny": <2*mB>}},
             {"level": 3, "A": {"nx": <4*nA>, "ny": <4*mA>}, "B": {"nx": <4*nB>, "ny": <4*mB>}}]
        where the keys under each participant's NAME, plus "level", are
        handed to that participant's PROCESS in the environment variable
        OPENPASO_CONFIG_JSON (a JSON object; OPENPASO_LEVEL carries the level
        alone). openPASO writes NO file into your directories: most served
        contracts merge OPENPASO_CONFIG_JSON over their own ./config.json, and
        any script must read it the same way
        (json.loads(os.environ.get("OPENPASO_CONFIG_JSON", "{}")) merged over
        its config) or be run with one couple() call per level instead; a
        level that hands keys to a script that never reads them is refused
        before anything runs. Halve h per
        level as the task prescribes, i.e. double every cell count; a
        transient side's step count travels the same way ("n_steps", which
        the served transient contracts read), so both sides change it
        together;
      * each level starts from the previous level's converged interface
        state (the driver's warm start), which is why the levels must run in
        the same work directories;
      * each level writes its measured iteration history to
        <history_dir>/<history_pattern with {k} = the level> -- pass
        `history_pattern` as the per-level history file name YOUR TASK
        prescribes (it must contain "{k}"); the default is a neutral name
        you would have to rename -- and keeps each side's solver console
        as participant_output_level<k>.log next to its exports.json;
      * the reply carries, per level, the verdict, the iteration count, the
        history path and the interface tables ready to save. It stops at the
        first level that does not converge; fix that level and call again
        from it (the earlier levels' files stay).
    
    Everything `couple` checks is checked here too (it IS couple, per level).
    `history_dir` defaults to the task's working directory. THE CRITIC REVIEW
    IS THE ONE `couple` TAKES: submit_critic_review(solver='couple',
    coupling_args=<{"participants": ..., "max_iter": ..., "tol": ...,
    "accelerator": ..., "theta": ..., "probe": ...} exactly as passed here>),
    then critic_approved=True; openPASO looks that review up for every level.
    ```
