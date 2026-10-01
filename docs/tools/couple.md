# `couple`

**Runs one problem split across two solvers and repeats the exchange until both sides agree.**

Group: Two solvers on one problem.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `participants` | string or array | yes |  |
| `max_iter` | integer | no | `50` |
| `tol` | number | no | `1e-06` |
| `accelerator` | string | no | `'auto'` |
| `theta` | number | no | `0.5` |
| `monolithic` | string | no | `''` |
| `probe` | boolean | no | `True` |
| `critic_approved` | boolean | no | `False` |
| `noise_replicates` | integer | no | `0` |
| `noise_floor` | number | no | `0.0` |
| `noise_block` | integer | no | `3` |
| `history_path` | string | no | `''` |
| `iface_level` | integer | no | `0` |
| `pde_sources` | string | no | `''` |
| `pde_check` | string | no | `''` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    GENERAL partitioned multi-code coupling — works for ANY physics/coupling.
    
    Have an independent critic review the setup before coupling; pass
    critic_approved=True only after that review (every simulation must be
    critic-reviewed first).
    
    Unlike coupled_solve (legacy, fixed toy geometries), this is physics-agnostic:
    you write one self-contained solver script per subdomain/participant and openPASO
    runs the fixed-point iteration, relaxation, convergence-or-fail, AND the
    silent-wrong validation a partitioned coupling needs, because a partitioned
    coupling's characteristic failure is not a crash — it is a clean, converged,
    confidently wrong number. openPASO checks, and reports in the verdict:
      * convergence, and per-block convergence (a large settled block, e.g. force,
        cannot hide a small moving one, e.g. displacement, inside one global norm);
      * finiteness of every exchanged array, including coordinates and fluxes;
      * interface flux balance, naming SIGN-CONVENTION and UNIT-MISMATCH signatures;
      * that every participant exited 0 — a diverged solver often writes its last
        iterate and then aborts;
      * that every participant's output actually MOVED when its imports moved —
        the test for a participant that exits 0 having done nothing, or re-serves
        a cached answer; such a run "converges" at iteration 2 with residual 0;
      * that the coupling graph is wired as declared — an `imports_from` name that
        matches no participant is REFUSED, not silently dropped into a one-way run;
      * whether the two interface discretisations match;
      * and, when you pass `monolithic`, that the coupled answer equals an
        independent un-split solve of the same problem. That last one is the only
        check that can catch a consistent unit error or a wrongly applied interface
        sign, so when it is not supplied the verdict SAYS it was not run.
    
    PARTICIPANT CONTRACT — each iteration the driver, per participant:
      1. writes <work_dir>/imports.json = {partner_name: InterfaceData} (boundary
         data this participant consumes; empty on iteration 1).
      2. runs your `command` in <work_dir>.
      3. reads <work_dir>/exports.json = the InterfaceData your script produced on
         the shared interface.
    Your script decides HOW to apply imports (Dirichlet/Neumann/Robin/traction/
    flux/...) and WHAT to export — opaque to the driver, so it generalizes.
    
    InterfaceData JSON shape (read imports, write exports):
      {"field_name": str, "n_points": N, "coordinates": [[x,y(,z)],...],
       "values": [...], "normal_fluxes": [...]  # optional, for conservation check}
    
    THE DRIVER IS JACOBI, NOT GAUSS-SEIDEL: within one iteration every
    participant reads the PREVIOUS iteration's exports, and the driver relaxes
    EVERY participant's export vector. A two-participant Dirichlet-Neumann loop
    therefore relaxes twice per cycle and converges geometrically — it does not
    finish in one step even for a linear problem. theta=1.0 (no relaxation)
    oscillates forever on a balanced interface; start at theta=0.5.
    
    FLUX SIGN: export `normal_fluxes` with respect to YOUR OWN outward normal.
    The two normals are anti-parallel, so the two participants' fluxes carry
    OPPOSITE signs and their sums cancel. The Dirichlet value you APPLY is the
    same number on both sides — the opposite rule. Getting this wrong is the
    single most common cause of the flux-balance finding.
    
    Args:
        participants: JSON list of {"name", "command":[argv...], "work_dir",
          "imports_from":[partner names], "timeout": seconds}. Every name in
          `imports_from` must be another participant's name.
        max_iter, tol: iteration controls.
        accelerator: "auto" (default: Aitken for a single-field exchange, Anderson for a
            multi-field one -- resolved from the first exports and reported in `theta.mode`),
            "aitken" (theta recomputed each iteration from the residual
            or "anderson" (Anderson mixing / interface quasi-Newton on the whole interface state, window 5:
            measured on a three-component thermo-elastic exchange to cut the iteration count several-fold)
          history, starting at `theta`) or "constant" (theta held at `theta` for
          the whole run). There is no per-field or per-participant theta.
        theta: the relaxation factor. Under-relaxation (theta < 1) is what makes
          a Dirichlet-Neumann or FSI coupling converge at all when the physical
          stiffness/density ratio makes the un-relaxed iteration diverge; 0.5 is
          a neutral default, not a recommendation for your problem.
        noise_replicates: for a STOCHASTIC participant (DSMC / Monte-Carlo /
          any sampled estimator). Use 4 or more — the floor is itself an
          estimate and three samples is a bad one. It makes the driver run
          every participant that many times on the SAME imports and MEASURE
          the residual noise floor — the residual a perfectly converged run
          would still report. Convergence is then judged against
          max(tol, floor), over a block mean, so a correct stochastic
          coupling is no longer reported as a failure just because tol sits
          under the sampler's own scatter. 0 (the default) switches the whole
          branch off. Deterministic participants measure a floor of exactly 0
          and are unaffected.
        noise_floor: declare a floor instead of measuring one (or raise a
          measured one), if you established it independently.
        noise_block: how many consecutive residuals must AVERAGE below the
          criterion before the run stops. Only in effect when a non-zero
          floor is; a single residual dipping into the noise means nothing.
        probe: after the iteration settles, spend ONE extra solve per
          participant perturbing its final imports and measuring how far its
          answer moves. This is the only check here that can tell a solver
          which reads its boundary data from one that merely looks as if it
          does; turn it off only if that solve is genuinely unaffordable, and
          the verdict will then record that the question was not asked.
        monolithic: OPTIONAL JSON {"command":[argv...], "work_dir": str,
          "timeout": int} — a solve of the SAME problem un-split, in ONE code,
          which writes <work_dir>/monolithic.json in InterfaceData shape on the
          same interface. Supplying it is the strongest verification available
          here and needs no external benchmark.
                    history_path: optional ABSOLUTE CSV path. When set, openPASO writes its
                        measured finite residuals there as
                        ``iteration,interface_residual``. Use the path of the per-level
                        residual-history file your task names; never retype the returned history.
    
    iface_level: optional level number stamped into the suggested_filename of the interface_csv blocks the reply carries on convergence (each participant's own final interface data, ready to save verbatim).
    
    pde_check: THE ONE CHECK THAT SEPARATES A RIGHT ANSWER FROM A CONVERGED WRONG ONE, run for you on every converged level. JSON keyed by participant name: {"A": {"solution_files": "<this side's per-level field files, comma-separated, in level order>", "equation": "<your task's EQUATION line, verbatim>", "source": "<that side's source, as your task wrote it>", "coefficient": "<that side's coefficient>", "domain": "[[x0,x1],[y0,y1]]"}, "B": {...}}. The files must be on the PROBE GRID your task prescribes, not your mesh nodes: the test is a midpoint quadrature and on a scatter it reports nothing rather than a misleading number. You name them; openPASO guesses no filename and opens no task file. Give it once and openPASO puts each side's own field back into that side's own equation at every level from the second on, and reports CONSISTENT or INCONSISTENT in `pde_consistency`. A refinement study cannot do this: a field that converges cleanly to the WRONG function is indistinguishable in one -- measured on three recorded coupled runs whose every self-consistency measure reported a converging run while the answer was wrong. Nothing here is read from any task file; these are YOUR strings, and openPASO neither stores nor supplies them. A side whose operator this check does not model is refused by name and the rest still run.
    
    pde_sources: OPTIONAL, public-only. JSON {"A": {"source": "<the forcing/coefficient you actually implemented>", "task_source": "<the task's stated source, verbatim>"}, "B": {...}}. When supplied, openPASO compares the two PUBLIC strings and flags a mismatch — a silent wrong forcing (right shape, wrong function) converges cleanly to a different answer and no self-consistency check can see it. Never required; openPASO reads no reference solution and never supplies the equation for you.
    
    Returns: JSON with converged, iterations, residual, per-block residuals,
        exports, the coupling graph, per-participant responsiveness and exit
        codes, a `validation` block, a `checks_not_run` block, and the verdict.
        A coupling that failed any check is reported as NOT VERIFIED, never as a
        trustworthy result — and one that could not be fully checked says so.
        With the stochastic branch on it also returns `noise_floor`,
        `tol_effective` and `stopped_at_noise_floor`. READ `noise_floor`
        BEFORE JUDGING CONVERGENCE: any tolerance applied to a result that
        carries one — including an acceptance tolerance — must be at least
        that floor, or it is measuring the sampler rather than the coupling.
    ```
