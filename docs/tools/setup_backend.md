# `setup_backend`

**Explains how to install a missing solver, with the route that works on this system.**

Group: Set up and develop.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `action` | string | no | `'status'` |
| `solver` | string | no | `''` |
| `route` | string | no | `''` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Guided backend setup: detect, plan, install, verify, persist.
    
    Helps a new user get any of the 8 FEM backends working on THIS
    machine — picking the fastest install route for the current OS
    (pip > conda > binary download > source build), executing it,
    running the backend's smoke test, and persisting the resolved
    paths into ~/.config/openpaso/sources.json so every future MCP
    session finds the install without re-discovery.
    
    Actions:
      status  — one-row-per-backend table: available? source tree?
                build dir? (no args needed)
      plan    — recommended install route for `solver` on this OS,
                incl. system deps (apt/brew), human notes, and
                whether the route is verified on this OS. Nothing
                executes. Optional `route` (pip|conda|binary|source|
                spack) forces an alternative.
      install — execute the planned route. pip/conda run inline
                (minutes); source builds start in the BACKGROUND
                (30-120 min) — re-run with action='verify' when
                done. binary routes return manual instructions.
                route='spack' (dealii, sparta, fourc) builds the solver
                and its dependencies with Spack in the BACKGROUND
                (minutes for SPARTA, over an hour for 4C) from the
                recipes openPASO ships; the result names the log. Re-run
                with action='verify' when the log's last line is `[+]`
                for the package itself.
      verify  — run the smoke test for `solver` and, on success,
                persist its paths.
    
    macOS note: darwin routes are structured but mostly UNVERIFIED
    (flagged in the plan output). They are extension points — when
    a route is validated on a Mac, its os_support['darwin'] entry
    in src/core/backend_setup.py should be updated with the working
    steps (e.g. the 4C brew/CMake settings).
    
    Args:
        action: status | plan | install | verify
        solver: backend name (required for plan/install/verify)
        route:  optional route kind override (pip|conda|binary|source|spack)
    ```
