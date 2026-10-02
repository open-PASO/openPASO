# `knowledge`

**Answers questions about a solver: physics, settings, known traps, materials, coupling, and what an error message means.**

Group: Find out.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `topic` | string | yes |  |
| `solver` | string | no | `''` |
| `physics` | string | no | `''` |
| `signal` | string | no | `''` |
| `category` | string | no | `''` |
| `index` | boolean | no | `False` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Get knowledge about solvers, physics, materials, coupling,
    post-mortems, or input formats.
    
    START WITH THE INDEX when you want pitfalls and do not yet know
    what to ask for: `knowledge(topic='pitfalls', solver=...,
    index=True)` returns a one-screen map — how many entries exist
    per physics and per category, and the exact call for each slice.
    Then narrow. If you already have an error message, skip the
    index and pass it as `signal=` directly.
    
    This is the single entry point for ALL domain knowledge — the
    catalog, the pitfall database, AND the post-mortem record
    store. Wiring post-mortems through this same tool closes the
    self-improvement loop: every prepare_simulation call also
    surfaces the relevant post-mortems so the critic-gate can
    retrieve them at planning time (the openPASO design paper §3.2 / §5
    self-correction loop).
    
    Args:
        topic: What you want to know. Options:
            - "physics" — physics-specific knowledge + matching
              post-mortems (needs solver + physics)
            - "pitfalls" — known pitfalls for a solver. Unfiltered
              returns every entry (comprehensive by design). Narrow
              with `signal=` (the error you saw), `physics=`, or
              `category=`; `index=True` maps what exists first. A
              narrowed answer always states how many entries it held
              back and how to get them.
            - "postmortems" — openPASO's formal post-mortem records,
              filtered by solver + physics + optional signal pattern
              (read them here: they are inside openPASO, not files in
              your working directory). These are the
              audit-trail entries that record WHY each pitfall
              exists; the critic-gate should retrieve them when
              the agent's plan touches the matching (solver,
              physics) area.
            - "materials" — material catalog for a solver
            - "coupling" — cross-solver coupling knowledge
            - "tsi" — thermo-structural interaction patterns
            - "precice" — preCICE comparison
            - "input_guide" — how to write input files for a solver
            - "solver_guidance" — which solver to use for a physics type
            - "hardware" — parallelism, GPU, and hardware acceleration capabilities
            - "overview" — backend-level reference catalog (element
              families, mesh types, solver catalogue, unique
              features). The content under the special "_general"
              knowledge key — for dealii ~5 KB, fenics / ngsolve /
              skfem / kratos / dune ~1-2 KB each. Needs solver=...
            - "cross_backend" — collation pitfalls that surface ONLY
              when porting a problem between two backends (units
              conventions, Tet10/Hex20 node ordering, 'linear
              elastic' semantic drift across backends, Dirichlet
              strong-vs-penalty enforcement, restart file
              incompatibility, MPI launch idioms). Pass the
              optional `physics` arg as a topic filter
              (e.g. 'units', 'mesh', 'bc', 'restart', 'mpi') to
              narrow the response. These pitfalls belong to no
              single backend's catalog because they only fire on
              the delta between two.
            - "install" — how to INSTALL a backend, how openPASO
              finds it, which environment variables matter, the
              first-run failures with the exact message each one
              produces, and — importantly — which claims depend on
              how the backend was COMPILED. Read this when a
              backend reports not_installed, when a run fails
              before any physics happens, or before trusting any
              claim whose signal is an assertion message (deal.II
              compiles those out in Release), a vendor linear
              solver (FEBio without MKL), a complex scalar type
              (dolfinx real vs complex builds) or an accelerator
              style (SPARTA without KOKKOS). Optional solver=...
              narrows it to one backend; with no solver you get
              every backend plus the probe commands. Also
              reachable as "setup", "dependencies", "build_config"
              and "portability".
        solver: Backend name (e.g. 'fenics', 'fourc', 'dealii', 'ngsolve')
        physics: Physics type (e.g. 'poisson', 'linear_elasticity', 'navier_stokes')
        signal: The error text you actually observed. Paste it raw —
            quoting, case and whitespace differences are folded, and
            a paraphrase still matches on distinctive terms. Filters
            `pitfalls`, `postmortems` AND `coupling`. Every result
            states the match mode, so a partial word-overlap is
            labelled weak instead of being presented as an
            identification. No match means the failure mode is not
            catalogued for that backend — it does NOT mean the setup
            is right.
            With topic='coupling' it is free text describing what
            you SAW, ranked against the coupling failure entries.
            Describe the observation, not the mechanism — "it
            converged but the answer is wrong", "the residual
            stops falling and stays there", "the two sides
            stopped agreeing" all route to the right entry. This
            is the fast path when a coupling misbehaves: it
            returns the two or three entries that explain the
            symptom instead of the whole payload.
        category: Narrow pitfalls by kind. In use, commonest first:
            Numerical, API, Input, Syntax, Physics, Integration,
            Performance, Output, Mesh, Validation. Spelling variants
            are folded, so 'numerics' finds 'Numerical'.
        index: For topic='pitfalls', return the map instead of the
            content — entry counts per physics and per category plus
            the call that fetches each. Use it to choose a filter
            before pulling the full set.
    ```
