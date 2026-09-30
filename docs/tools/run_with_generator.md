# `run_with_generator`

**Writes an input file from a template and runs a compiled solver (4C, deal.II, Kratos, FEBio, SPARTA).**

Group: Run.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `solver` | string | yes |  |
| `generator_script` | string | no | `''` |
| `generator_path` | string | no | `''` |
| `job_name` | string | no | `''` |
| `np` | integer | no | `1` |
| `critic_approved` | boolean | no | `False` |
| `critic_token` | string | no | `''` |
| `verify_pde` | string | no | `''` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Run a generator script that creates an input file, then execute the solver.
    
    Use this for solvers that need a COMPILED binary or separate input files:
    - 4C: generator creates .4C.yaml + mesh, then 4C binary runs on them
    - deal.II: generator creates main.cpp, then cmake + make + ./fem_solve
    - Kratos (with real binary): generator creates ProjectParameters.json +
      .mdpa + MainKratos.py, then Kratos Python runs MainKratos.py
    
    DO NOT use this for:
    - FEniCS, NGSolve, scikit-fem, DUNE-fem: use run_simulation() instead
    - Kratos manual-assembly scripts (numpy/scipy): use run_simulation()
      since those are standalone Python scripts, not input-file generators
    
    The generator script runs in the server's Python. It must produce an
    input file matching one of: *.4C.yaml, *.yaml, input.*, solve.py,
    MainKratos.py
    
    Args:
        solver: Backend name (fourc, dealii, kratos)
        generator_script: Python script that creates the input file
        generator_path: instead of generator_script, the name of a file in
            the run folder holding it. Use it to run a generator the critic
            reviewed: what runs is then exactly the reviewed text.
        job_name: Optional job directory name
        np: MPI processes (default 1)
        critic_approved: recorded, not trusted. The result is verified only
            if a critic review of THIS generator_script is on record — call
            submit_critic_review first.
        critic_token: optional token from submit_critic_review; makes the
            review single-use and binds it to this job.
        verify_pde: optional JSON declaring the problem being solved, so
            openPASO can check the result actually SATISFIES it rather than
            merely looking well-formed. Example:
            {"operator": "diffusion",
             "source": "2*pi**2*sin(pi*x)*sin(pi*y)",
             "coefficient": "1.0", "dim": 2, "domain_measure": 1.0}
            `source` and `coefficient` are numeric expressions in x, y, z.
            A field that does not satisfy the declared equations is NOT
            VERIFIED, whatever else the run did. Currently covers scalar
            diffusion on simplex meshes; anything else is reported as not
            checked, never as passed.
    ```
