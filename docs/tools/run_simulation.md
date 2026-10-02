# `run_simulation`

**Runs a Python-based solver script (FEniCSx, NGSolve, scikit-fem, DUNE-fem).**

Group: Run.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `solver` | string | yes |  |
| `input_content` | string | no | `''` |
| `input_path` | string | no | `''` |
| `job_name` | string | no | `''` |
| `np` | integer | no | `1` |
| `critic_approved` | boolean | no | `False` |
| `critic_token` | string | no | `''` |
| `verify_pde` | string | no | `''` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Run a simulation directly with input content.
    
    Use this for Python-based solvers (FEniCS, NGSolve, scikit-fem, DUNE-fem)
    where the input IS a Python script. The tool routes through the correct
    Python environment automatically (e.g., conda env for FEniCS).
    
    For 4C/deal.II/Kratos where a separate input file must be generated
    first, use run_with_generator() instead.
    
    Args:
        solver: Backend name (best for: fenics, ngsolve, skfem, dune)
        input_content: The input content (Python script / YAML / C++ / XML)
        input_path: instead of input_content, the name of a file in the run
            folder whose text is the input. Use it to run a file the critic
            reviewed: what runs is then exactly the reviewed text.
        job_name: Optional job name
        np: MPI processes
        critic_approved: recorded, not trusted. The result is verified only
            if a critic review of THIS input_content is on record — call
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
