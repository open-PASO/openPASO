# The tools the AI model gets

openPASO gives the AI model **25 tools**. You never call them yourself: you describe what you want, and the model decides which tool to use. This page tells you what each one is for, so you can follow what the model is doing.

Each tool page also shows the full description the model reads. It is long and technical on purpose: it is written for the model, not for you.

## Find out

| Tool | What it is for |
|---|---|
| [`discover`](discover.md) | Lists the solvers, which ones are installed on this machine, and what each can do. |
| [`knowledge`](knowledge.md) | Answers questions about a solver: physics, settings, known traps, materials, coupling, and what an error message means. |
| [`examples`](examples.md) | Finds real input files from the solvers' own test suites to start from. |
| [`prepare_simulation`](prepare_simulation.md) | The usual first step: collects knowledge, examples and a starting input file in one call. |
| [`session_insights`](session_insights.md) | Summarises what happened so far in this session. |

## Run

| Tool | What it is for |
|---|---|
| [`run_simulation`](run_simulation.md) | Runs a Python-based solver script (FEniCSx, NGSolve, scikit-fem, DUNE-fem). |
| [`run_with_generator`](run_with_generator.md) | Writes an input file from a template and runs a compiled solver (4C, deal.II, Kratos, FEBio, SPARTA). |
| [`check_input`](check_input.md) | Checks an input file for known mistakes before anything is run. |
| [`generate_mesh`](generate_mesh.md) | Builds a mesh with Gmsh: a rectangle, an L-shape, a plate with a hole, a channel, or your own shape. |
| [`visualize`](visualize.md) | Makes plots and statistics from a result, with automatic sanity checks. |

## Check the answer

| Tool | What it is for |
|---|---|
| [`verify_mesh_independence`](verify_mesh_independence.md) | Refines the mesh step by step and reports whether the answer stopped changing. |
| [`verify_pde_consistency`](verify_pde_consistency.md) | Checks that the computed field actually satisfies the equation it claims to solve. |
| [`verify_interface_flux`](verify_interface_flux.md) | For two coupled solvers: checks that what leaves one side arrives at the other. |
| [`audit_results`](audit_results.md) | Reads the result files and names anything missing, inconsistent or suspicious. |
| [`submit_critic_review`](submit_critic_review.md) | Records the independent review of the setup that has to happen before a run. |

## Two solvers on one problem

| Tool | What it is for |
|---|---|
| [`couple`](couple.md) | Runs one problem split across two solvers and repeats the exchange until both sides agree. |
| [`couple_levels`](couple_levels.md) | Does the same on several mesh sizes, so the accuracy of the coupled answer can be measured. |
| [`coupled_solve`](coupled_solve.md) | A ready-made coupled setup for common pairs of physics. |
| [`couple_precice`](couple_precice.md) | Couples two solvers through the preCICE library instead of openPASO's own driver. |
| [`transfer_field`](transfer_field.md) | Moves a result from one solver's output into another solver's input. |
| [`write_participant_contract`](write_participant_contract.md) | Writes the fill-in participant script for one solver of a coupled problem into your folder: everything around the solve, which you then write yourself. |

## Set up and develop

| Tool | What it is for |
|---|---|
| [`setup_backend`](setup_backend.md) | Explains how to install a missing solver, with the route that works on this system. |
| [`rediscover_backends`](rediscover_backends.md) | Looks for solvers again, for example after you installed one. |
| [`reload_catalog`](reload_catalog.md) | Reloads the knowledge catalogue after it changed. |
| [`developer`](developer.md) | Reads and changes a solver's own source code, then rebuilds it. |
