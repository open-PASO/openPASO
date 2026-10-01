# `write_participant_contract`

**Writes the fill-in participant script for one solver of a coupled problem into your folder: everything around the solve, which you then write yourself.**

Group: Two solvers on one problem.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `solver` | string | yes |  |
| `path` | string | yes |  |
| `variant` | string | no | `''` |
| `overwrite` | boolean | no | `False` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Write the served participant CONTRACT for `solver` to `path`, solve elided.
    
    The file is byte-for-byte the text `knowledge(topic='coupling', solver=...,
    signal='participant[:<variant>]:part<k>')` serves in parts, concatenated:
    the contract with its handshake, its checks and its recovery, and the
    SOLVE elided where the banner sits. It is not a runnable program; fill the
    marked hole(s) yourself. `variant` is '' for the base contract, or one of the words this install's contracts carry: fenics 'thermoelastic', 'elastic', 'transient', 'fsi_fluid', 'fsi_solid'; fourc 'thermoelastic', 'fsi_solid'; ngsolve 'elastic'; skfem 'elastic', 'fsi_solid'; dune 'elastic', '3d'; dealii 'elastic', 'transient'; febio 'elastic'; kratos 'neumann', '3d'; sparta none. The base contract serves both the Dirichlet and the Neumann side for fenics, fourc, ngsolve, skfem, dune, dealii, febio (SIDE in its edit block, or "side" in config.json where it reads config); kratos has a 'neumann' variant for that side. 'fsi' writes a code's one fluid-structure contract (fenics ships two, the fluid and the structure side: name 'fsi_fluid' or 'fsi_solid'); a word that names no contract of the code is refused with the words it has. The knowledge door takes the same words.
    Refuses to overwrite an existing file unless overwrite=True.
    ```
