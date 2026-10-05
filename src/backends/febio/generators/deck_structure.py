"""Executed-verified .feb deck-authoring surface for FEBio 4.12.

Everything in this module was obtained by RUNNING the installed
binary, not by reading documentation. Two ground-truth sources
were used and cross-checked:

  1. The binary's own factory registry. ``febio4`` has an
     interactive console; the ``list`` command dumps every
     registered factory as ``<module>.<type string> [SUPER_CLASS]``
     and ``list -m`` dumps the module names. Piping
     ``printf 'list\\nquit\\n' | febio4 -nosplash`` therefore
     yields the authoritative type-string vocabulary FOR THE
     INSTALLED BUILD — including which optional libraries were
     compiled in. This is strictly better than grepping
     REGISTER_FECORE_CLASS out of the sources, because a source
     tree lists features that a given cmake configuration never
     compiles (see ``linear_solvers`` below: ``pardiso`` is in
     the sources but absent from a USE_MKL=OFF build).

  2. Minimal .feb decks — one hex8 element, or a 2x2x2 patch —
     run through ``febio4 -i in.feb``, one deviation from a
     known-good baseline at a time, recording exit code, the
     "Reading file ... SUCCESS!/FAILED!" line, the verbatim
     ERROR box, the termination banner, and the numbers written
     to the logfile CSVs.

PROVENANCE: all entries below carry the marker
"live 2026-08-03 / FEBio 4.12.0.86045466d" and were produced on
a source build (github.com/febiosoftware/FEBio @ 8604546,
USE_MKL=OFF) symlinked at ~/FEBio/bin/febio4.

VERSION DRIFT WARNING: the installed build reports itself as
``version 4.12.0.86045466d``. Ten modules are registered. There
is NO ``heat`` module and NO ``biphasic-FSI`` module in this
build, although both strings appear in older openPASO catalog
entries and in FEBio documentation for earlier lines. Feeding
either string to ``<Module type=...>`` does not raise an error —
it segfaults the process (see ``module_types`` below).
"""

# ---------------------------------------------------------------
# Registered module names — `list -m` on the installed binary.
# `<Module type="X"/>` must be one of these, byte for byte.
# ---------------------------------------------------------------
FEBIO_MODULES: tuple[str, ...] = (
    "solid",
    "biphasic",
    "solute",
    "multiphasic",
    "fluid",
    "fluid-FSI",
    "multiphasic-FSI",
    "fluid-solutes",
    "thermo-fluid",
    "polar fluid",
)

# `<analysis>` is an enum whose vocabulary is installed BY THE
# ACTIVE MODULE (FE*Analysis::FindParameterFromData(&m_nanalysis)
# ->setEnums(...)), so the legal words differ per module. Verified
# live: a word from the wrong module's vocabulary is a hard parse
# error, not a silent fallback.
FEBIO_ANALYSIS_VALUES: dict[str, tuple[str, ...]] = {
    "solid": ("STATIC", "DYNAMIC"),
    "biphasic": ("STEADY-STATE", "TRANSIENT"),
    "solute": ("STEADY-STATE", "TRANSIENT"),
    "multiphasic": ("STEADY-STATE", "TRANSIENT"),
    "fluid": ("STEADY-STATE", "DYNAMIC"),
    "fluid-FSI": ("STEADY-STATE", "DYNAMIC"),
    "multiphasic-FSI": ("STEADY-STATE", "DYNAMIC"),
    "fluid-solutes": ("STEADY-STATE", "DYNAMIC"),
    "thermo-fluid": ("STEADY-STATE", "DYNAMIC"),
    "polar fluid": ("STEADY-STATE", "DYNAMIC"),
}


DECK_KNOWLEDGE = {
    "description": (
        "Executed-verified .feb authoring surface for the installed "
        "FEBio build. Every claim here was reproduced by running "
        "febio4 on a minimal deck; the counterexample decks are "
        "kept as Tier-2 fixtures under "
        "scripts/tier2_fixtures/febio/."),
    "verified_against": (
        "FEBio 4.12.0.86045466d (source build of "
        "github.com/febiosoftware/FEBio @ 8604546, USE_MKL=OFF, "
        "skyline default linear solver), live 2026-08-03."),

    "how_to_enumerate_features_yourself": (
        "febio4 has an interactive console. `printf 'list -m\\nquit\\n' "
        "| febio4 -nosplash` prints the registered MODULE names; "
        "`printf 'list\\nquit\\n' | febio4 -nosplash` prints every "
        "registered factory as `<module>.<type string> "
        "[SUPER_CLASS_ID]`; `list -c <SUPER_CLASS_ID>` filters and "
        "`list -s <substring>` greps. This is the only reliable way "
        "to learn which optional features a particular BUILD has — "
        "the source tree registers many more than any one cmake "
        "configuration compiles. On the verified build the dump has "
        "1429 factories: 159 FEMATERIAL_ID, 288 FEPLOTDATA_ID, "
        "286 FELOGELEMDATA_ID, 75 FELOAD_ID, 49 FEBC_ID, "
        "20 FELINEARSOLVER_ID, 10 FEANALYSIS_ID."),

    "required_skeleton": (
        "A deck that actually solves needs, at minimum: "
        "<febio_spec version=\"4.0\"> (the version attribute is "
        "mandatory) / <Module type=...> as the FIRST child / "
        "<Control> containing a <solver type=...> child / "
        "<Material> with at least one named <material> / <Mesh> "
        "with <Nodes> + <Elements> / <MeshDomains> binding the "
        "element block to the material BY NAME / <Boundary>. "
        "<Output> is optional: FEBio writes <jobname>.xplt with "
        "`displacement` and `stress` even with no <Output> section "
        "at all (verified by byte-comparing the .xplt of one deck "
        "run with and without the section: identical size, 1630 "
        "bytes on that deck — the SIZE is deck-specific, the "
        "IDENTITY is the claim). TEXT output is NOT free — CSV "
        "files only appear if you ask for them with "
        "<Output><logfile>."),

    "spec_version_attribute": (
        "<febio_spec version=\"X\"> SELECTS THE SCHEMA, and FEBio "
        "4.12 still reads several. Verified live 2026-08-03 by "
        "feeding one 4.0-shaped deck through every value: "
        "`1.2`, `2.0`, `2.5`, `3.0` and `4.0` are all ACCEPTED "
        "version strings; `1.0` and `5.0` are rejected outright "
        "with `Invalid version (line N)`; omitting the attribute "
        "gives tag \"febio_spec\" (line N) : `missing attribute "
        "\"version\"`. "
        "The danger is that an accepted-but-wrong version number "
        "does NOT fail on the version line — it fails much later, "
        "inside whichever section first differs, and the message "
        "points at that section instead of at the header. On the "
        "SAME 4.0 deck: labelling it `2.5` (or `2.0`, or `1.2`) "
        "dies with `tag \"analysis\" (line 5) : missing attribute "
        "\"type\"` because the 2.x schema wants "
        "<analysis type=\"static\"/>; labelling it `3.0` dies with "
        "`tag \"solver\" (line 8) : invalid attribute \"type\"` "
        "because in 3.0 <solver> is not a typed property. If a "
        "deck fails inside <Control> for no apparent reason, "
        "check the version attribute FIRST. Everything else in "
        "this catalog assumes version=\"4.0\"."),

    "section_order": (
        "The .feb reader is SINGLE-PASS: a section can only "
        "reference names that an EARLIER section defined. Verified "
        "live, the hard constraints are exactly four: "
        "(1) <Module> must be the first tag — any other first tag, "
        "or no <Module> at all, aborts with `First tag must be the "
        "Module section.`; "
        "(2) <Material> must precede <MeshDomains>, else "
        "tag \"SolidDomain\" (line N) : `invalid value for "
        "attribute \"mat\"`; "
        "(3) <Mesh> must precede <MeshDomains> — and this one has "
        "NO diagnostic at all: it SEGFAULTS (exit 139, rc -11), "
        "with stdout truncated at `Reading file in.feb ...`, no "
        "SUCCESS!/FAILED! line and no ERROR box; "
        "(4) <MeshDomains> must precede <Boundary> — node sets are "
        "only resolvable once the domains are built, so putting "
        "<Boundary> between <Mesh> and <MeshDomains> still fails. "
        "Constraint (4) alone produces tag \"bc\" (line N) : `"
        "invalid value for attribute \"node_set\"`, which names the "
        "BC rather than the section that is out of place — check "
        "the order of MeshDomains / Boundary before suspecting the "
        "node set itself. "
        "CORRECTED 2026-08-03 by running all six permutations of "
        "{Mesh, MeshDomains, Boundary} on one deck: the previously "
        "catalogued claim that constraints (3) and (4) SHARE the "
        "`tag \"bc\"` message is FALSE. Measured — "
        "Mesh<MeshDomains<Boundary: rc 0, normal termination; "
        "Mesh<Boundary<MeshDomains: rc 1, `tag \"bc\" ...`; "
        "MeshDomains<Mesh<Boundary: rc -11, SIGSEGV, silent; "
        "MeshDomains<Boundary<Mesh: rc -11, SIGSEGV, silent; "
        "Boundary<Mesh<MeshDomains: rc 1, `tag \"bc\" ...`; "
        "Boundary<MeshDomains<Mesh: rc 1, `tag \"bc\" ...`. "
        "So: whenever <MeshDomains> is the FIRST of the three the "
        "process dies with no output whatsoever — do not wait for "
        "a `tag \"bc\"` line that will never come. "
        "Everything else floats: <Control>, <LoadData> and "
        "<Output> were each verified to run correctly in first, "
        "middle and last position (bit-identical nodal "
        "displacements in all three placements)."),

    "matching_error_text": (
        "FEBio wraps its ERROR box at 71 columns, so a long "
        "diagnostic is SPLIT ACROSS LINES inside the star frame. "
        "`The selected linear solver does not support the "
        "requested matrix format.` is emitted as `... the "
        "requested matrix` / `format.` on two lines and a "
        "fixed-string grep for the whole sentence returns nothing. "
        "Match on a short distinctive fragment that cannot wrap "
        "(`does not support the requested`), or strip the frame "
        "and re-join before matching. Verified live 2026-08-03: "
        "the shorter messages quoted throughout this catalog "
        "(`Invalid load curve ID`, `First tag must be the Module "
        "section.`, `Invalid element type`, every `tag \"x\" (line "
        "N) : ...` form) all fit on one line and grep cleanly."),

    "module_types": {
        "registered": list(FEBIO_MODULES),
        "note": (
            "Byte-exact and case-SENSITIVE. `polar fluid` really "
            "does contain a space. `fluid-FSI` and "
            "`multiphasic-FSI` really do have an uppercase FSI. "
            "There is no `heat` module and no `biphasic-FSI` "
            "module in 4.12."),
        "failure_mode": (
            "FEBioModuleSection.cpp calls "
            "FEModelBuilder::SetActiveModule(szt) with NO existence "
            "check, so an unrecognised module string dereferences a "
            "null module and the process dies with SIGSEGV. "
            "Verified live for `heat`, `biphasic-FSI`, `Solid` and "
            "`SOLID` — all exit 139 with stdout truncated mid-line "
            "at `Reading file in.feb ...`, no FAILED!, no ERROR "
            "box, no log file."),
    },

    "analysis_values": {
        "per_module": {k: list(v) for k, v in
                       FEBIO_ANALYSIS_VALUES.items()},
        "note": (
            "Enum VALUES are matched case-insensitively (`static`, "
            "`Static` and `STATIC` all work) and the raw ordinal is "
            "also accepted (`<analysis>1</analysis>` == DYNAMIC in "
            "the solid module — verified by identical nodal "
            "displacements). Contrast with factory TYPE STRINGS in "
            "`type=` attributes, which are case-SENSITIVE. A word "
            "from another module's vocabulary is a clean parse "
            "error: tag \"analysis\" (line N) : invalid value: "
            "STEADY-STATE in a solid deck. An out-of-range ordinal "
            "passes the reader and fails at init with `Invalid "
            "value for parameter: .analysis`."),
    },

    "linear_solvers": {
        "registered_on_this_build": [
            "diagonal", "LU", "skyline", "pardiso-project", "fgmres",
            "boomeramg", "cg", "schur", "hypre_gmres", "hypre_pcg_amg",
            "block", "bipn", "bicgstab", "strategy", "test",
            "accelerate", "superlu_mt", "ilu0", "ilut", "ichol",
        ],
        "note": (
            "`pardiso` is NOT a valid type string on a USE_MKL=OFF "
            "build — `<linear_solver type=\"pardiso\">` fails at "
            "parse with tag \"linear_solver\" (line N) : `invalid "
            "value for attribute \"type\"`. The MKL-free spelling "
            "that IS registered is `pardiso-project`, and even that "
            "one rejects unsymmetric matrices in this build. The "
            "startup banner prints the fallback it picked: "
            "`Default linear solver: skyline`."),
        "unsymmetric": (
            "Anything that sets <symmetric_stiffness>non-symmetric"
            "</symmetric_stiffness> — which the biphasic, "
            "multiphasic, fluid and FSI solvers do by default — "
            "needs a solver that accepts an unsymmetric sparse "
            "format. Verified live on the USE_MKL=OFF build by "
            "sweeping all 20 registered solvers on a "
            "confined-compression biphasic deck: skyline, fgmres, "
            "cg, schur, bipn, superlu_mt, accelerate, diagonal, "
            "hypre_gmres, hypre_pcg_amg, boomeramg, ichol, "
            "pardiso-project and strategy abort with `The selected "
            "linear solver does not support the requested matrix "
            "format.` (the composite solvers schur / strategy / "
            "block may instead stop earlier at parse, demanding "
            "their sub-solver properties — `Component "
            "\"linear_solver\" needs to have property \"A_solver\" "
            "defined` etc. — depending on how they are "
            "configured); `LU` aborts with `Linear solver failed "
            "to find solution`; `ilu0` and `ilut` with `Fatal "
            "error in factorization of stiffness matrix`; `block` "
            "with `An error occurred during preprocessing of "
            "linear solver`. Two solvers RUN: `bicgstab`, which is "
            "the only real one, and `test`, which is not a solver "
            "at all (see `null_test_solver` below)."),
        "bicgstab_caveat": (
            "`bicgstab` is the recommended unsymmetric solver on a "
            "USE_MKL=OFF build, but it is registered with an "
            "OPTIONAL `pc_left` preconditioner property that "
            "defaults to none, and unpreconditioned BiCGStab is "
            "NOT robust. It fails with `Linear solver failed to "
            "find solution. Aborting run.` and "
            "`------- failed to converge at time : <t>`. "
            "CORRECTED 2026-08-03: this caveat previously blamed "
            "MESH SIZE ('converged on a single-element deck, "
            "failed on a 2x2x2'). Mesh size is not the "
            "discriminator — load magnitude is. Measured on ONE "
            "hex8 confined-compression biphasic deck, "
            "10 x 0.1 transient steps, neo-Hookean E=1000 v=0, "
            "perm-const-iso 1e-3, varying only the prescribed top "
            "displacement: uz=-0.001 converges (10 steps), "
            "uz=-0.01 converges (10 steps), uz=-0.05 FAILS at "
            "step 1, uz=-0.1 FAILS at step 1. So a single element "
            "is no guarantee. Treat bicgstab as a way to get a "
            "lightly-loaded model to run at all, not as a "
            "production unsymmetric solver, and re-check "
            "convergence after every load change. If real "
            "biphasic / fluid / FSI work is needed, rebuild FEBio "
            "with MKL (-DUSE_MKL=ON) so the `pardiso` "
            "factorisation is available. (Live-verified "
            "2026-08-03.)"),
        "null_test_solver": (
            "DO NOT USE <linear_solver type=\"test\"/>. It is a "
            "registered FELINEARSOLVER_ID factory whose "
            "TestSolver::BackSolve (NumCore/TestSolver.cpp) sets "
            "every entry of the solution vector to 0 and returns "
            "true. FEBio therefore takes zero Newton increments, "
            "declares convergence, completes every time step and "
            "ends with `N O R M A L   T E R M I N A T I O N` and "
            "exit 0. It is the one verified case on this build "
            "where the termination banner AND a non-zero "
            "completed-step count are both satisfied by a run that "
            "solved nothing. What comes back is exactly what the "
            "prescribed boundary conditions impose by themselves, "
            "with EVERY free degree of freedom left at zero: on a "
            "biphasic deck the fluid pressure — the entire point "
            "of a biphasic run — is identically zero at every "
            "node, and on a solid deck with a prescribed uz the "
            "null solver reproduces uz exactly and silently drops "
            "the lateral Poisson response (ux = uy = 0). The size "
            "of the resulting error is therefore deck-dependent "
            "and can be anything from enormous to invisible: DO "
            "NOT rely on the answer merely looking wrong. Detect "
            "it structurally instead — grep the deck for "
            "`type=\"test\"`, or check that a field which must be "
            "non-zero (a reaction, a pressure, a lateral "
            "displacement) is in fact non-zero. (Live-verified "
            "2026-08-03.)"),
    },

    "control_section": (
        "<Control> REQUIRES a <solver type=\"<module name>\"> child "
        "property in the 4.0 schema — omitting it aborts at parse "
        "with `Component \"\" needs to have property \"solver\" "
        "defined (line N)`. Quasi-Newton settings live INSIDE that "
        "solver, one level deeper than in 3.x: `max_ups` directly "
        "under <solver> is rejected (tag \"max_ups\" (line N) : `"
        "unrecognized tag`); the 4.0 spelling is "
        "<solver><qn_method type=\"BFGS\"><max_ups>10</max_ups>"
        "</qn_method></solver>. Five qn_method types are registered "
        "and all five are accepted: \"BFGS\", \"Broyden\", \"JFNK\", "
        "\"modified Newton\", \"full Newton\" (a sixth spelling gives "
        "tag \"qn_method\" (line N) : `invalid value for attribute "
        "\"type\"`). BUT <max_ups> is a parameter of the SECANT "
        "methods only: it is accepted under \"BFGS\" and \"Broyden\" "
        "and REJECTED under \"JFNK\", \"modified Newton\" and \"full "
        "Newton\" with tag \"max_ups\" (line N) : `unrecognized tag` — "
        "the same message as putting it directly under <solver>, so "
        "do not read that message as proof that the nesting is wrong; "
        "check which qn_method you named. Written bare, without "
        "<max_ups>, \"modified Newton\" and \"full Newton\" run "
        "normally; \"JFNK\" completes 0 time steps and exits 1 on a "
        "one-element solid deck. (Executed 2026-08-03, FEBio "
        "4.12.0.86045466d.)"),

    "final_time_formula": (
        "FINAL TIME = time_steps x step_size. FEAnalysis::Solve() "
        "loops `while (endtime - currentTime > eps)` with "
        "`endtime = m_dt0 * m_ntime`, so <time_steps> is NOT a step "
        "count when an auto <time_stepper> is active — it only sets "
        "the end of the interval together with <step_size>. "
        "Verified: time_steps=10 + step_size=1.0 runs to t = 10; "
        "time_steps=10 + step_size=0.1 runs to t = 1."),

    "output_naming": {
        "plot_variables": (
            "<Output><plotfile type=\"febio\"><var type=\"NAME\"/> "
            "names come from the FEPLOTDATA_ID factories and are "
            "MODULE-SCOPED and case-SENSITIVE. Solid-module basics: "
            "`displacement`, `velocity`, `acceleration`, `stress`, "
            "`relative volume`, `PK1 stress`, `PK2 stress`, "
            "`nodal stress`, `Lagrange strain`, `contact pressure`, "
            "`reaction forces`, `fiber vector`, `damage`. All "
            "thirteen EXECUTED 2026-08-03 on a solid deck: each one "
            "runs to normal termination. There is NO `von Mises "
            "stress` variable — request `stress` (the full tensor) "
            "and reduce it downstream. Asking for one that does not "
            "exist is an INIT-TIME failure, not a parse error: the "
            "deck reads `...SUCCESS!` first, then FEBio prints "
            "`FATAL ERROR: Output variable \"von Mises stress\" is "
            "not defined` and stops. A wrapper watching only for "
            "`Reading file ...FAILED!` will miss it."),
        "log_variables": (
            "<Output><logfile><node_data data=\"...\"> and "
            "<element_data data=\"...\"> take a SEMICOLON-separated "
            "list of FELOG*DATA_ID names, also module-scoped and "
            "case-sensitive. Solid nodes: x y z ux uy uz vx vy vz "
            "ax ay az Rx Ry Rz. Solid elements: J, sx sy sz sxy syz "
            "sxz (note the Voigt order is xy, yz, xz), Ex..Exz "
            "(Green-Lagrange, FEElasticMaterialPoint::Strain), "
            "ex..exz (INFINITESIMAL strain, "
            "FEElasticMaterialPoint::SmallStrain — not an "
            "Almansi measure), Ux..Uxz "
            "(right stretch), `effective stress`. EXECUTED "
            "2026-08-03: every one of those node and element lists "
            "runs to normal termination and writes its table. "
            "Omitting the file= attribute is legal and dumps the "
            "table into the run's .log instead of a separate CSV. "
            "Copy-ready: <Output><logfile><node_data data=\"x;y;z\" "
            "delim=\",\" file=\"pos.csv\"/><element_data "
            "data=\"sx;sy;sz;J\" delim=\",\" file=\"el.csv\"/>"
            "</logfile></Output>."),
        "small_strain_shear_bug": (
            "UPSTREAM BUG in FEBio 4.12: the `eyz` and `exz` log "
            "variables are SWAPPED. "
            "FEElasticMaterialPoint::SmallStrain() builds its "
            "return value as mat3ds(F00-1, F11-1, F22-1, "
            "0.5(F01+F10), 0.5(F02+F20), 0.5(F12+F21)) while the "
            "mat3ds constructor signature is (xx, yy, zz, xy, yz, "
            "xz) — so the xz shear is stored in the yz slot and "
            "vice versa. Reproduce it in one run: prescribe a "
            "homogeneous deformation with exactly ONE non-zero "
            "off-diagonal shear, say "
            "F = [[1.02,0.01,0],[0,0.99,0.005],[0,0,1.01]], which "
            "has a yz shear and no xz shear. The yz shear you "
            "prescribed comes back in the exz column and the eyz "
            "column reads machine zero — the two are transposed. "
            "The diagonals and exy are unaffected, and the "
            "Green-Lagrange Ex..Exz family reports the same "
            "deformation in the correct slots, which is what "
            "identifies this as an output-mapping bug rather than "
            "a wrong solution. Use Ex..Exz, or swap eyz/exz in "
            "post-processing. (Executed 2026-08-03, FEBio "
            "4.12.0.86045466d.)"),
        "csv_format": (
            "Each recorded step is a block of three header lines "
            "`*Step`, `*Time`, `*Data` followed by one row per "
            "item, `id,value,value,...`, at ~12 significant "
            "digits. Step 0 (the undeformed state) is always "
            "written, so a file that contains ONLY a step-0 block "
            "means nothing was solved."),
    },

    "silent_failure_checklist": (
        "febio4's exit code alone is not enough and neither is the "
        "presence of output files. Check, in this order: "
        "(1) exit status 0; "
        "(2) the literal banner `N O R M A L   T E R M I N A T I O N` "
        "in stdout/.log — `E R R O R   T E R M I N A T I O N` can "
        "appear with a fully-populated .xplt and CSVs already on "
        "disk; "
        "(3) `Number of time steps completed` > 0; "
        "(4) the last `*Time` in the logfile CSV equals "
        "time_steps x step_size — with an auto time-stepper FEBio "
        "writes every converged step, so a run that dies at t=0.335 "
        "of an intended t=1 leaves four perfectly valid-looking "
        "states behind; "
        "(5) any WARNING box. `1 isolated vertex removed.` means "
        "the connectivity is degenerate; `No force acting on the "
        "system.` means the residual FEBio logs for the free DOFs "
        "fell below min_residual (default 1e-20) before the normal "
        "convergence tests passed, so the iteration was accepted as "
        "converged. It also appears in "
        "correctly loaded models, e.g. on steps after a load curve "
        "plateaus, or when prescribed displacements alone leave the "
        "free DOFs in equilibrium, so treat it as a prompt to check "
        "that some load or BC was meant to act on the free DOFs in "
        "that step, not as proof that nothing loads the model; "
        "`Model has N unreferenced load controllers.` means a "
        "load curve is not wired to anything. "
        "SIX verified cases defeat checks (1)-(4) entirely, so they "
        "have to be caught at deck-generation time or by inspecting "
        "the RESULT, never by watching the run. All six executed on "
        "4.12.0.86045466d. "
        "(a) <linear_solver type=\"test\"/> — a null solver that "
        "zeroes the solution and reports success (see "
        "linear_solvers.null_test_solver). "
        "(b) A degenerate element whose connectivity repeats a node "
        "id — only a WARNING. "
        "(c) A <MeshAdaptor> block written with type= on the SECTION "
        "instead of on a <mesh_adaptor> child — the adaptor is never "
        "registered and the mesh is never refined, and if that block "
        "contains any child which itself has children it also "
        "SWALLOWS every following top-level section, giving a model "
        "with 0 nodes and 0 materials that still terminates normally "
        "(see _general.adaptive_mesh_refinement). "
        "(d) An anisotropic material with <mat_axis> omitted — it "
        "silently uses the global axes a=(1,0,0), d=(0,1,0), which "
        "for fibers that should point elsewhere changes the stress by "
        "a factor that grows with fiber stiffness (see "
        "fiber_reinforced). "
        "(e) <density> omitted on a DYNAMIC or body-loaded model — it "
        "silently defaults to 1.0 (see hyperelasticity). "
        "(f) A transient run whose physical diffusion or wave length "
        "scale is shorter than one element — the field oscillates in "
        "sign and overshoots the prescribed values with no diagnostic "
        "(see heat and hyperelasticity). "
        "THE COMMON SHAPE: FEBio reports on the SOLVER, not on the "
        "MODEL. So the last line of defence is always to log a field "
        "that must be non-zero, or must be monotone, or must be "
        "bounded, and assert it — "
        "<Output><logfile><node_data data=\"x;y;z\" delim=\",\" "
        "file=\"pos.csv\"/></logfile></Output>. "
        "Also note that three checks fire only AFTER the deck reads "
        "`...SUCCESS!`, at model-initialisation time rather than "
        "parse time: an out-of-range material parameter, an "
        "unregistered plot variable, and a missing <Globals> "
        "constant. A wrapper that greps only for `Reading file "
        "...FAILED!` misses all three."),

    "verified_working_baseline": (
        "COMPLETE RUNNABLE DECK. Copy it whole, save it as a .feb, run "
        "`febio4 -i <file>.feb`. It is a single hex8 cube fixed on its "
        "bottom face and stretched 10% along z in ten load steps, and "
        "it is the safest thing in this catalog to mutate from. "
        "EXECUTED 2026-08-03 on FEBio 4.12.0.86045466d: ten of ten "
        "time steps completed, `N O R M A L   T E R M I N A T I O N`, "
        "the top face reaches z = 1.1 and the free side faces pull "
        "inward by Poisson's effect, and both output tables are "
        "written. Verify the physics from pos.csv rather than the "
        "banner: the top nodes must sit at z = 1.1 and the side nodes "
        "must have moved inward, not outward.\n"
        "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n"
        "<febio_spec version=\"4.0\">\n"
        "  <Module type=\"solid\"/>\n"
        "  <Control>\n"
        "    <analysis>STATIC</analysis>\n"
        "    <time_steps>10</time_steps>\n"
        "    <step_size>0.1</step_size>\n"
        "    <solver type=\"solid\">\n"
        "      <symmetric_stiffness>symmetric</symmetric_stiffness>\n"
        "    </solver>\n"
        "  </Control>\n"
        "  <Material>\n"
        "    <material id=\"1\" name=\"M1\" type=\"isotropic elastic\">\n"
        "      <density>1.0</density>\n"
        "      <E>1000.0</E>\n"
        "      <v>0.3</v>\n"
        "    </material>\n"
        "  </Material>\n"
        "  <Mesh>\n"
        "    <Nodes name=\"N\">\n"
        "      <node id=\"1\">0,0,0</node>\n"
        "      <node id=\"2\">1,0,0</node>\n"
        "      <node id=\"3\">1,1,0</node>\n"
        "      <node id=\"4\">0,1,0</node>\n"
        "      <node id=\"5\">0,0,1</node>\n"
        "      <node id=\"6\">1,0,1</node>\n"
        "      <node id=\"7\">1,1,1</node>\n"
        "      <node id=\"8\">0,1,1</node>\n"
        "    </Nodes>\n"
        "    <Elements type=\"hex8\" name=\"P1\">\n"
        "      <elem id=\"1\">1,2,3,4,5,6,7,8</elem>\n"
        "    </Elements>\n"
        "    <NodeSet name=\"bot\">1,2,3,4</NodeSet>\n"
        "    <NodeSet name=\"top\">5,6,7,8</NodeSet>\n"
        "  </Mesh>\n"
        "  <MeshDomains>\n"
        "    <SolidDomain name=\"P1\" mat=\"M1\"/>\n"
        "  </MeshDomains>\n"
        "  <Boundary>\n"
        "    <bc name=\"fix\" type=\"zero displacement\" node_set=\"bot\">\n"
        "      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n"
        "    </bc>\n"
        "    <bc name=\"pull\" type=\"prescribed displacement\" "
        "node_set=\"top\">\n"
        "      <dof>z</dof>\n"
        "      <value lc=\"1\">0.1</value>\n"
        "      <relative>0</relative>\n"
        "    </bc>\n"
        "  </Boundary>\n"
        "  <LoadData>\n"
        "    <load_controller id=\"1\" name=\"LC1\" type=\"loadcurve\">\n"
        "      <interpolate>LINEAR</interpolate>\n"
        "      <extend>CONSTANT</extend>\n"
        "      <points><pt>0,0</pt><pt>1,1</pt></points>\n"
        "    </load_controller>\n"
        "  </LoadData>\n"
        "  <Output>\n"
        "    <logfile>\n"
        "      <node_data data=\"x;y;z\" delim=\",\" file=\"pos.csv\"/>\n"
        "      <element_data data=\"sx;sy;sz;J\" delim=\",\" "
        "file=\"el.csv\"/>\n"
        "    </logfile>\n"
        "  </Output>\n"
        "</febio_spec>\n"
        "\nThe abbreviated skeleton below shows the same section order "
        "with the bulk elided; use it only as a map of the deck, never "
        "as something to run:\n"
        "<febio_spec version=\"4.0\">\n"
        "  <Module type=\"solid\"/>\n"
        "  <Control><analysis>STATIC</analysis>"
        "<time_steps>1</time_steps><step_size>1.0</step_size>\n"
        "    <solver type=\"solid\">"
        "<symmetric_stiffness>symmetric</symmetric_stiffness>"
        "</solver></Control>\n"
        "  <Material><material id=\"1\" name=\"M1\" "
        "type=\"isotropic elastic\">"
        "<density>1.0</density><E>1000.0</E><v>0.3</v>"
        "</material></Material>\n"
        "  <Mesh><Nodes name=\"N\">...</Nodes>"
        "<Elements type=\"hex8\" name=\"P1\">...</Elements>"
        "<NodeSet name=\"bot\">1,2,3,4</NodeSet></Mesh>\n"
        "  <MeshDomains><SolidDomain name=\"P1\" mat=\"M1\"/>"
        "</MeshDomains>\n"
        "  <Boundary>...</Boundary>\n"
        "  <LoadData><load_controller id=\"1\" name=\"LC1\" "
        "type=\"loadcurve\">...</load_controller></LoadData>\n"
        "</febio_spec>"),

    "isotropic_elastic_constitutive_law": (
        "FEBio's `isotropic elastic` is NOT small-strain linear "
        "elasticity — it is St. Venant-Kirchhoff: "
        "S = lambda*tr(E)*I + 2*mu*E with E the Green-Lagrange "
        "strain, reported as Cauchy stress sigma = F S F^T / J "
        "(FEIsotropicElastic::Stress evaluates the algebraically "
        "identical b*(lambda*trE - mu)/J + b^2*mu/J). Verified "
        "numerically to 1e-12 relative by a hex8 patch test with "
        "F = [[1.02,0.01,0],[0,0.99,0.005],[0,0,1.01]] — see "
        "scripts/tier2_fixtures/febio/hex8_patch_test_exact_stress. "
        "Consequence: results agree with linear elasticity only to "
        "O(strain); a verification study must either keep strains "
        "tiny or compare against the SVK closed form."),

    "shipped_template_execution_status": (
        "All 19 templates were regenerated with default parameters "
        "and re-run on FEBio 4.12.0.86045466d (2026-08-05; again "
        "2026-10-01 with the same step counts). "
        "NINETEEN of 19 exit 0, print "
        "`N O R M A L   T E R M I N A T I O N` and complete at "
        "least one time step; none is killed by a signal. Completed "
        "steps: active_contraction_3d_fiber 20, biphasic_3d_confined "
        "10, biphasic_fsi_3d_block 10, damage_3d_cycle 60, "
        "elasticity_mms_3d_cube_hex8 1, fiber_reinforced_3d_hgo 10, "
        "fluid_3d_channel 10, fluid_fsi_3d_block 10, "
        "growth_remodeling_3d_isotropic 10, heat_3d_bar 1, "
        "hyperelasticity_3d_cube 10, linear_elasticity_3d_cube 1, "
        "multiphasic_3d_diffusion 10, plasticity_3d_uniaxial 20, "
        "polar_fluid_3d_channel 10, rigid_body_3d_pushdown 10, "
        "rigid_contact_3d_indentation 10, solute_3d_diffusion 10, "
        "viscoelasticity_3d_stress_relax 50. "
        "An earlier version of this entry said SIX of 17 ran and "
        "told the reader to treat ELEVEN as unverified stubs, "
        "naming a `heat` module segfault and a "
        "`tag \"micro_viscosity\" ... unrecognized tag`. Those "
        "defects were repaired; the entry was not. Do not "
        "re-derive a template's status from prose — regenerate and "
        "run it. "
        "RUNNING IS NOT THE SAME AS EXERCISING THE PHYSICS, and "
        "four of the 18 checked (solute_3d_diffusion was not part "
        "of that check) do not, proven by PARAMETER INVARIANCE "
        "rather than inspection (2026-08-05): "
        "(a) fiber_reinforced_3d_hgo — its `mat_axis` puts both "
        "fiber families transverse to the load, so the HGO fiber "
        "term is gated off. Sweeping <k1> over 0, 1 (shipped) and "
        "10000 leaves the final element row identical to all 12 "
        "printed digits (sz=98.2977075142, J=1.03954063461, "
        "sed=22.2559943744); <gamma>0</gamma> is identical too. "
        "Rotate <a> onto the load direction and k1 bites: sz goes "
        "98.2977 -> 99.2888 -> 107.749 -> 169.068 at k1 = 0, 1, 10, "
        "100. "
        "(b) polar_fluid_3d_channel — scaling <tau>, <beta> and "
        "<gamma> from 0.001 to 10.0 (10^4x) leaves the logged "
        "element data BYTE-IDENTICAL (same md5). The momentum "
        "balance is solved (fvx = -5.014 against the slug closed "
        "form -5.000) but the polar constitutive law contributes "
        "nothing. "
        "(c) biphasic_fsi_3d_block and fluid_fsi_3d_block — the "
        "dilatation DOF is constrained away so the logged `ef` "
        "column is identically 0.0 at every node and step; the "
        "z-motion the deck demonstrates is the prescribed "
        "compression, not consolidation. "
        "Use these four as syntax examples, not as physics "
        "demonstrations, and check the invariance yourself before "
        "quoting any number out of them. "
        "viscoelasticity_3d_stress_relax is not one of them: under "
        "the shipped DYNAMIC/STEP configuration its logged sz keeps "
        "its compressive sign on all 50 steps and relaxes (-146.7 to "
        "-95.9), but it carries a step-to-step oscillation of up to "
        "about 1.9 around the STATIC history (-146.6 to -94.06); see "
        "the [Numerical] RINGING pitfall in the viscoelasticity "
        "entry, and use STATIC for relaxation numbers."),
}
