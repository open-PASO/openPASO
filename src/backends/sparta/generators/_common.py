"""Cross-cutting SPARTA knowledge shared by every physics module.

Everything here was produced by feeding a deck to the installed serial build
and reading its output. The ``Signal:`` clause of each pitfall is a literal
substring that SPARTA itself prints (for printf-formatted messages the clause
quotes the format-invariant part), so a post-execution critic can retrieve the
pitfall by matching the failing run's stdout/log.sparta against it.

NOTE ON SCOPE: these entries are attached to every physics because they are
properties of the SPARTA *deck*, not of any one flow regime. Physics-specific
traps live in the per-physics modules next to this file.
"""

# ── the ordered skeleton every SPARTA deck follows ────────────────────────
DECK_SKELETON = [
    "seed <int>                       # mandatory, there is no default RNG seed",
    "dimension 2|3                    # before create_box",
    "boundary <x> <y> <z>             # before create_box; 1 letter = both faces",
    "create_box xlo xhi ylo yhi zlo zhi",
    "create_grid Nx Ny Nz [level ...] # 2d requires Nz = 1",
    "species <file> <ID> ...          # may precede create_box",
    "mixture <mixID> <species...> [vstream vx vy vz] [temp T] [frac f] [group SELF]",
    "global nrho <n> fnum <F>         # MUST precede create_particles",
    "collide vss <mixID> <file.vss>   # omit it and the gas is collisionless",
    "read_surf / surf_collide / surf_modify   # embedded geometry, if any",
    "create_particles <mixID> n 0     # n 0 == honour nrho/fnum",
    "compute / fix / stats / stats_style / dump",
    "timestep <dt>",
    "run <N>",
]

# ── ordering constraints the parser enforces (all reproduced) ─────────────
HARD_ORDERING_ERRORS = {
    "create_grid before create_box":
        "Cannot create grid before simulation box is defined (../create_grid.cpp:44)",
    "create_particles before create_grid":
        "Cannot create particles before grid is defined (../create_particles.cpp:53)",
    "run before a grid exists":
        "Run command before grid is defined (../run.cpp:43) — the message names "
        "the GRID even when the real omission is create_box",
    "boundary after create_box":
        "Boundary command after simulation box is defined (../domain.cpp:148)",
    "dimension after create_box":
        "Dimension command after simulation box is defined (../input.cpp:1442)",
    "units after create_box":
        "Units command after simulation box is defined (../input.cpp:1692)",
    "mixture naming an unloaded species":
        "Mixture species is not defined (../mixture.cpp:326)",
    "collide naming an unknown mixture":
        "Collision mixture does not exist (../collide.cpp:155) — raised at the "
        "START OF THE FIRST RUN, not when the collide line is parsed",
    "collide mixture missing a loaded species":
        "Collision mixture does not contain all species (../collide.cpp:159)",
    "no seed command":
        "Seed command has not been used (../random_mars.cpp:91)",
    "2d box whose z bounds do not straddle 0":
        "Create_box z box bounds must straddle 0.0 for 2d simulations "
        "(../create_box.cpp:51)",
    "2d create_grid with nz != 1":
        "Create_grid nz value must be 1 for a 2d simulation (../create_grid.cpp:205)",
}

# ── things SPARTA accepts without complaint (no error to rely on) ─────────
SILENTLY_ACCEPTED = {
    "no global nrho/fnum": "falls back to nrho = 1.0, fnum = 1.0, thermal "
                           "temperature 273.15 K; create_particles then makes "
                           "almost nothing. rc = 0, no warning.",
    "no species at all": "box + grid + run completes with Np = 0 and rc = 0. "
                         "An empty simulation is not an error.",
    "mixture with no temp keyword": "silently inherits the global thermal "
                                    "temperature (default 273.15 K).",
    "species/mixture before create_box": "legal — only box/grid/particle "
                                         "ordering is constrained.",
    "a timestep far above the mean collision time": "runs cleanly and inflates "
                                                    "transport; see the "
                                                    "rarefied_flow pitfalls.",
    "no timestep command at all": "the default timestep is 1.0 SECOND. On any "
                                  "micron- or millimetre-scale box that is not "
                                  "an error and not slow — it is a hang, "
                                  "because each particle crosses millions of "
                                  "cells per step. Always set 'timestep'.",
}

# ── how to get numbers out of a run ───────────────────────────────────────
READING_OUTPUT = {
    "stats": "'stats N' prints every N steps plus the first and last step of a "
             "run; 'stats_style' picks the columns; log.sparta in the cwd holds "
             "the same table as stdout.",
    "stats_style keywords": "step elapsed elaplong cpu tpcpu spcpu wall dt time "
                            "np ntouch ncomm nbound nexit nscoll nscheck ncoll "
                            "nattempt nreact nsreact (+ *ave variants) ngrid "
                            "nsplit maxlevel vol lx ly lz xlo xhi ylo yhi zlo "
                            "zhi, plus c_ID / f_ID / v_name. Anything else: "
                            "'Invalid keyword in stats_style command "
                            "(../stats.cpp:737)'.",
    "per-grid tally idiom": "compute <C> grid <grp> <mix> <values> -> fix <F> "
                            "ave/grid <grp> Nevery Nrepeat Nfreq c_C[*] -> "
                            "compute <R> reduce ave f_F[i] -> stats_style c_R. "
                            "A per-grid compute cannot go straight into "
                            "stats_style.",
    "per-surf tally idiom": "compute <C> surf <grp> <mix> <values> -> fix <F> "
                            "ave/surf <grp> Nevery Nrepeat Nfreq c_C[1] -> "
                            "compute <R> reduce sum f_F -> stats_style c_R.",
    "boundary tally idiom": "compute <C> boundary <mix> <values> (mixture ID "
                            "only, NO group ID) -> fix <F> ave/time Nevery "
                            "Nrepeat Nfreq c_C[*] mode vector ('mode vector' "
                            "is mandatory) -> stats_style f_F[face] when the "
                            "compute has one column, f_F[face][value] when it "
                            "has more. Row order is xlo=1 xhi=2 ylo=3 yhi=4 in "
                            "2d, plus zlo=5 zhi=6 in 3d — there are only FOUR "
                            "rows in a 2d run.",
    "vector vs array rule": "a fix ave/* fed ONE input value produces a VECTOR "
                            "read as f_ID (no bracket); fed two or more it "
                            "produces an ARRAY read as f_ID[1], f_ID[2], ... "
                            "compute surf and compute grid always produce "
                            "ARRAYS, so they are consumed as c_ID[i].",
    "dump format": "dump particle / grid / surf write LAMMPS-style text "
                   "snapshots: ITEM: TIMESTEP, ITEM: NUMBER OF "
                   "ATOMS|CELLS|SURFS, ITEM: BOX BOUNDS <flags>, then ITEM: "
                   "ATOMS|CELLS|SURFS with the requested column names.",
}


def output_idioms(*keys: str) -> dict:
    """The subset of READING_OUTPUT a given physics actually needs.

    'stats', 'stats_style keywords' and 'vector vs array rule' are always
    included — every deck hits them. The tally idioms are opt-in so a
    free-molecular box does not carry the surface-tally recipe and a
    conjugate-heat-transfer case does not carry the boundary one.
    """
    base = ("stats", "stats_style keywords", "vector vs array rule")
    return {k: READING_OUTPUT[k] for k in base + keys if k in READING_OUTPUT}

# ── what this build can and cannot do (no host paths, no run refs) ───
BUILD_FACTS = {
    "invocation": "spa_serial -in in.<case>, run from the directory that holds "
                  "the deck and its data files; log.sparta is written to cwd.",
    "parallelism": "serial build only on this install — nothing about MPI "
                   "decomposition or 'partition' is verified here.",
    "accelerators": "the KOKKOS /kk styles are NOT compiled in. 'package "
                    "kokkos' aborts with 'Package kokkos command without "
                    "KOKKOS package enabled (../input.cpp:1507)', a /kk "
                    "compute or fix gives 'Unrecognized compute style "
                    "(../modify.cpp:467)' / 'Unrecognized fix style "
                    "(../modify.cpp:370)', and 'suffix' is not a command in "
                    "this parser at all. Do not write /kk into a deck.",
    "compiled_collide_styles": ["vss"],
    "compiled_react_styles": ["qk", "tce", "tce/qk"],
    "compiled_surf_collide_styles": ["adiabatic", "cll", "diffuse", "impulsive",
                                     "piston", "specular", "td", "transparent",
                                     "vanish"],
    "compiled_surf_react_styles": ["adsorb", "global", "prob"],
    "n_compute_styles": 27,
    "n_fix_styles": 25,
    "n_input_commands": 66,
    "self_check": "'spa_serial -h' prints the exact style list this build "
                  "contains — use it instead of trusting a doc page.",
}

# ── pitfalls that apply to EVERY SPARTA deck ──────────────────────────────
UNIVERSAL_PITFALLS = [
    "[Setup] 'global nrho <n> fnum <F>' must be issued BEFORE "
    "'create_particles'. Placed after it, or left out entirely, the defaults "
    "nrho = 1.0 / fnum = 1.0 apply, create_particles makes (almost) no "
    "particles, and the run completes normally with an empty domain — rc = 0, "
    "no warning, a full stats table of zeros. So does a nrho that is not the "
    "gas's number density in molecules per m^3: 'create_particles <mix> n 0' "
    "makes nrho x V / fnum particles (V the flow volume, in 2-D the area), and "
    "a nrho written as a mass density or as a count per cell puts that below "
    "one. The same empty run follows when "
    "no grid cell lies on the flow side of a surface: SPARTA puts the gas on "
    "the side a line's normal N = (0,0,1) x (p2 - p1) points to (walking from "
    "p1 to p2, the gas is on your left), and with every line listed the other "
    "way round every cell is inside a body and create_particles makes 0 "
    "particles. Swap p1 and p2 of those lines. "
    "Signal: the setup line 'Created 0 particles' (printf 'Created %ld "
    "particles') followed by an Np column that is 0 on every stats line; for "
    "the surface cause, the read_surf line before it reads '0 <b> <c> = cells "
    "outside/inside/overlapping surfs'.",

    "[Numerical] 'create_particles <mix> n <N>' with a NONZERO N creates "
    "exactly N simulation particles and silently overrides 'global nrho' — the "
    "realised density becomes N*fnum/V, not the nrho you asked for. Only 'n 0' "
    "honours nrho and fnum. Most upstream example decks use an explicit n, so "
    "copying one silently changes your density. "
    "Signal: the 'Created <N> particles' line reproduces your n argument "
    "exactly instead of the value you get from nrho*V/fnum; recompute the "
    "realised density as Np*fnum/V and compare with the nrho you set. Adding "
    "a 'region' argument breaks even that check: the count then comes out at "
    "roughly N times the region's volume fraction, and SPARTA deliberately "
    "suppresses its 'Created unexpected # of particles:' warning whenever a "
    "region is present.",

    "[Numerical] A 'nrho' keyword on a MIXTURE overrides 'global nrho' for "
    "every particle drawn from that mixture — create_particles and fix "
    "emit/face both use the mixture value. Two nrho settings in one deck do "
    "not conflict-check; the mixture one simply wins, silently, rc = 0. "
    "Signal: 'Created <N> particles' is off from nrho*V/fnum by exactly the "
    "ratio of the two nrho values; grep the deck for a second 'nrho' on any "
    "mixture line before trusting a density.",

    "[Units] 'units cgs' is accepted silently and changes only the value of "
    "Boltzmann's constant SPARTA multiplies your numbers by — it does NOT "
    "convert the species, VSS, reaction or surface files, which ship in SI. A "
    "deck that merely gains a leading 'units cgs' line keeps the same particle "
    "count and the same stats table shape while every particle speed, and "
    "hence the collision rate, is scaled by a large constant factor, always "
    "upward, independently of density, grid and species. Nothing warns and "
    "rc = 0. A 'units' line AFTER create_box is a hard error, so only a "
    "leading one slips through. "
    "Signal: 'Cell-touches/particle/step' in the end-of-run block jumps from "
    "about 1 to well above 100, 'Collisions/particle/step' exceeds 1 (which no "
    "valid DSMC timestep can produce), and the run is orders of magnitude "
    "slower. Note that 'compute temp' is NOT a tell — Boltzmann cancels "
    "between the velocity draw and the temperature normalisation, so the "
    "temperature column reads correctly in both unit systems. Leave 'units' "
    "out (si is the default) unless every input file is genuinely cgs.",

    "[Physics] Omitting the 'collide' command is usually accepted and makes "
    "the gas collisionless (free-molecular), so any transport quantity you "
    "then measure describes ballistic particles rather than a continuum gas. "
    "It is not universally silent: 'collide_modify', 'fix vibmode' and 'react "
    "tce' each abort when no collide style is defined. "
    "Signal: the Ncoll column is identically 0 on every stats line while Np is "
    "large — but that signal is AMBIGUOUS, because a deck WITH a collide "
    "command also gives Ncoll = 0 at low enough density or small enough "
    "timestep. Two unambiguous checks: log.sparta echoes the input deck (the "
    "screen output does not), so grep it for a 'collide' line; and in the "
    "end-of-run timing table both the 'Coll' and the 'Sort' rows are exactly "
    "0 only when no collide style exists at all. The hard-error forms are "
    "'ERROR: Cannot use collide_modify with no collisions defined "
    "(../input.cpp:1425)', 'ERROR: Cannot use fix vibmode without collide "
    "style defined (../fix_vibmode.cpp:50)' and 'ERROR: React tce can only be "
    "used with collide vss (../react_tce.cpp:40)'.",

    "[Numerical] There is no default timestep worth having: SPARTA starts "
    "from dt = 1.0 SECOND, so a deck that omits the 'timestep' command on a "
    "micron- or millimetre-scale domain does not fail, it HANGS — every "
    "particle has to be traced across millions of cells in a single step. "
    "Signal: the run never reaches its first stats line and never returns. "
    "Measured on the installed build with a millimetre box: the run has to be "
    "killed, and BOTH log.sparta and the captured screen output are then ZERO "
    "BYTES — SPARTA's "
    "stdio is block-buffered, so when you redirect or pipe the run (which is "
    "how any driver runs it) you do not even get the setup banner to look at. "
    "An empty log.sparta from a process that is still alive means no "
    "'timestep' line, not a broken machine.",

    "[Numerical] DSMC is a Monte-Carlo method: changing only the 'seed' value "
    "changes every tallied number. Two runs that differ by a few percent in a "
    "collision or flux column have not necessarily been changed by your edit. "
    "Signal: rerun the unmodified deck with a different seed; if the "
    "difference you are attributing to a physics change is inside that spread, "
    "it is noise. Ncollave (the running mean) settles far faster than Ncoll, "
    "which is a per-step count.",

    "[Numerical] In a 2d run the grid-cell volume is dx*dy per ONE METRE of "
    "depth and the z extent handed to create_box is ignored for volume "
    "purposes. Sizing fnum for a thin slab therefore overshoots the particle "
    "count by 1/(zhi-zlo), which can be a factor of 10^4. "
    "Signal: two decks whose only difference is the z extent report the same "
    "'Created <N> particles' line; a 2d deck sized for a thin slab either "
    "hangs before the first stats block or leaves an empty log.sparta.",

    "[Output] Ncoll, Nattempt, Nscoll, Nscheck, Nbound, Nexit and Nreact are "
    "the count on THAT ONE TIMESTEP — not a cumulative total and not a sum "
    "over the stats interval. The *ave variants (Ncollave, ...) are running "
    "means since the start of the run. "
    "Signal: read the same column with 'stats 1' and with 'stats 100'; the "
    "per-step counters give the same order of magnitude in both, so a value "
    "you are treating as a total is wrong by a factor of the step count.",

    "[Syntax] The compute and fix STYLE names in the SPARTA documentation are "
    "page filenames, not deck commands. In a deck you write 'compute <ID> grid "
    "...', 'fix <ID> ave/surf ...', 'dump <ID> image ...', 'surf_react <ID> "
    "adsorb ...' — never 'compute_grid', 'fix_ave_surf', 'dump_image' or "
    "'surf_react_adsorb'. This build accepts 66 input-script commands. "
    "Signal: 'ERROR: Unknown command: ' followed by your line and "
    "'(../input.cpp:244)'.",

    "[Syntax] The dump command is 'dump <ID> <style> <group-or-mixture> "
    "<Nevery> <file> <attributes...>' — the STYLE is the SECOND token, after "
    "your own ID. Writing the group before the style, which is the order "
    "compute and fix use, makes SPARTA read the group word as the style and "
    "reject it by a name you did not type. Two more rules that bite: a "
    "filename containing '*' produces one file per snapshot and a filename "
    "without it puts every snapshot in ONE file; and a per-grid or per-surf "
    "compute named in a dump obeys the same bracket rule as everywhere else, "
    "so 'compute grid' must be written c_ID[i]. Of the six registered dump "
    "styles, 'movie' parses on this build but cannot run — it is compiled "
    "without the encoder — while 'image' works and writes PPM. "
    "Signal: 'ERROR: Unrecognized dump style (../output.cpp:538)' quoting no "
    "style name, which means the token in that slot was not a style; 'ERROR: "
    "Dump grid compute does not calculate per-grid vector "
    "(../dump_grid.cpp:515)' for a missing bracket; 'ERROR on proc 0: Support "
    "for writing movies not included (../dump_movie.cpp:52)' at the first "
    "dump step. (Verified 2026-08-07)",

    "[Integration] Two run-control fixes fail in ways a driver does not see. "
    "'fix <ID> halt <Nevery> <attribute> <op> <value>' defaults to 'error "
    "soft', which ENDS THE RUN EARLY and exits with status 0 — a driver that "
    "checks only the exit code records a truncated run as a success, and the "
    "only trace is one line in the log. 'error hard' turns the same event into "
    "a nonzero exit. Its attribute may only be 'tlimit' or an EQUAL-STYLE "
    "VARIABLE, so a bare stats keyword such as np is rejected and has to be "
    "wrapped ('variable n equal np' then 'fix h halt 100 v_n < 1000'). 'fix "
    "<ID> print <Nevery> \"...\"' substitutes ${var} at EXECUTION time, not "
    "when the line is parsed, so a variable that is never defined does not "
    "stop the deck at setup — the run starts, prints its first stats row, and "
    "dies at the first print step. "
    "Signal: 'Fix halt condition for fix-id <ID> met on step <N> with value "
    "<V> (../fix_halt.cpp:222)' with rc = 0 for the soft case and the same "
    "text at line 220 as an ERROR for the hard one; 'ERROR: Invalid fix halt "
    "attribute <name> (../fix_halt.cpp:62)'; 'ERROR on proc 0: Substitution "
    "for illegal variable (../input.cpp:531)' arriving AFTER the step-0 stats "
    "row. Compare the last Step value in the table against the argument of "
    "'run' before trusting any completed job. (Verified 2026-08-07)",
]
