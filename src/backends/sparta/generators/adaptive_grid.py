"""Static and dynamic grid adaptation."""

from ._common import output_idioms


def _adapt_circle_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Static refinement around a surface (adapt_grid, one-shot) followed by
    dynamic refinement on particle count during the run (fix adapt).
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", 20)
    vstream = params.get("vstream", 100.0)
    nrho = params.get("nrho", 1.0)
    fnum = params.get("fnum", 0.001)
    maxlevel = params.get("maxlevel", 3)
    dt = params.get("dt", 1.0e-4)
    nevery = params.get("nevery", 100)
    nsteps = params.get("nsteps", 500)
    return f"""\
# Grid adaptation around a surface - SPARTA DSMC
seed             12345
dimension        2
global           gridcut 0.0 nrho {nrho} fnum {fnum}
boundary         o r p
create_box       0 10 0 10 -0.5 0.5
create_grid      {nx} {ny} 1
species          air.species N O
mixture          air N O vstream {vstream} 0.0 0.0
read_surf        data.circle
surf_collide     wall diffuse 300.0 1.0
surf_modify      all collide wall
# WARNING, measured on this build: at the DEFAULT nrho 1.0 / fnum 0.001 this
# collide line is INERT — a run reports 'SurfColl occurs = 73827' but
# 'Collide occurs = 0 (0K)'. Grid adaptation driven by a collision-derived
# quantity (cell Knudsen number, mean free path) is therefore meaningless at
# these numbers: with no collisions the lambda field is the 1e+20 sentinel.
# Raise nrho and fnum before adapting on anything but particle count.
collide          vss air air.vss
fix              in emit/face air xlo
# ONE-SHOT static refinement near the surface. 'adapt_grid' acts immediately,
# so any fix it references must already have produced output -- with 'surf'
# style there is no such dependency.
adapt_grid       all refine surf all 0.15 iterate 2
# DYNAMIC refinement during the run. The fix Nfreq and the adapt Nevery must
# match, and maxlevel caps the refinement depth.
compute          npc grid all all n
fix              fnp ave/grid all 1 {nevery} {nevery} c_npc[1]
fix              ad adapt {nevery} all refine coarsen value f_fnp 20.0 5.0 &
                 maxlevel {maxlevel}
timestep         {dt}
stats            {nevery}
stats_style      step np ncoll ngrid maxlevel
run              {nsteps}
"""


KNOWLEDGE = {
    "adaptive_grid": {
        "description": "Static (adapt_grid) and dynamic (fix adapt) grid "
                       "adaptation to resolve shocks, boundary layers and "
                       "surface geometry",
        "spatial_dims": [2, 3],
        "key_commands": {
            "adapt_grid": "adapt_grid <grp> refine|coarsen [coarsen|refine] "
                          "particle <rcount> <ccount> | surf <surfgrp> <ssize> "
                          "| value <c_ID[i]|f_ID[i]> <rval> <cval> | random "
                          "<rfrac> <cfrac> [iterate <n>] [maxlevel <n>] "
                          "[minlevel <n>] [thresh less|more less|more] "
                          "[combine sum|min|max] [cells <nx> <ny> <nz>]",
            "fix adapt": "fix <ID> adapt <Nevery> <grp> ... — same style "
                         "arguments as adapt_grid, applied during the run",
            "create_grid ... levels": "create_grid Nx Ny Nz levels <N> then ONE "
                                      "of 'region <lev> <regID> <cx> <cy> <cz>' "
                                      "or 'subset <lev> <xrange> <yrange> "
                                      "<zrange> <cx> <cy> <cz>' for EVERY level "
                                      "2..N — static hierarchical refinement at "
                                      "setup time. There is NO 'level' keyword: "
                                      "an earlier catalog gave 'create_grid Nx "
                                      "Ny Nz level <n> <region> <nx> <ny> <nz>' "
                                      "and that form does not parse. The level "
                                      "index is 1-based with level 1 being the "
                                      "root grid, so the first refinable level "
                                      "is 2, and a level range may be written "
                                      "'2*3'.",
            "balance_grid": "balance_grid rcb cell|part|time — rebalance cells "
                            "across MPI ranks after adaptation (no effect in a "
                            "serial build)",
            "stats_style": "add 'ngrid' and 'maxlevel' so the adaptation is "
                           "visible in the stats table",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "output_idioms": output_idioms("per-grid tally idiom"),
        "pitfalls": [
            "[Setup] 'adapt_grid ... value f_<fixID>' placed BEFORE any 'run' "
            "reads every cell as literal 0, because the fix has produced no "
            "output yet. With the DEFAULT 'thresh more less' that means no "
            "cell qualifies for refinement and the grid is left alone — but "
            "the zeros are not harmless: with a 'coarsen' action, or with "
            "'thresh less', the same command instead coarsens or refines the "
            "ENTIRE grid. 'adapt_grid ... particle ...' and 'adapt_grid ... "
            "value c_<computeID>[N]' both work correctly before a run; the "
            "trap is specific to a fix that has not yet fired. "
            "Signal: 'Adapting grid ...' followed by '  no grid adaptation "
            "performed' instead of the '<N> cells refined, <M> cells "
            "coarsened' line. That message is AMBIGUOUS — it is also printed "
            "when the fix has data but no cell crosses the threshold — so "
            "confirm by moving the adapt_grid line after a short run.",

            "[Syntax] The bracket on an adapt_grid or fix adapt 'value' source "
            "must match its shape: a fix ave/grid with ONE input value is a "
            "per-grid VECTOR and must be written 'value f_<ID>' with no "
            "bracket, while a compute grid is always an ARRAY and must be "
            "written 'value c_<ID>[N]'. This is checked before and after a "
            "run alike, so it is easy to mistake for the "
            "fix-not-yet-computed trap above. "
            "Signal: 'ERROR: Adapt fix does not calculate a per-grid array "
            "(../adapt_grid.cpp:484)'.",

            "[Numerical] Refinement is unbounded unless capped: every level "
            "multiplies the cell count and divides the particles per cell, and "
            "below one particle per cell every per-cell diagnostic (including "
            "the one driving the adaptation) becomes noise, which can drive "
            "further refinement. Use 'maxlevel'. "
            "Signal: the Ngrid and Maxlevel columns keep climbing while Np is "
            "flat; reduce a per-cell particle count with a compute reduce in "
            "min mode (doc/compute_reduce.txt) and require it to stay >= 1.",

            "[Setup] 'fix adapt' can only act on steps where the fix it reads "
            "has produced output, so its Nevery must be a multiple of that "
            "fix's Nfreq — and the stats interval that prints any dependent "
            "quantity is subject to the same rule. There are THREE different "
            "messages depending on who reads the fix, and the one for the "
            "adapt itself is easy to miss — an earlier wording listed only the "
            "other two, so a guard built from it would not fire on the mistake "
            "the entry is about. They do not all abort in the same place, and "
            "an earlier wording said all three were 'a mid-run abort right "
            "after the step-0 stats line': only the compute reduce one is. "
            "The adapt's own complaint and the stats one fire at SETUP with no "
            "stats row printed at all. "
            "Signal: 'ERROR: Fix for adapt not computed at compatible time "
            "(../adapt_grid.cpp:502)' when the adapt Nevery is not a multiple "
            "of the fix Nfreq, before any stats row; 'ERROR: Stats and fix not "
            "computed at compatible times (../stats.cpp:203)' (line 209 in 27Aug2026) when stats_style "
            "prints f_<ID> for a fix producing a GLOBAL SCALAR such as "
            "fix ave/time, also before any stats row — printing a PER-GRID fix "
            "that way gives 'ERROR: Stats fix does not compute scalar "
            "(../stats.cpp:705)' (line 814 in 27Aug2026) instead, a shape complaint raised while the "
            "stats_style line is parsed and before any frequency is compared, "
            "so grepping for :203 after the per-grid mistake finds nothing; "
            "'ERROR: Fix used in compute reduce not computed at compatible time "
            "(../compute_reduce.cpp:805)' (line 809 in 27Aug2026) when a compute reduce reads it, and "
            "that one does abort mid-run, right after the step-0 stats line.",

            "[Physics] Adapting on particle count refines where particles "
            "already are, which is not where the gradients are: in a "
            "hypersonic case that means the post-shock region, not the shock "
            "itself. Adapting on the cell Knudsen number from 'compute "
            "lambda/grid ... knall' targets under-resolution directly. "
            "Signal: after adaptation the cell Knudsen minimum, reduced with a "
            "compute reduce in min mode over c_lam[2], has not improved — "
            "that alone is the "
            "observable, and the refinement went to the wrong cells. Do NOT "
            "also require Ngrid to have grown several-fold; an earlier wording "
            "bundled the two and only the first holds. Refining on particle "
            "count splits a cell's particles among its children, which drops "
            "them below the threshold immediately, so the criterion is "
            "SELF-LIMITING: measured on a Mach-5 cylinder case the cell count "
            "grew by about a quarter while the Knudsen-driven run on the same "
            "deck grew more than fivefold and lifted the Knudsen minimum by "
            "about a factor of three. Looking for several-fold growth AND no "
            "improvement, you would not find the first half and could wrongly "
            "conclude this does not apply to your case. Compare the Knudsen "
            "minimum before and after, nothing else.",

            "[Setup] 'balance_grid' distributes cells over MPI ranks; on a "
            "serial build it runs and moves nothing, so it is not a remedy for "
            "an unbalanced particle load within one rank. "
            "Signal: the line 'Balance grid migrated 0 cells' — it is printed "
            "for every balance_grid call, and a 0 there in a run you expected "
            "to rebalance means either a serial build or an already-balanced "
            "decomposition.",

            "[Syntax] Static multi-level refinement is a TWO-PART command and "
            "the parser will not tell you which part is missing. 'create_grid "
            "Nx Ny Nz levels <N>' only declares how many levels exist; every "
            "level from 2 to N must then be given its own 'region <lev> "
            "<regID> <cx> <cy> <cz>' or 'subset <lev> <xrange> <yrange> "
            "<zrange> <cx> <cy> <cz>' clause, and a level left unset is caught "
            "only at the END of parsing with a message that names no level. "
            "The keyword is 'levels' with an s — the singular 'level' does not "
            "exist and produces the generic Illegal message pointing at the "
            "keyword position. The 2d rule from create_grid still applies "
            "inside a level: cz must be 1. "
            "Signal: 'ERROR: Create_grid level was not set "
            "(../create_grid.cpp:208)' for a declared but unconfigured level, "
            "'ERROR: Illegal create_grid command (../create_grid.cpp:188)' for "
            "an unrecognised keyword such as 'level', 'ERROR: Create_grid cz "
            "value must be 1 for a 2d simulation (../create_grid.cpp:210)', "
            "and 'ERROR: Create_grid region ID does not exist' for a region "
            "that was never defined. When it works, the 'Created <N> child "
            "grid cells' line and the Maxlevel stats column both go up. "
            "(Verified 2026-08-07)",

            "[Setup] 'fix <ID> grid/check <Nevery> error|warn|silent [outside "
            "yes|no]' is an INTERNAL-CONSISTENCY assertion on SPARTA's own "
            "particle-to-cell bookkeeping, not a check that your setup is "
            "sound — and reading it as the latter is the trap, because it "
            "returns a clean bill of health on decks that are physically "
            "worthless. It stays at exactly zero on a grid far coarser than "
            "the mean free path, on a timestep so large the per-step collision "
            "count is comparable to the particle count, and on colliding "
            "moving surfaces (there the cutting routine fails first, with its "
            "own unrelated message). Upstream uses it only in decks that MOVE "
            "or ABLATE geometry, which is what it is for. Two details matter: "
            "the fix exposes a CUMULATIVE count of flagged particles as f_ID "
            "in all three modes, so 'silent' plus f_ID in stats_style is the "
            "usable form; and the check that a particle has not leaked INSIDE "
            "a surface is opt-in — without 'outside yes' only cell membership "
            "is tested. "
            "Signal: f_<ID> identically 0 for the whole run, and in warn mode "
            "no '<N> particles in wrong cells on timestep <step>' line. Do NOT "
            "conclude from that zero that the grid, the timestep or the "
            "geometry is adequate; those need lambda/grid, a timestep-halving "
            "comparison and the watertight check respectively. "
            "(Verified 2026-08-07)",

            "[Output] 'fix <ID> balance <Nevery> <thresh> <style>' is not a "
            "no-op on a serial build but it is not a diagnostic either: it "
            "runs, migrates nothing, and reports an imbalance factor of "
            "exactly 1 through f_ID on every step, because there is one "
            "partition to be imbalanced. A 1 there therefore says nothing "
            "about whether the particle load is uneven within the rank — that "
            "is what the per-cell particle count shows. The threshold argument "
            "is the imbalance ratio above which a rebalance is attempted, so "
            "on one rank it is never reached. "
            "Signal: the f_<ID> column pinned at 1 for the entire run on a "
            "serial build; the 'Running on 1 MPI task(s)' banner at the top of "
            "the log is the confirmation that this is what you are looking at. "
            "(Verified 2026-08-07)",

            "[Setup] 'fix <ID> move/surf <surf-group> <Nevery> <Nlarge> trans|"
            "rotate ...' re-cuts the grid every Nevery steps as the geometry "
            "moves, and the failure it produces when the motion is too "
            "aggressive comes from the CUTTING code, not from any check that "
            "names the fix. Two surface groups driven into each other, or a "
            "displacement large compared with a cell in one move interval, "
            "abort inside cut2d/cut3d with a message about surface topology "
            "that mentions neither move/surf nor the step size. 'fix "
            "grid/check', which upstream pairs with every moving-surface deck, "
            "does NOT catch it: it stays at zero right up to the abort. The "
            "controls are Nevery (how often the surface is re-cut, so how far "
            "it jumps each time) and Nlarge (the step over which the total "
            "displacement is spread); shrink the per-move displacement rather "
            "than the total. "
            "Signal: a 'Cut2d failed on proc 0 in cell ID: <n>' block naming "
            "the cell's corners and its surface list, followed by 'ERROR on "
            "proc 0: WB: Point appears last in more than one CLINE "
            "(../cut2d.cpp:289)' (line 291 in 27Aug2026) — or its cut3d equivalent. Read that as a "
            "geometry-motion problem, not as a bad surface file: the same file "
            "reads and cuts cleanly when it is stationary. "
            "(Verified 2026-08-07)",
        ],
    },
}

GENERATORS = {
    "adaptive_grid_circle_2d": _adapt_circle_2d,
    "adaptive_grid_2d": _adapt_circle_2d,
}
