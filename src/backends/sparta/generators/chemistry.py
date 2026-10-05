"""Gas-phase chemistry (TCE / QK) during DSMC collisions."""

from ._common import output_idioms


def _hot_air_box_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Hot five-species air in a 3d box with TCE gas-phase chemistry. Starts as
    80/20 N2/O2 and dissociates; the species counts make the chemistry visible
    in the stats table.
    """
    n = params.get("n", 10)
    lx = params.get("lx", 1.0e-4)
    temp = params.get("temp", 20000.0)
    nrho = params.get("nrho", 7.07043e22)
    fnum = params.get("fnum", 7.07043e7)
    dt = params.get("dt", 1.0e-9)
    nsteps = params.get("nsteps", 500)
    return f"""\
# Hot air in a 3d box with TCE gas-phase chemistry - SPARTA DSMC
seed             12345
dimension        3
boundary         rr rr rr
create_box       0 {lx} 0 {lx} 0 {lx}
create_grid      {n} {n} {n}
# EVERY species that can appear as a reaction product must be declared here,
# and the collide mixture must contain every declared species.
species          air.species N O N2 O2 NO
mixture          air N O N2 O2 NO
mixture          air N2 frac 0.8
mixture          air O2 frac 0.2
mixture          air N frac 0.0
mixture          air O frac 0.0
mixture          air NO frac 0.0
mixture          air vstream 0.0 0.0 0.0 temp {temp}
global           nrho {nrho} fnum {fnum}
collide          vss air air.vss
react            tce air.tce
create_particles air n 0
compute          cnt count N O N2 O2 NO
timestep         {dt}
stats            100
stats_style      step np ncoll nreact c_cnt[1] c_cnt[3] c_cnt[4]
run              {nsteps}
"""


KNOWLEDGE = {
    "chemistry": {
        "description": "Gas-phase chemical reactions (TCE / QK) evaluated "
                       "during DSMC collisions",
        "spatial_dims": [2, 3],
        "key_commands": {
            "react": "react tce|qk|tce/qk <reaction-file> — those three are the "
                     "styles compiled into this build",
            "react_modify": "react_modify [recomb <yes|no>] [rboost <factor>] "
                            "[compute_chem_rates <yes|no>]",
            "species": "species <file> <ID> ... — declare EVERY reactant and "
                       "product species",
            "mixture": "mixture <mixID> <species...> ; mixture <mixID> <sp> "
                       "frac <f> — fractions must sum to <= 1",
            "compute count": "compute <ID> count <species|mixture-group>... — "
                             "the cheapest way to watch composition change",
            "stats_style nreact": "nreact is the gas reactions on THAT "
                                  "timestep; nsreact is the surface ones",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "output_idioms": output_idioms(),
        "pitfalls": [
            "[Syntax] 'react <style> <file>' never checks that the file is a "
            "reaction file. It consumes non-blank, non-comment lines in PAIRS "
            "(formula line, then coefficient line), so what you get depends on "
            "the line COUNT, not on the file's purpose: a complete pair of "
            "junk aborts loudly, but a file whose data lines run out mid-pair "
            "— any file with an odd number of data lines, which includes "
            "single-entry .vss and .species files — ends silently with the "
            "dangling line discarded and ZERO reactions loaded. A trailing "
            "junk line after otherwise valid reactions is dropped the same "
            "way. "
            "Signal: the count line '  style <style> #-of-reactions <N>' under "
            "the 'Gas reaction tallies:' block, printed after a run — that is "
            "the only place SPARTA states how many reactions it parsed. The "
            "loud forms are 'ERROR: Invalid reaction formula in file "
            "(../react_bird.cpp:641)' (line 672 in 27Aug2026), 'ERROR: Invalid reaction type in file "
            "(../react_bird.cpp:659)' (line 690 in 27Aug2026) and, for an unreadable path, 'ERROR on "
            "proc 0: Cannot open reaction file <f> (../react_bird.cpp:552)' (line 583 in 27Aug2026). "
            "Do NOT use the end-of-run 'Gas reactions' counter as the test: "
            "SPARTA prints it in EVERY run, including runs with no react "
            "command at all.",

            "[Syntax] The react STYLE and the reaction FILE are not "
            "cross-checked: 'react qk <a tce file>' is accepted without error "
            "or warning and fires reactions at DIFFERENT rates than 'react tce "
            "<the same file>'. You cannot rely on SPARTA to tell you the model "
            "you asked for is the model you got. "
            "Signal: swap the style keyword and rerun — if the run still "
            "completes and Nreact still fills in, the style was never "
            "validated against the file, and the per-reaction tallies under "
            "'Gas reaction tallies:' will differ between the two.",

            "[Syntax] Mixture fractions are checked only for exceeding 1.0. A "
            "set of fractions that sums to LESS than 1 is accepted and the "
            "deficit is absorbed silently, so a deck that means 80/20 but "
            "writes 0.1/0.1 runs with a composition you did not choose. WHERE "
            "the deficit goes has two rules and you need both: species you left "
            "unset share it equally, AND the LAST species listed in the mixture "
            "absorbs whatever is still missing even when its own fraction was "
            "set explicitly, because init_fraction clamps the last entry of the "
            "cumulative array to 1.0 (mixture.cpp:278) (line 279 in 27Aug2026). So 0.1/0.1 over two "
            "species does not give 0.1/0.1 and does not give 0.5/0.5 — it gives "
            "roughly 0.09/0.91. An earlier wording said only that the remainder "
            "goes to the species you left unset, which does not explain the "
            "entry's own 0.1/0.1 example, where nothing is unset. "
            "Signal: 'ERROR: Mixture <ID> fractions exceed 1.0 "
            "(../mixture.cpp:225)' (line 226 in 27Aug2026) for the over-1 case, and nothing at all for "
            "the under-1 case — verify the realised composition with 'compute "
            "<ID> count <species...>' on step 0.",

            "[Physics] Chemistry needs collision energies above the activation "
            "energy, so at a modest temperature a correctly-configured TCE "
            "deck still reports Nreact = 0 for thousands of steps. A zero "
            "reaction count is therefore not by itself evidence of a "
            "misconfiguration. "
            "Signal: distinguish the two cases with the species counts, not "
            "with Nreact — if 'compute count' shows the product species "
            "strictly at 0 particles for the whole run AND Ncoll is large, "
            "check the setup; if the counts creep up, the chemistry is on and "
            "merely slow.",

            "[Physics] 'react tce/qk' is NOT tce and qk together, and on the "
            "distribution's own air.tce it fires nothing at all. Two separate "
            "facts. First, it DISPATCHES per reaction on the style letter in "
            "column 2 of the coefficient line — 'A' goes to its Arrhenius "
            "branch, 'Q' to its quantum-kinetic one — so a file whose lines "
            "are all 'A' never reaches the QK half. Second, its Arrhenius "
            "branch is a DIFFERENT expression from the one in 'react tce': "
            "react_tce.cpp carries the TCE gamma-function prefactor and the "
            "exponents coeff[3]-1+coeff[5] and z+1.5-coeff[5], while "
            "react_tce_qk.cpp uses a bare coeff[2]*(ecc-coeff[1])^coeff[3] * "
            "(1-coeff[1]/ecc)^coeff[5] with no prefactor. Bird coefficients "
            "calibrated for 'react tce' therefore do not transfer, and on a "
            "hot five-species air box the same file that dissociates under "
            "'react tce' and under 'react qk' produces zero reactions under "
            "'react tce/qk' — rc = 0, no warning, and a particle count that "
            "does not move. "
            "Signal: the run completes, the count line '  style tce/qk "
            "#-of-reactions <N>' shows the reactions WERE parsed, and yet the "
            "'Gas reaction tallies:' block carries no per-reaction lines at "
            "all and Np is flat. Compare against the same deck with 'react "
            "tce' before believing a tce/qk result. (Verified 2026-08-07)",

            "[Setup] 'react tce/qk' refuses two things the other reaction "
            "styles accept, and both checks fire at the START OF THE FIRST "
            "RUN rather than when the react line is parsed. A reaction file "
            "containing any RECOMBINATION entry is rejected outright — which "
            "is what happens with the distribution's mars.tce — unless you "
            "switch them off, and 'react_modify compute_chem_rates yes' is "
            "rejected as unsupported. 'react_modify' is a MODIFIER of the "
            "already-loaded reaction set and must come AFTER the react "
            "command. "
            "Signal: 'ERROR: React tce/qk does not currently support "
            "recombination reactions (../react_tce_qk.cpp:48)' and 'ERROR: "
            "React tce/qk does not currently support the 'react_modify "
            "compute_chem_rates' option (../react_tce_qk.cpp:52)'. 'react "
            "tce/qk' also requires collide vss: 'ERROR: React tce/qk can only "
            "be used with collide vss (../react_tce_qk.cpp:40)'. The escape "
            "for the first is 'react_modify recomb no', which marks the "
            "recombination entries inactive and lets the run proceed. "
            "(Verified 2026-08-07)",

            "[Output] The four tally computes — surf/collision/tally, "
            "surf/reaction/tally, gas/collision/tally, gas/reaction/tally — "
            "are EVENT LISTS, not accumulators, and 'dump tally' is their only "
            "consumer. Each snapshot holds the events of THAT ONE TIMESTEP: "
            "the 'ITEM: NUMBER OF TALLIES' count equals the Nscoll (or Ncoll / "
            "Nreact) printed for the same step, so a dump every 100 steps "
            "gives you one step in a hundred, not a hundred steps' worth. "
            "Nothing can time-average them: no fix ave/* accepts them, "
            "stats_style rejects them, and dump surf rejects them. For a "
            "converged surface quantity use compute surf plus fix ave/surf "
            "instead; use the tally computes when you need the individual "
            "events (which element, which species, when, where). "
            "Signal: 'ERROR: Stats compute does not compute scalar "
            "(../stats.cpp:678)' (line 787 in 27Aug2026) from stats_style and 'ERROR: Dump surf "
            "compute does not compute per-surf info (../dump_surf.cpp:567)' "
            "from dump surf; in a valid dump tally file, compare 'ITEM: NUMBER "
            "OF TALLIES' against the per-step counter in the stats table — "
            "they are equal. (Verified 2026-08-07)",
        ],
    },
}

GENERATORS = {
    "chemistry_box_3d": _hot_air_box_3d,
    "chemistry_3d": _hot_air_box_3d,
}
