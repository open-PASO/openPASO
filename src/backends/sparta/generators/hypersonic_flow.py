"""Hypersonic rarefied flow over a body: inflow, shock, surface heating."""

from ._common import output_idioms


def _hypersonic_circle_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Hypersonic argon over a 2d circle, with a diffuse wall, a surface
    energy-flux tally and Knudsen-driven grid adaptation.
    """
    vstream = params.get("vstream", 2634.1)
    t_free = params.get("t_free", 200.0)
    t_wall = params.get("t_wall", 500.0)
    nrho = params.get("nrho", 4.247e19)
    fnum = params.get("fnum", 7.0e14)
    dt = params.get("dt", 3.5e-7)
    nsteps = params.get("nsteps", 600)
    nevery = params.get("nevery", 200)
    return f"""\
# Hypersonic argon over a 2d circle with Kn-driven grid adaptation - SPARTA
seed             12345
dimension        2
global           nrho {nrho} fnum {fnum} gridcut 0.01
timestep         {dt}
# xlo/xhi outflow, ylo reflecting (symmetry line), yhi outflow, z periodic
boundary         o ro p
create_box       -0.2 0.65 0.0 0.4 -0.5 0.5
create_grid      30 15 1 block * * *
species          ar.species Ar
mixture          all vstream {vstream} 0.0 0.0 temp {t_free}
collide          vss all ar.vss
collide_modify   vremax 1000 yes
read_surf        circle.surf group 1
surf_collide     wall diffuse {t_wall} 1.0
surf_modify      1 collide wall
fix              in emit/face all xlo
create_particles all n 0
# surface heat flux (per-surf array -> index it)
compute          q surf all all etot
fix              fq ave/surf all 1 {nevery} {nevery} c_q[1] ave one
compute          qtot reduce sum f_fq
# cell Knudsen number, used both as a resolution check and to drive adaptation
compute          nr grid all all nrho
compute          tg thermal/grid all all temp
fix              fg ave/grid all 1 {nevery} {nevery} c_nr[*] c_tg[*] ave one
compute          lam lambda/grid f_fg[1] f_fg[2] lambda knall
fix              ad adapt {nevery} all refine coarsen value c_lam[2] 2.0 4.5 &
                 combine min thresh less more cells 2 2 1
stats_style      step np nattempt ncoll nscoll ngrid maxlevel c_qtot
stats            {nevery}
run              {nsteps}
"""


KNOWLEDGE = {
    "hypersonic_flow": {
        "description": "Hypersonic rarefied flow over a body: bow shock, "
                       "surface pressure and heat flux, DSMC",
        "spatial_dims": [2, 3],
        "key_commands": {
            "fix emit/face": "fix <ID> emit/face <mixID> <face...> [n <Np>] "
                             "[nevery <N>] [perspecies yes|no] [subsonic <P> "
                             "<T|NULL>] [region <regID>]",
            "collide_modify": "collide_modify vremax <N> <yes|no> — the "
                              "upstream hypersonic decks all set this",
            "read_surf": "read_surf <file> group <ID> [trans/scale/rotate/clip]",
            "compute surf": "compute <ID> surf <grp> <mix> press shx shy etot "
                            "— surface pressure, shear and heat flux",
            "fix adapt": "fix <ID> adapt <Nevery> <grp> refine coarsen value "
                         "c_lam[2] <rthresh> <cthresh> combine min thresh less "
                         "more cells 2 2 1",
            "compute lambda/grid": "compute <ID> lambda/grid <nrho-src> "
                                   "<temp-src> lambda knall — column 2 is "
                                   "Kn_cell, the natural adaptation criterion",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "output_idioms": output_idioms("per-surf tally idiom", "per-grid tally idiom"),
        "pitfalls": [
            "[Setup] 'fix emit/face' cannot inject through a PERIODIC face. "
            "The inflow face must not be declared periodic ('p') in the "
            "'boundary' command ('o', 'r' and 's' faces are accepted), and "
            "this is checked only when a run starts, not when the fix is "
            "defined, so it fires late in a long deck. "
            "Signal: 'ERROR: Cannot use fix emit/face on periodic boundary "
            "(../fix_emit_face.cpp:182)' (SPARTA 24Sep2025; 27Aug2026 raises "
            "it from line 183).",

            "[Physics] Emitting on ONE face of a box whose other faces are "
            "outflow need not hold the domain at the freestream density: "
            "particles leave through the outflow faces while only one face "
            "injects, and the particle count can settle below the seeded "
            "count, silently — the flow you are sampling is then thinner "
            "than the freestream you specified. How far below depends on the "
            "case. A seeded 2d nitrogen channel at 300 K emitting on xlo "
            "alone settles lower the slower the stream: about 53 % of the "
            "seeded count at 300 m/s, about 87 % at 1000 m/s. This module's "
            "own default deck holds its count (18416 particles at step 0, "
            "18070 at step 400, 19462 at step 600). "
            "Signal: NOT that the Np column keeps falling — an earlier "
            "wording said it 'decreases monotonically over the run instead of "
            "levelling off' and told you to drive the case to steady state and "
            "check Np has flattened. Executed, it DOES flatten, so that check "
            "passes on the broken deck. Compare instead against an otherwise "
            "identical deck that emits on every inflow face: in the 300 m/s "
            "channel that one holds about 92 % of the seeded count against "
            "53 %, and this module's default deck emitting on xlo and yhi "
            "reaches 19950 at step 600 against 19462. A plateau well below the "
            "density you asked for is the symptom, not the absence of a "
            "plateau.",

            "[Physics] Surface heat flux and pressure need statistical steady "
            "state, which for a hypersonic case is several flow-through times. "
            "Sampling the first stats block gives a wrong number while the run "
            "looks entirely healthy — rc = 0, no error, no warning, for the "
            "whole transient. "
            "How wrong: a QUARTER, not a factor of several. An earlier wording "
            "said 'wrong by a large factor'; measured on a Mach-5 argon "
            "cylinder with 'compute surf all all etot' through a fix ave/surf, "
            "the first 200-step window came out 1.26 times the converged "
            "plateau, reproducibly on two seeds. Told to look for a large "
            "factor you would see a quarter, conclude the transient had "
            "passed, and sample it anyway. "
            "Signal: do not judge it by the size of the offset at all. Tally "
            "the quantity in successive fix ave/surf windows and compare "
            "against the SEED-TO-SEED SPREAD of the late windows, which you "
            "measure by rerunning with a different seed. Measured here that "
            "spread is about half a percent, so the first window sits tens of "
            "times outside it while every late window agrees inside it. That "
            "separation is the convergence test — not the step count, not the "
            "wall-clock, and not how large the discrepancy looks.",

            "[Numerical] The bow shock and the wall boundary layer are far "
            "thinner than the freestream mean free path scale, so a uniform "
            "grid sized on freestream conditions is far too coarse there. "
            "SPARTA does not warn: the run is clean and the shock is simply "
            "smeared. "
            "Signal: 'compute <ID> lambda/grid ... knall' reduced by a compute "
            "reduce in min mode gives a cell Knudsen number below 1 "
            "somewhere in the domain, "
            "with rc = 0 and no warning; drive 'fix adapt' from that column "
            "rather than from particle count. Measured on a Mach-5 argon "
            "cylinder case, the uniform freestream-sized grid gives a cell "
            "Knudsen minimum well below 1; refining from the Knudsen column "
            "lifts it by about a factor of three while refining on particle "
            "count leaves it where it was. NOTE that upstream's "
            "examples/circle cannot show any of this: it uses nrho 1.0, so the "
            "flow is free molecular and the cell Knudsen numbers come out "
            "around 1e18. Scale the density physically before drawing a "
            "resolution conclusion from that deck.",

            "[Numerical] 'fix adapt' driven by a per-grid COMPUTE (c_ID[i]) "
            "re-evaluates it each adaptation step, but driven by a FIX (f_ID) "
            "it can only act on steps where that fix has produced output. Set "
            "the fix Nfreq and the adapt Nevery to the same value. Measured: "
            "an adapt Nevery of 30 against a fix Nfreq of 50 aborts, and the "
            "same Nevery of 30 driven by the per-grid COMPUTE instead runs to "
            "completion, which is the whole difference this entry is about. "
            "Signal: the adapt's OWN complaint is 'ERROR: Fix for adapt not "
            "computed at compatible time (../adapt_grid.cpp:502)' — that is the "
            "one this entry is about, and an earlier wording omitted it. The "
            "other two fire when something else reads the fix: 'ERROR: Fix used "
            "in compute reduce not computed at compatible time "
            "(../compute_reduce.cpp:805)' (line 809 in 27Aug2026) and 'ERROR: Stats and fix not "
            "computed at compatible times (../stats.cpp:203)' (line 209 in 27Aug2026), the latter only "
            "for a fix producing a global scalar such as fix ave/time; a "
            "per-grid fix in stats_style gives 'ERROR: Stats fix does not "
            "compute scalar (../stats.cpp:705)' (line 814 in 27Aug2026) instead. An earlier wording "
            "said all three abort 'mid-run, right after the step-0 stats "
            "line'; only the compute reduce one does. The adapt's own abort "
            "and the stats one happen at setup, before any stats row is "
            "printed.",

            "[Setup] Over-refinement is unbounded unless you cap it: each "
            "refinement level multiplies the cell count, and the particle "
            "count per cell falls with it until every per-cell diagnostic is "
            "noise. Use the 'maxlevel' keyword. "
            "Signal: the Ngrid and Maxlevel columns keep climbing between "
            "stats lines while Np is flat — cells are being split without new "
            "particles to fill them.",

            "[Syntax] 'compute <ID> boundary <mixID> <values...>' breaks the "
            "pattern of every other spatial compute: it takes a MIXTURE ID and "
            "NO group ID, where compute grid and compute surf both take a "
            "group first and a mixture second. Copying that pattern here "
            "consumes 'all' as the mixture and then rejects the second 'all' "
            "as a value, with a message that names the command rather than the "
            "argument. It is a GLOBAL ARRAY, not a scalar and not per-surf: "
            "rows are the box faces in the order xlo, xhi, ylo, yhi and then "
            "zlo, zhi in 3d — only FOUR rows in 2d — and columns are ngroup * "
            "nvalue. So it must go through 'fix ave/time ... c_ID[*] mode "
            "vector'; without 'mode vector' fix ave/time asks it for a scalar "
            "and aborts. Values are n nwt nflux mflux press shx shy shz ke "
            "erot evib etot. A periodic face tallies nothing and reads 0. "
            "Signal: 'ERROR: Illegal compute boundary command "
            "(../compute_boundary.cpp:64)' when a group ID is passed, and "
            "'ERROR: Fix ave/time compute does not calculate a scalar "
            "(../fix_ave_time.cpp:137)' (line 140 in 27Aug2026) when 'mode vector' is missing — the "
            "same message a per-grid compute produces there, so it does not by "
            "itself tell you which mistake you made. (Verified 2026-08-07)",

            "[Output] 'fix <F> ave/time Nevery Nrepeat Nfreq ...' enforces two "
            "constraints that a reader cannot tell apart from the message: "
            "Nfreq must be an exact multiple of Nevery, AND Nevery*Nrepeat "
            "must not exceed Nfreq. Both violations abort with the SAME "
            "generic text, which names neither rule and does not print the "
            "three numbers, so the fix is to check both by hand. The window "
            "the fix averages is the LAST Nrepeat samples ending at each "
            "Nfreq step, not the whole Nfreq interval — 'ave/time 10 5 1000' "
            "averages five samples out of the final fifty steps of every "
            "thousand and ignores the other 950. "
            "Signal: 'ERROR: Illegal fix ave/time command "
            "(../fix_ave_time.cpp:129)' (line 132 in 27Aug2026) for either violation. Widen the window "
            "with Nrepeat, not with Nfreq. (Verified 2026-08-07)",

"[Output] A fix's output can only be printed on steps where the "
            "fix has fresh data, and SPARTA checks this the STRICT way: "
            "'stats N' must be a multiple of the fix's Nfreq, so sampling the "
            "stats table MORE often than the averaging window is a hard error, "
            "not a repeated value. 'stats 50' against Nfreq 100 aborts; 'stats "
            "200' against Nfreq 100 runs. 'compute reduce' reading a fix "
            "raises the same objection with its own message. The check fires "
            "at the start of the run, after setup has printed, so a deck can "
            "look healthy for several screens before it stops. "
            "Signal: 'ERROR: Stats and fix not computed at compatible times "
            "(../stats.cpp:203)' (SPARTA 24Sep2025; line 209 in 27Aug2026), "
            "and from the reduce path 'ERROR: Fix used in "
            "compute reduce not computed at compatible time', which "
            "compute_reduce.cpp raises from three different call sites (lines "
            "778, 805 and 832 in SPARTA 24Sep2025; 782, 809 and 836 in "
            "27Aug2026) for a per-particle, a per-grid and a per-surf "
            "fix input (a global fix is refused earlier with 'Compute reduce "
            "fix calculates global values') — "
            "so match the text, not the line. (Verified 2026-08-07)",

"[Output] The FIRST stats row of a run prints every f_<ID> of a "
            "fix ave/* as a literal 0, because no averaging window has closed "
            "yet. There is no warning and no sentinel value — the zero is "
            "indistinguishable from a real measurement of zero, and it is the "
            "row an agent reading the top of the table sees first. The same "
            "zero reappears at the start of every subsequent 'run' command. "
            "Signal: the f_<ID> column reads exactly 0 on the step-0 row and "
            "jumps to a physical value on the first row at or after Nfreq. "
            "Discard the first row; if you need a time series on disk instead "
            "of in the table, add 'file <name>' to the fix ave/time line, "
            "which writes one row per Nfreq step. (Verified 2026-08-07)",
        ],
    },
}

GENERATORS = {
    "hypersonic_flow_circle_2d": _hypersonic_circle_2d,
    "hypersonic_flow_2d": _hypersonic_circle_2d,
}
