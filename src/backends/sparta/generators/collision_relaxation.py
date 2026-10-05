"""Particle-particle collisions (VSS) and internal-energy relaxation."""

from ._common import output_idioms


def _collisional_box_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Collisional argon in a 2d box, with the cell Knudsen number reported so the
    grid resolution is visible in the stats table.
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", 20)
    lx = params.get("lx", 1.0e-4)
    ly = params.get("ly", 1.0e-4)
    nrho = params.get("nrho", 7.07043e23)
    fnum = params.get("fnum", 6.0e11)
    temp = params.get("temp", 273.15)
    dt = params.get("dt", 1.0e-9)
    nsteps = params.get("nsteps", 500)
    return f"""\
# Collisional (VSS) argon in a 2d box - SPARTA DSMC
seed             12345
dimension        2
boundary         rr rr p
create_box       0 {lx} 0 {ly} -0.5 0.5
create_grid      {nx} {ny} 1
species          ar.species Ar
mixture          gas Ar vstream 0.0 0.0 0.0 temp {temp}
global           nrho {nrho} fnum {fnum}
# 'vss' is the only collide style compiled into this build. The mixture named
# here must contain EVERY species loaded by 'species', and the .vss file must
# contain a line for every one of them.
collide          vss gas ar.vss
create_particles gas n 0
compute          tk temp
compute          nr grid all species nrho
compute          tg thermal/grid all all temp
fix              fnr ave/grid all 1 100 100 c_nr[*]
fix              ftg ave/grid all 1 100 100 c_tg[*]
# one species: fnr averages one value, so it is a per-grid VECTOR, named
# f_fnr. With several species it is an array, named f_fnr[*]; SPARTA
# 27Aug2026 refuses f_fnr[*] on a vector ('Cannot use wildcard with
# f_fnr[*] because it does not produce multiple values').
compute          lam lambda/grid f_fnr f_ftg lambda knall
compute          knmin reduce min c_lam[2]
timestep         {dt}
stats            100
stats_style      step np ncoll ncollave c_tk c_knmin
run              {nsteps}
"""


def _internal_energy_box_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Diatomic nitrogen relaxing from a cold rotational mode toward the
    translational temperature. Demonstrates that trot/tvib only evolve when a
    collide command is present.
    """
    nx = params.get("nx", 10)
    ny = params.get("ny", 10)
    lx = params.get("lx", 1.0e-4)
    ly = params.get("ly", 1.0e-4)
    nrho = params.get("nrho", 7.07043e22)
    fnum = params.get("fnum", 7.07043e11)
    temp = params.get("temp", 273.15)
    trot = params.get("trot", 100.0)
    dt = params.get("dt", 1.0e-9)
    nsteps = params.get("nsteps", 500)
    return f"""\
# Rotational relaxation of N2 in a 2d box - SPARTA DSMC
seed             12345
dimension        2
boundary         rr rr p
create_box       0 {lx} 0 {ly} -0.5 0.5
create_grid      {nx} {ny} 1
species          air.species N2
# trot starts BELOW temp; collisions pull the two together
mixture          gas N2 vstream 0.0 0.0 0.0 temp {temp} trot {trot}
global           nrho {nrho} fnum {fnum}
collide          vss gas air.vss
create_particles gas n 0
compute          tg thermal/grid all all temp
compute          tr grid all all trot
compute          rtrans reduce ave c_tg[1]
compute          rrot reduce ave c_tr[1]
fix              keep ave/grid all 1 100 100 c_tg[*]
timestep         {dt}
stats            100
stats_style      step np ncoll c_rtrans c_rrot
run              {nsteps}
"""


KNOWLEDGE = {
    "collision_relaxation": {
        "description": "Particle-particle collisions with the VSS model plus "
                       "rotational / vibrational energy relaxation",
        "spatial_dims": [2, 3],
        "key_commands": {
            "collide": "collide vss <mixID> <file.vss> [relax variable] — vss "
                       "is the ONLY collide style compiled in this build",
            "collide_modify": "collide_modify [vremax <n> <yes|no>] [remain "
                              "<yes|no>] [vibrate <no|discrete|smooth>] "
                              "[rotate <no|smooth>] [ambipolar <yes|no>] "
                              "[nearcp <yes|no> <n>]",
            "mixture": "mixture <mixID> <species...> [temp T] [trot T] "
                       "[tvib T] [frac f] [group SELF]",
            "compute grid ... trot/tvib": "compute <ID> grid <grp> <mix> trot "
                                          "tvib — per-cell internal "
                                          "temperatures",
            "compute thermal/grid": "compute <ID> thermal/grid <grp> <mix> "
                                    "temp — per-cell translational temperature",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "vss_parameters": "The .vss file gives, per species, the reference "
                          "diameter (m), the viscosity exponent omega, the "
                          "reference temperature Tref (K) and the VSS scatter "
                          "parameter alpha. Those numbers are SI in the shipped "
                          "data files.",
        "output_idioms": output_idioms("per-grid tally idiom"),
        "pitfalls": [
            "[Syntax] 'collide vss <mixID>' without the .vss filename is a "
            "hard error, but a .vss file that simply does not list one of your "
            "species is caught only when the collision model is set up. "
            "Signal: 'ERROR: Illegal collide command (../collide_vss.cpp:47)' "
            "for the missing filename; 'ERROR on proc 0: Species <X> did not "
            "appear in VSS parameter file (../collide_vss.cpp:924)' (line 1075 in 27Aug2026) when the "
            "file is the wrong one for your species (e.g. handing air.vss to "
            "an argon deck).",

            "[Syntax] The mixture named on the collide line must contain EVERY "
            "species loaded by the 'species' command — a subset is rejected, "
            "not silently ignored — and the check happens at the start of the "
            "FIRST RUN, not when the collide line is parsed, so a deck that "
            "never reaches 'run' never reports it. "
            "Signal: 'ERROR: Collision mixture does not contain all species "
            "(../collide.cpp:159)' (line 185 in 27Aug2026); a mistyped mixture ID instead gives "
            "'ERROR: Collision mixture does not exist (../collide.cpp:155)' (line 181 in 27Aug2026).",

            "[Physics] Without a collide command, create_particles gives every "
            "particle ZERO rotational and vibrational energy even when the "
            "mixture asks for trot/tvib, so internal-energy diagnostics read "
            "exactly zero for the whole run and no relaxation is possible. "
            "Signal: a Trot or Tvib column that is exactly 0 rather than "
            "noisy, from step 0 onward, while Np is large.",

            "[Syntax] collide_modify silently accepts only its documented "
            "keywords; a typo aborts with a generic message that does not name "
            "the offending keyword, so check the spelling against the compiled "
            "build rather than against a doc page for another version. "
            "Signal: 'ERROR: Illegal collide_modify command "
            "(../collide.cpp:1727)' (line 2344 in 27Aug2026).",

            "[Physics] ROTATIONAL relaxation is active by default but "
            "VIBRATIONAL relaxation is not: with no collide_modify the "
            "vibrational mode is inert, and both 'compute <ID> grid <grp> "
            "<mix> tvib' and 'compute <ID> tvib/grid <grp> <mix>' return "
            "exactly 0.0 for the entire run while Trot visibly relaxes in the "
            "same run. 'collide_modify vibrate smooth' switches it on and also "
            "seeds the mode from the mixture's tvib at creation — but ONLY if "
            "the collide_modify line comes BEFORE create_particles. That "
            "ordering is not checked and getting it wrong is completely "
            "silent: with create_particles first, Tvib is exactly 0 at step 0 "
            "and creeps up from there by collisions alone, staying orders of "
            "magnitude below the mixture's tvib for the whole run, with rc = 0 "
            "and no warning. An earlier wording said only 'Tvib is nonzero "
            "from step 0', which is true of one of the two orderings. "
            "'collide_modify vibrate discrete' is NOT the "
            "quiet no-op an earlier wording claimed: for any species whose "
            "vibdof is > 2 and which has no 'vibfile' on the species line it "
            "ABORTS at run setup with 'ERROR: Discrete vibrational info for "
            "species <X> not read in (../particle.cpp:177)' (line 176 in 27Aug2026) "
            "(particle.cpp skips the check only for vibdof <= 2). For a "
            "vibdof <= 2 species it is accepted and starts at 0, but it does "
            "not stay there — it is zero for the first stats blocks and then "
            "collisions populate the levels. Only a species with no "
            "vibrational mode at all reads "
            "identically 0 under every setting. "
            "Signal: a Tvib column that is exactly 0.0 on every stats line "
            "while the Trot column moves — rc = 0, no warning — which "
            "identifies the DEFAULT (no collide_modify) case; the 'discrete' "
            "mistake announces itself with the particle.cpp:177 (line 176 in 27Aug2026) abort "
            "instead.",

            "[Output] 'compute grid <grp> <mix> tvib' and 'compute tvib/grid "
            "<grp> <mix>' are different estimators and do not agree: on the "
            "same smooth-vibration run they differ by a factor of several. "
            "Pick one and use it consistently; do not compare a number from "
            "one against a number from the other. "
            "Signal: two vibrational-temperature columns in the same stats "
            "table whose ratio is roughly constant and far from 1.",

            "[Setup] The two thermostats are not variants of one command and "
            "their argument lists differ in length as well as meaning. 'fix "
            "<ID> temp/rescale <Nevery> <Tstart> <Tstop> [ave yes|no]' "
            "rescales particle velocities per cell and RAMPS the target "
            "LINEARLY across the run: the target at any step is interpolated "
            "from Tstart at the run's first step to Tstop at its last, so a "
            "second 'run' command restarts the ramp from Tstart and a run "
            "length change silently changes the heating rate. 'fix <ID> "
            "temp/global/rescale <Nevery> <Tstart> <Tstop> <fraction>' takes a "
            "fourth argument, a relaxation fraction in [0,1] that damps how "
            "much of the gap is closed at each application, and acts on the "
            "whole domain rather than cell by cell. Neither is a physical "
            "process: they inject or remove energy to hit a number, so any "
            "transport coefficient measured while one is active is a property "
            "of the thermostat. "
            "Signal: 'ERROR: Illegal fix temp/rescale command "
            "(../fix_temp_rescale.cpp:31)' and 'ERROR: Illegal fix "
            "temp/global/rescale command (../fix_temp_global_rescale.cpp:29)' "
            "for the wrong argument count. When it is working, 'compute <ID> "
            "temp' tracks the interpolated target rather than the gas — plot "
            "it against step and it is a straight line from Tstart to Tstop, "
            "which is the confirmation that you are reading the thermostat and "
            "not the physics. (Verified 2026-08-07)",

            "[Output] 'fix <ID> ave/histo Nevery Nrepeat Nfreq lo hi Nbin "
            "<value>' needs 'mode vector' for any per-particle or per-grid "
            "input and refuses it otherwise, and the default scalar mode "
            "quietly does something much smaller than it looks: fed a GLOBAL "
            "compute it histograms Nrepeat numbers per window, not the "
            "particle population. Its global 4-vector is [1] = samples INSIDE "
            "[lo,hi], [2] = samples OUTSIDE the range, [3] = minimum, [4] = "
            "maximum, and the histogram bins themselves are the separate 2d "
            "ARRAY f_ID[bin][col]. Column [2] is the load-bearing one: values "
            "outside the range are DROPPED from every bin without a warning, "
            "so a range guessed too narrow silently truncates the tail you "
            "were looking for. Check [2] against [1] before reading any bin, "
            "and widen lo/hi rather than adding bins. "
            "Signal: 'ERROR: Fix ave/histo cannot input per-particle values in "
            "scalar mode (../fix_ave_histo.cpp:228)' (line 230 in 27Aug2026) and the per-grid twin at "
            "line 231; 'ERROR: Illegal fix ave/histo command "
            "(../fix_ave_histo.cpp:53)' when lo, hi or Nbin are omitted. A "
            "f_ID[1] equal to Nrepeat rather than to the particle count means "
            "you are histogramming a global scalar. (Verified 2026-08-07)",

"[Numerical] 'fix <ID> dt/reset <Nevery> <c_ID|f_ID> <weight> "
            "<resetflag>' can change the global timestep underneath a running "
            "simulation, and the last argument decides whether it does. With "
            "resetflag 0 the fix only COMPUTES a recommended timestep and "
            "publishes it as f_ID; with 1 or 2 it WRITES it into the global "
            "timestep, so the dt you set with the 'timestep' command is gone "
            "after the first Nevery steps and every subsequent result belongs "
            "to a step size you did not choose — a per-cell recommendation on "
            "a near-equilibrium box can be several times the value a user "
            "picked. <weight> in [0,1] is an exponential smoothing factor on "
            "the change, not a safety factor. The recommendation itself comes "
            "from 'compute dt/grid', so it inherits that compute's dependence "
            "on a fix ave/grid that has to have produced output first. "
            "Signal: put 'dt' in stats_style. A constant Dt column means "
            "resetflag 0 or no fix at all; a Dt column that steps at multiples "
            "of Nevery means the timestep is being rewritten, and the "
            "convergence study you thought you were running is not one. "
            "(Verified 2026-08-07)",
        ],
    },
}

GENERATORS = {
    "collision_relaxation_box_2d": _collisional_box_2d,
    "collision_relaxation_2d": _collisional_box_2d,
    "collision_relaxation_internal_energy_2d": _internal_energy_box_2d,
}
