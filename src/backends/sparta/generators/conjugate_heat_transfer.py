"""DSMC gas <-> solid conjugate heat transfer, in-code and via preCICE."""

from ._common import output_idioms


def _cht_circle_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Self-consistent wall temperature: the tallied per-element energy flux
    drives a radiative-equilibrium wall temperature (fix surf/temp), which the
    diffuse wall model then reads back through a custom per-surf attribute.
    """
    vstream = params.get("vstream", 2634.1)
    t_free = params.get("t_free", 200.0)
    t_init = params.get("t_init", 100.0)
    emis = params.get("emis", 0.9)
    nrho = params.get("nrho", 4.247e19)
    fnum = params.get("fnum", 7.0e14)
    dt = params.get("dt", 3.5e-7)
    nevery = params.get("nevery", 200)
    nsteps = params.get("nsteps", 600)
    return f"""\
# DSMC gas <-> surface conjugate heat transfer - SPARTA
# Order matters: compute surf -> fix ave/surf -> fix surf/temp (which CREATES
# the custom per-surf attribute) -> surf_collide diffuse s_<name> -> surf_modify
seed             12345
dimension        2
global           nrho {nrho} fnum {fnum} gridcut 0.01
timestep         {dt}
boundary         o ro p
create_box       -0.2 0.65 0.0 0.4 -0.5 0.5
create_grid      30 15 1 block * * *
species          ar.species Ar
mixture          all vstream {vstream} 0.0 0.0 temp {t_free}
collide          vss all ar.vss
collide_modify   vremax 1000 yes
read_surf        circle.surf group 1
compute          q surf all all etot
fix              fq ave/surf all 1 {nevery} {nevery} c_q[1] ave one
# the SOURCE must be a per-surf VECTOR: f_<avesurf>, not the raw compute ID
fix              tw surf/temp all {nevery} f_fq {t_init} {emis} temperature
surf_collide     wall diffuse s_temperature 1.0
surf_modify      1 collide wall
fix              in emit/face all xlo
create_particles all n 0
compute          qtot reduce sum f_fq
stats_style      step np nscoll c_qtot
stats            {nevery}
dump             dsurf surf all {nevery} dump.wall id s_temperature f_fq
run              {nsteps}
"""


KNOWLEDGE = {
    "conjugate_heat_transfer": {
        "description": "DSMC gas <-> solid conjugate heat transfer: SPARTA "
                       "tallies the surface heat flux and reads back a wall "
                       "temperature, either through fix surf/temp in-code or "
                       "through preCICE against an FEM solid",
        "spatial_dims": [2, 3],
        "key_commands": {
            "compute surf": "compute <ID> surf <grp> <mix> etot — total energy "
                            "flux per surface element (per-surf ARRAY)",
            "fix ave/surf": "fix <ID> ave/surf <grp> <Nevery> <Nrepeat> "
                            "<Nfreq> c_<ID>[1] [ave one|running]",
            "fix surf/temp": "fix <ID> surf/temp <grp> <Nevery> <f_source> "
                             "<Tinit> <emisurf> <custom-name> — CREATES the "
                             "custom per-surf attribute; 0 < emisurf <= 1",
            "surf_collide": "surf_collide <ID> diffuse s_<custom-name> <acc> — "
                            "reads the per-element temperature",
            "compute reduce": "compute <ID> reduce sum f_<avesurf> — a fix "
                              "ave/surf with ONE input is a per-surf VECTOR, "
                              "so no bracket",
            "dump surf": "dump <ID> surf all <N> <file> id s_<custom-name> "
                         "f_<avesurf> — writes the wall temperature field",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "coupling_notes": "For a two-code coupling, couple() runs the "
                          "participant knowledge(topic='coupling', "
                          "solver='sparta') serves: a Python wrapper that "
                          "writes the deck, runs the binary and exchanges one "
                          "value per surface element through files -- the "
                          "partner's wall temperature in through 'custom surf "
                          "... file' and 'surf_collide ... diffuse s_<name>', "
                          "the per-element etot tally out through a surf "
                          "dump. SPARTA's Python library (built with 'make "
                          "mode=shlib serial') exposes command / "
                          "extract_global / extract_compute / "
                          "extract_variable only, with no per-surface "
                          "scatter.",
        "output_idioms": output_idioms("per-surf tally idiom", "dump format"),
        "pitfalls": [
            "[Setup] The command order is a hard dependency chain: fix "
            "surf/temp is what CREATES the custom per-surf attribute, so the "
            "surf_collide line that consumes it as s_<name> must come "
            "afterwards. Writing them in the intuitive order (wall model "
            "first) aborts — with exactly the same message you get if the fix "
            "is missing altogether, so the error does not tell you which "
            "mistake you made. "
            "Signal: 'ERROR: Surf_collide tsurf could not find custom "
            "attribute (../surf_collide.cpp:141)' (line 167 in 27Aug2026).",

            "[Syntax] The <source> argument of fix surf/temp is BRACKET-driven, "
            "not compute-versus-fix: a bare id must name a per-surf VECTOR and "
            "a bracketed one must name a per-surf ARRAY. compute surf is "
            "always an array, so it must be given as c_<ID>[N] — the SPARTA "
            "doc page's own example passes a bare 'c_1' and does not work on "
            "this build, and 'c_<ID>[*]' fails the same way because the "
            "wildcard parses as index 0. A fix ave/surf with one input value "
            "is a vector, so it must be given bare as f_<ID>; f_<ID>[1] "
            "aborts. "
            "Signal: 'ERROR: Fix surf/temp compute does not compute per-surf "
            "vector (../fix_surf_temp.cpp:82)' for the bare-compute form, "
            "'ERROR: Fix surf/temp fix does not compute per-surf array "
            "(../fix_surf_temp.cpp:112)' for the over-indexed fix form.",

            "[Syntax] The emissivity argument is strictly inside (0, 1]. Zero "
            "is not 'no radiation', it is rejected — with emisurf = 0 the "
            "radiative-equilibrium temperature would be unbounded. "
            "Signal: 'ERROR: Fix surf/temp emissivity must be > 0.0 and <= 1 "
            "(../fix_surf_temp.cpp:125)'.",

            "[Output] A fix ave/surf fed ONE input value produces a per-surf "
            "VECTOR, referenced as f_ID with NO bracket; indexing it as f_ID[1] "
            "aborts. With two or more inputs it becomes an array read as "
            "f_ID[1], f_ID[2], ... The upstream compute surf feeding it is the "
            "opposite: always an array, so always indexed. "
            "Signal: 'ERROR: Compute reduce fix does not calculate a per-surf "
            "array (../compute_reduce.cpp:293)' (line 297 in 27Aug2026) when you index a "
            "single-value fix. Do not confuse it with line 280, which is the "
            "per-GRID twin of the same message ('... does not calculate a "
            "per-grid array') raised by a fix ave/grid.",

            "[Physics] The DSMC surface flux is statistically noisy and the "
            "wall temperature responds to it, so a short fix ave/surf window "
            "feeds noise straight into the wall temperature and the coupled "
            "system can oscillate without ever failing. Average over a window "
            "long compared with the DSMC fluctuation time and short compared "
            "with the solid's thermal time. "
            "Signal: the dumped s_<custom-name> field changes by a large "
            "fraction between consecutive fix surf/temp updates instead of "
            "drifting smoothly toward equilibrium.",

            "[Setup] SPARTA opens every data file relative to the CURRENT "
            "WORKING DIRECTORY, so a coupled run whose driver starts the "
            "participant in a fresh directory dies at setup unless the "
            "species, VSS and surface files were staged there first. "
            "Signal: 'ERROR on proc 0: Cannot open species file <name>' from "
            "particle.cpp at setup, before any stats line. The openPASO couple() "
            "copies every file listed in a participant's data_files into its "
            "work directory before the first iteration; it does not read the "
            "deck for file names.",

            "[Setup] A half-body surface used with a symmetry plane is an OPEN "
            "curve and fails the watertight test. Put the open endpoints "
            "exactly on a box face. The 'clip' keyword is NOT the remedy and "
            "an earlier wording implied it was: measured, an open curve in the "
            "interior fails the watertight test with 'clip' exactly as without "
            "it, and a curve whose endpoints sit on a box face is accepted "
            "either way. "
            "Signal: 'Watertight check failed' followed by the number of "
            "unmatched points.",

            "[Setup] 'fix <ID> controller Nevery alpha kp ki kd <process-var> "
            "<setpoint> <control-var>' takes exactly ten arguments, and its "
            "two variable slots have DIFFERENT requirements that are easy to "
            "get backwards. The CONTROL variable must be INTERNAL-style — "
            "'variable <name> internal <initial>' — because the fix writes to "
            "it, and an equal-style variable there is rejected. That same "
            "internal variable can then be used DIRECTLY as a surf_collide "
            "temperature ('surf_collide W diffuse v_<name> 1.0') with no "
            "equal-style wrapper, because Variable::equal_style() returns true "
            "for INTERNAL (variable.cpp:1016) (line 1061 in 27Aug2026) — a draft of this entry claimed a "
            "wrapper was required and execution showed it is not. The PROCESS "
            "variable must be a global SCALAR or ONE element of a global "
            "VECTOR; a compute producing a global ARRAY, such as 'compute "
            "boundary' named directly, is rejected. Worst of the three: a "
            "TWO-INDEX reference like f_ID[3][1] is NOT rejected — the parser "
            "truncates the name at the first '[', takes 3 as the index and "
            "silently discards '[1]' — so the controller quietly reads a "
            "different quantity than the one written. Feed it a fix ave/time "
            "in mode vector carrying exactly ONE compute value and index it "
            "once. "
            "Signal: 'ERROR: Fix controller variable is not internal-style "
            "variable (../fix_controller.cpp:135)' (line 136 in 27Aug2026) for the control variable, "
            "and 'ERROR: Fix controller compute does not calculate a global "
            "scalar or vector (../fix_controller.cpp:104)' (line 105 in 27Aug2026) (or the fix twin at "
            "line 118) for a process variable of the wrong shape. For the "
            "two-index case there is NO signal — print the process variable "
            "alongside f_<ID>[1] in stats_style and check the controller is "
            "reacting to the column you meant. (Verified 2026-08-07)",

            "[Numerical] 'fix controller' uses the OPPOSITE sign convention to "
            "textbook PID and the source says so in a comment: it computes "
            "cv -= kp*alpha*tau*err (and likewise for ki, kd), where err = "
            "current - setpoint, so the gain that stabilises a plant of "
            "positive gain is the one a control textbook would call negative. "
            "There is NO clamp on the control variable, no stability check and "
            "no warning: with the wrong sign the loop is positive feedback and "
            "the control variable grows WITHOUT BOUND while the run returns "
            "success. How fast depends on your gains and on the plant, so do "
            "not look for a particular size — look for a column that never "
            "turns around. On a wall-temperature "
            "loop that means an unbounded temperature and a completed run, or "
            "— on the other side of zero — an abort from the physics check "
            "rather than from the controller. Decide the sign by asking which "
            "way the process variable moves when the control variable rises, "
            "and print the control variable in stats_style from the first run. "
            "Signal: rc = 0 and the v_<control> column climbing monotonically "
            "by orders of magnitude, with f_<ID>[1], f_<ID>[2], f_<ID>[3] (the "
            "P, I and D contributions, which the fix exposes as a global "
            "3-vector) doing the same. The only guard that ever fires is the "
            "consumer's own: 'ERROR: Surf_collide tsurf <= 0.0 "
            "(../surf_collide.cpp:183)' (line 209 in 27Aug2026) when the runaway goes negative. "
            "(Verified 2026-08-07)",

            "[Numerical] The controller gains are not dimensionless and not "
            "independent of the schedule: the fix forms tau = Nevery * "
            "timestep and applies kp*alpha*tau, ki*alpha*tau^2 and kd*alpha, "
            "so changing Nevery or the timestep silently rescales the loop. "
            "Halving the timestep halves the proportional action and quarters "
            "the integral action at unchanged kp and ki; alpha exists to carry "
            "the units between the process variable and the control variable "
            "and has to be re-derived whenever either changes. Nevery is also "
            "the sampling interval, so it sets the loop's dead time at the "
            "same time as its gain. Retune after ANY change to timestep, "
            "Nevery or the averaging window feeding the process variable. "
            "Signal: a loop that was stable becomes oscillatory or sluggish "
            "after a timestep or Nevery change with the gains untouched — "
            "compare kp*alpha*tau before and after rather than kp. "
            "(Verified 2026-08-07)",

            "[Physics] SPARTA has NO heat-flux wall boundary condition. Every "
            "surf_collide style either fixes the wall TEMPERATURE (diffuse, "
            "cll, td, impulsive) or transfers no energy at all (specular, "
            "adiabatic), and there is no keyword anywhere that sets a flux. A "
            "prescribed flux can only be reached indirectly, by making the "
            "wall temperature respond to the flux you measure, and there are "
            "exactly two mechanisms in the code. (1) RADIATIVE EQUILIBRIUM: "
            "'fix surf/temp' sets each element's temperature from the tallied "
            "flux through the Stefan-Boltzmann balance, so the converged state "
            "has zero NET flux rather than a flux you chose. (2) PID FEEDBACK: "
            "'fix controller' drives an internal-style variable from any "
            "global scalar, and a surf_collide temperature given as "
            "'v_<name>' follows it — that is the only route to an ARBITRARY "
            "target. Neither is a boundary condition: both are controllers "
            "with a transient, and both need the flux tally to be averaged "
            "before it is fed back. "
            "Signal: grep the deck for the wall temperature. If it is a "
            "number, the flux is an output, not an input, whatever the problem "
            "statement asked for. There is no error to look for here — a deck "
            "that silently imposes a temperature where a flux was wanted runs "
            "perfectly. (Verified 2026-08-07)",
        ],
    },
}

GENERATORS = {
    "conjugate_heat_transfer_circle_2d": _cht_circle_2d,
    "conjugate_heat_transfer_2d": _cht_circle_2d,
}
