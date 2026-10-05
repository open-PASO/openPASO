"""Rarefied / free-molecular flow: SPARTA decks and knowledge."""

from ._common import output_idioms


def _free_molecular_box_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Collisionless thermal argon in a 2d box with specularly reflecting walls.
    There is deliberately no collide command.

    The DEFAULTS ARE NOT IN THE FREE-MOLECULAR LIMIT and must be changed
    before the name of this template is true of the run. At nrho 7.07043e22
    with lx = ly = 1e-4 m the VHS mean free path for argon is ~1.9e-5 m, i.e.
    Kn_box ~ 0.19 — transitional. Measured on the installed build: adding the
    single line 'collide vss gas ar.vss' to this exact deck yields
    'Collide attempts = 6212' and 'Collide occurs = 4893' over the default
    500 steps with 1000 particles. Omitting collide at these numbers is
    therefore suppressing ~4.9 real collisions per particle, not modelling a
    collisionless gas. For a genuine Kn >> 1 run lower nrho (and fnum with
    it) by several orders of magnitude, or enlarge the box.
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", 20)
    lx = params.get("lx", 1.0e-4)
    ly = params.get("ly", 1.0e-4)
    nrho = params.get("nrho", 7.07043e22)
    fnum = params.get("fnum", 7.07043e11)
    temp = params.get("temp", 273.15)
    dt = params.get("dt", 1.0e-9)
    nsteps = params.get("nsteps", 500)
    return f"""\
# Free-molecular (collisionless) argon in a 2d box - SPARTA DSMC
# No 'collide' command => Ncoll is 0 by construction, so Ncoll == 0 here proves
# nothing about the regime. Kn_box at the DEFAULT nrho/box above is only ~0.19:
# these numbers are transitional, not collisionless. Lower nrho and fnum by
# several decades (or enlarge the box) before calling a run free-molecular.
seed             12345
dimension        2
boundary         rr rr p
create_box       0 {lx} 0 {ly} -0.5 0.5
create_grid      {nx} {ny} 1
species          ar.species Ar
mixture          gas Ar vstream 0.0 0.0 0.0 temp {temp}
# global MUST precede create_particles, else 'Created 0 particles'
global           nrho {nrho} fnum {fnum}
create_particles gas n 0
# compute temp is the KINETIC temperature (bulk motion NOT removed);
# with vstream 0 it coincides with the thermal temperature.
compute          tk temp
timestep         {dt}
stats            100
stats_style      step np ncoll c_tk
run              {nsteps}
"""


def _fourier_channel_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Collisional argon between a cold (ylo) and a hot (yhi) fully accommodating
    diffuse wall — the standard wall-bounded conduction test. Reports the
    per-cell thermal temperature profile and the cell Knudsen number.
    """
    nx = params.get("nx", 8)
    ny = params.get("ny", 40)
    lx = params.get("lx", 2.0e-5)
    ly = params.get("ly", 1.0e-4)
    nrho = params.get("nrho", 7.07043e23)
    fnum = params.get("fnum", 2.0e11)
    t_cold = params.get("t_cold", 300.0)
    t_hot = params.get("t_hot", 1000.0)
    dt = params.get("dt", 1.0e-9)
    nsteps = params.get("nsteps", 2000)
    return f"""\
# 2d Fourier channel: argon between a cold ylo wall and a hot yhi wall.
# Both y faces are 'surface' boundaries ('s'), each bound to its own
# surf_collide model. x is periodic, so this is a pure 1d conduction problem.
seed             12345
dimension        2
boundary         p ss p
create_box       0 {lx} 0 {ly} -0.5 0.5
create_grid      {nx} {ny} 1
species          ar.species Ar
mixture          gas Ar vstream 0.0 0.0 0.0 temp {t_cold}
global           nrho {nrho} fnum {fnum}
# a 'diffuse' wall needs BOTH a temperature and an accommodation coefficient,
# in that order; 'specular' takes no temperature and exchanges no energy.
surf_collide     cold diffuse {t_cold} 1.0
surf_collide     hot  diffuse {t_hot} 1.0
bound_modify     ylo collide cold
bound_modify     yhi collide hot
collide          vss gas ar.vss
create_particles gas n 0
# temperature profile: per-cell thermal temperature, time-averaged
compute          tg thermal/grid all all temp
fix              ftg ave/grid all 10 20 200 c_tg[*]
compute          tmin reduce min f_ftg
compute          tmax reduce max f_ftg
# resolution check: cell Knudsen number must be >= 1 for the cells to be
# smaller than the local mean free path
compute          nr grid all species nrho
fix              fnr ave/grid all 10 20 200 c_nr[*]
# one species: fnr averages one value, so it is a per-grid VECTOR, named
# f_fnr. With several species it is an array, named f_fnr[*]; SPARTA
# 27Aug2026 refuses f_fnr[*] on a vector ('Cannot use wildcard with
# f_fnr[*] because it does not produce multiple values').
compute          lam lambda/grid f_fnr f_ftg lambda knall
compute          knmin reduce min c_lam[2]
timestep         {dt}
stats            200
stats_style      step np ncoll c_tmin c_tmax c_knmin
dump             dgrid grid all 1000 dump.profile id yc f_ftg
run              {nsteps}
"""


KNOWLEDGE = {
    "rarefied_flow": {
        "description": "Rarefied / free-molecular gas flow (high Knudsen "
                       "number) solved with DSMC simulator particles",
        "spatial_dims": [2, 3],
        "regime": "Kn = lambda/L. Kn >> 1 is free-molecular (drop the collide "
                  "command); Kn ~ 0.01-1 is the transitional regime DSMC is "
                  "built for; Kn << 0.01 is continuum and a Navier-Stokes "
                  "solver is cheaper and more accurate.",
        "key_commands": {
            "global": "global nrho <n> fnum <F> [gridcut <d>] — nrho is real "
                      "number density (1/m^3), fnum is real particles per "
                      "simulation particle",
            "create_grid": "create_grid Nx Ny Nz [levels <N> then a region or "
                           "subset clause per level] — there is no "
                           "'level' keyword; see adaptive_grid",
            "create_particles": "create_particles <mixID> n 0 [region <regID>]",
            "compute temp": "compute <ID> temp — GLOBAL kinetic temperature, "
                            "streaming velocity NOT removed",
            "compute thermal/grid": "compute <ID> thermal/grid <grp> <mix> "
                                    "temp — per-cell temperature with the "
                                    "per-cell mean velocity removed",
            "compute lambda/grid": "compute <ID> lambda/grid <nrho-src> "
                                   "<temp-src> lambda knall — column 1 is the "
                                   "mean free path, column 2 is Kn_cell. A fix "
                                   "as <nrho-src> is f_ID for one species, "
                                   "f_ID[*] for several",
            "compute dt/grid": "compute <ID> dt/grid <grp> <tfrac> <cfrac> "
                               "<tau> <temp> <usq> <vsq> <wsq> — recommended "
                               "per-cell timestep",
            "compute reduce": "compute <ID> reduce min|max|ave|sum "
                              "c_X[i]|f_X[i] — the only way to get a per-grid "
                              "or per-surf quantity into stats_style",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "unit_systems": "SI by default (m, kg, s, K). 'units cgs' switches the "
                        "constants but does NOT convert your input files.",
        "output_idioms": output_idioms("per-grid tally idiom", "boundary tally idiom"),
        "pitfalls": [
            "[Numerical] Grid cells much larger than the local mean free path "
            "run cleanly and give an over-diffusive answer, and a "
            "particle-count check does not catch it — a coarse grid has MORE "
            "particles per cell, not fewer. The cell Knudsen number is the "
            "quantity that must be checked. "
            "Signal: 'compute <C> lambda/grid <nrho-src> <temp-src> lambda "
            "knall' reduced by a compute reduce in min mode over c_C[2] "
            "returns a value below 1 "
            "(cells larger than a mean free path). SPARTA never checks this "
            "for you and never warns.",

            "[Numerical] compute lambda/grid reads from a fix ave/grid that "
            "has produced no output yet, so on the first stats line it "
            "returns SPARTA's no-data sentinel, not a physical value. The "
            "sentinel is BIG = 1e+20 (compute_lambda_grid.cpp:37), written "
            "into lambda whenever a cell's number density is zero; the "
            "per-cell Kn column then carries lambda/cell-size, so its "
            "magnitude depends on your cell size (1e+24 at a 1e-4 m cell, "
            "1e+25 at 1e-5 m), which a compute reduce max "
            "shows, while a compute reduce min reads exactly 1e+20 because "
            "reduce min starts from 1e+20. compute dt/grid fed from the same "
            "fixes returns 0 instead: it writes 0 for every cell with no "
            "particles or zero tau, temperature or speed "
            "(compute_dt_grid.cpp:594). "
            "Step 0 is NOT the only exposure: any cell that holds no particles "
            "keeps the sentinel on EVERY stats line, so on a grid refined past "
            "one particle per cell the maximum stays at 1e+20 forever while "
            "the minimum looks healthy. "
            "Signal: an absurd 1e+20-scale value in the lambda or Kn column. "
            "Never sample diagnostics at step 0, and reduce with BOTH a "
            "compute reduce in min mode and one in max mode — a max still "
            "pinned at "
            "1e+20 after the fix has produced output means empty cells, not a "
            "warm-up transient.",

            "[Numerical] A timestep larger than the mean collision time runs "
            "cleanly and inflates transport — SPARTA has no CFL check and "
            "never compares your 'timestep' with its own recommendation. "
            "'compute <D> dt/grid ...' computes a recommended per-cell "
            "timestep and reports the SAME value in a well-resolved run and in "
            "one using a hundred times too large a step. "
            "Signal: reduce compute dt/grid with a compute reduce in min mode "
            "and "
            "assert your timestep is below it; a run whose recommended dt sits "
            "far below the dt you set exits 0 regardless.",

            "[Numerical] An fnum that leaves well under one simulation "
            "particle per cell still runs, still reports a plausible collision "
            "rate, and turns every PER-CELL diagnostic into garbage — mean "
            "free paths tens of orders of magnitude too large. The collapse is "
            "specifically at sub-particle-per-cell occupancy; around one "
            "particle per cell the diagnostics are still sane. "
            "Signal: 'compute <C> grid all all n' reduced by a compute reduce "
            "in min mode over c_C[1] — require the cell MINIMUM to be >= 1, "
            "not just the "
            "average, before trusting any per-cell quantity.",

            "[Physics] 'compute <ID> temp' sums the kinetic energy of every "
            "particle WITHOUT subtracting any mean velocity, so it reports "
            "T_thermal + m*|vstream|^2/(3*kB) — a translational temperature "
            "inflated by the bulk motion. The inflation is proportional to "
            "SPECIES MASS and to the SQUARE of the stream speed, so it is "
            "enormous for a heavy species at hypersonic speed and negligible "
            "for a light species or a slow flow. The divisor is 3*kB in BOTH "
            "2d and 3d — compute_temp.cpp normalises by 3, not by 'dimension', "
            "because even a 2d run carries three velocity components — so it "
            "is independent of dimension and of whether collisions are on. It "
            "is also purely "
            "translational: for a molecular gas it drifts as translation "
            "exchanges energy with rotation. "
            "Signal: put 'compute <ID> temp' and a reduced 'compute <ID> "
            "thermal/grid <grp> <mix> temp' in the same stats_style; a gap of "
            "about m*|vstream|^2/(3*kB) between them means the first column "
            "is not a thermal temperature. With vstream 0 the two agree.",

            "[Numerical] compute thermal/grid removes the per-cell MEAN "
            "velocity, so it only removes the bulk motion to the extent the "
            "cell resolves the flow. Across an unresolved velocity gradient "
            "the sub-cell shear variance is counted as thermal motion and the "
            "reported temperature reads high; refining the grid drives it "
            "down toward the true value. "
            "Signal: the cell-averaged thermal/grid temperature falls when you "
            "refine the grid at fixed physics — but by a FEW PERCENT, not "
            "'substantially' as an earlier wording had it. Measured on a "
            "Couette slab with walls translating at plus and minus 1000 m/s, "
            "well above the thermal speed, a tenfold refinement moved it about "
            "four and a half percent. That follows from the mechanism and is "
            "not a property of this case: the sub-cell velocity spread is "
            "(dU/dy)*dy, so its variance enters as the SQUARE of the cell size "
            "and in only ONE of three velocity components. Two things make "
            "that small number unusable unless you control for them. First, "
            "read it through a 'fix ave/grid', never instantaneously: "
            "refining lowers the per-cell occupancy and the low-occupancy bias "
            "of compute thermal/grid pushes the reading DOWN as well, in the "
            "same direction, so an instantaneous comparison cannot tell the "
            "two apart. Second, run more than one seed and ask whether the "
            "coarse and fine CLUSTERS separate rather than whether two numbers "
            "differ — measured here the seed spread was about a percent at the "
            "coarse grid, so a single pair gives you only a few times the "
            "noise. Removing the wall motion entirely collapses the difference "
            "to a fraction of a percent and the clusters overlap, which is the "
            "control that shows the residual really is the shear.",

            "[Numerical] compute thermal/grid is biased LOW per timestep "
            "when cells hold few particles: it subtracts the per-cell SAMPLE "
            "mean, which costs one degree of freedom, and SPARTA writes "
            "exactly 0 K into any cell holding one particle or none. A "
            "'compute reduce ave' then averages those hard zeros in, so at a "
            "couple of particles per cell the reported cell-average can be a "
            "THIRD of the temperature you set — much worse than the naive "
            "one-degree-of-freedom estimate. Time-averaging through 'fix "
            "ave/grid' largely removes it, because the effective count is the "
            "particle count times the number of samples. "
            "Signal: an instantaneous cell-averaged thermal/grid temperature "
            "far below 'compute temp' in a gas at rest, which climbs back "
            "toward it when you lower fnum or route the compute through a fix "
            "ave/grid. Note the first tens of steps also fall on their own as "
            "the near-uniform fill from create_particles relaxes to Poisson "
            "occupancy — that is not physics either.",

            "[Physics] A homogeneous equilibrium box is worthless as a "
            "timestep or grid-resolution test: the collision statistics of a "
            "gas at rest are almost unchanged by a ten-times-too-large "
            "timestep or a ten-times-too-coarse grid, because there is no "
            "gradient for the errors to act on. Both errors show up only in a "
            "gradient-driven quantity. "
            "Signal: a dt or grid study whose Ncoll / Ncollave columns are flat "
            "is measuring nothing — move the study to a wall-bounded or "
            "flow-driven case before drawing a conclusion.",

            "[Physics] Wall and boundary fluxes in a driven run need several "
            "flow-through times; the first stats block is not an answer and "
            "the run looks perfectly healthy throughout the transient. "
            "Signal: at steady state the two opposed wall fluxes are equal and "
            "opposite. Tally both and use that balance as the convergence "
            "test, not the wall-clock or the step count.",

"[Output] 'compute <ID> property/grid <grid-group> <attrs>' is the "
            "exception to SPARTA's per-grid shape rule: with ONE attribute it "
            "produces a per-grid VECTOR read as c_ID with no bracket, and only "
            "with two or more does it produce an ARRAY read as c_ID[i]. "
            "'compute grid' and 'compute surf' are ALWAYS arrays, so the "
            "bracket convention you learn on those is wrong here and the two "
            "mistakes have opposite messages. It also holds GEOMETRY ONLY — "
            "id, proc, xlo/ylo/zlo, xhi/yhi/zhi, xc/yc/zc, vol — so any flow "
            "quantity asked of it is rejected; nrho and temperature come from "
            "'compute grid'. 'compute property/surf' takes a SURF group and "
            "the analogous element geometry (id, v1x..v3z, xc/yc/zc, area, "
            "normx/normy/normz). "
            "Signal: 'ERROR: Compute reduce compute does not calculate a "
            "per-grid array (../compute_reduce.cpp:232)' (line 236 in 27Aug2026) when you bracket a "
            "one-attribute property/grid, 'ERROR: Compute reduce compute does "
            "not calculate a per-grid vector (../compute_reduce.cpp:229)' (line 233 in 27Aug2026) when "
            "you do not bracket a multi-attribute one or a compute grid, "
            "'ERROR: Invalid keyword in compute property/grid command "
            "(../compute_property_grid.cpp:84)' for a flow quantity, and "
            "'ERROR: Invalid compute property/grid field for 2d simulation "
            "(../compute_property_grid.cpp:54)' for zlo/zhi/zc in 2d. "
            "(Verified 2026-08-07)",

            "[Output] 'compute reduce' is the ONLY bridge from a per-grid, "
            "per-surf or per-particle compute to a number in the stats table; "
            "stats_style accepts a compute only if it produces a global "
            "scalar. So 'compute ke/particle' — the per-particle kinetic "
            "energy, and the natural input to a velocity-distribution check — "
            "cannot be printed directly and has to be reduced (min, max, sum, "
            "ave) or histogrammed. 'replace <col1> <col2>' needs min or max "
            "mode and two DIFFERENT input columns, so it cannot be used on a "
            "single-value reduce. "
            "Signal: 'ERROR: Stats compute does not compute scalar "
            "(../stats.cpp:678)' (line 787 in 27Aug2026) when a per-particle or per-surf compute is "
            "named in stats_style, and 'ERROR: Illegal compute reduce command "
            "(../compute_reduce.cpp:168)' (line 170 in 27Aug2026) for a replace pair that names one "
            "column or a column past the end. (Verified 2026-08-07)",

],
    },
}

GENERATORS = {
    "rarefied_flow_box_2d": _free_molecular_box_2d,
    "rarefied_flow_2d": _free_molecular_box_2d,
    "rarefied_flow_free_molecular_2d": _free_molecular_box_2d,
    "rarefied_flow_channel_2d": _fourier_channel_2d,
}
