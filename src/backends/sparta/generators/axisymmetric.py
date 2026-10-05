"""2d axisymmetric DSMC: revolved geometry with radial particle weighting."""

from ._common import output_idioms


def _axisymmetric_body_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Axisymmetric flow past a body. The x axis is the axis of symmetry; the
    lower y face carries the axisymmetric boundary style and ylo must be 0.
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", 10)
    vstream = params.get("vstream", 3472.0)
    temp = params.get("temp", 300.0)
    nrho = params.get("nrho", 1.0e20)
    fnum = params.get("fnum", 1.0e17)
    dt = params.get("dt", 1.0e-6)
    nsteps = params.get("nsteps", 500)
    return f"""\
# 2d axisymmetric flow past a body - SPARTA DSMC
# The axisymmetric style 'a' is legal ONLY on the LOWER y face, so the y
# entry must be the TWO-letter form 'ar' (lower = axisymmetric, upper =
# reflect). A single 'a' would set BOTH y faces and is rejected.
seed             12345
dimension        2
global           gridcut 0.0
boundary         o ar p
# ylo MUST be exactly 0.0 for an axisymmetric model
create_box       -0.25 0.25 0.0 0.25 -0.5 0.5
create_grid      {nx} {ny} 1
# radial cell weighting keeps the particles-per-cell count roughly uniform
# in a revolved domain; it is REJECTED unless the model is axisymmetric
global           nrho {nrho} fnum {fnum} weight cell radius
species          air.species N2
mixture          air N2 vstream {vstream} 0.0 0.0 temp {temp}
fix              in emit/face air xlo
collide          vss air air.vss
read_surf        data.circle origin 5 5 0 trans -5 -5 0 scale 0.05 0.05 1 clip
surf_collide     wall specular
surf_modify      all collide wall
timestep         {dt}
stats            100
stats_style      step np ncoll nscoll nscheck
run              {nsteps}
"""


KNOWLEDGE = {
    "axisymmetric": {
        "description": "2d axisymmetric DSMC — the x axis is the symmetry "
                       "axis, cells are revolved annuli, radial weighting "
                       "available",
        "spatial_dims": [2],
        "key_commands": {
            "boundary": "boundary <x> <y> <z> — one letter sets BOTH faces of "
                        "that dimension, two letters set lower then upper. "
                        "Styles: o outflow, p periodic, r specular reflect, "
                        "a axisymmetric, s surface.",
            "create_box": "create_box xlo xhi 0.0 yhi zlo zhi — ylo must be "
                          "exactly 0.0",
            "global weight": "global weight cell radius — radial particle "
                             "weighting, axisymmetric models only",
            "create_particles": "create_particles <mixID> n 0 — cell volumes "
                                "are the revolved annulus volumes, so the "
                                "particle distribution is not uniform in y",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "output_idioms": output_idioms("per-grid tally idiom"),
        "pitfalls": [
            "[Syntax] Axisymmetry is a BOUNDARY style, not a global option. "
            "There is no 'global ... axisymmetric yes' — that form does not "
            "exist in the parser at all. "
            "Signal: 'ERROR: Illegal global command (../update.cpp:1805)' (line 2187 in 27Aug2026). "
            "Write 'boundary o ar p' instead.",

            "[Syntax] The single-letter form of a boundary entry sets BOTH "
            "faces of that dimension, and the axisymmetric style is legal only "
            "on the LOWER y face — so 'boundary p a p' is always an error. Use "
            "the two-letter form for y. "
            "Signal: 'ERROR: Only ylo boundary can be axi-symmetric "
            "(../domain.cpp:176)' (line 175 in 27Aug2026).",

            "[Setup] An axisymmetric model requires ylo exactly 0.0, a 2d "
            "simulation, and a non-periodic yhi — but the three checks fire at "
            "DIFFERENT times. ylo and the periodicity are caught at parse "
            "time; the 2d restriction is only checked at run setup, so a 3d "
            "deck carrying 'boundary o ar p' that never reaches a 'run' exits "
            "0 and a validation-only pass will not catch it. "
            "Signal: 'ERROR: Box ylo must be 0.0 for axi-symmetric model "
            "(../create_box.cpp:56)'; 'ERROR: Y cannot be periodic for "
            "axi-symmetric (../domain.cpp:181)' (line 180 in 27Aug2026); and, only once a run starts, "
            "'ERROR: Axi-symmetry only allowed for 2d simulation "
            "(../domain.cpp:80)' (line 79 in 27Aug2026).",

            "[Setup] 'global weight cell radius' has TWO ordering "
            "constraints, each with its own message: the model must already be "
            "axisymmetric (so the boundary command comes first) and the grid "
            "must already exist (so create_grid comes first too). "
            "Signal: 'ERROR: Cannot use weight cell radius unless "
            "axisymmetric (../grid.cpp:1997)' (line 2219 in 27Aug2026); 'ERROR: Cannot weight cells "
            "before grid is defined (../grid.cpp:1983)' (line 2205 in 27Aug2026).",

            "[Physics] In an axisymmetric run the cell volume grows linearly "
            "with radius, so with uniform weighting the cells near the axis "
            "hold very few particles and every per-cell diagnostic there is "
            "noise, while the outer cells are over-resolved. This is exactly "
            "the case radial weighting exists for. The size of the imbalance "
            "is set by the grid, not by a universal figure — occupancy tracks "
            "radius, so on an N-cell radial grid the outer/inner ratio is of "
            "order 2N, and an earlier wording's 'a hundred times' was this "
            "mechanism frozen at one grid. "
            "'global weight cell radius' IS NOT FREE, and this is the part "
            "that bites. For an axisymmetric run with 'global weight cell "
            "radius', grid.cpp sets the weight to the cell's MEAN RADIUS "
            "TIMES ITS X-WIDTH IN RUN LENGTH UNITS, "
            "0.5*(hi[1]+lo[1])*(hi[0]-lo[0]) (line 2028 in SPARTA 24Sep2025, "
            "2250 in 27Aug2026) — an absolute size, not a ratio against a "
            "reference radius. (The mean radius alone, grid.cpp:2032 in "
            "24Sep2025 and 2254 in 27Aug2026, is the 'radius/only' "
            "variant.) "
            "create_particles divides each cell's volume by that weight, so on "
            "any sub-metre domain every weight is far below 1 and the "
            "requested particle count is inflated by roughly 1/(r*dx). "
            "Measured on "
            "a millimetre box, create_particles does not complete at all. "
            "Measured on a metre box whose fnum cannot meet the inflated "
            "request, it does something worse than failing: it WARNS, creates "
            "nothing, exits 0 and runs to completion with an empty domain. "
            "Signal: tally 'compute <C> grid all all n' with BOTH a compute "
            "reduce in min mode and one in max mode, and use their RATIO, "
            "which "
            "is dimensionless and unaffected by fnum — without radial "
            "weighting it is more than tenfold on any usable grid, and with it "
            "the two come within a few percent. For the scale trap the only "
            "observable is 'WARNING: Created unexpected # of particles: 0 "
            "versus <N> (../create_particles.cpp:445)' followed by 'Created 0 "
            "particles' — no error, rc = 0, and a stats table of zeros. Check "
            "the created count against what you asked for; do not assume the "
            "weighting command is a no-op when the geometry is small.",

            "[Output] The end-of-run summary carries an 'Axisymm bad moves' "
            "counter that is specific to this mode: it counts particle moves "
            "the axisymmetric remap could not resolve. It is printed for every "
            "run, so a nonzero value is the thing to look for, not its "
            "presence. "
            "Signal: 'Axisymm bad moves = ' followed by anything other than 0 "
            "in the end-of-run block.",
        ],
    },
}

GENERATORS = {
    "axisymmetric_body_2d": _axisymmetric_body_2d,
    "axisymmetric_2d": _axisymmetric_body_2d,
}
