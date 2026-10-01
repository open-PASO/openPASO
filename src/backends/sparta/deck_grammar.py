"""The SPARTA input-script grammar — ONE copy, served wherever it is needed.

WHY THIS EXISTS. SPARTA is a BINARY driven by a command script, so the script is
this backend's run interface in exactly the way exports.json is the coupling
driver's: it cannot be guessed, and no run happens without it. openPASO elides "the
solve" by design — correct for a Python library, and for a file-driven code it
leaves the agent with nothing to start from.

MEASURED on the payload a single-code SPARTA agent received for `heat`: 1,084
characters in total, with no `fix` and no `run` — the two commands without which
the binary does nothing at all. The same shape of gap was measured for 4C (where
it cost three recorded runs of one coupled problem their whole attempt) and
for FEBio.

WHAT IS SERVED IS THE GRAMMAR, NOT A SOLVE. Every number is an arbitrary
placeholder; the agent still chooses its own domain, grid, species, mixture and
sampling. The command sequence and its ORDER are taken from this repo's own
executed SPARTA fixtures, and the skeleton runs as served on the installed
binary (tests/test_the_sparta_grammar_states_the_true_letters_and_the_surface_order.py).

THE BOUNDARY LETTERS AND THE SURFACE ORDER ARE STATED, BECAUSE THEY WERE WRONG
AND ABSENT. Measured on a coupled round: this text said "s specular, r
diffuse-reflect" (SPARTA's doc/boundary.txt: r is specular reflection, s makes
a face a surface that bound_modify must give a wall model), three runs repeated
it, and 22 runs died on the order of the surface commands, which the skeleton
did not show.

SPARTA IS A DSMC CODE, NOT A FEM CODE. It samples a stochastic estimate, so a
Monte-Carlo noise floor exists that no tolerance can go below — which is why a
residual that stops falling is not automatically a defect.
"""

SPARTA_INPUT_GRAMMAR = """\
THE SPARTA INPUT-SCRIPT GRAMMAR — you cannot guess it, and SPARTA is a BINARY
driven by this script. Every NUMBER and NAME is an ARBITRARY PLACEHOLDER; the
commands and THEIR ORDER are what is documented. Order matters: a command that
refers to something not yet defined is an error.

    seed            12345
    dimension       2
    global          gridcut 0.0 comm/sort yes
    boundary        p p p                 # x y z: the letters are below
    create_box      0 1e-3 0 1e-3 -0.5 0.5
    create_grid     10 10 1               # 2-D: the z count must be 1
    balance_grid    rcb cell

    species         ar.species Ar         # a species FILE must exist
    mixture         gas Ar temp 300.0
    global          nrho 1e21 fnum 1e11   # BEFORE create_particles
    collide         vss gas ar.vss        # a collision FILE must exist

    read_surf       body.surf group body  # after the grid, before custom and create_particles
    custom          surf create tb float 0 file tb.in 1 tb   # a value per element
    surf_collide    1 diffuse s_tb 1.0    # a wall model; s_tb reads that value
    surf_modify     body collide 1        # a group that exists, or all
    create_particles gas n 0              # n 0: honour nrho and fnum

    timestep        1e-8
    compute         gt temp
    fix             f ave/time 10 10 100 c_gt    # nevery nrepeat nfreq
    compute         s surf body gas etot         # the read_surf elements only
    fix             fs ave/surf body 1 100 100 c_s[1] ave one
    dump            d surf body 100 body.out id f_fs
    stats           100
    stats_style     step cpu np f_f
    run             300

  * `boundary` TAKES THREE ENTRIES, x y z. One letter sets both faces of that
    axis, two letters set the lower and the upper face: o outflow, p periodic
    (both faces), r SPECULAR reflection, s a surface whose wall model
    `bound_modify <face> collide <sc-ID>` names, a axisymmetric (lower y face
    only). In 2-D the z entry is p. 'Illegal boundary command' is a wrong count
    or an unknown letter; 'Box boundary not assigned a surf_collide ID' is an
    s face with no bound_modify.
  * A BOX FACE IS NOT A SURFACE. compute surf, fix ave/surf and dump surf see
    only the elements read_surf made ('Cannot use compute surf when surfs do
    not exist'). A face's flux is `compute <ID> boundary <mix> etot` through
    `fix <ID> ave/time ... mode vector`, row xlo=1 xhi=2 ylo=3 yhi=4, and one
    face takes one wall temperature.
  * THE SURFACE COMMANDS HAVE AN ORDER. read_surf needs the grid. custom surf
    needs read_surf first ('Cannot use custom surf command before surfaces are
    defined'). create_particles comes after read_surf ('Using read_surf
    particle none when particles exist' the other way round). surf_modify
    names `all`, a group read_surf made, or one `group <g> surf id <lo>:<hi>`
    made ('Surf_modify surface group is not defined'), and every element needs
    a wall model ('<N> surface elements not assigned to a collision model').
  * `global nrho` IS A NUMBER DENSITY, molecules per m^3 (not a mass density,
    not a count per cell), and `fnum` the molecules one simulated particle
    stands for. `create_particles <mix> n 0` makes nrho x V / fnum particles, V
    the flow volume (in 2-D the area: a 2-D cell has unit depth); `n N` makes N
    in all, not per cell, and the gas is then N x fnum / V. DSMC collides the
    particles of one grid cell with each other, so pick fnum = nrho x V /
    (particles per cell x grid cells) for several in each cell.
  * `create_box` TAKES THREE PAIRS EVEN IN 2-D. The z pair is still required;
    give it a unit-thickness slab such as -0.5 0.5.
  * `create_grid`'s THIRD COUNT MUST BE 1 in two dimensions, or the run is
    silently three-dimensional and every per-cell quantity changes meaning.
  * `species` AND `collide` READ FILES (ar.species, ar.vss ship with SPARTA).
    Copy them next to the script or give a path; a missing file aborts.
  * `fix ave/time nevery nrepeat nfreq` MUST SATISFY
        nevery * nrepeat <= nfreq   and   nfreq % nevery == 0
    or SPARTA rejects it. The averaging WINDOW is nevery*nrepeat steps ending at
    each nfreq multiple, so a quantity is an average over that window and not an
    instantaneous value — read a converged value from the LAST window, not from
    a single step. A dump of a fix writes on a multiple of that fix's nfreq
    ('Dump surf and fix not computed at compatible times').
  * NOTHING IS COMPUTED WITHOUT `run`, and nothing is SAMPLED without a `fix`
    or `compute`. A script that ends after `collide` exits cleanly having done
    nothing, which reads as success.
  * A SURFACE LINE HAS A FLOW SIDE: the side its normal points to. For a line
    from point p1 to point p2 the normal is N = (0,0,1) x (p2 - p1): walking
    from p1 to p2, the gas is on your left. A closed body in the gas is listed
    clockwise. A line listed the other way round puts the gas on its far side,
    and SPARTA does not warn.
  * NO CELL ON THE FLOW SIDE IS NOT AN ERROR. read_surf prints
        <a> <b> <c> = cells outside/inside/overlapping surfs
    and a = 0 means every grid cell is inside a body: create_particles then
    prints 'Created 0 particles', the run ends 'with 0 particles' and exits 0,
    and every surface tally is zero. Swap p1 and p2 of each line whose gas is
    on the wrong side.

  READING THE RESULT. SPARTA prints a stats table whose columns follow
  `stats_style`, and ends with
      Loop time of <t> on <p> procs for <n> steps with <m> particles
  which is the line that proves it ran and carries numbers to cross-check
  (steps, particle count). It also writes log.sparta in the working directory.

  IT IS A STOCHASTIC METHOD. Two runs with different `seed` values give
  different numbers, and the spread is physical, not a bug: there is a
  Monte-Carlo noise floor no tolerance can go below. Report a quantity with the
  sampling window that produced it, and if a residual stops falling at that
  floor say so rather than tightening the tolerance.
"""
