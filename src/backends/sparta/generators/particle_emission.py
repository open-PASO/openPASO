"""Particle injection from box faces or surfaces (inflow boundaries)."""

from ._common import output_idioms


def _emit_channel_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Continuous inflow through the xlo face of an otherwise open 2d box.
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", 20)
    lx = params.get("lx", 1.0e-2)
    ly = params.get("ly", 1.0e-2)
    vstream = params.get("vstream", 1000.0)
    temp = params.get("temp", 300.0)
    nrho = params.get("nrho", 1.0e20)
    fnum = params.get("fnum", 1.0e13)
    dt = params.get("dt", 1.0e-7)
    nsteps = params.get("nsteps", 1000)
    return f"""\
# Continuous inflow through a box face - SPARTA DSMC
# The emit face must NOT be periodic; 'o' (outflow) or a surface face is fine.
seed             12345
dimension        2
global           gridcut 0.0 nrho {nrho} fnum {fnum}
boundary         o o p
create_box       0 {lx} 0 {ly} -0.5 0.5
create_grid      {nx} {ny} 1
species          air.species N2
# the emit rate follows THIS mixture: its nrho (falling back to global nrho),
# its vstream and its temp, times the face area, divided by fnum
mixture          air N2 vstream {vstream} 0.0 0.0 temp {temp}
collide          vss air air.vss
fix              in emit/face air xlo
timestep         {dt}
stats            100
stats_style      step np ncoll nexit
run              {nsteps}
"""


KNOWLEDGE = {
    "particle_emission": {
        "description": "Particle injection / emission from box faces or "
                       "surfaces — the DSMC inflow boundary condition",
        "spatial_dims": [2, 3],
        "key_commands": {
            "fix emit/face": "fix <ID> emit/face <mixID> <face...> [n <Np>] "
                             "[nevery <N>] [perspecies yes|no] [region <regID>] "
                             "[modulate v_<name>] [subsonic <P> <T|NULL>]",
            "fix emit/surf": "fix <ID> emit/surf <mixID> <group-ID> [n <Np>] "
                             "[normal yes|no] [nevery <N>] [perspecies ...]",
            "fix emit/face/file": "fix <ID> emit/face/file <mixID> <face> "
                                  "<filename> <section-ID> [frac <f>] "
                                  "[nevery <N>] [perspecies yes|no] "
                                  "[region <regID>] — spatially varying "
                                  "inflow read from one named section of "
                                  "the file",
            "mixture": "mixture <mixID> <species...> nrho <n> vstream <vx> "
                       "<vy> <vz> temp <T> — these set the emitted flux",
            "boundary": "the emit face must not be periodic ('p' is rejected; "
                        "so is ylo in an axisymmetric model); 'o', 'r' and "
                        "'s' faces are accepted",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "output_idioms": output_idioms("boundary tally idiom"),
        "pitfalls": [
            "[Setup] 'fix emit/face' cannot be attached to a PERIODIC face — "
            "particles already re-enter there, so an emit fix would "
            "double-count. The check happens when a run starts "
            "(FixEmitFace::init), not when the fix is defined. "
            "Signal: 'ERROR: Cannot use fix emit/face on periodic boundary "
            "(../fix_emit_face.cpp:182)' (SPARTA 24Sep2025; 27Aug2026 raises "
            "it from line 183).",

            "[Numerical] The emitted flux is set by the MIXTURE, not by "
            "'global nrho': an 'nrho' keyword on the emit mixture silently "
            "overrides the global value for every emitted and created "
            "particle. Two densities in one deck do not conflict-check. "
            "Signal: the domain fills to a steady Np far from nrho*V/fnum; "
            "grep every mixture line for 'nrho' before trusting an inflow "
            "density.",

            "[Physics] Emitting on ONE face of an otherwise outflow box does "
            "not hold the domain at the freestream: particles leave through "
            "the other faces faster than one face injects, so Np falls "
            "throughout the run with rc = 0 and no warning. This is the "
            "difference between an inflow boundary and an initial condition. "
            "Signal: NOT flatness of the Np column — an earlier wording said "
            "it 'decreases monotonically instead of levelling off' and that is "
            "measurably false. A seeded box emitting on xlo alone drops at "
            "first and then FLATTENS, holding at a fraction of the seeded "
            "count that depends on the case (about 87 % on this module's "
            "default template seeded with 'create_particles air n 0', about "
            "53 % at vstream 300); once flat it moves "
            "by a fraction of a percent per interval when there are enough "
            "particles to beat the sampling noise. The usual advice — run to "
            "steady state and check that Np has flattened — therefore PASSES on the "
            "broken deck. The discriminating test is a second run that emits "
            "on every inflow face ('fix in emit/face <mix> xlo ylo yhi'): "
            "that one holds much closer to the seeded count (within a few "
            "percent on the default template, about 92 % at vstream 300), "
            "while the "
            "one-face run plateaus clearly below it and Nexit stays nonzero on "
            "both. Either seed the domain with create_particles as well, or "
            "emit on every inflow face.",

            "[Physics] A mixture with vstream 0 still emits — the thermal "
            "flux through a face is nonzero even for a gas at rest — but the "
            "inflow rate is then set by the thermal speed alone and is several "
            "times lower than the drift flux you probably intended. The deck "
            "runs and the domain fills, only slowly. "
            "Signal: the Np column plateaus far below nrho*V/fnum for the "
            "domain; compare a run with the vstream you meant against one with "
            "vstream 0 — if they differ by a large factor, the emit mixture's "
            "vstream is what you got wrong.",

            "[Numerical] 'fix emit/face ... n <Np>' overrides the flux-derived "
            "count entirely and injects Np particles per insertion step per "
            "face regardless of density, area or velocity — the same override "
            "trap as 'create_particles n', and the steady-state particle count "
            "then scales linearly with your n. It also cannot be combined with "
            "the default per-species insertion. "
            "Signal: 'ERROR: Cannot use fix emit/face n > 0 with perspecies yes "
            "(../fix_emit_face.cpp:100)' if you just add n; once you add "
            "'perspecies no' it is silent, and the steady Np is proportional to "
            "n rather than to nrho.",

            "[Setup] 'fix emit/surf' emits from a SURFACE group, so the "
            "surfaces must already have been read and grouped; the group is "
            "checked by name when the fix is defined. Use 'normal yes' if you "
            "want the flux directed along the element normal rather than drawn "
            "from the mixture's streaming velocity. "
            "Signal: 'ERROR: Fix emit/surf group ID does not exist "
            "(../fix_emit_surf.cpp:61)'.",

            "[Setup] A 'region' is a SELECTOR, not geometry. It reflects "
            "nothing, blocks nothing and creates no surface: a region defined "
            "in a deck and never named by another command changes the run not "
            "at all — same particle count, same collision count, same exit "
            "count, step for step. Only the commands that take a region "
            "argument see it (create_particles, create_grid, adapt_grid, "
            "group <ID> grid|surf region, custom, the fix emit family, fix "
            "ave/histo, dump particle). To obstruct a "
            "flow you need read_surf plus a surf_collide model. Styles are "
            "block, cylinder, sphere, plane, union and intersect; block takes "
            "six bounds and accepts INF (there is no EDGE keyword: SPARTA "
            "24Sep2025 silently reads a non-numeric bound such as EDGE as "
            "0.0 and aborts with 'Illegal region block command' only if that "
            "makes lo > hi, while SPARTA 27Aug2026 refuses it with 'ERROR: "
            "Expected floating point parameter in input script or data "
            "file'); union "
            "and intersect take the "
            "COUNT of sub-regions first ('region u union 2 a b'); 'side out' "
            "inverts the selection. In a 2d run a sphere and a z-cylinder of "
            "the same radius select exactly the same cells. "
            "Signal: there is NO signal for an unused region — the run is "
            "identical to one without it, which is why this has to be checked "
            "by reading the deck. The typo cases do speak: 'ERROR: "
            "Unrecognized region style (../domain.cpp:471)' for a style that "
            "does not exist (line 471 on a development build of 26 Jun "
            "2026; 450 in SPARTA 24Sep2025, 470 in 27Aug2026), and "
            "'ERROR: Create_particles region does not "
            "exist (../create_particles.cpp:122)' for a name that was never "
            "defined. (Verified 2026-08-07)",

            "[Setup] Tallying 'etot' on a wall whose collision model DELETES "
            "the particle crashes the process outright — no ERROR line, no "
            "message, just a signal, with log.sparta ending mid-run. It is "
            "specific to the etot keyword: n, nwt, nflux, mflux, press, shx, "
            "ke, erot, evib and the force keywords all run cleanly on the same "
            "wall, and etot is fine on diffuse, specular and adiabatic. The "
            "mechanism is in the source: SurfCollideVanish::collide and "
            "SurfCollideTransparent::collide are the only two styles that "
            "leave the 'reaction' out-parameter UNASSIGNED (the argument is "
            "unnamed in their signatures), where every other style writes "
            "reaction = 0 first; ComputeSurf::surf_tally's ETOT branch then "
            "indexes surf->sr[isr] on that stale value with no reaction model "
            "loaded. If you need the energy carried away by a vanishing "
            "stream, tally ke, erot and evib separately and add them. "
            "Signal: the run returns a negative status (SIGSEGV) with an empty "
            "stderr and no 'ERROR' string anywhere; under a debugger the "
            "faulting frame is ComputeSurf::surf_tally (src/compute_surf.cpp:"
            "263) called from Update::move (src/update.cpp:364) (line 573 in 27Aug2026) — those are "
            "names gdb reconstructs from the symbol table, both inside "
            "namespace SPARTA_NS, and SPARTA itself never prints them. "
            "A driver that only checks for the "
            "string 'ERROR' will report this run as clean. "
            "(Verified 2026-08-07)",
        ],
    },
}

GENERATORS = {
    "particle_emission_channel_2d": _emit_channel_2d,
    "particle_emission_2d": _emit_channel_2d,
}
