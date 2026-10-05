"""Weakly-ionized (ambipolar) DSMC: electrons carried along with their ions."""

from ._common import output_idioms


def _ambipolar_circle_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for
    your specific problem.

    Hypersonic ionizing air over a cylinder with the ambipolar approximation:
    electrons are attached to their parent ion rather than moved on their own
    (much shorter) timescale.
    """
    vstream = params.get("vstream", 12500.0)
    temp = params.get("temp", 217.63)
    nrho = params.get("nrho", 2.6404e20)
    fnum = params.get("fnum", 1.0e15)
    t_wall = params.get("t_wall", 615.0)
    dt = params.get("dt", 1.0e-8)
    nsteps = params.get("nsteps", 1000)
    return f"""\
# Weakly ionized air over a cylinder, ambipolar approximation - SPARTA DSMC
seed             12345
dimension        2
boundary         o ro p
global           gridcut 0.01
create_box       -0.2 0.65 0.0 0.4 -0.5 0.5
create_grid      30 15 1
global           fnum {fnum}
# a mixture-level nrho OVERRIDES global nrho for everything drawn from it
mixture          species nrho {nrho} vstream {vstream} 0.0 0.0 temp {temp}
# every ion AND the electron must be declared for fix ambipolar to find them
species          air.species N2 O2 N O NO N2+ O2+ N+ O+ NO+ e
# the mixture used for inflow/creation must EXCLUDE the electron: ambipolar
# electrons are created with their ion, never injected independently
mixture          species copy noelectron
mixture          noelectron delete e
mixture          noelectron N2 frac 0.8
mixture          noelectron O2 frac 0.2
# Measured on this distribution, not assumed: 'circle.surf' appears in TWO
# example dirs (adjust_temp, surf_react_heatflux) but both copies are
# byte-identical (md5 5db86f1f991caaae4a9e693640cd027a), so basename
# resolution cannot pick a wrong one. 'data.circle' appears in TWELVE dirs of
# which ELEVEN are identical and exactly ONE differs (examples/ambi, 46584 B
# vs 1977 B) — and because the stager takes the alphabetically first match,
# examples/adapt wins and the ambi geometry is the one that loses. Use
# circle.surf here.
read_surf        circle.surf group 1
surf_collide     wall diffuse {t_wall} 1.0
surf_modify      1 collide wall
fix              ambi ambipolar e N+ N2+ NO+ O+ O2+
collide          vss species air.vss relax variable
# 'vibrate discrete' is safe HERE only because every species in air.species has
# vibdof <= 2: particle.cpp:177 (line 176 in 27Aug2026) aborts with 'Discrete vibrational info for
# species <X> not read in' for any vibdof > 2 species that has no 'vibfile' on
# its species line. Do not copy this line onto a CO2 deck.
collide_modify   vremax 1000 yes vibrate discrete rotate smooth
# WARNING, measured on this build: at these defaults the ambipolar machinery
# never fires within the deck's own run length. 1000 steps gives
# 'Collide occurs = 61926' and 'Reactions = 995' (neutral dissociation only),
# while the ion columns c_cnt[6] (N2+) and c_cnt[8] (N+) are EXACTLY 0 on every
# stats line and nsreact stays 0. So 'fix ambipolar' and 'collide_modify
# ambipolar yes' are declared but untested by this run. Raise the stream speed
# or the run length until an ion column becomes nonzero before believing any
# plasma result — and note that ambipolar_plasma's own pitfall says the electron
# count must track the total ion count, which cannot be checked while both are 0.
collide_modify   ambipolar yes
react            tce air.tce
create_particles noelectron n 0
fix              in emit/face noelectron xlo
compute          cnt count species
compute          tk temp
timestep         {dt}
stats            250
# ionisation is slow: expect the ion counts to stay at 0 for hundreds of steps
stats_style      step np nattempt ncoll nsreact c_tk c_cnt[6] c_cnt[8]
run              {nsteps}
"""


KNOWLEDGE = {
    "ambipolar_plasma": {
        "description": "Weakly ionized flow in the ambipolar approximation: "
                       "each electron is carried with its parent ion instead "
                       "of being moved on the electron timescale",
        "spatial_dims": [2, 3],
        "key_commands": {
            "fix ambipolar": "fix <ID> ambipolar <e-species> <ion-species...> "
                             "— every named species must already be declared",
            "collide_modify": "collide_modify ambipolar yes — required for the "
                              "collision routines to treat the attached "
                              "electrons",
            "mixture copy/delete": "mixture <old> copy <new> ; mixture <new> "
                                   "delete <species> — the standard way to "
                                   "build an electron-free inflow mixture "
                                   "(examples/ambi/in.ambi: 'mixture species "
                                   "copy noelectron' then 'mixture "
                                   "noelectron delete e')",
            "react": "react tce <file> — the ionisation channels live in the "
                     "reaction file, so ambipolar without react produces no "
                     "ions at all",
            "compute count": "compute <ID> count species — per-species counts; "
                             "the only practical way to watch ionisation",
        },
        "solver": "SPARTA DSMC; run: spa_serial -in <deck>",
        "output_idioms": output_idioms(),
        "pitfalls": [
            "[Setup] Every species named on the 'fix ambipolar' line — the "
            "electron and each ion — must already be declared by a 'species' "
            "command, and the check is by name, so a typo in an ion name is "
            "reported with the same TEXT as a genuinely missing species but "
            "from a different line: the electron argument is checked at "
            "fix_ambipolar.cpp:44 and each ion argument at fix_ambipolar.cpp:57. "
            "Do not build a guard on the line number — an earlier wording quoted "
            "only :44, which would miss every ion typo, the likelier mistake. "
            "SPARTA also checks the CHARGE of what you named, so pointing the "
            "electron argument at a neutral species is caught separately. "
            "Signal: 'ERROR: Fix ambipolar species does not exist' from "
            "(../fix_ambipolar.cpp:44) for the electron argument and "
            "(../fix_ambipolar.cpp:57) for an ion argument; 'ERROR: Fix "
            "ambipolar electron species has charge >= 0.0 "
            "(../fix_ambipolar.cpp:46)' and 'ERROR: Fix ambipolar ion species "
            "has charge <= 0.0 (../fix_ambipolar.cpp:59)' when the name exists "
            "but carries the wrong charge.",

            "[Setup] The mixture used for create_particles and for the inflow "
            "must NOT contain the electron species: ambipolar electrons are "
            "created together with their ion and never injected on their own. "
            "Build the inflow mixture with 'mixture <old> copy <new>' followed "
            "by 'mixture <new> delete e' — the copy target is the SECOND "
            "argument, which is the opposite of how it reads. An earlier wording "
            "gave it as 'mixture <new> copy <old>', which aborts with 'ERROR: "
            "New mixture copy mixture already exists (../mixture.cpp:439)' (line 443 in 27Aug2026) "
            "because the name being created already exists. The upstream deck "
            "examples/ambi/in.ambi is the reference: 'mixture species copy "
            "noelectron' then 'mixture noelectron delete e'. "
            "Signal: an electron count from 'compute <ID> count species' that "
            "is nonzero at step 0, before any ionising collision has occurred.",

            "[Physics] Ionisation is a high-threshold channel, so a correctly "
            "configured ambipolar deck reports zero ions for the first stats "
            "blocks before the first ion appears. Measured on upstream's own "
            "examples/ambi/in.ambi, the ion columns are identically 0 for the "
            "first two 100-step stats blocks and only reach four figures by "
            "step 1000. A zero count early in the run is not evidence of a "
            "broken setup. Do NOT wait for the electron count to come up as "
            "confirmation — under the ambipolar approximation it never does; "
            "see the entry on 'collide_modify ambipolar yes'. "
            "Signal: watch the ION per-species columns from 'compute count', "
            "not Nreact and not the electron column; declare the setup broken "
            "only if the ion counts are still identically 0 once Ncoll has "
            "accumulated over the whole run.",

            "[Setup] 'fix ambipolar' and 'collide_modify ambipolar yes' are "
            "NOT symmetric, and an earlier wording had this backwards in both "
            "halves. The fix WITHOUT the collide_modify is silent — rc = 0, no "
            "error, no warning — and physically incomplete, because the "
            "ambipolar collision path is only switched on by the collide_modify "
            "line. The reverse is loud: 'collide_modify ambipolar yes' with no "
            "'fix ambipolar' defined aborts with 'ERROR: Collision ambipolar "
            "without fix ambipolar (../collide.cpp:293)' (line 328 in 27Aug2026). "
            "Signal: for the silent direction, do NOT use the electron count "
            "the way an earlier wording said. It claimed that 'once ions "
            "appear, the electron count must track the total ion count' and "
            "that an electron column stuck at 0 means the collide_modify line "
            "is missing. That is inverted. Under the ambipolar approximation "
            "the electron is carried as a per-particle attribute of its ion "
            "and is never a particle, so a compute count over the electron "
            "species reports 0 on every stats line of a CORRECT run — "
            "measured on examples/ambi/in.ambi unmodified, with the ion total "
            "climbing past 1000 over the same run. Delete the collide_modify "
            "line and the electron column becomes nonzero and grows "
            "monotonically, because the ionisation products are then ordinary "
            "particles. So: electron column identically 0 while ions "
            "accumulate = the ambipolar path is ON; a growing electron column "
            "= the collide_modify line is MISSING.",

            "[Setup] 'surf_react <ID> prob <file>' only parses the file when "
            "the command is read. At the start of the first run, reactions "
            "whose species are not declared are dropped silently, so a file "
            "written for a different species set is accepted with no error. "
            "The remaining reactions are then checked so that each reactant "
            "species' probabilities sum to <= 1. "
            "Signal: 'ERROR: Surface reaction probability for a species > 1.0 "
            "(../surf_react_prob.cpp:287)' at run start (SPARTA 24Sep2025; "
            "27Aug2026 raises it from line 292) — the active "
            "reactions of one reactant species have probabilities summing to "
            "more than 1.",

"[Setup] An external body force needs THREE lines that agree, and "
            "getting two of them right leaves a run that is silently "
            "force-free. 'fix <ID> field/grid <ax> <ay> <az>' (or "
            "field/particle) names GRID-style variables for field/grid and "
            "PARTICLE-style variables for field/particle, written as BARE "
            "names with no 'v_' prefix, with NULL for an unused component. The "
            "fix on its own does nothing at all: the mover only consults it "
            "after 'global field grid <fix-ID> <Nevery>' or 'global field "
            "particle <fix-ID>' activates it, and that first argument is a FIX "
            "ID, not a number — upstream's own example names its fix '1', so "
            "the line reads 'global field grid 1 0' and copies as if 1 were a "
            "flag. Without the global line the deck runs to completion with "
            "the particle statistics bit-identical to a deck with no field. "
            "The simpler alternative is 'global field constant <magnitude> "
            "<fx> <fy> <fz>', which needs no fix and no variable. "
            "Signal: for the silent case there is NO signal — compare a "
            "temperature or mean kinetic energy against the same deck with the "
            "fix deleted and see whether anything moved. The loud cases are "
            "'ERROR: External field fix ID not found (../update.cpp:221)' (line 240 in 27Aug2026) when "
            "the global line names something that is not a fix, 'ERROR: "
            "Variable for fix field/grid is invalid style "
            "(../fix_field_grid.cpp:105)' for an equal-style variable where a "
            "grid-style one is required, and 'ERROR: Variable name for fix "
            "field/grid does not exist (../fix_field_grid.cpp:103)' when the "
            "name carries a 'v_' prefix. (Verified 2026-08-07)",
        ],
    },
}

GENERATORS = {
    "ambipolar_plasma_circle_2d": _ambipolar_circle_2d,
    "ambipolar_plasma_2d": _ambipolar_circle_2d,
}
