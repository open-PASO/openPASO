"""SPARTA capability survey: what this build can actually construct, and what it costs.

WHAT THIS IS FOR. SPARTA is a DSMC code driven by an input deck, and its
capability surface is the set of commands and styles the BUILD registers --
which is not the set the manual documents, because styles are compiled in.
Before this module openPASO reached 27 of the 146 commands and styles this
build declares.

EVERY STYLE IS WRITTEN OUT AS A REAL DECK LINE. SPARTA aborts on the first
style it does not recognise, so a deck that reaches its final `run` has proved
every style in it exists in this build. That is why `run 0` is enough for the
construction phase and why the decks are not merely lists of names.

THREE DECKS, BECAUSE THE BUILD ITSELF SPLITS THEM THAT WAY
  box              - no surfaces: regions, grid/particle computes, fixes,
                     dumps, reactions, every surf_collide model, and the
                     scripting commands.
  surface          - explicit surfaces (data.circle), the surf computes and
                     fixes, and the surface file commands.
  implicit_surface - the distributed/implicit half, which is mutually
                     exclusive with the explicit half (see the pitfalls).

EACH DECK CARRIES A CHECK THAT IS NOT "IT RAN"
  box: a closed reflective box of a MONATOMIC gas with elastic collisions
       conserves particle count and total kinetic energy EXACTLY. Measured
       over 1000 steps with collisions actually occurring (Ncoll, the per-step
       count, is 10-11 at steps 500 and 1000; about 9900 collisions over the
       run): Np and sum(KE) both exactly constant, to every printed digit, at steps
       0, 500 and 1000.
  surface / implicit_surface: data.circle is a 50-segment polygon inscribed
       in a circle of radius 3, so the total surface length SPARTA builds must
       equal 2*50*3*sin(pi/50); the run prints both and they agree to every digit. A surface
       read at the wrong scale, or silently clipped by the box, fails this and
       nothing else in a DSMC run would tell you.

All three decks run to completion with exit status 0 on this build.
"""


def _survey_box(params: dict) -> str:
    """FORMAT TEMPLATE -- an inventory, not a physics problem to copy."""
    return _BOX


def _survey_surface(params: dict) -> str:
    """FORMAT TEMPLATE -- explicit-surface inventory."""
    return _SURFACE


def _survey_implicit(params: dict) -> str:
    """FORMAT TEMPLATE -- implicit/distributed-surface inventory."""
    return _IMPLICIT


_BOX = r"""# SPARTA capability survey, box phase: what this build can actually construct.
#
# PHASE 1 runs a CLOSED reflective box with elastic VSS collisions and checks
# two conserved quantities. PHASE 2 then constructs every command and style
# this build declares and runs 0 steps, so construction is proved separately
# from physics. Nothing here is a physics problem to copy; it is an inventory.
#
# units and package MUST precede create_box -- SPARTA refuses both once the
# simulation box exists ("Units command after simulation box is defined").
units            si
echo             both

seed             12345
dimension        2
boundary         rr rr p
create_box       0 1e-4 0 1e-4 -0.5 0.5

# ---- the planted-failure switch: SURVEY_MUTATE=1 (read through getenv) ------
variable         mutenv getenv SURVEY_MUTATE
variable         mutv string 0${mutenv}
create_grid      20 20 1
species          ar.species Ar
mixture          gas Ar vstream 0.0 0.0 0.0 temp 273.15
global           nrho 7.07043e22 fnum 7.07043e11
create_particles gas n 0
collide          vss gas ar.vss
timestep         1e-9

# ---- PHASE 1: two conserved quantities, measured -------------------------
# A closed reflective box of a MONATOMIC gas with elastic collisions conserves
# particle count and total kinetic energy EXACTLY. Argon is not an arbitrary
# choice: with the 5-species air mixture the same deck loses 25% of its
# translational energy in 1000 steps, and a similar share of its kinetic
# temperature, because VSS collisions feed energy into rotational and vibrational
# modes. Nothing is wrong there and nothing warns you; translational energy is
# simply not the conserved quantity for a molecular gas. Both are printed at the start and the end.
# Collisions really occur (Ncoll > 0), so this is conservation under activity,
# not conservation because nothing happened. Adding a thermostat or a reacting
# surface breaks it, which is what makes it a check rather than a formality.
compute          ct temp
compute          ck ke/particle
compute          cke reduce sum c_ck
stats            500
stats_style      step np ncoll c_ct c_cke
variable         np0 equal np
variable         ke0 equal c_cke
run              0
variable         np_ref equal ${np0}
variable         ke_ref equal ${ke0}
# the planted failure: a thermostat running through the conservation phase
# breaks the energy verdict and leaves the particle-count verdict standing
if               "${mutv} == 01" then "fix fmut temp/rescale 100 273.15 273.15" "print '[MUTATED: a thermostat runs during the conservation phase -- FAIL on the energy verdict is the control working]'"
run              1000
if               "${mutv} == 01" then "unfix fmut"
# ---- the verdicts, in the coverage harness's grammar --------------------
# (scripts/coverage_harness/definitions.py). The particle count is an exact
# identity; the total kinetic energy is a conservation law judged at
# roundoff, 1e-12 of its initial value.
variable         np_now equal np
variable         ke_now equal c_cke
variable         dnp equal abs(v_np_now-v_np_ref)
variable         dke equal abs(v_ke_now-v_ke_ref)
variable         ketol equal 1.0e-12*v_ke_ref
variable         f_np_ref format np_ref %.6e
variable         f_np_now format np_now %.6e
variable         f_ke_ref format ke_ref %.6e
variable         f_ke_now format ke_now %.6e
variable         f_ketol format ketol %.6e
if               "${dnp} <= 1e-10" then "print 'VERDICT sparta command:boundary exact_identity ref=${f_np_ref} got=${f_np_now} tol=1.000000e-10 PASS'" else "print 'VERDICT sparta command:boundary exact_identity ref=${f_np_ref} got=${f_np_now} tol=1.000000e-10 FAIL'"
if               "${dke} <= ${ketol}" then "print 'VERDICT sparta collide:vss conservation ref=${f_ke_ref} got=${f_ke_now} tol=${f_ketol} tol_from=roundoff_1e-12_of_KE0;elastic_VSS_collisions_in_a_specular_box_conserve_KE PASS'" else "print 'VERDICT sparta collide:vss conservation ref=${f_ke_ref} got=${f_ke_now} tol=${f_ketol} tol_from=roundoff_1e-12_of_KE0;elastic_VSS_collisions_in_a_specular_box_conserve_KE FAIL'"

# reset_timestep is refused once ANY time-dependent fix exists ("Cannot reset
# timestep with a time-dependent fix defined"), so it belongs here, before
# phase 2 defines them, not in the teardown.
reset_timestep   0

# ---- PHASE 2: construct every style this build declares -------------------
# regions
region           rblock block 0 5e-5 0 5e-5 INF INF
region           rcyl cylinder z 5e-5 5e-5 2e-5 INF INF
region           rsph sphere 5e-5 5e-5 0 2e-5
region           rplane plane 5e-5 5e-5 0 1 0 0
region           rinter intersect 2 rblock rcyl
region           runion union 2 rblock rcyl

# grid and particle computes
compute          cgrid grid all gas n nrho nfrac massrho u v w usq vsq wsq
compute          cbound boundary gas n press
compute          ccount count Ar
compute          cprop property/grid all id
compute          ceflux eflux/grid all gas heatx heaty heatz
compute          cpflux pflux/grid all gas momxx momyy momzz
compute          cson sonine/grid all gas a x 2
compute          cthermal thermal/grid all gas temp
compute          clambda lambda/grid c_cgrid[2] c_cthermal[1] lambda tau
compute          cdt dt/grid all 0.1 0.1 c_clambda[2] c_cthermal[1] c_cgrid[8] c_cgrid[9] c_cgrid[10]
compute          ctvib tvib/grid all gas
compute          cgcoll gas/collision/grid all gas
compute          cgctally gas/collision/tally all gas id1 type1

surf_react       srglobal global 0.1 0.1
compute          crbound react/boundary srglobal r:Ar

# every surface collision model this build registers -- these are DEFINED
# without surfaces present, which is legal and is how you check a build has them
surf_collide     scspec specular
surf_collide     scdiff diffuse 300.0 1.0
surf_collide     scadia adiabatic
surf_collide     scvanish vanish
surf_collide     sctrans transparent
surf_collide     sccll cll 300.0 0.5 0.5 0.5 0.5
surf_collide     scpiston piston 100.0
surf_collide     sctd td 300.0
surf_collide     scimp impulsive 300.0 softsphere 0.2 50 200 60 1.0 0.5

# custom per-particle attribute, then the fix that sets it
custom           particle create mytemp float 0
variable         vtemp particle 273.15
variable         vgrid grid 0.0
variable         vpart particle 0.0
variable         vinternal internal 1.0
variable         vequal equal 1.0

# fixes
fix              favegrid ave/grid all 10 5 50 c_cgrid[*]
fix              favetime ave/time 10 5 50 c_ct
fix              favehisto ave/histo 10 5 50 0 1e-18 20 c_ck mode vector
fix              favehistow ave/histo/weight 10 5 50 0 1e-18 20 c_ck c_ck mode vector
fix              fbalance balance 100 1.1 rcb part
fix              fgridcheck grid/check 100 warn
fix              fhalt halt 100 tlimit > 86400
fix              fprint print 100 "capability survey: phase 2 alive"
fix              ftemprescale temp/rescale 100 273.15 273.15
fix              ftempglobal temp/global/rescale 100 273.15 273.15 1.0
fix              femitface emit/face gas xlo
fix              fadapt adapt 100 all refine coarsen particle 50 20
fix              fcustom custom 100 particle set mytemp v_vtemp all NULL
fix              ffieldgrid field/grid vgrid vgrid NULL
fix              ffieldpart field/particle vpart vpart NULL
fix              fdtreset dt/reset 100 c_cdt 0.5 1
# fix controller takes the control variable as a BARE name, not v_name --
# passing v_vinternal gives "Variable name for fix controller does not exist"
fix              fcontroller controller 100 1.0 1.0 0.0 0.0 c_ct 273.15 vinternal

# dumps
dump             dpart particle all 1000 dump.particle id x y vx vy
dump             dgrid grid all 1000 dump.grid id
dump_modify      dpart pad 4

# scripting commands this build accepts
group            ggrid grid subset 10 1
print            "capability survey: definitions complete"
if               "1 > 0" then "print 'conditional accepted'"
variable         vindex index 1 2
label            surveytop
next             vindex
shell            echo capability-survey
stats_modify     flush yes
scale_particles  gas 1.0
balance_grid     rcb cell
restart          100000 tmp.restart
write_grid       survey.grid
write_restart    survey.restart

# construction is proved by reaching this line: SPARTA aborts on the first
# style it does not recognise, so 0 steps is enough.
run              0

# tear the physics-altering fixes down before anything else reads the state
unfix            ftemprescale
unfix            ftempglobal
unfix            femitface
undump           dpart
uncompute        ct
# `quit` is DELIBERATELY absent: see the pitfall below.
"""

_SURFACE = r"""# SPARTA capability survey, surface phase: everything that needs a surface.
#
# THE INDEPENDENT CHECK IS GEOMETRIC. data.circle is a 50-segment polygon
# inscribed in a circle of radius 3, so the total surface "area" (length, in
# 2d) SPARTA builds must equal 2*50*3*sin(pi/50) exactly; the run prints the
# measured total beside it. A surface read at the wrong scale, or silently clipped by the box,
# fails this and nothing else in a DSMC run would tell you.
#
# THIS VARIANT KEEPS SURFACES NON-DISTRIBUTED. The two modes are mutually
# exclusive in this build: create_isurf refuses non-distributed surfaces
# ("Create_isurf requires distributed explicit surfaces") while fix move/surf
# refuses distributed ones ("Cannot yet use fix move/surf with distributed
# surf elements"). The implicit_surface variant covers the distributed half.
seed             12345
dimension        2
global           gridcut 0.0 comm/sort yes
boundary         o r p
create_box       0 10 0 10 -0.5 0.5

# ---- the planted-failure switch: SURVEY_MUTATE=1 (read through getenv) ------
variable         mutenv getenv SURVEY_MUTATE
variable         mutv string 0${mutenv}
create_grid      20 20 1
global           nrho 1.0 fnum 0.001
species          air.species N O
mixture          air N O vstream 100.0 0 0
variable         sscale string 1.0
if               "${mutv} == 01" then "variable sscale string 1.0001" "print '[MUTATED: the surface is read at the wrong scale -- FAIL on the perimeter verdict is the control working]'"
read_surf        data.circle scale ${sscale} ${sscale} 1.0
collide          vss air air.vss
timestep         0.0001

# SPARTA refuses to run with any unassigned surface element ("50 surface
# elements not assigned to a collision model"), so the model and the
# surf_modify that binds it must precede the FIRST run, not merely exist.
surf_collide     scdiff diffuse 300.0 0.0
surf_react       srglobal global 0.1 0.1
surf_modify      all collide scdiff react srglobal

# ---- the geometric check ------------------------------------------------
compute          carea property/surf all area
compute          cperim reduce sum c_carea
stats            500
stats_style      step np nscoll c_cperim
run              0
# ---- the verdict: the polygon's perimeter, in the harness's grammar -----
variable         perim equal c_cperim
variable         perim_ref equal 2*50*3*sin(PI/50)
variable         dperim equal abs(v_perim-v_perim_ref)
variable         f_perim format perim %.6e
variable         f_perim_ref format perim_ref %.6e
if               "${dperim} <= 1e-10" then "print 'VERDICT sparta command:read_surf exact_identity ref=${f_perim_ref} got=${f_perim} tol=1.000000e-10 PASS'" else "print 'VERDICT sparta command:read_surf exact_identity ref=${f_perim_ref} got=${f_perim} tol=1.000000e-10 FAIL'"

# ---- a second reaction model this build registers -----------------------
surf_react       srprob prob air.surf

# a surface temperature that the run itself updates needs a CUSTOM per-surf
# attribute to live in; surf_collide reads it as s_<name>
custom           surf create tsurf float 0
compute          csurf surf all all n press ke
surf_collide     sctemp diffuse s_tsurf 0.0
fix              fsurftemp surf/temp all 100 c_csurf[1] 300.0 0.1 tsurf

# ---- surface computes ---------------------------------------------------
compute          cpropsurf property/surf all id area
compute          creactsurf react/surf all srglobal r:N
compute          csctally surf/collision/tally all air id/surf type
compute          csrtally surf/reaction/tally all air reaction
compute          cdistsurf distsurf/grid all all

# ---- surface fixes ------------------------------------------------------
fix              favesurf ave/surf all 10 5 50 c_csurf[*]
fix              femitsurf emit/surf air all
fix              fmovesurf move/surf all 1000 1000 trans 0.01 0 0

# ---- surface output and file commands ----------------------------------
dump             dsurf surf all 1000 dump.surf id
write_surf       survey_out.surf
group            gsurf surf all
move_surf        gsurf trans 0.0 0.0 0.0

run              0

remove_surf      gsurf
"""

_IMPLICIT = r"""# SPARTA capability survey, implicit-surface phase.
#
# WHY THIS IS A SEPARATE DECK. Explicit and implicit surfaces are not
# interchangeable in this build, and the split is not a style preference:
#   * create_isurf requires "global surfs explicit/distributed";
#   * fix move/surf REFUSES distributed surfaces
#     ("Cannot yet use fix move/surf with distributed surf elements");
#   * once surfaces are implicit, compute surf, compute property/surf,
#     compute react/surf, fix surf/temp and fix ave/surf refuse with
#     "Cannot use ... with implicit surfs", and fix emit/surf refuses with
#     "Fix emit/surf not allowed for implicit surfaces".
# So the surface variant covers the explicit half and this one the implicit
# half. Each refusal above was produced by running it, not read from a manual.
#
# THE INDEPENDENT CHECK runs BEFORE the conversion, while the surfaces are
# still explicit: data.circle is a 50-segment polygon inscribed in a circle of
# radius 3, so its total length must be 2*50*3*sin(pi/50).
seed             12345
dimension        2
global           gridcut 0.0 comm/sort yes
global           surfs explicit/distributed
boundary         o r p
create_box       0 10 0 10 -0.5 0.5

# ---- the planted-failure switch: SURVEY_MUTATE=1 (read through getenv) ------
variable         mutenv getenv SURVEY_MUTATE
variable         mutv string 0${mutenv}
create_grid      20 20 1
global           nrho 1.0 fnum 0.001
species          air.species N O
mixture          air N O vstream 100.0 0 0
variable         sscale string 1.0
if               "${mutv} == 01" then "variable sscale string 1.0001" "print '[MUTATED: the surface is read at the wrong scale -- FAIL on the perimeter verdict is the control working]'"
read_surf        data.circle scale ${sscale} ${sscale} 1.0
surf_collide     scdiff diffuse 300.0 0.0
surf_react       srglobal global 0.1 0.1
surf_modify      all collide scdiff react srglobal
collide          vss air air.vss
timestep         0.0001

compute          carea property/surf all area
compute          cperim reduce sum c_carea
stats            500
stats_style      step np c_cperim
run              0
# ---- the verdict: the polygon's perimeter, in the harness's grammar -----
variable         perim equal c_cperim
variable         perim_ref equal 2*50*3*sin(PI/50)
variable         dperim equal abs(v_perim-v_perim_ref)
variable         f_perim format perim %.6e
variable         f_perim_ref format perim_ref %.6e
if               "${dperim} <= 1e-10" then "print 'VERDICT sparta command:read_surf exact_identity ref=${f_perim_ref} got=${f_perim} tol=1.000000e-10 PASS'" else "print 'VERDICT sparta command:read_surf exact_identity ref=${f_perim_ref} got=${f_perim} tol=1.000000e-10 FAIL'"

# The surf computes must stop being referenced BEFORE the conversion, and the
# stats line is a reference too: leaving c_cperim in stats_style and then
# deleting the compute gives "Could not find stats compute ID".
stats_style      step np
uncompute        cperim
uncompute        carea

# ---- convert to implicit surfaces --------------------------------------
# create_isurf's second argument is the ID of an EXISTING fix ablate, not a
# new name; a name gives "Fix ID for create_isurf does not exist".
fix              fablate ablate all 0 0.0 random 0
create_isurf     all fablate 0.5 ave

# ---- what still works once the surfaces are implicit -------------------
compute          cisurf isurf/grid all all n
compute          cdistsurf distsurf/grid all all
compute          csctally surf/collision/tally all air id/surf type
compute          csrtally surf/reaction/tally all air reaction
surf_react       srprob prob air.surf
dump             dsurf surf all 1000 dump.surf id
write_surf       survey_implicit_out.surf
run              0
"""

GENERATORS = {
    "capability_survey_box": _survey_box,
    "capability_survey_surface": _survey_surface,
    "capability_survey_implicit_surface": _survey_implicit,
}

KNOWLEDGE = {
    "capability_survey": {
        "description":
            "Inventory of the commands and styles THIS SPARTA build registers. "
            "Three decks (box, explicit surface, implicit surface) construct "
            "every style and run, each carrying a check independent of the run "
            "itself: exact conservation of particle count and kinetic energy "
            "in a closed monatomic box, and the analytic perimeter of the "
            "50-segment circle surface. Each check prints a VERDICT line in "
            "the coverage harness's grammar (reference, measured value, "
            "tolerance, PASS or FAIL), computed by the deck itself through "
            "equal- and format-style variables and an if/then/else print: "
            "the particle count and the 2*50*3*sin(pi/50) perimeter as exact "
            "identities, the kinetic energy as a conservation law at 1e-12 of "
            "its initial value. SURVEY_MUTATE=1, read through a getenv "
            "variable, keeps a thermostat running through the box's "
            "conservation phase and reads the surface at the wrong scale: "
            "the planted failures that show those verdicts can fail. Answers "
            "'what can this build do' by running it rather than by reading "
            "the manual, which documents styles that may not be compiled in.",
        "minimal_working_example": _BOX,
        "key_commands": {
            "compute": "compute <ID> <style> ... — the STYLE is the THIRD "
                       "token, after your own ID. Same for fix, region, dump, "
                       "surf_collide and surf_react; collide and react put the "
                       "style second.",
            "create_isurf": "create_isurf <group> <fix-ablate-ID> <thresh> ave "
                            "— the second argument names an EXISTING fix "
                            "ablate, not a new name for the surface.",
            "global surfs": "global surfs explicit | explicit/distributed | "
                            "implicit — required before read_surf; "
                            "explicit/distributed is what create_isurf needs "
                            "and what fix move/surf refuses.",
            "surf_modify": "surf_modify <group> collide <sc-ID> [react "
                           "<sr-ID>] — defining a surf_collide is not enough, "
                           "a surf_modify must bind it before the first run.",
            "units": "units si|cgs — refused once create_box has run; same for "
                     "package. reset_timestep is refused once any "
                     "time-dependent fix exists.",
            "quit": "quit — exits with status 1 even on a fully successful "
                    "run. Do not use it; just end the deck.",
        },
    },
}


PITFALLS = [
    "[integration] The `quit` command makes SPARTA exit with status 1 even "
    "when the run succeeded completely. Measured on this build: the same deck "
    "returns 0 without `quit` and 1 with it, with no ERROR line either time. "
    "Signal: a nonzero exit status with a complete, error-free log and all "
    "expected output files written. Any harness that judges a run by its exit "
    "code will call a good run a failure. Just end the deck; do not use quit.",

    "[physics] Translational kinetic energy is conserved only for a "
    "MONATOMIC gas. In a closed reflective box with elastic VSS collisions, "
    "argon holds sum(KE) exactly constant over 1000 steps, to every printed "
    "digit, while the same deck with the 5-species air mixture loses about a "
    "quarter of its translational energy, and of its kinetic temperature, in "
    "the same 1000 steps. "
    "Nothing is wrong and nothing warns you: VSS collisions feed energy into "
    "rotational and vibrational modes, so translational energy is simply not "
    "the conserved quantity. Signal: a steadily falling `compute temp` in a "
    "closed box with no thermostat and no reactions.",

    "[setup] `units` and `package` are refused once the simulation "
    "box exists (\"Units command after simulation box is defined\"), and "
    "`reset_timestep` is refused once ANY time-dependent fix is defined "
    "(\"Cannot reset timestep with a time-dependent fix defined\"). Signal: "
    "an ERROR naming the command and the word 'after'. Put units and package "
    "at the very top, and reset_timestep before the fixes rather than in the "
    "teardown where it reads more naturally.",

    "[setup] A deck will not run while any surface element lacks a "
    "collision model: \"50 surface elements not assigned to a collision "
    "model\". Defining a surf_collide style is not enough -- a surf_modify "
    "must BIND it, and it must do so before the first run, not merely before "
    "the surfaces are used. Signal: the error names the element count, which "
    "is the whole surface, not a subset.",

    "[setup] Explicit and implicit surfaces are mutually exclusive "
    "in one deck. create_isurf requires `global surfs explicit/distributed`, "
    "but fix move/surf then refuses with \"Cannot yet use fix move/surf with "
    "distributed surf elements\"; and after conversion, compute surf, compute "
    "property/surf, compute react/surf, fix surf/temp and fix ave/surf "
    "refuse with \"Cannot use ... with implicit surfs\", and fix emit/surf "
    "refuses with \"Fix emit/surf not allowed for implicit surfaces\". Plan "
    "which half you need before writing the deck. Signal: a refusal naming "
    "'distributed', 'implicit surfs' or 'implicit surfaces' at the first run "
    "after conversion.",

    "[setup] create_isurf's second argument is the ID of an EXISTING "
    "`fix ablate`, not a new name for the implicit surface. Passing a name "
    "gives \"Fix ID for create_isurf does not exist\", which reads as if the "
    "surface were missing rather than the fix. Define `fix <id> ablate ...` "
    "first, then `create_isurf <group> <id> <thresh> ave`.",

    "[api] Deleting a compute that `stats_style` still names "
    "gives \"Could not find stats compute ID\" at the next run, not at the "
    "uncompute. A stats line is a live reference: reset stats_style first, "
    "then uncompute. Signal: the error arrives one command later than the "
    "mistake, naming stats rather than the compute you removed.",

    "[syntax] Several SPARTA commands take a variable by its BARE "
    "name where the rest of the deck uses the v_ prefix: `fix controller` and "
    "`fix field/grid` / `fix field/particle` all want `myvar`, not `v_myvar`, "
    "and passing the prefixed form gives \"Variable name for fix ... does not "
    "exist\" even though the variable is defined immediately above. Signal: a "
    "'does not exist' error for a name you can see in the deck.",
]
