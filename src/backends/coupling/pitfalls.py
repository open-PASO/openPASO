"""Coupling failure modes as indexed corpus entries.

WHY THIS FILE EXISTS
--------------------
The coupling knowledge openPASO serves is ~145 kB of prose across ten payloads,
and none of it carried a `[Category]` tag or a Signal clause. Retrieval is
built on exactly those two fields — `knowledge(topic='pitfalls', solver=...,
signal='<paste the error>')` matches the Signal clause and `category=` filters
the tag — so an agent whose coupling had just failed could not look its symptom
up. At the moment it most needed the knowledge, symptom lookup returned
nothing. The flagship capability was the only surface in the tree with that
hole.

WHAT WAS AND WAS NOT CUT INTO FIELDS
------------------------------------
Most of that prose is legitimate reference material — the participant contract,
the imports/exports shapes, one complete runnable participant script per
backend, the role/side capability table — and it stays prose. Measured over the
served payloads, 120,233 chars are reference and 24,884 chars describe a
FAILURE MODE: something that goes wrong, what it looks like, and what to do.
Only that second 17% belongs in this format. Cutting a runnable script into
Signal fields would destroy it and help nobody.

THE FORMAT, AND THE COMPOUND TAG
--------------------------------
Each entry is ONE string, in three parts: the compound tag, then the fact and
what to do about it in prose, then a clause introduced by the word Signal and a
colon saying what you observe when it bites, then the provenance in
parentheses. This docstring writes that word without its colon on purpose — the
freeze-criterion ledger counts every string constant here that carries the
literal marker, so a prose mention of it would inflate the denominator by one
claim that does not exist.

`src/backends/_cross.py` established the compound form `[Cross-Backend][Units]`
and the parser reads the LAST tag as the axis and everything before it as the
namespace, so `[Coupling][Numerical]` filters under `category='Numerical'` while
still declaring which corpus it came from.

THE SIGNAL IS THE WHOLE POINT, SO IT IS NEVER SHARED
----------------------------------------------------
Measured elsewhere in this tree: one backend has 63 entries sharing a single
generic "solver reports 'Convergence is not achieved'" string. A symptom query
returns all 63 and no differential, so every one of them is unfindable in
practice despite being present. The Signal has to be what distinguishes THIS
pitfall from its neighbours. Every clause below therefore names the specific
message, the specific field of the returned JSON, or the specific probe that
fires — and where the honest answer is that NOTHING fires, it says so.

THE SILENT ONES ARE THE VALUABLE ONES, AND THEY MUST NOT BE OVERSOLD
--------------------------------------------------------------------
A unit mismatch converges, balances, matches point by point, passes the
sensitivity probe and returns a confidently wrong number. Writing a Signal that
implies you would notice is worse than writing no entry, because the agent then
reads a clean run as evidence. For those modes the Signal states that every
internal check passes and names the only thing that separates them — a
monolithic re-solve of the same problem in one code.

NO MEASURED RESULT FROM OUR OWN RUNS IS IN HERE
-----------------------------------------------
No theta value from a sweep, no interface temperature, no flux, no iteration
count, no convergence rate, no error norm. Where a number appears it is either
a CRITERION (theta < 2/(1+rho)), a code constant (the driver keeps four
non-finite warnings before collapsing the rest), or a placeholder in a message
template. Message templates are written with <angle-bracket> placeholders for
the varying parts and verbatim text for the invariant parts, which is also what
makes them matchable: the invariant fragment is what an agent's paste and the
entry have in common.
"""
from __future__ import annotations


# ══════════════════════════════════════════════════════════════════════════
# THE TEN SILENT-WRONG MODES
#
# These converge cleanly and hand back a confidently wrong answer, which is
# why they are worth more than the modes that crash. Each is stated with the
# mechanism, the criterion, and what is ACTUALLY observable — including, for
# the ones where nothing is, that nothing is.
# ══════════════════════════════════════════════════════════════════════════

_SILENT_WRONG = [
    "[Coupling][BC] SIGN CONVENTION: the flux you APPLY and the flux you "
    "EXPORT are not the same number, and confusing them is the most common way "
    "to get a converged coupling that openPASO then refuses to verify. The "
    "receiving side applies the partner's flux value UNCHANGED — its outward "
    "normal points back at the sender, and that second sign flip cancels the "
    "first. The `normal_fluxes` array you EXPORT is taken with respect to YOUR "
    "OWN outward normal, and the two sides' outward normals are anti-parallel, "
    "so on a conservative interface the two exported fluxes CANCEL. Apply the "
    "same number; compute your own outward flux from your own assembled "
    "system and it will come out with the opposite sign. THEY CANCEL "
    "APPROXIMATELY, NOT EXACTLY, and that distinction is the whole evidence: "
    "two systems built on different meshes never cancel to the last bit, so a "
    "bit-exact zero jump means one array was negated into the other and the "
    "gate reads it as no proof of two solves. Measured on a live run that had "
    "otherwise driven an honest iteration: q_B = -q_A at every interface point, "
    "summing to exactly zero. Never construct one side's flux from the other's. "
    "Signal: THE TWO WAYS OF GETTING THIS WRONG HAVE OPPOSITE OBSERVABLES, and "
    "reading the loud one as the whole story is how the quiet one survives. "
    "(1) If only the EXPORTED diagnostic array is flipped, the coupling is "
    "physically right and gets failed for it — the conservation check reports "
    "`Interface flux NOT balanced` with the two net magnitudes equal to within "
    "the tolerance and their SIGNS AGREEING, and names it: 'The two magnitudes "
    "match but the signs agree, which is the signature of a SIGN-CONVENTION "
    "error rather than a conservation error'. Magnitudes that do NOT match are "
    "a different entry. (2) If the participant's WHOLE sign convention is "
    "flipped — the single `S = +1 or -1` factor that most participant scripts "
    "use for both the applied Neumann term and the exported flux — then NOTHING "
    "complains. The iteration converges normally, the interface flux balances "
    "to solver tolerance, the pointwise profile matches, the validation block "
    "comes back EMPTY, and the interface value is wrong by a large margin. "
    "Conservation is a property of the fixed point and holds for whatever "
    "problem the two scripts actually solved between them, so it certifies the "
    "wrong answer just as happily as the right one. Do not expect a growing "
    "residual or sign-alternating values from this: there are none. Only "
    "comparison against an independent answer — the closed form, or a "
    "monolithic re-solve of the un-split problem — separates case 2 from a "
    "correct run. "
    "(Verified 2026-08-06 against check_interface_balance in "
    "src/core/quality_checks.py, and against the tier-2 fixtures "
    "balance_check_both_directions and pair_ngsolve_skfem, which flip the "
    "participant sign factor and assert on the closed form because convergence "
    "and balance do not move)",

    "[Coupling][Units] UNIT MISMATCH between the two participants. The driver "
    "moves numbers and converts nothing — not units, not sign conventions, not "
    "field names. Two decks that each solve their own subdomain correctly in "
    "their own unit system produce a coupled answer in no unit system at all. "
    "Fix it at setup: pick ONE system before either deck is written, and state "
    "the unit of the exchanged quantity in both participant scripts. "
    "Signal: THIS IS THE ONE THAT PASSES EVERYTHING. When both exchanged "
    "quantities are scaled consistently on the two sides the iteration "
    "converges, the interface balance closes, the pointwise flux profile "
    "matches, the sensitivity probe reports a healthy response, the validation "
    "block comes back EMPTY, and the verdict is trustworthy — the answer is "
    "simply in the wrong units and no check in openPASO can see it. Do not expect "
    "a worse-looking run: conservation is a property of the fixed point, so the "
    "mismatched run can balance BETTER than the correct one, and reading a tight "
    "balance as reassurance is exactly the mistake. Only an independent answer "
    "separates "
    "it: re-solve the un-split problem in ONE code and compare, or estimate "
    "the exchanged number's magnitude by hand — a coupled answer that is off "
    "by a clean factor of 1000 or a million against a hand estimate is this "
    "and not a physics bug. The single visible case is a HALF-converted deck "
    "where only one of the two exchanged quantities was rescaled: the balance "
    "then fires with a ratio that is a clean 1e3, 1e6, 1e9, 60 or 3600, and "
    "the check says 'that is the signature of a UNIT MISMATCH between the two "
    "participants rather than a conservation error' instead of sending you "
    "after a physics bug. "
    "(Verified 2026-08-06 against _unit_ratio_hint in "
    "src/core/quality_checks.py, and against the tier-2 fixture "
    "unit_mismatch_survives_every_internal_check, which runs the same coupling "
    "twice changing one boundary temperature from kelvin to celsius and finds "
    "both runs converged, both validation blocks empty, both interfaces "
    "balanced, and only the monolithic check firing)",

    "[Coupling][Mesh] NON-MATCHING INTERFACE MESHES are legitimate and routine "
    "— they are the point of partitioning — but every exchange across them "
    "passes through an interpolation that is lossy and does not conserve the "
    "integrated quantity unless the mapping was built to, and a "
    "nearest-neighbour or plain linear map is not. The driver does NOT "
    "interpolate for you: each participant maps its partner's samples onto its "
    "own interface points, so the mapping error is yours to bound. Export "
    "`normal_fluxes` from BOTH sides — that is what turns the mapping loss into "
    "a checkable number instead of an assumption. "
    "Signal: nothing appears in the findings and the verdict is clean. The "
    "only trace is in the `not_checked` coverage list of the returned JSON: "
    "'conservation across a NON-MATCHING interface' followed by each side's "
    "point count, and either 'The interface flux balance is the only evidence "
    "here that it did conserve; the residual is silent about it' when both "
    "sides exported fluxes, or 'Neither side exported normal_fluxes, so "
    "NOTHING here checked whether the interpolation conserved' when they did "
    "not. An agent that reads `converged` and the findings and stops sees no "
    "symptom whatsoever. "
    "(Verified 2026-08-06 against check_interface_meshes in "
    "src/core/quality_checks.py)",

    "[Coupling][Mesh] THE TWO INTERFACES ARE NOT THE SAME SURFACE. Two "
    "subdomains that do not touch, or that touch at a coordinate one of them "
    "rounds differently, still exchange arrays happily: every number is mapped "
    "onto geometry it does not belong to and the iteration converges to a "
    "well-behaved solution of a different problem. Print each side's point "
    "count and its first and last interface coordinate on iteration 1 before "
    "trusting anything downstream. "
    "Signal: when the two coordinate sets are disjoint the geometry check "
    "names the axis and both spans — 'Interfaces do NOT overlap: along axis "
    "<d>' with each side's range and the size of the gap between them — and "
    "that is a finding, not a warning. When they DO overlap nothing fires, and "
    "the coverage list carries the limit instead: 'openPASO compared the "
    "coordinates the two participants REPORTED and they describe the same "
    "region of space. It cannot check that those coordinates are the surface "
    "each participant actually applied its boundary condition on'. A "
    "participant that reports correct coordinates for the wrong surface is "
    "invisible to every check at this level. "
    "(Verified 2026-08-06 against check_interfaces_are_the_same_surface in "
    "src/core/quality_checks.py)",

    "[Coupling][Validation] NO-OP PARTICIPANT: a script that never opens "
    "imports.json, opens it under the wrong partner name, or re-serves a "
    "cached result still exits 0 and writes a well-formed exports.json every "
    "iteration. Both participants then return their initial guess forever. "
    "Before writing any coupling, run each participant by hand twice with two "
    "different imports.json files and check the outputs differ. "
    "Signal: convergence at iteration 2 with a residual of exactly 0.0, and a "
    "`history` of length two whose first entry is NaN because iteration 1 has "
    "nothing to compare against. This is the most convincing wrong result the "
    "tool can produce — it reads as an instant, perfect convergence, which is "
    "what a fast linear problem is supposed to look like. Two checks name it: "
    "the per-iteration trace reports 'produced byte-identical output while the "
    "data handed to them CHANGED', and the post-convergence probe reports "
    "'is NOT COUPLED to its partner: perturbing every number handed to it "
    "moved its answer by a relative <S>'. Both need probe=True; with the probe "
    "off the coverage list says 'interface sensitivity: NOT probed' and the "
    "zero residual is the only thing you get. "
    "(Verified 2026-08-06 against check_participant_responsiveness and "
    "check_interface_sensitivity in src/core/quality_checks.py)",

    "[Coupling][Integration] A PARTICIPANT THAT CRASHES AFTER WRITING ITS "
    "OUTPUT. A solver that diverges commonly writes its last iterate and THEN "
    "aborts, so a complete, parseable exports.json can be the product of a "
    "failed solve. The driver deletes exports.json before every participant "
    "call and requires exit code 0 as well as the file, which is what stops a "
    "converged-looking coupling being built on a crashed participant. Write "
    "exports.json LAST in your own script, after the solve has succeeded. "
    "Signal: the run stops on a NONZERO EXIT STATUS with 'exited with code "
    "<rc> at iteration <k>; its exports.json is the output of a FAILED run and "
    "must not be coupled on', followed by the last 300 characters of that "
    "participant's stderr — read that tail, it names the solver's own error. "
    "Two "
    "neighbouring messages mean different things and must not be confused: "
    "'wrote no exports.json' with a return code means the process produced no "
    "file at all, and 'bad exports.json' followed by a JSON decode error means "
    "it produced a truncated or malformed one. Read which of the three you got "
    "before changing anything. "
    "(Verified 2026-08-06 against the participant loop in "
    "src/core/coupling_driver.py)",

    "[Coupling][Numerical] NaN OR INF IN THE EXCHANGED DATA does not stop the "
    "run. A non-finite entry in `values`, `normal_fluxes` or `coordinates` is "
    "recorded as a warning and the iteration continues to max_iter, so a "
    "coupling that has already gone numerically dead keeps burning solver "
    "time. The usual cause is a subdomain left with no essential boundary "
    "condition of its own, which is singular; the second is a solve that "
    "diverged internally without a non-zero exit code. "
    "Signal: an entry in the `warnings` list of the returned JSON reading "
    "'<participant>: non-finite export values/fluxes at iter <k>' — or the "
    "same line naming 'coordinates' instead, which is a different bug in the "
    "same script — while `error` stays empty and the run continues. The "
    "residual `history` turns to NaN from that iteration on, and the "
    "conservation check reports 'Interface flux balance could NOT be "
    "evaluated' naming a non-finite exchanged flux rather than an imbalance. "
    "Only the first four occurrences are kept; the rest collapse into a "
    "'further non-finite export warnings suppressed' count, so a low warning "
    "count does not mean it happened rarely. "
    "(Verified 2026-08-06 against the non-finite branch of "
    "src/core/coupling_driver.py and check_interface_balance)",

    "[Coupling][Numerical] NON-CONVERGENCE IS A FAILURE, NOT A RESULT — never "
    "report the numbers from a run that hit max_iter. Before touching the "
    "physics, check the budget: with a constant theta the driver's residual "
    "decays no faster than (1-theta) per iteration once the participants' raw "
    "outputs have settled, because the relaxed value is still catching up to "
    "the raw one, so the iteration count needed is about "
    "log(tol/d0)/log(1-theta) with d0 the initial relative mismatch. Evaluate "
    "that for YOUR theta and tol and give max_iter headroom: an "
    "under-budgeted run and a broken physics look identical from the outside. "
    "Signal: `converged` is false and `error` reads 'did not converge to "
    "tol=<tol> in <N> iters' with the last residual and the words 'result is "
    "NOT trustworthy'. The `history` is the discriminator and it has three "
    "distinct shapes: still falling steadily means the budget was short; flat "
    "or alternating in sign without growing means theta is too large; growing "
    "means divergence, which is a different entry with a different fix. "
    "(Verified 2026-08-06 against the max_iter exit of "
    "src/core/coupling_driver.py)",

    "[Coupling][Validation] ONE-WAY WHERE YOU MEANT TWO-WAY. A participant "
    "with an empty `imports_from` never sees its partner, so the residual "
    "falls to zero because nothing changes rather than because a coupled "
    "problem was solved. A deliberate one-way transfer is declared by asking "
    "for a single pass with max_iter=1; iterating a one-way graph is the "
    "confusion this catches. A MISSPELLED partner name is a different bug and "
    "is refused outright rather than silently dropped — dropping the edge "
    "would turn the run into exactly this failure. "
    "Signal: on a run that converged quickly with a small residual, the "
    "findings carry 'ONE-WAY coupling:' naming the isolated participants and "
    "the reason 'no information flows back to them and iterating to "
    "convergence is meaningless'. A misspelling gives a much earlier and "
    "louder failure, before any solver starts: 'imports_from names no such "
    "participant' listing the unknown name and the known participants, ending "
    "'Refusing to run'. If one-way IS what you meant, max_iter=1 moves the "
    "finding into the coverage list as a declared single pass. "
    "(Verified 2026-08-06 against check_coupling_directionality and the graph "
    "validation in src/core/coupling_driver.py)",

    "[Coupling][Numerical] THETA ABOVE THE STABILITY LIMIT — the wrong "
    "relaxation diverges rather than converging slowly. For a two-participant "
    "Dirichlet-Neumann split the iteration amplification is "
    "sqrt((1-theta)^2 + rho*theta^2), where rho is the interface conductance "
    "of the Dirichlet-side subdomain divided by that of the Neumann-side one, "
    "and interface conductance is the material coefficient divided by the "
    "distance from the interface to that subdomain's own outer boundary. So "
    "the criterion is theta < 2/(1+rho) and the fastest constant theta is "
    "theta = 1/(1+rho), exactly half the limit. WHAT CONVERGES IS AN INTERVAL, "
    "NOT A POINT — every theta below the limit converges, several of them at a "
    "typical unbalanced ratio, so finding one value that works does not mean it "
    "is the only one or the best one. The interval NARROWS as the split gets "
    "more unbalanced, which is why a coarse sweep can turn up a single working "
    "value: the spacing outgrew the interval, not the method. theta=1.0 is not a "
    "'no relaxation, exact for linear problems' setting on this driver: it "
    "sits at the limit for a balanced split and above it for any stiffer "
    "Dirichlet side. Halving theta always brings you back inside the interval; "
    "swapping which side is Dirichlet replaces rho by 1/rho and is usually a "
    "bigger win than any theta. "
    "Signal: the residual `history` GROWS from one iteration to the next and "
    "the exported interface values EXPLODE — they blow up by orders of "
    "magnitude — ending in non-finite export warnings and a 'did not converge "
    "to tol=' error. The values exploding while the residual grows is the "
    "signature; theta is above the stability limit for this conductance "
    "ratio, so halve it. The "
    "trap is that geometric growth starts small, so the first few iterations "
    "look exactly like a converging run and nothing warns you in advance — "
    "read the whole history, not its first entries. Distinguish from a short "
    "budget, where the history falls steadily, and from theta merely too large "
    "for speed, where it is flat or alternates in sign without growing. "
    "(Verified 2026-08-06 against the relaxation and residual arithmetic in "
    "src/core/coupling_driver.py, and against the tier-2 fixtures "
    "theta_stability_limit, theta_converging_set_matches_the_limit, "
    "theta_one_over_one_plus_rho_is_fastest and role_swap_beats_theta_tuning, "
    "which establish the limit as a derived criterion rather than a swept value)",

    "[Coupling][Validation] STALE DATA INSIDE A PARTICIPANT: one exported "
    "block frozen while the rest responds. A participant can echo an imported "
    "quantity straight back while holding the physics it is supposed to solve "
    "at a constant or a first-iteration field. The total export then responds "
    "fully to its imports and the frozen half is invisible in any global norm "
    "— the coupling converges around a quantity it never solved. "
    "Signal: on an otherwise clean converged run, the per-block sensitivity "
    "probe reports 'exports block(s) <names> that do NOT respond to its "
    "imports at all, while the rest of its export does', naming which of "
    "`values` or `normal_fluxes` is dead. It fires only when SOME block "
    "responds — a participant with every block frozen is the no-op case and "
    "produces the different 'is NOT COUPLED to its partner' finding instead. "
    "Neither fires without probe=True. "
    "(Verified 2026-08-06 against the per-block branch of "
    "check_interface_sensitivity in src/core/quality_checks.py)",
]


# ══════════════════════════════════════════════════════════════════════════
# THE PARTICIPANT CONTRACT — what the driver requires, and how each
# requirement announces itself when it is not met.
# ══════════════════════════════════════════════════════════════════════════

_CONTRACT = [
    "[Coupling][API] THE EXPORT LAYOUT MUST NOT CHANGE BETWEEN ITERATIONS. "
    "Export the same number of points, in the same order, every iteration, and "
    "keep `normal_fluxes` consistently present or consistently absent. The "
    "driver relaxes the export vector element by element, so a vector that "
    "changes length has no meaning to relax; the usual cause is an interface "
    "detection or remeshing step that depends on the imported data. "
    "Signal: the run stops with 'changed its export length from <N> to <M> at "
    "iteration <k>' and the explanation 'the exported interface must have the "
    "same layout every iteration or relaxation is meaningless'. It always "
    "fires at iteration 2 or later, never at iteration 1, because there is "
    "nothing to compare the first export against. "
    "(Verified 2026-08-06 against the layout guard in "
    "src/core/coupling_driver.py)",

    "[Coupling][API] AN EMPTY INTERFACE IS REFUSED, not converged. A "
    "well-formed exports.json carrying zero values stacks to an empty vector, "
    "whose relative change is zero, so before this was refused an empty "
    "exchange converged at iteration 2 with a residual of zero and every "
    "value-based check had nothing to look at and said nothing. An empty "
    "interface almost always means the participant's interface-selection "
    "predicate matched no entity — a coordinate tolerance too tight, or the "
    "wrong boundary id. "
    "Signal: the run stops with 'exported an EMPTY interface' naming the "
    "participant and the iteration, and the reason 'there is nothing to "
    "couple, and an empty exchange would otherwise report a residual of zero'. "
    "It fires at iteration 1, before any exchange has happened, which is how "
    "you tell it from a physics failure. "
    "(Verified 2026-08-06 against the empty-export guard in "
    "src/core/coupling_driver.py)",

    "[Coupling][Syntax] exports.json HAS THREE REQUIRED KEYS: `field_name`, "
    "`coordinates` and `values`. The VALUE of `field_name` is a free-form "
    "label that is never interpreted, but leaving the KEY out is a hard error. "
    "`normal_fluxes` is optional and `n_points` is ignored entirely. Nothing "
    "validates the SHAPES: a `values` list of a different length than "
    "`coordinates`, or a flat rather than nested `coordinates` list, is "
    "accepted and gives a wrong answer quietly, so check those in your own "
    "script. "
    "Signal: 'bad exports.json' naming the participant, followed by the "
    "underlying parse or key error. The same message covers three causes — "
    "malformed JSON, a file truncated by a crash mid-write, and a missing "
    "required key — so open the file itself and look rather than guessing "
    "from the message. A shape mismatch produces NO message at all. "
    "(Verified 2026-08-06 against InterfaceData.from_json and its call site in "
    "src/core/coupling_driver.py)",

    "[Coupling][API] imports.json IS EMPTY ON ITERATION 1. It is written "
    "before every participant call and is an empty object on the first "
    "iteration, because no partner has exported anything yet. Every "
    "participant script therefore needs a fallback initial value for whatever "
    "it imports — that fallback is a real modelling choice, not boilerplate, "
    "and a bad one costs iterations. "
    "Signal: the run dies on the FIRST iteration with 'wrote no exports.json' "
    "and a stderr tail ending in a KeyError or IndexError naming the partner, "
    "because the script indexed imports.json without a fallback. Always "
    "iteration 1: a script that survives iteration 1 has the fallback. "
    "(Verified 2026-08-06 against the imports write in "
    "src/core/coupling_driver.py)",

    "[Coupling][Integration] THE DRIVER STAGES NOTHING FOR YOU except files "
    "you list explicitly. It does not copy your participant script into the "
    "work directory — write the script there yourself and name it bare in "
    "`command` — and every mesh, species or config file your solver opens has "
    "to be there before the run starts. "
    "Signal: an iteration-1 failure with 'wrote no exports.json' whose stderr "
    "tail is an INTERPRETER-level error rather than a solver one: the "
    "interpreter reporting it cannot open the script file, with errno 2. A "
    "missing DATA file looks different — the script starts, and the solver's "
    "own file-not-found appears in the same stderr tail. Distinguishing the "
    "two tells you whether the script is in the wrong place or the data is. "
    "(Verified 2026-08-06 against the subprocess launch in "
    "src/core/coupling_driver.py)",

    "[Coupling][Input] A RELATIVE work_dir IS NOT REJECTED, and it does not "
    "resolve where you think. It is resolved against the SERVER PROCESS's own "
    "working directory, which the agent can neither see nor control, so the "
    "run happens somewhere you did not choose while the directory you prepared "
    "sits untouched. Always pass an absolute path. "
    "Signal: the failure looks like a missing script — 'wrote no exports.json' "
    "at iteration 1 with an interpreter error saying it cannot open the script "
    "file — but the discriminating observation is on YOUR disk: after the run, "
    "the work_dir you prepared contains no imports.json and no exports.json at "
    "all, because nothing ever ran in it. A path problem and a crashed solver "
    "give the same message; only the untouched directory separates them. "
    "(Verified 2026-08-06 against the participant construction in the couple "
    "tool in src/tools/consolidated.py, which calls mkdir on the path as given "
    "and does not test whether it is absolute)",

    "[Coupling][Integration] A LEFTOVER imports.json DEFEATS THE BY-HAND "
    "DEBUG RUN. The driver writes imports.json every iteration and never "
    "deletes it, so the file survives the run. Running a participant script by "
    "hand in its work directory is the first thing to do when a coupling "
    "fails, and it is supposed to exercise the iteration-1 fallback path — "
    "but with a leftover file present it silently consumes a previous "
    "attempt's data instead. Delete imports.json first, every time. "
    "Signal: a participant script run standalone produces a plausible answer "
    "although you supplied no partner data, and produces a DIFFERENT answer "
    "after you delete imports.json. This never fires inside a coupling, so the "
    "symptom appears only while debugging — which is exactly when you are "
    "trying to decide whether the failure is in the physics or in the "
    "handshake. "
    "(Verified 2026-08-06 against the imports/exports file handling in "
    "src/core/coupling_driver.py)",

    "[Coupling][Performance] THE TIMEOUT IS PER PARTICIPANT CALL, NOT PER RUN, "
    "and it defaults to an hour. Each coupling iteration launches a FRESH "
    "process per participant, so anything the script does at start-up — a "
    "JIT compile of a form, a mesh generation, an import of a heavy module — "
    "is paid on every iteration, not once. Keep the compiled or generated "
    "objects independent of the imported data so only the first iteration is "
    "slow, and set `timeout` per participant so a hung compiled code cannot "
    "stall the whole coupling. "
    "Signal: 'timed out after <N>s at iteration <k>' naming the participant, "
    "with 'the coupling was killed, no result'. If it fires at iteration 1 the "
    "start-up cost alone exceeds the budget; if it fires at a later iteration "
    "the solve itself is growing, which usually means the problem being handed "
    "over is getting harder — look at the residual history before raising the "
    "timeout. "
    "(Verified 2026-08-06 against the TimeoutExpired branch in "
    "src/core/coupling_driver.py)",

    "[Coupling][Numerical] THE DRIVER IS JACOBI, NOT GAUSS-SEIDEL, AND IT "
    "RELAXES EVERY PARTICIPANT. Within one iteration every participant reads "
    "the PREVIOUS iteration's exports — participant B does not see what A "
    "produced moments earlier in the same iteration — and each participant's "
    "own export is blended with its own previous export, so a two-participant "
    "loop applies relaxation twice per cycle: no theta makes this driver "
    "finish a two-code Dirichlet-Neumann split in one step, even for a linear "
    "problem. WHAT TO BUDGET is set by rho, the interface conductance (or "
    "stiffness) of the DIRICHLET-side subdomain over the Neumann-side one. "
    "Measured at the tool's default tol=1e-6 with accelerator='constant' and "
    "theta=1/(1+rho), a two-subdomain conduction split took 11 iterations at "
    "rho=1/10, 37 at rho=1, 66 at rho=2, 123 at rho=4, 299 at rho=10 and 3311 "
    "at rho=100 — roughly LINEAR in rho once the Dirichlet side is the "
    "stiffer one, and the same counts to the iteration over a fourfold "
    "INTERFACE refinement, so budget from rho and not from the mesh. Set "
    "max_iter=100 for rho<=1 and 400 for rho<=10, and size each participant's "
    "`timeout` for that many solves; the default max_iter=50 is already short "
    "at rho=2. Beyond rho=10 the budget is the wrong knob — theta=0.5 "
    "diverges from rho=4 up and the DEFAULT accelerator from rho=10 up — so "
    "swap which side is Dirichlet, which replaces rho by 1/rho and puts every "
    "ratio measured below 1 inside 25 iterations. "
    "Signal: reordering the participants changes nothing — swapping the order "
    "of the `participants` list produces a bit-identical residual history, "
    "which is the cheap way to confirm which driver you are on. And the "
    "residual keeps falling geometrically after the participants' "
    "RAW outputs have stopped changing, because the relaxed value is still "
    "catching up to the raw one — a decay that looks like slow physics and is "
    "actually bookkeeping. "
    "(Verified 2026-08-06 against run_coupling in "
    "src/core/coupling_driver.py)",

    "[Coupling][Output] THE EXPORTS THE TOOL RETURNS ARE THE RELAXED BLEND, "
    "not the last raw output of your solver. On a converged run the difference "
    "is below the tolerance and does not matter. On a failed run they are a "
    "mixture of two iterations that no participant ever computed, which is one "
    "more reason a non-converged run must not be reported as a result. "
    "Signal: on a NON-converged run the `exports` block of the returned JSON "
    "does not match the exports.json files left on disk in the work "
    "directories, and the mismatch is of the order of the last residual. On a "
    "converged run they agree. If you need the raw output your solver wrote, "
    "read the file on disk rather than the returned block. "
    "(Verified 2026-08-06 against the relaxation write-back in "
    "src/core/coupling_driver.py)",
]


# ══════════════════════════════════════════════════════════════════════════
# WHAT THE CHECKS CAN AND CANNOT SEE. Each entry here is a limit of the
# verification, stated as a failure mode, because a limit an agent does not
# know about is indistinguishable from a guarantee.
# ══════════════════════════════════════════════════════════════════════════

_CHECK_LIMITS = [
    "[Coupling][Validation] NO EXPORTED FLUX MEANS NO CONSERVATION EVIDENCE. "
    "`normal_fluxes` is optional, but if EITHER side omits it the interface "
    "conservation check cannot run at all. An empty findings list is then not "
    "the same thing as a check that looked and was happy. Export the flux from "
    "both sides whenever a flux exists — it is the only guard against a "
    "coupling that converges to a non-conservative answer. "
    "Signal: the findings are empty, the verdict is converged, and the "
    "`not_checked` coverage list carries 'at least one side exported no "
    "normal_fluxes, so only the total could have been compared, and here not "
    "even that'. The absence of a complaint is the symptom, which is why it "
    "has to be read out of the coverage list rather than the findings. "
    "(Verified 2026-08-06 against check_interface_flux_profile and "
    "interface-balance coverage in src/core/quality_checks.py)",

    "[Coupling][Validation] A BALANCED TOTAL WITH A WRONG DISTRIBUTION. The "
    "net interface balance is a single number, and a single number is easy to "
    "satisfy by accident: one side can pile a large flux onto one node and "
    "take it off the others, and the two totals still cancel exactly while the "
    "two sides disagree about the interface everywhere along it. The pointwise "
    "comparison is what catches that, and it can only run when both sides "
    "export the SAME interface points. "
    "Signal: 'Interface flux does NOT match POINT BY POINT' naming the worst "
    "node index and both sides' values there, with 'The TOTALS may still "
    "balance — a redistribution along the interface cancels in the sum'. When "
    "the two sides do NOT share interface points there is no finding at all "
    "and the coverage list says the flux could only be compared in total, "
    "which is the case where a wrong distribution stays invisible. "
    "(Verified 2026-08-06 against check_interface_flux_profile in "
    "src/core/quality_checks.py)",

    "[Coupling][Numerical] THE GLOBAL RESIDUAL CAN HIDE AN UNCONVERGED "
    "QUANTITY. The driver converges on ONE relative norm over every "
    "participant's stacked export vector. When the exchanged quantities live "
    "on different scales — forces against displacements in FSI, temperature "
    "against displacement in TSI — the large block sets the denominator and "
    "the small block can still be moving by a large fraction of ITSELF while "
    "the global number sits below tolerance. Converge each exchanged quantity "
    "in its own units, or scale the blocks before taking the norm. "
    "Signal: `converged` is true and the findings carry 'Global residual is "
    "NOT representative:' followed by the offending blocks with their own "
    "relative changes, naming the worst one to look at. The per-quantity "
    "breakdown is in `block_residuals` in the returned JSON; the single "
    "`residual` cannot show this and never will. "
    "(Verified 2026-08-06 against check_residual_blocks in "
    "src/core/quality_checks.py)",

    "[Coupling][Validation] THE SENSITIVITY PROBE HAS A FRONTIER, and knowing "
    "where it is decides whether a clean run means anything. A measured "
    "response above the floor establishes only that the export MOVES when the "
    "imports move — NOT that it moves by the right amount. A participant "
    "frozen at a badly wrong value that adds a token multiple of its import "
    "responds just enough to pass, and a genuinely stiff participant responds "
    "just as little, so no local measurement can separate the two. "
    "Signal: no finding at all. The run is converged, balanced, and stamped "
    "responsive, and the only trace is the coverage line 'interface "
    "sensitivity FRONTIER' with the floor it used and the sentence 'A "
    "participant frozen at a badly wrong value that adds a token multiple of "
    "its import responds just enough to pass'. Pass a monolithic re-solve if "
    "you need that distinction; nothing measured at the interface can make it. "
    "(Verified 2026-08-06 against check_interface_sensitivity in "
    "src/core/quality_checks.py)",

    "[Coupling][Validation] THE MONOLITHIC RE-SOLVE IS THE ONLY SILENT-WRONG "
    "DETECTOR. A unit mismatch, a wrongly applied interface sign, a "
    "participant that never reads its imports and a lossy mesh mapping all end "
    "in the same place: a coupled number that is clean, converged and wrong. "
    "Solving the same problem un-split in ONE code and comparing a scalar "
    "quantity of interest is the one check that needs no external benchmark "
    "and can see all four. It is worth a whole extra solve. "
    "Signal: the comparison reports 'coupled=<a> vs monolithic re-solve=<b>' "
    "with the relative difference and 'the coupled result is likely WRONG' "
    "when they disagree beyond the tolerance, and a separate message ending "
    "'the coupled result is not corroborated' when one of the two is "
    "non-finite. Without it the coverage list simply records that no "
    "independent answer was compared, and every silent-wrong mode above stays "
    "silent. "
    "(Verified 2026-08-06 against check_monolithic_consistency in "
    "src/core/quality_checks.py)",

    "[Coupling][Validation] A CRITIC-APPROVED FLAG IS NOT A CRITIC REVIEW. "
    "`critic_approved=True` on its own does nothing: openPASO looks the review up "
    "rather than believing the flag, and the review has to have been recorded "
    "for THIS exact set of arguments. Submit the review first, with the same "
    "coupling arguments as a JSON object, then couple. "
    "Signal: the run completes normally, `converged` is true, and "
    "`trustworthy_result` is false with the verification block saying no "
    "matching critic review is on record. Nothing about the physics is wrong "
    "and nothing in the numbers explains it — the failure is procedural, so "
    "look at the verification block rather than at the residual. "
    "(Verified 2026-08-06 against the critic gate wiring of the couple tool in "
    "src/tools/consolidated.py)",
]


# ══════════════════════════════════════════════════════════════════════════
# NEGATIVE CAPABILITY CLAIMS. An agent that tries one of these wastes a long
# run and, worse, builds the partner around a role that cannot exist. Every
# one was probed on the installed software rather than read off a docstring.
# ══════════════════════════════════════════════════════════════════════════

_CAPABILITY_LIMITS = [
    "[Coupling][BC] SPARTA HAS NO NATIVE FLUX BOUNDARY CONDITION, so it can "
    "only take the Dirichlet-type side: it imports a wall TEMPERATURE and "
    "exports the energy flux the gas deposits. Of the nine surf_collide styles "
    "the installed SPARTA registers — adiabatic, cll, diffuse, impulsive, "
    "piston, specular, td, transparent, vanish — the four that take a thermal "
    "datum all take a temperature, and none accepts a prescribed heat flux. A "
    "flux CAN be imposed INDIRECTLY: fix surf/temp converts a per-surface "
    "energy flux into a temperature through the gray-body Stefan-Boltzmann "
    "relation, taking the flux from any per-surf compute or fix, so an "
    "imported flux does reach it. Know what that is before using it — it "
    "prescribes the temperature that would RADIATE the imported flux, so it "
    "constrains the wall temperature rather than the gas-side flux, drags in "
    "an emissivity that has nothing to do with your coupling, and the flux "
    "SPARTA then reports is free to differ from the one you imposed. "
    "Signal: there is no error to match, because the deck cannot be written in "
    "the first place — the thermal surf_collide styles parse a temperature "
    "argument and there is no style keyword a flux can be given to. The cost "
    "lands at deck-writing time, after the partner has already been built "
    "around a Neumann SPARTA side, so check which side SPARTA can take BEFORE "
    "designing the coupling. Separately, DSMC output is a Monte-Carlo estimate "
    "whose sampling noise does not shrink as the coupling iterates, so the "
    "relative residual has a floor and a tol below it can never be met: the "
    "run ends as 'did not converge', which is honest rather than a bug. "
    "(Verified 2026-08-06 by enumerating the SurfCollideStyle registrations in "
    "the installed SPARTA source and reading the Stefan-Boltzmann conversion "
    "in fix_surf_temp.cpp)",

    "[Coupling][Integration] 4C CANNOT USE preCICE AT ALL. There is no preCICE "
    "entry point in 4C: it is not a configuration problem that can be fixed "
    "from the outside, because adding preCICE to 4C means writing and building "
    "an adapter into 4C's own source. Use the file-handshake couple driver "
    "instead, where 4C works as a Python wrapper around its own YAML deck and "
    "can take either side of a cross-code coupling. "
    "Signal: 4C does not support preCICE, and nothing fails at run time to "
    "tell you so, because there is nothing to run — the absence has to be "
    "established by probing the install, and an agent that plans a preCICE "
    "coupling around 4C loses the whole build before discovering it. The "
    "probes and their answers: the dynamic-library listing "
    "of the 4C binary names no preCICE library, its exported dynamic symbols "
    "contain no preCICE symbol, its printable strings contain no occurrence of "
    "the word, and the 4C source tree contains no file whose name or contents "
    "mention it. "
    "(Verified 2026-08-06 on the installed 4C binary and source tree: zero "
    "hits from the library listing, zero from the dynamic symbol table, zero "
    "from the printable strings, and zero matching files in the source tree)",

    "[Coupling][Integration] FEBio CANNOT USE preCICE AS INSTALLED, and it has "
    "no scripting API either. FEBio runs an XML deck to completion and exits; "
    "it does not expose a time loop, so there is nothing for a preCICE "
    "participant loop to drive even if the library were linked. A "
    "preCICE-enabled FEBio would have to be a compiled plugin using FEBio's "
    "own callback interface, which is a real route but is not built here — "
    "treat it as UNVERIFIED rather than supported. Use the file-handshake "
    "couple driver instead, with a wrapper that writes a complete deck, runs "
    "the binary and parses the log. "
    "Signal: FEBio does not support preCICE and ships no preCICE adapter, and "
    "again there is nothing to match at run time. The dynamic-library listing "
    "of the FEBio binary resolves only FEBio's own shared libraries, and "
    "neither the binary nor any of those libraries exports a preCICE symbol or "
    "contains the word among its printable strings; the FEBio source tree "
    "contains no file mentioning it. The cost of not knowing is a planned "
    "coupling that cannot be built. "
    "(Verified 2026-08-06 on the installed FEBio binary, each of its shared "
    "libraries and its source tree: zero hits in every probe)",

    "[Coupling][Physics] FEBio 4 HAS NO HEAT MODULE, so a conduction "
    "participant cannot be written in FEBio at all — the heat module was "
    "removed upstream and survives only as a plugin. The substitute for a "
    "thermal coupling demonstration is the exact linear analogue: a "
    "uniaxial-strain elastic bar, where displacement plays the role of "
    "temperature and the P-wave modulus the role of conductivity, which "
    "couples through the same Dirichlet-Neumann contract with a traction in "
    "place of the flux. "
    "Signal: an unknown module type does not produce an error message. The "
    "binary dies with SIGSEGV while reading the deck — the shell reports a "
    "segmentation fault and the exit status is 139 — which looks exactly like "
    "a corrupted input file and sends you to check the XML instead of the "
    "module name. The absence is also checkable before writing anything: the "
    "installed binary and its libraries contain no heat-module string at all. "
    "(Verified 2026-08-06 by running the installed febio4 on a minimal deck "
    "declaring a heat module and observing the segmentation fault)",

    "[Coupling][Physics] BOTH SIDES MUST SOLVE THE SAME PHYSICS. The driver "
    "moves numbers; it does not translate a temperature into a displacement. "
    "Coupling a heat code to a structural code is a thermo-structural problem "
    "and needs a real transfer relation written into the participant scripts — "
    "the exchange contract will not supply one and will not object to its "
    "absence. "
    "Signal: everything passes. The arrays are finite, the lengths agree, the "
    "balance closes — because balance is arithmetic on whatever numbers you "
    "exported, whatever they mean — and the iteration converges to a fixed "
    "point of two solvers exchanging numbers with no physical relation. No "
    "check in openPASO can see it, because nothing here knows what the exchanged "
    "quantity IS. The tell is at setup time, not in the output: read what each "
    "side actually applies its imported number as, and check the two are the "
    "same physical quantity in the same units. "
    "(Verified 2026-08-06 against the field-agnostic exchange in "
    "src/core/field_transfer.py and src/core/coupling_driver.py)",

    "[Coupling][Silent-Wrong] A TWO-WAY COUPLING WHOSE REVERSE DIRECTION IS "
    "INERT. In thermo-structural interaction the mechanical -> thermal "
    "direction is ONE term of the energy equation, T_ref*beta*d/dt tr(eps). "
    "Drop it, get its sign wrong, or leave the structural side with no route "
    "back, and what remains is a ONE-WAY coupling: temperature drives "
    "deformation and nothing returns. It converges faster and cleaner than the "
    "real thing. For a real metal the coupling parameter "
    "delta = T_ref*beta^2/(rho_c*(lam+2mu)) is ~1e-2, so the difference between "
    "one-way and two-way is a fraction of a percent of the answer — small "
    "enough to look like discretisation error and four orders of magnitude "
    "above any coupling tolerance. "
    "Signal: NOTHING fires. Convergence, per-block residuals, exit codes, "
    "responsiveness and the interface sensitivity probe are all satisfied by a "
    "one-way run, because each participant really is a function of its imports. "
    "The only test is to SUPPRESS the reverse direction on purpose — give the "
    "thermal participant `imports_from: []`, which `couple` then reports as "
    "ONE-WAY in `validation` — and check the answer MOVES, and moves by the "
    "amount the difference between a two-way and a one-way monolithic re-solve "
    "says it should. "
    "AND A MONOLITHIC COMPARISON ALONE IS NOT ENOUGH at a realistic coupling "
    "strength. Measured: with the reverse term's SIGN flipped at delta ~ 0.012, "
    "the coupling converged, both participants stayed responsive and sensitive, "
    "and the coupled answer sat 4.1e-03 from the monolithic re-solve against a "
    "5e-03 tolerance — it passed. Doubling an effect that is 0.15% of the "
    "answer stays inside the discretisation gap. What caught it was comparing "
    "the coupled two-way-MINUS-one-way difference against the same difference "
    "between two monolithic solves: that is a difference of differences, the "
    "discretisation error cancels, and the wrong sign showed up as 200% off. "
    "(Verified 2026-08-07 by the tsi_twoway_* fixtures under "
    "scripts/tier2_fixtures/coupling/, whose declared mutations were run: "
    "killing the term is caught by the responsiveness and sensitivity checks; "
    "flipping its sign is caught ONLY by the difference-of-differences control; "
    "moving the structural expansion coefficient 25% is caught by the "
    "monolithic comparison)",

    "[Coupling][Numerical] AITKEN DOES NOT CONVERGE ON A STRONGLY COUPLED TSI, "
    "and it is the default. A two-way thermoelastic coupling's composite "
    "fixed-point map is block-antidiagonal with product -delta, so its "
    "eigenvalues are +-i*sqrt(delta) — PURELY IMAGINARY. The residual turns by "
    "about a right angle each iteration instead of shrinking along a fixed "
    "direction, and Aitken extrapolates along a direction that does not exist. "
    "Constant relaxation at theta = 1/(1+delta) converges on the same problem. "
    "Signal: with accelerator='aitken' (the default) the run ends "
    "'did not converge to tol=... in N iters' with a residual that fell several "
    "orders and then stopped, while the same participants with "
    "accelerator='constant', theta=1/(1+delta) converge. At small delta Aitken "
    "is fine, so this only appears once the coupling is strong. "
    "(Verified 2026-08-07: at delta=1.25 aitken did not converge in 300 "
    "iterations from either starting theta while constant relaxation converged "
    "in 60-70; at delta=0.012 aitken converged in 14)",

    "[Coupling][Validation] EXCHANGING AN ABSOLUTE TEMPERATURE HIDES THE OTHER "
    "BLOCK. The driver's convergence test is a RELATIVE norm over the stacked "
    "export vector, so a quantity carrying a large constant offset — a "
    "temperature in kelvin — makes that norm small for free and makes its own "
    "block dominate the denominator. The partner's block, which in TSI is a "
    "volumetric strain five orders of magnitude smaller, can then still be "
    "moving while the global number reports convergence. Exchange the CHANGE "
    "from the reference temperature instead; it is also the only thing the "
    "constitutive law sees. "
    "Signal: `validation` carries 'Global residual is NOT representative: "
    "block(s) <name> are still changing by more than ...' naming the SMALL "
    "block, on a run whose global residual is orders of magnitude below tol. "
    "The same coupling exchanging kelvin and celsius reports residuals a factor "
    "of ~20 apart, which is the tell that the norm is measuring the offset. "
    "(Verified 2026-08-07: exporting absolute T the driver reported 6e-11 "
    "global while the strain block was still changing by 3e-09 per iteration; "
    "exporting T - T_ref brought the two blocks to within a factor of 1.2)",

    "[Coupling][Capability] 4C's TSI NODE MATCHING HAS AN ABSOLUTE 1e-3 "
    "TOLERANCE, so a millimetre-scale mesh cannot run at all. 4C builds the "
    "structure <-> thermo correspondence with a geometric octree whose default "
    "tolerance is 1e-3 in absolute units (Coupling::Adapter::Coupling::"
    "match_nodes). Below that element spacing, distinct nodes collapse into one "
    "match. Nothing in the deck is wrong; the geometry is simply too small. "
    "Signal: 'Did not get 1:1 correspondence. masternodes.size()=<N> "
    "(structure), coupling.size()=<M> (thermo)' with M < N, from "
    "4C_coupling_adapter.cpp, before the first time step. The message is about "
    "node COUNTS and the cause is geometric SCALE, so it does not move when you "
    "correct the mesh, the conditions or the physics. Re-pose the problem at "
    "metre scale (scale lengths by s and the time step by s^2 and the solution "
    "is unchanged), or coarsen until every element edge exceeds 1e-3. "
    "(Verified 2026-08-07: a 20 mm bar at 80 elements reported "
    "masternodes.size()=324, coupling.size()=320 and aborted; the same problem "
    "at 2 m ran at 80, 160, 320 and 640 elements)",

    "[Coupling][Verification] A PARTICIPANT SOLVING ON A SMALLER BODY THAN ITS "
    "PARTNER IS NOT DETECTED. The interface-identity check is a DISJOINTNESS "
    "test: it fires when the two exported point clouds do not overlap at all, "
    "which catches the two-different-surfaces mistake, and says nothing when "
    "one cloud is CONTAINED in the other. In a volume (field) coupling that is "
    "the likelier error — one side meshed to the wrong extent — and every "
    "exchange beyond the smaller body's edge then becomes an extrapolation. "
    "Signal: nothing. The interfaces 'describe the same region of space', the "
    "mesh-conformity note reports a non-matching interface as usual, and the "
    "iteration converges. Whether it changes the ANSWER depends on the physics: "
    "where the exchanged quantity is determined pointwise by the partner's "
    "field — tr(eps) = beta*theta/(lam+2mu) in uniaxial strain, for instance — "
    "even the monolithic comparison stays green, so a mesh-extent mistake can "
    "be genuinely harmless there and catastrophic elsewhere. Check the extents "
    "yourself against the geometry you meant to solve; nothing here does. "
    "(Established 2026-08-07 by shrinking one participant's body to 75% of its "
    "partner's as a declared mutation: every check survived, and the mutation "
    "is recorded as NON-DISCRIMINATING in "
    "scripts/tier2_fixtures/coupling/tsi_twoway_ngsolve_fenics/fixture.json "
    "rather than replaced quietly)",
]


# ══════════════════════════════════════════════════════════════════════════
# FLUID-STRUCTURE INTERACTION
#
# FSI is the coupling that most of the entries above were written for without
# naming it, plus three modes that belong to it alone: the traction sign, the
# added-mass instability, and the ALE mesh. Kept as their own group because an
# agent arriving with a diverging FSI should be able to ask for exactly this.
# ══════════════════════════════════════════════════════════════════════════

_FSI = [
    "[Coupling][BC] FSI TRACTION SIGN: which normal the exchanged traction is "
    "taken with respect to decides whether the wall bulges or collapses, and "
    "the derivation is short enough to do once and write down rather than "
    "re-derive on each side. Cauchy's t(n) = sigma.n is the traction exerted "
    "BY the material n points INTO, ON the material n points OUT OF. So with "
    "n_f the FLUID's outward normal on the interface, sigma_f . n_f is what the "
    "STRUCTURE does to the FLUID, and the load ON THE STRUCTURE is its "
    "negative: t = sigma_f . n_s with n_s = -n_f. Check it against the "
    "hydrostatic case before trusting any implementation: a fluid at pressure "
    "p > 0 has sigma_f = -p I, so t = -sigma_f . n_f = +p n_f, and the fluid "
    "pushes the wall AWAY from itself. Decide ONE convention for what the fluid "
    "participant puts in `values` — the recommendation is 'the load on the "
    "structure', so the structure applies it with NO further sign change — and "
    "say so in both scripts. "
    "Signal: a flip in ONE of the two places is loud: `Interface flux NOT "
    "balanced` with the two magnitudes equal and the SIGNS AGREEING. A flip in "
    "BOTH — the whole convention reversed — is SILENT, and has its own entry.",

    "[Coupling][Numerical] ADDED MASS: a partitioned FSI with an INCOMPRESSIBLE "
    "fluid can diverge for reasons that have nothing to do with your tolerance, "
    "and refining the time step makes it WORSE rather than better. The "
    "mechanism: incompressibility makes the interface pressure respond "
    "instantaneously to interface acceleration, so eliminating the fluid leaves "
    "an ADDED-MASS operator on the structure's interface equation. Its size "
    "grows with the fluid-to-structure density ratio and with the slenderness "
    "of the fluid domain (rho_f * L against rho_s * h), and it does not shrink "
    "with dt — the structure's own inertia enters as rho_s/dt^2 while the added "
    "mass enters at the same order, so a smaller dt does not buy stability. "
    "When the added mass exceeds the structural mass the UNRELAXED "
    "Dirichlet-Neumann iteration has an amplification factor above one and "
    "diverges no matter how many iterations you allow. Light structures in "
    "water, thin membranes, and haemodynamics are the standard cases. What to "
    "do, in order: under-relax hard (theta well below 1), then Aitken, then a "
    "quasi-Newton scheme (IQN-ILS through preCICE), then give up on the "
    "partitioned split and use a monolithic solver. Raising max_iter is the one "
    "thing that never helps. "
    "Signal: `did not converge to tol=<value> in <n> iters` with the residual "
    "history GROWING rather than stalling — check `history` in the returned "
    "JSON, a growing sequence is the added-mass signature and a flat one is a "
    "noise floor or a wrong tol. If the divergence is fast enough the fluid "
    "participant dies first, on an inverted ALE mesh; see the ALE entry.",

    "[Coupling][Verification] A ONE-WAY FSI PASSES EVERY CONSERVATION CHECK. If "
    "the fluid never applies the imported displacement — the mesh motion is "
    "switched off, the ALE lift is written but never added to the geometry, or "
    "the displacement is applied to a boundary that does not bound the flow — "
    "then the fluid solves the RIGID-WALL problem every iteration. That "
    "coupling converges in two or three iterations, the interface force "
    "balances to machine precision, the profiles match point by point, and the "
    "answer is the rigid-wall answer presented as FSI. Every force-side check "
    "is satisfied because the force side is genuinely correct; it is the "
    "kinematic side that is missing, and force checks do not look there. "
    "Signal: the ONE check that fires is the interface-sensitivity probe — "
    "`Participant <name> is NOT A FUNCTION of its imports`, with `signal` equal "
    "to `noise` in `interface_sensitivity`, because re-running the fluid on "
    "perturbed displacement returns a bit-identical answer. Do not run FSI with "
    "probe=False. The second, independent test is to compare the converged "
    "deflection against the same case with the mesh motion deliberately off: if "
    "the two agree, the coupling IS one-way whatever it is called.",

    "[Coupling][Verification] THE FORCE BALANCE CANNOT SEE A WHOLE-CONVENTION "
    "FLIP, AND NEITHER CAN A SHARED-SCRIPT REFERENCE. If the fluid's exported "
    "traction is reversed in BOTH the load the structure applies and the flux "
    "the conservation check sees — which is what a participant written with the "
    "opposite normal produces — the coupling converges, the interface force "
    "balances exactly, the profile matches point by point, and the structure "
    "deflects the wrong way. A reference re-solve does not help IF IT DRIVES "
    "THE SAME PARTICIPANT SCRIPTS: a Newton-Krylov or fixed-point re-solve of "
    "the coupled interface equation shares the flip and agrees with it to "
    "solver tolerance. What separates it is an answer computed OUTSIDE those "
    "scripts: a closed form on a configuration you can reason about (for a "
    "channel with the wall held rigid, plane Poiseuille gives both the normal "
    "and the tangential interface force), or the same case in an independent "
    "code's own FSI solver. "
    "Signal: NOTHING in the returned JSON. `validation` is empty, the verdict "
    "is VERIFIED, and the sign of the deflection is the only thing that is "
    "wrong. Check that the structure moves the way the physics says it should "
    "before reading any other number.",

    "[Coupling][Mesh] AN INVERTED ALE MESH HANGS RATHER THAN FAILS. When a "
    "partitioned FSI diverges, the interface displacement exceeds the fluid "
    "domain within a few iterations and the harmonic (or elastic) extension "
    "folds cells over. The flow solve that follows is on a mesh with negative "
    "cell volumes, where a Newton iteration typically neither converges nor "
    "raises: the participant sits inside its timeout and the coupling reads as "
    "a slow run rather than a divergence. Put an explicit check in the fluid "
    "participant right after moving the mesh — compare each cell's signed "
    "volume against ITS OWN undeformed value and exit non-zero if any changed "
    "sign or collapsed. Compare per cell, not against a global sign: a mesh "
    "generator need not orient every cell the same way, and a guard written "
    "against the median sign reports half the cells as inverted on a perfectly "
    "good mesh. "
    "Signal: without the guard, `participant <name> timed out after <n>s`, or "
    "no message at all until the whole coupling's deadline. With it, "
    "`participant <name> exited with code 1` and your own message naming the "
    "displacement that did it — which is the difference between a diagnosis and "
    "a mystery.",

    "[Coupling][Numerical] THE TRACTION RECOVERED FROM A STRUCTURE'S OWN STRESS "
    "FIELD IS NOT AN INTERFACE EQUILIBRIUM CHECK, on a bending body. It is "
    "tempting to have the structure export sigma_s . n_s computed from its "
    "displacement solution, as an INDEPENDENT statement that its internal "
    "stress balances the load. On a thin strip in bending it does not converge: "
    "the normal traction on the loaded face is a small difference of large "
    "bending stresses, so the recovery error swamps it, and where a Dirichlet "
    "boundary (a clamp) meets the loaded face the exact solution has a stress "
    "singularity — the pointwise traction there diverges under refinement by "
    "construction, and a nodal quadrature of it converges to the wrong number. "
    "Export the traction the structure RECEIVED instead and be clear about what "
    "that check then measures: the FORCE TRANSFER between two non-matching "
    "interface discretisations, which is the conservation question a "
    "partitioned FSI actually has to answer. "
    "Signal: the recovered traction's pointwise maximum GROWS as you refine "
    "while the applied one does not move, and its integral over the interface "
    "converges to a value that is not the applied net force. If you see that, "
    "the recovery is behaving correctly and the check is the thing that is "
    "wrong.",

    "[Coupling][Contract] EXCHANGE THE FSI INTERFACE ON THE REFERENCE "
    "CONFIGURATION. The interface is a material surface, so both participants "
    "should send and receive `coordinates` in the UNDEFORMED positions of their "
    "interface nodes and let the exchanged displacement carry the motion. If "
    "one side sends deformed coordinates while the other sends reference ones, "
    "each side's interpolation is chasing a parametrisation that moves with the "
    "answer. The same rule decides the fluid's interface velocity: steady, the "
    "wall is stationary in its DEFORMED position and the fluid sees u = 0 there "
    "with the mesh carrying the whole coupling; transient, the wall is moving "
    "and u = d/dt, and that kinematic term is where the structure's inertia "
    "reaches the fluid at all. "
    "Signal: NOTHING here detects a mixed parametrisation. "
    "`check_interfaces_are_the_same_surface` compares bounding boxes with a 5% "
    "tolerance and a small deformation stays well inside it. This is a setup "
    "rule to follow, not a symptom to look up.",
]


COUPLING_PITFALLS: dict[str, list[str]] = {
    "silent_wrong": _SILENT_WRONG,
    "participant_contract": _CONTRACT,
    "verification_limits": _CHECK_LIMITS,
    "capability_limits": _CAPABILITY_LIMITS,
    "fsi": _FSI,
}


def get_coupling_pitfalls(topic: str | None = None) -> dict[str, list[str]]:
    """Every coupling failure-mode entry, or the ones matching a topic.

    `topic` is matched case-insensitively against the group key AND against the
    entry text, so both `get_coupling_pitfalls('silent')` and
    `get_coupling_pitfalls('precice')` return something useful. A topic that
    matches nothing returns everything rather than an empty dict — an empty
    answer reads as "nothing is known", which here would be false.
    """
    if not topic:
        return dict(COUPLING_PITFALLS)
    t = topic.strip().lower()
    out: dict[str, list[str]] = {}
    for key, entries in COUPLING_PITFALLS.items():
        if t in key.lower():
            out[key] = list(entries)
            continue
        hits = [e for e in entries if t in e.lower()]
        if hits:
            out[key] = hits
    return out or dict(COUPLING_PITFALLS)


def coupling_failure_index() -> str:
    """These entries as one readable block, headed by the queries that reach
    them.

    NOT APPENDED TO ANY PAYLOAD, and it is worth saying so because it was, and
    the call site was lost when the coupling guide moved from
    src/tools/knowledge.py into src/tools/coupling_knowledge.py. Restoring it
    is the wrong repair: this block is 38 kB against a 35 kB guide, so
    `knowledge(topic='coupling')` would more than double and would then carry
    TWO symptom corpora — these entries and the 17-row failure table — for an
    agent that has one error message and wants one answer. The guide instead
    carries a cross-reference to the query below, and each corpus stays in one
    place. This function remains the way to read the whole set at once.
    """
    n = sum(len(v) for v in COUPLING_PITFALLS.values())
    lines = [
        "## Failure modes, indexed by symptom",
        "",
        f"The {n} entries below are the same failure modes this guide "
        "describes, written in the corpus format the retrieval layer indexes: "
        "a `[Coupling][Axis]` tag and a Signal clause naming what you "
        "actually observe. If a coupling has already failed, do not read them "
        "— paste what you saw:",
        "",
        "    knowledge(topic='pitfalls', solver='coupling', "
        "signal='<the message you got>')",
        "",
        "To browse a slice instead, `physics=` filters on the group name or on "
        "any word in the entries — 'silent_wrong', 'participant_contract', "
        "'verification_limits', 'capability_limits', or something like "
        "'precice' or 'theta':",
        "",
        "    knowledge(topic='pitfalls', solver='coupling', "
        "physics='silent_wrong')",
        "",
        "Axes in use: "
        + ", ".join(sorted({e.split("]")[1].lstrip("[")
                            for v in COUPLING_PITFALLS.values() for e in v}))
        + ".",
        "",
        f"THE FIRST GROUP IS THE SILENT ONES — "
        f"{len(COUPLING_PITFALLS['silent_wrong'])} of them. A unit mismatch, a "
        "wrong applied sign, a participant that never reads its imports, a "
        "lossy interface mapping and a stale exported block all converge, "
        "balance and come back trustworthy. Their Signal clauses say so rather "
        "than inventing a symptom you would not see — for those, the only "
        "separator is a monolithic re-solve of the same problem in one code.",
        "",
    ]
    for group, entries in COUPLING_PITFALLS.items():
        lines.append(f"### {group} ({len(entries)})")
        for e in entries:
            lines.append(f"- {e}")
        lines.append("")
    return "\n".join(lines)
