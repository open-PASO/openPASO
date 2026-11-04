# Data from another framework

Some problems need input that a finite-element model cannot compute itself: the stiffness of a crystal,
for example, which comes from quantum-mechanical calculations. Another framework produces it, often run by
another AI agent, and openPASO receives numbers. The tool `material_handoff` decides whether those numbers may
be used.

## The problem

A check can only compare calculations that exist. Whether elastic constants are converged shows when the same
calculations are repeated with finer settings. If those finer calculations never ran, a check that compares the
calculations it finds sees nothing wrong, and unchecked numbers look like checked ones.

## How openPASO handles it

1. **The card.** The producer hands over the values together with every calculation they rest on: settings,
   applied strain, resulting stress, and whether the calculation finished. Call
   `material_handoff(action='schema')` for the layout.
2. **The rule.** From the producer's own settings, not from what the card happens to contain, openPASO derives
   which calculations are required. For elastic constants from plane-wave calculations (Quantum ESPRESSO and
   similar programs): the reference cell and an opposite pair of stretches and shears, repeated once with a
   25 % higher cutoff and a 30 % finer k-point grid, and, if the calculation uses smearing, once more with half
   the smearing and a 50 % finer grid. Each repetition may change C' and C44 by at most 2 % and the pressure by
   at most 0.1 GPa. The values on the card must also follow from the stresses within 1 %.
3. **The decision.** *Accept* when every required calculation finished and every change is within its limit.
   *Reject* when a calculation ended in an error, a result is not a number, or a limit is exceeded.
   *Incomplete* when a required calculation is missing or was stopped. A missing calculation never passes.
4. **The plan.** For an incomplete card, `action='plan'` lists the calculations still to run, with the exact
   settings each must use and an estimate of its cost. A repetition with looser or different settings does not
   count.
5. **The continuum data.** Only an accepted card becomes a stiffness for the continuum model
   (`action='continuum'`): the full stiffness and, for a thin sheet, the plane-stress stiffness with the crystal
   turned by a given angle.

## Producers

A producer is a separate package; openPASO does not depend on it. It registers itself under the entry-point
group `openpaso.material_providers` and implements two calls: `produce` (make a card) and `complete` (run the
calculations a plan lists and return the card with them added). Then `action='produce'` and
`action='complete'` reach it from openPASO, and the decision stays on openPASO's side: the producer computes,
openPASO decides what has to be computed and whether the result may pass.

GENIUS, an agentic framework that prepares and runs Quantum ESPRESSO workflows, is the first intended
producer. Its adapter lives with GENIUS.

## Adding other kinds of data

A new kind of data (another crystal symmetry, thermal conductivity, parameters fitted to atomistic data) is one
new rule module in `src/scale_bridge/rules/`: it names its kind, derives the required calculations from the
producer's settings, and says how they are compared. The decision, the plan, the producers and the tool stay as
they are.

## What the check does not cover

It tests whether the numbers are converged for the method the producer used. It does not test the method
itself (the functional, the pseudopotential, the crystal structure): those belong in the task. The limits
suit results that are graded at a few percent; a finer tolerance needs finer limits.
