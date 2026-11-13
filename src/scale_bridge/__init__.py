"""Bridging scales: material data from an upstream producer into openPASO's continuum solvers.

WHY THIS EXISTS. A continuum model is only as good as the material data it receives, and the data usually comes
from another program run by another agent: elastic constants from quantum-mechanical calculations, for example. The
receiving side sees numbers. Checks that compare the calculations a producer happened to run cannot see a
calculation that never ran, so results whose required convergence calculations are missing pass as checked.

This package makes the handoff explicit and the check complete:

- ``card``: the handoff record, the values together with every calculation they rest on (``MaterialCard``);
- ``obligations``: a ledger that decides accept / reject / incomplete from a list of required calculations;
- ``rules``: per kind of data, the rule that derives that list from the producer's OWN settings, so a missing
  calculation counts as failed instead of being skipped;
- ``continuum``: turns accepted data into what a continuum model needs (e.g. a rotated plane-stress stiffness);
- ``providers``: the interface through which an upstream framework (e.g. GENIUS for Quantum ESPRESSO) produces a
  card and runs the calculations a rule still requires. openPASO does not depend on any provider.

The MCP tool ``material_handoff`` (tools/scale_bridge.py) exposes all of it to agents.
"""

from scale_bridge.card import Calculation, MaterialCard  # noqa: F401
from scale_bridge.obligations import Ledger  # noqa: F401
from scale_bridge.rules import check, plan, rule_for  # noqa: F401
