"""Ground-truth probes for NGSolve.

NGSolve's core surface lives on the top-level ``ngsolve`` module: FE
spaces (``H1``/``HCurl``/``HDiv``/``L2`` and specialised variants),
``Mesh``, ``GridFunction``, ``BilinearForm`` / ``LinearForm``,
the ``Integrate`` / ``InnerProduct`` operator family, the
``VOL`` / ``BND`` markers, and so on.

Catalog templates use ``from ngsolve import *`` plus separate
imports from sibling packages -- typically ``netgen.csg``,
``netgen.geom2d``, and ``ngsolve.webgui``.  The catalog-consistency
check covers identifiers expected to live on the top-level
``ngsolve`` module (the watchlist in
``tests/test_catalog_consistency.py``).  Identifiers belonging to
``netgen.*`` or ``ngsolve.webgui`` are not in scope here -- they
need their own probes or separate watchlist entries if drift is a
concern.

This is the second instance of the Python-introspection probe
family (``skfem.py`` was the first; 4C and deal.II use the
source-grep family).  Returns ``None`` when NGSolve is not
importable so the test skips rather than fails in environments
without it.
"""

from __future__ import annotations


def public_attrs() -> set[str] | None:
    """Public attribute names on the top-level ``ngsolve`` module.

    Returns ``None`` if NGSolve is not installed.  The returned set
    is the full public surface (~200 entries).  The catalog-consistency
    test checks against it only the names on its watchlist
    (``_NGSOLVE_CORE_IDENTIFIERS``) that the templates use, so a
    watch-listed name NGSolve drops fails the test, while a typo of a
    name that is not on the watchlist (e.g. ``BilinearFrom`` for
    ``BilinearForm``) is never collected and passes.  Add a name to the
    watchlist to have it checked.
    """
    try:
        import ngsolve  # type: ignore
    except ImportError:
        return None
    return {name for name in dir(ngsolve) if not name.startswith("_")}
