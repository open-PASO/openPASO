"""Ground-truth probes for scikit-fem.

scikit-fem ships every public element and mesh class on the top-level
``skfem`` module, so the source-of-truth check is straightforward Python
introspection: the catalog must not promise an ``Element*`` or ``Mesh*``
name that does not exist in the installed package.

The probes return ``None`` when scikit-fem is not importable so the test
suite can skip rather than fail in environments without it.

This module is the canonical example of the *Python-introspection* probe
family.  The same pattern transfers to FEniCSx/dolfinx, NGSolve, Kratos
and DUNE-fem — each exposes its catalog-relevant names as attributes on
a top-level package, and ``scripts/fingerprint_solvers.py`` already
captures most of those for drift-vs-prior-fingerprint comparisons.
"""

from __future__ import annotations

from typing import Set


def _classes_with_prefix(prefix: str) -> Set[str] | None:
    """Return the names of public classes or class-like callables (factory
    constructors) whose name starts with ``prefix`` on the ``skfem``
    module, or ``None`` if scikit-fem is not installed.

    scikit-fem ships some mesh names (notably ``MeshLine``) as factory
    instances (``MeshLineConstructor``) rather than the class itself —
    ``skfem.MeshLine()`` works, but ``inspect.isclass(skfem.MeshLine)``
    is False. The catalog correctly refers to ``MeshLine`` because that
    is the public API surface; this groundtruth filter accepts both
    classes AND callable instances so the catalog-consistency test
    does not flag legitimate factory references. (Audit 2026-06-01.)
    """
    try:
        import inspect

        import skfem  # type: ignore
    except ImportError:
        return None
    result: Set[str] = set()
    for name in dir(skfem):
        if not name.startswith(prefix):
            continue
        if name.startswith("_"):
            continue
        obj = getattr(skfem, name)
        if inspect.isclass(obj):
            result.add(name)
        elif callable(obj):
            # Accept factory-style constructors that produce a
            # mesh / element when called with no args. Avoids the
            # need for the catalog to switch between e.g. MeshLine
            # (the public alias) and MeshLine1 (the concrete class).
            result.add(name)
    return result


def element_classes() -> Set[str] | None:
    """Public ``Element*`` classes exposed by ``skfem`` (~80 in 12.0)."""
    return _classes_with_prefix("Element")


def mesh_classes() -> Set[str] | None:
    """Public ``Mesh*`` classes exposed by ``skfem`` (~20 in 12.0)."""
    return _classes_with_prefix("Mesh")
