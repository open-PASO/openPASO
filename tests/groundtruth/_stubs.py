"""Per-backend probe stubs.

Each function below should be replaced with a real probe.  Until then they
return ``None`` so the test suite skips the corresponding check rather
than failing or silently passing.

The stub-import dance also keeps these names visible to readers browsing
the package layout: a missing probe is now a TODO item with a clearly
named placeholder, not an absence.
"""

from __future__ import annotations


# deal.II finite-element classes -- implemented in ``dealii.py`` (source-grep
# family).  Future deal.II probes (Dirichlet BC types, quadrature classes,
# Manifold types) can either extend ``dealii.py`` or follow its pattern in
# additional modules.


# FEniCSx / dolfinx -- implemented in ``fenics.py`` (Python introspection,
# checks dolfinx submodule attributes via importlib walks).  Test skips when
# dolfinx is not importable; pip wheels are not available on most platforms
# so the canonical install is conda.


# NGSolve -- implemented in ``ngsolve.py`` (Python introspection family).
# Currently exposes ``public_attrs()``; broader probes (specific FE-space
# subclasses, solver factories) can either extend ``ngsolve.py`` or follow
# its pattern in additional modules.


# scikit-fem -- implemented in ``skfem.py`` (Python introspection family).
# Use that module directly; the stub below is removed.


# Kratos applications -- implemented in ``kratos.py`` (source-enumeration of
# the upstream ``applications/`` directory; no Kratos install needed).
# Finer-grained probes (constitutive laws, solver strategies) require a
# Kratos install and would extend ``kratos.py`` or follow its pattern.


# DUNE-fem -- implemented in ``dune.py`` (Python introspection via importlib
# dotted-path walk, shared with the FEniCSx probe through ``_introspect``).
# Test skips when ``dune`` is not importable (it builds C++ deps from source).


# ── FEBio (XML schema) ────────────────────────────────────────────────────
def febio_material_types() -> None:
    return None
