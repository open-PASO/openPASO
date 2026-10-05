"""Ground-truth probes for FEniCSx / dolfinx.

FEniCSx exposes its API as a Python package tree rooted at
``dolfinx``: mesh constructors under ``dolfinx.mesh``, function-space
machinery under ``dolfinx.fem``, IO under ``dolfinx.io``, PETSc
integration under ``dolfinx.fem.petsc``, and so on.  The catalog
references these by their dotted path (``dolfinx.mesh.create_rectangle``,
``dolfinx.fem.functionspace``, ``dolfinx.io.XDMFFile``), so the
verification is a per-path attribute check.

Unlike NGSolve / scikit-fem, dolfinx is rarely pip-installable in a
clean way -- the canonical install is conda, with optional system
wheel availability for some platform/Python combinations.  All probes
here return ``None`` (or ``False`` for individual checks) when
dolfinx is not importable so the test skips cleanly in environments
that lack it.

The catalog references for FEniCSx already exist in
``scripts/fingerprint_solvers.py`` (``fingerprint_fenics()``); this
module exposes the underlying primitives so the catalog-consistency
test can reuse the same API map.
"""

from __future__ import annotations

from tests.groundtruth._introspect import is_importable, resolve_dotted_path


def has_attr(dotted_path: str) -> bool | None:
    """Return ``True`` / ``False`` if the dotted-path attribute can be
    resolved on the installed ``dolfinx`` package; return ``None``
    when ``dolfinx`` is not importable or the path is not rooted at
    ``dolfinx``.

    Handles dolfinx's lazy submodule loading (``dolfinx.fem.petsc`` is
    only available after an explicit import) -- see
    ``_introspect.resolve_dotted_path`` for the walk strategy.

    Example::

        has_attr("dolfinx.mesh.create_rectangle")    # True / False / None
        has_attr("dolfinx.fem.petsc.LinearProblem")  # True when petsc-enabled
    """
    return resolve_dotted_path(dotted_path, "dolfinx")


def is_available() -> bool:
    """``True`` if ``dolfinx`` is importable in the current environment."""
    return is_importable("dolfinx")


# Hand-curated list of dotted paths the catalog mentions in template
# code and structured knowledge.  These are checked in addition to
# whatever the live source scan picks up, so a path that the catalog
# DOES use but that our regex happens to miss (broken across lines,
# embedded in a docstring fragment, etc.) is still verified.
#
# Partially overlaps with
# ``scripts/fingerprint_solvers.py::fingerprint_fenics`` (~9 entries
# in common); intentionally extends it because the catalog references
# a few names the fingerprint script does not (``create_unit_square``
# / ``create_unit_cube`` / ``locate_dofs_*`` / ``NonlinearProblem``).
# When the two lists drift, update both; there is no programmatic
# single source of truth.
CATALOG_API_PATHS: tuple[str, ...] = (
    "dolfinx.mesh.create_rectangle",
    "dolfinx.mesh.create_box",
    "dolfinx.mesh.create_unit_square",
    "dolfinx.mesh.create_unit_cube",
    "dolfinx.fem.functionspace",
    "dolfinx.fem.Function",
    "dolfinx.fem.dirichletbc",
    "dolfinx.fem.locate_dofs_topological",
    "dolfinx.fem.locate_dofs_geometrical",
    "dolfinx.fem.petsc.LinearProblem",
    "dolfinx.fem.petsc.NonlinearProblem",
    "dolfinx.io.XDMFFile",
    "dolfinx.io.VTKFile",
    "dolfinx.default_scalar_type",
)
