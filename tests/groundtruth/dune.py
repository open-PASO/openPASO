"""Ground-truth probes for DUNE-fem.

DUNE-fem's Python bindings expose a layered package tree rooted at
``dune``: grid creation under ``dune.grid``, function spaces and
schemes under ``dune.fem.space`` / ``dune.fem.scheme``, UFL
integration under ``dune.ufl``.  The catalog references these by
dotted path (``dune.fem.space``, ``dune.grid.reader.gmsh``,
``dune.ufl``), so verification is a per-path attribute check.

Like FEniCSx, DUNE-fem is rarely pip-installable cleanly (it builds
C++ modules such as ``dune-alugrid`` from source).  All probes return
``None`` when ``dune`` is not importable so the test skips in
environments that lack it.

Shares the dotted-path walk with the FEniCSx probe via
``_introspect.resolve_dotted_path``.
"""

from __future__ import annotations

from tests.groundtruth._introspect import is_importable, resolve_dotted_path


def has_attr(dotted_path: str) -> bool | None:
    """Resolve a ``dune.<path>`` dotted reference on the installed
    ``dune`` package.  ``True`` / ``False`` / ``None`` (last when
    ``dune`` is unimportable or the path is not rooted at ``dune``).

    Handles DUNE's lazy submodule loading (``dune.fem.space`` etc.
    are not all side-imported by ``dune.__init__``) -- see
    ``_introspect.resolve_dotted_path``.
    """
    return resolve_dotted_path(dotted_path, "dune")


def is_available() -> bool:
    """``True`` if ``dune`` is importable in the current environment."""
    return is_importable("dune")


# Dotted paths the DUNE-fem catalog references in template code.
# Hand-curated from src/backends/dune/generators/.  Checked in
# addition to the live source scan so a path the regex misses
# (split across lines, etc.) is still verified.
CATALOG_API_PATHS: tuple[str, ...] = (
    "dune.grid",
    "dune.fem",
    "dune.fem.space",
    "dune.fem.scheme",
    "dune.fem.function",
    "dune.ufl",
)
