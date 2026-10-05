"""Shared dotted-path resolution for Python-introspection probes.

Both the FEniCSx (``dolfinx.*``) and DUNE-fem (``dune.*``) probes need
to resolve a dotted attribute path against an installed package,
handling Python's lazy submodule loading: ``getattr`` on a parent
package only sees submodules that the parent's ``__init__`` has
already imported, so ``dolfinx.fem.petsc`` / ``dune.fem.space`` style
paths require an explicit ``importlib.import_module`` fallback.

Factored out so the walk logic lives in exactly one place -- a subtle
algorithm duplicated across probes is a bug waiting to drift.
"""

from __future__ import annotations

import importlib


def resolve_dotted_path(dotted_path: str, expected_root: str) -> bool | None:
    """Resolve ``dotted_path`` against the installed ``expected_root``
    package.

    Returns:
        * ``True``  -- every segment resolved.
        * ``False`` -- a segment is genuinely missing (neither an
          attribute nor an importable submodule).
        * ``None``  -- the root package is not importable at all, or
          ``dotted_path`` is not rooted at ``expected_root`` (caller
          passed an unrelated path).  Lets callers distinguish
          "missing attribute" from "package unavailable / wrong root".

    Walk strategy: for each segment try ``getattr``; on
    ``AttributeError`` fall back to ``importlib.import_module`` of the
    accumulated path (lazy submodule) before concluding the segment
    is missing.
    """
    parts = dotted_path.split(".")
    if not parts or parts[0] != expected_root:
        return None
    try:
        obj: object = importlib.import_module(expected_root)
    except ImportError:
        return None
    module_path = expected_root
    for segment in parts[1:]:
        next_path = f"{module_path}.{segment}"
        try:
            obj = getattr(obj, segment)
        except AttributeError:
            try:
                obj = importlib.import_module(next_path)
            except ImportError:
                return False
        module_path = next_path
    return True


def is_importable(package: str) -> bool:
    """``True`` if ``package`` can be imported in the current env."""
    try:
        importlib.import_module(package)
    except ImportError:
        return False
    return True
