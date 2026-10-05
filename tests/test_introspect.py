"""Unit tests for the shared dotted-path walk in
``tests/groundtruth/_introspect.py``.

The FEniCSx and DUNE-fem catalog-consistency tests both rely on
``resolve_dotted_path`` but SKIP in environments without dolfinx /
dune (neither pip-installs cleanly) -- so the walk's True / False /
None branches would otherwise never be asserted in CI.

These tests exercise the algorithm against ``numpy`` (always
installed as a base dependency), which has the same shape that
matters: a top-level package with lazy submodules
(``numpy.fft`` is not always side-imported by ``numpy.__init__``),
exactly the case the PR-#7 fix addressed.  Locking it in here means
a future refactor that re-breaks the lazy-submodule fallback fails a
real assertion instead of silently passing because the FEniCSx/DUNE
tests skip.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.groundtruth._introspect import (  # noqa: E402
    is_importable,
    resolve_dotted_path,
)


class TestResolveDottedPath(unittest.TestCase):
    def test_top_level_attribute(self):
        self.assertTrue(resolve_dotted_path("numpy.ndarray", "numpy"))

    def test_nested_attribute(self):
        self.assertTrue(resolve_dotted_path("numpy.linalg.norm", "numpy"))

    def test_lazy_submodule_fallback(self):
        # A package whose __init__ does not import its child: the child is
        # NOT an attribute until imported, so only the getattr-then-
        # import_module fallback (the PR-#7 fix) can resolve the path.
        # numpy.fft could not show this: where numpy exposes fft, getattr
        # resolves it and the test passed with the fallback removed.
        import importlib
        import tempfile
        import uuid
        name = f"lazy_pkg_{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / name
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "child.py").write_text("def leaf():\n    return 1\n")
            sys.path.insert(0, tmp)
            try:
                root = importlib.import_module(name)
                self.assertFalse(hasattr(root, "child"))
                self.assertTrue(resolve_dotted_path(f"{name}.child.leaf", name))
                self.assertFalse(resolve_dotted_path(f"{name}.child.missing", name))
            finally:
                sys.path.remove(tmp)
                for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
                    del sys.modules[mod]

    def test_bare_root(self):
        self.assertTrue(resolve_dotted_path("numpy", "numpy"))

    def test_missing_top_level_attribute(self):
        self.assertFalse(
            resolve_dotted_path("numpy.definitely_not_here_xyz", "numpy")
        )

    def test_missing_nested_attribute(self):
        self.assertFalse(
            resolve_dotted_path("numpy.linalg.definitely_not_here_xyz", "numpy")
        )

    def test_wrong_root_returns_none(self):
        # Path not rooted at the expected package -> None, distinct
        # from False, so callers can tell "wrong question" from
        # "missing attribute".
        self.assertIsNone(resolve_dotted_path("scipy.whatever", "numpy"))

    def test_unimportable_root_returns_none(self):
        self.assertIsNone(
            resolve_dotted_path(
                "definitely_not_a_pkg_xyz.x", "definitely_not_a_pkg_xyz"
            )
        )


class TestIsImportable(unittest.TestCase):
    def test_installed_package(self):
        self.assertTrue(is_importable("numpy"))

    def test_missing_package(self):
        self.assertFalse(is_importable("definitely_not_a_pkg_xyz"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
