"""Source-code ground-truth probes for catalog-consistency tests.

Each module here fetches a small slice of a backend's source-of-truth (a
local checkout, the upstream GitHub raw URL, or the installed Python
package) and extracts the canonical names that the backend's input parser
or API actually accepts.  The test suite in
``tests/test_catalog_consistency.py`` then cross-checks those against
what the MCP knowledge base claims.

The catalog can drift in two directions:
  * the catalog promises a keyword the solver no longer accepts (BREAKING)
  * the solver accepts a keyword the catalog never advertised
    (DEAD ENTRY -- agent never finds it)

This package surfaces both.

────────────────────────────────────────────────────────────────────────
HOW TO ADD A NEW BACKEND PROBE
────────────────────────────────────────────────────────────────────────

There are two recurring patterns.

(1) Source-grep — for backends whose canonical names live in compiled
    C++/Fortran source (4C, deal.II, FEBio).  Worked example:
    ``fourc.py``.  Pattern:

        - ``fetch_source(rel_path)`` -- read local checkout via
          ``$<BACKEND>_ROOT`` or raw.githubusercontent.com with a 24h
          on-disk cache; return ``None`` if both unavailable.
        - One parser per knowledge-base section (e.g.
          ``dynamictype_options()`` for DYNAMICTYPE,
          ``material_parameter_keys(path)`` for a MAT_* class).
        - Each parser must use a balanced-brace / explicit-token scan
          rather than ``[^}]+``; nested ``{...}`` will otherwise cut the
          match short.
        - Each parser returns ``None`` when the underlying file is
          unreachable so the test can ``skip`` gracefully.

(2) Introspection — for Python-importable backends (FEniCSx, NGSolve,
    scikit-fem, Kratos, DUNE-fem).  ``scripts/fingerprint_solvers.py``
    already implements this style for drift-against-prior-fingerprint
    comparisons; the catalog-consistency variant should reuse those
    fingerprints (or call ``importlib`` directly) and assert that every
    catalog entry maps to a real ``module.attribute``.

Each backend's module must expose:
  * A ``MATERIAL_SOURCE``-style map (or its equivalent) telling the test
    how to translate a catalog entry name into the source-of-truth probe.
  * One zero-arg ``<keyword>_options()`` function per categorical section
    the catalog promises (e.g. element types, time integrators).
  * A docstring noting which file/module is the source of truth so a
    future contributor can audit drift in either direction.

Existing modules:
  * ``fourc`` -- DYNAMICTYPE keywords, material parameter keys

Stubs (return ``None``; remove the stub once a real probe is added):
  * see ``_stubs.py``
"""
