"""Cross-check the MCP knowledge base against the backend's actual source.

The MCP catalog promises specific keyword strings to the agent: section names,
DYNAMICTYPE options, material classes, parameter keys, element types.  If any
of those drift away from what the solver's input parser actually accepts, the
agent will emit input files that the solver silently rejects -- exactly the
"silent input-mismatch" failure mode the paper's pitfall layer is meant to
prevent.

These tests close the loop by fetching the canonical names directly from the
solver source (local checkout via ``$<SOLVER>_ROOT`` or
raw.githubusercontent.com) and asserting set membership.

The tests SKIP -- they do not fail -- when neither a local source tree nor
network access is available.  This keeps the suite green in offline CI but
runs real checks whenever sources are reachable (e.g. on a weekly cron).
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tests.groundtruth import dealii as dealii_gt  # noqa: E402
from tests.groundtruth import dune as dune_gt  # noqa: E402
from tests.groundtruth import fenics as fenics_gt  # noqa: E402
from tests.groundtruth import fourc as fourc_gt  # noqa: E402
from tests.groundtruth import kratos as kratos_gt  # noqa: E402
from tests.groundtruth import ngsolve as ngsolve_gt  # noqa: E402
from tests.groundtruth import skfem as skfem_gt  # noqa: E402


def _normalise_key(raw: str) -> str:
    """Strip any ``" (annotation)"`` suffix the catalog appends for human
    readability.  The actual 4C input parser sees the bare key only."""
    return raw.split("(", 1)[0].strip()


def _parse_param_string(text: str) -> set[str]:
    """Parse a comma-separated parameter list as used by the older
    ``plasticity_models``-style entries.  Drops empty tokens."""
    return {
        _normalise_key(t)
        for t in (s.strip() for s in text.split(","))
        if t
    }


def _collect_fourc_catalog_materials() -> dict[str, set[str]]:
    """Walk every 4C physics generator, pull every materials-like block, and
    return {material_name: set_of_parameter_keys}.

    The material name is kept as the catalog literally writes it -- including
    composite forms like ``MAT_ElastHyper + ELAST_CoupNeoHooke`` -- because
    the source-of-truth check for composites is the union of each
    component's parameter set.  Stripping the suffix loses information.

    Two storage schemas coexist in the catalog:
      * structured ``parameters: {KEY: {description, range}}`` (preferred)
      * legacy ``parameters: "KEY1, KEY2, ..."`` comma-separated string
    Both are parsed here so the test catches the same bug class in either
    form -- otherwise free-form-string entries (which is where the original
    GTN-with-lowercase-keys bug lived) silently bypass verification.
    """
    from backends.fourc.generators import _GENERATOR_SPECS, get_generator

    # Collect material-like blocks at multiple nesting depths -- the legacy
    # ``plasticity_models`` block in solid_mechanics is a sibling of
    # ``materials`` but holds the same kind of name -> param mapping.
    interesting_blocks = ("materials", "plasticity_models")

    out: dict[str, set[str]] = {}
    for module_key in _GENERATOR_SPECS:
        try:
            gen = get_generator(module_key)
        except Exception:
            continue
        try:
            knowledge = gen.get_knowledge()
        except Exception:
            continue
        for block_name in interesting_blocks:
            block = knowledge.get(block_name)
            if not isinstance(block, dict):
                continue
            for mat_name, body in block.items():
                keys: set[str] = set()
                if isinstance(body, dict):
                    params = body.get("parameters")
                    if isinstance(params, dict):
                        keys = {_normalise_key(k) for k in params if isinstance(k, str)}
                    elif isinstance(params, str):
                        keys = _parse_param_string(params)
                # No params at all -> still record the entry with empty set so
                # the "no_catalog_key_is_unknown" check stays silent for it
                # while the down-stream "missing-keys" report can flag it.
                out.setdefault(mat_name, set()).update(keys)
    return out


def _allowed_keys_for(name: str, source_lookup: dict[str, set[str]]) -> set[str] | None:
    """Resolve the allowed-parameter set for a catalog entry name.

    For composite names (``MAT_ElastHyper + ELAST_CoupNeoHooke``) the
    allowed set is the union over components.  Returns ``None`` if any
    component is missing from ``source_lookup`` -- the caller should then
    skip that entry rather than miscount a partial union as ground truth.
    """
    from tests.groundtruth.fourc import composite_components

    parts = composite_components(name)
    allowed: set[str] = set()
    for p in parts:
        if p not in source_lookup:
            return None
        allowed |= source_lookup[p]
    return allowed


class TestFourcDynamictypeOptions(unittest.TestCase):
    """The catalog must not promise ``DYNAMICTYPE`` keywords that 4C rejects."""

    @classmethod
    def setUpClass(cls):
        cls.source_options = fourc_gt.dynamictype_options()
        if cls.source_options is None:
            raise unittest.SkipTest(
                "4C source unavailable (set FOURC_ROOT or enable network access)"
            )

    # Match only bullet-list lines that look like:  "  'KeyWord' -- ..."
    # so the test ignores stray quoted tokens elsewhere in the prose (e.g.
    # the MODE entry's "YN" | "Lame" mentioned for hyperelastic sub-materials).
    _BULLET_RE = re.compile(r"^\s+'([A-Z][A-Za-z0-9]+)' -- ", re.MULTILINE)

    def test_structural_dynamics_keywords_are_valid(self):
        """Every keyword introduced as a DYNAMICTYPE bullet ('Name' -- desc) in
        the structural_dynamics docstring must be a real 4C input keyword."""
        from backends.fourc.generators import get_generator

        gen = get_generator("structural_dynamics")
        doc = gen.get_knowledge()["time_integration"]["DYNAMICTYPE"]
        promised = set(self._BULLET_RE.findall(doc))
        self.assertTrue(
            promised,
            "Bullet-scanner found no DYNAMICTYPE keywords -- docstring "
            "format may have changed; update _BULLET_RE.",
        )
        unknown = promised - self.source_options
        self.assertFalse(
            unknown,
            f"\nstructural_dynamics catalog promises DYNAMICTYPE keywords that "
            f"4C does not accept: {sorted(unknown)}\n"
            f"4C input parser accepts: {sorted(self.source_options)}",
        )


class TestFourcMaterialParameters(unittest.TestCase):
    """Every parameter key the catalog lists for a 4C material must be one the
    material class actually reads via ``matdata.parameters.get<...>("KEY")``."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = _collect_fourc_catalog_materials()
        # Resolve source keys eagerly so a single unreachable source skips the
        # whole class rather than partially-running tests.
        cls.source: dict[str, set[str]] = {}
        skipped: list[str] = []
        for mat, path in fourc_gt.MATERIAL_SOURCE.items():
            keys = fourc_gt.material_parameter_keys(path)
            if keys is None:
                skipped.append(f"{mat} ({path})")
            else:
                cls.source[mat] = keys
        if not cls.source and skipped:
            raise unittest.SkipTest(
                f"4C material source unavailable: {skipped}"
            )

    def test_no_catalog_key_is_unknown_to_source(self):
        """Catalog parameter keys must be a subset of source keys.

        Failure means the agent will emit a parameter the 4C input parser
        rejects -- the same failure mode that the GTN/DYNAMICTYPE incidents
        manifested before this test existed.
        """
        violations: list[str] = []
        for mat_name, cat_keys in self.catalog.items():
            if not cat_keys:
                continue  # catalog entry declares no parameters at all
            allowed = _allowed_keys_for(mat_name, self.source)
            if allowed is None:
                continue  # component source unmapped; nothing to compare
            unknown = cat_keys - allowed
            if unknown:
                violations.append(
                    f"  {mat_name}: catalog lists {sorted(unknown)} "
                    f"which 4C source does not accept. "
                    f"Allowed: {sorted(allowed)}"
                )
        self.assertFalse(violations, "\n" + "\n".join(violations))

    def test_required_keys_present_in_catalog(self):
        """Soft check: surface 4C parameter keys that exist in source but are
        missing from the catalog, so an agent never finds them.

        These are not strictly errors (the catalog may legitimately omit
        obscure knobs) so the assertion only fails when
        ``OFA_STRICT_CATALOG=1`` is set.  Otherwise the gaps are printed to
        stderr so CI can surface them as warnings.
        """
        import os

        strict = os.environ.get("OFA_STRICT_CATALOG") == "1"
        missing: list[str] = []
        for mat_name, cat_keys in self.catalog.items():
            allowed = _allowed_keys_for(mat_name, self.source)
            if allowed is None:
                continue  # no source mapping for this entry; cannot compare
            # Note: we intentionally do NOT skip entries with empty
            # cat_keys here -- a catalog entry that promises a material
            # but lists no parameters is itself an under-declaration the
            # agent needs to know about.
            gap = allowed - cat_keys
            if gap:
                missing.append(
                    f"  {mat_name}: 4C source has parameters not in catalog: "
                    f"{sorted(gap)}"
                )
        if missing:
            msg = "Catalog under-declares 4C parameters:\n" + "\n".join(missing)
            if strict:
                self.fail(msg)
            else:
                print("\n[WARN catalog drift]\n" + msg, file=sys.stderr)


# ── scikit-fem ───────────────────────────────────────────────────────────────


_SKFEM_ELEMENT_RE = re.compile(r"\bElement[A-Z]\w*\b")
_SKFEM_MESH_RE = re.compile(r"\bMesh[A-Z]\w*\b")


def _collect_skfem_class_mentions() -> dict[str, set[str]]:
    """Walk every string value reachable under
    ``backends.skfem.generators.KNOWLEDGE`` and pull out all ``Element*``
    and ``Mesh*`` identifiers referenced by name.  These are the names the
    agent sees when it asks the MCP for guidance, so each one must be a
    real class on the installed ``skfem`` module.

    Returns ``{"Element": set, "Mesh": set}``.
    """
    from backends.skfem.generators import KNOWLEDGE

    def walk(obj, into: set[str], regex: re.Pattern[str]) -> None:
        if isinstance(obj, str):
            into.update(regex.findall(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v, into, regex)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                walk(v, into, regex)

    elements: set[str] = set()
    meshes: set[str] = set()
    walk(KNOWLEDGE, elements, _SKFEM_ELEMENT_RE)
    walk(KNOWLEDGE, meshes, _SKFEM_MESH_RE)
    return {"Element": elements, "Mesh": meshes}


class TestSkfemElementMeshClasses(unittest.TestCase):
    """Every ``Element*`` / ``Mesh*`` name the scikit-fem catalog mentions
    must be an actual class on the installed ``skfem`` package.

    This closes the same catalog-vs-source loop that
    ``TestFourcMaterialParameters`` does for 4C, but via Python
    introspection rather than C++ source-grep -- demonstrating the second
    probe family described in ``tests/groundtruth/__init__.py``.
    """

    @classmethod
    def setUpClass(cls):
        cls.source_elements = skfem_gt.element_classes()
        cls.source_meshes = skfem_gt.mesh_classes()
        if cls.source_elements is None or cls.source_meshes is None:
            raise unittest.SkipTest(
                "scikit-fem not installed (pip install scikit-fem)"
            )
        cls.mentions = _collect_skfem_class_mentions()

    def test_no_unknown_element_class_in_knowledge(self):
        promised = self.mentions["Element"]
        unknown = promised - self.source_elements
        self.assertFalse(
            unknown,
            f"\nscikit-fem catalog references Element* classes that do "
            f"not exist on `skfem`: {sorted(unknown)}\n"
            f"Typo or recent rename.  Available: "
            f"{sorted(self.source_elements)[:10]}... "
            f"({len(self.source_elements)} total)",
        )

    def test_no_unknown_mesh_class_in_knowledge(self):
        promised = self.mentions["Mesh"]
        unknown = promised - self.source_meshes
        self.assertFalse(
            unknown,
            f"\nscikit-fem catalog references Mesh* classes that do "
            f"not exist on `skfem`: {sorted(unknown)}\n"
            f"Available: {sorted(self.source_meshes)}",
        )


# ── deal.II ──────────────────────────────────────────────────────────────────


_DEALII_FE_RE = re.compile(r"\bFE_[A-Z][A-Za-z0-9_]*\b")


def _collect_dealii_fe_mentions() -> set[str]:
    """Return every ``FE_*`` class name referenced anywhere the
    deal.II catalog touches.

    Three places reference FE_* names and all three are scanned:

    * ``src/backends/dealii/generators/*.py`` -- per-physics template
      code and structured knowledge blocks (live in the same file).
    * ``src/backends/dealii/backend.py`` and any other top-level
      module in the deal.II backend package -- e.g. supported-
      physics declarations sometimes embed example element names.
    * ``src/tools/deep_knowledge.py`` -- the cross-cutting catalog
      module hosting ``_DEALII_KNOWLEDGE``.  A typo'd FE_* name added
      to that dict would be invisible to a generator-only scan.

    All three are read as raw .py source: the catalog embeds template
    code as multi-line string literals, and runtime knowledge dicts
    are defined in those same files, so a single grep over the .py
    text picks up both at once.
    """
    out: set[str] = set()
    repo_src = Path(__file__).parent.parent / "src"
    paths: list[Path] = []
    dealii_dir = repo_src / "backends" / "dealii"
    if dealii_dir.is_dir():
        paths.extend(dealii_dir.rglob("*.py"))
    deep_knowledge = repo_src / "tools" / "deep_knowledge.py"
    if deep_knowledge.is_file():
        paths.append(deep_knowledge)
    for py in paths:
        out.update(
            _DEALII_FE_RE.findall(
                py.read_text(encoding="utf-8", errors="replace")
            )
        )
    return out


class TestDealiiFiniteElementClasses(unittest.TestCase):
    """Every ``FE_*`` name the deal.II catalog mentions (in either
    knowledge prose or generated Python template code) must be a real
    concrete class under ``include/deal.II/fe/``."""

    @classmethod
    def setUpClass(cls):
        cls.source = dealii_gt.fe_class_names()
        if cls.source is None:
            raise unittest.SkipTest(
                "deal.II source unavailable "
                "(set DEALII_ROOT or enable network access)"
            )
        cls.mentions = _collect_dealii_fe_mentions()

    def test_no_unknown_fe_class_in_catalog(self):
        unknown = self.mentions - self.source
        # Tolerate a small denylist of "marker" / "abstract" / "alias"
        # references that show up in prose but never get instantiated
        # in template code. The catalog cites them for the LLM's
        # taxonomic understanding of deal.II's FE hierarchy:
        #   FE_Base, FE_Data     — abstract base classes
        #   FE_System            — alias / typo guidance for FESystem
        #                           (catalog warns LLMs against the
        #                           underscored form; the warning text
        #                           necessarily contains the bad name)
        #   FE_Series            — namespace FESeries (legendre / fourier
        #                           helpers), not a class
        #   FE_DG                — template base name covering
        #                           FE_DGRaviartThomas / FE_DGBDM /
        #                           FE_DGNedelec etc.; never instantiated
        #                           directly. (Audit 2026-06-01.)
        unknown -= {"FE_Base", "FE_Data",
                    "FE_System", "FE_Series", "FE_DG"}
        self.assertFalse(
            unknown,
            f"\ndeal.II catalog references FE_* classes that do not "
            f"exist in deal.II source: {sorted(unknown)}\n"
            f"Source has {len(self.source)} concrete FE_* classes. "
            f"First 15: {sorted(self.source)[:15]}",
        )

    def test_under_declared_fe_classes(self):
        """Soft check: ``FE_*`` classes that exist in deal.II source but
        are never mentioned by the catalog.  Each unmentioned class is
        a feature the agent will never propose -- not strictly wrong,
        but a gap worth surfacing.

        Mirrors the 4C ``test_required_keys_present_in_catalog`` --
        prints to stderr by default, fails the test only when
        ``OFA_STRICT_CATALOG=1`` is set so the under-declaration list
        does not block PRs.
        """
        import os

        strict = os.environ.get("OFA_STRICT_CATALOG") == "1"
        gap = self.source - self.mentions
        if gap:
            msg = (
                f"deal.II catalog under-declares FE_* classes "
                f"({len(gap)} unmentioned of {len(self.source)} in source): "
                f"{sorted(gap)}"
            )
            if strict:
                self.fail(msg)
            print(f"\n[WARN catalog drift]\n{msg}", file=sys.stderr)


# ── NGSolve ──────────────────────────────────────────────────────────────────


# Curated list of high-value ngsolve identifiers commonly emitted by
# the catalog templates.  Whitelist beats regex here because the
# templates also contain user-defined helper functions, single-letter
# math symbols (``A``, ``B``, ``E``, ``H``, ``J2``, ``Q``) used as
# variables, and submodule classes (``Sphere``, ``OrthoBrick`` from
# ``ngsolve.csg``).  An open regex over-matches all of those; this
# whitelist limits the check to identifiers whose drift (rename or
# removal upstream) would silently break agent-generated scripts.
#
# Extend this list when adding new templates that reference more of
# the ngsolve API.  Anything not in the list is simply skipped --
# wrong-name typos for already-listed names are still caught.
_NGSOLVE_CORE_IDENTIFIERS: tuple[str, ...] = (
    # FE spaces (vector / scalar / specialised)
    "H1", "HCurl", "HDiv", "L2", "NumberSpace",
    "VectorH1", "VectorL2", "FESpace", "Compress",
    "FacetFESpace", "HCurlCurl", "HDivDiv", "SurfaceL2",
    "Discontinuous", "Periodic",
    # Mesh
    "Mesh",
    # Forms and the solver pipeline
    "BilinearForm", "LinearForm", "GridFunction",
    "SymbolicEnergy", "Variation",
    # Operators / integrators / functional ops
    "Integrate", "InnerProduct", "grad", "div", "curl",
    "Grad", "Trace", "Norm", "Det", "Inv", "Id", "MatrixValued",
    "IfPos",
    # Linear algebra / solvers
    "BlockMatrix", "Matrix", "Vector", "ArnoldiSolver", "Solve",
    "HCurlAMG", "TaskManager",
    # Coefficient handling and IO
    "CoefficientFunction", "Parameter", "VTKOutput", "Draw",
    # Boundary / volume markers
    "VOL", "BND", "BBND",
)


def _collect_ngsolve_mentions() -> set[str]:
    """Return the subset of ``_NGSOLVE_CORE_IDENTIFIERS`` actually
    referenced anywhere in the ngsolve generator package.

    Scans the raw .py source for word-boundary matches.  That source
    contains both executable template strings (which the agent runs)
    and KNOWLEDGE-block prose / pitfall text (which it does not).
    A name appearing only in prose still counts as a "mention" here
    -- benign for the subset test because the same name's presence
    on ``dir(ngsolve)`` is verified either way, but worth knowing
    when reading the assertion message: an "unknown" entry could
    come from prose just as easily as from generated code.
    """
    out: set[str] = set()
    ngs_dir = Path(__file__).parent.parent / "src" / "backends" / "ngsolve" / "generators"
    if not ngs_dir.is_dir():
        return out
    blob = "".join(
        py.read_text(encoding="utf-8", errors="replace")
        for py in ngs_dir.rglob("*.py")
    )
    for name in _NGSOLVE_CORE_IDENTIFIERS:
        if re.search(rf"\b{re.escape(name)}\b", blob):
            out.add(name)
    return out


class TestNgsolveAttributes(unittest.TestCase):
    """Every watch-listed ngsolve identifier that the catalog uses must
    exist as a public attribute on the installed ``ngsolve`` module.

    Catches drift-style failures like the package renaming ``H1`` to
    ``H1FESpace`` or moving ``Integrate`` into a submodule -- both
    of which would silently break agent-generated scripts.  Wider
    coverage would require AST-parsing the multi-line template
    strings to separate user-defined helpers from imports; this
    targeted whitelist is the trade-off until that work is done.
    """

    @classmethod
    def setUpClass(cls):
        cls.source = ngsolve_gt.public_attrs()
        if cls.source is None:
            raise unittest.SkipTest(
                "NGSolve not installed (pip install ngsolve)"
            )
        cls.mentions = _collect_ngsolve_mentions()

    def test_watchlist_mentions_are_real(self):
        # A successful run will normally have a non-empty mention set
        # (the templates do use H1, BilinearForm, etc.).  An empty
        # mention set after the generators directory exists usually
        # signals that the templates moved, so we assert non-empty
        # too -- otherwise the test passes vacuously.
        self.assertTrue(
            self.mentions,
            "No watch-listed ngsolve identifiers found in templates -- "
            "either the catalog moved or _NGSOLVE_CORE_IDENTIFIERS "
            "needs updating.",
        )
        unknown = self.mentions - self.source
        self.assertFalse(
            unknown,
            f"\nNGSolve catalog references identifiers that are not "
            f"public attributes of ngsolve: {sorted(unknown)}\n"
            f"Source has ~{len(self.source)} public attrs.",
        )


# ── Kratos ───────────────────────────────────────────────────────────────────


# Three catalog forms collapse to one bare ``<Name>Application`` name:
#   * ``KratosMultiphysics.<Name>Application``  -- the actual import path
#     the agent's generated code executes.  Load-bearing: a typo here
#     makes the generated script crash with ``ModuleNotFoundError``.
#   * ``Kratos<Name>Application``               -- pip install hints and
#     prose lists.  Maps to the same bare name after stripping ``Kratos``.
#   * standalone ``<Name>Application``          -- prose, ``applications``
#     list entries, print messages.  Not load-bearing on its own, but a
#     typo here teaches the agent the wrong import name for next time.
_KRATOS_IMPORT_RE = re.compile(
    r"\bKratosMultiphysics\.([A-Z][A-Za-z0-9]*Application)\b"
)
_KRATOS_PREFIXED_RE = re.compile(
    r"\bKratos([A-Z][A-Za-z0-9]*Application)\b"
)
_KRATOS_BARE_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*Application)\b")


def _collect_kratos_application_mentions() -> set[str]:
    """Return every bare ``<Name>Application`` identifier referenced
    anywhere the Kratos catalog touches, normalising all three of the
    catalog conventions to the bare form.

    Scans the Kratos backend package recursively and the cross-cutting
    ``src/tools/deep_knowledge.py``.  The bare-form regex carries a
    small false-positive risk (any uppercase identifier ending in
    ``Application`` will match), but the scan is scoped to the
    Kratos directory so unrelated framework names are unlikely to
    appear; if one does, it would simply fail the subset check
    against the upstream apps listing, which is the desired signal.
    """
    out: set[str] = set()
    repo_src = Path(__file__).parent.parent / "src"
    paths: list[Path] = []
    kratos_dir = repo_src / "backends" / "kratos"
    if kratos_dir.is_dir():
        paths.extend(kratos_dir.rglob("*.py"))
    deep_knowledge = repo_src / "tools" / "deep_knowledge.py"
    if deep_knowledge.is_file():
        paths.append(deep_knowledge)
    for py in paths:
        text = py.read_text(encoding="utf-8", errors="replace")
        out.update(_KRATOS_IMPORT_RE.findall(text))
        out.update(_KRATOS_PREFIXED_RE.findall(text))
        # Bare regex over-matches: ``\b([A-Z]\w*Application)\b`` on
        # ``KratosChimeraApplication`` captures the whole thing
        # (including the ``Kratos`` prefix).  Filter those out --
        # they're already covered by ``_KRATOS_PREFIXED_RE`` which
        # captures just the bare portion.
        out.update(
            m for m in _KRATOS_BARE_RE.findall(text)
            if not m.startswith("Kratos")
        )
    # Names that are deliberately not real. The served text teaches that
    # you cannot reach an element through an application object by writing
    # `SomeApplication.ThermalFace2D2N(...)`, and the placeholder is the
    # point of the sentence. Counting it as a catalog claim is a false
    # positive of the same shape as reading "/home/user/4C" in a docstring
    # as somebody's real home directory.
    return out - {"SomeApplication", "MyApplication", "YourApplication"}


class TestKratosApplications(unittest.TestCase):
    """Every Kratos application the catalog mentions must be a real
    sub-application of Kratos -- i.e. correspond to a directory under
    ``applications/`` in the KratosMultiphysics/Kratos repo.

    Probe is source-enumeration: lists the GitHub contents API for the
    upstream ``applications/`` directory.  No Kratos install needed --
    the probe is purely a directory listing, so the test works in any
    CI environment that has network access.

    Mentions are matched in BOTH catalog conventions:
    ``KratosMultiphysics.<Name>Application`` (the agent's actual
    import path) and ``Kratos<Name>Application`` (pip install
    hints, prose lists); each is normalised to the bare
    ``<Name>Application`` form before the subset check.
    """

    @classmethod
    def setUpClass(cls):
        cls.source = kratos_gt.application_names()
        if cls.source is None:
            raise unittest.SkipTest(
                "Kratos applications listing unavailable "
                "(set KRATOS_ROOT to an upstream git checkout, "
                "or enable network access -- a pip install of "
                "KratosMultiphysics is NOT a substitute since the "
                "probe needs the applications/ directory listing)"
            )
        cls.mentions = _collect_kratos_application_mentions()

    def test_no_unknown_application_in_catalog(self):
        unknown = self.mentions - self.source
        self.assertFalse(
            unknown,
            f"\nKratos catalog references applications that do not "
            f"exist upstream: {sorted(unknown)}\n"
            f"Upstream has {len(self.source)} applications. "
            f"First 10: {sorted(self.source)[:10]}",
        )

    def test_under_declared_applications(self):
        """Soft check: applications that exist upstream but are never
        mentioned by the catalog.  Each unmentioned application is a
        capability the agent will never propose.  Mirrors the deal.II
        soft-warning test: stderr-print by default, hard-fail only
        when ``OFA_STRICT_CATALOG=1`` is set.
        """
        import os

        strict = os.environ.get("OFA_STRICT_CATALOG") == "1"
        gap = self.source - self.mentions
        if gap:
            msg = (
                f"Kratos catalog under-declares applications "
                f"({len(gap)} unmentioned of {len(self.source)} upstream): "
                f"{sorted(gap)}"
            )
            if strict:
                self.fail(msg)
            print(f"\n[WARN catalog drift]\n{msg}", file=sys.stderr)


# ── FEniCSx / dolfinx ────────────────────────────────────────────────────────


# Catalog mentions of dolfinx attributes typically follow the
# ``dolfinx.<submodule>.<name>`` pattern.  Match dotted paths of any
# depth so submodule additions (``dolfinx.fem.petsc.LinearProblem``,
# ``dolfinx.io.XDMFFile``) are picked up uniformly.
_DOLFINX_DOTTED_RE = re.compile(r"\bdolfinx(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")


def _collect_dolfinx_path_mentions() -> set[str]:
    """Return every ``dolfinx.<path>...`` dotted identifier referenced
    in any FEniCSx generator file under ``src/backends/fenics/``.

    Deliberately excludes ``src/tools/deep_knowledge.py`` because its
    ``_FENICS_KNOWLEDGE`` block contains an ``api_changes`` section
    listing OLD-vs-NEW dolfinx names side by side
    (``dolfinx.io.gmsh.model_to_mesh`` and the current
    ``dolfinx.io.gmshio.model_to_mesh``, etc.).  Those historical
    names are not all simultaneously present in any given dolfinx
    version, so a scan that includes them would falsely fail the
    test.  The catalog files under ``src/backends/fenics/`` only
    contain template code the agent actually executes, so paths
    there must resolve at runtime.
    """
    out: set[str] = set()
    repo_src = Path(__file__).parent.parent / "src"
    fenics_dir = repo_src / "backends" / "fenics"
    if fenics_dir.is_dir():
        for py in fenics_dir.rglob("*.py"):
            out.update(
                _DOLFINX_DOTTED_RE.findall(
                    py.read_text(encoding="utf-8", errors="replace")
                )
            )
    return out


class TestFenicsDolfinxAPIPaths(unittest.TestCase):
    """Every ``dolfinx.<path>`` dotted reference the FEniCSx catalog
    uses must resolve on the installed dolfinx package.

    Skipped when dolfinx is not importable (typical in a non-conda
    install -- the canonical dolfinx install path is conda, not pip).
    The test runs whenever dolfinx IS available: in developer
    environments using ``conda env create -f environment.yml``, in
    the weekly cron job that installs ``.[all-solvers,dev]``, or in
    any CI lane that explicitly conda-installs dolfinx.

    For the contributor: this test fails when dolfinx renames an
    API entry between releases (e.g. ``functionspace`` was
    ``FunctionSpace`` in earlier versions) or moves a submodule
    (e.g. ``dolfinx.fem.petsc.LinearProblem`` only exists when
    PETSc is built in).
    """

    @classmethod
    def setUpClass(cls):
        if not fenics_gt.is_available():
            raise unittest.SkipTest(
                "dolfinx not importable -- pip wheels are not available "
                "on most platforms; conda is the canonical install."
            )
        cls.mentions = _collect_dolfinx_path_mentions()
        # Union the catalog scan with the fingerprint-script's canonical
        # paths so a newly-added template that hasn't been scanned yet
        # is still covered, and a path that exists in the catalog but is
        # missing from the canonical list (drift the other direction)
        # is also surfaced.
        cls.checked = cls.mentions | set(fenics_gt.CATALOG_API_PATHS)

    def test_no_unresolved_dolfinx_path(self):
        # Each path is resolved via importlib walks; collect the
        # failing ones with their failure mode for an actionable
        # assertion message.
        unresolved: list[str] = []
        for path in sorted(self.checked):
            ok = fenics_gt.has_attr(path)
            if ok is False:
                unresolved.append(path)
            # ok is None -> dolfinx vanished mid-run; setUpClass would
            # have caught the import failure earlier, so this shouldn't
            # happen.  Leaving the branch implicit avoids hiding a
            # genuine ImportError under a soft warning.
        self.assertFalse(
            unresolved,
            f"\nFEniCSx catalog references dolfinx paths that do not "
            f"resolve on the installed package: {unresolved}\n"
            f"Likely an upstream rename or a build that omitted the "
            f"submodule (e.g. petsc-less dolfinx).",
        )


# ── DUNE-fem ─────────────────────────────────────────────────────────────────


# Match ``dune.<path>`` dotted references of any depth.  The trailing
# ``\b`` plus the greedy ``(?:\.…)+`` group captures the full dotted
# run (e.g. ``dune.grid.reader.gmsh``) rather than stopping early.
_DUNE_DOTTED_RE = re.compile(r"\bdune(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")


def _collect_dune_path_mentions() -> set[str]:
    """Return every ``dune.<path>...`` dotted identifier referenced in
    any DUNE-fem generator file under ``src/backends/dune/``.

    Scoped to the backend's own generators (template code the agent
    executes).  Like the FEniCSx scan, the cross-cutting
    ``deep_knowledge.py`` is intentionally excluded -- it can carry
    version-historical names that don't all resolve against one
    DUNE release.
    """
    out: set[str] = set()
    dune_dir = Path(__file__).parent.parent / "src" / "backends" / "dune"
    if dune_dir.is_dir():
        for py in dune_dir.rglob("*.py"):
            out.update(
                _DUNE_DOTTED_RE.findall(
                    py.read_text(encoding="utf-8", errors="replace")
                )
            )
    return out


class TestDuneFemAPIPaths(unittest.TestCase):
    """Every ``dune.<path>`` dotted reference the DUNE-fem catalog uses
    must resolve on the installed ``dune`` package.

    Skipped when ``dune`` is not importable -- DUNE-fem builds C++
    modules (``dune-alugrid`` etc.) from source and is not cleanly
    pip-installable, so a vanilla venv / CI runner skips.  Runs for
    real wherever a built DUNE-fem is available.
    """

    @classmethod
    def setUpClass(cls):
        if not dune_gt.is_available():
            raise unittest.SkipTest(
                "dune not importable -- DUNE-fem builds C++ deps from "
                "source and is not cleanly pip-installable."
            )
        cls.checked = (
            _collect_dune_path_mentions() | set(dune_gt.CATALOG_API_PATHS)
        )
        # The catalog deliberately NAMES a handful of paths that do not
        # exist — they are the phantom APIs a model would otherwise
        # reach for (dune.fem.space.product_space, the lowercase
        # raviartthomas, an importable dune.fem.solver). The exemption
        # list lives with the documentation, in
        # src/backends/dune/generators/verified_api.py, and
        # test_known_absent_paths_really_are_absent below turns it into
        # an extra positive check rather than a hole.
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from backends.dune.generators.verified_api import (  # noqa: E402
            JIT_GENERATED_PATH_PREFIXES, KNOWN_ABSENT_DUNE_PATHS)
        cls.known_absent = set(KNOWN_ABSENT_DUNE_PATHS)
        cls.jit_prefixes = tuple(JIT_GENERATED_PATH_PREFIXES)

    def test_no_unresolved_dune_path(self):
        unresolved = [
            path for path in sorted(self.checked)
            if dune_gt.has_attr(path) is False
            and path not in self.known_absent
            and not path.startswith(self.jit_prefixes)
        ]
        self.assertFalse(
            unresolved,
            f"\nDUNE-fem catalog references dune paths that do not "
            f"resolve on the installed package: {unresolved}\n"
            f"Likely an upstream rename or a build missing the module.\n"
            f"If the catalog names one of these BECAUSE it does not "
            f"exist, add it to KNOWN_ABSENT_DUNE_PATHS in "
            f"src/backends/dune/generators/verified_api.py — that list "
            f"is itself asserted to stay unresolvable.",
        )

    def test_known_absent_paths_really_are_absent(self):
        """The exemption list must never shelter a path that exists.

        If upstream adds e.g. a real ``dune.fem.space.product_space``,
        this fires and the corresponding "phantom API" claim in the
        catalog has to be retired.
        """
        resurrected = [
            path for path in sorted(self.known_absent)
            if dune_gt.has_attr(path) is not False
        ]
        self.assertFalse(
            resurrected,
            f"\nPaths listed as KNOWN_ABSENT now RESOLVE on the "
            f"installed dune: {resurrected}\n"
            f"The catalog documents them as phantom APIs — that claim "
            f"is no longer true for this version. Remove them from "
            f"KNOWN_ABSENT_DUNE_PATHS and update "
            f"verified_api.EXECUTED_API['phantom_apis_checked'].",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
