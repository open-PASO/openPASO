"""
MCP tools for searching and retrieving example input files across all backends.

For 4C: Searches ~2,872 test input files + tutorials
For FEniCS: Returns parametrized Python script templates
For deal.II: Returns C++ source templates based on tutorial steps
For FEBio: Returns XML templates
"""

import json
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from core.backend import detect_template_language
from core.registry import get_backend, available_backends

# Auto-detect 4C paths
FOURC_ROOT = Path(os.environ.get("FOURC_ROOT", ""))
FOURC_TESTS = FOURC_ROOT / "tests" if FOURC_ROOT else None
FOURC_TUTORIALS = FOURC_TESTS / "tutorials" if FOURC_TESTS else None


def _search_4c_input_files(keyword: str, max_results: int = 5) -> list[dict]:
    """Search 4C test input files by keyword in filename."""
    if not FOURC_TESTS or not FOURC_TESTS.is_dir():
        return []

    results = []
    keyword_lower = keyword.lower()
    for yaml_file in FOURC_TESTS.rglob("*.4C.yaml"):
        if keyword_lower in yaml_file.name.lower() or keyword_lower in str(yaml_file.parent.name).lower():
            rel = yaml_file.relative_to(FOURC_TESTS)
            results.append({
                "file": str(rel),
                "name": yaml_file.stem.replace(".4C", ""),
                "dir": yaml_file.parent.name,
                "size": yaml_file.stat().st_size,
            })
            if len(results) >= max_results:
                break
    return results


def _get_4c_tutorial_files() -> list[dict]:
    """List all 4C tutorial input files."""
    if not FOURC_TUTORIALS or not FOURC_TUTORIALS.is_dir():
        return []

    results = []
    for tut_dir in sorted(FOURC_TUTORIALS.iterdir()):
        if not tut_dir.is_dir():
            continue
        for yaml_file in tut_dir.glob("*.4C.yaml"):
            results.append({
                "tutorial": tut_dir.name,
                "file": yaml_file.name,
                "path": str(yaml_file),
            })
    return results


def register_example_tools(mcp: FastMCP):

    @mcp.tool()
    def get_example_inputs(module: str, solver: str = "fourc", max_results: int = 3) -> str:
        """Get working example input files for a given physics module.

        Returns the FULL content of real, tested input files from the solver's
        own test suite. These are the best reference for correct setup.

        IMPORTANT: Always call this before generating a new input file so you
        have a correct, validated template to work from.

        Args:
            module: Physics keyword (e.g. 'peridynamic', 'fsi', 'beam', 'contact',
                    'poisson', 'heat', 'fluid', 'elasticity')
            solver: Backend name (default: 'fourc')
            max_results: Maximum number of example files to return (default 3)
        """
        results = []

        # 4C: return full content of real test files
        if solver.lower() in ("fourc", "4c"):
            if FOURC_TESTS and FOURC_TESTS.is_dir():
                matches = []
                keyword_lower = module.lower()
                for f in sorted(FOURC_TESTS.rglob("*.4C.yaml")):
                    if keyword_lower in f.name.lower():
                        matches.append(f)
                        if len(matches) >= max_results:
                            break

                for f in matches:
                    try:
                        content = f.read_text()
                        rel_path = f.relative_to(FOURC_TESTS.parent)
                        results.append(
                            f"### Real test file: `{rel_path}`\n\n```yaml\n{content[:20000]}```\n"
                        )
                    except Exception:
                        pass

        # deal.II: return step tutorial content
        elif solver.lower() in ("dealii", "deal.ii"):
            dealii_dir = Path("/usr/share/doc/libdeal.ii-doc/examples")
            if dealii_dir.is_dir():
                keyword_lower = module.lower()
                for step_dir in sorted(dealii_dir.iterdir()):
                    if keyword_lower in step_dir.name.lower():
                        cc_file = step_dir / f"{step_dir.name}.cc"
                        if cc_file.is_file():
                            content = cc_file.read_text()
                            results.append(
                                f"### deal.II `{step_dir.name}`\n\n```cpp\n{content[:8000]}```\n"
                            )
                            if len(results) >= max_results:
                                break

        # FEniCS: return demo scripts from conda env
        elif solver.lower() in ("fenics", "fenicsx"):
            fenics_demo = Path.home() / "miniconda3" / "envs" / "fenics" / "share" / "dolfinx" / "demo"
            if not fenics_demo.is_dir():
                for p in Path.home().glob("miniconda3/envs/fenics/**/demo"):
                    if p.is_dir():
                        fenics_demo = p
                        break
            if fenics_demo.is_dir():
                keyword_lower = module.lower()
                for f in sorted(fenics_demo.rglob("*.py")):
                    if keyword_lower in f.name.lower() or keyword_lower in f.parent.name.lower():
                        content = f.read_text()
                        results.append(
                            f"### FEniCS demo: `{f.name}`\n\n```python\n{content[:8000]}```\n"
                        )
                        if len(results) >= max_results:
                            break

        # NGSolve, scikit-fem, Kratos, DUNE: generate from our templates as reference
        else:
            backend = get_backend(solver)
            if backend:
                keyword_lower = module.lower()
                # Normalize spaces/underscores/hyphens for fuzzy matching
                keyword_normalized = keyword_lower.replace(" ", "_").replace("-", "_")
                # Split into individual words for OR matching
                # "eigenvalue maxwell" matches physics containing "eigenvalue" OR "maxwell"
                keywords = [w.replace("-", "_") for w in keyword_lower.split()]
                for p in backend.supported_physics():
                    name_norm = p.name.lower().replace("-", "_")
                    desc_norm = p.description.lower()
                    match = (keyword_lower in name_norm or keyword_normalized in name_norm
                             or keyword_lower in desc_norm or keyword_normalized in desc_norm)
                    if not match and len(keywords) > 1:
                        # OR matching: any individual keyword matches
                        match = any(kw in name_norm or kw in desc_norm for kw in keywords)
                    if match:
                        for v in p.template_variants[:max_results]:
                            try:
                                content = backend.generate_input(p.name, v, {})
                                fmt = detect_template_language(content, backend.input_format().value)
                                results.append(
                                    f"### {backend.display_name()} template: `{p.name}/{v}`\n\n```{fmt}\n{content[:8000]}```\n"
                                )
                            except Exception:
                                pass
                        break

        if not results:
            return (
                f"No example files found for '{module}' in {solver}. "
                f"Try different keywords or use generate_input() for a template."
            )

        header = f"## {len(results)} example(s) for '{module}' from {solver}\n\n"
        return header + "\n---\n".join(results)

    @mcp.tool()
    def search_examples(keyword: str, solver: str = "", max_results: int = 5) -> str:
        """Search for example input files across all backends.

        For 4C: Searches ~2,872 test files + tutorials by keyword
        For FEniCS/deal.II/FEBio: Lists available template variants

        Args:
            keyword: Search term (e.g. 'poisson', 'fsi', 'beam', 'contact')
            solver: Limit to specific solver (empty = search all)
            max_results: Maximum results per backend (default 5)
        """
        results = {}

        # 4C: search actual test files AND include content preview
        if not solver or solver.lower() in ("fourc", "4c"):
            fourc_results = _search_4c_input_files(keyword, max_results)
            if fourc_results:
                # Include actual file content for the first match so the agent
                # can see real parameter values used in validated test cases
                for r in fourc_results[:2]:
                    fpath = FOURC_TESTS / r["file"] if FOURC_TESTS else None
                    if fpath and fpath.is_file():
                        try:
                            content = fpath.read_text()
                            # Include first 3000 chars of content
                            r["content_preview"] = content[:3000]
                        except Exception:
                            pass
                results["4C"] = {
                    "source": f"tests/ directory ({FOURC_TESTS})" if FOURC_TESTS else "not available",
                    "matches": fourc_results,
                }

        # All backends: search template variants
        backends_to_search = []
        if solver:
            b = get_backend(solver)
            if b:
                backends_to_search = [b]
        else:
            backends_to_search = available_backends()

        keyword_lower = keyword.lower()
        for b in backends_to_search:
            matches = []
            for p in b.supported_physics():
                if keyword_lower in p.name.lower() or keyword_lower in p.description.lower():
                    matches.append({
                        "physics": p.name,
                        "description": p.description,
                        "variants": p.template_variants,
                        "dims": p.spatial_dims,
                    })
            if matches:
                results[b.display_name()] = {"templates": matches[:max_results]}

        if not results:
            return f"No examples found for '{keyword}'. Try broader terms like 'elasticity', 'flow', 'heat'."

        return json.dumps(results, indent=2)

    @mcp.tool()
    def get_example_input(solver: str, physics: str, variant: str = "",
                          source: str = "template") -> str:
        """Retrieve a complete, runnable example input file.

        For 4C: Can return tutorial files or generated templates
        For FEniCS: Returns complete Python script
        For deal.II: Returns complete C++ source + CMakeLists
        For FEBio: Returns complete XML

        Args:
            solver: Backend name ('fenics', 'fourc', 'dealii', 'febio')
            physics: Physics type (e.g. 'poisson', 'linear_elasticity')
            variant: Template variant (e.g. '2d', '3d', 'poisson_2d')
            source: 'template' for generated, 'tutorial' for 4C tutorial files
        """
        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        # For 4C tutorials
        if source == "tutorial" and solver.lower() in ("fourc", "4c"):
            tutorials = _get_4c_tutorial_files()
            for t in tutorials:
                if physics.lower() in t["tutorial"].lower() or physics.lower() in t["file"].lower():
                    try:
                        content = Path(t["path"]).read_text()
                        return f"# 4C Tutorial: {t['tutorial']}/{t['file']}\n\n```yaml\n{content}\n```"
                    except Exception as e:
                        return f"Error reading {t['path']}: {e}"
            return f"No 4C tutorial found for '{physics}'. Available: {[t['tutorial'] for t in tutorials]}"

        # Generate from template
        if not variant:
            # Auto-select first variant
            for p in backend.supported_physics():
                if p.name == physics.lower() and p.template_variants:
                    variant = p.template_variants[0]
                    break

        if not variant:
            return f"No variant specified and no default found for {physics} in {solver}"

        try:
            content = backend.generate_input(physics, variant, {})
            fmt = detect_template_language(content, backend.input_format().value)
            return f"```{fmt}\n{content}\n```"
        except ValueError as e:
            return str(e)

    @mcp.tool()
    def list_tutorials(solver: str = "") -> str:
        """List available tutorial examples for a solver backend.

        For 4C: Lists complete tutorials with mesh files from the 4C test suite.
        For FEniCS/deal.II: Lists template variants available for generation.
        For all: Shows what ready-to-run examples exist.

        Args:
            solver: Backend name (empty = list all available)
        """
        lines = []

        # 4C tutorials (file-based)
        if not solver or solver.lower() in ("fourc", "4c"):
            tutorials = _get_4c_tutorial_files()
            if tutorials:
                lines.append("# 4C Tutorials\n")
                current_tut = ""
                for t in tutorials:
                    if t["tutorial"] != current_tut:
                        current_tut = t["tutorial"]
                        lines.append(f"\n## {current_tut}")
                    lines.append(f"- `{t['file']}`")
                lines.append("")

        # Other backends: list template variants
        from core.registry import available_backends, get_backend
        backends_to_show = []
        if solver and solver.lower() not in ("fourc", "4c"):
            b = get_backend(solver)
            if b:
                backends_to_show = [b]
        elif not solver:
            backends_to_show = available_backends()

        for b in backends_to_show:
            if b.name() == "fourc":
                continue  # already handled above
            b_lines = []
            for p in b.supported_physics():
                if p.template_variants:
                    b_lines.append(f"  - **{p.name}**: {', '.join(p.template_variants)} — {p.description}")
            if b_lines:
                lines.append(f"# {b.display_name()} Templates\n")
                lines.extend(b_lines)
                lines.append("")

        if not lines:
            return "No tutorials found. Check solver installation."
        return "\n".join(lines)

    @mcp.tool()
    def browse_solver_tests(solver: str, keyword: str = "", max_results: int = 10) -> str:
        """Browse real test/example files from a solver's own test suite.

        This is the PRIMARY source for understanding how each solver is actually
        used. These are real, validated input files from the solver's own test
        suite — not simplified templates.

        Args:
            solver: Backend name (fourc, fenics, dealii, ngsolve, kratos, dune, skfem)
            keyword: Filter by keyword in filename (empty = list all)
            max_results: Maximum number of results
        """
        results = []

        # Define where each solver's real tests/examples live
        test_dirs = {
            "fourc": FOURC_ROOT / "tests" / "input_files" if FOURC_ROOT else None,
            "4c": FOURC_ROOT / "tests" / "input_files" if FOURC_ROOT else None,
            "dealii": Path("/usr/share/doc/libdeal.ii-doc/examples"),
            "fenics": None,  # detected below
            "ngsolve": None,  # detected below
        }

        # FEniCS demos
        fenics_demo = Path.home() / "miniconda3" / "envs" / "fenics" / "share" / "dolfinx" / "demo"
        if not fenics_demo.is_dir():
            for p in Path.home().glob("miniconda3/envs/fenics/**/demo"):
                if p.is_dir():
                    fenics_demo = p
                    break
        if fenics_demo.is_dir():
            test_dirs["fenics"] = fenics_demo

        solver_key = solver.lower()
        test_dir = test_dirs.get(solver_key)

        if test_dir is None or not test_dir.is_dir():
            return (
                f"No test directory found for '{solver}'. "
                f"The agent can still use generate_input() for templates, "
                f"or read files directly if the path is known."
            )

        # Search
        extensions = {
            "fourc": "*.4C.yaml", "4c": "*.4C.yaml",
            "dealii": "*.cc",
            "fenics": "*.py",
            "ngsolve": "*.py",
        }
        ext = extensions.get(solver_key, "*")
        keyword_lower = keyword.lower()

        found_files = []
        for f in sorted(test_dir.rglob(ext)):
            if keyword and keyword_lower not in f.name.lower() and keyword_lower not in str(f.parent.name).lower():
                continue
            found_files.append(f)
            if len(found_files) >= max_results:
                break

        if not found_files:
            return f"No files matching '{keyword}' in {test_dir}"

        header = f"## {solver} test files in `{test_dir}`\n\n"
        header += f"Found {len(found_files)} matches" + (f" for '{keyword}'" if keyword else "") + ":\n\n"

        parts = [header]
        for f in found_files:
            rel = f.relative_to(test_dir)
            parts.append(f"### `{rel}` ({f.stat().st_size} bytes)\n")
            # Include content preview for ALL solvers
            try:
                content = f.read_text()
                preview = content[:3000]
                if len(content) > 3000:
                    preview += "\n... (truncated)"
                parts.append(f"```\n{preview}\n```\n")
            except Exception:
                parts.append("(could not read file)\n")

        return "\n".join(parts)

    @mcp.tool()
    def read_solver_test_file(solver: str, filepath: str) -> str:
        """Read the content of a specific test/example file from a solver.

        Use browse_solver_tests() first to find files, then this to read them.
        This gives the agent access to real, validated input files from each
        solver's own test suite — the best reference for correct usage.

        Args:
            solver: Backend name
            filepath: Relative path within the test directory (from browse_solver_tests output)
        """
        test_dirs = {
            "fourc": FOURC_ROOT / "tests" / "input_files" if FOURC_ROOT else None,
            "4c": FOURC_ROOT / "tests" / "input_files" if FOURC_ROOT else None,
            "dealii": Path("/usr/share/doc/libdeal.ii-doc/examples"),
        }

        solver_key = solver.lower()
        test_dir = test_dirs.get(solver_key)

        if test_dir is None or not test_dir.is_dir():
            return f"No test directory for '{solver}'"

        full_path = test_dir / filepath
        if not full_path.is_file():
            return f"File not found: {full_path}"

        try:
            content = full_path.read_text()
            if len(content) > 50000:
                content = content[:50000] + "\n\n... (truncated, file is very large)"
            return f"# {filepath}\n\n```\n{content}\n```"
        except Exception as e:
            return f"Error reading {full_path}: {e}"

    @mcp.tool()
    def get_input_file_guide(solver: str = "fourc") -> str:
        """Get a comprehensive guide for writing input files for a solver.

        Covers file structure, required sections, common patterns, and
        the most frequent mistakes.

        Args:
            solver: Backend name (default: 'fourc')
        """
        if solver.lower() in ("fourc", "4c"):
            return _4C_INPUT_GUIDE
        elif solver.lower() in ("fenics", "fenicsx"):
            return _FENICS_INPUT_GUIDE
        elif solver.lower() in ("dealii", "deal.ii"):
            return _DEALII_INPUT_GUIDE
        elif solver.lower() == "febio":
            return _FEBIO_INPUT_GUIDE
        else:
            return f"Unknown solver: {solver}"


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT FILE GUIDES
# ═══════════════════════════════════════════════════════════════════════════════

_4C_INPUT_GUIDE = """\
# 4C Input File Guide (.4C.yaml)

A 4C deck is a flat YAML mapping of ALL-CAPS section names. There is no
nesting of sections: subsections are spelled with a slash IN the name, e.g.
`STRUCTURAL DYNAMIC/GENALPHA`, as a separate top-level key.

4C's error messages are fmt templates: the binary contains
`Section '{}' is not a valid section name.`, not the rendered text. To check
whether a message can occur at all, grep for the TEMPLATE form, e.g.

    strings -n 6 <path-to-4C> <path-to-lib4C.so> \\
        | grep -F "Section '{}' is not a valid section name."

A few messages are assembled at runtime and are NOT findable this way (for
instance "Expected parameter 'DENS'"), and a few come from the C++ runtime
rather than from 4C, so a zero hit is weak evidence, not proof.

An unknown section name is fatal before anything runs:
    Section 'XFLUID DYNAMIC' is not a valid section name.
so guessing a section name costs the whole run. `4C --parameters` prints the
version-exact list of every accepted section and key; it is the only
authority.

## 1. GETTING A MESH IN — three routes, pick exactly one per field

### Route A — inline (no external files; use this unless told otherwise)
Three sections together:
    NODE COORDS:            list of "NODE <id> COORD <x> <y> <z>"
    <FIELD> ELEMENTS:       list of "<eid> <ELETYPE> <CELLTYPE> <nodes...> <KEY val>..."
    D<ENTITY>-NODE TOPOLOGY: list of "NODE <id> D<ENTITY> <design id>"
The element section name is per field and is one of:
    STRUCTURE ELEMENTS, THERMO ELEMENTS, FLUID ELEMENTS, TRANSPORT ELEMENTS,
    ALE ELEMENTS, LUBRICATION ELEMENTS, ARTERY ELEMENTS,
    REDUCED D AIRWAYS ELEMENTS, TRANSPORT2 ELEMENTS, PARTICLES
The topology sections are DNODE-NODE, DLINE-NODE, DSURF-NODE, DVOL-NODE
TOPOLOGY, and the entity words inside them are DNODE, DLINE, DSURFACE, DVOL.

### Route B — external Exodus mesh
    STRUCTURE GEOMETRY:
      FILE: mesh.e            # REQUIRED, absolute or relative to the input file
      SHOW_INFO: summary      # optional: none|summary|detailed_summary|detailed|full
      ELEMENT_BLOCKS:         # REQUIRED
        - ID: 1               # the block id inside the Exodus file
          SOLID:              # element type
            HEX8:             # cell type
              MAT: 1
              KINEM: nonlinear
Conditions then address node sets by name instead of by design-entity id:
    DESIGN SURF DIRICH CONDITIONS:
      - NODE_SET_NAME: wall
        NUMDOF: 3
        ONOFF: [1, 1, 1]
        VAL: [0, 0, 0]
        FUNCT: [null, null, null]
`E:` and `NODE_SET_NAME:` are mutually exclusive on one condition entry.
The per-field name is <FIELD> GEOMETRY: STRUCTURE / THERMO / FLUID /
TRANSPORT / ALE / ARTERY / LUBRICATION / TRANSPORT2 / REDUCED D AIRWAYS.

### Route C — built-in box generator (no node list at all)
    STRUCTURE DOMAIN:
      bottom_corner_point: [0.0, 0.0, 0.0]   # REQUIRED
      top_corner_point: [10.0, 1.0, 1.0]     # REQUIRED, each coord strictly greater
      subdivisions: [10, 2, 2]               # REQUIRED, elements per direction
      elements:                              # REQUIRED
        SOLID:
          HEX8:
            MAT: 1
            KINEM: nonlinear
      # optional: rotation_angle: [0,0,0]   auto_partition: false
Design entities are then attached SYMBOLICALLY — no node ids anywhere:
    DSURF-NODE TOPOLOGY:
      - "SIDE structure x- DSURFACE 1"
      - "SIDE structure x+ DSURFACE 2"
    DLINE-NODE TOPOLOGY:
      - "EDGE structure x- y- DLINE 1"
    DNODE-NODE TOPOLOGY:
      - "CORNER structure x- y- z- DNODE 1"
    DVOL-NODE TOPOLOGY:
      - "VOLUME structure DVOLUME 1"
The word after SIDE/EDGE/CORNER/VOLUME is the DISCRETISATION name (structure,
thermo, fluid, ale, scatra...), not the section name, and it is CASE
SENSITIVE and lowercase - "SIDE STRUCTURE x- DSURFACE 1" aborts with
"Could not find discretization 'STRUCTURE'." Mixing a DOMAIN section with a NODE COORDS
section does NOT fail - the extra nodes are ignored and the run exits 0. It
only surfaces if a RESULT DESCRIPTION happens to reference a node id the
generator renumbered, as "Node 1 does not belong to discretization
structure". Pick one route. The generator makes
HEX8/20/27 and WEDGE6/15 ONLY, so Route C is 3D-only and any 2D problem
must use Route A. Asking it for a 2D cell fails in three different places
depending on how you ask: "SOLID: QUAD4:" gives "Could not match this input"
(SOLID owns no quad4 at all); "WALL: QUAD4:" with the default
auto_partition: false gives "This map-partition is only available for
HEX-elements!"; and "WALL: QUAD4:" with auto_partition: true finally gives
the explicit "The discretization type quad4, is not implemented. Currently
only HEX(8,20,27) and WEDGE(6,15) are implemented for the box geometry
generation."

## 2. THE ELEMENT LINE — what each element type demands

Grammar: `<eid> <ELETYPE> <CELLTYPE> <node ids...> <KEY value> <KEY value>...`
Keys are positionless but every REQUIRED key must be present, and any token
the element does not own is fatal ("After parsing, the line still contains
'<token>'").

| ELETYPE | cell types | REQUIRED keys |
|---|---|---|
| SOLID  | HEX8 HEX18 HEX20 HEX27 TET4 TET10 WEDGE6 PYRAMID5 NURBS27 | MAT, KINEM |
| WALL   | QUAD4 QUAD8 QUAD9 TRI3 TRI6 | MAT, KINEM, EAS, THICK, STRESS_STRAIN, GP |
| THERMO | QUAD4 QUAD8 QUAD9 TRI3 HEX8 HEX20 HEX27 TET4 TET10 WEDGE6 PYRAMID5 LINE2 | MAT |
| TRANSP | QUAD4 QUAD8 QUAD9 TRI3 TRI6 HEX8 HEX20 HEX27 TET4 TET10 WEDGE6 WEDGE15 PYRAMID5 LINE2 LINE3 | MAT, TYPE (`TYPE Std` for plain convection-diffusion) |
| FLUID  | QUAD4 QUAD8 QUAD9 TRI3 TRI6 HEX8 HEX20 HEX27 TET4 TET10 WEDGE6 WEDGE15 PYRAMID5 | MAT, NA (`NA Euler` for a fixed mesh, `NA ALE` for a moving one) |
| SOLIDSCATRA | HEX8 HEX27 TET4 TET10 | MAT, KINEM, TYPE |
| ALE2 (2D) | QUAD4 QUAD8 QUAD9 TRI3 TRI6 | MAT |
| ALE3 (3D) | HEX8 HEX20 HEX27 TET4 TET10 WEDGE6 WEDGE15 PYRAMID5 | MAT |

The tables list the ordinary cell types only. Every element type above
also declares NURBS cells, which are omitted here because they need the
separate NURBS apparatus (SHAPEFCT, KNOTVECTORS, CP lines) described
below and are not drop-in replacements.

SOLID owns NO 2D cell type on this build — 2D structural cells belong to
WALL. Getting this backwards gives
"Element 'SOLID' does not seem to know cell type 'quad4'."
(the cell type is echoed in LOWERCASE).

BEING LISTED BY `4C --parameters` IS NOT THE SAME AS WORKING. `--parameters`
describes the PARSER; some cell types parse and then die at element
evaluation. Confirmed dead on this build: WALL NURBS4/NURBS9 (the registered
type for those is WALLNURBS, and the rejection misleadingly says "Unknown
type 'WALL' of finite element"), and THERMO TRI6/WEDGE15/LINE3/NURBS4/NURBS9,
which abort with "Element shape TRI6 (6 nodes) not activated. Just do it."
The tables above list only what was executed successfully.

WALL GP is per direction for quads ("GP 2 2", QUAD9 wants "GP 3 3") but for
TRI3/TRI6 the SECOND number must be 0: "GP 3 0". "GP 3 3" on a triangle
aborts with "Unknown number of Gauss points for tri element". WALL EAS full
is 4-node only.

KINEM takes exactly "linear" or "nonlinear". "nonlinearTotLag" is what 4C
echoes back internally after parsing "nonlinear"; writing it is rejected with
"Could not parse parameter 'KINEM': invalid value 'nonlinearTotLag'. Valid
options are: linear|nonlinear".

NURBS cells of any element type additionally need PROBLEM TYPE/SHAPEFCT:
"Nurbs", a "<DIS> KNOTVECTORS" section, and control points written as
"CP <id> COORD x y z <weight>" instead of "NODE". Omitting SHAPEFCT gives
"Received discretization which is not Nurbs!"; omitting the knotvectors gives
"cannot get ele knots when filled is false". Some element/cell combinations
fail hard instead: WALLNURBS with plain NODE lines segfaults with no message
at all (exit 139). Treat NURBS as a separate exercise, not a drop-in cell
type.
An element type that does not exist at all gives
"Unknown type 'BOGUS' of finite element".
A missing required key gives
"Required value 'KINEM' not found in input line".

SOLID optional keys: PRESTRESS_TECH (none|mulf), RAD/AXI/CIR, FIBER1..3,
TECH, INTEGRATION. TECH exists on only three cell types and with different
choices each: HEX8 -> none|fbar|eas_mild|eas_full|shell_ans|shell_eas|
shell_eas_ans; WEDGE6 -> none|shell_ans|shell_eas_ans; PYRAMID5 -> none|fbar.
Writing TECH on TET4 or HEX20 is fatal. INTEGRATION is usable ONLY where the
element is written as a YAML MAP - inside ELEMENT_BLOCKS (Route B) or inside
`STRUCTURE DOMAIN: elements:` (Route C), as
`INTEGRATION: {RESIDUUM: hex_27point, MASS: hex_8point}`. On an inline
`STRUCTURE ELEMENTS` string line there is NO syntax that works, and it fails
in two different ways, and the discriminator is the SYNTAX, not how many
sub-keys you write. Plain whitespace tokens
("INTEGRATION RESIDUUM hex_27point MASS hex_8point") abort with
"Key 'INTEGRATION' cannot be found in the container."; anything using the
brace form, and the bare keyword, produce NO 4C diagnostic at all - the
process dies with
"terminate called after throwing an instance of 'std::bad_any_cast'" and
shell exit status 134. That line comes from the C++ runtime's terminate
handler in libstdc++, not from 4C, so grepping the 4C binary for it finds
nothing - the absence is expected, not evidence it cannot happen.

## 3. COMPLETE RUNNABLE DECK — 3D linear-elastic cantilever, generated mesh

```yaml
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURE DOMAIN:
  bottom_corner_point: [0.0, 0.0, 0.0]
  top_corner_point: [10.0, 1.0, 1.0]
  subdivisions: [10, 2, 2]
  elements:
    SOLID:
      HEX8:
        MAT: 1
        KINEM: nonlinear
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-10        # update-norm tolerance
  TOLRES: 1.0e-09         # residual-norm tolerance
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: 0.3
      DENS: 1.0           # REQUIRED even under Statics
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, -1.0]
    FUNCT: [0, 0, 1]
    TYPE: "Live"
DSURF-NODE TOPOLOGY:
  - "SIDE structure x- DSURFACE 1"
  - "SIDE structure x+ DSURFACE 2"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 1
      QUANTITY: "dispz"
      VALUE: 0.0
      TOLERANCE: 1.0e30    # record mode: prints abs(diff) = the true value
```

## 4. COMPLETE RUNNABLE DECK — transient heat conduction (PROBLEMTYPE Thermo)

```yaml
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
THERMO DOMAIN:
  bottom_corner_point: [0.0, 0.0, 0.0]
  top_corner_point: [1.0, 0.2, 0.2]
  subdivisions: [10, 2, 2]
  elements:
    THERMO:
      HEX8:
        MAT: 1
THERMAL DYNAMIC:
  DYNAMICTYPE: "OneStepTheta"
  TIMESTEP: 0.01
  NUMSTEP: 20
  MAXTIME: 0.2
  INITIALFIELD: "zero_field"
  TOLTEMP: 1.0e-10
  TOLRES: 1.0e-08
  MAXITER: 30
  LINEAR_SOLVER: 1
THERMAL DYNAMIC/ONESTEPTHETA:
  THETA: 1.0               # 1.0 = backward Euler, 0.5 = Crank-Nicolson
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermo_Solver"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: 1.0
      CONDUCT:
        constant: [1.0]    # tensor-typed: a bare scalar is rejected
DESIGN SURF DIRICH CONDITIONS:   # PLAIN, *not* SURF THERMO DIRICH
  - E: 1
    NUMDOF: 1                    # one temperature DOF
    ONOFF: [1]
    VAL: [100.0]
    FUNCT: [0]
DSURF-NODE TOPOLOGY:
  - "SIDE thermo x- DSURFACE 1"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 5
THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_THERMO: true
  TEMPERATURE: true
RESULT DESCRIPTION:
  - THERMAL:                     # THERMAL, not THERMO
      DIS: "thermo"
      NODE: 1
      QUANTITY: "temp"
      VALUE: 0.0
      TOLERANCE: 1.0e30
```

For contact, ask for the `contact` physics knowledge — it carries its own
complete deck plus the three sections contact adds.

## 5. Key rules
1. Section names are ALL CAPS and exact. A typo is fatal, never ignored.
2. PROBLEMTYPE selects which *_DYNAMIC section is read; the rest are ignored.
3. Each field needs ONE mesh route (inline / GEOMETRY / DOMAIN), never two.
4. Materials live under MATERIALS as `- MAT: <id>` then the material's own
   key as a nested mapping.
5. Solvers are SOLVER 1, SOLVER 2, ... and every LINEAR_SOLVER integer in the
   deck must name one that exists.
6. ONOFF, VAL and FUNCT must each have exactly NUMDOF entries - THAT is what
   4C enforces, and an inconsistent block is rejected with "Could not match
   this input". The physical count (3 for 3D structure, 2 for 2D structure, 1
   for thermal and scalar transport) is what you SHOULD write, but a
   self-consistent oversized block - NUMDOF: 3 with three-entry arrays on a
   scalar field - runs with exit 0 and the same answer. So a wrong NUMDOF is
   a silent modelling error, not a caught one.
7. Convergence is controlled by TOLDISP+TOLRES (structure) or TOLTEMP+TOLRES
   (thermal); both members of the pair must be met by default
   (NORMCOMBI_RESFDISP / NORMCOMBI_RESFTEMP: "And"). ONLY the structural
   solver echoes them: it prints `Structure-Update-Norm = ... < <your
   TOLDISP>` and `Structure-F-Norm = ... < <your TOLRES>`. The thermal
   integrator is not NOX and prints no tolerance at all - just a numiter /
   abs-res-norm / abs-temp-norm table - so do not try to confirm a thermal
   tolerance by grepping the log.
8. Runtime VTK needs BOTH the parent `IO/RUNTIME VTK OUTPUT` section AND the
   per-field one with at least one field flag set; either alone writes
   nothing. Scalar transport is the exception: it writes .vtu automatically
   and has NO runtime-VTK section at all.
9. Add a RESULT DESCRIPTION entry so a wrong answer becomes a non-zero exit
   code instead of a silent success. Run once with TOLERANCE: 1.0e30 to read
   the true value out of `abs(diff)`, then tighten.
10. Capture output through `stdbuf -oL -eL` — 4C aborts via MPI_Abort and a
    block-buffered stdout (including a plain file redirect) loses the
    diagnostic.

## 6. Common mistakes
- Wrong section name: SCATRA DYNAMIC instead of SCALAR TRANSPORT DYNAMIC;
  XFLUID DYNAMIC instead of XFLUID DYNAMIC/GENERAL; EHL DYNAMIC instead of
  ELASTO HYDRO DYNAMIC; FBI DYNAMIC instead of FLUID BEAM INTERACTION;
  PORO GEOMETRY instead of POROELASTICITY DYNAMIC;
  IO/RUNTIME VTK OUTPUT/PARTICLES — no such section, particle output is
  configured inside PARTICLE DYNAMIC.
- `SOLID QUAD4` in 2D when SOLID owns no 2D cell type — use WALL with all
  six of its keys.
- Missing DENS in a structural material (zero mass matrix = singular).
- Forgetting KINEM: nonlinear for large-deformation problems.
- In a standalone PROBLEMTYPE: Thermo run, using the THERMO-prefixed
  condition sections (DESIGN SURF THERMO DIRICH CONDITIONS). They parse
  cleanly and are then silently dropped, leaving the temperature at its
  initial value with exit 0. Use the plain DESIGN SURF DIRICH CONDITIONS.
"""

_FENICS_INPUT_GUIDE = """\
# FEniCSx (dolfinx) Script Guide

## Script Structure
```python
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem  # or NonlinearProblem
import ufl
import numpy as np

# 1. Create mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32, mesh.CellType.triangle)

# 2. Define function space
V = fem.functionspace(domain, ("Lagrange", 1))

# 3. Apply boundary conditions
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

# 4. Define weak form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(domain, default_scalar_type(1.0)) * v * ufl.dx

# 5. Solve
problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="poisson_", petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# 6. Output (XDMF for dolfinx, convert to VTU for visualization)
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
```

## Key Rules (dolfinx 0.10.0)
1. Use `fem.functionspace()` (not `FunctionSpace()`)
2. Use `basix.ufl.element()` and `mixed_element()` for mixed spaces
3. `NonlinearProblem` requires `petsc_options_prefix` kwarg
4. `NonlinearProblem.solve()` directly — no separate NewtonSolver
5. P2 functions can't write to XDMF — interpolate to P1 first
6. BCs on sub-spaces: use `fem.Function` (not constant array)
7. Always use `default_scalar_type` for PETSc compatibility
"""

_DEALII_INPUT_GUIDE = """\
# deal.II C++ Source Guide

## Source Structure
```cpp
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
// ... other includes

using namespace dealii;

int main()
{
    // 1. Grid
    Triangulation<2> triangulation;
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(5);

    // 2. FE space
    FE_Q<2> fe(1);
    DoFHandler<2> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    // 3. Sparsity + matrices
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    SparsityPattern sp; sp.copy_from(dsp);
    SparseMatrix<double> system_matrix; system_matrix.reinit(sp);

    // 4. Assembly (loop over cells)
    // 5. Boundary conditions
    // 6. Solve (CG + preconditioner)
    // 7. Output (DataOut → VTU)
}
```

## Build (CMakeLists.txt)
```cmake
cmake_minimum_required(VERSION 3.13.4)
find_package(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()
project(my_problem)
add_executable(my_problem main.cpp)
deal_ii_setup_target(my_problem)
```
Configure with `cmake -S . -B build -DDEAL_II_DIR=<the tree discover(query='list') names for deal.II>`.
Without the HINTS, find_package ignores -DDEAL_II_DIR and takes any system deal.II it finds
(measured: "Using the deal.II-9.1.1 installation found at /usr" beside a 9.8 tree); a
compile that reads /usr/include/deal.II is building against that package.

## Key Rules
1. Always refine BEFORE distributing DOFs
2. Use DynamicSparsityPattern → copy_from → SparsityPattern
3. Vector FE: FESystem<dim>(FE_Q<dim>(1), dim)
4. Boundary IDs depend on GridGenerator: hyper_cube and hyper_rectangle give every face id 0;
   with colorize = true as their last argument, hyper_rectangle (and subdivided_hyper_rectangle)
   give 0 at x = x0, 1 at x = x1, 2 at y = y0, 3 at y = y1
5. DataOut for VTU output
"""

_DUNE_INPUT_GUIDE = """\
# DUNE-fem Input Guide (Python scripts using UFL)

## Core Imports
```python
from dune.grid import structuredGrid          # structured mesh
from dune.alugrid import aluConformGrid       # unstructured mesh
from dune.fem.space import lagrange, dglagrange  # FE spaces
from dune.fem.scheme import galerkin          # solver
from dune.fem.function import gridFunction    # post-processing
from dune.ufl import DirichletBC             # boundary conditions
from ufl import *                            # UFL form language
```

## Grid Creation
```python
# Structured grid on [0,1]^2
gridView = structuredGrid([0, 0], [1, 1], [N, N])

# Unstructured from Gmsh — the reader MUST be named as a tuple.
# aluConformGrid("mesh.msh", dimgrid=2) fails with
#   RuntimeError: IOError [checkMacroGridFile:...]: Wrong file format!
# because a bare string is read as ALUGrid's own DGF macro file.
from dune.grid import reader
from dune.alugrid import aluConformGrid
gridView = aluConformGrid((reader.gmsh, "mesh.msh"), dimgrid=2)
```

NOTE: structuredGrid gives CUBE cells (quadrilateral / hexahedron).
For simplices use dune.alugrid — aluConformGrid / aluSimplexGrid on a
cartesianDomain give 2x the cells in 2D; aluSimplexGrid in 3D gives 6x
(48 tets for a 2x2x2 domain). Do NOT infer the
cell shape from the space: a dune-fem space always reports a SIMPLEX
UFL cell ("<Lagrange1 on a triangle>") even on a cube grid. Read
gridView.type instead.

## Scalar vs Vector Spaces
```python
# Scalar (1 DOF per node)
space = lagrange(gridView, order=1)           # P1
space = lagrange(gridView, order=2)           # P2

# Vector-valued (dim DOFs per node, e.g. elasticity)
space = lagrange(gridView, dimRange=2, order=1)  # 2D vector P1

# DG space
space = dglagrange(gridView, order=1)         # DG-P1
```

## Weak Forms (same as FEniCS UFL)
```python
u = TrialFunction(space)
v = TestFunction(space)

# Bilinear + linear form
a = inner(grad(u), grad(v)) * dx
L = f * v * dx
scheme = galerkin([a == L, DirichletBC(space, 0)], solver="cg")
u_h = space.interpolate(0, name="solution")
scheme.solve(target=u_h)

# Nonlinear: galerkin([F == 0]) triggers Newton automatically, but the
# form must be NONLINEAR IN THE TRIAL FUNCTION u — dune-fem
# differentiates it symbolically to build the Jacobian and needs a form
# with two arguments. Writing it with the discrete function u_h instead
# raises
#   ValueError: Integrands model requires form with at least two arguments.
F = ((1 + u**2) * inner(grad(u), grad(v)) - f * v) * dx
scheme = galerkin([F == 0, DirichletBC(space, 0)])
scheme.solve(target=u_h)      # info["iterations"] is the Newton count
```

## Time Stepping
```python
# Semi-implicit: diffusion implicit, reaction explicit
u_n = space.interpolate(u0_expr, name="u")
u = TrialFunction(space)
v = TestFunction(space)

a = (u * v / dt + D * inner(grad(u), grad(v))) * dx
L = (u_n * v / dt + reaction(u_n) * v) * dx
scheme = galerkin([a == L], solver="cg")

for step in range(n_steps):
    scheme.solve(target=u_n)  # u_n updated in-place
```

## Coupled Systems (Two Scalar Fields)
```python
# For multi-species: use two separate scalar spaces + Gauss-Seidel
space = lagrange(gridView, order=1)
u_n = space.interpolate(u0, name="u")
v_n = space.interpolate(v0, name="v")

# Scheme for u (can reference v_n as a coefficient)
scheme_u = galerkin([a_u == L_u], solver="cg")
# Scheme for v (reference u_n — updated value for Gauss-Seidel)
scheme_v = galerkin([a_v == L_v], solver="cg")

for step in range(n_steps):
    scheme_u.solve(target=u_n)  # solve u first
    scheme_v.solve(target=v_n)  # then v using new u_n
```

## Coefficient Functions & Reassembly
```python
# DUNE-fem evaluates coefficients lazily at solve time.
# If u_n appears in a form, its CURRENT values are used each solve().
# No manual reassembly needed — just update the discrete function.
```

## Output
```python
# VTK output — writes filename.vtu
gridView.writeVTK("filename", pointdata={"u": u_h, "v": v_h})

# For order > 1, pass subsampling or the file is sampled at the grid
# VERTICES only: a P2 field on a 4x4 grid wrote 25 points, and
# subsampling=2 wrote 400. Nothing warns.
gridView.writeVTK("filename", pointdata={"u": u_h}, subsampling=2)

# Access DOF values as numpy array
vals = u_h.as_numpy  # returns a numpy view (read/write)
```

## Key Pitfalls
1. First run is slow due to JIT C++ compilation (measured: 439 s for a
   cold-cache script that ended up building four ALUGrid
   hierarchical grids on a loaded box; a fully warm 8x8 Poisson run
   is 0.89 s). Watch for "DUNE-INFO: Compiling <X> (new)" on stderr.
2. For coupled systems: dimRange=2 with Newton is possible but less documented.
   Safer approach: two scalar spaces with Gauss-Seidel coupling.
3. `galerkin([F == 0])` triggers Newton automatically, but F must be
   nonlinear in the TRIAL function; `galerkin([a == L])` solves linear.
4. DOF ordering for dimRange>1: components are interleaved (u0,v0,u1,v1,...).
   Verified by execution — slice `u_h.as_numpy[i::dimRange]`.
5. No built-in time integrator — manual time loop required.
6. Set timeout >= 600s for first run to allow JIT compilation (>= 900s
   if ALUGrid is involved).
7. A DirichletBC is only applied if it is IN THE LIST given to
   `galerkin([a == L, dbc])`. Omitting it, or giving it a subDomain
   conditional that matches nothing, leaves a singular pure-Neumann
   system that `scheme.solve()` still reports as
   `{'converged': True}` — measured L2 error 7.5e+14 after 23935 CG
   iterations. Check `info['converged']` AND a magnitude bound.
8. `solver=` accepts only 'cg', 'gmres', 'bicgstab' for the default
   storage; `('suitesparse', 'umfpack')` gives a direct solve (executed
   here: `linear_iterations: 1`, same answer as CG, but the direct
   solver is its own C++ type so it costs an extra ~60 s JIT build).
   The string is not validated in Python — a wrong name fails at scheme
   construction with a message about `'fem.solver.linear.method'`.
9. `dune.fem.threading.use` defaults to 1 even when
   `dune.fem.threading.max` reports every core; `threading.useMax()`
   opts in to all of them.
10. `dune.fem.globalRefine(level, uh)` is a SILENT NO-OP on a YaspGrid
   (element count, space size and dof array all unchanged, no
   exception). It only refines-and-prolongs on
   `adaptiveLeafGridView(aluConformGrid(...))`. Refining via
   `globalRefine(level, gridView.hierarchicalGrid)` works everywhere.
11. Unrecognised entries in `parameters={...}` are silently ignored —
   a typo like `'nonlinear.maxiter'` (the key is `maxiterations`)
   leaves the default in place with no warning.
"""

_SPARTA_INPUT_GUIDE = """\
# SPARTA Input Guide (DSMC decks)

A SPARTA deck is a plain-text list of commands, one per line, executed in
order. `&` at the end of a line continues it onto the next. `#` starts a
comment. Run it with `spa_serial -in in.<case>` from the directory that holds
the deck AND every data file it names — SPARTA opens data files relative to the
current working directory. log.sparta is written to that directory.

## Command order (the parser enforces most of this)
```
seed             12345          # MANDATORY, there is no default RNG seed
dimension        2              # before create_box
boundary         p ss p         # before create_box; ONE letter sets BOTH faces
create_box       xlo xhi ylo yhi zlo zhi
create_grid      Nx Ny Nz       # 2d requires Nz = 1
species          ar.species Ar  # may come before create_box
mixture          gas Ar vstream 0 0 0 temp 273.15
global           nrho 7.07e22 fnum 7.07e11   # MUST precede create_particles
collide          vss gas ar.vss              # omit it -> collisionless gas
read_surf        <file>                      # needs the grid
surf_collide     wall diffuse 300.0 1.0
surf_modify      all collide wall            # needs surfs AND the model
create_particles gas n 0                     # n 0 == honour nrho/fnum
compute / fix / stats / stats_style / dump
timestep         1e-9                        # default is 1.0 SECOND
run              1000
```

## Boundary styles
`o` outflow, `p` periodic, `r` specular reflect, `s` surface (needs
`bound_modify <face> collide <sc-ID>`), `a` axisymmetric (lower y face only,
so the y entry must be the two-letter form, e.g. `boundary o ar p`).

## Getting numbers out
A per-grid or per-surf compute cannot go straight into `stats_style`. The
chain is compute -> fix ave/* -> compute reduce -> stats_style:
```
compute  q surf all all etot          # compute surf is ALWAYS an array
fix      fq ave/surf all 1 100 100 c_q[1]
compute  qtot reduce sum f_fq         # one fix input -> VECTOR, no bracket
stats_style step np nscoll c_qtot
```
A fix ave/* fed ONE value is a vector (`f_ID`); fed two or more it is an array
(`f_ID[1]`, `f_ID[2]`). `compute boundary` takes a mixture ID only (no group
ID) and needs `fix ave/time ... mode vector`, read as `f_ID[face]` with
xlo=1 xhi=2 ylo=3 yhi=4 (and zlo=5 zhi=6 in 3d).

## Documentation-page names are not commands
`compute_grid`, `fix_ave_surf`, `dump_image`, `surf_react_adsorb` and `suffix`
are doc FILENAMES. In a deck you write `compute <ID> grid ...`,
`fix <ID> ave/surf ...`, `dump <ID> image ...`, `surf_react <ID> adsorb ...`.

## Before trusting any number
- `ncoll` in stats_style — 0 means the gas is collisionless.
- cell Knudsen number from `compute lambda/grid ... knall`, reduced with
  `compute reduce min`, must be >= 1.
- particles per cell from `compute grid all all n`, reduced with
  `compute reduce min`, must be >= 1.
- `compute dt/grid` gives a recommended timestep; SPARTA never compares it
  with yours.

Full per-command syntax: `SpartaBackend.get_command_reference('<command>')`.
Per-physics pitfalls, deck skeletons and runnable templates:
`prepare_simulation(solver='sparta', physics='<name>')`.
"""


_FEBIO_INPUT_GUIDE = """\
# FEBio Input File Guide (.feb XML)

## Structure (v4.0)
```xml
<?xml version="1.0"?>
<febio_spec version="4.0">
  <Module type="solid"/>  <!-- solid, biphasic, heat, etc. -->
  <Control>...</Control>
  <Globals>...</Globals>
  <Material>...</Material>
  <Mesh>
    <Nodes>...</Nodes>
    <Elements>...</Elements>
    <NodeSet>...</NodeSet>
  </Mesh>
  <MeshDomains>...</MeshDomains>
  <Boundary>...</Boundary>
  <LoadData>...</LoadData>
  <Output>...</Output>
</febio_spec>
```

## Key Rules
1. Poisson's ratio: lowercase 'v' (not 'nu')
2. All indices are 1-based
3. MeshDomains links elements to materials (required in v4.0)
4. LoadData with load_controller for time-varying BCs
5. Module type determines available materials and BCs
"""
