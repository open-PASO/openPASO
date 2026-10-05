"""
MCP tools for developer mode — read, modify, and extend solver source code.

This is what makes the openPASO a development PARTNER, not just an
operator. The agent can read solver source, understand architecture,
suggest modifications, and even implement new features.
"""

import json
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from core.registry import get_backend, available_backends


# ── Environment variables for source roots ──────────────────────────────
# Users can point the MCP at source trees by setting these env vars
# in .claude/settings.json under mcpServers.openpaso.env:
#
#   FOURC_ROOT:   /path/to/4C                  (C++, CMake)
#   DEALII_ROOT:  /path/to/dealii              (C++, CMake)
#   FENICS_ROOT:  /path/to/dolfinx             (Python/C++, CMake)
#   NGSOLVE_ROOT: /path/to/ngsolve             (Python/C++, CMake)
#   KRATOS_ROOT:  /path/to/Kratos              (C++/Python, CMake)
#   DUNE_ROOT:    /path/to/dune-fem            (Python/C++, CMake)
#   SKFEM_ROOT:   /path/to/scikit-fem          (pure Python)
#
# When set, the agent can browse, modify, and rebuild the solver source.
# When unset, the solver is assumed pip-installed (read-only).

def _source_root(env_var: str, fallback: str = "") -> str:
    """Get source root from environment, with descriptive fallback."""
    val = os.environ.get(env_var, "")
    if val and Path(val).is_dir():
        return val
    return fallback


def _make_key_dirs(root: str, dirs: dict) -> dict:
    """Build key_dirs dict, adding source root entries if available."""
    result = dict(dirs)
    if root and Path(root).is_dir():
        result["source_root"] = root
    return result


# Source code locations for each backend
_SOURCE_LOCATIONS = {
    "fourc": {
        "root": _source_root("FOURC_ROOT", "not set — set FOURC_ROOT env var to enable source access"),
        "source_env_var": "FOURC_ROOT",
        "build_system": "CMake",
        "language": "C++20",
        "build_command": "cd build && cmake --build . -j$(nproc)",
        "test_command": "cd build && ctest -j$(nproc)",
        "key_dirs": _make_key_dirs(os.environ.get("FOURC_ROOT", ""), {
            "source": "src/",
            "tests": "tests/input_files/",
            "modules": "src/",
            "solid_elements": "src/solid_ele/",
            "fluid_elements": "src/fluid_ele/",
            "materials": "src/mat/",
            "input_parser": "src/inpar/",
            "io": "src/core/io/",
            "templates": "openpaso/src/backends/fourc/",
            "kinematics_enum": "src/inpar/4C_inpar_structure.hpp (Inpar::Solid::KinemType)",
            "solid_element_factory": "src/solid_ele/4C_solid_ele_factory.cpp",
            "plane_2d_helpers": "src/solid_ele/4C_solid_ele_calc_lib_plane.{hpp,cpp}",
            "valid_materials": "src/global_legacy_module/4C_global_legacy_module_validmaterials.cpp",
        }),
        "extension_points": [
            "New material: add to src/mat/4C_mat_<name>.hpp/cpp.  Register the input "
            "parser group in src/global_legacy_module/4C_global_legacy_module_validmaterials.cpp "
            "and an enum value in src/core/legacy_enum_definitions/4C_legacy_enum_definitions_materials.cpp. "
            "Then add a catalog entry in openpaso/src/backends/fourc/generators/<physics>.py.",
            "New solid element kinematic mode (e.g. axisymmetric — not yet in 4C as of the "
            "tree the MCP was built against; grep -r 'axisym' src/ returns nothing): "
            "(1) extend the Inpar::Solid::KinemType enum in src/inpar/4C_inpar_structure.hpp "
            "(currently `vague, linear, nonlinearTotLag`); "
            "(2) the input mapping that exposes the new mode to users lives in "
            "src/solid_ele/4C_solid_ele.cpp inside get_kinem_type_input_spec(), which uses "
            "deprecated_selection<Inpar::Solid::KinemType>(\"KINEM\", {...}) — append a "
            "{kinem_type_string(KinemType::<new_mode>), KinemType::<new_mode>} entry there.  "
            "Also extend the switch in kinem_type_string() in src/inpar/4C_inpar_structure.cpp "
            "(which is only a stringifier, not the parser) so the new enum value has a label; "
            "(3) add a template specialization "
            "SolidCalculationFormulation<celltype, KinemType::<new_mode>, ...> in "
            "src/solid_ele/4C_solid_ele_factory.cpp.  The factory uses concrete-celltype-list "
            "constraints (e.g. `requires(celltype == Core::FE::CellType::hex8 || celltype == "
            "Core::FE::CellType::pyramid5)`); for axisymmetric, list the 2D cell types "
            "(quad4, quad8, quad9, tri3, tri6); "
            "(4) implement the new kinematics in a sibling library like "
            "src/solid_ele/4C_solid_ele_calc_lib_<mode>.hpp.  For axisymmetric specifically, "
            "mirror the existing plane-strain/plane-stress library "
            "(src/solid_ele/4C_solid_ele_calc_lib_plane.{hpp,cpp}) — its helpers are constrained "
            "to 2D via `requires(Core::FE::dim<celltype> == 2)` and already provide a 2D→3D "
            "tensor lift via transform_to_3d().  Axisymmetric differs from plane strain in two "
            "places: (a) the B-matrix gains the hoop row ε_θθ = u_r/r so the strain tensor is "
            "ε = [ε_rr, ε_zz, 2ε_rz, ε_θθ] and the constitutive update uses an "
            "axisymmetric-shaped 4-component lift in transform_to_3d; (b) the volume integration "
            "weight becomes 2πr·dA — multiply the Gauss weights by r at every integration point "
            "before assembling the stiffness and the internal force; "
            "(5) add an integration-test input under tests/input_files/ exercising the new mode "
            "(e.g. pressurized cylinder cross-section under axial restraint for an "
            "axisymmetric quad4 patch); "
            "(6) declare the new mode in the agent's catalog by extending the relevant "
            "physics generator (e.g. openpaso/src/backends/fourc/generators/solid_mechanics.py) "
            "so the catalog-consistency tests stay green.",
            "New fluid element: same pattern under src/fluid_ele/ with its own factory.",
            "New physics module: create src/<module_name>/ with CMakeLists.txt declaring "
            "the module's internal dependencies via four_c_add_internal_dependency() — "
            "see src/solid_ele/CMakeLists.txt for the canonical pattern.",
            "New boundary condition: register a ConditionDefinition via "
            "src/core/fem/src/condition/4C_fem_condition_definition.hpp from the relevant "
            "src/inpar/ module's set_valid_conditions() so the input parser exposes the new key.",
            "New inline mesh generator (Python side only): add a builder to "
            "openpaso/src/backends/fourc/inline_mesh.py and surface it via the relevant "
            "physics generator's template; no 4C source changes needed.",
        ],
    },
    "fenics": {
        "root": _source_root("FENICS_ROOT", "conda-installed — set FENICS_ROOT to enable source access"),
        "source_env_var": "FENICS_ROOT",
        "build_system": "CMake (from source) or conda/Spack/apt (pre-built; there is no PyPI package)",
        "language": "Python + C++ (PETSc)",
        "build_command": "conda create -n openpaso-fenicsx -y -c conda-forge fenics-dolfinx pyvista python=3.12  "
                         "# or from source: cmake .. && make install in cpp/build, then "
                         "pip install --check-build-dependencies --no-build-isolation . in python/",
        "reference_files": "Official demos at https://docs.fenicsproject.org/dolfinx/main/python/demos.html",
        "key_dirs": _make_key_dirs(os.environ.get("FENICS_ROOT", ""), {
            "templates": "openpaso/src/backends/fenics/backend.py",
        }),
        "extension_points": [
            "New physics: add template function _<physics>_<variant>() in backend.py",
            "New mesh type: use Gmsh Python API in template",
            "New solver config: modify petsc_options in template",
            "Custom weak form: modify UFL expressions in template",
        ],
    },
    "dealii": {
        "root": _source_root("DEALII_ROOT", "system-installed — set DEALII_ROOT to enable source access"),
        "source_env_var": "DEALII_ROOT",
        "build_system": "CMake",
        "language": "C++17",
        "build_command": "cmake -DDEAL_II_DIR=/usr/share/deal.II . && make",
        "reference_files": "step tutorials (the numbering has gaps: 88 up to step-97 in deal.II 9.7, 91 up to "
                           "step-104 in 9.8) at /usr/share/doc/libdeal.ii-doc/examples/step-*/ (the copy there is "
                           "the distribution package's: libdeal.ii-doc 9.1.1 on Ubuntu 20.04 ships 62, up to step-64)",
        "key_dirs": _make_key_dirs(os.environ.get("DEALII_ROOT", ""), {
            "templates": "openpaso/src/backends/dealii/backend.py",
            "include": "/usr/include/deal.II/",
            "step_tutorials": "/usr/share/doc/libdeal.ii-doc/examples/",
        }),
        "extension_points": [
            "New physics: add C++ template function in backend.py",
            "New element: use deal.II FE_Q, FE_DGQ, etc.",
            "New boundary condition: modify setup_system() in template",
            "Adaptive refinement: add Kelly error estimator loop",
        ],
    },
    "ngsolve": {
        "root": _source_root("NGSOLVE_ROOT", "pip-installed — set NGSOLVE_ROOT to enable source access"),
        "source_env_var": "NGSOLVE_ROOT",
        "build_system": "CMake (from source) or pip (pre-built wheels)",
        "language": "Python + C++ (Netgen)",
        "build_command": "pip install ngsolve  # or: cd build && cmake --build .",
        "reference_files": "i-tutorials at https://docu.ngsolve.org/latest/i-tutorials/index.html",
        "key_dirs": _make_key_dirs(os.environ.get("NGSOLVE_ROOT", ""), {
            "templates": "openpaso/src/backends/ngsolve/backend.py",
        }),
        "extension_points": [
            "New physics: add template function in backend.py",
            "New mesh: use Netgen geometry (SplineGeometry, CSG)",
            "New element: use H1, HDiv, HCurl, L2 spaces",
            "DG methods: use FacetFE spaces",
        ],
    },
    "skfem": {
        "root": _source_root("SKFEM_ROOT", "pip-installed — set SKFEM_ROOT to enable source access"),
        "source_env_var": "SKFEM_ROOT",
        "build_system": "pip (pure Python)",
        "language": "Python (numpy/scipy)",
        "build_command": "pip install scikit-fem",
        "key_dirs": _make_key_dirs(os.environ.get("SKFEM_ROOT", ""), {
            "templates": "openpaso/src/backends/skfem/backend.py",
        }),
        "extension_points": [
            "New physics: define @BilinearForm and @LinearForm",
            "New element: use skfem.element.* (ElementTriP1, ElementQuad2, etc.)",
            "New mesh: use MeshTri, MeshQuad, MeshHex, or import from meshio",
            "Custom assembly: direct matrix manipulation via scipy.sparse",
        ],
    },
    "kratos": {
        "root": _source_root("KRATOS_ROOT", "pip-installed — set KRATOS_ROOT to enable source access"),
        "source_env_var": "KRATOS_ROOT",
        "build_system": "CMake (from source) or pip (pre-built wheels)",
        "language": "C++ with Python bindings",
        "build_command": "pip install KratosMultiphysics  # or: cd build && cmake --build .",
        "source_repo": "https://github.com/KratosMultiphysics/Kratos",
        "reference_files": "Examples at https://kratosmultiphysics.github.io/Examples/",
        "key_dirs": _make_key_dirs(os.environ.get("KRATOS_ROOT", ""), {
            "templates": "openpaso/src/backends/kratos/backend.py",
            "applications": "kratos/applications/",
        }),
        "extension_points": [
            "New physics: add Kratos Application (C++) or use existing",
            "New element: implement in C++ and register with KRATOS_REGISTER_ELEMENT in the application's Register() "
            "(KM.KratosGlobals only looks up variables, flags and constitutive laws)",
            "CoSimulation: use CoSimulationApplication for multi-code coupling",
            "Custom process: derive from KM.Process",
        ],
    },
    "dune": {
        "root": _source_root("DUNE_ROOT", "pip-installed — set DUNE_ROOT to enable source access"),
        "source_env_var": "DUNE_ROOT",
        "build_system": "CMake (from source) or pip (JIT compiles C++)",
        "language": "Python (UFL) + C++ (JIT compiled)",
        "build_command": "pip install dune-fem  # or: cd build && cmake --build .",
        "reference_files": "Tutorials at https://www.dune-project.org/sphinx/content/sphinx/dune-fem/",
        "key_dirs": _make_key_dirs(os.environ.get("DUNE_ROOT", ""), {
            "templates": "openpaso/src/backends/dune/backend.py",
        }),
        "extension_points": [
            "New physics: define UFL weak forms (same syntax as FEniCS)",
            "New mesh: use structuredGrid, ALUGrid, or Gmsh",
            "DG methods: use dune-fem-dg module",
            "Adaptive refinement: built-in h-adaptivity via mark/adapt",
        ],
    },
}


def register_developer_tools(mcp: FastMCP):

    @mcp.tool()
    def get_solver_architecture(solver: str) -> str:
        """Get the source code architecture and extension points for a solver backend.

        This enables DEVELOPER MODE: understand how to modify and extend a solver.
        Returns source location, build system, language, key directories, and
        specific extension points for adding new physics, elements, or materials.

        Args:
            solver: Backend name (e.g. 'fourc', 'fenics', 'dealii', 'ngsolve')
        """
        if solver not in _SOURCE_LOCATIONS:
            available = ", ".join(sorted(_SOURCE_LOCATIONS.keys()))
            return f"Unknown solver '{solver}'. Available: {available}"

        info = _SOURCE_LOCATIONS[solver]
        lines = [
            f"## {solver} — Developer Architecture\n",
            f"**Language:** {info['language']}",
            f"**Build system:** {info['build_system']}",
            f"**Source root:** {info['root']}",
            f"**Build command:** {info.get('build_command', 'N/A')}",
        ]

        if "test_command" in info:
            lines.append(f"**Test command:** {info['test_command']}")

        if "source_repo" in info:
            lines.append(f"**Source repo:** {info['source_repo']}")

        if "key_dirs" in info:
            lines.append("\n### Key Directories")
            for name, path in info["key_dirs"].items():
                lines.append(f"- **{name}:** `{path}`")

        if "extension_points" in info:
            lines.append("\n### Extension Points")
            for ep in info["extension_points"]:
                lines.append(f"- {ep}")

        return "\n".join(lines)

    @mcp.tool()
    def list_solver_source_files(solver: str, pattern: str = "*.py") -> str:
        """List source files matching a pattern in a solver's template directory.

        Args:
            solver: Backend name
            pattern: Glob pattern (default: '*.py')
        """
        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        # Find the backend's source files in our repo
        base = Path(__file__).resolve().parents[1] / "backends" / solver
        if not base.exists():
            # Try alternative names
            for alt in [solver, solver.replace("-", ""), solver.replace("_", "")]:
                base = Path(__file__).resolve().parents[1] / "backends" / alt
                if base.exists():
                    break

        if not base.exists():
            return f"Backend source directory not found for '{solver}'"

        files = sorted(base.rglob(pattern))
        if not files:
            return f"No files matching '{pattern}' in {base}"

        lines = [f"## {solver} source files matching '{pattern}'\n"]
        for f in files:
            rel = f.relative_to(base)
            size = f.stat().st_size
            lines.append(f"- `{rel}` ({size} bytes)")

        return "\n".join(lines)

    @mcp.tool()
    def get_solver_capabilities_matrix() -> str:
        """Get a comprehensive capabilities matrix across all available backends.

        Shows which physics, elements, and features each solver supports.
        Essential for cross-solver comparison and solver selection.
        """
        backends = available_backends()
        if not backends:
            return "No backends available."

        lines = [
            "## Solver Capabilities Matrix\n",
            "| Solver | Physics | Element Types | Input Format | VTU Output | Coupling |",
            "|--------|---------|---------------|-------------|------------|----------|",
        ]

        for b in backends:
            physics = ", ".join(p.name for p in b.supported_physics())
            elements = set()
            for p in b.supported_physics():
                elements.update(p.element_types)
            elem_str = ", ".join(sorted(elements)[:3])
            if len(elements) > 3:
                elem_str += f" (+{len(elements)-3})"
            fmt = b.input_format().value
            vtu = "Yes" if b.name() != "febio" else "Needs binary"
            coupling = "TSI" if b.name() == "fourc" else "DN" if b.name() in ("fenics", "fourc") else "Via MCP"
            lines.append(
                f"| {b.display_name()} | {physics} | {elem_str} | {fmt} | {vtu} | {coupling} |"
            )

        lines.extend([
            "",
            f"**Total:** {len(backends)} available backends",
            "",
            "### Solver Selection Guide",
            "- **Fastest setup:** scikit-fem (pure Python, pip install)",
            "- **Most physics:** 4C (14 physics types, native TSI/FSI)",
            "- **Best for prototyping:** FEniCSx or DUNE-fem (UFL forms)",
            "- **Largest community:** Kratos (54 applications)",
            "- **Best for high-order:** NGSolve (arbitrary p)",
            "- **Best for coupling:** 4C (native) or via MCP agent (any pair)",
        ])

        return "\n".join(lines)

    @mcp.tool()
    def discover_sources(refresh: bool = False) -> str:
        """Find source trees + binaries for every backend by scanning the
        local filesystem. Reads ~/.config/openpaso/sources.json for
        pinned paths first, then walks $HOME / /opt / /usr/local/src.

        Each backend gets one of four statuses:
          - both:             source + working binary detected
          - source_tree:      source on disk, no built binary yet
          - installed_binary: importable / on PATH, no source on disk
          - missing:          neither found — call fetch_source to clone

        Args:
            refresh: bypass the 1-hour cache and re-scan disk
        """
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
        from core.source_discovery import discover as _discover
        res = _discover(use_cache=not refresh)
        lines = ["## Source / binary status\n",
                 "| Backend | Status | Source | Binary |",
                 "|---|---|---|---|"]
        for be, info in res.items():
            src = info.get("source_path") or "—"
            bn = info.get("binary_info") or "—"
            lines.append(f"| {be} | {info['status']} | `{src}` | `{bn}` |")
        lines.append("\nEdit `~/.config/openpaso/sources.json` to pin paths "
                     "globally, or set `$OPENPASO_SOURCE_CONFIG=<file>` for "
                     "session-local overrides.")
        return "\n".join(lines)

    @mcp.tool()
    def fetch_source(backend: str, background: bool = True) -> str:
        """Clone the canonical upstream source for a backend into
        upstream_sources/<backend>/. Idempotent: skips if directory exists.
        Shallow clone (--depth 1) by default; pass background=False to
        block until done.
        """
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
        from core.source_fetch import fetch as _fetch
        try:
            dest, procs = _fetch(backend, background=background, shallow=True)
        except Exception as ex:
            return f"FAILED: {ex}"
        pids = [p.pid for p in procs] or ["(already cloned)"]
        return f"{backend} → {dest}  pids: {pids}"

    @mcp.tool()
    def build_source(backend: str, background: bool = True) -> str:
        """Run the build recipe (CMake + ninja, or pip install -e .) for a
        backend's source tree. Source must already exist (call fetch_source
        first). Builds are LONG (minutes to hours); use background=True.
        """
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
        from core.source_discovery import discover as _discover
        from core.source_build import build as _build, recipe_summary
        res = _discover()
        src = res.get(backend, {}).get("source_path")
        if not src:
            return (f"No source path for {backend}. Run fetch_source first.")
        recipe = recipe_summary(backend)
        log, proc = _build(backend, _Path(src), background=background)
        return (f"## Building {backend}\n"
                f"**Recipe:** {recipe['description']}  "
                f"(~{recipe['typical_minutes']} min)\n"
                f"**Log:** `{log}`\n"
                f"**PID:** {proc.pid if proc else 'completed'}")

    @mcp.tool()
    def ensure_source(backend: str, build_if_no_binary: bool = False) -> str:
        """One-shot: discover → fetch if missing → optionally build if no
        binary. Single entry point for getting a backend usable.
        """
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
        from core.source_orchestrator import ensure_source as _ensure
        r = _ensure(backend, fetch_if_missing=True,
                    build_if_no_binary=build_if_no_binary, background=True)
        import json as _json
        return f"## ensure_source({backend})\n\n```json\n{_json.dumps(r, indent=2)}\n```"
