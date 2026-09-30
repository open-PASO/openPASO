"""Guided backend setup — detect, install, verify, persist.

The user-facing journey this module powers (task #227):

    1. DETECT   What is already on this machine? (pip packages,
                conda envs, source trees, binaries — reuses
                autodiscovery + source_discovery)
    2. PLAN     For a missing backend, which install route fits this
                OS best? pip (1 min) > conda (5 min) > binary
                download > source build (30-120 min). The plan is a
                structured, human-readable recipe — nothing executes.
    3. INSTALL  Execute a chosen route. pip/conda routes run inline;
                source builds delegate to source_orchestrator
                (background by default since they take 30-120 min).
    4. VERIFY   Run the backend's smoke test (core/smoke_tests.py)
                to confirm the install actually solves something.
    5. PERSIST  Write the resolved paths into
                ~/.config/openpaso/sources.json (single config
                entry point) so every future MCP session finds the
                backend without re-discovery.

OS-awareness: each route carries per-OS metadata under
`os_support` keyed by sys.platform ("linux", "darwin"). The linux
entries are tested on this Ubuntu machine. The darwin entries are
EXTENSION POINTS — deliberately seeded with the structure (brew
deps, compiler notes) so the user's Mac Claude instance can fill
them from his 4C-on-Mac compile thread without re-designing the
schema. A route with `"verified": False` for the current OS is
still shown in plans, flagged as untested.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .source_config import load as _load_source_config  # type: ignore
    from .source_config import _GLOBAL_CONFIG_PATH  # type: ignore
    from .source_config import _LEGACY_GLOBAL_CONFIG_PATH  # type: ignore
except ImportError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.source_config import load as _load_source_config  # type: ignore
    from core.source_config import _GLOBAL_CONFIG_PATH  # type: ignore
    from core.source_config import _LEGACY_GLOBAL_CONFIG_PATH  # type: ignore


def _persist_backend_paths(backend: str, **paths: str | None) -> str:
    """Read-modify-write the GLOBAL sources.json with resolved paths.

    Only the global layer is written (repo + env layers are read-only
    user surfaces). Keys with None values are skipped; existing keys
    not mentioned survive."""
    cfg_path = _GLOBAL_CONFIG_PATH
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            raw = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            raw = {}
    if not raw and _LEGACY_GLOBAL_CONFIG_PATH.exists():
        # First write to the new path: migrate the legacy global file
        # wholesale. load() reads new-OR-legacy, never both — creating
        # the new file with a single backend would silently shadow
        # every other backend configured in the legacy file.
        try:
            raw = json.loads(_LEGACY_GLOBAL_CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            raw = {}
    raw.setdefault("backends", {})
    entry = raw["backends"].setdefault(backend, {})
    changed = False
    for key, val in paths.items():
        if val and entry.get(key) != str(val):
            entry[key] = str(val)
            changed = True
    if changed:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(raw, indent=2))
    return str(cfg_path)


def _current_os() -> str:
    """'linux' or 'darwin' (macOS). Windows is unsupported for now."""
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    return sys.platform


# ── Route catalog ────────────────────────────────────────────────────────
#
# Order matters: routes are listed fastest-first; plan_setup() proposes
# the first route whose `os_support` covers the current OS.
#
# Schema per route:
#   kind:        "pip" | "conda" | "binary" | "source"
#   description: one line
#   commands:    list of argv lists (pip/conda routes only — source
#                routes delegate to source_orchestrator)
#   os_support:  {os_key: {"verified": bool,
#                          "system_deps": [...],   # apt / brew names
#                          "notes": [...]}}        # human guidance
#   typical_minutes: rough wall-clock
#
# The darwin "notes" lists are the landing zone for the user's Mac
# compile findings (e.g. the 4C discussion-thread settings).

SETUP_ROUTES: dict[str, list[dict[str, Any]]] = {
    "skfem": [
        {
            "kind": "pip",
            "description": "pip install scikit-fem (pure Python)",
            "commands": [[sys.executable, "-m", "pip", "install",
                          "scikit-fem", "meshio"]],
            "typical_minutes": 1,
            "os_support": {
                "linux": {"verified": True, "system_deps": [], "notes": []},
                "darwin": {"verified": False, "system_deps": [],
                           "notes": ["Pure Python — expected to work "
                                     "unchanged on macOS."]},
            },
        },
    ],
    "ngsolve": [
        {
            "kind": "pip",
            "description": "pip install ngsolve (binary wheels)",
            "commands": [[sys.executable, "-m", "pip", "install", "ngsolve"]],
            "typical_minutes": 2,
            "os_support": {
                "linux": {"verified": True, "system_deps": [], "notes": []},
                "darwin": {"verified": False, "system_deps": [],
                           "notes": ["Official wheels exist for macOS "
                                     "arm64; verify with the smoke test."]},
            },
        },
        {
            "kind": "source",
            "description": "cmake superbuild from source "
                           "(only if wheels unavailable)",
            "commands": [],
            "typical_minutes": 90,
            "os_support": {
                "linux": {"verified": True,
                          "system_deps": ["cmake", "ninja-build",
                                          "g++", "libopenmpi-dev"],
                          "notes": []},
                "darwin": {"verified": False,
                           "system_deps": ["cmake", "ninja"],
                           "notes": ["EXTENSION POINT: record working "
                                     "Apple Clang / brew toolchain "
                                     "settings here."]},
            },
        },
    ],
    "kratos": [
        {
            "kind": "pip",
            "description": "pip install KratosMultiphysics-all "
                           "(the metapackage — see the glibc note)",
            "commands": [[sys.executable, "-m", "pip", "install",
                          "KratosMultiphysics-all"]],
            "typical_minutes": 5,
            "os_support": {
                "linux": {"verified": True, "system_deps": [],
                          "notes": [
                              "KratosMultiphysics-all bundles the "
                              "application wheels and resolves them as a "
                              "set. Installing KratosMultiphysics ALONE "
                              "lets pip take the newest wheel, which is "
                              "where the next problem starts.",
                              "GLIBC: the 10.4.0-10.4.2 wheels are tagged "
                              "manylinux_2_28 but their Kratos.cpython-*.so "
                              "requires GLIBC_2.32 and GLIBC_2.34. pip "
                              "accepts them on any host with glibc >= 2.28, "
                              "the install reports success, and the import "
                              "then fails with \"version `GLIBC_2.32' not "
                              "found\" — followed by a Kratos message "
                              "blaming LD_LIBRARY_PATH, which cannot fix "
                              "it. Check `ldd --version | head -1`, then TRY "
                              "THE NEWEST RELEASE FIRST: 10.4.3 installs and "
                              "imports on glibc 2.31 with nothing above "
                              "GLIBC_2.17 anywhere in the package, verified "
                              "in a clean venv. Only if the newest is also "
                              "broken fall back to 10.3.x (its wheel is "
                              "manylinux_2_17 and tops out at GLIBC_2.16, "
                              "also verified). Do NOT pin backwards as a "
                              "reflex — this note said \"on glibc < 2.32 pin "
                              "'KratosMultiphysics==10.3.0'\" until an audit "
                              "found the mis-tag had been fixed upstream; the "
                              "advice froze users on an older line for no "
                              "reason. The range that is actually broken is "
                              "10.4.0-10.4.2, not the 10.4 line.",
                              "GLIBC, second half: a symbol sweep and a "
                              "platform tag answer different questions. "
                              "10.4.3 needs nothing above GLIBC_2.17 but is "
                              "still TAGGED manylinux_2_28, so pip refuses it "
                              "below glibc 2.28 even though the binary would "
                              "load. On such a host 10.3.x is the route.",
                              "The PFEM applications have no PyPI wheel at "
                              "all — `pip download "
                              "KratosPfemFluidDynamicsApplication` answers "
                              "\"No matching distribution found\". They "
                              "need a source build.",
                              "Kratos must be installed into the SAME "
                              "interpreter that runs openPASO; the backend has "
                              "no path override.",
                          ]},
                "darwin": {"verified": False, "system_deps": [],
                           "notes": ["Wheel coverage on macOS arm64 is "
                                     "partial — verify which application "
                                     "wheels resolve."]},
            },
        },
    ],
    "dune": [
        {
            "kind": "pip",
            "description": "pip install dune-fem + mpi4py (PyPI is the "
                           "working source; conda-forge has no dune-fem)",
            # mpi4py is listed explicitly because it is an UNDECLARED
            # dependency of the dune wheels: pip install dune-fem alone
            # succeeds, and the first `import dune.fem` then stops with
            # "Please run / pip install mpi4py / before rerunning your
            # Dune script." Verified in a clean venv.
            "commands": [[sys.executable, "-m", "pip", "install",
                          "dune-fem", "mpi4py"]],
            "typical_minutes": 10,
            "os_support": {
                "linux": {"verified": True,
                          "system_deps": ["cmake", "g++", "libopenmpi-dev"],
                          "notes": [
                              "DUNE compiles C++ on demand at first use, so "
                              "a C++ toolchain and CMake must be present and "
                              "the first solve takes tens of seconds.",
                              "The JIT builds against whatever Python CMake "
                              "finds, which is not necessarily the one "
                              "running it — a conda base earlier on PATH is "
                              "the usual culprit, and the symptom is an "
                              "`undefined symbol: Py...` ImportError from a "
                              "file under <env-prefix>/.cache/dune-py. Make "
                              "the interpreter unambiguous (Python3_ROOT_DIR "
                              "+ VIRTUAL_ENV or CONDA_PREFIX) before the "
                              "first import.",
                              "mpi4py needs a working mpicc to build.",
                              "There is NO conda route: `conda search -c "
                              "conda-forge --override-channels dune-fem` "
                              "answers `No match found for: dune-fem. "
                              "Search: *dune-fem*`, and an install attempt "
                              "ends in PackagesNotFoundError. A conda ENV is "
                              "still a convenient way to get a toolchain, "
                              "but the DUNE packages inside it come from "
                              "PyPI either way.",
                          ]},
                "darwin": {"verified": False, "system_deps": ["cmake"],
                           "notes": ["JIT compilation at first use needs "
                                     "Xcode CLT. EXTENSION POINT — not "
                                     "verified on macOS."]},
            },
        },
    ],
    "fenics": [
        {
            "kind": "conda",
            "description": "conda-forge fenics-dolfinx in a dedicated env",
            "commands": [["conda", "create", "-n", "openpaso-fenicsx", "-y",
                          "-c", "conda-forge", "fenics-dolfinx",
                          "pyvista", "python=3.12"]],
            "typical_minutes": 10,
            "os_support": {
                "linux": {"verified": True, "system_deps": ["conda"],
                          "notes": []},
                "darwin": {"verified": False, "system_deps": ["conda"],
                           "notes": ["conda-forge ships osx-arm64 "
                                     "dolfinx; MPI is the usual snag — "
                                     "record working env pins here."]},
            },
        },
    ],
    "dealii": [
        {
            "kind": "binary",
            "description": "apt install libdeal.ii-dev (Ubuntu/Debian)",
            # No inline commands: `sudo apt` inside an MCP subprocess
            # would hang on the password prompt. The user runs it
            # themselves; the notes carry the exact line.
            "commands": [],
            "typical_minutes": 5,
            "os_support": {
                "linux": {"verified": False,
                          "system_deps": [],
                          "notes": [
                              "Run manually: sudo apt install -y "
                              "libdeal.ii-dev.",
                              "CHECK THE VERSION before relying on it — "
                              "Ubuntu 20.04's package is 9.1.1, old enough "
                              "that many current APIs are missing: `grep "
                              "DEAL_II_PACKAGE_VERSION "
                              "/usr/include/deal.II/base/config.h`. Worse, "
                              "a system deal.II is what find_package silently "
                              "falls back to when DEAL_II_DIR is wrong or "
                              "unset, so an old one present alongside a newer "
                              "build is an active hazard, not just a spare.",
                              "Distribution packages are Release builds, so "
                              "every deal.II Assert is compiled out and no "
                              "assertion message can ever appear. Build from "
                              "source with -DCMAKE_BUILD_TYPE=DebugRelease if "
                              "you need those diagnostics.",
                          ]},
                "darwin": {"verified": False, "system_deps": [],
                           "notes": ["brew install dealii exists; "
                                     "EXTENSION POINT for verified "
                                     "settings."]},
            },
        },
        {
            "kind": "conda",
            "description": "conda-forge deal.II in a dedicated env",
            "commands": [["conda", "create", "-n", "openpaso-dealii", "-y",
                          "-c", "conda-forge", "dealii", "cmake",
                          "ninja", "cxx-compiler"]],
            "typical_minutes": 15,
            "os_support": {
                "linux": {"verified": True, "system_deps": ["conda"],
                          "notes": []},
                "darwin": {"verified": False, "system_deps": ["conda"],
                           "notes": []},
            },
        },
    ],
    "fourc": [
        {
            "kind": "source",
            "description": "clone + cmake preset + ninja "
                           "(no binary distribution exists)",
            "commands": [],
            "typical_minutes": 120,
            "os_support": {
                "linux": {"verified": True,
                          "system_deps": ["cmake", "ninja-build", "g++",
                                          "libopenmpi-dev",
                                          "libtrilinos-*-dev (or the "
                                          "4C-dependencies bundle)"],
                          "notes": [
                              "The build produces <4C-root>/build/4C plus "
                              "lib4C.so beside it, and links ~50 further "
                              "libraries from the dependency prefix.",
                              "SHARED LIBRARIES: read the list off the binary "
                              "rather than assembling it — `readelf -d "
                              "<4C-binary> | grep -E 'RUNPATH|RPATH'`. If a "
                              "RUNPATH is embedded, no LD_LIBRARY_PATH is "
                              "needed at all. If not, THREE directories are "
                              "required, not two: the BUILD directory (for "
                              "lib4C.so), the dependency prefix's lib (for "
                              "libteuchoscomm.so.16 and friends), and "
                              "<build>/_deps/ryml-build (for libryml.so.0.9.0, "
                              "a FetchContent sub-build that lives in neither "
                              "of the other two). This note said BOTH until an "
                              "audit ran every combination against a "
                              "RUNPATH-stripped copy: the dependency prefix "
                              "alone fails on lib4C.so, the build directory "
                              "alone fails on libteuchoscomm.so.16, and the "
                              "two together still fail on libryml.so.0.9.0.",
                              "4C takes `4C <input> <output-prefix>`; the "
                              "output prefix is mandatory and omitting it "
                              "aborts with a core dump rather than a usage "
                              "message.",
                          ]},
                "darwin": {"verified": False,
                           "system_deps": ["cmake", "ninja",
                                           "open-mpi (brew)"],
                           "notes": [
                               "EXTENSION POINT — the user documented "
                               "working macOS compile settings in a 4C "
                               "discussion thread (compiler pins + "
                               "changes needed on Mac). His Mac Claude "
                               "instance should replace this note with "
                               "the verified step list: brew deps, "
                               "CMake cache entries, and any source "
                               "patches."]},
            },
        },
    ],
    "febio": [
        {
            "kind": "binary",
            "description": "official installer from febio.org/downloads "
                           "(set FEBIO_BINARY afterwards)",
            "commands": [],
            "typical_minutes": 5,
            "os_support": {
                "linux": {"verified": True, "system_deps": [],
                          "notes": ["febio.org downloads require a (free) "
                                    "registered account — there are no "
                                    "direct download URLs, so this step "
                                    "is interactive. After unpacking, set "
                                    "FEBIO_BINARY to the febio4 "
                                    "executable; verify then persists "
                                    "that path."]},
                "darwin": {"verified": False, "system_deps": [],
                           "notes": ["Official mac installer exists."]},
            },
        },
        {
            "kind": "source",
            "description": "cmake + ninja from FEBio GitHub",
            "commands": [],
            "typical_minutes": 30,
            "os_support": {
                "linux": {"verified": True,
                          "system_deps": ["cmake", "ninja-build", "g++"],
                          "notes": [
                              "Working configure line without an Intel "
                              "toolchain: cmake -S FEBio -B FEBio/cbuild "
                              "-DUSE_MKL=OFF "
                              "-DCMAKE_EXE_LINKER_FLAGS='-fopenmp -ldl' "
                              "-DCMAKE_SHARED_LINKER_FLAGS='-fopenmp -ldl'",
                              "USE_MKL=OFF changes which linear solvers "
                              "exist. Detect it from a run, not from a "
                              "feature list: `febio4 -info` prints no build "
                              "options at all, so grepping it for MKL or MMG "
                              "returns nothing either way. The line "
                              "`Default linear solver: skyline` is what an "
                              "MKL-less build reports.",
                              "After building, either set FEBIO_BINARY to "
                              "the executable or symlink it somewhere "
                              "discovery looks; `febio4 -v` is NOT a valid "
                              "flag (use -info).",
                          ]},
                "darwin": {"verified": False, "system_deps": [],
                           "notes": []},
            },
        },
    ],
    "sparta": [
        {
            "kind": "source",
            "description": "make serial (or make mpi) from the SPARTA repo "
                           "— no wheel or package exists",
            # No inline commands: the build leaves the binary in src/ and
            # does not install it anywhere, so the useful part is the
            # follow-up (SPARTA_BINARY), not a one-shot command.
            "commands": [],
            "typical_minutes": 20,
            "os_support": {
                "linux": {"verified": True,
                          "system_deps": ["g++", "make", "libopenmpi-dev"],
                          "notes": [
                              "git clone https://github.com/sparta/"
                              "sparta.git && cd sparta/src && make serial",
                              "The org is `sparta`, NOT `sparta-sparta`. This "
                              "note carried the latter until an audit ran it: "
                              "`git ls-remote https://github.com/"
                              "sparta-sparta/sparta.git` answers `remote: "
                              "Repository not found.` (HTTP 404).",
                              "The binary is left in that same src directory "
                              "as spa_serial (or spa_mpi for `make mpi`) and "
                              "is NOT put on PATH — set SPARTA_BINARY to it, "
                              "or let discovery find the conventional "
                              "location.",
                              "Optional packages are compile-time. `make ps` "
                              "lists them as `Installed YES/NO: package "
                              "<NAME>`. KOKKOS is NOT in a default build, and "
                              "without it `-kokkos on`, the `package kokkos` "
                              "command and every kk-suffixed style are "
                              "refused.",
                              "Example decks reference data files from the "
                              "distribution's data/ and examples/ trees; "
                              "point SPARTA_DATA_DIR at them or run from a "
                              "directory that has them.",
                          ]},
                "darwin": {"verified": False, "system_deps": [],
                           "notes": ["EXTENSION POINT — not verified on "
                                     "macOS."]},
            },
        },
    ],
}


# ── Detection ────────────────────────────────────────────────────────────

def detect_backend(backend: str) -> dict:
    """What does this machine already have for `backend`?

    Combines the runtime registry view (is the backend importable /
    runnable right now?) with the source-tree view (is there a local
    clone / build dir?)."""
    out: dict[str, Any] = {"backend": backend, "available": False,
                           "source_tree": None, "build": None,
                           "details": ""}
    try:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        b = get_backend(backend)
        if b is not None and hasattr(b, "check_availability"):
            status, detail = b.check_availability()
            status_name = getattr(status, "name", str(status))
            out["available"] = status_name.upper() in (
                "AVAILABLE", "READY", "OK")
            out["details"] = f"{status_name}: {detail}"
    except Exception as e:  # registry import is environment-sensitive
        out["details"] = f"registry probe failed: {e}"

    try:
        cfg = _load_source_config()
        bp = cfg.backends.get(backend)
        if bp is not None:
            out["source_tree"] = str(bp.source) if bp.source else None
            out["build"] = str(bp.build) if bp.build else None
    except Exception:
        pass
    return out


def setup_status() -> list[dict]:
    """Per-backend one-row status across all known backends."""
    return [detect_backend(be) for be in sorted(SETUP_ROUTES)]


# ── Planning ─────────────────────────────────────────────────────────────

def plan_setup(backend: str, prefer: str | None = None) -> dict:
    """Return the recommended (or `prefer`-requested) install route for
    the current OS — structured, nothing executed."""
    if backend not in SETUP_ROUTES:
        return {"error": f"Unknown backend {backend!r}. Known: "
                         f"{sorted(SETUP_ROUTES)}"}
    osk = _current_os()
    state = detect_backend(backend)
    routes = SETUP_ROUTES[backend]
    if prefer:
        matching = [r for r in routes if r["kind"] == prefer]
        if not matching:
            return {"backend": backend, "os": osk, "state": state,
                    "error": f"No {prefer!r} route for {backend}. "
                             f"Available route kinds: "
                             f"{[r['kind'] for r in routes]}"}
        routes = matching

    chosen = None
    for r in routes:
        if osk in r["os_support"]:
            chosen = r
            break
    if chosen is None:
        return {"backend": backend, "os": osk, "state": state,
                "error": f"No setup route for {backend} on {osk}."}

    os_meta = chosen["os_support"][osk]
    return {
        "backend": backend,
        "os": osk,
        "already_available": state["available"],
        "state": state,
        "route": {
            "kind": chosen["kind"],
            "description": chosen["description"],
            "commands": [" ".join(c) for c in chosen["commands"]],
            "system_deps": os_meta.get("system_deps", []),
            "notes": os_meta.get("notes", []),
            "verified_on_this_os": os_meta.get("verified", False),
            "typical_minutes": chosen["typical_minutes"],
        },
        "alternatives": [
            {"kind": r["kind"], "description": r["description"]}
            for r in SETUP_ROUTES[backend] if r is not chosen
        ],
    }


# ── Execution ────────────────────────────────────────────────────────────

def execute_setup(backend: str, route_kind: str | None = None,
                  timeout: int = 1800) -> dict:
    """Execute the planned route inline (pip / conda) or delegate to the
    source orchestrator (source). Returns a structured result including
    the smoke-test verdict and what got persisted."""
    plan = plan_setup(backend, prefer=route_kind)
    if "error" in plan:
        return plan
    route = plan["route"]
    osk = plan["os"]
    result: dict[str, Any] = {"backend": backend, "os": osk,
                              "route": route["kind"], "steps": []}

    if plan["already_available"] and route_kind is None:
        result["status"] = "already_available"
        result["steps"].append({"step": "detect",
                                "detail": plan["state"]["details"]})
        # still verify + persist so a pre-existing install gets wired in
        result.update(_verify_and_persist(backend))
        return result

    if route["kind"] in ("pip", "conda"):
        # Always execute the commands of the route the plan chose —
        # routes[0] is not necessarily the one supported on this OS.
        for cmd in [c for r in SETUP_ROUTES[backend]
                    if r["kind"] == route["kind"] for c in r["commands"]]:
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=timeout, stdin=subprocess.DEVNULL)
                result["steps"].append({
                    "step": " ".join(cmd)[:120],
                    "rc": proc.returncode,
                    "elapsed_s": round(time.time() - t0, 1),
                    "stderr_tail": proc.stderr[-400:] if proc.returncode
                    else "",
                })
                if proc.returncode != 0:
                    result["status"] = "install_failed"
                    return result
            except FileNotFoundError as e:
                result["steps"].append({"step": " ".join(cmd)[:120],
                                        "error": str(e)})
                result["status"] = "tool_missing"
                return result
            except subprocess.TimeoutExpired:
                result["steps"].append({"step": " ".join(cmd)[:120],
                                        "error": f"timeout {timeout}s"})
                result["status"] = "timeout"
                return result
    elif route["kind"] == "source":
        from core.source_orchestrator import ensure_source
        r = ensure_source(backend, fetch_if_missing=True,
                          build_if_no_binary=True, background=True)
        result["steps"].append({"step": "ensure_source", "result": r})
        result["status"] = "build_started_background"
        result["note"] = ("Source build runs in the background "
                          f"(typically ~{route['typical_minutes']} min). "
                          "Re-run setup_backend(action='verify', "
                          f"solver='{backend}') when it finishes.")
        return result
    elif route["kind"] == "binary":
        result["status"] = "manual_step_required"
        result["note"] = route["description"]
        result["instructions"] = route["notes"] or [
            "Install the binary, then call "
            f"setup_backend(action='verify', solver='{backend}')."]
        return result

    result.update(_verify_and_persist(backend))
    return result


def _verify_and_persist(backend: str) -> dict:
    """Smoke-test the backend; on success persist paths to sources.json.

    Honesty contract: 'verified' requires a genuinely passing smoke
    test. Backends without a smoke test are only reported
    'installed_unverified' if detection says they are actually
    available — otherwise 'not_installed', and nothing is persisted."""
    out: dict[str, Any] = {}
    try:
        from core.smoke_tests import SMOKE_TESTS
        fn = SMOKE_TESTS.get(backend)
        if fn is None:
            out["smoke"] = {"skipped": f"no smoke test for {backend}"}
            state = detect_backend(backend)
            if state["available"]:
                out["status"] = "installed_unverified"
            else:
                out["status"] = "not_installed"
                out["detail"] = state["details"]
        else:
            sr = fn()
            out["smoke"] = sr.to_dict() if hasattr(sr, "to_dict") else vars(sr)
            out["status"] = ("verified" if getattr(sr, "passed", False)
                             else "smoke_failed")
    except Exception as e:
        out["smoke"] = {"error": str(e)}
        out["status"] = "smoke_errored"

    if out.get("status") in ("verified", "installed_unverified"):
        try:
            from core.source_discovery import discover
            disc = discover(use_cache=False)
            info = disc.get(backend) if isinstance(disc, dict) else None
            src = None
            if info is not None:
                src = getattr(info, "source_path", None) or \
                    (info.get("source_path") if isinstance(info, dict)
                     else None)
            binary = os.environ.get(f"{backend.upper()}_BINARY")
            if binary and not Path(binary).exists():
                binary = None
            path = _persist_backend_paths(backend, source=src,
                                          binary=binary)
            out["persisted"] = f"{path} updated for {backend}"
        except Exception as e:
            out["persisted"] = f"persist skipped: {e}"
    return out


# ── Rendering ────────────────────────────────────────────────────────────

def render_status_markdown() -> str:
    rows = setup_status()
    lines = ["| backend | available | source tree | build |",
             "|---------|-----------|-------------|-------|"]
    for r in rows:
        lines.append(
            f"| {r['backend']} "
            f"| {'YES' if r['available'] else 'no'} "
            f"| {r['source_tree'] or '—'} "
            f"| {r['build'] or '—'} |")
    return "\n".join(lines)


if __name__ == "__main__":  # manual probe
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["status", "plan", "install",
                                      "verify"])
    p.add_argument("--backend", default="")
    p.add_argument("--route", default=None)
    a = p.parse_args()
    if a.action == "status":
        print(render_status_markdown())
    elif a.action == "plan":
        print(json.dumps(plan_setup(a.backend, prefer=a.route), indent=2))
    elif a.action == "install":
        print(json.dumps(execute_setup(a.backend, route_kind=a.route),
                         indent=2, default=str))
    elif a.action == "verify":
        print(json.dumps(_verify_and_persist(a.backend), indent=2,
                         default=str))
