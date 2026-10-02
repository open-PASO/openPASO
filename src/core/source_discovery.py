"""Machine-wide source / binary discovery for all 8 FEM backends.

Loops through the filesystem, fingerprints each backend by characteristic
files, and returns where its source tree and binaries live. This is what
lets the cron walker actually walk source trees rather than just installed
Python wrappers.

Design points:
  * No env vars required. The walker finds trees the user already has,
    no matter where they were cloned.
  * Fast: prunes irrelevant trees (`.git/objects`, `.cache`, `node_modules`,
    `__pycache__`, conda/miniconda, .venv) before descending.
  * Honest: returns one of {INSTALLED_BINARY, SOURCE_TREE, BOTH, MISSING}
    per backend so the orchestrator knows whether fetch/build is needed.
  * Cached: results stored in data/source_discovery_cache.json so repeat
    cron ticks don't re-scan the disk.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from core.user_dirs import desktop_dirs
from core.session_journal import state_dir as _state_dir


_REPO = Path(__file__).resolve().parent.parent.parent
# The discovery cache is written at runtime; keep it out of the install (site-packages is shared
# and often read-only) and in the user's state dir instead.
_CACHE_PATH = _state_dir("cache") / "source_discovery_cache.json"

_load_user_config = None
try:
    from .source_config import load as _load_user_config  # type: ignore
except ImportError:
    # Allow `python src/core/source_discovery.py` direct invocation by
    # falling through to an absolute import.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from core.source_config import load as _load_user_config  # type: ignore
    except ImportError:
        pass


class SourceStatus(str, Enum):
    INSTALLED_BINARY = "installed_binary"  # importable / on PATH, no source on disk
    SOURCE_TREE = "source_tree"            # source on disk, no working binary
    BOTH = "both"                          # source AND working binary
    MISSING = "missing"                    # nothing on disk


@dataclass
class BackendSpec:
    """Per-backend rules for finding source / binary."""
    name: str
    # Fingerprint: a path under the candidate root that MUST exist for it
    # to count as that backend's source tree. Multiple = all required.
    source_fingerprints: list[str]
    # Directory-name regexes to filter the search (any match qualifies).
    name_patterns: list[str]
    # How to check if the backend is importable / on PATH.
    binary_check: str  # python expression evaluated in a subprocess, or shell cmd
    binary_check_kind: str  # "python" | "shell"
    # Canonical upstream for auto-fetch.
    git_url: str
    # Approx clone size in MB (informational).
    clone_size_mb: int = 0
    # Whether the backend ships pure-Python source (skfem) vs needs C++ build.
    needs_build: bool = True
    # Relative paths under source_path that indicate a working in-tree build.
    # If any exist + are non-empty, the source tree counts as a working binary too.
    build_artifact_globs: list[str] = None  # type: ignore


# ── Backend specs ────────────────────────────────────────────────────────

_SPECS: dict[str, BackendSpec] = {
    "skfem": BackendSpec(
        name="skfem",
        source_fingerprints=["skfem/__init__.py", "skfem/element/element.py"],
        name_patterns=[r"scikit[-_ ]?fem", r"skfem"],
        binary_check="import skfem; print(skfem.__version__)",
        binary_check_kind="python",
        git_url="https://github.com/kinnala/scikit-fem.git",
        clone_size_mb=15,
        needs_build=False,  # pure Python
        build_artifact_globs=[],
    ),
    "fenics": BackendSpec(
        name="fenics",
        source_fingerprints=["cpp/dolfinx/fem", "python/dolfinx/__init__.py"],
        name_patterns=[r"dolfinx", r"fenics"],
        binary_check="import dolfinx; print(dolfinx.__version__)",
        binary_check_kind="python",
        git_url="https://github.com/FEniCS/dolfinx.git",
        clone_size_mb=80,
        needs_build=True,
        build_artifact_globs=["build/cpp/libdolfinx*.so*",
                              "build-dir/cpp/libdolfinx*.so*"],
    ),
    "ngsolve": BackendSpec(
        name="ngsolve",
        # Real ngsolve repo has comp/, fem/, python/ at top level (no libsrc/).
        source_fingerprints=["comp", "fem", "python"],
        name_patterns=[r"ngsolve"],
        binary_check="import ngsolve; print(ngsolve.__version__)",
        binary_check_kind="python",
        git_url="https://github.com/NGSolve/ngsolve.git",
        clone_size_mb=120,
        needs_build=True,
        build_artifact_globs=["build/ngsolve/libngsolve*.so*"],
    ),
    "kratos": BackendSpec(
        name="kratos",
        source_fingerprints=["kratos/CMakeLists.txt", "applications"],
        name_patterns=[r"^Kratos$", r"kratos[-_]?multiphysics"],
        binary_check="import KratosMultiphysics; print(KratosMultiphysics.__version__)",
        binary_check_kind="python",
        git_url="https://github.com/KratosMultiphysics/Kratos.git",
        clone_size_mb=900,
        needs_build=True,
        build_artifact_globs=["bin/Release/libKratosCore.so",
                              "bin/Debug/libKratosCore.so"],
    ),
    "dealii": BackendSpec(
        name="dealii",
        source_fingerprints=["include/deal.II", "examples/step-1"],
        name_patterns=[r"^deal[\.\-_]?ii", r"^dealii"],
        binary_check="deal.II-VERSION || pkg-config --modversion deal.II",
        binary_check_kind="shell",
        git_url="https://github.com/dealii/dealii.git",
        clone_size_mb=600,
        needs_build=True,
        build_artifact_globs=["build/lib/libdeal_II*.so*"],
    ),
    "fourc": BackendSpec(
        name="fourc",
        source_fingerprints=["src/CMakeLists.txt", "src/inpar"],
        name_patterns=[r"^4C$", r"^four[-_]?c", r"fourc"],
        binary_check="which 4C || which fourc",
        binary_check_kind="shell",
        git_url="https://github.com/4C-multiphysics/4C.git",
        clone_size_mb=500,
        needs_build=True,
        build_artifact_globs=["build/4C", "build/4C_*"],
    ),
    "dune": BackendSpec(
        name="dune",
        # dune is multi-repo: look for any one of its core modules.
        source_fingerprints=["dune/fem"],
        name_patterns=[r"^dune[-_]?fem", r"^dune[-_]?common", r"^dune[-_]?grid",
                       r"^dune[-_]?localfunctions"],
        binary_check="import dune.fem; print('ok')",
        binary_check_kind="python",
        git_url="https://gitlab.dune-project.org/dune-fem/dune-fem.git",
        clone_size_mb=150,
        needs_build=True,
        build_artifact_globs=["build-cmake/lib/libdunefem*.so*",
                              "build-cmake/python/dune/fem"],
    ),
    "febio": BackendSpec(
        name="febio",
        # FEBioStudio is a separate repo; the core FEBio repo has FECore + FEBioMech.
        source_fingerprints=["FECore", "FEBioMech"],
        name_patterns=[r"^FEBio", r"^febio"],
        binary_check="which febio4 || which febio3 || which febio",
        binary_check_kind="shell",
        git_url="https://github.com/febiosoftware/FEBio.git",
        clone_size_mb=200,
        needs_build=True,
        build_artifact_globs=["build/bin/febio*", "cmbuild/bin/febio*"],
    ),
}


# ── Filesystem walker ────────────────────────────────────────────────────

# Directories to skip when scanning — they never contain backend source trees
# and slow the walk.
_PRUNE_DIRS = frozenset([
    ".git", ".cache", "node_modules", "__pycache__", ".venv",
    "venv", ".tox", ".mypy_cache", ".pytest_cache", "build_logs",
    "dist", "target", ".vscode", ".idea", "Trash",
    "snap", ".local",
    # conda installations are scanned separately as INSTALLED_BINARY.
    "miniconda3", "anaconda3", "conda", "mambaforge",
])

# Default scan roots (in priority order). User can extend via
# config file (see source_config.py) or OPENPASO_EXTRA_SOURCE_PATHS env var (OFA_EXTRA_SOURCE_PATHS still works).
def _default_roots() -> list[Path]:
    home = Path.home()
    candidates = [
        # Auto-cloned trees live here — highest priority.
        _REPO / "upstream_sources",
        *desktop_dirs(),
        home / "Documents",
        home / "projects",
        home / "src",
        home,  # last so deeper-first wins
        Path("/opt"),
        Path("/usr/local/src"),
        Path("/srv"),
        Path("/data"),
    ]
    # Merge in user-config scan paths.
    if _load_user_config is not None:
        try:
            user_cfg = _load_user_config()
            candidates.extend(user_cfg.scan_paths)
        except Exception:
            pass
    # Back-compat env var.
    extra = os.environ.get("OFA_EXTRA_SOURCE_PATHS", "")
    if extra:
        candidates.extend(Path(p) for p in extra.split(":") if p)
    # Dedupe while preserving order.
    seen: set[str] = set()
    out: list[Path] = []
    for c in candidates:
        if c.is_dir():
            k = str(c.resolve())
            if k not in seen:
                seen.add(k)
                out.append(c)
    return out


def _walk_for_sources(roots: list[Path], max_depth: int = 5,
                      time_budget_s: float = 60.0) -> list[Path]:
    """Walk roots, return candidate directories whose name matches ANY
    backend's name_patterns. Pruned for speed."""
    all_patterns = []
    for spec in _SPECS.values():
        all_patterns.extend(spec.name_patterns)
    combined = re.compile("|".join(f"({p})" for p in all_patterns), re.IGNORECASE)

    candidates: list[Path] = []
    deadline = time.monotonic() + time_budget_s

    def visit(d: Path, depth: int) -> None:
        if time.monotonic() > deadline:
            return
        if depth > max_depth:
            return
        try:
            entries = list(d.iterdir())
        except (PermissionError, OSError):
            return
        for e in entries:
            if not e.is_dir():
                continue
            if e.name in _PRUNE_DIRS or e.name.startswith("."):
                continue
            if combined.search(e.name):
                candidates.append(e)
            visit(e, depth + 1)

    for root in roots:
        visit(root, 0)
    return candidates


def _matches_fingerprint(candidate: Path, spec: BackendSpec) -> bool:
    """Does this candidate directory ACTUALLY contain backend source?"""
    return all((candidate / fp).exists() for fp in spec.source_fingerprints)


def _check_binary(spec: BackendSpec) -> tuple[bool, str]:
    """Try the binary_check; return (ok, version_or_error)."""
    if spec.binary_check_kind == "python":
        pythons = []
        # Probe across multiple known envs (.venv + ofa-fenicsx + ofa-dealii
        # + ofa-dune are common candidates).
        repo_venv = _REPO / ".venv/bin/python"
        if repo_venv.exists():
            pythons.append(str(repo_venv))
        home = Path.home()
        for env in (*(f"{p}-{b}" for p in ("openpaso", "ofa")
                      for b in ("fenicsx", "dealii", "dune", "ngsolve", "kratos", "febio")),):
            p = home / f"miniconda3/envs/{env}/bin/python"
            if p.exists():
                pythons.append(str(p))
        for py in pythons:
            try:
                r = subprocess.run([py, "-c", spec.binary_check],
                                   capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL)
                if r.returncode == 0:
                    return True, f"{py}: {r.stdout.strip()}"
            except Exception:
                continue
        return False, "no env has working import"
    else:
        try:
            r = subprocess.run(spec.binary_check, shell=True,
                               capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL)
            if r.returncode == 0:
                return True, r.stdout.strip() or "ok"
            return False, r.stderr.strip() or "not on PATH"
        except Exception as ex:
            return False, str(ex)


# ── Top-level API ────────────────────────────────────────────────────────

def discover(use_cache: bool = True, scan_time_budget_s: float = 90.0
             ) -> dict[str, dict]:
    """Discover source trees + binary status for every backend.

    Returns: {backend_name: {
        "status": SourceStatus,
        "source_path": str or None,
        "binary_info": str or None,
        "git_url": str,
        "needs_build": bool,
    }}
    """
    if use_cache and _CACHE_PATH.exists():
        # Cache is valid for 1 hour.
        age = time.time() - _CACHE_PATH.stat().st_mtime
        if age < 3600:
            try:
                return json.loads(_CACHE_PATH.read_text())
            except Exception:
                pass  # fall through to fresh scan

    # Layer 1: honour user-config pins (these skip the scan).
    user_pins: dict[str, dict] = {}  # backend → {source, build, python_env}
    if _load_user_config is not None:
        try:
            user_cfg = _load_user_config()
            for be, bp in user_cfg.backends.items():
                user_pins[be] = {
                    "source": bp.source if bp.source and bp.source.exists()
                              else None,
                    "build":  bp.build if bp.build and bp.build.exists()
                              else None,
                    "python_env": (bp.python_env
                                   if bp.python_env and bp.python_env.exists()
                                   else None),
                }
        except Exception:
            pass

    # Layer 2: filesystem scan for backends NOT pinned.
    unpinned = [name for name in _SPECS
                if not (user_pins.get(name) or {}).get("source")]
    if unpinned:
        roots = _default_roots()
        candidates = _walk_for_sources(roots, time_budget_s=scan_time_budget_s)
    else:
        candidates = []

    # Bucket candidates per backend by fingerprint check.
    found_sources: dict[str, Path] = {}
    for be, pins in user_pins.items():
        if pins.get("source"):
            found_sources[be] = pins["source"]
    for cand in candidates:
        for spec_name, spec in _SPECS.items():
            if spec_name in found_sources:
                continue  # already found / pinned; first match wins
            if not any(re.search(p, cand.name, re.IGNORECASE)
                       for p in spec.name_patterns):
                continue
            if _matches_fingerprint(cand, spec):
                found_sources[spec_name] = cand

    # Build result per backend.
    result: dict[str, dict] = {}
    for name, spec in _SPECS.items():
        src = found_sources.get(name)
        bin_ok, bin_info = _check_binary(spec)
        # Second-pass: if source has in-tree build artifacts, treat as binary
        # available even if not on PATH / importable.
        in_tree_artifacts: list[str] = []
        if src and spec.build_artifact_globs:
            for glob_pat in spec.build_artifact_globs:
                # Use Path.glob to expand simple wildcards.
                matches = list(src.glob(glob_pat))
                if matches:
                    in_tree_artifacts.extend(str(m) for m in matches[:3])
        if not bin_ok and in_tree_artifacts:
            bin_ok = True
            bin_info = f"in-tree build: {in_tree_artifacts[0]}"
        if src and bin_ok:
            status = SourceStatus.BOTH
        elif src:
            status = SourceStatus.SOURCE_TREE
        elif bin_ok:
            status = SourceStatus.INSTALLED_BINARY
        else:
            status = SourceStatus.MISSING
        result[name] = {
            "status": status.value,
            "source_path": str(src) if src else None,
            "binary_info": bin_info if bin_ok else None,
            "in_tree_artifacts": in_tree_artifacts,
            "git_url": spec.git_url,
            "clone_size_mb": spec.clone_size_mb,
            "needs_build": spec.needs_build,
        }

    # Cache.
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(result, indent=2))
    return result


def get_spec(backend: str) -> BackendSpec:
    if backend not in _SPECS:
        raise KeyError(f"Unknown backend {backend!r}. "
                       f"Known: {sorted(_SPECS)}")
    return _SPECS[backend]


def all_backends() -> list[str]:
    return list(_SPECS)


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Source / binary discovery")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--time-budget", type=float, default=90.0,
                    help="Max seconds for filesystem walk")
    args = ap.parse_args()

    print(f"Scanning {len(_default_roots())} roots (max "
          f"{args.time_budget:.0f}s)...", file=sys.stderr)
    res = discover(use_cache=not args.no_cache,
                   scan_time_budget_s=args.time_budget)
    print(f"\n{'backend':<10} {'status':<18} {'source':<60} binary")
    print("-" * 110)
    for name, info in res.items():
        src = info["source_path"] or "—"
        if len(src) > 58:
            src = "…" + src[-57:]
        bin_str = (info["binary_info"] or "—")[:30]
        print(f"{name:<10} {info['status']:<18} {src:<60} {bin_str}")
    print(f"\nCached to {_CACHE_PATH.relative_to(_REPO)}")
