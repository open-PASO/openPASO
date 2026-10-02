"""Auto-fetch missing backend source trees from canonical upstream.

Counterpart to source_discovery: when discovery reports MISSING or
INSTALLED_BINARY (no source on disk), this module clones the canonical
repo into <repository>/upstream_sources/<backend>/.

Design points:
  * Idempotent: clones only if destination doesn't exist.
  * Shallow by default (--depth 1) to save disk + time. Full history
    available via fetch_full=True.
  * Backgroundable: returns a Popen handle if run_in_background=True,
    so multiple clones run in parallel.
  * Includes special-case handling for DUNE (multi-repo) and
    NGSolve+netgen (two paired repos).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from .source_discovery import get_spec


_REPO = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DEST = _REPO / "upstream_sources"


# Special multi-repo bundles. DUNE needs all 5 core modules + dune-fem;
# NGSolve needs netgen too.
_BUNDLES: dict[str, list[tuple[str, str]]] = {
    "dune": [
        ("dune-common",     "https://gitlab.dune-project.org/core/dune-common.git"),
        ("dune-geometry",   "https://gitlab.dune-project.org/core/dune-geometry.git"),
        ("dune-grid",       "https://gitlab.dune-project.org/core/dune-grid.git"),
        ("dune-istl",       "https://gitlab.dune-project.org/core/dune-istl.git"),
        ("dune-localfunctions",
                            "https://gitlab.dune-project.org/core/dune-localfunctions.git"),
        ("dune-alugrid",    "https://gitlab.dune-project.org/extensions/dune-alugrid.git"),
        ("dune-fem",        "https://gitlab.dune-project.org/dune-fem/dune-fem.git"),
    ],
    "ngsolve": [
        ("netgen",  "https://github.com/NGSolve/netgen.git"),
        ("ngsolve", "https://github.com/NGSolve/ngsolve.git"),
    ],
}


def _git_clone(url: str, dest: Path, shallow: bool = True,
               recurse: bool = False,
               background: bool = False) -> Optional[subprocess.Popen]:
    """Clone url into dest. Returns Popen if background, else None
    after blocking."""
    if dest.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    if shallow:
        cmd.extend(["--depth", "1"])
    if recurse:
        cmd.append("--recursive")
    cmd.extend([url, str(dest)])
    log_path = dest.parent / f"{dest.name}.clone.log"
    if background:
        f = open(log_path, "w")
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True, stdin=subprocess.DEVNULL)
    else:
        with open(log_path, "w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            if r.returncode != 0:
                raise RuntimeError(
                    f"git clone {url} → {dest} failed (rc={r.returncode}); "
                    f"see {log_path}")
        return None


def fetch(backend: str, dest_root: Optional[Path] = None,
          shallow: bool = True, background: bool = False,
          recurse_submodules: bool = False
          ) -> tuple[Path, list[subprocess.Popen]]:
    """Clone the backend's canonical source tree.

    Returns (destination_path, list_of_running_procs). For multi-repo
    bundles (dune, ngsolve) destination is the containing directory.
    """
    spec = get_spec(backend)
    dest_root = dest_root or _DEFAULT_DEST
    procs: list[subprocess.Popen] = []

    if backend in _BUNDLES:
        bundle_dir = dest_root / backend
        bundle_dir.mkdir(parents=True, exist_ok=True)
        for sub_name, sub_url in _BUNDLES[backend]:
            sub_dest = bundle_dir / sub_name
            p = _git_clone(sub_url, sub_dest, shallow=shallow,
                          recurse=recurse_submodules, background=background)
            if p is not None:
                procs.append(p)
        return bundle_dir, procs

    dest = dest_root / backend
    p = _git_clone(spec.git_url, dest, shallow=shallow,
                   recurse=recurse_submodules, background=background)
    if p is not None:
        procs.append(p)
    return dest, procs


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Auto-fetch backend sources")
    ap.add_argument("--backend", required=True,
                    help="Backend name (kratos / febio / fenics / ngsolve / "
                         "dune / dealii / fourc / skfem) or 'all'")
    ap.add_argument("--full", action="store_true",
                    help="Full clone (no --depth 1) — much slower, larger")
    ap.add_argument("--background", action="store_true",
                    help="Spawn clone(s) in background and return immediately")
    ap.add_argument("--dest", default=str(_DEFAULT_DEST))
    ap.add_argument("--submodules", action="store_true",
                    help="--recursive for submodules (slower, much larger)")
    args = ap.parse_args()

    backends = (["skfem", "fenics", "ngsolve", "kratos", "dealii", "fourc",
                 "dune", "febio"] if args.backend == "all" else [args.backend])
    all_procs: list[tuple[str, subprocess.Popen]] = []
    for be in backends:
        try:
            dest, procs = fetch(be, dest_root=Path(args.dest),
                                shallow=not args.full,
                                background=args.background,
                                recurse_submodules=args.submodules)
            print(f"{be}: → {dest} ({len(procs)} clone(s) running)")
            for p in procs:
                all_procs.append((be, p))
        except Exception as ex:
            print(f"{be}: FAILED — {ex}", file=sys.stderr)
    if args.background and all_procs:
        print(f"\n{len(all_procs)} clone(s) running in background. "
              f"PIDs: {[p.pid for _, p in all_procs]}")
        print(f"Logs in: {Path(args.dest)}/")
