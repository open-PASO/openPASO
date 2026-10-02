"""Auto-build (compile) backend source trees.

Per-backend CMake / pip recipes that take a cloned source tree and produce
working binaries. Designed to be called from the MCP orchestrator
(`ensure_source` → build if no in-tree artifacts present).

Builds are long (minutes to hours per backend) so this module is
background-first: `build(backend, background=True)` returns a Popen
immediately and the caller polls.

Honesty:
  * No backend is auto-compiled without an explicit call. The orchestrator
    chooses when to build based on discovery status.
  * Each build writes to data/build_logs/<backend>.log so failures are
    inspectable.
  * pip-installable backends (skfem) are handled distinctly — "build" =
    `pip install -e .` into the configured env, no CMake involved.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_load_user_config = None
try:
    from .source_config import load as _load_user_config  # type: ignore
except ImportError:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from core.source_config import load as _load_user_config  # type: ignore
    except ImportError:
        pass


_REPO = Path(__file__).resolve().parent.parent.parent
try:
    from core.session_journal import state_dir as _state_dir
except ImportError:  # standalone `python src/core/source_build.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.session_journal import state_dir as _state_dir
# Build logs are runtime state, not shipped data: a pip install's site-packages is shared and
# often read-only, so they go to the user's state dir, never beside the code.
_LOG_DIR = _state_dir("build_logs")


@dataclass
class BuildRecipe:
    """A multi-step build, run sequentially in the source dir."""
    backend: str
    description: str
    # Each step: (subdir_relative_to_source_root, [shell_cmd_parts])
    steps: list[tuple[str, list[str]]]
    # Approx wall-clock on consumer hardware (informational).
    typical_minutes: int = 30
    # Environment overrides per step (None = inherit). Keys are env var names.
    env: dict[str, str] = field(default_factory=dict)


def _python_for(backend: str) -> str:
    """Pick the Python env for pip-installable backends. Consults user
    config first, then known conda envs."""
    if _load_user_config is not None:
        try:
            cfg = _load_user_config()
            bp = cfg.backends.get(backend)
            if bp and bp.python_env:
                py = bp.python_env / "bin" / "python"
                if py.exists():
                    return str(py)
        except Exception:
            pass
    # Fallbacks by env name.
    home = Path.home()
    candidates = {
        "fenics":  next((p for p in (home / "miniconda3/envs/openpaso-fenicsx/bin/python",
                                     home / "miniconda3/envs/ofa-fenicsx/bin/python") if p.exists()),
                        home / "miniconda3/envs/openpaso-fenicsx/bin/python"),
        "ngsolve": next((p for p in (home / "miniconda3/envs/openpaso-ngsolve/bin/python",
                                     home / "miniconda3/envs/ofa-ngsolve/bin/python") if p.exists()),
                        home / "miniconda3/envs/openpaso-ngsolve/bin/python"),
        "dune":    next((p for p in (home / "miniconda3/envs/openpaso-dune/bin/python",
                                     home / "miniconda3/envs/ofa-dune/bin/python") if p.exists()),
                        home / "miniconda3/envs/openpaso-dune/bin/python"),
        "skfem":   _REPO / ".venv/bin/python",
        "kratos":  _REPO / ".venv/bin/python",
    }
    cand = candidates.get(backend)
    if cand and cand.exists():
        return str(cand)
    return sys.executable


# ── Recipes ──────────────────────────────────────────────────────────────

def _recipe(backend: str) -> BuildRecipe:
    """Build recipe for `backend`. Each step's commands run in
    <source_root>/<subdir>."""
    if backend == "skfem":
        py = _python_for("skfem")
        return BuildRecipe(
            backend="skfem",
            description="pip install -e . (pure Python, no compile)",
            steps=[("", [py, "-m", "pip", "install", "-e", "."])],
            typical_minutes=1,
        )
    if backend == "fenics":
        py = _python_for("fenics")
        # dolfinx build is two-stage: C++ via CMake, Python via pip.
        prefix = str(Path(py).resolve().parent.parent)  # conda env root
        return BuildRecipe(
            backend="fenics",
            description="dolfinx: cmake C++ → ninja → pip install python",
            steps=[
                ("cpp", ["cmake", "-G", "Ninja", "-B", "build",
                         f"-DCMAKE_INSTALL_PREFIX={prefix}",
                         "-DCMAKE_BUILD_TYPE=Release"]),
                ("cpp", ["cmake", "--build", "build", "--target", "install"]),
                ("python", [py, "-m", "pip", "install",
                            "--no-build-isolation", "-e", "."]),
            ],
            typical_minutes=60,
            env={"CMAKE_PREFIX_PATH": prefix},
        )
    if backend == "ngsolve":
        py = _python_for("ngsolve")
        return BuildRecipe(
            backend="ngsolve",
            description="ngsolve+netgen: cmake superbuild",
            steps=[
                ("", ["cmake", "-G", "Ninja", "-B", "build",
                      "-DUSE_MPI=OFF", "-DCMAKE_BUILD_TYPE=Release"]),
                ("", ["cmake", "--build", "build", "--parallel"]),
                ("", [py, "-m", "pip", "install", "-e", "."]),
            ],
            typical_minutes=90,
        )
    if backend == "kratos":
        # Kratos uses a custom configure.sh; we replicate the minimal
        # CMake invocation it generates.
        return BuildRecipe(
            backend="kratos",
            description="Kratos: cmake + make Release with KratosCore + "
                        "core applications",
            steps=[
                ("", ["cmake", "-G", "Ninja", "-B", "bin/Release",
                      "-DCMAKE_BUILD_TYPE=Release",
                      "-DKRATOS_BUILD_TESTING=OFF",
                      "-DUSE_MPI=OFF",
                      "-DCMAKE_INSTALL_PREFIX=bin/Release"]),
                ("", ["cmake", "--build", "bin/Release", "--target", "install",
                      "--parallel"]),
            ],
            typical_minutes=120,
        )
    if backend == "dealii":
        return BuildRecipe(
            backend="dealii",
            description="deal.II: cmake + make release",
            steps=[
                ("", ["cmake", "-G", "Ninja", "-B", "build",
                      "-DCMAKE_BUILD_TYPE=Release",
                      "-DDEAL_II_WITH_MPI=OFF"]),
                ("", ["cmake", "--build", "build", "--parallel"]),
            ],
            typical_minutes=60,
        )
    if backend == "fourc":
        return BuildRecipe(
            backend="fourc",
            description="4C: cmake preset + ninja",
            steps=[
                ("", ["cmake", "--preset", "minimal-config", "-B", "build"]),
                ("", ["cmake", "--build", "build", "--parallel"]),
            ],
            typical_minutes=120,
        )
    if backend == "dune":
        return BuildRecipe(
            backend="dune",
            description="DUNE: dunecontrol all (builds all sibling modules)",
            steps=[("..", ["./dune-common/bin/dunecontrol",
                           "--module=dune-fem", "all"])],
            typical_minutes=60,
        )
    if backend == "febio":
        return BuildRecipe(
            backend="febio",
            description="FEBio: cmake + ninja",
            steps=[
                ("", ["cmake", "-G", "Ninja", "-B", "cmbuild",
                      "-DCMAKE_BUILD_TYPE=Release"]),
                ("", ["cmake", "--build", "cmbuild", "--parallel"]),
            ],
            typical_minutes=30,
        )
    raise ValueError(f"No build recipe for backend {backend!r}")


# ── Driver ───────────────────────────────────────────────────────────────

def build(backend: str, source_root: Path,
          background: bool = False) -> tuple[Path, Optional[subprocess.Popen]]:
    """Run the build recipe in source_root. Returns (log_path, popen).

    If background=True, returns immediately with popen running. Otherwise
    blocks until done (or step fails) and returns popen=None.
    """
    recipe = _recipe(backend)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root {source_root} does not exist")
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LOG_DIR / f"{backend}.log"

    # Combined shell script — easier to background and resume than a
    # multi-Popen approach.
    script_lines = [
        "#!/bin/bash",
        "set -e",
        f"cd {source_root}",
        f"echo '=== build {backend} ({recipe.description}) ==='",
        "echo \"start: $(date -Iseconds)\"",
    ]
    for k, v in recipe.env.items():
        script_lines.append(f"export {k}={v!r}")
    for subdir, cmd in recipe.steps:
        if subdir:
            script_lines.append(f"echo '--- cd {subdir} ---'")
            script_lines.append(f"cd {source_root / subdir if subdir != '..' else source_root.parent}")
            script_lines.append(f"echo '--- step: {' '.join(cmd)} ---'")
            script_lines.append(" ".join(_q(p) for p in cmd))
            script_lines.append(f"cd {source_root}")
        else:
            script_lines.append(f"echo '--- step: {' '.join(cmd)} ---'")
            script_lines.append(" ".join(_q(p) for p in cmd))
    script_lines.append("echo \"end: $(date -Iseconds)\"")
    script_path = _LOG_DIR / f"{backend}.sh"
    script_path.write_text("\n".join(script_lines))
    script_path.chmod(0o755)

    log_file = open(log_path, "w")
    if background:
        proc = subprocess.Popen(["bash", str(script_path)],
                                stdout=log_file, stderr=subprocess.STDOUT,
                                start_new_session=True, stdin=subprocess.DEVNULL)
        return log_path, proc
    else:
        r = subprocess.run(["bash", str(script_path)],
                           stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            raise RuntimeError(f"Build failed for {backend} "
                               f"(rc={r.returncode}); see {log_path}")
        return log_path, None


def _q(s: str) -> str:
    """Shell-quote a single argument."""
    if not s or any(c in s for c in " \t\n'\"$`\\"):
        return "'" + s.replace("'", "'\\''") + "'"
    return s


def recipe_summary(backend: str) -> dict:
    r = _recipe(backend)
    return {
        "backend": r.backend,
        "description": r.description,
        "typical_minutes": r.typical_minutes,
        "n_steps": len(r.steps),
        "steps": [(sd or ".", " ".join(c)) for sd, c in r.steps],
    }


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build a backend from source")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--source", help="Source root (else from config / "
                                      "discovery)")
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--show", action="store_true",
                    help="Show recipe without running it")
    args = ap.parse_args()

    if args.show:
        import json as _json
        print(_json.dumps(recipe_summary(args.backend), indent=2))
        sys.exit(0)

    src = Path(args.source) if args.source else None
    if src is None:
        from .source_discovery import discover
        r = discover()
        sp = r.get(args.backend, {}).get("source_path")
        if not sp:
            print(f"No source path for {args.backend}. Fetch first.",
                  file=sys.stderr)
            sys.exit(1)
        src = Path(sp)

    log, proc = build(args.backend, src, background=args.background)
    print(f"Build of {args.backend} → log: {log}")
    if proc:
        print(f"PID: {proc.pid} (running in background)")
