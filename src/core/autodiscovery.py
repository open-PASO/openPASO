"""Auto-discovery of solver backends.

Probes the system for installed solvers (pip, conda, binaries, source builds).
Results are persisted so the user only confirms once.

Usage:
    from core.autodiscovery import discover_backends, load_discovered_config

    # Probe system and return findings
    findings = discover_backends()

    # Load previously confirmed config
    config = load_discovered_config()
"""

from __future__ import annotations

import json
import logging
import os

from core.user_dirs import desktop_dirs as _desktop_dirs
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("openpaso.autodiscovery")

from core.session_journal import state_dir as _state_dir
# Discovery results are runtime state AND machine-specific (which backends exist HERE):
# never write them into the install (site-packages is shared and often read-only).
CONFIG_PATH = _state_dir("cache") / "discovered_config.json"


@dataclass
class ProbeResult:
    """Result of probing for a solver backend."""

    backend: str
    found: bool
    confidence: str = "unknown"  # "definite", "likely", "possible", "not_found"
    location: str = ""           # path to binary/python/env
    version: str = ""
    install_hint: str = ""       # how to install if not found
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v}


# ── Python solver probes ─────────────────────────────────────


def _probe_pip_package(name: str, import_name: str, backend: str,
                       install_cmd: str) -> ProbeResult:
    """Check if a Python package is importable in the current venv."""
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", "unknown"))
        return ProbeResult(
            backend=backend, found=True, confidence="definite",
            location=sys.executable, version=str(version),
        )
    except ImportError:
        return ProbeResult(
            backend=backend, found=False, confidence="not_found",
            install_hint=install_cmd,
        )


def _probe_conda_env(env_name_hints: list[str], import_name: str,
                     backend: str) -> Optional[ProbeResult]:
    """Search conda environments for a package."""
    # Find conda
    conda_bin = shutil.which("conda")
    if not conda_bin:
        for candidate in ["~/miniconda3", "~/miniforge3", "~/anaconda3",
                          "~/mambaforge", "/opt/conda"]:
            candidate = os.path.expanduser(candidate)
            if Path(candidate, "bin", "conda").exists():
                conda_bin = str(Path(candidate, "bin", "conda"))
                break
    if not conda_bin:
        return None

    try:
        result = subprocess.run(
            [conda_bin, "env", "list", "--json"],
            capture_output=True, text=True, timeout=10,
            stdin=subprocess.DEVNULL,
        )
        envs = json.loads(result.stdout).get("envs", [])
    except Exception:
        return None

    for env_path in envs:
        env_name = Path(env_path).name
        if not any(hint in env_name.lower() for hint in env_name_hints):
            continue
        python = Path(env_path) / "bin" / "python"
        if not python.exists():
            python = Path(env_path) / "python.exe"  # Windows
        if not python.exists():
            continue

        # Quick import check in that env
        try:
            r = subprocess.run(
                [str(python), "-c",
                 f"import {import_name}; print(getattr({import_name}, '__version__', 'ok'))"],
                capture_output=True, text=True, timeout=5,
                stdin=subprocess.DEVNULL,
            )
            if r.returncode == 0:
                version = r.stdout.strip()
                return ProbeResult(
                    backend=backend, found=True, confidence="definite",
                    location=str(python), version=version,
                    details={"conda_env": env_name, "env_path": env_path},
                )
        except Exception:
            continue
    return None


def _probe_binary(name: str, backend: str, search_patterns: list[str],
                  version_flag: str = "--version",
                  version_marker: str = "") -> ProbeResult:
    """Search for a compiled binary."""
    # Check env var first
    env_var = f"{backend.upper()}_BINARY"
    env_path = os.environ.get(env_var, "")
    if env_path and Path(env_path).exists():
        return ProbeResult(
            backend=backend, found=True, confidence="definite",
            location=env_path, details={"source": f"env:{env_var}"},
        )

    # Check PATH
    which_result = shutil.which(name)
    if which_result:
        return ProbeResult(
            backend=backend, found=True, confidence="definite",
            location=which_result, details={"source": "PATH"},
        )

    # Search common directories
    for pattern in search_patterns:
        pattern = os.path.expanduser(pattern)
        matches = list(Path("/").glob(pattern.lstrip("/"))) if not pattern.startswith("~") \
            else list(Path(os.path.expanduser("~")).glob(pattern.replace("~/", "")))
        # Direct path check
        p = Path(pattern)
        if p.exists() and p.is_file():
            return ProbeResult(
                backend=backend, found=True, confidence="likely",
                location=str(p), details={"source": "search"},
            )

    return ProbeResult(
        backend=backend, found=False, confidence="not_found",
        install_hint=f"Build from source — see {backend} documentation",
    )


def _probe_backend_finder(backend: str, module: str, function: str) -> Optional[ProbeResult]:
    """What the backend's own finder returns, as a probe result (None if nothing)."""
    try:
        import importlib
        found = getattr(importlib.import_module(module), function)()
    except Exception:  # noqa: BLE001 -- a finder that cannot run found nothing
        return None
    if not found:
        return None
    return ProbeResult(backend=backend, found=True, confidence="definite",
                       location=str(found), details={"source": f"resolver:{function}"})


def _probe_source_root(backend: str, env_var: str,
                       search_patterns: list[str]) -> Optional[dict]:
    """Check for source code root (developer mode)."""
    root = os.environ.get(env_var, "")
    if root and Path(root).exists():
        info = {"root": root, "source": f"env:{env_var}"}
        # Check git status
        git_info = _get_git_info(root)
        if git_info:
            info["git"] = git_info
        return info

    for pattern in search_patterns:
        p = Path(os.path.expanduser(pattern))
        if p.exists() and p.is_dir():
            info = {"root": str(p), "source": "search"}
            git_info = _get_git_info(str(p))
            if git_info:
                info["git"] = git_info
            return info
    return None


def _get_git_info(repo_path: str) -> Optional[dict]:
    """Get git branch, latest tag, and status for a source repo."""
    try:
        def _git(cmd):
            r = subprocess.run(
                ["git", "-C", repo_path] + cmd,
                capture_output=True, text=True, timeout=5,
                stdin=subprocess.DEVNULL,
            )
            return r.stdout.strip() if r.returncode == 0 else None

        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
        if branch is None:
            return None  # not a git repo
        tag = _git(["describe", "--tags", "--abbrev=0"])
        dirty = _git(["status", "--porcelain"])
        log = _git(["log", "--oneline", "-3", "--decorate"])

        return {
            "branch": branch,
            "latest_tag": tag,
            "uncommitted_changes": len(dirty.splitlines()) if dirty else 0,
            "recent_commits": log,
        }
    except Exception:
        return None


# ── Main discovery ───────────────────────────────────────────


def discover_backends() -> list[ProbeResult]:
    """Probe the system for all solver backends.

    Returns a list of ProbeResults (found or not).
    """
    results = []

    # Python pip-installable solvers
    results.append(_probe_pip_package(
        "ngsolve", "ngsolve", "ngsolve", "pip install ngsolve"))
    results.append(_probe_pip_package(
        "scikit-fem", "skfem", "skfem", "pip install scikit-fem"))
    results.append(_probe_pip_package(
        "KratosMultiphysics", "KratosMultiphysics", "kratos",
        "pip install KratosMultiphysics"))

    # DUNE-fem — like FEniCSx it usually lives in a conda env (ofa-dune),
    # not the server's venv (issue #40).
    dune_pip = _probe_pip_package(
        "dune-fem", "dune.fem", "dune", "pip install dune-fem")
    if dune_pip.found:
        results.append(dune_pip)
    else:
        dune_conda = _probe_conda_env(["dune"], "dune.fem", "dune")
        results.append(dune_conda or dune_pip)

    # FEniCSx — needs conda usually
    fenics_pip = _probe_pip_package(
        "dolfinx", "dolfinx", "fenics", "conda install -c conda-forge fenics-dolfinx")
    if fenics_pip.found:
        results.append(fenics_pip)
    else:
        fenics_conda = _probe_conda_env(
            ["fenics", "dolfinx", "fenicsx"], "dolfinx", "fenics")
        results.append(fenics_conda or fenics_pip)

    # Binary solvers
    fourc_result = _probe_binary(
        "4C", "fourc",
        search_patterns=[
            "~/4C/build/4C",
            "~/4c/build/4C",
            "/opt/4c/build/4C",
            "/opt/4C/build/4C",
            # ── Non-standard source-tree builds, including the user's
            #    desktop folder in their own language — keep these
            #    last so canonical locations win. ──
            "~/4C-src/4C/build/4C",
            *(str(d / sub) for d in _desktop_dirs()
              for sub in ("4C/build/4C", "4C-src/4C/build/4C")),
        ],
    )
    # The backends' own finders know more places than these lists (FOURC_ROOT,
    # a Spack build); ask them before reporting "not found", the way the
    # deal.II probe below does, so this tool and discover agree.
    if not fourc_result.found:
        fourc_result = _probe_backend_finder("fourc", "backends.fourc.backend",
                                             "_find_fourc_binary") or fourc_result
    results.append(fourc_result)
    sparta_result = _probe_binary("spa_serial", "sparta", search_patterns=[])
    if not sparta_result.found:
        sparta_result = _probe_backend_finder("sparta", "backends.sparta.backend",
                                              "_find_sparta_binary") or sparta_result
    results.append(sparta_result)
    # deal.II is often installed via conda-forge (env layout
    # ~/miniconda3/envs/<env>/include/deal.II/), but the
    # _probe_binary helper only walks hardcoded paths. Probe
    # conda envs directly here so the rediscover_backends
    # surface stays in sync with the dealii backend's own
    # check_availability (which already finds conda envs).
    # Audit 2026-06-01: discover('list') showed dealii
    # available while rediscover_backends listed it as
    # 'Not found' — the two LLM-facing tools disagreed.
    dealii_result = _probe_binary(
        "dealii", "dealii",
        search_patterns=[
            "/usr/share/deal.II",
            "/opt/dealii",
            "~/dealii/build",
            "~/deal.II/build",
        ],
    )
    if not dealii_result.found:
        # Defer to the dealii backend's own resolver — it is the
        # single source of truth and, since 2026-06-12, picks the
        # HIGHEST-version env when several conda envs contain
        # deal.II. The inline scan this replaces returned the first
        # env in directory order, so a stale 9.1.1 env could shadow
        # a newer 9.3.2 one and the saved discovered_config.json
        # then pinned every compile to the old headers.
        try:
            from backends.dealii.backend import (
                DealiiRootOverrideError, _find_dealii)
            try:
                root = _find_dealii()
            except DealiiRootOverrideError as exc:
                # DEAL_II_DIR / DEALII_ROOT is set and is not deal.II.
                # Do NOT fall back to some other install and record it as
                # discovered — that is exactly the substitution the
                # resolver now refuses. Report not-found with the reason.
                root = None
                dealii_result = ProbeResult(
                    backend="dealii", found=False,
                    confidence="definite", location="",
                    details={"source": "resolver", "error": str(exc)})
            if root is not None:
                dealii_result = ProbeResult(
                    backend="dealii", found=True,
                    confidence="definite",
                    location=str(root),
                    details={"source": f"resolver:{root.name}"})
        except ImportError:
            pass
    results.append(dealii_result)

    # Check for source roots (developer mode)
    source_roots = {
        "fourc": ("FOURC_ROOT", ["~/4C", "/opt/4C", "~/4C-src/4C",
                                 *(str(d / sub) for d in _desktop_dirs()
                                   for sub in ("4C", "4C-src/4C"))]),
        "fenics": ("FENICS_ROOT", ["~/dolfinx", "~/fenics"]),
        "dealii": ("DEALII_ROOT", ["~/dealii", "/opt/dealii"]),
        "ngsolve": ("NGSOLVE_ROOT", ["~/ngsolve"]),
        "kratos": ("KRATOS_ROOT", ["~/Kratos"]),
    }
    for r in results:
        if r.backend in source_roots:
            env_var, patterns = source_roots[r.backend]
            src = _probe_source_root(r.backend, env_var, patterns)
            if src:
                r.details["source_root"] = src

    return results


def format_discovery(results: list[ProbeResult]) -> str:
    """Format discovery results as human-readable text."""
    found = [r for r in results if r.found]
    missing = [r for r in results if not r.found]

    lines = [f"## Backend Discovery ({len(found)} found, {len(missing)} missing)\n"]

    if found:
        lines.append("**Available:**")
        for r in found:
            line = f"  - {r.backend}: {r.confidence.upper()} at `{r.location}`"
            if r.version:
                line += f" (v{r.version})"
            # Developer mode info
            src = r.details.get("source_root", {})
            if src:
                git = src.get("git", {})
                if git:
                    line += f"\n    Source: `{src['root']}` — branch `{git.get('branch', '?')}`"
                    if git.get("latest_tag"):
                        line += f", tag `{git['latest_tag']}`"
                    if git.get("uncommitted_changes", 0) > 0:
                        line += f" (**{git['uncommitted_changes']} uncommitted changes**)"
            lines.append(line)

    if missing:
        lines.append("\n**Not found:**")
        for r in missing:
            lines.append(f"  - {r.backend}: {r.install_hint}")

    return "\n".join(lines)


# ── Persistent config ────────────────────────────────────────


def save_discovered_config(results: list[ProbeResult]) -> Path:
    """Save confirmed discovery to persistent config."""
    config = {
        "version": 1,
        "discovered_at": datetime.now().isoformat(),
        "backends": {},
    }
    for r in results:
        if r.found:
            entry = {
                "location": r.location,
                "confidence": r.confidence,
                "version": r.version,
                "confirmed_at": datetime.now().isoformat(),
            }
            if r.details.get("conda_env"):
                entry["conda_env"] = r.details["conda_env"]
            if r.details.get("source_root"):
                entry["source_root"] = r.details["source_root"]["root"]
                git = r.details["source_root"].get("git", {})
                if git.get("branch"):
                    entry["git_branch"] = git["branch"]
            config["backends"][r.backend] = entry

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    logger.info(f"Discovery config saved: {CONFIG_PATH}")
    return CONFIG_PATH


def load_discovered_config() -> Optional[dict]:
    """Load previously saved discovery config. Returns None if not found."""
    if not CONFIG_PATH.exists():
        return None
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
