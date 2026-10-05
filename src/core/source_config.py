"""Single entry point for the user to declare where backend sources / binaries /
Python envs live.

Three layers, last-wins:

  1. Built-in defaults (hardcoded scan roots)
  2. Global config:       ~/.config/openpaso/sources.json
  3. Repo config:         <repo>/.openpaso.json
  4. Session config:      $OPENPASO_SOURCE_CONFIG=/path/to/file.json  (env var; the older OFA_ name still works)
  5. Direct env-var pins: $OPENPASO_<BACKEND>_SOURCE / _BUILD / _PYTHON_ENV
                          $OPENPASO_EXTRA_SOURCE_PATHS  (colon-separated)

Config schema (JSON):
{
  "scan_paths": ["~/projects", "/opt/fem"],
  "backends": {
    "kratos": {
      "source":     "/home/user/Kratos",
      "build":      "/home/user/Kratos/bin/Release",
      "python_env": "/home/user/miniconda3/envs/openpaso-kratos"
    },
    "fourc": {
      "source":     "/home/user/4C",
      "build":      "/home/user/4C/build"
    },
    "fenics": {
      "python_env": "/home/user/miniconda3/envs/openpaso-fenicsx"
    }
  }
}

All keys are optional. Per-backend pins skip the filesystem scan for that
backend (much faster, and lets users point at non-standard paths).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.user_dirs import desktop_dirs


_REPO = Path(__file__).resolve().parent.parent.parent
_GLOBAL_CONFIG_PATH = Path.home() / ".config" / "openpaso" / "sources.json"
# Pre-rebrand location (project renamed Open-FEM-agent -> openPASO
# 2026-06-04). The rename sed moved this constant to ~/.config/openpaso/
# but existing installs keep their config at the old path — without a
# fallback the rename silently orphans a working setup (found
# 2026-06-12: setup_status() showed every backend pathless although
# ~/.config/open-fem-agent/sources.json was fully populated). Readers
# fall back to the legacy path; writers always use the NEW path.
_LEGACY_GLOBAL_CONFIG_PATH = (Path.home() / ".config" / "open-fem-agent"
                              / "sources.json")
_REPO_CONFIG_PATH = _REPO / ".openpaso.json"
_ENV_CONFIG_VAR = "OFA_SOURCE_CONFIG"
_EXTRA_PATHS_VAR = "OFA_EXTRA_SOURCE_PATHS"


@dataclass
class BackendPaths:
    source: Optional[Path] = None
    build: Optional[Path] = None
    python_env: Optional[Path] = None


@dataclass
class SourceConfig:
    scan_paths: list[Path] = field(default_factory=list)
    backends: dict[str, BackendPaths] = field(default_factory=dict)
    # Where did each piece of config come from? Useful for diagnostics.
    sources_used: list[str] = field(default_factory=list)


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(p))).resolve()


def _load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as ex:
        print(f"WARN: failed to parse {path}: {ex}")
        return None


def _merge_into(cfg: SourceConfig, raw: dict, label: str) -> None:
    """Merge a parsed config dict into cfg, recording the source."""
    cfg.sources_used.append(label)
    for p in raw.get("scan_paths", []) or []:
        cfg.scan_paths.append(_expand(p))
    for be, paths in (raw.get("backends", {}) or {}).items():
        bp = cfg.backends.setdefault(be, BackendPaths())
        if paths.get("source"):
            bp.source = _expand(paths["source"])
        if paths.get("build"):
            bp.build = _expand(paths["build"])
        if paths.get("python_env"):
            bp.python_env = _expand(paths["python_env"])


def load() -> SourceConfig:
    """Load the merged config from all layers."""
    cfg = SourceConfig()

    # Layer 1: built-in defaults — none (defaults live in source_discovery).

    # Layer 2: global config (new path first, legacy pre-rebrand
    # path as fallback so existing installs keep working)
    raw = _load_json(_GLOBAL_CONFIG_PATH)
    if raw:
        _merge_into(cfg, raw, f"global:{_GLOBAL_CONFIG_PATH}")
    else:
        raw = _load_json(_LEGACY_GLOBAL_CONFIG_PATH)
        if raw:
            _merge_into(cfg, raw,
                        f"global-legacy:{_LEGACY_GLOBAL_CONFIG_PATH}")

    # Layer 3: repo config
    raw = _load_json(_REPO_CONFIG_PATH)
    if raw:
        _merge_into(cfg, raw, f"repo:{_REPO_CONFIG_PATH}")

    # Layer 4: session config (env var pointing at a file)
    sess_path = os.environ.get(_ENV_CONFIG_VAR)
    if sess_path:
        raw = _load_json(Path(sess_path))
        if raw:
            _merge_into(cfg, raw, f"session:{sess_path}")

    # Layer 5: direct env-var pins
    extra = os.environ.get(_EXTRA_PATHS_VAR, "")
    if extra:
        cfg.sources_used.append(f"env:{_EXTRA_PATHS_VAR}")
        for p in extra.split(":"):
            if p:
                cfg.scan_paths.append(_expand(p))
    for be in ("kratos", "fourc", "dealii", "fenics", "ngsolve", "skfem",
               "dune", "febio"):
        for kind, attr in (("SOURCE", "source"), ("BUILD", "build"),
                           ("PYTHON_ENV", "python_env")):
            var = f"OFA_{be.upper()}_{kind}"
            val = os.environ.get(var)
            if val:
                cfg.sources_used.append(f"env:{var}")
                bp = cfg.backends.setdefault(be, BackendPaths())
                setattr(bp, attr, _expand(val))

    return cfg


def example_config() -> str:
    """Return an example config JSON the user can copy and edit."""
    example = {
        "_comment": "openPASO source config — edit paths to match "
                    "your machine. All entries are optional.",
        "scan_paths": [
            "~/Desktop",
            "~/projects",
            "/opt/fem"
        ],
        "backends": {
            "kratos": {
                "source":     "~/Kratos",
                "build":      "~/Kratos/bin/Release",
                "python_env": "~/miniconda3/envs/openpaso-kratos"
            },
            "fourc": {
                "source": "~/4C",
                "build":  "~/4C/build"
            },
            "dealii": {
                "source": "~/dealii-src",
                "build":  "~/dealii-src/build"
            },
            "fenics": {
                "python_env": "~/miniconda3/envs/openpaso-fenicsx"
            }
        }
    }
    return json.dumps(example, indent=2)


def write_example(path: Optional[Path] = None) -> Path:
    """Write the example config to the global location (creating parent
    dirs) if it doesn't exist. Returns the path."""
    target = path or _GLOBAL_CONFIG_PATH
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(example_config())
    return target


def init_from_discovery(path: Optional[Path] = None,
                        overwrite: bool = False) -> Path:
    """Run live discovery, then write a config file pre-populated with the
    paths actually found on this machine. Gives the user a working config
    to edit instead of a generic example."""
    # Local import to avoid circular dependency at module load.
    from . import source_discovery as _sd

    target = path or _GLOBAL_CONFIG_PATH
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {target}. "
                              f"Pass overwrite=True to replace.")

    result = _sd.discover(use_cache=False, scan_time_budget_s=90.0)

    cfg: dict = {
        "_comment": "openPASO source config — generated from "
                    "live discovery. Edit freely. Per-backend keys "
                    "{source, build, python_env} are all optional.",
        "scan_paths": [*(str(d) for d in desktop_dirs()), "~/projects"],
        "backends": {},
    }
    for backend, info in result.items():
        entry: dict = {}
        if info.get("source_path"):
            entry["source"] = info["source_path"]
        # Derive build dir if we detected an in-tree artifact.
        for art in info.get("in_tree_artifacts") or []:
            # Walk up to the directory containing 'build' or 'cmbuild'.
            p = Path(art)
            for parent in p.parents:
                if parent.name in ("build", "cmbuild", "build-cmake",
                                   "build-dir", "Release", "Debug"):
                    entry["build"] = str(parent)
                    break
            if "build" in entry:
                break
        if entry:
            cfg["backends"][backend] = entry

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg, indent=2))
    return target


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Source config inspector")
    ap.add_argument("--show", action="store_true",
                    help="Show merged config from all layers")
    ap.add_argument("--example", action="store_true",
                    help="Print an example config JSON")
    ap.add_argument("--write-example", action="store_true",
                    help=f"Write example config to {_GLOBAL_CONFIG_PATH}")
    ap.add_argument("--init", action="store_true",
                    help="Run discovery + write a personalised config with "
                         "actual paths found on this machine")
    ap.add_argument("--overwrite", action="store_true",
                    help="With --init: overwrite existing global config")
    ap.add_argument("--paths", action="store_true",
                    help="Show the config-file lookup chain")
    args = ap.parse_args()

    if args.paths:
        print("Config lookup chain (last-wins):")
        print(f"  1. global:  {_GLOBAL_CONFIG_PATH}"
              f" {'(exists)' if _GLOBAL_CONFIG_PATH.is_file() else '(missing)'}")
        print(f"  2. repo:    {_REPO_CONFIG_PATH}"
              f" {'(exists)' if _REPO_CONFIG_PATH.is_file() else '(missing)'}")
        sess = os.environ.get(_ENV_CONFIG_VAR)
        print(f"  3. session: ${_ENV_CONFIG_VAR}"
              f" {'= ' + sess if sess else '(unset)'}")
        print(f"  4. env vars: OPENPASO_<BACKEND>_<SOURCE|BUILD|PYTHON_ENV>, "
              f"OPENPASO_EXTRA_SOURCE_PATHS")

    if args.example:
        print(example_config())

    if args.write_example:
        p = write_example()
        print(f"Wrote example config to {p}")

    if args.init:
        try:
            p = init_from_discovery(overwrite=args.overwrite)
            print(f"Wrote personalised config to {p}")
            print(Path(p).read_text())
        except FileExistsError as ex:
            print(f"ERROR: {ex}")
            print(f"Use --overwrite to replace.")

    if args.show or not (args.example or args.write_example or args.paths):
        cfg = load()
        layers = cfg.sources_used or "(none — built-in defaults only)"
        print(f"Layers loaded: {layers}")
        print(f"\nExtra scan paths ({len(cfg.scan_paths)}):")
        for p in cfg.scan_paths:
            print(f"  {p}")
        print(f"\nBackend pins ({len(cfg.backends)}):")
        for be, bp in cfg.backends.items():
            print(f"  {be}:")
            if bp.source:     print(f"    source:     {bp.source}")
            if bp.build:      print(f"    build:      {bp.build}")
            if bp.python_env: print(f"    python_env: {bp.python_env}")
