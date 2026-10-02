"""
Solver backend registry — discovers and manages available backends.

Backends register themselves at import time. The registry provides:
  - List of all known backends (installed or not)
  - List of available backends (installed and working)
  - Backend lookup by name or alias
  - Solver recommendation based on physics + constraints
"""

import logging
from typing import Optional

from .backend import SolverBackend, BackendStatus

logger = logging.getLogger("openpaso.registry")

# Global registry
_backends: dict[str, SolverBackend] = {}
_aliases: dict[str, str] = {}


def register_backend(backend: SolverBackend, aliases: list[str] | None = None):
    """Register a solver backend."""
    name = backend.name()
    _backends[name] = backend
    if aliases:
        for alias in aliases:
            _aliases[alias.lower()] = name
    logger.info(f"Registered backend: {backend.display_name()} ({name})")


def get_backend(name: str) -> Optional[SolverBackend]:
    """Look up a backend by name or alias."""
    key = name.lower()
    if key in _backends:
        return _backends[key]
    if key in _aliases:
        return _backends[_aliases[key]]
    return None


def list_backends() -> list[dict]:
    """List all registered backends with their status."""
    result = []
    for name, backend in _backends.items():
        status, message = backend.check_availability()
        physics = backend.supported_physics()
        result.append({
            "name": name,
            "display_name": backend.display_name(),
            "status": status.value,
            "message": message,
            "input_format": backend.input_format().value,
            "version": backend.get_version(),
            "physics_count": len(physics),
            "physics": [p.name for p in physics],
        })
    return result


def available_backends() -> list[SolverBackend]:
    """Return only backends that are installed and working."""
    return [
        b for b in _backends.values()
        if b.check_availability()[0] == BackendStatus.AVAILABLE
    ]


def all_backends() -> list[SolverBackend]:
    """Return EVERY registered backend regardless of availability.

    Use this for discovery surfaces that need to show LLMs which
    backends are known to the MCP — including ones the user
    has not installed yet. ``available_backends`` hides those,
    which is correct for "run this simulation" code paths but
    wrong for "what can this server do?" surfaces.
    """
    return list(_backends.values())


def recommend_backend(physics: str) -> Optional[SolverBackend]:
    """Recommend the best available backend for a given physics problem.

    Priority: FEniCS (easiest to use) > 4C (most capable) > deal.ii (most flexible).
    Only returns backends that are actually available and support the requested physics.
    """
    # Preference order for general use
    preference = ["fenics", "fourc", "dealii"]

    for pref_name in preference:
        backend = _backends.get(pref_name)
        if backend is None:
            continue
        status, _ = backend.check_availability()
        if status != BackendStatus.AVAILABLE:
            continue
        supported = [p.name for p in backend.supported_physics()]
        if physics.lower() in supported:
            return backend
    return None


def load_all_backends():
    """Import and register all known backends.

    Checks for a persistent discovered_config.json first (from rediscover_backends),
    then falls back to standard import-time registration.
    """
    # Load discovered config if available (sets env vars for backends)
    try:
        from core.autodiscovery import load_discovered_config
        config = load_discovered_config()
        if config and "backends" in config:
            import os
            for backend, info in config["backends"].items():
                loc = info.get("location", "")
                # A RECORDED INTERPRETER IS NOT PROMOTED to FENICS_PYTHON / DUNE_PYTHON.
                # Autodiscovery records the interpreter it probed WITH, so an old record
                # named the server's own venv as DUNE's Python; promoting it made it look
                # user-set, and every later lookup trusted it over the backend's own
                # verified finder. Those finders search the conda envs themselves. A
                # recorded 4C binary is still promoted, but only while it exists.
                if backend == "fourc" and loc and os.path.isfile(loc):
                    os.environ.setdefault("FOURC_BINARY", loc)
                src_root = info.get("source_root", "")
                # An UNBUILT checkout is not a root anyone should run from. Measured 2026-09-24:
                # discovered_config.json recorded /home/<user>/Kratos with needs_build: true, this
                # promoted it to KRATOS_ROOT, get_env_with_source_root prepended its install/ (a core
                # without ConvectionDiffusionApplication) to the PINNED interpreter's PYTHONPATH, and
                # the pin imported the wrong Kratos -- only inside the test suite, where the record
                # was read first. A source root is promoted only when the record says it is built.
                if src_root and not info.get("needs_build"):
                    env_var = f"{backend.upper()}_ROOT"
                    os.environ.setdefault(env_var, src_root)
            logger.info(f"Loaded discovered config: {len(config['backends'])} backends")
    except Exception as e:
        logger.debug(f"No discovered config: {e}")

    # Each backend module registers itself on import
    try:
        from backends.fourc.backend import register as register_fourc
        register_fourc()
    except Exception as e:
        logger.debug(f"4C backend not available: {e}")

    try:
        from backends.fenics.backend import register as register_fenics
        register_fenics()
    except Exception as e:
        logger.debug(f"FEniCS backend not available: {e}")

    try:
        from backends.dealii.backend import register as register_dealii
        register_dealii()
    except Exception as e:
        logger.debug(f"deal.ii backend not available: {e}")

    try:
        from backends.febio.backend import register as register_febio
        register_febio()
    except Exception as e:
        logger.debug(f"FEBio backend not available: {e}")

    try:
        from backends.ngsolve.backend import register as register_ngsolve
        register_ngsolve()
    except Exception as e:
        logger.debug(f"NGSolve backend not available: {e}")

    try:
        from backends.skfem.backend import register as register_skfem
        register_skfem()
    except Exception as e:
        logger.debug(f"scikit-fem backend not available: {e}")

    try:
        from backends.kratos.backend import register as register_kratos
        register_kratos()
    except Exception as e:
        logger.debug(f"Kratos backend not available: {e}")

    try:
        from backends.dune.backend import register as register_dune
        register_dune()
    except Exception as e:
        logger.debug(f"DUNE-fem backend not available: {e}")

    try:
        from backends.sparta.backend import register as register_sparta
        register_sparta()
    except Exception as e:
        logger.debug(f"SPARTA backend not available: {e}")
