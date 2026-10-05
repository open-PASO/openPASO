"""Ground-truth probes for Kratos Multiphysics.

Each sub-application of Kratos lives under ``applications/<Name>Application/``
in the ``KratosMultiphysics/Kratos`` repository.  The catalog refers to
them with a ``Kratos`` prefix (``KratosFluidDynamicsApplication``,
``KratosStructuralMechanicsApplication``, ...).  Verification therefore
maps each catalog mention to the corresponding directory entry.

This is the source-enumeration variant of the source-grep family --
no Kratos install needed.  Directory listing is fetched via the
GitHub contents API (using ``gh auth token`` when available to avoid
the unauthenticated 60-req/h rate limit) and cached locally for 24h.

Returns ``None`` when the listing is unobtainable so the test skips
rather than fails in offline / rate-limited environments.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_REPO = "KratosMultiphysics/Kratos"
_BRANCH = os.environ.get("KRATOS_BRANCH", "master")
_API_BASE = f"https://api.github.com/repos/{_REPO}"

_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", "/tmp")) / "openpaso" / "kratos-source"
_CACHE_TTL_SECONDS = 24 * 3600


def _list_applications_local() -> list[str] | None:
    """List ``applications/`` subdirectories of ``$KRATOS_ROOT``."""
    root = os.environ.get("KRATOS_ROOT")
    if not root:
        return None
    apps_dir = Path(root) / "applications"
    if not apps_dir.is_dir():
        return None
    return sorted(
        p.name for p in apps_dir.iterdir()
        if p.is_dir() and p.name.endswith("Application")
    )


def _list_applications_cached() -> list[str] | None:
    cached = _CACHE_DIR / _BRANCH / "_listing_applications.json"
    if not cached.is_file():
        return None
    if (time.time() - cached.stat().st_mtime) > _CACHE_TTL_SECONDS:
        return None
    try:
        return json.loads(cached.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None


def _list_applications_network() -> list[str] | None:
    """Fetch the ``applications/`` directory listing via the GitHub
    contents API at the configured ``$KRATOS_BRANCH``.  Falls back
    to unauthenticated if ``gh`` is not available."""
    # Pass the branch via the ``ref`` query parameter.  Without this,
    # the API returns the repository's default branch (``master``)
    # regardless of $KRATOS_BRANCH, and the cache key (which IS
    # branch-segmented) silently stores the wrong listing.
    url = (
        f"{_API_BASE}/contents/applications"
        f"?ref={urllib.parse.quote(_BRANCH, safe='')}"
    )
    token = ""
    try:
        token = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"token {token}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None
    if not isinstance(data, list):
        return None
    return sorted(
        e["name"] for e in data
        if isinstance(e, dict)
        and e.get("type") == "dir"
        and e.get("name", "").endswith("Application")
    )


def application_names() -> set[str] | None:
    """Set of valid bare ``<Name>Application`` strings, one per
    upstream ``applications/<Name>Application/`` directory.

    Catalog mentions appear in two forms and BOTH map to this set
    after normalising:

    * ``import KratosMultiphysics.<Name>Application as ...`` -- the
      actual import path the agent's generated code executes.  This
      is the load-bearing form: if ``<Name>Application`` isn't a
      real upstream directory, the import fails at runtime.
    * ``Kratos<Name>Application`` -- used in pip install hints and
      prose lists.  Maps to the same bare ``<Name>Application``
      after stripping the ``Kratos`` prefix.

    Returning bare names lets the test scanner normalise both forms
    before comparing.
    """
    local = _list_applications_local()
    if local is not None:
        return set(local)
    cached = _list_applications_cached()
    if cached is not None:
        return set(cached)
    fetched = _list_applications_network()
    if fetched is None:
        return None
    cached_path = _CACHE_DIR / _BRANCH / "_listing_applications.json"
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_text(json.dumps(fetched))
    return set(fetched)
