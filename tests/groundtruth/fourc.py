"""Ground-truth probes for 4C Multiphysics.

Fetches a single C++ source file from the 4C tree (local checkout via
``FOURC_ROOT`` or raw.githubusercontent.com) and extracts canonical input
keywords by regex.  The fetched files are cached on disk for 24 h to avoid
hammering GitHub in normal test runs.

The probes return ``None`` when neither a local checkout nor network is
available; callers translate that into the test framework's skip
mechanism (``unittest.SkipTest`` / ``pytest.skip`` -- both accept
``SkipTest``).
"""

from __future__ import annotations

import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

# Default upstream when no local checkout is configured.
_REPO = "4C-multiphysics/4C"
_BRANCH = os.environ.get("FOURC_BRANCH", "main")
_RAW_BASE = f"https://raw.githubusercontent.com/{_REPO}/{_BRANCH}"

_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", "/tmp")) / "openpaso" / "fourc-source"
_CACHE_TTL_SECONDS = 24 * 3600


def _read_local(rel_path: str) -> str | None:
    """Read ``rel_path`` from $FOURC_ROOT if that is set and the file exists."""
    root = os.environ.get("FOURC_ROOT")
    if not root:
        return None
    candidate = Path(root) / rel_path
    if not candidate.is_file():
        return None
    return candidate.read_text(encoding="utf-8", errors="replace")


def _read_cached(rel_path: str) -> str | None:
    """Read from the on-disk cache if a fresh copy is present."""
    cached = _CACHE_DIR / _BRANCH / rel_path
    if not cached.is_file():
        return None
    if (time.time() - cached.stat().st_mtime) > _CACHE_TTL_SECONDS:
        return None
    return cached.read_text(encoding="utf-8", errors="replace")


def _write_cache(rel_path: str, content: str) -> None:
    cached = _CACHE_DIR / _BRANCH / rel_path
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(content, encoding="utf-8")


def _read_network(rel_path: str) -> str | None:
    """Fetch from raw.githubusercontent.com.  Returns ``None`` on any failure."""
    url = f"{_RAW_BASE}/{rel_path}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            if resp.status != 200:
                return None
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def fetch_source(rel_path: str) -> str | None:
    """Return the contents of a 4C source file, or ``None`` if unavailable.

    Resolution order: local checkout ($FOURC_ROOT) → on-disk cache →
    network.  A successful network fetch is also written to the cache so
    subsequent runs within the TTL hit the local copy; the value returned
    on the network branch is the freshly-fetched content (not re-read from
    the cache).  Local hits bypass the cache entirely so a developer
    iterating on 4C sees their own changes immediately.
    """
    local = _read_local(rel_path)
    if local is not None:
        return local
    cached = _read_cached(rel_path)
    if cached is not None:
        return cached
    fetched = _read_network(rel_path)
    if fetched is not None:
        _write_cache(rel_path, fetched)
    return fetched


# ── Parsers ──────────────────────────────────────────────────────────────────


_SELECTION_HEAD_RE = re.compile(
    r'deprecated_selection<[^>]+>\(\s*"([A-Z_]+)"\s*,\s*\{'
)
_OPTION_PAIR_RE = re.compile(r'\{\s*"([^"]+)"\s*,\s*[A-Za-z:_0-9]+\s*\}')


def _balanced_brace_end(text: str, open_idx: int) -> int:
    """Index just past the closing ``}`` that matches the ``{`` at ``open_idx``."""
    assert text[open_idx] == "{"
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def selection_options(section_keyword: str, rel_path: str) -> set[str] | None:
    """Generic extractor: every ``{"...", ...}`` option pair inside the
    ``deprecated_selection<...>("<section_keyword>", { ... })`` block of
    ``rel_path``.  Multiple blocks in the same file are merged (4C uses
    multiple registration points -- primary + auxiliary -- for the same
    section keyword)."""
    src = fetch_source(rel_path)
    if src is None:
        return None
    options: set[str] = set()
    for head in _SELECTION_HEAD_RE.finditer(src):
        if head.group(1) != section_keyword:
            continue
        open_idx = head.end() - 1  # position of the '{' captured at end of head
        close_idx = _balanced_brace_end(src, open_idx)
        if close_idx == -1:
            continue
        body = src[open_idx:close_idx]
        options.update(p.group(1) for p in _OPTION_PAIR_RE.finditer(body))
    return options or None


def dynamictype_options() -> set[str] | None:
    """Strings 4C's input parser accepts for ``DYNAMICTYPE`` in
    ``src/inpar/4C_inpar_structure.cpp``."""
    return selection_options("DYNAMICTYPE", "src/inpar/4C_inpar_structure.cpp")


def kinem_options() -> set[str] | None:
    """Strings 4C's input parser accepts for ``KINEM`` (per-element kinematic
    nonlinearity flag) in ``src/inpar/4C_inpar_structure.cpp``."""
    return selection_options("KINEM", "src/inpar/4C_inpar_structure.cpp")


_PARAM_GET_RE = re.compile(
    # match across templated return types like std::vector<int>, so consume
    # up to the call '(' rather than the first '>'.
    r'matdata\.parameters\.(?:get|get_or)<[^(]+>\s*\(\s*"([A-Z_0-9]+)"',
)


def material_parameter_keys(rel_path: str) -> set[str] | None:
    """Extract every parameter key a 4C material class reads via
    ``matdata.parameters.get<...>("KEY")``.

    ``rel_path`` is relative to the 4C repo root, e.g.
    ``src/mat/4C_mat_plasticgtn.cpp``.
    """
    src = fetch_source(rel_path)
    if src is None:
        return None
    return set(_PARAM_GET_RE.findall(src)) or None


# Map catalog material names to the source file that declares them.  Composite
# entries that span multiple classes (e.g. "MAT_ElastHyper + ELAST_CoupNeoHooke")
# are listed against the wrapper class only; ELAST_* sub-materials live in
# separate files and are checked through their own mapping when present.
MATERIAL_SOURCE: dict[str, str] = {
    # Wrapper / standalone materials
    "MAT_Struct_StVenantKirchhoff": "src/mat/4C_mat_stvenantkirchhoff.cpp",
    "MAT_ElastHyper": "src/mat/4C_mat_elasthyper.cpp",
    "MAT_Struct_PlasticNlnLogNeoHooke": "src/mat/4C_mat_plasticnlnlogneohooke.cpp",
    "MAT_Struct_PlasticGTN": "src/mat/4C_mat_plasticgtn.cpp",
    "MAT_Struct_DruckerPrager": "src/mat/4C_mat_plasticdruckerprager.cpp",
    "MAT_Struct_PlasticLinElast": "src/mat/4C_mat_plasticlinelast.cpp",
    "MAT_PlasticElastHyper": "src/mat/4C_mat_plasticelasthyper.cpp",
    "MAT_Struct_Damage": "src/mat/4C_mat_damage.cpp",
    "MAT_crystal_plasticity": "src/mat/4C_mat_crystal_plasticity.cpp",
    # ELAST_* sub-materials -- used inside MAT_ElastHyper composites
    "ELAST_CoupNeoHooke": "src/mat/elast/4C_mat_elast_coupneohooke.cpp",
    "ELAST_CoupLogNeoHooke": "src/mat/elast/4C_mat_elast_couplogneohooke.cpp",
}


def composite_components(composite_name: str) -> list[str]:
    """Split a composite catalog name like ``MAT_ElastHyper + ELAST_CoupNeoHooke``
    into its component class names.  Returns a single-element list for
    non-composite names."""
    return [c.strip() for c in composite_name.split("+")]
