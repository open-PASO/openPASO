"""Which input grammar the discovered 4C build reads.

4C's input format moves between releases, and one template text cannot serve
all of them. openPASO's 4C templates were first written for, and executed on,
a 2026.2.0 development build from March 2026, which still has the WALL element.
Upstream changed the grammar in two steps, measured on 2026-09-30 by running
every template on three builds (that development build, the 2026.2.0 release
and the 2026.3.0 release) and by comparing their `4C -p` grammar dumps:

  at the 2026.2.0 release
  * The WALL element family is gone. A 2D solid is SOLID on a QUAD4, QUAD8,
    QUAD9, TRI3 or TRI6 cell with THICKNESS and PLANE_ASSUMPTION; WALLQ4PORO became
    SOLIDPORO_PRESSURE_VELOCITY_BASED and WALLSCATRA became SOLIDSCATRA.
  * The solver keys TEKO_XML_FILE, MUELU_XML_FILE, IFPACK_XML_FILE and
    AMGNXN_XML_FILE are one key, PRECONDITIONER_XML_FILE.
  * BEAM INTERACTION has no SEARCH_STRATEGY: every beam search uses ArborX.
  * A peridynamic pre-crack is a PRE_CRACK_LINES or PRE_CRACK_PLANES list, and
    a particle is held or driven by DIRICHLET_FUNCT together with
    DIRICHLET_BOUNDARY_CONDITION_FLAGGED; SPH open boundaries moved to
    PARTICLE DYNAMIC/OPEN BOUNDARIES.
  * MAT_InelasticDefgradLinScalarAniso has no NUMSPACEDIM.
  at the 2026.3.0 release
  * MAT_Struct_ThermoStVenantK has no THERMOMAT, MIX_Constituent_ElastHyper
    no NUMMAT.
  * MAT_electrode and MAT_newman take constant coefficients (DIFF_COEF, COND,
    TRANSFERENCE_NR, THERM_FAC) and optional scaling functions in place of
    the function-id-plus-parameter-list form; OCP_PARA_NUM is gone, and
    X_MIN and X_MAX moved into an optional LITHIATION_BOUNDS group.
  * reduced_lung reads reference_radius_r0, not reference_area_A0, and its
    boundary_conditions have no pulse_width.

So there are three grammars: LEGACY (WALL present), RELEASE_2026_2 (no WALL,
but MAT_electrode's DIFF_PARA_NUM form) and CURRENT (neither). The build itself
says which one it reads: `4C -p` dumps it in under a second. The answer is kept
per build (path, size and modification time of the binary and of its
lib4C.so), in memory and in openPASO's state directory, so the dump runs once
per build. When there is no binary, or the dump cannot be read, the answer is
LEGACY, which is what the templates have always been.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

LEGACY = "wall"               # the WALL element: development builds before 2026.2.0
RELEASE_2026_2 = "2026.2"     # 4C 2026.2.0: the new elements, the older materials
CURRENT = "2026.3"            # 4C 2026.3.0
DIALECTS = (LEGACY, RELEASE_2026_2, CURRENT)

# The marker element: in the grammar of the builds before 2026.2.0, in neither release.
_MARKER = "WALL"
# The marker parameter: MAT_electrode's function-id-plus-parameter-list form, in 2026.2.0
# and before, not in 2026.3.0.
_MATERIAL_MARKER = "DIFF_PARA_NUM"


def new_elements(dialect: str) -> bool:
    """Whether a grammar has the 2026.2.0 element, solver and particle forms
    (no WALL, SOLID with THICKNESS and PLANE_ASSUMPTION, PRECONDITIONER_XML_FILE)."""
    return dialect in (RELEASE_2026_2, CURRENT)


def new_materials(dialect: str) -> bool:
    """Whether a grammar has the 2026.3.0 material and reduced-lung forms."""
    return dialect == CURRENT
# `4C -p` takes about 0.7 s; a build that needs longer is not answered.
_DUMP_TIMEOUT = 20
# A build that could not be asked (hung, crashed, not 4C) is not asked again
# for this long. generate_input asks on every call, so without this a hung
# binary would stall every template request for the whole timeout. Kept in
# memory only: a failure should not outlive the server process.
FAILED_TTL = 300.0

_memo: dict[str, str] = {}
_failed: dict[str, float] = {}


def _key(binary: Path) -> str | None:
    """Everything that decides which grammar a run of `binary` loads.

    The `4C` executable is a small launcher; the grammar lives in lib4C.so,
    which a rebuild can replace without relinking the launcher, and which the
    loader may also take from LD_LIBRARY_PATH. So the key holds the launcher,
    the library path a run uses, and the path, size and modification time of
    every lib4C.so the loader could pick: beside the launcher, in the prefix's
    lib, and in each LD_LIBRARY_PATH directory."""
    try:
        real = binary.resolve()
        ld = _library_env(real).get("LD_LIBRARY_PATH", "")
        parts = [str(real), ld]
        candidates = [real, real.parent / "lib4C.so", real.parent.parent / "lib" / "lib4C.so"]
        candidates += [Path(d) / "lib4C.so" for d in ld.split(":") if d]
        for f in candidates:
            if f == real or f.is_file():
                st = f.stat()
                parts.append(f"{f.resolve()}|{st.st_size}|{st.st_mtime_ns}")
    except OSError:
        return None
    return "|".join(parts)


def _cache_file() -> Path:
    from core.session_journal import state_dir  # noqa: PLC0415
    # -v2: the first file stored the WALL grammar as "2026.2", the name the
    # 2026.2.0 release's grammar has now.
    return state_dir("fourc") / "grammar-dialect-v2.json"


def _read_cache() -> dict[str, str]:
    try:
        data = json.loads(_cache_file().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_cache(key: str, dialect: str) -> None:
    data = _read_cache()
    data[key] = dialect
    try:
        path = _cache_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    except OSError:
        pass


def element_types(dump: str) -> set[str]:
    """The element names under `legacy_element_specs:` in a `4C -p` dump."""
    names: set[str] = set()
    inside = False
    for line in dump.splitlines():
        if line.startswith("legacy_element_specs:"):
            inside = True
            continue
        if not inside:
            continue
        if line and not line.startswith(" "):
            break
        m = re.match(r"^  ([A-Za-z0-9_]+):\s*$", line)
        if m:
            names.add(m.group(1))
    return names


def dialect_of_elements(names, material_marker: bool = False) -> str | None:
    """The grammar for a dump's element names (and whether the dump names
    MAT_electrode's DIFF_PARA_NUM); None for no names."""
    names = set(names)
    if not names:
        return None
    if _MARKER in names:
        return LEGACY
    return RELEASE_2026_2 if material_marker else CURRENT


def dialect_of_dump(dump: str) -> str | None:
    """The grammar of a `4C -p` dump; None when it lists no elements."""
    marker = re.search(rf"^\s*- name: {_MATERIAL_MARKER}\s*$", dump, re.M) is not None
    return dialect_of_elements(element_types(dump), material_marker=marker)


def _library_env(binary: Path) -> dict:
    """The environment 4C runs in, so the probe loads what a run loads.

    backends.fourc.backend.fourc_library_env is that one rule for runs and
    output conversion (it arrives with the Spack install route); where this
    tree does not have it yet, the same rule is applied here."""
    try:
        from backends.fourc.backend import fourc_library_env  # noqa: PLC0415
    except ImportError:
        fourc_library_env = None
    if fourc_library_env is not None:
        return fourc_library_env(binary)
    env = os.environ.copy()
    # A source build finds its libraries in /opt/4C-dependencies/lib. A Spack
    # build (its prefix holds .spack/spec.json) carries its own search path and
    # must not see that directory: with Spack's runpath linking it would win.
    dep_lib = "/opt/4C-dependencies/lib"
    parts = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
    try:
        from_spack = (Path(binary).resolve().parent.parent / ".spack" / "spec.json").is_file()
    except OSError:
        from_spack = False
    if from_spack:
        parts = [p for p in parts if p.rstrip("/") != dep_lib]
    elif dep_lib not in parts:
        parts.insert(0, dep_lib)
    if parts:
        env["LD_LIBRARY_PATH"] = ":".join(parts)
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env


def _dump(binary: Path) -> str | None:
    env = _library_env(binary)
    env.pop("DISPLAY", None)
    try:
        done = subprocess.run([str(binary), "-p"], stdin=subprocess.DEVNULL,
                              capture_output=True, timeout=_DUMP_TIMEOUT, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.decode("utf-8", errors="replace")


def dialect_of(binary: Path | None) -> str:
    """The grammar this 4C binary reads: LEGACY, RELEASE_2026_2 or CURRENT.

    LEGACY whenever it cannot be found out, so an unreadable build gets the
    templates it got before this module existed."""
    if binary is None:
        return LEGACY
    key = _key(Path(binary))
    if key is None:
        return LEGACY
    if key in _memo:
        return _memo[key]
    failed_at = _failed.get(key)
    if failed_at is not None and time.monotonic() - failed_at < FAILED_TTL:
        return LEGACY
    cached = _read_cache().get(key)
    if cached in DIALECTS:
        _memo[key] = cached
        return cached
    dump = _dump(Path(binary))
    found = dialect_of_dump(dump) if dump else None
    if found is None:
        _failed[key] = time.monotonic()
        return LEGACY
    _memo[key] = found
    _write_cache(key, found)
    return found
