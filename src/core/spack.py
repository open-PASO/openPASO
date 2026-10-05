"""Spack as a way to install the solvers, and a place to find them afterwards.

Spack (https://spack.io) builds a solver with its whole dependency tree into a
prefix of its own. Four things live here:

  * where Spack is: ``$SPACK_ROOT/bin/spack``, then ``spack`` on PATH, then the
    Spack that last built a solver for openPASO (remembered in openPASO's state
    directory, because an AI app may start the server without SPACK_ROOT or
    PATH), then ``~/spack/bin/spack`` (where the clone command below puts it).
    A machine can have several: an install uses the first one that is new
    enough, and a lookup asks all of them;
  * where a Spack-built solver is: ``installed_prefix("dealii")`` asks Spack for
    the prefix of an installed package. The solver finders call it as one
    discovery step among the others;
  * what a Spack build brings along: ``build_prefix`` tells a Spack-built file
    from any other, and ``mpi_launcher`` finds the mpirun of the MPI the build
    links against, from the spec.json Spack writes into every prefix;
  * how openPASO's own recipes reach Spack. Upstream Spack has no recipe for
    several solvers (4C, SPARTA the DSMC code, ...), and its deal.II recipe
    leaves Spack's compiler wrapper in deal.IIConfig.cmake. openPASO ships the
    missing and the fixed recipes as a Spack package repository with the
    namespace ``openpaso`` under ``data/spack``. It is never added to the
    user's Spack configuration: openPASO's own Spack calls pass a scope that
    holds only that repository (``spack -C <scope>``), so the user's other
    Spack work keeps upstream's recipes. Spack treats such a scope as read
    only, so what Spack itself writes still goes to the user's configuration,
    as for any command: where no compiler is configured, the first
    concretization records the ones it finds in ~/.spack/packages.yaml.

Nothing here builds anything. The build is ``openpaso install <solver> --via
spack`` in the foreground, or ``setup_backend`` in the background.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

SPACK_RELEASE = "v1.2.2"
SPACK_CLONE = ["git", "clone", "--depth=2", f"--branch={SPACK_RELEASE}",
               "https://github.com/spack/spack.git"]
RECIPE_NAMESPACE = "openpaso"
MINIMUM_VERSION = (1, 0)
# Discovery must not hang on a Spack that is locked or on a slow file system;
# a lookup that takes longer counts as "not found" (0.6 s is typical).
LOOKUP_TIMEOUT = 15
# How long a "not installed" answer is kept. Every availability check asks for
# each Spack-built solver, and one `spack find` costs about 0.6 s per Spack, so
# without this a machine with Spack but none of these builds pays seconds on
# every discover. Kept short, and dropped by forget(), because setup_backend
# starts a build and asks again in the same server process when it finishes.
MISSING_TTL = 60.0
# How long a found build is kept before Spack is asked again, so a server that
# runs for days picks up a newer build (a rebuild with new variants gets a new
# prefix) without a restart.
FOUND_TTL = 300.0

_found: dict[str, tuple[Path, float]] = {}
_missing: dict[str, float] = {}


def _state(name: str) -> Path:
    from core.session_journal import state_dir  # noqa: PLC0415
    return state_dir("spack") / name


def spack_env(loads_recipes: bool = False) -> dict[str, str]:
    """Environment for Spack calls. A call that loads openPASO's recipes
    (install, info) writes no bytecode, because in a pip install they sit in
    site-packages, where it would outlive `pip uninstall`. A lookup loads no
    recipes, and keeping Spack's own bytecode makes it about three times
    faster."""
    env = os.environ.copy()
    if loads_recipes:
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def remember_executable(executable: str) -> None:
    try:
        path = _state("spack-executable")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(executable + "\n")
    except OSError:
        pass


def _remembered_executable() -> Path | None:
    try:
        text = _state("spack-executable").read_text().strip()
    except OSError:
        return None
    return Path(text) if text else None


def spack_candidates() -> list[str]:
    """Every Spack on this machine, in the order above; the same file once."""
    paths: list[Path] = []
    root = os.environ.get("SPACK_ROOT")
    if root:
        paths.append(Path(root) / "bin" / "spack")
    on_path = shutil.which("spack")
    if on_path:
        paths.append(Path(on_path))
    remembered = _remembered_executable()
    if remembered is not None:
        paths.append(remembered)
    paths.append(default_clone_target() / "bin" / "spack")
    found: list[str] = []
    seen: set[str] = set()
    for candidate in paths:
        if not (candidate.is_file() and os.access(candidate, os.X_OK)):
            continue
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key not in seen:
            seen.add(key)
            found.append(str(candidate))
    return found


def spack_executable() -> str | None:
    """The first Spack on this machine, or None when it has none."""
    found = spack_candidates()
    return found[0] if found else None


def usable_spack() -> tuple[str | None, list[str]]:
    """The first Spack new enough for openPASO's recipes, and why each Spack
    before it was passed over. (None, reasons) when none is."""
    passed_over: list[str] = []
    for executable in spack_candidates():
        problem = too_old(executable)
        if problem is None:
            return executable, passed_over
        passed_over.append(problem)
    return None, passed_over


def clone_advice() -> str:
    """How to get a current Spack when the ones found are too old."""
    target = default_clone_target()
    # git refuses to clone into an existing directory, Spack or not
    where = target if not target.exists() else target.with_name("spack-current")
    clone = shlex.join([*SPACK_CLONE, str(where)])
    return (f"Get a current Spack with `{clone}`"
            + ("" if where == target else f" and set SPACK_ROOT={where}"))


def default_clone_target() -> Path:
    return Path.home() / "spack"


def spack_version(executable: str) -> tuple[int, ...] | None:
    """`spack --version` prints e.g. "1.2.2 (3e19345b...)"; None if unreadable."""
    try:
        done = subprocess.run([executable, "--version"], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=LOOKUP_TIMEOUT,
                              env=spack_env())
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.match(r"\s*(\d+(?:\.\d+)*)", done.stdout)
    return tuple(int(part) for part in match.group(1).split(".")) if match else None


def too_old(executable: str) -> str | None:
    """Why this Spack cannot read openPASO's recipes, or None if it can."""
    version = spack_version(executable)
    if version is None:
        return f"`{executable} --version` did not answer"
    if version < MINIMUM_VERSION:
        return (f"Spack {'.'.join(map(str, version))} at {executable} is too old: openPASO's "
                f"recipes need Spack {'.'.join(map(str, MINIMUM_VERSION))} or newer")
    return None


def forget() -> None:
    """Drop what installed_prefix() remembered, e.g. after a build finished."""
    _found.clear()
    _missing.clear()


def _version_key(text: str) -> tuple:
    # Releases sort by their numbers. Anything else (a branch such as
    # "develop", a git commit) sorts below every release.
    if re.fullmatch(r"\d+(\.\d+)*", text):
        return (1, tuple(int(p) for p in text.split(".")))
    return (0, ())


def _installed_at(prefix: Path) -> float:
    """When Spack finished this build. install_manifest.json is written by the
    post-install hook; spec.json already exists when the build starts (a
    33-minute build showed exactly that gap)."""
    for marker in (prefix / ".spack" / "install_manifest.json",
                   prefix / ".spack" / "spec.json", prefix):
        try:
            return marker.stat().st_mtime
        except OSError:
            continue
    return 0.0


def installed_prefix(spec: str) -> Path | None:
    """Prefix of the preferred installed package matching `spec`, or None.

    Every Spack on the machine is asked, because the one first in the search
    order need not be the one that built the solver. A build from openPASO's
    own recipes wins over any other, then the newest release, then the most
    recent build of it (a changed recipe leaves the older build installed);
    the prefix path breaks what is left of a tie, so the answer never depends
    on Spack's listing order. None covers every way of not knowing: no Spack,
    nothing installed, Spack too slow or broken. A finder treats it like any
    other empty discovery step and moves on."""
    cached = _found.get(spec)
    if cached is not None and cached[0].is_dir() and time.monotonic() - cached[1] < FOUND_TTL:
        return cached[0]
    missing_since = _missing.get(spec)
    if missing_since is not None and time.monotonic() - missing_since < MISSING_TTL:
        return None
    found = []
    for spack in spack_candidates():
        try:
            done = subprocess.run(
                # --color=never: SPACK_COLOR=always would wrap each field in
                # escape codes and no version would parse
                [spack, "--color=never", "-E", "find", "--format",
                 "{version}\t{namespace}\t{prefix}", spec],
                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=LOOKUP_TIMEOUT, env=spack_env())
        except (OSError, subprocess.TimeoutExpired):
            continue
        if done.returncode != 0:
            continue
        for line in done.stdout.splitlines():
            version, namespace, prefix = (line.split("\t") + ["", ""])[:3]
            path = Path(prefix.strip())
            if prefix.strip() and path.is_dir():
                ours = namespace.strip() == RECIPE_NAMESPACE
                rank = (ours, _version_key(version.strip()), _installed_at(path), str(path))
                found.append((rank, path))
    if not found:
        _missing[spec] = time.monotonic()
        return None
    best = max(found, key=lambda item: item[0])[1]
    _found[spec] = (best, time.monotonic())
    return best


def build_prefix(path) -> Path | None:
    """The Spack prefix a file or directory lies in, or None when it is not
    part of a Spack build (Spack writes .spack/spec.json into every prefix)."""
    try:
        here = Path(path).resolve()
    except (OSError, RuntimeError):
        return None
    for prefix in (here, *here.parents):
        if (prefix / ".spack" / "spec.json").is_file():
            return prefix
    return None


def names_prefix(message: str, prefix: Path) -> bool:
    """Whether a finder's message names a path inside `prefix`.

    Spack reports prefixes under the install root it is configured with, which
    may run through a symlink; the deal.II finder reports the resolved path.
    Both count, and so does any other path in the message that resolves into
    the prefix."""
    try:
        root = prefix.resolve()
    except (OSError, RuntimeError):
        root = prefix
    if str(prefix) in message or str(root) in message:
        return True
    for token in re.findall(r"/[^\s'\"(),;]+", message):
        try:
            path = Path(token.rstrip(".:")).resolve()
        except (OSError, RuntimeError):
            continue
        if path == root or root in path.parents:
            return True
    return False


def _prefix_of_hash(dag_hash: str) -> Path | None:
    """Where the Spack that has it installed the build with this hash."""
    for spack in spack_candidates():
        try:
            done = subprocess.run(
                [spack, "--color=never", "-E", "find", "--format", "{prefix}", f"/{dag_hash}"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=LOOKUP_TIMEOUT, env=spack_env())
        except (OSError, subprocess.TimeoutExpired):
            continue
        found = done.stdout.strip()
        if done.returncode == 0 and found and Path(found).is_dir():
            return Path(found)
    return None


def mpi_launcher(executable) -> str | None:
    """The mpirun of the MPI a Spack-built executable links against, or None.

    Started by another MPI's launcher, such a binary runs as N unrelated
    one-process jobs that each solve the whole problem; with no launcher on
    PATH it runs as one process. Spack's spec.json names the build's MPI: an
    external one by its path, one Spack built by its hash. None when the
    executable is not a Spack build, has no MPI, or its MPI has no launcher
    there; the caller then keeps its usual search."""
    prefix = build_prefix(executable)
    if prefix is None:
        return None
    try:
        nodes = json.loads((prefix / ".spack" / "spec.json").read_text())["spec"]["nodes"]
        edges = nodes[0].get("dependencies", [])
        mpi = next((edge.get("hash") for edge in edges
                    if "mpi" in ((edge.get("parameters") or {}).get("virtuals") or [])), None)
        node = next((n for n in nodes if mpi and n.get("hash") == mpi), None)
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
        return None
    if node is None:
        return None
    external = node.get("external") or {}
    mpi_prefix = Path(external["path"]) if external.get("path") else _prefix_of_hash(mpi)
    if mpi_prefix is None:
        return None
    for name in ("mpirun", "mpiexec"):
        launcher = mpi_prefix / "bin" / name
        if launcher.is_file() and os.access(launcher, os.X_OK):
            return str(launcher)
    return None


def recipe_repository() -> Path | None:
    """openPASO's own Spack package repository, if this install carries it."""
    from core.paths import data_dir  # noqa: PLC0415
    repo = data_dir() / "spack" / "spack_repo" / RECIPE_NAMESPACE
    return repo if (repo / "repo.yaml").is_file() else None


def shipped_recipes() -> list[str]:
    """Package names openPASO ships recipes for (directory names are
    Python-safe: a leading digit gets an underscore, hyphens become
    underscores)."""
    repo = recipe_repository()
    if repo is None:
        return []
    names = []
    for recipe in sorted((repo / "packages").glob("*/package.py")):
        name = recipe.parent.name
        names.append((name[1:] if name.startswith("_") else name).replace("_", "-"))
    return names


def needs_recipe_repository(spec: str) -> bool:
    return spec.split()[0].startswith(RECIPE_NAMESPACE + ".")


def recipe_scope_path() -> Path:
    return _state("scope")


def write_recipe_scope() -> Path:
    """A Spack configuration scope holding only openPASO's recipe repository.

    Written fresh before each use, so it always names this install's recipes.
    Raises RuntimeError when this install carries no recipes, OSError when the
    state directory is not writable."""
    repo = recipe_repository()
    if repo is None:
        raise RuntimeError("this openPASO install carries no Spack recipes (data/spack is missing)")
    scope = recipe_scope_path()
    scope.mkdir(parents=True, exist_ok=True)
    # A JSON string is a valid YAML scalar, whatever the path contains.
    (scope / "repos.yaml").write_text(f"repos:\n  {RECIPE_NAMESPACE}: {json.dumps(str(repo))}\n")
    return scope


def install_command(spack: str, spec: str) -> list[str]:
    """`spack install` for a route's spec, ignoring any active Spack environment.

    --reuse-deps rebuilds the solver itself when its recipe changed and reuses
    everything below it; a plain install would keep an older build of the same
    version."""
    command = [spack, "-E"]
    if needs_recipe_repository(spec):
        command += ["-C", str(recipe_scope_path())]
    return command + ["install", "--reuse-deps", *shlex.split(spec)]
