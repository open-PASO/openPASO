"""openPASO calls spack-agent: an AI agent writes or repairs a Spack recipe, and a build verifies it.

spack-agent (https://github.com/Hereon-InstituteMS/spack-agent, MIT) runs an agent CLI that edits
one recipe in a writable package repository and writes a verification script; its runner then
builds the exact spec with no timeout and reads the script's verdict. A failed build goes back to
the agent with an excerpt of the real log, until the verdict passes or the iterations run out.

openPASO prepares everything such a run needs and starts it (`openpaso install <solver> --via
spack-agent`):

  * a git copy of openPASO's own recipe repository, under openPASO's state directory (never under
    the install), as the writable repository: the agent repairs a recipe openPASO ships, or
    writes one where openPASO has none (FEBio);
  * the solver's source checkout, which the agent reads to learn the build: the caller's own
    (`--source`), or a shallow clone of the release tag the spec names;
  * a host Spack wrapper that adds that recipe repository to every Spack call, the same `-E -C
    <scope>` the `--via spack` route uses, since spack-agent's host runner calls Spack as given;
  * spack-agent.toml with the spec and the solver's packaging notes.

spack-agent stays an external program: openPASO finds `spack-agent` on PATH, or the file
SPACK_AGENT names, and says how to install it when it is missing. Which agent plans the edits
(GitHub Copilot by default, Claude Code or an OpenAI-compatible endpoint where the installed
spack-agent offers them) is spack-agent's choice unless `--agent` names one.
"""
from __future__ import annotations

import contextlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

PROJECT_URL = "https://github.com/Hereon-InstituteMS/spack-agent"
INSTALL_COMMAND = f'python -m pip install "spack-agent @ git+{PROJECT_URL}"'

# What spack-agent needs to know about each solver beyond its Spack route: where the source lives
# (the agent reads it to learn the build) and what openPASO's templates need from the build.
# `spec`, `override` and `package_dir` are given only where the solver has no `spack` route of its
# own to take them from.
TARGETS: dict[str, dict] = {
    "fourc": {
        "git": "https://github.com/4C-multiphysics/4C.git",
        "ref": "v2026.3.0",
        "package_dir": "_4c",
        "context": (
            "openPASO runs 4C's input files with the 4C binary and converts native output with "
            "post_processor, so both must be installed in the prefix's bin/ (post_processor is not "
            "built by default). Keep the recipe in the openpaso namespace."
        ),
    },
    "dealii": {
        "git": "https://github.com/dealii/dealii.git",
        "ref": "v9.7.1",
        "package_dir": "dealii",
        "context": (
            "openPASO compiles small deal.II programs against this install with CMake "
            "(find_package(deal.II)), serially; keep the variants the spec names. Keep the recipe "
            "in the openpaso namespace."
        ),
    },
    "sparta": {
        "git": "https://github.com/sparta/sparta.git",
        "ref": "27Aug2026",
        "package_dir": "sparta_dsmc",
        "context": (
            "openPASO runs the serial SPARTA binary spa_serial from the prefix's bin/, with "
            "SPARTA's data files; keep the serial build the spec names. Keep the recipe in the "
            "openpaso namespace."
        ),
    },
    "febio": {
        "git": "https://github.com/febiosoftware/FEBio.git",
        "ref": "v4.13",
        "package_dir": "febio",
        "spec": "openpaso.febio@4.13",
        "override": ("FEBIO_BINARY", "bin/febio4"),
        "context": (
            "No Spack recipe exists for FEBio. Write one that builds FEBio 4.13 from this source "
            "with CMake and installs the febio4 executable in the prefix's bin/; openPASO runs "
            "FEBio input files (.feb) with it. Keep the recipe in the openpaso namespace."
        ),
    },
}


def find_executable() -> str | None:
    """spack-agent's command: the file SPACK_AGENT names, else `spack-agent` on PATH. Absolute,
    because openPASO starts it from the workspace: a relative name would then point elsewhere."""
    named = os.environ.get("SPACK_AGENT", "").strip()
    if named:
        return os.path.abspath(named) if os.path.isfile(named) and os.access(named, os.X_OK) else None
    found = shutil.which("spack-agent")
    return os.path.abspath(found) if found else None


def workspace_dir(solver: str) -> Path:
    """Absolute even when the state directory is configured as a relative path, for the same
    reason as find_executable."""
    from core.session_journal import state_dir  # noqa: PLC0415
    return Path(os.path.abspath(state_dir("spack-agent") / solver))


class WorkspaceBusy(Exception):
    """Another run is using the solver's workspace."""


@contextlib.contextmanager
def _exclusive(path: Path, busy: str):
    try:
        import fcntl  # noqa: PLC0415  (POSIX only, like spack-agent itself)
    except ImportError:                                   # pragma: no cover
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkspaceBusy(busy) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def hold_workspace(workspace: Path):
    """Held for a whole `openpaso install --via spack-agent`, so a second one for the same solver
    stops before it replaces the recipes or the source that a running build reads."""
    return _exclusive(workspace / "openpaso.lock",
                      f"another `openpaso install --via spack-agent` is using {workspace}")


def hold_session(workspace: Path):
    """spack-agent's own session lock, held while openPASO replaces the recipes and the
    configuration. Every spack-agent command on this workspace takes that lock (<config
    directory>/.spack-agent/session.lock, measured with spack-agent 01d2cd0), so one started by
    hand, such as a `run --resume`, stops openPASO before it changes anything, and cannot start
    meanwhile. openPASO lets go before it starts spack-agent, which takes the lock itself."""
    return _exclusive(workspace / ".spack-agent" / "session.lock",
                      f"a spack-agent command is running on {workspace}")


def recipe_copy_state(workspace: Path) -> str:
    """The workspace's recipe copy: "absent"; "unchanged" from the recipes it was copied from;
    "edited" by an earlier run's agent; or "unknown" when git cannot tell (git missing, the
    copy's repository unreadable). Only an "unchanged" copy may be deleted: a recipe edit must
    not be lost because the check failed."""
    recipes = workspace / "recipes"
    if not recipes.exists():
        return "absent"
    if not (recipes / ".git").is_dir():
        return "unknown"
    git = ["git", "-C", str(recipes)]
    try:
        status = subprocess.run([*git, "status", "--porcelain"], capture_output=True, text=True,
                                stdin=subprocess.DEVNULL)
        commits = subprocess.run([*git, "rev-list", "--count", "HEAD"], capture_output=True,
                                 text=True, stdin=subprocess.DEVNULL)
    except OSError:
        return "unknown"
    if status.returncode != 0 or commits.returncode != 0:
        return "unknown"
    return "edited" if status.stdout.strip() or commits.stdout.strip() != "1" else "unchanged"


def show_new_files(recipes: Path) -> None:
    """Mark the files the agent created (FEBio's whole recipe) with git's intent-to-add, so that
    `git diff` in the copy shows them: it shows no untracked file."""
    subprocess.run(["git", "-C", str(recipes), "add", "--intent-to-add", "--all"],
                   capture_output=True, stdin=subprocess.DEVNULL)


def target(solver: str, route: dict | None) -> dict | None:
    """Everything a run needs for this solver: the spack-agent entry, completed from its `spack`
    route where it has one. None when spack-agent has no entry for the solver."""
    entry = TARGETS.get(solver)
    if entry is None:
        return None
    out = dict(entry)
    if route is not None:
        out.setdefault("spec", route.get("spec"))
        out.setdefault("override", route.get("override"))
        if route.get("typical_minutes"):
            out.setdefault("typical_minutes", route["typical_minutes"])
    if not out.get("spec"):
        return None
    return out


def _toml_string(value: str) -> str:
    # A JSON string is a valid TOML basic string for what is written here: paths, a spec,
    # prose, and backslash escapes for quotes and newlines.
    return json.dumps(value)


def write_config(workspace: Path, *, source: Path, spec: str, recipe_path: str, context: str,
                 spack_wrapper: Path, agent: str | None = None, model: str | None = None,
                 max_iterations: int | None = None) -> Path:
    """spack-agent.toml for a host-runner session in `workspace`."""
    lines = [
        "[workspace]",
        f"source_repository = {_toml_string(str(source))}",
        f"writable_repository = {_toml_string(str(workspace / 'recipes'))}",
        'spack_repository = "spack_repo/openpaso"',
        f"recipe_path = {_toml_string(recipe_path)}",
        f"host_spack_executable = {_toml_string(str(spack_wrapper))}",
        "",
        "[goal]",
        f"spec = {_toml_string(spec)}",
        f"context = {_toml_string(context)}",
        "",
    ]
    # spack-agent requires the [agent] table even when it is empty (measured with upstream main
    # 01d2cd0 and the agent-backends branch: "must contain a [agent] table"). `backend` is written
    # only when named: upstream offers only its Copilot default and refuses the setting.
    lines += ["[agent]"]
    lines += [f"backend = {_toml_string(agent)}"] if agent else []
    lines += [f"model = {_toml_string(model)}"] if model else []
    lines += ["", "[runner]", 'backend = "host"']
    if max_iterations is not None:
        lines.append(f"max_iterations = {int(max_iterations)}")
    path = workspace / "spack-agent.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def prepare_recipes(workspace: Path) -> Path:
    """A fresh git copy of openPASO's recipe repository in the workspace; returns it.

    A new run starts from the recipes this openPASO ships. The copy is a git repository, so
    the agent's edits can be read with `git -C <it> diff` afterwards. A copy an earlier run's
    agent edited, or one git cannot check, is kept as recipes.previous (replacing an older
    one), not deleted."""
    from core.spack import recipe_repository  # noqa: PLC0415
    shipped = recipe_repository()
    if shipped is None:
        raise RuntimeError("this openPASO install carries no Spack recipes (data/spack is missing)")
    recipes = workspace / "recipes"
    state = recipe_copy_state(workspace)
    if state in ("edited", "unknown"):
        previous = workspace / "recipes.previous"
        if previous.exists():
            shutil.rmtree(previous)
        recipes.rename(previous)
    elif state == "unchanged":
        shutil.rmtree(recipes)
    shutil.copytree(shipped, recipes / "spack_repo" / shipped.name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    git = ["git", "-C", str(recipes), "-c", "user.name=openPASO", "-c", "user.email=openpaso@localhost"]
    subprocess.run([*git, "init", "-q"], check=True, stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # spack-agent's preflight compiles the recipe (measured: __pycache__/package.cpython-38.pyc),
    # which is no edit of it
    (recipes / ".git" / "info").mkdir(parents=True, exist_ok=True)
    (recipes / ".git" / "info" / "exclude").write_text("__pycache__/\n*.pyc\n")
    for args in (["add", "-A"], ["commit", "-q", "-m", "openPASO's shipped recipes"]):
        subprocess.run([*git, *args], check=True, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return recipes


def write_spack_wrapper(workspace: Path, spack: str, recipes: Path) -> Path:
    """A `spack` for spack-agent's host runner that adds the workspace's recipe repository to
    every call, and ignores an active Spack environment, as `openpaso install --via spack` does."""
    scope = workspace / "scope"
    scope.mkdir(parents=True, exist_ok=True)
    repo = recipes / "spack_repo" / "openpaso"
    (scope / "repos.yaml").write_text(f"repos:\n  openpaso: {json.dumps(str(repo))}\n")
    wrapper = workspace / "spack"
    wrapper.write_text("#!/bin/sh\n"
                       f"exec {shlex.quote(os.path.abspath(spack))} -E -C {shlex.quote(str(scope))} \"$@\"\n")
    wrapper.chmod(0o755)
    return wrapper


def clone_command(entry: dict, destination: Path) -> list[str]:
    return ["git", "clone", "--depth=1", f"--branch={entry['ref']}", entry["git"], str(destination)]


def run_command(executable: str, config: Path) -> list[str]:
    return [executable, "--config", str(config), "run"]


def check_command(executable: str, config: Path) -> list[str]:
    """spack-agent's own reading of the configuration: it resolves and checks every path and
    setting, starts no agent and builds nothing (seconds)."""
    return [executable, "--config", str(config), "config"]


def check_hint(output: str) -> str:
    """What to do about a configuration this spack-agent refused, for the refusals measured."""
    if "unknown [agent] setting(s): backend" in output:
        return ("this spack-agent drives only its default agent (the GitHub Copilot CLI): run "
                "again without --agent, or install a spack-agent that offers more backends")
    return ""
