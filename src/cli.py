"""The `openpaso` command: the server, and two things around it.

    openpaso                     start the MCP server on stdio (what an AI app runs)
    openpaso doctor              what this install can do: Python, openPASO, every solver,
                                 the mesh generator -- no key, no network
    openpaso install <solver>    check first; install only what is missing, the way
                                 setup_backend would; say what to do by hand where
                                 no package exists (4C, deal.II, FEBio, SPARTA)
    openpaso install <solver> --via spack
                                 build it with Spack instead (4C, deal.II, SPARTA),
                                 from the recipes openPASO ships (data/spack)

Both commands read the same registry the server uses (core.registry) and the
same setup routes the setup_backend tool uses (core.backend_setup.SETUP_ROUTES),
so what they report and what they run cannot drift from what the server does.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

OK, NO, HM = "[ok]", "[!!]", "[--]"


def _first_line(text: object, limit: int = 88) -> str:
    line = str(text or "").strip().splitlines()[0] if str(text or "").strip() else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _looks_like_a_version(text: object) -> bool:
    value = str(text or "").strip()
    return bool(re.match(r"^\d+\.\d", value)) and not re.search(r"fail|error|unknown|detect", value, re.I)


def _routes(name: str) -> list[dict]:
    try:
        from core.backend_setup import SETUP_ROUTES
    except Exception:                                    # noqa: BLE001
        return []
    return list(SETUP_ROUTES.get(name) or [])


def _route_command(route: dict) -> list[str] | None:
    """The first command of a setup route, with this interpreter where the route names one."""
    for command in route.get("commands") or []:
        parts = [str(p) for p in command]
        if parts and (parts[0].endswith("python") or parts[0].endswith("python3") or parts[0] == sys.executable):
            parts[0] = sys.executable
        return parts
    return None


def _hint(name: str) -> str:
    routes = _routes(name)
    if not routes:
        return ""
    cmd = _route_command(routes[0])
    if cmd:
        shown = list(cmd)
        if shown[0] == sys.executable:
            shown[0] = "python"
        return " ".join(shown)
    return str(routes[0].get("description", ""))


def _spack_hint(name: str) -> str:
    if any(r.get("kind") == "spack" for r in _routes(name)):
        return f"      or build it with Spack:  openpaso install {name} --via spack"
    return ""


def _rows() -> list[dict]:
    from core.registry import list_backends, load_all_backends
    load_all_backends()
    return list_backends()


def doctor() -> int:
    """What this install can do. Returns 0 when openPASO imports and at least one solver works."""
    logging.disable(logging.INFO)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):                    # pragma: no cover
        pass
    print("openPASO -- checking this install. No API key and no network are used.")
    print()
    v = sys.version_info
    if v < (3, 10):
        print(f"{NO} Python {v[0]}.{v[1]} is too old. openPASO needs 3.10 to 3.13.")
        return 1
    print(f"{OK} Python {v[0]}.{v[1]}" + (" -- supported." if v < (3, 14) else " -- newer than anything tested; 3.10 to 3.13 are."))
    try:
        rows = _rows()
    except Exception as exc:                             # noqa: BLE001
        print(f"{NO} openPASO itself does not import: {_first_line(exc)}")
        return 1
    print(f"{OK} openPASO imports, and its tools are registered.")
    usable = [r for r in rows if r["status"] == "available"]
    print()
    print(f"Solvers openPASO can see on this machine -- {len(usable)} of {len(rows)}:")
    print()
    for row in sorted(rows, key=lambda r: (r["status"] != "available", r["name"])):
        mark = OK if row["status"] == "available" else NO
        version = f" {row['version']}" if _looks_like_a_version(row.get("version")) else ""
        print(f"  {mark} {row['display_name']}{version}")
        if row["status"] == "available":
            print(f"      {_first_line(row.get('message'))}")
        else:
            hint = _hint(row["name"])
            print(f"      not installed -- to get it:  openpaso install {row['name']}"
                  + (f"   (runs: {hint})" if hint else ""))
            if _spack_hint(row["name"]):
                print(_spack_hint(row["name"]))
    print()
    try:
        import gmsh  # noqa: F401
        print(f"{OK} The mesh generator (Gmsh) is installed, so `generate_mesh` can build meshes.")
    except Exception:                                    # noqa: BLE001
        print(f"{HM} The mesh generator is not installed -- to get it:  pip install gmsh")
    print()
    if not usable:
        print(f"{NO} No solver works yet. Start with:  openpaso install skfem")
        return 1
    print(f"{OK} Ready: an AI app pointed at the `openpaso` command can use {len(usable)} solver(s).")
    return 0


def _confirm(question: str, yes: bool, unattended: bool = True) -> bool:
    """Ask on a terminal. Without one, `unattended` decides: pip-sized installs
    go ahead as before; a Spack build (an hour, a clone into $HOME) needs -y."""
    if yes:
        return True
    if not sys.stdin.isatty():
        if not unattended:
            print("    Not a terminal, so nobody can answer; pass --yes to run it unattended.")
        return unattended
    return input(f"    {question} [Y/n] ").strip().lower() in ("", "y", "yes")


def _probe(name: str, row: dict, smoke: bool = False) -> tuple[int, str]:
    """Measure, do not assume -- in a FRESH process, because a backend's finder
    may cache what it found before the install (measured: Kratos, installed a
    moment earlier into this very interpreter, still read "not found" here
    while `openpaso doctor` in a new process saw it). With smoke=True the
    backend's smoke test must pass too, where it has one: a 4C that cannot load
    its libraries still names itself in the error, which is all the
    availability check asks for. Returns the exit code and the backend's own
    message."""
    probe = ("import logging,sys; logging.disable(logging.CRITICAL)\n"
             "from core.registry import load_all_backends, get_backend\n"
             "load_all_backends(); st, msg = get_backend(%r).check_availability()\n"
             "print('AVAILABLE' if st.value == 'available' else 'MISSING', (msg or '').replace('\\n', ' ')[:400])") % name
    if smoke:
        probe += ("\nfrom core.smoke_tests import SMOKE_TESTS\n"
                  "if st.value == 'available' and %r in SMOKE_TESTS:\n"
                  "    r = SMOKE_TESTS[%r]()\n"
                  "    if not r.passed:\n"
                  "        print('MISSING', 'its smoke test failed: ' + (r.error or '').replace('\\n', ' ')[:300])") % (name, name)
    # This file's own directory holds the flat modules (core, backends) both in
    # an installed package (openpaso/) and in a checkout (src/); `import
    # openpaso` works only in the first.
    here = os.path.dirname(os.path.abspath(__file__))
    done = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {here!r})\n" + probe],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL)
    verdict = (done.stdout.strip().splitlines() or [""])[-1]
    if verdict.startswith("AVAILABLE"):
        message = verdict[len("AVAILABLE "):]
        print(f"{OK} {row['display_name']} is installed and working: {_first_line(message, 160)}")
        return 0, message
    failure = (done.stderr.strip().splitlines() or [""])[-1]
    message = verdict[len("MISSING "):] if verdict.startswith("MISSING") else failure
    print(f"{NO} The command finished, but openPASO still cannot use {row['display_name']}: {message}")
    return 1, message


def _clone_spack(yes: bool) -> str | None:
    import shlex
    from core import spack as sp
    target = sp.default_clone_target()
    clone = [*sp.SPACK_CLONE, str(target)]
    print(f"{HM} Spack is not installed. It is a git clone of about 30 MB:")
    print(f"      {shlex.join(clone)}")
    if target.exists():
        print(f"{NO} {target} already exists but holds no Spack; move it or clone elsewhere and set SPACK_ROOT.")
        return None
    if not _confirm("Clone Spack there now?", yes, unattended=False):
        print("    Spack not installed; nothing changed.")
        return None
    try:
        cloned = subprocess.run(clone, stdin=subprocess.DEVNULL).returncode == 0
    except FileNotFoundError:
        print(f"{NO} git is not installed, so Spack cannot be cloned. Nothing was changed.")
        return None
    if not cloned:
        print(f"{NO} The clone failed. Nothing else was changed.")
        return None
    exe = str(target / "bin" / "spack")
    if not Path(exe).is_file():
        print(f"{NO} The clone finished but {exe} is not there.")
        return None
    return exe


def _install_with_spack(name: str, row: dict, route: dict, yes: bool) -> int:
    """`spack install` the route's spec in the foreground, then measure."""
    import shlex
    from core import spack as sp
    exe, passed_over = sp.usable_spack()
    for reason in passed_over:
        print(f"{HM} Not using it: {reason}.")
    if exe is None and passed_over:
        print(f"{NO} No Spack here is new enough. {sp.clone_advice()}, then run this again.")
        return 1
    if exe is None:
        exe = _clone_spack(yes)
    if exe is None:
        return 1
    print(f"{OK} Spack: {exe}")
    if sys.platform == "darwin":
        print(f"{HM} This route is measured on Linux only; on macOS it is not verified.")
    spec = route["spec"]
    command = sp.install_command(exe, spec)
    print(f"    Route: {route.get('description', '')}"
          + (f" (about {route['typical_minutes']} min)" if route.get("typical_minutes") else ""))
    if sp.needs_recipe_repository(spec):
        print(f"    openPASO's recipes reach Spack through {sp.recipe_scope_path()} for this command "
              "only; they are not added to your Spack configuration. (Where no compiler is "
              "configured yet, Spack itself records the ones it finds in your ~/.spack, as it "
              "does for any spack command.)")
    print(f"    Will run:  {shlex.join(command)}")
    if not _confirm("Proceed?", yes, unattended=False):
        print("    Not installed.")
        return 1
    try:
        if sp.needs_recipe_repository(spec):
            sp.write_recipe_scope()
        done = subprocess.run(command, stdin=subprocess.DEVNULL, env=sp.spack_env(loads_recipes=True))
    except (OSError, RuntimeError) as exc:
        print(f"{NO} Could not start Spack: {exc}")
        return 1
    if done.returncode != 0:
        verbose = [*command[:-len(shlex.split(spec))], "-v", *shlex.split(spec)]
        print(f"{NO} spack install failed (exit {done.returncode}). To see the whole build: "
              f"{shlex.join(verbose)}")
        return done.returncode
    sp.remember_executable(exe)
    sp.forget()
    code, message = _probe(name, row, smoke=True)
    # The exact spec just built: the plain package name could also match
    # another variant of it (an MPI build of SPARTA has no spa_serial).
    prefix = sp.installed_prefix(spec)
    if prefix is not None and not sp.names_prefix(message, prefix):
        variable, inside = route["override"]
        target = prefix / inside if inside else prefix
        if target.exists():
            lead = ("openPASO still uses another " + row["display_name"] + ", found earlier in its search"
                    if code == 0 else "The Spack build is at " + str(prefix))
            print(f"{HM} {lead}. To use the Spack build, set {variable}={target}")
    return code


def install(name: str, yes: bool = False, via: str | None = None) -> int:
    """Check first; install only what is missing, the way setup_backend would."""
    logging.disable(logging.INFO)
    rows = {r["name"]: r for r in _rows()}
    if name not in rows:
        print(f"{NO} Unknown solver '{name}'. Known: {', '.join(sorted(rows))}")
        return 2
    row = rows[name]
    if row["status"] == "available":
        print(f"{OK} {row['display_name']} is already installed and working"
              f"{' (' + row['version'] + ')' if _looks_like_a_version(row.get('version')) else ''}: "
              f"{_first_line(row.get('message'))}")
        if not via:
            return 0
        # Named explicitly, the route runs anyway (setup_backend does the same):
        # replacing an install that works but is too old is what it is for.
        print(f"    Installing it again with {via}, as asked.")
    routes = _routes(name)
    if via:
        kinds = [r.get("kind") for r in routes]
        routes = [r for r in routes if r.get("kind") == via]
        if not routes:
            print(f"{NO} openPASO has no {via} route for {row['display_name']}. "
                  f"Its routes: {', '.join(kinds) or 'none'}.")
            return 2
    if not routes:
        print(f"{NO} {row['display_name']} is not installed, and openPASO has no automatic route for it.")
        return 1
    route = routes[0]
    if row["status"] != "available":
        print(f"{HM} {row['display_name']} is not installed: {_first_line(row.get('message'))}")
    if route.get("kind") == "spack":
        return _install_with_spack(name, row, route, yes)
    cmd = _route_command(route)
    print(f"    Route: {route.get('description', '')}"
          + (f" (about {route['typical_minutes']} min)" if route.get("typical_minutes") else ""))
    if not cmd:
        # no package to install: say exactly what to do by hand
        print("    No package exists for it. What to do:")
        for step in route.get("steps") or route.get("commands") or [route.get("description", "")]:
            print(f"      {step if isinstance(step, str) else ' '.join(str(p) for p in step)}")
        if any(r.get("kind") == "spack" for r in _routes(name)):
            print(f"    Or let Spack build it with its dependencies:  openpaso install {name} --via spack")
        return 1
    shown = " ".join("python" if p == sys.executable else p for p in cmd)
    print(f"    Will run:  {shown}")
    if not _confirm("Proceed?", yes):
        print("    Not installed.")
        return 1
    done = subprocess.run(cmd, stdin=subprocess.DEVNULL)
    if done.returncode != 0:
        print(f"{NO} The command failed (exit {done.returncode}). Nothing else was changed.")
        return done.returncode
    return _probe(name, row)[0]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        from server import main as serve   # the flat import; see openpaso/__init__.py
        serve()
        return 0
    ap = argparse.ArgumentParser(prog="openpaso", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("serve", help="start the MCP server on stdio (the default when no command is given)")
    sub.add_parser("doctor", help="what this install can do: Python, openPASO, every solver, the mesh generator")
    p_in = sub.add_parser("install", help="check first; install a solver only if it is missing")
    p_in.add_argument("solver", help="skfem, ngsolve, kratos, dune, fenics, dealii, fourc, febio or sparta")
    p_in.add_argument("-y", "--yes", action="store_true", help="do not ask before running the install command")
    p_in.add_argument("--via", choices=["pip", "conda", "binary", "source", "spack"],
                      help="use this kind of route instead of the default one")
    args = ap.parse_args(argv)
    if args.cmd in (None, "serve"):
        from server import main as serve
        serve()
        return 0
    if args.cmd == "doctor":
        return doctor()
    return install(args.solver, yes=args.yes, via=args.via)
