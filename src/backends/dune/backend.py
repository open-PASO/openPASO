"""
DUNE-fem solver backend.

DUNE-fem uses UFL (same form language as FEniCS) with Netgen/ALUGrid meshes.
Generates Python scripts, executes them, collects VTK output.
VTK output is native via gridView.writeVTK() — no conversion needed.

Key advantage: shares UFL with FEniCS, so physics descriptions are nearly
interchangeable. Different backend (DUNE C++ infrastructure vs PETSc).
"""

import asyncio
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from core.backend import (
    sorted_by_step,
    SolverBackend, BackendStatus, InputFormat,
    PhysicsCapability, JobHandle, get_python_executable,
)
from core.registry import register_backend
from .generators import GENERATORS, KNOWLEDGE

logger = logging.getLogger("openpaso.dune")

# Cache for the verified dune.fem interpreter. Probing candidates spawns a
# subprocess per candidate (`import dune.fem`, up to 30 s each), so we only
# resolve once per server lifetime. Cleared via _reset_dune_python_cache()
# (used by tests).
_DUNE_PYTHON_CACHE: dict = {}

# THIS MESSAGE USED TO RECOMMEND A COMMAND TWO OTHER SERVED PASSAGES CALL
# IMPOSSIBLE. It said `conda create -n ofa-dune -c conda-forge dune-fem`,
# while _setup.py says that command "cannot succeed" and server.py says
# "conda-forge has NO dune-fem package". An agent hits this text at exactly
# the moment DUNE has failed, so it is the worst place in the codebase to
# spend its budget on a command that cannot work.
_DUNE_INSTALL_HINT = (
    "dune.fem not importable in any candidate Python.\n"
    "Install from PyPI, which is the working source — conda-forge has no "
    "dune-fem package:\n"
    "  pip install dune-fem mpi4py    (mpi4py is an undeclared dependency; "
    "without it the first import stops)\n"
    "Or point openPASO at an existing install:\n"
    "  DUNE_PYTHON=/path/to/env/bin/python   (explicit interpreter)\n"
    "  DUNE_CONDA_PREFIX=/path/to/env        (conda env root)\n"
    "DUNE JIT-compiles C++ on first use, so a fresh install is slow before it "
    "is fast. A cache built against a different Python fails at USE, not at "
    "import: `import dune.fem` succeeds and the first grid dies with "
    "'undefined symbol'. Remove ~/.cache/dune-py to rebuild it.")


def _reset_dune_python_cache():
    """Clear the cached interpreter (tests / env changes)."""
    _DUNE_PYTHON_CACHE.clear()


def _find_dune_python() -> Optional[str]:
    """Locate and VERIFY the Python interpreter that can import dune.fem.

    This is the single source of truth for interpreter resolution — used by
    BOTH check_availability() and run(). (Issue #40: run() previously used
    the server's own venv Python unconditionally, so a dune.fem found in a
    conda env by check_availability() was ignored at execution time.)

    DUNE-fem is heavy enough that users typically install it into a
    dedicated conda env (ofa-dune is the convention mirroring
    ofa-fenicsx / ofa-dealii). The MCP server runs in the openpaso .venv
    which usually does NOT have dune.fem.

    Resolution order:
      * DUNE_PYTHON env var (explicit interpreter, verified too)
      * DUNE_CONDA_PREFIX env var (conda env root)
      * priority 0 — any conda env with "dune" in its name (ofa-dune,
        dune, dune-fem, ...) — the most likely place.
      * priority 1 — any other conda env (covers dune.fem installed
        alongside fenicsx in a single shared env).
      * priority 99 — the active interpreter (.venv) — checked last
        because dune.fem in the harness's own .venv is rare and probing
        it first would slow down the common case.

    Every candidate is verified with a subprocess `import dune.fem`
    before being returned; the first success is cached.
    """
    import subprocess

    if "python" in _DUNE_PYTHON_CACHE:
        return _DUNE_PYTHON_CACHE["python"]

    candidates: list[tuple[int, str]] = []

    # Explicit overrides first (mirrors FENICS_PYTHON / FENICS_CONDA_PREFIX).
    env_python = os.environ.get("DUNE_PYTHON", "")
    if env_python and Path(env_python).is_file():
        candidates.append((-2, env_python))
    env_prefix = os.environ.get("DUNE_CONDA_PREFIX", "")
    if env_prefix:
        p = Path(env_prefix) / "bin" / "python"
        if p.is_file():
            candidates.append((-1, str(p)))

    for conda_base in (
            Path.home() / "miniconda3" / "envs",
            Path.home() / "anaconda3" / "envs",
            Path.home() / "miniforge3" / "envs"):
        if not conda_base.is_dir():
            continue
        for env_dir in sorted(conda_base.iterdir()):
            py = env_dir / "bin" / "python"
            if py.is_file():
                # Prioritise *dune* envs but accept any
                name = env_dir.name.lower()
                priority = 0 if "dune" in name else 1
                candidates.append((priority, str(py)))

    active = get_python_executable()
    if active:
        candidates.append((99, active))

    # De-dup while preserving the best (lowest) priority per interpreter.
    seen = set()
    ordered = []
    for priority, py in sorted(candidates, key=lambda x: x[0]):
        if py in seen:
            continue
        seen.add(py)
        ordered.append(py)

    for python in ordered:
        try:
            result = subprocess.run(
                # The probe must BUILD a JIT module, not just import the
                # package. A conda env with a broken C-ABI passes
                # `import dune.fem` with rc=0 and then raises
                # 'undefined symbol: PyThreadState_GetUnchecked' the
                # moment a generated module loads — so openPASO selected the
                # poisoned interpreter (priority 0) over the working venv
                # (priority 99, never reached) and four coupled runs died
                # on it. structuredGrid triggers the JIT path.
                [python, "-c", "from dune.grid import structuredGrid; "
             "structuredGrid([0,0],[1,1],[2,2]); print('OK')"], stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                _DUNE_PYTHON_CACHE["python"] = python
                return python
        except Exception:
            continue
    # Cache the negative result too, so a missing/misconfigured dune.fem does
    # not re-probe every candidate (up to 30 s each) on every check/run.
    _DUNE_PYTHON_CACHE["python"] = None
    return None


def _interpreter_prefix(python: str) -> str:
    """Authoritative install prefix (sys.prefix) for an interpreter.

    Correct for venv, conda, and --system-site-packages alike, and — crucially —
    for a venv it returns the VENV, not the base interpreter a resolve()'d
    bin/python symlink would point at (issue #40 redux). Falls back, if the
    interpreter can't be queried, to resolving the bin DIRECTORY (never a
    symlink) rather than the interpreter FILE (the venv symlink we must not
    follow).
    """
    import subprocess
    try:
        out = subprocess.run(
            [python, "-c", "import sys; print(sys.prefix)"], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return str(Path(os.path.dirname(python) or ".").resolve().parent)


def _dune_subprocess_env(python: str) -> dict:
    """Environment for running dune scripts with the resolved interpreter.

    dune-fem JIT-compiles C++ modules at run time; its CMake needs to
    find the dune-* package configs that live under the interpreter's
    prefix (<prefix>/lib/cmake/dune-*). When the interpreter comes from
    a conda env that the server never activated, CMAKE_PREFIX_PATH does
    not contain that prefix and the very first JIT build fails with
    "Could not find ... dune-commonConfig.cmake" (issue #40 follow-up:
    picking the right python is necessary but not sufficient).

    Issue #44: CMAKE_PREFIX_PATH alone is still not enough. dune-py's JIT
    build runs ``find_package(Python3)``, and on macOS that grabs Xcode's
    bundled Python 3.9 (or any earlier interpreter on PATH) instead of the
    conda env's python — the compiled module then cannot ``import dune.fem``
    because it was built against the wrong Python. Make the env look like
    the conda env is activated and hand CMake explicit hints so its
    FindPython3 resolves to exactly the interpreter we picked.
    """
    env = os.environ.copy()
    # A venv's bin/python is a SYMLINK to the base interpreter, so
    # Path(python).resolve() would yield the BASE prefix, not the venv — every
    # dune-*Config.cmake under <venv>/lib/cmake then goes unfound and the JIT
    # build fails exactly as in issue #40 (audit finding). Ask the interpreter
    # for its own sys.prefix (authoritative for venv / conda / system-site),
    # with a fallback that resolves the bin DIRECTORY rather than following the
    # interpreter symlink.
    prefix = _interpreter_prefix(python)
    # <env>/bin — resolve the DIRECTORY (safe) but not the interpreter symlink.
    bindir = str(Path(os.path.dirname(python) or ".").resolve())

    def _prepend(var: str, value: str) -> None:
        cur = env.get(var, "")
        if value not in cur.split(os.pathsep):
            env[var] = value + (os.pathsep + cur if cur else "")

    # dune-*Config.cmake lookup (issue #40 follow-up)
    _prepend("CMAKE_PREFIX_PATH", prefix)

    # Mirror activation so CMake's FindPython3 (which honours CONDA_PREFIX /
    # VIRTUAL_ENV via Python3_FIND_VIRTUALENV) prefers THIS env, and put the
    # interpreter's bin first so a bare `python3` / CMake's PATH-based search
    # resolves here and not to a system / Xcode python.
    if (Path(prefix) / "conda-meta").is_dir():
        env["CONDA_PREFIX"] = prefix
        env["CONDA_DEFAULT_ENV"] = Path(prefix).name
        env.pop("VIRTUAL_ENV", None)
    else:
        # Non-conda venv: a stale CONDA_PREFIX inherited from the server's own
        # environment would otherwise win CMake's virtualenv detection and pull
        # in the WRONG python (audit finding). Clear it and point virtualenv
        # detection at the resolved interpreter instead.
        env.pop("CONDA_PREFIX", None)
        env.pop("CONDA_DEFAULT_ENV", None)
        env["VIRTUAL_ENV"] = prefix
    _prepend("PATH", bindir)

    # CMake's FindPython3 honours Python3_ROOT_DIR from the environment.
    # (Python3_EXECUTABLE is only read as a -D cache argument, never from the
    # environment, so setting it here would be dead weight.)
    env["Python3_ROOT_DIR"] = prefix
    return env


class DuneBackend(SolverBackend):

    def name(self) -> str:
        return "dune"

    def display_name(self) -> str:
        return "DUNE-fem"

    def check_availability(self) -> tuple[BackendStatus, str]:
        # Interpreter discovery + verification lives in
        # _find_dune_python() so that run() executes with EXACTLY the
        # interpreter reported here (issue #40).
        python = _find_dune_python()
        if python:
            return BackendStatus.AVAILABLE, f"DUNE-fem at {python}"
        return BackendStatus.NOT_INSTALLED, _DUNE_INSTALL_HINT

    def input_format(self) -> InputFormat:
        return InputFormat.PYTHON

    def supported_physics(self) -> list[PhysicsCapability]:
        return [
            PhysicsCapability(
                name="registry_survey",
                description=(
                    "Registry survey: builds every entry this dune-fem install "
                    "declares across all 8 axes (spaces, schemes, solvers, "
                    "storages, models, operators, functions, views) and checks "
                    "each against something independent of the run. Also "
                    "records the first-use JIT cost, which is 85-200s per new "
                    "space and 0.1s once cached."
                ),
                spatial_dims=[2],
                element_types=["all"],
                template_variants=["default"],
            ),
            PhysicsCapability(
                name="poisson",
                description="Poisson equation -Δu = f (UFL forms, DUNE backend)",
                spatial_dims=[2, 3],
                element_types=["Lagrange-P1", "Lagrange-P2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="poisson_mms",
                description=(
                    "3D variable-coefficient Poisson manufactured-solution "
                    "(MMS) convergence family — -div(kappa grad u) = f on "
                    "[0,L]^3, affine kappa, exact Dirichlet data, uniform "
                    "refinement with per-level L2/H1 error lines; "
                    "theoretical L2 order k+1 / H1 order k"),
                spatial_dims=[3],
                element_types=["Lagrange-P1", "Lagrange-P2",
                               "Lagrange-P3", "Lagrange-P4"],
                template_variants=["3d_varcoeff"],
            ),
            PhysicsCapability(
                name="heat",
                description="Steady heat conduction (UFL)",
                spatial_dims=[2],
                element_types=["Lagrange-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="linear_elasticity",
                description="Linear elasticity (UFL vector space)",
                spatial_dims=[2],
                element_types=["Lagrange-P1-vec"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="stokes",
                description="Stokes flow with Uzawa iteration (UFL)",
                spatial_dims=[2],
                element_types=["Lagrange-P2 + Lagrange-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="reaction_diffusion",
                description="Reaction-diffusion (transient)",
                spatial_dims=[2],
                element_types=["Lagrange-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="nonlinear",
                description="Nonlinear PDE via Newton method (UFL)",
                spatial_dims=[2],
                element_types=["Lagrange-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="dg_advection",
                description="DG method for pure advection equation (upwind flux)",
                spatial_dims=[2],
                element_types=["DG-Lagrange-P1", "DG-Lagrange-P2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="dg_advection_diffusion",
                description=("Steady advection-diffusion with upwind flux and "
                             "SIPG diffusion on triangles"),
                spatial_dims=[2],
                element_types=["DG-Lagrange-P1", "DG-Lagrange-P2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="adaptive_poisson",
                description="h-adaptive Poisson with residual error estimator",
                spatial_dims=[2],
                element_types=["Lagrange-P1", "Lagrange-P2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="maxwell",
                description="Maxwell equations: 2-D TE-mode scalar Helmholtz proxy (-Δu - k²u = f); full H(curl) via NGSolve",
                spatial_dims=[2],
                element_types=["Lagrange-P2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="eigenvalue",
                description="Eigenvalue problem -Δu = λu via shift-invert inverse iteration",
                spatial_dims=[2],
                element_types=["Lagrange-P1", "Lagrange-P2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="hyperelasticity",
                description="Neo-Hookean finite-strain hyperelasticity with automatic Newton via UFL",
                spatial_dims=[2],
                element_types=["Lagrange-P1-vec", "Lagrange-P2-vec"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="navier_stokes",
                description="Incompressible Navier-Stokes via Picard iteration (lid-driven cavity)",
                spatial_dims=[2],
                element_types=["Lagrange-P2-vec + Lagrange-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="helmholtz",
                description="Helmholtz equation -Δu - k²u = f with MMS verification",
                spatial_dims=[2],
                element_types=["Lagrange-P2", "Lagrange-P3"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="time_dependent_heat",
                description="Transient heat du/dt - alpha*Δu = f via implicit Euler time-stepping",
                spatial_dims=[2],
                element_types=["Lagrange-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="mixed_methods",
                description="Mixed Poisson with Raviart-Thomas RT0 flux and DG-P0 pressure (H(div) x L²)",
                spatial_dims=[2],
                element_types=["RT0 + DG-P0", "RT1 + DG-P1"],
                template_variants=["2d"],
            ),
        ]

    def get_knowledge(self, physics: str) -> dict:
        return KNOWLEDGE.get(physics, {})

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        key = f"{physics}_{variant}"
        gen = GENERATORS.get(key)
        if not gen:
            raise ValueError(f"No DUNE template for {key}")
        return gen(params)

    def validate_input(self, content: str) -> list[str]:
        errors = []
        if "dune" not in content:
            errors.append("Script should import from dune")
        try:
            compile(content, "<dune_input>", "exec")
        except SyntaxError as e:
            errors.append(f"Python syntax error: {e}")
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        # Use the SAME interpreter that check_availability() verified —
        # NOT the server's own venv Python (issue #40: the venv usually
        # lacks dune.fem; it lives in a conda env).
        python = _find_dune_python()
        if not python:
            return JobHandle(
                job_id=str(uuid.uuid4())[:8],
                backend_name="dune",
                work_dir=work_dir,
                status="failed",
                error=_DUNE_INSTALL_HINT,
            )

        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / "solve.py"
        script_path.write_text(input_content)

        job = JobHandle(
            job_id=str(uuid.uuid4())[:8],
            backend_name="dune",
            work_dir=work_dir,
            status="running",
        )

        cmd = [python, str(script_path)]
        start = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
                env=_dune_subprocess_env(python),
                start_new_session=True,
                stdin=asyncio.subprocess.DEVNULL,
            )
            # DUNE JIT compiles on first run — can be slow

            # TIMEOUT MUST KILL THE SOLVER, AND THE WHOLE GROUP. Without this, a
            # timed-out solve kept running forever: wait_for() abandoned the
            # process but never terminated it, and a sweep found one such
            # solver 3.2 CPU-hours later at 100%% of a core, its MPI daemon
            # (orted) beside it. start_new_session puts the solver and every child
            # it spawns into their own process group, so one killpg reaps MPI
            # ranks too — the same idiom precice_config.py already uses, for the
            # same reason. The kill re-raises, so each backend's own TimeoutError
            # handling below is unchanged.
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                import os as _os, signal as _signal
                try:
                    _os.killpg(proc.pid, _signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                await proc.wait()
                raise

            job.elapsed = time.time() - start
            job.return_code = proc.returncode
            job.pid = proc.pid

            if proc.returncode == 0:
                job.status = "completed"
            else:
                job.status = "failed"
                job.error = stderr.decode(errors="replace")[-2000:]

            (work_dir / "stdout.log").write_text(stdout.decode(errors="replace"))
            (work_dir / "stderr.log").write_text(stderr.decode(errors="replace"))
        except asyncio.TimeoutError:
            job.status = "failed"
            job.elapsed = timeout
            job.error = f"Timed out after {timeout}s (DUNE JIT compilation can be slow on first run)"
        except Exception as e:
            job.status = "failed"
            job.elapsed = time.time() - start
            job.error = str(e)

        return job

    def get_result_files(self, job: JobHandle) -> list[Path]:
        results = []
        for ext in ["*.vtu", "*.vtk", "*.pvd"]:
            results.extend(job.work_dir.rglob(ext))
        return sorted_by_step(results)


def register():
    backend = DuneBackend()
    register_backend(backend, aliases=["dune", "dune-fem", "dunefem"])
    logger.info("DUNE-fem backend registered")
