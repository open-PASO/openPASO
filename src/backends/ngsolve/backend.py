"""
NGSolve solver backend.

Generates Python scripts using the NGSolve/Netgen API, executes them,
and collects VTK output files. NGSolve produces VTK natively via
VTKOutput — no post-conversion needed (unlike FEniCS XDMF).
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
    PhysicsCapability, JobHandle,
)
from core.registry import register_backend
from .generators import GENERATORS, KNOWLEDGE

logger = logging.getLogger("openpaso.ngsolve")


_NGSOLVE_PYTHON_CACHE: dict[str, Optional[str]] = {}


def _find_ngsolve_python() -> Optional[Path]:
    """Locate and VERIFY the Python interpreter that can run ngsolve.

    THIS USED TO RETURN sys.executable AND NOTHING ELSE, which made NGSolve
    the one backend that could not be pointed anywhere. On a machine with
    ngsolve installed in two separate interpreters, openPASO reported
    "No Python with ngsolve found" and an agent given an NGSolve task spent
    its run trying to install what was already there. Both callers are fixed
    by fixing this one function: check_availability() and run() share it, so
    the interpreter that answers the question is the one that does the work.

    Resolution order, mirroring DUNE and FEniCSx:
      * NGSOLVE_PYTHON      -- an explicit interpreter
      * NGSOLVE_CONDA_PREFIX -- an explicit conda env root
      * the active interpreter, because `pip install ngsolve` into openPASO's
        own virtual environment is the documented easy route and needs no
        subprocess hop
      * a conda env whose name mentions ngsolve or netgen
      * any other conda env

    Every candidate is VERIFIED by building a mesh and a function space, not
    by importing the package. DUNE taught this the expensive way: an import
    can succeed against a broken C-ABI and fail the moment real work starts,
    so openPASO picked the poisoned interpreter and four coupled runs died on
    it. NGSolve has the same shape -- a Python package over a compiled core.
    """
    import subprocess
    import sys

    if "python" in _NGSOLVE_PYTHON_CACHE:
        found = _NGSOLVE_PYTHON_CACHE["python"]
        return Path(found) if found else None

    candidates: list[tuple[int, str]] = []

    env_python = os.environ.get("NGSOLVE_PYTHON", "")
    if env_python and Path(env_python).is_file():
        candidates.append((-2, env_python))
    env_prefix = os.environ.get("NGSOLVE_CONDA_PREFIX", "")
    if env_prefix:
        p = Path(env_prefix) / "bin" / "python"
        if p.is_file():
            candidates.append((-1, str(p)))

    # WHAT AUTODISCOVERY ALREADY RECORDED, and this is the entry that matters
    # inside the sandbox. The isolated shell sets HOME to the cell's own work
    # directory, so the conda scan below looks in an empty tree and finds
    # nothing: measured, a run with ngsolve on the machine spent its whole
    # budget calling setup_backend(action='install') and pip-installing what
    # was already there. The discovered config holds an ABSOLUTE path that
    # survives the tmpfs, and the repo it lives in is bound read-only.
    try:
        from core.autodiscovery import load_discovered_config
        entry = ((load_discovered_config() or {}).get("backends") or {}).get("ngsolve")
        recorded = (entry or {}).get("location")
        if recorded and Path(recorded).is_file():
            candidates.append((-3, str(recorded)))
    except Exception:                                        # noqa: BLE001
        pass

    candidates.append((0, sys.executable))

    for conda_base in (Path.home() / "miniconda3" / "envs",
                       Path.home() / "anaconda3" / "envs",
                       Path.home() / "miniforge3" / "envs"):
        if not conda_base.is_dir():
            continue
        for env_dir in sorted(conda_base.iterdir()):
            py = env_dir / "bin" / "python"
            if py.is_file():
                name = env_dir.name.lower()
                priority = 1 if ("ngsolve" in name or "netgen" in name) else 2
                candidates.append((priority, str(py)))

    seen: set[str] = set()
    ordered: list[str] = []
    for _priority, py in sorted(candidates, key=lambda x: x[0]):
        if py not in seen:
            seen.add(py)
            ordered.append(py)

    probe = ("from netgen.geom2d import unit_square\n"
             "from ngsolve import Mesh, H1\n"
             "H1(Mesh(unit_square.GenerateMesh(maxh=0.5)), order=1)\n"
             "print('OK')\n")
    for python in ordered:
        try:
            result = subprocess.run([python, "-c", probe],
                                    stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and "OK" in result.stdout:
                _NGSOLVE_PYTHON_CACHE["python"] = python
                return Path(python)
        except Exception:                                    # noqa: BLE001
            continue

    _NGSOLVE_PYTHON_CACHE["python"] = None
    return None


class NgsolveBackend(SolverBackend):

    def name(self) -> str:
        return "ngsolve"

    def display_name(self) -> str:
        return "NGSolve"

    def check_availability(self) -> tuple[BackendStatus, str]:
        python = _find_ngsolve_python()
        if not python:
            return BackendStatus.NOT_INSTALLED, "No Python with ngsolve found"

        import subprocess
        try:
            result = subprocess.run(
                [str(python), "-c", "import ngsolve; print(ngsolve.__version__)"], stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                ver = result.stdout.strip()
                return BackendStatus.AVAILABLE, f"NGSolve {ver} at {python}"
            else:
                return BackendStatus.NOT_INSTALLED, f"ngsolve import failed: {result.stderr.strip()}"
        except Exception as e:
            return BackendStatus.NOT_INSTALLED, f"Check failed: {e}"

    def input_format(self) -> InputFormat:
        return InputFormat.PYTHON

    def supported_physics(self) -> list[PhysicsCapability]:
        return [
            # Added 2026-09-19. NGSolve's capability surface IS its function
            # spaces and openPASO drove 6 of 31. This assembles a mass form on
            # every space that can carry one and verifies the matrix is SPD --
            # which caught three spaces assembling an EMPTY matrix without error.
            PhysicsCapability(
                name="space_survey",
                description=("Assemble a mass form on every NGSolve space "
                             "and verify it is symmetric positive definite"),
                spatial_dims=[2],
                element_types=["H1", "HCurl", "HDiv", "L2", "facet", "surface"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="poisson",
                description="Poisson equation -Δu = f with arbitrary-order H1 elements",
                spatial_dims=[2, 3],
                element_types=["H1-P1", "H1-P2", "H1-Pk"],
                template_variants=["2d", "3d"],
            ),
            PhysicsCapability(
                name="linear_elasticity",
                description="Linear elasticity (plane strain / 3D) with VectorH1",
                spatial_dims=[2, 3],
                element_types=["VectorH1-P1", "VectorH1-P2"],
                template_variants=["2d", "3d"],
            ),
            PhysicsCapability(
                name="heat",
                description="Heat conduction (steady and transient with implicit Euler)",
                spatial_dims=[2, 3],
                element_types=["H1-P1", "H1-Pk"],
                template_variants=["2d", "2d_steady", "2d_transient"],
            ),
            PhysicsCapability(
                name="stokes",
                description="Stokes flow with Taylor-Hood P2/P1 or HDG",
                spatial_dims=[2, 3],
                element_types=["VectorH1-P2 + H1-P1", "HDG"],
                template_variants=["2d", "2d_hdg"],
            ),
            PhysicsCapability(
                name="navier_stokes",
                description="Incompressible Navier-Stokes (IMEX time-stepping)",
                spatial_dims=[2],
                element_types=["VectorH1-P2 + H1-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="maxwell",
                description="Maxwell's equations with HCurl (Nedelec) elements",
                spatial_dims=[3],
                element_types=["HCurl-Nedelec"],
                template_variants=["3d_magnetostatics"],
            ),
            PhysicsCapability(
                name="helmholtz",
                description="Helmholtz equation with PML (complex-valued)",
                spatial_dims=[2],
                element_types=["H1-Pk-complex"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="hyperelasticity",
                description="Nonlinear hyperelasticity (Neo-Hookean) via SymbolicEnergy",
                spatial_dims=[2, 3],
                element_types=["VectorH1-P2"],
                template_variants=["2d", "3d"],
            ),
            PhysicsCapability(
                name="eigenvalue",
                description="Eigenvalue problems (Laplace, elasticity) via ArnoldiSolver",
                spatial_dims=[2],
                element_types=["H1-Pk"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="convection_diffusion",
                description="Convection-diffusion with DG upwind stabilization",
                spatial_dims=[2],
                element_types=["L2-DG"],
                template_variants=["2d_dg"],
            ),
            PhysicsCapability(
                name="mixed_poisson",
                description="Mixed Poisson with H(div)/L2 (flux recovery)",
                spatial_dims=[2],
                element_types=["HDiv-RT + L2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="thermal_structural",
                description="Coupled thermal-structural (heat -> elasticity with thermal strain)",
                spatial_dims=[2],
                element_types=["H1 + VectorH1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="surface_pde",
                description="PDE on curved surface manifold (Laplace-Beltrami)",
                spatial_dims=[3],
                element_types=["H1-surface"],
                template_variants=["3d"],
            ),
            PhysicsCapability(
                name="plasticity",
                description="Elasto-plasticity with isotropic hardening (J2/von Mises)",
                spatial_dims=[2],
                element_types=["VectorH1-P2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="dg_methods",
                description="Interior-penalty DG (SIPG) for advection-diffusion using L2 dglagrange space",
                spatial_dims=[2],
                element_types=["L2-DG-Pk"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="contact",
                description="Unilateral contact / obstacle problem via penalty method on elastic domain",
                spatial_dims=[2],
                element_types=["VectorH1-P2"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="time_dependent_ns",
                description="Transient incompressible Navier-Stokes with IMEX splitting (full channel/cavity)",
                spatial_dims=[2],
                element_types=["VectorH1-P2 + H1-P1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="mhd",
                description="Magnetohydrodynamics: coupled NS + Maxwell, 2.5-D low-Rm Hartmann problem",
                spatial_dims=[2],
                element_types=["VectorH1-P2 + H1-P1", "H1-P2 (magnetic scalar)"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="hdivdiv",
                description="HDivDiv Hellan-Herrmann-Johnson for Kirchhoff plate bending / biharmonic",
                spatial_dims=[2],
                element_types=["HDivDiv + H1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="nonlinear_elasticity",
                description="Large-deformation Neo-Hookean hyperelasticity with load stepping and Cauchy stress output",
                spatial_dims=[2, 3],
                element_types=["VectorH1-P2"],
                template_variants=["2d", "3d"],
            ),
            PhysicsCapability(
                name="phase_field",
                description="Phase-field: Allen-Cahn (interface motion) and fracture (Bourdin staggered scheme)",
                spatial_dims=[2],
                element_types=["H1-Pk", "VectorH1 + H1 (fracture)"],
                template_variants=["2d", "fracture_2d"],
            ),
        ]

    # THE TASK'S WORDS MUST RESOLVE TO THE BUCKET THAT HOLDS THE FACT.
    #
    # One task said "anisotropic diffusion" and "constant symmetric
    # positive-definite tensor K". The knowledge that decides that problem — a
    # constant matrix coefficient must be written
    # CoefficientFunction(..., dims=(2,2)), because the nested-list spelling
    # silently becomes a scalar — lives under `poisson`. Measured before this
    # map: get_knowledge("anisotropic_diffusion") returned {}, so
    # knowledge(topic='pitfalls', physics='anisotropic_diffusion') served
    # 5,685 chars, LESS than the 7,047 an unfiltered call gives, and the fact
    # was unreachable by any phrase in the task text.
    #
    # Aliases only, never a fallback to a default bucket: an unknown name still
    # returns {} so a genuinely unsupported physics is not answered with
    # confident advice about a different equation.
    _ALIASES = {
        "anisotropic_diffusion": "poisson",
        "anisotropic diffusion": "poisson",
        "diffusion": "poisson",
        "laplace": "poisson",
        "laplacian": "poisson",
        "steady_diffusion": "poisson",
        "conduction": "heat",
        "elasticity": "linear_elasticity",
        "linear elasticity": "linear_elasticity",
        "advection_diffusion": "convection_diffusion",
        "advection-diffusion": "convection_diffusion",
        "biharmonic": "hdivdiv",
    }

    def get_knowledge(self, physics: str) -> dict:
        key = (physics or "").strip()
        if key in KNOWLEDGE:
            return KNOWLEDGE[key]
        alias = self._ALIASES.get(key.lower().replace("-", "_"))
        if alias is None:
            alias = self._ALIASES.get(key.lower())
        return KNOWLEDGE.get(alias, {}) if alias else {}

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        key = f"{physics}_{variant}"
        gen = GENERATORS.get(key)
        if not gen:
            available = ", ".join(sorted(GENERATORS.keys()))
            raise ValueError(f"Unknown variant '{key}'. Available: {available}")
        return gen(params)

    def validate_input(self, content: str) -> list[str]:
        errors = []
        if "ngsolve" not in content and "from ngsolve" not in content:
            errors.append("Script should import from ngsolve")
        try:
            compile(content, "<ngsolve_input>", "exec")
        except SyntaxError as e:
            errors.append(f"Python syntax error: {e}")
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        python = _find_ngsolve_python()
        if not python:
            return JobHandle(
                job_id=str(uuid.uuid4())[:8],
                backend_name="ngsolve",
                work_dir=work_dir,
                status="failed",
                error="ngsolve not found",
            )

        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / "solve.py"
        script_path.write_text(input_content)

        job_id = str(uuid.uuid4())[:8]
        job = JobHandle(
            job_id=job_id,
            backend_name="ngsolve",
            work_dir=work_dir,
            status="running",
        )

        # NGSolve supports MPI via mpi4py but most problems run serial
        cmd = [str(python), str(script_path)]

        start = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
                start_new_session=True,
                stdin=asyncio.subprocess.DEVNULL,
            )

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
            job.error = f"Timed out after {timeout}s"
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


# ─── Registration ────────────────────────────────────────────────────────

def register():
    backend = NgsolveBackend()
    register_backend(backend, aliases=["ngsolve", "ngs"])
    logger.info("NGSolve backend registered")
