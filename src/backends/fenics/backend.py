"""
FEniCS (dolfinx) solver backend.

Generates Python scripts using the dolfinx API, executes them,
and collects VTU/XDMF output files.

Template generation is delegated to the ``generators`` sub-package.
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

logger = logging.getLogger("openpaso.fenics")

# Conda environment with dolfinx
_CONDA_PREFIX = os.environ.get("FENICS_CONDA_PREFIX", "")
_FENICS_PYTHON = os.environ.get("FENICS_PYTHON", "")


# THE INSTALLED DOLFINX VERSION, ASKED ONCE PER INTERPRETER. get_knowledge runs once per
# physics, and knowledge(topic='pitfalls') walks every physics: 226 calls, each of which
# started the FEniCSx interpreter to import dolfinx (about 0.5 s). Measured 2026-09-28: one
# pitfalls lookup took 107-115 s, and a critic's two lookups held a web-interface run for
# over two minutes. A version found is kept for the life of the process; a probe that
# failed is kept for a minute, so a broken or busy interpreter is not asked 226 times in
# a row and a repaired one is seen again.
_DOLFINX_VERSION: dict[str, tuple[Optional[str], float]] = {}
_DOLFINX_PROBE_RETRY_S = 60.0


def _installed_dolfinx_version(python: Path) -> Optional[str]:
    """dolfinx.__version__ as `python` imports it, or None."""
    import subprocess
    key = str(python)
    hit = _DOLFINX_VERSION.get(key)
    if hit is not None and (hit[0] is not None or time.time() - hit[1] < _DOLFINX_PROBE_RETRY_S):
        return hit[0]
    ver = None
    try:
        r = subprocess.run([key, "-c", "import dolfinx; print(dolfinx.__version__)"],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            ver = r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        ver = None
    _DOLFINX_VERSION[key] = (ver, time.time())
    return ver


def _mpi_launcher_for(python: Path) -> Optional[str]:
    """The MPI launcher that belongs to this interpreter's own MPI.

    A conda FEniCSx environment carries its own MPI (MPICH from conda-forge) and its own
    mpiexec beside the interpreter; the first mpirun on PATH is usually the system's. Measured
    2026-09-28: with np=4 the system's Open MPI 4.0.3 mpirun started the environment's MPICH
    interpreter and every rank died in MPI_Init ("launcher not compatible with PMI1 client"),
    0.4 s into run_simulation -- every FEniCSx run with np > 1 on this machine. The launcher
    beside the interpreter comes first; PATH is the fallback for an install that has none.
    """
    for name in ("mpiexec", "mpirun"):
        beside = python.parent / name
        if beside.is_file() and os.access(beside, os.X_OK):
            return str(beside)
    return shutil.which("mpirun")


def _find_fenics_python() -> Optional[Path]:
    """Locate the Python binary with dolfinx installed.

    FEniCS (dolfinx) is typically in a conda env, not in the server's venv.
    Resolution order: ``FENICS_PYTHON`` env var -> ``FENICS_CONDA_PREFIX``
    env var -> any conda env whose name contains "fenics" or "dolfinx"
    (case-insensitive) -> the server's own Python.

    The fuzzy name match covers the realistic naming spectrum users
    create -- ``fenics``, ``fenicsx``, ``dolfinx``, ``ofa-fenicsx``,
    ``my-fenics``, ``fenics-0.9``, etc.  A strictly-named ``fenics``
    env was the old behavior and quietly missed every other naming.
    """
    import sys

    # 1. Explicit env var (e.g., FENICS_PYTHON=/path/to/conda/envs/fenics/bin/python)
    if _FENICS_PYTHON and Path(_FENICS_PYTHON).is_file():
        return Path(_FENICS_PYTHON)

    # 2. Conda env path
    if _CONDA_PREFIX:
        p = Path(_CONDA_PREFIX) / "bin" / "python"
        if p.is_file():
            return p

    # 3. Search common conda env locations.  Match any env whose name
    # contains "fenics" or "dolfinx" so the user doesn't have to name
    # the env literally "fenics".  Iterating envs/* is cheap (filesystem
    # listing only; no import-testing at startup).  We sort the listing
    # so a machine with multiple matching envs always picks the same
    # one (Path.iterdir order is not guaranteed).
    home = Path.home()
    for conda_dir in [home / "miniconda3", home / "miniforge3", home / "anaconda3"]:
        envs_dir = conda_dir / "envs"
        if not envs_dir.is_dir():
            continue
        for env in sorted(envs_dir.iterdir()):
            if not env.is_dir():
                continue
            name = env.name.lower()
            if "fenics" in name or "dolfinx" in name:
                p = env / "bin" / "python"
                if p.is_file():
                    return p

    # 4. Last resort: the server's own Python -- but only if it can
    # actually import dolfinx.  Returning sys.executable unconditionally
    # would make ``check_availability``'s "No Python with dolfinx
    # found" branch dead code; we'd always reach the subprocess import
    # check and report a less actionable error.
    import subprocess
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import dolfinx"], stdin=subprocess.DEVNULL,
            capture_output=True, timeout=5,
        )
        if r.returncode == 0:
            return Path(sys.executable)
    except Exception:
        pass
    return None


# ---- Physics capabilities (used by supported_physics) ----
# Defined here so the backend class can return them without importing generators.
_PHYSICS_CAPABILITIES = [
    PhysicsCapability(
        name="poisson",
        description="Poisson equation / diffusion",
        spatial_dims=[2, 3],
        element_types=["triangle", "tetrahedron", "quadrilateral", "hexahedron"],
        template_variants=["2d", "3d", "l_domain", "rectangle"],
    ),
    PhysicsCapability(
        name="linear_elasticity",
        description="Linear elasticity (small strain)",
        spatial_dims=[2, 3],
        element_types=["triangle", "tetrahedron", "quadrilateral", "hexahedron"],
        template_variants=["2d", "3d", "plate_hole", "thick_beam"],
    ),
    PhysicsCapability(
        name="heat",
        description="Heat conduction (steady / transient)",
        spatial_dims=[2, 3],
        element_types=["triangle", "tetrahedron"],
        template_variants=["2d_steady", "2d_transient", "rectangle"],
    ),
    PhysicsCapability(
        name="navier_stokes",
        description="Incompressible Navier-Stokes (cavity, channel with obstacle; the unsteady wake "
                    "is a contract: mesh, forces and pictures served, the solve is yours)",
        spatial_dims=[2, 3],
        element_types=["triangle", "tetrahedron"],
        template_variants=["2d", "3d", "channel_cylinder", "channel_cylinder_transient"],
    ),
    PhysicsCapability(
        name="thermal_structural",
        description="Coupled thermal-structural (heat -> thermal expansion)",
        spatial_dims=[2],
        element_types=["triangle"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="hyperelasticity",
        description="Nonlinear hyperelasticity (Neo-Hookean, large deformation)",
        spatial_dims=[3],
        element_types=["tetrahedron"],
        template_variants=["3d"],
    ),
    PhysicsCapability(
        name="stokes",
        description="Stokes flow with Taylor-Hood P2/P1 (lid-driven cavity)",
        spatial_dims=[2],
        element_types=["triangle"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="convection_diffusion",
        description="Convection-diffusion (SUPG stabilized)",
        spatial_dims=[2],
        element_types=["triangle"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="eigenvalue",
        description="Eigenvalue problems (Laplace) via SLEPc",
        spatial_dims=[2],
        element_types=["triangle"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="biharmonic",
        description="Biharmonic equation (4th order) via interior penalty DG",
        spatial_dims=[2],
        element_types=["triangle"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="mixed_poisson",
        description="Mixed Poisson / Darcy flow (Raviart-Thomas + DG pressure)",
        spatial_dims=[2],
        element_types=["triangle"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="reaction_diffusion",
        description="Two-species reaction-diffusion system (coupled, transient)",
        spatial_dims=[2],
        element_types=["triangle"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="dg_methods",
        description="Discontinuous Galerkin for advection-dominated diffusion (upwind flux, interior penalty)",
        spatial_dims=[2],
        element_types=["triangle"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="contact",
        description="Contact / obstacle problem via smooth penalty method (Newton iteration)",
        spatial_dims=[2],
        element_types=["triangle"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="multiphase",
        description="Two-phase flow via Allen-Cahn phase-field (interface tracking, transient)",
        spatial_dims=[2],
        element_types=["triangle"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="time_dependent_heat",
        description="Transient heat equation with backward Euler, Robin convective BC, volumetric sources",
        spatial_dims=[2],
        element_types=["triangle"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="cahn_hilliard",
        description="Cahn-Hilliard phase separation: mixed (phi, mu) formulation, double-well potential",
        spatial_dims=[2],
        element_types=["triangle"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="nonlinear_pde",
        description="General nonlinear PDE with Newton solver and UFL automatic differentiation",
        spatial_dims=[2],
        element_types=["triangle"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="magnetostatics",
        description="Magnetostatics: 2D scalar Az curl-curl formulation, spatially varying permeability",
        spatial_dims=[2],
        element_types=["triangle"],
        template_variants=["2d"],
    ),
    # ── 2026-06-01: helmholtz + maxwell have detailed
    #    deep_knowledge entries but were not in the
    #    capabilities list, so users browsing discover(physics
    #    fenics) couldn't see them. Knowledge-tool fetch
    #    worked (deep_knowledge fallback), but discover did
    #    not list them. Exposing them here closes the gap.
    PhysicsCapability(
        name="helmholtz",
        description=(
            "Helmholtz equation: -laplacian(u) - k^2*u = f. "
            "Acoustic / optical wave propagation. Indefinite "
            "system — GMRES or direct, NOT CG. May be complex-"
            "valued; needs PETSc compiled with "
            "--with-scalar-type=complex."
        ),
        spatial_dims=[2, 3],
        element_types=["triangle", "tetrahedron"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="maxwell",
        description=(
            "Maxwell's equations (curl-curl). Requires H(curl) "
            "(Nedelec / N1curl, basix.ElementFamily.N1E) "
            "elements for tangential continuity. Complex-valued "
            "forms need a complex-PETSc build."
        ),
        spatial_dims=[2, 3],
        element_types=["triangle", "tetrahedron"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="nearly_incompressible_elasticity",
        description=(
            "Nearly-incompressible elasticity (Poisson ratio "
            "approaching 0.5). Standard primal P1/P2 locks; "
            "needs mixed (u, p) Taylor-Hood / MINI or a "
            "displacement-pressure split with stable element "
            "pair (otherwise volumetric locking)."
        ),
        spatial_dims=[2, 3],
        element_types=["triangle", "tetrahedron"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="fracture",
        description=(
            "Phase-field fracture mechanics. Coupled "
            "displacement / damage formulation with a diffuse "
            "crack representation (no remeshing). Extensions: "
            "PhaseFieldX library."
        ),
        spatial_dims=[2, 3],
        element_types=["triangle", "tetrahedron"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="stokes_darcy",
        description=(
            "Coupled Stokes-Darcy for free fluid / porous "
            "medium interaction. Beavers-Joseph-Saffman "
            "interface conditions."
        ),
        spatial_dims=[2, 3],
        element_types=["triangle", "tetrahedron"],
        template_variants=["2d"],
    ),
    PhysicsCapability(
        name="element_survey",
        description=(
            "Element survey: builds every basix element family this FEniCSx "
            "install declares and checks each one (its quadrature integrates "
            "the reference cell, its mass matrix is SPD, and the space "
            "reproduces a constant -- measured on a richer rule than the one "
            "that built the matrix). Answers 'which families does this install "
            "actually have, and how does each behave' without guessing."
        ),
        spatial_dims=[2],
        element_types=["all"],
        template_variants=["default"],
    ),
    PhysicsCapability(
        name="matrix_free_poisson",
        description=(
            "Matrix-free conjugate-gradient Poisson solver. "
            "Builds A as a callable action_A(x, y) via "
            "ufl.action(a, ui) — no global sparse assembly. "
            "Mirrors dolfinx demo_poisson_matrix_free.py."
        ),
        spatial_dims=[2],
        element_types=["Lagrange"],
        template_variants=["2d"],
    ),
]


class FenicsBackend(SolverBackend):

    def name(self) -> str:
        return "fenics"

    def display_name(self) -> str:
        return "FEniCSx (dolfinx)"

    def check_availability(self) -> tuple[BackendStatus, str]:
        python = _find_fenics_python()
        hint = (
            "  Set one of:\n"
            "    FENICS_PYTHON=/path/to/conda/envs/<env>/bin/python  (explicit, recommended)\n"
            "    FENICS_CONDA_PREFIX=/path/to/conda/envs/<env>        (env root)\n"
            "  Or rename your env to one whose name contains 'fenics' or 'dolfinx'\n"
            "  (auto-detected in ~/miniconda3, ~/miniforge3, ~/anaconda3 under envs/)."
        )
        if not python:
            return BackendStatus.NOT_INSTALLED, "No Python with dolfinx found.\n" + hint

        # Quick import check
        import subprocess
        try:
            result = subprocess.run(
                [str(python), "-c", "import dolfinx; print(dolfinx.__version__)"], stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                ver = result.stdout.strip()
                return BackendStatus.AVAILABLE, f"dolfinx {ver} at {python}"
            else:
                err = result.stderr.strip()
                # macOS + VPN (Mac stress audit 2026-07-18): mpich's libfabric
                # can pick a VPN tunnel interface (utun*) as the default NIC and
                # crash MPI on import/finalize — the probe then reports the
                # backend missing although dolfinx is fine. Surface the fix.
                if any(k in err for k in ("OFI poll failed", "utun",
                                          "MPIDI_OFI", "libfabric")):
                    hint = (
                        "  macOS/VPN: libfabric picked a VPN tunnel interface "
                        "(utun*). Set in the MCP server environment:\n"
                        "    FI_PROVIDER=sockets FI_SOCKETS_IFACE=en0\n"
                    ) + hint
                return BackendStatus.NOT_INSTALLED, (
                    f"dolfinx import failed at {python}: "
                    f"{err}\n" + hint
                )
        except subprocess.TimeoutExpired:
            return BackendStatus.NOT_INSTALLED, (
                f"dolfinx import probe timed out at {python} (10 s). On macOS "
                "the FIRST import after install can exceed this (Gatekeeper "
                "scan of fresh dylibs; VPN/libfabric interface selection). "
                "Warm it up once and retry:\n"
                f"    {python} -c 'import dolfinx'\n" + hint
            )
        except Exception as e:
            return BackendStatus.NOT_INSTALLED, f"Check failed at {python}: {e}\n" + hint

    @staticmethod
    def _convert_xdmf_to_vtu(work_dir: Path):
        """Convert XDMF+HDF5 output to VTU for universal PyVista compatibility.

        FEniCS XDMF uses XInclude which crashes PyVista's VTK reader.
        This reads the HDF5 data directly and writes a standard VTU via meshio.
        """
        import h5py
        import numpy as np

        xdmf_files = list(work_dir.glob("*.xdmf"))
        for xdmf_path in xdmf_files:
            h5_path = xdmf_path.with_suffix(".h5")
            if not h5_path.exists():
                continue

            try:
                import meshio
                with h5py.File(h5_path, "r") as h5:
                    # Read mesh topology and geometry
                    mesh_grp = h5["Mesh"]
                    mesh_name = list(mesh_grp.keys())[0]
                    topo = np.array(mesh_grp[mesh_name]["topology"])
                    geom = np.array(mesh_grp[mesh_name]["geometry"])

                    # Pad 2D geometry to 3D (meshio requires 3D points)
                    if geom.shape[1] == 2:
                        geom = np.column_stack([geom, np.zeros(geom.shape[0])])

                    # Determine cell type from topology shape and geometry dim
                    n_nodes_per_cell = topo.shape[1]
                    is_2d = (geom[:, 2].max() - geom[:, 2].min()) < 1e-10
                    if n_nodes_per_cell == 3:
                        cell_type = "triangle"
                    elif n_nodes_per_cell == 4:
                        cell_type = "quad" if is_2d else "tetra"
                    elif n_nodes_per_cell == 8:
                        cell_type = "hexahedron"
                    elif n_nodes_per_cell == 2:
                        cell_type = "line"
                    else:
                        cell_type = "triangle"  # fallback

                    cells = [meshio.CellBlock(cell_type, topo)]

                    # Read function data
                    point_data = {}
                    if "Function" in h5:
                        for func_name in h5["Function"]:
                            # Get the last timestep
                            timesteps = sorted(h5["Function"][func_name].keys())
                            data = np.array(h5["Function"][func_name][timesteps[-1]])
                            if data.ndim == 2 and data.shape[1] == 1:
                                data = data.flatten()
                            point_data[func_name] = data

                m = meshio.Mesh(points=geom, cells=cells, point_data=point_data)
                vtu_path = xdmf_path.with_suffix(".vtu")
                meshio.write(str(vtu_path), m)
                logger.info(f"Converted {xdmf_path.name} -> {vtu_path.name}")
            except Exception as e:
                logger.warning(f"XDMF->VTU conversion failed for {xdmf_path}: {e}")

    def input_format(self) -> InputFormat:
        return InputFormat.PYTHON

    def get_version(self) -> Optional[str]:
        python = _find_fenics_python()
        if not python:
            return None
        import subprocess
        try:
            r = subprocess.run(
                [str(python), "-c", "import dolfinx; print(dolfinx.__version__)"], stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=10
            )
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    def supported_physics(self) -> list[PhysicsCapability]:
        return list(_PHYSICS_CAPABILITIES)

    def get_knowledge(self, physics: str) -> dict:
        # Detect installed version and add API notes
        version_note = ""
        try:
            p = _find_fenics_python()
            if p:
                ver = _installed_dolfinx_version(p)
                if ver:
                    version_note = (
                        f"\n\n**Installed dolfinx version: {ver}**\n"
                        "API notes for 0.9+/0.10+:\n"
                        "- NonlinearProblem requires petsc_options_prefix kwarg\n"
                        "- Use problem.solve() directly, NOT separate NewtonSolver\n"
                        "- LinearProblem also requires petsc_options_prefix\n"
                        # 2026-08-03 adversarial re-verification against dolfinx
                        # 0.10.0 / basix 0.10.0: `interpolation_points` is NOT an
                        # attribute of a basix.ufl element at all any more (neither
                        # property nor method) — AttributeError:
                        # '_BasixElement' object has no attribute
                        # 'interpolation_points'. The points now live on the
                        # wrapped basix element as the `points` property.
                        "- element has NO .interpolation_points in basix 0.10; "
                        "use element.basix_element.points (ndarray property)\n"
                        "- For VTU output use VTXWriter or XDMFFile, read with pyvista (not meshio)\n"
                        "- fem.assemble_scalar returns the RANK-LOCAL value; wrap "
                        "in comm.allreduce(..., op=MPI.SUM) for a global norm\n"
                    )
        except Exception:
            pass

        # ─────────────────────────────────────────────────────────
        # Source-of-truth ordering (fenics-only, audit 2026-06-02):
        #   1. src/tools/deep_knowledge.py — CANONICAL for the 17
        #      physics covered there (hyperelasticity, stokes, poisson,
        #      heat, helmholtz, eigenvalue, reaction_diffusion,
        #      convection_diffusion, navier_stokes, cahn_hilliard,
        #      biharmonic, fracture, nearly_incompressible_elasticity,
        #      maxwell, stokes_darcy, thermal_structural, contact).
        #      verify_signal_clauses + the pitfall-DB audit gate read
        #      from HERE; the matching KNOWLEDGE dicts in
        #      src/backends/fenics/generators/*.py for these physics
        #      are DEAD CODE and may have drifted — do NOT edit them
        #      and expect a behaviour change.
        #   2. Generator-level KNOWLEDGE — canonical for the 6
        #      exposed physics NOT in deep_knowledge.py
        #      (mixed_poisson, dg_methods, multiphase,
        #      time_dependent_heat, nonlinear_pde, magnetostatics).
        #      These DO surface via prepare_simulation/knowledge.
        #      The elasticity generator file also has its own
        #      KNOWLEDGE but is shared-implementation code for the
        #      `linear_elasticity` template (which itself lives in
        #      deep_knowledge.py), so it is NOT canonical.
        # ─────────────────────────────────────────────────────────
        try:
            from tools.deep_knowledge import get_deep_fenics_knowledge
            deep = get_deep_fenics_knowledge(physics)
            if deep:
                if version_note:
                    deep["_version_info"] = version_note
                return deep
        except (ImportError, Exception):
            pass

        # Fall back to generator-level knowledge (canonical for the
        # 7 physics listed in the comment above; dead code for the
        # 17 deep_knowledge.py-covered physics).
        from .generators import get_knowledge as gen_knowledge, GENERAL_KNOWLEDGE
        try:
            return gen_knowledge(physics)
        except KeyError:
            pass

        # General knowledge fallback
        if physics == "_general":
            return GENERAL_KNOWLEDGE

        return {}

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        from .generators import generate_script
        try:
            return generate_script(physics, variant, params)
        except KeyError:
            from .generators import list_all_physics
            raise ValueError(
                f"No FEniCS generator for physics={physics!r}. "
                f"Available: {list_all_physics()}"
            )

    def validate_input(self, content: str) -> list[str]:
        errors = []
        if "import dolfinx" not in content and "from dolfinx" not in content:
            errors.append("Script does not import dolfinx")
        if "def " not in content and "solve" not in content.lower():
            errors.append("Script does not appear to solve anything")
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        python = _find_fenics_python()
        if not python:
            job = JobHandle(
                job_id=str(uuid.uuid4())[:8],
                backend_name="fenics",
                work_dir=work_dir,
                status="failed",
                error="dolfinx not found",
            )
            return job

        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / "solve.py"
        script_path.write_text(input_content)

        job_id = str(uuid.uuid4())[:8]
        job = JobHandle(
            job_id=job_id,
            backend_name="fenics",
            work_dir=work_dir,
            status="running",
        )

        mpirun = _mpi_launcher_for(Path(python))
        if np > 1 and mpirun:
            cmd = [mpirun, "-np", str(np), str(python), str(script_path)]
        else:
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
                # Convert XDMF->VTU for PyVista compatibility
                try:
                    self._convert_xdmf_to_vtu(work_dir)
                except Exception as e:
                    logger.warning(f"XDMF->VTU post-conversion failed: {e}")
            else:
                job.status = "failed"
                job.error = stderr.decode(errors="replace")[-2000:]

            # Save logs
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
        # Prefer VTU (converted from XDMF) over raw XDMF. Include ADIOS2 *.bp
        # (a directory) too: dolfinx forbids higher-order (e.g. P2 Taylor-Hood)
        # fields in XDMF and writes VTX/.bp instead, so without this a correct
        # P2 solve produced "no output" and was wrongly flagged NOT VERIFIED.
        for ext in ["*.vtu", "*.pvd", "*.pvtu", "*.xdmf", "*.bp"]:
            results.extend(job.work_dir.rglob(ext))
        return sorted_by_step(results)

    def precice_participant(self) -> dict:
        """FEniCSx as a preCICE participant — a dolfinx solve advanced each window,
        exchanging an interface field (e.g. read heat flux as Neumann BC, write surface
        temperature). Verified pattern (works with the fenicsxprecice adapter too)."""
        return {
            "description": "FEniCSx (dolfinx) preCICE participant for the FEM side of a coupling",
            "exchange_loop": (
                "import precice, numpy as np\n"
                "from dolfinx import fem, mesh; from dolfinx.fem.petsc import LinearProblem\n"
                "# build mesh + functionspace + the transient/steady weak form; the coupled\n"
                "# field enters as a BC: a Neumann flux (read) or Dirichlet value (read), and\n"
                "# the surface response (write) is sampled from the solution.\n"
                "qheat = fem.Constant(domain, 0.0)          # updated each window from preCICE\n"
                "p = precice.Participant('Solid','precice-config.xml',0,1)\n"
                "vid = p.set_mesh_vertices('Solid-Mesh', np.array([[0.0,0.0]]))\n"
                "p.initialize()\n"
                "while p.is_coupling_ongoing():\n"
                "    dt = p.get_max_time_step_size()\n"
                "    qheat.value = float(p.read_data('Solid-Mesh','Heat-Flux',vid,dt)[0])\n"
                "    Th = problem.solve()                    # advance the FEM solve one window\n"
                "    Tw = surface_value(Th)\n"
                "    p.write_data('Solid-Mesh','Wall-Temperature',vid,np.array([Tw])); p.advance(dt)\n"
                "p.finalize()"
            ),
            "notes": ("dolfinx 0.10: use fem.functionspace + LinearProblem(..., "
                      "petsc_options_prefix=...). For per-interface-node exchange use the "
                      "fenicsxprecice Adapter; for a scalar/lumped interface a 1-vertex mesh "
                      "suffices. Set LD_LIBRARY_PATH to /opt/precice/lib; pyprecice must match "
                      "the libprecice version."),
        }


def register():
    """Register the FEniCS backend with the global registry."""
    register_backend(
        FenicsBackend(),
        aliases=["fenics", "fenicsx", "dolfinx", "dolfin"],
    )
