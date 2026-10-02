"""
MCP tools for running simulations across backends.
"""

import json
import time
import uuid
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from core.registry import get_backend
from core.backend import JobHandle, error_excerpt

# Job tracking (in-memory)
_jobs: dict[str, JobHandle] = {}

# Output directory
from core.output_paths import output_dir as _output_dir  # noqa: E402
_OUTPUT_DIR = _output_dir("simulation_outputs")


def register_simulation_tools(mcp: FastMCP):

    @mcp.tool()
    async def run_with_generator(solver: str, generator_script: str,
                                  job_name: str = "", np: int = 1,
                                  timeout=None,
                                  critic_approved: bool = False) -> str:
        """Run a generator script first, then run the solver on its output.

        IMPORTANT — Before calling this tool, you MUST have an independent
        critic review the simulation setup. The critic should check every
        parameter against literature, verify units, discretization, boundary
        conditions, and loading. Set critic_approved=True only after a critic
        agent has explicitly approved the setup. Skipping this step is the
        most common source of wrong simulation results.

        Use this whenever the simulation input needs to be generated
        programmatically — meshes, particle grids, complex geometries,
        parametric studies, or any case where the input is too large or
        complex to write inline.

        The generator script (Python) creates the input file in the
        working directory, then the solver runs on it automatically.

        This is the PREFERRED workflow for production simulations across
        ALL solvers — it separates input generation from execution and
        allows arbitrarily complex setups.

        Args:
            solver: Backend name (e.g. 'fourc', 'fenics', 'dealii', etc.)
            generator_script: Python script that generates the input file.
                Must write the final input to 'input.4C.yaml' (for 4C),
                'solve.py' (for FEniCS/NGSolve/etc.), or equivalent.
            job_name: Optional name for the job directory
            np: Number of MPI processes (default 1)
            timeout: Maximum runtime in seconds (default 300)
            critic_approved: Set to True ONLY after a critic agent has
                reviewed and approved the simulation setup. If False,
                a warning is included in the output.

        Returns:
            Job status with output location.
        """
        import subprocess
        import sys

        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        status, msg = backend.check_availability()
        if status.value != "available":
            return f"Solver {solver} is not available: {msg}"

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = job_name or f"{solver}_gen_{ts}"
        work_dir = _OUTPUT_DIR / name
        work_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Write and run generator script
        gen_path = work_dir / "generate_input.py"
        gen_path.write_text(generator_script)

        python = sys.executable
        gen_result = subprocess.run(
            [python, str(gen_path)],
            capture_output=True, text=True,
            cwd=str(work_dir),
            stdin=subprocess.DEVNULL,
        )

        if gen_result.returncode != 0:
            return json.dumps({
                "status": "failed",
                "phase": "generator",
                "error": gen_result.stderr[-500:],
                "work_dir": str(work_dir),
            }, indent=2)

        # Step 2: Find the generated input file (backend-aware; see issue #39)
        from core.backend import find_generated_input
        input_file = find_generated_input(work_dir, backend)

        if not input_file:
            return json.dumps({
                "status": "failed",
                "phase": "generator",
                "error": "Generator did not produce an input file",
                "work_dir": str(work_dir),
                "files": sorted(f.name for f in work_dir.iterdir())[:50],
            }, indent=2)

        input_content = input_file.read_text()

        # Step 3: Run solver
        job = await backend.run(input_content, work_dir, np=np, timeout=timeout)
        _jobs[job.job_id] = job

        result = {
            "job_id": job.job_id,
            "solver": solver,
            "status": job.status,
            "work_dir": str(job.work_dir),
            "elapsed": f"{job.elapsed:.2f}s" if job.elapsed else None,
            "generator": "completed",
            "input_file": input_file.name,
            "critic_review": "approved" if critic_approved else "SKIPPED — no critic reviewed this setup. Results may be wrong.",
        }
        if job.error:
            result["error"] = error_excerpt(job.error)
        if job.status == "completed":
            result_files = backend.get_result_files(job)
            result["output_files"] = [f.name for f in result_files]

        return json.dumps(result, indent=2)

    @mcp.tool()
    async def run_simulation(solver: str, input_content: str,
                             job_name: str = "", np: int = 1,
                             timeout=None,
                             critic_approved: bool = False) -> str:
        """Run a FEM simulation using a specific solver backend.

        IMPORTANT — Before calling this tool, you MUST have an independent
        critic review the simulation setup. The critic should check every
        parameter against literature, verify units, discretization, boundary
        conditions, and loading. Set critic_approved=True only after a critic
        agent has explicitly approved the setup. Skipping this step is the
        most common source of wrong simulation results.

        Args:
            solver: Backend name (e.g. 'fenics', 'fourc', 'dealii', 'febio')
            input_content: The generated input (Python script / YAML / C++ / XML)
            job_name: Optional name for the job directory
            np: Number of MPI processes (default 1)
            timeout: Maximum runtime in seconds (default 300)
            critic_approved: Set to True ONLY after a critic agent has
                reviewed and approved the simulation setup. If False,
                a warning is included in the output.

        Returns:
            Job status with output location.
        """
        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        status, msg = backend.check_availability()
        if status.value != "available":
            return f"Solver {solver} is not available: {msg}"

        # Create work directory
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = job_name or f"{solver}_{ts}"
        work_dir = _OUTPUT_DIR / name
        work_dir.mkdir(parents=True, exist_ok=True)

        # Validate input before running
        validation_errors = backend.validate_input(input_content)
        validation_warnings = []
        if validation_errors:
            validation_warnings.extend(validation_errors)

        # Run
        job = await backend.run(input_content, work_dir, np=np, timeout=timeout)
        _jobs[job.job_id] = job

        result = {
            "job_id": job.job_id,
            "solver": solver,
            "status": job.status,
            "work_dir": str(job.work_dir),
            "elapsed": f"{job.elapsed:.2f}s" if job.elapsed else None,
            "critic_review": "approved" if critic_approved else "SKIPPED — no critic reviewed this setup. Results may be wrong.",
        }
        if job.error:
            result["error"] = error_excerpt(job.error)
        if job.status == "completed":
            result_files = backend.get_result_files(job)
            result["output_files"] = [f.name for f in result_files]
            if not result_files:
                result["warning"] = (
                    "Simulation completed but no VTU/VTK output files found. "
                    "Check if the IO/output section is configured correctly."
                )
        if validation_warnings:
            result["validation_warnings"] = validation_warnings

        return json.dumps(result, indent=2)

    @mcp.tool()
    def check_simulation_quality(
        solver: str,
        input_content: str,
        domain_size: float = 1.0,
        n_elements: int = 32,
        element_order: int = 1,
        physics: str = "poisson",
        time_step: float = 0.0,
        wave_speed: float = 0.0,
    ) -> str:
        """Run quality checks on a simulation BEFORE running it.

        Checks mesh resolution, time step stability, material consistency,
        and output configuration. Call this before run_simulation to catch
        common issues that lead to inaccurate or failed simulations.

        Args:
            solver: Backend name.
            input_content: Generated input content.
            domain_size: Characteristic length of the domain.
            n_elements: Number of elements/particles per direction.
            element_order: Polynomial order of elements.
            physics: Physics type for resolution guidelines.
            time_step: Time step size (0 = not applicable).
            wave_speed: Wave speed for CFL check (0 = not applicable).
        """
        from core.quality_checks import (
            check_mesh_resolution,
            check_particle_resolution,
            check_time_step,
        )

        warnings = []

        # Input validation
        backend = get_backend(solver)
        if backend:
            errors = backend.validate_input(input_content)
            if errors:
                warnings.extend([f"Validation: {e}" for e in errors])

        # Resolution check
        if physics == "peridynamics":
            h = domain_size / n_elements if n_elements > 0 else domain_size
            warnings.extend(check_particle_resolution(
                domain_size, h, physics="peridynamics"))
        else:
            warnings.extend(check_mesh_resolution(
                domain_size, n_elements, physics, element_order))

        # Time step check
        if time_step > 0 and wave_speed > 0:
            h = domain_size / n_elements if n_elements > 0 else domain_size
            warnings.extend(check_time_step(
                time_step, h, wave_speed=wave_speed, scheme="explicit"))

        # Output configuration check
        if solver == "fourc":
            # A particle deck is exempt: see directly below. Warning it to add the section told a
            # correct deck to go and look for one (found by Copilot review, Hereon PR #55).
            # NOTE: register_simulation_tools is not called by the server, and this tool imports
            # quality-check helpers that no longer exist, so it is unreachable as shipped.
            if ("IO/RUNTIME VTK OUTPUT" not in input_content
                    and "PARTICLE DYNAMIC" not in input_content):
                warnings.append(
                    "No IO/RUNTIME VTK OUTPUT section found in 4C input. "
                    "Add this section to get ParaView-readable VTU/VTP output files."
                )
            # A Particle deck needs NO output section: 4C writes the
            # particle .pvd/.vtu series unconditionally at RESULTSEVERY.
            # The failure mode runs the other way — the section some
            # sources recommend does not exist, and writing it aborts the
            # run before the first step.
            if "IO/RUNTIME VTK OUTPUT/PARTICLES" in input_content:
                warnings.append(
                    "IO/RUNTIME VTK OUTPUT/PARTICLES is not a valid 4C "
                    "section. The run will abort at input parsing with "
                    "\"Section 'IO/RUNTIME VTK OUTPUT/PARTICLES' is not "
                    "a valid section name.\" Delete it: particle output "
                    "is unconditional and its interval is RESULTSEVERY "
                    "in PARTICLE DYNAMIC."
                )

        if not warnings:
            return "All quality checks passed. Simulation is ready to run."

        result = "## Quality Check Warnings\n\n"
        for i, w in enumerate(warnings, 1):
            result += f"{i}. {w}\n"
        result += "\nFix these issues before running for best results."
        return result

    @mcp.tool()
    def get_job_status(job_id: str) -> str:
        """Check the status of a simulation job.

        Args:
            job_id: The job ID returned by run_simulation.
        """
        job = _jobs.get(job_id)
        if not job:
            return f"Unknown job: {job_id}"

        backend = get_backend(job.backend_name)
        result = {
            "job_id": job.job_id,
            "solver": job.backend_name,
            "status": job.status,
            "work_dir": str(job.work_dir),
            "elapsed": f"{job.elapsed:.2f}s" if job.elapsed else None,
        }
        if job.status == "completed" and backend:
            result_files = backend.get_result_files(job)
            result["output_files"] = [f.name for f in result_files]
        if job.error:
            result["error"] = error_excerpt(job.error)

        return json.dumps(result, indent=2)

    @mcp.tool()
    def list_jobs() -> str:
        """List all simulation jobs from this session."""
        if not _jobs:
            return "No jobs recorded."

        lines = ["| Job ID | Solver | Status | Duration |",
                 "|--------|--------|--------|----------|"]
        for jid, job in _jobs.items():
            dur = f"{job.elapsed:.1f}s" if job.elapsed else "-"
            lines.append(f"| {jid} | {job.backend_name} | {job.status} | {dur} |")

        return "\n".join(lines)
