"""
MCP tools for advanced simulation workflows across ALL backends.

Ported from 4c-ai-interface and generalized for multi-solver use:
- get_module_info: Detailed physics module information (ALL backends)
- list_template_variants: Available templates per physics per backend
- parameter_study: Sweep a parameter and compare results
- query_documentation: Search solver documentation
- restart_simulation: Continue a simulation from checkpoint
- validate_physics: Physics-aware parameter validation
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import yaml
from mcp.server.fastmcp import FastMCP

from core.backend import error_excerpt
from core.registry import get_backend, available_backends
from core.backend import BackendStatus

logger = logging.getLogger("openpaso.workflows")

FOURC_ROOT = Path(os.environ.get("FOURC_ROOT", ""))


def register_workflow_tools(mcp: FastMCP):

    # ═══════════════════════════════════════════════════════════════════════
    # get_module_info — detailed info for ANY physics on ANY backend
    # ═══════════════════════════════════════════════════════════════════════

    @mcp.tool()
    def get_module_info(solver: str, module: str) -> str:
        """Get detailed information about a specific physics module on a specific solver.

        Returns: description, problem type, required YAML sections / script structure,
        dynamics parameters, geometry section, element category, typical materials,
        difficulty level, and available template variants.

        This is the most detailed module-level info tool. Use it before setting up
        any simulation.

        Args:
            solver: Backend name ('fenics', 'fourc', 'dealii', 'febio')
            module: Physics key (e.g. 'poisson', 'navier_stokes', 'fsi', 'contact')
        """
        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        # Find matching physics
        physics_match = None
        for p in backend.supported_physics():
            if p.name == module.lower():
                physics_match = p
                break

        if not physics_match:
            avail = [p.name for p in backend.supported_physics()]
            return f"Module '{module}' not found in {solver}. Available: {', '.join(avail)}"

        # Pitfalls must come from the BACKEND (Table-1 promoted
        # source of truth). Supplementary fields (description,
        # weak_form, problem_type, materials, etc.) can come from
        # deep_knowledge.py. This stops the previous bug where
        # `module_knowledge` returned the prose deep_knowledge
        # pitfalls instead of the Table-1 ones the agent gets
        # via `prepare_simulation` — same backend, two answers.
        from tools.deep_knowledge import _4C_KNOWLEDGE, _FENICS_KNOWLEDGE, _DEALII_KNOWLEDGE
        # `febio` was removed 2026-06-02 — the febio backend's own
        # generators carry the canonical 16-physics catalog; the
        # 4-key deep_knowledge.py shim was shadowing it.
        knowledge_map = {
            "fourc": _4C_KNOWLEDGE, "fenics": _FENICS_KNOWLEDGE,
            "dealii": _DEALII_KNOWLEDGE,
        }
        deep_db = knowledge_map.get(solver.lower(), {})
        deep_knowledge = deep_db.get(module.lower(), {})

        # Backend-served (Table-1) view — overlays pitfalls.
        backend_knowledge = {}
        try:
            backend_knowledge = backend.get_knowledge(module.lower())
            if not isinstance(backend_knowledge, dict):
                backend_knowledge = {}
        except Exception:
            backend_knowledge = {}

        # Merge: deep_knowledge for supplementary fields, backend
        # for pitfalls (the Table-1 source of truth) and any
        # fields the backend explicitly provides.
        knowledge = dict(deep_knowledge)
        for k, v in backend_knowledge.items():
            if k == "pitfalls" or k not in knowledge:
                knowledge[k] = v

        lines = [f"# {physics_match.description}"]
        lines.append(f"**Solver:** {backend.display_name()}")
        lines.append(f"**Module key:** `{module}`")
        lines.append(f"**Input format:** {backend.input_format().value}")
        lines.append(f"**Dimensions:** {physics_match.spatial_dims}")
        lines.append(f"**Element types:** {', '.join(physics_match.element_types)}")
        lines.append(f"**Template variants:** {', '.join(physics_match.template_variants)}")

        if knowledge:
            if "description" in knowledge:
                lines.append(f"\n**Description:** {knowledge['description']}")
            if "problem_type" in knowledge:
                lines.append(f"**Problem type:** `{knowledge['problem_type']}`")
            if "required_sections" in knowledge:
                lines.append(f"**Required sections:** {', '.join(knowledge['required_sections'])}")
            if "weak_form" in knowledge:
                lines.append(f"**Weak form:** {knowledge['weak_form']}")
            if "function_space" in knowledge:
                lines.append(f"**Function space:** {knowledge['function_space']}")
            if "solver" in knowledge:
                lines.append(f"\n**Solver recommendations:**")
                if isinstance(knowledge["solver"], dict):
                    for k, v in knowledge["solver"].items():
                        lines.append(f"  - {k}: {v}")
                else:
                    lines.append(f"  {knowledge['solver']}")
            if "materials" in knowledge:
                lines.append(f"\n**Materials:**")
                for mat_name, mat_info in knowledge["materials"].items():
                    lines.append(f"  - `{mat_name}`: {json.dumps(mat_info, default=str)}")
            if "pitfalls" in knowledge:
                lines.append(f"\n**Pitfalls:**")
                for p in knowledge["pitfalls"]:
                    lines.append(f"  - {p}")
            if "variants" in knowledge:
                lines.append(f"\n**Available templates:** {', '.join(knowledge['variants'])}")

        lines.append(f"\nUse `generate_input('{solver}', '{module}', '<variant>')` to get a working template.")
        lines.append(f"Use `knowledge('physics', solver='{solver}', physics='{module}')` for the full knowledge dump.")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════
    # list_template_variants — what templates exist per module per backend
    # ═══════════════════════════════════════════════════════════════════════

    @mcp.tool()
    def list_template_variants(solver: str = "", module: str = "") -> str:
        """List all available template variants for a solver and/or physics module.

        Shows what ready-to-run input files can be generated.

        Args:
            solver: Filter by backend (empty = all)
            module: Filter by physics (empty = all)
        """
        if solver:
            backends = [get_backend(solver)]
            backends = [b for b in backends if b]
        else:
            backends = available_backends()

        if not backends:
            return "No backends available."

        lines = []
        for b in backends:
            b_lines = []
            for p in b.supported_physics():
                if module and p.name != module.lower():
                    continue
                if p.template_variants:
                    b_lines.append(f"  - **{p.name}**: {', '.join(p.template_variants)} — {p.description}")

            if b_lines:
                lines.append(f"## {b.display_name()}")
                lines.extend(b_lines)
                lines.append("")

        if not lines:
            return f"No templates found for solver='{solver}', module='{module}'."
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════
    # parameter_study — sweep a parameter across runs (ANY backend)
    # ═══════════════════════════════════════════════════════════════════════

    @mcp.tool()
    async def parameter_study(solver: str, physics: str, variant: str,
                               parameter_name: str, values: str,
                               base_params: str = "{}",
                               critic_approved: bool = False) -> str:
        """Run a parameter study: same simulation with different parameter values.

        IMPORTANT — Before calling this tool, you MUST have an independent
        critic review the simulation setup. The critic should check every
        parameter against literature, verify units, discretization, boundary
        conditions, and loading. Set critic_approved=True only after a critic
        agent has explicitly approved the setup. Skipping this step is the
        most common source of wrong simulation results.

        Generates input, runs, and compares results for each parameter value.
        Works with ANY backend.

        Args:
            solver: Backend name ('fenics', 'fourc', 'dealii', 'febio')
            physics: Physics type (e.g. 'poisson', 'linear_elasticity')
            variant: Template variant (e.g. '2d', '3d')
            parameter_name: Parameter to vary (e.g. 'kappa', 'E', 'Re', 'nx')
            values: JSON array of values (e.g. '[0.1, 1.0, 10.0]')
            base_params: JSON of other fixed parameters
            critic_approved: Set to True ONLY after a critic agent has
                reviewed and approved the simulation setup. If False,
                a warning is included in the output.

        Returns:
            Comparison table of results across parameter values.
        """
        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        status, msg = backend.check_availability()
        if status != BackendStatus.AVAILABLE:
            return f"{solver} not available: {msg}"

        try:
            val_list = json.loads(values)
        except json.JSONDecodeError:
            return f"Invalid values JSON: {values}. Use format: [0.1, 1.0, 10.0]"

        try:
            base = json.loads(base_params)
        except json.JSONDecodeError:
            return f"Invalid base_params JSON: {base_params}"

        from core.output_paths import output_dir as _output_dir
        output_dir = _output_dir("simulation_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        study_dir = output_dir / f"param_study_{solver}_{parameter_name}_{ts}"
        study_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, val in enumerate(val_list):
            params = dict(base)
            params[parameter_name] = val

            try:
                content = backend.generate_input(physics, variant, params)
            except ValueError as e:
                results.append({"value": val, "status": "gen_failed", "error": str(e)})
                continue

            work_dir = study_dir / f"run_{i}_{parameter_name}_{val}"
            work_dir.mkdir(parents=True, exist_ok=True)

            job = await backend.run(content, work_dir, np=1, timeout=None)

            entry = {"value": val, "status": job.status, "elapsed_s": job.elapsed}
            if job.status == "completed":
                vtu_files = [f for f in backend.get_result_files(job) if f.suffix == ".vtu"]
                if vtu_files:
                    from core.post_processing import post_process_file
                    pp = post_process_file(vtu_files[0], plot_fields=True, plot_dir=work_dir)
                    if pp.fields:
                        entry["field"] = pp.fields[0].name
                        entry["max"] = pp.fields[0].max
                        entry["min"] = pp.fields[0].min
            elif job.error:
                entry["error"] = error_excerpt(job.error, 400)

            results.append(entry)

        # Summary
        lines = [f"# Parameter Study: {parameter_name}\n"]
        lines.append(f"**Solver:** {backend.display_name()}")
        lines.append(f"**Physics:** {physics}/{variant}")
        lines.append(f"**Values:** {val_list}\n")
        lines.append(f"| {parameter_name} | Status | max(field) | Time |")
        lines.append("|---|---|---|---|")
        for r in results:
            mx = f"{r['max']:.6e}" if "max" in r else "-"
            t = f"{r.get('elapsed_s', 0):.2f}s" if r.get("elapsed_s") else "-"
            lines.append(f"| {r['value']} | {r['status']} | {mx} | {t} |")

        lines.append(f"\nResults saved to: {study_dir}")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════
    # query_documentation — search solver docs
    # ═══════════════════════════════════════════════════════════════════════

    @mcp.tool()
    def query_documentation(solver: str, topic: str) -> str:
        """Search solver documentation for a specific topic.

        For 4C: searches Sphinx docs and tutorial READMEs.
        For others: returns relevant knowledge from encoded domain expertise.

        Args:
            solver: Backend name ('fenics', 'fourc', 'dealii', 'febio')
            topic: Search term (e.g. 'boundary conditions', 'material models',
                   'time integration', 'mesh generation', 'parallel')
        """
        if solver.lower() in ("fourc", "4c"):
            return _query_4c_docs(topic)
        else:
            # For other solvers, search our knowledge base.
            # Pitfalls are sourced from the backend (Table-1
            # promoted) — deep_knowledge supplies supplementary
            # fields (weak_form, problem_type, materials, ...).
            from tools.deep_knowledge import _FENICS_KNOWLEDGE, _DEALII_KNOWLEDGE
            # `febio` removed 2026-06-02 — the febio backend's own
            # generators carry the canonical 16-physics catalog.
            knowledge_map = {"fenics": _FENICS_KNOWLEDGE, "dealii": _DEALII_KNOWLEDGE}
            db = knowledge_map.get(solver.lower())
            if not db:
                return f"Unknown solver: {solver}"
            backend_obj = get_backend(solver.lower())

            topic_lower = topic.lower()
            results = []
            for key, k in db.items():
                # Overlay backend pitfalls (Table-1 source) if
                # the backend exposes them for this physics.
                if backend_obj is not None:
                    try:
                        bk = backend_obj.get_knowledge(key)
                        if isinstance(bk, dict) and "pitfalls" in bk:
                            k = {**k, "pitfalls": bk["pitfalls"]}
                    except Exception:
                        pass
                k_str = json.dumps(k, default=str).lower()
                if topic_lower in k_str:
                    results.append(f"## {key}\n{json.dumps(k, indent=2, default=str)}")

            if not results:
                return f"No documentation found for '{topic}' in {solver}. Try: {', '.join(db.keys())}"
            return f"# Documentation: '{topic}' in {solver}\n\n" + "\n\n---\n\n".join(results[:5])

    # ═══════════════════════════════════════════════════════════════════════
    # restart_simulation — continue from checkpoint
    # ═══════════════════════════════════════════════════════════════════════

    @mcp.tool()
    async def restart_simulation(job_id: str, additional_steps: int = 10,
                                  timeout=None,
                                  critic_approved: bool = False) -> str:
        """Restart a completed simulation from its last checkpoint.

        IMPORTANT — Before calling this tool, you MUST have an independent
        critic review the simulation setup. The critic should check every
        parameter against literature, verify units, discretization, boundary
        conditions, and loading. Set critic_approved=True only after a critic
        agent has explicitly approved the setup. Skipping this step is the
        most common source of wrong simulation results.

        Currently supported for 4C backend: sets PROBLEM TYPE RESTART to the
        last step of the finished run; 4C reads that restart from the run's own
        output and writes the continuation under the prefix output-1. The
        finished run must have written a restart at that step (RESTARTEVERY in
        its dynamics section decides that; Scalar_Transport's default is every
        step).
        For FEniCS/deal.II: re-run with modified parameters.

        Args:
            job_id: Job ID from a previous run_simulation call.
            additional_steps: Number of additional time steps.
            timeout: Timeout for the continued simulation.
            critic_approved: Set to True ONLY after a critic agent has
                reviewed and approved the simulation setup. If False,
                a warning is included in the output.
        """
        from tools.simulation import _jobs
        job = _jobs.get(job_id)
        if not job:
            return f"Unknown job: {job_id}"
        if job.backend_name != "fourc":
            return f"Restart currently supported for 4C only. Job backend: {job.backend_name}"

        input_file = job.work_dir / "input.4C.yaml"
        if not input_file.exists():
            return f"No input file found in {job.work_dir}"

        try:
            data = yaml.safe_load(input_file.read_text())
        except yaml.YAMLError as e:
            return f"Could not parse input file: {e}"

        # Find dynamics section and update
        for sec_name in ["SCALAR TRANSPORT DYNAMIC", "STRUCTURAL DYNAMIC", "FLUID DYNAMIC"]:
            if sec_name in data:
                sec = data[sec_name]
                old_numstep = sec.get("NUMSTEP", 0)
                timestep = sec.get("TIMESTEP", 1.0)
                sec["NUMSTEP"] = old_numstep + additional_steps
                sec["MAXTIME"] = sec.get("MAXTIME", 0) + additional_steps * timestep
                # 4C has no RESTARTEVRY or RESTARTFROMSTEP key (a deck with them
                # stops with "Could not match this input"); PROBLEM TYPE RESTART
                # is the step to restart from when no --restart flag is given.
                data.setdefault("PROBLEM TYPE", {})["RESTART"] = old_numstep
                break
        else:
            return "Could not find dynamics section to modify for restart."

        new_yaml = yaml.dump(data, default_flow_style=False, sort_keys=False)
        restart_input = job.work_dir / "input_restart.4C.yaml"
        restart_input.write_text(new_yaml)

        backend = get_backend("fourc")
        new_job = await backend.run(new_yaml, job.work_dir, np=1, timeout=timeout)

        if new_job.status == "completed":
            return f"Restart completed in {new_job.elapsed:.1f}s. Extended to {old_numstep + additional_steps} steps."
        else:
            return f"Restart failed: {new_job.error}"

    # ═══════════════════════════════════════════════════════════════════════
    # validate_physics — physics-aware parameter validation
    # ═══════════════════════════════════════════════════════════════════════

    @mcp.tool()
    def validate_physics(solver: str, physics: str, params: str = "{}") -> str:
        """Validate simulation parameters against physics constraints.

        Checks for physically impossible or suspect values:
        - Negative Young's modulus
        - Poisson's ratio >= 0.5
        - Negative viscosity
        - Reynolds number too high for chosen mesh
        - Missing required parameters

        Args:
            solver: Backend name
            physics: Physics type
            params: JSON parameters to validate
        """
        try:
            param_dict = json.loads(params)
        except json.JSONDecodeError:
            return f"Invalid JSON: {params}"

        warnings = []

        # Universal physics checks
        if "E" in param_dict and param_dict["E"] <= 0:
            warnings.append("ERROR: Young's modulus E must be > 0")
        if "nu" in param_dict:
            nu = param_dict["nu"]
            if nu < 0:
                warnings.append("ERROR: Poisson's ratio nu must be >= 0")
            elif nu >= 0.5:
                warnings.append("ERROR: Poisson's ratio nu must be < 0.5 (incompressible limit)")
            elif nu > 0.49:
                warnings.append("WARNING: nu > 0.49 — nearly incompressible, may cause locking")

        if "kappa" in param_dict and param_dict["kappa"] <= 0:
            warnings.append("ERROR: Diffusion coefficient kappa must be > 0")
        if "conductivity" in param_dict and param_dict["conductivity"] <= 0:
            warnings.append("ERROR: Thermal conductivity must be > 0")

        # NS-specific
        if physics.lower() in ("navier_stokes", "fluid"):
            if "Re" in param_dict:
                re = param_dict["Re"]
                if re <= 0:
                    warnings.append("ERROR: Reynolds number must be > 0")
                elif re > 1000:
                    warnings.append("WARNING: Re > 1000 — may need very fine mesh or turbulence model")
                elif re > 500:
                    warnings.append("WARNING: Re > 500 — Newton solver may need continuation approach")

        # PD-specific
        if physics.lower() in ("particle_pd", "peridynamics"):
            if "CRITICAL_STRETCH" in param_dict:
                cs = param_dict["CRITICAL_STRETCH"]
                if cs <= 0 or cs > 0.1:
                    warnings.append(f"WARNING: CRITICAL_STRETCH={cs} unusual — typical range 0.001-0.05")

        # Mesh density checks
        if "nx" in param_dict and param_dict["nx"] < 4:
            warnings.append("WARNING: nx < 4 — mesh very coarse, results unreliable")
        if "refinements" in param_dict and param_dict["refinements"] < 2:
            warnings.append("WARNING: refinements < 2 — mesh very coarse")

        if not warnings:
            return "Physics validation PASSED — all parameters look reasonable."

        errors = [w for w in warnings if w.startswith("ERROR")]
        warns = [w for w in warnings if w.startswith("WARNING")]
        lines = []
        if errors:
            lines.append(f"ERRORS ({len(errors)}):")
            for e in errors:
                lines.append(f"  {e}")
        if warns:
            lines.append(f"WARNINGS ({len(warns)}):")
            for w in warns:
                lines.append(f"  {w}")
        return "\n".join(lines)


def _query_4c_docs(topic: str) -> str:
    """Search 4C Sphinx docs and tutorial READMEs."""
    if not FOURC_ROOT or not FOURC_ROOT.is_dir():
        return "4C docs not available (FOURC_ROOT not set)."

    topic_lower = topic.lower()
    results = []

    # Search Sphinx docs
    docs_dir = FOURC_ROOT / "doc" / "documentation" / "src"
    if docs_dir.is_dir():
        for md_file in sorted(docs_dir.rglob("*.md")):
            try:
                content = md_file.read_text(errors="replace")
                if topic_lower in content.lower():
                    rel_path = md_file.relative_to(FOURC_ROOT)
                    lines = content.split("\n")
                    relevant = []
                    for i, line in enumerate(lines):
                        if topic_lower in line.lower():
                            start = max(0, i - 3)
                            end = min(len(lines), i + 8)
                            relevant.append("\n".join(lines[start:end]))
                            if len(relevant) >= 2:
                                break
                    if relevant:
                        results.append(f"### `{rel_path}`\n\n" + "\n\n---\n\n".join(relevant))
            except Exception:
                pass
            if len(results) >= 5:
                break

    # Search tutorial READMEs
    tutorials_dir = FOURC_ROOT / "tests" / "tutorials"
    if tutorials_dir.is_dir():
        for readme in sorted(tutorials_dir.rglob("README*")):
            try:
                content = readme.read_text(errors="replace")
                if topic_lower in content.lower():
                    rel_path = readme.relative_to(FOURC_ROOT)
                    if len(content) > 2000:
                        content = content[:2000] + "\n... (truncated)"
                    results.append(f"### `{rel_path}`\n\n{content}")
            except Exception:
                pass
            if len(results) >= 7:
                break

    if not results:
        return f"No 4C documentation found for '{topic}'. Try broader terms."

    return f"# 4C Documentation: '{topic}'\n\nFound {len(results)} sections.\n\n" + "\n\n---\n\n".join(results)
