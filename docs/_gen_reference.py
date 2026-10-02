#!/usr/bin/env python3
"""Generate the Tools and Solvers reference pages of the documentation website from the code.

The pages are committed, so building the website needs no solver installed. Re-run this after a
tool or a backend changes:

    python docs/_gen_reference.py        # from the repository root, openPASO's own .venv active

Everything factual on these pages -- tool names, parameters, full descriptions, the physics each
solver supports, dimensions, elements, templates -- is read from the running code. Only the one
plain-language sentence per tool and the short solver introductions are written by hand, below.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
sys.path.insert(0, str(REPO / "src"))
logging.disable(logging.CRITICAL)

# One sentence per tool, for a reader who has never seen an MCP tool. Grouped for the index page.
TOOL_GROUPS = {
    "Find out": {
        "discover": "Lists the solvers, which ones are installed on this machine, and what each can do.",
        "knowledge": "Answers questions about a solver: physics, settings, known traps, materials, coupling, and what an error message means.",
        "examples": "Finds real input files from the solvers' own test suites to start from.",
        "prepare_simulation": "The usual first step: collects knowledge, examples and a starting input file in one call.",
        "session_insights": "Summarises what happened so far in this session.",
    },
    "Run": {
        "run_simulation": "Runs a Python-based solver script (FEniCSx, NGSolve, scikit-fem, DUNE-fem).",
        "run_with_generator": "Writes an input file from a template and runs a compiled solver (4C, deal.II, Kratos, FEBio, SPARTA).",
        "check_input": "Checks an input file for known mistakes before anything is run.",
        "generate_mesh": "Builds a mesh with Gmsh: a rectangle, an L-shape, a plate with a hole, a channel, or your own shape.",
        "visualize": "Makes plots and statistics from a result, with automatic sanity checks.",
    },
    "Check the answer": {
        "verify_mesh_independence": "Refines the mesh step by step and reports whether the answer stopped changing.",
        "verify_pde_consistency": "Checks that the computed field actually satisfies the equation it claims to solve.",
        "verify_interface_flux": "For two coupled solvers: checks that what leaves one side arrives at the other.",
        "audit_results": "Reads the result files and names anything missing, inconsistent or suspicious.",
        "submit_critic_review": "Records the independent review of the setup that has to happen before a run.",
    },
    "Two solvers on one problem": {
        "couple": "Runs one problem split across two solvers and repeats the exchange until both sides agree.",
        "couple_levels": "Does the same on several mesh sizes, so the accuracy of the coupled answer can be measured.",
        "coupled_solve": "A ready-made coupled setup for common pairs of physics.",
        "couple_precice": "Couples two solvers through the preCICE library instead of openPASO's own driver.",
        "transfer_field": "Moves a result from one solver's output into another solver's input.",
        "write_participant_contract": "Writes the fill-in participant script for one solver of a coupled problem into your folder: everything around the solve, which you then write yourself.",
    },
    "Set up and develop": {
        "setup_backend": "Explains how to install a missing solver, with the route that works on this system.",
        "rediscover_backends": "Looks for solvers again, for example after you installed one.",
        "reload_catalog": "Reloads the knowledge catalogue after it changed.",
        "developer": "Reads and changes a solver's own source code, then rebuilds it.",
    },
}

SOLVERS = {
    "skfem": ("scikit-fem", "Pure Python, and the easiest to start with. You see and control every step of the assembly.",
              "pip install scikit-fem", "Installs in seconds into openPASO's own `.venv`."),
    "ngsolve": ("NGSolve", "High-order elements; strong for electromagnetics, acoustics and wave problems.",
                "pip install ngsolve", "Installs in seconds into openPASO's own `.venv`."),
    "kratos": ("Kratos Multiphysics", "Structures, fluids, coupled problems and particle methods.",
               "pip install KratosMultiphysics-all",
               "Use the `-all` package. The plain `KratosMultiphysics` 10.4 wheels are labelled for an older system "
               "library than they need, so on an older system they install without error and then fail to import."),
    "dune": ("DUNE-fem", "Discontinuous Galerkin methods and adaptive meshes.",
             "pip install dune-fem mpi4py",
             "`mpi4py` is a hidden requirement: without it the first import stops with "
             "\"Please run pip install mpi4py before rerunning your Dune script.\""),
    "fenics": ("FEniCSx (dolfinx)", "Fast prototyping: you write the equation almost as on paper. Strong for fluid flow.",
               "conda create -n fenics -c conda-forge fenics-dolfinx",
               "FEniCSx needs its own conda environment, not openPASO's `.venv`. openPASO finds environments whose name "
               "contains `fenics` or `dolfinx` by itself; otherwise set `FENICS_PYTHON`. Real and complex numbers are "
               "separate environments."),
    "dealii": ("deal.II", "Adaptive mesh refinement and very large parallel computations, in C++.",
               "sudo apt install libdeal.ii-dev",
               "Version 9.3 or newer is needed for all examples (Ubuntu 20.04 ships 9.1.1). If you build it yourself, use "
               "`-DCMAKE_BUILD_TYPE=DebugRelease`: a Release build removes the internal checks, so mistakes fail silently."),
    "febio": ("FEBio", "Biomechanics: soft tissue, cartilage, muscle.",
              "export FEBIO_BINARY=/path/to/febio4",
              "Download the program from <https://febio.org/downloads/>, then point openPASO at the program file itself."),
    "fourc": ("4C Multiphysics", "Fluid–structure interaction, contact, beams, particles and many coupled problems.",
              "export FOURC_BINARY=/path/to/4C",
              "Built from source, which takes hours: <https://github.com/4C-multiphysics/4C>. You do not need it to start."),
    "sparta": ("SPARTA", "Rarefied gas with the particle method DSMC, where the usual flow equations stop working. Experimental.",
               "export SPARTA_BINARY=/path/to/spa_serial",
               "Built from source: <https://sparta.github.io/>. You do not need it to start."),
}


def _first_paragraph(text: str) -> str:
    return (text or "").strip().split("\n\n", 1)[0].replace("\n", " ").strip()


def _params_table(schema: dict) -> str:
    props = (schema or {}).get("properties", {})
    if not props:
        return "This tool takes no parameters.\n"
    required = set((schema or {}).get("required", []))
    rows = ["| Parameter | Type | Required | Default |", "|---|---|---|---|"]
    for name, spec in props.items():
        typ = spec.get("type") or " or ".join(x.get("type", "?") for x in spec.get("anyOf", [])) or "any"
        default = "" if name in required else f"`{spec.get('default')!r}`"
        rows.append(f"| `{name}` | {typ} | {'yes' if name in required else 'no'} | {default} |")
    return "\n".join(rows) + "\n"


def tools_pages() -> int:
    import server  # noqa: PLC0415
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    grouped = {n for g in TOOL_GROUPS.values() for n in g}
    missing = sorted(set(tools) - grouped)
    stale = sorted(grouped - set(tools))
    if missing or stale:
        raise SystemExit(f"TOOL_GROUPS is out of date: not described {missing}, no longer registered {stale}")
    out = DOCS / "tools"
    out.mkdir(exist_ok=True)
    index = [
        "# The tools the AI model gets",
        "",
        f"openPASO gives the AI model **{len(tools)} tools**. You never call them yourself: you describe what "
        "you want, and the model decides which tool to use. This page tells you what each one is for, so you "
        "can follow what the model is doing.",
        "",
        "Each tool page also shows the full description the model reads. It is long and technical on purpose: "
        "it is written for the model, not for you.",
        "",
    ]
    for group, members in TOOL_GROUPS.items():
        index += [f"## {group}", "", "| Tool | What it is for |", "|---|---|"]
        for name, plain in members.items():
            index.append(f"| [`{name}`]({name}.md) | {plain} |")
            t = tools[name]
            page = [
                f"# `{name}`", "", f"**{plain}**", "", f"Group: {group}.", "",
                "## Parameters", "", _params_table(t.inputSchema),
                "## What the model reads", "",
                "The text below is the tool's own description, exactly as the AI model receives it.", "",
                '??? note "Show the full description"', "",
                "    ```text",
                *("    " + line for line in (t.description or "").strip().splitlines()),
                "    ```", "",
            ]
            (out / f"{name}.md").write_text("\n".join(page))
        index.append("")
    (out / "index.md").write_text("\n".join(index))
    return len(tools)


def solver_pages() -> int:
    from core.registry import all_backends, load_all_backends  # noqa: PLC0415
    load_all_backends()
    backends = {b.name(): b for b in all_backends()}
    if set(backends) != set(SOLVERS):
        raise SystemExit(f"SOLVERS is out of date: code has {sorted(backends)}, page text has {sorted(SOLVERS)}")
    out = DOCS / "solvers"
    out.mkdir(exist_ok=True)
    index = [
        "# Solvers", "",
        "A **solver** is the program that actually computes the physics. openPASO can drive nine of them. "
        "**You need only one to start**, and scikit-fem is the easiest.", "",
        "| Solver | Good at | Physics openPASO knows | Install |", "|---|---|---|---|",
    ]
    for key, (title, good, cmd, note) in SOLVERS.items():
        phys = backends[key].supported_physics()
        index.append(f"| [{title}]({key}.md) | {good} | {len(phys)} | `{cmd}` |")
        page = [
            f"# {title}", "", good, "",
            "## Install", "", "```bash", cmd, "```", "", note, "",
            "Then check that openPASO sees it:", "", "```bash", "python check_install.py", "```", "",
            f"## What openPASO knows for {title}", "",
            f"{len(phys)} kinds of problem. Ask for any of them in plain words; the names below are what the "
            "model uses internally.", "",
            "| Physics | Description | Dimensions | Templates |", "|---|---|---|---|",
        ]
        for p in sorted(phys, key=lambda p: p.name):
            dims = ", ".join(f"{d}-D" for d in p.spatial_dims)
            tmpl = ", ".join(f"`{v}`" for v in p.template_variants) or "—"
            page.append(f"| `{p.name}` | {p.description} | {dims} | {tmpl} |")
        (out / f"{key}.md").write_text("\n".join(page) + "\n")
    index += ["", "Solvers you have not installed are simply reported as missing; openPASO uses the others."]
    (out / "index.md").write_text("\n".join(index) + "\n")
    return len(backends)


def sync_nav() -> None:
    """Rewrite the Tools section of mkdocs.yml from TOOL_GROUPS, so the menu never drifts."""
    cfg = REPO / "mkdocs.yml"
    text = cfg.read_text()
    start = text.index("  - Tools:\n")
    end = text.index("\n  - How it works:", start)
    lines = ["  - Tools:", "      - tools/index.md"]
    for group, members in TOOL_GROUPS.items():
        lines.append(f"      - {group}:")
        lines += [f"          - {name}: tools/{name}.md" for name in members]
    cfg.write_text(text[:start] + "\n".join(lines) + text[end:])


if __name__ == "__main__":
    print(f"{tools_pages()} tool pages, {solver_pages()} solver pages written under docs/")
    sync_nav()
