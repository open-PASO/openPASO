"""
Kratos Multiphysics solver backend.

Kratos is a framework for building multi-disciplinary simulation software.
It uses a three-file system:
  - MainKratos.py: Python driver script
  - ProjectParameters.json: solver settings, BCs, materials
  - mesh.mdpa: mesh data (nodes, elements, conditions)

This backend generates all three files and executes MainKratos.py.
VTK output is configured via ProjectParameters.json.
"""

import asyncio
import json
import logging
import os
import re
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

# `import x`, `from x import y`, and their continuation-free one-liners. Used
# by the honesty guard in validate_input to drop lines that merely NAME a
# symbol before looking for evidence that one was CALLED.
_IMPORT_LINE = re.compile(r"\s*(?:from\s+[\w.]+\s+)?import\s")

logger = logging.getLogger("openpaso.kratos")


# The applications openPASO's own catalogue names, which is what makes an
# interpreter useful rather than merely importable.
_CATALOGUE_APPS = (
    "FluidDynamics", "FSI", "GeoMechanics", "RANS", "CompressiblePotentialFlow",
    "Rom", "Iga", "StructuralMechanics", "Mapping", "MeshMoving", "Optimization",
    "CableNet", "DemStructuresCoupling", "TopologyOptimization",
    "PfemFluidDynamics", "ThermalDEM", "SwimmingDEM", "FemToDem", "Chimera",
    "DropletDynamics", "FluidDynamicsBiomedical", "ConstitutiveLaws",
    "ShallowWaterApplication".replace("Application", ""), "DEM", "MPM",
    "Contact"
)


_KRATOS_PYTHON_CACHE: dict = {}


def _find_kratos_python():
    """Locate and VERIFY the Python that can import KratosMultiphysics.

    THIS USED TO BE `get_python_executable()`, the server's own interpreter and
    nothing else. Kratos is a heavy compiled package that is normally installed
    somewhere of its own, so openPASO reported NOT_INSTALLED on a machine that
    had Kratos 10.3.0 sitting in a sibling virtual environment. Same shape as
    the NGSolve defect, and fixed the same way.

    Order: KRATOS_PYTHON, then what autodiscovery recorded (an ABSOLUTE path
    that survives the sandbox's tmpfs over $HOME, where a conda scan cannot
    look), then the active interpreter, then conda envs. Every candidate is
    verified by importing, because a recorded path can go stale.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    # The cache is keyed by the explicit pin. A KRATOS_PYTHON set later in the
    # process (the coverage harness pins through it) must win over an
    # interpreter resolved earlier under a different environment. Measured
    # 2026-09-24: the full test suite resolved and cached the server venv --
    # Kratos core, no ConvectionDiffusionApplication -- before the
    # planted-failure tests set the pin, and every Kratos run of theirs died on
    # the missing application while the same tests passed alone.
    cache_key = os.environ.get("KRATOS_PYTHON", "") or "python"
    if cache_key in _KRATOS_PYTHON_CACHE:
        return _KRATOS_PYTHON_CACHE[cache_key]

    candidates = []
    env_python = os.environ.get("KRATOS_PYTHON", "")
    if env_python and Path(env_python).is_file():
        # A NAMED INTERPRETER IS USED, NOT AUDITIONED. Until 2026-09-24 the pin
        # was only the first candidate to be probed, and when its probe failed
        # for any reason -- a timeout under a loaded box, a crashed import --
        # the finder fell through to conda environments and silently ran Kratos
        # somewhere else. Measured in two full test suites: the pinned venv
        # (Kratos 10.3, ConvectionDiffusionApplication present) was passed over
        # for a 3.12 build without that application, and every pinned run died
        # on the missing import while the same runs passed alone. Someone who
        # names an interpreter means it; if it cannot do the work, the run fails
        # loudly under that interpreter instead of succeeding under another.
        _KRATOS_PYTHON_CACHE[cache_key] = env_python
        return env_python
    try:
        from core.autodiscovery import load_discovered_config
        entry = ((load_discovered_config() or {}).get("backends") or {}).get("kratos")
        recorded = (entry or {}).get("location")
        if recorded and Path(recorded).is_file():
            candidates.append((-2, str(recorded)))
    except Exception:                                        # noqa: BLE001
        pass
    candidates.append((0, sys.executable))
    for base in (Path.home() / "miniconda3" / "envs",
                 Path.home() / "anaconda3" / "envs",
                 Path.home() / "miniforge3" / "envs"):
        if base.is_dir():
            for env_dir in sorted(base.iterdir()):
                py = env_dir / "bin" / "python"
                if py.is_file():
                    candidates.append((1 if "kratos" in env_dir.name.lower() else 2,
                                       str(py)))

    seen, ordered = set(), []
    for _prio, py in sorted(candidates, key=lambda x: x[0]):
        if py not in seen:
            seen.add(py)
            ordered.append(py)
    # PICK THE BUILD THAT CAN DO THE WORK, not the first that imports.
    #
    # Kratos is 48 applications and a build carries whichever were compiled.
    # "import KratosMultiphysics succeeds" says almost nothing about whether the
    # physics openPASO offers will run. Measured on this machine: the first
    # interpreter that imports Kratos carries 6 of the 21 applications the
    # catalogue names, while a second build on the same host carries 14 --
    # including OptimizationApplication, CableNetApplication and
    # DemStructuresCouplingApplication, which are three physics openPASO ALREADY
    # REGISTERS and could not actually run, and five more it does not yet offer.
    #
    # So candidates are scored by how many of those applications import, and the
    # best wins. (An explicit KRATOS_PYTHON never reaches this loop: it returned
    # above, unprobed, because someone who names an interpreter means it.)
    probe = (
        "import importlib.util as u\n"
        "apps = %r\n"
        "print(sum(1 for a in apps "
        "if u.find_spec('KratosMultiphysics.' + a + 'Application')))\n"
    ) % (_CATALOGUE_APPS,)

    best, best_score = None, -1
    for priority, python in [(pr, py) for pr, py in
                             sorted(candidates, key=lambda x: x[0])
                             if py in set(ordered)]:
        try:
            done = subprocess.run([python, "-c", probe], stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=180)
        except Exception:                                    # noqa: BLE001
            continue
        if done.returncode != 0:
            continue
        try:
            score = int(done.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            score = 0
        if priority <= -2:            # named outright: take it and stop
            _KRATOS_PYTHON_CACHE[cache_key] = python
            return python
        if score > best_score:
            best, best_score = python, score

    _KRATOS_PYTHON_CACHE[cache_key] = best
    return best


class KratosBackend(SolverBackend):

    def name(self) -> str:
        return "kratos"

    def display_name(self) -> str:
        return "Kratos Multiphysics"

    def check_availability(self) -> tuple[BackendStatus, str]:
        python = _find_kratos_python()
        if not python:
            return BackendStatus.NOT_INSTALLED, "No Python with KratosMultiphysics found"
        import subprocess
        try:
            result = subprocess.run(
                [python, "-c",
                 "import KratosMultiphysics as KM; "
                 "print(KM.KratosGlobals.Kernel.Version())"], stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                ver = result.stdout.strip().split('\n')[0]
                # SAY WHERE IT LIVES, NOT ONLY THAT IT LIVES. `discover` is the
                # one surface that resolves host paths -- the coupling knowledge
                # deliberately ships none and sends the agent here for the argv
                # `couple` needs -- and this backend was the only available one
                # whose line carried a version and no interpreter. Measured on a
                # C10 cell: the agent ran the Kratos participant with the wrong
                # python, got `ModuleNotFoundError: No module named
                # 'KratosMultiphysics'`, and had nowhere to read the right one.
                return BackendStatus.AVAILABLE, f"Kratos {ver} at {python}"
            return BackendStatus.NOT_INSTALLED, f"Kratos import failed: {result.stderr.strip()[:200]}"
        except Exception as e:
            return BackendStatus.NOT_INSTALLED, f"Check failed: {e}"

    def input_format(self) -> InputFormat:
        return InputFormat.JSON

    def supported_physics(self) -> list[PhysicsCapability]:
        # ---- 2026-06-26 honesty audit -----------------------------------
        # Every physics listed here now maps to a generator that runs a REAL
        # solve (model + mesh + solve + output). All former
        # "availability-probe" stub generators — scripts whose only action
        # was to import-check a Kratos sub-application and write
        # {"note": ...} with no solver run — and their PhysicsCapability rows
        # were REMOVED: rom, topology_optimization, iga, wind_engineering,
        # thermal_dem, swimming_dem, fem_to_dem, chimera, droplet_dynamics,
        # free_surface, fluid_biomedical, fluid_hydraulics, fluid, fsi,
        # geomechanics, compressible_potential, rans, pfem_fluid, pfem_solid,
        # pfem2.
        #
        # Importability tiers (verified on this host, python3 / Kratos 10.4):
        #   * The minimal pip stack importable here ships only
        #     StructuralMechanicsApplication, ConvectionDiffusionApplication,
        #     ContactStructuralMechanicsApplication and LinearSolversApplication.
        #     The generators that run end-to-end on that stack (verified
        #     rc=0 with physical output) are: poisson, heat, heat_transient,
        #     linear_elasticity (incl. the 2d_nonlinear Total-Lagrangian
        #     Newton solve), contact (penalty Signorini), structural_dynamics,
        #     plasticity / constitutive_laws (need ConstitutiveLawsApplication).
        #   * The remaining entries below (poromechanics, shallow_water, dam,
        #     dem, mpm, shape_optimization, cosimulation, dem_structures_coupling,
        #     cable_net, optimization) are GENUINE parameterized solves — NOT
        #     probe stubs — that require a fuller Kratos build whose
        #     application is not in the minimal pip stack. They are retained
        #     because they actually solve; they are not silent no-ops.
        return [
            PhysicsCapability(
                name="poisson",
                description="Poisson / convection-diffusion (LaplacianElement, EulerianConvDiff)",
                spatial_dims=[2, 3],
                element_types=["LaplacianElement2D3N", "EulerianConvDiff2D3N"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="linear_elasticity",
                description="Structural mechanics: linear/nonlinear, static/dynamic (StructuralMechanicsApplication)",
                spatial_dims=[2, 3],
                element_types=["SmallDisplacementElement2D3N/3D4N", "TotalLagrangianElement2D3N/3D4N",
                               "ShellThinElement3D3N", "CrBeamElement3D2N", "TrussElement3D2N"],
                template_variants=["2d", "2d_nonlinear"],
            ),
            PhysicsCapability(
                name="heat",
                description="Thermal convection-diffusion: steady and transient (ConvectionDiffusionApplication)",
                spatial_dims=[2, 3],
                element_types=["EulerianConvDiff2D3N", "LaplacianElement2D3N"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="contact",
                description="Contact mechanics: frictionless Signorini (penalty active-set) — real solve via ContactStructuralMechanicsApplication",
                spatial_dims=[2],
                element_types=["Element2D3N (CST) + penalty contact"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="structural_dynamics",
                description="Dynamic structural analysis with Newmark/Bossak time integration",
                spatial_dims=[2],
                element_types=["SmallDisplacementElement2D3N"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="curved_mms",
                description=("Curved-geometry manufactured solution: steady diffusion on a "
                             "Gmsh-meshed annulus, real ConvectionDiffusionApplication solve "
                             "(LaplacianElement2D3N), prints machine-readable L2_ERROR; "
                             "theoretical P1 L2 order 2 (live-verified). Meshing is "
                             "agent-driven via mesh_size or an agent-built .msh file."),
                spatial_dims=[2],
                element_types=["LaplacianElement2D3N"],
                template_variants=["annulus_2d"],
            ),
            PhysicsCapability(
                name="heat_transient",
                description="Transient heat conduction with backward Euler time integration",
                spatial_dims=[2],
                element_types=["EulerianConvDiff2D3N"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="dem",
                description="Discrete Element Method for granular/particle simulations (DEMApplication)",
                spatial_dims=[2, 3],
                element_types=["SphericParticle3D", "CylinderParticle2D"],
                template_variants=["2d"],
            ),
            # element_types corrected 2026-08-07. The row advertised
            # "UpdatedLagrangianPQ2D" and "UpdatedLagrangianAxisym"; neither is
            # a registered element. CreateNewElement on either raises 'The
            # Element "..." is not registered!' against MPMApplication 10.4.3,
            # while the MPM-prefixed spellings below construct. This is the
            # claim kratos.mpm::0 already made — the knowledge was corrected in
            # the DEM/MPM pass and the catalog row was left behind.
            #
            # The template this row points at is NOT MPMApplication: it is a
            # standalone numpy/scipy MPM that never imports KratosMultiphysics.
            # See the module docstring of generators/mpm.py.
            PhysicsCapability(
                name="mpm",
                description="Material Point Method for large-deformation solid mechanics (MPMApplication)",
                spatial_dims=[2, 3],
                element_types=["MPMUpdatedLagrangian2D3N",
                               "MPMUpdatedLagrangianAxisymmetry2D3N"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="shape_optimization",
                description="Shape optimization with gradient-based methods (ShapeOptimizationApplication)",
                spatial_dims=[2, 3],
                element_types=["SmallDisplacementElement2D3N"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="cosimulation",
                description="CoSimulation framework for multi-solver coupling (CoSimulationApplication)",
                spatial_dims=[2, 3],
                element_types=["Generic (wraps sub-solvers)"],
                template_variants=["2d"],
            ),
            # New applications
            # Added 2026-09-19. FluidDynamicsApplication is installed and was
            # unreached. The June audit deleted a 'fluid' row that was an
            # availability-probe stub; this one is a real monolithic VMS
            # Navier-Stokes solve that checks its own result against mass
            # conservation and the developed-channel peak/mean ratio -- both
            # derived from the geometry asked for, not retrieved.
            # Added 2026-09-19. FSIApplication is installed and was
            # unreached. Its substance is the convergence accelerators for a
            # partitioned fixed-point iteration, and this uses one for
            # exactly that across two real ConvectionDiffusion solves. It
            # checks the converged interface against the value derived from
            # the conductivities, lengths and boundary temperatures, and
            # checks that the accelerator moves only the iteration count and
            # not the answer.
            PhysicsCapability(
                name="fsi_partitioned",
                description=("Partitioned Dirichlet-Neumann coupling driven "
                             "by a KratosFSIApplication convergence "
                             "accelerator (FSIApplication); a contract: "
                             "each side's setup and solve are yours"),
                spatial_dims=[2],
                element_types=["LaplacianElement2D3N", "ThermalFace2D2N"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="fluid_dynamics",
                description=("Incompressible Navier-Stokes via "
                             "FluidDynamicsApplication, monolithic VMS "
                             "element (FluidDynamicsApplication)"),
                spatial_dims=[2],
                element_types=["VMS2D3N", "QSVMS2D3N"],
                template_variants=["channel_2d"],
            ),
            PhysicsCapability("poromechanics", "Poromechanics: fracture in porous media, dam/tunnel (PoromechanicsApplication)", [2, 3],
                              ["SmallStrainUPwDiffOrderElement2D6N"], ["2d"]),
            PhysicsCapability("shallow_water", "Shallow water equations: floods, dam breaks, coastal (ShallowWaterApplication)", [2],
                              ["ShallowWaterElement2D3N"], ["2d"]),
            PhysicsCapability("dam", "Dam engineering: thermal-mechanical, seepage, cracking", [2, 3],
                              ["SmallStrainElement2D3N"], ["2d"]),
            PhysicsCapability("plasticity", "Elasto-plasticity: MC, DP, VonMises, Tresca + 6 hardening laws (ConstitutiveLawsApplication)", [2, 3],
                              ["SmallDisplacementElement3D8N", "SmallDisplacementElement2D4N",
                               "TotalLagrangianElement3D8N"], ["3d"]),
            PhysicsCapability("constitutive_laws", "Extended constitutive laws: hyperelastic, plasticity, damage, viscoplastic", [2, 3],
                              ["SmallDisplacementElement2D3N"], ["2d"]),
            PhysicsCapability("dem_structures_coupling", "DEM-FEM coupling: impact on structures, blast", [2, 3],
                              ["SphericParticle3D + SmallDisplacement"], ["2d"]),
            PhysicsCapability("cable_net", "Cable and net structures: cables, membranes, form-finding", [3],
                              ["CableElement3D2N", "MembraneElement3D3N"], ["2d"]),
            PhysicsCapability("optimization", "General optimization: gradient-based, adjoint, multi-objective", [2, 3],
                              ["Generic"], ["2d"]),
            # ── 2026-06-01 (task #70): _auxiliary_overview in
            #    src/backends/kratos/generators/auxiliary_applications.py
            #    holds 6 substantive integration / API / numerical
            #    pitfalls about Kratos infrastructure apps
            #    (TrilinosApplication PyPI gap, MappingApplication
            #    FSI requirement, MapperFactory deprecation, DEM
            #    3D-only, contact apps PyPI absence). The leading
            #    underscore in the key historically kept this out
            #    of supported_physics; expose it now as
            #    'auxiliary_overview' so users browsing
            #    discover(physics, kratos) see the meta-catalog.
            PhysicsCapability(
                "auxiliary_overview",
                "[Reference] Kratos auxiliary applications "
                "(TrilinosApplication, MetisApplication, "
                "MappingApplication, MeshMovingApplication, "
                "HDF5Application, ...) — infrastructure that "
                "other Kratos analyses depend on. Catalog "
                "contains PyPI-publication status, deprecation "
                "notes, and FSI hidden-dependency warnings. "
                "Not a PDE physics — this is a meta-reference "
                "entry; the underlying KNOWLEDGE key is "
                "'_auxiliary_overview' (with the leading "
                "underscore preserved for backward "
                "compatibility).",
                [2, 3], ["N/A — meta-reference"], ["N/A"]),
        ]

    def get_knowledge(self, physics: str) -> dict:
        # Alias: PhysicsCapability surfaces 'auxiliary_overview'
        # (no leading underscore) but the on-disk KNOWLEDGE key
        # is '_auxiliary_overview'. Map back here so the public
        # name resolves.
        if physics == "auxiliary_overview":
            physics = "_auxiliary_overview"
        # Source-of-truth (audit 2026-06-02):
        # The kratos catalog has exactly ONE per-physics
        # source: backends.kratos.generators.KNOWLEDGE. A
        # previous version of this method tried
        # `from kratos_knowledge import KRATOS_KNOWLEDGE` —
        # but data/kratos_knowledge.py exports per-application
        # constants (KRATOS_APPLICATIONS, STRUCTURAL_MECHANICS,
        # FLUID_DYNAMICS, FSI, ...) and NOT a unified
        # `KRATOS_KNOWLEDGE` dict, so the import always raised
        # ImportError and the lookup was silent dead code.
        # Removed in 2026-06-02 audit; if cross-application
        # catalog access is needed in future, plumb it through
        # explicitly (e.g. a knowledge('kratos_application',
        # 'StructuralMechanicsApplication') surface) rather
        # than re-introducing a phantom flat-dict import.
        return KNOWLEDGE.get(physics, {})

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        # Meta-reference physics: catalog declares
        # 'auxiliary_overview' so it appears in discover() and
        # knowledge() surfaces, but it has no PDE template —
        # it's a documentation entry for Kratos infrastructure
        # apps (TrilinosApplication, MetisApplication, etc.).
        # Return a commentary script that points to the
        # underlying KNOWLEDGE key. Without this early-return,
        # calling generate_input('auxiliary_overview', 'N/A',
        # {}) would raise ValueError.
        if physics == "auxiliary_overview" or variant == "N/A":
            return (
                '"""Kratos auxiliary_overview — meta-reference"""\n'
                '# This is NOT a runnable Kratos analysis script.\n'
                '# The catalog advertises auxiliary_overview so '
                'it appears in\n'
                '# discover() and knowledge() results — it '
                'documents Kratos\n'
                '# infrastructure applications (TrilinosApplication,\n'
                '# MetisApplication, MappingApplication, '
                'MeshMovingApplication,\n'
                '# HDF5Application, ...) and their hidden FSI '
                'dependencies.\n'
                '#\n'
                '# Use knowledge(kratos, auxiliary_overview) for the\n'
                '# full reference. For a runnable Kratos analysis '
                'pick a\n'
                '# concrete physics: poisson, linear_elasticity, '
                'heat, fluid,\n'
                '# fsi, contact, mpm, dem, geomechanics, '
                'cosimulation, ...\n'
                'import KratosMultiphysics  # placeholder — see '
                'comment above\n'
                'print("auxiliary_overview is a meta-reference; '
                'see knowledge() output")\n'
            )
        key = f"{physics}_{variant}"
        gen = GENERATORS.get(key)
        if not gen:
            raise ValueError(f"No Kratos template for {key}")
        # Return the MainKratos.py script which embeds the JSON and mdpa inline
        return gen(params)

    def validate_input(self, content: str) -> list[str]:
        errors = []
        try:
            compile(content, "<kratos_input>", "exec")
        except SyntaxError as e:
            errors.append(f"Python syntax error: {e}")

        # ---- Honesty guard (2026-06-26 audit) ----------------------------
        # Reject "availability-probe" scripts: ones whose ONLY action is to
        # import-check a Kratos (sub-)application and write {"note": "..."}
        # without ever running a solve. Such stubs produce no solution yet
        # return rc=0, making the catalog silently wrong. A legitimate Kratos
        # analysis MUST actually solve something — via a Kratos strategy /
        # AnalysisStage, or (as several generators in this backend do) by
        # assembling the system and solving it with scipy/numpy while using
        # KratosMultiphysics for I/O.
        lowered = content.lower()

        # Markers that a genuine solve is present (any one suffices). Covers
        # both Kratos-native strategies and the scipy/numpy assemble-and-solve
        # pattern used by the poisson / heat / elasticity / contact / dynamics
        # generators in this backend.
        #
        # EVERY MARKER IS CALL-SHAPED, AND IMPORT LINES ARE STRIPPED FIRST.
        # Measured 2026-08-07: the MPM generator emitted a standalone numpy
        # material-point method that never touched Kratos, and this guard
        # passed it on the strength of one line —
        #
        #     from scipy.sparse.linalg import spsolve
        #
        # spsolve was never called; neither was lil_matrix. The marker list
        # held both the bare name `spsolve` and the module path
        # `scipy.sparse.linalg`, so an unused import was enough to certify a
        # solve. A guard satisfied by a symbol's PRESENCE rather than its USE
        # is the same defect shape as an expectation satisfied by a word the
        # fixture prints itself: it can only be passed, never failed, by the
        # thing it is supposed to be checking.
        #
        # `AnalysisStage` and `SolvingStrategy` are gone for the same reason:
        # naming a class is not running one. Both templates that used to be
        # covered by them (dem, dem_structures_coupling) call `.Run()` or
        # `.RunSolutionLoop(` and are still covered. Measured over all 21
        # Kratos templates on the 28-application build: 20 contain at least one
        # of the calls below, and mpm — before its rewrite — was the only one
        # that contained none.
        solve_calls = (
            ".Run()",                          # AnalysisStage.Run()
            ".RunSolutionLoop(",
            ".Solve()",                        # strategy / solver .Solve()
            ".SolveSolutionStep(",
            "CreateSolver(",
            "ResidualBasedNewtonRaphsonStrategy(",
            "ResidualBasedLinearStrategy(",
            "spsolve(",                        # scipy sparse direct solve
            "factorized(",                     # scipy prefactored solve
            "np.linalg.solve(",
            "numpy.linalg.solve(",
        )
        executable = "\n".join(
            line for line in content.splitlines()
            if not _IMPORT_LINE.match(line))
        has_solve = any(m in executable for m in solve_calls)

        # A "note"-only summary is the tell-tale signature of the old probe
        # stubs ({"note": "... available"} / {"note": "not installed"}).
        note_only_summary = (
            ('"note"' in content or "'note'" in content)
            and ("not installed" in lowered
                 or "available" in lowered
                 or "not pip-installable" in lowered)
        )
        imports_subapp = "import KratosMultiphysics." in content

        if not has_solve and (note_only_summary or imports_subapp):
            errors.append(
                "Script appears to be an availability-probe stub: it "
                "import-checks Kratos and/or writes a {\"note\": ...} "
                "summary but never runs a solve (no Kratos strategy / "
                "AnalysisStage and no scipy/numpy linear solve). A Kratos "
                "analysis must build a model and actually solve it, not just "
                "report availability."
            )

        # A valid Kratos input must EITHER use KratosMultiphysics (for the
        # model / output / strategy) OR perform a genuine solve (the
        # scipy-assembly generators in this backend import only numpy/scipy
        # and use KratosMultiphysics for VTK output). If it does neither it is
        # not a runnable analysis.
        if "KratosMultiphysics" not in content and not has_solve:
            errors.append(
                "Script neither uses KratosMultiphysics nor runs a solve — "
                "it is not a runnable Kratos analysis."
            )
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        # NO FALLBACK TO THE SERVER'S OWN INTERPRETER. That would let run()
        # try an interpreter check_availability() has already rejected, which
        # is the issue-#40 bug class: discovery and execution answering
        # different questions. tests/test_python_env_consistency.py pins it,
        # and caught exactly this when the fallback was written here.
        python = _find_kratos_python()
        if not python:
            return JobHandle(
                job_id=str(uuid.uuid4())[:8],
                backend_name="kratos",
                work_dir=work_dir,
                status="failed",
                error="Python not found",
            )

        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / "MainKratos.py"
        script_path.write_text(input_content)

        job = JobHandle(
            job_id=str(uuid.uuid4())[:8],
            backend_name="kratos",
            work_dir=work_dir,
            status="running",
        )

        cmd = [python, str(script_path)]

        # If KRATOS_ROOT has a source build, use it over pip-installed version --
        # UNLESS an interpreter was named. A named interpreter is used as it is,
        # with its own site-packages: prepending a source tree to its PYTHONPATH
        # swaps the Kratos it imports for whatever that tree holds. Measured
        # 2026-09-24 in three full test suites: the registry had promoted a
        # recorded, unbuilt source checkout to KRATOS_ROOT, its install/ held a
        # core without ConvectionDiffusionApplication, and every run under the
        # pinned venv (which has it) died on the import while the same runs
        # passed wherever that record did not exist.
        from core.backend import get_env_with_source_root
        if os.environ.get("KRATOS_PYTHON", "").strip():
            env = os.environ.copy()
        else:
            env = get_env_with_source_root("KRATOS_ROOT")

        start = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
                env=env,
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


def register():
    backend = KratosBackend()
    register_backend(backend, aliases=["kratos", "kratosmp"])
    logger.info("Kratos Multiphysics backend registered")
