"""Kratos auxiliary applications — utility, infrastructure, and legacy apps.

These do not own their own physics problem types, so they have no
GENERATORS — but the agent needs to know about them because other
generators reference them (e.g. FSI pulls in MappingApplication;
parallel runs need TrilinosApplication + MetisApplication; HDF5
output needs HDF5Application).

Source: upstream Kratos `applications/` directory listing.  Every name
in `KNOWLEDGE` below must correspond to a real sub-application.
"""


KNOWLEDGE = {
    "_auxiliary_overview": {
        "description": (
            "Kratos applications that the agent must know about even though "
            "they do not provide their own physics problem type: parallel "
            "infrastructure, I/O, meshing, mapping, statistics, and legacy "
            "predecessors of currently-active applications.  Pulled in as "
            "dependencies of physics applications rather than driven directly."
        ),
        "infrastructure_apps": {
            "TrilinosApplication": (
                "Trilinos linear-solver wrappers (Epetra, AztecOO, Amesos, ML, "
                "plus an MPI AMGCL; no MueLu) for distributed-memory MPI runs.  Required by any "
                "parallel Kratos analysis using iterative or AMG-preconditioned "
                "solvers.  Not published as a PyPI wheel — obtain it by "
                "building Kratos from source with the application added to "
                "KRATOS_APPLICATIONS (`add_app ${KRATOS_APP_DIR}/TrilinosApplication` "
                "in the configure script), `-DUSE_MPI=ON` and TRILINOS_ROOT "
                "pointing at an MPI-built Trilinos.  There is no "
                "`-DTRILINOS_APPLICATION=ON` option."
            ),
            "MetisApplication": (
                "Metis-based mesh partitioner for MPI runs.  Used by the model-"
                "part-IO splitter to produce per-rank .mdpa files.  Pulled in "
                "by every MPI workflow; no Python-facing physics on its own.  "
                "Pip install hint: KratosMetisApplication."
            ),
            "LinearSolversApplication": (
                "Linear-solver wrappers beyond Kratos core (Eigen-based "
                "sparse_qr/sparse_lu/sparse_cg, PARDISO, complex-valued solvers).  "
                "Used by structural and electromagnetic analyses needing direct "
                "factorisation or complex arithmetic.  Pip install hint: "
                "KratosLinearSolversApplication."
            ),
            "HDF5Application": (
                "Parallel HDF5 I/O.  Provides HDF5OutputProcess and HDF5IO for "
                "checkpointing and restart, plus XDMF/ParaView/VisIt-friendly "
                "result storage.  Used by long simulations and FSI restart "
                "workflows.  Pip install hint: KratosHDF5Application."
            ),
            "MedApplication": (
                "MED file format I/O (Salome ecosystem).  Provides import/"
                "export of Salome-generated meshes and post-processing into "
                "MED format.  Pip install hint: KratosMedApplication."
            ),
        },
        "meshing_and_mapping": {
            "MeshingApplication": (
                "Adaptive mesh refinement (h-refinement) and remeshing.  "
                "Provides MMG / PMMG bindings for tetrahedral and triangular "
                "remeshing driven by error indicators.  Used by large-"
                "deformation solid mechanics and adaptive CFD.  Pip install "
                "hint: KratosMeshingApplication."
            ),
            "MeshMovingApplication": (
                "ALE (arbitrary Lagrangian-Eulerian) mesh-moving strategies: "
                "Laplacian smoothing, structural-similarity, and rigid-body "
                "displacement of inner boundaries.  Required by FSI (the fluid "
                "mesh follows the structural interface) and by any moving-"
                "boundary CFD.  Pip install hint: KratosMeshMovingApplication."
            ),
            "MappingApplication": (
                "Inter-mesh field mapping for non-conforming or non-matching "
                "discretisations.  Provides nearest-neighbour, nearest-element, "
                "barycentric, coupling-geometry and projection (3D-2D) mappers "
                "(plus the beam mapper from Kratos 10.4.0 and the RBF mapper from "
                "10.4.2) used by FSI partitioned "
                "solvers, thermo-mechanical coupling, and CoSimulation.  A "
                "mortar mapper is upstream-flagged as under development and "
                "should not be relied on yet.  Pip install hint: "
                "KratosMappingApplication."
            ),
        },
        "analysis_utilities": {
            "StatisticsApplication": (
                "Statistical post-processing of time-series and ensemble "
                "fields: mean, variance, RMS, time-averaged Reynolds stresses, "
                "spatially-averaged quantities.  Used in turbulence post-"
                "processing and uncertainty quantification.  Pip install "
                "hint: KratosStatisticsApplication."
            ),
            "SystemIdentificationApplication": (
                "Sensor-based system identification.  Provides sensor types "
                "(displacement, strain, sensor_view) and a "
                "measurement_residual_response_function used for sensor "
                "placement, damage detection, and parameter calibration "
                "against measured response data.  Upstream README is empty "
                "as of writing; see `custom_sensors/` and `custom_responses/` "
                "for the actually-implemented surface area.  Pip install "
                "hint: KratosSystemIdentificationApplication."
            ),
        },
        "older_solid_and_contact_apps": {
            "SolidMechanicsApplication": (
                "Older solid-mechanics application focused on FEM for solids, "
                "shells and beams (per its upstream README).  "
                "StructuralMechanicsApplication is the modern path that the "
                "current Kratos generators target — prefer it for new work; "
                "use SolidMechanicsApplication only if you are reading an "
                "existing input deck that already targets it.  Not published "
                "on PyPI (there is no KratosSolidMechanicsApplication "
                "distribution); build it from source by adding it to "
                "KRATOS_APPLICATIONS."
            ),
            "ContactMechanicsApplication": (
                "Older contact-mechanics application.  For new structural-"
                "contact workflows the established path is "
                "ContactStructuralMechanicsApplication, which is what the "
                "agent's contact generator already targets.  Upstream README "
                "is empty; surface area is whatever lives under the app's "
                "`custom_*/` directories.  Not published on PyPI (there is no "
                "KratosContactMechanicsApplication distribution); source build "
                "only."
            ),
        },
        "pitfalls": [
            "[Integration] TrilinosApplication is not published on PyPI; to get it you must build Kratos from source with TrilinosApplication added to KRATOS_APPLICATIONS (add_app in the configure script), `-DUSE_MPI=ON` and TRILINOS_ROOT pointing at an MPI-built Trilinos; there is no `-DTRILINOS_APPLICATION=ON` option.  MetisApplication does have a PyPI wheel (KratosMetisApplication), but it is only useful in an MPI-parallel run. Signal: `import KratosMultiphysics.TrilinosApplication` raises ModuleNotFoundError and the package index offers no KratosTrilinosApplication distribution at any version, while KratosMetisApplication does resolve \u2014 the failure is one wheel's absence, not a broken stack.",
            "[Integration] MappingApplication and MeshMovingApplication are *required* (not optional) for partitioned FSI even if not named explicitly in the user-facing JSON \u2014 `FSIApplication`'s `partitioned_fsi_base_solver.py` imports both at runtime. Signal: the run fails inside FSIApplication's partitioned_fsi_base_solver on `import KratosMultiphysics.MappingApplication` or `...MeshMovingApplication` \u2014 a ModuleNotFoundError naming an application the user never wrote in the JSON, not an error on their own import line.",
            "[Numerical] For new analyses prefer StructuralMechanicsApplication and ContactStructuralMechanicsApplication over the older SolidMechanicsApplication / ContactMechanicsApplication paths; the current Kratos generators target the modern apps. Signal: `import KratosMultiphysics.SolidMechanicsApplication` and `...ContactMechanicsApplication` raise ModuleNotFoundError on a pip stack while the modern pair imports \u2014 the older path is unreachable, not merely discouraged.",
            "[API] MappingApplication.MapperFactory is DEPRECATED (moved to KratosMultiphysics core in Kratos 10.x). Calling MAP.MapperFactory.CreateMapper still works in 10.4 but prints a deprecation warning: \"[WARNING] DEPRECATION-Warning; MappingApplication: The \\\"MapperFactory\\\" was moved to the Core! (used for \\\"CreateMapper\\\")\". New code should use `KratosMultiphysics.MapperFactory.CreateMapper(origin, destination, params)` directly. Signal: KratosMultiphysics emits the literal string 'DEPRECATION-Warning' + 'MapperFactory' + 'moved to the Core' on stderr at the call site. (Verified empirically 2026-06-01.)",
            "[API] There is no SphericParticle2D, but Kratos DEM is NOT 3D-only — the 2D particle is registered under the Cylinder* stem. Use CylinderParticle2D (or CylinderContinuumParticle2D for bonded 2D), with the 2D discontinuum law DEM_D_Hertz_viscous_Coulomb2D. Signal: CreateNewElement('SphericParticle2D', ...) raises RuntimeError 'The Element \"SphericParticle2D\" is not registered!' and lists the available DEM element types, while CreateNewElement('CylinderParticle2D', ...) on the same model part succeeds. (Verified by execution 2026-08-07 on a build carrying DEMApplication — corrects the earlier 'DEM is always 3D internally' claim, which mis-advised a 3D sphere with constrained out-of-plane DOFs.)",
            "[Integration] The MODERN contact application IS on PyPI and the LEGACY pair is not, so there is a pip-only path to Contact physics in Kratos 10.x. `pip index versions KratosContactStructuralMechanicsApplication` lists 9.5 through 10.4.3; the legacy KratosContactMechanicsApplication and KratosSolidMechanicsApplication resolve to no distribution at all. Build from source, adding the legacy application to KRATOS_APPLICATIONS (e.g. `add_app ${KRATOS_APP_DIR}/ContactMechanicsApplication` or `SolidMechanicsApplication`), only if you need the legacy path; there is no `-D<APP>_APPLICATION=ON` flag. Signal: `import KratosMultiphysics.ContactStructuralMechanicsApplication` succeeds on a pip stack and prints its Initializing banner, while the legacy import raises ModuleNotFoundError: No module named 'KratosMultiphysics.ContactMechanicsApplication' and pip ends with an error line reporting no matching distribution found for it. (Re-measured by execution 2026-08-09 on Kratos 10.4.3 under /mnt/kratos-tier2/kv/bin/python \u2014 this entry previously claimed BOTH apps were absent from PyPI and quoted a pip 'could not find a version' error for the modern one; both were wrong, the modern wheel has been on PyPI since 9.5.)",
            "[API] Source-of-truth ordering for kratos get_knowledge(): backends.kratos.generators.KNOWLEDGE is the ONLY per-physics catalog consumed by KratosMultiphysics-using clients (the catalog drives every MainKratos.py / ProjectParameters.json the MCP emits). data/kratos_knowledge.py exports per-application constants (KRATOS_APPLICATIONS, STRUCTURAL_MECHANICS, FLUID_DYNAMICS, FSI, GEOMECHANICS, DEM, ...) and NOT a unified KRATOS_KNOWLEDGE dict \u2014 so importing it by-name will fail. Signal: a KratosMultiphysics backend whose get_knowledge wraps `from kratos_knowledge import KRATOS_KNOWLEDGE` in try/except ImportError silently runs the fallback branch \u2014 the import raises CPython's own ImportError, \"cannot import name 'KRATOS_KNOWLEDGE' from 'kratos_knowledge'\" followed by the module's file path — that text is written by the interpreter's import machinery, not by Kratos, and the names it quotes are the ones being reported as MISSING — and the dead-code ModelPart-/Variable-level knowledge lookup is invisible until you add an alignment test. (Removed in 2026-06-02 audit; gated by tests/test_kratos_source_of_truth.py.)",
        ],
    },
}


GENERATORS: dict = {}
