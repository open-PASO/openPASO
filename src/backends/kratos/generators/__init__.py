"""Kratos generator registry — maps physics_variant -> generator function."""

from .poisson import GENERATORS as _poisson_gen, KNOWLEDGE as _poisson_kn
from .heat import GENERATORS as _heat_gen, KNOWLEDGE as _heat_kn
from .linear_elasticity import GENERATORS as _elast_gen, KNOWLEDGE as _elast_kn
from .fluid import GENERATORS as _fluid_gen, KNOWLEDGE as _fluid_kn
from .contact import GENERATORS as _contact_gen, KNOWLEDGE as _contact_kn
from .fsi import GENERATORS as _fsi_gen, KNOWLEDGE as _fsi_kn
from .structural_dynamics import GENERATORS as _dyn_gen, KNOWLEDGE as _dyn_kn
from .dem import GENERATORS as _dem_gen, KNOWLEDGE as _dem_kn
from .mpm import GENERATORS as _mpm_gen, KNOWLEDGE as _mpm_kn
from .shape_optimization import GENERATORS as _shape_gen, KNOWLEDGE as _shape_kn
from .cosimulation import GENERATORS as _cosim_gen, KNOWLEDGE as _cosim_kn
# New generators
from .geomechanics import GENERATORS as _geo_gen, KNOWLEDGE as _geo_kn
from .compressible_flow import GENERATORS as _comp_gen, KNOWLEDGE as _comp_kn
from .rans import GENERATORS as _rans_gen, KNOWLEDGE as _rans_kn
from .pfem import GENERATORS as _pfem_gen, KNOWLEDGE as _pfem_kn
from .rom import GENERATORS as _rom_gen, KNOWLEDGE as _rom_kn
from .topology_optimization import GENERATORS as _topo_gen, KNOWLEDGE as _topo_kn
from .iga import GENERATORS as _iga_gen, KNOWLEDGE as _iga_kn
from .plasticity import GENERATORS as _plast_gen, KNOWLEDGE as _plast_kn
from .specialized import GENERATORS as _spec_gen, KNOWLEDGE as _spec_kn
from .auxiliary_applications import GENERATORS as _aux_gen, KNOWLEDGE as _aux_kn
from .curved_mms import GENERATORS as _curved_gen, KNOWLEDGE as _curved_kn
from .fluid_dynamics import (GENERATORS as _cfd_gen,
                             KNOWLEDGE as _cfd_kn)
from .fsi_partitioned import (GENERATORS as _fsip_gen,
                              KNOWLEDGE as _fsip_kn)

# Merged generator registry: physics_variant -> callable(params) -> str
GENERATORS: dict[str, callable] = {}
for _g in [
    _poisson_gen, _heat_gen, _elast_gen, _fluid_gen,
    _contact_gen, _fsi_gen, _dyn_gen, _dem_gen,
    _mpm_gen, _shape_gen, _cosim_gen,
    _geo_gen, _comp_gen, _rans_gen, _pfem_gen,
    _rom_gen, _topo_gen, _iga_gen, _plast_gen, _spec_gen,
    _aux_gen, _curved_gen, _cfd_gen, _fsip_gen,
]:
    GENERATORS.update(_g)

# Merged knowledge registry: physics_name -> dict
KNOWLEDGE: dict[str, dict] = {}
for _k in [
    _poisson_kn, _heat_kn, _elast_kn, _fluid_kn,
    _contact_kn, _fsi_kn, _dyn_kn, _dem_kn,
    _mpm_kn, _shape_kn, _cosim_kn,
    _geo_kn, _comp_kn, _rans_kn, _pfem_kn,
    _rom_kn, _topo_kn, _iga_kn, _plast_kn, _spec_kn,
    _aux_kn, _curved_kn, _cfd_kn, _fsip_kn,
]:
    KNOWLEDGE.update(_k)

# General knowledge (not physics-specific)
KNOWLEDGE["_general"] = {
    "description": "Kratos Multiphysics general capabilities",
    "installation": "pip install KratosMultiphysics-all (or individual: KratosStructuralMechanicsApplication, etc.)",
    "input_format": {
        "driver": "MainKratos.py (Python script that creates model and runs analysis)",
        "settings": "ProjectParameters.json (solver config, BCs, materials, output, processes)",
        "mesh": "mesh.mdpa (nodes, elements, conditions, SubModelParts, NodalData)",
        "materials": "StructuralMaterials.json or Materials.json (constitutive law + parameters)",
    },
    "mdpa_format": {
        "blocks": ["ModelPartData", "Properties", "Nodes", "Elements", "Conditions",
                   "NodalData", "ElementalData", "ConditionalData", "Table", "SubModelPart"],
        "element_naming": "ElementName + Dimension + NodeCount: SmallDisplacementElement2D3N",
        "submodelparts": "Nested hierarchy for boundary/region identification",
    },
    "applications_count": "47+ applications spanning structure, fluid, thermal, contact, DEM, MPM, geo, optimization, IGA, ROM, PFEM, RANS",
    "linear_solvers": {
        "direct": ["sparse_lu (LinearSolversApplication)", "skyline_lu_factorization",
                   "pardiso_lu (LinearSolversApplication MKL builds only; not in the 10.3.0 pip wheel)"],
        "iterative": "AMGCL (built-in, configurable smoother/krylov/coarsening)",
        "trilinos": ("aztec/cg/bicgstab/gmres (AztecOO), amesos/klu/super_lu_dist/mumps (Amesos), "
                     "multi_level (ML), amgcl (AMGCL); no MueLu (MPI parallel)"),
    },
    "application_loader_pattern": {
        "description": (
            "[Integration] Every Kratos application X ships a Python loader "
            "stub at "
            "applications/X/X.py that follows a 5-line uniform pattern: "
            "(1) from KratosMultiphysics import _ImportApplication, "
            "(2) imports its dependency applications (e.g. CableNet imports "
            "StructuralMechanics), (3) from KratosX import * (the C++ "
            "binding wheel), (4) application = KratosX(), "
            "(5) _ImportApplication(application, 'KratosX'). Users "
            "trigger this by `import KratosMultiphysics.X` — the loader "
            "loads the dependency apps FIRST. Signal: importing an app "
            "whose dependency wasn't built (minimal/custom Kratos build) "
            "fails opaquely on the inner `import "
            "KratosMultiphysics.<dep>` line, not on the user's own "
            "import line. File-walk audit 2026-06-02 catalogued 42 apps "
            "with this pattern."
        ),
    },
}
