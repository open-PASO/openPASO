"""NGSolve generator registry — maps physics_variant -> generator function."""

from .poisson import GENERATORS as _poisson_gen, KNOWLEDGE as _poisson_kn
from .space_survey import (GENERATORS as _ssurv_gen,
                           KNOWLEDGE as _ssurv_kn)
from .linear_elasticity import GENERATORS as _elast_gen, KNOWLEDGE as _elast_kn
from .heat import GENERATORS as _heat_gen, KNOWLEDGE as _heat_kn
from .stokes import GENERATORS as _stokes_gen, KNOWLEDGE as _stokes_kn
from .navier_stokes import GENERATORS as _ns_gen, KNOWLEDGE as _ns_kn
from .maxwell import GENERATORS as _maxwell_gen, KNOWLEDGE as _maxwell_kn
from .helmholtz import GENERATORS as _helmholtz_gen, KNOWLEDGE as _helmholtz_kn
from .hyperelasticity import GENERATORS as _hyper_gen, KNOWLEDGE as _hyper_kn
from .eigenvalue import GENERATORS as _eigen_gen, KNOWLEDGE as _eigen_kn
from .convection_diffusion import GENERATORS as _convdiff_gen, KNOWLEDGE as _convdiff_kn
from .mixed_poisson import GENERATORS as _mixed_gen, KNOWLEDGE as _mixed_kn
from .thermal_structural import GENERATORS as _therm_struct_gen, KNOWLEDGE as _therm_struct_kn
from .surface_pde import GENERATORS as _surface_gen, KNOWLEDGE as _surface_kn
from .plasticity import GENERATORS as _plasticity_gen, KNOWLEDGE as _plasticity_kn
from .advanced import GENERATORS as _advanced_gen, KNOWLEDGE as _advanced_kn

# Merged generator registry: physics_variant -> callable(params) -> str
GENERATORS: dict[str, callable] = {}
for _g in [
    _poisson_gen, _ssurv_gen, _elast_gen, _heat_gen, _stokes_gen, _ns_gen,
    _maxwell_gen, _helmholtz_gen, _hyper_gen, _eigen_gen,
    _convdiff_gen, _mixed_gen, _therm_struct_gen, _surface_gen,
    _plasticity_gen, _advanced_gen,
]:
    GENERATORS.update(_g)

# Merged knowledge registry: physics_name -> dict
KNOWLEDGE: dict[str, dict] = {}
for _k in [
    _poisson_kn, _ssurv_kn, _elast_kn, _heat_kn, _stokes_kn, _ns_kn,
    _maxwell_kn, _helmholtz_kn, _hyper_kn, _eigen_kn,
    _convdiff_kn, _mixed_kn, _therm_struct_kn, _surface_kn,
    _plasticity_kn, _advanced_kn,
]:
    KNOWLEDGE.update(_k)

# General knowledge (not physics-specific)
KNOWLEDGE["_general"] = {
    "description": "NGSolve general capabilities and unique features",
    "finite_element_spaces": {
        "H1": "Continuous Lagrange (any order, scalar or vector via VectorH1)",
        "HCurl": "Nedelec edge elements (Maxwell, H(curl) conforming)",
        "HDiv": "Raviart-Thomas/BDM (H(div) conforming, normal continuous)",
        "L2": "Discontinuous (DG methods, element-local)",
        "FacetFESpace": "DOFs on facets only (HDG hybrid variable)",
        "NumberSpace": "Single global DOF (Lagrange multiplier for constraints)",
        "HCurlCurl": "H(curl curl) space",
        "HDivDiv": "H(div div) space",
        "SurfaceL2": "L2 on surface manifolds",
        "Periodic": "Periodic wrapper around any space",
    },
    "mesh_types": {
        "unit_square": "Built-in 2D unit square",
        "unit_cube": "Built-in 3D unit cube (from netgen.csg)",
        "SplineGeometry": "Custom 2D domains with spline boundaries",
        "CSG": "3D constructive solid geometry (Sphere, OrthoBrick, Cylinder, Boolean ops)",
        "OCC": "OpenCASCADE: STEP/BREP/IGES CAD import, WorkPlane sketching",
    },
    "solvers": {
        "direct": ["sparsecholesky", "umfpack", "pardiso", "mumps (MPI)"],
        "iterative": ["CG", "GMRes", "MinRes", "QMR", "TFQMR"],  # ngsolve.solvers; BiCGStab is C++-only (no Python binding)
        "preconditioners": ["local (Jacobi)", "multigrid", "bddc", "h1amg", "hcurlamg"],
        "nonlinear": ["Newton (built-in)", "solvers.Newton()", "manual Newton loop"],
        "eigenvalue": ["ArnoldiSolver (shift-invert)", "PINVIT"],
    },
    "unique_features": [
        "Symbolic PDE description (write weak forms directly in Python)",
        "Automatic differentiation via Variation() for nonlinear problems",
        "Full De Rham complex: H1 -> HCurl -> HDiv -> L2 at arbitrary order",
        "Static condensation: condense=True eliminates interior DOFs (HDG)",
        "Matrix-free: one element matrix per equivalence class, not per element",
        "NewtonCF/MinimizationCF: nonlinear solves at integration points",
        "Built-in geometry + meshing (Netgen): no external mesher needed",
        "TaskManager for shared-memory parallelism, MPI for distributed",
        "GPU support via ngscuda",
        "Complex-valued problems: just add complex=True to any FESpace",
        "CAD import: OCC reads STEP, BREP, IGES directly",
    ],
}
