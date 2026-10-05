"""Generator registry for FEniCSx physics modules.

Each generator module exposes:
  - ``generate(variant, params) -> str`` — returns a runnable FEniCSx script
  - ``KNOWLEDGE`` — dict of domain knowledge for LLM consumption
  - ``VARIANTS`` — list of available variant names

Usage::

    from src.backends.fenics.generators import get_generator, list_all_physics

    script = get_generator("poisson", "2d")({})
    all_physics = list_all_physics()
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Maps physics name -> module name (relative import path within this package).
_PHYSICS_MODULES: dict[str, str] = {
    "poisson":              ".poisson",
    "linear_elasticity":    ".elasticity",
    "heat":                 ".heat",
    "navier_stokes":        ".navier_stokes",
    "stokes":               ".stokes",
    "hyperelasticity":      ".hyperelasticity",
    "thermal_structural":   ".thermal_structural",
    "convection_diffusion": ".convection_diffusion",
    "eigenvalue":           ".eigenvalue",
    "biharmonic":           ".biharmonic",
    "mixed_poisson":        ".mixed_poisson",
    "reaction_diffusion":   ".reaction_diffusion",
    "helmholtz":            ".helmholtz",
    "maxwell":              ".maxwell",
    "nearly_incompressible_elasticity": ".nearly_incompressible_elasticity",
    "fracture":             ".fracture",
    "stokes_darcy":         ".stokes_darcy",
    "matrix_free_poisson":  ".matrix_free_poisson",
    "element_survey":       ".element_survey",
    # Advanced physics — all served by .advanced (multi-physics module)
    "dg_methods":           ".advanced",
    "contact":              ".advanced",
    "multiphase":           ".advanced",
    "time_dependent_heat":  ".advanced",
    "cahn_hilliard":        ".advanced",
    "nonlinear_pde":        ".advanced",
    "magnetostatics":       ".advanced",
}

# Advanced physics names — these share a single module but need per-physics adapters.
_ADVANCED_PHYSICS: frozenset[str] = frozenset({
    "dg_methods",
    "contact",
    "multiphase",
    "time_dependent_heat",
    "cahn_hilliard",
    "nonlinear_pde",
    "magnetostatics",
})

# Cache of imported modules (or adapter objects for advanced physics).
_MODULE_CACHE: dict[str, Any] = {}


class _AdvancedPhysicsAdapter:
    """Thin adapter that makes a single physics entry from advanced.py look like
    a standalone module with ``generate(variant, params)``, ``KNOWLEDGE``, and
    ``VARIANTS`` attributes — matching the interface expected by the rest of
    this package.
    """

    def __init__(self, physics: str, advanced_mod: Any) -> None:
        self._physics = physics
        self._mod = advanced_mod

    @property
    def KNOWLEDGE(self) -> dict:  # noqa: N802
        return self._mod.KNOWLEDGE.get(self._physics, {})

    @property
    def VARIANTS(self) -> list[str]:  # noqa: N802
        return sorted(self._mod.GENERATORS.get(self._physics, {}).keys())

    def generate(self, variant: str, params: dict) -> str:
        return self._mod.generate(self._physics, variant, params)


def _load_module(physics: str) -> Any:
    """Lazily import a generator module by physics name.

    For physics served by the multi-physics ``advanced`` module, returns a
    per-physics :class:`_AdvancedPhysicsAdapter` that presents the standard
    ``generate(variant, params)`` / ``KNOWLEDGE`` / ``VARIANTS`` interface.
    """
    if physics in _MODULE_CACHE:
        return _MODULE_CACHE[physics]

    module_path = _PHYSICS_MODULES.get(physics)
    if module_path is None:
        raise KeyError(
            f"Unknown FEniCS physics: {physics!r}. "
            f"Available: {sorted(_PHYSICS_MODULES)}"
        )

    try:
        raw_mod = importlib.import_module(module_path, package=__name__)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import FEniCS generator module {module_path} "
            f"for physics {physics!r}: {exc}"
        ) from exc

    if physics in _ADVANCED_PHYSICS:
        # Wrap in a per-physics adapter so the rest of the package sees the
        # standard single-physics interface.
        adapter = _AdvancedPhysicsAdapter(physics, raw_mod)
        _MODULE_CACHE[physics] = adapter
        return adapter

    _MODULE_CACHE[physics] = raw_mod
    return raw_mod


def get_generator(physics: str, variant: str) -> Callable[[dict], str]:
    """Return a callable ``generator(params) -> script_str`` for the given physics+variant.

    Parameters
    ----------
    physics : str
        Physics name, e.g. ``"poisson"``, ``"linear_elasticity"``.
    variant : str
        Variant name, e.g. ``"2d"``, ``"3d"``, ``"l_domain"``.

    Returns
    -------
    Callable[[dict], str]
        A function that takes a params dict and returns a runnable script.

    Raises
    ------
    KeyError
        If *physics* is unknown.
    ValueError
        If *variant* is unknown for the given physics.
    """
    mod = _load_module(physics)
    # Return a partial that binds the variant
    def _gen(params: dict) -> str:
        return mod.generate(variant, params)
    return _gen


def generate_script(physics: str, variant: str, params: dict) -> str:
    """Generate a FEniCSx script for the given physics, variant, and parameters.

    Convenience wrapper around :func:`get_generator`.
    """
    mod = _load_module(physics)
    return mod.generate(variant, params)


def get_knowledge(physics: str) -> dict:
    """Return domain knowledge for the given physics.

    Parameters
    ----------
    physics : str
        Physics name.

    Returns
    -------
    dict
        Domain knowledge dictionary.
    """
    mod = _load_module(physics)
    return getattr(mod, "KNOWLEDGE", {})


def get_variants(physics: str) -> list[str]:
    """Return the list of available variants for a physics module."""
    mod = _load_module(physics)
    return getattr(mod, "VARIANTS", [])


def list_all_physics() -> list[str]:
    """Return sorted list of all registered physics names."""
    return sorted(_PHYSICS_MODULES.keys())


# General FEniCS knowledge (not tied to a specific physics module).
GENERAL_KNOWLEDGE = {'description': 'FEniCSx (dolfinx) general capabilities',
 'element_families': {'Lagrange (P/Q)': 'Continuous, arbitrary order, all cell types',
                      'DG': 'Discontinuous Lagrange, order 0+',
                      'Raviart-Thomas': 'H(div) conforming, for mixed methods',
                      'BDM': 'H(div) conforming, full polynomial',
                      'Nedelec 1st kind': 'H(curl) conforming, for '
                                          'Maxwell/electromagnetics',
                      'Nedelec 2nd kind': 'H(curl) conforming, full polynomial',
                      'Crouzeix-Raviart': 'Nonconforming, order 1 only',
                      'Bubble': 'For MINI element enrichment',
                      'Hermite': 'C1 conforming on simplices',
                      'Serendipity': 'Interval/quad/hex only (no triangle/tet/prism/pyramid), '
                                     'fewer DOFs than Q on quad/hex for degree >= 2',
                      'Regge': 'For elasticity complexes'},
 'mesh_types': ['create_unit_square, create_unit_cube, create_box, create_rectangle',
                'Gmsh: dolfinx.io.gmsh.model_to_mesh / dolfinx.io.gmsh.read_from_msh '
                '(2D/3D, mixed cell types) — the module is `gmsh`, NOT `gmshio`, which '
                'was removed in dolfinx 0.10',
                'XDMF import/export, refinement (dolfinx.mesh.refine — Plaza algorithm '
                'on triangles/tets — and dolfinx.mesh.uniform_refine)'],
 'solver_catalogue': {'direct': 'MUMPS, SuperLU_dist, UMFPACK (via PETSc)',
                      'iterative': 'CG, GMRES, BiCGStab, MinRes, Richardson',
                      'preconditioners': 'ILU, ICC, Jacobi, SOR, GAMG, '
                                         'hypre/BoomerAMG, BDDC, fieldsplit',
                      'nonlinear': 'PETSc SNES (newtonls, newtontr, ngmres)',
                      'eigenvalue': 'SLEPc EPS (krylovschur, arnoldi, lanczos)'},
 'unique_features': ['UFL: symbolic weak form language with automatic differentiation',
                     'PETSc/SLEPc: industrial-strength solver infrastructure',
                     'Gmsh integration: complex geometry meshing via Python API',
                     'MPI parallel: distributed assembly + solve out-of-box',
                     'Complex-valued problems: complex PETSc build',
                     'Mixed elements: arbitrary combinations via mixed_element()',
                     'Checkpointing via adios4dolfinx'],
 'petsc_index_size_solver_compat': {'description': 'PETSc-direct-solver choice in several '
                                                   'dolfinx demos depends on the PETSc '
                                                   "index size. PETSc's MUMPS interface "
                                                   'is supported with 64-bit PetscInt '
                                                   "(--with-64-bit-indices); dolfinx's "
                                                   'own test image builds its '
                                                   '64-bit-int PETSc with '
                                                   '--download-mumps (MUMPS is rejected '
                                                   'only with 64-bit BLAS indices). '
                                                   'Those demos nevertheless avoid '
                                                   'MUMPS when PetscInt is '
                                                   '64-bit: the C++ mixed_poisson '
                                                   'demo dispatches at compile time '
                                                   "(sizeof(PetscInt) == 4 ? 'mumps' : "
                                                   "'superlu_dist'), and "
                                                   'demo_half_loaded_waveguide / '
                                                   'demo_scattering_boundary_conditions '
                                                   'exit on int64 with >1 rank, citing '
                                                   "'memory errors in MUMPS'. Source: "
                                                   'cpp/demo/mixed_poisson/main.cpp:345-348.',
                                    'Signal': '[Solver] dolfinx generators that '
                                              'hardcode '
                                              "petsc_options={'pc_factor_mat_solver_type': "
                                              "'mumps', ...} (used in fracture, "
                                              'nearly_incompressible_elasticity, '
                                              'stokes_darcy, hyperelasticity, '
                                              'helmholtz, reaction_diffusion, '
                                              'mixed_poisson, and others) are not '
                                              'rejected by a PETSc build with 64-bit '
                                              'indices (--with-64-bit-indices, '
                                              'sizeof(PetscInt) == 8): PETSc has no '
                                              "'mumps does not support 64-bit "
                                              "integers' error. Several dolfinx demos "
                                              'still avoid MUMPS there (two exit in '
                                              'parallel on int64 citing memory errors '
                                              'in MUMPS). Diagnostic: `python -c '
                                              '"from petsc4py import PETSc; '
                                              'print(PETSc.IntType)"` returns int64 vs '
                                              'int32. To follow those demos on int64: '
                                              "switch to 'superlu_dist' (or 'pastix' / "
                                              "'mkl_pardiso' if available) in "
                                              'petsc_options, OR rebuild PETSc with '
                                              'the default 32-bit indices. The '
                                              'canonical compile-time dispatch from '
                                              'the C++ mixed_poisson demo (line '
                                              '345-348) is `sizeof(PetscInt) == 4 ? '
                                              "'mumps' : 'superlu_dist'` — a runtime "
                                              "Python equivalent is `'superlu_dist' if "
                                              'PETSc.IntType().itemsize == 8 else '
                                              "'mumps'`. Plus: 'mumps' requires the "
                                              'PETSc build to have actually configured '
                                              'MUMPS (--download-mumps); a 32-bit '
                                              "PETSc without MUMPS gives 'Could not "
                                              "locate solver type mumps' at runtime — "
                                              'separate failure mode. (File walk '
                                              'cpp/demo/mixed_poisson/main.cpp '
                                              '2026-06-03.)'},
 'cross_mesh_interpolation': {'description': 'Non-matching-mesh interpolation: take a '
                                             'Function on one mesh and evaluate it '
                                             'onto a Function on a DIFFERENT mesh '
                                             '(e.g., tet→hex transfer, mesh '
                                             'convergence studies on independent '
                                             'meshes, decoupled multiphysics with '
                                             'separate meshes per field). Source: '
                                             'cpp/demo/interpolation_different_meshes/main.cpp '
                                             '+ Python wrappers '
                                             'dolfinx.fem.create_interpolation_data '
                                             'and Function.interpolate_nonmatching.',
                              'python_api': {'step_1_build_pointownership': 'data = '
                                                                            'dolfinx.fem.create_interpolation_data(V_to, '
                                                                            'V_from, '
                                                                            'cells, '
                                                                            'padding=1e-14). '
                                                                            'V_to is '
                                                                            'the '
                                                                            'TARGET '
                                                                            'FunctionSpace '
                                                                            '(the one '
                                                                            'receiving '
                                                                            'values), '
                                                                            'V_from is '
                                                                            'the '
                                                                            'SOURCE. '
                                                                            'cells is '
                                                                            'the INT32 '
                                                                            'array of '
                                                                            'TARGET '
                                                                            'mesh cell '
                                                                            'indices '
                                                                            'to '
                                                                            'interpolate '
                                                                            'onto '
                                                                            '(typically '
                                                                            'all '
                                                                            'cells: '
                                                                            '`np.arange(cell_map.size_local '
                                                                            '+ '
                                                                            'cell_map.num_ghosts, '
                                                                            'dtype=np.int32)`). '
                                                                            'padding '
                                                                            '(default '
                                                                            '1e-14) is '
                                                                            'the '
                                                                            'geometric '
                                                                            'tolerance '
                                                                            'for '
                                                                            'point-in-cell '
                                                                            'ownership '
                                                                            'tests.',
                                             'step_2_interpolate': 'u_to.interpolate_nonmatching(u_from, '
                                                                   'cells, '
                                                                   'interpolation_data=data). '
                                                                   'NOT '
                                                                   'u_to.interpolate(u_from) '
                                                                   '— regular '
                                                                   'interpolate only '
                                                                   'works for '
                                                                   'same-mesh '
                                                                   'Function-to-Function '
                                                                   'transfer.'},
                              'Signal': '[API] FOUR common failure modes in cross-mesh '
                                        'interpolation: (1) Calling '
                                        'Function.interpolate(u_other_mesh) instead of '
                                        'Function.interpolate_nonmatching(u, cells, '
                                        'data) — the regular path tries same-mesh '
                                        'shape-function evaluation and silently '
                                        'produces garbage (sometimes zeros, sometimes '
                                        'uninitialized memory) when meshes differ. No '
                                        'clear error; the interpolated Function looks '
                                        'plausibly-shaped but values are wrong. (2) '
                                        'The Python create_interpolation_data default '
                                        'padding is 1e-14 (machine-eps-tight) while '
                                        'the C++ interpolation-different-meshes demo '
                                        'uses 1e-8. Points lying on the geometric '
                                        'boundary between source cells fall outside '
                                        'any cell with the Python default and get '
                                        'silently zeroed. For near-coincident meshes '
                                        '(FSI fluid/solid interfaces, h-refined target '
                                        'vs. coarser source) the 1e-8 default from the '
                                        'C++ demo is safer; bump padding explicitly to '
                                        '1e-10..1e-8 for boundary points. (3) The '
                                        "`cells` argument is the TARGET mesh's cell "
                                        "indices, NOT the source's. Common mistake: "
                                        'passing source-mesh cells gets you garbage '
                                        'ownership data with cells reading data they '
                                        "don't own. (4) Argument order in "
                                        'create_interpolation_data is (V_to, V_from, '
                                        'cells, padding) but the '
                                        'Function.interpolate_nonmatching signature is '
                                        '(u_from, cells, interpolation_data) — the '
                                        'FROM and TO directions are SWAPPED across the '
                                        'two calls. Reading the function names instead '
                                        'of the kwargs leads to swapped data. (File '
                                        'walk '
                                        'cpp/demo/interpolation_different_meshes/main.cpp '
                                        '2026-06-03.)'},
 'mixed_domain_assembly': {'description': 'Co-dimension-0 mixed assembly: assemble '
                                          'bilinear forms where trial and test spaces '
                                          'live on DIFFERENT meshes (a parent mesh and '
                                          'a submesh of a region of interest). Source: '
                                          'cpp/demo/codim_0_assembly/main.cpp + '
                                          'mixed_codim0.py.',
                           'workflow': ['1. Mark cells with a MeshTags scalar (e.g. 2 '
                                        'for the subregion, 1 elsewhere).',
                                        '2. submesh, emap, v_map, g_map = '
                                        'mesh.create_submesh(parent, tdim, subcells) — '
                                        'RETURNS 4-TUPLE, not just submesh; the '
                                        'EntityMap `emap` is REQUIRED for cross-mesh '
                                        'form assembly. (Re-verified 2026-08-03: '
                                        'dolfinx 0.10.0 returns tuple[Mesh, EntityMap, '
                                        'EntityMap, NDArray[int32]].)',
                                        '3. V = parent FunctionSpace, W = submesh '
                                        'FunctionSpace.',
                                        '4. integration_entities = '
                                        'fem.compute_integration_domains(IntegralType.cell, '
                                        'parent.topology, marker.find(2)).',
                                        '5. subdomain_data = {IntegralType.cell: '
                                        '[(<marker_id>, integration_entities)]} — '
                                        'marker_id (e.g. 3) chosen at UFL '
                                        "form-definition time as dx(3); it's the "
                                        'form-side tag, NOT the MeshTags value (2 '
                                        'above).',
                                        '6. fem.create_form(form, [V, W], V.mesh, '
                                        'subdomain_data, {}, {}, entity_maps=[emap]) — '
                                        'the dolfinx 0.10 signature is '
                                        'create_form(form, function_spaces, msh, '
                                        'subdomains, coefficient_map, constant_map, '
                                        'entity_maps=None). There is NO `parent_mesh=` '
                                        'kwarg (the mesh is the 3rd POSITIONAL arg) '
                                        'and no `coefficients=` / `constants=` kwargs '
                                        '(they are `coefficient_map` / `constant_map`, '
                                        'both positional-or-keyword and both '
                                        'REQUIRED). (Signature re-verified against the '
                                        'installed dolfinx 0.10.0 on 2026-08-03; the '
                                        'older kwarg spelling raises TypeError.)',
                                        'UFL-side idiom '
                                        '(cpp/demo/codim_0_assembly/mixed_codim0.py): '
                                        'the bilinear form is written as `a_mixed = '
                                        'inner(p, v) * dx(domain=mesh, '
                                        'subdomain_id=3)` with trial p on submesh W, '
                                        'test v on parent mesh V. BOTH kwargs are '
                                        'essential: domain=mesh picks the integration '
                                        "domain (parent, not the trial-space's "
                                        'submesh); subdomain_id=3 is the form-side '
                                        'numeric tag bridged by the subdomain_data '
                                        'dict at create_form time. The UFL file must '
                                        'also expose a top-level `forms = [a_mixed, '
                                        'a]` list for ffcx code generation.'],
                           'Signal': '[API] Two distinct tag-id namespaces in '
                                     'mixed-domain assembly that frequently confuse '
                                     'users: (a) the MeshTags value (e.g. 2 for the '
                                     'subregion in this demo) is what marks cells in '
                                     'the PYTHON topology, while (b) the dx(N) marker '
                                     "on the UFL FORM (e.g. 3 in the demo's "
                                     'subdomain_data) is what the form expects at '
                                     'assembly. The subdomain_data dict bridges them '
                                     'by mapping form-tag -> (MeshTags-selected entity '
                                     'list). Confusing the two yields either empty '
                                     'assembly (no marker match) or an unrelated '
                                     'subdomain getting integrated. Use distinct '
                                     'numeric values (e.g. 2 in MeshTags, 3 in dx) '
                                     "until you've validated the mapping. Plus: "
                                     'the C++ demo builds its mesh with '
                                     'mesh::create_cell_partitioner(mesh::GhostMode::shared_facet) '
                                     '(the optional integer argument of '
                                     'create_cell_partitioner is '
                                     'max_facet_to_cell_links, default 2, not an '
                                     'overlap depth). (File '
                                     'walk cpp/demo/codim_0_assembly/main.cpp '
                                     '2026-06-03.)'}}
