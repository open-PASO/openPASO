"""
Field transfer between FEM solvers via VTU/numpy/interpolation.

Foundation for cross-solver coupling: reads VTU output from any backend,
extracts field data at interface nodes, interpolates between non-matching
meshes, and formats data for the target solver.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("openpaso.field_transfer")


@dataclass
class InterfaceData:
    """Field values at an interface between subdomains."""
    coordinates: np.ndarray   # (N, dim) — interface node positions
    values: np.ndarray        # (N,) or (N, n_comp) — field values
    field_name: str
    normal_fluxes: Optional[np.ndarray] = None  # (N,) or (N, n_comp) — outward flux/traction per point; both shapes round-trip (measured)
    # Optional column names, one per column of `values` / `normal_fluxes`. They decide which
    # columns form one physical field for the convergence test (see coupling_driver.field_arrays).
    components: Optional[list] = None
    flux_components: Optional[list] = None

    def __post_init__(self):
        for label, names, arr in (("components", self.components, self.values),
                                  ("flux_components", self.flux_components, self.normal_fluxes)):
            if names is None:
                continue
            if arr is None:
                raise ValueError(f"'{label}' is given but its array is missing")
            if not isinstance(names, (list, tuple)) or not all(isinstance(s, str) and s for s in names):
                raise ValueError(f"'{label}' must be a list of non-empty column names")
            a = np.asarray(arr)
            ncol = a.shape[-1] if a.ndim >= 2 else 1
            if len(names) != ncol:
                raise ValueError(f"'{label}' names {len(names)} columns, but the array has {ncol}")
            if len(set(names)) != len(names):
                raise ValueError(f"'{label}' repeats a column name")

    def to_dict(self) -> dict:
        d = {
            "field_name": self.field_name,
            "n_points": len(self.coordinates),
            "coordinates": self.coordinates.tolist(),
            "values": self.values.tolist(),
        }
        if self.normal_fluxes is not None:
            d["normal_fluxes"] = self.normal_fluxes.tolist()
        if self.components is not None:
            d["components"] = list(self.components)
        if self.flux_components is not None:
            d["flux_components"] = list(self.flux_components)
        return d

    def to_json(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "InterfaceData":
        d = json.loads(path.read_text())
        coords = np.array(d["coordinates"])
        values = np.array(d["values"])
        fluxes = np.array(d["normal_fluxes"]) if "normal_fluxes" in d else None
        return cls(
            coordinates=coords,
            values=values,
            field_name=d["field_name"],
            normal_fluxes=fluxes,
            components=d.get("components"),
            flux_components=d.get("flux_components"),
        )


def extract_interface_from_vtu(
    vtu_path: Path,
    field_name: str,
    interface_coord: float,
    interface_axis: int = 0,
    tol: float = 1e-6,
) -> InterfaceData:
    """Extract field values at an interface plane from a VTU file.

    VOLUME MODE (`interface_axis = -1`). A surface coupling exchanges a field on
    a plane, and slicing one out is what this function was written for. A FIELD
    coupling does not: in thermo-structural interaction both participants own
    the WHOLE body and exchange volume fields — temperature one way, volumetric
    strain the other — so there is no plane to slice and the plane-slicing
    signature cannot express the exchange at all. With `interface_axis = -1`
    every point in the file is taken and `interface_coord` is ignored. The
    returned object is the same InterfaceData the `couple` driver moves; nothing
    downstream needs to know which mode produced it.

    Args:
        vtu_path: Path to VTU result file.
        field_name: Name of the field to extract (e.g. "temperature").
        interface_coord: Coordinate value defining the interface plane.
            Ignored when interface_axis is -1.
        interface_axis: Axis perpendicular to interface (0=x, 1=y, 2=z), or -1
            for the WHOLE VOLUME (field coupling; see above).
        tol: Tolerance for node matching.

    Returns:
        InterfaceData with coordinates, values, and optionally fluxes.
    """
    from core.post_processing import read_mesh

    mesh = read_mesh(vtu_path)
    points = np.asarray(mesh.points)

    if interface_axis == -1:
        mask = np.ones(len(points), dtype=bool)
    else:
        # Find nodes at the interface
        mask = np.abs(points[:, interface_axis] - interface_coord) < tol
    if not np.any(mask):
        raise ValueError(
            f"No nodes found at {['x','y','z'][interface_axis]}={interface_coord} "
            f"(tol={tol}). Range: [{points[:, interface_axis].min():.6f}, "
            f"{points[:, interface_axis].max():.6f}]"
        )

    interface_points = points[mask]
    indices = np.where(mask)[0]

    # Extract field values
    if field_name in mesh.point_data:
        arr = np.asarray(mesh.point_data[field_name])
        interface_values = arr[indices]
    else:
        available = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
        raise ValueError(f"Field '{field_name}' not found. Available: {available}")

    # Sort for a consistent, reproducible ordering. The driver relaxes export
    # vectors ENTRY BY ENTRY, so the order a participant exports in must be the
    # same on every iteration or relaxation is meaningless.
    if interface_axis == -1:
        # lexicographic over all coordinates — there is no tangential direction
        order = np.lexsort(tuple(interface_points[:, i]
                                 for i in reversed(range(points.shape[1]))))
        logger.info(f"Extracted {len(indices)} volume nodes (field coupling)")
    else:
        tangential_axes = [i for i in range(points.shape[1]) if i != interface_axis]
        sort_key = interface_points[:, tangential_axes[0]]
        order = np.argsort(sort_key)
        logger.info(
            f"Extracted {len(indices)} interface nodes at "
            f"{['x','y','z'][interface_axis]}={interface_coord}"
        )

    return InterfaceData(
        coordinates=interface_points[order],
        values=interface_values[order],
        field_name=field_name,
    )


def interpolate_to_points(
    source: InterfaceData,
    target_coords: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate interface data onto target coordinates.

    Uses scipy.interpolate.griddata for non-matching meshes.
    For 1D interfaces (e.g. y-values along x=const), uses interp1d.

    VECTOR FIELDS ARE MAPPED COMPONENT BY COMPONENT. `source.values` may be
    (N,) for a scalar or (N, n_comp) for a displacement, velocity or traction,
    and the two are NOT interchangeable here: `np.interp` takes only a 1-D
    `fp`, so the 1-D branch used to raise on any vector field, and the
    NaN-backfill in the 2-D branch indexed a coordinate array with a
    2-D boolean mask. Every component is interpolated on its own and the
    components are stacked back, so the returned shape always matches
    `source.values` in its trailing dimension.

    Args:
        source: InterfaceData from the source solver.
        target_coords: (M, dim) target node positions.
        method: Interpolation method ('nearest', 'linear', 'cubic').

    Returns:
        Interpolated values at target coordinates: (M,) for a scalar source,
        (M, n_comp) for a vector one.
    """
    src_coords = np.atleast_2d(np.asarray(source.coordinates, float))
    raw = np.asarray(source.values, float)
    vector = raw.ndim >= 2 and raw.shape[-1] > 1
    src_values = raw.reshape(len(src_coords), -1) if raw.size else raw.reshape(0, 1)
    target_coords = np.atleast_2d(np.asarray(target_coords, float))
    ncomp = src_values.shape[1]

    def _shape(a):
        return a if vector else a.ravel()

    # THE INTERFACE'S OWN DIMENSION, not the number of coordinate columns that
    # happen to vary. Dropping constant columns is right only for an
    # axis-aligned interface: a flat patch that is TILTED in 3-D varies in all
    # three columns while being a 2-D manifold, and handing those three columns
    # to Qhull raises "initial simplex is not full dimensional" — the same
    # class of mistake as measuring a surface with a path length. So the
    # intrinsic dimension comes from an SVD of the centred coordinates and the
    # interpolation happens in the interface's OWN principal coordinates, which
    # also fixes the 1-D branch for any interface that is not parallel to an
    # axis. Target points are projected with the SAME basis.
    if len(src_coords) == 0:
        return _shape(np.zeros((len(target_coords), ncomp)))
    origin = src_coords.mean(axis=0)
    centred = src_coords - origin
    try:
        u_, sv, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        sv, vt = np.zeros(1), np.eye(src_coords.shape[1])
    rank = int(np.sum(sv > 1e-8 * sv[0])) if sv.size and sv[0] > 0 else 0

    if rank == 0:
        # Single location — return that constant, per component
        return _shape(np.tile(src_values[0], (len(target_coords), 1)))

    src_p = centred @ vt[:rank].T
    # THE TWO SIDES NEED NOT CARRY THE SAME NUMBER OF COLUMNS. A source read
    # from a VTU always has three (VTK stores z even for a plane problem) while
    # a target assembled in Python commonly has two, and slicing the target to
    # the source's width does not widen it: `target[:, :3]` on an (M, 2) array
    # is still (M, 2), so subtracting the 3-vector `origin` raised
    # "operands could not be broadcast together with shapes (9,2) (3,)" and the
    # skfem<->skfem poisson_dd coupling died in its update step. The previous
    # implementation never met this because it indexed the target by the
    # source's VARYING axes only, which for a plane interface are columns the
    # target does have.
    #
    # A column the target omits is taken to sit at the interface's own position
    # along that axis — it contributes zero once centred, which is exactly
    # right for the constant axis this case is about, and is the same answer
    # the axis-dropping version gave.
    tgt = np.tile(origin, (len(target_coords), 1))
    shared = min(src_coords.shape[1], target_coords.shape[1])
    tgt[:, :shared] = target_coords[:, :shared]
    tgt_p = (tgt - origin) @ vt[:rank].T

    if rank == 1:
        # 1D interface — use numpy interp (robust, no scipy needed)
        s, t = src_p[:, 0], tgt_p[:, 0]
        order = np.argsort(s)
        out = np.column_stack([
            np.interp(t, s[order], src_values[order, c])
            for c in range(ncomp)])
        return _shape(out)

    # 2D+ interface — use scipy griddata in the interface's own coordinates
    from scipy.interpolate import griddata
    cols = []
    for c in range(ncomp):
        col = np.asarray(griddata(src_p, src_values[:, c], tgt_p,
                                  method=method), float).ravel()
        # Fill NaN (extrapolation) with nearest
        nans = ~np.isfinite(col)
        if np.any(nans):
            col[nans] = np.asarray(
                griddata(src_p, src_values[:, c], tgt_p[nans],
                         method="nearest"), float).ravel()
        cols.append(col)
    return _shape(np.column_stack(cols))


def format_for_fenics(
    interface_data: InterfaceData,
    bc_type: str = "dirichlet",
    interface_axis: int = 0,
    interface_coord: float = 0.5,
) -> str:
    """Format interface data as FEniCS Python code snippet.

    Generates code that applies Dirichlet or Neumann BC at the interface.

    SCALAR AND VECTOR fields take different code. For a vector field (a
    displacement, a velocity, a traction) the FEniCSx space is BLOCKED: one
    row of `tabulate_dof_coordinates()` per NODE, and component c of node n at
    scalar index `n*bs + c`. The scalar snippet writes
    `func.x.array[i] = np.interp(...)` with `i` a node index, which on a
    blocked space writes component 0 of node i and leaves every other component
    at zero — and `np.interp` refuses a 2-D `fp` in the first place. So the
    vector form loops over components and indexes `n*bs + c`.

    Args:
        interface_data: Field values at the interface. `values` may be (N,) or
            (N, n_comp).
        bc_type: 'dirichlet' or 'neumann'.
        interface_axis: Axis of the interface (0=x, 1=y).
        interface_coord: Coordinate of the interface.

    Returns:
        Python code snippet for FEniCS.
    """
    coords = np.atleast_2d(np.asarray(interface_data.coordinates, float))
    raw = np.asarray(interface_data.values, float)
    vector = raw.ndim >= 2 and raw.shape[-1] > 1
    values = raw.reshape(len(coords), -1) if raw.size else raw.reshape(0, 1)
    ncomp = values.shape[1]

    # Determine tangential axis
    tangential_axes = [i for i in range(coords.shape[1]) if i != interface_axis]
    tang_ax = tangential_axes[0] if tangential_axes else 0

    tang_vals = coords[:, tang_ax].tolist()
    field_vals = (values.tolist() if vector else values.ravel().tolist())

    axis_name = ['x', 'y', 'z'][interface_axis]
    what = "vals" if bc_type == "dirichlet" else "flux"
    kind = ("Dirichlet BC" if bc_type == "dirichlet"
            else "Neumann BC (flux/traction)")
    header = f"""\
# Interface {kind} from coupled solver
# {len(tang_vals)} nodes at {axis_name}={interface_coord}, \
{ncomp} component(s) per node
_interface_tang = np.array({tang_vals})
_interface_{what} = np.array({field_vals})

def interface_marker(x):
    return np.isclose(x[{interface_axis}], {interface_coord})
"""
    # ONE assignment body for both the scalar and the vector case: `bs` is the
    # space's own block size, so the scalar case (bs == 1) reduces to
    # `array[n]` exactly as before.
    fill = f"""\
_bs = V.dofmap.index_map_bs
_coords = V.tabulate_dof_coordinates()
_ncomp = {ncomp}
for _n in _iface_nodes:
    _tang = _coords[_n, {tang_ax}]
    for _c in range(min(_ncomp, _bs)):
        _col = (_interface_{what} if _ncomp == 1
                else _interface_{what}[:, _c])
        _fn.x.array[_n * _bs + _c] = np.interp(_tang, _interface_tang, _col)
"""
    if bc_type == "dirichlet":
        return header + f"""
interface_facets = mesh.locate_entities_boundary(domain, fdim, interface_marker)
_iface_nodes = fem.locate_dofs_topological(V, fdim, interface_facets)

# Interpolate coupled values onto FEniCS DOFs (component by component)
_fn = fem.Function(V)
{fill}
bc_interface = fem.dirichletbc(_fn, _iface_nodes)
"""
    return header + f"""
interface_facets = mesh.locate_entities_boundary(domain, fdim, interface_marker)
facet_tags = mesh.meshtags(domain, fdim, np.sort(interface_facets),
                            np.full(len(interface_facets), 99, dtype=np.int32))
ds_iface = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)(99)

# Interpolate flux/traction values (component by component)
_fn = fem.Function(V)
_coords0 = V.tabulate_dof_coordinates()
_iface_nodes = np.where(
    np.isclose(_coords0[:, {interface_axis}], {interface_coord}))[0]
{fill}
# Add to RHS:  L += ufl.inner(_fn, v) * ds_iface     (vector)
#              L += _fn * v * ds_iface               (scalar)
"""


def format_for_4c_neumann(
    interface_data: InterfaceData,
    line_id: int = 3,
) -> str:
    """Format interface flux data as 4C DESIGN LINE NEUMANN CONDITIONS.

    For uniform flux, returns a single Neumann condition.
    For non-uniform, returns per-node conditions via DESIGN POINT NEUMANN.

    Args:
        interface_data: Flux values at the interface.
        line_id: Design line ID for the interface.

    Returns:
        YAML snippet for 4C input file.
    """
    values = interface_data.values
    mean_flux = float(np.mean(values))

    # Check if flux is approximately uniform
    if np.std(values) < 1e-8 * abs(mean_flux) if mean_flux != 0 else np.std(values) < 1e-12:
        return f"""\
DESIGN LINE NEUMANN CONDITIONS:
  - E: {line_id}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{mean_flux:.10e}]
    FUNCT: [0]
"""
    else:
        # Non-uniform: use per-node point Neumann (approximate via uniform for now)
        # Full non-uniform would need FUNCT definition — use mean as approximation
        logger.warning(
            f"Non-uniform flux (std={np.std(values):.2e}) approximated by mean={mean_flux:.6e}"
        )
        return f"""\
DESIGN LINE NEUMANN CONDITIONS:
  - E: {line_id}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{mean_flux:.10e}]
    FUNCT: [0]
"""


def compute_interface_flux_fenics_code(
    interface_axis: int = 0,
    interface_coord: float = 0.5,
    field_name: str = "temperature",
    conductivity: float = 1.0,
) -> str:
    """Generate FEniCS code that computes the normal heat flux at the interface.

    The flux is q = -k * dT/dn evaluated at interface nodes.
    This code should be appended to the FEniCS solve script.

    Returns:
        Python code that writes interface_data.json with coordinates, values, and fluxes.
    """
    axis_name = ['x', 'y', 'z'][interface_axis]
    tang_axes = [i for i in range(3) if i != interface_axis]

    return f"""\

# === Compute interface flux ===
import json as _json

# Project gradient
V_grad = fem.functionspace(domain, ("DG", 0, (gdim,)))
grad_T = fem.Function(V_grad)

# Use L2 projection for gradient
_u_grad = ufl.TrialFunction(V_grad)
_v_grad = ufl.TestFunction(V_grad)
_a_grad = ufl.inner(_u_grad, _v_grad) * ufl.dx
_L_grad = ufl.inner(ufl.grad(uh), _v_grad) * ufl.dx
from dolfinx.fem.petsc import LinearProblem as _LP
_prob_grad = _LP(_a_grad, _L_grad, petsc_options_prefix="grad",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu"}})
grad_T = _prob_grad.solve()

# Extract interface node values and flux
_coords = V.tabulate_dof_coordinates()
_mask = np.abs(_coords[:, {interface_axis}] - {interface_coord}) < 1e-6
_iface_dofs = np.where(_mask)[0]
_iface_coords = _coords[_iface_dofs]
_iface_T = uh.x.array[_iface_dofs]

# Normal flux: q_n = -k * dT/d{axis_name}
# Use finite difference from nearby nodes for robustness
_h = 1e-4
_flux = np.zeros(len(_iface_dofs))
for _ii, _dof in enumerate(_iface_dofs):
    _pt = _coords[_dof].copy()
    # Find nearest interior node
    _dists = np.linalg.norm(_coords - _pt, axis=1)
    _dists[_mask] = 1e10  # exclude interface nodes
    _nearest = np.argmin(_dists)
    _dx = _coords[_nearest, {interface_axis}] - _pt[{interface_axis}]
    if abs(_dx) > 1e-12:
        _flux[_ii] = -{conductivity} * (uh.x.array[_nearest] - uh.x.array[_dof]) / _dx
    else:
        _flux[_ii] = 0.0

# Sort by tangential coordinate
_order = np.argsort(_iface_coords[:, {tang_axes[0]}])
_iface_data = {{
    "field_name": "{field_name}",
    "n_points": int(len(_iface_dofs)),
    "coordinates": _iface_coords[_order].tolist(),
    "values": _iface_T[_order].tolist(),
    "normal_fluxes": _flux[_order].tolist(),
}}
with open("interface_data.json", "w") as _f:
    _json.dump(_iface_data, _f, indent=2)
print(f"Interface: {{len(_iface_dofs)}} nodes, T=[{{_iface_T.min():.4f}}, {{_iface_T.max():.4f}}], flux=[{{_flux.min():.4e}}, {{_flux.max():.4e}}]")
"""


def extract_full_field_from_vtu(
    vtu_path: Path,
    field_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract field values at ALL nodes from a VTU file.

    Unlike extract_interface_from_vtu which filters to an interface plane,
    this returns the complete field for full-domain comparison or transfer.

    Args:
        vtu_path: Path to VTU result file.
        field_name: Name of the field to extract (e.g. 'displacement', 'temperature').
            Also tries prefix matching (e.g. 'disp' matches 'displacement').

    Returns:
        Tuple of (coordinates, values) where coordinates is (N, 3) and
        values is (N,) for scalar or (N, n_comp) for vector fields.
    """
    from core.post_processing import read_mesh

    mesh = read_mesh(vtu_path)
    points = np.asarray(mesh.points)

    if field_name in mesh.point_data:
        values = np.asarray(mesh.point_data[field_name])
    else:
        # Try prefix match
        available = list(mesh.point_data.keys())
        for key in available:
            if key.startswith(field_name) or field_name.startswith(key):
                values = np.asarray(mesh.point_data[key])
                logger.info(f"Field '{field_name}' matched as '{key}'")
                break
        else:
            raise ValueError(
                f"Field '{field_name}' not found in {vtu_path.name}. "
                f"Available: {available}"
            )

    logger.info(f"Extracted {len(points)} nodes, field shape {values.shape}")
    return points, values
