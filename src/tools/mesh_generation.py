"""
MCP tool for mesh generation using Gmsh.

Generates meshes on non-trivial geometries for use across ALL backends:
- L-shaped domain (corner singularity benchmark)
- Plate with circular hole (stress concentration)
- Channel with cylinder obstacle (CFD benchmark, Schäfer-Turek)
- Custom rectangular domains with refinement

Output formats: .msh (Gmsh), .xdmf (FEniCS), .e (Exodus for 4C)
Cross-solver mesh transfer: one mesh → multiple solvers.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("openpaso.mesh")

from core.output_paths import output_dir as _output_dir  # noqa: E402
_MESH_OUTPUT_DIR = _output_dir("meshes")

# THE BOUNDARIES HAVE NAMES, and the reply lists them. Measured 2026-09-29 in the web
# interface: `channel_cylinder` tagged its five curves 1..5 in whatever order the OCC cut
# returned them and named none of them, so a model reading $PhysicalNames found only "fluid",
# concluded the mesh had no boundaries, and could not put a boundary condition anywhere. Each
# curve is now named by where it lies, with a fixed tag, the same in every mesh of a geometry.
BOUNDARY_NAMES = {
    "l_domain": {"bottom": 1, "inner_vertical": 2, "inner_horizontal": 3, "right": 4,
                 "top": 5, "left": 6},
    "plate_with_hole": {"left": 1, "right": 2, "bottom": 3, "top": 4, "hole": 5},
    "channel_cylinder": {"left": 1, "right": 2, "bottom": 3, "top": 4, "cylinder": 5},
}
DOMAIN_NAMES = {"l_domain": ("domain", 100), "plate_with_hole": ("domain", 100),
                "channel_cylinder": ("fluid", 100)}
# MSH 2.2, because it is the one version every reader openPASO serves takes: netgen's ReadGmsh
# reads 2.2 only (measured: a 4.1 file raises "ValueError: invalid literal for int()"), while
# dolfinx, meshio and scikit-fem read both.
MSH_FILE_VERSION = 2.2


def _obstacle_inside(x0, y0, x1, y1, cx, cy, r, what):
    """Refuse an obstacle that touches or crosses the outer boundary: the cut would change the
    domain's topology, and its curves could no longer be named by where they lie."""
    if not (r > 0 and x0 + r < cx < x1 - r and y0 + r < cy < y1 - r):
        raise ValueError(f"the {what} (centre ({cx}, {cy}), radius {r}) must lie strictly inside "
                         f"the rectangle [{x0}, {x1}] x [{y0}, {y1}]")


def _name_curves_by_position(gmsh, x0, y0, x1, y1, obstacle):
    """Curve tags per name: a curve on x = x0 is "left", x = x1 "right", y = y0 "bottom",
    y = y1 "top", and the one curve off the outer boundary is the obstacle."""
    tol = 1e-6 * max(x1 - x0, y1 - y0)
    found = {"left": [], "right": [], "bottom": [], "top": [], obstacle: []}
    for dim, tag in gmsh.model.getEntities(1):
        x, y, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        name = ("left" if abs(x - x0) < tol else "right" if abs(x - x1) < tol else
                "bottom" if abs(y - y0) < tol else "top" if abs(y - y1) < tol else obstacle)
        found[name].append(tag)
    empty = [n for n, tags in found.items() if not tags]
    if empty:
        gmsh.finalize()
        raise RuntimeError(f"no curve was found for {empty}: the geometry is not a rectangle "
                           f"with one obstacle inside it")
    return found


def reading_note(geometry: str, path) -> str:
    """What the reply says about the written file: its boundary names, and how the codes openPASO
    serves read them. Each reading line is measured, by tests/test_a_mesh_names_its_boundaries.py,
    in that code's own interpreter."""
    names = BOUNDARY_NAMES.get(geometry)
    if not names:
        return ""
    dom, dom_tag = DOMAIN_NAMES[geometry]
    listed = ", ".join(f"{n}={t}" for n, t in names.items())
    one = next(iter(names))
    return (f"\nboundaries (Gmsh physical name = tag): {listed}; domain: {dom}={dom_tag}. "
            f"Format: MSH {MSH_FILE_VERSION}, ASCII. In a script, use the ABSOLUTE path above: "
            f"run_simulation starts every run in a job folder of its own, and a relative path "
            f"is read from there.\n"
            f"Read it with:\n"
            f"  FEniCSx: from dolfinx.io import gmsh as gmshio; md = gmshio.read_from_msh(path, "
            f"MPI.COMM_WORLD, rank=0, gdim=2) -> md.mesh, md.facet_tags (its values are the tags "
            f"above), md.physical_groups['{one}'].tag\n"
            f"  NGSolve: from netgen.read_gmsh import ReadGmsh; mesh = Mesh(ReadGmsh(path)) -> "
            f"mesh.Boundaries('{one}'), H1(mesh, order=..., dirichlet='{one}|...'). Mesh(path) "
            f"itself reads netgen's .vol format only: on a .msh it returns an EMPTY mesh "
            f"(mesh.ne == 0) and raises nothing\n"
            f"  scikit-fem: MeshTri.load(path) -> mesh.boundaries['{one}'] (facet indices)\n"
            f"  any other code: the tags above are the boundary ids its Gmsh reader hands over")


def _generate_l_domain_2d(mesh_size: float = 0.05, output_path: Path = None) -> Path:
    """Generate L-shaped domain mesh. Classic FEM benchmark with corner singularity."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("L-domain")

    # L-domain: [-1,1]^2 minus [0,1]x[-1,0] — matches deal.II hyper_L(-1,1)
    # Points (counterclockwise)
    p1 = gmsh.model.geo.addPoint(-1, -1, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(0, -1, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size * 0.3)  # Fine at re-entrant corner
    p4 = gmsh.model.geo.addPoint(1, 0, 0, mesh_size)
    p5 = gmsh.model.geo.addPoint(1, 1, 0, mesh_size)
    p6 = gmsh.model.geo.addPoint(-1, 1, 0, mesh_size)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p5)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p1)

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6])
    s = gmsh.model.geo.addPlaneSurface([cl])

    # Physical groups for BCs
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [l1], tag=1, name="bottom")
    gmsh.model.addPhysicalGroup(1, [l2], tag=2, name="inner_vertical")
    gmsh.model.addPhysicalGroup(1, [l3], tag=3, name="inner_horizontal")
    gmsh.model.addPhysicalGroup(1, [l4], tag=4, name="right")
    gmsh.model.addPhysicalGroup(1, [l5], tag=5, name="top")
    gmsh.model.addPhysicalGroup(1, [l6], tag=6, name="left")
    gmsh.model.addPhysicalGroup(2, [s], tag=100, name="domain")

    gmsh.model.mesh.generate(2)

    if output_path is None:
        output_path = _MESH_OUTPUT_DIR / "l_domain.msh"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.option.setNumber("Mesh.MshFileVersion", MSH_FILE_VERSION)
    gmsh.write(str(output_path))

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    n_elements = len(gmsh.model.mesh.getElements(2)[1][0]) if gmsh.model.mesh.getElements(2)[1] else 0
    gmsh.finalize()

    return output_path, n_nodes, n_elements


def _generate_plate_with_hole_2d(mesh_size: float = 0.05, radius: float = 0.2,
                                   width: float = 2.0, height: float = 1.0,
                                   output_path: Path = None) -> Path:
    """Plate with circular hole — stress concentration benchmark."""
    _obstacle_inside(-width / 2, -height / 2, width / 2, height / 2, 0.0, 0.0, radius, "hole")
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("plate-with-hole")

    # Rectangle
    rect = gmsh.model.occ.addRectangle(-width/2, -height/2, 0, width, height)
    # Circular hole at center
    hole = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
    # Boolean difference
    gmsh.model.occ.cut([(2, rect)], [(2, hole)])
    gmsh.model.occ.synchronize()
    named = _name_curves_by_position(gmsh, -width / 2, -height / 2, width / 2, height / 2, "hole")

    # Mesh refinement near the hole -- the HOLE only: measured from the distance to every
    # curve, the outer edges were refined as finely as the hole
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", named["hole"])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", mesh_size * 0.3)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", mesh_size)
    gmsh.model.mesh.field.setNumber(2, "DistMin", radius * 0.5)
    gmsh.model.mesh.field.setNumber(2, "DistMax", radius * 3)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # Physical groups, named by position (BOUNDARY_NAMES)
    surfaces = gmsh.model.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=100, name="domain")
    for name, tag in BOUNDARY_NAMES["plate_with_hole"].items():
        gmsh.model.addPhysicalGroup(1, named[name], tag=tag, name=name)

    gmsh.model.mesh.generate(2)

    if output_path is None:
        output_path = _MESH_OUTPUT_DIR / "plate_with_hole.msh"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.option.setNumber("Mesh.MshFileVersion", MSH_FILE_VERSION)
    gmsh.write(str(output_path))

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    n_elements = len(gmsh.model.mesh.getElements(2)[1][0]) if gmsh.model.mesh.getElements(2)[1] else 0
    gmsh.finalize()

    return output_path, n_nodes, n_elements


def _generate_channel_with_cylinder_2d(mesh_size: float = 0.05, cyl_radius: float = 0.05,
                                         cyl_center: tuple = (0.2, 0.2),
                                         channel_length: float = 2.2, channel_height: float = 0.41,
                                         output_path: Path = None) -> Path:
    """Channel with cylinder obstacle — DFG/Schäfer-Turek CFD benchmark geometry."""
    _obstacle_inside(0.0, 0.0, channel_length, channel_height, cyl_center[0], cyl_center[1],
                     cyl_radius, "cylinder")
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("channel-cylinder")

    # Channel
    rect = gmsh.model.occ.addRectangle(0, 0, 0, channel_length, channel_height)
    # Cylinder
    cyl = gmsh.model.occ.addDisk(cyl_center[0], cyl_center[1], 0, cyl_radius, cyl_radius)
    # Cut
    gmsh.model.occ.cut([(2, rect)], [(2, cyl)])
    gmsh.model.occ.synchronize()
    named = _name_curves_by_position(gmsh, 0.0, 0.0, channel_length, channel_height, "cylinder")

    # Refine near the cylinder -- the CYLINDER only. Measured 2026-09-29: with the distance
    # taken to every curve, the channel walls were refined as finely as the cylinder, and a
    # mesh_size of 0.01 on the default channel gave 521,884 nodes.
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", named["cylinder"])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", mesh_size * 0.2)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", mesh_size)
    gmsh.model.mesh.field.setNumber(2, "DistMin", cyl_radius)
    gmsh.model.mesh.field.setNumber(2, "DistMax", cyl_radius * 10)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    surfaces = gmsh.model.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=100, name="fluid")
    for name, tag in BOUNDARY_NAMES["channel_cylinder"].items():
        gmsh.model.addPhysicalGroup(1, named[name], tag=tag, name=name)

    gmsh.model.mesh.generate(2)

    if output_path is None:
        output_path = _MESH_OUTPUT_DIR / "channel_cylinder.msh"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.option.setNumber("Mesh.MshFileVersion", MSH_FILE_VERSION)
    gmsh.write(str(output_path))

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    n_elements = len(gmsh.model.mesh.getElements(2)[1][0]) if gmsh.model.mesh.getElements(2)[1] else 0
    gmsh.finalize()

    return output_path, n_nodes, n_elements


def _convert_msh_to_xdmf(msh_path: Path) -> Path:
    """Convert Gmsh .msh to XDMF for FEniCS."""
    import meshio
    mesh = meshio.read(str(msh_path))
    # Extract triangles only (2D)
    cells = [c for c in mesh.cells if c.type == "triangle"]
    if not cells:
        cells = [c for c in mesh.cells if c.type in ("quad", "tetra", "hexahedron")]
    if cells:
        # Keep only the cells and points
        out_mesh = meshio.Mesh(
            points=mesh.points[:, :3] if mesh.points.shape[1] > 3 else mesh.points,
            cells=cells,
        )
        xdmf_path = msh_path.with_suffix(".xdmf")
        meshio.write(str(xdmf_path), out_mesh)
        return xdmf_path
    return msh_path


def register_mesh_tools(mcp: FastMCP):

    @mcp.tool()
    def generate_mesh(geometry: str, mesh_size: float = 0.05,
                      params: str = "{}") -> str:
        """Generate a mesh on a non-trivial geometry using Gmsh.

        Available geometries:
        - 'l_domain': L-shaped domain (corner singularity benchmark)
        - 'plate_with_hole': Rectangle with circular hole (stress concentration)
        - 'channel_cylinder': Channel with cylinder obstacle (DFG CFD benchmark)
        - 'rectangle': Simple rectangle with custom dimensions

        The mesh is saved in Gmsh (.msh) format and also converted to
        XDMF for FEniCS compatibility. Use the returned path in subsequent
        simulation calls.

        Args:
            geometry: Geometry type (see list above)
            mesh_size: Target element size (smaller = finer mesh)
            params: JSON parameters for geometry customization
                    e.g. '{"radius": 0.3, "width": 4.0}' for plate_with_hole
        """
        try:
            import gmsh
        except ImportError:
            return "ERROR: Gmsh not installed. Run: pip install gmsh"

        try:
            param_dict = json.loads(params)
        except json.JSONDecodeError:
            return f"Invalid params JSON: {params}"

        generators = {
            "l_domain": _generate_l_domain_2d,
            "plate_with_hole": _generate_plate_with_hole_2d,
            "channel_cylinder": _generate_channel_with_cylinder_2d,
        }

        gen = generators.get(geometry.lower())
        if not gen:
            return f"Unknown geometry: {geometry}. Available: {list(generators.keys())}"

        try:
            msh_path, n_nodes, n_elements = gen(mesh_size=mesh_size, **param_dict)

            # Convert to XDMF for FEniCS
            try:
                xdmf_path = _convert_msh_to_xdmf(msh_path)
                xdmf_msg = f"XDMF: {xdmf_path}"
            except Exception as e:
                xdmf_msg = f"XDMF conversion failed: {e}"

            return json.dumps({
                "geometry": geometry,
                "mesh_file": str(msh_path),
                "format": "Gmsh (.msh)",
                "n_nodes": n_nodes,
                "n_elements": n_elements,
                "mesh_size": mesh_size,
                "xdmf": xdmf_msg,
                "usage": {
                    "fenics": f"mesh = meshio.read('{msh_path}') or gmsh.read('{msh_path}')",
                    "dealii": "Use GridIn to read .msh, or use built-in GridGenerator::hyper_L()",
                    "fourc": "Convert to Exodus (.e) via meshio for 4C",
                },
            }, indent=2)

        except Exception as e:
            return f"Mesh generation failed: {e}"
