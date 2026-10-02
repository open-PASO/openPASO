"""
Inline mesh generation for 4C input files.

Creates NODE COORDS + ELEMENTS sections directly in YAML,
bypassing the need for external Exodus mesh files.
This makes 4C fully self-contained for standard benchmark problems.

WHICH ELEMENT TYPE OWNS 2D STRUCTURAL CELLS IS VERSION-DEPENDENT, and the two
spellings share no keywords:

    WALL  QUAD4 <n..> MAT m KINEM k EAS e THICK t STRESS_STRAIN s GP a b
    SOLID QUAD4 <n..> MAT m KINEM k THICKNESS t PLANE_ASSUMPTION p

Only one of them is registered in any given build: `4C --parameters` lists the
cell types each element factory owns, and in a build where SOLID owns only 3D
cell types, `SOLID QUAD4` aborts before the first assembly with

    Element 'SOLID' does not seem to know cell type 'quad4'.

while in a build where 2D has been folded into SOLID, `WALL` aborts with

    Unknown type 'WALL' of finite element

STRUCT_QUAD4_TYPE / STRUCT_QUAD4_SUFFIX below carry the WALL spelling with all
six of its required keys, because that is what the deployed 4C accepts. Swap
both constants together if you target a build where SOLID owns quad4 — they are
not interchangeable one at a time, since neither keyword set is a subset of the
other.
"""

# 2D structural element spelling. Change BOTH constants together.
STRUCT_QUAD4_TYPE = "WALL QUAD4"
STRUCT_QUAD4_SUFFIX_LINEAR = (
    "MAT 1 KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"
)
STRUCT_QUAD4_SUFFIX_NONLINEAR = (
    "MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"
)


def generate_quad4_rectangle(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0,
                              element_section: str = "STRUCTURE",
                              element_type: str = STRUCT_QUAD4_TYPE,
                              element_suffix: str = STRUCT_QUAD4_SUFFIX_NONLINEAR):
    """Generate inline QUAD4 mesh on [0,lx]×[0,ly].

    Returns dict with:
        nodes: list of "NODE id COORD x y 0.0" strings
        elements: list of element definition strings
        node_grid: dict (i,j) -> node_id for boundary access
        left_nodes, right_nodes, bottom_nodes, top_nodes: boundary node lists
        all_nodes: all node IDs
        n_nodes, n_elements: counts
    """
    nodes = []
    node_grid = {}
    nid = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * lx / nx
            y = j * ly / ny
            nodes.append(f'NODE {nid} COORD {x:.6f} {y:.6f} 0.0')
            node_grid[(i, j)] = nid
            nid += 1

    elements = []
    eid = 1
    elem_section = element_section.upper()
    for j in range(ny):
        for i in range(nx):
            n1 = node_grid[(i, j)]
            n2 = node_grid[(i + 1, j)]
            n3 = node_grid[(i + 1, j + 1)]
            n4 = node_grid[(i, j + 1)]
            elements.append(f'{eid} {element_type} {n1} {n2} {n3} {n4} {element_suffix}')
            eid += 1

    left_nodes = [node_grid[(0, j)] for j in range(ny + 1)]
    right_nodes = [node_grid[(nx, j)] for j in range(ny + 1)]
    bottom_nodes = [node_grid[(i, 0)] for i in range(nx + 1)]
    top_nodes = [node_grid[(i, ny)] for i in range(nx + 1)]
    all_nodes = list(range(1, len(nodes) + 1))

    return {
        "nodes": nodes,
        "elements": elements,
        "node_grid": node_grid,
        "left_nodes": left_nodes,
        "right_nodes": right_nodes,
        "bottom_nodes": bottom_nodes,
        "top_nodes": top_nodes,
        "all_nodes": all_nodes,
        "n_nodes": len(nodes),
        "n_elements": len(elements),
        "geometry_section": f"{elem_section} ELEMENTS",
    }


def matched_poisson_input(nx: int = 32, ny: int = 32) -> str:
    """Poisson -Δu=1 on [0,1]², u=0 on ∂Ω. Matches FEniCS/deal.II setup."""
    mesh = generate_quad4_rectangle(nx, ny, element_section="TRANSPORT",
                                     element_type="TRANSP QUAD4",
                                     element_suffix="MAT 1 TYPE Std")

    yaml = f'''TITLE:
  - "Poisson -Δu=1 on [0,1]² — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 3
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 4
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    # Boundary topology: 4 lines (edges) + 1 surface (domain)
    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["bottom_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    for nid in mesh["right_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 2"\n'
    for nid in mesh["top_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 3"\n'
    for nid in mesh["left_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 4"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'

    return yaml


def matched_heat_input(nx: int = 32, ny: int = 32, T_left: float = 100.0, T_right: float = 0.0) -> str:
    """Heat conduction T_left on left, T_right on right. Matches FEniCS/deal.II."""
    mesh = generate_quad4_rectangle(nx, ny, element_section="TRANSPORT",
                                     element_type="TRANSP QUAD4",
                                     element_suffix="MAT 1 TYPE Std")

    yaml = f'''TITLE:
  - "Heat conduction — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_left}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_right}]
    FUNCT: [0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["left_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    for nid in mesh["right_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 2"\n'

    return yaml


def matched_poisson_rectangle_input(nx: int = 64, ny: int = 32, lx: float = 2.0, ly: float = 1.0) -> str:
    """Poisson -Δu=1 on [0,lx]×[0,ly], u=0 on ∂Ω. Matches FEniCS/deal.II setup."""
    mesh = generate_quad4_rectangle(nx, ny, lx=lx, ly=ly,
                                     element_section="TRANSPORT",
                                     element_type="TRANSP QUAD4",
                                     element_suffix="MAT 1 TYPE Std")

    yaml = f'''TITLE:
  - "Poisson -Δu=1 on [0,{lx}]×[0,{ly}] — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 3
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 4
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["bottom_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    for nid in mesh["right_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 2"\n'
    for nid in mesh["top_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 3"\n'
    for nid in mesh["left_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 4"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'

    return yaml


def matched_heat_rectangle_input(nx: int = 64, ny: int = 32, lx: float = 2.0, ly: float = 1.0,
                                   T_left: float = 100.0, T_right: float = 0.0) -> str:
    """Heat conduction on [0,lx]×[0,ly], T_left on left, T_right on right."""
    mesh = generate_quad4_rectangle(nx, ny, lx=lx, ly=ly,
                                     element_section="TRANSPORT",
                                     element_type="TRANSP QUAD4",
                                     element_suffix="MAT 1 TYPE Std")

    yaml = f'''TITLE:
  - "Heat conduction on [0,{lx}]×[0,{ly}] — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_left}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_right}]
    FUNCT: [0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["left_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    for nid in mesh["right_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 2"\n'

    return yaml


def generate_l_domain_mesh(n: int = 16):
    """Generate L-shaped domain [-1,1]²\\[0,1]×[-1,0] with QUAD4 elements.

    The L-domain is built from a 2n×2n grid covering [-1,1]², then excluding
    elements in the bottom-right quadrant [0,1]×[-1,0].

    Returns same dict format as generate_quad4_rectangle plus boundary info.
    """
    # Create full grid [-1,1]×[-1,1] with 2n subdivisions
    nn = 2 * n  # total subdivisions per direction
    dx = 2.0 / nn
    dy = 2.0 / nn

    # Generate all nodes (even those in excluded region — we'll skip unused ones)
    # First pass: determine which nodes are used
    used_nodes = set()
    for j in range(nn):
        for i in range(nn):
            # Skip elements in bottom-right quadrant: i >= n and j < n
            if i >= n and j < n:
                continue
            # This element uses 4 corners
            used_nodes.update([(i, j), (i+1, j), (i+1, j+1), (i, j+1)])

    # Assign node IDs (sequential)
    node_map = {}
    nid = 1
    for j in range(nn + 1):
        for i in range(nn + 1):
            if (i, j) in used_nodes:
                node_map[(i, j)] = nid
                nid += 1

    # Generate nodes
    nodes = []
    for j in range(nn + 1):
        for i in range(nn + 1):
            if (i, j) in used_nodes:
                x = -1.0 + i * dx
                y = -1.0 + j * dy
                nodes.append(f'NODE {node_map[(i, j)]} COORD {x:.6f} {y:.6f} 0.0')

    # Generate elements
    elements = []
    eid = 1
    for j in range(nn):
        for i in range(nn):
            if i >= n and j < n:
                continue
            n1 = node_map[(i, j)]
            n2 = node_map[(i+1, j)]
            n3 = node_map[(i+1, j+1)]
            n4 = node_map[(i, j+1)]
            elements.append(f'{eid} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std')
            eid += 1

    # Boundary nodes: all nodes on the exterior boundary of the L
    # The L-domain boundary consists of:
    # - Bottom edge: y=-1, x in [-1, 0] (j=0, i=0..n)
    # - Left edge of bottom-right cutout: x=0, y in [-1, 0] (i=n, j=0..n)
    # - Bottom edge of top-right: y=0, x in [0, 1] (j=n, i=n..2n)
    # - Right edge: x=1, y in [0, 1] (i=2n, j=n..2n)
    # - Top edge: y=1, x in [-1, 1] (j=2n, i=0..2n)
    # - Left edge: x=-1, y in [-1, 1] (i=0, j=0..2n)
    boundary_nodes = set()
    # Bottom: y=-1, x in [-1, 0]
    for i in range(n + 1):
        if (i, 0) in node_map:
            boundary_nodes.add(node_map[(i, 0)])
    # Re-entrant vertical: x=0, y in [-1, 0]
    for j in range(n + 1):
        if (n, j) in node_map:
            boundary_nodes.add(node_map[(n, j)])
    # Re-entrant horizontal: y=0, x in [0, 1]
    for i in range(n, nn + 1):
        if (i, n) in node_map:
            boundary_nodes.add(node_map[(i, n)])
    # Right: x=1, y in [0, 1]
    for j in range(n, nn + 1):
        if (nn, j) in node_map:
            boundary_nodes.add(node_map[(nn, j)])
    # Top: y=1, x in [-1, 1]
    for i in range(nn + 1):
        if (i, nn) in node_map:
            boundary_nodes.add(node_map[(i, nn)])
    # Left: x=-1, y in [-1, 1]
    for j in range(nn + 1):
        if (0, j) in node_map:
            boundary_nodes.add(node_map[(0, j)])

    all_nodes = list(range(1, len(nodes) + 1))

    return {
        "nodes": nodes,
        "elements": elements,
        "node_map": node_map,
        "boundary_nodes": sorted(boundary_nodes),
        "all_nodes": all_nodes,
        "n_nodes": len(nodes),
        "n_elements": len(elements),
    }


def matched_l_domain_poisson_input(n: int = 16) -> str:
    """Poisson -Δu=1 on L-domain [-1,1]²\\[0,1]×[-1,0], u=0 on ∂Ω.

    Matches FEniCS (Gmsh L-domain) and deal.II (hyper_L) setup.
    """
    mesh = generate_l_domain_mesh(n)

    yaml = '''TITLE:
  - "Poisson on L-domain — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [0]
'''

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    # All boundary nodes go to DLINE 1 (u=0 Dirichlet)
    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["boundary_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'

    # All nodes on DSURFACE 1 for Neumann source
    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'

    return yaml


def generate_hex8_cube(nx: int, ny: int, nz: int,
                       lx: float = 1.0, ly: float = 1.0, lz: float = 1.0,
                       element_section: str = "TRANSPORT",
                       element_type: str = "TRANSP HEX8",
                       element_suffix: str = "MAT 1 TYPE Std"):
    """Generate inline HEX8 mesh on [0,lx]×[0,ly]×[0,lz].

    Returns dict with nodes, elements, boundary node sets, etc.
    """
    nodes = []
    node_grid = {}
    nid = 1
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                x = i * lx / nx
                y = j * ly / ny
                z = k * lz / nz
                nodes.append(f'NODE {nid} COORD {x:.6f} {y:.6f} {z:.6f}')
                node_grid[(i, j, k)] = nid
                nid += 1

    elements = []
    eid = 1
    elem_section = element_section.upper()
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n1 = node_grid[(i, j, k)]
                n2 = node_grid[(i+1, j, k)]
                n3 = node_grid[(i+1, j+1, k)]
                n4 = node_grid[(i, j+1, k)]
                n5 = node_grid[(i, j, k+1)]
                n6 = node_grid[(i+1, j, k+1)]
                n7 = node_grid[(i+1, j+1, k+1)]
                n8 = node_grid[(i, j+1, k+1)]
                elements.append(
                    f'{eid} {element_type} {n1} {n2} {n3} {n4} {n5} {n6} {n7} {n8} {element_suffix}')
                eid += 1

    # Boundary nodes: all nodes on any face of the cube
    boundary_nodes = set()
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                if i == 0 or i == nx or j == 0 or j == ny or k == 0 or k == nz:
                    boundary_nodes.add(node_grid[(i, j, k)])

    all_nodes = list(range(1, len(nodes) + 1))

    return {
        "nodes": nodes,
        "elements": elements,
        "node_grid": node_grid,
        "boundary_nodes": sorted(boundary_nodes),
        "all_nodes": all_nodes,
        "n_nodes": len(nodes),
        "n_elements": len(elements),
        "geometry_section": f"{elem_section} ELEMENTS",
    }


def matched_poisson_3d_input(n: int = 8) -> str:
    """Poisson -Δu=1 on [0,1]³, u=0 on ∂Ω. Matches FEniCS/deal.II 3D setup."""
    mesh = generate_hex8_cube(n, n, n, element_section="TRANSPORT",
                               element_type="TRANSP HEX8",
                               element_suffix="MAT 1 TYPE Std")

    yaml = f'''TITLE:
  - "Poisson 3D on [0,1]³ — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DESIGN VOL NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [0]
'''

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    # All boundary nodes on DSURFACE 1 (u=0 Dirichlet)
    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["boundary_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'

    # All nodes in DVOL 1 for volumetric source
    yaml += 'DVOL-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DVOLUME 1"\n'

    return yaml


def matched_tsi_oneway_input(
    nx: int = 4, ny: int = 4, nz: int = 4,
    lx: float = 1.0, ly: float = 1.0, lz: float = 1.0,
    E: float = 200e3, nu: float = 0.3, alpha: float = 12e-6,
    T_left: float = 100.0, T_right: float = 0.0, T_ref: float = 0.0,
    conductivity: float = 1.0, capacity: float = 1.0,
    density: float = 1.0,
) -> str:
    """Generate 4C TSI one-way input: thermal expansion of a heated beam.

    3D beam [0,lx]x[0,ly]x[0,lz] with SOLIDSCATRA HEX8 elements.
    Thermal BCs: T_left on x=0, T_right on x=lx, insulated elsewhere.
    Structural BCs: Fix x=0 face (u=0).
    TSI one-way: thermal -> structural (no reverse coupling).

    Used for cross-solver coupling: FEniCS computes temperature, 4C computes
    structural response via native TSI with the same thermal BCs.
    """
    mesh = generate_hex8_cube(
        nx, ny, nz, lx, ly, lz,
        element_section="STRUCTURE",
        element_type="SOLIDSCATRA HEX8",
        element_suffix="MAT 1 KINEM linear TYPE Undefined",
    )
    ng = mesh["node_grid"]

    # Face node sets for boundary conditions
    left_face = sorted({ng[(0, j, k)] for j in range(ny + 1) for k in range(nz + 1)})
    right_face = sorted({ng[(nx, j, k)] for j in range(ny + 1) for k in range(nz + 1)})

    # Temperature function expression for INITIALFIELD
    t_expr = f"{T_left} + ({T_right} - {T_left}) * x / {lx}"

    yaml = f'''TITLE:
  - "TSI one-way: thermal expansion — cross-solver coupling benchmark"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-8
  TOLRES: 1e-8
  MAXITER: 10
  LINEAR_SOLVER: 2
  PREDICT: TangDis
THERMAL DYNAMIC:
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
  TIMESTEP: 1.0
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
TSI DYNAMIC:
  COUPALGO: "tsi_oneway"
  MAXTIME: 1.0
  TIMESTEP: 1.0
  ITEMAX: 1
# COUPVARIABLE Temperature = one-way direction thermo -> structure
# (thermal strain loads the structure). The 4C DEFAULT is Displacement
# (structure -> thermo): with the default, these one-way decks ran to
# rc=0 but produced ZERO displacement — silently wrong physics
# (verified live against a built 4C binary).
TSI DYNAMIC/PARTITIONED:
  COUPVARIABLE: "Temperature"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_ThermoStVenantK:
      YOUNGNUM: 1
      YOUNG: [{E}]
      NUE: {nu}
      DENS: {density}
      THEXPANS: {alpha}
      INITTEMP: {T_ref}
      THERMOMAT: 2
  - MAT: 2
    MAT_Fourier:
      CAPA: {capacity}
      CONDUCT:
        constant: [{conductivity}]
CLONING MATERIAL MAP:
  - SRC_FIELD: "structure"
    SRC_MAT: 1
    TAR_FIELD: "thermo"
    TAR_MAT: 2
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{t_expr}"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
DESIGN SURF THERMO DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_left}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_right}]
    FUNCT: [0]
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
'''

    # Node coordinates
    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    # Surface topology: DSURFACE 1 = left face, DSURFACE 2 = right face
    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in left_face:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'
    for nid in right_face:
        yaml += f'  - "NODE {nid} DSURFACE 2"\n'

    return yaml


def matched_elasticity_input(nx: int = 40, ny: int = 4, E: float = 1000.0, nu: float = 0.3,
                               lx: float = 10.0, ly: float = 1.0) -> str:
    """Cantilever beam lx×ly, fixed left, body force (0,-1). Matches FEniCS/deal.II.

    Uses KINEM linear (small-strain St. Venant-Kirchhoff) so that
    the 4C result matches the linear-elasticity formulations the
    FEniCS / deal.II / NGSolve / scikit-fem cantilever tests use.
    With KINEM nonlinear (finite-strain) the slender 10x1 beam
    deflection diverged from the linear answer by 40%+ (audit
    2026-06-01: u_y_max fourc=7.50 vs fenics=12.96 / dealii=13.23
    on a 10x1 cantilever under the same body force).
    """
    mesh = generate_quad4_rectangle(nx, ny, lx=lx, ly=ly,
                                     element_section="STRUCTURE",
                                     element_type=STRUCT_QUAD4_TYPE,
                                     element_suffix=STRUCT_QUAD4_SUFFIX_LINEAR)

    yaml = f'''TITLE:
  - "Cantilever {lx}x{ly} — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-8
  TOLRES: 1e-8
  # KINEM nonlinear (St. Venant-Kirchhoff finite strain) needs
  # > 2 Newton iters for moderate body loads. With MAXITER=2
  # NOX hits the StatusTest::MaxIters threshold and 4C aborts
  # with MPI_Abort(1). Audit 2026-06-01: the cantilever_10x1
  # e2e test failed in FourC::Solid::Nln::SOLVER::Nox::solve()
  # because the previous default was 2.
  MAXITER: 20
  LINEAR_SOLVER: 1
  PREDICT: TangDis
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E}
      NUE: {nu}
      DENS: 0.0
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
'''

    # Dirichlet: fix each left-edge node individually via DNODE
    yaml += 'DESIGN POINT DIRICH CONDITIONS:\n'
    for i in range(len(mesh["left_nodes"])):
        yaml += f'''  - E: {i+1}
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
'''

    # Body force via surface Neumann
    yaml += '''DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 6
    ONOFF: [0, 1, 0, 0, 0, 0]
    VAL: [0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0, 0, 0, 0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DNODE-NODE TOPOLOGY:\n'
    for i, nid in enumerate(mesh["left_nodes"]):
        yaml += f'  - "NODE {nid} DNODE {i+1}"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'

    return yaml


def generate_l_domain_hex8(n: int = 4, lz: float = 0.5,
                           element_type: str = "SOLIDSCATRA HEX8",
                           element_suffix: str = "MAT 1 KINEM linear TYPE Undefined"):
    """Generate 3D L-shaped domain by extruding the 2D L-domain in z.

    L-domain: [-1,1]^2 \\ [0,1]x[-1,0], extruded to thickness lz.
    Uses HEX8 elements (SOLIDSCATRA for TSI or SOLID for pure structure).

    Returns dict with nodes, elements, face node sets, all_nodes, etc.
    """
    nn = 2 * n
    used_nodes_2d = set()
    for j in range(nn):
        for i in range(nn):
            if i >= n and j < n:
                continue
            used_nodes_2d.update([(i, j), (i+1, j), (i+1, j+1), (i, j+1)])

    nz = max(n // 2, 1)
    dx = 2.0 / nn
    dy = 2.0 / nn
    dz = lz / nz

    node_map = {}
    nid = 1
    nodes = []
    for k in range(nz + 1):
        for j in range(nn + 1):
            for i in range(nn + 1):
                if (i, j) in used_nodes_2d:
                    x = -1.0 + i * dx
                    y = -1.0 + j * dy
                    z = k * dz
                    node_map[(i, j, k)] = nid
                    nodes.append(f'NODE {nid} COORD {x:.6f} {y:.6f} {z:.6f}')
                    nid += 1

    elements = []
    eid = 1
    for k in range(nz):
        for j in range(nn):
            for i in range(nn):
                if i >= n and j < n:
                    continue
                n1 = node_map[(i, j, k)]
                n2 = node_map[(i+1, j, k)]
                n3 = node_map[(i+1, j+1, k)]
                n4 = node_map[(i, j+1, k)]
                n5 = node_map[(i, j, k+1)]
                n6 = node_map[(i+1, j, k+1)]
                n7 = node_map[(i+1, j+1, k+1)]
                n8 = node_map[(i, j+1, k+1)]
                elements.append(
                    f'{eid} {element_type} {n1} {n2} {n3} {n4} '
                    f'{n5} {n6} {n7} {n8} {element_suffix}')
                eid += 1

    left_face = sorted({node_map[(0, j, k)]
                        for j in range(nn + 1) for k in range(nz + 1)
                        if (0, j) in used_nodes_2d})
    top_face = sorted({node_map[(i, nn, k)]
                       for i in range(nn + 1) for k in range(nz + 1)
                       if (i, nn) in used_nodes_2d})
    bottom_face = sorted({node_map[(i, 0, k)]
                          for i in range(n + 1) for k in range(nz + 1)
                          if (i, 0) in used_nodes_2d})
    right_face = sorted({node_map[(nn, j, k)]
                         for j in range(n, nn + 1) for k in range(nz + 1)
                         if (nn, j) in used_nodes_2d})

    all_nodes = list(range(1, len(nodes) + 1))

    return {
        "nodes": nodes,
        "elements": elements,
        "node_map": node_map,
        "left_face": left_face,
        "right_face": right_face,
        "top_face": top_face,
        "bottom_face": bottom_face,
        "all_nodes": all_nodes,
        "n_nodes": len(nodes),
        "n_elements": len(elements),
    }


def matched_l_bracket_tsi_input(
    n: int = 4, lz: float = 0.5,
    E: float = 200e3, nu: float = 0.3, alpha: float = 12e-6,
    T_hot: float = 100.0, T_cold: float = 0.0, T_ref: float = 0.0,
    conductivity: float = 1.0, capacity: float = 1.0, density: float = 1.0,
) -> str:
    """Generate 4C TSI one-way input on L-shaped bracket.

    Demonstrates thermal stress concentration at the re-entrant corner.
    Thermal BCs: T_hot on left face (x=-1), T_cold on right face (x=1).
    Structural BCs: Fix left face (x=-1).
    """
    mesh = generate_l_domain_hex8(
        n=n, lz=lz,
        element_type="SOLIDSCATRA HEX8",
        element_suffix="MAT 1 KINEM linear TYPE Undefined",
    )

    t_expr = f"{T_hot} + ({T_cold} - {T_hot}) * (x + 1.0) / 2.0"

    yaml = f'''TITLE:
  - "L-bracket TSI: thermal stress concentration — cross-solver benchmark"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-8
  TOLRES: 1e-8
  MAXITER: 10
  LINEAR_SOLVER: 2
  PREDICT: TangDis
THERMAL DYNAMIC:
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
  TIMESTEP: 1.0
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
TSI DYNAMIC:
  COUPALGO: "tsi_oneway"
  MAXTIME: 1.0
  TIMESTEP: 1.0
  ITEMAX: 1
# COUPVARIABLE Temperature = one-way direction thermo -> structure
# (thermal strain loads the structure). The 4C DEFAULT is Displacement
# (structure -> thermo): with the default, these one-way decks ran to
# rc=0 but produced ZERO displacement — silently wrong physics
# (verified live against a built 4C binary).
TSI DYNAMIC/PARTITIONED:
  COUPVARIABLE: "Temperature"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_ThermoStVenantK:
      YOUNGNUM: 1
      YOUNG: [{E}]
      NUE: {nu}
      DENS: {density}
      THEXPANS: {alpha}
      INITTEMP: {T_ref}
      THERMOMAT: 2
  - MAT: 2
    MAT_Fourier:
      CAPA: {capacity}
      CONDUCT:
        constant: [{conductivity}]
CLONING MATERIAL MAP:
  - SRC_FIELD: "structure"
    SRC_MAT: 1
    TAR_FIELD: "thermo"
    TAR_MAT: 2
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{t_expr}"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
DESIGN SURF THERMO DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_hot}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_cold}]
    FUNCT: [0]
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
'''

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["left_face"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'
    for nid in mesh["right_face"]:
        yaml += f'  - "NODE {nid} DSURFACE 2"\n'

    return yaml


def matched_heat_transient_input(nx: int = 16, ny: int = 16,
                                 T_left: float = 100.0,
                                 T_right: float = 0.0,
                                 numstep: int = 10,
                                 timestep: float = 0.01,
                                 theta: float = 0.66) -> str:
    """Transient heat conduction on [0,1]^2 with One-Step-Theta.

    Same setup as matched_heat_input (T_left / T_right Dirichlet ends)
    but integrates the parabolic problem for `numstep` steps from a
    zero initial field, so the catalog's heat_transient_2d variant is
    actually transient instead of routing to the stationary solve.
    Small default mesh (16x16) keeps the run < 5 s.
    """
    mesh = generate_quad4_rectangle(nx, ny, element_section="TRANSPORT",
                                     element_type="TRANSP QUAD4",
                                     element_suffix="MAT 1 TYPE Std")

    yaml = f'''TITLE:
  - "Transient heat conduction — One-Step-Theta"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "One_Step_Theta"
  THETA: {theta}
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: {timestep}
  NUMSTEP: {numstep}
  MAXTIME: {numstep * timestep}
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_left}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_right}]
    FUNCT: [0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["left_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    for nid in mesh["right_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 2"\n'

    return yaml


def matched_elasticity_genalpha_input(nx: int = 20, ny: int = 4,
                                      E: float = 1000.0, nu: float = 0.3,
                                      dens: float = 1.0,
                                      numstep: int = 10,
                                      timestep: float = 0.05,
                                      lx: float = 10.0,
                                      ly: float = 1.0) -> str:
    """Transient cantilever under sudden tip-side body load, GenAlpha.

    Same cantilever geometry as matched_elasticity_input, but
    DYNAMICTYPE GenAlpha with non-zero density so the structural
    inertia matters — the catalog's structural_dynamics/genalpha_2d
    variant gets a real transient instead of the placeholder
    template (which aborted in 4C's MatchTree, probe 2026-06-12).
    """
    mesh = generate_quad4_rectangle(nx, ny, lx=lx, ly=ly,
                                     element_section="STRUCTURE",
                                     element_type=STRUCT_QUAD4_TYPE,
                                     element_suffix=STRUCT_QUAD4_SUFFIX_LINEAR)

    yaml = f'''TITLE:
  - "Cantilever {lx}x{ly} transient — GenAlpha"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "GenAlpha"
  TIMESTEP: {timestep}
  NUMSTEP: {numstep}
  MAXTIME: {numstep * timestep}
  TOLDISP: 1e-8
  TOLRES: 1e-8
  MAXITER: 20
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E}
      NUE: {nu}
      DENS: {dens}
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
'''

    yaml += 'DESIGN POINT DIRICH CONDITIONS:\n'
    for i in range(len(mesh["left_nodes"])):
        yaml += f'''  - E: {i+1}
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
'''

    yaml += '''DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 6
    ONOFF: [0, 1, 0, 0, 0, 0]
    VAL: [0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0, 0, 0, 0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DNODE-NODE TOPOLOGY:\n'
    for i, nid in enumerate(mesh["left_nodes"]):
        yaml += f'  - "NODE {nid} DNODE {i+1}"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'

    return yaml


def matched_elasticity_3d_nonlinear_input(n: int = 4,
                                          E: float = 1000.0,
                                          nu: float = 0.3,
                                          load: float = -10.0) -> str:
    """3D unit-cube compression, St. Venant-Kirchhoff, KINEM nonlinear.

    Left face (x=0) fully fixed, right face (x=1) loaded in -y.
    Small n^3 HEX8 mesh so the nonlinear Newton solve completes in
    seconds. Gives the catalog's solid_mechanics/nonlinear_3d
    variant a real finite-strain solve instead of the placeholder
    template (probe 2026-06-12: MatchTree abort).
    """
    mesh = generate_hex8_cube(n, n, n,
                              element_section="STRUCTURE",
                              element_type="SOLID HEX8",
                              element_suffix="MAT 1 KINEM nonlinear")

    # Face node sets from the structured grid
    grid = mesh["node_grid"]
    left_face = sorted(nid for (i, j, k), nid in grid.items() if i == 0)
    right_face = sorted(nid for (i, j, k), nid in grid.items() if i == n)

    yaml = f'''TITLE:
  - "3D cube nonlinear elasticity — finite strain"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-8
  TOLRES: 1e-8
  MAXITER: 20
  LINEAR_SOLVER: 1
  PREDICT: TangDis
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E}
      NUE: {nu}
      DENS: 0.0
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 6
    ONOFF: [0, 1, 0, 0, 0, 0]
    VAL: [0.0, {load}, 0.0, 0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0, 0, 0, 0]
'''

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in left_face:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'
    for nid in right_face:
        yaml += f'  - "NODE {nid} DSURFACE 2"\n'

    return yaml


def matched_ale_2d_input(nx: int = 16, ny: int = 16,
                         E: float = 1.0, nu: float = 0.3,
                         dens: float = 1.0,
                         numstep: int = 10,
                         timestep: float = 0.001,
                         shear: float = 0.25) -> str:
    """Pure 2D ALE mesh motion on [0,1]² (catalog row ale/ale_2d).

    Bottom edge fixed, top edge sheared in x by a time-ramp FUNCT
    (reaches `shear` at MAXTIME); interior follows the
    laplace_material mesh-motion PDE. Mirrors the self-contained
    ale2d_laplace_*.4C.yaml regression inputs (ALE2 QUAD4 elements,
    MAT_Struct_StVenantKirchhoff pseudo-material, UMFPACK) but with
    an inline structured grid and DLINE boundary conditions instead
    of the two-element point-condition setup. Small default mesh
    keeps the run well under 30 s.
    """
    mesh = generate_quad4_rectangle(nx, ny, element_section="ALE",
                                     element_type="ALE2 QUAD4",
                                     element_suffix="MAT 1")
    maxtime = f"{numstep * timestep:g}"

    yaml = f'''TITLE:
  - "2D ALE mesh motion (laplace_material) — sheared top edge"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Ale"
ALE DYNAMIC:
  TIMESTEP: {timestep}
  NUMSTEP: {numstep}
  MAXTIME: {maxtime}
  ALE_TYPE: laplace_material
  RESULTSEVERY: 1
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E}
      NUE: {nu}
      DENS: {dens}
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t/{maxtime}"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
  - E: 2
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [{shear}, 0.0]
    FUNCT: [1, 0]
'''

    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["bottom_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    for nid in mesh["top_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 2"\n'

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'ALE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    return yaml


def matched_level_set_advection_input(nx: int = 16, ny: int = 16,
                                      numstep: int = 10,
                                      timestep: float = 0.01,
                                      radius: float = 0.25) -> str:
    """Level-set advection on [0,1]²: signed-distance circle
    phi0 = sqrt((x-0.5)^2+(y-0.5)^2) - radius advected by a rigid
    rotation velocity field about the domain center.

    Uses PROBLEMTYPE Level_Set with TRANSP HEX8 / TYPE Ls elements on a
    one-element-thick pseudo-2D layer, following the corpus pattern
    (levelset_gaussian_hill_pbc, levelset_zalesaks_disc_*): LEVEL-SET
    CONTROL time loop, VELOCITYFIELD "function" + VELFUNCNO,
    INITIALFIELD field_by_function, MAT_scatra with zero diffusivity.
    HEX8 (not QUAD4) is required: 4C's level-set zero-isosurface
    capture (4C_levelset_intersection_utils.cpp) only supports
    hex8/hex20/hex27 cells and aborts on QUAD4.

    The signed-distance field is shifted by +1.0 so it has no zero
    crossing inside the domain, following the corpus deck
    levelset_gaussian_hill ("shifted by +1.0 not to run into cut
    troubles"). The circular interface is thus the phi = 1 isoline;
    the advection dynamics are unchanged.

    CORRECTED 2026-08-07. This shift was previously justified here as
    a workaround for a missing Qhull ("builds without Qhull abort in
    Cut::TetMesh::call_q_hull when the zero isosurface cuts a hex").
    That is not true of this install: the build has
    FOUR_C_WITH_QHULL:BOOL=ON and lib4C.so links libqhull.so.6, and
    the same deck with the +1.0 removed -- so the zero isosurface
    really does cut the hexes -- runs to completion, rc 0. The shift
    is kept because it is what the corpus deck does, not because the
    unshifted case fails.
    """
    nx = max(2, min(int(nx), 32))
    ny = max(2, min(int(ny), 32))
    numstep = max(1, int(numstep))
    timestep = float(timestep)
    radius = float(radius)
    maxtime = numstep * timestep

    mesh = generate_hex8_cube(nx, ny, 1, lz=1.0 / nx,
                              element_section="TRANSPORT",
                              element_type="TRANSP HEX8",
                              element_suffix="MAT 1 TYPE Ls")

    yaml = f'''TITLE:
  - "Level-set advection on [0,1]² — signed-distance circle in rigid rotation"
PROBLEM TYPE:
  PROBLEMTYPE: "Level_Set"
LEVEL-SET CONTROL:
  NUMSTEP: {numstep}
  TIMESTEP: {timestep}
  MAXTIME: {maxtime}
SCALAR TRANSPORT DYNAMIC:
  NUMSTEP: {numstep}
  TIMESTEP: {timestep}
  MAXTIME: {maxtime}
  MATID: 1
  VELOCITYFIELD: "function"
  VELFUNCNO: 1
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 2
  LINEAR_SOLVER: 1
SCALAR TRANSPORT DYNAMIC/STABILIZATION:
  DEFINITION_TAU: "Taylor_Hughes_Zarins"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Sca_Tra_Solver"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 0.0
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "-(y-0.5)"
  - COMPONENT: 1
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "(x-0.5)"
  - COMPONENT: 2
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
FUNCT2:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "sqrt((x-0.5)^2+(y-0.5)^2)-{radius}+1.0"
DESIGN SURF DIRICH CONDITIONS: []
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'

    return yaml

def matched_nernst_planck_3d_input(n: int = 4,
                                   c_left: float = 2.0, c_right: float = 1.0,
                                   d_cation: float = 2.0, d_anion: float = 1.0,
                                   numstep: int = 10, timestep: float = 0.001) -> str:
    """Nernst-Planck binary electrolyte on [0,1]³ (4C Electrochemistry).

    Two ionic species (MAT_ion, valences +1/-1) plus electric potential
    closed by the electroneutrality condition (ELCH CONTROL EQUPOT "ENC",
    element TYPE ElchNP — syntax matched to the 4C regression corpus,
    e.g. elch_Kwok_Wu_BDF2.4C.yaml). The cation concentration is fixed on
    the x=0 / x=1 faces (c_left / c_right); the anion follows from the
    ENC constraint there. Pinning BOTH species would make the nodal ENC
    row (the potential dof's equation, z1*c1 + z2*c2 = 0) an exact linear
    combination of the two Dirichlet identity rows — a singular Jacobian
    (first Newton update explodes). All other faces are insulated, and
    the potential is grounded at a single corner node (required gauge
    fixing for the ENC formulation). Unequal diffusivities
    establish a liquid-junction potential. The initial field is the linear
    electroneutral profile c = c_left + (c_right - c_left)*x, phi = 0.
    TEMPERATURE 11604.506 K makes F/(R*T) = 1 (corpus convention).
    Transient one-step-theta with nonlinear solver, a few fast steps.
    """
    mesh = generate_hex8_cube(n, n, n, element_section="TRANSPORT",
                              element_type="TRANSP HEX8",
                              element_suffix="MAT 3 TYPE ElchNP")
    ng = mesh["node_grid"]
    left_face = sorted(ng[(0, j, k)] for j in range(n + 1) for k in range(n + 1))
    right_face = sorted(ng[(n, j, k)] for j in range(n + 1) for k in range(n + 1))
    maxtime = numstep * timestep

    yaml = f'''TITLE:
  - "Nernst-Planck binary electrolyte on [0,1]³ — inline-mesh benchmark"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Electrochemistry"
SCALAR TRANSPORT DYNAMIC:
  SOLVERTYPE: "nonlinear"
  TIMEINTEGR: "One_Step_Theta"
  THETA: 1.0
  MAXTIME: {maxtime}
  NUMSTEP: {numstep}
  TIMESTEP: {timestep}
  MATID: 3
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
  LINEAR_SOLVER: 1
SCALAR TRANSPORT DYNAMIC/NONLINEAR:
  ITEMAX: 20
  CONVTOL: 1e-08
SCALAR TRANSPORT DYNAMIC/STABILIZATION:
  STABTYPE: "no_stabilization"
ELCH CONTROL:
  TEMPERATURE: 11604.506
  EQUPOT: "ENC"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Sca_Tra_Solver"
MATERIALS:
  - MAT: 1
    MAT_ion:
      DIFFUSIVITY: {d_cation}
      VALENCE: 1
  - MAT: 2
    MAT_ion:
      DIFFUSIVITY: {d_anion}
      VALENCE: -1
  - MAT: 3
    MAT_matlist:
      LOCAL: false
      NUMMAT: 2
      MATIDS: [1, 2]
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{c_left}+({c_right}-{c_left})*x"
  - COMPONENT: 1
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{c_left}+({c_right}-{c_left})*x"
  - COMPONENT: 2
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 0, 0]
    VAL: [{c_left}, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: 2
    NUMDOF: 3
    ONOFF: [1, 0, 0]
    VAL: [{c_right}, 0.0, 0.0]
    FUNCT: [0, 0, 0]
'''

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'TRANSPORT ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    # Potential ground at one corner node (on the x=0 face)
    yaml += 'DNODE-NODE TOPOLOGY:\n'
    yaml += f'  - "NODE {ng[(0, 0, 0)]} DNODE 1"\n'

    # Concentration Dirichlet faces: x=0 -> DSURFACE 1, x=1 -> DSURFACE 2
    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in left_face:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'
    for nid in right_face:
        yaml += f'  - "NODE {nid} DSURFACE 2"\n'

    return yaml


def matched_low_mach_heated_channel_input(
    nx: int = 32, ny: int = 8,
    lx: float = 4.0, ly: float = 1.0,
    u_max: float = 0.3,
    T_in: float = 293.0, T_wall: float = 350.0,
    numstep: int = 5, timestep: float = 0.1,
) -> str:
    """Low-Mach variable-density 2D heated channel (PROBLEMTYPE
    Low_Mach_Number_Flow).

    Channel [0,lx]x[0,ly]: parabolic inlet velocity (peak u_max) with
    cold inlet temperature T_in, heated bottom wall (thermal Dirichlet
    T_wall, inlet corner excluded), no-slip top/bottom walls, natural
    outflow at x=lx. FLUID QUAD4 elements with MAT_sutherland; the
    temperature equation runs on a TRANSP discretization cloned via
    CLONING MATERIAL MAP — section combination mirrors the working
    corpus case loma_2d_heated_chan_30x100.4C.yaml, with Belos/MueLu
    solvers replaced by self-contained UMFPACK (no XML files).

    Replaces the placeholder low_mach generator template (probe
    2026-06-12: literal <...> scalars + external Exodus mesh ->
    MatchTree abort).

    VTU note: 4C's fluid runtime writer packs nodal dofs assuming 3D
    (3 velocity dofs + pressure 4th), so for this 2D case the
    "velocity" point array carries (vx, vy, p) — the pressure rides in
    the z-component. A standalone PRESSURE array would be all-NaN in
    2D, so it is not requested.
    """
    mesh = generate_quad4_rectangle(
        nx, ny, lx, ly,
        element_section="FLUID",
        element_type="FLUID QUAD4",
        element_suffix="MAT 1 NA Euler",
    )

    # Parabolic profile vanishing at both walls so the inlet velocity
    # Dirichlet is consistent with no-slip at the shared corner nodes.
    profile = f"{4.0 * u_max / (ly * ly):.10g}*y*({ly:.10g}-y)"
    maxtime = numstep * timestep

    yaml = f'''TITLE:
  - "Low-Mach 2D heated channel {nx}x{ny} — self-contained inline mesh"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Low_Mach_Number_Flow"
FLUID DYNAMIC:
  PHYSICAL_TYPE: "Loma"
  LINEAR_SOLVER: 1
  TIMEINTEGR: "Af_Gen_Alpha"
  INITIALFIELD: "field_by_function"
  STARTFUNCNO: 2
  NUMSTEP: {numstep}
  TIMESTEP: {timestep}
  MAXTIME: {maxtime}
  ITEMAX: 10
  ALPHA_M: 0.83333333333333
  ALPHA_F: 0.66666666666666
  GAMMA: 0.66666666666666
FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES:
  TOL_VEL_RES: 1e-06
  TOL_VEL_INC: 1e-06
  TOL_PRES_RES: 1e-06
  TOL_PRES_INC: 1e-06
SCALAR TRANSPORT DYNAMIC:
  SOLVERTYPE: "nonlinear"
  TIMEINTEGR: "Gen_Alpha"
  NUMSTEP: {numstep}
  TIMESTEP: {timestep}
  MAXTIME: {maxtime}
  ALPHA_M: 0.83333333333333
  ALPHA_F: 0.66666666666666
  GAMMA: 0.66666666666666
  MATID: 1
  VELOCITYFIELD: "Navier_Stokes"
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 3
  LINEAR_SOLVER: 2
SCALAR TRANSPORT DYNAMIC/NONLINEAR:
  ITEMAX: 10
  CONVTOL: 1e-06
LOMA CONTROL:
  NUMSTEP: {numstep}
  TIMESTEP: {timestep}
  MAXTIME: {maxtime}
  ITEMAX: 1
  CONVTOL: 0.0001
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Fluid_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Sca_Tra_Solver"
MATERIALS:
  - MAT: 1
    MAT_sutherland:
      REFVISC: 0.01178
      REFTEMP: 293
      SUTHTEMP: 110.4
      SHC: 1004.5
      PRANUM: 1
      THERMPRESS: 98100
      GASCON: 287
CLONING MATERIAL MAP:
  - SRC_FIELD: "fluid"
    SRC_MAT: 1
    TAR_FIELD: "scatra"
    TAR_MAT: 1
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{profile}"
FUNCT2:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{profile}"
  - COMPONENT: 1
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
  - COMPONENT: 2
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
FUNCT3:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{T_in:.10g}"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/FLUID:
  OUTPUT_FLUID: true
  VELOCITY: true
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
  - E: 2
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
  - E: 3
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [1, 0, 0]
    FUNCT: [1, 0, 0]
DESIGN LINE TRANSPORT DIRICH CONDITIONS:
  - E: 3
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_in:.10g}]
    FUNCT: [0]
  - E: 4
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_wall:.10g}]
    FUNCT: [0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'FLUID ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    # DLINE 1 = bottom wall (no-slip), DLINE 2 = top wall (no-slip),
    # DLINE 3 = inlet (x=0), DLINE 4 = heated part of bottom wall
    # (inlet corner excluded so the cold-inlet and hot-wall thermal
    # Dirichlet conditions never disagree on a shared node).
    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["bottom_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    for nid in mesh["top_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 2"\n'
    for nid in mesh["left_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 3"\n'
    for nid in mesh["bottom_nodes"][1:]:
        yaml += f'  - "NODE {nid} DLINE 4"\n'

    return yaml


def matched_porofluid_single_phase_3d_input(
    n: int = 4,
    permeability: float = 1.0,
    viscosity: float = 0.01,
    density: float = 1.0,
    bulk_modulus: float = 100.0,
    p_in: float = 1.0,
    p_out: float = 0.0,
    p_init: float = 0.1,
    numstep: int = 10,
    timestep: float = 0.01,
) -> str:
    """Single-phase Darcy flow through a unit cube (porofluid_pressure_based).

    Pressure-driven flow: p = p_in on the inlet face (x=0), p = p_out on
    the outlet face (x=1), uniform initial pressure p_init.  Single fluid
    phase via the full 4C material hierarchy (MAT_FluidPoroMultiPhase ->
    MAT_FluidPoroSinglePhase -> DofPressure/PhaseLawConstraint/density/
    viscosity/rel-permeability laws), matching the corpus single-phase
    setup in porofluid_pressure_based_elast_3D_hex27.4C.yaml.

    Element lines follow the corpus convention for the porofluid field:
    "FLUID ELEMENTS" section with "POROFLUIDMULTIPHASE HEX8 ... MAT 1"
    (no TYPE suffix) — the old generator template used
    "TRANSPORT ELEMENTS" + "TYPE PoroFluidMultiPhase", which 4C's input
    matcher rejects (MPI_Abort).
    """
    n = max(2, int(n))
    mesh = generate_hex8_cube(
        n, n, n,
        element_section="FLUID",
        element_type="POROFLUIDMULTIPHASE HEX8",
        element_suffix="MAT 1",
    )
    ng = mesh["node_grid"]
    inlet_nodes = sorted(ng[(0, j, k)] for j in range(n + 1) for k in range(n + 1))
    outlet_nodes = sorted(ng[(n, j, k)] for j in range(n + 1) for k in range(n + 1))

    total_time = numstep * timestep

    yaml = f'''TITLE:
  - "Single-phase Darcy flow through a 3D porous unit cube"
PROBLEM TYPE:
  PROBLEMTYPE: "porofluid_pressure_based"
DISCRETISATION:
  NUMSTRUCDIS: 0
  NUMALEDIS: 0
  NUMTHERMDIS: 0
porofluid_dynamic:
  total_simulation_time: {total_time}
  time_integration:
    number_of_time_steps: {numstep}
    time_step_size: {timestep}
    theta: 1
  nonlinear_solver:
    linear_solver_id: 1
  output:
    porosity: false
  initial_condition:
    type: by_function
    function_id: 1
  flux_reconstruction:
    active: true
    solver_id: 1
IO:
  VERBOSITY: "Minimal"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "porofluid_solver"
MATERIALS:
  - MAT: 1
    MAT_FluidPoroMultiPhase:
      LOCAL: false
      PERMEABILITY: {permeability}
      NUMMAT: 1
      MATIDS: [10]
      NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE: 1
  - MAT: 10
    MAT_FluidPoroSinglePhase:
      DENSITYLAWID: 103
      DENSITY: {density}
      RELPERMEABILITYLAWID: 105
      VISCOSITY_LAW_ID: 104
      DOFTYPEID: 101
  - MAT: 101
    MAT_FluidPoroSinglePhaseDofPressure:
      PHASELAWID: 102
  - MAT: 102
    MAT_PhaseLawConstraint: {{}}
  - MAT: 103
    MAT_PoroDensityLawExp:
      BULKMODULUS: {bulk_modulus}
  - MAT: 104
    MAT_FluidPoroViscosityLawConstant:
      VALUE: {viscosity}
  - MAT: 105
    MAT_FluidPoroRelPermeabilityLawConstant:
      VALUE: 1.0
  - MAT: 2
    MAT_StructPoro:
      MATID: 501
      POROLAWID: 502
      INITPOROSITY: 0.4
  - MAT: 501
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 10
      NUE: 0.35
      DENS: 0.1
  - MAT: 502
    MAT_PoroLawDensityDependent:
      DENSITYLAWID: 503
  - MAT: 503
    MAT_PoroDensityLawExp:
      BULKMODULUS: 100
CLONING MATERIAL MAP:
  - SRC_FIELD: "porofluid"
    SRC_MAT: 1
    TAR_FIELD: "structure"
    TAR_MAT: 2
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{p_init}"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{p_in}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{p_out}]
    FUNCT: [0]
RESULT DESCRIPTION:
  - POROFLUIDMULTIPHASE:
      DIS: "porofluid"
      NODE: {inlet_nodes[0]}
      QUANTITY: "phi1"
      VALUE: {p_in}
      TOLERANCE: 1e-06
'''

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'FLUID ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in inlet_nodes:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'
    for nid in outlet_nodes:
        yaml += f'  - "NODE {nid} DSURFACE 2"\n'

    return yaml


def matched_tsi_monolithic_3d_input(
    nx: int = 4, ny: int = 4, nz: int = 4,
    lx: float = 1.0, ly: float = 1.0, lz: float = 1.0,
    E: float = 200e3, nu: float = 0.3, alpha: float = 12e-6,
    T_left: float = 100.0, T_right: float = 0.0, T_ref: float = 0.0,
    conductivity: float = 1.0, capacity: float = 1.0,
    density: float = 1.0,
    numstep: int = 10, timestep: float = 0.001,
) -> str:
    """Generate 4C TSI monolithic input: thermal expansion of a heated beam.

    3D beam [0,lx]x[0,ly]x[0,lz] with SOLIDSCATRA HEX8 elements.
    Thermal BCs: T_left on x=0, T_right on x=lx, insulated elsewhere.
    Structural BCs: Fix x=0 face (u=0).

    Fully MONOLITHIC two-way coupling (COUPALGO tsi_monolithic): the
    structural and thermal residuals are assembled into one block
    system and solved together each Newton step. Layout follows the 4C
    corpus example tsi_lincompression_monolithic_mergeTSImatrix.4C.yaml:
    OneStepTheta (THETA=1) dynamics in both fields, the TSI block
    matrix merged (MERGE_TSI_BLOCK_MATRIX) so the direct UMFPACK
    solver (SOLVER 3) can handle the coupled system without the
    Belos/Teko block-preconditioner XML files.
    """
    mesh = generate_hex8_cube(
        nx, ny, nz, lx, ly, lz,
        element_section="STRUCTURE",
        element_type="SOLIDSCATRA HEX8",
        element_suffix="MAT 1 KINEM linear TYPE Undefined",
    )
    ng = mesh["node_grid"]

    # Face node sets for boundary conditions
    left_face = sorted({ng[(0, j, k)] for j in range(ny + 1) for k in range(nz + 1)})
    right_face = sorted({ng[(nx, j, k)] for j in range(ny + 1) for k in range(nz + 1)})

    maxtime = numstep * timestep

    # Temperature function expression for INITIALFIELD
    t_expr = f"{T_left} + ({T_right} - {T_left}) * x / {lx}"

    yaml = f'''TITLE:
  - "TSI monolithic 3D: thermal expansion of a heated beam (two-way coupling)"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "OneStepTheta"
  TIMESTEP: {timestep}
  NUMSTEP: {numstep}
  MAXTIME: {maxtime}
  TOLDISP: 1e-8
  TOLRES: 1e-6
  MAXITER: 20
  LINEAR_SOLVER: 2
STRUCTURAL DYNAMIC/ONESTEPTHETA:
  THETA: 1
THERMAL DYNAMIC:
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
  TIMESTEP: {timestep}
  NUMSTEP: {numstep}
  MAXTIME: {maxtime}
  TOLTEMP: 1e-7
  TOLRES: 1e-6
  LINEAR_SOLVER: 1
THERMAL DYNAMIC/ONESTEPTHETA:
  THETA: 1
TSI DYNAMIC:
  COUPALGO: "tsi_monolithic"
  NUMSTEP: {numstep}
  MAXTIME: {maxtime}
  TIMESTEP: {timestep}
  ITEMAX: 20
TSI DYNAMIC/MONOLITHIC:
  CONVTOL: 1e-6
  TOLINC: 1e-6
  NORM_RESF: "Rel"
  LINEAR_SOLVER: 3
  MERGE_TSI_BLOCK_MATRIX: true
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
SOLVER 3:
  SOLVER: "UMFPACK"
  NAME: "TSI_Monolithic_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_ThermoStVenantK:
      YOUNGNUM: 1
      YOUNG: [{E}]
      NUE: {nu}
      DENS: {density}
      THEXPANS: {alpha}
      INITTEMP: {T_ref}
      THERMOMAT: 2
  - MAT: 2
    MAT_Fourier:
      CAPA: {capacity}
      CONDUCT:
        constant: [{conductivity}]
CLONING MATERIAL MAP:
  - SRC_FIELD: "structure"
    SRC_MAT: 1
    TAR_FIELD: "thermo"
    TAR_MAT: 2
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{t_expr}"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
DESIGN SURF THERMO DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_left}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_right}]
    FUNCT: [0]
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
'''

    # Node coordinates
    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    # Surface topology: DSURFACE 1 = left face, DSURFACE 2 = right face
    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in left_face:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'
    for nid in right_face:
        yaml += f'  - "NODE {nid} DSURFACE 2"\n'

    return yaml


def matched_tsi_plane_strain_input(
    nx: int = 16, ny: int = 4,
    lx: float = 2.0, ly: float = 0.25, thickness: float | None = None,
    E: float = 200e9, nu: float = 0.3, alpha: float = 12e-6,
    T_ref: float = 293.0, T_left: float = 293.0, T_right: float = 450.0,
    temp_expr: str | None = None,
    density: float = 7850.0,
    conductivity: float = 1.0, capacity: float = 1.0,
) -> str:
    """4C thermo-elastic PLANE STRAIN via a pseudo-2D thin slab (one-way TSI).

    Why this exists: 4C has NO 2D TSI elements — every TSI corpus test is
    3D SOLIDSCATRA. The two 2D structural eletypes both dead-end when a
    thermo material is attached:
      * WALL QUAD4  + MAT_Struct_ThermoStVenantK
          -> "Invalid type of material law for wall element" (4C_w1_mat.cpp:179)
      * SOLID QUAD4 (any material, current builds)
          -> "Element 'SOLID' does not seem to know cell type 'quad4'"
    Both reproduced live against a built 4C binary. The correct route is
    the standard thin-slab trick implemented here:

      - 3D mesh [0,lx]x[0,ly]x[0,t] with ONE SOLIDSCATRA HEX8 layer in z
        (t defaults to ly/ny so elements stay well-shaped),
      - plane strain enforced exactly by fixing u_z on ALL nodes
        (DESIGN VOL DIRICH, ONOFF [0,0,1]),
      - the x=0 face clamped (structural Dirichlet, all 3 dofs),
      - the temperature FIELD imposed on the whole volume via
        DESIGN VOL THERMO DIRICH + FUNCT1 = ``temp_expr`` (also the thermal
        initial field), so a partner code's temperature solution can be
        passed in as a symbolic expression of x/y — the one-way coupled
        (partner -> 4C) use case,
      - COUPALGO tsi_oneway, Statics, single step: thermal expansion of
        the imposed field drives the structural solve.

    ``temp_expr`` defaults to the linear profile
    "T_left + (T_right-T_left)*x/lx". The deck runs rc=0 on a built 4C
    binary; validate the result yourself against the analytic
    plane-strain thermal-expansion estimate for the parameters you pass
    (see tests/test_fourc_inline_tsi.py for the deck-level checks).
    """
    nx = max(1, int(nx)); ny = max(1, int(ny))
    lz = float(thickness) if thickness else float(ly) / ny
    mesh = generate_hex8_cube(
        nx, ny, 1, lx, ly, lz,
        element_section="STRUCTURE",
        element_type="SOLIDSCATRA HEX8",
        element_suffix="MAT 1 KINEM linear TYPE Undefined",
    )
    ng = mesh["node_grid"]

    clamp_face = sorted({ng[(0, j, k)] for j in range(ny + 1) for k in (0, 1)})
    all_nodes = sorted(ng.values())

    if not temp_expr:
        temp_expr = f"{T_left} + ({T_right} - {T_left}) * x / {lx}"

    yaml = f'''TITLE:
  - "TSI plane strain (pseudo-2D thin slab): imposed temperature field -> thermo-elastic expansion"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-8
  TOLRES: 1e-8
  MAXITER: 20
  LINEAR_SOLVER: 2
  PREDICT: TangDis
THERMAL DYNAMIC:
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
  TIMESTEP: 1.0
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
TSI DYNAMIC:
  COUPALGO: "tsi_oneway"
  MAXTIME: 1.0
  TIMESTEP: 1.0
  ITEMAX: 1
# COUPVARIABLE Temperature = one-way direction thermo -> structure
# (thermal strain loads the structure). The 4C DEFAULT is Displacement
# (structure -> thermo): with the default, these one-way decks ran to
# rc=0 but produced ZERO displacement — silently wrong physics
# (verified live against a built 4C binary).
TSI DYNAMIC/PARTITIONED:
  COUPVARIABLE: "Temperature"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_ThermoStVenantK:
      YOUNGNUM: 1
      YOUNG: [{E}]
      NUE: {nu}
      DENS: {density}
      THEXPANS: {alpha}
      INITTEMP: {T_ref}
      THERMOMAT: 2
  - MAT: 2
    MAT_Fourier:
      CAPA: {capacity}
      CONDUCT:
        constant: [{conductivity}]
CLONING MATERIAL MAP:
  - SRC_FIELD: "structure"
    SRC_MAT: 1
    TAR_FIELD: "thermo"
    TAR_MAT: 2
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{temp_expr}"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
# Impose the (partner-supplied) temperature field on the whole volume:
# thermal Dirichlet everywhere = the thermal solve reproduces FUNCT1
# exactly, and one-way TSI turns it into thermal strain.
DESIGN VOL THERMO DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
# Plane strain: u_z = 0 on every node of the slab.
DESIGN VOL DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
# Clamped edge x=0 (all displacement dofs).
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
'''

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'
    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in clamp_face:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'
    yaml += 'DVOL-NODE TOPOLOGY:\n'
    for nid in all_nodes:
        yaml += f'  - "NODE {nid} DVOL 1"\n'

    return yaml


# ── Beam line meshes (BEAM3R cantilevers) ──────────────────────────────


def generate_beam_line_mesh(n_elem: int, length: float = 10.0,
                            order: int = 1) -> dict:
    """Straight beam centerline along the x-axis from 0 to ``length``.

    order=1 -> LINE2 connectivity (n1, n2); n_elem+1 nodes.
    order=2 -> LINE3 connectivity in the 4C convention
               (endpoint1, endpoint2, midpoint); 2*n_elem+1 nodes.

    Returns dict with "nodes" ('NODE id COORD x y z' strings),
    "connectivity" (list of node-id tuples) and "n_nodes".
    """
    n_elem = max(1, int(n_elem))
    order = 2 if int(order) == 2 else 1
    n_nodes = order * n_elem + 1
    dx = float(length) / (n_nodes - 1)
    nodes = [f"NODE {i + 1} COORD {i * dx:.10g} 0.0 0.0"
             for i in range(n_nodes)]
    connectivity = []
    for e in range(n_elem):
        if order == 1:
            connectivity.append((e + 1, e + 2))
        else:
            # LINE3: endpoint1, endpoint2, midpoint (NOT sequential)
            connectivity.append((2 * e + 1, 2 * e + 3, 2 * e + 2))
    return {"nodes": nodes, "connectivity": connectivity,
            "n_nodes": n_nodes}


def matched_beam_cantilever_static_input(
    n_elem: int = 10,
    length: float = 10.0,
    radius: float = 0.1,
    E: float = 1.0e7,
    nu: float = 0.3,
    load_factor: float = 1.0,
    numstep: int = 5,
) -> str:
    """Static cantilever: 10 BEAM3R LINE2 elements, clamped at x=0,
    transverse tip force F_z at x=L, ramped over ``numstep`` load
    steps (Statics + TangDis predictor).

    Circular cross-section of radius r: A = pi r^2,
    I_yy = I_zz = pi r^4 / 4, J = pi r^4 / 2, shear correction 6/7.

    The tip force is chosen so that the LINEAR tip deflection
    F L^3 / (3 E I) equals ``load_factor`` — i.e. the load scales
    with E, so a probe overriding E (e.g. E=1000) converges exactly
    like the default E=1e7 case.

    Element/material syntax mirrors the working corpus case
    beam3r_line2_static_test2.4C.yaml (plain BEAM3R LINE2, 6 TRIADS
    values, NUMDOF 6, MAT_BeamReissnerElastHyper with SHEARMOD).
    Replaces the placeholder beams generator template (probe
    2026-06-12: literal <...> scalars -> 4C MatchTree abort).
    """
    import math

    n_elem = max(1, int(n_elem))
    numstep = max(1, int(numstep))
    G = E / (2.0 * (1.0 + nu))
    A = math.pi * radius ** 2
    I = math.pi * radius ** 4 / 4.0
    J = 2.0 * I
    shearcorr = 6.0 / 7.0
    tip_force = load_factor * 3.0 * E * I / length ** 3
    timestep = 1.0 / numstep

    mesh = generate_beam_line_mesh(n_elem, length, order=1)
    n_nodes = mesh["n_nodes"]

    yaml = f'''TITLE:
  - "Static cantilever beam — {n_elem} BEAM3R LINE2 elements, tip force"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
IO:
  VERBOSITY: "Standard"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/BEAMS:
  OUTPUT_BEAMS: true
  DISPLACEMENT: true
  STRAINS_GAUSSPOINT: true
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  PREDICT: "TangDis"
  TIMESTEP: {timestep:.10g}
  NUMSTEP: {numstep}
  MAXTIME: 1
  TOLRES: 1e-06
  TOLDISP: 1e-08
  MAXITER: 25
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_BeamReissnerElastHyper:
      YOUNG: {E:.10g}
      SHEARMOD: {G:.10g}
      DENS: 1.0
      CROSSAREA: {A:.10g}
      SHEARCORR: {shearcorr:.10g}
      MOMINPOL: {J:.10g}
      MOMIN2: {I:.10g}
      MOMIN3: {I:.10g}
DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 6
    ONOFF: [1, 1, 1, 1, 1, 1]
    VAL: [0, 0, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 0, 0]
DESIGN POINT NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 6
    ONOFF: [0, 0, 1, 0, 0, 0]
    VAL: [0, 0, {tip_force:.10g}, 0, 0, 0]
    FUNCT: [0, 0, 1, 0, 0, 0]
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_TIME: "t"
DNODE-NODE TOPOLOGY:
  - "NODE 1 DNODE 1"
  - "NODE {n_nodes} DNODE 2"
DLINE-NODE TOPOLOGY:
'''
    for nid in range(1, n_nodes + 1):
        yaml += f'  - "NODE {nid} DLINE 1"\n'

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'

    yaml += 'STRUCTURE ELEMENTS:\n'
    for eid, (n1, n2) in enumerate(mesh["connectivity"], start=1):
        yaml += (f'  - "{eid} BEAM3R LINE2 {n1} {n2} MAT 1 '
                 f'TRIADS 0 0 0 0 0 0"\n')

    return yaml


def matched_beam_cantilever_dynamic_input(
    n_elem: int = 10,
    length: float = 10.0,
    radius: float = 0.1,
    E: float = 1.0e7,
    nu: float = 0.3,
    dens: float = 1.0,
    moment_factor: float = 0.2,
    numstep: int = 5,
    timestep: float = 0.01,
    rho_inf: float = 0.9,
) -> str:
    """Dynamic cantilever: 10 BEAM3R LINE3 elements with Hermite
    centerline interpolation (9 DOFs/node), clamped at x=0, bending
    moment M_y at the tip ramped linearly over the run,
    GenAlphaLieGroup time integration (MASSLIN rotations, RHO_INF).

    Circular cross-section of radius r (A = pi r^2,
    I = pi r^4 / 4, J = pi r^4 / 2). The tip moment is chosen so the
    static tip rotation M L / (E I) equals ``moment_factor`` radians
    (linear tip deflection M L^2 / (2 E I) = moment_factor * L / 2) —
    the load scales with E, so a probe overriding E (e.g. E=1000)
    behaves like the default E=1e7 case.

    Element/integrator syntax mirrors the working corpus case
    beam3r_herm2line3_genalpha_liegroup_lineload_dynamic.4C.yaml
    (LINE3 connectivity endpoint1-endpoint2-midpoint, 9 TRIADS
    values + HERMITE_CENTERLINE true, NUMDOF 9 conditions).
    Replaces the placeholder beams generator template (probe
    2026-06-12: literal <...> scalars -> 4C MatchTree abort).
    """
    import math

    n_elem = max(1, int(n_elem))
    numstep = max(2, int(numstep))
    G = E / (2.0 * (1.0 + nu))
    A = math.pi * radius ** 2
    I = math.pi * radius ** 4 / 4.0
    J = 2.0 * I
    shearcorr = 6.0 / 7.0
    tip_moment = moment_factor * E * I / length
    maxtime = numstep * timestep

    mesh = generate_beam_line_mesh(n_elem, length, order=2)
    n_nodes = mesh["n_nodes"]

    yaml = f'''TITLE:
  - "Dynamic cantilever beam — {n_elem} BEAM3R HERM2LINE3 elements, tip moment"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
IO:
  VERBOSITY: "Standard"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/BEAMS:
  OUTPUT_BEAMS: true
  DISPLACEMENT: true
  TRIAD_VISUALIZATIONPOINT: true
  STRAINS_GAUSSPOINT: true
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "GenAlphaLieGroup"
  TIMESTEP: {timestep:.10g}
  NUMSTEP: {numstep}
  MAXTIME: {maxtime:.10g}
  TOLRES: 1e-06
  TOLDISP: 1e-09
  MAXITER: 80
  MASSLIN: "rotations"
  LINEAR_SOLVER: 1
STRUCTURAL DYNAMIC/GENALPHA:
  RHO_INF: {rho_inf:.10g}
STRUCT NOX/Printing:
  Inner Iteration: false
  Outer Iteration StatusTest: false
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_BeamReissnerElastHyper:
      YOUNG: {E:.10g}
      SHEARMOD: {G:.10g}
      DENS: {dens:.10g}
      CROSSAREA: {A:.10g}
      SHEARCORR: {shearcorr:.10g}
      MOMINPOL: {J:.10g}
      MOMIN2: {I:.10g}
      MOMIN3: {I:.10g}
DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 9
    ONOFF: [1, 1, 1, 1, 1, 1, 0, 0, 0]
    VAL: [0, 0, 0, 0, 0, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 0, 0, 0, 0, 0]
DESIGN POINT NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 9
    ONOFF: [0, 0, 0, 0, 1, 0, 0, 0, 0]
    VAL: [0, 0, 0, 0, {tip_moment:.10g}, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 1, 0, 0, 0, 0]
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_TIME: "t/{maxtime:.10g}"
DNODE-NODE TOPOLOGY:
  - "NODE 1 DNODE 1"
  - "NODE {n_nodes} DNODE 2"
DLINE-NODE TOPOLOGY:
'''
    for nid in range(1, n_nodes + 1):
        yaml += f'  - "NODE {nid} DLINE 1"\n'

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'

    yaml += 'STRUCTURE ELEMENTS:\n'
    for eid, (n1, n2, nmid) in enumerate(mesh["connectivity"], start=1):
        yaml += (f'  - "{eid} BEAM3R LINE3 {n1} {n2} {nmid} MAT 1 '
                 f'TRIADS 0 0 0 0 0 0 0 0 0 HERMITE_CENTERLINE true"\n')

    return yaml


def matched_thermo_2d_input(nx: int = 16, ny: int = 16,
                            T_left: float = 100.0, T_right: float = 0.0,
                            conductivity: float = 1.0,
                            capacity: float = 1.0) -> str:
    """Pure-thermal (PROBLEMTYPE "Thermo") 2D steady heat conduction.

    [0,1]^2 QUAD4 mesh with THERMO QUAD4 elements + MAT_Fourier,
    T_left on x=0, T_right on x=1, insulated top/bottom. Steady state
    (THERMAL DYNAMIC -> DYNAMICTYPE Statics), so the exact solution is
    the linear profile T(x) = T_left + (T_right - T_left) * x.
    Self-contained: inline NODE COORDS, no external mesh files.
    """
    mesh = generate_quad4_rectangle(nx, ny,
                                    element_section="THERMO",
                                    element_type="THERMO QUAD4",
                                    element_suffix="MAT 1")

    yaml = f'''TITLE:
  - "Pure thermal 2D steady conduction - inline mesh benchmark"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
THERMAL DYNAMIC:
  DYNAMICTYPE: Statics
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: {capacity}
      CONDUCT:
        constant: [{conductivity}]
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_left}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_right}]
    FUNCT: [0]
'''

    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["left_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    for nid in mesh["right_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 2"\n'

    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'THERMO ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    return yaml


def matched_thermo_3d_input(n: int = 6,
                            T_left: float = 100.0, T_right: float = 0.0,
                            conductivity: float = 1.0,
                            capacity: float = 1.0,
                            numstep: int = 5,
                            timestep: float = 0.1) -> str:
    """Pure-thermal (PROBLEMTYPE "Thermo") 3D transient heat conduction.

    Unit cube [0,1]^3 with n x n x n THERMO HEX8 elements + MAT_Fourier.
    T_left on the x=0 face, T_right on the x=1 face, insulated elsewhere;
    zero initial field, one-step-theta time integration for a few steps.
    Resolution is the single parameter "n" (NOT nx/ny/nz) so a generic
    parameter sweep cannot inflate the 3D mesh. Self-contained: inline
    NODE COORDS, no external mesh files.
    """
    mesh = generate_hex8_cube(n, n, n,
                              element_section="THERMO",
                              element_type="THERMO HEX8",
                              element_suffix="MAT 1")
    ng = mesh["node_grid"]
    left_face = sorted({ng[(0, j, k)] for j in range(n + 1)
                        for k in range(n + 1)})
    right_face = sorted({ng[(n, j, k)] for j in range(n + 1)
                         for k in range(n + 1)})

    maxtime = numstep * timestep

    yaml = f'''TITLE:
  - "Pure thermal 3D transient conduction - inline mesh benchmark"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
THERMAL DYNAMIC:
  DYNAMICTYPE: OneStepTheta
  INITIALFIELD: "zero_field"
  TIMESTEP: {timestep}
  NUMSTEP: {numstep}
  MAXTIME: {maxtime}
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: {capacity}
      CONDUCT:
        constant: [{conductivity}]
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_left}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_right}]
    FUNCT: [0]
'''

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in left_face:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'
    for nid in right_face:
        yaml += f'  - "NODE {nid} DSURFACE 2"\n'

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'THERMO ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    return yaml


def matched_thermo_transient_mms_input(n: int = 48,
                                       lx: float = 1.0, ly: float = 1.0,
                                       kappa: float = 1.0,
                                       rho: float = 1.0, c: float = 1.0,
                                       theta: float = 0.5,
                                       dt: float = 0.02,
                                       t_end: float = 0.4,
                                       temp_offset: float = 1.0,
                                       amp: float = 1.0,
                                       grad_amp: float = 0.5,
                                       mode_x: int = 1, mode_y: int = 1,
                                       omega: float = 6.283185307179586,
                                       time_profile: str = "cos") -> str:
    """Transient thermo MMS deck for TEMPORAL convergence studies.

    Manufactured solution on [0,lx] x [0,ly] (PROBLEMTYPE "Thermo",
    THERMO QUAD4 + MAT_Fourier, One-Step-Theta):

        u*(x,y,t) = temp_offset
                    + (amp * sin(kx x) * sin(ky y)
                       + grad_amp * x / lx) * f(t)

    with kx = mode_x*pi/lx, ky = mode_y*pi/ly and
    f(t) = cos(omega t) (time_profile "cos") or exp(-omega t)
    (time_profile "exp"). The matching volumetric source

        q = rho*c * du*/dt - kappa * lap(u*)

    is emitted as FUNCT2 and applied through a plain
    "DESIGN SURF NEUMANN CONDITIONS" on a DSURF covering the whole
    domain — in a 2D discretization that condition's geometry is the
    QUAD4 elements themselves, so 4C's thermo body-load path
    (TemperImpl::radiation, which looks up the PLAIN "SurfaceNeumann"
    condition name) integrates it as a volumetric heat source. The
    THERMO-prefixed Neumann sections are registered under
    "ThermoSurfaceNeumann" and are SILENTLY IGNORED in standalone
    Thermo (verified empirically against a built 4C binary).

    Time-dependent Dirichlet u* on the whole boundary (VAL 1.0 scaled
    by FUNCT 1 = exact solution) and INITIALFIELD field_by_function
    from the same function (evaluated at t=0), so the only model error
    is discretization error. On a fixed mesh, halving dt grades the
    temporal order of One-Step-Theta: ~2 for theta=0.5
    (Crank-Nicolson; the 4C OST residual theta-weights the external
    force at t_n / t_{n+1}, i.e. trapezoid quadrature, and consistent
    initial rates are computed), ~1 for theta=1 (backward Euler).
    Note the error vs the exact solution saturates at the spatial Q1
    floor, so an error-vs-exact dt table flattens at small dt no matter
    how good the time integrator is. Grade the order from Richardson
    differences of consecutive-dt solutions on the SAME mesh: the
    spatial error is identical in both and cancels in the difference,
    leaving the temporal order alone (exercised live 2026-08-01).

    Final-time nodal temperatures land in
    <prefix>-vtk-files/thermo-<step>-0.vtu (point_data key
    "temperature", readable with meshio); INTERVAL_STEPS is set to
    NUMSTEP so step 0 and the final step are written.
    """
    import math

    n = max(2, min(int(n), 96))
    dt = float(dt)
    numstep = max(1, round(float(t_end) / dt))
    maxtime = numstep * dt

    kx = mode_x * math.pi / lx
    ky = mode_y * math.pi / ly
    rhoc = rho * c
    lap_coeff = kappa * amp * (kx * kx + ky * ky)

    spatial = (f"({amp:.16g}*sin({kx:.16g}*x)*sin({ky:.16g}*y)"
               f" + {grad_amp / lx:.16g}*x)")
    if time_profile == "exp":
        f_t = f"exp(-{omega:.16g}*t)"
        fp_t = f"(-{omega:.16g}*exp(-{omega:.16g}*t))"
    else:  # "cos"
        f_t = f"cos({omega:.16g}*t)"
        fp_t = f"(-{omega:.16g}*sin({omega:.16g}*t))"

    u_exact = f"{temp_offset:.16g} + {spatial}*{f_t}"
    source = (f"{rhoc:.16g}*{spatial}*{fp_t}"
              f" + {lap_coeff:.16g}*sin({kx:.16g}*x)*sin({ky:.16g}*y)"
              f"*{f_t}")

    mesh = generate_quad4_rectangle(n, n, lx, ly,
                                    element_section="THERMO",
                                    element_type="THERMO QUAD4",
                                    element_suffix="MAT 1")
    boundary = sorted(set(mesh["left_nodes"]) | set(mesh["right_nodes"])
                      | set(mesh["bottom_nodes"]) | set(mesh["top_nodes"]))

    yaml = f'''TITLE:
  - "Transient thermo MMS — temporal convergence (One-Step-Theta)"
  - "u* = {temp_offset:.6g} + ({amp:.6g}*sin({kx:.6g}x)*sin({ky:.6g}y) + {grad_amp / lx:.6g}x)*f(t), f = {time_profile}(omega t), omega = {omega:.6g}"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
THERMAL DYNAMIC:
  DYNAMICTYPE: OneStepTheta
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
  TIMESTEP: {dt:.16g}
  NUMSTEP: {numstep}
  MAXTIME: {maxtime:.16g}
  RESULTSEVERY: {numstep}
  RESTARTEVERY: 0
  LINEAR_SOLVER: 1
THERMAL DYNAMIC/ONESTEPTHETA:
  THETA: {theta:.16g}
THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_THERMO: true
  TEMPERATURE: true
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: {numstep}
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: {rhoc:.16g}
      CONDUCT:
        constant: [{kappa:.16g}]
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{u_exact}"
FUNCT2:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{source}"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [2]
DLINE-NODE TOPOLOGY:
'''
    for nid in boundary:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'
    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'THERMO ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    return yaml


def matched_lubrication_slider_bearing_input(nx: int = 16, ny: int = 1) -> str:
    """2-D slider bearing (Reynolds equation) — self-contained inline mesh.

    Ports the authoritative 4C corpus case
    tests/input_files/lubrication_sb_2d.4C.yaml into an inline-mesh
    generator so the catalog row lubrication/slider_bearing_2d runs
    without the external Exodus file the placeholder generator
    template emitted (which aborted 4C's MatchTree, probe 2026-06-12).

    PURE_LUB Reynolds solve on a thin converging film: lower surface
    moves at constant velocity (FUNCT2 VELOCITYFIELD), the linear
    film height h(x) is prescribed (FUNCT3 HEIGHTFEILD), and the
    pressure is pinned to 0 (Dirichlet) at the inlet (x=-15) and
    outlet (x=+15) line. The domain spans x in [-15,15] so the
    corpus height/velocity expressions stay physical; LUBRICATION
    QUAD4 elements (2-D surface elements in the film plane) carry
    MAT_lubrication with MAT_lubrication_law_constant viscosity.
    Mesh is a single strip (default 16x1) so the run is < 30 s.

    NB: the corpus uses SOLVER "Superlu"; this build's Superlu
    segfaults at the first Reynolds linear solve (the corpus file
    itself crashes with it), so we pin SOLVER 3 to UMFPACK — the
    same direct solver every other working inline input uses here.
    Verified live: rc=0 in ~0.5 s (2026-06-12).
    """
    # x in [-15, 15], thin strip in y. The corpus uses a single row of
    # 8 LUBRICATION QUAD4 elements over this span; we keep that layout
    # (nx columns, ny rows) but emit a clean structured grid.
    nx = max(2, int(nx))
    ny = max(1, int(ny))
    lx = 30.0
    ly = 5.0
    x0 = -15.0
    y0 = -2.5
    mesh = generate_quad4_rectangle(
        nx, ny, lx=lx, ly=ly,
        element_section="LUBRICATION",
        element_type="LUBRICATION QUAD4",
        element_suffix="MAT 3",
    )
    # Shift the generated [0,lx]x[0,ly] grid into [-15,15]x[-2.5,2.5]
    # so the corpus FUNCTs (which reference x) stay valid.
    shifted_nodes = []
    for nd in mesh["nodes"]:
        # "NODE id COORD x y z"
        parts = nd.split()
        nid = parts[1]
        x = float(parts[3]) + x0
        y = float(parts[4]) + y0
        shifted_nodes.append(f'NODE {nid} COORD {x:.10e} {y:.10e} 0.0000000000e+00')

    # Pressure Dirichlet line = inlet (left, x=-15) + outlet (right, x=+15)
    dirich_nodes = sorted(set(mesh["left_nodes"]) | set(mesh["right_nodes"]))

    yaml = '''TITLE:
  - "2-D slider bearing (Reynolds equation) — inline-mesh benchmark"
  - "Ported from 4C corpus lubrication_sb_2d.4C.yaml"
PROBLEM TYPE:
  PROBLEMTYPE: "Lubrication"
DISCRETISATION:
  NUMSTRUCDIS: 0
  NUMALEDIS: 0
  NUMTHERMDIS: 0
IO:
  STRUCT_DISP: false
LUBRICATION DYNAMIC:
  MAXTIME: 1
  NUMSTEP: 5
  TIMESTEP: 1e-05
  CALCERRORNO: 0
  VELOCITYFIELD: "function"
  VELFUNCNO: 2
  HEIGHTFEILD: "function"
  HFUNCNO: 3
  LINEAR_SOLVER: 3
  CONVTOL: 1e-06
  PENALTY_CAVITATION: 1e+08
  ROUGHNESS_STD_DEVIATION: 0.001
  PURE_LUB: true
SOLVER 3:
  SOLVER: "UMFPACK"
  NAME: "Direct_Solver"
MATERIALS:
  - MAT: 3
    MAT_lubrication:
      LUBRICATIONLAWID: 30
      DENSITY: 1
  - MAT: 30
    MAT_lubrication_law_constant:
      VISCOSITY: 5e-07
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
FUNCT2:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "20000"
  - COMPONENT: 1
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
  - COMPONENT: 2
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
FUNCT3:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "(0.045)-(x*0.5e-3)"
  - COMPONENT: 1
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
  - COMPONENT: 2
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0]
    FUNCT: [0]
'''

    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in dirich_nodes:
        yaml += f'  - "NODE {nid} DLINE 1"\n'

    yaml += 'NODE COORDS:\n'
    for nd in shifted_nodes:
        yaml += f'  - "{nd}"\n'

    yaml += 'LUBRICATION ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    return yaml


def matched_mixture_3d_input(n: int = 4,
                             E: float = 1000.0,
                             nu: float = 0.3,
                             density: float = 0.1,
                             load: float = 5.0) -> str:
    """3D unit-cube under tension with a MAT_Mixture material.

    Routes the catalog's mixture/mixture_3d row (previously a one-line
    comment template that failed validate_input — "Input is not a YAML
    dictionary", probe 2026-06-12). Builds a self-contained inline HEX8
    cube whose single material is the 4C Mixture toolbox:

        MAT_Mixture
          -> MIX_Rule_Simple (one constituent, MASSFRAC [1.0])
          -> MIX_Constituent_ElastHyper
          -> ELAST_CoupLogNeoHooke (MODE "YN": C1=E, C2=nu)

    The block mirrors mixture_elast_hyper.4C.yaml / mixture_solid_material
    .4C.yaml in the 4C corpus but with a single isotropic NeoHooke
    constituent so there is no fibre direction to set (the mass
    fractions sum to 1, as the mixture rule requires). KINEM nonlinear
    finite-strain SOLID HEX8 elements, Statics, UMFPACK.

    Left face (x=0) fully fixed, right face (x=n's lx) pulled in +x by a
    surface Neumann traction `load`. The traction scales modestly with
    the probe's E=1000 so the single static Newton step converges.
    Resolution keyed off `n` (NOT nx/ny/nz) so the probe's nz=16 cannot
    inflate the cube; capped small for a < 40 s run.
    """
    mesh = generate_hex8_cube(n, n, n,
                              element_section="STRUCTURE",
                              element_type="SOLID HEX8",
                              element_suffix="MAT 1 KINEM nonlinear")

    grid = mesh["node_grid"]
    left_face = sorted(nid for (i, j, k), nid in grid.items() if i == 0)
    right_face = sorted(nid for (i, j, k), nid in grid.items() if i == n)

    yaml = f'''TITLE:
  - "3D cube tension — MAT_Mixture (single NeoHooke constituent)"
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
PROBLEM SIZE:
  DIM: 3
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-8
  TOLRES: 1e-8
  MAXITER: 25
  LINEAR_SOLVER: 1
  PREDICT: TangDis
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_Mixture:
      MATIDMIXTURERULE: 10
      MATIDSCONST: [11]
  - MAT: 10
    MIX_Rule_Simple:
      DENS: {density}
      MASSFRAC:
        constant: [1.0]
  - MAT: 11
    MIX_Constituent_ElastHyper:
      NUMMAT: 1
      MATIDS: [101]
  - MAT: 101
    ELAST_CoupLogNeoHooke:
      MODE: "YN"
      C1: {E}
      C2: {nu}
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 6
    ONOFF: [1, 0, 0, 0, 0, 0]
    VAL: [{load}, 0.0, 0.0, 0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0, 0, 0, 0]
'''

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in left_face:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'
    for nid in right_face:
        yaml += f'  - "NODE {nid} DSURFACE 2"\n'

    return yaml


def matched_constraint_3d_input(n: int = 4,
                                E: float = 1000.0,
                                nu: float = 0.3,
                                load: float = 5.0) -> str:
    """3D cube with a multi-point coupling constraint (DESIGN POINT
    COUPLING CONDITIONS) tying a set of nodes to move identically.

    Routes the catalog's constraint/constraint_3d row (previously a
    one-line comment template that failed validate_input — "Input is not
    a YAML dictionary", probe 2026-06-12). Builds a self-contained
    inline HEX8 cube (St. Venant-Kirchhoff, Statics, UMFPACK) and adds a
    real constraint: a DESIGN POINT COUPLING CONDITION that gathers the
    free-face (x=lx) nodes into one design point set and couples their
    y- and z-displacements so the loaded face translates rigidly in the
    transverse directions — a linear multi-point constraint enforced
    directly by 4C (mirrors sohex8_distributed-pointcoupling.4C.yaml in
    the corpus, the DESIGN POINT COUPLING CONDITIONS / DNODE-NODE
    pattern, on a structured cube).

    Left face (x=0) fully fixed, right face (x=lx) pulled in +x by a
    surface Neumann traction `load` scaling modestly with E so the
    single static Newton step converges at the probe's E=1000.
    Resolution keyed off `n` (NOT nx/ny/nz) so the probe's nz=16 cannot
    inflate the cube; capped small for a < 40 s run.
    """
    mesh = generate_hex8_cube(n, n, n,
                              element_section="STRUCTURE",
                              element_type="SOLID HEX8",
                              element_suffix="MAT 1 KINEM nonlinear")

    grid = mesh["node_grid"]
    left_face = sorted(nid for (i, j, k), nid in grid.items() if i == 0)
    right_face = sorted(nid for (i, j, k), nid in grid.items() if i == n)

    yaml = f'''TITLE:
  - "3D cube — multi-point coupling constraint on loaded face"
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
PROBLEM SIZE:
  DIM: 3
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-8
  TOLRES: 1e-8
  MAXITER: 25
  LINEAR_SOLVER: 1
  PREDICT: TangDis
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E}
      NUE: {nu}
      DENS: 0.0
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 6
    ONOFF: [1, 0, 0, 0, 0, 0]
    VAL: [{load}, 0.0, 0.0, 0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0, 0, 0, 0]
DESIGN POINT COUPLING CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [0, 1, 1]
'''

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    # Couple the loaded-face nodes into one design point set: their
    # y/z displacements are tied together (the MPC).
    yaml += 'DNODE-NODE TOPOLOGY:\n'
    for nid in right_face:
        yaml += f'  - "NODE {nid} DNODE 1"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in left_face:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'
    for nid in right_face:
        yaml += f'  - "NODE {nid} DSURFACE 2"\n'

    return yaml


def matched_membrane_2d_input(nx: int = 8, ny: int = 8,
                              lx: float = 1.0, ly: float = 1.0,
                              E: float = 1000.0, nu: float = 0.3,
                              thick: float = 0.01,
                              stretch: float = 0.1) -> str:
    """Flat MEMBRANE4 QUAD4 patch under a prescribed uniaxial stretch.

    Self-contained inline-mesh structural membrane (PROBLEMTYPE
    "Structure"), modelled on tests/input_files/membrane_patch_new_struct
    .4C.yaml. Membranes carry NO bending stiffness, so an unconstrained
    membrane under a traction load has a singular stiffness matrix
    (zero-eigenvalue out-of-plane modes - the membrane.py pitfalls call
    this out). The corpus patch avoids this by prescribing the FULL
    displacement field of every node via Dirichlet conditions: a uniaxial
    stretch in x (u_x = stretch * x) plus zero y/z. That makes the solve
    unconditionally well-posed (no free DOFs left to buckle), so it
    converges at any E. The membrane material is MAT_Membrane_ElastHyper
    wrapping an ELAST_IsoNeoHooke (MUE = E / (2*(1+nu))), element token
    "MEMBRANE4 QUAD4 ... KINEM nonlinear THICK <t> STRESS_STRAIN
    plane_stress" copied from the corpus element line.

    Parameters are capped (nx,ny <= 16) so a generic sweep cannot inflate
    the mesh; runtime is well under 30 s.
    """
    nx = max(1, min(int(nx), 16))
    ny = max(1, min(int(ny), 16))
    mue = E / (2.0 * (1.0 + nu))

    mesh = generate_quad4_rectangle(
        nx, ny, lx=lx, ly=ly,
        element_section="STRUCTURE",
        element_type="MEMBRANE4 QUAD4",
        element_suffix=f"MAT 1 KINEM nonlinear THICK {thick} "
                       f"STRESS_STRAIN plane_stress")

    maxtime = 1.0
    # NB: no "PROBLEM SIZE: DIM:" section — MEMBRANE4 elements are
    # surface elements embedded in 3D ambient space (the corpus
    # membrane_patch_new_struct.4C.yaml omits PROBLEM SIZE too).
    # Forcing DIM: 2 makes 4C SIGFPE in initialize_elements().
    yaml = f"""TITLE:
  - "Membrane patch {lx}x{ly} - prescribed uniaxial stretch (inline mesh)"
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
IO:
  STRUCT_STRESS: "Cauchy"
  STRUCT_STRAIN: "GL"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "OneStepTheta"
  TIMESTEP: 0.1
  NUMSTEP: 10
  MAXTIME: {maxtime}
  TOLDISP: 1e-9
  TOLRES: 1e-9
  MAXITER: 25
  LOADLIN: true
  LINEAR_SOLVER: 1
STRUCTURAL DYNAMIC/ONESTEPTHETA:
  THETA: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_Membrane_ElastHyper:
      NUMMAT: 1
      MATIDS: [2]
      DENS: 1
  - MAT: 2
    ELAST_IsoNeoHooke:
      MUE:
        constant: {mue}
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "a*x"
  - COMPONENT: 1
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
  - COMPONENT: 2
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
  - VARIABLE: 0
    NAME: "a"
    TYPE: "multifunction"
    NUMPOINTS: 2
    TIMES: [0, {maxtime}]
    DESCRIPTION: ["{stretch}*t"]
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [1, 1, 1]
    FUNCT: [1, 1, 1]
"""

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    return yaml


def matched_shell_3d_input(nx: int = 8, ny: int = 4,
                           lx: float = 1.0, ly: float = 0.5,
                           E: float = 1000.0, nu: float = 0.3,
                           thick: float = 0.05,
                           load: "float | None" = None) -> str:
    """Flat SHELL7P QUAD4 clamped cantilever under transverse pressure.

    Self-contained inline-mesh structural shell (PROBLEMTYPE
    "Structure"). The SHELL7P element + material/BC pattern is copied
    from tests/input_files/shell7p_spring_dashpot.4C.yaml: element token
    "SHELL7P QUAD4 ... MAT 1 THICK <t> EAS N_7 N_7 N_4 N_4 N_4 SDC 1.0
    USE_ANS true", material MAT_ElastHyper wrapping ELAST_CoupNeoHooke,
    and a DESIGN SURF NEUMANN orthopressure load. The shell lies in the
    z=0 plane (a flat QUAD4 patch); the left edge (x=0) is fully clamped
    (all 6 shell DOFs) and a transverse orthopressure is applied to the
    surface, producing a bending cantilever.

    SHELL7P carries bending stiffness, so unlike the membrane it is
    well-posed under a traction load once one edge is clamped. The load
    is scaled with E (load = E*5e-5 by default) so the deflection stays
    moderate and Newton converges at the probe E=1000. nx,ny are capped
    (<=16) for a sub-30 s runtime.
    """
    nx = max(1, min(int(nx), 16))
    ny = max(1, min(int(ny), 16))
    if load is None:
        load = E * 5.0e-5

    mesh = generate_quad4_rectangle(
        nx, ny, lx=lx, ly=ly,
        element_section="STRUCTURE",
        element_type="SHELL7P QUAD4",
        element_suffix=f"MAT 1 THICK {thick} EAS N_7 N_7 N_4 N_4 N_4 "
                       f"SDC 1.0 USE_ANS true")

    left_nodes = mesh["left_nodes"]

    yaml = f"""TITLE:
  - "Shell7p clamped cantilever {lx}x{ly} - transverse pressure (inline mesh)"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
IO:
  STRUCT_STRESS: "Cauchy"
  STRUCT_STRAIN: "GL"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 10
  MAXTIME: 1.0
  TOLDISP: 1e-9
  TOLRES: 1e-9
  MAXITER: 25
  LOADLIN: true
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_ElastHyper:
      NUMMAT: 1
      MATIDS: [2]
      DENS: 0
  - MAT: 2
    ELAST_CoupNeoHooke:
      YOUNG: {E}
      NUE: {nu}
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 6
    ONOFF: [1, 1, 1, 1, 1, 1]
    VAL: [0, 0, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 6
    ONOFF: [0, 0, 1, 0, 0, 0]
    VAL: [0, 0, {load}, 0, 0, 0]
    FUNCT: [0, 0, 1, 0, 0, 0]
    TYPE: "orthopressure"
"""

    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in left_nodes:
        yaml += f'  - "NODE {nid} DLINE 1"\n'

    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in mesh["all_nodes"]:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    return yaml


def matched_cardiovascular0d_windkessel_input(
    n: int = 2,
    E: float = 10.0, nu: float = 0.3, density: float = 2e-6,
    C: float = 1.5, R_p: float = 5.0, Z_c: float = 0.0,
    L: float = 0.0, p_ref: float = 0.0, p_0: float = 10.0,
    pressure: float = 1.0, numstep: int = 3, timestep: float = 0.1,
) -> str:
    """0-D cardiovascular 4-element Windkessel coupled to a 3-D solid.

    Ports tests/input_files/cardiovascular0d_4elementwindkessel_structure_direct_stat.4C.yaml:
    a structural HEX8 cube (StVenantKirchhoff) whose x=lx face is closed off
    against a lumped-parameter 4-element Windkessel model (C, R_p, Z_c, L)
    via DESIGN SURF CARDIOVASCULAR 0D 4-ELEMENT WINDKESSEL + 0D-STRUCTURE
    COUPLING conditions. The cavity volume seen by the 0D model is integrated
    over that surface (DESIGN SURFACE VOLUME MONITOR 3D). The x=0 face is
    clamped; a small orthopressure (FUNCT 15*t) on the coupled face drives a
    cavity-volume change so the monolithic 0D-3D system is genuinely exercised.

    Self-contained: inline NODE COORDS, no external mesh and no external NOX
    "Status Test" XML - omitting STRUCT NOX/Status Test makes 4C build the
    convergence test from STRUCTURAL DYNAMIC TOLDISP/TOLRES (including the
    Cardiovascular0D quantity), see structure_new_nln_solver_factory.cpp.
    Statics integrator (the simplest of the corpus _stat/_ost/_genalpha set).
    Resolution is the single param "n" (NOT nx/ny/nz) so a parameter sweep
    cannot inflate the monolithic 0D-3D solve.
    """
    n = max(1, min(int(n), 4))
    mesh = generate_hex8_cube(
        n, n, n, lx=10.0, ly=10.0, lz=10.0,
        element_section="STRUCTURE",
        element_type="SOLID HEX8",
        element_suffix="MAT 1 KINEM nonlinear")
    ng = mesh["node_grid"]
    # x=0 face: clamped Dirichlet.   x=lx face: 0D-windkessel-coupled cavity.
    fixed_face = sorted({ng[(0, j, k)] for j in range(n + 1)
                         for k in range(n + 1)})
    coupled_face = sorted({ng[(n, j, k)] for j in range(n + 1)
                           for k in range(n + 1)})

    maxtime = numstep * timestep

    yaml = f'''TITLE:
  - "0D 4-element Windkessel coupled to a 3D solid - inline mesh benchmark"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
IO:
  STRUCT_STRESS: "Cauchy"
  STRUCT_STRAIN: "GL"
SOLVER 1:
  SOLVER: "UMFPACK"
CARDIOVASCULAR 0D-STRUCTURE COUPLING:
  TOL_CARDVASC0D_RES: 1e-06
  TIMINT_THETA: 1
  LINEAR_COUPLED_SOLVER: 1
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  PRESTRESSTIME: {maxtime + timestep + 0.001}
  TIMESTEP: {timestep}
  NUMSTEP: {numstep}
  MAXTIME: {maxtime}
  M_DAMP: 0
  K_DAMP: 0.0001
  TOLDISP: 1e-07
  TOLRES: 1e-05
  MAXITER: 500
  LOADLIN: true
  LINEAR_SOLVER: 1
# Runtime VTK output for the structure: without it 4C writes only its native binary result
# files, which neither a model nor openPASO's own result gate can open (found by the coverage
# run of 2026-09-24; the 0D windkessel values stay in the native output, the 3D field is here).
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E}
      NUE: {nu}
      DENS: {density}
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{pressure * 15.0}*t"
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 6
    ONOFF: [1, 0, 0, 0, 0, 0]
    VAL: [-1, 0, 0, 0, 0, 0]
    FUNCT: [1, 0, 0, 0, 0, 0]
    TYPE: "orthopressure"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DESIGN SURFACE VOLUME MONITOR 3D:
  - E: 2
    ConditionID: 1
DESIGN SURF CARDIOVASCULAR 0D 4-ELEMENT WINDKESSEL CONDITIONS:
  - E: 2
    id: 0
    C: {C}
    R_p: {R_p}
    Z_c: {Z_c}
    L: {L}
    p_ref: {p_ref}
    p_0: {p_0}
DESIGN SURF CARDIOVASCULAR 0D-STRUCTURE COUPLING CONDITIONS:
  - E: 2
    coupling_id: 0
'''

    # DSURFACE 1 = clamped face (E:1), DSURFACE 2 = windkessel face (E:2).
    yaml += 'DSURF-NODE TOPOLOGY:\n'
    for nid in fixed_face:
        yaml += f'  - "NODE {nid} DSURFACE 1"\n'
    for nid in coupled_face:
        yaml += f'  - "NODE {nid} DSURFACE 2"\n'

    yaml += 'NODE COORDS:\n'
    for nd in mesh["nodes"]:
        yaml += f'  - "{nd}"\n'
    yaml += 'STRUCTURE ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'

    return yaml


def _fluid_quad4_mesh(nx: int, ny: int, lx: float, ly: float) -> dict:
    """QUAD4 mesh tagged as FLUID elements (MAT 1, NA Euler)."""
    return generate_quad4_rectangle(
        nx, ny, lx, ly,
        element_section="FLUID",
        element_type="FLUID QUAD4",
        element_suffix="MAT 1 NA Euler",
    )


def _fluid_common_sections(nx: int, ny: int, title: str,
                           viscosity: float, density: float,
                           numstep: int, timestep: float) -> str:
    """Shared PROBLEM/SOLVER/MATERIALS block for a 2D incompressible
    Navier-Stokes case (PROBLEMTYPE Fluid, Np_Gen_Alpha + UMFPACK).

    Mirrors the corpus case f2_channel20x20_drt_weak.4C.yaml but with
    a self-contained UMFPACK solver (no external XML) and strong
    Dirichlet conditions instead of weak ones.
    """
    maxtime = numstep * timestep
    return f'''TITLE:
  - "{title}"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Fluid"
FLUID DYNAMIC:
  PHYSICAL_TYPE: "Incompressible"
  LINEAR_SOLVER: 1
  TIMEINTEGR: "Np_Gen_Alpha"
  NONLINITER: Newton
  NUMSTEP: {numstep}
  TIMESTEP: {timestep:.10g}
  MAXTIME: {maxtime:.10g}
  ITEMAX: 10
  ALPHA_M: 0.8333
  ALPHA_F: 0.6666
  GAMMA: 0.6666
FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
  STABTYPE: "residual_based"
  CHARELELENGTH_PC: "root_of_volume"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Fluid_Solver"
MATERIALS:
  - MAT: 1
    MAT_fluid:
      DYNVISCOSITY: {viscosity:.10g}
      DENSITY: {density:.10g}
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/FLUID:
  OUTPUT_FLUID: true
  VELOCITY: true
'''


def matched_fluid_cavity_input(
    nx: int = 16, ny: int = 16,
    lx: float = 1.0, ly: float = 1.0,
    u_lid: float = 1.0,
    viscosity: float = 0.01, density: float = 1.0,
    numstep: int = 10, timestep: float = 0.1,
) -> str:
    """2D lid-driven cavity, incompressible Navier-Stokes.

    Square cavity [0,lx]x[0,ly]: no-slip on the bottom, left and right
    walls; the top lid moves with a prescribed tangential velocity
    u_lid (vy=0 on the lid). Self-contained inline FLUID QUAD4 mesh,
    MAT_fluid, Np_Gen_Alpha time integration, UMFPACK solver. No
    external mesh file, no placeholder scalars.

    DLINE topology: 1 bottom (no-slip), 2 top (lid), 3 left (no-slip),
    4 right (no-slip). The lid line excludes its two corner nodes so the
    moving-lid and no-slip wall Dirichlet values never disagree on a
    shared corner.
    """
    nx = max(2, int(nx))
    ny = max(2, int(ny))
    mesh = _fluid_quad4_mesh(nx, ny, lx, ly)
    pin_node = mesh["bottom_nodes"][0]  # bottom-left corner
    title = f"Lid-driven cavity {nx}x{ny} — self-contained inline mesh"
    yaml = _fluid_common_sections(
        nx, ny, title, viscosity, density, numstep, timestep)
    # NUMDOF 3 = (vx, vy, p) in 2D fluid. The cavity is fully enclosed by
    # velocity Dirichlet conditions, so the incompressible pressure is
    # determined only up to a constant (singular pressure null space ->
    # divide-by-zero in the solver). Pin the pressure at one corner node
    # (DNODE 1, ONOFF only on the 3rd = pressure dof) to remove it.
    yaml += f'''DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
  - E: 3
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
  - E: 4
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
  - E: 2
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [{u_lid:.10g}, 0, 0]
    FUNCT: [0, 0, 0]
'''
    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'FLUID ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'
    yaml += 'DNODE-NODE TOPOLOGY:\n'
    yaml += f'  - "NODE {pin_node} DNODE 1"\n'
    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["bottom_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    # Lid: interior nodes only (corners belong to side walls).
    for nid in mesh["top_nodes"][1:-1]:
        yaml += f'  - "NODE {nid} DLINE 2"\n'
    for nid in mesh["left_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 3"\n'
    for nid in mesh["right_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 4"\n'
    return yaml


def matched_fluid_channel_input(
    nx: int = 24, ny: int = 8,
    lx: float = 3.0, ly: float = 1.0,
    u_max: float = 1.0,
    viscosity: float = 0.01, density: float = 1.0,
    numstep: int = 10, timestep: float = 0.1,
) -> str:
    """2D plane channel flow, incompressible Navier-Stokes.

    Channel [0,lx]x[0,ly]: parabolic inlet velocity (peak u_max) at
    x=0, no-slip on the top and bottom walls, natural (do-nothing)
    outflow at x=lx. Self-contained inline FLUID QUAD4 mesh, MAT_fluid,
    Np_Gen_Alpha + UMFPACK. No external mesh, no placeholder scalars.

    DLINE topology: 1 bottom wall, 2 top wall, 3 inlet (x=0). The inlet
    parabola vanishes at both walls so it is consistent with no-slip on
    the shared corner nodes; the outlet (x=lx) carries no Dirichlet
    condition (natural outflow).
    """
    nx = max(2, int(nx))
    ny = max(2, int(ny))
    mesh = _fluid_quad4_mesh(nx, ny, lx, ly)
    profile = f"{4.0 * u_max / (ly * ly):.10g}*y*({ly:.10g}-y)"
    title = f"Plane channel flow {nx}x{ny} — self-contained inline mesh"
    yaml = _fluid_common_sections(
        nx, ny, title, viscosity, density, numstep, timestep)
    yaml += f'''FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{profile}"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
  - E: 2
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
  - E: 3
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [1, 0, 0]
    FUNCT: [1, 0, 0]
'''
    yaml += 'NODE COORDS:\n'
    for n in mesh["nodes"]:
        yaml += f'  - "{n}"\n'
    yaml += 'FLUID ELEMENTS:\n'
    for e in mesh["elements"]:
        yaml += f'  - "{e}"\n'
    yaml += 'DLINE-NODE TOPOLOGY:\n'
    for nid in mesh["bottom_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 1"\n'
    for nid in mesh["top_nodes"]:
        yaml += f'  - "NODE {nid} DLINE 2"\n'
    # Inlet: interior nodes only (corners belong to the walls so the
    # parabola-vs-no-slip values agree at y=0 and y=ly anyway, but
    # excluding them keeps the wall no-slip authoritative).
    for nid in mesh["left_nodes"][1:-1]:
        yaml += f'  - "NODE {nid} DLINE 3"\n'
    return yaml


def matched_reduced_airways_input(
    peak_pressure: float = 30.0,
    numstep: int = 200, period: float = 100.0,
    wall_elasticity: float = 500.0,
    acinus_stiffness: float = 0.001,
) -> str:
    """1-D reduced-dimensional airway tree (PROBLEMTYPE
    ReducedDimensionalAirWays), self-contained inline mesh.

    A 2-generation bifurcating tree: 3 RED_AIRWAY LINE2 segments
    feeding 2 RED_ACINUS LINE2 compartments (6 nodes, 5 elements). The
    inlet pressure is driven sinusoidally from 0 to peak_pressure
    (cmH2O) with the given period; reduced fractions of that pressure
    are prescribed at the two acinar outlets, mapped to the airways via
    the airway-acinus interdependency (COMPAWACINTER).

    Ported from the corpus case red_airway_3airway_2acinus_awacinter
    .4C.yaml; the RESULT DESCRIPTION (baked expected pressures) is
    intentionally NOT included. Solver is self-contained UMFPACK.
    """
    numstep = max(1, int(numstep))
    amp = float(peak_pressure) / 2.0
    return f'''TITLE:
  - "2-generation airway tree (3 airways + 2 acini) — self-contained inline mesh"
PROBLEM SIZE:
  ELEMENTS: 5
  NODES: 6
  MATERIALS: 2
  NUMDF: 1
PROBLEM TYPE:
  PROBLEMTYPE: "ReducedDimensionalAirWays"
REDUCED DIMENSIONAL AIRWAYS DYNAMIC:
  SOLVERTYPE: Nonlinear
  NUMSTEP: {numstep}
  RESTARTEVERY: 100000
  RESULTSEVERY: 100000
  MAXITERATIONS: 40
  TOLERANCE: 1e-07
  LINEAR_SOLVER: 1
  COMPAWACINTER: true
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Reduced_dimensional_Airways_Solver"
MATERIALS:
  - MAT: 1
    MAT_fluid:
      DYNVISCOSITY: 0.04
      DENSITY: 1.176e-06
      GAMMA: 1
  - MAT: 2
    MAT_0D_MAXWELL_ACINUS_EXPONENTIAL:
      Stiffness1: {acinus_stiffness:.10g}
      Stiffness2: {acinus_stiffness:.10g}
      Viscosity1: {acinus_stiffness:.10g}
      Viscosity2: {acinus_stiffness:.10g}
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_TIME: "{amp:.10g}*(sin(pi*t/{period / 2.0:.10g}-pi/2)+1)"
DESIGN NODE Reduced D AIRWAYS PRESCRIBED CONDITIONS:
  - E: 1
    boundarycond: "pressure"
    VAL: [1]
    curve: [1, null]
  - E: 2
    boundarycond: "pressure"
    VAL: [0.1]
    curve: [1, null]
  - E: 3
    boundarycond: "pressure"
    VAL: [0.2]
    curve: [1, null]
REDUCED D AIRWAYS ELEMENTS:
  - "1 RED_AIRWAY LINE2 1 2 MAT 1 ElemSolvingType NonLinear TYPE ConvectiveViscoElasticRLC Resistance Poiseuille PowerOfVelocityProfile 2 WallElasticity {wall_elasticity:.10g} PoissonsRatio 0.4 ViscousTs 2.0 ViscousPhaseShift 0.13 WallThickness 0.1 Area 1.0 Generation 0"
  - "2 RED_AIRWAY LINE2 2 3 MAT 1 ElemSolvingType NonLinear TYPE ConvectiveViscoElasticRLC Resistance Poiseuille PowerOfVelocityProfile 2 WallElasticity {wall_elasticity:.10g} PoissonsRatio 0.4 ViscousTs 2.0 ViscousPhaseShift 0.13 WallThickness 0.1 Area 1.0 Generation 1"
  - "3 RED_AIRWAY LINE2 2 4 MAT 1 ElemSolvingType NonLinear TYPE ConvectiveViscoElasticRLC Resistance Poiseuille PowerOfVelocityProfile 2 WallElasticity {wall_elasticity:.10g} PoissonsRatio 0.4 ViscousTs 2.0 ViscousPhaseShift 0.13 WallThickness 0.1 Area 1.0 Generation 1"
  - "4 RED_ACINUS LINE2 3 5 MAT 2 TYPE Exponential AcinusVolume 1.0 AlveolarDuctVolume 1.0 E1_0 8.0 E1_LIN 1.0 E1_EXP 0.022 TAU 7"
  - "5 RED_ACINUS LINE2 4 6 MAT 2 TYPE Exponential AcinusVolume 1.0 AlveolarDuctVolume 1.0 E1_0 8.0 E1_LIN 1.0 E1_EXP 0.022 TAU 7"
NODE COORDS:
  - "NODE 1 COORD 0.000 0.000 0.000"
  - "NODE 2 COORD 10.00 0.000 0.000"
  - "NODE 3 COORD 15.00 2.000 0.000"
  - "NODE 4 COORD 15.00 -2.000 0.000"
  - "NODE 5 COORD 17.50 2.000 0.000"
  - "NODE 6 COORD 17.00 -2.000 0.000"
DNODE-NODE TOPOLOGY:
  - "NODE 1 DNODE 1"
  - "NODE 5 DNODE 2"
  - "NODE 6 DNODE 3"
'''


def matched_contact_3d_input(nx: int = 2, ny: int = 2, nz: int = 2,
                             gap: float = 0.1, indent: float = 0.3,
                             penalty: float = 1.0e4,
                             E: float = 1000.0, nu: float = 0.3,
                             n_steps: int = 10) -> str:
    """Two stacked HEX8 blocks driven into mortar-penalty contact.

    Self-contained: inline nodes, inline elements, no external mesh.
    Runs to completion on the deployed 4C (rc=0, all load steps
    finalised, contact active set non-empty from the step where the
    prescribed displacement closes ``gap``).

    The contact-specific ingredients — and there are exactly three
    sections — are CONTACT DYNAMIC, MORTAR COUPLING, and one Master +
    one Slave DESIGN SURF MORTAR CONTACT CONDITIONS 3D entry sharing an
    InterfaceID. CONTACT DYNAMIC/LINEAR_SOLVER may reuse the structural
    solver id; the separate SOLVER 2 block below is convention, not a
    requirement.

    ``penalty`` is a physical interface stiffness for STRATEGY Penalty and
    scales with ``E``. If Newton fails, shrink ``n_steps``' step size
    (raise ``n_steps``) rather than lowering ``penalty``: the failure mode
    is active-set chatter, and a smaller ``penalty`` buys convergence by
    permitting more penetration. A deck that exhausts MAXITER at ten load
    steps completes at a hundred, unchanged otherwise.
    """
    nodes, elements = [], []
    nid, eid = 1, 1
    face = {"bot_lo": [], "bot_hi": [], "top_lo": [], "top_hi": []}

    for blk, (z0, z1, lo_key, hi_key) in enumerate(
            [(0.0, 1.0, "bot_lo", "bot_hi"),
             (1.0 + gap, 2.0 + gap, "top_lo", "top_hi")]):
        grid = {}
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    x = i / nx
                    y = j / ny
                    z = z0 + k * (z1 - z0) / nz
                    nodes.append(f'NODE {nid} COORD {x:.6f} {y:.6f} {z:.6f}')
                    grid[(i, j, k)] = nid
                    if k == 0:
                        face[lo_key].append(nid)
                    if k == nz:
                        face[hi_key].append(nid)
                    nid += 1
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    c = [grid[(i, j, k)], grid[(i + 1, j, k)],
                         grid[(i + 1, j + 1, k)], grid[(i, j + 1, k)],
                         grid[(i, j, k + 1)], grid[(i + 1, j, k + 1)],
                         grid[(i + 1, j + 1, k + 1)], grid[(i, j + 1, k + 1)]]
                    ids = " ".join(str(c_) for c_ in c)
                    elements.append(
                        f'{eid} SOLID HEX8 {ids} MAT {blk + 1} KINEM nonlinear')
                    eid += 1

    def topo(ids, dsurf):
        return "\n".join(f'  - "NODE {n} DSURFACE {dsurf}"' for n in ids)

    node_block = "\n".join(f"  - \"{n}\"" for n in nodes)
    ele_block = "\n".join(f"  - \"{e}\"" for e in elements)
    topo_block = "\n".join([
        topo(face["bot_lo"], 1),
        topo(face["bot_hi"], 2),
        topo(face["top_lo"], 3),
        topo(face["top_hi"], 4),
    ])

    return f'''TITLE:
  - "Two blocks in mortar-penalty contact -- inline mesh, self-contained"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: {1.0 / n_steps}
  NUMSTEP: {n_steps}
  MAXTIME: 1.0
  TOLDISP: 1.0e-08
  TOLRES: 1.0e-06
  MAXITER: 50
  LINEAR_SOLVER: 1
CONTACT DYNAMIC:
  LINEAR_SOLVER: 2
  STRATEGY: "Penalty"
  PENALTYPARAM: {penalty}
MORTAR COUPLING:
  LM_DUAL_CONSISTENT: "none"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Contact_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E}
      NUE: {nu}
      DENS: 1.0
  - MAT: 2
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E}
      NUE: {nu}
      DENS: 1.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: 4
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, {-abs(indent)}]
    FUNCT: [0, 0, 1]
DESIGN SURF MORTAR CONTACT CONDITIONS 3D:
  - E: 2
    InterfaceID: 1
    Side: "Master"
  - E: 3
    InterfaceID: 1
    Side: "Slave"
DSURF-NODE TOPOLOGY:
{topo_block}
NODE COORDS:
{node_block}
STRUCTURE ELEMENTS:
{ele_block}
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  OUTPUT_CONTACT: true
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 1
      QUANTITY: "dispz"
      VALUE: 0.0
      TOLERANCE: 1.0e30
'''
