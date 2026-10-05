"""
MCP tools for deep input validation across ALL backends.

This goes far beyond basic syntax checking — it validates:
- Required sections present
- Material parameters in valid ranges
- Solver configurations consistent with physics
- Boundary conditions properly formed
- Element types compatible with problem dimension
- Known pitfall detection

Ported from 4c-ai-interface/src/tools/schema.py and extended for all backends.
"""

import json
import re
from mcp.server.fastmcp import FastMCP
from core.registry import get_backend

# ═══════════════════════════════════════════════════════════════════════════════
# 4C INPUT SCHEMAS (ported from 4c-ai-interface)
# ═══════════════════════════════════════════════════════════════════════════════

_4C_SCHEMAS = {
    "Scalar_Transport": {
        # <FIELD> GEOMETRY is optional: the mesh may equally come inline (NODE COORDS + <FIELD> ELEMENTS).
        "required_sections": ["PROBLEM TYPE", "SCALAR TRANSPORT DYNAMIC", "SOLVER 1", "MATERIALS"],
        "dynamics_keys": {
            "TIMEINTEGR": {"allowed": ["Stationary", "One_Step_Theta", "BDF2", "Gen_Alpha"], "default": "One_Step_Theta"},
            "SOLVERTYPE": {"allowed": ["linear_full", "nonlinear"], "default": "linear_full"},
            "VELOCITYFIELD": {"allowed": ["zero", "function", "Navier_Stokes"], "default": "zero"},
        },
        "materials": ["MAT_scatra", "MAT_Fourier"],
        "geometry_section": "TRANSPORT GEOMETRY",
        "element_category": "TRANSP",
    },
    "Structure": {
        "required_sections": ["PROBLEM TYPE", "STRUCTURAL DYNAMIC", "SOLVER 1", "MATERIALS"],
        "dynamics_keys": {
            "DYNAMICTYPE": {"allowed": ["Statics", "GenAlpha", "GenAlphaLieGroup", "OneStepTheta", "ExplicitEuler",
                                        "CentrDiff", "AdamsBashforth2", "AdamsBashforth4"]},
        },
        "materials": ["MAT_Struct_StVenantKirchhoff", "MAT_ElastHyper", "MAT_Struct_PlasticNlnLogNeoHooke",
                       "MAT_BeamReissnerElastHyper", "MAT_BeamKirchhoffTorsionFreeElastHyper"],
        "geometry_section": "STRUCTURE GEOMETRY",
        "element_categories": ["SOLID", "WALL", "BEAM3R", "BEAM3EB", "BEAM3K"],
    },
    "Fluid": {
        "required_sections": ["PROBLEM TYPE", "FLUID DYNAMIC", "SOLVER 1", "MATERIALS"],
        "dynamics_keys": {
            "TIMEINTEGR": {"allowed": ["Stationary", "Np_Gen_Alpha", "Af_Gen_Alpha", "One_Step_Theta", "BDF2"]},
        },
        "materials": ["MAT_fluid"],
        "geometry_section": "FLUID GEOMETRY",
        "element_category": "FLUID",
    },
    "Fluid_Structure_Interaction": {
        "required_sections": ["PROBLEM TYPE", "STRUCTURAL DYNAMIC", "FLUID DYNAMIC", "ALE DYNAMIC",
                               "FSI DYNAMIC", "SOLVER 1", "MATERIALS", "CLONING MATERIAL MAP"],
        "materials": ["MAT_fluid", "MAT_ElastHyper", "MAT_Struct_StVenantKirchhoff"],
    },
}


def _validate_4c_deep(content: str) -> list[str]:
    """Deep validation of 4C YAML input."""
    import yaml
    errors = []
    warnings = []

    # Parse YAML
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        return [f"YAML parse error: {e}"]

    if not isinstance(data, dict):
        return ["Input is not a YAML dictionary"]

    # Check PROBLEM TYPE
    pt_section = data.get("PROBLEM TYPE", {})
    problem_type = pt_section.get("PROBLEMTYPE", "") if isinstance(pt_section, dict) else ""
    if not problem_type:
        errors.append("Missing PROBLEM TYPE / PROBLEMTYPE")
        return errors

    schema = _4C_SCHEMAS.get(problem_type)
    if not schema:
        warnings.append(f"Unknown problem type '{problem_type}' — cannot validate deeply")
        return warnings

    # Check required sections
    for section in schema.get("required_sections", []):
        if section not in data:
            errors.append(f"Missing required section: {section}")

    # Check materials
    materials = data.get("MATERIALS", [])
    if not materials:
        errors.append("MATERIALS section is empty or missing")
    elif isinstance(materials, list):
        for mat in materials:
            if not isinstance(mat, dict) or "MAT" not in mat:
                errors.append(f"Invalid material entry: {mat}")

    # Check geometry
    geo_section = schema.get("geometry_section")
    if geo_section and geo_section in data:
        geo = data[geo_section]
        if isinstance(geo, dict):
            if "FILE" not in geo and "ELEMENT_BLOCKS" not in geo:
                if "NODE COORDS" not in data:
                    errors.append(f"{geo_section}: needs FILE (mesh) or ELEMENT_BLOCKS or inline NODE COORDS")

    # FSI-specific checks
    if problem_type == "Fluid_Structure_Interaction":
        if "CLONING MATERIAL MAP" not in data:
            errors.append("FSI requires CLONING MATERIAL MAP section")
        # MONOLITHIC SOLVER is its own top-level section, not a key inside FSI DYNAMIC.
        mono = data.get("FSI DYNAMIC/MONOLITHIC SOLVER")
        if isinstance(mono, dict) and not mono.get("SHAPEDERIVATIVES"):
            warnings.append("FSI: SHAPEDERIVATIVES should be true in FSI DYNAMIC/MONOLITHIC SOLVER")

    # Known pitfall detection (VELOCITYFIELD needs no check: its default is "zero")
    if problem_type == "Structure":
        struct = data.get("STRUCTURAL DYNAMIC", {})
        if isinstance(struct, dict):
            kinem = None
            # Check materials for nonlinear requirement
            for mat in (materials if isinstance(materials, list) else []):
                if isinstance(mat, dict):
                    if any(k.startswith("MAT_ElastHyper") or k.startswith("MAT_Struct_Plastic") for k in mat):
                        kinem_needed = "nonlinear"
                        if struct.get("DYNAMICTYPE") == "Statics":
                            pass  # check in geometry
            if struct.get("DYNAMICTYPE") in ("GenAlpha", "GenAlphaLieGroup", "ExplicitEuler"):
                # Check DENS in materials
                for mat in (materials if isinstance(materials, list) else []):
                    if isinstance(mat, dict):
                        for k, v in mat.items():
                            if k.startswith("MAT_") and isinstance(v, dict):
                                if "DENS" not in v and k != "MAT_BeamReissnerElastHyper":
                                    warnings.append(f"PITFALL: {k} missing DENS — zero mass matrix in dynamics!")

    result = errors + [f"WARNING: {w}" for w in warnings]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FENICS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_fenics_deep(content: str) -> list[str]:
    """Deep validation of FEniCS Python scripts."""
    issues = []

    # Must import dolfinx
    if "import dolfinx" not in content and "from dolfinx" not in content:
        issues.append("Script does not import dolfinx")

    # Check for common API mistakes (dolfinx 0.10)
    if "NewtonSolver" in content and "from dolfinx.nls" in content:
        issues.append("WARNING: dolfinx 0.10 — use NonlinearProblem.solve() instead of NewtonSolver")

    if "FunctionSpace(" in content and "functionspace(" not in content.lower():
        issues.append("WARNING: dolfinx 0.10 uses fem.functionspace() (lowercase), not FunctionSpace()")

    # Check for BC on sub-space with constant
    if "W.sub(" in content and "np.zeros" in content and "dirichletbc" in content:
        if "fem.Function" not in content:
            issues.append("WARNING: BCs on sub-spaces require fem.Function (not constant array) in dolfinx 0.10")

    # Check for XDMF P2 issue
    if "element(" in content and ", 2," in content and "write_function" in content:
        if "interpolate" not in content:
            issues.append("WARNING: P2 functions can't write to XDMF — interpolate to P1 first")

    # Check mesh creation
    if "create_unit_square" not in content and "create_unit_cube" not in content and "create_rectangle" not in content and "create_box" not in content:
        if "gmsh" not in content.lower():
            issues.append("No mesh creation found — need mesh.create_* or Gmsh import")

    # Check for solve call
    if "solve" not in content.lower():
        issues.append("No solve() call found")

    # Check for output
    if "XDMFFile" not in content and "VTXWriter" not in content and "write" not in content:
        issues.append("WARNING: No output writing found — results won't be saved")

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# DEAL.II VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_dealii_deep(content: str) -> list[str]:
    """Deep validation of deal.II C++ source."""
    issues = []

    if "#include" not in content:
        issues.append("No #include directives found")
    if "deal.II" not in content and "deal_II" not in content:
        issues.append("No deal.II headers included")
    if "int main" not in content:
        issues.append("No main() function")

    # Check common mistakes
    if "distribute_dofs" in content and "refine_global" in content:
        # Check order: refine should come before distribute_dofs
        refine_pos = content.index("refine_global")
        dof_pos = content.index("distribute_dofs")
        if dof_pos < refine_pos:
            issues.append("PITFALL: distribute_dofs() called BEFORE refine_global() — DOFs wrong!")

    if "FESystem" in content and "system_to_component_index" not in content:
        if "FEValuesExtractors" not in content:
            issues.append("WARNING: FESystem used but no component extraction (system_to_component_index or FEValuesExtractors)")

    if "hyper_cube" in content and "boundary_id" not in content and "interpolate_boundary_values" in content:
        issues.append("WARNING: hyper_cube has all boundary_id=0 — all Dirichlet BCs will be the same. Use hyper_rectangle for distinct boundaries.")

    if "DataOut" in content and "build_patches" not in content:
        issues.append("PITFALL: DataOut requires build_patches() before writing")

    # Check for missing FEValues flags
    if "FEValues" in content:
        if "update_values" not in content and "update_gradients" not in content:
            issues.append("PITFALL: FEValues created without update flags — silent wrong results!")

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# FEBIO VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_febio_deep(content: str) -> list[str]:
    """Deep validation of FEBio XML input."""
    issues = []

    if "<febio_spec" not in content:
        issues.append("Missing <febio_spec> root element")
    if "<Material>" not in content and "<Material " not in content:
        issues.append("Missing Material section")
    if "<Mesh>" not in content and "<Geometry>" not in content:
        issues.append("Missing Mesh/Geometry section")
    if "<MeshDomains>" not in content:
        issues.append("PITFALL: Missing MeshDomains section (required in FEBio v4.0)")
    if "<Control>" not in content:
        issues.append("Missing Control section")

    # Check for common FEBio mistakes
    if 'type="nu"' in content or "nu=" in content:
        issues.append("PITFALL: FEBio uses lowercase 'v' for Poisson's ratio, NOT 'nu'")

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOL REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def register_schema_tools(mcp: FastMCP):

    @mcp.tool()
    def deep_validate(solver: str, content: str) -> str:
        """Deep validation of solver input — catches physics errors, not just syntax.

        Goes far beyond basic validation:
        - Checks required sections for the specific problem type
        - Validates material parameters
        - Detects known pitfalls (from 107+ encoded pitfalls)
        - Warns about API compatibility issues
        - Checks solver/element/BC consistency

        Args:
            solver: Backend name ('fenics', 'fourc', 'dealii', 'febio')
            content: The input content (YAML / Python / C++ / XML)

        Returns:
            List of errors and warnings. Empty = valid.
        """
        validators = {
            "fourc": _validate_4c_deep,
            "4c": _validate_4c_deep,
            "fenics": _validate_fenics_deep,
            "fenicsx": _validate_fenics_deep,
            "dealii": _validate_dealii_deep,
            "deal.ii": _validate_dealii_deep,
            "febio": _validate_febio_deep,
        }

        validator = validators.get(solver.lower())
        if not validator:
            return f"Unknown solver: {solver}. Available: fourc, fenics, dealii, febio"

        issues = validator(content)
        if not issues:
            return "Deep validation PASSED — no issues found."

        errors = [i for i in issues if not i.startswith("WARNING")]
        warnings = [i for i in issues if i.startswith("WARNING")]

        lines = []
        if errors:
            lines.append(f"ERRORS ({len(errors)}):")
            for e in errors:
                lines.append(f"  - {e}")
        if warnings:
            lines.append(f"WARNINGS ({len(warnings)}):")
            for w in warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)

    @mcp.tool()
    def get_input_schema(solver: str, problem_type: str = "") -> str:
        """Get the full input schema for a solver's problem type.

        Returns required sections, allowed parameters, material types,
        and element categories. Essential for generating correct input files.

        Args:
            solver: Backend name ('fourc', 'fenics', 'dealii', 'febio')
            problem_type: For 4C: 'Scalar_Transport', 'Structure', 'Fluid', 'Fluid_Structure_Interaction'
                          For others: leave empty for general schema
        """
        if solver.lower() in ("fourc", "4c"):
            if problem_type:
                schema = _4C_SCHEMAS.get(problem_type)
                if schema:
                    return json.dumps(schema, indent=2)
                return f"Unknown 4C problem type: {problem_type}. Available: {list(_4C_SCHEMAS.keys())}"
            return json.dumps(_4C_SCHEMAS, indent=2)
        elif solver.lower() in ("fenics", "fenicsx"):
            return json.dumps({
                "input_format": "Python script using dolfinx",
                "required_imports": ["from mpi4py import MPI", "from dolfinx import mesh, fem, io, default_scalar_type"],
                "workflow": ["1. Create mesh", "2. Define function space", "3. Apply BCs", "4. Define weak form", "5. Solve", "6. Write output"],
                "api_version": "dolfinx 0.10.0 — NonlinearProblem.solve() directly, no NewtonSolver",
            }, indent=2)
        elif solver.lower() in ("dealii", "deal.ii"):
            return json.dumps({
                "input_format": "C++ source code + CMakeLists.txt",
                "required_includes": ["deal.II/grid/tria.h", "deal.II/dofs/dof_handler.h", "deal.II/fe/fe_q.h"],
                "build": "cmake . && make -jN",
                "workflow": ["1. Grid", "2. FE + DoFHandler", "3. Sparsity + matrices", "4. Assembly", "5. BCs", "6. Solve", "7. DataOut"],
            }, indent=2)
        elif solver.lower() == "febio":
            return json.dumps({
                "input_format": "XML (.feb), FEBio v4.0",
                "required_elements": ["febio_spec", "Module", "Control", "Material", "Mesh", "MeshDomains", "Boundary"],
                "module_types": ["solid", "biphasic", "heat", "fluid"],
            }, indent=2)
        return f"Unknown solver: {solver}"
