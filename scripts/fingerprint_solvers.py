#!/usr/bin/env python3
"""Solver API fingerprinting — detect when backend APIs change.

Probes the actual API surface of each installed solver:
- Import checks (modules, classes, functions)
- Version strings
- Key function signatures

Saves fingerprints to data/fingerprints/<solver>.json.
Compare against previous fingerprints to detect drift.

Usage:
    python scripts/fingerprint_solvers.py              # Generate all
    python scripts/fingerprint_solvers.py --compare     # Compare against saved
    python scripts/fingerprint_solvers.py --solver fenics  # Single solver
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


FINGERPRINT_DIR = Path(__file__).parent.parent / "data" / "fingerprints"


def _check_import(module: str) -> dict[str, Any]:
    """Check if a module is importable and get its version."""
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", "unknown"))
        return {"importable": True, "version": str(version)}
    except ImportError:
        return {"importable": False, "version": None}


def _check_attr(module: str, attr: str) -> bool:
    """Check if a module has an attribute (class, function, constant)."""
    try:
        mod = importlib.import_module(module)
        return hasattr(mod, attr)
    except ImportError:
        return False


def fingerprint_fenics() -> dict[str, Any]:
    """Fingerprint FEniCSx/dolfinx API."""
    fp = {"solver": "fenics", "timestamp": datetime.now().isoformat()}
    fp["dolfinx"] = _check_import("dolfinx")
    fp["basix"] = _check_import("basix")
    fp["ufl"] = _check_import("ufl")
    fp["gmsh"] = _check_import("gmsh")

    # Key API surface
    fp["api"] = {
        "dolfinx.mesh.create_rectangle": _check_attr("dolfinx.mesh", "create_rectangle"),
        "dolfinx.mesh.create_box": _check_attr("dolfinx.mesh", "create_box"),
        "dolfinx.fem.functionspace": _check_attr("dolfinx.fem", "functionspace"),
        "dolfinx.fem.Function": _check_attr("dolfinx.fem", "Function"),
        "dolfinx.fem.dirichletbc": _check_attr("dolfinx.fem", "dirichletbc"),
        "dolfinx.fem.petsc.LinearProblem": _check_attr("dolfinx.fem.petsc", "LinearProblem"),
        "dolfinx.io.XDMFFile": _check_attr("dolfinx.io", "XDMFFile"),
        "dolfinx.io.VTKFile": _check_attr("dolfinx.io", "VTKFile"),
        "dolfinx.default_scalar_type": _check_attr("dolfinx", "default_scalar_type"),
    }
    return fp


def fingerprint_ngsolve() -> dict[str, Any]:
    """Fingerprint NGSolve API."""
    fp = {"solver": "ngsolve", "timestamp": datetime.now().isoformat()}
    fp["ngsolve"] = _check_import("ngsolve")
    fp["netgen"] = _check_import("netgen")

    fp["api"] = {
        "ngsolve.H1": _check_attr("ngsolve", "H1"),
        "ngsolve.HCurl": _check_attr("ngsolve", "HCurl"),
        "ngsolve.HDiv": _check_attr("ngsolve", "HDiv"),
        "ngsolve.VectorH1": _check_attr("ngsolve", "VectorH1"),
        "ngsolve.BilinearForm": _check_attr("ngsolve", "BilinearForm"),
        "ngsolve.LinearForm": _check_attr("ngsolve", "LinearForm"),
        "ngsolve.GridFunction": _check_attr("ngsolve", "GridFunction"),
        "ngsolve.Mesh": _check_attr("ngsolve", "Mesh"),
        "ngsolve.VTKOutput": _check_attr("ngsolve", "VTKOutput"),
        "netgen.occ.OCCGeometry": _check_attr("netgen.occ", "OCCGeometry"),
    }
    return fp


def fingerprint_skfem() -> dict[str, Any]:
    """Fingerprint scikit-fem API."""
    fp = {"solver": "skfem", "timestamp": datetime.now().isoformat()}
    fp["skfem"] = _check_import("skfem")

    fp["api"] = {
        "skfem.MeshTri": _check_attr("skfem", "MeshTri"),
        "skfem.MeshQuad": _check_attr("skfem", "MeshQuad"),
        "skfem.MeshTet": _check_attr("skfem", "MeshTet"),
        "skfem.ElementTriP1": _check_attr("skfem", "ElementTriP1"),
        "skfem.ElementTriP2": _check_attr("skfem", "ElementTriP2"),
        "skfem.Basis": _check_attr("skfem", "Basis"),
        "skfem.asm": _check_attr("skfem", "asm"),
        "skfem.solve": _check_attr("skfem", "solve"),
        "skfem.condense": _check_attr("skfem", "condense"),
    }
    return fp


def fingerprint_kratos() -> dict[str, Any]:
    """Fingerprint Kratos Multiphysics API."""
    fp = {"solver": "kratos", "timestamp": datetime.now().isoformat()}
    fp["KratosMultiphysics"] = _check_import("KratosMultiphysics")

    fp["applications"] = {
        "StructuralMechanicsApplication": _check_import(
            "KratosMultiphysics.StructuralMechanicsApplication"),
        "FluidDynamicsApplication": _check_import(
            "KratosMultiphysics.FluidDynamicsApplication"),
        "ConstitutiveLawsApplication": _check_import(
            "KratosMultiphysics.ConstitutiveLawsApplication"),
        "DEMApplication": _check_import(
            "KratosMultiphysics.DEMApplication"),
        "ContactStructuralMechanicsApplication": _check_import(
            "KratosMultiphysics.ContactStructuralMechanicsApplication"),
        "CoSimulationApplication": _check_import(
            "KratosMultiphysics.CoSimulationApplication"),
    }

    fp["api"] = {
        "KM.Model": _check_attr("KratosMultiphysics", "Model"),
        "KM.ModelPart": _check_attr("KratosMultiphysics", "ModelPart"),
        "KM.DISPLACEMENT": _check_attr("KratosMultiphysics", "DISPLACEMENT"),
        "KM.YOUNG_MODULUS": _check_attr("KratosMultiphysics", "YOUNG_MODULUS"),
    }
    return fp


def fingerprint_dune() -> dict[str, Any]:
    """Fingerprint DUNE-fem API."""
    fp = {"solver": "dune", "timestamp": datetime.now().isoformat()}
    fp["dune.fem"] = _check_import("dune.fem")
    fp["dune.grid"] = _check_import("dune.grid")
    fp["dune.ufl"] = _check_import("dune.ufl")

    fp["api"] = {
        "dune.fem.space": _check_attr("dune.fem", "space") if _check_import("dune.fem")["importable"] else False,
        "dune.fem.function": _check_attr("dune.fem", "function") if _check_import("dune.fem")["importable"] else False,
    }
    return fp


def fingerprint_fourc() -> dict[str, Any]:
    """Fingerprint 4C — check binary existence and version."""
    import os
    import subprocess
    fp = {"solver": "fourc", "timestamp": datetime.now().isoformat()}

    binary = os.environ.get("FOURC_BINARY", "")
    if not binary:
        # Try common locations
        for candidate in [
            os.path.expanduser("~/4C/build/4C"),
            "/opt/4c/bin/4C",
        ]:
            if Path(candidate).exists():
                binary = candidate
                break

    fp["binary_found"] = bool(binary) and Path(binary).exists()
    fp["binary_path"] = binary

    if fp["binary_found"]:
        try:
            result = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=10
            )
            fp["version_output"] = result.stdout[:200] + result.stderr[:200]
        except Exception as e:
            fp["version_output"] = f"Error: {e}"
    return fp


def fingerprint_dealii() -> dict[str, Any]:
    """Fingerprint deal.II — check installation."""
    import os
    fp = {"solver": "dealii", "timestamp": datetime.now().isoformat()}

    root = os.environ.get("DEALII_ROOT", "")
    fp["root_found"] = bool(root) and Path(root).exists()
    fp["root_path"] = root

    # Check for cmake config
    if fp["root_found"]:
        cmake_dirs = list(Path(root).rglob("deal.IIConfig.cmake"))
        fp["cmake_config_found"] = len(cmake_dirs) > 0
        if cmake_dirs:
            fp["cmake_config_path"] = str(cmake_dirs[0])
    return fp


# ── Registry ────────────────────────────────────────────────

SOLVERS = {
    "fenics": fingerprint_fenics,
    "ngsolve": fingerprint_ngsolve,
    "skfem": fingerprint_skfem,
    "kratos": fingerprint_kratos,
    "dune": fingerprint_dune,
    "fourc": fingerprint_fourc,
    "dealii": fingerprint_dealii,
}


def generate_all(solvers: list[str] | None = None,
                  run_smoke: bool = True) -> dict[str, dict]:
    """Generate fingerprints for all (or specified) solvers.

    Args:
        solvers: List of solver names, or None for all.
        run_smoke: If True, also run smoke tests (actual FEM solves).
    """
    targets = solvers or list(SOLVERS.keys())
    results = {}
    for name in targets:
        if name in SOLVERS:
            print(f"  Fingerprinting {name}...", end=" ", flush=True)
            try:
                fp = SOLVERS[name]()
                results[name] = fp
                print("OK")
            except Exception as e:
                results[name] = {"solver": name, "error": str(e)}
                print(f"ERROR: {e}")

    if run_smoke:
        print("\nRunning smoke tests...")
        try:
            # Add src to path for smoke_tests module
            src_dir = str(Path(__file__).parent.parent / "src")
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            from core.smoke_tests import run_all_smoke_tests
            smoke_results = run_all_smoke_tests(targets)
            for name, sr in smoke_results.items():
                print(f"  {name}: {'PASS' if sr.passed else 'FAIL'}"
                      f" ({sr.duration_ms:.0f}ms)"
                      f"{'' if sr.passed else f' — {sr.error}'}")
                if name in results:
                    results[name]["smoke_test"] = sr.to_dict()
        except Exception as e:
            print(f"  Smoke tests unavailable: {e}")

    return results


def save_fingerprints(fingerprints: dict[str, dict]):
    """Save fingerprints to data/fingerprints/."""
    FINGERPRINT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fp in fingerprints.items():
        path = FINGERPRINT_DIR / f"{name}.json"
        path.write_text(json.dumps(fp, indent=2, default=str))
    # Also save a combined timestamp
    meta = FINGERPRINT_DIR / "_meta.json"
    meta.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(),
        "solvers": list(fingerprints.keys()),
    }, indent=2))


def compare_fingerprints(current: dict, previous: dict) -> list[str]:
    """Compare two fingerprints and return list of drift messages."""
    drifts = []
    solver = current.get("solver", "unknown")

    # Check API surface
    curr_api = current.get("api", {})
    prev_api = previous.get("api", {})

    for key, was_available in prev_api.items():
        now_available = curr_api.get(key)
        if was_available and not now_available:
            drifts.append(f"BREAKING: {solver} — {key} no longer available (was: {was_available})")
        elif not was_available and now_available:
            drifts.append(f"NEW: {solver} — {key} now available")

    # Check for entirely new API entries not in previous
    for key in curr_api:
        if key not in prev_api and curr_api[key]:
            drifts.append(f"NEW: {solver} — {key} now available")

    # Check version changes
    for module in ["dolfinx", "ngsolve", "skfem", "KratosMultiphysics", "dune.fem"]:
        if module in current and module in previous:
            cv = current[module].get("version")
            pv = previous[module].get("version")
            if cv and pv and cv != pv:
                drifts.append(f"VERSION: {solver} — {module} changed {pv} → {cv}")

    return drifts


def load_previous_fingerprints() -> dict[str, dict]:
    """Load previously saved fingerprints."""
    results = {}
    if not FINGERPRINT_DIR.exists():
        return results
    for f in FINGERPRINT_DIR.glob("*.json"):
        if f.name.startswith("_"):
            continue
        try:
            results[f.stem] = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return results


def main():
    parser = argparse.ArgumentParser(description="Solver API fingerprinting")
    parser.add_argument("--compare", action="store_true",
                        help="Compare against saved fingerprints")
    parser.add_argument("--solver", type=str, default="",
                        help="Fingerprint a single solver")
    args = parser.parse_args()

    solvers = [args.solver] if args.solver else None

    print("Generating solver fingerprints...")
    current = generate_all(solvers)

    if args.compare:
        previous = load_previous_fingerprints()
        all_drifts = []
        for name, fp in current.items():
            if name in previous:
                drifts = compare_fingerprints(fp, previous[name])
                all_drifts.extend(drifts)

        if all_drifts:
            print(f"\n⚠️  {len(all_drifts)} API drift(s) detected:")
            for d in all_drifts:
                print(f"  {d}")
            sys.exit(1)
        else:
            print("\n✓ No API drift detected.")
    else:
        save_fingerprints(current)
        print(f"\nFingerprints saved to {FINGERPRINT_DIR}/")


if __name__ == "__main__":
    main()
