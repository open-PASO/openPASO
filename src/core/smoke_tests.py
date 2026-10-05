"""Solver smoke tests — actually run trivial FEM problems to verify the pipeline.

Each test takes <2 seconds: mesh → function space → assemble/solve → check result.
Goes beyond hasattr by testing the full solver pipeline end-to-end.

Results are stored in the fingerprint alongside the API surface check.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class SmokeResult:
    """Result of a solver smoke test."""
    solver: str
    passed: bool
    error: Optional[str] = None
    version: Optional[str] = None
    duration_ms: float = 0.0
    tested_at: str = ""

    def __post_init__(self):
        if not self.tested_at:
            self.tested_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _run_script(python: str, script: str, timeout: int = 15) -> tuple[bool, str, str]:
    """Run a Python script in a subprocess. Returns (success, first_line_stdout, stderr).

    Only returns the first line of stdout to avoid banners/cleanup output.
    """
    try:
        r = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        # Take only first non-empty line (avoids Kratos/4C banners on cleanup)
        first_line = ""
        for line in r.stdout.splitlines():
            line = line.strip()
            if line and line.startswith("{"):
                first_line = line
                break
        return r.returncode == 0, first_line, r.stderr[-500:] if r.stderr else ""
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)


def smoke_ngsolve() -> SmokeResult:
    """Smoke test: solve Laplace on unit square with NGSolve."""
    t0 = time.time()
    script = '''
import ngsolve as ngs
from netgen.geom2d import unit_square
mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
V = ngs.H1(mesh, order=1, dirichlet="bottom|right|top|left")
u, v = V.TnT()
a = ngs.BilinearForm(ngs.grad(u)*ngs.grad(v)*ngs.dx).Assemble()
f = ngs.LinearForm(1*v*ngs.dx).Assemble()
gfu = ngs.GridFunction(V)
gfu.vec.data = a.mat.Inverse(V.FreeDofs()) * f.vec
print(f'{{"ok": true, "max": {max(gfu.vec):.6f}, "version": "{ngs.__version__}"}}')
'''
    # NOT `sys.executable`. NGSolve usually lives in its own environment, and
    # the backend already resolves which interpreter that is -- so running the
    # smoke script in openPASO's own Python asks a question about the wrong
    # machine. Measured the moment the backend learned to look outside its own
    # venv: check_availability() reported NGSolve 6.2.2604 and this smoke test
    # answered "No module named 'ngsolve'" in the same second. Same defect the
    # DUNE smoke test carried below, one backend over.
    from backends.ngsolve.backend import _find_ngsolve_python
    interpreter = str(_find_ngsolve_python() or sys.executable)
    ok, stdout, stderr = _run_script(interpreter, script)
    dt = (time.time() - t0) * 1000
    if ok:
        try:
            data = json.loads(stdout)
            return SmokeResult("ngsolve", True, version=data.get("version"),
                               duration_ms=round(dt, 1))
        except json.JSONDecodeError:
            return SmokeResult("ngsolve", False, error=f"Bad output: {stdout[:200]}",
                               duration_ms=round(dt, 1))
    return SmokeResult("ngsolve", False, error=stderr[:300], duration_ms=round(dt, 1))


def smoke_skfem() -> SmokeResult:
    """Smoke test: solve Poisson on unit square with scikit-fem."""
    t0 = time.time()
    script = '''
import skfem, json, numpy as np
from skfem import *
from skfem.models.poisson import laplace, unit_load
m = MeshTri.init_symmetric()
e = ElementTriP1()
ib = Basis(m, e)
A = asm(laplace, ib)
b = asm(unit_load, ib)
# Use boundary nodes by coordinate (symmetric mesh has no named boundaries)
boundary_dofs = ib.get_dofs(lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0))
u = solve(*condense(A, b, D=boundary_dofs.all()))
print(json.dumps({"ok": True, "max": float(u.max()), "version": skfem.__version__}))
'''
    ok, stdout, stderr = _run_script(sys.executable, script)
    dt = (time.time() - t0) * 1000
    if ok:
        try:
            data = json.loads(stdout)
            return SmokeResult("skfem", True, version=data.get("version"),
                               duration_ms=round(dt, 1))
        except json.JSONDecodeError:
            return SmokeResult("skfem", False, error=f"Bad output: {stdout[:200]}",
                               duration_ms=round(dt, 1))
    return SmokeResult("skfem", False, error=stderr[:300], duration_ms=round(dt, 1))


def smoke_kratos() -> SmokeResult:
    """Smoke test: create a model and check Kratos core works."""
    t0 = time.time()
    script = '''
import KratosMultiphysics as KM, json
model = KM.Model()
mp = model.CreateModelPart("test")
mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
mp.CreateNewNode(1, 0.0, 0.0, 0.0)
mp.CreateNewNode(2, 1.0, 0.0, 0.0)
ver = str(KM.Kernel.Version()).replace('"', '')
print(json.dumps({"ok": True, "nodes": mp.NumberOfNodes(), "version": ver}))
'''
    # NOT `sys.executable`. Kratos is a heavy compiled package normally
    # installed in an environment of its own, and the backend now finds it
    # there. Measured the moment it learned to: check_availability() said
    # Kratos 10.3.0 and this smoke test said "No module named
    # 'KratosMultiphysics'" in the same second. Third instance of this
    # defect after DUNE and NGSolve -- the check and the work must ask the
    # same question of the same interpreter.
    from backends.kratos.backend import _find_kratos_python
    interpreter = _find_kratos_python() or sys.executable
    ok, stdout, stderr = _run_script(interpreter, script)
    dt = (time.time() - t0) * 1000
    if ok:
        try:
            data = json.loads(stdout)
            return SmokeResult("kratos", True, version=data.get("version"),
                               duration_ms=round(dt, 1))
        except json.JSONDecodeError:
            return SmokeResult("kratos", False, error=f"Bad output: {stdout[:200]}",
                               duration_ms=round(dt, 1))
    return SmokeResult("kratos", False, error=stderr[:300], duration_ms=round(dt, 1))


def smoke_dune() -> SmokeResult:
    """Smoke test: import dune.fem (JIT compile may take time on first run).

    Known limitation: import-only — does NOT solve anything, because a
    real dune.fem solve JIT-compiles C++ modules on first use (minutes).
    A passing result means importable, not solve-capable; upgrade to a
    solve-based test once a working install is available to validate it.
    """
    t0 = time.time()
    script = '''
import json
try:
    import dune.fem
    print(json.dumps({"ok": True}))
except ImportError as e:
    print(json.dumps({"ok": False, "error": str(e)}))
'''
    # DUNE LIVES IN ITS OWN INTERPRETER, AND THIS RAN IN OURS.
    #
    # This was `sys.executable`, the Python running openPASO. DUNE-fem is
    # installed in a separate environment by design -- that is what DUNE_PYTHON
    # and the conda auto-detection are for -- so the script could only ever
    # report "No module named 'dune'" unless openPASO itself happened to run
    # inside the DUNE env. A correct install therefore failed its own smoke
    # test, which is worse than having no smoke test: it says the solver is
    # broken when the solver is fine. The backend already resolves this.
    from backends.dune.backend import _find_dune_python
    interpreter = _find_dune_python() or sys.executable
    ok, stdout, stderr = _run_script(interpreter, script, timeout=60)
    dt = (time.time() - t0) * 1000
    if ok:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return SmokeResult("dune", False, error=f"Bad output: {stdout[:200]}",
                               duration_ms=round(dt, 1))
        # The inner script exits 0 even on ImportError — the verdict
        # lives in the "ok" flag, not the return code.
        if data.get("ok") is True:
            return SmokeResult("dune", True, duration_ms=round(dt, 1))
        return SmokeResult("dune", False,
                           error=str(data.get("error", "unknown"))[:300],
                           duration_ms=round(dt, 1))
    return SmokeResult("dune", False, error=stderr[:300], duration_ms=round(dt, 1))


def smoke_fourc() -> SmokeResult:
    """Smoke test: the 4C binary starts and prints its usage.

    `4C --help` exits 0 and names the program ("Comprehensive Computational
    Community Code") on a development build before 2026.2.0, on 4C 2026.2.0 and
    on 4C 2026.3.0 alike. `--version` is not an option of 4C: all three exit 109
    with "The following argument was not expected: --version", so the verdict
    comes from the exit code and that text, never from the call merely
    returning."""
    t0 = time.time()
    # The backend's own finder, so the smoke test checks the binary openPASO
    # will run (a Spack-built 4C included), not a shorter list of its own.
    from backends.fourc.backend import _find_fourc_binary
    found = _find_fourc_binary()
    binary = str(found) if found else ""
    if not binary or not Path(binary).exists():
        return SmokeResult("fourc", False, error="Binary not found",
                           duration_ms=0)
    # The libraries of a source build, but never for a Spack build.
    from backends.fourc.backend import fourc_library_env
    env = fourc_library_env(binary)
    env.pop("DISPLAY", None)
    try:
        r = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30,
                           stdin=subprocess.DEVNULL, env=env)
    except Exception as e:
        return SmokeResult("fourc", False, error=str(e),
                           duration_ms=round((time.time()-t0)*1000, 1))
    dt = round((time.time() - t0) * 1000, 1)
    text = r.stdout + r.stderr
    if r.returncode != 0 or "Comprehensive Computational Community Code" not in text:
        tail = " ".join(text.split())[-300:]
        return SmokeResult("fourc", False,
                           error=f"`{binary} --help` exited {r.returncode}: {tail}",
                           duration_ms=dt)
    return SmokeResult("fourc", True, version="found", duration_ms=dt)


# ── Registry ─────────────────────────────────────────────────

SMOKE_TESTS = {
    "ngsolve": smoke_ngsolve,
    "skfem": smoke_skfem,
    "kratos": smoke_kratos,
    "dune": smoke_dune,
    "fourc": smoke_fourc,
}


def run_all_smoke_tests(solvers: list[str] | None = None) -> dict[str, SmokeResult]:
    """Run smoke tests for all (or specified) solvers."""
    targets = solvers or list(SMOKE_TESTS.keys())
    results = {}
    for name in targets:
        if name in SMOKE_TESTS:
            try:
                results[name] = SMOKE_TESTS[name]()
            except Exception as e:
                results[name] = SmokeResult(name, False, error=str(e))
    return results
