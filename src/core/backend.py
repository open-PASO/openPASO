"""
Abstract base class for FEM solver backends.

Each backend (4C, FEniCS, deal.ii, ...) implements this interface,
providing solver-independent access to:
  - Installation check
  - Physics knowledge
  - Input generation (YAML, Python script, C++ code, ...)
  - Simulation execution
  - Result retrieval
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class BackendStatus(Enum):
    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    MISCONFIGURED = "misconfigured"


class InputFormat(Enum):
    YAML = "yaml"           # 4C
    PYTHON = "python"       # FEniCS/dolfinx
    CPP = "cpp"             # deal.ii
    XML = "xml"             # FEBio
    JSON = "json"           # generic
    SPARTA = "sparta"       # SPARTA DSMC input script (LAMMPS-style)


# Filename patterns a generator script may write for the actual solver input,
# keyed by the backend's declared input format. Ordered most-specific first.
# Note: deal.II writes main.cpp and SPARTA writes in.sparta; neither is matched
# by the old ["*.4C.yaml","*.yaml","input.*","solve.py","MainKratos.py"] glob,
# which silently broke run_with_generator for those backends (issue #39).
_GENERATED_INPUT_PATTERNS = {
    InputFormat.CPP:    ["main.cpp", "*.cpp", "*.cc", "*.cxx"],
    InputFormat.YAML:   ["*.4C.yaml", "*.yaml", "*.yml"],
    InputFormat.PYTHON: ["solve.py", "main.py", "*.py"],
    InputFormat.XML:    ["input.feb", "*.feb", "*.xml"],
    # Kratos is the only InputFormat.JSON backend, and its declared format is
    # its CONFIG format, not the thing you execute — every Kratos generator
    # emits a python script. `input.py` sits ahead of `*.json` because the DEM
    # generator is a file writer: it leaves MaterialsDEM.json,
    # ProjectParametersDEM.json, two .mdpa files and the input.py that drives
    # DEMAnalysisStage. Without this entry the glob picked MaterialsDEM.json,
    # which KratosBackend.run then wrote into MainKratos.py and executed as
    # python — a SyntaxError on the first brace. Measured 2026-08-07: the same
    # deck that scripts/audit_two_stage_templates.py runs to completion could
    # not be run through run_with_generator at all.
    InputFormat.JSON:   ["MainKratos.py", "input.py", "*.json"],
    InputFormat.SPARTA: ["in.sparta", "in.*", "*.sparta"],
}

# Legacy patterns kept as a fallback so previously-working backends still match.
_GENERATED_INPUT_LEGACY = ["*.4C.yaml", "*.yaml", "input.*", "solve.py",
                           "MainKratos.py"]

# Artifacts a generator run leaves behind that are never the input itself.
_GENERATED_INPUT_EXCLUDE_NAMES = {
    "generate_input.py", "CMakeLists.txt", "CMakeCache.txt", "Makefile",
    "cmake_install.cmake", "stdout.log", "stderr.log", "log.sparta",
}
_GENERATED_INPUT_EXCLUDE_SUFFIXES = {
    ".log", ".pyc", ".o", ".obj", ".vtk", ".vtu", ".vtp", ".pvd", ".h5", ".xdmf",
}


def find_generated_input(work_dir, backend=None) -> Optional[Path]:
    """Locate the solver-input file a generator script wrote in ``work_dir``.

    Backend-aware: prefers the file type matching ``backend.input_format``
    (e.g. ``main.cpp`` for deal.II, ``in.sparta`` for SPARTA), then falls back
    to the legacy pattern list, then to any plausible non-script artifact the
    generator produced. Returns the matching Path, or ``None`` if nothing
    looks like an input file.
    """
    work_dir = Path(work_dir)

    def _ok(p: Path) -> bool:
        return (p.is_file()
                and p.name not in _GENERATED_INPUT_EXCLUDE_NAMES
                and p.suffix not in _GENERATED_INPUT_EXCLUDE_SUFFIXES)

    patterns: list[str] = []
    if backend is not None:
        try:
            # input_format is a method on real backends but may be a plain
            # attribute on stubs; accept either.
            fmt = backend.input_format
            if callable(fmt):
                fmt = fmt()
            patterns = list(_GENERATED_INPUT_PATTERNS.get(fmt, []))
        except Exception:
            patterns = []
    patterns += _GENERATED_INPUT_LEGACY

    for pattern in patterns:
        matches = sorted(m for m in work_dir.glob(pattern) if _ok(m))
        if matches:
            return matches[0]

    # Last resort: any file the generator produced that is not a script/aux.
    leftovers = sorted(f for f in work_dir.iterdir() if _ok(f))
    return leftovers[0] if leftovers else None


def sorted_by_step(paths: list) -> list:
    """Sort result files by NUMERIC step index, not lexicographically.

    Fixes a silent-wrong bug: plain sorted() puts 'field_10.vtu' before
    'field_2.vtu', so callers taking [-1] as "final timestep" can read an EARLIER
    step (step 9 while step 10 exists) and report it as the converged result.
    Sort by the LAST integer in each filename stem, then name.
    """
    import re
    def _key(p):
        nums = re.findall(r"\d+", Path(p).stem)
        return (int(nums[-1]) if nums else -1, str(p))
    return sorted(paths, key=_key)


def detect_template_language(content: str, default: str) -> str:
    """Sniff a template's first-line markers and return the markdown
    fence language tag that matches the *content*, not the backend's
    declared ``input_format``.

    Catches the Kratos case: ``KratosBackend.input_format() == JSON``
    (its native config-file format), but its catalog generators emit
    Python scripts that use the ``KratosMultiphysics`` library
    directly.  Tagging those as ``json`` confuses any client that
    strips fences by language.

    Falls back to ``default`` (typically ``backend.input_format().value``)
    when no marker matches.
    """
    head = content.lstrip()[:200]
    if not head:
        return default
    python_markers = (
        '"""', "'''",
        "#!/",
        "from ", "import ",
        "def ", "class ",
        "# -*-",
    )
    if head.startswith(python_markers):
        return "python"
    if head.startswith(("<?xml", "<")):
        return "xml"
    if head.startswith(("{", "[")):
        return "json"
    if head.startswith(("//", "#include", "template<", "namespace ")):
        return "cpp"
    return default


@dataclass
class PhysicsCapability:
    """A physics problem this backend can solve."""
    name: str                          # e.g. "poisson", "linear_elasticity"
    description: str                   # human-readable
    spatial_dims: list[int]            # [2], [3], or [2, 3]
    element_types: list[str]           # e.g. ["QUAD4", "HEX8", "TRI3"]
    template_variants: list[str] = field(default_factory=list)


@dataclass
class JobHandle:
    """Tracks a running or completed simulation."""
    job_id: str
    backend_name: str
    work_dir: Path
    status: str = "pending"            # pending, running, completed, failed
    pid: Optional[int] = None
    return_code: Optional[int] = None
    elapsed: Optional[float] = None
    error: Optional[str] = None
    # When the run started (time.time()), for a backend that tells this run's output
    # files from those an earlier run left in a reused work directory.
    started_at: Optional[float] = None


class SolverBackend(ABC):
    """Abstract interface for a FEM solver backend."""

    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'fourc', 'fenics', 'dealii'."""

    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name, e.g. '4C Multiphysics', 'FEniCSx (dolfinx)'."""

    @abstractmethod
    def check_availability(self) -> tuple[BackendStatus, str]:
        """Check if the solver is installed and usable.

        Returns (status, message) where message explains the status.
        """

    @abstractmethod
    def input_format(self) -> InputFormat:
        """What kind of input this backend expects."""

    @abstractmethod
    def supported_physics(self) -> list[PhysicsCapability]:
        """List of physics problems this backend can solve."""

    @abstractmethod
    def get_knowledge(self, physics: str) -> dict:
        """Return domain knowledge for a physics module.

        Keys: description, materials, solver, pitfalls, unit_systems, etc.
        """

    @abstractmethod
    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        """Generate solver-specific input (YAML / Python / C++ / XML).

        Args:
            physics: Physics module name (e.g. "poisson")
            variant: Template variant (e.g. "2d", "3d")
            params: User-specified parameters to override defaults

        Returns:
            Input file content as string.
        """

    @abstractmethod
    def validate_input(self, content: str) -> list[str]:
        """Validate input content. Returns list of errors (empty = valid)."""

    @abstractmethod
    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        """Execute the simulation.

        Args:
            input_content: Generated input (YAML/Python/C++)
            work_dir: Directory for input/output files
            np: Number of MPI processes
            timeout: Maximum runtime in seconds

        Returns:
            JobHandle with status and output location.
        """

    @abstractmethod
    def get_result_files(self, job: JobHandle) -> list[Path]:
        """Return paths to VTU/XDMF output files from a completed job."""

    def get_version(self) -> Optional[str]:
        """Return solver version string, if detectable."""
        return None

    def precice_participant(self) -> dict:
        """How to make this backend a preCICE coupling participant.

        Returns a dict describing the participant-adapter pattern so an agent (or the
        general orchestrator) can wrap this solver into an arbitrary cross-code coupling:
          {description, exchange_loop, notes}
        Backends override this with solver-specific code; the default is the generic
        preCICE time-loop that any participant follows.
        """
        return {
            "description": f"Generic preCICE participant for {self.name()}",
            "exchange_loop": (
                "import precice, numpy as np\n"
                "p = precice.Participant(NAME, 'precice-config.xml', 0, 1)\n"
                "vid = p.set_mesh_vertices(MESH, coords)   # interface coordinates\n"
                "p.initialize()\n"
                "while p.is_coupling_ongoing():\n"
                "    dt = p.get_max_time_step_size()\n"
                "    read_vals = p.read_data(MESH, READ_DATA, vid, dt)   # inputs from partner\n"
                "    # ... advance this solver one window using read_vals ...\n"
                "    p.write_data(MESH, WRITE_DATA, vid, out_vals)        # outputs to partner\n"
                "    p.advance(dt)\n"
                "p.finalize()"
            ),
            "notes": "Set LD_LIBRARY_PATH to libprecice; match pyprecice to the libprecice version.",
        }


def get_python_executable() -> str:
    """Get the Python executable that's running this MCP server.

    This is critical: when the MCP server runs inside a venv, we must use
    sys.executable (the venv Python) rather than shutil.which('python3')
    (which may find the system Python that lacks our packages).
    """
    import sys
    return sys.executable


def get_env_with_source_root(env_var: str) -> dict:
    """Get an environment dict that includes a source build from *_ROOT.

    If the env var (e.g. KRATOS_ROOT, NGSOLVE_ROOT) points to a directory
    with a build/, that build path is prepended to PYTHONPATH so the
    source-built version takes priority over pip-installed packages.

    This enables the develop workflow: modify source → build → use.

    Returns a copy of os.environ with PYTHONPATH adjusted.
    """
    import os
    env = os.environ.copy()
    root = os.environ.get(env_var, "")
    if not root or not Path(root).is_dir():
        return env

    # Check for Python source builds — cmake install/ takes priority, then build/
    for build_dir in ["install", "install/lib", "build/lib", "build",
                      "build/Release/lib", "build/Release", "lib"]:
        candidate = Path(root) / build_dir
        if candidate.is_dir():
            # Check if it contains Python packages
            has_python = any(candidate.rglob("*.py"))
            if has_python:
                pp = env.get("PYTHONPATH", "")
                build_path = str(candidate)
                if build_path not in pp:
                    env["PYTHONPATH"] = f"{build_path}:{pp}" if pp else build_path
                break

    # Also add the root itself (for editable installs / src layouts)
    pp = env.get("PYTHONPATH", "")
    if root not in pp:
        env["PYTHONPATH"] = f"{root}:{pp}" if pp else root

    # Add shared library paths (for compiled C++ extensions)
    for libs_dir in ["install/libs", "install/lib", "build/libs", "build/lib", "lib"]:
        candidate = Path(root) / libs_dir
        if candidate.is_dir() and any(candidate.glob("*.so")):
            ld = env.get("LD_LIBRARY_PATH", "")
            libs_path = str(candidate)
            if libs_path not in ld:
                env["LD_LIBRARY_PATH"] = f"{libs_path}:{ld}" if ld else libs_path
            break

    return env


def error_excerpt(text, limit: int = 1200) -> str:
    """Excerpt a solver failure so the CAUSE survives the truncation.

    A traceback is written cause-last. Cutting it from the front with
    ``job.error[:500]`` therefore keeps the stack frames and throws away the one
    line that says what went wrong. Measured on a real FEniCSx failure (a vector
    source assembled against a scalar space): the traceback is 765 characters,
    the first 500 are ``ufl/operators.py`` frames, and
    ``ValueError: Shapes do not match`` — the whole diagnosis — falls outside
    them. A model handed that reads the plumbing and not the fault, which is the
    opposite of what openPASO exists to do.

    So keep both ends: enough of the head to see which line of the input failed,
    and the whole tail, where the exception is. Short errors are returned
    untouched.
    """
    if not text:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    return (text[:head].rstrip() + "\n\n[... " + str(len(text) - limit)
            + " characters of stack omitted; the cause is below ...]\n\n"
            + text[-tail:].lstrip())
