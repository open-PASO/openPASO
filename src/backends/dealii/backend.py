"""
deal.ii solver backend.

Generates C++ source files based on deal.ii tutorial step patterns,
compiles them with CMake, and runs the resulting executables.

deal.ii tutorials used:
  - step-3/4/5: Poisson / Laplace equation
  - step-8/17:  Linear elasticity
  - step-26:    Heat equation (transient)
  - step-22:    Stokes flow
"""

import asyncio
import logging
import re
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from core.backend import (
    sorted_by_step,
    SolverBackend, BackendStatus, InputFormat,
    PhysicsCapability, JobHandle,
)
from core.registry import register_backend

logger = logging.getLogger("openpaso.dealii")


class DealiiRootOverrideError(RuntimeError):
    """DEAL_II_DIR / DEALII_ROOT names something that is not deal.II.

    Raised rather than ignored, for the same reason FEBIO_BINARY raises:
    silently searching elsewhere means the install openPASO reports is not
    the install the user named. `DEAL_II_DIR=/tmp` used to log a warning
    and CONTINUE discovery, so the backend reported
    `available — deal.II 9.8.0-pre at {DEALII_BUILD}` while
    the variable pointed at /tmp. Every surface then agreed the backend
    was fine, and the only place the wrong path would surface was the
    compile error, much later, naming neither variable.
    """


# Cache of verification verdicts, keyed by resolved path. A verdict is
# True (proven deal.II), False (proven not), or None (could not look).
_VERIFY_CACHE: dict[str, tuple[Optional[bool], str]] = {}


def verify_dealii_install(root: Path) -> tuple[Optional[bool], str]:
    """Prove that ``root`` really is a deal.II installation.

    deal.II is a C++ library with no executable to interrogate, so the
    equivalent of "run it and look for its own vocabulary" is to read
    the version macro that deal.II's OWN generated header carries, or
    the version its own CMake package file declares. A directory that
    merely exists, or that happens to contain a differently-named
    ``include/`` tree, cannot produce either string.

    Returns ``(verdict, detail)``:
      * ``(True,  "9.8.0-pre (include/deal.II/base/config.h)")`` —
        proven; ``detail`` names the version and the evidence.
      * ``(False, reason)`` — looked, and the evidence is absent.
      * ``(None,  reason)`` — could NOT look (unreadable path,
        permission error). Callers must FAIL OPEN on None: an
        unreadable install is not a wrong install, and refusing to
        run because we could not inspect it would be its own bug.

    Verdicts are cached per path; the filesystem does not change under
    us within a session and ``discover`` may ask repeatedly.
    """
    key = str(root)
    if key in _VERIFY_CACHE:
        return _VERIFY_CACHE[key]

    result: tuple[Optional[bool], str]
    unreadable = False
    version = ""
    evidence = ""

    # (a) The generated header, which carries deal.II's own
    #     DEAL_II_PACKAGE_VERSION macro. It exists in an installed
    #     prefix, and in the BUILD tree of a source checkout — note a
    #     source checkout's own include/ has only config.h.in, so a
    #     bare checkout root is not by itself proof.
    for rel in (Path("include"),):
        cfg = root / rel / "deal.II" / "base" / "config.h"
        if not cfg.is_file():
            continue
        try:
            m = re.search(
                r'#\s*define\s+DEAL_II_PACKAGE_VERSION\s+"([^"]+)"',
                cfg.read_text(errors="replace"))
            if m:
                version = m.group(1)
                evidence = str(rel / "deal.II" / "base" / "config.h")
                break
        except OSError as exc:
            unreadable = True
            evidence = f"could not read {cfg}: {exc}"

    # (b) deal.II's own CMake package-version file, in any of the
    #     layouts it ships (installed prefix, or a build tree).
    if not version:
        for rel in (Path("lib") / "cmake" / "deal.II",
                    Path("share") / "deal.II" / "cmake",
                    Path("cmake") / "config"):
            vf = root / rel / "deal.IIConfigVersion.cmake"
            if not vf.is_file():
                continue
            try:
                m = re.search(r'set\(\s*PACKAGE_VERSION\s+"([^"]+)"',
                              vf.read_text(errors="replace"))
                if m:
                    version, evidence = m.group(1), str(rel / vf.name)
                    break
            except OSError as exc:
                unreadable = True
                evidence = f"could not read {vf}: {exc}"

    if version:
        result = (True, f"deal.II {version} ({evidence})")
    elif unreadable:
        result = (None, evidence)
    else:
        result = (False,
                  "no DEAL_II_PACKAGE_VERSION in "
                  "include/deal.II/base/config.h and no "
                  "deal.IIConfigVersion.cmake under lib/cmake/deal.II, "
                  "share/deal.II/cmake or cmake/config")

    _VERIFY_CACHE[key] = result
    return result


def resolve_dealii_root(candidate: Path) -> Optional[Path]:
    """Return the sub-path of ``candidate`` that find_package can use.

    A source checkout is not itself usable: its ``include/`` holds
    ``config.h.in``, and the generated header plus the CMake package
    files live in the build directory. Try the candidate first, then
    the usual build/install layouts, and return the first that
    verifies (or that we could not inspect, since we fail open).
    """
    unreadable: Optional[Path] = None
    for rel in (Path("."), Path("build"), Path("install"),
                Path("build") / "install"):
        cand = (candidate / rel).resolve()
        if not cand.is_dir():
            continue
        verdict, _ = verify_dealii_install(cand)
        if verdict is True:
            return cand
        if verdict is None and unreadable is None:
            unreadable = cand      # fail open, but prefer a proven hit
    return unreadable


# The templates' cmake_minimum_required (see _generate_cmakelists).
_CMAKE_MINIMUM = (3, 13, 4)
_CMAKE_VERSIONS: dict[str, Optional[tuple[int, ...]]] = {}


def _cmake_version(cmake: str) -> Optional[tuple[int, ...]]:
    """`cmake --version` as numbers, once per executable; None if it cannot say."""
    if cmake not in _CMAKE_VERSIONS:
        import subprocess
        try:
            out = subprocess.run([cmake, "--version"], capture_output=True, text=True,
                                 timeout=10, stdin=subprocess.DEVNULL).stdout
            m = re.search(r"cmake version (\d+)\.(\d+)\.(\d+)", out)
            _CMAKE_VERSIONS[cmake] = tuple(int(x) for x in m.groups()) if m else None
        except (OSError, subprocess.TimeoutExpired):
            _CMAKE_VERSIONS[cmake] = None
    return _CMAKE_VERSIONS[cmake]


def _cmake_executable() -> Optional[str]:
    """The cmake that builds the templates: the one on PATH if it is new enough
    for their cmake_minimum_required, else the one Spack built for deal.II (a
    machine set up only through Spack has no other). An older cmake on PATH is
    still returned when there is no other, so that its own message says what is
    wrong instead of "CMake not found"."""
    on_path = shutil.which("cmake")
    if on_path and (_cmake_version(on_path) or (0,)) >= _CMAKE_MINIMUM:
        return on_path
    from core.spack import installed_prefix   # noqa: PLC0415
    prefix = installed_prefix("cmake")
    if prefix is not None and (prefix / "bin" / "cmake").is_file():
        return str(prefix / "bin" / "cmake")
    return on_path


# Where Spack keeps its compiler wrappers: in the compiler-wrapper package's
# prefix (<prefix>/libexec/spack/gcc/g++) from Spack 1.0 on, inside the Spack
# checkout itself (<spack>/lib/spack/env/gcc/g++) before that.
_SPACK_WRAPPER_DIRS = ("/libexec/spack/", "/lib/spack/env/")


def _records_spack_wrapper(root: Path) -> bool:
    """Whether deal.IIConfig.cmake under `root` names Spack's compiler wrapper."""
    config = root / "lib" / "cmake" / "deal.II" / "deal.IIConfig.cmake"
    try:
        text = config.read_text(errors="replace")
    except OSError:
        return False
    return any(marker in text for marker in _SPACK_WRAPPER_DIRS)


def _describe_exit(rc: int) -> str:
    """The program's return code in words; a signal is named, and a crash is
    said to be the program's (measured: code 139 / -11 is a segfault inside it)."""
    import signal as _signal
    sig = -rc if rc < 0 else (rc - 128 if rc > 128 else 0)
    try:
        name = _signal.Signals(sig).name if sig else ""
    except ValueError:
        name = ""
    if not name:
        return f"the program exited with return code {rc}"
    what = ("a crash inside your program, not an install fault. This deal.II may be a Release "
            "build that asserts nothing: an FEValues accessor whose update flag was not requested, "
            "an unsized index vector, or a SparseMatrix copy-constructed from a filled one (it is left "
            "empty) crashes with no message. Add "
            "target_compile_definitions(<target> PRIVATE DEBUG) after deal_ii_setup_target and "
            "run again: deal.II's header checks then name a missing flag, while the empty matrix and "
            "the unsized vector, whose checks are compiled into its library, crash with no message even then"
            if name in ("SIGSEGV", "SIGBUS", "SIGFPE", "SIGABRT", "SIGILL") else "")
    return f"the program was KILLED BY {name} (return code {rc})" + (f": {what}." if what else ".")


def build_tree_note() -> str:
    """One line naming the deal.II tree to build against on this install, and a
    system package that is not it; '' when there is no tree to name."""
    try:
        root = _find_dealii()
    except DealiiRootOverrideError as exc:
        return str(exc)
    if root is None:
        return ""
    verdict, detail = verify_dealii_install(root)
    note = (f"{detail.split(' (')[0]} at {root}" if verdict else f"deal.II at {root}") + \
        f": build against THIS tree, DEAL_II_DIR={root}"
    sys_cfg = Path("/usr/include/deal.II/base/config.h")
    try:
        sys_ver = next((ln.split('"')[1] for ln in sys_cfg.read_text().splitlines()
                        if "DEAL_II_PACKAGE_VERSION" in ln and '"' in ln), "")
    except OSError:
        sys_ver = ""
    if sys_ver and Path(root).resolve() != Path("/usr"):
        note += (f". The system package at /usr (deal.II {sys_ver}) is NOT the one to build "
                 f"against: a compile that reads /usr/include/deal.II found that package, or ran "
                 f"without deal_ii_setup_target")
    return note


def _find_dealii() -> Optional[Path]:
    """Locate a deal.II installation root.

    Discovery order (first hit wins):
      1. ``DEAL_II_DIR`` env variable (explicit override).
      2. ``DEALII_ROOT`` env variable (alternate spelling).
      3. Conda envs at ``~/miniconda3/envs/*`` and
         ``~/anaconda3/envs/*`` that contain ``include/deal.II/``.
      4. User-source dirs: ``~/dealii``, ``~/deal.II``,
         ``~/src/dealii``, ``~/src/deal.II``, and ``dealii`` / ``deal.II``
         in the user's desktop folder (core.user_dirs, locale-aware).
      5. A Spack-built deal.II (core.spack), ahead of the system paths
         because an old distribution deal.II is the hazard those carry.
      6. System paths: ``/opt/dealii``,
         ``/usr/lib/x86_64-linux-gnu/cmake/deal.II``,
         ``/usr/share/cmake/deal.II``.
      7. ``cmake --find-package`` for system-installed deal.II.

    Returns the install ROOT (the path that contains
    ``include/deal.II/`` or ``share/deal.II/cmake/``). Callers
    pass this to ``find_package(deal.II HINTS ...)``.
    """
    def _looks_like_dealii_root(p: Path) -> bool:
        """A path is a deal.II install root if it has either
        include/deal.II/ headers or share/deal.II/cmake/ macros
        or lib/cmake/deal.II/ config."""
        if not p.is_dir():
            return False
        return ((p / "include" / "deal.II").is_dir()
                or (p / "share" / "deal.II" / "cmake").is_dir()
                or (p / "lib" / "cmake" / "deal.II").is_dir())

    # 1. Explicit env override. The path is VERIFIED, not merely
    #    tested for existence: this branch used to `return
    #    Path(env_dir)` for any directory that happened to exist, so
    #    `DEAL_II_DIR=/tmp` (or DEALII_ROOT=/tmp) made
    #    check_availability report "deal.II found at /tmp" and the
    #    backend advertise itself as available with no deal.II
    #    anywhere near it. An availability check that a wrong path
    #    can satisfy is worse than no check: it turns a clear
    #    "not installed" into template generation that fails much
    #    later, at compile time, with an unrelated error.
    #
    #    Verifying is only half of it. The first fix WARNED and then
    #    continued discovery, which restored the same false verdict by
    #    a longer route: `DEAL_II_DIR=/tmp` fell through to the real
    #    build tree and the backend reported AVAILABLE again — now with
    #    an accurate-looking version string, from an install the user
    #    had not named. An explicit override that resolves elsewhere is
    #    how you debug the install you are not running, so a set-but-
    #    wrong override is an ERROR, not a hint.
    for env_var in ("DEAL_II_DIR", "DEALII_ROOT"):
        env_dir = os.environ.get(env_var)
        if not env_dir:
            continue
        cand = Path(env_dir)
        if not cand.is_dir():
            raise DealiiRootOverrideError(
                f"{env_var} is set to {env_dir!r}, which is not a "
                f"directory. Refusing to fall back to a different "
                f"install: point it at a prefix containing "
                f"include/deal.II/base/config.h (or at the build "
                f"directory of a source checkout), or unset it.")
        resolved = resolve_dealii_root(cand)
        if resolved is None:
            # resolve_dealii_root already fails OPEN on a path it could
            # not inspect (it returns the unreadable candidate), so None
            # means we DID look and deal.II's own version evidence is
            # absent. Fail closed.
            raise DealiiRootOverrideError(
                f"{env_var} is set to {env_dir!r}, which does not look "
                f"like a deal.II install: neither "
                f"include/deal.II/base/config.h nor "
                f"lib/cmake/deal.II/deal.IIConfigVersion.cmake is there, "
                f"under it, or under its build/ or install/ subdirectory. "
                f"Refusing to fall back to a different install and report "
                f"that one as available. Fix the path or unset the "
                f"variable.")
        # Proven, or unreadable -> fail OPEN and trust the override.
        return resolved

    # 2 + 3. Conda envs (deal.II often lives in a dedicated env).
    # When several envs contain deal.II, prefer the HIGHEST version:
    # iterdir() order is arbitrary, and on this machine an old
    # ofa-dealii (9.1.1, serial, missing fe_interface_values.h /
    # hp/refinement.h / count_dofs_per_fe_block) used to shadow the
    # newer ofa-dealii-93 (9.3.2) depending on directory order —
    # 10 of 39 catalog templates failed to compile purely from
    # losing that race (probe 2026-06-12).
    def _dealii_version_of(root: Path) -> tuple:
        cfg = root / "include" / "deal.II" / "base" / "config.h"
        try:
            for line in cfg.read_text().splitlines():
                if "DEAL_II_PACKAGE_VERSION" in line and '"' in line:
                    ver = line.split('"')[1]
                    return tuple(int(x) for x in ver.split(".")[:3])
        except (OSError, ValueError):
            pass
        return (0, 0, 0)

    candidates: list[Path] = []
    for conda_base in (Path.home() / "miniconda3" / "envs",
                       Path.home() / "anaconda3" / "envs",
                       Path.home() / "miniforge3" / "envs"):
        if not conda_base.is_dir():
            continue
        for env_dir in conda_base.iterdir():
            if _looks_like_dealii_root(env_dir):
                candidates.append(env_dir)
    if candidates:
        return max(candidates, key=_dealii_version_of)

    # 4. User-source dirs (in case the user built from source). A
    #    checkout root is NOT usable on its own — the generated header
    #    and the CMake package files live in the build tree — so each
    #    candidate goes through resolve_dealii_root().
    from core.user_dirs import desktop_dirs   # noqa: PLC0415
    _cands = [Path.home() / s for s in ("dealii", "deal.II", "src/dealii", "src/deal.II")]
    _cands += [d / s for d in desktop_dirs() for s in ("dealii", "deal.II")]
    for candidate in _cands:
        if not candidate.is_dir():
            continue
        resolved = resolve_dealii_root(candidate)
        if resolved is not None:
            return resolved

    # 5. Spack. A deal.II built by `openpaso install dealii --via spack`
    #    lives in a hashed prefix no fixed list can name; Spack knows it.
    #    openPASO's own build first. Any other Spack deal.II only if its
    #    deal.IIConfig.cmake names real compilers: upstream's recipe leaves
    #    Spack's compiler wrapper there, and every template built against it
    #    stops at CMake's compiler check ("Spack compiler must be run from
    #    Spack!").
    from core.spack import installed_prefix   # noqa: PLC0415
    for spack_spec in ("openpaso.dealii", "dealii"):
        spack_prefix = installed_prefix(spack_spec)
        if (spack_prefix is not None and _looks_like_dealii_root(spack_prefix)
                and not _records_spack_wrapper(spack_prefix)):
            return spack_prefix

    # 6. System paths.
    for cand in (Path("/opt/dealii"), Path("/opt/deal.II"),
                 Path("/usr/local/dealii"),
                 Path("/usr/local/deal.II"),
                 Path("/usr")):
        if _looks_like_dealii_root(cand):
            return cand

    # 7. CMake fall-back.
    cmake = _cmake_executable()
    if cmake:
        import subprocess
        try:
            r = subprocess.run(
                [cmake, "--find-package", "-DNAME=deal.II",
                 "-DCOMPILER_ID=GNU", "-DLANGUAGE=CXX",
                 "-DMODE=COMPILE"], stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=10
            )
            if r.returncode == 0:
                return Path("/usr")  # system-installed
        except Exception:
            pass

    return None


class DealiiBackend(SolverBackend):

    def name(self) -> str:
        return "dealii"

    def display_name(self) -> str:
        return "deal.II"

    def check_availability(self) -> tuple[BackendStatus, str]:
        # Check for cmake
        cmake = _cmake_executable()
        if not cmake:
            return BackendStatus.NOT_INSTALLED, "CMake not found"

        # Check for deal.II headers/library
        try:
            dealii = _find_dealii()
        except DealiiRootOverrideError as exc:
            # Report as MISCONFIGURED rather than propagating: `discover`
            # must keep working and list the other backends, but this one
            # must not read as available, and must not read as simply
            # absent either — the user set a variable and it is wrong.
            # NOT `_check_via_compile()`: find_package(deal.II) consults
            # DEAL_II_DIR itself, so probing would answer for whatever
            # CMake finds next and re-hide the bad override.
            return BackendStatus.MISCONFIGURED, str(exc)
        if not dealii:
            # Try a test compile
            return self._check_via_compile()

        # Report WHAT was found, not just WHERE. A message of the form
        # "deal.II <version> at <path>" cannot be produced by a path
        # that is not a deal.II install, which is the point: the old
        # message ("deal.II found at {dealii}") was satisfied by any
        # directory reachable through DEAL_II_DIR / DEALII_ROOT.
        verdict, detail = verify_dealii_install(dealii)
        if verdict is False:
            # _find_dealii's non-env branches already require deal.II
            # markers, so getting here means those markers exist but
            # no version does — a broken or partial install. Fall back
            # to the compile probe, which is the authority.
            logger.warning("%s has deal.II markers but no version (%s)",
                           dealii, detail)
            return self._check_via_compile()
        if verdict is None:
            # Could not look. FAIL OPEN, and say so rather than
            # claiming a verification that did not happen.
            return (BackendStatus.AVAILABLE,
                    f"deal.II at {dealii} (version not verifiable: "
                    f"{detail})")
        return BackendStatus.AVAILABLE, f"{detail} at {dealii}"

    def _check_via_compile(self) -> tuple[BackendStatus, str]:
        """Try to compile a minimal deal.II program to check availability."""
        import subprocess
        import tempfile

        test_cpp = '#include <deal.II/base/utilities.h>\nint main(){return 0;}\n'
        test_cmake = (
            'cmake_minimum_required(VERSION 3.13.4)\n'
            'find_package(deal.II REQUIRED)\n'
            'deal_ii_initialize_cached_variables()\n'
            'project(test)\n'
            'deal_ii_setup_target(test)\n'
            'add_executable(test test.cpp)\n'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.cpp").write_text(test_cpp)
            Path(tmpdir, "CMakeLists.txt").write_text(test_cmake)
            try:
                r = subprocess.run(
                    [_cmake_executable() or "cmake", "."], stdin=subprocess.DEVNULL,
                    capture_output=True, text=True, cwd=tmpdir, timeout=30
                )
                if r.returncode == 0:
                    return BackendStatus.AVAILABLE, "deal.II found via CMake"
                else:
                    return BackendStatus.NOT_INSTALLED, f"CMake cannot find deal.II: {r.stderr[:200]}"
            except Exception as e:
                return BackendStatus.NOT_INSTALLED, f"Check failed: {e}"

    def input_format(self) -> InputFormat:
        return InputFormat.CPP

    def get_version(self) -> Optional[str]:
        try:
            dealii = _find_dealii()
        except DealiiRootOverrideError:
            # No version to report for an override that is not deal.II;
            # check_availability is the surface that explains why.
            return None
        if not dealii:
            return None
        # Try to read version from cmake config
        for f in dealii.rglob("deal.IIConfig.cmake"):
            text = f.read_text()
            for line in text.splitlines():
                if "DEAL_II_VERSION" in line and "SET" in line:
                    parts = line.split('"')
                    if len(parts) >= 2:
                        return parts[1]
        return "9.x (version detection failed)"

    def supported_physics(self) -> list[PhysicsCapability]:
        return [
            PhysicsCapability(
                name="element_survey",
                description="Finite-element survey: builds every FE_* class "
                            "this deal.II install declares and checks each one "
                            "(quadrature integrates the cell, mass matrix is "
                            "SPD, space reproduces a constant). Answers 'which "
                            "elements does this build actually have, and how "
                            "does each behave' without guessing from docs.",
                spatial_dims=[2, 3],
                element_types=["all"],
                template_variants=["default"],
            ),
            PhysicsCapability(
                name="poisson",
                description="Poisson / Laplace equation (step-3/6/7, with AMR, "
                            "L-domain, rectangle, 3D mixed Dirichlet-Neumann "
                            "MMS convergence study)",
                spatial_dims=[2, 3],
                element_types=["Q1", "Q2"],
                template_variants=["2d", "3d", "l_domain", "rectangle",
                                   "2d_adaptive", "3d_mixed_bc"],
            ),
            PhysicsCapability(
                name="linear_elasticity",
                description="Linear elasticity (step-8, with thick beam variant)",
                spatial_dims=[2],
                element_types=["Q1"],
                template_variants=["2d", "thick_beam"],
            ),
            PhysicsCapability(
                name="heat",
                description="Heat equation (transient step-26 and steady-state, with rectangle)",
                spatial_dims=[2],
                element_types=["Q1"],
                template_variants=["2d_transient", "2d_steady", "rectangle"],
            ),
            PhysicsCapability(
                name="stokes",
                description="Stokes flow (step-22, Taylor-Hood Q2/Q1, block preconditioner)",
                spatial_dims=[2],
                element_types=["Q2-Q1 (Taylor-Hood)"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="convection_diffusion",
                description="Convection-diffusion with SUPG stabilization (step-9 based)",
                spatial_dims=[2],
                element_types=["Q1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="nonlinear",
                description="Nonlinear PDE (minimal surface, step-15, Newton method)",
                spatial_dims=[2],
                element_types=["Q1"],
                template_variants=["2d_minimal_surface"],
            ),
            PhysicsCapability(
                name="helmholtz",
                description="Helmholtz equation (complex-valued, step-29 inspired)",
                spatial_dims=[2],
                element_types=["Q1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="eigenvalue",
                description="Eigenvalue problems via SLEPc (step-36 inspired)",
                spatial_dims=[2],
                element_types=["Q1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="wave",
                description="Wave equation with Newmark time integration (step-23 inspired)",
                spatial_dims=[2],
                element_types=["Q1"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="hp_adaptive",
                description="hp-adaptive FEM with automatic smoothness estimation (step-27 pattern)",
                spatial_dims=[2],
                element_types=["FE_Q(1..7)", "hp::FECollection"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="dg_transport",
                description="Discontinuous Galerkin for advection problems (step-12 pattern)",
                spatial_dims=[2],
                element_types=["FE_DGQ(1)"],
                template_variants=["2d"],
            ),
            PhysicsCapability(
                name="hyperelasticity",
                description="Finite-strain hyperelasticity with Neo-Hookean material (step-44 pattern)",
                spatial_dims=[3],
                element_types=["Q1"],
                template_variants=["3d"],
            ),
            PhysicsCapability(
                name="parallel_poisson",
                description="MPI-parallel Poisson solver with p4est (step-40 pattern)",
                spatial_dims=[2],
                element_types=["Q2"],
                template_variants=["2d"],
            ),
            # New physics
            PhysicsCapability("navier_stokes", "Navier-Stokes: stationary + transient (step-57, step-35)", [2, 3],
                              ["Q2-Q1 (Taylor-Hood)"], ["2d"]),
            PhysicsCapability("mixed_laplacian", "Mixed Laplacian with Raviart-Thomas H(div) (step-20)", [2],
                              ["FE_RaviartThomas + FE_DGQ"], ["2d"]),
            PhysicsCapability("time_dependent_heat", "Transient heat with AMR (step-26)", [2],
                              ["Q1"], ["2d"]),
            PhysicsCapability("time_dependent_wave", "Wave equation (step-23, step-48)", [2, 3],
                              ["Q1"], ["2d"]),
            PhysicsCapability("time_dependent_ns", "Transient Boussinesq flow (Boussinesq tutorials: step-31, step-32)", [2],
                              ["Q2-Q1"], ["2d"]),
            PhysicsCapability("matrix_free", "Matrix-free high-performance FEM (step-37, step-59)", [2, 3],
                              ["Q1-Q4 (tensor product)"], ["2d"]),
            PhysicsCapability("multigrid", "Geometric multigrid preconditioner (step-16, step-50)", [2, 3],
                              ["Q1-Q2"], ["2d"]),
            PhysicsCapability("obstacle_problem", "Variational inequality / contact (step-41)", [2],
                              ["Q1"], ["2d"]),
            PhysicsCapability("error_estimation", "Adaptive error estimation, Kelly + AMR (step-6, step-14)", [2],
                              ["Q1-Q2"], ["2d"]),
            PhysicsCapability("phase_field", "Phase-field / ADR with SUPG (step-63)", [2],
                              ["Q1"], ["2d"]),
            PhysicsCapability("dg_advection_reaction", "DG advection-reaction (step-12, step-39)", [2],
                              ["FE_DGQ"], ["2d"]),
            # ── 2026-06-01: three _DEALII_KNOWLEDGE keys had
            #    detailed pitfalls but no PhysicsCapability entry,
            #    so users browsing discover never saw them.
            #    Catalog content is distinct from the nearby
            #    similarly-named entries (dg_advection_reaction /
            #    obstacle_problem / hyperelasticity) — keep both
            #    surfaces. Closes task #69.
            PhysicsCapability(
                "advection_dg",
                "Pure DG advection (step-9, step-12). Distinct "
                "from dg_advection_reaction (step-12, step-39) — "
                "advection_dg covers step-9 transport without "
                "reaction term. DoFTools::make_flux_sparsity_"
                "pattern required for face coupling.",
                [2], ["FE_DGQ"], ["2d"]),
            PhysicsCapability(
                "contact",
                "Contact / variational inequalities (step-41, "
                "step-42). Active-set strategy. Related to "
                "obstacle_problem (the dealii backend's primary "
                "name for this class) — distinct deep_knowledge "
                "entry kept for active-set-strategy specifics.",
                [2, 3], ["Q1", "Q2"], ["2d"]),
            PhysicsCapability(
                "nonlinear_elasticity",
                "Nonlinear solid mechanics (step-44). Neo-"
                "Hookean three-field (u, p, J) formulation for "
                "quasi-incompressible materials. Distinct from "
                "hyperelasticity (broader catalog) — this entry "
                "focuses on the step-44 three-field method.",
                [3], ["Q1", "Q2"], ["3d"]),
        ]

    def get_knowledge(self, physics: str) -> dict:
        # ATTACHED ON EVERY RETURN PATH, VIA A WRAPPER, NOT PER-BRANCH.
        #
        # This method resolves through three catalogs and returns from four
        # places. Measured across the nine served backends, deal.II was the
        # only one whose payload told an agent no way to read its solution at a
        # point that is not a mesh node — and "the solve worked and was never
        # read back" is the largest single failure bucket across the
        # recorded runs, 60 runs at 12.9%. Adding the recipe to one branch
        # would have left the other three silent, which is the defect class
        # this file has already been repaired for twice.
        k = self._get_knowledge_inner(physics)
        if isinstance(k, dict):
            from backends.dealii.probe_recipe import DEALII_PROBE_RECIPE
            k = {**k, "probe_recipe": DEALII_PROBE_RECIPE}
        return k

    def _get_knowledge_inner(self, physics: str) -> dict:
        # Resolution order (2026-06-01 audit closes task #69):
        #
        #   1. data/dealii_knowledge.py:DEALII_KNOWLEDGE — the
        #      course-level catalog (overview/tutorials/etc.).
        #      Usually does NOT hold per-physics keys, but some
        #      entries do live here.
        #   2. generator-embedded KNOWLEDGE — the primary 96-pitfall
        #      source-of-truth that the dealii Tier-2 fixtures
        #      were built against. This is the catalog the
        #      cross-backend signal-verification test scores
        #      against.
        #   3. tools.deep_knowledge._DEALII_KNOWLEDGE — fallback
        #      ONLY for keys NOT in either of the above. This is
        #      where {advection_dg, contact, nonlinear_elasticity}
        #      live; without this fallback they were orphaned.
        try:
            import sys
            from core.paths import data_dir as _data_dir
            data_dir = str(_data_dir())
            if data_dir not in sys.path:
                sys.path.insert(0, data_dir)
            from dealii_knowledge import DEALII_KNOWLEDGE as deep
            if physics in deep:
                return deep[physics]
        except ImportError:
            pass
        # Primary fallback: generator-embedded knowledge.
        from backends.dealii.generators import get_knowledge
        gen_k = get_knowledge(physics)
        if isinstance(gen_k, dict) and gen_k.get("pitfalls"):
            return gen_k
        # Last fallback: tools.deep_knowledge per-physics catalog,
        # for entries (advection_dg / contact / nonlinear_
        # elasticity) that ONLY live in _DEALII_KNOWLEDGE.
        try:
            from tools.deep_knowledge import _DEALII_KNOWLEDGE
            if physics in _DEALII_KNOWLEDGE:
                tier3 = _DEALII_KNOWLEDGE[physics]
                # The always-served essentials block is injected by
                # the GENERATOR path only, so physics resolved here
                # used to be served without it: advection_dg, contact
                # and nonlinear_elasticity are advertised by
                # supported_physics() and do serve pitfalls (4/3/3),
                # yet reached the client with none of the Debug-vs-
                # Release scoping, required call order or JxW check —
                # 24 of 27 payloads carried the block, these 3 did
                # not. Inject on the same condition the generator
                # path uses, so an unknown physics still cannot look
                # covered.
                if isinstance(tier3, dict) and tier3.get("pitfalls"):
                    from backends.dealii.generators._critical import (
                        CRITICAL_KNOWLEDGE)
                    return {"deal_II_essentials": CRITICAL_KNOWLEDGE,
                            **tier3}
                return tier3
        except ImportError:
            pass
        return gen_k

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        from backends.dealii.generators import get_template
        key = f"{physics}_{variant}"
        generator = get_template(key)
        return generator(params)

    def validate_input(self, content: str) -> list[str]:
        """Reject not just malformed C++ but *silent-wrong* programs:
        a deal.II source that compiles and exits 0 while doing nothing
        (no mesh, no solve, no output). Until the 2026-06-26 overhaul,
        16 advanced generators emitted exactly such print-and-exit stubs
        and passed every check while producing no result file. The extra
        rules below make that class of program fail validation."""
        errors = []
        if "#include" not in content:
            errors.append("C++ source does not contain any #include directives")
        if "deal.II" not in content and "deal_II" not in content:
            errors.append("Source does not include deal.II headers")
        if "int main" not in content:
            errors.append("Source does not contain main()")

        # A real solver must build a mesh.
        if ("Triangulation" not in content
                and "GridGenerator" not in content
                and "GridIn" not in content):
            errors.append(
                "Source builds no mesh (no Triangulation / GridGenerator / "
                "GridIn) — it cannot solve anything. Looks like a "
                "placeholder/stub program.")

        # A real solver must write output we can collect (.vtu/.vtk/.pvd
        # via DataOut / DataOutBase, or a gnuplot dump).
        has_output = any(tok in content for tok in (
            "DataOut", "write_vtu", "write_vtk", "write_pvd",
            "write_gnuplot", "output_results", "DataOutBase"))
        if not has_output:
            errors.append(
                "Source produces no output (no DataOut/write_vtu/write_vtk/"
                "write_pvd/output_results) — results cannot be collected. "
                "Looks like a placeholder/stub program.")

        # Catch the classic stub whose main() only prints a 'see the
        # tutorial' message and returns. Heuristic: a main body that has
        # std::cout/printf but none of the verbs a real solve uses.
        lowered = content
        solve_tokens = (
            "distribute_dofs", "assemble", "solve(", ".solve",
            "vmult", "FEValues", "system_matrix", "cell_loop",
        )
        placeholder_phrases = (
            "see deal.II tutorial", "see step-", "for full implementation",
            "Placeholder", "placeholder:",
        )
        if not any(tok in lowered for tok in solve_tokens):
            errors.append(
                "Source never assembles or solves a system (no "
                "distribute_dofs/assemble/solve/vmult/FEValues) — it does "
                "no finite-element work. Looks like a placeholder/stub.")
        if any(p.lower() in lowered.lower() for p in placeholder_phrases):
            errors.append(
                "Source contains placeholder boilerplate ('see the "
                "tutorial' / 'full implementation' / 'Placeholder') — "
                "this is a stub, not a real solver.")
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        # Write source and CMakeLists
        src_path = work_dir / "main.cpp"
        src_path.write_text(input_content)

        cmake_content = _generate_cmakelists("fem_solve")
        (work_dir / "CMakeLists.txt").write_text(cmake_content)
        configure = [_cmake_executable() or "cmake", "."]
        config_dir = _dealii_config_dir(_find_dealii())
        if config_dir is not None:
            # CMake searches a CMAKE_PREFIX_PATH from the environment (spack
            # load, a Spack view, environment modules) before any HINTS, but
            # it takes deal.II_DIR as given.
            configure.insert(1, f"-Ddeal.II_DIR={config_dir}")

        job_id = str(uuid.uuid4())[:8]
        job = JobHandle(job_id=job_id, backend_name="dealii", work_dir=work_dir, status="running")

        start = time.time()

        # Refused before anything is compiled.
        if np > 1 and _dealii_without_mpi(config_dir):
            job.status = "failed"
            job.error = (f"np={np} was asked for, but this deal.II ({config_dir}) is built "
                         "without MPI (DEAL_II_WITH_MPI OFF): each of the processes would "
                         "solve the whole problem and write the same files. Run it with np=1.")
            job.elapsed = time.time() - start
            return job

        # Step 1: CMake configure
        try:
            proc = await asyncio.create_subprocess_exec(
                *configure,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
                start_new_session=True,
                stdin=asyncio.subprocess.DEVNULL,
            )

            # TIMEOUT MUST KILL THE SOLVER, AND THE WHOLE GROUP. Without this, a
            # timed-out solve kept running forever: wait_for() abandoned the
            # process but never terminated it, and a sweep found one such
            # solver 3.2 CPU-hours later at 100%% of a core, its MPI daemon
            # (orted) beside it. start_new_session puts the solver and every child
            # it spawns into their own process group, so one killpg reaps MPI
            # ranks too — the same idiom precice_config.py already uses, for the
            # same reason. The kill re-raises, so each backend's own TimeoutError
            # handling below is unchanged.
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            except asyncio.TimeoutError:
                import os as _os, signal as _signal
                try:
                    _os.killpg(proc.pid, _signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                await proc.wait()
                raise

            if proc.returncode != 0:
                job.status = "failed"
                job.error = f"CMake configure failed:\n{stderr.decode(errors='replace')}"
                job.elapsed = time.time() - start
                return job
        except asyncio.TimeoutError:
            job.status = "failed"
            job.error = "CMake configure timed out"
            job.elapsed = time.time() - start
            return job

        # Step 2: Make
        nproc = os.cpu_count() or 4
        try:
            proc = await asyncio.create_subprocess_exec(
                "make", f"-j{min(nproc, 8)}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
                stdin=asyncio.subprocess.DEVNULL,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode != 0:
                job.status = "failed"
                err = stderr.decode(errors='replace')
                hint = ""
                import platform
                if platform.system() == "Darwin" and (
                        "MacOSX.sdk" in err or "isinf" in err
                        or ("abs" in err and "ambiguous" in err)):
                    hint = ("\n\nHint (macOS + deal.II.app): this looks like the Xcode "
                            "SDK header conflict (a deal.II.app packaging issue, not an "
                            "openPASO bug). Make the sysroot consistent and re-run:\n"
                            "    export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
                job.error = f"Compilation failed:\n{err[-2000:]}{hint}"
                job.elapsed = time.time() - start
                return job
        except asyncio.TimeoutError:
            job.status = "failed"
            job.error = "Compilation timed out"
            job.elapsed = time.time() - start
            return job

        # Step 3: Run
        executable = work_dir / "fem_solve"
        if not executable.is_file():
            job.status = "failed"
            job.error = "Executable not found after compilation"
            job.elapsed = time.time() - start
            return job

        mpirun = shutil.which("mpirun")
        if np > 1 and mpirun:
            cmd = [mpirun, "-np", str(np), str(executable)]
        else:
            cmd = [str(executable)]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
                stdin=asyncio.subprocess.DEVNULL,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            job.elapsed = time.time() - start
            job.return_code = proc.returncode
            job.status = "completed" if proc.returncode == 0 else "failed"
            if proc.returncode != 0:
                # THE RETURN CODE AND THE SIGNAL, ALWAYS. This deal.II is a Release build
                # and asserts nothing: an FEValues accessor whose update flag was not
                # requested segfaults with an empty stderr, and a bare "failed" read as
                # an install fault (measured: a run gave up 0.4 min after one).
                job.error = (_describe_exit(proc.returncode) + "\n"
                             + stderr.decode(errors="replace")[-2000:]).strip()
            (work_dir / "stdout.log").write_text(stdout.decode(errors="replace"))
            (work_dir / "stderr.log").write_text(stderr.decode(errors="replace"))
        except asyncio.TimeoutError:
            job.status = "failed"
            job.elapsed = timeout
            job.error = f"Execution timed out after {timeout}s"
        except Exception as e:
            job.status = "failed"
            job.elapsed = time.time() - start
            job.error = str(e)

        return job

    def get_result_files(self, job: JobHandle) -> list[Path]:
        results = []
        for ext in ["*.vtu", "*.pvd", "*.vtk", "*.gnuplot", "*.gpl"]:
            results.extend(job.work_dir.rglob(ext))
        return sorted_by_step(results)


def _dealii_config_dir(root: Optional[Path]) -> Optional[Path]:
    """The directory holding deal.IIConfig.cmake for a deal.II the finder
    reported: an install keeps it in lib/cmake/deal.II, a source tree in its
    build directory."""
    if root is None:
        return None
    for sub in ("lib/cmake/deal.II", "build/lib/cmake/deal.II", "build",
                "build/release/lib/cmake/deal.II", "build/release",
                "build/Release/lib/cmake/deal.II", "build/Release",
                "install/lib/cmake/deal.II", "share/deal.II/cmake"):
        cfg = root / sub
        if (cfg / "deal.IIConfig.cmake").exists():
            return cfg
    return None


def _dealii_without_mpi(config_dir: Optional[Path]) -> bool:
    """True when deal.IIConfig.cmake says the build has no MPI. False when it
    has MPI or cannot be read: then mpirun is used as before."""
    if config_dir is None:
        return False
    try:
        text = (config_dir / "deal.IIConfig.cmake").read_text(errors="replace")
    except OSError:
        return False
    return re.search(r"^\s*set\(DEAL_II_WITH_MPI OFF\)", text, re.M) is not None


def _generate_cmakelists(target_name: str) -> str:
    # The deal.II a template compiles against is the one _find_dealii()
    # reports: it checks DEAL_II_DIR, then DEALII_ROOT, then the usual places.
    # This used to read DEALII_ROOT first, so with both set, discover reported
    # DEAL_II_DIR's deal.II while every template compiled against DEALII_ROOT's
    # (measured: "deal.II 9.7.1 at <spack prefix>" reported, "Using the
    # deal.II-9.8.0-pre build directory" compiled).
    #
    # The hint points at the directory that ACTUALLY holds
    # deal.IIConfig.cmake. For a SOURCE tree the config lives under
    # build/lib/cmake/deal.II (not <root>/lib/cmake/deal.II), so a bare
    # root hint makes find_package silently fall back to a *system*
    # deal.II of a different (older) version — e.g. advertising a 9.8
    # source build but compiling against /usr's 9.1.1, which breaks every
    # template that uses a post-9.1 API (issue #39 class). Without any hint,
    # cmake aborts with "Could not find a package configuration file provided
    # by 'deal.II'" on conda-forge installs, which discover still reports
    # AVAILABLE (audit 2026-06-01).
    extra_hints = ""
    discovered = _find_dealii()
    if discovered is not None:
        extra_hints = f" {_dealii_config_dir(discovered) or discovered}"

    # Honour CC/CXX from the environment so that conda-forge deal.II
    # packages (whose deal.IIConfig.cmake bakes in a feedstock-only
    # compiler path like
    # /home/conda/feedstock_root/build_artifacts/.../x86_64-conda_cos6-linux-...)
    # do not force the build to use a non-existent toolchain when
    # `deal_ii_initialize_cached_variables()` runs (that macro
    # unconditionally writes the cache without FORCE, so any pre-seeded
    # cache entry wins).
    #
    # Pre-seed without FORCE and only when not already defined so an
    # explicit `-DCMAKE_CXX_COMPILER=…` on the user's cmake command line
    # still wins.  Use CACHE STRING to match deal.II's own macro type so
    # CMake does not emit a type-mismatch warning.  Use plain
    # `if(NOT DEFINED CMAKE_*_COMPILER)` rather than the `CACHE{}`
    # operand form, which only works on CMake >= 3.14 (the file declares
    # a minimum of 3.13.4).  A `-D` from the command line is visible as a
    # regular variable too, so this still respects user overrides.
    cc = os.environ.get("CC", "")
    cxx = os.environ.get("CXX", "")
    compiler_cache = ""
    if cc:
        compiler_cache += (
            f'if(NOT DEFINED CMAKE_C_COMPILER)\n'
            f'  set(CMAKE_C_COMPILER "{cc}" CACHE STRING "C compiler")\n'
            f'endif()\n'
        )
    if cxx:
        compiler_cache += (
            f'if(NOT DEFINED CMAKE_CXX_COMPILER)\n'
            f'  set(CMAKE_CXX_COMPILER "{cxx}" CACHE STRING "C++ compiler")\n'
            f'endif()\n'
        )

    return f"""\
cmake_minimum_required(VERSION 3.13.4)
{compiler_cache}find_package(deal.II 9.0 REQUIRED
  HINTS ${{DEAL_II_DIR}} ${{deal.II_DIR}}{extra_hints} /usr /usr/local
)
deal_ii_initialize_cached_variables()
project({target_name})
add_executable({target_name} main.cpp)
deal_ii_setup_target({target_name})
"""


def register():
    register_backend(
        DealiiBackend(),
        aliases=["deal.ii", "deal_ii", "dealii", "deal"],
    )
