"""
FEBio solver backend.

FEBio is an open-source FEM code for biomechanics. Uses XML input files (.feb).
Specialized for soft tissue mechanics, biphasic/multiphasic problems, and
biological applications.

FEBio website: https://febio.org
GitHub: https://github.com/febiosoftware/FEBio
"""

import asyncio
import logging
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
from .generators import GENERATORS as _TEMPLATES, KNOWLEDGE as _FEBIO_KNOWLEDGE

logger = logging.getLogger("openpaso.febio")


class FebioBinaryOverrideError(RuntimeError):
    """FEBIO_BINARY names something that is not a file.

    Raised rather than ignored, because silently searching elsewhere means the
    binary openPASO tests is not the binary the user named.
    """


_FEBIO_IDENT_CACHE: dict[str, tuple[bool, str]] = {}


def _identifies_as_febio(binary) -> tuple[bool, str]:
    """Does this executable actually identify itself as FEBio?

    `-info` prints `FEBio version  = <x>` before its banner — measured on
    4.12.0.86045466d. Two details are load-bearing and both were learned the
    hard way on the 4C equivalent:

    stdin MUST be closed. `-info` falls into FEBio's interactive prompt on an
    open stdin and hangs; and under an MCP stdio server the inherited stdin is
    the JSON-RPC stream, so a probe that reads it consumes the protocol.

    ValueError is NOT caught. `UnicodeDecodeError` is a `ValueError`, so
    catching it in the cannot-look branch sent every binary with non-text
    output down the fail-open path and ACCEPTED it. Decoding is explicit.

    Fails OPEN on a genuine inability to look — a check that cannot run must not
    condemn a working install. Cached per path, because `check_availability` is
    called by `discover` and every knowledge surface.
    """
    import subprocess

    key = str(binary)
    if key in _FEBIO_IDENT_CACHE:
        return _FEBIO_IDENT_CACHE[key]

    verdict: tuple[bool, str]
    try:
        r = subprocess.run([key, "-info"], capture_output=True, timeout=25,
                           stdin=subprocess.DEVNULL)
        blob = (r.stdout + r.stderr).decode("utf-8", errors="replace")
        if "FEBio version" in blob:
            verdict = (True, "identified itself")
        elif not blob.strip():
            verdict = (False, "-info produced no output")
        else:
            verdict = (False, "its -info output carries no `FEBio version` "
                              "line: " + " ".join(blob.split())[:120])
    except (subprocess.TimeoutExpired, OSError) as exc:
        verdict = (True, f"identity not checked ({type(exc).__name__})")

    _FEBIO_IDENT_CACHE[key] = verdict
    return verdict


def _find_febio_binary() -> Optional[Path]:
    """Locate the FEBio binary."""
    # An EXPLICIT override that does not resolve is an error, not a hint. This
    # used to fall through silently to the search path, so a user with a stale
    # or mistyped FEBIO_BINARY got a DIFFERENT binary tested than the one they
    # named, with nothing said — and then debugged the wrong install. It also
    # invalidated an audit's own acceptance test: setting
    # FEBIO_BINARY=/nonexistent to check that fixtures go red instead found the
    # real binary and passed, so the verification measured nothing.
    env_path = os.environ.get("FEBIO_BINARY")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        raise FebioBinaryOverrideError(
            f"FEBIO_BINARY is set to {env_path!r}, which is not a file. "
            f"Refusing to fall back to a different binary: an explicit "
            f"override that silently resolves elsewhere is how you end up "
            f"debugging an install you are not running. Fix the path or unset "
            f"the variable.")

    # Common locations
    candidates = [
        Path.home() / "FEBio" / "bin" / "febio4",
        Path.home() / "FEBioStudio" / "bin" / "febio4",
        Path("/opt/febio/bin/febio4"),
        Path("/usr/local/bin/febio4"),
    ]
    for c in candidates:
        if c.is_file():
            return c

    p = shutil.which("febio4") or shutil.which("febio3") or shutil.which("febio")
    if p:
        return Path(p)
    # A FEBio built by `openpaso install febio --via spack-agent` (the recipe is `febio` in the
    # openpaso namespace) sits in a hashed prefix; Spack knows where. Asked last, because it
    # starts a process where every step above costs a stat.
    from core.spack import installed_prefix   # noqa: PLC0415
    prefix = installed_prefix("febio")
    if prefix is not None and (prefix / "bin" / "febio4").is_file():
        return prefix / "bin" / "febio4"
    return None


_BODY_FORCE_TRAP = (
    "A BODY FORCE IN FEBio IS NOT THE FORCE YOU WROTE. "
    "<body_load type=\"body force\"> assembles -H[a]*density*f*J0 and is "
    "assembled the way INTERNAL forces are, so <force>f</force> applies a "
    "physical body force of MINUS f, and it is a SPECIFIC force: the value is "
    "multiplied by the material's <density>, so the same number means "
    "different loads in different materials. Measured on this install: the "
    "deck route and an equivalent consistent nodal load agree to 1e-15 ONLY "
    "after negating the deck value. "
    "FEBio's MATH PARSER HAS NO ** OPERATOR, and this is the single most "
    "likely way a manufactured source term fails. Problem statements, sympy "
    "and numpy all write powers as x**2; FEBio's parser wants x^2, or just "
    "write the multiplication out. Pasting a stated source term verbatim "
    "gives `Error evaluating math expression: Token expected (position N)`, "
    "where N is a CHARACTER OFFSET and never names the operator — so it reads "
    "as a mysterious parse failure, or as the expression being too long. It "
    "is not: measured 2026-08-30, `-1*X**2` fails at position 6, while "
    "`-1*X^2`, `-1*X*X`, a five-term polynomial "
    "(-1*X*X*X*X*Y + 2*X*X*X - 3*X*X*Y*Y + 4*X*Y - 5*Y*Y*Y) and `-1*X*sin(t)` "
    "all reach normal termination. Length is fine and time dependence is "
    "fine; the two asterisks are the whole problem. Convert every ** before "
    "you put an expression in a deck. (Uppercase X, Y, Z are the reference "
    "coordinates — see the math-symbol pitfall. Lowercase x, y, z are not: "
    "in a `body force` <force type=\"math\"> the run stops at initialisation "
    "with `Don't understand variable name \"x\" in math expression.`, and in "
    "a `non-const` load they read that load's own <x>/<y>/<z> components, so "
    "a component that names itself segfaults (exit 139) at the first time "
    "step and one that names another component silently takes that "
    "component's value.)\n"
    "A BODY LOAD LIVES IN <Loads>, NOT IN <LoadData>. The two sections sound "
    "interchangeable and are not: <Loads> holds the loads themselves "
    "(body_load, nodal_load, surface_load) and <LoadData> holds ONLY the "
    "<load_controller> curves that scale them. A <body_load> placed inside "
    "<LoadData> is a HARD failure — `Reading file ... FAILED!` with `tag "
    "\"body_load\" (line N) : unrecognized tag`, which names the tag but not "
    "the reason, so it reads as \"FEBio has no such tag\" rather than \"that "
    "tag is in the wrong section\".\n"
    "THE DANGEROUS STEP IS WHAT COMES NEXT. The obvious response to an "
    "unrecognized tag is to delete it, and a deck with no load DOES NOT "
    "RELIABLY COMPLAIN. Measured 2026-08-30 on a minimal clamped block: with "
    "the load in <Loads> the peak displacement is 3.99e-04; with the load "
    "removed the run reaches N O R M A L   T E R M I N A T I O N and returns "
    "9.19e-17, machine zero, with one line — `No force acting on the system` "
    "— buried in pages of banner. (On a stiffer deck the same deletion "
    "error-terminates instead, so this is deck-dependent: the point is that "
    "a missing load may or may not announce itself, and you cannot rely on "
    "it doing so.) Treat that warning as an error: it means you are solving a "
    "different problem from the one you were given. This cost a real run — "
    "the agent hit the unrecognized tag, removed the body_load, saw the "
    "warning, wrote \"the run is working but there's a warning\", and "
    "concluded that \"FEBio's XML input format does not appear to support "
    "body_load\". It does; the tag was one section over.\n"
    "A POSITION-DEPENDENT BODY FORCE IS WRITABLE IN THE DECK, and this entry "
    "used to say the opposite — \"a math string cannot make it vary per "
    "element, so a POSITION-DEPENDENT source cannot be written in the deck at "
    "all\". That is false, and agents believed it: FB2 runs reported back "
    "\"FEBio 4.12.0 does not support spatially-varying body forces through "
    "its XML interface\" and stopped. Two forms work, and they give "
    "bit-identical answers:\n"
    "  <body_load type=\"body force\">\n"
    "    <force type=\"math\">fx(X,Y,Z), fy, fz</force>\n"
    "  </body_load>\n"
    "  <body_load type=\"non-const\"><x>fx</x><y>fy</y><z>fz</z></body_load>\n"
    "THE type=\"math\" ATTRIBUTE IS NOT OPTIONAL, AND OMITTING IT IS USUALLY "
    "SILENT. `force` is a FEParamVec3 and the expression is read by the "
    "FEMathValueVec3 valuator, registered under \"math\". Without the "
    "attribute the string is read as a plain vec3, and what happens then "
    "depends on WHICH COMPONENT carries the expression:\n"
    "  <force>-1*X, 0, 0</force>   -> hard failure, `Reading file ... "
    "FAILED!` with `syntax error (line N)`, because the first token cannot "
    "be read as a number at all\n"
    "  <force>0, 0, -1*X</force>   -> NORMAL TERMINATION, no warning, and the "
    "load is the CONSTANT (0, 0, -1): the numeric prefix is taken and `*X` is "
    "discarded\n"
    "Measured 2026-08-30: the second form gives per-element sz of -1.0627e-4, "
    "2.7784e-4, 2.4880e-4, 8.4147e-5 — bit-identical to an explicit constant "
    "(0,0,-1) — against -1.3652e-3, 1.8972e-3, 2.3403e-3, 9.2593e-4 with the "
    "attribute present. So the usual mistake does not announce itself; it "
    "silently replaces your source term with a uniform load, and the only "
    "symptom is a manufactured-solution study whose order collapses. If you "
    "are debugging a convergence order that will not come out, check this "
    "attribute before you doubt the mathematics. \"non-const\" is a "
    "FEBio-2 class marked obsolete since 3.0 but still registered and still "
    "correct; \"const\" likewise. Measured 2026-08-30 on FEBio 4.12.0, four "
    "hex8 elements along X clamped at x=0: the math and non-const forms both "
    "give per-element sz of -1.365e-3, 1.897e-3, 2.340e-3, 9.258e-4, agreeing "
    "to every digit, while a constant force of the same mean gives -4.251e-4, "
    "1.111e-3, 9.952e-4, 3.366e-4 — a different spatial profile, not a "
    "rescaling, so the expression really is evaluated per element. "
    "WHAT DOES BITE: an over-large body force makes the nonlinear solve "
    "diverge with `failed to converge at time : 1`, which reads like a deck "
    "problem and is not one. Check the deflection is small before concluding "
    "the load was rejected. "
    "The consistent nodal-load route below remains valid and is still the "
    "right choice when you need the load vector itself (compute "
    "F_i = int b.phi_i dV and apply it as a nodal_force map — "
    "data/coupling_participants/participant_febio_elastic.py does exactly "
    "this, with Gauss quadrature over the hex8 elements), but it is now an "
    "alternative rather than the only way through. "
    "ON SEPARABLE LOADS, which is a genuine convenience either way: a load "
    "of the form f(x,y)*g(t) — and every manufactured solution's source in "
    "practice factors this way or is a short sum of such terms — needs NO "
    "time-varying deck mechanism at all. Put the SPATIAL part f(x,y) into the "
    "consistent nodal load map (computed once, as above), and put the TIME "
    "part g(t) into the load controller curve that scales the map: "
    "<load_controller type=\"loadcurve\"> with <points> sampling g(t) at "
    "each output time, interpolate LINEAR (or SMOOTH for C1). That is the "
    "standard FEBio idiom — the shipped participant's LoadData block does "
    "exactly this. A sum of separable terms is one nodal_load per term, each "
    "with its own curve. 'The deck cannot express it' is true only of a "
    "genuinely non-separable f(x,y,t), and even there a per-timestep restart "
    "loop works before you declare a task impossible. "
    "If you take that route, remember FEBio's reported reaction Rx/Ry is "
    "m_Fr, accumulated only on the ELEMENT path: FENodalLoad::LoadVector "
    "calls the scalar Assemble(), which at a prescribed dof adds to nothing. "
    "So a nodal load never reaches the reported reaction, and any use of "
    "r = A u - b must subtract the consistent load itself. Skipping that "
    "correction leaves a spurious FIRST-ORDER error in a recovered traction "
    "that is otherwise exact.")

# Merged into every physics row, because a mechanics fact filed under one row
# is invisible to the other eleven and this trap applies to all of them.
_CROSS_CUTTING = {
    "body_force_trap": _BODY_FORCE_TRAP,
}


class FebioBackend(SolverBackend):

    def name(self) -> str:
        return "febio"

    def display_name(self) -> str:
        return "FEBio"

    def check_availability(self) -> tuple[BackendStatus, str]:
        try:
            binary = _find_febio_binary()
        except FebioBinaryOverrideError as exc:
            # Report as MISCONFIGURED rather than propagating: `discover` must
            # keep working and list the other backends, but this one must not
            # read as available or as simply absent.
            return BackendStatus.MISCONFIGURED, str(exc)
        if not binary:
            return BackendStatus.NOT_INSTALLED, (
                "FEBio binary not found. Install from https://febio.org/downloads/, "
                "or build from source (github.com/febiosoftware/FEBio: cmake "
                "-DUSE_MKL=OFF -DCMAKE_EXE_LINKER_FLAGS='-fopenmp -ldl' "
                "-DCMAKE_SHARED_LINKER_FLAGS='-fopenmp -ldl' && make; verified "
                "working recipe, see elasticity_mms KNOWLEDGE) and symlink to "
                "~/FEBio/bin/febio4, or set FEBIO_BINARY env var."
            )
        # Confirm the binary IS FEBio. This check previously accepted any
        # existing executable, so `FEBIO_BINARY=/bin/true` reported
        # "available — FEBio at /bin/true" — the same defect found and fixed in
        # 4C. An agent asks `discover`, is told the backend works, and every run
        # then fails for a cause the availability report has already excluded.
        #
        # `-info` prints `FEBio version  = <x>`, measured on 4.12. stdin is
        # closed because `-info` falls into FEBio's interactive prompt on an open
        # stdin and hangs — and under an MCP stdio server the inherited stdin is
        # the JSON-RPC stream, so the probe would consume the protocol.
        ident_ok, ident_why = _identifies_as_febio(binary)
        if not ident_ok:
            return BackendStatus.MISCONFIGURED, (
                f"the binary at {binary} does not identify itself as FEBio "
                f"({ident_why}). Point FEBIO_BINARY at a real FEBio build; note "
                f"`pip install febio` installs an unrelated third-party wrapper, "
                f"not FEBio.")
        return BackendStatus.AVAILABLE, f"FEBio at {binary}"

    def input_format(self) -> InputFormat:
        return InputFormat.XML

    def get_version(self) -> Optional[str]:
        binary = _find_febio_binary()
        if not binary:
            return None
        import subprocess
        try:
            r = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL)
            return r.stdout.strip() or r.stderr.strip()
        except Exception:
            return None

    def supported_physics(self) -> list[PhysicsCapability]:
        return [
            PhysicsCapability(
                name="linear_elasticity",
                description="Linear elasticity (small strain solid mechanics)",
                spatial_dims=[3],
                element_types=["hex8", "tet4", "tet10"],
                template_variants=["3d_cube"],
            ),
            PhysicsCapability(
                name="elasticity_mms",
                description=("3D linear-elasticity manufactured-solution "
                             "(MMS) verification family — structured hex8 "
                             "cube [0,L]^3, exact trig body force "
                             "-div(sigma(u*)), per-node Dirichlet maps, "
                             "displacement L2 order 2 expected"),
                spatial_dims=[3],
                element_types=["hex8"],
                template_variants=["3d_cube_hex8"],
            ),
            PhysicsCapability(
                name="hyperelasticity",
                description="Nonlinear hyperelasticity (Neo-Hookean, Mooney-Rivlin)",
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_cube"],
            ),
            PhysicsCapability(
                name="biphasic",
                description="Biphasic poroelasticity (solid + fluid phases)",
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_confined"],
            ),
            PhysicsCapability(
                name="heat",
                description="Heat conduction (steady-state)",
                spatial_dims=[3],
                element_types=["hex8"],
                template_variants=["3d_bar"],
            ),
            # Added 2026-09-19. FEBio registers 'solute' (Biphasic Solute
            # Analysis) as a module in its own right, between 'biphasic' and
            # 'multiphasic'. openPASO reached the two neighbours and not this
            # one. Derived from the multiphasic template and verified by
            # execution on FEBio 4.12.0: NORMAL TERMINATION with defaults and
            # again with E and diffusivity overridden.
            PhysicsCapability(
                name="solute",
                description=("Biphasic poroelasticity carrying EXACTLY ONE "
                             "solute (FEBio module 'solute'); use "
                             "'multiphasic' for several solutes or a fixed "
                             "charge density"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_diffusion"],
            ),
            PhysicsCapability(
                name="multiphasic",
                description=("Biphasic poroelasticity + solute transport "
                             "(charged-hydrated cartilage, electrolyte "
                             "diffusion, drug delivery)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_diffusion"],
            ),
            PhysicsCapability(
                name="fluid",
                description=("Incompressible Newtonian fluid via FEBio's "
                             "pressure-velocity fluid solver "
                             "(cardiovascular CFD)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_channel"],
            ),
            PhysicsCapability(
                name="fluid_fsi",
                description=("Strongly-coupled monolithic FSI "
                             "(arterial wall hemodynamics, cardiac "
                             "chamber dynamics, valve modeling)"),
                spatial_dims=[3],
                element_types=["hex8"],
                template_variants=["3d_block"],
            ),
            PhysicsCapability(
                name="rigid_body",
                description=("Rigid-body material (impactors, fixtures, "
                             "articulating joints, contact prescription)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_pushdown"],
            ),
            PhysicsCapability(
                name="viscoelasticity",
                description=("Prony-series viscoelastic stress relaxation / "
                             "creep response (cartilage, ligament, tendon)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_stress_relax"],
            ),
            PhysicsCapability(
                name="plasticity",
                description=("Rate-independent plasticity (J2 / Hill / "
                             "user-curve hardening) — cortical bone, "
                             "metal implants, surgical tools"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_uniaxial"],
            ),
            PhysicsCapability(
                name="fiber_reinforced",
                description=("Anisotropic fiber-reinforced hyperelasticity "
                             "(HGO, transversely isotropic) — arterial "
                             "wall, ligament, tendon, myocardium"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_hgo"],
            ),
            PhysicsCapability(
                name="active_contraction",
                description=("Active contractile fibers on a passive "
                             "elastic base (cardiac chamber, skeletal "
                             "muscle, peristalsis)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_fiber"],
            ),
            PhysicsCapability(
                name="biphasic_fsi",
                description=("Coupled biphasic tissue + free-fluid FSI "
                             "(blood-tissue perfusion, drug elution, "
                             "cartilage-synovial fluid)"),
                spatial_dims=[3],
                element_types=["hex8"],
                template_variants=["3d_block"],
            ),
            PhysicsCapability(
                name="polar_fluid",
                description=("Micropolar (Cosserat) fluid with "
                             "independent micro-rotation DOFs "
                             "(blood-rheology, polymer suspensions, "
                             "near-wall turbulence corrections)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_channel"],
            ),
            PhysicsCapability(
                name="damage",
                description=("Continuum damage mechanics — progressive "
                             "stiffness degradation under repeated "
                             "loading (tissue tearing, cartilage "
                             "wear, elastomer fatigue)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_cycle"],
            ),
            PhysicsCapability(
                name="growth_remodeling",
                description=("Multiplicative growth-and-remodeling "
                             "F = F_e * F_g (vascular adaptation, "
                             "tissue scaffolds, muscle hypertrophy, "
                             "tumor mechanobiology)"),
                spatial_dims=[3],
                element_types=["hex8", "tet4"],
                template_variants=["3d_isotropic"],
            ),
        ]

    def get_knowledge(self, physics: str) -> dict:
        # THE DECK GRAMMAR GOES OUT WITH EVERY PHYSICS, INCLUDING THE EMPTY
        # CASE.
        #
        # Measured on the `heat` payload (21,787 characters): no `<febio_spec`,
        # no `<MeshDomains`, no `<Boundary`, no `<node id=`, no `fix=`. The
        # corpus is rich on material models and pitfalls and silent on the
        # document that carries them, so an agent cannot begin. Same shape of
        # gap as 4C, where it cost three recorded runs of one coupled
        # problem their whole attempt.
        #
        # It is attached even when there is no per-physics entry: an unknown
        # physics is exactly when the agent most needs to know how a deck is
        # shaped, and returning {} taught it nothing at all.
        try:
            from backends.febio.deck_grammar import FEBIO_DECK_GRAMMAR
        except Exception:                                # pragma: no cover
            FEBIO_DECK_GRAMMAR = ""
        kn = _FEBIO_KNOWLEDGE.get(physics)
        out = dict(kn) if kn else {}
        if kn:
            out.update(_CROSS_CUTTING)
        if FEBIO_DECK_GRAMMAR:
            out["deck_grammar"] = FEBIO_DECK_GRAMMAR
        return out

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        key = f"{physics}_{variant}"
        generator = _TEMPLATES.get(key)
        if not generator:
            raise ValueError(f"No FEBio template for {key}. "
                             f"Available: {list(_TEMPLATES.keys())}")
        return generator(params)

    def validate_input(self, content: str) -> list[str]:
        errors = []
        if "<febio_spec" not in content:
            errors.append("Missing <febio_spec> root element")
        if "<Material>" not in content and "<Material " not in content:
            errors.append("Missing Material section")
        if "<Geometry>" not in content and "<Mesh>" not in content:
            errors.append("Missing Geometry/Mesh section")
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        binary = _find_febio_binary()
        if not binary:
            return JobHandle(
                job_id=str(uuid.uuid4())[:8],
                backend_name="febio",
                work_dir=work_dir,
                status="failed",
                error="FEBio binary not found",
            )

        work_dir.mkdir(parents=True, exist_ok=True)
        input_file = work_dir / "input.feb"
        input_file.write_text(input_content)

        cmd = [str(binary), "-i", str(input_file)]
        job_id = str(uuid.uuid4())[:8]
        job = JobHandle(job_id=job_id, backend_name="febio", work_dir=work_dir, status="running")

        start = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
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
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                import os as _os, signal as _signal
                try:
                    _os.killpg(proc.pid, _signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                await proc.wait()
                raise

            job.elapsed = time.time() - start
            job.return_code = proc.returncode
            job.status = "completed" if proc.returncode == 0 else "failed"
            if proc.returncode != 0:
                job.error = stderr.decode(errors="replace")[-2000:]
            (work_dir / "stdout.log").write_text(stdout.decode(errors="replace"))
            (work_dir / "stderr.log").write_text(stderr.decode(errors="replace"))
        except asyncio.TimeoutError:
            job.status = "failed"
            job.elapsed = timeout
            job.error = f"Timed out after {timeout}s"
        except Exception as e:
            job.status = "failed"
            job.elapsed = time.time() - start
            job.error = str(e)

        return job

    def get_result_files(self, job: JobHandle) -> list[Path]:
        results = []
        for ext in ["*.xplt", "*.vtk", "*.vtu", "*.log"]:
            results.extend(job.work_dir.rglob(ext))
        return sorted_by_step(results)



def register():
    register_backend(
        FebioBackend(),
        aliases=["febio", "FEBio", "febio4"],
    )
