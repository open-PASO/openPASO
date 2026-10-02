"""
4C Multiphysics solver backend.

Self-contained 4C interface with 10 physics generators, domain knowledge,
and input validation. Uses YAML input files (.4C.yaml).
Generators are at backends/fourc/generators/ (10 physics modules).
"""

import asyncio
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from core.user_dirs import desktop_dirs as _desktop_dirs

from core.backend import (
    sorted_by_step,
    SolverBackend, BackendStatus, InputFormat,
    PhysicsCapability, JobHandle,
)
from core.registry import register_backend

logger = logging.getLogger("openpaso.fourc")

# Path resolution
FOURC_ROOT = Path(os.environ["FOURC_ROOT"]) if os.environ.get("FOURC_ROOT") else None


_FOURC_IDENT_CACHE: dict[str, tuple[bool, str]] = {}


# EVERY MARKER HERE WAS TAKEN FROM A REAL 4C FAILURE ON THIS MACHINE.
#
# The first version of this regex knew only `PROC n ERROR` and friends, and so
# it MISSED the very failure that motivated it: 4C's YAML parse error, which is
# a bare `ERROR:` at line start followed by a line:col location,
#
#     ERROR: could not find ':' colon after key
#     53:17: "NODE 7 DLINE 1"  (size=16)
#
# i.e. the fix did not reach the case it was built for. Caught only by running
# the failing deck and feeding its real log through this function.
_FOURC_ERROR_RE = re.compile(
    r"^.*?(?:PROC\s+\d+\s+ERROR"
    r"|^ERROR:"                      # YAML/input parse errors
    r"|^\d+:\d+:\s"                 # the line:col that follows one
    r"|Could not match this input"
    r"|terminate called after throwing|Caught exception"
    r"|\bFOUR_C_THROW\b|Segmentation fault|dserror)",
    re.M | re.I)

# Lines 4C prints on EVERY run, successful ones included, so they never explain
# a failure. `Invalid MIT-MAGIC-COOKIE-1 key` is an X11 authority warning from
# the display, not the solver: `4C -p` prints it and then dumps the whole input
# grammar successfully. Reporting it as the error is what convinced two agents
# the binary was broken.
_FOURC_NOISE = ("Invalid MIT-MAGIC-COOKIE-1 key",)


def _fourc_diagnostic(stdout_text: str, stderr_text: str,
                      window: int = 1600) -> str:
    """4C's real error, found by CONTENT rather than by position.

    4C prints its diagnostic BEFORE the MPI_ABORT boilerplate, so the last N
    characters of stdout are always the boilerplate and never the cause. This
    locates the first genuine error marker and returns the window that follows
    it, which is where the offending input block is printed.

    The tails are appended afterwards as a fallback, because a crash with no
    marker at all (a signal, an MPI-level failure) still needs reporting.
    """
    parts = []
    # A SEGFAULT IS NOT A DIAGNOSTIC, SO NAME THE CAUSE THAT PRODUCES IT.
    #
    # Measured on one recorded run that was lost entirely to this. 4C dies
    # with "Signal: Segmentation fault (11) / Address code: Address not mapped"
    # and prints no error, no line number and no mention of conditions. The
    # crash lands during "Read/generate conditions", so it reads as a problem
    # with the condition's CONTENT — and the run responded by rewriting section
    # names, swapping DESIGN LINE TRANSPORT DIRICH for DESIGN LINE DIRICH, and
    # changing the element TYPE, five attempts that all crashed identically.
    #
    # The cause was one digit. Design entity ids are ONE-based; `E: 0` with
    # `NODE n DLINE 0` indexes past the end of the array. Verified by running
    # that agent's own deck twice with only that digit changed: E:0/DLINE 0
    # segfaults at exit 139, E:1/DLINE 1 finishes normally at exit 0.
    blob = (stdout_text or "") + "\n" + (stderr_text or "")
    if re.search(r"Signal:\s*Segmentation fault|signal 11|SIGSEGV", blob):
        parts.append(
            "--- 4C crashed with a signal, not an error message ---\n"
            "A segmentation fault carries no diagnostic, so read the deck, not "
            "the log. In this codebase the commonest cause by far is a "
            "ZERO-BASED DESIGN ENTITY ID: `E:` in a condition block and the "
            "DLINE/DNODE/DSURF number in the topology block are ONE-based, and "
            "`E: 0` with `NODE n DLINE 0` indexes past the end of the array and "
            "dies with exactly this signal. Measured on one real deck: "
            "E:0/DLINE 0 -> Segmentation fault, exit 139; the SAME deck with "
            "E:1/DLINE 1 -> 'processor 0 finished normally', exit 0.\n"
            "Check that digit BEFORE rewriting section names, swapping "
            "DESIGN LINE TRANSPORT DIRICH for DESIGN LINE DIRICH, or changing "
            "the element TYPE: each of those leaves the fault in place and "
            "crashes identically.")
    for name, text in (("stdout", stdout_text), ("stderr", stderr_text)):
        m = _FOURC_ERROR_RE.search(text or "")
        if m:
            block = text[m.start():m.start() + window].strip()
            parts.append(f"--- 4C diagnostic ({name}) ---\n{block}")
    if not parts:
        noise_only = all(
            (not (stdout_text or "").strip()
             or all(n in stdout_text for n in _FOURC_NOISE))
            for _ in (0,))
        hint = ("\nNOTE: 4C aborted without printing a recognised diagnostic. "
                "The `Invalid MIT-MAGIC-COOKIE-1 key` line, if present, is an "
                "X11 warning that 4C prints on SUCCESSFUL runs too and never "
                "explains a failure. Read work_dir/stdout.log from the TOP."
                if noise_only else "")
        return ((stderr_text or "")[-1000:] + "\n--- stdout tail ---\n"
                + (stdout_text or "")[-1000:])[-2000:] + hint
    parts.append("--- stderr tail ---\n" + (stderr_text or "")[-400:])
    parts.append("--- stdout tail ---\n" + (stdout_text or "")[-400:])
    return "\n".join(parts)


def _identifies_as_fourc(binary) -> tuple[bool, str]:
    """Does this executable actually identify itself as 4C?

    Cheap and conservative. 4C prints its usage to stderr and exits non-zero
    when run with no arguments, and that text names the program — so the check
    is "run it with no arguments and look for 4C's own vocabulary". Anything
    that produces neither is not 4C.

    Deliberately fails OPEN on an inability to look (timeout, permission,
    OSError): a check that cannot run must not condemn a working install. It
    fails CLOSED only when the program ran and said something that is not 4C,
    which is the case that matters — `/bin/true` runs, says nothing, and used to
    be reported as an available solver.

    Cached per path, because `check_availability` is called repeatedly by
    `discover` and every knowledge surface, and this spawns a process.
    """
    import subprocess

    key = str(binary)
    if key in _FOURC_IDENT_CACHE:
        return _FOURC_IDENT_CACHE[key]

    verdict: tuple[bool, str]
    try:
        # stdin MUST be closed. An audit pointed this at `/bin/cat`, which
        # consumed the PARENT's entire stdin and made the verdict a function of
        # that text; pointed at a real solver it hung the full timeout and then
        # failed open. Under an MCP stdio server the parent's stdin is the
        # JSON-RPC stream, so an identity probe could eat the protocol.
        r = subprocess.run([key], capture_output=True, timeout=20,
                           stdin=subprocess.DEVNULL)
        blob = (r.stdout + r.stderr).decode("utf-8", errors="replace").lower()
        # 4C's banner names itself in full. The first version matched "4c",
        # "dat file" and "input file", all of which are far too weak: "4c" is
        # two characters, so `/bin/pwd` and `/bin/ls` pass whenever the working
        # directory contains it, and `/usr/bin/env` passes whenever ANY
        # environment variable does — which is the normal state for a 4C user,
        # since LD_LIBRARY_PATH=/opt/4C-dependencies/lib contains it. And
        # "input file" passes `/usr/bin/gcc`, whose no-argument output is "no
        # input files". Four measured false positives from three sloppy tokens.
        # What 4C ACTUALLY emits with no arguments, measured rather than
        # assumed: it does not print a banner at all. It throws
        # `FourC::Core::Exception` with "Please provide both <input> and
        # <output> arguments." and aborts. An audit recommended matching the
        # project's full name, "Comprehensive Computational Community Code" —
        # that appears in the banner on a successful start, NOT on this path, so
        # matching it refused the real binary. Both the first token set and its
        # proposed replacement were wrong in opposite directions, which is why
        # this now uses strings taken from the observed output.
        #
        # `fourc::` is the C++ namespace and is specific enough on its own: no
        # ordinary executable emits it. The argument message is a second,
        # independent witness.
        markers = ("fourc::", "provide both <input> and <output>", "lib4c.so")
        if not blob.strip():
            verdict = (False, "produced no output at all when run with no "
                              "arguments; 4C aborts with a named exception")
        elif any(t in blob for t in markers):
            verdict = (True, "identified itself")
        else:
            verdict = (False, "its output carries none of 4C's own markers: "
                              + " ".join(blob.split())[:120])
    except (subprocess.TimeoutExpired, OSError) as exc:
        # Could not look — do not accuse a possibly-working install.
        # ValueError is deliberately NOT caught here: UnicodeDecodeError is a
        # ValueError, so catching it sent every non-UTF-8 binary down the
        # fail-open path and ACCEPTED it (`/usr/bin/gzip` passed). Decoding is
        # now explicit with errors="replace", so there is nothing left for a
        # ValueError to mean except a real bug, which should surface.
        verdict = (True, f"identity not checked ({type(exc).__name__})")

    _FOURC_IDENT_CACHE[key] = verdict
    return verdict


def _find_fourc_binary() -> Optional[Path]:
    """Locate the 4C binary."""
    # An explicit override that does not resolve must not fall through to the
    # search path — see the FEBio equivalent for what that costs. Kept as a
    # warning-and-None here rather than an exception, because this finder is
    # called from more places than FEBio's and a raise would change behaviour
    # in paths I have not tested.
    env_path = os.environ.get("FOURC_BINARY")
    if env_path and not Path(env_path).is_file():
        logger.warning(
            "FOURC_BINARY is set to %r, which is not a file; NOT falling back "
            "to the search path, because the binary openPASO tests must be the one "
            "you named", env_path)
        return None
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    if FOURC_ROOT:
        for d in ["build", "build/release", "build/debug"]:
            p = FOURC_ROOT / d / "4C"
            if p.is_file():
                return p
    # Fall back to the same search paths used by the
    # autodiscovery scanner (src/core/autodiscovery.py). Without
    # this, the discover MCP tool reports 4C as installed via the
    # scanner, but get_backend('fourc').check_availability() still
    # returns NOT_INSTALLED — the two surfaces drift out of sync.
    for cand in (
        "~/4C/build/4C",
        "~/4c/build/4C",
        "/opt/4c/build/4C",
        "/opt/4C/build/4C",
        "~/4C-src/4C/build/4C",
        *(str(d / sub) for d in _desktop_dirs()
          for sub in ("4C/build/4C", "4C-src/4C/build/4C")),
    ):
        p = Path(cand).expanduser()
        if p.is_file():
            return p
    p = shutil.which("4C")
    return Path(p) if p else None


def _get_generators():
    """Import the 4C generators — self-contained in openpaso."""
    # The generators package is at backends/fourc/generators/ (copied from 4c-ai-interface)
    from backends.fourc.generators import get_generator, list_generators
    return get_generator, list_generators


# A defined function is INERT until a condition references it, and VAL is a
# MULTIPLIER on it. Filed as a cross-cutting entry because it is true of every
# 4C physics, and because filing a universal fact under one physics row is a
# mistake this project has now made three times.
# Filed cross-cutting for the same reason as _FUNCT_WIRING: it is true of every
# 4C physics, and it answers the question that has now cost whole runs.
_GRAMMAR_DUMP = """\
NEVER GUESS A 4C SECTION NAME. THE BINARY WILL LIST THEM ALL, IN UNDER A
SECOND.

    4C -p > grammar.txt          # 68,931 lines here, 0.76 s
    grep -oE "DESIGN VOL [A-Z ]*CONDITIONS" grammar.txt | sort -u

That is the whole technique, and it answers section names, key names, defaults
and enum values for any problem type on THIS build -- which is the only build
whose answer matters. Documentation and forum posts describe other versions.

It is worth saying because the alternative is expensive. One run spent its
budget and gave up with: "The 4C binary requires a specific section name for
volume source conditions that I could not determine after extensive
debugging." The answer was two greps away:

    DESIGN VOL NEUMANN CONDITIONS          mechanical body force
    DESIGN VOL THERMO NEUMANN CONDITIONS   heat source
    DESIGN VOL TRANSPORT NEUMANN CONDITIONS
    DESIGN VOL PORO NEUMANN CONDITIONS

and the matching DIRICH / INITIAL FIELD / LOCSYS variants. Note the pattern:
the physics word sits between VOL and the condition type, so a THERMO problem
does not use the plain mechanical section, and a deck that names the wrong one
is rejected rather than silently ignored.

When a section is rejected, dump the grammar and grep for the words you expect
rather than trying spellings. Trying spellings is how an afternoon
disappears.
"""

_FUNCT_WIRING = """\
A `FUNCT` YOU DEFINE DOES NOTHING UNTIL A CONDITION POINTS AT IT, AND `VAL`
SCALES IT.

A common and expensive failure: the deck defines the
complete manufactured source as FUNCT1 —

    FUNCT1:
      - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "exp(t/2)*(t*x^3*y/2 + ...)"

— and then wired every condition with

    VAL: [0.0]
    FUNCT: [0]

The solver runs, converges, and writes a field of exactly 0.0 at every probe
point. Nothing errors, because nothing is wrong: the deck asked for zero
source and got it.

THREE switches, and ALL must be right — `ONOFF`, `VAL`, `FUNCT`. The scatra
body-force path evaluates `onoff * val * functfac`
(4C_scatra_ele_calc.cpp), so `ONOFF: [0]` zeroes the load just as surely as
the other two, and a deck that passes a grep for the other two can still
apply nothing:
  * `FUNCT: [n]` selects FUNCT<n>. `FUNCT: [0]` means NO FUNCTION — the
    literal zero index is 4C's "none", not "the first one".
  * `VAL: [v]` is the AMPLITUDE MULTIPLYING that function, not an alternative
    to it. The applied quantity is v * FUNCT<n>(x, t). So `VAL: [0.0]` with a
    perfectly correct FUNCT applies exactly nothing, and `VAL: [1.0]` with
    `FUNCT: [0]` applies a constant 1.0, not your function.

For a manufactured source the pair you almost always want is
`VAL: [1.0], FUNCT: [1]` — unit amplitude on the function that carries the
whole spatial and temporal shape.

CHECK IT BEFORE YOU BELIEVE A RESULT: a field that is identically zero (or
identically your Dirichlet value) on a driven problem is this bug until proven
otherwise. Grep your own deck for `ONOFF: [0]`, `VAL: [0.0]` and `FUNCT: [0]`
next to a function you spent effort deriving — all three, because checking two
of three is how this bug survives a review."""


class FourcBackend(SolverBackend):

    def name(self) -> str:
        return "fourc"

    def display_name(self) -> str:
        return "4C Multiphysics"

    def check_availability(self) -> tuple[BackendStatus, str]:
        binary = _find_fourc_binary()
        if not binary:
            return BackendStatus.NOT_INSTALLED, "4C binary not found (set FOURC_BINARY)"
        # Check that local generators are present (self-contained)
        local_gen = Path(__file__).parent / "generators" / "__init__.py"
        if not local_gen.exists():
            return BackendStatus.MISCONFIGURED, "4C generators not found in openpaso"

        # Confirm the binary IS 4C, not merely that a file exists and is
        # executable. `FOURC_BINARY=/bin/true` used to report
        # "available — 4C at /bin/true", so a stale path, a wrong build, or a
        # same-named program on PATH was indistinguishable from a working
        # install. An agent consults `discover`, believes it, and every run then
        # fails for a reason the availability report has already ruled out —
        # which is the worst place to be wrong.
        ident_ok, ident_why = _identifies_as_fourc(binary)
        if not ident_ok:
            return BackendStatus.MISCONFIGURED, (
                f"the binary at {binary} does not identify itself as 4C "
                f"({ident_why}). Point FOURC_BINARY at a real 4C build; a file "
                f"that merely exists and is executable is not a solver.")
        return BackendStatus.AVAILABLE, f"4C at {binary}"

    def input_format(self) -> InputFormat:
        return InputFormat.YAML

    def get_version(self) -> Optional[str]:
        binary = _find_fourc_binary()
        if not binary:
            return None
        import subprocess
        try:
            r = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL)
            for line in r.stdout.splitlines():
                if "version" in line.lower():
                    return line.strip()
        except Exception:
            pass
        return None

    def supported_physics(self) -> list[PhysicsCapability]:
        return [
            PhysicsCapability("poisson", "Poisson / scalar transport", [2, 3],
                              ["QUAD4", "HEX8", "TRI3", "TET4"],
                              ["poisson_2d", "heat_2d", "poisson_3d"]),
            PhysicsCapability("linear_elasticity", "Linear elasticity", [2, 3],
                              ["QUAD4", "HEX8"],
                              ["linear_2d", "nonlinear_3d"]),
            PhysicsCapability("plasticity", "Elasto-plasticity: J2/von Mises, Drucker-Prager, GTN damage, crystal plasticity", [2, 3],
                              ["QUAD4", "HEX8"],
                              ["linear_2d", "nonlinear_3d"]),
            # heat_transient_2d was reachable through generate_input()
            # and produced a running One-Step-Theta deck, but was absent
            # from template_variants, so no tool could select it and no
            # caller could discover it. Registered 2026-08-03 after
            # executing both variants on the installed 4C (both rc=0).
            PhysicsCapability("heat", "Heat conduction", [2, 3],
                              ["QUAD4", "HEX8"],
                              ["heat_2d", "heat_transient_2d"]),
            PhysicsCapability("fluid", "Incompressible Navier-Stokes", [2, 3],
                              ["QUAD4", "HEX8"],
                              ["channel_2d", "cavity_2d"]),
            PhysicsCapability("fsi", "Fluid-structure interaction", [2, 3],
                              ["QUAD4", "HEX8"],
                              ["fsi_2d"]),
            PhysicsCapability("structural_dynamics", "Structural dynamics", [2, 3],
                              ["QUAD4", "HEX8"],
                              ["genalpha_2d"]),
            PhysicsCapability("beams", "Beam elements", [2, 3],
                              ["BEAM3R", "BEAM3EB"],
                              ["cantilever_static", "cantilever_dynamic"]),
            # inline_penalty_3d FIRST: it is the only contact variant
            # that is self-contained (inline nodes + elements) and runs
            # without FOURC_ROOT or an external Exodus mesh. penalty_3d
            # is the tutorial/format-template route and needs both.
            PhysicsCapability("contact", "Contact mechanics", [3],
                              ["HEX8"],
                              ["inline_penalty_3d", "penalty_3d"]),
            PhysicsCapability("particle_pd", "Peridynamics (bond-based)", [2],
                              ["particle"],
                              ["plate_2d", "impact_2d"]),
            # Variants renamed 2026-08-07 to match the deck that now ships.
            # "poiseuille_2d" served a hydrostatic column and
            # "normal_impact_1d" a 3-D settling pack; a name that describes a
            # different experiment than the deck is the same class of defect
            # as a wrong key, and harder to notice because nothing errors.
            PhysicsCapability("particle_sph", "Smoothed particle hydrodynamics", [2],
                              ["particle"],
                              ["hydrostatic_2d", "dam_break_2d"]),
            PhysicsCapability("particle_dem",
                              "Discrete element method (granular contact, "
                              "friction, rolling, adhesion, walls)",
                              [1, 2, 3], ["particle"],
                              ["settling_3d"]),
            PhysicsCapability("tsi", "Thermo-structure interaction", [2, 3],
                              ["SOLIDSCATRA HEX8"],
                              ["monolithic_3d", "oneway_3d",
                               "plane_strain_2d"]),
            PhysicsCapability("ssi", "Structure-scalar interaction (battery/electrode)", [3],
                              ["SOLIDSCATRA HEX8"],
                              ["monolithic_elch_3d"]),
            PhysicsCapability("ale", "ALE mesh movement", [2, 3],
                              ["ALE2", "ALE3"],
                              ["ale_2d"]),
            PhysicsCapability("electrochemistry", "Electrochemistry (Nernst-Planck)", [2, 3],
                              ["TRANSP QUAD4", "TRANSP HEX8"],
                              ["nernst_planck_3d"]),
            PhysicsCapability("level_set", "Level-set interface tracking", [2, 3],
                              ["TRANSP QUAD4"],
                              ["advection_2d"]),
            PhysicsCapability("low_mach", "Low Mach number flow (buoyancy)", [2, 3],
                              ["FLUID QUAD4"],
                              ["heated_channel_2d"]),
            PhysicsCapability("ssti", "Structure-scalar-thermo interaction (3-field)", [3],
                              ["SOLIDSCATRA HEX8"],
                              ["monolithic_3d"]),
            PhysicsCapability("sti", "Scalar-thermo interaction", [3],
                              ["TRANSP HEX8"],
                              ["monolithic_3d"]),
            PhysicsCapability("fbi", "Fluid-beam interaction (immersed)", [3],
                              ["FLUID HEX8", "BEAM3R LINE2"],
                              ["penalty_3d"]),
            PhysicsCapability("fpsi", "Fluid-porous-structure interaction", [3],
                              ["FLUID HEX8", "SOLIDPORO HEX8"],
                              ["monolithic_3d"]),
            PhysicsCapability("pasi", "Particle-structure interaction", [3],
                              ["SOLID HEX8", "particle"],
                              ["dem_impact_3d"]),
            PhysicsCapability("lubrication", "Lubrication (Reynolds equation)", [2],
                              ["LUBRICATION QUAD4"],
                              ["slider_bearing_2d"]),
            PhysicsCapability("cardiac_monodomain", "Cardiac monodomain (electrophysiology)", [3],
                              ["TRANSP HEX8"],
                              ["monodomain_3d"]),
            PhysicsCapability("arterial_network", "Arterial network (1-D blood flow)", [1],
                              ["ARTERY LINE2"],
                              ["single_artery_1d"]),
            PhysicsCapability("xfem_fluid", "XFEM fluid (embedded interfaces)", [3],
                              ["FLUID HEX8"],
                              ["xfem_3d"]),
            PhysicsCapability("fsi_xfem", "FSI XFEM (fixed-grid fluid-structure)", [3],
                              ["FLUID HEX8", "SOLID HEX8"],
                              ["xfem_fsi_3d"]),
            PhysicsCapability("fs3i", "FS3I (fluid-structure-scalar-scalar, 5-field)", [3],
                              ["FLUID HEX8", "SOLID HEX8", "TRANSP HEX8"],
                              ["fs3i_3d"]),
            PhysicsCapability("ehl", "Elastohydrodynamic lubrication", [3],
                              ["LUBRICATION QUAD4", "SOLID HEX8"],
                              ["ehl_3d"]),
            PhysicsCapability("reduced_airways", "Reduced-dimensional airways (lung)", [1],
                              ["REDAIRWAY LINE2"],
                              ["airways_1d"]),
            PhysicsCapability("beam_interaction", "Beam interaction (contact/meshtying)", [3],
                              ["BEAM3R LINE2", "SOLID HEX8"],
                              ["beam_contact_3d", "beam_solid_meshtying_3d"]),
            PhysicsCapability("multiscale", "Multiscale FE-squared (computational homogenisation)", [3],
                              ["SOLID HEX8"],
                              ["fe2_3d"]),
            # single_phase_3d listed FIRST because it is the only one of
            # the three that generates a complete deck and runs (rc=0 on
            # the installed 4C, 2026-08-03); terzaghi_2d and
            # consolidation_3d fall through to the ~1 kB reference-stub
            # template and are documentation, not runnable input.
            # Added 2026-09-19. 4C declares porofluid_pressure_based_elasticity
            # as a problem type DISTINCT from Poroelasticity and from
            # porofluid_pressure_based, with its own section names. openPASO
            # reached the third but not this one, so a user asking for
            # monolithic pressure-based poro-elasticity was told 4C could not
            # do it. Deck executed on the installed binary (commit 89519cfe76):
            # rc=0, 26 VTU steps to the configured end time, porosity and solid
            # displacement both varying.
            # Added 2026-09-19, both verified by running the exact text
            # openPASO serves on the installed binary.
            PhysicsCapability("poroelast_scatra",
                              "Poroelasticity + reacting scalar transport "
                              "through the pore fluid", [3],
                              ["HEX8"],
                              ["homogeneous_3d"]),
            PhysicsCapability("fluid_ale",
                              "Fluid on a deforming (ALE) mesh -- the moving-"
                              "domain setting every FSI and free-surface "
                              "problem needs", [2],
                              ["QUAD4", "TRI3"],
                              ["hdg_2d"]),
            PhysicsCapability("porofluid_elasticity_scatra",
                              "Multiphase porous flow + deformable skeleton + "
                              "scalar transport, monolithically coupled", [3],
                              ["HEX8"],
                              ["monolithic_3d"]),
            PhysicsCapability("porofluid_elasticity",
                              "Pressure-based porous-media flow monolithically "
                              "coupled to an elastic skeleton", [3],
                              ["HEX27", "HEX8"],
                              ["monolithic_3d"]),
            PhysicsCapability("porous_media", "Poroelasticity (Biot/mixture theory, consolidation)", [2, 3],
                              ["WALLQ4PORO", "WALLQ9PORO", "SOLIDH8PORO", "SOLIDT4PORO"],
                              ["single_phase_3d", "terzaghi_2d", "consolidation_3d"]),
            # New physics
            PhysicsCapability("membrane", "Membrane elements (inflatable, fabric, tissue)", [2, 3],
                              ["MEMBRANE TRI3", "MEMBRANE QUAD4"], ["membrane_2d"]),
            PhysicsCapability("shell", "Shell elements (Kirchhoff-Love, Reissner-Mindlin)", [3],
                              ["SHELL REISSNER QUAD4", "SHELL KIRCHHOFF TRI3", "SOLIDSHELL HEX8"], ["shell_3d"]),
            PhysicsCapability("thermo", "Pure thermal analysis (standalone heat conduction)", [2, 3],
                              ["THERMO QUAD4", "THERMO HEX8"], ["thermo_2d", "thermo_3d"]),
            PhysicsCapability("thermo_transient_mms",
                              "Transient thermal MMS on a fixed mesh for "
                              "TEMPORAL-order dt-halving studies "
                              "(One-Step-Theta: order 2 at theta=0.5, "
                              "order 1 at theta=1)", [2],
                              ["THERMO QUAD4"], ["temporal_mms_2d"]),
            PhysicsCapability("mixture", "Mixture/composite materials (fiber-reinforced, biological)", [3],
                              ["SOLID HEX8 with MAT_Mixture"], ["mixture_3d"]),
            PhysicsCapability("constraint", "Constraints: MPC, rigid body, periodic BCs, mortar coupling", [2, 3],
                              ["Generic"], ["constraint_3d"]),
            PhysicsCapability("brownian_dynamics", "Brownian dynamics of fiber/biopolymer networks", [3],
                              ["BEAM3R LINE2"], ["brownian_3d"]),
            PhysicsCapability("cardiovascular0d", "0-D cardiovascular: windkessel, closed-loop circulation, heart models", [3],
                              ["coupled to 3D fluid/structure"], ["windkessel_3d"]),
            PhysicsCapability("reduced_lung", "Reduced lung model: 1D airways + 0D alveoli + optional 3D parenchyma", [1, 3],
                              ["REDAIRWAY LINE2 + 0D acini"], ["lung_1d"]),
            PhysicsCapability("fluid_turbulence", "Fluid turbulence: LES (Smagorinsky, dynamic, WALE) and DNS", [2, 3],
                              ["FLUID QUAD4", "FLUID HEX8"], ["les_channel_3d"]),
            # ── 2026-06-01: umbrella catalogs from data/fourc_knowledge.py
            #    that aggregate pitfalls across families of specific
            #    physics. Previously orphaned (catalog reachable via
            #    knowledge(physics=...) but not listed in
            #    discover(physics, fourc)). Exposed here so users see
            #    the umbrella name alongside the specific ones.
            PhysicsCapability(
                "scalar_transport",
                "[Umbrella] Scalar-transport family pitfalls "
                "(applies to poisson, heat, electrochemistry, "
                "level-set, low-mach scalars). For specific "
                "physics use poisson/heat/electrochemistry "
                "directly.",
                [2, 3], ["TRANSP QUAD4", "TRANSP HEX8"],
                ["umbrella"]),
            PhysicsCapability(
                "structural_mechanics",
                "[Umbrella] Structural-mechanics family pitfalls "
                "(applies to linear_elasticity, plasticity, "
                "structural_dynamics, beams, contact). For "
                "specific physics use linear_elasticity / "
                "plasticity / structural_dynamics directly.",
                [2, 3], ["SOLID HEX8", "SOLID QUAD4"],
                ["umbrella"]),
            PhysicsCapability(
                "thermal",
                "[Umbrella] Thermal-analysis family pitfalls "
                "(applies to heat, thermo, tsi). For specific "
                "physics use heat / thermo / tsi directly.",
                [2, 3], ["THERMO QUAD4", "THERMO HEX8"],
                ["umbrella"]),
            PhysicsCapability(
                "input_format",
                "[Reference] Cross-physics general 4C input "
                "pitfalls (ExodusII 1-indexed block IDs, "
                "SYMBOLIC_FUNCTION_OF_SPACE_TIME COMPONENT "
                "requirement, NUMDOF conflicts on shared "
                "FSI/TSI nodes, .yaml-only extension, "
                "post_vtu vs IO/RUNTIME VTK OUTPUT, WALL→SOLID "
                "rename, etc.). Not a PDE physics — meta-"
                "reference entry. Underlying KNOWLEDGE key in "
                "data/fourc_knowledge.py is 'input_format'.",
                [2, 3], ["N/A — meta-reference"], ["N/A"]),
            PhysicsCapability(
                "particles",
                "[Umbrella] Particle-methods family pitfalls "
                "(applies to particle_pd, particle_sph, "
                "pasi, dem). For specific physics use "
                "particle_pd / particle_sph / pasi directly.",
                [2, 3], ["particle"],
                ["umbrella"]),
        ]

    def get_knowledge(self, physics: str) -> dict:
        # THE DECK GRAMMAR REACHES THE SINGLE-CODE PROBLEMS TOO.
        #
        # It was served only from the coupling payload, and a single-code 4C
        # problem never calls knowledge(topic='coupling') -- it has no
        # coupling. So agents on single-code tasks received nothing about how
        # to write a runnable deck, which is the same defect that killed three
        # recorded runs of one coupled problem.
        #
        # One copy, in backends/fourc/deck_grammar.py, served from both paths.
        # The interface-probe text taught why that matters: it existed in four
        # places, two of them dead, and a fix to the wrong one looked correct.
        # Try deep knowledge from data file first
        # Resolution: merge data/fourc_knowledge.py (rich
        # course-level dict — description / methods / variants /
        # constitutive_laws / etc.) with the generator's
        # per-physics pitfalls list. Previously the data file
        # SHADOWED the generator: if FOURC_KNOWLEDGE had an
        # entry without a 'pitfalls' field, get_knowledge
        # returned that entry and the generator's pitfalls
        # were unreachable. Critic-audit 2026-06-01 finding #14
        # (fourc::contact had 0 pitfalls reachable; the actual
        # 8 contact.py pitfalls were silently shadowed).
        data_entry: dict = {}
        try:
            import sys
            from core.paths import data_dir as _data_dir
            data_dir = str(_data_dir())
            if data_dir not in sys.path:
                sys.path.insert(0, data_dir)
            from fourc_knowledge import FOURC_KNOWLEDGE
            data_entry = FOURC_KNOWLEDGE.get(physics, {})
        except ImportError:
            pass

        gen_entry: dict = {}
        try:
            get_gen, _ = _get_generators()
            gen = get_gen(physics)
            gen_entry = gen.get_knowledge()
        except Exception:  # noqa: BLE001
            pass

        try:
            from backends.fourc.deck_grammar import FOURC_DECK_GRAMMAR
        except Exception:                                # pragma: no cover
            FOURC_DECK_GRAMMAR = ""

        def _with_grammar(d: dict) -> dict:
            if FOURC_DECK_GRAMMAR and isinstance(d, dict):
                d = dict(d)
                d["deck_grammar"] = FOURC_DECK_GRAMMAR
            return d

        if data_entry and gen_entry:
            # Merge: data_entry wins for shared keys (its
            # description / methods / variants are richer).
            merged = dict(data_entry)
            # PITFALLS ARE UNIONED, NOT CHOSEN BETWEEN. The 2026-06-01
            # fix only fell back to the generator's list when the data
            # file had none, so wherever BOTH carried pitfalls the
            # generator's were still discarded — 49 verified entries
            # across contact (8), fsi (13), tsi (9), scalar_transport
            # (7), fluid (6) and thermal (6). The two lists are
            # complementary, not competing: the data file's contact
            # entries are section-and-key facts read off `4C -p`, the
            # generator's are the interface/penalty/KINEM behaviours
            # met while building a deck. Neither is a superset.
            #
            # It stayed invisible because the shadowing key and the
            # visible key are the same one. It surfaced only through the
            # ALIASES, which have no data-file entry of their own:
            # `get_knowledge('mortar')` and `get_knowledge('penalty_contact')`
            # returned the generator's 8 contact pitfalls, and those 8
            # texts appeared under NO enumerated area — an agent could
            # reach them only by guessing the alias.
            merged_pitfalls = list(data_entry.get("pitfalls") or [])
            seen = {p for p in merged_pitfalls if isinstance(p, str)}
            for p in (gen_entry.get("pitfalls") or []):
                if isinstance(p, str) and p in seen:
                    continue      # identical text, not a second entry
                merged_pitfalls.append(p)
                if isinstance(p, str):
                    seen.add(p)
            if merged_pitfalls:
                merged["pitfalls"] = merged_pitfalls
            # Carry over any other gen-only keys.
            for k, v in gen_entry.items():
                if k not in merged:
                    merged[k] = v
            merged = dict(merged)
            merged["funct_wiring"] = _FUNCT_WIRING
            merged["grammar_dump"] = _GRAMMAR_DUMP
            return _with_grammar(merged)
        if data_entry:
            data_entry = dict(data_entry)
            data_entry["funct_wiring"] = _FUNCT_WIRING
            data_entry["grammar_dump"] = _GRAMMAR_DUMP
            return _with_grammar(data_entry)
        if gen_entry:
            gen_entry = dict(gen_entry)
            gen_entry["funct_wiring"] = _FUNCT_WIRING
            gen_entry["grammar_dump"] = _GRAMMAR_DUMP
            return _with_grammar(gen_entry)
        # Last resort: the deck catalog. Four physics rows are served only
        # from there and had no knowledge at all -- the templates ran and
        # get_knowledge() returned an error for the same name.
        try:
            from backends.fourc import decks as _decks
            deck_entry = _decks.knowledge_for(physics)
        except Exception:                                # pragma: no cover
            deck_entry = {}
        if deck_entry:
            deck_entry = dict(deck_entry)
            deck_entry["funct_wiring"] = _FUNCT_WIRING
            deck_entry["grammar_dump"] = _GRAMMAR_DUMP
            return _with_grammar(deck_entry)
        return {"error": f"no knowledge for {physics!r} in fourc"}

    def generate_input(self, physics: str, variant: str, params: dict) -> str:
        # Umbrella / meta-reference physics: catalog declares
        # these so they appear in discover() and knowledge()
        # surfaces (e.g. scalar_transport groups poisson + heat +
        # electrochemistry + level-set + low-mach scalars). They
        # are documentation-only — generate_input returns a YAML
        # commentary block pointing to the concrete physics names
        # in the same family. Without this early-return, calling
        # generate_input('scalar_transport', 'umbrella', {}) would
        # cascade through the inline / tutorial / generator chain
        # and raise ValueError.
        if variant in ("umbrella", "N/A"):
            return self._umbrella_template(physics, variant)

        # Executed deck templates come first. Every entry in
        # backends.fourc.decks was run on the installed binary, exited 0 AND
        # wrote a finite field (checked by scripts/coverage_harness; exit 0
        # alone let fsi_2d serve a NaN pressure at every node of every step
        # until 2026-09-23), which is a stronger guarantee than any other
        # branch below offers,
        # and the pairs it covers are exactly the ones that used to fall
        # through to the "Not a runnable input" stub.
        from backends.fourc import decks as _decks
        _deck = _decks.render(physics, variant)
        if _deck is not None:
            return _deck

        # First try inline mesh generators (self-contained, no external files)
        try:
            return self._generate_inline(physics, variant, params)
        except ValueError:
            pass

        # Then try tutorial-based templates (these include mesh files)
        try:
            return self._generate_from_tutorial(physics, variant, params)
        except ValueError:
            pass

        # Honest reference stub BEFORE the generator fallback. Deep
        # multiphysics rows (xfem, fs3i, fpsi, ehl, fbi, pasi, ssi/
        # ssti/sti, cardiac_monodomain, arterial/airway/lung 1-D,
        # multiscale fe2, beam_interaction, particle_pd/sph, LES,
        # brownian) DO have a generator template, but it is a
        # placeholder full of literal <...> scalars + external mesh
        # references that aborts 4C in MatchTree (probe 2026-06-12).
        # A documented stub the user can read beats a guaranteed
        # MPI_Abort, so the stub catalog takes precedence over the
        # broken placeholder. The stub omits MATERIALS on purpose:
        # validate_input() flags it as non-runnable, so the probe
        # never reports it as a completed run — it is honestly
        # "documented, not runnable".
        stub = self._reference_stub_template(physics, variant)
        if stub is not None:
            return stub

        # Fallback: try generator-based templates
        try:
            get_gen, _ = _get_generators()
            gen = get_gen(physics)
            content = gen.get_template(variant)
            content = self._resolve_mesh_references(content)
            return content
        except Exception as e:
            # Last-resort reference stub. The catalog advertises
            # (physics, variant) in supported_physics() so it
            # appears in discover() — without something runnable
            # here, calling generate_input on that pair raised
            # ValueError unconditionally. Many 4C problems
            # (plasticity, particle_pd impact, particle_sph
            # dam_break, porous_media terzaghi/consolidation) need
            # case-specific mesh + parameters that cannot be
            # baked into a generic template. The stub is a
            # valid YAML reference that documents what's
            # required so an LLM agent or human user knows
            # what to fill in. See _reference_stub_template for
            # the list of stub-eligible (physics, variant) pairs.
            stub = self._reference_stub_template(
                physics, variant)
            if stub is not None:
                return stub
            raise ValueError(f"No 4C template for {physics}/{variant}: {e}")

    def _reference_stub_template(self, physics: str,
                                  variant: str) -> str | None:
        """Reference-stub fallback for catalog-advertised
        (physics, variant) pairs that need case-specific
        mesh + parameters (and thus can't be baked into a
        generic generator). Returns a YAML commentary
        block that documents what the user must fill in.

        Returns None for pairs not in the stub catalog —
        the caller falls through to its original
        ValueError.
        """
        # Map (physics, variant) → (problemtype, description,
        # required sections, pitfalls).
        stubs: dict[tuple[str, str], dict] = {
            # ── Deep multiphysics rows that genuinely need a
            #    case-specific mesh (often TWO meshes), a second
            #    input file, patient-derived topology, an explicit
            #    particle cloud, or a build feature this 4C lacks.
            #    A generic inline QUAD4/HEX8 mesh cannot carry them,
            #    so they are honest reference stubs instead of a
            #    guaranteed MPI_Abort from the placeholder template
            #    (probe 2026-06-12).
        }
        # A stub must never shadow an executed deck. generate_input already
        # consults backends.fourc.decks first, so a pair present in both would
        # be dead code that silently rots out of date — and the whole point of
        # the deck catalog is that its content is the content that ran.
        from backends.fourc import decks as _decks
        if _decks.get(physics, variant) is not None:
            raise AssertionError(
                f"stub catalog still carries {physics}/{variant}, which now "
                f"has an executed deck in backends.fourc.decks")
        spec = stubs.get((physics, variant))
        if spec is None:
            return None
        problemtype = spec["problemtype"]
        summary = spec["summary"]
        needs = "\n".join(
            f"#   {i+1}. {n}" for i, n in enumerate(
                spec["needs"]))
        pitfalls = "\n".join(
            f"#   * {p}" for p in spec["pitfalls"])
        return (
            f"# ============================================\n"
            f"# 4C reference stub: {physics} / {variant}\n"
            f"# ============================================\n"
            f"# {summary}\n"
            f"#\n"
            f"# Not a runnable input — the user must supply\n"
            f"# the case-specific mesh + material parameters.\n"
            f"# This stub lists what's required:\n"
            f"#\n"
            f"{needs}\n"
            f"#\n"
            f"# Pitfalls (see knowledge() for the full set):\n"
            f"{pitfalls}\n"
            f"# ============================================\n"
            f"TITLE:\n"
            f'  - "4C {physics}/{variant} reference stub"\n'
            f"PROBLEM TYPE:\n"
            f'  PROBLEMTYPE: "{problemtype}"\n'
        )

    def _umbrella_template(self, physics: str, variant: str) -> str:
        """Return a YAML-commentary template for umbrella /
        meta-reference physics (scalar_transport,
        structural_mechanics, thermal, particles, input_format).
        These aren't runnable physics inputs — they're a
        catalog cross-reference. The returned YAML is parseable
        and validates against 4C 2026.3 (no PROBLEM TYPE means
        4C reports a 'PROBLEMTYPE missing' diagnostic, but the
        file itself is valid YAML)."""
        family_redirects = {
            "scalar_transport": ("poisson, heat, "
                                 "electrochemistry, level_set, "
                                 "low_mach"),
            "structural_mechanics": ("linear_elasticity, "
                                     "plasticity, "
                                     "structural_dynamics, "
                                     "beams, contact"),
            "thermal": "heat, thermo, tsi",
            "particles": ("particle_pd, particle_sph, pasi, "
                          "dem (use kratos for dem instead)"),
            "input_format": ("meta-reference only — see "
                             "data/fourc_knowledge.py['input_format']"),
        }
        family = family_redirects.get(physics,
                                       "<unknown umbrella>")
        return (
            f"# =====================================================\n"
            f"# 4C umbrella / meta-reference physics: '{physics}'\n"
            f"# variant: '{variant}'\n"
            f"# =====================================================\n"
            f"# This is NOT a runnable 4C input. The catalog\n"
            f"# advertises '{physics}' so it appears in discover()\n"
            f"# and knowledge() results, where it groups related\n"
            f"# physics under a shared documentation umbrella.\n"
            f"#\n"
            f"# For a RUNNABLE input pick one of the concrete\n"
            f"# physics names in the same family:\n"
            f"#\n"
            f"#   {family}\n"
            f"#\n"
            f"# Below this header is the RUNNABLE template of the\n"
            f"# first concrete child, verbatim, so a reader who\n"
            f"# asked the umbrella name still gets something they\n"
            f"# can execute instead of a dead end.\n"
            f"# For the other children call prepare_simulation with\n"
            f"# their name.\n"
            f"# =====================================================\n"
        ) + self._umbrella_child_template(physics, family)

    def _umbrella_child_template(self, physics: str, family: str) -> str:
        """The smallest concrete child's template that renders whole.

        An umbrella row used to return nothing but a redirect. That is a dead
        end for exactly the reader it is meant to help: a small model that
        asked for `thermal` gets a list of names and no deck, and has to guess
        which name to ask for next. Serving a child's executed template under a
        header that says plainly which physics it is costs nothing and removes
        a round trip.

        The size rule is not cosmetic. prepare_simulation truncates a template
        at 12000 characters, and several children are far above that (fourc's
        poisson ships a 32x32 inline mesh, ~135 KB). A deck cut off mid-mesh is
        worse than the redirect it replaced, because it looks complete. So the
        smallest child that fits with headroom wins, and if none fits the
        redirect stands.
        """
        fallback = (f"TITLE:\n"
                    f"  - \"4C umbrella reference for {physics}\"\n")
        # prepare_simulation truncates a template at 12000 characters; leave
        # headroom for the header this method's caller prepends.
        BUDGET = 10000
        # Several inline generators default to a display-hostile mesh (fourc's
        # poisson_2d is 32x32, ~134 KB). They accept a coarser one, and a
        # coarse deck that renders whole teaches more than a fine one cut off
        # mid-mesh, so try the coarse form before giving up on a child.
        PARAM_TRIES = ({}, {"nx": 4, "ny": 4})
        best: tuple[int, str, str, str] | None = None
        for name in (n.strip() for n in family.split(",")):
            if not name or name.startswith("meta-reference"):
                continue
            row = next((p for p in self.supported_physics()
                        if p.name == name and p.template_variants), None)
            if row is None:
                continue
            for variant in row.template_variants:
                for params in PARAM_TRIES:
                    try:
                        child = self.generate_input(name, variant, dict(params))
                    except Exception:
                        continue
                    if not child or child.lstrip().startswith("# ====="):
                        continue
                    if "Not a runnable" in child:
                        continue
                    # An external mesh reference makes the template unrunnable
                    # for anyone who does not have that file. `\bFILE:` matches
                    # the mesh-file key and deliberately NOT TEKO_XML_FILE /
                    # MICROFILE, which resolve through FOURC_ROOT and are
                    # documented on the two decks that need them.
                    if re.search(r"\bFILE:", child):
                        continue
                    if len(child) > BUDGET:
                        continue
                    if best is None or len(child) < best[0]:
                        best = (len(child), name, variant, child)
                    break
        if best is None:
            return fallback
        _, name, variant, child = best
        return (f"# ---- concrete child, shown in full: "
                f"{name} / {variant} ----\n" + child)

    def _generate_inline(self, physics: str, variant: str, params: dict) -> str:
        """Generate self-contained input with inline mesh (no external files)."""
        from backends.fourc.inline_mesh import (
            matched_poisson_input, matched_heat_input,
            matched_elasticity_input, matched_poisson_3d_input,
            matched_l_domain_poisson_input,
            matched_heat_transient_input,
            matched_elasticity_genalpha_input,
            matched_elasticity_3d_nonlinear_input,
            matched_level_set_advection_input,
            matched_ale_2d_input,
            matched_nernst_planck_3d_input,
            matched_low_mach_heated_channel_input,
            matched_porofluid_single_phase_3d_input,
            matched_tsi_monolithic_3d_input,
            matched_tsi_oneway_input,
            matched_tsi_plane_strain_input,
            matched_beam_cantilever_static_input,
            matched_beam_cantilever_dynamic_input,
            matched_thermo_2d_input,
            matched_thermo_3d_input,
            matched_thermo_transient_mms_input,
            matched_lubrication_slider_bearing_input,
            matched_mixture_3d_input,
            matched_constraint_3d_input,
            matched_membrane_2d_input,
            matched_shell_3d_input,
            matched_cardiovascular0d_windkessel_input,
            matched_fluid_cavity_input,
            matched_fluid_channel_input,
            matched_reduced_airways_input,
            matched_contact_3d_input,
        )
        key = f"{physics}_{variant}"

        def _elasticity(p):
            return matched_elasticity_input(
                nx=p.get("nx", 40), ny=p.get("ny", 4),
                E=p.get("E", 1000.0), nu=p.get("nu", 0.3),
                lx=p.get("lx", 10.0), ly=p.get("ly", 1.0))

        inline_generators = {
            "poisson_2d": lambda p: matched_poisson_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32)),
            "poisson_poisson_2d": lambda p: matched_poisson_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32)),
            # scalar_transport is the catalog umbrella for the same
            # physics — route its concrete variants to the proven
            # matched inputs instead of the placeholder generator
            # templates (probe 2026-06-12: those abort in 4C's
            # MatchTree with un-substituted <...> placeholders).
            "scalar_transport_poisson_2d": lambda p: matched_poisson_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32)),
            "heat_2d": lambda p: matched_heat_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32),
                T_left=p.get("T_left", 100.0), T_right=p.get("T_right", 0.0)),
            "heat_heat_2d": lambda p: matched_heat_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32),
                T_left=p.get("T_left", 100.0), T_right=p.get("T_right", 0.0)),
            "poisson_heat_2d": lambda p: matched_heat_input(
                nx=p.get("nx", 32), ny=p.get("ny", 32),
                T_left=p.get("T_left", 100.0),
                T_right=p.get("T_right", 0.0)),
            "scalar_transport_heat_transient_2d":
                lambda p: matched_heat_transient_input(
                    nx=p.get("nx", 16), ny=p.get("ny", 16),
                    T_left=p.get("T_left", 100.0),
                    T_right=p.get("T_right", 0.0),
                    numstep=p.get("numstep", 10),
                    timestep=p.get("timestep", 0.01)),
            "heat_heat_transient_2d":
                lambda p: matched_heat_transient_input(
                    nx=p.get("nx", 16), ny=p.get("ny", 16),
                    T_left=p.get("T_left", 100.0),
                    T_right=p.get("T_right", 0.0),
                    numstep=p.get("numstep", 10),
                    timestep=p.get("timestep", 0.01)),
            "linear_elasticity_linear_2d": _elasticity,
            "linear_elasticity_2d": _elasticity,
            # solid_mechanics is the structural umbrella physics;
            # its linear_2d variant is the same cantilever the
            # linear_elasticity row uses (probe 2026-06-12).
            "solid_mechanics_linear_2d": _elasticity,
            "solid_mechanics_nonlinear_3d":
                lambda p: matched_elasticity_3d_nonlinear_input(
                    n=p.get("n", 4),
                    E=p.get("E", 1000.0), nu=p.get("nu", 0.3)),
            # linear_elasticity/nonlinear_3d previously fell through to
            # the generator template with <placeholder> scalars + an
            # external Exodus mesh (probe 2026-06-26: MatchTree abort).
            # It is the same finite-strain HEX8 cube the solid_mechanics
            # umbrella uses; route it to the proven inline input.
            "linear_elasticity_nonlinear_3d":
                lambda p: matched_elasticity_3d_nonlinear_input(
                    n=p.get("n", 4),
                    E=p.get("E", 1000.0), nu=p.get("nu", 0.3)),
            "structural_dynamics_genalpha_2d":
                lambda p: matched_elasticity_genalpha_input(
                    nx=p.get("nx", 20), ny=p.get("ny", 4),
                    E=p.get("E", 1000.0), nu=p.get("nu", 0.3),
                    dens=p.get("dens", 1.0),
                    numstep=p.get("numstep", 10),
                    timestep=p.get("timestep", 0.05)),
            # low_mach/heated_channel_2d fell through to the generator
            # template with <placeholder> scalars + an external Exodus
            # mesh (probe 2026-06-12: MatchTree abort). Route to the
            # self-contained inline heated-channel Loma input.
            "low_mach_heated_channel_2d":
                lambda p: matched_low_mach_heated_channel_input(
                    nx=min(int(p.get("nx", 32)), 64),
                    ny=min(int(p.get("ny", 8)), 32),
                    u_max=p.get("u_max", 0.3),
                    T_in=p.get("T_in", 293.0),
                    T_wall=p.get("T_wall", 350.0),
                    numstep=p.get("numstep", 5),
                    timestep=p.get("timestep", 0.1)),
            "poisson_3d": lambda p: matched_poisson_3d_input(n=p.get("n", 8)),
            "poisson_poisson_3d": lambda p: matched_poisson_3d_input(n=p.get("n", 8)),
            "poisson_l_domain": lambda p: matched_l_domain_poisson_input(
                n=p.get("n", 16)),
            # electrochemistry/nernst_planck_3d previously fell through
            # to the generator template with <placeholder> scalars + an
            # external Exodus mesh reference (probe 2026-06-12:
            # MatchTree abort). Route to the self-contained inline-mesh
            # Nernst-Planck input. Resolution uses "n" (not nx/ny/nz)
            # so the probe's nz=16 cannot inflate the 3-species
            # nonlinear 3D solve.
            "electrochemistry_nernst_planck_3d":
                lambda p: matched_nernst_planck_3d_input(
                    n=p.get("n", 4),
                    c_left=p.get("c_left", 2.0),
                    c_right=p.get("c_right", 1.0),
                    d_cation=p.get("d_cation", 2.0),
                    d_anion=p.get("d_anion", 1.0),
                    numstep=p.get("numstep", 10),
                    timestep=p.get("dt", 0.001)),
            # ale/ale_2d previously fell through to the generator
            # template with <placeholder> scalars + external Exodus
            # mesh (probe 2026-06-12: MatchTree abort). Inline 2D
            # mesh-motion problem instead.
            "ale_ale_2d": lambda p: matched_ale_2d_input(
                nx=min(int(p.get("nx", 16)), 32),
                ny=min(int(p.get("ny", 16)), 32),
                E=p.get("E", 1.0), nu=p.get("nu", 0.3),
                dens=p.get("rho", 1.0),
                numstep=max(1, round(p.get("T_end", 0.01)
                                     / p.get("dt", 0.001))),
                timestep=p.get("dt", 0.001)),
            # level_set/advection_2d previously fell through to the
            # placeholder generator template (literal <...> scalars +
            # external Exodus mesh → 4C MatchTree abort, probe
            # 2026-06-12). Route to the self-contained inline input.
            "level_set_advection_2d":
                lambda p: matched_level_set_advection_input(
                    nx=p.get("nx", 16), ny=p.get("ny", 16),
                    numstep=p.get("numstep", 10),
                    timestep=p.get("timestep", 0.01),
                    radius=p.get("radius", 0.25)),
            # porous_media/single_phase_3d previously used the generator
            # template with "TRANSPORT ELEMENTS" + a "TYPE
            # PoroFluidMultiPhase" element suffix that 4C's input
            # matcher rejects (probe 2026-06-12: MPI_Abort). Route to
            # the corpus-matched inline input (FLUID ELEMENTS /
            # POROFLUIDMULTIPHASE HEX8 ... MAT 1). Resolution uses "n"
            # (not nx/ny/nz) so the probe's nz=16 cannot inflate the
            # 3D mesh.
            "porous_media_single_phase_3d":
                lambda p: matched_porofluid_single_phase_3d_input(
                    n=p.get("n", 4),
                    permeability=p.get("kappa", 1.0),
                    viscosity=p.get("mu", 0.01),
                    density=p.get("rho", 1.0),
                    numstep=p.get("numstep", 10),
                    timestep=p.get("timestep", 0.01)),
            # tsi/monolithic_3d previously fell through to the generator
            # template with <placeholder> scalars + external Exodus mesh
            # (probe 2026-06-12: MatchTree abort). Route to the inline
            # SOLIDSCATRA cube with genuinely MONOLITHIC two-way coupling
            # (COUPALGO tsi_monolithic, merged TSI block matrix +
            # UMFPACK). Mesh capped at 8^3: the probe passes nx=ny=nz=16
            # and a 16^3 monolithic SOLIDSCATRA solve is too big.
            # tsi/plane_strain_2d: pseudo-2D thin-slab route for 2D
            # plane-strain thermo-mechanics. 4C has NO 2D TSI elements
            # (module solid_scatra_3D_ele; every TSI corpus test is 3D)
            # and the 2D structural eletypes both dead-end with the
            # thermo material on current builds (WALL QUAD4 -> "Invalid
            # type of material law for wall element"; SOLID QUAD4 ->
            # "Element 'SOLID' does not seem to know cell type
            # 'quad4'"). One SOLIDSCATRA HEX8
            # layer with u_z fixed everywhere is exact plane strain;
            # temp_expr imposes a (partner-computed) temperature field.
            "tsi_plane_strain_2d":
                lambda p: matched_tsi_plane_strain_input(
                    nx=min(int(p.get("nx", 16)), 64),
                    ny=min(int(p.get("ny", 4)), 32),
                    lx=p.get("lx", 2.0), ly=p.get("ly", 0.25),
                    thickness=p.get("thickness"),
                    E=p.get("E", 200e9), nu=p.get("nu", 0.3),
                    alpha=p.get("alpha", 12e-6),
                    T_ref=p.get("T_ref", 293.0),
                    T_left=p.get("T_left", 293.0),
                    T_right=p.get("T_right", 450.0),
                    temp_expr=p.get("temp_expr"),
                    density=p.get("rho", 7850.0),
                    conductivity=p.get("kappa", 1.0)),
            # tsi/oneway_3d: the corrected one-way (thermo->structure)
            # heated-beam input. Existed in inline_mesh since the
            # coupled_solve era but was never exposed as a variant —
            # and carried the silent COUPVARIABLE default bug (ran
            # rc=0, zero displacement) until that was fixed.
            "tsi_oneway_3d":
                lambda p: matched_tsi_oneway_input(
                    nx=min(int(p.get("nx", 4)), 8),
                    ny=min(int(p.get("ny", 4)), 8),
                    nz=min(int(p.get("nz", 4)), 8),
                    E=p.get("E", 200e3), nu=p.get("nu", 0.3),
                    alpha=p.get("alpha", 12e-6),
                    T_left=p.get("T_left", 100.0),
                    T_right=p.get("T_right", 0.0),
                    T_ref=p.get("T_ref", 0.0),
                    density=p.get("rho", 1.0),
                    conductivity=p.get("kappa", 1.0)),
            "tsi_monolithic_3d":
                lambda p: matched_tsi_monolithic_3d_input(
                    nx=min(int(p.get("nx", 4)), 8),
                    ny=min(int(p.get("ny", 4)), 8),
                    nz=min(int(p.get("nz", 4)), 8),
                    E=p.get("E", 200e3), nu=p.get("nu", 0.3),
                    density=p.get("rho", 1.0),
                    conductivity=p.get("kappa", 1.0),
                    numstep=max(1, round(p.get("T_end", 0.01)
                                         / p.get("dt", 0.001))),
                    timestep=p.get("dt", 0.001)),
            # beams/cantilever_* previously fell through to the
            # generator templates with <placeholder> scalars (probe
            # 2026-06-12: MatchTree abort). Route to corpus-matched
            # inline BEAM3R cantilevers; the tip load scales with E,
            # so the probe's E=1000 override converges like the
            # default E=1e7.
            # contact/inline_penalty_3d: the ONLY self-contained contact
            # variant. contact/penalty_3d needs FOURC_ROOT and an external
            # Exodus mesh; without them it falls through to the
            # <placeholder> format template, which cannot run (4C's
            # MatchTree rejects "TIMESTEP: <load_step_size>"). This one
            # carries its own nodes and elements and completes every load
            # step on the deployed 4C (verified by execution 2026-08-03).
            "contact_inline_penalty_3d":
                lambda p: matched_contact_3d_input(
                    nx=int(p.get("nx", 2)), ny=int(p.get("ny", 2)),
                    nz=int(p.get("nz", 2)),
                    gap=p.get("gap", 0.1),
                    indent=p.get("indent", 0.3),
                    penalty=p.get("penalty", 1.0e4),
                    E=p.get("E", 1000.0), nu=p.get("nu", 0.3),
                    n_steps=int(p.get("n_steps", 10))),
            "beams_cantilever_static":
                lambda p: matched_beam_cantilever_static_input(
                    n_elem=p.get("n_elem", 10),
                    length=p.get("length", 10.0),
                    radius=p.get("radius", 0.1),
                    E=p.get("E", 1.0e7), nu=p.get("nu", 0.3),
                    load_factor=p.get("load_factor", 1.0),
                    numstep=p.get("numstep", 5)),
            "beams_cantilever_dynamic":
                lambda p: matched_beam_cantilever_dynamic_input(
                    n_elem=p.get("n_elem", 10),
                    length=p.get("length", 10.0),
                    radius=p.get("radius", 0.1),
                    E=p.get("E", 1.0e7), nu=p.get("nu", 0.3),
                    dens=p.get("rho", 1.0),
                    moment_factor=p.get("moment_factor", 0.2),
                    numstep=p.get("numstep", 5),
                    timestep=p.get("timestep", 0.01)),
            # thermo/thermo_2d + thermo/thermo_3d previously fell
            # through to a one-line comment template ("# Thermal
            # template ...") that is not even a YAML dict, so
            # validate_input failed before the run stage (probe
            # 2026-06-12). Route to genuine PROBLEMTYPE "Thermo"
            # inline-mesh inputs (THERMO QUAD4/HEX8 + MAT_Fourier).
            # The 3D row keys resolution off "n" (NOT nx/ny/nz) so
            # the probe's nz=16 cannot inflate the cube mesh.
            "thermo_thermo_2d": lambda p: matched_thermo_2d_input(
                nx=min(int(p.get("nx", 16)), 32),
                ny=min(int(p.get("ny", 16)), 32),
                T_left=p.get("T_left", 100.0),
                T_right=p.get("T_right", 0.0),
                conductivity=p.get("kappa", 1.0)),
            "thermo_thermo_3d": lambda p: matched_thermo_3d_input(
                n=min(int(p.get("n", 6)), 8),
                T_left=p.get("T_left", 100.0),
                T_right=p.get("T_right", 0.0),
                conductivity=p.get("kappa", 1.0),
                capacity=p.get("capacity", 1.0),
                numstep=max(1, min(20, round(p.get("T_end", 0.5)
                                             / p.get("dt", 0.1)))),
                timestep=p.get("dt", 0.1)),
            # thermo_transient_mms/temporal_mms_2d: unsteady-heat MMS
            # family measured on TEMPORAL convergence order. Fixed fine
            # mesh keyed off "n" (capped in the inline builder),
            # dt-halving to the same T_end, so the only thing varying
            # is dt. One-Step-Theta is 2nd-order in time at theta=0.5
            # and 1st-order at theta=1; get the theta=0.5 order from Richardson
            # differences of consecutive-dt solutions rather than from
            # an error-vs-exact table, which saturates at the fixed
            # mesh's spatial floor. The volumetric
            # MMS source goes through the PLAIN "DESIGN SURF NEUMANN
            # CONDITIONS" — the THERMO-prefixed Neumann sections are
            # silently ignored in standalone Thermo (see the
            # thermo_transient_mms generator's pitfalls).
            "thermo_transient_mms_temporal_mms_2d":
                lambda p: matched_thermo_transient_mms_input(
                    n=int(p.get("n", 48)),
                    lx=p.get("lx", 1.0), ly=p.get("ly", 1.0),
                    kappa=p.get("kappa", 1.0),
                    rho=p.get("rho", 1.0), c=p.get("c", 1.0),
                    theta=p.get("theta", 0.5),
                    dt=p.get("dt", 0.02),
                    t_end=p.get("T_end", p.get("t_end", 0.4)),
                    temp_offset=p.get("temp_offset", 1.0),
                    amp=p.get("amp", 1.0),
                    grad_amp=p.get("grad_amp", 0.5),
                    mode_x=int(p.get("mode_x", 1)),
                    mode_y=int(p.get("mode_y", 1)),
                    omega=p.get("omega", 6.283185307179586),
                    time_profile=p.get("time_profile", "cos")),
            # Lubrication (Reynolds eq.) slider bearing: the placeholder
            # generator template emitted literal <...> scalars + an
            # external Exodus mesh, aborting 4C's MatchTree (probe
            # 2026-06-12). Route to the inline-mesh port of the corpus
            # case lubrication_sb_2d.4C.yaml (PURE_LUB, LUBRICATION
            # QUAD4, MAT_lubrication). Mesh capped small for < 30 s.
            "lubrication_slider_bearing_2d":
                lambda p: matched_lubrication_slider_bearing_input(
                    nx=min(int(p.get("nx", 16)), 32),
                    ny=min(int(p.get("ny", 1)), 4)),
            # mixture/mixture_3d previously returned a one-line comment
            # ("# Mixture template ...") — not a YAML dict, so
            # validate_input failed with "Input is not a YAML
            # dictionary" before the run stage (probe 2026-06-12). Route
            # to a self-contained inline HEX8 cube whose material is the
            # 4C Mixture toolbox (MAT_Mixture -> MIX_Rule_Simple ->
            # MIX_Constituent_ElastHyper -> ELAST_CoupLogNeoHooke).
            # Resolution keyed off "n" (NOT nx/ny/nz) so the probe's
            # nz=16 cannot inflate the cube.
            "mixture_mixture_3d": lambda p: matched_mixture_3d_input(
                n=min(int(p.get("n", 4)), 6),
                E=p.get("E", 1000.0), nu=p.get("nu", 0.3),
                density=p.get("rho", 0.1)),
            # constraint/constraint_3d previously returned a one-line
            # comment ("# Constraint template ...") — not a YAML dict, so
            # validate_input failed with "Input is not a YAML
            # dictionary" before the run stage (probe 2026-06-12). Route
            # to a self-contained inline HEX8 cube with a real
            # DESIGN POINT COUPLING CONDITION (multi-point coupling) that
            # ties the loaded face's transverse DOFs together.
            # Resolution keyed off "n" (NOT nx/ny/nz) so the probe's
            # nz=16 cannot inflate the cube.
            "constraint_constraint_3d":
                lambda p: matched_constraint_3d_input(
                    n=min(int(p.get("n", 4)), 6),
                    E=p.get("E", 1000.0), nu=p.get("nu", 0.3)),
            # membrane/membrane_2d + shell/shell_3d previously returned a
            # one-line comment from generators/membrane.py & shell.py
            # ("# Membrane template ...", "# Shell template ...") — not a
            # YAML dict, so validate_input failed with "Input is not a
            # YAML dictionary" before the run stage (probe 2026-06-12).
            # Route to self-contained inline structural inputs: a flat
            # MEMBRANE4 QUAD4 patch under a prescribed uniaxial stretch
            # (membranes are singular without prestress / full Dirichlet),
            # and a flat SHELL7P QUAD4 clamped cantilever under transverse
            # orthopressure. nx,ny capped <=16 for sub-30 s runtime; the
            # shell load scales with E so Newton converges at probe E.
            "membrane_membrane_2d":
                lambda p: matched_membrane_2d_input(
                    nx=min(int(p.get("nx", 8)), 16),
                    ny=min(int(p.get("ny", 8)), 16),
                    E=p.get("E", 1000.0), nu=p.get("nu", 0.3)),
            "shell_shell_3d":
                lambda p: matched_shell_3d_input(
                    nx=min(int(p.get("nx", 8)), 16),
                    ny=min(int(p.get("ny", 4)), 16),
                    E=p.get("E", 1000.0), nu=p.get("nu", 0.3)),
            # cardiovascular0d/windkessel_3d previously fell through to a
            # one-line comment generator template that is not even a YAML
            # dict, so validate_input failed before the run stage (probe
            # 2026-06-12). Route to the corpus-matched inline 0D-3D input:
            # a structural HEX8 cube coupled to a 4-element Windkessel via
            # DESIGN SURF CARDIOVASCULAR 0D conditions. Resolution keys off
            # "n" (NOT nx/ny/nz) so the probe's nz=16 cannot inflate the
            # monolithic 0D-3D solve; n is capped <=4 inside the helper.
            "cardiovascular0d_windkessel_3d":
                lambda p: matched_cardiovascular0d_windkessel_input(
                    n=min(int(p.get("n", 2)), 4),
                    E=p.get("E", 10.0), nu=p.get("nu", 0.3),
                    density=p.get("rho", 2e-6),
                    numstep=max(1, min(10, round(p.get("T_end", 0.3)
                                                 / p.get("dt", 0.1)))),
                    timestep=p.get("dt", 0.1)),
            # fluid/{channel_2d,cavity_2d} previously fell through to the
            # generator template with <placeholder> scalars + an external
            # Exodus mesh (FILE: "channel_2d.e", never produced) — 4C
            # aborted in MatchTree (probe 2026-06-26). Route to the
            # self-contained inline incompressible Navier-Stokes inputs
            # (FLUID QUAD4 + MAT_fluid + Np_Gen_Alpha + UMFPACK), based on
            # the corpus case f2_channel20x20_drt_weak.4C.yaml. nx,ny
            # capped for a sub-30 s monolithic solve.
            "fluid_cavity_2d": lambda p: matched_fluid_cavity_input(
                nx=min(int(p.get("nx", 16)), 32),
                ny=min(int(p.get("ny", 16)), 32),
                u_lid=p.get("u_lid", p.get("u_max", 1.0)),
                viscosity=p.get("mu", p.get("viscosity", 0.01)),
                density=p.get("rho", p.get("density", 1.0)),
                numstep=p.get("numstep", 10),
                timestep=p.get("dt", p.get("timestep", 0.1))),
            "fluid_channel_2d": lambda p: matched_fluid_channel_input(
                nx=min(int(p.get("nx", 24)), 48),
                ny=min(int(p.get("ny", 8)), 24),
                u_max=p.get("u_max", 1.0),
                viscosity=p.get("mu", p.get("viscosity", 0.01)),
                density=p.get("rho", p.get("density", 1.0)),
                numstep=p.get("numstep", 10),
                timestep=p.get("dt", p.get("timestep", 0.1))),
            # reduced_airways/airways_1d previously returned a reference
            # stub (no MATERIALS — validate_input flagged it not
            # runnable). The corpus case red_airway_3airway_2acinus_
            # awacinter.4C.yaml is small and fully self-contained (6
            # nodes, 5 elements, inline mesh, UMFPACK), so it ports to a
            # genuinely runnable inline input. numstep capped for a
            # sub-30 s solve.
            "reduced_airways_airways_1d":
                lambda p: matched_reduced_airways_input(
                    peak_pressure=p.get("peak_pressure", 30.0),
                    numstep=min(int(p.get("numstep", 200)), 2000),
                    period=p.get("period", 100.0)),
        }
        gen = inline_generators.get(key)
        if gen is None:
            raise ValueError(f"No inline generator for {key}")
        return gen(params)

    def _generate_from_tutorial(self, physics: str, variant: str, params: dict) -> str:
        """Generate input from 4C tutorial examples (with mesh files)."""
        tutorials = {
            # Poisson / scalar transport
            "poisson_poisson_2d": ("tutorials/poisson/tutorial_poisson_scatra.4C.yaml",
                                    "tutorials/poisson/tutorial_poisson_geo.e"),
            "poisson_heat_2d": ("tutorials/poisson/tutorial_poisson_thermo.4C.yaml",
                                 "tutorials/poisson/tutorial_poisson_geo.e"),
            "heat_heat_2d": ("tutorials/poisson/tutorial_poisson_thermo.4C.yaml",
                              "tutorials/poisson/tutorial_poisson_geo.e"),
            # Solid mechanics
            "linear_elasticity_linear_2d": ("tutorials/solid/tutorial_solid.4C.yaml",
                                             "tutorials/solid/tutorial_solid_geo_coarse.e"),
            "linear_elasticity_solid_tutorial": ("tutorials/solid/tutorial_solid.4C.yaml",
                                                  "tutorials/solid/tutorial_solid_geo_coarse.e"),
            # Fluid
            "fluid_channel_2d": ("tutorials/fluid/tutorial_fluid.4C.yaml",
                                  "tutorials/fluid/tutorial_fluid.e"),
            "fluid_cavity_2d": ("tutorials/fluid/tutorial_fluid.4C.yaml",
                                 "tutorials/fluid/tutorial_fluid.e"),
            # FSI
            "fsi_fsi_2d": ("tutorials/fsi/tutorial_fsi_2d.4C.yaml",
                            "tutorials/fsi/tutorial_fsi_2d.e"),
            "fsi_fsi_monolithic": ("tutorials/fsi/tutorial_fsi_monolithic.4C.yaml",
                                    "tutorials/fsi/tutorial_fsi_2d.e"),
            "fsi_fsi_3d": ("tutorials/fsi/tutorial_fsi_3d.4C.yaml",
                            "tutorials/fsi/tutorial_fsi_3d.e"),
            # Contact
            "contact_penalty_3d": ("tutorials/contact/tutorial_contact_3d.4C.yaml",
                                    "tutorials/contact/tutorial_contact_3d.e"),
        }
        key = f"{physics}_{variant}"
        if key not in tutorials:
            raise ValueError(f"No 4C tutorial for {key}")

        if not FOURC_ROOT:
            raise ValueError("FOURC_ROOT not set")

        yaml_path = FOURC_ROOT / "tests" / tutorials[key][0]
        if not yaml_path.exists():
            raise ValueError(f"Tutorial file not found: {yaml_path}")

        content = yaml_path.read_text()
        mesh_rel = tutorials[key][1]
        mesh_path = FOURC_ROOT / "tests" / mesh_rel
        if mesh_path.exists():
            # Give the RELATIVE location as well as the resolved one. The
            # absolute path is what this host needs to run the deck, but it is
            # also what an agent copies — and a served deck naming
            # `/home/<someone>/4C/tests/...` is a dead end on every other
            # machine. These meshes ship WITH 4C, so anyone who has 4C has them;
            # only the prefix differs. Naming FOURC_ROOT turns an unusable
            # absolute path into a locatable one.
            content = (f"# MESH_FILE: {mesh_path}\n"
                       f"# MESH_FILE_RELATIVE: $FOURC_ROOT/tests/{mesh_rel}"
                       f"  (ships with 4C; the absolute path above is this "
                       f"host's — resolve it against your own FOURC_ROOT)\n"
                       + content)
        return content

    def _resolve_mesh_references(self, content: str) -> str:
        """Find mesh file references in YAML and add MESH_FILE metadata."""
        import re
        # Look for FILE: xxx.e pattern
        match = re.search(r'FILE:\s*(\S+\.e)\b', content)
        if match and FOURC_ROOT:
            mesh_name = match.group(1)
            # Search for the mesh in tests/
            for mesh_path in FOURC_ROOT.rglob(mesh_name):
                try:
                    rel = mesh_path.relative_to(FOURC_ROOT)
                except ValueError:
                    rel = mesh_path.name
                content = (f"# MESH_FILE: {mesh_path}\n"
                           f"# MESH_FILE_RELATIVE: $FOURC_ROOT/{rel}"
                           f"  (the absolute path above is this host's — "
                           f"resolve it against your own FOURC_ROOT)\n"
                           + content)
                break
        return content

    def validate_input(self, content: str) -> list[str]:
        import re
        import yaml
        errors = []

        # ── Hard rejections that would make 4C abort ──────────────────
        # 1. Un-substituted <...> placeholders (e.g. <number_of_steps>,
        #    <dynamic_viscosity>). 4C's input matcher aborts on these.
        #    Skip matches inside comment lines so prose like "<...>" in a
        #    reference stub is not flagged; scan only non-comment text.
        non_comment = "\n".join(
            ln for ln in content.splitlines()
            if not ln.lstrip().startswith("#"))
        placeholders = re.findall(
            r"<[A-Za-z_][A-Za-z0-9_ ]*>", non_comment)
        if placeholders:
            uniq = sorted(set(placeholders))
            errors.append(
                "Un-substituted placeholder(s) found "
                f"({', '.join(uniq[:5])}): the deck is not runnable — "
                "every <...> must be replaced with a concrete value.")

        # 2. External mesh FILE: references (e.g. FILE: \"channel_2d.e\").
        #    These abort 4C unless the mesh travels with the deck. A
        #    legitimate tutorial deck carries a leading
        #    '# MESH_FILE: <path>' header that run() copies into the work
        #    dir; allow the FILE: ref only when that header resolves to an
        #    existing file.
        mesh_refs = re.findall(
            r'FILE:\s*["\']?([^\s"\']+\.(?:e|exo|dat|bin))\b',
            non_comment)
        if mesh_refs:
            header_path = None
            first_line = content.splitlines()[0] if content else ""
            if first_line.startswith("# MESH_FILE: "):
                header_path = Path(
                    first_line.split(": ", 1)[1].strip())
            if header_path is None or not header_path.exists():
                errors.append(
                    "External mesh FILE: reference(s) "
                    f"({', '.join(sorted(set(mesh_refs))[:3])}) with no "
                    "bundled/resolvable mesh — the deck references a mesh "
                    "file that is not produced. Use an inline NODE COORDS "
                    "mesh or a '# MESH_FILE:' header pointing at an "
                    "existing file.")

        # ── Structural YAML checks ────────────────────────────────────
        try:
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                errors.append("Input is not a YAML dictionary")
                return errors
            if "PROBLEM TYPE" not in data:
                errors.append("Missing PROBLEM TYPE section")
            if "MATERIALS" not in data:
                errors.append("Missing MATERIALS section")
        except yaml.YAMLError as e:
            errors.append(f"YAML parse error: {e}")
        return errors

    async def run(self, input_content: str, work_dir: Path,
                  np: int = 1, timeout=None) -> JobHandle:
        binary = _find_fourc_binary()
        if not binary:
            return JobHandle(
                job_id=str(uuid.uuid4())[:8],
                backend_name="fourc",
                work_dir=work_dir,
                status="failed",
                error="4C binary not found",
            )

        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        # Extract mesh file path if embedded in content
        mesh_src = None
        lines = input_content.splitlines()
        if lines and lines[0].startswith("# MESH_FILE: "):
            mesh_src = Path(lines[0].split(": ", 1)[1].strip())
            input_content = "\n".join(lines[1:])

        input_file = work_dir / "input.4C.yaml"
        input_file.write_text(input_content)

        # Copy mesh file if referenced
        if mesh_src and mesh_src.exists():
            import shutil as _shutil
            _shutil.copy2(mesh_src, work_dir / mesh_src.name)

        output_prefix = str(work_dir / "output")

        mpirun = shutil.which("mpirun")
        max_procs = int(os.environ.get("FOURC_MAX_PROCS", "4"))
        np = min(np, max_procs)

        # Wrap with stdbuf -oL to force line-buffered stdout.
        # 4C writes errors to stdout (buffered) then calls MPI_Abort which
        # kills the process before flushing — stdbuf prevents lost messages.
        stdbuf = shutil.which("stdbuf")

        if np > 1 and mpirun:
            base_cmd = [mpirun, "-np", str(np), str(binary), str(input_file), output_prefix]
        else:
            base_cmd = [str(binary), str(input_file), output_prefix]

        cmd = [stdbuf, "-oL"] + base_cmd if stdbuf else base_cmd

        job_id = str(uuid.uuid4())[:8]
        job = JobHandle(job_id=job_id, backend_name="fourc", work_dir=work_dir, status="running")

        start = time.time()
        job.started_at = start
        try:
            env = os.environ.copy()
            # Ensure 4C dependencies are on the library path
            ld_path = env.get("LD_LIBRARY_PATH", "")
            dep_lib = "/opt/4C-dependencies/lib"
            if dep_lib not in ld_path:
                ld_path = f"{dep_lib}:{ld_path}" if ld_path else dep_lib
            env["LD_LIBRARY_PATH"] = ld_path

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
                env=env,
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
                # 4C WRITES THE REAL ERROR BEFORE THE MPI BOILERPLATE, SO A
                # TAIL IS EXACTLY THE WRONG END.
                #
                # This line already knew "4C often writes the real error to
                # stdout, not stderr" -- and then took stdout_text[-1000:].
                # Measured on a deliberately malformed deck, 4C prints:
                #
                #   Invalid MIT-MAGIC-COOKIE-1 key         <- X11 noise
                #   ***** 4C version 2026.2.0-dev *****    <- banner
                #   Trilinos Version: ...
                #   ====================================
                #   PROC 0 ERROR in 4C_io_input_spec_builders.cpp, line 633:
                #   Could not match this input
                #   STRUCTURAL DYNAMIC:
                #     INT_STRATEGY: "Standard"             <- the actual defect
                #   ... then ~40 lines of MPI_ABORT boilerplate
                #
                # The last 1000 characters are entirely that boilerplate, so the
                # agent was handed "Invalid MIT-MAGIC-COOKIE-1 key ... MPI_ABORT
                # was invoked on rank 0" and nothing else.
                #
                # THE COST, MEASURED: two recorded runs of one coupled
                # problem concluded from exactly that string that the 4C BINARY
                # was broken on this machine, wrote a could-not-finish report
                # with zero deliverables, and stopped at 30 and 37 tool calls
                # having used 22-27% of their wall budget. Another run of the
                # same problem, running 4C directly and reading the head of its
                # own log, produced a complete three-level study from the same
                # binary in the same minutes. The binary was never broken: `4C
                # -p` prints the cookie line too and succeeds.
                #
                # So the diagnostic is now located BY CONTENT. The tails are
                # kept as a fallback, after it.
                stdout_text = stdout.decode(errors="replace")
                stderr_text = stderr.decode(errors="replace")
                job.error = _fourc_diagnostic(stdout_text, stderr_text)
            else:
                # 4C's runtime VTK writer covers STRUCTURE, FLUID and BEAMS only. A run that
                # produced no runtime output (Thermo, Ale, Lubrication, ...) is converted to VTU
                # here so its result can be opened at all; a run that has VTK is left alone.
                # (The comment this replaces claimed every template requests runtime output;
                # twelve served decks did not, measured 2026-09-24.)
                await self._convert_native_output_to_vtu(work_dir, binary, since=start)
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

    async def _convert_native_output_to_vtu(self, work_dir: Path, binary=None,
                                            since: float | None = None) -> None:
        """Make a run that wrote only 4C-native output readable, by converting it to VTU.

        4C's runtime VTK writer exists for STRUCTURE, FLUID and BEAMS only. A Thermo, Ale,
        Lubrication or ReducedAirways run -- and any deck that requests no runtime output --
        leaves just <prefix>.control plus .result/.mesh binaries, which neither a model nor
        openPASO's own result gate can open. Measured 2026-09-24: twelve served decks did exactly
        that and every one read "output unreadable" to the coverage judge. This backend already
        carried a converter step that never fired: it looked for a binary named post_vtu (this
        build ships post_processor --filter=vtu), launched it fire-and-forget with its output on
        /dev/null, and run() never called it, behind a comment saying every template requests
        runtime output.

        Runs ONLY when no runtime VTK exists and a control file does; waits for the converter
        (bounded); keeps its console in post_processor.log beside the results. A failed
        conversion (the 1-D airways result type is one, measured) leaves the solve as it was:
        the result simply stays unreadable and the log says why.

        `since` is when this run started. A reused work directory can hold an earlier
        run's VTU: counted, it made this run skip its conversion and hand back the
        earlier run's fields as its own result (Copilot review of the org PR, verified
        in the code). Only files written since the run started count.
        """
        def _this_run(paths):
            return [q for q in paths if since is None or q.stat().st_mtime >= since - 1.0]
        # PER FIELD, NOT PER FOLDER. Measured 2026-09-28 on 4C 2026.3.0 (the interface
        # session): a poro-fluid run writes a runtime VTK series "fluid-poro" by itself, while
        # its structure, fluid and porofluid fields exist only as native results; the folder
        # held a .vtu, so nothing was converted and three fields stayed unreadable. A field is
        # covered when a VTU series of its own name exists (series = the file name without its
        # trailing -NNNNN step and rank numbers); convert when any native field is not.
        import re as _re
        native = {m.group(1) for q in _this_run(work_dir.glob("*.result.*"))
                  for m in [_re.search(r"\.result\.([A-Za-z0-9_]+)\.s\d+", q.name)] if m}
        series = set()
        for q in _this_run(list(work_dir.rglob("*.vtu")) + list(work_dir.rglob("*.pvtu"))
                           + list(work_dir.rglob("*.pvd"))):
            stem = _re.sub(r"(-\d+)+$", "", q.stem)
            series.add(stem)
            series.add(stem.split("-", 1)[1] if "-" in stem else stem)   # "<prefix>-<field>"
        if series and native <= series:
            return                                   # every native field has runtime output
        controls = sorted(_this_run(work_dir.glob("*.control")))
        if not controls:
            return
        candidates: list[Path] = []
        if binary:
            here = Path(str(binary)).parent
            candidates += [here / "post_processor", here / "post_vtu"]
        if FOURC_ROOT:
            for d in ("build", "build/release"):
                candidates += [FOURC_ROOT / d / "post_processor", FOURC_ROOT / d / "post_vtu"]
        for name in ("post_processor", "post_vtu"):
            found = shutil.which(name)
            if found:
                candidates.append(Path(found))
        tool = next((c for c in candidates if c.is_file()), None)
        log = work_dir / "post_processor.log"
        if tool is None:
            log.write_text("no post_processor (or post_vtu) beside the 4C binary, under FOURC_ROOT/build or on "
                           "PATH: the native result files could not be converted to VTU\n")
            logger.warning("4C run wrote only native output and no post_processor was found; result stays unreadable")
            return
        env = os.environ.copy()
        env.pop("DISPLAY", None)                      # the converter links VTK; a stray X display only adds noise
        dep_lib = "/opt/4C-dependencies/lib"
        ld = env.get("LD_LIBRARY_PATH", "")
        if dep_lib not in ld:
            env["LD_LIBRARY_PATH"] = f"{dep_lib}:{ld}" if ld else dep_lib
        for ctrl in controls:
            prefix = str(ctrl)[: -len(".control")]
            args = [str(tool), f"--file={prefix}"]
            if tool.name == "post_processor":
                args += ["--filter=vtu", "--postprocessor_deprecation_warning_off"]
            elif tool.read_bytes()[:2] == b"#!":
                # this build's post_vtu is a script that adds --filter=vtu and passes the
                # rest on to post_processor, which otherwise stops at "Press Enter to
                # continue" (measured); an old compiled post_vtu does not know the flag
                args += ["--postprocessor_deprecation_warning_off"]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                    cwd=str(work_dir), env=env, stdin=asyncio.subprocess.DEVNULL)
                try:
                    out, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
                except asyncio.TimeoutError:
                    proc.kill()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=30)   # reaped, not left behind
                    except asyncio.TimeoutError:
                        pass
                    with log.open("a") as fh:
                        fh.write(f"$ {' '.join(args)}\nTIMED OUT after 300 s\n")
                    logger.warning(f"post_processor timed out on {ctrl.name}")
                    continue
                with log.open("a") as fh:
                    fh.write(f"$ {' '.join(args)}\nexit {proc.returncode}\n{out.decode(errors='replace')}\n")
                if proc.returncode != 0:
                    logger.warning(f"post_processor exit {proc.returncode} on {ctrl.name}; see post_processor.log")
            except Exception as e:  # noqa: BLE001
                with log.open("a") as fh:
                    fh.write(f"$ {' '.join(args)}\nFAILED TO LAUNCH: {e}\n")
                logger.warning(f"post_processor launch failed on {ctrl.name}: {e}")

    def get_result_files(self, job: JobHandle) -> list[Path]:
        results = []
        for ext in ["*.vtu", "*.pvd", "*.pvtu"]:
            results.extend(job.work_dir.rglob(ext))
        if job.started_at is not None:
            # an earlier run's files in a reused work directory are not this run's result
            results = [q for q in results if q.stat().st_mtime >= job.started_at - 1.0]
        return sorted_by_step(results)


def register():
    register_backend(
        FourcBackend(),
        aliases=["4c", "4C", "fourc", "four_c"],
    )
