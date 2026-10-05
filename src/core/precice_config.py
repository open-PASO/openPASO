"""
preCICE coupling — a first-class member of openPASO.

preCICE (precice.org) is the field-standard library for black-box coupling of
independent simulation codes. In openPASO it is the native coupling path for genuinely
forced multi-paradigm couplings — e.g. a DSMC particle code (SPARTA) coupled to a FEM
solid (FEniCS) for conjugate heat transfer — that the file-handshake driver cannot
express as cleanly.

This module (1) detects the preCICE Python bindings + C++ library, (2) generates valid
preCICE-config.xml for standard coupling scenarios (heat DN, TSI, and generic), and
(3) verifies an end-to-end coupling actually runs. The libprecice shared library lives
at PRECICE_LIB_DIR (default /opt/precice/lib); pyprecice must match its version.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger("openpaso.precice")

# Location of libprecice.so.* — pyprecice loads it at import time.
PRECICE_LIB_DIR = os.environ.get("PRECICE_LIB_DIR", "/opt/precice/lib")


def _ensure_lib_on_path() -> None:
    """Make libprecice loadable for the pyprecice import.

    Sets LD_LIBRARY_PATH (for child processes) AND ctypes-preloads libprecice into the
    current process with RTLD_GLOBAL — necessary because changing LD_LIBRARY_PATH at
    runtime does NOT affect the already-initialized dynamic linker of this process.
    """
    if not Path(PRECICE_LIB_DIR).exists():
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if PRECICE_LIB_DIR not in cur.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (PRECICE_LIB_DIR + ":" + cur).rstrip(":")
    import ctypes
    for cand in ("libprecice.so.3", "libprecice.so"):
        try:
            ctypes.CDLL(str(Path(PRECICE_LIB_DIR) / cand), mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


def check_precice_available() -> tuple[bool, str]:
    """Check that the preCICE Python bindings + C++ library are usable.

    Returns (ok, message). Sets LD_LIBRARY_PATH so the bindings can find libprecice.
    """
    _ensure_lib_on_path()
    try:
        import precice
        info = precice.get_version_information()
        ver = info.decode().split(";")[0] if isinstance(info, bytes) else str(info)
        return True, f"preCICE {ver} (lib at {PRECICE_LIB_DIR})"
    except ImportError as e:
        return False, (
            f"preCICE Python bindings not importable ({e}). Install a pyprecice whose "
            f"version matches libprecice in {PRECICE_LIB_DIR}: pip install 'pyprecice==<libver>'"
        )
    except Exception as e:
        return False, f"preCICE present but unusable: {e}"


# Data names that carry a flux/force/traction — an INTEGRATED quantity. Mapping
# those with constraint="consistent" across non-matching meshes preserves the
# nodal value and NOT the integral, so the total crossing the interface is not
# conserved; the run completes cleanly and nothing says so.
_CONSERVATIVE_HINTS = ("flux", "force", "traction", "load", "heatflux",
                       "heat-flux", "heat_flux", "pressure-force")


def validate_precice_spec(participants: list, data: list, exchanges: list, *,
                          scheme: str = "serial-explicit",
                          time_window: float = 1.0,
                          max_time: float = 10.0) -> list[str]:
    """Refuse a preCICE configuration that would run but not couple.

    Every check here corresponds to a config the generator used to emit happily:
    an empty exchange list (preCICE rejects that config in each participant's
    own Participant constructor, before its version banner and without waiting
    for a partner: "Tag <exchange> was not found but is required to occur at
    least once."), a participant that reads a field nobody sends it (a bare
    KeyError out of the generator, or worse a full receive-mesh block with no
    m2n behind it), a third participant that appears in no exchange and is
    therefore in no coupling scheme, which preCICE aborts in its Participant
    constructor ("No coupling scheme defined for participant ...", exit 255),
    and a serial-implicit acceleration on data flowing the wrong way, which
    preCICE refuses to load.

    Returns advisory notes (worth saying, not worth refusing over) and raises
    ValueError, naming the offending entry, for the rest.
    """
    notes: list[str] = []
    names = [p.get("name") for p in participants]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate participant names: {names}")
    if len(participants) < 2:
        raise ValueError("a preCICE coupling needs at least 2 participants")
    data_names = {d["name"] for d in data}
    if not exchanges:
        raise ValueError(
            "no exchanges: an empty exchange list generates a config with no data "
            "transfer at all. preCICE rejects it only at run time, as each solver "
            "constructs its Participant (\"Tag <exchange> was not found but is "
            "required to occur at least once.\"), not when the config is generated.")
    for ex in exchanges:
        for key in ("data", "from", "to"):
            if key not in ex:
                raise ValueError(f"exchange {ex!r} is missing '{key}'")
        if ex["data"] not in data_names:
            raise ValueError(f"exchange sends undeclared data {ex['data']!r}; "
                             f"declared: {sorted(data_names)}")
        for key in ("from", "to"):
            if ex[key] not in names:
                raise ValueError(f"exchange {key}={ex[key]!r} is not a participant; "
                                 f"participants: {names}")
        if ex["from"] == ex["to"]:
            raise ValueError(f"exchange {ex['data']!r} goes from {ex['from']} to itself")
    delivered = {(ex["data"], ex["to"]) for ex in exchanges}
    sent = {(ex["data"], ex["from"]) for ex in exchanges}
    for p in participants:
        for dn in p.get("reads", []):
            if (dn, p["name"]) not in delivered:
                raise ValueError(
                    f"participant {p['name']!r} declares reads={dn!r} but no "
                    "exchange delivers it there. preCICE would still be handed a "
                    "receive-mesh and a read-data tag, and the participant would "
                    "read zeros forever without any error.")
        for dn in p.get("writes", []):
            if (dn, p["name"]) not in sent:
                notes.append(f"{p['name']} writes {dn} but no exchange carries it "
                             "anywhere — that data goes nowhere")
    touched = {n for ex in exchanges for n in (ex["from"], ex["to"])}
    orphan = [n for n in names if n not in touched]
    if orphan:
        raise ValueError(
            f"participant(s) {orphan} take part in no exchange. The generated "
            "config gives them no m2n and no place in the coupling scheme, and "
            "preCICE aborts each of them in its Participant constructor with "
            "\"No coupling scheme defined for participant ...\" (exit 255) while "
            "the others couple normally.")
    if len(participants) > 2:
        raise ValueError(
            f"{len(participants)} participants: this generator emits a two-participant "
            "coupling scheme (<participants first=... second=.../>), so a third "
            "participant is silently left out of it. A genuine N>2 coupling needs "
            "<coupling-scheme:multi>, which is not generated here.")
    if time_window > max_time:
        raise ValueError(f"time-window {time_window} exceeds max-time {max_time}: "
                         "no coupling window would ever complete")
    if "implicit" in scheme and "serial" in scheme:
        if not [ex for ex in exchanges if ex["to"] == names[0]]:
            raise ValueError(
                f"serial-implicit needs data flowing from the second participant "
                f"({names[1]}) back to the first ({names[0]}) to accelerate on; no "
                "exchange does. preCICE refuses to load such a config.")
    for ex in exchanges:
        if any(h in ex["data"].lower() for h in _CONSERVATIVE_HINTS):
            notes.append(
                f"{ex['data']} looks like an integrated quantity (flux/force/"
                'traction) but is mapped with constraint="consistent", which '
                "preserves nodal values and NOT the integral. On non-matching "
                "interface meshes the total crossing the interface is then not "
                "conserved, and nothing in the run reports it.")
    return notes


def generate_precice_config(
    participants: list,
    data: list,
    exchanges: list,
    *,
    scheme: str = "serial-explicit",
    dimensions: int = 2,
    time_window: float = 1.0,
    max_time: float = 10.0,
    max_iterations: int = 20,
    convergence_tol: float = 1e-6,
    acceleration: dict = None,
    mapping: str = "nearest-neighbor",
    strict: bool = False,
    initial_relaxation: float = 0.5,
) -> str:
    """Generate a preCICE XML config for an ARBITRARY cross-code coupling.

    Fully general — any number of participants, any data fields, any exchange pattern.
    This is the backend for all coupling scenarios (heat, TSI, FSI, DSMC-FEM, ...).

    Args:
        participants: list of dicts, one per coupled code:
            {"name": str, "mesh": str, "writes": [data_name, ...], "reads": [data_name, ...]}
        data: list of dicts describing exchanged fields:
            {"name": str, "type": "scalar" | "vector"}
        exchanges: list of dicts, one per coupled field transfer:
            {"data": str, "from": participant_name, "to": participant_name}
        scheme: "serial-explicit" | "serial-implicit" | "parallel-explicit" | "parallel-implicit"
        dimensions: spatial dimension of the coupling meshes (2 or 3)
        time_window, max_time: coupling time control
        max_iterations, convergence_tol: implicit-scheme controls (ignored for explicit)
        acceleration: {"type": "aitken"|"IQN-ILS", "data": data_name, "mesh": mesh_name,
                       "initial_relaxation": float} — for implicit schemes. Without it,
                       serial-implicit relaxes a datum sent from the second participant to
                       the first (preCICE requires that), and parallel-implicit relaxes every
                       exchanged datum.
        mapping: "nearest-neighbor" | "nearest-projection" | "rbf"

    Returns:
        Complete preCICE XML configuration as a string.
    """
    implicit = "implicit" in scheme
    names = [p["name"] for p in participants]
    mesh_of = {p["name"]: p["mesh"] for p in participants}
    validate_precice_spec(participants, data, exchanges, scheme=scheme,
                          time_window=time_window, max_time=max_time)
    # writer (participant, mesh) of each data field, from the exchange list
    writer_mesh = {}
    for ex in exchanges:
        writer_mesh[ex["data"]] = (ex["from"], mesh_of[ex["from"]])

    # --- <data> tags ---
    data_xml = "\n".join(
        f'  <data:{d.get("type","scalar")} name="{d["name"]}" />' for d in data)

    # --- <mesh> tags (each participant provides one mesh, using all data it touches) ---
    mesh_xml = []
    for p in participants:
        used = sorted(set(p.get("writes", []) + p.get("reads", [])))
        uses = "".join(f'\n    <use-data name="{dn}" />' for dn in used)
        mesh_xml.append(f'  <mesh name="{p["mesh"]}" dimensions="{dimensions}">{uses}\n  </mesh>')
    mesh_xml = "\n".join(mesh_xml)

    # --- <participant> tags ---
    part_xml = []
    for p in participants:
        lines = [f'  <participant name="{p["name"]}">',
                 f'    <provide-mesh name="{p["mesh"]}" />']
        # meshes this participant must receive (sources of data it reads)
        recv = {}
        for dn in p.get("reads", []):
            src_p, src_m = writer_mesh[dn]
            recv.setdefault((src_p, src_m), []).append(dn)
        for (src_p, src_m) in recv:
            lines.append(f'    <receive-mesh name="{src_m}" from="{src_p}" />')
        for dn in p.get("writes", []):
            lines.append(f'    <write-data name="{dn}" mesh="{p["mesh"]}" />')
        for dn in p.get("reads", []):
            lines.append(f'    <read-data name="{dn}" mesh="{p["mesh"]}" />')
        for (src_p, src_m) in recv:
            lines.append(f'    <mapping:{mapping} direction="read" from="{src_m}" '
                         f'to="{p["mesh"]}" constraint="consistent" />')
        lines.append("  </participant>")
        part_xml.append("\n".join(lines))
    part_xml = "\n".join(part_xml)

    # --- m2n connections (pairwise between participants that exchange) ---
    pairs = set()
    for ex in exchanges:
        pairs.add(tuple(sorted((ex["from"], ex["to"]))))
    m2n_xml = "\n".join(
        f'  <m2n:sockets acceptor="{a}" connector="{b}" exchange-directory="." />'
        for (a, b) in sorted(pairs))

    # --- coupling scheme ---
    exch_xml = "\n".join(
        f'    <exchange data="{ex["data"]}" mesh="{writer_mesh[ex["data"]][1]}" '
        f'from="{ex["from"]}" to="{ex["to"]}" />' for ex in exchanges)
    cs = [f'  <coupling-scheme:{scheme}>',
          f'    <time-window-size value="{time_window}" />',
          f'    <max-time value="{max_time}" />',
          f'    <participants first="{names[0]}" second="{names[1]}" />',
          exch_xml]
    if implicit:
        cs.append(f'    <max-iterations value="{max_iterations}" />')
        # A convergence measure PER EXCHANGED FIELD. With a measure on only one
        # field (this used to be exchanges[0], i.e. whichever the caller happened
        # to list first) the scheme declares convergence while every other
        # exchanged quantity is still moving — the same masking the file-handshake
        # driver guards against with per-block residuals.
        for ex in exchanges:
            dn = ex["data"]
            cs.append(f'    <relative-convergence-measure limit="{convergence_tol}" '
                      f'data="{dn}" mesh="{writer_mesh[dn][1]}"'
                      + (' strict="true"' if strict else '') + ' />')
        # preCICE requires the acceleration datum of a SERIAL-implicit scheme to
        # be exchanged second -> first. The old default (exchanges[0]) produced a
        # config preCICE refuses to load, which is how both convenience wrappers
        # in this file shipped broken.
        if "serial" in scheme:
            back = [ex["data"] for ex in exchanges if ex["to"] == names[0]]
            if acceleration and acceleration.get("data") not in back:
                raise ValueError(
                    "serial-implicit acceleration must use data exchanged from "
                    f"{names[1]} to {names[0]}; {acceleration.get('data')!r} is not "
                    f"(candidates: {back}). preCICE refuses such a config.")
            acc_data = [acceleration["data"] if acceleration else back[0]]
        elif acceleration:
            acc_data = [acceleration["data"]]
        else:
            # A PARALLEL-implicit scheme relaxes EVERY exchanged field by default. Both
            # participants solve from the other's previous iterate, so a relaxation on one
            # field leaves the other unrelaxed. Measured 2026-09-30 with the served FEniCSx
            # loop (a transient heat solid against a lumped wall model, 10 windows,
            # max-iterations 50): Aitken on the heat flux alone ran every window to
            # max-iterations, and the wall temperature ended NaN while both participants
            # exited 0; Aitken on both fields converged to the serial-implicit wall
            # temperature within 3e-9. preCICE 3.1.2 accepts several data in one Aitken block.
            acc_data = [ex["data"] for ex in exchanges]
        acc = acceleration or {"type": "aitken", "initial_relaxation": initial_relaxation}
        cs.append(f'    <acceleration:{acc["type"]}>')
        for dn in acc_data:
            mesh_of = acc["mesh"] if acceleration else writer_mesh[dn][1]
            cs.append(f'      <data mesh="{mesh_of}" name="{dn}" />')
        cs.append(f'      <initial-relaxation value="{acc.get("initial_relaxation",0.5)}" />')
        cs.append(f'    </acceleration:{acc["type"]}>')
    cs.append(f'  </coupling-scheme:{scheme}>')
    cs_xml = "\n".join(cs)

    return (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        '<precice-configuration>\n'
        '  <!-- generated by openPASO generate_precice_config (general) -->\n'
        f'{data_xml}\n\n{mesh_xml}\n\n{part_xml}\n\n{m2n_xml}\n\n{cs_xml}\n'
        '</precice-configuration>\n'
    )


def generate_heat_coupling_config(
    mesh_name_a: str = "FEniCS-Mesh",
    mesh_name_b: str = "4C-Mesh",
    data_name: str = "Temperature",
    flux_name: str = "Heat-Flux",
    max_iterations: int = 20,
    convergence_tol: float = 1e-6,
    relaxation: float = 0.5,
    time_window: float = 1.0,
    output_dir: str = "precice-output",
) -> str:
    """Generate preCICE XML config for Dirichlet-Neumann heat coupling.

    This produces a configuration equivalent to our MCP-orchestrated DN
    domain decomposition, enabling direct comparison.

    Args:
        mesh_name_a: Mesh name for the Dirichlet participant (FEniCS).
        mesh_name_b: Mesh name for the Neumann participant (4C).
        data_name: Name of the temperature data field.
        flux_name: Name of the heat flux data field.
        max_iterations: Maximum implicit coupling iterations.
        convergence_tol: Convergence criterion for the coupling.
        relaxation: Initial relaxation factor for Aitken acceleration.
        time_window: Size of coupling time window.
        output_dir: Directory for preCICE output.

    Returns:
        Complete preCICE XML configuration as a string.
    """
    return generate_precice_config(
        participants=[
            {"name": "Dirichlet", "mesh": mesh_name_a,
             "writes": [data_name], "reads": [flux_name]},
            {"name": "Neumann", "mesh": mesh_name_b,
             "writes": [flux_name], "reads": [data_name]},
        ],
        data=[{"name": data_name, "type": "scalar"},
              {"name": flux_name, "type": "scalar"}],
        exchanges=[{"data": data_name, "from": "Dirichlet", "to": "Neumann"},
                   {"data": flux_name, "from": "Neumann", "to": "Dirichlet"}],
        scheme="serial-implicit", dimensions=2, time_window=time_window,
        max_time=time_window, max_iterations=max_iterations,
        convergence_tol=convergence_tol,
        # preCICE accelerates a SERIAL-implicit scheme on data flowing from the
        # second participant back to the first, i.e. the FLUX here. Naming the
        # temperature produced a config preCICE refuses to load.
        acceleration={"type": "aitken", "data": flux_name, "mesh": mesh_name_b,
                      "initial_relaxation": relaxation},
    )


def generate_tsi_coupling_config(
    mesh_name_a: str = "Thermal-Mesh",
    mesh_name_b: str = "Structure-Mesh",
    temperature_name: str = "Temperature",
    displacement_name: str = "Displacement",
    heat_flux_name: str = "Heat-Flux",
    traction_name: str = "Force",
    max_iterations: int = 30,
    convergence_tol: float = 1e-6,
    time_window: float = 1.0,
) -> str:
    """Generate preCICE XML for thermal-structural interaction coupling.

    Two-way coupling: temperature and displacement exchanged between solvers.

    Returns:
        Complete preCICE XML configuration string.
    """
    return generate_precice_config(
        participants=[
            {"name": "Thermal", "mesh": mesh_name_a,
             "writes": [temperature_name], "reads": [displacement_name]},
            {"name": "Structure", "mesh": mesh_name_b,
             "writes": [displacement_name], "reads": [temperature_name]},
        ],
        data=[{"name": temperature_name, "type": "scalar"},
              {"name": displacement_name, "type": "vector"}],
        exchanges=[{"data": temperature_name, "from": "Thermal", "to": "Structure"},
                   {"data": displacement_name, "from": "Structure", "to": "Thermal"}],
        scheme="serial-implicit", dimensions=3, time_window=time_window,
        max_time=time_window, max_iterations=max_iterations,
        convergence_tol=convergence_tol,
        # Same serial-implicit rule: accelerate on Structure -> Thermal data.
        acceleration={"type": "aitken", "data": displacement_name,
                      "mesh": mesh_name_b, "initial_relaxation": 0.5},
    )


def save_precice_config(config_xml: str, output_dir: Path, filename: str = "precice-config.xml") -> Path:
    """Save preCICE configuration to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(config_xml)
    logger.info(f"preCICE config saved to {path}")
    return path


# Verification config — built via the GENERAL generator so the general path is exercised.
_VERIFY_CONFIG = generate_precice_config(
    participants=[{"name": "A", "mesh": "A-Mesh", "writes": ["Val"], "reads": []},
                  {"name": "B", "mesh": "B-Mesh", "writes": [], "reads": ["Val"]}],
    data=[{"name": "Val", "type": "scalar"}],
    exchanges=[{"data": "Val", "from": "A", "to": "B"}],
    scheme="serial-explicit", dimensions=2, time_window=1.0, max_time=3.0)

_VERIFY_PARTICIPANT = """\
import sys, numpy as np, precice
name, other = sys.argv[1], sys.argv[2]
p = precice.Participant(name, "precice-config.xml", 0, 1)
mesh = f"{name}-Mesh"
vid = p.set_mesh_vertices(mesh, np.array([[0.0, 0.0]]))
p.initialize()
while p.is_coupling_ongoing():
    dt = p.get_max_time_step_size()
    if name == "B":
        v = p.read_data(mesh, "Val", vid, dt)
        print(f"B received {float(v[0])}", flush=True)
    if name == "A":
        p.write_data(mesh, "Val", vid, np.array([42.0]))
    p.advance(dt)
p.finalize()
"""


def run_precice_coupling(
    participants: list,
    data: list,
    exchanges: list,
    work_dir: Path,
    *,
    scheme: str = "serial-explicit",
    dimensions: int = 2,
    max_time: float = 10.0,
    time_window: float = 1.0,
    timeout: int = 1800,
    extra_env: dict = None,
    **config_kw,
) -> dict:
    """Run a GENERAL preCICE coupling of arbitrary codes, end-to-end.

    openPASO generates the preCICE config and launches every participant's solver command,
    then waits for the coupling to finish. This is the physics-agnostic, code-agnostic
    coupling orchestrator — the same call drives heat DN, TSI, FSI, DSMC<->FEM, etc.

    Args:
        participants: one dict per coupled code, combining the config spec and how to run it:
            {"name": str, "mesh": str, "writes": [data], "reads": [data], "command": [argv...]}
        data:      [{"name": str, "type": "scalar"|"vector"}]
        exchanges: [{"data": str, "from": name, "to": name}]
        work_dir:  directory to run in (config + participant cwd)
        scheme/dimensions/max_time/time_window/**config_kw: passed to generate_precice_config
        extra_env: extra environment for participant processes (e.g. LD_LIBRARY_PATH for
                   libprecice / solver libraries)

    Returns:
        {"converged": bool, "returncodes": {name: rc}, "config": path, "logs": {name: tail}}
    """
    import signal
    import subprocess
    import time as _time
    _ensure_lib_on_path()
    work_dir = Path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    notes = validate_precice_spec(participants, data, exchanges, scheme=scheme,
                                  time_window=time_window, max_time=max_time)
    cfg = generate_precice_config(
        participants=[{k: p[k] for k in ("name", "mesh", "writes", "reads")} for p in participants],
        data=data, exchanges=exchanges, scheme=scheme, dimensions=dimensions,
        max_time=max_time, time_window=time_window, **config_kw)
    cfg_path = work_dir / "precice-config.xml"
    cfg_path.write_text(cfg)
    # preCICE re-creates (truncates) precice-<name>-iterations.log and
    # -convergence.log only when an implicit scheme initializes. An explicit
    # run, or one that dies before initialize(), leaves an earlier run's logs in
    # place, where they would be read as this run's evidence.
    for stale in list(work_dir.glob("precice-*-iterations.log")) + \
            list(work_dir.glob("precice-*-convergence.log")):
        try:
            stale.unlink()
        except OSError:
            pass
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = PRECICE_LIB_DIR + ":" + env.get("LD_LIBRARY_PATH", "")
    if extra_env:
        for k, v in extra_env.items():
            env[k] = v + ":" + env.get(k, "") if k.endswith("PATH") else v

    procs: dict = {}
    rcs: dict = {}
    logs: dict = {}
    error = None

    def _kill_all():
        for pr, _lf in procs.values():
            if pr.poll() is None:
                try:
                    os.killpg(pr.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pr.kill()

    try:
        for p in participants:
            lf = open(work_dir / f"{p['name']}.out", "w")
            # start_new_session so an MPI-launched participant's whole process
            # group can be killed; pr.kill() alone orphans the ranks.
            try:
                pr = subprocess.Popen(
                    p["command"], cwd=work_dir, env=env, stdout=lf,
                    stderr=subprocess.STDOUT, text=True, start_new_session=True,
                    stdin=subprocess.DEVNULL)
            except OSError as exc:
                # A command that cannot be started at all (an interpreter path
                # that does not exist). The participants already started wait in
                # initialize() for it forever, so kill them before reporting.
                # Measured before this: the call raised after a 10 s wait and
                # left the partner running.
                lf.close()
                _kill_all()
                raise OSError(exc.errno, f"participant {p['name']!r} could not be "
                              f"started ({exc.strerror}: {p['command'][0]!r}); the "
                              f"participants already started were killed") from exc
            procs[p["name"]] = (pr, lf)
        # ONE shared deadline, and stop the moment a participant dies badly.
        # Waiting per participant with the FULL timeout meant a partner that
        # crashed in milliseconds still cost the whole timeout (measured with
        # that loop: 15.0 s at timeout=15 when one participant exited 1 at once;
        # the first wait that timed out killed the rest, so one timeout, not N)
        # while the survivor sat blocked on the m2n handshake — and
        # the hung participant's log, the only one that could explain the hang,
        # was never captured, because it was read only after a successful wait.
        # A command that cannot be started at all is handled in the launch loop.
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            for name, (pr, _lf) in procs.items():
                if name not in rcs and pr.poll() is not None:
                    rcs[name] = pr.returncode
            if len(rcs) == len(procs):
                break
            if any(rc != 0 for rc in rcs.values()):
                dead = {n: rc for n, rc in rcs.items() if rc != 0}
                error = (f"participant(s) {dead} exited non-zero; the rest were "
                         "killed rather than left blocked on the coupling socket")
                _kill_all()
                break
            _time.sleep(0.05)
        else:
            error = f"coupling timed out after {timeout}s"
            _kill_all()
    finally:
        for name, (pr, lf) in procs.items():
            if name not in rcs:
                try:
                    rcs[name] = pr.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    rcs[name] = None
            try:
                lf.close()
            except OSError:
                pass
            f = work_dir / f"{name}.out"
            logs[name] = f.read_text(errors="replace") if f.exists() else ""

    ev = precice_run_evidence(work_dir, [p["name"] for p in participants], logs,
                              implicit="implicit" in scheme)
    out = {"returncodes": rcs, "config": str(cfg_path),
           "logs": {n: t[-600:] for n, t in logs.items()},
           "exit_codes_ok": bool(rcs) and all(rc == 0 for rc in rcs.values()),
           "exchanged": ev["exchanged"],
           "coupling_converged": ev["converged"],
           "evidence": ev["detail"],
           "config_notes": notes}
    if error:
        out["error"] = error
    # `converged` used to mean "every process exited 0", which two no-op scripts
    # that never touch preCICE also satisfy. It now means what it says, and is
    # ABSENT when preCICE measured nothing (an explicit scheme has no notion of
    # convergence at all) rather than defaulting to True.
    if ev["converged"] is not None:
        out["converged"] = bool(ev["converged"]) and out["exit_codes_ok"]
    return out


def precice_run_evidence(work_dir, names: list, logs: dict,
                         implicit: bool = False) -> dict:
    """Read preCICE's OWN record of what happened, not the process exit codes.

    preCICE writes `precice-<Participant>-iterations.log` next to the config, with
    a `Convergence` column (1/0) per time window for implicit schemes. An implicit
    scheme that exhausts `max-iterations` without meeting its measure logs
    `Convergence 0`, prints NO warning, and EXITS 0 — so exit codes cannot tell a
    converged coupling from one that gave up, and two scripts that never call
    preCICE at all satisfy them equally.

    Returns {"exchanged": bool, "converged": bool|None, "detail": [...], "windows": int}.
    converged is None when preCICE measured no convergence (explicit schemes), so
    "not measured" stays distinguishable from "measured and fine".
    """
    from pathlib import Path as _P
    wd = _P(work_dir)
    detail: list[str] = []
    ran: dict[str, bool] = {}
    conv: list[bool] = []
    windows = 0
    for name in names:
        text = logs.get(name, "") or ""
        it_log = wd / f"precice-{name}-iterations.log"
        started = ("This is preCICE version" in text) or it_log.exists()
        ran[name] = started
        if not started:
            detail.append(f"{name}: no preCICE banner and no iterations log — this "
                          "participant never initialised preCICE, so nothing was "
                          "exchanged with it")
            continue
        if "ERROR:" in text:
            line = next((ln.strip() for ln in text.splitlines() if "ERROR:" in ln), "")
            detail.append(f"{name}: preCICE reported an error — {line[:200]}")
        # EXPLICIT schemes write no iterations log at all, so its absence is not
        # evidence of a dead coupling; preCICE's own per-window log line is.
        completed = text.count("Time window completed")
        if not it_log.exists():
            if completed:
                windows = max(windows, completed)
                detail.append(f"{name}: {completed} time window(s) completed; this "
                              "scheme keeps no iteration record, so convergence "
                              "was NOT established")
            else:
                ran[name] = False
                detail.append(f"{name}: preCICE started but completed no coupling "
                              "time window")
            continue
        rows = [ln.split() for ln in it_log.read_text(errors="replace").splitlines()
                if ln.strip()]
        if not rows:
            windows = max(windows, completed)
            if not completed:
                ran[name] = False
            detail.append(f"{name}: iterations log is empty — {completed} time "
                          "window(s) completed per the run log")
            continue
        header = rows[0]
        # Index by HEADER NAME: the column count differs per participant (the
        # second one carries extra quasi-Newton columns), so a positional read of
        # the last column silently reports a different quantity entirely.
        if "Convergence" not in header:
            detail.append(f"{name}: {len(rows) - 1} time window(s) completed; this "
                          "scheme measures no convergence")
            windows = max(windows, len(rows) - 1)
            continue
        ci = header.index("Convergence")
        flags = [r[ci] == "1" for r in rows[1:] if len(r) > ci]
        windows = max(windows, len(flags))
        if not flags:
            detail.append(f"{name}: no completed time window in the iterations log")
            continue
        conv.append(all(flags))
        if not all(flags):
            bad = [i + 1 for i, f in enumerate(flags) if not f]
            detail.append(
                f"{name}: preCICE recorded NON-CONVERGENCE in time window(s) {bad} "
                "— the implicit scheme hit max-iterations without meeting its "
                "convergence measure. preCICE logs this and exits 0.")
    exchanged = bool(names) and all(ran.get(n) for n in names) and windows > 0
    if not exchanged and all(ran.get(n) for n in names):
        detail.append("no completed coupling time window was recorded — the "
                      "participants started preCICE but exchanged nothing")
    converged = (all(conv) if conv else None)
    if implicit and converged is None:
        detail.append("this implicit scheme recorded no convergence column, so "
                      "whether it converged was NOT established")
    return {"exchanged": exchanged, "converged": converged, "detail": detail,
            "windows": windows}


def verify_precice_coupling(work_dir: Path = None, timeout: int = 60) -> tuple[bool, str]:
    """Run a real 2-participant preCICE coupling and confirm data is exchanged.

    Launches participants A and B (A writes 42.0, B reads it) over 3 time windows via
    the preCICE sockets m2n. Returns (ok, message). This is the complete end-to-end
    verification that preCICE works in this environment, not just that it imports.
    """
    import subprocess, sys, tempfile, time
    _ensure_lib_on_path()
    wd = Path(work_dir or tempfile.mkdtemp(prefix="precice_verify_"))
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "precice-config.xml").write_text(_VERIFY_CONFIG)
    (wd / "participant.py").write_text(_VERIFY_PARTICIPANT)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = PRECICE_LIB_DIR + ":" + env.get("LD_LIBRARY_PATH", "")
    py = sys.executable
    try:
        pa = subprocess.Popen([py, "participant.py", "A", "B"], cwd=wd, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, stdin=subprocess.DEVNULL)
        pb = subprocess.Popen([py, "participant.py", "B", "A"], cwd=wd, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, stdin=subprocess.DEVNULL)
        out_b, _ = pb.communicate(timeout=timeout)
        pa.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pa.kill(); pb.kill()
        return False, "preCICE coupling timed out (deadlock?)"
    except Exception as e:
        return False, f"preCICE coupling launch failed: {e}"
    got = out_b.count("B received 42.0")
    if pa.returncode == 0 and pb.returncode == 0 and got >= 1:
        return True, f"preCICE coupling verified: B received the coupled value over {got} window(s)"
    return False, f"preCICE coupling failed (rc_A={pa.returncode}, rc_B={pb.returncode}); output:\n{out_b[-500:]}"
