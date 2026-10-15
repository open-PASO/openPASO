"""The traction a coupling EXCHANGES is not the traction a result file REPORTS.

Every served Dirichlet-Neumann participant exports `normal_fluxes` as minus the
conormal derivative on its OWN outward normal n_own: q = -(k grad u) . n_own for
heat, which IS the outward heat flux, and q = -(sigma . n_own) for elasticity,
which is MINUS the outward traction sigma . n_own. The exchange needs exactly
that -- the two sides' exports cancel, and the Neumann side applies its
partner's numbers unchanged -- so it must stay as it is.

FOUND ON AN ELASTIC COUPLING: after a converged ladder the reply told the agent
to write the task's interface files from these columns and said nothing about
their sign, so a file that asked for the outward traction sigma . n held its
negative. The fix is in what openPASO SAYS, not in what the participants
EXCHANGE, and these tests hold the two apart:

  * the served elastic and scalar contracts, RUN on fields whose stress and flux
    are known in closed form, on both orientations of the interface: the export
    is -(sigma . n_own) and -(k grad u) . n_own, the Neumann side reproduces the
    field from its partner's numbers applied unchanged, and the outward traction
    obtained by the served rule, -q, is sigma . n_own and not q;
  * a real converged coupling of two served elastic participants: the replies
    that send the agent from the exported columns to the task's files carry the
    rule, and the interface tables name one column per component.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
PARTICIPANTS = REPO / "data" / "coupling_participants"

pytest.importorskip("skfem")

from tools.coupling_knowledge import FLUX_COLUMNS_SIGN, FLUX_COLUMNS_SIGN_SHORT  # noqa: E402

# A plane-strain displacement LINEAR in x and y: P1 elements hold it exactly, the stress is
# one constant tensor with every component nonzero, and the body force is zero. Written in
# the contract's own config keys (udx, udy: u = c0 + c1 x + c2 y + c3 y^2).
E_MOD, NU = 1000.0, 0.3
UDX = (0.001, 0.002, 0.0005, 0.0)
UDY = (-0.0005, 0.001, -0.0015, 0.0)
LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E_MOD / (2.0 * (1.0 + NU))
_EXX, _EYY, _EXY = UDX[1], UDY[2], 0.5 * (UDX[2] + UDY[1])
SIGMA = np.array([[2.0 * MU * _EXX + LAM * (_EXX + _EYY), 2.0 * MU * _EXY],
                  [2.0 * MU * _EXY, 2.0 * MU * _EYY + LAM * (_EXX + _EYY)]])


def _u(x, y):
    return (UDX[0] + UDX[1] * x + UDX[2] * y, UDY[0] + UDY[1] * x + UDY[2] * y)


# The two subdomains of each arrangement: (box x0, x1, y0, y1; where the interface is on it;
# its outward normal there). The first side is the Dirichlet side, the second the Neumann side.
ARRANGEMENTS = {
    "x": (((0.0, 0.5, 0.0, 0.4), "right", (1.0, 0.0)),
          ((0.5, 1.0, 0.0, 0.4), "left", (-1.0, 0.0))),
    "y": (((0.0, 0.6, 0.0, 0.3), "top", (0.0, 1.0)),
          ((0.0, 0.6, 0.3, 0.7), "bottom", (0.0, -1.0))),
}
IFACE_AT = {"x": 0.5, "y": 0.3}          # the interface is the line x = 0.5, or y = 0.3


def _run(script: Path, work: Path, config: dict, imports: dict) -> dict:
    """Run a served participant once in `work`, the way the driver does, and return its exports."""
    work.mkdir(parents=True)
    shutil.copy(script, work / "participant.py")
    (work / "config.json").write_text(json.dumps(config))
    (work / "imports.json").write_text(json.dumps(imports))
    r = subprocess.run([sys.executable, "participant.py"], cwd=work, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:])
    return json.loads((work / "exports.json").read_text())


def _dump(work: Path, level: int = 1):
    """(header, rows) of the side's own interface_level<k>.csv."""
    lines = (work / f"interface_level{level}.csv").read_text().splitlines()
    return lines[0].split(","), np.array([[float(c) for c in ln.split(",")] for ln in lines[1:]])


def _elastic_config(box, iface, side, partner):
    return {"side": side, "partner": partner, "x0": box[0], "x1": box[1], "y0": box[2],
            "y1": box[3], "iface": iface, "E": E_MOD, "nu": NU, "udx": list(UDX),
            "udy": list(UDY), "level": 1, "nx": 8, "ny": 6}


def test_the_served_statement_says_what_the_columns_hold():
    """The rule itself: the exchange convention, and how a file gets the outward quantity."""
    for needle in ("-(sigma . n_own)", "NEGATIVE of the outward traction sigma . n_own",
                   "gets -q", "-(k grad u) . n_own", "IS the outward normal flux",
                   "ITS OWN outward normal", "leave the exchange as it is",
                   "ONE normal n for both sides", "tx, ty already hold sigma . n_own", "FSI"):
        assert needle in FLUX_COLUMNS_SIGN, needle
    for needle in ("-(sigma . n_own)", "MINUS the outward traction sigma . n_own", "gets -q",
                   "-(k grad u) . n_own", "the outward normal flux itself", "FSI"):
        assert needle in FLUX_COLUMNS_SIGN_SHORT, needle


@pytest.mark.parametrize("axis", sorted(ARRANGEMENTS))
def test_elastic_export_is_minus_the_outward_traction(tmp_path, axis):
    (box_d, iface_d, n_d), (box_n, iface_n, n_n) = ARRANGEMENTS[axis]
    script = PARTICIPANTS / "participant_skfem_elastic.py"
    # points along the interface, with the field's own displacement on them
    at = IFACE_AT[axis]
    s = np.linspace(box_d[2], box_d[3], 13) if axis == "x" else np.linspace(box_d[0], box_d[1], 13)
    pts = [[at, float(t)] if axis == "x" else [float(t), at] for t in s]
    ex_d = _run(script, tmp_path / "D", _elastic_config(box_d, iface_d, "dirichlet", "N"),
                {"N": {"coordinates": pts, "values": [list(_u(*p)) for p in pts]}})
    # the coupling's fixed point: the Neumann side applies its partner's export UNCHANGED
    ex_n = _run(script, tmp_path / "N", _elastic_config(box_n, iface_n, "neumann", "D"),
                {"D": {"coordinates": ex_d["coordinates"], "normal_fluxes": ex_d["normal_fluxes"]}})
    scale = float(np.abs(SIGMA).max())
    for ex, n_own, tag in ((ex_d, n_d, "D"), (ex_n, n_n, "N")):
        q = np.asarray(ex["normal_fluxes"], float)
        outward = SIGMA @ np.asarray(n_own)                   # sigma . n_own, both components nonzero
        assert np.abs(outward).min() > 0.1 * scale
        # 1. WHAT THE COUPLING EXCHANGES: -(sigma . n_own), at every interface point
        assert np.abs(q - (-outward)).max() < 1e-9 * scale, (tag, q[:3], outward)
        # 2. WHAT THE CONTRACT DUMPS: the same numbers, under the q column names
        head, rows = _dump(tmp_path / tag)
        assert head[-2:] == ["qx", "qy"], head
        assert np.abs(rows[:, -2:] - q).max() < 1e-9 * scale
        # 3. WHAT A FILE THAT ASKS FOR THE TRACTION RECEIVES, by the served rule: -q
        t = -q
        assert np.abs(t - outward).max() < 1e-9 * scale
        assert np.abs(t - q).min() > 1.0 * np.abs(outward).min(), "the two readings must differ"
    q_d = np.asarray(ex_d["normal_fluxes"], float)
    q_n = np.asarray(ex_n["normal_fluxes"], float)
    # the exports cancel -- the conservation statement the exchange convention exists for
    assert np.abs(q_d + q_n).max() < 1e-9 * scale
    # the partner's numbers, applied unchanged, give the Neumann side the right field
    u_n = np.asarray(ex_n["values"], float)
    u_ref = np.array([_u(*c) for c in ex_n["coordinates"]])
    assert np.abs(u_n - u_ref).max() < 1e-12
    # ONE normal n for both sides (here the Dirichlet side's): sigma . n is -q there and +q on
    # the side whose outward normal is -n, and the two agree, as a traction across a surface must
    fixed = SIGMA @ np.asarray(n_d)
    assert np.abs(-q_d - fixed).max() < 1e-9 * scale
    assert np.abs(+q_n - fixed).max() < 1e-9 * scale


def test_scalar_export_is_the_outward_flux(tmp_path):
    """The heat contract's qn IS the outward normal flux: a file that asks for it takes it as it is.

    T = 300 + 50 x on both sides of the line x = 0.6, held at its own value on each side's outer
    face; the contract's interface sits at x = 0.6 and its y-faces are free (zero flux, which this
    T has). The outward heat flux there is -k dT/dx * n_x."""
    k = 0.8
    script = PARTICIPANTS / "participant_skfem.py"
    ys = np.linspace(0.0, 0.4, 9)
    pts = [[0.6, float(y)] for y in ys]
    for tag, box, n_x in (("left", (0.0, 0.6, 0.0, 0.4), 1.0), ("right", (0.6, 1.2, 0.0, 0.4), -1.0)):
        outer = 300.0 + 50.0 * (box[0] if n_x > 0 else box[1])
        ex = _run(script, tmp_path / tag,
                  {"x0": box[0], "x1": box[1], "y0": box[2], "y1": box[3], "k": k, "outer": outer,
                   "level": 1, "nx": 6, "ny": 4},
                  {"right": {"coordinates": pts, "values": [330.0] * len(pts)}})
        qn = np.asarray(ex["normal_fluxes"], float)
        outward_flux = -k * 50.0 * n_x                        # (-k grad T) . n_own
        assert np.abs(qn - outward_flux).max() < 1e-9 * abs(outward_flux), (tag, qn[:3])
        head, rows = _dump(tmp_path / tag)
        assert head[-1] == "qn", head
        assert np.abs(rows[:, -1] - qn).max() < 1e-9 * abs(outward_flux)


# ── A REAL CONVERGED COUPLING, AND WHAT ITS REPLIES TELL THE AGENT ──────────────────────────


class _Tools:
    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


def _call(fn, **kw):
    r = fn(**kw)
    if inspect.iscoroutine(r):
        r = asyncio.run(r)
    return json.loads(str(r))


@pytest.fixture(scope="module")
def tools():
    from core.registry import load_all_backends
    from tools.consolidated import register_consolidated_tools
    load_all_backends()
    reg = _Tools()
    register_consolidated_tools(reg)
    return reg.tools


def _elastic_pair(work: Path):
    """Two served elastic participants side by side on x = 0.5: A Dirichlet, B Neumann."""
    script = PARTICIPANTS / "participant_skfem_elastic.py"
    (box_a, iface_a, _), (box_b, iface_b, _) = ARRANGEMENTS["x"]
    for name, box, iface, side, partner in (("A", box_a, iface_a, "dirichlet", "B"),
                                            ("B", box_b, iface_b, "neumann", "A")):
        d = work / f"side_{name}"
        d.mkdir(parents=True)
        shutil.copy(script, d / "participant.py")
        (d / "config.json").write_text(json.dumps(_elastic_config(box, iface, side, partner)))
    return [{"name": n, "command": [sys.executable, "participant.py"],
             "work_dir": str(work / f"side_{n}"), "imports_from": [o]}
            for n, o in (("A", "B"), ("B", "A"))]


@pytest.fixture(scope="module")
def coupled(tools, tmp_path_factory):
    """One couple_levels call and one couple call, each a real coupling to convergence.

    Their working directories hold a solved coupling; they are removed afterwards."""
    out = {}
    roots = []
    for kind in ("ladder", "couple"):
        work = tmp_path_factory.mktemp(f"traction_sign_{kind}")
        roots.append(work)
        with pytest.MonkeyPatch.context() as mp:
            mp.chdir(work)
            mp.setenv("OPENPASO_CELL_WORKDIR", str(work))
            mp.setenv("OPENPASO_JOURNAL_LIVE_DIR", str(work / "journal"))
            parts = json.dumps(_elastic_pair(work))
            if kind == "ladder":
                reply = _call(tools["couple_levels"], participants=parts, critic_approved=True,
                              levels=json.dumps([{"level": 1, "A": {"nx": 8, "ny": 6},
                                                  "B": {"nx": 8, "ny": 6}}]),
                              max_iter=80, tol=1e-8, probe=False,
                              history_pattern="residual_level{k}.csv")
            else:
                reply = _call(tools["couple"], participants=parts, critic_approved=True,
                              max_iter=80, tol=1e-8, probe=False, iface_level=1,
                              history_path=str(work / "residual_level1.csv"))
            from tools.result_audit import coupled_ladder
            out[kind] = (reply, coupled_ladder(work) or {})
    yield out
    for work in roots:
        shutil.rmtree(work, ignore_errors=True)


def _tables(blocks: dict) -> dict:
    """{side: (header, rows)} of the interface tables a reply carries inline."""
    got = {}
    for name, block in blocks.items():
        if isinstance(block, dict) and isinstance(block.get("csv"), str):
            lines = block["csv"].strip().splitlines()
            got[name] = ([c.strip() for c in lines[0].split(",")],
                         np.array([[float(c) for c in ln.split(",")] for ln in lines[1:]]))
    return got


def _check_tables(blocks: dict):
    """Both sides' tables: one column name per component, and q = -(sigma . n_own) in them."""
    tables = _tables(blocks)
    assert set(tables) == {"A", "B"}, blocks
    scale = float(np.abs(SIGMA).max())
    for name, n_own in (("A", (1.0, 0.0)), ("B", (-1.0, 0.0))):
        head, rows = tables[name]
        assert head == ["x", "y", "u1", "u2", "q1", "q2"], head
        assert rows.shape[1] == len(head)
        outward = SIGMA @ np.asarray(n_own)
        # the table holds the EXCHANGED quantity; the outward traction is its negative
        assert np.abs(rows[:, 4:6] + outward).max() < 1e-5 * scale, (name, rows[:2, 4:6], outward)
    assert FLUX_COLUMNS_SIGN_SHORT in blocks["_how_to_use"]


def test_a_converged_ladder_states_the_sign_where_it_sends_the_agent_to_the_files(coupled):
    reply, step = coupled["ladder"]
    assert reply["all_levels_converged"], reply
    # the after-convergence recipe that sends the agent from the dumps to the task's files
    assert "EVERY REQUESTED LEVEL CONVERGED" in reply["next_step"]
    assert FLUX_COLUMNS_SIGN in reply["next_step"]
    _check_tables(reply["levels"][0]["interface_csv"])
    # the audit's next step for the same directory: the worker brief that writes those files
    assert step.get("step") == 4, step
    assert FLUX_COLUMNS_SIGN in step["brief"]


def test_a_converged_couple_reply_states_the_sign(coupled):
    reply, _ = coupled["couple"]
    assert reply["converged"], reply
    assert FLUX_COLUMNS_SIGN in reply["reporting"]
    _check_tables(reply["interface_csv"])
    # the next step this reply hands over is the brief that writes the task's files
    assert FLUX_COLUMNS_SIGN in reply["next_step"]
