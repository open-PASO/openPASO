"""Kratos FSIApplication: partitioned coupling with a convergence accelerator.

FSIApplication's substance is its CONVERGENCE ACCELERATORS -- Aitken, MVQN,
IBQN-MVQN, constant relaxation -- for the fixed-point iteration of a
partitioned coupling. This template uses one for exactly that, driving a
Dirichlet-Neumann exchange between two real Kratos ConvectionDiffusion
solves.

THE SERVED SCRIPT LEAVES EACH SIDE'S SETUP AND SOLVE TO THE AGENT (2026-09-25):
openPASO serves no working solve, and a complete Dirichlet-Neumann pair of
Kratos solves is exactly what a coupled problem asks its Kratos side to write.
Served: the accelerator, the exchange, the flux recovery and the checks.
tests/test_the_kratos_partitioned_contract_leaves_the_solve_out.py fills the
gap with a validation fill (never served) and runs it.

VERIFIED BY EXECUTION (with that fill) on Kratos 10.3.0 against two things
independent of the run. The converged interface temperature must equal the value derived from
the two conductivities, the two lengths and the two boundary temperatures --
it does, to six figures, at conductivity ratios of 1, 10 and 100. And the
accelerator must not move that fixed point, only reach it sooner -- at a
ratio of 10 Aitken takes 3 iterations where constant relaxation takes 25,
and both land on the same value.

THE FOUR DEFECTS IT TOOK TO GET THERE are the knowledge below. Three of them
produce a confident wrong answer rather than an error, and the fourth
produces no error message at all.
"""
from string import Template


_PARTITIONED = Template(r"""
'''$title

Partitioned Dirichlet-Neumann coupling of two Kratos ConvectionDiffusion
solves, with the fixed-point iteration driven by a KratosFSIApplication
convergence accelerator.

THIS IS A CONTRACT, NOT A PROGRAM: the accelerator loop, the exchange order,
the flux recovery and the checks are served; each side's model part,
material, boundary conditions and solve are yours, in the marked region.

It checks itself against two things independent of the run: the converged
interface temperature must equal the value derived from the two
conductivities, the two lengths and the two boundary temperatures, and the
accelerated iteration must not change that fixed point.
'''
import numpy as np, math
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication as CDA
import KratosMultiphysics.FSIApplication as FSI
import KratosMultiphysics.LinearSolversApplication  # noqa
from KratosMultiphysics import python_linear_solver_factory as plsf

import os
MUTATE = os.environ.get("SURVEY_MUTATE", "") == "1"   # planted failure, constant-relaxation branch only
NX, NY = $nx, $ny
L1, L2, H = $l1, $l2, $height
K1, K2 = $k1, $k2
T_LEFT, T_RIGHT = $t_left, $t_right

# ── SIDE SETUP AND SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
# Yours to write: the two sides as real Kratos ConvectionDiffusion solves. The loop below needs:
#   left, right       ModelParts on x in [0, L1] and [L1, L1 + L2], conductivity K1 and K2,
#                     NX x NY cells of height H, with TEMPERATURE and REACTION_FLUX as dof pair;
#   sl, sr            one object per side whose .Solve() solves it (the loop calls them);
#   iface_l, iface_r  the interface nodes (x == L1) of each side in the SAME order (by y), so
#                     index i is the same point on both sides;
#   and the boundary conditions: TEMPERATURE fixed to T_LEFT at x = 0 and to T_RIGHT at
#   x = L1 + L2, and fixed on the left side's interface nodes (the Dirichlet side: the loop
#   sets their values each iteration).
# Measured facts the fill has to get right -- the first gives no message at all, the other two
# a clean convergence to a wrong answer:
#   * LaplacianElement2D3N reads CONVECTION_DIFFUSION_SETTINGS (unknown, diffusion, volume and
#     surface source, density, specific heat); without it the process segfaults, exit 139.
#   * the strategy must compute reactions (CalculateReactionsFlag): otherwise REACTION_FLUX
#     stays zero, nothing crosses the interface and the coupling "converges" in one iteration.
#   * FACE_HEAT_FLUX on nodes is a density that only FACE CONDITIONS integrate: the Neumann
#     side needs ThermalFace2D2N conditions on its interface, or it sees no source at all.
raise SystemExit("fill the marked region: each side's model part, material, boundary "
                 "conditions and solver -- the comments above it say what the loop needs")
# ── SIDE SETUP AND SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

def coupled(acc, label):
    for n in iface_l: n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    acc.Initialize(); acc.InitializeSolutionStep()
    x = KM.Vector(len(iface_l))
    for i in range(len(iface_l)): x[i] = 0.0
    for it in range(1, 60):
        for i, n in enumerate(iface_l): n.SetSolutionStepValue(KM.TEMPERATURE, x[i])
        sl.Solve()
        # A NODAL REACTION IS AN INTEGRATED QUANTITY; FACE_HEAT_FLUX IS A
        # DENSITY THAT THE CONDITION INTEGRATES AGAIN. Passing the reaction
        # straight across double-integrates it, and the coupling converges
        # -- cleanly, with the accelerator working -- to the WRONG fixed
        # point. Divide by the node's tributary length: h for an interior
        # node, h/2 at each end.
        h = H / NY
        q = []
        for i, n in enumerate(iface_l):
            trib = h if 0 < i < len(iface_l) - 1 else 0.5 * h
            if MUTATE and label == "constant":
                trib = 1.0    # the planted wrong answer: the reaction passed across as if it were a density
            q.append(n.GetSolutionStepValue(KM.REACTION_FLUX) / trib)
        for i, n in enumerate(iface_r):
            n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, -q[i])
        sr.Solve()
        t_new = [n.GetSolutionStepValue(KM.TEMPERATURE) for n in iface_r]
        r = KM.Vector(len(iface_l))
        for i in range(len(iface_l)): r[i] = t_new[i] - x[i]
        res = max(abs(r[i]) for i in range(len(iface_l)))
        if res < 1e-9: break
        acc.InitializeNonLinearIteration(); acc.UpdateSolution(r, x)
        acc.FinalizeNonLinearIteration()
    return it, [x[i] for i in range(len(iface_l))]

it_a, t_a = coupled(FSI.AitkenConvergenceAccelerator(0.5), "aitken")
it_c, t_c = coupled(FSI.ConstantRelaxationConvergenceAccelerator(0.5), "constant")
t_exact = (T_LEFT*K1/L1 + T_RIGHT*K2/L2) / (K1/L1 + K2/L2)
print("FSI aitken   %2d iterations, interface T = %.6f" % (it_a, sum(t_a)/len(t_a)))
print("FSI constant %2d iterations, interface T = %.6f" % (it_c, sum(t_c)/len(t_c)))
print("FSI analytic interface T = %.6f  (derived from k, L and the two BCs)" % t_exact)
# One VERDICT line per accelerator, in the coverage harness's grammar
# (scripts/coverage_harness/definitions.py). The reference is the closed-form
# interface temperature of two conducting strips in series; P1 elements on
# these meshes are nodally exact for that piecewise-linear profile, so the only
# distance left is the coupling's stopping residual (1e-9 on the interface
# temperature), and the tolerance is a thousand times that.
for _label, _t in (("aitken", sum(t_a)/len(t_a)), ("constant", sum(t_c)/len(t_c))):
    _ok = abs(_t - t_exact) <= 1e-6
    print("VERDICT kratos fsi_partitioned closed_form ref=%.6e got=%.6e tol=%.6e "
          "tol_from=coupling_stop_residual_1e-9_x1000;P1_nodally_exact_for_a_piecewise_linear_profile %s"
          % (t_exact, _t, 1e-6, "PASS" if _ok else "FAIL"))

ok = abs(sum(t_a)/len(t_a) - t_exact) < 1e-6 and abs(sum(t_c)/len(t_c) - t_exact) < 1e-6
print("FSI interface matches the derived value:", ok)
print("FSI accelerator saved %d iterations" % (it_c - it_a))
""")


def _fsi_partitioned_2d(params: dict) -> str:
    """Two conducting strips coupled Dirichlet-Neumann across an interface."""
    return _PARTITIONED.substitute(
        title=params.get("title", "Partitioned coupling with an FSI accelerator"),
        nx=params.get("nx", 8), ny=params.get("ny", 4),
        l1=params.get("length_left", 1.0), l2=params.get("length_right", 1.0),
        height=params.get("height", 1.0),
        k1=params.get("conductivity_left", 1.0),
        k2=params.get("conductivity_right", 10.0),
        t_left=params.get("temperature_left", 1.0),
        t_right=params.get("temperature_right", 0.0),
    )


GENERATORS = {
    "fsi_partitioned_2d": _fsi_partitioned_2d,
}

KNOWLEDGE = {
    "fsi_partitioned": {
        "description": (
            "Partitioned Dirichlet-Neumann coupling driven by a "
            "KratosFSIApplication convergence accelerator (Aitken, MVQN, "
            "IBQN-MVQN, constant relaxation). The served script is a "
            "contract: each side's setup and solve are yours."
        ),
        "application": "FSIApplication",
        "pitfalls": [
            "[Numerical] A NODAL REACTION IS AN INTEGRATED QUANTITY AND A "
            "FACE FLUX IS A DENSITY. Passing REACTION_FLUX straight across "
            "an interface double-integrates it: the coupling still "
            "converges, the accelerator still works, and it lands on the "
            "WRONG fixed point. Measured: an interface that should sit at "
            "0.5 converged cleanly to 0.177 instead. Divide by the node's "
            "tributary length -- h for an interior node, h/2 at each end -- "
            "and the same run lands on 0.500000. Nothing about the failure "
            "looks like a failure. Signal: the interface TEMPERATURE "
            "converges cleanly under the Aitken loop to a value you can "
            "also derive analytically, and misses it by a factor of "
            "roughly the tributary length -- 0.177 where 0.5 was "
            "derivable. Derive the interface value and compare; a clean "
            "convergence to the wrong number has no other symptom.",

            "[API] REACTION_FLUX IS IDENTICALLY ZERO UNLESS THE STRATEGY WAS "
            "BUILT TO COMPUTE IT. The first flag of ResidualBasedLinearStrategy "
            "is CalculateReactionsFlag; with it False the solve is entirely "
            "correct and the reactions are all zero. A Dirichlet-Neumann "
            "exchange then transfers nothing, the residual is zero on the "
            "first pass, and the coupling reports CONVERGED IN ONE ITERATION. "
            "A coupling that converges immediately is exchanging nothing "
            "until proven otherwise. Signal: the iteration count is 1, and every "
            "transferred value reads back as exactly zero while each side 's"
            "own solve is correct.",

            "[API] A NODAL FLUX IS NOT A LOAD. Setting FACE_HEAT_FLUX on the "
            "receiving nodes contributes nothing unless FACE CONDITIONS "
            "integrate it over the boundary -- ThermalFace2D2N for a 2-D "
            "line. Without them the receiving side sees no source at all and "
            "sits at its initial value, while every variable you set reads "
            "back exactly as assigned. Signal: the receiving field is uniformly "
            "its initial value while the variable you set reads back "
            "correctly on every node -- assignment succeeded, assembly "
            "never happened.",

            "[Setup] A LaplacianElement WITHOUT CONVECTION_DIFFUSION_SETTINGS "
            "SEGFAULTS. The settings object on ProcessInfo names which "
            "variable is the unknown, which is the diffusivity and which are "
            "the sources; without it the element dereferences an unset "
            "pointer. There is no exception and no message -- the process "
            "dies with exit 139. Signal: the process terminates with SIGSEGV and no "
            "Python traceback, immediately at the first solve rather than "
            "during setup. Set the settings object before creating any "
            "element.",

            "[Numerical] THE ACCELERATOR EARNS ITS KEEP ONLY WHEN THE "
            "COUPLING IS STIFF. With matched conductivities both Aitken and "
            "constant relaxation converge in 2 iterations, so a comparison "
            "on an easy case proves nothing about the accelerator. Raise the "
            "conductivity ratio and the gap opens: measured 3 against 25 at "
            "a ratio of 10. Judge an accelerator on the case that needs one. "
            "Signal: two accelerators reporting the SAME iteration count "
            "on your test case -- that is the case being too easy, not the "
            "accelerators being equivalent.",
        ],
    },
}
