"""Fluid turbulence (LES/DNS) generator for 4C.

Covers large-eddy simulation and direct numerical simulation capabilities.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class FluidTurbulenceGenerator(BaseGenerator):
    """Generator for turbulent flow (LES/DNS) in 4C."""

    module_key = "fluid_turbulence"
    display_name = "Fluid Turbulence (LES/DNS)"
    problem_type = "Fluid"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Large-Eddy Simulation (LES) and Direct Numerical Simulation (DNS) "
                "for turbulent incompressible flow.  Uses the fluid module with "
                "additional subgrid-scale modeling."
            ),
            "sgs_models": {
                "Smagorinsky": "Classic constant-coefficient SGS model",
                "Smagorinsky_with_van_Driest_damping": (
                    "Smagorinsky with van Driest wall damping (channel of "
                    "height 2 in y only)"
                ),
                "Dynamic_Smagorinsky": "Germano dynamic procedure for C_s",
                "Vreman": "Vreman SGS model",
                "Dynamic_Vreman": "Vreman model with one coefficient computed dynamically from volume-averaged filtered quantities (clipped at zero)",
                "Multifractal_Subgrid_Scales": "Multifractal SGS model",
            },
            "stabilization": [
                "Residual-based VMS (variational multiscale) — built into fluid elements",
                "SUPG/PSPG for coarse LES",
            ],
            "applications": ["channel flow DNS/LES", "backward-facing step",
                             "cylinder wake", "jet flow", "mixing layers"],
            "pitfalls": [
                (
                    '[Numerical] DNS requires resolution down to the Kolmogorov '
                    'scale eta = (nu^3/epsilon)^(1/4), and the cost grows steeply '
                    'with Reynolds number. 4C will not help you notice an '
                    'under-resolved run: TURBULENCE_APPROACH has only two values, '
                    'CLASSICAL_LES and DNS_OR_RESVMM_LES, and the default lumps DNS '
                    "together with residual-based VMM LES, so writing 'DNS' is "
                    'rejected with a possible-values list and no resolution check '
                    'exists anywhere. Signal: to check y+ at the first cell you '
                    'must set DUMPING_PERIOD > 0 in FLUID DYNAMIC/TURBULENCE MODEL, '
                    'which writes <output>.flow_statistics with a y+ column and a '
                    "'(u_tau)^2 = tau_W/rho' line; with the usual DUMPING_PERIOD: 0 "
                    'that file has a header and no rows. (Audit 2026-06-02; '
                    'corrected by execution 2026-08-06.)'
                ),
                (
                    '[Numerical] An LES mesh should resolve most of the turbulent '
                    'kinetic energy, but 4C reports no resolved/total TKE ratio and '
                    'no TKE at all. Neither the log nor the statistics file '
                    "contains 'kinetic energy', 'TKE' or 'resolved', with the "
                    'subgrid model on or off. Signal: what you get, and only when '
                    'DUMPING_PERIOD > 0, is <output>.flow_statistics headed '
                    "'Statistics for turbulent incompressible channel flow (first- "
                    "and second-order moments)' with plane-averaged mean u^2, mean "
                    'v^2, mean w^2 and cross moments. A resolved TKE has to be '
                    'assembled from those columns by hand; there is no total to '
                    'divide by. (Audit 2026-06-02; corrected by execution '
                    '2026-08-06.)'
                ),
                (
                    '[Numerical] Keep the time step small enough to resolve the '
                    "eddies you care about. The 'CFL < 1 for explicit' half of the "
                    'usual rule does not apply to 4C: FLUID DYNAMIC TIMEINTEGR '
                    'offers Af_Gen_Alpha, BDF2, Np_Gen_Alpha, One_Step_Theta and '
                    'Stationary, all implicit, and asking for an explicit scheme is '
                    'rejected with that list. Signal: an oversized step is '
                    'therefore accepted and silent. The run completes, produces no '
                    'NaN, and prints no CFL or Courant warning of any kind, while '
                    'the results drift. Only a reference solution or a step-size '
                    'study will catch it. (Audit 2026-06-02; corrected by execution '
                    '2026-08-06.)'
                ),
                (
                    '[Input] Periodic boundary conditions are needed for the '
                    'homogeneous directions of a channel or box flow. The sections '
                    'are DESIGN LINE PERIODIC BOUNDARY CONDITIONS and DESIGN SURF '
                    'PERIODIC BOUNDARY CONDITIONS, whose entries pair a Master and '
                    'a Slave through a shared ID with PLANE, LAYER, ANGLE and '
                    "ABSTREETOL; 'DESIGN PERIODIC CONDITIONS' is not a 4C section "
                    'and is rejected as not a valid section name. Signal: omitting '
                    'the real block does not give a merely wrong mean profile. A '
                    'streamwise-periodic channel has no Dirichlet data in that '
                    "direction, so the run aborts with 'Nullspace check for sysmat_ "
                    "failed' from 4C_fluid_implicit_integration.cpp. (Audit "
                    '2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Numerical] Turning on a subgrid-scale model takes THREE '
                    'settings, and any one of them missing leaves the model '
                    'silently inert: TURBULENCE_APPROACH: CLASSICAL_LES and '
                    'PHYSICAL_MODEL: <model> in FLUID DYNAMIC/TURBULENCE MODEL, '
                    'plus a non-zero C_SMAGORINSKY in FLUID DYNAMIC/SUBGRID '
                    'VISCOSITY, which defaults to zero. There is no '
                    'TURBULENCE_MODEL key; that spelling is rejected as unused '
                    'input. Signal: PHYSICAL_MODEL together with a real constant '
                    'but the DEFAULT approach reproduces the no-model answer '
                    'exactly, with no warning that the model was skipped. '
                    'Statistics windows are SAMPLING_START, SAMPLING_STOP and '
                    "DUMPING_PERIOD; 'running_mean' is not 4C vocabulary. (Audit "
                    '2026-06-02; corrected by execution 2026-08-06.)'
                ),
                (
                    '[Input] A laminar inlet on a turbulent LES wastes a long '
                    "development length, so use 4C's turbulent-inflow generator, "
                    'but note it is a precursor DOMAIN, not a recycling or '
                    'synthetic-eddy operator, and neither Lund-Wu-Squires nor '
                    'Jarrin appears anywhere in 4C. Signal: FLUID DYNAMIC/TURBULENT '
                    'INFLOW with TURBULENTINFLOW: true also needs CANONICAL_INFLOW, '
                    'INFLOW_HOMDIR, NUMINFLOWSTEP and a geometric separation '
                    'declared by FLUID TURBULENT INFLOW VOLUME with DESIGN SURF '
                    'TURBULENT INFLOW TRANSFER. Setting only the flag aborts with '
                    "4C's own typo 'homogeneuous plane for channel flow was "
                    "specified incorrectly.'; fixing that reaches 'Nodes with "
                    "separation condition expected!' from "
                    '4C_fluid_discret_extractor.cpp. (Audit 2026-06-02; corrected '
                    'by execution 2026-08-06.)'
                ),
                (
                    # Split from the entry above (index 3) rather than folded
                    # into it. That entry is about the SECTION: its name, its
                    # Master/Slave pairing, and the nullspace abort you get by
                    # deleting it. This one is about the RANK COUNT, which the
                    # section entry says nothing about, and the two fixtures
                    # that probe them were both keyed to index 3 — crediting
                    # one claim twice and leaving this failure mode with no
                    # entry of its own to defend.
                    '[Input] A deck with periodic boundary conditions is a '
                    'RANK-COUNT constraint on this build, not only a modelling '
                    'choice: the same file, byte for byte, ABORTS on one MPI '
                    'rank and completes on two and on four. The throw comes out '
                    'of Epetra_CrsGraph::MakeIndicesLocal, reached through '
                    'Core::LinAlg::SparseMatrix::complete inside '
                    'Conditions::PeriodicBoundaryConditions::balance_load, and '
                    'it is a bare int, so 4C emits no diagnostic of its own at '
                    'all. Signal: shell status 134 (SIGABRT) and a C++ runtime '
                    'terminate line naming a thrown int, which is in no 4C '
                    'source file; there is no PROC 0 ERROR banner, no source '
                    'line, and the string DESIGN SURF PERIODIC appears nowhere '
                    'in the log, so the reader is sent to inspect the periodic '
                    'block — which is correct — instead of the mpirun -np. The '
                    'only 4C-side anchor is the frame name '
                    "'Conditions::PeriodicBoundaryConditions::balance_load', "
                    'which is in 4C_fem_condition_periodic.cpp; on 2 and on 4 '
                    'ranks every rank reaches the exit banner and prints '
                    "'finished normally'. This is why the LES channel deck this "
                    'project ships records np=2. (Verified by execution '
                    '2026-08-09 against 4C 2026.2.0-dev, git 89519cf, on 1, 2 '
                    'and 4 ranks.)'
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "les_channel_3d", "description": "LES of turbulent channel flow"}]

    def get_template(self, variant: str = "les_channel_3d") -> str:
        # Not self-contained-runnable: a meaningful LES needs a
        # wall-resolved, periodic, graded HEX8 channel mesh — a coarse
        # inline grid "runs" but produces physically meaningless
        # statistics. Return a valid-YAML reference stub rather than a
        # comment-only one-liner.
        return (
            "# =====================================================\n"
            "# 4C fluid turbulence / LES (variant: les_channel_3d)\n"
            "# =====================================================\n"
            "# Not a self-contained runnable input. Requires:\n"
            "#   * a wall-resolved 3-D HEX8 channel mesh graded to\n"
            "#     the wall (y+ ~ 1) with periodic stream/spanwise\n"
            "#     boundary surfaces\n"
            "#   * FLUID DYNAMIC + a TURBULENCE MODEL section\n"
            "#     (Smagorinsky / Dynamic_Smagorinsky / Vreman /\n"
            "#     Multifractal_Subgrid_Scales) + periodic BCs +\n"
            "#     turbulence statistics sampling\n"
            "#   * MAT_fluid at the target Reynolds number\n"
            "# Pitfalls (see knowledge() for the full set):\n"
            "#   * LES on an under-resolved mesh is meaningless — it\n"
            "#     'runs' but the statistics are wrong\n"
            "#   * needs a long sampling time to converge mean /\n"
            "#     Reynolds-stress profiles\n"
            "# =====================================================\n"
            "TITLE:\n"
            "  - \"4C fluid turbulence (LES) reference stub\"\n"
            "PROBLEM TYPE:\n"
            "  PROBLEMTYPE: \"Fluid\"\n"
        )

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
