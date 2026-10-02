"""Runnable 4C deck templates — every one executed on the installed binary.

WHY THIS MODULE EXISTS
----------------------
`prepare_simulation(fourc, <physics>)` used to answer 27 of 4C's 49 catalog
physics with a comment block that said, in as many words, "Not a runnable
input — the user must supply the case-specific mesh + material parameters."
This module is what it answers with instead: 29 decks over 24 physics, every
one of them executed.
Every other backend in this project ships a fillable skeleton for every physics
it advertises (deal.II 27/27, scikit-fem 22/22, FEBio 17/17, SPARTA 10/10). 4C,
the backend the project is named around, was the exception — and it is the one
whose input format a model is least able to reconstruct from prose: 478
sections, 7383 paths, 2728 distinct keys.

WHAT IS IN HERE, AND WHAT "RUNNABLE" MEANS
------------------------------------------
Each entry is a complete deck, not a skeleton with `<...>` holes. Every one was
executed with `{FOURC_BINARY}` at the rank count recorded in
`np` and exited 0. A deck that 4C rejects is worse than no deck at all,
because it looks like help; so the rule for adding an entry here is that the
exact bytes shipped are the exact bytes that ran.

Each deck is also checked for *physics*, not only for exit status — a run that
completes without the coupling doing anything is a silent failure. The evidence
per deck lives in the `evidence` field and was measured from that deck's own
output (VTU fields, reaction forces, interface fluxes), never assumed.

HOW THE DECKS WERE BUILT
------------------------
From the upstream corpus, not from the grammar. `{FOURC_ROOT}/tests/
input_files` holds 1978 parseable decks that 4C's own CI runs; each template
starts from a named one of those (`upstream`) and is reduced — mesh shrunk,
regression `RESULT DESCRIPTION` deleted, Belos+XML solvers replaced by direct
UMFPACK — until it is self-contained and small enough to render whole. Two
(`particle_pd_*`) had no upstream deck to start from and were built from the
grammar dump plus the deck shape used by the author of 4C's PD module.

THE TWO DEPENDENCIES THAT COULD NOT BE REMOVED
-----------------------------------------------
`fs3i` and `multiscale` name a file outside the deck. Both were established by
execution, not assumed:

  * FS3I refuses any direct solver — `4C_fs3i_partitioned.cpp:604` throws
    "Iterative solver expected" unless COUPLED_LINEAR_SOLVER is Belos, `:610`
    demands AZPREC Teko, and `TekoPreconditioner::setup` then demands
    TEKO_XML_FILE. The deck therefore names 4C's own recommended block
    preconditioner by absolute path.
  * FE2 multiscale reads the RVE from a second, standalone `InputFile` through
    `Global::read_micro_fields`, one per macro multiscale material, so
    `MICROFILE` cannot be inlined. The deck names 4C's own RVE by path.

Both resolve through `FOURC_ROOT`, the same environment variable this backend
already uses to resolve Exodus meshes. When it is unset the deck still renders
and still teaches the layout, but it will not run — `requires_fourc_root` marks
those two so callers can say so instead of letting a user discover it at
MPI_Abort time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_DECK_DIR = Path(__file__).resolve().parent / "decks"

# Placeholder that the loader rewrites to an absolute path under FOURC_ROOT.
FOURC_ROOT_TOKEN = "@FOURC_ROOT@"


@dataclass(frozen=True)
class Deck:
    """One executed template."""

    physics: str
    variant: str
    filename: str
    np: int                      # rank count the deck was verified at
    upstream: str                # the tests/input_files deck it derives from
    summary: str                 # one line, what it simulates
    evidence: str                # what was measured to show the physics is live
    requires_fourc_root: bool = False
    pitfalls: tuple[str, ...] = field(default_factory=tuple)

    def path(self) -> Path:
        return _DECK_DIR / self.filename

    def text(self) -> str:
        raw = self.path().read_text()
        return _resolve_root(raw)


def _resolve_root(text: str) -> str:
    """Rewrite @FOURC_ROOT@ to the configured 4C source tree.

    Left verbatim when FOURC_ROOT is unset: a wrong absolute path is a worse
    failure than a visible placeholder, because 4C reports the missing file
    and not the missing configuration.

    EVERY SUBSTITUTED LINE SAYS WHERE ITS PATH CAME FROM. The resolved deck is
    handed to an agent by `generate_input`, so an unannotated
    `/home/<someone>/4C/tests/...` in it reads as a fact about 4C rather than
    as a fact about this machine, and is wrong on every other one. The deck
    header already says to set FOURC_ROOT, but the substituted lines sit
    thousands of characters below it — far outside the window
    test_no_served_payload_hard_codes_a_host_path looks in, and far outside
    what a reader skimming the middle of a deck would connect it to.
    A trailing YAML comment is ignored by 4C, so the deck still runs verbatim.
    """
    if FOURC_ROOT_TOKEN not in text:
        return text
    root = os.environ.get("FOURC_ROOT")
    if not root:
        return text
    resolved = str(Path(root).resolve())
    out = []
    for line in text.split("\n"):
        if FOURC_ROOT_TOKEN not in line:
            out.append(line)
            continue
        line = line.replace(FOURC_ROOT_TOKEN, resolved)
        # Only where a comment cannot change the value: a `#` inside a quoted
        # scalar would be part of the string, and a line that already carries
        # one needs nothing.
        if "#" not in line:
            line = f"{line}    # path from $FOURC_ROOT on this machine"
        out.append(line)
    return "\n".join(out)


# ── the catalog ────────────────────────────────────────────────────────────
# Ordered by physics name. `np` is the rank count the deck was RUN at, not a
# guess: two decks genuinely need more than one rank (see their pitfalls).

DECKS: tuple[Deck, ...] = (
    Deck(
        physics="arterial_network", variant="single_artery_1d",
        filename="arterial_network.4C.yaml", np=1,
        upstream="one_d_3_artery_network.4C.yaml",
        summary="One-dimensional blood flow through a small arterial tree: a "
                "parent vessel bifurcating into two compliant daughter "
                "vessels, with a periodic cardiac inflow waveform.",
        evidence="Explicit Taylor-Galerkin integration of the area/flow "
                 "equations completes with the junction and both reflective "
                 "outlets active; area and flow-rate results written.",
        pitfalls=(
            "The reference area comes from the element line's DIAM, not from "
            "an AREA0 material parameter — there is no AREA0 key.",
            "The junction is closed by DESIGN NODE 1D ARTERY JUNCTION "
            "CONDITIONS entries sharing one ConditionID; a missing partner is "
            "not diagnosed, the tree just leaks.",
        ),
    ),
    Deck(
        physics="beam_interaction", variant="beam_contact_3d",
        filename="beam_interaction_beam_contact.4C.yaml", np=2,
        upstream="beam3eb_static_contact_penalty_linposquadpen_"
                 "beamslidingoverarc.4C.yaml",
        summary="Beam-to-beam penalty contact: a straight Kirchhoff beam is "
                "pressed onto a bent one and slid across it over 18 "
                "quasi-static steps.",
        evidence="4C reports 'currently monitors 3 beam contact pairs' in "
                 "every step, so pairs are found and the penalty law is "
                 "evaluated.",
        pitfalls=(
            "SEARCH_STRATEGY: bounding_volume_hierarchy needs ArborX. On a "
            "build without it every beam-interaction deck aborts in "
            "4C_geometric_search_bounding_volume.hpp:79. The default "
            "bruteforce_with_binning has no such dependency.",
            "BINNING STRATEGY DOMAINBOUNDINGBOX must enclose the DEFORMED "
            "geometry — pairs outside it are silently never searched.",
        ),
    ),
    Deck(
        physics="beam_interaction", variant="beam_solid_meshtying_3d",
        filename="beam_interaction_beam_to_solid.4C.yaml", np=1,
        upstream="beam3r_herm2line3_static_beam_to_solid_volume_meshtying_"
                 "beam_along_solid_boundary_segmentation.4C.yaml",
        summary="Beam-to-solid volume meshtying: a Simo-Reissner beam embedded "
                "in a solid column, tied to it by a Gauss-point-to-segment "
                "penalty constraint, is pulled at its overhanging tip.",
        evidence="Meshtying pairs are monitored every step, and a solid node on the "
                 "FAR side of the column moves with the beam -- load really crosses "
                 "the tie rather than the two meshes sitting independently.",
        pitfalls=(
            "Both condition sections (…VOLUME MESHTYING VOLUME and …LINE) and "
            "both topology sections are mandatory. Drop either and beam and "
            "solid pass through each other with no diagnostic.",
            "See the beam_contact_3d note on ArborX.",
        ),
    ),
    Deck(
        physics="brownian_dynamics", variant="brownian_3d",
        filename="brownian_dynamics.4C.yaml", np=1,
        upstream="beam3r_herm2line3_backweuler_browndyn_crosslinking_"
                 "beam3rline2_additional_fixed_crosslink.4C.yaml",
        summary="Brownian dynamics of semiflexible filaments: crosslinked "
                "beams in a periodic box under thermal forcing, integrated "
                "with overdamped backward Euler.",
        evidence="Re-run with KT: 0.0 as a control: the stochastic case must differ "
                 "from it by a substantial fraction of the deterministic curvature. "
                 "If the two agree, the thermal forcing is parsed and not applied.",
        pitfalls=(
            "KT defaults to 0. BROWNDYNPROB: true with KT unset is a "
            "deterministic run wearing a Brownian label — the upstream deck "
            "this derives from had exactly that defect.",
            "Turning KT on with the default Cylinder_geometry_approx drag "
            "diverges in step 2 (residual 2.67e2 -> 8.74e6, abort in "
            "4C_solver_nonlin_nox_problem.cpp:165). Specifying "
            "BEAMS_DAMPING_COEFF_PER_UNITLENGTH explicitly is what makes the "
            "stochastic step size survivable.",
            "np 1 only: at 2 ranks this aborts in 4C_binstrategy.cpp:1008 "
            "('Node … resides outside the binning domain'). The unmodified "
            "upstream deck fails identically, so it is inherited.",
        ),
    ),
    Deck(
        physics="cardiac_monodomain", variant="monodomain_3d",
        filename="cardiac_monodomain.4C.yaml", np=1,
        upstream="scatra_myocard_MV_material.4C.yaml",
        summary="Cardiac monodomain: the reaction-diffusion equation for the "
                "transmembrane potential coupled to the Bueno-Orovio minimal "
                "ventricular ionic model, stimulated twice.",
        evidence="Read phi at a node through the time loop: it must rise and fall "
                 "through an action potential rather than sitting at its resting "
                 "value. A field that never leaves rest means the stimulus is not "
                 "reaching the tissue.",
        pitfalls=(
            "The element line needs TYPE CardMono and a FIBER1 direction; "
            "DIFF1/DIFF2/DIFF3 are along fibre and the two cross-fibre "
            "directions, so an isotropic-looking material is still "
            "orientation-dependent.",
        ),
    ),
    Deck(
        physics="contact", variant="penalty_3d",
        filename="contact_penalty_3d.4C.yaml", np=1,
        upstream="contact3D_lin_penalty.4C.yaml + "
                 "contact3D_symmetry_penalty_new_struct.4C.yaml",
        summary="Mortar penalty contact between two separate bodies: a stiff "
                "punch descends across an initial gap onto a clamped soft "
                "foundation and indents it.",
        evidence="4C prints 'Total ACTIVE nodes' per step: it must read 0 while the "
                 "gap is open and jump to a non-zero count on the step where the "
                 "prescribed descent closes it. A node under the punch then moves "
                 "while one outside it does not.",
        pitfalls=(
            "LM_SHAPEFCN: Dual needs LM_DUAL_CONSISTENT: none, or "
            "contact_strategy_factory.cpp:263 throws 'Consistent dual shape "
            "functions in boundary elements only for Lagrange multiplier "
            "strategy.'",
            "This is the two-body mortar route. The other contact variant, "
            "inline_penalty_3d, is a single inline block and teaches the "
            "self-contact/simple case instead.",
        ),
    ),
    Deck(
        physics="ehl", variant="ehl_3d",
        filename="ehl.4C.yaml", np=2,
        upstream="ehl3d_mixed.4C.yaml",
        summary="Elastohydrodynamic lubrication: a soft neo-Hookean pad "
                "pressed onto a sliding rigid plate with the oil film "
                "resolved by a Reynolds equation on the pad underside, solved "
                "monolithically with mortar contact.",
        evidence="Newton converges quadratically with 6 to 8 active contact "
                 "nodes in all 20 steps — the mixed lubricated/dry patch is "
                 "genuinely resolved.",
        pitfalls=(
            "CONTACT DYNAMIC STRATEGY must be 'Ehl'; the ordinary contact "
            "strategies do not carry the film coupling.",
            "A viscosity unit slip is not a silent scaling error here — the "
            "monolithic Newton diverges.",
            "[Output] Do NOT add IO/RUNTIME VTK OUTPUT sections to this deck: this coupling runs on the old structure time integration and 4C aborts the run with 'Runtime output is not available in the old structure time integration! ... set INT_STRATEGY: Standard' (which this coupling does not offer). The run writes 4C-native output (<prefix>.control, .result.*, .mesh.*); openPASO converts it to VTU after a normal finish with 4C's post_processor, and the fields appear under output-files/*.vtu (post_processor.log beside them records the conversion). Signal: a run that requests runtime output dies before its first step with the message above. (Measured 2026-09-24 by running the served deck both ways.)",
        ),
    ),
    Deck(
        physics="fbi", variant="penalty_3d",
        filename="fbi.4C.yaml", np=2,
        upstream="fbi_mortar_solidcoupling.4C.yaml",
        summary="Fluid-beam interaction: a slender beam immersed in a 3-D "
                "flow, tied to the fluid by a penalty-regularised mortar "
                "constraint, on a fluid mesh that does not conform to it.",
        evidence="Beam displacement must grow MONOTONICALLY as the free stream ramps "
                 "up, by orders of magnitude over the ramp -- that is the fluid-to- "
                 "beam force transfer being live rather than a one-off transient.",
        pitfalls=(
            "SEARCH_RADIUS must cover a fluid element diagonal; too small and "
            "the beam couples to nothing, with no error — a beam lying "
            "entirely outside the fluid mesh also raises none.",
            "A Dirichlet FUNCT must be SYMBOLIC_FUNCTION_OF_SPACE_TIME; the "
            "time-only form aborts in 4C_utils_function_manager.hpp:143.",
        ),
    ),
    Deck(
        physics="fpsi", variant="monolithic_3d",
        filename="fpsi.4C.yaml", np=1,
        upstream="fpsi_ofsiinterface.4C.yaml",
        summary="Monolithic fluid-porous-structure interaction: free ALE flow "
                "next to a Darcy-saturated neo-Hookean poroelastic block and "
                "an elastic solid, with all three interface families active.",
        evidence="All four fields (fluid, poro structure, porofluid, ALE) "
                 "assemble into one Newton system that converges over the "
                 "driven ramp.",
        pitfalls=(
            "Three distinct condition families are needed and are easy to "
            "confuse: DESIGN FPSI COUPLING SURF (free fluid to porous), "
            "DESIGN FSI COUPLING SURF (free fluid to solid), and DESIGN VOLUME "
            "POROCOUPLING CONDITION (skeleton to pore fluid).",
            "The upstream deck declares a DSURFACE built from nodes that do "
            "not exist; 4C accepts it because no condition references it.",
            "[Output] Do NOT add IO/RUNTIME VTK OUTPUT sections to this deck: this coupling runs on the old structure time integration and 4C aborts the run with 'Runtime output is not available in the old structure time integration! ... set INT_STRATEGY: Standard' (which this coupling does not offer). The run writes 4C-native output (<prefix>.control, .result.*, .mesh.*); openPASO converts it to VTU after a normal finish with 4C's post_processor, and the fields appear under output-files/*.vtu (post_processor.log beside them records the conversion). Signal: a run that requests runtime output dies before its first step with the message above. (Measured 2026-09-24 by running the served deck both ways.)",
        ),
    ),
    Deck(
        physics="fsi_xfem", variant="xfem_fsi_3d",
        filename="fsi_xfem.4C.yaml", np=2,
        upstream="xfsi_3D_boxes.4C.yaml",
        summary="Monolithic fixed-grid FSI: the structure surface cuts a fixed "
                "Eulerian fluid mesh and Nitsche coupling enforces the "
                "interface conditions, with both meshes generated inline.",
        evidence="Traction-driven flow past the immersed rotated box completes "
                 "with the XFEM monolithic coupling active on all six "
                 "structure faces.",
        pitfalls=(
            "COUPALGO must be iter_xfem_monolithic — the ALE-based FSI "
            "algorithms do not apply to a cut mesh.",
            "STRUCTURE DOMAIN and FLUID DOMAIN generate both meshes inline; "
            "the structure must lie inside the fluid box or it cuts nothing.",
            "[Output] Do NOT add IO/RUNTIME VTK OUTPUT sections to this deck: this coupling runs on the old structure time integration and 4C aborts the run with 'Runtime output is not available in the old structure time integration! ... set INT_STRATEGY: Standard' (which this coupling does not offer). The run writes 4C-native output (<prefix>.control, .result.*, .mesh.*); openPASO converts it to VTU after a normal finish with 4C's post_processor, and the fields appear under output-files/*.vtu (post_processor.log beside them records the conversion). Signal: a run that requests runtime output dies before its first step with the message above. (Measured 2026-09-24 by running the served deck both ways.)",
        ),
    ),
    Deck(
        physics="multiscale", variant="fe2_3d",
        filename="multiscale.4C.yaml", np=2,
        upstream="sohex8_multiscale_macro.4C.yaml",
        summary="FE2 computational homogenisation: a macro block with no "
                "constitutive law of its own, whose stress at every Gauss "
                "point comes from solving a boundary value problem on a micro "
                "RVE driven by the local deformation gradient.",
        evidence="64 independent micro problems are opened and written "
                 "(out_microdis{1,2,3}_el*_gp*), one per macro Gauss point per "
                 "micro discretisation — the homogenisation really runs rather "
                 "than the macro material falling back to something local.",
        requires_fourc_root=True,
        pitfalls=(
            "FE2 is intrinsically a TWO-file problem type. "
            "Global::read_micro_fields opens a second, standalone InputFile "
            "per macro multiscale material, so the RVE cannot be inlined; this "
            "deck names 4C's own RVE by absolute path. Omitting MICROFILE does "
            "not help — its default is the literal placeholder 'filename.dat', "
            "so you get the same 'Input file does not exist' abort with a "
            "different name.",
            "Pointing MICROFILE at the deck itself segfaults: the recursive "
            "read corrupts the problem registry and the backtrace surfaces in "
            "an unrelated field's setup.",
            "MICRODIS_NUM numbers the independent micro discretisations; two "
            "macro materials sharing a number share one RVE.",
        ),
    ),
    Deck(
        physics="particle_dem", variant="settling_3d",
        filename="particle_dem.4C.yaml", np=2,
        upstream="particle_dem_1d_normalcontact_linspring_stiffset.4C.yaml",
        summary="Discrete element method: 48 rigid spheres of two sizes fall "
                "under gravity, collide with each other and the six walls of "
                "the bounding box, and settle into a static pack.",
        evidence="Potential energy must fall, kinetic energy decay towards zero, and "
                 "contact energy stay small but NON-zero -- the pack lands and comes "
                 "to rest. Zero contact energy means the particles are passing "
                 "through each other.",
        pitfalls=(
            "PARTICLE_WALL_SOURCE: BoundingBox turns the six faces of "
            "DOMAINBOUNDINGBOX into rigid walls, so a container needs no mesh "
            "at all — but it also needs PARTICLE_WALL_MAT.",
            "The normal stiffness is derived from MAX_VELOCITY and "
            "REL_PENETRATION, not given directly; raising the stiffness "
            "without lowering TIMESTEP silently violates the stability limit.",
        ),
    ),
    Deck(
        physics="particle_sph", variant="hydrostatic_2d",
        filename="particle_sph_hydrostatic.4C.yaml", np=2,
        upstream="particle_sph_1d_hydrostatic_freesurface_densityintegration_"
                 "cubicspline_adami.4C.yaml",
        summary="Weakly compressible SPH: a fluid column on three boundary "
                "layers settles under gravity to the hydrostatic density and "
                "pressure profile.",
        evidence="At the final time the velocity must have decayed to numerical zero "
                 "and the pressure must follow rho0*g*(H-x) through the bulk. Derive "
                 "that profile from your own rho0, g and H and compare; it is the "
                 "point of the case.",
        pitfalls=(
            "There is no SOUNDSPEED key. The artificial speed of sound is "
            "sqrt(BULK_MODULUS/rho0) from MAT_ParticleSPHFluid.",
            "INITRADIUS is the kernel support radius and must match the "
            "kernel: 2x the spacing for a cubic spline, 3x for a quintic.",
            "The upstream deck this derives from stops while its gravity ramp "
            "is still at 35% of g, so it never reaches equilibrium; the ramp "
            "was shortened here.",
        ),
    ),
    Deck(
        physics="particle_sph", variant="dam_break_2d",
        filename="particle_sph_dambreak.4C.yaml", np=2,
        upstream="particle_sph_2d_dambreak_freesurface_"
                 "densitynormalizedreinit.4C.yaml",
        summary="Two-dimensional dam break: a water column released onto a dry "
                "bed inside a closed tank, the standard free-surface SPH "
                "benchmark.",
        evidence="The surge front must advance and the column height drop while the "
                 "density stays within a fraction of a per cent of rho0. A drifting "
                 "density means the equation of state or the time step is wrong, not "
                 "that the dam broke.",
        pitfalls=(
            "TIMESTEP must stay below roughly 0.2*spacing/c with "
            "c = sqrt(BULK_MODULUS/rho0).",
            "Changing the kernel changes how many boundary layers the wall "
            "needs: two for a cubic spline, three for a quintic.",
        ),
    ),
    Deck(
        physics="pasi", variant="dem_impact_3d",
        filename="pasi.4C.yaml", np=2,
        upstream="pasi_twoway_norelax_particle_dem_1d_normalcontact_linspring_"
                 "walldiscretcond.4C.yaml",
        summary="Particle-structure interaction, partitioned two-way: a DEM "
                "sphere presses into an elastic plate, the plate deflects, and "
                "the deformed wall is fed back to the particle solver each "
                "coupling iteration.",
        evidence="The outer loop must converge in every step without hitting ITEMAX, "
                 "and the particle must sink while the plate centre deflects the "
                 "OTHER way -- force goes one way and displacement the other, which "
                 "is the coupling.",
        pitfalls=(
            "PARTICLE_WALL_MOVING and PARTICLE_WALL_LOADED are what make the "
            "coupling two-way; with them false the structure is a rigid "
            "obstacle and PASI silently degenerates to one-way.",
            "Rayleigh K_DAMP scales with the stiffness: the upstream K_DAMP 1 "
            "gives the plate a relaxation time 13x the simulated time, so it "
            "never responds and the run still exits 0.",
        ),
    ),
    Deck(
        physics="plasticity", variant="linear_2d",
        filename="plasticity_linear_2d.4C.yaml", np=1,
        upstream="plastic_pressurisedcylinder.4C.yaml",
        summary="Small-strain J2 (von Mises) elastoplasticity with linear "
                "isotropic hardening, plane strain, displacement controlled "
                "past yield.",
        evidence="Re-run with the yield stress raised out of reach as a control: the "
                 "reaction force must be several times LOWER in the yielding case. "
                 "Identical forces mean the specimen is still elastic and the "
                 "plasticity is not engaging.",
        pitfalls=(
            "There is no 2-D plasticity element on this build. WALL QUAD4 with "
            "a plasticity material aborts in 4C_w1_mat.cpp:179 ('Invalid type "
            "of material law for wall element'); plane strain is done the way "
            "the upstream deck does it, one layer of SOLID HEX8 with u_z "
            "locked by a volume Dirichlet.",
            "Removing that u_z condition turns the deck into a 3-D bar without "
            "any diagnostic.",
        ),
    ),
    Deck(
        physics="plasticity", variant="nonlinear_3d",
        filename="plasticity_nonlinear_3d.4C.yaml", np=1,
        upstream="plastic_necking_eas.4C.yaml",
        summary="Finite-strain J2 elastoplasticity: the classic necking "
                "tensile bar, one eighth modelled with symmetry planes and a "
                "2% taper so localisation picks a plane deterministically.",
        evidence="Against a raised-yield control the reaction force must be lower by "
                 "more than an order of magnitude, the load must pass a MAXIMUM "
                 "part-way through, and the section must contract substantially more "
                 "than the elastic control -- that trio is necking with isochoric "
                 "plastic flow.",
        pitfalls=(
            "TECH eas_mild and TECH fbar abort with SIGFPE inside "
            "evaluate_eas_kinematics for this material (the same EAS with "
            "StVenantKirchhoff runs), so this ships TECH none and accepts HEX8 "
            "volumetric locking.",
            "The hardening law is Voce plus linear: "
            "sigma_y = YIELD + ISOHARD*e_p + (SATHARDENING - YIELD)*"
            "(1 - exp(-HARDEXPO*e_p)).",
        ),
    ),
    Deck(
        physics="poroelast_scatra", variant="homogeneous_3d",
        filename="poroelast_scatra_3d.4C.yaml", np=1,
        upstream="poro_3D_scatra_homogeneous_coupling_reacstart.4C.yaml",
        summary="Poroelasticity with scalar transport through the pore fluid "
                "and a reaction source: the Biot problem of a deforming "
                "saturated solid, carrying a reacting species in the fluid "
                "phase.",
        evidence="The species field must respond to the FLOW, not merely "
                 "diffuse: start the reaction and watch the scalar "
                 "distribution change where the pore fluid is moving. A "
                 "scalar that evolves identically whether the solid deforms "
                 "or not is transport that never received a velocity, and "
                 "the run reports nothing wrong.",
        pitfalls=(
            "[setup] Poroelastic_scalar_transport is a DIFFERENT problem type from "
            "porofluid_pressure_based_elasticity_scatra. Both couple a "
            "deforming porous solid to a transported species; they are "
            "different formulations with different section names, and "
            "picking the wrong one gives section errors that read as a "
            "malformed deck."
            "Signal: measured, BOTH formulations bring up the SAME three discretisations (structure, porofluid, scatra), each with its own fill_complete() line at setup -- so the setup output cannot tell them apart and only the PROBLEMTYPE line can. Check that line before blaming a section name.",
            "[input] RETRACTED AND REPLACED, 2026-09-20. This entry used to "
            "say that the two SMALLEST upstream Poroelastic_scalar_transport "
            "decks (poro_2D_scatra_quad4_partint, poro_2D_scatra_quad9) abort "
            "with std::runtime_error while the larger 3-D ones run clean. That "
            "is FALSE: re-run on the installed binary, both finish with "
            "'processor 0 finished normally' and exit 0. The abort was mine, "
            "not the deck's -- I had invoked 4C with only the input file. "
            "THE REAL FACT, which is worth more: 4C takes BOTH an input and an "
            "output argument, and omitting the output one aborts after the "
            "banner with 'terminate called after throwing an instance of "
            "'FourC::Core::Exception'' and exit 134, core dumped. It does say "
            "what is wrong: 4C_global_full_main.cpp line 457 prints "
            "\"Please provide both\" followed by the two argument names. But "
            "that line sits ABOVE a long MPI backtrace, so a driver that reads "
            "only the tail sees a crash and blames the deck. Signal: exit 134 "
            "with a FourC::Core::Exception and no solver output at all. Invoke "
            "4C with TWO arguments, the input file and an output prefix, and "
            "read the HEAD of the output rather than the tail."        ),
    ),
    Deck(
        physics="fluid_ale", variant="hdg_2d",
        filename="fluid_ale_hdg_2d.4C.yaml", np=1,
        upstream="hdg_weakly_compressible_etienne_cfd.4C.yaml",
        summary="Weakly compressible flow on a DEFORMING domain: the fluid is "
                "solved on a mesh that moves, which is the Arbitrary "
                "Lagrangian-Eulerian setting every moving-boundary and FSI "
                "problem needs. Discretised with 4C's hybridisable "
                "discontinuous Galerkin fluid.",
        evidence="Check that the ALE displacement field is NON-ZERO and "
                 "changes over the run: PROBLEMTYPE Fluid_Ale parses and "
                 "solves perfectly well with a mesh that never moves, which "
                 "is an ordinary fixed-grid fluid run wearing an ALE label. "
                 "The moving mesh is the thing being tested, so look at it.",
        pitfalls=(
            "[setup] Fluid_Ale is a DISTINCT problem type from Fluid. Choosing Fluid "
            "and then adding a mesh-motion section does not give you ALE; 4C "
            "reads the sections its problem type declares and silently "
            "ignores the rest."
            "Signal: measured, Fluid_Ale brings up TWO discretisations, ale and fluid, each with its own fill_complete() line at setup. Plain Fluid brings up no ale discretisation at all, so counting those lines tells you which problem type you actually got.",
            "[setup] An ALE run needs BOTH a fluid and an ALE discretisation, and the "
            "two must cover the same region. A missing ALE domain is not a "
            "parse error -- the mesh simply does not move."
            "Signal: the setup must bring up an ALE discretisation of its own -- look for its fill_complete() line beside the fluid one. A run that lists only the fluid has no mesh motion, whatever the mesh-motion section says.",
            "[Output] The HDG fluid ignores IO/RUNTIME VTK OUTPUT (the run finishes normally and writes no VTK), and post_processor --filter=vtu fails on its mixed variables (4C_post_vtk_vtu_writer.cpp); only --filter=ensight writes a readable .case/.geo pair, which openPASO's result gate does not read. Signal: no output-vtk-files/ or *.vtu after a normal finish; post_processor.log ends in an MPI abort naming the VTU writer. Judge the run from the log. (Measured 2026-09-24 by running the served deck.)",
        ),
    ),
    Deck(
        physics="porofluid_elasticity_scatra", variant="monolithic_3d",
        filename="porofluid_elasticity_scatra_monolithic_3d.4C.yaml", np=1,
        upstream="porofluid_pressure_based_elast_scatra_3D_hex8_mono_FD.4C.yaml",
        summary="Multiphase flow through a deformable porous medium WITH "
                "scalar transport on top: three fields -- porofluid, solid "
                "skeleton and one or more transported species -- solved as a "
                "single monolithic Newton system.",
        evidence="Watch all THREE fields, not two: the scalar must move "
                 "BECAUSE the porofluid moves. A scatra field that stays at "
                 "its initial value while the pressure and displacement "
                 "evolve means the transport is riding on a velocity it never "
                 "received, and the run reports nothing wrong. The upstream "
                 "deck this derives from also enables 4C's finite-difference "
                 "check of the monolithic matrix, which is worth keeping on "
                 "while you develop: it compares the assembled Jacobian "
                 "against a numerical one and reports the largest relative "
                 "error.",
        pitfalls=(
            "[setup] porofluid_pressure_based_elasticity_scatra is a THIRD problem "
            "type, distinct from porofluid_pressure_based_elasticity and from "
            "Poroelastic_scalar_transport. All three exist, all three couple "
            "flow to a solid, and they take different section names."
            "Signal: measured, this deck brings up exactly two discretisations, porofluid and structure. A run that brings up a third, or names something else, is not in the formulation you think it is.",
            "[performance] 4C's finite-difference check of the monolithic system matrix is "
            "a development tool that costs a full extra assembly per step. It "
            "is what tells you an off-diagonal coupling block is wrong, which "
            "is otherwise invisible -- the run converges to a plausible "
            "answer with a Jacobian that is merely approximate."
            "Signal: it multiplies the cost of every Newton step, so compare the wall time of a single step with it on and off before leaving it enabled anywhere but a debugging run.",
        ),
    ),
    Deck(
        physics="porofluid_elasticity", variant="monolithic_3d",
        filename="porofluid_elasticity_monolithic_3d.4C.yaml", np=1,
        upstream="porofluid_pressure_based_elast_3D_hex27.4C.yaml",
        summary="Pressure-based porous-media flow coupled monolithically to "
                "an elastic solid skeleton: one fluid phase in a deforming "
                "porous solid, solved as a single Newton system rather than "
                "by staggering the two fields.",
        evidence="Run it and watch the porosity and the solid displacement "
                 "TOGETHER over the time history: a monolithic poro-elastic "
                 "solve must show the porosity changing BECAUSE the skeleton "
                 "deforms. Porosity that stays at its initial value while the "
                 "solid moves means the two fields are not actually coupled -- "
                 "the deck parses and runs either way.",
        pitfalls=(
            "[setup] PROBLEMTYPE is 'porofluid_pressure_based_elasticity', which is a "
            "DIFFERENT problem type from 'Poroelasticity' and from "
            "'porofluid_pressure_based'. All three exist in 4C and they take "
            "different section names; picking the wrong one gives section "
            "errors that read as though your deck is malformed."
            "Signal: measured, this brings up three discretisations -- porofluid, scatra and structure. Poroelastic_scalar_transport brings up the same three, so this list separates it from the two-discretisation formulations but NOT from that one.",
            "[setup] The coupling is selected by coupling_scheme: twoway_monolithic "
            "under porofluid_elasticity_dynamic. The staggered alternative is "
            "a different scheme keyword, and switching it changes which "
            "nonlinear solver block 4C reads."
            "Signal: a monolithic run solves ONE linear system per Newton step and a staggered one alternates two, so the per-step solver output tells you which you actually got.",
            "[validation] 4C's RESULT DESCRIPTION block is its regression self-check, not "
            "part of running a simulation. This deck ships without one on "
            "purpose. If you add one, every VALUE you write is an assertion "
            "4C will enforce to the TOLERANCE you give -- a wrong value fails "
            "the run rather than being ignored."
            "Signal: removing the block changes nothing else in the run -- measured here by running the deck with and without it and comparing the fields. If a number moves when you remove it, the deck was leaning on the self-check, which it must never do.",
        ),
    ),
    Deck(
        physics="porous_media", variant="terzaghi_2d",
        filename="porous_media_terzaghi_2d.4C.yaml", np=1,
        upstream="poro_2D_quad4_br_stsplit_nbc.4C.yaml + "
                 "poro_2D_quad4_linporo.4C.yaml",
        summary="Terzaghi one-dimensional consolidation: a saturated soil "
                "column loaded at the drained top surface, with the pore "
                "pressure carrying the load initially and dissipating over "
                "time as the skeleton takes it up.",
        evidence="The base pore pressure must start near the applied load q (the "
                 "undrained limit, where the fluid carries everything) and fall "
                 "MONOTONICALLY towards zero as the skeleton takes it up, with the "
                 "settlement growing towards the drained oedometric value q*H/E_oed "
                 "that you evaluate for your own parameters. Pressure that does not "
                 "dissipate means the top is not draining.",
        pitfalls=(
            "Poroelasticity requires the SAME THETA in STRUCTURAL "
            "DYNAMIC/ONESTEPTHETA and in FLUID DYNAMIC, or "
            "poroelast_base.cpp:182 throws 'porous media problem is limited in "
            "functionality'.",
            "Runtime VTK output needs STRUCTURAL DYNAMIC INT_STRATEGY: "
            "Standard; the old integrator throws 'Runtime output is not "
            "available in the old structure time integration!'",
            "In 2-D the porofluid VTU 'pressure' array is all NaN and the pore "
            "pressure lands in the THIRD component of 'velocity' — "
            "FluidImplicitTimeInt::write_runtime_output hardcodes three "
            "velocity components. The 3-D output is fine.",
        ),
    ),
    Deck(
        physics="porous_media", variant="consolidation_3d",
        filename="porous_media_consolidation_3d.4C.yaml", np=1,
        upstream="poro_3D_hex8_stat.4C.yaml + poro_2D_quad4_linporo.4C.yaml",
        summary="Three-dimensional consolidation under a surface load, the "
                "HEX8 counterpart of the Terzaghi column.",
        evidence="Same signals as the 2-D deck, and the two must AGREE: the physics "
                 "here is one-dimensional, so a 3-D column and a 2-D one should give "
                 "the same base pressure history and the same settlement. That "
                 "agreement is the cross-check.",
        pitfalls=(
            "There is no SOLIDH8PORO element. It appears in zero files of the "
            "4C source, zero upstream decks and is absent from the grammar "
            "index; earlier openPASO knowledge named it as the 3-D poro element. "
            "The real ones are SOLIDPORO_PRESSURE_VELOCITY_BASED (used here, "
            "and the only one the 25 upstream Poroelasticity decks use), "
            "SOLIDPORO_PRESSURE_VELOCITY_BASED_P1 (porosity as a 4th nodal "
            "unknown, needs PHYSICAL_TYPE: Poro_P1) and "
            "SOLIDPORO_PRESSURE_BASED (no fluid-velocity field; belongs to the "
            "pressure-based multiphase module, not to Poroelasticity).",
            "See the terzaghi_2d note on matching THETA.",
        ),
    ),
    Deck(
        physics="reduced_lung", variant="lung_1d",
        filename="reduced_lung.4C.yaml", np=2,
        upstream="reduced_lung_1d_pipe_flow_continuous.4C.yaml",
        summary="One-dimensional compliant-tube airway flow: a pressure pulse "
                "propagating along a pipe whose diameter halves midway, so it "
                "partially reflects at the area change.",
        evidence="Wave propagation and partial reflection resolved over 10000 "
                 "steps with the flow inlet and reflecting outlet active.",
        pitfalls=(
            "This is PROBLEMTYPE Reduced_Lung_1D_Pipe_Flow. The other lung "
            "problem type, Reduced_Lung (the lung-tree model), CANNOT be "
            "written as a single file on this build: its topology comes "
            "through from_file / from_mesh / field_reference, and the "
            "top-level `fields:` section only offers separate_file or "
            "from_mesh, so there is no way to give node coordinates inline. "
            "The `constant:` alternative assigns one value to every index, "
            "which collapses the tree — measured: 'Multiple pressure boundary "
            "conditions assigned to node 1', then SIGFPE from a zero reference "
            "volume once all nodes coincide.",
            "Nearly all the physics is in one lowercase top-level section, "
            "`reduced_lung:`, not in the usual upper-case DYNAMIC sections.",
        ),
    ),
    Deck(
        physics="ssi", variant="monolithic_elch_3d",
        filename="ssi.4C.yaml", np=1,
        upstream="ssi_2D_quad4.4C.yaml",
        summary="Structure-scalar interaction: a scalar transported on a "
                "deforming mesh, with the scatra discretisation cloned from "
                "the structure so both fields share nodes.",
        evidence="Staggered solid-to-scatra loop runs to completion with the "
                 "scalar transported in conservative form on the stretched "
                 "element.",
        pitfalls=(
            "The structure elements must be a *SCATRA type (WALLSCATRA, "
            "SOLIDSCATRA, …) with a meaningful TYPE. A plain WALL/SOLID aborts "
            "in 4C_ssi_clonestrategy.cpp:97, naming ImplType 'Undefined'.",
            "COUPALGO chooses one-way, staggered or monolithic; the one-way "
            "variants run happily and simply do not feed the scalar back.",
        ),
    ),
    Deck(
        physics="ssti", variant="monolithic_3d",
        filename="ssti.4C.yaml", np=1,
        upstream="ssti_mono_3D_3hex8_elch_s2i_butlervolmerthermo_"
                 "growthlaw.4C.yaml",
        summary="Structure-scalar-thermo interaction: a 1-D lithium-ion cell "
                "(anode / electrolyte / cathode) solved monolithically for "
                "displacement, lithium concentration, potential and "
                "temperature at once.",
        evidence="Four-field monolithic Newton converges with Butler-Volmer "
                 "kinetics plus thermal contact resistance active on both "
                 "electrode-electrolyte interfaces.",
        pitfalls=(
            "S2I kinetics and SSTI interface meshtying are separate condition "
            "families and both are needed at each interface.",
            "Electrode swelling comes from the inelastic factors of "
            "MAT_MultiplicativeSplitDefgradElastHyper, not from a thermal "
            "expansion coefficient on the elastic material.",
        ),
    ),
    Deck(
        physics="sti", variant="monolithic_3d",
        filename="sti.4C.yaml", np=1,
        upstream="sti_mono_3D_hex8_elch_s2i_butlervolmerpeltier_adiabatic_"
                 "mortar_standard.4C.yaml",
        summary="Monolithic scatra-thermo interaction: electrochemistry "
                "(lithium concentration plus potential) and temperature solved "
                "in one Newton system, coupled by Butler-Volmer-Peltier "
                "kinetics across non-conforming mortar interfaces.",
        evidence="Cell voltage must fall and state of charge decrease over the "
                 "discharge, the interface current density must come back EXACTLY as "
                 "the value you applied, and the Peltier and Joule heat fluxes must "
                 "both be non-zero across the interfaces. A current density that "
                 "does not match what you imposed is the clearest sign the "
                 "electrochemistry is not coupled to the thermal field.",
        pitfalls=(
            "Without ELCH CONTROL the run stops with 'Invalid type of closing "
            "equation for electric potential'.",
            "Set SORET to 0 and the species field still moves, not just the "
            "temperature — Soret is a cross-coupling, not the whole coupling.",
        ),
    ),
    Deck(
        physics="xfem_fluid", variant="xfem_3d",
        filename="xfem_fluid.4C.yaml", np=2,
        upstream="xfluid_ls_neumann_inflow_stab.4C.yaml",
        summary="XFEM fluid: transient incompressible Navier-Stokes on a fixed "
                "Eulerian mesh cut by a level-set circle, with a Neumann "
                "traction imposed on the embedded interface.",
        evidence="Cut elements are enriched and the level-set Neumann "
                 "condition is integrated on the embedded circle over the "
                 "whole time loop.",
        pitfalls=(
            "The interface is a FUNCT level set, so refining the interface "
            "means refining FLUID DOMAIN subdivisions — there is no interface "
            "mesh to refine.",
            "Ghost-penalty and mass-conservation parameters live in XFLUID "
            "DYNAMIC/STABILIZATION, separately from the ordinary fluid "
            "stabilisation.",
        ),
    ),
    Deck(
        physics="fluid_turbulence", variant="les_channel_3d",
        filename="fluid_turbulence.4C.yaml", np=2,
        upstream="f3_cha_8x8x8_recongradl2.4C.yaml + "
                 "f3_stokes_residualbased_rotboxgeom.4C.yaml",
        summary="Large-eddy simulation of turbulent channel flow of height 2: "
                "Smagorinsky subgrid model, periodic in x and z, driven by a "
                "constant streamwise body force.",
        evidence="4C reports 'Turbulence model : Smagorinsky with Smagorinsky "
                 "constant Cs= 0.1' and opens plane-and-time averaged channel "
                 "statistics over the sampling window.",
        pitfalls=(
            "On ONE MPI rank this deck aborts inside "
            "Core::Conditions::PeriodicBoundaryConditions::balance_load with "
            "'terminate called after throwing an instance of int' (SIGABRT, "
            "rc 134). It runs clean on 2 and 4 ranks; reproduced three times. "
            "Periodic boundary conditions need np > 1 on this build.",
            "Every boundary here is periodic or Dirichlet, so the pressure has "
            "a null space. Without DESIGN VOL MODE FOR KRYLOV SPACE PROJECTION "
            "the run stops with 'Nullspace check for sysmat_ failed'.",
            "8x8x8 is a demonstration resolution. An LES on an under-resolved "
            "mesh runs and produces wrong statistics silently.",
        ),
    ),
    Deck(
        physics="fs3i", variant="fs3i_3d",
        filename="fs3i.4C.yaml", np=1,
        upstream="fsi_fp_mono_fs_ga_ga.4C.yaml + fs3i_part_1wc_finperm.4C.yaml",
        summary="Fluid-structure-scalar interaction: a scalar transported in "
                "the fluid and in the solid, exchanging mass across the FSI "
                "interface through a permeability condition.",
        evidence="The fluid scalar must fall and the solid scalar rise from zero "
                 "over the run, read from the scatra1/scatra2 VTU output, while the "
                 "ALE displacement reaches its prescribed value. A solid scalar that "
                 "stays at zero means the interface transfer is not happening.",
        requires_fourc_root=True,
        pitfalls=(
            "FS3I rejects direct solvers. 4C_fs3i_partitioned.cpp:604 throws "
            "'Iterative solver expected' unless COUPLED_LINEAR_SOLVER names a "
            "Belos solver, :610 demands AZPREC Teko, and "
            "4C_linear_solver_preconditioner_teko.cpp:48 then throws "
            "'TEKO_XML_FILE parameter not set!'. This is the one deck here "
            "that cannot use UMFPACK and cannot be a single file.",
            "A plain SOLID element aborts in 4C_ssi_clonestrategy.cpp:97 — the "
            "structure elements must be SOLIDSCATRA / WALLSCATRA / SHELLSCATRA "
            "/ TRUSS3SCATRA carrying a meaningful TYPE.",
            "Np_Gen_Alpha for the fluid aborts in 4C_fs3i.cpp:204; BDF2 and "
            "Stationary are rejected as well. One_Step_Theta in all three "
            "fields works.",
        ),
    ),
    Deck(
        physics="fsi", variant="fsi_2d",
        filename="fsi_2d.4C.yaml", np=1,
        upstream="volmortar2D_fsi.4C.yaml + fsi_fp_mono_fs_ga_ga.4C.yaml",
        summary="Partitioned (Dirichlet-Neumann) 2-D fluid-structure "
                "interaction: an elastic block pushed on its far edge drives "
                "an incompressible channel flow on a deforming ALE mesh.",
        evidence="The FSI outer loop must converge with a NON-ZERO interface "
                 "increment, and structure, fluid and ALE result files must all be "
                 "written. A zero increment is the classic sign of a coupling that "
                 "is running but exchanging nothing.",
        pitfalls=(
            "A Dirichlet FUNCT must be a SYMBOLIC_FUNCTION_OF_SPACE_TIME. "
            "Giving it a SYMBOLIC_FUNCTION_OF_TIME aborts in "
            "4C_utils_function_manager.hpp:143 with 'You tried to query "
            "function 1 as a function of type FunctionOfSpaceTime. Actually, "
            "it has type FunctionOfTime.'",
            "With COUPALGO iter_monolithicfluidsplit the FLUID is the slave "
            "field and may carry no Dirichlet condition on interface dofs — "
            "4C_fsi_monolithicfluidsplit.cpp:135 prints a boxed diagnostic "
            "naming master and slave.",
        ),
    ),
    Deck(
        physics="particle_pd", variant="plate_2d",
        filename="particle_pd_plate.4C.yaml", np=1,
        upstream="none — no bond-based PD deck exists upstream; built from the "
                 "grammar dump and 4C's own PD generator script",
        summary="Bond-based peridynamics: a pre-cracked plate pulled in "
                "tension until the crack runs from the notch tip.",
        evidence="The mean damage must start at whatever the pre-crack alone "
                 "accounts for and RISE through the run -- bonds breaking beyond the "
                 "notch. Damage that never moves means the loading is not reaching "
                 "the bonds.",
        pitfalls=(
            "PD is not a separate interaction mode. INTERACTION must be SPH "
            "and PD_BODY_INTERACTION true; the PD parameters then live in "
            "PARTICLE DYNAMIC/PD.",
            "A 2-D PD_DIMENSION requires PARTICLE DYNAMIC/INITIAL AND BOUNDARY "
            "CONDITIONS CONSTRAINT: Projection2D. Without it 4C aborts in "
            "4C_particle_interaction_sph_peridynamic.cpp:92 with 'Plane stress "
            "or plane strain for peridynamic requested. CONSTRAINT must be set "
            "to Projection2D!'",
            "PDFIXED 1 pins a particle at its reference position; PDFIXED 2 "
            "makes it part of a rigid body moved at IMPACTOR_VELOCITY.",
        ),
    ),
    Deck(
        physics="particle_pd", variant="impact_2d",
        filename="particle_pd_impact.4C.yaml", np=1,
        upstream="none — see plate_2d",
        summary="Kalthoff-Winkler edge impact: a rigid impactor strikes a "
                "doubly-notched plate between the notches.",
        evidence="Both peridynamic bodies must be present, the mean damage must rise "
                 "over the run, and particle speeds must reach the order of the "
                 "impactor velocity you imposed. A second body that is absent is the "
                 "usual failure here.",
        pitfalls=(
            "The impactor is a second PDBODYID whose particles carry PDFIXED 2; "
            "contact between bodies is the NORMALCONTACTLAW / NORMAL_STIFF "
            "penalty pair in PARTICLE DYNAMIC/PD.",
            "dx = 12.5 mm here is a teaching resolution. The published "
            "Kalthoff-Winkler study resolves the same specimen at dx = 0.5 mm; "
            "crack paths at this spacing are indicative only.",
        ),
    ),
)


_BY_KEY = {(d.physics, d.variant): d for d in DECKS}


def get(physics: str, variant: str) -> Deck | None:
    return _BY_KEY.get((physics, variant))


def render(physics: str, variant: str) -> str | None:
    """Return the deck text with a short provenance header, or None."""
    d = get(physics, variant)
    if d is None:
        return None
    head = [
        f"# 4C {d.physics} / {d.variant} — runnable template",
        f"# {d.summary}",
        f"# Verified: executed on the installed 4C binary with "
        f"{d.np} MPI rank{'s' if d.np > 1 else ''}, exit 0 — on the 4C build "
        f"this catalog was last verified against, which is not necessarily "
        f"yours. Re-run it before trusting it on a different build.",
        # The deck teaches 4C's INPUT GRAMMAR, which cannot be guessed and is
        # the reason these exist. It is not a worked answer: what the run
        # produces is for the run to produce. This header used to read
        # "Evidence the physics is live: <the measured result>", which handed
        # the agent the number before it had run anything.
        f"# What to check yourself, to confirm the physics is live rather "
        f"than merely parsing: {d.evidence}",
        f"# Derived from upstream deck(s): {d.upstream}",
    ]
    if d.requires_fourc_root:
        head.append(
            "# NOTE: this deck names a file from the 4C source tree; set "
            "FOURC_ROOT so the path resolves.")
    for p in d.pitfalls:
        head.append(f"# Pitfall: {p}")
    return "\n".join(head) + "\n" + d.text()


def variants_for(physics: str) -> list[str]:
    return [d.variant for d in DECKS if d.physics == physics]


def knowledge_for(physics: str) -> dict:
    """The deck catalog's own description and pitfalls for one physics.

    Four physics rows (poroelast_scatra, fluid_ale, porofluid_elasticity and
    porofluid_elasticity_scatra) are served ENTIRELY from this catalog: they
    have no generator class, so get_knowledge() found neither a data-file
    entry nor a generator entry and returned {"error": ...} while the
    templates ran fine. The summaries and pitfalls were already here, measured
    by running each deck on the installed binary; they were simply not wired
    to the knowledge path.
    """
    decks = [d for d in DECKS if d.physics == physics]
    if not decks:
        return {}
    pitfalls: list[str] = []
    seen: set[str] = set()
    for d in decks:
        for p in (d.pitfalls or ()):
            if p not in seen:
                pitfalls.append(p)
                seen.add(p)
    out: dict = {"description": decks[0].summary,
                 "variants": [d.variant for d in decks]}
    if pitfalls:
        out["pitfalls"] = pitfalls
    checks = [d.evidence for d in decks if getattr(d, "evidence", None)]
    if checks:
        out["what_to_check"] = checks
    return out


def physics_covered() -> list[str]:
    seen: list[str] = []
    for d in DECKS:
        if d.physics not in seen:
            seen.append(d.physics)
    return seen
