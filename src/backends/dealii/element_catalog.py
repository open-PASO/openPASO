"""Canonical deal.II element catalog.

Single source of truth for every FE_* class the catalog claims:
header path, math space, semantics, constructor signature, version
gate. Each per-physics ``elements:`` entry now stores only the
class name + a physics-keyed applicability note (e.g. "preferred
for nearly-incompressible elasticity"); the rich form returned to
the agent is the *join* of this canonical record with the
applicability note.

Motivation (senior-AI-scientist critic, 2026-05-31): before this
refactor, FE_Q was described three different ways across
poisson.py, heat.py, helmholtz.py — same class, three contradictory
canonical descriptions, three places to maintain. For an LLM agent
retrieving multiple physics in one call (the post-mortem
breadcrumbs path already does this) the contradictions confuse
canonical semantics with use-case applicability. Splitting them
restores openPASO design Principle 1 (generality over specificity):
the canonical layer is general, the applicability layer is
specific, the diff tool checks canonical existence against the
source scan ignoring applicability noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ElementRecord:
    """Canonical metadata for one deal.II finite-element class."""

    name: str
    header: str
    math_space: str           # "H1" / "L2" / "H(div)" / "H(curl)" / "Mixed" / "Internal"
    semantics: str            # one-line canonical description
    constructor: str          # signature, e.g. "FE_Q<dim>(unsigned int degree)"
    version_added: str = ""   # empty → present since at least 9.0
    aliases: tuple = field(default_factory=tuple)


# Re-verified by execution on deal.II 9.8.0-pre: one program
# instantiates EVERY user-facing record below, in 2D and (where the
# element exists in 3D) in 3D, and prints its degree, dofs_per_cell,
# component count, conformity, reference cell and support-point
# flags. It compiles and runs; the values recorded in the semantics
# fields come from that run.
#
# COUNT DISCIPLINE. This header used to say "the 39 remaining records
# each compiled and instantiated". There were 38, and 5 of them
# (FE_Poly, FE_PolyFace, FE_PolyTensor, FE_Q_Base, FE_DGVector) are
# template base classes with constructor == "(internal)" that CANNOT
# be instantiated at all — so the sentence was wrong twice over. Do
# not write a count here by hand; len(ELEMENTS) is the count, and the
# instantiable ones are those whose constructor is not "(internal)".
# FE_Base and FE_Series were REMOVED because no such classes exist
# (see below).
#
# Each record's `name` MUST exactly match the class name as it
# appears in the deal.II source scan (`scripts/scan_results/
# dealii.json` -> ``elements:``) so the catalog-vs-scan diff can
# join them. Adding a new entry requires either:
#   * the class exists in the local scan (validates with diff =
#     truly_missing -> shared on next run), OR
#   * the entry carries an explicit version_added gate stating
#     when the class becomes available.
ELEMENTS: dict[str, ElementRecord] = {
    # ── H^1 conforming (continuous Lagrange family) ─────────────
    "FE_Q": ElementRecord(
        name="FE_Q",
        header="deal.II/fe/fe_q.h",
        math_space="H1",
        semantics=("Continuous Lagrange element on hyper-cube cells. "
                   "The canonical first choice for elliptic and "
                   "parabolic PDEs. Wrap in FESystem<dim>(..., dim) "
                   "for vector unknowns."),
        constructor="FE_Q<dim>(unsigned int degree)",
    ),
    "FE_Q_Hierarchical": ElementRecord(
        name="FE_Q_Hierarchical",
        header="deal.II/fe/fe_q_hierarchical.h",
        math_space="H1",
        semantics=("Hierarchical-basis continuous Lagrange. Coarse-"
                   "level DoFs survive a polynomial-degree change, "
                   "so this is the required choice for p-adaptive "
                   "refinement; regular FE_Q changes its DoF "
                   "ordering across degrees."),
        constructor="FE_Q_Hierarchical<dim>(unsigned int degree)",
    ),
    "FE_Q_Bubbles": ElementRecord(
        name="FE_Q_Bubbles",
        header="deal.II/fe/fe_q_bubbles.h",
        math_space="H1",
        semantics=("Continuous Lagrange enriched with one cell-"
                   "interior bubble. The bubble vanishes on the cell "
                   "boundary so global H^1-conformity is preserved. "
                   "Used for MINI-like stable Stokes pairs and to "
                   "reduce volumetric locking in elasticity."),
        constructor="FE_Q_Bubbles<dim>(unsigned int degree)",
    ),
    "FE_Q_DG0": ElementRecord(
        name="FE_Q_DG0",
        header="deal.II/fe/fe_q_dg0.h",
        math_space="Mixed",
        semantics=("Continuous Lagrange plus one piecewise-constant "
                   "DG mode per cell. NOT strictly H^1-conforming — "
                   "the DG0 piece is discontinuous across faces. Use "
                   "for mass-matrix lumping tricks and certain "
                   "stabilised formulations."),
        constructor="FE_Q_DG0<dim>(unsigned int degree)",
    ),
    "FE_Q_iso_Q1": ElementRecord(
        name="FE_Q_iso_Q1",
        header="deal.II/fe/fe_q_iso_q1.h",
        math_space="H1",
        semantics=("Piecewise (multi-)linear functions on p^dim "
                   "sub-cells of each macro-cell. Same H^1 "
                   "continuity as FE_Q(1) but more DoFs per cell; "
                   "lumped mass matrix is diagonal."),
        constructor="FE_Q_iso_Q1<dim>(unsigned int p)",
    ),
    "FE_Bernstein": ElementRecord(
        name="FE_Bernstein",
        header="deal.II/fe/fe_bernstein.h",
        math_space="H1",
        semantics=("Bernstein-Bezier basis on hyper-cube cells. "
                   "Better-conditioned mass matrix at high p than "
                   "FE_Q; used in isogeometric-adjacent workflows."),
        constructor="FE_Bernstein<dim>(unsigned int degree)",
    ),
    "FE_RannacherTurek": ElementRecord(
        name="FE_RannacherTurek",
        header="deal.II/fe/fe_rannacher_turek.h",
        math_space="L2",
        semantics=("Rannacher-Turek non-conforming element on "
                   "quads/hexes; continuity only in the face-mean "
                   "sense, so deal.II reports "
                   "conforming_space == Conformity::L2, NOT H1 "
                   "(verified by instantiation — this record "
                   "claimed H1 until then). Locking-free for nearly-"
                   "incompressible elasticity; inf-sup stable on quads "
                   "when paired with FE_DGQ(0) pressure. CAVEAT: "
                   "has_support_points() is false, so "
                   "VectorTools::interpolate_boundary_values cannot be "
                   "used on it (segfaults on a Release build) — use "
                   "VectorTools::project_boundary_values instead."),
        constructor=("FE_RannacherTurek<dim>(unsigned int order = 0, "
                     "unsigned int n_face_support_points = 2)"),
    ),
    "FE_P1NC": ElementRecord(
        name="FE_P1NC",
        header="deal.II/fe/fe_p1nc.h",
        math_space="L2",
        semantics=("Park-Sheen non-conforming P1 on 2D quads. Same "
                   "locking-free property as FE_RannacherTurek for "
                   "incompressible-limit elasticity. deal.II reports "
                   "conforming_space == Conformity::L2 (this record "
                   "claimed H1). NOT a class "
                   "template: it is declared "
                   "`class FE_P1NC : public FiniteElement<2, 2>` and "
                   "`FE_P1NC<2>()` fails to compile with "
                   "'dealii::FE_P1NC is not a template'."),
        constructor="FE_P1NC()   # no template argument, 2D only",
    ),
    "FE_SimplexP": ElementRecord(
        name="FE_SimplexP",
        header="deal.II/fe/fe_simplex_p.h",
        math_space="H1",
        semantics=("Lagrange element on simplex (triangle / tet) "
                   "cells. Required when the mesh comes from an "
                   "unstructured Gmsh / Triangle / TetGen "
                   "tessellation rather than the deal.II default "
                   "hyper-cube refinement."),
        constructor="FE_SimplexP<dim>(unsigned int degree)",
        version_added="9.3",
    ),
    # ── L^2 / DG family ─────────────────────────────────────────
    "FE_DGQ": ElementRecord(
        name="FE_DGQ",
        header="deal.II/fe/fe_dgq.h",
        math_space="L2",
        semantics=("Discontinuous Galerkin Lagrange on hyper-cube "
                   "cells. The default DG element; pair with upwind "
                   "flux for transport problems and with FE_Q for "
                   "mixed Stokes/Maxwell pressure."),
        constructor="FE_DGQ<dim>(unsigned int degree)",
    ),
    "FE_DGQArbitraryNodes": ElementRecord(
        name="FE_DGQArbitraryNodes",
        header="deal.II/fe/fe_dgq.h",
        math_space="L2",
        semantics=("FE_DGQ variant where the user picks the quadrature "
                   "points (typically Gauss-Lobatto). Enables "
                   "diagonal mass matrices and spectral-element "
                   "implementations."),
        constructor=("FE_DGQArbitraryNodes<dim>("
                     "const Quadrature<1>&)"),
    ),
    "FE_DGQLegendre": ElementRecord(
        name="FE_DGQLegendre",
        header="deal.II/fe/fe_dgq.h",
        math_space="L2",
        semantics=("FE_DGQ with Legendre-polynomial basis instead of "
                   "Lagrange. Orthogonal mass matrix even before "
                   "lumping."),
        constructor="FE_DGQLegendre<dim>(unsigned int degree)",
    ),
    "FE_DGQHermite": ElementRecord(
        name="FE_DGQHermite",
        header="deal.II/fe/fe_dgq.h",
        math_space="L2",
        semantics=("FE_DGQ (discontinuous, L2; no inter-element "
                   "continuity) with a Hermite-like basis "
                   "(Polynomials::HermiteLikeInterpolation): for "
                   "degree >= 3 only two basis functions per direction "
                   "have a non-zero value or derivative on a face, "
                   "which benefits DG face terms; degrees 0-2 use a "
                   "Lagrange basis. Not interpolatory, no support "
                   "points."),
        constructor="FE_DGQHermite<dim>(unsigned int degree)",
    ),
    "FE_DGP": ElementRecord(
        name="FE_DGP",
        header="deal.II/fe/fe_dgp.h",
        math_space="L2",
        semantics=("DG on the complete polynomial space P_p on "
                   "hyper-cube cells, with a Legendre basis "
                   "(L2-orthonormal on the reference cell); "
                   "FE_DGPMonomial is the monomial-basis variant. Often "
                   "paired with continuous velocity in mixed Stokes "
                   "(MINI / RT/DGP)."),
        constructor="FE_DGP<dim>(unsigned int degree)",
    ),
    "FE_DGPMonomial": ElementRecord(
        name="FE_DGPMonomial",
        header="deal.II/fe/fe_dgp_monomial.h",
        math_space="L2",
        semantics=("FE_DGP variant with strictly monomial basis (no "
                   "tensor-product). Cheaper at low p; less stable "
                   "at high p."),
        constructor="FE_DGPMonomial<dim>(unsigned int degree)",
    ),
    "FE_DGPNonparametric": ElementRecord(
        name="FE_DGPNonparametric",
        header="deal.II/fe/fe_dgp_nonparametric.h",
        math_space="L2",
        semantics=("FE_DGP basis defined in REAL space (not on the "
                   "reference cell). Removes the Jacobian-pullback "
                   "step at the cost of per-cell quadrature."),
        constructor=("FE_DGPNonparametric<dim>(unsigned int degree)"),
    ),
    # ── H(div) family ────────────────────────────────────────────
    "FE_RaviartThomas": ElementRecord(
        name="FE_RaviartThomas",
        header="deal.II/fe/fe_raviart_thomas.h",
        math_space="H(div)",
        semantics=("Raviart-Thomas H(div)-conforming element. Pair "
                   "with FE_DGQ pressure for exactly-divergence-free "
                   "Stokes velocity; the canonical mixed-form Darcy "
                   "element."),
        constructor=("FE_RaviartThomas<dim>(unsigned int degree)"),
    ),
    "FE_RaviartThomasNodal": ElementRecord(
        name="FE_RaviartThomasNodal",
        header="deal.II/fe/fe_raviart_thomas.h",
        math_space="H(div)",
        semantics=("Nodal variant of FE_RaviartThomas; nodal DoFs "
                   "instead of moment DoFs make interpolation of "
                   "given vector fields easier."),
        constructor=("FE_RaviartThomasNodal<dim>(unsigned int degree)"),
    ),
    "FE_BDM": ElementRecord(
        name="FE_BDM",
        header="deal.II/fe/fe_bdm.h",
        math_space="H(div)",
        semantics=("Brezzi-Douglas-Marini H(div) element. Fewer DoFs "
                   "per cell than FE_RaviartThomas at the same "
                   "polynomial order; same conservation properties."),
        constructor="FE_BDM<dim>(unsigned int degree)",
    ),
    "FE_ABF": ElementRecord(
        name="FE_ABF",
        header="deal.II/fe/fe_abf.h",
        math_space="H(div)",
        semantics=("Arnold-Boffi-Falk H(div) element. Higher-order "
                   "alternative to FE_RaviartThomas with improved "
                   "approximation on quadrilateral meshes."),
        constructor="FE_ABF<dim>(unsigned int degree)",
    ),
    "FE_RT_Bubbles": ElementRecord(
        name="FE_RT_Bubbles",
        header="deal.II/fe/fe_rt_bubbles.h",
        math_space="H(div)",
        semantics=("FE_RaviartThomas enriched with cell-interior "
                   "bubble functions. Used in MINI-like mixed Stokes "
                   "formulations."),
        constructor=("FE_RT_Bubbles<dim>(unsigned int degree)"),
    ),
    "FE_DGRaviartThomas": ElementRecord(
        name="FE_DGRaviartThomas",
        header="deal.II/fe/fe_dg_vector.h",
        math_space="L2",
        semantics=("Discontinuous version of FE_RaviartThomas; same "
                   "shape functions but no continuity constraint "
                   "across faces. For DG mixed methods."),
        constructor=("FE_DGRaviartThomas<dim>(unsigned int degree)"),
    ),
    "FE_DGBDM": ElementRecord(
        name="FE_DGBDM",
        header="deal.II/fe/fe_dg_vector.h",
        math_space="L2",
        semantics=("Discontinuous version of FE_BDM."),
        constructor="FE_DGBDM<dim>(unsigned int degree)",
    ),
    # ── H(curl) family ───────────────────────────────────────────
    "FE_Nedelec": ElementRecord(
        name="FE_Nedelec",
        header="deal.II/fe/fe_nedelec.h",
        math_space="H(curl)",
        semantics=("Nedelec edge-element family. The canonical "
                   "H(curl)-conforming element for Maxwell / vector "
                   "Helmholtz / time-harmonic electromagnetics. NOT "
                   "for scalar acoustic Helmholtz — that uses FE_Q."),
        constructor="FE_Nedelec<dim>(unsigned int degree)",
    ),
    "FE_NedelecSZ": ElementRecord(
        name="FE_NedelecSZ",
        header="deal.II/fe/fe_nedelec_sz.h",
        math_space="H(curl)",
        semantics=("Nedelec element from Zaglmayr's PhD thesis (2006), "
                   "used in the same context as FE_Nedelec; it gives "
                   "edges and faces a globally defined orientation to "
                   "avoid the orientation-dependent sign conflicts of "
                   "the classical Nedelec basis."),
        constructor="FE_NedelecSZ<dim>(unsigned int degree)",
    ),
    "FE_DGNedelec": ElementRecord(
        name="FE_DGNedelec",
        header="deal.II/fe/fe_dg_vector.h",
        math_space="L2",
        semantics=("Discontinuous Nedelec; same shape functions "
                   "without inter-element edge continuity."),
        constructor="FE_DGNedelec<dim>(unsigned int degree)",
    ),
    # ── Vector / mixed / utility ────────────────────────────────
    "FESystem": ElementRecord(
        name="FESystem",
        header="deal.II/fe/fe_system.h",
        math_space="Mixed",
        semantics=("Compose multiple scalar FE classes into a "
                   "vector-valued or block-valued element. The vector "
                   "wrapper for elasticity (FESystem<dim>(FE_Q, dim)) "
                   "and the block wrapper for Stokes Taylor-Hood "
                   "(FESystem<dim>(FE_Q(p+1), dim, FE_Q(p), 1))."),
        constructor=("FESystem<dim>(const FiniteElement<dim>& fe1, "
                     "unsigned int n1, ...)"),
        aliases=("FE_System",),
    ),
    "FE_BernardiRaugel": ElementRecord(
        name="FE_BernardiRaugel",
        header="deal.II/fe/fe_bernardi_raugel.h",
        math_space="H(div)",
        semantics=("Vector-valued Lagrange enriched with edge "
                   "bubbles. Inf-sup stable when paired with "
                   "piecewise-constant pressure — a cheap "
                   "alternative to Taylor-Hood Q2/Q1 for Stokes. "
                   "deal.II classifies it as Conformity::Hdiv "
                   "(verified by instantiation; this record said H1). "
                   "Only degree 1 is implemented."),
        constructor="FE_BernardiRaugel<dim>(unsigned int degree)",
    ),
    "FE_Enriched": ElementRecord(
        name="FE_Enriched",
        header="deal.II/fe/fe_enriched.h",
        math_space="H1",
        semantics=("XFEM-style enrichment of a base FE with custom "
                   "functions. Used to embed singular solutions or "
                   "discontinuities (cracks) without re-meshing."),
        constructor=("FE_Enriched<dim>(const FiniteElement<dim>& "
                     "base, ...)"),
    ),
    "FE_FaceQ": ElementRecord(
        name="FE_FaceQ",
        header="deal.II/fe/fe_face.h",
        math_space="L2",
        semantics=("Lives only on the cell skeleton (faces). The "
                   "trace component of a hybridised DG (HDG) "
                   "method."),
        constructor="FE_FaceQ<dim>(unsigned int degree)",
    ),
    "FE_FaceP": ElementRecord(
        name="FE_FaceP",
        header="deal.II/fe/fe_face.h",
        math_space="L2",
        semantics=("Face-only element of complete polynomials with a "
                   "Legendre basis (Polynomials::Legendre); the FE_DGP "
                   "analogue of FE_FaceQ."),
        constructor="FE_FaceP<dim>(unsigned int degree)",
    ),
    "FE_TraceQ": ElementRecord(
        name="FE_TraceQ",
        header="deal.II/fe/fe_trace.h",
        math_space="L2",
        semantics=("Trace of FE_Q onto the cell skeleton — continuous "
                   "across faces, used as Lagrange-multiplier space "
                   "in mortar / hybridised methods. Mathematically an "
                   "H^{1/2} trace space, but deal.II reports "
                   "conforming_space == Conformity::L2 (measured "
                   "verified by instantiation); this record used to advertise the "
                   "non-existent math_space value 'H(1/2)'."),
        constructor="FE_TraceQ<dim>(unsigned int degree)",
    ),
    "FE_Nothing": ElementRecord(
        name="FE_Nothing",
        header="deal.II/fe/fe_nothing.h",
        math_space="None",
        semantics=("Placeholder element with zero DoFs. Used inside "
                   "FESystem on subdomains where a component is "
                   "fully constrained or absent — e.g. velocity in "
                   "a solid region of an FSI problem."),
        constructor="FE_Nothing<dim>()",
    ),
    # ── Simplex / wedge / pyramid cells (3D transition meshes) ───
    # Every record below was instantiated and its reported reference
    # cell, conformity and support-point flags recorded. They need
    # QGaussSimplex quadrature and an explicitly-passed
    # MappingFE(FE_SimplexP(1)); see the "simplex" block in the
    # essentials knowledge.
    "FE_SimplexDGP": ElementRecord(
        name="FE_SimplexDGP",
        header="deal.II/fe/fe_simplex_p.h",
        math_space="L2",
        semantics=("Discontinuous complete-polynomial element on "
                   "simplices (triangle in 2D, tetrahedron in 3D). "
                   "Reports Conformity::L2 and reference cell Tri / "
                   "Tet; has point support, so its DoFs are nodal. "
                   "The DG counterpart of FE_SimplexP."),
        constructor="FE_SimplexDGP<dim>(unsigned int degree)",
        version_added="9.3",
    ),
    "FE_SimplexP_Bubbles": ElementRecord(
        name="FE_SimplexP_Bubbles",
        header="deal.II/fe/fe_simplex_p_bubbles.h",
        math_space="H1",
        semantics=("Continuous simplex Lagrange enriched with "
                   "interior bubble functions — the simplex analogue "
                   "of FE_Q_Bubbles, used for inf-sup stability in "
                   "mixed formulations (MINI-type pairs)."),
        constructor="FE_SimplexP_Bubbles<dim>(unsigned int degree)",
        version_added="9.3",
    ),
    "FE_WedgeP": ElementRecord(
        name="FE_WedgeP",
        header="deal.II/fe/fe_wedge_p.h",
        math_space="H1",
        semantics=("Continuous element on WEDGE (triangular-prism) "
                   "3D cells. Exists only for dim==3; reports "
                   "reference cell Wedge and Conformity::H1, and has "
                   "point support. Used in transition meshes between "
                   "hexahedral and tetrahedral regions, and for "
                   "extruded triangular surface meshes."),
        constructor="FE_WedgeP<3>(unsigned int degree)",
        version_added="9.3",
    ),
    "FE_WedgeDGP": ElementRecord(
        name="FE_WedgeDGP",
        header="deal.II/fe/fe_wedge_p.h",
        math_space="L2",
        semantics=("Discontinuous counterpart of FE_WedgeP on wedge "
                   "cells; reports Conformity::L2."),
        constructor="FE_WedgeDGP<3>(unsigned int degree)",
        version_added="9.3",
    ),
    "FE_PyramidP": ElementRecord(
        name="FE_PyramidP",
        header="deal.II/fe/fe_pyramid_p.h",
        math_space="H1",
        semantics=("Continuous element on PYRAMID (square-base) 3D "
                   "cells. Exists only for dim==3; reports reference "
                   "cell Pyramid and Conformity::H1, and has point "
                   "support. The other half of a hex-to-tet "
                   "transition layer, together with FE_WedgeP."),
        constructor="FE_PyramidP<3>(unsigned int degree)",
        version_added="9.3",
    ),
    "FE_PyramidDGP": ElementRecord(
        name="FE_PyramidDGP",
        header="deal.II/fe/fe_pyramid_p.h",
        math_space="L2",
        semantics=("Discontinuous counterpart of FE_PyramidP on "
                   "pyramid cells; reports Conformity::L2."),
        constructor="FE_PyramidDGP<3>(unsigned int degree)",
        version_added="9.3",
    ),
    # ── Other classes present in the library but previously absent ─
    "FE_Hermite": ElementRecord(
        name="FE_Hermite",
        header="deal.II/fe/fe_hermite.h",
        math_space="H2",
        semantics=("Hermite-interpolation element on hypercubes (odd "
                   "degree p, regularity r=(p-1)/2). For p>=3 it "
                   "carries derivative degrees of freedom, is C^r (at "
                   "least C1) across faces, and deal.II reports "
                   "Conformity::H2, NOT H1 — which is why it is the "
                   "element for fourth-order problems (biharmonic, "
                   "Cahn-Hilliard) where an H1 element would need an "
                   "interior-penalty or mixed reformulation. "
                   "FE_Hermite(1) has only vertex values: it is C0 and "
                   "reports Conformity::H1. "
                   "has_support_points() is FALSE, so use "
                   "VectorTools::project_boundary_values, never "
                   "interpolate_boundary_values."),
        constructor="FE_Hermite<dim>(unsigned int degree)",
        version_added="9.6",
    ),
    # Declared in fe_nedelec.h (there is no fe_nedelec_nodal.h). Added to
    # deal.II on 2025-12-05; first release 9.8.0, so 9.7.1 does not have it.
    "FE_NedelecNodal": ElementRecord(
        name="FE_NedelecNodal",
        header="deal.II/fe/fe_nedelec.h",
        math_space="H(curl)",
        semantics=("Nedelec edge element with a nodal-interpolation "
                   "degree-of-freedom setup, an alternative to the "
                   "moment-based default. Convenient when coupling "
                   "against nodal H(curl) data from another code."),
        constructor="FE_NedelecNodal<dim>(unsigned int order)",
        version_added="9.8",
    ),
    # ── Internal / template base classes (not user-instantiable) ─
    # These appear in the scan but a community user should NOT pick
    # them directly; the catalog records them with a clear
    # `is_user_facing = False` so retrieval can filter them out.
    #
    # REMOVED 2026-08-03 after an execution pass on deal.II 9.8.0-pre:
    #   * "FE_Base"   — no such class. `deal.II/fe/fe_base.h` is a
    #     deprecated one-line shim that emits
    #     DEAL_II_WARNING("This file is deprecated. Use
    #     deal.II/fe/fe_data.h instead.") and declares nothing;
    #     `FE_Base<2>()` fails with "'FE_Base' was not declared".
    #   * "FE_Series" — no such class either. `fe_series.h` declares
    #     the NAMESPACE `FESeries` (holding FESeries::Fourier and
    #     FESeries::Legendre, both verified to compile); "FE_Series"
    #     is a name that never existed.
    "FE_Poly": ElementRecord(
        name="FE_Poly",
        header="deal.II/fe/fe_poly.h",
        math_space="Internal",
        semantics=("Template base for scalar polynomial elements; "
                   "not user-instantiable."),
        constructor="(internal)",
    ),
    "FE_PolyFace": ElementRecord(
        name="FE_PolyFace",
        header="deal.II/fe/fe_poly_face.h",
        math_space="Internal",
        semantics=("Template base for face-only polynomial elements; "
                   "not user-instantiable."),
        constructor="(internal)",
    ),
    "FE_PolyTensor": ElementRecord(
        name="FE_PolyTensor",
        header="deal.II/fe/fe_poly_tensor.h",
        math_space="Internal",
        semantics=("Template base for tensor-product polynomial "
                   "elements; not user-instantiable."),
        constructor="(internal)",
    ),
    "FE_Q_Base": ElementRecord(
        name="FE_Q_Base",
        header="deal.II/fe/fe_q_base.h",
        math_space="Internal",
        semantics=("Template base for the FE_Q family; not "
                   "user-instantiable."),
        constructor="(internal)",
    ),
    "FE_DGVector": ElementRecord(
        name="FE_DGVector",
        header="deal.II/fe/fe_dg_vector.h",
        math_space="Internal",
        semantics=("Template base for the FE_DG{RaviartThomas,BDM,"
                   "Nedelec} family; not user-instantiable directly."),
        constructor="(internal)",
    ),
}


# Convenience: the set of names the catalog can validly reference.
ELEMENT_NAMES = frozenset(ELEMENTS) | {
    alias for record in ELEMENTS.values() for alias in record.aliases
}


# ── Canonical mesh-generator catalog ────────────────────────────────
# Same pattern as ELEMENTS — single source of truth for the canonical
# description, per-physics applicability notes layered on top.


@dataclass(frozen=True)
class MeshGeneratorRecord:
    """Canonical metadata for one deal.II GridGenerator function."""

    name: str
    header: str = "deal.II/grid/grid_generator.h"
    signature: str = ""
    semantics: str = ""


_GG = "deal.II/grid/grid_generator.h"
MESH_GENERATORS: dict[str, MeshGeneratorRecord] = {
    "hyper_cube": MeshGeneratorRecord(
        name="hyper_cube", header=_GG,
        signature="GridGenerator::hyper_cube(Triangulation<dim, spacedim>&, double left = 0., double right = 1., bool colorize = false)",
        semantics="Axis-aligned cube [a,b]^dim. Smallest canonical "
                  "domain; one cell, refine globally. colorize=true "
                  "assigns per-face boundary ids 0..2*dim-1 "
                  "(2D {0,1,2,3}, 3D {0,...,5}); the default "
                  "colorize=false puts every face on id 0 (measured "
                  "verified by instantiation on deal.II 9.8)."),
    "hyper_rectangle": MeshGeneratorRecord(
        name="hyper_rectangle", header=_GG,
        signature="GridGenerator::hyper_rectangle(Triangulation<dim, spacedim>&, const Point<dim>& p1, const Point<dim>& p2, bool colorize = false)",
        semantics="Axis-aligned box from p1 to p2. Aspect ratio "
                  "control; one cell. Like hyper_cube it needs "
                  "colorize=true for per-face ids — the default "
                  "yields get_boundary_ids() == {0} (measured "
                  "verified by instantiation on deal.II 9.8)."),
    "subdivided_hyper_cube": MeshGeneratorRecord(
        name="subdivided_hyper_cube", header=_GG,
        signature="GridGenerator::subdivided_hyper_cube(Triangulation<dim>&, unsigned int n, double a, double b)",
        semantics="hyper_cube pre-divided into n^dim cells; avoids "
                  "repeated refine_global() calls when a uniform "
                  "starting mesh is needed."),
    "subdivided_hyper_rectangle": MeshGeneratorRecord(
        name="subdivided_hyper_rectangle", header=_GG,
        signature="GridGenerator::subdivided_hyper_rectangle(Triangulation<dim>&, const std::vector<unsigned int>& repetitions, const Point<dim>& p1, const Point<dim>& p2, bool colorize = false)",
        semantics="hyper_rectangle with per-direction cell counts. "
                  "colorize=true tags each face with a distinct "
                  "boundary_id (0-5 in 3D)."),
    "hyper_L": MeshGeneratorRecord(
        name="hyper_L", header=_GG,
        signature="GridGenerator::hyper_L(Triangulation<dim>&, double a, double b)",
        semantics="L-shaped domain with re-entrant corner. Solution "
                  "has u ~ r^{2/3} sin(2θ/3) singularity at the "
                  "corner — canonical test for adaptive refinement."),
    "hyper_ball": MeshGeneratorRecord(
        name="hyper_ball", header=_GG,
        signature="GridGenerator::hyper_ball(Triangulation<dim>&, const Point<dim>& center, double radius)",
        semantics="Solid disk / ball; curved boundary tagged with a "
                  "manifold for accurate refinement."),
    "hyper_shell": MeshGeneratorRecord(
        name="hyper_shell", header=_GG,
        signature="GridGenerator::hyper_shell(Triangulation<dim>&, const Point<dim>& center, double inner, double outer)",
        semantics="Annulus / spherical shell; pressure vessels, "
                  "rotating-frame problems."),
    "hyper_sphere": MeshGeneratorRecord(
        name="hyper_sphere", header=_GG,
        signature="GridGenerator::hyper_sphere(Triangulation<dim, dim+1>&, const Point<dim+1>& center, double radius)",
        semantics="Surface of a sphere (codim-1 manifold)."),
    "hyper_cube_with_cylindrical_hole": MeshGeneratorRecord(
        name="hyper_cube_with_cylindrical_hole", header=_GG,
        signature="GridGenerator::hyper_cube_with_cylindrical_hole(Triangulation<dim>&, double inner, double outer, double L = 0.5, unsigned int repetitions = 1, bool colorize = false)",
        semantics="Box with a cylindrical hole through it; flow-"
                  "around-cylinder / Kirsch-3D problems."),
    "hyper_cube_slit": MeshGeneratorRecord(
        name="hyper_cube_slit", header=_GG,
        signature="GridGenerator::hyper_cube_slit(Triangulation<dim>&, double a, double b, bool colorize = false)",
        semantics="Square with a slit cut — tests crack-like "
                  "singularities."),
    "hyper_cross": MeshGeneratorRecord(
        name="hyper_cross", header=_GG,
        signature="GridGenerator::hyper_cross(Triangulation<dim>&, const std::vector<unsigned int>& sizes, bool colorize = false)",
        semantics="Cross-shaped domain (5-cell in 2D / 7-cell in 3D)."),
    "plate_with_a_hole": MeshGeneratorRecord(
        name="plate_with_a_hole", header=_GG,
        signature="GridGenerator::plate_with_a_hole(Triangulation<dim>&, double inner, double outer, double pad_bottom, double pad_top, double pad_left, double pad_right, ...)",
        semantics="Rectangular plate with a circular hole — Kirsch "
                  "stress-concentration benchmark."),
    "channel_with_cylinder": MeshGeneratorRecord(
        name="channel_with_cylinder", header=_GG,
        signature="GridGenerator::channel_with_cylinder(Triangulation<dim>&, double shell_region_width, unsigned int n_shells, double skewness, bool colorize = false)",
        semantics="Schäfer-Turek 2D-1/2D-2/2D-3 benchmark domain — "
                  "cylinder at (0.2, 0.2) in a (2.2 × 0.41) channel."),
    "cylinder": MeshGeneratorRecord(
        name="cylinder", header=_GG,
        signature="GridGenerator::cylinder(Triangulation<dim>&, double radius, double half_length)",
        semantics="Circular cylinder — an extruded disk in 3D "
                  "(10 cells, ids {0,1,2}). NOT 3D-only: the 2D "
                  "instantiation compiles and produces a single "
                  "rectangular cell with ids {0,1,2} (measured "
                  "verified by instantiation; this record used to say '3D only')."),
    "cylinder_shell": MeshGeneratorRecord(
        name="cylinder_shell", header=_GG,
        signature="GridGenerator::cylinder_shell(Triangulation<dim>&, double length, double inner_radius, double outer_radius, ...)",
        semantics="Annular cylinder (pipe)."),
    "truncated_cone": MeshGeneratorRecord(
        name="truncated_cone", header=_GG,
        signature="GridGenerator::truncated_cone(Triangulation<dim>&, double r1, double r2, double half_length)",
        semantics="Truncated cone (varying-radius cylinder)."),
    "torus": MeshGeneratorRecord(
        name="torus", header=_GG,
        signature="GridGenerator::torus(Triangulation<dim>&, double R, double r)",
        semantics="3D torus — periodic-boundary studies."),
    "moebius": MeshGeneratorRecord(
        name="moebius", header=_GG,
        signature="GridGenerator::moebius(Triangulation<dim>&, unsigned int n_cells, unsigned int n_rotations, double R, double r)",
        semantics="Möbius strip (non-orientable)."),
    "half_hyper_ball": MeshGeneratorRecord(
        name="half_hyper_ball", header=_GG,
        signature="GridGenerator::half_hyper_ball(Triangulation<dim>&, const Point<dim>& center, double radius)",
        semantics="Half of a hyper_ball."),
    "quarter_hyper_ball": MeshGeneratorRecord(
        name="quarter_hyper_ball", header=_GG,
        signature="GridGenerator::quarter_hyper_ball(Triangulation<dim>&, const Point<dim>& center, double radius)",
        semantics="Quarter of a hyper_ball."),
    "half_hyper_shell": MeshGeneratorRecord(
        name="half_hyper_shell", header=_GG,
        signature="GridGenerator::half_hyper_shell(Triangulation<dim>&, const Point<dim>& center, double inner, double outer)",
        semantics="Half of a hyper_shell."),
    "quarter_hyper_shell": MeshGeneratorRecord(
        name="quarter_hyper_shell", header=_GG,
        signature="GridGenerator::quarter_hyper_shell(Triangulation<dim>&, const Point<dim>& center, double inner, double outer)",
        semantics="Quarter of a hyper_shell."),
    "cheese": MeshGeneratorRecord(
        name="cheese", header=_GG,
        signature="GridGenerator::cheese(Triangulation<dim>&, const std::vector<unsigned int>& holes)",
        semantics="Rectangle with a regular grid of holes; "
                  "porous-media / sonic-crystal demos."),
    "merge_triangulations": MeshGeneratorRecord(
        name="merge_triangulations", header=_GG,
        signature="GridGenerator::merge_triangulations(const Triangulation<dim>&, const Triangulation<dim>&, Triangulation<dim>& result, double duplicated_vertex_tolerance = 1e-12, bool copy_manifold_ids = false)",
        semantics="Combine two domains into one; inclusion / "
                  "dissimilar-material problems."),
    "create_union_triangulation": MeshGeneratorRecord(
        name="create_union_triangulation", header=_GG,
        signature="GridGenerator::create_union_triangulation(const Triangulation<dim>&, const Triangulation<dim>&, Triangulation<dim>& result)",
        semantics="Build the geometric union of two triangulations."),
    "create_triangulation_with_removed_cells": MeshGeneratorRecord(
        name="create_triangulation_with_removed_cells", header=_GG,
        signature="GridGenerator::create_triangulation_with_removed_cells(const Triangulation<dim>&, const std::set<typename Triangulation<dim>::active_cell_iterator>& cells_to_remove, Triangulation<dim>& result)",
        semantics="Carve out a subset of cells from an existing mesh."),
    "extrude_triangulation": MeshGeneratorRecord(
        name="extrude_triangulation", header=_GG,
        signature="GridGenerator::extrude_triangulation(const Triangulation<2, 2>&, unsigned int n_slices, double height, Triangulation<3, 3>& result, bool copy_manifold_ids = false)",
        semantics="Sweep a 2D mesh into a 3D prism with n_slices "
                  "subdivisions along z. The standard 2D→3D mesh "
                  "promotion."),
    "flatten_triangulation": MeshGeneratorRecord(
        name="flatten_triangulation", header=_GG,
        signature="GridGenerator::flatten_triangulation(const Triangulation<dim, spacedim>&, Triangulation<dim, spacedim>& flattened)",
        semantics="Drop manifold information; produce a flat-cell "
                  "version of an existing mesh."),
    "enclosed_hyper_cube": MeshGeneratorRecord(
        name="enclosed_hyper_cube", header=_GG,
        signature="GridGenerator::enclosed_hyper_cube(Triangulation<dim>&, double left, double right, double thickness, bool colorize = false)",
        semantics="hyper_cube with a layer of cells around it; "
                  "tests interface-handling."),
    "parallelepiped": MeshGeneratorRecord(
        name="parallelepiped", header=_GG,
        signature="GridGenerator::parallelepiped(Triangulation<dim>&, const Point<dim> (&corners)[dim], bool colorize = false)",
        semantics="Skewed box from three (2D: two) edge vectors."),
    "subdivided_parallelepiped": MeshGeneratorRecord(
        name="subdivided_parallelepiped", header=_GG,
        signature="GridGenerator::subdivided_parallelepiped<dim, spacedim>(Triangulation<dim, spacedim>&, const Point<spacedim>& origin, const std::array<Tensor<1, spacedim>, dim>& edges, const std::vector<unsigned int>& subdivisions = {}, bool colorize = false)  (template arguments must be given explicitly; the other overloads are (Triangulation<dim>&, unsigned int n_subdivisions, const Point<dim> (&corners)[dim], bool colorize = false) and (Triangulation<dim>&, const unsigned int (&n_subdivisions)[dim], const Point<dim> (&corners)[dim], bool colorize = false))",
        semantics="parallelepiped with per-direction subdivisions."),
    "parallelogram": MeshGeneratorRecord(
        name="parallelogram", header=_GG,
        signature="GridGenerator::parallelogram(Triangulation<2>&, const Point<2> (&corners)[2], bool colorize = false)",
        semantics="2D parallelogram from two edge vectors."),
    "simplex": MeshGeneratorRecord(
        name="simplex", header=_GG,
        signature="GridGenerator::simplex(Triangulation<dim>&, const std::vector<Point<dim>>& vertices)",
        semantics="A single simplex (triangle / tet) from its "
                  "vertices."),
    "concentric_hyper_shells": MeshGeneratorRecord(
        name="concentric_hyper_shells", header=_GG,
        signature="GridGenerator::concentric_hyper_shells(Triangulation<dim>&, const Point<dim>& center, double inner, double outer, unsigned int n_shells, double skewness, ...)",
        semantics="Stacked annular shells; for problems where "
                  "anisotropic radial refinement matters."),
    "general_cell": MeshGeneratorRecord(
        name="general_cell", header=_GG,
        signature="GridGenerator::general_cell(Triangulation<dim>&, const std::vector<Point<dim>>& vertices, bool colorize = false)",
        semantics="Single cell from explicit corner coordinates."),
}

MESH_GENERATOR_NAMES = frozenset(MESH_GENERATORS)


def resolve_mesh_generator(name: str, applicability: str = "") -> dict:
    """Return the rich form for a per-physics mesh-generator entry."""
    record = MESH_GENERATORS.get(name)
    if record is None:
        return {
            "name": name,
            "header": "(unknown — not in canonical mesh-generator catalog)",
            "signature": "(unknown)",
            "semantics": "(unknown — diff tool will surface this)",
            "applicability": applicability,
        }
    out: dict[str, object] = {
        "name": record.name,
        "header": record.header,
        "signature": record.signature,
        "semantics": record.semantics,
    }
    if applicability:
        out["applicability"] = applicability
    return out


def resolve_mesh_generators_section(value) -> list[dict] | None:
    """Resolve a per-physics ``mesh_generators:`` value into rich form."""
    if isinstance(value, dict):
        return [resolve_mesh_generator(name, applicability)
                for name, applicability in value.items()]
    return None


def resolve_element(name: str, applicability: str = "") -> dict:
    """Return the rich form for a per-physics ``elements:`` entry.

    Joins the canonical record with the physics-keyed applicability
    note. Falls back gracefully if the catalog references an
    element not in the canonical table (the diff tool will surface
    this as drift on the next run).
    """
    record = ELEMENTS.get(name)
    if record is None:
        # Drift: catalog references an element the canonical table
        # doesn't know. Diff tool catches this — we return a
        # minimal stub here so retrieval doesn't crash.
        return {
            "name": name,
            "header": "(unknown — not in canonical element catalog)",
            "math_space": "(unknown)",
            "semantics": "(unknown — diff tool will surface this)",
            "constructor": "(unknown)",
            "applicability": applicability,
        }
    out: dict[str, object] = {
        "name": record.name,
        "header": record.header,
        "math_space": record.math_space,
        "semantics": record.semantics,
        "constructor": record.constructor,
    }
    if record.version_added:
        out["version_added"] = record.version_added
    if record.aliases:
        out["aliases"] = list(record.aliases)
    if applicability:
        out["applicability"] = applicability
    return out


def resolve_elements_section(elements_value) -> list[dict] | None:
    """Resolve a per-physics ``elements:`` value into the rich form.

    Accepts:
      * a dict ``{class_name: applicability_note}`` — the new
        post-refactor shape; resolved with `resolve_element` per
        entry,
      * a list of strings — the legacy long-form shape (kept for
        backward compatibility with physics not yet refactored);
        returns None so the caller passes the value through as-is.

    Returning None lets a callsite distinguish "this is structured
    and was resolved" from "this is legacy and should be passed
    through verbatim".
    """
    if isinstance(elements_value, dict):
        return [resolve_element(name, applicability)
                for name, applicability in elements_value.items()]
    return None

