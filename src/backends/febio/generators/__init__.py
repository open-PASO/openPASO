"""FEBio generator registry — maps physics_variant -> generator function.

Mirrors the layout used by skfem, fenics, ngsolve etc.: one file per
FEBio physics module exposes a per-module ``GENERATORS`` + ``KNOWLEDGE``
dict, and this aggregator merges them into the top-level ``GENERATORS``
and ``KNOWLEDGE`` dicts that ``backend.py`` consumes.

When adding a new FEBio physics module (multiphasic, fluid, fluid-FSI,
etc.), drop a new file alongside this one and add it to the imports and
the merge loops below.
"""
from .linear_elasticity import GENERATORS as _le_gen, KNOWLEDGE as _le_kn
from .hyperelasticity import GENERATORS as _he_gen, KNOWLEDGE as _he_kn
from .biphasic import GENERATORS as _bi_gen, KNOWLEDGE as _bi_kn
from .heat import GENERATORS as _ht_gen, KNOWLEDGE as _ht_kn
from .multiphasic import GENERATORS as _mp_gen, KNOWLEDGE as _mp_kn
from .solute import GENERATORS as _sol_gen, KNOWLEDGE as _sol_kn
from .fluid import GENERATORS as _fl_gen, KNOWLEDGE as _fl_kn
from .fluid_fsi import GENERATORS as _fsi_gen, KNOWLEDGE as _fsi_kn
from .rigid_body import GENERATORS as _rb_gen, KNOWLEDGE as _rb_kn
from .viscoelasticity import GENERATORS as _ve_gen, KNOWLEDGE as _ve_kn
from .plasticity import GENERATORS as _pl_gen, KNOWLEDGE as _pl_kn
from .fiber_reinforced import GENERATORS as _fr_gen, KNOWLEDGE as _fr_kn
from .active_contraction import GENERATORS as _ac_gen, KNOWLEDGE as _ac_kn
from .biphasic_fsi import GENERATORS as _bfs_gen, KNOWLEDGE as _bfs_kn
from .polar_fluid import GENERATORS as _pf_gen, KNOWLEDGE as _pf_kn
from .damage import GENERATORS as _dm_gen, KNOWLEDGE as _dm_kn
from .growth_remodeling import GENERATORS as _gr_gen, KNOWLEDGE as _gr_kn
from .elasticity_mms import GENERATORS as _mms_gen, KNOWLEDGE as _mms_kn
from .deck_structure import (
    DECK_KNOWLEDGE as _deck_kn,
    FEBIO_MODULES,
    FEBIO_ANALYSIS_VALUES,
)


GENERATORS: dict[str, callable] = {}
for _g in (_le_gen, _he_gen, _bi_gen, _ht_gen,
           _mp_gen, _fl_gen, _fsi_gen, _rb_gen,
           _ve_gen, _pl_gen, _fr_gen, _ac_gen,
           _bfs_gen, _pf_gen, _dm_gen, _gr_gen,
           _mms_gen, _sol_gen):
    GENERATORS.update(_g)


KNOWLEDGE: dict[str, dict] = {}
for _k in (_le_kn, _he_kn, _bi_kn, _ht_kn,
           _mp_kn, _fl_kn, _fsi_kn, _rb_kn,
           _ve_kn, _pl_kn, _fr_kn, _ac_kn,
           _bfs_kn, _pf_kn, _dm_kn, _gr_kn,
           _mms_kn, _sol_kn):
    KNOWLEDGE.update(_k)


KNOWLEDGE["_general"] = {
    "description": "FEBio general capabilities and C++ embedding surface",
    # Executed-verified .feb authoring surface (deck_structure.py).
    # Kept as a sub-block of _general rather than a new top-level
    # KNOWLEDGE key so the catalog's physics-key count is unchanged
    # — this is reference material, not a physics row.
    "deck_authoring": _deck_kn,
    "adaptive_mesh_refinement": {
        "description": (
            "FEAMR module (FEBio::FEAMR library). Mesh adaptation is "
            "driven from a top-level <MeshAdaptor> section whose "
            "children are <mesh_adaptor> elements. EXECUTED on FEBio "
            "4.12.0.86045466d: a uniaxial-stretch hex8 cube refines "
            "under `hex_refine`, and the node count grows on each "
            "adaptor iteration. The XML shape below is the one that "
            "works — read it before anything else in this entry, "
            "because the obvious shape is silently ignored."
        ),
        # ---- EXECUTED. Load-bearing; read this first. -------------
        "required_xml_shape": (
            "The section is <MeshAdaptor> and its CHILDREN are "
            "<mesh_adaptor> elements. The type= and elem_set= "
            "attributes go on the CHILD, never on the section. "
            "Complete runnable block, place it after </LoadData>:\n"
            "  <MeshAdaptor>\n"
            "    <mesh_adaptor type=\"hex_refine\" "
            "elem_set=\"AllElems\">\n"
            "      <max_iters>1</max_iters>\n"
            "      <criterion type=\"element_selection\">\n"
            "        <element_list>1,2</element_list>\n"
            "      </criterion>\n"
            "    </mesh_adaptor>\n"
            "  </MeshAdaptor>\n"
            "REQUIRED: the <mesh_adaptor> child, its type= attribute, "
            "and a <criterion> child. OPTIONAL: elem_set= (defaults to "
            "the whole mesh; it accepts either an <ElementSet> name or "
            "an <Elements> block name — both executed) and <max_iters> "
            "(how many adaptor passes per converged time step)."),
        "how_to_tell_it_worked": (
            "Three things appear in the .log ONLY when the adaptor was "
            "actually registered and applied. Check for all three; the "
            "termination banner tells you nothing here. "
            "(1) a ` MESH ADAPTOR DATA` block in the model echo, "
            "(2) `=== Applying mesh adaptors: iteration N`, "
            "(3) a `Refinement info:` block with `Elements to refine`, "
            "`Facets to refine`, `Edges to refine`, followed by "
            "`Hanging nodes ............ : N`. The decisive check is "
            "the `Number of nodes` line: FEBio re-echoes the model "
            "after each adaptation, so a working refinement prints an "
            "increasing sequence of node counts. If it prints the "
            "original count once, nothing was refined. "
            "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"),
        "pitfalls": [
            (
                "[Syntax] Putting type= on the <MeshAdaptor> SECTION "
                "instead of on a <mesh_adaptor> CHILD is silently "
                "ignored. FEBioMeshAdaptorSection::Parse only looks at "
                "children named `mesh_adaptor`; anything else in the "
                "section is walked past without comment, so no adaptor "
                "is ever added to the model. "
                "WRONG: <MeshAdaptor type=\"hex_refine\" "
                "elem_set=\"AllElems\"><max_iters>1</max_iters>"
                "</MeshAdaptor>. "
                "RIGHT: <MeshAdaptor><mesh_adaptor type=\"hex_refine\" "
                "elem_set=\"AllElems\"><max_iters>1</max_iters>"
                "<criterion type=\"element_selection\">"
                "<element_list>1,2</element_list></criterion>"
                "</mesh_adaptor></MeshAdaptor>. "
                "Signal: NONE. The deck reads `...SUCCESS!`, the run "
                "ends in `N O R M A L   T E R M I N A T I O N` with "
                "exit 0, and the mesh is never touched. Even a "
                "deliberately non-existent adaptor type is not "
                "diagnosed in this form. Detect it by the absence of "
                "the three log markers listed under "
                "how_to_tell_it_worked. In the CORRECT form a bad type "
                "IS diagnosed: tag mesh_adaptor (line N) : `invalid "
                "value for attribute \"type\"`. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] A wrong-form <MeshAdaptor type=...> that "
                "contains ANY CHILD WHICH ITSELF HAS CHILDREN "
                "swallows every top-level section after it. Placed "
                "before <Material> "
                "and <Mesh>, it produces an EMPTY MODEL — zero nodes, "
                "zero materials — that still exits 0 with a normal "
                "termination banner. This is the worst silent failure "
                "found in the FEBio surface. "
                "WRONG: a <MeshAdaptor type=\"hex_refine\"> block "
                "holding a <criterion> child, placed anywhere above "
                "the sections you need. "
                "RIGHT: the <mesh_adaptor> child form, placed after "
                "</LoadData>. "
                "Signal: `Number of nodes ...... : 0` and `Number of "
                "materials ...... : 0` in the model echo, with "
                "`N O R M A L   T E R M I N A T I O N` and exit 0. "
                "ALWAYS check the echoed node and material counts "
                "against the deck you wrote. Executed across five "
                "shapes of the wrong form: a <criterion> child and an "
                "arbitrary made-up nested child BOTH destroy the "
                "model, while a leaf child such as "
                "<max_iters>1</max_iters>, an empty block and a "
                "self-closing tag are merely ignored — so the trigger "
                "is nesting depth, not the particular tag, and the two "
                "failure modes look different from the outside. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] `max_elems` is a parameter of the "
                "`erosion` adaptor -- that exact lower-case spelling "
                "is the registered type string, "
                "REGISTER_FECORE_CLASS(FEErosionAdaptor, \"erosion\") "
                "in FEAMR/FEAMR.cpp:49 -- and is "
                "not accepted by hex_refine. FEHexRefine registers "
                "max_elem_refine, max_value and criterion; its base "
                "FERefineMesh adds max_iters, max_elements, map_data, "
                "nnc, nsdim, transfer_method. Note max_elementS on the "
                "refiners versus max_elemS on erosion. "
                "WRONG: <mesh_adaptor type=\"hex_refine\">"
                "<max_elems>0</max_elems></mesh_adaptor>. "
                "RIGHT: <mesh_adaptor type=\"hex_refine\">"
                "<max_elements>0</max_elements></mesh_adaptor>. "
                "Signal: tag \"max_elems\" (line N) : `unrecognized "
                "tag` and `Reading file ...FAILED!` — but ONLY in the "
                "correct <mesh_adaptor> child form. In the wrong form "
                "above it is accepted in silence like everything else. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Syntax] The `relative error` and `min-max filter` "
                "criteria REQUIRE a nested <data> property naming "
                "another criterion, and that inner type must itself be "
                "registered — `element stress` is not, `stress` is. "
                "`max_variable` takes NO <data>: a <data> child is an "
                "`unrecognized tag`, and its only parameter is <dof>. "
                "WRONG: <criterion type=\"relative error\">"
                "<error>0.05</error></criterion>, or a <data "
                "type=\"element stress\"/> child, or "
                "<criterion type=\"max_variable\"><max>1e-12</max>"
                "</criterion>. "
                "RIGHT: <criterion type=\"relative error\">"
                "<error>0.05</error><data type=\"stress\"/></criterion>. "
                "Signal: for the missing property, `Component \"\" "
                "needs to have property \"data\" defined (line N)`; for "
                "the unregistered inner type, tag \"data\" (line N) : `"
                "invalid value for attribute \"type\"`; for the wrong "
                "parameter name, tag \"max\" (line N) : `unrecognized "
                "tag`. The simplest criterion that needs no <data> "
                "child at all is `element_selection` with an "
                "<element_list>. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d.)"
            ),
            (
                "[Numerical] hex_refine splits elements without "
                "splitting their neighbours, so every refinement "
                "introduces HANGING NODES which FEBio ties back with "
                "linear constraints. Refining on a stress criterion "
                "escalates fast: on a uniaxial-stretch cube a "
                "`relative error` criterion with a loose tolerance "
                "refined the whole domain twice in one time step and "
                "the node count grew by more than an order of "
                "magnitude, whereas an `element_selection` criterion "
                "naming two elements grew it by a few dozen. Bound the "
                "work with <max_iters> and <max_elements>, not with the "
                "criterion alone. "
                "Signal: `Hanging nodes ............ : N` and `Added "
                "linear constraints . : N` after each `Refinement "
                "info:` block; a run whose node count multiplies "
                "between adaptor iterations is about to become "
                "unaffordable. "
                "(Executed 2026-08-03, FEBio 4.12.0.86045466d, on a "
                "hex8 cube at two criterion settings.)"
            ),
        ],
        "note_on_the_material_below": (
            "Everything after this point is SOURCE-DERIVED (file walks "
            "of FEAMR/*.cpp), not executed. It is retained because it "
            "documents parameter semantics that no single run reveals — "
            "sort orders, enum spellings, default values, and known "
            "header/implementation disagreements. Treat the XML shapes "
            "in it as needing the <mesh_adaptor> child wrapper "
            "documented above: several of the older notes describe an "
            "<Adaptive> / <Step> placement that FEBio 4.12 does not "
            "use."),
        "mesh_adaptors": {
            "erosion":       "FEErosionAdaptor — remove elements failing a criterion",
            "hex_refine":    "FEHexRefine — uniform 3D hex split",
            "hex_refine2d":  "FEHexRefine2D — 2D analogue",
            "tet_refine":    "FETetRefine — uniform tet split",
            "mmg_remesh":    "FEMMGRemesh — anisotropic remesh via MMG library (requires MMG build)",
            "test_refine":   "FETestRefine — EXPERIMENTAL adaptor flagged FECORE_EXPERIMENTAL in source; do not use in production",
        },
        "mesh_adaptor_details": {
            "erosion (FEErosionAdaptor)": {
                "description": (
                    "Iteratively deactivates elements that match a "
                    "criterion. Operates IN-PLACE on the mesh — no "
                    "remeshing, just .setInactive() on selected "
                    "elements + optional surface/topology cleanup. "
                    "Source: FEAMR/FEErosionAdaptor.cpp"),
                "parameters": {
                    "max_iters": (
                        "int, default -1 = UNLIMITED. Apply() only "
                        "self-terminates via 'Max iterations reached' "
                        "when max_iters >= 0 and iteration crosses "
                        "it. Default behaviour: erosion never stops "
                        "voluntarily — relies on the outer time-"
                        "step / load controller to bound the loop."),
                    "max_elems": (
                        "int, default 0 = NO CAP. Caps how many "
                        "elements get deactivated per Apply() call. "
                        "When 0, the entire criterion-selection is "
                        "processed in one call."),
                    "sort": (
                        "int enum {0=none (default), 1=largest first "
                        "(DECREASING), 2=smallest first (INCREASING)}. "
                        "Only consulted when max_elems > 0 — with "
                        "max_elems == 0 the sort value is IGNORED, "
                        "no log line. Sorts by the criterion's "
                        "per-element scalar. Header-comment drift: "
                        "FEErosionAdaptor.h:58 says '1 = smallest to "
                        "largest, 2 = largest to smallest' — the "
                        "OPPOSITE of what the .cpp dispatch "
                        "(FEErosionAdaptor.cpp:89-100) implements "
                        "(1 -> SORT_DECREASING, 2 -> SORT_INCREASING). "
                        "Trust the .cpp; the header doc is wrong."),
                    "remove_islands": (
                        "bool, default false. When true, calls "
                        "RemoveIslands(FEMeshTopo) after deactivating "
                        "the selection — flood-fills active elements "
                        "and removes any disconnected component whose "
                        "ALL nodes have BC == DOF_OPEN on DOFs 0/1/2."),
                    "erode_surfaces": (
                        "enum, default 'yes' (ERODE). Values: "
                        "{'no' (DONT_ERODE), 'yes' (ERODE), 'grow' "
                        "(GROW), 'reconstruct' (RECONSTRUCT)}. "
                        "Registered via setEnums(\"no\\0yes\\0grow\\0"
                        "reconstruct\\0\")."),
                    "criterion": (
                        "REQUIRED <criterion> child property "
                        "(FEMeshAdaptorCriterion). Without it, "
                        "Apply() returns false silently — NO log "
                        "line, no error."),
                },
                "erode_surfaces_semantics": {
                    "no": (
                        "DONT_ERODE — leave surfaces untouched; faces "
                        "attached to deactivated elements remain "
                        "active. Useful when surfaces represent "
                        "external loading patches that should persist."),
                    "yes": (
                        "ERODE (default) — ErodeSurfaces(): for each "
                        "surface, deactivate any face whose owning "
                        "element became inactive. Calls surf.Init() "
                        "only when faces were actually eroded."),
                    "grow": (
                        "GROW — GrowErodedSurfaces(): same as 'yes' "
                        "PLUS rebuild boundary as the eroded surface "
                        "grows inward. New faces are INVERTED "
                        "(node[l] = node[nf-1-l]) to keep normals "
                        "outward-pointing. Materializes new FE_TRI3G3 "
                        "/ FE_QUAD4G4 surface elements."),
                    "reconstruct": (
                        "RECONSTRUCT — discard existing surface faces, "
                        "rebuild from FEMesh::DomainBoundary(domainList) "
                        "where the surface name matches a registered "
                        "FEDomainList name. Surfaces NOT generated "
                        "from part lists with matching names are "
                        "silently skipped — no log, no error."),
                },
            },
            "hex_refine (FEHexRefine)": {
                "description": (
                    "Subdivides each selected HEX8 element into 8 "
                    "smaller HEX8 sub-elements (uniform 1-to-8 split) "
                    "via FEAMR/FEHexRefine.cpp. Inherits from "
                    "FERefineMesh. Hanging nodes at non-conforming "
                    "interfaces get FELinearConstraint MPCs to tie "
                    "them to parent-edge endpoints."),
                "parameters": {
                    "max_elem_refine": (
                        "int, default 0 = UNLIMITED. Caps how many "
                        "elements get refined per RefineMesh() call. "
                        "When 0, every element matching the criterion "
                        "threshold is refined. When >0 and the "
                        "criterion-selected count exceeds the cap, "
                        "BuildSplitLists trims the list "
                        "deterministically (preserves first-N-matched)."),
                    "max_value": (
                        "double, default 0.0. Threshold compared "
                        "STRICTLY (selection[i].m_elemValue > "
                        "m_maxValue). Default 0.0 means any positive "
                        "criterion value qualifies; with a value-"
                        "returning criterion (e.g. 'min-max filter' in "
                        "clamp=false mode or 'mean-ratio') the user "
                        "sets this to the actual quality threshold. "
                        "Negative values are accepted (refines even "
                        "more aggressively)."),
                    "criterion": (
                        "REQUIRED FEMeshAdaptorCriterion property in "
                        "any realistic use, but missing-criterion is "
                        "NOT an error: when m_criterion == nullptr "
                        "(BuildSplitLists line 155-159: 'just do'em "
                        "all'), EVERY element gets flagged for "
                        "refinement. Omitting <criterion> on a large "
                        "mesh refines globally to 8N elements in one "
                        "step — likely OOM."),
                },
                "init_requirements": (
                    "Init() checks mesh.IsType(ET_HEX8). Mixed meshes "
                    "or non-HEX8 element types (HEX20, HEX27, TET4, "
                    "etc.) fail with feLogError('Cannot apply hex "
                    "refinement: Mesh is not a HEX8 mesh.') and return "
                    "false. For mixed meshes use FETetRefine or "
                    "FEHexRefine2D in combination, or restrict the "
                    "adaptor to a HEX8-only sub-domain via "
                    "FEElementSet."),
                "internal_algorithm_notes": (
                    "RefineMesh(): BuildSplitLists → UpdateNewNodes → "
                    "FindHangingNodes (assigns FELinearConstraint) → "
                    "BuildNewDomains → recreate element sets / node "
                    "sets / surfaces. Elements that already have "
                    "hanging nodes (FENode::HANGING flag) are "
                    "REJECTED from refinement (logged: 'Elements "
                    "rejected: N'). Coincident-node detection in "
                    "findNodeInMesh uses HARDCODED tolerance 1e-12 — "
                    "meshes scaled to >1e6 length units may miss "
                    "coincident-node merges. Two error sites use the "
                    "generic exception 'Error in FEHexRefine!' "
                    "(lines 108, 127) — no diagnostic detail."),
            },
            "hex_refine2d (FEHexRefine2D)": {
                "description": (
                    "Anisotropic 2D-in-plane variant of hex_refine. "
                    "Splits each selected HEX8 element into 4 smaller "
                    "HEX8 sub-elements (1-to-4 split, NOT 1-to-8 like "
                    "the 3D hex_refine), refining ONLY the in-plane "
                    "XY directions while leaving the Z direction "
                    "unchanged. Used to refine extruded 2D-like "
                    "meshes (e.g. layered shells, plane-stress slabs "
                    "represented as a single HEX8 layer) without "
                    "exploding the through-thickness DOF count. "
                    "Faces with normals near ±Z (|n.z| > 0.999) are "
                    "split in 4; faces with normals NOT perpendicular "
                    "to XY are split in 2. Source: "
                    "FEAMR/FEHexRefine2D.cpp. Registered as "
                    "FECoreClass 'hex_refine2d' in FEAMR/FEAMR.cpp:51."),
                "parameters": {
                    "max_elem_refine": (
                        "int, default 0 = UNLIMITED. SAME parameter "
                        "name as hex_refine, but with a real upstream "
                        "bug in the cap logic — see Signal below."),
                    "max_value": (
                        "double, default 0.0. Threshold compared "
                        "STRICTLY (selection[i].m_elemValue > "
                        "m_maxValue). Same semantics as hex_refine."),
                    "criterion": (
                        "REQUIRED FEMeshAdaptorCriterion property; "
                        "when omitted (nullptr), every HEX8 element "
                        "is flagged for refinement (line 145: "
                        "'just do'em all'). Same fall-back as "
                        "hex_refine."),
                },
                "init_requirements": (
                    "Init() checks mesh.IsType(ET_HEX8). Mixed "
                    "meshes or non-HEX8 (HEX20, HEX27, TET, etc.) "
                    "fail with feLogError('Cannot apply hex "
                    "refinement: Mesh is not a HEX8 mesh.') — note "
                    "the error text is shared with the 3D hex_refine, "
                    "so it does not disambiguate which adaptor "
                    "rejected the mesh."),
                "Signal": (
                    "[Validation] FEHexRefine2D::BuildSplitLists "
                    "(lines 185-201) has a REAL UPSTREAM BUG in "
                    "the max_elem_refine cap-enforcement loop: "
                    "  if ((m_elemRefine > 0) && "
                    "      (m_splitElems > m_elemRefine)) { "
                    "    m_splitElems = 0; "
                    "    for (int i = 0; i < m_elemList.size(); ++i){ "
                    "      if (m_elemList[i] == 0) {                 "
                    "// ← BUG: should be != -1 or >= 0 "
                    "        ... "
                    "      } "
                    "    } "
                    "    assert(m_splitElems == m_elemRefine); "
                    "  } "
                    "Elements eligible for refinement are marked "
                    "with value 1 (line 138: m_elemList[lid] = 1; "
                    "line 145: m_elemList.assign(NEL, 1) when no "
                    "criterion). They are NEVER marked with 0. The "
                    "cap-loop checks for 0 and so NEVER executes its "
                    "body — m_splitElems stays at 0. Compare with "
                    "the 3D FEHexRefine.cpp:205 which uses "
                    "'if (m_elemList[i] >= 0)' — the correct "
                    "condition. User-visible failure: when a user "
                    "sets max_elem_refine=N to cap a step and the "
                    "criterion flags MORE than N elements, the "
                    "adaptor silently performs ZERO refinements in "
                    "release builds (the post-loop "
                    "'assert(m_splitElems == m_elemRefine)' aborts "
                    "in debug builds with NDEBUG unset). Log line "
                    "'\\t  Elements to refine: 0' is the only "
                    "signal. Workaround until upstream fixes: leave "
                    "max_elem_refine at the default 0 (unlimited) "
                    "and pre-restrict the candidate set via a "
                    "tight criterion. "
                    "[Mesh] hex_refine2d splits HEX8 cells into 4 "
                    "smaller HEX8 cells (in-plane only), so it "
                    "expects a SINGLE-LAYER through-thickness mesh "
                    "with face normals aligned to ±Z. The "
                    "XY-vs-non-XY detection uses fabs(n.z) > 0.999 "
                    "(line 219) — meshes whose 'layer' axis is "
                    "rotated more than ~2.6° off the global Z axis "
                    "fall through to the 'split in 2' branch and "
                    "the refinement does not match the user's "
                    "mental model. Pre-align meshes to Z before "
                    "applying hex_refine2d. "
                    "(File walk FEAMR/FEHexRefine2D.cpp 2026-06-03.)"
                ),
            },
            "mmg_remesh (FEMMGRemesh)": {
                "description": (
                    "Adaptive TETRAHEDRAL remesher wrapping the "
                    "external MMG3D library "
                    "(https://www.mmgtools.org/). Inherits from "
                    "FERefineMesh. Registered as 'mmg_remesh' "
                    "(FEAMR/FEAMR.cpp:53). For each refinement "
                    "step: builds an MMG metric from the user-"
                    "supplied criterion + optional size_function, "
                    "calls MMG3D_mmg3dlib to remesh, then "
                    "transfers nodal/integration-point data from "
                    "the old to the new mesh via a configurable "
                    "interpolator (TRANSFER_SHAPE = "
                    "FEMeshShapeInterpolator, TRANSFER_MLQ "
                    "(default) = FELeastSquaresInterpolator)."),
                "parameters": {
                    "min_element_size": (
                        "double, default 0.0. MMG3D_DPARAM_hmin — "
                        "minimum allowed element edge length. "
                        "0.0 lets MMG choose; set to a positive "
                        "value to prevent over-refinement."),
                    "hausdorff": (
                        "double, default 0.01. MMG3D_DPARAM_hausd "
                        "— maximum chordal deviation from curved "
                        "boundaries during remeshing. Small "
                        "values preserve curvature; large values "
                        "let MMG flatten."),
                    "gradation": (
                        "double, default 1.3. MMG3D_DPARAM_hgrad "
                        "— maximum allowed size-ratio between "
                        "adjacent edges. 1.3 is MMG's default; "
                        "values > 2.0 produce very anisotropic "
                        "meshes."),
                    "relative_size": (
                        "bool, default true. Controls whether "
                        "criterion values are interpreted as "
                        "absolute target sizes or relative size "
                        "multipliers."),
                    "mesh_coarsen": (
                        "bool, default false. Allow MMG to "
                        "REMOVE elements (coarsen) where the "
                        "criterion permits, not just refine."),
                    "normalize_data": (
                        "bool, default false. Normalize "
                        "criterion values to [0, 1] before "
                        "passing to MMG."),
                },
                "properties": {
                    "criterion": (
                        "REQUIRED FEMeshAdaptorCriterion. "
                        "Without it, build_mmg_mesh asserts "
                        "(release no-op) and returns false → "
                        "RefineMesh fails silently."),
                    "size_function": (
                        "OPTIONAL FEFunction1D (registration "
                        "index 0). Maps criterion value → target "
                        "edge size. When omitted, MMG uses the "
                        "raw criterion value."),
                },
                "init_requirements": (
                    "Init() only accepts mesh.IsType(ET_TET4) — "
                    "for non-tet meshes (HEX8, prisms, mixed) it "
                    "returns false without a message of its own, "
                    "and the run then stops at initialisation with "
                    "the generic 'Mesh adaptor 1 (<unnamed>) failed "
                    "to initialize' (rc 1, measured on a HEX8 deck "
                    "2026-10-01), which does not say why. "
                    "Compare to FEHexRefine.cpp which DOES "
                    "feLogError on the wrong element type. Users "
                    "with a hex mesh wanting MMG-style remeshing "
                    "must first convert to TET4 (or use a "
                    "different refiner)."),
                "Signal": (
                    "[Integration]+[Mesh] Three real user-facing "
                    "edges on mmg_remesh: "
                    "(1) Build-time gate: the ENTIRE FEMMGRemesh "
                    "implementation is wrapped in #ifdef HAS_MMG. "
                    "If FEBio was compiled without MMG support, "
                    "RefineMesh() returns false from line 147 "
                    "(`#else return false; #endif`) and the "
                    "constructor leaves the `mmg` member pointer "
                    "uninitialized — NOT nullptr — so any "
                    "downstream code path that touches `mmg->` "
                    "without first checking HAS_MMG would segv. "
                    "No log message tells the user 'FEBio "
                    "without MMG cannot do mmg_remesh' — they "
                    "see silent zero-refinement. Check whether "
                    "MMG was enabled at build time in the FEBio "
                    "CMake config (`USE_MMG:BOOL=ON` in "
                    "CMakeCache.txt); `febio4 -info` prints the "
                    "compile date and the version but no list of "
                    "build features, so grepping it for MMG finds "
                    "nothing. "
                    "(2) Mesh-type gate: Init() returns false on "
                    "non-TET4 meshes without its own feLogError, "
                    "and FEBio then aborts: an mmg_remesh set up "
                    "like hex_refine on a hex mesh reads SUCCESS "
                    "and then stops with "
                    "'Mesh adaptor 1 (<unnamed>) failed to "
                    "initialize' and 'Model initialization "
                    "failed', exit 1. On a TET4 mesh a remesh that "
                    "does nothing (e.g. a build without MMG) logs "
                    "'-- Starting Mesh refinement.' followed by "
                    "'Nothing to refine.', with no '-- Mesh "
                    "refinement completed.' and no 'Mesh "
                    "Statistics' block; check the log for that "
                    "after the first refinement step. 'Elements "
                    "to refine' is printed only by hex_refine and "
                    "hex_refine2d, and mmg_remesh prints no 'MMG "
                    "remesh' progress line. "
                    "(3) Default-data interpolator (m_transferMethod "
                    "= TRANSFER_MLQ at line 85) uses "
                    "FELeastSquaresInterpolator under the hood — "
                    "which #160 flagged as having a commented-"
                    "out KDTree.build (brute-force "
                    "findNeirestNeighbors fallback) and "
                    "non-fatal assert(M > 4) (potentially "
                    "singular MLS matrix in release builds). For "
                    "large remeshing steps with many nodes, the "
                    "MLQ path is O(N²) per remesh; users wanting "
                    "shape-function-based interpolation can hand-"
                    "edit the source to set TRANSFER_SHAPE — but "
                    "that path isn't a user-exposed parameter "
                    "yet (m_transferMethod is hard-coded in the "
                    "ctor). "
                    "(File walk FEAMR/FEMMGRemesh.cpp + "
                    "FEMMGRemesh.h 2026-06-03.)"
                ),
            },
        },
        "adaptor_criteria": {
            "max_variable":      "FEVariableCriterion — threshold on a field variable",
            "element_selection": (
                "FEElementSelectionCriterion — UNDERSCORE tag "
                "name. Required children: <element_list> (vector "
                "of element IDs, 1-based, GetID() not "
                "GetLocalID()) and <value> (double, default 1.0 "
                "= refinement scale assigned to each matched "
                "element). Iterates over active elements only "
                "(el.isActive() gate; inactive/erased elements "
                "silently skipped). LOOKUP IS O(N·M) per call: "
                "for N total elements and M IDs in the list, a "
                "linear scan runs N×M times — source comment "
                "(FEAMR/FEElementSelectionCriterion.cpp:52) "
                "explicitly says 'TODO: This is really slow. "
                "Need to speed this up!'. Acceptable for "
                "M ≤ ~100 on small meshes; becomes the dominant "
                "cost for large meshes with long lists. Source: "
                "FEAMR/FEElementSelectionCriterion.cpp"),
            "math":              "FEScaleAdaptorCriterion — math-expression scaling",
            "min-max filter": (
                "FEMinMaxFilterAdaptorCriterion — SPACED-AND-HYPHEN "
                "tag name. Parameters: <min> (double, default -1e37), "
                "<max> (double, default +1e37), <clamp> (bool, default "
                "true). Required child: <data> (FEMeshAdaptorCriterion "
                "property). Behavior depends on <clamp>: when true "
                "(default), values are CLAMPED to [min,max] and ALL "
                "elements remain in the selection; when false, "
                "elements with values outside [min,max] are REJECTED "
                "(GetElementValue returns false). Default range "
                "[-1e37,+1e37] makes the criterion a silent pass-"
                "through regardless of clamp mode — users must set "
                "min/max explicitly. A missing <data> property is "
                "NOT silent: the input stops with 'Component \"\" "
                "needs to have property \"data\" defined (line N)', "
                "rc 1 (measured 2026-10-01). "
                "Source: FEAMR/FEFilterAdaptorCriterion.cpp"),
            "relative error": (
                "FEDomainErrorCriterion — SPACED tag name. Required "
                "child params: <error> (double, target relative-error "
                "tolerance) and <data> (FEMeshAdaptorCriterion property "
                "providing per-Gauss-point scalar via "
                "GetMaterialPointValue). Error formula per element: "
                "max_j |s_j - s_hat_j| / (smax - smin), where s_hat is "
                "the recovered nodal projection evaluated at Gauss "
                "point j. Returns size-scale s = error/max_err when "
                "max_err>error (shrink), else s=1 (keep). Source: "
                "FEAMR/FEDomainErrorCriterion.cpp"),
            "element data": (
                "FEElementDataCriterion — SPACED tag name on the "
                "outer <criterion type=\"element data\"> wrapper. "
                "Required child uses UNDERSCORE form: "
                "<element_data>...</element_data>. The string value "
                "is dispatched via fecore_new<FELogElemData>(name, "
                "fem) to look up a registered log-element-data "
                "variant; an unknown name returns nullptr → Init() "
                "returns false → model initialisation aborts with "
                "'The property \"criterion\" failed to initialize or "
                "was not defined.' and 'Mesh adaptor N (...) failed "
                "to initialize' (exit 1), even though the reader "
                "reported SUCCESS. Vocabulary matches the "
                "<plot> tag set (use the same registered "
                "FELogElemData names you see in .feb plot output). "
                "Source: FEAMR/FEElementDataCriterion.cpp"),
            "tet-quality":       "FETetQualityCriterion",
            "mean-ratio": (
                "FEMeanRatioQualityCriterion — HYPHEN tag name. "
                "Required child: <min_quality> (double, default 1.0 "
                "= refines everything; mean-ratio metric ∈ [0,1] "
                "with 1=perfect equilateral). Per-node formula: "
                "MR = 3·det(J)^(2/3) / (|e0|² + |e1|² + |e2|²), "
                "returns min over nodes. Element-shape gate "
                "(GetElementValue): ONLY ET_TET4 / ET_HEX8 / "
                "ET_PENTA6 — linear quads, higher-order elements "
                "(ET_TET10, ET_HEX20, ...), and all 2D shells/beams "
                "return false silently. Source: "
                "FEAMR/FEElementQualityCriterion.cpp:35-127"),
            "scaled Jacobian": (
                "FEScaledJacobianQualityCriterion — SPACED-AND-"
                "CAMELCASE tag name (outer); ADD_PARAMETER uses "
                "UNDERSCORE \"min_quality\" inner form (default "
                "1.0). Per-node formula: SJ = det(J) / "
                "(|e0|·|e1|·|e2|), returns min over nodes (range "
                "[-1, 1] with 1=perfect, near 0=bad, negative "
                "=inverted element). Same element-shape gate as "
                "mean-ratio: ONLY TET4/HEX8/PENTA6. Unsupported "
                "shapes return false silently. Source: "
                "FEAMR/FEElementQualityCriterion.cpp:131-194"),
        },
        "plot_variables": {
            "tet-quality":     ("FEPlotTetQuality — silently writes 0.0 for "
                                "non-ET_TET4 elements on mixed meshes "
                                "(FEAMR/FEAMRPlot.cpp:44 explicit else branch); "
                                "tag name warns but no log message"),
            "mean-ratio":      "FEPlotMeanRatio — applies to ALL element shapes",
            "scaled-Jacobian": "FEPlotScaledJacobian — HYPHEN-AND-CAMELCASE; applies to ALL element shapes",
        },
        "Signal": (
            "[Input] Some FEAMR .feb tag names contain SPACES and "
            "hyphens that look non-XML-like — they ARE the canonical "
            "strings the FECoreKernel registers and parses. Examples: "
            "'min-max filter', 'relative error', 'element data', "
            "'scaled Jacobian', 'scaled-Jacobian' (plot). Replacing "
            "spaces with underscores ('min_max_filter') silently fails "
            "to match in the FECoreKernel registry — FEBio reports the "
            "unknown tag at parse time. Also note test_refine has "
            "FECORE_EXPERIMENTAL flag in the source — production use "
            "is unsupported. Plus a quiet behavior in 'relative error' "
            "(FEDomainErrorCriterion::GetElementSelection): if the "
            "data-field range across the element set is below 1e-12 "
            "(fabs(smin - smax) < 1e-12 — e.g. an undisturbed step-0 "
            "state with zero stress everywhere), the function returns "
            "an EMPTY FEMeshAdaptorSelection and refinement is silently "
            "a no-op. Users probing the criterion before the first "
            "solve see no refinement and no warning. Two more quirks "
            "in the 'mean-ratio' and 'scaled Jacobian' criteria "
            "(FEElementQualityCriterion.cpp): (a) BOTH default "
            "<min_quality> to 1.0 — since mean-ratio ∈ [0,1] and "
            "scaled-Jacobian ∈ [-1,1] both peak at 1, users who omit "
            "<min_quality> trigger refinement on every non-perfect "
            "element (essentially the entire mesh on the first call). "
            "(b) BOTH criterion classes silently SKIP elements whose "
            "Shape() is not ET_TET4 / ET_HEX8 / ET_PENTA6 — "
            "GetElementValue returns false. Higher-order solids "
            "(ET_TET10, ET_HEX20), 2D shells, beams, etc. are "
            "invisible to these criteria. Mixed meshes get partial "
            "coverage with no warning. Five FEErosionAdaptor "
            "edges users routinely hit on damage / erosion runs: "
            "(c) <erode_surfaces> is a 4-VALUE enum {'no', 'yes', "
            "'grow', 'reconstruct'} registered via setEnums("
            "\"no\\0yes\\0grow\\0reconstruct\\0\"). Any other "
            "spelling (e.g. 'true', 'erode', 'remove') silently "
            "FAILS the enum gate at XML parse time — the parameter "
            "stays at its default ERODE. (d) Default <max_iters> "
            "is -1 (UNLIMITED) — Apply() only emits 'Max "
            "iterations reached' when max_iters >= 0; with the "
            "default the loop NEVER self-terminates and relies on "
            "the outer load step. (e) <sort> is silently IGNORED "
            "when <max_elems> == 0 (default). Users setting "
            "sort=1 expecting biggest-failures-first behaviour "
            "must ALSO set max_elems > 0. (f) <criterion> is "
            "REQUIRED — omitting it makes Apply() return false "
            "silently (no feLog, no error). The mesh adaptor is "
            "effectively a no-op with no diagnostic. (g) "
            "RemoveIslands has a HARDCODED 'TODO: mechanics "
            "only!' check — it tests nj.get_bc(0/1/2) != DOF_OPEN. "
            "For biphasic / multiphasic / thermal / scalar-only "
            "problems where mechanics DOFs are open by default, "
            "EVERY component looks isolated → remove_islands=true "
            "deletes the entire mesh. Source comment line "
            "FEAMR/FEErosionAdaptor.cpp:206. Three "
            "FEMinMaxFilterAdaptorCriterion edges users routinely "
            "hit on the 'min-max filter' criterion: "
            "(h) Default <min> = -1e37 and <max> = +1e37 — with "
            "the defaults the filter is a SILENT PASS-THROUGH "
            "regardless of <clamp>. Users must set both bounds "
            "explicitly; forgetting them yields a refinement "
            "selection equal to the input (no filtering happens). "
            "(i) <clamp> default is TRUE — meaning out-of-range "
            "values are clamped to the bounds and the element "
            "stays in the selection. Users who want to FILTER "
            "OUT out-of-range elements (the natural "
            "interpretation of 'filter') must explicitly set "
            "<clamp>false</clamp>; otherwise the criterion is a "
            "soft-clamp that keeps every element. "
            "(j) Missing <data> child property silently returns "
            "false (no log, no error) — same pattern as "
            "FEElementDataCriterion's missing-data and "
            "FEErosionAdaptor's missing-criterion (edge f above). "
            "Three FECORE criterion / adaptor classes share this "
            "silent-no-op fault pattern, and they all consult the "
            "same nullable property field. "
            "Four FEHexRefine edges users routinely hit on "
            "hex AMR runs: "
            "(k) Init() REJECTS non-HEX8 meshes with "
            "feLogError('Cannot apply hex refinement: Mesh is "
            "not a HEX8 mesh.'). Mixed meshes (HEX8 + TET4), "
            "higher-order hex (HEX20, HEX27), and pure-tet/"
            "wedge/pyramid all fail at Init time. Workaround: "
            "restrict to a HEX8 element set via FEElementSet, "
            "or pair with FETetRefine for mixed meshes. "
            "(l) Missing <criterion> is NOT an error — when "
            "m_criterion == nullptr, BuildSplitLists falls "
            "through to 'just do'em all' (line 155-159) and "
            "flags EVERY element for refinement. On a large "
            "mesh this means 8× node count in one step — likely "
            "OOM or extreme runtime. Always set <criterion>. "
            "(m) Default <max_value> = 0.0 + STRICTLY-GREATER "
            "comparison (line 147: 'm_elemValue > m_maxValue'). "
            "Criterion values of exactly 0 are NOT refined. "
            "With a quality-criterion that returns 0 for "
            "'don't-refine', the default works; for thresholds "
            "(e.g. relative-error 0.05) the user MUST set "
            "<max_value> to the actual cutoff. "
            "(n) Coincident-node detection in findNodeInMesh "
            "uses HARDCODED tolerance 1e-12 (line 267 default "
            "arg). Meshes scaled to length units > 1e6 may "
            "miss coincident-node merges, leading to "
            "duplicated mid-edge / face / element-center nodes. "
            "Symptom: 'mesh refinement produced N more nodes "
            "than expected' with no error. Plus two generic "
            "std::runtime_error('Error in FEHexRefine!') sites "
            "(lines 108, 127) with no diagnostic detail — when "
            "they fire, the user has no information about "
            "which sub-step (element-set recreation / surface "
            "update) failed. "
            "(File walks "
            "FEAMR/FEAMR.cpp 2026-06-02, "
            "FEAMR/FEDomainErrorCriterion.cpp 2026-06-03, "
            "FEAMR/FEElementQualityCriterion.cpp 2026-06-03, "
            "FEAMR/FEErosionAdaptor.cpp 2026-06-03, "
            "FEAMR/FEFilterAdaptorCriterion.cpp 2026-06-03, "
            "FEAMR/FEHexRefine.cpp 2026-06-03.)"
        ),
    },
    "cmake_embedding": {
        "find_package": (
            "find_package(FEBio) — defined by FEBioConfig.cmake at the "
            "FEBio SOURCE TREE ROOT (not in a lib/cmake subdirectory), "
            "so REQUIRED alongside it is the hint that says where that "
            "root is:\n"
            "  cmake -S . -B build -DFEBio_DIR=/path/to/febio-src\n"
            "It detects in-tree-build vs install layout automatically "
            "(_FEBIO_IS_SOURCE_TREE, decided by whether "
            "<config>/../../../include exists) and then globs for a "
            "*build* directory containing libfecore.so, so a cmake "
            "build directory named cbuild or build2 is found as "
            "readily as one named build. EXECUTED 2026-08-03 against "
            "the FEBio 4.12.0.86045466d source tree: FEBio_FOUND=1 and "
            "all eleven imported targets resolve."),
        "imported_targets": [
            "FEBio::FECore     — core FE framework",
            "FEBio::FEBioMech  — solid-mechanics module",
            "FEBio::FEBioMix   — biphasic / multiphasic mixture mechanics",
            "FEBio::FEBioFluid — fluid + biphasic-fluid",
            "FEBio::FEBioRVE   — RVE / multiscale support",
            "FEBio::FEBioPlot  — output plot file writer (.xplt)",
            "FEBio::FEBioXML   — .feb XML input parser",
            "FEBio::FEBioLib   — top-level orchestration / module registry",
            "FEBio::FEAMR      — adaptive mesh refinement",
            "FEBio::FEBioOpt   — parameter optimization",
            "FEBio::FEImgLib   — image-based meshing / DICOM support",
        ],
        "build_configurations": "_febio_find_library probes lib/Release "
                                "+ lib/Debug + lib/MinSizeRel + "
                                "lib/RelWithDebInfo, with a fallback that "
                                "matches anything under lib/. Multi-config "
                                "users (Visual Studio) get correct per-config "
                                "binaries.",
        "Signal": (
            "[Output] FEBioConfig.cmake demands ALL ELEVEN libraries and "
            "gives up on the first one it cannot find: it prints "
            "`FEBio library '<name>' could not be found.` and `Expected "
            "under: <prefix>`, then sets FEBio_FOUND=FALSE and returns. "
            "A partial build — say no FEImgLib because OpenCV was not "
            "installed — makes find_package(FEBio) report not-found even "
            "though the remaining subset would be enough for "
            "mechanics-only embedding. "
            "THE TRAP: the imported targets created BEFORE the missing "
            "one still exist. Executed 2026-08-03 with libfeimglib.so "
            "removed from a copy of the prefix: FEBio_FOUND came back 0 "
            "and `if(TARGET FEBio::FECore)` was nevertheless TRUE. So a "
            "CMakeLists that calls find_package(FEBio) WITHOUT REQUIRED "
            "and then tests for a target concludes FEBio is usable and "
            "fails later at link time. Always branch on FEBio_FOUND, or "
            "pass REQUIRED and let the configure step fail loudly. "
            "Workaround for a deliberate partial build: keep a local "
            "fork of FEBioConfig.cmake with the unwanted names removed "
            "from _FEBIO_LIBS. "
            "(Executed 2026-08-03 against FEBio 4.12.0.86045466d, both "
            "the complete prefix and a prefix with one library removed.)"
        ),
    },
}
