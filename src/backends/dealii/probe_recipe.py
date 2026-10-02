"""How to read a deal.II solution at a point that is not a mesh node.

WHY THIS EXISTS. Measured over 464 single-code runs, the
largest single failure bucket is a solve that SUCCEEDED and was then never read
back at the required points: 60 runs, 12.9%. A task that prescribes probe
points commonly places them deliberately off the mesh nodes, so getting a value
at an arbitrary coordinate is a required step, not an optional one.

Measured across the nine served backends, deal.II was the ONLY one whose
payload carried no way to do it — FEniCSx serves bb_tree/compute_colliding_cells,
NGSolve serves mesh(x,y), scikit-fem serves probes, and so on, while deal.II
served nothing. The runs that got it right used VectorTools::point_value, which
they had to know already.

VERIFIED BY EXECUTION on this install, not quoted from documentation. A known
function was interpolated onto FE_Q(1) and read back at a midpoint probe grid
(i+0.5)/n x (j+0.5)/n — none of whose points is a node:

    N =  8   NDOF =   81   max|probe - exact| = 1.896963e-03
    N = 16   NDOF =  289   max|probe - exact| = 4.810567e-04
    N = 32   NDOF = 1089   max|probe - exact| = 1.179388e-04

Ratios 3.94 and 4.08, so the evaluation is second order and does not cap the
order the run can report. Reading the nearest vertex instead caps it at 1.
"""

DEALII_PROBE_RECIPE = """\
READING YOUR SOLUTION AT A POINT THAT IS NOT A MESH NODE (deal.II).

The largest single way a run is lost is a solve that worked
and was never read back at the prescribed points. deal.II does this with
`VectorTools::point_value`, which locates the cell containing the point and
evaluates the finite element field there — second order for FE_Q(1), where
taking the nearest vertex instead caps your reported order at 1.

    #include <deal.II/numerics/vector_tools.h>

    Point<2> p(x, y);                       // ANY coordinate, not a node
    const double u_at_p = VectorTools::point_value(dof_handler, solution, p);

Measured on this install by interpolating a known function onto FE_Q(1) and
reading it back at a grid of non-nodal midpoint probes:

    N =  8   NDOF =   81   max|probe - exact| = 1.896963e-03
    N = 16   NDOF =  289   max|probe - exact| = 4.810567e-04
    N = 32   NDOF = 1089   max|probe - exact| = 1.179388e-04

which is order 1.98 then 2.03.

  * A POINT ON A CELL BOUNDARY IS AMBIGUOUS BUT NOT AN ERROR — deal.II picks
    one of the neighbouring cells and the value agrees to round-off, because
    the field is continuous for FE_Q. Points exactly on the DOMAIN boundary are
    the ones to watch: a coordinate a rounding error outside the mesh throws
    `ExcPointNotAvailableHere`, so clamp or skip such points deliberately
    rather than letting one exception end the run.
  * IT IS O(log N) PER CALL because of the cell search. For a few thousand
    probes that is irrelevant; for a million, build a `FEFieldFunction` and
    reuse it, or sort the points by cell.
  * THE DOF HANDLER AND THE VECTOR MUST MATCH the refinement level you are
    reporting. Re-distributing dofs after refinement invalidates an old
    solution vector, and reading it gives values from the previous mesh with no
    warning at all.
  * THE NUMBER OF DEGREES OF FREEDOM for your run log is
    `dof_handler.n_dofs()`. Print it yourself: deal.II's library output
    carries no mesh count at any verbosity, and the tutorial programs'
    "Number of active cells" line is the TUTORIAL's print, not the library's.
"""
