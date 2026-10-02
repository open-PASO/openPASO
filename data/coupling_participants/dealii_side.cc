/* deal.II side of a coupling run by the openPASO `couple` driver:
 * -div(k grad u) + c u = f on ONE rectangle, holding the partner's interface
 * trace (DIRICHLET role) or applying its outward flux (NEUMANN role).
 * participant_dealii.py writes dealii_input.txt, builds this file into
 * ./build and runs ./build/dealii_side <input file> <interface output file>,
 * both absolute paths (the field file goes beside the interface file).
 *
 * AS SERVED IT DOES NOT COMPILE: its five holes are yours (1 mesh and
 * boundary ids, 2 coefficient and source, 3 assembly, 4 boundary data,
 * 5 solve). The comment above each hole is its contract: the names it may
 * use, what it must leave set, and what the served lines after it check.
 * How to write it in deal.II is yours; the deal.II calls, one fact each and
 * measured on this install, are in the reply of
 * knowledge(topic='coupling', solver='dealii'). The rest -- input, interface
 * nodes, checks, consistent flux recovery, output -- is served and runs as it
 * stands once the holes are filled. Holes 1, 3, 4 and 5 fill variables
 * declared above them: keep their own helpers inside a { } block, since the
 * served lines after them declare names of their own.
 *
 * INPUT lines: side dirichlet|neumann; k (one number, or four: the full
 * 2x2 conductivity row by row); reaction; box x0 x1 y0 y1;
 * interface x|y <position> (the line x = .. or y = ..); outer (the value on
 * the held non-interface edges); full_outer 0|1 (1: the two edges the
 * interface ends on are held too); mesh nx ny; degree; level; samples n, then
 * n lines "s value" (the partner's trace on the Dirichlet role, its outward
 * flux on the Neumann role, at s along the interface); source mx my, then my
 * lines of mx values (f on a uniform grid of the box).
 * OUTPUT: dealii_interface.txt, "x y u q" per interface node in order along
 * it (q = this side's OUTWARD flux density); dealii_field.txt, "x y u" per
 * support point. On the console: deal.II's own version line (the library's
 * version and git revision, from its headers, through its log stream), then
 * "NDOF = <n>", "VOLUME_SOURCE on|off", the residual the served solve check
 * measured, and a closing line.
 */
#include <deal.II/base/config.h>
static_assert(DEAL_II_VERSION_GTE(9, 5, 0), "this program needs deal.II 9.5 or newer, and the build found "
                                            "an older one: set DEAL_II_DIR to the tree discover names");
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace dealii;

struct Input
{
  std::string side, axis_name;
  double reaction = 0, outer = 0, x0 = 0, x1 = 1, y0 = 0, y1 = 1, position = 0;
  Tensor<2, 2> K;                // the conductivity; k times the identity when the input gives one k
  unsigned int nx = 0, ny = 0, degree = 1, level = 1, mx = 0, my = 0;
  int full_outer = 0;
  std::vector<double> s, v, f;   // the partner's samples along the interface; the source grid
};

[[noreturn]] void fail(const std::string &what)
{
  std::cerr << "dealii_side: " << what << std::endl;
  std::exit(1);
}

std::string num(const double v)
{
  char b[32];
  std::snprintf(b, sizeof b, "%.3e", v);
  return b;
}

Input read_input(const std::string &name)
{
  std::ifstream file(name);
  Input in;
  std::string line, key, need = " side k box interface outer full_outer mesh samples source ";
  while (std::getline(file, line))
  {
    std::istringstream ls(line);
    if (!(ls >> key))
      continue;
    if (need.find(" " + key + " ") != std::string::npos)
      need.erase(need.find(" " + key + " "), key.size() + 1);
    if (key == "side") ls >> in.side;
    else if (key == "k")
    {
      std::vector<double> k;
      for (double c; ls >> c;)
        k.push_back(c);
      if (k.size() != 1 && k.size() != 4)
        fail("the k line needs one number or four (k11 k12 k21 k22)");
      for (unsigned int i = 0; i < 4; ++i)
        in.K[i / 2][i % 2] = k.size() == 1 ? (i % 3 == 0 ? k[0] : 0.0) : k[i];
      ls.clear();
    }
    else if (key == "reaction") ls >> in.reaction;
    else if (key == "outer") ls >> in.outer;
    else if (key == "box") ls >> in.x0 >> in.x1 >> in.y0 >> in.y1;
    else if (key == "interface") ls >> in.axis_name >> in.position;
    else if (key == "full_outer") ls >> in.full_outer;
    else if (key == "mesh") ls >> in.nx >> in.ny;
    else if (key == "degree") ls >> in.degree;
    else if (key == "level") ls >> in.level;
    else if (key == "samples")
    {
      unsigned int count = 0;
      ls >> count;
      in.s.resize(count), in.v.resize(count);
      for (unsigned int i = 0; i < count; ++i)
        file >> in.s[i] >> in.v[i];
    }
    else if (key == "source")
    {
      ls >> in.mx >> in.my;
      in.f.resize(in.mx * in.my);
      for (double &val : in.f)
        file >> val;
    }
    else
      fail("input key '" + key + "' is not one this program reads");
    if (ls.fail() || (file.fail() && !file.eof()))
      fail("input line '" + line + "' (or the values after it) did not parse");
  }
  if (need != " " || (in.side != "dirichlet" && in.side != "neumann") || in.s.empty() || in.mx < 2 || in.my < 2)
    fail("cannot read " + name + ": it lacks" + need + "or has no side, samples or source grid");
  return in;
}

// The partner's samples at s along the interface: piecewise linear, held
// constant beyond their ends (numpy.interp).
double partner_at(const Input &in, const double s)
{
  std::vector<std::pair<double, double>> sv;
  for (unsigned int i = 0; i < in.s.size(); ++i)
    sv.emplace_back(in.s[i], in.v[i]);
  std::sort(sv.begin(), sv.end());
  for (unsigned int i = 1; i < sv.size(); ++i)
    if (s <= sv[i].first)
    {
      const double w = std::max(0.0, (s - sv[i - 1].first) / (sv[i].first - sv[i - 1].first));
      return (1 - w) * sv[i - 1].second + w * sv[i].second;
    }
  return sv.back().second;
}

// The source the wrapper sampled, at a point: bilinear on its grid.
double source_sample(const Input &in, const Point<2> &p)
{
  auto at = [](double t, double a, double b, unsigned int m, unsigned int &i) {
    const double r = std::min(std::max((t - a) / (b - a), 0.0), 1.0) * (m - 1);
    i = std::min(static_cast<unsigned int>(r), m - 2);
    return r - i;
  };
  unsigned int i, j;
  const double wx = at(p[0], in.x0, in.x1, in.mx, i), wy = at(p[1], in.y0, in.y1, in.my, j);
  const double *f = &in.f[j * in.mx + i];
  return (1 - wy) * ((1 - wx) * f[0] + wx * f[1]) + wy * ((1 - wx) * f[in.mx] + wx * f[in.mx + 1]);
}

// The partner's samples as a Function<2> of a point on the interface (the
// trace on the Dirichlet role, the outward flux on the Neumann role), ready for
// interpolate_boundary_values or a face quadrature: partner.value(p).
class PartnerData : public Function<2>
{
public:
  PartnerData(const Input &in, const unsigned int along) : in(in), along(along) {}
  double value(const Point<2> &p, const unsigned int = 0) const override { return partner_at(in, p[along]); }

private:
  const Input &in;
  const unsigned int along;
};

int main(int argc, char **argv)
{
  try
  {
    deallog.depth_console(2);   // a SolverControl(n, tol, true, true) then logs its steps here
    // served: the library's own line, its version and git revision from its headers, so the
    // console shows which code ran whatever solver hole 5 uses (a direct one prints nothing)
    deallog << DEAL_II_PACKAGE_NAME << ' ' << DEAL_II_PACKAGE_VERSION << ", git revision " << DEAL_II_GIT_REVISION
            << std::endl;
    const Input in = read_input(argc > 1 ? argv[1] : "dealii_input.txt");
    const std::string out_if = argc > 2 ? argv[2] : "dealii_interface.txt";
    const std::string out_field = out_if.substr(0, out_if.find_last_of('/') + 1) + "dealii_field.txt";
    const bool dirichlet = in.side == "dirichlet";
    const unsigned int AX = in.axis_name == "y", AL = 1 - AX;   // AX: fixed on the interface; AL: along it
    const double lo = AL ? in.y0 : in.x0, hi = AL ? in.y1 : in.x1;   // the interface runs lo..hi
    const double a0 = AX ? in.y0 : in.x0, a1 = AX ? in.y1 : in.x1;
    const double far = std::abs(in.position - a0) < std::abs(in.position - a1) ? a1 : a0;   // the opposite edge
    const double tol = 1e-9 * std::max(in.x1 - in.x0, in.y1 - in.y0);
    const types::boundary_id INTERFACE_ID = 1;
    const PartnerData partner(in, AL);

    Triangulation<2> triangulation;
    // HOLE 1 OF 5 -- THE MESH AND ITS BOUNDARY IDS.
    // Uses: the box in.x0 to in.x1 by in.y0 to in.y1, the cell counts in.nx
    // and in.ny, the interface (the line where coordinate AX equals
    // in.position), far (the value of that coordinate on the opposite edge)
    // and tol.
    // Leaves: `triangulation`, declared above and empty, holding the box
    // meshed with in.nx by in.ny quadrilateral cells. Every boundary face on
    // the interface line carries boundary id INTERFACE_ID, and every other
    // boundary face another id, so that hole 4 can tell the interface from
    // the outer edges.
    // The lines after it stop when a face on the line lacks INTERFACE_ID, a
    // face off it carries it, the marked faces do not cover the interface,
    // or the mesh does not have in.nx cells along x and in.ny along y.
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    const std::vector<unsigned int> repetitions{in.nx, in.ny};
    GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, Point<2>(in.x0, in.y0),
                                              Point<2>(in.x1, in.y1));
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
        {
          const double c = face->center()[AX];
          face->set_boundary_id(std::abs(c - in.position) < tol ? INTERFACE_ID
                                : (std::abs(c - far) < tol || in.full_outer) ? 0 : 2);
        }
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    double iface_length = 0;   // served: INTERFACE_ID marks exactly the interface line
    unsigned int on_edge[2] = {0, 0};   // served: the boundary faces on the edges y = y0 and x = x0
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
        {
          const bool on_line = std::abs(face->center()[AX] - in.position) < tol;
          if (on_line != (face->boundary_id() == INTERFACE_ID))
            fail(std::string("BOUNDARY IDS: a face ") + (on_line ? "on" : "off") + " the interface line has id " +
                 std::to_string(face->boundary_id()) + "; INTERFACE_ID goes on exactly the interface faces");
          iface_length += on_line ? face->measure() : 0.0;
          on_edge[0] += std::abs(face->center()[1] - in.y0) < tol;
          on_edge[1] += std::abs(face->center()[0] - in.x0) < tol;
        }
    if (std::abs(iface_length - (hi - lo)) > 1e-6 * (hi - lo))
      fail("BOUNDARY IDS: the INTERFACE_ID faces are " + num(iface_length) + " long, the interface " + num(hi - lo));
    if (on_edge[0] != in.nx || on_edge[1] != in.ny || triangulation.n_active_cells() != in.nx * in.ny)
      fail("MESH: the mesh's edge y = in.y0 has " + std::to_string(on_edge[0]) + " boundary faces and its edge x = "
           "in.x0 has " + std::to_string(on_edge[1]) + " (" + std::to_string(triangulation.n_active_cells()) +
           " cells in all), and the input asks for in.nx = " + std::to_string(in.nx) + " cells along x and in.ny = " +
           std::to_string(in.ny) + " along y: hole 1 meshes the box with in.nx by in.ny cells");

    FE_Q<2> fe(in.degree);
    DoFHandler<2> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);
    const MappingQ<2> mapping(1);
    const unsigned int n = dof_handler.n_dofs(), dpc = fe.n_dofs_per_cell();
    std::cout << "\nNDOF = " << n << std::endl;
    std::vector<Point<2>> support(n);
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support);
    DynamicSparsityPattern dsp(n);
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    SparsityPattern sparsity;
    sparsity.copy_from(dsp);
    SparseMatrix<double> system_matrix(sparsity);
    Vector<double> volume_rhs(n), system_rhs(n), solution(n);
    std::map<types::global_dof_index, double> boundary_values;

    // HOLE 2 OF 5 -- THE COEFFICIENT AND THE SOURCE, for hole 3.
    // Uses: in.K (the conductivity the wrapper passed, a Tensor<2, 2> that is
    // k times the identity when the input gave one k) and source_sample(in, p)
    // (the wrapper's samples of f, interpolated to a point p).
    // Leaves: `coefficient` and `source`, each callable with a const Point<2>
    // p: coefficient(p) the conductivity at p as a Tensor<2, 2>, and source(p)
    // the value of f at p as a double. Hole 3 reads both.
    // The lines after it stop unless source(p) equals the wrapper's samples
    // at every sample point.
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    auto coefficient = [&](const Point<2> &) { return in.K; };
    auto source = [&](const Point<2> &p) { return source_sample(in, p); };
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    double f_max = 0, f_gap = 0;   // served: `source` is the source the wrapper passed
    for (unsigned int j = 0; j < in.my; ++j)
      for (unsigned int i = 0; i < in.mx; ++i)
      {
        const double fij = in.f[j * in.mx + i];
        const Point<2> p(in.x0 + (in.x1 - in.x0) * i / (in.mx - 1), in.y0 + (in.y1 - in.y0) * j / (in.my - 1));
        f_max = std::max(f_max, std::abs(fij)), f_gap = std::max(f_gap, std::abs(source(p) - fij));
      }
    if (f_gap > 1e-8 * std::max(1.0, f_max))
      fail("SOURCE: source(p) differs from the wrapper's samples by up to " + num(f_gap) +
           "; one f must reach both (config source_expr or F_SRC in the wrapper)");

    // HOLE 3 OF 5 -- THE ASSEMBLY, WITH NO BOUNDARY CONDITION IN IT.
    // Uses: dof_handler (FE_Q of degree in.degree on the mesh), fe, mapping,
    // dpc (the dofs of one cell), n, in.reaction (c below), and coefficient
    // and source from hole 2.
    // Leaves: system_matrix (built on `sparsity`, zero on entry) holding the
    // finite element matrix of -div(K grad u) + c u over the whole box, and
    // volume_rhs (zero on entry) holding the load vector of f. Neither carries
    // a boundary value or a constraint: the consistent flux recovery reads
    // system_matrix u - volume_rhs on the interface rows, and a row a boundary
    // condition emptied has lost the flux it carried.
    // The lines after it stop when f is non-zero and volume_rhs is zero, when
    // volume_rhs does not sum to the integral of f over the box (its test
    // functions sum to one) or, weighted by f at the support points, to the
    // integral of f times its interpolant, and, with no reaction, when a row
    // of system_matrix at a dof inside the box does not sum to zero.
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    {
      FEValues<2> fe_values(mapping, fe, QGauss<2>(fe.degree + 1),
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);
      FullMatrix<double> cell_matrix(dpc, dpc);
      Vector<double> cell_rhs(dpc);
      std::vector<types::global_dof_index> local_dof_indices(dpc);
      for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;
        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
        {
          const Point<2> &x = fe_values.quadrature_point(q);
          const Tensor<2, 2> kq = coefficient(x);
          const double fq = source(x), dx = fe_values.JxW(q);
          for (unsigned int i = 0; i < dpc; ++i)
          {
            for (unsigned int j = 0; j < dpc; ++j)
              cell_matrix(i, j) += ((kq * fe_values.shape_grad(j, q)) * fe_values.shape_grad(i, q) +
                                    in.reaction * fe_values.shape_value(i, q) * fe_values.shape_value(j, q)) * dx;
            cell_rhs(i) += fq * fe_values.shape_value(i, q) * dx;
          }
        }
        cell->get_dof_indices(local_dof_indices);
        system_matrix.add(local_dof_indices, cell_matrix);
        for (unsigned int i = 0; i < dpc; ++i)
          volume_rhs(local_dof_indices[i]) += cell_rhs(i);
      }
    }
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    // served: the interface dofs in order along it, w_i = the integral of
    // phi_i over the interface, and the load the partner's flux makes there
    std::vector<types::global_dof_index> iface, dofs(dpc);
    for (unsigned int i = 0; i < n; ++i)
      if (std::abs(support[i][AX] - in.position) < tol)
        iface.push_back(i);
    std::sort(iface.begin(), iface.end(), [&](auto a, auto b) { return support[a][AL] < support[b][AL]; });
    Vector<double> weight(n), flux_load(n);
    FEFaceValues<2> fe_face(mapping, fe, QGauss<1>(fe.degree + 2),
                            update_values | update_quadrature_points | update_JxW_values);
    for (const auto &cell : dof_handler.active_cell_iterators())
      for (const unsigned int f : cell->face_indices())
        if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == INTERFACE_ID)
        {
          fe_face.reinit(cell, f);
          cell->get_dof_indices(dofs);
          for (unsigned int q = 0; q < fe_face.n_quadrature_points; ++q)
            for (unsigned int i = 0; i < dpc; ++i)
            {
              const double phi_ds = fe_face.shape_value(i, q) * fe_face.JxW(q);
              weight(dofs[i]) += phi_ds;
              flux_load(dofs[i]) += partner.value(fe_face.quadrature_point(q)) * phi_ds;
            }
        }
    if (f_max > 0 && volume_rhs.linfty_norm() == 0)
      fail("VOLUME_SOURCE: the source is not zero and volume_rhs is: hole 3 never integrated f");
    // served: for a finite element function w = sum_i w_i phi_i, a load vector of f has sum_i
    // volume_rhs_i w_i equal to the integral of f w over the box, whatever quadrature built it. Two are
    // checked, the integrals taken with a finer Gauss rule than an assembly needs: w = 1 (the test
    // functions sum to one, so volume_rhs sums to the integral of f) and w = f at the support points
    // (f times its interpolant cannot cancel over the box, as f alone can). Measured on coarse meshes,
    // f written out or sampled: a right load meets the first to within 8 % of the integral of |f| and
    // the second to within 35 % of that of |f w| with one Gauss point per cell, both to within 2e-2 with
    // two; a load summed without its test function is about the dofs of a cell times the right one.
    Vector<double> f_node(n);
    for (unsigned int i = 0; i < n; ++i)
      f_node(i) = source(support[i]);
    double b_sum[2] = {0, 0}, f_int[2] = {0, 0}, f_abs[2] = {0, 0};
    FEValues<2> fe_sum(mapping, fe, QGauss<2>(fe.degree + 2), update_values | update_quadrature_points | update_JxW_values);
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_sum.reinit(cell);
      cell->get_dof_indices(dofs);
      for (unsigned int q = 0; q < fe_sum.n_quadrature_points; ++q)
      {
        double f_interp = 0;
        for (unsigned int i = 0; i < dpc; ++i)
          f_interp += f_node(dofs[i]) * fe_sum.shape_value(i, q);
        const double f_here = source(fe_sum.quadrature_point(q)), w = fe_sum.JxW(q);
        f_int[0] += f_here * w, f_abs[0] += std::abs(f_here) * w;
        f_int[1] += f_here * f_interp * w, f_abs[1] += std::abs(f_here * f_interp) * w;
      }
    }
    for (unsigned int i = 0; i < n; ++i)
      b_sum[0] += volume_rhs(i), b_sum[1] += volume_rhs(i) * f_node(i);
    const double within[2] = {0.15, 0.5};
    for (unsigned int k = 0; k < 2; ++k)
      if (std::abs(b_sum[k] - f_int[k]) > within[k] * f_abs[k])
        fail(std::string("VOLUME_SOURCE: volume_rhs ") + (k ? "weighted by f at the support points " : "") + "sums to " +
             num(b_sum[k]) + ", and " + (k ? "f times its interpolant" : "f") + " integrates to " + num(f_int[k]) +
             " over the box" + (std::abs(f_int[k]) > 1e-3 * f_abs[k] ? ", so volume_rhs is " +
             num(b_sum[k] / f_int[k]) + " times the load of f" : "") + ". A load vector of f, weighted by any finite "
             "element function w at its support points, sums to the integral of f w, to within " +
             (k ? "50" : "15") + " % of that of |f w| (" + num(f_abs[k]) + ") whatever the quadrature: hole 3's "
             "volume_rhs is not the load vector of f");
    if (in.reaction == 0)
    {
      double worst = 0;
      unsigned int row_at = 0;
      for (unsigned int i = 0; i < n; ++i)
      {
        const Point<2> &p = support[i];
        if (p[0] < in.x0 + tol || p[0] > in.x1 - tol || p[1] < in.y0 + tol || p[1] > in.y1 - tol)
          continue;                   // a dof on the boundary
        double sum = 0, size = 0;
        for (auto it = system_matrix.begin(i); it != system_matrix.end(i); ++it)
          sum += it->value(), size += std::abs(it->value());
        if (size > 0 && std::abs(sum) > worst * size)
          worst = std::abs(sum) / size, row_at = i;
      }
      if (worst > 1e-9)
        fail("MATRIX: with no reaction every row of the matrix of -div(K grad u) sums to zero (the gradients of its "
             "test functions sum to zero), and the row of the dof at (" + num(support[row_at][0]) + ", " +
             num(support[row_at][1]) + ") sums to " + num(worst) + " of the size of its entries: hole 3's system_matrix "
             "carries more than that operator");
    }
    std::cout << "VOLUME_SOURCE " << (f_max > 0 ? "on" : "off") << ": max|f| = " << f_max
              << ", max|volume_rhs| = " << volume_rhs.linfty_norm() << "; volume_rhs sums to " << b_sum[0]
              << " (the integral of f: " << f_int[0] << ") and, weighted by f at the support points, to " << b_sum[1]
              << " (the integral of f times its interpolant: " << f_int[1] << ")" << std::endl;
    system_rhs = volume_rhs;
    const double matrix_size = system_matrix.frobenius_norm();   // served: hole 4 leaves system_matrix as it is

    // HOLE 4 OF 5 -- THE BOUNDARY DATA.
    // Uses: support (the support point of each dof), n, dirichlet (the role),
    // in.outer, in.full_outer, AX and AL (the coordinates the interface fixes
    // and runs along), in.position, far, lo and hi (the interface ends), tol,
    // INTERFACE_ID, partner (the partner's data along the interface, a
    // Function<2>: its trace on the Dirichlet role, its outward flux on the
    // Neumann role) and flux_load (per dof, the integral of that data times
    // phi_i over the interface faces).
    // Leaves: boundary_values (dof to value, empty on entry) holding exactly
    // the held dofs, all of them on the boundary: every dof inside the box
    // stays free. The edge opposite the interface is held at in.outer, and so
    // are the two edges the interface ends on when in.full_outer is 1 (when
    // it is 0 they are free). On the Dirichlet role every interface dof is
    // held at partner.value of its support point, on the Neumann role none
    // is. An interface end that lies on a held edge is held too: on the
    // Dirichlet role at in.outer or at partner.value of its support point,
    // on the Neumann role at in.outer. That partner value is the partner's
    // own value at its end: in.outer when the partner holds its end too, the
    // value its solve left there when it does not, and the wrapper's starting
    // value on a coupling's first iteration. The lines after it accept
    // either, so they cannot see a partner end left free. Held at in.outer,
    // the end does not depend on the partner. On the
    // Neumann role system_rhs (equal to volume_rhs on entry) also carries the
    // partner's flux as a load with its sign unchanged, the partner's outward
    // flux being this side's inward one: system_rhs minus volume_rhs equals
    // flux_load on the interface dofs. system_matrix stays as hole 3 left it.
    // The lines after it check each of these.
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    VectorTools::interpolate_boundary_values(mapping, dof_handler, 0, Functions::ConstantFunction<2>(in.outer),
                                             boundary_values);
    if (dirichlet)
      VectorTools::interpolate_boundary_values(
        mapping, dof_handler, INTERFACE_ID, partner, boundary_values);
    else
    {
      FEFaceValues<2> fv(mapping, fe, QGauss<1>(fe.degree + 1),
                         update_values | update_quadrature_points | update_JxW_values);
      for (const auto &cell : dof_handler.active_cell_iterators())
        for (const unsigned int f : cell->face_indices())
          if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == INTERFACE_ID)
          {
            fv.reinit(cell, f);
            cell->get_dof_indices(dofs);
            for (unsigned int q = 0; q < fv.n_quadrature_points; ++q)
              for (unsigned int i = 0; i < dpc; ++i)
                system_rhs(dofs[i]) += partner.value(fv.quadrature_point(q)) * fv.shape_value(i, q) * fv.JxW(q);
          }
    }
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    // served: what hole 4 left, against the input, at every dof by where it lies: inside the box, on
    // an outer edge, on the interface between its ends, or at an interface end
    auto held = [&](types::global_dof_index i) { return boundary_values.count(i) > 0; };
    auto held_at = [&](types::global_dof_index i, const double v) {
      return held(i) && std::abs(boundary_values.at(i) - v) <= 1e-9 * (1 + std::abs(v));
    };
    auto left = [&](types::global_dof_index i) {   // the dof, and what hole 4 left on it
      return "the dof at (" + num(support[i][0]) + ", " + num(support[i][1]) + ") is " +
             (held(i) ? "held at " + num(boundary_values.at(i)) : std::string("free"));
    };
    for (unsigned int i = 0; i < n; ++i)
    {
      const Point<2> &p = support[i];
      const bool on_if = std::abs(p[AX] - in.position) < tol, on_far = std::abs(p[AX] - far) < tol;
      const bool at_end = std::abs(p[AL] - lo) < tol || std::abs(p[AL] - hi) < tol;
      const bool edge_held = on_far || in.full_outer != 0;
      const double trace = on_if ? partner.value(p) : 0.0;
      if (!on_if && !on_far && !at_end && held(i))
        fail("INSIDE THE BOX: " + left(i) + " and lies inside the box; hole 4 holds boundary dofs only and leaves "
             "every dof inside the box free");
      if ((on_far || (at_end && !on_if)) && (edge_held ? !held_at(i, in.outer) : held(i)))
        fail("OUTER EDGES: " + left(i) + " and lies on " +
             (on_far ? "the edge opposite the interface" : "an edge the interface ends on") + "; hole 4 " +
             (edge_held ? "holds it at in.outer (" + num(in.outer) + ")" : std::string("leaves it free, full_outer being 0")));
      if (on_if && !at_end && (dirichlet ? !held_at(i, trace) : held(i)))
        fail("INTERFACE: on the " + in.side + " role " + left(i) + " and lies on the interface between its ends; hole 4 " +
             (dirichlet ? "holds it at partner.value(p) (" + num(trace) + ")"
                        : std::string("leaves it free, the partner's flux being a load")));
      if (on_if && at_end &&
          !(in.full_outer ? held_at(i, in.outer) || (dirichlet && held_at(i, trace)) : dirichlet ? held_at(i, trace) : !held(i)))
        fail("INTERFACE ENDS: on the " + in.side + " role " + left(i) + " and is an end of the interface, on an edge " +
             (in.full_outer ? "the outer condition holds (full_outer 1); hole 4 holds it at " +
                                (dirichlet ? "partner.value(p) (" + num(trace) + ") or " : std::string()) + "in.outer (" +
                                num(in.outer) + ")"
                            : "left free (full_outer 0); hole 4 " +
                                (dirichlet ? "holds it at partner.value(p) (" + num(trace) + ")" : std::string("leaves it free")) +
                                ", as every interface dof"));
    }
    if (std::abs(system_matrix.frobenius_norm() - matrix_size) > 1e-12 * matrix_size)
      fail("MATRIX: hole 4 changed system_matrix (the root of the sum of its squared entries went from " + num(matrix_size) +
           " to " + num(system_matrix.frobenius_norm()) + "); hole 4 leaves it as hole 3 left it and holds dofs in "
           "boundary_values alone");
    double gap = 0, flip = 0, size = 0, got = 0;
    for (const auto i : iface)
    {
      const double added = system_rhs(i) - volume_rhs(i);
      size = std::max(size, std::abs(flux_load(i))), got = std::max(got, std::abs(added));
      gap = std::max(gap, std::abs(added - flux_load(i))), flip = std::max(flip, std::abs(added + flux_load(i)));
    }
    if (!dirichlet && gap > 0.05 * size)
      fail("NEUMANN LOAD: on the interface system_rhs - volume_rhs misses the integral of the partner's flux times "
           "phi_i by " + num(gap) + " (its size " + num(size) + ")" +
           (got < 1e-12 * size ? ": the flux never entered system_rhs"
            : flip < 0.05 * size ? ": it entered with the opposite sign; apply the partner's number unchanged" : ""));

    // served: the system hole 5 answers for, as holes 3 and 4 left it
    const Vector<double> rhs_before_solve(system_rhs);

    // HOLE 5 OF 5 -- THE SOLVE.
    // Uses: system_matrix and system_rhs as holes 3 and 4 left them,
    // boundary_values, sparsity and n.
    // Leaves: `solution` (length n) equal to boundary_values on every held
    // dof and satisfying system_matrix solution = system_rhs, to round-off,
    // on every other row. system_matrix and volume_rhs leave this hole as
    // they came in: the flux recovery after it reads both
    // (r = system_matrix u - volume_rhs). system_rhs is not read after it:
    // the lines after it judge `solution` against rhs_before_solve, the copy
    // taken above.
    // The lines after it stop when a held row of system_matrix was emptied,
    // when every dof is held, or when `solution` misses either condition.
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    {
      SparseMatrix<double> m(sparsity);
      m.copy_from(system_matrix);
      Vector<double> rhs(system_rhs);
      MatrixTools::apply_boundary_values(boundary_values, m, solution, rhs);
      SparseDirectUMFPACK direct;
      direct.initialize(m);
      direct.vmult(solution, rhs);
    }
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    // served: the solve answers for the system the recovery reads
    Vector<double> r(n);
    system_matrix.vmult(r, solution);
    double row_max = 0, r_free = 0, kept = 0;
    unsigned int n_free = 0, held_inside = 0;   // the rows the solve answers for; held dofs inside the box
    for (unsigned int i = 0; i < n; ++i)
    {
      double row = 0, off = 0;
      for (auto it = system_matrix.begin(i); it != system_matrix.end(i); ++it)
        row += std::abs(it->value()), off += it->column() != i ? std::abs(it->value()) : 0.0;
      row_max = std::max(row_max, row);
      if (held(i) && off == 0)
        fail("SOLVE: a held row of system_matrix has no off-diagonal entries left: the boundary values went "
             "into system_matrix itself. Apply them to a copy: a SparseMatrix<double> built on `sparsity` (or "
             "given reinit(sparsity)), then filled by copy_from(system_matrix). A SparseMatrix copy-constructed "
             "from system_matrix is left empty, and a library call on it crashes with no message");
      if (held(i))
      {
        const Point<2> &p = support[i];
        kept = std::max(kept, std::abs(solution(i) - boundary_values.at(i)));
        held_inside += p[0] > in.x0 + tol && p[0] < in.x1 - tol && p[1] > in.y0 + tol && p[1] < in.y1 - tol;
      }
      else
        ++n_free, r_free = std::max(r_free, std::abs(r(i) - rhs_before_solve(i)));
    }
    if (n_free == 0 && held_inside > 0)
      fail("SOLVE: every dof is held: boundary_values holds all " + std::to_string(n) + " of them, " +
           std::to_string(held_inside) + " inside the box, so no row of system_matrix u = system_rhs was left to "
           "solve and nothing was solved: `solution` is boundary_values alone. A dof inside the box is never held");
    const double scale = std::max(rhs_before_solve.linfty_norm(), row_max * solution.linfty_norm());
    if (!std::isfinite(solution.l2_norm()) || kept > 1e-9 * (1 + solution.linfty_norm()) || r_free > 1e-8 * scale)
      fail("SOLVE: `solution` misses boundary_values by " + num(kept) + " and leaves system_matrix u - system_rhs at " +
           num(r_free) + " on the free dofs, system_rhs as holes 3 and 4 left it (system scale " + num(scale) +
           "): it does not solve this system");
    std::cout << "residual of the solve: " << num(r_free) << " on the " << n_free << " free rows, " << num(kept)
              << " on the " << n - n_free << " held dofs (system scale " << num(scale) << ")"
              << (n_free ? "" : "; every dof is held, and nothing was solved") << std::endl;

    // served: the CONSISTENT outward flux q_i = -r_i / w_i, r = system_matrix u
    // - volume_rhs (the volume load alone, no boundary condition). An interface
    // end the outer condition also holds carries that reaction too, so it
    // takes the value of its nearest interior neighbour.
    r -= volume_rhs;
    std::vector<double> q(iface.size());
    std::vector<int> good;
    for (unsigned int j = 0; j < iface.size(); ++j)
    {
      const double s = support[iface[j]][AL];
      if (std::abs(weight(iface[j])) > 1e-14 && !(in.full_outer && (std::abs(s - lo) < tol || std::abs(s - hi) < tol)))
        good.push_back(j), q[j] = -r(iface[j]) / weight(iface[j]);
    }
    for (unsigned int j = 0; j < iface.size() && !good.empty(); ++j)
      q[j] = q[*std::min_element(good.begin(), good.end(),
                                 [&](int a, int b) { return std::abs(a - int(j)) < std::abs(b - int(j)); })];

    std::ofstream if_file(out_if), field_file(out_field);
    if (!if_file || !field_file)
      fail("cannot write " + out_if + " or " + out_field);
    char buf[128];
    for (unsigned int j = 0; j < iface.size(); ++j)
    {
      const Point<2> &p = support[iface[j]];
      std::snprintf(buf, sizeof buf, "%.17g %.17g %.17g %.17g\n", p[0], p[1], solution(iface[j]), q[j]);
      if_file << buf;
    }
    for (unsigned int i = 0; i < n; ++i)
    {
      std::snprintf(buf, sizeof buf, "%.17g %.17g %.17g\n", support[i][0], support[i][1], solution(i));
      field_file << buf;
    }
    std::cout << "deal.II " << in.side << " side, level " << in.level << ": " << iface.size()
              << " interface nodes, max|u| = " << solution.linfty_norm() << std::endl;
  }
  catch (std::exception &exc)
  {
    std::cerr << "dealii_side: " << exc.what() << std::endl;
    return 1;
  }
  return 0;
}
