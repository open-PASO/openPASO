/* deal.II side of a coupling run by the openPASO `couple` driver: plane-strain
 * linear elasticity -div(sigma(u)) = b on ONE rectangle, with
 * sigma = lam tr(eps) I + 2 mu eps and eps = (grad u + grad u^T) / 2, holding
 * the partner's interface displacement (DIRICHLET role) or applying its
 * interface traction (NEUMANN role). participant_dealii_elastic.py writes
 * dealii_input.txt, builds this file into ./build and runs
 * ./build/dealii_side_elastic <input file> <interface output file>, both
 * absolute paths (the field file goes beside the interface file).
 *
 * AS SERVED IT DOES NOT COMPILE: its five holes are yours (1 mesh and
 * boundary ids, 2 material law and body force, 3 assembly, 4 boundary data,
 * 5 solve). The comment above each hole is its contract: the names it may
 * use, what it must leave set, and what the served lines after it check.
 * How to write it in deal.II is yours; the deal.II calls, one fact each and
 * measured on this install, are in the reply of
 * knowledge(topic='coupling', solver='dealii', physics='elasticity'). The
 * rest -- input, the two-component element and its dofs, interface nodes,
 * checks, consistent traction recovery, output -- is served and runs as it
 * stands once the holes are filled. Holes 1, 3, 4 and 5 fill variables
 * declared above them: keep their own helpers inside a { } block, since the
 * served lines after them declare names of their own.
 *
 * THE DISPLACEMENT HAS TWO COMPONENTS. fe is FESystem<2>(FE_Q<2>(degree), 2):
 * every support point carries one dof per component, fe.n_dofs_per_cell()
 * counts both (8 on a degree-1 cell), component[i] (served) is the component
 * of dof i (0 for u_x, 1 for u_y), and twin[i] the dof of the other component
 * at the same support point.
 *
 * SIGN. The exported traction is q = -(sigma . n_own), n_own this side's
 * outward normal: the two sides' exports cancel, and the Neumann role applies
 * the partner's q unchanged, as the load +integral(q . phi_i) over the
 * interface.
 *
 * INPUT lines: side dirichlet|neumann; lame lam mu (the plane-strain Lame
 * constants); box x0 x1 y0 y1; interface x|y <position> (the line x = .. or
 * y = ..); full_outer 0|1 (1: the two edges the interface ends on are held
 * too, 0: they are traction free); mesh nx ny; degree; level; samples n, then
 * n lines "s v_x v_y" (at s along the interface, the partner's displacement on
 * the Dirichlet role, its traction q on the Neumann role); source mx my, then
 * my lines of 2 mx values (b_x b_y at each point of a uniform grid of the box,
 * x fastest); outer mx my, the same for the displacement held on the outer
 * edges.
 * OUTPUT: dealii_interface.txt, "x y u_x u_y q_x q_y" per interface node in
 * order along it; dealii_field.txt, "x y u_x u_y" per support point. On the
 * console: deal.II's own version line (the library's version and git
 * revision, from its headers, through its log stream), then "NDOF = <n>"
 * (both components counted), "BODY_FORCE on|off", the residual the served
 * solve check measured, and a closing line.
 */
#include <deal.II/base/config.h>
static_assert(DEAL_II_VERSION_GTE(9, 5, 0), "this program needs deal.II 9.5 or newer, and the build found "
                                            "an older one: set DEAL_II_DIR to the tree discover names");
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
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
#include <array>
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

struct Grid                      // two values (x and y component) per point of a uniform grid of the box
{
  unsigned int mx = 0, my = 0;
  std::vector<double> v;
};

struct Input
{
  std::string side, axis_name;
  double lam = 0, mu = 0, x0 = 0, x1 = 1, y0 = 0, y1 = 1, position = 0;
  unsigned int nx = 0, ny = 0, degree = 1, level = 1;
  int full_outer = 0;
  std::vector<double> s;         // the partner's sample points along the interface, ascending
  std::vector<std::array<double, 2>> v;   // its two components at each
  Grid f, g;                     // the body force; the displacement held on the outer edges
};

[[noreturn]] void fail(const std::string &what)
{
  std::cerr << "dealii_side_elastic: " << what << std::endl;
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
  std::string line, key, need = " side lame box interface full_outer mesh samples source outer ";
  auto grid = [&](std::istringstream &ls, Grid &g) {
    ls >> g.mx >> g.my;
    g.v.resize(2 * std::size_t(g.mx) * g.my);
    for (double &val : g.v)
      file >> val;
  };
  while (std::getline(file, line))
  {
    std::istringstream ls(line);
    if (!(ls >> key))
      continue;
    if (need.find(" " + key + " ") != std::string::npos)
      need.erase(need.find(" " + key + " "), key.size() + 1);
    if (key == "side") ls >> in.side;
    else if (key == "lame") ls >> in.lam >> in.mu;
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
      std::vector<std::pair<double, std::array<double, 2>>> sv(count);
      for (auto &e : sv)
        file >> e.first >> e.second[0] >> e.second[1];
      std::sort(sv.begin(), sv.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
      for (const auto &e : sv)
        in.s.push_back(e.first), in.v.push_back(e.second);
    }
    else if (key == "source") grid(ls, in.f);
    else if (key == "outer") grid(ls, in.g);
    else
      fail("input key '" + key + "' is not one this program reads");
    if (ls.fail() || (file.fail() && !file.eof()))
      fail("input line '" + line + "' (or the values after it) did not parse");
  }
  if (need != " " || (in.side != "dirichlet" && in.side != "neumann") || in.s.empty() || in.f.mx < 2 ||
      in.f.my < 2 || in.g.mx < 2 || in.g.my < 2 || !(in.mu > 0) || !(in.lam + in.mu > 0))
    fail("cannot read " + name + ": it lacks" + need + "or its side, samples, source or outer grid, or its Lame "
         "constants (mu > 0 and lam + mu > 0) are missing");
  return in;
}

// The partner's samples at s along the interface, component c: piecewise
// linear, held constant beyond their ends (numpy.interp on each component).
double partner_at(const Input &in, const double s, const unsigned int c)
{
  for (unsigned int i = 1; i < in.s.size(); ++i)
    if (s <= in.s[i])
    {
      const double d = in.s[i] - in.s[i - 1];
      const double w = d > 0 ? std::max(0.0, (s - in.s[i - 1]) / d) : 1.0;
      return (1 - w) * in.v[i - 1][c] + w * in.v[i][c];
    }
  return in.v.back()[c];
}

// One of the wrapper's grids at a point, component c: bilinear on its grid of the box.
double grid_at(const Input &in, const Grid &g, const Point<2> &p, const unsigned int c)
{
  auto at = [](double t, double a, double b, unsigned int m, unsigned int &i) {
    const double r = std::min(std::max((t - a) / (b - a), 0.0), 1.0) * (m - 1);
    i = std::min(static_cast<unsigned int>(r), m - 2);
    return r - i;
  };
  unsigned int i, j;
  const double wx = at(p[0], in.x0, in.x1, g.mx, i), wy = at(p[1], in.y0, in.y1, g.my, j);
  auto v = [&](unsigned int a, unsigned int b) { return g.v[2 * (std::size_t(b) * g.mx + a) + c]; };
  return (1 - wy) * ((1 - wx) * v(i, j) + wx * v(i + 1, j)) + wy * ((1 - wx) * v(i, j + 1) + wx * v(i + 1, j + 1));
}

// The body force the wrapper sampled, at a point, as a Tensor<1, 2>.
Tensor<1, 2> source_sample(const Input &in, const Point<2> &p)
{
  Tensor<1, 2> b;
  for (unsigned int c = 0; c < 2; ++c)
    b[c] = grid_at(in, in.f, p, c);
  return b;
}

// The partner's samples as a Function<2> with two components, of a point on
// the interface (its displacement on the Dirichlet role, its traction on the
// Neumann role), ready for interpolate_boundary_values or a face quadrature:
// partner.value(p, c).
class PartnerData : public Function<2>
{
public:
  PartnerData(const Input &in, const unsigned int along) : Function<2>(2), in(in), along(along) {}
  double value(const Point<2> &p, const unsigned int c = 0) const override { return partner_at(in, p[along], c); }

private:
  const Input &in;
  const unsigned int along;
};

// The displacement the wrapper gave for the held outer edges, as a Function<2>
// with two components: outer.value(p, c).
class OuterData : public Function<2>
{
public:
  explicit OuterData(const Input &in) : Function<2>(2), in(in) {}
  double value(const Point<2> &p, const unsigned int c = 0) const override { return grid_at(in, in.g, p, c); }

private:
  const Input &in;
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
    const OuterData outer(in);
    const char *const NAME[2] = {"u_x", "u_y"};

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
    // face off it carries it, or the marked faces do not cover the interface.
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    GridGenerator::subdivided_hyper_rectangle(triangulation, std::vector<unsigned int>{in.nx, in.ny},
                                              Point<2>(in.x0, in.y0), Point<2>(in.x1, in.y1));
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
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
        {
          const bool on_line = std::abs(face->center()[AX] - in.position) < tol;
          if (on_line != (face->boundary_id() == INTERFACE_ID))
            fail(std::string("BOUNDARY IDS: a face ") + (on_line ? "on" : "off") + " the interface line has id " +
                 std::to_string(face->boundary_id()) + "; INTERFACE_ID goes on exactly the interface faces");
          iface_length += on_line ? face->measure() : 0.0;
        }
    if (std::abs(iface_length - (hi - lo)) > 1e-6 * (hi - lo))
      fail("BOUNDARY IDS: the INTERFACE_ID faces are " + num(iface_length) + " long, the interface " + num(hi - lo));

    // served: the displacement's element, two FE_Q components, and what each dof is
    const FESystem<2> fe(FE_Q<2>(in.degree), 2);
    DoFHandler<2> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);
    const MappingQ<2> mapping(1);
    const unsigned int n = dof_handler.n_dofs(), dpc = fe.n_dofs_per_cell();
    std::cout << "\nNDOF = " << n << std::endl;
    std::vector<Point<2>> support(n);
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support);
    std::vector<unsigned int> component(n);
    std::vector<types::global_dof_index> twin(n), dofs(dpc);
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(dofs);
      for (unsigned int i = 0; i < dpc; ++i)
      {
        component[dofs[i]] = fe.system_to_component_index(i).first;
        for (unsigned int k = 0; k < dpc; ++k)
          if (k != i && fe.system_to_component_index(k).second == fe.system_to_component_index(i).second)
            twin[dofs[i]] = dofs[k];
      }
    }
    DynamicSparsityPattern dsp(n);
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    SparsityPattern sparsity;
    sparsity.copy_from(dsp);
    SparseMatrix<double> system_matrix(sparsity);
    Vector<double> volume_rhs(n), system_rhs(n), solution(n);
    std::map<types::global_dof_index, double> boundary_values;

    // HOLE 2 OF 5 -- THE MATERIAL LAW AND THE BODY FORCE, for hole 3.
    // Uses: in.lam and in.mu (the plane-strain Lame constants the wrapper
    // passed) and source_sample(in, p) (the wrapper's samples of the body
    // force b, interpolated to a point p, as a Tensor<1, 2>).
    // Leaves: `stress` and `body_force`. stress(e) is the stress of linear
    // elasticity for a symmetric strain e, both a SymmetricTensor<2, 2>: lam
    // times the trace of e times the identity, plus 2 mu times e.
    // body_force(p) is b at a const Point<2> p, as a Tensor<1, 2>. Hole 3
    // reads both.
    // The lines after it stop unless stress gives that law for the three
    // unit strains, and body_force(p) equals the wrapper's samples at every
    // sample point.
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    auto stress = [&](const SymmetricTensor<2, 2> &e) {
      return in.lam * trace(e) * unit_symmetric_tensor<2>() + 2 * in.mu * e;
    };
    auto body_force = [&](const Point<2> &p) { return source_sample(in, p); };
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    double law_gap = 0;   // served: `stress` is the law of the lam and mu the wrapper passed
    for (unsigned int k = 0; k < 3; ++k)
    {
      SymmetricTensor<2, 2> e;   // the unit strains: e_xx = 1; e_yy = 1; e_xy = e_yx = 1
      e[k == 2 ? 0 : k][k == 2 ? 1 : k] = 1;
      const SymmetricTensor<2, 2> got = stress(e);
      const double want[3][3] = {{in.lam + 2 * in.mu, in.lam, 0}, {in.lam, in.lam + 2 * in.mu, 0}, {0, 0, 2 * in.mu}};
      law_gap = std::max({law_gap, std::abs(got[0][0] - want[k][0]), std::abs(got[1][1] - want[k][1]),
                          std::abs(got[0][1] - want[k][2])});
    }
    if (!(law_gap <= 1e-9 * (std::abs(in.lam) + 2 * in.mu)))
      fail("MATERIAL LAW: stress(e) misses lam tr(e) I + 2 mu e by up to " + num(law_gap) +
           " on a unit strain, with lam = " + num(in.lam) + " and mu = " + num(in.mu) + " from the wrapper");
    double b_max = 0, b_gap = 0;   // served: `body_force` is the body force the wrapper passed
    for (unsigned int j = 0; j < in.f.my; ++j)
      for (unsigned int i = 0; i < in.f.mx; ++i)
      {
        const Point<2> p(in.x0 + (in.x1 - in.x0) * i / (in.f.mx - 1), in.y0 + (in.y1 - in.y0) * j / (in.f.my - 1));
        const Tensor<1, 2> bp = body_force(p);
        for (unsigned int c = 0; c < 2; ++c)
        {
          const double bij = in.f.v[2 * (std::size_t(j) * in.f.mx + i) + c];
          b_max = std::max(b_max, std::abs(bij)), b_gap = std::max(b_gap, std::abs(bp[c] - bij));
        }
      }
    if (!(b_gap <= 1e-8 * std::max(1.0, b_max)))
      fail("BODY FORCE: body_force(p) differs from the wrapper's samples by up to " + num(b_gap) +
           "; one b must reach both (config source_ux and source_uy, or B_SRC in the wrapper)");

    // HOLE 3 OF 5 -- THE ASSEMBLY, WITH NO BOUNDARY CONDITION IN IT.
    // Uses: dof_handler (fe on the mesh: two FE_Q components of degree
    // in.degree), fe, mapping, dpc (the dofs of one cell, both components
    // counted), n, and stress and body_force from hole 2.
    // Leaves: system_matrix (built on `sparsity`, zero on entry) holding the
    // stiffness matrix of plane-strain elasticity: entry (i, j) is the
    // integral over the box of stress(eps(phi_j)) contracted with eps(phi_i),
    // eps the symmetric gradient of a vector shape function phi. And
    // volume_rhs (zero on entry) holding the load vector of the body force:
    // entry i is the integral of b . phi_i. Neither carries a boundary value
    // or a constraint: the consistent traction recovery reads system_matrix u
    // - volume_rhs on the interface rows, and a row a boundary condition
    // emptied has lost the traction it carried.
    // The lines after it stop when b is non-zero and volume_rhs is zero, when
    // system_matrix does not vanish on the three rigid motions (the two
    // translations and the rotation), and when it does not give the strain
    // energy of the three uniform unit strains.
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    {
      FEValues<2> fe_values(mapping, fe, QGauss<2>(fe.degree + 1),
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);
      const FEValuesExtractors::Vector displacement(0);
      FullMatrix<double> cell_matrix(dpc, dpc);
      Vector<double> cell_rhs(dpc);
      std::vector<types::global_dof_index> local_dof_indices(dpc);
      std::vector<SymmetricTensor<2, 2>> eps(dpc);
      for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;
        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
        {
          const double dx = fe_values.JxW(q);
          const Tensor<1, 2> bq = body_force(fe_values.quadrature_point(q));
          for (unsigned int i = 0; i < dpc; ++i)
            eps[i] = fe_values[displacement].symmetric_gradient(i, q);
          for (unsigned int i = 0; i < dpc; ++i)
          {
            for (unsigned int j = 0; j < dpc; ++j)
              cell_matrix(i, j) += scalar_product(stress(eps[j]), eps[i]) * dx;
            cell_rhs(i) += (bq * fe_values[displacement].value(i, q)) * dx;
          }
        }
        cell->get_dof_indices(local_dof_indices);
        system_matrix.add(local_dof_indices, cell_matrix);
        for (unsigned int i = 0; i < dpc; ++i)
          volume_rhs(local_dof_indices[i]) += cell_rhs(i);
      }
    }
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    if (b_max > 0 && volume_rhs.linfty_norm() == 0)
      fail("BODY_FORCE: the body force is not zero and volume_rhs is: hole 3 never integrated b");
    {   // served: system_matrix is the stiffness of that law, read off the linear displacements
      const double xc = 0.5 * (in.x0 + in.x1), yc = 0.5 * (in.y0 + in.y1);
      const double area = (in.x1 - in.x0) * (in.y1 - in.y0);
      auto field = [&](const unsigned int k, Vector<double> &u) {   // 0-2 rigid, 3-5 the unit strains
        for (unsigned int i = 0; i < n; ++i)
        {
          const double x = support[i][0] - xc, y = support[i][1] - yc, c = component[i];
          const double ux[6] = {1, 0, -y, x, 0, y}, uy[6] = {0, 1, x, 0, y, x};
          u(i) = c == 0 ? ux[k] : uy[k];
        }
      };
      const char *const RIGID[3] = {"the translation along x", "the translation along y", "the rotation"};
      const double energy[3][3] = {{in.lam + 2 * in.mu, in.lam, 0}, {in.lam, in.lam + 2 * in.mu, 0},
                                   {0, 0, 4 * in.mu}};
      Vector<double> u(n), w(n), image(n);
      for (unsigned int k = 0; k < 3; ++k)
      {
        field(k, u);
        system_matrix.vmult(image, u);
        const double scale = system_matrix.linfty_norm() * u.linfty_norm();
        if (!(image.linfty_norm() <= 1e-9 * scale))
          fail("ASSEMBLY: system_matrix times " + std::string(RIGID[k]) + " is " + num(image.linfty_norm()) +
               " (the matrix's scale " + num(scale) + "), not zero: it is not the stiffness of stress : eps, "
               "which vanishes on every rigid motion");
      }
      for (unsigned int a = 0; a < 3; ++a)
        for (unsigned int b = a; b < 3; ++b)
        {
          field(3 + a, u);
          field(3 + b, w);
          const double got = system_matrix.matrix_scalar_product(u, w), want = area * energy[a][b];
          if (!(std::abs(got - want) <= 1e-8 * area * (std::abs(in.lam) + 2 * in.mu)))
            fail("ASSEMBLY: the strain energy of the uniform unit strains " + std::to_string(a) + " and " +
                 std::to_string(b) + " (0: e_xx, 1: e_yy, 2: e_xy) is " + num(got) + " from system_matrix and " +
                 num(want) + " from the law over the box: system_matrix is not the stiffness of stress : eps");
        }
    }
    std::cout << "BODY_FORCE " << (b_max > 0 ? "on" : "off") << ": max|b| = " << b_max
              << ", max|volume_rhs| = " << volume_rhs.linfty_norm() << std::endl;

    // served: the interface nodes in order along it, one dof per component (iface[c][j]),
    // w_i = the integral of phi_i over the interface, and the load the partner's data makes there
    std::vector<types::global_dof_index> iface[2];
    for (unsigned int i = 0; i < n; ++i)
      if (std::abs(support[i][AX] - in.position) < tol)
        iface[component[i]].push_back(i);
    for (auto &list : iface)
      std::sort(list.begin(), list.end(), [&](auto a, auto b) { return support[a][AL] < support[b][AL]; });
    const unsigned int m = iface[0].size();
    for (unsigned int j = 0; j < m; ++j)
      if (iface[1].size() != m || twin[iface[0][j]] != iface[1][j])
        fail("INTERFACE NODES: the u_x and u_y dofs on the interface do not pair up point by point");
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
              flux_load(dofs[i]) += partner.value(fe_face.quadrature_point(q), component[dofs[i]]) * phi_ds;
            }
        }
    system_rhs = volume_rhs;

    // HOLE 4 OF 5 -- THE BOUNDARY DATA.
    // Uses: support (the support point of each dof), component (its
    // component, 0 for u_x and 1 for u_y), n, dirichlet (the role),
    // in.full_outer, AX and AL (the coordinates the interface fixes and runs
    // along), in.position, far, lo and hi (the interface ends), tol,
    // INTERFACE_ID, outer (the displacement held on the outer edges, a
    // Function<2> with two components: outer.value(p, c)), partner (the
    // partner's data along the interface, a Function<2> with two components:
    // its displacement on the Dirichlet role, its traction on the Neumann
    // role) and flux_load (per dof, the integral of that data's component
    // times phi_i over the interface faces).
    // Leaves: boundary_values (dof to value, empty on entry) holding exactly
    // the held dofs, both components of every held point. The edge opposite
    // the interface is held at outer.value of each dof's support point and
    // component, and so are the two edges the interface ends on when
    // in.full_outer is 1 (when it is 0 they are free and traction free). On
    // the Dirichlet role every interface dof is held at partner.value of its
    // support point and component, on the Neumann role none is (an interface
    // end that lies on a held edge may take either value). On the Neumann role
    // system_rhs (equal to volume_rhs on entry) also carries the partner's
    // traction as a load with its sign unchanged: system_rhs minus volume_rhs
    // equals flux_load on the interface dofs. system_matrix stays as hole 3
    // left it.
    // The lines after it check each of these.
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin
    VectorTools::interpolate_boundary_values(mapping, dof_handler, 0, outer, boundary_values);
    if (dirichlet)
      VectorTools::interpolate_boundary_values(mapping, dof_handler, INTERFACE_ID, partner, boundary_values);
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
                system_rhs(dofs[i]) +=
                  partner.value(fv.quadrature_point(q), component[dofs[i]]) * fv.shape_value(i, q) * fv.JxW(q);
          }
    }
    // ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end

    auto held = [&](types::global_dof_index i) { return boundary_values.count(i) > 0; };
    for (unsigned int i = 0; i < n; ++i)   // served: what hole 4 left, against the input
    {
      const Point<2> &p = support[i];
      const unsigned int c = component[i];
      const std::string at = std::string("the ") + NAME[c] + " dof at (" + num(p[0]) + ", " + num(p[1]) + ")";
      const bool on_if = std::abs(p[AX] - in.position) < tol, on_far = std::abs(p[AX] - far) < tol;
      const bool at_end = std::abs(p[AL] - lo) < tol || std::abs(p[AL] - hi) < tol;
      if ((on_far && !held(i)) || (at_end && !on_if && !on_far && held(i) != (in.full_outer != 0)) ||
          (held(i) && !on_if && !on_far && !at_end))
        fail("OUTER EDGES: " + at + " is " + (held(i) ? "held" : "free") + "; hole 4 holds both components on the "
             "edge opposite the interface, and on the two it ends on exactly when full_outer is 1, and no other");
      if (held(i) && !on_if && std::abs(boundary_values.at(i) - outer.value(p, c)) > 1e-9 * (1 + std::abs(outer.value(p, c))))
        fail("OUTER VALUES: " + at + " is held at " + num(boundary_values.at(i)) + ", and the outer displacement "
             "there is " + num(outer.value(p, c)));
      const double trace = partner.value(p, c);
      if (on_if && !at_end &&
          (held(i) != dirichlet || (dirichlet && std::abs(boundary_values.at(i) - trace) > 1e-9 * (1 + std::abs(trace)))))
        fail("INTERFACE: on the " + in.side + " role " + at + " must be " +
             (dirichlet ? "held at partner.value(p, c)" : "free, the partner's traction being a load"));
    }
    double gap = 0, flip = 0, size = 0, got = 0;
    for (const auto &list : iface)
      for (const auto i : list)
      {
        const double added = system_rhs(i) - volume_rhs(i);
        size = std::max(size, std::abs(flux_load(i))), got = std::max(got, std::abs(added));
        gap = std::max(gap, std::abs(added - flux_load(i))), flip = std::max(flip, std::abs(added + flux_load(i)));
      }
    if (!dirichlet && gap > 0.05 * size)
      fail("NEUMANN LOAD: on the interface system_rhs - volume_rhs misses the integral of the partner's traction "
           "times phi_i by " + num(gap) + " (its size " + num(size) + ")" +
           (got < 1e-12 * size ? ": the traction never entered system_rhs"
            : flip < 0.05 * size ? ": it entered with the opposite sign; apply the partner's number unchanged" : ""));

    // HOLE 5 OF 5 -- THE SOLVE.
    // Uses: system_matrix and system_rhs as holes 3 and 4 left them,
    // boundary_values, sparsity and n.
    // Leaves: `solution` (length n) equal to boundary_values on every held
    // dof and satisfying system_matrix solution = system_rhs, to round-off,
    // on every other row. system_matrix and system_rhs leave this hole as they
    // came in: the traction recovery after it reads their held rows.
    // The lines after it stop when a held row of system_matrix was emptied,
    // or when `solution` misses either condition.
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
        kept = std::max(kept, std::abs(solution(i) - boundary_values.at(i)));
      else
        r_free = std::max(r_free, std::abs(r(i) - system_rhs(i)));
    }
    const double scale = std::max(system_rhs.linfty_norm(), row_max * solution.linfty_norm());
    if (!std::isfinite(solution.l2_norm()) || kept > 1e-9 * (1 + solution.linfty_norm()) || r_free > 1e-8 * scale)
      fail("SOLVE: `solution` misses boundary_values by " + num(kept) + " and leaves system_matrix u - system_rhs at " +
           num(r_free) + " on the free dofs (system scale " + num(scale) + "): it does not solve this system");
    std::cout << "residual of the solve: " << num(r_free) << " on the free rows, " << num(kept)
              << " on the held dofs (system scale " << num(scale) << ")" << std::endl;

    // served: the CONSISTENT traction q = -(sigma . n_own), per component q_i = -r_i / w_i
    // with r = system_matrix u - volume_rhs (the body force's load alone, no boundary
    // condition). An interface end the outer condition also holds carries that reaction
    // too, so it takes the value of its nearest interior neighbour.
    r -= volume_rhs;
    std::vector<std::array<double, 2>> q(m, std::array<double, 2>{{0, 0}});
    std::vector<int> good;
    for (unsigned int j = 0; j < m; ++j)
    {
      const double s = support[iface[0][j]][AL];
      if (std::abs(weight(iface[0][j])) > 1e-14 && !(in.full_outer && (std::abs(s - lo) < tol || std::abs(s - hi) < tol)))
      {
        good.push_back(j);
        for (unsigned int c = 0; c < 2; ++c)
          q[j][c] = -r(iface[c][j]) / weight(iface[c][j]);
      }
    }
    for (unsigned int j = 0; j < m && !good.empty(); ++j)
      q[j] = q[*std::min_element(good.begin(), good.end(),
                                 [&](int a, int b) { return std::abs(a - int(j)) < std::abs(b - int(j)); })];

    std::ofstream if_file(out_if), field_file(out_field);
    if (!if_file || !field_file)
      fail("cannot write " + out_if + " or " + out_field);
    char buf[160];
    for (unsigned int j = 0; j < m; ++j)
    {
      const Point<2> &p = support[iface[0][j]];
      std::snprintf(buf, sizeof buf, "%.17g %.17g %.17g %.17g %.17g %.17g\n", p[0], p[1], solution(iface[0][j]),
                    solution(iface[1][j]), q[j][0], q[j][1]);
      if_file << buf;
    }
    for (unsigned int i = 0; i < n; ++i)
      if (component[i] == 0)
      {
        std::snprintf(buf, sizeof buf, "%.17g %.17g %.17g %.17g\n", support[i][0], support[i][1], solution(i),
                      solution(twin[i]));
        field_file << buf;
      }
    std::cout << "deal.II " << in.side << " side, level " << in.level << ": " << m
              << " interface nodes, max|u| = " << solution.linfty_norm() << std::endl;
  }
  catch (std::exception &exc)
  {
    std::cerr << "dealii_side_elastic: " << exc.what() << std::endl;
    return 1;
  }
  return 0;
}
