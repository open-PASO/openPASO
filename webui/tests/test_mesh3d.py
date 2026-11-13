"""The 3D view reads a mesh's outer surface, the values on it, and the series a file belongs to."""
import base64

import meshio
import numpy as np
import pytest

from webui import mesh3d


def _arr(s, dt):
    return np.frombuffer(base64.b64decode(s), dtype=dt)


def test_two_tetrahedra_show_only_their_outer_faces(tmp_path):
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1.0]])
    cells = [("tetra", np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))]
    u = np.arange(15, dtype=float).reshape(5, 3)
    f = tmp_path / "two.vtu"
    meshio.write(f, meshio.Mesh(pts, cells, point_data={"u": u}, cell_data={"p": [np.array([1.0, 3.0])]}))
    s = mesh3d.surface(f)
    assert s["n_triangles"] == 6 and s["n_points"] == 5          # the shared face is inside
    tri = _arr(s["triangles"], np.uint32).reshape(-1, 3)
    assert tri.max() == 4
    fields = {x["name"]: x for x in s["fields"]}
    assert fields["u"]["ncomp"] == 3 and fields["u"]["from"] == "point"
    assert fields["u"]["max"] == pytest.approx(np.linalg.norm(u, axis=1).max())
    p = _arr(fields["p"]["values"], np.float32)
    assert fields["p"]["from"] == "cell" and p.min() >= 1 and p.max() <= 3


def test_every_outer_face_points_away_from_the_body(tmp_path):
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1.0]])
    f = tmp_path / "two.vtu"
    meshio.write(f, meshio.Mesh(pts, [("tetra", np.array([[0, 1, 2, 3], [2, 1, 3, 4]]))]))
    s = mesh3d.surface(f)
    P = _arr(s["positions"], np.float32).reshape(-1, 3).astype(float)
    T = _arr(s["triangles"], np.uint32).reshape(-1, 3)
    mid = P.mean(0)
    n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    assert (np.einsum("ij,ij->i", n, P[T].mean(1) - mid) > 0).all()


def test_a_hexahedron_is_twelve_triangles_and_a_quad_two(tmp_path):
    cube = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1.0]])
    f = tmp_path / "hex.vtu"
    meshio.write(f, meshio.Mesh(cube, [("hexahedron", np.arange(8)[None])]))
    assert mesh3d.surface(f)["n_triangles"] == 12
    g = tmp_path / "sheet.vtu"
    meshio.write(g, meshio.Mesh(cube[:4, :2], [("quad", np.arange(4)[None])]))
    s = mesh3d.surface(g)
    assert s["n_triangles"] == 2 and s["bounds"][0][2] == 0.0      # a 2D mesh is drawn in its plane


def test_a_cut_through_a_cube_is_the_whole_square_with_values_in_between(tmp_path):
    cube = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1.0]])
    f = tmp_path / "hex.vtu"
    meshio.write(f, meshio.Mesh(cube, [("hexahedron", np.arange(8)[None])],
                                point_data={"T": cube[:, 2] * 10}, cell_data={"k": [np.array([7.0])]}))
    c = mesh3d.cross_section(f, 2, 0.25)
    P = _arr(c["positions"], np.float32).reshape(-1, 3, 3).astype(float)
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1).sum()
    assert area == pytest.approx(1.0) and np.allclose(P[..., 2], 0.25)
    vals = {x["name"]: _arr(x["values"], np.float32) for x in c["fields"]}
    assert np.allclose(vals["T"], 2.5) and np.allclose(vals["k"], 7.0)
    assert mesh3d.cross_section(f, 2, 5.0)["n_triangles"] == 0          # a plane that misses the body


def test_numbered_files_are_one_series_in_number_order(tmp_path):
    for n in (10, 2, 1):
        (tmp_path / f"field_level{n}.vtu").write_text("x")
    (tmp_path / "other_level3.vtu").write_text("x")
    got = [q.name for q in mesh3d.series(tmp_path / "field_level2.vtu")]
    assert got == ["field_level1.vtu", "field_level2.vtu", "field_level10.vtu"]
    assert mesh3d.series(tmp_path / "other_level3.vtu")[0].name == "other_level3.vtu"


def test_a_pvd_collection_lists_its_files_in_time_order_and_stays_in_its_folder(tmp_path):
    for n in (0, 1):
        (tmp_path / f"s{n}.vtu").write_text("x")
    (tmp_path / "c.pvd").write_text(
        '<VTKFile type="Collection"><Collection>'
        '<DataSet timestep="2" file="s1.vtu"/><DataSet timestep="1" file="s0.vtu"/>'
        '<DataSet timestep="3" file="../elsewhere.vtu"/></Collection></VTKFile>')
    assert [q.name for q in mesh3d.series(tmp_path / "c.pvd")] == ["s0.vtu", "s1.vtu"]
