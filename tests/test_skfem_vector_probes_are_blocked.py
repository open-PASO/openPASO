"""scikit-fem's probes() of a VECTOR basis returns its values BLOCKED BY COMPONENT.

The dof vector of an ElementVector basis is interleaved per node (u_x and u_y of node 0,
then of node 1, ...), but `basis.probes(P) @ x` is not: for n points it holds u_x at all n
points, then u_y at all n points. Read as interleaved, with [0::2] and [1::2], the values
scramble the components without any error, and a converged solve is reported as a wrong
field. The served text said only that the operator has shape (n_points, N), which holds for
a scalar basis.

These tests MEASURE the layout on the installed scikit-fem, for triangles and tetrahedra and
for the velocity basis that split() returns from a Taylor-Hood basis, and require every
served text that teaches point evaluation with probes() to state it.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

skfem = pytest.importorskip("skfem")
from skfem import (Basis, ElementTetP1, ElementTriP1, ElementTriP2,  # noqa: E402
                   ElementVector, MeshTet, MeshTri)

N_POINTS = 6
# Vector fields that each element holds exactly, so probing them is exact.
CASES = {
    "P1 triangles": (lambda: MeshTri().refined(3), ElementTriP1,
                     (lambda X: 3 * X[0] + 2 * X[1] + 1, lambda X: -X[0] + 5 * X[1] - 2)),
    "P2 triangles": (lambda: MeshTri().refined(2), ElementTriP2,
                     (lambda X: X[0] ** 2 + X[1], lambda X: X[0] * X[1] - X[1] ** 2)),
    "P1 tetrahedra": (lambda: MeshTet().refined(2), ElementTetP1,
                      (lambda X: X[0] + 2 * X[1], lambda X: X[2] - X[0], lambda X: 3 * X[1] + X[2])),
}


def _points(dim: int) -> np.ndarray:
    return 0.1 + 0.8 * np.random.default_rng(0).random((dim, N_POINTS))


def _layouts(values: np.ndarray, exact: np.ndarray) -> tuple[bool, bool]:
    """(blocked, interleaved): which reading of the probed values gives the exact field."""
    dim = exact.shape[0]
    return (np.allclose(values.reshape(dim, N_POINTS), exact, atol=1e-12),
            np.allclose(values.reshape(N_POINTS, dim).T, exact, atol=1e-12))


@pytest.mark.parametrize("case", sorted(CASES))
def test_vector_probes_are_blocked_by_component(case):
    mesh, element, fields = CASES[case]
    dim = len(fields)
    basis = Basis(mesh(), ElementVector(element()))
    x = basis.project(lambda X: np.stack([f(X) for f in fields]))
    P = _points(dim)
    exact = np.stack([f(P) for f in fields])

    # the dof vector IS interleaved per node: node 0 owns dofs 0..dim-1
    assert list(basis.nodal_dofs[:, 0]) == list(range(dim))
    assert basis.probes(P).shape == (dim * N_POINTS, basis.N)
    assert _layouts(basis.probes(P) @ x, exact) == (True, False)
    # interpolator() needs no reshape: it returns (dim, n_points)
    assert np.allclose(basis.interpolator(x)(P), exact, atol=1e-12)


def test_split_taylor_hood_velocity_probes_are_blocked():
    mesh = MeshTri().refined(3)
    composite = Basis(mesh, ElementVector(ElementTriP2()) * ElementTriP1())
    velocity = Basis(mesh, ElementVector(ElementTriP2())).project(
        lambda X: np.stack([X[0] ** 2 + X[1], X[0] * X[1] - X[1] ** 2]))
    pressure = Basis(mesh, ElementTriP1()).project(lambda X: 2 * X[0] - X[1] + 1)
    x = np.zeros(composite.N)
    u_dofs, p_dofs = composite.split_indices()
    x[u_dofs], x[p_dofs] = velocity, pressure
    (u, u_basis), (p, p_basis) = composite.split(x)
    P = _points(2)

    assert isinstance(u_basis.elem, ElementVector)
    assert _layouts(u_basis.probes(P) @ u,
                    np.stack([P[0] ** 2 + P[1], P[0] * P[1] - P[1] ** 2])) == (True, False)
    assert np.allclose(p_basis.probes(P) @ p, 2 * P[0] - P[1] + 1, atol=1e-12)


def _served_texts() -> dict:
    """Each served text that teaches `basis.probes(...) @ x` for scikit-fem: the facts that
    lead every scikit-fem reply of prepare_simulation and of a coupled side, and the
    installed-API reference."""
    from backends._installed_api import render
    from tools.consolidated import _deciding_block

    return {"deciding facts": _deciding_block("skfem", "linear_elasticity"),
            "installed API": render("skfem")}


@pytest.mark.parametrize("name", ["deciding facts", "installed API"])
def test_served_texts_state_the_vector_layout(name):
    text = _served_texts()[name]
    assert "basis.probes(" in text, f"the {name} no longer teaches probes(); update this test"
    # the rule must sit with the call it corrects, not anywhere in a long text
    assert re.search(r"basis\.probes\(.{0,600}?blocked", text, re.S | re.I), (
        f"the {name} does not say, next to probes(), that vector probes are blocked by component")
    assert any(r in text for r in ("reshape(dim, n_points)", "reshape(dim, npoints)")), (
        f"the {name} does not show the reshape that reads vector probes correctly")
