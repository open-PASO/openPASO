"""From accepted handoff data to what a continuum model needs.

Cubic elastic constants in Voigt notation, rotated about the z axis by an angle, and condensed to plane stress for
a thin sheet in the x-y plane. Voigt order: xx, yy, zz, yz, xz, xy, with engineering shear strains.
"""
from __future__ import annotations

import math

import numpy as np

_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def cubic_voigt(c11: float, c12: float, c44: float) -> np.ndarray:
    c = np.zeros((6, 6))
    c[:3, :3] = c12
    np.fill_diagonal(c[:3, :3], c11)
    c[3, 3] = c[4, 4] = c[5, 5] = c44
    return c


def _tensor(cv: np.ndarray) -> np.ndarray:
    t = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(_PAIRS):
        for b, (k, l) in enumerate(_PAIRS):
            for p, q in {(i, j), (j, i)}:
                for r, s in {(k, l), (l, k)}:
                    t[p, q, r, s] = cv[a, b]
    return t


def _voigt(t: np.ndarray) -> np.ndarray:
    return np.array([[t[i, j, k, l] for (k, l) in _PAIRS] for (i, j) in _PAIRS])


def rotate_about_z(cv: np.ndarray, theta_deg: float) -> np.ndarray:
    """Stiffness in a frame rotated by theta about z (the crystal axes turned by theta in the sheet plane)."""
    th = math.radians(theta_deg)
    r = np.array([[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]])
    t = np.einsum("ia,jb,kc,ld,abcd->ijkl", r, r, r, r, _tensor(cv))
    return _voigt(t)


def plane_stress(cv: np.ndarray) -> np.ndarray:
    """3x3 reduced stiffness (xx, yy, xy) for sigma_zz = sigma_yz = sigma_xz = 0."""
    inplane, outplane = [0, 1, 5], [2, 3, 4]
    a = cv[np.ix_(inplane, inplane)]
    b = cv[np.ix_(inplane, outplane)]
    c = cv[np.ix_(outplane, outplane)]
    return a - b @ np.linalg.solve(c, b.T)


def for_continuum(values: dict, theta_deg: float = 0.0) -> dict:
    cv = cubic_voigt(values["C11"], values["C12"], values["C44"])
    rot = rotate_about_z(cv, theta_deg)
    return {"voigt_order": "xx yy zz yz xz xy (engineering shear)", "units": "GPa", "theta_deg": theta_deg,
            "stiffness_voigt": np.round(rot, 6).tolist(),
            "plane_stress_xx_yy_xy": np.round(plane_stress(rot), 6).tolist()}
