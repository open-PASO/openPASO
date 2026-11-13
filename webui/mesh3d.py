"""A mesh or result file as a surface the browser can draw in 3D.

The browser draws with its own GPU; the server only reads the file once (meshio), keeps the
boundary of the domain and the point values on it, and sends them as packed arrays. A volume
mesh is reduced to its outer faces, so a million-cell result is still a few hundred thousand
triangles. Files in one folder whose names differ only by a number (field_level1.vtu,
field_level2.vtu; out_0001.vtu, ...) or a .pvd collection are one series the viewer steps through."""
from __future__ import annotations

import base64
import re
import threading
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

import numpy as np

MAX_TRIANGLES = 800_000
_VOLUME_FACES = {   # each cell type's faces, as corner indices (quads split later)
    "tetra": [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)],
    "tetra10": [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)],
    "hexahedron": [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    "hexahedron20": [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    "hexahedron27": [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    "wedge": [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
    "wedge15": [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
    "pyramid": [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
}
_SURFACE = {"triangle": 3, "triangle6": 3, "triangle7": 3, "quad": 4, "quad8": 4, "quad9": 4}

_cache: OrderedDict = OrderedDict()
_lock = threading.Lock()


class TooLarge(ValueError):
    pass


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _boundary(cells) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Triangles of the outer surface, and for each the (cell block, cell) it came from."""
    faces, owners = [], []
    for bi, block in enumerate(cells):
        t, data = block.type, np.asarray(block.data)
        if t in _VOLUME_FACES:
            for f in _VOLUME_FACES[t]:
                faces.append(data[:, f] if len(f) == 4 else np.column_stack([data[:, f], np.full(len(data), -1)]))
                owners.append(np.column_stack([np.full(len(data), bi), np.arange(len(data))]))
    tris, own = [], []
    if faces:
        F = np.vstack(faces)
        O = np.vstack(owners)
        key = np.sort(np.where(F < 0, np.iinfo(np.int64).max, F), axis=1)
        _, first, count = np.unique(key, axis=0, return_index=True, return_counts=True)
        outer = first[count == 1]                      # a face shared by two cells is inside
        F, O = F[outer], O[outer]
        tri = F[:, 3] < 0
        tris += [F[tri][:, :3], F[~tri][:, [0, 1, 2]], F[~tri][:, [0, 2, 3]]]
        own += [O[tri], O[~tri], O[~tri]]
    for bi, block in enumerate(cells):
        n = _SURFACE.get(block.type)
        if not n:
            continue
        data = np.asarray(block.data)[:, :n]
        idx = np.column_stack([np.full(len(data), bi), np.arange(len(data))])
        if n == 3:
            tris.append(data); own.append(idx)
        else:
            tris += [data[:, [0, 1, 2]], data[:, [0, 2, 3]]]; own += [idx, idx]
    if not tris:
        return np.zeros((0, 3), np.int64), np.zeros((0, 2), np.int64)
    return np.vstack(tris).astype(np.int64), np.vstack(own).astype(np.int64)


def _range(a: np.ndarray) -> tuple[float, float, list, list]:
    """(min, max) of the size (or the value), and per part, over every finite entry."""
    mag = np.linalg.norm(a, axis=1) if a.shape[1] > 1 else a[:, 0]
    ok = np.isfinite(mag)
    if not ok.any():
        return float("nan"), float("nan"), [], []
    return (float(mag[ok].min()), float(mag[ok].max()),
            [float(np.nanmin(a[:, i])) for i in range(a.shape[1])],
            [float(np.nanmax(a[:, i])) for i in range(a.shape[1])])


def _fields(m, used: np.ndarray, tri_owner: np.ndarray, tri_new: np.ndarray) -> list[dict]:
    out = []
    for name, arr in m.point_data.items():
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            a = a[:, None]
        if a.ndim != 2 or a.shape[0] != len(m.points) or a.shape[1] > 9:
            continue
        out.append((name, a[used], "point", _range(a)))
    # cell values become point values on the surface: the mean over the surface triangles at each point
    for name, blocks in m.cell_data.items():
        try:
            per = [np.asarray(b, dtype=float).reshape(len(b), -1) for b in blocks]
        except (TypeError, ValueError):
            continue
        ncomp = per[0].shape[1] if per else 0
        if not ncomp or ncomp > 9 or any(p.shape[1] != ncomp for p in per) or not len(tri_owner):
            continue
        offsets = np.concatenate([[0], np.cumsum([len(q) for q in per])])
        vals = np.vstack(per)[offsets[tri_owner[:, 0]] + tri_owner[:, 1]]
        acc = np.zeros((len(used), ncomp)); cnt = np.zeros(len(used))
        for k in range(3):
            np.add.at(acc, tri_new[:, k], vals); np.add.at(cnt, tri_new[:, k], 1)
        out.append((name, acc / np.maximum(cnt, 1)[:, None], "cell", _range(np.vstack(per))))
    fields = []
    for name, a, where, (wmin, wmax, wmin_c, wmax_c) in out:
        if not np.isfinite(a).any():
            continue
        comps = a.astype(np.float32)
        mag = np.linalg.norm(comps, axis=1) if comps.shape[1] > 1 else comps[:, 0]
        fin = mag[np.isfinite(mag)]
        fields.append({"name": name, "from": where, "ncomp": int(comps.shape[1]),
                       "values": _b64(comps), "min": float(fin.min()), "max": float(fin.max()),
                       "min_c": [float(np.nanmin(comps[:, i])) for i in range(comps.shape[1])],
                       "max_c": [float(np.nanmax(comps[:, i])) for i in range(comps.shape[1])],
                       # the whole body's range: a peak inside the body is not on the drawn surface
                       "body_min": wmin, "body_max": wmax, "body_min_c": wmin_c, "body_max_c": wmax_c})
    return fields


def surface(p: Path) -> dict:
    """The boundary surface and its fields; cached per file version."""
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    m, pts = _read(p)
    tris, owner = _boundary(m.cells)
    if len(tris) > MAX_TRIANGLES:
        raise TooLarge(f"{len(tris):,} surface triangles: more than this view draws ({MAX_TRIANGLES:,})")
    # every outer face of a volume points away from its cell, so shading and edges are right
    is_vol = np.array([b.type in _VOLUME_FACES for b in m.cells], bool)
    vol = is_vol[owner[:, 0]] if len(owner) else np.zeros(0, bool)
    if vol.any():
        offs = np.concatenate([[0], np.cumsum([len(b.data) for b in m.cells])])
        cen = np.vstack([pts[np.asarray(b.data)].mean(1) for b in m.cells])
        t, o = tris[vol], owner[vol]
        cc = cen[offs[o[:, 0]] + o[:, 1]]
        p0, p1, p2 = pts[t[:, 0]], pts[t[:, 1]], pts[t[:, 2]]
        inward = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), (p0 + p1 + p2) / 3 - cc) < 0
        t[inward] = t[inward][:, [0, 2, 1]]
        tris[vol] = t
    used, inv = np.unique(tris.ravel(), return_inverse=True)
    tri_new = inv.reshape(-1, 3)
    P = pts[used]
    lo, hi = P.min(0), P.max(0)
    out = {"kind": "mesh3d", "name": p.name,
           "n_points": int(len(used)), "n_triangles": int(len(tri_new)),
           "n_cells": int(sum(len(b.data) for b in m.cells)),
           "cell_types": sorted({b.type for b in m.cells}),
           "bounds": [lo.tolist(), hi.tolist()],
           "positions": _b64(P.astype(np.float32)),
           "triangles": _b64(tri_new.astype(np.uint32)),
           "fields": _fields(m, used, owner, tri_new)}
    with _lock:
        _cache[key] = out
        while len(_cache) > 8:
            _cache.popitem(last=False)
    return out


_meshes: OrderedDict = OrderedDict()


def _read(p: Path):
    """The file as meshio reads it, with 3D points; the last few are kept for cuts."""
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    with _lock:
        if key in _meshes:
            _meshes.move_to_end(key)
            return _meshes[key]
    import meshio
    m = meshio.read(p)
    pts = np.asarray(m.points, dtype=float)
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    with _lock:
        _meshes[key] = (m, pts)
        while len(_meshes) > 3:
            _meshes.popitem(last=False)
    return m, pts


# each volume cell as tetrahedra (corner nodes only), for cutting
_HEX5 = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 4, 5, 6), (3, 6, 7, 4), (1, 3, 4, 6)]
_WEDGE3 = [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]
_TETS = {"tetra": [(0, 1, 2, 3)], "tetra10": [(0, 1, 2, 3)], "hexahedron": _HEX5, "hexahedron20": _HEX5,
         "hexahedron27": _HEX5, "wedge": _WEDGE3, "wedge15": _WEDGE3, "pyramid": [(0, 1, 2, 4), (0, 2, 3, 4)]}


def cross_section(p: Path, axis: int, at: float) -> dict:
    """The plane {axis} = at through the inside of the body, with every quantity on it: point values
    interpolated along the cut edges, element values constant on each cut element."""
    m, pts = _read(p)
    tets, own = [], []
    for bi, b in enumerate(m.cells):
        for t in _TETS.get(b.type, []):
            data = np.asarray(b.data)
            tets.append(data[:, t]); own.append(np.column_stack([np.full(len(data), bi), np.arange(len(data))]))
    if not tets:
        return {"kind": "cross_section", "n_triangles": 0, "positions": "", "fields": [],
                "note": "this file has no volume cells to cut"}
    T, O = np.vstack(tets), np.vstack(own)
    d = pts[:, axis] - at
    s = d[T] > 0
    cnt = s.sum(1)
    cut = (cnt > 0) & (cnt < 4)
    T, O, s, cnt = T[cut], O[cut], s[cut], cnt[cut]
    order = np.argsort(~s, axis=1, kind="stable")          # nodes above the plane first
    V = np.take_along_axis(T, order, 1)
    edges, owner_tri = [], []
    one, three, two = cnt == 1, cnt == 3, cnt == 2
    if one.any():                                          # one node above: a triangle round it
        v = V[one]; edges.append(np.stack([v[:, [0, 1]], v[:, [0, 2]], v[:, [0, 3]]], 1)); owner_tri.append(O[one])
    if three.any():                                        # one node below
        v = V[three]; edges.append(np.stack([v[:, [3, 0]], v[:, [3, 1]], v[:, [3, 2]]], 1)); owner_tri.append(O[three])
    if two.any():                                          # two and two: a quad, as two triangles
        v = V[two]
        ac, ad, bd, bc = v[:, [0, 2]], v[:, [0, 3]], v[:, [1, 3]], v[:, [1, 2]]
        edges += [np.stack([ac, ad, bd], 1), np.stack([ac, bd, bc], 1)]; owner_tri += [O[two], O[two]]
    if not edges:
        return {"kind": "cross_section", "axis": axis, "at": at, "n_triangles": 0, "positions": "", "fields": [],
                "note": "the plane does not pass through the body"}
    E = np.vstack(edges)                                   # (ntri, 3, 2): the cut edge behind each corner
    Ow = np.vstack(owner_tri)
    if len(E) > MAX_TRIANGLES:
        raise TooLarge(f"{len(E):,} triangles in this cut: more than this view draws")
    i, j = E[..., 0].ravel(), E[..., 1].ravel()
    w = (d[i] / (d[i] - d[j]))[:, None]
    P = pts[i] + w * (pts[j] - pts[i])
    fields = []
    for name, arr in m.point_data.items():
        a = np.asarray(arr, dtype=float)
        a = a[:, None] if a.ndim == 1 else a
        if a.ndim != 2 or a.shape[0] != len(pts) or a.shape[1] > 9:
            continue
        fields.append((name, "point", a[i] + w * (a[j] - a[i])))
    offs = np.concatenate([[0], np.cumsum([len(b.data) for b in m.cells])])
    for name, blocks in m.cell_data.items():
        try:
            per = np.vstack([np.asarray(b, dtype=float).reshape(len(b), -1) for b in blocks])
        except (TypeError, ValueError):
            continue
        if per.shape[0] != offs[-1] or per.shape[1] > 9:
            continue
        fields.append((name, "cell", np.repeat(per[offs[Ow[:, 0]] + Ow[:, 1]], 3, axis=0)))
    return {"kind": "cross_section", "axis": axis, "at": at, "n_triangles": int(len(E)),
            "positions": _b64(P.astype(np.float32)),
            "fields": [{"name": n, "from": f, "ncomp": int(v.shape[1]), "values": _b64(v.astype(np.float32))}
                       for n, f, v in fields]}


_NUM = re.compile(r"^(.*?)(\d+)(\.[A-Za-z0-9]+)$")


def series(p: Path) -> list[Path]:
    """The files of the series this file belongs to, in order (just the file when it is alone)."""
    if p.suffix.lower() == ".pvd":
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError:
            return []
        sets = sorted(((float(d.get("timestep") or 0), d.get("file") or "") for d in root.iter("DataSet")),
                      key=lambda x: x[0])
        out = [(p.parent / f).resolve() for _, f in sets if f]
        return [q for q in out if q.is_file() and q.is_relative_to(p.parent.resolve())]
    mt = _NUM.match(p.name)
    if not mt:
        return [p]
    stem, _, ext = mt.groups()
    found = []
    for q in p.parent.iterdir():
        m2 = _NUM.match(q.name)
        if m2 and m2.group(1) == stem and m2.group(3) == ext and q.is_file():
            found.append((int(m2.group(2)), q))
    return [q for _, q in sorted(found)] or [p]
