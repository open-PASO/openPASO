"""Turn a solver's field into something a browser can animate, with provenance.

Why this exists in the repository at all: a figure shown by a scientific product
must come from code that can be audited. Before this module the animations the
interface displayed were produced by a script outside the project, and there was
no record of what had been sampled, clipped or quantised. A reader could not
check the picture, and neither could we.

Everything that would change what a viewer concludes is recorded in the file:
the true range of the field against the range the colours actually span, how much
of the domain is saturated at that clip, the grid it was sampled onto, the
quantisation step in physical units, and the run that produced it.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np

FORMAT_VERSION = 1


def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL)
        return out.stdout.strip() or None
    except Exception:
        return None


def write_field_series(
    path: str | Path,
    frames: np.ndarray,
    times: Sequence[float],
    *,
    field: str,
    unit: str,
    extent: tuple[float, float, float, float],
    clip_percentile: float | None = 92.0,
    clip: tuple[float, float] | None = None,
    fps: int = 25,
    solver: str | None = None,
    source: str | None = None,
    notes: str | None = None,
) -> dict:
    """Write one animated field and return the metadata that went with it.

    ``frames`` is ``(n_times, ny, nx)``. Cells outside the domain, such as the
    inside of an obstacle, are ``NaN`` and become the mask.

    ``extent`` is ``(x0, x1, y0, y1)`` in physical units, so the picture can
    carry a scale rather than being a decoration.

    A signed field is clipped symmetrically, because a diverging colour ramp has
    to put zero at its centre or the sign is misread.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError(f"frames must be (n_times, ny, nx), got {frames.shape}")
    n_times, ny, nx = frames.shape
    if len(times) != n_times:
        raise ValueError(f"{len(times)} times for {n_times} frames")

    valid = ~np.isnan(frames[0])
    finite = frames[:, valid]
    true_lo, true_hi = float(np.nanmin(finite)), float(np.nanmax(finite))

    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        clip_percentile = None
    elif clip_percentile is None:
        lo, hi = true_lo, true_hi
    else:
        signed = true_lo < 0.0 < true_hi
        mag = float(np.percentile(np.abs(finite), clip_percentile))
        lo, hi = (-mag, mag) if signed else (
            float(np.percentile(finite, 100 - clip_percentile)),
            float(np.percentile(finite, clip_percentile)))

    if hi <= lo:
        raise ValueError(f"empty colour range: {lo} to {hi}")

    # How much of the picture is pinned at an end of the ramp. A saturated
    # plateau reads as a value and is only a bound, so the number is recorded.
    saturated = float(np.count_nonzero(
        (finite <= lo) | (finite >= hi)) / finite.size)

    q = np.clip((frames - lo) / (hi - lo), 0.0, 1.0)
    q = np.nan_to_num(q, nan=0.5)
    q8 = np.round(q * 255).astype(np.uint8)

    x0, x1, y0, y1 = extent
    meta = {
        "kind": "field_series",
        "format_version": FORMAT_VERSION,
        "field": field,
        "unit": unit,
        "nx": int(nx), "ny": int(ny),
        "x0": float(x0), "y0": float(y0),
        "dx": (float(x1) - float(x0)) / nx,
        "dy": (float(y1) - float(y0)) / ny,
        "vmin": round(lo, 6), "vmax": round(hi, 6),
        "fps": int(fps),
        "times": [round(float(t), 6) for t in times],

        # Everything below is what makes the picture checkable.
        "provenance": {
            "true_min": round(true_lo, 6),
            "true_max": round(true_hi, 6),
            "clip_percentile": clip_percentile,
            "saturated_fraction": round(saturated, 5),
            "quantisation_step": round((hi - lo) / 255.0, 8),
            "levels": 256,
            "interpolation": "none; values sampled at cell centres",
            "solver": solver,
            "source": source,
            "notes": notes,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": platform.node(),
            "commit": _git_commit(Path(__file__).resolve().parents[2]),
        },
    }

    mask_bytes = valid.astype(np.uint8).tobytes()
    frame_bytes = q8.tobytes()
    meta["mask"] = base64.b64encode(mask_bytes).decode()
    meta["frames"] = base64.b64encode(frame_bytes).decode()
    meta["provenance"]["sha256"] = hashlib.sha256(frame_bytes).hexdigest()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta))

    summary = dict(meta)
    summary.pop("frames"); summary.pop("mask"); summary.pop("times")
    summary["bytes"] = os.path.getsize(path)
    return summary


__all__ = ["write_field_series", "FORMAT_VERSION"]
