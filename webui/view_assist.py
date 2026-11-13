"""Plain words about a result picture, turned into view settings.

The person types how they want the picture ("colour by pressure from 0 to 50 Pa", "show the
bending 10 times larger"); a model call turns that into the same small settings object the
viewer's controls write, and nothing else: no tools, no code, no file contents. The model is
told only the field names and the current settings, never values or paths. Whatever comes back
is checked against the fields the file really has before the browser applies it, and the browser
shows the change in the controls, where it can be undone. A request that needs a new calculation
(flow lines, a cross-section, a quantity the file does not hold) is answered with what it would
need, not attempted."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import httpx

from . import config

PARTS = ("size", "x", "y", "z")
SCALES = ("auto", "sequential", "diverging")
VIEWS = ("reset", "front", "top", "side")

_SYSTEM = """You set how a 3D picture of a simulation result is shown. You only choose display settings.
The file has these quantities (name: number of parts; 1 = a single value, 2 or 3 = a vector):
{fields}
It has {steps} step(s) ({step_kind}).
Current settings: {current}

Reply with ONE JSON object and nothing else:
{{"settings": {{ ...only the settings to change... }}, "said": "<one short plain sentence starting with 'Changed:' saying what changed, or why nothing changed>", "needs_calculation": null or "<plain words: what new calculation the request needs>"}}

Settings you may set:
- "field": one of the quantity names above, exactly as written
- "part": "size" (length of a vector, or the value itself), "x", "y" or "z" (a vector's part)
- "range": {{"auto": true}} or {{"auto": false, "min": <number>, "max": <number>}} in the file's own units
- "scale": "auto", "sequential" (for values that only grow) or "diverging" (for values around zero)
- "warp_field": a vector quantity name to move the shape by, or null for no movement
- "warp_factor": a number >= 0; 1 is the true size, 10 shows movements ten times larger
- "edges": true or false (draw the mesh lines)
- "step": a step number starting at 0
- "same_range": true or false (one colour range for all steps)
- "view": "reset", "front", "top" or "side"
- "cut_axis": "none", "x", "y" or "z" (a cut through the inside, across that direction; it shows the values inside)
- "cut_at": where to cut, from 0 (one end) to 1 (other end); 0.5 is the middle
- "cut_flip": true or false (keep the other side of the cut)

Never invent a quantity the file does not have. If the request needs something these settings cannot do
(flow lines, a new quantity such as vorticity or a stress the file does
not hold, a plot across files, a movie), leave "settings" empty and say so in "needs_calculation"."""


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def validate(raw: dict, fields: list[dict], steps: int) -> tuple[dict, list[str]]:
    """Keep only settings this file can take; say what was dropped."""
    names = {f["name"]: int(f.get("ncomp") or 1) for f in fields if isinstance(f, dict) and "name" in f}
    out: dict = {}
    dropped: list[str] = []
    s = raw if isinstance(raw, dict) else {}
    if "field" in s:
        if s["field"] in names:
            out["field"] = s["field"]
        else:
            dropped.append(f"no quantity named {s['field']!r} in this file")
    if "part" in s:
        n = names.get(out.get("field") or "", 3)
        if "field" in s and "field" not in out:
            dropped.append(f"part {s['part']!r} of a quantity the file does not have")
        elif s["part"] in PARTS and (s["part"] == "size" or (n >= 2 and PARTS.index(s["part"]) <= n)):
            out["part"] = s["part"]
        else:
            dropped.append(f"part {s['part']!r}")
    if isinstance(s.get("range"), dict):
        r = s["range"]
        if r.get("auto") is True:
            out["range"] = {"auto": True}
        else:
            lo, hi = _num(r.get("min")), _num(r.get("max"))
            if lo is not None and hi is not None and hi > lo:
                out["range"] = {"auto": False, "min": lo, "max": hi}
            else:
                dropped.append("a colour range whose top is not above its bottom")
    if s.get("scale") in SCALES:
        out["scale"] = s["scale"]
    if "warp_field" in s:
        w = s["warp_field"]
        if w is None or (w in names and names[w] >= 2):
            out["warp_field"] = w
        else:
            dropped.append(f"cannot move the shape by {w!r}")
    if "warp_factor" in s:
        f = _num(s["warp_factor"])
        if f is not None and 0 <= f <= 1e6:
            out["warp_factor"] = f
        else:
            dropped.append("deformation factor")
    for key in ("edges", "same_range", "cut_flip"):
        if isinstance(s.get(key), bool):
            out[key] = s[key]
    if s.get("cut_axis") in ("none", "x", "y", "z"):
        out["cut_axis"] = s["cut_axis"]
    if "cut_at" in s:
        c = _num(s["cut_at"])
        if c is not None and 0 <= c <= 1:
            out["cut_at"] = c
        else:
            dropped.append("a cut position outside the body")
    if "step" in s:
        st = s["step"]
        if isinstance(st, int) and 0 <= st < max(1, steps):
            out["step"] = st
        else:
            dropped.append(f"step {st!r}")
    if s.get("view") in VIEWS:
        out["view"] = s["view"]
    return out, dropped


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("the reply held no settings")
    return json.loads(m.group(0))


async def assist(model: str, text: str, fields: list[dict], current: dict, steps: int, step_kind: str,
                 log_dir: Path) -> dict:
    if model not in config.OPENROUTER_MODELS:
        raise LookupError("The picture assistant needs an OpenRouter model; this run uses another kind. "
                          "Use the settings on the right.")
    key = config.openrouter_key()
    if not key:
        raise LookupError("No OpenRouter key is set, so the picture assistant cannot answer.")
    listing = "\n".join(f"- {f['name']}: {int(f.get('ncomp') or 1)}" for f in fields[:60]) or "- (none)"
    system = _SYSTEM.format(fields=listing, steps=steps, step_kind=step_kind or "files",
                            current=json.dumps(current)[:1500])
    body = {"model": model, "temperature": 0, "max_tokens": 400,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": text[:2000]}]}
    t0 = time.time()
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{config.OPENROUTER_URL}/chat/completions", json=body,
                         headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    data = r.json()
    reply = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    try:
        got = _parse(reply)
    except ValueError:
        got = {"settings": {}, "said": "Nothing changed: the assistant's answer could not be read. Try other words."}
    settings, dropped = validate(got.get("settings") or {}, fields, steps)
    said = str(got.get("said") or ("Changed the settings." if settings else "Nothing changed."))[:300]
    if dropped:
        said += " Not applied: " + "; ".join(dropped) + "."
    need = got.get("needs_calculation")
    out = {"settings": settings, "said": said,
           "needs_calculation": str(need)[:400] if need else None,
           "model": model, "seconds": round(time.time() - t0, 2),
           "tokens": {"in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens")}}
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "view_requests.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "text": text[:2000], **out}) + "\n")
    except OSError:
        pass
    return out
