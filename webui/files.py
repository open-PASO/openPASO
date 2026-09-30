"""Sandbox file browser — safe directory traversal under a fixed root.

Every path coming from the client is resolved and checked to be inside
SANDBOX_ROOT. Symlink escape attempts return 403.
"""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from . import config

MAX_LIST = 5000
TEXT_PREVIEW_BYTES = 200_000


def _safe(rel: str) -> Path:
    """Resolve ``rel`` (string, possibly relative to SANDBOX_ROOT) to an
    absolute Path that is guaranteed to live inside SANDBOX_ROOT.

    Raises PermissionError on escape attempts.
    """
    p = Path(rel) if rel else config.SANDBOX_ROOT
    if not p.is_absolute():
        p = config.SANDBOX_ROOT / p
    p = p.resolve()
    root = config.SANDBOX_ROOT.resolve()
    if root not in p.parents and p != root:
        raise PermissionError(f"path escapes sandbox: {p}")
    return p


def list_dir(rel: str = "") -> dict:
    p = _safe(rel)
    if not p.exists():
        return {"path": str(p), "entries": [], "exists": False}
    if not p.is_dir():
        return {"path": str(p), "entries": [], "exists": True,
                "is_file": True}
    entries = []
    for child in sorted(p.iterdir(),
                        key=lambda x: (x.is_file(), x.name.lower())):
        try:
            stat = child.stat()
        except OSError:
            continue
        entries.append({
            "name": child.name,
            "abs_path": str(child),
            "rel_path": str(child.relative_to(config.SANDBOX_ROOT)),
            "is_dir": child.is_dir(),
            "size": stat.st_size if child.is_file() else None,
            "mtime": stat.st_mtime,
            "kind": classify(child),
        })
        if len(entries) >= MAX_LIST:
            break
    return {"path": str(p), "rel": str(p.relative_to(config.SANDBOX_ROOT))
            if p != config.SANDBOX_ROOT.resolve() else "",
            "entries": entries, "exists": True}


def classify(p: Path) -> str:
    if p.is_dir():
        return "dir"
    suf = p.suffix.lower()
    if suf in (".vtu", ".vtk", ".pvtu", ".pvd", ".vtp"):
        return "vtk"
    if suf in (".xdmf", ".h5", ".hdf5"):
        return "hdf"
    if suf in (".png", ".jpg", ".jpeg", ".gif", ".svg"):
        return "image"
    if suf in (".csv", ".tsv"):
        return "table"
    if suf in (".json",):
        return "json"
    if suf in (".yaml", ".yml", ".4c", ".4cyaml"):
        return "yaml"
    if suf in (".msh", ".mesh", ".inp", ".dat"):
        return "mesh"
    if suf in (".py", ".cc", ".cpp", ".c", ".h", ".hpp",
               ".txt", ".log", ".md", ".sh", ".tex"):
        return "text"
    if suf == "" and p.name.lower().startswith(("result", "readme",
                                                 "makefile")):
        return "text"
    return "binary"


def read_text(rel: str) -> dict:
    p = _safe(rel)
    if not p.is_file():
        return {"path": str(p), "error": "not a file"}
    try:
        data = p.read_bytes()[:TEXT_PREVIEW_BYTES]
        for drop in range(0, 4):
            # the cut can land in the middle of a character, which is not a
            # reason to call a text file binary and refuse to show it
            try:
                text = (data[:len(data) - drop] if drop else data).decode("utf-8")
            except UnicodeDecodeError:
                continue
            return {"path": str(p), "text": text,
                    "truncated": p.stat().st_size > TEXT_PREVIEW_BYTES,
                    "kind": classify(p)}
        return {"path": str(p), "binary": True,
                "size": p.stat().st_size, "kind": classify(p)}
    except OSError as e:
        return {"path": str(p), "error": str(e)}
