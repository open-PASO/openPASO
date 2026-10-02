"""Where tool output goes — resolved from the CALLER, not the installation.

Every output directory in openPASO used to be computed from the package's own
location:

    _OUTPUT_DIR = Path(__file__).resolve().parents[2] / "simulation_outputs"

That is a single directory shared by every caller, inside the install. Two
failures follow, both silent:

  1. A caller working in its own directory — an evaluation harness, a CI job, a
     user with two projects open — has its results written somewhere else
     entirely. Anything that later reads the working directory sees nothing
     and concludes the run produced nothing.
  2. The namespace is shared, so two callers using the same job name
     overwrite each other, and either can read the other's results.

Both were observed: 8 of 14 coupled runs in one batch of recorded runs
wrote through this path and were read as having produced nothing, and 168
simulation directories plus 58 coupling directories had accumulated inside
the repository.

This module is the ONE place that answers the question. Fixing it in a single
tool module and not the others is how the defect survived its first repair —
`tools/consolidated.py` was corrected while `tools/coupling.py`,
`tools/simulation.py` and `tools/mesh_generation.py` kept their own copies.

Each directory honours an environment variable and falls back to the historic
location, so nothing changes for a user who sets nothing.
"""
from __future__ import annotations

import os
from pathlib import Path

_INSTALL_ROOT = Path(__file__).resolve().parents[2]

# env var -> path relative to the install root, used when the var is unset
_DIRS = {
    "simulation_outputs": ("OPENPASO_OUTPUT_DIR", "simulation_outputs"),
    "coupling": ("OPENPASO_COUPLING_DIR", "benchmarks/coupling"),
    "meshes": ("OPENPASO_MESH_DIR", "meshes"),
    "benchmark_results": ("OPENPASO_BENCHMARK_DIR", "benchmarks/results"),
}


def output_dir(kind: str) -> Path:
    """The directory for `kind`, from the environment or the install root."""
    try:
        var, default = _DIRS[kind]
    except KeyError:
        raise KeyError(f"unknown output kind {kind!r}; "
                       f"known: {sorted(_DIRS)}") from None
    return Path(os.environ.get(var) or (_INSTALL_ROOT / default))
