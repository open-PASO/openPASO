"""
openPASO — MCP Server

Connects any LLM to multiple open-source FEM codes via the Model Context Protocol.
Supported backends (catalog ships 9, runtime depends on local installs):
4C Multiphysics, FEniCSx (dolfinx), deal.II, NGSolve, scikit-fem, Kratos
Multiphysics, DUNE-fem, FEBio, SPARTA. Call `discover(query='list')` after start-up
for the current availability and per-backend install hints.
"""

import os
import sys
import logging

# Old and new spellings of this project's variables are the same variable.
# This runs before the guard below, which reads one of them.
from core.env_compat import install_aliases as _install_env_aliases
_install_env_aliases()

# ── PROTECT THE PROTOCOL CHANNEL BEFORE ANYTHING ELSE IS IMPORTED ──────────
# Over stdio transport, file descriptor 1 IS the JSON-RPC stream. Libraries
# this server loads write banners to that descriptor from C code — importing
# KratosMultiphysics prints its ASCII logo — and one such banner, emitted
# while `rediscover_backends` probed installs mid-session, corrupted the
# stream ("Invalid JSON: input ' |  / ...'") and killed the client's whole
# session. Python-level capture cannot stop a C-level write, so the guard is
# at descriptor level: keep a private duplicate of the real channel for the
# protocol, and point fd 1 at stderr so anything naive lands in the log
# instead of the wire. sys.stdout (which the MCP transport writes through)
# is rebound to the duplicate, so the protocol is unaffected.
if os.environ.get("OPENPASO_NO_FD_GUARD") != "1":       # escape hatch for tests
    _real_stdout_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(_real_stdout_fd, "w", buffering=1)

from mcp.server.fastmcp import FastMCP

# OFA_DISABLE_PITFALLS=1 → knowledge surfaces strip pitfall-DB content
# (per-backend pitfalls incl. Signal: anchors, post-mortems, cross-backend
# collation catalog); implemented in tools/consolidated.py. It only ever makes
# openPASO weaker, so it cannot inflate a result.
#
# There was an OFA_DISABLE_CRITIC toggle here that stripped the critic
# paragraph. The critic requirement is no longer a paragraph an agent may
# choose to follow — the server holds the review record and checks it — so the
# toggle is gone along with the runtime bypass it paired with.

# The instructions text (and the critic block it leads with) live in
# core.instructions so that a harness which must put the same text in
# front of a model imports it rather than carrying a drifting copy.
from core.instructions import INSTRUCTIONS, _CRITIC_BLOCK  # noqa: E402,F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("openpaso")

mcp = FastMCP(
    "openPASO",
    instructions=INSTRUCTIONS,
)

# Register consolidated tools (12 tools)
from tools.consolidated import register_consolidated_tools
register_consolidated_tools(mcp)

# Material data handed over from another framework (src/scale_bridge)
from tools.scale_bridge import register_scale_bridge_tools
register_scale_bridge_tools(mcp)

# Load all backends
from core.registry import load_all_backends
load_all_backends()


def main():
    logger.info("Starting openPASO MCP server")
    try:
        mcp.run(transport="stdio")
    finally:
        # Persist session journal on shutdown
        try:
            from core.session_journal import get_journal
            from pathlib import Path
            journal = get_journal()
            if journal.events:
                from core.session_journal import live_session_dir
                sessions_dir = live_session_dir()     # the same answer the live writer uses
                path = journal.save(sessions_dir)
                logger.info(f"Session journal saved: {path} ({len(journal.events)} events)")
        except Exception as e:
            logger.warning(f"Could not save session journal: {e}")


if __name__ == "__main__":
    main()
