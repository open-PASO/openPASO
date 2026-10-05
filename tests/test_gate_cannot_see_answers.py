"""Structural guard: nothing the agent touches may see an exact solution.

Three surfaces must stay clean, and this test enforces the separation
structurally rather than by inspection, so it cannot quietly regress:

  1. THE VERIFICATION GATE — its checks must work from the PROBLEM STATEMENT
     (source term, coefficients, domain) and from the run's own output. They
     must never receive a reference or exact solution, because a gate that
     needs the answer cannot verify a real engineering problem, where no
     answer exists. This is also why the gate's convergence evidence is a
     residual and a mesh-refinement study: both are reference-free.
  2. THE SYSTEM PROMPTS — the agent must not be handed a solution.
  3. THE KNOWLEDGE — covered by test_knowledge_not_contaminated.py.

Comparing against a manufactured exact solution is legitimate ONLY in the
external grader, after the fact, which the agent never sees.

Word boundaries matter here: "resolution" contains "solution", and flagging it
would pressure someone into mangling correct code.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# The verification gate and the reference-free checks it relies on.
GATE_MODULES = [
    REPO / "src" / "core" / "quality_checks.py",
    REPO / "src" / "core" / "residual_check.py",
    REPO / "src" / "core" / "fabrication_gate.py",
    REPO / "src" / "core" / "mesh_independence.py",
]

# Everything the agent is told, in either arm.
PROMPT_SOURCES = [
    REPO / "langgraph_eval" / "agent.py",
    REPO / "src" / "server.py",
]

# Names that would mean "the answer was handed in". \b prevents "resolution".
ANSWER_NAMES = re.compile(
    r"\b(exact_solution|u_exact|uexact|analytic_solution|analytical_solution|"
    r"reference_solution|true_solution|ground_truth|manufactured_solution)\b",
    re.IGNORECASE)

# A parameter of a gate function must not be an answer.
ANSWER_PARAM = re.compile(
    r"^(exact|u_exact|analytic|analytical|reference|truth|ground_truth|"
    r"expected_solution|exact_fn|analytic_fn|reference_fn)$", re.IGNORECASE)


def _functions(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            names = [a.arg for a in
                     (*args.posonlyargs, *args.args, *args.kwonlyargs)]
            if args.vararg:
                names.append(args.vararg.arg)
            if args.kwarg:
                names.append(args.kwarg.arg)
            yield node.name, names


@pytest.mark.parametrize("module", GATE_MODULES, ids=lambda p: p.name)
def test_gate_functions_never_accept_a_solution(module: Path):
    """No verification-gate function may take the answer as an argument.

    The gate's inputs are the problem statement and the run's own output. A
    residual needs the source term, not the solution; a refinement study needs
    neither.
    """
    if not module.exists():
        pytest.skip(f"{module.name} not present")
    offenders = [f"{fn}({', '.join(params)})"
                 for fn, params in _functions(module)
                 if any(ANSWER_PARAM.match(p) for p in params)]
    assert not offenders, (
        f"{module.name}: a verification-gate function accepts an exact "
        f"solution. The gate must verify without knowing the answer, or it "
        f"cannot verify a real problem.\n  " + "\n  ".join(offenders))


@pytest.mark.parametrize("module", GATE_MODULES, ids=lambda p: p.name)
def test_gate_code_contains_no_solution(module: Path):
    if not module.exists():
        pytest.skip(f"{module.name} not present")
    text = module.read_text()
    hits = []
    for m in ANSWER_NAMES.finditer(text):
        line_no = text[:m.start()].count("\n") + 1
        line = text.splitlines()[line_no - 1].strip()
        # Prose explaining WHY the gate must not see the answer is fine.
        if line.lstrip().startswith(("#", '"', "'", "*")):
            continue
        hits.append(f"{module.name}:{line_no}: {line[:100]}")
    assert not hits, (
        "A verification-gate module references an exact solution in code.\n  "
        + "\n  ".join(hits))


@pytest.mark.parametrize("source", PROMPT_SOURCES, ids=lambda p: p.name)
def test_prompts_hand_the_agent_no_solution(source: Path):
    """Neither arm's instructions may contain a solution."""
    if not source.exists():
        pytest.skip(f"{source.name} not present")
    text = source.read_text()
    hits = [f"{source.name}:{text[:m.start()].count(chr(10)) + 1}"
            for m in ANSWER_NAMES.finditer(text)]
    assert not hits, (
        f"{source.name} mentions an exact solution in agent-facing text.\n  "
        + "\n  ".join(hits))


def test_gate_cannot_import_the_answer_key():
    """The gate must have no path to a sealed key or a problem builder."""
    forbidden = re.compile(r"\b(keys|answer_key|build_problems|build_coupled|"
                           r"build_extra|grade_blind)\b")
    offenders = []
    for module in GATE_MODULES:
        if not module.exists():
            continue
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            for n in names:
                if forbidden.search(n or ""):
                    offenders.append(f"{module.name} imports {n}")
    assert not offenders, (
        "A verification-gate module can reach the answer key.\n  "
        + "\n  ".join(offenders))
