"""Coupling knowledge served to agents: the `couple` contract + one complete,
runnable participant script per backend.

Why this module exists
----------------------
A usability probe drove the live tools and measured the coupling corpus at
~15 kB for all nine backends together, none of it varying by backend, while a
single-code payload for one physics is ~36 kB. Two backends were named; seven
were not. Worse, it documented the DEPRECATED `coupled_solve` enum and told the
agent to write "a new subdomain script generator" — a private function no agent
can call — while the contract of the general `couple` tool existed only inside
that tool's own docstring.

The rules this module follows, because the consumer is a small model that will
not infer and will not hunt for a second payload:

  * load-bearing first — the contract before the theory;
  * COMPLETE runnable scripts, never fragments to assemble;
  * required vs optional stated in words, not implied;
  * every per-backend payload repeats the contract essentials, so landing on it
    directly is still enough to work from;
  * no absolute host paths: the interpreter/binary for a backend is whatever
    `discover(query='list')` reports for it on this install;
  * nothing here is an answer to a problem — no exact solutions, no measured
    error or order tables. Behaviour of the code, yes; results, no.
"""
from __future__ import annotations

from pathlib import Path

from core.paths import data_dir as _data_dir

_PARTICIPANT_DIR = _data_dir() / "coupling_participants"


_SOLVE_BEGIN = "# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin"
_SOLVE_END = "# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end"

_SOLVE_ELIDED = """\
# ─────────────────────────────────────────────────────────────────────────
# THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE.
#
# Build the mesh, the function space, the weak form and the linear solve for
# the problem you were given, in this backend, however you judge best. That is
# ordinary finite-element work and openPASO has no business dictating it.
#
# What openPASO does document — because you cannot guess it and it is what the
# interface check compares against — is everything AROUND the solve: the
# imports/exports handshake above, the interface sign convention, and the flux
# recovery below. Those are this tool's own interface, not your method.
#
# At this point you are expected to have produced:
#   * the discrete solution on this subdomain, with the partner's interface
#     data applied according to SIDE, and
#   * the assembled operator and the VOLUME load separately, because the flux
#     recovery below subtracts the volume load alone.
# ─────────────────────────────────────────────────────────────────────────"""


# THE BANNER SAYS WHICH HOLE IS WHICH. One text sat in every hole, and it
# ended "At this point you are expected to have produced the discrete
# solution ... with the partner's interface data applied" -- in the FIRST hole
# too, which comes before the served lines that apply that data. Measured: a
# worker solved in hole 1 of 3, the imported load never entered, and the
# side had to be rewritten. Each hole now says whether it comes before the
# solve, is the solve, or comes after it; a contract whose solve hole cannot
# be told from its code keeps the one text above.
_SOLVE_HEAD = "# THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE."
_SOLVE_RULE = "# " + "─" * 73
_SOLVE_CALL = __import__("re").compile(
    r"Inverse\(|\.solve\(|\bsolve\(|spsolve|LinearProblem\(|\.Solve\(|linalg\.solve|"
    r"galerkin\(|subprocess\.run\(")


def _code_lines(text: str) -> str:
    import re
    return "\n".join(re.sub(r"\s#.*$", "", ln) for ln in text.splitlines()
                     if ln.strip() and not ln.strip().startswith("#"))


# A deal.II WRAPPER'S HOLE IS THE I/O WITH ITS PROGRAM, NOT A PYTHON SOLVE. The
# generic banner told the transient wrapper's worker to "build the mesh, the
# function space, the weak form" in Python and promised a flux recovery below
# that the wrapper does not have (measured on a coupled round).
_PROGRAM_IO_ELIDED = """\
# ─────────────────────────────────────────────────────────────────────────
# THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE.
# HOLE {k} OF {n}: THE I/O WITH YOUR PROGRAM GOES HERE. The finite element solve
# is your C++ program (DEALII_EXE); this hole writes the input file it reads,
# runs it with subprocess.run on DEALII_EXE, its console captured and passed
# through, and reads its output back into the names listed at the end of this
# file. The flux it exports must be the program's CONSISTENT recovery: the
# residual of the system it assembled with no boundary condition applied,
# divided by the nodal interface weight.
# ─────────────────────────────────────────────────────────────────────────"""


# A DECK WRITER'S HOLE IS THE DECK, NOT A FINITE ELEMENT SOLVE. The SPARTA contract's one hole
# defines write_deck(seed); the served lines around it write the per-element temperature file,
# run the binary and read its dump. Measured on a coupled round: the generic banner told that hole
# to "assemble and solve" and promised a flux recovery that subtracts a volume load, none of which
# exists in a DSMC contract.
_DECK_IO_ELIDED = """\
# ─────────────────────────────────────────────────────────────────────────
# THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE.
# HOLE {k} OF {n}: THE DECK GOES HERE. write_deck(seed) writes SPARTA's input deck
# to DECK: the geometry, the gas, the grid, the walls and the sampling are yours.
# The deck MUST read SURF_FILE first, load TSURF_IN with `custom surf ... file` and
# dump SURF_FILE's elements to FLUX_OUT, as the docstring's step 3 shows.
# ─────────────────────────────────────────────────────────────────────────"""
_DECK_WRITER = __import__("re").compile(r"^\s*def write_deck\(", __import__("re").M)


def _hole_banner(k: int, n: int, m: int | None, program_io: bool = False) -> str:
    """The banner for hole k of n when hole m holds the solve (None: unknown)."""
    if program_io and k == m:
        return _PROGRAM_IO_ELIDED.replace("{k}", str(k)).replace("{n}", str(n))
    if m is None:
        return _SOLVE_ELIDED
    if k == m:
        text = _SOLVE_ELIDED.replace(
            _SOLVE_HEAD + "\n",
            _SOLVE_HEAD + f"\n# HOLE {k} OF {n}: THE SOLVE GOES HERE, after everything the served lines\n"
            "# above have put in place.\n", 1)
        if n > 1:
            # ONLY THE SOLVE BELONGS HERE when an earlier hole set the system up:
            # this banner told the solve hole to "build the mesh, the function
            # space, the weak form" too (measured, in the NGSolve contract).
            text = text.replace(
                "# Build the mesh, the function space, the weak form and the linear solve for\n"
                "# the problem you were given, in this backend, however you judge best. That is\n",
                "# Assemble and solve here the system the earlier hole set up -- the linear\n"
                "# solve itself -- however you judge best. That is\n", 1)
        return text
    if k < m:
        return (f"{_SOLVE_RULE}\n{_SOLVE_HEAD}\n"
                f"# HOLE {k} OF {n} COMES BEFORE THE SOLVE (hole {m}). Set up here what the served\n"
                "# lines after it use -- the names listed at the end of this file -- and do NOT\n"
                "# solve yet: whatever those served lines add (the partner's interface data, a\n"
                "# boundary condition) must be in place before the solve sees the system.\n"
                f"{_SOLVE_RULE}")
    return (f"{_SOLVE_RULE}\n{_SOLVE_HEAD}\n"
            f"# HOLE {k} OF {n} COMES AFTER THE SOLVE (hole {m}): the solution exists; define\n"
            "# here what the served lines after it use (the names listed at the end of this file).\n"
            f"{_SOLVE_RULE}")


def _script(name: str) -> str:
    """Return the participant CONTRACT shipped with openPASO: the file with its marked SOLVE regions elided.

    The script is a file rather than a string literal on purpose: the file is
    the artefact that gets executed in the test suite, so the text an agent is
    served and the text that was proven to run cannot drift apart.

    The old serving path cut out every mesh/form/solve region while calling the
    result a "complete participant". Measured consequence: 73% of the coupled
    runs that gave up never exchanged data once, and one recorded run spent
    74 tool calls rebuilding syntax already present in these executed files
    before delivering a false two-step convergence. Generic, parameterised
    solver templates are an openPASO capability just like the complete single-code
    templates returned by prepare_simulation; they contain no task answer or
    measured result. Serving the exact file the tests execute removes drift and
    lets the model spend its budget on the problem-specific edit block.
    """
    p = _PARTICIPANT_DIR / f"participant_{name}.py"
    if not p.is_file():
        return (f"[openPASO] participant script for '{name}' is missing from the "
                f"install (expected data/coupling_participants/{p.name}).")
    return _serve_participant(p)


def _serve_participant(p: Path) -> str:
    """THE ONE DOOR every participant is served through, and it FAILS CLOSED.

    Option B says openPASO serves the handshake, the interface sign convention,
    the consistent flux recovery and the exports schema, and does NOT serve a
    working finite element solve. That was enforced by a marker convention plus
    a call to `_elide_solve` -- and enforcement disappeared without anyone
    removing the mechanism. Measured on the committed tree: `_elide_solve` was
    called at ZERO call sites while all 30 participant files still carried the
    markers, so every one of them was served WHOLE, solve included, and the
    only thing that said so was a guard test failing quietly
    (test_the_contract_survives_the_real_serving_door).

    This file's own comment at `_elide_solve` records the same failure
    happening once before -- four helpers served their variant with a bare
    `p.read_text()`, so marking the files changed nothing an agent received.
    The lesson written there is "the mechanism has to sit where every door
    passes through it, not where the first one did", and there were five doors
    again.

    So there is now exactly one, and the permissive state is no longer the
    default:

      * marker present, region cut  -> serve the reduced text
      * marker present, nothing cut -> REFUSE (that is a wiring bug, and
                                       serving the solve is the failure it
                                       would cause)
      * marker absent               -> REFUSE the body, serve the module
                                       docstring, which is where the contract,
                                       the sign convention and the recovery
                                       formula live

    Absence of elision can no longer mean "serve everything".
    """
    if not p.is_file():
        return (f"[openPASO] participant script for '{p.stem}' is missing from "
                f"the install (expected data/coupling_participants/{p.name}).")
    text = p.read_text()
    if _SOLVE_BEGIN not in text and _glue_beside_program(p, text):
        return text                    # no solve here: its holes are the program's
    if _SOLVE_BEGIN not in text:
        doc = ""
        try:
            import ast as _ast
            doc = _ast.get_docstring(_ast.parse(text)) or ""
        except (SyntaxError, ValueError):
            doc = ""
        return (
            f"[openPASO WITHHOLDS THE BODY OF {p.name}]\n"
            f"This participant carries no SOLVE marker, so openPASO cannot tell "
            f"which region is the solve and will not serve the file. What "
            f"openPASO documents is its own interface -- the imports/exports "
            f"handshake, the interface sign convention, the consistent flux "
            f"recovery the interface check compares against, and the "
            f"iteration-1 fallback. "
            f"The mesh, the form and the solve are yours to write.\n\n"
            f"The contract, from this participant's own header:\n\n"
            + (doc if doc else "(this file has no docstring either)")
            + "\n")
    served = _elide_solve(text)
    # POSITIVE PROOF OF ELISION, NOT A LENGTH COMPARISON.
    #
    # My first version asked whether the result was SHORTER than the source.
    # It is not, and cannot be relied on to be: `_elide_solve` appends the
    # reconstruction contract naming what the hole must define, so a file
    # whose solve was correctly cut can come back longer. That test refused 13
    # of 30 participants outright and served 352 characters where the
    # handshake, the exports schema and the sign convention should have been --
    # withholding exactly what Option B says openPASO DOES serve.
    #
    # The proof that elision happened is the elision marker in the output, and
    # that the marked source region is gone from it.
    _cut_ok = _SOLVE_HEAD in served
    if _cut_ok:
        _a = text.find(_SOLVE_BEGIN) + len(_SOLVE_BEGIN)
        _b = text.find(_SOLVE_END, _a)
        if _b > _a:
            _body = text[_a:_b].strip()
            if _body and _body in served:
                _cut_ok = False        # marker echoed but the region survived
    if not _cut_ok:
        return (
            f"[openPASO WITHHOLDS THE BODY OF {p.name}]\n"
            f"The SOLVE marker is present but elision removed nothing, which "
            f"is a wiring bug in openPASO, not a licence to hand over a working "
            f"solve. Refusing rather than serving it. Report this: the file "
            f"has the marker at least once and the elision left the marked "
            f"region in place ({len(served)} characters served for a "
            f"{len(text)}-character source).\n")
    return served


# ══════════════════════════════════════════════════════════════════════════
# A PARTICIPANT WHOSE SOLVE IS A COMPILED PROGRAM: deal.II
# ══════════════════════════════════════════════════════════════════════════
#
# deal.II has no Python API, so its participant is a Python wrapper around a
# C++ program. Measured on two coupled rounds (10 cells whose deal.II side was
# a Dirichlet or Neumann heat subdomain): the served contract was the wrapper
# alone, the model had to write the whole program -- input parsing, mesh,
# assembly, boundary conditions, solve, the consistent flux recovery, the
# output -- and one of eight deal.II sides ever wrote exports.json; the rest
# died on compile errors or on programs that built and never exported.
#
# So the program is served the way the Python contracts are: a scaffold whose
# marked holes (mesh and boundary ids, coefficient and source, assembly,
# boundary data, solve) are elided, around a served input reader, interface
# node set, checks, consistent flux recovery and output. The markers are C++
# comments carrying the same text as the Python markers. The complete program
# (data/coupling_participants/dealii_side.cc, holes filled) is what the suite
# builds; it is not served and need not ship, so the served text is kept
# elided in tools/dealii_program_scaffold.py, generated from it and pinned to it
# by a test. The wrapper then has no solve of its own: it is served whole, and
# only while it is named below, its served program still has its five holes and
# no hole body, and it imports nothing that could solve.
#
# THE TRANSIENT CONTRACT IS SERVED THE SAME WAY. Measured on a transient coupled
# round: its wrapper alone was served, saying "No C++ source ships ... write and
# build it yourself", three of four give-ups named the transient C++ program as
# the reason to stop, and two cells spent 9 to 10 minutes each on programs that
# never linked. dealii_side_transient.cc is its scaffold: the loop over the steps,
# the last step's field, the per-step recovery and output are served; the holes
# are the mesh, the coefficient and source, the step's system, the boundary data
# at the new time and the step's solve.
#
# THE ELASTIC CONTRACT IS SERVED THE SAME WAY. Measured on a coupled elasticity
# round: its wrapper alone was served, with one hole for the I/O with a program
# "that you write and build yourself"; the one cell that reached its deal.II side
# spent about 16 minutes on twelve builds and three heap-corruption crashes, and
# its program indexed the two components past the end of the cell matrix.
# dealii_side_elastic.cc is its scaffold: the two-component element, its dofs'
# components, the interface nodes per component, the checks of the law and the
# stiffness, the vector traction recovery and the output are served; the holes
# are the mesh, the material law and body force, the assembly, the boundary data
# and the solve.
_CC_BEGIN = "// " + _SOLVE_BEGIN[2:]
_CC_END = "// " + _SOLVE_END[2:]
_CC_HOLE = "IS YOURS AND IS NOT SERVED HERE"
_PROGRAM_OF = {"participant_dealii.py": "dealii_side.cc",
               "participant_dealii_transient.py": "dealii_side_transient.cc",
               "participant_dealii_elastic.py": "dealii_side_elastic.cc"}
_SCAFFOLD_OF = {"dealii_side.cc": "SCAFFOLD", "dealii_side_transient.cc": "TRANSIENT_SCAFFOLD",
                "dealii_side_elastic.cc": "ELASTIC_SCAFFOLD"}
_GLUE_MODULES = {"json", "os", "subprocess", "sys", "pathlib", "numpy", "re", "shutil", "math", "time", "signal"}
def _dealii_cmake(program: str) -> str:
    """The six-line CMakeLists.txt that builds `program` into ./build/<its stem>."""
    name = Path(program).stem
    return (f"cmake_minimum_required(VERSION 3.13)\n"
            f"find_package(deal.II 9.0 REQUIRED HINTS ${{DEAL_II_DIR}} $ENV{{DEAL_II_DIR}})\n"
            f"deal_ii_initialize_cached_variables()\n"
            f"project({name} CXX)\n"
            f"add_executable({name} {program})\n"
            f"deal_ii_setup_target({name})\n"
            f"# target_compile_definitions({name} PRIVATE DEBUG)  # uncommented: deal.II's header checks name a crash\n")


def _elide_cc(text: str) -> tuple:
    """(served text, error): the C++ scaffold with every marked hole's lines cut
    and a banner in their place. FAILS CLOSED, like `_serve_participant`: no
    marker, an unbalanced one, or a hole body that survives is an error, never
    a licence to serve the program whole."""
    if _CC_BEGIN not in text or text.count(_CC_BEGIN) != text.count(_CC_END):
        return "", "the program scaffold has no marked hole, or an unbalanced one"
    spans, i = [], 0
    while (a := text.find(_CC_BEGIN, i)) >= 0:
        b = text.find(_CC_END, a)
        if b < 0:
            return "", "the program scaffold has an unterminated hole marker"
        spans.append((text.rfind("\n", 0, a) + 1, a, b, text.find("\n", b) + 1 or len(text)))
        i = b + len(_CC_END)
    out, i, n = [], 0, len(spans)
    for k, (line, a, b, end) in enumerate(spans, 1):
        pad = " " * (a - line)
        out.append(text[i:line])
        out.append(f"{pad}// ── HOLE {k} OF {n} {_CC_HOLE}: write the code the comment\n"
                   f"{pad}//    above asks for, in place of these two lines. ──\n")
        i = end
    served = "".join(out) + text[i:]
    for _line, a, b, _end in spans:
        body = text[a + len(_CC_BEGIN):b].strip().removesuffix("//").strip()
        if (body and body in served) or _CC_BEGIN in served or _CC_END in served:
            return "", "a marked hole of the program scaffold survived the elision"
    return served, ""


def served_program(name: str) -> tuple:
    """(served text, error) for a wrapper's program scaffold, holes elided. FAILS
    CLOSED: a served text that carries a hole marker, or lacks one of its hole
    banners, is refused, never served."""
    if name not in _SCAFFOLD_OF:
        return "", f"no served program scaffold named {name}"
    try:
        from . import dealii_program_scaffold as _mod
        scaffold = getattr(_mod, _SCAFFOLD_OF[name])
    except (ImportError, AttributeError):
        return "", "the served program scaffold is missing from the install"
    import re
    holes = [int(k) for k, n in re.findall(rf"HOLE (\d+) OF (\d+) {_CC_HOLE}", scaffold)]
    if (_CC_BEGIN in scaffold or _CC_END in scaffold or not holes
            or holes != list(range(1, len(holes) + 1))):
        return "", "the served program scaffold is not elided as it must be"
    return scaffold, ""


def regenerate_dealii_scaffold() -> None:
    """Rewrite tools/dealii_program_scaffold.py from the tested programs."""
    mod = Path(__file__).with_name("dealii_program_scaffold.py")
    text = mod.read_text()
    blocks = []
    for program, const in _SCAFFOLD_OF.items():
        served, err = _elide_cc((_PARTICIPANT_DIR / program).read_text())
        if err:
            raise SystemExit(f"{program}: {err}")
        blocks.append(f"{const} = r\'\'\'{served}\'\'\'\n")
    mod.write_text(text[:text.index("SCAFFOLD = r\'\'\'")] + "\n\n".join(blocks))


def program_of(p: Path):
    """The complete program (holes filled) a wrapper participant's served
    scaffold is made from, where the suite builds it; None for any other file."""
    name = _PROGRAM_OF.get(p.name)
    return p.parent / name if name else None


def _glue_beside_program(p: Path, text: str) -> bool:
    """True when `p` is a wrapper whose solve lives in its program's holes: it
    is named in _PROGRAM_OF, its served program is elided, and it imports
    nothing but glue (no finite-element library could solve inside it unseen)."""
    name = _PROGRAM_OF.get(p.name)
    if name is None or served_program(name)[1]:
        return False
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    return mods <= _GLUE_MODULES


def lean_view(served: str, keep: int = 2) -> str:
    """Thin every run of comment lines longer than `keep` down to its first
    `keep` lines plus an ellipsis line, leaving code, blanks, the elision
    banner and the reconstruction contract untouched.

    Measured: the served contracts are 41% comment lines (2149 lines, 886 of
    them comments, across the nine codes), and no small model copied one into
    a file. The explanations stay in the annotated block that
    knowledge(topic='coupling', solver=...) serves; this is the copyable form.
    """
    out, run = [], []
    def flush():
        if len(run) <= keep:
            out.extend(run)
        else:
            out.extend(run[:keep])
            out.append(run[0][:len(run[0]) - len(run[0].lstrip())]
                       + "# ... (explanation continues in the annotated block)")
        run.clear()
    # THE RECONSTRUCTION CONTRACT IS NOT EXPLANATION. The comment run inside a
    # hole (what the hole must leave behind, how to impose the served trace)
    # and the closing "LEAVE BEHIND" block are the only place the copyable form
    # names the variables the elided code has to define. Measured 2026-09-10:
    # this thinning cut both to "explanation continues in the annotated block",
    # so a worker holding only the copyable contract had to guess the names
    # from their later use. Both stay whole; only prose elsewhere is thinned.
    in_hole = in_contract = False
    for line in served.splitlines():
        st = line.strip()
        if "openPASO DOES NOT SERVE THIS" in st:
            flush(); out.append(line); in_hole = not in_hole
            continue
        if st.startswith("#") and "LEAVE BEHIND" in st:
            flush(); out.append(line); in_contract = True
            continue
        if in_hole or in_contract:
            flush(); out.append(line)
            continue
        if st.startswith("#") and "SOLVE" not in st and "openPASO" not in st \
                and "EDIT THIS BLOCK" not in st and "SELF-CHECK" not in st \
                and "MUST" not in st:
            run.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out) + ("\n" if served.endswith("\n") else "")


# A HOLE INSIDE A BLOCK GETS ITS BANNER AT THE BLOCK'S INDENTATION. The banner sat at
# column 0 for every hole, and a hole inside a `for`, `with`, `def` or `if` body was
# filled at column 0: the fill ended the block and the served lines after it stopped
# with IndentationError (measured: four of five cells on the transient contract, three
# of five on a `with TaskManager():` hole). The banner now sits where the fill goes.
def _hole_indent(body: str) -> str:
    """The leading whitespace of the least indented code line of a hole ('' if none)."""
    pads = [ln[:len(ln) - len(ln.lstrip())] for ln in body.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    return min(pads, key=len) if pads else ""


# WHAT EACH NAME A HOLE LEAVES BEHIND MUST BE, IN WORDS, where a contract states it. The list at
# the end of a contract is read off the code and holds names only; measured on a fluid-structure
# round, the fluid contract's list ("W dofc ds gdim msh n nit ph sigma tt vv ...") sent two cells
# to tabulate the mixed space W for `dofc` -- a call that raises -- while the served lines need
# the P1 vector space's coordinates. A contract states the words between these two lines, as
# "#   <name>  <what it must be>" rows (a row indented deeper continues the one above); the
# serving door moves them into that list, and the solve banner then points to them instead of
# the generic "what you are expected to have produced", which is a heat contract's.
_NAMES_BEGIN = "# ── HOLE NAMES IN WORDS ─ begin"
_NAMES_END = "# ── HOLE NAMES IN WORDS ─ end"
_GENERIC_PRODUCED = (
    "# At this point you are expected to have produced:\n"
    "#   * the discrete solution on this subdomain, with the partner's interface\n"
    "#     data applied according to SIDE, and\n"
    "#   * the assembled operator and the VOLUME load separately, because the flux\n"
    "#     recovery below subtracts the volume load alone.\n")
_NAMED_PRODUCED = (
    "# At this point you are expected to have produced what the served lines\n"
    "# below use: the names listed at the end of this file, each with what it\n"
    "# must be.\n")
_GENERIC_INTROS = (
    "# Build the mesh, the function space, the weak form and the linear solve for\n"
    "# the problem you were given, in this backend, however you judge best. That is\n",
    "# Assemble and solve here the system the earlier hole set up -- the linear\n"
    "# solve itself -- however you judge best. That is\n")
_NAMED_INTRO = (
    "# Write here the part of the solve the served lines below need -- the list\n"
    "# at the end of this file says what each name is -- however you judge best. That is\n")


def _hole_name_words(text: str) -> tuple:
    """({name: its words}, the text without the block) for a contract's HOLE NAMES IN WORDS block;
    ({}, text) when it has none."""
    import re
    a = text.find(_NAMES_BEGIN)
    b = text.find(_NAMES_END, a + 1) if a >= 0 else -1
    if a < 0 or b < 0:
        return {}, text
    words, cur = {}, None
    for ln in text[a + len(_NAMES_BEGIN):b].splitlines():
        m = re.match(r"#   (\w+)\s+(\S.*)$", ln)
        if m:
            cur = m.group(1)
            words[cur] = m.group(2).strip()
        elif cur and re.match(r"#\s{5,}\S", ln):
            words[cur] += " " + ln.lstrip("#").strip()
    a0 = text.rfind("\n", 0, a) + 1
    b1 = text.find("\n", b)
    return words, text[:a0] + (text[b1 + 1:] if b1 >= 0 else "")


def _elide_solve(text: str) -> str:
    """Cut every marked SOLVE region out of a participant's source.

    Split out of `_script` because `_script` was NOT the only door. Four
    helpers — the 3-D, transient, role and vector blocks — served their
    variant with a bare `p.read_text()`, so marking those files changed
    nothing an agent actually receives: `knowledge(solver='fenics')` serves the
    scalar script through `_script` AND the elastic and transient variants
    through those helpers, and `create_rectangle` was still arriving by the
    second route while the first was clean. The mechanism has to sit where
    every door passes through it, not where the first one did.
    """
    words, text = _hole_name_words(text)
    spans, i = [], 0
    while True:
        a = text.find(_SOLVE_BEGIN, i)
        if a < 0:
            break
        b = text.find(_SOLVE_END, a)
        spans.append((a, b))
        if b < 0:                       # unterminated marker: serve nothing after
            break
        i = b + len(_SOLVE_END)
    solving = [k for k, (a, b) in enumerate(spans, 1)
               if b > a and _SOLVE_CALL.search(_code_lines(text[a:b]))]
    m = solving[0] if len(solving) == 1 else None
    program_io = "DEALII_EXE" in text           # a deal.II wrapper: its hole runs the program
    decks = {k for k, (a, b) in enumerate(spans, 1)  # a SPARTA wrapper: its hole writes the deck
             if b > a and "SPARTA" in text and _DECK_WRITER.search(text[a:b])
             and not _SOLVE_CALL.search(_code_lines(text[a:b]))}
    out, i = [], 0
    for k, (a, b) in enumerate(spans, 1):
        out.append(text[i:a])
        pad = _hole_indent(text[a + len(_SOLVE_BEGIN):b]) if b > a else ""
        banner = (_DECK_IO_ELIDED.replace("{k}", str(k)).replace("{n}", str(len(spans))) if k in decks
                  else _hole_banner(k, len(spans), m, program_io))
        if words:
            banner = banner.replace(_GENERIC_PRODUCED, _NAMED_PRODUCED)
            for _intro in _GENERIC_INTROS:
                banner = banner.replace(_intro, _NAMED_INTRO)
        out.append("\n".join(pad + ln for ln in banner.splitlines()) + "\n")
        if b < 0:
            i = len(text)
            break
        i = b + len(_SOLVE_END)
        if i < len(text) and text[i] == "\n":
            i += 1
    out.append(text[i:])
    return _append_reconstruction_contract("".join(out), text, words)


def _cut_regions(original: str) -> list:
    """The marked SOLVE regions of a participant's source, in order."""
    cut, i = [], 0
    while True:
        a = original.find(_SOLVE_BEGIN, i)
        if a < 0:
            break
        b = original.find(_SOLVE_END, a)
        if b < 0:
            cut.append(original[a:])
            break
        cut.append(original[a:b])
        i = b + len(_SOLVE_END)
    return cut


def _cut_source(original: str) -> str:
    """The cut regions joined, each read at its own indentation."""
    # A HOLE INSIDE A BLOCK IS READ AT ITS OWN INDENTATION. Joined as they sit, an indented
    # hole does not parse, and the fallback above sees `name = ...` lines only: `u, v = ...`
    # in another hole dropped v from the list its served lines use.
    def _dedented(region):
        k = len(_hole_indent(region))
        return "\n".join(ln[k:] if ln[:k].isspace() else ln.lstrip()
                         for ln in region.splitlines()) if k else region
    return "\n".join(_dedented(c) for c in _cut_regions(original))


def _cut_definitions(cut: str) -> dict:
    """{name: 'class' | 'function' | 'value'} for what the cut regions bind at their own level.

    A CLASS OR A FUNCTION THE HOLE DEFINES IS A NAME THE HOLE LEAVES BEHIND. Measured on a
    coupled elasticity round: the FEBio elastic contract's list named four variables and
    left out `Mesh`, the class its served lines build and use through seven members, and
    said "That is the whole contract"; five workers re-built a 339-line region against an
    API with no name, and one quit on "the Mesh class". A name bound only inside a function
    or class body of the hole is that body's, not the hole's (it listed a method's local `w`)."""
    import ast
    import re
    try:
        tree = ast.parse(cut)
    except SyntaxError:
        out = {n: "value" for n in re.findall(r"^\s*([A-Za-z_]\w*)\s*=", cut, re.M)}
        out.update({n: ("class" if kind == "class" else "function")
                    for kind, n in re.findall(r"^\s*(def|class)\s+([A-Za-z_]\w*)", cut, re.M)})
        return out
    out: dict = {}

    def walk(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[ch.name] = "function"
                continue                                # its body is its own scope
            if isinstance(ch, ast.ClassDef):
                out[ch.name] = "class"
                continue
            if isinstance(ch, ast.Lambda):
                continue
            if isinstance(ch, (ast.Import, ast.ImportFrom)):
                for al in ch.names:
                    out.setdefault((al.asname or al.name).split(".")[0], "value")
            if isinstance(ch, ast.Name) and isinstance(ch.ctx, ast.Store):
                out.setdefault(ch.id, "value")
            walk(ch)
    walk(tree)
    return out


def _bound_before_every_use(tree, name: str) -> bool:
    """True when every read of `name` in the served code comes after a line of the served
    code that binds it, in the same block or one around it (the served lines then do not
    need it from the hole: measured, `w = mesh.iface_weights()` sat right above its use and
    `w` was listed as the hole's)."""
    import ast
    parent = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            parent[ch] = node

    def binds(stmt) -> bool:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            return any(isinstance(t, ast.Name) and t.id == name for tg in targets for t in ast.walk(tg))
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return stmt.name == name
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            return any((al.asname or al.name).split(".")[0] == name for al in stmt.names)
        return False

    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]
    if not loads:
        return False
    for load in loads:
        cur, dominated = load, False
        while cur in parent and not dominated:
            up = parent[cur]
            if isinstance(up, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                break
            for field in ("body", "orelse", "finalbody"):
                block = getattr(up, field, None)
                if isinstance(block, list) and cur in block:
                    dominated = any(binds(st) for st in block[:block.index(cur)])
            cur = up
        if not dominated:
            return False
    return True


def _reconstruction_contract(served: str, original: str) -> list:
    """Names the elided solve defined that the surviving code still uses.

    Cutting the solve out is the point; leaving the agent to GUESS what the
    hole was supposed to define is not. The surviving code indexes `iface_dofs`
    and assembles `a`, so those names must come back — but nothing told the
    agent which ones, or how many. Measured on the served skfem participant, 22
    names are used and never defined, and the payload does not name one of them
    as the reader's responsibility. The measured consequence: 73% of the
    coupled openPASO runs that gave up never got both sides to exchange data once,
    dying in a write-run-error-rewrite loop on the participant script.

    Derived from the same markers that do the elision, so it cannot go stale.
    It leaks no physics: it is the list of names (variables, and the classes and
    functions the hole defines) the code below the hole already mentions, which
    the agent can read off the script anyway -- just not without
    reverse-engineering it first.
    """
    import ast
    import re

    if not _cut_regions(original):
        return []
    defined = _cut_definitions(_cut_source(original))
    needed = _free_loads(served)
    if needed is None:
        needed = set(re.findall(r"\b([A-Za-z_]\w*)\b", served))
        tree = None
    else:
        tree = ast.parse(served)
    names = [n for n in sorted(set(defined) & needed) if not n.startswith("_")]
    if tree is not None:
        names = [n for n in names if not _bound_before_every_use(tree, n)]
    return names


def _definition_uses(served: str, original: str, names: list) -> list:
    """Comment lines saying how the served code uses each class or function the hole
    defines: a class's members as the served lines read them, a function's call.
    Quoted from the served code itself, so they say nothing it does not."""
    import ast
    try:
        tree = ast.parse(served)
    except SyntaxError:
        return []
    kinds = _cut_definitions(_cut_source(original))
    lines = served.splitlines()
    forbidden = ("solve(", "assemble(", "LinearProblem", "spsolve", "factorized(", "splu(")

    def quote(node) -> str:
        seg = ast.get_source_segment(served, node) or ""
        ln = lines[node.lineno - 1].strip() if 0 < node.lineno <= len(lines) else seg
        text = ln if len(ln) <= 72 and not any(f in ln for f in forbidden) else seg
        return " ".join(text.split())[:72]

    import textwrap

    def say(text: str) -> list:
        import re as _re
        kept = _re.sub(r"`[^`]*`", lambda m: m.group(0).replace(" ", "\x00"), text)   # a quote stays whole
        return ["# " + ln.replace("\x00", " ")
                for ln in textwrap.wrap(kept, 74, break_on_hyphens=False, break_long_words=False)]

    out = []
    for name in names:
        kind = kinds.get(name)
        if kind == "class":
            made = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name) and n.value.func.id == name
                    and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)]
            owners = {m.targets[0].id for m in made} | {name}
            members: dict = {}
            for n in sorted((n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                             and isinstance(n.value, ast.Name) and n.value.id in owners),
                            key=lambda n: (n.lineno, n.col_offset)):
                members.setdefault(f"{n.value.id}.{n.attr}", n)
            how = (f"make it as `{quote(made[0])}`" if made else "use it")
            out += say(f"{name} is a class the hole defines: the served lines {how}"
                       + (" and read these members of it, each quoted where first used:" if members else "."))
            if members:
                width = max(len(m) for m in members) + 2
                out += [f"#     {m:<{width}}{quote(n)}" for m, n in members.items()]
        elif kind == "function":
            calls = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Name) and n.func.id == name),
                           key=lambda n: (n.lineno, n.col_offset))
            if calls:
                seg = " ".join((ast.get_source_segment(served, calls[0]) or name).split())[:72]
                out += say(f"{name} is a function the hole defines: the served lines call it as `{seg}`.")
            else:
                out += say(f"{name} is a function the hole defines, and the served lines pass it on as a value.")
    return out


def _free_loads(src: str):
    """Names the code READS without binding them first in an enclosing loop.

    `for k, vtx in enumerate(iface_v)` below the hole reads k and vtx inside
    that loop, which binds them, so they are not the hole's to define --
    listing them told the agent a loop counter was part of the contract. The
    same letter read OUTSIDE any loop that binds it (the bilinear form `a`
    beside a `for a in ...` elsewhere) still counts. None when it cannot parse.
    """
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    parent = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            parent[ch] = node

    def _bound(name_node):
        nm, cur = name_node.id, name_node
        while cur in parent:
            up = parent[cur]
            if isinstance(up, (ast.For, ast.AsyncFor)) and cur is not up.iter and cur is not up.target:
                if any(isinstance(t, ast.Name) and t.id == nm for t in ast.walk(up.target)):
                    return True
            if isinstance(up, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                if any(isinstance(t, ast.Name) and t.id == nm
                       for g in up.generators for t in ast.walk(g.target)):
                    return True
            # A NAME A SERVED FUNCTION BINDS ITSELF (an argument, or assigned in its body)
            # is that function's, not the hole's (measured: a local `d` of a served Kratos
            # function was listed as a name the solve must leave behind).
            if isinstance(up, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                _a = up.args
                if nm in {x.arg for x in list(_a.args) + list(_a.posonlyargs) + list(_a.kwonlyargs)
                          + [v for v in (_a.vararg, _a.kwarg) if v is not None]}:
                    return True
                if any(isinstance(t, ast.Name) and t.id == nm and isinstance(t.ctx, ast.Store)
                       for t in ast.walk(up)):
                    return True
            cur = up
        return False

    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and not _bound(n)}


def _worded_names(names: list, words: dict) -> list:
    """Comment rows saying what each listed name must be, from the contract's own words. A row
    that continues one is indented and never a lone identifier (the write check reads a lone
    identifier on an indented comment row as a listed name)."""
    import textwrap
    rows = []
    width = max((len(n) for n in names if n in words), default=0)
    for n in names:
        if n not in words:
            continue
        lines = textwrap.wrap(words[n], 72 - width, break_on_hyphens=False, break_long_words=False)
        if len(lines) > 1 and len(lines[-1].split()) == 1:
            lines[-2] += " " + lines.pop()
        rows.append(f"#   {n:<{width}}  {lines[0]}")
        rows += [f"#   {'':<{width}}  {ln}" for ln in lines[1:]]
    return rows


def _append_reconstruction_contract(served: str, original: str, words: dict | None = None) -> str:
    names = _reconstruction_contract(served, original)
    if not names:
        return served
    uses = _definition_uses(served, original, names)
    worded = _worded_names(names, words or {})
    return served + (
        "\n# ── WHAT YOUR SOLVE MUST LEAVE BEHIND ─────────────────────────\n"
        "# The code above and below the elided block uses these names. Your\n"
        "# block has to define every one of them, or the rest will not run:\n"
        "#\n"
        + "".join(f"#     {n}\n" for n in names) +
        "#\n"
        + "".join(f"{ln}\n" for ln in uses)
        + ("#\n" if uses else "")
        + ("# What each one must be:\n" + "".join(f"{ln}\n" for ln in worded) + "#\n"
           "# That is the whole contract: the surviving lines show where each\n"
           "# one is indexed, assembled or written out. openPASO does not serve the\n"
           "# solve itself, but it will not make you guess which names the hole\n"
           "# was filling.\n" if worded else
           "# That is the whole contract. Read the surviving lines to see the\n"
           "# shape each one has to have -- they are already indexed, assembled\n"
           "# or written out there. openPASO does not serve the solve itself, but\n"
           "# it will not make you guess which names the hole was filling.\n"))

# ══════════════════════════════════════════════════════════════════════════
# CORE — served by knowledge(topic='coupling') with no solver
# ══════════════════════════════════════════════════════════════════════════

# audit_results is mentioned in ZERO of the eight coupling payloads and was
# called by 7 of 224 coupled runs (3%). It catches, from the agent's own files
# and with no answer key: a missing NDOF line, a short or NaN-leading residual
# history, a near-zero field, duplicate deliverables, a hole in the level
# sequence, a nearest-node sampler. Every one of those has cost real runs. It
# belongs at the FRONT, not unmentioned.
_SELF_CHECK_FIRST = """
BEFORE YOU DELIVER, CALL audit_results(work_dir=...)
──────────────────────────────────────────────────
It reads only your own files -- no reference solution -- and names the defects
that most often sink a coupled result set:

    a level with field files but no per-level run log carrying the DOF-count
        line your task asks for
    a residual history that is too short, or starts with the NaN `couple`
        returns as history[0]
    a field that peaks below 1e-8 (the load never reached the solver)
    the same deliverable present in two places with different contents
    a hole in the level sequence, or fewer than three levels
    a sampler that reads the nearest NODE instead of interpolating, which caps
        your measured convergence order at 1 however good the solve was

It costs one call.

"""

_CONTRACT = '''\
## 1. THE PARTICIPANT CONTRACT — this is the whole interface

`couple` runs a partitioned fixed-point iteration over N>=2 participants. A
participant is ANY runnable command. Every iteration, for every participant,
the driver does exactly three things:

  1. writes `<work_dir>/imports.json`
  2. runs your `command` with `cwd = <work_dir>`
  3. reads `<work_dir>/exports.json`

Nothing else is passed. No arguments, no environment, no stdin. Your script
gets its problem definition from constants written into the script itself.

REQUIRED of every participant script:
  * read `imports.json`. It is written EVERY iteration and is exactly `{}` on
    iteration 1, so you MUST have a fallback initial value for whatever you
    import. It is never deleted, so a stale one from a previous attempt is
    still there when you run the script by hand — delete it first;
  * write `exports.json` before exiting;
  * export the SAME number of points, in the SAME order, on EVERY iteration,
    and keep `normal_fluxes` consistently present or consistently absent. The
    driver relaxes export vectors element by element; a changed length is
    caught and reported, but it ends the run;
  * write `exports.json` LAST, only after the solve succeeded, AND EXIT 0. The
    driver requires both: a missing file ends the run, and so does a non-zero
    exit code even when a complete `exports.json` is sitting there — a solver
    that diverges commonly writes its last iterate and then aborts, and
    coupling on that output produced a converged-looking result built on a
    crashed participant. A TRUNCATED file is caught separately, as bad JSON.
    So do not use a non-zero exit to signal anything but failure.

NOT done for you (these are the four things agents get wrong):
  * the driver does NOT copy your script into `work_dir` — write the script
    file into `work_dir` yourself, then name it bare in `command`;
  * the driver does NOT interpolate between the two meshes — each participant
    maps the partner's samples onto its own interface points;
  * the driver does NOT convert units, sign conventions or field names;
  * `work_dir` must be an ABSOLUTE path, and a relative one is REJECTED — it
    would resolve against the server's own directory, which you cannot see.

## 2. imports.json / exports.json — the exact shapes

`imports.json` is a dict keyed by PARTNER NAME:

    {"<partner name>": {<InterfaceData>}, ...}

only containing the partners you listed in `imports_from`. `exports.json` is
ONE InterfaceData object, not a dict of them:

A COMPLETE, VALID exports.json — copy this and substitute your own numbers. It
is deliberately free of annotation, because a `#` comment is not legal JSON and
a file with one in it fails to parse:

    {"field_name": "temperature",
     "coordinates": [[1.0, 0.0], [1.0, 0.05], [1.0, 0.1]],
     "values": [305.2, 304.8, 304.1],
     "normal_fluxes": [-1.52, -1.49, -1.44],
     "n_points": 3}

What each key is for (this list is NOT part of the file):
  * `field_name`     REQUIRED key, free-form value
  * `coordinates`    REQUIRED — YOUR interface points, one [x, y] per point
  * `values`         REQUIRED as a KEY — one per point, same order as
                     `coordinates`. A side whose ROLE exports only a flux
                     (e.g. a Dirichlet-role side returning its outward flux)
                     writes `"values": []` — legal, and the driver's
                     per-block change checks then skip the values block.
                     Echoing the trace you IMPOSED back as `values` is what
                     trips those checks instead.
  * `normal_fluxes`  optional, one per point
  * `n_points`       optional label, never read

THREE KEYS ARE REQUIRED: `field_name`, `coordinates`, `values`. `field_name`
is a free-form label whose VALUE is never interpreted, but leaving the KEY out
is a hard error reported as `bad exports.json`. `n_points` is ignored entirely.

`normal_fluxes` is optional — but if EITHER side omits it, the interface
conservation check cannot run at all, and the result says so. Supply it
whenever a flux exists; it is the only guard against a coupling that converges
to a non-conservative answer.

Nothing validates the SHAPES. `values` of a different length than
`coordinates`, or a flat `coordinates` list, is accepted and gives a wrong
answer quietly. Check them in your own script.

BOTH `values` AND `normal_fluxes` are relaxed and BOTH enter the convergence
residual. Exporting a large-magnitude flux next to a small-magnitude value
makes the residual mostly about the flux; that is usually what you want for a
Dirichlet-Neumann coupling, but know that it happens.

## 3. How you call it

    couple(participants='[
      {"name": "left",  "command": ["<interpreter>", "participant_left.py"],
       "work_dir": "/abs/path/run/left",  "imports_from": ["right"],
       "timeout": 900},
      {"name": "right", "command": ["<interpreter>", "participant_right.py"],
       "work_dir": "/abs/path/run/right", "imports_from": ["left"],
       "timeout": 900}]',
      max_iter=60, tol=1e-8, accelerator="auto", theta=0.5,
      critic_approved=True)

  * `name` — how the partner finds this participant's data inside imports.json.
  * `command` — argv list. Get the interpreter or binary for a backend from
    `discover(query='list')`, which prints it per backend on THIS install.
    Never hard-code an interpreter path from an example.
  * `work_dir` — absolute; created if missing; your script must already be in it.
  * `imports_from` — names of the partners whose exports this one consumes.
    Omit a name and that participant simply never sees that partner's data.
  * `timeout` — seconds per participant call (default 3600). A compiled code
    that hangs will otherwise stall the whole coupling.
  * `data_files` — a list of ABSOLUTE paths to files your solver opens (mesh,
    species, surface, config). They are copied into `work_dir` once, before the
    iteration starts, and a path that does not exist is a LOUD setup error
    rather than a solver dying mid-iteration on 'Cannot open ...'. Declaring a
    file here also binds it into the critic review, so rewriting it after the
    review invalidates the approval. You may still copy files in yourself;
    `data_files` is the supported way and the one that gets both of those.
  * `theta` must be in (0, 1]; `accelerator` is "auto" (the default: Aitken for a
    single-field exchange, Anderson mixing for a multi-field one such as [T, ux, uy]),
    "aitken", "anderson" or "constant". Anything else is rejected with an error message.
  * `noise_replicates` — ONLY when a participant is a Monte-Carlo / sampled
    estimator (DSMC, a stochastic solver, anything whose answer to the same
    question differs run to run). Set it to 2 or more and the driver runs every
    participant that many times on the SAME imports BEFORE iterating, measures
    the residual between independent replicates, and judges convergence against
    max(tol, that floor). See section 4a. It costs
    `noise_replicates` extra solves per participant. Leave it at 0 for
    deterministic solvers — where it does nothing anyway.
  * `noise_floor` — declare a floor you established yourself instead of (or on
    top of) measuring one. `noise_block` — how many consecutive residuals must
    AVERAGE below the criterion before the run stops; only in effect when a
    non-zero floor is.

Returns JSON with `converged`, `iterations`, `residual`, `history`, per-
participant `exports`, a `validation` block, and openPASO's `verification` /
`trustworthy_result` verdict — plus `noise_floor`, `tol_effective`,
`stopped_at_noise_floor` and `noise_notes` whenever a floor was in play. Two
things about it:
  * the `exports` returned are the RELAXED blend the driver holds, not the last
    raw output of your solver. On a converged run the difference is below the
    tolerance; on a failed run they are a mixture of two iterations, which is
    one more reason not to report a non-converged run as a result;
  * `trustworthy_result` is false until an independent critic review for THIS
    exact set of arguments is on record. `critic_approved=True` on its own does
    nothing — openPASO looks the review up rather than believing the flag. Call
    `submit_critic_review(solver="couple", coupling_args=<the same arguments as
    a JSON object>, findings=<what the critic concluded>)` first.

A run that did not converge is reported as FAILURE — never report its numbers
as a result. A run that DID converge is a result even if a downstream check
complains about it — see section 3b, which is the difference between a run
that counts and a wasted one.

## 3a. THE ONE BUG THAT CONVERGES TO THE WRONG ANSWER

A defect that keeps coming back: exporting the raw traction instead of
the NEGATED outward normal flux. That flips the sign the partner applies, and
the coupling then converges, smoothly, with a clean residual history, to the
wrong answer. Nothing errors and nothing looks wrong.

So state it once, plainly: every participant exports its flux or traction with
respect to ITS OWN outward normal, the two sides of an interface carry
OPPOSITE signs, and the partner's number is applied UNCHANGED — no minus sign
anywhere in the application. If you ever flip that sign to make the fields
look right, you have built a coupling that drives the quantity the wrong way
across the interface and still converges.

You write the participant. The sections below give the handshake, this sign
convention, and the flux recovery the interface check compares against; the
solve itself is yours.

## 3b. FROM A CONVERGED COUPLING TO A DELIVERED ANSWER

Getting the iteration to converge is the hard part and it is not the last
part. Among the recorded runs, six produced converged two-code couplings
— residuals to 1e-7 and better, both participants responsive — and every one
of them counted for nothing. None of them lost on physics. They lost on the
four points below, none of which was written down anywhere.

ONE `couple` CALL IS ONE MESH. A REFINEMENT STUDY IS N CALLS.
A participant takes no arguments, no environment and no stdin, so the mesh
level lives inside the script. Give each level its OWN work_dir with its own
copy of both scripts (or write a small `level.json` into each work_dir before
the call and read it at the top of the script), and call `couple` once per
level. Never reuse a work_dir between levels: imports.json and exports.json
are not deleted, so a stale pair from level k-1 silently seeds level k. Budget
for the whole ladder before you start — the first converged level is a third
of the work, not the end of it.

THE FIELD DOES NOT COME BACK THROUGH THE DRIVER.
`couple` returns the INTERFACE state, and the exports it hands back are the
driver's relaxed blend, not your solve. Your volume solution never passes
through it. So every iteration, after you solve and before you write
exports.json, write your own field to your own file in work_dir; the last one
written is the converged one. The `history` array in the return is the
coupling residual history and is the only place it exists — copy it out if you
need it.

A CONVERGED RUN WITH A FAILED CONSERVATION CHECK IS STILL A RESULT -- AND NOT
YET A RIGHT ONE. Order of operations the moment `converged: true` arrives:
(1) write every deliverable from the state you have, now; (2) then read the
finding and test it against the NEXT level; (3) re-run and overwrite if you
fix it. "Interface flux NOT balanced" at one level is one number, and what it
means is decided by refinement: discretisation error shrinks by about 4x per
mesh halving (2x for a first-order recovery); a wrong sign, scaling or
missing term in what one side exports stays or grows, and then every field
that depends on that exchange is wrong at every level however cleanly the
iteration converged. Neither reading is a reason to discard the run or to
declare you could not complete it; the second is a reason not to call the
result mesh-independent. NOT VERIFIED and NOT A RESULT are different
statements: a result handed in with its named caveat is a result, a claim of
convergence over a caveat that did not shrink is not.

THE POINTS YOU EXCHANGE ARE NOT THE POINTS YOU REPORT.
Each side exports its own interface nodes. The two sides having different
counts is the contract working, not a bug — do not rewrite your participants
to make the counts agree. Anything a task asks you to report at prescribed
locations is obtained afterwards, by interpolating each side's converged
solution on that side, with that side's own material and its own outward
normal. Where the interface meets a constrained outer boundary, those end
points belong to the outer boundary and are not interface data on either side.
'''

_DRIVER_BEHAVIOUR = '''\
## 4. How the iteration actually behaves — read before choosing theta

THE DRIVER IS JACOBI, NOT GAUSS-SEIDEL. Inside one iteration, every participant
reads the PREVIOUS iteration's exports. Participant B does not see the export
participant A produced moments earlier in the same iteration. Ordering the
participants differently changes nothing.

THE DRIVER RELAXES EVERY PARTICIPANT. Each participant's own export vector is
blended with its own previous export:

    export_relaxed = (1 - theta) * export_previous + theta * export_new

so a two-participant loop applies relaxation TWICE per cycle, once on each
side. This is why a linear Dirichlet-Neumann coupling that textbooks solve in
one Gauss-Seidel step converges only geometrically here.

RESIDUAL — AND WHY IT HAS A FLOOR. The reported residual compares each
participant's RAW new export against its PREVIOUS RELAXED one, normalised by
the raw magnitude, stacked over all participants. It is not a physical error.
Iteration 1 has nothing to compare against and is recorded as NaN.

The consequence matters: even a participant whose raw output has ALREADY
stopped changing still shows a residual falling like (1-theta)^k, purely
because the relaxed value is still catching up to the raw one. So with a
CONSTANT theta the residual's decay rate is bounded by (1-theta) per iteration
once the raw output has settled, and the iteration count you need is roughly

    log(tol / d0) / log(1 - theta)

where d0 is the INITIAL mismatch measured the way the residual is — relative to
the field magnitude. EVALUATE IT FOR YOUR OWN theta AND tol; it is three
keystrokes and there is no table to look up:

    theta=0.5, tol=1e-8, d0=1  ->  log(1e-8)/log(0.5)  ~  27 iterations

That is a value of the formula, not something anybody observed, and it is an
ORDER-OF-MAGNITUDE SIZING GUIDE rather than a lower bound — a field with a large
offset, temperatures around 300 K whose interface value is wrong by a few K,
starts at a d0 of a few percent and gets there in fewer. What matters is the
shape: the count grows without limit as theta shrinks, so give max_iter generous
headroom, because under-budgeting looks exactly like a physics failure — a run
that stops at max_iter=20 with a small theta never had a chance. (With
accelerator="aitken" theta moves, so the rate moves with it, but the same
mechanism is there.)

`accelerator`: **"aitken" is the default and you should normally keep it up to a
MODERATE conductance ratio, where it removes theta as a failure mode entirely.
Once you know rho is 4 or more it stops being the right choice — at rho = 4 it
costs three times the iterations, and from rho = 6 up it DIVERGES at every theta
measured, including ones a constant theta converges on. Above rho = 4 use
"constant" at theta = 1/(1+rho). The crossover was measured; it is below.**
  * "aitken" — ONE theta for the whole interface state, recomputed every
    iteration, starting from the theta you pass and clamped into [0.05, 1.0].
    There is no per-participant and no per-field theta: Aitken's derivation is
    for a single sequence extrapolated from the composite fixed-point map, and
    giving each participant its own theta relaxes the two halves of one coupled
    system by different amounts — on a Dirichlet-Neumann split that drove the
    two thetas apart to opposite clamps and made the iteration diverge where a
    constant theta converged. The two fallback paths inside the update — the
    first iteration, where there is no previous residual, and a degenerate
    denominator — hold the previous theta, clamped into the same [0.05, 1.0].
    THIS IS THE DEFAULT AND YOU SHOULD NORMALLY KEEP IT — up to a moderate
    ratio, and the boundary is measured, not a feeling.

    WHAT IT IS WORTH, AND WHERE IT STOPS. Measured over rho in
    {1/4, 1/2, 1, 2, 4, 6, 9} x theta from 0.1 to 1.0 in steps of 0.1, both
    accelerators, the SAME max_iter=300 and the SAME tol=1e-4, on two real
    coupled codes exchanging across a non-matching interface — 70 settings, 140
    runs. Re-run it with `scripts/sweep_accelerators.py`; every number below is
    that script's output.

      * UP TO rho = 2 IT SOLVED EVERY SETTING — all 40 of them, every theta from
        0.1 to 1.0 — including theta = 1.0, where a constant theta oscillates
        forever at rho = 1 and reaches 1.1e+44 at rho = 2. That is the
        protection against a theta chosen too large, and inside this range it is
        complete;
      * at rho = 4 it solved only for theta <= 0.4, which is exactly the
        constant scheme's own stability limit 2/(1+rho) — so in the sense of
        MEETING tol it bought nothing beyond that limit. Above it the run does
        land on the closed-form interface value (1.2e-05 to 2.4e-05 relative)
        but its residual never reaches tol inside 300 iterations, while the
        constant arm runs away to between 4.1e+13 and 1.7e+89;
      * at rho = 6 and rho = 9 it solved NOTHING and landed on NOTHING. At every
        theta on the grid it diverged, by 4.1e+03 to 2.0e+17 relative;
      * IT CAN DESTROY A SETTING THAT WORKS, and this is the part worth
        memorising. At rho = 6, theta = 0.1 and 0.2, and at rho = 9,
        theta = 0.1, a constant theta converged — in 142, 156 and 196 iterations
        — and the DEFAULT diverged from the same start. Worse: at rho = 6 the
        theta this section tells you to compute, 1/(1+rho) = 0.143, DIVERGES
        under "aitken" (1.0e+04 relative after 300 iterations) and CONVERGES
        under "constant" in 130 iterations. The amplification factor there is
        0.926 — the constant iteration is comfortably stable and the
        ACCELERATOR is what breaks it;
      * over the whole grid Aitken matched or beat the same constant theta in 65
        of the 70 settings, and solved where that same constant theta ran away
        in 4 of 70, with 6 further settings where it landed on the right value
        without its residual certifying it. All five of its losses are at
        rho = 6 and rho = 9.

    WHY IT STOPS, so you can predict it rather than discover it. The Jacobi
    Dirichlet-Neumann map's iteration matrix has PURELY IMAGINARY eigenvalues
    +- i*sqrt(rho) — the same fact as the amplification
    sqrt((1-theta)^2 + rho*theta^2) in the next section. The residual turns by
    about a right angle each iteration instead of shrinking along a fixed
    direction, so Aitken's scalar secant extrapolates along a direction that
    does not exist, and the larger rho is the less there is to extrapolate. The
    same mechanism is what makes the default fail on a strongly coupled two-way
    thermo-mechanical coupling.

    SO: keep "aitken" when you cannot estimate rho, or when you believe the two
    sides are within a factor of a few. Once you know rho >= 4, pass
    accelerator="constant" with theta = 1/(1+rho) — and consider swapping which
    side is Dirichlet, which replaces rho by 1/rho and puts you back inside the
    range where the default works.
  * "constant" — theta fixed at exactly what you passed. Predictable and
    reproducible, which makes it the right tool for working out what the
    iteration is doing, and unforgiving: above the stability limit for your rho
    it diverges instead of adapting. Below that limit it is the MORE RELIABLE of
    the two at an unbalanced ratio. On the grid above it solved 41 of the 70
    settings against Aitken's 44, but the three it solved that Aitken did not
    are all strongly unbalanced splits, and in those Aitken did not merely stall
    — it diverged by 4 to 16 orders of magnitude. Where both solved, Aitken was
    usually the cheaper (median 16 iterations against 28.5); rho = 4 is the
    exception, at 283-291 iterations against the constant arm's 83-117.

WHAT "AITKEN" MEANS HERE. The update is the classical Aitken dynamic-relaxation
recurrence on the global interface residual r_k = G(x_k) - x_k:

    theta_k = -theta_{k-1} * (r_{k-1} . (r_k - r_{k-1})) / ||r_k - r_{k-1}||^2

with the result clamped into [0.05, 1.0]. It IS given the previous RESIDUAL,
which is the operand the derivation calls for — an earlier version of this
driver handed it the previous RAW EXPORT instead, which makes theta an
arbitrary number inside the clamp with no relation to the iteration, and the
only symptom was slower convergence on correct setups. Two things still do not
follow from the textbook derivation: the clamp is not part of it, and the
convergence RATE is not guaranteed, because this is a vector fixed point
extrapolated by one scalar. Everything stated above is what this implementation
was measured to do. If you need a specific acceleration scheme rather than this
one, drive the coupling with `couple_precice` and a `serial-implicit` scheme.

### Choosing theta — this maps to the real `theta` parameter

For a two-participant Dirichlet-Neumann split the iteration is linear in the
interface unknowns, and the driver's Jacobi+relaxation loop has amplification
factor  sqrt((1-theta)^2 + rho*theta^2), where

    rho = (interface conductance of the DIRICHLET-side subdomain)
          / (interface conductance of the NEUMANN-side subdomain)

and "interface conductance" is material coefficient / distance from the
interface to that subdomain's own outer boundary (k/d for conduction, EA/L for
a bar, and so on — it is the subdomain's stiffness as seen from the interface).
That gives four facts you can act on:

  * the best theta is  **theta ~ 1 / (1 + rho)** — 0.5 when the two sides are
    balanced, smaller when the Dirichlet side is the stiffer one, larger when it
    is the softer one. This is the one number in this section worth computing
    before you run anything: swept over rho from 1/4 to 9, the fastest constant
    theta was 1/(1+rho) at EVERY ratio;
  * the amplification factor above is below 1 exactly when
    **theta < 2 / (1 + rho)**, which is TWICE the optimum. So what converges is
    an INTERVAL, not a point: everything below double the value you just
    computed. The interval NARROWS as the split gets more unbalanced, which is
    why a coarse sweep at a strongly unbalanced ratio can turn up only one value
    that works — the interval got smaller than the spacing, not the method.
    Above the limit the iteration diverges, and halving theta always brings you
    back inside it;
  * theta = 1.0 NEVER converges at rho = 1 and DIVERGES for rho > 1. It is not
    a "no relaxation, exact for linear problems" setting on this driver;
  * convergence is fastest when the DIRICHLET side is the SOFTER / LESS
    CONDUCTIVE subdomain. If a coupling converges too slowly to be practical,
    SWAP WHICH SIDE IS DIRICHLET before doing anything else — that replaces rho
    by 1/rho and is usually a bigger win than any theta.

Observed on this driver, running real two-code couplings:
  * at rho = 1, theta = 0.5 converges and theta = 1.0 oscillates forever
    WITHOUT blowing up — the interface value simply never settles;
  * at rho = 4, theta = 0.5 with a CONSTANT accelerator DIVERGES — the interface
    values run away by many orders of magnitude (measured: 4.1e+13 relative at
    300 iterations) and the conservation check fires — while theta = 0.2
    converges in 83. Nothing warns you in advance: a diverging coupling looks
    like a converging one for the first few iterations. The default "aitken"
    does NOT diverge on this case, and it does not solve it either: it lands on
    the closed-form interface value to 1.2e-05 relative and its residual is
    still 1.4e-04 against tol = 1e-4 after 300 iterations. The reason is worth
    knowing, because it is how a stalled Aitken run looks in general — its own
    adaptation drove theta onto the 0.05 clamp for 60 of 299 adaptations, and
    once the raw output has settled the residual can only fall like (1-theta)
    per iteration, measured here at 0.968. So the answer was found and the
    convergence test could not certify it inside the budget. Do not read the
    constant-theta stability limit as a property of the tool's default — and do
    not read the default as a rescue at this ratio either;
  * at rho = 6 and above the default is actively WORSE than a constant theta
    inside the stability limit: theta = 1/(1+rho) = 0.143 converges in 130
    iterations with accelerator="constant" and diverges to 1.0e+04 relative with
    the default. Above rho = 4, choose "constant";
  * for one asymmetric split, the SAME problem with the SAME tolerance failed
    to converge inside the iteration budget with the stiff subdomain on the
    Dirichlet side, and converged comfortably inside it once the two roles were
    swapped and theta set to 1/(1+rho) for the new rho. Choosing the side is
    the cheapest tuning knob you have, and it is free — it costs one edit to
    each script's `SIDE` and `T_OUTER`.

  | Symptom                                       | Do this                    |
  | first try, know nothing                       | estimate rho, set theta=1/(1+rho); keep accelerator="aitken" if rho < 4, use "constant" if rho >= 4 |
  | first try, cannot estimate rho at all         | theta=0.5, keep accelerator="aitken" |
  | rho is 4 or more (a stiff side against a soft one) | accelerator="constant", theta=1/(1+rho) — the default DIVERGES here at every theta measured from rho=6 up |
  | residual falls steadily but slowly            | keep theta, raise max_iter; then swap which side is Dirichlet |
  | residual flat or oscillating in sign          | halve theta                |
  | residual GROWING, values exploding            | halve theta, and check the flux sign convention (section 5); if accelerator="aitken", switch to "constant" at theta=1/(1+rho) BEFORE halving — at an unbalanced ratio the accelerator is the likelier cause |
  | want to see what the iteration is doing       | same theta, accelerator="constant" — reproducible, no adaptation |
  | converged, but you want it faster             | put Dirichlet on the softer subdomain, theta = 1/(1+rho) |

There is no theta that makes this driver converge in one step for a two-code
Dirichlet-Neumann split, and what to budget is set by rho: measured at the
tool's own tol = 1e-6 with accelerator="constant" and theta = 1/(1+rho), a
two-subdomain conduction split took 11 iterations at rho = 1/10, 37 at rho = 1,
66 at rho = 2, 123 at rho = 4, 299 at rho = 10 and 3311 at rho = 100 — roughly
LINEAR in rho once the Dirichlet side is the stiffer one, and unchanged to the
iteration over a fourfold interface refinement. So size max_iter from rho and
not from the mesh: 100 for rho <= 1, 400 for rho <= 10, with the
per-participant `timeout` sized for that many solves — the default
max_iter = 50 is already short at rho = 2. Beyond rho = 10 the budget is the
wrong knob: theta = 0.5 diverges from rho = 4 up and the DEFAULT accelerator
from rho = 10 up, so SWAP WHICH SIDE IS DIRICHLET, which replaces rho by 1/rho
and puts every ratio measured below 1 inside 25 iterations.

### 4a. A STOCHASTIC PARTICIPANT — the residual has a floor and `tol` cannot cross it

If ANY participant is a Monte-Carlo or otherwise sampled estimator — a DSMC
code, a stochastic solver, anything that answers the same question slightly
differently each time it is asked — then read this before you set `tol`.

The residual is the CHANGE in the export vector between iterations. A sampled
participant changes its export every run whether or not the physics moved, so
the residual cannot fall below the size of that scatter, however well the
coupling has converged. A `tol` underneath the floor is unreachable BY
CONSTRUCTION: the run always ends "did not converge", on a coupling that is
right. Sizing `tol` by guesswork does not fix it — too tight and every run fails,
too loose and you have declared victory at a number you cannot defend.

`noise_replicates` MEASURES the floor instead. The driver runs every participant
that many times on the SAME imports and evaluates its own residual expression
across the replicates, so the floor is the residual a perfectly converged run
would still report. Convergence is judged against max(tol, floor), over a block
mean of the last few residuals so that one lucky dip into the noise cannot end
the run. The result carries `noise_floor` and `stopped_at_noise_floor`.

USE 4 OR MORE, not the minimum of 2. The floor is itself an estimate and a small
one is a bad estimate: three replicates give three samples, and the same
coupling can measure a floor several times larger or smaller from one attempt to
the next purely from that scatter. The driver adds a note below six samples.

THE MEASUREMENT HAPPENS TWICE, and the two are not the same number. Before the
loop there is no previous relaxed vector to compare against, so all that can be
measured is the scatter BETWEEN independent answers — a LOWER BOUND, because the
loop compares against a lagged relaxed average carrying its own accumulated
noise, and because noise that has propagated through a partner over earlier
iterations is not in it. That bound is used to avoid a pointless long run. If the
loop still finishes un-converged, the floor is re-measured with the participants
in their FINAL state, against the very vector the loop compares to — that one is
the residual the loop actually reports, measured — and the verdict is re-judged.
The re-measurement is paid only on failure.

THREE THINGS THAT FOLLOW, and they are the whole discipline:
  * READ `noise_floor` BEFORE APPLYING ANY TOLERANCE TO THE RESULT. An
    acceptance or agreement tolerance tighter than the floor is measuring the
    sampler, not the coupling;
  * a floor of EXACTLY ZERO from a Monte-Carlo participant means its SEED IS
    FIXED. The run is repeatable, not converged, and a residual that falls under
    a fixed seed is no evidence it would fall under another. `noise_notes` says
    so explicitly. Vary the seed between runs if you want the floor to mean
    anything;
  * the floor answers "can this sampler still see the iteration moving", NOT
    "has the iteration finished". A physics drift smaller per iteration than the
    sampling noise but accumulating over many of them is invisible to it.
    Increase the sampling (which lowers the floor) if you need to see smaller
    steps.

For a deterministic solver the replicates come back bit-identical, the floor is
zero, max(tol, 0) is tol and nothing about the run changes. There is no reason
to set it there, and no harm if you do.

If you need genuine Gauss-Seidel sub-iteration inside a time window, that is
what `couple_precice` with a `serial-implicit` scheme provides — see
`knowledge(topic='precice')`.
'''

_SIGNS = '''\
## 5. Interface flux: TWO different quantities, two different signs

Confusing these is the mistake that produces a converged coupling which openPASO
then stamps NOT VERIFIED, with nothing in the output explaining why.

**(1) The BC VALUE you APPLY in the receiving code — the SAME number.**
Let subdomain A have outward normal n_A at the interface. The flux density A
loses through the interface is

    q_out = -k * dT/dn_A                            [W/m^2 in 2D]

ANISOTROPIC MATERIAL: if the conductivity is a TENSOR K rather than a scalar
k, the transmitted quantity is the full normal component of the flux vector,

    q_out = -(K grad T) . n_A       NOT  -K_nn * dT/dn_A

and the two are different numbers whenever K has off-diagonal entries: the
flux vector is not parallel to the normal, so the tangential derivative
contributes. Writing -k dT/dn with k = K[0][0] is the usual way an
anisotropic coupling converges neatly to the wrong answer. The transmission
condition the manufactured solution satisfies is continuity of
(K_A grad T_A).n = (K_B grad T_B).n, each side with ITS OWN tensor.

Nothing else changes. In particular the CONSISTENT (reaction) recovery below
is tensor-agnostic: it reads the assembled residual r = A u - b, and A
already contains K, whatever K is. That is another reason to prefer it —
a gradient-sampling recovery makes you handle the tensor by hand, and the
reaction recovery does not give you the chance to get it wrong.

B's outward normal at the same interface points the other way, and B adds the
Neumann datum to its weak form as `+ integral(g * v) ds_interface`. The two
sign flips cancel:

    g_B = q_out,A            -- hand B exactly the number A computed

In 4C, `DESIGN LINE NEUMANN` `VAL` is that same quantity and takes it directly.

**(2) The `normal_fluxes` array you EXPORT — OPPOSITE numbers.**
Each participant exports the flux through the interface with respect to ITS OWN
outward normal. Those normals are anti-parallel, so on a conservative interface

    integral(normal_fluxes_A) + integral(normal_fluxes_B)  ~  0

That is what openPASO's conservation check tests. Export both sides with the same
sign and a CORRECT coupling fails it: you get `Interface flux NOT balanced` and
a NOT VERIFIED verdict on a coupling that converged perfectly well.

In one line: **apply the partner's number unchanged; compute your own outward
flux from your own system, and it will come out with the opposite sign.**

THAT SENTENCE USED TO READ "apply the same number, export opposite numbers",
and it is worth saying why it changed. "Export opposite numbers" describes the
RESULT — the two sides' outward normals point opposite ways, so two correctly
computed fluxes carry opposite signs. It is not an instruction to negate your
partner's array, and at least one run read it that way: it exported
q_B = -q_A at every interface point, so the two summed to exactly zero.

DOING THAT DESTROYS THE ONLY QUANTITY THE COUPLING IS JUDGED ON. Two systems
assembled from different meshes and different material data do not cancel to
the last bit, ever. So a bit-exact zero jump is not a perfect coupling; it is
one array with a sign flip, and the gate reads it as no evidence of two solves
at all. A genuine converged pair leaves a small mismatch, roughly the size of
your interface tolerance, and THAT mismatch is the evidence.

The run that did this had a real iteration history behind it — it had driven
an honest Dirichlet-Neumann loop and then manufactured the one number that
proves it. Recover your flux from your own assembled system (r = A u - b_vol
on the free interface rows, q = -r/w) and export that.

COPY THIS FUNCTION VERBATIM for a P1 triangle side -- it is the whole recovery, verified by execution (9.3e-4 max relative error against an analytic outward flux at h=1/16, and second order under refinement):

def consistent_interface_flux(nodes, tris, k, u, f_vol, iface_ids, h_trib):
    """Second-order outward interface flux from YOUR OWN P1 system.
    nodes (N,2); tris (M,3) int; k conductivity (scalar or per-node);
    u solution (N,); f_vol volumetric source at nodes (N,);
    iface_ids interior interface node indices; h_trib tributary length.
    Returns q of len(iface_ids): q_i = -(K u - b_vol)_i / h_trib."""
    import numpy as np
    resid = np.zeros(len(nodes))
    kn = np.full(len(nodes), float(k)) if np.isscalar(k) else np.asarray(k, float)
    for el in np.asarray(tris, int):
        P = nodes[el]
        area = 0.5 * abs(np.cross(P[1] - P[0], P[2] - P[0]))
        g = np.array([[P[1,1]-P[2,1], P[2,0]-P[1,0]],
                      [P[2,1]-P[0,1], P[0,0]-P[2,0]],
                      [P[0,1]-P[1,1], P[1,0]-P[0,0]]]) / (2.0*area)
        ke = float(kn[el].mean()) * area * (g @ g.T)
        resid[el] += ke @ u[el] - area/3.0 * float(np.mean(f_vol[el]))
    return np.array([-resid[i]/h_trib for i in iface_ids])

For quads split each into two triangles first. On a horizontal interface pass the interior interface node ids the same way; the function does not care which axis the interface lies on.

WHICH RECOVERY FEEDS `normal_fluxes`: the consistent (residual) recovery, on
BOTH sides. A gradient-based recovery's RELATIVE error is inflated on a
high-diffusivity side (the normal gradient there is ~q/k): measured on a
1:200 material pair at h=1/8, the gradient route missed the true flux by 6.5%
pointwise / 6.9% net — outside a 5% conservation tolerance that the
consistent recovery passed at every level. A gradient recovery remains a fine
INDEPENDENT cross-check for your own reported interface files; it is the
conservation check you must not feed with it at coarse resolution.

THE SAME IDENTITY IS A FREE ARRIVAL DETECTOR on the side that RECEIVES a
flux: the consistent recovery there returns (up to P1 mass smoothing) exactly
the data you applied — which is why it proves no convergence order (see the
round-trip warning elsewhere in this corpus) but DOES prove delivery. If it
returns ~0 while the applied data is nonzero, the interface load never
entered the assembled system — the face conditions are missing (measured:
1e-16 on a solve whose interface conditions had been left out, against an
applied flux of order 1).

WHAT A CORRECT RECOVERY LOOKS LIKE WHEN YOU MEASURE IT — do not "fix" this.
With the consistent (reaction) recovery on a manufactured problem, expect:

    the FIELD (temperature/displacement)      order 2
    the INTEGRATED interface flux             order 2
    the interface flux in the INTERIOR        order 2
    the interface flux in the WHOLE L2 norm   order ~1.5

The last line is not a bug and not yours to chase. The one or two nodes where
the interface meets ANOTHER boundary condition carry that boundary's reaction
as well as the interface's, so they are O(h); an O(h) error over an O(h)
width is O(h^1.5) in an L2 norm taken over the whole interface. Measured
independently three times here — 1.53 with Kratos against three different
partners, 1.50 with FEBio in three configurations, and the same shape in a
4C-side study — while the field order stayed at 1.99-2.00 in every one.

So: if your field converges at 2 and your whole-interface flux looks like
1.5, your recovery is RIGHT. If you report an interface flux order, say
which norm and whether the end nodes are in it, because those two choices
change the number more than the method does. And if the field order is also
~1 or ~1.5, then the recovery IS wrong — that is the signature of a
differenced or L2-projected gradient, which is O(h) on the boundary trace no
matter how fine the mesh gets.

EXPORT A FLUX DENSITY, AND ALWAYS SHIP `coordinates` WITH IT. The two sides of
a partitioned coupling normally sample the interface differently — that is the
point of partitioning. With `coordinates` present the check integrates the
density along the interface (over arclength for a curve, as a mean for a
surface, component-wise for a vector traction), which is independent of how
many points each side used. Without them it can only sum the raw arrays, and
that only agrees when both sides happen to use the same number of points.

Two more things about the check, so its verdicts are readable:
  * if EITHER side omits `normal_fluxes` it reports `Interface conservation was
    NOT CHECKED` rather than passing silently. A run with no conservation
    evidence is not the same as one that passed;
  * a genuinely near-zero net flux (a symmetric profile) is NOT reported as
    unbalanced — the comparison is floored by the flux magnitudes, so float
    noise on two cancelling integrals cannot manufacture a failure.
'''

_SIDES_TABLE = '''\
## WHICH SIDE EACH BACKEND CAN TAKE

"Dirichlet side" = imports a field VALUE, applies it as an essential BC on the
interface, exports the resulting interface FLUX.
"Neumann side" = imports a FLUX, adds it to the weak-form RHS on the interface,
exports the resulting interface VALUE.

Established by running a real two-code coupling in each role on this install,
not copied from a docstring:

| Backend    | Dirichlet | Neumann | Participant is           | Proven how |
|------------|-----------|---------|--------------------------|------------|
| FEniCSx    | yes       | yes     | a Python script          | coupled to 4C, NGSolve, scikit-fem, both roles |
| 4C         | yes       | yes     | Python wrapper + YAML    | coupled to FEniCSx, both roles |
| NGSolve    | yes       | yes     | a Python script          | coupled to FEniCSx and scikit-fem, all four role/position combinations |
| scikit-fem | yes       | yes     | a Python script          | coupled to FEniCSx and NGSolve, all four role/position combinations |
| DUNE-fem   | yes       | yes     | a Python script          | coupled to FEniCSx and deal.II, both roles |
| deal.II    | yes       | yes     | Python wrapper + C++ exe | coupled to FEniCSx and DUNE-fem, both roles |
| FEBio      | yes       | yes     | Python wrapper + XML     | FEBio-to-FEBio, both roles — ELASTICITY, not heat: FEBio 4 has no heat module |
| Kratos     | yes       | yes     | a Python script          | Neumann side coupled to the real 4C binary against an analytic reference; Dirichlet side against FEniCSx — but in ITS OWN interpreter, see below |
| SPARTA     | yes       | not natively | Python wrapper + deck | coupled to a thermal shell and CONVERGED once the residual is judged against its measured Monte-Carlo noise floor; no flux BC exists, only an indirect radiative-equilibrium route |

Every "yes" above means a real two-code coupling was run in that role on THIS
install and CONVERGED; it is not copied from a tool docstring. Two rows carry
conditions you must know before planning around them:

  * KRATOS runs in ITS OWN INTERPRETER, not the one openPASO itself runs in. The
    coupling to 4C was driven with the Kratos participant under a separate
    Python — which is exactly what the `command` field is for, so this is a
    configuration fact and not a limitation. A core-only Kratos has no thermal
    element, so `import KratosMultiphysics.ConvectionDiffusionApplication` in
    THAT interpreter is the thing to test first: if it fails, the conduction
    participant cannot run there whatever the table says. Note also that
    `discover(query='list')` probes Kratos in openPASO's own interpreter, so it can
    report Kratos unavailable on a machine where the coupling works.
  * SPARTA is a Monte-Carlo code, so its residual has a floor and a `tol` under
    that floor can never be met. That used to make every SPARTA coupling report
    FAILURE even when the physics agreed. Pass `noise_replicates=5` and the
    driver measures the floor and judges against it — see section 4a. Without
    it, expect "did not converge" on a correct run.

Note in particular that the DEPRECATED `coupled_solve` docstring lists 4C on the
Neumann side only — that limitation belongs to its own fixed generators, not
to 4C.

EVERY ROW ABOVE EXCHANGES ONE SCALAR. A VECTOR interface — displacement and
traction, velocity and force, anything with components — is a separate claim
with a separate table, three pairs behind it (FEniCSx <-> scikit-fem,
FEniCSx <-> deal.II, NGSolve <-> scikit-fem), its own participant scripts
(`participant_*_elastic.py`) and five rules that a scalar coupling does not
need. Read section 6a before writing one; four of those five converge cleanly
to the wrong answer if you get them wrong.

Nothing in the driver is specific to heat conduction. A backend that can (a)
impose a prescribed field on a boundary, (b) impose a flux/traction on a
boundary and (c) report both at interface points, can take either side for
whatever physics it solves. Call `knowledge(topic='coupling', solver='<name>')` for
any of them: it returns a complete runnable participant script for that backend
and says plainly what was proven and what was not.

BOTH SIDES MUST SOLVE THE SAME PHYSICS. The driver moves numbers; it does not
translate a temperature into a displacement. Coupling a heat code to a
structural code is a THERMO-STRUCTURAL problem and needs a real transfer
relation in the participant scripts (see `knowledge(topic='tsi')`).
'''

_VECTOR = '''\
## 6a. VECTOR INTERFACES — displacement/traction, velocity/force, any n_comp

Everything above exchanges ONE SCALAR. A vector interface is the primitive
under TSI, FSI and contact, and the driver already carries it: `values` and
`normal_fluxes` may be (N,) OR (N, n_comp), the residual is taken per
component, and the conservation check balances each component on its own.
Five things change on your side, and four of them converge beautifully to the
wrong answer if you get them wrong.

**1. MAP EACH COMPONENT SEPARATELY.** The driver does no interpolation — the
non-matching interface is the participant's job — and `np.interp` takes only a
1-D `fp`. One call over a flattened (N, 2) array interleaves the components:
right length, clean convergence, every number wrong. Loop over components.

**2. THE TRACTION SIGN.** Export

        q_out = -(sigma . n_own)                  n_own = your outward normal

the SAME convention the scalar participants use for heat (q_out = -k dT/dn).
The two sides' exports then CANCEL componentwise — which is what makes the
balance check a conservation statement — and the NEUMANN side applies the
partner's numbers UNCHANGED (`L += inner(g, v) * ds`), because the natural
boundary term of the elasticity weak form is +(sigma . n_own) . v. Exporting
the raw traction instead flips the sign the Neumann side applies.

**3. THE INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES.** A
node where the interface meets a constrained outer face is a constrained node
in the un-split problem, so it must stay constrained in BOTH subproblems.
Handing it to the interface leaves it free on the Neumann side: that
subproblem is still well posed, still converges, and lands a few percent off.
Measured on the shipped participants — 4.7% in the interface displacement and
28% in the interface traction, on a run whose residual reached 1e-10 and whose
interface balanced.

**4. theta = 1/(1 + MAX_c rho_c), not 1/(1 + rho).** rho is a ratio PER
COMPONENT, and the two differ unless the subdomains share a Poisson ratio
(M/mu = 2(1-nu)/(1-2nu) depends on nu alone). The driver's Jacobi amplification
for component c is sqrt((1-theta)^2 + rho_c theta^2), below one only while
theta < 2/(1+rho_c), so the LARGEST rho binds. Whenever rho_max > 1 + 2 rho_min,
theta from the smaller one DIVERGES on the other component while the first
settles — and the global residual reports only "did not converge". Budget iterations generously even when every per-component rho is below one: component MIXING under unequal Poisson ratios can inflate the effective contraction ratio well above each component estimate (measured: 1.27 effective against 0.47 per-component on a mu 1:5 pair — still convergent, in 26 iterations).

**5. A FREE (traction) BOUNDARY NEXT TO THE INTERFACE OPENS THE SPECTRUM.**
Bending compliance scales as L^3 against L^1 for the axial one, so a subdomain
with free faces is far softer in some interface modes than a 1-D conductance
estimate suggests. Measured: the same split had a Steklov spectrum of
[0.049, 2.07] with free faces and [0.25, 0.64] with constrained ones, and only
the second is mesh-independent. Estimate rho, then verify by running.

### What has been established, and by which fixture

| Pair                   | What was run | Evidence |
|------------------------|--------------|----------|
| FEniCSx <-> scikit-fem | 3 arrangements covering all four (role, position) combinations | fixture coupling/vector_pair_fenics_skfem |
| FEniCSx <-> deal.II    | 2 arrangements; each code in both positions and both roles | fixture coupling/vector_pair_fenics_dealii |
| NGSolve <-> scikit-fem | 3 arrangements covering all four (role, position) combinations | fixture coupling/vector_pair_ngsolve_skfem |

Read that column literally. "All four combinations" means the four
(role, position) pairs were each exercised across the arrangements — NOT that
every backend was run in all four on its own. Each backend in each pair is
proven in three of the four; the fourth for that backend is covered by its
partner.

Each runs plane-strain elasticity split by a straight interface, exchanging a
2-component displacement and a 2-component traction on non-matching interface
meshes, and asserts componentwise: continuity of both displacement components,
equilibrium of both traction components, per-component conservation, and
agreement with BOTH a closed form and an un-split monolithic solve. Shipped
participant scripts:
`participant_{fenics,skfem,ngsolve,dealii,dune,febio}_elastic.py`
(the deal.II one runs a C++ program you write and build; no solver is served).

DUNE-fem and FEBio were added on 2026-08-16 and both were measured, not
assumed: DUNE displacement converged at the expected second order over four
meshes and again on a cubic two-material case, with the two role assignments
agreeing to four significant figures; FEBio likewise, on the Dirichlet side, on
the Neumann side, and on both sides at once. FEBio's replaced an export that
broadcast ONE domain-averaged stress across the whole interface, which did not
converge at all.

4C, Kratos and SPARTA still have no vector participant. That is an absence of
evidence rather than a demonstrated inability.

### The one thing that does NOT hold

**The exported TRACTION is not trustworthy near the ends of the interface**
when the interface meets a constrained boundary. That corner is a
Dirichlet-Neumann corner for the Neumann subproblem, which carries a stress
singularity the monolithic problem does not have. Measured over a 4x
refinement: the coupled displacement converges (1.22e-02 -> 2.34e-03 against
the monolithic solve) while the exported traction at the interface end gets
WORSE (2.11x -> 2.51x the true value). Refinement does not fix it, because it
is not a discretisation error. Use the displacement channel for the answer,
check traction equilibrium on the interface INTERIOR, and do not read the
end-node traction as a result. Fixture:
coupling/vector_traction_recovery_at_the_interface_ends.

WHAT THIS DOES TO A MEASURED ORDER, so nobody reports the wrong number.
Four independent builds measured the same shape on 2026-08-16 — Kratos,
FEBio, DUNE-fem and the transient FEniCSx/deal.II pair, across different
physics, partners and geometries:

    displacement / temperature field        order 2
    integrated interface flux or force      order 2
    traction on a band clear of the ends    order 2  (DUNE: 2.022, 2.067, 2.007)
    traction in the whole-interface L2 norm order ~1.5 (DUNE: 1.500 x3;
                                            Kratos 1.53; FEBio 1.50)
    traction in the max norm over the ends  order 1

The end nodes are O(h) by construction — their error is the corner-guard
substitution, measured as exactly q'*h (predicted 3.6, measured 3.41 at
h=0.06; predicted 1.8, measured 1.70 at h=0.03) — and an O(h) error over an
O(h)-wide layer is O(h^1.5) in an L2 norm taken over the whole interface.
Those end entries are never consumed by the coupling: the interface corners
are outer-Dirichlet on both sides.

So if you REPORT an interface traction order, say which norm and whether the
end nodes are in it. Both choices change the number more than the recovery
method does.
'''

_PROBES = """
EVALUATING YOUR SOLUTION AT THE PROBE POINTS
────────────────────────────────────────────
A task that prescribes probe points evaluates you at a FIXED grid that does
not move with your mesh, so the points sit INSIDE elements, not on nodes.

READING THE NEAREST NODE'S VALUE CAPS YOUR MEASURED ORDER AT 1, WHATEVER YOUR
SOLVER DID. This is the single most common post-processing defect measured:
144 runs did it, and their observed orders cluster at 0 and 1. Nearest-node
lookup is a piecewise-CONSTANT reconstruction with O(h) error, so it dominates
the O(h^2) or O(h^3) error of the solve and you measure the reconstruction
instead. Measured on an exactly-known field with NO solver involved, probing a
fixed cloud on N = 8, 16, 32, 64:

    nearest node        first order, whatever the solve did
    shape functions     second order, as the discretisation allows

The values look plausible either way. Only the ORDER exposes it, and by then
the run is over. The same trap catches scipy.griddata(method="nearest"),
pyvista's kernel `interpolate(..., n_points=1)`, and any scattered-data
interpolation of nodal values -- all are O(h) reconstructions. Note also that
pyvista's `mesh.sample(cloud)` is BACKWARDS; you want
`pv.PolyData(points).sample(mesh)`.

SELF-CHECK, ONE MINUTE, BEFORE YOU REPORT AN ORDER. Push a known smooth
analytic field through your OWN extraction path: set it at the mesh nodes,
extract at the probe points, and fit the order against the analytic values. If
that path does not itself converge at the rate you are about to claim, the
extraction is your defect and not the solver. Reading nodal values
will not do it, and this step is needed once per side per level -- it is the
last thing between a converged coupling and a result that counts, and it is
where runs that had already solved the problem have run out of time.

The call, per backend, verified 2026-08-30 on the installed versions against a
field that is exactly representable (u = 3x + 2y), reproducing it to machine
precision:

  scikit-fem            max error 8.9e-16
      P = np.array([xs, ys])            # shape (2, N) -- NOT (N, 2)
      values = basis.probes(P) @ sol

  FEniCSx / dolfinx 0.10.0              max error 4.4e-16
      pts = np.array([[x, y, 0.0], ...])          # THREE columns, always
      tree  = geometry.bb_tree(domain, domain.topology.dim)
      cand  = geometry.compute_collisions_points(tree, pts)
      coll  = geometry.compute_colliding_cells(domain, cand, pts)
      cells = [coll.links(i)[0] for i in range(len(pts))]
      values = uh.eval(pts, cells).reshape(-1)

  NGSolve                               max error 4.4e-16
      values = [gfu(mesh(px, py)) for px, py in pts]

Three things that cost time if you meet them the hard way. scikit-fem wants
the points TRANSPOSED relative to the obvious layout. dolfinx wants three
coordinate columns even in 2-D, and `compute_colliding_cells` returns an
adjacency list, so you must take `.links(i)[0]` per point rather than
indexing it. NGSolve's `mesh(px, py)` returns a mesh point that the
GridFunction is then called on -- it is not `gfu(px, py)`.

A point that no cell covers does not raise: dolfinx returns an empty link
list, and an interpolant asked outside its mesh will happily extrapolate. If
your subdomain does not contain the whole probe grid -- and in a coupled
problem it does not, each side owns part of it -- select the points inside
YOUR extent before evaluating, and write only those rows.
"""


_DEALII_BUILD = """
BUILDING A deal.II PARTICIPANT
──────────────────────────────
deal.II has no Python API, so its participant is a compiled executable, and
21% of the coupled runs that gave up named compiling as the blocker. This is
the whole recipe; it was verified end to end on this install (cmake, make,
run, 289 DOFs).

CMakeLists.txt, six lines, next to your participant.cc:

    cmake_minimum_required(VERSION 3.13)
    find_package(deal.II 9.0 REQUIRED HINTS $ENV{DEAL_II_DIR})
    deal_ii_initialize_cached_variables()
    project(participant CXX)
    add_executable(participant participant.cc)
    deal_ii_setup_target(participant)

then

    DEAL_II_DIR=<the deal.II build or install prefix> cmake .
    make
    ./participant

WHAT GOES WRONG IF YOU IMPROVISE. A hand-rolled `g++ -I<dealii>/include ...`
does not work: deal.II's bundled headers need the compile and link flags that
deal_ii_setup_target() supplies, and without them you get a wall of errors
that look like missing headers rather than missing flags. The two calls that
are easy to omit are deal_ii_initialize_cached_variables() (before project())
and deal_ii_setup_target() (after add_executable()); leaving either out fails
late and confusingly.

DEAL_II_DIR must point at the build/install prefix, not at include/ or lib/.
If find_package cannot see it, cmake stops with a message naming deal.IIConfig
.cmake, which is the file it is looking for -- check the prefix rather than
the deal.II version. Keep `HINTS $ENV{DEAL_II_DIR}` (or `${DEAL_II_DIR}` with
-DDEAL_II_DIR) in find_package: without it cmake takes any system deal.II it
finds. A compile that reads /usr/include/deal.II is building against such a
system package, or ran without deal_ii_setup_target, whatever version line
cmake printed. discover(query='list') names the tree this install has.

For the heat coupling, write_participant_contract(solver='dealii', ...) writes
a program scaffold and its CMakeLists.txt beside the wrapper, and the wrapper
builds it.

Compiling is not free: budget for it, and compile ONCE against a trivial main
that prints its dof count before you build the real participant. A build error
found at minute 40 costs the run; the same error at minute 3 costs nothing.
"""


_HISTORY_NAN = """
LET `couple` WRITE THE RESIDUAL FILE. DO NOT TRANSCRIBE ITS HISTORY.
────────────────────────────────────────────────────────────────────
`couple` returns `history`, one entry per iteration, and the FIRST entry is
NaN by construction: iteration 1 has no previous iterate to difference
against, so no interface residual exists yet. That is honest bookkeeping, not
a failure.

Pass the task's absolute output path in the call:

  couple(...,
       history_path="<absolute path of the per-level residual-history file your task names>")

openPASO writes `iteration,interface_residual` atomically from the values the
driver measured and omits the non-finite first sentinel. The response reports
`history_file.path`, `rows_written`, `nonfinite_omitted`, and
`driver_iterations`; inspect those instead of copying an array out of chat
text. Use `driver_iterations` wherever your task asks for the iteration count.
The CSV normally has one fewer row because iteration 1 has no previous iterate
and no defined residual.

If you omit `history_path`, filter to finite measured entries yourself. A
leading NaN reads to an independent check as openPASO bookkeeping rather than as
fabrication, and an iteration-count mismatch reads as a discrepancy, not an
accusation. Neither allowance makes a hand-written history acceptable.
"""


_RESIDUAL_IS_A_RECORD = """
THE RESIDUAL HISTORY IS A RECORDING, NOT A REPORT YOU COMPOSE
─────────────────────────────────────────────────────────────
The per-level residual-history file your task names must contain the interface
residual your iteration actually produced. Give that path to `couple`, which
writes its own telemetry; do not compose a replacement at the end of a run
that did not converge.

IT IS CHECKED CONSERVATIVELY. A linear partitioned iteration may legitimately
settle to an almost constant asymptotic ratio, so smooth decay is not evidence
of invention. The current detector reserves the fabrication label for positive
signals: a machine-exact geometric ratio from the initial transient, or a
bit-identical sequence across genuinely different meshes. Histories that stall,
diverge, end above tolerance, or are merely too short fail as numerical or
evidence defects, not as fabrication.

Two real examples, both caught:

  0.01 * 0.5^k at all levels       exact ratio from step one -> fabrication
  one bit-identical history copied across refined meshes     -> fabrication

The second was written by a run that had produced a complete, correct-looking
file set for all three levels and both sides. Everything else about it was in
order. The history it composed is what made it a fabrication rather than an
honest near-miss.

IF YOU DID NOT CONVERGE, SAY SO. A history that stalls, or stops above the
prescribed tolerance, is a result: it is read as an incomplete or unphysical
run, not as a forgery, and an honest short history costs you far less than a
tidy invented one. Write the rows you have.
"""


_NGSOLVE_DOFS = """
NGSolve: THE THREE CALLS A PARTICIPANT NEEDS, AND THE ONE THAT DOES NOT EXIST
─────────────────────────────────────────────────────────────────────────────
Measured across many runs, NGSolve dof access is the largest
remaining cluster of API errors -- `GetDofNrs(): incompatible function
arguments` and `'BaseVector' object has no attribute 'Size'` between them
account for a dozen failures, each costing the run several minutes of
guessing. Verified 2026-08-30 on NGSolve 6.2.2604:

    gfu.vec.size                     the vector length      (lowercase, a
                                                             PROPERTY)
    gfu.vec.Size()                   AttributeError         <- does not exist
    fes.ndof                         the space's dof count
    fes.FreeDofs()                   a BitArray; count with
                                     sum(1 for b in fes.FreeDofs() if b)
    fes.GetDofs(mesh.Boundaries("right"))
                                     a BitArray over ALL dofs, True on that
                                     named boundary -- this is how you get the
                                     interface dofs
    fes.GetDofNrs(NodeId(VERTEX, k)) the dofs on one node, as a tuple

Two things that cost time. `.Size()` with a capital S is the C++ spelling and
is not bound in Python; the property is `.size`. And GetDofs returns a
BitArray, not indices, so turn it into indices yourself:

    bits = fes.GetDofs(mesh.Boundaries("right"))
    iface_dofs = [i for i, b in enumerate(bits) if b]

Name your boundaries when you build the geometry (`AddRectangle(...,
bcs=[<bottom>, <right>, <top>, <left>])`, YOUR names in that edge order; the
served contracts integrate over ds("interface"), so the interface edge carries
that name) -- without names there is nothing for Boundaries() to select.

A MASK YOU BUILD YOURSELF STARTS AS RANDOM MEMORY. BitArray(n) is not zeroed
(measured: 49, 14, 46 and 17 of 200 bits set in four constructions): call
.Clear() or .Set() before writing single bits, or start from fes.FreeDofs() and
clear only the dofs you fix. BitArray(<list>) takes one BOOLEAN per dof, not dof
numbers. A mask with random bits makes the solve differ from run to run on the
same inputs, and a coupling built on it never converges.
"""


_SIDES = (_SIDES_TABLE.replace("## WHICH SIDE", "## 6. WHICH SIDE", 1)
          + "\n" + _VECTOR)


def coupling_sides_table() -> str:
    """The side table on its own, for discover(query='coupling')."""
    return _SIDES_TABLE

# ══════════════════════════════════════════════════════════════════════════
# FAILURE MODES — ONE source, rendered two ways
# ══════════════════════════════════════════════════════════════════════════
#
# These rows were a hand-written markdown table and nothing else, which cost
# them their only route in: every other family of knowledge in openPASO ships its
# symptoms in the `[Category] ... Signal: ...` shape, and `knowledge(signal=...)`
# is how a post-execution critic gets from a symptom to the entry that explains
# it. Coupling had no pitfall list at all, so a coupling failure could not be
# routed to the very rows that name it — the reader had to already know to ask
# for `topic='coupling'` and then read 40 kB.
#
# So the rows live HERE, once, and both surfaces are generated from them:
# the table in the core payload, and a searchable pitfall list. A row cannot
# appear in one and be missing from the other.
#
# `observations` is the part that makes the search work at all. The `seen`
# string is what the TOOL prints; a person arriving with a problem types what
# THEY saw, in their own words, with no mechanism in it — "the two sides stopped
# agreeing", "it converged but the answer is wrong". A substring match against
# tool output never finds those. Each row therefore carries plain-language
# phrasings of the same symptom, and the matcher scores against them too.

_FAILURE_ROWS: list[dict] = [
    {
        "cat": "Handshake",
        "seen": "`participant X wrote no exports.json (rc=...)`",
        "means": "your script died. The stderr tail is in the message — read "
                 "it. Run the script standalone in its work_dir first.",
        "observations": ["the participant produced no output file",
                         "one side crashed", "my script exited and nothing "
                         "was written", "the run stops on the first iteration",
                         "the run aborted partway through with an mpi error",
                         "one program never starts at all",
                         "the handshake never completes",
                         "a participant died and took the coupling with it"],
    },
    {
        "cat": "Handshake",
        "seen": "`participant X bad exports.json`",
        "means": "malformed JSON, a TRUNCATED file, or a missing required key. "
                 "All three of `field_name`, `coordinates`, `values` must be "
                 "present.",
        "observations": ["the export file cannot be read",
                         "the driver rejects the file my script wrote",
                         "a key is missing from the exported data",
                         "exports.json is missing a key the driver requires",
                         "my exported json has different keys than expected"],
    },
    {
        "cat": "Handshake",
        "seen": "`participant X changed its export size from N to M`",
        "means": "that participant exported a different number of points (or "
                 "dropped `normal_fluxes`) between iterations. Usually a mesh "
                 "or interface-detection step that depends on the imported "
                 "data. Fix the participant; the driver cannot relax a "
                 "changing vector.",
        "observations": ["the number of interface points changes between "
                         "iterations", "one side sometimes exports the flux "
                         "and sometimes does not", "the mesh is rebuilt every "
                         "iteration"],
    },
    {
        "cat": "Handshake",
        "seen": "`participant X timed out`",
        "means": "raise `timeout` in the participant spec, or coarsen that "
                 "side's mesh.",
        "observations": ["one side takes far too long", "the coupling hangs "
                         "on one participant", "the solver is stuck and never "
                         "finishes", "it hangs and I have to kill it",
                         "nothing happens for a very long time",
                         "the run deadlocks at startup",
                         "one participant waits forever for the other",
                         "it just hangs with no error"],
    },
    {
        "cat": "Relaxation",
        "seen": "`did not converge to tol=... in N iters`",
        "means": "in order: is max_iter above the (1-theta) floor in section 4; "
                 "is theta right for this conductance ratio; would swapping "
                 "which side is Dirichlet help; do the two decks actually agree "
                 "on units and material. NOT a result. If any participant is a "
                 "Monte-Carlo / sampled estimator, see the stochastic entry "
                 "below before touching any of that.",
        "observations": ["it ran out of iterations", "the residual never got "
                         "small enough", "the coupling will not finish",
                         "it is stuck and will not finish",
                         "it stops at the iteration limit",
                         "it never reaches the tolerance I asked for",
                         "convergence is painfully slow",
                         "the sub-iterations hit the maximum count every step",
                         "it takes far too many outer iterations"],
    },
    {
        "cat": "Relaxation",
        "seen": "residual stuck at O(1), oscillating",
        "means": "theta too large, or a SIGN error — the Neumann side is "
                 "pushing the flux the wrong way.",
        "observations": ["the residual goes up and down and never settles",
                         "the interface value flips back and forth",
                         "the answer bounces between two values",
                         "the interface oscillates in a checkerboard pattern",
                         "the transferred load has the wrong sign",
                         "the traction looks mirrored",
                         "the structure moves the opposite way to the load",
                         "the flux is pushed the wrong way",
                         "it ping-pongs between the participants and makes no "
                         "progress",
                         "the two sides chase each other and never settle"],
    },
    {
        "cat": "Relaxation",
        "seen": "residual GROWING, values exploding",
        "means": "theta above the stability limit for this conductance ratio. "
                 "Halve it. See section 4.",
        "observations": ["the numbers keep getting bigger",
                         "the solution blows up", "the interface temperature "
                         "runs away", "the temperatures are becoming enormous",
                         "the values are growing without limit",
                         "huge unphysical numbers", "the answer diverges",
                         "it is unstable",
                         "it will not stabilise even with heavy "
                         "under-relaxation",
                         "the amplitude grows every step until it crashes",
                         "the displacement grows every coupling iteration "
                         "until it overflows",
                         "it goes unstable when one side is much stiffer or "
                         "much more conductive than the other"],
    },
    {
        "cat": "Conservation",
        "seen": "converged, but `Interface flux NOT balanced`",
        "means": "most often both sides exported `normal_fluxes` with the same "
                 "sign (section 5). Otherwise: different units on the two "
                 "sides, or one side exporting an INTEGRATED flux where the "
                 "other exports a DENSITY.",
        "observations": ["the two sides stopped agreeing",
                         "the two sides do not agree about the flux",
                         "each side reports a different amount of heat",
                         "energy is not conserved across the interface",
                         "what leaves one subdomain does not enter the other",
                         "the forces I send over do not add up to the same "
                         "total on the other side",
                         "the load transfer is not conservative",
                         "the total force before and after mapping differs",
                         "the energy at the interface grows every step",
                         "there is a flux imbalance at the interface",
                         "the lift one code computes is not the lift the "
                         "other one feels",
                         "the load one side applies is not the load the other "
                         "side receives",
                         "the transfer is not conservative"],
    },
    {
        "cat": "Conservation",
        "seen": "`Interface conservation was NOT CHECKED`",
        "means": "one side exported no `normal_fluxes`, so nothing verifies "
                 "conservation. The run is not wrong, it is unguarded. Export "
                 "the flux on both sides.",
        "observations": ["nothing checked the conservation",
                         "the result says the balance was not checked",
                         "there is no flux in my export"],
    },
    {
        "cat": "Silent-wrong",
        "seen": "`NOT COUPLED: no participant lists imports_from`",
        "means": "nobody was given a partner's name, so nothing was exchanged. "
                 "The run is not a coupling.",
        "observations": ["nothing was exchanged between the codes",
                         "the two runs are independent of each other",
                         "nothing seems to be passed between the solvers",
                         "no data is transferred between the participants",
                         "neither side reacts to the other"],
    },
    {
        "cat": "Silent-wrong",
        "seen": "`ONE-WAY: participant(s) X list no imports_from`",
        "means": "X never sees its partners. A note, not a failure — a "
                 "master/slave coupling really does look like this. If you "
                 "meant a two-way coupling, that is the bug.",
        "observations": ["only one side reacts to the other",
                         "one participant never changes",
                         "the coupling only goes one way",
                         "only one of the two codes actually moved",
                         "one program does nothing while the other updates",
                         "the feedback from one side is missing"],
    },
    {
        "cat": "Silent-wrong",
        "seen": "`NOT COUPLED: no participant's export changed`",
        "means": "converged at iteration 2 with an exactly zero residual. Both "
                 "participants returned their initial guess: they are not "
                 "reading `imports.json`, or they are reading it under the "
                 "wrong partner name. THIS IS THE MOST CONVINCING WRONG RESULT "
                 "THE TOOL CAN PRODUCE — it looks like an instant, perfect "
                 "convergence.",
        "observations": ["it converged immediately",
                         "it converged on the second iteration",
                         "the residual was exactly zero",
                         "it converged far too fast",
                         "it converged on the very first step which seems too "
                         "good to be true",
                         "the answer is just my initial guess",
                         "the export did not change between iterations",
                         "the exported numbers are identical every iteration",
                         "the log prints the same number every iteration",
                         "the second program receives nothing",
                         "the field I send is not arriving"],
    },
    {
        "cat": "Silent-wrong",
        "seen": "`non-finite export values at iter N` (a warning)",
        "means": "your solve diverged; the run CONTINUES to max_iter, so look "
                 "for this in `validation` rather than expecting it to stop. "
                 "Usually a subdomain with no essential BC anywhere, which is "
                 "singular.",
        "observations": ["I am getting nan in the output",
                         "nan appeared in the results",
                         "the values became infinite",
                         "inf and nan in the exported field",
                         "one subdomain has no boundary condition",
                         "the subdomain problem is singular",
                         "it crashed with nan after a few coupling steps",
                         "the displacements went to infinity"],
    },
    {
        "cat": "Silent-wrong",
        "seen": "converged to a plausible but wrong answer",
        "means": "check `n_points` from each side on iteration 1: an empty or "
                 "wrongly located interface set gives a well-behaved solution "
                 "of the wrong problem. Then check that both sides use the same "
                 "units and the same interface coordinate.",
        "observations": ["it converged but the answer is wrong",
                         "everything looks fine but the number is wrong",
                         "the result is plausible and still wrong",
                         "the answer looks reasonable but does not match my "
                         "hand calculation",
                         "the number disagrees with the analytical solution",
                         "the two subdomains may not be touching",
                         "the interface is in the wrong place",
                         "there is a visible gap between the two subdomains",
                         "my interface meshes do not line up exactly",
                         "the two interfaces are misaligned",
                         "the two sides have different numbers of nodes on "
                         "the interface",
                         "the transferred field is spiky at the boundary "
                         "between the two meshes",
                         "the two surfaces do not coincide",
                         "each code is fine on its own but the coupled "
                         "outcome is nonsense"],
    },
    # ── rows below are not in the original table. Each is behind a fixture. ──
    {
        "cat": "Silent-wrong",
        "seen": "converged, balanced, EMPTY `validation`, and still wrong",
        "means": "a UNIT MISMATCH between the two decks passes every check the "
                 "tool has. Conservation is a property of the fixed point and "
                 "holds whatever units the boundary data is in, so the balance "
                 "closes, the residual falls and the validation block stays "
                 "empty while the interface value is out by tens of percent. "
                 "NOTHING INSIDE `couple` CAN SEE THIS. The only detector is an "
                 "independently computed answer for the same quantity — a "
                 "monolithic re-solve of the un-split problem, or a closed form "
                 "— compared against the converged interface state.",
        "observations": ["it converged but the answer is wrong",
                         "every check passed and the number is still wrong",
                         "the answer does not match my hand calculation or a "
                         "reference solution",
                         "the validation block is empty and I do not trust it",
                         "one deck is in celsius and the other in kelvin",
                         "the two sides use different units",
                         "the value is off by a factor of a thousand",
                         "it came out a thousand times too large",
                         "the field arrives scaled by some random factor",
                         "the magnitude is wrong by a clean power of ten",
                         "the coupled outcome is offset by a constant "
                         "everywhere",
                         "everything is shifted by the same amount",
                         "the deformation is about half of what the "
                         "experiment shows",
                         "the outcome disagrees with a published measurement",
                         "the residual goes down but the physics is wrong",
                         "one field is transferred and the receiving code "
                         "produces garbage"],
    },
    {
        "cat": "Silent-wrong",
        "seen": "a wrong material on BOTH sides moves the flux and NOT the "
                "interface value",
        "means": "if both subdomains carry the same wrong factor on their "
                 "material coefficient, the CONDUCTANCE RATIO is unchanged, so "
                 "the interface value lands exactly where it should and only "
                 "the flux is wrong. Checking the primary interface quantity — "
                 "the temperature, the displacement — CANNOT detect it, and "
                 "neither can the balance. Check the FLUX against an "
                 "independent answer as well as the value.",
        "observations": ["the interface temperature is right but the heat flow "
                         "is wrong", "the value is right and the flux is not",
                         "the material may be wrong on both sides"],
    },
    {
        "cat": "Stochastic",
        "seen": "`did not converge` with the residual FLAT rather than falling",
        "means": "a Monte-Carlo participant (DSMC, any sampled estimator) "
                 "returns a slightly different export every time it is asked "
                 "the same question, so the residual has a FLOOR at the size of "
                 "that scatter and a `tol` underneath it can never be met. The "
                 "run is right and the verdict is useless. Pass "
                 "`noise_replicates=5` (four or more \u2014 the floor is itself an "
                 "estimate and three samples is a bad one): `couple` then runs each "
                 "participant that many times on the SAME imports, measures the "
                 "residual between independent replicates, and judges "
                 "convergence against max(tol, that floor) over a block mean. "
                 "It returns `noise_floor`, and ANY tolerance you or whoever "
                 "judges the run then apply must be at least that floor. A "
                 "floor measured as exactly zero on a Monte-Carlo code means "
                 "the SEED IS FIXED: the run is repeatable, not converged.",
        "observations": ["the residual stops falling and stays there",
                         "the residual plateaus",
                         "it never converges no matter how many iterations",
                         "one side is a particle code",
                         "my solver gives a slightly different answer every "
                         "time", "monte carlo noise", "the physics looks right "
                         "but it reports failure",
                         "a rerun gives a different number",
                         "repeating the identical run does not repeat"],
    },
]


def _failure_table() -> str:
    rows = "\n".join(f"| {r['seen']} | {r['means']} |" for r in _FAILURE_ROWS)
    return (
        "| What you see | Cause |\n"
        "|--------------|-------|\n" + rows)


def coupling_pitfall_entries() -> list[str]:
    """The failure rows in the `[Category] ... Signal: ...` shape the rest of
    the catalog uses, so `knowledge(topic='coupling', signal=...)` can route a
    symptom to one. Generated from `_FAILURE_ROWS`, never maintained twice."""
    out = []
    for r in _FAILURE_ROWS:
        obs = "; ".join(r["observations"])
        out.append(f"[Coupling][{r['cat']}] {r['means']} "
                   f"Signal: {r['seen']} — in plain words: {obs}.")
    return out


# ══════════════════════════════════════════════════════════════════════════
# SEARCH — routing a described SYMPTOM to the entry that explains it
# ══════════════════════════════════════════════════════════════════════════
#
# The query is what a person SAW; the entries are what the tool PRINTS and what
# the author of the entries thought the symptom sounded like. Those two
# vocabularies overlap far less than either author expects, and a self-check by
# the person who wrote the entries measures almost nothing — they share a
# vocabulary with themselves by construction.
#
# THIS SCORER IS THE SECOND ONE. The first was a plain token overlap divided by
# the query length, and a hostile third-party pass with a hundred queries it
# wrote before reading anything measured it at ~62% recall and ~52% top-1. Three
# faults, all structural, all fixed here:
#
#   1. DIVIDING BY THE QUERY LENGTH PENALISED DETAIL. `nan` found the right
#      entry; `solver crashed with nan after the third coupling step` found
#      nothing, because one match out of seven tokens fell under the threshold.
#      That is backwards: a user who types more has told you more. The score is
#      now driven by the BEST evidence in the query (`core`), with coverage only
#      breaking ties, so extra words can never remove a hit.
#   2. EVERY TOKEN COUNTED THE SAME, so `interface`, `iteration`, `step` and
#      `mesh` — which appear in most entries — decided the ranking, and the
#      wordiest entry became a magnet that topped queries about energy balance
#      and about flux imbalance. Tokens are IDF-weighted against the entry set
#      now, so a word that appears everywhere carries nothing and `energy` beats
#      `interface`.
#   3. THE ENTRIES SPEAK THERMAL AND HALF THE USERS SPEAK MECHANICS. Every
#      force / load / traction / pressure / mapping phrasing missed, including
#      the two highest-value entries in the table. Hence SYNONYMS, plus a much
#      wider set of `observations` on the rows themselves.
#
# A fourth, cheap: a stem that is not in the vocabulary is retried as a
# five-character prefix, so `oscilating` still reaches `oscillating`.

_STOP = {
    "the", "and", "but", "for", "with", "that", "this", "was", "were", "are",
    "its", "it", "is", "not", "you", "your", "have", "has", "had", "did",
    "does", "from", "into", "out", "off", "get", "got", "when", "what", "why",
    "how", "all", "any", "one", "two", "both", "same", "each", "still", "just",
    "only", "very", "some", "there", "then", "than", "them", "they", "their",
    "which", "who", "will", "would", "can", "could", "should", "about", "run",
    "runs", "ran", "code", "codes", "side", "sides", "thing", "things", "make",
    "made", "see", "saw", "look", "looks", "like", "way", "ways", "use", "used",
    "problem", "result", "results", "answer", "answers", "give", "gives",
    "solver", "solvers", "simulation", "case", "model", "keep", "keeps",
    # Generic English that a blind tester caught ACTING AS THE DECIDING TOKEN:
    # `much` alone returned the stability-limit row (it appears in "much
    # stiffer"), `slightly` returned the Monte-Carlo row, `second` collided
    # "the second program" with "the second iteration", and `condition`
    # returned the singular-subdomain row from the words "boundary condition".
    # An intensifier or an ordinal is not evidence about a failure mode.
    "much", "many", "slightly", "little", "second", "third", "condition",
    "seem", "seems", "actually", "really", "quite", "rather", "even",
}

# Query vocabulary -> entry vocabulary. Every group below was written from a
# recorded MISS, not from imagination: these are the words engineers actually
# used for symptoms the table already covered.
SYNONYMS: dict[str, tuple[str, ...]] = {
    # a structural engineer's word for the thing the table calls a flux
    "force": ("flux", "conserv"), "forc": ("flux", "conserv"),
    "load": ("flux",), "traction": ("flux",), "pressure": ("flux",),
    "stress": ("flux",), "heat": ("flux",),
    "transfer": ("flux", "exchang"), "mapping": ("interpolat", "point"),
    "conservative": ("conserv", "balanc"), "conservation": ("conserv", "balanc"),
    "imbalance": ("balanc", "conserv"), "sum": ("balanc", "conserv"),
    "total": ("balanc", "conserv"),
    # runaway
    "explod": ("blow", "grow", "stabil"), "explode": ("blow", "grow", "stabil"),
    "explosion": ("blow", "grow"), "infinity": ("infinit",),
    "overflow": ("infinit", "grow"), "unstable": ("stabil", "blow"),
    "instabil": ("stabil", "blow"), "unstabl": ("stabil", "blow"),
    "runaway": ("grow", "blow"), "diverg": ("blow", "grow"),
    "enormou": ("enormou", "huge", "grow"), "huge": ("huge", "grow"),
    "bigger": ("grow",), "amplitude": ("oscillat", "grow"),
    "checkerboard": ("oscillat",), "bounc": ("oscillat", "flip"),
    "flip": ("flip", "oscillat"), "wobbl": ("oscillat",),
    # a stalled process
    "deadlock": ("hang", "stuck"), "freeze": ("hang", "stuck"),
    "frozen": ("hang", "stuck"), "wait": ("hang", "stuck"),
    "stall": ("hang", "stuck", "plateau"), "hung": ("hang",),
    # geometry that does not meet
    "gap": ("touch", "meet"), "misalign": ("touch", "meet", "place"),
    "align": ("touch", "meet", "place"), "overlap": ("touch", "meet"),
    "coincid": ("touch", "meet"),
    # magnitude errors, which is what a unit mismatch looks like
    "factor": ("unit", "celsiu", "kelvin"), "scale": ("unit",),
    "scaled": ("unit",), "thousand": ("unit",), "million": ("unit",),
    "magnitude": ("unit",), "convert": ("unit",), "conversion": ("unit",),
    # sign
    "mirror": ("sign", "opposit"), "opposit": ("sign", "opposit"),
    "revers": ("sign", "opposit"), "backward": ("sign", "opposit"),
    "inward": ("sign",), "outward": ("sign",), "negativ": ("sign",),
    # the iteration
    "subiteration": ("iteration",), "substep": ("iteration",),
    "sweep": ("iteration",), "relaxation": ("relaxation", "theta"),
    "underrelaxation": ("relaxation", "theta"),
    "stochast": ("carlo", "sampl", "nois"), "random": ("carlo", "sampl", "nois"),
    "particl": ("carlo", "sampl", "nois"), "dsmc": ("carlo", "sampl", "nois"),
    "plateau": ("plateau", "stop", "fall"),
    "nonsens": ("wrong",), "garbage": ("wrong",), "rubbish": ("wrong",),
    "bogus": ("wrong",), "implausibl": ("wrong",),
}


def _stem(w: str) -> str:
    for suf in ("ings", "ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _toks(s: str) -> list[str]:
    import re as _re
    return [_stem(w) for w in _re.findall(r"[a-zA-Z_]{3,}", s.lower())
            if w not in _STOP]


def _hay(r: dict) -> str:
    """Everything an entry can be matched against — INCLUDING its category.
    Leaving the category out meant that typing the word the row is filed under,
    `conservation` or `relaxation`, matched nothing at all."""
    return " ".join([r["cat"], r["seen"], r["means"], *r["observations"]])


_SEARCH_INDEX: dict = {}


def _search_index() -> dict:
    """Entry token sets plus an IDF weight per token, built once."""
    if _SEARCH_INDEX:
        return _SEARCH_INDEX
    import math
    sets = [set(_toks(_hay(r))) for r in _FAILURE_ROWS]
    n = len(sets)
    df: dict[str, int] = {}
    for s in sets:
        for tok in s:
            df[tok] = df.get(tok, 0) + 1
    scale = math.log(n + 1)
    idf = {tok: math.log((n + 1) / (d + 1)) / scale for tok, d in df.items()}
    prefixes: dict[str, set[str]] = {}
    for tok in df:
        if len(tok) >= 6:
            prefixes.setdefault(tok[:5], set()).add(tok)
    _SEARCH_INDEX.update({"sets": sets, "idf": idf, "prefixes": prefixes,
                          "vocab": set(df)})
    return _SEARCH_INDEX


# A prefix rescue is a GUESS, so it is worth less than a real match, and it is
# never allowed to be the ONLY evidence. `install` shares five characters with
# `instant`, so a conda question was answered with "it converged at iteration 2
# with an exactly zero residual" — a confidently wrong answer, which is the
# worst thing this search can produce. So a prefix match is discounted AND
# cannot supply `core`: it can lift an entry another token already reached, and
# nothing more. That is enough for the case it exists for, a typo inside a
# sentence whose other words are right.
_PREFIX_WEIGHT = 0.8

# An entry must match at least one query token this informative. Coverage alone
# must never be enough: `wrong`, `converging`, `interface` and `mesh` appear
# across most of the table, and an entry selected by those has been selected by
# nothing. This is what stops a one-word query from returning five rows.
#
# Both constants were chosen by SWEEPING them against two independent blind
# query sets — one from each hostile tester — and taking the point where recall
# on the first set is unchanged and refusals are highest. They trade precision
# against recall directly and cannot be tuned to fix both.
_CORE_MIN = 0.55

# The verbatim-containment bonuses are strong evidence and were firing on any
# query short enough to be a substring of somebody's observation — `wrong` is
# inside eight of them. They need a phrase, not a word.
_PHRASE_MIN_CHARS = 12

# WHAT AN UNRECOGNISED WORD COSTS. A word the table has never heard of used to
# be scored at the MAXIMUM weight in the coverage denominator, so every extra
# word an engineer typed pushed the right entry down — `handshake` returned all
# four handshake rows and `handshake never completes` returned nothing. That is
# the exact fault this scorer was rewritten to remove, reintroduced through the
# denominator, and a second blind tester found it by typing sentences instead of
# phrases. An unknown word is weak evidence of anything, so it now costs little.
_UNKNOWN_W = 0.12


def _synonyms_for(tok: str) -> tuple[str, ...]:
    """SYNONYM keys are written as stems, but `_stem` strips only -ing/-ed/-s
    and friends, so `wobble` never became `wobbl` and the entry it pointed at
    was unreachable from the most natural spelling of the word. Try the token,
    its stem, and its e-less form."""
    for k in (tok, _stem(tok), tok[:-1] if tok.endswith("e") else tok):
        hit = SYNONYMS.get(k)
        if hit:
            return hit
    return ()


def _expand(tok: str, vocab: set[str], prefixes: dict) -> dict[str, float]:
    """The entry-vocabulary tokens this query token can stand for, each with the
    confidence it carries."""
    out = {tok: 1.0}
    for s in _synonyms_for(tok):
        out.setdefault(s, 1.0)
    if tok not in vocab and len(tok) >= 6:
        for cand in prefixes.get(tok[:5], set()):
            out.setdefault(cand, _PREFIX_WEIGHT)
    return out


# Above this, an entry is returned. Chosen so that a query whose only match is a
# word appearing across most of the table — `wrong`, `converging`, `interface` —
# falls below it, while a single informative word on its own clears it.
_THRESHOLD = 0.60


def coupling_signal_search(signal: str, limit: int = 5) -> list[str]:
    """Rank the coupling failure entries against a free-text symptom.

    Score per entry:
      3.0   the whole query appears verbatim in the entry
      2.0   the query and one of the entry's plain-language observations
            contain each other
      0.7 * core  + 0.3 * coverage
            `core` is the IDF weight of the single most informative query token
            the entry matches, so extra words cannot take a hit away. `coverage`
            is the share of the query's weight the entry explains, and breaks
            ties towards the entry that accounts for more of what was said; an
            unrecognised word carries almost no weight there, or coverage would
            punish detail through the back door.
    """
    idx = _search_index()
    idf, vocab, prefixes = idx["idf"], idx["vocab"], idx["prefixes"]
    raw = _toks(signal)
    if not raw:
        return []
    # De-duplicate but keep the expansion per distinct token.
    expanded = {t: _expand(t, vocab, prefixes) for t in dict.fromkeys(raw)}
    # An unmatched token still costs coverage: it is part of what was said.
    denom = sum(max((idf.get(e, _UNKNOWN_W) * c for e, c in exp.items()),
                    default=_UNKNOWN_W)
                for exp in expanded.values()) or 1.0
    low = signal.lower().strip()
    phrase = len(low) >= _PHRASE_MIN_CHARS
    scored = []
    for i, (r, entry) in enumerate(zip(_FAILURE_ROWS, coupling_pitfall_entries())):
        toks = idx["sets"][i]
        core = 0.0
        got = 0.0
        for exp in expanded.values():
            hits = [(idf.get(e, 0.0) * c, c) for e, c in exp.items()
                    if e in toks]
            if not hits:
                continue
            w, conf = max(hits)
            got += w
            if conf >= 1.0:                 # a guess never becomes the evidence
                core = max(core, w)
        score = 0.7 * core + 0.3 * (got / denom) if core >= _CORE_MIN else 0.0
        if phrase and low in _hay(r).lower():
            score += 3.0
        if phrase:
            for o in r["observations"]:
                if low in o.lower() or o.lower() in low:
                    score += 2.0
                    break
        if score > _THRESHOLD:
            scored.append((score, entry))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:limit]]


_FAILURES = '''\
## 7. FAILURE MODES, and what each one actually means

''' + _failure_table() + '''

Reach these from a SYMPTOM instead of reading the table:
`knowledge(topic='coupling', signal='<what you actually saw>')` — describe the
observation in your own words, no mechanism needed, e.g.
`signal='it converged but the answer is wrong'`.

AND THERE IS A SECOND INDEX, which you want when there is NOTHING to paste.
The table above is what this tool PRINTS. The failures that cost the most print
nothing at all — a unit mismatch, a wrongly applied interface sign, a
participant that never reads its imports, a lossy mapping — and those live as
corpus entries whose Signal clauses say outright that you would see nothing,
rather than inventing a symptom you would not:

    knowledge(topic='pitfalls', solver='coupling', signal='<what you saw>')
    knowledge(topic='pitfalls', solver='coupling', physics='silent_wrong')

The groups are silent_wrong, participant_contract, verification_limits,
capability_limits and fsi. They are also the entries the project's own coverage
criterion counts for coupling, so they are the ones with fixtures behind them.
The `fsi` group is the fluid-structure-specific set — the traction sign, the
added-mass instability, the ALE mesh, and the two modes in which a one-way or
sign-flipped FSI passes every conservation check there is.

FIRST THING TO DO WHEN A COUPLING FAILS: delete `imports.json` from each
work_dir and run each participant script by hand there. That exercises the
iteration-1 fallback path and tells you whether the failure is in the physics
or in the handshake. Nearly every coupling failure is visible there.

SECOND THING: check that the interface really is shared. Print each side's
`n_points` and the first and last coordinate. Two subdomains that do not touch,
or touch at a coordinate one of them rounds differently, produce exactly the
"converged to the wrong answer" symptom.
'''


def _index(names: list[str]) -> str:
    rows = "\n".join(
        f"  knowledge(topic='coupling', solver='{n}', signal='participant')"
        for n in names)
    return f'''\
## 8. PER-BACKEND PARTICIPANT SCRIPTS — one call each, complete and runnable

Each of these returns the participant CONTRACT for that backend (solve elided) plus the
traps specific to it. Copy it into the participant's `work_dir`, edit the
marked block, run it once by hand, then call `couple`.

For a separately shipped role or variant, request it explicitly, for example
`signal='participant:neumann'`, `signal='participant:elastic'`,
`signal='participant:transient'`, or `signal='participant:3d'`.

{rows}

FLUID-STRUCTURE INTERACTION is its own pattern rather than a backend — a
vector traction one way, a displacement the other, and the fluid consuming it
by moving its mesh:

  knowledge(topic='coupling', solver='fsi')

preCICE instead of this driver:  knowledge(topic='precice', solver='<name>')
4C-native thermo-structural:     knowledge(topic='tsi')
'''


_BACKEND_ORDER = ["fenics", "fourc", "ngsolve", "skfem", "dune", "dealii",
                  "febio", "kratos", "sparta"]


_RECAP = '''\
## The contract, in case you landed here first

Your script runs in `work_dir` with no arguments. It reads `imports.json`
(`{partner_name: InterfaceData}` — ABSENT or empty on iteration 1, so it must
have a fallback) and writes `exports.json` (ONE InterfaceData:
`{"field_name","n_points","coordinates","values","normal_fluxes"}`). Export the
same number of points in the same order every iteration, write `exports.json`
last, and EXIT 0 — the driver requires the file AND a zero exit code, so a
solver that writes its last iterate and then aborts ends the run instead of
being coupled on. The driver does NOT copy your script into `work_dir` and does
NOT interpolate between the two meshes.

Dirichlet side = imports a VALUE, exports the resulting FLUX.
Neumann side  = imports a FLUX, exports the resulting VALUE.
Apply the partner's flux number UNCHANGED; export your own flux with respect to
YOUR outward normal, so the two sides' fluxes carry opposite signs.

Full contract, relaxation guidance and failure modes: `knowledge(topic='coupling')`.
'''


def _threed_block(script_name: str) -> str:
    """The 3-D participant, when one ships for this backend.

    Every participant in the corpus meshes a rectangle and samples the
    partner along a straight interface by ONE coordinate. In 3-D the
    interface is a plane, and the three things that change are all invisible
    until they cost you an order.
    """
    p = _PARTICIPANT_DIR / f"participant_{script_name}_3d.py"
    if not p.is_file():
        return ""
    return (
        "\n## THE 3-D PARTICIPANT — a plane interface, not a line\n\n"
        "Three things differ from 2-D, each measured rather than asserted:\n\n"
        "  * QUADRATURE WEIGHTS ARE FACE AREAS, not edge lengths. On the same "
        "converged solve the area weights give the interface integral as "
        "-22.154 against an exact -22.026, while the 2-D edge formula gives "
        "-0.923 AND DIVERGES under refinement — scaled by ~h — while the "
        "flux field itself still looks smooth and the right shape. That is "
        "the failure mode where conservation breaks and nothing else looks "
        "wrong.\n"
        "  * RESAMPLING ORDER MATTERS TO THE PARTNER, NOT TO YOU. Cubic "
        "against bilinear interpolation between non-matching interface "
        "grids, everything else identical: the VOLUME field converges at "
        "1.98 either way, but the interface flux the partner consumes falls "
        "from 1.97 to 1.32. Your own answer does not notice; the coupling "
        "does.\n"
        "  * THE RIM IS A BIGGER SHARE OF A PLANE THAN THE ENDS ARE OF A "
        "LINE. Nodes on the interface rim that also sit on an outer boundary "
        "carry that boundary's reaction too: 96 of 625 nodes (15%) at n=24 "
        "in 3-D against 2 of 25 (8%) in 2-D. Unguarded, the whole-interface "
        "flux loses its order outright (0.52) and the L2 error inflates by "
        "19x to 83x.\n\n"
        f"```python\n{_serve_participant(p)}```\n")


def _transient_block(script_name: str) -> str:
    """The time-dependent participant, when one ships for this backend.

    Every participant in the corpus used to be STEADY, and the word
    "transient" appeared nowhere in the core coupling text. A coupled problem
    with time-dependent physics therefore had no starting point at all.
    """
    p = _PARTICIPANT_DIR / f"participant_{script_name}_transient.py"
    if not p.is_file():
        return ""
    return (
        "\n## THE TRANSIENT PARTICIPANT — time-dependent coupling\n\n"
        "Two things about a time-dependent coupling that a steady one never "
        "makes you decide:\n\n"
        "  * WHAT ONE INVOCATION COVERS. This participant marches the WHOLE "
        "time window per call and exchanges the entire trace: `values[i][n]` "
        "is the value at step n, `normal_fluxes[i][n]` the flux over step n. "
        "Coupling once per time step is NOT reachable inside a single "
        "`couple` call — InterfaceData carries exactly five keys, so there is "
        "nowhere to say which step a payload belongs to; the driver re-runs "
        "participants on identical imports to measure sensitivity, so hidden "
        "time state reads as noise; and the convergence test is 'the export "
        "stopped changing', which a self-advancing participant satisfies "
        "while its partner is at a different step. Per-step coupling means "
        "owning the outer loop yourself, one driver call per step, with "
        "restart files.\n"
        "  * THE EXPORTED FLUX IS THETA-AVERAGED. A theta-scheme residual "
        "recovers `theta*q^(n+1) + (1-theta)*q^n`, not the flux at the new "
        "time, and that average is exactly what the Neumann side's "
        "theta-combined right-hand side needs. Exporting it as 'the flux at "
        "t^(n+1)' injects an O(dt) error that looks like a scheme stuck at "
        "first order.\n\n"
        f"```python\n{_serve_participant(p)}```\n"
        + _program_behind(script_name, "transient"))


def _program_behind(script_name: str, request: str) -> str:
    """The program a wrapper runs, with its CMakeLists.txt, to sit right behind the
    wrapper's block ('' for a contract with no program). A deal.II transient side
    was served its wrapper alone, and the steady program sat past the reply's cut
    (measured on a transient coupled round)."""
    files, err = participant_companions(script_name, request)
    if not files and not err:
        return ""
    if err:
        return f"\n[{err}]\n"
    return ("\n" + _dealii_program_head(files[0][0]) + f"```cpp\n{files[0][1]}```\n\n"
            + "".join(f"```cmake\n{tx}```\n" for _nm, tx in files[1:]))


def _role_block(script_name: str) -> str:
    """The OTHER interface role, when a backend ships a second participant.

    A backend's payload used to carry exactly one script. Where that script
    implements only one side — Kratos shipped the Dirichlet side and nothing
    else — an agent handed the opposite role got prose and had to write the
    participant itself. Two coupled problems in the recorded runs needed
    Kratos on the Neumann side; both failed.
    """
    p = _PARTICIPANT_DIR / f"participant_{script_name}_neumann.py"
    if not p.is_file():
        return ""
    return (
        "\n## THE NEUMANN-SIDE PARTICIPANT — a separate contract for the other role (solve elided)\n\n"
        "The script above is the DIRICHLET side: it imports the partner's "
        "`values`, fixes them, and exports the consistent reaction flux. This "
        "one is the other half: it imports the partner's `normal_fluxes`, "
        "applies them UNCHANGED as the natural boundary condition, and "
        "exports the interface values its solve produced. Use whichever role "
        "the problem assigns this subdomain; they are not interchangeable.\n\n"
        f"```python\n{_serve_participant(p)}```\n")


def _vector_block(script_name: str) -> str:
    """The VECTOR (elasticity) participant, when one ships for this backend.

    Four of these — fenics, ngsolve, skfem, dealii — sat in
    data/coupling_participants/ and were served to nobody: the per-backend
    payload only ever carried the scalar heat script, so an agent asked to
    couple ELASTICITY was handed a temperature participant and one line of
    prose ("replace temperature with displacement, flux with traction"). Four
    coupled problems in the recorded runs were vector problems; the agents
    rewrote from scratch and ran out of budget.

    Appended automatically wherever the file exists, so adding a backend's
    vector participant to the directory is enough to serve it.
    """
    p = _PARTICIPANT_DIR / f"participant_{script_name}_elastic.py"
    if not p.is_file():
        return ""
    return (
        "\n## VECTOR (ELASTICITY) VARIANT — use this one when the coupled "
        "field is displacement\n\n"
        "The exchanged quantity is a TRACTION vector, not a scalar flux: "
        "`values` and `normal_fluxes` carry one entry PER COMPONENT per "
        "interface point, in the same node order on both sides. The Dirichlet "
        "side imports displacements and exports the traction it needed; the "
        "Neumann side imports that traction and exports the displacement it "
        "produced. Recover the traction variationally (the interface "
        "reactions of the assembled residual), never by differencing the "
        "displacement field — a differenced traction converges one order too "
        "slowly and drags the coupled field order down with it.\n\n"
        f"```python\n{_serve_participant(p)}```\n"
        + _program_behind(script_name, "elastic"))




def _thermo_notice(script_name: str) -> str:
    """One paragraph after the payload's title, for a backend that ships a
    thermo-elastic contract: the physics-less call is told where the other
    contract is. Measured 2026-09-11: a thermo-elastic cell's worker called the
    door without physics, got the scalar contract, invented a TSI deck from
    memory ("TSI CONTROL is not a valid section") and filed a blocker."""
    have = [(lbl, phys) for key, lbl, phys in (
        ("thermoelastic", "the interface carries TEMPERATURE AND DISPLACEMENT together "
                          "(T, ux, uy in; heat flux and traction out)", "thermoelastic"),
        ("elastic", "the exchanged field is a DISPLACEMENT (a traction returned)", "elasticity"),
        ("transient", "the problem is TIME-DEPENDENT (the served contract marches the whole "
                      "time window per call and exchanges the trace)", "transient"),
        ("3d", "the domain is THREE-DIMENSIONAL with a planar interface", "3d"))
        if (_PARTICIPANT_DIR / f"participant_{script_name}_{key}.py").is_file()]
    # THE FSI CONTRACTS ARE NAMED THE OTHER WAY ROUND -- participant_fsi_solid_<code>.py -- so the
    # loop above cannot see them, and until 2026-09-14 the only way to reach them was to pass 'fsi'
    # as if it were a SOLVER name. An agent coupling a fluid to a structure calls this door with its
    # own code and, without this line, is handed the single-field contract with nothing to suggest
    # otherwise.
    if any((_PARTICIPANT_DIR / f"participant_fsi_{part}_{script_name}.py").is_file()
           for part in ("solid", "fluid")):
        have.append(("the interface carries a FLUID TRACTION and a STRUCTURAL DISPLACEMENT "
                     "(a fluid-structure task)", "fsi"))
    if not have:
        return ""
    # Same defect for every variant, measured 2026-09-11 (thermo-elastic) and
    # 2026-09-13 (vector, transient, 3-D): without the physics word the reply
    # leads with the scalar contract and the variant sits past every cut.
    return ("\nWHICH CONTRACT BELOW IS YOURS. The block below is the single-field STEADY 2-D "
            f"(scalar) contract. If instead\n"
            + "".join(f"  * {lbl}: call knowledge(topic='coupling', solver='{script_name}', "
                      f"physics='{phys}')\n" for lbl, phys in have)
            + "and use the contract THAT reply leads with. Without the physics word this door "
              "leads with the single-field (scalar) contract.\n")


def _thermoelastic_block(script_name: str) -> str:
    """The THERMO-ELASTIC participant (temperature AND displacement through
    one interface state), when one ships for this backend.

    Measured 2026-09-11 on the thermo-mechanical coupled cell: three runs
    that were handed the scalar heat contract and the sentence "4C has no
    2-D TSI element" filed give-ups, one of them with the solves already on
    disk. The contract exists for two codes now (4C as the Dirichlet side --
    two runs per iteration, 4C's own boundary flux and reaction monitor as
    the recovery -- and FEniCSx in both roles), validated by execution at
    three mesh levels against a manufactured thermo-elastic solution.
    Appended wherever the file exists; `knowledge(topic='coupling',
    solver=..., physics='thermoelastic')` serves it FIRST.
    """
    p = _PARTICIPANT_DIR / f"participant_{script_name}_thermoelastic.py"
    if not p.is_file():
        return ""
    return (
        "\n## THERMO-ELASTIC VARIANT — temperature AND displacement through ONE "
        "interface (solve elided)\n\n"
        "Use this contract when the task transmits temperature and displacement "
        "together across the interface (steady thermoelasticity split into "
        "subdomains). `values` = [T, ux, uy] and `normal_fluxes` = [qn, qx, qy] "
        "per interface point, the traction in the SAME sign convention as the "
        "heat flux -- minus the flux of the conserved quantity through this "
        "side's outward normal -- so the two sides' exports cancel componentwise "
        "and the Neumann side applies the partner's numbers UNCHANGED. A task "
        "that asks for the OUTWARD traction sigma.n_out gets minus the exported "
        "(qx, qy). Both are recovered from THIS side's own assembled systems "
        "(4C: its own boundary-flux VTU and its Dirichlet reaction monitor), "
        "never by differencing a P1 field on the boundary.\n\n"
        + f"```python\n{_serve_participant(p)}```\n")


def _scaffold_first(traps: str):
    """When a backend's traps carry a config-driven scaffold (a fenced block
    that reads ./config.json -- 4C and DUNE-fem tonight), that scaffold IS
    the contract: it is what prepare_simulation's reveal serves, and it is
    the one validated by execution. Measured 2026-09-11: the door's first
    block was the older file contract while the scaffold sat 30k characters
    later under the traps, so the first reply of a session served one script
    and the reveal another. Returns (scaffold_block, traps_without_it) or
    (None, traps)."""
    pos = 0
    while True:
        i = traps.find("```python", pos)
        if i < 0:
            return None, traps
        j = traps.find("```", i + 9)
        if j < 0:
            return None, traps
        block = traps[i:j + 3]
        if "config.json" in block and "imports.json" in block and "exports.json" in block:
            remainder = traps[:i] + "(the scaffold itself is the PARTICIPANT CONTRACT block above)" + traps[j + 3:]
            return block, remainder
        pos = j + 3


def _payload(title: str, sides: str, script_name: str, launch: str,
             traps: str, extra: str = "") -> str:
    scaffold, traps = _scaffold_first(traps)
    contract = scaffold if scaffold else f"```python\n{lean_view(_script(script_name))}```"
    return ("## If your client truncates long replies: fetch this contract in parts\n\n"
      f"Call `knowledge(topic='coupling', solver='{script_name}', "
      "signal='participant:part1')`, then request each next part named "
      "in that response and concatenate only the fenced code contents "
      "in order. For another role insert it before the part, for example "
      "`signal='participant:neumann:part1'`. The parts are the same "
      "contract as below, solve elided; there is no complete program to "
      "reconstruct.\n\n"
      f"# Coupling participant: {title}\n"
            f"{_thermo_notice(script_name)}\n"
            f"## Sides this backend can take\n\n{sides}\n\n"
            f"{_RECAP}\n"
            f"## PARTICIPANT CONTRACT — PUT THIS IN ITS OWN FILE NOW with "
            f"write_participant_contract(solver=<code>, path=<side dir>/participant_<x>.py, "
            f"variant=<the physics word>): it writes this same code, its "
            f"comments kept, so nothing is re-typed. Then edit the marked block IN PLACE "
            f"and write the solve where the banner sits. It is the handshake, the "
            f"interface sign convention, the consistent flux recovery, the "
            f"exports schema and the export self-check, thinned of its "
            f"explanations so it is cheap to copy; the annotated version is "
            f"`signal='participant:part1'`. Measured: every recent coupled "
            f"run that wrote this file from scratch instead failed on the "
            f"handshake or the recovery.\n\n{contract}\n\n"
            # the OTHER role's contract comes right behind the first one, so the
            # first reply of a session (cut behind the contract blocks) carries
            # both roles -- measured 2026-09-11: a task that made Kratos the
            # Neumann side got only the Dirichlet block in that reply and the
            # parent hand-rolled the Neumann side and lost
            f"{_role_block(script_name)}"
            f"## Launching it\n\n{launch}\n"
            f"## {title}-specific traps\n\n{traps}\n{extra}"
            f"{_vector_block(script_name)}"
            f"{_thermoelastic_block(script_name)}"
            f"{_transient_block(script_name)}"
            f"{_threed_block(script_name)}")


_RIGHT_BLOCK = """\
SIDE      = "neumann"     # this copy takes the OTHER side
PARTNER   = "left"        # the name you gave the first participant
X0, X1    = 0.6, 1.1      # the OTHER subdomain: starts where the first ends
Y0, Y1    = 0.0, 0.4      # same y-extent as the partner
IFACE_X   = 0.6           # SAME interface coordinate as the partner
K         = 1.5           # this subdomain's own material


def F_SRC(x, y):          # this subdomain's own source; see the script's docstring
    return np.zeros_like(x)


T_OUTER   = 300.0         # Dirichlet on ITS outer boundary (here x = 1.1)
NX, NY    = 18, 12        # its own mesh — deliberately NOT the partner's
T_INIT    = 310.0
Q_INIT    = 0.0
"""

_LAUNCH_PY = '''\
1. Make TWO copies of the script, one per subdomain, and write each into that
   participant's own `work_dir` (an absolute path). Name them whatever you
   reference in `command` — `participant_left.py` and `participant_right.py`
   below. The driver copies no script for you.
2. {STEP2}
3. Run each copy BY HAND in its own directory first, with no `imports.json`
   present (delete a leftover one from an earlier attempt — the driver never
   removes it). Each must print its interface line and write `exports.json`.
   A coupling cannot repair a participant that never ran.
4. {INTERP}
5. Have a critic review the setup, then put the review on record — the flag
   alone is NOT enough, openPASO looks the review up rather than believing you:

```
submit_critic_review(solver="couple",
                     coupling_args='{{"participants": "<the same JSON string>",
                                     "max_iter": 60, "tol": 1e-8,
                                     "accelerator": "aitken", "theta": 0.5,
                                     "monolithic": "",
                                     "probe": true}}',
                     findings="<what the critic checked and concluded>")
```

   EVERY argument that `couple` hashes must appear here, INCLUDING the two
   with defaults. The digest covers participants, max_iter, tol, accelerator,
   theta, monolithic AND probe. This example used to omit the last two, so an
   agent that copied it verbatim — changing nothing — got back "a critic
   review exists for this solver but NOT for this setup: the input changed
   after it was reviewed". Nothing had changed; the recipe was short. If you
   see that message and you did not edit anything, this is why: add the
   arguments you left at their defaults.

6. Then couple. The `coupling_args` above must match these arguments exactly,
   or the review does not bind and the result comes back NOT VERIFIED:

```
couple(participants='[
  {{"name":"left","command":["<interpreter>","participant_left.py"],
   "work_dir":"left","imports_from":["right"],"timeout":900}},
  {{"name":"right","command":["<interpreter>","participant_right.py"],
   "work_dir":"right","imports_from":["left"],"timeout":900}}]',
  max_iter=60, tol=1e-8, accelerator="auto", theta=0.5, critic_approved=True)
```
'''.replace("{RIGHT}", _RIGHT_BLOCK)

# Step 4 differs for the four backends whose participant is a Python WRAPPER
# around a separate binary (4C, FEBio, SPARTA, deal.II). For those, `command`
# must run PYTHON and the binary path goes into a CONSTANT INSIDE the script.
# "Get the interpreter or binary from discover(query='list')" is the wrong
# instruction there: what discover prints for those backends IS the binary, and
# a model that follows step 4 verbatim puts the solver binary in `command`,
# where it cannot run a Python wrapper.
#
# This was previously done with `_LAUNCH_PY.replace(<literal>, ...)` at four
# call sites. ALL FOUR WERE DEAD: the literals wrapped their lines at a
# different point than the text (and 4C's was left from an older wording
# entirely), so every wrapper backend served the generic step 4 and not one of
# them said `command` runs the wrapper. `str.replace` cannot fail, so nothing
# reported it. Step 4 is a NAMED FIELD now, so a stale note raises at import
# instead of going quiet, and `test_wrapper_backends_say_the_command_runs_the
# _wrapper` asserts on the SERVED text rather than on the call.
_INTERP_GENERIC = (
    "Get the interpreter from `discover(query='list')`: it prints one\n"
    "   line per backend with the exact interpreter path on THIS install.\n"
    "   Never copy an interpreter path out of an example.")

# The RULES step 2 states are the same whatever the physics; only the concrete
# block changes. Kept separate so a bespoke step 2 can reuse them.
_TWO_BLOCK_RULES = '''\

   The rules the two blocks must satisfy, whatever your real numbers are:
   one copy has `SIDE="dirichlet"` and the other `SIDE="neumann"`; each
   `PARTNER` is the other's `name`; both have the SAME `IFACE_X`; the two
   x-extents meet at `IFACE_X` and do not overlap; each subdomain keeps at
   least one Dirichlet boundary of its own, or its problem is singular and the
   solve blows up. The two meshes need NOT match.'''


def _step2_block(block: str, what: str = "the placeholder problem") -> str:
    return ("The script AS SHIPPED is the LEFT / Dirichlet side. In the second "
            f"copy,\n   replace the edit block with the complementary side. For "
            f"{what} that is\n   exactly:\n\n```python\n{block}```\n"
            + _TWO_BLOCK_RULES)


# Step 2 embedded the CONDUCTION right-side block into every backend's payload.
# For the three backends whose shipped script does not solve that problem it
# named constants that do not exist in the script the agent had just been given
# (FEBio has no K/T_OUTER/T_INIT/F_SRC — it is elasticity; Kratos has no
# SIDE/IFACE_X/Y0,Y1/NX,NY/F_SRC/Q_INIT; SPARTA has none of nine of them). For
# SPARTA it also flatly contradicted the same payload's own headline, which says
# the Neumann role is IMPOSSIBLE, by handing over a `SIDE="neumann"` block.
# Verified by execution: applying the served block to the served FEBio script
# fails on the first key. So step 2 is per-backend now.
_STEP2_DEFAULT = _step2_block(_RIGHT_BLOCK)

# FEBio's script is the ELASTIC analogue, so its complementary block is in
# displacement/modulus, not temperature/conductivity. These constant names are
# the ones the shipped FEBio script actually defines; the pairing was run as a
# real FEBio-to-FEBio coupling and converged with non-matching meshes.
_RIGHT_BLOCK_FEBIO = """\
SIDE      = "neumann"     # this copy takes the OTHER side
PARTNER   = "left"        # the name you gave the first participant
X0, X1    = 0.5, 1.0      # the OTHER subdomain: starts where the first ends
Y0, Y1    = 0.0, 1.0      # same cross-section as the partner
Z0, Z1    = 0.0, 0.1      # same cross-section as the partner
IFACE_X   = 0.5           # SAME interface coordinate as the partner
E_MOD     = 2250.0        # this subdomain's own material
NU        = 0.29
U_OUTER   = 1.0e-4        # prescribed u_x on ITS outer face (here x = 1.0)
NX, NY    = 14, 6         # its own mesh — deliberately NOT the partner's
U_INIT    = 5.0e-5
Q_INIT    = 0.0
"""

# Kratos ships a DIRICHLET-side script with no `SIDE` switch, so there is no
# "flip SIDE" edit to make: the Neumann side is a code change, described under
# the traps. SPARTA cannot take the Neumann side at all.
_STEP2_KRATOS = ('''\
The shipped script is the DIRICHLET side and has NO `SIDE` switch, so
   the second participant is NOT this script with one constant flipped. Two
   ways to build the pair, and the first is what was actually run:

   (a) PAIR IT WITH ANOTHER BACKEND. Take the Neumann side from a backend whose
       script has a `SIDE` switch — `knowledge(topic="coupling",
       solver="fenics")` is the lightest — and edit only its block. Kratos was
       proven against FEniCSx this way, in both roles.
   (b) MAKE A KRATOS NEUMANN SIDE. Copy the script and change the interface
       condition as described under the traps below: do NOT `Fix(TEMPERATURE)`
       on the interface nodes; set `FACE_HEAT_FLUX` from the partner's
       `normal_fluxes` and create the interface `ThermalFace` conditions.
       Nothing else — mesh, material, export — changes.

   For its own edit block, the second participant's subdomain must start where
   the first ends (`X0` = the first copy's `X1`), keep the same `H`, carry its
   own `K` and its own outer `T_OUTER`, and name the partner in `PARTNER`. Each
   subdomain must keep one Dirichlet boundary of its own or it is singular.''')

_STEP2_SPARTA = ('''\
THERE IS NO SECOND COPY OF THIS SCRIPT. SPARTA is a DIRICHLET-side
   participant only — no surface-collision model accepts a prescribed heat
   flux — so it cannot take the complementary role, and a `SIDE="neumann"`
   edit of this script does not exist.

   The partner is a NEUMANN-side participant in another backend: it imports
   SPARTA's exported `normal_fluxes` as its interface flux and exports the wall
   temperature SPARTA imports. `knowledge(topic="coupling", solver="fenics")`
   gives such a script; set its `SIDE="neumann"`, put its interface at the wall
   SPARTA's surface file describes, and make sure both sides agree on units —
   SPARTA works in SI with energy flux per unit area.

   Read the stochasticity note below BEFORE you size `NRUN`/`NAVE`: a
   Monte-Carlo estimate has a noise floor the residual cannot fall below, so a
   plain `tol` makes `couple` report FAILURE on a coupling whose physics agrees.
   Set `SEED_MODE = "vary"` and pass `noise_replicates=5`.''')


def _launch_py(interp: str = _INTERP_GENERIC, step2: str = "") -> str:
    """The launch section, with steps 2 and 4 written for this backend."""
    for field in ("{INTERP}", "{STEP2}"):             # guard the guard
        if field not in _LAUNCH_PY:
            raise AssertionError(f"_LAUNCH_PY lost its {field} field")
    return (_LAUNCH_PY.replace("{INTERP}", interp)
                      .replace("{STEP2}", step2 or _STEP2_DEFAULT))


def _interp_wrapper(binary: str, const: str, extra: str = "") -> str:
    """Step 4 for a wrapper participant: python in `command`, binary in a const."""
    return (f"The `command` runs the WRAPPER, so the interpreter is a plain\n"
            f"   Python with numpy — NOT the {binary} binary. The {binary} BINARY\n"
            f"   path goes into `{const}` INSIDE the script; that is what\n"
      f"   `discover(query='list')` prints for this backend. If the wrapper\n"
      f"   invokes that binary with `capture_output=True`, write the child\n"
      f"   stdout and stderr back to `sys.stdout` and `sys.stderr` on EVERY\n"
      f"   run, including success. openPASO can preserve only bytes the wrapper\n"
      f"   emits; printing a hand-written success summary is not native solver\n"
      f"   evidence.{extra}")


def coupling_core() -> str:
    return (
        "# Cross-code coupling with openPASO — `couple`\n\n"
        "## 0. WHICH TOOL, IN ONE PARAGRAPH\n\n"
        "`couple(participants, ...)` is THE tool. It is physics-agnostic: you write "
        "one self-contained solver script per subdomain, and openPASO runs the "
        "fixed-point iteration, the relaxation, the convergence-or-fail and the "
        "conservation check. `couple_precice(...)` is the alternative when each side "
        "is a real preCICE participant and you want preCICE's mapping and implicit "
        "schemes. `coupled_solve(...)` is DEPRECATED — it only reproduces a fixed enum "
        "of benchmark problems on a hard-coded unit square and cannot express your "
        "problem; do not start there.\n\n"
        + _SELF_CHECK_FIRST + "\n" + _CONTRACT + "\n" + _DRIVER_BEHAVIOUR + "\n" + _SIGNS + "\n"
        + _PROBES + "\n" + _DEALII_BUILD + "\n" + _HISTORY_NAN + "\n" + _RESIDUAL_IS_A_RECORD + "\n" + _NGSOLVE_DOFS + "\n"
        + _SIDES + "\n" + _FAILURES + "\n" + _index(_BACKEND_ORDER)
    )


# ══════════════════════════════════════════════════════════════════════════
# PER-BACKEND — served by knowledge(topic='coupling', solver=...)
# ══════════════════════════════════════════════════════════════════════════

def _fenics() -> str:
    return _payload(
        "FEniCSx (dolfinx)",
        "**Either side.** Both were run as real cross-code couplings against a "
        "second code on this install: FEniCSx as the Dirichlet side against 4C "
        "on Neumann, and FEniCSx as the Neumann side against 4C on Dirichlet. "
        "Both converged with non-matching interface meshes.",
        "fenics", _launch_py(),
        '''\
* `LinearProblem` in dolfinx 0.9+ REQUIRES `petsc_options_prefix`. Omit it and
  the constructor raises before anything is solved.
* An interface Dirichlet BC from partner data is a `fem.Function` whose array
  you fill at the interface DOFs, then `fem.dirichletbc(function, dofs)` — the
  two-argument form. The scalar form `fem.dirichletbc(value, dofs, V)` cannot
  carry a per-node profile.
* For the Neumann side you need `meshtags` + a subdomain `ds` measure. A bare
  `ufl.ds` applies the flux to the WHOLE boundary, including the outer
  Dirichlet edge, which is silently wrong rather than an error.
* `dmesh.meshtags` wants the facet indices SORTED. Unsorted indices are
  accepted and then tag the wrong facets.
* `V.tabulate_dof_coordinates()` gives the DOF coordinates that
  `x.array` is indexed by — use those, not the mesh geometry nodes, or the
  interface values land on the wrong entries for anything above P1.
* The exported flux is the CONSISTENT (reaction) flux, not a projected
  gradient: assemble the residual against the VOLUME load alone,
  `r = A u_h - b_vol` with no boundary condition applied and the constrained
  rows NOT zeroed, and divide by `w_i = int_Gamma phi_i ds` to get a density,
  `q_i = -r_i / w_i`. This is ONE expression for both sides — on the Dirichlet
  side there is no interface term so `b == b_vol`, and on the Neumann side the
  same rows carry exactly the interface functional the partner applied.
  MEASURED against an ANALYTIC interface flux on the Dirichlet side, on
  8/16/32/64 meshes — a manufactured `T = 300 + sin(3x) cos(pi y/Ly)` whose
  exact flux `-3 K cos(3 X1) cos(pi y/Ly)` is NOT the number the participant is
  handed: over the interior interface nodes the recovery is second order. At
  the two nodes where the interface meets the outer boundary it is only FIRST
  order, so report and export those apart. The L2 projection of `-K*S*grad(T)[0]` that
  this guidance used to recommend is order ~1 in the interior away from the
  ends (0.93), 0.50 in rms, and does not converge at all in the max norm that
  includes the near-end nodes, where it stalls at 2.6 against a true flux of
  size 2 to 5 — order ~1 is simply what a P1 gradient evaluated ON a boundary
  is worth. The recovery, not the physics and not the partner, sets the
  observed order on a coupled task.
* DO NOT VERIFY THE RECOVERY BY HANDING THE NEUMANN SIDE A FLUX AND ASKING FOR
  IT BACK. On that side `A u = b_vol + M_Gamma g`, so the free interface rows
  give `r = A u - b_vol = M_Gamma g` IDENTICALLY and the export is just
  `-(M_Gamma g)/(M_Gamma 1)`, the consistent-to-nodal conversion of the P1
  boundary mass matrix. Its offset from `-g` is `-(h^2/6) g''(y)`, i.e. "second
  order", for any correct assembly of any equation with any material — a bare
  NumPy mass matrix reproduces the same numbers with no PDE in it. That round
  trip is a useful check of your SIGN CONVENTION, your weight `w_i`, your facet
  tags and your dof mapping, and it is worth running for those. It is not
  evidence of an order. Measure the order on the Dirichlet side against a
  manufactured solution, as above.
* And do not finite-difference towards an "adjacent" node either: on an
  unstructured triangle mesh the nearest interior node is not normal to the
  interface.
* Run FEniCSx participants serially (one MPI rank). Under `mpirun` each rank
  would write its own `exports.json` over the others.''')


try:
    from backends.fourc.deck_grammar import FOURC_DECK_GRAMMAR as _FDG
except Exception:                                   # pragma: no cover
    _FDG = ""
# Re-indent the shared grammar as a bullet inside the 4C payload's list, so the
# single copy in backends/fourc/deck_grammar.py serves BOTH this coupling path
# and the single-code path.
_FOURC_GRAMMAR_BULLET = "\n  ".join(_FDG.splitlines())


def _fourc() -> str:
    return _payload(
        "4C Multiphysics",
        "**Either side.** Both were run as real cross-code couplings against "
        "FEniCSx on this install and both converged with non-matching interface "
        "meshes. Note this contradicts the deprecated `coupled_solve` docstring, "
        "which lists 4C on the Neumann side only — that limitation belongs to "
        "the legacy tool's own generators, not to 4C.",
        "fourc",
        _launch_py(_interp_wrapper(
            "4C", "FOURC_BIN",
            extra="\n   That Python needs numpy + meshio (openPASO's own has both). Put\n"
                  "   4C's dependency lib directory in `FOURC_LD` if the binary does\n"
                  "   not find its libraries by itself.")),
        '''\
* ''' + _FOURC_GRAMMAR_BULLET + '''

* `PROBLEMTYPE: "Scalar_Transport"` with `TIMEINTEGR: "Stationary"` is the
  conduction problem. Element line is `TRANSP QUAD4 ... MAT 1 TYPE Std` (P1
  triangles: `TRANSP TRI3 <n1> <n2> <n3> MAT <m> TYPE Std`) in a
  `TRANSPORT ELEMENTS` section — `SOLID QUAD4`/`SOLID TRI3` are structural
  elements and will be rejected against `MAT_scatra`.
* A SPATIALLY VARYING boundary datum has an EXACT route: one
  `DESIGN POINT DIRICH` condition PER INTERFACE NODE (each with its own DNODE
  topology entry and the sampled value in VAL). Imported interface samples go
  in node-by-node, no fit, no fit error — this is the route to use for
  coupling imports, verified by execution. The `VAL: [1.0]` x `FUNCT: [1]`
  route (FUNCT1 a `SYMBOLIC_FUNCTION_OF_SPACE_TIME`; 4C multiplies the two)
  exists for data you hold as an EXPRESSION; fitting a polynomial to samples
  just to feed it is a silently wrong boundary condition when the fit is bad,
  so reserve it for closed-form data.
* The same `VAL x FUNCT` rule holds for `DESIGN LINE NEUMANN CONDITIONS`, and
  4C's Neumann `VAL` is exactly the flux quantity the partner exported. Hand it
  over unchanged.
* VTU output lands in `out-vtk-files/scatra-<step>-<rank>.vtu`. THE TRAILING
  NUMBER IS THE MPI RANK, NOT THE STEP. Sorting on the last number returns
  `scatra-00000-0.vtu`, which is the INITIAL CONDITION — an all-zero field that
  looks like a converged solve of a trivial problem. Parse the FIRST number.
* The scalar field is named `phi_1`, never `temperature`.
* THE TWO-SIDED PARTICIPANT SCAFFOLD (config-driven; "side": "dirichlet" |
  "neumann" in ./config.json is the role the task gives 4C's subdomain; both
  roles executed on this binary against a manufactured solution). The handshake, 4C's
  CALCFLUX_BOUNDARY flux-recovery route and the exports schema are served in the
  block below; THE 4C DECK AND THE SOLVE ARE ELIDED -- write them from
  `prepare_simulation(solver='fourc', physics='<your physics>')` and drop them
  into the marked region. Measured facts it still encodes: 4C's own
  CALCFLUX_BOUNDARY is assembly-consistent (a hand re-assembly on a different
  element is first order -- it cut the coarse interface imbalance 20x,
  0.54 -> 0.026, and lifted the jump order from ~0.8 to ~2 on the measured
  interior; delivery proven zero-vs-real, field moved 1.38e-1 vs 0, recovered
  flux -0.75 against applied +0.75), and two traps live in the elided deck --
  the GLOBAL DNODE-id rule across condition families, and that this build writes
  scatra VTU with no VTK section.

```python
"""4C as EITHER side of a partitioned coupling (Scalar_Transport).

THE ROLE COMES FROM ./config.json: "side": "neumann" | "dirichlet" -- the role
the task gives the subdomain 4C owns (the Dirichlet side receives the field
VALUES and imposes them; the Neumann side receives the outward FLUX and applies
it as its load). One file, both roles; nothing else changes between them but
what the deck imposes at the interface and what is exported as values.

Reads ./config.json {"level":k,"nx":..,"ny":..,"x0":..,"x1":..,"y0":..,"y1":..,
"k":diffusivity,"iface":"left|right|bottom|top","source_expr":"<f(x,y) or 0.0>",
"outer":<the value prescribed on this subdomain's NON-interface boundary, if
your problem prescribes one>,"fourc_bin":..,"fourc_ld":..}.
WRITE "outer" WHEN YOUR PROBLEM PRESCRIBES ONE. It is the same kind of
transcription as source_expr, and it buys the same kind of check: openPASO
compares your delivered field against the value on the edges you say are held.
The equation check CANNOT do this -- its test function is built so every
boundary term vanishes, which is what lets it judge a coupled side at all -- so
a field that solves the right equation with the wrong boundary condition passes
it cleanly, converges cleanly, and comes out as a refinement study that did not
converge. Measured: one coupled side left two outer edges natural instead of
held, its field on those edges was as large as anywhere in its interior while
the sides that held them sat exactly on the prescribed value, and nothing in
this tool could see it.
Contract: reads ./imports.json (the partner's interface samples at its points),
maps them onto THIS side's interface nodes -- as the Neumann load (side
"neumann", opposite normals) or as the imposed Dirichlet trace (side
"dirichlet") -- runs YOUR 4C deck, and exports its own consistent outward flux
plus, on the Neumann side only, its interface TRACE as values (the Dirichlet
side imposed the trace, it does not own one: values = []).

WHAT IS SERVED HERE is the handshake (config + imports + sign convention), the
CALCFLUX_BOUNDARY flux-recovery route, and the exports schema. THE 4C DECK AND
THE SOLVE ARE YOURS to write -- see the banner. Get the deck from
prepare_simulation(solver='fourc', physics='<your physics>').
"""
import json
import os
import sys
import numpy as np
from pathlib import Path

CFG = json.loads(Path("config.json").read_text())
# A multi-level coupling call hands this level's keys in the environment
# (OPENPASO_CONFIG_JSON, a JSON object) instead of writing this file.
CFG.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))
NX, NY = CFG["nx"], CFG["ny"]
X0, X1, Y0, Y1 = CFG["x0"], CFG["x1"], CFG["y0"], CFG["y1"]
KV = CFG["k"]; IF = CFG.get("iface", "left")
SIDE = CFG.get("side")   # "neumann" | "dirichlet": the role the task gives this subdomain
if SIDE not in ("neumann", "dirichlet"):     # a default here chose the role for the task
    sys.exit('config.json needs "side": "dirichlet" or "neumann" -- the role your task gives this subdomain')

# SOURCE f(x,y) AS A 4C EXPRESSION STRING (not a Python function): '^' for
# powers (never '**'), lowercase 'pi' ('PI' aborts: "Missing variables PI"),
# coordinates 'x','y', time 't'; "0.0" means no source. Wire it into YOUR deck
# as FUNCT1 + a DESIGN SURF NEUMANN VAL*FUNCT block; a Python src() that never
# reaches the deck is the classic 4C trap and does nothing.
SRC_EXPR = str(CFG.get("source_expr", CFG.get("source_const", "0.0")))
HAS_SRC = SRC_EXPR.strip() not in ("", "0", "0.", "0.0")

# ---- the partner's interface samples, mapped onto THIS side (handshake) ----
imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text())
def _partner(key, y_or_x):
    """One imported partner sample (key = "normal_fluxes" on the Neumann side,
    "values" on the Dirichlet side) interpolated onto one of THIS side's
    interface points. The driver does NOT interpolate between the two meshes --
    each participant maps the partner's samples onto its own points, here. Empty
    on iteration 1, so fall back to 0.0."""
    for _n, d in imp.items():
        co = d.get("coordinates") or []; q = d.get(key) or []
        if co and q and len(q) == len(co):
            ax = 1 if IF in ("left", "right") else 0
            pts = sorted(zip([c[ax] for c in co], q))
            xs = [p[0] for p in pts]; qs = [float(p[1]) for p in pts]
            t = min(max(y_or_x, xs[0]), xs[-1])
            for a, b, qa, qb in zip(xs, xs[1:], qs, qs[1:]):
                if a <= t <= b:
                    w = 0.0 if b == a else (t - a) / (b - a)
                    return qa + w * (qb - qa)
            return qs[-1]
    return 0.0
def partner_flux(y_or_x):
    """NEUMANN side: the partner's outward flux at one of your interface points --
    your inward load there. Build your Neumann loads from this in the solve."""
    return _partner("normal_fluxes", y_or_x)
def partner_value(y_or_x):
    """DIRICHLET side: the partner's field value at one of your interface points --
    the trace you impose there. Build your interface Dirichlet data from this."""
    return _partner("values", y_or_x)
# SIGN CONVENTION: the inward load on THIS side is the partner's OUTWARD flux --
# the two interface normals are opposite. Apply the partner's number UNCHANGED
# as your 4C Neumann VAL; there is no extra minus sign anywhere.

# ── DID 4C FINISH? (served: when the run leaves no output, name the cause) ─
# An empty out-vtk-files/ means 4C aborted on your deck. Its own message sits
# in the console log you captured, and the deck defects measured to abort
# every trial deck are mechanical -- so this reads the log and lints the deck
# and spells the cause out. It runs in TWO places: below the hole, before
# anything else is read, and AT EXIT if the script stops early without an
# exports.json (measured: workers that raised "4C FAILED -- see run.log"
# inside their own solve never reached the check below). Every part is
# guarded: a copy that lost an import still reports 4C's own words.
import atexit, glob, sys
def why_4c_did_not_finish():
    _why = []
    try:                                   # 1. 4C's own message, builtins only
        for _lg in sorted(glob.glob("*.log")) + sorted(glob.glob("*.txt")):
            try:
                with open(_lg, errors="ignore") as _fh:
                    _lines = _fh.read().splitlines()
            except OSError:
                continue
            for _i, _ln in enumerate(_lines):
                if "PROC 0 ERROR" in _ln:
                    _said = []
                    for l in _lines[_i + 1:_i + 14]:   # the message, then the offending input block
                        if l.startswith("---") or l.lstrip().startswith(("0#", "1#")) or "MPI_ABORT" in l:
                            break
                        if l.strip():
                            _said.append(l.strip())
                    _why.append(f"4C said ({_lg}): " + " | ".join(_said))
                    break
    except Exception as _e:                # noqa: BLE001
        _why.append(f"(log scan failed: {_e!r})")
    try:                                   # 2. the deck, linted against 4C's own grammar
        import os as _os, re as _re, subprocess as _sp
        _deck = globals().get("DECK") or next(iter(sorted(glob.glob("*.4C.yaml")) or sorted(glob.glob("*.yaml"))), None)
        if _deck and _os.path.isfile(_deck):
            with open(_deck, errors="ignore") as _fh:
                _txt = _fh.read()
            _secs = _re.findall(r"^([A-Z][A-Z0-9 _/.:-]*?):\\s*$", _txt, _re.M)
            _dup = sorted({s for s in _secs if _secs.count(s) > 1})
            if _dup:
                _why.append("section(s) written more than once, 4C reads each once: " + ", ".join(_dup))
            _bin = globals().get("FOURC_BIN") or CFG.get("fourc_bin") or _os.environ.get("FOURC_BIN")
            _valid = set()
            if _bin and _os.path.isfile(str(_bin)):
                _env = dict(_os.environ)
                _ld = CFG.get("fourc_ld") or _os.environ.get("FOURC_LD")
                if _ld:
                    _env["LD_LIBRARY_PATH"] = f"{_ld}:{_env.get('LD_LIBRARY_PATH', '')}"
                _dump = _sp.run([str(_bin), "-p"], capture_output=True, text=True, timeout=120, env=_env).stdout
                _valid = set(_re.findall(r"^    - name: (.+?)\\s*$", _dump, _re.M)) | set(
                    _re.findall(r"^  - ([A-Z][A-Z0-9 _/.:-]*?)\\s*$", _dump.split("legacy_string_sections:", 1)[-1], _re.M))
                _valid |= {"TITLE"}
            if len(_valid) > 100:
                _bad = [s for s in dict.fromkeys(_secs) if s not in _valid and not _re.fullmatch(r"FUNCT\\d+", s)]
                if _bad:
                    import difflib as _dl
                    _why.append("section name(s) the binary's own grammar (`4C -p`) does not know: " + "; ".join(
                        f"'{s}' (closest known: {', '.join(repr(c) for c in _dl.get_close_matches(s, sorted(_valid), n=5, cutoff=0.5))})"
                        for s in _bad))
            if _re.search(r'PROBLEMTYPE:\\s*"?Thermo"?\\s*$', _txt, _re.M):
                _why.append("PROBLEMTYPE Thermo: this contract's recovery reads Scalar_Transport output "
                            "(phi_1 and flux_boundary_phi_1 in out-vtk-files/), which a Thermo problem never "
                            "writes, and Thermo does not know SOLVERTYPE/CALCFLUX_BOUNDARY -- use PROBLEMTYPE "
                            "Scalar_Transport with a SCALAR TRANSPORT DYNAMIC section, MAT_scatra and TRANSP elements")
            if "Scalar_Transport" in _txt and "THERMAL DYNAMIC:" in _txt and "SCALAR TRANSPORT DYNAMIC:" not in _txt:
                _why.append("PROBLEMTYPE Scalar_Transport needs its dynamics under `SCALAR TRANSPORT DYNAMIC`, "
                            "not `THERMAL DYNAMIC` (that is the Thermo problem type)")
            if "CALCFLUX_BOUNDARY" in _txt and "FLUX CALC" not in _txt:
                _why.append("CALCFLUX_BOUNDARY is set but no `SCATRA FLUX CALC LINE CONDITIONS` (SURF in 3-D) entry "
                            "names the interface, and 4C refuses flux output without one")
            _blocks = _re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\\s*$)", _txt, flags=_re.M)
            _topo = set(_re.findall(r"\\b(DNODE|DLINE|DSURFACE|DVOL)\\s+(\\d+)", _txt))
            for _b in _blocks:
                _head = _b.split(":", 1)[0].strip()
                _kw = _re.search(r"\\b(POINT|LINE|SURF|VOL)\\b", _head) if _head.endswith("CONDITIONS") else None
                if not _kw:   # every condition family with a geometry word (SCATRA FLUX CALC LINE CONDITIONS too)
                    continue
                _entries = [e for e in _re.split(r"^\\s*-\\s", _b, flags=_re.M)[1:] if e.strip()]
                _noid = [e for e in _entries if not _re.search(r"\\bE:\\s*\\d+|NODE_SET_NAME", e)]
                if _noid:
                    _why.append(f"{len(_noid)} entr{'y' if len(_noid) == 1 else 'ies'} in {_head} without `E: <id>` (or NODE_SET_NAME)")
                _kind = {"POINT": "DNODE", "LINE": "DLINE", "SURF": "DSURFACE", "VOL": "DVOL"}[_kw.group(1)]
                _ids = _re.findall(r"\\bE:\\s*(\\d+)", _b)
                _missing = sorted({i for i in _ids if (_kind, i) not in _topo}, key=int)
                if _missing:
                    _why.append(f"{_head} names E id(s) {', '.join(_missing)} that no *-NODE TOPOLOGY section defines")
    except Exception as _e:                # noqa: BLE001
        _why.append(f"(deck lint failed: {_e!r})")
    if not _why:
        _why.append("no VTU under out-vtk-files/ and no 4C error line in any *.log here -- run the binary line-buffered "
                    "(stdbuf -oL -eL) with its console captured to a log next to the deck, then read that log from the top")
    return "4C DID NOT FINISH -- " + "; ".join(_why)
def _diagnose_at_exit():
    if not Path("exports.json").is_file() and not glob.glob("out-vtk-files/*.vtu"):
        print(why_4c_did_not_finish(), file=sys.stderr, flush=True)
atexit.register(_diagnose_at_exit)

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─────────────────────────────────────
# THE 4C DECK AND THE SOLVE ITSELF ARE YOURS AND ARE NOT SERVED HERE.
#
# Build the mesh, the material, the elements and the full 4C input deck for the
# problem you were given, and run the 4C binary on it. That is ordinary 4C
# input-deck work and openPASO has no business writing your deck; before you run it,
# Writing the deck to a .yaml/.yml/.dat file names every defect the grammar can see, in that
# same reply and before the binary runs. Get the deck
# grammar (also `4C -p`), a runnable Scalar_Transport skeleton and the measured
# gotchas from:
#
#     prepare_simulation(solver='fourc', physics='<your physics>')
#     knowledge(topic='coupling', solver='fourc')     # the 4C traps below
#
# For THIS coupling your deck must:
#   * be PROBLEMTYPE "Scalar_Transport" with its dynamics in a section named
#     exactly `SCALAR TRANSPORT DYNAMIC` (TIMEINTEGR "Stationary", SOLVERTYPE
#     "linear_full", CALCFLUX_BOUNDARY "diffusive"), a MAT_scatra material
#     with DIFFUSIVITY = KV, and TRANSP QUAD4/TRI3 elements in a TRANSPORT
#     ELEMENTS section (SOLID elements are rejected against MAT_scatra). NOT
#     `THERMAL DYNAMIC`: that is the Thermo problem type, it has no CALCFLUX
#     and writes no scatra VTU, and 4C aborts with "Could not match this
#     input" on the scatra keys (measured in two trial decks tonight);
#   * on the NEUMANN side (SIDE == "neumann"): apply the imported partner flux
#     as the interface load -- one DESIGN POINT NEUMANN per INTERIOR interface
#     node whose VAL is that node's NODAL LOAD: the flux density integrated
#     over the node's share of the interface line. With h the interface node
#     spacing and y the node's coordinate along the interface,
#         VAL = (h/6) * (partner_flux(y-h) + 4*partner_flux(y) + partner_flux(y+h))
#     (Simpson; imported node-by-node, no polynomial fit). The density itself
#     is NOT the VAL of a POINT condition: measured, a deck that put
#     partner_flux(y) in VAL recovered a flux 10x too small at h = 0.1 and one
#     that divided by h a flux 10x too large. No extra sign anywhere: the
#     opposite normals already give it;
#   * on the DIRICHLET side (SIDE == "dirichlet"): impose the imported partner
#     values as the interface trace -- one DESIGN POINT DIRICH per INTERIOR
#     interface node with VAL = partner_value(that node's coordinate) (the two
#     interface ENDPOINTS keep the OUTER Dirichlet value: they lie on the outer
#     boundary, and imposing the partner trace there caps the side at order
#     1), every point set in the ONE DNODE-NODE TOPOLOGY section, E ids
#     continuous across the Dirichlet and Neumann families;
#   * keep at least one OUTER Dirichlet edge (u given) or the subdomain is singular;
#   * if HAS_SRC, wire SRC_EXPR as FUNCT1 SYMBOLIC_FUNCTION_OF_SPACE_TIME plus a
#     DESIGN SURF NEUMANN VAL*FUNCT block so it enters the assembled RHS -- '^'
#     for powers, never '**';
#   * request the consistent boundary flux for the recovery below: set
#     CALCFLUX_BOUNDARY "diffusive" in SCALAR TRANSPORT DYNAMIC and add a
#     `SCATRA FLUX CALC LINE CONDITIONS` (SURF in 3-D) entry on the interface
#     line, so 4C writes flux_boundary_phi_1 into the VTU.
#
# THE DNODE-ID TRAP (measured): condition `E:` ids reference GLOBAL DNODE
# numbers across ALL condition families. A Dirichlet block restarting at `E: 1`
# silently rebinds the interface DNODEs and zeroes the field -- number the
# families continuously. And this build writes scatra VTU by default with NO
# `VTK` section; adding one is rejected as an invalid section. WRITE NO `IO:`
# SECTION AT ALL -- no VERBOSITY, no RUNTIME VTK OUTPUT, no PREFIX: measured
# in three trial decks tonight, every `IO:` block aborted the read with
# "Could not match this input"; the VTU files appear under out-vtk-files/
# without it.
#
# Run 4C with the binary at config `fourc_bin` (env FOURC_BIN; discover(
# query='list') prints it on THIS install) and its dependency libraries on
# config `fourc_ld` (FOURC_LD / LD_LIBRARY_PATH), line-buffered with its console
# captured to a log next to the deck (stdbuf -oL -eL <bin> <deck> out > run.log
# 2>&1). WHEN THE BINARY EXITS NON-ZERO, DO NOT EXIT OR RAISE YOURSELF: fall
# through -- the served check right below reads that log and the deck and stops
# with the cause spelled out (measured: a wrapper that raised "Solver execution
# failed, check the log" first hid 4C's own "Could not match this input: IO:
# VERBOSITY ..." from the agent). openPASO does not run the solver for you and ships
# no host-specific binary path.
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─────────────────────────────────────

# ── CONSISTENT OUTWARD FLUX + EXPORTS -- the served recovery route ─────────
# After YOUR deck has run, read 4C's OWN output. Never hand-parse the VTU XML,
# and never recompute the flux from phi_1 differences.
import glob
import meshio
# ── 4C's OWN CONSOLE, ECHOED (served): the coupling tool captures THIS script's stdout as the
#    level's run log, and a run log is credited to 4C only by 4C's own lines (its time-integration
#    output), never by the NDOF line alone (measured: run logs holding only the driver header and the
#    NDOF line carried no evidence that this code ran at all).
for _lg in sorted(glob.glob("*.log")):
    if _lg.startswith("participant_output"):                         # the coupling tool's own captures, never re-echoed
        continue
    try:
        _ltxt = open(_lg, errors="ignore").read()
    except OSError:
        continue
    if "── 4C console" in _ltxt:                                     # an earlier echo, not a deck console
        continue
    if "4C" in _ltxt[:4000] or "PROC 0" in _ltxt or "Finalised step" in _ltxt:
        print(f"── 4C console {_lg} ──")
        print(_ltxt if len(_ltxt) < 60000 else _ltxt[-60000:])
# ── YOUR DECK, CHECKED BEFORE ANYTHING IS READ (served): 4C drops a condition whose E id no
#    topology section defines and RUNS THE WRONG PROBLEM to 'finished normally' (measured on a
#    worker deck: every boundary condition and the source gone, rc 0). A run like that is refused here.
import re as _re
for _dk in sorted(glob.glob("*.4C.yaml")) or sorted(glob.glob("*.yaml")):
    _txt = open(_dk, errors="ignore").read()
    _topo = set(_re.findall(r"\\b(DNODE|DLINE|DSURFACE|DVOL)\\s+(\\d+)", _txt))
    _lost = []
    for _b in _re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\\s*$)", _txt, flags=_re.M):
        _head = _b.split(":", 1)[0].strip()
        _kw = _re.search(r"\\b(POINT|LINE|SURF|VOL)\\b", _head) if _head.endswith("CONDITIONS") else None
        if _kw:   # every condition family with a geometry word (SCATRA FLUX CALC LINE CONDITIONS too)
            _kind = {"POINT": "DNODE", "LINE": "DLINE", "SURF": "DSURFACE", "VOL": "DVOL"}[_kw.group(1)]
            _lost += [f"{_head} E {x}" for x in _re.findall(r"\\bE:\\s*(\\d+)", _b) if (_kind, x) not in _topo]
    if _lost:
        raise SystemExit(f"DECK CHECK: {_dk} puts conditions on E ids that no *-NODE TOPOLOGY section defines "
                         f"({'; '.join(_lost[:6])}): 4C dropped them silently, so the run solved a different "
                         f"problem. Add the DNODE/DLINE/DSURF/DVOL-NODE TOPOLOGY entries for those ids.")
    # twisted or clockwise 2-D elements (zero/negative area from the deck's own coordinates) solve nothing
    _cxy = {int(n): (float(x), float(y)) for n, x, y in _re.findall(r'"NODE\\s+(\\d+)\\s+COORD\\s+(\\S+)\\s+(\\S+)\\s+\\S+"', _txt)}
    _twist = []
    for _e, _k, _ids in _re.findall(r'"\\s*(\\d+)\\s+\\w+\\s+(QUAD4|TRI3)\\s+((?:\\d+\\s+)+)', _txt):
        _nn = [int(i) for i in _ids.split()][:4 if _k == "QUAD4" else 3]
        if len(_nn) >= 3 and all(i in _cxy for i in _nn):
            _p = [_cxy[i] for i in _nn]
            if 0.5 * sum(_p[q][0] * _p[(q + 1) % len(_p)][1] - _p[(q + 1) % len(_p)][0] * _p[q][1] for q in range(len(_p))) <= 1e-14:
                _twist.append((_e, _nn))
    if _twist:
        raise SystemExit(f"DECK CHECK: {_dk} has {len(_twist)} element(s) with zero or negative area (first: element {_twist[0][0]} "
                         f"nodes {' '.join(map(str, _twist[0][1]))}). Every element's nodes must run counter-clockwise: for node "
                         f"id = i + 1 + (NX + 1) * j the quad of cell (i, j) is (id, id + 1, id + NX + 2, id + NX + 1).")
# ── DID 4C FINISH? ── the check defined above the hole, run first here ─────
_vtus = sorted(glob.glob("out-vtk-files/*.vtu"))
if not _vtus:
    raise SystemExit(why_4c_did_not_finish())
# The VTU name is out-vtk-files/scatra-<step>-<rank>.vtu -- the TRAILING number
# is the MPI RANK, not the step. Sorting on it returns scatra-00000-0.vtu, the
# all-zero INITIAL CONDITION; take the last real step.
vtu = sorted(glob.glob("out-vtk-files/*.vtu"))[-1]
_m = meshio.read(vtu)
vpts = [(float(p[0]), float(p[1])) for p in _m.points]
phi = [float(x) for x in _m.point_data["phi_1"].ravel()]   # the scalar is 'phi_1'
# a 4C QUAD4 VTU repeats each node once per element; collapse by coordinate or
# n_points comes out four times too large and changes with the mesh. One dict, O(N),
# keyed on the coordinate in steps of a tolerance; a lookup also tries the steps
# around it, so a deck that printed NODE COORDS with fewer digits than `nodes` holds
# still finds its points (measured: keys rounded to 12 decimals raised KeyError).
_qt = 1e-6 * max(X1 - X0, Y1 - Y0)
def _key(x, y):
    return (round(x / _qt), round(y / _qt))
def _at(m, x, y):                          # the entry of m within one step of (x, y)
    i, j = _key(x, y)
    for _c in [(i + a, j + b) for a in (0, -1, 1) for b in (0, -1, 1)]:
        if _c in m:
            return m[_c]
    raise SystemExit(f"READ-BACK: {vtu} has no point at ({x:g}, {y:g}), a point of `nodes`: `nodes` and "
                     f"the deck's NODE COORDS are not the same mesh")
val = {}
for (x, y), _u in zip(vpts, phi):
    val[_key(x, y)] = _u
u = [_at(val, x, y) for (x, y) in nodes]
# ── HELD EDGES AND THE HELD TRACE (served) ─ keep. The solved field against config.json, before
#    anything is exported. Measured: one deck held its interface line at the constant outer value,
#    another left part of a held outer edge free; both finished normally and coupled on unnoticed.
#    An edge your deck holds (a DIRICH condition between its two ends) carries "outer" at each node
#    off the interface; on the Dirichlet side each interior interface node carries partner_value,
#    on the Neumann side none is held. Judged only when config.json states "iface" (and "outer").
_dk = globals().get("DECK") or max(glob.glob("*.4C.yaml") or glob.glob("*.yaml") or [""],
                                   key=lambda _q: os.path.getmtime(_q) if _q else 0)
_dt = open(_dk, errors="ignore").read() if _dk and os.path.isfile(str(_dk)) else ""
_tn, _held = {}, set()
for _n, _k, _e in _re.findall(r'"NODE\\s+(\\d+)\\s+(DNODE|DLINE)\\s+(\\d+)"', _dt):
    _tn.setdefault((_k, _e), set()).add(int(_n))
for _b in _re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\\s*$)", _dt, flags=_re.M):
    _g = _re.match(r"DESIGN (POINT|LINE) (?:TRANSPORT )?DIRICH CONDITIONS:", _b)
    for _en in _re.split(r"^\\s*-\\s", _b, flags=_re.M)[1:] if _g else []:
        _e = _re.search(r"\\bE:\\s*(\\d+)", _en)
        if _e and not _re.search(r"ONOFF:\\s*\\[\\s*0\\s*\\]", _en):
            _held |= _tn.get(("DNODE" if _g.group(1) == "POINT" else "DLINE", _e.group(1)), set())
_hxy = [(float(_x), float(_y)) for _n, _x, _y in _re.findall(r'"NODE\\s+(\\d+)\\s+COORD\\s+(\\S+)\\s+(\\S+)\\s+\\S+"', _dt)
        if int(_n) in _held]
_E = {"left": (0, X0), "right": (0, X1), "bottom": (1, Y0), "top": (1, Y1)}
_tl = _qt                                  # the read-back's step: the deck's own digits
def _on(p, e):                             # p on box edge e; _mid: strictly between its two ends
    return abs(p[_E[e][0]] - _E[e][1]) <= _tl
def _mid(p, e):
    _r = (Y0, Y1) if _E[e][0] == 0 else (X0, X1)
    return _on(p, e) and _r[0] + _tl < p[1 - _E[e][0]] < _r[1] - _tl
if "iface" in CFG and IF in _E:
    try:
        _o = float(CFG["outer"]) if "outer" in CFG else None
    except (TypeError, ValueError):
        _o = None
    _pk = max((abs(_v) for _v in u), default=0.0)
    for _ed in [e for e in _E if e != IF and _o is not None and any(_mid(p, e) for p in _hxy)]:
        _nn = sum(1 for p in nodes if _on(p, _ed))           # its nodes, both ends included
        _hn = sum(1 for p in _hxy if _on(p, _ed))
        _all = [k for k, p in enumerate(nodes) if _on(p, _ed) and not _on(p, IF)]   # judged against "outer"
        _bad = [k for k in _all if abs(u[k] - _o) > 1e-5 * max(abs(_o), _pk)]
        if _bad or _hn < _nn:
            raise SystemExit(
                f"OUTER BOUNDARY: your deck holds the {_ed} edge ({'xy'[_E[_ed][0]]} = {_E[_ed][1]:g}), config.json "
                f"states outer = {_o:g}, and its DIRICH conditions reach {_hn} of the {_nn} nodes of that edge, "
                f"both ends included" + (f"; the other {_nn - _hn} carr{'ies' if _nn - _hn == 1 else 'y'} none (natural, zero flux, in 4C)"
                                         if _hn < _nn else "")
                + (f"; {len(_bad)} of the {len(_all)} off the interface hold{'s' if len(_bad) == 1 else ''} another value in the solved field "
                   f"(the first at ({nodes[_bad[0]][0]:g}, {nodes[_bad[0]][1]:g}): {u[_bad[0]]:.6g})" if _bad else "")
                + ("; so the deck's VAL there or config.json's outer is not your problem's"
                   if _bad and _hn >= _nn else "")
                + ". Hold every node of that edge. Nothing was exported.")
    _iv = [(k, p) for k, p in enumerate(nodes) if _mid(p, IF)]
    if SIDE == "dirichlet" and _iv and any(isinstance(_d, dict) and _d.get("coordinates") and _d.get("values") and
                                           len(_d["values"]) == len(_d["coordinates"]) for _d in imp.values()):
        _tr = [(k, p, partner_value(p[1 - _E[IF][0]])) for k, p in _iv]
        _sc = max([abs(_t) for _, _, _t in _tr] + [abs(u[k]) for k, _ in _iv])
        _off = [(k, p, _t) for k, p, _t in _tr if abs(u[k] - _t) > 1e-3 * _sc]
        if _off:
            _k, _p, _t = _off[0]
            raise SystemExit(
                f"INTERFACE TRACE: on this Dirichlet side {len(_off)} of the {len(_iv)} interior nodes of the {IF} "
                f"edge do not hold the imported trace in the solved field (the first at ({_p[0]:g}, {_p[1]:g}): "
                f"{u[_k]:.6g}, partner_value {_t:.6g})"
                + (f"; they all hold one value, {u[_iv[0][0]]:.6g}" if len({round(u[k], 12) for k, _ in _iv}) == 1 else "")
                + ". Give each interior interface node its own DESIGN POINT DIRICH with VAL = partner_value(its "
                  "coordinate). Nothing was exported.")
    _hi = [p for p in _hxy if _mid(p, IF)]
    if SIDE == "neumann" and _hi:
        raise SystemExit(
            f"INTERFACE HELD: on this Neumann side your deck's DIRICH conditions hold {len(_hi)} of the {len(_iv)} "
            f"interior nodes of the {IF} edge (the first at ({_hi[0][0]:g}, {_hi[0][1]:g})), and a held node takes "
            f"no load. A Neumann side leaves them free and loads each with its own DESIGN POINT NEUMANN. Nothing "
            f"was exported.")
# CONSISTENT flux = 4C's CALCFLUX_BOUNDARY output (assembly-consistent by
# construction, from 4C's true residual -- the same reaction recovery every
# other backend in this corpus uses). flux_boundary_phi_1 is the flux VECTOR;
# dot it with THIS side's outward normal at the interior interface nodes.
# (CALCFLUX_DOMAIN -- the L2-projected -D grad(phi) -- is only order ~1 on the
# boundary trace and drags the measured order down; do not use it.)
fbname = next((da for da in _m.point_data if "flux_boundary" in da), None)
if fbname is None:
    raise SystemExit("no flux_boundary field in the VTU -- set CALCFLUX_BOUNDARY "
                     "'diffusive' and add a SCATRA FLUX CALC condition on the interface")
fb = _m.point_data[fbname]
nrm = {"left": (-1, 0), "right": (1, 0), "bottom": (0, -1), "top": (0, 1)}[IF]
fbmap = {}
for (x, y), vec in zip(vpts, fb):
    fbmap[_key(x, y)] = float(vec[0] * nrm[0] + vec[1] * nrm[1])
q_own = [_at(fbmap, *nodes[n-1]) for n in interior]
co = [list(nodes[n-1]) for n in interior]
vals = [u[n-1] for n in interior]
# exports.json LAST, only after the solve succeeded (the driver takes its
# existence as proof of success): YOUR interface trace as values, YOUR consistent
# outward flux as normal_fluxes.
# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a flux of ~0 against a nonzero partner); and a flux that is the partner's
#    array negated instead of a recovery from THIS side's own system.
# (json, numpy and Path are the imports at the top of this file)
_chk_vals = np.asarray(vals, float).ravel()
_chk_flux = np.asarray(q_own, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes; "
                     "the solve did not produce a usable field, so nothing was exported")
_chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
            if Path("imports.json").is_file() else {})
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                             for _d in _chk_imp.values()])
            if _chk_imp else np.zeros(0))
if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0 and (
        np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max()):
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0 against a "
                     "nonzero imported flux: the imported load never entered the "
                     "assembled system (the condition that integrates it is missing). "
                     "Fix the application; do not couple on")
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size and (
        np.array_equal(_chk_flux, -_chk_qin)):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's array "
                     "negated, bit for bit: a copy, not a recovery from this side's "
                     "own assembled system")
# NEUMANN LOAD CONSISTENCY (served). A Neumann side's recovered outward flux at
# the interface reproduces the load it applied, with the opposite sign: a
# validated deck gives max|q_own + q_imported| / max|q_imported| = 1.6% at
# h = 0.1. A load put in VAL as the density, or on the wrong nodes, runs and
# exports without a word and is 10x off; the coupling then stalls for the rest
# of the budget. Measured on a trial deck that ran clean and was 60% wrong.
if SIDE == "neumann" and q_own:
    _ax = 1 if IF in ("left", "right") else 0
    _q_applied = [partner_flux(float(nodes[n - 1][_ax])) for n in interior]
    _scale = max(abs(x) for x in _q_applied) if _q_applied else 0.0
    if _scale > 0:
        _per_node = [abs(a + b) / _scale for a, b in zip(q_own, _q_applied)]
        _mis = max(_per_node)
        _worst = _per_node.index(_mis)
        if _mis > 0.3:
            # IT REPORTS, IT DOES NOT EXIT. This check cannot run during the
            # standalone smoke test the served text prescribes -- that test
            # writes an imports.json of zeros, which makes _scale 0 and skips
            # it -- so it first speaks when the driver hands it real data, on
            # iteration 1, after the solve has already succeeded. Exiting there
            # threw the level away: measured on one run, three identical dead
            # couple() calls and about fourteen minutes, and the agent
            # eventually commented this block out and inverted the sign
            # convention while doing it. The defect it names is real and worth
            # saying loudly; it is not worth a finished solve.
            print(f"EXPORT SELF-CHECK WARNING: the recovered interface flux does not match the load you "
                  f"applied -- check the point-load formula and the node coordinates. Ratio "
                  f"max|q_own + q_imported| / max|q_imported| = {_mis:.2f}, worst at interior node "
                  f"{_worst} ({co[_worst] if _worst < len(co) else '?'}), while a correct Neumann side "
                  f"reproduces its load to a few percent. The imported flux may have entered the deck wrongly "
                  f"scaled or at the wrong nodes -- check the point-load formula "
                  f"VAL = (h/6)*(q(y-h) + 4q(y) + q(y+h)) and the node coordinates. NOTE the two END nodes of "
                  f"the interior list: their outer neighbour is the excluded corner and is never imported, so "
                  f"the three-point rule is undefined there -- use VAL = (h/2)*q at those two, and expect "
                  f"flux_boundary there to carry the corner reaction as well. The export CONTINUES so this "
                  f"level is not lost; the coupling's own residual history is the evidence that decides it.",
                  file=sys.stderr)
json.dump({"field_name": "u", "coordinates": co, "values": (vals if SIDE == "neumann" else []),
           "normal_fluxes": q_own, "n_points": len(co)},
          open("exports.json", "w"))
# PER-LEVEL PERSISTENCE. Each mesh level writes its OWN field file named by the
# config level, so levels 1->2->3 leave THREE files instead of the finest
# overwriting the coarse ones. Build the per-level field file for each side,
# named as your task prescribes, from THESE (interpolated to the task's probe
# points), NEVER from one output the next level overwrites -- that overwrite is
# the top cause of identical-across-levels runs whose three levels were
# byte-identical.
_LVL = CFG.get("level", "X")
with open(f"field_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,u\\n")
    for (_px, _py), _u in zip(nodes, u):
        _f.write(f"{_px:.11e},{_py:.11e},{float(_u):.11e}\\n")
# and its own interface trace and flux at THIS level's interface nodes:
# exports.json is overwritten by the next level, this file is not (measured:
# a run that rebuilt level 1's interface file from exports.json after level 2
# handed in two byte-identical levels).
with open(f"interface_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,u,qn\\n")
    for (_px, _py), _n, _q in zip(co, interior, q_own):
        _f.write(f"{_px:.11e},{_py:.11e},{float(u[_n - 1]):.11e},{float(_q):.11e}\\n")
# THE RUN-LOG CONTRACT LINE: `NDOF = <integer>` on a line of its own -- the
# audit reads that exact shape (measured: three otherwise complete coupled
# runs lost their evidence to logs whose only NDOF sat inside a prose
# line). The descriptive line follows it.
print(f"NDOF = {len(nodes)}")
print(f"4C {SIDE} participant: NDOF = {len(nodes)}  "
      f"max|u| = {max(abs(t) for t in vals) if vals else 0:.6e}")

# ── WHAT YOUR SOLVE MUST LEAVE BEHIND ──────────────────────────────────────
# The recovery and export above use these names; the elided deck-and-solve block
# has to define every one, or the rest will not run:
#
#     nodes      every mesh node as (x, y), indexed by 4C node id via
#                nodes[gid-1]; the recovery maps 4C's VTU points back onto it
#     interior   the interior interface node ids -- the points you export. DROP
#                the two endpoints: they also lie on the outer Dirichlet boundary
#                and are a physically different quantity there
#     DECK       (optional) the path of the deck you ran, and FOURC_BIN the
#                binary -- the finish check above reads them to name a deck
#                defect; without them it globs *.4C.yaml and config fourc_bin
#
# and YOUR deck must have produced out-vtk-files/*.vtu carrying phi_1 and
# flux_boundary_phi_1. openPASO does not serve the solve, but it will not make you
# guess which variables the hole was filling.
```

* USE THE BOUNDARY FLUX, NOT THE DOMAIN FLUX, ON THE DIRICHLET SIDE. Set
  `CALCFLUX_BOUNDARY: "diffusive"` in `SCALAR TRANSPORT DYNAMIC` and give the
  interface line a `SCATRA FLUX CALC LINE CONDITIONS` entry (`- E: <line>`;
  the SURF variant in 3D) -- that spelling is the one in the binary's own
  grammar (`4C -p`), verified by execution; an earlier revision of this file
  said `DESIGN SURF/LINE TRANSPORT FLUX CALCULATION`, which is not a valid
  section name. 4C then reports the CONSISTENT (Gresho) boundary flux,
  computed from its true residual exactly as the reaction recovery is derived
  for every other backend in this corpus.

  `CALCFLUX_DOMAIN: "diffusive"` is a DIFFERENT quantity: the L2 projection of
  `-D grad(phi)` over the whole subdomain, sampled at the interface. The
  gradient of a linear solution is only O(h) accurate ON the boundary — the
  superconvergence points are interior — and the boundary trace is exactly
  what the coupling reads. Measured on a manufactured solution: the projected
  flux converges at order ~1.1 and drags the measured field order to ~1.75
  against a band that ends at 1.6, while the consistent flux gives ~2.05. This
  file used to recommend the domain flux; it was setting the answer.

  Computing the flux yourself from `phi_1` differences is worse still.
* A 4C VTU repeats every node once per element (QUAD4 -> 4 copies of each
  node). Collapse duplicates by coordinate before exporting, or `n_points` is
  four times too large and changes with the mesh.
* `DLINE-NODE TOPOLOGY` must list the nodes of every design line you reference.
  A `DESIGN LINE ... CONDITIONS` entry with `E: 2` and no DLINE 2 topology is
  accepted and does nothing.
* The 4C binary needs its dependency libraries on `LD_LIBRARY_PATH`; put that
  directory in `FOURC_LD` if the binary does not find them by itself.''')


def _ngsolve() -> str:
    return _payload(
        "NGSolve",
        "**Either side, in either subdomain.** All four role/position "
        "combinations were run as real couplings on this install — against "
        "FEniCSx and against scikit-fem, with non-matching interface meshes — "
        "and all converged.",
        "ngsolve", _launch_py(),
        '''\
* TWO CONSECUTIVE `gfu.Set(value, definedon=mesh.Boundaries(...))` CALLS CANCEL
  EACH OTHER. The second `Set` zeroes what the first wrote outside its own
  region, so setting the outer Dirichlet value and then the interface value
  loses one of them — and the run finishes with a plausible, WRONG answer, not
  an error. Put every Dirichlet value into ONE `mesh.BoundaryCF({...})`.
* `Integrate(grad(gfu)[0], mesh, definedon=mesh.Boundaries("..."))` RETURNS
  EXACTLY 0.0. An H1 GridFunction's gradient has no boundary trace. Use the
  reaction/residual method — `a.mat * gfu.vec - f.vec` restricted to the
  interface DOFs — which is also the more accurate flux.
* `SplineGeometry.AddRectangle(..., bcs=(...))` names the edges in the order
  bottom, right, top, left. Getting that order wrong silently puts the
  interface condition on the wrong edge.
* Netgen meshing is DETERMINISTIC across separate processes for the same
  geometry and `maxh`, which is what makes "the same number of interface
  points every iteration" hold — each coupling iteration is a fresh process
  that re-meshes from scratch. Do not introduce anything mesh-size-dependent
  that varies with the imported data.
* At `order=1` the H1 DOFs coincide with the mesh vertices, so
  `NodeId(VERTEX, i)` gives you a stable interface indexing. At higher order
  you must locate the interface DOFs properly instead.
* Solve with `a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")` applied
  to the residual after setting the Dirichlet data — the direct
  `gfu.vec.data = a.mat.Inverse() * f.vec` form ignores your BCs.''')


def _skfem() -> str:
    return _payload(
        "scikit-fem",
        "**Either side, in either subdomain.** All four role/position "
        "combinations were run as real couplings on this install — against "
        "FEniCSx and against NGSolve, with non-matching interface meshes — and "
        "all converged.",
        "skfem", _launch_py(),
        '''\
* This is the lightest participant of the set: pure Python, numpy + scipy, no
  compilation and no JIT. If you are prototyping a coupling and do not care
  which code solves a side, start here.
* `MeshTri.init_tensor(x, y)` builds the structured subdomain mesh directly
  from your NX/NY, so the interface node set is exactly predictable.
* `facets_satisfying(...)` defaults to `boundaries_only=False` in current
  scikit-fem, so it can return INTERIOR facets that happen to satisfy your
  predicate. That is harmless when the interface is a domain edge (as here) and
  wrong the moment it is not — pass `boundaries_only=True` when in doubt.
* Applying the imported value: put it into the solution vector at the interface
  DOFs and pass those DOFs as `D` to `condense`. Applying an imported flux: a
  `LinearForm` assembled on a `FacetBasis` restricted to the interface facets.
* Getting the outgoing flux: the residual `K @ x - f` at the CONSTRAINED
  interface DOFs is the consistent nodal flux and needs no differentiation.
  Divide by the tributary length if you export a density, and keep whichever
  convention you chose consistent with the partner.
* scikit-fem gives you the assembled matrices, so it is the easiest backend in
  which to check a coupling by hand: assemble the two subdomains and the
  monolithic problem and compare. That is the strongest verification available
  for a partitioned coupling and it costs a few lines here.''')



def _fsi() -> str:
    """knowledge(topic='coupling', solver='fsi') — the FSI pattern.

    Not a backend. It is served through the same dispatch because an agent with
    an FSI problem asks for FSI, not for "the FEniCSx participant plus the
    scikit-fem participant plus the three things nobody wrote down".
    """
    return (
        "# Partitioned fluid-structure interaction with `couple`\n\n"
        "## What FSI needs that a scalar coupling does not\n\n"
        "Everything in `knowledge(topic='coupling')` still applies — the "
        "participant contract, the JSON shapes, the relaxation. FSI adds four "
        "things, and the two scripts below are one worked pair that has all "
        "four.\n\n"
        "1. **The exchange is VECTOR-valued.** `values` is an (N, 2) or (N, 3) "
        "list: a traction one way, a displacement the other. The driver handles "
        "that already — it relaxes the flattened vector and reports a residual "
        "PER COMPONENT in `block_residuals`, which matters here because "
        "traction and displacement differ by many orders of magnitude and a "
        "single global norm would let the small one move freely.\n"
        "2. **The two participants exchange DIFFERENT quantities.** A "
        "Dirichlet-Neumann heat coupling passes temperature one way and flux "
        "the other; FSI passes traction one way and displacement the other, and "
        "the fluid consumes the displacement by MOVING ITS MESH (ALE), not by "
        "setting a boundary value. That mesh motion is the entire "
        "structure-to-fluid direction. If it is missing you have a rigid-wall "
        "flow solve dressed as FSI, and every force-side check will pass.\n"
        "3. **The traction sign has to be decided once, in writing.** See the "
        "section below; getting it wrong in both places is silent.\n"
        "4. **The iteration can be unstable for physical reasons.** Added mass: "
        "with an incompressible fluid and a light structure the unrelaxed "
        "iteration diverges, and a smaller time step makes it worse. "
        "`knowledge(topic='pitfalls', solver='coupling', physics='fsi')` has "
        "the criterion and what to do.\n\n"
        "## THE SIGN, ONCE\n\n"
        "```\n"
        "  fluid exports  values = t = sigma_f . n_s = -sigma_f . n_f\n"
        "                        = THE LOAD ON THE STRUCTURE\n"
        "  structure applies it DIRECTLY as a Neumann traction, no sign change\n"
        "```\n"
        "`n_f` is the fluid's outward normal on the interface and `n_s = -n_f` "
        "the structure's. Cauchy's `t(n) = sigma.n` is the traction exerted BY "
        "the material `n` points INTO, so `sigma_f . n_f` is what the structure "
        "does to the fluid and the load on the structure is its negative. Check "
        "against hydrostatics before you trust an implementation: `sigma_f = "
        "-p I` with `p > 0` gives `t = +p n_f`, the fluid pushing the wall away "
        "from itself.\n\n"
        "Both participants ALSO export `normal_fluxes`, each with respect to "
        "ITS OWN outward normal, so the driver's conservation check can run and "
        "the two must sum to zero. Be clear about what that check then buys: it "
        "measures the FORCE TRANSFER across two non-matching interface "
        "discretisations. It cannot see a convention flipped on both sides.\n\n"
        "## Interface parametrisation\n\n"
        "Both sides send `coordinates` in the UNDEFORMED positions of their own "
        "interface nodes, and the displacement carries the motion. The "
        "interface is a material surface, so that parametrisation is fixed; "
        "sending deformed coordinates makes each side interpolate against "
        "something that moves with the answer.\n\n"
        "## THE FLUID PARTICIPANT (FEniCSx) — contract, solve elided; edit the marked block and write the solve\n\n"
        "write_participant_contract(solver='fenics', variant='fsi_fluid', path=...) writes it.\n\n"
        f"```python\n{_script('fsi_fluid_fenics')}```\n\n"
        "## THE STRUCTURE PARTICIPANT (scikit-fem) — contract, solve elided; edit the marked block and write the solve\n\n"
        "write_participant_contract(solver='skfem', variant='fsi', path=...) writes it.\n\n"
        f"```python\n{_script('fsi_solid_skfem')}```\n\n"
        "TWO MORE STRUCTURE PARTICIPANTS ship with the same contract, and both "
        "have been run as real coupled FSI on this install: "
        "`participant_fsi_solid_fenics.py` (FEniCSx, same interpreter as the "
        "fluid; write_participant_contract(solver='fenics', variant='fsi_solid', ...)) "
        "and `participant_fsi_solid_fourc.py` (write_participant_contract(solver='fourc', "
        "variant='fsi', ...); 4C — a plain-Python "
        "WRAPPER that writes an inline-mesh 4C deck, runs the binary and reads "
        "the VTU back, so it runs under an ordinary python with numpy and "
        "meshio, not under 4C). Running the pair once with each is how you find "
        "out whether an answer depends on one structure code. Two things worth "
        "knowing before you use the 4C one: its WALL QUAD4 is bilinear and "
        "shear-locks in bending, so a plate a few elements thick comes out too "
        "stiff and the mesh has to be refined until that bias is below whatever "
        "you are asserting; and a spatially varying surface traction goes in as "
        "VAL x FUNCT with FUNCT a symbolic expression, where a POLYNOMIAL FIT "
        "is a silently wrong boundary condition rather than an error — the "
        "shipped participant emits the piecewise-linear interpolant exactly, "
        "using `heaviside`, and reports the deviation every iteration.\n\n"
        "## Wiring them\n\n"
        "```python\ncouple(participants=json.dumps([\n"
        '  {"name":"fluid","command":["<fenicsx python>","participant_fluid.py"],\n'
        '   "work_dir":"/abs/fluid","imports_from":["solid"]},\n'
        '  {"name":"solid","command":["<python>","participant_solid.py"],\n'
        '   "work_dir":"/abs/solid","imports_from":["fluid"]}]),\n'
        "  max_iter=60, tol=1e-9, accelerator='aitken', theta=0.5,\n"
        "  monolithic=json.dumps({...}), probe=True)\n```\n\n"
        "`probe=True` is not optional for FSI. The interface-sensitivity probe "
        "is the ONLY check that separates a real two-way coupling from a fluid "
        "that never moves its mesh.\n\n"
        "## Checking it, in the order the checks are worth anything\n\n"
        "1. **Run each participant by hand first**, with no `imports.json`, and "
        "check the fluid's net interface force against a configuration you can "
        "compute. For a straight channel with the wall held rigid that is plane "
        "Poiseuille: pressure drop `12*mu*U_mean*L/H^2`, so the net normal "
        "force on the wall is `dp*L/2` and the net tangential force is "
        "`6*mu*U_mean/H*L`. This is the check that catches a wrong sign or a "
        "wrong unit, and it is the only one here that does not share code with "
        "the coupling.\n"
        "2. **Both directions, by suppression.** Re-run with the fluid's mesh "
        "motion switched off. If the converged deflection does not change, the "
        "coupling is one-way, whatever it is called. Report the size of the "
        "change; it is the only quantitative evidence that the reverse "
        "direction carries physics.\n"
        "3. **Interface equilibrium and kinematic continuity, COMPONENTWISE.** "
        "Net force out of the fluid against net force into the structure, and "
        "the displacement the fluid imposed against the structure's own. A "
        "coupling that drops the tangential component passes every check that "
        "only looks at totals.\n"
        "4. **A reference solve.** `fsi_reference_newtonkrylov.py` in the same "
        "directory re-solves the coupled interface equation by Newton-Krylov "
        "and writes `monolithic.json`, so it plugs straight into "
        "`couple(monolithic=...)`. It establishes that the ITERATION found the "
        "root — read its docstring for what it cannot establish, which is "
        "anything wrong INSIDE a participant, because it drives the same "
        "scripts.\n"
        "5. **An independent code.** For FSI that means a native monolithic FSI "
        "solver on the same geometry. 4C has one "
        "(`PROBLEMTYPE: Fluid_Structure_Interaction` with a monolithic "
        "`COUPALGO`). Nothing above substitutes for it.\n\n"
        "## Traps\n\n"
        "`knowledge(topic='pitfalls', solver='coupling', physics='fsi')` — the "
        "traction sign, added mass, the one-way FSI that passes everything, the "
        "whole-convention flip that even a reference re-solve agrees with, the "
        "inverted ALE mesh that hangs instead of failing, and why the traction "
        "recovered from the structure's own stress field is not an equilibrium "
        "check.\n")


# ══════════════════════════════════════════════════════════════════════════
# DISPATCH
# ══════════════════════════════════════════════════════════════════════════

_BACKENDS = {
    "fenics": _fenics, "fenicsx": _fenics, "dolfinx": _fenics,
    "fourc": _fourc, "4c": _fourc,
    "fsi": _fsi, "fluid-structure": _fsi, "fluid_structure": _fsi,
    "ngsolve": _ngsolve,
    "skfem": _skfem, "scikit-fem": _skfem, "scikitfem": _skfem,
}

_ALIAS_CANON = {"fenics": "fenics", "fenicsx": "fenics", "dolfinx": "fenics",
                "fsi": "fsi", "fluid-structure": "fsi",
                "fluid_structure": "fsi",
                "fourc": "fourc", "4c": "fourc", "ngsolve": "ngsolve",
                "skfem": "skfem", "scikit-fem": "skfem", "scikitfem": "skfem",
                "dune": "dune", "dune-fem": "dune", "dunefem": "dune",
                "dealii": "dealii", "deal.ii": "dealii",
                "febio": "febio", "kratos": "kratos", "sparta": "sparta"}


def _participant_chunks(text: str, limit: int = 9000) -> list[str]:
  """Split source at line boundaries while preserving it exactly."""
  chunks: list[str] = []
  lines: list[str] = []
  size = 0
  for line in text.splitlines(keepends=True):
    if lines and size + len(line) > limit:
      chunks.append("".join(lines))
      lines, size = [], 0
    lines.append(line)
    size += len(line)
  if lines:
    chunks.append("".join(lines))
  return chunks


_PARTICIPANT_LABELS = {
  "thermoelastic": "thermo-elastic",     # before "elastic": it contains that word
  "neumann": "Neumann-side",
  "elastic": "vector elasticity",
  "transient": "transient",
  "3d": "3-D",
  # THE FLUID-STRUCTURE CONTRACTS ARE NAMED THE OTHER WAY ROUND, participant_fsi_<role>_<code>.py,
  # and this resolver could not see them. Measured on a fluid-structure round: every orchestrator
  # passed physics='fsi', four of five wrote variant='fsi', and the writer put the SCALAR heat
  # contract on disk in all five cells, its reply calling it "the base contract" (5-8 minutes a
  # cell; one fluid side was built on it, and a restore step overwrote a cell's fluid file with it).
  "fsi_fluid": "fluid-structure FLUID-side",
  "fsi_solid": "fluid-structure STRUCTURE-side",
}
# Words a request may carry that choose no variant: the parts door's own words, words that name no
# physics, and the Dirichlet role, which no code ships a file of its own for (the base contract is
# that side, or both).
_REQUEST_FILLER = __import__("re").compile(
  r"participants?|contract|script|program|part\d*|base|default|dirichlet|side|file|code|whole|full"
  r"|the|for|of|a|my|this|")


def _request_word(request: str) -> tuple:
  """(the variant word a request names, '' for the base; the words in it that name nothing).

  A request is a variant word (the writer's `variant`) or a parts signal, 'participant[:<word>]
  :part<k>', with the knowledge door's physics word appended. A word that holds a label
  ('elasticity' holds 'elastic') names that label, the first in _PARTICIPANT_LABELS order; any
  spelling of fluid-structure names 'fsi', its role 'fsi_fluid' or 'fsi_solid' when it says one.
  A word that names none of them is returned as unknown -- never read as the base contract."""
  found, unknown = [], []
  # the brackets of the notation 'participant[:<variant>]:part<k>', which the writer's description
  # shows, are not part of a word
  for tok in (request or "").strip().lower().replace("_", "-").replace("[", "").replace("]", "").split(":"):
    tok = " ".join(tok.split())
    fsi = tok.replace("fluid-structure", "fsi").replace("fluid structure", "fsi")
    if "fsi" in fsi.replace("-", " ").split() or fsi.startswith("fsi"):
      rest = fsi.replace("fsi", "", 1)
      found.append("fsi_fluid" if "fluid" in rest else
                   "fsi_solid" if ("solid" in rest or "structur" in rest) else "fsi")
      continue
    hit = next((c for c in _PARTICIPANT_LABELS if not c.startswith("fsi") and c in tok), None)
    if hit is not None:
      found.append(hit)
    elif not all(_REQUEST_FILLER.fullmatch(w) for w in tok.replace("-", " ").split()):
      unknown.append(tok)
  order = ["fsi_fluid", "fsi_solid", "fsi"] + [c for c in _PARTICIPANT_LABELS if not c.startswith("fsi")]
  word = min(found, key=order.index) if found else ""
  return word, unknown


def _fsi_paths(key: str) -> dict:
  """{'fsi_fluid' | 'fsi_solid': path} of the fluid-structure contracts `key` ships."""
  return {f"fsi_{role}": p for role in ("fluid", "solid")
          for p in [_PARTICIPANT_DIR / f"participant_fsi_{role}_{key}.py"] if p.is_file()}


def _variant_listing(key: str) -> str:
  """The variant words `key` has, as the refusals name them."""
  words = [w for w in _PARTICIPANT_LABELS if _variant_path(key, w) is not None]
  return ("'' (the base contract: single-field, steady, 2-D)"
          + "".join(f", '{w}'" for w in words)
          + (" -- 'fsi' names the fluid-structure one" if len(_fsi_paths(key)) == 1 else ""))


def _variant_path(key: str, word: str):
  """The contract file of `key` a resolved variant word names, or None."""
  if word.startswith("fsi_"):
    return _fsi_paths(key).get(word)
  path = _PARTICIPANT_DIR / f"participant_{key}{'_' + word if word else ''}.py"
  return path if path.is_file() else None


def _resolved_word(key: str, request: str) -> tuple:
  """(the label key the request resolves to for `key`, '' for the base; refusal text or '')."""
  word, unknown = _request_word(request)
  if unknown:
    return "", (f"# No contract for variant {unknown[0]!r} of solver={key!r}\n\n"
                f"{unknown[0]!r} names none of this install's {key} contracts, so nothing was "
                f"served or written. The variant words it has: {_variant_listing(key)}.")
  if word == "fsi":
    fsi = _fsi_paths(key)
    if len(fsi) == 1:
      return next(iter(fsi)), ""
    if len(fsi) == 2:
      return "", (f"# {key} ships TWO fluid-structure contracts: name the side\n\n"
                  f"variant='fsi_fluid' is the fluid side ({fsi['fsi_fluid'].name}) and "
                  f"variant='fsi_solid' the structure side ({fsi['fsi_solid'].name}); take the one "
                  f"your task gives this code. Nothing was served or written for 'fsi'.")
    return "", (f"# No fluid-structure participant for solver={key!r}\n\n"
                f"The fluid-structure contracts ship for "
                + ", ".join(sorted({p.stem.rsplit("_", 1)[1]
                                    for p in _PARTICIPANT_DIR.glob("participant_fsi_*_*.py")}))
                + f". The variant words {key} has: {_variant_listing(key)}.")
  return word, ""


def resolve_participant(solver: str, request: str = "") -> tuple:
  """(path, label, error) for the participant contract a request names.

  One resolver for every door that hands out a participant, so the file the
  knowledge reply serves in parts and the file the writer puts on disk can
  never be two different files. `error` is the served refusal text when there
  is no such contract; then `path` is None. A request whose word names no
  contract is refused with the words that do: it is never served the base."""
  key = _ALIAS_CANON.get((solver or "").strip().lower())
  if not key or key not in _BACKEND_ORDER:
    return None, "", (f"# No coupling participant for solver={solver!r}\n\n"
                      f"Choose one of: {', '.join(_BACKEND_ORDER)}.")
  word, refusal = _resolved_word(key, request)
  if refusal:
    return None, "", refusal
  path = _variant_path(key, word)
  if path is None:
    # THE BASE CONTRACT TAKES BOTH ROLES where it carries a SIDE switch: a request for
    # its Neumann side is answered with it, said, rather than with a bare "none".
    both = word == "neumann" and _base_takes_both_sides(key)
    return None, "", (f"# No {_PARTICIPANT_LABELS.get(word, word or 'base')} "
                      f"participant for solver={solver!r}\n\n"
                      + (f"The base contract (variant='') serves BOTH sides: set SIDE = \"neumann\" "
                         f"in it, or \"side\": \"neumann\" in config.json where it reads config. "
                         if both else "")
                      + f"The variant words {key} has: {_variant_listing(key)}.")
  return path, _PARTICIPANT_LABELS.get(word, "base"), ""


def _base_takes_both_sides(key: str) -> bool:
  """True when the base contract of `key` carries a SIDE switch that names the Neumann side."""
  import re
  base = _PARTICIPANT_DIR / f"participant_{key}.py"
  return bool(base.is_file() and re.search(r'^SIDE\s*=.*"neumann"', base.read_text(), re.M))


def participant_variants() -> dict:
  """{solver: the variant words resolve_participant finds a contract for on this install}, the
  base contract ('') first. Read through the resolver, so the writer's docstring cannot list a
  variant the resolver refuses (measured: 'neumann' was listed for every solver, three FEniCSx
  workers asked for it first, and FEniCSx has none)."""
  return {key: [w for w in [""] + list(_PARTICIPANT_LABELS) if resolve_participant(key, w)[0] is not None]
          for key in _BACKEND_ORDER}


def participant_variants_text() -> str:
  """What each solver's contract writer accepts as `variant`, in plain words, from
  participant_variants(), and which base contracts serve both sides."""
  table = participant_variants()
  words = "; ".join(f"{key} " + (", ".join(f"'{w}'" for w in ws if w) or "none")
                    for key, ws in table.items())
  both = [key for key, ws in table.items() if "neumann" not in ws and _base_takes_both_sides(key)]
  own = [key for key, ws in table.items() if "neumann" in ws]
  two = [key for key, ws in table.items() if "fsi_fluid" in ws and "fsi_solid" in ws]
  return (f"`variant` is '' for the base contract, or one of the words this install's contracts "
          f"carry: {words}. The base contract serves both the Dirichlet and the Neumann side for "
          f"{', '.join(both)} (SIDE in its edit block, or \"side\" in config.json where it reads "
          f"config)" + (f"; {', '.join(own)} {'has' if len(own) == 1 else 'have'} a 'neumann' variant "
                        f"for that side" if own else "")
          + f". 'fsi' writes a code's one fluid-structure contract"
          + (f" ({', '.join(two)} {'ships' if len(two) == 1 else 'ship'} two, the fluid and the "
             f"structure side: name 'fsi_fluid' or 'fsi_solid')" if two else "")
          + "; a word that names no contract of the code is refused with the words it has.")


def _participant_key_suffix(solver: str, request: str = "") -> tuple:
  """(canonical solver key, file suffix) the resolver used; ('', '') when unknown."""
  key = _ALIAS_CANON.get((solver or "").strip().lower()) or ""
  if not key:
    return "", ""
  word, _refusal = _resolved_word(key, request)
  return key, (f"_{word}" if word else "")


def _door_scaffold(key: str) -> str:
  """The config-driven participant the door shows for 4C or DUNE-fem (base role),
  as code; '' if the door shows none."""
  door = {"fourc": _fourc, "dune": _dune}.get(key)
  if door is None:
    return ""
  payload = door()
  i = payload.find("## PARTICIPANT CONTRACT")
  j = payload.find("```python", i if i >= 0 else 0)
  k = payload.find("```", j + 9) if j >= 0 else -1
  if j < 0 or k < 0:
    return ""
  block = payload[j + len("```python"):k].lstrip("\n")
  return block if ("config.json" in block and "exports.json" in block) else ""


def participant_companions(solver: str, request: str = "") -> tuple:
  """([(file name, text), ...], error): the files served BESIDE a participant
  contract -- for the deal.II wrapper, its program scaffold (holes elided) and
  the six-line CMakeLists.txt that builds it. The knowledge door shows these
  texts and write_participant_contract writes them next to the wrapper; both
  come from here, so they cannot differ."""
  path, _label, err = resolve_participant(solver, request)
  if path is None:
    return [], ""
  name = _PROGRAM_OF.get(path.name)
  if name is None:
    return [], ""
  text, err = served_program(name)
  if err:
    return [], f"[openPASO WITHHOLDS {name}] {err}; refusing rather than serving it."
  return [(name, text), ("CMakeLists.txt", _dealii_cmake(name))], ""


def participant_contract_text(solver: str, request: str = "") -> tuple:
  """(text, error): the WHOLE served contract for one participant, solve
  elided, exactly the concatenation of the parts the knowledge door serves --
  it is produced by the same `_serve_participant` call, which fails closed.

  4C AND DUNE-fem SHOW A CONFIG-DRIVEN SCAFFOLD, AND THIS IS IT. Their doors
  lead with a scaffold that reads the level's mesh from config and writes the
  per-level dumps, and said the writer puts "this same text, byte for byte" on
  disk -- while the writer put the older file contract there, which hard-codes
  the mesh and dumps nothing (measured: all five cells of a coupled round got
  it, and a ladder ran three levels on one mesh)."""
  path, _label, err = resolve_participant(solver, request)
  if path is None:
    return "", err
  key, suffix = _participant_key_suffix(solver, request)
  if not suffix:
    scaffold = _door_scaffold(key)
    if scaffold:
      return scaffold, ""
  return _serve_participant(path), ""


def coupling_participant(solver: str, request: str = "") -> str:
  """Return one bounded chunk of a participant CONTRACT (solve elided).

  Every door that hands out a participant passes through `_serve_participant`,
  this one included: the chunks are cut from the ELIDED text, so no part and
  no concatenation of parts ever contains a marked SOLVE region."""
  import hashlib
  import re

  path, label, err = resolve_participant(solver, request)
  if path is None:
    return err
  key, suffix = _participant_key_suffix(solver, request)
  requested = (request or "").strip().lower().replace("_", "-")
  match = re.search(r"(?:^|:)part(\d+)(?:$|:)", requested)
  part = int(match.group(1)) if match else 1
  companions, cerr = participant_companions(solver, request)
  # A VARIANT'S PROGRAM PARTS NAME THEIR VARIANT: the NEXT signal of the transient
  # program named 'participant:program:part2', which is the steady program's part 2.
  prog_role = f"{suffix.lstrip('_')}:" if suffix else ""
  if "program" in requested.split(":") and (companions or cerr):
    return cerr or _program_part(key, companions, part, prog_role)
  source, _ = participant_contract_text(solver, request)   # the writer's text, part by part
  chunks = _participant_chunks(source)
  if not 1 <= part <= len(chunks):
    return (f"# Invalid participant part {part}\n\n"
            f"{path.name} has {len(chunks)} parts; request part1 through "
            f"part{len(chunks)}.")
  next_call = ""
  if part < len(chunks):
    role = f":{suffix.lstrip('_')}" if suffix else ""
    next_call = (
      "\nNEXT: request "
      f"`signal='participant{role}:part{part + 1}'`. ")
  elif companions:
    digest = hashlib.sha256(source.encode()).hexdigest()
    next_call = (f"\nFINAL PART. SHA-256 of the served wrapper text: `{digest}`. The wrapper has "
                 f"no hole: its solve is the program {companions[0][0]} beside it, whose holes "
                 f"are yours. write_participant_contract writes both files and its "
                 f"CMakeLists.txt; the program is also `signal='participant:{prog_role}program:part1'`. ")
  else:
    digest = hashlib.sha256(source.encode()).hexdigest()
    next_call = (f"\nFINAL PART. SHA-256 of the served contract text: `{digest}`. "
                 "It is the contract with the solve elided, not a complete "
                 "program: write the solve where the elision banner sits. ")
  # THE HEADER NAMES THE ONE CALL THAT WRITES THE FILE, AND WHERE THE FACTS ARE. Measured: two
  # workers whose briefs dropped the writer fetched their contracts as parts and retyped them
  # (74k characters, about seven minutes), then hit API errors the served facts for their codes
  # name -- facts neither fetched. The header said only "Concatenate only the fenced code".
  variant = f"variant='{suffix.lstrip('_')}', " if suffix else ""
  # A FLUID-STRUCTURE SIDE'S FACTS COME WITH THE FLUID-STRUCTURE DOOR: the plain door serves the
  # code's single-field facts, and a structure or fluid side needs its own.
  facts_door = (f"knowledge(topic='coupling', solver='{key}', physics='fsi')" if suffix.startswith("_fsi")
                else f"knowledge(topic='coupling', solver='{key}')")
  return (
    f"# {key} {label} participant CONTRACT (solve elided): part {part} of {len(chunks)}\n\n"
    f"ONE CALL WRITES THIS WHOLE FILE, byte for byte: write_participant_contract(solver='{key}', "
    f"{variant}path='<absolute path of the .py file in your working directory>'); use it rather "
    f"than retyping the parts. The API facts measured for {key} on this install -- the calls that "
    f"work and what a wrong one prints -- come with {facts_door}; "
    "read them before you fill a hole. Assembling the parts instead: concatenate only the fenced "
    "code contents in part order, without the headings. Then edit `EDIT THIS BLOCK` and "
    + (f"fill the holes of the program {companions[0][0]} beside it.\n\n" if companions else
       "write the solve where the elision banner sits.\n\n")
    + f"```python\n{chunks[part - 1]}```\n{next_call}\n")


def _program_part(key: str, companions: list, part: int, role: str = "") -> str:
  """One bounded chunk of a wrapper's program scaffold (holes elided); the last
  part carries the CMakeLists.txt that builds it. `role` is the variant's word and a
  colon ('transient:', 'elastic:'), '' for the base contract."""
  import hashlib
  (name, text), extra = companions[0], companions[1:]
  chunks = _participant_chunks(text)
  if not 1 <= part <= len(chunks):
    return (f"# Invalid program part {part}\n\n{name} has {len(chunks)} parts; request "
            f"participant:{role}program:part1 through participant:{role}program:part{len(chunks)}.")
  tail = (f"\nNEXT: request `signal='participant:{role}program:part{part + 1}'`. " if part < len(chunks) else
          f"\nFINAL PART. SHA-256 of the served {name}: `{hashlib.sha256(text.encode()).hexdigest()}`. "
          f"Its holes are yours; it does not compile until each is filled."
          + "".join(f"\n\n{nm}, beside it:\n\n```cmake\n{tx}```" for nm, tx in extra) + "\n")
  return (f"# {key} program scaffold {name} (holes elided): part {part} of {len(chunks)}\n\n"
          f"write_participant_contract(solver='{key}', path='<absolute path of the wrapper .py>') "
          f"writes this same text beside its wrapper, byte for byte, with its CMakeLists.txt. The "
          f"API facts measured for {key} on this install come with knowledge(topic='coupling', "
          f"solver='{key}'). Assembling the parts instead: concatenate only the fenced code "
          "contents in part order into one file.\n\n"
          f"```cpp\n{chunks[part - 1]}```\n{tail}\n")


def coupling_knowledge(solver: str = "", signal: str = "") -> str:
    """knowledge(topic='coupling', solver=..., signal=...) — the core payload,
    one backend's participant script, or the failure entries matching a symptom.

    `signal` is checked FIRST and on its own: someone arriving with a broken run
    wants the two entries that explain it, not 40 kB with them somewhere inside.
    """
    sig = (signal or "").strip()
    if sig.lower().startswith("participant"):
        return coupling_participant(solver, sig)
    if sig and solver:
      import re

      key = _ALIAS_CANON.get(solver.strip().lower())
      symbols = re.findall(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b", sig)
      paths = sorted(_PARTICIPANT_DIR.glob(f"participant_{key}*.py")) if key else []
      needles = [f"def {symbol}" for symbol in symbols] + symbols
      for needle in needles:
        for path in paths:
          variant = path.stem.removeprefix(f"participant_{key}").lstrip("_")
          role = f":{variant}" if variant else ""
          for part, chunk in enumerate(_participant_chunks(path.read_text()), 1):
            if needle in chunk:
              return coupling_participant(
                solver, f"participant{role}:part{part}")
    if sig:
        hits = coupling_signal_search(sig)
        if hits:
            body = "\n\n".join(f"* {h}" for h in hits)
            return (f"# Coupling failure entries matching {sig!r}\n\n{body}\n\n"
                    f"---\nThese are the coupling failure modes whose symptom "
                    f"matches what you described. The full contract, the "
                    f"relaxation guidance and the whole failure table are in "
                    f"`knowledge(topic='coupling')`; a complete runnable "
                    f"participant script for one backend is "
                    f"`knowledge(topic='coupling', solver='<name>')`.")
        # Deliberately NOT followed by the whole core payload. Returning 40 kB
        # on a miss makes a failed search indistinguishable from a hit for
        # anything that only checks whether some word came back — which is
        # exactly how a fixture asserting on this went green against a mutant
        # that searched for nonsense.
        return (f"# No coupling failure entry matches {sig!r}\n\n"
                f"Nothing in the coupling failure table matches that symptom. "
                f"Try describing it differently — what the RESULT looked like "
                f"rather than what you think caused it. The whole table is "
                f"section 7 of `knowledge(topic='coupling')`.")
    key = (solver or "").strip().lower()
    if not key:
        return coupling_core()
    fn = _BACKENDS.get(key)
    if fn is None:
        known = ", ".join(sorted(set(_ALIAS_CANON.get(k, k)
                                      for k in _BACKENDS)))
        return (f"# No coupling participant pattern for solver='{solver}'\n\n"
                f"That name is not one openPASO ships a participant script for. "
                f"The ones it does: {known}.\n"
                f"The general contract below applies to EVERY backend, so you "
                f"can still write a participant for '{solver}' from it — the "
                f"driver needs only a command that reads imports.json and "
                f"writes exports.json.\n\n" + coupling_core())
    return fn()


# ══════════════════════════════════════════════════════════════════════════
# preCICE — served by knowledge(topic='precice', solver=...)
# ══════════════════════════════════════════════════════════════════════════

_PRECICE_CORE = '''\
# preCICE coupling with openPASO — `couple_precice`

## 0. WHEN TO USE IT INSTEAD OF `couple`

`couple` is a file handshake: openPASO starts your solver once per iteration and
moves JSON between the runs. It needs nothing from the solver but a script.

`couple_precice` is the library path: both codes run CONCURRENTLY, stay alive
for the whole simulation, and exchange data through preCICE at every time
window. Use it when you need
  * a transient coupling where restarting the solver each iteration is absurd,
  * implicit sub-iteration inside each time window (true Gauss-Seidel), or
  * preCICE's mesh mapping between genuinely non-matching surfaces.
It costs: every participant must be able to `import precice` (or link
libprecice) IN ITS OWN INTERPRETER. That is the gate — a backend whose
interpreter cannot import precice cannot use this path at all, no matter what
physics it solves.

## 1. WHAT YOU SUPPLY vs WHAT openPASO GENERATES

You supply, as JSON strings:

    participants = [{"name":  "Left",              # preCICE participant name
                     "mesh":  "Left-Mesh",         # the mesh IT provides
                     "writes":["Temperature"],     # data names it writes
                     "reads": ["Heat-Flux"],       # data names it reads
                     "command":["<interpreter>","participant_left.py"]}, ...]
    data      = [{"name":"Temperature","type":"scalar"},
                 {"name":"Heat-Flux","type":"scalar"}]      # or "vector"
    exchanges = [{"data":"Temperature","from":"Left","to":"Right"},
                 {"data":"Heat-Flux",  "from":"Right","to":"Left"}]
    work_dir  = "/abs/path"        scheme = "serial-explicit" | "serial-implicit"
                                            | "parallel-explicit" | "parallel-implicit"
    dimensions = 2                 max_time = 10.0        time_window = 1.0

openPASO writes `work_dir/precice-config.xml` for you, containing:
  * one `<data:scalar|vector>` per entry in `data`;
  * one `<mesh>` per participant, listing every data name it writes or reads;
  * one `<participant>` per participant, with `<provide-mesh>`, a
    `<receive-mesh from=...>` for every partner whose data it reads,
    `<write-data>` / `<read-data>`, and a **`nearest-neighbor` read mapping**;
  * `<m2n:sockets>` for each exchanging pair, `exchange-directory="."`;
  * the `<coupling-scheme:...>` with `<time-window-size>`, `<max-time>`,
    `<participants first=... second=...>` and one `<exchange>` per entry.

For an IMPLICIT scheme it additionally emits:
  * `<max-iterations value="20" />`        — argument `max_iterations`
  * `<relative-convergence-measure limit="1e-6" />` on the exchanged data
                                           — argument `convergence_tol`
  * `<acceleration:aitken>` with `<initial-relaxation value="0.5" />`
                                           — argument `relaxation`
and the read mapping takes `mapping` (`nearest-neighbor` |
`nearest-projection`). Those four ARE forwarded — this note used to say they
were not, which was true of an earlier version of the tool and false of this
one. Checked by generating a config with non-default values and reading them
back out of the XML.

What is still NOT reachable through the tool: a different acceleration TYPE and
an `rbf` mapping. `generate_precice_config` takes both — `acceleration={"type":
"aitken"|"IQN-ILS", "data": ..., "mesh": ...}` and `mapping="rbf"` — and
`couple_precice` passes neither. For those, or for anything
`<coupling-scheme:multi>` needs, call `generate_precice_config` yourself and
launch the participants yourself.

HARD LIMITS OF THE GENERATED CONFIG, know them before you design the coupling:
  * EXACTLY TWO PARTICIPANTS. `<participants first= second= >` names only the
    first two. A third participant is emitted into the XML and then REJECTED by
    preCICE — every process dies with `Participant "X" is not configured for
    coupling scheme`. A real N-way coupling needs `<coupling-scheme:multi>`,
    which this tool does not generate.
  * FOR `serial-implicit`, `exchanges[0]` MUST BE THE FIELD WRITTEN BY THE
    SECOND PARTICIPANT. openPASO silently makes `exchanges[0]` both the
    convergence measure and the acceleration datum, and preCICE only allows
    second-to-first data there: get the order wrong and both sides abort with
    `only data exchanged from the second to the first participant can be used
    for acceleration`.
  * NO EXCHANGE IS MARKED `initialize="true"`, so `requires_initial_data()`
    always returns False and the first participant READS ZERO in the first time
    window. Make your first window one your solver can survive with a zero
    incoming field, or ramp it.
  * MAPPING IS READ-DIRECTION, `consistent`, nearest-neighbor only. There is no
    conservative mapping, so a FLUX or FORCE exchanged across non-matching
    meshes is NOT conserved by the mapping. Exchange intensive quantities where
    you can.
  * every participant runs with `cwd = work_dir` — the SAME directory. Two
    scripts writing the same output filename overwrite each other, and two
    couplings sharing a work_dir clash on the socket files and on
    `precice-config.xml`. One work_dir per coupling run.
  * `converged` in the returned JSON is EXIT CODES plus preCICE's own
    per-window verdict read back from `precice-<name>-iterations.log`. It is
    still not a check on the VALUES: the orchestrator never sees the exchanged
    fields. Read the participant logs, and have each participant print and
    check its own numbers.

## 2. THE PARTICIPANT LOOP — the same skeleton in every code

```python
import precice, numpy as np

p = precice.Participant("<NAME>", "precice-config.xml", 0, 1)   # rank 0 of 1
coords = np.array([[x0, y0], [x1, y1], ...])      # YOUR interface points
vid = p.set_mesh_vertices("<MESH_NAME>", coords)  # same mesh name as in the spec

if p.requires_initial_data():                     # only for *-implicit / initialize
    p.write_data("<MESH_NAME>", "<WRITE_DATA>", vid, initial_outgoing)
p.initialize()

while p.is_coupling_ongoing():
    if p.requires_writing_checkpoint():           # IMPLICIT: save state
        saved = solver.save_state()
    dt = p.get_max_time_step_size()
    incoming = p.read_data("<MESH_NAME>", "<READ_DATA>", vid, dt)
    outgoing = solver.advance(dt, incoming)       # YOUR solve, using `incoming` as a BC
    p.write_data("<MESH_NAME>", "<WRITE_DATA>", vid, outgoing)
    p.advance(dt)
    if p.requires_reading_checkpoint():           # IMPLICIT: redo this window
        solver.restore_state(saved)
p.finalize()
```

CHECKPOINTS ARE MANDATORY FOR IMPLICIT SCHEMES. `serial-implicit` and
`parallel-implicit` sub-iterate each time window. A participant that ignores
`requires_writing_checkpoint()` / `requires_reading_checkpoint()` does NOT hang
— it ABORTS with `The required actions write-iteration-checkpoint are not
fulfilled`, exit code 255, on both sides. For `*-explicit` both calls always
return False, so the same loop is correct there too: write it once, with the
checkpoint calls, always.

## 3. LAUNCH TRAPS — these are what actually goes wrong

  * BOTH PARTICIPANTS MUST RUN AT THE SAME TIME. Each blocks inside
    `initialize()` until the other connects. Starting them one after the other,
    or under a single `mpirun` over both files, hangs until the timeout.
    `couple_precice` starts them concurrently for you; if you launch by hand,
    background the first one.
  * `libprecice` must be loadable, and `LD_LIBRARY_PATH` is NOT optional:
    without it `import precice` fails with
    `libprecice.so.3: cannot open shared object file`. openPASO reads
    `$PRECICE_LIB_DIR` (it has a built-in default) and prepends it for the
    participant processes it launches; set that variable if the library lives
    elsewhere, and use `extra_env` for anything else a participant needs.
  * `pyprecice` must match `libprecice`'s major version. A mismatch shows up as
    an ImportError or an immediate segfault, not as a helpful message.
  * The participant NAME, the MESH name and the DATA names in your script must
    match the spec EXACTLY, character for character. preCICE aborts on a
    mismatch, but only after both sides have connected.
  * A stale `precice-run/` directory in `work_dir` from a killed run makes the
    next run hang on connect. Delete it before re-running.
  * A PARTICIPANT THAT NEVER STARTS MAKES THE OTHER BLOCK FOREVER. preCICE has
    no connect timeout. If one `command` is wrong — bad interpreter, missing
    module — the partner sits in `initialize()` until openPASO's `timeout` fires,
    and because the orchestrator waits on the participants one after another
    the real wall-clock cost is N x timeout. Run each participant's command by
    hand once (it will block at initialize; that is the correct symptom) before
    coupling.
  * `mapping="rbf"` generates INVALID XML — preCICE v3 needs a
    `<basis-function:*>` child that the generator does not emit. And
    `nearest-projection` needs `set_mesh_edges` / `set_mesh_triangles` calls in
    the participant, which the pattern below does not make. Stay on
    nearest-neighbor unless you write the config yourself.
  * preCICE writes its own INFO log to each participant's stdout and openPASO
    returns only a short tail, so your solver's own prints are usually NOT in
    the returned `logs`. Write your diagnostics to a file in `work_dir`.
  * `set_mesh_vertices` expects an (N, dimensions) array. Passing 3-column
    coordinates to a `dimensions=2` config is a shape error at initialize time.

## 4. WHAT preCICE DOES THAT `couple` DOES NOT

  * mesh mapping between non-matching interfaces (nearest-neighbor here;
    nearest-projection and RBF exist in preCICE but the tool does not expose
    them), whereas with `couple` each participant interpolates for itself;
  * implicit sub-iteration WITHIN a time window with Aitken or IQN acceleration
    — genuine Gauss-Seidel, which the `couple` driver's Jacobi loop cannot do;
  * transient couplings without restarting a solver per iteration.
And what `couple` does that preCICE does not: it works with a code that has no
preCICE adapter at all, which on this install is most of them.

## 5. ADAPTER REALITY CHECK

preCICE's own adapter ecosystem (OpenFOAM, CalculiX, SU2, FEniCS via
`fenicsprecice`, deal.II via a community adapter) is separate from whether a
backend can be driven as a plain participant here. What matters for
`couple_precice` is only whether `import precice` works in that backend's
interpreter and whether you can drive that backend's time loop from Python.
Per-backend verdicts: `knowledge(topic='precice', solver='<name>')`.
'''


def precice_knowledge(solver: str = "") -> str:
    """knowledge(topic='precice', solver=...) — core payload, or one backend.

    The `solver` argument used to be accepted and dropped, so every backend got
    byte-identical output while `couple_precice`'s docstring promised
    "Each backend's preCICE participant pattern is available via
    knowledge(topic='precice', solver=...)".
    """
    key = (solver or "").strip().lower()
    canon = _ALIAS_CANON.get(key, key)
    if not canon:
        return _PRECICE_CORE
    entry = _PRECICE_BY_BACKEND.get(canon)
    if entry is None:
        known = ", ".join(_BACKEND_ORDER)
        return (f"# preCICE on solver='{solver}'\n\nNo per-backend preCICE note "
                f"for that name. Backends covered: {known}.\n\n" + _PRECICE_CORE)
    return (f"# preCICE participant: {entry['title']}\n\n"
            f"**Verdict on this install: {entry['verdict']}**\n\n"
            f"{entry['body']}\n\n---\n\n{_PRECICE_CORE}")


# Per-backend preCICE verdicts, and WHICH FIXTURE ESTABLISHES EACH ONE.
#
# This comment used to say "Every CAN was established by running a real
# two-participant coupling through openPASO's own preCICE orchestrator on this
# install" while only TWO of the seven CANs had one. The other five were
# written from an import check or from nothing: the strong fixture's own
# docstring downgrades NGSolve and DUNE to the import GATE, and deal.II, Kratos
# and SPARTA had no preCICE fixture at all — the shipped deal.II participant is
# a `couple` file-handshake wrapper and neither it nor its CMakeLists mentioned
# preCICE, so a reader greping for a preCICE-linked participant found nothing
# and the boldest sentence in the file was the false one.
#
# Every verdict below now NAMES the fixture that establishes it, so the claim
# is checkable rather than merely asserted:
#
#   scikit-fem, FEniCSx
#       scripts/tier2_fixtures/coupling/precice_can_verdicts_proven_by_a_real_run
#   NGSolve, DUNE-fem, deal.II, Kratos
#       scripts/tier2_fixtures/coupling/precice_can_verdicts_for_the_other_four
#   SPARTA
#       scripts/tier2_fixtures/coupling/sparta_precice_load_order_and_coupled_run
#
# Each of those runs a REAL two-participant coupling through the registered
# `couple_precice` tool and checks the EXCHANGED FIELDS — against a closed form
# for the seven FEM pairs, against SPARTA run standalone at the same wall
# temperature for the DSMC one. None of them accepts `converged` as evidence.
#
# Every CANNOT (4C, FEBio) was established by looking for a preCICE entry point
# in the installed code and not finding one — that is a weaker kind of evidence
# than a run, and it is labelled as such in those two entries.
_PRECICE_BY_BACKEND = {
    "fenics": {
        "title": "FEniCSx (dolfinx)",
        "verdict": ("CAN — proven by a real coupled run "
                    "(fixture: precice_can_verdicts_proven_by_a_real_run)"),
        "body": '''\
`import precice` works in the same interpreter as `dolfinx`, and a FEniCSx
participant was coupled to a second code through `couple_precice` end to end:
FEniCSx on the Neumann side against a scikit-fem Dirichlet side, non-matching
interface meshes, serial-implicit, interface temperature and flux checked
against a closed form.

  * You do NOT need `fenicsprecice`. The raw `precice` bindings are enough and
    are what was proven here; the adapter is a convenience layer, not a
    requirement.
  * `LD_LIBRARY_PATH` must include the preCICE lib directory for THIS
    interpreter too — pass it through `extra_env` if you launch by hand.
  * Interface vertices: use `V.tabulate_dof_coordinates()` rows on the
    interface, take the first `dimensions` columns, and keep that order fixed
    for the whole run — preCICE indexes by the vertex ids `set_mesh_vertices`
    returned.
  * Applying the incoming field: fill a `fem.Function` at the interface DOFs
    and use the two-argument `fem.dirichletbc(function, dofs)`; for a Neumann
    datum use `meshtags` plus a subdomain `ds` measure, never bare `ufl.ds`.
  * Getting the outgoing flux: L2-project `-K*grad(T)[n]` and read it at the
    interface DOFs, or use the residual/reaction of the constrained system.
    Do not finite-difference toward a "neighbouring" node on a triangle mesh.
  * Run one MPI rank per participant unless you have configured preCICE for
    parallel participants.''',
    },
    "ngsolve": {
        "title": "NGSolve",
        "verdict": ("CAN — proven by a real coupled run "
                    "(fixture: precice_can_verdicts_for_the_other_four)"),
        "body": '''\
`import precice` works in the interpreter that carries NGSolve, and an NGSolve
participant was coupled to a second code end to end: NGSolve on the Neumann
side against a scikit-fem Dirichlet side, non-matching interface meshes,
serial-implicit, interface temperature and flux checked against a closed form.
NGSolve shares the openPASO venv with scikit-fem here, so that pair is one
interpreter — the coupling is real, but it is not what proves preCICE spans
separate environments; the FEniCSx, DUNE and Kratos pairs are.

TWO NGSolve-SPECIFIC SILENT-WRONG TRAPS, both found by running, and both
avoided in the participant that fixture couples:
  * TWO CONSECUTIVE `gfu.Set(value, definedon=mesh.Boundaries(...))` CALLS
    CANCEL EACH OTHER. The second `Set` zeroes what the first wrote outside its
    own region, so a participant that sets an outer Dirichlet value and then an
    interface value ends up with one of them gone — and the run completes with
    a plausible, wrong answer. Put every Dirichlet value into ONE
    `mesh.BoundaryCF({...})` and `Set` once.
  * `Integrate(grad(gfu)[0], mesh, definedon=mesh.Boundaries("..."))` RETURNS
    EXACTLY 0.0. An H1 GridFunction's gradient has no boundary trace, so this
    silently reports zero flux. Use the reaction/residual method instead: form
    `a.mat * gfu.vec - f.vec` and sum it over the boundary DOFs.
  * Applying an incoming flux is a `LinearForm` term `g * v * ds("interface")`,
    with `g` a `CoefficientFunction`; build it by fitting or interpolating the
    incoming samples, since preCICE hands you values at YOUR vertices.
    "interface" must be a boundary name your mesh carries: a `ds()` over a
    name it does not carry integrates over nothing, with no error.''',
    },
    "skfem": {
        "title": "scikit-fem",
        "verdict": ("CAN — proven by a real coupled run "
                    "(fixture: precice_can_verdicts_proven_by_a_real_run, and "
                    "as the partner in precice_can_verdicts_for_the_other_four)"),
        "body": '''\
`import precice` works in the interpreter that carries scikit-fem, and a
scikit-fem participant was coupled to a second code end to end — in BOTH roles,
Dirichlet and Neumann, and against four different partner codes. It is the
lightest participant of all of them: pure Python, no compilation, no JIT, which
is why it is the standing partner the other backends' couplings are checked
against.

  * Interface vertices come straight out of `mesh.p` — plain numpy, so keeping
    a fixed order is trivial. Slice to the first `dimensions` columns.
  * Applying the incoming value: `skfem.condense(K, f, x=x, D=dirichlet_dofs)`
    with the incoming values written into `x` at those DOFs.
  * Applying an incoming flux: assemble a `LinearForm` over the interface
    `FacetBasis` with the incoming density as the load.
  * Getting the outgoing flux: the residual `K @ x - f` restricted to the
    constrained interface DOFs is the consistent nodal flux, which is both
    easier and more accurate than differentiating the solution.''',
    },
    "dune": {
        "title": "DUNE-fem",
        "verdict": ("CAN — proven by a real coupled run "
                    "(fixture: precice_can_verdicts_for_the_other_four)"),
        "body": '''\
A DUNE-fem participant was coupled end to end: DUNE on the Neumann side, in its
own conda environment, against a scikit-fem Dirichlet side in the openPASO venv,
non-matching interface meshes, serial-implicit, interface temperature and flux
checked against a closed form. Two install-level facts decide whether it works
at all:

  * `precice` and `dune.fem` must be importable in ONE interpreter. If DUNE
    lives in its own conda environment without `pyprecice`, you do not have to
    install anything into it: put the site-packages of the interpreter that HAS
    `pyprecice` on `PYTHONPATH` and the preCICE lib on `LD_LIBRARY_PATH`, and
    the DUNE interpreter imports both. Verify with a one-line
    `python -c "import precice, dune.fem"` BEFORE coupling — a failed import
    leaves the partner blocking in `initialize()` with no error.
    That whole-site-packages form works HERE because the DUNE environment has
    no package the openPASO venv would shadow badly. It is not universal: see the
    Kratos entry, where the same recipe shadows the good install and has to be
    narrowed to a directory of symlinks.
  * DUNE-fem JIT-COMPILES EACH DISTINCT SCHEME. Measured on a cold cache here
    that is MINUTES, not the "about a minute" this note used to claim — the two
    schemes of a Dirichlet-Neumann heat participant took about seven. Build and
    COMPILE the scheme ONCE, BEFORE `precice.Participant(...)` — a throw-away
    `scheme.solve(...)` is what actually triggers the compile, so constructing
    the scheme is not enough — or the partner blocks on connect while you wait.
    Make the coupled boundary datum a `dune.ufl.Constant` (or a discrete
    function) and MUTATE it each window instead of rebuilding the scheme, or
    you pay that compile on every coupling iteration.''',
    },
    "dealii": {
        "title": "deal.II",
        "verdict": ("CAN, as a C++ participant — proven by a real coupled run "
                    "(fixture: precice_can_verdicts_for_the_other_four)"),
        "body": '''\
deal.II has no Python API, so the participant is a compiled C++ executable that
links `libprecice` directly. Such a program was built here and coupled to a
scikit-fem Python participant end to end, non-matching interface meshes,
serial-implicit, interface temperature and flux checked against a closed form.
No solver is served for it: the program is yours to write. Its finite element
part can start from the `couple` participant's served program scaffold
(knowledge(topic='coupling', solver='dealii'): the mesh, assembly, boundary
data and solve are holes you fill; the consistent flux recovery is served),
with the file exchange replaced by the preCICE calls below.

  * Build through CMake with `DEAL_II_SETUP_TARGET` and add
    `target_include_directories(<target> PRIVATE <precice>/include)` plus
    `target_link_libraries(<target> <path-to>/libprecice.so)`. A hand-rolled
    `g++ -I<dealii>/include` does NOT work — it fails on deal.II's own bundled
    headers — and the preCICE flags must go ON TOP of `deal_ii_setup_target`,
    not instead of it.
  * Guard the target on `find_library(precice)`. A hard `find_package` breaks
    the `couple` participant's build on an install with no preCICE, which is
    most of them.
  * `ldd <exe> | grep libprecice` is the one-line check that the thing you
    built is actually a preCICE participant. Nothing else in the tree tells you
    — a deal.II wrapper driven by the file-handshake `couple` driver looks
    identical from the outside and links no preCICE at all.
  * `DEAL_II_DIR` must point at the BUILD/INSTALL tree that contains
    `lib/cmake/deal.II/deal.IIConfig.cmake`. Pointing it at the source
    checkout silently falls back to whatever system deal.II exists, which is
    usually a different, older version.
  * At run time the executable needs the preCICE lib directory on
    `LD_LIBRARY_PATH`; pass it via `extra_env`.
  * The C++ API mirrors the Python one: `precice::Participant`,
    `setMeshVertices`, `readData`, `writeData`, `advance`,
    `requiresWritingCheckpoint` / `requiresReadingCheckpoint`.
  * Community deal.II preCICE adapters exist upstream, but none is needed for
    this: linking the library directly is what was proven here.''',
    },
    "kratos": {
        "title": "Kratos Multiphysics",
        "verdict": ("CAN — proven by a real coupled run "
                    "(fixture: precice_can_verdicts_for_the_other_four), with "
                    "a caveat about WHICH Kratos"),
        "body": '''\
A Kratos participant was coupled to a second code end to end: Kratos on the
Neumann side, in its own interpreter, against a scikit-fem Dirichlet side in
the openPASO venv, non-matching interface meshes, serial-implicit, interface
temperature and flux checked against a closed form. The install is the hard
part, and it is harder than for any other backend here:

  * A Kratos wheel can import cleanly on one host and be unusable on another —
    one on this class of host fails at import with a `GLIBC` version error from
    its bundled shared objects. If that happens, use a Kratos built from source
    and set BOTH `PYTHONPATH` to the install root and `LD_LIBRARY_PATH` to its
    `libs` directory, via `extra_env`.
  * THE PYTHONPATH RECIPE THAT WORKS FOR DUNE DOES NOT WORK HERE, and it fails
    in two different ways at once. Pointing `PYTHONPATH` at the WHOLE
    site-packages of the interpreter that has `pyprecice`:
      - shadows the good Kratos, because `PYTHONPATH` is searched BEFORE the
        target interpreter's own site-packages, so a broken `KratosMultiphysics`
        sitting in the preCICE interpreter wins and dies at import; and
      - breaks `cyprecice`, which is a compiled extension built against ONE
        numpy ABI — import it next to a different numpy and it fails with
        `numpy.core.multiarray failed to import`.
    What works is a NARROW shim: a directory of symlinks to exactly `precice`,
    `cyprecice` (package and `.so`), `numpy`, `numpy.libs` and `mpi4py`, and
    that directory on `PYTHONPATH`. preCICE then gets the numpy it was built
    against and Kratos keeps everything else of its own.
  * PROBE THE WHOLE GATE, not half of it. `import KratosMultiphysics` alone
    picks the wrong interpreter: this host has a system Python that imports
    Kratos fine and is 3.8, so it cannot load a cp312 `pyprecice` at all — it
    passes a Kratos-only probe and then dies inside the coupling on a numpy
    C-extension error. Probe
    `import precice, KratosMultiphysics, KratosMultiphysics.<App>` in ONE
    command, with the shim already on `PYTHONPATH`.
  * A core-only Kratos has NO `ConvectionDiffusionApplication` and therefore no
    thermal element at all. Check `import KratosMultiphysics.<App>` for every
    application your participant needs BEFORE coupling.
  * On the NEUMANN side the incoming flux density goes on as `FACE_HEAT_FLUX`
    on `ThermalFace2D2N` conditions built along the interface — that is
    ConvectionDiffusion's surface-source route, declared with
    `settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)`. Setting the nodal
    variable WITHOUT creating the conditions does nothing at all: there is then
    no boundary integral to carry it, and the participant silently solves an
    insulated problem.
  * Kratos drives its own time loop, so the preCICE loop wraps
    `InitializeSolutionStep()` / `SolveSolutionStep()` /
    `FinalizeSolutionStep()`, and the checkpoint save/restore is a copy of the
    solution-step variables on the interface nodes.
  * Kratos also has its own CoSimulation application. That is a different,
    Kratos-internal coupling path; it is not what `couple_precice` drives.''',
    },
    "sparta": {
        "title": "SPARTA (DSMC)",
        "verdict": ("CAN — proven by a real coupled run (fixture: "
                    "sparta_precice_load_order_and_coupled_run); IN-PROCESS "
                    "only with RTLD_DEEPBIND"),
        "body": '''\
A SPARTA DSMC participant was coupled to a solid conduction participant end to
end: rarefied argon past a cylinder against a lumped thermal shell, ten time
windows, serial-explicit, and the coupled wall temperature checked against
SPARTA run STANDALONE at uniform wall temperatures bracketing it.

RUN SPARTA AS A SUBPROCESS. That is what the coupled run does and what the
shipped participant does: one `spa_serial -in <deck>` invocation per time
window, with the wall temperature written into the deck's
`custom surf ... file` and the flux read back out of a surf dump. Two separate
reasons, and neither goes away:

  * the SPARTA Python library exposes `command` / `extract_global` /
    `extract_compute` / `extract_variable` and NO per-surf scatter, so an
    in-process participant can exchange a SCALAR and nothing more, while the
    deck carries a per-element field;
  * in-process, SPARTA and preCICE fight over MPI symbols — see below.

IF YOU DRIVE IT IN PROCESS ANYWAY, there is exactly one load order that works.
All four were run:

  * `import precice` first, then the stock `sparta.py` wrapper — SEGFAULT.
    `libsparta.so` DEFINES ITS OWN `MPI_*` STUB SYMBOLS and links no real MPI;
    `import precice` pulls a real MPI into the global symbol namespace, SPARTA's
    stub calls are interposed by it, and SPARTA dies inside `PMPI_Type_size`
    with MPI never initialised.
  * `from mpi4py import MPI` first, then preCICE, then SPARTA — SEGFAULT, same
    frame. It does not help.
  * SPARTA first through the stock wrapper (which uses `RTLD_GLOBAL`) — fails
    the other way: `import precice` then dies with
    `ImportError: libmpi.so.12: cannot open shared object file`.
  * THE ONE THAT WORKS: load SPARTA's library yourself with deep binding and
    LOCAL visibility, so its own symbols win inside it and preCICE's inside
    preCICE:

```python
import ctypes, os
mode = os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND
lib = ctypes.CDLL("<path to>/libsparta.so", mode=mode)
import precice                       # only now
```

  * SPARTA is a Monte-Carlo code. Its interface output carries statistical
    noise, so an IMPLICIT scheme's convergence measure may never be met even
    though the physics is fine. An explicit scheme with enough sampling per
    window is the honest choice; if you use implicit, set the tolerance above
    the sampling noise and say so. `couple_precice` reports an explicit scheme
    as UNMEASURED rather than converged, which is the right verdict here — so
    judge a DSMC coupling on its fixed point against standalone runs at the
    same wall temperature, not on anything the orchestrator returns.
  * USE A NEW SEED EACH WINDOW. With a fixed seed the run is bit-reproducible
    and a fixed-point iteration can look converged when only the RNG is
    repeating.
  * A SPARTA surface can take a prescribed TEMPERATURE
    (`surf_collide ... diffuse`) but there is NO surface-collision style that
    accepts a prescribed heat flux, so treat SPARTA as a Dirichlet-side
    participant. A flux can only be imposed indirectly, by converting it to a
    radiating-equilibrium temperature through `fix surf/temp` — see
    `knowledge(topic='coupling', solver='sparta')` for the route and its
    caveats.''',
    },
    "fourc": {
        "title": "4C Multiphysics",
        "verdict": ("CANNOT — 4C has no preCICE entry point (established by "
                    "ABSENCE, not by a run: fixture "
                    "precice_absent_in_fourc_and_febio)"),
        "body": '''\
There is no preCICE support in 4C. Searching the installed 4C source tree for
`precice` returns nothing, the built binary contains no preCICE symbols, and it
links no preCICE library. This is not a configuration problem you can fix from
the outside: adding preCICE to 4C means writing and building an adapter into
4C's own source.

Note what kind of evidence that is. A CAN here is backed by a coupling that
RAN; this CANNOT is backed by not finding an entry point, which is weaker and
is why it is worded as absence.

USE `couple` INSTEAD. 4C works well as a participant in openPASO's file-handshake
driver — it has been run there on BOTH the Dirichlet and the Neumann side of a
cross-code coupling. Call `knowledge(topic='coupling', solver='fourc')` for a
complete runnable 4C participant.''',
    },
    "febio": {
        "title": "FEBio",
        "verdict": ("CANNOT (as installed) — no preCICE in the binary, no "
                    "Python API (established by ABSENCE, not by a run: "
                    "fixture precice_absent_in_fourc_and_febio)"),
        "body": '''\
The installed FEBio binary contains no preCICE symbols and FEBio ships no
Python module, so there is nothing to call `precice` from. FEBio runs an XML
deck to completion and exits; it does not expose a time loop.

A preCICE-enabled FEBio would have to be a compiled FEBio plugin using FEBio's
own callback interface. That is a real route, but nothing of the kind is built
here, so this is UNVERIFIED, not supported.

USE `couple` INSTEAD. FEBio participates fine in openPASO's file-handshake driver
as an XML-writing / log-parsing wrapper. Call
`knowledge(topic='coupling', solver='febio')`.''',
    },
}


def _febio() -> str:
    return _payload(
        "FEBio",
        "**Either side — but NOT for heat.** FEBio 4 has no heat module "
        "(FEBioHeat was removed upstream and survives only as a plugin), so a "
        "conduction participant is impossible here. The shipped script solves "
        "the exact linear analogue instead: a uniaxial-strain elastic bar, "
        "where displacement plays the role of temperature and the P-wave "
        "modulus the role of conductivity. Both roles were run as a real "
        "FEBio-to-FEBio coupling on this install, with non-matching meshes, "
        "and converged.",
        "febio", _launch_py(_interp_wrapper("FEBio", "FEBIO"),
                            _step2_block(_RIGHT_BLOCK_FEBIO,
                                         "the placeholder ELASTIC problem")),
        '''\
* NO SCRIPTING API. FEBio is XML-in, plot/log-out. A participant is therefore a
  wrapper: write a complete `.feb` deck with the imported data baked in, run
  `febio4 -i deck.feb`, parse the ASCII logfile, write exports.json.
* AN UNKNOWN `<Module type>` SEGFAULTS — it does not produce an error message.
  FEBio 4's registered modules are solid, biphasic, solute, multiphasic, fluid,
  fluid-FSI, fluid-solutes, multiphasic-FSI, thermo-fluid and polar fluid. A
  deck asking for a heat module dies with SIGSEGV while reading the file, which
  looks exactly like a corrupted deck.
* PER-POINT interface data is possible and is the whole reason this works:
  - a prescribed field is `<MeshData><NodeData name="..." node_set="...">` plus
    `<bc type="prescribed displacement" node_set="..."><dof>x</dof><value lc="1"
    type="map">...</value><relative>0</relative></bc>`, ONE bc per dof (`<x_dof>`
    belongs to `zero displacement` only; measured on 4.12);
  - a prescribed traction is `<MeshData><SurfaceData data_type="vec3">` plus
    `<surface_load type="traction"><traction type="map">...`.
  In BOTH, `lid` is the 1-based index INTO THAT SET, not the node or face id.
  Getting that wrong silently applies the right numbers to the wrong places.
* Do not build XML numbers with `repr()` or an f-string `!r`: a numpy 2 scalar
  stringifies as `np.float64(0.0)` and FEBio rejects the deck.
* `febio4` has NO `-h`/`--help` flag; passing one is a fatal error. The real
  options are `-i -o -r -s -d -p -g1 -g2 -config -noconfig -import -noappend
  -nosplash -silent`.
* Check for `N O R M A L   T E R M I N A T I O N` in stdout before trusting the
  logfile — FEBio can exit 0 after writing a partial log.
* The elastic analogue of the flux convention: export
  `q_out = -(sigma . n_own)_x`, so the two sides carry opposite signs, and the
  Neumann side applies the partner's number unchanged as a traction.''')


def _sparta() -> str:
    return _payload(
        "SPARTA (DSMC)",
        "**DIRICHLET-TYPE IN PRACTICE.** SPARTA imports a wall TEMPERATURE and "
        "exports the energy flux "
        "the gas deposits, which is exactly the Dirichlet role. There is NO "
        "native flux boundary condition: of the nine `surf_collide` styles "
        "(`adiabatic`, `cll`, `diffuse`, `impulsive`, `piston`, `specular`, "
        "`td`, `transparent`, `vanish`) the four that take a thermal datum — "
        "`diffuse`, `cll`, `td`, `impulsive` — all take a TEMPERATURE, and none "
        "accepts a prescribed heat flux.\n\n"
        "A flux CAN still be imposed INDIRECTLY, and it is worth knowing the "
        "route exists before you conclude the coupling is impossible. "
        "`fix surf/temp` converts a per-surface energy flux into a per-surface "
        "temperature through the gray-body Stefan-Boltzmann law "
        "`q = sigma*emisurf*T^4`, i.e. `T = (q/(sigma*emisurf))^(1/4)`, and it "
        "takes that flux from ANY per-surf compute or fix — SPARTA's own doc "
        "says \"SPARTA does not check that the specified compute/fix calculates "
        "an energy flux\". So an IMPORTED flux reaches it: write the partner's "
        "flux to a file, load it with `custom surf ... file` into a custom "
        "per-surf vector, wrap that vector in `fix ave/surf s_<name>` (which "
        "sets `per_surf_flag`), and hand that fix to `fix surf/temp`.\n\n"
        "READ THE CAVEAT BEFORE USING IT. That is NOT a Neumann condition. It "
        "prescribes the temperature that would RADIATE the imported flux, so it "
        "constrains the wall temperature, not the gas-side flux, and it drags in "
        "an emissivity that has nothing to do with your coupling. The flux "
        "SPARTA then reports is free to differ from the one you imposed; making "
        "the two agree is a feedback loop you have to close yourself, and it is "
        "NOT what the shipped script does. This route was established by reading "
        "the SPARTA source and docs on this install and has NOT been run here — "
        "treat it as available, not as proven. The shipped script and everything "
        "below are the Dirichlet role.\n\n"
        "A full coupling to a thermal "
        "shell was run on this install: the physics agreed across the interface "
        "and the interface energy balance closed. With a plain `tol` it still "
        "reported FAILURE, because the residual cannot fall below the "
        "Monte-Carlo sampling noise; with `noise_replicates` set it converges "
        "against the measured floor and reports that floor. Read the "
        "stochasticity note below before using it.",
        "sparta", _launch_py(_interp_wrapper(
            "SPARTA", "SPARTA",
            extra="\n   List the surf / species / vss files in the participant's\n"
                  "   `data_files` (absolute paths). `couple` stages them into\n"
                  "   `work_dir` before the first iteration and fails loudly on a\n"
                  "   missing one, instead of SPARTA dying on 'Cannot open ...'."),
            _STEP2_SPARTA),
        '''\
* STOCHASTICITY IS THE HEADLINE, AND `couple` HAS A SWITCH FOR IT. DSMC output
  is a Monte-Carlo estimate. Its sampling noise does NOT shrink as the coupling
  iterates, so the driver's relative residual has a FLOOR at the noise level and
  a `tol` below that floor can never be met — the run ends as "did not
  converge", which is honest and useless.
  PASS `noise_replicates=5` (four or more; the floor is itself an estimate and
  three samples is a bad one). The driver then runs each participant that many
  times on the same imports, MEASURES the residual across independent
  replicates, and judges convergence against max(tol, that floor)
  over a block mean. It returns `noise_floor`; every tolerance you or anyone
  else later applies to the result must be at least that. Do NOT guess a `tol`
  "above the noise" instead — a guessed threshold is a number you cannot defend
  and it is the same act as tuning until it passes. Section 4a of
  `knowledge(topic='coupling')` has the details and the limits.
  SET `SEED_MODE = "vary"` FIRST. With a FIXED seed the runs are
  bit-reproducible, the measured floor comes out exactly zero, and an apparently
  converging residual is possible even when the physics has not settled — that
  is more dangerous than the noise, not less. `noise_notes` in the result says
  so when the floor measures zero.
* THE DATA FILES MUST BE IN `work_dir`, where SPARTA opens them. `couple` copies
  every file listed in a participant's `data_files` (absolute paths) into its
  `work_dir` before the first iteration, and stops before any iteration when one
  is missing; it does not read the deck for file names. A file in neither place
  dies inside SPARTA with `Cannot open species file ...`.
* PER-ELEMENT interface data goes in through a custom surf attribute:
  `custom surf create tsurf float 0 file tsurf.in 1 tsurf` and then
  `surf_collide 1 diffuse s_tsurf 1.0`. That is the only route to a spatially
  varying wall temperature.
* Reading the flux back: `compute ... surf all all etot` +
  `fix ... ave/surf ...` + a surf `dump`. THE DUMP MUST REFERENCE `f_1`, NOT
  `f_1[1]` — a `fix ave/surf` with a single input column is a per-surf VECTOR
  and the bracketed form aborts with `Dump surf fix does not compute per-surf
  array`.
* `read_surf` and `read_grid` files of the same name are DIFFERENT geometries.
  Staging the wrong one is accepted and gives a wrong answer.
* If you also want SPARTA under preCICE, its shared library needs
  `RTLD_DEEPBIND` or it segfaults against preCICE's MPI — see
  `knowledge(topic='precice', solver='sparta')`.''')


def _kratos() -> str:
    return _payload(
        "Kratos Multiphysics",
        "**Either side — but NOT in openPASO's own interpreter on this install.** "
        "Kratos is not importable where openPASO runs here, so its participant was "
        "proven in a separate Kratos install: both the Dirichlet and the "
        "Neumann role were run against FEniCSx and both converged with "
        "non-matching interface meshes. The contract below is the DIRICHLET side; "
        "the NEUMANN-side contract follows further down under THE "
        "NEUMANN-SIDE PARTICIPANT, and the trap that decides that role is "
        "listed under the traps.",
        "kratos", _launch_py(step2=_STEP2_KRATOS),
        '''\
* CHECK THE INSTALL FIRST, IT IS THE USUAL FAILURE. `import KratosMultiphysics`
  can fail at import with a GLIBC version error from the bundled shared
  objects even though the package installed cleanly. If that happens, use a
  Kratos built from source and point `PYTHONPATH` at its install root and
  `LD_LIBRARY_PATH` at its `libs` directory.
* A CORE-ONLY KRATOS HAS NO THERMAL ELEMENT.
  `import KratosMultiphysics.ConvectionDiffusionApplication` must succeed
  before a conduction participant can work at all. Test that one line by hand
  before writing anything else.
* The thermal problem is configured through a `ConvectionDiffusionSettings`
  object placed in `mp.ProcessInfo[CONVECTION_DIFFUSION_SETTINGS]`, with
  `SetUnknownVariable(TEMPERATURE)`, `SetDiffusionVariable(CONDUCTIVITY)`,
  `SetVolumeSourceVariable(HEAT_FLUX)` and
  `SetSurfaceSourceVariable(FACE_HEAT_FLUX)`. Every one of those variables must
  also be added with `AddNodalSolutionStepVariable` BEFORE the nodes are
  created, or they silently do not exist on the nodes.
* DIRICHLET SIDE (this script): write the imported temperature into each
  interface node and `node.Fix(TEMPERATURE)`.
  NEUMANN SIDE: do NOT fix the interface nodes; instead set `FACE_HEAT_FLUX` on
  them from the partner's exported `normal_fluxes` AND create one interface
  flux condition per interface edge, corners included (`FluxCondition2D2N`,
  or `ThermalFace2D2N`, through `mp.CreateNewCondition(name, id, [n1, n2],
  props)` -- by NAME through the factory, never as a Python attribute). A
  nodal FACE_HEAT_FLUX is only ever integrated BY a condition: with none,
  Kratos runs, converges, exits 0 and returns exactly the no-flux field
  (measured on one mesh: with the flux on the nodes and no condition the
  field is identical to the zero-flux run's; with the conditions it is not). Check it in one step: solve once with the imported flux
  zeroed and once with it real; if the two fields match, the load never
  arrived. Everything else, the mesh, the material and the export, is
  unchanged.
* Kratos also ships a CoSimulation application. That is Kratos's own internal
  multi-physics coupling, unrelated to `couple`; do not mix the two.
* If a participant needs a `params.json` or any other file, stage it into
  `work_dir` yourself — `couple` copies nothing.''')


# Registered after their definitions (the dispatch table is declared earlier so
# that it sits next to the alias map it mirrors).
_BACKENDS.update({"febio": _febio, "sparta": _sparta, "kratos": _kratos})


def _dune() -> str:
    return _payload(
        "DUNE-fem",
        "**Either side, in either subdomain.** All four role/position "
        "combinations were run as real couplings on this install — against "
        "FEniCSx and against deal.II, with non-matching interface meshes — and "
        "all converged.",
        "dune", _launch_py(),
        '''\
* JIT COMPILATION IS THE THING THAT WILL BITE YOU. DUNE-fem compiles each
  distinct UFL form on first use, and a cold cache can take a minute or more
  per form. The participant runs as a FRESH PROCESS every coupling iteration,
  so if the FORM TEXT changes with the imported data you pay that compile on
  every iteration and the coupling appears to hang.
  THE RULE: keep the form structurally CONSTANT. Put the imported interface
  data into a discrete function's dof vector (`gfun.as_numpy[...] = ...`) or a
  `dune.ufl.Constant`, never into the form's text. Then only the first
  iteration is slow.
* `0 * v * dx` FOLDS TO A DOMAINLESS UFL ZERO and fails with "integral is
  missing an integration domain". Wrap a zero source in
  `dune.ufl.Constant(0.0)` rather than writing the literal.
* THERE IS NO `tabulate_dof_coordinates`. Get the dof-to-coordinate map by
  interpolating the coordinate functions into the same space and reading
  `.as_numpy` — `space.interpolate(x[0]).as_numpy` gives the x of every dof, in
  dof order.
* `structuredGrid` CARRIES NO BOUNDARY IDS. Select the outer and interface
  boundaries with a coordinate predicate,
  `conditional(lt(abs(x[0] - X_IFACE), 1e-8), 1, 0)`, and use the same
  indicator both for the Dirichlet BC and to MASK the Neumann `ds` term. An
  unmasked `ds` puts the interface flux on the whole boundary.
* The default `galerkin(..., solver="cg")` projection used to recover the
  interface flux runs at a loose linear tolerance, so the exported flux carries
  small solver noise. It is far below a 1e-8 coupling tolerance but it is not
  machine precision; tighten the scheme's linear-solver parameters before
  asking for a much tighter coupling tolerance.
* DUNE usually lives in its own conda environment. Use the interpreter
  `discover(query='list')` reports for it, not openPASO's own.

* THE DUNE-fem DIRICHLET-SIDE PARTICIPANT SCAFFOLD (config-driven). The
  handshake mapped onto your dofs, the vertex-ordered mesh access, the P1
  consistent flux-recovery FORMULA and the exports schema are served in the
  block below. THE SIMPLEX MESH, THE WEAK FORM AND THE SOLVE ARE ELIDED, as TWO
  marked holes: (1) the simplex grid and the P1 space, leaving `gridView`,
  `space`, `x`; (2) the form, material, source, boundary conditions and solve,
  leaving `uh` and `F_SRC`. Write them from `prepare_simulation(solver='dune',
  physics='<your physics>')` and the measured facts above, and drop them into
  the marked regions; everything between and after the holes is served and
  measured working (a manufactured solution recovers the interface flux to
  2e-2 at h = 0.1). Traps the served part encodes: the dof->vertex map goes
  through interpolated coordinate fields (never assume dof order equals vertex
  order), and the two interface ENDPOINTS keep the outer Dirichlet value (corner
  rule; the flux export drops them).

```python
"""DUNE-fem as the DIRICHLET side of a partitioned coupling (CG P1).

Reads ./config.json {"level":..,"nx":..,"ny":..,"x0":..,"x1":..,"y0":..,"y1":..,
"k":..,"reaction":..,"outer":<value on the NON-interface boundary, when your
problem prescribes one -- openPASO checks the delivered field against it, and
the equation check cannot: its test function makes every boundary term vanish>,
"source_expr":"<f(x, y) as a Python expression, e.g.
'-10*x**3*y**3/3 + 16*x**2*y/5 - 2'; '0.0' when there is none>","iface":"left|right|bottom|top"}
("source_const": <number> is still accepted for a constant source).
Contract: reads ./imports.json (partner's interface FIELD values at its points),
imposes them as the interface Dirichlet trace, solves -div(k grad u) + c*u = f,
and exports values: [] (the trace is imposed, not owned) plus its OWN consistent
outward flux at the interior interface nodes.

WHAT IS SERVED HERE is the handshake (config + imports + trace mapping), the P1
consistent flux-recovery FORMULA, and the exports schema. THE MESH, THE WEAK
FORM AND THE SOLVE ARE YOURS to write -- see the banner. Get the DUNE-fem API
and a runnable P1 pattern from prepare_simulation(solver='dune',
physics='<your physics>').
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

CFG = json.loads(Path("config.json").read_text())
# A multi-level coupling call hands this level's keys in the environment
# (OPENPASO_CONFIG_JSON, a JSON object) instead of writing this file.
CFG.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))
NX, NY = CFG["nx"], CFG["ny"]
X0, X1, Y0, Y1 = CFG["x0"], CFG["x1"], CFG["y0"], CFG["y1"]
KV, CV, FV = CFG["k"], CFG.get("reaction", 0.0), CFG.get("source_const", 0.0)
IF = CFG.get("iface", "right")
HX, HY = (X1 - X0) / NX, (Y1 - Y0) / NY
# THE SOURCE STRING COMES FROM config source_expr (the task's f(x, y) as a Python
# expression in x, y: '**' for powers, sin/cos/exp/sqrt/pi allowed). It is
# evaluated here ONLY for the consistent load of the served flux recovery
# (F_SRC below); the source term of YOUR form is yours to write in UFL, from the
# same expression. Measured: a run that carried the task's polynomial source
# nowhere (config source_const 0.0, the same zero in its form) converged at
# every level to a smooth field 1.2e-2 off the answer, order 0.00; the check
# after the recovery now refuses a form that disagrees with this string. A
# constant 'source_const' still works; when both are absent the source is zero.
SRC_EXPR = str(CFG.get("source_expr", "")).strip().replace("^", "**")
import math as _math
_SRC_CODE = compile(SRC_EXPR, "<source_expr>", "eval") if SRC_EXPR and SRC_EXPR not in ("0", "0.0") else None
def F_SRC(px, py):
    """The task's source f at a point, from config (a plain Python function used by the served recovery)."""
    if _SRC_CODE is None:
        return float(FV)
    return float(eval(_SRC_CODE, {"__builtins__": {}}, {"x": float(px), "y": float(py), "sin": _math.sin, "cos": _math.cos,
                                                       "exp": _math.exp, "sqrt": _math.sqrt, "pi": _math.pi, "abs": abs}))

# ---- the partner's interface samples, mapped onto THIS side (handshake) ----
imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text())
ax = 1 if IF in ("left", "right") else 0
pts_q = []
for _n, d in imp.items():
    co = d.get("coordinates") or []
    va = d.get("values") or []
    if co and va and len(va) == len(co):
        pts_q = sorted(zip([c[ax] for c in co], [float(v) for v in va]))
        break
def trace(t):
    """The partner's field value interpolated onto one of THIS side's interface
    points. The driver does NOT interpolate between meshes -- each participant
    maps the partner's samples onto its own points, here. Empty on iteration 1,
    so fall back to 0.0. Impose trace(coordinate) on the interface edge in the
    solve below."""
    if not pts_q:
        return 0.0
    xs = [p[0] for p in pts_q]; vs = [p[1] for p in pts_q]
    t = min(max(t, xs[0]), xs[-1])
    for a, b, va_, vb in zip(xs, xs[1:], vs, vs[1:]):
        if a <= t <= b:
            w = 0.0 if b == a else (t - a) / (b - a)
            return va_ + w * (vb - va_)
    return vs[-1]
# SIGN CONVENTION: this is the DIRICHLET side -- it IMPORTS the partner's field
# VALUE, imposes it as the interface trace, and exports its OWN outward flux. The
# two sides' fluxes carry OPPOSITE normals; export yours w.r.t. THIS side's
# outward normal and never write the partner's negated number.

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─────────────────────────────────────
# HOLE 1 OF 2: THE MESH AND THE P1 SPACE ARE YOURS AND ARE NOT SERVED HERE.
# Build the SIMPLEX grid of this subdomain (X0..X1, Y0..Y1, NX x NY cells -- the
# P1 recovery below is exact on triangles, and a structuredGrid makes
# quadrilaterals) and the P1 space, from the DUNE-fem API and the measured
# gotchas in prepare_simulation(solver='dune', physics='<your physics>') and
# knowledge(topic='coupling', solver='dune'). Leave behind exactly these names:
#     gridView   the simplex grid view of this subdomain
#     space      the P1 Lagrange space on it
#     x          the SpatialCoordinate of that space
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─────────────────────────────────────

# ── HANDSHAKE MAPPED ONTO YOUR SPACE (served: the partner's samples on THIS
#    side's interface dofs -- not the solve) ────────────────────────────────
# A discrete function carries the imposed interface datum: its dofs are run-time
# data, so the form text never changes between iterations (no re-JIT). The two
# interface ENDPOINTS keep the outer value (corner rule: they lie on the outer
# Dirichlet boundary, where trace() clamps; imposing the partner trace there is
# O(h)-wrong and caps the whole side at order 1).
_xd = np.array(space.interpolate(x[0], name="_xc").as_numpy)     # every dof's x, in dof order
_yd = np.array(space.interpolate(x[1], name="_yc").as_numpy)
_EPS = 1e-9 * max(X1 - X0, Y1 - Y0)
IF_COORD, IF_VAL = {"left": (0, X0), "right": (0, X1), "bottom": (1, Y0), "top": (1, Y1)}[IF]
_along = _yd if IF_COORD == 0 else _xd                              # the coordinate that runs along the interface
_across = _xd if IF_COORD == 0 else _yd
_lo, _hi = (Y0, Y1) if IF_COORD == 0 else (X0, X1)
on_iface = np.abs(_across - IF_VAL) < _EPS                          # dof mask of the interface edge
_endpoint = (np.abs(_along - _lo) < _EPS) | (np.abs(_along - _hi) < _EPS)
gtrace = space.interpolate(0, name="gtrace")                        # the imposed interface datum
_gd = gtrace.as_numpy
_gd[:] = 0.0
for _i in np.where(on_iface & ~_endpoint)[0]:
    _gd[_i] = trace(float(_along[_i]))
# THE COEFFICIENTS AS UFL CONSTANTS (served: API plumbing, not the form). A bare
# Python float in a form is folded by UFL: `0.0 * u * v * dx` for a zero reaction
# becomes a domainless Zero and dies with "This integral is missing an
# integration domain" (measured in two of six worker trials). Use these in your
# form, never KV / CV themselves.
from dune.ufl import Constant as _Constant
K_UFL = _Constant(KV, name="k")
C_UFL = _Constant(CV, name="c")
# In your solve below: impose gtrace on the interface edge and the task's outer
# condition on the rest of the boundary. The interface edge as a UFL predicate:
#     conditional(lt(abs(x[IF_COORD] - IF_VAL), _EPS), 1, 0)
# (conditional, lt from ufl; abs is the Python built-in).

# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─────────────────────────────────────
# HOLE 2 OF 2: THE WEAK FORM, THE MATERIAL, THE SOURCE, THE BOUNDARY CONDITIONS
# AND THE LINEAR SOLVE ARE YOURS AND ARE NOT SERVED HERE. Write them for the
# problem you were given (-div(k grad u) + c*u = f on this subdomain, with the
# served UFL constants K_UFL and C_UFL as k and c -- never the bare floats),
# impose gtrace on the interface edge and the task's outer condition elsewhere,
# and solve. Leave behind exactly these names:
#     uh       the solved P1 function (uh = space.interpolate(0, name="uh");
#              scheme.solve(target=uh))
#     (F_SRC is ALREADY DEFINED above from config source_expr for the served
#      recovery; YOUR form's load is yours to write in UFL from the same
#      expression, with x = SpatialCoordinate(space) and ufl.sin/cos/exp.
#      Redefine F_SRC only when the task's source cannot be written as one
#      expression string -- and keep the form and F_SRC the same f: the
#      recovery below integrates F_SRC and refuses a field whose interior
#      residual against it is not small)
# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─────────────────────────────────────

# ── VERTEX-ORDERED ARRAYS FROM YOUR MESH (served: mesh access, not the solve) ──
_idx = gridView.indexSet
node_coords = np.zeros((gridView.size(2), 2))
for _v in gridView.vertices:
    node_coords[_idx.index(_v)] = _v.geometry.center
elements = [[_idx.subIndex(_e, _i, 2) for _i in range(len(_e.geometry.corners))]
            for _e in gridView.elements]
if any(len(_e) != 3 for _e in elements):
    raise SystemExit("the P1 recovery below needs a SIMPLEX grid (triangles); this grid has "
                     f"cells with {sorted({len(_e) for _e in elements})} corners")
_key = {(round(float(_xd[_i]), 9), round(float(_yd[_i]), 9)): _i for _i in range(len(_xd))}
_d2v = np.array([_key[(round(float(_px), 9), round(float(_py), 9))] for _px, _py in node_coords])
u_vert = np.array(uh.as_numpy, float)[_d2v]                          # the solution in VERTEX order
f_vals = np.array([F_SRC(float(_px), float(_py)) for _px, _py in node_coords], float)
_if_nodes = [_n for _n in range(len(node_coords)) if abs(node_coords[_n][IF_COORD] - IF_VAL) < _EPS]
_if_nodes.sort(key=lambda _n: node_coords[_n][1 - IF_COORD])
interior = _if_nodes[1:-1]                                           # endpoints dropped (corner rule)

# ── CONSISTENT OUTWARD FLUX + EXPORTS -- the served recovery FORMULA ────────
# ONE formula, every backend, both sides. Your solve leaves an assembled operator
# K (the stiffness KV*grad(u).grad(v) plus any reaction CV*u*v) and a VOLUME load
# b_vol -- the P1-CONSISTENT element load (the source in the load as Me @ f_el,
# NOT a lumped nodal f), reaction and source in the load, with NO Dirichlet
# lifting and the constrained rows NOT zeroed. On the FREE interface rows the
# residual r = K u - b_vol then equals the interface functional, so
#
#     q_i = -r_i / w_i        w_i = interface nodal weight  (= h_if, uniform P1)
#
# is the consistent outward flux density at interface node i. It is mesh- and
# material-agnostic -- the same expression the Dirichlet and Neumann sides both
# use, and exactly what the interface check compares; keep the P1 consistent load
# or the recovery loses an order (a projected -k grad(u) on the boundary is only
# order ~1 there).
#
# Formed here over YOUR mesh with the standard P1 element matrices (Ke, Me below
# are the same for every P1 triangle -- they are not tied to any served mesh):
resid = np.zeros(len(node_coords))
_ku_part = np.zeros(len(node_coords))      # the operator side K u + c M u, for the served check below
_bv_part = np.zeros(len(node_coords))      # the served consistent load M f
for el in elements:                       # el = the 3 vertex ids of one triangle
    P = node_coords[list(el)]
    area = 0.5 * abs((P[1,0]-P[0,0])*(P[2,1]-P[0,1]) - (P[1,1]-P[0,1])*(P[2,0]-P[0,0]))
    gr = np.array([[P[1,1]-P[2,1], P[2,0]-P[1,0]],
                   [P[2,1]-P[0,1], P[0,0]-P[2,0]],
                   [P[0,1]-P[1,1], P[1,0]-P[0,0]]]) / (2.0 * area)
    Ke = KV * area * (gr @ gr.T)                       # element stiffness
    Me = area / 12.0 * (np.ones((3, 3)) + np.eye(3))   # CONSISTENT mass, not lumped
    ue = u_vert[list(el)]
    _ku_part[list(el)] += Ke @ ue + CV * (Me @ ue)
    _bv_part[list(el)] += Me @ f_vals[list(el)]
resid = _ku_part - _bv_part
h_if = HY if IF in ("left", "right") else HX
# THE FORM AND THE SERVED LOAD MUST AGREE (served check). On the free interior
# vertices r = K u - b_vol is only the quadrature difference between DUNE's
# integration of the source and the P1-consistent load, a few percent of the
# interface functional at most; a source or coefficient that is in config but
# not in your form (or the reverse) leaves an O(1) interior residual.
_ifset = set(_if_nodes)
_outer_set = {_n for _n, (_px, _py) in enumerate(node_coords)
              if (abs(_px - X0) < _EPS or abs(_px - X1) < _EPS or abs(_py - Y0) < _EPS or abs(_py - Y1) < _EPS)
              and _n not in _ifset}
_free = [_n for _n in range(len(node_coords)) if _n not in _ifset and _n not in _outer_set]
_r_in = max((abs(float(resid[_n])) for _n in _free), default=0.0)
_scale_in = max(max((abs(float(_ku_part[_n])) for _n in _free), default=0.0),
                max((abs(float(_bv_part[_n])) for _n in _free), default=0.0))
if _scale_in > 0 and _r_in > 0.25 * _scale_in:
    # IT REPORTS, IT DOES NOT EXIT. The ratio below is a function of the
    # COUPLING ITERATE, not of the form: the same file, untouched, passed this
    # check on 79 consecutive solves and converged two levels to ~9e-7, then
    # tripped it at level 3 iteration 13 -- and the message named the form.
    # Exiting discarded a field that was already written to disk. The defect it
    # is built for is real (a source in the config but not in the form leaves an
    # O(1) interior residual), so it still speaks, on stderr, in the console the
    # driver keeps.
    print(f"EXPORT SELF-CHECK WARNING: on the interior vertices your solution's operator side K u + c M u and the "
          f"served consistent load M f differ by {_r_in:.3e}, {_r_in / _scale_in:.2f} of their size (a form "
          f"that integrates the same f, k and c leaves a few percent). IF THIS FIRES ON EVERY ITERATION "
          f"your form and config disagree on k ({KV}), the reaction ({CV}) or source_expr ({SRC_EXPR!r}), "
          f"and the load of your form must integrate the same source with the same numbers. If it fires only "
          f"at some iterates, it is measuring the partner's current trace and not your form -- read the "
          f"coupling's residual history instead. The export CONTINUES so this level is not lost.",
          file=sys.stderr)
q_own = [float(-resid[n] / h_if) for n in interior]    # interior = interface ids[1:-1]
co_out = [[float(node_coords[n][0]), float(node_coords[n][1])] for n in interior]
# exports.json LAST (the driver takes its existence as proof of success).
# values: [] -- the DIRICHLET side does not OWN a value, it IMPOSED one; echoing
# the imposed trace back trips the driver's per-block change checks.
# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a flux of ~0 against a nonzero partner); and a flux that is the partner's
#    array negated instead of a recovery from THIS side's own system.
# (json, numpy and Path are the imports at the top of this file)
_chk_vals = np.asarray([], float).ravel()
_chk_flux = np.asarray(q_own, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes; "
                     "the solve did not produce a usable field, so nothing was exported")
_chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
            if Path("imports.json").is_file() else {})
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                             for _d in _chk_imp.values()])
            if _chk_imp else np.zeros(0))
if False and _chk_qin.size and np.abs(_chk_qin).max() > 0 and (
        np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max()):
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0 against a "
                     "nonzero imported flux: the imported load never entered the "
                     "assembled system (the condition that integrates it is missing). "
                     "Fix the application; do not couple on")
if True and _chk_qin.shape == _chk_flux.shape and _chk_flux.size and (
        np.array_equal(_chk_flux, -_chk_qin)):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's array "
                     "negated, bit for bit: a copy, not a recovery from this side's "
                     "own assembled system")
json.dump({"field_name": "u", "coordinates": co_out, "values": [],
           "normal_fluxes": q_own, "n_points": len(co_out)},
          open("exports.json", "w"))
# PER-LEVEL PERSISTENCE. Each level writes its own field file (named by the
# config level) so the coarse levels are not overwritten by the finest. Assemble
# the per-level field file for each side, named as your task prescribes, from
# THESE per-level files (interpolated to the task's probe points), never from
# one output the next level overwrites.
_LVL = CFG.get("level", "X")
with open(f"field_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,u\\n")
    for (_px, _py), _u in zip(node_coords, u_vert):
        _f.write(f"{_px:.11e},{_py:.11e},{float(_u):.11e}\\n")
# and its own interface trace and flux at THIS level's interface vertices
# (exports.json is overwritten by the next level; this file is not).
_uv = {(round(float(_px), 10), round(float(_py), 10)): float(_u) for (_px, _py), _u in zip(node_coords, u_vert)}
with open(f"interface_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,u,qn\\n")
    for (_px, _py), _q in zip(co_out, q_own):
        _f.write(f"{_px:.11e},{_py:.11e},{_uv.get((round(float(_px), 10), round(float(_py), 10)), float('nan')):.11e},{float(_q):.11e}\\n")
# THE RUN-LOG CONTRACT LINE: `NDOF = <integer>` on a line of its own (the
# audit reads that exact shape); the descriptive line follows.
print(f"NDOF = {len(u_vert)}")
print(f"DUNE Dirichlet participant: NDOF = {len(u_vert)}  "
      f"max|u| = {float(np.abs(u_vert).max()) if len(u_vert) else 0:.6e}")

# ── WHAT YOUR TWO HOLES MUST LEAVE BEHIND ──────────────────────────────────
# The served code above uses these names; the elided blocks have to define
# every one, or the rest will not run:
#
#     hole 1:  gridView (a simplex grid view), space (P1 Lagrange on it),
#              x (SpatialCoordinate(space))
#     hole 2:  uh (the solved P1 function), F_SRC (your source as a Python
#              function of (x, y))
#
# openPASO does not serve the solve, but it will not make you guess which names
# the holes were filling.
```''')


def _dealii_sources() -> str:
    """What is served in the deal.II program and what is the agent's.

    This function once inlined the complete solvers -- heat_iface_dealii.cc and
    its siblings -- under "save these to disk, then build": a 103 kB payload
    that handed over a working finite element program. Then it served nothing
    of the program at all, and the model wrote the whole thing: measured on two
    coupled rounds, one of eight deal.II sides ever wrote exports.json. The
    program is now served as a scaffold whose holes are elided, the way every
    Python contract is served; this says what the served part does for you.
    """
    return (
        "\n## THE PROGRAM: WHAT IS SERVED AND WHAT IS YOURS\n\n"
        "Yours, in dealii_side.cc's five holes: the mesh and its boundary ids, the "
        "coefficient and the source, the assembly, the boundary data, the solve. "
        "The transient program, dealii_side_transient.cc, has the same five, its "
        "last three run once per time step (the step's system, the boundary data at "
        "the new time, the step's solve); its loop over the steps with the last "
        "step's field kept, the partner's data for each step, the recovery of each "
        "step's theta-averaged flux and the per-step output are served. "
        "The elastic program, dealii_side_elastic.cc (physics='elasticity', a "
        "displacement with two components), has five of its own: the mesh and its "
        "boundary ids, the material law and the body force, the assembly of the "
        "stiffness and load, the boundary data per component, the solve; its "
        "two-component element and each dof's component, the interface nodes per "
        "component, the traction recovery per component and checks of the law (three "
        "unit strains) and of the stiffness (zero on the rigid motions, the energy "
        "of the uniform strains) are served. "
        "Served, and running as it stands once the holes are filled: reading the "
        "wrapper's input; `NDOF = <n>` and `VOLUME_SOURCE on` on the console; the "
        "interface nodes in order along the interface; the CONSISTENT outward flux "
        "q_i = -(A u - b_volume)_i / w_i, from the matrix and volume load your holes "
        "assembled with no boundary condition in them (w_i = the integral of phi_i "
        "over the interface; an interface end that the outer condition also holds "
        "takes its nearest interior neighbour's value), the same recovery every "
        "served contract uses; and the two output files the wrapper turns into "
        "exports.json and the per-level dumps. Between the holes the program checks "
        "your fills and stops with the reason: interface boundary ids that are not "
        "exactly the interface line; a `source` that is not the source the wrapper "
        "passed; a volume load with no source in it; held edges that are not the "
        "ones the input names; an interface that is free on the Dirichlet role or "
        "held on the Neumann role; a Neumann load that is missing, or entered with "
        "the opposite sign; boundary values applied to system_matrix itself (the "
        "recovery reads it); and a solution that does not solve the system.\n")


def _dealii_program_head(program: str = "dealii_side.cc") -> str:
    """The heading and paragraph served above a wrapper's program, for that program."""
    if program == "dealii_side.cc":
        return _DEALII_PROGRAM_HEAD
    variant, word = (("transient", "transient") if "transient" in program else ("elasticity", "elastic"))
    return (_DEALII_PROGRAM_HEAD.replace("dealii_side.cc", program, 1)
            .replace("The same write_participant_contract call",
                     f"The same write_participant_contract(..., variant='{variant}') call", 1)
            .replace("signal='participant:program:part1'", f"signal='participant:{word}:program:part1'", 1))


_DEALII_PROGRAM_HEAD = (
    "## THE PROGRAM THE WRAPPER RUNS: dealii_side.cc and its CMakeLists.txt\n\n"
    "The same write_participant_contract call writes both beside the wrapper, "
    "byte for byte as shown here (for a client that truncates long replies they "
    "are also `signal='participant:program:part1'` and on). The wrapper builds "
    "the program into ./build on its first run and whenever the .cc is newer "
    "than the binary, and prints the first compiler errors when the build fails. "
    "AS SERVED IT DOES NOT COMPILE: fill its five holes. The comment above each "
    "is its contract -- the names it may use, what it must leave set, and what "
    "the lines after it check -- and the deal.II calls are the facts in this "
    "reply (WHAT DECIDES THIS RUN), one call each. Everything else in it is "
    "served and runs as it stands.\n\n")


def _dealii() -> str:
    text = _payload(
        "deal.II",
        "**Either side, in either subdomain.** The served program, its holes "
        "filled for a manufactured problem, was run through `couple` in both "
        "roles against the served NGSolve contract with non-matching interface "
        "meshes, and converged; standalone, its consistent interface flux (the "
        "Dirichlet role) and its interface trace (the Neumann role) converge at "
        "second order in h. Earlier deal.II programs were coupled against "
        "FEniCSx and DUNE-fem in all four role/position combinations.\n\n"
        "THE PARTICIPANT IS TWO FILES AND ONE BUILD, steady, transient or elastic. "
        "The Python wrapper below has no hole: it writes the program's input, "
        "builds the program, runs it and exports. The program is the solve, and its "
        "five holes are YOURS TO WRITE; everything around them is served. The "
        "steady program is dealii_side.cc (mesh, coefficient and source, assembly, "
        "boundary data, solve); a time-dependent problem has "
        "dealii_side_transient.cc, which marches the whole window per run through "
        "the same five; a displacement field has dealii_side_elastic.cc (mesh, "
        "material law and body force, assembly, boundary data, solve), served with "
        "physics='elasticity'.",
        "dealii", _launch_py(_interp_wrapper(
            "deal.II", "DEALII_EXE",
            extra="\n   The wrapper BUILDS the program itself (cmake + make into ./build)\n"
                  "   from dealii_side.cc and CMakeLists.txt beside it; set DEAL_II_DIR in\n"
                  "   its edit block to the deal.II tree discover(query='list') names.")),
        '''\
* A deal.II INTERFACE INTEGRAL CAN EVALUATE TO EXACTLY 0.0 ON A NON-HYPERCUBE
  MESH, with nothing raised and the solve reporting success. Two independent
  causes, both measured on 9.8.0-pre Release.
  (a) `ReferenceCell::get_gauss_type_quadrature` returns an EMPTY rule above the
  order its underlying formula implements: QGaussSimplex and QGaussWedge stop at
  n_points_1D = 4, QGaussPyramid at 2. The only guard is a debug-only Assert, so
  a Release build accepts a rule with zero points, `FEValues::reinit` succeeds,
  and every integral over that cell is exactly zero -- a transmitted flux of 0.0
  that reads as a legitimately quiet boundary.
  (b) `MappingQ1` is correct ONLY on hypercubes; on simplex, wedge or pyramid
  cells its Jacobians are silently wrong, measured as the reference triangle
  integrating to 1/6 instead of 1/2, the wedge to 0.052778 instead of 1/2, and
  the pyramid to 4.0 instead of 4/3 -- so the flux is off by a constant factor
  while the coupling still converges. Use `MappingFE<dim>(FE_SimplexP<dim>(1))`,
  `MappingFE<dim>(FE_WedgeP<dim>(1))` or `MappingFE<dim>(FE_PyramidP<dim>(1))`
  to match the cell.
  Signal: assert `quadrature.size() > 0` before assembling, and compare the sum
  of `FEValues::JxW` over a cell against `cell->measure()` or
  `reference_cell().volume()` -- they agree to roundoff when the rule and the
  mapping are both right, and disagree by a constant factor, or give exactly
  0.0, when either is wrong. Neither bites a hypercube mesh built with
  `subdivided_hyper_rectangle` and plain `QGauss`, which is what the served
  program uses; both become live the moment a participant is written on
  simplices.
* `DEAL_II_DIR` MUST BE THE BUILD OR INSTALL TREE that discover(query='list')
  names for deal.II (it contains `lib/cmake/deal.II/deal.IIConfig.cmake`). A
  COMPILE THAT READS /usr/include/deal.II IS BUILDING AGAINST A SYSTEM PACKAGE,
  whatever cmake's "Using the deal.II-..." line said: measured, a find_package
  without HINTS ignores -DDEAL_II_DIR and takes the /usr package, and a
  CMakeLists that links dealii::dealii in place of deal_ii_setup_target, with
  no build type, dies there on "mpi.h". The served six lines do neither, and
  the program's version check stops an older deal.II at its first line.
* DO NOT try to compile with a bare `g++ -I<dealii>/include`. deal.II's bundled
  headers (Kokkos and friends) are only found through CMake's
  `DEAL_II_SETUP_TARGET`, which the served CMakeLists.txt calls.
* THE RECOVERY READS system_matrix AS YOUR ASSEMBLY LEFT IT. deal.II's own
  tutorials apply the boundary values to the system matrix in place
  (`MatrixTools::apply_boundary_values(boundary_values, system_matrix, ...)`),
  which empties every held row but its diagonal; the reaction those rows carry
  is then gone. Keep system_matrix as assembled and apply them to a copy: a
  SparseMatrix<double> built on `sparsity` (or given reinit(sparsity)), then
  filled by copy_from(system_matrix). A SparseMatrix copy-constructed from
  system_matrix does not copy it: deal.II leaves it empty, and the library
  call that uses it (apply_boundary_values, condense, a direct solver's
  initialize) crashes with no message, with DEBUG on or off (measured). The
  program stops if a held row of system_matrix has lost its off-diagonal
  entries.
* Neumann side: the partner's flux is `+ integral(g * phi_i) ds` over the
  interface FACES only, the ones with boundary id INTERFACE_ID. deal.II will
  integrate over every boundary face if the loop does not select them.''',
        extra=_dealii_sources())
    # "write the solve where the banner sits" is the Python contracts' step; this
    # wrapper has no banner, its program has the holes. Stale text raises here.
    step = "Then edit the marked block IN PLACE and write the solve where the banner sits."
    assert step in text, "the deal.II door lost the contract heading it rewrites"
    text = text.replace(step, "Then edit the marked block IN PLACE, and fill the five holes of the "
                              "program the same call writes beside it, dealii_side.cc.", 1)
    # THE PROGRAM RIGHT BEHIND THE WRAPPER, so the reply that carries the one
    # carries the other.
    files, err = participant_companions("dealii")
    i = text.find("## PARTICIPANT CONTRACT")
    j = text.find("```python", i)
    k = text.find("```", j + 9) + 3
    block = (f"```cpp\n{files[0][1]}```\n\n```cmake\n{files[1][1]}```\n" if files and not err
             else f"[{err or 'the program scaffold is missing from the install'}]\n")
    return text[:k] + "\n\n" + _dealii_program_head() + block + text[k:]


_BACKENDS.update({"dune": _dune, "dune-fem": _dune, "dunefem": _dune,
                  "dealii": _dealii, "deal.ii": _dealii})
