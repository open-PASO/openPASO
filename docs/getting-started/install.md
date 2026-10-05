# Install

!!! important "openPASO is not useful on its own"
    openPASO is the part that drives the solvers. The thinking is done by an AI model, which you
    bring. Before you install, know which of the two ways below you will use. The install itself is
    the same either way.

| | Option A: an AI app | Option B: your own API key |
|---|---|---|
| **You need** | Claude Code, Claude Desktop or Cursor | an account at openrouter.ai |
| **Extra cost** | none beyond your subscription | you pay for what you use |
| **Choice of model** | whatever the app offers | any model on OpenRouter |
| **Good for** | trying it out, everyday work | scripting, experiments, cheap models |

Not sure whether your app works? If you have Claude Code, type `claude mcp list` in a terminal. If
the command exists, use Option A.

## What you need

- **Python 3.10 to 3.13**
- **at least one** solver. You do not need all nine; openPASO tells you what is missing and how to get it.

Check your Python:

```bash
python3 --version    # 3.10, 3.11 or 3.12: go straight on. 3.13: read the box below.
```

??? warning "On Python 3.13 you need a C compiler"
    The install takes about two minutes longer. openPASO keeps numpy below version 2, because
    pyprecice 3.1.2, the Python binding that pairs with the preCICE 3.1.2 coupling library, requires
    that (only 3.1.2 and its release candidate pin numpy below 2; the pyprecice 3 releases before
    and after them do not), and for Python 3.13 no ready-made numpy below version 2
    exists, so `pip` has to compile it.

    - Debian or Ubuntu: `sudo apt install build-essential`
    - macOS: `xcode-select --install`
    - Fedora or RHEL: `sudo dnf install gcc gcc-c++ make`
    - Windows: install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
      with "Desktop development with C++", **or, much simpler, install Python 3.12** from
      <https://www.python.org/downloads/>.

    Without a compiler the install stops on numpy with `Unknown compiler(s)` and
    `metadata-generation-failed`. Python 3.14 and newer are untested.

    **If your Python is older than 3.10 or newer than 3.13**, install a supported one beside it. They
    live side by side without conflict. With conda: `conda create -n paso python=3.12 && conda activate paso`.

## Install openPASO

From PyPI, into a fresh virtual environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install openpaso              # the server, with scikit-fem as a first solver
openpaso doctor                   # which solvers openPASO can use on this machine; no key, no network
openpaso                          # starts the MCP server on stdio; Ctrl-C stops it
```

Missing a solver? `openpaso install ngsolve` (or `kratos`, `dune`, `fenics`, ...) checks first and
installs only what is missing, the same way the server's `setup_backend` tool would. 4C and
SPARTA have no pip or conda package, and deal.II none on pip: `openpaso install fourc --via spack`
(or `dealii`, `sparta`) builds them with Spack, see [With Spack](#with-spack). For FEBio it says
what to do by hand.

`openpaso` is the command your AI app's MCP configuration points at
([Option A](../use/ai-app.md)). scikit-fem comes with it; the other solvers that pip can
install are extras -- `pip install "openpaso[ngsolve]"`, `[kratos]`, `[dune]`, or `[all-solvers]`
-- FEniCSx comes from conda, 4C, deal.II and SPARTA from Spack or your own build, and FEBio
from its installer; openPASO finds all of them on your machine, see [More solvers](#more-solvers).

From a checkout instead -- to change the code, or to run `check_install.py`, which lists the solvers
openPASO can use on your machine without a key or network:

```bash
git clone https://github.com/open-PASO/openPASO.git
cd openPASO
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install scikit-fem            # the easiest solver to start with
python check_install.py
```

Then install the agent packages. Only Option B uses them, and they need the checkout:

```bash
pip install -r langgraph_eval/requirements-langgraph.txt
```

!!! note "Windows"
    Use `python -m venv .venv` and then `.venv\Scripts\activate`. A few other commands differ:
    `copy` instead of `cp`, `dir` instead of `ls`, `set` instead of `export`. openPASO is developed
    and tested on Linux. It should work on Windows, but we do not measure that, and the solvers that
    build from source (4C, SPARTA, deal.II) are the least likely to.

## More solvers

Each solver has its own page with the exact install command and what openPASO knows about it:
see [Solvers](../solvers/index.md). In short:

```bash
pip install ngsolve scikit-fem meshio      # into the same .venv
pip install KratosMultiphysics-all         # Kratos: use the -all package
pip install dune-fem mpi4py                # DUNE-fem: mpi4py is a hidden requirement
conda create -n fenics -c conda-forge fenics-dolfinx   # FEniCSx: its own conda environment
```

Any `export NAME=value` line only lasts until you close the terminal. To keep it, add the same line
to the end of `~/.bashrc` (or `~/.zshrc`).

### With Spack

For the codes pip cannot install, [Spack](https://spack.io) builds them from source, with their
whole dependency tree, into a directory of its own:

```bash
openpaso install dealii --via spack     # deal.II 9.7.1, serial, the features the templates use
openpaso install sparta --via spack     # SPARTA, the DSMC code; a few minutes
openpaso install fourc --via spack      # 4C with its whole dependency tree; over an hour
```

- **No Spack yet?** The command offers to clone it to `~/spack` (about 30 MB). A Spack older
  than 1.0 is passed over; if it is the only one, the command prints the clone command for a
  current one.
- **Build times** were measured with the system's cmake, perl, openssl and Open MPI registered
  in Spack (`spack external find`). Spack builds whatever is not registered, which takes longer.
- **The recipes come with openPASO:** 4C (upstream Spack has none), SPARTA (upstream's `sparta`
  is a different program), and deal.II (upstream's, fixed so programs can build against it outside
  Spack). They reach Spack for openPASO's own commands only and are not added to your Spack
  configuration. Spack itself still writes there as for any command: where no compiler is
  configured yet, the first build records the compilers it finds in `~/.spack`.
- **No variable to set**, as long as no other build of the same solver comes first in openPASO's
  search (a variable such as `DEAL_II_DIR`, a conda environment, a build in your home folder). If
  one does, the command prints the variable that selects the Spack build, and so does the server's
  `setup_backend` when it verifies the install. openPASO remembers which
  Spack built a solver, so an AI app that starts the server without Spack on its PATH still finds
  it.
- **No terminal to answer in** (a script, a CI job): pass `--yes`, or the Spack build is not
  started.
- **Which solvers have a Spack route?** For a missing solver, `openpaso doctor` names it.

### When a recipe needs work: spack-agent

[spack-agent](https://github.com/Hereon-InstituteMS/spack-agent) lets an AI agent repair or write
a Spack recipe, and builds the spec after each change until a build passes. openPASO prepares such
a run and starts it:

```bash
openpaso install sparta --via spack-agent   # repair openPASO's recipe where it fails here
openpaso install febio --via spack-agent    # write a recipe FEBio does not have yet
```

- **Targets:** 4C, deal.II and SPARTA (openPASO's recipes) and FEBio (a new recipe). openPASO
  finds each build through Spack afterwards.
- **Install spack-agent first** (Python 3.11 or newer), with the agent program it drives logged in.
  The spack-agent this command installs drives the GitHub Copilot CLI and no other agent.
  `--agent claude` or `--agent openai` need a spack-agent that offers them; openPASO asks the
  installed one before anything starts and stops if it does not. `openpaso doctor` says whether
  spack-agent is installed.
  ```bash
  python -m pip install "spack-agent @ git+https://github.com/Hereon-InstituteMS/spack-agent"
  ```
- **What openPASO prepares**, under its state directory and never in its install:
  - a git copy of its recipes, where the agent edits;
  - the solver's source, cloned at the release tag unless `--source` names your own checkout;
  - a Spack wrapper that adds the recipes, as `--via spack` does;
  - the spack-agent configuration.
- **Afterwards**, `git -C <the copy> diff` shows what the agent changed, and the command re-checks
  the solver. A later run starts again from openPASO's recipes and keeps the edited copy as
  `recipes.previous`. Only one run per solver can go at a time; a second one stops before it
  changes anything.
- **The agent edits and builds on your machine, with your rights.** Each round is a full build
  (over an hour for 4C), so the command asks first; `--max-iterations` caps the rounds.

**If a solver will not install**, ask openPASO once it is connected:

> How do I install 4C on Ubuntu? Use the knowledge tool.

Next: [check that it works](check.md).
