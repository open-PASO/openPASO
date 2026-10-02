# openPASO web interface

A browser interface for openPASO: describe a simulation in plain language, and
watch openPASO choose a solver, write the input, run it and report back, with
every step visible.

## Start it

```bash
# the interface's own dependencies, declared in pyproject.toml
pip install -e ".[webui]"

# run
uvicorn webui.app:app --port 8080
# open http://localhost:8080
```

The built interface is committed in `webui/static/`, so Node is not needed to
use it. To change the interface, edit `ui/src/` and rebuild:

```bash
cd ui && npm install && npm run build     # writes webui/static/
```

## Models

The model picker groups models by where they run, and says for each whether it
works right now:

| group | what it is | needs |
|---|---|---|
| Hosted on OpenRouter | pay-per-token models; price shown, read live from OpenRouter | `OPENROUTER_API_KEY` in the environment, in `.env` in the openPASO folder, or in a file named by `OPENPASO_ENV_FILE` |
| Claude Code | the `claude` command on this machine, on your own Claude login | Claude Code installed and signed in; runs without asking only |
| On this machine | local models behind an OpenAI-compatible server (vLLM) | the server running on the listed port |

A model that cannot work is shown with the reason and cannot be picked.

## Steps

* **Run without asking**: openPASO runs tools, shell commands and solvers in
  the run's folder.
* **Ask before each step**: every tool call waits for *Run this step* or
  *Skip it*. Not available with Claude Code.

## Runs

* A run lives on the server, not in the browser tab. Close the tab, go back,
  open another run: it keeps working. Up to four runs work at once
  (`OPENPASO_MAX_RUNNING`).
* **Stop** ends the run and every process it started (solvers included), and
  says how many.
* A message sent while a run works is a **correction**: it is handed to the
  model when the current step finishes. A message after it ends is a
  **follow-up** in the same conversation. Claude Code runs as one command that
  cannot be spoken to while it works, so a correction to it is sent the moment
  the turn ends; the interface says so where you type it.
* **Attach files** (geometry, meshes, input decks, data) puts them in the run's
  `uploads/` folder and tells the model where they are.
* **Files** shows the run's own folder; **Download the run record** gives the
  prompt, model, every event and a checksum for every file.
* Home directories are removed from everything the browser is shown, and from
  text files it downloads. A binary file (mesh, HDF5, image, PDF) is handed over
  exactly as the run wrote it, so a path can still be inside one.
* A run starts in its own folder, but its shell and file tools can reach
  anything your account can. Deleting a run removes its folder, not files it
  wrote elsewhere. Run openPASO as a separate user or in a container if that
  matters to you.
* A turn ends as *Finished* only when an openPASO solver ran and openPASO
  verified its result. A solver result openPASO did not verify ends as *Ran,
  not verified*; a turn without a solver result says what happened instead
  (no tools used, a solver call that computed nothing, or numbers from scripts
  the model ran itself).
* Runs are listed in the left panel (which can be hidden) and can be deleted,
  one or several at a time; deleting removes the record and the run's folder.

Records are JSON files in `data/webui_sessions/`; run folders are in
`eval_interactive/webui_<id>/`.

## Tests

The fake model that answers without a model and runs no solver exists only for
these tests. The server refuses to create a run with it unless it was started
with `OPENPASO_TEST_MODEL=1`, so nothing on a normal machine can produce a
fabricated run through the API.

```bash
# fast, no model and nothing leaving this machine: config, file safety,
# uploads, outcome rules, Stop ending processes, corrections, a run that
# outlives its tab. (It does ask localhost whether a model server is
# listening, which answers at once either way.)
.venv-lg/bin/pytest webui/tests/test_app.py -q

# live, with a hosted model (a few cents) and the server running
.venv-lg/bin/python webui/tests/live_flows.py        # survive tab close, parallel, correction, memory, Stop, plan
.venv-lg/bin/python webui/tests/live_concurrency.py  # several runs starting at once
.venv/bin/python webui/tests/browser_journey.py      # a person's journey in a real browser, with screenshots
```

## Layout

| path | what |
|---|---|
| `app.py` | HTTP endpoints and the WebSocket a tab uses to follow a run |
| `runs.py` | a run: its turns, corrections, Stop, and its openPASO connection |
| `runner.py` | the agent: model, tools, the approval gate, streaming |
| `claude_code.py` | the Claude Code path |
| `catalog.py` | what can be selected: models with status and price, installed solvers |
| `proctree.py` | finding and ending a run's processes |
| `outcome.py` | how a turn ended, decided in one place |
| `ui/src/` | the interface (React, TypeScript, Tailwind) |
