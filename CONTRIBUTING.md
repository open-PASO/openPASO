# Contributing to openPASO

Thank you for your interest. openPASO is a community project under active development, and it gets
better every time someone reports what went wrong.

**Development happens in the development repository, <https://github.com/alhermann/openPASO>.** It carries
the test suite, the fixtures that back each served claim, and the measurement tooling. This
repository is the released product: what installing and running openPASO needs. Please open issues
here or there, and pull requests there.

The browser interface's fast tests ship with it and need no key:

```bash
pytest webui/tests/test_app.py -q
```

So do the knowledge gates that the weekly drift check runs:

```bash
pip install -e ".[skfem,ngsolve,test]" && pytest tests -q
```

The full guide, in plain language, is on the website: <https://open-paso.github.io/openPASO/contribute/>.

The rules a change has to meet:

- **Every improvement helps all simulations, not one example.** Templates use placeholders, never
  the dimensions of a specific case.
- Every new or changed parameter key or keyword is cited to the solver's own source, file and line.
- The pull request says what changed, which failure it catches, and how it was checked.
