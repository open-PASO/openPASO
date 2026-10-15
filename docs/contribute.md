# Contribute

!!! success "openPASO is a community project, and it is under active development"
    It is young, it changes quickly, and it gets better every time someone tells us what went wrong.
    **You do not need to be a programmer or a simulation expert to help.** If you tried it, your
    experience is already useful.

## Ways to help

| You can | How |
|---|---|
| **Report a problem** | Something did not install, a run failed, a page here was unclear: [open an issue](#report-a-problem) |
| **Share a solver trap** | You know a setting that silently gives a wrong answer, or an error message that means something other than it says. This is the most valuable knowledge openPASO has |
| **Improve these pages** | Every page has an edit button (the pencil at the top right). Plain-language fixes are very welcome |
| **Add a solver or a physics** | A new backend, element list, template or coupling participant |
| **Test on your system** | Windows and macOS are the least tested. A report that it works is useful too |

## Report a problem

Open an issue on GitHub and include:

1. what you ran (the command, or what you asked the AI);
2. what you expected, and what happened instead;
3. the output of `python check_install.py`.

Leave out API keys, and paths or file contents you do not want to share.

## The one rule for changes

**Every improvement must help all simulations, not one example.**

- **Welcome:** solver traps, element lists, new solvers, new coupling participants, clearer error
  messages, better documentation.
- **Not welcome:** numbers tuned for one benchmark, or templates built around one particular problem.
  Templates use placeholders, never the dimensions of a specific case.

A changed parameter key or keyword must be **cited to the solver's own source**, a file and line,
because one wrong key in the catalogue produces silently broken input for everyone who uses it.
Please say in the pull request which failure your change catches and how you checked it.

## Where development happens

Please open issues and pull requests in <https://github.com/open-PASO/openPASO>, which carries the
released product. The full test suite, the fixtures and the measurement tooling live in a development
repository that is not public.

## Regenerating the reference pages

The [Tools](tools/index.md) and [Solvers](solvers/index.md) pages are generated from the code:

```bash
python docs/_gen_reference.py
```

To preview this website locally: `pip install mkdocs-material`, then `mkdocs serve`.
