# `examples`

**Finds real input files from the solvers' own test suites to start from.**

Group: Find out.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `keyword` | string | no | `''` |
| `solver` | string | no | `'fourc'` |
| `action` | string | no | `'search'` |
| `max_results` | integer | no | `3` |
| `variant` | string | no | `''` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Find and retrieve example input files from solver test suites.
    
    IMPORTANT: Always call this before writing new input files to study
    real, validated configurations.
    
    Args:
        keyword: Search term (e.g. 'peridynamic', 'fsi', 'poisson', 'heat')
        solver: Backend name (default: 'fourc')
        action: What to do. Options:
            - "search" — find matching test files with content preview
            - "template" — get a generated template for this physics
            - "tutorials" — list available tutorials
        max_results: Maximum results (default 3)
        variant: For action="template": which of the physics' template
            variants to generate (e.g. '2d', '3d'). action="tutorials"
            lists them. With a variant and no keyword, the physics that
            has that variant is used.
    ```
