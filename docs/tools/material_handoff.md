# `material_handoff`

**Checks material data from another framework, such as elastic constants from quantum-mechanical calculations, and passes it to the continuum model only when every calculation it rests on is complete.**

Group: Data from other frameworks.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `action` | string | no | `'schema'` |
| `card` | string | no | `''` |
| `theta_deg` | number | no | `0.0` |
| `provider` | string | no | `''` |
| `request` | string | no | `''` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Use material data from another framework (e.g. elastic constants from quantum-mechanical calculations)
            in a continuum model, but only once the calculations it rests on are complete.
    
            The card holds the values and every calculation behind them (call action='schema' for the layout). From
            the producer's OWN settings the rule derives which convergence calculations are required; a missing one
            counts as failed, never as passed.
    
            Args:
                action: 'schema' (card layout and rules), 'check' (accept / reject / incomplete, with the measured
                    changes), 'plan' (the calculations still to run, with their settings and estimated core-hours),
                    'continuum' (stiffness for the continuum model; refused unless the check accepts),
                    'providers' (installed upstream frameworks), 'produce' (ask a provider for a new card),
                    'complete' (ask a provider to run the planned calculations, then check again).
                card: path to a card file, or the card's JSON text (check, plan, continuum, complete).
                theta_deg: rotation of the crystal axes about z in the sheet plane (continuum).
                provider: provider name (produce, complete).
                request: JSON request for the provider (produce), e.g. {"material": "Al", "kind": "elastic.cubic"}.
    ```
