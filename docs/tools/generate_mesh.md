# `generate_mesh`

**Builds a mesh with Gmsh: a rectangle, an L-shape, a plate with a hole, a channel, or your own shape.**

Group: Run.

## Parameters

| Parameter | Type | Required | Default |
|---|---|---|---|
| `geometry` | string | yes |  |
| `mesh_size` | number | no | `0.1` |
| `output_dir` | string | no | `''` |
| `params` | string | no | `''` |

## What the model reads

The text below is the tool's own description, exactly as the AI model receives it.

??? note "Show the full description"

    ```text
    Generate a mesh using Gmsh for non-trivial geometries.
    
    THE SHAPE IS YOURS TO SET. Each geometry below has DIMENSIONS, and
    `params` sets them. Without it you get this server's defaults, which
    are one particular published configuration and almost certainly not
    the one in your problem -- so the reply always states the dimensions
    it actually built, whether you passed any or not.
    
    Args:
        geometry: One of the built-in geometries:
            - "l_domain"          — 2D L-shaped domain. FIXED at
              [-1,1]^2 minus [0,1]x[-1,0] (it matches deal.II's
              hyper_L(-1,1)); it takes no shape parameters, and asking
              for some is refused rather than ignored.
            - "plate_with_hole"   — 2D plate with circular hole.
              params: radius, width, height.
            - "channel_cylinder"  — 2D channel with cylindrical
              obstacle. params: cyl_radius, cyl_center ([x, y]),
              channel_length, channel_height.
            (No "custom" passthrough yet — passing any other name
            returns a 'Unknown geometry' message with this list.)
        mesh_size: Target element size
        output_dir: Where to save (auto if empty)
        params: JSON object of the shape parameters above, e.g.
            {"channel_length": 5.0, "channel_height": 1.0,
             "cyl_center": [1.0, 0.5], "cyl_radius": 0.1}. A key this
            geometry does not take is REFUSED with the list of the ones
            it does -- never dropped, because a dropped key hands you the
            default geometry while you believe you asked for another.
    ```
