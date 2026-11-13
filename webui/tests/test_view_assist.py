"""The picture assistant's answer is applied only where the file can take it."""
from webui.view_assist import validate

FIELDS = [{"name": "pressure", "ncomp": 1}, {"name": "displacement", "ncomp": 3}, {"name": "u2d", "ncomp": 2}]


def test_settings_the_file_can_take_pass_through():
    got, dropped = validate({"field": "displacement", "part": "z", "range": {"auto": False, "min": 0, "max": 0.02},
                             "warp_field": "displacement", "warp_factor": 10, "edges": True, "step": 2,
                             "same_range": True, "view": "front", "scale": "sequential"}, FIELDS, steps=3)
    assert dropped == []
    assert got["field"] == "displacement" and got["part"] == "z" and got["warp_factor"] == 10
    assert got["range"] == {"auto": False, "min": 0.0, "max": 0.02} and got["step"] == 2


def test_invented_or_impossible_settings_are_dropped_and_named():
    got, dropped = validate({"field": "vorticity", "part": "x", "range": {"min": 5, "max": 1},
                             "warp_field": "pressure", "warp_factor": -3, "step": 7, "view": "inside"},
                            FIELDS, steps=3)
    assert got == {}
    assert len(dropped) == 6                       # every rejected setting is said, the unknown view ignored


def test_a_value_has_no_parts_and_a_flat_vector_no_z():
    assert validate({"field": "pressure", "part": "x"}, FIELDS, 1)[0] == {"field": "pressure"}
    assert validate({"field": "u2d", "part": "z"}, FIELDS, 1)[0] == {"field": "u2d"}
    assert validate({"field": "u2d", "part": "y"}, FIELDS, 1)[0] == {"field": "u2d", "part": "y"}
