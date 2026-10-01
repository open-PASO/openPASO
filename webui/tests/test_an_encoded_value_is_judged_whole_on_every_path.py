"""An encoded value is judged whole, by one rule, on every path a file or an answer takes.

The scrubber passes the values of "frames", "mask", "data" and "image" on untouched when they are
encoded data, so that a field's numbers reach the page as the run wrote them. Copilot on the org PR
found such a value judged by its start: first in the object walk, then in the streamed download,
which looked at the first 64 characters and copied the rest unread. Measured before the fix: a value
of 64 letters, a space and the home directory, and a long path a run wrote under "data", went out as
they were in the streamed download AND in the whole-file download; and where the stream cut a piece
inside a path, the rest of the user name went out after "~". Copilot's next round: with "=" allowed
anywhere, 40 letters, "=" and the home directory passed as encoded data, and so do 40 letters glued
to the home directory, which IS valid base64: the form cannot settle it, the home check does.
"""
from __future__ import annotations

import base64
import getpass
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pytest
from fastapi.testclient import TestClient

from webui import app as webui_app
from webui import config
from webui.privacy import scrub, scrub_text

HOME, USER = str(Path.home()), getpass.getuser()
PAD = "the run wrote a long note here. " * 300_000          # about 9.6 MB: over the whole-file limit
TEXT_VALUES = {
    "a value that only starts encoded": "A" * 64 + f" {HOME}/secret",
    "a long path a run wrote": f"{HOME}/projects/openpaso/eval/interactive/webui/work/results.csv",
    "a path in base64's own characters": f"{HOME}/projects/openpaso/runs/work/output/fields/velocity",
    "padding before the home directory": "A" * 40 + "=" + f"{HOME}/secret",
    "letters glued to the home directory": "A" * 40 + f"{HOME}/secret",
    "the home directory as a valid block": "A" * 38 + "==" + f"{HOME}/secret" + "A" * (-len(f"{HOME}/secret") % 4),
}


@pytest.fixture
def served():
    """GET a file the run wrote, through the download route, and remove it afterwards."""
    client = TestClient(webui_app.app)
    work = config.SANDBOX_ROOT / "webui_abc123def" / "work"
    work.mkdir(parents=True, exist_ok=True)
    made = []

    def get(name: str, text: str) -> str:
        p = work / name
        p.write_text(text)
        made.append(p)
        r = client.get(f"/sandbox-file/webui_abc123def/work/{name}")
        assert r.status_code == 200
        return r.text
    yield get
    for p in made:
        p.unlink(missing_ok=True)


@pytest.mark.parametrize("what", list(TEXT_VALUES))
def test_text_under_an_encoded_key_is_scrubbed_on_every_path(served, what):
    value = TEXT_VALUES[what]
    streamed = served("streamed.json", json.dumps({"note": PAD, "data": value}))
    assert len(streamed) > webui_app._SCRUB_WHOLE, "this is the streamed path"
    whole = served("whole.json", json.dumps({"data": value}))
    for path, out in (("streamed download", streamed), ("whole-file download", whole),
                      ("document", scrub_text(json.dumps({"data": value}))),
                      ("object walk", json.dumps(scrub({"data": value})))):
        leaked = HOME in out                    # a bool first: pytest would diff megabytes
        assert not leaked, f"{what}: the {path} handed out the home directory"


def test_encoded_data_comes_back_byte_for_byte_on_every_path(served):
    """The other half: real frames, with a name and a home path occurring in them by chance (the
    cases the earlier fixes were for), are left exactly as written on every path."""
    frames = base64.b64encode(os.urandom(9_000_000)).decode()           # over the limit on its own
    name = USER if re.fullmatch(r"[A-Za-z0-9]{3,}", USER) else "someone"
    # what chance can put into real frames, in place so that the value stays base64: a second block
    # that begins with a home path after the first block's padding, and a name between "+" and "/"
    first = base64.b64encode(os.urandom(1001)).decode()               # ends in "="
    inner = "/home/" + "someone/x"
    planted = (first + inner + frames[len(inner):200] + "+" + name + "/"
               + frames[200 + len(name) + 2:])
    doc = json.dumps({"kind": "field_series", "note": f"written in {HOME}/run",
                      "frames": planted, "mask": frames[:4000]})
    streamed = json.loads(served("field.json", doc))
    intact = streamed["frames"] == planted and streamed["mask"] == frames[:4000]
    assert intact, "the streamed download changed the frames"
    assert HOME not in streamed["note"] and streamed["note"].endswith("~/run")
    small = {"note": f"written in {HOME}/run", "frames": planted[:6000], "mask": frames[:4000]}
    for path, out in (("whole-file download", json.loads(served("small.json", json.dumps(small)))),
                      ("document", json.loads(scrub_text(json.dumps(small)))),
                      ("object walk", scrub(small))):
        intact = out["frames"] == small["frames"] and out["mask"] == small["mask"]
        assert intact, f"the {path} changed the frames"
        assert HOME not in out["note"], path


def test_a_path_across_the_end_of_a_piece_is_scrubbed_whole(served):
    """The stream hands out prose up to a point shortly before the end of what it has read. At a
    fixed position that point fell inside a path, and the two halves were judged apart: "/home/"
    with the first letters of the name became "~", and the rest of the name went out after it."""
    if len(USER) < 4 or not HOME.endswith("/" + USER) or not USER.isalpha():
        pytest.skip("needs a home directory ending in an alphabetic user name of four letters or more")
    cut = getattr(webui_app, "_PIECE", 1024 * 1024) - webui_app._CARRY   # where a fixed cut falls
    n = cut - (len(HOME) - len(USER) + 2)          # so that it falls two letters into the name
    doc = ("7 " * (n // 2 + 1))[-n:] + f"{HOME}/secret " + "7 " * 4_200_000
    assert doc.index(HOME) == n
    out = served("long.log", doc)
    assert len(out) > webui_app._SCRUB_WHOLE, "this is the streamed path"
    rest_out, whole = USER[2:] in out, " ~/secret " in out
    assert not rest_out, "the rest of the user name went out"
    assert whole, "the path was not scrubbed whole"


def test_the_rule_itself():
    from webui.privacy import is_encoded
    frames = base64.b64encode(os.urandom(301)).decode()            # ends in "=="
    assert is_encoded(frames) and is_encoded(frames + frames) and is_encoded(frames.rstrip("="))
    for value in TEXT_VALUES.values():
        assert not is_encoded(value), value
    assert not is_encoded("QUJD" * 9), "shorter than 40 characters"
    assert not is_encoded(frames[:100] + " " + frames[100:]), "a space is not base64"
    assert not is_encoded("QUJD" * 10 + "A"), "a last group of one character encodes nothing"
    assert not is_encoded("QUJD" * 10 + "Q=JD"), "padding stands only at the end of a group"
    assert not is_encoded("QUJD" * 10 + "Q==="), "at most two padding characters"
    assert is_encoded("QUJD" * 10 + "QUI="), "one padded block"


def test_a_data_uri_is_judged_by_the_same_rule(served):
    """A data: URI's payload was protected as far as base64's characters reach, so a home path written
    straight after it went out with it."""
    payload = base64.b64encode(os.urandom(3000)).decode()
    ok = f'<img src="data:image/png;base64,{payload}">'
    bad = f'<img src="data:image/png;base64,{payload}{HOME}/secret">'
    for path, out in (("document", scrub_text(bad)), ("whole-file download", served("page.html", bad)),
                      ("streamed download", served("big.html", PAD + bad))):
        leaked = HOME in out
        assert not leaked, f"the {path} handed out the home directory after a data: URI"
    kept = served("ok.html", ok) == ok and scrub_text(ok) == ok
    assert kept, "a real data: URI must come back byte for byte"
