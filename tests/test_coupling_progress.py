"""The coupling driver leaves one line per iteration beside its participants, for a live view."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.coupling_driver import PROGRESS_FILE, Participant, _progress_writer  # noqa: E402


def test_each_iteration_is_one_line_in_the_participants_common_folder(tmp_path):
    ps = [Participant("fluid", ["x"], tmp_path / "side_fluid"), Participant("solid", ["x"], tmp_path / "side_solid")]
    write = _progress_writer(ps, 1e-6)
    write(1, float("nan"))
    write(2, 0.25)
    rows = [json.loads(s) for s in (tmp_path / PROGRESS_FILE).read_text().splitlines()]
    assert [(r["iteration"], r["residual"]) for r in rows] == [(1, None), (2, 0.25)]
    assert rows[0]["run"] == rows[1]["run"] and rows[0]["participants"] == ["fluid", "solid"]
    assert rows[0]["tol"] == 1e-6


def test_a_folder_that_cannot_be_written_never_stops_the_coupling(tmp_path):
    ps = [Participant("a", ["x"], tmp_path / "gone" / "a"), Participant("b", ["x"], tmp_path / "gone" / "b")]
    write = _progress_writer(ps, 1e-6)
    write(2, 0.1)                       # the folder does not exist: nothing written, nothing raised
    assert not (tmp_path / "gone").exists()
