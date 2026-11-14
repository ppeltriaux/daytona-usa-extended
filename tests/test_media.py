"""Media helpers for the public page (Phase 13 Task 3): MAME AVI captures to clips, stills, posters, the share card and
the hub card (dtools/mame/media.py, ported from virtua-racing's vrtools/mame/media.py), and the autopilot's road file
(dtools/mame/road.py, moved from tests/test_courses_mame.py)."""
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                                  reason="ffmpeg and ffprobe are needed")

NATIVE = (496, 384)
"""MAME `daytona`'s frame as `-aviwrite` records it: measured on the Phase 13 captures (ffprobe of out/site/raw/*.avi:
rawvideo 496x384, 57.524 fps, pcm_s16le 48 kHz stereo), the same as the snapshots the select tests crop."""


def _synthetic_avi(path: Path, seconds: float = 12.0) -> Path:
    """What MAME writes: raw video 496x384 at 57.524 fps with 48 kHz stereo PCM audio."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=496x384:rate=57.52416",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-ac", "2", "-t", str(seconds),
                    "-c:v", "rawvideo", "-pix_fmt", "bgr24", "-c:a", "pcm_s16le", str(path)], check=True)
    return path


@needs_ffmpeg
def test_encode_clip_concatenates_cuts_at_2x_and_fits(tmp_path):
    from dtools.mame.media import encode_clip
    avi = _synthetic_avi(tmp_path / "in.avi")
    out = tmp_path / "clip.mp4"
    p = encode_clip(avi, out, [(1.0, 3.0), (6.0, 4.0)])
    assert p["codec"] == "h264" and p["pix_fmt"] == "yuv420p" and (p["width"], p["height"]) == (2 * NATIVE[0],
                                                                                                 2 * NATIVE[1])
    assert 6.8 <= p["duration"] <= 7.2 and p["audio"] == "aac" and p["size"] < 6_000_000
    assert p["color_space"] == "smpte170m"                  # tagged as the BT.601 matrix swscale used
    head = out.read_bytes()[:65536]
    assert b"moov" in head and (b"mdat" not in head or head.index(b"moov") < head.index(b"mdat"))


@needs_ffmpeg
def test_encode_clip_refuses_what_cannot_fit(tmp_path):
    from dtools.mame.media import MediaError, encode_clip
    avi = _synthetic_avi(tmp_path / "in.avi", seconds=4.0)
    with pytest.raises(MediaError):
        encode_clip(avi, tmp_path / "clip.mp4", [(0.0, 4.0)], max_bytes=20_000)


@needs_ffmpeg
def test_still_poster_card_and_crop_sizes(tmp_path):
    from PIL import Image
    from dtools.mame.media import card_crop, poster, share_card, still
    avi = _synthetic_avi(tmp_path / "in.avi", seconds=3.0)
    s = still(avi, 1.0, tmp_path / "s.png")
    assert Image.open(s).size == NATIVE
    p = poster(avi, 1.0, tmp_path / "p.png")
    assert Image.open(p).size == (2 * NATIVE[0], 2 * NATIVE[1])
    og = share_card(s, "DAYTONA USA EXTENDED", "The Dreamcast's courses in the Model 2 arcade game",
                    tmp_path / "og.jpg")
    im = Image.open(og)
    assert im.size == (1200, 630) and im.format == "JPEG" and og.stat().st_size < 1_000_000
    right = np.asarray(im.convert("RGB").crop((780, 0, 1200, 630)), dtype=int)
    green = int((np.abs(right - (0x2F, 0xD3, 0x6A)).max(axis=2) < 24).sum())
    assert green >= 800                                     # the accent rule, #2fd36a by default (140 x 8 px)
    c = card_crop(s, tmp_path / "card.png")
    assert Image.open(c).size == (320, 213)


def test_run_mame_passes_extra_args(tmp_path, monkeypatch):
    import dtools.mame.runner as runner
    seen = {}

    class Done:
        returncode = 0
        stdout = "Average speed: 100% (5 seconds)"
        stderr = ""

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return Done()
    monkeypatch.setattr(runner, "running_mame", lambda: [])
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.run_mame(Path("x.lua"), 5, tmp_path, tmp_path, extra_args=["-aviwrite", "x.avi"])
    assert seen["cmd"][-2:] == ["-aviwrite", "x.avi"]


MERMAID_LAKE = 7
"""Dreamcast course n 7 (Mermaid Lake, course id 7: the one gate the autopilot drives, test_mermaid_lake_id7)."""


def test_road_file_matches_the_test_helper(tmp_path, disc):  # noqa: F811
    """The moved road_file writes what tests/test_courses_mame.py's helper wrote: compared byte for byte against a copy
    of the helper for one run (Phase 13 Task 3, then the copy deleted); now its line count and its first and last
    lines on Mermaid Lake's road line from the disc (500 road points)."""
    from dtools.convert.tables import road_line
    from dtools.dc.data import decode_data
    from dtools.dc.disc import course_files
    from dtools.mame.road import road_file
    name = course_files(MERMAID_LAKE, kind="data")
    c = SimpleNamespace(road=road_line(decode_data(disc.read(name), name)))
    path = road_file(c, tmp_path / "road.txt")
    lines = path.read_text().split("\n")
    assert path == tmp_path / "road.txt" and lines[-1] == ""          # every line ends in a newline
    assert len(lines) - 1 == len(c.road) == 500
    assert lines[0] == "-501.5240 150.0000" and lines[-2] == "-509.2480 150.0000"
