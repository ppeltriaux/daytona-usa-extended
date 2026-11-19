"""Offline tests of the Course Direction screen's arrows, frame and captions (dtools/arcade/dirgfx.py; M9 stage 3,
plan docs/superpowers/plans/2026-11-18-reverse-stage3-direction.md, Task 2). Need DAYTONA_ROMDIR and DAYTONA_DC."""

import numpy as np

from dtools.arcade import dirgfx as G
from dtools.arcade import select as S
from tests.test_dirscreen import _release
from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)


def test_arrow_senses_mirror():
    """The anticlockwise arrow is the clockwise one mirrored; the arrow is inside its area, the frame on its edge."""
    assert np.array_equal(G.arrow_pens(False), G.arrow_pens(True)[:, ::-1])
    framed, plain = G.area_pens(True, True), G.area_pens(True, False)
    assert framed.shape == (8 * G.AREA_ROWS, 8 * G.AREA_COLS)
    assert set(np.unique(framed[0])) == {2} and set(np.unique(framed[1, 1:-1])) == {3}
    assert not plain[0].any() and not plain[:, 0].any()


def test_side_art_blocks(romset_zip, disc):  # noqa: F811
    """Each side's chars fill at most its block from CAPTION_FIRST; the area maps' tiles have the priority bit (over
    the polygons) and name the side's chars; the caption's map is the same for either arrow."""
    rs = _release(romset_zip, disc)
    for side in (0, 1):
        arts = [G.side_art(rs, side, cw) for cw in (True, False)]
        assert arts[0].caption.tiles == arts[1].caption.tiles
        for a in arts:
            assert len(a.chars) <= G.BLOCK
            lo, hi = G.CAPTION_FIRST[side], G.CAPTION_FIRST[side] + len(a.chars)
            for m in (a.plain, a.framed):
                assert all(t & G.PRIORITY for t in m.tiles)
                assert all(lo <= t & ~G.PRIORITY < hi for t in m.tiles)
            assert all(lo <= t < hi for t in a.caption.tiles)


def test_course_senses(romset_zip, disc):  # noqa: F811
    """The laps' sense on their select maps, as the screen's NORMAL arrows show them (seen in MAME, 18 Nov 2026):
    Three Seven Speedway, Circuit Pixie, Rin Rin Rink and Mermaid Lake clockwise, the other four anticlockwise."""
    rs = _release(romset_zip, disc)
    assert [G.course_clockwise(rs, e.course) for e in S.ENTRIES] == [True, False, False, False, False, True, True, True]
