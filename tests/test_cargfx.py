"""The Car Select screen's tile art (v0.4 stage 2; dtools/arcade/cargfx.py)."""
import numpy as np
import pytest

from dtools.arcade import cargfx, dirgfx as G
from tests.test_convert_build import _romset


def _rs():
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    return rs


def test_banner_pens():
    """The "Car Select" banner: 48 rows, a multiple of 8 columns, the banners' pens only, narrower than Course
    Direction's and not it; its chars fit the two blocks from CHAR_FIRST."""
    rs = _rs()
    b, d = cargfx.banner_pens(rs), G.banner_pens(rs)
    assert b.shape[0] == 48 and b.shape[1] % 8 == 0 and b.shape[1] < d.shape[1]
    assert set(np.unique(b)) <= set(np.unique(G.render_map(rs, G.CIRCUIT_BANNER)[1])) | set(np.unique(
        G.render_map(rs, G.MISSION_BANNER)[1]))
    assert len(G.tiled(b, cargfx.CHAR_FIRST).chars) <= 2 * G.BLOCK
    assert (b[:, :G.FRAME_ENDS] == d[:, :G.FRAME_ENDS]).all()            # the same frame end


def test_name_pens():
    """The five names: 8 rows by NAME_COLS tiles, the fill and outline pens only, centred on a tile boundary so the
    letters' chars are shared: all five maps' chars fit one block."""
    rs = _rs()
    assert cargfx.NAMES == ("HORNET", "GRASSHOPPER", "FALCON", "LIGHTNING", "UNICORN")
    pens = [cargfx.name_pens(rs, k) for k in range(5)]
    for k, p in enumerate(pens):
        assert p.shape == (8, 8 * cargfx.NAME_COLS) and set(np.unique(p)) <= {0, 2, cargfx.NAME_FILL}
        cols = np.nonzero(p.any(0))[0]
        assert cols[-1] - cols[0] + 1 <= 8 * len(cargfx.NAMES[k]) and cols[0] // 8 * 8 % 8 == 0
    assert len(G.tiled(np.concatenate(pens, 0), 0).chars) <= G.BLOCK
    assert cargfx.name_palette(True)[cargfx.NAME_FILL] == G.YELLOW and cargfx.name_palette(False)[cargfx.NAME_FILL] == G.WHITE
    for bad in (-1, 5, True, "0"):
        with pytest.raises(cargfx.CargfxError):
            cargfx.name_pens(rs, bad)


def test_bars():
    """The bars: a count of BAR_SEGMENTS per label from a value in a range (at least one); two rows of label, gap and
    squares, the first `filled` full; refused: counts that are not one per label within 0..BAR_SEGMENTS."""
    rs = _rs()
    assert cargfx.BAR_LABELS == ("SPEED", "ACCEL") and len(cargfx.TOP_SPEED) == len(cargfx.NAMES)
    assert [cargfx.segments(v, *cargfx.SPEED_RANGE) for v in cargfx.TOP_SPEED] == [5, 3, 4, 6, 8]
    assert cargfx.segments(0.0, 1.0, 2.0) == 1 and cargfx.segments(9.0, 1.0, 2.0) == cargfx.BAR_SEGMENTS
    p = cargfx.bars_pens(rs, (3, 8))
    assert p.shape == (24, 8 * cargfx.BAR_COLS) and set(np.unique(p)) <= {0, 2, cargfx.NAME_FILL} and not p[8:16].any()
    squares = lambda row: [int((p[row:row + 8, 48 + 8 * k:56 + 8 * k] == cargfx.NAME_FILL).any()) for k in range(8)]  # noqa: E731
    assert squares(0) == [1, 1, 1, 0, 0, 0, 0, 0] and squares(16) == [1] * 8
    for bad in ((3,), (3, 9), (True, 2), "38"):
        with pytest.raises(cargfx.CargfxError):
            cargfx.bars_pens(rs, bad)
