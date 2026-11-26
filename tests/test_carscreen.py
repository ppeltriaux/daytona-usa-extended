"""The Car Select screen's code and tables (v0.4 stage 2; dtools/arcade/carscreen.py)."""
import struct

import pytest

from dtools.arcade import carscreen as K, course as C, dirscreen as D, romset as R


def test_constants():
    assert any((lo, n) == (K.CODE_AT, K.CODE_SIZE) for lo, n, _ in R.RESERVED["program"])
    assert any((C.DATA_BASE + lo, n) == (K.DATA_AT, K.DATA_SPAN) for lo, n, _ in R.RESERVED["data"])
    assert C.CAR_CODE + C.CAR_SIZE <= K.CODE_AT and K.CODE_AT + K.CODE_SIZE <= C.FAR_CODE and (K.STATE_SETUP, K.STATE_RUN, K.STATE_HOLD) == (20, 21, 22)
    assert K.CHOICE == C.CAR_CHOICE_RAM and K.ANGLE + 2 <= C.CAR_CHOICE_RAM + 4
    assert K.BANDS == tuple(sorted(K.BANDS)) and len(K.BANDS) == K.CARS - 1 and K.BANDS[-1] < 127
    assert len(K.NAME_DST) == len(K.CAR_X) == K.CARS and len(set(K.NAME_DST)) == K.CARS
    assert K.HORNET_PARTS[0] == C.CAR_PARTS_LIST


def test_code_fits():
    """The code and its two tables fit CODE_SIZE; its length does not depend on the layout; the tables' labels lie
    at its end, 4 * STATES apart."""
    blk = K.code()
    code = blk.bytes()
    assert len(code) <= K.CODE_SIZE and K.LINKED_TABLE_AT - K.TABLE_AT == 4 * K.STATES
    assert K.TABLE_AT + 8 * K.STATES == K.CODE_AT + len(code)
    other = K.code(K.Layout(*(K.DATA_AT + 0x1234,) * 9))
    assert len(other.bytes()) == len(code) and other.labels == blk.labels
    for name in ("car7", "car19", "car20", "car21", "car22", "car_draw", "car_names"):
        assert K.CODE_AT <= blk.labels[name] < K.TABLE_AT


def test_tables_synthetic():
    """_tables: each of the direction screen's tables with states 7 and 19 wrapped and 20-22 added; a table that is
    not the direction screen's is refused."""
    class Fake:
        def __init__(self, tables):
            self.tables = tables

        def read_i960(self, at, n):
            return self.tables[at][:n]

    blk = K.code()
    stock = list(range(0x100, 0x100 + 4 * D.STATES, 4))
    stock[7], stock[D.STATE_HOLD] = D._BLOCK.labels["dir7"], D._BLOCK.labels["dir19"]
    fake = Fake({D.TABLE_AT: struct.pack(f"<{D.STATES}I", *stock),
                 D.LINKED_TABLE_AT: struct.pack(f"<{D.STATES}I", *(v + 0x1000 if k not in (7, D.STATE_HOLD) else v
                                                                   for k, v in enumerate(stock)))})
    out = K._tables(fake, blk)
    assert len(out) == 8 * K.STATES
    for half, base in ((out[:4 * K.STATES], 0), (out[4 * K.STATES:], 0x1000)):
        t = struct.unpack(f"<{K.STATES}I", half)
        assert t[7] == blk.labels["car7"] and t[19] == blk.labels["car19"]
        assert t[20:] == (blk.labels["car20"], blk.labels["car21"], blk.labels["car22"])
        assert all(t[k] == stock[k] + base for k in range(D.STATES) if k not in (7, 19))
    stock[7] = 0x1234
    with pytest.raises(K.CarScreenError, match="direction"):
        K._tables(Fake({D.TABLE_AT: struct.pack(f"<{D.STATES}I", *stock)}), blk)


def test_patch_refusals():
    from tests.test_convert_build import _romset
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    with pytest.raises(K.CarScreenError, match="direction screen"):
        K.patch_cars(rs)
    with pytest.raises(K.CarScreenError, match="RomSet"):
        K.patch_cars("rs")
