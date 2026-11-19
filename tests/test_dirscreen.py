"""Offline tests of the Course Direction screen (M9 stage 3; docs/superpowers/plans/2026-11-18-reverse-stage3-direction.md):
the patch's words on the release set. Need DAYTONA_ROMDIR and DAYTONA_DC."""

import struct

import pytest

from dtools.arcade import course as C
from dtools.arcade import dirscreen as D
from dtools.arcade import select as S
from dtools.arcade.romset import RomSet
from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)

_SETS: dict = {}


def _release(romset_zip, disc):  # noqa: F811
    key = str(romset_zip)
    if key not in _SETS:
        from dtools.release import release_set
        _SETS[key] = release_set(RomSet.from_zip(romset_zip), disc, direction=False)
    return _SETS[key]


def _words(rs, at, n):
    return struct.unpack(f"<{n}I", rs.read_i960(at, 4 * n))


def test_patch_direction_words(romset_zip, disc):  # noqa: F811
    """Both dispatchers read their new state tables (D.TABLE_AT, D.LINKED_TABLE_AT: each its stock 17 handlers but
    state 7's, which is D's wrapper, then states 17-19), and nothing else of the set changes outside the direction
    code (D.CODE_AT to the end of the linked table), the two dispatchers' words and the graphics data (D.DATA_AT to
    D.graphics' end)."""
    rs = _release(romset_zip, disc)
    out = D.patch_direction(rs)
    stock = _words(rs, D.STOCK_TABLE, D.STOCK_STATES)
    assert _words(out, D.DISPATCH_WORD, 1) == (D.TABLE_AT,)
    table = _words(out, D.TABLE_AT, D.STATES)
    blk = D.code()
    assert table[:7] == stock[:7] and table[8:17] == stock[8:17]
    assert table[7] == blk.labels["dir7"] and table[17:] == tuple(blk.labels[f"dir{k}"] for k in (17, 18, 19))
    linked, stock_linked = _words(out, D.LINKED_TABLE_AT, D.STATES), _words(rs, D.LINKED_TABLE, D.STOCK_STATES)
    assert _words(out, D.LINKED_WORD, 1) == (D.LINKED_TABLE_AT,)
    assert linked[:7] == stock_linked[:7] and linked[8:17] == stock_linked[8:17] and linked[7] == table[7]
    assert linked[17:] == table[17:]
    end = D.LINKED_TABLE_AT + 4 * D.STATES
    for region in ("program", "data", "copro_data", "polygons", "textures"):
        a, b = getattr(rs, region), getattr(out, region)
        diff = [i for i in range(0, min(len(a), len(b)), 4096) if a[i:i + 4096] != b[i:i + 4096]]
        if region == "program":
            changed = {i for blk_at in diff for i in range(blk_at, blk_at + 4096) if a[i] != b[i]}
            word = C.rom_word_at(D.DISPATCH_WORD)[1]                # the program-window word's image offset
            allowed = set(range(C.rom_word_at(D.CODE_AT)[1], C.rom_word_at(D.CODE_AT)[1] + end - D.CODE_AT))
            linked_word = C.rom_word_at(D.LINKED_WORD)[1]
            allowed |= set(range(word, word + 4)) | set(range(linked_word, linked_word + 4))
            assert changed <= allowed, sorted(changed - allowed)[:8]
        elif region == "data":
            lay, _ = D.graphics(rs)
            changed = {i for blk_at in diff for i in range(blk_at, blk_at + 4096) if a[i] != b[i]}
            lo = D.DATA_AT - C.DATA_BASE
            assert changed <= set(range(lo, lay.end - C.DATA_BASE)), sorted(changed)[:8]
        else:
            assert diff == [], region


def test_patch_direction_code_disassembles(romset_zip, disc):  # noqa: F811
    """The direction code as unidasm reads it on the patched set: every word decodes (no `?`), the three state
    handlers and the wrapper begin where the table names them."""
    from dtools.arcade.disasm import listing
    out = D.patch_direction(_release(romset_zip, disc))
    blk = D.code()
    lines = listing(out, D.CODE_AT, len(blk.bytes()) // 4)
    assert all(not ln.text.startswith("?") for ln in lines if ln.addr < blk.labels["table"]), \
        [ln for ln in lines if ln.text.startswith("?")][:3]


def test_patch_direction_refusals(romset_zip, disc):  # noqa: F811
    """Refused on a set without the select patch or widen_ids, and on a set already patched."""
    stock = RomSet.from_zip(romset_zip)
    with pytest.raises(D.DirectionError, match="select"):
        D.patch_direction(C.widen_ids(C.guard_new_ids(C.grow_ids(stock))))
    out = D.patch_direction(_release(romset_zip, disc))
    with pytest.raises(D.DirectionError, match="already"):
        D.patch_direction(out)
    assert S.patched(out) and D.patched(out)
