"""Tests for the Dreamcast executable's checkpoint table (dtools/dc/exe.py)."""

import random
import struct

import numpy as np
import pytest

from dtools.dc import exe
from dtools.dc.disc import VARIANTS, DcDisc, course_files
from dtools.dc.errors import DcError
from dtools.dc.exe import CheckpointTable

# the real table's shape: 16 stored lists (n 0-7 plain, then reversed), entries 16-31 reuse 0-15
LISTS = ((), (150, 257, 404, 539, 730), (142, 275), (136, 263), (125, 257), (), (263,), (120, 363),
         (), (260, 451, 586, 733, 840), (211, 344), (201, 328), (168, 300), (), (236,), (136, 379))
ENTRIES = tuple(range(16)) * 2


def _exe(t: CheckpointTable, tail: bytes = b"\xAA" * 16) -> bytes:
    """A stand-in 1ST_READ.BIN: filler, then the encoded lists and pointers ending at CHECKPOINT_END, then `tail`."""
    blob = exe.encode_checkpoint_table(t)
    start = exe.CHECKPOINT_END - len(blob)
    return b"\x55" * start + blob + tail


def test_checkpoint_table_round_trip_synthetic():
    t = CheckpointTable(LISTS, ENTRIES)
    blob = exe.encode_checkpoint_table(t)
    assert len(blob) == 4 * sum(1 + len(c) for c in LISTS) + 4 * exe.ENTRIES
    start = exe.CHECKPOINT_END - len(blob)
    assert start == 0x17E8EC                                    # the disc's: the lists fill 0x17e8ec-0x17e99b
    b = _exe(t)
    assert exe.decode_checkpoint_table(b) == t
    # the layout: each list a u32 count then that many u32 segments, packed; then 32 u32 addresses
    assert struct.unpack_from("<3I", b, start) == (0, 5, 150)
    ptrs = struct.unpack_from("<32I", b, exe.CHECKPOINT_POINTERS)
    assert ptrs[0] == exe.EXE_BASE + start and ptrs[1] == exe.EXE_BASE + start + 4 and ptrs[16] == ptrs[0]
    for n in range(8):
        for v in VARIANTS:
            want = LISTS[n + 8 * (VARIANTS.index(v) % 2)]
            assert exe.checkpoints(b, n, v) == want
    assert exe.checkpoints(bytearray(b), np.int64(3)) == (136, 263)
    assert exe.checkpoints(memoryview(b), 1, "R") == LISTS[9]
    # a table whose lists are not shared, in another order
    t2 = CheckpointTable(((7,), (), (1, 2, 3)), (2, 0, 1) + (0,) * 29)
    b2 = _exe(t2)
    assert exe.decode_checkpoint_table(b2) == t2
    assert exe.checkpoints(b2, 0) == (1, 2, 3) and exe.checkpoints(b2, 1) == (7,) and exe.checkpoints(b2, 2) == ()


def test_checkpoint_table_accepts_numpy_numbers():
    t = CheckpointTable(tuple(np.array(c, dtype=np.uint32) if c else () for c in LISTS),
                        np.array(ENTRIES, dtype=np.int64))
    assert exe.encode_checkpoint_table(t) == exe.encode_checkpoint_table(CheckpointTable(LISTS, ENTRIES))


def _refuses(t, match):
    with pytest.raises(DcError, match=match):
        exe.encode_checkpoint_table(t)


def test_checkpoint_table_refuses_encode():
    _refuses(None, "not a CheckpointTable")
    _refuses(CheckpointTable(LISTS, ENTRIES[:31]), "32 items")
    _refuses(CheckpointTable(LISTS, ENTRIES[:31] + (16,)), "entry 31.*outside 0..15")
    _refuses(CheckpointTable(LISTS, ENTRIES[:31] + (True,)), "entry 31")
    _refuses(CheckpointTable(LISTS, ENTRIES[:31] + (-1,)), "entry 31")
    _refuses(CheckpointTable(LISTS, None), "entries")
    _refuses(CheckpointTable(None, ENTRIES), "lists")
    _refuses(CheckpointTable((), (0,) * 32), "entry 0.*outside")
    _refuses(CheckpointTable(LISTS[:1] + ((1, True),) + LISTS[2:], ENTRIES), "list 1.*segment 1")
    _refuses(CheckpointTable(LISTS[:1] + ((1, 2 ** 32),) + LISTS[2:], ENTRIES), "list 1.*segment 1")
    _refuses(CheckpointTable(LISTS[:1] + ((1, 2.0),) + LISTS[2:], ENTRIES), "list 1.*segment 1")
    _refuses(CheckpointTable(LISTS[:1] + (None,) + LISTS[2:], ENTRIES), "list 1")
    # the first list must be named by an entry: the decoder starts at the lowest pointer
    _refuses(CheckpointTable(((5,),) + LISTS, tuple(e + 1 for e in ENTRIES)), "list 0 .*no entry")
    # the lists must fit before the pointer table
    big = (tuple(range(exe.CHECKPOINT_POINTERS // 4)),)
    _refuses(CheckpointTable(big, (0,) * 32), "bytes before the pointer table")


def test_checkpoints_refuses_bad_arguments():
    b = _exe(CheckpointTable(LISTS, ENTRIES))
    for n in (-1, 8, True, np.bool_(True), 1.0, "1", None):
        with pytest.raises(DcError, match="course"):
            exe.checkpoints(b, n)
    for v in ("X", "r", None, 0):
        with pytest.raises(DcError, match="variant"):
            exe.checkpoints(b, 0, v)
    for bad in (None, "1ST_READ", 5, [0, 1]):
        with pytest.raises(DcError, match="not bytes"):
            exe.checkpoints(bad, 0)
        with pytest.raises(DcError, match="not bytes"):
            exe.decode_checkpoint_table(bad)


def test_checkpoint_table_refuses_malformed():
    t = CheckpointTable(LISTS, ENTRIES)
    b = _exe(t)
    start = exe.CHECKPOINT_END - len(exe.encode_checkpoint_table(t))
    P = exe.CHECKPOINT_POINTERS

    def patched(off, word):
        c = bytearray(b)
        struct.pack_into("<I", c, off, word)
        return bytes(c)

    def refused(blob, match):
        with pytest.raises(DcError, match=match):
            exe.decode_checkpoint_table(blob)
        with pytest.raises(DcError, match=match):
            exe.checkpoints(blob, 0)

    # truncated: the pointer table needs CHECKPOINT_END bytes
    refused(b[:exe.CHECKPOINT_END - 1], "pointer table")
    refused(b"", "pointer table")
    # a pointer past the end of the file, before the load address, misaligned, or into the pointer table itself
    refused(patched(P + 4 * 5, exe.EXE_BASE + len(b) + 64), "entry 5: pointer 0x")
    refused(patched(P + 4 * 5, exe.EXE_BASE - 4), "entry 5: pointer 0x")
    refused(patched(P + 4 * 5, 0), "entry 5: pointer 0x")
    refused(patched(P + 4 * 5, exe.EXE_BASE + start + 2), "entry 5: pointer 0x")
    refused(patched(P + 4 * 5, exe.EXE_BASE + P), "entry 5: pointer 0x")
    # a pointer into the middle of a list
    refused(patched(P + 4 * 5, exe.EXE_BASE + start + 8), "entry 5: pointer .* not the start of a list")
    # a count past the end: list 1 (at start + 4) claims more segments than lie before the pointer table
    refused(patched(start + 4, 1000), r"list at 0x17e8f0.*count 1000.*past")
    refused(patched(start + 4, 0xFFFFFFFF), r"list at 0x17e8f0.*count 4294967295.*past")
    # a count that swallows the next list's count: the walk still ends at the pointers, but entry 1's pointer no
    # longer lands on a list's start
    t2 = CheckpointTable(((7,), (), (1, 2, 3)), (0, 1, 2) + (0,) * 29)
    c = bytearray(_exe(t2))
    struct.pack_into("<I", c, exe.CHECKPOINT_END - len(exe.encode_checkpoint_table(t2)), 2)
    refused(bytes(c), "entry 1: pointer .* not the start of a list")


def test_checkpoint_decode_never_raises_other_errors():
    t = CheckpointTable(LISTS, ENTRIES)
    b = bytearray(_exe(t))
    start = exe.CHECKPOINT_END - len(exe.encode_checkpoint_table(t))
    P = exe.CHECKPOINT_POINTERS
    hot = [P + 4 * e + k for e in range(32) for k in range(4)] + list(range(start, start + 24))
    rng = random.Random(7)
    seen = 0
    for i in range(1500):
        saved = []
        for _ in range(rng.randint(1, 3)):
            at = rng.choice(hot) if rng.random() < 0.7 else rng.randrange(start, exe.CHECKPOINT_END)
            saved.append((at, b[at]))
            b[at] = rng.choice((0, 1, 0x8C, 0xE8, 0x18, 0xFF, rng.randrange(256)))
        blob = memoryview(b)[:rng.randrange(exe.CHECKPOINT_END - 64, len(b))] if i % 25 == 0 else b
        try:
            got = exe.decode_checkpoint_table(blob)
        except DcError:
            got = None
        if got is not None:
            seen += 1
            enc = exe.encode_checkpoint_table(got)
            assert bytes(b[exe.CHECKPOINT_END - len(enc):exe.CHECKPOINT_END]) == enc
            for n in range(8):
                assert exe.checkpoints(blob, n, rng.choice(VARIANTS)) in got.lists
        for at, old in reversed(saved):
            b[at] = old
    assert seen > 50                        # some mutations still decode (a changed segment, a shared pointer)


# ---- the disc ----

@pytest.fixture(scope="module")
def disc(request):
    from dtools import config
    p = config.dc_path()
    if p is None or not p.is_file():
        pytest.skip("set DAYTONA_DC to the Dreamcast disc .cue or .7z")
    d = DcDisc.open(p)
    request.addfinalizer(d.close)
    return d


@pytest.fixture(scope="module")
def exe_blob(disc):
    return disc.read(exe.EXE_NAME)


def test_checkpoints_lossless(disc):
    assert exe.verify_lossless_checkpoints(disc) == []


def test_checkpoint_table_real(exe_blob):
    """The layout, confirmed: 16 lists packed from 0x17e8ec up to the pointer table (each a u32 count, then that many
    u32 road segments), then 32 pointers; entries 16-31 (M, MR) point at 0-15's lists."""
    t = exe.decode_checkpoint_table(exe_blob)
    assert t == CheckpointTable(LISTS, ENTRIES)
    assert len(exe.encode_checkpoint_table(t)) == exe.CHECKPOINT_END - 0x17E8EC
    assert exe.checkpoints(exe_blob, 0) == () and exe.checkpoints(exe_blob, 5) == ()
    assert exe.checkpoints(exe_blob, 3) == (136, 263)
    assert exe.checkpoints(exe_blob, 1) == (150, 257, 404, 539, 730)
    assert exe.checkpoints(exe_blob, 2) == (142, 275)


def test_checkpoints_match_the_arcade(exe_blob, romset_zip):
    """Dreamcast courses 0-2 are the arcade's Beginner (id 0), Expert (1) and Advanced (2): their checkpoint lists
    are the arcade's checkpoint tables' segments (golden.decode_course, the lap line left out)."""
    from dtools.arcade import golden
    from dtools.arcade.romset import RomSet
    rs = RomSet.from_zip(romset_zip)
    for n in range(3):
        arcade = tuple(cp.segment for cp in golden.decode_course(rs, n).checkpoints if not cp.lap_line)
        assert exe.checkpoints(exe_blob, n) == arcade
    assert exe.checkpoints(exe_blob, 0) == ()                   # the arcade Beginner has none either


def test_checkpoints_variants_real(disc, exe_blob):
    """Every list's segments lie on its course (0 < s < segments); the R lists are the plain lists run backwards
    (segment s -> segments - s, on the R data file's segment count); M and MR share "" and R's lists."""
    from dtools.dc import data
    for n in range(8):
        segs = {}
        for v in VARIANTS:
            name = course_files(n, v, "data")
            segs[v] = len(data.decode_data(disc.read(name), name).road) - 1
            assert all(0 < s < segs[v] for s in exe.checkpoints(exe_blob, n, v))
        plain = exe.checkpoints(exe_blob, n)
        assert exe.checkpoints(exe_blob, n, "R") == tuple(segs["R"] - s for s in reversed(plain))
        assert exe.checkpoints(exe_blob, n, "M") == plain
        assert exe.checkpoints(exe_blob, n, "MR") == exe.checkpoints(exe_blob, n, "R")
