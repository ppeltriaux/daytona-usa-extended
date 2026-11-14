import struct

import pytest

from dtools.arcade.ramdiff import Diff, Dump, RamDiffError, diff_dumps, find_smooth_vectors, load_dump

BASE = 0x00500000


def _dump(frame, data, base=BASE):
    return Dump(frame, base, bytes(data))


def test_diff_reports_changed_bytes():
    a = bytearray(64)
    b = bytearray(64)
    b[5] = 7
    b[9] = 1
    b[10] = 2
    assert diff_dumps(_dump(750, a), _dump(750, b)) == [Diff(BASE + 5, 0, 7), Diff(BASE + 9, 0, 1),
                                                        Diff(BASE + 10, 0, 2)]
    # width 4: little-endian words at aligned addresses
    assert diff_dumps(_dump(750, a), _dump(750, b), width=4) == [Diff(BASE + 4, 0, 0x700), Diff(BASE + 8, 0, 0x20100)]
    assert diff_dumps(_dump(750, a), _dump(750, a)) == []


def test_bad_width_raises():
    with pytest.raises(RamDiffError, match="width"):
        diff_dumps(_dump(1, bytes(8)), _dump(1, bytes(8)), width=3)


def test_size_mismatch_raises():
    with pytest.raises(RamDiffError, match="size"):
        diff_dumps(_dump(750, bytes(64)), _dump(750, bytes(32)))


def test_base_mismatch_raises():
    with pytest.raises(RamDiffError, match="base"):
        diff_dumps(_dump(750, bytes(64)), _dump(750, bytes(64), base=0x00200000))


def test_frame_mismatch_raises_unless_allowed():
    a, b = _dump(750, bytes(64)), _dump(1500, bytes([1]) + bytes(63))
    with pytest.raises(RamDiffError, match="frame"):
        diff_dumps(a, b)
    assert diff_dumps(a, b, allow_frame_mismatch=True) == [Diff(BASE, 0, 1)]


def test_load_dump_reads_frame_and_base(tmp_path):
    p = tmp_path / "t.txt.ram.750.bin"
    p.write_bytes(bytes(0x100000))
    d = load_dump(p)
    assert (d.frame, d.base, len(d.data)) == (750, 0x00500000, 0x100000)
    p2 = tmp_path / "t.txt.ram2.1500.bin"
    p2.write_bytes(bytes(0x20000))
    d2 = load_dump(p2)
    assert (d2.frame, d2.base, len(d2.data)) == (1500, 0x00200000, 0x20000)


def test_load_dump_rejects_bad_name_or_size(tmp_path):
    p = tmp_path / "t.txt.ram.bin"
    p.write_bytes(bytes(0x100000))
    with pytest.raises(RamDiffError, match="name"):
        load_dump(p)
    q = tmp_path / "t.txt.ram.750.bin"
    q.write_bytes(bytes(100))
    with pytest.raises(RamDiffError, match="size"):
        load_dump(q)


def test_find_smooth_vectors_picks_moving_float():
    dumps = []
    for i in range(5):
        data = bytearray(64)
        struct.pack_into("<f", data, 8, 100.0 + 3.0 * i)       # moving 3 units per dump: the candidate
        struct.pack_into("<f", data, 16, 42.0)                 # constant: never changes
        struct.pack_into("<f", data, 24, 1000.0 * i)           # jumping 1000 per dump
        struct.pack_into("<f", data, 32, float("nan") if i == 2 else 1.0 + i)   # smooth but not finite once
        dumps.append(_dump(1000 + i, data))
    got = find_smooth_vectors(dumps)
    assert [a for a, _ in got] == [BASE + 8]
    assert got[0][1] == [100.0, 103.0, 106.0, 109.0, 112.0]


def test_find_smooth_vectors_needs_matching_dumps():
    with pytest.raises(RamDiffError, match="two"):
        find_smooth_vectors([_dump(1, bytes(8))])
    with pytest.raises(RamDiffError, match="size"):
        find_smooth_vectors([_dump(1, bytes(8)), _dump(2, bytes(4))])
