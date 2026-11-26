import struct
import zipfile
import zlib

import pytest

from dtools.arcade.romset import (COPRO_PAIR, CRCS, DATA_PAIRS, OTHER_FILES, POLY_PAIRS, PROGRAM_PAIR, TEX_PAIRS,
                                  RomSet, RomSetError)

N = 8  # tiny synthetic ROM size

ALL = (*PROGRAM_PAIR, *COPRO_PAIR, *(f for p, _ in DATA_PAIRS for f in p), *(f for p in POLY_PAIRS for f in p),
       *(f for p, _ in TEX_PAIRS for f in p), *OTHER_FILES)


def _files(**override):
    files = {name: bytes([i + 1] * N) for i, name in enumerate(ALL)}
    files.update(override)
    return files


def test_program_interleaves_word_lanes():
    a, b = bytes([0xA0, 0xA1, 0xA2, 0xA3] + [0] * 4), bytes([0xB0, 0xB1, 0xB2, 0xB3] + [0] * 4)
    rs = RomSet.from_files(_files(**{"epr-16722a.12": a, "epr-16723a.13": b}), verify=False)
    assert rs.program[:8] == bytes([0xA0, 0xA1, 0xB0, 0xB1, 0xA2, 0xA3, 0xB2, 0xB3])
    assert len(rs.program) == 256 * 1024


def test_image_sizes_and_other_files():
    rs = RomSet.from_files(_files(), verify=False)
    assert len(rs.data) == 16 << 20 and len(rs.copro_data) == 4 << 20
    assert len(rs.polygons) == 16 << 20 and len(rs.textures) == 16 << 20
    assert set(rs.extra) == set(OTHER_FILES)
    assert rs.extra["epr-16726.bin"] == _files()["epr-16726.bin"]


def test_data_pairs_offsets_and_1mb_copies():
    rs = RomSet.from_files(_files(), verify=False)
    assert [off for _, off in DATA_PAIRS] == [0, 0x400000, 0x800000]
    assert rs.data[0x400000] == _files()["mpr-16808.8"][0] and rs.data[0x400002] == _files()["mpr-16809.9"][0]
    assert rs.data[0x800000:0x800008] == rs.data[0x900000:0x900008]
    for k in range(1, 8):
        assert rs.data[0x800000:0x900000] == rs.data[0x800000 + k * 0x100000:0x900000 + k * 0x100000]
    assert rs.data[0x800010:0x800020] == bytes(16)           # the rest of the 1 MB block stays zero


def test_copro_halves_and_polygon_textures_placement():
    f = _files()
    rs = RomSet.from_files(f, verify=False)
    assert rs.copro_data[0] == f["mpr-16537.ic28"][0] and rs.copro_data[2] == f["mpr-16536.ic29"][0]
    assert rs.polygons[0xC00000] == f["mpr-16772.ic19"][0] and rs.polygons[0xC00002] == f["mpr-16771.ic23"][0]
    assert rs.textures[0x800000] == f["mpr-16770.27"][0] and rs.textures[0x800002] == f["mpr-16769.26"][0]
    assert rs.textures[0x400000:0x800000] == bytes(0x400000)


def test_read_i960_maps_regions():
    rs = RomSet.from_files(_files(), verify=False)
    assert rs.read_i960(0x02000004, 4) == rs.data[4:8]
    assert rs.read_i960(0x00000002, 4) == rs.program[2:6]
    assert rs.read_i960(0x00220000, 4) == rs.program[0x20000:0x20004]
    assert rs.read_i960(0x02FFFFFC, 4) == rs.data[0xFFFFFC:0x1000000]
    for bad in (0x00040000, 0x00500000, 0x0003FFFE, 0x03000000):
        with pytest.raises(RomSetError):
            rs.read_i960(bad, 4)


def test_poly_word_masks_index():
    a = bytes([0xA0, 0xA1, 0xA2, 0xA3] + [0] * 4)
    b = bytes([0xB0, 0xB1, 0xB2, 0xB3] + [0] * 4)
    rs = RomSet.from_files(_files(**{"mpr-16523.ic16": a, "mpr-16518.ic20": b}), verify=False)
    assert rs.poly_word(0) == 0xB1B0A1A0
    assert rs.poly_word(0x400000) == rs.poly_word(0)
    assert rs.poly_float(0) == struct.unpack("<f", bytes([0xA0, 0xA1, 0xB0, 0xB1]))[0]


def test_missing_file_named():
    f = _files()
    del f["mpr-16771.ic23"]
    del f["epr-16488a.ic12"]
    with pytest.raises(RomSetError, match="mpr-16771.ic23") as e:
        RomSet.from_files(f, verify=False)
    assert "epr-16488a.ic12" in str(e.value)


def test_bad_crc_reported_with_expected_value():
    with pytest.raises(RomSetError, match=r"epr-16722a\.12: crc ") as e:
        RomSet.from_files(_files())
    assert f"expected {CRCS['epr-16722a.12']:08x}" in str(e.value)
    assert "mpr-16771.ic23: crc" in str(e.value)               # every mismatch is listed


def test_pair_size_mismatch_rejected():
    with pytest.raises(RomSetError, match="mpr-16518.ic20"):
        RomSet.from_files(_files(**{"mpr-16518.ic20": b"\0" * (N - 2)}), verify=False)


def test_crcs_cover_every_file():
    assert set(CRCS) == set(ALL) and len(ALL) == 36


def test_zip_reads_root_members_only(tmp_path):
    p = tmp_path / "daytona.zip"
    with zipfile.ZipFile(p, "w") as z:
        for name, blob in _files().items():
            z.writestr(name, blob)
        z.writestr("daytona93/epr-16722a.12", b"clone")
    rs = RomSet.from_zip(p, verify=False)
    assert rs.extra["epr-16726.bin"] == _files()["epr-16726.bin"]
    p2 = tmp_path / "clone_only.zip"
    with zipfile.ZipFile(p2, "w") as z:
        for name, blob in _files().items():
            z.writestr(("daytona93/" if name == "mpr-16771.ic23" else "") + name, blob)
    with pytest.raises(RomSetError, match="mpr-16771.ic23"):
        RomSet.from_zip(p2, verify=False)


def test_dump_regions_writes_named_files(tmp_path):
    rs = RomSet.from_files(_files(), verify=False)
    written = rs.dump_regions(tmp_path)
    assert {p.name for p in written} == {"program.bin", "data.bin", "copro_data.bin", "polygons.bin", "textures.bin"}
    assert (tmp_path / "program.bin").stat().st_size == 256 * 1024


def test_real_romset(romset_zip):
    rs = RomSet.from_zip(romset_zip)
    lane_a = bytearray(0x20000)
    lane_a[0::2], lane_a[1::2] = rs.program[0::4], rs.program[1::4]
    assert zlib.crc32(lane_a) == 0x48B94318
    assert rs.program[0x3CE98:0x40000] == b"\xff" * 12648
    assert rs.copro_data[:0x200000] == rs.copro_data[0x200000:]
    # free runs of 0xff, measured on the real set (the end of each run is the next non-0xff byte)
    assert rs.data[0x65D120:0x65D120 + 667360] == b"\xff" * 667360
    assert rs.copro_data[0x12893E:0x12893E + 882370] == b"\xff" * 882370
    assert rs.polygons[0xE3C364:0xE3C364 + 1850524] == b"\xff" * 1850524
    assert rs.polygons[0xE3C363] != 0xFF
    assert rs.textures[0xB52D48:0xB52D48 + 709304] == b"\xff" * 709304
    assert len(rs.textures) == 16 * 2**20


def _full_files():
    """Synthetic files at the real sizes (identical copro halves, as on the real set)."""
    import numpy as np
    rng = np.random.default_rng(5)
    sizes = {PROGRAM_PAIR: 0x20000, COPRO_PAIR: 0x200000}
    sizes.update({p: 0x200000 for p, _ in DATA_PAIRS[:2]})
    sizes[DATA_PAIRS[2][0]] = 0x80000
    sizes.update({p: 0x200000 for p in (*POLY_PAIRS, *(q for q, _ in TEX_PAIRS))})
    f = _files()
    for pair, n in sizes.items():
        for name in pair:
            f[name] = rng.integers(0, 256, n, dtype="uint8").tobytes()
    for name in COPRO_PAIR:
        f[name] = f[name][:0x100000] * 2
    return f


def test_split_files_inverts_from_files():
    f = _full_files()
    rs = RomSet.from_files(f, verify=False)
    out = rs.split_files()
    assert set(out) == set(CRCS)
    assert out == f


def test_with_patch_bounds():
    rs = RomSet.from_files(_files(), verify=False)
    with pytest.raises(RomSetError, match="outside program"):
        rs.with_patch("program", 256 * 1024 - 2, b"abcd")
    with pytest.raises(RomSetError, match="outside data"):
        rs.with_patch("data", -1, b"a")
    with pytest.raises(RomSetError, match="unknown region"):
        rs.with_patch("nope", 0, b"a")
    assert rs.with_patch("program", 256 * 1024 - 4, b"abcd").program[-4:] == b"abcd"


def test_with_patch_does_not_mutate_original():
    rs = RomSet.from_files(_files(), verify=False)
    before = rs.polygons[8:10]
    p = rs.with_patch("polygons", 8, b"\x01\x02")
    assert p.polygons[8:10] == b"\x01\x02" and rs.polygons[8:10] == before != b"\x01\x02"
    assert p.program is rs.program


def test_with_patch_copies_go_into_every_copy():
    rs = RomSet.from_files(_full_files(), verify=False)
    p = rs.with_patch("data", 0x800010, b"\xaa\xbb")
    for k in range(8):
        assert p.data[0x800010 + k * 0x100000:0x800012 + k * 0x100000] == b"\xaa\xbb"
    p.split_files()
    q = rs.with_patch("data", 0xf00020, b"\xcc")           # a patch in copy 7 lands in all copies too
    assert q.data[0x800020] == 0xCC
    with pytest.raises(RomSetError, match="spans two"):
        rs.with_patch("data", 0x8FFFFF, b"ab")
    with pytest.raises(RomSetError, match="straddles"):
        rs.with_patch("data", 0x7FFFFF, b"ab")
    c = rs.with_patch("copro_data", 0x1000, b"\x11")
    assert c.copro_data[0x1000] == 0x11 == c.copro_data[0x201000]
    c.split_files()
    with pytest.raises(RomSetError, match="halves"):
        rs.with_patch("copro_data", 0x1FFFFF, b"ab")


def test_split_files_refuses_differing_copies():
    import dataclasses
    rs = RomSet.from_files(_full_files(), verify=False)
    d = bytearray(rs.data)
    d[0xA00000] ^= 1
    with pytest.raises(RomSetError, match="data copy 2"):
        dataclasses.replace(rs, data=bytes(d)).split_files()
    c = bytearray(rs.copro_data)
    c[5] ^= 1
    with pytest.raises(RomSetError, match="copro_data halves"):
        dataclasses.replace(rs, copro_data=bytes(c)).split_files()


def test_write_zip_holds_exactly_the_36_files(tmp_path):
    rs = RomSet.from_files(_full_files(), verify=False)
    p = rs.write_zip(tmp_path / "out" / "daytona.zip")
    with zipfile.ZipFile(p) as z:
        assert set(z.namelist()) == set(CRCS)
    assert RomSet.from_zip(p, verify=False) == rs


def test_split_files_real_crcs(romset_zip):
    rs = RomSet.from_zip(romset_zip)
    for name, blob in rs.split_files().items():
        assert zlib.crc32(blob) == CRCS[name], name


def test_write_zip_loads_back(romset_zip, tmp_path):
    rs = RomSet.from_zip(romset_zip)
    back = RomSet.from_zip(rs.write_zip(tmp_path / "daytona.zip"))
    assert back == rs


def test_with_patches_equals_fold():
    rs = RomSet.from_files(_files(), verify=False)
    entries = {("data", 0x100): b"\x01\x02\x03", ("data", 0x2000): b"\xaa", ("data", 0x3000): b"\x07" * 16,
               ("data", 0xb00010): b"\xcc\xdd", ("copro_data", 0x2fff00): b"\x55\x66", ("polygons", 4): b"\x09"}
    want = rs
    for (region, off), blob in sorted(entries.items()):
        want = want.with_patch(region, off, blob)
    got = rs.with_patches(entries)
    for r in ("program", "data", "copro_data", "polygons", "textures"):
        assert getattr(got, r) == getattr(want, r), r
    assert rs.with_patches({}) == rs


def test_with_patches_overlap_refused():
    rs = RomSet.from_files(_files(), verify=False)
    with pytest.raises(RomSetError, match="overlap"):
        rs.with_patches({("data", 0x100): b"abcd", ("data", 0x102): b"x"})
    with pytest.raises(RomSetError, match="overlap"):
        rs.with_patches({("data", 0x100): b"abcd", ("data", 0x103): b"y"})
    rs.with_patches({("data", 0x100): b"abcd", ("data", 0x104): b"x", ("polygons", 0x100): b"z"})   # adjacent: fine
    with pytest.raises(RomSetError):
        rs.with_patches({("program", 256 * 1024 - 2): b"abcd"})
    with pytest.raises(RomSetError, match="spans two"):
        rs.with_patches({("data", 0x8ffffe): b"abcd"})
    with pytest.raises(RomSetError):
        rs.with_patches({("nope", 0): b"a"})


def test_with_patches_speed(romset_zip):
    import time

    from dtools.arcade.golden import decode_course, encode_course
    rs = RomSet.from_zip(romset_zip)
    entries = encode_course(decode_course(rs, 0), rs)
    assert len(entries) > 40000
    t = time.perf_counter()
    got = rs.with_patches(entries)
    assert time.perf_counter() - t < 5.0
    assert got.split_files()


def test_with_patches_agreeing_overlap_accepted():
    """Two entries of a region may overlap where their bytes agree (golden's nested texel blocks); the result is the
    fold. Copies of a copied region (the data ROM's 1 MB block and its seven copies, the two copro_data halves) are
    the same bytes, so entries there overlap through the alias."""
    rs = RomSet.from_files(_files(), verify=False)
    nested = {("data", 0x200): b"abcdefgh", ("data", 0x202): b"cd", ("data", 0x206): b"ghij"}
    got = rs.with_patches(nested)
    assert got.data[0x200:0x20A] == b"abcdefghij"
    aliased = {("data", 0x800100): b"wxyz", ("data", 0x900102): b"yz", ("copro_data", 0x10): b"1234",
               ("copro_data", 0x200012): b"34"}
    got = rs.with_patches(aliased)
    assert got.data[0x800100:0x800104] == got.data[0xF00100:0xF00104] == b"wxyz"
    assert got.copro_data[0x10:0x14] == got.copro_data[0x200010:0x200014] == b"1234"


def test_with_patches_differing_overlap_refused():
    rs = RomSet.from_files(_files(), verify=False)
    with pytest.raises(RomSetError, match=r"overlap.*differ"):
        rs.with_patches({("data", 0x200): b"abcdefgh", ("data", 0x202): b"cd", ("data", 0x206): b"gX"})
    with pytest.raises(RomSetError, match=r"overlap.*differ"):                   # aliased: copy 1 of the 1 MB block
        rs.with_patches({("data", 0x800100): b"wxyz", ("data", 0x900102): b"yZ"})
    with pytest.raises(RomSetError, match=r"overlap.*differ"):                   # aliased: the second copro half
        rs.with_patches({("copro_data", 0x10): b"1234", ("copro_data", 0x200011): b"2X"})


@pytest.mark.parametrize("course", [0, 1, 2])
def test_with_patches_course_is_identity(romset_zip, course):
    """encode_course on the decoded course writes the set's own bytes back: with_patches of them is the set (Expert's
    texel entries nest: e.g. data 0x440200 + 128 and 0x440220 + 32, their bytes equal)."""
    from dtools.arcade.golden import decode_course, encode_course
    rs = RomSet.from_zip(romset_zip)
    assert rs.with_patches(encode_course(decode_course(rs, course), rs)) == rs


# ---- the custom MAME build's larger regions (Phase 8, Task 2): docs/spike/2026-11-sixteen.md, "Room" ----

def _custom_files(stock: dict[str, bytes]) -> dict[str, bytes]:
    """`stock` plus the five custom files, each a distinct byte pattern at its exact size."""
    from dtools.arcade.romset import CUSTOM_FILES
    out = dict(stock)
    for k, (name, (_, _, n)) in enumerate(CUSTOM_FILES.items()):
        out[name] = bytes([0x10 + k]) * n
    return out


def test_custom_files_and_free_runs_layout():
    """Five custom files, one per new area, each a plain (not interleaved) load of image bytes; CUSTOM_AREAS is exactly
    their spans and CUSTOM_FREE those less RESERVED (the sixteen-entry select's data, grow_ids' home and the SRAM sink);
    REGION_SIZES_CUSTOM the grown
    images (the MAME regions of the patch, copro_data counted from 0)."""
    from dtools.arcade.romset import CUSTOM_AREAS, CUSTOM_FILES, CUSTOM_FREE, REGION_SIZES, REGION_SIZES_CUSTOM
    assert REGION_SIZES == {"program": 0x40000, "data": 0x1000000, "copro_data": 0x400000, "polygons": 0x1000000,
                            "textures": 0x1000000}
    assert REGION_SIZES_CUSTOM == {"program": 0x40000, "data": 0x2000000, "copro_data": 0x2000000,
                                   "polygons": 0x2000000, "textures": 0x1000000}
    assert CUSTOM_FILES == {"dx-data.bin": ("data", 0x1000000, 0x1000000),
                            "dx-copro.bin": ("copro_data", 0x400000, 0x1C00000),
                            "dx-poly.bin": ("polygons", 0x1000000, 0x1000000),
                            "dx-tex1.bin": ("textures", 0x400000, 0x400000),
                            "dx-tex2.bin": ("textures", 0xC00000, 0x400000)}
    assert CUSTOM_AREAS == {"data": ((0x1000000, 0x1000000),), "copro_data": ((0x400000, 0x1C00000),),
                            "polygons": ((0x1000000, 0x1000000),),
                            "textures": ((0x400000, 0x400000), (0xC00000, 0x400000))}
    assert CUSTOM_FREE == {**CUSTOM_AREAS, "data": ((0x10017E0, 0xEFE820), (0x1F20000, 0xD0000), (0x1FF4000, 0xBA00))}   # and the Dreamcast field's lists (0x1FFFA00, 0x300)
    # v0.4: less the Car Select screen's graphics (0x1FF0000, 0x4000) and the cars' table (0x1FFFE00, 0x200)
    assert not set(CUSTOM_FILES) & set(CRCS)


def test_as_custom_keeps_stock_bytes_and_fills_new_areas_ff():
    from dtools.arcade.romset import CUSTOM_FREE, REGION_SIZES_CUSTOM
    rs = RomSet.from_files(_full_files(), verify=False)
    assert rs.custom is False
    c = rs.as_custom()
    assert c.custom is True and c.as_custom() is c
    for r, n in REGION_SIZES_CUSTOM.items():
        assert len(getattr(c, r)) == n, r
    assert c.program == rs.program and c.extra == rs.extra
    assert c.data[:0x1000000] == rs.data and c.copro_data[:0x400000] == rs.copro_data
    assert c.polygons[:0x1000000] == rs.polygons
    assert c.textures[:0x400000] == rs.textures[:0x400000] and c.textures[0x800000:0xC00000] == \
        rs.textures[0x800000:0xC00000]
    for r, runs in CUSTOM_FREE.items():
        for lo, n in runs:
            assert getattr(c, r)[lo:lo + n] == b"\xff" * n, (r, hex(lo))


def test_romset_custom_round_trip(tmp_path):
    """A custom RomSet writes the 36 stock files, byte-identical to the stock set's, plus the five custom files, and
    reads back equal; a patch in a new area survives the trip; a zip without the custom files reads as stock."""
    from dtools.arcade.romset import CUSTOM_FILES
    rs = RomSet.from_files(_full_files(), verify=False)
    c = rs.as_custom().with_patches({("data", 0x1234560): b"new!", ("copro_data", 0x1FFFFFC): b"tail",
                                     ("polygons", 0x1000000): b"\x01", ("textures", 0xFFFFFE): b"zz",
                                     ("data", 0x800010): b"\xaa"})
    files = c.split_files()
    assert set(files) == set(CRCS) | set(CUSTOM_FILES)
    stock = rs.with_patch("data", 0x800010, b"\xaa").split_files()
    assert {n: files[n] for n in CRCS} == stock
    for name, (region, lo, n) in CUSTOM_FILES.items():
        assert files[name] == getattr(c, region)[lo:lo + n], name
    p = c.write_zip(tmp_path / "custom" / "daytona.zip")
    with zipfile.ZipFile(p) as z:
        assert set(z.namelist()) == set(CRCS) | set(CUSTOM_FILES)
    back = RomSet.from_zip(p, verify=False)
    assert back == c and back.custom is True
    assert back.data[0x1234560:0x1234564] == b"new!" and back.textures[0xFFFFFE:0x1000000] == b"zz"
    plain = RomSet.from_zip(rs.write_zip(tmp_path / "stock" / "daytona.zip"), verify=False)
    assert plain == rs and plain.custom is False


def test_custom_from_files_places_each_file():
    from dtools.arcade.romset import CUSTOM_FILES
    f = _custom_files(_full_files())
    c = RomSet.from_files(f, verify=False)
    assert c.custom is True
    for name, (region, lo, n) in CUSTOM_FILES.items():
        assert getattr(c, region)[lo:lo + n] == f[name], name
    assert c.split_files() == f


def test_custom_from_files_refuses_partial_or_wrong_sizes():
    f = _custom_files(_files())
    partial = dict(f)
    del partial["dx-poly.bin"], partial["dx-tex2.bin"]
    with pytest.raises(RomSetError, match="dx-poly.bin") as e:
        RomSet.from_files(partial, verify=False)
    assert "dx-tex2.bin" in str(e.value)
    with pytest.raises(RomSetError, match=r"dx-tex1\.bin.*4194303"):
        RomSet.from_files({**f, "dx-tex1.bin": f["dx-tex1.bin"][:-1]}, verify=False)
    with pytest.raises(RomSetError, match="not bytes"):
        RomSet.from_files({**f, "dx-data.bin": 7}, verify=False)


def test_custom_flag_must_be_bool():
    import dataclasses
    rs = RomSet.from_files(_files(), verify=False)
    for bad in (1, 0, None, "yes"):
        with pytest.raises(RomSetError, match="custom"):
            dataclasses.replace(rs, custom=bad)


def test_custom_split_files_checks_sizes():
    import dataclasses
    c = RomSet.from_files(_full_files(), verify=False).as_custom()
    with pytest.raises(RomSetError, match="polygons is 16777216 bytes, expected 33554432"):
        dataclasses.replace(c, polygons=c.polygons[:0x1000000]).split_files()
    rs = RomSet.from_files(_full_files(), verify=False)
    with pytest.raises(RomSetError, match="data is 33554432 bytes, expected 16777216"):
        dataclasses.replace(rs, data=c.data).split_files()


def test_custom_patches_have_no_copies_in_the_new_areas():
    """The data ROM's 1 MB copies end at 0xffffff and copro_data's mirrored halves at 4 MB: a patch in a new area is
    written once; a patch across either boundary is refused; patches below them are still copied."""
    c = RomSet.from_files(_full_files(), verify=False).as_custom()
    p = c.with_patch("data", 0x1000010, b"\xaa\xbb")
    assert p.data[0x1000010:0x1000012] == b"\xaa\xbb"
    assert p.data[0x800010:0x800012] == c.data[0x800010:0x800012] and p.data[:0x1000000] == c.data[:0x1000000]
    assert c.with_patch("data", 0x10FFFFE, b"abcd").data[0x10FFFFE:0x1100002] == b"abcd"    # across a 1 MB boundary
    with pytest.raises(RomSetError, match="0x1000000"):
        c.with_patch("data", 0xFFFFFE, b"abcd")
    q = c.with_patch("copro_data", 0x400000, b"\x11\x22")
    assert q.copro_data[0x400000:0x400002] == b"\x11\x22" and q.copro_data[:0x400000] == c.copro_data[:0x400000]
    with pytest.raises(RomSetError, match="0x400000"):
        c.with_patch("copro_data", 0x3FFFFE, b"abcd")
    r = c.with_patch("copro_data", 0x10, b"\x33")
    assert r.copro_data[0x10] == 0x33 == r.copro_data[0x200010]
    s = c.with_patch("data", 0x900020, b"\x44")
    assert all(s.data[0x800020 + k * 0x100000] == 0x44 for k in range(8)) and s.data[0x1000020] == 0xFF
    s.split_files()
    both = c.with_patches({("data", 0x1000100): b"wxyz", ("data", 0x1000102): b"yz", ("copro_data", 0x400010): b"12"})
    assert both.data[0x1000100:0x1000104] == b"wxyz"
    with pytest.raises(RomSetError, match=r"overlap.*differ"):
        c.with_patches({("data", 0x1000100): b"wxyz", ("data", 0x1000102): b"yZ"})
    c.with_patches({("data", 0x800100): b"ab", ("data", 0x1800100): b"cd"})     # not aliases: no overlap


def test_custom_read_i960_and_poly_word():
    """read_i960 serves the data ROM's 32 MB window (0x02000000-0x03ffffff) on a custom set, 16 MB on the stock one;
    poly_word masks the word index as MAME does, by the region (0x7fffff on 32 MB, 0x3fffff on 16 MB)."""
    rs = RomSet.from_files(_full_files(), verify=False)
    c = rs.as_custom().with_patches({("data", 0x1FFFFFC): b"\x01\x02\x03\x04",
                                     ("polygons", 0x1000000): b"\x05\x06\x07\x08"})
    assert c.read_i960(0x03FFFFFC, 4) == b"\x01\x02\x03\x04"
    assert c.read_i960(0x02000000, 4) == rs.data[:4]
    with pytest.raises(RomSetError):
        rs.read_i960(0x03000000, 4)
    with pytest.raises(RomSetError):
        c.read_i960(0x03FFFFFE, 4)
    assert c.poly_word(0x400000) == 0x08070605 and rs.poly_word(0x400000) == rs.poly_word(0)
    assert c.poly_word(0x800000) == c.poly_word(0)
    assert c.poly_word(5) == rs.poly_word(5)


def test_custom_real_set_stock_files_unchanged(romset_zip, tmp_path):
    """On the real set: the custom set's 36 stock files keep their MAME CRCs; it reads back equal."""
    from dtools.arcade.romset import CUSTOM_FILES
    c = RomSet.from_zip(romset_zip).as_custom()
    files = c.split_files()
    for name, want in CRCS.items():
        assert zlib.crc32(files[name]) == want, name
    assert all(files[n] == b"\xff" * CUSTOM_FILES[n][2] for n in CUSTOM_FILES)
    assert RomSet.from_zip(c.write_zip(tmp_path / "daytona.zip")) == c


def test_free_runs_custom(romset_zip):
    """Review Focus 5 for the new runs: on the real set made custom every CUSTOM_FREE run is 0xff; on the stock set
    the same bytes were never ROM (past the image, or the textures' zero fill: no file loads there); no table of the
    three courses (golden.coverage, golden.excluded_spans) reaches into one; tgp.write_free writes there once (the
    free-bytes rule: a second write over the same bytes is refused, so is one on the stock set)."""
    from dtools.arcade import golden as G
    from dtools.arcade import tgp
    from dtools.arcade.romset import CUSTOM_FREE
    rs = RomSet.from_zip(romset_zip)
    c = rs.as_custom()
    for region, runs in CUSTOM_FREE.items():
        img, stock = getattr(c, region), getattr(rs, region)
        for lo, n in runs:
            assert img[lo:lo + n] == b"\xff" * n, (region, hex(lo))
            assert lo >= len(stock) or stock[lo:lo + n] == bytes(n), (region, hex(lo))
    spans = [s for i in range(3) for s in G.coverage(rs, i) + G.excluded_spans(rs, i)]
    assert len(spans) > 1000
    for region, off, n, name in spans:
        for lo, size in CUSTOM_FREE.get(region, ()):
            assert off + n <= lo or lo + size <= off, (region, hex(off), n, name)
    for region in ("data", "copro_data"):
        lo, n = CUSTOM_FREE[region][0]
        for at in (lo, lo + n - 8):
            w = tgp.write_free(c, region, at, b"\x01\x02\x03\x04\x05\x06\x07\x08")
            assert getattr(w, region)[at:at + 8] == b"\x01\x02\x03\x04\x05\x06\x07\x08"
            assert getattr(w, region)[:lo] == getattr(c, region)[:lo]
            with pytest.raises(tgp.TgpError, match="not free"):
                tgp.write_free(w, region, at + 4, b"\0\0\0\0")
        with pytest.raises(tgp.TgpError, match="past the region"):
            tgp.write_free(rs, region, lo, b"\0\0\0\0")
    w = tgp.write_free(c, "data", 0x10FFFFC, b"abcdefgh")              # across a 1 MB boundary: no copies up here
    assert w.data[0x10FFFFC:0x1100004] == b"abcdefgh"
    with pytest.raises(tgp.TgpError):
        tgp.write_free(c, "copro_data", 0x3FFFFC, b"abcdefgh")          # the second half's end into the new area
