import struct

import pytest

from dtools.arcade.geo import Cmd, PolyRam, count_polygons, parse_log, walk
from dtools.arcade.polyrom import PolyError
from dtools.arcade.romset import RomSet

BUF = 0x8000                                   # buffer RAM words (0x20000 bytes)
MAT = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 10.0, 20.0, 30.0]


def op(kind: int) -> int:
    return kind << 23


def f32(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def f12(values) -> list[int]:
    return [f32(v) for v in values]


def words(*items, size: int = BUF) -> list[int]:
    """A buffer image: the items (ints or lists of ints) from word 0, padded with zeros (NOPs) to `size`."""
    out: list[int] = []
    for it in items:
        out.extend(it if isinstance(it, list) else [it])
    return out + [0] * (size - len(out))


def strip_words(*attrs: int, normals: bool = True) -> list[int]:
    """An object as the parser reads it: two seed points, one record per attr (normal if `normals`, two points),
    then an end record (attr 0, plus its normal in the modes with normals)."""
    out = f12([0, 0, 0, 1, 0, 0])
    for a in attrs:
        out += [a] + (f12([0, 1, 0]) if normals else [0, 0, 0]) + f12([0, 0, 1, 1, 0, 1])
    return out + [0] + (f12([0, 1, 0]) if normals else [])


def romset(poly_words: list[int]) -> RomSet:
    blob = struct.pack(f"<{len(poly_words)}I", *poly_words)
    return RomSet(program=b"", data=b"", copro_data=b"", polygons=blob, textures=b"", extra={})


def test_matrix_object_end():
    w = words(op(0x0b), *f12(MAT), op(0x01), 0x10, 0x20, 0x800100, 2, op(0x0f))
    cmds = walk(w, 0)
    assert cmds[0].kind == 0x0b and cmds[0].offset == 0
    assert cmds[0].args["mat"] == pytest.approx(MAT)
    assert cmds[1] == Cmd(0x01, 13, {"tpa": 0x10, "tha": 0x20, "oba": 0x800100, "obc": 2})
    assert cmds[2].kind == 0x0f and cmds[2].offset == 18
    assert len(cmds) == 3


def test_command_is_bits_23_to_27_and_0x1f_ends():
    # the low 23 bits and bits 28-30 do not change the command (geo_w ORs the 20-bit data into the word)
    w = words(op(0x11) | 0x12345, 1, 2, 0x800000, 1, op(0x1f) | (1 << 28))
    cmds = walk(w, 0)
    assert [c.kind for c in cmds] == [0x11, 0x1f]
    assert cmds[0].args["oba"] == 0x800000


def test_start_is_a_word_index():
    w = words(0, 0, 0, op(0x07), 2, op(0x0f))
    cmds = walk(w, 3)
    assert [(c.kind, c.offset) for c in cmds] == [(0x07, 3), (0x0f, 5)]
    assert cmds[0].args == {"mode": 2}


def test_jump_is_followed():
    # a word with bit 31 set: continue at (word & 0x1ffff) / 4, a byte offset into the buffer
    w = words(0x80000000 | 0x100, op(0x0f))
    w[0x40] = op(0x08)
    w[0x41] = 0x300
    w[0x42] = op(0x0f)
    cmds = walk(w, 0)
    assert [(c.kind, c.offset) for c in cmds] == [(0x08, 0x40), (0x0f, 0x42)]
    assert cmds[0].args == {"zsort": 0x300}


def test_jump_ignores_bits_17_to_30():
    w = words(0x80000000 | 0x7ffe0000 | 0x10)
    w[4] = op(0x0f)
    assert [(c.kind, c.offset) for c in walk(w, 0)] == [(0x0f, 4)]


def test_jump_to_itself_terminates():
    cmds = walk(words(0x80000000), 0)
    assert cmds[-1].kind == -1
    assert cmds[-1].args == {"stopped_on": None, "reason": "0x8000 commands"}


def test_command_cap():
    # 0x7fff NOPs then the end: the end is the 0x8000th command and still runs
    w = [0] * BUF
    w[0x7fff] = op(0x0f)
    assert walk(w, 0)[-1].kind == 0x0f
    # a loop through two jumps (each costs one command) ends at the cap
    w = words(op(0x07), 0, 0x80000000 | 0x10)
    w[4] = 0x80000000
    cmds = walk(w, 0)
    assert cmds[-1] == Cmd(-1, 4, {"stopped_on": None, "reason": "0x8000 commands"})   # at the next word to read
    assert len(cmds) == 0x8000 // 3 + 1 + 1      # 10923 mode writes (one cycle is 3 commands), then the stop


def test_end_of_buffer_stops():
    cmds = walk(words(op(0x07), 0, size=0x10), 0)      # a short image: no end command
    assert cmds[-1] == Cmd(-1, 0x10, {"stopped_on": None, "reason": "buffer end"})


def test_unknown_payload_stops():
    # a payload that runs past the buffer: MAME reads past bufferram; the walker stops on the command
    cmds = walk(words(op(0x05), 0, 0x10000, op(0x0f)), 0)
    assert cmds == [Cmd(-1, 0, {"stopped_on": 0x05, "reason": "past buffer end"})]
    w = words(op(0x07), 0)
    w[-2] = op(0x0b)                            # a matrix with one word left
    cmds = walk(w, BUF - 2)
    assert cmds == [Cmd(-1, BUF - 2, {"stopped_on": 0x0b, "reason": "past buffer end"})]


def test_direct_data_inline_strip():
    rec_quad = [0x101, 7, 8] + [1, 2, 3] + [4, 5, 6]          # attr & 1: a second point follows
    rec_tri = [0x202, 9, 10] + [11, 12, 13]
    w = words(op(0x02), 0x10, 0x20, 1, 2, 3, 4, 5, 6, *rec_quad, *rec_tri, 0x4, op(0x0f))
    cmds = walk(w, 0)
    assert cmds[0].kind == 0x02
    a = cmds[0].args
    assert (a["tpa"], a["tha"], a["p0"], a["p1"]) == (0x10, 0x20, (1, 2, 3), (4, 5, 6))
    assert a["records"] == [{"attr": 0x101, "luma": 7, "distance": 8, "points": [(1, 2, 3), (4, 5, 6)]},
                            {"attr": 0x202, "luma": 9, "distance": 10, "points": [(11, 12, 13)]}]
    assert cmds[1].kind == 0x0f and cmds[1].offset == 1 + 8 + 9 + 6 + 1


def test_window_texture_upload_and_parameters():
    win = [0x00100020 + i for i in range(6)]
    w = words(op(0x03), *win, op(0x04), 0xabc, 2, 11, 22, op(0x14), 0xdef, 1, 33,
              op(0x05), 0x01000010, 3, 7, 8, 9, op(0x06), 0x8, 2, 0x11, f32(0.5), 0x22, f32(0.25), op(0x0f))
    cmds = walk(w, 0)
    assert [c.kind for c in cmds] == [0x03, 0x04, 0x14, 0x05, 0x06, 0x0f]
    assert cmds[0].args["coords"] == [(0x010, 0x020 + i) for i in range(6)] and cmds[0].args["words"] == win
    assert cmds[1].args == {"address": 0xabc, "count": 2, "data": [11, 22]}
    assert cmds[2].args == {"address": 0xdef, "count": 1, "data": [33]}
    assert cmds[3].args == {"address": 0x01000010, "ram": 1, "index": 0x10, "count": 3, "data": [7, 8, 9]}
    assert cmds[4].args == {"index": 2, "index_low": 0, "count": 2, "params": [(0x11, 0.5), (0x22, 0.25)]}
    assert cmds[5].offset == 7 + 5 + 4 + 6 + 7


def test_fixed_size_commands():
    w = words(op(0x00), op(0x07), 1, op(0x17), 3, op(0x08), 0x100, op(0x09), f32(1.0), f32(2.0),
              op(0x0a), *f12([0, 1, 0]), op(0x0c), *f12([4, 5, 6]), op(0x1c), *f12([7, 8, 9]), op(0x0d), 0x55, 9,
              op(0x10), 0x99, op(0x16), f32(3.0), op(0x1e), 0x123, op(0x1b), *f12(MAT), op(0x0f))
    cmds = walk(w, 0)
    assert [c.kind for c in cmds] == [0x00, 0x07, 0x17, 0x08, 0x09, 0x0a, 0x0c, 0x1c, 0x0d, 0x10, 0x16, 0x1e,
                                      0x1b, 0x0f]
    by = {c.kind: c.args for c in cmds}
    assert by[0x09] == {"focus": (1.0, 2.0)} and by[0x0a] == {"light": (0.0, 1.0, 0.0)}
    assert by[0x0c] == {"translate": (4.0, 5.0, 6.0)} and by[0x16] == {"lod": 3.0}
    assert by[0x0d] == {"address": 0x55, "count": 9}          # MAME reads the two words only (the push is disabled)
    assert by[0x10] == {"value": 0x99} and by[0x1e] == {"address": 0x123}
    assert cmds[-1].offset == 1 + 2 + 2 + 2 + 3 + 4 + 4 + 4 + 3 + 2 + 2 + 2 + 13


def test_test_block_and_code_upload():
    fifo = [1 << i for i in range(32)]
    w = words(op(0x0e), *fifo, 2, 0x800010, 4, 0xaaaa, 0x20, 1, 0xbbbb,
              op(0x1d), 0x400, 2, 1, 2, 3, 4, 5, 6, op(0x0f))
    cmds = walk(w, 0)
    assert [c.kind for c in cmds] == [0x0e, 0x1d, 0x0f]
    assert cmds[0].args == {"fifo_ok": True, "fifo": fifo, "blocks": [(0x000010, 4, 0xaaaa), (0x20, 1, 0xbbbb)],
                            "block_high": [0x800000, 0]}
    assert cmds[1].args == {"flags": 0x400, "count": 2, "data": [1, 2, 3, 4, 5, 6]}
    assert cmds[2].offset == 1 + 32 + 1 + 6 + 3 + 6


def test_count_polygons_reads_rom_and_uploaded_ram():
    obj = strip_words(0x101, 0x201)                          # two drawn quads
    rs = romset(obj + [0] * 16)
    ram = PolyRam()
    w = words(op(0x05), 0, len(obj), *obj,
              op(0x01), 0, 0, 0x000000, 4,                  # RAM 0 at word 0
              op(0x01), 0, 0, 0x800000, 4,                  # ROM at word 0
              op(0x0f))
    cmds = walk(w, 0)
    assert count_polygons(cmds, rs, ram) == 4


def test_count_polygons_skips_link0_and_counts_direct_data():
    obj = strip_words(0x001, 0x101, 0x301)                   # the link-0 record is culled by the hardware
    rs = romset(obj)
    direct = [0x001, 0, 0, 1, 2, 3, 4, 5, 6, 0x102, 0, 0, 7, 8, 9, 0x0]
    w = words(op(0x11), 0, 0, 0x800000, 3, op(0x12), 0, 0, 1, 2, 3, 4, 5, 6, *direct, op(0x0f))
    assert count_polygons(walk(w, 0), rs, PolyRam()) == 2 + 1


def test_poly_ram_persists_and_ram1_is_selected():
    obj = strip_words(0x101)
    ram = PolyRam()
    upload = walk(words(op(0x15), 0x01000100, len(obj), *obj, op(0x0f)), 0)
    assert count_polygons(upload, romset([0] * 8), ram) == 0
    later = walk(words(op(0x01), 0, 0, 0x01000100, 1, op(0x01), 0, 0, 0x100, 1, op(0x0f)), 0)
    # the RAM 1 object counts; RAM 0 at 0x100 is all zeros: a strip that ends at its first record
    assert count_polygons(later, romset([0] * 8), ram) == 1


def test_mode_is_state_carried_across_lists():
    obj = strip_words(0x101, 0x101, normals=False)           # the no-normal layout of modes 2-3
    rs = romset(obj)
    ram = PolyRam()
    assert count_polygons(walk(words(op(0x07), 2, op(0x0f)), 0), rs, ram) == 0
    assert ram.mode == 2
    assert count_polygons(walk(words(op(0x01), 0, 0, 0x800000, 0, op(0x0f)), 0), rs, ram) == 2


def test_obc_zero_means_until_end():
    obj = strip_words(*([0x101] * 5))
    rs = romset(obj)
    w = words(op(0x01), 0, 0, 0x800000, 0, op(0x0f))
    assert count_polygons(walk(w, 0), rs, PolyRam()) == 5
    w = words(op(0x01), 0, 0, 0x800000, 3, op(0x0f))
    assert count_polygons(walk(w, 0), rs, PolyRam()) == 3


def test_rom_index_is_masked_like_mame():
    obj = strip_words(0x101)
    rs = romset(obj)
    w = words(op(0x01), 0, 0, 0x800000 | 0x400000, 1, op(0x0f))  # bit 22 is above the 0x3fffff mask
    assert count_polygons(walk(w, 0), rs, PolyRam()) == 1


def test_object_whose_records_never_end_raises():
    rs = romset(strip_words(*([0x101] * 4))[:-4])            # no end record before the ROM ends
    w = words(op(0x01), 0, 0, 0x800000, 0, op(0x0f))
    with pytest.raises(PolyError):
        count_polygons(walk(w, 0), rs, PolyRam())


def test_parse_log_round_trip(tmp_path):
    w = words(op(0x07), 0, op(0x0f))
    hexline = "".join(f"{v:08x}" for v in w)
    log = tmp_path / "geo.txt"
    log.write_text(f"X junk\nG 1500 400 1 {hexline}\nG 1560 0 0 {hexline}\n")
    frames = list(parse_log(log))
    assert [(f, rs, vc) for f, rs, vc, _ in frames] == [(1500, 0x400, 1), (1560, 0, 0)]
    assert frames[0][3] == w
    assert [c.kind for c in walk(frames[1][3], frames[1][1] // 4)] == [0x07, 0x0f]


# --- emit: the inverse of walk (Task 9) -----------------------------------------------------------------------------
from dtools.arcade.geo import GeoError, emit


def _walked(w: list[int], cmds: list[Cmd]) -> list[int]:
    """The words `walk` read for `cmds`, jumps left out: each command's words from its offset, checked to tile the
    list (the next command starts where this one ends, or the word there is a jump)."""
    out = []
    for i, c in enumerate(cmds):
        n = len(emit([c]))
        out += w[c.offset:c.offset + n]
        if i + 1 < len(cmds) and cmds[i + 1].offset != c.offset + n:
            assert w[c.offset + n] & 0x80000000, f"gap after the command at {c.offset:#x} is not a jump"
    return out


def _every_command() -> list[int]:
    """A list holding every command kind once (0x00-0x1f), with payload bits the walker's fields do not show: opcode
    data bits, window words with bits outside their 12-bit fields, a texture-parameter index with its low bits, a
    direct-data end attribute other than 0, raw test-block addresses, code-upload words, and a jump."""
    fifo = [1 << i for i in range(32)]
    cmds = [
        [op(0x00) | 0x5], [op(0x01), 0x10, 0x20, 0x800100, 2],
        [op(0x02), 0x10, 0x20, *f12([1, 2, 3, 4, 5, 6]), 0x101, 7, 8, *f12([1, 2, 3, 4, 5, 6]), 0x202, 9, 10,
         *f12([7, 8, 9]), 0x4],
        [op(0x03), 0xFFFF0080, 0x01F00200, 0x00F8010E, 0x00F8013C, 0x01600098, 0x00F8013C],
        [op(0x04), 0x80A000, 2, 11, 22], [op(0x05), 0x01000010, 3, 7, 8, 9],
        [op(0x06), 0x9, 2, 0x11, f32(0.5), 0x22, f32(0.25)], [op(0x07), 1], [op(0x08), 0x40800000],
        [op(0x09), *f12([1.5, -2.0])], [op(0x0a), *f12([0, 1, 0])], [op(0x0b), *f12(MAT)], [op(0x0c), *f12([4, 5, 6])],
        [op(0x0d), 0x55, 9], [op(0x0e), *fifo, 2, 0xFF800010, 4, 0xAAAA, 0x20, 1, 0xBBBB],
        [op(0x10), 0x99], [op(0x11) | (1 << 29), 1, 2, 0x900000, 0], [op(0x12), 0, 0, *f12([0] * 6), 0x0],
        [op(0x13), *range(6)], [op(0x14), 0xDEF, 1, 33], [op(0x15), 0x10, 1, 5], [op(0x16), f32(3.0)],
        [op(0x17), 2], [op(0x18), 0x300], [op(0x19), *f12([1, 1])], [op(0x1a), *f12([1, 0, 0])],
        [op(0x1b), *f12(MAT)], [op(0x1c), *f12([7, 8, 9])], [op(0x1d), 0x400, 2, 1, 2, 3, 4, 5, 6], [op(0x1e), 0x123],
    ]
    out: list[int] = []
    for c in cmds[:10]:
        out += c
    jump_to = 0x400
    out += [0x80000000 | 4 * jump_to]                    # a jump over a gap of junk words
    out += [0xDEADBEEF] * (jump_to - len(out))
    for c in cmds[10:]:
        out += c
    return words(out + [op(0x0f) | 0x77])


def test_emit_round_trip_synthetic():
    w = _every_command()
    cmds = walk(w, 0)
    assert {c.kind for c in cmds} == set(range(0x20)) - {0x1f}
    assert emit(cmds) == _walked(w, cmds)
    assert emit(cmds)[:1] == [op(0x00) | 0x5]                      # the opcode word as stored, data bits included
    w2 = words(op(0x1f) | 1)
    assert emit(walk(w2, 0)) == [op(0x1f) | 1]


def test_emit_builds_commands_made_without_a_walk():
    # a converter's command list: no opcode words, no raw companion fields
    cmds = [Cmd(0x0b, 0, {"mat": MAT}), Cmd(0x01, 0, {"tpa": 1, "tha": 2, "oba": 0x800004, "obc": 5000}),
            Cmd(0x03, 0, {"coords": [(0x10, 0x20 + i) for i in range(6)]}),
            Cmd(0x06, 0, {"index": 2, "count": 1, "params": [(0x11, 0.5)]}), Cmd(0x0f, 0, {})]
    out = emit(cmds)
    assert out == [op(0x0b), *f12(MAT), op(0x01), 1, 2, 0x800004, 5000, op(0x03),
                   *[0x00100020 + i for i in range(6)], op(0x06), 8, 1, 0x11, f32(0.5), op(0x0f)]
    assert emit(walk(words(out), 0)) == out


@pytest.mark.parametrize("cmd, match", [
    (Cmd(-1, 0, {"stopped_on": None, "reason": "buffer end"}), "stopped"),
    (Cmd(0x20, 0, {}), "kind"),
    (Cmd(0x01, 0, {"tpa": 1 << 32, "tha": 0, "oba": 0, "obc": 0}), "tpa"),
    (Cmd(0x07, 0, {"mode": -1}), "mode"),
    (Cmd(0x03, 0, {"coords": [(0x1000, 0)] * 6}), "coords"),
    (Cmd(0x03, 0, {"coords": [(0, 0)] * 5}), "coords"),
    (Cmd(0x04, 0, {"address": 0, "count": 2, "data": [1]}), "count"),
    (Cmd(0x06, 0, {"index": 1 << 30, "count": 0, "params": []}), "index"),
    (Cmd(0x0b, 0, {"mat": [0.0] * 11}), "mat"),
    (Cmd(0x0b, 0, {"mat": [1e39] + [0.0] * 11}), "mat"),
    (Cmd(0x02, 0, {"tpa": 0, "tha": 0, "p0": (0, 0, 0), "p1": (0, 0, 0),
                   "records": [{"attr": 0x100, "luma": 0, "distance": 0, "points": [(0, 0, 0)]}]}), "attr"),
    (Cmd(0x01, 0, {"tpa": 0, "tha": 0, "oba": 0, "obc": 0}, opcode=0x07 << 23), "opcode"),
])
def test_emit_refuses_what_does_not_fit(cmd, match):
    with pytest.raises(GeoError, match=match):
        emit([cmd])


@pytest.mark.mame
def test_emit_round_trip_logged_list(tmp_path, romset_zip):
    """Integration: the display lists of a 30 s geo_log run (attract mode, frames 1500-1680) emit back to the words
    walk read, every command, jumps left out."""
    from dtools.mame.runner import LUA_DIR, run_mame
    log = tmp_path / "geo.txt"
    run_mame(LUA_DIR / "geo_log.lua", 30, tmp_path, romset_zip.parent,
             env={"DAYTONA_GEO_LOG": str(log), "DAYTONA_GEO_FROM": "1500", "DAYTONA_GEO_EVERY": "60"})
    lists = list(parse_log(log))
    assert lists
    kinds = set()
    for frame, start, _, w in lists:
        cmds = walk(w, start // 4)
        assert cmds[-1].kind in (0x0f, 0x1f), frame                # a whole list, not a stopped walk
        assert emit(cmds) == _walked(w, cmds), frame
        kinds |= {c.kind for c in cmds}
    assert {0x01, 0x03, 0x0b} <= kinds                             # objects, the window, matrices
