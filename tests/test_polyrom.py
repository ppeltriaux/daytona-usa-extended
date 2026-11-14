import struct

import pytest

from dtools.arcade.polyrom import Object, Poly, PolyError, Record, decode_object, encode_object, polygons

N = (0.0, 1.0, 0.0)
S0, S1 = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
A, B, C, D = (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 0.0, 2.0), (1.0, 0.0, 2.0)
E, F = (0.0, 0.0, 3.0), (1.0, 0.0, 3.0)
Z = (9.0, 9.0, 9.0)


def strip(mode: int, *records, seeds=(S0, S1)) -> bytes:
    """Two seed points, then ten words per record: attr, normal (read in modes 0-1, skipped in 2-3), point, point.
    `mode` only documents the layout the test means; the words are the same in every mode."""
    assert mode in (0, 1, 2, 3)
    out = b"".join(struct.pack("<3f", *p) for p in seeds)
    for attr, n, p0, p1 in records:
        out += struct.pack("<I", attr) + struct.pack("<9f", *n, *p0, *p1)
    return out


def test_mode0_quad_then_triangle_then_end():
    words = strip(0, (0x1 | (3 << 18), N, A, B), (0x2, N, C, D), (0x0, N, Z, Z))
    obj = decode_object(words, 0)
    assert obj.seed0 == S0 and obj.seed1 == S1
    assert len(obj.records) == 2
    # seeds 6, quad 10, triangle 10 (its second point skipped by `input += 3`), end record: attr + normal = 4
    # (geo_parse_np_ns reads the normal before testing attr & 3, then breaks)
    assert obj.word_count == 6 + 10 + 10 + 4
    assert obj.records[0] == Record(0x1 | (3 << 18), N, A, B)
    assert obj.records[0].is_quad and obj.records[0].tex_param == 3
    assert obj.records[1].p1 is None                       # triangle: its second point is skipped, not kept
    assert not obj.records[1].is_quad and obj.records[1].p0 == C


def test_attr_fields():
    r = Record((0x1f << 27) | (0x15 << 18) | (3 << 8) | 3, None, A, B)
    assert r.is_quad and not r.ends
    assert r.tex_param == 0x15 and r.coef == 0x1f and r.link == 3
    assert Record(0x4, None, A, None).ends


def test_mode1_reads_like_mode0():
    words = strip(1, (0x1, N, A, B), (0x0, N, Z, Z))
    obj = decode_object(words, 0, mode=1)
    assert obj.records == [Record(0x1, N, A, B)] and obj.word_count == 6 + 10 + 4


@pytest.mark.parametrize("mode", [2, 3])
def test_mode2_records_have_no_normal(mode):
    words = strip(mode, (0x1, N, A, B), (0x2, N, C, D), (0x0, N, Z, Z))
    obj = decode_object(words, 0, mode=mode)
    assert obj.records == [Record(0x1, None, A, B), Record(0x2, None, C, None)]
    # the normal words are still there (`input += 3`); the end record is the attribute alone
    assert obj.word_count == 6 + 10 + 10 + 1


def test_mode2_end_record_needs_only_its_attribute():
    words = strip(2, (0x1, N, A, B)) + struct.pack("<I", 0)
    assert decode_object(words, 0, mode=2).word_count == 6 + 10 + 1
    with pytest.raises(PolyError):                         # mode 0 reads the end record's normal too
        decode_object(words, 0, mode=0)


def test_index_is_a_word_index():
    words = b"\xff" * 8 + strip(0, (0x1, N, A, B), (0x0, N, Z, Z))
    assert decode_object(words, 2).records == [Record(0x1, N, A, B)]


def test_count_limits_records():
    words = strip(0, (0x1, N, A, B), (0x1, N, C, D), (0x1, N, E, F))
    obj = decode_object(words, 0, count=1)
    assert len(obj.records) == 1 and obj.word_count == 6 + 10


def test_count_zero_reads_only_the_seeds():
    obj = decode_object(strip(0, (0x1, N, A, B)), 0, count=0)
    assert obj.records == [] and obj.word_count == 6


def test_stops_at_end_record():
    words = strip(0, (0x1, N, A, B), (0x0, N, Z, Z), (0x1, N, C, D))
    assert len(decode_object(words, 0).records) == 1


def test_runs_off_end_raises():
    words = strip(0, (0x1, N, A, B), (0x1, N, C, D))
    with pytest.raises(PolyError):
        decode_object(words, len(words) // 4 - 3)
    with pytest.raises(PolyError):                         # no end record before the buffer ends
        decode_object(words, 0)


def test_max_records_guard_raises():
    words = strip(0, *([(0x1, N, A, B)] * 5), (0x0, N, Z, Z))
    with pytest.raises(PolyError):
        decode_object(words, 0, max_records=3)
    assert len(decode_object(words, 0, max_records=5).records) == 5


def test_polygons_join_the_previous_two_points():
    obj = decode_object(strip(0, (0x1, N, A, B), (0x0, N, Z, Z)), 0)
    q = polygons(obj)
    assert set(q[0].points) == {S0, S1, A, B}
    # corner order of model2_3d_process_polygon: v0 = P1(n-1), v1 = P0(n-1), v2 = P0(n), v3 = P1(n)
    assert q == [Poly((S1, S0, A, B), N, 0x1)]


def test_link0_and_link2_carry_both_new_points():
    for link in (0, 2):
        obj = decode_object(strip(0, (0x1 | link << 8, N, A, B), (0x1, N, C, D), (0x0, N, Z, Z)), 0)
        assert [p.points for p in polygons(obj)] == [(S1, S0, A, B), (B, A, C, D)]


def test_link1_replaces_old_p1_with_p0():
    obj = decode_object(strip(0, (0x101, N, A, B), (0x1, N, C, D), (0x0, N, Z, Z)), 0)
    assert [p.points for p in polygons(obj)] == [(S1, S0, A, B), (A, S0, C, D)]


def test_link3_replaces_old_p0_with_p1():
    obj = decode_object(strip(0, (0x301, N, A, B), (0x1, N, C, D), (0x0, N, Z, Z)), 0)
    assert [p.points for p in polygons(obj)] == [(S1, S0, A, B), (S1, B, C, D)]


def test_triangle_is_three_corners_and_links_with_its_point_twice():
    obj = decode_object(strip(0, (0x2, N, A, Z), (0x1, N, C, D), (0x0, N, Z, Z)), 0)
    assert [p.points for p in polygons(obj)] == [(S1, S0, A), (A, A, C, D)]


def test_polygons_of_normalless_mode_carry_no_normal():
    obj = decode_object(strip(2, (0x1, N, A, B), (0x0, N, Z, Z)), 0, mode=2)
    assert polygons(obj) == [Poly((S1, S0, A, B), None, 0x1)]


def test_link0_polygon_is_emitted_but_not_drawn():
    # check_culling (model2_v.cpp 253-255) culls every link-0 polygon; the link update still runs
    obj = decode_object(strip(0, (0x001, N, A, B), (0x101, N, C, D), (0x201, N, E, F), (0x302, N, A, Z),
                              (0x0, N, Z, Z)), 0)
    assert [p.drawn for p in polygons(obj)] == [False, True, True, True]


# --- encode_object ------------------------------------------------------------------------------------------------

def _w(*vals: int) -> bytes:
    return struct.pack(f"<{len(vals)}I", *vals)


def _np_strip_with_padding() -> bytes:
    """Mode 0/1 layout with every link rule, a triangle whose skipped point words are not zero, and an end record whose
    three 'normal' words hold arbitrary data (the real ROM has e.g. 0x00040401 there)."""
    out = struct.pack("<6f", *S0, *S1)
    for attr, p0, p1 in ((0x001, A, B), (0x101 | 3 << 18, C, D), (0x201 | 0x1f << 27, E, F), (0x301, A, B)):
        out += _w(attr) + struct.pack("<9f", *N, *p0, *p1)
    out += _w(0x202) + struct.pack("<6f", *N, *C) + _w(0x3F10A0D7, 0xDEADBEEF, 0x00000001)   # triangle
    return out + _w(0x0000F000, 0x00040401, 0, 7)                     # end: attr & 3 == 0, then three words


@pytest.mark.parametrize("mode", [0, 1])
def test_encode_object_round_trip_synthetic(mode):
    words = _np_strip_with_padding() + _w(0x1, 0x2)                  # trailing words are not part of the object
    obj = decode_object(words, 0, mode=mode)
    assert [r.link for r in obj.records] == [0, 1, 2, 3, 2] and obj.word_count == 6 + 50 + 4
    enc = encode_object(obj)
    assert isinstance(enc, list) and all(isinstance(w, int) for w in enc)
    assert struct.pack(f"<{len(enc)}I", *enc) == words[:4 * obj.word_count]


def test_encode_object_round_trip_count_limited_and_normalless():
    words = _np_strip_with_padding()
    obj = decode_object(words, 0, count=3)                            # obc ran out: no end record
    assert struct.pack(f"<{obj.word_count}I", *encode_object(obj)) == words[:4 * obj.word_count]
    for mode in (2, 3):                                               # skipped normal words and the 1-word end kept
        nn = words[:-16] + _w(0x0000F000)
        obj = decode_object(nn, 0, mode=mode)
        assert struct.pack(f"<{obj.word_count}I", *encode_object(obj)) == nn


def test_encode_object_built_by_hand_uses_zero_padding():
    obj = Object(S0, S1, [Record(0x1, N, A, B), Record(0x2, N, C, None)], 6 + 20 + 4)
    words = struct.pack(f"<{obj.word_count}I", *encode_object(obj))
    assert words == strip(0, (0x1, N, A, B), (0x2, N, C, (0.0, 0.0, 0.0))) + _w(0, 0, 0, 0)
    back = decode_object(words, 0)
    assert back.records == obj.records and back.word_count == obj.word_count


@pytest.mark.parametrize("obj, match", [
    (Object(S0, S1, [Record(0x4, N, A, B)], 6 + 10 + 4), "end"),                 # attr & 3 == 0 inside the records
    (Object(S0, S1, [Record(0x1, N, A, None)], 6 + 10 + 4), "quad"),             # a quad needs its second point
    (Object(S0, S1, [Record(0x2, N, A, B)], 6 + 10 + 4), "triangle"),            # a triangle's second point is never read
    (Object(S0, S1, [Record(0x1, N, A, B), Record(0x1, None, C, D)], 6 + 20 + 4), "normal"),   # mixed layouts
    (Object(S0, S1, [Record(0x1, N, (1e40, 0.0, 0.0), B)], 6 + 10 + 4), "float"),
    (Object(S0, S1, [Record(1 << 32 | 1, N, A, B)], 6 + 10 + 4), "attr"),
    (Object(S0, S1, [Record(0x1, N, A, B)], 6 + 10 + 2), "word_count"),
    (Object(S0, (0.0, 1.0), [], 6 + 4), "point"),
])
def test_encode_object_refuses_what_cannot_be_encoded(obj, match):
    with pytest.raises(PolyError, match=match):
        encode_object(obj)


def test_encode_object_real_course_tiles(romset_zip):
    from dtools.arcade.course import check_rom_object, course_table, scenery
    from dtools.arcade.romset import RomSet
    rs = RomSet.from_zip(romset_zip)
    placements = scenery(rs, course_table(rs)[0], allow_unmodelled=True)
    seen = set()
    for p in placements:
        if p.oba in seen:
            continue
        seen.add(p.oba)
        index = check_rom_object(p.oba)
        obj = decode_object(rs.polygons, index, p.obc, mode=1)        # the race runs geometry mode 1
        enc = encode_object(obj)
        assert len(enc) == obj.word_count, hex(p.oba)
        assert struct.pack(f"<{len(enc)}I", *enc) == rs.polygons[4 * index:4 * (index + obj.word_count)], hex(p.oba)
    assert len(seen) >= 103
