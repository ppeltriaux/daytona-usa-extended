"""Decode Model 2 polygon ROM / RAM objects into records and polygons.

Ports MAME 0.289 src/mame/sega/model2_v.cpp (BSD-3-Clause, the MAME team): the word layout read by the geometry
engine's object parsers geo_parse_np_ns / np_s / nn_ns / nn_s, and the strip rule of the rasterizer's
model2_3d_push / model2_3d_process_polygon. See docs/formats/model2-polygon-rom.md. `encode_object` is the inverse
of `decode_object`: the words the parsers skip (a triangle's second point, the normal in modes 2-3, the end record's
three words after its attribute in modes 0-1) are kept raw on `Record.skipped` and `Object.end`, so a decoded object
encodes back to the same words.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from dtools.arcade.tablebase import DecodeError, _f32, _seq, is_int

Point = tuple[float, float, float]


class PolyError(DecodeError):
    pass


@dataclass(frozen=True)
class Record:
    attr: int
    normal: Point | None           # None in modes 2-3 (the normal words are skipped)
    p0: Point
    p1: Point | None               # None for a triangle (its second point is skipped)
    # The raw words the parser skips in this record, in stream order: modes 2-3 the three normal words, then for a
    # triangle the three second-point words. Kept for encode_object; not part of equality.
    skipped: tuple[int, ...] = field(default=(), compare=False)

    @property
    def is_quad(self) -> bool:
        return bool(self.attr & 1)

    @property
    def ends(self) -> bool:
        return (self.attr & 3) == 0

    @property
    def link(self) -> int:
        return (self.attr >> 8) & 3

    @property
    def tex_param(self) -> int:
        return (self.attr >> 18) & 0x1F

    @property
    def coef(self) -> int:
        return self.attr >> 27


@dataclass(frozen=True)
class Object:
    seed0: Point
    seed1: Point
    records: list[Record]
    word_count: int                # words the parser advances over, end record included
    # The end record's words as read: (attr, w1, w2, w3) in modes 0-1, (attr,) in modes 2-3, () when the count ran
    # out first. Kept for encode_object; not part of equality.
    end: tuple[int, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class Poly:
    points: tuple[Point, ...]      # 3 or 4 corners in the rasterizer's order
    normal: Point | None
    attr: int

    @property
    def drawn(self) -> bool:
        """False for link 0: check_culling (model2_v.cpp 253-255) always culls it; only the link update runs."""
        return (self.attr >> 8) & 3 != 0


def decode_object(words: bytes, index: int, count: int = 0xFFFFF, mode: int = 0,
                  max_records: int = 65536) -> Object:
    """`index` is a word index into `words`, already masked by the caller; `mode` is the geometry mode & 3."""
    total = len(words) // 4
    pos = index

    def take(n: int, fmt: str) -> tuple:
        nonlocal pos
        if pos < 0 or pos + n > total:
            raise PolyError(f"object at word {index:#x} runs past the end of the buffer at word {pos:#x}")
        out = struct.unpack_from(f"<{n}{fmt}", words, pos * 4)
        pos += n
        return out

    has_normal = (mode & 3) < 2
    seed0, seed1 = take(3, "f"), take(3, "f")
    records: list[Record] = []
    end: tuple[int, ...] = ()
    for _ in range(count):
        attr = take(1, "I")[0]
        normal = take(3, "f") if has_normal else None      # np modes read it before testing attr & 3
        if attr & 3 == 0:
            # the three words after an np end attribute are read as a normal but are not one; keep them raw
            end = (attr, *struct.unpack_from("<3I", words, (pos - 3) * 4)) if has_normal else (attr,)
            break
        if len(records) >= max_records:
            raise PolyError(f"object at word {index:#x} exceeds {max_records} records")
        skipped: tuple[int, ...] = () if has_normal else take(3, "I")   # nn modes: `input += 3`
        p0 = take(3, "f")
        if attr & 1:
            p1 = take(3, "f")
        else:
            p1, skipped = None, skipped + take(3, "I")     # a triangle's second point: `input += 3`
        records.append(Record(attr, normal, p0, p1, skipped))
    return Object(seed0, seed1, records, pos - index, end)


def polygons(obj: Object) -> list[Poly]:
    """One polygon per record, as model2_3d_process_polygon builds it: corners (P1(n-1), P0(n-1), P0(n)[, P1(n)]),
    then the link bits pick which new points become the next record's P0(n-1) / P1(n-1). Link-0 polygons are
    emitted for their points but always culled (`Poly.drawn` is False): count drawn polygons only."""
    old_p0, old_p1 = obj.seed0, obj.seed1
    out: list[Poly] = []
    for r in obj.records:
        if r.is_quad:
            out.append(Poly((old_p1, old_p0, r.p0, r.p1), r.normal, r.attr))
            p1 = r.p1
        else:
            out.append(Poly((old_p1, old_p0, r.p0), r.normal, r.attr))
            p1 = r.p0                                      # a triangle's P1(n) is its P0(n)
        if r.link in (0, 2):
            old_p0, old_p1 = r.p0, p1
        elif r.link == 1:
            old_p1 = r.p0
        else:
            old_p0 = p1
    return out


def _point_words(p, what: str) -> list[int]:
    """The three words of a point: finite numbers that fit single precision (the shared number rule, tablebase)."""
    return list(struct.unpack("<3I", _f32(p, 3, "point", what, PolyError)))


def _raw_words(words, n: int, what: str) -> list[int]:
    words = list(_seq(words, None, "words", what, PolyError)) or [0] * n
    if len(words) != n or not all(is_int(w) and 0 <= w <= 0xFFFFFFFF for w in words):
        raise PolyError(f"{what}: expected {n} u32 words, got {tuple(words)!r}")
    return [int(w) for w in words]


def encode_object(obj: Object) -> list[int]:
    """The 32-bit words `decode_object` reads for `obj` (the inverse for the same mode): seeds, ten words per record,
    then the end record (four words in modes 0-1, one in modes 2-3, none when `word_count` leaves no room for it).
    The layout follows the records: normals present means modes 0-1, absent means modes 2-3. Skipped words come from
    `Record.skipped` and `Object.end`, zero (end attribute 0) when they are empty. PolyError for a record that cannot
    be encoded: an attribute that is not a u32 or would end the object (attr & 3 == 0), a quad without its second
    point, a triangle with one (the parser never reads it), records of both layouts, a point that is not three
    finite numbers that fit single precision (NaN and infinity refused), or a `word_count` that does not match."""
    recs = _seq(obj.records, None, "records", "object", PolyError)
    if not is_int(obj.word_count):
        raise PolyError(f"word_count {obj.word_count!r} is not an integer")
    layouts = {r.normal is not None for r in recs}
    if len(layouts) > 1:
        raise PolyError("records mix the normal-present (modes 0-1) and normal-less (modes 2-3) layouts")
    tail = obj.word_count - 6 - 10 * len(recs)
    has_normal = layouts.pop() if recs else (tail != 1)
    end_len = 4 if has_normal else 1
    if tail not in (0, end_len):
        raise PolyError(f"word_count {obj.word_count} does not fit 6 + 10 x {len(recs)} records + an end record of "
                        f"{end_len} words (or none)")
    out = _point_words(obj.seed0, "seed point 0") + _point_words(obj.seed1, "seed point 1")
    for k, r in enumerate(recs):
        what = f"record {k}"
        if not is_int(r.attr) or not 0 <= r.attr <= 0xFFFFFFFF:
            raise PolyError(f"{what}: attr {r.attr!r} is not a u32")
        if r.attr & 3 == 0:
            raise PolyError(f"{what}: attr {r.attr:#x} has attr & 3 == 0, which the parser reads as the end record")
        quad = bool(r.attr & 1)
        skipped = _raw_words(r.skipped, (0 if has_normal else 3) + (0 if quad else 3), f"{what} skipped words")
        out.append(int(r.attr))
        if has_normal:
            out += _point_words(r.normal, f"{what} normal")
        else:
            out += skipped[:3]
            skipped = skipped[3:]
        out += _point_words(r.p0, f"{what} point 0")
        if quad:
            if r.p1 is None:
                raise PolyError(f"{what}: a quad (attr & 1) needs its second point")
            out += _point_words(r.p1, f"{what} point 1")
        else:
            if r.p1 is not None:
                raise PolyError(f"{what}: a triangle's second point is never read by the parser; pass p1=None")
            out += skipped
    if tail:
        end = _raw_words(obj.end, end_len, "end record")
        if end[0] & 3:
            raise PolyError(f"end record: attr {end[0]:#x} has attr & 3 != 0")
        out += end
    elif obj.end:
        raise PolyError(f"end record {obj.end!r} given but word_count {obj.word_count} leaves no room for it")
    return out
