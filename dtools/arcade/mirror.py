"""The mirror of milestone M1c: one decoded course (`golden.Course`) mirrored in the plane z = -0.5 (`PLANE`: world
z -> -1 - z, so that the game's rounded cells mirror exactly), the result encoded into the set in place
(`mirrored_set`), or, since M4b, as a course id of its own beside the original (`mirrored_entry`: id + 8, its
changed tables in free space, the rest shared; docs/spike/2026-11-mirrors-live.md, "Mirrors of the originals").

The rules, per table: docs/formats/model2-course-tables.md, "What the mirror (M1c) must transform"; the ground's
layout and rules: docs/formats/model2-tgp-ground.md (sections 4 and 10). Each rule is one function here, so a rule
MAME contradicts is corrected in one place. What the offline tests measured: docs/spike/2026-11-mirror.md,
"Offline"; what the MAME runs checked and corrected (the lane, the road's attitude, the plane): "Rules checked".

- world points (`mirror_world`: road and AI points, actor positions, ground corners and centres, the points of
  objects drawn in world coordinates): (x, y, -1 - z); directions (stored normals) and object-local points
  (`mirror_point`): (x, y, -z);
- polygon objects (`golden.scenery_objects`, keyed by polygon-ROM index, so an object reached from several lists is
  mirrored once): seed points and points as world points, or as local points for the objects a STATIC_PROP actor
  draws under T(position) Ry(angle) (`_local_objects`: 2 of Beginner's 325), normals negated;
  corner order, attributes (bit 17 never set), the skipped and end words kept. Car models (`Course.car_objects`)
  are shared by every course: untouched;
- scenery cell grid: row r -> 15 - r (the row comes from z), each cell's list with it; object lists unchanged;
- actors (the modelled and the unmodelled records): position mirrored, angle -> -angle (`mirror_angle`); the game
  recomputes each actor's cell from its position. The unmodelled handlers' data is not decoded, so not mirrored:
  the objects 31 of Beginner's 35 actors draw (15 x 0x2201bc, 10 x 0x221700, 6 others) keep the original's local
  geometry (docs/spike/2026-11-mirror.md, "Not found, not covered");
- road line: positions mirrored, lane slot k -> 7 - k (`mirror_lane_slot`; lane_lo, lane_hi -> 7 - lane_hi,
  7 - lane_lo), each slot's pitch, heading and roll from slot 8 - k (`mirror_road_point`: the game reads them at
  slot (lane >> 8) + 1), heading h -> 0x2000 - h (`mirror_heading`), roll and curve negated (`mirror_signed`),
  pitch kept;
- AI lines: positions mirrored, lane -> 0x6ff - lane (`mirror_lane`: 7 - lane in the game's f / 255 blend),
  heading as the road's, roll and steer negated, pitch, speed and step kept;
- grid lanes: `mirror_lane` each (0x180, 0x580 -> 0x57f, 0x17f);
- the HUD map (`mirror_hud_map`): its outline objects as world objects (the cars' markers are drawn in the outline's
  frame at the cars' world positions, so they follow the mirrored cars); the list, place record and zooms kept;
- the ground (`mirror_ground`): corners and centres mirrored, the winding restored by a corner order (QUAD_ORDER,
  TRIANGLE_ORDER), direction word d -> 0x8000 - d (`mirror_direction`), grid row r -> 127 - r, lists kept;
- the view-angle billboards: variant k -> 63 - k (`mirror_billboard_lists`), their reference point BILLBOARD_REF
  mirrored as a world point;
- the backdrop (`mirror_backdrop`): panorama column q -> the original's column (MIRROR_COLUMN - q) & 255, each char
  flipped left to right into a char of its own in the same palette block; the mirrored chars written as new char
  blocks in the data ROM's free run (tgp.DATA_FREE), or on a custom set in a run of the custom data area the caller
  gives (`mirrored_set`'s `run`: the race-time uploader reads char blocks there, docs/spike/2026-11-mirrors-live.md,
  "Backdrops"); `mirrored_set` takes the run's first bytes that are free in its set; the char list pointed at them;
- unchanged: step table, checkpoint table, time tables, laps, car count, rolling start, grid base, spawn records,
  per-segment flags, switch lists, textures, select screen. The views (`Course` docstring) are recomputed from the
  mirrored sources: each grid slot's segment, lane and position (course.grid_slot), each checkpoint's position
  (course.checkpoint_views).

Every integer rule maps s16 to s16 and undoes itself, and a negation is exact in f32; -1 - z is exact but where it
rounds to f32 (within one ulp, `mirror_world`). So `mirror_course` is an involution up to those ulps, except the
road's lane_lo attitude (see `mirror_road_point`) and the backdrop's char blocks, which come back with the original's
chars in the free run's blocks, not at the original blocks' addresses.
"""

from __future__ import annotations

import dataclasses
import struct

import numpy as np

from dtools.arcade import course as C
from dtools.arcade import golden as G
from dtools.arcade import polyrom as P
from dtools.arcade import tgp
from dtools.arcade import tiles as TL
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import CUSTOM_AREAS, RomSet
from dtools.arcade.tablebase import DATA_BASE, DATA_SIZE, DATA_WINDOW, CourseError, _int, _seq, is_int, is_number


class MirrorError(DecodeError):
    pass


_S16 = (-0x8000, 0x7FFF)

QUAD_ORDER = (0, 3, 2, 1)
"""The mirrored polygon's corners, as indices of the original's: negating z flips the signed XZ area to positive,
and the TGP's edge test is one-sided (the originals' negative area), so the winding is reversed. This order keeps
corners 0 and 2, the diagonal function 0x36 splits a quad along (its height is the plane of 0-1-2 or of 0-2-3)."""

TRIANGLE_ORDER = (1, 0, 3, 2)
"""The order for a triangle (corner 3 = corner 2, every original triangle): QUAD_ORDER would repeat corners 1 and 2,
and the TGP's alpha / beta solve (corners 0, 1, 2, apex 2) would have no extent (the plane test reads 1/0). This
order keeps the repeat at corners 2-3 and corner 2 as the apex: the same cone as the original's (corners 0 and 1
swap) and the same degenerate plane 0-2-3 (docs/spike/2026-11-mirror.md, "Offline")."""

PLANE = -0.5
"""The mirror plane, z = -0.5: a world point's z becomes -1 - z (`mirror_world`). The game rounds world z to an
integer before it takes a cell (`tgp.cell`: (round(z) + 1024) >> 4 for the ground; 0x172a0: (round(z) + 1024) >> 7
for the scenery grid; 0x1770c: round(z) & ~0x7f for the view test's cell corner), and round(-1 - z) = -1 - round(z),
so a mirrored point lands in exactly the mirrored row on every grid, cell edges included (about z = 0 a point whose
z rounds onto a row edge lands one row off). The one exception is a z exactly halfway between integers, which both
sides round to even (an f32 z of exactly k + 0.5 on a row edge lands one row off). Directions (normals) and
object-local points keep the plain negation (`mirror_point`)."""


def _s16(v: int) -> int:
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


def _num(v, what: str) -> int:
    return _int(v, what, *_S16, None, MirrorError)


# --- the rules ---------------------------------------------------------------------------------------------------

def mirror_point(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """(x, y, -z): a direction (a stored normal) or an object-local point. MirrorError for anything but three finite
    numbers."""
    p = _seq(p, 3, "point", None, MirrorError)
    if not all(is_number(v) for v in p):
        raise MirrorError(f"point {p!r:.80} is not three finite numbers")
    return p[0], p[1], -p[2]


def mirror_world(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """(x, y, -1 - z), z rounded to f32: a point in world coordinates mirrored in PLANE (road and AI points, actor
    positions, ground corners and centres, the points of objects drawn in world coordinates). -1 - z is exact in f32
    unless z's last bits fall below the result's precision (|z| < 1, or z + 1 crossing a power of two), where it
    rounds by at most half a unit in the last place, so mirroring twice gives z back to within one ulp. MirrorError
    as `mirror_point`."""
    x, y, z = mirror_point(p)
    try:
        return x, y, struct.unpack("<f", struct.pack("<f", -1.0 + z))[0]
    except OverflowError as e:
        raise MirrorError(f"point {p!r:.80}: -1 - z is outside f32") from e


def mirror_heading(h: int) -> int:
    """A road / AI heading (s16; `4 h` is the yaw, 0x10000 per turn): 0x2000 - h, kept in s16 (mod 0x10000, so the
    yaw 4h becomes 0x8000 - 4h and (-sin, cos) of it is the mirrored direction of travel). The format doc writes
    `(0x2000 - h) & 0x3fff`; the stored headings are continuous s16 values spanning more than 0x4000 (Beginner
    -5462..10922, Advanced -17750..0), so the mask would make them jump between neighbours and would not undo
    itself; the yaw is the same."""
    return _s16(0x2000 - _num(h, "heading"))


def mirror_direction(d: int) -> int:
    """A ground polygon's direction word (sign-extended s16, 1/65536 turn): 0x8000 - d, as s16, the travel yaw's
    mirror. Read on Beginner: the standing cars take their yaw from it, and with 0x8000 - d they face the mirror's
    way (MAME, Phase 4 Task 5; docs/formats/model2-tgp-ground.md, section 10)."""
    return _s16(0x8000 - _num(d, "direction"))


def mirror_angle(a: int) -> int:
    """An actor's angle (s16 about y, 0x10000 per turn): -a (T(p) Ry(a) becomes T(mirror p) Ry(-a)); -0x8000, a half
    turn, stays."""
    return _s16(-_num(a, "angle"))


def mirror_signed(v: int) -> int:
    """A roll, curve or steer value (s16): -v (-0x8000 stays)."""
    return _s16(-_num(v, "value"))


def mirror_lane(lane: int) -> int:
    """A lateral position in lanes (8.8 fixed, k.f; AI lines, GRID_LANES): 0x6ff - lane, i.e. (6 - k).(255 - f).
    Lane numbers grow to the left of travel, which a mirror turns to the right, so the point at k + f / 255 (the
    game blends lanes k and k + 1 by f / 255, `course.lane_position`, 0x140f0) mirrors to 7 - k - f / 255 =
    (6 - k) + (255 - f) / 255: exact for every f (f = 0 maps to (6 - k).255, a whole lane, the same point), and an
    involution. That form also keeps the slot 0x140f0 takes the pitch, heading and roll from, (lane >> 8) + 1, on the
    mirror of the original's (`mirror_road_point`). The format doc's 0x700 - lane put every grid car 1/255 lane
    (0.012 units) off its mirrored place in MAME (docs/spike/2026-11-mirror.md, "Rules checked")."""
    return _num(0x6FF - _num(lane, "lane"), "mirrored lane")


def mirror_lane_slot(k: int) -> int:
    """A road point's lane slot (0..LANES-1): LANES - 1 - k."""
    return C.LANES - 1 - _int(k, "lane slot", 0, C.LANES - 1, None, MirrorError)


# --- one function per table kind ---------------------------------------------------------------------------------

def mirror_object(o: P.Object, world: bool = True) -> P.Object:
    """A polygon object mirrored: its seed points and every record's points by `mirror_world` (`world`, an object
    drawn in world coordinates: 323 of Beginner's 325 scenery objects) or `mirror_point` (an object drawn under a
    STATIC_PROP actor's T(position) Ry(angle): the other 2, `_local_objects`); its stored normals by `mirror_point`; corner order, attributes and the words the
    parser skips kept."""
    if not isinstance(o, P.Object):
        raise MirrorError(f"{o!r:.60} is not a polyrom.Object")
    at = mirror_world if world else mirror_point
    records = []
    for r in _seq(o.records, None, "records", "object", MirrorError):
        if not isinstance(r, P.Record):
            raise MirrorError(f"object record {r!r:.60} is not a polyrom.Record")
        records.append(dataclasses.replace(
            r, normal=None if r.normal is None else mirror_point(r.normal), p0=at(r.p0),
            p1=None if r.p1 is None else at(r.p1)))
    return dataclasses.replace(o, seed0=at(o.seed0), seed1=at(o.seed1), records=records)


def mirror_hud_map(m: C.HudMap) -> C.HudMap:
    """The race HUD's course map mirrored: every outline object by `mirror_object(o, world=True)` (the outline is
    drawn in world coordinates, the frame in which 0x13184-0x133a8 place the cars' markers by their world x and z:
    docs/spike/2026-11-mirror.md, "HUD map"); the list, the place record (translation, angle) and the zooms kept.
    On screen world z is the horizontal axis, so the map is flipped left to right about the screen x of z = -0.5."""
    if not isinstance(m, C.HudMap):
        raise MirrorError(f"{m!r:.60} is not a course.HudMap")
    if not isinstance(m.objects, dict):
        raise MirrorError(f"HUD map objects {m.objects!r:.60} is not a dict")
    return dataclasses.replace(m, objects={k: mirror_object(o, True) for k, o in m.objects.items()})


def mirror_billboard_lists(lists) -> tuple[int, ...]:
    """A view-angle billboard's BILLBOARD_VARIANTS object lists in the mirror's order: variant k becomes variant
    63 - k. The update 0x221e60 draws variant ((-a) >> 5) & 63, a the angle of the camera from BILLBOARD_REF
    (atan2(dz, dx), 0x10000 a turn); with the camera and the point mirrored (`mirror_world`) dz is negated, a becomes
    -a, and ((a) >> 5) & 63 = 63 - (((-a) >> 5) & 63) but where a is a multiple of 32. The variants' objects are
    mirrored with the scenery (world coordinates), so the mirror shows the mirrored variant the original shows from
    the mirrored camera (MAME: docs/spike/2026-11-mirror.md, "Billboards"). An involution. MirrorError for another
    count."""
    lists = _seq(lists, C.BILLBOARD_VARIANTS, "billboard lists", None, MirrorError)
    return tuple(reversed(lists))


MIRROR_COLUMN = 189
"""Column q of the mirrored backdrop panorama shows column (MIRROR_COLUMN - q) & 255 of the original, flipped left to
right (docs/spike/2026-11-mirror.md, "Backdrop"). The streamer 0x1dfd0 writes panorama column p (of 256, one turn)
to tile-layer 2 column p & 63, and the layer's scroll (tile RAM word 0x5002, from 0x501308 = (Y >> 5) & 0x1ff at
0x1dfa4; MAME segaic24.cpp: screen x s shows layer pixel (s - scroll) & 0x1ff) puts panorama pixel s - Y / 32 at
screen x s, where Y is 0x5fe11a, the camera yaw 0x5014a8 (0x10000 a turn; 0x4500 adds 0x50174c, the yaw on a single
cabinet at 0x6fe8; traced equal on every frame). The mirror's camera yaw is 0x8000 - Y (Task 5), so the mirror's
screen x s must show the original's screen x 495 - s flipped: the mirror's panorama pixel v is the original's
(495 - 1024 - v) mod 2048, i.e. pixel 7 - i of column (189 - q) & 255 for v = 8 q + i. 189 = 496 / 8 - 1 - 128 (the
screen's 62 columns, half a turn of 128): for another screen width there is no whole-column offset."""


def mirror_backdrop(b: C.Backdrop, chars: dict, free_char: int = 0x3000) -> tuple[C.Backdrop, dict[int, np.ndarray]]:
    """The backdrop mirrored: the panorama's column q is the original's column (MIRROR_COLUMN - q) & 255 with every
    tile's char flipped left to right. `chars` is the char RAM the backdrop's upload set fills (char index -> (8, 8)
    pixels, `Backdrop.chars`). System 24 tiles have no flip bit, so a flipped char is a char of its own, and a tile's
    palette follows from its char index (colour = (tile >> 7) & 0xff), so it must sit in the 128-char block of the
    char it flips. Per char of the panorama: a symmetric char is kept; a char whose flip is a char of `chars` in the
    same block takes that char (deduplicated); every other distinct flip (per block) is a new char, at the lowest free
    index of the block at or above `free_char` (free: not kept by a tile of the mirror; the uploaded indices first,
    then the block's others, below 0x4000), so with the default 0x3000 at most 4,096 new chars, and on Beginner each
    in the place of the char it flips. Tile bits 14 and 15 are kept. The fill tiles (`top_tile`, `bottom_tile`)
    follow from the mirrored segment 0.

    Returns the mirrored Backdrop (upload set unchanged) and the new chars (char index -> (8, 8) pixels), to be
    uploaded over `chars`; applied to the result and `{**chars, **new}`, it gives `b` and `chars` back. MirrorError
    for a malformed backdrop, a tile whose char `chars` lacks, chars that are not 8 x 8 pixels 0-15, or a block with
    no free index for its new chars."""
    if not isinstance(b, C.Backdrop):
        raise MirrorError(f"{b!r:.60} is not a course.Backdrop")
    segs = _seq(b.segments, TL.BACKDROP_SEGMENTS, "segments", "backdrop", MirrorError)
    for s in segs:
        if (not isinstance(s, TL.BackdropSegment) or s.cols != TL.BACKDROP_SEGMENT_COLS or s.rows != segs[0].rows
                or not isinstance(s.rows, int) or s.rows < 1 or len(_seq(s.tiles, None, "tiles", "backdrop segment",
                                                                         MirrorError)) != s.rows * s.cols):
            raise MirrorError(f"backdrop segment {s!r:.60}: not {TL.BACKDROP_SEGMENT_COLS} columns of "
                              f"{segs[0].rows!r} rows like segment 0")
        for t in s.tiles:
            _int(t, "tile", 0, 0xFFFF, "backdrop segment", MirrorError)
    if not isinstance(chars, dict):
        raise MirrorError(f"chars {chars!r:.60} is not a dict (char index -> 8 x 8 pixels)")
    for k, px in chars.items():
        if (not isinstance(k, int) or not 0 <= k < TL.CHAR_COUNT or not isinstance(px, np.ndarray)
                or px.shape != (8, 8) or not np.issubdtype(px.dtype, np.integer) or px.min() < 0 or px.max() > 15):
            raise MirrorError(f"char {k!r:.20}: not 8 x 8 pixels 0-15 at a char index below {TL.CHAR_COUNT:#x}")
    free_char = _int(free_char, "free_char", 0, TL.CHAR_COUNT, None, MirrorError)
    pan = b.panorama()[:, (MIRROR_COLUMN - np.arange(256)) & 255]

    def key(k: int, px: np.ndarray) -> tuple[int, bytes]:
        return k >> 7, np.ascontiguousarray(px, np.uint8).tobytes()
    held: dict[tuple[int, bytes], int] = {}
    for k in sorted(chars):
        held.setdefault(key(k, chars[k]), k)
    target: dict[int, int] = {}
    need: dict[tuple[int, bytes], list[int]] = {}
    for k in sorted({int(v) & TL.CHAR_MASK for v in np.unique(pan)}):
        if k not in chars:
            raise MirrorError(f"backdrop of course {b.course!r}: char {k:#x} is not in the backdrop's char RAM")
        flipped = C.flip_chars(chars[k])
        if np.array_equal(flipped, chars[k]):
            target[k] = k
        elif key(k, flipped) in held:
            target[k] = held[key(k, flipped)]
        else:
            need.setdefault(key(k, flipped), []).append(k)
    kept = set(target.values())
    new: dict[int, np.ndarray] = {}
    for (block, image), ks in sorted(need.items(), key=lambda kv: kv[1][0]):
        lo, hi = max(block << 7, free_char), min((block + 1) << 7, TL.CHAR_COUNT)
        slot = next((i for i in sorted(range(lo, hi), key=lambda i: (i not in chars, i))
                     if i not in kept and i not in new), None)
        if slot is None:
            raise MirrorError(f"backdrop of course {b.course!r}: no free char in block {block << 7:#x}-"
                              f"{((block + 1) << 7) - 1:#x} at or above {free_char:#x} for the flip of char {ks[0]:#x}")
        new[slot] = np.frombuffer(image, np.uint8).reshape(8, 8).copy()
        for k in ks:
            target[k] = slot
    out = (pan & 0xC000) | np.vectorize(lambda v: target[v & TL.CHAR_MASK], otypes=[np.uint16])(pan)
    cols = TL.BACKDROP_SEGMENT_COLS
    return b._replace(segments=tuple(s._replace(tiles=tuple(int(t) for t in out[:, cols * i:cols * (i + 1)].ravel()))
                                     for i, s in enumerate(segs))), new


def mirror_cells(cells) -> tuple[int, ...]:
    """The scenery cell grid (GRID_CELLS list pointers, cell = row << 4 | col, row from z): row r -> 15 - r."""
    cells = _seq(cells, C.GRID_CELLS, "cell grid", None, MirrorError)
    last = C.GRID_DIM - 1
    return tuple(cells[(last - (c >> 4)) << 4 | (c & last)] for c in range(C.GRID_CELLS))


def mirror_actor(a: C.Actor) -> C.Actor:
    if not isinstance(a, C.Actor):
        raise MirrorError(f"actor {a!r:.60} is not a course.Actor")
    return dataclasses.replace(a, position=mirror_world(a.position), angle=mirror_angle(a.angle))


def mirror_road_point(p: C.RoadPoint) -> C.RoadPoint:
    """A road point mirrored: lane slot k's position (and pad) is slot 7 - k's with z negated, lane_lo, lane_hi ->
    7 - lane_hi, 7 - lane_lo, curve negated. Its pitch, heading (`mirror_heading`) and roll (negated) are slot
    8 - k's, slot lane_hi's on the new lane_lo (slots outside the used range keep 7 - k's): 0x140f0 hands a car at
    lane k.f the attitude of slot k + 1 (clamped, not blended; 0x14334-0x14340), and its mirror (6 - k).(255 - f)
    (`mirror_lane`) reads slot 7 - k, which must hold the original's k + 1. MAME (Phase 4 Task 5) showed the
    reversal alone tilting the cars by the neighbour slot's pitch and roll (Beginner segment 161, lane 0x580:
    pitch -64, roll -3039 against -297, 4557 mirrored). Slot lane_lo's attitude is read only by a car below lane_lo
    (lanes clamp), whose mirror reads the new lane_hi, which holds lane_lo + 1's: no slot can serve both, so it is
    not kept and the second mirror gives lane_lo lane_lo + 1's attitude (a third equals the first)."""
    if not isinstance(p, C.RoadPoint):
        raise MirrorError(f"road point {p!r:.60} is not a course.RoadPoint")
    lanes = _seq(p.lanes, C.LANES, "lanes", "road point", MirrorError)
    for lane in lanes:
        if not isinstance(lane, C.Lane):
            raise MirrorError(f"road point lane {lane!r:.60} is not a course.Lane")
    lo, hi = (_int(v, n, 0, C.LANES - 1, "road point", MirrorError) for v, n in ((p.lane_lo, "lane_lo"),
                                                                                  (p.lane_hi, "lane_hi")))
    new_lo, new_hi = mirror_lane_slot(hi), mirror_lane_slot(lo)

    def slot(k: int) -> C.Lane:
        at = lanes[min(C.LANES - k, hi) if new_lo <= k <= new_hi else mirror_lane_slot(k)]
        return lanes[mirror_lane_slot(k)]._replace(
            position=mirror_world(lanes[mirror_lane_slot(k)].position), pitch=at.pitch,
            heading=mirror_heading(at.heading), roll=mirror_signed(at.roll))
    return dataclasses.replace(p, lane_lo=new_lo, lane_hi=new_hi, curve=mirror_signed(p.curve),
                               lanes=tuple(slot(k) for k in range(C.LANES)))


def mirror_ai_point(p: C.AiPoint) -> C.AiPoint:
    if not isinstance(p, C.AiPoint):
        raise MirrorError(f"AI point {p!r:.60} is not a course.AiPoint")
    return p._replace(position=mirror_world(p.position), heading=mirror_heading(p.heading),
                      roll=mirror_signed(p.roll), steer=mirror_signed(p.steer), lane=mirror_lane(p.lane))


def mirror_ground_poly(p: tgp.GroundPoly) -> tgp.GroundPoly:
    """Corners and centre mirrored (`mirror_world`) in QUAD_ORDER (TRIANGLE_ORDER when corner 3 = corner 2), the direction word
    mirrored; `raw` (the words as read) re-made from the mirrored polygon when the original has it, so a set
    written with it decodes to it. A mirrored polygon that the rounding of -1 - z leaves not convex gets the
    converter's "ulps" repair (`_rounding_convex`); every other polygon's corners are exactly the mirrored ones."""
    if not isinstance(p, tgp.GroundPoly):
        raise MirrorError(f"ground polygon {p!r:.60} is not a tgp.GroundPoly")
    corners = _seq(p.corners, 4, "corners", "ground polygon", MirrorError)
    order = TRIANGLE_ORDER if corners[2] == corners[3] else QUAD_ORDER
    q = dataclasses.replace(p, corners=_rounding_convex(tuple(mirror_world(corners[k]) for k in order)), raw=(),
                            centre=None if p.centre is None else mirror_world(p.centre),
                            direction=mirror_direction(p.direction))
    if p.raw:
        try:
            q = dataclasses.replace(q, raw=struct.unpack(f"<{tgp.POLY_WORDS}I", tgp._encode_poly(q, 0)))
        except tgp.TgpError as e:
            raise MirrorError(f"mirrored ground polygon: {e}") from e
    return q


def _rounding_convex(corners: tuple) -> tuple:
    """`corners` (a mirrored ground polygon's) as they are when tgp's encoder finds them convex (every corner's turn
    at most tgp.MIN_AREA: then nothing moves, the rule every mirror before Phase 10 Task 5 was built with). Else, when
    the notch is the rounding of the mirror's -1 - z (a corner on its neighbours' line within f32 that the rounding
    puts on the notch side: convert.ground's collinear case, at most NUDGE_ULPS ulps from the line), the converter's
    own "ulps" repair: the corner moved outward by the fewest f32 ulps (at most NUDGE_ULPS on each axis) that make the
    polygon convex (Mermaid Lake's cell 1123, itself repaired "ulps" by the converter, Phase 10 Task 2). MirrorError
    for a wider notch (no mirror makes one from a convex polygon but by rounding)."""
    from dtools.convert import ConvertError
    from dtools.convert import ground as GR
    if max(GR._turns(corners)) <= tgp.MIN_AREA:
        return corners
    what = "mirrored ground polygon"
    try:
        fixed, how = GR._convex(corners, what)
    except ConvertError as e:
        raise MirrorError(str(e)) from e
    if how != "ulps":
        raise MirrorError(f"{what}: not convex, and the notch is wider than the mirror's rounding ({how})")
    return fixed


def mirror_ground(g: tgp.Ground) -> tgp.Ground:
    """The ground mirrored: every polygon by `mirror_ground_poly` (same index, attributes unchanged, the road first in
    step order), the grid's rows r -> 127 - r, each cell with its list; the lists and the list block (`raw`) kept.
    Mirrored in PLANE, a point lands in exactly the mirrored row (`tgp.cell` rounds z: round(-1 - z) =
    -1 - round(z) but for exact halves), so permuting the rows is the whole rule, edges included, and
    `mirror_ground` is an involution (the corners to within an f32 ulp, `mirror_world`; a polygon the rounding left
    not convex within convert.ground.NUDGE_ULPS ulps, `_rounding_convex`). MirrorError for a ground
    `tgp.ground_tables` refuses, before or after."""
    if not isinstance(g, tgp.Ground):
        raise MirrorError(f"{g!r:.60} is not a tgp.Ground")
    try:
        tgp.ground_tables(g)
    except tgp.TgpError as e:
        raise MirrorError(f"ground of course {g.course!r}: {e}") from e
    rows = tgp.GRID_CELLS // 128
    grid = tuple(int(g.grid[(rows - 1 - (t >> 7)) * 128 + (t & 127)]) for t in range(tgp.GRID_CELLS))
    out = dataclasses.replace(g, grid=grid, polys=tuple(mirror_ground_poly(p) for p in g.polys))
    try:
        tgp.ground_tables(out)
    except tgp.TgpError as e:
        raise MirrorError(f"mirrored ground of course {g.course!r}: {e}") from e
    return out


# --- the course --------------------------------------------------------------------------------------------------

def mirror_course(c: G.Course, data_at: int | None = None, others=(), run=None) -> G.Course:
    """`c` with every rule of the module docstring applied: the sources mirrored, the views recomputed from them.
    The mirrored backdrop's char blocks are placed at data byte `data_at` of the data run `run` (`_backdrop`;
    default tgp.DATA_FREE, and the run's start); `others` are the other courses' backdrops (`_own_blocks`).

    MirrorError naming the fields when a table field is unset (`golden.unset_fields`: TABLE_FIELDS and FILLED_VIEWS
    at their default): `encode_course(m, rs)` fills such a field from the ROM, unmirrored, so a partial mirror would
    mix mirrored and unmirrored tables. billboards, billboard_ref and switch_lists may be empty for real (Expert,
    Advanced): no rule changes switch_lists, so the ROM's value that `complete` fills for an empty one is the
    mirrored value; the billboard rules change the other two, so they are refused when unset on a course with a
    VIEW_BILLBOARD actor (`_billboards`). MirrorError (naming the cause) for a
    Course `golden.check_course` refuses, before or after, and for malformed tables."""
    if not isinstance(c, G.Course):
        raise MirrorError(f"{c!r:.60} is not a golden.Course")
    name = c.entry.name if isinstance(c.entry, C.CourseEntry) else f"course {c.course!r}"
    if missing := G.unset_fields(c):
        raise MirrorError(f"{name}: unset fields {', '.join(missing)}: a partial Course cannot be mirrored "
                          f"(encode_course would fill them from the set, unmirrored; complete it first: "
                          f"golden.complete)")
    road = tuple(mirror_road_point(p) for p in _seq(c.road, None, "road", name, MirrorError))   # shapes first
    lanes = tuple(mirror_lane(v) for v in _seq(c.grid_lanes, 2, "grid lanes", name, MirrorError))
    try:
        G.check_course(c)
    except DecodeError as e:
        raise MirrorError(f"{name}: {e}") from e
    scenery = G.scenery_objects(c)          # keyed by polygon-ROM index: each mirrored once, however often listed
    local = _local_objects(c, name)
    objects = {k: mirror_object(o, k not in local) if k in scenery else o for k, o in c.objects.items()}
    out = dataclasses.replace(
        c, road=road, ai_lines=tuple(tuple(mirror_ai_point(p) for p in _seq(line, None, "AI line", name, MirrorError))
                                     for line in c.ai_lines),
        cells=mirror_cells(c.cells), grid_lanes=lanes, ground_data=mirror_ground(c.ground_data),
        actor_lists={a: tuple(mirror_actor(x) for x in _seq(v, None, "actor list", name, MirrorError))
                     for a, v in c.actor_lists.items()},
        unmodelled=tuple(mirror_actor(x) for x in _seq(c.unmodelled, None, "unmodelled", name, MirrorError)),
        objects=objects, hud_map=mirror_hud_map(c.hud_map),
        spawn=_grid_slots(c, road, lanes), checkpoints=_checkpoints(c, road),
        **_backdrop(c, name, data_at, others, run), **_billboards(c, name))
    try:
        G.check_course(out)
    except DecodeError as e:
        raise MirrorError(f"{name}: the mirrored Course: {e}") from e
    return out


def _billboards(c: G.Course, name: str) -> dict:
    """The billboards and billboard_ref fields of the mirror: every billboard's lists in `mirror_billboard_lists`'
    order, the reference point mirrored as a world point. Both are optional fields (empty on Expert and Advanced),
    but a rule changes them, so a Course with a VIEW_BILLBOARD actor whose lists or reference point are unset is
    refused, naming the field: `encode_course` would fill them from the set, unmirrored."""
    if not isinstance(c.billboards, dict):
        raise MirrorError(f"{name}: billboards {c.billboards!r:.60} is not a dict (array address -> lists)")
    objs = {a.obj for v in c.actor_lists.values() for a in v
            if isinstance(a, C.Actor) and a.handler == C.VIEW_BILLBOARD}
    unset = (["billboards"] if objs - set(c.billboards) else []) + (["billboard_ref"] if objs and c.billboard_ref
                                                                    is None else [])
    if unset:
        raise MirrorError(f"{name}: unset fields {', '.join(unset)} with view-angle billboards at "
                          f"{[hex(o) for o in sorted(objs)]}: the mirror reorders their variants and mirrors their "
                          f"reference point, and encode_course would fill them from the set, unmirrored")
    return {"billboards": {a: mirror_billboard_lists(v) for a, v in c.billboards.items()},
            "billboard_ref": None if c.billboard_ref is None else mirror_world(c.billboard_ref)}


def _own_blocks(c: G.Course, others=()) -> dict[int, int]:
    """The char blocks only the course's backdrop uploads (not BOOT_CHAR_LIST, not a screen's upload set): data-ROM
    offset -> size in bytes (the 4-byte count and 32 bytes per char). The mirror drops them from char_blocks, and
    `mirrored_set` may write the new blocks over them (a mirrored course mirrored again). `others` are the other
    courses' backdrops (`course.Backdrop`): MirrorError for an own block in a run the mirrored blocks are placed in
    (tgp.DATA_FREE, the custom data area: `_data_run`) that one of them also uploads, since neither course could then
    reuse or overwrite it alone."""
    screens = {x.src for x in c.boot_chars} | {x.src for _, u in c.screen_uploads for x in u.chars}
    own = {x.src - DATA_BASE: 4 + TL.CHAR_SIZE * len(c.char_blocks[x.src]) for x in c.backdrop.uploads.chars
           if x.src not in screens and x.src in c.char_blocks}
    named = set()
    for b in _seq(others, None, "other backdrops", c.name, MirrorError):
        if not isinstance(b, C.Backdrop):
            raise MirrorError(f"{c.name}: other backdrop {b!r:.40} is not a course.Backdrop")
        named.update(x.src - DATA_BASE for x in b.uploads.chars)
    runs = (tgp.DATA_FREE,) + CUSTOM_AREAS["data"]
    if shared := sorted(k for k in own if k in named and any(lo <= k < lo + n for lo, n in runs)):
        raise MirrorError(f"{c.name}: free-run char block(s) at data {[hex(k) for k in shared]} are also uploaded by "
                          f"another course's backdrop, so neither course owns them")
    return own


def _char_bytes(b: C.Backdrop) -> int:
    """The data-ROM bytes the mirrored backdrop's char blocks take: one block per upload (its 4-byte count and 32
    bytes per char)."""
    return sum(4 + TL.CHAR_SIZE * u.count for u in b.uploads.chars)


def _data_run(run, name: str) -> tuple[int, int, str]:
    """The data run (offset, end, name for messages) the mirrored char blocks go to: tgp.DATA_FREE for `run` None,
    else `run` = (data-ROM offset, length), a 4-byte-aligned, non-empty run inside tgp.DATA_FREE or the custom data
    area (romset.CUSTOM_AREAS["data"]: i960 0x03000000+, which the char uploader 0x1dd70 reads; the placement keeps
    out of romset.RESERVED there, as of every owned span: `_char_run`). MirrorError otherwise (a bool is not an
    int)."""
    if run is None:
        lo, n = tgp.DATA_FREE
        return lo, lo + n, f"the data ROM's free run (tgp.DATA_FREE, {lo:#x}-{lo + n:#x})"
    lo, n = (_int(v, f, 0, DATA_WINDOW, name, MirrorError)
             for v, f in zip(_seq(run, 2, "run (offset, length)", name, MirrorError), ("run offset", "run length")))
    if lo % 4 or not n or not any(a <= lo and lo + n <= a + m for a, m in (tgp.DATA_FREE,) + CUSTOM_AREAS["data"]):
        raise MirrorError(f"{name}: run {lo:#x}+{n:#x} is not a non-empty 4-byte-aligned run inside tgp.DATA_FREE or "
                          f"the custom data area ({', '.join(f'{a:#x}+{m:#x}' for a, m in CUSTOM_AREAS['data'])})")
    return lo, lo + n, f"the free run {lo:#x}-{lo + n:#x}"


def _backdrop(c: G.Course, name: str, data_at: int | None, others=(), region_run=None) -> dict:
    """The backdrop and char_blocks fields of the mirror: `mirror_backdrop` on the backdrop's char RAM, then the
    mirrored char RAM written as new char blocks in the data run `region_run` (`_data_run`: default the data ROM's
    free run tgp.DATA_FREE; on a custom set a run of the custom data area) from data byte `data_at` (default the
    run's start; `mirrored_set` passes the first bytes there that are free in its set), one per upload of the
    original (same destination and count: the char list keeps its length and place), the char list pointing at
    them. The original blocks stay in the ROM untouched (the M4 entry beside the original needs them) and leave
    char_blocks (the decoded set reaches only the new ones); the screens' blocks are kept. MirrorError when a new char
    falls outside the upload ranges, the run is not one (`_data_run`), or `data_at` is unaligned or the blocks from it
    do not fit the run (the message states the bytes needed and the bytes there)."""
    b = c.backdrop
    if not isinstance(b, C.Backdrop) or not isinstance(c.char_blocks, dict):
        raise MirrorError(f"{name}: backdrop {b!r:.40} / char_blocks are not a course.Backdrop and a dict")
    try:
        chars = b.chars(c.char_blocks)
    except CourseError as e:
        raise MirrorError(f"{name}: {e}") from e
    lo, hi, run = _data_run(region_run, name)
    data_at = lo if data_at is None else _int(data_at, "data_at", 0, DATA_SIZE if region_run is None else DATA_WINDOW,
                                              name, MirrorError)
    need = _char_bytes(b)
    if data_at % 4 or not lo <= data_at or data_at + need > hi:
        raise MirrorError(f"{name}: the mirrored backdrop's char blocks need {need} bytes from data {data_at:#x}; "
                          f"{run} has {max(0, hi - data_at) if lo <= data_at else 0} bytes from there"
                          f"{' (unaligned: 4-byte words)' if data_at % 4 else ''}")
    mb, new = mirror_backdrop(b, chars)
    ram = {**chars, **new}
    at = DATA_BASE + data_at
    blocks, uploads, covered = {}, [], set()
    for u in mb.uploads.chars:
        blocks[at] = np.stack([ram[u.first_char + k] for k in range(u.count)])
        uploads.append(u._replace(src=at))
        covered.update(range(u.first_char, u.first_char + u.count))
        at += 4 + TL.CHAR_SIZE * u.count
    if outside := sorted(set(new) - covered):
        raise MirrorError(f"{name}: new backdrop chars {[hex(k) for k in outside[:4]]} lie outside the upload ranges")
    old = {DATA_BASE + k for k in _own_blocks(c, others)}
    return {"backdrop": mb._replace(uploads=mb.uploads._replace(chars=tuple(uploads))),
            "char_blocks": {**{k: v for k, v in c.char_blocks.items() if k not in old}, **blocks}}


def _local_objects(c: G.Course, name: str) -> set[int]:
    """The objects drawn under an actor's T(position) Ry(angle), whose points are object-local: the objects of every
    STATIC_PROP actor's list (`course.scenery`, 0x220118: V * T(position) * Ry(angle)). Taken from the actors, the
    readers that decide how a list is drawn, not from `Placement.in_world` in `object_lists`: `golden.decode_course`
    reads every list with `course.object_list`, which sets in_world True, so the field there does not say how the
    game draws it. Every other object (cell lists, switch and billboard lists, spawn and map-object lists) is drawn in
    world coordinates. MirrorError for an object drawn both ways (one set of points cannot be both) or a STATIC_PROP
    actor whose list `object_lists` lacks. Objects are keyed by their polygon-ROM word index up to
    course.POLY_WINDOW (`course.rom_object_index`: the Course carries no set, and a converted course's objects lie in a
    custom set's polygon area past the original 16 MB, as Desert City's at id 3 do)."""
    local_lists = set()
    for actors in c.actor_lists.values():
        for a in _seq(actors, None, "actor list", name, MirrorError):
            if isinstance(a, C.Actor) and a.handler == C.STATIC_PROP:
                if a.obj not in c.object_lists:
                    raise MirrorError(f"{name}: actor at {a.addr!r} draws list {a.obj!r}, which object_lists lacks")
                local_lists.add(a.obj)
    world, local = set(), set()
    for ptr, places in c.object_lists.items():
        for p in _seq(places, None, "object list", name, MirrorError):
            if not isinstance(p, C.Placement):
                raise MirrorError(f"{name}: object list entry {p!r:.60} is not a course.Placement")
            (local if ptr in local_lists else world).add(C.rom_object_index(p.oba))     # no set: the custom area's too
    if both := world & local:
        raise MirrorError(f"{name}: objects {sorted(both)[:4]} are drawn both in world coordinates and under an "
                          f"actor: their points cannot be mirrored both ways")
    return local


def _grid_slots(c: G.Course, road, lanes) -> tuple[tuple[C.GridSlot, ...], ...]:
    """The spawn records as stored; their segment, lane and position (views) from the mirrored road and lanes."""
    return tuple(tuple(C.grid_slot(s.car, s.objects, s.line, s.start, s.speed, s.kind, list(road), c.grid_base[1],
                                   lanes, c.rolling_start[1]) for s in slots) for slots in c.spawn)


def _checkpoints(c: G.Course, road) -> tuple[C.Checkpoint, ...]:
    """The checkpoint rows as stored; their views (position, extensions) from the mirrored road line and the time
    table check_course reads them against."""
    key = (C.DEFAULT_MODE, C.DEFAULT_DIFFICULTY)
    if key not in c.time_tables or C.DEFAULT_MODE not in c.laps:
        raise MirrorError(f"{c.name}: the checkpoints' views need the time table of (mode, difficulty) {key} and the "
                          f"laps of mode {C.DEFAULT_MODE}: time_tables has {sorted(c.time_tables)!r:.60}, laps "
                          f"{sorted(c.laps)!r:.60}")
    rows = [(cp.segment, cp.raw) for cp in c.checkpoints if not cp.lap_line]
    return tuple(C.checkpoint_views(rows, list(road), c.time_tables[key][2], c.laps[C.DEFAULT_MODE][1]))


def _char_run(rs: RomSet, c: G.Course, others, run=None) -> np.ndarray:
    """Per byte of the data run `run` (`_data_run`: default the data ROM's free run tgp.DATA_FREE): True where the
    mirrored backdrop's char blocks may go: free (0xff) and owned by no table or block (golden.owned_spans, through
    the per-set cache golden.owned that the converter's allocator and tgp.write_free read: a byte a table owns is busy
    whatever its value, e.g. a new id's object lists in DATA_FREE, its texel block in the custom data area), or one of
    course `c`'s own char blocks (`_own_blocks`, which it rewrites). MirrorError (from DecodeError) when a course of
    the set does not decode, and for a run past the set's data image (a custom run on a stock set)."""
    lo, hi, name = _data_run(run, c.name)
    size = hi - lo
    if len(rs.data) < hi:
        raise MirrorError(f"{c.name}: {name} is past the set's data image ({len(rs.data):#x} bytes): a run of the "
                          f"custom data area is a custom set's")
    usable = np.frombuffer(rs.data, np.uint8, size, lo) == 0xFF
    try:
        rows = G.owned(rs).get("data")
    except DecodeError as e:
        raise MirrorError(f"{c.name}: the set's owned space (golden.owned_spans): {e}") from e
    if rows is not None:
        for a, z in rows[(rows[:, 1] > lo) & (rows[:, 0] < hi)]:
            usable[max(int(a), lo) - lo:min(int(z), hi) - lo] = False
    for off, n in _own_blocks(c, others).items():
        if off < hi and off + n > lo:
            usable[max(off, lo) - lo:min(off + n, hi) - lo] = True
    return usable



def _char_at(rs: RomSet, c: G.Course, others, run=None, data_at: int | None = None) -> int | None:
    """Where the mirrored backdrop of course `c` (decoded from `rs`) puts its char blocks in the data run `run`
    (`_data_run`): `data_at` checked, or by default the run's first 4-byte-aligned bytes long enough for them that are
    all usable (`_char_run`: free and owned by no table, or the course's own blocks). A `data_at` that is not an int
    inside the run is returned as it is (`_backdrop` refuses it). MirrorError when no such span exists or `data_at`'s
    span holds another byte."""
    lo, hi, name = _data_run(run, c.name)
    size = hi - lo
    usable = _char_run(rs, c, others, run)
    need = _char_bytes(c.backdrop)
    if data_at is None:
        bad = np.concatenate(([0], np.cumsum(~usable)))
        starts = np.arange(0, size - need + 1, 4)
        fits = starts[bad[starts + need] == bad[starts]] if need <= size else starts[:0]
        if not fits.size:
            raise MirrorError(f"{c.name}: no {need} bytes in {name} are free (0xff) or the course's own char blocks: "
                              f"{int(usable.sum())} such bytes in all")
        return lo + int(fits[0])
    if is_int(data_at) and lo <= data_at and data_at + need <= hi:
        span = usable[data_at - lo:data_at - lo + need]
        if not span.all():
            k = data_at + int(np.argmin(span))
            raise MirrorError(f"{c.name}: data {k:#x} (byte {rs.data[k]:#04x}) is not free (0xff) nor one of the "
                              f"course's own char blocks: the char blocks' target {data_at:#x}+{need} must be")
    return data_at


def mirrored_set(rs: RomSet, course: int = 0, data_at: int | None = None, *, run=None) -> RomSet:
    """`rs` with course `course` mirrored in place: `rs.with_patches(encode_course(mirror_course(decode_course(rs,
    course), data_at, others, run), rs))`.

    GRID_LANES (0x231b30) and BILLBOARD_REF (0x235bb4) are single values the game reads for every course: this build
    swaps GRID_LANES, so Expert and Advanced start on swapped grid sides in it (their tables are otherwise untouched),
    and mirrors BILLBOARD_REF (only Beginner draws a view-angle billboard). A per-course value is M4.

    The mirrored backdrop's char blocks go to the data run `run`: by default the data ROM's free run (tgp.DATA_FREE);
    on a custom set (`rs.custom`) a run of the custom data area may be given instead (`_data_run`; e.g. the new ids'
    build.ID_DATA_FREE: the race-time char uploader reads blocks at i960 0x03000000+, docs/spike/2026-11-mirrors-
    live.md, "Backdrops"). In the run: at `data_at`, or by default at the first 4-byte-aligned bytes long enough for
    them that are all free (0xff, as `tgp.write_free` requires, and owned by no table: golden.owned_spans) or the
    course's own current char blocks (`_own_blocks`: a mirrored course mirrored again writes over its first mirror's
    blocks; `_char_run`). So a second mirrored course, or a ground `tgp.relocate_ground` wrote to the run's start, is
    not overwritten. MirrorError when no such span exists, when `data_at`'s span holds another byte, for a run that is
    not one (`_data_run`) or a custom run on a stock set, and when the result does not decode to the mirrored
    ground."""
    if not isinstance(rs, RomSet):
        raise MirrorError(f"{rs!r:.60} is not a RomSet")
    c = G.decode_course(rs, course)
    lo, hi, name = _data_run(run, c.name)
    if not rs.custom and not (tgp.DATA_FREE[0] <= lo and hi <= tgp.DATA_FREE[0] + tgp.DATA_FREE[1]):
        raise MirrorError(f"{c.name}: {name} is a run of the custom data area; the set is a stock one (rs.custom "
                          f"False): its backdrop goes to tgp.DATA_FREE")
    others = [C.backdrop(rs, e) for e in C.course_table(rs) if e.course != c.course]
    data_at = _char_at(rs, c, others, run, data_at)
    m = mirror_course(c, data_at, others, run)  # MirrorError for a data_at outside the run (`_backdrop`)
    out = rs.with_patches(G.encode_course(m, rs))
    if tgp.decode_ground(out, course) != m.ground_data:
        raise MirrorError(f"{c.name}: the mirrored set does not decode to the mirrored ground")
    return out


# --- the mirror as a course id of its own (M4b, Phase 9 Task 3) ------------------------------------------------------

MIRROR_ID_OFFSET = 8
"""A mirror's course id is its original's + 8 (binding: docs/superpowers/plans/2026-11-09-phase9-mirrors-live.md)."""
MIRROR_ORIGINALS = (0, 1, 2, 3, 4, 7)
"""The originals `mirrored_entry` mirrors: Beginner, Expert, Advanced, the course at id 3 (Desert City, Phase 9 Task 4)
and, since Phase 10 Task 5, the courses at ids 4 and 7 (National Park Speedway and Mermaid Lake: mirrors 12 and 15).
A converted original (3 and up) must be on the set first."""

OWN_ARRAYS = frozenset({"COURSE_POINTERS", "GRID_TABLES", "ACTOR_TABLES", "HUD_MAP_LISTS"})
"""The per-id arrays whose entry of a mirror names a table of its own (`mirrored_entry`): its record, its cell grid
(the rows reversed, new object lists), its actor table (mirrored actors; an original with none shares the one empty
table, build.EMPTY_ACTORS, so the entry is the original's) and its HUD outline's list (the outline mirrored)."""
MIRRORED_ARRAYS = frozenset({"GRID_LANES", "BILLBOARD_REF", "CAMERA_SCRIPTS", "TUNNEL_LIGHTS", "PROGRESS_MASKS"})
"""The per-id arrays whose entry of a mirror holds the original's value mirrored (when the value has a geometric
part): the grid lanes (`mirror_lane`), the view-angle billboards' reference point (a world point, on a course with a
billboard), the pre-race camera script (`mirror_camera_record`: a new script), the tunnel light zones' positions
(`mirror_tunnel_lights`: a new position table, on a course with zones) and the progress-mask records (`mirror_masks`:
rows reversed, on a course with records). An original without the geometric part (no billboard, no zone, the empty
mask) keeps its entry, so the mirror's equals the original's."""
UNSHARED_ARRAYS = frozenset(C.SRAM_SINK_ARRAYS) | {"SELECT_PANEL_OF_COURSE"}
"""The per-id arrays whose entry of a mirror is not its original's: the SRAM words (course.SRAM_SINK_ARRAYS: the
rankings, the newest-entry word, the time-attack and best-lap words keep the guard's sink, so a mirror never writes its
original's SRAM: Review Focus 1; course.grow_rankings, later in the chain, gives each id its own ranking table and
newest-entry word) and the select's panel of course (the sixteen-entry select's inverse of its records,
select.patch_select's)."""
VISIT_OF_MIRROR = {0: 0, 1: 1, 2: C.VISIT_NONE, 3: 3, 4: 4, 7: 7}
"""course.VISIT_IDS of a mirror, by original: the original's own value for Beginner, Expert and ids 3, 4 and 7
(course.VISIT_DEFAULTS: id & 7, never Advanced's 2, so the general visit order at every cell); none for
Advanced (the controller's ruling: Advanced's second visit order is taken at its cells 72 and 88, which the mirror's
reversed rows put elsewhere, and the order only matters at the polygon cap: measured in MAME, discovery log, "Mirrors
of the originals")."""


def mirror_camera_record(raw: bytes) -> bytes:
    """One 64-byte camera keyframe (course.Keyframe's layout) mirrored: a point in a car's frame (car index not
    CAMERA_ABSOLUTE) as (-x, y, z) (the car's frame is T(position) R(attitude) with z its direction of travel; the
    mirrored car's yaw is 0x8000 - yaw, so a point at the car's left is at its right), a world point (CAMERA_ABSOLUTE)
    by `mirror_world`, the roll (+0x28) negated; the time, skip, cars, fov and the rest kept. A jump record
    (time CAMERA_JUMP) is returned as it is (`mirror_camera_script` repoints its target). MirrorError for a record
    that is not 64 bytes or a point that is not finite."""
    if not isinstance(raw, (bytes, bytearray)) or len(raw) != C.KEYFRAME_SIZE:
        raise MirrorError(f"camera keyframe {raw!r:.40} is not {C.KEYFRAME_SIZE} bytes")
    time, = struct.unpack_from("<i", raw, 0)
    if time == C.CAMERA_JUMP:
        return bytes(raw)
    out = bytearray(raw)
    for car_at, point_at in ((0x08, 0x0C), (0x18, 0x1C)):
        car, = struct.unpack_from("<i", raw, car_at)
        p = struct.unpack_from("<3f", raw, point_at)
        if not all(is_number(v) for v in p):
            raise MirrorError(f"camera keyframe: the point at +{point_at:#x} {p!r:.60} is not finite")
        struct.pack_into("<3f", out, point_at, *(mirror_world(p) if car == C.CAMERA_ABSOLUTE else (-p[0], p[1], p[2])))
    roll, = struct.unpack_from("<f", raw, 0x28)
    struct.pack_into("<f", out, 0x28, -roll)
    return bytes(out)


def _camera_records(rs: RomSet, start: int, name: str) -> tuple[list[tuple[int, bytes]], int]:
    """The camera script at `start` as played: (address, 64 bytes) per record in play order, jump records included,
    until a jump returns to a record already walked; and the index of that record (the loop). MirrorError as
    course.camera_script refuses the script (it walks it first)."""
    try:
        C.camera_script(rs, start)
        records, seen, at = [], {}, start
        while at not in seen:
            seen[at] = len(records)
            raw = rs.read_i960(at, C.KEYFRAME_SIZE)
            records.append((at, raw))
            time, _, target = struct.unpack_from("<iIi", raw, 0)
            at = target & 0xFFFFFFFF if time == C.CAMERA_JUMP else at + C.KEYFRAME_SIZE
    except DecodeError as e:
        raise MirrorError(f"{name}: camera script at {start:#x}: {e}") from e
    return records, seen[at]


def mirror_camera_script(rs: RomSet, start: int, at: int, name: str = "camera script") -> bytes:
    """The camera script at i960 `start` mirrored and laid out from i960 `at`: every record of its play
    (`_camera_records`, the shared loop it jumps to included) in play order, each keyframe by `mirror_camera_record`,
    each jump pointed at its target's new place, and a closing jump to the loop's new place when the walk ended on a
    plain keyframe. MirrorError as `_camera_records`."""
    records, loop = _camera_records(rs, start, name)
    index = {a: k for k, (a, _) in enumerate(records)}
    out = bytearray()
    for _, raw in records:
        rec = bytearray(mirror_camera_record(raw))
        time, = struct.unpack_from("<i", raw, 0)
        if time == C.CAMERA_JUMP:
            struct.pack_into("<I", rec, 8, at + C.KEYFRAME_SIZE * index[struct.unpack_from("<I", raw, 8)[0]])
        out += rec
    last = records[-1][1]
    if struct.unpack_from("<i", last, 0)[0] != C.CAMERA_JUMP:
        jump = bytearray(last)
        struct.pack_into("<iIi", jump, 0, C.CAMERA_JUMP, 0, at + C.KEYFRAME_SIZE * loop)
        out += jump
    return bytes(out)


TUNNEL_POSITION_SIZE = 12
"""A tunnel light zone's position: f32 x, y, z, world coordinates (0x7be4: zone * 3 words from the record's fifth
pointer, used only while the zone's bit of the mask, the fourth pointer's word, is set)."""


def mirror_tunnel_lights(rs: RomSet, record: bytes, count: int, name: str = "tunnel") -> bytes:
    """The position table of a tunnel light record (course.ID_ARRAYS["TUNNEL_LIGHTS"]: five pointers, the fourth to
    the zone mask word, the fifth to `count` positions) mirrored: every lit zone's position (its mask
    bit set) by `mirror_world`, the unlit ones (never read) kept. MirrorError for a record that is not five words or a
    position that is not finite."""
    if not isinstance(record, (bytes, bytearray)) or len(record) != 20:
        raise MirrorError(f"{name}: tunnel light record {record!r:.40} is not five words")
    *_, mask_at, positions = struct.unpack("<5I", record)
    try:
        mask = struct.unpack("<I", rs.read_i960(mask_at, 4))[0]
        table = rs.read_i960(positions, TUNNEL_POSITION_SIZE * count)
    except DecodeError as e:
        raise MirrorError(f"{name}: tunnel light zones: {e}") from e
    out = bytearray(table)
    for z in range(count):
        if mask >> z & 1:
            struct.pack_into("<3f", out, TUNNEL_POSITION_SIZE * z,
                             *mirror_world(struct.unpack_from("<3f", table, TUNNEL_POSITION_SIZE * z)))
    return bytes(out)


MASK_RECORD_SIZE = C.PROGRESS_MASK_SIZE
"""A progress-mask record (0x17688): u16 first and last segment, then a u16 column mask per scenery row (16 rows): the
cells of the 5 x 5 block it drops while the player's segment is in the range."""
MASK_END = C.PROGRESS_MASK_END
"""The last segment of the records' end: 0x17688 walks records while the player's segment is past a record's last
segment, so a record whose last segment is MASK_END ends the table (the original's is followed by code)."""


def mask_records(rs: RomSet, start: int, name: str = "progress masks") -> list[bytes]:
    """The progress-mask records at i960 `start` (a PROGRESS_MASKS entry) before the end record (last segment
    MASK_END): empty for Beginner's and Expert's entry. MirrorError past 256 records or for an unreadable table."""
    out, at = [], start
    try:
        while struct.unpack("<2H", rs.read_i960(at, 4))[1] < MASK_END:
            if len(out) >= 256:
                raise MirrorError(f"{name} at {start:#x}: no end record within 256 records")
            out.append(rs.read_i960(at, MASK_RECORD_SIZE))
            at += MASK_RECORD_SIZE
    except DecodeError as e:
        raise MirrorError(f"{name} at {start:#x}: {e}") from e
    return out


def mirror_masks(records) -> bytes:
    """Progress-mask records mirrored, then an end record: each record's row masks in reverse row order (row r -> 15 -
    r, the scenery grid's rule `mirror_cells`), its segments and column bits kept; the end record its two segment words
    (MASK_END, MASK_END) only, the 4 bytes 0x17688 reads of it (its last segment stops the walk, its first, above any
    segment, skips its rows; golden.extra_spans owns the same 4 bytes)."""
    out = bytearray()
    for r in _seq(records, None, "progress-mask records", None, MirrorError):
        if not isinstance(r, (bytes, bytearray)) or len(r) != MASK_RECORD_SIZE:
            raise MirrorError(f"progress-mask record {r!r:.40} is not {MASK_RECORD_SIZE} bytes")
        rows = struct.unpack_from(f"<{C.GRID_DIM}H", r, 4)
        out += r[:4] + struct.pack(f"<{C.GRID_DIM}H", *rows[::-1])
    return bytes(out + struct.pack("<2H", MASK_END, MASK_END))


def _layers(rs: RomSet) -> None:
    """CourseError naming the first layer `rs` lacks: grow_ids, guard_new_ids, widen_ids (in that order)."""
    for ok, layer in ((C.grown, "grown (course.grow_ids)"), (C.guarded, "guarded (course.guard_new_ids)"),
                      (C.widened, "widened (course.widen_ids)")):
        if not ok(rs):
            raise CourseError(f"a mirror needs a set that is {layer}: it is not")


def _alias(rs: RomSet, o: int, k: int) -> dict[tuple[str, int], bytes]:
    """The patches that make id `k` its original `o` under another id: o's record at k's, every per-id array entry of
    k o's but COURSE_POINTERS (k's record) and UNSHARED_ARRAYS, TGP slot k the slot words of o, TUNES[k] o's and
    VISIT_IDS[k] VISIT_OF_MIRROR[o]."""
    arrays = C.id_tables(rs)
    out: dict[tuple[str, int], bytes] = {}

    def copy(src: int, dst: int, n: int) -> None:
        out[C.rom_word_at(dst)] = rs.read_i960(src, n)

    copy(C.record_address(o), C.record_address(k), C.COURSE_ENTRY_SIZE)
    for name, a in C.ID_ARRAYS.items():
        if name != "COURSE_POINTERS" and name not in UNSHARED_ARRAYS:
            copy(C.id_entry(arrays, name, o), C.id_entry(arrays, name, k), a.stride)
    for w in (tgp.SLOT_LIST, tgp.SLOT_POOL):
        out[("copro_data", 4 * (w + k))] = rs.copro_data[4 * (w + o):4 * (w + o) + 4]
    copy(C.TUNES + o, C.TUNES + k, 1)
    out[C.rom_word_at(C.VISIT_IDS + k)] = bytes([VISIT_OF_MIRROR[o]])
    return out


def _own_words(rs: RomSet, k: int) -> list[tuple[str, int, int]]:
    """(region, offset, length) of the words that are id `k`'s own outside the free runs: its record, its entry of
    every per-id array, its TGP slot words, its TUNES and VISIT_IDS bytes."""
    arrays = C.id_tables(rs)
    out = [(*C.rom_word_at(C.record_address(k)), C.COURSE_ENTRY_SIZE)]
    out += [(*C.rom_word_at(C.id_entry(arrays, n, k)), a.stride) for n, a in C.ID_ARRAYS.items()]
    out += [("copro_data", 4 * (w + k), 4) for w in (tgp.SLOT_LIST, tgp.SLOT_POOL)]
    out += [(*C.rom_word_at(t + k), 1) for t in (C.TUNES, C.VISIT_IDS)]
    return out


class _Space:
    """The spans `mirrored_entry` takes from the free runs (build._first_fit: free, owned by nothing, not taken
    already), each with its bytes to write; one write per span (build._write_free)."""

    def __init__(self, rs: RomSet, name: str):
        self.rs, self.name, self.spans = rs, name, []

    def busy(self) -> list[tuple[str, int, int]]:
        return [(r, at, len(b)) for r, at, b in self.spans]

    def take(self, region: str, run: tuple[int, int], need: int, what: str, safe=None) -> int:
        from dtools.convert import ConvertError
        from dtools.convert import build as B
        try:
            if safe is None:
                at = B._first_fit(self.rs, region, run, need, f"{self.name}: {what}", self.busy())
            else:
                fits = B._fits(B._usable(self.rs, region, self.busy(), run), need)
                at = next((run[0] + int(f) for f in fits if safe(run[0] + int(f))), None)
                if at is None:
                    raise MirrorError(f"{self.name}: {what}: no {need} free bytes in the {region} run "
                                      f"{run[0]:#x}+{run[1]:#x} with no list or polygon across a bank window")
        except ConvertError as e:
            raise MirrorError(str(e)) from e
        self.spans.append((region, at, bytearray(b"\xff" * need)))
        return at

    def put(self, region: str, off: int, blob: bytes) -> bool:
        """Write `blob` into the span holding it: True, or False when no span holds all of it."""
        for r, at, buf in self.spans:
            if r == region and at <= off and off + len(blob) <= at + len(buf):
                buf[off - at:off - at + len(blob)] = blob
                return True
        return False


def _window(off: int) -> int:
    """The i960 program-window address of program ROM offset `off` (0x20000+)."""
    return C.PROGRAM_WINDOW[0] + off - 0x20000


def mirrored_entry(rs: RomSet, original: int) -> RomSet:
    """`rs` with the mirror of course `original` (MIRROR_ORIGINALS) as course id `original + 8` (MIRROR_ID_OFFSET),
    its own entry beside the original: `mirror_course` of the original, with every table the mirror changes written into
    free space (build._first_fit, build._write_free: free, owned by no table) and named by id `original + 8`'s own
    words. `rs` must be grown, guarded and widened (course.grow_ids, guard_new_ids, widen_ids; select.patch_select may
    come before or after; the course at id 3, 4 or 7 converted first for that `original`: a lifted course, build.LIFT,
    is mirrored as the set holds it, lifted).

    Shared with the original (the same bytes, named again): the texel block (COURSE_SHEETS), texture headers and
    points, the step table, checkpoint table, time tables, segment flags, laps, car counts, rolling start, grid base,
    the HUD map's place record and zooms, spawn records and car models, the backdrop's palette list, the unmodelled
    actors' handler data, the switch's three lists (program code: their objects stay the original's, and so does the
    switch actor, a known loss: discovery log, "Mirrors of the originals"), and every per-id array entry (`_alias`)
    but UNSHARED_ARRAYS. Its own: the record (COURSE_POINTERS), the ground (grid and attributes in
    build.ID_DATA_FREE, lists and polygons in the custom copro area, window-safe, TGP slot `original + 8`), road line,
    line table, cell grid (GRID_TABLES), every object list that names a scenery object (in tgp.DATA_FREE, the list
    home, with course.LIBRARY_BOUND_PATCH) and the objects (the custom polygon area), the actor table and lists
    (PROGRAM_FREE, the program window: ACTOR_TABLES; an original with no actor, a converted course, shares the one
    empty table instead, build.EMPTY_ACTORS, written by build.empty_actors if no course named it yet), the
    billboards' list arrays, the HUD outline's list and objects (HUD_MAP_LISTS), the backdrop's upload set, char list,
    segments and char blocks (build.ID_DATA_FREE), the grid lanes and billboard reference (mirrored, per id), and
    MIRRORED_ARRAYS' tables where the original has them (the camera script, the tunnel light positions, the progress
    masks: `mirror_camera_script`, `mirror_tunnel_lights`, `mirror_masks`). VISIT_IDS of the mirror VISIT_OF_MIRROR.

    CourseError for `rs` not a RomSet or missing a layer, `original` not one of MIRROR_ORIGINALS (bool refused), id
    `original + 8` filled already (golden.own_tables), or `original` 3, 4 or 7 on a set whose id holds no course of
    its own. MirrorError for a table that would change a byte outside the free runs and the mirror's own words, no room
    in a run, or a set that does not decode back to the mirror."""
    from dtools.convert import build as B
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    if not is_int(original) or int(original) not in MIRROR_ORIGINALS:
        raise CourseError(f"original {original!r:.40} is not one of {MIRROR_ORIGINALS} (a mirror is its original's "
                          f"id + {MIRROR_ID_OFFSET})")
    o = int(original)
    k = o + MIRROR_ID_OFFSET
    _layers(rs)
    if G.own_tables(rs, k):
        raise CourseError(f"course id {k}: its slot is filled already (its record or a per-id entry is not Beginner's "
                          f"copy): one mirror per original")
    if o >= C.COURSE_COUNT and not G.own_tables(rs, o):
        raise CourseError(f"course id {o} holds no course of its own (Beginner's copy): convert it first")
    rs1 = rs.with_patches(_alias(rs, o, k))
    base = G.decode_course(rs1, k)
    name = f"the mirror of {C.course_name(o)} (id {k})"
    others = [C.backdrop(rs1, e) for e in C.course_table(rs1) if e.course != k]
    run = B.ID_DATA_FREE
    char_at = _char_at(rs1, base, others, run)
    m = mirror_course(base, char_at, others, run)
    space = _Space(rs1, name)
    space.spans.append(("data", char_at, bytearray(b"\xff" * _char_bytes(base.backdrop))))
    arrays = C.id_tables(rs1)

    # what moves: the lists naming a scenery object (not the switch's: program code), their objects, the HUD outline
    scenery = set(G.scenery_objects(base))
    switch = set(base.switch_lists)
    moved = [p for p, places in m.object_lists.items()
             if p not in switch and any(C.check_rom_object(q.oba, rs1) in scenery for q in places)]
    objs = sorted({C.check_rom_object(q.oba, rs1) for p in moved for q in m.object_lists[p]} & scenery)
    hud = m.hud_map
    huds = sorted(hud.objects)
    words = {i: P.encode_object(m.objects[i]) for i in objs}
    words.update({("hud", i): P.encode_object(hud.objects[i]) for i in huds})
    poly_at = space.take("polygons", B.NEW_ID_RUNS["polygons"], 4 * sum(map(len, words.values())), "objects")
    index, at = {}, poly_at
    for key, w in words.items():
        index[key] = at // 4
        at += 4 * len(w)

    def oba(v: int, key) -> int:
        return (v & ~0x7FFFFF) | index[key]

    sizes = {p: C.OBJECT_ENTRY_SIZE * len(m.object_lists[p]) + 4 for p in moved}
    list_at = space.take("data", tgp.DATA_FREE, sum(sizes.values()) + C.OBJECT_ENTRY_SIZE * len(hud.entries) + 4,
                         "object lists")
    new_list, at = {}, list_at
    for p in moved:
        new_list[p] = DATA_BASE + at
        at += sizes[p]
    hud_list = DATA_BASE + at
    lists = {}
    for p, places in m.object_lists.items():
        if p in new_list:
            lists[new_list[p]] = tuple(dataclasses.replace(
                q, oba=oba(q.oba, C.check_rom_object(q.oba, rs1)) if C.check_rom_object(q.oba, rs1) in scenery
                else q.oba, entry=new_list[p] - DATA_BASE + C.OBJECT_ENTRY_SIZE * n) for n, q in enumerate(places))
        else:
            lists[p] = base.object_lists[p]
    kept = {C.check_rom_object(q.oba, rs1) for places in lists.values() for q in places} - {index[i] for i in objs}
    objects = {i: base.objects[i] for i in kept if i in base.objects}
    objects.update({index[i]: m.objects[i] for i in objs})
    hud = dataclasses.replace(
        hud, outline_list=hud_list,
        entries=tuple(dataclasses.replace(q, oba=oba(q.oba, ("hud", C.check_rom_object(q.oba, rs1))),
                                          entry=hud_list - DATA_BASE + C.OBJECT_ENTRY_SIZE * n)
                      for n, q in enumerate(hud.entries)),
        objects={index[("hud", i)]: hud.objects[i] for i in huds})

    # the actors (program window): the table, then the lists in table order; no actor, the shared empty table
    order = list(dict.fromkeys(s for s in m.actor_slots if s))
    new_actors = {}
    if order:
        prog_at = space.take("program", B.PROGRAM_FREE, 4 * C.GRID_CELLS
                             + sum(4 + C.ACTOR_SIZE * len(m.actor_lists[a]) for a in order), "actor table and lists")
        actor_table, at = _window(prog_at), prog_at + 4 * C.GRID_CELLS
        for a in order:
            new_actors[a] = _window(at)
            at += 4 + C.ACTOR_SIZE * len(m.actor_lists[a])
    else:
        actor_table = B.EMPTY_ACTORS

    # the data tables: the ground's grid and attributes, then the rest one after another
    g = m.ground_data
    attrs = len(g.attributes) + -len(g.attributes) % 4
    ground_at = space.take("data", run, 2 * tgp.GRID_CELLS + attrs, "ground grid and attributes")
    n_polys, block = len(g.polys), tgp.list_block_words(g.lists)
    copro_at = space.take("copro_data", B.NEW_ID_RUNS["copro_data"], 4 * (block + tgp.POLY_WORDS * n_polys),
                          "ground lists and polygons", lambda a: tgp.window_safe(g.lists, n_polys, a // 4, block))
    g = tgp.placed_ground(g, copro_at // 4, ground_at)
    u = m.backdrop.uploads
    try:
        blobs = {"ground pointers": bytes(8), "road line": C.encode_road_line(list(m.road)),
                 "line table": C.encode_line_table(0, [list(x) for x in m.ai_lines]), "cell grid": bytes(1024),
                 "upload set": TL.encode_upload_set(DATA_BASE, DATA_BASE, (DATA_BASE,) * len(u.segments)),
                 "char list": TL.encode_char_list([(x.src, x.dst) for x in u.chars])}
        for i, s in enumerate(m.backdrop.segments):
            blobs[f"segment {i}"] = TL.encode_backdrop_segment(s)
        for b in m.billboards:
            blobs[("billboard", b)] = bytes(4 * C.BILLBOARD_VARIANTS)
        extra = _extras(rs1, arrays, o, name)
        blobs.update({x: bytes(size) for x, (size, *_) in extra.items()})
    except CourseError as e:
        raise MirrorError(f"{name}: {e}") from e
    data_at = space.take("data", run, sum(len(v) + -len(v) % 4 for v in blobs.values()), "race tables")
    put, at = {}, data_at
    for x, v in blobs.items():
        put[x] = at
        at += len(v) + -len(v) % 4

    entry = dataclasses.replace(
        m.entry, upload_list=put["upload set"], block_ref_a=put["ground pointers"],
        block_ref_b=put["ground pointers"] + 4, line_table=put["line table"], record_table=put["road line"],
        raw=tgp.repoint_record_raw(m.entry.raw, DATA_BASE + base.ground_data.grid_at, DATA_BASE + g.grid_at))
    uploads = u._replace(addr=DATA_BASE + put["upload set"], char_list=DATA_BASE + put["char list"],
                         segments=tuple(DATA_BASE + put[f"segment {i}"] for i in range(len(u.segments))))
    bill = {b: DATA_BASE + put[("billboard", b)] for b in m.billboards}

    def actor(a: C.Actor, n: int, lst: int) -> C.Actor:
        src = a
        if a.handler == C.SWITCH:                   # the switch stays the original's (its lists are program code)
            src = base.actor_lists[lst][n]
        obj = (new_list.get(a.obj, a.obj) if a.handler == C.STATIC_PROP else bill[a.obj]
               if a.handler == C.VIEW_BILLBOARD else a.obj)
        return dataclasses.replace(src, addr=new_actors[lst] + 4 + C.ACTOR_SIZE * n, obj=obj)
    actor_lists = {new_actors[a]: tuple(actor(x, n, a) for n, x in enumerate(m.actor_lists[a])) for a in order}
    by_addr = {x.addr: (a, n) for a in order for n, x in enumerate(m.actor_lists[a])}
    unmodelled = tuple(actor_lists[new_actors[by_addr[x.addr][0]]][by_addr[x.addr][1]] for x in m.unmodelled)
    out_c = dataclasses.replace(
        m, entry=entry, backdrop=m.backdrop._replace(uploads=uploads), ground_data=g,
        ground=(DATA_BASE + g.grid_at, DATA_BASE + g.attributes_at), grid=DATA_BASE + put["cell grid"],
        cells=tuple(new_list.get(p, p) for p in m.cells), object_lists=lists, actor_table=actor_table,
        actor_slots=tuple(new_actors[s] if s else 0 for s in m.actor_slots), actor_lists=actor_lists,
        billboards={bill[b]: tuple(new_list.get(p, p) for p in v) for b, v in m.billboards.items()},
        unmodelled=unmodelled, objects=objects, hud_map=hud)

    # the bytes: every table into its span, or unchanged, or one of the id's own words
    own = _own_words(rs1, k)
    in_place: dict[tuple[str, int], bytes] = {}
    try:
        G.check_course(out_c)
        entries = G.tables(out_c)
    except DecodeError as e:
        raise MirrorError(f"{name}: {e}") from e
    for x, (_, make, entry_at, entry_of) in extra.items():
        entries.append(("data", put[x], make(DATA_BASE + put[x]), x))
        entries.append((*C.rom_word_at(entry_at), entry_of(DATA_BASE + put[x]), f"{x} entry"))
    shared = ("program", B.EMPTY_AT, B.EMPTY_TABLE, "actor table")    # written by build.empty_actors
    for region, off, blob, table in entries:
        if (region, off, blob, table) == shared and actor_table == B.EMPTY_ACTORS:
            continue
        if space.put(region, off, blob) or getattr(rs1, region)[off:off + len(blob)] == blob:
            continue
        if not any(r == region and a <= off and off + len(blob) <= a + n for r, a, n in own):
            raise MirrorError(f"{name}: the {table} at {region}+{off:#x} ({len(blob)} bytes) would change the set "
                              f"outside the free runs and the mirror's own words")
        in_place[(region, off)] = blob
    out = rs1
    try:
        for region, at, buf in space.spans:
            out = B._write_free(out, region, at, bytes(buf))
        if actor_table == B.EMPTY_ACTORS:
            out = B.empty_actors(out)
    except DecodeError as e:
        raise MirrorError(f"{name}: {e}") from e
    out = C.patch_library_bound(out.with_patches(in_place))
    got = G.decode_course(out, k)
    if differ := _decodes_differ(got, out_c):
        raise MirrorError(f"{name}: the written set decodes with a different {', '.join(differ)}")
    return out


def _decodes_differ(a: G.Course, b: G.Course) -> list[str]:
    """The TABLE_FIELDS, entry and record_pointer of two Courses that differ (by type and value, build._same); the
    texels compared by geometry (their order and the header each keeps follow the lists' walk, which a dict of lists
    does not fix)."""
    from dtools.convert import build as B

    def texels(c: G.Course) -> dict:
        return {G._texel_key(t.header): (t.indices, t.mips) for t in c.texels}
    return [f for f in G.TABLE_FIELDS + ("entry", "record_pointer")
            if not (B._same(texels(a), texels(b)) if f == "texels" else B._same(getattr(a, f), getattr(b, f)))]


def _extras(rs: RomSet, arrays: dict[str, int], o: int, name: str) -> dict:
    """MIRRORED_ARRAYS' tables the mirror of course `o` writes for itself, by name: (bytes, a function of the table's
    i960 address giving them, the i960 address of the mirror's array entry, a function of the table's address giving
    the entry). The camera script always (`mirror_camera_script`); the tunnel light positions when the original has
    zones (its TUNNEL_BYTES count: `mirror_tunnel_lights`, the entry the original's record with the fifth word
    repointed); the progress masks when it has records (Advanced: `mirror_masks`). Without them the mirror's entry
    stays the original's (`_alias`)."""
    k = o + MIRROR_ID_OFFSET

    def word(at: int) -> bytes:
        return struct.pack("<I", at)

    def entry(n: str) -> tuple[int, int]:
        return C.id_entry(arrays, n, o), C.id_entry(arrays, n, k)
    out = {}
    script = struct.unpack("<I", rs.read_i960(entry("CAMERA_SCRIPTS")[0], 4))[0]
    out["camera script"] = (len(mirror_camera_script(rs, script, DATA_BASE, name)),
                            lambda at: mirror_camera_script(rs, script, at, name), entry("CAMERA_SCRIPTS")[1], word)
    count = rs.read_i960(entry("TUNNEL_BYTES")[0], 1)[0]
    record = rs.read_i960(entry("TUNNEL_LIGHTS")[0], 20)
    if count:
        lights = mirror_tunnel_lights(rs, record, count, name)
        out["tunnel light positions"] = (len(lights), lambda at: lights, entry("TUNNEL_LIGHTS")[1],
                                         lambda at: record[:16] + word(at))
    masks = mask_records(rs, struct.unpack("<I", rs.read_i960(entry("PROGRESS_MASKS")[0], 4))[0], name)
    if masks:
        blob = mirror_masks(masks)
        out["progress masks"] = (len(blob), lambda at: blob, entry("PROGRESS_MASKS")[1], word)
    return out
