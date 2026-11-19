"""M9, the arcade's three in reverse (ids 8-10; docs/superpowers/plans/2026-11-17-reverse-stage2-originals.md):
the map from a reverse road to its plain road, and the arcade's segment-keyed tables put onto the reverse road.

The Dreamcast's R file of course n holds the plain road backwards: its road point r is the plain road point
(c - r) mod L (L road points, c = `point_offset`), its lane j the plain lane LANES - 1 - j, exactly. So its segment s
(point s to s + 1, the last point to the first: the lap line) is the plain segment `plain_segment(c, L, s)` =
(c - 1 - s) mod L, a map that is its own inverse. The arcade keys some tables by the player's segment: the segment
flags (one u16 per segment), the progress masks (records over a segment range) and the tunnel light zones (zone
segment starts and ends, a lit range); `reversed_*` give them on the reverse road."""

from __future__ import annotations

import math
import struct

from dtools.arcade import course as C
from dtools.convert import ConvertError

LANE_MATCH = 1e-3           # units: an R lane and the plain lane it mirrors are the same point (they are exact)


def _lane(rp, j: int) -> tuple[float, float, float]:
    ln = rp.lanes[j]
    return float(ln.x), float(ln.y), float(ln.z)


def _matches(rp, pp) -> bool:
    """Every used lane j of road point `rp` is the plain road point `pp`'s used lane LANES - 1 - j."""
    for j in range(rp.lane_lo, rp.lane_hi + 1):
        m = C.LANES - 1 - j
        if not pp.lane_lo <= m <= pp.lane_hi or math.dist(_lane(rp, j), _lane(pp, m)) > LANE_MATCH:
            return False
    return True


def point_offset(plain, rev) -> int:
    """c such that every reverse road point r is the plain road point (c - r) mod L (`_matches`), the two CourseData
    with their lane ranges fixed (tables.with_lane_ranges_fixed). ConvertError when the roads' point counts differ,
    when no single plain point (or several) matches reverse point 0, or when a later point does not match."""
    L = len(plain.road)
    if len(rev.road) != L:
        raise ConvertError(f"reverse road: {len(rev.road)} road points, the plain road {L}")
    found = [p for p in range(L) if _matches(rev.road[0], plain.road[p])]
    if len(found) != 1:
        raise ConvertError(f"reverse road: no plain road point matches reverse road point 0" if not found else
                           f"reverse road: plain road points {found} all match reverse road point 0")
    c = found[0]
    for r in range(1, L):
        p = (c - r) % L
        if not _matches(rev.road[r], plain.road[p]):
            raise ConvertError(f"reverse road point {r}: its lanes mirrored are not plain road point {p}'s "
                               f"(offset {c})")
    return c


def plain_segment(c: int, L: int, s: int) -> int:
    """The plain segment of reverse segment `s` (and the reverse segment of plain segment `s`): (c - 1 - s) mod L."""
    return (c - 1 - s) % L


def reversed_flags(flags, c: int) -> tuple[int, ...]:
    """The plain road's per-segment flags (L entries) on the reverse road: entry s the plain entry plain_segment(s)."""
    L = len(flags)
    return tuple(int(flags[plain_segment(c, L, s)]) for s in range(L))


def reversed_masks(records, c: int, L: int) -> bytes:
    """Progress-mask records (course.PROGRESS_MASK_SIZE bytes: u16 first and last segment, then the row masks) on
    the reverse road, then the end record's two segment words (as mirror.mirror_masks): each record's segments
    [a, b] -> [plain_segment(b), plain_segment(a)], its rows kept, the records in ascending order (0x17688 walks them
    forward). ConvertError for a record that is not PROGRESS_MASK_SIZE bytes, or one whose reversed range would
    wrap the lap line."""
    out = []
    for r in records:
        if not isinstance(r, (bytes, bytearray)) or len(r) != C.PROGRESS_MASK_SIZE:
            raise ConvertError(f"progress-mask record {r!r:.40} is not {C.PROGRESS_MASK_SIZE} bytes")
        a, b = struct.unpack_from("<2H", r)
        lo, hi = plain_segment(c, L, b), plain_segment(c, L, a)
        if lo > hi:
            raise ConvertError(f"progress-mask record over segments {a}-{b}: reversed it would wrap the lap line "
                               f"({lo}-{hi})")
        out.append(struct.pack("<2H", lo, hi) + bytes(r[4:]))
    return b"".join(sorted(out)) + struct.pack("<2H", C.PROGRESS_MASK_END, C.PROGRESS_MASK_END)


def reversed_tunnel(starts, ends, rng, mask: int, positions: dict[int, bytes], c: int, L: int) -> tuple[bytes, ...]:
    """A tunnel light record's tables on the reverse road, packed as the record names them (zone starts, zone ends,
    range, mask, positions). The walk (0x7b1c) holds a zone index 0..K (K the record's count, TUNNEL_BYTES): it steps
    to z + 1 when the car's segment reaches starts[z + 1], back to z - 1 when the segment is at most ends[z - 1] (K + 1
    words each, starts[0] and ends[K] unread); 0x7b88 lights the car from positions[z] (12 bytes) while bit z of the
    mask is set and the segment is in the range [lo, hi). On the reverse road zone z' is K - z: starts'[j] =
    plain_segment(ends[K - j]) (j 1..K), ends'[i] = plain_segment(starts[K - i]) (i 0..K-1), starts'[0] and ends'[K] 0
    as the plain tables'; mask bit z' the plain bit K - z' (a plain bit past K, which the walk never reaches, dropped);
    positions'[z'] = positions[K - z'] for the lit zones (`positions`: lit plain zone -> its 12 bytes), zeros for the
    unlit, K entries; the range [lo, hi) -> [plain_segment(hi - 1), plain_segment(lo) + 1). ConvertError for starts and
    ends of unequal length, a lit zone with no position, a lit reverse zone K (past the K positions), or a range that
    would wrap the lap line."""
    K = len(starts) - 1
    if K < 0 or len(ends) != K + 1:
        raise ConvertError(f"tunnel zones: {len(starts)} starts, {len(ends)} ends (K + 1 each)")
    lit = [z for z in range(K + 1) if mask >> z & 1]
    if any(z not in positions for z in lit):
        raise ConvertError(f"tunnel zones: lit zones {lit}, positions for {sorted(positions)}")
    if 0 in lit and K:
        raise ConvertError("tunnel zones: plain zone 0 is lit, which would be reverse zone K, past its K positions")
    s2 = [0] + [plain_segment(c, L, ends[K - j]) for j in range(1, K + 1)]
    e2 = [plain_segment(c, L, starts[K - i]) for i in range(K)] + [0]
    lo, hi = rng
    lo2, hi2 = plain_segment(c, L, hi - 1), plain_segment(c, L, lo) + 1
    if lo2 >= hi2:
        raise ConvertError(f"tunnel range {lo}-{hi}: reversed it would wrap the lap line ({lo2}-{hi2})")
    mask2 = sum(1 << (K - z) for z in lit)
    pos2 = b"".join(bytes(positions[K - z]) if mask2 >> z & 1 else bytes(12) for z in range(K))
    return (struct.pack(f"<{K + 1}H", *s2), struct.pack(f"<{K + 1}H", *e2), struct.pack("<2H", lo2, hi2),
            struct.pack("<I", mask2), pos2)
