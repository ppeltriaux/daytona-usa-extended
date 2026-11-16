"""The Dreamcast executable `1ST_READ.BIN`: its checkpoint table, a lossless reader and encoder; its pit stalls, a
reader.

The executable loads at `EXE_BASE` (address = file offset + 0x8C010000). The checkpoint table (docs/formats/
dreamcast-disc.md, "Executable tables"; found by searching for the arcade's checkpoint segments, Phase 5) is:

- 32 u32 pointers at file offset `CHECKPOINT_POINTERS` (0x17e99c, address 0x8c18e99c), indexed by entry
  `e = 8 * VARIANTS.index(v) + n` like the executable's three other per-course tables (debug labels, course files,
  ghosts). Entries 16-31 (M, MR) hold the same addresses as 0-15.
- the lists they point at, packed directly before the pointers (0x17e8ec-0x17e99b on the disc, 16 lists in entry
  order 0-15): each a u32 count, then that many u32 road segments, the next list straight after. The first word is
  a count: every pointer is the one before it plus 4 * (1 + its first word), and the last list ends exactly at the
  pointer table. The segments' high halves are all 0 (the arcade's entries keep a raw high u16 there,
  `course.Checkpoint.raw`; here the word is read whole).

What each word means was confirmed by its values, not traced in the running game. The basis, by variant:

- `""`: courses 0-2's lists are the arcade's checkpoint segments value for value (0 Beginner none, 1 Expert, 2
  Advanced; golden.decode_course), so a list is the checkpoints in race order, the lap line not included.
- `R`: each list is the plain list run backwards, segment s becoming `segments - s` on the R data file's segment
  count (all 8 courses); a reversed course passes the same places in the other order.
- `M`, `MR`: the entries point at the `""` and `R` lists (a mirror keeps the segment numbering).
- Which entry the game reads was not traced: the code at 0x8c035a42 loads the pointer table's address (a literal
  at 0x8c035aa8) and indexes it by a byte at 0x8c292e98, masked, times 4 (static reading of the SH-4 code).

Every malformed input is a `DcError` naming the entry or the list and the offset; the encoder refuses what the
layout cannot hold, so whatever decodes re-encodes byte for byte.

The pit stalls (v0.1.2; found by searching for the arcade's, course.pit_stalls, by their speeds 150, 100, 60): the
waypoints the pit stop steers the player's car through, the arcade's CAR_TABLES_230E50 lists without the pad word.

- `PIT_GROUPS` x `PIT_STALLS` u32 pointers at file offset `PIT_POINTERS` (0x17e2ec, address 0x8c18e2ec): group g's
  stall k (one per car ident, a linked cabinet's) at index PIT_STALLS * g + k.
- each pointer names `PIT_WAYPOINTS` records of x, z, speed f32 (12 bytes), packed in pointer order up to the
  pointer table (0x17ba3c-0x17e2eb on the disc).
- the groups go in pairs, a course's plain stalls then its R stalls (the same lane run the other way), the pairs in
  the order Beginner, Advanced, Expert, then courses 3-7 (`PIT_PAIRS[n]` is course n's pair; Expert is n 1).

Confirmed by value, not traced in the running game: the plain groups of courses 0-2 are the arcade's stalls value for
value, in the arcade's coordinates (the courses' x, z as the converter writes them: the data files' z negated); the
plain groups of 3-7 lie on the converted courses' pit lanes and run with the race, the R groups against it
(tests/test_release.py `test_release_pit_stalls`). A few stalls repeat others (National Park Speedway's 4-7 are 0-3)
and one ends at (0, 0) (Desert City's stall 4, after its stop): kept as the disc has them."""

from __future__ import annotations

import struct
import math
from dataclasses import dataclass
from typing import NamedTuple

from dtools.arcade.tablebase import _int, _seq
from dtools.dc.disc import COURSE_COUNT, VARIANTS
from dtools.dc.errors import DcError

EXE_NAME = "1ST_READ.BIN"
EXE_BASE = 0x8C010000               # load address: address = file offset + EXE_BASE
ENTRIES = 32                        # 8 courses x 4 variants: entry e = 8 * VARIANTS.index(v) + n
CHECKPOINT_POINTERS = 0x17E99C      # file offset of the 32 u32 pointers (address 0x8c18e99c)
CHECKPOINT_END = CHECKPOINT_POINTERS + 4 * ENTRIES
PIT_POINTERS = 0x17E2EC             # file offset of the pit stalls' u32 pointers (address 0x8c18e2ec)
PIT_GROUPS = 16                     # 8 courses x (plain, R)
PIT_STALLS = 8                      # per group: one per car ident (car +0x4 & 7 on the arcade)
PIT_WAYPOINTS = 7                   # per stall
PIT_RECORD = 12                     # x, z, speed f32
PIT_END = PIT_POINTERS + 4 * PIT_GROUPS * PIT_STALLS
PIT_PAIRS = (0, 2, 1, 3, 4, 5, 6, 7)
"""Course n's pair of pit groups (plain 2 * PIT_PAIRS[n], R the next): Beginner, Advanced, Expert, then 3-7."""


@dataclass(frozen=True)
class CheckpointTable:
    """The checkpoint table. `lists`: the stored lists in address order, each the road segments of one list (the
    lap line not included); `entries`: ENTRIES indices into `lists`, entry e's list. On the disc 16 lists and
    entries (0, 1, ..., 15) twice. Encoded, the lists are packed from the first and the pointers follow them."""
    lists: tuple[tuple[int, ...], ...]
    entries: tuple[int, ...]


class PitWaypoint(NamedTuple):
    """One waypoint of a pit stall: where the car steers (x, z, world units) and the speed it is set to there."""
    x: float
    z: float
    speed: float


def _bytes(blob) -> bytes | bytearray:
    if isinstance(blob, (bytes, bytearray)):
        return blob
    if isinstance(blob, memoryview):
        return blob.tobytes()
    raise DcError(f"{EXE_NAME}: {type(blob).__name__} is not bytes")


def decode_checkpoint_table(exe) -> CheckpointTable:
    """The checkpoint table of `exe` (the bytes of 1ST_READ.BIN). DcError for a file too short for the pointer table,
    a pointer that is not a 4-aligned address before the pointer table (past the end of the file, before the load
    address), a list whose count runs past the pointer table, or a pointer that does not land on a list's start."""
    b = _bytes(exe)
    if len(b) < CHECKPOINT_END:
        raise DcError(f"{EXE_NAME}: the file is {len(b):#x} bytes; the checkpoint pointer table at "
                      f"{CHECKPOINT_POINTERS:#x} needs {CHECKPOINT_END:#x}")
    offsets = []
    for e, p in enumerate(struct.unpack_from(f"<{ENTRIES}I", b, CHECKPOINT_POINTERS)):
        off = p - EXE_BASE
        if not 0 <= off < CHECKPOINT_POINTERS or off % 4:
            raise DcError(f"{EXE_NAME}: checkpoint entry {e}: pointer {p:#x} (at offset "
                          f"{CHECKPOINT_POINTERS + 4 * e:#x}) is not a 4-aligned address of a list before the "
                          f"pointer table ({EXE_BASE:#x}..{EXE_BASE + CHECKPOINT_POINTERS - 4:#x})")
        offsets.append(off)
    lists: list[tuple[int, ...]] = []
    starts: dict[int, int] = {}
    pos = min(offsets)
    while pos < CHECKPOINT_POINTERS:
        count = struct.unpack_from("<I", b, pos)[0]
        end = pos + 4 + 4 * count
        if end > CHECKPOINT_POINTERS:
            raise DcError(f"{EXE_NAME}: checkpoint list at {pos:#x} (address {pos + EXE_BASE:#x}): count {count} runs "
                          f"to {end:#x}, past the pointer table at {CHECKPOINT_POINTERS:#x}")
        starts[pos] = len(lists)
        lists.append(struct.unpack_from(f"<{count}I", b, pos + 4))
        pos = end
    entries = []
    for e, off in enumerate(offsets):
        if off not in starts:
            raise DcError(f"{EXE_NAME}: checkpoint entry {e}: pointer {off + EXE_BASE:#x} is not the start of a list "
                          f"(the lists from {min(offsets):#x} start at "
                          f"{', '.join(f'{s:#x}' for s in sorted(starts)[:8])}{' ...' if len(starts) > 8 else ''})")
        entries.append(starts[off])
    return CheckpointTable(tuple(lists), tuple(entries))


def encode_checkpoint_table(t: CheckpointTable) -> bytes:
    """The lists (packed, in order) then the ENTRIES pointers: the bytes from the first list to CHECKPOINT_END, which
    belong at file offset CHECKPOINT_END - len(result). DcError for a segment outside u32, an entry that names no
    list, a first list that no entry names (the decoder starts at the lowest pointer) or lists too long for the
    bytes before the pointer table."""
    if not isinstance(t, CheckpointTable):
        raise DcError(f"checkpoint table: {type(t).__name__} is not a CheckpointTable")
    lists = _seq(t.lists, None, "lists", "checkpoint table", DcError)
    entries = _seq(t.entries, ENTRIES, "entries", "checkpoint table", DcError)
    idx = tuple(_int(x, f"entry {e}: list", 0, len(lists) - 1, "checkpoint table", DcError)
                for e, x in enumerate(entries))
    if lists and 0 not in idx:
        raise DcError("checkpoint table: list 0 is named by no entry (the decoder starts at the lowest pointer)")
    body = bytearray()
    starts = []
    for k, c in enumerate(lists):
        segs = _seq(c, None, f"list {k}", "checkpoint table", DcError)
        starts.append(len(body))
        body += struct.pack("<I", len(segs))
        body += b"".join(struct.pack("<I", _int(s, f"list {k}: segment {i}", 0, 0xFFFFFFFF, "checkpoint table",
                                                DcError)) for i, s in enumerate(segs))
    first = CHECKPOINT_POINTERS - len(body)
    if first < 0:
        raise DcError(f"checkpoint table: the lists need {len(body):#x} bytes, more than the {CHECKPOINT_POINTERS:#x} "
                      f"bytes before the pointer table")
    return bytes(body) + struct.pack(f"<{ENTRIES}I", *(EXE_BASE + first + starts[k] for k in idx))


def checkpoints(exe, n: int, v: str = "") -> tuple[int, ...]:
    """The checkpoints of course n (0-7), variant v (one of VARIANTS) in `exe` (the bytes of 1ST_READ.BIN): road
    segments in race order, the lap line not included; () for a course without checkpoints (n 0 and 5). All four
    variants are read; their basis is in the module docstring (`""` against the arcade, `R` against the R data
    files, `M` and `MR` the same lists). DcError for an n or v outside those, or a table `decode_checkpoint_table`
    refuses."""
    n = _int(n, "course", 0, COURSE_COUNT - 1, "checkpoints", DcError)
    if not isinstance(v, str) or v not in VARIANTS:
        raise DcError(f"checkpoints: variant {v!r} is not one of {VARIANTS}")
    t = decode_checkpoint_table(exe)
    return t.lists[t.entries[8 * VARIANTS.index(v) + n]]


def verify_lossless_checkpoints(disc) -> list[str]:
    """One line if `encode_checkpoint_table(decode_checkpoint_table(1ST_READ.BIN))` does not reproduce the file's bytes
    from the first list to the end of the pointer table, or if the table does not decode ([] when lossless)."""
    blob = disc.read(EXE_NAME)
    try:
        back = encode_checkpoint_table(decode_checkpoint_table(blob))
    except DcError as exc:
        return [str(exc)]
    have = blob[CHECKPOINT_END - len(back):CHECKPOINT_END]
    if have != back:
        k = next(k for k, (a, c) in enumerate(zip(have, back)) if a != c)
        return [f"{EXE_NAME}: checkpoint table differs at offset {CHECKPOINT_END - len(back) + k:#x}"]
    return []


def decode_pit_stalls(exe) -> tuple[tuple[tuple[PitWaypoint, ...], ...], ...]:
    """The pit stalls of `exe` (the bytes of 1ST_READ.BIN): PIT_GROUPS groups of PIT_STALLS stalls of PIT_WAYPOINTS
    waypoints. DcError for a file too short for the pointer table, a pointer that is not a 4-aligned address whose
    PIT_WAYPOINTS records end at or before the pointer table, or a value that is not finite."""
    b = _bytes(exe)
    if len(b) < PIT_END:
        raise DcError(f"{EXE_NAME}: the file is {len(b):#x} bytes; the pit pointer table at {PIT_POINTERS:#x} needs "
                      f"{PIT_END:#x}")
    need = PIT_WAYPOINTS * PIT_RECORD
    groups = []
    for g in range(PIT_GROUPS):
        stalls = []
        for k in range(PIT_STALLS):
            at = PIT_POINTERS + 4 * (PIT_STALLS * g + k)
            p = struct.unpack_from("<I", b, at)[0]
            off = p - EXE_BASE
            if not 0 <= off <= PIT_POINTERS - need or off % 4:
                raise DcError(f"{EXE_NAME}: pit group {g}, stall {k}: pointer {p:#x} (at offset {at:#x}) is not a "
                              f"4-aligned address of {need} bytes before the pointer table ({EXE_BASE:#x}.."
                              f"{EXE_BASE + PIT_POINTERS - need:#x})")
            pts = []
            for j in range(PIT_WAYPOINTS):
                w = PitWaypoint(*struct.unpack_from("<3f", b, off + PIT_RECORD * j))
                if not all(math.isfinite(v) for v in w):
                    raise DcError(f"{EXE_NAME}: pit group {g}, stall {k}, waypoint {j} (offset "
                                  f"{off + PIT_RECORD * j:#x}): {tuple(w)} is not finite")
                pts.append(w)
            stalls.append(tuple(pts))
        groups.append(tuple(stalls))
    return tuple(groups)


def pit_stalls(exe, n: int, v: str = "") -> tuple[tuple[PitWaypoint, ...], ...]:
    """Course n's (0-7) pit stalls in `exe` (the bytes of 1ST_READ.BIN): its plain group (v `""`) or its R group (v
    `"R"`), PIT_STALLS stalls of PIT_WAYPOINTS waypoints. DcError for an n outside 0-7, a v other than those, or a
    table decode_pit_stalls refuses."""
    n = _int(n, "course", 0, COURSE_COUNT - 1, "pit stalls", DcError)
    if not isinstance(v, str) or v not in ("", "R"):
        raise DcError(f"pit stalls: variant {v!r} is not one of ('', 'R')")
    return decode_pit_stalls(exe)[2 * PIT_PAIRS[n] + (v == "R")]
