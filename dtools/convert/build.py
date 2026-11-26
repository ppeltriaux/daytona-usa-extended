"""The converted course: a Dreamcast course's race tables (`tables`) and ground (`ground`) assembled as a
`golden.Course` in an arcade slot (`converted_course`), and written into the arcade set's free runs (`encode`,
`converted_set`).

The slot's decoded Course is the base. Replaced together (Review Focus 4: a slot never races on mixed tables):
step_n and the steps (and the pit, their view), the road line, the AI lines, the ground, the checkpoint rows (and
their views), the grid base, the start (ROLLING_STARTS[n]: rolling on n 0 only) and the per-segment flags (n + 1
zeros, `tables.segment_flags`). The new tables do not fit the slot's (n 3's road line is 465 points against
Beginner's 196), so each is written into free space and its pointer repointed (docs/spike/2026-11-convert.md,
"Layout"):

- data ROM, `tgp.DATA_FREE`, from `Layout.data_at`, back to back and 4-byte aligned: the ground's grid and attribute
  block (as `tgp.placed_ground` places them), the step table, the road line, the line table, the checkpoint
  table and the per-segment flags (DATA_ORDER);
- copro_data, `tgp.COPRO_FREE`, from `Layout.copro_at`: the ground's list block, its pool right after it, no list
  and no polygon across a 64K-word bank window (`tgp.window_safe`);
- repointed in place: the course record's words 1, 4 and 5 (step table, line table, road line) and its words 6-7
  where they hold the grid pointer (`tgp.repoint_record_raw`, as `tgp.relocate_ground` does), the two ground
  pointer words (*block_ref_a, *block_ref_b), slot `slot`'s two copro_data slot words, CHECKPOINT_TABLES[slot],
  course.SEGMENT_FLAGS[slot], the grid-base word and course.ROLLING_STARTS[slot] (REWRITTEN_IN_PLACE). Slots
  4-15's words keep the slot's old ground (golden writes slot `base`'s only; no reader of them is known:
  golden.Course docstring), and the old tables stay where they were.

Kept as the slot's (M6 tunes them): laps, the time tables (their pointers: on a course with more checkpoints than
the slot's, the race reads on past the slot's stored table, see `converted_course`), the car count, the time
corrections; shared by every course and unchanged: GRID_LANES, the two spawn tables, BILLBOARD_REF, the car
models. The slot's old per-segment flags stay in the ROM (the id-3 stub's pointer still names Beginner's).

Scenery (Review Focus 5): `scenery="keep"` keeps the slot's; `scenery="empty"` replaces it, together
(SCENERY_FIELDS; docs/spike/2026-11-convert.md, "Scenery emptied" and "HUD map"):

- the cell grid, rewritten in place at the slot's GRID_TABLES word: its 256 cells name EMPTY_LIST, a list of the
  slot's own (Beginner's cell 0) rewritten to its terminator (the object library has no free run and no empty
  list);
- the actor table: the shared empty one (EMPTY_ACTORS, 256 zero words at PROGRAM_FREE's first bytes, written by
  the first course that names it), ACTOR_TABLES[slot] repointed. Nothing spawns the slot's actors then, so its actor
  lists, billboard array, switch instructions and BILLBOARD_REF stay in the ROM unchanged (not written: the Course's
  actor fields are empty);
- the HUD map: the new course's outline (`hud_outline`, at most OUTLINE_QUADS quads) in the polygon run
  (POLYGON_FREE, from `Layout.polygons_at`), named by the slot's outline list, its one entry rewritten in place; the
  place record and zooms framing the new course as the slot's frame the slot's road (`hud_frame`).

With `scenery="empty"` the Course's objects, headers, texture points and texels stay the slot's: the slot's scenery
objects, texture headers, texture points and texels are left in the ROM unchanged (re-emitted as they are); the set
decodes to fewer (only what is reached; the new HUD outline reads its points at the slot's outline tpa, further on
over the slot's unchanged words).

`scenery="model"` (MODEL_FIELDS; docs/spike/2026-11-convert.md, "Object library") is the empty scenery's actor table
and HUD map with real scenery drawn from a `Scenery`: the cell grid rewritten in place to name its object lists,
which live in DATA_FREE (LIST_HOME "data": the draw routines' lower bound is patched down to DATA_FREE's start,
course.LIBRARY_BOUND_PATCH, two program words); its objects in the polygon run (POLYGON_FREE); its texture headers
and texture points in the textures run (TEXTURES_FREE); its texels in the slot's 1 MB course sheet block, rewritten in
place (the common block and the other courses' blocks are never written). The Course's objects, headers, texture
points and texels are then what the set reaches: the scenery's, the car models' and the HUD outline's. A course's own
scenery is `model_scenery` (its model and textures, its decals dropped, one header per drawn polygon in its mode's
variant, the tables at the free runs' ends; docs/spike/2026-11-convert.md, "Scenery assembled"), which
`converted_set(..., scenery="model")` builds from the disc.

A new id (Phase 8, Task 5: `slot=NEW_ID`, Desert City's course id 3 on a grown set with the new ids' guards; docs/
spike/2026-11-sixteen.md, "Desert City at id 3"): the course gets its own of every table it changes, since id 3's
per-id words name Beginner's tables until filled. The race tables as above, plus its two ground pointer words (record
words 2-3 moved), its per-difficulty time words and time tables (the slot's words copied, TIME_TABLES' three id-3
level entries repointed), with its own scenery its cell grid (GRID_TABLES[3]), its 1 MB course sheet block
(COURSE_SHEETS[3]; the slot's block copied there first, the new texels over it) and its HUD outline's own list, all in
the custom set's new areas (NEW_ID_RUNS: data ID_DATA_FREE, copro_data from 4 MB, polygons from 16 MB, textures the
first gap), its object lists and HUD list in the list home DATA_FREE (the draw routines' upper bound), its actor table
in PROGRAM_FREE (the shared EMPTY_ACTORS). What it rewrites in place is only its own: its record and its entries of
the grown arrays (ID_IN_PLACE besides REWRITTEN_IN_PLACE), TGP slot 3's words. Ids 0-2 stay byte-identical. The
empty scenery is slot 0's only.

Every public function validates its inputs and raises ConvertError; a DcError from the Dreamcast readers passes
through."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import pickle
import struct
from pathlib import Path
from typing import NamedTuple

import numpy as np

from dtools import config
from dtools.arcade import course as C
from dtools.arcade import golden
from dtools.arcade import polyrom as P
from dtools.arcade import texture as T
from dtools.arcade import tgp
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import COPRO_SIZE, CUSTOM_FREE, RomSet, reserved
from dtools.arcade.tablebase import DATA_BASE, PROGRAM_WINDOW, _f32, is_int, is_number
from dtools.convert import ConvertError, ground, model, reverse, tables, textures
from dtools.dc import pol, tex
from dtools.dc.data import CourseData, decode_data, road_cells
from dtools.dc.disc import COURSE_COUNT, DcDisc, course_files
from dtools.dc import exe as dc_exe
from dtools.dc.exe import EXE_NAME

GRID_BASES = (100, 984, 480, 458, 418, 123, 493, 493)
"""The Dreamcast's grid bases, courses n 0-7: 1ST_READ.BIN u16[8] at file offset GRID_BASES_AT (repeated after it
for the R, M and MR variants). Phase 5 matched them to the arcade's by likeness: exact on n 0 (Beginner's 100) and
n 1 (Expert's 984), 2 off on n 2 (480 against Advanced's 482). The converter writes GRID_BASES[n]."""
GRID_BASES_AT = 0x17E814

ROLLING_STARTS = (True, False, False, False, False, False, False, False)
"""The start of courses n 0-7: True rolling, False standing (golden.Course.rolling_start's value; stored as
course.encode_rolling_start: 0 / -1). The arcade's own on n 0-2 (Beginner rolling, Expert and Advanced standing);
standing on n 3-7 by ruling: the rolling start's countdown steps at segments n - 29, n - 19, n - 9 and the
Dreamcast's grid bases put the player a few segments behind the lap line (n 3: 458 of 464), past all three, so the
race fired at once and the player's car crashed (docs/spike/2026-11-convert.md, "Course-dependent reads", "Desert
City in MAME")."""

PROGRAM_FREE = (0x3CE98, 12648)
"""Program ROM's free run (byte offset, length): 0xff to the region's end (0x40000; program window 0x0023ce98).
Its first 1,024 bytes are the shared empty actor table (EMPTY_ACTORS, reserved); the rest holds the actor tables and
lists of the courses that have actors (the mirrors of the originals: mirror.mirrored_entry). `program_room` counts
what is left."""
EMPTY_AT = PROGRAM_FREE[0]
"""The program ROM offset of EMPTY_ACTORS: PROGRAM_FREE's first bytes, where the first converted course's table was
written before the table was shared (so Desert City's bytes are unchanged)."""
EMPTY_ACTORS = PROGRAM_WINDOW[0] + EMPTY_AT - 0x20000   # the window shows program ROM from 0x20000
"""The one shared empty actor table (program window 0x0023ce98): 256 zero words, the table of every course that has
no actors. A conversion (it spawns no actor: SCENERY_FIELDS) and a mirror of a course without actors
(mirror.mirrored_entry) point ACTOR_TABLES[id] here instead of writing 1,024 bytes of their own; a course with actors
keeps a table of its own. Its span is reserved (romset.RESERVED: owned by golden.owned_spans whatever its bytes, so no
allocator nor free-space writer takes it) and written once, by the first course that names it (`empty_actors`).
Sound because the table's one reader only reads it: the spawner at 0x22002c loads each of its 256 words once at the
race's load, `ld (r13),r10`, and follows only the non-zero ones (docs/spike/2026-11-courses-4-7.md, "Program room")."""
POLYGON_FREE = (0xE3C364, 1850524)
"""Polygon ROM's free run (byte offset, length): 0xff to the image's end (word index 0x38f0d9, `oba` 0x00b8f0d9).
The new HUD outline goes there (`Layout.polygons_at`), and the model scenery's objects."""
TEXTURES_FREE = (0xB52D48, 709304)
"""Textures region's free run (byte offset, length): 0xff to 0xc00000 (word indices 0x5a96a4-0x5fffff). The model
scenery's texture headers and texture points go there."""

DATA_ORDER = ("ground grid", "ground attributes", "step table", "road line", "line table", "checkpoint table",
              "segment flags")
"""The tables written into DATA_FREE, in layout order (golden.tables names)."""
COPRO_ORDER = ("ground lists", "ground polygons")
"""The tables written into COPRO_FREE, in layout order."""

REWRITTEN_IN_PLACE = frozenset({"course record", "checkpoint pointer", "grid base", "ground grid pointer",
                                "ground attribute pointer", "ground slot: list block address",
                                "ground slot: polygon pool address", "segment flags pointer", "rolling start"})
"""The slot's tables (golden.tables names) whose bytes the conversion changes where they are; `encode` refuses a
Course that changes any other byte outside the free runs. Sound because `encode` also fixes where each of them is:
the record and its pointer fields but the three moved tables (CourseEntry), the grid-base word (grid_base's
address), the start word (rolling_start's address) and the slot words, the checkpoint pointer and the segment flags
pointer (at the slot's id)."""

RACE_FIELDS = ("step_n", "steps", "pit", "ground", "ground_data", "road", "ai_lines", "checkpoint_table",
               "checkpoints", "time_tables", "grid_base", "rolling_start", "spawn", "segment_flags")
"""The golden.Course fields `converted_course` replaces (the views pit, ground, checkpoints and spawn with their
sources; time_tables for the words read with the new checkpoint count): the fields `encode` lets differ from the
slot's decoded Course by default."""
SCENERY_FIELDS = ("cells", "object_lists", "actor_table", "actor_slots", "actor_lists", "billboards",
                  "billboard_ref", "switch_lists", "unmodelled", "hud_map")
"""The golden.Course fields `scenery="empty"` replaces, together (`encode` refuses a part of them): with
RACE_FIELDS the fields `converted_set` lets `encode` change. `grid` is not among them (the grid is rewritten in
place at the slot's address), nor are objects, headers, points and texels (the slot's stay in the ROM)."""
SCENERY_IN_PLACE = frozenset({"cell grid", "object list", "actor table pointer", "HUD map"})
"""The slot's tables (golden.tables names) the empty scenery rewrites in place, on top of REWRITTEN_IN_PLACE. Each
is fixed by address (`_check_scenery`): the grid is the slot's (`grid` is not a replaced field), among the object
lists only EMPTY_LIST changes, ACTOR_TABLES[slot] is at the slot's id, and the HUD map's list is the slot's with as
many entries (its place record and zooms are at the slot's id)."""
SCENERY_PLACED = ("actor table", "HUD map outline")
"""The empty scenery's placed tables: each must lie inside its free run. The actor table is the shared EMPTY_ACTORS
(PROGRAM_FREE's first bytes, reserved: written by `empty_actors`, never by the free-space writer); the HUD outline is
new, in POLYGON_FREE."""

MODEL_FIELDS = SCENERY_FIELDS + ("objects", "headers", "points", "texels")
"""The golden.Course fields `scenery="model"` replaces, together: SCENERY_FIELDS (the actor table emptied, the HUD
outline the new course's; the cells and object lists the scenery's) and the objects, headers, texture points and
texels the set then reaches (texels last: `encode` tells the model scenery by it). The texture points are golden's
`points` since Phase 7 (they were `encode`'s `points` argument before golden decoded them). The model scenery also
writes one thing that is not a golden.Course field: the object-library bound patch (course.LIBRARY_BOUND_PATCH),
which `encode` applies."""
MODEL_IN_PLACE = frozenset({"cell grid", "actor table pointer", "HUD map", "texels", "texels (mips)"})
"""The slot's tables (golden.tables names) the model scenery rewrites in place, on top of REWRITTEN_IN_PLACE: the cell
grid (the slot's, `grid` is not replaced), the actor table pointer and the HUD map as the empty scenery's, and the
texels and their mip levels only inside the slot's course sheet block (sheet_sources[0], texture.SHEET_BLOCK bytes:
`encode` refuses a changed texel byte elsewhere). Object lists, objects and headers are new tables in the free runs."""
LIST_HOME = "data"
"""Where the model scenery's object lists are written: "data", in DATA_FREE (i960 0x0265d120-0x026fffff). The ruling of
docs/spike/2026-11-convert.md, "Object library": the only readers of the object library's bounds are the draw routines
0x17acc and 0x17cf8, each a range compare of the list pointer against two `lda` constants, so one program word each
(course.LIBRARY_BOUND_PATCH, the lower bound moved down to DATA_FREE's start) admits DATA_FREE; with it applied alone
the three original courses draw identical geometry lists and render-rate logs in MAME (tests/test_convert_mame.py,
test_library_bound_patch_draws_originals). The other choice, "library" (one merged object per cell, its one-entry list
written over a Beginner-only list in place), is not needed."""

EMPTY_LIST = 0x0284CCEC
"""The empty scenery's object list: Beginner's cell 0 list (one link-0 stub, never drawn, in the far corner),
rewritten to its 4-byte terminator. The object library has no list that decodes to no placement (walked as
back-to-back lists: each of its 3,190 has an entry), so the ruling's fallback: the shortest list of Beginner's that
no other course names (every one has one entry), the lowest of the 256 that only Beginner's grid names (not the
actors', billboard's or switch's, whose bytes stay in place). tests/test_convert_build.py, test_empty_list_choice."""

OUTLINE_WIDTH = 22.0
"""The HUD outline's stroke, world units: Beginner's drawn quads are a median 21.9 wide (area over the mean of the
two long sides; Expert's 23.2, Advanced's 26.8)."""
OUTLINE_ATTR = 0x010C0A01
"""The outline's quad attribute: Beginner's link-2 quads' (0x010c1a01: a quad, link 2, z-sort max, texture parameter
3, coefficient 2) with the texture-header step (bits 12-16) 0 instead of 1, so every polygon reads the list's one
header (Beginner's tha 0x10acf4: untextured, colour 1) instead of walking on past Beginner's 24 headers."""
OUTLINE_QUADS = 64
"""The most quads a HUD outline has (`hud_outline`): its budget word (its record count) is added to the frame's
polygon count before any scenery cell is tested against the 5,000 cap (docs/spike/2026-11-convert.md, "Confirmed with
scenery"), so a course's outline is merged to this many (Beginner's has 25 records; the map is about 53 x 88 px)."""
OUTLINE_NORMAL = (0.0, -1.0, 0.0)
"""The normal of every drawn polygon of the real outlines."""
OUTLINE_END = (0, 0x40401, 0, 0)
"""The end record's words as the real outlines store them (attribute 0, then three words read as a normal)."""
HUD_LEVEL = 1
"""The zoom level the HUD set-up 0x12e58 starts the race at: the framing is exact at it (the translation is applied
before the eased scale, so the centre drifts at the other levels, as Beginner's own)."""
_QUARTER = 0x4000               # TGP angles: 0x10000 a turn
_F32_NORMAL = 2.0 ** -126       # the smallest normal single-precision value

_MOVED = ("step_table", "record_table", "line_table")       # the CourseEntry pointer fields a conversion moves

SCENERY = ("keep", "empty", "model")
_RUNS = {"data": tgp.DATA_FREE, "copro_data": tgp.COPRO_FREE, "program": PROGRAM_FREE, "polygons": POLYGON_FREE,
         "textures": TEXTURES_FREE}
_REGIONS = ("data", "copro_data", "program", "polygons")        # Layout's order
_FREE = 0xFF

# ---- a new course id (Phase 8, Task 5): docs/spike/2026-11-sixteen.md, "Desert City at id 3" ----

NEW_ID = 3
"""The new course id a conversion fills besides Beginner's slot 0: Desert City's, on a grown set (course.grow_ids) with
the new ids' guards (course.guard_new_ids). `slot=NEW_ID` gives the course its own of every table it changes (module
docstring, the paragraph on a new id); slot 0 keeps Phase 6-7's rewrite of Beginner's tables in place."""
CONVERTED_IDS = (NEW_ID, 4, 7, 6, 5)
"""The new course ids a conversion fills on a grown set with the new ids' guards, each the Dreamcast course n of the
same number (`slot=n`), in the order they were added: Desert City (n 3, Phase 8), National Park Speedway (n 4, Phase 10
Task 3), Mermaid Lake (n 7, Phase 10 Task 4), Rin Rin Rink (n 6, Phase 11 Task 3, shifted: SHIFT) and Circuit Pixie
(n 5, Phase 11 Task 4). Each takes the new-id path NEW_ID takes (its own of every table it changes, in NEW_ID_RUNS,
each at the first bytes free: the course converted later lands after the earlier one), so nothing per id is added to
NEW_ID_RUNS, ID_DATA_ORDER or ID_FIELDS; the program sites that test the id already send ids 4-15 down the path id 3
takes (course.ID3_LITERALS, course.widen_ids)."""
REVERSE_IDS = tuple(range(8, 16))
"""The reverse course ids (M9, docs/superpowers/specs/2026-11-16-reverse-courses-design.md): Dreamcast course n driven
the other way, converted from its R files (`variant="R"`), is id n + 8."""
VARIANTS = ("", "R")
"""The Dreamcast variants a conversion reads: the plain course, or its reverse (into REVERSE_IDS)."""
SLOTS = (0,) + CONVERTED_IDS + REVERSE_IDS
_ID_DATA_AT = (C.SRAM_SINK + C.SRAM_SINK_SIZE - DATA_BASE + 0xFF) & ~0xFF
ID_DATA_FREE = (_ID_DATA_AT,
                next(lo + n for lo, n in CUSTOM_FREE["data"] if lo <= _ID_DATA_AT < lo + n) - _ID_DATA_AT)
"""The data run a new id's tables take (data-ROM offset, length): the custom data area's first free run
(romset.CUSTOM_FREE["data"][0]) from past grow_ids' tables (course.ID_HOME, 0x03000000-0x0300157f) and the new
ids' SRAM sink (course.SRAM_SINK), both reserved (romset.RESERVED), to the select's reserved data (0x01f00000).
Its race tables, cell grid, ground pointer words, time tables and 1 MB course sheet block."""
NEW_ID_RUNS = {"data": ID_DATA_FREE, "copro_data": CUSTOM_FREE["copro_data"][0], "program": PROGRAM_FREE,
               "polygons": CUSTOM_FREE["polygons"][0], "textures": CUSTOM_FREE["textures"][0]}
"""A new id's free runs per region (byte offset, length): the custom set's new areas (romset.CUSTOM_FREE: data
ID_DATA_FREE, copro_data from 4 MB, polygons from 16 MB, textures the first 4 MB gap) and the program run
(PROGRAM_FREE: the actor table must be a program-window address). Its object lists (and its HUD outline's list) stay
in the list home DATA_FREE: the draw routines' upper bound (course.OBJECT_LIBRARY[1], 0x0284f4ec) is below the custom
data area, so a list there would not be drawn."""
ID_DATA_ORDER = ("ground pointers", "time pointers", "time tables", "cell grid")
"""A new id's data tables after DATA_ORDER, in layout order: its two ground pointer words (the record's block_ref_a /
block_ref_b, a slot-0 conversion rewrites Beginner's in place), its per-difficulty time-table words (one array of
course.DIFFICULTIES per mode) and the time tables they name (the slot's words, read as converted_course reads them),
and, with its own scenery, its cell grid."""
ID_FIELDS = ("grid", "sheet_sources")
"""The golden.Course fields a new id's model scenery replaces besides MODEL_FIELDS: its own cell grid (GRID_TABLES[id]
repointed) and its own 1 MB course sheet block (COURSE_SHEETS[id] repointed)."""
_ID_MOVED = ("block_ref_a", "block_ref_b")      # a new id's record also moves its two ground pointer words
ID_PLACED = ("ground grid pointer", "ground attribute pointer", "time table pointer", "time table", "cell grid")
"""A new id's tables (golden.tables names) written into its free runs besides DATA_ORDER, COPRO_ORDER and
SCENERY_PLACED."""
ID_IN_PLACE = frozenset({"cell grid pointer", "sheet source", "time levels"})
"""A new id's per-id words rewritten in place besides REWRITTEN_IN_PLACE and MODEL_IN_PLACE (each its own entry of a
grown array): GRID_TABLES[id], COURSE_SHEETS[id] and its three TIME_TABLES levels (written by `encode`)."""


def _runs(slot: int) -> dict[str, tuple[int, int]]:
    return NEW_ID_RUNS if slot else _RUNS


_RUN_NAMES = {"data": "DATA_FREE", "copro_data": "COPRO_FREE", "program": "PROGRAM_FREE", "polygons": "POLYGON_FREE",
              "textures": "TEXTURES_FREE"}


def _run_name(region: str, slot: int) -> str:
    """The free run of `region` for messages: its name and span."""
    lo, n = _runs(slot)[region]
    return f"{_RUN_NAMES[region] if not slot else f'NEW_ID_RUNS[{region!r}]'} {lo:#x}+{n:#x}"


class Layout(NamedTuple):
    """The first byte used in each free run: data ROM offset in tgp.DATA_FREE, copro_data byte offset in
    tgp.COPRO_FREE (its first 2 MB copy), program ROM offset of the actor table (EMPTY_AT, the default and the only
    value: a converted course has no actors, so its table is the shared EMPTY_ACTORS), polygon ROM offset in
    POLYGON_FREE (default its first byte). Each 4-byte aligned. The tables follow each other from there (module
    docstring).
    For a new id (a slot of CONVERTED_IDS) the runs are NEW_ID_RUNS' and two more places: `list_at`, the data-ROM
    offset of its HUD outline's list in the list home DATA_FREE (its own scenery), and `sheet_at`, the data-ROM
    offset of its 1 MB course sheet block in ID_DATA_FREE (1 MB-aligned); None for slot 0 (and chosen by default for
    a new id)."""
    data_at: int
    copro_at: int
    program_at: int = EMPTY_AT
    polygons_at: int = POLYGON_FREE[0]
    list_at: int | None = None
    sheet_at: int | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class Scenery:
    """What a converted course draws with `scenery="model"`, addresses in golden.Course's conventions:

    - objects: polygon-ROM word index -> polyrom.Object (geometry mode 1's layout, normals present), each inside
      POLYGON_FREE;
    - lists: i960 address -> the object list's placements as golden.decode_course reads them back (`object_list_at`),
      each list inside the list home (LIST_HOME "data": DATA_FREE), 4-aligned; every one named by a cell (an empty
      list, its terminator only, serves the cells with nothing to draw);
    - cells: the 256 cells' list addresses (keys of `lists`);
    - headers: textures word index -> texture.TexHeader, each inside TEXTURES_FREE (8 bytes). Their four words are
      the header; `tha` and `step` are set by `converted_course` from the walk (as decode_course reads them: the
      address sent and the first reaching record's step). A placement may also read a header of the slot's
      (unchanged in the ROM) that is not in `headers`;
    - points: textures word index (a cell placement's tpa) -> its texture points as golden.Course.points holds them
      (texture.texture_points: per record of the object, 3 or 4 (v, u) tuples of ints, 1/8 texel;
      `textures.course_points` makes them), each inside TEXTURES_FREE;
    - texels: golden.Texels, one per texture geometry the headers read (level 0 and its mip levels), each in sheet 0:
      the slot's course block, rewritten in place;
    - lift: the world units the model it was made from was raised by (`lifted_model`; 0.0 for a model as decoded),
      which `converted_course` checks against LIFT for the course (final review of Phase 10, Minor 3);
    - shift: likewise the world units it was moved by in x (`shifted_model`), checked against SHIFT (Phase 11 Task
      3)."""
    objects: dict
    lists: dict
    cells: tuple
    headers: dict
    points: dict
    texels: tuple
    lift: float = 0.0
    shift: float = 0.0


def object_list_at(addr, entries) -> tuple[C.Placement, ...]:
    """The placements of an object list written at i960 address `addr`, as golden.decode_course reads them back
    (course.object_list): one per (oba, tpa, tha, budget) of `entries`, obc course.OBC, cell -1, entry the data-ROM
    offset of its 16 bytes, in_world True, no actor. ConvertError for an address that is not a 4-aligned data-ROM
    address, or entries that are not (oba, tpa, tha, budget) u32 quadruples with oba a polygon-ROM object."""
    if not is_int(addr) or addr % 4 or not DATA_BASE <= addr < DATA_BASE + 0x1000000:
        raise ConvertError(f"object list address {addr!r:.40} is not a 4-aligned data-ROM address")
    if not isinstance(entries, (tuple, list)):
        raise ConvertError(f"object list entries {entries!r:.60} are not a sequence of (oba, tpa, tha, budget)")
    out = []
    for k, e in enumerate(entries):
        if not isinstance(e, (tuple, list)) or len(e) != 4 or not all(is_int(w) and 0 <= w <= 0xFFFFFFFF for w in e):
            raise ConvertError(f"object list entry {k} {e!r:.60} is not (oba, tpa, tha, budget) u32 words")
        try:
            C.rom_object_index(e[0])
        except DecodeError as err:
            raise ConvertError(f"object list entry {k}: {err}") from err
        out.append(C.Placement(int(e[0]), C.OBC, int(e[1]), int(e[2]), (int(e[3]),), -1,
                               int(addr) - DATA_BASE + C.OBJECT_ENTRY_SIZE * k, True, None))
    return tuple(out)


# ---- the course model as scenery (Phase 7, Task 5) ----

TSP_SRC, TSP_DST = 29, 26
"""A Dreamcast mesh's TSP word (`Mesh.params[2]`, PowerVR): the source blend at bits 29-31, the destination at 26-28."""
BLEND_SRC_ALPHA, BLEND_INV_SRC_ALPHA = 4, 5
TSP_IGNORE_ALPHA = 1 << 19      # the texture's alpha is not used (the mesh draws it opaque)
TSP_FLIP_U, TSP_FLIP_V = 1 << 18, 1 << 17       # mirrored repeat on u / v
TSP_CLAMP_U, TSP_CLAMP_V = 1 << 16, 1 << 15     # clamped to the texture's edge on u / v (takes precedence over flip)
TSP_SHADING = 6                 # bits 6-7: the texture/shading instruction (1, 3: modulate the texture by the colour)
MODULATE = (1, 3)
MESH_COLOUR = slice(1, 5)
"""`Mesh.colours[1:5]`: the face colour (alpha, red, green, blue, 0-1) of the PowerVR intensity colour modes
(parameter word 0 bits 4-5: 2, intensity mode 1, the mesh's own face colour; 3, intensity mode 2, the face colour of
the last mode-1 mesh before it, in file order). `_face_colours` reads it. An untextured mesh is drawn in it (CS3:
black, greys, browns, a red, a yellow); a textured mesh whose shading modulates (TSP_SHADING) is tinted by it."""
COLOUR_MODE = 4                 # parameter word 0 bits 4-5: 0 packed, 1 floating, 2 intensity 1, 3 intensity 2
FLAT_HEADER = dict(tha=0, step=0, width=32, height=32, tex_x=0, tex_y=0, sheet=0, textured=False, translucent=False,
                   checker=False, wrap_x=False, wrap_y=False, mirror_x=False, mirror_y=False, utex=False,
                   utex_min_lod=0, utex_x=0, utex_y=0, luma_base=0, raw=(0, 0, 0))
"""An untextured header's fields but its colour: the arcade's convention (4,523 of Beginner's 4,800 untextured
headers: every field 0, 32 x 32, raw (0, 0, 0); the other 277 add the checker bit)."""


def _blends(me: pol.Mesh) -> bool:
    """True when the mesh alpha-blends its texture: source alpha, one minus source alpha, the texture's alpha used."""
    w = me.params[2]
    return ((w >> TSP_SRC) & 7, (w >> TSP_DST) & 7) == (BLEND_SRC_ALPHA, BLEND_INV_SRC_ALPHA) and \
        not w & TSP_IGNORE_ALPHA


def _modes(me: pol.Mesh) -> tuple[str, str]:
    """The mesh's (u, v) addressing: "clamp", "mirror" or "repeat" each (clamp before flip, as the PowerVR's)."""
    w = me.params[2]
    return tuple("clamp" if w & clamp else "mirror" if w & flip else "repeat"
                 for clamp, flip in ((TSP_CLAMP_U, TSP_FLIP_U), (TSP_CLAMP_V, TSP_FLIP_V)))


def _meshes(m: pol.Model) -> dict[tuple[int, int], pol.Mesh]:
    return {(oi, mi): me for oi, ob in enumerate(m.objects) for mi, me in enumerate(ob.meshes)}


DECAL_STEP = 17
"""One ARGB4444 step (255 / 15): a decal's visible texels may stray this far from its modal colour, per channel."""
DECAL_SHARE = 0.99
"""... and its modal colour holds at least this share of them (v0.1.2: CS5's and CS7's shadow texture 67, 994 of 1000
texels (17, 17, 17), six one step off; CS6's near-white texture 131, 77 % one colour, is no decal)."""


def decals(m, tex_blob) -> tuple[int, ...]:
    """The Dreamcast textures whose meshes the scenery drops (Phase 7 ruling, docs/spike/2026-11-convert.md, "Scenery
    assembled"): the arcade's translucency is binary (index 15 transparent, everything else opaque), so an alpha-blended
    decal whose picture is in its alpha alone cannot be drawn and would show as a dark square. A texture is one when
    every mesh drawing it alpha-blends it (`_blends`), its alpha is not all opaque, and either its visible texels
    (alpha above 0) are all one colour (a shadow: Desert City's 55, 56 and 74, (17, 17, 17) at 40-67 % alpha), or
    nearly (v0.1.2: the modal colour holds DECAL_SHARE of them and none is more than DECAL_STEP from it per channel:
    Mermaid Lake's and Circuit Pixie's shadow 67, drawn solid black under gantries and bridges when kept), or none
    reaches 50 % alpha (textures.OPAQUE: Desert City's 81, wholly transparent once converted). Ascending. ConvertError for an `m` that is
    not a pol.Model or a blob that is not bytes; DcError from tex.decode_tex."""
    if not isinstance(m, pol.Model) or not isinstance(m.textures, tuple) or not isinstance(m.objects, tuple):
        raise ConvertError(f"m {type(m).__name__} is not a pol.Model")
    if not isinstance(tex_blob, (bytes, bytearray)):
        raise ConvertError(f"tex_blob {type(tex_blob).__name__} is not bytes (the course's _TEX.BIN)")
    model.triangle_meshes(m)                                    # m checked as convert_model checks it
    users: dict[int, list[bool]] = {}
    for me in _meshes(m).values():
        if is_int(me.texture) and me.texture >= 0:
            users.setdefault(int(me.texture), []).append(_blends(me))
    out = []
    for i, t in enumerate(tex.decode_tex(bytes(tex_blob), m.textures, "textures")):
        if not users.get(i) or not all(users[i]):
            continue
        rgba = tex.to_rgba(t)
        alpha = rgba[..., 3]
        if (alpha == 255).all():
            continue                                            # opaque: drawn as it is, whatever the blending
        visible = rgba[..., :3][alpha > 0]
        if not (alpha >= textures.OPAQUE).any() or _one_colour(visible):
            out.append(i)
    return tuple(out)


def _one_colour(visible: np.ndarray) -> bool:
    """True when the visible texels (an (n, 3) array) are one colour, or nearly: DECAL_SHARE of them the modal
    colour and every one within DECAL_STEP of it per channel (`decals`)."""
    colours, counts = np.unique(visible, axis=0, return_counts=True)
    if len(colours) <= 1:
        return True
    mode = colours[np.argmax(counts)].astype(int)
    return counts.max() >= DECAL_SHARE * len(visible) and int(np.abs(visible.astype(int) - mode).max()) <= DECAL_STEP


SKY_RADIUS = 400.0
"""A sky dome's bounding radius is past this (Phase 1's density proxy rule, spike/dc_course_scan.py `sky_objects`; on
the six models the converter takes, 0, 2, 3, 4, 5 and 7, the domes are objects 0 and 1, radius 567, and no other)."""
SKY_SHELL = 0.1
"""... and every vertex of a dome lies within this fraction of the radius from its bounding sphere's surface."""
VOLUME_GRID = 0.01
"""Positions are rounded to this (world units) when a shadow volume's edges are paired."""


def sky_domes(m) -> tuple[int, ...]:
    """The objects of `m` that are a sky dome (Phase 7 Task 6, docs/spike/2026-11-convert.md, "Desert City seen"):
    bounding radius past SKY_RADIUS and every vertex within SKY_SHELL of the radius from the bounding centre. The arcade
    draws its own sky (the slot's backdrop), and a dome converted as static scenery sits in the cells round the world
    origin, drawn only where those cells are seen: in MAME a violet sheet over part of the sky. Ascending. ConvertError
    for an `m` that is not a pol.Model of the pol types (`model.triangle_meshes`' checks) or an object whose bounding
    sphere is not four finite numbers."""
    model.triangle_meshes(m)
    out = []
    for oi, ob in enumerate(m.objects):
        sphere = ob.sphere
        if not isinstance(sphere, (tuple, list)) or len(sphere) != 4 or not all(is_number(v) for v in sphere):
            raise ConvertError(f"object {oi}: bounding sphere {sphere!r:.60} is not four finite numbers")
        x, y, z, r = (float(v) for v in sphere)
        if not r > SKY_RADIUS:
            continue
        ps = np.array([(e.x, e.y, e.z) for me in ob.meshes for st in me.strips for e in st.entries
                       if isinstance(e, pol.Vertex)], dtype=float)
        if len(ps) and (np.abs(np.linalg.norm(ps - (x, y, z), axis=1) - r) <= SKY_SHELL * r).all():
            out.append(oi)
    return tuple(out)


VOLUME_KIND = 3
"""The object kind word of the Dreamcast's shadow volumes (and its sky domes, which are textured): v0.1.1, out/debug/
np-start/report.md. National Park Speedway (CS4) colours each volume with a per-volume ID (a grey ramp, some pinkish)
and Rin Rin Rink (CS6) one (0, 0.02, 0), so black alone misses them."""


def shadow_volumes(m) -> tuple[tuple[int, int], ...]:
    """The meshes of `m` that are shadow volumes (Phase 7 Task 6, "Desert City seen"): untextured, black (face colour
    (0, 0, 0), `_face_colours`) or of an object of kind VOLUME_KIND (v0.1.1: CS4's 23 and CS6's 5 volumes are not
    black; the black test stays for CS4's 19:0, a kind-1 object) and closed (every edge of its non-degenerate
    triangles, positions rounded to VOLUME_GRID, shared by an even number of them: two, or four where two volumes meet
    along it, as three edges of Desert City's 35:0; at least six edges, a tetrahedron's). On the Dreamcast these enclose the road and the
    shoulders where a cliff or a wall casts its shadow and are not seen (the PowerVR's modifier volumes darken what
    lies inside them); drawn as polygons by the arcade, which sorts polygons instead of testing depth, they are black
    walls standing across the road. (object, mesh) pairs, ascending. ConvertError as `sky_domes`."""
    where = model.triangle_meshes(m)
    face = _face_colours(m)
    meshes = _meshes(m)
    edges: dict[tuple[int, int], dict] = {}
    for (a, b, c, _), key in zip(pol.triangles(m), where):
        me = meshes[key]
        if me.texture != pol.NO_TEXTURE or (face[id(me)] != (0.0, 0.0, 0.0)
                                            and m.objects[key[0]].kind != VOLUME_KIND):
            continue
        t = [tuple(round(float(v) / VOLUME_GRID) for v in (p.x, p.y, p.z)) for p in (a, b, c)]
        e = edges.setdefault(key, {})
        if len(set(t)) == 3:
            for i in range(3):
                k = tuple(sorted((t[i], t[(i + 1) % 3])))
                e[k] = e.get(k, 0) + 1
    return tuple(sorted(k for k, e in edges.items() if len(e) >= 6 and all(n % 2 == 0 for n in e.values())))


PROP_RADIUS = 32.0
"""An object is a run-time prop when every vertex lies within this (world units, x and z) of the world origin (the
Dreamcast's: on a shifted model, `shifted_model`, the origin moved by its shift)."""


def origin_props(m, d: CourseData | None = None) -> tuple[int, ...]:
    """The objects of `m` that are run-time props (Phase 7 Task 7 ruling 1, docs/spike/2026-11-convert.md,
    "Thinning"): every vertex within PROP_RADIUS of the world origin in x and z. The Dreamcast models its small props
    at the origin (on Desert City objects 2-4, 8-28 and 54, none reaching 22 units from it; kind 1 but for 22, 24 and
    54, so the kind word does not mark them) and places them at run time; they are actors, out of scope, and
    converted as static scenery they stand in the cells round the origin (83 units from Desert City's road line)
    costing polygons. An object without a vertex is not one. The origin is the Dreamcast's, where its props are
    modelled: on a model moved by `shifted_model` it is (`_shift_of(m)`, 0), the origin moved with the course (Phase 11
    Task 3: CS6's 20 objects about the origin, RINK_NEAR_ORIGIN in the tests, would all be missed at the arcade's).

    The road guard (Phase 10 Task 3, docs/spike/2026-11-courses-4-7.md, "National Park Speedway"): with `d`, the
    course the model is drawn on, an object with a vertex within ROAD_NEAR of `d`'s road line (`_road_line`, the
    exact point-to-segment distance in x, y and z, as `road_triangles` measures it) is road-side scenery, not a prop,
    and is kept: where a road passes the origin (CS4 9 units from it, CS7 31) a real object there would otherwise be
    taken. On the convertible models it keeps none: CS4's objects 52 and 53, 2.7-4.1 units from the road line in x and
    z only, lie 12.6-16.8 units below it, buried under the ground there (a tree and a frame modelled at y 0-15, the
    road and the ground at y 23-27), so they are props. On CS7 the rule finds 130 objects, all 29 units or more from
    the road line: two waving flags of 64 frames each, a marshal and his shadow (Phase 10 Task 4); the run-time props
    it misses are named per course (NAMED_PROPS).

    Ascending. ConvertError for an `m` that is not a pol.Model of the pol types (`model.triangle_meshes`' checks), a
    `d` that is neither None nor a CourseData, or road points without finite positions."""
    model.triangle_meshes(m)
    if d is not None and not isinstance(d, CourseData):
        raise ConvertError(f"d {type(d).__name__} is not a dc.data.CourseData")
    line = _road_line(d) if d is not None else None
    ox = _shift_of(m)
    out = []
    for oi, ob in enumerate(m.objects):
        ps = np.array([(e.x, e.y, e.z) for me in ob.meshes for st in me.strips for e in st.entries
                       if isinstance(e, pol.Vertex)], dtype=float).reshape(-1, 3)
        if not len(ps) or not (np.hypot(ps[:, 0] - ox, ps[:, 2]) <= PROP_RADIUS).all():
            continue
        if line is not None and _near_line(ps, *line).any():
            continue                                            # the road guard: road-side scenery
        out.append(oi)
    return tuple(out)


def kept_model(m, tex_blob, d: CourseData | None = None, props=()) -> pol.Model:
    """`m` as the scenery draws it (`model.drop_meshes`): without the meshes of the decals (`decals`), of the sky domes
    (`sky_domes`), the shadow volumes (`shadow_volumes`) and the run-time props at the origin (`origin_props`, Task
    7; with `d`, the course it is drawn on, past the road guard), each found on `m` itself, and of the objects `props`
    names (object indices of `m`: the run-time props the origin rule does not find, NAMED_PROPS' for the course).
    ConvertError as those, and for a `props` that is not a sequence of object indices of `m`."""
    gone = set(decals(m, tex_blob))
    named = _props(m, props)
    keys = {(oi, mi) for oi, ob in enumerate(m.objects) for mi, me in enumerate(ob.meshes) if me.texture in gone}
    keys |= {(oi, mi) for oi in sky_domes(m) + origin_props(m, d) + named for mi in range(len(m.objects[oi].meshes))}
    keys |= set(shadow_volumes(m))
    return model.drop_meshes(m, sorted(keys))


def _props(m: pol.Model, props) -> tuple[int, ...]:
    """`props` (kept_model's) as a tuple of object indices of `m`; ConvertError otherwise."""
    if not isinstance(props, (tuple, list)) or not all(is_int(k) and 0 <= k < len(m.objects) for k in props):
        raise ConvertError(f"props {props!r:.60} is not a sequence of object indices of the model "
                           f"(0..{len(m.objects) - 1})")
    return tuple(int(k) for k in props)


NAMED_PROPS: dict[int, tuple[int, ...]] = {4: (3,), 7: (2,), 6: (2, 3, 4) + tuple(range(20, 51)),
                                           5: tuple(range(3, 34)) + tuple(range(35, 98))}
"""Per course n, the objects of its model that are run-time props though the origin rule (`origin_props`: every
vertex within PROP_RADIUS of the origin) does not find them, named after inspection; `kept_model` drops them with the
props (converted_set passes NAMED_PROPS.get(n)). Mermaid Lake (Phase 10 Task 4, docs/spike/2026-11-courses-4-7.md,
"Mermaid Lake"): object 2, the Dreamcast airship (the DREAMCAST / Deep Space Planet banner, the gondola's windows,
the propeller disc), modelled about the origin (its bounding sphere's centre x 0, z -16.5) and 139 units long, so it
reaches 86 units out; converted as scenery it lay half in the ground (its keel 24 units below it) beside the road at
segment 74, the course's flag-waving marshal (the origin rule's props 3-132) standing over it. National Park Speedway
(Task 4's fix round 1, the controller's ruling on I1): object 3, a big wheel's frame (100 x 100 x 14 units, textures 46
and 54) modelled about its hub at the origin, reaching 50 units out; converted as scenery it stands across the road
near segment 394 (its strips one-sided: in MAME drawn, 25 polygons at the lap's busiest frame, but turned away from
the lap's views). Rin Rin Rink (Phase 11 Task 3, Pascal's decision 2026-11-12, "Try shorter view first", step 1):
objects 20-50, the frames after the first of one animated advertising banner (32 objects of 70 vertices, texture 117,
at one place, x 176..286 as the disc stores it, their vertices differing by up to 4.5 units, their u, v the same),
which the Dreamcast shows one at a time and the static scenery drew all at once: the banner stays as one still frame
(object 19); and objects 2-4, three copies of one prop (193 vertices each, identical positions and u, v, textures 30
and 86) modelled about the origin (every vertex within PROP_RADIUS), which the origin rule finds and the road guard
keeps because the road passes the origin. Circuit Pixie (Phase 11 Task 4, docs/spike/2026-11-courses-4-7.md, "Circuit
Pixie"): the frames after the first of its two animations, placed in the course (so the origin rule misses them) and
drawn all at once by the static scenery (on the unreduced lap they filled cell 0x68 with 3,824 records and 0x97 with
2,006): objects 35-97, a waving flag (64 objects of 48 vertices, texture 0, the US flag and SEGA sheet, the same shape
as Mermaid Lake's 64-frame flags, here at x 34..50, y 26..35, z 193..201), which stays as frame 34; and objects 3-33, a
waving sponsor banner (32 objects of 84 vertices in two meshes, texture 23), which stays as frame 2; each animation's
frames have the same u, v and lie within 4.6 units of its first. A course not named drops none more: Desert City
keeps its bytes."""


ROAD_NEAR = 8.0
"""World units: a triangle with a corner this near the road line is never thinned (`road_triangles`); an object at
the origin with a vertex this near it is not a prop (`origin_props`' road guard)."""
ROAD_RISE = 1.0
"""World units: a triangle whose centroid lies on a road cell within this above or below it is the road surface."""


def _road_line(d: CourseData) -> tuple[np.ndarray, np.ndarray]:
    """The road line's segments (start points, end points; Dreamcast frame): each lane used at both ends of a
    segment, point i to i + 1, the last point to the first (the lap line)."""
    road = d.road
    if not isinstance(road, (tuple, list)) or len(road) < 2:
        raise ConvertError(f"d.road {road!r:.40} is not a sequence of at least two road points")
    ends = []
    for i, rp in enumerate(road):
        lo, hi, lanes = getattr(rp, "lane_lo", None), getattr(rp, "lane_hi", None), getattr(rp, "lanes", None)
        if not (is_int(lo) and is_int(hi) and 0 <= lo <= hi and isinstance(lanes, (tuple, list)) and hi < len(lanes)):
            raise ConvertError(f"road point {i}: {rp!r:.60} is not a RoadPoint with lanes lane_lo..lane_hi")
        pts = {}
        for j in range(int(lo), int(hi) + 1):
            xyz = tuple(getattr(lanes[j], k, None) for k in "xyz")
            if not all(is_number(v) for v in xyz):
                raise ConvertError(f"road point {i} lane {j}: {lanes[j]!r:.60} has no finite x, y, z")
            pts[j] = tuple(float(v) for v in xyz)
        ends.append(pts)
    a, b = [], []
    for i, here in enumerate(ends):
        there = ends[(i + 1) % len(ends)]
        for j in sorted(set(here) & set(there)):
            a.append(here[j])
            b.append(there[j])
    return np.array(a, dtype=float).reshape(-1, 3), np.array(b, dtype=float).reshape(-1, 3)


def _on_cells(points: np.ndarray, cells) -> np.ndarray:
    """Per point (x, y, z): True when it lies over a cell (its quad split 0-1-2 and 0-2-3, in x and z) within
    ROAD_RISE of the cell's height there."""
    hit = np.zeros(len(points), dtype=bool)
    for k, cell in enumerate(cells):
        corners = getattr(cell, "corners", None)
        if not isinstance(corners, (tuple, list)) or len(corners) != 4 or not all(
                isinstance(c, (tuple, list)) and len(c) == 3 and all(is_number(v) for v in c) for c in corners):
            raise ConvertError(f"road cell {k}: {cell!r:.60} has no four finite corners")
        q = np.array(corners, dtype=float)
        lo, hi = q[:, [0, 2]].min(axis=0), q[:, [0, 2]].max(axis=0)
        near = np.flatnonzero((points[:, 0] >= lo[0]) & (points[:, 0] <= hi[0]) & (points[:, 2] >= lo[1])
                              & (points[:, 2] <= hi[1]))
        for t in ((0, 1, 2), (0, 2, 3)):
            a, b, c = q[list(t)]
            m2 = np.array([[b[0] - a[0], c[0] - a[0]], [b[2] - a[2], c[2] - a[2]]])
            if abs(np.linalg.det(m2)) < 1e-12:
                continue
            st = np.linalg.solve(m2, (points[near][:, [0, 2]] - a[[0, 2]]).T).T
            inside = (st >= -1e-9).all(axis=1) & (st.sum(axis=1) <= 1 + 1e-9)
            y = a[1] + st[:, 0] * (b[1] - a[1]) + st[:, 1] * (c[1] - a[1])
            hit[near[inside & (np.abs(points[near][:, 1] - y) <= ROAD_RISE)]] = True
    return hit


def _near_line(flat: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per point (x, y, z) of `flat`: True when it lies within ROAD_NEAR of one of the segments a[i]-b[i] (`_road_line`;
    the exact point-to-segment distance)."""
    order = np.argsort(flat[:, 0])
    xs = flat[order, 0]
    near = np.zeros(len(flat), dtype=bool)
    for p, q in zip(a, b):
        i, j = np.searchsorted(xs, min(p[0], q[0]) - ROAD_NEAR), np.searchsorted(xs, max(p[0], q[0]) + ROAD_NEAR,
                                                                                  side="right")
        idx = order[i:j]
        pts = flat[idx]
        e = q - p
        t = np.clip((pts - p) @ e / (e @ e), 0.0, 1.0) if e @ e > 0 else np.zeros(len(pts))
        near[idx[np.linalg.norm(pts - (p + t[:, None] * e), axis=1) <= ROAD_NEAR]] = True
    return near


def road_triangles(m, d) -> frozenset[int]:
    """The triangles of `m` (indices into `pol.triangles(m)`) that thinning never removes (Phase 7 Task 7,
    `model.thin`'s `protected`): those with a corner within ROAD_NEAR of the road line of `d` (the course the scenery
    is drawn on: every used lane's polyline, `_road_line`; exact point-to-segment distances), and the road surface:
    those whose centroid lies on one of `d`'s road cells (`dc.data.road_cells`: the ground's road and pit polygons) in
    x and z, within ROAD_RISE of its height. Both in the Dreamcast's frame. ConvertError for an `m` that is not a
    pol.Model of the pol types (`model.triangle_meshes`' checks), a `d` that is not a CourseData, or road points or
    road cells without finite positions."""
    model.triangle_meshes(m)
    if not isinstance(d, CourseData):
        raise ConvertError(f"d {type(d).__name__} is not a dc.data.CourseData")
    a, b = _road_line(d)
    try:
        cells = road_cells(d)
    except DecodeError as err:
        raise ConvertError(f"road cells: {err}") from err
    corners = np.array([(v.x, v.y, v.z) for t in pol.triangles(m) for v in t[:3]], dtype=float).reshape(-1, 3, 3)
    near = _near_line(corners.reshape(-1, 3), a, b)
    hit = near.reshape(-1, 3).any(axis=1) | _on_cells(corners.mean(axis=1), cells)
    return frozenset(int(k) for k in np.flatnonzero(hit))


THIN = {3: {34: 0.75, 36: 0.7, 38: 0.75, 40: 0.7, 42: 0.9, 50: 0.7, 56: 0.9, 59: 0.8, 61: 0.79, 62: 0.85},
        6: {136: 0.5}}
"""Per course n, the kept fraction per object of its kept model (`kept_model`) that `model_scenery` thins to
(`model.thin`; objects not named keep 1). Chosen from Task 6's counts (Phase 7 Task 7, docs/spike/2026-11-convert.md,
"Thinning"), frozen: Task 6's lap (every frame's visible cells and count from the cap tap, the drawn count and the
cars' share from the geometry log) replayed on the thinned cells, the object most drawn on the frames still within
150 of the bar (4,251) or of the cap (5,000) thinned by 0.05 at a time, least first, until none is. Desert City:
objects 34-42 along segments 40-140 (the canyon through the hairpin to the first checkpoint), 50-62 along segments
230-420 (the far loop); 61 stops at 0.79, where its road-side triangles and seams leave no collapse. National Park
Speedway (Phase 10 Task 3) needs none: by the same recipe on its re-meshed model (REMESH) with its quad tolerances
(QUAD), its replayed lap is already within the target, the bar less THIN_MARGIN (tests/test_courses_mame.py, 300:
3,951 drawn, a count of 4,700), set before measuring (fix round 1: 3,909 and 4,423). Rin Rin Rink (Phase 11 Task 3,
with its view shape, VIEW): object 136, a building beside the track (glass and facade textures 30-32, 158), at 0.5,
the one object whose thinning earns its place on the frames still over the target (165 of its 413 records go; the
course's other objects lose 0-67 each: strips, as National Park Speedway's)."""


QUAD: dict[int, tuple[float, float]] = {4: (10.0, 0.125), 6: (10.0, 0.125), 5: (10.0, 0.125), 7: (10.0, 0.125)}
"""Per course n, the quad tolerances `model.convert_model` merges its scenery's triangles with (degrees between the two
normals; the fourth corner's shear off the affine prediction as a fraction of its texture's smaller side) in place of
model.QUAD_ANGLE and model.QUAD_UV (Phase 10 Task 3, the controller's lever 2; fix round 1's ruling I1, set before
measuring: a fold of at most 10 degrees, a shear of at most 1/8 of the texture; docs/spike/2026-11-courses-4-7.md,
"National Park Speedway"; Rin Rin Rink and Circuit Pixie at the same bound, Phase 11 Tasks 3 and 4; Mermaid Lake too,
v0.1.2, to keep CELL_CUT's cut in budget). A course not
named keeps those."""
CELL_CUT: dict[int, float] = {7: 32.0}
"""Per course n, the world units a scenery polygon may reach past its own cell before `model.convert_model` cuts it on
the cell grid (v0.1.2; docs/follow-ups/v0.1.2.md, "Mermaid Lake's start camera"): Mermaid Lake's start straight is a
few strips about 232 units long filed one cell ahead of the grid, which the start's fly-by never opens, so the
backdrop showed through. 32 (Pascal took the recommendation, 15 Nov 2026): 420 of CS7's polygons reach further (the
start straight's long strips 82-146 units). The cut alone put the lap's maximum at 3,985 drawn at frame 1715 (64: 3,963),
over the 3,951 target, so CS7 also takes QUAD's tolerances (as 4-6): 3,790 (tests/test_courses_mame.py LAKE_MAX). A
course not named is not cut (its bytes unchanged)."""
ROAD_COPY: dict[int, tuple[float, float]] = {n: (16.0, 10.0) for n in (3, 4, 5, 6, 7)}
"""Per course n (its NORMAL conversion and its reverse alike), (margin, band) for `model.convert_model`'s `road_copy`
(v0.2; Pascal's drive of Rin Rin Rink: the road vanished for a moment at three spots a lap): each near-flat scenery
polygon within `band` units of the road line is filed also in each neighbouring cell it comes within `margin` units of,
so the road under the camera draws when the game's culling has dropped the cell it lies in (docs/follow-ups/
v0.2.md). Measured with dtools/mame/lua/holes_log.lua on the autopilot's 200 s race: Rin Rin Rink's hole runs 14 -> 0
on the road (2 left where the car is off the road in the grass) and its reverse's 3 -> 0, the polygon count's peak
4,933 -> 4,983 and 4,843 -> 4,883, under OBC's 5,000 (a band of 20 units peaked at 5,102; cutting on the cell grid
first, CELL_CUT 0, cost some 600 more). On all five (and their reverses), the hole runs on the road at (16, 10):
Desert City 4 -> 0 (reverse 9 -> 0), National Park Speedway 8 -> 4 (its other four a gap in the ground beside the road,
not the culling: a wider band, or CELL_CUT 32, changes nothing; reverse 4 -> 0), Circuit Pixie 10 -> 4 (the four with the
car off the road on its infield; reverse 6 -> 3, the same), Mermaid Lake 3 -> 0 (reverse 2 -> 0). A course not named
copies nothing (its bytes unchanged); and only `converted_set(..., road_copy=True)` copies (release.release_set with
`direction`, v0.2's chain), so every earlier set and pin, and the release chain without the screen (v0.1.2), hold."""
REMESH: dict[int, float] = {4: 0.1, 6: 0.1, 5: 0.1}
"""Per course n, the bound (world units) `model.remesh` re-meshes its kept model to before thinning (Phase 10 Task 3,
lever 3: 0.1, set before measuring; Rin Rin Rink and Circuit Pixie the same, Phase 11 Tasks 3 and 4). A course not
named is not re-meshed."""

REVERSE_THIN: dict[int, dict[int, float]] = {6: THIN[6]}
REVERSE_QUAD: dict[int, tuple[float, float]] = {3: (10.0, 0.125), 4: (10.0, 0.125), 5: (10.0, 0.125), 6: (10.0, 0.125),
                                                7: (10.0, 0.125)}
REVERSE_REMESH: dict[int, float] = {3: 0.4, 4: 0.1, 5: 0.1, 6: 0.4}
REVERSE_NAMED_PROPS: dict[int, tuple[int, ...]] = {n: NAMED_PROPS[n] for n in (4, 5, 6, 7)}
"""Per course n, the budget settings of its reverse model (CSnR_POL.BIN, `converted_set(..., variant="R")`) in place of
THIN, QUAD, REMESH and NAMED_PROPS (M9 stage 1): the R model has the plain one's object count but other detail (Desert
City: 20 of 68 objects differ in mesh count, among them THIN[3]'s), so the plain model's per-object fractions and props
do not carry over; CELL_CUT (geometric) does. Desert City's reverse unreduced drew 4,507 at frame 1771 (the start),
556 over BAR - THIN_MARGIN: the levers in the project's order, quad merge and re-mesh first (docs/spike/
2026-11-reverse.md, "Desert City in reverse").

Courses 4-7 (M9 stage 2, step 3, measured in MAME 17 Nov 2026): their R models hold the plain models' objects at the
same indices (same object count; NAMED_PROPS' and THIN[6]'s objects identical, or the same object with its texture
indices one off where the R texture table is renumbered: National Park's object 3, Rin Rin Rink's 136), so their props
and THIN carry over; Circuit Pixie's and Rin Rin Rink's animations drawn all at once filled the polygon cap without
them (4,683 / 4,817 drawn, counts 5,659 / 5,876). With the plain QUAD and REMESH and the props the laps drew 3,811
(count 4,406), 3,567 (4,312, VIEW_12), 3,805 (4,725: the count over, VIEW_12) and 3,894 (4,579); Rin Rin Rink's
reverse then takes REMESH 0.4: 3,660 (4,564). Circuit Pixie's reverse fits a larger view (REVERSE_VIEW)."""

VIEW_12 = frozenset({(0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (2, -2),
                     (2, 2)})
"""12 of the culling half-block's 15 cells, (d, c) as VIEW names them: the row the reference cell is in and the next (5
cells each) and the two far corners; the far row's middle three are left out. Rin Rin Rink's and Circuit Pixie's view."""
VIEW: dict[int, frozenset] = {6: VIEW_12, 5: VIEW_12}
"""Per course n, the cells of the culling block its view opens, in every heading quadrant: (d, c), d cells ahead of
the reference cell's row or column (0-2: the half-block a quadrant's view-test table reaches) and c across (-2..2;
left-right symmetric). The course's own view-test tables are the shared ones (course.VIEW_TESTS) with each point's
mask cut to those cells (`view_tables`), read through course.view_ids (`viewed_set`).

- Rin Rin Rink (Phase 11 Task 3, Pascal's decision 2026-11-12, "Per-course view, largest fit";
  docs/spike/2026-11-courses-4-7.md, "Rin Rin Rink"): VIEW_12, the largest shape (most cells, then the farthest reach
  ahead) whose lap meets the target with QUAD[6], REMESH[6], NAMED_PROPS[6] and THIN[6] in MAME (fix round 1: the
  replay put it 47 drawn and 82 counted over, MAME measured it within: 3,816 drawn, a count of 4,694); every shape
  keeping one of the far row's middle three is over.
- Circuit Pixie (Phase 11 Task 4, Pascal's ruling 2026-11-12, option A: the smaller view, all scenery kept;
  docs/spike/2026-11-courses-4-7.md, "Circuit Pixie"): VIEW_12 too. With QUAD[5], REMESH[5] and NAMED_PROPS[5] at the
  stock view only the start (40 cars bunched on the grid) is over (4,387 drawn, a count of 5,293); with VIEW_12 MAME
  draws 3,723 and counts 4,521; no 13- or 14-cell shape fits.

A course not named keeps the shared view, and a set without a VIEW course is built without view_ids (its bytes
unchanged)."""


VIEW_8 = frozenset({(0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (1, -1), (1, 0), (1, 1)})
"""8 cells: the reference cell's row (5) and the middle three of the next."""
VIEW_9 = VIEW_8 | {(2, 0)}
"""9 cells: VIEW_8 and the far row's middle cell."""
VIEW_13 = VIEW_12 | {(2, 0)}
"""13 cells: VIEW_12 and the far row's middle cell."""
REVERSE_VIEW: dict[int, frozenset] = {**VIEW, 9: VIEW_9, 10: VIEW_9, 13: VIEW_13, 14: VIEW_12}
"""VIEW with the reverse ids' own views (M9 stage 2, step 2; the reverse chain's viewed_set, not the release's). Sea-Side
Street Galaxy and Dinosaur Canyon in reverse (ids 9, 10), under the arcade's scenery, go past BAR and hit the cap driven
backwards; Pascal's rule for them is BAR and no cap hit (17 Nov 2026), per-id views first, and as on Rin Rin Rink the
largest shape whose lap fits: VIEW_9 on both (the only symmetric shape between VIEW_8 and rows 0-1). Measured in MAME over the autopilot's lap (drawn maximum, frames over BAR, cap hits):

| view | id 9 | id 10 |
|---|---|---|
| all 15 cells | 4,535+ (the lap not measured to its end), 212, 134 | 4,753, 410, 264 |
| VIEW_12 | 4,817, 73, 70 | 4,657, 39, 22 |
| rows 0-1 (10 cells) | 4,567, 40, 25 | 4,262, 11, 0 |
| VIEW_9 | 4,102, 0, 0 | 4,059, 0, 0 |
| VIEW_8 | 4,003, 0, 0 | 3,869, 0, 0 |

Circuit Pixie and Rin Rin Rink in reverse (ids 13, 14; the converted courses' rule: drawn at most BAR - THIN_MARGIN
3,951 and the count at most 4,700), with REVERSE_NAMED_PROPS, REVERSE_THIN and REVERSE_REMESH, drawn max / count max:
all 15 cells 4,098 / 4,918 and 4,330 / 5,218 (235 cap hits); VIEW_12 plus the far row's two flanking cells (14)
3,916 / 4,702 (the count 2 over) and 4,079 / 5,057; VIEW_13 3,749 / 4,528 and 4,059 / 4,837; VIEW_12 3,567 / 4,312
and 3,660 / 4,564. The largest that fits: VIEW_13 on id 13, VIEW_12 on id 14 (the plain course's)."""


def view_tables(rs: RomSet, cells) -> bytes:
    """The four view-test tables (course.VIEW_TESTS' quadrant order, course.VIEW_TEST_SIZE bytes each) of a view that
    opens `cells` ((d, c) as VIEW names them): the shared tables of `rs` with every point's mask cut to the block
    cells those name in its quadrant (quadrant 0 looks to +row, 1 to -column, 2 to -row, 3 to +column; block index i
    is row i // 5 - 2, column i % 5 - 2 from the reference cell); offsets and the unread words as they are.
    ConvertError for `rs` not a RomSet, `cells` that is not a collection of (d, c) with d 0-2 and c -2..2, or a shape
    that is not left-right symmetric (the rotation names "across" in absolute terms, so a one-sided shape would flip
    sides between opposite headings)."""
    rs = _romset(rs)
    if not isinstance(cells, (set, frozenset, tuple, list)) or not all(
            isinstance(k, tuple) and len(k) == 2 and is_int(k[0]) and is_int(k[1]) and 0 <= k[0] <= 2
            and -2 <= k[1] <= 2 for k in cells):
        raise ConvertError(f"view cells {cells!r:.60} are not (ahead 0-2, across -2..2) pairs")
    if {(d, -c) for d, c in cells} != set(cells):
        raise ConvertError(f"view cells {sorted(cells)!r:.80} are not left-right symmetric: the quadrants' rotation names "
                           f"across in absolute terms (quadrants 0 and 2 both send c to column c), so a one-sided shape "
                           f"would flip sides between opposite headings")
    turn = (lambda d, c: (d, c), lambda d, c: (c, -d), lambda d, c: (-d, c), lambda d, c: (c, d))
    out = b""
    for q in range(4):
        keep = {turn[q](d, c) for d, c in cells}
        mask = sum(1 << i for i in range(C.BLOCK_CELLS) if (i // C.BLOCK - 2, i % C.BLOCK - 2) in keep)
        ptr = struct.unpack("<I", rs.read_i960(C.VIEW_TESTS + 4 * q, 4))[0]
        table = bytearray(rs.read_i960(ptr, C.VIEW_TEST_SIZE))
        for off in C.VIEW_TEST_POINTS:
            word = struct.unpack_from("<I", table, off + 8)[0]
            struct.pack_into("<I", table, off + 8, word & mask)
        out += bytes(table)
    return out


def viewed_set(rs: RomSet, views: dict | None = None) -> RomSet:
    """`rs` (a widened set, every course converted) with each course `views` names (default VIEW) given its view:
    its tables (`view_tables`) at the first free bytes of ID_DATA_FREE, then course.view_ids. `rs` itself when no
    course is named. The last step of a chain (after the select): nothing after it allocates. ConvertError for a
    `views` that is not a dict of course ids to view cells and as `view_tables` and the allocator; CourseError from
    course.view_ids."""
    rs = _romset(rs)
    views = VIEW if views is None else views
    if not isinstance(views, dict) or not all(is_int(k) and 0 <= k < C.COURSE_IDS for k in views):
        raise ConvertError(f"views {views!r:.60} is not a dict of course ids to view cells")
    if not views:
        return rs
    ptrs = {}
    for n in sorted(views):
        blob = view_tables(rs, views[n])
        at = _first_fit(rs, "data", ID_DATA_FREE, len(blob), f"course {n}'s view-test tables")
        rs = rs.with_patch("data", at, blob)
        ptrs[int(n)] = tuple(DATA_BASE + at + C.VIEW_TEST_SIZE * q for q in range(4))
    return C.view_ids(rs, ptrs)


VIEW_15 = frozenset((d, c) for d in range(3) for c in range(-2, 3))
"""The culling half-block's 15 cells: the stock view (`view_tables` of it are the shared tables' bytes)."""
FAR_DEPTH, FAR_HALF = 4, 3
"""The far view's reach in cells from the reference cell: rows ahead (the game's block: 2) and columns to each side
(the block: 2). 20 cells a quadrant past the block, 45 with it (the visible list holds 63)."""
FAR_SHAPE = tuple([(d, c) for d in range(3, FAR_DEPTH + 1) for c in range(-FAR_HALF, FAR_HALF + 1)]
                  + [(d, c) for d in range(3) for c in range(-FAR_HALF, FAR_HALF + 1) if abs(c) > 2])
"""The far view's cells past the block, (d, c) as VIEW names them: rows 3..FAR_DEPTH ahead across
-FAR_HALF..FAR_HALF, and the columns past the block beside rows 0-2."""
BLOCK_OFFSETS = tuple((r, c) for r in range(-2, 3) for c in range(-2, 3) if (r, c) != (0, 0))
"""The 5 x 5 block's cells but the reference cell, (row, column): a far id's far table opens them all first, whatever
the heading (the game's view test opens up to 15 of them; the rest lie beside and behind, where v0.2's road holes
were: the camera trails the car)."""
FAR_IDS = (3, 4, 5, 6, 7, 11, 12, 13, 14, 15)
"""The ids `far_viewed_set` gives the whole block, the far view and POLYGON_CAP: the converted courses and their
reverses. The arcade's own courses and their reverses (0-2, 8-10) get an empty far table and the cap course.OBC."""
POLYGON_CAP = 28000
"""The frame's polygon cap of a far id (course.CAP_MAX 30,000). Measured in MAME 20 Nov 2026 on the autopilot's
160 s races with FAR_SHAPE: Desert City's count peaked at 13,168 (docs/follow-ups/v0.3.md)."""


def far_offsets(shape, q: int) -> tuple[tuple[int, int], ...]:
    """`shape`'s cells ((d ahead, c across)) as (row, column) offsets in heading quadrant `q` (`view_tables`' turn:
    quadrant 0 looks to +row, 1 to -column, 2 to -row, 3 to +column)."""
    turn = (lambda d, c: (d, c), lambda d, c: (c, -d), lambda d, c: (-d, c), lambda d, c: (c, d))
    return tuple(turn[q](d, c) for d, c in shape)


def far_viewed_set(rs: RomSet, views: dict | None = None, far_ids=FAR_IDS, shape=FAR_SHAPE,
                   cap: int = POLYGON_CAP) -> RomSet:
    """`rs` (a widened set, every course converted) with v0.3's view, in place of `viewed_set` (the chain's last
    allocating step): every id gets its own four view-test tables, `views`' as `viewed_set` writes them (default
    REVERSE_VIEW; VIEW_15, the stock view, for an id not named), so the game's own view test and what the race reads
    of it are v0.2's; each is followed by its far table (course.far_table): for an id of `far_ids` the cap `cap`,
    BLOCK_OFFSETS and `shape` turned to the quadrant, for any other id course.OBC and no cell. Then course.view_ids
    and course.far_view. ConvertError for `views` as `viewed_set`, `far_ids` not ids 0-15, a `shape` that is not
    left-right symmetric (as `view_tables`) or reaches into the block; CourseError from the course steps."""
    rs = _romset(rs)
    views = REVERSE_VIEW if views is None else views
    if not isinstance(views, dict) or not all(is_int(k) and 0 <= k < C.COURSE_IDS for k in views):
        raise ConvertError(f"views {views!r:.60} is not a dict of course ids to view cells")
    if not isinstance(far_ids, (tuple, list)) or not all(is_int(k) and 0 <= k < C.COURSE_IDS for k in far_ids):
        raise ConvertError(f"far ids {far_ids!r:.60} are not course ids")
    if not isinstance(shape, (tuple, list)) or {(d, -c) for d, c in shape} != set(shape) \
            or any(max(abs(d), abs(c)) <= 2 for d, c in shape):
        raise ConvertError(f"far shape {shape!r:.60} is not a left-right symmetric tuple of (d, c) cells past the block")
    step = C.VIEW_TEST_SIZE + C.FAR_TABLE_SIZE
    ptrs = {}
    for n in range(C.COURSE_IDS):
        tables = view_tables(rs, views.get(n, VIEW_15))
        blob = b"".join(tables[C.VIEW_TEST_SIZE * q:C.VIEW_TEST_SIZE * (q + 1)]
                        + (C.far_table(cap, BLOCK_OFFSETS + far_offsets(shape, q)) if n in far_ids
                           else C.far_table(C.OBC, ())) for q in range(4))
        at = _first_fit(rs, "data", ID_DATA_FREE, len(blob), f"course {n}'s view-test and far tables")
        rs = rs.with_patch("data", at, blob)
        ptrs[n] = tuple(DATA_BASE + at + step * q for q in range(4))
    return C.far_view(C.view_ids(rs, ptrs))


LIFT: dict[int, float] = {7: 41.0}
"""Per course n, the world units its data and model are raised by before converting (`lifted_data`, `lifted_model`;
converted_set applies them), so that every corner of its ground lies above 0 (Phase 10 Task 4, docs/spike/
2026-11-courses-4-7.md, "Mermaid Lake"). The TGP chooses, of the polygons a height query hits, the one with the
smallest | |height| - y | (docs/formats/model2-tgp-ground.md, "Choice"): above 0 the nearest to the car, below 0 the
nearest to 0. Mermaid Lake's road dives to -36 in the tunnel under segment 81 (segments 316-345; its lowest ground
corner -40.5), where the arcade's ground answered the surface above it on every lane point of segments 322-335 and
112 AI points: a car met a wall there and turned back. Raised by 41 every lane and AI point gets its own polygon. The
arcade's originals lie at 0-139, Desert City at -57.5..57.5 (no polygon below 0 under another), National Park
Speedway at -1.2..53.7: none is named, so they keep their bytes."""


def lifted_data(d: CourseData, dy) -> CourseData:
    """`d` with every y raised by `dy` world units, each rounded to single precision as the disc stores it: the used
    lanes of each road point (lane_lo..lane_hi; the unused slots keep their placeholder), every AI point, every cell's
    corners and centre. Nothing else changes (x, z, angles, the grid, lists, steps, attributes); a shift's mark
    (`shifted_data`) is kept. ConvertError for a `d` that is not a CourseData of the dc.data types or a `dy` that is not
    a finite number, or a raised y past single precision."""
    if not isinstance(d, CourseData):
        raise ConvertError(f"d {type(d).__name__} is not a dc.data.CourseData")
    return _moved_data(d, 1, _offset(dy, "y", "raised"))


def lifted_model(m: pol.Model, dy) -> pol.Model:
    """`m` with every vertex's y and every object's bounding-sphere centre y raised by `dy` world units, rounded to
    single precision as the file stores them (normals, u, v, re-uses and everything else unchanged; a shift's mark
    kept). ConvertError for an `m` that is not a pol.Model of the pol types (`model.triangle_meshes`' checks) or a `dy`
    that is not a finite number, or a raised y past single precision."""
    model.triangle_meshes(m)
    out = _moved_model(m, 1, _offset(dy, "y", "raised"))
    object.__setattr__(out, "_lift", _lift_of(m) + float(dy))      # not a field: eq, repr and the re-mesh key unchanged
    return out


def _lift_of(m) -> float:
    """The world units `m` was raised by through `lifted_model` (0.0 for a model as decoded; a model rebuilt from a
    lifted one by `dataclasses.replace` loses the mark, so it reads as unlifted)."""
    return getattr(m, "_lift", 0.0)


def _check_lifted(d: CourseData, n: int) -> None:
    """ConvertError when course `n` is one LIFT names and a corner of `d`'s ground lies below 0, so `d` was not raised
    by `lifted_data(d, LIFT[n])` (fix round 1 of Phase 10 Task 4, M4: converted_course refuses rather than lifts, so a
    caller cannot build the course with its tunnel answering the roof, nor lift it twice; converted_set lifts it)."""
    if n not in LIFT:
        return
    low = min(c[1] for cell in d.cells for c in cell.corners)
    if low < 0:
        raise ConvertError(f"course {n}: its ground reaches {low:.1f}, below 0: convert it lifted by LIFT[{n}] "
                           f"({LIFT[n]}: lifted_data and lifted_model, as converted_set does)")


SHIFT: dict[int, float] = {6: -64.0}
"""Per course n, the world units added to every x of its data and model before converting (`shifted_data`,
`shifted_model`; converted_set applies them), so that every vertex, lane, AI point and ground corner lies inside the
world (model.WORLD, -1024..1023 on x and the arcade's z: the scenery cell rule and the TGP grid wrap outside it). Rin
Rin Rink (Phase 11 Task 3, docs/spike/2026-11-courses-4-7.md, "Rin Rin Rink"): its model reaches x -928.9..1073.1
(objects 158 and 159 of 169 past 1023, by 12 and 4 vertices), its road -651.7..636.8, its ground -735.9..657.9, every z
inside; any shift in [-95.1, -49.1] fits, and -64 (four TGP grid squares, half a scenery cell) about centres it
(-992.9..1009.1). The arcade's z is the Dreamcast's negated, its x the Dreamcast's, so the shift is in the disc's x
too. What the conversion takes from elsewhere is position-free (the checkpoints and the grid base are segments, the
fly-by keyframes are in car 0's frame, the HUD map's place record is framed on the shifted road) or moves with the
road line; the origin rule measures from the Dreamcast's origin, moved (`origin_props`). No other course is named, so
they keep their bytes."""


def shifted_data(d: CourseData, dx) -> CourseData:
    """`d` with every x moved by `dx` world units, each rounded to single precision as the disc stores it: the used
    lanes of each road point (the unused slots keep their placeholder), every AI point, every cell's corners and
    centre. Nothing else changes (y, z, angles, the grid and lists (the Dreamcast's, which the conversion checks but
    rebuilds, ground.convert_ground), steps, attributes). The result carries the shift (`_shift_of`, a mark, not a
    field: eq and repr are the plain data's; `lifted_data` keeps it, a `dataclasses.replace` loses it), which
    converted_course checks against SHIFT (`_check_shifted`). ConvertError for a `d` that is not a CourseData of the
    dc.data types or a `dx` that is not a finite number, or a moved x past single precision."""
    if not isinstance(d, CourseData):
        raise ConvertError(f"d {type(d).__name__} is not a dc.data.CourseData")
    out = _moved_data(d, 0, _offset(dx, "x", "shifted"))
    object.__setattr__(out, "_shift", _shift_of(d) + float(dx))
    return out


def shifted_model(m: pol.Model, dx) -> pol.Model:
    """`m` with every vertex's x and every object's bounding-sphere centre x moved by `dx` world units, rounded to
    single precision as the file stores them (normals, u, v, re-uses and everything else unchanged), marked with the
    shift as `lifted_model` marks the lift (`_shift_of`; the lift's mark kept). The model may lie outside the world
    before (the reason to shift it), so its types are checked here and its positions by the scenery's converter after.
    ConvertError for an `m` that is not a pol.Model of the pol types or a `dx` that is not a finite number, or a moved
    x past single precision."""
    if not isinstance(m, pol.Model):
        raise ConvertError(f"m {type(m).__name__} is not a pol.Model")
    move = _offset(dx, "x", "shifted")
    if not isinstance(m.objects, tuple):
        raise ConvertError(f"model: objects {m.objects!r:.60} are not a tuple of pol.Object")
    for oi, ob in enumerate(m.objects):
        if not isinstance(ob, pol.Object) or not isinstance(ob.meshes, tuple) or not all(
                isinstance(me, pol.Mesh) and isinstance(me.strips, tuple) and all(
                    isinstance(st, pol.Strip) and isinstance(st.entries, tuple) for st in me.strips)
                for me in ob.meshes):
            raise ConvertError(f"model: object {oi} {ob!r:.60} is not a pol.Object of pol meshes and strips")
    out = _moved_model(m, 0, move)
    object.__setattr__(out, "_shift", _shift_of(m) + float(dx))    # not a field: eq, repr and the re-mesh key unchanged
    return out


def _shift_of(x) -> float:
    """The world units `x` (a CourseData or a pol.Model) was moved by in x through `shifted_data` / `shifted_model`
    (0.0 as decoded; one rebuilt by `dataclasses.replace` loses the mark, so it reads as unshifted)."""
    return getattr(x, "_shift", 0.0)


def _check_shifted(d: CourseData, n: int) -> None:
    """ConvertError when `d` was not moved by SHIFT's amount for course `n` (`_shift_of`; 0 for a course SHIFT does
    not name): converted_course refuses rather than shifts, as it does a course LIFT names (`_check_lifted`), so a
    caller cannot build course `n` with its road and ground off its scenery, nor shift it twice; converted_set shifts
    it. The data alone cannot show it, as the lift's ground below 0 does: Rin Rin Rink's ground lies inside the world
    either way, only its model does not (`scenery_model` refuses that, naming the objects)."""
    want, got = float(SHIFT.get(n, 0.0)), _shift_of(d)
    if got != want:
        named = (f"SHIFT[{n}] ({want}: shifted_data and shifted_model, as converted_set does)" if n in SHIFT else
                 "SHIFT names none for it: convert it as the disc stores it")
        raise ConvertError(f"course {n}: its data was shifted by {got} in x: {named}")


def _outside_world(m) -> dict[int, tuple[float, float]]:
    """{object: (its least, its largest x)} of the objects of `m` with a vertex outside model.WORLD (x, or the
    arcade's z, the negated z); objects, meshes or entries that are not the pol types are skipped (the converter's own
    checks name them)."""
    lo, hi = model.WORLD
    out = {}
    for oi, ob in enumerate(getattr(m, "objects", ()) or ()):
        ps = [(e.x, e.z) for me in getattr(ob, "meshes", ()) or () for st in getattr(me, "strips", ()) or ()
              for e in getattr(st, "entries", ()) or () if isinstance(e, pol.Vertex) and is_number(e.x)
              and is_number(e.z)]
        if any(not (lo <= x <= hi and lo <= -z <= hi) for x, z in ps):
            out[oi] = (min(x for x, _ in ps), max(x for x, _ in ps))
    return out


def _offset(dv, axis: str, verb: str):
    """v -> v + dv as a single-precision value (a Python float), `axis` and `verb` naming it in messages; ConvertError
    for a `dv` that is not a finite number (bool refused), a `v` that is not, or a result past single precision. The
    lift's and the shift's rule."""
    if not is_number(dv):
        raise ConvertError(f"d{axis} {dv!r:.30} is not a finite number")

    def move(v):
        if not is_number(v):
            raise ConvertError(f"{axis} {v!r:.30} is not a finite number")
        with np.errstate(over="ignore"):
            r = np.float32(float(v) + float(dv))
        if not np.isfinite(r):
            raise ConvertError(f"{axis} {v!r:.30} {verb} by {dv!r:.30} is past single precision")
        return float(r)
    return move


def _moved_data(d: CourseData, k: int, move) -> CourseData:
    """`d` with coordinate `k` (0 x, 1 y) of the used lanes, the AI points and the cells' corners and centres through
    `move` (`_offset`), the shift's mark kept; ConvertError for parts that are not the dc.data types."""
    axis = "xyz"[k]

    def pt(p):
        return tuple(move(v) if j == k else v for j, v in enumerate(p))
    try:
        road = tuple(dataclasses.replace(rp, lanes=tuple(dataclasses.replace(ln, **{axis: move(getattr(ln, axis))})
                                                         if rp.lane_lo <= j <= rp.lane_hi else ln
                                                         for j, ln in enumerate(rp.lanes)))
                     for rp in d.road)
        lines = tuple(tuple(dataclasses.replace(q, **{axis: move(getattr(q, axis))}) for q in line)
                      for line in d.lines)
        cells = tuple(dataclasses.replace(c, corners=tuple(pt(p) for p in c.corners), centre=pt(c.centre))
                      for c in d.cells)
    except (TypeError, ValueError, AttributeError) as e:
        raise ConvertError(f"d: {e}") from e
    out = dataclasses.replace(d, road=road, lines=lines, cells=cells)
    object.__setattr__(out, "_shift", _shift_of(d))
    return out


def _moved_model(m: pol.Model, k: int, move) -> pol.Model:
    """`m` with coordinate `k` (0 x, 1 y) of every vertex and bounding-sphere centre through `move` (`_offset`), the
    lift's and the shift's marks kept (the caller adds its own)."""
    axis = "xyz"[k]

    def entry(e):
        return dataclasses.replace(e, **{axis: move(getattr(e, axis))}) if isinstance(e, pol.Vertex) else e
    try:
        out = dataclasses.replace(m, objects=tuple(
            dataclasses.replace(ob, sphere=tuple(move(v) if j == k else v for j, v in enumerate(ob.sphere)),
                                meshes=tuple(dataclasses.replace(me, strips=tuple(
                                    dataclasses.replace(st, entries=tuple(entry(e) for e in st.entries))
                                    for st in me.strips)) for me in ob.meshes))
            for ob in m.objects))
    except (TypeError, ValueError, AttributeError) as e:
        raise ConvertError(f"m: {e}") from e
    for mark in ("_lift", "_shift"):
        object.__setattr__(out, mark, getattr(m, mark, 0.0))
    return out


REMESH_CACHE = Path("cache") / "remesh"
"""Where `remeshed` keeps `model.remesh`'s results, under config.out_dir() (git-ignored, never committed; Phase 10 Task
3 fix round 2: an id 4 re-mesh takes about five minutes)."""


def _remesh_key(m: pol.Model, eps) -> str:
    """The cache key of `model.remesh(m, eps)`: the sha1 of the model (its repr: every value), eps, and the source of
    the code that computes it (dtools/convert/model.py, dtools/dc/pol.py) and numpy's version, so a change to any of
    them is a miss."""
    h = hashlib.sha1()
    for part in (repr(m).encode(), repr(float(eps)).encode(), Path(model.__file__).read_bytes(),
                 Path(pol.__file__).read_bytes(), np.__version__.encode()):
        h.update(hashlib.sha1(part).digest())
    return h.hexdigest()


def remeshed(m, eps, cache: Path | None = None) -> pol.Model:
    """`model.remesh(m, eps)`, kept on disk: the result is stored as `<cache>/<key>.pkl` (`cache` default
    config.out_dir() / REMESH_CACHE; the key `_remesh_key`) and read back on the next call with the same inputs and
    code. A file that does not hold (its key, a pol.Model) is ignored and rewritten; the write goes to a temporary file
    renamed into place (removed again if the write fails). ConvertError as model.remesh (checked before the cache is
    consulted).

    The cache is local and trusted: a file is read with `pickle.loads`, so a file planted in `out/cache/remesh` runs
    code when it is loaded (the key and type checks come after the load). Never share the folder between users or
    machines, or fill it from anywhere but this function (final review of Phase 10, Minor 4). What is cheap to
    validate is: the stored key, the type, and that the file's name is the key of this call."""
    if not isinstance(m, pol.Model):
        raise ConvertError(f"m {type(m).__name__} is not a pol.Model")
    if not is_number(eps) or not eps > 0:
        raise ConvertError(f"eps {eps!r:.30} is not a positive finite number")
    key = _remesh_key(m, eps)
    folder = Path(cache) if cache is not None else config.out_dir() / REMESH_CACHE
    path = folder / f"{key}.pkl"
    try:
        stored = pickle.loads(path.read_bytes())
        if (isinstance(stored, tuple) and len(stored) == 2 and stored[0] == key and isinstance(stored[1], pol.Model)
                and path.name == f"{stored[0]}.pkl"):
            return stored[1]
    except Exception:                                   # noqa: BLE001 (a missing or damaged file is a miss)
        pass
    out = model.remesh(m, eps)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{key}.{os.getpid()}.tmp"
    try:
        tmp.write_bytes(pickle.dumps((key, out), protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)                     # a failed write leaves no .tmp behind
        raise
    return out


def scenery_model(m, tex_blob, d, keep=None, remesh=None, props=()) -> pol.Model:
    """`m` as the scenery converts it: `kept_model(m, tex_blob, d, props)`, re-meshed to `remesh` world units
    (`model.remesh` through its disk cache, `remeshed`; None: not re-meshed), then thinned (`model.thin`) to `keep`
    (object index -> kept fraction, the others whole; None or empty: no thinning) with the road triangles of `d`
    protected (`road_triangles`, of the re-meshed model). ConvertError as those, and for a `keep` that is not a dict of
    object indices of `m` to numbers in (0, 1]."""
    return _scenery_models(m, tex_blob, d, keep, remesh, props)[1]


def _scenery_models(m, tex_blob, d, keep, remesh=None, props=()) -> tuple[pol.Model, pol.Model]:
    """(kept_model(m, tex_blob, d, props), scenery_model's). A model past the world is refused naming its objects
    outside and SHIFT (a course that reaches past it converts shifted)."""
    if not isinstance(d, CourseData):
        raise ConvertError(f"d {type(d).__name__} is not a dc.data.CourseData")
    try:
        kept = kept_model(m, tex_blob, d, props)
    except ConvertError as e:
        if not (far := _outside_world(m)):
            raise
        lo, hi = model.WORLD
        raise ConvertError(f"model: objects {', '.join(map(str, far))} reach outside the world {lo}..{hi} (x "
                           f"{min(v[0] for v in far.values()):.1f}..{max(v[1] for v in far.values()):.1f}): a course "
                           f"past it converts shifted in x by SHIFT ("
                           + ", ".join(f"SHIFT[{k}] {v}" for k, v in SHIFT.items())
                           + ": shifted_data and shifted_model, as converted_set does)") from e
    if keep is None:
        keep = {}
    if not isinstance(keep, dict) or not all(is_int(k) and 0 <= k < len(m.objects) for k in keep):
        raise ConvertError(f"keep {keep!r:.60} is not a dict of object indices to kept fractions")
    work = kept if remesh is None else remeshed(kept, remesh)
    if not keep:
        return kept, work
    fractions = {int(k): v for k, v in keep.items()}
    return kept, model.thin(work, lambda oi: fractions.get(oi, 1.0), protected=road_triangles(work, d))


def mesh_colours(m, kept, drawn) -> dict[int, tuple[float, float, float]]:
    """`_face_colours(m)` extended to `drawn`'s meshes: a mesh `model.thin` rebuilt takes the face colour of the
    kept mesh it came from (same object and mesh index in `kept`, `kept_model(m, ...)`). ConvertError for an `m` that is
    not a pol.Model of the pol types (`model.triangle_meshes`' checks) or as `_face_colours`, and for `kept` and
    `drawn` that are not models whose objects and meshes pair up, `kept`'s being `m`'s own."""
    for name, x in (("m", m), ("kept", kept), ("drawn", drawn)):
        if not isinstance(x, pol.Model) or not isinstance(x.objects, tuple):
            raise ConvertError(f"{name} {type(x).__name__} is not a pol.Model")
    model.triangle_meshes(m)
    face = _face_colours(m)
    if len(kept.objects) != len(drawn.objects) or any(
            not isinstance(a, pol.Object) or not isinstance(b, pol.Object) or len(a.meshes) != len(b.meshes)
            for a, b in zip(kept.objects, drawn.objects)):
        raise ConvertError("drawn is not kept thinned: their objects and meshes do not pair up")
    for a, b in zip(kept.objects, drawn.objects):
        for ma, mb in zip(a.meshes, b.meshes):
            if id(ma) not in face:
                raise ConvertError("kept is not m less some meshes: a kept mesh is not one of m's")
            face[id(mb)] = face[id(ma)]
    return face


def _fit_coefficients(pieces: dict, sizes, headers: dict) -> tuple[dict, int]:
    """Ruling 4: `model.convert_model` sets each textured record's distance coefficient (attr bits 27-31) from the
    Dreamcast texture's size (`sizes`, (width, height) per Dreamcast index); where the converted texture (`headers`,
    Dreamcast index -> TexHeader) is another size, the coefficient is computed again from the polygon's corners and
    (u, v) on the converted size (model's rule, `model.lod_coef` of the texel density: one floor, not a shift of the
    already floored value; about LOD_STEPS / 2 per halving or doubling of a side). (pieces with the records'
    coefficients refitted, the records whose coefficient changed). The same pieces when no texture changed size
    (Desert City: all 86 kept theirs). ConvertError for a textured record whose texture has no header."""
    if not isinstance(headers, dict) or not all(isinstance(h, T.TexHeader) for h in headers.values()):
        raise ConvertError(f"headers {headers!r:.40} are not a dict of Dreamcast texture index -> TexHeader")
    shift = {}
    for k, h in headers.items():
        if not is_int(k) or not 0 <= k < len(sizes):
            raise ConvertError(f"headers: texture {k!r:.20} is not an index into the {len(sizes)}-entry table")
        w, hh = sizes[k]
        shift[k] = (h.width, h.height) != (w, hh)
    used = {t for ps in pieces.values() for pc in ps for t in pc.textures if t >= 0}
    if missing := sorted(used - set(shift)):
        raise ConvertError(f"textures {missing[:4]} of the converted model have no converted header")
    if not any(shift[t] for t in used):
        return pieces, 0
    moved, out = 0, {}
    for cell, ps in pieces.items():
        new = []
        for pc in ps:
            recs = []
            for r, poly, t, uv in zip(pc.obj.records, P.polygons(pc.obj), pc.textures, pc.uvs):
                if t >= 0 and shift[t]:
                    coef = model._coef(poly.points, uv, (headers[t].width, headers[t].height))
                    moved += coef != r.coef
                    r = dataclasses.replace(r, attr=(r.attr & 0x07FFFFFF) | coef << 27)
                recs.append(r)
            new.append(pc._replace(obj=dataclasses.replace(pc.obj, records=recs)))
        out[cell] = tuple(new)
    return out, moved


def _face_colours(m: pol.Model) -> dict[int, tuple[float, float, float]]:
    """id(mesh) -> its face colour (red, green, blue, each clamped to 0-1) for every mesh of `m`, walked in file order
    (the order the Dreamcast sends the polygon headers): an intensity mode 1 mesh (COLOUR_MODE bits 2) its own
    MESH_COLOUR, an intensity mode 2 mesh (3) the last mode-1 mesh's before it (its own when there is none; on CS3 the
    654 mode-2 meshes store their predecessor's anyway), a packed or floating colour mesh (0, 1: the per-vertex
    colours, not carried; CS3's 22 are the raw-normal meshes, all 1.0) its own. ConvertError for colours that are not
    nine finite numbers. (`drop_textures` keeps the Mesh objects, so a kept model's meshes look up their colour
    here.)"""
    out, last = {}, None
    for oi, ob in enumerate(m.objects):
        for mi, me in enumerate(ob.meshes):
            c = me.colours
            if not isinstance(c, (tuple, list)) or len(c) != 9 or not all(is_number(v) for v in c):
                raise ConvertError(f"object {oi} mesh {mi}: colours {c!r:.60} are not nine finite numbers")
            own = tuple(min(max(float(v), 0.0), 1.0) for v in c[MESH_COLOUR][1:])
            mode = (me.params[0] >> COLOUR_MODE) & 3
            if mode == 2:
                last = own
            out[id(me)] = last if mode == 3 and last is not None else own
    return out


def _tint(me: pol.Mesh, face: dict) -> tuple[float, float, float]:
    """The colour a textured mesh multiplies its texture by: its face colour where its shading modulates
    (TSP_SHADING in MODULATE), else none (1, 1, 1)."""
    return face[id(me)] if (me.params[2] >> TSP_SHADING) & 3 in MODULATE else (1.0, 1.0, 1.0)


def _clamped(uvs, modes) -> tuple:
    """A record's (u, v) per corner with each clamped axis held to 0..1 (the Dreamcast stretches the edge texel past
    it; the arcade's header has no clamp, so the points stay inside the texture and need no wrap bit)."""
    return tuple(tuple(min(max(float(x), 0.0), 1.0) if mode == "clamp" else x for x, mode in zip(p, modes))
                 for p in uvs)


def model_scenery(rs: RomSet, d: CourseData, m: pol.Model, tex_blob: bytes, keep: dict | None = None, *,
                  slot: int = 0, quad: tuple[float, float] | None = None, remesh: float | None = None,
                  props=(), cell_cut: float | None = None, road_copy: tuple[float, float] | None = None,
                  depth: bool = False) -> Scenery:
    """A Dreamcast course model (`m`, a `CS{n}_POL.BIN`; `tex_blob` its `CS{n}_TEX.BIN`) as the Scenery of a converted
    course in slot `slot` of `rs` (0, Beginner's, on the original set; NEW_ID on a grown set: the colour tables, which
    pick the hues, are the original's on both). Phase 7 Task 5's rulings
    (docs/spike/2026-11-convert.md, "Scenery assembled"):

    - the alpha-blended decals (`decals`), the sky domes (`sky_domes`), the shadow volumes (`shadow_volumes`), the
      run-time props at the origin (`origin_props`) and the objects `props` names (NAMED_PROPS', Phase 10 Task 4) are
      dropped (`kept_model`, Tasks 6 and 7); the objects `keep`
      names thinned to their fractions, the road protected (`scenery_model`, Task 7; converted_set passes THIN's),
      after a re-mesh to `remesh` world units where given (`model.remesh`; converted_set passes REMESH's, Phase 10
      Task 3); the rest converted (`model.convert_model`: one object per cell, more past model.MAX_RECORDS; `quad`
      its quad tolerances where given, converted_set passing QUAD's; `road_copy` (margin, band) its road copied across
      cell edges where given (converted_set passing ROAD_COPY's; the road line's centres from `d`); `cell_cut` its cell-grid cut where given,
      converted_set passing CELL_CUT's) and textured
      (`textures.course_textures`, every texture packed in sheet 0, the slot's course block); each record's distance
      coefficient fitted to its converted texture's size (`_fit_coefficients`);
    - one texture header per drawn record (each drawn record steps the header by 1, a move record by 0: a move reads
      the next drawn record's), the header of the record's (texture, mode, translucency, tint): the converted
      texture's with mirror_x / mirror_y where its mesh flips that axis (and does not clamp it), translucent where the
      texture is and the mesh alpha-blends it (an opaque user of a translucent texture, Desert City's 51 and 57, gets
      an opaque copy), its palette entry `textures.tint_entry`'s for the mesh's tint (`_tint`: its face colour where
      its shading modulates the texture; untinted, the texture's own), the wrap bits `textures.course_points` finds
      each variant needs; a clamped axis's (u, v) held to 0..1, so its points stay inside the texture with no wrap
      bit. Every variant shares its texture's texels;
    - an untextured record: a flat header (FLAT_HEADER) in the palette entry nearest its mesh's face colour
      (`textures.flat_entry`, `_face_colours`), its texture points zeros;
    - the texture points `textures.course_points` gives (moves and untextured records zeros);
    - placed at the end of the free runs, clear of the race tables and the HUD outline that `converted_course` puts at
      their start: the objects back to back at the end of POLYGON_FREE, in cell order; the headers (in walk order,
      piece by piece) then the texture points from TEXTURES_FREE's start; the object lists at the end of DATA_FREE
      (the list home), one per occupied cell in cell order, then one empty list (its terminator only) that every
      other cell names. Each list entry: (oba, tpa, tha, budget), budget the object's record count (the convention:
      Beginner's every entry; the frame cap OBC is checked against their sum at run time).

    For a new id (`slot` NEW_ID) the tables go into its runs (NEW_ID_RUNS) instead, each at the first bytes free in
    `rs` that take it (`rs` holds whatever was placed before: the next course's scenery lands after this one's): the
    objects in the custom polygon run, the headers then the points in the custom textures run, the lists in the list
    home DATA_FREE (the draw routines' bound).

    `d` is the course the scenery is drawn on (its road is never thinned). ConvertError for an `rs` that is not a
    RomSet, a `d` that is not a CourseData, an `m` that is not a pol.Model or one `model.convert_model` refuses, a
    blob that is not bytes, a `keep` `scenery_model` refuses, a mesh whose colours are not nine numbers, a scenery that
    does not fit its runs, a `slot` that is not 0 or one of CONVERTED_IDS (a new id: on a set that is not grown);
    DcError from the texture decoder."""
    rs = _romset(rs)
    slot = _slot(slot, rs)
    if not isinstance(d, CourseData):
        raise ConvertError(f"d {type(d).__name__} is not a dc.data.CourseData")
    if not isinstance(m, pol.Model):
        raise ConvertError(f"m {type(m).__name__} is not a pol.Model")
    if not isinstance(tex_blob, (bytes, bytearray)):
        raise ConvertError(f"tex_blob {type(tex_blob).__name__} is not bytes (the course's _TEX.BIN)")
    base_kept, kept = _scenery_models(m, tex_blob, d, keep, remesh, props)
    where, meshes = model.triangle_meshes(kept), _meshes(kept)
    base, indices = textures.course_textures(m, tex_blob, rs)
    copy = None if road_copy is None else (road_centres(tables.road_line(d), "road copy"), *road_copy)
    pieces, _ = _fit_coefficients(model.convert_model(kept, quad, {t: (h.width, h.height) for t, h in base.items()},
                                                      cell_cut, copy, depth),
                                  [(t.width, t.height) for t in m.textures], base)
    tables_ = T.colour_tables(rs)
    face = mesh_colours(m, base_kept, kept)
    flat: dict[tuple, int] = {}                         # mesh colour -> its palette entry
    tinted: dict[tuple, int] = {}                       # (texture, tint) -> its palette entry

    variants: dict[tuple, int] = {}                     # (texture, u mode, v mode, translucent, tint) -> variant id
    headers_v: dict[int, T.TexHeader] = {}
    keys: dict[tuple[int, int], list] = {}              # (cell, piece) -> per record: variant id, ("flat", entry), None
    remapped = {}
    for cell, ps in pieces.items():
        out = []
        for i, pc in enumerate(ps):
            texs, uvs, ks = [], [], []
            for r, t, uv, src in zip(pc.obj.records, pc.textures, pc.uvs, pc.sources):
                if not src:
                    if _step(r.attr):
                        raise ConvertError(f"cell {cell:#x}: a move record steps the header by {_step(r.attr)}, not 0")
                    texs.append(-1), uvs.append(None), ks.append(None)
                    continue
                if _step(r.attr) != 1:
                    raise ConvertError(f"cell {cell:#x}: a drawn record steps the header by {_step(r.attr)}, not 1")
                me = meshes[where[src[0]]]
                if t < 0:
                    rgb = tuple(255 * c for c in face[id(me)])
                    if rgb not in flat:
                        flat[rgb] = textures.flat_entry(rgb, tables_)
                    texs.append(-1), uvs.append(uv), ks.append(("flat", flat[rgb]))
                    continue
                modes = _modes(me)
                translucent = bool(base[t].translucent) and _blends(me)
                tint = _tint(me, face)
                if (t, tint) not in tinted:
                    tinted[(t, tint)] = textures.tint_entry(indices[t], base[t], tint, tables_)
                vid = variants.setdefault((t, *modes, translucent, tint), len(variants))
                headers_v.setdefault(vid, dataclasses.replace(base[t], mirror_x=modes[0] == "mirror",
                                                              mirror_y=modes[1] == "mirror", translucent=translucent,
                                                              colour=tinted[(t, tint)]))
                texs.append(vid), uvs.append(_clamped(uv, modes)), ks.append(vid)
            keys[(cell, i)] = ks
            out.append(pc._replace(textures=tuple(texs), uvs=tuple(uvs)))
        remapped[cell] = tuple(out)
    points_v, headers_v = textures.course_points(remapped, headers_v)

    # the textures run: every piece's headers (one per drawn record), then every piece's texture points
    runs = _runs(slot)
    words = 4 * sum(k is not None for ks in keys.values() for k in ks) + sum(
        2 * sum(map(len, p)) for ps in points_v.values() for p in ps)
    tex_lo, tex_n = runs["textures"]
    word, end = tex_lo // 2, (tex_lo + tex_n) // 2
    if slot:
        word = _first_fit(rs, "textures", runs["textures"], 2 * words, "scenery: texture headers and points") // 2
    tha, tpa, headers, points = {}, {}, {}, {}
    for (cell, i), ks in keys.items():
        tha[(cell, i)] = word
        for k in ks:
            if k is not None:
                headers[word] = headers_v[k] if isinstance(k, int) else T.TexHeader(colour=k[1], **FLAT_HEADER)
                word += 4
    for cell, ps in remapped.items():
        for i in range(len(ps)):
            tpa[(cell, i)] = word
            points[word] = points_v[cell][i]
            word += 2 * sum(map(len, points_v[cell][i]))
    if word > end:
        raise ConvertError(f"scenery: {2 * words} bytes of texture headers and points; the textures run "
                           f"{_run_name('textures', slot)} holds {tex_n}")
    # the polygon run: the objects at its end (a new id: at its first free bytes)
    sized = {key: len(P.encode_object(pc.obj)) for key, pc in
             ((k, pieces[k[0]][k[1]]) for k in keys)}
    poly_lo, poly_n = runs["polygons"]
    if slot:
        word = _first_fit(rs, "polygons", runs["polygons"], 4 * sum(sized.values()), "scenery: objects") // 4
    else:
        word = (poly_lo + poly_n) // 4 - sum(sized.values())
    if 4 * word < poly_lo:
        raise ConvertError(f"scenery: {4 * sum(sized.values())} bytes of objects; the polygon run "
                           f"{_run_name('polygons', slot)} holds {poly_n}")
    objects, oba = {}, {}
    for (cell, i), n in sized.items():
        objects[word], oba[(cell, i)] = pieces[cell][i].obj, 0x800000 | word
        word += n
    # the list home: one list per occupied cell, then the empty list, at DATA_FREE's end (a new id: its first free)
    empty = len(pieces) < C.GRID_CELLS
    need = sum(C.OBJECT_ENTRY_SIZE * len(ps) + 4 for ps in pieces.values()) + 4 * empty
    at = DATA_BASE + (_first_fit(rs, "data", tgp.DATA_FREE, need, "scenery: object lists") if slot
                      else tgp.DATA_FREE[0] + tgp.DATA_FREE[1] - need)
    if at < DATA_BASE + tgp.DATA_FREE[0]:
        raise ConvertError(f"scenery: {need} bytes of object lists; the list home DATA_FREE holds {tgp.DATA_FREE[1]}")
    lists, cell_list = {}, {}
    for cell, ps in pieces.items():
        lists[at] = object_list_at(at, [(oba[(cell, i)], tpa[(cell, i)], tha[(cell, i)], len(pc.obj.records))
                                        for i, pc in enumerate(ps)])
        cell_list[cell] = at
        at += C.OBJECT_ENTRY_SIZE * len(ps) + 4
    if empty:
        lists[at] = ()
    cells = tuple(cell_list.get(k, at) for k in range(C.GRID_CELLS))
    used = sorted({t for t, *_ in variants})
    texels = textures.course_texels({t: base[t] for t in used}, {t: indices[t] for t in used})
    return Scenery(objects, lists, cells, headers, points, texels, lift=_lift_of(m), shift=_shift_of(m))


# ---- validation ----

def _course_n(n) -> int:
    if not is_int(n) or not 0 <= n < COURSE_COUNT:
        raise ConvertError(f"course {n!r} is not a Dreamcast course 0..{COURSE_COUNT - 1}")
    return int(n)


def _slot(slot, rs: RomSet | None = None) -> int:
    """`slot` as an int: 0 (Beginner's), one of CONVERTED_IDS or one of REVERSE_IDS (a new id: with `rs`, on a grown
    set with the new ids' guards: course.grow_ids, course.guard_new_ids). ConvertError otherwise."""
    if not is_int(slot) or slot not in SLOTS:
        raise ConvertError(f"slot {slot!r}: a conversion fills Beginner's slot (0) or a new id "
                           f"{', '.join(map(str, CONVERTED_IDS + REVERSE_IDS))}")
    slot = int(slot)
    try:
        if slot and rs is not None and not (C.grown(rs) and C.guarded(rs)):
            raise ConvertError(f"slot {slot}: the set has no course id {slot} of its own (course.grow_ids, then "
                               f"course.guard_new_ids)")
    except DecodeError as e:
        raise ConvertError(f"slot {slot}: {e}") from e
    return slot


def _variant(variant, n: int, slot: int, scenery: str = "keep") -> str:
    """`variant` checked against course `n`, `slot` and `scenery`: "" fills slot 0 or a CONVERTED_IDS id, never a
    reverse id; "R" fills exactly the reverse id n + 8, for a converted course n 3-7, or for the arcade's three
    (n 0-2) with scenery "keep" only (their own scenery shared: `_reverse_alias`). ConvertError otherwise."""
    if not isinstance(variant, str) or variant not in VARIANTS:
        raise ConvertError(f"variant {variant!r:.30} is not one of {VARIANTS}")
    if variant == "R":
        if n < 3 and scenery != "keep":
            raise ConvertError(f"course {n} in reverse keeps the arcade's scenery (scenery='keep'), not "
                               f"{scenery!r:.30}")
        if slot != n + 8:
            raise ConvertError(f"slot {slot}: course {n} in reverse fills its reverse id {n + 8}")
    elif slot in REVERSE_IDS:
        raise ConvertError(f"slot {slot}: a reverse id takes a course's R files (variant='R')")
    return variant


def _reverse_alias(rs: RomSet, o: int, k: int) -> dict[tuple[str, int], bytes]:
    """The patches that make id `k` (a reverse id 8-10) share the arcade course `o`'s record, scenery and per-id
    arrays by pointer, zero bytes (mirror._alias: every per-id array entry but COURSE_POINTERS and the unshared
    SRAM and select words, TGP slot k, the tune), the culling walk `o`'s own (VISIT_IDS[k] = o, the widened default
    k & 7: the same cells, unlike a mirror's). The race tables come after (converted_set)."""
    from dtools.arcade import mirror
    out = mirror._alias(rs, o, k)
    out[C.rom_word_at(C.VISIT_IDS + k)] = bytes([o])
    return out


def _reverse_tables(rs: RomSet, n: int, k: int, offset: int, L: int, tunnel: bool) -> RomSet:
    """`rs` with reverse id `k`'s own copies of arcade course `n`'s segment-keyed tables, on the reverse road
    (`reverse`; `offset` its point_offset, L its road points): the progress masks when `n` has records (Advanced),
    the tunnel light record's tables when the reverse road has the tunnel bit (`tunnel`: Expert's segment flags,
    Advanced's step flags); each written into the new ids' data run (ID_DATA_FREE) and named by id k's own entry.
    Without them id k keeps id n's entry (`_reverse_alias`). ConvertError as reverse's and for no room."""
    from dtools.arcade import mirror
    arrays = C.id_tables(rs)
    patches: dict[tuple[str, int], bytes] = {}
    blobs: list[tuple[int, bytes]] = []

    def place(blob: bytes, what: str) -> int:
        off = _first_fit(rs, "data", ID_DATA_FREE, len(blob), f"course {k}: {what}",
                         busy=[("data", o, len(b)) for o, b in blobs])
        blobs.append((off, blob))
        return DATA_BASE + off

    masks = mirror.mask_records(rs, struct.unpack("<I", rs.read_i960(C.id_entry(arrays, "PROGRESS_MASKS", n), 4))[0])
    if masks:
        at = place(reverse.reversed_masks(masks, offset, L), "progress masks")
        patches[C.rom_word_at(C.id_entry(arrays, "PROGRESS_MASKS", k))] = struct.pack("<I", at)
    if tunnel:
        count = rs.read_i960(C.id_entry(arrays, "TUNNEL_BYTES", n), 1)[0]
        starts, ends, rng, mask, positions = struct.unpack("<5I", rs.read_i960(C.id_entry(arrays, "TUNNEL_LIGHTS", n),
                                                                               20))
        bits = struct.unpack("<I", rs.read_i960(mask, 4))[0]
        tables_ = reverse.reversed_tunnel(
            struct.unpack(f"<{count + 1}H", rs.read_i960(starts, 2 * count + 2)),
            struct.unpack(f"<{count + 1}H", rs.read_i960(ends, 2 * count + 2)),
            struct.unpack("<2H", rs.read_i960(rng, 4)), bits,
            {z: rs.read_i960(positions + 12 * z, 12) for z in range(count + 1) if bits >> z & 1}, offset, L)
        names = ("tunnel zone starts", "tunnel zone ends", "tunnel range", "tunnel zone mask", "tunnel light positions")
        words = [place(b, w) if count or w in ("tunnel range", "tunnel zone mask") else a
                 for b, w, a in zip(tables_, names, (starts, ends, rng, mask, positions))]
        patches[C.rom_word_at(C.id_entry(arrays, "TUNNEL_LIGHTS", k))] = struct.pack("<5I", *words)
    for off, blob in blobs:
        rs = _write_free(rs, "data", off, blob)
    return rs.with_patches(patches) if patches else rs


def _romset(rs) -> RomSet:
    if not isinstance(rs, RomSet):
        raise ConvertError(f"{type(rs).__name__} is not a RomSet")
    return rs


def _owned(rs: RomSet) -> dict[str, np.ndarray]:
    """golden.owned_spans(rs) merged per region into ascending [start, end) rows: golden.owned, the per-set cache the
    writers share (tgp.write_free, mirror._char_run). ConvertError when a course of the set does not decode."""
    try:
        return golden.owned(rs)
    except DecodeError as e:
        raise ConvertError(f"the set's owned space (golden.owned_spans): {e}") from e


def _usable(rs: RomSet, region: str, busy=(), run: tuple[int, int] | None = None) -> np.ndarray:
    """Per byte of the region's free run (`run`, default _RUNS[region]): True where it is free (0xff; copro_data's
    stock run: in both 2 MB copies), not in one of the `busy` (region, offset, length) spans (the model scenery's
    tables, placed by the caller) and owned by nothing (golden.owned_spans: romset.RESERVED, a new id's course sheet
    block, every course's tables; a byte they own is busy whatever its value, 0xff included)."""
    lo, size = run or _RUNS[region]
    image = getattr(rs, region)
    if len(image) < lo + size:
        raise ConvertError(f"the {region} run {lo:#x}+{size:#x} is past the set's image ({len(image):#x} bytes): a "
                           f"new id's runs are a custom set's")
    out = np.frombuffer(image, np.uint8, size, lo) == _FREE
    if region == "copro_data" and len(image) >= COPRO_SIZE and lo + size <= COPRO_SIZE // 2:
        out &= np.frombuffer(image, np.uint8, size, lo + COPRO_SIZE // 2) == _FREE
    rows = _owned(rs).get(region)
    if rows is not None:
        for a, z in rows[(rows[:, 1] > lo) & (rows[:, 0] < lo + size)]:
            out[max(int(a) - lo, 0):int(z) - lo] = False
    for r, off, n in busy:
        if r == region:
            out[max(off - lo, 0):max(off + n - lo, 0)] = False
    return out


def _first_fit(rs: RomSet, region: str, run: tuple[int, int], need: int, what: str, busy=(),
               align: int = 4) -> int:
    """The first `align`-aligned offset in `run` of the region from which `need` bytes are free (`_usable`).
    ConvertError when none is."""
    lo, size = run
    usable = _usable(rs, region, busy, run)
    fits = _fits(usable, need)
    fits = fits[(lo + fits) % align == 0]
    if not len(fits):
        raise ConvertError(f"{what}: no {need} free bytes in the {region} run {lo:#x}-{lo + size:#x}")
    return lo + int(fits[0])


def last_fit(rs: RomSet, region: str, run: tuple[int, int], need: int, what: str, busy=(), align: int = 4) -> int:
    """The last `align`-aligned offset in `run` of the region from which `need` bytes are free (`_usable`): the
    run's last bytes, which `_first_fit` reaches last (select.patch_select's map objects and lists, so that the
    converter's and the mirrors' first fits place the same before the select or after it). ConvertError when none
    is."""
    lo, size = run
    fits = _fits(_usable(rs, region, busy, run), need)
    fits = fits[(lo + fits) % align == 0]
    if not len(fits):
        raise ConvertError(f"{what}: no {need} free bytes in the {region} run {lo:#x}-{lo + size:#x}")
    return lo + int(fits[-1])


def _fits(usable: np.ndarray, need: int) -> np.ndarray:
    """The 4-aligned offsets into the run from which `need` bytes are all free, ascending."""
    bad = np.concatenate(([0], np.cumsum(~usable)))
    starts = np.arange(0, len(usable) - need + 1, 4)
    return starts[bad[starts + need] == bad[starts]] if need <= len(usable) else starts[:0]


def _check_span(rs: RomSet, region: str, at, need: int, field: str, busy=(), run: tuple[int, int] | None = None,
                align: int = 4) -> int:
    """`at` as an int: `align`-aligned, `need` bytes from it inside the region's run (`run`, default _RUNS[region])
    and free. ConvertError otherwise."""
    lo, size = run or _RUNS[region]
    if not is_int(at):
        raise ConvertError(f"layout: {field} {at!r} is not a byte offset (int)")
    at = int(at)
    if at % align:
        raise ConvertError(f"layout: {field} {at:#x} is unaligned ({align}-byte aligned)")
    if not lo <= at or at + need > lo + size:
        raise ConvertError(f"layout: {field} {at:#x}: the {need} bytes from it are not inside the {region} free run "
                           f"{lo:#x}-{lo + size:#x}")
    span = _usable(rs, region, busy, run)[at - lo:at - lo + need]
    if not span.all():
        k = at + int(np.argmin(span))
        raise ConvertError(f"layout: {field} {at:#x}: {region} {k:#x} is not free (not 0xff, the scenery's, or "
                           f"owned: golden.owned_spans): the {need} bytes from {at:#x} must be")
    return at


def _layout(rs: RomSet, layout, need: dict[str, int], g: tgp.Ground, busy=(), slot: int = 0) -> Layout:
    """`layout` checked (each span inside its run and free, the ground window-safe), or by default each run's first
    4-aligned bytes that are free for what it takes (copro_data: also window-safe for `g`). Bytes in `busy` spans
    (the model scenery's tables) are not free. The runs are `_runs(slot)`'s; a new id with its own scenery also places
    its course sheet block (need["sheet"]: 1 MB-aligned in ID_DATA_FREE, first) and its HUD outline's list
    (need["list"]: in DATA_FREE), each kept out of the other tables' way; without (need 0) both are None."""
    n, block = len(g.polys), tgp.list_block_words(g.lists)
    runs = _runs(slot)

    def safe(at: int) -> bool:
        return tgp.window_safe(g.lists, n, at // 4, block)

    if layout is not None and not isinstance(layout, Layout):
        raise ConvertError(f"layout {layout!r:.60} is not a build.Layout")
    if not slot and layout is not None and (layout.list_at, layout.sheet_at) != (None, None):
        raise ConvertError(f"layout: list_at {layout.list_at!r}, sheet_at {layout.sheet_at!r}: slot 0 has neither "
                           f"(its list and sheet block are Beginner's, rewritten in place)")
    busy = list(busy)
    extra = {}
    for field, region, run, size, align in (("sheet_at", "data", runs["data"], need.get("sheet", 0), T.SHEET_BLOCK),
                                            ("list_at", "data", tgp.DATA_FREE, need.get("list", 0), 4)):
        given = None if layout is None else getattr(layout, field)
        if not size:
            if given is not None:
                raise ConvertError(f"layout: {field} {given!r}: the conversion places no such table")
            extra[field] = None
            continue
        at = _first_fit(rs, region, run, size, f"layout: {field}", busy, align) if given is None else \
            _check_span(rs, region, given, size, field, busy, run, align)
        extra[field] = at
        busy.append((region, at, size))
    if layout is not None and not (is_int(layout.program_at) and layout.program_at == EMPTY_AT):
        raise ConvertError(f"layout: program_at {layout.program_at!r:.40} is not EMPTY_AT {EMPTY_AT:#x}: a converted "
                           f"course has no actors, its actor table is the shared EMPTY_ACTORS")
    if layout is None:
        out = []
        for region in _REGIONS:
            if region == "program":                     # the shared table: nothing to place
                out.append(EMPTY_AT)
                continue
            lo, size = runs[region]
            fits = _fits(_usable(rs, region, busy, runs[region]), need[region])
            at = next((lo + int(k) for k in fits if region != "copro_data" or safe(lo + int(k))), None)
            if at is None:
                raise ConvertError(f"layout: no {need[region]} free bytes in the {region} free run {lo:#x}-"
                                   f"{lo + size:#x}" + (" with no list or polygon across a bank window"
                                                        if region == "copro_data" else ""))
            out.append(at)
        return Layout(*out, **extra)
    out = Layout(*(_check_span(rs, region, at, need[region], field, busy, runs[region]) if region != "program"
                   else EMPTY_AT for region, at, field in zip(_REGIONS, layout, Layout._fields)), **extra)
    if not safe(out.copro_at):
        raise ConvertError(f"layout: copro_at {out.copro_at:#x}: a list or polygon of the ground would cross a "
                           f"64K-word bank window (the TGP wraps to the window's start)")
    return out


# ---- the HUD map ----

def _f32s(values, what: str) -> tuple[float, ...]:
    """`values` rounded to single precision (as the ROM holds them): ConvertError for one that is not a finite number
    or does not fit."""
    blob = _f32(values, len(values), "values", what, ConvertError)
    return struct.unpack(f"<{len(values)}f", blob)


def road_centres(road, what: str) -> list[tuple[float, float]]:
    """(x, z) of each road point's centre (course.RoadPoint.centre: the midpoint of the outer used lanes), every field
    it reads validated. ConvertError for a road that is not a sequence of at least three course.RoadPoint, used lanes
    outside 0..LANES-1, lanes that are not LANES course.Lane, or a used lane's position that is not three finite
    numbers."""
    if not isinstance(road, (tuple, list)):
        raise ConvertError(f"{what}: {road!r:.60} is not a sequence of course.RoadPoint")
    if len(road) < 3:
        raise ConvertError(f"{what}: {len(road)} road points; the road line is a closed loop of at least 3")
    out = []
    for k, p in enumerate(road):
        if not isinstance(p, C.RoadPoint):
            raise ConvertError(f"{what}: road point {k} {p!r:.60} is not a course.RoadPoint")
        lanes = p.lanes
        if not isinstance(lanes, (tuple, list)) or len(lanes) != C.LANES or not all(isinstance(v, C.Lane)
                                                                                   for v in lanes):
            raise ConvertError(f"{what}: road point {k}: lanes {lanes!r:.60} are not {C.LANES} course.Lane")
        if not is_int(p.lane_lo) or not is_int(p.lane_hi) or not 0 <= p.lane_lo <= p.lane_hi < C.LANES:
            raise ConvertError(f"{what}: road point {k}: used lanes {p.lane_lo!r}..{p.lane_hi!r} are not inside "
                               f"0..{C.LANES - 1}")
        a, b = lanes[p.lane_lo].position, lanes[p.lane_hi].position
        for pos in (a, b):
            if not isinstance(pos, (tuple, list)) or len(pos) != 3 or not all(is_number(v) for v in pos):
                raise ConvertError(f"{what}: road point {k}: lane position {pos!r:.60} is not three finite numbers")
        x, z = (float(a[0]) + float(b[0])) / 2, (float(a[2]) + float(b[2])) / 2
        if not math.isfinite(x) or not math.isfinite(z):
            raise ConvertError(f"{what}: road point {k}: its centre ({x}, {z}) is not finite")
        out.append((x, z))
    return out


def _chord_distance(p, a, b) -> float:
    """The distance from point `p` to the segment a-b (all (x, z))."""
    ex, ez = b[0] - a[0], b[1] - a[1]
    length2 = ex * ex + ez * ez
    t = 0.0 if not length2 else max(0.0, min(1.0, ((p[0] - a[0]) * ex + (p[1] - a[1]) * ez) / length2))
    return math.hypot(p[0] - a[0] - t * ex, p[1] - a[1] - t * ez)


def outline_points(cs: list[tuple[float, float]], quads: int = OUTLINE_QUADS) -> tuple[list[int], float]:
    """(kept, deviation): the indices of the road points (x, z) of the closed loop `cs` an outline of at most `quads`
    quads keeps, in order from 0, and the largest distance of a skipped point from the chord between the kept points
    round it. All of them when there are at most `quads`; otherwise point 0 and the point farthest from it, then, one
    at a time, the skipped point farthest from its chord (ties: the longer span, then nearer its middle), until
    `quads` are kept: the loop merged where it is straightest. ConvertError for fewer than 3 points, a point that
    is not two finite numbers, or a `quads` that is not an int of at least 3 (bool refused)."""
    if not isinstance(cs, (tuple, list)) or len(cs) < 3:
        raise ConvertError(f"HUD outline: {cs!r:.60} is not a loop of at least 3 points")
    for k, p in enumerate(cs):
        if not isinstance(p, (tuple, list)) or len(p) != 2 or not all(is_number(v) for v in p):
            raise ConvertError(f"HUD outline: point {k} {p!r:.60} is not two finite numbers (x, z)")
    n = len(cs)
    if not is_int(quads) or quads < 3:
        raise ConvertError(f"HUD outline: quads {quads!r} is not an int of at least 3")
    if n <= quads:
        return list(range(n)), 0.0
    kept = [0, max(range(n), key=lambda k: (math.dist(cs[0], cs[k]), -k))]
    while len(kept) < quads:
        best = None
        for i, a in enumerate(kept):
            b = kept[(i + 1) % len(kept)]
            span = (b - a) % n
            for j in range(1, span):
                key = (_chord_distance(cs[(a + j) % n], cs[a], cs[b]), span, -abs(2 * j - span))
                if best is None or key > best[0]:
                    best = (key, (a + j) % n)
        kept = sorted(kept + [best[1]])
    worst = max(_chord_distance(cs[(a + j) % n], cs[a], cs[b]) for i, a in enumerate(kept)
                for b in (kept[(i + 1) % len(kept)],) for j in range(1, (b - a) % n))
    return kept, worst


def hud_outline(road) -> P.Object:
    """The HUD map's outline of a road line (a sequence of course.RoadPoint, a closed loop): flat polygons at y 0
    along the road line's centre (RoadPoint.centre), OUTLINE_WIDTH wide (Beginner's strokes). A strip of one edge
    per kept road point (`outline_points`: every point of a road of at most OUTLINE_QUADS, else OUTLINE_QUADS of
    them, the loop merged where it is straightest), square to the centre line there (the direction from the road
    point before to the one after), its two ends OUTLINE_WIDTH / 2 either side of the centre: the seeds are point
    0's edge, then one quad record per kept point after it and a last one back to point 0's edge (as many quads as
    kept points, every one drawn). Each record: OUTLINE_ATTR, the normal OUTLINE_NORMAL, the edge's two points; the
    end record OUTLINE_END; points rounded to single precision (the object decodes back equal in geometry mode 1, as
    the HUD map is read). Every skipped road centre lies within OUTLINE_WIDTH / 2 of its chord, inside the stroke.

    ConvertError for a road `road_centres` refuses (empty, fewer than three points, a point that is not a RoadPoint,
    used lanes out of range, a position that is not finite), two neighbours of a point at the same place (no
    direction), an edge that does not fit single precision, or a road whose merged outline leaves a road centre
    outside its stroke."""
    cs = road_centres(road, "HUD outline")
    n, half = len(cs), OUTLINE_WIDTH / 2
    kept, deviation = outline_points(cs)
    if deviation > half:
        raise ConvertError(f"HUD outline: merged to {len(kept)} quads, a road centre lies {deviation:.2f} from the "
                           f"outline's centre line, outside its stroke ({half})")
    edges = []
    for k in kept:
        x, z = cs[k]
        (x0, z0), (x1, z1) = cs[k - 1], cs[(k + 1) % n]
        tx, tz = x1 - x0, z1 - z0
        length = math.hypot(tx, tz)
        if not length or not math.isfinite(length):
            raise ConvertError(f"HUD outline: road point {k}: its neighbours {k - 1 if k else n - 1} and "
                               f"{(k + 1) % n} are at one place, so the road has no direction there")
        nx, nz = -tz / length * half, tx / length * half
        what = f"HUD outline: road point {k}"
        edges.append((_f32s((x + nx, 0.0, z + nz), what), _f32s((x - nx, 0.0, z - nz), what)))
    records = [P.Record(OUTLINE_ATTR, OUTLINE_NORMAL, p0, p1) for p0, p1 in edges[1:] + edges[:1]]
    return P.Object(edges[0][0], edges[0][1], records, 6 + 10 * len(kept) + len(OUTLINE_END), OUTLINE_END)


def _turn(angle: int) -> tuple[float, float]:
    """(cos, sin) of a TGP angle (0x10000 a turn), exact on the quarter turns."""
    exact = {0: (1.0, 0.0), 1: (0.0, 1.0), 2: (-1.0, 0.0), 3: (0.0, -1.0)}
    if angle % _QUARTER == 0:
        return exact[angle // _QUARTER % 4]
    a = 2 * math.pi * angle / 0x10000
    return math.cos(a), math.sin(a)


def _box(cs: list[tuple[float, float]], angle: int) -> tuple[float, float, float, float]:
    """(width, height, centre x, centre y): the view-plane box of the points turned by `angle` at zoom 1. The HUD's
    matrix gives view x = zoom (x cos a - z sin a) + tx, view y = zoom (x sin a + z cos a) + ty: at Beginner's angle
    0x4000 the traced view x = -zoom z + tx, view y = zoom x + ty (docs/spike/2026-11-mirror.md, "HUD map")."""
    c, s = _turn(angle)
    xs = [x * c - z * s for x, z in cs]
    ys = [x * s + z * c for x, z in cs]
    return max(xs) - min(xs), max(ys) - min(ys), (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2


def hud_frame(ref: C.HudMap, ref_road, road) -> tuple[tuple[float, float, float], int, tuple[float, ...]]:
    """(position, angle, zooms) of the HUD map's place record and zooms for `road`, framing it as `ref` (the slot's
    HudMap) frames `ref_road` (the slot's road line; Beginner's on the real set). The rule, measured on Beginner (it
    gives Beginner's own record for Beginner's road; docs/spike/2026-11-convert.md, "HUD map"):

    - angle: the road's longer axis (x or z, of its centre's extent) turned as the reference's: `ref.angle` when it
      is the same world axis as the reference's, a quarter turn less otherwise (Beginner, long in x: 0x4000, x
      vertical on screen; a road long in z: 0, as Advanced's);
    - zooms: the reference's, every level times one factor: the largest at which the road's centre, turned, fits
      the box the reference's centre fills (its width and height in the view plane), so one of the two is equal;
    - position: x and y such that at zoom level HUD_LEVEL (the race's) the box's centre is the reference's; z (the
      depth) the reference's. Single precision, as stored.

    ConvertError for a `ref` that is not a course.HudMap, a position or zooms that are not finite single-precision
    numbers, an angle that is not an s16, a level-HUD_LEVEL zoom that is not positive; a road or reference road
    `road_centres` refuses; a road whose centre is one point (nothing to frame) or so large that its zoom is not a
    normal single-precision number."""
    if not isinstance(ref, C.HudMap):
        raise ConvertError(f"HUD map {ref!r:.60} is not a course.HudMap")
    if not isinstance(ref.position, (tuple, list)) or len(ref.position) != 3:
        raise ConvertError(f"HUD map: position {ref.position!r:.60} is not three numbers")
    if not isinstance(ref.zooms, (tuple, list)) or len(ref.zooms) != C.HUD_MAP_ZOOM_LEVELS:
        raise ConvertError(f"HUD map: zooms {ref.zooms!r:.60} are not {C.HUD_MAP_ZOOM_LEVELS} numbers")
    position, zooms = _f32s(ref.position, "HUD map position"), _f32s(ref.zooms, "HUD map zooms")
    if not is_int(ref.angle) or not -0x8000 <= ref.angle < 0x8000:
        raise ConvertError(f"HUD map: angle {ref.angle!r} is not an s16")
    if not zooms[HUD_LEVEL] > 0:
        raise ConvertError(f"HUD map: the zoom of level {HUD_LEVEL} {zooms[HUD_LEVEL]} is not positive")
    rc, nc = road_centres(ref_road, "HUD map: the reference road"), road_centres(road, "HUD map: the road")

    def long_x(cs):
        return max(x for x, _ in cs) - min(x for x, _ in cs) >= max(z for _, z in cs) - min(z for _, z in cs)

    angle = int(ref.angle)
    if long_x(rc) != long_x(nc):
        angle = (angle - _QUARTER + 0x8000) % 0x10000 - 0x8000
    rw, rh, rx, ry = _box(rc, int(ref.angle))
    w, h, x, y = _box(nc, angle)
    fits = [r / v for r, v in ((rw, w), (rh, h)) if v > 0]
    if not fits or not all(math.isfinite(k) and k > 0 for k in fits):
        raise ConvertError(f"HUD map: the road's centre spans {w} x {h} in the view plane, the reference's {rw} x "
                           f"{rh}: nothing to frame")
    k = min(fits)
    new = _f32s(tuple(z * k for z in zooms), "HUD map zooms")
    if not new[HUD_LEVEL] >= _F32_NORMAL:
        raise ConvertError(f"HUD map: the road's centre spans {w} x {h} in the view plane: its zoom "
                           f"{zooms[HUD_LEVEL] * k} is below single precision's normal range")
    sr, sn = zooms[HUD_LEVEL], new[HUD_LEVEL]
    dx, dy = sr * rx - sn * x, sr * ry - sn * y         # 0 for the reference's own road: its position exactly
    return _f32s((position[0] + dx, position[1] + dy, position[2]), "HUD map position"), angle, new


# ---- the course ----

def _empty_scenery(base: golden.Course, road, outline: P.Object, lay: Layout) -> dict:
    """The SCENERY_FIELDS of the empty scenery (converted_course) on the slot's decoded Course `base`."""
    name = f"{base.entry.name}: empty scenery"
    reached = {o for slots in base.spawn for s in slots for o in s.objects} | {base.map_object.objects}
    if EMPTY_LIST not in base.cells or EMPTY_LIST in reached:
        raise ConvertError(f"{name}: EMPTY_LIST {EMPTY_LIST:#x} is not a cell list of the slot's only (its grid "
                           f"must name it, no spawn record nor map object): it is rewritten to its terminator")
    hud = base.hud_map
    if len(hud.entries) != 1:
        raise ConvertError(f"{name}: the slot's HUD map list holds {len(hud.entries)} entries, not one")
    index = lay.polygons_at // 4
    position, angle, zooms = hud_frame(hud, base.road, road)
    entry = dataclasses.replace(hud.entries[0], oba=0x800000 | index, raw=(len(outline.records),))
    lst = hud.outline_list
    if lay.list_at is not None:                         # a new id: its own list, its one entry there
        lst, entry = DATA_BASE + lay.list_at, dataclasses.replace(entry, entry=lay.list_at)
    return dict(cells=(EMPTY_LIST,) * C.GRID_CELLS,
                object_lists={EMPTY_LIST: (), **{k: base.object_lists[k] for k in reached}},
                actor_table=EMPTY_ACTORS, actor_slots=(0,) * C.GRID_CELLS,
                actor_lists={}, billboards={}, billboard_ref=None, switch_lists=(), unmodelled=(),
                hud_map=C.HudMap(base.course, lst, (entry,), {index: outline}, position, angle, hud.pad, zooms))


def _slot_lists(base: golden.Course) -> set:
    """The slot's object lists a converted set still reaches: its spawn records' (the car models) and its map
    object's (the select screen's)."""
    return {o for slots in base.spawn for s in slots for o in s.objects} | {base.map_object.objects}


def _u32ok(v) -> bool:
    return is_int(v) and 0 <= v <= 0xFFFFFFFF


def _step(attr: int) -> int:
    """The texture-header step of a record's attribute: bits 12-16, signed (texture.tex_header)."""
    step = (attr >> 12) & 0x1F
    return step - 32 if step & 0x10 else step


def _header_walk(placements) -> dict[int, tuple[int, int]]:
    """textures word index -> (tha as sent, step) of every header the polygons of `placements` ((Placement, objects
    by word index) pairs, in order) read, in the order golden.decode_course meets them (texture.object_headers: the
    header at tha for every record, then step * 4 words on); the first meeting wins, as decode_course keeps it."""
    out: dict[int, tuple[int, int]] = {}
    for p, objects in placements:
        tha = p.tha
        for r in objects[C.rom_object_index(p.oba)].records:
            out.setdefault(tha & T.TEX_ADDR_MASK, (tha, _step(r.attr)))
            tha = (tha + 4 * _step(r.attr)) & 0xFFFFFFFF
    return out


def _point_words(obj: P.Object) -> int:
    """The texture point words an object's placement reads: 2 per vertex, 3 or 4 vertices per record."""
    return 2 * sum(4 if r.is_quad else 3 for r in obj.records)


def _texel_key(h: T.TexHeader) -> tuple[int, int, int, int, int]:
    return h.sheet, h.tex_x, h.tex_y, h.width, h.height


def _drawn(cells, lists: dict, base: golden.Course, hud: C.HudMap, objects: dict):
    """(Placement, objects) pairs in golden.decode_course's order: the cells' lists (each once, in cell order), the
    slot's spawn lists (the car models), then the HUD map's entries."""
    order = list(dict.fromkeys(cells)) + [o for slots in base.spawn for s in slots for o in s.objects]
    return ([(p, objects) for k in dict.fromkeys(order) for p in lists[k]]
            + [(p, hud.objects) for p in hud.entries])


def _point_ok(p, none: bool = False) -> bool:
    """A tuple of three plain numbers (int or float, not bool, not numpy), or None where `none` allows it (a
    triangle's second point, a modes 2-3 normal)."""
    if p is None:
        return none
    return type(p) is tuple and len(p) == 3 and all(type(v) in (int, float) for v in p)


def _objects_ok(objects) -> bool:
    """A dict of polygon-ROM word index -> polyrom.Object whose seeds are points (`_point_ok`) and whose records are
    polyrom.Record with u32 attributes and points (p1 and the normal may be None)."""
    return isinstance(objects, dict) and all(
        is_int(k) and isinstance(o, P.Object) and _point_ok(o.seed0) and _point_ok(o.seed1)
        and isinstance(o.records, (list, tuple))
        and all(isinstance(r, P.Record) and _u32ok(r.attr) and _point_ok(r.p0) and _point_ok(r.p1, True)
                and _point_ok(r.normal, True) for r in o.records) for k, o in objects.items())


def _points_ok(points) -> bool:
    """A dict of textures word index (an int) -> texture points as golden.Course.points holds them: a tuple per
    record of 3 or 4 (v, u) tuples of plain ints 0-0xffff (the model scenery's texture points)."""
    return isinstance(points, dict) and all(
        type(k) is int and type(v) is tuple
        and all(type(q) is tuple and len(q) in (3, 4)
                and all(type(c) is tuple and len(c) == 2 and all(type(w) is int and 0 <= w <= 0xFFFF for w in c)
                        for c in q) for q in v)
        for k, v in points.items())


def _overlap(spans) -> tuple | None:
    """The first two of `spans` ((offset, length, name)) that share a byte, or None."""
    spans = sorted(spans)
    return next(((x, y) for x, y in zip(spans, spans[1:]) if x[0] + x[1] > y[0]), None)


def _lists_ok(lists) -> bool:
    """A dict of list address -> tuple of course.Placement whose oba, tpa, tha are u32 and raw one u32."""
    return isinstance(lists, dict) and all(
        is_int(k) and isinstance(v, tuple) and all(
            isinstance(p, C.Placement) and all(_u32ok(w) for w in (p.oba, p.tpa, p.tha))
            and isinstance(p.raw, tuple) and len(p.raw) == 1 and _u32ok(p.raw[0]) for p in v)
        for k, v in lists.items())


def _header_ok(h) -> bool:
    """A texture.TexHeader whose fields are ints and bools and whose raw is a tuple of ints (texture's encoders and
    the walk read them; their ranges are encode_tex_header's)."""
    return (isinstance(h, T.TexHeader) and all(type(getattr(h, f.name)) in (int, bool)
                                               for f in dataclasses.fields(T.TexHeader) if f.name != "raw")
            and type(h.raw) is tuple and all(type(x) is int for x in h.raw))


def _headers_ok(headers) -> bool:
    """A dict of textures word index -> texture.TexHeader (`_header_ok`)."""
    return isinstance(headers, dict) and all(is_int(k) and _header_ok(h) for k, h in headers.items())


def _texels_ok(texels) -> bool:
    return isinstance(texels, tuple) and all(isinstance(t, golden.Texels) and _header_ok(t.header) for t in texels)


def _scenery_data(sd, name: str) -> list[tuple[str, int, int]]:
    """`sd` checked to be a Scenery whose fields have the types `converted_course` reads (ConvertError otherwise);
    its tables' (region, offset, length) spans, kept out of the default layout. Where they go is `_check_model`'s."""
    def fail(field: str, detail: str):
        raise ConvertError(f"{name}: scenery {field}: {detail}")

    if not isinstance(sd, Scenery):
        raise ConvertError(f"{name}: scenery_data {sd!r:.60} is not a build.Scenery (scenery=\"model\" draws one)")
    if not _objects_ok(sd.objects):
        fail("objects", f"{sd.objects!r:.60} are not polyrom.Object (of polyrom.Record) by word index")
    if not _lists_ok(sd.lists):
        fail("lists", f"{sd.lists!r:.60} are not tuples of course.Placement (u32 words) by list address")
    if not isinstance(sd.cells, (tuple, list)) or len(sd.cells) != C.GRID_CELLS or not all(is_int(k)
                                                                                          for k in sd.cells):
        fail("cells", f"{sd.cells!r:.60} are not {C.GRID_CELLS} list addresses")
    if bad := [k for k in sd.cells if k not in sd.lists]:
        fail("cells", f"cell lists {', '.join(f'{k:#x}' for k in bad[:4])} are not in the scenery's lists")
    if not _headers_ok(sd.headers):
        fail("headers", f"{sd.headers!r:.60} are not texture.TexHeader by textures word index")
    if not _points_ok(sd.points):
        fail("points", f"{sd.points!r:.60} are not texture points (per record 3 or 4 (v, u) tuples of u16 ints) by "
             f"textures word index")
    if not _texels_ok(sd.texels):
        fail("texels", f"{sd.texels!r:.60} are not a tuple of golden.Texels")
    for k, v in sd.lists.items():
        for p in v:
            try:
                index = C.rom_object_index(p.oba)
            except DecodeError as e:
                fail("lists", f"the list at {k:#x}: {e}")
            if index not in sd.objects:
                fail("lists", f"the list at {k:#x} names object {p.oba:#x}, not one of the scenery's objects")
    spans = [("data", k - DATA_BASE, C.OBJECT_ENTRY_SIZE * len(v) + 4) for k, v in sd.lists.items()]
    try:
        spans += [("polygons", 4 * k, 4 * len(P.encode_object(o))) for k, o in sd.objects.items()]
    except DecodeError as e:
        fail("objects", str(e))
    return spans


def _model_scenery(rs: RomSet, base: golden.Course, empty: dict, sd: Scenery, name: str) -> dict:
    """The MODEL_FIELDS of `scenery="model"` on the slot's decoded Course `base` (of `rs`): the empty scenery's
    actor table and HUD map (`empty`, `_empty_scenery`'s), the scenery's cells and lists with the slot's spawn and
    map-object lists, the scenery's objects with the car models, and the headers, texture points and texels the set then
    reaches, in golden.decode_course's order (`_drawn`, `_header_walk`): each header's words the scenery's (else the
    slot's), its tha and step the walk's; each placement's points the scenery's (a cell list's, required; the scenery
    has no others), else read from `rs` as the placement's object reads them (the car models' and the HUD outline's,
    whose words are not rewritten; where placements share a tpa, the one reading the most words, as decode_course); each
    texture's texels the scenery's (else the slot's), by geometry."""
    lists = {**{k: base.object_lists[k] for k in _slot_lists(base)}, **sd.lists}
    objects = {**{k: base.objects[k] for k in base.car_objects}, **sd.objects}
    cells = tuple(sd.cells)
    hud = empty["hud_map"]
    headers, texels, seen = {}, [], set()
    have = {_texel_key(t.header): t for t in base.texels}
    have.update({_texel_key(t.header): t for t in sd.texels})
    for k, (tha, step) in _header_walk(_drawn(cells, lists, base, hud, objects)).items():
        words = sd.headers.get(k) or base.headers.get(k)
        if words is None:
            raise ConvertError(f"{name}: the texture header at {k:#x} (tha {tha:#x}) a placement reads is neither the "
                               f"scenery's nor the slot's")
        h = headers[k] = dataclasses.replace(words, tha=tha, step=step)
        if h.textured and (key := _texel_key(h)) not in seen:
            seen.add(key)
            if key not in have:
                raise ConvertError(f"{name}: the texture {key} of the header at {k:#x} has no texels (neither the "
                                   f"scenery's nor the slot's)")
            texels.append(golden.Texels(h, have[key].indices, have[key].mips))
    points: dict[int, tuple] = {}
    mine = {p.tpa & T.TEX_ADDR_MASK for k in set(cells) for p in lists[k]}
    if extra := sorted(set(sd.points) - mine):
        raise ConvertError(f"{name}: texture points at {[hex(k) for k in extra[:4]]} are not reached: the scenery's "
                           f"points are those of its cell placements (their tpa), exactly")
    for p, objs in _drawn(cells, lists, base, hud, objects):
        k, obj = p.tpa & T.TEX_ADDR_MASK, objs[C.rom_object_index(p.oba)]
        if k in mine:
            if k not in sd.points:
                raise ConvertError(f"{name}: no texture points at {k:#x}, the tpa of a cell placement (the scenery's "
                                   f"points are written for every cell placement)")
            pts = sd.points[k]
        else:
            try:
                pts = T.texture_points(rs, p.tpa, obj)
            except DecodeError as e:
                raise ConvertError(f"{name}: the slot's texture points at {k:#x}: {e}") from e
        if sum(map(len, pts)) > sum(map(len, points.get(k, ()))):
            points[k] = pts
    return dict(empty, cells=cells, object_lists=lists, objects=objects, headers=headers, points=points,
                texels=tuple(texels))


def _sizes(step_n, steps, road, lines, rows, flags) -> dict[str, int]:
    """The bytes each data-ROM table takes, as its encoder writes it (offset-independent lengths; the per-segment
    flags padded to 4 bytes, the last table's end left 0xff)."""
    try:
        n = len(C.encode_segment_flags(flags))
        return {"step table": len(C.encode_step_table(step_n, list(steps))),
                "road line": len(C.encode_road_line(list(road))),
                "line table": len(C.encode_line_table(0, [list(line) for line in lines])),
                "checkpoint table": len(C.encode_checkpoint_table(list(rows))),
                "segment flags": n + -n % 4}
    except DecodeError as e:
        raise ConvertError(f"converted tables: {e}") from e


def converted_course(rs: RomSet, d: CourseData, exe, n: int, *, slot: int = 0, scenery: str = "keep",
                     layout: Layout | None = None, scenery_data: Scenery | None = None,
                     variant: str = "", plain: CourseData | None = None) -> golden.Course:
    """Dreamcast course `n` (0-7; its data `d` and `exe`, the bytes of 1ST_READ.BIN) as a golden.Course in arcade
    slot `slot` of `rs`: the slot's decoded Course with every race table replaced together (module docstring) and
    each placed in the free runs from `layout` (default: each run's first bytes that are free in `rs` and take the
    tables; the copro_data start also window-safe), its pointers repointed:

    - entry: step_table, record_table, line_table at their places in DATA_FREE, raw words 6-7 holding the old grid
      pointer set to the new one; ground_data (`ground.convert_ground`) placed by `tgp.placed_ground`, and the
      `ground` view its two pointer words;
    - step_n, steps (`tables.steps`), pit (their view: course._pit_of, so the set decodes); road
      (`tables.road_line`), ai_lines (`tables.ai_lines`);
    - checkpoint_table: the new table's i960 address in DATA_FREE; checkpoints: `tables.checkpoint_rows` with their
      views (course.checkpoint_views on the new road line);
    - grid_base: (the slot's grid-base word, GRID_BASES[n]); rolling_start: (the slot's ROLLING_STARTS word,
      ROLLING_STARTS[n]); spawn: the slot's records with their views (segment, lane, position) on the new road line,
      grid base and start;
    - time_tables: the slot's (mode, difficulty) pointers and tables, each read for 1 + laps * (checkpoints + 1)
      words, what the race reads with the new checkpoint count. Where the new course has more checkpoints than the
      slot (n 3: 2 against Beginner's 0) that runs past the slot's stored table: n 3's mode 0 (25 words) stays
      inside Beginner's own tables, modes 1 and 2 (61 and 241 words) read on into the next tables' words. Kept as
      the slot's until M6 tunes them, like laps, car count and the time corrections;
    - segment_flags: (the new table's i960 address in DATA_FREE, after the checkpoint table, `tables.segment_flags`:
      n + 1 zeros; the Dreamcast has none, docs/spike/2026-11-convert.md, "Per-segment flags"), SEGMENT_FLAGS[slot]
      repointed to it. Always a new table, even where the count equals the slot's (n 0): a race table at the slot's
      own address is refused by `encode` (Review Focus 4).

    golden.check_course passes on the result (ConvertError otherwise).

    `scenery="empty"` also replaces SCENERY_FIELDS (`_empty_scenery`, module docstring): cells all EMPTY_LIST;
    object_lists the lists the set then reaches (EMPTY_LIST, empty, and the slot's spawn and map-object lists);
    actor_table the shared EMPTY_ACTORS, actor_slots 256 zeros, no actor lists,
    unmodelled actors, billboards, billboard reference or switch; hud_map the slot's list with its one entry naming
    `hud_outline(road)` at `Layout.polygons_at` (its budget the outline's record count, as the real entries'), the
    place record and zooms `hud_frame(slot's HudMap, slot's road, road)`. objects, headers, points and texels stay
    the slot's. "keep" leaves every scenery field the slot's.

    `scenery="model"` replaces MODEL_FIELDS: the empty scenery's actor table and HUD map, and the scenery of
    `scenery_data` (a Scenery, required then and only then; `_model_scenery`): cells its cells, object_lists its
    lists with the slot's spawn and map-object lists, objects its objects with the car models, headers and texels
    and texture points those the set then reaches in golden.decode_course's order (`_model_scenery`). The
    scenery's tables are not free for the default layout (nor for `layout`), so the race tables and the HUD outline
    are placed around them. `_check_model` passes on the result.

    `slot=NEW_ID` (the module docstring's paragraph on a new id): the record also moves block_ref_a / block_ref_b to
    two pointer words of its own; time_tables are its own (each (mode, difficulty)'s word in one array per mode, the
    tables after them, the slot's words); with "model" also grid (its own cell grid), sheet_sources[0] (its own block,
    `Layout.sheet_at`) and the HUD map's list (its own, `Layout.list_at`); every table in NEW_ID_RUNS (the lists in
    DATA_FREE).

    ConvertError for `rs` not a RomSet, an `n` that is not an integer 0-7, `slot` other than 0 or one of
    CONVERTED_IDS (a new id on a set that is not grown and guarded, or with "empty"), `scenery` other than "keep",
    "empty" or "model", `scenery_data` given without "model" or missing with it, a Scenery `_scenery_data` or
    `_check_model` refuses, a `layout` that is not a Layout or whose spans are not 4-aligned, inside their runs and
    free (or where the ground's lists or polygons would cross a bank window), no room in a run for the default
    layout, a `d` the converters refuse, a course LIFT names whose `d` is not lifted (`_check_lifted`) or whose
    scenery_data was made from a model not lifted by the same amount (Scenery.lift), a `d` not shifted by SHIFT's
    amount for the course (`_check_shifted`: 0 for a course SHIFT does not name) or a scenery_data made from a model
    not shifted by it (Scenery.shift), or a slot the set cannot decode; DcError for an `exe` the Dreamcast reader
    refuses. `plain` (a reverse's plain course data, lifted and shifted as `d`: converted_set passes it for variant
    "R") lends the ground a cell neither repair makes convex (ground.convert_ground's plain=: 4R's cell 1765)."""
    rs = _romset(rs)
    n = _course_n(n)
    slot = _slot(slot, rs)
    variant = _variant(variant, n, slot, scenery)
    if not isinstance(scenery, str) or scenery not in SCENERY:
        raise ConvertError(f"scenery {scenery!r:.60} is not one of {SCENERY}")
    if slot and scenery == "empty":
        raise ConvertError(f"scenery \"empty\" in slot {slot}: a new id takes \"keep\" or \"model\" (the empty "
                           f"scenery rewrites one of Beginner's lists, EMPTY_LIST, in place)")
    if (scenery == "model") != (scenery_data is not None):
        raise ConvertError(f"scenery_data {scenery_data!r:.40} with scenery {scenery!r}: a build.Scenery is drawn with "
                           f"scenery=\"model\", and only then")
    busy = _scenery_data(scenery_data, f"course {n}") if scenery == "model" else []
    if layout is not None and not isinstance(layout, Layout):
        raise ConvertError(f"layout {layout!r:.60} is not a build.Layout")
    road, lines = tables.road_line(d), tables.ai_lines(d)
    _check_lifted(d, n)
    if scenery_data is not None and scenery_data.lift != LIFT.get(n, 0.0):
        raise ConvertError(f"course {n}: its scenery was made from a model raised by {scenery_data.lift}, LIFT names "
                           f"{LIFT.get(n, 0.0)} for it: its data and model are lifted together (lifted_data and "
                           f"lifted_model, as converted_set does)")
    _check_shifted(d, n)
    if scenery_data is not None and scenery_data.shift != SHIFT.get(n, 0.0):
        raise ConvertError(f"course {n}: its scenery was made from a model shifted by {scenery_data.shift} in x, "
                           + (f"not SHIFT[{n}] ({SHIFT[n]})" if n in SHIFT else "SHIFT names none for it")
                           + ": its data and model are shifted together (shifted_data and shifted_model, as "
                           "converted_set does)")
    step_n, steps = tables.steps(d)
    flags = tables.segment_flags(d)
    rows = tables.checkpoint_rows(exe, n, variant)
    g = ground.convert_ground(d, slot, plain=plain)          # plain: a reverse's lender (4R's cell 1765)
    outline = hud_outline(road) if scenery != "keep" else None
    try:
        base = golden.decode_course(rs, slot)
        pit = C._pit_of(list(steps), step_n, f"course {n}: step table")
    except DecodeError as e:
        raise ConvertError(f"course {n} into slot {slot}: {e}") from e
    count = len(rows)
    read = {}                           # (mode, difficulty) -> the slot's time-table words the race reads
    try:
        for key, (_, table, _) in base.time_tables.items():
            size = 1 + base.laps[key[0]][1] * (count + 1)
            read[key] = struct.unpack(f"<{size}I", rs.read_i960(table, 4 * size))
    except DecodeError as e:
        raise ConvertError(f"course {n} into slot {slot}: the slot's time table at {table:#x} for {count} "
                           f"checkpoints: {e}") from e

    own = slot and scenery == "model"                   # a new id's own grid, sheet block and HUD list
    sizes = {"ground grid": 2 * tgp.GRID_CELLS, "ground attributes": len(g.attributes) + -len(g.attributes) % 4,
             **_sizes(step_n, steps, road, lines, rows, flags)}
    order = DATA_ORDER
    if slot:
        sizes.update({"ground pointers": 8, "time pointers": 4 * C.MODES * C.DIFFICULTIES,
                      "time tables": sum(4 * len(w) for w in read.values()), "cell grid": 4 * C.GRID_CELLS * own})
        order = DATA_ORDER + ID_DATA_ORDER
    need = {"data": sum(sizes.values()),
            "copro_data": 4 * (tgp.list_block_words(g.lists) + tgp.POLY_WORDS * len(g.polys)),
            "program": 0, "polygons": 4 * outline.word_count if outline else 0,
            "sheet": T.SHEET_BLOCK if own else 0, "list": (C.OBJECT_ENTRY_SIZE + 4) if own else 0}
    lay = _layout(rs, layout, need, g, busy, slot)
    at, k = {}, lay.data_at
    for name in order:
        at[name] = k
        k += sizes[name]
    g = tgp.placed_ground(g, lay.copro_at // 4, lay.data_at)

    moved = dict(block_ref_a=at["ground pointers"], block_ref_b=at["ground pointers"] + 4) if slot else {}
    entry = dataclasses.replace(base.entry, step_table=at["step table"], record_table=at["road line"],
                                line_table=at["line table"], **moved,
                                raw=tgp.repoint_record_raw(base.entry.raw, DATA_BASE + base.ground_data.grid_at,
                                                           DATA_BASE + g.grid_at))
    time_tables = {}
    if slot:                            # a new id's own words and tables, the slot's words copied
        tables_at = DATA_BASE + at["time tables"]
        for key in sorted(read):
            ptr = DATA_BASE + at["time pointers"] + 4 * (C.DIFFICULTIES * key[0] + key[1])
            time_tables[key] = (ptr, tables_at, read[key])
            tables_at += 4 * len(read[key])
    else:
        time_tables = {key: (ptr, table, read[key]) for key, (ptr, table, _) in base.time_tables.items()}
    mode = (C.DEFAULT_MODE, C.DEFAULT_DIFFICULTY)
    grid_base = (base.grid_base[0], GRID_BASES[n] if variant == "" else dc_exe.grid_base(exe, n, variant))
    rolling_start = (base.rolling_start[0], ROLLING_STARTS[n])
    try:
        checkpoints = tuple(C.checkpoint_views([(r.segment, r.raw) for r in rows], list(road), time_tables[mode][2],
                                               base.laps[C.DEFAULT_MODE][1]))
        spawn = tuple(tuple(C.grid_slot(s.car, s.objects, s.line, s.start, s.speed, s.kind, list(road),
                                        grid_base[1], tuple(base.grid_lanes), rolling_start[1]) for s in slots)
                      for slots in base.spawn)
    except DecodeError as e:
        raise ConvertError(f"course {n} into slot {slot}: the checkpoints' and grid's views: {e}") from e
    out = dataclasses.replace(
        base, entry=entry, step_n=step_n, steps=steps, pit=pit, ground=(DATA_BASE + g.grid_at,
                                                                        DATA_BASE + g.attributes_at),
        ground_data=g, road=road, ai_lines=lines, checkpoint_table=DATA_BASE + at["checkpoint table"],
        checkpoints=checkpoints, time_tables=time_tables, grid_base=grid_base, rolling_start=rolling_start,
        spawn=spawn,
        segment_flags=(DATA_BASE + at["segment flags"], flags),
        **(_empty_scenery(base, road, outline, lay) if outline else {}),
        **(dict(grid=DATA_BASE + at["cell grid"], sheet_sources=(lay.sheet_at,) + base.sheet_sources[1:])
           if own else {}))
    if scenery == "model":
        name = f"course {n} into slot {slot}"
        out = dataclasses.replace(out, **_model_scenery(rs, base, {f: getattr(out, f) for f in SCENERY_FIELDS},
                                                        scenery_data, name))
        _check_model(out, base, name)
    try:
        golden.check_course(out)
    except DecodeError as e:
        raise ConvertError(f"course {n} into slot {slot}: {e}") from e
    return out


# ---- the set ----

def _in_run(region: str, off: int, size: int, slot: int = 0) -> bool | None:
    """True when the bytes lie inside one of slot `slot`'s free runs of the region (slot 0: DATA_FREE, COPRO_FREE,
    PROGRAM_FREE, POLYGON_FREE, TEXTURES_FREE; a new id: NEW_ID_RUNS, and DATA_FREE, its list home), False when they
    lie outside them, None when they straddle a run's edge."""
    runs = _runs(slot)
    if region not in runs:
        return False
    for lo, n in [runs[region]] + ([tgp.DATA_FREE] if slot and region == "data" else []):
        if lo <= off and off + size <= lo + n:
            return True
        if not (off + size <= lo or lo + n <= off):
            return None
    return False


def _in_list_home(addr: int, size: int) -> bool:
    """True when `size` bytes from i960 address `addr` lie inside the list home (LIST_HOME: DATA_FREE)."""
    lo, n = tgp.DATA_FREE
    return DATA_BASE + lo <= addr and addr + size <= DATA_BASE + lo + n


def _write_free(rs: RomSet, region: str, off: int, blob: bytes) -> RomSet:
    """`rs` with `blob` written into a free run where every byte is 0xff, none is reserved (romset.RESERVED) and none
    is owned by a table or block (golden.owned_spans, through the per-set cache golden.owned, carried to the result:
    golden.carry_owned); data, copro_data: tgp.write_free; program, polygons, textures: here. DecodeError (TgpError,
    ConvertError) otherwise."""
    if region in ("data", "copro_data"):
        return tgp.write_free(rs, region, off, blob)
    if owner := reserved(region, off, len(blob)):
        raise ConvertError(f"{region} {off:#x}+{len(blob):#x}: reserved for {owner} (romset.RESERVED)")
    if getattr(rs, region)[off:off + len(blob)] != bytes([_FREE]) * len(blob):
        raise ConvertError(f"{region} {off:#x}+{len(blob):#x}: not free (a byte that is not 0xff)")
    try:
        owner = golden.owner(rs, region, off, len(blob))
    except DecodeError as e:
        raise ConvertError(f"{region} {off:#x}+{len(blob):#x}: the set's owned space (golden.owned_spans): {e}") from e
    if owner:
        raise ConvertError(f"{region} {off:#x}+{len(blob):#x}: owned by {owner} (golden.owned_spans), whatever its "
                           f"bytes")
    out = rs.with_patch(region, off, blob)
    golden.carry_owned(rs, out)
    return out


EMPTY_TABLE = bytes(4 * C.GRID_CELLS)
"""EMPTY_ACTORS' bytes: 256 zero words (course.encode_actor_table of no list)."""


def empty_actors(rs: RomSet) -> RomSet:
    """`rs` with the shared empty actor table (EMPTY_ACTORS: EMPTY_TABLE at program EMPTY_AT, reserved) written when
    its span is free (every byte 0xff), `rs` itself when it holds the table already (an earlier course named it).
    ConvertError for `rs` not a RomSet or the span holding other bytes (a set this rule did not write)."""
    rs = _romset(rs)
    held = rs.program[EMPTY_AT:EMPTY_AT + len(EMPTY_TABLE)]
    if held == EMPTY_TABLE:
        return rs
    if held != bytes([_FREE]) * len(EMPTY_TABLE):
        k = next(i for i, v in enumerate(held) if v != _FREE)
        raise ConvertError(f"EMPTY_ACTORS {EMPTY_ACTORS:#x}: program {EMPTY_AT + k:#x} holds {held[k]:#04x}: the "
                           f"shared empty actor table's span is neither free (0xff) nor the table (256 zero words)")
    return rs.with_patch("program", EMPTY_AT, EMPTY_TABLE)


def program_room(rs: RomSet) -> int:
    """The free bytes of PROGRAM_FREE on `rs`: 0xff and owned by nothing (`_usable`: golden.owned_spans, so the
    reserved EMPTY_ACTORS span never counts), what courses with actors can still take for their tables and lists.
    ConvertError for `rs` not a RomSet or a set whose courses do not decode."""
    return int(_usable(_romset(rs), "program").sum())


def _merged(patches: list[tuple[str, int, bytes]]) -> list[tuple[str, int, bytes]]:
    """(region, offset, bytes) patches in (region, offset) order with each one that starts where the one before it
    ends joined onto it (the model scenery's headers, one per drawn polygon, are one run written with one copy of the
    region, not one each). Written the same: every byte must still be free. A data or copro_data patch is joined only at a 4-aligned
    offset, so one `tgp.write_free` would refuse alone is still refused."""
    out: list[tuple[str, int, bytearray]] = []
    for region, off, blob in patches:
        if out and out[-1][0] == region and out[-1][1] + len(out[-1][2]) == off and (
                region not in ("data", "copro_data") or off % 4 == 0):
            out[-1][2].extend(blob)
        else:
            out.append((region, off, bytearray(blob)))
    return [(region, off, bytes(blob)) for region, off, blob in out]


def _same(a, b) -> bool:
    """Equal and of the same type all the way down (tuples, lists, dicts, dataclasses, numpy arrays): never raises,
    so junk (a numpy value where an int is, an array where a tuple is) is unequal, not an error."""
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        try:
            if len(a) != len(b) or set(a) != set(b):
                return False
            return all(_same(v, b[k]) for k, v in a.items())
        except Exception:                   # a key whose == is not a bool (an array): not the slot's
            return False
    if dataclasses.is_dataclass(a):
        return all(_same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a) if f.compare)
    try:
        return (a == b) is True
    except Exception:                       # an object whose == raises: not the slot's
        return False


_PLAIN = (type(None), bool, int, float, str, bytes)


def _plain(v, depth: int = 0) -> bool:
    """True for values made only of None, bool, int, float, str, bytes, tuples, lists and dicts of them, NamedTuples
    and dataclasses (what converted_course and golden.decode_course give): no numpy value, no other object."""
    if depth > 8:
        return False
    if type(v) in _PLAIN:
        return True
    if isinstance(v, (tuple, list)):
        return all(_plain(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return all(_plain(k, depth + 1) and _plain(x, depth + 1) for k, x in v.items())
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return all(_plain(getattr(v, f.name), depth + 1) for f in dataclasses.fields(v))
    return False


def _fields(fields) -> tuple[str, ...]:
    names = {f.name for f in dataclasses.fields(golden.Course)} - {"course", "entry"}
    if not isinstance(fields, (tuple, frozenset, set, list)) or not fields or not all(isinstance(f, str)
                                                                                       for f in fields):
        raise ConvertError(f"fields {fields!r:.60} are not golden.Course field names")
    if bad := sorted(set(fields) - names):
        raise ConvertError(f"fields {bad} are not golden.Course fields a conversion may replace")
    if (part := [f for f in SCENERY_FIELDS if f in fields]) and len(part) != len(SCENERY_FIELDS):
        raise ConvertError(f"fields: the scenery fields are replaced together (Review Focus 5), not "
                           f"{', '.join(part)} without {', '.join(f for f in SCENERY_FIELDS if f not in part)}")
    if (part := [f for f in MODEL_FIELDS if f in fields]) and len(part) != len(MODEL_FIELDS) and \
            any(f in fields for f in MODEL_FIELDS[len(SCENERY_FIELDS):]):
        raise ConvertError(f"fields: the model scenery's fields are replaced together (MODEL_FIELDS), not "
                           f"{', '.join(part)} without {', '.join(f for f in MODEL_FIELDS if f not in part)}")
    return tuple(dict.fromkeys(fields))


def _check_against_slot(c: golden.Course, ref: golden.Course, fields: tuple[str, ...], name: str) -> None:
    """Every field outside `fields` the slot's (`ref`); the record the slot's but for the moved tables' pointers and
    words 6-7 as tgp.repoint_record_raw changes them; the replaced fields plain values (`_plain`), grid_base,
    rolling_start and the time tables at the slot's addresses, ground_data a tgp.Ground. A new id also moves the
    record's two ground pointer words (block_ref_a, block_ref_b) and its time tables (their words and tables new:
    `encode` places them). ConvertError naming the field otherwise."""
    slot = ref.course
    moved = _MOVED + (_ID_MOVED if slot else ())
    for f in dataclasses.fields(golden.Course):
        if f.name not in fields + ("course", "entry") and not _same(getattr(c, f.name), getattr(ref, f.name)):
            raise ConvertError(f"{name}: {f.name} differs from the slot's: the conversion replaces only "
                               f"{', '.join(fields)} (and the record's {', '.join(moved)})")
    for f in fields:
        if f != "texels" and not _plain(getattr(c, f)):         # texels hold arrays: _check_model checks them
            raise ConvertError(f"{name}: {f} {getattr(c, f)!r:.60} holds a value that is not a plain int, float, "
                               f"bytes, sequence or record (a numpy value or another object)")
    if "grid_base" in fields:
        gb = c.grid_base
        if not isinstance(gb, tuple) or len(gb) != 2 or not _same(gb[0], ref.grid_base[0]):
            raise ConvertError(f"{name}: grid_base {gb!r:.60} is not (the slot's grid-base word "
                               f"{ref.grid_base[0]:#x}, base)")
    if "rolling_start" in fields:
        rl = c.rolling_start
        if not isinstance(rl, tuple) or len(rl) != 2 or not _same(rl[0], ref.rolling_start[0]) \
                or not isinstance(rl[1], bool):
            raise ConvertError(f"{name}: rolling_start {rl!r:.60} is not (the slot's start word "
                               f"{ref.rolling_start[0]:#x}, a bool)")
    if "time_tables" in fields:
        tt = c.time_tables
        if not isinstance(tt, dict) or set(tt) != set(ref.time_tables) or not all(
                isinstance(v, tuple) and len(v) == 3 and (
                    _same(v[:2], ref.time_tables[k][:2]) if not slot else all(is_int(a) and a % 4 == 0 for a in v[:2]))
                for k, v in tt.items()):
            raise ConvertError(f"{name}: time_tables {tt!r:.60} are not the slot's (mode, difficulty) tables at the "
                               + ("slot's pointer and table addresses" if not slot else "4-aligned addresses"))
        if slot and not all(_same(v[0], tt[(m, 0)][0] + 4 * d) for (m, d), v in tt.items()):
            raise ConvertError(f"{name}: time_tables: each mode's per-difficulty words are not one array (the mode's "
                               f"level entry names its first word)")
    g = c.ground_data
    if "ground_data" in fields and (not isinstance(g, tgp.Ground) or not is_int(g.grid_at)):
        raise ConvertError(f"{name}: ground_data {g!r:.60} is not a tgp.Ground with a grid address")
    e, want = c.entry, ref.entry
    for f in dataclasses.fields(C.CourseEntry):
        if f.name not in moved + ("raw",) and not _same(getattr(e, f.name), getattr(want, f.name)):
            raise ConvertError(f"{name}: entry.{f.name} {getattr(e, f.name)!r:.40} is not the slot's "
                               f"{getattr(want, f.name)!r:.40}: a conversion moves only {', '.join(moved)}")
    raw = tgp.repoint_record_raw(want.raw, DATA_BASE + ref.ground_data.grid_at, DATA_BASE + int(g.grid_at)) \
        if isinstance(g, tgp.Ground) and is_int(g.grid_at) else want.raw
    if not _same(e.raw, raw):
        raise ConvertError(f"{name}: entry.raw {e.raw!r:.40} is not {raw!r:.40}: the record's words 6-7 change "
                           f"only where they hold the grid pointer (tgp.repoint_record_raw)")


def _check_scenery(c: golden.Course, ref: golden.Course, name: str) -> None:
    """The empty scenery's fields (SCENERY_FIELDS) of `c` against the slot's decoded `ref`, before golden reads them
    (so junk is refused here, never a raw error there), each table fixed by address: the 256 cells all EMPTY_LIST,
    one of the slot's own cell lists; object_lists keyed by the slot's lists only, every one the slot's but
    EMPTY_LIST, which is empty (and present); actor_table the shared EMPTY_ACTORS; actor_slots 256 zeros; no actor
    lists, unmodelled actors, billboards, billboard reference or switch; hud_map a course.HudMap of the slot's id naming
    the slot's outline list with as many entries, each the slot's but its object and budget, every outline object a
    polyrom.Object of polyrom.Record that encodes (polyrom.encode_object) inside POLYGON_FREE and is
    `hud_outline(c.road)`, the place record, angle and zooms `hud_frame`'s (the pad the slot's), and the map as
    encode_hud_map takes it (Review Focus 5: the map is the new course's). ConvertError naming the field otherwise."""
    def fail(field: str, detail: str):
        raise ConvertError(f"{name}: {field}: {detail}")

    if not _same(c.cells, (EMPTY_LIST,) * C.GRID_CELLS):
        fail("cells", f"{c.cells!r:.60} do not all name EMPTY_LIST {EMPTY_LIST:#x}: the empty scenery's 256 cells "
             f"name the one empty list (Review Focus 5)")
    if EMPTY_LIST not in ref.cells:
        fail("cells", f"EMPTY_LIST {EMPTY_LIST:#x} is not one of the slot's cell lists: only a list of the slot's "
             f"own is rewritten to its terminator")
    lists = c.object_lists
    if not isinstance(lists, dict) or not all(is_int(k) for k in lists):
        fail("object_lists", f"{lists!r:.60} is not a dict of list addresses")
    if not _same(lists.get(EMPTY_LIST), ()):
        fail("object_lists", f"EMPTY_LIST {EMPTY_LIST:#x} is not there, empty")
    for k, v in lists.items():
        if k != EMPTY_LIST and (k not in ref.object_lists or not _same(v, ref.object_lists[k])):
            fail("object_lists", f"the list at {k:#x} is not the slot's: the empty scenery changes no list but "
                 f"EMPTY_LIST (the object library has no free run)")
    _check_actors_hud(c, ref, name)


def _check_actors_hud(c: golden.Course, ref: golden.Course, name: str) -> None:
    """The actor table and HUD map of the empty and the model scenery (`_check_scenery`'s docstring): ACTOR_TABLES[slot]
    the shared empty table EMPTY_ACTORS (256 zeros), no actor list, billboard or switch, the HUD map the new course's
    outline in POLYGON_FREE named by the slot's list, framed by `hud_frame`. A new id (`ref.course` NEW_ID): its outline
    in the custom polygon run, named by a list of its own (one entry, at its own offset) in the list home. ConvertError
    naming the field otherwise."""
    def fail(field: str, detail: str):
        raise ConvertError(f"{name}: {field}: {detail}")

    slot = ref.course

    t = c.actor_table
    if not _same(t, EMPTY_ACTORS):
        fail("actor_table", f"{t!r:.20} is not EMPTY_ACTORS {EMPTY_ACTORS:#x}: a converted course has no actors, its "
             f"table is the shared empty one")
    if not _same(c.actor_slots, (0,) * C.GRID_CELLS):
        fail("actor_slots", f"{c.actor_slots!r:.60} are not 256 zeros: the empty scenery's actor table names no list "
             f"(Review Focus 5)")
    for f, empty in (("actor_lists", {}), ("billboards", {}), ("billboard_ref", None), ("switch_lists", ()),
                     ("unmodelled", ())):
        if not _same(getattr(c, f), empty):
            fail(f, f"{getattr(c, f)!r:.60} is not {empty!r}: the empty scenery's actor table names no list")
    h, want = c.hud_map, ref.hud_map
    if not isinstance(h, C.HudMap) or not _same(h.course, want.course) or not (
            _same(h.outline_list, want.outline_list) if not slot else
            is_int(h.outline_list) and h.outline_list % 4 == 0
            and _in_list_home(h.outline_list, C.OBJECT_ENTRY_SIZE * len(want.entries) + 4)):
        fail("HUD map", f"{h!r:.60} is not a course.HudMap of course {want.course} naming "
             + (f"the slot's outline list {want.outline_list:#x} (rewritten in place)" if not slot else
                "a 4-aligned list of its own in the list home DATA_FREE"))
    if (not isinstance(h.entries, tuple) or len(h.entries) != len(want.entries)
            or not all(isinstance(e, C.Placement) for e in h.entries)):
        fail("HUD map", f"entries {h.entries!r:.60} are not {len(want.entries)} course.Placement: the slot's list is "
             f"rewritten in place, as long as it was")
    for j, (e, w) in enumerate(zip(h.entries, want.entries)):
        if slot and not _same(e.entry, h.outline_list - DATA_BASE + C.OBJECT_ENTRY_SIZE * j):
            fail("HUD map", f"entry {j} at {e.entry!r:.20} is not its list's entry {j}")
        for f in dataclasses.fields(C.Placement):
            if f.name not in ("oba", "raw", "entry" if slot else "") and not _same(getattr(e, f.name),
                                                                                  getattr(w, f.name)):
                fail("HUD map", f"entry {f.name} {getattr(e, f.name)!r:.40} is not the slot's: only the object and "
                     f"its budget change")
    if not isinstance(h.objects, dict) or not all(
            is_int(k) and isinstance(o, P.Object) and isinstance(o.records, (list, tuple))
            and all(isinstance(r, P.Record) for r in o.records) for k, o in h.objects.items()):
        fail("HUD map", f"objects {h.objects!r:.60} are not polyrom.Object (of polyrom.Record) by word index")
    try:
        named = {C.rom_object_index(e.oba) for e in h.entries}
    except DecodeError as e:
        fail("HUD map", f"entries: {e}")
    if named != set(h.objects):
        fail("HUD map", f"the entries name objects {sorted(named)} but the map holds {sorted(h.objects)}: each entry "
             f"names one of its outline objects, and nothing else is written")
    try:
        C.encode_hud_map(h, c.id_arrays or None)
        words = {k: P.encode_object(o) for k, o in h.objects.items()}
    except DecodeError as e:
        raise ConvertError(f"{name}: HUD map: {e}") from e
    for k, w in words.items():
        if _in_run("polygons", 4 * k, 4 * len(w), slot) is not True:
            fail("HUD map", f"outline object {k:#x} ({len(w)} words) is not inside the polygon run "
                 f"{_run_name('polygons', slot)}: a new outline is written into free space")
    outline = hud_outline(c.road)
    if not all(_same(o, outline) for o in h.objects.values()):
        fail("HUD map", "the outline is not hud_outline(road): the map would not show the course's road")
    position, angle, zooms = hud_frame(want, ref.road, c.road)
    if not _same((h.position, h.angle, h.pad, h.zooms), (position, angle, want.pad, zooms)):
        fail("HUD map", f"place {h.position!r:.60}, angle {h.angle!r}, pad {h.pad!r} and zooms {h.zooms!r:.60} are not "
             f"hud_frame's for the road (and the slot's pad)")


def _check_model(c: golden.Course, ref: golden.Course, name: str) -> None:
    """The model scenery's fields (MODEL_FIELDS) of `c` against the slot's decoded `ref`, before
    golden reads them (junk refused here, never a raw error there), each table fixed by address:

    - the actor table and HUD map as the empty scenery's (`_check_actors_hud`);
    - cells: 256 list addresses, none of them a list of the slot's spawn records or map object; object_lists: exactly
      the cells' lists and those slot lists (the latter the slot's); each cell list 4-aligned and, with its
      terminator, inside the list home (DATA_FREE), each entry a course.Placement as decode_course reads it back
      (`object_list_at`: obc OBC, cell -1, its own entry offset, in world, no actor) with u32 words, its object one of
      `objects`, its tpa and tha not texture RAM (bit 23);
    - objects: exactly the car models (the slot's) and the objects the cell lists name, each a polyrom.Object of
      normal-present records (geometry mode 1, as the race decodes them) that encodes inside POLYGON_FREE;
    - headers: exactly those the placements read (`_header_walk`, the cell lists, the spawn lists, the HUD map), each
      with the walk's tha and step; a header of the slot's (`ref.headers`) keeps its words, every other lies inside
      TEXTURES_FREE and a textured one is in sheet 0;
    - texels: one golden.Texels per texture geometry the headers read, in the walk's order, its header the first
      reaching one; the slot's texels where they are the slot's, the others in sheet 0 (the slot's course block,
      `encode` writes nothing else of the sheets);
    - points: exactly the tpa word indices of the placements the set reaches (the cell lists', the spawn lists' and
      the HUD outline's), each the reading of the placement there that reads the most words (decode_course's rule):
      one entry per record of its object, 3 or 4 (v, u) by the record (texture.encode_texture_points), u16; a cell
      list's inside TEXTURES_FREE, overlapping no other points nor a new header (the others are the slot's words,
      which `encode` refuses to change);
    - no two cell lists overlap, nor two of the new objects and the HUD outline: a Course that passes encodes.

    A new id (`ref.course` NEW_ID): its objects in the custom polygon run, its headers and points in the custom
    textures run (NEW_ID_RUNS), its lists in the list home as slot 0's (none overlapping its HUD outline's list), its
    sheet 0 its own course block (`c.sheet_sources[0]`, 1 MB in ID_DATA_FREE: `encode` copies the slot's block there
    first, so the slot's texels it keeps read the same).

    ConvertError naming the field otherwise."""
    def fail(field: str, detail: str):
        raise ConvertError(f"{name}: {field}: {detail}")

    _check_actors_hud(c, ref, name)
    slot = ref.course
    if slot:
        sheet = c.sheet_sources
        if not isinstance(sheet, tuple) or len(sheet) != 2 or not _same(sheet[1], ref.sheet_sources[1]) \
                or not is_int(sheet[0]) or sheet[0] % T.SHEET_BLOCK \
                or not ID_DATA_FREE[0] <= sheet[0] <= sum(ID_DATA_FREE) - T.SHEET_BLOCK:
            fail("sheet_sources", f"{sheet!r:.60} are not (a 1 MB-aligned course block of its own in "
                 f"{ID_DATA_FREE[0]:#x}+{ID_DATA_FREE[1]:#x}, the common block {ref.sheet_sources[1]:#x})")
    cells, lists, objects, headers = c.cells, c.object_lists, c.objects, c.headers
    if not isinstance(cells, tuple) or len(cells) != C.GRID_CELLS or not all(is_int(k) for k in cells):
        fail("cells", f"{cells!r:.60} are not {C.GRID_CELLS} list addresses")
    if not _lists_ok(lists):
        fail("object_lists", f"{lists!r:.60} are not tuples of course.Placement (u32 words) by list address")
    if not _objects_ok(objects):
        fail("objects", f"{objects!r:.60} are not polyrom.Object (of polyrom.Record) by word index")
    if not _headers_ok(headers):
        fail("headers", f"{headers!r:.60} are not texture.TexHeader by textures word index")
    if not _texels_ok(c.texels):
        fail("texels", f"{c.texels!r:.60} are not a tuple of golden.Texels")
    points = c.points
    if not _points_ok(points):
        fail("points", f"{points!r:.60} are not texture points (per record 3 or 4 (v, u) tuples of u16 ints) by "
             f"textures word index")
    keep, new = _slot_lists(ref), set(cells)
    if new & keep:
        fail("cells", f"a cell names the slot's list {min(new & keep):#x} (a spawn or map-object list): the model "
             f"scenery's cells name its own lists")
    if set(lists) != new | keep:
        extra, missing = sorted(set(lists) - new - keep), sorted(new | keep - set(lists))
        fail("object_lists", f"lists {[hex(k) for k in extra[:4]]} are not reached, {[hex(k) for k in missing[:4]]} "
             f"are missing: the set reaches the cells' lists and the slot's spawn and map-object lists, exactly")
    for k in keep:
        if not _same(lists[k], ref.object_lists[k]):
            fail("object_lists", f"the slot's list at {k:#x} differs from the slot's: it is not rewritten")
    cars = ref.car_objects
    for k in cars:
        if not _same(objects.get(k), ref.objects[k]):
            fail("objects", f"the car model at {k:#x} differs from the slot's")
    named, cell_tpas = set(), set()
    for k in sorted(new):
        v = lists[k]
        size = C.OBJECT_ENTRY_SIZE * len(v) + 4
        if k % 4 or not _in_list_home(k, size):
            fail("object_lists", f"the list at {k:#x} ({size} bytes) is not 4-aligned inside the list home "
                 f"(LIST_HOME {LIST_HOME!r}: DATA_FREE {DATA_BASE + tgp.DATA_FREE[0]:#x}+{tgp.DATA_FREE[1]:#x})")
        for j, p in enumerate(v):
            what = f"the list at {k:#x} entry {j}"
            if not _same((p.obc, p.cell, p.entry, p.in_world, p.actor),
                         (C.OBC, -1, k - DATA_BASE + C.OBJECT_ENTRY_SIZE * j, True, None)):
                fail("object_lists", f"{what}: {p!r:.80} is not as decode_course reads it back (object_list_at)")
            try:
                index = C.rom_object_index(p.oba)
            except DecodeError as e:
                fail("object_lists", f"{what}: {e}")
            if index not in objects or index in cars:
                fail("objects", f"{what} names object {p.oba:#x}, not one of the scenery's objects")
            if (p.tpa | p.tha) & T.TEX_RAM_SELECT:
                fail("object_lists", f"{what}: tpa {p.tpa:#x} / tha {p.tha:#x} select texture RAM (bit 23), not "
                     f"the ROM")
            named.add(index)
            cell_tpas.add(p.tpa & T.TEX_ADDR_MASK)
    if set(objects) != cars | named:
        fail("objects", f"objects {[hex(k) for k in sorted(set(objects) - cars - named)[:4]]} are not reached: the set "
             f"decodes the car models and the objects the cell lists name, exactly")
    hud_list = [(c.hud_map.outline_list - DATA_BASE, C.OBJECT_ENTRY_SIZE * len(c.hud_map.entries) + 4,
                 "HUD outline's list")] if slot else []
    if pair := _overlap([(k - DATA_BASE, C.OBJECT_ENTRY_SIZE * len(lists[k]) + 4, f"list {k:#x}") for k in new]
                        + hud_list):
        fail("object_lists", f"the {pair[0][2]} and the {pair[1][2]} overlap in DATA_FREE")
    taken, placed = [], [(4 * k, 4 * o.word_count, f"HUD outline {k:#x}") for k, o in c.hud_map.objects.items()]
    for k in sorted(named):
        o = objects[k]
        if any(r.normal is None for r in o.records):
            fail("objects", f"object {k:#x} has normal-less records: the race decodes objects in geometry mode 1 "
                 f"(normals present)")
        try:
            words = P.encode_object(o)
        except DecodeError as e:
            fail("objects", f"object {k:#x}: {e}")
        if _in_run("polygons", 4 * k, 4 * len(words), slot) is not True:
            fail("objects", f"object {k:#x} ({len(words)} words) is not inside the polygon run "
                 f"{_run_name('polygons', slot)}: a new object is written into free space")
        if not _same(P.decode_object(struct.pack(f"<{len(words)}I", *words), 0, C.OBC, golden.GEOMETRY_MODE), o):
            fail("objects", f"object {k:#x} does not decode back equal in geometry mode {golden.GEOMETRY_MODE} (its "
                 f"values must be single precision, as the ROM holds them)")
        placed.append((4 * k, 4 * len(words), f"object {k:#x}"))
    if pair := _overlap(placed):
        fail("objects", f"the {pair[0][2]} and the {pair[1][2]} overlap in the polygon run")
    walk = _header_walk(_drawn(cells, lists, ref, c.hud_map, objects))
    if set(headers) != set(walk):
        fail("headers", f"headers {[hex(k) for k in sorted(set(headers) ^ set(walk))[:4]]}: the set decodes exactly "
             f"the headers the placements read")
    for k, (tha, step) in walk.items():
        h = headers[k]
        if not _same((h.tha, h.step), (tha, step)):
            fail("headers", f"the header at {k:#x}: tha {h.tha!r}, step {h.step!r} are not {tha:#x}, {step} (the "
                 f"address sent and the first reaching record's step, as decode_course reads them)")
        try:
            blob = T.encode_tex_header(h)
            mine = k in ref.headers and blob == T.encode_tex_header(ref.headers[k])
        except DecodeError as e:
            fail("headers", f"the header at {k:#x}: {e}")
        if not mine and (_in_run("textures", 2 * k, 8, slot) is not True or (h.textured and h.sheet != 0)):
            fail("headers", f"the header at {k:#x} is not the slot's and not a sheet-0 header inside the textures run "
                 f"{_run_name('textures', slot)}")
        if not mine:
            taken.append((2 * k, 8, f"header {k:#x}"))
    textured = [headers[k] for k in walk if headers[k].textured]
    order = list(dict.fromkeys(_texel_key(h) for h in textured))
    if not _same([_texel_key(t.header) for t in c.texels], order):
        fail("texels", f"textures {[_texel_key(t.header) for t in c.texels][:3]} are not {order[:3]}: one golden.Texels "
             f"per texture the headers read, in decode_course's order")
    slot_texels = {_texel_key(t.header): t for t in ref.texels}
    for t in c.texels:
        h = next(h for h in textured if _texel_key(h) == _texel_key(t.header))
        if not _same(t.header, h):
            fail("texels", f"the texture {_texel_key(h)}: its header is not the first one reaching it ({h.tha:#x})")
        theirs = slot_texels.get(_texel_key(h))
        if theirs is not None and _same((t.indices, t.mips), (theirs.indices, theirs.mips)):
            continue
        if h.sheet != 0:
            fail("texels", f"the texture {_texel_key(h)} is not the slot's and not in sheet 0 (the slot's course block, "
                 f"rewritten in place; sheet 1 is never written)")
        if not isinstance(t.mips, tuple):
            fail("texels", f"the texture {_texel_key(h)}: mips {t.mips!r:.40} are not a tuple of levels")
        try:
            T.encode_texels(h, t.indices)
            levels = T.mip_levels(h)
            if len(t.mips) != len(levels):
                fail("texels", f"the texture {_texel_key(h)}: {len(t.mips)} mip levels, it has {len(levels)}")
            for level, m in zip(levels, t.mips):
                T.encode_mip_bytes(h, level, m, c.sheet_sources[0])
        except DecodeError as e:
            fail("texels", f"the texture {_texel_key(h)}: {e}")
    readers: dict[int, P.Object] = {}                  # tpa word index -> the object read there the furthest
    for p, objs in _drawn(cells, lists, ref, c.hud_map, objects):
        k, o = p.tpa & T.TEX_ADDR_MASK, objs[C.rom_object_index(p.oba)]
        if k not in readers or _point_words(o) > _point_words(readers[k]):
            readers[k] = o
    if set(points) != set(readers):
        fail("points", f"points at {[hex(k) for k in sorted(set(points) ^ set(readers))[:4]]}: the set decodes "
             f"exactly the texture points its placements read (their tpa)")
    for k, v in points.items():
        try:
            blob = T.encode_texture_points(v, readers[k])
        except DecodeError as e:
            fail("points", f"the points at {k:#x} (for the object read there the furthest): {e}")
        if k in cell_tpas:
            if _in_run("textures", 2 * k, len(blob), slot) is not True:
                fail("points", f"the points at {k:#x} (a cell placement's) are not inside the textures run "
                     f"{_run_name('textures', slot)}")
            taken.append((2 * k, len(blob), f"points {k:#x}"))
    if pair := _overlap(taken):
        fail("points", f"the {pair[0][2]} and the {pair[1][2]} overlap in the textures run")


def encode(rs: RomSet, c: golden.Course, fields: tuple[str, ...] = RACE_FIELDS) -> RomSet:
    """`rs` with the converted Course `c` written: its new tables into the free runs (`_write_free`: each byte must
    be 0xff), every other table as golden.encode_course writes it. `fields` are the Course fields the
    conversion may replace (RACE_FIELDS; RACE_FIELDS + SCENERY_FIELDS for the empty scenery). Refused with
    ConvertError naming the field or table (Review Focus 4: a slot never races on a race table left as the slot's;
    Review Focus 1: nothing but the free runs and the in-place words changes; Review Focus 5: the empty scenery
    names no list but the empty one):

    - `rs` not a RomSet, `c` not a golden.Course, `c.course` or `c.entry.course` other than slot 0 or one of
      CONVERTED_IDS (a new id: on a grown, guarded set), `fields` not Course field names, or a part of SCENERY_FIELDS
      without the rest; for a new id the empty scenery, ID_FIELDS without MODEL_FIELDS, or MODEL_FIELDS without
      ID_FIELDS (its new tables, ID_PLACED, must lie in its runs; what it rewrites in place adds ID_IN_PLACE; its
      sheet block is the slot's block with its texels written over it, `_own_block`; its TIME_TABLES levels are
      written, `_time_levels`);
    - any field golden would fill from the set (golden.unset_fields): a replaced field so filled is the slot's own
      table;
    - a field outside `fields` that differs from the slot's decoded Course (`golden.decode_course(rs, slot)`;
      compared by type and value, so junk is unequal, never an error); a record that differs from the slot's but
      in the moved tables' pointers (step_table, record_table, line_table) and words 6-7 as
      `tgp.repoint_record_raw` repoints them to the new grid; a replaced field holding a value that is not plain
      (`_plain`: a numpy value or another object); grid_base, rolling_start or the time tables away from the
      slot's addresses (rolling_start's value not a bool);
    - with SCENERY_FIELDS, scenery fields that are not the empty scenery's or are away from their places
      (`_check_scenery`, which validates them before golden reads them: junk is refused there, never a raw error);
    - with MODEL_FIELDS (`fields` RACE_FIELDS + MODEL_FIELDS: the model scenery), scenery fields (texture points
      included) that `_check_model` refuses (the same rule for junk); a set whose object-library bounds are neither
      the original's nor patched (course.library_bounds);
    - a Course golden.check_course refuses (the tables against each other);
    - a new table (DATA_ORDER, COPRO_ORDER; with the scenery SCENERY_PLACED) not inside its free run: at the slot's
      address it would be written over the slot's table and whatever follows it;
    - a table outside the free runs whose bytes differ from `rs` that is not one the conversion rewrites in place
      (REWRITTEN_IN_PLACE; with the scenery also SCENERY_IN_PLACE; with the model scenery MODEL_IN_PLACE, the texels
      only inside the slot's course sheet block);
    - a free-run byte that is not free in `rs`;
    - a result whose slot does not decode to `c`'s replaced fields and record (with the model scenery its texture
      points among them: they read back).

    With the model scenery the set also gets course.LIBRARY_BOUND_PATCH (`course.patch_library_bound`: the two
    lower-bound words of the draw routines, so the lists in DATA_FREE are drawn)."""
    rs = _romset(rs)
    if not isinstance(c, golden.Course):
        raise ConvertError(f"{c!r:.60} is not a golden.Course")
    slot = _slot(c.course, rs)
    if not isinstance(c.entry, C.CourseEntry) or not is_int(c.entry.course) or c.entry.course != slot:
        raise ConvertError(f"converted course: entry {c.entry!r:.60} is not a course.CourseEntry of slot {slot}")
    fields = _fields(fields)
    name = f"{c.entry.name} (converted)"
    if missing := golden.unset_fields(c):
        race = [f for f in missing if f in fields]
        raise ConvertError(f"{name}: unset fields {', '.join(missing)}: encode_course would fill them from the set"
                           + (f", leaving {', '.join(race)} as the slot's own" if race else ""))
    try:
        ref = golden.decode_course(rs, slot)
    except DecodeError as e:
        raise ConvertError(f"{name}: the set's slot {slot}: {e}") from e
    scenery, model = SCENERY_FIELDS[0] in fields, MODEL_FIELDS[-1] in fields
    own = bool(slot) and model                          # a new id's own grid and sheet block
    if slot and scenery and not model:
        raise ConvertError(f"{name}: the empty scenery in slot {slot}: a new id keeps the slot's scenery or draws "
                           f"its own (MODEL_FIELDS)")
    if any(f in fields for f in ID_FIELDS) and not (own and all(f in fields for f in ID_FIELDS)):
        raise ConvertError(f"{name}: fields {', '.join(ID_FIELDS)} are a new id's own scenery's, replaced together "
                           f"with MODEL_FIELDS (slot {slot})")
    if own and not all(f in fields for f in ID_FIELDS):
        raise ConvertError(f"{name}: a new id's model scenery replaces {', '.join(ID_FIELDS)} too (its own cell grid "
                           f"and course sheet block: the slot's are Beginner's)")
    _check_against_slot(c, ref, fields, name)
    if model:
        _check_model(c, ref, name)
        try:
            if C.library_bounds(rs) not in (C.OBJECT_LIBRARY, C.PATCHED_LIBRARY):
                raise ConvertError(f"{name}: the set's object-library bounds {C.library_bounds(rs)} are neither the "
                                   f"original's nor course.PATCHED_LIBRARY")
        except DecodeError as e:
            raise ConvertError(f"{name}: {e}") from e
    elif scenery:
        _check_scenery(c, ref, name)
    try:
        golden.check_course(c)
        entries = golden.tables(c)
    except DecodeError as e:
        raise ConvertError(f"{name}: {e}") from e
    placed = set(DATA_ORDER + COPRO_ORDER + (SCENERY_PLACED if scenery else ())
                 + (tuple(t for t in ID_PLACED if own or t != "cell grid") if slot else ()))
    in_place = REWRITTEN_IN_PLACE | (MODEL_IN_PLACE if model else SCENERY_IN_PLACE if scenery else frozenset())
    if slot:
        in_place = (in_place | ID_IN_PLACE) - set(ID_PLACED)
    block = c.sheet_sources[0]
    for region, off, blob, table in entries:
        inside = _in_run(region, off, len(blob), slot)
        if table in placed and not inside:
            lo, size = _runs(slot).get(region, (0, 0))
            raise ConvertError(f"{name}: the {table} at {region}+{off:#x} ({len(blob)} bytes) is not inside the free "
                               f"run {lo:#x}-{lo + size:#x}: a converted table is written into free space, not over "
                               f"the slot's")
        if inside is None:
            raise ConvertError(f"{name}: the {table} at {region}+{off:#x} ({len(blob)} bytes) straddles the edge of "
                               f"the {region} free run")
        if not inside and getattr(rs, region)[off:off + len(blob)] != blob:
            if table not in in_place:
                raise ConvertError(f"{name}: the {table} at {region}+{off:#x} differs from the set's, and it is not a "
                                   f"table the conversion rewrites in place ({', '.join(sorted(in_place))})")
            if table.startswith("texels") and not (region == "data" and block <= off
                                                   and off + len(blob) <= block + T.SHEET_BLOCK):
                raise ConvertError(f"{name}: the {table} at {region}+{off:#x} differ from the set's outside the slot's "
                                   f"course sheet block data+{block:#x}: only that block is rewritten in place")
    try:
        patches = golden.encode_course(c)
        free = [(region, off, blob) for (region, off), blob in sorted(patches.items())
                if _in_run(region, off, len(blob), slot)]
        if own:
            free = _own_block(rs, free, block, ref.sheet_sources[0], name)
        shared = ("program", EMPTY_AT, EMPTY_TABLE)     # the shared table: reserved, written by empty_actors
        if scenery:
            if shared not in free:
                raise ConvertError(f"{name}: the actor table is not EMPTY_ACTORS' 256 zero words")
            free.remove(shared)
        out = empty_actors(rs) if scenery else rs   # the free runs first: the in-place words (COURSE_SHEETS[id]
        for region, off, blob in _merged(free):     # among them) then name what they hold, which write_free refuses
            out = _write_free(out, region, off, blob)
        out = out.with_patches({k: v for k, v in patches.items() if not _in_run(k[0], k[1], len(v), slot)})
        if slot:
            out = out.with_patches(_time_levels(c))
        if model:
            out = C.patch_library_bound(out)
        got = golden.decode_course(out, slot)
    except DecodeError as e:
        raise ConvertError(f"{name}: {e}") from e
    for f in fields + ("entry",):
        if not _same(getattr(got, f), getattr(c, f)):
            raise ConvertError(f"{name}: the written set decodes with a different {f}")
    return out


def _own_block(rs: RomSet, free: list, block: int, src: int, name: str) -> list:
    """`free` (region, offset, bytes) patches with those inside a new id's own 1 MB course sheet block (data `block`)
    replaced by the whole block: the slot's block (data `src`) with them written over it, so every texel the slot's
    block holds and the new course does not rewrite (the car models', the slot's other textures) reads as before."""
    image, out = bytearray(rs.data[src:src + T.SHEET_BLOCK]), []
    for region, off, blob in free:
        if region == "data" and block <= off < block + T.SHEET_BLOCK:
            if off + len(blob) > block + T.SHEET_BLOCK:
                raise ConvertError(f"{name}: data {off:#x}+{len(blob):#x} runs past the course sheet block "
                                   f"{block:#x}")
            image[off - block:off - block + len(blob)] = blob
        else:
            out.append((region, off, blob))
    return out + [("data", block, bytes(image))]


def _time_levels(c: golden.Course) -> dict[tuple[str, int], bytes]:
    """A new id's three TIME_TABLES level entries (its own entries of the grown arrays, data ROM) naming its own
    per-difficulty words: mode m's entry holds the address of (m, 0)'s word ("time levels", ID_IN_PLACE)."""
    out = {}
    for m in range(C.MODES):
        at = C.array_at(c.id_arrays, f"TIME_TABLES[{m}]") + 4 * c.course
        out[("data", at - DATA_BASE)] = struct.pack("<I", c.time_tables[(m, 0)][0])
    return out


def converted_set(rs: RomSet, disc: DcDisc, n: int, *, slot: int = 0, scenery: str = "keep",
                  layout: Layout | None = None, scenery_data: Scenery | None = None, variant: str = "",
                  road_copy: bool = False, thin: bool = True, depth: bool = False) -> RomSet:
    """`rs` with Dreamcast course `n` of `disc` (its CS_DATAn.BIN and 1ST_READ.BIN) converted into slot `slot`
    (`thin=False`, v0.3's far view: the model unthinned, THIN and REVERSE_THIN left out; the polygon cap is raised;
    `depth=True`, v0.3: the model's polygons ask the custom MAME for the per-pixel depth test, model.DEPTH_TEST):
    `encode(rs, converted_course(rs, d, exe, n, ...), fields)`, `fields` RACE_FIELDS (+ SCENERY_FIELDS for
    `scenery="empty"`, + MODEL_FIELDS for `scenery="model"`: the set then has course.LIBRARY_BOUND_PATCH, applied
    for the model scenery only). With `scenery="model"` and no `scenery_data` the scenery is the course's own model:
    `model_scenery(rs, d, CS{n}_POL.BIN, CS{n}_TEX.BIN, THIN.get(n), quad=QUAD.get(n), remesh=REMESH.get(n),
    props=NAMED_PROPS.get(n, ()), cell_cut=CELL_CUT.get(n))` from the disc; a course LIFT names is converted lifted (its data and model by
    `lifted_data` and `lifted_model`, Phase 10 Task 4), a course SHIFT names shifted in x (`shifted_data` and
    `shifted_model`, Phase 11 Task 3; after the lift) (`rs` should be the original
    set, never an earlier conversion's output; for a new id the grown, guarded set, `model_scenery(..., slot=slot)`
    and fields + ID_FIELDS). `variant="R"` (M9) converts the course's reverse into its reverse id n + 8 (`_variant`):
    its CSDATnR.BIN, CSnR_POL.BIN and CSnR_TEX.BIN, its R checkpoints and grid base, the same LIFT, SHIFT and budget
    settings as the plain course. ConvertError as those and for a `disc` that is not a DcDisc; DcError from the
    disc's readers passes through.
    `road_copy` (v0.2; release.release_set passes True with `direction`): a `scenery="model"` conversion copies its road
    across cell edges (model_scenery's `road_copy`, build.ROAD_COPY's setting for course n); False (the default)
    copies nothing, as every set before v0.2 was built."""
    rs = _romset(rs)
    n = _course_n(n)
    slot = _slot(slot, rs)
    variant = _variant(variant, n, slot, scenery)
    if not isinstance(disc, DcDisc):
        raise ConvertError(f"{type(disc).__name__} is not a DcDisc")
    if variant == "R" and n < 3:
        rs = rs.with_patches(_reverse_alias(rs, n, slot))
    data_name = course_files(n, variant, kind="data")
    d = tables.with_lane_ranges_fixed(decode_data(disc.read(data_name), data_name))     # 1R only (none elsewhere)
    plain = None                    # a reverse's plain course data, lifted and shifted as d (its ground's lender)
    if variant == "R":
        plain_name = course_files(n, kind="data")
        plain = tables.with_lane_ranges_fixed(decode_data(disc.read(plain_name), plain_name))
    if n in LIFT:
        d = lifted_data(d, LIFT[n])
        plain = plain and lifted_data(plain, LIFT[n])
    if n in SHIFT:
        d = shifted_data(d, SHIFT[n])
        plain = plain and shifted_data(plain, SHIFT[n])
    exe = disc.read(EXE_NAME)
    if scenery == "model" and scenery_data is None:
        thin_ = thin
        thin, quad, remesh, props = ((THIN.get(n), QUAD.get(n), REMESH.get(n), NAMED_PROPS.get(n, ())) if variant == ""
                                     else (REVERSE_THIN.get(n), REVERSE_QUAD.get(n), REVERSE_REMESH.get(n),
                                           REVERSE_NAMED_PROPS.get(n, ())))
        thin = thin if thin_ else None
        mn, tn = course_files(n, variant, kind="model"), course_files(n, variant, kind="textures")
        m = pol.decode_pol(disc.read(mn), mn)
        if n in LIFT:
            m = lifted_model(m, LIFT[n])
        if n in SHIFT:
            m = shifted_model(m, SHIFT[n])
        scenery_data = model_scenery(rs, d, m, disc.read(tn), thin, slot=slot, quad=quad, remesh=remesh, props=props,
                                     cell_cut=CELL_CUT.get(n), road_copy=ROAD_COPY.get(n) if road_copy else None,
                                     depth=depth)
    c = converted_course(rs, d, exe, n, slot=slot, scenery=scenery, layout=layout, scenery_data=scenery_data,
                         variant=variant, plain=plain)
    if variant == "R" and n < 3:                    # the arcade's segment-keyed tables on the reverse road
        offset = reverse.point_offset(plain, d)
        flags = reverse.reversed_flags(golden.decode_course(rs, n).segment_flags[1], offset)
        c = dataclasses.replace(c, segment_flags=(c.segment_flags[0], flags))
        out = encode(rs, c, RACE_FIELDS)
        tunnel = any(f & C.SEGMENT_TUNNEL for f in flags) or any(st.flags & C.SEGMENT_TUNNEL for st in c.steps)
        return _reverse_tables(out, n, slot, offset, len(flags), tunnel)
    if scenery == "model":
        return encode(rs, c, RACE_FIELDS + MODEL_FIELDS + (ID_FIELDS if slot else ()))
    return encode(rs, c, RACE_FIELDS + (SCENERY_FIELDS if scenery == "empty" else ()))


LONG_TIME = 12000   # ticks: a start time past a 180 s run (the playtest set's and the long-time MAME runs')


def long_time(s: RomSet, c: golden.Course, ticks: int = LONG_TIME) -> RomSet:
    """`s` with the first word (the start time) of every time table of `c` set to `ticks`, so a human can drive a
    lap: a converted course keeps Beginner's start time (3200 ticks, 50 s to the first checkpoint). The release set's
    real times are `race_times` (v0.1.2); this stays for the MAME tests' long-time runs."""
    s = _romset(s)
    if not isinstance(c, golden.Course):
        raise ConvertError(f"{type(c).__name__} is not a golden.Course")
    if not is_int(ticks) or not 0 < ticks < 1 << 32:
        raise ConvertError(f"start time {ticks!r} is not a positive u32")
    return s.with_patches({("data", t - DATA_BASE): struct.pack("<I", ticks) for _, t, _ in c.time_tables.values()})


RACE_TIMES: dict[int, dict[tuple[int, int], tuple[int, tuple[int, ...]]]] = {
    3: {
        (0, 0): (7588, (856, 1188, 804)),
        (0, 1): (6897, (778, 1080, 731)),
        (0, 2): (6992, (789, 1095, 741)),
        (0, 3): (7045, (795, 1103, 747)),
        (1, 0): (31144, (610, 610, 610)),
        (1, 1): (30924, (606, 606, 606)),
        (1, 2): (32364, (634, 634, 634)),
        (1, 3): (33564, (658, 658, 658)),
        (2, 0): (31645, (1012, 1012, 1012)),
        (2, 1): (30547, (977, 977, 977)),
        (2, 2): (30619, (979, 979, 979)),
        (2, 3): (30543, (977, 977, 977)),
    },
    4: {
        (0, 0): (6436, (753, 929, 616)),
        (0, 1): (5850, (684, 844, 560)),
        (0, 2): (5930, (693, 856, 568)),
        (0, 3): (5976, (699, 862, 572)),
        (1, 0): (25870, (491, 491, 491)),
        (1, 1): (25687, (488, 488, 488)),
        (1, 2): (26883, (511, 511, 511)),
        (1, 3): (27880, (530, 530, 530)),
        (2, 0): (28227, (816, 816, 816)),
        (2, 1): (27248, (788, 788, 788)),
        (2, 2): (27312, (790, 790, 790)),
        (2, 3): (27245, (788, 788, 788)),
    },
    5: {
        (0, 0): (4634, (1178,)),
        (0, 1): (4212, (1071,)),
        (0, 2): (4270, (1086,)),
        (0, 3): (4302, (1094,)),
        (1, 0): (15424, (735,)),
        (1, 1): (15315, (730,)),
        (1, 2): (16028, (764,)),
        (1, 3): (16622, (792,)),
        (2, 0): (15855, (1271,)),
        (2, 1): (15305, (1227,)),
        (2, 2): (15341, (1230,)),
        (2, 3): (15303, (1227,)),
    },
    6: {
        (0, 0): (6416, (1050, 1087)),
        (0, 1): (5832, (954, 988)),
        (0, 2): (5912, (967, 1002)),
        (0, 3): (5957, (974, 1009)),
        (1, 0): (23973, (686, 686)),
        (1, 1): (23803, (681, 681)),
        (1, 2): (24911, (713, 713)),
        (1, 3): (25835, (739, 739)),
        (2, 0): (25020, (1140, 1140)),
        (2, 1): (24152, (1100, 1100)),
        (2, 2): (24209, (1103, 1103)),
        (2, 3): (24149, (1100, 1100)),
    },
    7: {
        (0, 0): (6430, (1105, 676, 606)),
        (0, 1): (5844, (1004, 614, 551)),
        (0, 2): (5924, (1018, 622, 559)),
        (0, 3): (5969, (1026, 627, 563)),
        (1, 0): (26337, (512, 512, 512)),
        (1, 1): (26151, (508, 508, 508)),
        (1, 2): (27369, (532, 532, 532)),
        (1, 3): (28383, (551, 551, 551)),
        (2, 0): (27351, (848, 848, 848)),
        (2, 1): (26402, (819, 819, 819)),
        (2, 2): (26464, (821, 821, 821)),
        (2, 3): (26399, (819, 819, 819)),
    },
    8: {
        (0, 0): (3315, (955,)),
        (0, 1): (3013, (868,)),
        (0, 2): (3054, (880,)),
        (0, 3): (3078, (887,)),
        (1, 0): (12082, (596,)),
        (1, 1): (11997, (592,)),
        (1, 2): (12556, (620,)),
        (1, 3): (13021, (643,)),
        (2, 0): (12452, (1030,)),
        (2, 1): (12020, (994,)),
        (2, 2): (12048, (996,)),
        (2, 3): (12018, (994,)),
    },
    9: {
        (0, 0): (5360, (1161, 907, 902, 670, 850, 1756)),
        (0, 1): (4872, (1055, 824, 820, 609, 773, 1596)),
        (0, 2): (4939, (1069, 835, 831, 617, 784, 1618)),
        (0, 3): (4977, (1078, 842, 838, 622, 790, 1630)),
        (1, 0): (17419, (669, 669, 669, 669, 669, 669)),
        (1, 1): (17296, (664, 664, 664, 664, 664, 664)),
        (1, 2): (18101, (695, 695, 695, 695, 695, 695)),
        (1, 3): (18772, (721, 721, 721, 721, 721, 721)),
        (2, 0): (18325, (1111, 1111, 1111, 1111, 1111, 1111)),
        (2, 1): (17689, (1072, 1072, 1072, 1072, 1072, 1072)),
        (2, 2): (17731, (1075, 1075, 1075, 1075, 1075, 1075)),
        (2, 3): (17687, (1072, 1072, 1072, 1072, 1072, 1072)),
    },
    10: {
        (0, 0): (4665, (678, 769, 1314)),
        (0, 1): (4240, (616, 699, 1194)),
        (0, 2): (4298, (624, 709, 1210)),
        (0, 3): (4331, (629, 714, 1220)),
        (1, 0): (15548, (591, 591, 591)),
        (1, 1): (15438, (587, 587, 587)),
        (1, 2): (16157, (614, 614, 614)),
        (1, 3): (16756, (637, 637, 637)),
        (2, 0): (16441, (981, 981, 981)),
        (2, 1): (15871, (947, 947, 947)),
        (2, 2): (15908, (949, 949, 949)),
        (2, 3): (15869, (947, 947, 947)),
    },
    11: {
        (0, 0): (8209, (863, 837, 1203)),
        (0, 1): (7461, (784, 761, 1093)),
        (0, 2): (7563, (795, 771, 1108)),
        (0, 3): (7621, (801, 777, 1116)),
        (1, 0): (32105, (621, 621, 621)),
        (1, 1): (31878, (617, 617, 617)),
        (1, 2): (33362, (646, 646, 646)),
        (1, 3): (34599, (670, 670, 670)),
        (2, 0): (33783, (1032, 1032, 1032)),
        (2, 1): (32611, (996, 996, 996)),
        (2, 2): (32688, (998, 998, 998)),
        (2, 3): (32607, (996, 996, 996)),
    },
    12: {
        (0, 0): (6698, (736, 643, 926)),
        (0, 1): (6088, (669, 584, 842)),
        (0, 2): (6172, (678, 592, 854)),
        (0, 3): (6219, (683, 597, 860)),
        (1, 0): (25875, (493, 493, 493)),
        (1, 1): (25692, (490, 490, 490)),
        (1, 2): (26888, (513, 513, 513)),
        (1, 3): (27885, (532, 532, 532)),
        (2, 0): (28178, (819, 819, 819)),
        (2, 1): (27201, (791, 791, 791)),
        (2, 2): (27265, (793, 793, 793)),
        (2, 3): (27198, (791, 791, 791)),
    },
    13: {
        (0, 0): (4000, (1122,)),
        (0, 1): (3636, (1020,)),
        (0, 2): (3686, (1034,)),
        (0, 3): (3714, (1042,)),
        (1, 0): (14302, (701,)),
        (1, 1): (14201, (696,)),
        (1, 2): (14862, (728,)),
        (1, 3): (15413, (755,)),
        (2, 0): (14729, (1211,)),
        (2, 1): (14218, (1169,)),
        (2, 2): (14251, (1172,)),
        (2, 3): (14216, (1169,)),
    },
    14: {
        (0, 0): (6154, (1061, 982)),
        (0, 1): (5593, (964, 893)),
        (0, 2): (5670, (977, 905)),
        (0, 3): (5713, (985, 912)),
        (1, 0): (23189, (657, 657)),
        (1, 1): (23025, (652, 652)),
        (1, 2): (24097, (682, 682)),
        (1, 3): (24990, (708, 708)),
        (2, 0): (25372, (1090, 1090)),
        (2, 1): (24492, (1052, 1052)),
        (2, 2): (24550, (1054, 1054)),
        (2, 3): (24489, (1052, 1052)),
    },
    15: {
        (0, 0): (6420, (1095, 558, 725)),
        (0, 1): (5835, (995, 507, 659)),
        (0, 2): (5915, (1009, 514, 668)),
        (0, 3): (5960, (1016, 518, 673)),
        (1, 0): (26200, (509, 509, 509)),
        (1, 1): (26015, (505, 505, 505)),
        (1, 2): (27226, (529, 529, 529)),
        (1, 3): (28236, (548, 548, 548)),
        (2, 0): (27091, (845, 845, 845)),
        (2, 1): (26151, (816, 816, 816)),
        (2, 2): (26212, (818, 818, 818)),
        (2, 3): (26148, (816, 816, 816)),
    },
}
"""The converted courses' own time tables (v0.1.2, docs/superpowers/specs/2026-11-15-race-times-design.md): id 3-7 ->
(mode 0-2, difficulty 0-3) -> (start, one lap's pass words: the checkpoints in order, then the lap line); the table is
the start then that lap repeated for every lap (the phantom last word included), `race_table`. Frozen from
`racetimes.derive` on the AI field's measured sector times (tests/data/race_times_measured.json, spike/race_times.py,
docs/spike/2026-11-race-times.md); tests/test_release.py test_race_times_frozen recomputes them. Ids 8-15 (M9 stage
4): the reverses, measured from the Course Direction screen and derived by the same rule (racetimes.REVERSE). The start makes the
whole race as generous as the arcade's own courses against their AI field (racetimes module docstring)."""


def race_table(start: int, lap: tuple[int, ...], n_words: int) -> tuple[int, ...]:
    """The time table of `n_words` words: `start`, then `lap` repeated (n_words - 1 must be a whole number of laps).
    ConvertError otherwise."""
    if not lap or (n_words - 1) % len(lap) or n_words < 1 + len(lap):
        raise ConvertError(f"a table of {n_words} words is not a start plus whole laps of {len(lap)} passes")
    return (start,) + tuple(lap) * ((n_words - 1) // len(lap))


def race_times(s: RomSet, c: golden.Course, tables: dict[tuple[int, int], tuple[int, ...]] | None = None) -> RomSet:
    """`s` with every time table of `c` rewritten in place with `tables` (default RACE_TIMES[c.course], each expanded
    by `race_table` to the length of the course's table), word for word:
    the same table addresses and lengths, no repointing. ConvertError for a RomSet or Course of the wrong kind, a
    (mode, difficulty) of `c` that `tables` lacks or has extra, a table whose length differs from the course's (the race
    reads 1 + laps x (checkpoints + 1) words), a word that is not a positive u32, or two tables that overlap without
    being the same table with the same words (Sega's own tables overlap across difficulties: a course converted into an
    original's slot without `slot=` keeps them, a new id's own tables are disjoint)."""
    s = _romset(s)
    if not isinstance(c, golden.Course):
        raise ConvertError(f"{type(c).__name__} is not a golden.Course")
    if tables is None:
        if c.course not in RACE_TIMES:
            raise ConvertError(f"course {c.course}: no race times (RACE_TIMES holds {sorted(RACE_TIMES)})")
        tables = {key: race_table(*v, len(c.time_tables[key][2])) if key in c.time_tables else v
                  for key, v in RACE_TIMES[c.course].items()}
    if not isinstance(tables, dict) or set(tables) != set(c.time_tables):
        got = sorted(tables) if isinstance(tables, dict) else tables
        raise ConvertError(f"course {c.course}: race times for {got!r:.80}, the course has {sorted(c.time_tables)}")
    patches: dict[tuple[str, int], bytes] = {}
    for key, (_, t, words) in sorted(c.time_tables.items()):
        new = tables[key]
        if not isinstance(new, (tuple, list)) or len(new) != len(words):
            got = len(new) if isinstance(new, (tuple, list)) else new
            raise ConvertError(f"course {c.course} {key}: {got!r:.40} words, the table holds {len(words)}")
        if not all(is_int(w) and 0 < w < 1 << 32 for w in new):
            raise ConvertError(f"course {c.course} {key}: a word of {tuple(new)!r:.80} is not a positive u32")
        blob = struct.pack(f"<{len(new)}I", *new)
        at = ("data", t - DATA_BASE)
        for (_, o), b in patches.items():
            if (o, b) != (at[1], blob) and o < at[1] + len(blob) and at[1] < o + len(b):
                raise ConvertError(f"course {c.course} {key}: the table at {t:#x} overlaps the one at "
                                   f"{o + DATA_BASE:#x} with different words")
        patches[at] = blob
    return s.with_patches(patches)


# ---- the pit stalls (v0.1.2) ----

PIT_IDS = (3, 4, 5, 6, 7)
"""The course ids pit_stalls writes: the converted courses (the originals keep the arcade's own stalls)."""
REVERSE_OFFSET = 8
"""M9: course n's reverse is at id n + REVERSE_OFFSET (release.REVERSE_IDS, dirscreen.REVERSE_OFFSET)."""
REVERSE_PIT_COURSES = tuple(range(8))
"""M9: the courses whose reverse (id n + REVERSE_OFFSET) pit_stalls(..., "R") gives the disc's R stalls: all eight."""


def pit_stalls(s: RomSet, disc: DcDisc, n: int, variant: str = "") -> RomSet:
    """`s` (a grown set) with course id `n` (one of PIT_IDS, converted at its own id) given Dreamcast course n's pit
    stalls (dc.exe.pit_stalls of the disc's 1ST_READ.BIN, the plain group; each x moved by SHIFT[n] as the course is,
    single precision): course.encode_pit_stalls at the last free bytes of ID_DATA_FREE (`last_fit`, so the first fits
    after it, the views', place as before), its CAR_TABLES_230E50 entry repointed there. Before v0.1.2 every new id
    read Beginner's stalls (grow_ids' fill), which lie behind the car on the box of a converted course, so the stop
    skipped all seven waypoints in 7 frames and the car drove on (docs/follow-ups/v0.1.2.md, "Pit stops").
    M9 stage 4: `variant="R"` gives course n's reverse, id n + 8 (n 0-7, REVERSE_PIT_COURSES), the R group (x moved
    by SHIFT[n] as the plain; the arcade's three have none: the disc's plain stalls of 0-2 are the arcade's own).
    ConvertError for `s` not a RomSet, `disc` not a DcDisc, an `n` outside PIT_IDS (REVERSE_PIT_COURSES for "R"), a
    variant other than "" and "R", or a set whose CAR_TABLES_230E50 is not the grown table; DcError and CourseError
    from the readers and the encoder pass through."""
    s = _romset(s)
    if not isinstance(disc, DcDisc):
        raise ConvertError(f"{type(disc).__name__} is not a DcDisc")
    if variant not in ("", "R"):
        raise ConvertError(f"pit stalls: variant {variant!r} is not '' or 'R'")
    ids = REVERSE_PIT_COURSES if variant else PIT_IDS
    if not is_int(n) or isinstance(n, bool) or n not in ids:
        raise ConvertError(f"course {n!r}: {'reverse ' if variant else ''}pit stalls are written for {ids}")
    slot = n + (REVERSE_OFFSET if variant else 0)
    table = C.id_table(s, "CAR_TABLES_230E50")
    if table != C.ID_ARRAYS["CAR_TABLES_230E50"].new:
        raise ConvertError(f"course {n}: CAR_TABLES_230E50 is read at {table:#x}, not the grown table (grow_ids)")
    move = _offset(SHIFT.get(n, 0.0), "x", "shifted")
    stalls = tuple(tuple(C.PitWaypoint(move(w.x), w.z, w.speed) for w in st)
                   for st in dc_exe.pit_stalls(disc.read(EXE_NAME), n, variant))
    at = last_fit(s, "data", ID_DATA_FREE, C.PIT_STALLS_SIZE, f"course id {slot}'s pit stalls")
    return s.with_patches({("data", at): C.encode_pit_stalls(stalls, DATA_BASE + at),
                           ("data", table + 4 * slot - DATA_BASE): struct.pack("<I", DATA_BASE + at)})


# ---- the Dreamcast's own mirror as a check (Phase 9 Task 4) ----

MIRROR_SHIFT = -1.0
"""The Dreamcast's `M` files mirror in its plane z = 0, the arcade's mirror in z = -0.5 (mirror.PLANE). In the
arcade's frame (z negated: `tables`, `ground`) the `M` file's point has the plain point's z unnegated, the arcade's
mirror -1 - z: the `M` file converted, its z moved by MIRROR_SHIFT, is the arcade's mirror (docs/formats/
dreamcast-disc.md, "Mirror files")."""
MIRROR_TOLERANCES = {"road": 0.004, "point": 2.0 ** -14, "centre": 0.35, "heading": 2, "lane": 134}
"""`mirror_check`'s bounds, set before the check is run (a distance is the largest of |dx|, |dy|, |dz|). Where Phase
5 gives a number for the Dreamcast's `M` files (docs/spike/2026-11-dc-decode.md, "Mirror files"; docs/formats/
dreamcast-disc.md, "Mirror files"), it is that number, as Phase 5 states it (MIRROR_TOLERANCE_SOURCES):

- road: a used road slot's position. Phase 5, course 3's road lanes: 0 of 2,790 exact, offset 0.004;
- centre: a ground cell's centre. Phase 5: course 3's cells "near (the road within 0.35) but not exact";
- heading: road and AI headings, modulo 0x4000 (the yaw is 4h). Phase 5, courses 3-7: "within 2 of 0x2000 - h";
- lane: an AI lane (8.8). Phase 5, courses 3-7: "near 0x6ff - lane (within 60-134)", the range's top.

Where Phase 5 gives none, the bound is derived, not fitted:

- point: an AI line point and a ground corner. Phase 5 has the line points exact (3,720 of 3,720 on course 3) and
  gives no number for the corners as a set (its cells are compared in one corner order: 763 of 3,073 exact); that
  every corner is exact as a set is this task's measurement (Phase 9 Task 4). What is left of an exact point is the
  arcade's -1 - z rounded to f32 (mirror.mirror_world): half an f32 unit, at most 2^-15 for |z| < 1024; the bound is
  one unit, 2^-14.

`mirror_check` reports what lies outside a bound ("outside") rather than widening it."""
MIRROR_TOLERANCE_SOURCES = {"road": "Phase 5", "centre": "Phase 5", "heading": "Phase 5", "lane": "Phase 5",
                            "point": "derived: one f32 unit (the corners' exactness measured in Phase 9 Task 4)"}
"""Where each of MIRROR_TOLERANCES comes from: Phase 5's documented numbers (a priori), or derived."""


def _mirror_tables(d: CourseData) -> tuple[tuple, tuple, tgp.Ground]:
    """(road line, AI lines, ground) of the arcade's mirror of course data `d` as converted: mirror.mirror_course's
    rules on `tables.road_line`, `tables.ai_lines` and `ground.convert_ground` (placed at the new ids' runs:
    mirror.mirror_ground wants a placed ground; the polygons, grid values and lists do not depend on where). What a
    set's mirror of the converted course holds (test_convert_build.test_mirror_check_desert_city; for a course LIFT
    names, `d` lifted, as converted_set converts it: tests/test_mirror.py test_mirror_check_4_and_7)."""
    from dtools.arcade import mirror as M
    try:
        road = tuple(M.mirror_road_point(p) for p in tables.road_line(d))
        lines = tuple(tuple(M.mirror_ai_point(p) for p in line) for line in tables.ai_lines(d))
        g = tgp.placed_ground(ground.convert_ground(d, NEW_ID), NEW_ID_RUNS["copro_data"][0] // 4,
                              NEW_ID_RUNS["data"][0])
        return road, lines, M.mirror_ground(g)
    except M.MirrorError as e:
        raise ConvertError(f"the arcade's mirror: {e}") from e


def _shifted(p) -> np.ndarray:
    """Points (n, 3) as float64 with z moved by MIRROR_SHIFT."""
    a = np.array(p, dtype=np.float64).reshape(-1, 3)
    a[:, 2] += MIRROR_SHIFT
    return a


def _turn_error(a: int, b: int, period: int = 0x4000) -> int:
    """|a - b| modulo `period`, the nearer way round."""
    return abs((a - b + period // 2) % period - period // 2)


def mirror_check(disc: DcDisc, n: int = 3) -> dict:
    """The arcade's mirror of Dreamcast course `n` as converted (`_mirror_tables`: mirror.mirror_course's rules on the
    converter's tables, what id `n`'s mirror holds on a set) against the Dreamcast's own mirror, `CSDATnM.BIN`
    converted the same way (`tables`, `ground.convert_ground`) and moved by MIRROR_SHIFT in z. Per table, the pairs
    compared, how many are within MIRROR_TOLERANCES and the largest difference (a distance is the largest of |dx|,
    |dy|, |dz|):

    - "lift": LIFT[n] (0.0 for a course not lifted): a lifted course (Mermaid Lake, n 7) is compared lifted, the plain
      and the `M` file alike (`lifted_data`), as the set's course and its mirror hold it (the mirror keeps every
      height, so it is the arcade's mirror of the lifted course);
    - "plane": the measured offset, the median of (arcade z - `M` z) over the AI points to 0.001 (MIRROR_SHIFT);
    - "road": every used lane slot (slot for slot: both mirrors reverse the slots), "count", "within" (road), "max",
      "outside" ((road point, slot) of each slot past the road bound);
      "lane_ranges" (equal, total); "headings_within" (heading) and "headings_max" (modulo 0x4000);
    - "ai_lines": "order" (the `M` line paired with each arcade line: the nearest by its largest distance), "count",
      "within" (point), "max", "outside" ((arcade line, point) of each point past the point bound), "steps"
      (equal), "lanes_within" (lane), "lanes_max", "headings_within" (heading), "headings_max";
    - "ground": per polygon (index for index) "corners_within" and "corners_max" (the corners as a set: the orders
      differ, the arcade's (0, 3, 2, 1) of the plain cell, the Dreamcast's (1, 0, 3, 2)), "centres_within" (centre)
      and "centres_max", "centres_outside" (the polygon indices past the bound), "directions_equal" and
      "directions_max" (modulo 0x10000);
    - "grid": the arcade's mirrored grid against the grid the arcade's rule builds from the `M` polygons moved
      (`ground`), square by square as sets of polygons: "count", "equal", "differ" ((row, column) of each square that
      differs);
    - "steps", "attributes": (equal, total); "checkpoints": the arcade's segments (the plain list's: the mirror keeps
      them) and the Dreamcast's `M` list;
    - "differs": the names of what is not equal (within "point" for positions, exactly for the rest), in or out of
      tolerance; "outside": the names of what lies past its bound (a finding, not a failure of the check).

    ConvertError for a `disc` that is not a DcDisc, an `n` that is not an int 0-7 (bool refused), tables the
    converter or the mirror refuse, or an empty table; DcError from the disc's readers passes through."""
    if not isinstance(disc, DcDisc):
        raise ConvertError(f"{type(disc).__name__} is not a DcDisc")
    n = _course_n(n)
    from dtools.dc import exe as X

    def data(v: str) -> CourseData:
        name = course_files(n, v, kind="data")
        return decode_data(disc.read(name), name)
    plain, dm = data(""), data("M")
    if n in LIFT:                       # like with like: the set's course and its mirror are lifted (converted_set)
        plain, dm = lifted_data(plain, LIFT[n]), lifted_data(dm, LIFT[n])
    road, lines, g = _mirror_tables(plain)
    m_road, m_lines, m_ground = tables.road_line(dm), tables.ai_lines(dm), ground.convert_ground(dm, NEW_ID)
    tol = MIRROR_TOLERANCES
    if len(m_road) != len(road) or len(m_ground.polys) != len(g.polys) or len(m_lines) != len(lines):
        raise ConvertError(f"course {n}: the M file's tables ({len(m_road)} road points, {len(m_lines)} lines, "
                           f"{len(m_ground.polys)} cells) are not the plain file's ({len(road)}, {len(lines)}, "
                           f"{len(g.polys)})")
    if not road or not lines or not g.polys or not all(lines):
        raise ConvertError(f"course {n}: an empty table (road {len(road)}, lines {len(lines)}, cells {len(g.polys)})")
    out: dict = {"lift": float(LIFT.get(n, 0.0))}
    differs: list[str] = []
    outside: list[str] = []

    def note(name: str, differ, beyond=False) -> None:
        """`name` into differs when it is not equal, into outside when past its bound (numpy or Python truths)."""
        if differ:
            differs.append(name)
        if beyond:
            outside.append(name)

    def count(mask) -> int:
        return int(np.count_nonzero(mask))

    # the road: every used slot
    slots = [(i, k) for i, b in enumerate(m_road) for k in range(b.lane_lo, b.lane_hi + 1)]
    pairs = [(road[i].lanes[k], m_road[i].lanes[k]) for i, k in slots]
    if not pairs:
        raise ConvertError(f"course {n}: no used road slot")
    d = np.abs(np.array([p.position for p, _ in pairs]) - _shifted([q.position for _, q in pairs])).max(axis=1)
    heads = np.array([_turn_error(p.heading, q.heading) for p, q in pairs])
    ranges = count([(a.lane_lo, a.lane_hi) == (b.lane_lo, b.lane_hi) for a, b in zip(road, m_road)])
    out["road"] = {"count": len(pairs), "within": count(d <= tol["road"]), "max": float(d.max()),
                   "outside": tuple(slots[j] for j in np.flatnonzero(d > tol["road"])),
                   "lane_ranges": (ranges, len(road)), "headings_within": count(heads <= tol["heading"]),
                   "headings_max": int(heads.max())}
    note("road positions", (d > tol["point"]).any(), (d > tol["road"]).any())
    note("road lane ranges", ranges < len(road))
    note("road headings", heads.any(), (heads > tol["heading"]).any())
    # the AI lines: each paired with the M line nearest to it
    pa = [np.array([p.position for p in line], dtype=np.float64) for line in lines]
    pm = [_shifted([q.position for q in line]) for line in m_lines]
    order = tuple(min(range(len(pm)), key=lambda j: (np.abs(pa[k] - pm[j]).max() if len(pm[j]) == len(pa[k])
                                                     else math.inf)) for k in range(len(pa)))
    if any(len(pm[j]) != len(pa[k]) for k, j in enumerate(order)):
        raise ConvertError(f"course {n}: the M file's lines do not pair with the plain file's (lengths)")
    pairs = [(p, q) for k, j in enumerate(order) for p, q in zip(lines[k], m_lines[j])]
    where = [(k, i) for k in range(len(order)) for i in range(len(lines[k]))]
    d = np.concatenate([np.abs(pa[k] - pm[j]).max(axis=1) for k, j in enumerate(order)])
    plane = np.median(np.concatenate([pa[k][:, 2] - (pm[j][:, 2] - MIRROR_SHIFT) for k, j in enumerate(order)]))
    out["plane"] = round(float(plane), 3) + 0.0
    lanes = np.array([abs(p.lane - q.lane) for p, q in pairs])
    heads = np.array([_turn_error(p.heading, q.heading) for p, q in pairs])
    steps = count([p.step == q.step for p, q in pairs])
    out["ai_lines"] = {"order": order, "count": len(pairs), "within": count(d <= tol["point"]),
                       "max": float(d.max()), "outside": tuple(where[j] for j in np.flatnonzero(d > tol["point"])),
                       "steps": steps, "lanes_within": count(lanes <= tol["lane"]),
                       "lanes_max": int(lanes.max()), "headings_within": count(heads <= tol["heading"]),
                       "headings_max": int(heads.max())}
    note("AI positions", (d > tol["point"]).any(), (d > tol["point"]).any())
    note("AI steps", steps < len(pairs), steps < len(pairs))
    note("AI lanes", lanes.any(), (lanes > tol["lane"]).any())
    note("AI headings", heads.any(), (heads > tol["heading"]).any())
    note("AI line order", order != tuple(range(len(order))))
    # the ground, polygon for polygon
    corners, centres, dirs = [], [], []
    for a, b in zip(g.polys, m_ground.polys):
        x, y = np.array(a.corners, dtype=np.float64), _shifted(b.corners)
        gap = np.abs(x[:, None, :] - y[None, :, :]).max(axis=2)
        corners.append(max(gap.min(axis=1).max(), gap.min(axis=0).max()))
        centres.append(float(np.abs(np.array(a.centre, dtype=np.float64) - _shifted(b.centre)[0]).max()))
        dirs.append(_turn_error(a.direction, b.direction, 0x10000))
    corners, centres, dirs = np.array(corners), np.array(centres), np.array(dirs)
    out["ground"] = {"count": len(g.polys), "corners_within": count(corners <= tol["point"]),
                     "corners_max": float(corners.max()), "centres_within": count(centres <= tol["centre"]),
                     "centres_max": float(centres.max()),
                     "centres_outside": tuple(int(i) for i in np.flatnonzero(centres > tol["centre"])),
                     "directions_equal": count(dirs == 0), "directions_max": int(dirs.max())}
    note("ground corners", (corners > tol["point"]).any(), (corners > tol["point"]).any())
    note("ground centres", (centres > tol["point"]).any(), (centres > tol["centre"]).any())
    note("ground directions", dirs.any())
    # the grid: the arcade's mirrored one against the arcade's rule on the M polygons moved
    moved = [dataclasses.replace(p, corners=tuple(tuple(map(float, c)) for c in _shifted(p.corners))) for p in
             m_ground.polys]
    m_grid, m_lists, _ = ground._grid(moved)

    def square(grid, lists, k: int) -> frozenset:
        return frozenset(lists[grid[k]]) if grid[k] else frozenset()
    bad = tuple((k >> 7, k & 127) for k in range(tgp.GRID_CELLS)
                if square(g.grid, g.lists, k) != square(m_grid, m_lists, k))
    out["grid"] = {"count": tgp.GRID_CELLS, "equal": tgp.GRID_CELLS - len(bad), "differ": bad}
    note("grid squares", len(bad) > 0)
    # steps, attributes, checkpoints
    _, s, _, ms = *tables.steps(plain), *tables.steps(dm)
    out["steps"] = (count([x == y for x, y in zip(s, ms)]), len(s))
    a, b = tables.attributes(plain), tables.attributes(dm)
    out["attributes"] = (count([x == y for x, y in zip(a, b)]), len(a))
    exe = disc.read(EXE_NAME)
    out["checkpoints"] = {"arcade": tuple(r.segment for r in tables.checkpoint_rows(exe, n)),
                          "dreamcast": tuple(X.checkpoints(exe, n, "M"))}
    note("steps", out["steps"][0] < len(s) or len(s) != len(ms))
    note("attributes", out["attributes"][0] < len(a) or len(a) != len(b))
    note("checkpoints", out["checkpoints"]["arcade"] != out["checkpoints"]["dreamcast"])
    out["differs"], out["outside"] = tuple(differs), tuple(outside)
    return out


# ---- the Dreamcast's cars (v0.4; docs/spike/2026-11-cars.md) ----

CAR_TABLE = 0x1FFFE00
CAR_TABLE_SIZE = 0x200
"""The cars' table (data-ROM offset; i960 DATA_BASE + CAR_TABLE), reserved (romset.RESERVED): CAR_COUNT rows of four
u32, one per car choice (0 the arcade's Hornet: zeros; 1-4 the Dreamcast's Grasshopper, Falcon, Lightning, Unicorn):
the body list, the cockpit list, the address of its wheel-list table and of its hubs. Then at CAR_WHEELS a wheel-list
table per car (four list pointers, in the arcade's order: front x < 0, front x > 0, rear x < 0, rear x > 0, as its own
table at window 0x230486) and at CAR_HUBS the hubs per car (four x, y, z f32, the same order, as the arcade's at
window 0x230d54). The car hook (course.car_choice) reads the rows."""
CAR_COUNT = 5
CAR_HANDLING = 0x1FFFD00
"""The cars' handling rows (data-ROM offset, reserved; course.CAR_HANDLING_ADDR): per choice four gear ratios and
the four times the car's torque factor, f32. Row 0 is the arcade's own ratios twice (data `0x86f780`); a Dreamcast
car's are its own (`dc.car.handling`: the Dreamcast's Hornet row equals the arcade's ratios, so its cars' values are
already relative to the arcade's Hornet)."""
GEAR_RATIOS_AT = 0x86F780
CAR_WHEELS = 0x50
CAR_HUBS = 0xA0
CAR_TEXTURES = CUSTOM_FREE["textures"][1]
"""The run the cars' texels, texture headers and texture points take: the textures region's second custom area."""


FIELD_TABLE = 0x1FFFA00
FIELD_ROWS = 44
"""The Dreamcast field's lists (data-ROM offset, reserved; course.FIELD_TABLE_ADDR): per car index 1..FIELD_ROWS-1
four list pointers, near, mid, far, far. Car n drives car `(n // 4) % 10` of opponents' set `n % 4` (the disc's
forty: four shapes in five liveries with two numbers; the spread is this project's, the Dreamcast's was not traced)."""


class CarEntry(NamedTuple):
    body: int
    cockpit: int
    wheels: tuple[int, int, int, int]
    hubs: tuple[tuple[float, float, float], ...]


def _car_header(tex: np.ndarray, base: int, checker: bool) -> bytes:
    """A true-colour polygon's texture header (model.RGB_TEX): word 0 the size, the textured bit and the checker bit,
    word 1 zero, words 2-3 the word address of the texels."""
    h, w = tex.shape
    if any(s < 32 or s > 4096 or s & (s - 1) for s in (w, h)):
        raise ConvertError(f"a car texture of {w} x {h} is not a header size (32 << n)")
    h0 = (w // 32).bit_length() - 1 | ((h // 32).bit_length() - 1) << 3 | 1 << 14 | (1 << 15 if checker else 0)
    return struct.pack("<4H", h0, 0, base & 0xFFFF, base >> 16)


def car_set(rs: RomSet, disc, cars=(1,), field: bool = False) -> RomSet:
    """`rs` (a custom set with the library bound patched: the release chain's) with the Dreamcast cars `cars`
    (choices 1-4, each once) converted from `disc` (`convert.car.car_from_disc`) and filed: the baked texels, then per
    piece its texture headers (one per drawn record) and texture points in CAR_TEXTURES; the objects in the custom
    polygon area (NEW_ID_RUNS); the body, cockpit and two wheel lists in the list home (tgp.DATA_FREE); the car's row,
    wheel-list table and hubs in CAR_TABLE. ConvertError for an `rs` that is not a custom RomSet, `cars` that are not
    distinct choices 1-4, a set that holds cars already (all are filed in one call), or room that is short (naming
    the run)."""
    from dtools.convert import car as ccar
    from dtools.dc.car import handling as dcar_handling
    from dtools.dc.errors import DcError
    rs = _romset(rs)
    if not rs.custom:
        raise ConvertError("car_set: the set is not a custom set (romset.as_custom): the cars need its new areas")
    if (not isinstance(cars, (tuple, list)) or not cars or len(set(cars)) != len(cars)
            or not all(is_int(c) and not isinstance(c, bool) and 1 <= c < CAR_COUNT for c in cars)):
        raise ConvertError(f"car_set: cars {cars!r:.40} are not distinct choices 1..{CAR_COUNT - 1}")
    if rs.data[FIELD_TABLE:CAR_TABLE + CAR_TABLE_SIZE] != b"\xff" * (CAR_TABLE + CAR_TABLE_SIZE - FIELD_TABLE):
        raise ConvertError("car_set: the set holds cars already (they are filed in one call: a car's white texels "
                           "read as free bytes, so a later call could not tell them from room)")
    busy: list[tuple[str, int, int]] = []              # every span placed in this call, across the cars
    if not isinstance(field, bool):
        raise ConvertError(f"car_set: field {field!r:.40} is not a bool")
    for car in cars:
        conv = ccar.car_from_disc(disc, int(car))
        patches: dict[tuple[str, int], bytes] = {}

        def place(region, run, blob, what, align=4, patches=patches, busy=busy):
            at = _first_fit(rs, region, run, len(blob), f"car {car}: {what}", tuple(busy), align)
            patches[(region, at)] = bytes(blob)
            busy.append((region, at, len(blob)))
            return at

        bases = [place("textures", CAR_TEXTURES, t.astype("<u2").tobytes(), "texels") // 2 for t in conv.textures]
        lists = []
        for pieces, what in ((conv.body, "body"), (conv.cockpit, "cockpit"), (conv.wheel_left, "left wheel"),
                             (conv.wheel_right, "right wheel")):
            entries = []
            for pc in pieces:
                heads = b"".join(_car_header(conv.textures[t], bases[t], chk)
                                 for r, t, chk in zip(pc.obj.records, pc.textures, pc.checker) if t >= 0)
                tha = place("textures", CAR_TEXTURES, heads, f"{what} headers") // 2
                tpa = place("textures", CAR_TEXTURES, T.encode_texture_points(pc.points, pc.obj), f"{what} points") // 2
                words = P.encode_object(pc.obj)
                oba = place("polygons", NEW_ID_RUNS["polygons"], struct.pack(f"<{len(words)}I", *words),
                            f"{what} object") // 4
                entries.append((0x800000 | oba, tpa, tha, len(pc.obj.records)))
            need = C.OBJECT_ENTRY_SIZE * len(entries) + 4
            at = DATA_BASE + _first_fit(rs, "data", tgp.DATA_FREE, need, f"car {car}: {what} list", tuple(busy))
            busy.append(("data", at - DATA_BASE, need))
            patches[("data", at - DATA_BASE)] = C.encode_object_list(object_list_at(at, entries))
            lists.append(at)
        body, cockpit, left, right = lists
        try:
            hd = dcar_handling(disc.read("1ST_READ.BIN"), int(car))
        except DcError as err:
            raise ConvertError(str(err)) from err
        own = struct.unpack_from("<4f", rs.data, GEAR_RATIOS_AT)
        patches[("data", CAR_HANDLING)] = struct.pack("<8f", *own, *own)
        patches[("data", CAR_HANDLING + 32 * car)] = struct.pack("<8f", *hd.gears, *(g * hd.torque for g in hd.gears))
        row = CAR_TABLE + 16 * car
        wheels, hubs = CAR_TABLE + CAR_WHEELS + 16 * car, CAR_TABLE + CAR_HUBS + 48 * car
        patches[("data", row)] = struct.pack("<4I", body, cockpit, DATA_BASE + wheels, DATA_BASE + hubs)
        patches[("data", wheels)] = struct.pack("<4I", left, right, left, right)
        patches[("data", hubs)] = struct.pack("<12f", *(v for h in conv.hubs for v in h))
        rs = rs.with_patches(patches)
    return _field_set(rs, disc, busy) if field else rs


def _field_set(rs: RomSet, disc, busy: list) -> RomSet:
    """`rs` with the Dreamcast's opponents filed (car_set's `field`): each opponents' set's baked texels in
    CAR_TEXTURES, each of its objects once (headers and points in the textures region's first custom run or, when that is
    full, CAR_TEXTURES; the object in the custom polygon area), and per car index its near, mid and far lists in
    the list home and its row of FIELD_TABLE."""
    from dtools.convert import car as ccar
    from dtools.dc import car as dcar
    patches: dict[tuple[str, int], bytes] = {}

    def place(region, runs, blob, what, align=4):
        err = None
        for run in runs:
            try:
                at = _first_fit(rs, region, run, len(blob), f"field: {what}", tuple(busy), align)
            except ConvertError as e:
                err = e
                continue
            patches[(region, at)] = bytes(blob)
            busy.append((region, at, len(blob)))
            return at
        raise err

    tex_runs = (NEW_ID_RUNS["textures"], CAR_TEXTURES)     # the texels need CAR_TEXTURES: headers and points elsewhere first
    lists: dict[tuple[int, int], tuple[int, int, int]] = {}
    for k in range(dcar.ENEMY_FILES):
        es = ccar.enemies_from_disc(disc, k)
        bases = [place("textures", (CAR_TEXTURES,), t.astype("<u2").tobytes(), "texels") // 2 for t in es.textures]
        entries: dict[int, list] = {}
        for idx, pieces in es.objects.items():
            entries[idx] = []
            for pc in pieces:
                heads = b"".join(_car_header(es.textures[t], bases[t], chk)
                                 for r, t, chk in zip(pc.obj.records, pc.textures, pc.checker) if t >= 0)
                tha = place("textures", tex_runs, heads, "headers") // 2
                tpa = place("textures", tex_runs, T.encode_texture_points(pc.points, pc.obj), "points") // 2
                words = P.encode_object(pc.obj)
                oba = place("polygons", (NEW_ID_RUNS["polygons"],), struct.pack(f"<{len(words)}I", *words), "object") // 4
                entries[idx].append((0x800000 | oba, tpa, tha, len(pc.obj.records)))
        for c, parts in enumerate(es.cars):
            got = []
            for level in parts:
                ents = [e for idx in level for e in entries[idx]]
                need = C.OBJECT_ENTRY_SIZE * len(ents) + 4
                at = DATA_BASE + _first_fit(rs, "data", tgp.DATA_FREE, need, "field: a car's list", tuple(busy))
                busy.append(("data", at - DATA_BASE, need))
                patches[("data", at - DATA_BASE)] = C.encode_object_list(object_list_at(at, ents))
                got.append(at)
            lists[(k, c)] = tuple(got)
    table = bytearray(16 * FIELD_ROWS)
    for n in range(1, FIELD_ROWS):
        near, mid, far = lists[(n % dcar.ENEMY_FILES, (n // dcar.ENEMY_FILES) % dcar.ENEMY_CARS)]
        struct.pack_into("<4I", table, 16 * n, near, mid, far, far)
    patches[("data", FIELD_TABLE)] = bytes(table)
    return rs.with_patches(patches)


def field_lists(rs: RomSet, n: int) -> tuple[int, int, int, int]:
    """Car index `n`'s row of the Dreamcast field's table: its near, mid, far and far list. ConvertError for an index
    outside 1..FIELD_ROWS-1 or a set without the field."""
    rs = _romset(rs)
    if not is_int(n) or isinstance(n, bool) or not 1 <= n < FIELD_ROWS:
        raise ConvertError(f"field_lists: car {n!r} is not 1..{FIELD_ROWS - 1}")
    row = struct.unpack_from("<4I", rs.data, FIELD_TABLE + 16 * n)
    if not all(DATA_BASE <= p < DATA_BASE + 0x1000000 for p in row):
        raise ConvertError("field_lists: the set holds no Dreamcast field (build.car_set with field=True)")
    return row


def car_entry(rs: RomSet, car: int) -> CarEntry:
    """Car `car`'s row of the car table read back. ConvertError for a car outside 1..CAR_COUNT-1 (car 0 is the
    arcade's own Hornet and has no row) or one that is not filed."""
    rs = _romset(rs)
    if not is_int(car) or isinstance(car, bool) or not 1 <= car < CAR_COUNT:
        raise ConvertError(f"car_entry: car {car!r} has no row (the Dreamcast's cars are 1..{CAR_COUNT - 1})")
    body, cockpit, wheels, hubs = struct.unpack_from("<4I", rs.data, CAR_TABLE + 16 * car)
    if (wheels, hubs) != (DATA_BASE + CAR_TABLE + CAR_WHEELS + 16 * car, DATA_BASE + CAR_TABLE + CAR_HUBS + 48 * car):
        raise ConvertError(f"car_entry: car {car} is not filed in this set")
    h = struct.unpack_from("<12f", rs.data, hubs - DATA_BASE)
    return CarEntry(body, cockpit, struct.unpack_from("<4I", rs.data, wheels - DATA_BASE),
                    tuple(h[3 * k:3 * k + 3] for k in range(4)))


def car_room(before: RomSet, after: RomSet) -> str:
    """The bytes a car_set took per region, as a sentence (the spike log's)."""
    out = []
    for region in ("textures", "polygons", "data"):
        a, b = np.frombuffer(getattr(before, region), np.uint8), np.frombuffer(getattr(after, region), np.uint8)
        out.append(f"{region} {int((a != b).sum()):,} bytes")
    return ", ".join(out)

