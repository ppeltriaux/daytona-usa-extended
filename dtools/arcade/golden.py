"""The golden re-emit of milestone M1: every table of one arcade Daytona USA course decoded into a `Course`, encoded
back (`encode_course`), and compared with the ROM (`verify_lossless`). `coverage` says which bytes that proves.

The tables, their layouts and their encoders: docs/formats/model2-course-tables.md (course.py, tiles.py),
docs/formats/model2-textures.md (texture.py), docs/formats/model2-polygon-rom.md (polyrom.py),
docs/formats/model2-tgp-ground.md (tgp.py). This module only composes them; it adds no layout of its own beyond the
pointer words that tie the tables together.

What a Course re-emits (the `tables` names), per course id:

- the course pointer (program 0x4770[id]) and the course record;
- record word 0: the backdrop's upload set (ten words), its char list, char blocks, palette list and the eight
  segments;
- word 1: the step table (pit flags included); words 2-3: the two ground pointer words and the ground
  (tgp.encode_ground): the grid (*block_ref_a, 32 KB) and the attribute block (*block_ref_b, its 0xcd padding to 4
  bytes included) in data ROM; in copro_data (its first 2 MB copy) the course's slot words (list block and pool
  addresses), the list block and the polygon pool;
- words 4-5: the line table (pairs and lists, wrap points included) and the road line;
- the per-segment flags: SEGMENT_FLAGS[id] (course.SEGMENT_FLAGS) and the table it names, one u16 per road segment;
- scenery: GRID_TABLES[id], the cell grid (256 list pointers), every object list it reaches (entries and the 4-byte
  terminator), ACTOR_TABLES[id], the actor table (256 pointers), every actor list (count and 24-byte records, the
  unmodelled handlers' records included), the modelled actors' object lists, billboard list arrays (and, with a
  billboard, its reference point BILLBOARD_REF) and the switch's three `lda` instructions;
- race values: CHECKPOINT_TABLES[id] and the checkpoint table; for every mode (0-2) and difficulty (0-3) the
  per-difficulty TIME_TABLES word and the time-table words the race reads; the lap count of every mode; the car
  count (CAR_COUNTS entry 0), rolling-start word, grid base and the shared grid lanes; the spawn records 0 to car
  count - 1 of both spawn tables and their object lists;
- select screen: the course's panel record in each mode table, every tile map those records name, its map-object
  record and object list; shared by all courses: the panel-of-course array, the wheel-zone instructions, the boot
  and select upload sets (their pointer words, sets, char lists, char blocks, palette lists) and BOOT_CHAR_LIST;
- the ranking-table pointers (shared);
- the race HUD's course map (course.hud_map): its list pointer (HUD_MAP_LISTS[id]), object list, place record and
  zooms, and its outline object in polygon ROM ("HUD map outline");
- geometry: every polygon-ROM object of the object lists above (geometry mode 1, obc 5000), except the map
  objects (EXCLUDED); every texture header their records read (the HUD map outline's included); the texture points
  each of those placements reads at its tpa (the HUD outline's included: 2 words per vertex of every record, move
  records too); the two sheet-source words; every textured header's level-0 texels and its mip levels in its
  sheet's data-ROM block.

Course-shared bytes (the object library, the common sheet, the select-screen uploads, the shared pointer arrays) are
covered by each course whose tables reach them. What is not covered is listed in EXCLUDED, by name.
"""

from __future__ import annotations

import bisect
import dataclasses
import struct
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from dtools.arcade import course as C
from dtools.arcade import polyrom as P
from dtools.arcade import select as S
from dtools.arcade import texture as T
from dtools.arcade import tgp
from dtools.arcade import tiles as TL
from dtools.arcade import title as TT
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import I960_PROGRAM, I960_WINDOW, RESERVED, RomSet
from dtools.arcade.tablebase import DATA_BASE, DATA_WINDOW, CourseError, _int, _seq, _u32, is_int, is_number

GEOMETRY_MODE = 1               # the race runs geometry mode 1 (Phase 1); objects decode in it

EXCLUDED: dict[str, str] = {
    "unmodelled actor handler data": "the `obj` of actors whose handlers are not modelled (spike log, \"Scenery "
                                     "placement / Not found / not modelled\"); their 24-byte records are covered",
    "map objects in polygon ROM": "the select screen's 3D course maps: geometry mode on Circuit Select not traced; "
                                  "the map-object records and their object lists are covered",
    "time corrections": "data 0x280587c, 0x280583c: not decoded",
    "culling tables": "Advanced's progress mask (0x175e0, 0x175f4), the visit orders (0x17594, 0x175ad), the "
                      "view-test tables (0x286f7e0-): not decoded",
    "race-value pointer levels": "TIME_TABLES[mode] and its per-id words, LAP_COUNTS[mode], CAR_COUNTS[setting]: "
                                 "fixed, not re-emitted (docs/formats/model2-course-tables.md, \"Writing these "
                                 "values\"); CAR_COUNTS entry 1 is not decoded",
    "spawn records past the car count": "not read for this course",
    "other course ids": "the original set's id-3 stub (record 0x600080, its select records, the fourth entry of "
                        "every per-id array); on a grown set (course.grow_ids) the entries of ids 3-15 other than "
                        "the course's own and their records",
    "per-id arrays not decoded": "the course's entry of each course.ID_ARRAYS array no decoder reads (camera "
                                 "scripts, the tunnel lights, the music, the time-attack and best-lap SRAM words, "
                                 "the ranking screen's and select screen's per-id words, the other per-id words "
                                 "and tables the discovery log names): read, not decoded",
    "colour tables": "the luma tables (data 0x802fc4-) and polygon palette (0x84f4ec-): decoded by "
                     "texture.colour_tables, no encoder; shared by all courses",
    "rankings beyond the table pointers": "SRAM contents, RANKING_LAST, RANKING_DEFAULTS, the default time bases "
                                          "0x23909c (on a ranked set their sixteen-id copies RANK_DEFAULTS, "
                                          "RANK_BASES): a layout only",
    "course-dependent reads not tied to a table": "read per course, identified but not decoded (format doc, \"Not "
                                                  "covered by M1's golden test\"): program 0x65d4-0x6a34 (the "
                                                  "pre-race camera script 0x6c98[id], pages 0x2806b00 / 0x2807200 / "
                                                  "0x2807100), 0xf548 and 0x1149c (shared tables, indexed by the "
                                                  "race), the window code 0x221b80-0x221c34 (handler 0x221700's "
                                                  "data, 0x2850c00 / - / 0x2850d00), the per-id table at window "
                                                  "0x23895c (16-byte stride, read by 0x226c20: the select screens' "
                                                  "palette animation); extents unknown",
}

ALWAYS_COVERED = frozenset({
    "course pointer", "course record", "upload set", "char list", "char block", "palette list", "backdrop segment",
    "step table", "ground grid pointer", "ground attribute pointer", "road line", "line table", "cell grid pointer",
    "cell grid", "object list", "actor table pointer", "actor table", "actor list", "checkpoint pointer",
    "checkpoint table", "time table pointer", "time table", "laps", "car count", "rolling start", "grid base",
    "grid lanes", "spawn record", "select panel", "tile map", "map object", "panel of course", "wheel zones",
    "screen upload pointer", "ranking pointers", "polygon object", "texture header", "texture points",
    "sheet source", "texels", "texels (mips)", "ground grid", "ground attributes", "ground slot: list block address",
    "ground slot: polygon pool address", "ground lists", "ground polygons", "HUD map", "HUD map outline",
    "segment flags pointer", "segment flags"})
"""The table names every real course covers (`tables`); "billboard lists", "billboard reference" and "switch lists"
only where a course has such an actor (Beginner)."""


class Texels(NamedTuple):
    header: T.TexHeader              # any header with this geometry (sheet, tex_x, tex_y, width, height)
    indices: np.ndarray              # level 0, (height, width) 4-bit indices
    mips: tuple[np.ndarray, ...]     # levels mip_levels(header), as stored


@dataclass(frozen=True)
class Course:
    """Every table of one course, decoded. `check_course` and `encode_course` take the Course with the set (`rs`): a
    field left at its default is first filled from `decode_course(rs, course)` (`complete`, over TABLE_FIELDS), so a
    caller sets just the tables it changes and the rules between tables are judged on the whole; without `rs`, a
    Course with a table field unset is refused. They check the tables against each other first (`check_course`).
    An empty value is unset too: `complete` refills a field set to an empty dict, tuple or list (e.g.
    `actor_lists={}`, `unmodelled=()`, `billboards={}`) with the set's own value, so such a field cannot be emptied;
    except the fields reached only through the actor table (ACTOR_FIELDS) when it names no list (every slot 0).

    Addresses, per field (three conventions):

    - i960 addresses (data ROM 0x02000000+, program 0x0+, program window 0x00220000+): record_pointer, grid, cells,
      the keys of object_lists, actor_lists and billboards, actor_table, actor_slots, switch_lists, Actor.addr,
      checkpoint_table, the two addresses of each time_tables value, segment_flags[0], the address halves of laps,
      car_count, rolling_start and grid_base, GridSlot.objects, the record addresses of select_panels, the keys of
      tile_maps and char_blocks, the pointer-word addresses and UploadSet fields of screen_uploads and backdrop,
      boot_chars, rankings (SRAM), MapObject.objects, the two values of ground, hud_map.outline_list.
    - data-ROM offsets (RomSet.data offset = i960 address - 0x02000000): the CourseEntry pointer fields (entry.*),
      Placement.entry, sheet_sources, ground_data.grid_at and ground_data.attributes_at.
    - word indices: the keys of objects are polygon-ROM word indices (RomSet.polygons byte offset / 4; Placement.oba
      is 0x800000 | index), and those of hud_map.objects; the keys of headers are textures-region word indices
      (RomSet.textures byte offset / 2, = tha & TEX_ADDR_MASK), and those of points (= tpa & TEX_ADDR_MASK);
      ground_data.list_base and pool_base are copro_data word indices (byte offset / 4 in its first 2 MB copy), the
      keys of ground_data.lists word offsets from list_base (the grid values).

    ground_data (a tgp.Ground) is a source: the grid, the attribute block and the TGP side (tgp.encode_ground). Its
    attributes hold one byte per polygon, without the block's padding: the real blocks are 1,184 / 7,240 / 2,988
    bytes for 1,184 / 7,239 / 2,986 polygons, the rest 0xcd to a multiple of 4, which encode_ground writes (so
    check_course wants len(attributes) == len(polys)). A ground moved by editing ground_data (its four addresses) is
    written with slot `base`'s two words only, and entry.raw (the record's words 6-7, which hold the grid pointer
    on the real set) as it is: tgp.relocate_ground also rewrites those words and slots 4-15 (which repeat slot 0's).
    No reader of either is known (words 6-7: none found; slots 4-15: function 0x0d is sent the course id 0-2), so
    the difference is not expected to matter (not run in MAME); docs/formats/model2-tgp-ground.md section 9.5.

    Sources and views. Most fields are sources: what `tables` writes. These are views, kept for reading and never
    written; `check_course` recomputes each from its sources and refuses a Course where they differ (edit the
    source instead):

    - pit: from the steps' flags and step_n (write a new pit into the steps with course.encode_pit);
    - every Checkpoint's index, lap_line, position and extensions: from the row order, the road line's centre and
      the time table of (DEFAULT_MODE, DEFAULT_DIFFICULTY) with laps[DEFAULT_MODE] (course.checkpoint_views; the
      lap line's last extension is the phantom word, see course.Checkpoint); the checkpoint table writes only the
      rows' segment and raw;
    - every GridSlot's segment, lane and position: from its start, grid_base, grid_lanes, rolling_start and the
      road line (course.grid_slot); the spawn record writes the other fields;
    - record_pointer: COURSE_POINTERS[id] must hold course.record_address(id) (COURSE_TABLE + 0x20 * id for ids
      0-2, ID_RECORDS + 0x20 * id for 3-15), where `encode_course` writes the record (a record elsewhere is not
      supported);
    - id_arrays: where the set's per-id arrays are (course.id_tables: the original addresses, or the sixteen-entry
      tables of a grown set). `tables` writes the course's per-id words (course pointer, cell grid, actor table,
      checkpoint, segment flags, sheet source, HUD map, panel of course and ranking pointers) into these arrays;
      the course ids it accepts follow them (course.ids_of: 0-2, or 0-15 when every array is at its new address);
    - panels: mode 0's records, used to check panel_of_course (not written: select_panels holds the course's own);
    - ground, when ground_data is set: the words at block_ref_a / block_ref_b, DATA_BASE + ground_data.grid_at
      and DATA_BASE + ground_data.attributes_at; `tables` writes them from ground_data (to move the ground, edit
      both). Without ground_data, ground is a source: `tables` writes the two words from it;
    - car_objects (a property): the keys of `objects` reached from the spawn records, the car models shared by every
      course; `scenery_objects(c)` is the rest.
    """
    course: int
    entry: C.CourseEntry
    record_pointer: int | None = None                       # COURSE_POINTERS[id]
    id_arrays: dict[str, int] = field(default_factory=dict)  # per-id array name -> address in the set (a view)
    backdrop: C.Backdrop | None = None                      # record word 0: upload set and segments
    step_n: int | None = None                               # record word 1: the step table's n
    steps: tuple[C.Step, ...] = ()
    pit: C.Pit | None = None                                # a view of the steps' flags (not re-emitted)
    ground: tuple[int, int] | None = None                   # the words at block_ref_a, block_ref_b
    ground_data: tgp.Ground | None = None                   # what they point at, and the TGP side (copro_data)
    road: tuple[C.RoadPoint, ...] = ()
    ai_lines: tuple[tuple[C.AiPoint, ...], ...] = ()
    grid: int | None = None                                 # GRID_TABLES[id]
    cells: tuple[int, ...] = ()                             # the grid's 256 list pointers
    object_lists: dict[int, tuple[C.Placement, ...]] = field(default_factory=dict)
    actor_table: int | None = None                          # ACTOR_TABLES[id]
    actor_slots: tuple[int, ...] = ()                       # the table's 256 list pointers
    actor_lists: dict[int, tuple[C.Actor, ...]] = field(default_factory=dict)
    billboards: dict[int, tuple[int, ...]] = field(default_factory=dict)
    billboard_ref: tuple[float, float, float] | None = None     # BILLBOARD_REF, when the course has a billboard
    switch_lists: tuple[int, ...] = ()                      # when the course has a SWITCH actor
    unmodelled: tuple[C.Actor, ...] = ()                    # actors whose handler data is not followed
    checkpoint_table: int | None = None
    checkpoints: tuple[C.Checkpoint, ...] = ()
    time_tables: dict[tuple[int, int], tuple[int, int, tuple[int, ...]]] = field(default_factory=dict)
    #   (mode, difficulty) -> (address of the per-difficulty word, table address, the words the race reads)
    laps: dict[int, tuple[int, int]] = field(default_factory=dict)      # mode -> (address, laps)
    car_count: tuple[int, int] | None = None                # (address, count)
    rolling_start: tuple[int, bool] | None = None
    grid_base: tuple[int, int] | None = None
    grid_lanes: tuple[int, int] | None = None
    spawn: tuple[tuple[C.GridSlot, ...], ...] = ()          # per spawn table, cars 0 .. car_count - 1
    segment_flags: tuple[int, tuple[int, ...]] | None = None    # (the table SEGMENT_FLAGS[id] names, one u16 per
    #                                                             road segment: course.segment_flags)
    select_panels: tuple[tuple[int, C.SelectPanel], ...] = ()   # per mode: (record address, record)
    tile_maps: dict[int, TL.TileMap] = field(default_factory=dict)
    map_object: C.MapObject | None = None
    panels: tuple[C.SelectPanel, ...] = ()                  # mode 0's records in panel order (for panel_of_course)
    panel_of_course: tuple[int, ...] = ()
    wheel: C.WheelZones | None = None
    screen_uploads: tuple[tuple[int, TL.UploadSet], ...] = ()   # (pointer word address, set): boot, then select
    boot_chars: tuple[TL.CharUpload, ...] = ()              # BOOT_CHAR_LIST
    char_blocks: dict[int, np.ndarray] = field(default_factory=dict)    # block address -> (n, 8, 8) pixels
    rankings: tuple[int, ...] = ()                          # RANKING_TABLES (ids 0-2)
    hud_map: C.HudMap | None = None                         # the race HUD's course map (its outline: hud_map.objects)
    objects: dict[int, P.Object] = field(default_factory=dict)          # polygon-ROM word index -> object
    #   (scenery and car models alike: car_objects names the cars, scenery_objects(c) gives the rest)
    headers: dict[int, T.TexHeader] = field(default_factory=dict)       # textures-region word index -> header
    points: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = field(default_factory=dict)
    #   textures-region word index (a placement's tpa) -> its texture points: per record 3 or 4 (v, u), 1/8 texel
    #   (texture.texture_points); where placements of different objects share a tpa, the one reading the most words
    sheet_sources: tuple[int, ...] = ()                     # data-ROM offsets of sheet 0 (the course's), sheet 1
    texels: tuple[Texels, ...] = ()

    @property
    def name(self) -> str:
        return self.entry.name

    @property
    def car_objects(self) -> frozenset[int]:
        """The polygon-ROM word indices (keys of `objects`) of the car models: the objects of the lists the spawn
        records name (GridSlot.objects, through object_lists). The spawn tables are shared by every course, per car
        index, so these objects are too (a view of spawn and object_lists; empty when spawn is not set). A mirror
        leaves them alone: it transforms `scenery_objects(c)`."""
        return frozenset(C.check_rom_object(p.oba) for slots in self.spawn for s in slots for lst in s.objects
                         for p in self.object_lists.get(lst, ()))


_VIEWS = ("record_pointer", "id_arrays", "pit", "panels", "ground")
TABLE_FIELDS: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(Course)
                                      if f.name not in ("course", "entry") + _VIEWS)
"""The fields `complete` fills and `check_course` / `encode_course` need set: every field of `Course` except `course`,
`entry` and the views the Course docstring names. `ground` is left out although it is a source when `ground_data` is
unset: `ground_data` is always set by `complete`, which makes `ground` its view (a caller who edits `ground` alone is
refused as a view that differs; edit `ground_data`). A field added to `Course` joins automatically."""

FILLED_VIEWS = ("record_pointer", "id_arrays", "panels")
"""Views that are not TABLE_FIELDS but that `tables` reads (`record_pointer` is written as the course pointer,
`id_arrays` places the per-id words, `panels` is what `encode_panel_of_course` checks the panel order against):
`complete` fills them too, so a partial Course encodes what the complete one does, and a Course without a set needs
them set (but id_arrays: empty means the original set's arrays). `pit` stays a view that is checked when set
(nothing reads it)."""

OPTIONAL_FIELDS = frozenset({"billboards", "billboard_ref", "switch_lists"})
"""Table fields that are legitimately empty on a complete Course (Expert and Advanced have neither billboards, nor so
the billboards' reference point, nor the SWITCH actor): an empty value is not "unset" for a Course without a set,
but `complete` still fills it from the set when the caller left it empty."""

ACTOR_FIELDS = frozenset({"actor_lists", "unmodelled", "billboards", "billboard_ref", "switch_lists"})
"""Table fields reached only through the actor table: on a Course whose actor table names no list (`actor_slots`
GRID_CELLS zeros, as the converter's empty scenery writes) they are empty for real, neither unset nor filled by
`complete` (`_actorless`)."""


def _is_unset(value) -> bool:
    """A field at its default: None or an empty tuple, list or dict."""
    return value is None or (isinstance(value, (tuple, list, dict)) and len(value) == 0)


def _actorless(c: Course) -> bool:
    """True when `c`'s actor table is set and names no list: GRID_CELLS slots, every one the integer 0."""
    slots = c.actor_slots
    return (isinstance(slots, (tuple, list)) and len(slots) == C.GRID_CELLS
            and all(is_int(s) and s == 0 for s in slots))


def unset_fields(c: Course) -> tuple[str, ...]:
    """The TABLE_FIELDS and FILLED_VIEWS of `c` left at their default (OPTIONAL_FIELDS excluded: they may be empty
    for real; so are ACTOR_FIELDS when the actor table names no list; and id_arrays, whose empty value means the
    original set's arrays: `complete` still fills it from a set)."""
    actorless = _actorless(c)
    return tuple(f for f in TABLE_FIELDS + FILLED_VIEWS if f not in OPTIONAL_FIELDS and _is_unset(getattr(c, f))
                 and not (actorless and f in ACTOR_FIELDS) and f != "id_arrays")


def complete(c: Course, rs: RomSet) -> Course:
    """`c` with every field of TABLE_FIELDS and FILLED_VIEWS left at its default (None or empty) filled from
    `decode_course(rs, c.course)`; `c` itself when none is. The set is read only when something is unset. Fields the
    caller set are kept as they are, so the result is the caller's edit on top of the set's course, and `check_course`
    then judges the whole: tables that must agree (time tables with laps and checkpoints, the pit with the steps) are
    compared with the set's. ACTOR_FIELDS are not filled when the actor table names no list (they are empty for
    real).
    `course` and `entry` are the caller's (CourseError for a course id outside 0..n-1, n = course.id_count(rs): 3,
    or 16 on a grown set). A set id_arrays must be the set's own (course.id_tables(rs)), else CourseError: a Course
    decoded from the original set and encoded against a grown one would write its per-id words into arrays the
    grown readers no longer read."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    n = C.id_count(rs)
    if not is_int(c.course) or not 0 <= c.course < n:
        raise CourseError(f"course id {c.course!r} is not 0..{n - 1}")
    if not _is_unset(c.id_arrays) and c.id_arrays != (have := C.id_tables(rs)):
        names = sorted(n for n in set(have) | set(c.id_arrays) if c.id_arrays.get(n) != have.get(n)) \
            if isinstance(c.id_arrays, dict) else ["(not a dict)"]
        raise CourseError(f"{c.name if isinstance(c.entry, C.CourseEntry) else f'course {c.course}'}: id_arrays "
                          f"differ from the set's per-id arrays ({', '.join(names[:4])}"
                          f"{', ...' if len(names) > 4 else ''}): its per-id words would go where this set does not "
                          f"read them (a Course decoded from another set: leave id_arrays empty to take the set's)")
    actorless = _actorless(c)
    missing = [f for f in TABLE_FIELDS + FILLED_VIEWS
               if _is_unset(getattr(c, f)) and not (actorless and f in ACTOR_FIELDS)]
    if not missing:
        return c
    decoded = decode_course(rs, c.course)
    return dataclasses.replace(c, **{f: getattr(decoded, f) for f in missing})


def scenery_objects(c: Course) -> dict[int, P.Object]:
    """The polygon objects of `c` that are not car models (`Course.car_objects`): the course's own scenery (grid
    cells, actors, billboards, the switch). On the real set the two never share an object, and every object a course
    shares with another is a car model. CourseError when `spawn` is unset: the cars cannot be told from scenery
    (complete the Course first: `complete(c, rs)`)."""
    if not c.spawn:
        raise CourseError(f"{c.name}: spawn is unset: car models cannot be told from scenery; complete the Course "
                          f"first (golden.complete)")
    cars = c.car_objects
    return {k: o for k, o in c.objects.items() if k not in cars}


def _texel_key(h: T.TexHeader) -> tuple[int, int, int, int, int]:
    return h.sheet, h.tex_x, h.tex_y, h.width, h.height


def decode_course(rs: RomSet, index: int) -> Course:
    """Every table of course `index` (0 Beginner, 1 Expert, 2 Advanced; on a grown set, course.grow_ids, also 3-15,
    which hold Beginner's values until an entry is filled). Per-id words are read where the set's readers read them
    (course.id_tables, kept in Course.id_arrays). CourseError (TextureError, PolyError, tgp.TgpError) as the decoders
    it calls; NotImplementedError never: unmodelled actors are kept in `Course.unmodelled`."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    n = C.id_count(rs)
    if not is_int(index) or not 0 <= index < n:
        raise CourseError(f"course {index!r} is not a course id 0..{n - 1}")
    index = int(index)
    arrays = C.id_tables(rs)
    e = C.course_table(rs)[index]
    lists: dict[int, tuple[C.Placement, ...]] = {}

    def take_list(ptr: int, what: str) -> None:
        if ptr not in lists:
            lists[ptr] = tuple(C.object_list(rs, ptr, what))

    grid, cells = C.cell_grid(rs, e)
    for c, ptr in enumerate(cells):
        take_list(ptr, f"{e.name}: cell {c}")
    table, slots = C.actor_table(rs, e)
    actor_lists: dict[int, list[C.Actor]] = {}
    billboards, switch = {}, ()
    for a in C.actors(rs, e):
        actor_lists.setdefault(slots[a.slot], []).append(a)
        if a.handler == C.STATIC_PROP:
            take_list(a.obj, f"{e.name}: actor at {a.addr:#x}")
        elif a.handler == C.SWITCH:
            switch = C.switch_lists(rs)
            for ptr in switch:
                take_list(ptr, f"{e.name}: switch list")
        elif a.handler == C.VIEW_BILLBOARD:
            billboards[a.obj] = C.billboard_lists(rs, a.obj)
            for ptr in billboards[a.obj]:
                take_list(ptr, f"{e.name}: billboard at {a.obj:#x}")
    at = C.race_value_addresses(rs, e)
    spawn = tuple(tuple(C.start_grid(rs, e, table=t)) for t in range(len(C.SPAWN_TABLES)))
    for slots_ in spawn:
        for s in slots_:
            for o in s.objects:
                take_list(o, f"{e.name}: spawn record {s.car}")
    drawn = dict(lists)                     # the lists whose objects are decoded (not the map object's)

    sa = C.select_assets(rs)
    cs = sa.courses[index]
    select_panels = _select_fields(sa, index)["select_panels"]
    tile_maps = {a: TL.tile_map(rs, a) for _, p in select_panels for a in p.maps}
    take_list(cs.map_object.objects, f"{e.name}: map object")
    screen_uploads = tuple((slot, u) for slot, u in zip(C.BOOT_SCREEN + C.SELECT_SCREEN, sa.uploads))
    boot_chars = TL.char_list(rs, C.BOOT_CHAR_LIST)
    bd = C.backdrop(rs, e)
    blocks = {}
    for c in boot_chars + bd.uploads.chars + tuple(c for _, u in screen_uploads for c in u.chars):
        blocks.setdefault(c.src, TL.decode_chars(c.data(rs)))

    objects: dict[int, P.Object] = {}
    headers: dict[int, T.TexHeader] = {}
    for ptr, places in drawn.items():
        for p in places:
            index_ = C.check_rom_object(p.oba, rs)
            obj = objects.get(index_) or P.decode_object(rs.polygons, index_, p.obc, GEOMETRY_MODE)
            objects[index_] = obj
            for h in T.object_headers(rs, p.tha, obj):
                headers.setdefault(h.tha & T.TEX_ADDR_MASK, h)
    hud = C.hud_map(rs, e)
    for p in hud.entries:                   # the outline's headers (flat on the real set), read as a list's
        for h in T.object_headers(rs, p.tha, hud.objects[C.check_rom_object(p.oba, rs)]):
            headers.setdefault(h.tha & T.TEX_ADDR_MASK, h)
    points: dict[int, tuple] = {}
    for p, obj in [(p, objects[C.check_rom_object(p.oba, rs)]) for places in drawn.values() for p in places] + \
            [(p, hud.objects[C.check_rom_object(p.oba, rs)]) for p in hud.entries]:
        pts, k = T.texture_points(rs, p.tpa, obj), p.tpa & T.TEX_ADDR_MASK
        if sum(map(len, pts)) > sum(map(len, points.get(k, ()))):     # the longest reading covers the others' words
            points[k] = pts
    texels, seen = [], set()
    for h in headers.values():
        if h.textured and _texel_key(h) not in seen:
            seen.add(_texel_key(h))
            texels.append(Texels(h, T.decode_indices(rs, h, index),
                                 tuple(T.mip_indices(rs, h, lv, index) for lv in T.mip_levels(h))))

    return Course(
        course=index, entry=e, record_pointer=_u32(rs, arrays["COURSE_POINTERS"] + 4 * index), id_arrays=arrays,
        backdrop=bd,
        step_n=_u32(rs, DATA_BASE + e.step_table), steps=tuple(C.step_table(rs, e)), pit=C.pit(rs, e),
        ground=(_u32(rs, DATA_BASE + e.block_ref_a), _u32(rs, DATA_BASE + e.block_ref_b)),
        ground_data=tgp.decode_ground(rs, index),
        road=tuple(C.road_line(rs, e)), ai_lines=tuple(tuple(l) for l in C.ai_lines(rs, e)),
        grid=grid, cells=cells, object_lists=lists, actor_table=table, actor_slots=slots,
        actor_lists={k: tuple(v) for k, v in actor_lists.items()}, billboards=billboards, switch_lists=switch,
        billboard_ref=C.billboard_ref(rs, index) if billboards else None,
        unmodelled=tuple(C.unmodelled_actors(rs, e)),
        checkpoint_table=at["checkpoint_table"], checkpoints=tuple(C.checkpoints(rs, e)),
        time_tables={(m, d): ((a := C.race_value_addresses(rs, e, m, d))["time_pointer"], a["time_table"],
                              tuple(C.time_table(rs, e, m, d)))
                     for m in range(C.MODES) for d in range(C.DIFFICULTIES)},
        laps={m: (C.race_value_addresses(rs, e, m)["laps"], C.laps(rs, e, m)) for m in range(C.MODES)},
        car_count=(at["car_count"], C.car_count(rs, e)), rolling_start=(at["rolling_start"], C.rolling_start(rs, e)),
        grid_base=(at["grid_base"], C.grid_base(rs, e)), grid_lanes=C.grid_lanes(rs, index), spawn=spawn,
        segment_flags=(_u32(rs, arrays["SEGMENT_FLAGS"] + 4 * index), C.segment_flags(rs, e)),
        select_panels=select_panels, tile_maps=tile_maps, map_object=cs.map_object, panels=sa.panels[0],
        panel_of_course=sa.panel_of_course, wheel=sa.wheel, screen_uploads=screen_uploads, boot_chars=boot_chars,
        char_blocks=blocks, rankings=C.ranking_pointers(rs), hud_map=hud, objects=objects, headers=headers,
        points=points, sheet_sources=(T.sheet_source(rs, 0, index), T.sheet_source(rs, 1)), texels=tuple(texels))


def _select_fields(sa: C.SelectAssets, index: int) -> dict:
    """DEAD_ON_PATCHED's values of course `index` as decode_course reads them from the select assets `sa`."""
    cs = sa.courses[index]
    return {"select_panels": tuple((t + C.SELECT_RECORD_SIZE * cs.panel, sa.panels[m][cs.panel])
                                   for m, t in enumerate(C.SELECT_TABLES)),
            "wheel": sa.wheel, "map_object": cs.map_object}


DEAD_ON_PATCHED = ("select_panels", "wheel", "map_object")
"""The Course fields that hold the original Circuit Select's own tables: the course's panel record per mode (from
course.SELECT_TABLES, 0x23862c), the wheel-zone instructions (course.wheel_zones) and its 3D-map record (0x23889c).
On a set with the sixteen-entry select (select.patched) nothing reads them: the new select draws, steps and votes
from its own tables (select.select_row), and ids 3-15 decode here as Beginner's panel. They still decode and
re-emit (the bytes are there), but an edit would only change dead bytes, so check_course and encode_course with
such a set refuse one (`_refuse_dead_select`). The panels' tile maps, the panel-of-course array and the upload sets
are not here: the new select still reads them."""


def _refuse_dead_select(c: Course, rs: RomSet) -> None:
    """CourseError when `rs` carries select.patch_select and `c` sets a DEAD_ON_PATCHED field to a value other than
    the set's (an edit nothing would read)."""
    try:
        if not S.patched(rs):
            return
        have = _select_fields(C.select_assets(rs), c.course)
    except DecodeError as e:
        raise CourseError(f"course {c.course}: the set's Circuit Select: {e}") from e
    for f in DEAD_ON_PATCHED:
        if not _is_unset(got := getattr(c, f)) and got != have[f]:
            raise CourseError(f"{c.name if isinstance(c.entry, C.CourseEntry) else f'course {c.course}'}: {f} "
                              f"differs from the set's, which carries the sixteen-entry select (select.patched): "
                              f"nothing reads the original select's {f} there, so the edit is refused "
                              f"(golden.DEAD_ON_PATCHED)")


# --- emit ------------------------------------------------------------------------------------------------------

def _where(addr: int, size: int, what: str) -> tuple[str, int]:
    """The RomSet region and offset of i960 ROM address `addr` (program, program window, data: the i960's whole
    window, DATA_WINDOW; RomSet.with_patch refuses a write past a set's image)."""
    for (lo, hi), region, base in ((I960_PROGRAM, "program", 0), (I960_WINDOW, "program", 0x20000),
                                   ((DATA_BASE, DATA_BASE + DATA_WINDOW), "data", 0)):
        if lo <= addr and addr + size <= hi:
            return region, base + addr - lo
    raise CourseError(f"{what}: {addr!r} (+{size}) is not in a ROM region")


def _window_word(addr, what: str) -> bytes:
    if not is_int(addr) or not I960_WINDOW[0] <= addr < I960_WINDOW[1]:
        raise CourseError(f"{what} {addr!r} is not a program-window address")
    return struct.pack("<I", addr)


def tables(c: Course) -> list[tuple[str, int, bytes, str]]:
    """(region, offset, bytes, table name) for every table of `c` that is set, in a fixed order (see the module
    docstring). Regions are RomSet attribute names (program, data, polygons, textures, copro_data), so an entry can
    be written with RomSet.with_patch (copro_data offsets are in its first 2 MB copy; with_patch writes both).
    Errors of the encoders propagate (CourseError, TextureError, PolyError, tgp.TgpError)."""
    out: list[tuple[str, int, bytes, str]] = []

    def at(addr: int, blob: bytes, name: str) -> None:
        region, off = _where(addr, len(blob), name)
        out.append((region, off, blob, name))

    def uploads(u: TL.UploadSet, segments=()) -> None:
        at(u.addr, TL.encode_upload_set(u.char_list, u.palette_list, segments), "upload set")
        at(u.char_list, TL.encode_char_list([(x.src, x.dst) for x in u.chars]), "char list")
        at(u.palette_list, TL.encode_palette_list(u.palette), "palette list")

    i, arrays = c.course, c.id_arrays

    def per_id(name: str, stride: int = 4) -> int:
        return C.array_at(arrays, name) + stride * i

    if c.record_pointer is not None:
        at(per_id("COURSE_POINTERS"), C.encode_table_pointer(c.record_pointer), "course pointer")
    at(C.record_address(i), C.encode_course_entry(c.entry), "course record")
    e = c.entry
    if c.backdrop is not None:
        uploads(c.backdrop.uploads, c.backdrop.uploads.segments)
        for a, s in zip(c.backdrop.uploads.segments, c.backdrop.segments):
            at(a, TL.encode_backdrop_segment(s), "backdrop segment")
    if c.step_n is not None:
        at(DATA_BASE + e.step_table, C.encode_step_table(c.step_n, list(c.steps)), "step table")
    ground, words = [], c.ground
    if c.ground_data is not None:
        ground = tgp.ground_tables(c.ground_data)            # encode_ground's entries, named; TgpError when malformed
        words = (DATA_BASE + c.ground_data.grid_at, DATA_BASE + c.ground_data.attributes_at)
    if words is not None:
        at(DATA_BASE + e.block_ref_a, C.encode_table_pointer(words[0]), "ground grid pointer")
        at(DATA_BASE + e.block_ref_b, C.encode_table_pointer(words[1]), "ground attribute pointer")
    out += ground
    if c.road:
        at(DATA_BASE + e.record_table, C.encode_road_line(list(c.road)), "road line")
    if c.ai_lines:
        at(DATA_BASE + e.line_table, C.encode_line_table(e.line_table, [list(l) for l in c.ai_lines]), "line table")
    if c.segment_flags is not None:
        table, flags = _seq(c.segment_flags, 2, "segment_flags", "segment flags (table address, flags)")
        at(per_id("SEGMENT_FLAGS"), C.encode_table_pointer(table), "segment flags pointer")
        at(table, C.encode_segment_flags(flags), "segment flags")
    if c.grid is not None:
        at(per_id("GRID_TABLES"), C.encode_table_pointer(c.grid), "cell grid pointer")
        at(c.grid, C.encode_cell_grid(c.cells), "cell grid")
    for a, places in c.object_lists.items():
        at(a, C.encode_object_list(places), "object list")
    if c.actor_table is not None:
        at(per_id("ACTOR_TABLES"), _window_word(c.actor_table, "actor table pointer"), "actor table pointer")
        at(c.actor_table, C.encode_actor_table(c.actor_slots), "actor table")
    for a, records in c.actor_lists.items():
        at(a, C.encode_actor_list(records), "actor list")
    for a, lists in c.billboards.items():
        at(a, C.encode_billboard_lists(lists), "billboard lists")
    if c.billboard_ref is not None:
        at(C.id_entry(arrays, "BILLBOARD_REF", i), C.encode_billboard_ref(c.billboard_ref), "billboard reference")
    if c.switch_lists:
        for a, blob in C.encode_switch_lists(c.switch_lists).items():
            at(a, blob, "switch lists")
    if c.checkpoint_table is not None:
        at(per_id("CHECKPOINT_TABLES"), C.encode_table_pointer(c.checkpoint_table), "checkpoint pointer")
        at(c.checkpoint_table, C.encode_checkpoint_table(list(c.checkpoints)), "checkpoint table")
    for (_, _), (ptr_at, table, words) in c.time_tables.items():
        at(ptr_at, C.encode_table_pointer(table), "time table pointer")
        at(table, C.encode_time_table(list(words)), "time table")
    for a, n in c.laps.values():
        at(a, C.encode_laps(n), "laps")
    if c.car_count is not None:
        at(c.car_count[0], C.encode_car_count(c.car_count[1]), "car count")
    if c.rolling_start is not None:
        at(c.rolling_start[0], C.encode_rolling_start(c.rolling_start[1]), "rolling start")
    if c.grid_base is not None:
        at(c.grid_base[0], C.encode_grid_base(c.grid_base[1]), "grid base")
    if c.grid_lanes is not None:
        at(C.id_entry(arrays, "GRID_LANES", i), C.encode_grid_lanes(c.grid_lanes), "grid lanes")
    for t, slots in enumerate(c.spawn):
        for s in slots:
            at(C.SPAWN_TABLES[t] + C.SPAWN_RECORD_SIZE * s.car, C.encode_spawn_record(s), "spawn record")
    for a, p in c.select_panels:
        at(a, C.encode_select_panel(p), "select panel")
    for a, m in c.tile_maps.items():
        at(a, TL.encode_tile_map(m), "tile map")
    if c.map_object is not None:
        at(C.SELECT_MAP_OBJECTS + 32 * c.map_object.panel, C.encode_map_object(c.map_object), "map object")
    if c.panel_of_course:
        at(C.array_at(arrays, "SELECT_PANEL_OF_COURSE"), C.encode_panel_of_course(c.panel_of_course, c.panels),
           "panel of course")
    if c.wheel is not None:
        for a, blob in C.encode_wheel_zones(c.wheel).items():
            at(a, blob, "wheel zones")
    for slot, u in c.screen_uploads:
        at(slot, C.encode_table_pointer(u.addr), "screen upload pointer")
        uploads(u)
    if c.boot_chars:
        at(C.BOOT_CHAR_LIST, TL.encode_char_list([(x.src, x.dst) for x in c.boot_chars]), "char list")
    for a, px in c.char_blocks.items():
        at(a, TL.encode_char_block(TL.encode_chars(px)), "char block")
    if c.rankings:
        at(C.array_at(arrays, "RANKING_TABLES"), C.encode_ranking_pointers(c.rankings), "ranking pointers")
    if c.hud_map is not None:
        for a, blob in C.encode_hud_map(c.hud_map, arrays):
            at(a, blob, "HUD map")
        for index, obj in c.hud_map.objects.items():
            words = P.encode_object(obj)
            out.append(("polygons", 4 * index, struct.pack(f"<{len(words)}I", *words), "HUD map outline"))
    for index, obj in c.objects.items():
        words = P.encode_object(obj)
        out.append(("polygons", 4 * index, struct.pack(f"<{len(words)}I", *words), "polygon object"))
    for index, h in c.headers.items():
        out.append(("textures", 2 * index, T.encode_tex_header(h), "texture header"))
    for index, pts in c.points.items():
        out.append(("textures", 2 * index, T.encode_texture_points(pts), "texture points"))
    if c.sheet_sources:
        at(per_id("COURSE_SHEETS"), T.encode_sheet_source(c.sheet_sources[0]), "sheet source")
        at(T.COMMON_SHEET_AT, T.encode_sheet_source(c.sheet_sources[1]), "sheet source")
    for tx in c.texels:
        out.extend(("data", off, b, name) for off, b, name in texel_writes(tx, c.sheet_sources[tx.header.sheet]))
    return out


def texel_writes(tx: Texels, base: int) -> list[tuple[int, bytes, str]]:
    """(data-ROM offset, bytes, "texels" or "texels (mips)") for one texture whose sheet block starts at data-ROM
    offset `base`: its level-0 texels (`texture.texel_spans_at`) then each mip level of `mip_levels(header)` that
    `tx.mips` holds (`texture.encode_mip_bytes`). The one writer of texels: `tables` uses it, and so does the
    converter (dtools/convert/textures.py). TextureError as the encoders."""
    out: list[tuple[int, bytes, str]] = []
    blob, pos = T.encode_texels(tx.header, tx.indices), 0
    for off, n in T.texel_spans_at(tx.header, base):
        out.append((off, blob[pos:pos + n], "texels"))
        pos += n
    for level, m in zip(T.mip_levels(tx.header), tx.mips):
        for off, b in T.encode_mip_bytes(tx.header, level, m, base):
            out.append((off, b, "texels (mips)"))
    return out


def _views_differ(name: str, got, want, fields, sources: str) -> None:
    for f in fields:
        g, w = getattr(got, f), getattr(want, f)
        if g != w:
            raise CourseError(f"{name}: {f} {g!r} differs from {w!r}, what {sources} give (a view: edit the source)")


def check_course(c: Course, rs: RomSet | None = None) -> None:
    """The rules between the tables of `c`. With `rs`, every field left at its default is first filled from the set
    (`complete`); without, a Course with any table field unset is refused (CourseError naming them): a rule runs only
    when the fields it needs are set, so a partial Course would encode tables the game mis-reads against the ROM's
    unchanged ones. CourseError naming the rule and the values when one fails: a Course that breaks one would encode
    without error and read out of bounds in the game (or drop an edit). The rules (`_check_rules`, which applies
    each when the fields it needs are set):

    - the course id is one the id_arrays hold (0-2, or 0-15 on a grown set: course.ids_of) and entry.course;
      record_pointer, when set, is course.record_address(id); when id_arrays is set, the address halves of
      rolling_start, grid_base, car_count and laps are entry `id` of ROLLING_STARTS, GRID_BASES, CAR_COUNTS[0] and
      LAP_COUNTS[mode] at the addresses it gives;
    - every AI line has len(road) points (a car's segment indexes both); step_n is len(road) - 1; step and checkpoint
      segments are <= n; every AiPoint.step is < len(steps) when the step table is set (step_n); segment_flags
      holds one value per road point (0x10f7c / 0x13c7c read entry `segment`);
    - every (mode, difficulty) time table holds 1 + laps[mode] * (checkpoints + 1) words, the final lap line's
      phantom word included (course.Checkpoint), when laps[mode] and the checkpoints are set;
    - spawn: at most len(SPAWN_TABLES) tables, cars 0..SPAWN_RECORDS-1 once each, lines < AI_LINE_COUNT; every table
      holds a record for each of cars 0 .. car_count - 1 when car_count is set;
    - at most course.ACTOR_POOL actors in all (the lists the actor table names, each as often as it is named);
    - ground_data, when set: a tgp.Ground of this course id, in the slot of this id (base == id: the game selects
      the slot with the course id); one attribute byte per polygon (len(attributes) ==
      len(polys), the block's 0xcd padding not stored: Course docstring); when the step table is set (step_n), the
      polygons of kind 1 (road, attributes[i] & 15) are exactly the first len(steps), one per step; every grid
      value is a u16 that is 0 (off the ground) or names a list, and every list entry names a polygon that exists;
      encode_ground's own checks (polygon shape, layout) raise tgp.TgpError from `tables`;
    - the views (Course docstring): pit equals what the steps' flags give; the checkpoints are the rows then one lap
      line (segment 0, raw 0), with index, position and extensions as course.checkpoint_views gives them; every
      GridSlot's segment, lane and position as course.grid_slot gives them; ground (the two pointer words) is
      DATA_BASE + ground_data.grid_at and DATA_BASE + ground_data.attributes_at, when both are set;
    - hud_map, when set: a course.HudMap of this course id (encode_hud_map writes at its own id);
    - with `rs` carrying the sixteen-entry select (select.patched): select_panels, wheel and map_object are the set's
      (DEAD_ON_PATCHED: nothing reads the original select's tables there, so an edit is refused)."""
    _check_rules(_resolve(c, rs))


def _resolve(c: Course, rs: RomSet | None) -> Course:
    """`c` completed from `rs`, or `c` when no table field is unset (CourseError naming them when there is no set)."""
    if not isinstance(c, Course):
        raise CourseError(f"{c!r:.60} is not a golden.Course")
    if rs is not None:
        out = complete(c, rs)
        _refuse_dead_select(out, rs)
        return out
    if missing := unset_fields(c):
        name = c.entry.name if isinstance(c.entry, C.CourseEntry) else f"course {c.course!r}"
        raise CourseError(f"{name}: unset fields {', '.join(missing)}: a partial Course cannot be checked or "
                          f"encoded without a set (pass `rs`: they are filled from decode_course)")
    return c


def _race_shapes(c: Course, fail) -> None:
    """The type checks `_check_rules` needs before any arithmetic on a field (the encoders' own checks run later)."""
    def ints(v, n, what):
        if not isinstance(v, (tuple, list)) or len(v) != n or not all(is_int(x) for x in v):
            fail(what, f"{v!r:.60} is not {n} integers")

    def seq(v, what):
        try:
            return _seq(v, None, what)
        except CourseError as e:
            fail(what, str(e))

    for f in ("car_count", "grid_base", "rolling_start"):          # (address, value); rolling_start's is a bool
        v = getattr(c, f)
        if v is not None and (not isinstance(v, (tuple, list)) or len(v) != 2 or not is_int(v[0])
                              or (f != "rolling_start" and not is_int(v[1]))):
            fail(f, f"{v!r:.60} is not (address, value)")
    if c.grid_lanes is not None:
        ints(c.grid_lanes, 2, "grid lanes")
    if not isinstance(c.laps, dict):
        fail("laps", f"{c.laps!r:.60} is not a dict (mode -> (address, laps))")
    for m, v in c.laps.items():
        ints(v, 2, f"laps[{m!r}]")
    if not isinstance(c.time_tables, dict):
        fail("time table", f"{c.time_tables!r:.60} is not a dict ((mode, difficulty) -> (pointer, table, words))")
    for key, v in c.time_tables.items():
        if not isinstance(key, tuple) or len(key) != 2:
            fail("time table", f"key {key!r:.60} is not (mode, difficulty)")
        if not isinstance(v, (tuple, list)) or len(v) != 3 or not is_int(v[0]) or not is_int(v[1]):
            fail("time table", f"{key}: {v!r:.60} is not (pointer address, table address, words)")
        seq(v[2], f"time table {key} words")
    for what, items, cls in (("steps", c.steps, C.Step), ("checkpoints", c.checkpoints, C.Checkpoint)):
        for k, x in enumerate(seq(items, what)):
            if not isinstance(x, cls):
                fail(what, f"item {k} {x!r:.60} is not a course.{cls.__name__}")
    for k, s in enumerate(c.steps):
        if not is_int(s.flags):
            fail("step table", f"step {k}: flags {s.flags!r} are not an integer")
    for k, line in enumerate(seq(c.ai_lines, "AI lines")):
        seq(line, f"AI line {k}")
    sf = c.segment_flags
    if sf is not None and (not isinstance(sf, (tuple, list)) or len(sf) != 2 or not is_int(sf[0])
                           or not isinstance(sf[1], (tuple, list)) or not all(is_int(f) for f in sf[1])):
        fail("segment flags", f"{sf!r:.60} is not (table address, flags): an address and a sequence of integers")
    for k, p in enumerate(seq(c.road, "road line")):       # centreline (grid slots, checkpoints) reads every one
        if not isinstance(p, C.RoadPoint):
            fail("road line", f"point {k} {p!r:.60} is not a course.RoadPoint")
        lanes = seq(p.lanes, f"road point {k} lanes")
        if len(lanes) != C.LANES or not all(isinstance(v, C.Lane) for v in lanes):
            fail("road line", f"point {k}: lanes {p.lanes!r:.60} are not {C.LANES} course.Lane")
        if not is_int(p.lane_lo) or not is_int(p.lane_hi) or not 0 <= p.lane_lo <= p.lane_hi < C.LANES:
            fail("road line", f"point {k}: lane_lo {p.lane_lo!r}, lane_hi {p.lane_hi!r} are not 0 <= lo <= hi "
                 f"<= {C.LANES - 1}")
        for j, v in enumerate(lanes):
            if (not isinstance(v.position, (tuple, list)) or len(v.position) != 3
                    or not all(is_number(x) for x in v.position)):
                fail("road line", f"point {k} lane {j}: position {v.position!r:.60} is not three finite numbers")
    if not isinstance(c.actor_lists, dict):
        fail("actors", f"actor_lists {c.actor_lists!r:.60} is not a dict (list address -> actors)")
    for a, v in c.actor_lists.items():
        seq(v, f"actor list {a!r}")
    for k, ptr in enumerate(seq(c.actor_slots, "actor table")):
        if not is_int(ptr):
            fail("actor table", f"slot {k}: {ptr!r:.60} is not a list address")
    for t, slots in enumerate(seq(c.spawn, "spawn")):
        for s in seq(slots, f"spawn table {t}"):
            if not isinstance(s, C.GridSlot):
                fail("spawn", f"table {t}: {s!r:.60} is not a course.GridSlot")
            if not is_int(s.start):
                fail("spawn", f"table {t} car {s.car!r}: start {s.start!r} is not an integer")
            seq(s.objects, f"spawn table {t} car {s.car!r} objects")


def _check_rules(c: Course) -> None:
    """check_course's rules on `c` as it is: each is applied when the fields it needs are set, so a partial Course
    passes the rules whose fields are missing (check_course and encode_course refuse a partial Course first)."""
    name = c.name if isinstance(c.entry, C.CourseEntry) else f"course {c.course!r}"

    def fail(rule: str, detail: str):
        raise CourseError(f"{name}: {rule}: {detail}")

    if not isinstance(c.entry, C.CourseEntry):
        fail("course entry", f"{c.entry!r:.60} is not a course.CourseEntry")
    _race_shapes(c, fail)
    i = c.course
    if not isinstance(c.id_arrays, dict) or not all(isinstance(k, str) and is_int(v) for k, v in c.id_arrays.items()):
        fail("id arrays", f"{c.id_arrays!r:.60} is not a dict of per-id array names to addresses")
    if unknown := sorted(set(c.id_arrays) - set(C.ID_ARRAYS)):
        fail("id arrays", f"{unknown} are not per-id arrays (course.ID_ARRAYS)")
    n = C.ids_of(c.id_arrays)
    if not is_int(i) or not 0 <= i < n:
        fail("course id", f"{i!r} is not 0..{n - 1}")
    if c.entry.course != i:
        fail("course id", f"entry.course {c.entry.course!r} is not the Course's id {i}")
    if c.id_arrays:                     # the per-id values' addresses are where the set's arrays put them
        at = [(f, array, stride, getattr(c, f)) for f, array, stride in
              (("rolling_start", "ROLLING_STARTS", 2), ("grid_base", "GRID_BASES", 2),
               ("car_count", "CAR_COUNTS[0]", 4))]
        at += [(f"laps[{m}]", f"LAP_COUNTS[{m}]", 4, v) for m, v in c.laps.items() if m in range(C.MODES)]
        for f, array, stride, v in at:
            if v is not None and v[0] != (w := C.array_at(c.id_arrays, array) + stride * i):
                fail("id arrays", f"{f}'s address {v[0]:#x} is not {w:#x}, entry {i} of {array} where id_arrays puts "
                     f"it (a value read from another set's arrays: leave it unset to take the set's)")
    want = C.record_address(i)
    if c.record_pointer is not None and c.record_pointer != want:
        fail("record pointer", f"{c.record_pointer!r:} ({c.record_pointer:#x}) is not {want:#x}: encode_course writes "
             f"the record at course.record_address(id), so COURSE_POINTERS[{i}] must point there"
             if is_int(c.record_pointer) else f"{c.record_pointer!r} is not {want:#x}")

    if c.hud_map is not None and (not isinstance(c.hud_map, C.HudMap) or c.hud_map.course != i):
        fail("HUD map", f"{c.hud_map!r:.60} is not a course.HudMap of course {i} (its tables are written at the "
             f"HudMap's own id)")

    road, steps = list(c.road), list(c.steps)
    n = len(road) - 1 if road else None
    if c.ground_data is not None:
        _check_ground(c, i, steps, fail)
    if road:
        for k, line in enumerate(c.ai_lines):
            if len(line) != len(road):
                fail("AI lines", f"line {k} has {len(line)} points, the road line {len(road)} (a car's segment "
                     f"indexes both)")
        if c.step_n is not None and c.step_n != n:
            fail("step table", f"step_n {c.step_n!r} is not the road line's n {n} ({len(road)} points)")
        for k, s in enumerate(steps):
            if not is_int(s.segment) or not 0 <= s.segment <= n:
                fail("step table", f"step {k}: segment {s.segment!r} is past the road line's n {n}")
        for cp in c.checkpoints:
            if not is_int(cp.segment) or not 0 <= cp.segment <= n:
                fail("checkpoints", f"checkpoint {cp.index!r}: segment {cp.segment!r} is past the road line's n {n}")
        if c.segment_flags is not None and len(c.segment_flags[1]) != len(road):
            fail("segment flags", f"{len(c.segment_flags[1])} values, the road line {len(road)} points (one per "
                 f"segment: 0x10f7c and 0x13c7c read entry `segment`)")
    if c.step_n is not None:
        for k, line in enumerate(c.ai_lines):
            for j, p in enumerate(line):
                if not is_int(p.step) or not 0 <= p.step < len(steps):
                    fail("AI lines", f"line {k} point {j}: step {p.step!r} is past the {len(steps)} steps")

    rows = [cp for cp in c.checkpoints if not cp.lap_line]
    if c.checkpoints:
        lap = c.checkpoints[-1]
        if len(rows) != len(c.checkpoints) - 1 or not lap.lap_line:
            fail("checkpoints", "the lap line must be the last checkpoint, once")
        if (lap.segment, lap.raw) != (0, 0):
            fail("checkpoints", f"the lap line has segment {lap.segment!r}, raw {lap.raw!r}; it is segment 0, raw 0 "
                 f"(the wrap from n to 0; not stored)")
    for (m, d), (_, _, words) in c.time_tables.items():
        if m not in range(C.MODES) or d not in range(C.DIFFICULTIES):
            fail("time table", f"({m!r}, {d!r}) is not a (mode 0..{C.MODES - 1}, difficulty 0..{C.DIFFICULTIES - 1})")
        if c.checkpoints and m in c.laps:
            laps = c.laps[m][1]
            need = 1 + laps * (len(rows) + 1)
            if len(words) < need:
                fail("time table", f"mode {m}, difficulty {d}: {len(words)} words, needs {need} (1 + laps {laps} x "
                     f"(checkpoints {len(rows)} + 1): what the race reads plus the final lap line's phantom word)")

    if len(c.spawn) > len(C.SPAWN_TABLES):
        fail("spawn", f"{len(c.spawn)} spawn tables, the game has {len(C.SPAWN_TABLES)}")
    for t, slots in enumerate(c.spawn):
        seen = set()
        for s in slots:
            if not is_int(s.car) or not 0 <= s.car < C.SPAWN_RECORDS:
                fail("spawn", f"spawn table {t}: car {s.car!r} is not a record 0..{C.SPAWN_RECORDS - 1}")
            if s.car in seen:
                fail("spawn", f"spawn table {t}: car {s.car} twice")
            seen.add(s.car)
            if not is_int(s.line) or not 0 <= s.line < C.AI_LINE_COUNT:
                fail("spawn", f"spawn table {t} car {s.car}: line {s.line!r} is past the {C.AI_LINE_COUNT} AI lines")
        if c.car_count is not None and (missing := sorted(set(range(c.car_count[1])) - seen)):
            fail("car count", f"car count {c.car_count[1]}, but spawn table {t} has no record for cars {missing}")

    lists = [c.actor_lists.get(p, ()) for p in c.actor_slots if p] if c.actor_slots else c.actor_lists.values()
    if (count := sum(map(len, lists))) > C.ACTOR_POOL:
        fail("actors", f"{count} actors; the spawner creates at most {C.ACTOR_POOL} (data 0x02802f74)")

    if c.pit is not None and c.step_n is not None:
        got = C._pit_of(steps, c.step_n, f"{name}: step table")
        if got != c.pit:
            fail("pit", f"{c.pit} differs from what the steps' flags give ({got}); the pit is a view: write it into "
                 f"the steps with course.encode_pit")
    key = (C.DEFAULT_MODE, C.DEFAULT_DIFFICULTY)
    if c.checkpoints and road and key in c.time_tables and C.DEFAULT_MODE in c.laps:
        want_cps = C.checkpoint_views([(cp.segment, cp.raw) for cp in rows], road, c.time_tables[key][2],
                                      c.laps[C.DEFAULT_MODE][1])
        for got, w in zip(c.checkpoints, want_cps):
            _views_differ(f"{name}: checkpoint {got.index!r}", got, w, ("index", "lap_line", "position", "extensions"),
                          f"the road line and the mode {key[0]} / difficulty {key[1]} time table")
    if c.spawn and road and None not in (c.grid_base, c.grid_lanes, c.rolling_start):
        if n < 1:
            fail("spawn", f"a road line of {len(road)} point(s) cannot hold a grid")
        for t, slots in enumerate(c.spawn):
            for s in slots:
                w = C.grid_slot(s.car, s.objects, s.line, s.start, s.speed, s.kind, road, c.grid_base[1],
                                tuple(c.grid_lanes), c.rolling_start[1])
                _views_differ(f"{name}: spawn table {t} car {s.car}", s, w, ("segment", "lane", "position"),
                              "its start, the grid base, the grid lanes, the start type and the road line")


def _check_ground(c: Course, i: int, steps: list, fail) -> None:
    """check_course's ground rules (its docstring); every field type-checked before it is used."""
    g = c.ground_data
    if not isinstance(g, tgp.Ground):
        fail("ground", f"ground_data {g!r:.60} is not a tgp.Ground")
    if not is_int(g.course) or g.course != i:
        fail("ground", f"ground_data.course {g.course!r} is not the Course's id {i}")
    if not is_int(g.base) or g.base != i:
        fail("ground", f"ground_data.base (the slot) {g.base!r:.60} is not the Course's id {i}: the game sends "
             f"the course id to function 0x0d, which selects the slot")
    try:
        tgp.ground_tables(g)                  # the ground's own validity: one message per violation, in one place
    except tgp.TgpError as e:
        fail("ground", str(e))
    count = len(g.polys)
    if c.ground is not None:
        if not isinstance(c.ground, (tuple, list)) or len(c.ground) != 2:
            fail("ground pointer", f"ground {c.ground!r:.60} is not the two pointer words")
        for got, f, label in zip(c.ground, ("grid_at", "attributes_at"), ("grid", "attribute block")):
            want = DATA_BASE + getattr(g, f)
            if not is_int(got):
                fail("ground pointer", f"the {label} pointer word {got!r:.60} is not {want:#x}, DATA_BASE + "
                     f"ground_data.{f}")
            if got != want:
                fail("ground pointer", f"the {label} pointer word {got:#x} differs from {want:#x}, DATA_BASE + "
                     f"ground_data.{f} (a view: edit ground_data and ground together)")
    if c.step_n is not None:
        k = len(steps)
        if count < k:
            fail("ground", f"{count} polygons, but {k} steps: the first {k} polygons are the road, one per step")
        for p, a in enumerate(g.attributes):
            if (a & 15 == 1) != (p < k):
                fail("ground", f"polygon {p} has kind {a & 15}, but the first {k} polygons (one per step) are the "
                     f"road (kind 1) and no other is")


def encode_course(c: Course, rs: RomSet | None = None) -> dict[tuple[str, int], bytes]:
    """(region, offset) -> bytes, one entry per table and per object (`tables`). With `rs`, fields left at their
    default are first filled from the set (`complete`); without, a partial Course is refused (`check_course`).
    CourseError when a rule between the tables fails (`check_course`, first; with `rs`, an edit of a DEAD_ON_PATCHED
    field on a set with the sixteen-entry select among them) or when two tables overlap with different bytes (naming
    both): a course whose tables disagree cannot be written."""
    c = _resolve(c, rs)
    _check_rules(c)
    return _encode(c)


def _encode(c: Course) -> dict[tuple[str, int], bytes]:
    """`tables` merged into (region, offset) -> bytes, overlaps checked; no rule check (encode_course runs them)."""
    out: dict[tuple[str, int], bytes] = {}
    entries = sorted(tables(c), key=lambda t: (t[0], t[1], -len(t[2])))
    open_: list[tuple[str, int, bytes, str]] = []       # earlier entries of the region that may still overlap
    for region, off, blob, name in entries:
        open_ = [t for t in open_ if t[0] == region and t[1] + len(t[2]) > off]
        for _, o2, b2, n2 in open_:
            lo, hi = off, min(off + len(blob), o2 + len(b2))
            if blob[:hi - lo] != b2[lo - o2:hi - o2]:
                raise CourseError(f"{c.name}: {name} at {region}+{off:#x} overlaps {n2} at {region}+{o2:#x} "
                                  f"with different bytes")
        open_.append((region, off, blob, name))
        key = (region, off)
        if key not in out or len(blob) > len(out[key]):       # same start: the longer holds the shorter
            out[key] = blob
    return out


def mismatches(rs: RomSet, c: Course) -> list[str]:
    """One line per table of `c` whose bytes differ from `rs`, naming the table, where it is and the first byte that
    differs; empty when every table re-emits identically."""
    out = []
    for region, off, blob, name in tables(c):
        rom = getattr(rs, region)[off:off + len(blob)]
        if rom == blob:
            continue
        if len(rom) < len(blob):
            out.append(f"{c.name}: {name} at {region}+{off:#x} ({len(blob)} bytes) runs past the {region} region")
            continue
        diff = [k for k in range(len(blob)) if rom[k] != blob[k]]
        out.append(f"{c.name}: {name} at {region}+{off:#x} ({len(blob)} bytes): {len(diff)} bytes differ, first at "
                   f"{region}+{off + diff[0]:#x}")
    return out


def verify_lossless(rs: RomSet, index: int) -> list[str]:
    """Decode course `index`, re-emit every table (`encode_course`: overlapping tables must agree) and compare with
    the ROM: empty when every table and every course object re-emits identically, else one line per mismatch
    naming the table and offset (`mismatches`).

    Bytes covered: exactly `coverage(rs, index)`, i.e. the tables the module docstring lists. The race-value pointer
    trees are covered as docs/formats/model2-course-tables.md says a converter rewrites them: the per-id
    CHECKPOINT_TABLES word, the per-difficulty TIME_TABLES words, the lap-count, car-count, rolling-start and
    grid-base values in place; their upper levels are not. Everything not covered is in EXCLUDED."""
    c = decode_course(rs, index)
    encode_course(c)
    return mismatches(rs, c)


def coverage(rs: RomSet, index: int) -> list[tuple[str, int, int, str]]:
    """(region, offset, length, table name) of every table `verify_lossless` compares for course `index`, sorted.
    Entries may overlap (shared lists, texels of headers with the same geometry); `covered_bytes` counts the
    union."""
    return sorted((r, o, len(b), n) for r, o, b, n in tables(decode_course(rs, index)))


NO_EXTENT = frozenset({"unmodelled actor handler data", "course-dependent reads not tied to a table"})
"""EXCLUDED entries without a known extent: an unmodelled handler's data layout and the reads whose tables are not
identified are what is not known."""

DECODED_ARRAYS = frozenset({"COURSE_POINTERS", "COURSE_SHEETS", "ACTOR_TABLES", "GRID_BASES", "SELECT_PANEL_OF_COURSE",
                            "RANKING_TABLES", "ROLLING_STARTS", "TIME_TABLES[0]", "TIME_TABLES[1]", "TIME_TABLES[2]",
                            "LAP_COUNTS[0]", "LAP_COUNTS[1]", "LAP_COUNTS[2]", "CAR_COUNTS[0]", "CAR_COUNTS[1]",
                            "CHECKPOINT_TABLES", "GRID_TABLES", "SEGMENT_FLAGS", "HUD_MAP_LISTS", "HUD_MAP_PLACES",
                            "HUD_MAP_ZOOMS", "GRID_LANES", "BILLBOARD_REF"})
"""The course.ID_ARRAYS a decoder reads (a Course re-emits the course's entry, or the race-value pointer levels name
it); the others are EXCLUDED's "per-id arrays not decoded"."""
SHARED_ARRAYS = frozenset({"SELECT_PANEL_OF_COURSE", "RANKING_TABLES"})
"""Decoded arrays a Course re-emits for ids 0-2 whatever its own id (panel_of_course, rankings)."""


def excluded_spans(rs: RomSet, index: int) -> list[tuple[str, int, int, str]]:
    """(region, offset, length, EXCLUDED name) for the excluded tables whose extent is known, for course `index`:
    the bytes `coverage` must not reach. Extents taken from the format docs: the time corrections (u32[4][4] each),
    the culling tables, the race-value pointer levels, the spawn records past the car count, the other course ids'
    entries of the decoded per-id arrays (DECODED_ARRAYS; SEGMENT_FLAGS' among them) and their records (the original
    set's id-3 stub, or a grown set's ids 3-15 but `index`), the per-id arrays no decoder reads (whole), the colour
    tables, the rankings' other program words, the select map objects in polygon ROM. On a grown set the arrays are
    the sixteen-entry ones (Course.id_arrays); the original arrays it leaves behind are read by nothing and named by
    neither."""
    c = decode_course(rs, index)
    e = c.entry
    out: list[tuple[str, int, int, str]] = []

    def at(addr: int, n: int, name: str) -> None:
        region, off = _where(addr, n, name)
        out.append((region, off, n, name))

    arrays = c.id_arrays
    ids = C.ids_of(arrays)
    grown = ids > C.COURSE_COUNT
    at(0x0280583C, 0x80, "time corrections")
    at(0x0286F7E0, 4 * 0xA0, "culling tables")
    for arr, n in ((C.TIME_TABLES, 4 * C.MODES), (C.LAP_COUNTS, 4 * C.MODES), (C.CAR_COUNTS, 8)):
        at(arr, n, "race-value pointer levels")
    for m in range(C.MODES):
        at(_u32(rs, C.TIME_TABLES + 4 * m) + 4 * index, 4, "race-value pointer levels")
    at(_u32(rs, C.CAR_COUNTS + 4) + 4 * index, 4, "race-value pointer levels")
    for t in C.SPAWN_TABLES:
        n = c.car_count[1]
        at(t + C.SPAWN_RECORD_SIZE * n, C.SPAWN_RECORD_SIZE * (C.SPAWN_RECORDS - n), "spawn records past the car count")
    if not grown:
        at(_u32(rs, C.array_at(arrays, "COURSE_POINTERS") + 4 * 3), C.COURSE_ENTRY_SIZE, "other course ids")
    else:
        for k in (k for k in range(C.COURSE_COUNT, ids) if k != index):
            at(C.record_address(k), C.COURSE_ENTRY_SIZE, "other course ids")
    for name, a in C.ID_ARRAYS.items():
        base = C.array_at(arrays, name)
        entries = ids if grown and not (name in C.WIDE_ARRAYS and base == a.old) else a.entries
        if name in DECODED_ARRAYS:
            for k in range(C.COURSE_COUNT, entries):
                if k != index or name in SHARED_ARRAYS:
                    at(base + a.stride * k, a.stride, "other course ids")
        else:
            at(base, a.stride * entries, "per-id arrays not decoded")
    for t in C.SELECT_TABLES:
        at(t + C.SELECT_RECORD_SIZE * C.SELECT_PANELS, C.SELECT_RECORD_SIZE, "other course ids")
    out += [("data", T.LUMA_COUNT_AT, 4 + 128 * _u32(rs, DATA_BASE + T.LUMA_COUNT_AT), "colour tables"),
            ("data", T.PALETTE_START, T.PALETTE_END - T.PALETTE_START, "colour tables")]
    at(C.array_at(arrays, "RANKING_LAST"), 4 * (ids if grown else 4), "rankings beyond the table pointers")
    if C.ranked(rs):                        # the reset's sixteen-id copies (course.grow_rankings, RANK_HOME)
        at(C.RANK_DEFAULTS, 8 * C.COURSE_IDS, "rankings beyond the table pointers")
        at(C.RANK_BASES, 4 * C.COURSE_IDS * C.MODES, "rankings beyond the table pointers")
    else:
        at(C.RANKING_DEFAULTS, 24, "rankings beyond the table pointers")
        at(C.RANKING_TIME_BASES, 48, "rankings beyond the table pointers")
    for p in c.object_lists[c.map_object.objects]:
        index_ = C.check_rom_object(p.oba, rs)
        if index_ not in c.objects:
            obj = P.decode_object(rs.polygons, index_, p.obc, GEOMETRY_MODE)
            out.append(("polygons", 4 * index_, 4 * obj.word_count, "map objects in polygon ROM"))
    return sorted(out)


def overlaps(a, b) -> list[tuple[tuple, tuple]]:
    """Pairs (x, y), x from `a`, y from `b`, of (region, offset, length, name) intervals that share a byte."""
    out = []
    for region in {t[0] for t in b}:
        ys = sorted(t for t in b if t[0] == region and t[2])
        starts = [y[1] for y in ys]
        reach, m = [], -1                      # reach[j]: the largest end among ys[0..j]
        for y in ys:
            m = max(m, y[1] + y[2])
            reach.append(m)
        for x in a:
            if x[0] != region or not x[2]:
                continue
            j = bisect.bisect_left(starts, x[1] + x[2]) - 1
            while j >= 0 and reach[j] > x[1]:
                if ys[j][1] + ys[j][2] > x[1]:
                    out.append((x, ys[j]))
                j -= 1
    return out


def covered_bytes(cov) -> dict[str, int]:
    """Bytes covered per region: the union of the (region, offset, length, name) intervals."""
    out: dict[str, int] = {}
    for region in sorted({r for r, *_ in cov}):
        total, end = 0, -1
        for _, o, n, _ in sorted(t for t in cov if t[0] == region):
            if o + n > end:
                total += o + n - max(o, end)
                end = o + n
        out[region] = total
    return out


OWN_TABLE_ARRAYS = DECODED_ARRAYS - SHARED_ARRAYS - {"COURSE_POINTERS"}
"""The decoded per-id arrays whose entry names a course's own tables or values: a grown id whose entries here are
entry 0's (Beginner's, as grow_ids copies them) and whose record is Beginner's has no table of its own."""


def _own_tables(rs: RomSet, arrays: dict[str, int], k: int) -> bool:
    """True when grown id `k` has tables of its own: its record (course.record_address) or its entry of an
    OWN_TABLE_ARRAYS array differs from Beginner's."""
    if rs.read_i960(C.record_address(k), C.COURSE_ENTRY_SIZE) != rs.read_i960(C.record_address(0),
                                                                             C.COURSE_ENTRY_SIZE):
        return True
    for name in sorted(OWN_TABLE_ARRAYS):
        a = C.ID_ARRAYS[name]
        if rs.read_i960(C.id_entry(arrays, name, k), a.stride) != rs.read_i960(C.id_entry(arrays, name, 0), a.stride):
            return True
    return False


def own_tables(rs: RomSet, k: int) -> bool:
    """True when course id `k` of the grown set `rs` has tables of its own (`_own_tables`): its slot is filled (a
    converted course, a mirror). CourseError for `rs` not a grown RomSet or `k` not an id 3-15 (bool refused)."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    if not C.grown(rs):
        raise CourseError("own tables: the set is not grown (course.grow_ids): ids 3-15 have no slot")
    k = _int(k, "course id", C.COURSE_COUNT, C.COURSE_IDS - 1, "own tables")
    return _own_tables(rs, C.id_tables(rs), k)


ID_EXTRA_ARRAYS = ("CAMERA_SCRIPTS", "TUNNEL_LIGHTS", "PROGRESS_MASKS")
"""The per-id arrays no decoder reads whose entries name tables a mirror writes for itself (mirror.mirrored_entry):
the pre-race camera script, the tunnel light zones' position table and the progress-mask records. owned_spans adds
their extents for every grown id 3-15 with tables of its own (`extra_spans`)."""


def extra_spans(rs: RomSet, k: int) -> list[tuple[str, int, int, str]]:
    """(region, offset, length, name) of the ID_EXTRA_ARRAYS tables course id `k` names: every record of its camera
    script's play (64 bytes each, the loop it jumps to included: course.camera_script), its tunnel light record's
    tables (the positions, 12 bytes per zone of its TUNNEL_BYTES count, at its fifth word; the zone starts and ends,
    K + 1 u16 for a count K, its first and second; the range, two u16, its third; the zone mask, a u32, its fourth: a reverse
    id 9-10 writes its own, build._reverse_tables) and its progress-mask
    records (course.PROGRESS_MASK_SIZE bytes each) and the end record's two segment words, the 4 bytes 0x17688 reads
    of it (mirror.mirror_masks writes those). CourseError as the readers (a camera script course.camera_script
    refuses)."""
    arrays = C.id_tables(rs)
    out: list[tuple[str, int, int, str]] = []

    def at(addr: int, n: int, name: str) -> None:
        region, off = _where(addr, n, name)
        out.append((region, off, n, name))

    a = C.camera_script(rs, _u32(rs, C.id_entry(arrays, "CAMERA_SCRIPTS", k))).start    # refuses a bad script
    seen: set[int] = set()
    while a not in seen:                    # every record the play reads, the jump records too
        seen.add(a)
        at(a, C.KEYFRAME_SIZE, "camera script")
        time, _, target = struct.unpack("<iIi", rs.read_i960(a, 12))
        a = target & 0xFFFFFFFF if time == C.CAMERA_JUMP else a + C.KEYFRAME_SIZE
    count = rs.read_i960(C.id_entry(arrays, "TUNNEL_BYTES", k), 1)[0]
    starts, ends, rng, mask, positions = struct.unpack("<5I", rs.read_i960(C.id_entry(arrays, "TUNNEL_LIGHTS", k), 20))
    if count:
        at(positions, 12 * count, "tunnel light positions")
        at(starts, 2 * count + 2, "tunnel zone starts")     # K + 1 words each (reverse.reversed_tunnel)
        at(ends, 2 * count + 2, "tunnel zone ends")
    at(rng, 4, "tunnel range")                  # read under the tunnel bit with or without zones (0x7b88)
    at(mask, 4, "tunnel zone mask")
    start = a = _u32(rs, C.id_entry(arrays, "PROGRESS_MASKS", k))
    while struct.unpack("<2H", rs.read_i960(a, 4))[1] < C.PROGRESS_MASK_END:
        if a - start >= C.PROGRESS_MASK_SIZE * 256:
            raise CourseError(f"course {k}: progress masks at {start:#x}: no end record within 256 records")
        a += C.PROGRESS_MASK_SIZE
    at(start, a - start + 4, "progress masks")
    return out


def owned_spans(rs: RomSet) -> list[tuple[str, int, int, str]]:
    """(region, offset, length, owner) of every byte of `rs` that a table or block owns, whatever its value: what an
    allocator or a writer must treat as busy although it may hold 0xff (the converter's free-space search,
    build._usable; mirror.mirrored_set's; tgp.write_free, build._write_free), each through the per-set cache `owned`.
    Sorted; spans overlap. They are:

    - the space reserved for a patch (romset.RESERVED: the select's code and data, grow_ids' tables and records,
      the new ids' SRAM sink), whether written yet or not;
    - every course sheet block a grown set's ids 3-15 name in the custom data area (texture.custom_sheet_blocks:
      a new id's own 1 MB texel block, Desert City's at id 3, every byte of it);
    - every table `coverage` names for ids 0-2, and for each grown id 3-15 with tables of its own (its record or its
      OWN_TABLE_ARRAYS entries not Beginner's: grow_ids' copies own nothing beyond grow_ids' home), with, for such an
      id, the tables of ID_EXTRA_ARRAYS it names (`extra_spans`: a mirror's camera script, tunnel light positions
      and progress masks, mirror.mirrored_entry);
    - on a set carrying the sixteen-entry select (select.patched), the 3D map lists and objects it wrote outside its
      reserved data (select.map_spans: the mirrors of the originals' maps, in the list home and the custom polygon
      area);
    - on a set carrying the Extended title (title.extended_title), its spans (title.title_spans: the copied logo map,
      its chars and the boot upload set's copy in title.TITLE_RUN, and the two words it repointed).

    An owner is the RESERVED owner, the sheet block's, "<table> of course <id>", select.map_spans' or title_spans'.
    CourseError (or the decoders' DecodeError subclasses) for `rs` not a RomSet or a course that does not decode."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    out = [(r, o, n, owner) for r, spans in RESERVED.items() for o, n, owner in spans]
    out += T.custom_sheet_blocks(rs)
    ids = list(range(C.COURSE_COUNT))
    if C.grown(rs):
        arrays = C.id_tables(rs)
        ids += [k for k in range(C.COURSE_COUNT, C.COURSE_IDS) if _own_tables(rs, arrays, k)]
    for k in ids:
        out += [(r, o, n, f"{t} of course {k}") for r, o, n, t in coverage(rs, k)]
        if k >= C.COURSE_COUNT:
            out += [(r, o, n, f"{t} of course {k}") for r, o, n, t in extra_spans(rs, k)]
    if S.patched(rs):
        out += S.map_spans(rs)
    out += TT.title_spans(rs)
    return sorted(out)


_OWNED: list = [None, {}, []]
"""The last `owned` answer: [the set, region -> (k, 2) array of merged [start, end) rows, the owned_spans list]. A
RomSet is immutable, so the same object has the same answer; the allocators (build._usable, mirror._char_run) and the
writers (tgp.write_free, build._write_free) ask for one set many times (each run, each table, each write)."""


def owned(rs: RomSet) -> dict[str, np.ndarray]:
    """owned_spans(rs) merged per region into ascending, disjoint [start, end) rows ((k, 2) int64 arrays), cached for
    the last set asked (or carried to it: `carry_owned`). CourseError (or the decoders' DecodeError subclasses) as
    owned_spans: `rs` not a RomSet, a course that does not decode."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    if _OWNED[0] is not rs:
        spans = owned_spans(rs)
        merged: dict[str, list[list[int]]] = {}
        for r, off, n, _ in spans:          # sorted by region, then offset
            rows = merged.setdefault(r, [])
            if rows and off <= rows[-1][1]:
                rows[-1][1] = max(rows[-1][1], off + n)
            elif n:
                rows.append([off, off + n])
        _OWNED[:] = [rs, {r: np.array(rows, np.int64).reshape(-1, 2) for r, rows in merged.items()}, spans]
    return _OWNED[1]


def owner(rs: RomSet, region: str, offset: int, n: int) -> str | None:
    """The owner (owned_spans' name) of an owned span that bytes offset..offset+n-1 of `region` overlap, else None:
    the first such span in owned_spans' order. Through the cache (`owned`). CourseError for an argument of the wrong
    type (a bool is not an int) and as `owned`."""
    if not isinstance(region, str):
        raise CourseError(f"region {region!r:.60} is not a str")
    for v, what in ((offset, "offset"), (n, "length")):
        if not is_int(v):
            raise CourseError(f"{what} {v!r:.60} is not an int")
    rows = owned(rs).get(region)
    if rows is None or not n or not len(rows):
        return None
    k = int(np.searchsorted(rows[:, 1], offset, side="right"))
    if k == len(rows) or rows[k, 0] >= offset + n:
        return None
    return next(w for r, o, m, w in _OWNED[2] if r == region and o < offset + n and offset < o + m)


def carry_owned(rs: RomSet, out: RomSet) -> None:
    """Record that `out` owns what `rs` owns, when `rs`'s answer is the cached one (else nothing): `out` is `rs` with
    bytes written that no table or block of `rs` owns (tgp.write_free; build._write_free), so none of its tables
    changed a byte. A chain of such writes keeps the owned space it started from: the new tables are owned once a
    pointer names them, and that pointer is written after them (with_patches, which drops the cache), not through
    write_free; between the two a decoder may refuse the half-written set (a ground pool bounded by its area's end
    must be followed by free records: tgp.decode_ground). CourseError for an argument that is not a RomSet."""
    for s in (rs, out):
        if not isinstance(s, RomSet):
            raise CourseError(f"{type(s).__name__} is not a RomSet")
    if _OWNED[0] is rs:
        _OWNED[0] = out
