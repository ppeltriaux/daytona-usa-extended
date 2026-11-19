"""The patched Circuit Select (Phase 8, M4a; panels, MIRROR marking, mirrored maps and the idle wheel: Phase 9 Task 5,
M4b; the row of eight: Phase 11 Task 2, M5b; the courses' names, the originals' over their name box: Phase 12 Task 2,
M7): a row of entries (ROW_LENGTHS: 8, the shipped row since Phase 11, or
16, the frozen rows of Phases 8-10) the wheel scrolls behind the three panel positions of the arcade's screen (Virtua
Racing's ten-course select is the model), the course vote moved to free work RAM for ids 0-15, generated panels on
tile layer 1, locked entries greyed and refused. Layouts and the evidence: docs/formats/model2-course-tables.md,
"Select screen", "The sixteen-entry select" and "The eight-entry select"; docs/spike/2026-11-sixteen.md, "Select";
docs/spike/2026-11-mirrors-live.md, "Panels"; docs/spike/2026-11-courses-4-7.md, "The row of eight".

How the patched select works (all on top of `course.grow_ids`):

- **Entries and the row.** `ENTRIES` (the release row, Phase 12: every course under its own name; Phase 11's row is the
  frozen M5B_ENTRIES) in row order: the three originals in panel order, then Desert City and the four courses M5
  converts; a row of 16 (the frozen M4A_ENTRIES, M4B_ENTRIES, M5A_ENTRIES) adds the eight mirrors in the same order.
  Each has a course id (originals 0-2, Desert City 3, M5's 4-7, the mirror of id `i` is `i + 8`), a name, a label (the
  placeholders' line), a mirror flag, a live flag and a difficulty label (decision 11). The row's length n (ROW_LENGTHS)
  is the number of entries; the tables the select indexes by entry hold n, those it indexes by course id
  (SELECT_PANEL_OF_COURSE, the vote) sixteen whatever n (course.COURSE_IDS).
- **The wheel.** The wheel value `w` (-127..127, `wheel_value`) is cut into n bands by `bands_of(n)` (BANDS for 16,
  16 wide; 32 wide for 8; `entry_of_wheel`): the entry index E goes to 0x501468 (the game's panel half-word, now an
  entry index) and 0x5043de (0x5e(g13)), the entry's course id to 0x501460. The three panel positions show entries
  `first_of(E, n)` .. +2: the cursor entry in the middle position except at the ends of the row (`slot_of`); the red
  cursor frame eases to that position. The idle
  wheel (the controller's ruling, approved by Pascal): the select's state 0 puts the cursor on IDLE_ENTRY (0,
  Beginner) and clears FOLLOW_RAM; the wheel leaves it there while the wheel value stays in the centre band (|w| <=
  IDLE_BAND) and follows the wheel once it has left it (`follow` is the model).
- **Records.** Three per-mode tables of n 44-byte records (`course.SelectPanel` layout) in the select's data
  area: an entry's cursor words are those of the position it is shown in when chosen, word 2 its course id, its
  maps the original panel's for the originals and their mirrors, else the placeholder panel (Beginner's frame round
  a transparent interior). `SELECT_PANEL_OF_COURSE` (grown by grow_ids) is the inverse of the records: course id ->
  entry.
- **Panels.** `draw` (called from 0x2263e4) blits, per position, the entry's panel map into tile layer 0 and its window
  map into tile layer 1 at the same tiles, and sets the window mask (MASK_RAM) over the rows the entry's window covers
  (`window_of`): layer 1 is System 24's "window" tilemap, drawn only where the mask is set and layer 0 not drawn there.
  The originals keep their own art (layer 0, mask clear); on the release row (no difficulty, WINDOW_NAME) their courses'
  names cover their name box on layer 1 (rows 1-5, the mask over those rows only: the band and its lap count stay Sega's
  and blink with the cursor), drawn as a generated panel's name in the original's colours, and the draw takes one more
  instruction for it (`code`'s `name_window`, only on such a row). A generated panel (Desert City, M5's courses, their
  mirrors) is Beginner's interior with its lettering erased, the name in the box (rows 1-5) and the difficulty in the
  band (rows 6-8) in the arcade's own 8 x 8 font, in its scheme's colours (`panel_pixels`); a mirror's band adds MIRROR
  in the accent pen, and a mirror of an original keeps its original's art with MIRROR drawn over the band (the window
  covers rows 6-8 only). The chars are cut from those pixels into ten blocks of 128 from char 0x3b00, one palette per
  block (0x76-0x7f: the original panels' palettes, read from the set, or the new courses' purple); a locked entry's
  tiles set bit 14, which takes palettes 0xf6-0xff, the grey copies. The draw runs in states 0-3 and 6 only; `exit`
  (state 4, the frame before the panels' wipe in state 5) runs it once with the state at 4, which clears the mask and
  blanks layer 1, so the windows do not stay on screen while layer 0 wipes away. The select's upload set (data word
  0x2700004) is copied and extended with the chars and palettes. An own panel (`own_panel`, Phase 12 Task 6: the
  release row's Dreamcast courses) has no window: its records' maps are its own, one pair (normal, blink: the
  placeholder's frame) per game mode, its name in the box and in the band its course's length and laps for that mode
  (`band_pixels`: the word of the original whose lap is nearest in length, `length_panel`; the set's laps; Sega's
  band glyphs), so it shows and wipes as an original does; its chars go to its scheme's block like a window's.
- **3D maps.** The three course-map objects below the panels follow the row: a per-(first, position) record table
  `MAPS` (the position's place, the entry's object list; 0, drawn as nothing, for an entry without a map). A mirror of
  an original shows its original's map quad mirrored (`_mirrored_maps`, `map_records`, `map_spans`); an own panel's
  course is drawn from its road line and a painting of its own scenery in flat polygons on the map quad (`_own_maps`,
  dtools/arcade/selmap.py; Phase 12 Tasks 6 and 7); any other entry has an empty record (Phase 8: Desert City's map
  from its road was Not found then).
- **Start.** The gas pedal on a locked entry does nothing (`step`); when the timer runs out on a locked entry the
  choice moves to IDLE_ENTRY (`timeout`; entry 0 must be live).
- **The vote** (0x32ac, after the select) keeps sixteen half-word slots, one per course id whatever n, in free work
  RAM (`VOTE_RAM`, proven unused by a whole run's memory trace) instead of four at 0x501474.

Errors are SelectError (a DecodeError)."""

from __future__ import annotations

import dataclasses
import math
import struct
from typing import NamedTuple

import numpy as np

from dtools.arcade import course as C
from dtools.arcade import i960
from dtools.arcade import tiles as TL
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import RESERVED, RESERVED_RAM, RomSet
from dtools.arcade.tablebase import CourseError, _u32, is_int


class SelectError(DecodeError):
    pass


# --- the entries ---------------------------------------------------------------------------------------------------

class Entry(NamedTuple):
    course: int         # course id 0-15
    name: str           # drawn on a generated panel's name rows (uppercase ASCII the arcade's font draws; at most
    #                     two lines of NAME_LINE glyphs)
    label: str          # the band's line when the entry has no difficulty ("" for none; "COMING SOON" on the
    #                     placeholders)
    mirror: bool
    live: bool          # selectable; a locked entry shows its panel greyed and refuses the step
    difficulty: str = ""    # the difficulty label (decision 11): the originals' own, the Dreamcast's for its courses,
    #                         a mirror its original's; drawn on the band of a generated panel
    own_panel: bool = False     # Phase 12 Task 6: a generated entry drawn as the originals are, on layer 0 (no window):
    #                             per game mode its own panel, the name in the box and in the band its course's length
    #                             and laps in that mode (`band_pixels`), and its course's map below it (selmap)


ENTRY_COUNT = 16
"""The longest row (ROW_LENGTHS' last) and the number of course ids the select's id-indexed tables hold (ids 0-15,
course.COURSE_IDS: SELECT_PANEL_OF_COURSE and the vote's slots, whatever the row's length)."""
ROW_LENGTHS = (8, 16)
"""The row lengths patch_select takes: 8, the eight courses (the shipped row since Phase 11: mirror mode dropped,
Pascal 2026-11-11), and 16, the frozen rows of Phases 8-10 with the mirrors (M4A_ENTRIES, M4B_ENTRIES, M5A_ENTRIES and
the tests' frozen rows), whose sets keep their bytes."""
ENTRIES: tuple[Entry, ...] = (
    Entry(0, "THREE SEVEN SPEEDWAY", "", False, True),
    Entry(2, "DINOSAUR CANYON", "", False, True),
    Entry(1, "SEA-SIDE STREET GALAXY", "", False, True),
    Entry(3, "DESERT CITY", "", False, True, own_panel=True),
    Entry(4, "NATIONAL PARK SPEEDWAY", "", False, True, own_panel=True),
    Entry(5, "CIRCUIT PIXIE", "", False, True, own_panel=True),
    Entry(6, "RIN RIN RINK", "", False, True, own_panel=True),
    Entry(7, "MERMAID LAKE", "", False, True, own_panel=True),
)
"""The release row (Phase 12 Task 2, M7): the eight courses, each under its own name, in Phase 11's order and ids (the
three originals in panel order, entries 0-2 being panels 0-2, check_entries; Desert City, id 3; the four courses M5
converted, ids 4-7), all live, no mirror (mirror mode is dropped, Pascal 2026-11-11; ids 8-15 stay free for reverse,
M9). Built by dtools/release.py `release_set`; Phase 11's row is M5B_ENTRIES, frozen.
The originals (Beginner, Advanced, Expert) carry their courses' names, Three Seven Speedway, Dinosaur Canyon and
Sea-Side Street Galaxy, and no difficulty: window_of gives them WINDOW_NAME, the name on layer 1 over their own name
box (rows 1-5), their band (course length and laps, which change with the mode) and their maps Sega's in every mode.
No entry carries a difficulty or a label line, CIRCUIT EDITION included (the plan's call: it filled the difficulty
slot, decision 11, the originals show no origin line, and only two of eight would carry one), so the generated panels'
bands hold no label. Since Phase 12 Task 6 entries 3-7 draw their own panels (`own_panel`, Pascal's request): in every
game mode the band shows their course's length and laps as the originals' band does (`band_pixels`), and their
course's map is drawn below the panel (dtools/arcade/selmap.py). Discovery log: docs/spike/2026-11-release.md, "Names
on the select", "Maps and laps" and "Painted maps"."""
M5B_ENTRIES: tuple[Entry, ...] = (
    Entry(0, "BEGINNER", "", False, True, "BEGINNER"),
    Entry(2, "ADVANCED", "", False, True, "ADVANCED"),
    Entry(1, "EXPERT", "", False, True, "EXPERT"),
    Entry(3, "DESERT CITY", "", False, True, "CIRCUIT EDITION"),
    Entry(4, "NATIONAL PARK SPEEDWAY", "", False, True, "CIRCUIT EDITION"),
    Entry(5, "CIRCUIT PIXIE", "", False, True),
    Entry(6, "RIN RIN RINK", "", False, True),
    Entry(7, "MERMAID LAKE", "", False, True),
)
"""Phase 11's row (M5b), frozen: ENTRIES as main 1684e10 left it, the row of eight. A literal, as M5A_ENTRIES is: no
entry is taken from ENTRIES, so a change to the live row (Phase 12 names the entries) moves no byte of the shipped set
(tests/test_courses_mame.py `m5b_set`, M5B_SHA1) or of the M5b playtest set (tools/desert_city_playtest.py `--m5b`)."""
M5A_ENTRIES: tuple[Entry, ...] = (
    Entry(0, "BEGINNER", "", False, True, "BEGINNER"),
    Entry(2, "ADVANCED", "", False, True, "ADVANCED"),
    Entry(1, "EXPERT", "", False, True, "EXPERT"),
    Entry(3, "DESERT CITY", "", False, True, "CIRCUIT EDITION"),
    Entry(4, "NATIONAL PARK SPEEDWAY", "", False, True, "CIRCUIT EDITION"),
    Entry(5, "CIRCUIT PIXIE", "COMING SOON", False, False),
    Entry(6, "RIN RIN RINK", "COMING SOON", False, False),
    Entry(7, "MERMAID LAKE", "", False, True),
    Entry(8, "BEGINNER", "", True, True, "BEGINNER"),
    Entry(10, "ADVANCED", "", True, True, "ADVANCED"),
    Entry(9, "EXPERT", "", True, True, "EXPERT"),
    Entry(11, "DESERT CITY", "", True, True, "CIRCUIT EDITION"),
    Entry(12, "NATIONAL PARK SPEEDWAY", "", True, False, "CIRCUIT EDITION"),
    Entry(13, "CIRCUIT PIXIE", "COMING SOON", True, False),
    Entry(14, "RIN RIN RINK", "COMING SOON", True, False),
    Entry(15, "MERMAID LAKE", "COMING SOON", True, False),
)
"""Phase 10's row (M5a), frozen: ENTRIES as Phase 10 left it (main ef7ad08), 16 entries: the originals, Desert City,
National Park Speedway (entry 4) and Mermaid Lake (7) live, Circuit Pixie and Rin Rin Rink (5, 6) placeholders, the
mirrors 8-11 live, 12-15 locked. For the sets that still carry the mirrors of 4 and 7 under this select: the M5a
playtest set (tools/desert_city_playtest.py --m5a, which keeps the bytes it had) and the tests' Phase 10 final set
(tests/test_mirrors_mame.py final_set). A literal, as M4B_ENTRIES is: no entry is taken from ENTRIES, so Phase 11's
new live row cannot move it (tests/test_select.py test_pinned_rows_frozen)."""
M4B_ENTRIES: tuple[Entry, ...] = (
    Entry(0, "BEGINNER", "", False, True, "BEGINNER"),
    Entry(2, "ADVANCED", "", False, True, "ADVANCED"),
    Entry(1, "EXPERT", "", False, True, "EXPERT"),
    Entry(3, "DESERT CITY", "", False, True, "CIRCUIT EDITION"),
    Entry(4, "NATIONAL PARK SPEEDWAY", "COMING SOON", False, False),
    Entry(5, "CIRCUIT PIXIE", "COMING SOON", False, False),
    Entry(6, "RIN RIN RINK", "COMING SOON", False, False),
    Entry(7, "MERMAID LAKE", "COMING SOON", False, False),
    Entry(8, "BEGINNER", "", True, True, "BEGINNER"),
    Entry(10, "ADVANCED", "", True, True, "ADVANCED"),
    Entry(9, "EXPERT", "", True, True, "EXPERT"),
    Entry(11, "DESERT CITY", "", True, True, "CIRCUIT EDITION"),
    Entry(12, "NATIONAL PARK SPEEDWAY", "COMING SOON", True, False),
    Entry(13, "CIRCUIT PIXIE", "COMING SOON", True, False),
    Entry(14, "RIN RIN RINK", "COMING SOON", True, False),
    Entry(15, "MERMAID LAKE", "COMING SOON", True, False),
)
"""Phase 9's row (M4b), frozen: ENTRIES as Phase 9 left it, National Park Speedway (entry 4), Mermaid Lake (7) and
their mirrors (12, 15) placeholders. For the sets that hold no course at ids 4 and 7: the M4b playtest set
(tools/desert_city_playtest.py --mirrors, which keeps the bytes it had) and the tests' Phase 9 reference sets. A
literal since Phase 10 Task 5 (the Task 4 review's M5): no entry is taken from ENTRIES, so a change to the live row
cannot move it (tests/test_select.py test_pinned_rows_frozen)."""
M4A_ENTRIES: tuple[Entry, ...] = M4B_ENTRIES[:8] + tuple(e._replace(live=False) for e in M4B_ENTRIES[8:])
"""Phase 8's row (M4a): Phase 9's row (M4B_ENTRIES) with every mirror locked, for the M4a playtest set
(tools/desert_city_playtest.py --sixteen), which has no mirrors; Desert City (entry 3) is live, so the set's id 3 is
converted first. Since Phase 9 Task 5 it carries the generated panels and the idle wheel too (the select's code and
data are patch_select's, whatever the row)."""

ORIGINAL_PANEL = {0: 0, 2: 1, 1: 2}
"""Course id -> the original Circuit Select panel showing it (course.PANEL_COURSES inverted): the art an entry of an
original course (not a mirror) keeps."""


# --- the wheel -----------------------------------------------------------------------------------------------------

BANDS = (-112, -96, -80, -64, -48, -32, -16, 0, 16, 32, 48, 64, 80, 96, 112)
"""The row of 16's thresholds (bands_of(16)): entry E is the number of thresholds below the wheel value w: w <= -112
is entry 0, w > 112 entry 15. w runs -127..127 (wheel_value), so the bands are 16 values wide (the last 15)."""


def row_length(n) -> int:
    """`n` checked as a row length (ROW_LENGTHS). SelectError otherwise (bool refused)."""
    if not is_int(n) or n not in ROW_LENGTHS:
        raise SelectError(f"row length {n!r:.60} is not one of ROW_LENGTHS {ROW_LENGTHS}")
    return int(n)


def bands_of(n: int) -> tuple[int, ...]:
    """The n - 1 wheel thresholds of a row of `n` (ROW_LENGTHS): 256 // n wide, symmetric about the centred wheel
    (BANDS for 16; (-96, -64, -32, 0, 32, 64, 96) for 8: the wheel value 0 is entry n / 2 - 1). SelectError for a
    length not in ROW_LENGTHS."""
    n = row_length(n)
    return tuple(256 // n * (k - n // 2) for k in range(1, n))
WHEEL_DEAD = 5                  # 0x57e0-0x5830: |paddle - 128| - 5, clamped to 0..75, times 255 / 150, signed
WHEEL_SPAN = 75


def wheel_value(paddle: int) -> int:
    """The game's wheel value for a paddle byte held still (0x57cc-0x5844: 0x501050; the select averages it with the
    previous frame's, so a held paddle gives this). SelectError for a paddle outside 0..255."""
    p = _int(paddle, "paddle", 0, 255)
    d = p - 128
    a = min(max(abs(d) - WHEEL_DEAD, 0), WHEEL_SPAN) * 255 // 150
    return a if d > 0 else -a


def entry_of_wheel(w: int, n: int = ENTRY_COUNT) -> int:
    """The entry the select picks for wheel value `w` on a row of `n` (ROW_LENGTHS). SelectError for a `w` that is
    not an int or an `n` not in ROW_LENGTHS."""
    w = _int(w, "wheel value", -0x80000000, 0x7FFFFFFF)
    return sum(1 for t in bands_of(n) if w > t)


def paddle_for_entry(k: int, n: int = ENTRY_COUNT) -> int:
    """The paddle byte autoplay.lua holds to pick entry `k` of a row of `n` (DAYTONA_ENTRY; DAYTONA_ROW 8 for a row of
    8): the middle of the paddle values that give it. SelectError for `n` not in ROW_LENGTHS or `k` outside it."""
    n = row_length(n)
    k = _int(k, "entry", 0, n - 1)
    ps = [p for p in range(256) if entry_of_wheel(wheel_value(p), n) == k]
    return ps[len(ps) // 2]


IDLE_ENTRY = 0
"""Where an idle player's select starts and ends (the idle-wheel ruling, plan "Ruling (controller, 2026-11-09): the
idle wheel"; Pascal's approval): the cursor starts on entry 0 (Beginner, as on the stock select) and follows the wheel
only once the wheel value has left the centre band; a timeout on a locked entry races entry 0."""
IDLE_BAND = 16
"""The centre band, |w| <= IDLE_BAND (the controller's ruling, fix round 1: symmetric about the centred wheel), which
the cursor ignores until the wheel has left it: paddle 114-142 (wheel_value -15..15), the dead zone and the first steps
either side. Once it has left, the bands decide as before (on the row of 16 w in (0, 16] is entry 8; on the row of 8
w in (-32, 0] is entry 3, (0, 32] entry 4). The same band for either row length (a wheel value, not an entry)."""
CENTRE_ENTRY = sum(1 for t in BANDS if 0 > t)
"""The entry of the centred wheel (wheel value 0) on the row of 16, entry 7 (on the row of 8, entry 3)."""


def follow(ws, n: int = ENTRY_COUNT) -> list[int]:
    """The entry the patched select's cursor shows on a row of `n` for each of the wheel values `ws` from the select's
    first frame (the code's rule, `code`'s `wheel`): IDLE_ENTRY while the wheel has not yet left the centre band, then
    entry_of_wheel. SelectError for a value that is not an int (bool refused) or `n` not in ROW_LENGTHS."""
    n = row_length(n)
    out, moved = [], False
    for w in ws:
        w = _int(w, "wheel value", -0x80000000, 0x7FFFFFFF)
        moved = moved or abs(w) > IDLE_BAND
        out.append(entry_of_wheel(w, n) if moved else IDLE_ENTRY)
    return out


def first_of(e: int, n: int = ENTRY_COUNT) -> int:
    """The entry shown in the first panel position when entry `e` of a row of `n` (3..ENTRY_COUNT) is chosen: the
    chosen one in the middle position, except at the ends of the row. SelectError for `n` or `e` out of range."""
    n = _int(n, "row length", 3, ENTRY_COUNT)
    e = _int(e, "entry", 0, n - 1)
    return max(0, min(n - 3, e - 1))


def slot_of(e: int, n: int = ENTRY_COUNT) -> int:
    """The panel position (0-2) of chosen entry `e` of a row of `n`. SelectError as first_of."""
    return e - first_of(e, n)


# --- addresses -----------------------------------------------------------------------------------------------------

SELECT_RAM = (RESERVED_RAM[0][0], RESERVED_RAM[0][0] + RESERVED_RAM[0][1])
"""Work RAM the patched select uses: 0x501ae4-0x501fff is read and written by nothing in a whole run (boot,
attract, select, race, game over: the trace in docs/spike/2026-11-sixteen.md, "Select")."""
VOTE_RAM = 0x00501B03           # the vote's count byte; slot of id i: the half-word at VOTE_RAM + 1 + 2 * i
FIRST_RAM = 0x00501B24          # byte: first_of(E), stored by DRAW, read by the 3D-map code
FOLLOW_RAM = 0x00501B25         # byte: 0 until the wheel leaves the centre band this select (the idle wheel)
CODE_AT = 0x0001E800
CODE_FREE = (RESERVED["program"][0][0], RESERVED["program"][0][0] + RESERVED["program"][0][1])
"""The program's 0xff run after the last code (nothing references it), reserved for this code (romset.RESERVED)."""
DATA_AT = C.DATA_BASE + RESERVED["data"][0][0]
DATA_SPAN = RESERVED["data"][0][1]
"""The select's tables, maps and chars: the custom data area's top (custom sets only, CUSTOM_FREE), away from the
id tables (course.ID_HOME, 0x03000000-0x0300157f) and from Task 5's allocations from below."""
CHAR_HOLE = 0x3B00              # an all-transparent char: the placeholder panel's interior
CHAR_FIRST = 0x3B01             # the generated chars, to 0x3fff (no boot or select upload covers 0x3a82-0x3fff;
CHAR_END = 0x4000               # 0x3b00 on, colours 0x76-0x7f, nobody's palettes)
PALETTE_NORMAL = 0x76           # colours of chars 0x3b00-0x3fff, one per block of 128
PALETTE_GREY = 0xF6             # the same chars with tile bit 14
GREY_BIT = 0x4000
FRONT = 0x8000                  # tile bit 15: in front of the polygons
LAYER1_BLANK = 0x0020           # what the select fills tile layer 1 with (0x1bba0): the font's space, behind
LAYER_BYTES = 0x2000
MASK_RAM = 0x0100C000           # tile RAM word 0x6000: per screen row four u16 of one bit per 8-pixel column; a set
MASK_ROWS = (80, 144)           # bit hides layer 0 there and shows layer 1 (segaic24.cpp draw_rect: the odd, "window"
#                                 tilemap draws the inverted mask); the select clears it (0x2728). Rows 80-143 are the
#                                 panels' interiors (tile rows 10-17).
SELECT_UPLOAD_WORD = 0x02700004     # the second select upload set (0x5d68), repointed to the extended one
BLIT = 0x00019304               # (map, tile-RAM address): copies a tile map
VOICE_DEFAULT = 0x00AE216C      # the voice table's fourth word (0x238ec8): the step sound of entries 3-15

# the original select's code and tables (program window)
_STATE = 0x62                   # 0x62(g13): the select's state byte (g13 = 0x504380)
_CURSOR = 0x5E                  # 0x5e(g13): the cursor panel byte, now the entry
PANEL = 0x00501468
COURSE_ID = 0x00501460
_STEP_INPUT = 0x00501068        # the gas pedal's press this frame (0x226004)
_VARIANT = 0x005FE59A
_FRAMES = 0x00540058
_LINKED = 0x005FE588
_LINK_PACKET = 0x00540C00

# --- the panel graphics --------------------------------------------------------------------------------------------

WINDOW_NONE, WINDOW_LABEL, WINDOW_ALL, WINDOW_NAME = 0, 1, 2, 3
"""What an entry's window map (tile layer 1) covers of its panel (`window_of`, the layout data's u8 per entry): nothing
(an original with its difficulty: its own art in layer 0), the label rows 6-8 (a mirror of an original: its original's
art with MIRROR drawn over the band), the interior, rows 1-8 (a generated panel round the placeholder's frame), or the
name rows 1-5 (WINDOW_NAME, Phase 12: an original under its course's name, the name box over its own art, the band and
its lap count, which change with the mode, Sega's). The draw sets the window mask over the same rows: NAME_MASK_ROWS
for rows 1-5 (WINDOW_ALL, WINDOW_NAME), LABEL_MASK_ROWS for rows 6-8 (WINDOW_ALL, WINDOW_LABEL); the WINDOW_NAME branch
is emitted only for a row holding such an entry (`code`'s `name_window`)."""
NAME_MASK_ROWS = (80, 120)      # screen rows of panel rows 1-5 (tile rows 10-14)
LABEL_MASK_ROWS = (120, 144)    # panel rows 6-8 (tile rows 15-17)
WINDOW_ROWS = {WINDOW_LABEL: range(6, 9), WINDOW_ALL: range(1, 9), WINDOW_NAME: range(1, 6)}
"""The panel rows each window kind's map draws (the rest of its map LAYER1_BLANK)."""

ART_PANEL = 0
"""The original panel whose interior the generated panels copy (Beginner's, mode 0, variant 0, the normal map): the
three originals' interiors have the same pens in the same places, only their palettes differ (discovery log,
"Panels")."""
INTERIOR = (8, 72, 8, 136)      # panel pixels of rows 1-8, columns 1-16: y0, y1, x0, x1 (end exclusive)
BOX = (4, 41, 4, 124)           # interior pixels of the name box's fill, inside its four-pixel border
BAND = (45, 64, 0, 128)         # interior pixels of the label band, under the box's border (its corners rounded)
LABEL_TOP = 40                  # the label rows' first interior pixel row (panel row 6)
TEXT, ACCENT, OUTLINE, SHADOW = 5, 7, 8, 6
"""Pens of the original panels' palettes (the same roles in all three): white (the lettering), orange (the lap
count's digits: here the MIRROR mark), black, and the band's dark colour (here the box lettering's outline, as the
original letters' shadow)."""
FONT_FIRST, FONT_END = 0x20, 0x60
"""The arcade's own 8 x 8 font (chars 0x20-0x5f of the boot upload, space to underscore: the ranking and test-screen
lettering): pen 1 the letter, pen 15 its outline. Names and labels are drawn in it (decision 11: the arcade's existing
lettering)."""
FONT_FILL = 1
NAME_CELLS = (1, 15)            # the char columns a name line may use: the box's fill spans columns 1-14 whole
NAME_ROWS = ((2,), (1, 3))      # the char row each name line starts on (two rows each): one line, two lines
BAND_CELLS = (0, 16)            # the char columns a band line may use (0 and 15 hold the band's rounded corners)
BAND_ROWS = (6, 7)              # the band's char rows (interior rows 48-63): a tall line takes both, a short one one
NAME_LINE = NAME_CELLS[1] - NAME_CELLS[0]
LABEL_LINE = BAND_CELLS[1] - BAND_CELLS[0]
UPLOAD_MAX = 288
"""The most chars the select's upload set may add: what Phase 8's names uploaded (287 and the hole), proven not to
cost the select's set-up a video frame. The set-up copies the chars a half-word at a time in the frame the select
opens; 664 chars made it one frame longer, the game's frame counters (0x5010b0, 0x540058) ran one behind from then on
and Beginner's race parted from the stock set's at frame 1339 (discovery log, "Panels"). Glyphs are therefore drawn on
the char grid, one font cell per char (each letter's chars are shared by every panel of its scheme), and only the
used chars of each block are uploaded. Proven in MAME (Beginner from the select call for call with the stock set):
288 chars in one run (Phase 8) and 288 in four runs (fix round 1: this task's four runs, their counts raised to 128,
64, 48, 48); UPLOAD_RUNS_MAX keeps to the four."""
UPLOAD_RUNS_MAX = 4
"""The most char blocks (runs) the select's upload set may add: the count proven with UPLOAD_MAX chars."""
MIRROR_MARK = "MIRROR"
NEW_SCHEME = "new"
"""The colour scheme of a course without an original panel (Desert City, M5's courses and their mirrors): Beginner's
palette with its blues turned purple (`_new_scheme`); the originals' mirrors keep their original's palette."""
BLOCKS = (CHAR_END - CHAR_HOLE) // 128
"""Char blocks of 128 from CHAR_HOLE, one palette each (a tile's colour is its char >> 7, + 0x80 with bit 14):
palettes PALETTE_NORMAL + k, greyed PALETTE_GREY + k."""


def _int(value, name: str, lo: int, hi: int) -> int:
    if not is_int(value) or not lo <= value <= hi:
        raise SelectError(f"{name} {value!r:.60} is outside {lo}..{hi}")
    return int(value)


def _line_ok(text, n: int, what: str) -> str:
    if not isinstance(text, str) or len(text) > n or any(not FONT_FIRST <= ord(ch) < FONT_END for ch in text):
        raise SelectError(f"{what} {text!r:.60} is not a str of at most {n} of the arcade font's characters "
                          f"(uppercase ASCII {FONT_FIRST:#x}-{FONT_END - 1:#x})")
    return text


def name_lines(name: str) -> tuple[str, ...]:
    """`name` wrapped by words into one or two lines of at most NAME_LINE glyphs. SelectError for a name that is not
    a non-empty str of the font's characters, or needs more than two lines."""
    if not isinstance(name, str) or not name.strip():
        raise SelectError(f"name {name!r:.60} is not a non-empty str")
    _line_ok(name, 2 * NAME_LINE + 1, "name")
    lines: list[str] = []
    for word in name.split():
        if len(word) > NAME_LINE:
            raise SelectError(f"name {name!r}: the word {word!r} is longer than {NAME_LINE} glyphs")
        if lines and len(lines[-1]) + 1 + len(word) <= NAME_LINE:
            lines[-1] += " " + word
        else:
            lines.append(word)
    if len(lines) > 2:
        raise SelectError(f"name {name!r} needs {len(lines)} lines of {NAME_LINE} glyphs; the panel holds 2")
    return tuple(lines)


def is_original(e: Entry) -> bool:
    """True for an entry that keeps an original panel's art: not a mirror, course id 0-2. SelectError for a
    non-Entry."""
    if not isinstance(e, Entry):
        raise SelectError(f"{e!r:.60} is not an Entry")
    course = _int(e.course, "course id", 0, ENTRY_COUNT - 1)
    if not isinstance(e.mirror, bool):
        raise SelectError(f"mirror {e.mirror!r:.60} is not a bool")
    return not e.mirror and course in ORIGINAL_PANEL


def _mirror_of_original(e: Entry) -> bool:
    return not is_original(e) and e.mirror and e.course - MIRROR_OFFSET in ORIGINAL_PANEL


MIRROR_OFFSET = 8               # a mirror's course id is its original's + 8 (mirror.MIRROR_ID_OFFSET)


def window_of(e: Entry) -> int:
    """WINDOW_NONE for an original with a difficulty (every original of the frozen rows: its own art, word and all)
    and for an entry drawing its own panel (`own_panel`: its name and band are in its layer-0 maps, per mode),
    WINDOW_NAME for an original without one (the release row: its course's name over its name box), WINDOW_LABEL for a
    mirror of an original, WINDOW_ALL for every other entry. SelectError as is_original."""
    if is_original(e):
        return WINDOW_NONE if e.difficulty else WINDOW_NAME
    if e.own_panel is True:
        return WINDOW_NONE
    return WINDOW_LABEL if _mirror_of_original(e) else WINDOW_ALL


def scheme_of(e: Entry):
    """The palette an entry's panel takes: its original panel (0-2) for an original or a mirror of one, NEW_SCHEME
    otherwise. SelectError as is_original."""
    if is_original(e):
        return ORIGINAL_PANEL[e.course]
    return ORIGINAL_PANEL[e.course - MIRROR_OFFSET] if _mirror_of_original(e) else NEW_SCHEME


def band_lines(e: Entry) -> tuple[tuple[str, int, int], ...]:
    """The label band's lines of an entry's window as (text, vertical scale, pen): a mirror of an original MIRROR,
    twice as tall, in the accent pen (its original's art names the difficulty); a generated mirror its difficulty (or
    its label) and MIRROR under it, one line each; any other generated entry its difficulty (or its label) twice as
    tall; nothing for an entry with neither. SelectError for an original (its band is its own art: no window, or
    WINDOW_NAME's over the name rows only) or a malformed entry."""
    w = window_of(e)
    if w in (WINDOW_NONE, WINDOW_NAME):
        raise SelectError(f"course {e.course}: no window over its band (an original keeps its own; an own_panel "
                          f"entry's is in its layer-0 maps, `band_pixels`)")
    if w == WINDOW_LABEL:
        return ((MIRROR_MARK, 2, ACCENT),)
    line = _line_ok(e.difficulty, LABEL_LINE, "difficulty") or _line_ok(e.label, LABEL_LINE, "label")
    if e.mirror:
        return ((line, 1, TEXT), (MIRROR_MARK, 1, ACCENT)) if line else ((MIRROR_MARK, 2, ACCENT),)
    return ((line, 2, TEXT),) if line else ()


class PanelArt(NamedTuple):
    """What the generated panels are drawn from, read from the set's Circuit Select (course.select_assets): the
    original panels' interior (ART_PANEL's) and each original's label rows with their lettering erased, the arcade's
    font and the colour schemes (each original panel's palette, NEW_SCHEME's, and the grey every block's bit-14 copy
    takes)."""
    interior: np.ndarray                    # (64, 128) pens of rows 1-8
    labels: dict[int, np.ndarray]           # original panel -> (24, 128) pens of its rows 6-8
    font: dict[str, np.ndarray]             # char -> (8, 8) pens
    schemes: dict                           # 0, 1, 2, NEW_SCHEME -> 16 palette words
    grey: tuple[int, ...]
    bands: dict                             # (original panel, mode) -> BandArt: its band in that game mode (Task 6)


def _rgb5(v: int) -> tuple[int, int, int]:
    return v & 31, (v >> 5) & 31, (v >> 10) & 31


def _new_scheme(words: tuple[int, ...]) -> tuple[int, ...]:
    """`words` (Beginner's) with the box and band colours (every entry with more blue than red and green) turned
    purple: red raised to three quarters of the blue."""
    out = []
    for v in words:
        r, g, b = _rgb5(v)
        out.append(v & 0x8000 | b << 10 | g << 5 | max(r, 3 * b // 4) if b > max(r, g) else v)
    return tuple(out)


def _grey(words: tuple[int, ...]) -> tuple[int, ...]:
    """`words` as dim greys (six tenths of the luma): the locked entries' colours."""
    out = []
    for v in words:
        r, g, b = _rgb5(v)
        y = (299 * r + 587 * g + 114 * b) // 1000 * 6 // 10
        out.append(v & 0x8000 | y << 10 | y << 5 | y)
    return tuple(out)


def palette_of(rs: RomSet, colour: int) -> tuple[int, ...]:
    """The 16 palette words of tile colour `colour` on Circuit Select: the last write of the boot and select upload
    sets (course.select_assets' uploads). SelectError for a colour outside 0..0xff, or one no upload writes in
    full."""
    colour = _int(colour, "colour", 0, 0xFF)
    try:
        uploads = C.select_assets(rs).uploads
    except DecodeError as e:
        raise SelectError(f"the select screen's uploads: {e}") from e
    words: dict[int, int] = {}
    for u in uploads:
        for run in u.palette:
            first = (run.dst - TL.PALETTE_RAM[0]) // 2
            words.update({first + i: w for i, w in enumerate(run.words)})
    if any(16 * colour + i not in words for i in range(16)):
        raise SelectError(f"colour {colour:#x}: no upload writes its 16 palette entries")
    return tuple(words[16 * colour + i] for i in range(16))


def panel_art(rs: RomSet) -> PanelArt:
    """The PanelArt of `rs` (a stock, grown or patched set: the original panels and the boot font are read as
    course.select_assets finds them). SelectError for a set whose panels or font do not decode, a panel map in more
    than one colour, or a font char outside the boot upload."""
    if not isinstance(rs, RomSet):
        raise SelectError(f"{type(rs).__name__} is not a RomSet")
    try:
        a = C.select_assets(rs)

        def char(ch: int) -> np.ndarray:
            if ch not in a.char_source:
                raise SelectError(f"char {ch:#x} is not uploaded on Circuit Select")
            return TL.decode_chars(rs.read_i960(a.char_source[ch], TL.CHAR_SIZE))[0]

        def interior(m: TL.TileMap) -> np.ndarray:
            img = np.zeros((8 * m.rows, 8 * m.cols), np.uint8)
            for r in range(m.rows):
                for c in range(m.cols):
                    img[8 * r:8 * r + 8, 8 * c:8 * c + 8] = char(m.tile(r, c) & TL.CHAR_MASK)
            y0, y1, x0, x1 = INTERIOR
            return img[y0:y1, x0:x1].copy()

        pens, schemes, strips = {}, {}, {}
        table = C.course_table(rs)
        for p in range(C.SELECT_PANELS):
            m = TL.tile_map(rs, a.panels[0][p].maps[0])
            colours = {(t >> 7) & 0xFF for t in m.tiles()}
            if len(colours) != 1:
                raise SelectError(f"panel {p}'s map {a.panels[0][p].maps[0]:#x} uses colours {sorted(colours)}, "
                                  f"not one")
            pens[p] = interior(m)
            schemes[p] = palette_of(rs, colours.pop())
            for mode in range(C.MODES):
                strips[(p, mode)] = (interior(TL.tile_map(rs, a.panels[mode][p].maps[0]))[LABEL_TOP:],
                                     C.laps(rs, table[C.PANEL_COURSES[p]], mode))
        font = {chr(c): char(c) for c in range(FONT_FIRST, FONT_END)}
    except DecodeError as e:
        if isinstance(e, SelectError):
            raise
        raise SelectError(f"the select screen's art: {e}") from e

    def erased(img: np.ndarray) -> np.ndarray:
        out = img.copy()
        y0, y1, x0, x1 = BAND
        band = out[y0:y1, x0:x1]
        band[(band == TEXT) | (band == ACCENT)] = out[y0, x0 + 8]
        return out
    interior = erased(pens[ART_PANEL])
    y0, y1, x0, x1 = BOX
    interior[y0:y1, x0:x1] = interior[y0, x0]
    labels = {p: erased(img)[LABEL_TOP:] for p, img in pens.items()}
    schemes[NEW_SCHEME] = _new_scheme(schemes[ART_PANEL])
    bands = {key: _band_art(strip, laps, f"panel {key[0]}'s band in mode {key[1]}")
             for key, (strip, laps) in strips.items()}
    return PanelArt(interior, labels, font, schemes, _grey(schemes[ART_PANEL]), bands)


def _text(img: np.ndarray, art: PanelArt, text: str, row: int, scale: int, pen: int, outline: int,
          cells: tuple[int, int]) -> None:
    """`text` in the arcade's font into `img` on the char grid: from char row `row`, each glyph one char column (8
    pixels), centred in the columns `cells` (rounded left), each font row `scale` pixel rows tall: the letters in
    `pen`, their outlines in `outline`."""
    left = cells[0] + (cells[1] - cells[0] - len(text)) // 2
    for i, ch in enumerate(text):
        g = np.repeat(art.font[ch], scale, 0)
        sub = img[8 * row:8 * row + 8 * scale, 8 * (left + i):8 * (left + i) + 8]
        sub[(g != 0) & (g != FONT_FILL)] = outline
        sub[g == FONT_FILL] = pen


def panel_pixels(art: PanelArt, e: Entry) -> np.ndarray:
    """The pens of an entry's window: WINDOW_ALL (64, 128), rows 1-8 (the copied interior, the name centred in the
    box, each font row two pixel rows tall, on char rows 2-3 for one line, 1-2 and 3-4 for two; the band's lines,
    `band_lines`, on char rows 6-7); WINDOW_NAME (40, 128), rows 1-5 of the copied interior (the name box, its
    lettering erased) with the name placed as on a generated panel, drawn in the original's scheme (the three
    originals' boxes have the same pens, PanelArt); WINDOW_LABEL (24, 128), rows 6-8 of its original's art with the
    band's lines. Every glyph sits on the char grid (`_text`). SelectError for an original with a difficulty (its own
    art) or a malformed entry."""
    if not isinstance(art, PanelArt):
        raise SelectError(f"{art!r:.60} is not a PanelArt")
    w = window_of(e)
    if w == WINDOW_NONE:
        raise SelectError(f"course {e.course}: no window (an original with its difficulty keeps its own art; an "
                          f"own_panel entry draws on layer 0, `own_pixels`)")
    lines = () if w == WINDOW_NAME else band_lines(e)
    if w == WINDOW_LABEL:
        img, shift = art.labels[scheme_of(e)].copy(), LABEL_TOP // 8
    else:
        img, shift = art.interior[:LABEL_TOP if w == WINDOW_NAME else None].copy(), 0
        names = name_lines(e.name)
        for ln, row in zip(names, NAME_ROWS[len(names) - 1]):
            _text(img, art, ln, row, 2, TEXT, SHADOW, NAME_CELLS)
    row = BAND_ROWS[0] - shift
    for text, scale, pen in lines:
        _text(img, art, text, row, scale, pen, OUTLINE, BAND_CELLS)
        row += scale
    return img


# --- the band: course length and laps (Phase 12 Task 6) ------------------------------------------------------------

LENGTH_PANELS = (0, 1, 2)
"""The original panels whose bands give an own panel its length word: SHORT (panel 0, Three Seven Speedway), MEDIUM
(1, Dinosaur Canyon), LONG (2, Sea-Side Street Galaxy). `length_panel` picks the one whose course's lap is nearest in
length."""
BAND_TEXT_TOP = BAND[0] - LABEL_TOP
"""The first row of a band strip (rows 6-8, LABEL_TOP on) the band's lettering may use: the band's own rows."""
BAND_WORD_GAP = 3
"""Pixels between the length word and the times sign at the narrowest Sega draws them (SHORT x 20 and x 80, MEDIUM x
40: 3; LONG x 20: 4). `band_shift` moves a length's word left, in every band of that length on the row, until each
band keeps at least this gap."""
DIGIT_GAP = 1
"""Pixels between two digits `band_pixels` composes from Sega's two-digit glyphs (Sega's own pairs leave 1 or 2)."""


class BandArt(NamedTuple):
    """One original band (rows 6-8 of an original panel's normal map, variant 0, in one game mode) cut into its parts
    by columns of lettering (the band's rows only): the length word, the times sign, the lap count's digits (the
    accent pen: one column run each) and LAPS. Columns are [lo, hi), in the band's 128 pixels."""
    pens: np.ndarray                    # (24, 128)
    laps: int                           # the lap count it shows: its course's (the stock set's) in that mode
    word: tuple[int, int]
    times: tuple[int, int]
    digits: tuple[tuple[int, int], ...]
    laps_word: tuple[int, int]


def _runs(cols) -> list[tuple[int, int]]:
    out, c = [], 0
    while c < len(cols):
        if cols[c]:
            lo = c
            while c < len(cols) and cols[c]:
                c += 1
            out.append((lo, c))
        else:
            c += 1
    return out


def _band_art(strip: np.ndarray, laps: int, what: str) -> BandArt:
    """`strip` (an original's rows 6-8) cut as BandArt: the lettering's column runs, those holding the accent pen the
    digits (as many as `laps` has), the runs before them the word's letters and, last, the times sign, the runs after
    them LAPS. SelectError when the band is not laid out that way."""
    band = strip[BAND_TEXT_TOP:]
    runs = _runs(np.isin(band, (TEXT, ACCENT)).any(0))
    digits = [r for r in runs if (band[:, r[0]:r[1]] == ACCENT).any()]
    if len(digits) != len(str(laps)):
        raise SelectError(f"{what}: {len(digits)} digit runs for its {laps} laps")
    pre = [r for r in runs if r[1] <= digits[0][0]]
    post = [r for r in runs if r[0] >= digits[-1][1]]
    if len(pre) < 2 or not post or len(pre) + len(digits) + len(post) != len(runs):
        raise SelectError(f"{what}: its lettering is not a word, a times sign, {len(digits)} digits and LAPS")
    return BandArt(strip, laps, (pre[0][0], pre[-2][1]), pre[-1], tuple(digits), (post[0][0], post[-1][1]))


def _paste(dst: np.ndarray, src: np.ndarray, lo: int, hi: int, at: int) -> None:
    """The lettering (TEXT, ACCENT) of `src`'s band rows, columns lo..hi, into `dst` at column `at`."""
    part = src[BAND_TEXT_TOP:, lo:hi]
    ink = np.isin(part, (TEXT, ACCENT))
    dst[BAND_TEXT_TOP:, at:at + hi - lo][ink] = part[ink]


def _digit_block(art: PanelArt, laps: int, length: int) -> np.ndarray:
    """The lap count's digits as Sega draws them: the digits of a band showing `laps` when one does (its columns,
    whole; `length`'s own band first: Sega draws the same number a little differently on different panels), else
    each digit's glyph from a band with as many digits (a single digit is drawn wide, two narrow), DIGIT_GAP apart.
    SelectError for a digit no band of that width draws."""
    text = str(laps)
    for _, b in sorted(art.bands.items(), key=lambda kv: (kv[0][0] != length, kv[0])):
        if b.laps == laps:
            return b.pens[:, b.digits[0][0]:b.digits[-1][1]]
    parts = []
    for ch in text:
        src = next(((b, b.digits[str(b.laps).index(ch)]) for _, b in sorted(art.bands.items())
                    if len(str(b.laps)) == len(text) and ch in str(b.laps)), None)
        if src is None:
            raise SelectError(f"laps {laps}: no original band draws a {ch} among {len(text)}-digit lap counts "
                              f"(Sega's bands show {sorted({b.laps for b in art.bands.values()})})")
        b, (lo, hi) = src
        parts.append(b.pens[:, lo:hi])
    gap = np.zeros((parts[0].shape[0], DIGIT_GAP), np.uint8)
    return np.concatenate([x for p in parts for x in (p, gap)][:-1], 1)


def _band_layout(art: PanelArt, length: int, laps: int) -> tuple[BandArt, np.ndarray, int, int]:
    """(source band, digit block, the digits' first column, the times sign's first column) of `length`'s band for
    `laps`: laid out on the length panel's band of a mode whose lap count has as many digits, the digits ending where
    its digits end and the times sign before them at its gap. SelectError for a length that is not LENGTH_PANELS', a
    lap count outside 1..course.MAX_LAPS or with more digits than any band of that length."""
    if not isinstance(art, PanelArt) or not art.bands:
        raise SelectError(f"{art!r:.60} is not a PanelArt with the originals' bands")
    if not is_int(length) or length not in LENGTH_PANELS:
        raise SelectError(f"length {length!r:.60} is not one of LENGTH_PANELS {LENGTH_PANELS}")
    laps = _int(laps, "laps", 1, C.MAX_LAPS)
    src = next((b for (p, _), b in sorted(art.bands.items()) if p == length and len(str(b.laps)) == len(str(laps))),
               None)
    if src is None:
        raise SelectError(f"laps {laps}: panel {length}'s bands show no {len(str(laps))}-digit lap count to lay it "
                          f"out on")
    block = _digit_block(art, laps, length)
    d0 = src.digits[-1][1] - block.shape[1]
    t0 = d0 - (src.digits[0][0] - src.times[1]) - (src.times[1] - src.times[0])
    return src, block, d0, t0


def band_shift(art: PanelArt, length: int, laps) -> int:
    """The pixels `length`'s word moves left so that every band of that length for the lap counts `laps` (an iterable
    of ints) keeps BAND_WORD_GAP before its times sign (0 when Sega's place does): one place for the word in all of
    them, so they share its chars. SelectError as `_band_layout`, or for an empty `laps`."""
    need = [_band_layout(art, length, n) for n in laps]
    if not need:
        raise SelectError("band_shift: no lap counts")
    return max(0, max(src.word[1] + BAND_WORD_GAP - t0 for src, _, _, t0 in need))


def band_pixels(art: PanelArt, length: int, laps: int, shift: int = 0) -> np.ndarray:
    """The pens of an own panel's rows 6-8 (24, 128): the box's lower edge and the band of the copied interior
    (PanelArt.interior, lettering erased) with `length`'s word (the original panel LENGTH_PANELS[length]'s, moved
    `shift` pixels left: `band_shift`), the times sign, `laps` in Sega's digits (`_digit_block`) and LAPS, each from
    Sega's bands (`_band_layout`), in their pens (TEXT, the digits ACCENT). With `shift` 0 and a lap count an original
    shows, the original's band itself. SelectError as `_band_layout`, for a `shift` that is not an int 0..127, or a
    word that would leave the band or come nearer the times sign than BAND_WORD_GAP."""
    src, block, d0, t0 = _band_layout(art, length, laps)
    shift = _int(shift, "shift", 0, 127)
    w0, w1 = src.word[0] - shift, src.word[1] - shift
    if w0 < 1 or w1 + BAND_WORD_GAP > t0:
        raise SelectError(f"length {length}, laps {laps}: the word at {w0}..{w1} does not fit before the times sign "
                          f"at {t0} (shift {shift}; band_shift gives the room)")
    out = art.interior[LABEL_TOP:].copy()
    _paste(out, src.pens, *src.word, w0)
    _paste(out, src.pens, *src.times, t0)
    _paste(out, block, 0, block.shape[1], d0)
    _paste(out, src.pens, *src.laps_word, src.laps_word[0])
    return out


def road_length(rs: RomSet, course: int) -> float:
    """The length of course `course`'s lap in world units: its road line's centres (course.RoadPoint.centre) joined
    as the closed loop the race runs. SelectError for a course id the set does not hold or a road that does not
    decode."""
    try:
        table = C.course_table(rs)
        course = _int(course, "course id", 0, len(table) - 1)
        cs = [p.centre for p in C.road_line(rs, table[course])]
    except DecodeError as e:
        if isinstance(e, SelectError):
            raise
        raise SelectError(f"course {course!r:.20}'s road: {e}") from e
    return sum(math.dist(cs[k - 1], cs[k]) for k in range(len(cs)))


def length_panel(rs: RomSet, course: int) -> int:
    """The original panel (LENGTH_PANELS) whose course's lap is nearest in length to course `course`'s
    (`road_length`; a tie to the shorter): its band's word is `course`'s length word (the controller's call: SHORT,
    MEDIUM, LONG by the nearest original). SelectError as `road_length`."""
    mine = road_length(rs, course)
    return min(LENGTH_PANELS, key=lambda p: (abs(road_length(rs, C.PANEL_COURSES[p]) - mine), p))


def own_pixels(art: PanelArt, e: Entry, length: int, laps: int, shift: int = 0) -> np.ndarray:
    """The pens of an own panel's rows 1-8 (64, 128) in one game mode: rows 1-5 the name in the box as on a generated
    panel (`panel_pixels`), rows 6-8 `band_pixels(art, length, laps, shift)`. SelectError for an entry that is not an
    own_panel one, or as band_pixels."""
    if not isinstance(e, Entry) or e.own_panel is not True or window_of(e) != WINDOW_NONE:
        raise SelectError(f"{e!r:.60} is not an own_panel entry")
    box = panel_pixels(art, e._replace(own_panel=False))[:LABEL_TOP]
    return np.concatenate([box, band_pixels(art, length, laps, shift)])


class Graphics(NamedTuple):
    chars: np.ndarray               # (n, 8, 8) pens from CHAR_HOLE (all 0), block by block (128 each; unused: 0)
    name_maps: tuple[tuple[int, ...] | None, ...]   # per entry: the 10 x 18 window map, None without a window
    hole: int                       # the placeholder interior's tile
    windows: tuple[int, ...]        # per entry: window_of
    palettes: tuple[tuple[int, ...], ...]   # per block (BLOCKS): its scheme's 16 words (palette PALETTE_NORMAL + k)
    grey: tuple[int, ...]           # every block's greyed palette (PALETTE_GREY + k)
    schemes: dict                   # PanelArt.schemes
    runs: tuple[tuple[int, int], ...] = ()      # (first char, count) of each block's used chars: what is uploaded
    panels: tuple = ()              # per entry: None, or for an own_panel entry per game mode its layer-0 maps (normal,
    #                                 blink: 10 x 18 tiles each, the placeholder's frame round its own interior)
    bands: tuple = ()               # per entry: None, or for an own_panel entry per game mode (length panel, laps,
    #                                 word shift)

    def uploaded(self) -> int:
        return sum(n for _, n in self.runs)


def _own_bands(rs: RomSet, art: PanelArt, entries) -> dict[int, tuple[tuple[int, int, int], ...]]:
    """{entry index: per game mode (length panel, laps, word shift)} for the own_panel entries: the length panel by
    road length (`length_panel`), the laps the set holds for the course in that mode (course.laps), the shift one per
    length over every own band of the row (`band_shift`). SelectError for a course that does not decode."""
    own = [k for k, e in enumerate(entries) if e.own_panel]
    if not own:
        return {}
    try:
        table = C.course_table(rs)
        laps = {k: tuple(C.laps(rs, table[entries[k].course], m) for m in range(C.MODES)) for k in own}
    except DecodeError as e:
        raise SelectError(f"the own panels' laps: {e}") from e
    length = {k: length_panel(rs, entries[k].course) for k in own}
    shift = {p: band_shift(art, p, sorted({n for k in own if length[k] == p for n in laps[k]}))
             for p in set(length.values())}
    return {k: tuple((length[k], n, shift[length[k]]) for n in laps[k]) for k in own}


def graphics(rs: RomSet, entries) -> Graphics:
    """The generated chars, each entry's window map and each own_panel entry's layer-0 maps: per `window_of` the
    panel_pixels of its rows cut into chars, tiles in front (bit 15) and greyed (bit 14) when the entry is locked, the
    rest of the map LAYER1_BLANK; for an own_panel entry (no window) per game mode `own_pixels` (its band's length,
    laps and shift `_own_bands`) cut the same way into rows 1-8 of a map whose frame is the placeholder's (Beginner's
    frame, normal and blink: the record's two maps). The chars are packed, in row order (an own panel's mode by mode),
    into BLOCKS blocks of 128 from CHAR_HOLE (char CHAR_HOLE, all transparent, the placeholder's interior), each
    entry's into blocks of its scheme (the block's palette), live or locked, so the chars and palettes do not depend
    on the live flags (a course's chars stay where they are when it goes live, and Phase 8's row uploads what the full
    row does); identical chars of a block are stored once, and each block's used chars are one upload run (`runs`).
    SelectError for entries that are not check_entries', a set whose art does not decode (panel_art) or whose own
    panels' courses do not (`_own_bands`), chars that do not fit the blocks, or more than UPLOAD_MAX chars or
    UPLOAD_RUNS_MAX runs to upload."""
    entries = check_entries(entries)
    art = panel_art(rs)
    blocks: list[list] = [[None, [np.zeros((8, 8), np.uint8)]]] + [[None, []] for _ in range(BLOCKS - 1)]
    where: dict[tuple[int, bytes], int] = {(0, bytes(64)): CHAR_HOLE}
    maps: list[tuple[int, ...] | None] = [None] * len(entries)
    panels: list = [None] * len(entries)
    windows = tuple(window_of(e) for e in entries)
    bands = _own_bands(rs, art, entries)

    def cut(k: int, img: np.ndarray, rows, tiles: list[int]) -> None:
        """`img`'s char cells into `tiles` (a 10 x 18 map) at panel rows `rows`, columns 1-16."""
        e = entries[k]
        scheme = scheme_of(e)
        flag = FRONT | (0 if e.live else GREY_BIT)
        for i, r in enumerate(rows):
            for c in range(16):
                blk = np.ascontiguousarray(img[8 * i:8 * i + 8, 8 * c:8 * c + 8])
                key = blk.tobytes()
                mine = [b for b in range(BLOCKS) if blocks[b][0] == scheme]
                ch = CHAR_HOLE if not blk.any() else next((where[(b, key)] for b in mine if (b, key) in where), None)
                if ch is None:
                    room = [b for b in mine + [b for b in range(BLOCKS) if blocks[b][0] is None]
                            if len(blocks[b][1]) < 128]
                    if not room:
                        raise SelectError(f"entry {k}: the generated chars do not fit the {BLOCKS} blocks of 128 from "
                                          f"{CHAR_HOLE:#x}")
                    b = room[0]
                    blocks[b][0] = scheme
                    ch = CHAR_HOLE + 128 * b + len(blocks[b][1])
                    blocks[b][1].append(blk)
                    where[(b, key)] = ch
                tiles[r * C.PANEL_COLS + 1 + c] = flag | ch

    frames = None
    for k in range(len(entries)):
        e, w = entries[k], windows[k]
        if k in bands:
            if frames is None:
                try:
                    a = C.select_assets(rs)
                    frames = tuple(TL.tile_map(rs, a.panels[0][0].maps[i]).tiles() for i in (0, 1))
                except DecodeError as err:
                    raise SelectError(f"the placeholder's frame: {err}") from err
            per_mode = []
            for length, laps, shift in bands[k]:
                tiles = list(frames[0])
                cut(k, own_pixels(art, e, length, laps, shift), range(1, 9), tiles)
                inner = {r * C.PANEL_COLS + c for r in range(1, 9) for c in range(1, 17)}
                per_mode.append((tuple(tiles), tuple(tiles[i] if i in inner else frames[1][i]
                                                     for i in range(len(tiles)))))
            panels[k] = tuple(per_mode)
            continue
        if w == WINDOW_NONE:
            continue
        tiles = [LAYER1_BLANK] * (C.PANEL_ROWS * C.PANEL_COLS)
        cut(k, panel_pixels(art, e), WINDOW_ROWS[w], tiles)
        maps[k] = tuple(tiles)
    last = max(b for b in range(BLOCKS) if blocks[b][1])
    chars = []
    for b in range(last + 1):
        cs = blocks[b][1]
        chars += cs + ([np.zeros((8, 8), np.uint8)] * (128 - len(cs)) if b < last else [])
    palettes = tuple(art.schemes[blocks[b][0] if blocks[b][0] is not None else ART_PANEL] for b in range(BLOCKS))
    runs = tuple((CHAR_HOLE + 128 * b, len(blocks[b][1])) for b in range(BLOCKS) if blocks[b][1])
    if sum(n for _, n in runs) > UPLOAD_MAX or len(runs) > UPLOAD_RUNS_MAX:
        raise SelectError(f"{sum(n for _, n in runs)} generated chars in {len(runs)} runs to upload: more than "
                          f"UPLOAD_MAX ({UPLOAD_MAX}) or UPLOAD_RUNS_MAX ({UPLOAD_RUNS_MAX}), what is proven not to cost "
                          f"the select's set-up a video frame")
    return Graphics(np.array(chars, np.uint8), tuple(maps), FRONT | CHAR_HOLE, windows, palettes, art.grey,
                    art.schemes, runs, tuple(panels), tuple(bands.get(k) for k in range(len(entries))))


def check_entries(entries) -> tuple[Entry, ...]:
    """`entries` as a tuple of n Entry, n one of ROW_LENGTHS (8 or 16), checked: course ids a permutation of 0..n - 1
    (bool refused), names, labels and difficulties drawable (name_lines; at most LABEL_LINE of the font's
    characters), mirror and live bools, no mirror entry in a row of 8 (the eight courses: mirror mode is dropped),
    entries 0-2 the original panels 0-2 (Beginner, Advanced, Expert, not mirrors), entry IDLE_ENTRY live (where the
    idle player and the timer's fallback end), a mirror's difficulty its original's (decision 11), own_panel a bool
    and only on a generated entry that is not a mirror and has no difficulty or label (Phase 12 Task 6). SelectError
    otherwise."""
    if not isinstance(entries, (tuple, list)) or len(entries) not in ROW_LENGTHS:
        raise SelectError(f"entries {entries!r:.60} is not a sequence of 8 or 16 Entry (ROW_LENGTHS {ROW_LENGTHS})")
    n = len(entries)
    out = []
    for k, e in enumerate(entries):
        if not isinstance(e, Entry):
            raise SelectError(f"entry {k} {e!r:.60} is not an Entry")
        _int(e.course, f"entry {k}: course id", 0, n - 1)
        name_lines(e.name)
        _line_ok(e.label, LABEL_LINE, f"entry {k}: label")
        _line_ok(e.difficulty, LABEL_LINE, f"entry {k}: difficulty")
        for f in ("mirror", "live", "own_panel"):
            if not isinstance(getattr(e, f), bool):
                raise SelectError(f"entry {k}: {f} {getattr(e, f)!r:.60} is not a bool")
        if e.own_panel and (is_original(e) or e.mirror or e.difficulty or e.label):
            raise SelectError(f"entry {k}: own_panel is for a generated entry (not an original, not a mirror) with no "
                              f"difficulty or label: its band shows its length and laps")
        out.append(e)
    if sorted(e.course for e in out) != list(range(n)):
        raise SelectError(f"course ids {[e.course for e in out]} are not a permutation of 0..{n - 1}")
    if n < ENTRY_COUNT and (mirrors := [k for k, e in enumerate(out) if e.mirror]):
        raise SelectError(f"entries {mirrors} are mirrors: a row of {n} is the eight courses (mirror mode is dropped)")
    if not out[IDLE_ENTRY].live:
        raise SelectError(f"entry {IDLE_ENTRY} is locked: it is where the idle player and the timer's fallback end")
    for k in range(C.SELECT_PANELS):
        if out[k].mirror or out[k].course != C.PANEL_COURSES[k]:
            raise SelectError(f"entry {k} is course {out[k].course} (mirror {out[k].mirror}), not the original panel "
                              f"{k}'s course {C.PANEL_COURSES[k]}: entries 0-2 must be panels 0-2, so the code that "
                              f"still indexes by panel (and the linked path's copy at 0x5718) sees what it did")
    of = {e.course: e for e in out}
    for k, e in enumerate(out):
        if e.course >= MIRROR_OFFSET and e.difficulty != of[e.course - MIRROR_OFFSET].difficulty:
            raise SelectError(f"entry {k} (course {e.course}): difficulty {e.difficulty!r} is not its original's "
                              f"{of[e.course - MIRROR_OFFSET].difficulty!r} (a mirror takes its original's label)")
    return tuple(out)


# --- the code ------------------------------------------------------------------------------------------------------

TABLE_SITES = {
    0x0022544C: (0x0023862C, 0, "0x225448 lda 0x23862c,r13: the first record's cursor (select init)"),
    0x00225650: (0x0023862C, 0, "0x22564c lda 0x23862c(r3),g0: the cursor record of state 0"),
    0x002256C8: (0x0023862C, 0, "0x2256c4 lda: state 0 (linked)"),
    0x002256F8: (0x0023862C, 0, "0x2256f4 lda: state 1"),
    0x00225730: (0x0023862C, 0, "0x22572c lda: state 2"),
    0x00225764: (0x0023862C, 0, "0x225760 lda: state 3"),
    0x002257A0: (0x0023862C, 0, "0x22579c lda: state 3 (linked)"),
    0x002263B4: (0x0023862C, 0, "0x2263b0 lda 0x23862c,g0: mode 0's records to the draw"),
    0x002263CC: (0x002386E0, 1, "0x2263c8 lda 0x2386e0,g0: mode 1's"),
    0x002263DC: (0x00238790, 2, "0x2263d8 lda 0x238790,g0: the other modes'"),
}
"""The record-table sites: displacement word -> (original table, the mode table it now names, what)."""
COURSE_OF_PANEL_SITE = (0x00226C80, 0x00238634)     # 0x226c7c ldob 0x238634(g0),g0: the record's course byte
VOICE_SITE = (0x002261CC, 0x00238EBC)               # 0x2261c8 ld 0x238ebc[r3*4],g0: the step's voice per panel
PANEL_COUNT_SITE = (0x002263E0, 0x5C881E03)         # mov 3,g1 -> mov 16,g1
VOTE_SITES = {
    0x0000304C: 0x00501473,     # 0x3048 lda 0x501473,r4: the clear loop's start (count byte and slots)
    0x00003074: 0x00501474,     # 0x3070 stob r3,0x501474: Beginner's priority 2
    0x0000307C: 0x00501478,     # 0x3078 stob r4,0x501478: Advanced's priority 1 (id 2)
    0x00003084: 0x00501476,     # 0x3080 stob r5,0x501476: Expert's priority 0 (id 1)
    0x000032B8: 0x00501474,     # 0x32b4 lda 0x501474,r11: the vote's first slot
    0x00003474: 0x00501473,     # 0x3470 lda 0x501473,r11: the count (and slots at +1, 0x3514)
}
"""The vote's RAM words: displacement word -> original address; each moves by VOTE_RAM - 0x501473."""
VOTE_WORDS = {
    0x00003050: (0x5C281E09, "mov 9,r5 -> lda 33,r5: the clear loop clears the count and sixteen slots"),
    0x000032BC: (0x5CC81E04, "mov 4,g9 (course.ID_PATCHES) -> mov 16,g9: sixteen slots"),
    0x000032D0: (0x59911199, "subi g9,4,g2 (course.ID_PATCHES) -> subi g9,16,g2: the slot's id, 16 - g9"),
}
_OLD_VOTE = 0x00501473

# branch sites: i960 address -> (original word, label in the code block, what)
HOOKS = {
    0x00225F8C: (0x5C201E00, "wheel", "b: the wheel zones (mov 0,r4 ... addo 1,r4,r4) and the stores to 0x225fc0"),
    0x00225FFC: (0x09000000 | (0x2261C0 - 0x225FFC), "timeout", "call: the timer's end, before the step voice"),
    0x00226004: (0x90183000, "step", "b: the gas pedal's step (ld 0x501068,r3)"),
    0x002263E4: (0x09000000 | (0x2269B4 - 0x2263E4), "draw", "call: the panel draw (was 0x2269b4)"),
    0x00226464: (0x5A012086, "fix_a", "b: cmpi r6,r4, the 3D map's lit test, with the entry made a position"),
    0x00226540: (0x59CC0E03, "maprec", "b: shlo 3,g0,g9 + lda 0x23889c[g9*4],g9, the 3D map record"),
    0x00226580: (0x35214174, "fix_b", "b: cmpobne r4,r5,0x2266f4, the chosen map's test"),
    0x002259AC: (0x09000000 | (0x2259C8 - 0x2259AC), "exit", "call: state 4 (the wipe's set-up), before 0x2259c8"),
    0x002259C0: (0x09000000 | (0x2259C8 - 0x2259C0), "exit", "call: state 4 (linked), before 0x2259c8"),
    0x0022563C: (0x88183000, "idle", "b: state 0's ldos 0x501468,r3 (the cursor record of the select's first frame)"),
    0x002256B4: (0x88183000, "idle_linked", "b: state 0's ldos 0x501468,r3 (linked)"),
}
CALLED = ("draw", "timeout", "exit")             # the routines entered by `call`; the others by `b`
IDLE_BACK = {"idle": 0x00225644, "idle_linked": 0x002256BC}    # where state 0 goes on (the ldos's next instruction)


class Layout(NamedTuple):
    """The select's data area (i960 addresses), after a directory of these seventeen words at DATA_AT, for a row of n
    entries (ROW_LENGTHS; the directory holds no n: `row_of` reads it from the tables' spacing)."""
    tables: tuple[int, int, int]    # the three mode tables of n records
    maps: int                       # MAPS: 3 * (n - 2) records of 32 bytes, index 3 * first + position
    live: int                       # u8[n]
    voices: int                     # u32[n]
    names: int                      # u32[n]: the layer-1 map of each entry
    slot_dst: int                   # u32[3]: tile-RAM address of each position's top-left tile in layer 0
    generated: int                  # u8[n]: each entry's window (WINDOW_NONE, WINDOW_LABEL, WINDOW_ALL, WINDOW_NAME)
    slot_masks: int                 # u32[3][2]: each position's interior bits in a mask row (words 0-1, 2-3)
    bands: int                      # s32[n - 1]: bands_of(n)
    hole_maps: tuple[int, int]      # the placeholder panel (normal, blink)
    blank: int                      # the blank layer-1 map
    name_maps: int                  # the entries' layer-1 maps, back to back
    upload: int                     # the extended upload set, its char list and palette list
    char_block: int


def _table_span(n: int) -> int:
    """The bytes one mode table of a row of `n` takes in the data area (n records, 16-aligned)."""
    return -(-n * C.SELECT_RECORD_SIZE // 16) * 16


def row_of(lay: Layout) -> int:
    """The row length (ROW_LENGTHS) of the data at `lay`: the spacing of its three mode tables (n records each, laid
    back to back). SelectError for a Layout whose tables fit no row length."""
    if not isinstance(lay, Layout):
        raise SelectError(f"{lay!r:.60} is not a Layout")
    gaps = {lay.tables[1] - lay.tables[0], lay.tables[2] - lay.tables[1]}
    for n in ROW_LENGTHS:
        if gaps == {_table_span(n)}:
            return n
    raise SelectError(f"the mode tables {[hex(t) for t in lay.tables]} are spaced {sorted(gaps)}: no row of "
                      f"{ROW_LENGTHS}")


MAP_BYTES = 12 + 2 * C.PANEL_ROWS * C.PANEL_COLS
"""The bytes of one 10 x 18 panel map as `_tile_map` encodes it (a header, then a u16 per tile)."""


def _layout(n: int, n_names: int, n_chars: int, n_char_pairs: int, n_palette_bytes: int) -> Layout:
    def take(size: int) -> int:
        nonlocal at
        a = at
        at += -(-size // 16) * 16
        return a

    at = DATA_AT + DIRECTORY_SIZE
    tables = tuple(take(n * C.SELECT_RECORD_SIZE) for _ in range(3))
    maps = take(32 * 3 * (n - 2))
    live, voices, names = take(n), take(4 * n), take(4 * n)
    slot_dst, bands = take(12), take(4 * (n - 1))
    generated, slot_masks = take(n), take(24)
    hole_maps = (take(MAP_BYTES), take(MAP_BYTES))
    blank = take(MAP_BYTES)
    name_maps = take(MAP_BYTES * n_names)
    upload = take(8 + 8 * n_char_pairs + 4 + n_palette_bytes)
    char_block = take(4 + TL.CHAR_SIZE * n_chars)
    if at > DATA_AT + DATA_SPAN:
        raise SelectError(f"the select's data ({at - DATA_AT} bytes) does not fit {DATA_SPAN} bytes from {DATA_AT:#x}")
    return Layout(tables, maps, live, voices, names, slot_dst, generated, slot_masks, bands, hole_maps, blank,
                  name_maps, upload, char_block)


def slot_mask(x: int) -> bytes:
    """The mask row bits of a panel at tile column `x` (its interior: columns x+1..x+16), as two u32: words 0-1 and
    2-3 of the row (little-endian, word k covering columns 16k..16k+15, bit 15 the leftmost). SelectError for an `x`
    where the panel does not fit the 64-column layer."""
    x = _int(x, "tile column", 0, TL.TILE_LAYER - C.PANEL_COLS)
    words = [0, 0, 0, 0]
    for c in range(x + 1, x + 1 + C.PANEL_COLS - 2):
        words[c >> 4] |= 0x8000 >> (c & 15)
    return struct.pack("<4H", *words)


DIRECTORY_SIZE = 128


def _flat(lay: Layout) -> tuple[int, ...]:
    return (*lay.tables, lay.maps, lay.live, lay.voices, lay.names, lay.slot_dst, lay.generated, lay.slot_masks,
            lay.bands, *lay.hole_maps, lay.blank, lay.name_maps, lay.upload, lay.char_block)


def _encode_directory(lay: Layout) -> bytes:
    w = _flat(lay)
    return struct.pack(f"<{len(w)}I", *w).ljust(DIRECTORY_SIZE, b"\0")


def _decode_directory(blob: bytes) -> Layout:
    w = struct.unpack_from("<17I", blob)
    return Layout(w[0:3], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11:13], w[13], w[14], w[15], w[16])


def code(lay: Layout, name_window: bool = False) -> i960.Block:
    """The routines at CODE_AT for the data at `lay` (a row of n, `row_of`). Each `b`-entered one runs in its caller's
    register frame and branches back; the CALLED ones (`draw`, `timeout`, `exit`) are called. The same instructions
    for every row length: n reaches the code as data (the bands' count, the wheel loop's literal, from the data's
    bands_of(n); the draw's row length in g1, PANEL_COUNT_SITE's `mov n,g1`), and no instruction is added.
    `name_window` (Phase 12; patch_select passes True only for a row holding a WINDOW_NAME entry) adds one
    instruction to the draw, `cmpobe WINDOW_NAME,r5,draw_name_rows`: such an entry's position joins the name rows'
    mask words and not the label rows'; False writes the bytes every earlier row has (the frozen rows' pins).
    SelectError for `lay` not a Layout of a row length or a `name_window` that is not a bool."""
    n = row_of(lay)
    if not isinstance(name_window, bool):
        raise SelectError(f"name_window {name_window!r:.60} is not a bool")
    b = i960.Block(CODE_AT)
    g = _STATE

    # wheel: r3 = the wheel value (0x225f74-0x225f88) -> E; 0x501468 = E, 0x5e(g13) = E, 0x501460 = its course id.
    # Until the wheel leaves the centre band this select (FOLLOW_RAM 0), E stays IDLE_ENTRY (the idle wheel).
    b.label("wheel")
    b.mem("ldob", "r5", FOLLOW_RAM)
    b.cobr("cmpobne", 0, "r5", "wheel_follow")
    b.cobr("cmpibl", IDLE_BAND, "r3", "wheel_left")     # IDLE_BAND < w: left
    b.reg("addi", IDLE_BAND, "r3", "r6")
    b.cobr("cmpibg", 0, "r6", "wheel_left")             # w + IDLE_BAND < 0, w < -IDLE_BAND: left
    b.reg("mov", IDLE_ENTRY, "r4")
    b.ctrl("b", "wheel_done")
    b.label("wheel_left")
    b.reg("mov", 1, "r5")
    b.mem("stob", "r5", FOLLOW_RAM)
    b.label("wheel_follow")
    b.reg("mov", 0, "r4")
    b.mem("lda", "r5", lay.bands)
    b.label("wheel_loop")
    b.mem("ld", "r6", None, "r5")
    b.cobr("cmpibge", "r6", "r3", "wheel_done")         # threshold >= w: stop
    b.reg("addo", 1, "r4", "r4")
    b.reg("addo", 4, "r5", "r5")
    b.cobr("cmpobg", len(bands_of(n)), "r4", "wheel_loop")
    b.label("wheel_done")
    b.mem("stos", "r4", PANEL)
    b.mem("stob", "r4", _CURSOR, "g13")
    b.reg("mov", "r4", "g0")
    b.ctrl("call", 0x00226C74)                          # g0 = the record's course id (TABLE_SITES' table 0)
    b.label("wheel_store")
    b.mem("stob", "g0", COURSE_ID)
    b.ctrl("b", 0x00225FC0)                             # the link packet store, then ret

    # idle (0x22563c, linked 0x2256b4, in state 0's frame): the select starts on IDLE_ENTRY, not following the wheel
    for label, back in IDLE_BACK.items():
        b.label(label)
        b.reg("mov", 0, "r3")
        b.mem("stob", "r3", FOLLOW_RAM)                 # not following the wheel yet
        if IDLE_ENTRY:
            b.reg("mov", IDLE_ENTRY, "r3")
        b.mem("stos", "r3", PANEL)
        b.mem("stob", "r3", _CURSOR, "g13")
        b.ctrl("b", back)

    # step (0x226004, frame of 0x225fdc): the gas pedal counts only on a live entry
    b.label("step")
    b.mem("ld", "r3", _STEP_INPUT)
    b.mem("ldis", "r4", PANEL)
    b.mem("ldob", "r4", lay.live, None, "r4")
    b.cobr("cmpobne", 0, "r4", "step_live")
    b.reg("mov", 0, "r3")
    b.label("step_live")
    b.ctrl("b", 0x0022600C)

    # timeout (called from 0x225ffc): a locked entry gives way to IDLE_ENTRY (the idle-wheel ruling), then the voice
    b.label("timeout")
    b.mem("ldis", "r4", PANEL)
    b.mem("ldob", "r6", lay.live, None, "r4")
    b.cobr("cmpobne", 0, "r6", "timeout_voice")
    b.reg("mov", IDLE_ENTRY, "r4")
    b.mem("stos", "r4", PANEL)
    b.mem("stob", "r4", _CURSOR, "g13")
    b.reg("mov", "r4", "g0")
    b.ctrl("call", 0x00226C74)
    b.label("timeout_store")
    b.mem("stob", "g0", COURSE_ID)
    b.mem("ldob", "r3", _LINKED)
    b.cobr("cmpobe", 0, "r3", "timeout_voice")
    b.mem("lda", "r3", _LINK_PACKET)
    b.mem("stob", "g0", 0x14, "r3")
    b.label("timeout_voice")
    b.ctrl("call", 0x002261C0)
    b.ctrl("ret")

    # draw (called from 0x2263e4 with g0 = the mode's table, g1 = n): three positions from first_of(E, n).
    # The window mask: r9/r10 the name rows' words (WINDOW_ALL, WINDOW_NAME), r14/r15 the label rows' (WINDOW_ALL,
    # WINDOW_LABEL).
    b.label("draw")
    b.reg("mov", "g0", "r12")
    b.mem("ldis", "r3", PANEL)
    b.reg("subi", 1, "r3", "r4")
    b.cobr("cmpible", 0, "r4", "draw_lo")
    b.reg("mov", 0, "r4")
    b.label("draw_lo")
    b.reg("subi", 3, "g1", "r5")
    b.cobr("cmpible", "r4", "r5", "draw_hi")
    b.reg("mov", "r5", "r4")
    b.label("draw_hi")
    b.mem("stob", "r4", FIRST_RAM)
    b.mem("ldos", "r11", _VARIANT)
    b.mem("lda", "r8", C.SELECT_RECORD_SIZE)
    b.reg("mov", 0, "r13")
    b.reg("mov", 0, "r9")
    b.reg("mov", 0, "r10")
    b.reg("mov", 0, "r14")
    b.reg("mov", 0, "r15")
    b.label("draw_slot")
    b.reg("addo", "r4", "r13", "r6")                    # the entry
    b.reg("mulo", "r8", "r6", "r7")
    b.reg("addo", "r12", "r7", "r7")                    # its record
    b.mem("ld", "g0", 0x14, "r7", "r11", 8)             # the normal map of the variant
    b.cobr("cmpobne", "r6", "r3", "draw_map")
    b.mem("ld", "r5", _FRAMES)
    b.reg("and", 7, "r5", "r5")
    b.cobr("cmpobge", 4, "r5", "draw_map")
    b.mem("ld", "g0", 0x18, "r7", "r11", 8)             # the cursor's blink map
    b.label("draw_map")
    b.mem("ld", "g1", lay.slot_dst, None, "r13", 4)
    b.ctrl("call", BLIT)
    b.mem("lda", "g0", lay.blank)
    b.mem("ldob", "r5", g, "g13")
    b.cobr("cmpobl", 3, "r5", "draw_name")              # state 4 on: blank windows, layer 0 everywhere
    b.mem("ld", "g0", lay.names, None, "r6", 4)
    b.mem("ldob", "r5", lay.generated, None, "r6")     # the entry's window (WINDOW_*)
    b.cobr("cmpobe", WINDOW_NONE, "r5", "draw_name")
    b.reg("shlo", 3, "r13", "r7")
    b.mem("ld", "g2", lay.slot_masks, "r7")
    b.mem("ld", "g3", lay.slot_masks + 4, "r7")
    if name_window:
        b.cobr("cmpobe", WINDOW_NAME, "r5", "draw_name_rows")     # an original's name: rows 1-5 only
    b.reg("or", "g2", "r14", "r14")
    b.reg("or", "g3", "r15", "r15")
    b.cobr("cmpobne", WINDOW_ALL, "r5", "draw_name")
    if name_window:
        b.label("draw_name_rows")
    b.reg("or", "g2", "r9", "r9")
    b.reg("or", "g3", "r10", "r10")
    b.label("draw_name")
    b.mem("ld", "g1", lay.slot_dst, None, "r13", 4)
    b.mem("lda", "g1", LAYER_BYTES, "g1")
    b.ctrl("call", BLIT)
    b.reg("addo", 1, "r13", "r13")
    b.cobr("cmpobg", 3, "r13", "draw_slot")
    b.mem("lda", "r5", MASK_RAM + 8 * NAME_MASK_ROWS[0])
    for (lo, hi), (w0, w1) in ((NAME_MASK_ROWS, ("r9", "r10")), (LABEL_MASK_ROWS, ("r14", "r15"))):
        b.mem("lda", "r6", hi - lo)
        b.label(f"draw_mask_{lo}")
        b.mem("st", w0, None, "r5")
        b.mem("st", w1, 4, "r5")
        b.reg("addo", 8, "r5", "r5")
        b.reg("subo", 1, "r6", "r6")
        b.cobr("cmpobne", 0, "r6", f"draw_mask_{lo}")
    b.ctrl("ret")

    # exit (called from state 4's handlers 0x2259a0 / 0x2259b4, entering the wipe): the draw once more, now with the
    # state at 4 (layer 1 blank, the mask clear), then what they called (0x2259c8: state 5, the wipe)
    b.label("exit")
    b.ctrl("call", 0x002263B0)
    b.ctrl("call", 0x002259C8)
    b.ctrl("ret")

    # fix_a (0x226464, frame of 0x226438): r6 = the lit entry (0x5e(g13), or the chosen course's entry) -> position
    b.label("fix_a")
    b.mem("ldob", "r7", FIRST_RAM)
    b.reg("subo", "r7", "r6", "r6")
    b.reg("cmpi", "r6", "r4")
    b.ctrl("be", 0x00226478)
    b.ctrl("b", 0x0022646C)

    # maprec (0x226540, frame of 0x22653c): g9 = MAPS[3 * first + g0]
    b.label("maprec")
    b.mem("ldob", "r3", FIRST_RAM)
    b.reg("mulo", 3, "r3", "r3")
    b.reg("addo", "g0", "r3", "r3")
    b.reg("shlo", 5, "r3", "r3")
    b.mem("lda", "g9", lay.maps, "r3")
    b.ctrl("b", 0x0022654C)

    # fix_b (0x226580): r5 = the chosen course's entry -> position; not this position: the shrinking path
    b.label("fix_b")
    b.mem("ldob", "r3", FIRST_RAM)
    b.reg("subo", "r3", "r5", "r5")
    b.reg("cmpo", "r4", "r5")
    b.ctrl("bne", 0x002266F4)
    b.ctrl("b", 0x00226584)
    return b


# --- the patch -----------------------------------------------------------------------------------------------------

def _word(rs: RomSet, addr: int) -> int:
    try:
        return _u32(rs, addr)
    except CourseError as e:
        raise SelectError(str(e)) from e


def patched(rs: RomSet) -> bool:
    """True when `rs` carries patch_select: the panel draw at 0x2263e4 calls the code at CODE_AT."""
    if not isinstance(rs, RomSet):
        raise SelectError(f"{type(rs).__name__} is not a RomSet")
    return _word(rs, 0x002263E4) != HOOKS[0x002263E4][0]


def _check_original(rs: RomSet) -> None:
    if not isinstance(rs, RomSet):
        raise SelectError(f"{type(rs).__name__} is not a RomSet")
    if not C.grown(rs):
        raise SelectError("the set is not grown (course.grow_ids): the select needs course ids 0-15")
    if patched(rs):
        raise SelectError("the select is already patched (0x2263e4 calls the new draw)")
    want = {a: (w, what) for a, (w, _, what) in TABLE_SITES.items()}
    want[COURSE_OF_PANEL_SITE[0]] = (COURSE_OF_PANEL_SITE[1], "0x226c7c")
    want[VOICE_SITE[0]] = (VOICE_SITE[1], "0x2261c8")
    want[PANEL_COUNT_SITE[0]] = (PANEL_COUNT_SITE[1], "0x2263e0 mov 3,g1")
    want.update({a: (w, f"vote site {a:#x}") for a, w in VOTE_SITES.items()})
    want.update({a: (w, what) for a, (w, what) in VOTE_WORDS.items()})
    want.update({a: (w, what) for a, (w, _, what) in HOOKS.items()})
    for a, (w, what) in want.items():
        if (got := _word(rs, a)) != w:
            raise SelectError(f"{what}: the word at {a:#x} is {got:#x}, not {w:#x}")


def _tile_map(tiles, base: int = -0x8000) -> bytes:
    entries = [((t - base) & 0xFFFF) - (0x10000 if (t - base) & 0x8000 else 0) for t in tiles]
    return TL.encode_tile_map(TL.TileMap(base, True, C.PANEL_ROWS, C.PANEL_COLS, tuple(entries)))


FILLED_FIRST = 3
"""The lowest course id a live entry may name only once the set holds a course of its own there (`_unfilled`): ids
0-2 are the originals; id 3 (Desert City) is converted before the select (build.converted_set, then patch_select:
the same bytes as Phase 8's order, which converted it after, test_convert_build.test_desert_city_before_select), and
ids 4-15 (M5's courses and the mirrors, mirror.mirrored_entry) are written before it too. Unconverted, id 3 is
Beginner's copy: a live entry 3 would race Beginner's rolling word on Desert City's standing path (Phase 6's crash),
so it is checked like the rest (Phase 9's final review, item 3; until then FILLED_FIRST was 4)."""


def _unfilled(rs: RomSet, entries) -> list[tuple[int, int]]:
    """(entry index, course id) of every live entry whose course id (FILLED_FIRST or above) holds no course of its own
    in `rs` (golden.own_tables: its record and per-id entries are still grow_ids' copies of Beginner's)."""
    from dtools.arcade import golden as G         # golden imports this module
    out = []
    for k, e in enumerate(entries):
        if e.live and e.course >= FILLED_FIRST:
            try:
                if not G.own_tables(rs, e.course):
                    out.append((k, e.course))
            except DecodeError as err:
                raise SelectError(f"entry {k} (course id {e.course}): {err}") from err
    return out


def patch_select(rs: RomSet, entries=ENTRIES, *, faceted_maps: bool = True) -> RomSet:
    """`rs` (a grown set, course.grow_ids) with the Circuit Select of `entries`, a row of 8 or 16 (ROW_LENGTHS;
    ENTRIES, the release row of eight, by default; the draw's WINDOW_NAME branch only for a row holding a WINDOW_NAME
    entry, `code`; the module docstring has the design; a row of 16 writes the bytes it wrote
    before rows of 8 existed): the record tables, maps, names, 3D map records and code written into free
    space (CODE_FREE, DATA_AT..+DATA_SPAN: every byte written there must be 0xff before), the sites of TABLE_SITES,
    COURSE_OF_PANEL_SITE, VOICE_SITE, PANEL_COUNT_SITE, HOOKS, VOTE_SITES and VOTE_WORDS rewritten,
    SELECT_PANEL_OF_COURSE (course.id_table) made the inverse of the records' course ids and the select's second
    upload set (SELECT_UPLOAD_WORD) replaced by an extended copy; with a mirror of an original in the row, its
    mirrored map objects and list at the last free bytes of the custom polygon area and of tgp.DATA_FREE
    (`_mirrored_maps`), with an own_panel entry its per-mode maps after the window maps and its course's map (object,
    headers and list at the last free bytes of the custom polygon and textures areas and of tgp.DATA_FREE,
    `_own_maps`), and with either course.LIBRARY_BOUND_PATCH applied. SelectError for `rs` not a grown, unpatched RomSet
    (any site not holding its original word), bad entries (check_entries), a live entry with a course id 3-15 on a set
    without the new ids' guards (course.guard_new_ids: unguarded, that id would race through the literal id tests
    and write its play counters over Beginner's SRAM; guard first, then patch), a live entry with a course id 3-15
    the set holds no course at (`_unfilled`: convert Desert City, build.converted_set, and write the mirrors,
    mirror.mirrored_entry, first, then patch; Phase 8's row M4A_ENTRIES locks the mirrors) or space that is not free.
    The tables the select indexes by entry hold n (the records, MAPS, live, voices, names, windows, bands; the draw's
    row length at PANEL_COUNT_SITE); those indexed by course id hold sixteen whatever n (SELECT_PANEL_OF_COURSE, whose
    ids without an entry in a row of 8, 8-15, name the entry of id - 8; the vote's VOTE_WORDS): docs/spike/
    2026-11-courses-4-7.md, "The row of eight"."""
    _check_original(rs)
    entries = check_entries(entries)
    n = len(entries)
    if live := [(k, e.course) for k, e in enumerate(entries) if e.live and e.course >= C.COURSE_COUNT]:
        try:
            ok = C.guarded(rs)
        except DecodeError as e:
            raise SelectError(f"the new ids' guards: {e}") from e
        if not ok:
            raise SelectError(f"entries {', '.join(f'{k} (course id {i})' for k, i in live)} are live, but the set "
                              f"has no new ids' guards: apply course.guard_new_ids before patch_select")
    if empty := _unfilled(rs, entries):
        raise SelectError(f"entries {', '.join(f'{k} (course id {i})' for k, i in empty)} are live, but the set has "
                          f"no course of its own at those ids (Beginner's copy: Desert City's or a mirror's id "
                          f"would race it with Beginner's rolling flag on a standing path): write them first "
                          f"(build.converted_set for a new course, mirror.mirrored_entry for a mirror), or lock them")
    gr = graphics(rs, entries)
    old_set = TL.upload_set(rs, _word(rs, SELECT_UPLOAD_WORD), segments=False)
    runs = old_set.palette + (TL.PaletteUpload(TL.PALETTE_RAM[0] + 2 * 16 * PALETTE_NORMAL, sum(gr.palettes, ())),
                              TL.PaletteUpload(TL.PALETTE_RAM[0] + 2 * 16 * PALETTE_GREY, gr.grey * BLOCKS))
    pal_bytes = len(TL.encode_palette_list(runs))
    named = [k for k, m in enumerate(gr.name_maps) if m is not None]
    owned_maps = sum(2 * len(pm) for pm in gr.panels if pm is not None)    # an own panel's (normal, blink) per mode
    lay = _layout(n, len(named) + owned_maps, gr.uploaded() + len(gr.runs), len(old_set.chars) + len(gr.runs),
                  pal_bytes)
    patches: dict[tuple[str, int], bytes] = {}

    def put(addr: int, blob: bytes, free: bool) -> None:
        region, off = C.rom_word_at(addr)
        if free:
            have = (rs.program if region == "program" else rs.data)[off:off + len(blob)]
            if have != b"\xff" * len(blob):
                raise SelectError(f"{addr:#x} (+{len(blob)}): not free (0xff)")
        patches[(region, off)] = blob

    # records, per mode: (cursor, course, tile, maps)
    old = [[C.decode_select_panel(rs.read_i960(t + C.SELECT_RECORD_SIZE * p, C.SELECT_RECORD_SIZE), p)
            for p in range(C.SELECT_PANELS)] for t in C.SELECT_TABLES]
    hole_n, hole_b = old[0][0].maps[0], old[0][0].maps[1]
    holes = []
    for src in (hole_n, hole_b):
        m = TL.tile_map(rs, src)
        tiles = [m.tile(r, c) if r in (0, C.PANEL_ROWS - 1) or c in (0, C.PANEL_COLS - 1) else gr.hole
                 for r in range(C.PANEL_ROWS) for c in range(C.PANEL_COLS)]
        holes.append(_tile_map(tiles, m.base))
    for a, blob in zip(lay.hole_maps, holes):
        put(a, blob, True)
    placeholder = tuple(lay.hole_maps) * C.SELECT_VARIANTS
    # an own panel's maps (per mode, normal and blink) after the window maps, in the name maps' area
    own: dict[int, list[tuple[int, int]]] = {}
    at = lay.name_maps + MAP_BYTES * len(named)
    for k, pm in enumerate(gr.panels):
        if pm is not None:
            own[k] = []
            for normal, blink in pm:
                put(at, _tile_map(normal), True)
                put(at + MAP_BYTES, _tile_map(blink), True)
                own[k].append((at, at + MAP_BYTES))
                at += 2 * MAP_BYTES
    for mode, table in enumerate(lay.tables):
        blob = b""
        for k, e in enumerate(entries):
            s = slot_of(k, n)
            w = gr.windows[k]
            if k in own:
                maps = own[k][mode] * C.SELECT_VARIANTS                                  # its own, this mode's
            else:
                maps = placeholder if w == WINDOW_ALL else old[mode][scheme_of(e)].maps     # the original's art
            blob += C.encode_select_panel(C.SelectPanel(k, e.course, old[0][s].cursor, old[0][s].tile, maps))
        put(table, blob, True)
    # the 3D maps: MAPS[3 * first + position]; the mirrors of the originals' quads mirrored (`_mirrored_maps`), an own
    # panel's course drawn from its road (`_own_maps`)
    words, map_patches = _mirrored_maps(rs, entries)
    own_words, own_patches = _own_maps(rs, entries, map_patches, faceted_maps)
    words.update(own_words)
    blob = b""
    for first in range(n - 2):
        for s in range(3):
            pos = rs.read_i960(C.SELECT_MAP_OBJECTS + 32 * s, 12)
            e = entries[first + s]
            if is_original(e):
                blob += pos + rs.read_i960(C.SELECT_MAP_OBJECTS + 32 * ORIGINAL_PANEL[e.course] + 12, 20)
            else:
                blob += pos + words.get(e.course, b"\0" * 20)
    put(lay.maps, blob, True)
    patches.update(map_patches)
    patches.update(own_patches)
    put(lay.live, bytes(1 if e.live else 0 for e in entries), True)
    old_voices = struct.unpack("<3I", rs.read_i960(VOICE_SITE[1], 12))
    put(lay.voices, b"".join(struct.pack("<I", old_voices[ORIGINAL_PANEL[e.course]] if is_original(e)
                                         else VOICE_DEFAULT) for e in entries), True)
    blank = _tile_map([LAYER1_BLANK] * (C.PANEL_ROWS * C.PANEL_COLS))
    put(lay.blank, blank, True)
    names, at = [], lay.name_maps
    for k, m in enumerate(gr.name_maps):
        if m is None:
            names.append(lay.blank)
        else:
            put(at, _tile_map(m), True)
            names.append(at)
            at += len(blank)
    put(lay.names, struct.pack(f"<{n}I", *names), True)
    put(lay.slot_dst, b"".join(struct.pack("<I", TL.TILE_RAM[0] + 2 * p.tile[0] + 0x80 * p.tile[1])
                               for p in old[0]), True)
    put(lay.bands, struct.pack(f"<{n - 1}i", *bands_of(n)), True)
    put(lay.generated, bytes(gr.windows), True)
    put(lay.slot_masks, b"".join(slot_mask(p.tile[0]) for p in old[0]), True)
    # the upload set: the old set's chars and palettes, then one char block per used block of generated chars and the
    # two palette runs
    cl, pl = lay.upload + 8, lay.upload + 8 + 8 * (len(old_set.chars) + len(gr.runs)) + 4
    put(lay.upload, TL.encode_upload_set(cl, pl), True)
    pairs, at = [], lay.char_block
    for first, count in gr.runs:
        blob = TL.encode_char_block(TL.encode_chars(gr.chars[first - CHAR_HOLE:first - CHAR_HOLE + count]))
        put(at, blob, True)
        pairs.append((at, TL.CHAR_RAM[0] + TL.CHAR_SIZE * first))
        at += len(blob)
    put(cl, TL.encode_char_list([(u.src, u.dst) for u in old_set.chars] + pairs), True)
    put(pl, TL.encode_palette_list(runs), True)
    put(DATA_AT, _encode_directory(lay), True)
    put(SELECT_UPLOAD_WORD, struct.pack("<I", lay.upload), False)
    # the inverse of the records' ids, by course id (sixteen words): on a row of 8 ids 8-15 have no entry and name
    # the entry of id - 8 (read only for the chosen course, 0x501460, which no entry of the row sets to 8-15)
    poc = C.id_table(rs, "SELECT_PANEL_OF_COURSE")
    entry_of = {e.course: k for k, e in enumerate(entries)}
    put(poc, struct.pack(f"<{ENTRY_COUNT}I", *(entry_of[c if c < n else c - MIRROR_OFFSET]
                                               for c in range(ENTRY_COUNT))), False)
    # code and sites
    blk = code(lay, WINDOW_NAME in gr.windows)
    body = blk.bytes()
    if not (CODE_FREE[0] <= CODE_AT and CODE_AT + len(body) <= CODE_FREE[1]):
        raise SelectError(f"the code ({len(body)} bytes) does not fit the program's free run")
    put(CODE_AT, body, True)
    for a, (_, mode, _) in TABLE_SITES.items():
        put(a, struct.pack("<I", lay.tables[mode]), False)
    put(COURSE_OF_PANEL_SITE[0], struct.pack("<I", lay.tables[0] + 8), False)
    put(VOICE_SITE[0], struct.pack("<I", lay.voices), False)
    put(PANEL_COUNT_SITE[0], i960.reg("mov", n, "g1"), False)
    for a, (_, label, _) in HOOKS.items():
        op = "call" if label in CALLED else "b"
        put(a, i960.ctrl(op, blk.labels[label] - a), False)
    for a, old_addr in VOTE_SITES.items():
        put(a, struct.pack("<I", old_addr - _OLD_VOTE + VOTE_RAM), False)
    put(0x00003050, i960.mem("lda", "r5", 1 + 2 * ENTRY_COUNT), False)     # the vote: sixteen course ids, any n
    put(0x000032BC, i960.reg("mov", ENTRY_COUNT, "g9"), False)
    put(0x000032D0, i960.reg("subi", "g9", ENTRY_COUNT, "g2"), False)
    out = rs.with_patches(patches)
    return C.patch_library_bound(out) if map_patches or own_patches else out


# --- the 3D maps ---------------------------------------------------------------------------------------------------

MAP_MODE = 1
"""The geometry mode the map objects are decoded in. Each original's map is a textured quad, x and z -2..2, y 0 (the
course map is its texture), drawn as one to four quads at the same corners with move records between them. The mode
on Circuit Select is not traced (golden.EXCLUDED, "map objects in polygon ROM"), and need not be: the four modes
read a quad's points at the same words (modes 2-3 skip the normal words, which `mirror_point` would turn into the
same normals), so the mirror is the same in every mode (discovery log, "Panels")."""


class MapRecord(NamedTuple):
    first: int                          # first_of of the row shown
    position: int                       # the panel position 0-2
    place: tuple[float, float, float]   # the position's place (the original record's of that position)
    objects: int                        # the object list (0: no map)
    raw: tuple[int, int, int, int]      # words 4-7 (word 4 the list again on the real set, no reader found)


def _stock_map_lists(rs: RomSet) -> tuple[int, ...]:
    return tuple(_word(rs, C.SELECT_MAP_OBJECTS + 32 * p + 12) for p in range(C.SELECT_PANELS))


def _last_fit(rs: RomSet, region: str, run: tuple[int, int], need: int, what: str, busy=()) -> int:
    """The highest 4-aligned offset in `run` from which `need` bytes are free and owned by nothing (build.last_fit),
    and clear of `busy` ((region, offset, length) spans this patch already takes): the run's last bytes, so a
    converter's or a mirror's first fit, before the select or after it, places the same."""
    from dtools.convert import build as B
    try:
        return B.last_fit(rs, region, run, need, what, busy)
    except DecodeError as e:
        raise SelectError(str(e)) from e


def _mirrored_maps(rs: RomSet, entries) -> tuple[dict[int, bytes], dict[tuple[str, int], bytes]]:
    """({course id of a mirror of an original: its map record's 20 bytes after the place}, patches): each such
    entry's original's map objects mirrored (mirror.mirror_object, local: z -> -z about the quad's centre, the texture
    flipped left to right in the quad's frame) and a list of its own naming them, the objects at the last free
    bytes of the custom polygon area (build.NEW_ID_RUNS), the lists at the last of the list home (tgp.DATA_FREE: the
    draw routines skip a list above the object library, course.PATCHED_LIBRARY); the record's words naming the
    original's list name the new one."""
    from dtools.arcade import mirror as M
    from dtools.arcade import polyrom as P
    from dtools.arcade import tgp
    from dtools.convert import build as B
    todo = [e.course for e in entries if window_of(e) == WINDOW_LABEL]
    if not todo:
        return {}, {}
    stock = _stock_map_lists(rs)
    made: list[tuple[int, list, list[bytes], tuple[int, ...]]] = []
    try:
        for c in todo:
            p = ORIGINAL_PANEL[c - MIRROR_OFFSET]
            rec = struct.unpack("<5I", rs.read_i960(C.SELECT_MAP_OBJECTS + 32 * p + 12, 20))
            places = C.object_list(rs, stock[p], f"panel {p}'s map")
            objs = []
            for q in places:
                o = P.decode_object(rs.polygons, C.check_rom_object(q.oba, rs), q.obc, MAP_MODE)
                w = P.encode_object(M.mirror_object(o, world=False))
                objs.append(struct.pack(f"<{len(w)}I", *w))
            made.append((c, places, objs, rec))
    except DecodeError as e:
        if isinstance(e, SelectError):
            raise
        raise SelectError(f"the mirrored maps: {e}") from e
    poly = _last_fit(rs, "polygons", B.NEW_ID_RUNS["polygons"], sum(len(b) for _, _, objs, _ in made for b in objs),
                     "the mirrored maps' objects")
    data = _last_fit(rs, "data", tgp.DATA_FREE, sum(C.OBJECT_ENTRY_SIZE * len(pl) + 4 for _, pl, _, _ in made),
                     "the mirrored maps' lists")
    words, patches = {}, {}
    for c, places, objs, rec in made:
        new = []
        for q, b in zip(places, objs):
            patches[("polygons", poly)] = b
            new.append(dataclasses.replace(q, oba=(q.oba & ~0x7FFFFF) | poly // 4))
            poly += len(b)
        lst = C.encode_object_list(new)
        patches[("data", data)] = lst
        at = C.DATA_BASE + data
        data += len(lst)
        words[c] = struct.pack("<5I", at, *(at if w == rec[0] else w for w in rec[1:]))
    return words, patches

def _own_maps(rs: RomSet, entries, taken: dict, faceted: bool = True) -> tuple[dict[int, bytes], dict[tuple[str, int], bytes]]:
    """({course id of an own_panel entry: its map record's 20 bytes after the place}, patches): each such course's map
    drawn from its own road and scenery (selmap.course_map: flat polygons in the map quad's frame) and its flat
    headers (one a part, selmap.parts), the objects at the last free bytes of the custom polygon area, the headers at
    the last of the custom textures area (build.NEW_ID_RUNS), the lists (one entry each, panel 0's map placement with
    the object and headers swapped in) at the last of the list home tgp.DATA_FREE, all clear of `taken`
    (`_mirrored_maps`' patches); the record's words as panel 0's, its list's address in place of panel 0's list."""
    from dtools.arcade import selmap
    from dtools.arcade import polyrom as P
    from dtools.arcade import texture as T
    from dtools.arcade import tgp
    from dtools.convert import build as B
    todo = [e.course for e in entries if e.own_panel]
    if not todo:
        return {}, {}
    try:
        stock = _stock_map_lists(rs)
        rec = struct.unpack("<5I", rs.read_i960(C.SELECT_MAP_OBJECTS + 12, 20))
        place = C.object_list(rs, stock[0], "panel 0's map")[0]
        made = {c: selmap.course_map(rs, c, faceted) for c in todo}
    except DecodeError as e:
        if isinstance(e, SelectError):
            raise
        raise SelectError(f"the own panels' maps: {e}") from e
    objs = {c: struct.pack(f"<{len(w)}I", *w) for c, (o, _) in made.items() for w in (P.encode_object(o),)}
    heads = {c: h for c, (_, h) in made.items()}
    busy = [(r, off, len(b)) for (r, off), b in taken.items()]
    poly = _last_fit(rs, "polygons", B.NEW_ID_RUNS["polygons"], sum(map(len, objs.values())),
                     "the own panels' map objects", busy)
    tex = _last_fit(rs, "textures", B.NEW_ID_RUNS["textures"], sum(map(len, heads.values())),
                    "the own panels' map headers", busy)
    data = _last_fit(rs, "data", tgp.DATA_FREE, len(todo) * (C.OBJECT_ENTRY_SIZE + 4), "the own panels' map lists",
                     busy)
    words, patches = {}, {}
    for c in todo:
        patches[("polygons", poly)] = objs[c]
        patches[("textures", tex)] = heads[c]
        lst = C.encode_object_list([dataclasses.replace(place, oba=(place.oba & ~0x7FFFFF) | poly // 4,
                                                        tha=(place.tha & ~T.TEX_ADDR_MASK) | tex // 2)])
        patches[("data", data)] = lst
        at = C.DATA_BASE + data
        words[c] = struct.pack("<5I", at, *(at if w == rec[0] else w for w in rec[1:]))
        poly, tex, data = poly + len(objs[c]), tex + len(heads[c]), data + len(lst)
    return words, patches


def map_records(rs: RomSet) -> tuple[MapRecord, ...]:
    """The 3 * (n - 2) map records patch_select wrote for its row of n (MAPS, index 3 * first + position).
    SelectError for an unpatched set."""
    lay = layout_of(rs)
    out = []
    for i in range(3 * (row_of(lay) - 2)):
        try:
            w = rs.read_i960(lay.maps + 32 * i, 32)
        except DecodeError as e:
            raise SelectError(f"map record {i}: {e}") from e
        words = struct.unpack_from("<5I", w, 12)
        out.append(MapRecord(i // 3, i % 3, struct.unpack_from("<3f", w), words[0], words[1:]))
    return tuple(out)


def map_spans(rs: RomSet) -> list[tuple[str, int, int, str]]:
    """(region, offset, length, owner) of the map lists, objects and headers patch_select wrote (the lists its records
    name other than the original panels', their objects, and the headers those objects read in the custom textures
    area, build.NEW_ID_RUNS: an own panel's map's; a mirror's map reads its original's headers, Sega's): owned
    (golden.owned_spans). SelectError for an unpatched set or a list, object or header that does not decode."""
    from dtools.arcade import polyrom as P
    from dtools.arcade import texture as T
    from dtools.convert import build as B
    stock = set(_stock_map_lists(rs))
    lo, size = B.NEW_ID_RUNS["textures"]
    out = []
    try:
        for lst in sorted({r.objects for r in map_records(rs)} - stock - {0}):
            places = C.object_list(rs, lst, "select map list")
            out.append(("data", lst - C.DATA_BASE, C.OBJECT_ENTRY_SIZE * len(places) + 4, "select map list"))
            for q in places:
                i = C.check_rom_object(q.oba, rs)
                obj = P.decode_object(rs.polygons, i, q.obc, MAP_MODE)
                n = len(P.encode_object(obj))
                out.append(("polygons", 4 * i, 4 * n, "select map object"))
                for at in sorted({2 * (h.tha & T.TEX_ADDR_MASK) for h in T.object_headers(rs, q.tha, obj)}):
                    if lo <= at < lo + size:
                        out.append(("textures", at, 8, "select map header"))
    except DecodeError as e:
        if isinstance(e, SelectError):
            raise
        raise SelectError(f"the select's map spans: {e}") from e
    return out


# --- decoding a patched set ----------------------------------------------------------------------------------------

class Row(NamedTuple):
    tables: tuple[int, int, int]
    records: tuple[tuple[C.SelectPanel, ...], ...]  # per mode, n in row order (a row of n: ROW_LENGTHS)
    panel_of_course: tuple[int, ...]                # course id -> entry (sixteen ids, whatever n)
    live: tuple[bool, ...]
    voices: tuple[int, ...]
    names: tuple[int, ...]                          # the layer-1 map per entry
    maps: int
    bands: tuple[int, ...]
    vote: int                                       # the vote's count byte (VOTE_RAM on a patched set)


def select_row(rs: RomSet) -> Row:
    """The patched select of `rs`, read from its sites: the mode tables the record sites name, their n records (the
    row's length, `row_of`), SELECT_PANEL_OF_COURSE (sixteen course ids), the live bytes, voices and name maps the
    code reads, the bands and the vote's RAM. SelectError for an unpatched set or sites that disagree with each
    other (the draw's row length at PANEL_COUNT_SITE among them)."""
    if not patched(rs):
        raise SelectError("the select is not patched (patch_select)")
    tables = tuple(_word(rs, a) for a in (0x002263B4, 0x002263CC, 0x002263DC))
    for a, (_, mode, what) in TABLE_SITES.items():
        if _word(rs, a) != tables[mode]:
            raise SelectError(f"{what}: names {_word(rs, a):#x}, not mode {mode}'s table {tables[mode]:#x}")
    if _word(rs, COURSE_OF_PANEL_SITE[0]) != tables[0] + 8:
        raise SelectError("0x226c7c does not read mode 0's records' course byte")
    lay = layout_of(rs)
    if lay.tables != tables:
        raise SelectError(f"the directory's tables {lay.tables} are not the sites' {tables}")
    n = row_of(lay)
    blk = code(lay, WINDOW_NAME in rs.read_i960(lay.generated, n))
    if rs.read_i960(CODE_AT, len(blk.bytes())) != blk.bytes():
        raise SelectError(f"the code at {CODE_AT:#x} is not patch_select's for the data at {DATA_AT:#x}")
    if _word(rs, PANEL_COUNT_SITE[0]) != struct.unpack("<I", i960.reg("mov", n, "g1"))[0]:
        raise SelectError(f"0x2263e0 does not give the draw the row's length {n} (mov {n},g1)")
    records = tuple(tuple(C.decode_select_panel(rs.read_i960(t + C.SELECT_RECORD_SIZE * k, C.SELECT_RECORD_SIZE), k)
                          for k in range(n)) for t in tables)
    poc_at = C.id_table(rs, "SELECT_PANEL_OF_COURSE")
    poc = struct.unpack(f"<{ENTRY_COUNT}I", rs.read_i960(poc_at, 4 * ENTRY_COUNT))
    live = tuple(bool(b) for b in rs.read_i960(lay.live, n))
    voices = struct.unpack(f"<{n}I", rs.read_i960(_word(rs, VOICE_SITE[0]), 4 * n))
    names = struct.unpack(f"<{n}I", rs.read_i960(lay.names, 4 * n))
    bands = struct.unpack(f"<{n - 1}i", rs.read_i960(lay.bands, 4 * (n - 1)))
    return Row(tables, records, poc, live, voices, names, lay.maps, bands, _word(rs, 0x0000304C))


def layout_of(rs: RomSet) -> Layout:
    """The Layout patch_select wrote into `rs`, from the directory at DATA_AT. SelectError for an unpatched set."""
    if not patched(rs):
        raise SelectError("the select is not patched (patch_select)")
    try:
        return _decode_directory(rs.read_i960(DATA_AT, DIRECTORY_SIZE))
    except DecodeError as e:
        raise SelectError(f"no select directory at {DATA_AT:#x}: {e}") from e


def tile_maps(rs: RomSet, row: Row) -> dict[int, tuple[TL.TileMap, ...]]:
    """Per entry its (layer-0 normal map of mode 0 variant 0, layer-1 name map). SelectError for `row` not a Row of
    three modes of n records and n names, n one of ROW_LENGTHS; CourseError (a DecodeError) for a map that does not
    decode."""
    n = len(row.names) if isinstance(row, Row) and isinstance(row.names, tuple) else None
    if n not in ROW_LENGTHS or len(row.records) != 3 or any(len(r) != n for r in row.records):
        raise SelectError(f"{row!r:.60} is not a Row of three modes of n records and n names (n in {ROW_LENGTHS})")
    return {k: (TL.tile_map(rs, row.records[0][k].maps[0]), TL.tile_map(rs, row.names[k])) for k in range(n)}

