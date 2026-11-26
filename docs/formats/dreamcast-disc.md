# Dreamcast disc (Daytona USA, USA, product MK-51037)

Text only: layout, header fields and file sizes. No game data is stored in the repository.

## Source and the `.7z`

- Source dump: Redump, `Daytona USA (USA)-Dreamcast.7z`, containing a `.cue` and three tracks.
  - Track 1: single-density data. Track 2: audio.
  - Track 3: `Daytona USA (USA) (Track 3).bin`, MODE1/2352, 1,185,760,800 bytes, CRC32 `aa3f9cde`. This is the high-density area and holds the game files.
- `dtools check` / `dtools build` (Phase 12; `release.check_inputs`) accept the `.7z` or the `.cue` with its tracks
  beside it, nothing else (a `.gdi`, `.iso` or `.chd` is refused naming the two shapes). The file the cue names as track
  3 must be `release.DISC_TRACK3` (the size and CRC32 above; the whole track is read, about 1.2 GB), then
  `DcDisc.verify()` (product MK-51037, every course file present). A `.7z` is extracted under `<out>/work/dc/`.
- `DcDisc.open(path)` accepts the `.cue` (track 3 is the `.bin` named by the `FILE` line before `TRACK 03`, beside the cue) or the `.7z`.
- For a `.7z` it runs `tar -xf <archive> <cue>` (macOS bsdtar reads 7z) into `<out>/dc/<archive stem>.partial/`, checks the cue (track 3 must be `MODE1/2352`, see below), then `tar -xf <archive> <track 3>`, and renames the directory to `<out>/dc/<archive stem>/`; `<out>` is the `out` argument or `DAYTONA_OUT` (default `<repo>/out`). Nothing is written beside the archive. If the cue and track 3 are already there, extraction is skipped. A missing `tar` or an unreadable cue raises `DiscError`.

## GD-ROM layout and the 45000 base

- The ISO9660 extents on track 3 are absolute GD-ROM sectors counted from LBA 45000.
- File sector index n holds LBA 45000 + n.
  - IP header: LBA 45000 (file sector 0).
  - Primary volume descriptor: LBA 45016 (file sector 16).
  - A directory or file extent written as LBA L is at file sector L - 45000.
- `Iso2352(path, lba_base=45000)` implements this: `sector(lba)` reads frame `lba - lba_base`; each frame is 2352 bytes, with 2048 user bytes after a 16-byte sync and header.
- The listing has 744 entries: 580 files in the root plus 164 under the browser directories `DPETC`, `DPFONT`, `DPSS`, `DPTEX`, `DPWWW`. There are no other directories.
- Entry names carry a `;1` version suffix on disc; the reader strips it.

## IP header (the 2048 user bytes of LBA 45000)

| offset | size | field | value on this disc |
|---|---|---|---|
| 0x00 | 16 | hardware id | `SEGA SEGAKATANA ` |
| 0x10 | 16 | maker | `SEGA ENTERPRISES` |
| 0x20 | 16 | device info | CRC, `GD-ROM1/1` |
| 0x30 | 8 | area symbols | region flags |
| 0x38 | 8 | peripherals | supported controllers |
| 0x40 | 10 | product number | `MK-51037  ` (`DcDisc.product`, stripped) |
| 0x4a | 6 | version | `V1.004` |
| 0x50 | 16 | release date | 2001-02-14 |
| 0x60 | 16 | boot file | `1ST_READ.BIN` |
| 0x70 | 16 | company | `SEGA ENTERPRISES` |
| 0x80 | 128 | title | `DAYTONAUSA` padded with spaces (`DcDisc.title`, stripped) |

A real Saturn dump never reaches `verify()`. The Redump Saturn discs (`Daytona USA (USA).7z`, MK-81200, 22 tracks;
`Daytona USA - Championship Circuit Edition (USA).7z`, MK-81213, 27 tracks) have one data track, track 1
(`MODE1/2352`, the Saturn IP header `SEGA SEGASATURN ` at user byte 0x00, the product at 0x20-0x29), and audio from
track 2 on, so their `(Track 03).bin` is CD audio (10 MB and 44 MB). `DcDisc.open` reads the cue first: when
`TRACK 03` is not `MODE1/2352` it reads track 1's first sector (beside the cue, or streamed from the `.7z` with
`tar -xOf`, nothing written) and raises `DiscError`, for example
`not a Dreamcast disc: SEGA SEGASATURN, product MK-81200 (Daytona USA (USA).cue: track 3 is AUDIO, not MODE1/2352)`
(a header that is neither Saturn's nor Dreamcast's: `not a Dreamcast disc (header b'...')`). From a `.7z` only the cue
has been extracted by then, and it is removed with the `.partial` directory.

`verify()` first checks the hardware id; a different header in a GD-ROM-shaped image is reported as
`not a Dreamcast disc (header ...)`. Then it checks the product number and that every course file exists.

## Course file families

Course n is 0-7; variant v is one of `""`, `R`, `M`, `MR`, in that order (`dtools.dc.disc.VARIANTS`, the executable's order: entries 8-15 are `(REV)`, 16-23 `(MIR)`, 24-31 `(MIR-REV)`; see "Executable tables"). The variant names are reverse (R), mirror (M) and both (MR). `course_files(n, v)` gives the names:

| kind | name | content | state (Phase 5) |
|---|---|---|---|
| model | `CS{n}{v}_POL.BIN` | course geometry (world-space triangle strips) | decoded, lossless: `dtools/dc/pol.py` |
| textures | `CS{n}{v}_TEX.BIN` | course textures, 16-bit texels (twiddled or VQ, some mipmapped) | decoded, lossless: `dtools/dc/tex.py` |
| data | `CS_DATA{n}.BIN` (v empty), `CSDAT{n}{v}.BIN` | the arcade's course tables with z negated: AI lines, road line, steps, ground polygons, grid | decoded, lossless: `dtools/dc/data.py` |
| model_v | `VCS{n}{v}_POL.BIN` | second model set, the same objects with fewer strips; loaded in a Single Race (RAM), so not only a split-screen copy | decoded, lossless (same format as `CS`); purpose **Not found** (below) |
| textures_v | `VCS{n}{v}_TEX.BIN` | textures for the second set | decoded, lossless |
| ghost | `GHOST_{8*variant_index+n}.BIN` | ghost replay (40,120 or 32,768 bytes) | **Not found**: not decoded (outside M2's list; no session spent) |
| bg_lg | `LG_BG{n}.BIN` | the loading screen ("LADIES & GENTLEMEN START YOUR ENGINES"); see "Background screens" | decoded, lossless: `dtools/dc/screens.py` |
| bg_chp | `CHP_BG{n}.BIN` | a course still, probably Championship mode's; see "Background screens" | decoded, lossless; where it is shown **Not found** |
| bg_name | `NAME_BG{n:02d}.BIN` | a course still (two-digit number on disc); see "Background screens" | decoded, lossless; where it is shown **Not found** |
| map | `GAME_MAP.BIN`, `GAME_MAP2.BIN`, `GAME_MAP3.BIN` | course outlines and an icon sheet; a message sheet | decoded, lossless: `tex.game_map`, `tex.game_map3` |

**Not found** (families and questions that the M2 time boxes or scope left open; the discovery log,
`docs/spike/2026-11-dc-decode.md`, has the rest):

- *Ghosts.* `GHOST_*.BIN` were not opened; they are not in M2's list and no converter step needs them.
- *The `VCS` set's purpose.* Its model is in RAM during a Single Race on the same course as `CS` (Task 2), so it is not
  split-screen only. When the game picks which set to draw was not traced. A converter probably can leave it
  out, as the arcade has one model set.
- *Other root files* (`CSE_*`, `EC0`-`EC3_*`, `CAR*`, menus, fonts, audio): not part of M2.

Entry numbering: ghosts are numbered by `8 * VARIANTS.index(v) + n` with `VARIANTS = ("", "R", "M", "MR")`, giving `GHOST_0` to `GHOST_31` (`GHOST_8` is course 0 reverse, `GHOST_16` course 0 mirror). Background files are per course only (no variant) and always 614,400 bytes.

Other root files seen: `1ST_READ.BIN` (2,010,720), `GAME_MAP.BIN` (77,824), `GAME_MAP2/3.BIN`, `CSE_POL.BIN` (205,120) and `CSE_TEX.BIN` (course-select or ending scene), `EC0`-`EC3_POL/TEX.BIN`, car sets `CAR*`, menus, fonts, and `ADXB_*`/`ADXV_*.ADX` audio (excluded from `extract_all` by default along with the `DP*` directories: the exclude is a prefix of the path's first component, so `2_DP.BIN` and `SG_DPLDR.BIN` are kept).

### Sizes in bytes

Model (`CS*_POL.BIN`):

| variant | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 |
|---|---|---|---|---|---|---|---|---|
| (none) | 474,248 | 2,334,752 | 1,287,008 | 1,009,168 | 1,101,200 | 794,360 | 1,202,128 | 1,288,312 |
| M | 474,328 | 2,335,224 | 1,287,008 | 1,009,168 | 1,100,672 | 1,288,368 | 1,199,592 | 1,288,312 |
| R | 476,992 | 2,332,088 | 1,280,504 | 1,022,440 | 1,106,120 | 746,304 | 1,188,704 | 1,285,784 |
| MR | 477,008 | 2,333,016 | 1,280,504 | 1,022,440 | 1,106,048 | 798,832 | 1,189,352 | 1,285,784 |

Textures (`CS*_TEX.BIN`):

| variant | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 |
|---|---|---|---|---|---|---|---|---|
| (none) | 845,056 | 810,656 | 969,728 | 654,624 | 670,304 | 609,536 | 873,664 | 774,432 |
| M | 845,056 | 811,616 | 969,728 | 654,624 | 666,496 | 608,832 | 873,664 | 774,432 |
| R | 845,056 | 814,688 | 956,032 | 651,200 | 669,568 | 608,832 | 870,240 | 779,200 |
| MR | 903,072 | 814,688 | 956,032 | 651,200 | 669,568 | 608,832 | 870,240 | 779,200 |

Data (`CS_DATA*.BIN` / `CSDAT*.BIN`):

| variant | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 |
|---|---|---|---|---|---|---|---|---|
| (none) | 200,030 | 941,512 | 443,944 | 436,716 | 360,318 | 151,396 | 465,614 | 435,990 |
| M | 200,030 | 941,512 | 443,944 | 436,716 | 360,318 | 151,396 | 465,614 | 435,990 |
| R | 200,030 | 941,512 | 443,944 | 436,716 | 360,318 | 151,476 | 465,614 | 435,990 |
| MR | 200,030 | 941,512 | 443,944 | 436,716 | 360,318 | 151,476 | 465,614 | 435,990 |

Second model set (`VCS*_POL.BIN`):

| variant | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 |
|---|---|---|---|---|---|---|---|---|
| (none) | 328,216 | 2,165,320 | 1,125,912 | 744,576 | 900,120 | 699,144 | 857,176 | 1,105,760 |
| M | 327,752 | 2,165,632 | 1,125,912 | 742,832 | 899,424 | 699,144 | 854,728 | 1,105,760 |
| R | 329,368 | 2,162,992 | 1,127,488 | 741,192 | 904,896 | 703,368 | 842,896 | 1,105,968 |
| MR | 329,288 | 2,163,208 | 1,127,488 | 741,192 | 904,896 | 703,016 | 842,896 | 1,105,968 |

Second textures (`VCS*_TEX.BIN`):

| variant | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 |
|---|---|---|---|---|---|---|---|---|
| (none) | 787,488 | 783,680 | 939,552 | 640,224 | 651,872 | 596,480 | 841,376 | 914,016 |
| M | 787,488 | 784,640 | 939,552 | 640,224 | 644,992 | 596,480 | 841,376 | 914,016 |
| R | 787,488 | 787,712 | 938,208 | 636,800 | 648,064 | 596,480 | 837,952 | 919,488 |
| MR | 787,488 | 787,712 | 938,208 | 636,800 | 648,064 | 596,480 | 837,952 | 919,488 |

Ghosts (`GHOST_*.BIN`; 40,120 or 32,768 bytes):

| variant | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 |
|---|---|---|---|---|---|---|---|---|
| (none) | 40,120 | 40,120 | 40,120 | 40,120 | 40,120 | 40,120 | 40,120 | 40,120 |
| M | 40,120 | 32,768 | 32,768 | 32,768 | 40,120 | 32,768 | 40,120 | 32,768 |
| R | 40,120 | 40,120 | 40,120 | 40,120 | 32,768 | 40,120 | 32,768 | 32,768 |
| MR | 40,120 | 40,120 | 40,120 | 32,768 | 40,120 | 32,768 | 40,120 | 32,768 |

`LG_BG*`, `CHP_BG*` and `NAME_BG*` are 614,400 bytes for every course.

## Course table (Phase 1, Task 9; names Phase 5, Task 2)

Line lengths and outlines from `spike/dc_course_scan.py` (`maps.txt`, `overlay.txt` under `out/spike/dc/`); details
and pictures in `docs/spike/2026-11-beginner-overlay.md`, section "Dreamcast". "Map" is the outline's index in
`GAME_MAP.BIN`, measured (Phase 1's outline score, rerun in Phase 5 Task 4 on `tex.game_map`'s decode; each course's
best map is a different map; `GAME_MAP2.BIN` holds course n as its map n). Names, origin lines and select entries
from MAME's `dc` (`docs/spike/2026-11-dc-decode.md`, "Course names"): the names and origin lines read off the course select *(by hand)*; "entry" is the place on the select, and
which n an entry loads was measured (the race's main RAM holds that course's data, model and `VCS` model).

| n | debug label (`1ST_READ.BIN`) | line 0 length (DC units), turn with x right, z up | map | course | select entry, origin line | basis |
|---|---|---|---|---|---|---|
| 0 | `00:OVAL` | 1,564, clockwise | 7 (= 8) | **Three Seven Speedway** (arcade Beginner) | 0, FROM ARCADE BEGINNER | select + RAM; overlays the arcade's Beginner course (Phase 1: RMS 1.58 arcade units; Phase 5: exact, RMS 0), label, tri-oval outline |
| 1 | `01:CANYON` | 7,566, counter-clockwise | 0 | **Sea-Side Street Galaxy** (arcade Expert) | 2, FROM ARCADE EXPERT | select + RAM; geometry (the arcade Expert camera path, median 8.5 units). **Contradicts the label.** |
| 2 | `02:E_SIGHT` | 3,834, counter-clockwise | 1 | **Dinosaur Canyon** (arcade Advanced) | 1, FROM ARCADE ADVANCED | select + RAM; geometry (the arcade Advanced camera path, median 11.4 units). **Contradicts the label.** |
| 3 | `03:SAT1` | 3,438, counter-clockwise | 2 | **Desert City** (Saturn CCE) | 3, FROM CIRCUIT EDITION | select + RAM (Phase 1 guessed National Park Speedway) |
| 4 | `04:SAT2` | 2,980, counter-clockwise | 3 | **National Park Speedway** (Saturn CCE) | 4, FROM CIRCUIT EDITION | select + RAM (Phase 1 guessed Desert City) |
| 5 | `05:OVAL_TAB` | 1,612, clockwise | 4 | **Circuit Pixie** (Dreamcast) | 5, none | select + RAM; outline a rounded oval, 1 mile on the select |
| 6 | `06:OVAL_IKE` | 2,956, clockwise | 5 | **Rin Rin Rink** (Dreamcast) | 6, none | select + RAM; outline a triangle with a tail |
| 7 | `07:OVAL_TAB2` | 3,803, clockwise | 6 | **Mermaid Lake** (Dreamcast) | 7, none | select + RAM; outline a figure eight, as the race HUD's map |

- The select's order is not n's: entries 1 and 2 swap Dinosaur Canyon and Sea-Side Street Galaxy. Courses 1 and 2
  were first named by geometry against the arcade runs (Spike A's `budget/c1`, 4 laps, canyon walls: Advanced;
  `budget/c2`, 2 laps, the suspension bridge: Expert); the select and RAM agree. The executable's debug labels do not
  name the files reliably (`CANYON` on Sea-Side's geometry).
- Difficulty labels: the select shows none. The arcade's Beginner / Advanced / Expert appear only in the origin line
  of courses 0-2 (`FROM ARCADE BEGINNER` and so on); courses 3 and 4 say `FROM CIRCUIT EDITION`; 5-7 have no origin
  line. The circuit lengths the select gives (miles): 1, 4.922, 2.415, 1.850, 2.149, 1, 1.857, 2.389 for n 0-7.
- Variants: the select's TYPE item offers NORMAL, REVERSE, MIRROR, MIRROR REVERSE (the order of `VARIANTS`), with no
  separate entries; which files a REVERSE or MIRROR race loads was not measured.
- The executable holds no course names, nor do the background screens (only `LG_BG` carries text, the same line on
  all eight; "Background screens").

## Executable tables

`1ST_READ.BIN` loads at `0x8C010000`. Three pointer tables are indexed by entry `e = 8 * v + n`, with the variants
in the order `""`, `R`, `M`, `MR`:

| table | file offset | entries |
|---|---|---|
| debug labels | `0x18003c` | 32: `00:OVAL` … `07:OVAL_TAB2`, `08:OVAL(REV)` …, `16:OVAL(MIR)` …, `24:OVAL(MIR-REV)` … `31:OVAL_TAB2(MIR-REV)` |
| course files | `0x154700` | 32 (`CSnv_POL.BIN`, `CSnv_TEX.BIN`) pairs: 0 `CS0`, 8 `CS0R`, 16 `CS0M`, 24 `CS0MR`; then the `VCS` pairs |
| ghosts | `0x154680` | `GHOST_0.BIN` … `GHOST_31.BIN` |

The ghost table is an identity table (entry e points at `GHOST_e.BIN`), so the ghost of course n, variant v is
`GHOST_{8 * VARIANTS.index(v) + n}` with `dtools.dc.disc.VARIANTS = ("", "R", "M", "MR")`, the executable's order:
`GHOST_8` is course 0 reverse, `GHOST_16` course 0 mirror, `GHOST_24` course 0 mirror-reverse. The ghost-size table
above uses this numbering.

Course rules (Phase 5, Task 5; `docs/spike/2026-11-dc-decode.md`, "Course data"). Found by searching for the
arcade's values; "entry" is `e = 8 * v + n` as above:

| table | file offset (address) | layout | basis |
|---|---|---|---|
| checkpoints | lists `0x17e8ec`-`0x17e99b`, pointers `0x17e99c` (`0x8c18e99c`) | 32 u32 pointers to u32 lists (a count, then that many road segments), the 16 lists packed in entry order 0-15 straight before the pointers; entries 16-31 (M, MR) reuse 0-15's lists. Reader `dtools/dc/exe.py` (below) | n 1: 150, 257, 404, 539, 730 and n 2: 142, 275, the arcade's Expert and Advanced; n 0 and n 5: none. The pointer table's address is a literal at `0x8c035aa8` (code) |
| spawn records | two tables of 44 records of 24 bytes, `0x155078` (`0x8c165078`) and `0x155498` | u32 (5,048 or 5,049 in the first table, 6,128-6,130 in the second; not learned), u32 AI line, u32 start (segments ahead of the grid base), u32 speed setting (an integer, 260-336), u32 kind (0-2), u32 a sixth word (0-39; 768 in the last record; not learned) | the race's 20 cars hold pointers to the starts of AI lines 0, 1, 2, 3, 6, 5, 4, 7, 0, 1, 6, 7, 6, 7, 6, 7, 6, 7, 0, 1: records 0-19's line column (RAM, course 1). The line, start and kind columns are the arcade spawn table's with a few cars changed |
| grid bases | `0x17e814`, u16 | 32, by entry `8 * v + n` (`dc.exe.grid_base`) | 100, 984, 480, 458, 418, 123, 493, 493 (arcade: 100, 984, 482), the same in all four variants (M9 stage 1): a reverse course keeps its course's grid base, counted on its own road. Not traced |
| laps? | `0x17eb7c`, 32 u32 | 8, 2, 4, 4, 4, 8, 6, 6 by course, the same in all four variants (`dc.exe.laps`) | the arcade's mode-0 laps for n 0-2; the Single Race run had 4 laps on n 1, so it is not that mode's. Not traced |
| cars? | `0x17ebfc`, 32 u32 | 40, 30, 20, 30, 40, 40, 40, 20, the same in all four variants (`dc.exe.cars`) | the arcade's car counts for n 0-2; the Single Race run had 20 cars on n 1. Not traced |
| pit stalls (v0.1.2) | lists `0x17ba3c`-`0x17e2eb`, pointers `0x17e2ec` (`0x8c18e2ec`) | 128 u32 pointers: 16 groups of 8 stalls (one per car ident), each 7 waypoints of x, z, speed f32 (12 bytes), packed in pointer order up to the pointers. The groups go in pairs, a course's plain stalls then its R stalls, the pairs in the order Beginner, Advanced, Expert, then n 3-7 (`dc.exe.PIT_PAIRS`). Reader `dtools/dc/exe.py` (`decode_pit_stalls`, `pit_stalls`) | n 0-2's plain groups are the arcade's `CAR_TABLES_230E50` stalls value for value, in the arcade's frame (the courses' x and z as the converter writes them); n 3-7's lie on the converted pit lanes and run with the race, the R groups against it (`tests/test_release.py` `test_release_pit_stalls`). Found by the speeds 150, 100, 60. National Park Speedway's stalls 4-7 repeat 0-3; Desert City's stall 4 ends at (0, 0). Not traced in the Dreamcast program |

**The checkpoint table, word by word** (Phase 5 final fix wave; `dtools/dc/exe.py`, `tests/test_dc_exe.py`). Confirmed
by the values, not traced in the running game:

- A list's first word is a count: each of the 16 distinct pointers is the one before it plus 4 × (1 + its first word),
  and the last list (entry 15) ends exactly at the pointer table. Then come `count` u32 road segments, in race order,
  the lap line not included (the arcade's checkpoint tables hold the same segments for n 0-2, `golden.decode_course`
  ids 0-2). The segments' high halves are all 0 (the arcade keeps a raw high u16 there; read here as whole u32s).
- Entries 0-7 (plain): `()`, `(150, 257, 404, 539, 730)`, `(142, 275)`, `(136, 263)`, `(125, 257)`, `()`, `(263,)`,
  `(120, 363)`. Entries 8-15 (R) are the plain lists run backwards on the R course: segment s becomes
  `segments − s`, with `segments` from `CSDATnR.BIN` (990, 486, 464, 425, 499, 499 on n 1-4, 6, 7; all eight checked).
  Entries 16-31 hold the same 16 addresses again (a mirror keeps the segment numbering).
- Which entry the game reads: the code at `0x8c035a42` loads the literal at `0x8c035aa8` (the pointer table's
  address) and indexes it by a byte at `0x8c292e98`, masked, times 4 (a static reading of the SH-4 code; two more
  per-entry tables, `0x8c18ed04` and `0x8c18ee3c`, are read the same way and were not looked at).
- Reader: `exe.checkpoints(exe, n, v)` (the bytes of `1ST_READ.BIN`, course, variant) gives the segments;
  `exe.decode_checkpoint_table` / `encode_checkpoint_table` give the whole table and its bytes back (the lists from
  the first, then the pointers), and `exe.verify_lossless_checkpoints(disc)` checks the disc's table re-encodes byte
  for byte. Every malformed table is a `DcError` (a pointer outside the file or not on a list's start, a count past the
  pointer table).

Course-specific sound names in the executable: `SD_777_*` (Three Seven), `SD_SSS_SEAGULL`, `SD_SSS_HORSE_CRY`
(Sea-Side Street), `SD_DESERT_SAND/CHURCH/TRAIN`, `SD_RIN3_WATERFALL`, `SD_PIXY_HELI`.

## `GAME_MAP.BIN` (77,824 bytes), `GAME_MAP2.BIN`, `GAME_MAP3.BIN`

38,912 ARGB4444 texels in PowerVR twiddled order (Morton: bit 2i of the index is y bit i, bit 2i+1 is x bit i):
a 32 × 64 sheet (two 32 × 32 squares stacked: six coloured icons) then nine 64 × 64 maps, nothing left over. The
maps are the course outlines: map k is course k + 1 for k = 0–6, maps 7 and 8 (identical) are course 0 (measured: the
outline score of `maps.txt`, rerun on `tex.game_map`'s decode; `docs/spike/2026-11-dc-decode.md`, "Textures"). Each
map is drawn with x right and z up in the course's own coordinates.

- `GAME_MAP2.BIN` (69,632): the same sheet, byte for byte, then eight 64 × 64 maps of its own (thicker outlines),
  map k = course k for k = 0-7 (measured the same way).
- `GAME_MAP3.BIN` (131,072): not this layout; one 256 × 256 twiddled ARGB4444 sheet (network-play messages, player
  tags 1P-4P, icons), stored bottom row first like the course textures. The 256 × 256 reading rests on the picture
  looking right *(by hand)*; nothing else in the file confirms it.
- `dtools/dc/tex.py`: `game_map(blob)` returns the sheet (64 rows × 32) and the maps as uint16 texels, untwiddled,
  rows as stored; `encode_game_map` gives the file back byte for byte (`GAME_MAP.BIN`, `GAME_MAP2.BIN`);
  `game_map3(blob)` / `encode_game_map3` read and write `GAME_MAP3.BIN` (one (256, 256) uint16 array; any other
  length is a `DcError` naming the file and the offset).

## Background screens (`LG_BGn.BIN`, `CHP_BGn.BIN`, `NAME_BGnn.BIN`)

Phase 5, Task 6: `dtools/dc/screens.py` (`decode_screen`, `encode_screen`, `texels`, `to_rgba`,
`screen_from_texels`, `verify_lossless_screens`; all 24 files byte for byte). Discovery in
`docs/spike/2026-11-dc-decode.md`, "Screens".

Each file is one 640 × 480 RGB565 picture, 614,400 bytes, no header: eight textures one after the other, each
twiddled like the course textures (a rectangle as squares of its short side, in order) and each stored bottom row
first. The pieces cut the picture into bands from the top:

| piece | first texel | picture rows | picture columns | size |
|---|---|---|---|---|
| 0 | 0 | 0-255 | 0-511 | 512 × 256 (two 256-squares, left first) |
| 1 | 131,072 | 0-255 | 512-639 | 128 × 256 (two 128-squares, the lower one first: rows as stored) |
| 2 | 163,840 | 256-383 | 0-511 | 512 × 128 |
| 3 | 229,376 | 256-383 | 512-639 | 128 × 128 |
| 4 | 245,760 | 384-447 | 0-511 | 512 × 64 |
| 5 | 278,528 | 384-447 | 512-639 | 128 × 64 |
| 6 | 286,720 | 448-479 | 0-511 | 512 × 32 |
| 7 | 303,104 | 448-479 | 512-639 | 128 × 32 |

- "Stored bottom row first" is per piece: piece 0's first texel is picture (row 255, column 0), its second (254, 0),
  its third (255, 1). `screens.texels` / `to_rgba` return the picture top row first.
- Confirmed against MAME's frame buffer: seven loading screens (`LG_BG1`-`LG_BG7`), snapshotted during "LADIES &
  GENTLEMEN START YOUR ENGINES", match the decode at every pixel: each channel is the texel's or one step below.

| file | content *(by hand)* | indexed by | basis |
|---|---|---|---|
| `LG_BGn` | the loading screen: "LADIES & GENTLEMEN START YOUR ENGINES" over a blurred race scene, a colour per course | course n | measured: the Single Race on n shows `LG_BGn` (n 1-7) |
| `CHP_BGn` | a blue-tinted course still with cars, no text | the select's entry order, by its pictures (`CHP_BG1` Dinosaur Canyon, `CHP_BG2` Sea-Side Street) | not traced; named beside `CHP_NEXT*`, `CHP_CIR`, `CHP_SRS`, `ENDBG_*` in the executable ("CHP": Championship, probably) |
| `NAME_BGnn` | the same stills, brighter and softer (another still for 0), no text | course n, by its pictures (`NAME_BG01` Sea-Side Street, `NAME_BG02` Dinosaur Canyon) | not traced; named beside `NET_RANK`, `PTV1`, `PTV2` |

- The executable's name tables (identity, file offsets): `LG_BG` `0x154c24`, `CHP_BG` `0x154bac`, `NAME_BG`
  `0x154b54`.
- None of the 24 is the course-select backdrop (a 3D fly-over in a panel) or the Championship's name-entry backdrop.

## `CSnv_POL.BIN` and `VCSnv_POL.BIN` (course models)

All 64 files parse completely with this layout (every object ends exactly where the next begins; `headers.txt`).
Little-endian throughout.

Header (0x30 bytes):

| offset | field | values |
|---|---|---|
| 0x00 | format | `0x200` in all 64 files |
| 0x04 | build date, BCD `yyyymmdd` | e.g. `0x20001117` (2000-11-17) |
| 0x08 | build time, BCD `00hhmmss` | e.g. `0x00213342`; differs per variant |
| 0x0C | `ntex << 16 \| nobj` | CS0 116 objects, 117 textures; CS1 178, 176 |
| 0x10 | object table offset | `0x30` |
| 0x14 | texture table offset | `0x30 + 4 * nobj` |
| 0x18 | file size | equals the file length (all 64) |
| 0x1C | size of the matching `_TEX.BIN` | equals it (all 64) |
| 0x20 | 4 words | 0 |

Decoded and re-encoded byte for byte by `dtools/dc/pol.py` (`verify_lossless_pol`, all 64 files). Words 0-2 and 7-11
stay raw; words 3-6 are derived from the contents on encode. The objects start on an 8-byte boundary: the texture
table's end is followed by 0 or 4 zero bytes (39 / 25 files).

- Object table: `nobj` words, each `offset | 1` (bit 0 set in all 8,481 objects of the 64 files).
- Texture table: `ntex` × 16 bytes: u16 width, u16 height (16–256; square in all 64 files), u32 format word
  (`0x100`–`0x402`), u32 offset into the `_TEX.BIN`, u32 0. The format word and the texture data: see
  "`CSnv_TEX.BIN` and `VCSnv_TEX.BIN`".
- Object: u32 1, u32 kind: 1, 3 or 5 (7,449 / 1,000 / 32 objects over the 64 files; the kind is kept raw), bounding
  sphere (4 floats: centre, radius), then meshes until a zero word, then one word = the object's number of strip
  entries (true for all 8,481). Kind 3 holds the shadow (modifier) volumes and the sky domes (v0.1.1, on the five
  converted courses; CS4's 19:0, a volume, is kind 1). A volume's face colour is not a shade: black on CS3, CS5 and
  CS7, a per-volume ID on CS4 (a grey ramp 0.05-0.9, some pinkish) and (0, 0.02, 0) on CS6.
- Mesh: a 0x50-byte header then strip data:
  - (Every mesh float is finite; the texture index is -1 or inside the texture table, in all 64 files.)
  - +0x00 four words that look like PowerVR polygon parameters (`0x8000002c`/`0x8000003c`/`0x800000bc`…,
    `0x83000000`, `0x20xxxxxx`, `0xcc000000`/`0x88000000`/`0xd4000000`…); word 0 & 0x30 = 0x10 means the vertices
    carry no normal (three non-float words instead), 0x20/0x30 a unit normal.
  - +0x10 bounding sphere; +0x20 the texture index, `pol.Mesh.texture` (−1 none, else into the texture table: true
    of every mesh in the 64 files, and every texture is named by some mesh; Phase 5 Task 4); +0x24 0, 5, 2, −1 or
    −3, meaning not learned, kept raw as `pol.Mesh.kind` (Task 3 first took it for the texture index); +0x28 nine
    floats (colours, mostly 1.0; bit 0 of the first sometimes set); +0x4C byte length of the strip data.
  - Strips: u32 flags, u32 count, entries. Flag bit 3 set: a triangle list, `count` triangles (3 entries each);
    clear: a triangle strip of `count` entries. An entry is a vertex, 32 bytes (x, y, z, normal or three words,
    u, v), or an 8-byte re-use of an earlier vertex: `0x5ffxxxxx` then a negative byte offset `w`, the vertex being at
    `entry + 8 + w` (all 278,567 re-uses in the 64 files land on an earlier vertex; the first word's low 20 bits are
    not understood: they are not a function of `w`, and are kept raw).
  - In the meshes with `word 0 & 0x30 == 0x10` (5,791, exactly those whose +0x24 word is −3) the three normal words
    are not floats (two of them are NaN or infinite when read as floats); the decoder keeps them as three u32.
- Coordinates are world coordinates, y up; objects carry no transform. CS0 holds 11,867 vertices, 8,961 triangles;
  the sky is objects 0–1, a dome (bounding sphere centre (0, 118.6, 0), radius 566.8, `road.txt`; vertices within ±547.8 on x and z).
- `VCSnv_POL.BIN`: same header and layout; the same objects (CS0: 116, 112 with the same bounding sphere), some with
  fewer strips; 70 % of the triangles on course 0, 93 % on course 1. A reduced-detail copy (split screen, probably).

## `CSnv_TEX.BIN` and `VCSnv_TEX.BIN` (textures)

No header: the texture data that the matching `_POL.BIN`'s texture table points at (CS with CS, VCS with VCS, each
variant with its own model). Decoded and re-encoded byte for byte by `dtools/dc/tex.py` (`verify_lossless_tex`, all
64 files). Discovery in `docs/spike/2026-11-dc-decode.md`, "Textures".

Format word (the PVR file's, not the PowerVR texture control word):

| bits | field | values in the 64 files |
|---|---|---|
| 0-7 | pixel format (the control word's codes) | 0 ARGB1555, 1 RGB565, 2 ARGB4444 |
| 8-15 | layout | 1 twiddled, 2 twiddled + mipmaps, 3 VQ, 4 VQ + mipmaps |
| 16-31 | — | 0 |

Any other word (palettised, YUV422, bump, rectangular, stride, other bits) is refused by the decoder.

| layout | stored form | textures (RGB565 / ARGB4444 / ARGB1555) |
|---|---|---|
| twiddled | w × h texels, twiddled | 743 (479 / 199 / 65) |
| twiddled + mipmaps | 2 bytes (0), then the levels 1 × 1, 2 × 2, … w × w, each twiddled | 1,439 (1,137 / 250 / 52) |
| VQ | codebook (256 entries × 4 texels = 2,048 bytes), then (w/2) × (h/2) index bytes, twiddled | 1,917 (1,640 / 77 / 200) |
| VQ + mipmaps | codebook, then index levels 1 × 1 (1 byte), 2 × 2 (1), 4 × 4 (4), … w × w | 3,669 (2,683 / 796 / 190) |

- Texels are 16-bit little-endian. Twiddled order: bit 2i of the index is y bit i, bit 2i+1 is x bit i. A VQ entry's
  four texels are a 2 × 2 block in the same order, given as (row, column): (0,0), (1,0), (0,1), (1,1), i.e. the
  texel at x 0 / y 1 comes second.
- The mip chain's offsets are not the PowerVR's documented ones (1 × 1 at byte 6, 2 × 2 at byte 8): in these files
  the 1 × 1 is at byte 2 (measured by box-filtering each level to the next; the 2-byte preamble is kept).
- Placement: textures in table order, by increasing offset, from 0; each starts on a 32-byte boundary and the bytes
  up to it are zero (65,470 bytes over the 64 files). No data lies outside a texture; no two entries share bytes.
- Pictures are stored bottom row first (the lettering on signs reads upside down as stored; *by hand*). `to_rgba`
  returns rows as stored, so its row 0 is the picture's bottom row; the flip is decided in M3 (which converts Desert
  City's textures first) together with the UV convention.
- All 7,768 textures are square, 16-256.

## `CS_DATAn.BIN` / `CSDATnv.BIN` (course data)

Decoded and re-encoded byte for byte, all 32 files: `dtools/dc/data.py` (`decode_data`, `encode_data`,
`verify_lossless_data`, `road_cells`). Discovery: `docs/spike/2026-11-dc-decode.md`, "Course data".

**The file is the arcade's course tables.** Courses 0-2 hold the arcade Beginner, Expert and Advanced tables
(`docs/formats/model2-course-tables.md`, `model2-tgp-ground.md`) value for value with z negated, at scale 1: the AI
lines, the road line, the steps (pit flags included), the ground polygons and their attribute bytes. Only the grid and
its lists are rebuilt (for the negated z), and a few off-road cells moved (2 of 1,184 on course 0, 43 of 7,239 on
course 1, 3 of 2,986 on course 2; up to 22 units; no road cell).

Header: eight little-endian u32, the file offsets of sections H0-H7. H0's word is 0 (its section starts at 0x20,
always first). The other sections follow in an order that varies by course: H0, H4, H5, H3, H1, H6, H7, H2 on
courses 0 and 2-7; H0, H4, H5, H3, H1, H7, H2, H6 on course 1. Each starts where the one before it ends, rounded up
to its alignment (2 bytes for the u16 sections H2 and H7, 4 for the rest; the pad is zero). The file ends with the
last section. Every variant file has its own offsets and sizes: course 5's `R` and `MR` files hold one more cell and
four more list words (80 bytes).

| section | content | the arcade's |
|---|---|---|
| H0 | the 8 AI lines in storage order, each `count` points of 28 bytes, then its first 8 points again (the wrap) | the line table's lists (`AiPoint`, `AI_LINE_WRAP`) |
| H4 | the line directory: 8 × (u32 offset, u32 count); entry k is line k. Plain and `R` files list the stored lines in order; the `M` and `MR` files' directories vary (reversed: 0M-2M; in order: 0MR-2MR and 3M; rotated by one, entry k naming stored line k + 1 mod 8: 3MR and 4M-7MR) | the line table's pairs |
| H5 | the road line, `segments + 1` records of 164 bytes: u8 `lane_lo`, u8 `lane_hi`, s16 `curve`, 8 lanes of 20 bytes (x, y, z f32, pitch, heading, roll s16, u16 pad 0). Unused lanes are (0, 0, -0.0) with heading `0x1fff` | the record table (`RoadPoint`, `Lane`), without its header |
| H3 | one attribute byte per cell (the low nibble the surface kind: 1 road, then 2, 3, 5, 8, 10), a zero pad to an even size, then the steps: per road cell a u16 segment and u16 flags | `*block_ref_b` (padded with 0, not `0xcd`); the step table |
| H1 | 3 u32: segments (road points - 1), steps (= road cells), the file offset of the steps (inside H3) | the record table's n; the step count |
| H6 | the cells, 64 bytes: 4 corners and a centre (x, y, z f32 each), and a u32 direction word (the low half an angle, the arcade's + `0x8000`; the high half 0). The first `steps` cells are the road and the pit lane in step order; the rest (681 on course 0) are the other ground polygons (attribute kinds 2, 3, 5, 8, 10; which surface each is was not looked at) | the TGP ground polygons |
| H7 | the cell lists: a u16 count, then that many u16 cell indices. List 0 (offset 0) is empty, the list of an off-ground square | the TGP list block (u32 there) |
| H2 | the grid, 128 × 128 u16 row-major (row from z, column from x, 16-unit squares from -1024): the u16 offset in H7 of the list of the cells whose bounding box comes within 1.4 units of the square, 0 off the ground | `*block_ref_a` |

Field rules against the arcade (courses 0-2, every value):

| field | Dreamcast = |
|---|---|
| x, y, z (lines, lanes, cells) | x, y, −z of the arcade; exact |
| AI point speed | the arcade's within 6e-6 |
| AI point and lane pitch | negated |
| AI point and lane heading | the arcade's + `0x1fff` (mod `0x10000`) |
| AI point and lane roll; AI steer, lane (8.8), step; road `lane_lo`, `lane_hi`, `curve`; steps; attributes | equal |
| cell direction | `(arcade s16 + 0x8000) & 0xffff`, stored as a u32 (1,176 of 1,184 cells on course 0, all on courses 1 and 2) |

Step flags: `0x01` road, `0x42` the pit lane, `0x82` the pit box (one cell per course: course 0's is cell 152,
segment 5), `0x101` course 2's segments 174-232 (the arcade's unexplained `0x100` on Advanced). Every course 0-7 has a
pit lane and one box.

In RAM (MAME, course 1 racing): the file is loaded whole at `0x8c96f000`. The loader turns the header's offsets, the
directory's offsets and H1's third word into pointers, and writes 1 into H0's word. Every other byte stays as on disc,
and the game reads the sections in place.

## Coordinates against the arcade

**One frame, scale 1 (Phase 5, Task 7, `dtools/dc/match.py`).** arcade (x, y, z) = (x, y, −z)_DC: z negated, scale 1,
no rotation, no translation. The Dreamcast's world is the arcade's with z negated: right-handed against the arcade's
left-handed; y is up in both.

| Dreamcast n | arcade id | fit on the road cells (`match.fit_course`) | RMS | lap order | median road cell width |
|---|---|---|---|---|---|
| 0 | 0 Beginner | scale 1.0, angle 0.0, t (0, 0) | 0.0 (gate 1e-6) | cell 0 = step 0, same direction | 17.0 |
| 1 | 1 Expert | scale 1.0, angle 0.0, t (0, 0) | 0.0 (gate 1e-6) | cell 0 = step 0, same direction | 24.0 |
| 2 | 2 Advanced | scale 1.0, angle 0.0, t (0, 0) | 0.0 (gate 1e-6) | cell 0 = step 0, same direction | 20.0 |

- The fit pairs Dreamcast road cell k (four corners and the centre) with the arcade's ground road polygon k, in step
  order. It negates z explicitly and never fits a reflection, and it refuses a residual over its gate (`MatchError`).
- **The models are in the same frame.** Through the same transform, 88 % of CS0's road-surface vertices lie on an
  arcade road polygon edge (within 1e-3). A similarity fitted to the 493 that sit on a polygon corner is scale
  0.99999985, RMS 1.2e-3. The model's road rides +0.04 (median) above the arcade's ground.
- Phase 1's estimate, arcade (x, z) = s · (x, −z)_DC + t with s ≈ 1.01–1.04 (1.0396 from the mean racing line, 1.0286
  from line 0, 1.0120 from the road edges, `overlay.txt`), came from the AI lines against MAME's car lap and a trimmed
  nearest-vertex ICP onto the arcade's drawn polygons. It does not hold: at s = 1.01-1.04, 2-3 of the 2,038 road
  vertices lie on an edge.
- Model winding: `pol.triangles`' triangles run clockwise about their stored normals in the Dreamcast's frame and
  anticlockwise after the z negation (94 % of CS0's road triangles; docs/spike/2026-11-dc-decode.md, "Match").

## Mirror files

The `M` files (`CSnM_POL.BIN`, `VCSnM_POL.BIN`, `CSDATnM.BIN`) and `MR` files have the plain files' formats. They
hold the plain (or `R`) course mirrored in the Dreamcast's plane z = 0: every z is negated. Measured on all 8 courses
by `dtools/dc/match.py` (`mirror_report`; `tests/test_dc_match.py`, `test_mirror_files_all` and
`test_mirror_files_mr`). Per-course numbers and the evidence are in docs/spike/2026-11-dc-decode.md, "Mirror files".

| family | `M` against plain |
|---|---|
| model vertices | position and normal z negated (a raw-normal mesh's word 0 packs the normal as signed bytes x << 16 \| y << 8 \| z; z's byte negated), u, v kept: 90-100 % of the vertices (measured on the `CS` models); the rest moved or remodelled |
| model triangles | vertex order kept; the strip flags' low two bits swapped 2 ↔ 3 (1 kept). On the PowerVR these are the culling mode's values, by likeness |
| cells (H6) | index for index: corners in the order (1, 0, 3, 2) and the centre, z negated (exact on courses 0-2 but 2-4 off-road cells, and on 4-7; course 3 near); the direction word `0x8000 − d` |
| grid, lists (H2, H7) | row r → 127 − r, each square's list the same cells |
| steps, attributes (H3) | kept |
| AI lines (H0, H4) | `M` line k is plain line 7 − k (courses 0-3) or (k + 1) mod 8 (courses 4-7): every point's position exact, z negated (course 4: 56 points up to 0.81 off: points 140-146 of all eight lines, a seven-segment stretch the `M` file moved, with road slots (139, 6) and (144, 1) up to 0.016 off; Phase 10 Task 5). Courses 0-2: speed × 16/15, pitch kept, heading `0x1fff − h`, roll and steer negated, lane `0x700 − lane`, step kept. Courses 3-7: step and (but course 4) pitch kept; the other fields recomputed (headings within 2 of `0x2000 − h` modulo 0x4000) |
| road line (H5) | lane slot k is plain slot 7 − k, its attitude from that slot; `lane_lo`, `lane_hi` kept, even where asymmetric (course 1). Courses 0-2: positions exact, curve and roll negated, pitch kept, heading `0x1fff − h`. Courses 3-7: positions within 0.016, angles recomputed |

`MR` against `R`: the same geometry. The lines are kept in order on 0-2 and rotated on 3-7. On 0-2 the road's lane
slots are kept, its curve becomes `0x8000 − c` and its heading `0x2000 − h`.

**Against the arcade's mirror** (Phase 4, `dtools/arcade/mirror.py`). These agree: the normals, the direction word,
the grid flip, the steps and attributes, the road's slot reversal, pitch, and (courses 0-2) roll, curve and steer
negated. These differ:

- the plane: the arcade's is z = −0.5 (z → −1 − z);
- the ground's quad corner order: the arcade's is (0, 3, 2, 1);
- the AI lane: the arcade's is `0x6ff − lane`;
- the heading: the arcade's is `0x2000 − h`, one unit from the Dreamcast's in either frame;
- the AI line order: the arcade keeps it;
- the slot attitude: the arcade takes slot min(8 − k, `lane_hi`);
- `lane_lo` and `lane_hi`: the arcade reverses them;
- the speed: the arcade keeps it;
- the cull bits: the arcade's mirror keeps a polygon's corner order and attributes (`mirror_object`), and its
  mirrored Beginner drew so in MAME.

M4 derives its mirrors with the arcade's rules and uses the `M` files only as a check.

## What a converter reads

Phase 5, Task 9. For M3: the files and calls that give a converter each input, and where each input goes. The arcade
side is `docs/formats/model2-course-tables.md` ("What the converter must produce", the checklist) and
`docs/formats/model2-tgp-ground.md` (section 9). Read a file with `DcDisc.read(name)` (open the disc with `DcDisc.open`,
`.cue` or `.7z`; `course_files(n, v)` gives the names). Every decoder raises only `DcError` (a `DecodeError`), naming
the file and the offset. The converter takes the plain file (`v = ""`) for a course's base form; `R` and `M` files are
not needed (M4 mirrors with the arcade's rules).

**The transform.** arcade (x, y, z) = (x, y, −z)_DC: z negated, scale 1, no rotation, no translation (see "Coordinates
against the arcade"). Directions and angles: the arcade's pitch is the Dreamcast's negated, and its heading is the
Dreamcast's minus `0x1fff`; the cell direction word is the Dreamcast's minus `0x8000`; roll, steer, lane, step, curve,
`lane_lo`, `lane_hi` and the steps are equal. Measured on courses 0-2 against the arcade's own tables, value for value;
courses 3-7 have no arcade twin and use the same rules *(inferred: same files, same tool)*; on n 3 the model and data
frame and the AI heading rule are pinned (`tests/test_dc_match.py`, `test_course3_*`: 659 CS3 and 574 VCS3 vertices on
road-cell corners; headings less `0x1fff` within 2.2 units of `0x4000` a turn of the next point's direction). A
reflection reverses winding: see "Model winding" below.

### Course data, to the arcade's tables

`data.decode_data(blob, name)` gives a `CourseData`; `data.road_cells(d)` the road cells.

| Dreamcast | arcade table (`model2-course-tables.md` checklist; `model2-tgp-ground.md`) | encoder | notes | Phase 6 (M3a) |
|---|---|---|---|---|
| `lines` (H0, H4): 8 AI lines of `LinePoint` | AI lines (course record word 4) | `course.encode_line_table` | the wrap (the first 8 points again) is part of the arcade layout and of the file; the transform above; `speed` equal within 6e-6 | **Measured**: `convert.tables.ai_lines`; n 0-2 equal ids 0-2 field for field; n 3 drives in MAME. n 3's headings fit `0x2000` (+0.92 unit); the converter writes - `0x1fff` on every course |
| `road` (H5): `RoadPoint`, 8 `Lane`s | Road line (word 5) | `course.encode_road_line` | `segments = len(road) - 1`; unused lanes are zeros | **Measured**: `convert.tables.road_line`; n 0-2 equal ids 0-2 byte for byte; n 3: 465 points. Lane headings: one per road point, the across-road direction less a quarter turn (±1), n 0-2 and n 3 alike |
| `steps` (H3): `Step(segment, flags)` | Step table (word 1) | `course.encode_step_table`, `course.encode_pit` | flags `0x01` road, `0x42` pit lane, `0x82` the one pit box (every course has one); `0x101` on course 2 (the arcade's Advanced) | **Measured**: `convert.tables.steps`; n 0-2 equal; n 3: 527 steps |
| `attributes` (H3) | attribute block (`*block_ref_b`) | `tgp.encode_ground` | one byte per cell, low nibble the surface kind, 1 = road first | **Measured**: `convert.tables.attributes`; n 0-2 equal |
| `cells` (H6): `Cell(corners, centre, direction)` | ground polygons (TGP pool; `model2-tgp-ground.md` §9 item 1) | `tgp.encode_ground` | the first `len(steps)` cells are the road, in step order; corners and centre with z negated, direction minus `0x8000`; section 9 wants each polygon convex with negative signed XZ area, four finite corners inside −1024 ≤ x, z < 1024 at the road's height: check the converted cells against it | **Measured**: `convert.ground.convert_ground`; n 0-2 equal ids 0-2 but for the pinned off-road cells; n 3: 3,073 polygons, every AI and road lane point queries one; n 4-7 convert with their notched cells repaired (`ground.REPAIRED`, `model2-tgp-ground.md` §9 item 1), heights within 6e-5 of the TGP's rule on the unrepaired cells; the Dreamcast's own height routine (`0x8c034ee8`, a ruled surface from edge 0-1 to edge 3-2) differs from any diagonal split on non-planar cells (up to 0.24 units on Desert City's road) |
| `grid` (H2), lists (H7) | grid (`*block_ref_a`) and list block (§9 items 2-3) | `tgp.encode_ground` | the Dreamcast's are u16 and rebuilt for the negated z; rebuild them from the polygons with the TGP encoder and use these as a check (every cell is in its own centre square's list) | **Measured**: rebuilt with the arcade's rule (exact on ids 0-2); the Dreamcast's differ (526 of n 0's squares), kept as a check only |
| `1ST_READ.BIN` checkpoint list (`0x17e99c` pointers → u32 count, then road segments): `exe.checkpoints(disc.read("1ST_READ.BIN"), n)` | Checkpoint table (`CHECKPOINT_TABLES[id]`) | `course.encode_checkpoint_table` | read from the executable ("Executable tables", "The checkpoint table, word by word"; `tests/test_dc_exe.py`); n 0-2 equal the arcade's ids 0-2, n 3 is (136, 263); not traced in the game | **Measured in MAME**: n 3's list (136, 263), read at entry `8 * v + n`, drives the race clock (it rises one frame after the player reaches each segment and the lap line, `test_desert_city_drives`) |
| `1ST_READ.BIN` laps, car counts, grid base (low confidence) | laps, car count, grid base | `course.encode_laps`, `encode_car_count`, `encode_grid_base` | which mode reads these tables is **Not found**; take the arcade's values for the slot and tune in M6 | grid base **measured** (`build.GRID_BASES` from `0x17e814`, n 3: 458; the grid sits just behind the lap line, hence the standing start); laps and car counts still Not found, kept as the slot's |
| time extensions | Time tables | `course.encode_time_table` | **Not found** in the executable; M3 keeps the slot's own times, M6 balances | still **Not found**; the slot's times are kept (n 3's playtest set raises the start words to 12000) |
| per-car spawn records (`1ST_READ.BIN` `0x155078`) | the two spawn tables, shared by every course | `course.encode_spawn_record` | do not copy: the arcade's are shared, and changing them changes every course | **Measured**: not copied, shared tables unchanged |

Not in the Dreamcast data and not decoded by M2, so M3 sets them from the arcade side. Measured in Phase 6 (M3a): the
per-segment flags (Not found on the disc; `n + 1` zeros), the course record's words 6-7 (the grid pointer,
`tgp.repoint_record_raw`), the HUD map (traced from the road line, `build.hud_outline`, not the `GAME_MAP*.BIN`
outline), the view-test and culling tables (shared, kept; confirmed with scenery in Phase 7). Texture points (from the
model's UVs) are measured in Phase 7 (M3b): see "Models and textures" below. Not used: the backdrop (`LG_BG`/`CHP_BG`/`NAME_BG` are loading and menu pictures,
not the in-race backdrop).

### Models and textures

- `pol.decode_pol(blob, name)` gives a `Model`; `pol.triangles(model)` yields `(a, b, c, object index)` with re-uses
  resolved, each vertex `Vertex(x, y, z, normal, u, v)`. Every mesh names its texture by `Mesh.texture` (an index into
  `Model.textures`, −1 for none). The sky is objects 0-1 on course 0 (a dome of radius ~567); the road surface is a
  few objects (on course 0, 89-110). Lists and strips: the layout in "`CSnv_POL.BIN`".
- **Model winding.** `triangles` follows the stored strip order with every odd triangle swapped. In the Dreamcast's
  frame the faces are clockwise about the stored normals; after the z negation they are anticlockwise with the right-hand
  normal up (94 % of CS0's road triangles). **Measured (Phase 7, M3b):** the strip flags' bits 0-1 are the PowerVR
  culling mode and decide the seen side (mode 2 keeps the corner order after the z negation, mode 3 reverses it, modes
  0-1 are double-sided); the converter writes corners anticlockwise about the seen side with the stored normal
  -Newell / |Newell|. `convert.model.convert_model`; `docs/spike/2026-11-convert.md`, "Model".
- `tex.decode_tex(blob, entries, name)` gives the textures of one `_TEX.BIN` (`entries` is the model's texture table,
  `Model.textures`). The converter must read these texture formats, all square, 16 to 256 texels:

  | layout (bits 8-15 of the format word) | pixel (bits 0-7) | textures in the 64 files |
  |---|---|---|
  | 1 twiddled | 0 ARGB1555, 1 RGB565, 2 ARGB4444 | 743 |
  | 2 twiddled + mipmaps | the same three | 1,439 |
  | 3 VQ (256-entry codebook of 2 × 2 blocks) | the same three | 1,917 |
  | 4 VQ + mipmaps | the same three | 3,669 |

  Palettised, YUV422, bump, rectangular and stride textures do not occur, and the decoder refuses any other word.
  `tex.to_rgba(t, level)` gives RGBA; rows are as stored, **bottom row first**, so the picture is `rgba[::-1]`; the flip
  belongs with the UV convention (M3's texture step). The arcade side is 4-bit indices with a hue from the header's
  palette entry and a luma table (`model2-textures.md`), widths `32 << n`: the converter quantises the 16-bit colours,
  and a 16-texel texture needs enlarging to 32.
  **Measured (Phase 7, M3b):** each texture keeps its own size (none enlarged or reduced on Desert City), its palette
  entry and luma table are the pair whose decoded mean colour is nearest the Dreamcast's mean (one hue per texture;
  a face-colour tint chooses another entry per mesh), rows flipped once; the Dreamcast's
  alpha-blended decals (shadows) have no arcade equivalent and are dropped, and the sky domes, shadow volumes and
  run-time props at the origin are not scenery. The model's per-vertex (u, v) become texture points at 1/8 texel with
  the mirror and wrap bits per polygon. `convert.textures`, `convert.model`, `build.model_scenery`;
  `docs/spike/2026-11-convert.md`, "Textures (M3b)", "Texture points (M3b)", "Scenery assembled".
- `tex.game_map(blob)` / `tex.game_map3(blob)`: the outlines (64 × 64 ARGB4444) and the message sheet. The map of
  course n in `GAME_MAP.BIN`: map n − 1 for n ≥ 1, maps 7 and 8 (identical) for course 0 (map k is course k + 1). In
  `tex.game_map`'s tuple, whose index 0 is the sheet, that is index n for n ≥ 1 and indices 8 and 9 for course 0. In
  `GAME_MAP2.BIN` map n is course n (tuple index n + 1). The arcade's HUD map is a polygon outline
  (`course.encode_hud_map`), so a converter traces the outline's polygons from the road cells or the map picture.

### The density proxy

The Phase 1 recommendations name it ("Calibrate the density proxy"). The proxy is the quad-equivalents of a Dreamcast
course within 200 arcade units of points along its racing line (strip triangles count 1/2, list triangles 1, sky
left out; `docs/spike/2026-11-beginner-overlay.md`, "Polygon count against the budget"). The arcade's bar is drawn
polygons per frame (Beginner 3,533, Expert 4,153, Advanced 4,251). The pairs already in hand, for the courses that
exist in both games (the densest stretch's proxy against the arcade's bar for the same course):

| course | proxy, Dreamcast form (max within 200) | arcade bar (drawn per frame) | bar / proxy |
|---|---|---|---|
| 0 Beginner | 1,875 | 3,533 | 1.88 |
| 1 Expert | 6,920 | 4,153 | 0.60 |
| 2 Advanced | 3,302 | 4,251 | 1.29 |
| 3 Desert City, converted (Phase 7 Task 6) | 2,222 | 4,779 drawn (over the bar) | 2.15 |
| 3 Desert City, thinned (Phase 7 Task 7) | 2,222 | 4,078 drawn | 1.84 |
| 3 Desert City, thinned, HUD outline merged (Phase 7 Task 8; final) | 2,222 | 3,677 drawn | 1.65 |

The ratio runs from 0.60 to 1.88, so the proxy alone does not predict the bar. The calibration for M3: convert the
first course, run it in MAME with the geometry loggers (`geo_log.lua`, `renderrate.lua`; the drawn-polygon count per
display list), and add its (proxy, drawn) pair as the fourth point. The thinning target is
then the drawn count against the bar, with the proxy used only to place where to thin. (Derived from the Phase 1
table; the arcade side is a maximum over a recorded lap, the Dreamcast side the densest stretch of line 0.)

Desert City's rows are the converted course's own drawn maximum over the player's full lap (cars and HUD included;
`docs/spike/2026-11-convert.md`, "Desert City seen" and "Thinning"), not a bar. Converted whole (less the sky domes,
shadow volumes and decals) it draws 2.15 per proxy polygon, past every original's ratio; with the run-time props
dropped and ten objects thinned to 70-90 % (`build.THIN`) it draws 1.84, within the bar (4,251) and Beginner's ratio; with the HUD outline merged from 465 to 64 quads (Task 8) every count falls by 401: 3,677, 1.65.
So for a course converted the same way the proxy times about 2 estimates the drawn maximum before thinning, and a
proxy past about 2,100 (4,251 / 2) says thinning will be needed; where to thin is found from the drawn count per
frame (Task 7 replayed the measured lap), not from the proxy.

## Car sets (`CAR{c}{level}_POL.BIN`, `_TEX.BIN`; v0.4)

Reader: `dtools/dc/car.py` (`car_files`, `car_parts`, `wheel_positions`). The executable's file table at `0x154908`:
`CARCOM`, then level 0 and level 1 of H, G, F, L, U (Hornet, Grasshopper, Falcon, Lightning, Unicorn), level 0 and 1
of P, N, C, J, B (the bonus cars), then level 2 of all ten. The files are course model files (`dtools.dc.pol`); their
textures add the layout `0x0d`, a twiddled rectangle (`tex.LAYOUTS`, e.g. a 64 x 16 `0xd01`). `CARCOM_POL.BIN` does
not decode (a non-finite vertex at `0x3ad98`); **not found:** what it holds.

A level-0 set has 80 objects, the same roles by index on the five stock cars (read from a rendered sheet of each
object, `spike/v04/car_sheet.py`; the same scale as the arcade's car, about 2.0 x 1.3 x 4.8, y up):

| objects | part |
|---|---|
| 0-8 | the body shell (about 640 triangles): 0 undamaged, 1-8 dented versions |
| 9-16 | the floor with the bumpers: 9 undamaged, 10-16 damaged versions |
| 17 | the inside of the shell |
| 18-26 | the decal layer over the body, one per body version |
| 27, 28 | the windscreen; the cabin with its cage |
| 29-31 | three untextured hulls (use **not found**: shadow or collision) |
| 32-51 | twenty roof-and-door number panels (two sets of ten colours) |
| 52-57 | bumpers and skirts as loose pieces |
| 58-62, 63-67 | the wheels, left and right, round the origin (160 triangles): a still one and four more blurred |
| 68, 69-73, 74-77, 78 | the driver's seat view: the windscreen frame, five bonnets, the dashboard (four), the cabin |
| 79 | a 4-triangle marker |

Level 2 is a lighter build of the same (54 objects). **Not found:** what level 1 is for.

Wheel hubs: `1ST_READ.BIN` `0x124ac0`, per car four (x, y, z) f32: front left, front right, rear left, rear right
(the Hornet ±0.81, 0.343, 1.41 / −1.34; the Grasshopper ±0.75, 0.343, 1.333 / −1.34). The cars' corner boxes are at
`0x11fc0c` (per car four (x, 0.33, z)).
