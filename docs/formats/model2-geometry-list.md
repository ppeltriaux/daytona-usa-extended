# Model 2 geometry display list

Derived from MAME 0.289 `src/mame/sega/model2_v.cpp` and `src/mame/sega/model2.cpp`, BSD-3-Clause, copyright the
MAME team: `geo_parse` (model2_v.cpp line 2327), `geo_process_command` (line 2286) and the `geo_*` command
functions (lines 1704–2285); `geo_w` (model2.cpp line 822), the memory map (`model2_base_mem`, line 1027;
`model2_tgp_mem`, line 1254) and `screen_vblank` (line 2420). Implemented in `dtools/arcade/geo.py` (`walk`,
`emit`, `count_polygons`, `parse_log`); logged by `dtools/mame/lua/geo_log.lua` and `renderrate.lua`.

## The buffer and the i960 addresses

| i960 address | What | MAME |
|---|---|---|
| `0x00900000`–`0x0091ffff` | buffer RAM: 128 KB = `0x8000` little-endian 32-bit words, the display list (mirrored by `0x60000`) | `bufferram` |
| `0x00800000`–`0x00800fff` | command port, function `(offset >> 4) & 0x3f`: a write at `offset & 0xf == 0` pushes `function << 23 \| (data & 0xfffff)`, a write with data bit 31 set pushes `function << 23 \| (data & 0x800fffff)` (a jump), at the write start address; other writes are dropped | `geo_w` |
| `0x00801008` | write start address (byte offset, `& 0xfffff`); every pushed word goes there, then it advances by 4 | `geo_w` |
| `0x00802008` / `0x00803008` | reads return the write / read start address | `geo_r` |
| `0x00803008` | **read start address** (byte offset, `& 0xfffff`): where `geo_parse` starts | `geo_w` |
| `0x00804000`–`0x00807fff` | FIFO: each write is pushed at the write start address (unless geo control bit 31 selects program upload) | `geo_prg_w` |
| `0x0098000c` | video control (`videoctl_w`); reads return `(frame-number bit) << 2 \| (videoctl & 3)` | `videoctl_r/w` |

The read start address is a **byte offset** into the buffer: `geo_parse` begins at word `(read_start & 0x1ffff) / 4`,
so `walk(words, read_start // 4)`. `parse_log` yields `read_start` as logged (bytes).

## When the list is parsed

At every vblank (`screen_vblank`), `geo_parse` runs when video control bit 0 is clear, and on even frame numbers
only when it is set (the 30 Hz mode). The parse reads the whole list in one go: the i960 can rebuild the buffer
between parses. The screen runs at 16 MHz / (656 × 424) = 57.52 Hz, visible lines 0–383, vblank lines 384–423.

The screen is configured `VIDEO_UPDATE_AFTER_VBLANK` (model2.cpp line 2517), so `screen_vblank(1)` and with it
`geo_parse` run at vblank **start** (screen.cpp `vblank_begin`), while `frame_update` and Lua's `frame_done` run at
vblank **end** (`vblank_end`), 40 lines later. `screen_vblank` raises the vblank IRQ right after `geo_parse`, and
the i960 starts the next list inside that gap (measured below), so a buffer read at `frame_done` is not, in general,
what was parsed. The loggers therefore sample at vblank start (next section).

`geo_parse` loop: while no end command was seen, the read pointer is inside the buffer (`< 0x8000` words) and fewer
than `0x8000` words have been read as opcodes (jumps included):

- an opcode with **bit 31 set is a jump**: continue at word `(opcode & 0x1ffff) / 4` (a byte offset);
- otherwise the command is **`(opcode >> 23) & 0x1f`**; the other bits are ignored by the parser (the command port
  puts its 20 data bits in the low bits, bits 29–30 carry an eye mode for 0x01).

A non-looping list always reaches its end before the cap (each word is read at most once); the cap stops a loop.
A command whose payload runs past the buffer's end makes MAME read past `bufferram`; `walk` stops there.

## Commands

Word counts are after the opcode. `count` is the word read as such; "f" = IEEE float.

| Cmd | Name (MAME function) | Words | Carries / `Cmd.args` |
|---|---|---|---|
| 00 | NOP (`geo_nop`) | 0 | — |
| 01, 11 | Object Data (`geo_object_data`) | 4 | `tpa` texture point address, `tha` texture header address, `oba` object address, `obc` object count |
| 02, 12 | Direct Data (`geo_direct_data`) | 8 + records + 1 | `tpa, tha`, two points of 3 words (`p0, p1`), then records `attr, luma, distance, point (3)[, point (3) if attr & 1]` until an `attr` with `(attr & 3) == 0` (consumed). `records` |
| 03, 13 | Window Data (`geo_window_data`) | 6 | six `0x0XXX0YYY` coordinates: start, end, four vanishing points. `coords` = `[(x, y)]` |
| 04 | Texture Data Write (`geo_texture_data`) | 2 + count | `address` (start address / DSP id), `count`, `count` words |
| 14 | Log Data Write (`geo_log_data`) | 2 + count | as 04 (each word is four luma-table bytes) |
| 05, 15 | Polygon Data (`geo_polygon_data`) | 2 + count | `address`, `count`, `count` words written to polygon RAM 1 if `address & 0x01000000` else RAM 0, from word `address & 0x7fff` |
| 06 | Texture Parameters (`geo_texture_parameters`) | 2 + 2·count | `index` (word >> 2), `count`, then per entry a parameter word (diffuse, ambient, specular scale, specular control bytes) and a distance coefficient (f) |
| 07, 17 | Geo Mode (`geo_mode`) | 1 | `mode`; `mode & 3` picks the object layout (polygon-ROM doc). Engine state: it holds across commands and frames |
| 08, 18 | Z-Sort Mode (`geo_zsort_mode`) | 1 | `zsort` (pushed `>> 8` to the rasterizer) |
| 09, 19 | Focal Distance (`geo_focal_distance`) | 2 | `focus` = (x, y) f |
| 0a, 1a | Light Source Vector (`geo_light_source`) | 3 | `light` = (x, y, z) f |
| 0b, 1b | Transformation Matrix (`geo_matrix_write`) | 12 | `mat`: 12 f, rotation `[0..8]`, translation `[9..11]` |
| 0c, 1c | Parallel Transfer Vector (`geo_translate_write`) | 3 | `translate`: rewrites `mat[9..11]` |
| 0d | Data Memory Push (`geo_data_mem_push`) | 2 | `address`, `count`; the push loop is commented out in MAME, so only these two words are read |
| 0e | Geo Test (`geo_test`) | 33 + 3·blocks | 32 FIFO-test words (`1 << i`), `blocks`, then per block `address` (`& 0x7fffff`), `count`, `checksum` (over polygon ROM). `fifo_ok`, `blocks` |
| 0f, 1f | End (`geo_end`) | 0 | ends the list |
| 10 | Dummy (`geo_dummy`) | 1 | `value`, ignored |
| 16 | LOD (`geo_lod`) | 1 | `lod` f |
| 1d | Code Upload (`geo_code_upload`) | 2 + 3·count | `flags`, `count`, three words per 64-bit program word (unimplemented in MAME) |
| 1e | Code Jump (`geo_code_jump`) | 1 | `address` (unimplemented in MAME) |

All 32 values are defined, so there is no unknown command: `walk` stops early only at the cap, the buffer end or a
payload past the buffer (`Cmd(-1, offset, {"stopped_on": kind | None, "reason": ...})`).

## Writing a list (`emit`)

`emit(cmds)` is the inverse of `walk`: the words of a command list in order, each command's opcode word then its
payload, with **no jumps** (for a walked list, exactly the words walk read, jump words left out). Walk keeps what its
fields do not show, so that the round trip is exact: `Cmd.opcode` (the opcode word as read, data bits included; not
part of equality; a hand-built `Cmd` without it is written as `kind << 23`) and these `args` keys: window `words`
(the six words, for the bits outside the 12-bit fields), texture parameters `index_low` (index bits 0-1), direct
data `end` (the attribute word that ended the records), geo test `fifo` and `block_high` (address bits above 23),
code upload `data`. Missing keys are written as 0 (the FIFO pattern for `fifo`); floats as single precision.
`GeoError` for a stopped walk's `END` entry (not a whole list), a kind outside `0x00-0x1f`, an opcode that is a
jump or whose bits 23-27 are not the kind, a word past its field (u32; window coordinates 12 bits, a texture
parameter index 30 bits, a test-block address 23 bits), a count that does not match its data, a float that does not
fit single precision, and a direct-data record attribute with `attr & 3 == 0`. Checked on every list of a 30 s
`geo_log.lua` run (`tests/test_geo.py`).

## Objects and polygon RAM

Command 01/11 selects the object's source, in order: `oba & 0x01000000` → polygon RAM 1 at `oba & 0x7fff`;
`oba & 0x00800000` → polygon ROM at word `oba & 0x3fffff`; else polygon RAM 0 at `oba & 0x7fff`. `obc == 0` means
`0xfffff`. The object layout is in `model2-polygon-rom.md`.

The two polygon RAMs are `0x8000` words each, filled only by 05/15 and kept across frames; in MAME's `geo_state`
RAM 0 is followed by RAM 1, so an object or upload running past RAM 0 continues into RAM 1. `PolyRam` keeps that
image and the geometry mode; `count_polygons` applies a list's uploads and mode writes in order, so pass every
logged list through it in log order (uploads in frames that were not logged are missed).

`count_polygons` counts **drawn** polygons: records whose link bits `(attr >> 8) & 3` are not 0 (the rasterizer
culls link 0), in objects and in direct data alike.

## Log formats

- `geo_log.lua` (`DAYTONA_GEO_LOG`, from `DAYTONA_GEO_FROM` every `DAYTONA_GEO_EVERY` frames):
  `G <frame> <read_start_hex> <videoctl_hex> <0x8000 words as 8 hex digits each>`; `read_start` is the last value
  written to `0x00803008` (a write tap on the i960 program space), `videoctl` a read of `0x0098000c`.
- `renderrate.lua` (`DAYTONA_RATE_LOG`, every frame): `R <frame> <writes to 0x00803008 this frame>
  <videoctl & 3> <FNV-1a 32 of the 256 words at the read start, each as 4 little-endian bytes>`.

Both take the buffer, the read start and video control **as `geo_parse` saw them**, through
`dtools/mame/lua/geo_sample.lua` (shared, state in `_G.daytona_geo`): write taps on `0x00800000`–`0x00807fff`
(command port, write and read start, FIFO; the FIFO and the port reach the buffer through `push_geo_data`, which a
buffer-only tap would miss) and on `0x00900000`–`0x0097ffff` (the buffer and its mirrors). The state is captured at
the first such write that falls in the vblank (a tap runs before its write lands, so no i960 write has reached the
buffer between vblank start and that write), or at `frame_done` when the vblank saw none. The TGP is not tapped: its
bank register (bit `0x400000`) can map the buffer into its data space (`model2.cpp` lines 471–514), so a TGP write
in the gap would go unseen: "nothing writes the buffer before the first tapped write" holds for the i960 only. The
vblank test compares `screen:time_until_vblank_end()` with `time_until_vblank_start()`: inside the i960's timeslice
the scheduler time is the CPU's local time, and the CPU does not run past a pending timer by more than its current
instruction, so a write is classed before or after `geo_parse` to within a few cycles (an i960 store in the last
instruction of a timeslice can land just past vblank start; negligible). The write count in `R` lines runs from one `frame_done` to the next.

## Observed on Daytona USA (2026-11-07, MAME 0.289)

From a 60 s `autoplay.lua` run on course 0 (geometry log from frame 1400 every 10 frames, rate log every frame), a
3 s `geo_log.lua` run logging every frame from frame 1, a 34 s run logging every frame from 1900, and a 60 s run with
probe taps classing every geometry write by whether it fell in the vblank (all sampled at vblank start):

- **Writes between vblank start and `frame_done`** (taps as above, 60 s run): in each of the 1,561 race frames the
  i960 writes there: the write start address once (to the other half of the buffer, see the read start), about 2
  command-port and 7 FIFO words (the head of the next list, pushed into that other half) and 7–8 direct writes of
  `0xffffffff` to word `0x7ffc`. Over the whole run 2,985 of 3,452 frames had such writes, and 741 read-start writes
  fell in the gap. In this run the gap writes left the parsed list itself intact (a `frame_done` sample differed from
  the vblank-start sample only in words `0x7ffc`–`0x7ffe`, and the counts below came out the same), but nothing in
  the driver guarantees that, so the loggers sample at vblank start.
- **Boot uploads.** Polygon RAM is filled once, at boot: the lists parsed at frames 79–110 carry sixteen 05 uploads
  of 2,048 words to RAM 0 (`0x0000`–`0x7fff`), each upload in two consecutive lists; RAM 1 is never written. Two
  runs filled identical RAM images. The same lists
  set the mode (07) to 3, then 1: the mode is 1 from then on (mode 1 reads the same words as mode 0). No upload or
  mode write follows in the attract mode, the selects or the race. The race draws RAM 0 objects (`oba` `0x4`,
  `0x91`, `0x10a`, `obc` 5000) every frame, so a polygon count needs the boot lists: fill a `PolyRam` from a
  `DAYTONA_GEO_FROM=1 DAYTONA_GEO_EVERY=1` log of the first 3 s (the boot is the same in every run), then count
  the later log with it.
- **Read start: double-buffered.** The i960 builds each list in one half of the buffer while the other is parsed,
  and points the read start at it: in the race it alternates **`0x0`** (even frames) and **`0x10000`** (odd frames,
  word `0x4000`), checked on 57 consecutive frames (1900–1956); the boot log holds 66 lists at `0x0` and 107 at
  `0x10000`. A log taken every 10 frames therefore sees one half only (all 206 lists of the 60 s run were at `0x0`).
  It writes `0x00803008` on 54 of the boot's 473 frames (27–499); from ~500 on once per frame, except 27 frames
  without a write (around the coin, the selects and the race intro) and 4 frames with two.
- **Video control** `& 3` is **0** in every frame: `geo_parse` runs at every vblank, 57.52 Hz, never the 30 Hz mode.
  In the race (frames 1890–3452) the rate log has 1,563 frames, 1,563 read-start writes and 1,562 changes of the
  list-head hash: one new list per frame, i.e. the render cadence is the full 57.52 Hz.
- **Commands per list** (race): 01 objects 24–76 (97 % from polygon ROM, the rest RAM 0), 0b matrices 14–53,
  09 and 0a 4–7 each, 03, 04, 08 once, then 0f. No 02 (direct data), no jump; every logged list ends with 0f.
- **Drawn polygons per list** in the race (frames 1890–3450, 157 lists, polygon RAM filled from the boot log):
  median 2,358, 95th percentile 3,264, maximum 3,488.
