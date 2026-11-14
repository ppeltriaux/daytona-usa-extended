# Model 2 Daytona USA: the TGP ground (`copro_data`)

The ground the cars stand on: the i960 asks the TGP (MB86234, MAME `mb86233`, device `:copro_tgp`) for the height
and the polygon under (x, z) with function `0x36` (`docs/formats/model2-course-tables.md`, "The ground under the
road"), and the TGP answers from per-course lists and polygons in its data ROM. Decoder and encoder:
`dtools/arcade/tgp.py` (`decode_ground`, `encode_ground`, `ground_spans`, `verify_lossless_ground`; lossless on the
three courses). The runs and listings behind it: `docs/spike/2026-11-tgp-ground.md`, "Layout" and "Course
selection". The layout is the one of Virtua Racing's Model 1 ground (`../virtua-racing/docs/formats/
model1-tgp-collision.md`), checked here word by word.

## 1. Addressing

TGP addresses are 32-bit **words**. `copro_data` word `w` is the little-endian u32 at byte `4 w` of
`RomSet.copro_data`, whose two 2 MB halves are identical (`0x80000` words each; `RomSet.with_patch` writes both;
the encoder writes word addresses of the first half). The TGP reads it through a bank register (rf word 3): I/O
address `offset16` reads word `(bank & 0xff0000) | offset16`, and bank bit `0x800000` selects `copro_data` (MAME
`model2.cpp`, `copro_tgp_memory_r`). The program sets the bank to the full word address plus `0x800000` and walks
on with a 16-bit post-incremented index, so a list or a polygon that crosses a multiple of `0x10000` words wraps to
the start of its window. No original crosses one; the encoder refuses one that does.

## 2. Layout

| Words | Content |
|---|---|
| `0x00`-`0x0f` | 4 x 4 identity matrix (f32) |
| `0x10`-`0x1f` | slot table: list block address of slots 0-15 |
| `0x20`-`0x2f` | slot table: polygon pool address of slots 0-15 |
| `0x30`-`0x1299` | Beginner list block (slot 0) |
| `0x8000`-`0xd639` | Expert list block (slot 1) |
| `0x10000`-`0x12831` | Advanced list block (slot 2) |
| `0x18000`-`0x19c88` | slot 3's list block (no course selects it) |
| `0x19c90`-`0x1e68f` | Beginner pool, 1,184 polygons |
| `0x1e690`-`0x3aaff` | Expert pool, 7,239 polygons |
| `0x3ab00`-`0x4659f` | Advanced pool, 2,986 polygons |
| `0x465a0`-`0x4a24f` | slot 3's pool, 971 polygons |
| `0x4a250`-`0x7ffff` | `0xffffffff`: free (882,368 bytes from byte `0x128940`) |

Between the list blocks: zeros, and `0xcdcdcdcd` from `0x19c89` to the first pool. Slots 4-15 repeat slot 0's
addresses. Slot 3 has 1,936 lists and 971 polygons; no course record points at a grid for it.

## 3. Course selection

**Slot = course id.** At course load the i960 (`0x4534`: `ldib 0x501460,r10`; `0x46d4`-`0x46dc`) writes `0x0d0d`
to `0x008800d0` (TGP function `0x0d`) and `r10` to the FIFO. The TGP's handler (`0x1b8`) calls `0x7cb` with it,
which sets `$0x69 = ROM[0x10 + slot] + 0x800000` (the list block) and `$0x6a = ROM[0x20 + slot] + 0x800000` (the
pool). Init calls `0x7cb` with slot 0, so queries made before a course loads (the attract mode's) use Beginner's.
`Ground.base` is the slot. A new course with id `n` needs slot `n` filled.

## 4. Grid and lists

The i960's grid (`*block_ref_a`, 128 x 128 `u16`, 16-unit cells from -1024; `GRID_CELLS`) holds per cell 0 (off the
ground) or the **word offset of the cell's list in the list block**: `count:u32`, then `count` polygon indices
(`u32`). Word 0 of a block is a count of 0, the list that grid value 0 reaches. On the real set the lists follow
each other from word 1 in value order with no gap, each non-zero cell has its own value, and every polygon is in a
list (`decode_ground` takes the polygon count as the highest index named, plus one).

| | Beginner (0) | Expert (1) | Advanced (2) |
|---|---|---|---|
| slot; list block, pool | 0; `0x30`, `0x19c90` | 1; `0x8000`, `0x1e690` | 2; `0x10000`, `0x3ab00` |
| non-zero cells = lists | 626 | 2,315 | 1,288 |
| list block words | 4,714 | 22,074 | 10,290 |
| longest list | 23 | 33 | 32 |
| polygons | 1,184 | 7,239 | 2,986 |
| of which triangles (a repeated corner) | 16 | 274 | 42 |
| attribute block bytes (with `0xcd` padding) | 1,184 | 7,240 (1) | 2,988 (2) |
| road polygons (`steps`, kind 1, first) | 503 | 1,064 | 818 |
| height range (y) | 0-19.5 | 0-60.6 | 0-138.9 |

Example: Beginner's block starts `0, 1, 0x32d, 3, 0x326, 0x32d, 0x330, 3, ...` (word 0, the list at grid value 1,
the list at grid value 3).

## 5. Polygons

16 words per polygon (`POLY_WORDS`), at `pool + 16 * index`:

| Words | Field | Content |
|---|---|---|
| 0-11 | `corners` | four (x, y, z) f32, world units; the only words function `0x36` reads (`rep #0xc` at `0x48e`) |
| 12-14 | `centre` | (x, y, z) f32: the corners' mean within 0.0005 on 98.5 % (up to 8.4 off on the rest) |
| 15 | `direction` | sign-extended s16, 1/65536 turn (`0x4000`, `0x3fff`, `0x3555` common; 2 / 6 / 523 are 0) |

Words 12-15 are read by index through routine `0x7be` (all 16 words into `$0x56`-`$0x65`) for functions `0x18`,
`0x2a`, `0x2b` (word 15 to the sin/cos unit), `0x2c` (returns the centre), `0x30` (returns word 15) and `0x33`.
What the game does with them is not traced here.

Every original polygon is convex (a repeated corner makes a triangle) with **negative signed XZ area**
(`sum x_k z_k+1 - x_k+1 z_k < 0`), all corners inside -1024 <= x, z < 1024, smallest area 0.28 / 0.36 / 2.64.
Example: Beginner polygon 0 has corners (-91.009, 0, -82.042), (-101.509, 0, -100.229), (-108.437, 0, -96.229),
(-97.937, 0, -78.042), centre (-99.723, 0, -89.135), direction `0xffffaaab` (-120 degrees).

## 6. The attribute block

`*block_ref_b`: one byte per polygon, its low nibble the surface kind (`docs/formats/model2-course-tables.md`: kind 1
is the road and exactly the first `steps` polygons). The i960 reads it with the TGP's polygon index
(`ldob (r3)[g4]` at `0x11ca0`). It is padded to 4 bytes with `0xcd`, which the encoder writes too.

## 7. Function `0x36` (reproduced by `tgp.query`)

FIFO after the call (`0x880360`, data `0x3636`): `dest` (`0x17ffc`), `y` f32 (the car's height), the grid value of
the cell, `x`, `z` f32.

1. **Hit test**, each listed polygon in list order: for k = 0..3 (`P_4 = P_0`) the TGP computes in f32, in this
   order, `d = z X_k - x Z_k - z X_k+1 + x Z_k+1 - Z_k+1 X_k + X_k+1 Z_k` and rejects the polygon unless `d >= 0`
   (`0x499`-`0x4a5`). That is the inner side of every edge for the originals' winding, edges included; a polygon of
   the other winding is never hit.
2. **Height** of a hit (`0x684`, plane `0x72b`). With u, v = corner 1 - corner 2, p, q = corner 0 - corner 2 and
   s, t = (x, z) - corner 2 (X and Z differences), the point is alpha (p, q) + beta (u, v). The routine branches on
   `Z0 == Z2` (q == 0, tested as q + 0.0 == 0): for q != 0, beta = (q s - t p) / (q u - v p); for q == 0, beta =
   t / v. Then alpha = (s - u beta) / p, or for p = `X0 - X2` == 0, alpha = (t u - s v) / (q u). The bits of alpha
   and beta are OR-ed and `0.0` added (`orad`, `fadd`, `brif ged`): a result that is not negative-and-non-zero
   (the TGP's sign flag ignores a -0.0) selects the plane of corners 0-1-2, else the plane of 0, 2, 3 (corner 1's
   side of the 0-2 diagonal, else the other). The plane: n = (B - A) x (C - A), r = (((nx x + nz z) - nx Xa) -
   ny Ya) - nz Za in that order, height = -1 * (r * (1 / ny)) (the TGP's `$5` is -1.0). Every `/` is a numerator
   times the 1/x unit's answer R(a) = T1 - m |T0| (MAME `copro_inv_r`; m is a with exponent 0x7f, T0 and T1 two words
   of the 1/x table in the tables ROMs `opr-14742a.45` / `opr-14743a.46`, words `0x8000`+): `tgp.inv_table(rs)`
   builds it, `query(..., inv)` uses it; without it the model divides in plain f32. **ny == 0** answers the constant
   at TGP word `0x206`, which init `0x38`-`0x3c` loads as exponent 0x7b, mantissa 0xcccccd: -0.1 (f32
   `0xbdcccccd`), the same "none" value the i960 uses. It is reachable on originals: a triangle (corner 3 equal to
   corner 2; 16 / 274 / 42 of them) has plane 0-2-3 with no XZ extent, and a point on its 0-2 edge whose f32 alpha
   or beta comes out negative gets -0.1 where the polygon's height is about 82. It matches every probed call within
   0.00014. On flat ground at y 0 the height is -0.0 (the -1 factor times +0), and the probe logs -0.0000, not the
   word 0 (no call read none): a model must keep the sign of a zero. On the originals 7 (Expert) and 4
   (Advanced) polygons have `Z0 == Z2` and none of Beginner's; no probed call chose one, so that branch is verified
   on synthetic polygons only (`test_query_sloped_plane_both_triangles_and_both_branches`).
3. **Choice** (`0x4e9`): the hit with the smallest `| |height| - y |`; a later hit replaces an equal one.
   Because of `|height|`, below y = 0 this favours the polygon nearest 0, not the one nearest the car: where ground
   below 0 lies under other ground, the TGP answers the surface overhead (Phase 10, M5a: Mermaid Lake's tunnel,
   segments 316-345 down to y -36 under segment 81 at 0.1, answered the road above on all 82 lane points of segments
   322-335, and the car met an invisible wall). Overlapping ground must lie above 0; the converter raises such a
   course whole (`build.LIFT`; `docs/spike/2026-11-courses-4-7.md`, "Mermaid Lake"). Ground below 0 with nothing over
   it (Desert City -55.6, Rin Rin Rink -35.5) answers right. A course past the grid in x moves whole in x
   (`build.SHIFT`, Rin Rin Rink -64: four squares, so every polygon keeps its place on the grid, the lists the same
   four columns over; `docs/spike/2026-11-courses-4-7.md`, "Rin Rin Rink").
4. **Result** in buffer RAM: `dest & 0xffff` + `0x400000` is word `D` (i960 `0x91fff0`), bits 16-22 a signed stride
   `s` (1). `D + s` gets the hit count, then the chosen height over it; `D + 2s` the chosen polygon index; `D` is set
   to 0 last (done). No hit leaves the count 0 in the height word: the i960's "none" (it then uses -0.1).

### The query model

`tgp.cell(x, z)` is the i960's `0x11bec`: per axis round to nearest even (`cvtri`), + 1024, `>> 4`, `& 0x7f`; the
row (times 128) is z and the column x. A point outside -1024 <= x, z < 1024 wraps, it is not clamped or refused, so a
converted course's car that leaves the map reads another cell's list (usually none). `tgp.query(g, x, y, z, inv)`
returns `(height, polygon index)` or `None` (hit count 0); it evaluates the hit test, height and choice above in
numpy float32, in the order of the listing (the MB86233 semantics read from MAME's core: `fsbd` d - a, `fmrd`
d - p, `fspd`, `fsmd`, the sign flag, the 1/x unit), with the list of grid value `g.grid[cell(x, z)]` (0: none).
`tgp.surface(g, poly)` is `attributes[poly] & 15`. All raise `TgpError` for a non-finite or non-number argument or a
polygon index outside the attribute bytes, and for a malformed `Ground` where they read it: `g` not a `Ground`; for
`query` a grid that is not 16,384 values, the cell's grid value not a u16 or without a list, a list entry that is
not an index into the polygons, a listed polygon that is not a `GroundPoly` or whose corners are not four finite f32
triples; for `surface` attributes that are not bytes. Only what the call reads is checked, not the whole layout.

Measured against the probe (`DAYTONA_GROUND_LOG`, 45 s autoplay per course, calls from frame 1330) with the 1/x
table: `cell` gives the grid value the i960 sent on every call; Beginner (9,487 calls) and Advanced (11,535) agree on
every call (polygon, height within 0.01), Expert on 99.03 % (109 of 11,232 differ). Each of the 109 is reproduced
by some x, z within the log's +-5e-5 rounding (a point within about 0.003 of an edge two polygons share, or two
stacked polygons whose heights differ by 1-2 f32 ulps, where the log's four decimals decide the choice), so each is
consistent with the model under the log's rounding; `test_query_reproduces_probe` asserts that for every miss and at least 99 % per course. Without the table (plain
f32 division) Expert is 99.00 % (112) and one miss (frame 2293) cannot be reproduced from any rounding of the log:
the table is what closes it. Player calls on Beginner while racing are on surface 1 (road) or 2, 5, 8, 10, 13
(`test_road_surface`).

## 8. Decoder and encoder

`decode_ground(rs, course) -> Ground` reaches the data as the game does (course record, then slot `course`).
`Ground` fields: `course`, `grid`, `attributes` (one byte per polygon, padding not included), `polys`
(`GroundPoly(corners, raw, centre, direction)`), `lists` (grid value -> indices), `base` (the slot), `raw` (the list
block as read), `grid_at`, `attributes_at` (data offsets), `list_base`, `pool_base` (words). `TgpError` (a
`DecodeError`) names the cell and its grid value when a list starts or runs past the course's list block, or names a
polygon past its pool (both bounded by the next higher slot-table address, else the half's end: another course's
bytes), and when a list or a polygon crosses a window. The polygon count is the highest index the lists name,
plus one; so that no polygon is dropped unseen, every record (16 words) from that count up to the pool's bound (the
next higher slot-table address, else the half's end) must be free (every word `0xffffffff`), else `TgpError` naming
the first record that is not. On the originals the three pools end exactly at the next address (no such record); a
pool whose following block moved away and was erased reaches to the next address over free records, and the last
pool is followed by the free run. Words after the bound's last whole record hold no polygon and are not checked.

`encode_ground(g)` gives `{(region, byte offset): bytes}` for six blocks: the grid and the attribute block (data
ROM), the slot's two words, the list block and the pool (copro_data). `ground_spans(rs, course)` names the same
blocks with their lengths; no block is shared between courses. `verify_lossless_ground(rs, course)` is `[]` on the
three courses. The encoder refuses, with `TgpError`, what the TGP cannot answer for or a field cannot hold: corners
that are not four finite f32 triples, a corner outside -1024 <= x, z < 1024, zero or positive XZ area, a polygon
that is not convex (the edge test would miss the notch), a centre that is not finite or a direction outside s16; a
grid value past u16 or without a list (naming the cell), a list key past u16 or that no grid cell reaches, an
index past the polygon count, a polygon that no list names,
overlapping lists, a non-empty off-road list, a list or polygon crossing a 64K-word window, blocks past their
region (copro_data's first half, data ROM) or overlapping each other or the header (words 0-`0x2f`), a slot past
15, attributes that are not one byte per polygon.

## 9. What a new course must supply

1. **Polygons.** Convex (a repeated corner makes a triangle), each with the originals' winding (negative signed XZ
   area; the TGP's edge test is one-sided), four finite corners inside -1024 <= x, z < 1024, at the road's height.
   Words 12-15 per polygon: the centre (the corners' mean) and the direction word (s16, 1/65536 turn). The road
   polygons come first, in step order (the step table and the attribute kinds follow the index).
   **A Dreamcast cell with a notch** (one corner turned the wrong way; courses 3-7 have 2 / 11 / 19 / 10 / 13, all
   with exactly one such corner, one road cell among them: n 5's 130) is repaired by `convert.ground`, one polygon per
   cell (so no index, list or step moves): a corner within `NUDGE_ULPS` 4 f32 ulps of its neighbours' line is
   nudged outward by the fewest ulps (`"ulps"`); a wider notch (one cell a course on 4-7: n 4 cell 2082, n 5 410,
   n 6 2256, n 7 2334; 0.00012 to 0.049 units) has its corner moved onto that line (`"line"`, at most `MAX_MOVE`
   0.0625 units) at the height of the plane the TGP uses there. The TGP's height splits along the 0-2 diagonal
   (section 7), so a notch at corner 1 or 3 puts the whole cell in the plane of the other three corners and the
   moved corner on that plane changes no height of the cell; a notch at corner 0 or 2 takes the mean of the two
   triangles' planes (n 5's cell is flat). Refused when half the planes' disagreement plus the steepest slope times
   the move passes `HEIGHT_TOLERANCE` 0.005 units. `ground.REPAIRED` lists every repaired cell;
   `test_repaired_ground_heights` measures the answer against the TGP's rule on the unrepaired cells (worst 6e-5
   units). The Dreamcast's own height (`1ST_READ.BIN` `0x8c034ee8`) blends a non-planar cell as a ruled surface,
   from edge 0-1 to edge 3-2, which no diagonal split gives. So any converted course differs from it by its cells'
   non-planarity: up to 0.24 units on Desert City's road. The repair adds at most the tolerance to that
   (`test_repaired_ground_against_the_dreamcast`).
   Splitting the cell along the notch diagonal was the other way and was not chosen: for a notch at 1 or 3 that
   diagonal is not the TGP's, so the split moves heights by the corner's non-planarity (0.011 on n 6's cell); and
   any split adds polygons and triangles (whose degenerate plane answers -0.1 at the edge, section 7).
2. **Lists.** One list per distinct grid value: `count:u32` then the indices of every polygon that overlaps the
   cell, laid out in a list block at the cell's grid value (a word offset). Every polygon is in at least one list
   (the decoder takes the polygon count as the highest index named, plus one). Word 0 of the block is a count of 0:
   the list of grid value 0 (off the ground).
3. **Grid and attribute block**, in data ROM, at the course record's `*block_ref_a` (128 x 128 `u16`, 32 KB) and
   `*block_ref_b` (one byte per polygon, the surface kind in the low nibble, road = 1 first, padded to 4 with
   `0xcd`).
4. **Selector entry.** Slot `n` (= the course id) of the slot table: words `0x10 + n` (list block) and `0x20 + n`
   (pool) in the header, `n` <= 15. Every list and every polygon inside one 64K-word window (a block may span
   windows; the program re-banks per list and per polygon). Every record between the pool's last polygon and the
   next slot-table address above it (or the half's end) is free (16 x `0xffffffff`); the game sends the course id
   to function `0x0d`, so course `n`'s ground is found only in slot `n` (`golden.check_course` refuses another).
5. **Where it fits.** Only free bytes (`0xff`), written through `tgp.write_free`: `tgp.COPRO_FREE` = byte `0x128940`,
   882,368 bytes in copro_data (word `0x4a250` to the end of each identical 2 MB half; the encoder writes the first
   half and `RomSet.with_patch` the second), `tgp.DATA_FREE` = byte `0x65d120`, 667,360 bytes in the data ROM
   (to `0x700000`). `tgp.relocate_ground(rs, course, copro_at, data_at, erase_original=False)` does this for an
   existing ground (every pointer rewritten, its output decoded and compared; Beginner relocated, and relocated with
   the original erased, drives identically in MAME: `docs/spike/2026-11-tgp-ground.md`, "Relocation"). Its default
   `copro_at` is the first word from `COPRO_FREE`'s start at which no list and no polygon crosses a window: the
   start for Beginner, 6 / 14 words on for Expert / Advanced (word `0x4a256` / `0x4a25e`: their pools then start on
   a 16-word boundary; from `0x4a250` a polygon would cross `0x50000`). Expert and Advanced move only with
   `erase_original=True` (offline tests; not run in MAME): once their slot words change, the pool before the old one
   (Beginner's before Expert's, Expert's before Advanced's) is bounded by the next address up and would reach over
   the old polygons, neither listed nor free; erased, they are free records. Without it `relocate_ground` refuses
   with a `TgpError` that names the neighbour and says `erase_original` frees the old blocks. Beginner moves either
   way (no pool ends at its old blocks). `relocate_ground` also rewrites the record's words 6-7 (where they hold the
   grid pointer) and slots 4-15 that repeat the course's addresses; `golden.tables`, for a ground moved by editing
   `Course.ground_data`, writes only slot `base`'s two words and keeps words 6-7 as `entry.raw` holds them. No
   reader of either is known (Unknowns), so the difference is not expected to matter. A new course needs the same
   steps with its own `Ground`, then `encode_ground`; `write_free` refuses what is not free.

### Budget

Bytes per course at arcade density (Task 6's `coverage`; data = the 32,768-byte grid plus the attribute block;
copro_data = 8 bytes of slot words plus the list block plus the pool):

| | Beginner | Expert | Advanced |
|---|---|---|---|
| polygons | 1,184 | 7,239 | 2,986 |
| list block | 18,856 | 88,296 | 41,160 |
| pool (64 bytes per polygon) | 75,776 | 463,296 | 191,104 |
| **copro_data** | **94,640** | **551,600** | **232,272** |
| **data ROM** | **33,952** | **40,008** | **35,756** |

Against `COPRO_FREE` (882,368) and `DATA_FREE` (667,360), before the other tables a course needs (road line, scenery
and the rest also draw on `DATA_FREE`; polygons and textures have their own free spans):

| Density of the five new courses | copro_data for five | Fit in `COPRO_FREE` | data ROM for five (grid + attributes) |
|---|---|---|---|
| Beginner's (about 1,200 polygons) | 473,200 (54 % of free) | all five (up to 9) | 169,760 (25 % of free) |
| Advanced's (about 3,000) | 1,161,360 | three (up to 3; 696,816) | 178,780 |
| Expert's (about 7,200) | 2,758,000 | one (551,600; two need 1,103,200) | 200,040 |

The copro_data span is the binding one: the data ROM holds five courses of any density many times over (19 at
Beginner's, 16 at Expert's). Polygon count drives the cost (80 bytes per polygon in the pool and its lists on the
originals: 79.9 / 76.2 / 77.8).

**A custom MAME build** (spec decision 9) would change the copro_data side only: the board populates 4 MB of the
8 MB region (`mpr-16536/16537`), MAME masks the word index to 8 MB, so the unpopulated 4 MB would add up to
4,194,304 bytes (derived, not run: the bank bits that select it are not traced here), about 7 courses at Expert's
density on top of the free span (5,076,672 bytes in all, 9 at Expert's). The data ROM would not need it.

### Unknowns

- What the game does with polygon words 12-15 (the centre and the direction word): read by functions `0x18`, `0x2a`,
  `0x2b`, `0x2c`, `0x30`, `0x33`, not traced. A new course supplies the originals' relations (centre = corner mean;
  direction = the road's travel angle). Phase 4 showed the direction word is read on Beginner: the standing cars
  behind the lap line take their yaw from it (section 10); which function reads it is still not traced.
- Slot 3 is a fourth ground (1,936 lists, 971 polygons) that no course record selects; its owner is unknown.
- The course record's words 6-7 hold the grid pointer and slots 4-15 repeat slot 0's addresses; no reader found.
  `relocate_ground` rewrites them anyway; `golden.tables` does not (section 9, item 5).
- The `0x684` branch on Z0 == Z2: modelled from the listing, covered by synthetic polygons only (Expert has 7 such
  real polygons, Advanced 4; none was chosen in a probed call).

## 10. The mirror (M1c)

`dtools/arcade/mirror.py`, `mirror_ground` and `mirror_ground_poly` (the course's other tables:
`docs/formats/model2-course-tables.md`, "What the mirror (M1c) must transform"). The mirror is taken in the plane
z = -0.5 (`mirror.PLANE`): a world z becomes `-1 - z` (`mirror.mirror_world`, rounded to f32). `cell` rounds z before
it takes the row, and `round(-1 - z) = -1 - round(z)`, so a mirrored point lands in exactly row `127 - r`, row edges
included (about z = 0 a z that rounds onto a row edge lands one row lower, which Task 4 first answered by rebuilding
the edge rows' lists). The one exception is an f32 z of exactly k + 0.5 on a row edge: both sides round to even. On
the ground, as implemented and as the MAME runs found it (`docs/spike/2026-11-mirror.md`, "Offline", "Rules
checked", "The mirror plane"):

| Part | Rule | State |
|---|---|---|
| corners, centre | z -> `-1 - z` (words 2, 5, 8, 11, 14); x, y kept | checked in MAME (Phase 4): every call of the mirrored run answered, heights equal to the original's wherever the calls pair |
| corner order, quads | `(0, 3, 2, 1)`: negating z makes the signed XZ area positive and the edge test is one-sided (section 7), so the winding is reversed; this order keeps corners 0 and 2, the diagonal the height splits along | checked in MAME (Phase 4) |
| corner order, triangles (corner 3 = corner 2) | `(1, 0, 3, 2)`: the quads' order would repeat corners 1 and 2 and leave the alpha / beta solve (corners 0, 1, 2) without extent; this keeps the repeat at 2-3 and corner 2 as the apex | checked in MAME (Phase 4) |
| direction word (word 15) | `0x8000 - d`, kept in s16 | checked in MAME (Phase 4): read on Beginner, the standing cars' yaw (with `d` unchanged they face the original's way, `0xaaac` instead of `0xd554` at frame 1330, and the runs part at 1369); with `0x8000 - d` they face the mirror's within one unit |
| polygon order, attribute bytes, list indices | unchanged (the road first, in step order) | checked in MAME (Phase 4) |
| grid rows | `r -> 127 - r` (the column kept), each cell with its list; the lists, the list block and every block's place and size kept (nothing is rebuilt, so a mirror fits wherever the original does, Expert included) | checked in MAME (Phase 4): no call missed the ground; offline every query point far from a polygon edge answers as the original's (`test_mirror_ground_queries`) |

A point exactly on an edge two polygons share may answer the neighbour polygon (the reversed winding changes the
f32 rounding), at the same height; a point on a non-planar quad's 2-3 edge may take the other triangle's plane.
Offline, 129 of Beginner's 1,176 road lane points (road polygon corners) and 15 integer edge points answer
differently, every one within 0.01 of a polygon edge and a hit on both sides. Mirrored twice, 74 of the 4,736
corner z values come back one f32 ulp off (where `-1 - z` rounds); grid and lists come back exactly.
