# Model 2 (original) textures

Derived from MAME 0.289 `src/mame/sega/model2_v.cpp` (`model2_3d_process_polygon` lines 381-423,
`model2_3d_render` lines 567-601), `model2rd.ipp` (`get_texel` lines 17-40, `fetch_bilinear_texel`, `draw_scanline_tex`
lines 244-353) and `model2.cpp` (memory map lines 1029-1066 and 1258-1278, `tex0_w` / `tex1_w`, `video_start`),
BSD-3-Clause, copyright the MAME team; and from Daytona USA's own upload code, traced and read in
`docs/spike/2026-11-arcade-course-tables.md`, "Textures". Implemented in `dtools/arcade/texture.py`.

## Where things live

| What | Where | Filled by |
|---|---|---|
| Texture points (per-vertex u, v) and texture headers | **textures region** (`RomSet.textures`, MAME `"textures"`), 16-bit LE words; `tpa` / `tha` are word indices | ROM, read directly by the rasterizer |
| Points / headers with `tpa` / `tha` bit 23 set | rasterizer texture RAM (`raster->texture_ram`, 64 K words, not on the i960 bus) | geometry command `0x04` at run time; not decoded here |
| Texels | texture RAM 0 / 1, i960 `0x12000000` / `0x12400000` (the game writes the mirrors `0x12200000` / `0x12600000`), 1 MB each | the i960 at boot and at the course load, **from data ROM** (below) |
| Luma tables | luma RAM, i960 `0x12800000-0x1281ffff` (`umask32 0xff`: 0x8000 bytes) | the i960 at boot (frame 28), from data ROM `0x802fc8` |
| Polygon palette | palette RAM `0x01800000`, entries `0x1000-0x13ee` (bytes at `0x01802000`) | the i960 at boot (frame 20), from data ROM `0x84f4ec-0x84fcc9` |
| Colour translation | `0x01810000-0x0181bfff`: r, g, b tables of 32 x 256 u16 | computed by program `0x0a74-0x0b40` at boot (frame 20) |
| Gamma | MAME only (`video_start`): `max((i - 64) * 255 / 191, 0)` | monitor emulation, not game data |

So the textures region does **not** hold texels: on this board the texel sheets are RAM, uploaded from data ROM.
Everything the colour path needs is in ROM (copied) or in program code (computed); `colour_tables(rs)` rebuilds
the three RAM tables and matches MAME's RAM at frame 2400 of a Beginner race byte for byte, except four palette
entries (`0x10d3`, `0x11ae`, `0x11b5`, `0x11b6`) the game rewrites at run time (writers at `0xecf0`, `0x2256b4`,
`0x225c00`; not decoded).

## Texture header (4 words at `tha`)

`tha & 0x800000` selects texture RAM, else the ROM word `tha & 0x7fffff` (`texture_rom_mask` = region bytes / 2 - 1).
After each polygon the rasterizer adds `step * 4` to the header address, `step` being **attr bits 12-16**, signed
(line 416). Every record moves it, a culled link-0 record included (the update happens before `check_culling`).
`object_headers(rs, tha, obj)` walks an object's records this way.

| Word | Bits | `TexHeader` field | Meaning (MAME line) |
|---|---|---|---|
| h0 | 0-2 | `width` | `32 << n` (592) |
| | 3-5 | `height` | `32 << n` (593) |
| | 6, 7 | `wrap_x`, `wrap_y` | smooth wrap; ignored when the axis is mirrored (587-588) |
| | 8, 9 | `mirror_x`, `mirror_y` | mirrored repeat (583-584) |
| | 10-11 | `utex_min_lod` | microtexture blend shift (599) |
| | 12 | `utex` | microtexture enable (598) |
| | 13 | `translucent` | renderer bit 0: texel 15 is transparent (568; model2rd.ipp 187-196) |
| | 14 | `textured` | renderer bit 1: textured, else flat colour (568) |
| | 15 | `checker` | draw every other pixel (575) |
| h1 | 0-7 | `luma_base` | luma table, `lumabase = n << 7` (576) |
| | 8-15 | `raw[0]` | not read by MAME; 0 on every Beginner / Expert / Advanced header |
| h2 | 0-5 | `tex_x` | `32 * n`, logical sheet x (594) |
| | 6-10 | `tex_y` | `32 * n`, logical sheet y (595) |
| | 11 | `raw[1]` | not read by MAME. **Observed** set on every textured header of the three courses and clear on every untextured one; its meaning is not known |
| | 12 | `sheet` | level 0 in texture RAM 1 (else 0); odd mip levels in the other sheet (590-591) |
| | 13 | `utex_x` | `128 * n` (600) |
| | 14-15 | `utex_y` | `128 * n` (601) |
| h3 | 0-5 | `raw[2]` | not read by MAME; 0 on every course header |
| | 6-15 | `colour` | palette entry `0x1000 + n` (577; model2rd.ipp 270) |

Example (Beginner, `tha 0x1da0`): `44c9 0000 0bac 8a40` = 64 x 64, wrap x and y, `utex_min_lod` 1, textured, luma table
0, sheet 0 at (1408, 448), bit 11 set, colour `0x229`.

`encode_tex_header(h)` writes the four words back (`tha` and `step` are not part of them); it refuses a size that is
not `32 << 0..7`, a position that is not a multiple of 32 inside its field, a field past its width (`TextureError`).

**Format.** MAME's Model 2 renderer has one texel format, **4 bits per texel** (`get_texel` returns a nibble); there
is no 8 bpp path. "Format" in a header is only the renderer choice: flat (`textured` clear), textured, and either
one translucent. MAME draws nothing for a flat translucent polygon (`draw_scanline_solid<true>` returns).

## Texture points (`tpa`)

Decoded by `texture_points(rs, tpa, obj)` and written by `encode_texture_points(points, obj=None)`; covered by the
golden test since Phase 7 (`golden.Course.points`, table "texture points").

**Layout** (model2_v.cpp lines 383-402, `model2_3d_process_polygon<NumVerts>`): per polygon `2 * vertices` u16 words,
in vertex order (the rasterizer's corners, `polyrom.polygons`: `P1(n-1), P0(n-1), P0(n)[, P1(n)]`), each vertex
**`v` then `u`**, unsigned, in 1/8 texel (the renderer scales by 1/8 and works in 1/256: `u = pu / 8` texels);
`tpa` advances by `2 * vertices` words per polygon. A quad (attr bit 0) reads 4 vertices, a triangle 3. `tpa & 0x800000`
selects texture RAM (not in ROM; refused), else the ROM word `tpa & 0x7fffff`.

**Every record reads them**, before culling: a link-0 (move) record and an untextured record too. On ids 0-2 each
placement's points re-emit byte for byte this way (539 / 430 / 390 placements, the HUD outline's included), and
Beginner's cell placements' `tpa` are one object's all-record points apart. What the originals store where nothing is
sampled: Beginner's move records (all quads) hold zeros (4,712 of 4,713) and so do its untextured drawn records
(4,521 of 4,566). The converter writes zeros there.

`decode_course` keys the points by `tpa & 0x7fffff`; where two placements share a `tpa` it keeps the reading of the
most words (it covers the other's). Runs of different `tpa` may overlap: the car models' placements read overlapping
runs (words `0x16a276`-`0x16a278`, `0x5a9032`-`0x5a904e`), which Beginner's and Advanced's HUD outlines also reach;
their union is 451,068 / 689,804 / 440,468 bytes on ids 0-2.

Example (Beginner, `tpa 0x550a8`): `0000 0000 0200 0000 0200 0200 0000 0200` = the corners of a 64 x 64 texture.

**Outside the texture.** The rasterizer repeats every texture (model2rd.ipp `fetch_bilinear_texel`: `u0 = (u >> 8) &
(width - 1)`): a point past the texture never reads a neighbouring texture's texels. The header's wrap bit only chooses
what the bilinear filter blends at the texture's edge: with it, the next repeat (smooth); without it, the edge texel
(clamp). A mirror bit flips every other repeat (`u & (width << 8)`) and turns the wrap bit off. The originals leave wrap
and mirror clear on many polygons whose points pass the texture: over the placements golden decodes (car models
included), 7,054 of Beginner's 19,426 textured drawn records pass their texture on an axis whose wrap and mirror bits
are both clear (Expert 8,495 of 32,652, Advanced 6,422 of 22,677). The converter sets the wrap bit wherever a polygon's
points leave its texture (docs/spike/2026-11-convert.md, "Texture points (M3b)").

**The range.** A point is a u16 in 1/8 texel: at most 8,191.875 texels from the polygon's least point once that is
moved into the first repeat (`textures.polygon_points`), so 255 repeats of a 32-texel texture, 127 of a 64-texel one.
A Dreamcast polygon may repeat its texture more (Rin Rin Rink's stripes: 300 times on 32 texels, 140 on 64). The
converter cuts such a polygon (`model.convert_model`'s `points`, Phase 11 Task 3) along lines of constant u or v at
every multiple of `model.band(size)` repeats (the largest even count that fits: 254 for 32 texels, 126 for 64, 30 for
256), each piece then within one band; the cut points are on the original edges with the cut's own coordinate, so the
picture is the same (docs/spike/2026-11-courses-4-7.md, "Rin Rin Rink").

## Texel sheets

A sheet is a **2048 x 1024** logical texel grid stored as 1024 x 2048: logical `(x, y)` is stored at `(x, y)` for
`x < 1024` and at `(x - 1024, y ^ 1024)` otherwise. Stored `(x, y)` is sheet word `(y // 2) * 512 + x // 2`; each
16-bit word holds a 2 x 2 block (`get_texel`; parity of the texture-local x, y, which is that of the sheet's for
level 0 since `tex_x`, `tex_y` are multiples of 32):

| Bits of the word | Texel |
|---|---|
| 12-15 | even x, even y |
| 8-11 | odd x, even y |
| 4-7 | even x, odd y |
| 0-3 | odd x, odd y |

So in little-endian bytes, the high byte is the even row and the low byte the odd row, the high nibble the even column.

**Mip levels** (`fetch_bilinear_texel`): level `L` of a texture at `(x, y)` is at
`(((x - 2048) >> L) & 2047, ((y - 1024) >> L) & 1023)` in sheet `sheet ^ (L & 1)`: level 1 at `(x / 2 + 1024,
y / 2 + 512)` in the other sheet, level 2 at `(x / 4 + 1536, y / 4 + 768)` in the same sheet, and so on. All of them
fall in the **mip area**, logical x >= 1024 and y >= 512 (stored words `0x60000-0x7ffff`). Level-0 textures use the
rest; `texel_spans` refuses a texture that reaches into the mip area. No course header does. A level whose base
is odd (only from level 5, i.e. textures of 128+ on both sides) still works: `get_texel` picks the word from the
sheet position and the nibble from the texture-local parity, and `mip_texels` does the same.

### The game's upload

Routine `0x146c` (program ROM), entered from `0x1454` at boot (sheet 1) and from `0x1438` at boot and at every course
load (sheet 0), copies **one 1 MB block of data ROM** into the sheets, one 16-bit word per 32-bit bus write (`tex0_w`
stores the low 16 bits of write `k` into sheet word `k`):

1. words 0-`0x5ffff` of the block -> words 0-`0x5ffff` of its own sheet, unchanged (level 0);
2. the remaining `0x20000` words -> the mip area (words `0x60000-0x7ffff`), in 256 rows of 512 words: 128 rows, then
   64, 32, 16, 8, 4, 2, 1, 1 (one batch per level `r9 = 0..8`). The first batch goes entirely to the **other** sheet;
   from the second on, each row is cut into runs from the table at program `0x1578` (`0x200, 0x100, 0x80, ... 1`;
   the first run is `0x100`, run `k` is `table[k]`, doubled once `k` is past the level), alternating sheets run by
   run, starting with the other sheet.

| Sheet | Source block (data ROM offset) | Chosen by |
|---|---|---|
| 0 (course) | Beginner `0x200000`, Expert `0x400000`, Advanced `0x300000` | `u32[4]` at program `0x15a0` (`0x02200000, 0x02400000, 0x02300000, 0x02200000`), indexed by the course id byte `0x00501460` |
| 1 (common) | `0x500000` | `lda 0x2500000,g5` at `0x145c` (displacement word at `0x1460`) |

Level-0 texels of a header therefore come straight from data ROM: `block(sheet) + 2 * word`. `upload_sheets(rs,
course)` ports the routine (common block first, then the course block, the boot order) and reproduces MAME's texture
RAM 0 and 1 **byte for byte** at frames 1330 and 2400 of a Beginner run. The two loops are `0x149c` / `0x14c0`
(level 0; `st` to sheet 1, `stos` to sheet 0) and `0x14f0-0x1570` (mip area).

## Colour

`draw_scanline_tex` per pixel, `t` the bilinear-filtered texel times 16 (8 bits; for an unfiltered texel
`t = index << 4`):

    luma  = min(luma_ram[luma_base * 128 + (t >> 1)] * object.luma // 256, 63)
    c     = palette[0x1000 + colour] & 0x7fff                 # x bbbbb ggggg rrrrr
    r     = gamma[xlat[0x0000 + (c & 31) * 256 + luma] & 0xff]
    g     = gamma[xlat[0x2000 + (c >> 5 & 31) * 256 + luma] & 0xff]
    b     = gamma[xlat[0x4000 + (c >> 10 & 31) * 256 + luma] & 0xff]

`object.luma` is the polygon's lighting, `(command_buffer[9] >> 15) & 0xff` from the geometry engine (diffuse /
ambient / specular from the texture parameters, attr bits 18-22): run-time state, so `decode_texels` takes it as
`light` (default 255). A texture is thus a 4-bit **intensity** map: the luma table turns it into a 6-bit shade, the
header's single palette entry gives the hue. In a translucent texture index 15 is transparent (`decode_texels` alpha 0).

- **Luma RAM**: `u32` table count at data `0x802fc4` (66), then 66 x 128 bytes at `0x802fc8`, copied byte by byte by
  program `0x1400` to the start of luma RAM; the rest of luma RAM stays 0. Values 0-63. Course headers use tables 0
  and 1 only.
- **Palette**: the 1,007 words at data `0x84f4ec-0x84fcc9` (right after the object library) -> entries
  `0x1000-0x13ee` (program `0x5c7c`). Course headers use colours up to `0x3ee`.
- **Colour translation**: program `0x0a74-0x0b40`, per table (r, g, b identical) and 5-bit component `c`: 64 shade
  entries `m = 0, k, 2k, ...`: `(c * m + base) >> 6` (0 when `c * m = 0`; `0xffff` from 256), then 192 copies of
  `(k * c + bias) >> 1` (the entry MAME's tile-layer palette reads, `0x80 / 2` into the row). With the half-word at
  work RAM `0x005fe5e2` non-zero: `k` 10, base `0x1920`, bias `0xc9`; zero: 12, `0x1160`, `0x8b`. The MAME run had the
  non-zero variant (the generated table equals its RAM); what sets `0x5fe5e2` was not traced (`colour_xlat(flag)`).

## Encoding

`encode_texels(h, indices)` packs a `(height, width)` array of 0-15 into the bytes of `texel_spans(rs, h, course)`
(one run per two texel rows, split at logical x = 1024). `texel_spans` first runs `encode_tex_header(h)`, so a size
that is not `32 << n` or a position that is not 32-aligned is refused rather than written misaligned.

`with_texels(rs, h, indices, course, mips=...)` writes them into data ROM. `mips` is required (no default), because
the game copies its mip levels from ROM and never computes them:

- `"keep"`: the mip area is left alone, so at a distance the texture still shows the **old** texels;
- `"decimate"`: every level of `mip_levels(h)` (1 .. log2(min(width, height)) - 1, down to 2 x 2, MAME's
  `max_level`) is rewritten from the new level 0 by **nearest sampling**: level k takes every 2^k-th texel in x and
  y (4-bit indices cannot be averaged). **Sega's own filter is not known; "decimate" is our choice.** (Inferred from
  the ROM, not from code: the stored level-1 texels are within 1 of the mean of their 2 x 2 level-0 block for 78% of
  the Beginner texels, so the originals look box-filtered on the index values.)
- anything else: `TextureError`.

`mip_texels(rs, h, level, course)` gives each mip texel's data-ROM byte and nibble; `mip_spans` the merged byte runs.
They invert the upload: mip word `w` of a sheet is word `w` of one of the two blocks (the source and destination
pointers advance together), the block chosen by the run pattern (`_mip_other`). A mip texel whose word would come
from the other sheet's block is refused; none of the Expert, Advanced or Beginner textures has one (each texture's
mips sit in its own block). A synthetic test runs the game's upload (`upload_sheets`) on the written ROM and reads
every level back through a port of `get_texel`.

For a caller that holds the sheet blocks' offsets rather than a RomSet (a `golden.Course`): `texel_spans_at(h,
base)` and `mip_texels_at(h, level, base)`, with `base` the block's data-ROM offset (`sheet_source`); `mip_indices`
reads a mip level as stored and `encode_mip_bytes(h, level, indices, base)` writes it back as byte runs (refusing a
byte that would hold one nibble of the level only: none on the real set); `encode_sheet_source` writes a sheet-source
word. `upload_sheets` equals MAME's texture RAM 0 and 1 after the load on all three courses (frame 1335,
`tests/test_texture.py`, one MAME run per course).

`TextureError` (all encoders): wrong shape, non-integer values, values outside 0-15, an untextured header, a header
field `encode_tex_header` refuses, a texture outside the sheet or in the mip area, a `mips` other than the two.

## Tools

- `dtools/mame/lua/texdump.lua` (`DAYTONA_TEXDUMP_DIR`, `DAYTONA_TEXDUMP_AT`; run from `autoplay.lua`): dumps texture
  RAM 0 / 1, luma, palette and colour-translation RAM (and tile and char RAM, `tile.<f>.bin`, `char.<f>.bin`) at given
  frames; `load_dump(dir, frame)` reads the colour tables.
- `python -m dtools.arcade.texture <out.png> [course] [light]`: contact sheet of a course's scenery textures (game
  pixels: write it under `out/`).
