# Model 2 Daytona USA — the attract logo (DAYTONA USA)

Recon 13 Nov 2026 (Phase 12 Task 3; MAME 0.289, the custom build; tile, char and palette RAM dumped at the logo's
frames on the stock set and on the M5b set, identical). Decoded by `dtools/arcade/title.py`; the runs are in
`docs/spike/2026-11-release.md`, "The title". Tile layers, chars, palettes and upload sets as in
`docs/formats/model2-course-tables.md`, "Tile layers and uploads".

## Where the game shows its name

- **The attract loop's title screen**, once per loop with no coin: frames 3265-3668 and 6432-6835 after a stock boot
  (the loop: the demo race, Beginner's ranking, the V.R. demo, the logo). A white field between two checker bands,
  the four flags, the black italic **DAYTONA** over **USA** with a trademark, under them the HUD (CREDIT, INSERT
  COIN(S), © SEGA 1994, the SEGA logo). Tile layers only; no polygons.
- **Circuit Select's background** is the logo screen's layer-2 map (below): the field, the bands and the letters'
  grey shadow behind the panels. No black letters there.
- Nowhere else: the boot's settings page, the races, the rankings and the V.R. demo carry no title (the HUD's SEGA
  logo and copyright only).

## The logo screen

The logo state `0x222fd0` (the first entry of the state table `0x222fb8`, its byte `0x36(g13)`) blits three maps with
`0x19304(map, tile RAM)` and moves on:

| Program | Map (data) | Size | To | What |
|---|---|---|---|---|
| `0x222fd0` | `0x02835a78` (by `0x192c4`, another blit) | | layer 2 | not traced (overwritten by the third) |
| `0x222fe4` | `0x0282a270` | 34 x 62 | layer 0, tile row 7 (`0x01000380`) | black letters, trademark, flags |
| `0x222ff8` | `0x0282499e` | 48 x 62 | layer 2, row 0 (`0x01004000`) | field, bands, the letters' grey shadow |

Each blit is `lda <tile RAM>,g1; lda <map>,g0; call 0x19304` (program offsets `0x22fe4` and `0x22ff8`; the map
words at `+4` and `+12`). The layer-0 map is read there only; the layer-2 map is also blitted by `0x226a4c` (Circuit
Select). Both are wide maps with base 0 in the data ROM's copied 1 MB block (each byte has seven copies).

- **Layer 0** uses chars `0x1800-0x19fe` (all 511 of the block, palettes `0x30-0x33`, four copies of one 16-word
  palette: pen 1 black, pens 2-8 a grey ramp from 68 to 228 (the letters' anti-aliasing towards white), pens 10-13
  the flags' red, blue, green and yellow). Its tiles are behind the polygons (bit 15 clear), bit 14 clear. Outside
  the letters the map holds char `0x1800`, which has no pixel (transparent).
- **Layer 2** uses chars `0x1a00-0x1b7a` (all 379, palettes `0x34-0x36`: pen 6 the shadow's grey 214, pen 7 white).
  The shadow is the letters' silhouette, one or two pixels lower: grey shows only along the letters' lower edges.
- **The chars are loaded once, at boot**, by the second boot upload set (data word `0x02700000` -> `0x02700008`):
  `0x1800+511` and `0x1a00+379`, with the palette runs that fill `0x30-0x36`. Nothing reloads them: every later
  upload (screens, courses, the select) writes chars from `0x3000`.
- **The wipe.** Layers 0 and 1 are in the chip's split mode (word `0x5004` = `0x4000`): word `0x5000` steps from
  `0x0100` to `0x0400` over the first frames (3265-3300), showing layer 0 scrolled left of the split and layer 1 (all
  the font's space) right of it, so the black letters slide in from the left over the static field and shadow. At
  rest the scroll is 0 and the window masks are clear. The palettes do not change while the logo shows (no fade); the
  next screen replaces it.
- **On screen**, layer 2 then layer 0 over it (pen 0 transparent), each palette word through the colour translation
  table's tile entry and MAME's gamma (`title.palette_rgb`): equal to MAME's snapshot pixel for pixel.

## Room for "Extended"

- **In the logo's blocks: none.** Layer 0's block has one unloaded char (`0x19ff`), layer 2's five (`0x1b7b-0x1b7f`);
  the palette of a char is its block's, so no other char can take the logo's palettes.
- **The boot's holes.** Below `0x3000` the boot uploads leave 27 runs unloaded (1,583 chars) that nothing writes after
  boot and no screen draws. A tile with bit 14 set draws its char with palette `0x80 + (char >> 7)`; the boot writes
  `0x80-0x84` only, the select `0xf6-0xff`, nobody `0x85-0xf5`. `title.ROOM` is the holes less blocks 2-4 (whose
  bit-14 palettes `0x82-0x84` the boot writes): 24 runs, 1,445 chars (`title.BUDGET_CHARS`).
- **The budget.** Loaded at boot by a copy of the boot upload set, all 1,583 hole chars keep the frame counters
  `0x5010b0` / `0x540058` on every frame to 1200 and Beginner from the select call for call with the M5b set, so the
  room is the limit, not the boot's time. A variant needs 74-129 chars.
- **No instruction is added.** The chars and one 16-word palette per hole block used go into a copy of the boot
  upload set (its char list plus the new pairs, its palette list plus the new runs) in the custom data run
  `0x01f20000-0x01ffffff` (outside `build.ID_DATA_FREE`), and the data word `0x02700000` names the copy. A copy of the
  layer-0 map (read by the logo state only) takes the word's cells, tile `0x4000 | char`, in the same run, and the
  logo state's map word (program `0x22ff0`, the 32-bit displacement word of the two-word `lda 0x282a270,g0` at
  `0x22fec`; the opcode word is unchanged) names the copy; the original map
  bytes stay as they are (Task 4: every byte the title changes was free or is one of the two words). Layer 2 is left
  alone, so Circuit Select's background does not change. The word's shadow is drawn on layer 0 with its letters (its
  palette carries the shadow grey), so during the wipe it slides in with them, where the logo's own shadow waits on
  layer 2.
- **The HUD's band.** While the logo shows, the HUD writes layer 0's tile rows 38-40, columns 12-49 (INSERT COIN(S),
  blinking; its tiles `0x8280-0x8285` in place of the map's transparent `0x1800`), so a word cell there would be
  overwritten. `title.HUD_ROWS` / `HUD_COLS`; `_drawn` refuses such a cell. The HUD's other rows (41-46) are below the
  map, which ends on row 40.

## What title.py draws

`extended_pixels(rs, variant)`: the logo's palette indices at rest with "Extended" drawn in the white room left of
USA, in the logo's own pens (layer 0's black and grey ramp by coverage, layer 2's grey for the shadow), its slant
(0.172 px right per px up, measured on the D's stem) and weight (stems about 0.39 of the cap height, small counters).
The letters are drawn (vector shapes, 8 x 8 samples a pixel): the logo's D, T and N are fused with their neighbours
in DAYTONA, one run of ink with no free column even unslanted, so none can be lifted whole. Four variants
(`title.VARIANTS`: box, cap height, case, slant); `new_chars` lists the distinct chars a variant needs; `mockup`
puts the logo and a variant side by side at 2x, through the set's own palette.

## What extended_title writes (Task 4)

Pascal's pick (`title.PICKED`): **v3**, "Extended" under DAYTONA's left half, cap 24 px, slant 0.30, box
(8, 229)-(248, 255): 74 chars. `title.extended_title(rs)` on a custom set, its word's cells still blank:

| Where | What |
|---|---|
| data `0x01f20000` (i960 `0x03f20000`) | the layer-0 map's copy (wide, base 0, 34 x 62): the logo's tiles, the word's cells `0x4000 \| char` |
| after it | one char block per ROOM run used (v3: 74 chars at `0x0698-0x06e1`, block `0xd`) |
| after them | the char list: the boot set's pairs, then the title's |
| after it | the palette list: the boot set's runs, then one 16-word run per hole block (v3: palette `0x8d`) |
| after it | the upload set: the two list addresses |
| data `0x00700000` (i960 `0x02700000`) | repointed: the boot's second upload set is the copy |
| program `0x22ff0` | repointed: the logo state's layer-0 map is the copy |

- **One palette.** The word's pixels come from two logo palettes (layer 0's black and seven-step grey ramp, pens 1-8
  of `0x30`; layer 2's shadow grey, pen 6 of `0x34`), but a bit-14 char draws through its block's palette alone.
  `title.title_palette` writes layer 0's palette with the shadow grey on pen 15 (no ramp step is that grey; the word
  uses pens 1-8), so every pixel shows the word it showed in `extended_pixels`.
- **Read back.** `title.titled(rs)`: the boot word names TITLE_RUN. `title.title_spans(rs)`: TITLE_RUN from its
  start to the upload set's end (8,240 bytes for v3) and the two words; `golden.owned_spans` owns them. Golden's
  course decode follows the repointed boot word too ("screen upload pointer" of every course), as it follows the
  select's repointed `0x02700004`.
- **MAME** (`tests/test_release_mame.py` `test_attract_says_extended`): at frames 3450 and 6617 tile, char and palette
  RAM hold what was written, the snapshot is `title.render` of the set above the HUD, and the attract loop's display
  lists, render rate and frame counters equal the M5b playtest set's (the same chain without the title or the names) on every frame from boot to 6900 and through
  Circuit Select's opening.

