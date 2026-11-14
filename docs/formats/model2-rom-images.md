# Model 2 ROM images (`daytona.zip`)

Layout follows MAME's `src/mame/sega/model2.cpp` (`ROM_START( daytona )` and the `MODEL2_CPU_BOARD` macro),
BSD-3-Clause, copyright the MAME team. Loader: `dtools/arcade/romset.py`. The zip is a MAME merged set: the
parent's files are at the zip root, clones' files in subdirectories (ignored).

## Interleave rule

Every pair is `ROM_LOAD32_WORD` at offset `o` and `o+2`: ROM A supplies bytes 0-1 and ROM B bytes 2-3 of each
little-endian 32-bit word.

    out[o+0::4], out[o+1::4] = A[0::2], A[1::2]
    out[o+2::4], out[o+3::4] = B[0::2], B[1::2]

A pair of N-byte ROMs spans 2N bytes of the image. Unfilled image space is zero.

## Images

| Image | Size | Pair (A, B) | Offset |
|---|---|---|---|
| `program` | 256 KB | `epr-16722a.12`, `epr-16723a.13` | 0 |
| `data` | 16 MB | `mpr-16528.10`, `mpr-16529.11` | 0 (4 MB) |
| | | `mpr-16808.8`, `mpr-16809.9` | 0x400000 (4 MB) |
| | | `epr-16724a.6`, `epr-16725a.7` | 0x800000 (1 MB), then copied to 0x900000, 0xa00000 ... 0xf00000 (7 `ROM_COPY` copies) |
| `copro_data` | 4 MB | `mpr-16537.ic28`, `mpr-16536.ic29` | 0 |
| `polygons` | 16 MB | `mpr-16523.ic16`, `mpr-16518.ic20` | 0 |
| | | `mpr-16524.ic17`, `mpr-16519.ic21` | 0x400000 |
| | | `mpr-16525.ic18`, `mpr-16520.ic22` | 0x800000 |
| | | `mpr-16772.ic19`, `mpr-16771.ic23` | 0xc00000 |
| `textures` | 16 MB | `mpr-16522.25`, `mpr-16521.24` | 0 |
| | | `mpr-16770.27`, `mpr-16769.26` | 0x800000 (rest zero) |

Kept untouched in `RomSet.extra`: `epr-16726.bin` (comm), `epr-16720.7`, `epr-16721.8` (sound 68K),
`mpr-16491.32`, `mpr-16492.33`, `mpr-16493.4`, `mpr-16494.5` (samples), `epr-16488a.ic12` (drive board),
`opr-14742a.45`, `opr-14743a.46`, `opr-14744.58`, `opr-14745.59`, `opr-14746.62`, `opr-14747.63` (CPU board
TGP tables). CRCs for all 36 files are in `CRCS`.

`polygons` is addressed by 32-bit word index masked with `0x3fffff` (MAME's `polygon_rom_mask`, region bytes / 4 -
1): `poly_word(i)` reads LE32 at `(i & poly_mask()) * 4`; on a custom set the mask is `0x7fffff`.

`textures` holds texture **points and headers** (16-bit words addressed by `tpa` / `tha`), not texels: the texel
sheets are texture RAM, uploaded by the i960 from 1 MB blocks of `data` (`0x200000`, `0x300000`, `0x400000` per
course, `0x500000` common). See `docs/formats/model2-textures.md`.

## Free space (runs of 0xff) measured on the real set

| Image | Run | Start | Note |
|---|---|---|---|
| `program` | 12,648 bytes | 0x3ce98 | to end of image |
| `data` | 667,360 bytes | 0x65d120 | |
| `copro_data` | 882,370 bytes | 0x12893e | in each 2 MB half; the two halves are identical (mirrored) |
| `polygons` | 1,850,524 bytes | 0xe3c364 | to end of image; about 4.2 MB (4,398,096 bytes) of 0xff runs of 100 KB+ in all |
| `textures` | 709,304 bytes | 0xb52d48 | also 0x80ff10 (590,064) and 0xa3e2a8 (400,728); 0x400000-0x7fffff and 0xc00000-0xffffff are the zero fill |

(Earlier notes gave starts 0x80 higher for copro_data, polygons and textures; the starts above are the measured
ones, each run ending exactly where the next non-0xff byte begins.)

## The custom set (custom MAME build)

`tools/mame-daytona-rom-space.patch` (applied by `tools/build-mame.sh`, binary `build/mame/model2`, found by
`dtools.config.custom_mame()`) grows the `daytona` regions as far as the game's address decoding reaches
(`docs/spike/2026-11-sixteen.md`, "Room") and loads five more files. `RomSet.as_custom()` makes one from a stock set,
`RomSet.custom` tells it, `from_zip` reads either, `split_files` / `write_zip` write the 36 stock files (unchanged,
their CRCs as above) plus the five. Stock MAME ignores the five; the custom build runs the stock set unchanged (the
new areas then zero).

| Image | Stock size | Custom size (`REGION_SIZES_CUSTOM`) | File | At (image offset) | Length | Decoding limit |
|---|---|---|---|---|---|---|
| `program` | 256 KB | 256 KB | | | | not grown |
| `data` | 16 MB | 32 MB | `dx-data.bin` | 0x1000000 | 16 MB | the i960 data ROM window 0x02000000-0x03ffffff |
| `copro_data` | 4 MB | 32 MB | `dx-copro.bin` | 0x400000 | 28 MB | TGP word address 23 bits under bank bit 0x800000 |
| `polygons` | 16 MB | 32 MB | `dx-poly.bin` | 0x1000000 | 16 MB | `oba` word index 23 bits under bit 0x800000 |
| `textures` | 16 MB | 16 MB | `dx-tex1.bin`, `dx-tex2.bin` | 0x400000, 0xc00000 | 4 MB each | already the limit (23-bit word address); the two zero gaps |

Each file is a plain `ROM_LOAD` of exactly those image bytes (no interleave) declared `NO_DUMP`: MAME loads it when
present without a CRC check ("NO GOOD DUMP KNOWN"; `mame.build.run_set` takes that line per file as the proof the
custom set was loaded) and leaves the area zero when absent (the regions are `ROMREGION_ERASE00`). The new areas have
no copies: the data ROM's 1 MB copies stay 0x800000-0xffffff (`DATA_COPY_END`), copro_data's identical halves stay the
first 4 MB; `with_patch` writes a new area once and refuses a patch across 0x1000000 (data) or 0x400000 (copro_data).
`read_i960` serves the whole 32 MB window on a custom set.

### Free space of a custom set (`CUSTOM_FREE`)

| Image | Run | Start |
|---|---|---|
| `data` | 16,777,216 bytes | 0x1000000 (i960 0x03000000) |
| `copro_data` | 29,360,128 bytes | 0x400000 (TGP word 0x100000) |
| `polygons` | 16,777,216 bytes | 0x1000000 (word index 0x400000) |
| `textures` | 4,194,304 + 4,194,304 bytes | 0x400000 and 0xc00000 (words 0x200000, 0x600000) |

The table gives the areas before the Phase 8 reservations. `CUSTOM_FREE["data"]` is two runs, the area less
`romset.RESERVED`: grow_ids' sixteen-entry tables and records (`course.ID_HOME`, `0x1000000`-`0x100157f`, i960
`0x03000000`; record slots 0-2 are unused and stay 0xff), the new ids' SRAM sink (`0x1001580`, `0x260` bytes) and the
select's tables, maps and chars (`0x1f00000`-`0x1f1ffff`, i960 `0x03f00000`):
`(0x10017e0, 0xefe820)`, `(0x1f20000, 0xe0000)`. The program also has reserved runs `0x1e780`-`0x1f7ff` (the
select's code), `0x1f800`-`0x1fdff` (`course.widen_ids`' tables and code, `WIDE_HOME`, Phase 9) and `0x1fe00`-`0x1ffff`
(`course.grow_rankings`' sixteen-id ranking defaults and time bases, `RANK_HOME`, Phase 9) and `0x3ce98`-`0x3d297` (the
first 1,024 bytes of the program's free run: the shared empty actor table `build.EMPTY_ACTORS`, Phase 10), and work RAM
`0x501b00`-`0x501b3f` and `0x501b40`-`0x501b53` (`RESERVED_RAM`: the select's, widen_ids' shadows). `tgp.write_free` and the converter refuse reserved bytes. A free run is not "every 0xff byte": a byte a table or block owns is busy whatever its value
(`golden.owned_spans`: the reservations, each new id's own 1 MB texel block in the custom area,
`texture.custom_sheet_blocks`, and every course's tables), and the allocators (the converter's `_usable`,
`mirror.mirrored_set`) skip it; `tgp.write_free` also refuses a new id's texel block. Room per course on the custom set: see `docs/follow-ups/phase-08-m4a.md`, "Phase 8 (M4a)", "What M5
starts from".

All 0xff in `as_custom()`, outside every table of the three courses (`tests/test_romset.py`, `test_free_runs_custom`).
`tgp.write_free` writes the data and copro_data runs once (no copies).

## What `dtools check` verifies and `dtools build` writes (Phase 12)

`release.check_inputs` reads the root of `daytona.zip` (a merged set's clone folders are ignored): each of the 36 files
of `CRCS` present, its size `release.ROM_SIZES` (program, data and TGP pairs 128 KB / 2 MB / 512 KB / 2 MB as in the
table above; the sound, I/O and protection files 64 KB to 2 MB) and its CRC32 from the zip's directory, then the whole
set read (`RomSet.from_zip`, which checks each member's CRC as it inflates). A set holding `dx-*.bin` is refused: it is
a built set, not MAME's. `model1io.zip` beside it is not read (a warning when absent: MAME needs it). `dtools build`
writes `<out>/daytona.zip` = `split_files()` of the release set: the 36 stock names (the changed ones under their stock
names, so MAME lists them under WRONG CHECKSUMS) plus the five `dx-*.bin`; `release.member_sha1` is the sha1 of each
(the zip's own bytes vary with its timestamps), listed in `<out>/build-report.txt`.

## i960 memory map (excerpt)

| Address | Region |
|---|---|
| 0x00000000 | program ROM (256 KB, `program`) |
| 0x00200000 | RAM |
| 0x00220000 | program window (`program` from offset 0x20000) |
| 0x00500000 | work RAM |
| 0x00800000 | geometry engine |
| 0x00900000 | buffer RAM |
| 0x01000000 | tile RAM |
| 0x01800000 | palette |
| 0x01d00000 | backup SRAM |
| 0x02000000-0x03ffffff | data ROM (`read_i960` serves the first 16 MB, 0x02000000-0x02ffffff; all 32 MB on a custom set) |
