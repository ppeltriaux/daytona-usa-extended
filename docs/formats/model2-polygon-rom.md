# Model 2 polygon ROM — object format

Derived from MAME 0.289 `src/mame/sega/model2_v.cpp`, BSD-3-Clause, copyright the MAME team: the geometry
engine's object parsers and the 3D rasterizer's input state machine. Implemented in `dtools/arcade/polyrom.py`
(`decode_object`, `polygons`). Line numbers below are into that file.

## Addressing

Geometry command `0x01` (Object Data, `geo_object_data`, line 1715) takes four words: `tpa` (texture point
address), `tha` (texture header address), `oba` (object address), `obc` (object count).

| `oba` test (in order) | Object read from | Word index |
|---|---|---|
| `oba & 0x01000000` | polygon RAM 1 (fast) | `oba & 0x7fff` |
| `oba & 0x00800000` | polygon ROM (`RomSet.polygons`) | `oba & polygon_rom_mask` = `0x3fffff` for the 16 MB ROM |
| neither | polygon RAM 0 (slow) | `oba & 0x7fff` |

All addresses are indices of **32-bit little-endian words**. `obc` is the maximum number of records; `obc == 0`
becomes `0xfffff`. The parser is chosen by `geo->mode & 3`:

| mode | parser | lines | normal words |
|---|---|---|---|
| 0 | `geo_parse_np_ns` (normals present, no specular) | 1006–1152 | read |
| 1 | `geo_parse_np_s` (normals present, specular) | 1153–1308 | read |
| 2 | `geo_parse_nn_ns` (no normals, no specular) | 1309–1498 | skipped (`input += 3`); normal computed from the points |
| 3 | `geo_parse_nn_s` (no normals, specular) | 1499–1703 | skipped (`input += 3`); normal computed from the points |

The specular variants read exactly the words of their non-specular twins; they differ only in the lighting maths.

`decode_object(words, index, count, mode)` takes the already-masked word index and the effective count (the
caller does the RAM/ROM select and the `0 → 0xfffff` mapping).

## Layout

Floats are IEEE single (`<f`), the attribute is `<I`.

| Words | Content |
|---|---|
| 0–2 | seed point 0 `x y z` |
| 3–5 | seed point 1 `x y z` |
| then per record, 10 words | `attr`, normal `nx ny nz`, point `x y z`, point `x y z` |

Per record, what each parser reads (`input` advances):

| Record | modes 0–1 (np) | modes 2–3 (nn) |
|---|---|---|
| quad (`attr & 1`) | attr, normal, point, point (10) | attr, skip 3, point, point (10) |
| triangle (`attr & 3 == 2`) | attr, normal, point, skip 3 (10) | attr, skip 3, point, skip 3 (10) |
| end (`attr & 3 == 0`) | attr, normal, stop (**4**) | attr, stop (**1**) |

The np parsers read the normal before testing `attr & 3` (np_ns lines 1048–1061), so an end record in modes 0–1
needs its three normal words; the nn parsers test `attr & 3` first (nn_ns lines 1357–1370). Parsing stops at the end
record or after `obc` records, whichever comes first. A triangle's second point is never kept
(np_ns line 1139: `input += 3`).

`Object.word_count` is the number of words the parser advances over, end record included:
`6 + 10·records + (4 | 1 | 0)` (0 when `obc` ran out before an end record). `decode_object` also requires the
skipped words to lie inside the buffer and raises `PolyError` when anything runs past the end, or past
`max_records`.

## Attribute bits

Only `attr & 0x3ffff` is pushed to the rasterizer (np_ns line 1050); bits 18–31 are used by the geometry engine.

| Bits | Meaning | Used at |
|---|---|---|
| 0–1 | 0 end; 1 or 3 quad; 2 triangle (`Record.is_quad = attr & 1`, `Record.ends`) | raster lines 793–822 |
| 8–9 | link mode (`Record.link`, see below) | lines 529–557; nn parsers 1466–1491 |
| 10–11 | z-sort value: 0 previous polygon's, 1 min z, 2 max z, 3 error (1e10) | lines 436–451 |
| 12–16 | texture header offset, signed 5 bits, ×4 words added to `tha` after the polygon | lines 416–423 |
| 17 | double-sided: when clear, a back-facing polygon is culled | `check_culling` lines 245–250 |
| 18–22 | texture parameter index: diffuse, ambient, specular control (`Record.tex_param`) | np_ns line 1090 |
| 27–31 | distance coefficient table index, for texture LOD (`Record.coef`) | np_ns line 1105 |

Culling also reads `attr` (`check_culling`, line 433; defined at 243).

### The depth test (bit 7; the custom MAME only, v0.3)

Bit 7 is one of the 18 the geometry engine passes on and none of the original set's records uses (101,462 records of
the three courses read: the bits ever set are `0x21f03`). `tools/mame-daytona-rom-space.patch` gives it a meaning in
`model2_v.cpp` / `model2rd.ipp`: a polygon with it is depth tested per pixel. MAME draws the frame's polygons
nearest sort key first and a pixel once filled is never drawn again (the fill map). A depth-tested polygon marks the
pixels it draws (fill `0x7f`) and stores its 1/z there; a later depth-tested polygon nearer at that pixel replaces
it. Every other polygon keeps the rule as it was, against depth-tested pixels too: an untested polygon drawn first
keeps its pixels, and one drawn later never replaces a filled pixel. So a set with no such polygon draws exactly as
before (checked: a recorded drive of a set without the bit replays to the same pictures on the patched and the
unpatched build), and the converter sets it only on the Dreamcast models' polygons (`model.DEPTH_TEST`,
`convert_model(..., depth=True)`; `release_set`'s v0.3 chain). Stock MAME and the hardware ignore the bit.

## Strip and corner rule (rasterizer)

The geometry engine pushes the stream to `model2_3d_push` (line 769); command `0x01` collects it in
`command_buffer` (line 781):

| `command_buffer` | Content |
|---|---|
| 0, 1 | `tpa`, `tha` |
| 2–4 | **P0(n-1)**, first the seed point 0 |
| 5–7 | **P1(n-1)**, first the seed point 1 |
| 8 | `attr & 0x3ffff` (end when `& 3 == 0`, line 793) |
| 9, 10 | luma, distance (LOD) |
| 11–13 | **P0(n)**, the record's first point |
| 14–16 | **P1(n)**, the record's second point (quads only) |

When a quad (17 words) or a triangle (14 words) is complete it calls `model2_3d_process_polygon<4|3>` and resets
`command_index` to 8 (lines 786–822), so the next record overwrites slots 8–16 while 2–7 hold the previous points.

**Every record produces a polygon and updates the links**, whatever its link mode, but `check_culling`
(line 243, called at 433) culls some of them before drawing (only `if (cull == false)`, line 455, draws):

- **link 0 is always culled** (lines 253–255): a link-0 record only moves the strip's two points, as on Model 1;
- a back face is culled unless bit 17 (double-sided) is set (lines 245–250); the face bit is `0x00800000` of
  `command_buffer[9]`, i.e. the `0x100` the geometry engine adds to the luma when `dotp < 0`
  (np_ns lines 1086–1102, pushed as `luma << 15`);
- also culled: `1 / min_z` beyond the master z clip, or `max_z < 0` (lines 257–264); the rest is clipped.

`model2_3d_process_polygon` (line 328) builds its corners as (lines 340–367):

    v0 = P1(n-1)    v1 = P0(n-1)    v2 = P0(n)    v3 = P1(n)   (quads only)

so a quad is `(P1(n-1), P0(n-1), P0(n), P1(n))` and a triangle `(P1(n-1), P0(n-1), P0(n))`. For a triangle the
raster then copies P0(n) into P1(n)'s slot (lines 364–367: "the rope of P1(n) is achieved by P0(n-1)").

After drawing (or culling) the link bits `(attr >> 8) & 3` choose the next P0(n-1)/P1(n-1) (lines 529–557):

| link | update | next polygon starts on edge |
|---|---|---|
| 0, 2 | `P0(n-1) := P0(n)`, `P1(n-1) := P1(n)` | (P1(n), P0(n)) — v3, v2 |
| 1 | `P1(n-1) := P0(n)` | (P0(n), P0(n-1)) — v2, v1 |
| 3 | `P0(n-1) := P1(n)` | (P1(n-1), P1(n)) — v0, v3 |

(The comment on case 3, line 553, says "P1(n) -> P1(n-1)", but the code writes slots 2–4, i.e. P0(n-1).) For a
triangle P1(n) is P0(n) in every row. The nn parsers keep their own copy of the same four points with the same
link rule (lines 1466–1491) to compute the normal as `vector_cross3(p0, p1, p2)` (line 1384) of P0(n-1), P1(n-1), P0(n).

`polygons(obj)` returns one `Poly(points, normal, attr)` per record in this corner order; `normal` is the record's
normal (modes 0–1) or `None` (modes 2–3). `Poly.drawn` is `False` for link 0 (always culled) and `True` otherwise;
a polygon count should count drawn polygons only. The back-face and z culls depend on the view and are not
modelled.

**A culled polygon still costs** (Phase 10, M5a; `docs/spike/2026-11-courses-4-7.md`, "National Park Speedway"):
MAME's `check_culling` runs in the rasterizer, after the geometry engine has transformed the polygon, so a back face
culled at draw counts on the frame-rate bar (`geo.count_polygons`, every record whose link bits are not 0) and on the
5,000-word frame cap like a drawn one. Removing never-front-facing polygons lowers both only by what it removes from
the list, not by what the rasterizer skips. **A quad is drawn as one convex polygon** (`poly_manager::render_polygon`,
`build/mame/src/devices/video/poly.h`): its outline is the four projected corners and 1/z, u/z, v/z are interpolated
along the edges and across each span, so a folded or non-affine quad keeps the union outline, one normal (culling,
lighting) and a scanline-interpolated texture. That is MAME's model; real hardware was not checked.

## Measured for a converter (Phase 7, M3b)

Desert City's model written as scenery (`dtools/convert/model.py`; `docs/spike/2026-11-convert.md`, "Model"):

- **Winding and normal.** On Beginner's 7,669 drawn polygons the Newell normal of the rasterizer's corners points
  against the stored normal on every one, and the stored normal is -Newell / |Newell| within 0.03 degrees. So the
  corners run anticlockwise about the side the face is seen from, and the stored normal points away from the viewer
  (a road's is (0, -1, 0)). The converter writes that.
- **Seen side from the Dreamcast.** The strip flags' bits 0-1 (the PowerVR cull mode) decide it: after the z negation
  (a reflection) mode 2 keeps the Dreamcast corner order and mode 3 reverses it; modes 0-1 are double-sided (bit 17)
  and oriented by their stored normals.
- **Links.** After a polygon (v0, v1, v2[, v3]) the next starts on (P1, P0) = (v2, v1) for link 1, (v3, v2) for a quad
  on link 2 (a triangle's link 2 is degenerate, (v2, v2)) and (v0, v3) for a quad on link 3 ((v0, v2) for a triangle).
  A polygon no exit reaches is preceded by a link-0 move record (Beginner's `0x00040401`, zero normal, step 0). Move
  records read their texture points too (8 words), but a move steps the texture header by 0, a drawn record by 1.
- **Objects.** A scenery object is geometry mode 1 (normals present), written back to back at the end of the polygon
  run in cell order, `oba` = `0x800000 | word index`; one object per cell, more past `model.MAX_RECORDS`.

## Observation on real data

Decoding objects found by scanning the Daytona polygon ROM in mode 0 (word indices 3783, 4753, 6014, 6984) gave
353 link-0 records, every one with a zero normal, against 905 link-1/2/3 records with unit normals. That is
consistent with MAME's link-0 cull: the data carries no normal for polygons that are never drawn, so link-0
records only move the strip (about 28% of the records in this sample).

## Encoding

`encode_object(obj)` (`dtools/arcade/polyrom.py`) is the inverse of `decode_object` for the same mode: it writes the
seeds, ten words per record and the end record, and on the real ROM it reproduces every Beginner, Expert and Advanced
scenery object word for word (325 + 256 + 256 objects, mode 1). Three things the decoder used to drop carry data, so
`decode_object` now keeps them raw (not part of `Record` / `Object` equality):

- **A triangle's skipped second point** (`input += 3`) is not padding: in the 308 ROM objects of Phase 1 the 371
  triangles' skipped words look like real coordinates (the most common triple occurs 24 times). Kept as
  `Record.skipped`.
- **The end record's three words after its attribute** (modes 0-1 read them as a normal) are not a normal: they are
  `0x00040401, 0, 0` in 255 of the 308 objects, `0x00041401, 0, 0` in 29, zero in 15. Kept as `Object.end`, with the
  end attribute (0 in all 308).
- In modes 2-3, the skipped normal words (`Record.skipped`, before a triangle's skipped point).

The layout comes from the records (normal present: modes 0-1; absent: modes 2-3) and `word_count` (end record of
4 words, 1 word, or none when `obc` ran out). Empty `skipped` / `end` encode as zeros, so a hand-built object needs
only its points and attributes. `PolyError`: an attribute that is not a u32 or has `attr & 3 == 0` (the parser would
end there), a quad without its second point, a triangle with one (never read), records of both layouts, a point that is
not three single-precision floats, a `word_count` that does not fit. Floats go through Python doubles, which is exact
for every non-NaN single; no NaN or infinity occurs in those objects.
