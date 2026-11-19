# Model 2 Daytona USA course tables

The arcade game's per-course data, as the i960 program finds it. Decoder and encoder: `dtools/arcade/course.py`.
The tile-layer primitives the select screen and backdrop are built from are in `dtools/arcade/tiles.py`.
The runs and disassembly behind each section: `docs/spike/2026-11-arcade-course-tables.md`.

Addresses are i960 bus addresses. Data ROM `0x02000000-0x02ffffff` is `RomSet.data` at offset `addr - 0x02000000`;
`CourseEntry` fields hold those offsets. Program ROM `0x00000000-0x0003ffff` is `RomSet.program`; the window
`0x00220000-0x0023ffff` is program offset `0x20000+`. All words are 32-bit little-endian.

Per-course tables are indexed by the game's **course id**, the byte at work RAM `0x00501460`:

| Id | Course | Circuit Select panel |
|---|---|---|
| 0 | Beginner (Three Seven Speedway) | 0 |
| 1 | Expert (Seaside Street Galaxy) | 2 |
| 2 | Advanced (Dinosaur Canyon) | 1 |

## Course table

`COURSE_POINTERS = 0x00004770` (program ROM): four words, one per id, each the address of that id's course record.
The loader at `0x46e8` reads `0x4770[id*4]`, keeps the record address at work RAM `0x5fe5e4` and copies record words
0-5 to `0x501400-0x501414`; `0x4534` then follows them.

`COURSE_TABLE = 0x02600020` (data ROM, offset `0x600020`), `COURSE_ENTRY_SIZE = 0x20`: record `id` is at
`0x02600020 + 0x20 * id`. The 32 bytes before it, `0x02600000-0x0260001f`, are eight upload-list pointers for the
screens that are not courses (the select screens point `0x5fe5e4` at `0x02600008`-`0x02600010`, through
`0x23899c[cursor]`, and the code at `0x5cbc` reads `0x02600000`).

| Word | Offset | `CourseEntry` field | What it points at | Read by |
|---|---|---|---|---|
| 0 | `0x00` | `upload_list` | a struct of list pointers (word 0 a list for `0x1dd78`, word 1 for `0x1dde8`, word 2 on further lists); the lists copy course data, a half-word at a time, while the course loads | `0x1de40`, `0x1dfd0` (through `0x5fe5e4`) |
| 1 | `0x04` | `step_table` | `u32 n`, `u32 steps`, then `steps` words, one per road polygon of the ground table: `u16` segment, `u16` flags (`0x00010000`, `0x00010001`, ... on the real set; see "Road line") | `0x4534`: `n` to `0x501434`, `steps` to `0x501438`, the word array's address to `0x501430` |
| 2 | `0x08` | `block_ref_a` | the address of a word that holds a pointer to a 32 KB (`0x8000`) block: the ground-query grid, 128 x 128 `u16`, one per 16-unit cell (see "Road line") | `0x4668` (`ld (r3)`) to `0x501428` |
| 3 | `0x0c` | `block_ref_b` | the next word: a pointer to the block after that one, which runs to the next course's `step_table`: one attribute byte per ground polygon | `0x467c` to `0x50142c` |
| 4 | `0x10` | `line_table` | the AI lines: eight (`u32` pointer, `u32` count) pairs; count = `n + 1`; the lists follow the pairs ("AI lines") | `0xe8a0`: the pair indexed by the word at `0x20(g0)`, via `0x501410` -> `0x50144c` |
| 5 | `0x14` | `record_table` | the road line: `u32 n`, `u32 8` (lanes per record), then `n + 1` records of 164 bytes, ending where the 32 KB block starts ("Road line") | `0x4690` -> `0x501448` (= the record array, word 5 + 8) |
| 6 | `0x18` | `raw[0]` | unknown; equals the pointer stored at word 2's address | no reader found |
| 7 | `0x1c` | `raw[1]` | unknown; same value as word 6 | no reader found |

Field names are provisional, after the structure and the readers; words 4 and 5 are now decoded as the AI lines and
the road line (sections below), and words 1-3 are described there as far as the road line needs them. The names are
kept so that code written against them keeps working.

Each course's blobs from `step_table` on are one contiguous run: `step_table`, the two words at `block_ref_a`/`_b`,
`record_table`, the 32 KB block, the block after it, then the next course's `step_table`. Ids 0, 1, 2 (then the
id-3 stub) follow each other in data ROM from offset 0. The records, then each `line_table` with its lists, are a second
run from `0x600000` up to the free run at `0x65d120`; the `upload_list` structs and their lists a third, from
`0x073f9c`.

`course_table(rs)` returns the records of ids 0-2. It checks that `0x4770[id]` is `COURSE_TABLE + 0x20 * id` and
that words 0-5 are inside data ROM, and raises `CourseError` naming the address otherwise. `encode_course_entry`
refuses offsets outside `0-0xffffff` and `raw` that is not two `u32` words.

### Example values (MAME 0.289 `daytona`)

Data-ROM offsets (add `0x02000000` for the i960 address):

| Field | Beginner (0) | Expert (1) | Advanced (2) |
|---|---|---|---|
| record | `0x600020` | `0x600040` | `0x600060` |
| `upload_list` | `0x074240` | `0x074268` | `0x081180` |
| `step_table` | `0x000000` | `0x010a24` | `0x043200` |
| `block_ref_a` | `0x0007e4` | `0x011acc` | `0x043ed0` |
| `block_ref_b` | `0x0007e8` | `0x011ad0` | `0x043ed4` |
| `line_table` | `0x6000a0` | `0x60b360` | `0x641dc0` |
| `record_table` | `0x0007ec` | `0x011ad4` | `0x043ed8` |
| `raw` | `0x02008584`, `0x02008584` | `0x020395b8`, `0x020395b8` | `0x020576dc`, `0x020576dc` |
| `n` (step and record tables) | 195 | 990 | 486 |
| `steps` | 503 | 1064 | 818 |
| 32 KB block (`*block_ref_a`) | `0x008584` | `0x0395b8` | `0x0576dc` |
| block after it (`*block_ref_b`) | `0x010584` (1,184 bytes) | `0x0415b8` (7,240 bytes) | `0x05f6dc` (2,988 bytes) |
| line-table count | 196 | 991 | 487 |

The stub record for id 3 (`0x600080`) has `step_table` `0x060288`, `n` 249, and a `line_table` at `0x65cf20` that
repeats one (pointer, count `0xfffffff8`) pair eight times. Circuit Select never produced id 3 on the original set; on a Phase 8 set it does (Desert City, below).

### Other per-course arrays

The course load also indexes these arrays by id (four entries each, id 3 included). `course.py` decodes, for ids
0-2: `0x2805298` ("Scenery placement"); `0x2823c10`, and `0x2805228` / `0x2805234` at one mode (and difficulty) per
call ("Checkpoints"); `0x2805240` at cabinet entry 0 only, `0x28050e8` ("Start grid") and, since Phase 6,
`0x28058bc` ("Per-segment flags"). It does not decode `0x15a0`, `0x233bfc` beyond the actors or the other entries of
`0x2805240`:

| Array | Entry | Beginner / Expert / Advanced | Stored to |
|---|---|---|---|
| program `0x15a0` | data-ROM address of a 1 MB block | `0x2200000` / `0x2400000` / `0x2300000` | copied by `0x146c` to `0x12200000+` |
| window `0x233bfc` | program-window address | `0x233c0c` / `0x234374` / `0x234d90` | `0x501444` |
| data `0x28058bc` | data-ROM address | `0x28058cc` / `0x2805a54` / `0x2806212` | `0x501450` |
| data `0x2823c10` | data-ROM address | `0x2823c20` / `0x2823c24` / `0x2823c3c` | `0x50143c` |
| data `0x2805298` | data-ROM address | `0x284fccc` / `0x28500cc` / `0x28504cc` | `0x501420` |
| data `0x2805228`, `0x2805234` | per mode, an array by id (`0x2805228`: then by difficulty) | | `0x501440`, `0x5010ec` |
| data `0x2805240` | per cabinet setting, an array by id | 40 / 30 / 20 (entry 0) | `0x501080` |
| data `0x28050e8` | `s16` | 0 / -1 / -1 | `0x501850` |

### Course ids 0-15 (grown sets)

A grown set (`course.grow_ids`, on the custom build: `docs/formats/model2-rom-images.md`) holds course ids 0-15
(`course.COURSE_IDS`): 0 Beginner, 1 Expert, 2 Advanced as before, 3 Desert City (Phase 8, Task 5), 4-7 M5's courses,
8-15 the mirrors (M4b). Every array the game indexes by the course id moves to a sixteen-entry table in the custom
data area (`course.ID_HOME` = i960 `0x03000000`, data offset `0x1000000`) and its readers are repointed; ids 0-2's
own tables (records, road lines, grounds, scenery, select records) stay where they are. Until an entry is filled,
ids 3-15 hold entry 0's bytes (Beginner's); an array whose fourth entry is real keeps it (`MUSIC`). Discovery and
evidence: `docs/spike/2026-11-sixteen.md`, "Ids".

- **How each array is found and repointed.** Every reader loads the array's address whole: an i960 MEMB instruction
  with a 32-bit displacement (`ld 0x4770[g0*4],g0`, `lda 0x6c98,r5`, ...), whose displacement word (instruction + 4)
  is a *site*, or a pointer word one level up (`TIME_TABLES[mode]`, `0x230a6c[gear]`, `0x2813974[setting]`), also a
  site. So any address works, and every table goes to the data area: none needs the program window's free run
  (11,624 bytes, unused). The sites are the only ROM words that hold an array's address (`test_id_arrays_readers_found`
  checks every program word at any offset and every data word; data words in `0x02900000-0x02ffffff` are the
  `ROM_COPY` mirrors of `0x028xxxxx`, which `RomSet.with_patch` rewrites with their first copy).
- **The course records.** `COURSE_POINTERS[i]` points at `course.record_address(i)`: `COURSE_TABLE + 0x20 * i` for
  ids 0-2, `ID_RECORDS + 0x20 * i` (`ID_RECORDS` = `ID_HOME`, the first three slots unused) for ids 3-15, whose records
  start as copies of Beginner's. There is no room after `0x02600060`: Beginner's line table starts at `0x026000a0`
  right after the id-3 stub record (`0x02600080`, now unread).
- **Masks.** Six readers mask the id before indexing (`and 3,rX,rX`: an id 4-15 would read entry `id & 3`); grow_ids
  widens them to `and 15` (`course.ID_PATCHES`: `0xd474`, `0xd49c`, `0x195cc`, `0x19700`, `0x225b50`, `0x227594`).
- **The time-attack records `0x230b9c`.** `0xb864` indexes them `(id - 1) * 8 + gear * 16 + section` (Expert and
  Advanced only; 32 SRAM word pointers), guarded at `0xb818` by `and 3,g9,g9; cmpi g9,0; be` (id 0 skips). An id 3
  or more would index past the table, so the `and` becomes `cmpobl 2,g9,0xb848` (`ID_PATCHES`): ids 3-15 skip it like
  Beginner. The second reader of the same guard's `g9`, `0xbad8 ld 0x230ccc[g9*4]`, is a site of `WORDS_230CCC`.
- **The course vote.** After Circuit Select and the link wait, `0x32ac` stores to `0x501460` the id whose half-word
  at `0x501474 + 2 * id` is largest: high byte the cabinets' picks (`0x3504`-`0x3524` count each link packet's id,
  byte `0x14`), low byte a priority (`0x3064`-`0x3080`: Beginner 2, Advanced 1, Expert 0). It scanned ids 0-2 only, so
  a picked id 3 raced as id 0 (traced: the vote's store at frame 766 rewrote 3 to 0). The nine bytes `0x3048` clears
  (`0x501473`: a count byte and four half-words) already hold id 3's slot, so `ID_PATCHES` makes the scan four slots
  (`0x32bc mov 4,g9`, `0x32d0 subi g9,4,g2`); ids 0-2 vote as before (id 3's slot is 0 unless picked; a tie keeps the
  earlier slot). Ids 4-15 could not pass this four-slot vote (their slots would lie on `0x501480` and `0x501488`, which other code uses);
  the Phase 8 select moves the vote to free work RAM with sixteen slots (`0x501b03`, `0x501b04 + 2 * id`, below).
- **The TGP.** Its slot table already has sixteen slots (the course id is sent to function `0x0d`, slot = id; slots
  4-15 repeat slot 0's). grow_ids gives slot 3 slot 0's two words; the stub's list block and pool (`0x18000`,
  `0x465a0`) stay in place and still end Advanced's blocks (`tgp.STUB_BLOCKS`).
- **The one non-id reader of an original array.** The texel upload's mip loop (`0x1534 ld 0x1578[r10*4],r11`) reads
  its run-length table (ten words at `0x1578`) one word past its end, into `COURSE_SHEETS`' entry 0 (`0x02200000`, a
  run that never ends inside a 512-word row). That is not a per-id read, so the original word stays as it is and
  must not be rewritten (`course.OLD_ARRAY_READERS`).

The arrays grow_ids moves (`course.GROWN_ARRAYS`, in layout order; "decoded" = read by a `course` / `golden` decoder,
`golden.DECODED_ARRAYS`; `course.ID_ARRAYS` also holds `GRID_LANES` and `BILLBOARD_REF`, which `widen_ids` makes per
id: "A mirror tests as its original"):

| Array | Original | Stride x entries | Sites patched | New (16 entries) | Reader, use |
|---|---|---|---|---|---|
| `COURSE_POINTERS` (decoded) | `0x4770` | 4 x 4 | `0x46f4` | `0x3000200` | 0x46f0 ld [id*4]: the course record (ids 3-15: ID_RECORDS) |
| `COURSE_SHEETS` (decoded) | `0x15a0` | 4 x 4 | `0x1448` | `0x3000240` | 0x1444 ld [id*4]: the texel block 0x146c uploads (texture.py) |
| `ATTRACT_NEXT` | `0x2160` | 1 x 4 | `0x2120` | `0x3000280` | 0x211c ldob [id]: the next course id (then id rem 4, 0x2138) |
| `OUTPUT_CODES` | `0x48a4` | 1 x 4 | `0x4794` | `0x3000290` | 0x4790 ldob [id]: or-ed with 0x48a8[0x501464] / 0x70 into the 0x502900 queue (not traced) |
| `CAMERA_SCRIPTS` | `0x6c98` | 4 x 4 | `0x63e8` | `0x30002a0` | 0x63e4 lda, 0x6418 ld [id*4]: the pre-race camera script |
| `SHIFTS_8410` | `0x8410` | 1 x 4 | `0x83fc` | `0x30002e0` | 0x83f8 ldob [id]: a shift count, result to 0x5014f7 |
| `TABLES_DF44` | `0xdf44` | 4 x 4 | `0xdee8` | `0x30002f0` | 0xdee4 ld [id*4]: a per-id u8 threshold table (0xdeec) |
| `PROGRESS_MASKS` | `0x175e0` | 4 x 4 | `0x176a0` | `0x3000330` | 0x1769c ld [id*4]: the progress-mask records (0x17688) |
| `ACTOR_TABLES` (decoded) | `0x233bfc` | 4 x 4 | `0x46b8` | `0x3000370` | 0x46b4 ld [id*4] -> 0x501444 |
| `GRID_BASES` (decoded) | `0x230e48` | 2 x 4 | `0xe8e0` | `0x30003b0` | 0xe8dc ldis [id*2]: the grid's base segment |
| `CAR_TABLES_230E50` | `0x230e50` | 4 x 4 | `0xe758` | `0x30003d0` | 0xe754 ld [id*4]: the pit stalls, u32[8] per car ident (pit_stalls; "Pit") |
| `OFFSETS_230B18` | `0x230b18` | 4 x 4 | `0x63bc` | `0x3000410` | 0x63b8 ld [id*4]: an index offset into 0x230b28 |
| `TUNNEL_BYTES` | `0x230742` | 1 x 4 | `0x6d0c` | `0x3000450` | 0x6d08 ldob [id] -> car +0x85 |
| `TUNNEL_STRIDE` | `0x230746` | 4 x 1 (keeps 1) | `0x6d18` | `0x3000460` | 0x6d14 ld: the stride of TUNNEL_LIGHTS (20), not per id; moved with the records it strides (entry 0 is the one read) |
| `TUNNEL_LIGHTS` | `0x23074a` | 20 x 4 | `0x6d24` | `0x30004a0` | 0x6d20 lda (id * the word at 0x230746, 20): five words to +0x70-+0x80 |
| `WORDS_230CCC` | `0x230ccc` | 4 x 4 | `0xb668`, `0xbadc` | `0x30005e0` | 0xb664 ld [id*4] -> 0x501ac4; 0xbad8 ld [g9*4] (g9 = the id, from 0xb818: time attack) |
| `TABLES_231960` | `0x231960` | 4 x 3 | `0xced0` | `0x3000620` | 0xcecc ld [id*4]: a per-id table (id 3: not a pointer) |
| `OFFSETS_231B34` | `0x231b34` | 4 x 4 | `0xd420` | `0x3000660` | 0xd41c ld [id*4]: an index offset into 0x2880004 |
| `FLOATS_231C44` | `0x231c44` | 4 x 4 | `0xd47c` | `0x30006a0` | 0xd478 ld [(id & 3)*4] -> +0x20 (mask widened) |
| `FLOATS_231C54` | `0x231c54` | 4 x 4 | `0xd4a4` | `0x30006e0` | 0xd4a0 ld [(id & 3)*4] (mask widened) |
| `FLAGS_23329C` | `0x23329c` | 4 x 4 | `0x195d4`, `0x19708` | `0x3000720` | 0x195d0, 0x19704 ld [(id & 3)*4] (masks widened) |
| `TABLES_2332AC` | `0x2332ac` | 4 x 4 | `0x19880` | `0x3000760` | 0x1987c ld [id*4]: per lap still to run (0x5010ec less the player's lap + 1, clamped 0-15) a list of 16-byte records (segment, a value, a handler, a pad): 0x19868 fires the next record when the player's segment reaches its segment (the handler called, or the value stored at 0x501a02), so segment-keyed; not remapped for the reverse ids (M9 stage 2, step 2) |
| `TABLES_236F58` | `0x236f58` | 4 x 4 | `0x1c4f4` | `0x30007a0` | 0x1c4f0 ld [id*4]: per car ident & 7, 16 bytes |
| `TABLES_2370E8` | `0x2370e8` | 4 x 4 | `0x1c534` | `0x30007e0` | 0x1c530 ld [id*4]: per car ident, 32 bytes |
| `TABLES_232CB0` | `0x232cb0` | 4 x 4 | `0x16bec` | `0x3000820` | 0x16be8 ld [id*4] (r4 = the id from 0x16b94) |
| `TABLES_236B40` | `0x236b40` | 4 x 4 | `0x2201c8` | `0x3000860` | 0x2201c4 ld [id*4]: per actor ident, 16 bytes |
| `SELECT_238EFC` | `0x238efc` | 4 x 4 | `0x225b58` | `0x30008a0` | 0x225b54 ld [(id & 3)*4] (mask widened): select screen |
| `SELECT_UPLOADS` | `0x23899c` | 4 x 4 | `0x225ea8` | `0x30008e0` | 0x225ea4 ld [id*4] -> 0x5fe5e4 (data 0x02600008-) |
| `SELECT_SOUNDS` | `0x238ee4` | 4 x 4 | `0x225f68` | `0x3000920` | 0x225f64 ld [id*4]: a sound call (0x1a064) |
| `SELECT_PANEL_OF_COURSE` (decoded) | `0x238f0c` | 4 x 4 | `0x226460`, `0x22657c` | `0x3000960` | 0x22645c, 0x226578 ld [id*4] |
| `SELECT_2389AC` | `0x2389ac` | 4 x 4 | `0x226bfc` | `0x30009a0` | 0x226bf8 ld [id*4]: an upload (0x19304) |
| `SELECT_PALETTES` | `0x23895c` | 16 x 4 | `0x226c2c` | `0x30009e0` | 0x226c28 lda [id*16]: the select palette animation |
| `RANKING_SCREEN_2273E8` | `0x2273e8` | 8 x 4 | `0x22725c` | `0x3000ae0` | 0x227258 lda [id*8]: two words (tile position) |
| `RANKING_SCREEN_2273D8` | `0x2273d8` | 4 x 4 | `0x227284` | `0x3000b60` | 0x227280 ld [id*4]: an upload (0x19304) |
| `RANKING_TABLES` (decoded) | `0x2390e4` | 4 x 4 | `0x22759c` | `0x3000ba0` | 0x227598 ld [(id & 3)*4] -> 0x50138c (mask widened) |
| `RANKING_LAST` | `0x2390f4` | 4 x 4 | `0x2275ac` | `0x3000be0` | 0x2275a8 ld [(id & 3)*4] -> 0x5013b4 |
| `TIME_ATTACK_230A90` | `0x230a90` | 4 x 4 | `0x230a88` | `0x3000c20` | 0xbf8c ld (0x230a88[gear])[id*4]: SRAM words |
| `TIME_ATTACK_230AA0` | `0x230aa0` | 4 x 4 | `0x230a8c` | `0x3000c60` | 0xbf8c ld (0x230a88[gear])[id*4]: SRAM words |
| `TIME_ATTACK_230AB8` | `0x230ab8` | 8 x 4 | `0x230a70`, `0x230ab0` | `0x3000ca0` | 0xb524, 0xc0b4 lda (per gear)[id*8]: SRAM word pairs |
| `TIME_ATTACK_230AD8` | `0x230ad8` | 8 x 4 | `0x230a78`, `0x230ab4` | `0x3000d20` | 0xb524, 0xc0b4 lda (per gear)[id*8]: SRAM word pairs |
| `ROLLING_STARTS` (decoded) | `0x28050e8` | 2 x 4 | `0x3a90` | `0x3000da0` | 0x3a8c ldis [id*2] -> 0x501850 |
| `MUSIC` | `0x28050f0` | 4 x 4 (keeps 4) | `0x3be4` | `0x3000dc0` | 0x3be0 ld [music*4]: the music id 0x501a98 is the course id unless a view button picks 0-3, so all four entries are kept |
| `BEST_LAPS_2805100` | `0x2805100` | 4 x 4 | `0x4260` | `0x3000e00` | 0x425c ld [id*4]: an SRAM word (written) |
| `BEST_LAPS_2805110` | `0x2805110` | 4 x 4 | `0x4274` | `0x3000e40` | 0x4270 ld [id*4]: an SRAM word (written) |
| `TIME_TABLES[0]` (decoded) | `0x281f458` | 4 x 4 | `0x2805228` | `0x3000e80` | 0x45ec ld (TIME_TABLES[mode])[id*4] |
| `TIME_TABLES[1]` (decoded) | `0x281f468` | 4 x 4 | `0x280522c` | `0x3000ec0` | 0x45ec ld (TIME_TABLES[mode])[id*4] |
| `TIME_TABLES[2]` (decoded) | `0x281f478` | 4 x 4 | `0x2805230` | `0x3000f00` | 0x45ec ld (TIME_TABLES[mode])[id*4] |
| `LAP_COUNTS[0]` (decoded) | `0x2805248` | 4 x 4 | `0x2805234` | `0x3000f40` | 0x45f0 ld (LAP_COUNTS[mode])[id*4] |
| `LAP_COUNTS[1]` (decoded) | `0x2805258` | 4 x 4 | `0x2805238` | `0x3000f80` | 0x45f0 ld (LAP_COUNTS[mode])[id*4] |
| `LAP_COUNTS[2]` (decoded) | `0x2805268` | 4 x 4 | `0x280523c` | `0x3000fc0` | 0x45f0 ld (LAP_COUNTS[mode])[id*4] |
| `CAR_COUNTS[0]` (decoded) | `0x2805278` | 4 x 4 | `0x2805240` | `0x3001000` | 0x45f4 ld (CAR_COUNTS[setting])[id*4] |
| `CAR_COUNTS[1]` (decoded) | `0x2805288` | 4 x 4 | `0x2805244` | `0x3001040` | 0x45f4 ld (CAR_COUNTS[setting])[id*4] |
| `CHECKPOINT_TABLES` (decoded) | `0x2823c10` | 4 x 4 | `0x462c`, `0x463c` | `0x3001080` | 0x4628, 0x4638 ld [id*4] -> 0x50143c |
| `GRID_TABLES` (decoded) | `0x2805298` | 4 x 4 | `0x464c` | `0x30010c0` | 0x4648 ld [id*4] -> 0x501420 |
| `SEGMENT_FLAGS` (decoded) | `0x28058bc` | 4 x 4 | `0x46c8` | `0x3001100` | 0x46c4 ld [id*4] -> 0x501450 |
| `HUD_MAP_LISTS` (decoded) | `0x28052dc` | 4 x 4 | `0x12ee4` | `0x3001140` | 0x12ee0 ld [id*4] |
| `HUD_MAP_PLACES` (decoded) | `0x28052fc` | 16 x 4 | `0x12ef4` | `0x3001180` | 0x12ef0 lda [(id << 2)*4] |
| `HUD_MAP_ZOOMS` (decoded) | `0x280534c` | 16 x 4 | `0x12ed0`, `0x130d4` | `0x3001280` | 0x12ecc, 0x130d0 ld [((id << 2) + level)*4] |
| `TIME_CORRECTIONS_SUB` | `0x280587c` | 4 x 4 | `0x13b2c` | `0x3001380` | 0x13b28 ld [id*4]: per difficulty, subtracted |
| `TIME_CORRECTIONS_ADD` | `0x280583c` | 4 x 4 | `0x13b5c` | `0x30013c0` | 0x13b58 ld [id*4]: per difficulty, added |
| `WORDS_280539C` | `0x280539c` | 4 x 4 | `0x13bb0` | `0x3001400` | 0x13bac ld [id*4]: times the laps (0x13bbc) |
| `TABLES_28053AC` | `0x28053ac` | 4 x 4 | `0x13be0` | `0x3001440` | 0x13bdc ld [id*4]: sound calls per lap |
| `TABLES_280538C` | `0x280538c` | 4 x 4 | `0x13c00` | `0x3001480` | 0x13bfc ld [id*4]: half-words per lap -> 0x501a02 |
| `TABLES_281399C` | `0x281399c` | 4 x 4 | `0x226dac` | `0x30014c0` | 0x226da8 ld [id*4] -> 0x501858 |
| `TABLES_281397C` | `0x281397c` | 4 x 4 | `0x2813978` | `0x3001500` | 0x226e00 ld (0x2813974[setting])[id*4] -> 0x501858 |
| `TABLES_281398C` | `0x281398c` | 4 x 4 | `0x2813974` | `0x3001540` | 0x226e00 ld (0x2813974[setting])[id*4] -> 0x501858 |

Not grown, with the reason:

| What | Why |
|---|---|
| `RANKING_DEFAULTS` `0x2390cc`, the time bases `0x23909c` | read only by the boot-time ranking reset `0x2270d0`, a loop over ids 0-2 (`mov 3,r15`; `0x22713c` indexes `0x23909c[mode * 4 + 3 - r15]`). Ids 3-15 read the guard's sink (`RANKING_TABLES[3..15]`) until `course.grow_rankings` (Phase 9) gives the reset sixteen-id copies (`RANK_HOME`) and each id its own table ("Rankings") |
| `0x230b28`, `0x2880004` | indexed by an offset a grown array gives (`OFFSETS_230B18`, `OFFSETS_231B34`); Beginner's offset for ids 3-15 |
| `0x230b9c` | the time-attack records of ids 1-2, guarded (above) |
| `GRID_LANES`, `BILLBOARD_REF` | shared, not indexed by the id: made per id by `widen_ids` (Phase 9: "A mirror tests as its original"), not by grow_ids |
| `CAMERA_SCRIPTS_SLOT0` (`0x6ca8`, Beginner's linked-cabinet scripts) | shared, not indexed by the id (linked mode) |
| SRAM `0x1d00140`, `0x1d0014c`, `0x1d00158`, `0x1d00164` (`0x229010`-`0x229078`) | four bookkeeping counter arrays in backup SRAM, `u32[3]` back to back, read-modify-written by the id (`ldos 0x501460`, `& 0xffff`): an id 3-15 writes the next array's words (id 3: `0x1d0014c[0]`, Beginner's) and up to `0x1d001a3`, inside the system area (`rankings_layout().used[0]`). Not ROM: SRAM is Phase 9's (the rankings and the signature); Review Focus 4 (Task 5) |
| the per-difficulty time tables, the checkpoint tables, the time-correction tables | reached through a grown per-id entry; ids 3-15 point at Beginner's |

**Literal id comparisons.** The program also branches on the id itself, so an id 3-15 takes the "other" path where
Beginner (id 0) takes its own: `0x3eac` (`cmpibe 0`), `0x63ec` (id 0 with a linked cabinet: `0x6ca8`), `0x7370`
(`cmpobe 0`), `0x7ad4` (`cmpobne 2`: Advanced's sound), `0xc2a4` / `0xc2d0` (`cmpi 0`: the rolling start's steps,
`0xc334`), `0x15d90` (`cmpi 0`: a lap threshold 3 instead of 1), `0x17420` (`cmpobne 2`: Advanced's visit order),
`0x19ae4`-`0x19af8` (ids 0, 1, 2, else `0x19ed4`), `0x224cf0` (`cmpibne 0`); and **id 3 itself** skips code at
`0x102e4` and `0x10b54` (`cmpobe 3`). The music id `0x501a98` is the course id unless a view button picks 0-3
(`0x33d8`, `0x3b9c`). None reads a table. Phase 9 widens them for the mirrors (`course.widen_ids`, "A mirror tests as
its original" below). Phase 8 decided them for Desert City (id 3): all kept except `0x102e4` and `0x10b54` (`cmpobe 3` becomes
`cmpobe 31`, so id 3 runs the player car's heading assist and event timers), `course.ID3_LITERALS`; id 3 thereby gets
Expert's standing-start countdown and lights (`0x3eac` kept; the clock starts at 1651). See "Desert City as id 3 and the
SRAM guard".

Decoders on a grown set: `course.id_table(rs, name)` gives the address the set's readers read (the new one when the
first site holds it, else the original), `course.id_count(rs)` the ids (3, or 16 grown), and every decoder follows
them (`course_table`, `cell_grid`, `actors`, `segment_flags`, checkpoints, `rolling_start`, `grid_base`,
`race_value_addresses`, `select_assets`, `hud_map`, `ranking_pointers`, `texture.sheet_source`, `tgp.decode_ground`,
`golden.decode_course`); `golden.Course.id_arrays` carries the addresses so `tables` writes a course's per-id words
where the set reads them. Data bounds follow the set's image (`tablebase.data_size`: 16 MB, 32 MB custom; encoders
check against the i960 window, `DATA_WINDOW`), as do the polygon ROM (`course.check_rom_object(oba, rs)`,
`geo` with `RomSet.poly_mask()`) and the char-block ranges (`tiles._rom_ranges`).

## Scenery placement

The course's visible geometry is placed two ways, and both are culled by one **16 x 16 grid of 128-unit cells**.
Decoder: `scenery(rs, entry)` (a `Placement` per object), `actors(rs, entry)`; encoders `encode_placement`,
`encode_actor`; helpers `cell_of`, `cell_box`, `check_rom_object`. The structures around the entries:
`cell_grid` / `encode_cell_grid` (the 256 list pointers), `object_list` / `encode_object_list` (entries, then the
4-byte terminator; lists are 20 bytes apart on the real set), `actor_table` / `encode_actor_table` (256 list
pointers, 0 = none), `encode_actor_list` (count, then records), `billboard_lists` / `encode_billboard_lists`,
`switch_lists` / `encode_switch_lists` (the three `lda` instructions). The runs and disassembly:
`docs/spike/2026-11-arcade-course-tables.md`, "Scenery placement".

### The cell grid

`cell = row << 4 | col` with `col = (round(x) + 1024) >> 7 & 15`, `row = (round(z) + 1024) >> 7 & 15` (`0x172a0`:
`cvtri`, so round to nearest; world positions outside -1024..1023 wrap). Cell `c` covers x from
`-1024 + 128 * (c & 15)` and z from `-1024 + 128 * (c >> 4)`, 128 units each way. Beginner's course (x -386...350,
z -237...229) spans columns 4-10 and rows 6-9.

A converted course must lie inside -1024..1023 on x and the arcade's z (the Dreamcast's z negated; `model.WORLD`,
`ground.WORLD`). The converter moves a course that does not: `build.LIFT` raises every y (Mermaid Lake, for the TGP's
choice), and **`build.SHIFT`** adds to every x of the data and the model before converting (`shifted_data`,
`shifted_model`; Rin Rin Rink -64.0: its model reached x 1073.1). What moves: the used lanes, the AI points, the
ground cells' corners and centres, the model's vertices and bounding-sphere centres (single precision as stored);
what is a segment or in a car's frame does not need to (checkpoints, the grid base and spawn records, the fly-by
keyframes), and the HUD map's place record is framed on the shifted road (`hud_frame`), so the map and its markers
draw where they did. The origin rule for run-time props (`build.origin_props`) measures from the Dreamcast's origin,
moved by the shift. `converted_course` refuses a course whose data or scenery is not shifted by `SHIFT[n]` (a mark on
what `shifted_data` / `shifted_model` made, `Scenery.shift`), as it refuses a course `LIFT` names unlifted.

`GRID_TABLES = 0x02805298` (data ROM): `u32[4]`, per course id the address of the course's grid, stored to `0x501420`
at the load. A grid is `u32[256]`, one **object-list pointer per cell**. An object list is 16-byte entries ended by an
`oba` of 0:

| Offset | Field | Meaning |
|---|---|---|
| `0x0` | `oba` | object address (geometry command `0x01`); every grid object is a polygon-ROM object |
| `0x4` | `tpa` | texture point address |
| `0x8` | `tha` | texture header address |
| `0xc` | `budget` (`Placement.raw[0]`) | added to the frame's polygon count at `0x5010e8`; the draw stops emitting objects once the count is past `0x5010f4` (5000). On Beginner it equals the object's record count for the 22 tiles and is 0 for the one-record stubs |

The draw routine `0x17cf8` sends each entry as command `0x01` with `obc` = the word at `0x5010f4` (`0x1388` = 5000,
set at `0x12f8`): `obc` is not stored per object, which is why Phase 1 saw 5000 on every object. Grid objects are in
world coordinates and drawn under the view matrix V.

All object lists live in one **object library**, data ROM `0x0283fbb4`-`0x0284f4eb` (the bounds `0x17acc` and
`0x17cf8` check a list pointer against); the grids are right after it (`0x0284fccc` / `0x028500cc` / `0x028504cc`).
On the real set every cell of every course holds exactly one object; the cells away from the track hold one-record
link-0 stubs (never drawn: the Phase 1 "stubs" are the empty cells the camera passes).

| | Beginner (0) | Expert (1) | Advanced (2) |
|---|---|---|---|
| grid | `0x0284fccc` | `0x028500cc` | `0x028504cc` |
| cells with a non-zero budget (tiles) | 22 | 95 | 58 |

### Actors

`ACTOR_TABLES = 0x00233bfc` (program window): `u32[4]`, per course id the address of `u32[256]` actor-list pointers
(0 = none), stored to `0x501444` at the load and walked by the spawner at `0x22002c` (up to the count at data
`0x02802f74`, 65). An actor list is a `u32` count, then 24-byte records:

| Offset | Type | Field (`Actor`) | Copied to the actor at |
|---|---|---|---|
| `0x00` | 3 x f32 | `position` x, y, z | `+0x1c`, `+0x20`, `+0x24` |
| `0x0c` | s16 | `angle` (rotation about y; `0x10000` = one turn, inferred) | `+0x2a` |
| `0x0e` | u16 | `ident` (per-handler index) | `+0x04` (byte) |
| `0x10` | u32 | `handler`: the init routine | `+0x0c` |
| `0x14` | u32 | `obj`: handler data | `+0x14` |

The init routine computes the actor's cell from its position (`0x172a0`, to `+0x2e`) and installs an update routine
that draws only when that cell is in the frame's visible set. `scenery` models the three handlers Beginner's
view-matrix objects need; the others are returned by `actors` with no placement:

| Handler | `obj` | Draws (`Placement.in_world`) | Beginner |
|---|---|---|---|
| `0x220118` static prop (update `0x220138`) | an object list | the list under V x T(position) x Ry(angle) (False) | 2 |
| `0x220a54` switch (update `0x220a74`) | unused | one of three lists held by `lda <list>,g0` at `0x220abc` / `0x220ac8` / `0x220ad4`, chosen by race state, under V (True) | 1 |
| `0x221e40` view-angle billboard (update `0x221e60`) | `u32[64]` object lists | the list at index `((-a) >> 5) & 63`, `a` the TGP angle (function `0x0f`, `0x10000` a turn: `atan2(dz, dx)` of the camera from the reference point `BILLBOARD_REF`, window `0x235bb4`, (210, 0, -380), 12 bytes before Beginner's array; traced in Phase 4), under V (True) | 1 |
| others (`0x2201bc` animated props via the per-course table at `0x236b40`, `0x221700`, ...) | | not modelled | 31 |

Expert's and Advanced's actors (63 and 32) all use other handlers.

**Shared empty actor tables** (Phase 10, `docs/spike/2026-11-courses-4-7.md`, "Program room"). The table's readers
only read it: `ACTOR_TABLES[id]` is read once, by the race's load (`0x46b4`, stored to `0x501444`), and that word once,
by the spawner (`0x220040`), which loads each of the 256 words once (`0x22004c ld (r13),r10`) at the race's load and
follows only the non-zero ones; it reads no further than the 256th word (traced: 256 reads, frame 1328, nothing past
the table, no write). So every course without actors names one table, `build.EMPTY_ACTORS` (program window
`0x0023ce98`, program ROM `build.EMPTY_AT` `0x3ce98`, the first 1,024 bytes of `build.PROGRAM_FREE`), reserved in
`romset.RESERVED` (owned by `golden.owned_spans` whatever its bytes, skipped by every allocator, refused by
`build._write_free`) and written once by the first course that names it (`build.empty_actors`): every conversion (it
spawns no actor) and every mirror of a course without actors (`mirror.mirrored_entry`: Desert City's, id 11). A
course with actors keeps a table and lists of its own in `PROGRAM_FREE`. Two ids naming one table decode, cover and
re-emit as one table each (`golden.coverage` lists the span per id; `owned_spans` merges them). `build.program_room(rs)`
counts the run's free bytes: 11,624 on a stock or grown set, 5,312 on Phase 9's final chain (the mirrors of the
originals' tables and lists: 1,896 / 2,588 / 1,828).

**How the gap is signalled.** `scenery(rs, entry)` raises `NotImplementedError` when the course has any actor of
an unmodelled handler (every real course does: 31 / 63 / 32 actors). The message names the discovery-log entry
("Scenery placement / Not found / not modelled" in `docs/spike/2026-11-arcade-course-tables.md`), the count and
the handler addresses. `scenery(rs, entry, allow_unmodelled=True)` returns the modelled placements only (the whole
grid plus the modelled actors' objects), and `unmodelled_actors(rs, entry)` lists exactly the actors it skips, so
a caller that takes the partial result knows what is missing.

### Culling (per frame)

`0x172e8` then `0x17a80` (called at `0x6380`), for a reference point 10 units ahead of the viewer along the heading
(`0x5014a8`):

1. The reference point's cell (`0x172a0`), and the 5 x 5 cells around it (offsets `16 * dr + dc`, dr, dc in -2..2),
   visited in a fixed order (program `0x17594`: centre, the 4 edge neighbours, the 4 diagonals, then the outer
   ring; Advanced uses `0x175ad` when the cell is 72 or 88).
2. Cells off the grid are dropped (border masks at `0x17374`-`0x173e8`); on Advanced only, a mask chosen by race
   progress drops more (`0x17688`, table `0x175e0[id]`; Beginner's and Expert's tables are empty).
3. A view test (`0x1770c`): per heading quadrant (`(heading + 0x2000) >> 14 & 3`) a table of test points (data ROM
   `0x0286f7e0`, `0x0286f880`, `0x0286f920`, `0x0286f9c0`; shared by all courses), offsets from the reference
   point's cell corner, each with a 25-bit mask of the cells it opens. A point whose offset, rotated by 45 degrees
   minus the heading (TGP functions `0x1d`/`0x1e` at `0x17918`), has both coordinates >= 0, i.e. lies in the
   90-degree wedge ahead, ORs its mask in. The centre cell is always visible.
   (v0.2, measured on the converted courses: where the road crosses a cell edge at about 45 degrees, the quadrant's
   wedge can drop the row of cells the camera, trailing the car, still sees, for a few frames; the road there is not
   drawn and the backdrop or the ground under it shows. `build.ROAD_COPY` files the road's near-flat polygons in the
   neighbouring cells too; docs/follow-ups/v0.2.md, "The road holes".)
4. Visible cells go to the bit set `0x501520` and the list at `0x5016c0` (count byte, then cell bytes); the first
   10 cells of the order also to the "near" sets `0x501500` / `0x501540` and the list `0x501600`.
5. `0x17a80` draws, for each cell of `0x5016c0`, the cell's object list (`0x501420[cell]`); actors test their own
   cell against `0x501520` (or `0x501540`, the animated props).

On Beginner at frames 2400-2459, 10-13 cells were visible per frame and the grid objects drawn were exactly those
cells' objects, in list order.

Decoder: `culling(rs)` (`Culling`: the four tables as `ViewPoint(dx, dz, mask)`, 12 per table at `VIEW_TEST_POINTS`,
the two visit orders, the block offsets `0x175c6`, `16 * dr + dc`). The tables are geometry in cell units: every point
lies a whole number of cells (multiples of 128, within -256..384) from the reference cell's low corner, and every mask
names cells of the 5 x 5 block; nothing names a cell index or a world extent, so no part of them is sized for one
course. **Confirmed with scenery (Phase 7, Desert City's lap):** the view test runs on every race frame (on the M3a
set with no scenery too: it is called unconditionally at `0x17400`) and reads the heading quadrant's table whatever
the cell; the reference cell is a cell of the road line on every frame (Desert City: the road's 31 cells, rows 3-12,
columns 5-10, all four quadrants). Only the 5 x 5 block round the reference cell is drawn; as the reference cell
was a road cell on that autoplay lap (measured, not derived: the camera follows the car, which can leave the road),
scenery more than two cells from every road cell was not drawn there (Desert City: cells `0x08` and `0x09`, 10
budget words). `docs/spike/2026-11-convert.md`, "Confirmed
with scenery".

**A view per course id (Phase 11, `course.view_ids`, `build.VIEW`, `build.viewed_set`).** The table load `0x17744`
(`ld 0x17908[r8*4],r8`, r8 the quadrant) is the only reader of `0x17908`. On a set with `view_ids` its displacement
names `VIEW_RAM` (`0x501b54`, u32[4], `romset.RESERVED_RAM`): the same opcode, so the same cycles. `widen_ids`'
`update` (after every program store of the id: boot, attract rotation, vote, linked copies) ends in a branch to `view`
(`VIEW_CODE` `0x1fd00`), which copies the id's four pointers from `VIEW_PTRS` (`0x1fc00`, u32[4] x 16 ids). Every id but
the ones `VIEW` names gets the shared four, so its race reads the tables it read. `VIEW` names Circuit Pixie and Rin
Rin Rink (5 and 6, both `build.VIEW_12`); their tables are the shared ones with each point's mask cut to the view (in
each quadrant only 15 cells can open: the reference cell's row or column and the two ahead; `VIEW_12` keeps 12, that
row, the next and the two far corners, left-right symmetric as `build.view_tables` requires;
`docs/spike/2026-11-courses-4-7.md`, "Rin Rin Rink" and "Circuit Pixie"). `viewed_set` writes 640 bytes per course in
`ID_DATA_FREE` (id 5's, then id 6's: the same bytes twice). They are not in `golden.owned_spans`, so `viewed_set` is a
chain's last step (nothing allocates after it). `course.view_tables(rs, k)` gives an id's four pointers. Sets built
without it (every frozen set) keep their bytes.

### Camera scripts

`0x63d8` (the race's start) picks the script `CAMERA_SCRIPTS[id]` (program `0x6c98`, `u32[4]`: Beginner `0x02806b78`,
Expert `0x02807274`, Advanced `0x028071a8`, id 3 none) or, on slot 0 when the byte `0x540027` is past 1, the variant
`CAMERA_SCRIPTS_SLOT0[8 - that byte]` (program `0x6ca8`: `0x0280701c`, `0x0280701c`, `0x02806e90`, `0x02806e90`).
`0x65b4` plays it, one tick a frame (`0x5014a0`), between the two keyframes round the tick; `0x6a10` places each
point. A keyframe is 64 bytes (`course.camera_script`, `Keyframe`):

| Offset | Type | Field | Meaning |
|---|---|---|---|
| `0x00` | s32 | `time` | ticks from the script's start; `-1` (CAMERA_JUMP): the script goes on at the word at `+0x08` |
| `0x04` | u32 | `skip` | the keyframe is skipped when its bit `0x501752` is set (0 on every keyframe of the three courses) |
| `0x08` | s32 | `position_car` | the car whose frame the position is in (`T(car +0x1c) R(car +0x28)`); `-1` (CAMERA_ABSOLUTE): a world position |
| `0x0c` | 3 x f32 | `position` | the camera |
| `0x18` | s32 | `target_car` | as `position_car`, for the target |
| `0x1c` | 3 x f32 | `target` | the point looked at |
| `0x28` | f32 | `roll` | to `0x50173c` (named after its use) |
| `0x2c` | f32 | `fov` | times `0x230b94`, to `0x501744` (named after its use) |
| `0x30` | 4 x u32 | `rest` | a value to `0x501740`, a byte to `0x501746`, a hook (`+0x38`), padding |

Slot 0's three scripts are six keyframes on car 0 (the player) for position and target, the fly-by from 14 units over
the car (ticks 0-410, 0-280 or 0-310) down to 4 behind it, held to tick 3600, then a jump back to the start: the
fly-by again every 3600 frames. Expert's and Advanced's run 6,336 ticks and jump to a shared loop at `0x02808360`
whose first two keyframes are absolute ((10, 8, 145) and (5, 4, 145), targeting car 4); Expert's first keyframe is
absolute too ((400, 80, -70)). The scripts are shared program and data, not course tables: a course converted into
slot 0 is filmed round the player's car wherever its road is (confirmed on Desert City, Phase 7); one converted into
Expert's or Advanced's slot would be filmed from those worlds' places for part of the loop.

`Placement` fields: `oba`, `obc` (always 5000), `tpa`, `tha`, `raw` (`(budget,)`), `cell` (the grid cell, or the
actor's), `entry` (data-ROM offset of the 16-byte entry), `in_world`, `actor` (`None` for grid objects).
`check_rom_object(oba)` returns the polygon-ROM word index `oba & 0x7fffff` and refuses RAM 1 (bit 24), RAM 0 (bit 23
clear), bits 25-31 and an index >= `0x400000`. `encode_placement` refuses those, an `obc` other than 5000 (it is not
stored) and fields past 32 bits; `encode_actor` refuses a position that is not three f32, an angle outside s16 and an
ident outside u16.

## Road line

The record table (`CourseEntry.record_table`) is the road line: a closed loop of cross-sections, one per **segment**,
about 8 units apart. Decoder `road_line(rs, entry)` (a `RoadPoint` per record), encoders `encode_road_point`,
`encode_road_line` (the whole table); `centreline(points)` gives one centre point per segment. The runs and
disassembly: `docs/spike/2026-11-arcade-course-tables.md`, "Road and AI lines".

A car's segment is the `u16` at car `+0x12c` (cars are `0x300`-byte structs from work RAM `0x505180`; the player is
the first). It indexes the road line and every AI line alike, which is why the line table's counts equal the road
line's `n + 1`.

Header: `u32 n`, `u32 8` (`LANES`, the lane slots per record), then `n + 1` records of `ROAD_POINT_SIZE` = 164 bytes:

| Offset | Type | `RoadPoint` field | Meaning |
|---|---|---|---|
| `0x0` | u8 | `lane_lo` | first used lane; `0x140f0` clamps the car's lane to `lane_lo..lane_hi` |
| `0x1` | u8 | `lane_hi` | last used lane |
| `0x2` | s16 | `curve` | signed bend, 0 on the straights (Beginner: 90 of 196 records); `0x15848` tests its sign; unit not known |
| `0x4` | 8 x 20 bytes | `lanes` | `LANES` lane points, `Lane` below; lanes outside `lane_lo..lane_hi` are all zero on the real set |

Lane point (`Lane`, 20 bytes):

| Offset | Type | Field | Meaning |
|---|---|---|---|
| `0x00` | 3 x f32 | `position` | x, y, z |
| `0x0c` | s16 | `pitch` | to car `+0x28` |
| `0x0e` | s16 | `heading` | `heading << 2` is the yaw stored to car `+0x2a` (`0x10000` per turn). The direction of travel in (x, z) is (-sin, cos) of that angle: checked against the centreline on all three courses (median dot product 1.0). Every lane of a record has the same heading |
| `0x10` | s16 | `roll` | to car `+0x2c` |
| `0x12` | u16 | `pad` | 0 on the real set; no reader found |

The lanes are `LANE_UNITS` = 3 units apart (2.92-3.09 on the real set). The car's lateral position is the `s16` at car
`+0x142`, in lanes as 8.8 fixed point (`0x000`-`0x700`). `0x140f0` places a car from it: lanes `pos >> 8` and
`(pos >> 8) + 1`, clamped to the used range, blended by the table at data `0x02871c80[pos & 0xff]` (`frac / 255`), and
hands the car the pitch, heading and roll of the second, unblended (`0x14334`-`0x14340`); `0x14018` goes the
other way, from the car's offset to lane 4 rotated by the heading, times -85.0 (256 / 3.01), plus `0x400`. Lane
numbers grow towards the left of the direction of travel, as seen in the top view of `out/spike/arcade/topdown.png`
(+x right, +z up).

`RoadPoint.centre` is the midpoint of lanes `lane_lo` and `lane_hi` (the lanes are evenly spaced, so it is also their
mean); `RoadPoint.width` is the distance between them. That is the span the lateral coordinate covers, not the road
from kerb to kerb: on Beginner the player car at the outer wall sits at lane 0.3, outside lane 1.

| | Beginner (0) | Expert (1) | Advanced (2) |
|---|---|---|---|
| road points (`n + 1`) | 196 | 991 | 487 |
| centreline lap (closed, XZ) | 1,555 | 7,670 | 3,783 |
| spacing (min / median / max) | 7.25 / 8.0 / 9.14 | 4.54 / 8.0 / 10.26 | 5.57 / 8.0 / 8.54 |
| lane ranges | 1-6 (all) | 1-6 (566), 0-7 (353), 2-5 (34), others | 0-7 (286), 1-6 (201) |
| `width` | 15.0-15.3 | 9.0-21.0 | 15.0-21.4 |

Example (Beginner record 0): lanes 1-6, curve 0, lane 1 at (-93.1, 0, -101.6), heading `0xeaaa` (yaw `-0x5558`,
travel towards +x, -z).

`road_line` raises `CourseError` naming the address when the header's second word is not 8, the records run past data
ROM or past the `u16` segment index, or a record's lane range is not `lo <= hi` inside 0-7. `encode_road_point`
refuses a lane range like that, a curve or angle outside s16, a pad outside u16, a lane count other than 8, and a
position that is not three finite single-precision floats.

### The ground under the road (for M1b)

The road line is not what the cars stand on. The car's height and its **step** (car `+0x30`) come from a ground
query, `0x11c28`, run for every car every frame:

1. The cell of (x, z) on a 128 x 128 grid of 16-unit cells from -1024 (`0x11bec`: `cvtri`, + 1024, `>> 4`, `& 0x7f`;
   cell = row * 128 + col).
2. The grid's `u16` for that cell, from the 32 KB block at `*block_ref_a` (`0x501428`): 626 / 2,315 / 1,288 non-zero
   cells on Beginner / Expert / Advanced.
3. TGP function `0x36` through the function port (`st 0x3636` at `0x11c4c`) with `0x17ffc`, y, the grid value, x and z;
   the i960 then polls buffer RAM `0x91fff0` until the TGP writes the result: a height (`0x91fff4`, or none: the code
   then uses -0.1) and a polygon index (`0x91fff8`).
4. The attribute byte of that polygon, from the block at `*block_ref_b` (`0x50142c`), `& 15`: the surface kind. Kind 1
   is road; the polygon index is then the car's step.

The `*block_ref_b` block has one byte per ground polygon: 1,184 / 7,239 / 2,986 on Beginner / Expert / Advanced,
padded with `0xcd` to 4 bytes (1,184 / 7,240 / 2,988 to the next step table). Kind 1 (road) is exactly the first
`steps` polygons (503 / 1,064 / 818), the other kinds are 2, 5, 8, 10 and 13. The step table then maps each road
polygon to its segment (low `u16`, to car `+0x12c` at `0x13c60` / `0x10f80`) and flags (high `u16`, ORed with the
per-segment `u16` at `0x501450[segment]`, to car `+0x13c`). The polygons and the lists the grid values address are in
the TGP's data ROM (`copro_data`, slot = course id): `docs/formats/model2-tgp-ground.md`, decoder and encoder
`dtools/arcade/tgp.py` (`decode_ground`, `encode_ground`).

### Per-segment flags

`SEGMENT_FLAGS = 0x028058bc` (data ROM): `u32[4]`, per course id the address of one `u16` per road segment (`n + 1`),
stored to `0x501450` by the course load (`0x46cc` reads the word). Decoder `course.segment_flags(rs, entry)`, encoder
`course.encode_segment_flags(flags)`; `golden.Course.segment_flags` is (table address, flags). The step code
(`0x10f7c`, `0x13c7c`) reads entry `segment` (`ldos`, the car's segment from its step) and ORs it with the step's
flags into car `+0x13c` (the previous value to `+0x13e`), so the bits share the step flags' space ("Pit"). The runs
and disassembly: `docs/spike/2026-11-convert.md`, "Per-segment flags".

| | Beginner (0) | Expert (1) | Advanced (2) |
|---|---|---|---|
| table | `0x028058cc` | `0x02805a54` | `0x02806212` |
| values | 196, all 0 | 991: `0x400` on segments 190-240, `0x500` on 208-218 | 487, all 0 |

The tables are back to back; the id-3 stub's pointer repeats Beginner's. The readers of car `+0x13c`:

| Bit | Name | Readers, meaning (inferred) |
|---|---|---|
| 0 (`0x01`) | `STEP_ROAD` | `0xe18c`, `0x162c4`, `0x16504`: the AI resyncs a car to its line ("Pit") |
| 7 (`0x80`) | `STEP_PIT_BOX` | `0xdf9c`: the pit stop ("Pit") |
| 8 (`0x100`) | `SEGMENT_TUNNEL` | a tunnel (inferred). `0x7a08` (from `0x74ec`) switches a sound effect on entering it (`0x1a238` with `0x03c003c0`, `0x1fff0c0c`, on Advanced only) and off on leaving it, and weighs a term of the car's speed four times in it (`0x7a7c`); `0x7b88` lights the car from the course's tunnel light zones instead of the default at `0x232db8`. Expert's segments 208-218 (its zone range 209-217), Advanced's step flags (`0x101`, segments 174-232) |
| 10 (`0x400`) | `SEGMENT_AI_FREE` | `0x1129c`: when set, the AI car skips `0x13ccc`, its lane reaction to the player (moving in front, making room): the AI keeps its line (inferred). Expert's segments 190-240 |

The tunnel light zones are a per-id record in the program window: `0x230742[id]` a zone count (byte; 20 / 0 / 20 /
20), `0x23074a + 20 * id` five pointers (segment starts, segment ends, the tunnel's range, a zone bit mask, a light
position per zone, world coordinates). Beginner, Advanced and the stub share one record (Advanced's tunnel, segments
175-232); Expert's has no zone (range 209-217). `0x7b1c` walks the zone index by the car's segment; `0x7b88` uses it
only while bit 8 is set, so on a course without the bit (Beginner, a converted course) the record is never used.

No Dreamcast file holds the per-segment flags (`docs/spike/2026-11-convert.md`, "Per-segment flags"): the converter
writes `n + 1` zeros (`dtools.convert.tables.segment_flags`), exact for Beginner and Advanced.

## AI lines

The line table (`CourseEntry.line_table`) holds `AI_LINE_COUNT` = 8 AI lines, one point per segment. Decoder
`ai_lines(rs, entry)` (8 lists of `AiPoint`, in pair order), encoders `encode_ai_point`, `encode_ai_line` (one list as
stored), `encode_line_table(offset, lines)` (the whole table, pairs then lists, as on the real set); `ai_path(points)`
gives the positions. The runs and disassembly: `docs/spike/2026-11-arcade-course-tables.md`, "Road and AI lines".

Header: eight (`u32` pointer, `u32` count) pairs, count = the road line's `n + 1`. On the real set the lists follow the
pairs, each `count + 8` points long: the points, then its first `AI_LINE_WRAP` = 8 points again, so that the look-ahead
past the last point needs no wrap. Point (`AiPoint`, `AI_POINT_SIZE` = 28 bytes):

| Offset | Type | Field | Meaning |
|---|---|---|---|
| `0x00` | 3 x f32 | `position` | x, y, z: the car's target (`0x112d8`) |
| `0x0c` | f32 | `speed` | multiplied with the car's speed setting (`+0x23c`) into `+0x44` at `0x114a0`; read as a speed factor (inferred); 0.59-1.42 on the real set |
| `0x10` | s16 | `pitch` | to car `+0x28` |
| `0x12` | s16 | `heading` | as `Lane.heading` (`<< 2` at `0x112f4`, to `+0x2a`) |
| `0x14` | s16 | `roll` | to car `+0x2c` |
| `0x16` | s16 | `steer` | shifted by the car's speed and added to `+0x48` at `0x11648`; read as a steering bias (inferred) |
| `0x18` | s16 | `lane` | lateral position on the road line in lanes, 8.8 (to `+0x142` at `0x112f8`); `0x000`-`0x700` |
| `0x1a` | u16 | `step` | the point's step (road polygon), to car `+0x30` at `0x111f8`; below the step table's `steps` |

How a car follows its line: `0xe8a0` (car set-up) takes the pair indexed by the word at `+0x20` of the car's spawn
record, stores the list pointer to car `+0x134` and the count to `+0x130`, and the start segment
`(start + 0x230e48[id]) mod n` to `+0x12c` and `+0x138` (as a byte offset, `* 28`). Each frame `0x1117c` reads the
point at `+0x138`, asks the TGP for the angle from the car to it (function `0x0f`) and moves to the next point
(wrapping at `count * 28`) once that angle, less the point's heading, is more than a quarter turn (the point is
behind); `0x1127c` then loads the fields of the point after it into the car.

| | Beginner (0) | Expert (1) | Advanced (2) |
|---|---|---|---|
| points per line | 196 | 991 | 487 |
| line 0 lap (closed, XZ) | 1,563 | 7,548 | 3,798 |
| all eight laps | 1,533-1,564 | 7,501-7,566 | 3,717-3,798 |
| farthest point from the centreline (same segment, XZ) | 14.2 | 19.6 | 17.9 |
| identical lists | 0 = 4 = 6, 1 = 3 | 1 = 3, 5 = 7 | none |
| cars with a line pointer at frame 2500-2600 (player included) | 40 | 30 | 20 |
| cars per line 0-7 | 4, 4, 2, 2, 2, 2, 12, 12 | 4, 4, 2, 2, 2, 2, 7, 7 | 3, 3, 1, 1, 1, 1, 5, 5 |

`ai_lines` raises `CourseError` naming the address when a count is not the road line's `n + 1`, a list (with its
wrap) is outside data ROM, or a list's wrap points differ from its first ones. `encode_ai_point` refuses a position or
speed that is not finite single precision, a pitch, heading, roll, steer or lane outside s16 and a step outside u16;
`encode_ai_line` a list shorter than the wrap or of 0x10000 points and more; `encode_line_table` a count of lines other than 8, lines of different
lengths and a table that does not fit in data ROM.

## Checkpoints

The time extensions. Decoder `checkpoints(rs, entry, mode=0, difficulty=1)` (a `Checkpoint` per checkpoint, the lap
line last), `time_table(...)`, `laps(...)`; encoders `encode_checkpoint_table`, `encode_time_table`. The runs:
`docs/spike/2026-11-arcade-course-tables.md`, "Checkpoints, start grid, pit".

The race timer is the `u32` at work RAM `0x005010d0`, in **ticks**: it counts down one per frame (`0x4010`-`0x4028`; not while `0x501a8c` is non-zero, which the pit stop sets) and
the HUD shows `ticks >> 6` (`0x3b04`; `TICKS_PER_SECOND` = 64, inferred from the shift). The course load sets it to
the time table's word 0 (`0x3ad8`); each time extension adds a later word (`0x13b70`-`0x13b7c`).

`CHECKPOINT_TABLES = 0x02823c10` (data ROM): `u32[4]`, per course id the address of its checkpoint table, stored to
`0x50143c` (`0x4628`). A checkpoint table is `u32 count`, then `count` entries of (`u16 segment`, `u16 raw`):

| Course (id) | Table | Count | Segments |
|---|---|---|---|
| Beginner (0) | `0x02823c20` | 0 | (the lap line only) |
| Expert (1) | `0x02823c24` | 5 | 150, 257, 404, 539, 730 |
| Advanced (2) | `0x02823c3c` | 2 | 142, 275 |

`raw` is 0 on the real set (no reader found). How a checkpoint is passed (`0x1386c`, the player car's segment step):

1. When the player's segment (car `+0x12c`) changes, `0x1395c` compares it with every entry; on a match it calls
   `0x13a10` with bit number `count - j` for entry `j` (entry 0 gets the highest bit).
2. The lap line is the wrap from segment `n` to segment 0 (`0x13898`); it counts (bit 0) only when the car's progress
   `+0x132` reached `n` without a jump (`0x13924`: it follows the segment in steps under 7, or resyncs at `n >> 1` or
   `(n >> 1) + 4`), and only when the bit set at `0x5014b0` equals `0x5014b4` = `(1 << (count + 1)) - 1`, i.e. every
   checkpoint of the lap was passed. It then counts the lap (`0x5014b9`, car `+0x6f`) and clears the set for the next
   lap; on the final lap it is not cleared, so the last lap line gives no extension.
3. A bit not yet set is set and the car is flagged (`+0x0` bit 21); next frame `0x13aac` adds the time table word
   number `++car[+0x71]` (the count of passes in the race) to the timer. Two corrections can apply (`0x13b14`-
   `0x13b6c`): minus `0x280587c[id][difficulty]` when `0x501a80` is 1, plus `0x280583c[id][difficulty]` when the
   player car's `+0x0` bit 28 is set (both not traced; neither applied in the runs).

So the time table is indexed by pass number, not by checkpoint: pass `k = lap * (count + 1) + j + 1` for checkpoint `j`
(the lap line is `j = count`) on lap `lap` (0-based). `Checkpoint.extensions[lap]` is that word. The lap line's last
one (`k = laps * (count + 1)`, the final lap) is a **phantom**: the race ends at the final lap line and never adds it.
`time_table` still reads it and the golden re-emit keeps it, so every checkpoint has one value per lap;
`golden.check_course` requires the word to exist (its value is free).

`TIME_TABLES = 0x02805228`: `u32[3]` per game mode (work RAM `0x5fe5e0`; 0 in every run), each the address of `u32[4]`
per course id, each the address of `u32[4]` per difficulty (`0x5fe59c`; 1 in every run), each the address of a time
table: `u32` words, the initial time then one extension per pass. `time_table` returns the `1 + laps * (count + 1)`
words the race reads (the stored tables are longer: the 80-lap mode 2 reads further).
`LAP_COUNTS = 0x02805234`: `u32[3]` per mode, each the address of `u32[4]` laps per id (to the byte `0x5010ec`).

| Mode 0, difficulty 1 | Beginner (0) | Expert (1) | Advanced (2) |
|---|---|---|---|
| time table | `0x02820b50` | `0x02821ca8` | `0x0281f9f8` |
| laps | 8 | 2 | 4 |
| initial time (ticks) | 3,200 (50 s) | 3,840 (60 s) | 3,840 (60 s) |
| first extensions | 1,201, 841 (laps 1, 2) | 560 (checkpoint 150, lap 1) | 1,083, 723 (142, 275, lap 1) |

`Checkpoint.position` is the road line's centre at the checkpoint's segment. `checkpoints` raises `CourseError` naming
the address for a pointer outside data ROM, a count past `MAX_CHECKPOINTS` (30: the bit set is one `u32`) or a
segment past the road line's `n`; `time_table` for a mode outside 0-2 or a difficulty outside 0-3; `laps` for a count
outside 1-127 (`MAX_LAPS`: laps go to a byte that `0xdfa4` reads signed against the car's signed lap byte `+0x70`).
`encode_checkpoint_table` refuses more than 30 checkpoints and a segment or `raw` outside `u16`;
`encode_time_table` an empty table and a word that is not a `u32`; `encode_laps` a count outside 1-127.

**Writing these values** (for a converter). `race_value_addresses(rs, entry, mode, difficulty)` gives the address of
every per-course value. The fixed-size values are rewritten in place and the pointer arrays above them are not
re-emitted: the lap count (`u32`, `encode_laps`), the car count (`u32`, `encode_car_count`), the rolling-start word
(`s16`, `encode_rolling_start`) and the grid base (`s16`, `encode_grid_base`). The checkpoint and time tables vary in
size (the checkpoint tables are packed: Beginner's 4-byte table is followed by Expert's), so a new table is written
where it fits and the one pointer word that holds its address is rewritten (`encode_table_pointer`, refusing an
address outside data ROM): `CHECKPOINT_TABLES[id]` for the checkpoint table, the per-difficulty word of `TIME_TABLES`
for the time table. The upper levels of `TIME_TABLES`, `LAP_COUNTS` and `CAR_COUNTS` (per mode, per cabinet setting)
are fixed.

## Start grid

Decoder `start_grid(rs, entry, table=1)` (a `GridSlot` per car, the player first), `car_count`, `grid_base`,
`grid_lanes`, `lane_position`; encoders `encode_spawn_record`, `encode_grid_base`, `encode_grid_lanes`. The runs:
`docs/spike/2026-11-arcade-course-tables.md`, "Checkpoints, start grid, pit".

Every car is set up by `0xe898` from a **spawn record**, one per car index (the car's ident, car `+0x4`; the player
is 0). `SPAWN_TABLES = (0x0281e3d4, 0x0281ec14)` (data ROM): two tables of `SPAWN_RECORDS` = 44 records of 48 bytes;
`0xeb00` takes the first when work RAM `0x5fe588` is 0, the second otherwise (MAME's default settings give 1, so every
run used the second). The two differ only in the object words of records 0-9, 17 and 19.

| Offset | Type | `GridSlot` field | Use |
|---|---|---|---|
| `0x00` | 8 x u32 | `objects` | object lists in the object library (the car's models; one 16-byte entry and the terminator each on the real set, all inside `0x0283fbb4`-`0x0284a293`); `0xec64` copies words 5, 4, 3, 2, 7 to car `+0x74`-`+0x88` |
| `0x20` | u32 | `line` | the AI line: `0xe8a0` takes pair `line` of the line table (car `+0x134`, `+0x130`) |
| `0x24` | u32 | `start` | segments ahead of the course's grid base; to car `+0x12e` and `+0x132` as a `u16` |
| `0x28` | f32 | `speed` | the car's speed setting, scaled by `0x0286f68c` / `0x0286f6fc` into `+0x38`, `+0x44`, `+0x23c` |
| `0x2c` | u32 | `kind` | read as a byte, to car `+0x23a`; 0, 1 or 2 |

The course decides only three things: the car count, the base segment and the start type.

- `CAR_COUNTS = 0x02805240`: `u32[2]` by cabinet setting (`0x4570`-`0x45d0`: from `0x5fe588` and `0x540027`; 0 in the
  runs), each the address of `u32[4]` per id: **40 / 30 / 20** cars on Beginner / Expert / Advanced (entry 1 gives
  16 each). The car set-up reads records 0 to count - 1 (traced: the second table's pages up to `0x0281f300` /
  `0x0281f100` / `0x0281ef00`).
- `GRID_BASES = 0x00230e48` (program window): `s16[4]` per id, **100 / 984 / 482**. A car's segment is
  `(start + base) mod n` (`0xe8d4`-`0xe8f4`, `remo` by `0x501434` = `n`). The starts run 0, then pairs 4, 8, ..., 28 (4 apart) and
  36, 44, ..., 76 (8 apart), then 84, 88, 92, 92, 96, 100, ..., 144: the player is at the back of the field, the others
  in rows of two ahead of it (single cars from record 27 on).
- `GRID_LANES = 0x00231b30` (program window): `u16[2]` = `0x180`, `0x580` (lanes 1.5 and 5.5, 12 units apart). The lane
  (car `+0x142`) is `GRID_LANES[(car & 1) ^ 1]` (`0xe980`): even cars, the player among them, on 5.5, odd cars on 1.5.
- `ROLLING_STARTS = 0x028050e8`: `s16[4]` per id, to `0x501850`; 0 on Beginner (rolling), -1 on Expert and Advanced
  (standing). `0x501850` is the rolling start's countdown state (`0xc334` steps it 0, 1, 2, 3 at segments `n - 29`,
  `n - 19`, `n - 9`), so `rolling_start` refuses any other stored value. On a rolling start the player's handler (`0xd800`) and the opponents' (`0xdafc`, for a car whose
  segment is `>= n >> 1`, i.e. its lap counter `+0x70` was set to -1 at `0xe90c`) swap the side:
  `GRID_LANES[car & 1]`, and step the car one segment on. Beginner's cars 31-39 (starts 96-128, past segment 0 after
  the mod) keep the standing sides.

`lane_position(road_point, lane)` is `0x140f0`'s placement: lanes `lane >> 8` and the next, each clamped to
`lane_lo..lane_hi`, blended by data `0x02871c80[lane & 0xff]` (`frac / 255` as f32). The set-up then drops the car on
the ground (`0x11c28`), so `GridSlot.position` has the road line's y, not the ground's.

| | Beginner (0) | Expert (1) | Advanced (2) |
|---|---|---|---|
| cars | 40 | 30 | 20 |
| grid base / `n` | 100 / 195 | 984 / 990 | 482 / 486 |
| player | segment 100, lane `0x180` (rolling) | segment 984, 6 behind the lap line | segment 482, 4 behind |
| last car | car 39, segment 33 (start 128) | car 29, segment 86 (start 92) | car 19, segment 48 (start 52) |

Checked against the RAM dumps at frame 1335 (standing starts, the cars have not moved): every car's x and z equal
`GridSlot.position` on Expert and Advanced (y within 0.01). The field straddles the lap line on all three courses.

`start_grid` raises `CourseError` for a table index other than 0 or 1, a car count outside 1-44 (the car pool: the boot sets
`0x501080` to 44 at `0x12e0`, the ident loop `0x3d1c` reads 44-byte rows at `0x3d4c`, the spawn tables hold 44
records), a spawn record whose line is past 7 and an object pointer outside the object library (the list draw
`0x17cf8` skips a list outside `0x0283fbb4`-`0x0284f4eb`). `encode_spawn_record` refuses objects that are not 8 pointers
into the object library, a line outside 0-7, a start outside `u16`
(the game keeps it in a `u16`), a speed that is not finite single precision and a kind outside `u8` (the game reads
one byte); `encode_grid_base` a base outside 0-`0x7fff`; `encode_car_count` a count outside 1-44;
`encode_rolling_start` anything but a bool (written as 0 / -1).

## Pit

Every course has a pit lane. It is not a separate table: it is in the **step table's flags**
(`CourseEntry.step_table`; "The ground under the road"), one `u16` per road polygon, which the step code ORs into car
`+0x13c`. Decoder `step_table(rs, entry)` (a `Step` per polygon), `pit(rs, entry)` (`Pit` or `None`); encoders
`encode_step_table(n, steps)`, `encode_pit(n, steps, pit)` (writes a pit into the flags). The runs:
`docs/spike/2026-11-arcade-course-tables.md`, "Checkpoints, start grid, pit".

| Flag | Name | Meaning (reader) |
|---|---|---|
| `0x01` | `STEP_ROAD` | the road: the AI resyncs a car to its line here (`0xe18c`, `0x162c4`, `0x16504`) |
| `0x02` | `STEP_PIT` | a pit polygon (lane or box) |
| `0x40` | `STEP_PIT_LANE` | the pit lane (`0x42` together with `STEP_PIT`; no direct reader found) |
| `0x80` | `STEP_PIT_BOX` | the pit box: `0xdf9c` starts the pit stop (handler `0xe288`) when the car is on it, not on the final lap, just arrived (`+0x13c` != `+0x13e`) and heading the right way |
| `0x100` | `SEGMENT_TUNNEL` | Advanced only, segments 174-232 (59 polygons): its tunnel (inferred; `0x7a08`, `0x7b88`; "Per-segment flags", where Expert's is) |

On the real set the flags are exactly `0x01`, `0x42`, `0x82` (and `0x101` on Advanced): the pit polygons are their
own polygons beside the road's, not road polygons with an extra bit. `Pit.entry_segment` and `exit_segment` are the
first and last segment of the pit polygons' run in the direction of travel, where the lane leaves and rejoins the road:

| | Beginner (0) | Expert (1) | Advanced (2) |
|---|---|---|---|
| pit-lane polygons (`0x42`) | 96 (indices 147-501) | 72 (190-262) | 96 (51-807) |
| box polygon (`0x82`) | 152, segment 5 | 209, segment 136 | 800, segment 479 |
| entry - exit segment | 175 - 75 (across the lap line) | 117 - 189 | 465 - 50 (across the lap line) |

Cross-check (Dreamcast): `CS_DATA0/1/2.BIN` hold exactly as many road cells as the arcade has steps (503, 1,064, 818)
and the same `n`; taking the cells by the arcade polygon index, the pit-flagged ones lie 13-125 / 8-37 / 33-48 units
from the road line's centre at their segment (rough transform, x and -z scaled by 1.0286), apart from the road's (on
Beginner and Expert closest at the entry and exit). The pit lane's geometry itself is on the TGP side (M1b).

### The pit stop and its stalls (v0.1.2)

Traced in MAME (`spike/pit_drive.py`, the pit probe `dtools/mame/lua/pit_log.lua`; `docs/follow-ups/v0.1.2.md`, "Pit
stops"). The player car's handler (car `+0xc`, normally `0xdc64`) calls the trigger `0xdf9c`, which sets the handler to
`0xe288` when the car stands on a `STEP_PIT_BOX` step (`+0x13c` bit 7), not on the final lap (`+0x70` < the laps at
`0x5010ec` - 1), just arrived (`+0x13e` != `+0x13c`), heading within `[-0x17ff, +0x800]` of the box polygon's direction
(`+0x48` - `+0x1c6` + 0x3fff in `0x2800..0x47ff`; `+0x1c6` is the ground polygon's direction word) and with `+0xd2`,
`+0x9c`, `+0x9d`, `+0xd8`, `+0xf0` all 0. `0xe288` sets `0x501a8c` to 640 (the race timer stops while it is non-zero),
the car's heading to the box's direction and the handler to the stop `0xe384`. Each frame the stop steers (`0xe744`)
towards waypoint `+0xcc` of the car's stall and sets the car's speed from it; the waypoint is passed once it lies more
than a quarter turn from the box's direction; at the seventh (`+0xcc` = 7, `0xe3b4`) the stop ends. Once the car stands (inferred)
(`0x501220`'s `+0x18` = 0), `0x501a8c` counts down one a frame ("PIT WORK!"); at 0 the handler is `0xdc64` again
(`0xe598`). Beginner, measured: the box at frame 3239, the car standing at (213.09, -59.01) for 640 frames, back on
the road at 4489.

The stalls are the per-id array `CAR_TABLES_230E50` (`course.pit_stalls`, `course.encode_pit_stalls`): per course id a
pointer to `u32[8]`, one stall per car ident (`+0x4 & 7`, the cabinets of a link; a single cabinet is 0), each 7
waypoints of 16 bytes (x, z, speed f32, a 0 word; the speed times the f32 at `0x286f68c` goes to `+0x44`). Beginner's
stall 0: in at speed 150 along z = -62 from x 92.36, 100 at 162.36, 60 at 182.36, over to z = -59 at 202.36 (60),
20 at 212.36, 0 at 242.36, then (300, -59, 0), the lane's end; the eight stalls about 21 units apart along the lane.
Id 3 (the stub) repeats Advanced's. grow_ids gives ids 3-15 Beginner's, so until v0.1.2 a converted course's stop
aimed at waypoints far behind the car and ended 7 frames after the box. The Dreamcast keeps the same lists for its
eight courses (`dc.exe.pit_stalls`; "Executable tables" in docs/formats/dreamcast-disc.md); `build.pit_stalls` writes
them for ids 3-7 (`release.RELEASE_PIT_IDS`).

**The angle arithmetic (M9 stage 4).** The trigger's window (`0xdfd0`-`0xdffb`), the stop's steering (`0xe79c`: the
turn to the waypoint, its angle less the car's yaw, held to -0x80..0x80 and added to the yaw) and its waypoint-passed
test (`0xe7fc`: the waypoint's angle less the box's direction, plus 0x3fff, outside 0..0x7fff) each subtract two
angles loaded sign-extended and compare the 32-bit result: none wraps. A box facing near +-0x8000 (Dinosaur Canyon and
Desert City in reverse, -32768) starts the stop only for a yaw on one side of the wrap, and a waypoint dead ahead on
the other side counts as passed at once. `course.patch_pit_heading` keeps each difference's low 16 bits (shlo 16,
then shro 16 or shri 16; MAME's i960 has no `extract`), each site in its stock length (the compare-and-branch
instructions free the room); `release_set` applies it with the reverses. The arcade's boxes face -16383, -4, 21587
and the like: the trigger's narrow window never meets the wrap on them. The passed test's half-turn window and the
steering's difference do on a box beyond 0x4000: on National Park Speedway (21587, in practice the only course
affected) stock counted a waypoint about 61-90 degrees to one side of the box's direction (angles -32768 to -27565)
as passed and turned the long way towards it; the patch gives the quarter-turn answer and the short turn.

`pit` raises `CourseError` when there is not exactly one box polygon or the pit polygons' segments are not one run of
the loop. `encode_pit` refuses a polygon outside the steps, listed twice or flagged `STEP_ROAD`, another polygon already
flagged as pit, and a pit whose entry, exit or box segment differs from what its polygons give; `encode_step_table` a
segment past `n` and flags outside `u16`.

## Tile layers and uploads (select screen and backdrop)

The Circuit Select panels and the race horizon are drawn by the System 24 tilemap chip (MAME `segaic24.cpp`,
BSD-3-Clause, copyright the MAME team), not by polygons. The decoders and encoders of this section are in
`dtools/arcade/tiles.py` (they raise `CourseError` like `course.py`; both take it from `dtools/arcade/tablebase.py`):

| What | Where | Layout |
|---|---|---|
| Tile RAM | `0x01000000-0x0100ffff` (mirrors `+0x10000`, `+0x100000`) | four 64 x 64 layers of `u16` tiles, layer `k` at byte `0x2000 * k`, row-major (`0x80` bytes per row); scroll registers at word `0x5000` |
| Char RAM | `0x01080000-0x010fffff` (mirror `+0x100000`) | `0x4000` chars of 8 x 8 pixels, 4 bpp, 32 bytes each; a row is two `u16`, pixel 0 the high nibble of the first |
| Palette | `0x01800000`, entries `0x000-0xfff` for the tiles | `x bbbbb ggggg rrrrr`, shown through the colour-translation table's entry `0x40` of each component row (`model2.cpp palette_w`) |

A tile `v` draws char `v & 0x3fff` with colour `(v >> 7) & 0xff` (pens `colour * 16 + pixel`, pen 0 transparent):
the fields overlap, so **a char's palette follows from its index** (chars `0x3480-0x34ff` use colour `0x69`, or
`0xe9` with bit 14 set). Bit 15 is the priority: clear, the tile is drawn behind the polygons (layers 3 and 2 opaque,
then 1 and 0); set, in front. `decode_chars` / `encode_chars` convert char bytes to and from `(n, 8, 8)` pixels.

**Tile map** (`TileMap`, `tile_map`, `encode_tile_map`), the blob the blit `0x19304(src, dst)` copies to tile RAM:

| Offset | Type | Field |
|---|---|---|
| `0x0` | `s16` | `base`, added to every entry |
| `0x2` | `s16` | `wide`: 0 = entries are `s8`, else `s16` |
| `0x4` | `u32` | `rows` |
| `0x8` | `u32` | `cols` |
| `0xc` | | `rows * cols` entries, row-major; tile = `(entry + base) & 0xffff`, rows `0x80` bytes apart in tile RAM |

**Upload set** (`UploadSet`, `upload_set`, `encode_upload_set`): two words, a char list and a palette list (a
course's set has eight more, the backdrop segments). Applied by `0x5cc4` (screens) and the course load (`0x1de40`):

- char list (`char_list`, `encode_char_list`; copied by `0x1dd70`): (`src`, `dst`) word pairs ended by a zero
  `src`. `src` is a **char block** (`encode_char_block`): `u32 count`, then `count` chars, copied a half-word at a
  time to char RAM `dst`. Blocks are in data ROM, except `BOOT_CHAR_LIST`'s (program window).
- palette list (`palette_list`, `encode_palette_list`; `0x1dde0`): (`dst`, `u32 count`, `count` `u16` entries)
  runs ended by a zero `dst`; the copy moves `count / 2` words.

Later uploads overwrite earlier ones and char RAM is never cleared, so chars no upload of the current screen covers
keep whatever an earlier screen left. The encoders refuse a source outside ROM, a destination that is not
char-aligned in char RAM or outside palette RAM, an odd palette count and an entry past `u16`.

## Select screen

Decoder `select_assets(rs)` (`SelectAssets`): per game mode the three panel records, per course id (`CourseSelect`)
its panel, tile position, the 18 tile maps it can show, the chars of its **name** (map rows 1-5 inside the frame)
and **label** (rows 6-8: "SHORT x 8 LAPS" and the like), its 3D course map, plus the wheel zones and, for every char
on the screen, the data-ROM address of its 32 bytes (`char_source`). Encoders `encode_select_panel`,
`encode_tile_map`, `encode_map_object`, `encode_wheel_zones` and the upload encoders above. The runs:
`docs/spike/2026-11-arcade-course-tables.md`, "Select screen".

How a panel is drawn, each frame of Circuit Select (`0x2263b0` -> `0x2269b4(table, 3)`):

1. the record table is picked by the game mode, the half-word `0x5fe5e0`: `SELECT_TABLES` = `0x0023862c` (mode 0),
   `0x002386e0` (1), `0x00238790` (other). The tables differ only in the label (the lap count of that mode);
2. for panel `i` (0-2) the record at `table + 44 * i` gives the tile position `(x, y)` and three (normal, blink)
   map pairs; the pair is picked by the half-word `0x5fe59a` (`variant`), the blink map when `i` is the cursor and
   `0x540058 & 7 > 4`;
3. `0x19304(map, 0x01000000 + 2 * x + 0x80 * y)` copies the 10 x 18 map into **tile layer 0**;
4. separately, `0x226438` draws each panel's **course map as a 3D object** (`SELECT_MAP_OBJECTS` + 32 * panel:
   position `(x, y, z)` f32, then an object-library list: Beginner `0x0284490c`, Advanced `0x028448e4`, Expert
   `0x028448f8`, at x = -5, 0, 5), and `0x226244` eases the 3D window centre (`0x50280c`) towards the record's
   cursor words, where the red frame object `0x02843818` is drawn (read, not traced).

Panel record (`SelectPanel`, 44 bytes, program window):

| Word | Field | Real values (panels 0 / 1 / 2) |
|---|---|---|
| 0, 1 | `cursor` (x, y), packed as 12 bits each | (95, 396) / (247, 396) / (399, 396) |
| 2 | `course` id (`0x226c74` reads the low byte: the **wheel-position -> course table**) | 0 / 2 / 1 |
| 3, 4 | `tile` (x, y) in layer 0 | (3, 9) / (22, 9) / (41, 9) |
| 5-10 | maps: (normal, blink) for variants 0, 1, 2 (data ROM) | e.g. mode 0 panel 0: `0x2832164`, `0x28322d8`, `0x2831e7c`, `0x2831ff0`, `0x2832164`, `0x28322d8` |

By course id (the same records, in id order):

| | Beginner (0) | Expert (1) | Advanced (2) |
|---|---|---|---|
| panel | 0 | 2 | 1 |
| mode 0 record | `0x0023862c` | `0x00238684` | `0x00238658` |
| tile (x, y) in layer 0 | (3, 9) | (41, 9) | (22, 9) |
| map-object list (`SELECT_MAP_OBJECTS` `0x0023889c` + 32 * panel) | `0x0284490c` | `0x028448f8` | `0x028448e4` |

Each table has a fourth record (course id 3, the stub, a copy of panel 2) that `0x2269b4` never draws. Variants 0
and 2 are the same maps; variant 1 holds the Japanese names (初級 / 中級 / 上級), shown by the MAME runs (COUNTRY
JPN); what sets `0x5fe59a` was not traced. `SELECT_PANEL_OF_COURSE` (`0x00238f0c`, `u32[4]` = 0, 2, 1, 3; encoder
`encode_panel_of_course`, which refuses an array that does not invert the records) is the inverse, course id -> panel (`0x22653c` uses it to animate the chosen course's map after the step).

Wheel -> panel (`WheelZones`, code at `0x225f74-0x225fb8`): `w = (s32 [0x501050] + s32 [0x50105c]) >> 1` (the
steering value of this frame and the last; paddle `0x50` gives -73); panel 0 when `w <= 0`, 1 when `w <= 108`, else
2; the panel is stored to `0x501468` and `0x005043de`, the course id to `0x00501460`. The thresholds are instruction
literals: `cmpibge 0,r3` at `WHEEL_LO_AT` (`0x00225f90`, 5 bits) and `shlo 2,27,r5` at `WHEEL_HI_AT` (`0x00225f98`,
`27 << 2`); `encode_wheel_zones` rewrites them and refuses what those literals cannot hold and a `hi` past
`0x7fffffff` (`shlo` keeps 32 bits and the compare at `0x225f9c` is signed); `wheel_zones` wraps such a word the
way the register does.

Panel map (10 x 18, `base` -32768, wide): row 0, row 9 and columns 0, 17 are the frame; rows 1-5 the name; rows 6-8
the label. All tiles have bit 15 set (in front of the 3D maps). The maps' chars are `0x3480-0x39c5` (colours
`0x69-0x73`). The three mode tables' maps differ only in the label rows.

Chars and palette on Circuit Select, in the order applied (all `UploadSet`s are in `SelectAssets.uploads`):

| Step | Set | Chars | Palette |
|---|---|---|---|
| boot `0x5d38` | `BOOT_CHAR_LIST` `0x0023a270` | 16 chars | |
| boot `0x5cb8` | `[0x02600000]` = `0x02073f9c` | 30 blocks from `0x1080000` (fonts, HUD, ...) | entries `0x000-0x46f`, `0x800-0x84f` |
| boot `0x5d38` | `[0x02700000]` = `0x02700008` | 13 blocks | `0x0b0-0x0cf`, `0x230-0x29f`, `0x300-0x3af`, `0x470-0x5ff` |
| select `0x2d5c` | `[0x02600004]` = `0x020740b8` | 7 blocks to `0x10e0000-0x10f503f` | `0x600-0x75f` (and an empty run) |
| select `0x5d68` | `[0x02700004]` = `0x02700030` | 5 blocks from `0x10e0000`, over the previous | `0x600-0x65f`, `0x690-0x73f` |

Then `0x2824(0xfd02)` sets pen 0 of every palette (the blue backdrop colour). With these, `select_assets`'
`char_source` reproduces every uploaded char of MAME's char RAM at frame 750 (0 of 13,006 differ) and the panel
palettes `0x690-0x6bf`.

**For M4** (sixteen panels in a wheel-scrolled row, generated panels, MIRROR marking; the row, its names and the vote are built in Phase 8,
the generated panels, the MIRROR marking and the mirrored maps in Phase 9: below, "The sixteen-entry select"):

- panel count: `0x2263e0` passes 3 to `0x2269b4` (`mov 3,g1`); `0x226438` loops 3 (`mov 3,r5`) over the 3D maps;
  the cursor is clamped to 0-2 by the zone code (`0x225f8c-0x225fa0`), whose 5-bit literals cannot express sixteen
  zones: that code has to be replaced, not re-parameterised. The course id per panel is word 2 of the record
  (`0x226c74`), and `0x238f0c` must stay its inverse. Records are 44 bytes in three per-mode tables of four, laid out
  back to back in the program window (`0x0023862c-0x0023883f`), so a 16-panel table needs new space and new
  `lda` displacements at `0x2263b0`, `0x2263c8`, `0x2263d8`, `0x225448`, `0x22564c`-`0x22579c` and `0x226c7c`.
- drawing a panel = a 10 x 18 tile map in data ROM + the chars it names in char RAM (uploaded by a select upload
  set) + a 16-colour palette per 128-char block (the char index decides the palette). Scrolling sixteen panels
  through the visible 62 columns means re-blitting maps at new tile positions (`0x19304` takes any destination) or
  scrolling layer 0 (its scroll registers are at tile-RAM word `0x5000`).
- name and label graphics: rows 1-5 and 6-8 of each map; `CourseSelect.name` / `.label` list their chars and
  `char_source` where each char's 32 bytes are. A MIRROR mark fits in the label rows (their chars change by mode
  already) or in the name rows.
- free space on Circuit Select: tile layer 1 is empty; layer 0 uses rows 1-6 (title, timer), 9-18 (panels), 28-34
  and 37-44 (prompts). Chars no boot or select set uploads: 3,378, the largest run `0x3a82-0x3fff` (1,406 chars,
  colours `0x75-0x7f`, or `0xf5-0xff` with tile bit 14); others `0x3416-0x347f` (106), `0x3109-0x317f` (119),
  `0x2fb2-0x2fff` (78). Tile palettes no upload writes: `0x76-0x7f`, `0x85-0xff`. Data ROM has the 667,360-byte
  free run at `0x65d120` for maps, char blocks and upload lists.

### The sixteen-entry select (patched sets)

The rows of 16 (Phases 8-10; since Phase 11 the frozen rows `M4A_ENTRIES`, `M4B_ENTRIES`, `M5A_ENTRIES`: the shipped
row is the row of eight, below; the table here is Phase 9's row).
`dtools/arcade/select.py`: `patch_select(rs, entries)` on a grown set (`course.grow_ids`), `select_row(rs)` reads it
back, `layout_of(rs)` gives its data addresses; the evidence is in `docs/spike/2026-11-sixteen.md`, "Select".
`course.select_assets` still reads the original panels of ids 0-2 on a patched set and gives ids 3-15 Beginner's
panel (the row's own records are `select_row`'s).

**Entries** (`ENTRIES`, row order; `Entry(course, name, label, mirror, live, difficulty)`):

| Entry | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| course | Beginner | Advanced | Expert | Desert City | National Park Speedway | Circuit Pixie | Rin Rin Rink | Mermaid Lake | the eight, mirrored, same order | | | | | | | |
| course id | 0 | 2 | 1 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 9 | 11 | 12 | 13 | 14 | 15 |
| live | yes | yes | yes | yes (Phase 8 Task 5) | no | no | no | no | yes (Phase 9 Task 3) | yes | yes | yes (Phase 9 Task 4) | no | no | no | no |
| difficulty | BEGINNER | ADVANCED | EXPERT | CIRCUIT EDITION | | | | | BEGINNER | ADVANCED | EXPERT | CIRCUIT EDITION | | | | |
| window | none | none | none | all | all | all | all | all | label | label | label | all | all | all | all | all |

The difficulty labels (decision 11) are the originals' own words and, for Desert City, the Dreamcast's: its course
select shows no difficulty label, only the origin line FROM CIRCUIT EDITION (Phase 5), drawn without FROM (the band
holds 17 glyphs); a mirror takes its original's (`check_entries` refuses another). M5's courses carry none yet: their
band shows the label, COMING SOON. `M4A_ENTRIES` (Phase 8's row, the M4a playtest set) is `ENTRIES` with the mirrors
locked.

**The wheel and the row.** The wheel value `w` (`0x57cc`: paddle `p`, `d = p - 128`, `sign(d) * min(max(|d| - 5,
0), 75) * 255 / 150`, so -127..127) gives entry `E` = the number of `BANDS` (-112, -96, ..., 112, steps of 16)
below `w` (`autoplay.lua`'s `DAYTONA_ENTRY` holds `select.paddle_for_entry(E)`). The three panel positions show
entries `first`, `first + 1`, `first + 2` with `first = clamp(E - 1, 0, 13)`: the chosen entry in the middle,
except entry 0 (left) and 15 (right). The red cursor frame eases to the chosen entry's position (its record's
cursor words).

**The idle wheel** (Phase 9 Task 5; the controller's ruling, approved by Pascal: an idle player races Beginner, as on
the stock select). The select's state 0 puts the cursor on `IDLE_ENTRY` (entry 0) and clears `FOLLOW_RAM`; the wheel
routine keeps `E = 0` while `FOLLOW_RAM` is 0 and the wheel value is in the centre band, `|w| <= 16`
(`select.IDLE_BAND`, symmetric: paddle 114-142, `cmpibl 16,r3` and `addi 16,r3,r6; cmpibg 0,r6`); the first frame it
is outside sets `FOLLOW_RAM`, and from then on `E` follows the bands (w in (0, 16] is entry 8). A timeout on a locked
entry races entry 0 (Phase 8 walked left to the nearest live entry). `select.follow` is the rule as a model. Entries 7
and 8, whose middle paddles (124, 138) lie in the band, are reached by leaving it and coming back (`autoplay.lua`'s
`LEAD`: entry 6's or entry 9's paddle for 20 frames first).

**RAM.**

| Address | Was | Is |
|---|---|---|
| `0x501468` (s16) | the cursor panel 0-2 | the entry `E` 0-15 (Mission Select still reuses it for its two panels) |
| `0x5043de` (`0x5e(g13)`) | the cursor panel | `E` (kept through Mission Select) |
| `0x501460` | the course id | the chosen entry's course id (a locked one's too, while the cursor is on it) |
| `0x501b03`, `0x501b04 + 2 * id` | free | the course vote's count and its sixteen slots (were `0x501473`, `0x501474 + 2 * id`, four slots) |
| `0x501b24` | free | `first`, written by the draw every frame of the select, read by the 3D maps |
| `0x501b25` | free | `FOLLOW_RAM`: 0 from state 0 until the wheel leaves the centre band (Phase 9 Task 5) |

`0x501ae4`-`0x501fff` is touched by nothing in a whole run but the boot's clear of work RAM (trace session 1);
`select.SELECT_RAM` reserves `0x501b00`-`0x501b3f`.

Code `0x1e780`-`0x1f7ff` (`0x1f800`-`0x1fdff` is `widen_ids`' and `0x1fe00`-`0x1ffff` `grow_rankings`', Phase 9),
data `0x03f00000`-`0x03f1ffff` and RAM `0x501b00`-`0x501b3f` are registered in
`romset.RESERVED` / `RESERVED_RAM`: left out of `CUSTOM_FREE`, refused by `tgp.write_free` and the converter.
Entries 0-2 must be the original panels 0-2 (`check_entries`): the code that still indexes by panel and the linked
path's copy at `0x5718` then see what they did.

**Data** (`select.DATA_AT` = `0x03f00000`, custom sets; a directory of seventeen words, then): three per-mode
tables of sixteen 44-byte records (`SelectPanel`: entry `k`'s cursor and tile words are those of the position it
is chosen in, word 2 its course id, its six maps the original panel's for the originals and their mirrors, else the
placeholder panel: Beginner's frame round a transparent interior, char `0x3b00`); `MAPS`, 42 records of 32 bytes for
`3 * first + position` (the position's place, the entry's object list or 0: an empty map); `u8 live[16]`,
`u32 voices[16]` (the step's voice: the originals' `0xae215e`, `0xae215f`, `0xae2160`, entries 3-15 the table's
fourth word `0xae216c`), `u32 names[16]` (each entry's layer-1 map), `u32 slot_dst[3]` (layer-0 tile address of
each position), `u8 generated[16]` (each entry's window: 0 none, 1 label rows, 2 all, 3 name rows since Phase 12;
Phase 8 stored 0 or 1, a name map or none), the positions' mask bits (two `u32` each), `s32 bands[15]`, the placeholder maps (normal,
blink), the blank layer-1 map (all `0x0020`), the window maps, the extended upload set, the char block.
`SELECT_PANEL_OF_COURSE` (grown by `grow_ids`) holds course id -> entry, the inverse of the records' ids.

**Panels** (Phase 9 Task 5; the spec's open question "Layout of the generated panels and the MIRROR marking";
discovery log `docs/spike/2026-11-mirrors-live.md`, "Panels"). Every entry's panel position shows its record's
layer-0 map and, where its window covers it, its layer-1 window map (10 x 18, the panel's tile position; tiles with
bit 15, in front of the polygons; bit 14 for a locked entry). Layer 1 is the "window" tilemap: it shows only where
the window mask (tile RAM `0x100c000`, four `u16` per screen row, one bit per 8-pixel column, bit 15 leftmost; layer 0
is hidden there) is set, so the draw sets, every frame, rows 80-119 (panel rows 1-5) to the interiors of the positions
whose window is `all` and rows 120-143 (rows 6-8) to those whose window is `all` or `label`.

| Window (`window_of`) | Entries | Layer 0 (the records' maps) | Layer 1 (rows) |
|---|---|---|---|
| none | the originals, 0-2 | the original panel's maps (every mode and variant) | blank |
| label | the mirrors of the originals, 8-10 | the original's maps (its art, blink and language variants) | 6-8: the original's label rows with the band's lettering erased and MIRROR over it |
| all | 3-7, 11-15 | the placeholder: Beginner's frame (normal, blink) round char `0x3b00` (transparent) | 1-8: the generated interior |

The generated interior (`select.panel_pixels`, 64 x 128 pixels) is the original panels' (Beginner's, mode 0, variant
0: the three panels' interiors have the same pens in the same places, only their palettes differ) with its lettering
erased: the name box (pixel rows 4-40, columns 4-123 of the interior; a four-pixel border round it) filled with its
fill pen, the band (rows 45-63) with its band pen where the letters were (pens 5 and 7). Then, in the arcade's own 8 x
8 font (chars `0x20`-`0x5f` of the boot upload, the ranking and test screens' lettering: pen 1 the letter, pen 15 its
outline), **on the char grid**: each glyph fills one char column (8 pixels), each font row two pixel rows tall for a
tall line (a glyph is then two chars, its top and bottom halves) or one for a short one, so a letter's chars are shared
by every panel of its scheme (`select.UPLOAD_MAX`, below):

- the name in the box, centred in char columns 1-14 (the box's fill spans them whole), wrapped by words into at most
  two lines of 14 glyphs, tall: one line on char rows 2-3, two on rows 1-2 and 3-4; letters in pen 5 (white),
  outlines in pen 6;
- the band's lines (`band_lines`) on char rows 6-7, centred in columns 0-15 (16 glyphs), outlines in pen 8 (black):
  an entry that is not a mirror its difficulty (or its label when it has none), one tall line, pen 5; a mirror its
  difficulty (or label) on row 6 and MIRROR on row 7, short lines, MIRROR in pen 7 (the orange of the original lap
  counts); a mirror of an original only MIRROR, tall, pen 7, over its original's band (its art names the difficulty).

Colours: the original panels' palettes (Beginner `0x69`, Advanced `0x6e`, Expert `0x71`, read from the set's upload
sets) for the originals' mirrors, and for every other course `NEW_SCHEME`, Beginner's with its blues turned purple
(red raised to three quarters of the blue); a locked entry's tiles take the grey copy (six tenths of the luma of
Beginner's). The chars (`select.graphics`) are cut 8 x 8 from those pixels and packed into ten blocks of 128 from char
`0x3b00` (`0x3b00` itself the transparent hole), one palette per block: block `k`'s chars take colour `0x76 + k`
(`0xf6 + k` with bit 14). Every entry's chars go into blocks of its scheme, live or locked, so the chars and palettes
do not depend on the live flags (Phase 8's row uploads what the full row does); identical chars of a block are stored
once. The second select upload set (`[0x02700004]`) is replaced by a copy with one more char block per used block
(each block's used chars, from its first char) and two palette runs (`0x760`-`0x7ff`, the ten blocks' palettes;
`0xf60`-`0xfff`, the grey ten times). `ENTRIES` uploads 127 chars in four blocks (85 purple, 14 for each original's
MIRROR band). **The upload is capped** (`select.UPLOAD_MAX`, 288 chars, and `UPLOAD_RUNS_MAX`, 4 runs: 288 chars were
proven in one run by Phase 8 and in four runs by Phase 9 Task 5's fix round): the select's
set-up copies the chars a half-word at a time in the frame the select opens, and 664 chars made that set-up one
video frame longer, which put the game's frame counters (`0x5010b0`, `0x540058`) one behind for the rest of the
session and made Beginner's race part from the stock set's at frame 1339 (`docs/spike/2026-11-mirrors-live.md`,
"Panels"). The draw runs in states
0-3 and 6 only; state 4's handlers (`0x2259ac`, `0x2259c0`) call `exit`, which runs it once with the state at 4 (mask
clear, layer 1 blank) before the wipe of state 5. A mirror of an original's MIRROR band does not blink with the
cursor (layer 1 is drawn over the blink map's band); its name box does.

**3D maps** (`MAPS`, `select.map_records`). Each original's map object is a textured quad, x and z -2..2, y 0, drawn
as one to four quads at the same corners with move records between (Beginner one, Advanced three, Expert four: the
course map is the textures). A mirror of an original's record names a list of its own whose objects are its
original's mirrored (`mirror.mirror_object`, local: z -> -z, the texture flipped in the quad's frame; the same in any
geometry mode, so the select's mode, not traced, does not matter). The objects go to the last free bytes of the
custom polygon area, the lists to the last of the list home `tgp.DATA_FREE` (the draw routines skip a list above the
object library: `patch_select` applies `course.LIBRARY_BOUND_PATCH`), so a converter's or a mirror's first fit,
before the select or after it, places the same bytes; `golden.owned_spans` owns them (`select.map_spans`). Every other
non-original entry keeps an empty record (no map); Desert City's from its road was not built in the time box.

**Code** (`select.code`, at `0x1e800` in the program's free run `0x1e780`-`0x1f7ff`; `b`-entered routines run in
their caller's register frame and branch back):

| Site | Was | Now |
|---|---|---|
| `0x225f8c` | the zones `mov 0,r4` ... `addo 1,r4,r4` and the stores | `b wheel`: `E` from the bands (0 until the wheel leaves the centre band), stores `0x501468`, `0x5e(g13)`, `0x501460` (`0x226c74`), back to `0x225fc0` (the link packet) |
| `0x226004` | `ld 0x501068,r3` (the step) | `b step`: the step counts only on a live entry |
| `0x225ffc` | `call 0x2261c0` (timer end: voice) | `call timeout`: a locked entry gives way to entry 0 (`IDLE_ENTRY`), then the voice |
| `0x22563c`, `0x2256b4` | `ldos 0x501468,r3` (state 0, single and linked: the cursor record of the select's first frame) | `b idle`: `E = 0` to `0x501468` and `0x5e(g13)`, `FOLLOW_RAM` cleared, back to `0x225644` / `0x2256bc` |
| `0x2263e0`, `0x2263e4` | `mov 3,g1`, `call 0x2269b4` | `mov 16,g1`, `call draw`: the three positions' panel maps (layer 0) and window maps (layer 1, blank from state 4), stores `first`, writes the mask rows (80-119 and 120-143 by window) |
| `0x2259ac`, `0x2259c0` | `call 0x2259c8` (state 4: on to the wipe) | `call exit`: the draw at state 4 (names and mask cleared), then `0x2259c8` |
| `0x226464` | `cmpi r6,r4` (the lit map) | `b fix_a`: the lit entry minus `first` is a position |
| `0x226540` | `shlo 3,g0,g9`; `lda 0x23889c[g9*4],g9` | `b maprec`: `g9 = MAPS + 32 * (3 * first + position)` |
| `0x226580` | `cmpobne r4,r5,0x2266f4` (the chosen map) | `b fix_b`: the chosen course's entry minus `first` |
| `0x2261c8` | `ld 0x238ebc[r3*4],g0` | the voices by entry |
| `0x225448`, `0x22564c`, `0x2256c4`, `0x2256f4`, `0x22572c`, `0x225760`, `0x22579c` | `lda 0x23862c(...)` | mode 0's sixteen records (cursor words) |
| `0x2263b0`, `0x2263c8`, `0x2263d8` | `lda` of the three mode tables | the three sixteen-record tables |
| `0x226c7c` | `ldob 0x238634(g0),g0` | mode 0's table + 8 (the course byte) |
| `0x3048`, `0x3070`, `0x3078`, `0x3080`, `0x32b4`, `0x3470` | vote addresses `0x501473`-`0x501478` | `0x501b03`-`0x501b08` |
| `0x3050`, `0x32bc`, `0x32d0` | `mov 9,r5`; `mov 4,g9`; `subi g9,4,g2` (Task 3) | `lda 33,r5`; `mov 16,g9`; `subi g9,16,g2` |

`course.LIBRARY_BOUND_AT` (the object library's lower bound, `course.LIBRARY_BOUND_PATCH`) is applied when a mirror of
an original is in the row (its map list lives in `tgp.DATA_FREE`). The original record tables (`0x23862c`-`0x23883f`),
map objects and wheel-zone literals stay in place, unread except by the decoders. The wheel-zone words (`WHEEL_LO_AT`, `WHEEL_HI_AT`) are dead code on a patched set.

### The eight-entry select (patched sets, Phase 11)

Mirror mode is dropped (Pascal, 2026-11-11), so the shipped select is one row of the eight courses: Phase 11's row
(`select.ENTRIES` from Phase 11 Task 2, frozen as `select.M5B_ENTRIES` by Phase 12 Task 1; the release row below is
`select.ENTRIES` since), `patch_select(rs, entries)` with a row of 8 (`select.ROW_LENGTHS` = (8, 16); a row of 16, the
frozen `M4A_ENTRIES`, `M4B_ENTRIES`, `M5A_ENTRIES`, writes the bytes it wrote before: the sixteen-entry select above).
A row of 8: course ids a permutation of 0-7, no mirror entry (`check_entries`). The shipped chain builds no mirror
(grow, guard, widen, convert 3, 4 and 7, `course.grow_rankings`, select): ids 8-15 hold no course and no entry
reaches them (free for reverse, M9: since M9 stage 3 the release chain builds the reverses there, reached by the Course Direction screen below). Since Phase 11 Tasks 3-4 the chain converts 3, 4, 7, 6 and 5, all eight entries
are live, and `build.viewed_set` (the per-course views, below) is its last step. Evidence:
`docs/spike/2026-11-courses-4-7.md`, "The row of eight".

| Entry | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| course | Beginner | Advanced | Expert | Desert City | National Park Speedway | Circuit Pixie | Rin Rin Rink | Mermaid Lake |
| course id | 0 | 2 | 1 | 3 | 4 | 5 | 6 | 7 |
| live | yes | yes | yes | yes | yes | yes (Task 4; Task 2: COMING SOON) | yes (Task 3; Task 2: COMING SOON) | yes |
| difficulty | BEGINNER | ADVANCED | EXPERT | CIRCUIT EDITION | CIRCUIT EDITION | | | |
| window | none | none | none | all | all | all | all | all |

**What follows the row's length n, and what stays sixteen.** Indexed by entry (n): the three mode tables of n records,
`MAPS` (3 x (n - 2) records), `live[n]`, `voices[n]`, `names[n]`, `generated[n]`, `bands[n - 1]`, the wheel loop's
count (`cmpobg n - 1,r4`) and the draw's row length (`0x2263e0 mov n,g1`: `first = clamp(E - 1, 0, n - 3)`, 0-5 on
the row of 8). Indexed by course id (sixteen, whatever n): `SELECT_PANEL_OF_COURSE` (on a row of 8 ids 8-15 name the
entry of id - 8; it is read only for the chosen course, never 8-15 there) and the vote (`0x3050 lda 33,r5`, `0x32bc
mov 16,g9`, `0x32d0 subi g9,16,g2`, sixteen slots in `VOTE_RAM`; the slots of 8-15 stay 0 and never win). The
directory keeps its seventeen words; `select.row_of` reads n from the mode tables' spacing (`ceil16(44 n)`: 704 bytes
for 16, 352 for 8). The code is the same instructions for both lengths (n reaches it as data).

**The wheel.** `select.bands_of(8)` = -96, -64, -32, 0, 32, 64, 96 (32 wide): entry 0 is paddle 0-66, entry 7 paddle
191-255, entry 3 (the centred wheel) paddle 105-133, the others nineteen paddles each. The idle wheel is unchanged
(`IDLE_BAND` 16: an idle player races Beginner). `autoplay.lua` with `DAYTONA_ROW=8`: `ENTRY_WHEEL_8` (33, 76, 95,
119, 143, 162, 181, 223) and `LEAD_8` {3: 2} (entry 3's paddle lies in the centre band).

**Panels and upload.** No mirror band, so no mirrored 3D map (no map objects or lists; `map_spans` empty) and only
the new courses' purple chars are uploaded (within `UPLOAD_MAX` / `UPLOAD_RUNS_MAX`).

### The release row: the courses' names (Phase 12)

The release set (`dtools/release.py` `release_set`) carries `select.ENTRIES`, the **release row**: Phase 11's row
(now the frozen `select.M5B_ENTRIES`) with every course under its own name and no difficulty or label line. Evidence:
`docs/spike/2026-11-release.md`, "Names on the select".

| Entry | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| name | THREE SEVEN SPEEDWAY | DINOSAUR CANYON | SEA-SIDE STREET GALAXY | DESERT CITY | NATIONAL PARK SPEEDWAY | CIRCUIT PIXIE | RIN RIN RINK | MERMAID LAKE |
| course id | 0 | 2 | 1 | 3 | 4 | 5 | 6 | 7 |
| window | name | name | name | none (own panel) | none | none | none | none |
| band | Sega's (layer 0) | Sega's | Sega's | MEDIUM x N LAPS (own, layer 0) | MEDIUM | SHORT | MEDIUM | MEDIUM |
| map | Sega's | Sega's | Sega's | drawn from the road | drawn | drawn | drawn | drawn |

(Task 2 gave entries 3-7 window `all` with an empty band and no map; Task 6 gave them their own panels and maps,
below.)

**The originals' window, `name` (`WINDOW_NAME` = 3).** An original (course id 0-2, not a mirror) with no difficulty
keeps its own art and records (the original panel's six maps for each mode) and gets a layer-1 window over rows 1-5
only, the name box: `panel_pixels` is the copied interior's rows 1-5 (the box, its lettering erased, 40 x 128) with
the name wrapped and placed as on a generated panel (THREE SEVEN / SPEEDWAY, DINOSAUR / CANYON, SEA-SIDE / STREET
GALAXY), drawn in the original panel's palette. Rows 6-8, the band, stay layer 0: "SHORT x 8 LAPS" and the like,
whose lap count changes with the game mode (8 / 20 / 80, 4 / 10 / 40, 2 / 5 / 20 for modes 0 / 1 / other), so no
per-entry layer-1 map could follow it. On the panels' own maps: rows 1-5 change with the variant (the language; the
difficulty word, BEGINNER or 初級, lies in interior pixel rows 6-38) and the blink only, rows 6-8 with the mode and the
blink only, the frame with the blink only. An original with a difficulty (every frozen row's) is `none`, as before.

**The draw's branch.** The window mask sets the name rows (80-119) for `all` and `name`, the label rows (120-143) for
`all` and `label`. `select.code(lay, name_window)` adds, only for a row holding a `name` entry, one instruction after
the position's two mask words are loaded: `cmpobe 3,r5,draw_name_rows`, which skips the label rows' `or`s and joins
the name rows' (4 bytes; every label after it moves by 4). Any other row writes the bytes it wrote (the frozen rows'
pins, `M5B_SHA1`). `select_row` checks the code for the row it reads (`name_window` from the `generated` bytes).

**The blink.** The cursor's blink map (`0x540058 & 7 > 4`) lights the whole panel (the box light blue, the band white);
under the name window the box stays in its normal colours while the frame and the band blink, as a generated
panel's interior does.

**Upload.** Task 2: 129 generated chars in four runs (31, 24, 29, 45: the originals' three palettes, then the new
courses' purple), inside `UPLOAD_MAX` 288 / `UPLOAD_RUNS_MAX` 4. Since Task 6: 210 (31, 24, 29, 126).

### The release row: own panels and maps (Phase 12 Tasks 6 and 7)

Entries 3-7 of the release row carry `Entry.own_panel`: a generated entry (not an original, not a mirror, no
difficulty or label) drawn as the originals are. Evidence: `docs/spike/2026-11-release.md`, "Maps and laps" and
"Painted maps".

**Records.** No window (`window_of` = `none`, the `generated` byte 0): the entry's records name its own maps, one pair
per game mode (normal, blink; the three variants the same pair), written after the window maps in the name maps'
area (`Layout.name_maps`; the directory is unchanged, so a row without an own panel keeps its bytes). Each map is
Beginner's frame (normal or blink, as the placeholder's) round rows 1-8 of the entry's own chars: rows 1-5 the name
in the box as on a generated panel, rows 6-8 the band (`select.band_pixels`) for that mode. The frame blinks with the
cursor, the interior does not (as a generated panel's). The panel wipes with layer 0 like an original.

**The band.** `<LENGTH> x <N> LAPS`, every part Sega's own glyphs (`PanelArt.bands`: the nine original bands, cut by
lettering column runs into word, times sign, digits in the accent pen, LAPS):

- LENGTH: the word of the original whose lap is nearest in length (`select.length_panel`, `road_length`: the road
  line's centres as a closed loop): SHORT (Three Seven Speedway, 1,555 world units), MEDIUM (Dinosaur Canyon, 3,820),
  LONG (Sea-Side Street Galaxy, 7,688). Desert City 3,460, National Park Speedway 2,986, Rin Rin Rink 2,990, Mermaid
  Lake 3,845: MEDIUM; Circuit Pixie 1,644: SHORT.
- N: `course.laps(set, course, mode)` of the set as built (8 / 20 / 80 on the release set: the slot's laps).
- Layout: on the length panel's band of a mode whose lap count has as many digits; the digits end where its digits
  end (a group Sega draws whole, the length panel's own first, else Sega's glyphs of that width one pixel apart:
  single digits are drawn wide, two narrow; a digit no band draws is refused); the times sign at the source's gap
  before them; LAPS where Sega puts it; the word where its panel puts it, moved left (`band_shift`, one place per
  length on the row) until it keeps `BAND_WORD_GAP` (3 px, Sega's narrowest) before the times sign: MEDIUM x 20 and
  x 80 need 2 px, so MEDIUM's three bands share its chars. With no shift and a lap count an original shows, the
  original's band pixel for pixel.

**The map.** The originals' maps are 128 x 128 four-bit textures in texture sheet 1 (the common sheet, uploaded at
boot; one colour per layer, 1-4 layers per map: Beginner colours 44 and 376, Advanced 44, 372, 373, Expert 44, 374,
363, 375), and sheet 1 has no free 128 x 128 room. An own panel's map is flat polygons instead
(`dtools/arcade/selmap.py` `course_map`), on the map quad's place, in drawing order:

- the ground: the quad (x and z -2..2) in the colour of the course's uncovered surroundings (`selmap.OUTSIDE`, the
  light shade of its main natural family; brown's is Advanced's sand);
- the painting (Task 7): the course's own scenery seen from straight above (every drawn polygon of the cell grid's
  placements, the highest surface winning, each in its texture's mean colour), each pixel classed into the nearest of
  the originals' map colours (`selmap.FAMILIES`: grass, olive, sand, brown, water in dark / mid / light, white and
  two greys), on a grid of 128 cells (one per texel; 64 or 32 when a map would pass `PAINT_MAX_DRAWN`, 600 drawn
  polygons), each colour's cells cut into y-monotone pieces, one strip of flat quads each (a move to its first row
  edge, then one quad per kept edge), the colours by area;
- a two-texel dark frame, the road's edge strip (white, 10 texels wide) and the road strip (grey, 6 texels) along the
  road line's centres (merged to at most 48 points, `build.outline_points`);
- the start stripe: a white quad across the road strip at road point 0 (the lap line), 2 texels along the road, as
  Advanced's map marks its lap line.

Fitted with the longer axis upright over 104 texels of 128 (the originals' turn: a road long in x has quad x = -z,
quad z = -x, as Beginner's; long in z quad x = x, quad z = -z, as Advanced's; on screen quad x runs right to left and z
downwards: the originals' texture points put (2, -2) at texel (0, 0)). The first polygon has z-sort min, every later
one "old value", so the map is one z bucket, in which a later polygon draws over an earlier one. Every quad winds as
the ground quad (single-sided, facing up). One flat header per part (`selmap.parts`: ground, each painted colour,
frame, edge, road, stripe; palette entries nearest each colour, `textures.flat_entry`), back to back, each part's last
polygon stepping to the next. Placement: objects at the last free bytes of the custom polygon area, headers at the
last of the custom textures area, lists (one entry: panel 0's map placement with the object and headers swapped in)
at the last of `tgp.DATA_FREE`; `select.map_spans` owns all three (headers only in the custom textures area). On the
release set the maps are 251-443 drawn polygons (282-545 records); with three on screen the select draws at most
1,079 polygons a frame (a race up to 3,488) and its frame counters keep step with the M5b set's.

### The Course Direction screen (M9 stage 3, patched sets)

`dirscreen.patch_direction` (after `patch_select` and `widen_ids`; the release chain's last step) adds select states
17-19 between Circuit Select and Mission Select. Code at 0x1ef00 (1,216 bytes, then the two tables of 20 words: the
stock 17 handlers but state 7's, then 17-19); the single and linked dispatchers' table words (0x225480, 0x2254b8)
repointed. Data from 0x03fc0000 (DATA_SPAN 0x40000; 122,368 bytes on the release set).

| State | Handler | Does |
|---|---|---|
| 7 | `dir7` | the stock 0x225ab8, then 17 instead of 8 when the course id is below 8 and no other cabinet's vote |
| 17 | `dir17` | saves Mission Select's timer; uploads and blits the art, NORMAL, not armed, the timer at 640 (one frame; it runs over into the next) |
| 18 | `dir18` | the two maps, the timer (0x226d40), the wheel (D6 rule), the band / frame / captions on a change, the gas from the timer's 0x270 or the timeout |
| 19 | `dir19` | the maps; the 32-frame hold (0x56(g13)); REVERSE: id + 8 and widen_ids' update; Mission Select's layers (0x225ad0), its timer restored, state 8 |

RAM 0x501b26-0x501b37: the choice (+0), armed (+1), the entry's directory record (u32 +2), the band shown (+6), the
map object list (u32 +10), the arrow variant (u32 +14), Mission Select's timer as state 7 left it (u32 +18: 320 on
the linked path MAME's TWIN cabinet takes, 640 on the single-cabinet one; restored by state 19).

Chars 0x3d00-0x3fff (unused by the boot and the select; 0x3d00-0x3dfc are also the course backdrop's last chars, `0x1de40`'s
lists to 0x3a00-0x3d8e and 0x3b00-0x3dfc, uploaded again by every course load before the race: Mission Select keeps
the last backdrop's tiles on layer 2 rows 49-54 only, below the 48 visible rows): the banner from 0x3d00 (150 chars, blocks 0x7a-0x7b), the
entry's panel from 0x3e00 (at most 120, blocks 0x7c-0x7d), the NORMAL side from 0x3f00 and the REVERSE side from
0x3f80 (85 each: caption, arrow, frame; one palette per block: pen 1 the arrow, 2 black, 3 the frame's yellow, 4 the
caption, swapped for the chosen side). The panels: 24 variants (8 entries x 3 game modes), a directory of 20-byte
records (char list, palette list, panel map, the normal and the REVERSE band maps); two arrow variants (NORMAL
clockwise or not), each a char list and the four area maps [choice][side]; per entry its map object list and its
arrow variant.

## Backdrop

Decoder `backdrop(rs, entry)` (`Backdrop`, in `course.py`; the segment format in `tiles.py`): the course's upload set at `CourseEntry.upload_list` (record word 0)
holds the backdrop's chars and palette and, in words 2-9, eight **segments** of one horizon panorama. Encoders
`encode_backdrop_segment`, `encode_upload_set`, `encode_char_list`, `encode_char_block`, `encode_palette_list`.

At the course load (frames 1328-1330) `0x1de40` uploads the chars (to `0x10e0000+`, chars `0x3000+`) and palette
(entries `0x600+`), then fills tile layer 2 rows 0-20 with the **top tile** (segment 0, row 0, col 0) and rows 21-63
with the **bottom tile** (segment 0, last row, col 0), layer 3 likewise (rows 0-19 bottom, 20-63 top). Every frame
`0x1dfd0` streams the panorama into **layer 2 rows 6 ..** as a 64-column ring: panorama column `p` (0-255) goes to
layer column `p & 63`, from segment `p >> 5`, column `p & 31`.

**Scroll** (Phase 4, traced): `0x4500` stores `Y` = `[0x50174c] + ([0x501748] << 5)` to `0x5fe11a` (the previous to
`0x5fe118`); on a single cabinet `0x6fd0`-`0x6ff0` sets `0x501748` to 0 and `0x50174c` to the camera yaw `0x5014a8`
(`0x10000` a turn), so `Y` is the camera yaw (equal on every frame of the trace, e.g. `0xd65d` at GO + 5 s). The
streamer `0x1df1c` fills the 62 columns from panorama column `(-Y) >> 8` (one turn = 256 columns), and `0x1dfa4`
stores `(Y >> 5) & 0x1ff` to `0x501308`, which `0x1b9b0` copies to tile RAM word `0x5002`, layer 2's horizontal
scroll (MAME `segaic24.cpp`: screen x `s` shows layer pixel `(s - scroll) & 0x1ff`). So screen x `s` shows panorama
pixel `s - Y / 32` (mod 2048; the register lags the yaw by a frame), checked against the snapshot at GO + 5 s within
2 pixels. Vertically the band's first row (layer row 6) is at screen y 25 (`0x50130a`, the vertical scroll, `0x17`).

Segment (`BackdropSegment`):

| Offset | Type | Field | Real values |
|---|---|---|---|
| `0x0` | `u32` | `flags` | `0x10000` on all 24 (no reader found) |
| `0x4` | `u32` | `rows` | Beginner 49, Expert 54, Advanced 43 (all eight segments of a course alike) |
| `0x8` | `u32` | `cols` | 32 (the streamer's `& 31`) |
| `0xc` | `u16[rows * cols]` | tiles | char `0x3000+`, colour `0x60+`, bit 15 clear (behind the polygons) |

| | Beginner (0) | Expert (1) | Advanced (2) |
|---|---|---|---|
| upload set | `0x2074240` | `0x2074268` | `0x2081180` |
| char blocks (chars) | 3 (3,327) | 2 (3,548) | 2 (2,429) |
| palette entries | 448 at `0x600` | 448 at `0x600` | 320 at `0x600` |
| segment 0 | `0x20742c0` | `0x207a520` | `0x20811f8` |

Checked against MAME at frame 2400 of a race on each course: all 64 layer-2 columns of the band equal the panorama
column with the same index mod 64, the rows above and below equal the top and bottom tiles, and every uploaded char
and palette entry equals RAM. `encode_backdrop_segment` refuses columns other than 32, rows outside 1-58 (the band
must fit layer 2 from row 6), a tile count that is not `rows * 32` and tiles past `u16`. For M4: a new course's
backdrop is a new upload set (its `upload_list`), up to 4,096 chars from char `0x3000` and palettes `0x60-0x7f`
(in the Beginner race at frame 2400 only layers 2 and 3, the backdrop, use chars `0x3000+`; the HUD in layer 0 uses
chars below `0x2446`).

**The mirror** (`mirror.mirror_backdrop`, Phase 4): the mirrored camera yaw is `0x8000 - Y`, so the mirror's screen
x `s` must show the original's `495 - s` flipped: column `q` of the mirrored panorama is column `(189 - q) & 255` of
the original (`mirror.MIRROR_COLUMN`; 189 = 62 screen columns - 1 - 128) with every char flipped left to right. System
24 tiles have no flip bit, and a tile's palette follows from its char index (128 chars per palette), so each flipped
char is a char of its own in the 128-char block of the char it flips: a symmetric char is kept, a char whose flip is
another char of its block takes that char, every other distinct flip is a new char at a free index of the block
(Beginner: 3,327 chars, 11 symmetric, 4 swapped, 3,312 new, each in the place of the char it flips). `mirror_course`
writes the mirrored char RAM as one new char block per upload of the original (same destination and count) at the
start of the data ROM's free run (`0x65d120`, 106,476 bytes for Beginner) and points the char list at them; the
original blocks stay in place. Beginner's top and bottom fill tiles are uniform and stay.

## HUD map

Decoder `hud_map(rs, entry)` (`HudMap`, in `course.py`), encoder `encode_hud_map` (the i960 tables) and
`polyrom.encode_object` (the outline). The runs and the disassembly: `docs/spike/2026-11-mirror.md`, "HUD map".

The race HUD's course map (below the CONDITION box, right of the screen) is **one polygon object per course, in world
coordinates**: a strip of flat black polygons (colour 1, untextured) at the course's (x, z), y 0. The set-up `0x12e58`
copies the course's three per-id entries into the HUD struct; every frame `0x1308c` sends the TGP translate
(`position`), scale (the zoom, eased by 0.2 a frame towards `zooms[level]`; `level` is the struct's `+0x4a`, 1 from
the set-up), a quarter turn (function `0x14`, `0x4000`), the course's `angle` (function `0x15`), then draws the
outline list (`0x17acc`). The **cars' markers** are drawn in that same frame: translated by the car's world x
(`+0x1c`) and z (`+0x24`) (y 0, the player's 1.0), turned by its yaw (`+0x58`) and scaled by `6e-8 / zoom` (a fixed
size on screen), from program lists (`0x13764[0x501464]` the player, `0x13794[]` the leader or, when the player
leads, the second, `0x1379c` / `0x137bc` linked cars; routines `0x13184`, `0x13250`, `0x132e0`, chosen by `0x5fe588`
and `0x540027`). No course table places a marker: a car's world position is its place on the map.

| Table | Where | Layout | Beginner / Expert / Advanced |
|---|---|---|---|
| `HUD_MAP_LISTS` | data `0x028052dc`, `u32[4]` by id | the outline's object list (16-byte entries, then an `oba` of 0), inside the object library | `0x028437b4` / `0x028458d4` / `0x02843890` (id 3: Beginner's) |
| outline list | that address | one entry: `oba`, `tpa`, `tha`, `budget` | `0x8d5da4` (25 records, 260 words) / `0x94dd59` (145, 1,460) / `0x8d61bf` (106, 1,070) |
| `HUD_MAP_PLACES` | data `0x028052fc`, 16 bytes by id | translation x, y, z (`f32`, view coordinates), `angle` (`s16`, `0x10000` per turn), `pad` (`u16`, 0) | z 9e-5 / 1.1e-4 / 9e-5; angle `0x4000` / -20480 / 0 |
| `HUD_MAP_ZOOMS` | data `0x0280534c`, `f32[4]` by id | world units to view units, by zoom level (0-3) | 7e-8, 5e-8, 3.5e-8, 0 / 2.7e-8, 2.4e-8, 1.7e-8, 0 / 4e-8, 3e-8, 2e-8, 0 |

The outline's extents hold the course's road line (Beginner x -319..319, z -195..156 against the road's -311..310,
-184..145; Expert's reaches 1,088, past the -1024..1023 grids). On the Beginner screen (496 x 384) the outline is drawn
at x 401-455, y 177-276, world x vertical and world z horizontal (view x = -zoom z + translation x). Reads of these
tables were listed as the course-dependent read `0x12efc` before Phase 4.

`encode_hud_map` refuses a course id outside 0-2, a list outside the object library, a list without an entry, an
entry whose object `objects` lacks (or an object no entry names), a position or zooms that are not finite `f32`, an
angle outside `s16` and a pad outside `u16`. **For M3/M4**: a new course needs its outline (an object in world
coordinates, flat polygons, in the polygon run, and a list in the object library, which has no free run: "Free
space"), its place record and zooms in its slot's entries; a course past id 3 needs the three arrays grown.

## Rankings

`rankings_layout()` (`RankingsLayout`) describes where the race rankings live in **backup SRAM**
`0x01d00000-0x01d03fff` (MAME `backup1`, NVRAM). A layout only: SRAM contents are never decoded or committed.

| Course (id) | Table (`RANKING_TABLES` `0x002390e4[id]`) | Newest-entry pointer (`RANKING_LAST` `0x002390f4[id]`) |
|---|---|---|
| Beginner (0) | `0x01d00360` | `0x01d00a20` |
| Expert (1) | `0x01d005a0` | `0x01d00a24` |
| Advanced (2) | `0x01d007e0` | `0x01d00a2c` |

Each table is 24 entries of 24 bytes, sorted by `time`:

| Offset | Field | Meaning (writer) |
|---|---|---|
| `0x00` | `rank` | `u32` 1-24, the slot's number; never moved |
| `0x04` | `time` | `u32` race time (`0x5010cc`, one tick per frame; a 1-lap Beginner race: 1,683), the sort key |
| `0x08` | `view` | `u32` the most used view 0-3 (largest of `0x501700-0x50170c`) |
| `0x0c` | `flags` | `u32` bit 0 = word `0x501a80`, bit 1 = word `0x501a90` (not interpreted) |
| `0x10` | `shown_time` | `u32` the time the ranking screen shows (`0x2285d0`); written equal to `time` |
| `0x14` | `name` | 3 ASCII bytes in the low 24 bits (display order not checked); `0x202020` until the name entry writes it (`0x228490`) |

At the goal (`0x227408`, when `0x5010d8` bit 1 is set) the new time replaces the last entry if it is lower and is
bubbled up (`0x227508` swaps fields `0x04-0x14`); the slot it ends in goes to the newest-entry pointer, and a rank of
5 or better sets bit 0 of `0x501398`, which lets `0x228470` store the entered name. At boot (`0x5a98`) a 24-byte
signature at `0x01d00200` ("S-RAM CHECK OK (4.0)", from program `0x5b78`) is checked; on a mismatch `0x2270d0` resets
the tables from `RANKING_DEFAULTS` (`0x002390cc`: (defaults, table) pairs, 24 x (u8 rank, name, flags) each) with
times `0x23909c[mode * 4 + id] + (rank - 1) * 320` (mode 0: 160 / 240 / 240 s in ticks of 1/64 s), and the other areas
below.

Other SRAM areas (`RankingsLayout.used`): `0x01d00000-0x01d001ff` settings, a copy and counters (written every few
frames by `0x22925c`-`0x2292cc`, `0x1db74`); `0x01d00220-0x01d0035f` sixteen 20-byte view presets (floats; read every
frame by `0x9c5c`, `0x9984` with the view index `0x501710`; a provisional name); `0x01d00b60-0x01d00ba3`,
`0x01d00be0-0x01d00c27` settings from `0x2270d0`. Nothing found uses **`0x01d00c28-0x01d03fff`** (13,272 bytes): no
direct address in the program, no access in the runs, and the NVRAM MAME saved after them is still `0xff` (its power-
on fill) from `0x01d00c28` on. (Phase 9's sweep of the ROM's pointer words: the time-attack section records
`0x230b9c` name `0x01d00c28` as padding, never reached; the ranked layout below leaves that word alone.)

`ranking_pointers(rs)` / `encode_ranking_pointers` read and write the per-id table array (ids 0-2, shared by every
Course).

**A table per course id (M4b, `course.grow_rankings`; `ranked(rs)`).** On the final chain (after the mirrors, before
the select; it commutes with `select.patch_select`) every course id has its own table and newest-entry word, ids 0-2
moved too; `rankings_layout(rs)` gives the ranked set's layout:

| What | Where | Size |
|---|---|---|
| ranking table of id k (`RANKING_TABLES[k]`) | SRAM `0x01d00c30` + 576 * k (`RANKING_SRAM`) | 16 x 576 = 9,216 |
| newest-entry word of id k (`RANKING_LAST[k]`) | SRAM `0x01d03030` + 4 * k (`RANKING_LAST_SRAM`) | 16 x 4 |
| left alone | SRAM `0x01d00c28` (`RANKING_PAD`): the time-attack section records' padding pointer (`0x230b9c`) names it | 4 (+4) |
| free | SRAM `0x01d03070-0x01d03fff` | 3,984 |
| (defaults list, table) per id (`RANK_DEFAULTS`) | program `0x1fe00` (`RANK_HOME`, reserved) | 16 x 8 |
| time bases `[mode * 16 + id]` (`RANK_BASES`) | program `0x1fe80` | 3 x 16 x 4 |

The reset `0x2270d0` reads them with the same instructions (`course.RANKING_PATCHES`: `mov 16,r15`, `lda
RANK_DEFAULTS,r14`, `subo r15,16,g1`; `0x22713c`'s `shlo 4,r5,r5`, `ld RANK_BASES[r5*4],r4`). Defaults
(`RANK_SOURCES`): ids 0-2 their own (byte-equal to the stock reset's), the mirrors 8-10 their original's, the others
Beginner's. The signature becomes `S-RAM CHECK OK (4.0/16)` (`SIGNATURE_RANKED`): an older SRAM (the stock set's,
Phase 8's) fails the boot's compare, which writes the new one and calls the reset, so the sixteen tables start fresh.
The reset never writes the view words or the newest-entry words; no older set wrote the new area, so they are `0xff`
as on a fresh NVRAM. The old tables and words (`0x01d00360-0x01d00a2f`) are read by nothing and keep an old SRAM's
bytes. The reset's other callers (`0x12dc`, a changed difficulty setting; the test menu's backup clear `0x22edcc`)
reset all sixteen. The guard's sink still stands in for ids 3-15's time-attack and best-lap words
(`SRAM_SINK_ARRAYS` but the two ranking arrays). Evidence: `docs/spike/2026-11-mirrors-live.md`, "Rankings"
(`test_rankings_per_entry`).

## Desert City as id 3 and the SRAM guard (Phase 8)

`build.converted_set(base, disc, 3, slot=3, scenery="model")` with `base = patch_select(guard_new_ids(grow_ids(stock)))`
(custom sets only) puts Desert City beside the originals as course id 3, with a table of its own for everything the
game reads by id (evidence: `docs/spike/2026-11-sixteen.md`, "Desert City at id 3"). Ids 0-2 are byte-identical to
`grow_ids`' output.

| Table | Where | Named by |
|---|---|---|
| ground grid and attributes, step table, road line, AI lines, checkpoint table, segment flags, time words and tables | data `build.ID_DATA_FREE` (`0x1001800`-`0x1efffff`) | record 3 (`ID_RECORDS + 0x60`); `CHECKPOINT_TABLES[3]`, `SEGMENT_FLAGS[3]`, `TIME_TABLES[m][3]` |
| two ground pointer words | `0x1037804` / `0x1037808` | record 3 words 2-3 |
| cell grid | `0x1038cac` | `GRID_TABLES[3]` |
| 1 MB course texel block | data `0x1100000` (the block a copy of Beginner's, Desert City's texels over it) | `COURSE_SHEETS[3]` |
| ground list block and pool | copro_data custom area, word `0x100000`+ | TGP slot 3's words |
| 69 objects and the HUD outline | polygons custom area `0x1000000`+ (index `0x400000`+) | the lists |
| headers and points | textures first gap `0x400000`+ | the lists' `tha` / `tpa` |
| 69 object lists, empty list, the outline's list | `DATA_FREE` from `0x65d120` (the lists must lie below the draw routines' upper bound `0x0284f4ec`; the custom data area is above it) | the cell grid, `HUD_MAP_LISTS[3]` |
| actor table (256 zeros) | program free run `0x3ce98` (an actor table must be a program-window address); since Phase 10 the shared `build.EMPTY_ACTORS` there ("Shared empty actor tables") | `ACTOR_TABLES[3]` |
| HUD place and zooms, grid base, rolling start, laps, car count | id 3's own entries of the grown arrays | |

`build.model_scenery(..., slot=3)` places at the first free bytes of the new runs; slot 0 keeps Phase 7's end-of-run
placement. `tgp` accepts a ground in the custom copro area (`tgp.CUSTOM_WORDS`); `course.rom_object_index` bounds
an object index by the 32 MB window. Empty scenery is refused for slot 3.

**The SRAM guard** (`course.guard_new_ids`, on a grown set): no id above 2 writes backup SRAM.

- The four per-id counter arrays (`0x1d00140`...; `0x228fb0`, called at race end `0x5680`): `course.COUNTER_GUARD` turns
  `0x229020` / `0x229040` into `cmpobl 2,g6,0x229094; mov g6,gN; mov g6,gN` over the no-op `lda 0xffff,g3; and g6,g3,gN`;
  ids above 2 jump to the tail (checksum and backup copy). Ids 0-2 are unchanged.
- The rankings, newest-entry words, best laps and time-attack words: entries 3-15 of the eight `SRAM_SINK_ARRAYS` point at
  a read-only ROM copy of Beginner's fresh-reset values, `course.SRAM_SINK` (`0x03001580`, `0x260` bytes, built by
  `sram_sink` from `fresh_ranking` / `fresh_settings`, which replay the reset `0x2270d0` / `0x227168` and refuse a time
  outside u32). The game's writes there are ROM writes and are dropped; reads give the fresh defaults. The sink is in
  `romset.RESERVED` and out of `CUSTOM_FREE`.
- The SRAM layout and the signature at `0x5b78` are unchanged: SRAM from before reads as it did. Per-entry rankings are
  M4b's: `course.grow_rankings` (Phase 9) gives every id its own table and newest-entry word and changes the signature
  ("Rankings"); the sink then stands in for the time-attack and best-lap words only.
- Proven offline (`test_guard_new_ids`: disassembly of the four sites, every entry 3-15 naming ROM) and at run time only as
  far as a lap goes (backup SRAM stays fresh through Desert City's lap). A finished race or the time attack, where the
  writers run, was not driven: Not found (time box).

## A mirror tests as its original (widened sets, Phase 9)

A mirror's course id is its original's + 8. `course.widen_ids(rs)` (on a grown set; idempotent, `course.widened`)
makes every test of the id against 0, 1 or 2 see a mirror as its original, makes `GRID_LANES` and `BILLBOARD_REF` per
id and picks the tune per id. Ids 0-7 take the paths and read the values they did (decodes and MAME runs:
`test_widened_originals`). The chain: `grow_ids`, `guard_new_ids`, `widen_ids`, then `select.patch_select` (widen and
the guards commute; Phase 8's `--sixteen` playtest set does not widen). Evidence: `docs/spike/2026-11-mirrors-live.md`,
"Literal ids".

**How.** No site gains an instruction (the race is cycle-exact work: a trampoline of three cycles at a site run every
frame moved Desert City's AI within a minute; MAME's i960 charges cycles by opcode). Each site's 8-byte instruction
(a MEMB load of the id `0x501460`, a lane reader, the billboard `lda`) keeps its opcode word; its displacement names a
shadow in work RAM `course.WIDE_RAM` (`0x501b40`, `romset.RESERVED_RAM`) instead (`course.WIDE_SHADOWS`). The code at
`course.WIDE_CODE` (`update`, called) refreshes the shadows from the id and the per-id tables; it runs after every
program store of the id, each moved into a trampoline (`course.ID_WRITERS`: the boot's clear `0x17a4`, the attract's
next course `0x213c`, the vote `0x32dc`, linked mode's `0x5720` and `0x1b084`). The select's own stores (every frame
of the wheel) are not hooked: no widened site reads the id before the vote. `course.id_table` gives a `WIDE_ARRAYS`
array's per-id table on a widened set (its sites name the shadow).

| Shadow (work RAM) | Address | Holds | Read by |
|---|---|---|---|
| `SH_PATH` | `0x501b40` | `u8` id & 7 | the and7 loads |
| `SH_TUNE` | `0x501b41` | `u8` `TUNES[id]` | the music id's loads `0x33d8`, `0x3b9c` |
| `SH_VISIT` | `0x501b42` | `u8` `VISIT_IDS[id]` | the culling walk's load `0x17418` |
| `SH_LANES` | `0x501b44` | `u16[2]` the id's `GRID_LANES` | `0xd840`, `0xdb0c`, `0xe980` |
| `SH_REF` | `0x501b48` | `f32[3]` the id's `BILLBOARD_REF` | `0x221e88` (`lda`, then `ld 0x0(r8)`, `ld 0x8(r8)`) |

| Tables (program ROM) | Address | Layout | Values `widen_ids` writes |
|---|---|---|---|
| `GRID_LANES` per id (`ID_ARRAYS["GRID_LANES"].new`, `course.WIDE_LANES`) | `0x1f800` | `u16[2]` per id | the shared pair (`0x180`, `0x580`) for every id |
| `BILLBOARD_REF` per id (`course.WIDE_BILLBOARD`) | `0x1f840` | `f32[3]` per id | the shared point (210, 0, -380) for every id |
| `course.TUNES` | `0x1f900` | `u8` per id: the music id | `TUNE_DEFAULTS`: 0, 1, 2, 3, then 0 for 4-7; 8-15 as 0-7 |
| `course.VISIT_IDS` | `0x1f910` | `u8` per id: tested against 2 | `VISIT_DEFAULTS`: id & 7 (2 for ids 2 and 10: Advanced's order `0x175ad` at cells 72 and 88) |
| code (`course.WIDE_CODE`) | `0x1fa00` | 204 bytes: `update`, five writer trampolines | |

The sites (`course.LITERAL_ID_SITES`, by the test's address; the load is the instruction replaced):

| Test | Load | Guards | Rule |
|---|---|---|---|
| `0x3eb4 cmpibe 0,r3` | `0x3eac` | state 20: Beginner skips the standing start's 320-frame countdown | and7 |
| `0x7370 cmpobe 0,r8` | `0x7368` | the chase camera's pitch correction, skipped on Beginner | and7 |
| `0x7ad4 cmpobne 2,r3` | `0x7acc` | a sound call with Advanced's parameters | and7 |
| `0xc2a4`, `0xc2d0 cmpi r3,0` | `0xc29c`, `0xc2c8` | the rolling start's steps | and7 |
| `0x15d90 cmpi r3,0` | `0x15d88` | the slipstream message's lap (Beginner 3, others 1) | and7 |
| `0x17420 cmpobne 2,r3` | `0x17418` | Advanced's cell visit order at cells 72 and 88 | table: tests `VISIT_IDS[id]` (the cells are code literals) |
| `0x19ae4`, `0x19aec`, `0x19af4 cmpi r4,0/1/2` | `0x19adc` | a positional sound's segment ranges per course | and7 (the mirror keeps the segment order) |
| `0x224cf0 cmpibne 0,r3` | `0x224ce8` | Beginner's rolling-start pace task at once | and7 |
| `0x3bec cmpi r3,2` (the music id) | `0x3b9c`, `0x33d8` | music 2 adds the sound `0xb70700` | table: the music id is `TUNES[id]` |

"and7": the load reads `id & 7` (its register is dead after the test on every path), so ids 8-15 test as 0-7 and 11-15
as 3-7 (which take the generic path as 3-7 do). Left (`course.LITERAL_ID_KEPT`): `0x63f4`
(linked mode's camera script, decision 20), `0xc3e0` (a no-op branch), `0x102e4` / `0x10b54` (`cmpobe 31`), `0xb820` /
`0xb824` (the time-attack section records: SRAM words of Expert and Advanced, which a mirror must not share) and the
counter guard `0x229020` / `0x229040`.

**The tune.** The music id `0x501a98` (`0x340c`, `0x3bd0`: the course id unless a view button picks 0-3) indexes
`MUSIC` `0x28050f0` (`0x3be0`), `0x280eeb8` (`0x223da8`) and `0x2810d58` (`0x224644`), four sound codes each; on a
widened set its source is `TUNES[id]`, so it stays 0-3 (a grown set's ids 4-15 read the last two past their ends) and a mirror
plays its original's tune (decision 12).

**Lanes and billboard per id.** `course.grid_lanes(rs, course)` and `billboard_ref(rs, course)` read the course's entry
(`course.id_entry`: a `WIDE_ARRAYS` array at its original address is one entry every id reads), `start_grid` lays a
course out on its own lanes, and `golden` decodes and writes them per id (`DECODED_ARRAYS`; a Course's
`id_arrays` names the per-id tables, so a Course from a set that is not widened is refused on a widened one).
`ID_ARRAYS` = `GROWN_ARRAYS` (grow_ids') + `WIDE_ARRAYS` (these two).

**Before a mirror is written.** An id without a course of its own holds Beginner's tables (grow_ids' copies), and
its `ROLLING_STARTS` entry Beginner's rolling word: id 8 takes the rolling path that word asks for; ids 9-15 would take
it on the standing path (Phase 6's crash). `mirrored_entry` (below) gives ids 8-11 their originals' words; ids 12-15
keep Beginner's until M5 converts their originals, and `select.patch_select` refuses a live entry 3-15 whose id has no
course of its own (`select.FILLED_FIRST`, `golden.own_tables`; id 3 too since Phase 9's final review: Desert City is
converted before the select), so they stay locked.

## A mirror beside its original (`mirror.mirrored_entry`, Phase 9)

`mirror.mirrored_entry(rs, o)` writes the mirror of course id `o` (0-3) as course id `o + 8`
(`mirror.MIRROR_ID_OFFSET`) on a grown, guarded and widened set, with its original already in place (for `o` = 3,
Desert City converted at id 3 first). Evidence: `docs/spike/2026-11-mirrors-live.md`, "Mirrors of the originals",
"Desert City mirrored". The final chain: `grow_ids`, `guard_new_ids`, `widen_ids`, `build.converted_set(..., 3,
slot=3, scenery="model")`, `mirrored_entry` of 0, 1, 2, 3, `course.grow_rankings`, `select.patch_select(ENTRIES)`
(`tools/desert_city_playtest.py --mirrors`, `tests/test_mirrors_mame.py` `final_set`). Each call takes the first free
bytes of each run, so the call order fixes the layout. Refused (`CourseError`): an original outside 0-3 (a bool too),
a set missing a layer (grown, guarded, widened, checked in that order), an id already filled (`golden.own_tables`),
original 3 while id 3 is still Beginner's copy.

**The per-id copy rule.** First the alias: id `k`'s record becomes the original's, TGP slot `k` takes slot `o`'s two
words, `TUNES[k]` the original's, `VISIT_IDS[k]` `mirror.VISIT_OF_MIRROR[o]` (8: 0, 9: 1, 10: none `course.VISIT_NONE`,
11: 3), and every per-id array entry of `k` the original's, except:

| Group | Entries | Value at id `k` |
|---|---|---|
| own (`mirror.OWN_ARRAYS`) | `COURSE_POINTERS`, `GRID_TABLES`, `ACTOR_TABLES`, `HUD_MAP_LISTS` | the mirror's record, cell grid (rows reversed; lists naming scenery new), actor table (in `PROGRAM_FREE`; an original with no actor, Desert City, gives the shared `build.EMPTY_ACTORS`, the original's own entry, so id 11's `ACTOR_TABLES` equals id 3's), HUD outline list |
| mirrored (`mirror.MIRRORED_ARRAYS`) | `GRID_LANES` | `mirror_lane` of each (`0x180`, `0x580` -> `0x57f`, `0x17f`) |
| | `BILLBOARD_REF` | the point mirrored as a world point where the original has a view-angle billboard (Beginner); else the original's (unread) |
| | `CAMERA_SCRIPTS` | a new script (`mirror_camera_script`): walked in play order (the shared loop `0x2808360` copied, its jumps repointed); a keyframe point in the car's frame (-x, y, z), an absolute one (`CAMERA_ABSOLUTE`) mirrored in the plane, roll negated |
| | `TUNNEL_LIGHTS` | the original's five words, the fifth naming a mirrored position table (`mirror_tunnel_lights`, 240 bytes) where the record has lit zones (Beginner, Advanced) |
| | `PROGRESS_MASKS` | rows reversed (`r` -> `15 - r`) and an end record (its two segment words) where there are records (Advanced); the others keep the shared end record `0x175f0` |
| unshared (`mirror.UNSHARED_ARRAYS`) | `RANKING_TABLES`, `RANKING_LAST`, `TIME_ATTACK_*`, `BEST_LAPS_*` | left as they are: the guard's sink, then `grow_rankings`' own table and word per id ("Rankings"); a mirror never writes its original's SRAM |
| | `SELECT_PANEL_OF_COURSE` | the select's |

Copied as the original's (the course is the same race run the other way round): the checkpoint table, segment flags,
time tables, laps, car counts, grid base, `ROLLING_STARTS` (so Beginner's mirror starts rolling, the others
standing), `COURSE_SHEETS` (the original's texel block: texture headers and points are shared too), the HUD place and
zooms, the sound and AI tables (`TABLES_28053AC` and the rest), `TUNNEL_BYTES`, `ATTRACT_NEXT`, `MUSIC`, the select's
per-id uploads, sounds and palettes.

**Where the mirror's own tables go** (through `build._first_fit` / `_usable` / `_fits`, written by
`build._write_free`; every byte owned by a table is busy, `golden.owned_spans`):

| Region | Run | Tables |
|---|---|---|
| data | `build.ID_DATA_FREE` | ground grid and attributes (the record's two ground pointer words name them), road line, line table, cell grid, backdrop upload set, char list and char blocks (`mirror._backdrop`, Phase 9 Task 1: the race-time uploader `0x1dd70` reads anywhere in the data window), billboard list arrays, camera script, tunnel light positions, progress masks |
| data | `tgp.DATA_FREE` (below the object library's bound, `course.LIBRARY_BOUND_PATCH`) | every object list naming a scenery object, the HUD outline's list |
| copro_data | the custom copro area, window-safe | ground lists and polygons (TGP slot `k`) |
| polygons | the custom polygon area | the mirrored scenery objects and HUD outline (`C.rom_object_index`: Desert City's objects lie past 16 MB) |
| program | `build.PROGRAM_FREE` | the actor table and actor lists (none when the original has no actor: the shared `build.EMPTY_ACTORS`) |

The check: every byte `golden.tables` gives id `k` lies in a span just taken, equals the set's, or is one of `k`'s own
words (record, per-id entries, TGP slot words, `TUNES` / `VISIT_IDS` bytes); else `MirrorError`. The set must decode id
`k` back to the course built (`_decodes_differ`). `golden.extra_spans` owns the three tables no decoder reads (the
camera script's records, the tunnel positions, the progress-mask end words). Kept unmirrored: Beginner's switch (its
handler `0x220a54` names its three lists by `lda` literals in shared code), the positional sound's source points
`0x19f1e`-`0x19f92`, the unmodelled actors' handler data. Two more known losses of the copy (Phase 9's final review):
`OFFSETS_231B34` is the original's (`mirror._alias` copies it unchanged), and its offset indexes the "D.K." bank
keyframes at `0x2880004` that the task at `0x501220` writes (handler `0xd104`, set at the race's end by `0x50e0`):
probably the post-goal camera, so a mirror's goal sequence is its original's, not mirrored (unverified in MAME). And
`WORDS_230CCC` at ids 9 and 10 holds Expert's and Advanced's time-attack section counts; its readers sit behind the
`0xb820` guard, which ids 3-15 skip (harmless), so a mirror gets no time-attack section splits. For a course M5
converts at id `n` (4-7), its mirror is one more call, `mirrored_entry(rs, n)` (today `MIRROR_ORIGINALS` is 0-3: the
call refuses 4-7 until it is widened).

## Golden re-emit

`dtools/arcade/golden.py` decodes every table above for one course into a `Course` (`decode_course(rs, id)`),
re-emits them (`encode_course`: `(region, offset) -> bytes`, refusing tables that overlap with different bytes) and
compares (`verify_lossless(rs, id)`: one line per mismatch, naming the table and the first differing byte).
`coverage(rs, id)` lists the bytes compared, `golden.EXCLUDED` what is not (by name, with the reason) and
`excluded_spans` the excluded bytes whose extent is known.

`encode_course` first runs `check_course`, the rules between tables that no single encoder sees, and refuses
(`CourseError` naming the rule and the values) a Course that breaks one: every AI line has the road line's
`n + 1` points and `step_n` is that `n`; step and checkpoint segments are at most `n`; every AI point's `step` is
below the step count; every time table holds `1 + laps * (checkpoints + 1)` words (the phantom word included); the
spawn tables give a record for every car of the car count, lines below 8; at most 65 actors (`course.ACTOR_POOL`,
the count at data `0x02802f74`); the record pointer is `0x02600020 + 0x20 * id`; the ground (`Course.ground_data`,
a `tgp.Ground`) has the Course's id, holds one attribute byte per polygon (`len(attributes) == len(polys)`: the
block's `0xcd` padding to 4 bytes is not stored, `tgp.encode_ground` writes it), its polygons of kind 1
(`attributes[i] & 15`, road) are exactly the first `len(steps)` (503 / 1,064 / 818), every grid value is 0 or names a
list and every list entry a polygon that exists (the encoder's own checks, polygon shape and layout, raise
`tgp.TgpError`). A Course also carries **views**,
fields decoded for reading and never written: `pit` (from the step flags), each `Checkpoint`'s position and
extensions (from the road line and the mode 0 / difficulty 1 time table), each `GridSlot`'s segment, lane and position
(from its start, the grid base, the grid lanes, the start type and the road line), and with `ground_data` set the two
ground pointer words `Course.ground` (`0x02000000 + ground_data.grid_at` / `attributes_at`; `tables` writes them from
`ground_data`). `check_course` recomputes them
(`course.checkpoint_views`, `course.grid_slot`, the pit from the steps) and refuses a Course whose views differ from
their sources: an edit goes into the source (`encode_pit` writes a pit into the steps). The `Course` docstring lists
the address convention of every field (i960 addresses, data-ROM offsets, polygon-ROM, textures and `copro_data` word indices). All three courses re-emit identically
(`tests/test_golden.py`), including the structures: the cell grid and list terminators, the actor tables, lists and
counts, the line table's pairs, the upload sets and their lists, the panel records and maps, the pointer words,
and the ground (milestone M1b, `docs/formats/model2-tgp-ground.md`): the grid (`*block_ref_a`, 32 KB) and the
attribute block (`*block_ref_b`, padding included) in data ROM, and in `copro_data` (its first 2 MB copy;
`RomSet.with_patch` writes both) the course's two slot words, its list block and its polygon pool. The ground's
extents are `tgp.ground_spans`, not "to the next course's step table", so a set whose ground was moved
(`tgp.relocate_ground`) still decodes, re-emits identically and gives `coverage` and `excluded_spans`
(`tests/test_golden.py`, `test_golden_on_a_relocated_ground`). Since Phase 4 (M1c) a Course also holds the HUD
map (`Course.hud_map`, "HUD map"): its four i960 tables ("HUD map") and its outline object ("HUD map outline", kept
out of `Course.objects`), whose texture headers join `Course.headers`; `check_course` refuses a `HudMap` of another
course id.

Bytes covered per region (the union of `coverage`, pinned in `tests/test_golden.py`):

| Region | Beginner (0) | Expert (1) | Advanced (2) |
|---|---|---|---|
| data | 1,846,912 | 2,310,270 | 2,022,428 |
| copro_data | 94,640 | 551,600 | 232,272 |
| polygons | 1,169,760 | 1,768,040 | 1,186,080 |
| textures | 193,808 | 298,288 | 206,208 |
| program | 2,934 | 3,346 | 2,586 |

The ground added to data the grid (32,768 bytes) and the attribute block (1,184 / 7,240 / 2,988; Phase 2's data
counts were 1,812,904 / 2,270,206 / 1,986,616) and all of `copro_data`: the two slot words, the list block (18,856 /
88,296 / 41,160 bytes) and the pool (75,776 / 463,296 / 191,104). The HUD map (Phase 4) added 56 bytes to data (list
pointer 4, list 20, place record 16, zooms 16), the outline to polygons (1,040 / 5,840 / 4,280 bytes) and its texture
headers to textures (192 / 1,064 / 784).

## What the converter must produce

This is the list a converter (M3) plans from: for one new course in an existing slot (M3 swaps it in place of
Beginner, id 0), every table the game reads for that id. Each item says what the table is, its fields and units, the
encoder that writes it, whether the bytes are **rewritten in place** or **written elsewhere and repointed** (a new
blob, then the one word that points at it rewritten), and whether M1's golden test covers it. `golden.Course` with
`golden.encode_course` composes the covered encoders into `(region, offset) -> bytes`; `RomSet.with_patch` and
`write_zip` (`dtools/arcade/romset.py`) write the patched set. The encoders raise `CourseError`, `TextureError`,
`PolyError`, `GeoError` or `RomSetError`, all subclasses of `DecodeError` (`dtools/arcade/errors.py`, re-exported by
`tablebase`): catch `DecodeError`. Every encoder takes Python or numpy integers (never bool) for integer fields and
finite Python or numpy numbers that fit single precision for float fields (NaN, infinity and bool are refused).

### Units

| Quantity | Unit |
|---|---|
| positions (road line, AI lines, actors, polygon points) | arcade units, f32, world x, y, z |
| Dreamcast to arcade | one arcade unit per Dreamcast unit, z negated: arcade (x, y, z) = (x, y, -z)_DC, no rotation or offset; exact on courses 0-2's data and true of the CS0 model (Phase 5, Task 7: dreamcast-disc.md, "Coordinates against the arcade"; Phase 1's overlay estimate of 1.01–1.04 does not hold) |
| world extent | x and z inside -1024..1023: the scenery grid (16 x 16 cells of 128) and the ground grid (128 x 128 cells of 16) both start at -1024 and wrap outside it |
| segment | one road-line record, about 8 units apart (median 8.0 on all three courses); a `u16` index |
| lane | lane points 3 units apart (2.92-3.09 on the real set); a car's lateral position in lanes as 8.8 fixed point, `0x000`-`0x700` |
| yaw | `heading << 2`, `0x10000` per turn; the direction of travel in (x, z) is (-sin, cos) of it |
| actor angle | `s16` about y, `0x10000` per turn (inferred) |
| race time | ticks, `u32` (the tick's length is in "Unknowns") |
| texture points | 1/8 texel, unsigned, v then u per vertex |
| texels | 4-bit indices; the hue is the header's palette entry, the shade comes from the luma table |

### Free space

Runs of `0xff` on the real set (`docs/formats/model2-rom-images.md`):

| Image | Bytes | Start (image offset) | As the game addresses it |
|---|---|---|---|
| program | 12,648 | `0x3ce98` | i960 `0x0003ce98`, program window `0x0023ce98` |
| data | 667,360 | `0x65d120` | i960 `0x0265d120` (the run ends at `0x700000`) |
| polygons | 1,850,524 | `0xe3c364` | word index `0x38f0d9`, so `oba` `0x00b8f0d9`; the run ends at the image end |
| textures | 709,304 | `0xb52d48` | word index `0x5a96a4` (`tpa`, `tha`) |

The M1c build (`mirror.mirrored_set`) takes the data run's first 106,476 bytes (`0x65d120`-`0x67710b`) for the
mirrored backdrop's char blocks ("Backdrop"); nothing else in it uses the run.

The polygons image has about 4.2 MB of `0xff` runs of 100 KB and more in all; the textures image also has runs at
`0x80ff10` (590,064 bytes) and `0xa3e2a8` (400,728 bytes). `copro_data` has 882,370 bytes at `0x12893e` in each of
its two identical halves (M1b).

What this means for a new course:

- A course's texel sheet is one whole 1 MB block of data ROM (`0x15a0[id]`). The data run is smaller, so a course
  swapped into Beginner's slot rewrites Beginner's block (`0x200000`) in place. A block of its own needs 1 MB that no
  free run has (spec decision 9: a custom MAME build enlarges the regions).
- Data ROM `0x800000-0xffffff` is one 1 MB block loaded eight times. The object library, the polygon palette, the
  view-test tables and the per-id arrays at `0x2805xxx` are in it. `with_patch` writes every copy and refuses a patch
  that spans two.
- The object library (`0x0283fbb4`-`0x0284f4eb`) has no free run: the polygon palette starts right after it, then the
  grids. New object lists must replace lists inside it, or the bound checks at `0x17acc` and `0x17cf8` must be patched.
  The converter patches them (`course.LIBRARY_BOUND_PATCH`: the lower bound's `lda` word of each, `0x17ad0` and
  `0x17cfc`, moved down to DATA_FREE's start `0x0265d120`) and writes its lists into DATA_FREE; the originals draw
  the same with it (docs/spike/2026-11-convert.md, "Object library").
- Actor tables and lists are in the program window, and `encode_actor_table` refuses a pointer outside it; the
  program run (12,648 bytes) holds them. Whether the spawner would accept data-ROM addresses is not known.

### Checklist

Legend (Phase 6, updated in Phase 7): `[x]` the converter produces it (rule and test given); `[~]` kept as the slot's, or emptied, by ruling; `[ ]` not produced yet. Phase 6 (M3a) converted n 3 into Beginner's slot with no scenery (`scenery="empty"`); Phase 7 (M3b) adds the model scenery (`scenery="model"`).

Course record:

- [x] **Course record** (`0x02600020 + 0x20 * id`, 32 bytes). Words 0-5: data-ROM pointers to the tables below.
  Words 6-7 (`raw`): equal to the pointer stored at word 2's address on the real set; no reader found. Encoder
  `course.encode_course_entry`. Rewritten in place; `COURSE_POINTERS` (program `0x4770`) stays. Golden: covered.
  Converter (Phase 6): `build.converted_course` moves words 1, 4, 5 (`step_table`, `line_table`, `record_table`) to the new tables and rewrites words 6-7 with `tgp.repoint_record_raw`; `encode` refuses any other change. Test: `test_converted_round_trip`, `test_converted_only_free_bytes` (record words changed = {1, 4, 5, 6, 7}).

Road, AI and ground:

- [x] **Road line** (word 5). `u32 n`, `u32 8`, then `n + 1` records of 164 bytes, one per segment: `lane_lo`,
  `lane_hi` (u8, the used lanes), `curve` (s16, a signed bend, unit unknown), then 8 lanes of position (3 x f32,
  units), pitch, heading, roll (s16 angles) and `pad` (u16, 0). Encoder `course.encode_road_line`. Written elsewhere,
  word 5 repointed. Golden: covered.
  Converter: `convert.tables.road_line`, written in `DATA_FREE`. Rule: position (x, y, -z), pitch negated, heading - `0x1fff` (road-lane headings: one per road point, the across-road direction less a quarter turn), roll, `curve`, `lane_lo`, `lane_hi` equal. Test: `test_tables_beginner`, `test_tables_expert_advanced` (n 0-2 equal ids 0-2 byte for byte), `test_lane_headings_are_the_cross_section_normal`, `test_tables_desert_city` (465 points).
- [x] **AI lines** (word 4). Eight (pointer, count) pairs, count `n + 1`; each list is `n + 1` points of 28 bytes,
  then its first 8 points again. Point: target position (3 x f32, units), `speed` (f32 factor, 0.59-1.42 on the real
  set), pitch, heading, roll (s16), `steer` (s16 bias), `lane` (s16, 8.8 lanes), `step` (u16, a road polygon below
  `steps`). Encoder `course.encode_line_table(offset, lines)`. Written elsewhere, word 4 repointed. Golden: covered.
  Converter: `convert.tables.ai_lines`, in `DATA_FREE`. Rule: as the road line; `speed`, `steer`, `lane`, `step` equal (the speed within 6e-6: the arcade's exact decimal is Not found). Test: `test_tables_beginner`, `test_tables_expert_advanced`, `test_converted_beginner_from_dc`, `test_dc_beginner_drives_identically` (MAME).
- [x] **Step table** (word 1). `u32 n`, `u32 steps`, then per road polygon a `u16` segment and `u16` flags: `0x01`
  road, `0x42` pit lane, `0x82` the one pit box. Encoders `course.encode_step_table`, `course.encode_pit`. Written
  elsewhere, word 1 repointed. Golden: covered. The road polygons are the first `steps` polygons of the ground (M1b).
  Converter: `convert.tables.steps`, in `DATA_FREE`; pit by `course.encode_pit`. Rule: steps equal (`0x01`, `0x42`, `0x82`; `0x101` on Advanced). Test: `test_tables_beginner`, `test_tables_expert_advanced`, `test_tables_desert_city` (527 steps).
- [x] **Ground** (words 2-3, M1b; `docs/formats/model2-tgp-ground.md`). The grid (`*block_ref_a`, 128 x 128 `u16`
  grid values, a word offset into the course's list block or 0), the attribute block (`*block_ref_b`, one byte per
  polygon, `& 15` the surface kind, padded to 4 bytes with `0xcd`) and on the TGP side the course's slot words
  (`copro_data` words `0x10 + id`, `0x20 + id`), its list block and its polygon pool (16 words per polygon: four
  corners, the centre, the direction); the first `steps` polygons are the road (kind 1). Decoder and encoder
  `tgp.decode_ground`, `tgp.encode_ground`. Written elsewhere (`tgp.relocate_ground`: into free `copro_data` and data
  ROM; the two pointer words at `block_ref_a` / `block_ref_b`, the record's words 6-7 and the slot words
  rewritten). Golden: covered.
  Converter: `convert.ground.convert_ground`, placed by `build.converted_course` (grid and attributes in `DATA_FREE`, lists and pool in `COPRO_FREE`). Rule: corners and centre with z negated, corners in the same order (no reordering; negating z alone gives the negative XZ area), direction - `0x8000`; grid and lists rebuilt with the arcade's rule, not copied from H2 / H7; a corner under an f32 ulp inside its neighbours' line is moved outward (at most 4 ulps; n 3: cells 1689 and 1915 by 1 ulp). Test: `test_ground_beginner`, `test_ground_expert_advanced`, `test_ground_queries_beginner`, `test_ground_desert_city` (every AI and road lane point queries a polygon), `test_converted_only_free_bytes`.
- [x] **Per-segment flags** ("Per-segment flags"). `SEGMENT_FLAGS[id]` (data `0x028058bc`) points at one `u16` per
  road segment (`n + 1`), ORed with the step flags into car `+0x13c`: `0x100` a tunnel (inferred), `0x400` the AI keeps its
  line (inferred). Encoder `course.encode_segment_flags`. Written elsewhere, the `SEGMENT_FLAGS` word repointed with
  `encode_table_pointer` (the converter writes `n + 1` zeros: the Dreamcast has none). Golden: covered (Phase 6).
  Converter: `convert.tables.segment_flags`, `n + 1` zeros in `DATA_FREE` (Not found on the Dreamcast; exact on n 0 and 2, short of Expert's 51 segments on n 1). Test: `test_converted_segment_flags`.

Scenery:

- [x] **Cell grid**. `GRID_TABLES[id]` (data `0x02805298`) points at `u32[256]` object-list pointers, cell
  `row << 4 | col` from z and x. Encoder `course.encode_cell_grid` (refuses lists outside the object library, or outside `PATCHED_LIBRARY` when the bound patch is applied). A new
  grid, then the `GRID_TABLES` word rewritten in place. Golden: covered.
  Converter: emptied, not converted. With `scenery="empty"` all 256 cells name `build.EMPTY_LIST` (`0x0284ccec`, Beginner's cell 0 stub rewritten to its terminator); the grid is rewritten in place. With `scenery="model"` (Phase 7) the model's polygons are grouped by `model.cell_of` into the 256 cells (Desert City: 69 occupied, the rest name one empty list in `DATA_FREE`); the grid is rewritten in place. Test: `test_empty_scenery`, `test_empty_list_choice`, `test_desert_city_scenery_round_trip`.
- [x] **Object lists**. 16-byte entries (`oba`, `tpa`, `tha`, `budget`), then an `oba` of 0. `budget` is added to
  the frame's polygon count, and the draw stops once the count passes 5,000; on Beginner it is the object's record
  count for the 22 tiles and 0 for the one-record stubs. Encoders `course.encode_object_list`,
  `course.encode_placement`. Inside the object library (above). Golden: covered.
  Converter: with `scenery="empty"` the one empty list; with `scenery="model"` (Phase 7) one list per occupied cell, one entry per object (`oba`, `tpa`, `tha`, budget = the object's record count), written at the end of `DATA_FREE` (`build.LIST_HOME` = "data") behind the two-word bound patch `course.LIBRARY_BOUND_PATCH` (program words `0x17ad0`, `0x17cfc`: the lower bound `0x0283fbb4` becomes `0x0265d120`, so lists outside the object library are drawn; the 153 call sites of the checks rely on no admitted range). Test: `test_library_bound_patch_draws_originals` (MAME), `test_desert_city_scenery_round_trip`, `test_desert_city_only_free_bytes`.
- [x] **Polygon objects**. Geometry mode 1 layout (normals present): two seed points, then 10-word records
  (attribute, normal, two points; units). Encoder `polyrom.encode_object`. In the polygon run; `oba` =
  `0x800000 | word index`. Golden: covered.
  Converter: the HUD outline (`build.hud_outline`, polygon run, at the start) and, with `scenery="model"`, the course's objects (`model.convert_model`: one object per cell, more past `model.MAX_RECORDS`; the polygon run's end). Rule: triangles and quads from the Dreamcast strips, corners anticlockwise about the side the Dreamcast draws (its cull mode decides), stored normal -Newell, links chosen greedily with move records where no exit reaches the next polygon (Desert City: 12,061 drawn polygons after thinning, 14,223 records). Dropped first: the alpha-blended decals, the sky domes, the shadow volumes and the run-time props at the origin (`build.kept_model`). Test: `test_model_desert_city`, `test_thin_keeps_road`, `test_desert_city_thinned`, `test_desert_city_frame_rate` (MAME).
- [x] **Actors**. `ACTOR_TABLES[id]` (window `0x00233bfc`) points at `u32[256]` actor-list pointers (0 = none); a
  list is a `u32` count, then 24-byte records: position (3 x f32, units), angle (s16), ident (u16), handler (u32, the
  init routine), `obj` (u32, handler data). Up to 65 actors are spawned (the count at data `0x02802f74`). Encoders
  `course.encode_actor_table`, `encode_actor_list`, `encode_actor`, and for the modelled handlers' data
  `encode_object_list`, `encode_billboard_lists` (with `encode_billboard_ref`: the billboard's reference point,
  `BILLBOARD_REF`, one for every billboard), `encode_switch_lists`. In the program run, the `ACTOR_TABLES` word
  rewritten in place. A course swapped into Beginner's slot must replace Beginner's 35 actors, or they draw
  Beginner's props at Beginner's positions. Golden: covered, except the handler data of unmodelled handlers.
  Converter: with `scenery="empty"` (and `"model"`) the actor table is the shared empty one, `build.EMPTY_ACTORS` (256 zero words at the program run's first 1,024 bytes, reserved; "Shared empty actor tables" under "Actors"), `ACTOR_TABLES[slot]` rewritten, so no actor spawns; `golden.ACTOR_FIELDS` counts an all-zero actor table as set. Test: `test_empty_scenery`, `test_encode_scenery_fields`, `test_empty_actor_table_shared`.
- [x] **Texture headers**. Four `u16` at `tha` (size, wrap, mirror, translucent, textured, luma table, sheet
  position, palette entry). Encoder `texture.encode_tex_header`. In the textures run. Golden: covered.
  Converter: `build.model_scenery` writes one header per drawn record (a move record reads the next drawn record's), one variant per (texture, mirror and clamp mode, translucency, face-colour tint) sharing its texture's texels: the texture's own palette entry and luma table (the pair nearest the Dreamcast's mean colour), `textures.tint_entry` for a tinted mesh, a flat header (`FLAT_HEADER`) in the palette entry nearest an untextured mesh's colour; in the textures run from its start. Desert City: 12,061 headers. Test: `test_model_scenery_modes`, `test_desert_city_scenery_round_trip`.
- [x] **Texture points**. Per record 3 or 4 (v, u), 1/8 texel, at each placement's `tpa` (`model2-textures.md`,
  "Texture points"). Decoder `texture.texture_points`, encoder `texture.encode_texture_points`; in the textures run.
  Golden: covered since Phase 7 (`golden.Course.points`). Converter: `textures.course_points` from the Dreamcast UVs
  (Phase 7, Task 4); written as golden's `points` with the model scenery (`build.MODEL_FIELDS`).
- [x] **Texels**. 4-bit indices in the course sheet (level 0, then the mip levels). Encoder
  `texture.with_texels(rs, h, indices, course, mips="decimate")` (or `"keep"`), rewritten in place in the course's
  1 MB block; `texture.encode_sheet_source` writes `0x15a0[id]` if the block moves. Golden: covered.
  Converter: `textures.course_textures`, written by `encode` -> `golden.tables` -> `texel_writes` (`write_texels` is used by tests only): every texture of the course's `_TEX.BIN` (Desert City: 86, at their own sizes, 52 % of the level-0 area) quantised to 4-bit indices with one hue and a luma table (the colour ruling: second colours are lost) and packed into sheet 0 outside the mip area; Beginner's 1 MB course block rewritten in place, the common block (sheet 1) and Expert's and Advanced's blocks never written. Test: `test_desert_city_only_free_bytes`, `test_beginner_textures_round_trip`.

Race values (`course.race_value_addresses(rs, entry, mode, difficulty)` gives each address):

- [x] **Checkpoint table**. `u32 count` (at most 30), then (`u16` segment, `u16` raw, 0 on the real set); the lap
  line is implicit. Encoder `course.encode_checkpoint_table`. Written where it fits, `CHECKPOINT_TABLES[id]` (data
  `0x02823c10`) repointed with `encode_table_pointer`. Golden: covered.
  Converter: `convert.tables.checkpoint_rows` (`exe.checkpoints(exe, n)`: n 3 = segments 136, 263), 12 bytes in `DATA_FREE`, `CHECKPOINT_TABLES[0]` repointed. Test: `test_checkpoint_rows`, `test_converted_round_trip`; MAME: `test_desert_city_drives` (the clock rises one frame after the player reaches 136, 263 and the lap line).
- [~] **Time tables**, per mode (0-2) and difficulty (0-3). `u32` words in ticks: the initial time, then one extension
  per pass; the race reads `1 + laps * (count + 1)` words. Encoder `course.encode_time_table`. Written where it fits,
  the per-difficulty `TIME_TABLES` word repointed. Golden: covered (the words the race reads).
  Converter: kept as the slot's (ruling; M6 sets them). n 3's 2 checkpoints make the race read 1 + laps x 3 words, past Beginner's stored tables in modes 1 and 2. The playtest set raises the start words to 12000 (`tools/desert_city_playtest.py`).
- [~] **Laps** per mode (`u32`, 1-127). Encoder `course.encode_laps`. In place. Golden: covered.
  Converter: kept as the slot's (M6).
- [~] **Car count** (`u32`, 1-44, `CAR_COUNTS` entry 0). Encoder `course.encode_car_count`. In place. Golden:
  covered; `CAR_COUNTS` entry 1 is not decoded.
  Converter: kept as the slot's (M6).
- [x] **Rolling start** (`s16` at `0x028050e8[id]`: 0 rolling, -1 standing). Encoder `course.encode_rolling_start`.
  In place. Golden: covered.
  Converter: `build.ROLLING_STARTS` = rolling on n 0 only, standing (-1) on n 1-7; a standing grid sits just behind the lap line, as n 3's does. Test: `test_standing_start`; MAME: `test_desert_city_drives`.
- [x] **Grid base** (`s16` segments at window `0x00230e48[id]`, 0-`0x7fff`). Encoder `course.encode_grid_base`. In
  place. Golden: covered.
  Converter: `build.GRID_BASES` from `1ST_READ.BIN` `0x17e814` (n 3: 458). Test: `test_grid_bases_from_exe`, `test_grid_bases_and_runs`.
- [~] **Shared by every course**: `GRID_LANES` (window `0x00231b30`, `0x180` and `0x580`) and the two spawn tables
  (44 records of 48 bytes, per car index: object lists, AI line, start, speed, kind). Encoders
  `course.encode_grid_lanes`, `course.encode_spawn_record`. Changing them changes every course. Golden: covered.
  `BILLBOARD_REF` (window `0x235bb4`, one point for every view-angle billboard) is shared the same way, but only
  Beginner's `Course` covers it.
  Converter: not changed (`GRID_LANES`, the spawn tables, `BILLBOARD_REF`, the car models; Expert and Advanced stay byte-identical). Test: `test_converted_leaves_other_courses`.
- [ ] **Time corrections**. Not covered: see the next section.
  Converter: kept as the slot's (read, not traced).

Screens:

- [~] **Backdrop** (word 0). An upload set of ten words (char list, palette list, eight segment addresses); segments
  of `u32` flags (`0x10000`), rows (1-58), columns (32), then `rows * 32` `u16` tiles (char `0x3000+`, colour `0x60+`,
  bit 15 clear); char blocks to char RAM `0x10e0000+` (up to 4,096 chars); palette entries from `0x600`. Encoders
  `tiles.encode_upload_set`, `encode_char_list`, `encode_char_block`, `encode_palette_list`,
  `encode_backdrop_segment`. Written in the data run, word 0 repointed. Golden: covered.
  Converter: Beginner's kept (the Dreamcast's screens are loading and menu pictures).
- [~] **Select panel, wheel zones, rankings**: M4 ("For M4" under "Select screen" and "Rankings"). A course swapped
  into Beginner's slot keeps Beginner's panel and ranking table. Golden: covered (the existing ones).
  Converter: Beginner's kept (M4).
- [x] **HUD map** ("HUD map"): the outline, a polygon object in world coordinates (flat polygons at y 0) named by a
  one-entry list at `HUD_MAP_LISTS[id]`; the place record (`HUD_MAP_PLACES[id]`: view translation, angle) and the
  zooms (`HUD_MAP_ZOOMS[id]`). Encoders `course.encode_hud_map`, `polyrom.encode_object`. The list inside the object
  library (above), the object in the polygon run, the per-id words rewritten in place. A course swapped into
  Beginner's slot without its own outline shows Beginner's map with its own cars' markers. Golden: covered (M1c).
  Converter: `build.hud_outline` (a polygon object in the polygon run: one quad per road point, merged to at most `OUTLINE_QUADS` = 64 where the road is straightest, `outline_points`, since Phase 7 Task 8: its budget word counts against the frame cap; Desert City's 465 road points give 64 quads, every road centre within 1.36 units of the outline's centre line) and `build.hud_frame` (place record and zooms); the list is Beginner's with its one entry naming the new outline. Test: `test_hud_outline`, `test_frame_of_the_new_course`; MAME: the outline shows Desert City with the player marker on it (`docs/spike/2026-11-convert.md`, "HUD map").

### Not covered by M1's golden test: explicit M3 work

"M1 green" (`tests/test_golden.py` passing) means that every table marked covered above re-emits byte for byte on all
three courses. It does **not** cover the tables below, and a new course needs each of them. None has a decoder or an
encoder yet (each decoder that exists raises `NotImplementedError` naming its discovery-log entry:
`course.time_corrections`), so each is M3 work, on top of the checklist. The ground grid
and attribute block (M1b) left this list in Phase 3: decoded with the TGP side by `tgp.decode_ground` and covered
(checklist, "Ground"); Phase 2's stubs for them (`course.ground_grid`, `course.ground_attributes`) are removed. The
HUD map left it in Phase 4 (M1c): decoded by `course.hud_map` and covered ("HUD map"; checklist), and with it the
read `0x12efc` (its place record). The per-segment flags left it in Phase 6 (M3a): decoded by `course.segment_flags`
and covered ("Per-segment flags"; checklist). The texture points left it in Phase 7 (M3b): decoded by
`texture.texture_points` and covered (checklist, "Texture points").

| Table | Where | What a new course needs | State |
|---|---|---|---|
| view-test and culling tables | view-test points `0x0286f7e0`, `0x0286f880`, `0x0286f920`, `0x0286f9c0` (one table per heading quadrant, shared); visit orders `0x17594`, `0x175ad` (program); Advanced's progress mask `0x175e0[id]`, `0x175f4` | nothing: the shared tables serve a new course inside -1024..1023 as they are. **Confirmed with scenery (Phase 7, Desert City):** read on every race frame of the player's lap (8,571 frames), the heading quadrant's table whatever the cell, the reference cell always a road cell (31 cells, rows 3-12, columns 5-10, all four quadrants); the points are whole cells from the reference cell and the masks name the 5 x 5 block, so no extent is Beginner-shaped (`culling`; `test_culling_real`, `test_desert_city_culling_and_camera`, `test_desert_city_scenery_reads`). Phase 6's "no read recorded" is contradicted: the M3a set (no scenery) reads them every frame too. What it implies: only the 5 x 5 block round the reference cell is drawn, and on the autoplay lap the reference cell was always a road cell, so scenery more than two cells from every road cell was not drawn there (Desert City: cells `0x08`, `0x09`, 10 budget words; a car off the road could reach them). The per-id progress mask must be empty or fit the new course. Beginner's and Expert's entry (`0x175f0`) is one record of 36 bytes whose segment range is (9999, 9999): `0x17688` walks records while the player's segment is past the range's end and uses none below its start, so it is empty for any course under 9,999 segments (traced on the converted n 3, Phase 6: read every frame, no mask) | view tests, visit orders and block offsets decoded (`culling`), shared, unchanged by the converter; the progress mask not decoded |
| time corrections | data `0x0280587c[id][difficulty]` (subtracted when `0x501a80` is 1) and `0x0280583c[id][difficulty]` (added when the player car's `+0x0` bit 28 is set) | one word per difficulty, or the original course's kept | read, not traced |
| course-dependent reads not tied to a table | program `0x65d4`-`0x6a34` (pages `0x2806b00` / `0x2807200` / `0x2807100`), `0xf548`, `0x1149c`, the window code `0x221b80`-`0x221c34` (`0x2850c00` / - / `0x2850d00`), the per-id table at window `0x23895c` (16-byte stride, read by `0x226c20` on the select screens) | traced on the converted n 3 (Phase 6, `docs/spike/2026-11-convert.md`, "Course-dependent reads"): none is shaped by Beginner's road. `0x65d4`-`0x6a34` plays the camera script `0x6c98[id]` (Beginner's `0x02806b78`: keyframes relative to the player's car; decoded, "Camera scripts"; Phase 7: on Desert City only slot 0's script is read, from frame 1328, looping every 3600 frames, and the fly-by frames the player's car in Desert City's scenery); `0xf548` and `0x1149c` index shared tables by the car's speed and by its distance to the player in segments (the page differs with the race, not the course); the window code is handler `0x221700`'s update (data at `0x2850cd0` / `0x2850d10`, fixed: the init at `0x221700` loads them, disassembly), reached only through the actor table, so it was not exercised on the converted course (Phase 7 confirms once actors return); `0x23895c[id]` is the select screens' palette animation (count, source rows, palette `0x01800c00`; colours; concluded from disassembly, not the trace, whose taps started at frame 1250, after the select screens) | identified, not decoded; extents unknown (`golden.NO_EXTENT`) |

`golden.EXCLUDED` also lists the unmodelled actor handlers' data (needed only if a new course uses those handlers),
the select screen's map objects in polygon ROM (M4), the race-value pointer levels (fixed), the spawn records past
the car count, the id-3 stub, the colour tables (shared; a new course whose hues are not in the polygon palette needs
an encoder for it) and the rankings beyond the table pointers (M4).

### What the mirror (M1c) must transform

The mirror is a reflection of world z (Phase 1: `docs/spike/2026-11-beginner-overlay.md`, "Mirror"), taken in the plane
z = -0.5 since Phase 4 (`mirror.PLANE`): a world point's z becomes `-1 - z` (rounded to f32), a direction's (a stored
normal) or an object-local point's `-z`. The game rounds world z before it takes a cell (the ground's `cell`, the
scenery grid's `0x172a0`, the view test's corner `0x1770c`), and `round(-1 - z) = -1 - round(z)`, so every mirrored
point lands in exactly the mirrored row of every grid, cell edges included (an f32 z of exactly k + 0.5 excepted).
Below, "z negated" on a world point means this. "Phase 1" rows come
from Phase 1's checks (the Dreamcast's `M` files against its normal ones) and MAME's culling code; "derived" rows
follow from the layouts in this document and are for M1c to check in MAME. Phase 4 implemented them in
`dtools/arcade/mirror.py` (one function per rule) and drove the mirrored Beginner in its slot against the original
(`docs/spike/2026-11-mirror.md`, "Symmetry", "Rules checked"): each row says "checked in MAME (Phase 4)" or gives the
rule as corrected there. The symmetric part of those runs ends at frame 1441, before GO, at asymmetric code (the
TGP's angle rounding), not at a table rule.

| Table | Transform | Basis |
|---|---|---|
| polygon objects (grid tiles, actor objects: `golden.scenery_objects(c)`) | the seed points and points mirrored as world points (`-1 - z`) when the object is drawn in world coordinates (323 of Beginner's 325) or as local points (`-z`) when a STATIC_PROP actor draws it under T(position) Ry(angle) (Beginner's objects 875514 and 875859; `mirror._local_objects` takes this from the actors, since `golden`'s lists all carry `in_world` True); stored normals `-z`; keep the corner order and the attributes; never set attribute bit 17 | Phase 1; the plane Phase 4 |
| car models (the objects the spawn records' lists reach: `Course.car_objects`, 213 / 173 / 133 objects on Beginner / Expert / Advanced) | not mirrored: the spawn tables are shared by every course, per car index, so these objects are the other courses' cars too (every object Beginner shares with Expert or Advanced is one). `Course.objects` holds them beside the scenery; transform `scenery_objects` only | derived |
| cell grid | row `r` becomes `15 - r` (the row comes from `(round(z) + 1024) >> 7`, `0x172a0`); the column stays; each cell's list moves with it. In the plane z = -0.5 a z on a cell edge lands in the mirrored row too | derived; checked in MAME (Phase 4): the mirror draws the original's objects in the mirrored cells |
| object-list entries | unchanged: `oba`, `tpa`, `tha` and `budget` name the mirrored objects | derived |
| actors | position z negated; angle becomes `-angle` (the object is drawn under T(position) x Ry(angle), and negating z turns Ry(a) into Ry(-a)), kept in s16; the game recomputes each actor's cell from its position. **The unmodelled handlers' objects are not mirrored** (31 of Beginner's 35 actors: 15 x `0x2201bc`, 10 x `0x221700`, 6 others): their handler data is not decoded, so each draws the original's object under its mirrored position and angle; in which space each handler draws is not traced (`docs/follow-ups/phase-04-m1c.md`, "What M4 starts from") | derived (not compared in MAME: no actor position is logged, no snapshot covers the unmodelled actors) |
| view-angle billboards (`Course.billboards`, `Course.billboard_ref`) | the 64 variants in reverse order (variant `k` becomes `63 - k`, `mirror.mirror_billboard_lists`) and the reference point `BILLBOARD_REF` mirrored as a world point ((210, 0, -380) to (210, 0, 379)): the update `0x221e60` shows variant `((-a) >> 5) & 63` of the camera's angle `a = atan2(dz, dx)` from that point, which the mirror negates; the variants' objects are mirrored with the scenery. The reference point is read by absolute address (one for every billboard; only Beginner has one). The animated billboard (37 variants, an animation, not a view angle) is left as it is | checked in MAME (Phase 4): with both, the mirror shows variant `63 - k` of what the original shows from the mirrored camera on 775 of 813 frames and one off on the rest (angle rounding at a variant edge); with the reference point kept, the variants scatter (`docs/spike/2026-11-mirror.md`, "Billboards") |
| road line | positions z negated; lane slots reversed (slot `k`'s position is slot `7 - k`'s, so `lane_lo`, `lane_hi` become `7 - lane_hi`, `7 - lane_lo`), since lane numbers grow to the left of travel; **each slot's pitch, heading and roll come from slot `8 - k`** (slot `lane_hi`'s on the new `lane_lo`): `0x140f0` hands a car at lane `k.f` the attitude of slot `k + 1`, not blended (`0x14334`-`0x14340`), and the mirrored lane reads slot `7 - k`. The original's `lane_lo` attitude (read only by a car below `lane_lo`) has no slot in the mirror. Heading `h` becomes `0x2000 - h` kept in s16 (the same yaw `4h -> 0x8000 - 4h` as mod `0x4000`, without the mask's jumps between neighbours: the stored headings span more than `0x4000`); roll negated; pitch kept; `curve` negated | derived; corrected in MAME (Phase 4): the attitude's slot (with the reversal alone a car at lane `0x580` on segment 161 leaned by the neighbour slot's pitch and roll, -297 / 4557 against -64 / -3039); positions, heading, roll sign checked in MAME (Phase 4) |
| AI lines | positions z negated; **`lane` becomes `0x6ff - lane`**: lane `k.f` is placed at `k + f / 255` (`0x140f0` blends by data `0x02871c80[f]` = `f / 255`), whose mirror is `(6 - k) + (255 - f) / 255`; heading as on the road line; roll and `steer` negated; pitch, `speed` and `step` kept. The Dreamcast's mirrored lines are up to 10 units off an exact mirror (Phase 1) *(corrected in Phase 5: Phase 1 paired the wrong lines; `M` line k is plain line 7 - k on courses 0-3 and (k + 1) mod 8 on 4-7, positions exact in the plane z = 0, one unit from this table's z = -0.5, course 4 up to 0.81 off; lane `0x700 - lane` on courses 0-2; `dreamcast-disc.md`, "Mirror files")*, so they check this only roughly | derived; corrected in MAME (Phase 4): `0x700 - lane` put every car 3 / 255 = 0.0118 units off its mirrored place, to the same side of travel; heading, roll and steer signs checked in MAME (Phase 4) |
| step table, checkpoints, time tables, laps, car count, rolling start, grid base | unchanged: the direction of travel and the segment numbering stay | derived; checked in MAME (Phase 4): laps and the lap line's extension as on the original |
| grid lanes | `0x180`, `0x580` become **`0x57f`, `0x17f`** (`0x6ff - lane`, as the AI lines). `GRID_LANES` is shared by all courses: **the M1c build swaps it for every course**, so Expert and Advanced start on swapped grid sides in that build (`mirror.mirrored_set`); a mirrored entry beside its original needs a per-course value (a code patch, M4) | derived; corrected in MAME (Phase 4): with `0x6ff - lane` 39 of the 40 cars stand exactly on their mirrored places at frame 1330; car 18's standing place lies on a shared road-polygon edge, the TGP answers the neighbour polygon and the step table puts it one segment behind (`docs/spike/2026-11-mirror.md`, "Car 18") |
| spawn records | unchanged (shared, per car index) | derived; checked in MAME (Phase 4) |
| ground grid, attribute block, TGP polygons and lists | M1b (`docs/formats/model2-tgp-ground.md`, section 10): the polygons' z negated on corners and centre (y, x unchanged). The area's sign then flips to positive, but the TGP's edge test is one-sided and needs the originals' negative area, so the corner order is reversed: quads `(0, 3, 2, 1)` (keeping the 0-2 diagonal the height test splits along), triangles (corner 3 = corner 2) `(1, 0, 3, 2)` (keeping the repeat at 2-3 and corner 2 as the solve's apex). Polygon order, the attribute bytes and every list (and the list block) are unchanged (the road stays first, in step order). The grid's rows reverse (`r` -> `127 - r`): in the plane z = -0.5 that is exact at the row edges too, so nothing is rebuilt (about z = 0 the edge rows' lists had to be: Beginner 108 cells, 122 entries more, which might not fit Expert). Direction word (polygon word 15, s16, 1/65536 turn) becomes `0x8000 - d` (kept in s16) | derived; checked in MAME (Phase 4): every ground call of both runs answered, heights equal wherever the calls pair; the direction word is read on Beginner (the standing cars' yaw: with it unchanged they face the original's direction) and `0x8000 - d` turns them to the mirror's |
| per-segment flags | unchanged (per segment: a tunnel's and the AI's segments stay). The tunnel light zones (window `0x23074a`, "Per-segment flags") hold world positions and are not mirrored: Beginner never sets the tunnel bit, so they are not used on it | derived |
| culling | the view-test tables and the first visit order are shared and geometric; Advanced's progress mask and its second visit order (taken at cells 72 and 88) name cells, which move with the rows | derived; **contradicted in MAME (Phase 4)**: the mirror draws cells the original culls (up to 3,579 polygons in one list against the bar's 3,533); the four view-test tables are mirror-symmetric and the camera yaw is exactly mirrored, the cause is not found (`docs/spike/2026-11-mirror.md`, "Rules checked") |
| texture headers, texture points, texels | unchanged; lettering comes out mirrored (accepted, spec decision 8) | derived |
| backdrop (`Course.backdrop`, `Course.char_blocks`) | column `q` of the panorama is the original's column `(189 - q) & 255` (`mirror.MIRROR_COLUMN`: the scroll is the camera yaw, which the mirror turns to `0x8000 - yaw`), each char flipped left to right. System 24 tiles have no flip bit and a char's palette follows from its index, so a flip is a char of its own in the same 128-char block: symmetric chars kept, a flip that is another char of the block reuses it, the others new (Beginner: 3,312 of 3,327, at most 4,096 from `0x3000`). The mirrored char RAM goes into new char blocks in the data ROM's free run (106,476 bytes on Beginner), at the run's first span that is free (`0xff`) or the course's own current blocks (its start on an unpatched set), one per upload of the original; the char list points at them, the original blocks stay ("Backdrop") | checked in MAME (Phase 4): at GO + 5 s the original's sky pixels above the horizon match the mirror's flipped 0.920 (0.564 with the backdrop unmirrored, 0.092 unflipped); the best match is 1 pixel off, the mirror's yaw being 31 units short of the exact mirror at that frame |
| select screen, rankings | M4: the MIRROR marking and an own ranking table per entry (spec decision 19) | spec; done in Phase 9 ("The sixteen-entry select", "Rankings") |
| HUD map (`course.hud_map`: outline object, list, place record, zooms) | the outline's points mirrored as world points (`-1 - z`), its normals `-z`, corner order and attributes kept (`mirror.mirror_hud_map`); the list, place record and zooms unchanged: the cars' markers are drawn in the outline's frame at the cars' world (x, z), so they follow the mirrored cars. The outline is reached from no other table (mirrored once; Beginner's is also the id-3 stub's). World z is the screen's horizontal axis: the map is flipped left to right about the screen x of z = -0.5 (x 425), and its outline, whose z extent is not centred on the plane, moves 6 px left (x 401-455 to 395-449) | checked in MAME (Phase 4): at GO + 5 s the outline's pixels over (385, 176, 466, 284) match the original's flipped (0.996 in the discovery run) and the markers sit on the mirrored road (`docs/spike/2026-11-mirror.md`, "HUD map") |

Two things the M1c build changes are shared with courses it does not mirror (`mirror.mirrored_set`):

- `GRID_LANES` (`0x00231b30`) is read for every course, so Expert and Advanced start on swapped grid sides in that
  build; `verify_lossless` on courses 1 and 2 reports exactly this span.
- `BILLBOARD_REF` (program window `0x235bb4`, 12 bytes) is read, by absolute address, only by the view-angle billboard
  handler, which Expert and Advanced do not use. **`verify_lossless` on courses 1 and 2 does not show it**: only
  Beginner's `Course.billboard_ref` covers the bytes. A later course with a view-angle billboard reads the mirrored
  point unless the point is made per course: on a widened set (`course.widen_ids`, Phase 9) both it and `GRID_LANES`
  are per id, and a mirror beside its original writes its own entries.

## Unknowns

Where the sources disagree:

- **The race timer's unit.** The timer at `0x005010d0` is traced counting down one per frame (`0x4010`-`0x4028`),
  and the HUD shows `ticks >> 6` (`0x3b04`). `TICKS_PER_SECOND` = 64 is inferred from the shift, and this document
  writes "3,200 (50 s)" and "ticks of 1/64 s" on that basis. The screen runs at 57.52 Hz, so one tick per frame makes
  3,200 ticks 55.6 s of real time. The rankings' race time (`0x5010cc`) is also "one tick per frame" (code comment:
  frames). Neither reading was checked on the HUD.
- **The second SRAM settings area's writer.** "Rankings" above names `0x2270d0` for both settings areas
  (`0x01d00b60`-`0x01d00ba3` and `0x01d00be0`-`0x01d00c27`); `course.rankings_layout()` labels the second one
  `0x2271fc`.

Not decoded or not traced (the runs behind each: `docs/spike/2026-11-arcade-course-tables.md`, the "Not found"
entries of each section):

- Course table: record words 6-7 (no reader).
- Course-dependent data-ROM reads not tied to any table here ("Per-course blobs found"): `0x65d4`-`0x6a34` (pages
  `0x2806b00` / `0x2807200` / `0x2807100`), `0xf548`, `0x1149c`, the window code `0x221b80`-`0x221c34`
  (`0x2850c00` / - / `0x2850d00`), and the per-id table at window `0x23895c` (16-byte stride, read by `0x226c20` on
  the select screens). Identified in Phase 6 ("Not covered by M1's golden test"), not decoded: the palette animation's
  layout, handler `0x221700`'s data (the camera script's layout is decoded since Phase 7: "Camera scripts"; its
  `roll`, `fov` and `rest` words are named after their use).
- Scenery: the 31 / 63 / 32 actors of unmodelled handlers and their data; the switch's state rule; the view test's
  TGP functions `0x1d` / `0x1e` (the wedge reading is inferred; the tables are decoded and their reads confirmed with
  scenery in Phase 7); Advanced's progress mask and second visit order. (The
  billboard's angle function was traced in Phase 4: `atan2(dz, dx)` of the camera from `BILLBOARD_REF`.)
- Road and AI lines: `RoadPoint.curve` (unit), `AiPoint.speed` and `AiPoint.steer` (named after their use),
  `Lane.pad` (no reader); the road from kerb to kerb (the ground polygons, M1b).
- Checkpoints and grid: the time corrections; what sets the mode `0x5fe5e0` and the difficulty `0x5fe59c`; the spawn
  table choice `0x5fe588`, `CAR_COUNTS` entry 1 and the link path of `0xeb00`; the spawn `kind` and `speed`
  (named after their use); the checkpoint `raw` half-word (0, no reader).
- Pit: the pit lane's geometry (TGP side), the readers of `STEP_PIT` and `STEP_PIT_LANE`; the meaning of the trigger's
  `+0xd2`, `+0x9c`, `+0x9d`, `+0xd8`, `+0xf0` (all 0 in the driven stops). The stop itself was driven in v0.1.2 ("The
  pit stop and its stalls"). (The `0x100` step flag on Advanced is its tunnel since Phase 6: "Per-segment
  flags".)
- Per-segment flags: the meanings of bits 8 (`SEGMENT_TUNNEL`, a tunnel) and 10 (`SEGMENT_AI_FREE`, the AI keeps
  its line) are inferred from their readers, not checked in a run that sets them; no Dreamcast source for Expert's (the converter writes zeros).
- Textures: what sets `0x5fe5e2` (the colour-translation variant); the four
  palette entries rewritten at run time; headers and points in the rasterizer's texture RAM (bit 23); the mip filter
  (`"decimate"` is our choice); header bit 11 of h2 (set on every textured header, meaning unknown).
- Select screen: what sets the variant `0x5fe59a` and the mode `0x5fe5e0`; the cursor words' use; how `0x501050`
  comes from the paddle; map-object words 4-7; the geometry mode of the map objects on Circuit Select.
- Backdrop: segment word 0 (`0x10000`, no reader); the layer-3 split. (The scroll's source, the camera yaw, was traced
  in Phase 4: "Backdrop".)
- Rankings: the name store `0x228490` (no name entered in a run); the `view` and `flags` sources; the Expert and
  Advanced tables were not written in a run.
- HUD map ("HUD map"): what changes the zoom level (HUD struct `+0x4a`, 1 from the set-up; no writer traced);
  the place record's pad (0, no reader); the marker routines' choice (`0x5fe588`, `0x540027`) and the marker lists
  were read, not traced. The outline's texture points (flat polygons: read by the rasterizer, never sampled) are
  covered with every `tpa` since Phase 7.
