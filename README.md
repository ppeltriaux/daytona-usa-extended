# Daytona USA Extended

Daytona USA Extended is the 1994 Sega Model 2 arcade *Daytona USA* with eight courses: its own three, byte for byte
(Three Seven Speedway, Dinosaur Canyon, Sea-Side Street Galaxy) and five converted from the Dreamcast's *Daytona USA*
(2001): Desert City, National Park Speedway, Circuit Pixie, Rin Rin Rink and Mermaid Lake. The five race on the arcade's
own engine in MAME: its field of cars, its checkpoints, its render rate. The project page:
https://continuearcade.com/daytona-usa-extended/

This repository is a converter: one command builds the game on your machine from your own copies of the arcade game and
of the Dreamcast disc. It holds none of either (see "Legal").

> **v0.1 is a very early release, still under development.** All eight courses load and race on the arcade's engine,
> but time limits, laps and opponents are not tuned yet and the new courses have rendering defects: see "Known
> issues".

## What you need

- **Your own copy of the arcade game:** MAME 0.289's `daytona` ROM set, `daytona.zip` (its 36 files are checked by size
  and CRC32 against MAME 0.289's list; a merged set's clone folders are ignored) with `model1io.zip` beside it.
- **Your own copy of the Dreamcast game:** the US *Daytona USA* disc (`Daytona USA (USA)`, MK-51037) as the Redump `.7z`,
  or its `.cue` with the track `.bin` files beside it. Other releases and regions are refused.
- **Python 3.12 or newer** (numpy is installed by `pip`).
- **On Linux, libarchive's `bsdtar`** to read the `.7z` (Debian/Ubuntu: `sudo apt install libarchive-tools`; macOS
  has it), or pass the disc's `.cue` instead.
- **The custom MAME build.** The new courses outgrow the arcade's ROM regions, so **stock MAME 0.289 will not run the
  set.** `tools/build-mame.sh` clones MAME 0.289, applies `tools/mame-daytona-rom-space.patch` and builds
  `build/mame/model2`; it takes 25 to 60 minutes from scratch (about 25 on a 12-core Mac), once. It needs macOS with
  Homebrew's `sdl3` and `pkgconf`, git and the Xcode command-line tools. On Linux, apply
  `tools/mame-daytona-rom-space.patch` to MAME 0.289 (git tag `mame0289`) and build it yourself (`SUBTARGET=model2
  SOURCES=src/mame/sega/model2.cpp`, as the script does); this is untested, `tools/build-mame.sh` is macOS-only.

None of the game is included here, linked to, or sent on request.

## How to use it

```sh
git clone https://github.com/ppeltriaux/daytona-usa-extended.git
cd daytona-usa-extended
python3 -m venv .venv && .venv/bin/pip install -e .
tools/build-mame.sh       # once: 25-60 minutes
.venv/bin/dtools check --roms ~/roms/daytona --disc ~/dc/"Daytona USA (USA)-Dreamcast.7z" --out ~/daytona-extended
.venv/bin/dtools build --roms ~/roms/daytona --disc ~/dc/"Daytona USA (USA)-Dreamcast.7z" --out ~/daytona-extended
build/mame/model2 daytona -rompath ~/daytona-extended -nvram_directory ~/daytona-extended/nvram
```

- `--roms` is the folder holding `daytona.zip` and `model1io.zip`; `--disc` the Redump `.7z` or the `.cue`; `--out` a new
  folder (it must not be the ROM folder: your files are only read, never overwritten). `--roms` and `--disc` default to
  `DAYTONA_ROMDIR` and `DAYTONA_DC`.
- `dtools check` names anything wrong with the inputs (a missing file, a file with another CRC32 than MAME 0.289's, a
  disc of another release) and `dtools build` will not start until it passes. `check` creates `--out` and, for a `.7z`,
  extracts the disc's track 3 under `<out>/work` (about 1.2 GB, reused by `build`).
- `dtools build` writes `daytona.zip` (the patched set), copies `model1io.zip` beside it, and writes
  `build-report.txt`: the commit, the Python and MAME versions tested, the input checks, the sha1 of every file in the
  zip, and the MAME command to run. The first build took 407 s on an Apple silicon Mac (the disc's extraction and the
  re-meshes of three courses); a rebuild into the same `--out` took 105 s.
- Run MAME from the repository root, and **keep `-nvram_directory`**: MAME's default `nvram/daytona` belongs to the
  stock game, and this set's first boot resets the backup RAM (its rankings have a new layout), which would wipe the
  stock game's best times. MAME shows a ROM warning once per launch (the five `dx-*.bin` files have no known good
  dump): press a key and the game runs.
- Exit codes: 0 done, 1 an input is wrong, 2 a usage error.

**In the game:** the attract logo reads DAYTONA USA with "Extended" under DAYTONA. Circuit Select scrolls through eight
courses as you turn the wheel: THREE SEVEN SPEEDWAY, DINOSAUR CANYON, SEA-SIDE STREET GALAXY, then DESERT CITY,
NATIONAL PARK SPEEDWAY, CIRCUIT PIXIE, RIN RIN RINK and MERMAID LAKE. Every panel shows its course's name; the five new
ones also show their laps for the game mode and a painted map of the course.

## Known issues

- **Time limits are not tuned:** courses 3 to 7 (the five new ones) start with a long clock (12,000 ticks) so each can
  be finished; their laps and opponents are not tuned either. The arcade's three keep their own.
- **Two courses draw a shorter view:** Circuit Pixie and Rin Rin Rink draw less far ahead (far scenery appears a little
  later), so they stay within the arcade's polygon budget.
- **Mermaid Lake** was raised 41 units to fit the arcade's ground; an invisible lift is possible.
- **Rendering defects on the five new courses:** the ground is not drawn (black) under the car for a few frames under
  bridges and in Mermaid Lake's tunnel, a flat grey frame appears now and then, and thin black seams cross the road
  (seen on National Park Speedway and Circuit Pixie).
- **Tested only in MAME:** on macOS on Apple silicon only. Nothing was tried on real hardware. `dtools build` was
  tested on macOS on Apple silicon (Python 3.15.0, numpy 2.5.4) and on Linux x86-64, emulated (Docker `linux/amd64`
  `python:3.12-slim` on an Apple silicon host; Python 3.12.16, numpy 2.5.4): both write the same `daytona.zip`
  members, byte for byte. `tools/build-mame.sh` is for macOS.

If `dtools check` fails it prints one `problem:` line per fault naming the file and the expected value.

## Tests

```sh
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

The offline tests run as they are. The tests that need your ROM set or your disc (`DAYTONA_ROMDIR`, `DAYTONA_DC`) and
the ones that run MAME skip without them and say so. The whole suite takes well over an hour with the inputs (MAME
runs one at a time). Docstrings and the format notes cite the project's working notes (`docs/spike/`,
`docs/follow-ups/`, `out/`), which are not published.

## Licence

- The converter and its tests and notes (`dtools/`, `tests/`, `docs/formats/`) are under the MIT licence (`LICENSE`).
- `tools/build-mame.sh` and `tools/mame-daytona-rom-space.patch` are changes to MAME and fall under MAME's licence,
  GPL-2.0 (https://github.com/mamedev/mame, COPYING).

## Legal

This is a personal, non-commercial technical study; nothing is sold.

- *Daytona USA* and its Dreamcast version, with their code, art, sound and courses, belong to Sega. This project is not
  affiliated with or endorsed by Sega; the names are used only to say which games it concerns.
- No ROM set, disc image, patched set or extracted game data is offered here or anywhere else, none is linked to, and
  nothing will be sent on request. Build it from your own lawfully obtained copies.
- If you hold rights in the games and object to anything here, open an issue and it will be taken down promptly.
