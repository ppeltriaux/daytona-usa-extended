# Daytona USA Extended

Daytona USA Extended is the 1994 Sega Model 2 arcade *Daytona USA* with eight courses: its own three, byte for byte
(Three Seven Speedway, Dinosaur Canyon, Sea-Side Street Galaxy) and five converted from the Dreamcast's *Daytona USA*
(2001): Desert City, National Park Speedway, Circuit Pixie, Rin Rin Rink and Mermaid Lake. Each of the eight also runs
in reverse. The five race on the arcade's own engine in MAME: its field of cars, its checkpoints. Since v0.3 they draw
further and in more detail than the arcade board could (see "Changes"). The project page:
https://continuearcade.com/daytona-usa-extended/

This repository is a converter: one command builds the game on your machine from your own copies of the arcade game and
of the Dreamcast disc. It holds none of either (see "Legal").

> **v0.3 is an early release, still under development.** All eight courses and their reverses load and race on the
> arcade's engine, but laps and opponents are not tuned yet and the new courses have rendering defects: see "Known
> issues".

## What you need

- **Your own copy of the arcade game:** MAME 0.289's `daytona` ROM set, `daytona.zip` (its 36 files are checked by size
  and CRC32 against MAME 0.289's list; a merged set's clone folders are ignored) with `model1io.zip` beside it.
- **Your own copy of the Dreamcast game:** the US *Daytona USA* disc (`Daytona USA (USA)`, MK-51037) as the Redump `.7z`,
  or its `.cue` with the track `.bin` files beside it. Other releases and regions are refused.
- **Python 3.12 or newer** (numpy is installed by `pip`).
- **On Windows:** Python 3.12 or newer from python.org (tick "Add python.exe to PATH"), and the prebuilt MAME from this
  repository's Releases page (https://github.com/ppeltriaux/daytona-usa-extended/releases; the latest release: v0.3
  needs v0.3's MAME): `daytona-usa-extended-mame-0.289-win64.zip` (`mame.exe`, linked statically so it needs no DLLs, MAME's
  licences, the patch and a README.txt). Unzip it anywhere. The `.7z` works as on the other systems: no `bsdtar` is
  needed. Windows' own `tar.exe` is tried first (current Windows reads the `.7z`), then py7zr (installed with
  `dtools` on Python 3.12 to 3.14). On Python 3.15 there is no py7zr yet, so the `.7z` is read by `tar.exe`; if your
  Windows cannot, extract it with 7-Zip and pass the `.cue` instead.
- **On Linux, libarchive's `bsdtar`** to read the `.7z` (Debian/Ubuntu: `sudo apt install libarchive-tools`; macOS
  has it), or pass the disc's `.cue` instead.
- **The custom MAME build** (Windows: the zip above, nothing to build). The new courses outgrow the arcade's ROM regions, so **stock MAME 0.289 will not run the
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
- On macOS and Linux, run MAME from the repository root. On Windows, run it from the unzipped MAME folder (`mame.exe`
  is there; the command names `.\mame.exe`, which PowerShell and cmd both run from the current folder): `cd` there first. Keep the output and MAME folders free
  of `"`, `%`, `$` and backticks (`dtools` refuses them in the output folder). And **keep `-nvram_directory`**: MAME's default `nvram/daytona` belongs to the
  stock game, and this set's first boot resets the backup RAM (its rankings have a new layout), which would wipe the
  stock game's best times. MAME shows a ROM warning once per launch (the five `dx-*.bin` files have no known good
  dump): press a key and the game runs.
- **On Windows** (PowerShell, from the repository folder; the exe is unsigned, so SmartScreen may warn on the first
  launch: click "More info", then "Run anyway"):

  ```powershell
  py -m venv .venv
  .venv\Scripts\pip install -e .
  .venv\Scripts\dtools build --roms C:\roms\daytona --disc "C:\dc\Daytona USA (USA)-Dreamcast.7z" --out C:\daytona-extended
  ```

  then run the command `dtools build` prints (it begins `.\mame.exe daytona -rompath "..." -nvram_directory
  "...\nvram"`) from the folder you unzipped the MAME zip into. `dtools check` takes the same flags.
- Exit codes: 0 done, 1 an input is wrong, 2 a usage error.

**In the game:** the attract logo reads DAYTONA USA with "Extended" under DAYTONA. Circuit Select scrolls through eight
courses as you turn the wheel: THREE SEVEN SPEEDWAY, DINOSAUR CANYON, SEA-SIDE STREET GALAXY, then DESERT CITY,
NATIONAL PARK SPEEDWAY, CIRCUIT PIXIE, RIN RIN RINK and MERMAID LAKE. Every panel shows its course's name; the five new
ones also show their laps for the game mode and a map of the course. After Circuit Select a Course Direction screen
shows the course's map twice: turn the wheel to the right for REVERSE (the red arrow) or leave it for NORMAL (the
white one), then press the gas; REVERSE runs the course the other way round, with its own time limits, pit stalls and
best times.

## Changes

- **v0.3:** the five Dreamcast courses and their reverses draw further and cleaner. They show two more rows of
  scenery cells ahead and one more to each side, and always the whole block round the car (the arcade opens at most
  15 cells a frame; these courses now draw 45), with the frame's polygon limit raised from 5,000 to 28,000 for them
  and their models no longer thinned: up to about 13,000 polygons a frame on Desert City. Their scenery is drawn with
  a per-pixel depth test, as on the Dreamcast, instead of the Model 2's polygon sorting: on Rin Rin Rink the grass no
  longer paints over the road on the bridge and the trackside panels' feet no longer show through the road. The depth
  test is a change to MAME's Model 2 renderer, used only by polygons that ask for it; the arcade's three courses and
  their reverses do not, and race and draw as before. **v0.3 needs v0.3's MAME:** rebuild it with
  `tools/build-mame.sh` (or take v0.3's Windows zip), and rebuild the set; with an older patched MAME the set runs
  but without the depth test. This goes past what a real Model 2 board could draw: v0.3 is for MAME. Best times are
  kept.
- **v0.2:** every course in reverse. A Course Direction screen after Circuit Select picks NORMAL or REVERSE; the eight
  courses in reverse (the arcade's three included) race with their own time limits (measured, as the five converted
  courses', against the computer-driven field), the Dreamcast's reverse pit stalls and their own best-time tables. The
  arcade's pit stop compared angles without wrapping, so on Dinosaur Canyon and Desert City in reverse the stop failed;
  it is patched. Circuit Select's maps for the five Dreamcast courses are drawn as a faceted landscape of each course's
  own ground. The road no longer vanishes for a moment at a few spots on the Dreamcast courses (the game's culling
  dropped the cell the camera still saw; the road is now filed in the neighbouring cells too). v0.2 changes the set,
  so rebuild; best times from v0.1.x are kept.
- **v0.1.2:** Mermaid Lake and Circuit Pixie no longer draw the Dreamcast's road shadows as black bands under the start
  gantry, the bridge and the checkpoint arch; the race is unchanged. Mermaid Lake's road no longer vanishes around the
  car while the camera comes down onto it before the start. Pit stops work on courses 3 to 7: the car is guided to its
  stall for the pit work, as on the arcade's own courses. Windows: a prebuilt, patched MAME 0.289 for 64-bit Windows
  (`daytona-usa-extended-mame-0.289-win64.zip`, built by this repository's GitHub Actions workflow, with the matching
  source `daytona-usa-extended-mame-0.289-src.tar.gz`), and `dtools build` reads the `.7z` on Windows. Windows builds the same
  set as macOS and Linux. Courses 3 to 7 race against a real clock: a start time and checkpoint extensions of their
  own in every game mode and difficulty, as generous as the arcade's three courses measured against the
  computer-driven field (the long clock is gone). v0.1.2 changes the set (the shadow decal, Mermaid Lake's scenery,
  the pit stalls and the times), so rebuild.
- **v0.1.1:** National Park Speedway and Rin Rin Rink no longer draw some of the Dreamcast's shadow volumes as walls
  (a grey slab behind National Park Speedway's start gantry, grey walls beside its pits and a dark band late in the
  lap; a black wall on Rin Rin Rink). The race is unchanged; both courses draw fewer polygons.
- **v0.1:** the first release.

## Known issues

- **Time limits are measured, not played in:** courses 3 to 7 and all eight reverses take their times from the
  computer-driven field's pace, with the arcade's own courses' slack; the normal race at the default difficulty was driven in MAME, the long races
  and other difficulties are derived. Their laps and opponents are not tuned. The arcade's three keep their own.
- **Beyond the arcade board:** since v0.3 the five new courses and their reverses draw up to about 13,000 polygons
  a frame and use a depth test the Model 2 does not have. MAME does not model the board's drawing time, so the game
  holds its rate there; a real board would not. Animated scenery (flags, banners) still appears at the arcade's
  distance, later than the scenery around it.
- **Mermaid Lake** was raised 41 units to fit the arcade's ground; an invisible lift is possible.
- **Rendering defects on the five new courses:** these were seen on v0.2 and have not been looked at again on v0.3,
  which draws these courses differently: one flat grey frame about a second after the start, thin black seams across
  the road (National Park Speedway and Circuit Pixie), and the background showing for a moment beside the road on
  National Park Speedway and at the edge of the ground when you run wide. v0.3's depth test was checked on Rin Rin
  Rink only. On Three Seven Speedway in reverse the background still shows once a lap (the arcade's own course: not
  changed by v0.3).
- **Windows:** built and tested in GitHub Actions on Windows Server (`windows-latest`); the set has not yet been built
  and played on a Windows PC.
- **Tested only in MAME:** the set was played on macOS on Apple silicon only. Nothing was tried on real hardware. `dtools build` was
  tested on macOS on Apple silicon (Python 3.15.0, numpy 2.5.4) and on Linux x86-64, emulated (Docker `linux/amd64`
  `python:3.12-slim` on an Apple silicon host; Python 3.12.16, numpy 2.5.4): both write the same `daytona.zip`
  members, byte for byte. On Windows the offline tests ran in GitHub Actions (`windows-latest`); the build of the set
  itself has not yet been run on Windows. `tools/build-mame.sh` is for macOS.

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
