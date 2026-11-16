"""Run MAME headless on the `daytona` set with a Lua autoboot script; parse the render-rate log.

Adapted from vrtools/mame/runner.py (the virtua-racing project).
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

LUA_DIR = Path(__file__).resolve().parent / "lua"
MAME_WAIT = 600.0               # seconds run_mame waits for another MAME to end before giving up
MAME_POLL = 5.0                 # seconds between checks


class MameError(Exception):
    pass


MAME_NAMES = ("mame", "model2")
"""Process names of a MAME: the installed one and the custom build (tools/build-mame.sh, SUBTARGET=model2)."""


def running_mame() -> list[str]:
    """`pid command` of every running process named exactly one of MAME_NAMES (`pgrep -x` each, then `ps` for the
    command lines); [] when there is none or pgrep is missing. MAME runs one instance at a time on this machine: a
    second one exits -13 (probably the comm-board socket), and other projects run MAME too."""
    pids: list[str] = []
    try:
        for name in MAME_NAMES:
            pids += subprocess.run(["pgrep", "-x", name], capture_output=True, text=True).stdout.split()
    except FileNotFoundError:
        return []
    if not pids:
        return []
    try:
        ps = subprocess.run(["ps", "-o", "pid=,command=", "-p", ",".join(pids)], capture_output=True, text=True)
        lines = [ln.strip() for ln in ps.stdout.splitlines() if ln.strip()]
    except FileNotFoundError:
        lines = []
    return lines or pids


def wait_for_other_mame(timeout: float = MAME_WAIT, poll: float = MAME_POLL, *, sleep=None, clock=None) -> None:
    """Return once no MAME process runs, checking every `poll` seconds; MameError naming the other process(es) when
    one still runs after `timeout` seconds. `sleep` and `clock` default to time.sleep and time.monotonic."""
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    deadline = clock() + timeout
    while others := running_mame():
        left = deadline - clock()
        if left <= 0:
            raise MameError(f"another MAME is still running after {timeout:.0f} s (MAME runs one at a time; a second "
                            f"instance exits -13); not started. Running: " + "; ".join(others))
        sleep(min(poll, left))


def run_mame(script: Path, seconds: int, workdir: Path, rom_dir: Path, env: dict[str, str] | None = None,
             mame: str | Path = "mame", system: str = "daytona", wait: float = MAME_WAIT,
             extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Run `system` for `seconds` emulated seconds with the Lua `script`, MAME's cwd, snapshot, NVRAM and cfg
    directories inside `workdir`; `extra_args` go after the others (the Dreamcast's `-cdrom <image>`). `env` is
    merged into the environment, with DAYTONA_LUA_DIR set to LUA_DIR.
    MameError (with MAME's output) when it exits non-zero or does not print "Average speed", and when a Lua
    callback failed: MAME prints errors inside frame or tap callbacks as `[LUA ERROR] ...` and runs on (only
    load-time and top-level script errors are fatal), so such a run would otherwise pass with empty or partial
    logs.

    `mame` is the binary (a name on PATH or a path): `config.custom_mame()` for the custom build, whose None (not
    built) is a MameError here, as is anything that is not a str or Path; MAME is not started then.

    Before starting, it waits up to `wait` seconds while another MAME process runs (wait_for_other_mame; MameError
    naming it after that): parallel runs exit -13."""
    if not isinstance(mame, (str, Path)) or not str(mame):
        raise MameError(f"mame must be the MAME binary (a str or Path), not {mame!r}: "
                        f"{'the custom build is not built (tools/build-mame.sh)' if mame is None else 'bad type'}")
    wait_for_other_mame(wait)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(mame), system, "-rompath", str(rom_dir), "-video", "none", "-sound", "none", "-window", "-nothrottle",
           "-skip_gameinfo", "-str", str(seconds), "-autoboot_script", str(script),
           "-snapshot_directory", str(workdir / "snap"), "-nvram_directory", str(workdir / "nvram"),
           "-cfg_directory", str(workdir / "cfg"), *(extra_args or [])]
    full_env = {**os.environ, "DAYTONA_LUA_DIR": str(LUA_DIR), **(env or {})}
    proc = subprocess.run(cmd, cwd=workdir, env=full_env, capture_output=True, text=True)
    if proc.returncode != 0 or "Average speed" not in proc.stdout:
        raise MameError(f"mame exited {proc.returncode}: {proc.stderr[-2000:]}\n{proc.stdout[-1000:]}")
    lua_errors = [ln for ln in (proc.stdout + "\n" + proc.stderr).splitlines() if "[LUA ERROR]" in ln]
    if lua_errors:
        raise MameError("Lua callback error(s) during the run: " + "\n".join(lua_errors[:20]))
    return proc


def parse_rate(path: Path) -> list[tuple[int, int, int, int]]:
    """(frame, writes to the read start address that frame, videoctl & 3, FNV-1a of the 256 words at the read
    start) per `R` line of renderrate.lua."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 5 and p[0] == "R":
                out.append((int(p[1]), int(p[2]), int(p[3]), int(p[4], 16)))
    return out


@dataclass(frozen=True)
class GroundSample:
    """One TGP ground query (function 0x36) as ground_log.lua saw it: the call number `n` within `frame`, the
    five arguments (`dest`, `y`, the grid value `id`, `x`, `z`) and the result the i960 read (`height`, `poly`;
    None when it read none: a height word of 0, which the game turns into -0.1)."""
    frame: int
    n: int
    dest: int
    y: float
    id: int
    x: float
    z: float
    height: float | None
    poly: int | None


def parse_ground(path: Path) -> tuple[list[GroundSample], dict[int, tuple[float, float, float]]]:
    """(samples, player position per frame) from a ground_log.lua log: `G` and `P` lines; `END` is ignored, so a
    log of a run that never reached the race parses to ([], {})."""
    samples: list[GroundSample] = []
    positions: dict[int, tuple[float, float, float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 10 and p[0] == "G":
                none = p[8] == "-"
                samples.append(GroundSample(int(p[1]), int(p[2]), int(p[3], 16), float(p[4]), int(p[5], 16),
                                            float(p[6]), float(p[7]), None if none else float(p[8]),
                                            None if p[9] == "-" else int(p[9], 16)))
            elif len(p) == 5 and p[0] == "P":
                positions[int(p[1])] = (float(p[2]), float(p[3]), float(p[4]))
    return samples, positions


def player_samples(samples: list[GroundSample], positions: dict[int, tuple[float, float, float]],
                   tolerance: float = 0.25) -> list[GroundSample]:
    """Per frame with a player position, the call nearest to it in (x, z), if within `tolerance` on both axes.
    The `P` position is read at the frame's end, after the car moved on from the centre query (up to 0.05 per axis
    per frame at racing speed, measured), and the player's wheel-point queries are at least 0.85 away, so 0.25
    picks the centre call without confusing it."""
    out = []
    for frame in sorted(positions):
        px, _, pz = positions[frame]
        near = [(max(abs(s.x - px), abs(s.z - pz)), s) for s in samples if s.frame == frame]
        near = [t for t in near if t[0] <= tolerance]
        if near:
            out.append(min(near, key=lambda t: t[0])[1])
    return out


def drive_summary(samples: list[GroundSample]) -> dict:
    """calls, distance (sum of XZ steps between consecutive samples), offroad (calls with grid value 0), ymin and
    ymax of the query y (None when there are no calls)."""
    ys = [s.y for s in samples]
    return {"calls": len(samples),
            "distance": sum(math.hypot(b.x - a.x, b.z - a.z) for a, b in zip(samples, samples[1:])),
            "offroad": sum(1 for s in samples if s.id == 0),
            "ymin": min(ys, default=None), "ymax": max(ys, default=None)}


def require_calls(samples: list[GroundSample]) -> None:
    """MameError when the run logged no ground call (it never reached the race, or the probe saw nothing)."""
    if not samples:
        raise MameError("no ground calls")


@dataclass(frozen=True)
class Progress:
    """One car's race progress in one frame, as progress_log.lua saw it: the road segment (car +0x12c) and the lap
    (car +0x6f); car 0 is the player."""
    frame: int
    car: int
    segment: int
    lap: int


def _malformed(path: Path, n: int, line: str, e: Exception) -> MameError:
    return MameError(f"{path}: line {n}: malformed ({e}): {line.strip()!r}")


def parse_progress(path: Path) -> tuple[list[Progress], dict[int, int]]:
    """(rows, timer per frame) from a progress_log.lua log: `S <frame> <car> <segment> <lap>` and `T <frame>
    <timer>` lines (the timer u32 at 0x005010d0, in ticks); other lines (C, END, and S / T lines of another field
    count) are ignored, so a log of a run that never reached frame 1330 parses to ([], {}). MameError naming the line
    for an S or T line whose fields are not decimal integers."""
    rows: list[Progress] = []
    timer: dict[int, int] = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            p = line.split()
            try:
                if len(p) == 5 and p[0] == "S":
                    rows.append(Progress(*(int(v) for v in p[1:])))
                elif len(p) == 3 and p[0] == "T":
                    timer[int(p[1])] = int(p[2])
            except ValueError as e:
                raise _malformed(path, n, line, e) from None
    return rows, timer


def parse_ground_slots(path: Path) -> list[tuple[int, int]]:
    """(frame, slot) of every `S <frame> <slot>` line of a ground_log.lua log: the course loads' TGP function 0x0d
    calls, in order."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 3 and p[0] == "S":
                out.append((int(p[1]), int(p[2])))
    return out


def parse_camera(path: Path) -> dict[int, tuple[tuple[float, float, float], int, int, int]]:
    """{frame: ((x, y, z), pitch, yaw, roll)} from the `K <frame> <x> <y> <z> <pitch> <yaw> <roll>` lines of a
    progress_log.lua log: the scripted camera's world position (0x501720) and its s16 angles (0x501738-0x50173d)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 8 and p[0] == "K":
                out[int(p[1])] = ((float(p[2]), float(p[3]), float(p[4])), int(p[5]), int(p[6]), int(p[7]))
    return out


def parse_car_positions(path: Path) -> dict[int, dict[int, tuple[float, float, float]]]:
    """{frame: {car: (x, y, z)}} from the `C <frame> <car> <x> <y> <z>` lines of a progress_log.lua log."""
    out: dict[int, dict[int, tuple[float, float, float]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 6 and p[0] == "C":
                out.setdefault(int(p[1]), {})[int(p[2])] = (float(p[3]), float(p[4]), float(p[5]))
    return out


def parse_cells(path: Path) -> dict[int, tuple[int, tuple[int, ...]]]:
    """{frame: (the frame's polygon count, the visible cells in draw order)} from the `V <frame> <count> <cells>`
    lines of a cells_log.lua log (cells two-digit hex joined by commas, "-" for none); other lines (and V lines of
    another field count) are ignored. MameError naming the line for a V line whose frame or count is not a decimal
    integer or whose cells are not hex bytes (0-ff)."""
    out: dict[int, tuple[int, tuple[int, ...]]] = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            p = line.split()
            if len(p) == 4 and p[0] == "V":
                try:
                    cells = () if p[3] == "-" else tuple(int(c, 16) for c in p[3].split(","))
                    if any(not 0 <= c <= 0xFF for c in cells):
                        raise ValueError("a cell outside 00-ff")
                    out[int(p[1])] = (int(p[2]), cells)
                except ValueError as e:
                    raise _malformed(path, n, line, e) from None
    return out
