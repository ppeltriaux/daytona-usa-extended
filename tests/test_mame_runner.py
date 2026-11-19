import collections
import math
import re
import sys

import pytest

from dtools.arcade.geo import parse_log, walk
from dtools.mame import runner
from dtools.mame.runner import LUA_DIR, MameError, parse_rate, run_mame


@pytest.fixture
def no_other_mame(monkeypatch):
    """The fake-MAME tests start no MAME: no job limit, so run_mame never looks for other MAME processes."""
    monkeypatch.delenv("DAYTONA_MAME_JOBS", raising=False)


@pytest.mark.mame
def test_attract_mode_feeds_the_geometry_engine(tmp_path, romset_zip):
    log = tmp_path / "geo.txt"
    proc = run_mame(LUA_DIR / "geo_log.lua", 35, tmp_path, romset_zip.parent,
                    env={"DAYTONA_GEO_LOG": str(log), "DAYTONA_GEO_FROM": "1500", "DAYTONA_GEO_EVERY": "60"})
    assert "LUA ERROR" not in proc.stdout + proc.stderr          # a failing frame callback does not stop MAME
    frames = list(parse_log(log))
    assert frames
    assert all(len(words) == 0x8000 for _, _, _, words in frames)
    assert any(c.kind in (0x01, 0x11) for _, start, _, words in frames for c in walk(words, start // 4))


@pytest.mark.mame
def test_missing_ioboard_rom_raises(tmp_path, romset_zip):
    roms = tmp_path / "roms"
    roms.mkdir()
    (roms / "daytona.zip").symlink_to(romset_zip)                  # no model1io.zip beside it
    with pytest.raises(MameError, match="missing|not found"):
        run_mame(LUA_DIR / "geo_log.lua", 1, tmp_path / "run", roms)


@pytest.mark.mame
def test_runner_writes_only_inside_the_workdir(tmp_path, romset_zip):
    wd = tmp_path / "run"
    before = set(tmp_path.iterdir())
    run_mame(LUA_DIR / "autoplay.lua", 2, wd, romset_zip.parent, env={"DAYTONA_SNAPS": "60"})
    assert set(tmp_path.iterdir()) - before == {wd}
    assert (wd / "snap" / "daytona" / "0000.png").is_file()


def test_parse_rate_reads_render_rate_lines(tmp_path):
    log = tmp_path / "rate.txt"
    log.write_text("X junk\nR 1 0 0 811c9dc5\nR 2 1 1 0000beef\nR 3\n")
    assert parse_rate(log) == [(1, 0, 0, 0x811C9DC5), (2, 1, 1, 0xBEEF)]


def _fake_mame(tmp_path, body: str) -> str:
    mame = tmp_path / "fake-mame"
    mame.write_text("#!/bin/sh\n" + body)
    mame.chmod(0o755)
    return str(mame)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only: a shell-script fake MAME / pgrep and ps (dtools does not run MAME for players)")
def test_lua_error_in_a_callback_fails_the_run(tmp_path, no_other_mame):
    mame = _fake_mame(tmp_path, "echo '[LUA ERROR] boom'\necho 'Average speed: 100%'\nexit 0\n")
    with pytest.raises(MameError, match=r"\[LUA ERROR\] boom"):
        run_mame(tmp_path / "x.lua", 1, tmp_path / "run", tmp_path, mame=mame)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only: a shell-script fake MAME / pgrep and ps (dtools does not run MAME for players)")
def test_lua_error_on_stderr_fails_the_run(tmp_path, no_other_mame):
    mame = _fake_mame(tmp_path, "echo '[LUA ERROR] on stderr' >&2\necho 'Average speed: 100%'\nexit 0\n")
    with pytest.raises(MameError, match="on stderr"):
        run_mame(tmp_path / "x.lua", 1, tmp_path / "run", tmp_path, mame=mame)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only: a shell-script fake MAME / pgrep and ps (dtools does not run MAME for players)")
def test_clean_fake_run_passes(tmp_path, no_other_mame):
    mame = _fake_mame(tmp_path, "echo 'Average speed: 100%'\nexit 0\n")
    assert "Average speed" in run_mame(tmp_path / "x.lua", 1, tmp_path / "run", tmp_path, mame=mame).stdout


class _Clock:
    """A fake clock that `sleep` advances, so the wait is tested without waiting."""
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s

    def __call__(self):
        return self.t


def test_wait_returns_once_a_slot_is_free(monkeypatch):
    two, one = ["1 model2 daytona", "2 model2 daytona"], ["2 model2 daytona"]
    seen = iter([two, two, one])
    monkeypatch.setattr(runner, "running_daytona_mame", lambda: next(seen))
    clock = _Clock()
    runner.wait_for_mame_slot(2, 600, 5, sleep=clock.sleep, clock=clock)
    assert clock.sleeps == [5, 5]


def test_wait_gives_up_naming_the_daytona_mame(monkeypatch):
    monkeypatch.setattr(runner, "running_daytona_mame", lambda: ["4242 /x/build/mame/model2 daytona -str 30"])
    clock = _Clock()
    with pytest.raises(MameError, match=r"(?s)DAYTONA_MAME_JOBS=1.*4242 /x/build/mame/model2 daytona"):
        runner.wait_for_mame_slot(1, 12, 5, sleep=clock.sleep, clock=clock)
    assert clock.t == 12 and clock.sleeps == [5, 5, 2]          # bounded: polls up to the timeout, not past it


@pytest.mark.parametrize("value, jobs", [(None, None), ("", None), ("1", 1), ("6", 6)])
def test_mame_jobs_reads_the_environment(monkeypatch, value, jobs):
    if value is None:
        monkeypatch.delenv("DAYTONA_MAME_JOBS", raising=False)
    else:
        monkeypatch.setenv("DAYTONA_MAME_JOBS", value)
    assert runner.mame_jobs() == jobs


@pytest.mark.parametrize("value", ["0", "-2", "two"])
def test_mame_jobs_refuses_a_bad_value(monkeypatch, value):
    monkeypatch.setenv("DAYTONA_MAME_JOBS", value)
    with pytest.raises(MameError, match="DAYTONA_MAME_JOBS must be a positive integer"):
        runner.mame_jobs()


def _fake_ps(tmp_path, monkeypatch, lines: dict[str, list[str]]):
    """pgrep and ps on PATH: `pgrep -x NAME` prints the pids of lines[NAME]; `ps` prints every `pid command` line."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    pg = "\n".join(f'[ "$2" = "{name}" ] && printf "%s\\n" {" ".join(ln.split()[0] for ln in ls)}'
                   for name, ls in lines.items())
    ps = "\n".join(f'echo " {ln}"' for ls in lines.values() for ln in ls)
    for name, body in (("pgrep", pg + "\nexit 0"), ("ps", ps)):
        (bin_ / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_ / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only: a shell-script fake MAME / pgrep and ps (dtools does not run MAME for players)")
def test_running_daytona_mame_ignores_other_projects(tmp_path, monkeypatch):
    """Only the MAME processes running this project's systems (daytona, dc) count; another project's MAME (vr,
    genesis) does not, whether it is named mame or model2."""
    _fake_ps(tmp_path, monkeypatch, {
        "mame": ["4242 /opt/homebrew/bin/mame vr -str 30", "4243 /opt/homebrew/bin/mame genesis",
                 "4244 mame dc -cdrom x"],
        "model2": ["5151 /x/build/mame/model2 daytona -rompath /r", "5152 /y/model2 vr"]})
    assert runner.running_daytona_mame() == ["4244 mame dc -cdrom x", "5151 /x/build/mame/model2 daytona -rompath /r"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only: a shell-script fake MAME / pgrep and ps (dtools does not run MAME for players)")
def test_running_daytona_mame_without_pgrep(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))                     # no pgrep at all
    assert runner.running_daytona_mame() == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only: a shell-script fake MAME / pgrep and ps (dtools does not run MAME for players)")
def test_run_mame_does_not_wait_for_other_mame(tmp_path, monkeypatch, no_other_mame):
    """No job limit: run_mame starts at once, with another project's MAME and even another daytona MAME running
    (each run has its own comm port, so they no longer meet on 15112)."""
    _fake_ps(tmp_path, monkeypatch, {"mame": ["4242 /opt/homebrew/bin/mame vr"],
                                     "model2": ["5151 /x/build/mame/model2 daytona"]})
    monkeypatch.setattr(runner.time, "sleep", lambda s: pytest.fail("run_mame waited"))
    mame = _fake_mame(tmp_path, "echo 'Average speed: 100%'\nexit 0\n")
    run_mame(tmp_path / "x.lua", 1, tmp_path / "run", tmp_path, mame=mame)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only: a shell-script fake MAME / pgrep and ps (dtools does not run MAME for players)")
def test_run_mame_waits_for_a_slot_with_a_job_limit(tmp_path, monkeypatch):
    marker = tmp_path / "started"
    states = iter([["1 model2 daytona"], []])

    def running():
        assert not marker.exists(), "MAME started while the job limit was reached"
        return next(states)

    monkeypatch.setenv("DAYTONA_MAME_JOBS", "1")
    monkeypatch.setattr(runner, "running_daytona_mame", running)
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    mame = _fake_mame(tmp_path, f"touch {marker}\necho 'Average speed: 100%'\nexit 0\n")
    run_mame(tmp_path / "x.lua", 1, tmp_path / "run", tmp_path, mame=mame)
    assert marker.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only: a shell-script fake MAME / pgrep and ps (dtools does not run MAME for players)")
def test_run_mame_gives_each_run_its_own_comm_port(tmp_path, monkeypatch, no_other_mame):
    """Daytona switches its comm board on at boot and MAME's default comm ports are 15112 for every instance, so two
    runs at once met on that port and one died of SIGPIPE (-13; measured 2026-11-17, three race tests at once). Each
    run listens on and connects to a free port of its own (local = remote, as a lone run on 15112), before
    extra_args."""
    ports = iter([15301, 15302])
    monkeypatch.setattr(runner, "free_port", lambda: next(ports))
    args = tmp_path / "args"
    mame = _fake_mame(tmp_path, f"printf '%s\\n' \"$@\" >> {args}\necho END >> {args}\necho 'Average speed: 100%'\n")
    for _ in range(2):
        run_mame(tmp_path / "x.lua", 1, tmp_path / "run", tmp_path, mame=mame, extra_args=["-cdrom", "d.cue"])
    runs = [r.splitlines() for r in args.read_text().split("END\n") if r]
    for argv, port in zip(runs, ("15301", "15302")):
        assert argv[argv.index("-comm_localport") + 1] == port
        assert argv[argv.index("-comm_remoteport") + 1] == port
        assert argv[-2:] == ["-cdrom", "d.cue"]


def test_free_port_is_free():
    import socket
    port = runner.free_port()
    assert 1024 <= port <= 65535
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:   # nothing holds it: it binds again
        s.bind(("0.0.0.0", port))


def test_run_mame_refuses_a_missing_custom_build(tmp_path, no_other_mame):
    """run_mame(..., mame=config.custom_mame()) with the custom build absent: MameError, MAME not started."""
    for bad in (None, 7, True):
        with pytest.raises(MameError, match="mame"):
            run_mame(tmp_path / "x.lua", 1, tmp_path / "run", tmp_path, mame=bad)


GROUND_LOG = """\
G 10 1 17ffc 1.5000 5 -10.0000 20.0000 0.2500 3f
G 10 2 17ffc 2.0000 0 100.0000 -4.0000 - -
P 10 -10.0000 1.5000 20.0000
G 11 1 17ffc 1.5000 5 -10.0050 20.0000 0.2500 3f
END 12
"""


def test_parse_ground_round_trip(tmp_path):
    log = tmp_path / "ground.txt"
    log.write_text(GROUND_LOG)
    samples, positions = runner.parse_ground(log)
    assert samples[0] == runner.GroundSample(10, 1, 0x17ffc, 1.5, 5, -10.0, 20.0, 0.25, 0x3f)
    assert samples[1] == runner.GroundSample(10, 2, 0x17ffc, 2.0, 0, 100.0, -4.0, None, None)
    assert len(samples) == 3
    assert positions == {10: (-10.0, 1.5, 20.0)}


def test_player_samples_match_position(tmp_path):
    log = tmp_path / "ground.txt"
    log.write_text(GROUND_LOG)
    samples, positions = runner.parse_ground(log)
    assert runner.player_samples(samples, positions) == [samples[0]]      # frame 11 has no P line; frame 10 matches call 1


def test_drive_summary(tmp_path):
    log = tmp_path / "ground.txt"
    log.write_text(GROUND_LOG)
    samples, _ = runner.parse_ground(log)
    s = runner.drive_summary(samples)
    assert s["calls"] == 3 and s["offroad"] == 1 and s["ymin"] == 1.5 and s["ymax"] == 2.0
    assert s["distance"] == pytest.approx(math.hypot(110, 24) + math.hypot(110.005, 24))


def test_empty_log_summary(tmp_path):
    log = tmp_path / "ground.txt"
    log.write_text("END 0\n")
    samples, positions = runner.parse_ground(log)
    assert samples == [] and positions == {}
    assert runner.drive_summary(samples)["calls"] == 0
    with pytest.raises(MameError, match="no ground calls"):
        runner.require_calls(samples)


@pytest.mark.mame
def test_ground_log_refused_with_trace_log(tmp_path, romset_zip):
    with pytest.raises(MameError, match="DAYTONA_GROUND_LOG cannot be combined"):
        run_mame(LUA_DIR / "autoplay.lua", 1, tmp_path / "run", romset_zip.parent,
                 env={"DAYTONA_GROUND_LOG": str(tmp_path / "g.txt"), "DAYTONA_TRACE_LOG": str(tmp_path / "t.txt")})


@pytest.mark.mame
def test_tgp_log_refused_with_trace_and_ground_logs(tmp_path, romset_zip):
    for other in ("DAYTONA_GROUND_LOG", "DAYTONA_TRACE_LOG"):
        with pytest.raises(MameError, match="DAYTONA_TGP_LOG cannot be combined"):
            run_mame(LUA_DIR / "autoplay.lua", 1, tmp_path / "run", romset_zip.parent,
                     env={"DAYTONA_TGP_LOG": str(tmp_path / "tgp.txt"), other: str(tmp_path / "other.txt")})


@pytest.mark.mame
def test_tgp_trace_sees_the_boot_slot(tmp_path, romset_zip):
    """At boot the TGP's init calls routine 0x7cb with slot 0: it sets the bank to copro_data and reads slot 0's
    two words through the bank, at 0x800010 and 0x800020 (docs/spike/2026-11-tgp-ground.md, "Layout"). So the read
    tap on the io space fires through MAME's bank view, and the bank-register tap on rf word 3 fires."""
    log = tmp_path / "tgp.txt"
    run_mame(LUA_DIR / "autoplay.lua", 2, tmp_path / "run", romset_zip.parent,
             env={"DAYTONA_TGP_LOG": str(log), "DAYTONA_TGP_TO": "100"})
    lines = [ln.split() for ln in log.read_text().splitlines()]
    assert lines[0][0] == "S" and {"io", "rf"} <= set(lines[0][1:])
    assert lines[-1][0] == "END"
    reads = {(int(p[2], 16), int(p[3], 16)) for p in lines if p[0] == "R"}
    assert (0x7CF, 0x800010) in reads and (0x7D3, 0x800020) in reads
    banks = [int(p[3], 16) for p in lines if p[0] == "B"]
    assert banks and banks[0] & 0xC00000 == 0x800000


def test_player_samples_follow_a_moving_car():
    """The P line is read after the car moved on from its query: the nearest call within the tolerance matches."""
    samples = [runner.GroundSample(5, 1, 0x17ffc, 0, 1, 9.0, 9.0, 0.0, 1),          # another car
               runner.GroundSample(5, 2, 0x17ffc, 0, 1, 10.00, 20.00, 0.0, 2),      # the centre call
               runner.GroundSample(5, 3, 0x17ffc, 0, 1, 10.85, 20.00, 0.0, 3)]      # a wheel point
    assert runner.player_samples(samples, {5: (10.04, 0.0, 20.03)}) == [samples[1]]
    assert runner.player_samples(samples, {5: (50.0, 0.0, 50.0)}) == []


@pytest.mark.mame
def test_original_beginner_ground(tmp_path, romset_zip):
    """Course 0, 60 s emulated (about 3,450 frames, racing from 2400), original ROMs. Measured 2026-11-07: the game
    does not query the ground of all 40 cars each frame; while racing it makes 8-11 calls every frame up to 2700, 9 mostly (5-13 after):
    the player's four wheel points and centre plus the nearby cars (docs/spike/2026-11-tgp-ground.md, "Ground
    probe")."""
    log = tmp_path / "ground.txt"
    run_mame(LUA_DIR / "autoplay.lua", 60, tmp_path / "run", romset_zip.parent,
             env={"DAYTONA_COURSE": "0", "DAYTONA_GROUND_LOG": str(log)})
    samples, positions = runner.parse_ground(log)
    runner.require_calls(samples)
    last = max(positions) - 1                       # the final frame's calls may be cut off by the end of the run
    racing = [f for f in positions if 2400 <= f <= last]
    per_frame = collections.Counter(s.frame for s in samples if 2400 <= s.frame <= last)
    assert last - 2400 > 900 and sorted(per_frame) == list(range(2400, last + 1))      # a call set every frame
    assert all(8 <= c <= 11 for f, c in per_frame.items() if f < 2700)      # later: 5-13 as cars come and go
    player = [s for s in runner.player_samples(samples, positions) if 2400 <= s.frame <= last]
    assert len(player) >= 0.9 * len(racing) and len(player) > 400
    assert runner.drive_summary(player)["offroad"] == 0
    assert all(s.poly is not None and s.poly < 1184 for s in player)
    assert all(s.height is not None and math.isfinite(s.height) for s in player)


PROGRESS_LOG = """\
S 1330 0 104 0
C 1330 0 94.6979 -0.0000 150.4941
S 1330 1 107 0
C 1330 1 64.0000 0.0000 138.4941
T 1330 3200
X junk
S 1331 0 105 1
T 1331 3199
S 1332 0
END 1333
"""


def test_parse_progress_round_trip(tmp_path):
    log = tmp_path / "progress.txt"
    log.write_text(PROGRESS_LOG)
    rows, timer = runner.parse_progress(log)
    assert rows == [runner.Progress(1330, 0, 104, 0), runner.Progress(1330, 1, 107, 0), runner.Progress(1331, 0, 105, 1)]
    assert timer == {1330: 3200, 1331: 3199}
    assert rows[2].segment == 105 and rows[2].lap == 1 and rows[2].car == 0 and rows[2].frame == 1331
    assert runner.parse_car_positions(log) == {1330: {0: (94.6979, -0.0, 150.4941), 1: (64.0, 0.0, 138.4941)}}
    empty = tmp_path / "empty.txt"
    empty.write_text("END 0\n")
    assert runner.parse_progress(empty) == ([], {}) and runner.parse_car_positions(empty) == {}


def test_parse_cells(tmp_path):
    """cells_log.lua's lines: the frame's polygon count and its visible cells in draw order (Phase 7 Task 7)."""
    log = tmp_path / "cells.txt"
    log.write_text("V 1330 4118 75,76,65\nX junk\nV 1331 0 -\nV 1332 5577 89,a7\nV 1333\n")
    assert runner.parse_cells(log) == {1330: (4118, (0x75, 0x76, 0x65)), 1331: (0, ()), 1332: (5577, (0x89, 0xa7))}


@pytest.mark.parametrize("line", ["V 13x0 4118 75,76", "V 1330 41.5 75", "V 1330 4118 75,zz", "V 1330 4118 75,,76",
                                  "V 1330 4118 1ff", "V 1330 4118 -5"])
def test_parse_cells_malformed(tmp_path, line):
    """A V line of the right shape whose frame, count or cells do not parse: MameError naming the line, not a raw
    ValueError (Phase 7 Task 8)."""
    log = tmp_path / "cells.txt"
    log.write_text(f"V 1329 0 -\n{line}\n")
    with pytest.raises(MameError, match=rf"line 2: malformed .*{re.escape(line)}"):
        runner.parse_cells(log)


@pytest.mark.parametrize("line", ["S 1330 0 10x 0", "S 1330 0 104 0.5", "T 1330 ff", "T x 3200"])
def test_parse_progress_malformed(tmp_path, line):
    """An S or T line of the right shape whose fields are not decimal integers: MameError naming the line, not a raw
    ValueError (Phase 7 Task 8)."""
    log = tmp_path / "progress.txt"
    log.write_text(f"S 1330 0 104 0\n{line}\n")
    with pytest.raises(MameError, match=rf"line 2: malformed .*{re.escape(line)}"):
        runner.parse_progress(log)


@pytest.mark.mame
def test_progress_log_refused_with_trace_log(tmp_path, romset_zip):
    with pytest.raises(MameError, match="DAYTONA_PROGRESS_LOG cannot be combined"):
        run_mame(LUA_DIR / "autoplay.lua", 1, tmp_path / "run", romset_zip.parent,
                 env={"DAYTONA_PROGRESS_LOG": str(tmp_path / "p.txt"), "DAYTONA_TRACE_LOG": str(tmp_path / "t.txt")})
