"""v0.3 in MAME: the far view and the raised polygon cap on the release set (tests/test_far_view.py has the bytes;
docs/follow-ups/v0.3.md the measurements)."""
import pytest

from dtools.arcade import course as C
from dtools.arcade import select as S
from dtools.convert import build
from tests.test_convert_tables import disc  # noqa: F401
from tests.test_release_mame import _release

FAR_SECONDS = 160
FAR_FROM, FAR_UPTO = 1800, 7000         # race frames: past the start's fly-by, before a race can end
FAR_PEAK = {3: 13168, 14: 10636, 0: 4990}
"""The frame's polygon count at most, per course id, on the autopilot's FAR_SECONDS race (measured 20 Nov 2026):
Desert City, Rin Rin Rink in reverse, and Three Seven Speedway, which keeps the arcade's view and stays under OBC."""


def _holes(path) -> list[tuple[int, int]]:
    """Runs of 3+ frames with over 5 % of the holes probe's samples magenta: (first frame, frames)."""
    bad = [int(w[1]) for w in (line.split() for line in open(path))
           if w and w[0] == "H" and FAR_FROM <= int(w[1]) < FAR_UPTO and int(w[2]) / int(w[3]) > 0.05]
    out, i = [], 0
    while i < len(bad):
        j = i
        while j + 1 < len(bad) and bad[j + 1] - bad[j] <= 2:
            j += 1
        if bad[j] - bad[i] + 1 >= 3:
            out.append((bad[i], bad[j] - bad[i] + 1))
        i = j + 1
    return out


@pytest.mark.mame
@pytest.mark.custom_mame
@pytest.mark.parametrize("course", (3, 14, 0))
def test_far_view_driven(tmp_path, romset_zip, disc, course):  # noqa: F811
    """The release set raced by the autopilot for FAR_SECONDS on Desert City (3), Rin Rin Rink in reverse (14) and
    Three Seven Speedway (0), the cells log and the holes probe on every race frame FAR_FROM..FAR_UPTO:

    - a far id (build.FAR_IDS): the visible list is the whole 5 x 5 block round the reference cell (its first entry)
      that lies on the grid (the game's own cells, then the hook's), then exactly the cells of build.FAR_SHAPE for
      one heading quadrant that lie on the grid, each once, at most 63; the road never shows a hole (no run of the
      probe);
    - Three Seven Speedway: only cells of its block, at most 15, as on v0.2;
    - the polygon count never reaches the id's cap (build.POLYGON_CAP on a far id, nothing dropped; course.OBC on
      Three Seven Speedway) and peaks at FAR_PEAK."""
    from dtools.arcade import golden as G
    from dtools.mame import build as mame_build, runner
    from dtools.mame.road import road_file
    from tests.test_custom_mame import _custom
    rel = _release(romset_zip, disc)
    cap = C.polygon_cap(rel, course)
    assert C.far_viewed(rel) and cap == (build.POLYGON_CAP if course in build.FAR_IDS else C.OBC)
    entry = next(k for k, e in enumerate(S.ENTRIES) if e.course == course % 8)
    env = {"DAYTONA_ROW": "8", "DAYTONA_ENTRY": str(entry), "DAYTONA_SNAPS": "0", "DAYTONA_AUTOPILOT_LAPS": "99",
           "DAYTONA_DIRECTION": "reverse" if course >= 8 else "normal", "DAYTONA_CELLS_FROM": str(FAR_FROM),
           "DAYTONA_HOLES_FROM": str(FAR_FROM),
           "DAYTONA_AUTOPILOT": str(road_file(G.decode_course(rel, course), tmp_path / "road.txt"))}
    out = mame_build.run_set(rel, tmp_path, FAR_SECONDS, romset_zip, env, logs=("cells", "holes"), mame=_custom())
    seen = {f: v for f, v in runner.parse_cells(out["cells"]).items() if FAR_FROM <= f < FAR_UPTO}
    assert len(seen) > 5000
    far = [{(r, c) for r, c in build.far_offsets(build.FAR_SHAPE, q)} for q in range(4)]
    for f, (count, cells) in seen.items():
        assert count < cap and len(cells) <= 63 and len(set(cells)) == len(cells), f
        ref = cells[0]
        offs = [((k >> 4) - (ref >> 4), (k & 15) - (ref & 15)) for k in cells]
        block = [o for o in offs if max(abs(o[0]), abs(o[1])) <= 2]
        rest = set(offs) - set(block)
        if course not in build.FAR_IDS:
            assert not rest and len(cells) <= 15, f
            continue
        on_grid = lambda o: 0 <= (ref >> 4) + o[0] <= 15 and 0 <= (ref & 15) + o[1] <= 15  # noqa: E731
        assert set(block) == {(r, c) for r in range(-2, 3) for c in range(-2, 3) if on_grid((r, c))}, f
        assert offs[:len(block)] == block, f                       # the far cells come after the block's
        assert any(rest == {o for o in far[q] if on_grid(o)} for q in range(4)), (f, sorted(rest))
    assert max(v[0] for v in seen.values()) == FAR_PEAK[course]
    if course in build.FAR_IDS:
        assert _holes(out["holes"]) == []
