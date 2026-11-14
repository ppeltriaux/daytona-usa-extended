"""The autopilot's road file (autoplay.lua's DAYTONA_AUTOPILOT): one "x z" line per road segment, the road line's
centre in the arcade's frame. Moved from tests/test_courses_mame.py (Phase 13 Task 3) so the page's capture script
(spike/site_media.py) drives the converted courses the way the MAME gates do."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def road_file(c, path: Path) -> Path:
    """`path` with the road line's centre of the course `c` (a golden.Course, or anything whose `road` is a sequence
    of course.RoadPoint: the mean of each road point's used lanes, lane_lo..lane_hi), one "%.4f %.4f" (x z) line per
    segment; returns `path`."""
    path = Path(path)
    path.write_text("".join("%.4f %.4f\n" % tuple(np.mean([ln.position for ln in rp.lanes[rp.lane_lo:rp.lane_hi + 1]],
                                                          axis=0)[[0, 2]]) for rp in c.road))
    return path
