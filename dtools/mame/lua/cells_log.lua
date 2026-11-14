-- The visible scenery cells and the frame's polygon count, one line per frame from DAYTONA_CELLS_FROM (dofile'd by
-- autoplay.lua when DAYTONA_CELLS_LOG is set; parsed by runner.parse_cells). Allowed with every log but
-- DAYTONA_TRACE_LOG (its own taps on work RAM).
-- Env: DAYTONA_CELLS_LOG (absolute output path), DAYTONA_CELLS_FROM (default 1330, the race HUD).
-- Line: "V <frame> <count> <cells>": cells the visible-cell list at work RAM 0x5016c0 (a count byte, then one byte
--   per cell; docs/formats/model2-course-tables.md, "Culling (per frame)"), as two-digit hex joined by commas ("-" when
--   empty), in the order the draw walks it; count the largest value written to the frame's polygon count (u32 at
--   0x5010e8: each object drawn adds its list entry's budget word; the draw stops emitting objects once it is past
--   0x5010f4, 5000) since the last line.
-- Both are taken where geo_log.lua takes its list (geo_sample.lua: at vblank start, before the i960 starts the next
-- list) and the line carries geo_log.lua's frame number, so line f describes the list geo_log.lua logs as frame f
-- (checked on Desert City's Task 6 set, 8,550 race frames: every scenery object a list draws is a cell of its line's
-- list, and the 285 frames with cells left out and the count's 5,577 are Task 6's scratch tap's; docs/spike/
-- 2026-11-convert.md, "Thinning").
local path = os.getenv("DAYTONA_CELLS_LOG")
if path then
  if os.getenv("DAYTONA_TRACE_LOG") then
    error("DAYTONA_CELLS_LOG cannot be combined with DAYTONA_TRACE_LOG")
  end
  if not _G.daytona_geo then dofile(os.getenv("DAYTONA_LUA_DIR") .. "/geo_sample.lua") end
  local mem = manager.machine.devices[":maincpu"].spaces["program"]
  local geo = _G.daytona_geo
  local out = assert(io.open(path, "w"))
  local from = tonumber(os.getenv("DAYTONA_CELLS_FROM") or "1330")
  local LIST, COUNT = 0x005016c0, 0x005010e8
  local frame, top = 0, 0
  local tap = mem:install_write_tap(COUNT, COUNT + 3, "daytona_cells_count", function(offset, data, mask)
    local v = mem:read_u32(COUNT)                              -- the tap runs before the write lands: merge it
    v = (v & ~mask) | (data & mask)
    if v > top then top = v end
  end)
  table.insert(geo.listeners, function(read_start)
    local f = frame + 1                                        -- the frame whose frame_done comes next (geo_log.lua)
    if f >= from then
      local n = mem:read_u8(LIST)
      local cells = {}
      for i = 1, n do cells[i] = string.format("%02x", mem:read_u8(LIST + i)) end
      out:write(string.format("V %d %d %s\n", f, top, n > 0 and table.concat(cells, ",") or "-"))
    end
    top = 0
  end)
  emu.register_frame_done(function()
    geo.capture()
    frame = frame + 1
    if frame % 60 == 0 then out:flush() end
  end)
  _G.daytona_cells = {tap = tap, stop = emu.add_machine_stop_notifier(function() out:close() end)}
end
