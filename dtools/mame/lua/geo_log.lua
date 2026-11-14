-- Dump the geometry engine's buffer RAM every DAYTONA_GEO_EVERY frames from DAYTONA_GEO_FROM on.
-- Env: DAYTONA_GEO_LOG (absolute path), DAYTONA_GEO_FROM (default 1400), DAYTONA_GEO_EVERY (default 10).
-- Line: "G <frame> <read_start_hex> <videoctl_hex> <0x8000 hex u32 words of 0x00900000..0x0091ffff>".
-- read_start is the last value the i960 wrote to 0x00803008 (geo_w "Read Start Address", & 0xfffff): a byte offset
-- into the buffer. videoctl is a read of 0x0098000c (videoctl_r: frame-number bit 2 | video control & 3).
-- All three are taken as geo_parse saw them at the frame's vblank start, not at frame_done (see geo_sample.lua).
local out_path = os.getenv("DAYTONA_GEO_LOG")
if out_path then
  if not _G.daytona_geo then dofile(os.getenv("DAYTONA_LUA_DIR") .. "/geo_sample.lua") end
  local from  = tonumber(os.getenv("DAYTONA_GEO_FROM") or "1400")
  local every = tonumber(os.getenv("DAYTONA_GEO_EVERY") or "10")
  local mem = manager.machine.devices[":maincpu"].spaces["program"]
  local geo = _G.daytona_geo
  local out = assert(io.open(out_path, "w"))
  local fmt, concat = string.format, table.concat
  local frame, line = 0, nil
  table.insert(geo.listeners, function(read_start)
    local f = frame + 1                                      -- the frame whose frame_done comes next
    if f < from or (f - from) % every ~= 0 then return end
    local parts = {}
    for i = 0, 0x7fff do parts[i + 1] = fmt("%08x", mem:read_u32(0x00900000 + 4 * i)) end
    line = fmt("G %d %x %x %s\n", f, read_start, mem:read_u32(0x0098000c), concat(parts))
  end)
  emu.register_frame_done(function()
    geo.capture()
    frame = frame + 1
    if line then out:write(line); out:flush(); line = nil end
  end)
  emu.add_machine_stop_notifier(function() out:close() end)
end
