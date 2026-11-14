-- Render-rate probe, one line per frame. Env: DAYTONA_RATE_LOG (absolute path).
-- Line: "R <frame> <writes to 0x00803008 this frame> <videoctl & 3> <fnv1a32 of the 256 words at the read start>".
-- Writes are counted from one frame_done to the next. The video control and the hash are taken as geo_parse saw them
-- at the frame's vblank start (see geo_sample.lua): the hash covers the buffer words at the read start (wrapping in
-- the 0x20000-byte buffer), each word's four little-endian bytes in order, so it changes when a new list is parsed.
-- video control bit 0 set = geo_parse only on even frames (30 Hz), clear = every vblank.
local path = os.getenv("DAYTONA_RATE_LOG")
if path then
  if not _G.daytona_geo then dofile(os.getenv("DAYTONA_LUA_DIR") .. "/geo_sample.lua") end
  local mem = manager.machine.devices[":maincpu"].spaces["program"]
  local geo = _G.daytona_geo
  local out = assert(io.open(path, "w"))
  local frame, seen, ctl, h = 0, 0, 0, 0
  table.insert(geo.listeners, function(read_start)
    h = 0x811c9dc5
    local base = read_start & 0x1ffff
    for i = 0, 255 do
      local w = mem:read_u32(0x00900000 + ((base + 4 * i) & 0x1ffff))
      for s = 0, 24, 8 do h = ((h ~ ((w >> s) & 0xff)) * 0x01000193) & 0xffffffff end
    end
    ctl = mem:read_u32(0x0098000c) & 3
  end)
  emu.register_frame_done(function()
    geo.capture()
    frame = frame + 1
    out:write(string.format("R %d %d %d %08x\n", frame, geo.writes - seen, ctl, h))
    seen = geo.writes
    if frame % 60 == 0 then out:flush() end
  end)
  emu.add_machine_stop_notifier(function() out:close() end)
end
