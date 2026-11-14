-- Race-progress probe (dofile'd by autoplay.lua when DAYTONA_PROGRESS_LOG is set; parsed by runner.parse_progress).
-- Env: DAYTONA_PROGRESS_LOG (absolute output path). Allowed with DAYTONA_GROUND_LOG and DAYTONA_RATE_LOG; refused
-- with DAYTONA_TRACE_LOG.
-- Lines, every frame from 1330 (the race HUD; docs/formats/model2-course-tables.md, "Road line", "Checkpoints"):
--   S <frame> <car> <segment> <lap>   one per car in the race: the car count is the u32 at work RAM 0x501080 (the
--     course's CAR_COUNTS entry, set by the course load), cars are 0x300-byte structs from 0x00505180 (the player is
--     car 0); segment = u16 at +0x12c, lap = u8 at +0x6f.
--   C <frame> <car> <x> <y> <z>       the same car's position, f32 at +0x1c / +0x20 / +0x24 (%.4f).
--   T <frame> <timer>                 the race timer, u32 at 0x005010d0 (ticks; counts down one per frame, rises by
--     the time table's word at each checkpoint pass).
--   K <frame> <x> <y> <z> <pitch> <yaw> <roll>   the camera the script player 0x65b4 places (0x6a10): its world
--     position, f32 at 0x00501720 / 24 / 28 (%.4f), and the angles it derives from the target and the roll, s16 at
--     0x00501738 / 3a / 3c (docs/spike/2026-11-mirrors-live.md, "Mirrors of the originals").
--   END <frames>                      frames seen.
-- Frames are counted like ground_log.lua: a line belongs to the count of frames finished before this frame_done.
local path = os.getenv("DAYTONA_PROGRESS_LOG")
if path then
  if os.getenv("DAYTONA_TRACE_LOG") then
    error("DAYTONA_PROGRESS_LOG cannot be combined with DAYTONA_TRACE_LOG")
  end
  local mem = manager.machine.devices[":maincpu"].spaces["program"]
  local out = assert(io.open(path, "w"))
  local fmt = string.format
  local function f32(bits) return (string.unpack("<f", string.pack("<I4", bits & 0xffffffff))) end
  local FROM, CARS, BASE, STRIDE = 1330, 0x00501080, 0x00505180, 0x300
  local frame = 0
  emu.register_frame_done(function()
    if frame >= FROM then
      local n = mem:read_u32(CARS)
      if n < 1 or n > 44 then n = 44 end           -- the car pool (the boot's 44, course.SPAWN_RECORDS)
      for car = 0, n - 1 do
        local s = BASE + STRIDE * car
        out:write(fmt("S %d %d %d %d\n", frame, car, mem:read_u16(s + 0x12c), mem:read_u8(s + 0x6f)))
        out:write(fmt("C %d %d %.4f %.4f %.4f\n", frame, car, f32(mem:read_u32(s + 0x1c)), f32(mem:read_u32(s + 0x20)),
                      f32(mem:read_u32(s + 0x24))))
      end
      out:write(fmt("T %d %d\n", frame, mem:read_u32(0x005010d0)))
      out:write(fmt("K %d %.4f %.4f %.4f %d %d %d\n", frame, f32(mem:read_u32(0x00501720)),
                    f32(mem:read_u32(0x00501724)), f32(mem:read_u32(0x00501728)), mem:read_i16(0x00501738),
                    mem:read_i16(0x0050173a), mem:read_i16(0x0050173c)))
    end
    frame = frame + 1
    if frame % 60 == 0 then out:flush() end
  end)
  _G.daytona_progress = {stop = emu.add_machine_stop_notifier(function()
    out:write(fmt("END %d\n", frame))
    out:close()
  end)}
end
