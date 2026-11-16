-- Pit probe (dofile'd by autoplay.lua when DAYTONA_PIT_LOG is set). The player car's fields read by the pit-stop
-- trigger 0xdf9c (docs/formats/model2-course-tables.md, "Pit"), every frame from 1330:
--   P <frame> <handler +0xc> <seg +0x12c> <flags +0x13c> <prev +0x13e> <lap +0x70> <laps 0x5010ec> <yaw +0x48>
--     <ref +0x1c6> <+0xd2> <+0x9c> <+0x9d> <+0xd8> <+0xf0> <x> <y> <z> <timer 0x5010d0> <pit 0x501a8c>
--   END <frames>
local path = os.getenv("DAYTONA_PIT_LOG")
if path then
  local mem = manager.machine.devices[":maincpu"].spaces["program"]
  local out = assert(io.open(path, "w"))
  local fmt = string.format
  local function f32(bits) return (string.unpack("<f", string.pack("<I4", bits & 0xffffffff))) end
  local S, frame = 0x00505180, 0
  emu.register_frame_done(function()
    if frame >= 1330 then
      out:write(fmt("P %d %x %d %x %x %d %d %d %d %d %d %d %d %d %.3f %.3f %.3f %d %d\n", frame,
        mem:read_u32(S + 0xc), mem:read_u16(S + 0x12c), mem:read_u16(S + 0x13c), mem:read_u16(S + 0x13e),
        mem:read_i8(S + 0x70), mem:read_i8(0x5010ec), mem:read_i16(S + 0x48), mem:read_i16(S + 0x1c6),
        mem:read_i16(S + 0xd2), mem:read_u8(S + 0x9c), mem:read_u8(S + 0x9d), mem:read_u8(S + 0xd8), mem:read_u8(S + 0xf0),
        f32(mem:read_u32(S + 0x1c)), f32(mem:read_u32(S + 0x20)), f32(mem:read_u32(S + 0x24)),
        mem:read_u32(0x005010d0), mem:read_u32(0x00501a8c)))
    end
    frame = frame + 1
    if frame % 60 == 0 then out:flush() end
  end)
  _G.daytona_pit = {stop = emu.add_machine_stop_notifier(function()
    out:write(fmt("END %d\n", frame))
    out:close()
  end)}
end
