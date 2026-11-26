-- Log the Car Select screen and the player's car (v0.4, dtools/arcade/carscreen.py), one line per frame from
-- DAYTONA_CAR_FROM (default 900):
--   K <frame> <state> <choice> <paddle> <speed> <gear> <engine speed>
-- state the select's state byte (0x62(g13), g13 = 0x504380: 17-19 Course Direction, 20-22 Car Select, 8-11 Mission
-- Select), choice the byte course.CAR_CHOICE_RAM (0x501b64), paddle the wheel value the screens read
-- ((0x501050 + 0x50105c) >> 1, s32), then the player's car (0x505180): its speed (+0x18, s32, km/h as the HUD
-- shows), its gear (+0x14c, 0-3) and its engine speed (+0x154, low word). Run by autoplay.lua when DAYTONA_CAR_LOG
-- names the file.
local path = os.getenv("DAYTONA_CAR_LOG")
local from = tonumber(os.getenv("DAYTONA_CAR_FROM") or "900")
local mem = manager.machine.devices[":maincpu"].spaces["program"]
local out = assert(io.open(path, "w"))
local n = 0
local function s32(v) if v >= 0x80000000 then return v - 0x100000000 end return v end
emu.register_frame_done(function()
  n = n + 1
  if n < from then return end
  local w = (s32(mem:read_u32(0x00501050)) + s32(mem:read_u32(0x0050105c))) // 2
  out:write(string.format("K %d %d %d %d %d %d %d\n", n, mem:read_u8(0x504380 + 0x62), mem:read_u8(0x00501b64), w,
    s32(mem:read_u32(0x505180 + 0x18)), mem:read_u32(0x505180 + 0x14c), mem:read_u32(0x505180 + 0x154) & 0xffff))
  if n % 300 == 0 then out:flush() end
end)
emu.add_machine_stop_notifier(function() out:close() end)
