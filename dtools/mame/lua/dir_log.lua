-- Log the Course Direction screen (M9 stage 3, dtools/arcade/dirscreen.py), one line per frame from
-- DAYTONA_DIR_FROM (default 600) to DAYTONA_DIR_TO (default 2400):
--   D <frame> <state> <choice> <armed> <timer> <course id> <paddle> <linked 0x5fe588> <votes 0x501b03>
--     <transmission 0x501464: Mission Select's choice, 0 automatic, 1 manual>
-- state the select's state byte (0x62(g13), g13 = 0x504380: 17-19 the direction screen, 8-11 Mission Select),
-- choice 0x501b26 (0 NORMAL, 1 REVERSE), armed 0x501b27, timer 0x5010a8 (s32), course id 0x501460, paddle the
-- wheel value Mission Select reads ((0x501050 + 0x50105c) >> 1, s32). Run by autoplay.lua when DAYTONA_DIR_LOG
-- names the file.
-- DAYTONA_DIR_POKE (test-only): "addr:value:from:to[,...]" (hex addr, decimal rest) writes the byte `value` at `addr`
-- at the end of each frame from `from` to `to` (the game reads it in the frame after): the guard tests' forced vote
-- count and link flag.
local path = os.getenv("DAYTONA_DIR_LOG")
local from = tonumber(os.getenv("DAYTONA_DIR_FROM") or "600")
local to = tonumber(os.getenv("DAYTONA_DIR_TO") or "2400")
local mem = manager.machine.devices[":maincpu"].spaces["program"]
local out = assert(io.open(path, "w"))
local n = 0
local pokes = {}
for spec in string.gmatch(os.getenv("DAYTONA_DIR_POKE") or "", "[^,]+") do
  local a, v, f, t = string.match(spec, "^(%x+):(%d+):(%d+):(%d+)$")
  if not a then error("DAYTONA_DIR_POKE: bad entry " .. spec) end
  pokes[#pokes + 1] = {tonumber(a, 16), tonumber(v), tonumber(f), tonumber(t)}
end
local function s32(v) if v >= 0x80000000 then return v - 0x100000000 end return v end
emu.register_frame_done(function()
  n = n + 1
  for _, pk in ipairs(pokes) do
    if n >= pk[3] and n <= pk[4] then mem:write_u8(pk[1], pk[2]) end
  end
  if n < from or n > to then return end
  local w = (s32(mem:read_u32(0x00501050)) + s32(mem:read_u32(0x0050105c))) // 2
  out:write(string.format("D %d %d %d %d %d %d %d %d %d %d\n", n, mem:read_u8(0x504380 + 0x62), mem:read_u8(0x00501b26),
    mem:read_u8(0x00501b27), s32(mem:read_u32(0x005010a8)), mem:read_u8(0x00501460), w, mem:read_u8(0x005fe588),
    mem:read_u8(0x00501b03), mem:read_u8(0x00501464)))
  if n == to then out:flush() end
end)
emu.add_machine_stop_notifier(function() out:close() end)
