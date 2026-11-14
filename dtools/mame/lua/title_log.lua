-- The attract logo's logger (dofile'd by autoplay.lua when DAYTONA_TITLE_LOG is set); parsed by
-- tests/test_release_mame.py (Phase 12 Task 4, the Extended title). Env: DAYTONA_TITLE_LOG (absolute output path),
-- DAYTONA_TITLE_DUMPS (comma-separated frames), DAYTONA_TITLE_DIR (absolute directory, with DAYTONA_TITLE_DUMPS).
-- Frames are counted like autoplay.lua.
-- Lines (hex fields), one per frame:
--   C <frame> <state 0x5043e2> <frame counter 0x5010b0> <frame counter 0x540058>
--   END <frames>
-- At each frame of DAYTONA_TITLE_DUMPS: tile RAM (0x01000000, 64 KB), char RAM (0x01080000, 512 KB) and palette RAM
-- (0x01800000, 16 KB), u16 words little-endian, written raw to DAYTONA_TITLE_DIR/<frame>.tile, .char and .pal.
local path = os.getenv("DAYTONA_TITLE_LOG")
local dir = os.getenv("DAYTONA_TITLE_DIR")
local dumps = {}
for f in string.gmatch(os.getenv("DAYTONA_TITLE_DUMPS") or "", "%d+") do dumps[tonumber(f)] = true end
if next(dumps) and not dir then error("DAYTONA_TITLE_DUMPS needs DAYTONA_TITLE_DIR") end
local mem = manager.machine.devices[":maincpu"].spaces["program"]
local out = assert(io.open(path, "w"))

local function raw(file, lo, n)
  local f = assert(io.open(file, "wb"))
  local t = {}
  for i = 0, n - 1, 2 do
    local w = mem:read_u16(lo + i)
    t[#t + 1] = string.char(w & 0xff, w >> 8)
    if #t == 4096 then f:write(table.concat(t)); t = {} end
  end
  f:write(table.concat(t))
  f:close()
end

local frame = 0
emu.register_frame_done(function()
  frame = frame + 1
  out:write(string.format("C %d %x %x %x\n", frame, mem:read_u8(0x5043e2), mem:read_u32(0x5010b0),
    mem:read_u32(0x540058)))
  if dumps[frame] then
    raw(string.format("%s/%d.tile", dir, frame), 0x01000000, 0x10000)
    raw(string.format("%s/%d.char", dir, frame), 0x01080000, 0x80000)
    raw(string.format("%s/%d.pal", dir, frame), 0x01800000, 0x4000)
  end
end)
_G.daytona_title_stop = emu.add_machine_stop_notifier(function()   -- kept: a dropped subscription would end with its
  -- garbage collection
  out:write(string.format("END %d\n", frame))
  out:close()
end)
