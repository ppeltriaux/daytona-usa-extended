-- Circuit Select logger for the sixteen-entry select (dofile'd by autoplay.lua when DAYTONA_SELECT_LOG is set);
-- parsed by tests/test_sixteen_mame.py. Env: DAYTONA_SELECT_LOG (absolute output path), DAYTONA_SELECT_FROM /
-- DAYTONA_SELECT_TO (frames logged with F lines, inclusive; default 600 / 3000), DAYTONA_SELECT_TILES (comma-separated
-- frames at which tile rows 9-18 of layers 0 and 1 are dumped), DAYTONA_SELECT_CHARS (one frame at which char RAM
-- 0x3b00-0x3fff and palettes 0x76-0x7f and 0xf6-0xff are dumped), DAYTONA_SELECT_WATCH ("lo-hi", hex: every read and
-- write there, from the start of the run, summed per (kind, ip, aligned address, byte-lane mask), this script's own
-- reads left out (a tap sees a Lua read too, with whatever ip the CPU is at), DAYTONA_SELECT_MODE (0-2: the game mode
-- half-word 0x5fe5e0 is written with it at the end of every frame from DAYTONA_SELECT_FROM to DAYTONA_SELECT_TO, so the
-- select draws that mode's panel records; the operator's setting otherwise). Frames are counted like autoplay.lua.
-- Lines (hex fields):
--   F <frame> <state 0x5043e2> <entry 0x501468 s16> <cursor 0x5043de> <course id 0x501460> <first 0x501b24>
--     <timer 0x5010a8> <frame counter 0x5010b0> <frame counter 0x540058> (the game's two frame counters: a frame the
--     game drops shows as these falling behind another set's at the same frame)
--   T <frame> <layer> <row> <64 u16 tiles, 4 hex digits each>
--   M <frame> <the window mask rows 80-143 (0x100c280-0x100c47f): 64 rows of four u16, 4 hex digits each>
--   S <frame> <tile RAM words 0x5000-0x5007: the four layers' horizontal, then vertical scroll and control words>
--   C <frame> <chars from 0x3b00, 32 bytes each, 2 hex digits per byte>
--   P <frame> <first entry> <u16 entries, 4 hex digits each>
--   A <R|W> <ip> <aligned address> <mask> <first frame> <count>   (DAYTONA_SELECT_WATCH, at the end)
--   END <frames>
local path = os.getenv("DAYTONA_SELECT_LOG")
local from = tonumber(os.getenv("DAYTONA_SELECT_FROM") or "600")
local to = tonumber(os.getenv("DAYTONA_SELECT_TO") or "3000")
local tiles_at = {}
for f in string.gmatch(os.getenv("DAYTONA_SELECT_TILES") or "", "%d+") do tiles_at[tonumber(f)] = true end
local chars_at = tonumber(os.getenv("DAYTONA_SELECT_CHARS") or "-1")
local mode = tonumber(os.getenv("DAYTONA_SELECT_MODE") or "")
if os.getenv("DAYTONA_SELECT_MODE") and not (mode and mode >= 0 and mode <= 2) then
  error("DAYTONA_SELECT_MODE must be 0, 1 or 2")
end
local mem = manager.machine.devices[":maincpu"].spaces["program"]
local out = assert(io.open(path, "w"))
local TILE_RAM, CHAR_RAM, PALETTE_RAM = 0x01000000, 0x01080000, 0x01800000

local function row_hex(base, n, step, read)
  local t = {}
  for i = 0, n - 1 do t[#t + 1] = string.format(step == 2 and "%04x" or "%02x", read(base + step * i)) end
  return table.concat(t)
end

local frame = 0
local watch, watch_order = {}, {}
local own = false       -- set while this script reads memory itself: its reads would fire the watch (with any ip)
local watch_lo, watch_hi = string.match(os.getenv("DAYTONA_SELECT_WATCH") or "", "^(%x+)%-(%x+)$")
if os.getenv("DAYTONA_SELECT_WATCH") and not watch_lo then error("bad DAYTONA_SELECT_WATCH range") end
if watch_lo then
  local ip = manager.machine.devices[":maincpu"].state["ip"]
  local function note(kind, offset, mask)
    if own then return end
    local key = string.format("%s %x %x %x", kind, ip.value, offset, mask)
    local w = watch[key]
    if w then w.n = w.n + 1 else watch[key] = {first = frame, n = 1}; watch_order[#watch_order + 1] = key end
  end
  _G.daytona_select_watch_r = mem:install_read_tap(tonumber(watch_lo, 16), tonumber(watch_hi, 16), "select_watch_r",
    function(offset, data, mask) note("R", offset, mask) end)
  _G.daytona_select_watch_w = mem:install_write_tap(tonumber(watch_lo, 16), tonumber(watch_hi, 16), "select_watch_w",
    function(offset, data, mask) note("W", offset, mask) end)
end
emu.register_frame_done(function()
  frame = frame + 1
  own = true
  if frame >= from and frame <= to then
    out:write(string.format("F %d %x %x %x %x %x %x %x %x\n", frame, mem:read_u8(0x5043e2),
      mem:read_i16(0x501468) & 0xffff, mem:read_u8(0x5043de), mem:read_u8(0x501460), mem:read_u8(0x501b24),
      mem:read_u32(0x5010a8), mem:read_u32(0x5010b0), mem:read_u32(0x540058)))
    if mode then mem:write_u16(0x5fe5e0, mode) end
  end
  if tiles_at[frame] then
    out:write(string.format("S %d %s\n", frame,
      row_hex(TILE_RAM + 0xa000, 8, 2, function(a) return mem:read_u16(a) end)))
    out:write(string.format("M %d %s\n", frame,
      row_hex(TILE_RAM + 0xc280, 256, 2, function(a) return mem:read_u16(a) end)))
    for layer = 0, 1 do
      for row = 9, 18 do
        local base = TILE_RAM + 0x2000 * layer + 0x80 * row
        out:write(string.format("T %d %d %d %s\n", frame, layer, row,
          row_hex(base, 64, 2, function(a) return mem:read_u16(a) end)))
      end
    end
  end
  if frame == chars_at then
    out:write(string.format("C %d %s\n", frame,
      row_hex(CHAR_RAM + 32 * 0x3b00, 32 * 0x500, 1, function(a) return mem:read_u8(a) end)))
    for _, first in ipairs({0x760, 0xf60}) do
      out:write(string.format("P %d %x %s\n", frame, first,
        row_hex(PALETTE_RAM + 2 * first, 160, 2, function(a) return mem:read_u16(a) end)))
    end
  end
  own = false
end)
_G.daytona_select_stop = emu.add_machine_stop_notifier(function()   -- kept: a dropped
  -- subscription would end with its garbage collection
  for _, key in ipairs(watch_order) do
    out:write(string.format("A %s %d %d\n", key, watch[key].first, watch[key].n))
  end
  out:write(string.format("END %d\n", frame))
  out:close()
end)
