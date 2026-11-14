-- i960 memory tracer (dofile'd by autoplay.lua when DAYTONA_TRACE_LOG is set); parsed by dtools/arcade/trace.py.
-- Env: DAYTONA_TRACE_LOG (absolute output path), DAYTONA_TRACE_FROM / DAYTONA_TRACE_TO (frame window, inclusive /
-- exclusive; defaults 0 / forever), DAYTONA_TRACE_READ and DAYTONA_TRACE_WRITE (comma-separated hex ranges
-- "start-end", i960 addresses, inclusive; a "!" suffix on a write range also logs every write there as a V line with
-- the value written), DAYTONA_TRACE_DUMP (comma-separated frames at which work RAM 0x00500000-0x005fffff and extra RAM
-- 0x00200000-0x0021ffff are written to <log>.ram.<frame>.bin / <log>.ram2.<frame>.bin).
-- Lines: R/W <frame> <ip> <page> <count> aggregated per frame per (ip, addr & ~0xff); V <frame> <ip> <addr> <value>;
-- D <frame> <kind> <path>; END <frames_traced> <events>. Frames are counted like autoplay.lua (frame_done calls).
-- V lines log what the CPU actually wrote: MAME hands the tap the 32-bit-aligned bus address, the whole data word and
-- a byte-lane mask (offset & 3 is always 0; a byte store to 0x...1 arrives as mask 0x0000ff00, checked 2026-11-07).
-- So <addr> is the lowest byte address written (aligned address + lowest lane in the mask) and <value> is
-- (data & mask) shifted down to that byte, printed with two hex digits per byte the mask spans (2 for a byte store,
-- 4 for a half-word, 8 for a word). Little-endian, so <value> reads as the stored integer of that width.
-- ip is the address of the instruction after the accessing one (see dtools/arcade/trace.py).
-- One tap per range (a second tap on the same range may not fire); handlers are kept in _G.daytona_trace.
local log_path = os.getenv("DAYTONA_TRACE_LOG")
if os.getenv("DAYTONA_GEO_LOG") or os.getenv("DAYTONA_RATE_LOG") or os.getenv("DAYTONA_GROUND_LOG") then
  error("DAYTONA_TRACE_LOG cannot be combined with DAYTONA_GEO_LOG, DAYTONA_RATE_LOG or DAYTONA_GROUND_LOG")
end
local from = tonumber(os.getenv("DAYTONA_TRACE_FROM") or "0")
local to = tonumber(os.getenv("DAYTONA_TRACE_TO") or "") or math.huge
local cpu = manager.machine.devices[":maincpu"]
local mem = cpu.spaces["program"]
local ipstate = cpu.state["ip"]
local out = assert(io.open(log_path, "w"))

local function ranges(spec, allow_bang)
  local list = {}
  for item in string.gmatch(spec or "", "[^,]+") do
    local lo, hi, bang = string.match(item, "^%s*(%x+)%-(%x+)(!?)%s*$")
    if not lo then error("bad trace range: " .. item) end
    if bang == "!" and not allow_bang then error("'!' is only valid on DAYTONA_TRACE_WRITE: " .. item) end
    list[#list + 1] = {lo = tonumber(lo, 16), hi = tonumber(hi, 16), exact = bang == "!"}
  end
  return list
end

local frame = 0
local traced = 0
local events = 0
local agg = {}          -- key "R|ip|page" -> count, for the current frame
local values = {}       -- V lines of the current frame
local active = false    -- the frame window, evaluated at each frame_done for the NEXT frame

local function note(kind, addr)
  if not active then return end
  local ip = ipstate.value
  local key = kind .. "|" .. ip .. "|" .. (addr & ~0xff)
  agg[key] = (agg[key] or 0) + 1
end

local taps = {}
for _, r in ipairs(ranges(os.getenv("DAYTONA_TRACE_READ"), false)) do
  taps[#taps + 1] = mem:install_read_tap(r.lo, r.hi, "daytona_trace_r", function(offset, data, mask)
    note("R", offset)
  end)
end
for _, r in ipairs(ranges(os.getenv("DAYTONA_TRACE_WRITE"), true)) do
  local exact = r.exact
  taps[#taps + 1] = mem:install_write_tap(r.lo, r.hi, "daytona_trace_w", function(offset, data, mask)
    note("W", offset)
    if exact and active then
      local lo, hi = 0, 3                              -- lowest and highest byte lane set in the mask
      while lo < 3 and (mask >> (8 * lo)) & 0xff == 0 do lo = lo + 1 end
      while hi > lo and (mask >> (8 * hi)) & 0xff == 0 do hi = hi - 1 end
      local value = ((data & mask) >> (8 * lo)) & ((1 << (8 * (hi - lo + 1))) - 1)
      values[#values + 1] = string.format("V %d %x %x %0" .. (2 * (hi - lo + 1)) .. "x", frame, ipstate.value,
                                          offset + lo, value)
    end
  end)
end
_G.daytona_trace = {taps = taps}

local dumps = {}
for f in string.gmatch(os.getenv("DAYTONA_TRACE_DUMP") or "", "%d+") do dumps[tonumber(f)] = true end

local function dump(kind, lo, hi, f)
  local path = string.format("%s.%s.%d.bin", log_path, kind, f)
  local fh = assert(io.open(path, "wb"))
  fh:write(mem:read_range(lo, hi, 8))
  fh:close()
  out:write(string.format("D %d %s %s\n", f, kind, path))
end

local function flush()
  local lines = {}
  for key, count in pairs(agg) do
    local k, ip, page = string.match(key, "^(%a)|(%d+)|(%d+)$")
    lines[#lines + 1] = string.format("%s %d %x %x %d", k, frame, tonumber(ip), tonumber(page), count)
  end
  table.sort(lines)
  for _, l in ipairs(lines) do out:write(l, "\n") end
  for _, l in ipairs(values) do out:write(l, "\n") end
  events = events + #lines + #values
  agg, values = {}, {}
end

emu.register_frame_done(function()
  if active then flush(); traced = traced + 1 end
  frame = frame + 1                                  -- same count as autoplay.lua's frame_done callback
  if dumps[frame] then
    dump("ram", 0x00500000, 0x005fffff, frame)
    dump("ram2", 0x00200000, 0x0021ffff, frame)
  end
  active = frame >= from and frame < to
end)

-- Keep the notifier reachable: a collected one never runs (seen in a run that allocates a lot).
_G.daytona_trace.stop = emu.add_machine_stop_notifier(function()
  local ok, err = pcall(function()
    if active then flush(); traced = traced + 1 end
    out:write(string.format("END %d %d\n", traced, events))
  end)
  if not ok then out:write("ERR " .. tostring(err) .. "\n") end
  out:close()
end)
