-- TGP-side tracer (dofile'd by autoplay.lua when DAYTONA_TGP_LOG is set; refused with DAYTONA_TRACE_LOG and
-- DAYTONA_GROUND_LOG, whose taps may overlap). Device :copro_tgp (MB86234, MAME's mb86233 core), MAME 0.289.
-- Env: DAYTONA_TGP_LOG (absolute output path), DAYTONA_TGP_FROM / DAYTONA_TGP_TO (frame window, inclusive /
-- exclusive; defaults 0 / forever).
-- What it taps (src/mame/sega/model2.cpp, copro_tgp_rf_map / copro_tgp_io_map): a write tap on rf word 3 (the bank
-- register, copro_tgp_bank_w) and a read tap on io words 0x0000-0xffff (the bank view, copro_tgp_memory_r: word
-- address (bank & 0xff0000) | offset; bank bit 0x800000 = copro_data, 0x400000 = buffer RAM).
-- Lines:
--   S <space> ...                     first line: the device's address spaces, by their Lua names
--   B <frame> <tgp_pc> <value>        every bank-register write (hex)
--   R <frame> <tgp_pc> <word_address> <count>   reads through the bank, aggregated per frame per (pc, address), hex;
--                                     only while the bank selects copro_data or buffer RAM (bank & 0xc00000)
--   END <frames_traced> <events>
-- <tgp_pc> is the core's "CURPC" state (m_ppc: the word address of the instruction making the access; its "PC"
-- state is not exposed to Lua). Frames are counted like memtrace.lua (frame_done calls).
-- One tap per range; handlers are kept in _G.daytona_tgp.
local path = os.getenv("DAYTONA_TGP_LOG")
if path then
  if os.getenv("DAYTONA_TRACE_LOG") or os.getenv("DAYTONA_GROUND_LOG") then
    error("DAYTONA_TGP_LOG cannot be combined with DAYTONA_TRACE_LOG or DAYTONA_GROUND_LOG")
  end
  local from = tonumber(os.getenv("DAYTONA_TGP_FROM") or "0")
  local to = tonumber(os.getenv("DAYTONA_TGP_TO") or "") or math.huge
  local fmt = string.format
  local tgp = manager.machine.devices[":copro_tgp"] or error("no device :copro_tgp")
  local out = assert(io.open(path, "w"))
  local names = {}
  for name, _ in pairs(tgp.spaces) do names[#names + 1] = name end
  table.sort(names)
  out:write("S " .. table.concat(names, " ") .. "\n")
  local pc = tgp.state["CURPC"] or error("no CURPC state on :copro_tgp")

  local frame, traced, events = 0, 0, 0
  local active = from <= 0 and to > 0
  local bank = 0
  local agg, lines = {}, {}

  local taps = {}
  taps[#taps + 1] = tgp.spaces["rf"]:install_write_tap(3, 3, "daytona_tgp_bank", function(offset, data, mask)
    bank = data & 0xffffffff
    if active then lines[#lines + 1] = fmt("B %d %x %x", frame, pc.value, bank) end
  end)
  taps[#taps + 1] = tgp.spaces["io"]:install_read_tap(0x0000, 0xffff, "daytona_tgp_read", function(offset, data, mask)
    if not active or bank & 0xc00000 == 0 then return end
    local key = pc.value .. "|" .. ((bank & 0xff0000) | (offset & 0xffff))
    agg[key] = (agg[key] or 0) + 1
  end)
  _G.daytona_tgp = {taps = taps}

  local function flush()
    local reads = {}
    for key, count in pairs(agg) do
      local p, a = string.match(key, "^(%d+)|(%d+)$")
      reads[#reads + 1] = {tonumber(p), tonumber(a), count}
    end
    table.sort(reads, function(x, y) return x[2] < y[2] or (x[2] == y[2] and x[1] < y[1]) end)
    for _, l in ipairs(lines) do out:write(l, "\n") end
    for _, r in ipairs(reads) do out:write(fmt("R %d %x %x %d\n", frame, r[1], r[2], r[3])) end
    events = events + #lines + #reads
    agg, lines = {}, {}
  end

  emu.register_frame_done(function()
    if active then flush(); traced = traced + 1 end
    frame = frame + 1
    active = frame >= from and frame < to
  end)
  _G.daytona_tgp.stop = emu.add_machine_stop_notifier(function()
    local ok, err = pcall(function()
      if active then flush(); traced = traced + 1 end
      out:write(fmt("END %d %d\n", traced, events))
    end)
    if not ok then out:write("ERR " .. tostring(err) .. "\n") end
    out:close()
  end)
end
