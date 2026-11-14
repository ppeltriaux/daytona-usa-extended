-- Ground-query probe (dofile'd by autoplay.lua when DAYTONA_GROUND_LOG is set; parsed by runner.parse_ground).
-- Env: DAYTONA_GROUND_LOG (absolute output path). Allowed with DAYTONA_RATE_LOG; refused with DAYTONA_TRACE_LOG.
-- The i960 ground query 0x11c28 (docs/formats/model2-course-tables.md, "The ground under the road") does:
--   st -1 -> 0x91fff0; st 0x3636 -> 0x880360 (TGP function 0x36); five FIFO words at 0x884000+: dest 0x17ffc, y, grid
--   value, x, z (f32 y, x, z as bit patterns); poll 0x91fff0 until it is 0 (the TGP's answer is ready); ld height from
--   0x91fff4 and polygon index from 0x91fff8 (both always read); height word == 0 means "no ground": the code
--   (cmpobe 0,g0 at 0x11c70) then returns -0.1 (0xbdcccccd) and ignores the polygon.
-- "Read none" here is exactly that: the i960 read a height word of 0, so height and poly are logged as "-".
-- Lines:
--   G <frame> <n> <dest_hex> <y> <id_hex> <x> <z> <height> <poly_hex>   one per function-0x36 call; n counts calls in
--     the frame from 1; floats %.4f; height/poly are the values the i960 read at 0x91fff4 / 0x91fff8 (read tap).
--   P <frame> <x> <y> <z>   player car position (car struct 0x00505180, f32 at +0x1c/+0x20/+0x24), once per frame.
--   S <frame> <slot>        a course load's TGP function 0x0d call (0x0d0d to 0x8800d0, the slot word to the FIFO:
--     docs/formats/model2-tgp-ground.md, section 3): the TGP takes the list block and pool from slot-table words
--     0x10 + slot and 0x20 + slot.
--   END <frames>            frames seen.
-- Frames are counted like memtrace.lua (frame_done calls; an event belongs to the count of frames finished before
-- it). A call whose answer the i960 never read (the run ended mid-call) is not logged.
-- Taps: write 880360-88036f (function 0x36 call; arms a capture of the next five FIFO words), write 8800d0-8800df
-- (function 0x0d; arms a capture of the next FIFO word), write 884000-887fff
-- (the FIFO; other functions' words pass when not armed), read 91fff0-91fffb (results). One tap per range; handlers
-- are kept in _G.daytona_ground.
local path = os.getenv("DAYTONA_GROUND_LOG")
if path then
  if os.getenv("DAYTONA_TRACE_LOG") then
    error("DAYTONA_GROUND_LOG cannot be combined with DAYTONA_TRACE_LOG")
  end
  local mem = manager.machine.devices[":maincpu"].spaces["program"]
  local out = assert(io.open(path, "w"))
  local fmt = string.format
  local function f32(bits) return (string.unpack("<f", string.pack("<I4", bits & 0xffffffff))) end

  local frame, n = 0, 0
  local args, nargs = nil, 0        -- FIFO words of the call being captured (nil: not armed)
  local slot_armed = false          -- function 0x0d called: the next FIFO word is the slot
  local call = nil                  -- a captured call waiting for its result: {args..., height, poly}
  local taps = {}

  taps[#taps + 1] = mem:install_write_tap(0x880360, 0x88036f, "daytona_ground_fn", function(offset, data, mask)
    args, nargs, call = {}, 0, nil
  end)
  taps[#taps + 1] = mem:install_write_tap(0x8800d0, 0x8800df, "daytona_ground_slot", function(offset, data, mask)
    slot_armed = true
  end)
  taps[#taps + 1] = mem:install_write_tap(0x884000, 0x887fff, "daytona_ground_fifo", function(offset, data, mask)
    if slot_armed then
      slot_armed = false
      out:write(fmt("S %d %d\n", frame, data & 0xffffffff))
      return
    end
    if not args then return end
    nargs = nargs + 1
    args[nargs] = data & 0xffffffff
    if nargs == 5 then call, args = {a = args}, nil end
  end)
  taps[#taps + 1] = mem:install_read_tap(0x91fff0, 0x91fffb, "daytona_ground_res", function(offset, data, mask)
    if not call then return end
    local word = offset & ~3
    if word == 0x91fff4 then
      call.height = data & 0xffffffff
    elseif word == 0x91fff8 and call.height then
      local a = call.a
      n = n + 1
      local h, p = "-", "-"
      if call.height ~= 0 then h, p = fmt("%.4f", f32(call.height)), fmt("%x", data & 0xffffffff) end
      out:write(fmt("G %d %d %x %.4f %x %.4f %.4f %s %s\n", frame, n, a[1], f32(a[2]), a[3], f32(a[4]), f32(a[5]), h, p))
      call = nil
    end
  end)
  _G.daytona_ground = {taps = taps}

  emu.register_frame_done(function()
    out:write(fmt("P %d %.4f %.4f %.4f\n", frame, f32(mem:read_u32(0x0050519c)), f32(mem:read_u32(0x005051a0)),
                  f32(mem:read_u32(0x005051a4))))
    frame = frame + 1
    n = 0
    if frame % 60 == 0 then out:flush() end
  end)
  _G.daytona_ground.stop = emu.add_machine_stop_notifier(function()
    out:write(fmt("END %d\n", frame))
    out:close()
  end)
end
