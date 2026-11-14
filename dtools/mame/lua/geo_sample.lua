-- Shared by geo_log.lua and renderrate.lua (dofile'd once; state in _G.daytona_geo): sample the geometry buffer as
-- MAME's geo_parse reads it, and track the read start address.
--
-- The model2 screen has VIDEO_UPDATE_AFTER_VBLANK (model2.cpp line 2517): geo_parse runs in screen_vblank at vblank
-- START (line 384 of 424, screen.cpp vblank_begin), while frame_update, and with it Lua's frame_done, runs at vblank
-- END (vblank_end). The vblank IRQ raised right after geo_parse lets the i960 start the next list in between.
-- Measured 2026-11-07 (60 s autoplay run on course 0, write taps on 0x00800000-0x00807fff and 0x00900000-0x0097ffff,
-- each write classed by whether it fell in the vblank): in every one of the 1,561 race frames the i960 writes in
-- that gap (per frame: the write start address once, ~2 command-port words, ~7 FIFO words, ~7.5 direct buffer
-- writes of 0xffffffff to word 0x7ffc): the head of the NEXT list, written into the other half of the double buffer
-- (the read start alternates 0x0 / 0x10000), not over the parsed one; in that run a frame_done sample differed from
-- the vblank-start sample only in words 0x7ffc-0x7ffe. Outside the race the read start address is sometimes written
-- in the gap too. Nothing in the driver guarantees the parsed half stays intact, so a frame_done read is not trusted.
--
-- So the state is captured at the first geometry write after vblank start (a write tap runs before the write lands),
-- or at frame_done when the vblank saw no write. No i960 write reaches the buffer before that first tapped write;
-- the TGP is not tapped and could write it too (its bank register, bit 0x400000, maps the buffer into its data space:
-- model2.cpp lines 471-514), which this method cannot rule out. The vblank test is exact to within an instruction: an
-- i960 store in the last instruction of a timeslice can land a few cycles past vblank start (negligible).
-- st.capture() is idempotent per frame; each logger calls it at the start of its frame_done. Listeners registered with
-- table.insert(st.listeners, fn) are called once per frame as fn(read_start), with the buffer still as parsed.
-- In the 30 Hz mode (video control bit 0 set) geo_parse runs on even frames only; the capture still runs every frame.
if not _G.daytona_geo then
  local mem = manager.machine.devices[":maincpu"].spaces["program"]
  local screen = manager.machine.screens[":screen"]
  local st = {read_start = 0, writes = 0, listeners = {}, key = nil}

  local function in_vblank()           -- during the CPU's timeslice the scheduler time is the CPU's local time
    return screen:time_until_vblank_end():as_double() < screen:time_until_vblank_start():as_double()
  end

  function st.capture()
    local k = screen:frame_number()    -- incremented after frame_update: the same from vblank start to frame_done
    if st.key == k then return end
    st.key = k
    for _, fn in ipairs(st.listeners) do fn(st.read_start) end
  end

  -- Kept in the global table: a collected tap is removed. One tap per range (a second tap on a range may not fire).
  st.taps = {
    -- command port 0x00800000-0x00800fff, write start 0x00801008, read start 0x00803008, FIFO 0x00804000-0x00807fff
    mem:install_write_tap(0x00800000, 0x00807fff, "daytona_geo_port", function(offset, data, mask)
      if in_vblank() then st.capture() end
      if (offset & 0x7ffc) == 0x3008 then
        st.read_start = data & 0xfffff
        st.writes = st.writes + 1
      end
    end),
    -- buffer RAM and its mirrors (push_geo_data writes it directly, so the port tap covers the FIFO path)
    mem:install_write_tap(0x00900000, 0x0097ffff, "daytona_geo_buffer", function(offset, data, mask)
      if in_vblank() then st.capture() end
    end),
  }
  _G.daytona_geo = st
end
