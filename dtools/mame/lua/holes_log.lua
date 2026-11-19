-- Holes probe (dofile'd by autoplay.lua when DAYTONA_HOLES_LOG is set; v0.2, Pascal's Rin Rin Rink report): every
-- frame the tile layers' palette (palette RAM words 0x0000-0x0fff; the polygons' palette starts at word 0x1000,
-- texture.POLY_PALETTE) is painted magenta, so the 2D backdrop behind the polygons draws magenta wherever no polygon
-- covers it; then a grid of screen pixels in the picture's lower part (rows ROW0..ROW1, every STEP pixels; the HUD's
-- tiles turn magenta too, but none sits there) is read and the magenta ones counted. A ground hole (the backdrop
-- showing through where the road or the ground should be) is magenta at the bottom of the screen.
-- Env: DAYTONA_HOLES_LOG (absolute output path), DAYTONA_HOLES_FROM (default 1700).
-- Line: "H <frame> <magenta> <samples>"; END <frames>.
local path = os.getenv("DAYTONA_HOLES_LOG")
if path then
  local mem = manager.machine.devices[":maincpu"].spaces["program"]
  local screen = manager.machine.screens[":screen"]
  local out = assert(io.open(path, "w"))
  local from = tonumber(os.getenv("DAYTONA_HOLES_FROM") or "1700")
  local ROW0, ROW1, STEP = 260, 380, 8
  local MAGENTA = 0x7C1F                        -- xBGR555: red 31, blue 31
  local frame = 0
  local function paint()
    for a = 0x01800000, 0x01801FFE, 2 do mem:write_u16(a, MAGENTA) end
  end
  emu.register_frame_done(function()
    frame = frame + 1
    if frame < from - 2 then return end
    paint()
    if frame >= from then
      local n, m = 0, 0
      local w = screen.width
      for y = ROW0, ROW1, STEP do
        for x = 4, w - 4, STEP do
          local p = screen:pixel(x, y)
          local r, g, b = (p >> 16) & 0xff, (p >> 8) & 0xff, p & 0xff
          n = n + 1
          if r > 200 and b > 200 and g < 60 then m = m + 1 end
        end
      end
      out:write(string.format("H %d %d %d\n", frame, m, n))
      if frame % 60 == 0 then out:flush() end
    end
  end)
  _G.daytona_holes = {stop = emu.add_machine_stop_notifier(function()
    out:write(string.format("END %d\n", frame))
    out:close()
  end)}
end
