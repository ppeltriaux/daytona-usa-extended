-- Race the Dreamcast Daytona USA (MAME `dc`): dc_select.lua picks the course and starts the race, then this script
-- drives the player's car round a route, closed loop: the gas held, the analog stick steered towards the route.
-- Env: everything dc_select.lua reads, plus DC_ROUTE (a file of "x z" lines, Dreamcast coordinates, in driving
-- order: the course's AI line 0), DC_DRIVE_FROM (the frame the gas goes down), DC_PLAYER (hex address of the player
-- car's x; y and z follow as f32; default 0x0c34416c, measured below). Prints "CAR <frame> <x> <z>" (the player's
-- position, Dreamcast coordinates) every frame from DC_DRIVE_FROM.
--
-- Measured 2026-11-09 (MAME 0.289, Desert City, select entry 3, `DC_START=1`): the lights go green and the lap clock
-- starts at frame ~5879; three RAM dumps on the main straight (frames 6000, 6100, 6200, gas from 5880) put 20 car
-- records 0x30c bytes apart from 0x0c34416c, each (x, y, z) f32; the player's is the first (x -337.7, y 25.1, z
-- falling 53.2, 14.9, -44.3: driving down the straight, the last on the grid, its x fixed by the centred stick).
-- The pad: the R trigger (`:P1:A0`) is the accelerator and the stick (`:P1:A2`) steers, above 128 to the right (both
-- seen in snapshots); the L trigger (`:P1:A1`) is pressed as the brake (its effect not measured on its own). With this
-- driver the car reaches segment 136 at frame 7209 and completes the lap (docs/spike/2026-11-convert.md, "Desert City
-- seen").
local lua_dir = os.getenv("DAYTONA_LUA_DIR")
dofile(lua_dir .. "/dc_select.lua")

local route = {}
local route_path = os.getenv("DC_ROUTE") or error("DC_ROUTE is required")
for line in io.lines(route_path) do
  local x, z = line:match("^(%S+)%s+(%S+)$")
  if x then table.insert(route, {tonumber(x), tonumber(z)}) end
end
assert(#route > 2, "DC_ROUTE holds no route")
local drive_from = tonumber(os.getenv("DC_DRIVE_FROM") or "5880")
local at = tonumber(os.getenv("DC_PLAYER") or "0x0c34416c")

local function one(tag)
  for _, f in pairs(manager.machine.ioport.ports[tag].fields) do return f end
end
local gas, brake, stick = one(":P1:A0"), one(":P1:A1"), one(":P1:A2")
local mem = manager.machine.devices[":maincpu"].spaces["program"]
local function f32(a) return (string.unpack("<f", string.pack("<I4", mem:read_u32(a)))) end

local LOOK = 5                  -- route points ahead (about 35 units)
local GAIN = 260                -- stick units per radian of heading error
local n, near, prev, heading = #route, nil, nil, nil
local frame = 0
emu.register_frame_done(function()
  frame = frame + 1
  if frame < drive_from then return end
  local x, z = f32(at), f32(at + 8)
  print(string.format("CAR %d %.3f %.3f", frame, x, z))
  if near == nil then
    local best
    for i = 1, n do
      local d = (route[i][1] - x) ^ 2 + (route[i][2] - z) ^ 2
      if not best or d < best then best, near = d, i end
    end
  else
    local best, bi = nil, near
    for k = -3, 12 do
      local i = (near - 1 + k) % n + 1
      local d = (route[i][1] - x) ^ 2 + (route[i][2] - z) ^ 2
      if not best or d < best then best, bi = d, i end
    end
    near = bi
  end
  if prev and ((x - prev[1]) ^ 2 + (z - prev[2]) ^ 2) > 0.0025 then
    heading = math.atan(x - prev[1], -(z - prev[2]))            -- the arcade's frame: z' = -z, right turns positive
  end
  prev = {x, z}
  local t = route[(near - 1 + LOOK) % n + 1]
  local err = 0
  if heading then
    err = math.atan(t[1] - x, -(t[2] - z)) - heading
    err = (err + math.pi) % (2 * math.pi) - math.pi
  end
  stick:set_value(math.max(0, math.min(255, math.floor(128 + GAIN * err + 0.5))))
  local a = math.abs(err)
  gas:set_value(a > 0.35 and 0 or (a > 0.2 and 0x80 or 0xff))
  brake:set_value(a > 0.5 and 0xff or 0)
end)
