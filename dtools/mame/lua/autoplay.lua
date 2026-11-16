-- Coin up, pick a course with the wheel, floor the gas (MAME `daytona`). Adapted from vrtools' autoplay.lua.
-- Env: DAYTONA_COURSE (0 Beginner, 1 Advanced, 2 Expert; default 0), DAYTONA_ENTRY (an entry of the patched select
-- of dtools/arcade/select.py, patched sets only; overrides DAYTONA_COURSE), DAYTONA_ROW (the patched select's row
-- length: 16, the default, the frozen rows of sixteen; 8, the shipped row of eight since Phase 11), DAYTONA_WHEEL
-- (absolute paddle value held on the Circuit Select screen, overrides the course's zone), DAYTONA_GEAR ("auto"
-- default, "manual"), DAYTONA_COIN_AT, DAYTONA_WHEEL_FROM, DAYTONA_GAS_FROM (frames), DAYTONA_SNAPS (comma-separated
-- frames to snapshot; default the two select screens and 20 s into the race; MAME also writes one when the run
-- ends), DAYTONA_WHEEL_SCRIPT ("frame:paddle,...": from each frame on the paddle holds that value, replacing the
-- select and mission wheel until the gas; frames ascending), DAYTONA_STEPS (comma-separated frames of the 10-frame
-- gas presses, replacing the course and mission steps; "none" for no step), DAYTONA_SECOND ("entry:frame" or
-- "course:frame" with a "c" before the course, e.g. "0:7000" or "c0:7000": a second game in the same session, its coin
-- at that frame and the first game's offsets after it; the first game's gas is let go and the wheel centred 30 frames
-- before that coin, which must land in the attract mode), DAYTONA_AUTOPILOT (a file of "x z" lines, one per road
-- segment: the road line's centre in the arcade's frame; from the gas on, the paddle steers the player's car instead
-- of staying centred, see "The autopilot" below), DAYTONA_AUTOPILOT_LAPS (with DAYTONA_AUTOPILOT: once the player's
-- lap count reaches it, the gas is let go and the paddle centred; default never).
-- Runs select_log.lua when DAYTONA_SELECT_LOG is set (see there).
-- Runs memtrace.lua when DAYTONA_TRACE_LOG is set (see there; refuses to run with the two below).
-- Runs texdump.lua when DAYTONA_TEXDUMP_DIR is set (see there).
-- Runs ground_log.lua when DAYTONA_GROUND_LOG is set (see there; refused with DAYTONA_TRACE_LOG).
-- Runs progress_log.lua when DAYTONA_PROGRESS_LOG is set (see there; refused with DAYTONA_TRACE_LOG).
-- Runs tgp_trace.lua when DAYTONA_TGP_LOG is set (see there; refused with DAYTONA_TRACE_LOG and DAYTONA_GROUND_LOG).
-- Runs geo_log.lua when DAYTONA_GEO_LOG is set and renderrate.lua when DAYTONA_RATE_LOG is set (from DAYTONA_LUA_DIR).
-- Runs cells_log.lua when DAYTONA_CELLS_LOG is set (see there; refused with DAYTONA_TRACE_LOG).
-- Runs pit_log.lua when DAYTONA_PIT_LOG is set (see there).
-- Runs title_log.lua when DAYTONA_TITLE_LOG is set (see there).
--
-- What the game does (measured 2026-11-07, MAME 0.289, fresh NVRAM; snapshots read by eye):
--   frames 1-~500  boot: the operator settings page (LINK ID MASTER, CABINET TWIN, COUNTRY JPN), "NETWORK CHECKING".
--                  A coin at frame 100 is lost; one at 200-500 is kept and the select opens when the boot ends (~515).
--   attract        Japanese text (the parent set; the country is the game's own setting, see the note at the end).
--   coin           one credit starts the game at once: CREDIT never reads 1/1. A coin pressed at frame 600 opens
--                  "Circuit Select" at frame 615 (attract until 612), panels drawn by 618, 15 s timer.
--   Circuit Select three panels, the wheel moves the cursor, "STEP TO CHOOSE": a press of the gas pedal picks the
--                  course (the timer jumps to 0), then the game waits for linked cabinets ("accepting entries from
--                  other cars") until ~900; a gas pedal held down picks only once.
--   Mission Select from ~935: Automatic (left) / Manual-4 (right) by the wheel, "STEP TO START", 4 s left. Its
--                  footer warns: START + gas pedal here gives the time-attack mode (this script never does that).
--   race           "GENTLEMEN START YOUR ENGINES" ~1140-1320, race HUD and the fly-by from ~1330. Advanced and
--                  Expert: standing start, the lights turn green between 1640 and 1650. Beginner: rolling start
--                  ("AUTO DRIVE"), counts 3-2-1 from ~1750, GO! at ~1890 (the race clock starts there).
-- Unless a step is pressed, each select screen picks what is under the cursor when its timer ends (race at ~2700).
local lua_dir = os.getenv("DAYTONA_LUA_DIR")
if os.getenv("DAYTONA_GROUND_LOG") and os.getenv("DAYTONA_TRACE_LOG") then
  error("DAYTONA_GROUND_LOG cannot be combined with DAYTONA_TRACE_LOG")
end
if os.getenv("DAYTONA_PROGRESS_LOG") and os.getenv("DAYTONA_TRACE_LOG") then
  error("DAYTONA_PROGRESS_LOG cannot be combined with DAYTONA_TRACE_LOG")
end
if os.getenv("DAYTONA_TGP_LOG") and (os.getenv("DAYTONA_TRACE_LOG") or os.getenv("DAYTONA_GROUND_LOG")) then
  error("DAYTONA_TGP_LOG cannot be combined with DAYTONA_TRACE_LOG or DAYTONA_GROUND_LOG")
end
if os.getenv("DAYTONA_GROUND_LOG") and lua_dir then dofile(lua_dir .. "/ground_log.lua") end
if os.getenv("DAYTONA_PROGRESS_LOG") and lua_dir then dofile(lua_dir .. "/progress_log.lua") end
if os.getenv("DAYTONA_TGP_LOG") and lua_dir then dofile(lua_dir .. "/tgp_trace.lua") end
if os.getenv("DAYTONA_GEO_LOG") and lua_dir then dofile(lua_dir .. "/geo_log.lua") end
if os.getenv("DAYTONA_RATE_LOG") and lua_dir then dofile(lua_dir .. "/renderrate.lua") end
if os.getenv("DAYTONA_CELLS_LOG") and lua_dir then dofile(lua_dir .. "/cells_log.lua") end
if os.getenv("DAYTONA_PIT_LOG") and lua_dir then dofile(lua_dir .. "/pit_log.lua") end
if os.getenv("DAYTONA_TRACE_LOG") and lua_dir then
  if os.getenv("DAYTONA_GEO_LOG") or os.getenv("DAYTONA_RATE_LOG") then
    error("DAYTONA_TRACE_LOG cannot be combined with DAYTONA_GEO_LOG or DAYTONA_RATE_LOG")
  end
  dofile(lua_dir .. "/memtrace.lua")
end
if os.getenv("DAYTONA_TEXDUMP_DIR") and lua_dir then dofile(lua_dir .. "/texdump.lua") end
if os.getenv("DAYTONA_SELECT_LOG") and lua_dir then dofile(lua_dir .. "/select_log.lua") end
if os.getenv("DAYTONA_TITLE_LOG") and lua_dir then dofile(lua_dir .. "/title_log.lua") end

local function field(tag, name)
  local port = manager.machine.ioport.ports[tag]
  for fname, f in pairs(port.fields) do
    if fname == name then return f end
  end
  error("no field " .. name .. " on " .. tag)
end

local coin  = field(":IN0", "Coin 1")
local start = field(":IN0", "1 Player Start")
local wheel = field(":STEER", "Paddle")
local accel = field(":ACCEL", "P1 Pedal 1")

-- Paddle zones on Circuit Select (2026-11-07): 0x20 and 0x80 (centre) -> Beginner, 0x90..0xc0 -> Advanced,
-- 0xd0..0xe0 -> Expert. The values below sit inside each zone.
local ZONES = {[0] = 0x50, [1] = 0xa8, [2] = 0xe0}
-- Paddle on Mission Select (2026-11-07): 0x80 leaves the cursor on Automatic, 0xe0 moves it to Manual-4.
local GEAR_WHEEL = {auto = 0x80, manual = 0xe0}
-- The sixteen-entry select (select.paddle_for_entry: the middle paddle value of each entry's band; checked against
-- the Python model by tests/test_select.py).
local ENTRY_WHEEL = {[0] = 29, 62, 71, 81, 90, 100, 109, 124, 138, 148, 157, 167, 176, 186, 195, 228}
-- The entries whose paddle lies in the centre band (|wheel value| <= select.IDLE_BAND): the patched select ignores the
-- band until the wheel has left it (the idle wheel, Phase 9 Task 5), so DAYTONA_ENTRY of one of them first holds the
-- paddle of the entry beside it, outside the band, for LEAD_IN frames (checked by tests/test_select.py).
local LEAD, LEAD_IN = {[7] = 6, [8] = 9}, 20
-- The row of eight (Phase 11, the shipped row: DAYTONA_ROW=8; select.paddle_for_entry(k, 8), bands 32 wide) and its
-- entries whose paddle lies in the centre band (entry 3's, 119; entry 4's, 143, gives wheel value 17, outside it).
local ENTRY_WHEEL_8 = {[0] = 33, 76, 95, 119, 143, 162, 181, 223}
local LEAD_8 = {[3] = 2}
local row = os.getenv("DAYTONA_ROW") or "16"
if row == "8" then
  ENTRY_WHEEL, LEAD = ENTRY_WHEEL_8, LEAD_8
elseif row ~= "16" then
  error("DAYTONA_ROW must be 8 or 16")
end
local course = tonumber(os.getenv("DAYTONA_COURSE") or "0")
local entry = tonumber(os.getenv("DAYTONA_ENTRY") or "")
if os.getenv("DAYTONA_ENTRY") and (entry == nil or ENTRY_WHEEL[entry] == nil) then
  error("DAYTONA_ENTRY must be an entry of the row (0-" .. row - 1 .. ")")
end
local gear = os.getenv("DAYTONA_GEAR") or "auto"
local wheel_pos = tonumber(os.getenv("DAYTONA_WHEEL") or "") or (entry and ENTRY_WHEEL[entry]) or ZONES[course]
  or error("DAYTONA_COURSE must be 0, 1 or 2")
local wheel_script = {}
for f, v in string.gmatch(os.getenv("DAYTONA_WHEEL_SCRIPT") or "", "(%d+):(%d+)") do
  wheel_script[#wheel_script + 1] = {tonumber(f), tonumber(v)}
end
local gear_wheel = GEAR_WHEEL[gear] or error("DAYTONA_GEAR must be auto or manual")

-- 2026-11-07: a coin at 600 (attract on screen) opens Circuit Select at 615; the offsets below assume a coin in the
-- attract mode (from ~600 on); an earlier coin opens the select only when the boot ends (~515), not at coin + 15.
local coin_at = tonumber(os.getenv("DAYTONA_COIN_AT") or "600")
-- START 120 frames after the coin lands on Circuit Select, where it changes nothing (2026-11-07); kept from vrtools.
local start_at = coin_at + 120
-- 2026-11-07: the panels are drawn by coin + 18; the cursor follows the paddle within 20 frames.
local wheel_from = tonumber(os.getenv("DAYTONA_WHEEL_FROM") or tostring(coin_at + 40))
local course_step = coin_at + 160            -- 2026-11-07: a 10-frame step at 760 picks the course under the cursor
local mission_from = coin_at + 200           -- the paddle moves to Automatic / Manual before Mission Select (~935)
local mission_step = coin_at + 360           -- 2026-11-07: a step at 960 starts the race (GO at ~1650)
-- 2026-11-07: the race HUD is up from ~1330; the gas is held and the paddle centred from here to the end. Keep it
-- after the mission step (coin + 370): a pedal already held down does not register the step.
local gas_from = tonumber(os.getenv("DAYTONA_GAS_FROM") or tostring(coin_at + 700))
-- 2026-11-07: the race begins (green light / GO!) at 1645 on Advanced and Expert, at 1890 on Beginner.
local RACE = {[0] = coin_at + 1290, [1] = coin_at + 1045, [2] = coin_at + 1045}
-- With DAYTONA_ENTRY (ENTRIES' order): entries 1 and 2 are Advanced and Expert, entry 3 Desert City (standing start:
-- the race begins with Expert's countdown, 2026-11-09), the others Beginner's data (rolling start).
local race_at = entry and ((entry >= 1 and entry <= 3) and RACE[1] or RACE[0]) or RACE[course] or RACE[1]
local steps = {}
if os.getenv("DAYTONA_STEPS") then
  for f in string.gmatch(os.getenv("DAYTONA_STEPS"), "%d+") do steps[tonumber(f)] = true end
else
  steps[course_step] = true
  steps[mission_step] = true
end
-- The second game (DAYTONA_SECOND): the first game's script again from its coin, on its own entry or course.
local second = os.getenv("DAYTONA_SECOND")
local coin2, wheel2, lead2 = nil, nil, false
if second then
  local kind, which, at = string.match(second, "^(c?)(%d+):(%d+)$")
  which, coin2 = tonumber(which), tonumber(at)
  if not coin2 then error("DAYTONA_SECOND must be entry:frame or c<course>:frame") end
  if kind == "c" then
    wheel2 = ZONES[which] or error("DAYTONA_SECOND: no such course " .. which .. " (0, 1 or 2)")
  else
    wheel2 = ENTRY_WHEEL[which] or error("DAYTONA_SECOND: no such entry " .. which .. " (0-" .. row - 1 .. ")")
    lead2 = LEAD[which] or false
  end
  if coin2 <= gas_from + 60 then error("DAYTONA_SECOND: the second coin must come after the first race") end
  steps[coin2 + 160] = true
  steps[coin2 + 360] = true
end
local releases = {}
for f in pairs(steps) do releases[f + 10] = true end
local snaps = {}                              -- Circuit Select with the cursor placed, Mission Select, race + 20 s
local default_snaps = string.format("%d,%d,%d", coin_at + 150, coin_at + 350, race_at + 1150)   -- 57.52 Hz
for f in string.gmatch(os.getenv("DAYTONA_SNAPS") or default_snaps, "%d+") do snaps[tonumber(f)] = true end

-- The autopilot (DAYTONA_AUTOPILOT; Phase 10 Task 4: Mermaid Lake's gate drives one lap at race pace, then stops;
-- the centred paddle there finishes its lap only off the walls at half the pace and runs on into lap 2). The player's
-- car is the 0x300-byte struct at 0x00505180 (progress_log.lua): segment u16 at +0x12c, lap u8 at +0x6f, position
-- f32 at +0x1c (x) and +0x24 (z). Its heading is its motion (the position's change from frame to frame, smoothed by
-- AP_SMOOTH); the paddle turns it towards the road centre AP_AHEAD segments past its own: 0x80 - AP_GAIN * turn, the
-- turn the signed angle (radians, atan2 of the x-z cross and dot products) from the motion to that point (its sign
-- measured: from +z towards +x is a paddle above 0x80), held within AP_LOCK of the centre, centred below AP_SLOW units
-- a frame (the standing start). It reads the emulated state only, so a run is as deterministic as without it.
local AP_AHEAD, AP_GAIN, AP_LOCK, AP_SLOW, AP_SMOOTH = 6, 160, 0x60, 0.05, 0.5
local autopilot, ap_laps = nil, tonumber(os.getenv("DAYTONA_AUTOPILOT_LAPS") or "")
if os.getenv("DAYTONA_AUTOPILOT") then
  autopilot = {}
  for line in io.lines(os.getenv("DAYTONA_AUTOPILOT")) do
    local x, z = string.match(line, "^%s*(%S+)%s+(%S+)%s*$")
    autopilot[#autopilot + 1] = {assert(tonumber(x)), assert(tonumber(z))}
  end
  if #autopilot < 2 then error("DAYTONA_AUTOPILOT: fewer than two road points") end
end
local AP_CAR = 0x00505180
local ap_mem, ap_last, ap_hx, ap_hz = manager.machine.devices[":maincpu"].spaces["program"], nil, 0.0, 0.0
local function ap_f32(bits) return (string.unpack("<f", string.pack("<I4", bits & 0xffffffff))) end
local function ap_paddle()
  local x, z = ap_f32(ap_mem:read_u32(AP_CAR + 0x1c)), ap_f32(ap_mem:read_u32(AP_CAR + 0x24))
  local seg, last = ap_mem:read_u16(AP_CAR + 0x12c), ap_last
  ap_last = {x, z}
  if not last then return 0x80 end
  ap_hx = AP_SMOOTH * ap_hx + (1 - AP_SMOOTH) * (x - last[1])
  ap_hz = AP_SMOOTH * ap_hz + (1 - AP_SMOOTH) * (z - last[2])
  if math.sqrt(ap_hx * ap_hx + ap_hz * ap_hz) < AP_SLOW or seg >= #autopilot then return 0x80 end
  local t = autopilot[(seg + AP_AHEAD) % #autopilot + 1]
  local tx, tz = t[1] - x, t[2] - z
  local turn = math.atan(ap_hx * tz - ap_hz * tx, ap_hx * tx + ap_hz * tz)
  return math.floor(0x80 + 0.5 - math.max(-AP_LOCK, math.min(AP_LOCK, AP_GAIN * turn)))
end

local frame = 0
emu.register_frame_done(function()
  frame = frame + 1
  if coin2 and frame >= coin2 - 30 then         -- the second game: the first one's script from coin2
    local f = frame - coin2
    if f == -30 then accel:set_value(0x20); wheel:set_value(0x80) end
    if f == 0 then coin:set_value(1) elseif f == 10 then coin:set_value(0) end
    if f == 120 then start:set_value(1) elseif f == 130 then start:set_value(0) end
    if f >= 40 and f < 700 then
      wheel:set_value(f < 200 and (lead2 and f < 40 + LEAD_IN and ENTRY_WHEEL[lead2] or wheel2) or gear_wheel)
    end
    if steps[frame] then accel:set_value(0xe0) elseif releases[frame] then accel:set_value(0x20) end
    if f == 700 then wheel:set_value(0x80) end
    if f >= 700 then accel:set_value(0xe0) end
    if snaps[frame] then manager.machine.video:snapshot() end
    return
  end
  if frame == coin_at then coin:set_value(1) elseif frame == coin_at + 10 then coin:set_value(0) end
  if frame == start_at then start:set_value(1) elseif frame == start_at + 10 then start:set_value(0) end
  if #wheel_script > 0 then
    if frame < gas_from then
      local v = nil
      for _, fv in ipairs(wheel_script) do if frame >= fv[1] then v = fv[2] end end
      if v then wheel:set_value(v) end
    end
  elseif frame >= wheel_from and frame < gas_from then
    local lead = entry and LEAD[entry] and not os.getenv("DAYTONA_WHEEL") and frame < wheel_from + LEAD_IN
    wheel:set_value(frame < mission_from and (lead and ENTRY_WHEEL[LEAD[entry]] or wheel_pos) or gear_wheel)
  end
  if steps[frame] then accel:set_value(0xe0)
  elseif releases[frame] then accel:set_value(0x20) end
  if frame == gas_from then wheel:set_value(0x80) end
  if frame >= gas_from then accel:set_value(0xe0) end
  if autopilot and frame >= gas_from then
    if ap_laps and ap_mem:read_u8(AP_CAR + 0x6f) >= ap_laps then
      accel:set_value(0x20); wheel:set_value(0x80)
    else
      wheel:set_value(ap_paddle())
    end
  end
  if snaps[frame] then manager.machine.video:snapshot() end
end)
-- Region: the I/O board's three dipswitch banks are all "unused" in MAME 0.289 (model2.cpp ioboard_dipswitches), so
-- -dipswitch cannot change the language; the boot page shows COUNTRY JPN, a game setting kept in NVRAM (test menu).
-- Manual: the car starts in 1st (the HUD says SHIFT at ~112 km/h); this script does not shift (the :GEARS fields).
