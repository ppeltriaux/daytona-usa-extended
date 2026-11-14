-- Boot the Dreamcast Daytona USA (MAME `dc`, BIOS 101d, fresh NVRAM) to the course select by timed presses on
-- controller 1, step through the courses, and optionally start the race.
-- Env: DC_COURSE (select entry 0-7, default 0: the number of RIGHT steps on the select), DC_START ("1": confirm the
-- course and start the race), DC_SNAPS (comma-separated frames to snapshot; default the select at 3690 and each
-- entry after its step; MAME also writes one when the run ends), DC_RAM_DUMPS (frames at which to write main RAM,
-- 16 MB from 0x0c000000, to ram_<frame>.bin in MAME's cwd). Prints "SNAP <frame>" and "DUMP <frame>" lines.
--
-- What the game does (measured 2026-11-08, MAME 0.289; snapshots read by eye; frames counted by frame_done, ~49 per
-- emulated second):
--   1-~550      BIOS: the Dreamcast logo.
--   ~600        BIOS "Set Date/Time" (empty NVRAM): cursor on the month; RIGHT x5 reaches "Select", A leaves.
--   ~900-2000   SEGA licence, Hasbro Interactive, AV / SEGA, Genki / Wave Master, ADX logos.
--   ~2100       MEMORY CARD: no VMU, "start without saving? Yes" pre-selected: A.
--   ~2300-2700  NOW LOADING, the attract ("PRESS START BUTTON"); START at 2600 shows the title logo, START at 2900
--               opens MODE SELECT (~3000) with SINGLE RACE under the cursor: A at 3050.
--   ~3300       Single Race setup (CAR / COURSE / START, the course Three Seven Speedway): DOWN to COURSE, A.
--   3690        COURSE SETTING, "SELECT COURSE!", entry 0 (Three Seven Speedway). RIGHT (held 20 frames; a 6-frame
--               press is enough in the menus, but a 20-frame hold on a menu item repeats) steps one entry; the
--               preview is drawn by +100. Order: Three Seven Speedway, Dinosaur Canyon, Sea-Side Street Galaxy,
--               Desert City, National Park Speedway, Circuit Pixie, Rin Rin Rink, Mermaid Lake, then round again.
--   DC_START    DOWN x4 (6 frames, 100 apart) from SELECT to OK, A: back on the setup; DOWN to START, A: "LADIES &
--               GENTLEMEN START YOUR ENGINES" ~+850, the race from ~+1450.
local function fields(tag)
  local out = {}
  for name, f in pairs(manager.machine.ioport.ports[tag].fields) do out[name] = f end
  return out
end
local pad = fields(":P1:0")
local function button(name) return pad[name] or error("no field " .. name .. " on :P1:0") end

local presses = {}                            -- {frame, field, hold}
local function press(frame, name, hold) table.insert(presses, {frame, button(name), hold or 6}) end

for i = 0, 4 do press(700 + 20 * i, "P1 RIGHT") end   -- date screen: month -> day -> year -> hour -> minute -> Select
press(830, "P1 A")
press(2150, "P1 A")                           -- memory card: Yes
press(2600, "P1 START")
press(2900, "P1 START")
press(3050, "P1 A")                           -- SINGLE RACE
press(3400, "P1 DOWN")
press(3460, "P1 A")                           -- COURSE

local course = tonumber(os.getenv("DC_COURSE") or "0")
if course == nil or course < 0 or course > 7 or course % 1 ~= 0 then error("DC_COURSE must be 0-7") end
local SELECT_AT, STEP = 3690, 150
local default_snaps = {SELECT_AT}
local f = SELECT_AT + 10
for _ = 1, course do
  press(f, "P1 RIGHT", 20)
  table.insert(default_snaps, f + 100)
  f = f + STEP
end
if os.getenv("DC_START") == "1" then
  for _ = 1, 4 do press(f, "P1 DOWN"); f = f + 100 end
  press(f, "P1 A")                            -- OK
  press(f + 300, "P1 DOWN")
  press(f + 400, "P1 A")                      -- START
end

local snaps = {}
local snap_list = os.getenv("DC_SNAPS") or table.concat(default_snaps, ",")
for s in string.gmatch(snap_list, "%d+") do snaps[tonumber(s)] = true end
local dumps = {}
for s in string.gmatch(os.getenv("DC_RAM_DUMPS") or "", "%d+") do dumps[tonumber(s)] = true end

local frame = 0
emu.register_frame_done(function()
  frame = frame + 1
  for _, p in ipairs(presses) do
    if frame == p[1] then p[2]:set_value(1) elseif frame == p[1] + p[3] then p[2]:set_value(0) end
  end
  if snaps[frame] then
    manager.machine.video:snapshot()
    print(string.format("SNAP %d", frame))
  end
  if dumps[frame] then
    local ram = manager.machine.devices[":maincpu"].spaces["program"]:read_range(0x0c000000, 0x0cffffff, 8)
    local fh = assert(io.open(string.format("ram_%d.bin", frame), "wb"))
    fh:write(ram)
    fh:close()
    print(string.format("DUMP %d", frame))
  end
end)
