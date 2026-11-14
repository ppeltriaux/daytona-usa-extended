-- Texture-side RAM dumper (dofile'd by autoplay.lua when DAYTONA_TEXDUMP_DIR is set); read by
-- dtools/arcade/texture.py (`load_dump`). Env: DAYTONA_TEXDUMP_DIR (absolute output directory), DAYTONA_TEXDUMP_AT
-- (comma-separated frames, counted like autoplay.lua). At each frame it writes, read through the i960 program space
-- (MAME src/mame/sega/model2.cpp, model2_base_mem / model2_tgp_mem):
--   tex0.<f>.bin   0x12000000-0x120fffff  texture RAM 0 (the 1 MB sheet m_textureram0[0..0x3ffff])
--   tex1.<f>.bin   0x12400000-0x124fffff  texture RAM 1
--   luma.<f>.bin   0x12800000-0x1281ffff  luma RAM (umask32 0xff: byte k of the table is byte 4k of the dump)
--   pal.<f>.bin    0x01800000-0x01803fff  palette RAM (u16 LE)
--   xlat.<f>.bin   0x01810000-0x0181bfff  colour translation RAM (u16 LE)
--   tile.<f>.bin   0x01000000-0x0100ffff  tile RAM (System 24 tilemap chip, u16 LE; MAME segaic24.cpp tile_r)
--   char.<f>.bin   0x01080000-0x010fffff  char RAM (8 x 8 4 bpp characters, 32 bytes each; segaic24.cpp char_r)
-- The rasterizer's own texture RAM (tpa/tha with bit 23 set, filled by geometry command 0x04) is not on the i960
-- bus and is not dumped.
local dir = assert(os.getenv("DAYTONA_TEXDUMP_DIR"), "DAYTONA_TEXDUMP_DIR")
local mem = manager.machine.devices[":maincpu"].spaces["program"]
local at = {}
for f in string.gmatch(os.getenv("DAYTONA_TEXDUMP_AT") or "", "%d+") do at[tonumber(f)] = true end
local REGIONS = {{"tex0", 0x12000000, 0x120fffff}, {"tex1", 0x12400000, 0x124fffff},
                 {"luma", 0x12800000, 0x1281ffff}, {"pal", 0x01800000, 0x01803fff}, {"xlat", 0x01810000, 0x0181bfff},
                 {"tile", 0x01000000, 0x0100ffff}, {"char", 0x01080000, 0x010fffff}}
local frame = 0
emu.register_frame_done(function()
  frame = frame + 1
  if not at[frame] then return end
  for _, r in ipairs(REGIONS) do
    local fh = assert(io.open(string.format("%s/%s.%d.bin", dir, r[1], frame), "wb"))
    fh:write(mem:read_range(r[2], r[3], 8))
    fh:close()
  end
end)
