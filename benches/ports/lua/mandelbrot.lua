-- mandelbrot 200x100x200, ported from acvus-interpreter-test/benches/mandelbrot.rs.
--
-- Obligation across artifacts: the three nested loops and the compound loop
-- condition are that file's `rust_mandelbrot`, which is what the bench's
-- `rust/us` column times.
--
-- Usage: lua mandelbrot.lua <row> [reps]

local REPS = 4

local function mandelbrot(w, h, max_i)
  local total = 0
  local py = 0
  while py < h do
    local px = 0
    while px < w do
      local cx = -2.0 + 3.0 * px / w
      local cy = -1.2 + 2.4 * py / h
      local x = 0.0
      local y = 0.0
      local i = 0
      while i < max_i and x * x + y * y < 4.0 do
        local xt = x * x - y * y + cx
        y = 2.0 * x * y + cy
        x = xt
        i = i + 1
      end
      total = total + i
      px = px + 1
    end
    py = py + 1
  end
  return total
end

local row, reps = ...
reps = tonumber(reps) or REPS
local w, h, max_i = row:match("^(%d+)x(%d+)x(%d+)$")
w, h, max_i = tonumber(w), tonumber(h), tonumber(max_i)

mandelbrot(w, h, max_i)
local samples = {}
local value
for r = 1, reps do
  local start = os.clock()
  value = mandelbrot(w, h, max_i)
  samples[r] = os.clock() - start
end
table.sort(samples)
local median = samples[math.floor(#samples / 2) + 1]
print(string.format("%s\t%.1f\t%.0f", row, median * 1e6, value))
