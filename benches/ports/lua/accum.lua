-- accum rows at n = 1 000 000, ported from acvus-interpreter-test/benches/accum.rs.
--
-- Obligation across artifacts: each row here is the semantics of that file's
-- `rust_*` twin, which is what the bench's `rust/us` column times.
--
-- Decisions not to build, so that one file parses and runs under both Lua 5.4
-- and LuaJIT (which is Lua 5.1 and rejects `//` at parse time):
--   * `collatz while` halves with `/`, taken only where the numerator is even,
--     so the quotient is exact in both; under 5.4 that makes the accumulator a
--     float, and every partial sum of this row is below 2^53.
--   * Lua has no lazy pipeline, so `range | sum` is the numeric `for` that
--     `for range` is, and the two rows are the same program here.
--
-- Usage: lua accum.lua <row> [n] [reps]

local REPS = 4
local N = 1000000

local function id_of(i)
  return i
end

local function even_of(i)
  return i % 2 == 0
end

local function step(x)
  return x + 1
end

local function add_one(x)
  return x + 1
end

local rows = {}

rows["int while"] = function(n)
  local acc = 0
  local i = 0
  while i < n do
    acc = acc + i
    i = i + 1
  end
  return acc
end

rows["float while"] = function(n)
  local acc = 0.0
  local i = 0
  while i < n do
    acc = acc + (i + 0.0)
    i = i + 1
  end
  return acc
end

rows["range | sum"] = function(n)
  local acc = 0
  for i = 0, n - 1 do
    acc = acc + i
  end
  return acc
end

rows["map id | sum"] = function(n)
  local acc = 0
  for i = 0, n - 1 do
    acc = acc + id_of(i)
  end
  return acc
end

rows["map add | sum"] = function(n)
  local acc = 0
  for i = 0, n - 1 do
    acc = acc + add_one(i)
  end
  return acc
end

rows["branch while"] = function(n)
  local acc = 0
  local i = 0
  while i < n do
    if even_of(i) then
      acc = acc + i
    end
    i = i + 1
  end
  return acc
end

rows["call while"] = function(n)
  local i = 0
  while i < n do
    i = step(i)
  end
  return i
end

rows["collatz while"] = function(n)
  local acc = 0
  local i = 0
  while i < n do
    local d
    if i % 2 == 0 then
      d = i / 2
    else
      d = i * 3 + 1
    end
    acc = acc + d
    i = i + 1
  end
  return acc
end

rows["grade while"] = function(n)
  local a = 0
  local b = 0
  local i = 0
  while i < n do
    if i % 3 == 0 then
      a = a + 1
    elseif i % 3 == 1 then
      b = b + 1
    else
      a = a + 2
    end
    i = i + 1
  end
  return a + b
end

rows["while let vec"] = function(n)
  local v = {}
  for i = 0, n - 1 do
    v[i + 1] = i
  end
  local acc = 0
  for _, x in ipairs(v) do
    acc = acc + x
  end
  return acc
end

rows["for range"] = function(n)
  local acc = 0
  for i = 0, n - 1 do
    acc = acc + i
  end
  return acc
end

rows["for slice add"] = function(n)
  local v = {}
  for i = 0, n - 1 do
    v[i + 1] = i
  end
  local acc = 0
  for i = 1, #v do
    acc = acc + v[i] + 1
  end
  return acc
end

local row, n, reps = ...
n = tonumber(n) or N
reps = tonumber(reps) or REPS
local body = rows[row]
if body == nil then
  io.stderr:write("no row named " .. tostring(row) .. "\n")
  os.exit(1)
end

body(n)
local samples = {}
local value
for r = 1, reps do
  local start = os.clock()
  value = body(n)
  samples[r] = os.clock() - start
end
table.sort(samples)
local median = samples[math.floor(#samples / 2) + 1]
local shown
if row == "float while" then
  shown = string.format("%.1f", value)
else
  shown = string.format("%.0f", value)
end
print(string.format("%s\t%.1f\t%s", row, median * 1e6, shown))
