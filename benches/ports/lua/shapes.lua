-- shapes rows at n = 1 000 000, ported from acvus-interpreter-test/benches/shapes.rs.
--
-- Obligation across artifacts: each row is the semantics of that file's `rust_*`
-- twin. Lua has one aggregate, so the twin's `struct Point` is a table with the
-- same two named fields and the twin's `enum E` is a table `{ t = tag, v = payload }`
-- with an integer tag standing for the discriminant; Lua carries no tagged
-- union of its own.
--
-- Usage: lua shapes.lua <row> [n] [reps]

local REPS = 4
local N = 1000000

local A = 0
local B = 1

local function some_of(i)
  if i % 2 == 0 then
    return i
  end
  return nil
end

local rows = {}

rows["field read"] = function(n)
  local p = { x = 1, y = 2 }
  local acc = 0
  local i = 0
  while i < n do
    acc = acc + p.x + p.y
    i = i + 1
  end
  return acc
end

rows["field write"] = function(n)
  local p = { x = 0, y = 0 }
  local i = 0
  while i < n do
    p.x = p.x + i
    i = i + 1
  end
  return p.x
end

rows["construct"] = function(n)
  local acc = 0
  local i = 0
  while i < n do
    local q = { x = i, y = i + 1 }
    acc = acc + q.x
    i = i + 1
  end
  return acc
end

rows["enum match"] = function(n)
  local acc = 0
  local i = 0
  while i < n do
    local e
    if i % 2 == 0 then
      e = { t = A, v = i }
    else
      e = { t = B, v = i + 1 }
    end
    if e.t == A then
      acc = acc + e.v
    else
      acc = acc + e.v
    end
    i = i + 1
  end
  return acc
end

rows["option match"] = function(n)
  local acc = 0
  local i = 0
  while i < n do
    local v = some_of(i)
    if v ~= nil then
      acc = acc + v
    end
    i = i + 1
  end
  return acc
end

rows["vec of objects"] = function(n)
  local v = {}
  for k = 0, 999 do
    v[k + 1] = { x = k, y = k + 1 }
  end
  local m = #v
  local acc = 0
  local r = 0
  local outer = math.floor(n / 1000)
  while r < outer do
    local i = 1
    while i <= m do
      acc = acc + v[i].x
      i = i + 1
    end
    r = r + 1
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
print(string.format("%s\t%.1f\t%.0f", row, median * 1e6, value))
