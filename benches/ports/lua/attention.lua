-- attention 256x128, ported from acvus-interpreter-test/benches/attention.rs.
--
-- Obligation across artifacts: `inputs` and the five loops are that file's
-- `inputs` and `rust_attention`, in the same order, and the timed region
-- excludes building the inputs exactly as the bench's does. `sin`, `cos` and
-- `exp` are the platform libm's and are not bit-reproducible across runtimes;
-- the bench's own criterion for this row is an absolute difference below 1e-9.
--
-- Usage: lua attention.lua <row> [reps]

local REPS = 4

local function inputs(n, d)
  local query = {}
  for i = 0, d - 1 do
    query[i + 1] = math.sin(i)
  end
  local keys = {}
  for t = 0, n - 1 do
    local row = {}
    for i = 0, d - 1 do
      row[i + 1] = math.cos(t + i)
    end
    keys[t + 1] = row
  end
  local values = {}
  for t = 0, n - 1 do
    local row = {}
    for i = 0, d - 1 do
      row[i + 1] = (t * d + i) / (n * d)
    end
    values[t + 1] = row
  end
  return query, keys, values
end

local function attention(query, keys, values)
  local d = #query
  local n = #keys
  local scale = 1.0 / math.sqrt(d)

  local scores = {}
  for t = 1, n do
    local s = 0.0
    for i = 1, d do
      s = s + query[i] * keys[t][i]
    end
    scores[t] = s * scale
  end

  local m = -math.huge
  for t = 1, n do
    if scores[t] > m then
      m = scores[t]
    end
  end

  local weights = {}
  for t = 1, n do
    weights[t] = math.exp(scores[t] - m)
  end

  local z = 0.0
  for t = 1, n do
    z = z + weights[t]
  end

  local out = {}
  for j = 1, d do
    local acc = 0.0
    for t = 1, n do
      acc = acc + weights[t] / z * values[t][j]
    end
    out[j] = acc
  end
  return out[1]
end

local row, reps = ...
reps = tonumber(reps) or REPS
local n, d = row:match("^(%d+)x(%d+)$")
n, d = tonumber(n), tonumber(d)
local query, keys, values = inputs(n, d)

attention(query, keys, values)
local samples = {}
local value
for r = 1, reps do
  local start = os.clock()
  value = attention(query, keys, values)
  samples[r] = os.clock() - start
end
table.sort(samples)
local median = samples[math.floor(#samples / 2) + 1]
print(string.format("%s\t%.1f\t%.17g", row, median * 1e6, value))
