-- programs rows at n = 1 000 000, ported from acvus-interpreter-test/benches/programs.rs.
--
-- Obligation across artifacts: `B`, `STEPS_PER_OUTER`, `TAPE`, the opcode
-- numbers, `program`, `bracket_table` and the three machine bodies are that
-- file's `rust_table`, `rust_scan` and `rust_call`, whose timed region includes
-- building the program, decoding it and building the bracket table.
--
-- Two decisions not to build: Lua tables index from 1, so `prog`, `jumps` and
-- `tape` are 1-based and `pc`/`ptr` start at 1, which shifts every index by one
-- and changes no step the machine takes; and Lua has no `switch`, so the twin's
-- eight-variant `match` is an `if`/`elseif` chain in the twin's arm order.
--
-- Usage: lua programs.lua <row> [n] [reps]

local REPS = 3
local N = 1000000

local B = 32
local STEPS_PER_OUTER = 6 * B + 6
local MIN_OUTER = 1
local TAPE = 30000

local INC = 0
local DEC = 1
local LEFT = 2
local RIGHT = 3
local OPEN = 4
local CLOSE = 5
local OUT = 6
local IN = 7

local function outer_count(n)
  local a = math.floor(n / STEPS_PER_OUTER)
  if a < MIN_OUTER then
    return MIN_OUTER
  end
  return a
end

local function program(n)
  local code = {}
  local at = 0
  local function push(c)
    at = at + 1
    code[at] = c
  end
  for _ = 1, outer_count(n) do
    push(INC)
  end
  push(OPEN)
  push(RIGHT)
  for _ = 1, B do
    push(INC)
  end
  for _, c in ipairs({ OPEN, DEC, RIGHT, INC, LEFT, CLOSE }) do
    push(c)
  end
  for _, c in ipairs({ LEFT, DEC, CLOSE }) do
    push(c)
  end
  for _, c in ipairs({ RIGHT, RIGHT, OUT }) do
    push(c)
  end
  return code
end

local function decode(code)
  local prog = {}
  for i = 1, #code do
    if code[i] <= OUT then
      prog[i] = code[i]
    else
      prog[i] = IN
    end
  end
  return prog
end

local function bracket_table(code)
  local jumps = {}
  for i = 1, #code do
    jumps[i] = 0
  end
  local stack = {}
  local top = 0
  for i = 1, #code do
    if code[i] == OPEN then
      top = top + 1
      stack[top] = i
    end
    if code[i] == CLOSE and top > 0 then
      local o = stack[top]
      top = top - 1
      jumps[o] = i
      jumps[i] = o
    end
  end
  return jumps
end

local function blank_tape()
  local tape = {}
  for i = 1, TAPE do
    tape[i] = 0
  end
  return tape
end

local rows = {}

rows["bf table"] = function(n)
  local code = program(n)
  local prog = decode(code)
  local jumps = bracket_table(code)
  local tape = blank_tape()
  local pc = 1
  local ptr = 1
  local steps = 0
  local out = 0
  local plen = #prog
  while pc <= plen do
    local op = prog[pc]
    if op == INC then
      tape[ptr] = tape[ptr] + 1
    elseif op == DEC then
      tape[ptr] = tape[ptr] - 1
    elseif op == LEFT then
      ptr = ptr - 1
    elseif op == RIGHT then
      ptr = ptr + 1
    elseif op == OPEN then
      if tape[ptr] == 0 then
        pc = jumps[pc]
      end
    elseif op == CLOSE then
      if tape[ptr] ~= 0 then
        pc = jumps[pc]
      end
    elseif op == OUT then
      out = out + tape[ptr]
    end
    pc = pc + 1
    steps = steps + 1
  end
  return steps + out
end

rows["bf scan"] = function(n)
  local code = program(n)
  local prog = decode(code)
  local tape = blank_tape()
  local pc = 1
  local ptr = 1
  local steps = 0
  local out = 0
  local plen = #prog
  while pc <= plen do
    local op = prog[pc]
    if op == INC then
      tape[ptr] = tape[ptr] + 1
    elseif op == DEC then
      tape[ptr] = tape[ptr] - 1
    elseif op == LEFT then
      ptr = ptr - 1
    elseif op == RIGHT then
      ptr = ptr + 1
    elseif op == OPEN then
      if tape[ptr] == 0 then
        local d = 1
        while d > 0 do
          pc = pc + 1
          if prog[pc] == OPEN then
            d = d + 1
          elseif prog[pc] == CLOSE then
            d = d - 1
          end
        end
      end
    elseif op == CLOSE then
      if tape[ptr] ~= 0 then
        local d = 1
        while d > 0 do
          pc = pc - 1
          if prog[pc] == OPEN then
            d = d - 1
          elseif prog[pc] == CLOSE then
            d = d + 1
          end
        end
      end
    elseif op == OUT then
      out = out + tape[ptr]
    end
    pc = pc + 1
    steps = steps + 1
  end
  return steps + out
end

local function bump(x, d)
  return x + d
end

rows["bf call"] = function(n)
  local code = program(n)
  local prog = decode(code)
  local jumps = bracket_table(code)
  local tape = blank_tape()
  local pc = 1
  local ptr = 1
  local steps = 0
  local out = 0
  local plen = #prog
  while pc <= plen do
    local op = prog[pc]
    if op == INC then
      tape[ptr] = bump(tape[ptr], 1)
    elseif op == DEC then
      tape[ptr] = bump(tape[ptr], -1)
    elseif op == LEFT then
      ptr = ptr - 1
    elseif op == RIGHT then
      ptr = ptr + 1
    elseif op == OPEN then
      if tape[ptr] == 0 then
        pc = jumps[pc]
      end
    elseif op == CLOSE then
      if tape[ptr] ~= 0 then
        pc = jumps[pc]
      end
    elseif op == OUT then
      out = out + tape[ptr]
    end
    pc = pc + 1
    steps = steps + 1
  end
  return steps + out
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
