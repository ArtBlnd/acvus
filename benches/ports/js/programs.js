// programs rows at n = 1 000 000, ported from acvus-interpreter-test/benches/programs.rs.
//
// Obligation across artifacts: `B`, `STEPS_PER_OUTER`, `TAPE`, the opcode
// numbers, `program`, `bracketTable` and the three machine bodies are that
// file's `rust_table`, `rust_scan` and `rust_call`, whose timed region includes
// building the program, decoding it and building the bracket table. The twin
// dispatches an eight-variant `enum` through `match`; the port dispatches the
// integer opcode through a dense `switch`, arms in the twin's order.
//
// Usage: node programs.js <row> [n] [reps]

const REPS = 3;
const N = 1000000;

const B = 32;
const STEPS_PER_OUTER = 6 * B + 6;
const MIN_OUTER = 1;
const TAPE = 30000;

const INC = 0;
const DEC = 1;
const LEFT = 2;
const RIGHT = 3;
const OPEN = 4;
const CLOSE = 5;
const OUT = 6;
const IN = 7;

function outerCount(n) {
  return Math.max(Math.floor(n / STEPS_PER_OUTER), MIN_OUTER);
}

function program(n) {
  const code = [];
  for (let k = 0; k < outerCount(n); k++) code.push(INC);
  code.push(OPEN);
  code.push(RIGHT);
  for (let k = 0; k < B; k++) code.push(INC);
  code.push(OPEN, DEC, RIGHT, INC, LEFT, CLOSE);
  code.push(LEFT, DEC, CLOSE);
  code.push(RIGHT, RIGHT, OUT);
  return code;
}

function decode(code) {
  return code.map((c) => (c <= OUT ? c : IN));
}

function bracketTable(code) {
  const jumps = new Array(code.length).fill(0);
  const stack = [];
  for (let i = 0; i < code.length; i++) {
    if (code[i] === OPEN) stack.push(i);
    if (code[i] === CLOSE && stack.length > 0) {
      const o = stack.pop();
      jumps[o] = i;
      jumps[i] = o;
    }
  }
  return jumps;
}

function bfTable(n) {
  const code = program(n);
  const prog = decode(code);
  const jumps = bracketTable(code);
  const tape = new Array(TAPE).fill(0);
  let pc = 0;
  let ptr = 0;
  let steps = 0;
  let out = 0;
  const plen = prog.length;
  while (pc < plen) {
    switch (prog[pc]) {
      case INC:
        tape[ptr] += 1;
        break;
      case DEC:
        tape[ptr] -= 1;
        break;
      case LEFT:
        ptr -= 1;
        break;
      case RIGHT:
        ptr += 1;
        break;
      case OPEN:
        if (tape[ptr] === 0) pc = jumps[pc];
        break;
      case CLOSE:
        if (tape[ptr] !== 0) pc = jumps[pc];
        break;
      case OUT:
        out += tape[ptr];
        break;
      default:
        break;
    }
    pc += 1;
    steps += 1;
  }
  return steps + out;
}

function bfScan(n) {
  const code = program(n);
  const prog = decode(code);
  const tape = new Array(TAPE).fill(0);
  let pc = 0;
  let ptr = 0;
  let steps = 0;
  let out = 0;
  const plen = prog.length;
  while (pc < plen) {
    switch (prog[pc]) {
      case INC:
        tape[ptr] += 1;
        break;
      case DEC:
        tape[ptr] -= 1;
        break;
      case LEFT:
        ptr -= 1;
        break;
      case RIGHT:
        ptr += 1;
        break;
      case OPEN:
        if (tape[ptr] === 0) {
          let d = 1;
          while (d > 0) {
            pc += 1;
            if (prog[pc] === OPEN) d += 1;
            else if (prog[pc] === CLOSE) d -= 1;
          }
        }
        break;
      case CLOSE:
        if (tape[ptr] !== 0) {
          let d = 1;
          while (d > 0) {
            pc -= 1;
            if (prog[pc] === OPEN) d -= 1;
            else if (prog[pc] === CLOSE) d += 1;
          }
        }
        break;
      case OUT:
        out += tape[ptr];
        break;
      default:
        break;
    }
    pc += 1;
    steps += 1;
  }
  return steps + out;
}

function bump(x, d) {
  return x + d;
}

function bfCall(n) {
  const code = program(n);
  const prog = decode(code);
  const jumps = bracketTable(code);
  const tape = new Array(TAPE).fill(0);
  let pc = 0;
  let ptr = 0;
  let steps = 0;
  let out = 0;
  const plen = prog.length;
  while (pc < plen) {
    switch (prog[pc]) {
      case INC:
        tape[ptr] = bump(tape[ptr], 1);
        break;
      case DEC:
        tape[ptr] = bump(tape[ptr], -1);
        break;
      case LEFT:
        ptr -= 1;
        break;
      case RIGHT:
        ptr += 1;
        break;
      case OPEN:
        if (tape[ptr] === 0) pc = jumps[pc];
        break;
      case CLOSE:
        if (tape[ptr] !== 0) pc = jumps[pc];
        break;
      case OUT:
        out += tape[ptr];
        break;
      default:
        break;
    }
    pc += 1;
    steps += 1;
  }
  return steps + out;
}

const ROWS = { "bf table": bfTable, "bf scan": bfScan, "bf call": bfCall };

const row = process.argv[2];
const n = process.argv[3] === undefined ? N : Number(process.argv[3]);
const reps = process.argv[4] === undefined ? REPS : Number(process.argv[4]);
const body = ROWS[row];
if (body === undefined) {
  console.error(`no row named ${JSON.stringify(row)}`);
  process.exit(1);
}

body(n);
const samples = [];
let value;
for (let r = 0; r < reps; r++) {
  const start = process.hrtime.bigint();
  value = body(n);
  samples.push(Number(process.hrtime.bigint() - start));
}
samples.sort((a, b) => a - b);
const median = samples[samples.length >> 1];
console.log(`${row}\t${(median / 1000).toFixed(1)}\t${value}`);
