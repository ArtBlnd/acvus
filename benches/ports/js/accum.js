// accum rows at n = 1 000 000, ported from acvus-interpreter-test/benches/accum.rs.
//
// Obligation across artifacts: each row here is the semantics of that file's
// `rust_*` twin, which is what the bench's `rust/us` column times. Every value
// the rows carry is below 2^53, so a JS double holds each of them exactly;
// `black_box` has no JS spelling, so where V8 hoists work the twin is stopped
// from hoisting, the row says so in docs/performance.md.
//
// Usage: node accum.js <row> [n] [reps]

const REPS = 4;
const N = 1000000;

function idOf(i) {
  return i;
}

function evenOf(i) {
  return i % 2 === 0;
}

function step(x) {
  return x + 1;
}

function* range(n) {
  for (let i = 0; i < n; i++) yield i;
}

function intWhile(n) {
  let acc = 0;
  let i = 0;
  while (i < n) {
    acc = acc + i;
    i = i + 1;
  }
  return acc;
}

function floatWhile(n) {
  let acc = 0.0;
  let i = 0;
  while (i < n) {
    acc = acc + i;
    i = i + 1;
  }
  return acc;
}

function rangeSum(n) {
  return range(n).reduce((a, b) => a + b, 0);
}

function mapIdSum(n) {
  return range(n)
    .map((x) => x)
    .reduce((a, b) => a + b, 0);
}

function mapAddSum(n) {
  return range(n)
    .map((x) => x + 1)
    .reduce((a, b) => a + b, 0);
}

function branchWhile(n) {
  let acc = 0;
  let i = 0;
  while (i < n) {
    if (evenOf(i)) acc = acc + i;
    i = i + 1;
  }
  return acc;
}

function callWhile(n) {
  let i = 0;
  while (i < n) i = step(i);
  return i;
}

function collatzWhile(n) {
  let acc = 0;
  let i = 0;
  while (i < n) {
    const d = i % 2 === 0 ? i / 2 : i * 3 + 1;
    acc = acc + d;
    i = i + 1;
  }
  return acc;
}

function gradeWhile(n) {
  let a = 0;
  let b = 0;
  let i = 0;
  while (i < n) {
    if (i % 3 === 0) a = a + 1;
    else if (i % 3 === 1) b = b + 1;
    else a = a + 2;
    i = i + 1;
  }
  return a + b;
}

function whileLetVec(n) {
  const v = [];
  for (let i = 0; i < n; i++) v.push(i);
  const it = v[Symbol.iterator]();
  let acc = 0;
  let next = it.next();
  while (!next.done) {
    acc = acc + next.value;
    next = it.next();
  }
  return acc;
}

function forRange(n) {
  let acc = 0;
  for (let i = 0; i < n; i++) acc = acc + i;
  return acc;
}

function forSliceAdd(n) {
  const v = [];
  for (let i = 0; i < n; i++) v.push(i);
  let acc = 0;
  for (const x of v) acc = acc + x + 1;
  return acc;
}

const ROWS = {
  "int while": intWhile,
  "float while": floatWhile,
  "range | sum": rangeSum,
  "map id | sum": mapIdSum,
  "map add | sum": mapAddSum,
  "branch while": branchWhile,
  "call while": callWhile,
  "collatz while": collatzWhile,
  "grade while": gradeWhile,
  "while let vec": whileLetVec,
  "for range": forRange,
  "for slice add": forSliceAdd,
};

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
const shown = row === "float while" ? value.toFixed(1) : String(value);
console.log(`${row}\t${(median / 1000).toFixed(1)}\t${shown}`);
