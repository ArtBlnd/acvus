// shapes rows at n = 1 000 000, ported from acvus-interpreter-test/benches/shapes.rs.
//
// Obligation across artifacts: each row is the semantics of that file's `rust_*`
// twin. The twin's record is a `struct` with named fields, which in JS is an
// object literal of the same shape; the twin's `enum E` is a tagged value,
// which is `{ t, v }` with an integer tag standing for the discriminant.
//
// Usage: node shapes.js <row> [n] [reps]

const REPS = 4;
const N = 1000000;

const A = 0;
const B = 1;

function someOf(i) {
  return i % 2 === 0 ? i : null;
}

function fieldRead(n) {
  const p = { x: 1, y: 2 };
  let acc = 0;
  let i = 0;
  while (i < n) {
    acc = acc + p.x + p.y;
    i = i + 1;
  }
  return acc;
}

function fieldWrite(n) {
  const p = { x: 0, y: 0 };
  let i = 0;
  while (i < n) {
    p.x = p.x + i;
    i = i + 1;
  }
  return p.x;
}

function construct(n) {
  let acc = 0;
  let i = 0;
  while (i < n) {
    const q = { x: i, y: i + 1 };
    acc = acc + q.x;
    i = i + 1;
  }
  return acc;
}

function enumMatch(n) {
  let acc = 0;
  let i = 0;
  while (i < n) {
    const e = i % 2 === 0 ? { t: A, v: i } : { t: B, v: i + 1 };
    if (e.t === A) acc = acc + e.v;
    else acc = acc + e.v;
    i = i + 1;
  }
  return acc;
}

function optionMatch(n) {
  let acc = 0;
  let i = 0;
  while (i < n) {
    const v = someOf(i);
    if (v !== null) acc = acc + v;
    i = i + 1;
  }
  return acc;
}

function vecOfObjects(n) {
  const v = [];
  for (let k = 0; k < 1000; k++) v.push({ x: k, y: k + 1 });
  const m = v.length;
  let acc = 0;
  let r = 0;
  while (r < Math.floor(n / 1000)) {
    let i = 0;
    while (i < m) {
      acc = acc + v[i].x;
      i = i + 1;
    }
    r = r + 1;
  }
  return acc;
}

const ROWS = {
  "field read": fieldRead,
  "field write": fieldWrite,
  construct: construct,
  "enum match": enumMatch,
  "option match": optionMatch,
  "vec of objects": vecOfObjects,
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
console.log(`${row}\t${(median / 1000).toFixed(1)}\t${value}`);
