// attention 256x128, ported from acvus-interpreter-test/benches/attention.rs.
//
// Obligation across artifacts: `inputs` and the five loops are that file's
// `inputs` and `rust_attention`, in the same order, and the timed region
// excludes building the inputs exactly as the bench's does. V8 carries its own
// `sin`, `cos` and `exp`, so this row's value is not expected to be bit-equal
// to the Rust twin's; the bench's own criterion is an absolute difference
// below 1e-9.
//
// Usage: node attention.js <row> [reps]

const REPS = 4;

function inputs(n, d) {
  const query = [];
  for (let i = 0; i < d; i++) query.push(Math.sin(i));
  const keys = [];
  for (let t = 0; t < n; t++) {
    const row = [];
    for (let i = 0; i < d; i++) row.push(Math.cos(t + i));
    keys.push(row);
  }
  const values = [];
  for (let t = 0; t < n; t++) {
    const row = [];
    for (let i = 0; i < d; i++) row.push((t * d + i) / (n * d));
    values.push(row);
  }
  return { query, keys, values };
}

function attention({ query, keys, values }) {
  const d = query.length;
  const n = keys.length;
  const scale = 1.0 / Math.sqrt(d);

  const scores = [];
  for (let t = 0; t < n; t++) {
    let s = 0.0;
    for (let i = 0; i < d; i++) s = s + query[i] * keys[t][i];
    scores.push(s * scale);
  }

  let m = -Infinity;
  for (const s of scores) if (s > m) m = s;

  const weights = [];
  for (const s of scores) weights.push(Math.exp(s - m));

  let z = 0.0;
  for (const w of weights) z = z + w;

  const out = [];
  for (let j = 0; j < d; j++) {
    let acc = 0.0;
    for (let t = 0; t < n; t++) acc = acc + (weights[t] / z) * values[t][j];
    out.push(acc);
  }
  return out[0];
}

const row = process.argv[2];
const reps = process.argv[3] === undefined ? REPS : Number(process.argv[3]);
const [n, d] = row.split("x").map(Number);
const built = inputs(n, d);

attention(built);
const samples = [];
let value;
for (let r = 0; r < reps; r++) {
  const start = process.hrtime.bigint();
  value = attention(built);
  samples.push(Number(process.hrtime.bigint() - start));
}
samples.sort((a, b) => a - b);
const median = samples[samples.length >> 1];
console.log(`${row}\t${(median / 1000).toFixed(1)}\t${value.toPrecision(17)}`);
