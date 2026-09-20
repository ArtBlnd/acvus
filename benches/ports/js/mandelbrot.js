// mandelbrot 200x100x200, ported from acvus-interpreter-test/benches/mandelbrot.rs.
//
// Obligation across artifacts: the three nested loops and the compound loop
// condition are that file's `rust_mandelbrot`, which is what the bench's
// `rust/us` column times.
//
// Usage: node mandelbrot.js <row> [reps]

const REPS = 4;

function mandelbrot(w, h, maxI) {
  let total = 0;
  let py = 0;
  while (py < h) {
    let px = 0;
    while (px < w) {
      const cx = -2.0 + (3.0 * px) / w;
      const cy = -1.2 + (2.4 * py) / h;
      let x = 0.0;
      let y = 0.0;
      let i = 0;
      while (i < maxI && x * x + y * y < 4.0) {
        const xt = x * x - y * y + cx;
        y = 2.0 * x * y + cy;
        x = xt;
        i = i + 1;
      }
      total = total + i;
      px = px + 1;
    }
    py = py + 1;
  }
  return total;
}

const row = process.argv[2];
const reps = process.argv[3] === undefined ? REPS : Number(process.argv[3]);
const [w, h, maxI] = row.split("x").map(Number);

mandelbrot(w, h, maxI);
const samples = [];
let value;
for (let r = 0; r < reps; r++) {
  const start = process.hrtime.bigint();
  value = mandelbrot(w, h, maxI);
  samples.push(Number(process.hrtime.bigint() - start));
}
samples.sort((a, b) => a - b);
const median = samples[samples.length >> 1];
console.log(`${row}\t${(median / 1000).toFixed(1)}\t${value}`);
