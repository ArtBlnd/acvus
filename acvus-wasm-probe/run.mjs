// Runs the probe module under node, each run on a fresh instance, and prints
// one JSON line.
//
// node run.mjs <module.wasm> <steps>...
//   The straight body, once per step count: per run, its marks and the
//   linear stack pointer around it, or the trap, and then whether dropping
//   the prepared body trapped.
// node run.mjs <module.wasm> recursion <stack budget in bytes | default> <frames>...
//   A recursion nesting that many frames, on a host given that stack budget
//   or none: per run, its value and its deepest call's mark, or the trap.
import { readFileSync } from 'node:fs';

const [path, ...rest] = process.argv.slice(2);
const module = new WebAssembly.Module(readFileSync(path));

Error.stackTraceLimit = Infinity;
const engineFrames = () => new Error().stack.split('\n').length - 1;

// Every import besides the probe's is linked by a dependency (`uuid`'s `js`
// feature) that the straight body never calls; each is linked as a throw.
const imports = {};
for (const { module: from, name } of WebAssembly.Module.imports(module)) {
  imports[from] ??= {};
  imports[from][name] = () => {
    throw new Error(`import ${from}.${name} reached`);
  };
}
imports.probe.engine_frames = engineFrames;

const panicMessage = (ex) =>
  new TextDecoder().decode(
    new Uint8Array(ex.memory.buffer, ex.panic_message_ptr(), ex.panic_message_len()),
  );

const recursions = (budget, frames) => {
  const runs = [];
  for (const deep of frames.map(Number)) {
    const ex = new WebAssembly.Instance(module, imports).exports;
    let value;
    try {
      value = budget === 'default' ? ex.recurse(deep) : ex.recurse_within(deep, Number(budget));
    } catch (error) {
      runs.push({ frames: deep, budget, trap: String(error), panic: panicMessage(ex) });
      continue;
    }
    const view = new DataView(ex.memory.buffer, ex.recursion_marks(), 16);
    runs.push({
      frames: deep,
      budget,
      value: String(value),
      deepest: {
        engine_frames: view.getUint32(8, true),
        linear_sp: view.getUint32(12, true),
      },
    });
  }
  return runs;
};

if (rest[0] === 'recursion') {
  const [, budget, ...frames] = rest;
  console.log(JSON.stringify(recursions(budget, frames)));
  process.exit(0);
}

const counts = rest;
const runs = [];
for (const steps of counts.map(Number)) {
  const ex = new WebAssembly.Instance(module, imports).exports;
  const spBefore = ex.linear_stack_pointer();
  let count;
  try {
    count = ex.run(steps);
  } catch (error) {
    runs.push({ steps, trap: String(error), panic: panicMessage(ex) });
    continue;
  }
  const spAfter = ex.linear_stack_pointer();
  const view = new DataView(ex.memory.buffer, ex.marks(), count * 16);
  const marks = [];
  for (let i = 0; i < count; i++) {
    marks.push({
      value: String(view.getBigInt64(i * 16, true)),
      engine_frames: view.getUint32(i * 16 + 8, true),
      linear_sp: view.getUint32(i * 16 + 12, true),
    });
  }
  let release = 'ok';
  try {
    ex.release();
  } catch (error) {
    release = `${error} ${panicMessage(ex)}`;
  }
  runs.push({ steps, marks, sp_before: spBefore, sp_after: spAfter, release });
}
console.log(JSON.stringify(runs));
