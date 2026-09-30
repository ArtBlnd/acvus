// RFC-0106 rule 2. Compiles a source of each kind of level, at half the
// bound, at the bound and one level past it, each on a fresh instance of the
// `nesting_wasm` module, and prints one JSON object: per compile its status,
// or the trap, and the linear stack it took; per kind the bytes a level took.
// The stack is painted below the stack pointer before each compile, and what
// the compile took is how far down the paint was overwritten.
// Usage: node nesting_wasm.mjs <nesting_wasm.wasm>
import { readFileSync } from 'node:fs';

const [path] = process.argv.slice(2);
const module = new WebAssembly.Module(readFileSync(path));

// Every import is linked by a dependency (`uuid`'s `js` feature) that no
// compile calls; each is linked as a throw.
const imports = {};
for (const { module: from, name } of WebAssembly.Module.imports(module)) {
  imports[from] ??= {};
  imports[from][name] = () => {
    throw new Error(`import ${from}.${name} reached`);
  };
}

const PAINT = 0xa5;
const STACK_LOW = 1024;
const STATUS = ['compiled', 'parse refused', 'check refused', 'no such kind'];
const NAMES = ['parens', 'chain', 'blocks', 'statements', 'lambdas', 'patterns', 'sections'];

const first = new WebAssembly.Instance(module, imports).exports;
const max = first.nesting_max();
const kinds = first.kind_count();
const half = Math.floor(max / 2);

const compile = (kind, levels) => {
  const ex = new WebAssembly.Instance(module, imports).exports;
  const top = ex.stack_pointer();
  new Uint8Array(ex.memory.buffer, STACK_LOW, top - STACK_LOW).fill(PAINT);
  let status;
  let trap;
  try {
    status = STATUS[ex.compile(kind, levels)];
  } catch (error) {
    trap = String(error);
  }
  const memory = new Uint8Array(ex.memory.buffer);
  let lowest = top;
  for (let at = STACK_LOW; at < top; at++) {
    if (memory[at] !== PAINT) {
      lowest = at;
      break;
    }
  }
  return { levels, status, trap, used: top - lowest, stack: top };
};

const report = { nesting_max: max, kinds: [] };
for (let kind = 0; kind < kinds; kind++) {
  const runs = [half, max, max + 1].map((levels) => compile(kind, levels));
  const [atHalf, atMax] = runs;
  report.kinds.push({
    kind: NAMES[kind],
    per_level: (atMax.used - atHalf.used) / (max - half),
    runs,
  });
}
console.log(JSON.stringify(report, null, 1));
