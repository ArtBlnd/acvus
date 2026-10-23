import { readFileSync } from 'node:fs';

function refusing(module, name) {
  return () => {
    throw new Error(`import ${module}.${name} reached`);
  };
}

const module = new WebAssembly.Module(readFileSync(process.argv[2]));
const imports = {};
for (const imp of WebAssembly.Module.imports(module)) {
  imports[imp.module] ??= {};
  imports[imp.module][imp.name] = refusing(imp.module, imp.name);
}
const { main } = new WebAssembly.Instance(module, imports).exports;
const status = main(0, 0);
if (status !== 0) {
  throw new Error(`main returned ${status}`);
}
console.log('clockless_host: compiled and ran under node, 5050');
