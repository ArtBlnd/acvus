# Building for wasm32

A chain's operations tail-call their successor (RFC-0052 rule 1). On
`wasm32-unknown-unknown` that call is `return_call_indirect`, which exists
only with the target's `tail-call` feature. A module built without it nests
one engine frame per operation, on the engine's own stack, which a page
cannot size. RFC-0105 is the decision.

## The flags

The workspace's `.cargo/config.toml` sets both flags for its own builds.
Cargo reads that file only for builds inside this workspace, so a crate
that depends on these crates and builds its own module sets the same two
flags in its own `.cargo/config.toml`:

```toml
[target.wasm32-unknown-unknown]
rustflags = [
    "-C", "target-feature=+tail-call",
    "-C", "link-arg=-zstack-size=16777216",
]
```

- `-C target-feature=+tail-call` compiles a chain's tail call as
  `return_call_indirect`. The pinned stable toolchain (1.97.1) accepts it;
  no nightly is needed. A `wasm32` build of `acvus-interpreter` without it
  is refused by a `compile_error!` that names the flag.
- `-C link-arg=-zstack-size=16777216` links the linear-memory stack at
  16 MiB instead of the linker's 1 MiB. The target links the stack first,
  so `__stack_pointer` starts at 16777216. An embedder may link a larger
  one.
- A `RUSTFLAGS` in the environment, an empty one included, replaces every
  `rustflags` a config file sets, so a build driven that way carries both
  flags in the variable.
- A tool that rewrites the module after it is linked must accept the
  tail-call proposal, since the module holds `return_call_indirect`.
- No other target feature is needed. `multivalue` is on by default in
  this toolchain, and turning it off changes nothing the probe reads: it
  lets a function return several values, but the wasm32 ABI still returns
  a value wider than one word through a slot in the caller's frame, and
  the module holds no function type with two results either way.

## Engines

An engine without Wasm tail calls cannot compile the module and is
unsupported. The module is not built without the feature for such an
engine.

The first version that ships Wasm tail calls, per MDN's
browser-compat-data (`webassembly/tail-calls.json`) and the WebAssembly
feature table (`features.json` in `WebAssembly/website`), both read on
2026-09-26:

- Chrome 112; Edge mirrors Chrome in the compat data;
- Firefox 121;
- Safari 18.2; Safari on iOS mirrors it in the compat data;
- Node.js 20.0, in the feature table.

## What a module does today

- A straight body of more than 40 000 operations runs at constant engine
  depth and leaves the linear stack pointer where it found it, under
  node 26.
- Every operation's `run` ends in `return_call_indirect` to its
  successor, or ends a chain, or is of a family that holds the address of
  a slot in its linear-stack frame across a callee. LLVM does not
  tail-call out of a function whose frame a callee may still reach. Those
  families are extern calls (the handler takes its argument run and result
  slot by reference), body and closure calls, `Fused`, spawns,
  `MakeClosure`, storage `Fetch` and `Commit`, the path writes `SetStep`
  and `SetPath`, and `Concat`.
  `wasm_probe` lists each with the address and the callee it reaches. A
  long run of those still nests one engine frame per operation.
- A region whose part can escape (a `break`, `continue`, `?` or `return`
  inside it) ends in one call, of its successor or of `Yield` with the
  verdict, so it tail-calls on both paths.
- Dropping a prepared body is a loop over its operations, so a body that
  runs also drops.

## The depth bound

A module cannot read how much of its engine's stack is left, so a call
chain is bounded by a count of frames instead (RFC-0100 rule 5). The host
computes the bound when it compiles, from the stack budget its embedder
gives it:

```rust
let program = Host::new(registries)
    .stack_budget(4 << 20)
    .entry::<(), i64>("main", Source::Script(script))
    .compile(SequentialExecutor)?;
```

- The budget is the engine's stack in bytes, which the embedder sizes:
  node takes it from `--stack-size` (in KiB), a browser from its own
  setting. `Host::stack_budget` exists only on `wasm32`; a native host
  reads each thread's stack instead.
- The bound is the budget less a headroom of 128 KiB, over the most stack
  a counted frame was measured to cost on the build (2464 bytes for the
  `wasm` profile), charged half again. A budget no larger than the headroom
  bounds every chain at zero frames, so every call traps.
- A host given no budget assumes the stack the build's figures were
  measured on: V8's default 984 KiB for the `wasm` profile, which bounds a
  chain at 237 frames.
- A chain past the bound ends in the depth trap, whose message names the
  frames and the bound. Under a budget larger than the stack the engine
  gives, a chain can run that stack out first, which ends in the engine's
  `RangeError` instead.
- The linear stack is not part of the budget. On the `wasm` profile a
  level of `wasm_probe`'s recursion spends 256 bytes of it, so the 16 MiB
  stack holds that recursion at the bound of any budget under 200 MiB. What
  a level of the costliest path spends of it has not been measured.
- A debug build's figures were measured on the linear stack when it was
  linked at 1 MiB, and a debug host given no budget still assumes 1 MiB,
  bounding a chain at 73 frames. Linked at 16 MiB, a debug module runs the
  engine's stack out first: `wasm_probe`'s recursion reaches V8's default
  limit between 400 and 800 levels while the linear stack has spent under
  0.5 MiB. How much of the engine's stack a debug frame costs on the
  costliest path has not been measured.

## Checking a build

`cargo bench -p acvus-interpreter-test --bench wasm_probe` builds
`acvus-wasm-probe` for `wasm32-unknown-unknown` with the `wasm` profile,
the one an embedder deploys (fat LTO, `opt-level = "z"`), in a cargo of
its own, and reads the module. The `wasm32-unknown-unknown` target must
be installed. It is a bench so that `cargo test` never builds a fat-LTO
module. It fails:

- when `__stack_pointer` does not start at 16 MiB;
- when an operation's `run` calls its successor and is neither a chain's
  end nor an instance, holding a linear-stack frame, of a listed family.

It prints the counts as tail calls, chain ends and listed instances.
Where `/usr/bin/node` exists, it runs a straight body of more than 40 000
operations, which must reach its last operation at the engine depth and
linear stack pointer of a body of one step, and then drops it. It then
runs a recursion through `Host`: on a host given no budget, one frame
below the bound returns and one past it ends in the depth trap; on a host
given 512 KiB, the same recursion one frame below the default bound traps;
on a host given 4 MiB under `node --stack-size=4096`, a recursion past the
default bound returns, and one past the larger bound traps. Where node does
not exist, it says so.
