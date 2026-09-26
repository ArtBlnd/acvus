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
  no nightly is needed.
- `-C link-arg=-zstack-size=16777216` links the linear-memory stack at
  16 MiB instead of the linker's 1 MiB. The target links the stack first,
  so `__stack_pointer` starts at 16777216. An embedder may link a larger
  one.
- A `RUSTFLAGS` in the environment, an empty one included, replaces every
  `rustflags` a config file sets, so a build driven that way carries both
  flags in the variable.
- A tool that rewrites the module after it is linked must accept the
  tail-call proposal, since the module holds `return_call_indirect`.

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

- A straight body of arithmetic chains and one-word extern calls runs at
  constant engine depth and leaves the linear stack pointer where it found
  it; 20 000 steps were run under node 26.
- An operation that hands a callee the address of a slot on the linear
  stack does not tail-call its successor. On `wasm32` a result wider than
  one value comes back through such a slot, so this holds for many extern
  calls and for `For`, `Fused` and the direct and indirect body calls. A
  region that can escape, whose successor call and escape exit share one
  return, does not either. A long run of those still nests one engine
  frame per operation.
- Dropping a prepared body nests one engine frame per operation of a
  chain, so a body of about 16 000 operations or more runs node's default
  stack out when it is dropped.

## Checking a build

`cargo test -p acvus-interpreter-test --test wasm_probe`, part of
`cargo test --workspace`, builds `acvus-wasm-probe` for
`wasm32-unknown-unknown` with `--release` and reads the module; the
`wasm32-unknown-unknown` target must be installed. It fails when
`__stack_pointer` does not start at 16 MiB, and when an operation's `run`
calls its successor instead of tail-calling it, naming the operation.
Where `/usr/bin/node` exists, it runs a straight body of more than 20 000
operations, which must reach its last operation at the engine depth and
linear stack pointer of a body of one step; where it does not, it says so.
`RUSTFLAGS= cargo test -p acvus-interpreter-test --test wasm_probe` builds
the module without the flags, and the probe fails.

With the flags it fails today too, on the operations the section above
names; the straight body and the stack size pass.
