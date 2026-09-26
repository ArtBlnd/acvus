# The nesting bound

A source nests at most `acvus_ast::NESTING_MAX` levels, and every parse
refuses a deeper one as `NestingTooDeep`, at the innermost construct that
holds more than the bound (RFC-0106 rule 1). The doc of `NESTING_MAX` states
what one level is. The parser counts each construct as it builds it, so no
tree past the bound is built, walked or dropped.

Past the parser, the compiler's walks recurse once per level. On a native
target each recursive entry, each parse and each pipeline stage runs through
`acvus_utils::grow`, which moves the work onto a fresh heap segment when the
thread's stack runs low (RFC-0106 rule 3). On `wasm32` there is no second
stack, and the bound alone keeps the compile inside the linear stack
(rule 2). This page records the measurement the bound was chosen from.

## Measuring on wasm32

```sh
RUSTFLAGS="-C link-arg=-zstack-size=16777216" \
  cargo build -p acvus-mir-test --example nesting_wasm \
  --target wasm32-unknown-unknown --release
node acvus-mir-test/examples/nesting_wasm.mjs \
  target/wasm32-unknown-unknown/release/examples/nesting_wasm.wasm
```

- The release profile has no LTO. `CARGO_PROFILE_RELEASE_OPT_LEVEL=z`
  measures the size-optimized build.
- The stack flag links the 16 MiB linear stack a `wasm32` build is linked
  with. A `RUSTFLAGS` in the environment replaces the `rustflags` of a
  config file, so the flag is given here in full.
- The example is not part of `cargo test --workspace`. The runner compiles a
  source of each kind of level at half the bound, at the bound and one level
  past it, each on a fresh instance. It paints the linear stack below the
  stack pointer, and reads what each compile took as how far down the paint
  was overwritten. The compile is the whole pipeline at `Opt::Full`.

## What was measured

Measured 2026-09-26 with `NESTING_MAX = 256`, node 26.7.0, the pinned
toolchain. Each source at 256 levels compiled without trapping, and each at
257 was refused by the parser. Stack is in bytes; per level is the slope
between 128 and 256 levels.

| Kind | Per level, opt 3 | At 256, opt 3 | Per level, opt z | At 256, opt z |
|---|---|---|---|---|
| parenthesized expressions | 2464 | 644 195 | 2080 | 536 227 |
| a chain of `+` | 2464 | 647 840 | 2080 | 536 227 |
| blocks | 2464 | 644 195 | 2080 | 536 227 |
| statements in blocks | 1824 | 481 635 | 1384 | 366 179 |
| lambdas, applied | 3536 | 915 179 | 2928 | 759 091 |
| patterns under a `match` | 2512 | 656 387 | 2080 | 536 227 |
| `% if` template sections | 1824 | 478 056 | 1384 | 364 627 |

The deepest compile at the bound takes 915 179 bytes, 5.5 % of the 16 MiB
stack. The bound is 256 for these reasons:

- It leaves a margin of 18 times over the deepest kind measured. The margin
  covers a construct that costs more per level than the seven kinds here,
  and the frames of whatever calls the compiler.
- The deepest source of `acvus-interpreter-test`'s `par_corpus` and
  `app_corpus`, all 180 files, nests 16 levels.
- A larger bound raises what a native build needs between two entries
  through `grow`. The AST is cloned and dropped by derived recursion, which
  `grow` cannot enter; a debug build overflowed a 256 KiB thread cloning a
  script about a hundred blocks deep before the pipeline stages entered
  through `grow`.

## On a native thread

`acvus-interpreter-test`'s `nesting_bound` compiles each kind at exactly the
bound on a thread of 256 KiB, at both optimization levels, and runs it to its
value on a thread of 2 MiB. The run is not on the small thread: a debug
build's machine keeps more headroom below a call than 256 KiB holds
(RFC-0100 rule 5), so its first closure call traps there.

`grow`'s red zone is 1 MiB, and a fresh segment is 4 MiB. The largest stretch
between two entries through `grow` is the parser's `__reduce`, one frame of
668 KiB in a debug build.

## What the bound does not bound

These recursions follow the length of a source, not its nesting, so neither
the bound nor the measurement above covers them on `wasm32`:

- A type's depth. `let a1 = Some(a0); let a2 = Some(a1);` builds a deeper
  `Option` with each statement at the same nesting. The walks over types run
  through `grow`, so a native thread holds them; the derived `Clone`, `Hash`
  and `PartialEq` of a type do not.
- The optimizer's walks along a control-flow path or a chain of SSA
  definitions, and union-find root chasing in the solver.
