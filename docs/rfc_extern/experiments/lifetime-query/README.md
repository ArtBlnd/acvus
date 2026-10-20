# Probes: a compile-time query on lifetimes

The evidence for declarations §3, collecting lifetimes. `lq.rs` holds the query
and stand-ins for the extern crate's and an author's types; each probe poses
the queries a macro would generate, against types as rustc resolves them.

| Probe | What it checks | Expected |
|---|---|---|
| `q01_signature.rs` | a whole signature: a slice, an opaque type over a held reference, a kept function with its effect variable left as itself, two lifetimes and two type arguments, a structural type, a result | compiles |
| `q02_alias_swaps_lifetimes.rs` | an alias that reverses its lifetimes | lifetime error |
| `q03_alias_flips_type_args.rs` | an alias that reverses its type arguments | lifetime error |
| `q04_elided_in_path.rs` | a lifetime elided in a path, which the syntax does not show | E0277 |
| `q05_type_variable_unplaced.rs` | a type variable left as itself rather than put as `Var` | E0277 |
| `q06_closure_binder.rs` | a closure parameter's signature, its binder's lifetimes as the check's own | compiles |
| `q07_alias_in_closure_arg.rs` | a reversing alias inside a closure's argument | lifetime error |
| `q08a`, `q08b` | covariant markers at one position with `'x: 'y` implied, each direction | lifetime error, both |
| `q09_deep.rs` | sixteen lifetimes through nested two-lifetime types and reference chains | compiles |

Run `docs/rfc_extern/experiments/lifetime-query/run_probes.sh`; it prints each probe's
compile time too. Build outputs go to `target/` here, which git ignores.

Result, 2026-10-06, with stable rustc 1.97.1: all ten as expected, each in
twelve milliseconds or less.
