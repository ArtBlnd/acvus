# Probes: an extern-defined type's parameters, named and bounded

The evidence for types §1: an extern-defined type whose attribute names every
parameter, each bounded by its role, so that fields holding an `Instance` and
a kept function are well formed and nothing of the type is read. `model.rs`
holds stand-ins for the runtime, `Uniform`, the sealed `TypeVar`, the effect
and identity markers, a kept function's type and an `Instance`; they are the
probe's means, not decided primitives. Each probe writes by hand what the
attribute would write.

| Probe | What it checks | Expected |
|---|---|---|
| `p01_named_with_bounds.rs` | each named parameter bounded by its role: fields holding an `Instance` and a kept function, each carrying the type's effect variable | compiles |
| `p02_named_without_bounds.rs` | the same fields with no bound | E0277 |
| `p03_arguments_without_runtime.rs` | a type argument bounded with no runtime parameter to name | E0425 |
| `p04_instantiations_share_layout.rs` | a generic and a concrete instantiation of one type: same size and alignment | compiles |
| `p05_runtime_only_in_bounds.rs` | a runtime that only the type arguments' bounds name | E0392 |
| `p06_native_argument.rs` | the type written with a native argument | E0277 |

Run `docs/rfc_extern/experiments/opaque-attribute/run_probes.sh`. Build
outputs go to `target/` here, which git ignores.

Result, 2026-10-06, with stable rustc 1.97.1: all six as expected.
- p02 is refused at the fields: `R: Runtime`, `It: TypeVar<R>` and `E: EffectVar` are not satisfied.
- p03 is "cannot find type `R` in this scope".
- p05 is "type parameter `R` is never used": a runtime the fields do not name must still be used, by a marker field or by the fields.
- p06 is "the trait bound `i64: TypeVar<Probe>` is not satisfied": a type whose arguments are bounded by `TypeVar` has no instantiation at a native argument.
