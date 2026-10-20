# Probes: closure parameters as Rust's own closure types

The evidence for functions §1, closure parameters, and functions §2, a
function kept past the call. The probes compile
author signatures that take Rust's own closure types, and functions kept past
the call, against the model of `../reference-markers`.

| Probe | What it checks | Expected |
|---|---|---|
| `c01_rust_closure_params.rs` | `impl Fn(&T) -> bool`, a result borrowing the call's argument by elision, two arguments with lifetimes of their own, `impl FnMut(&mut T)`, all generic over `T: TypeVar` and the declaration's effect variable `E: EffectVar`, and an entry that writes the closures it hands over | compiles |
| `c02_lent_closure_is_not_kept.rs` | a closure lent for the call, boxed to be kept past it | E0310 |
| `c03_result_names_its_argument.rs` | a result that borrows an argument its signature does not name | E0621 |
| `c04_kept_function_projection.rs` | what `#[fn_type]` would expand to, by hand: a marker carrying the function's effect variable and a projection trait over `Uniform` of it; a function kept in an author's type and called later through the projection, generic and at a concrete instantiation, with arguments of their own lifetimes | compiles |
| `c05_projection_names_its_argument.rs` | a projection whose result borrows an argument its signature does not name | E0621 |
| `c06_inherent_impl_on_uniform.rs` | the projection as an inherent method on `Uniform`, written in the author's crate | E0116 |

Run `docs/rfc_extern/experiments/closure-params/run_probes.sh`. Build outputs go to
`target/` here, which git ignores.

Result, 2026-10-06, with stable rustc 1.97.1: all six as expected.
