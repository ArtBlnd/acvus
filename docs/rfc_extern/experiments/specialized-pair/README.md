# Probes: a `#` member checked against its uniform member

The evidence for types §3: whether Rust can confirm that a declaration in the
whole native form is its uniform member at the marked type, the choice its
`specializes = f<..>` writes. `pair.rs` holds the instrument, a trait giving
each type its native form, what a `#` member writes where its uniform member
writes the type, and stand-ins for the extern crate's and an author's types;
the trait is the probe's means, not a decided primitive. A `Uniform` is the
value it holds, written natively; a `Ref` has no native form, since a `#`
value holds no `Value` to lend.

Each probe writes what the two macros would emit: the uniform member's family
over its lifetimes and type variables, and the `#` member's check, which puts
its choice into the family and asks rustc, through trait matching, whether
the family's native form is its own signature.

Two ways of writing the family were tried.

**s: a higher-ranked function pointer**, `for<'l0> fn(<P as Native>::Out, ..) -> <R as Native>::Out`.
It fails. A lifetime that appears in the inputs only inside a projection does
not constrain the result, so a result that borrows a parameter is refused
(E0581, s09), and so is a probe meant to fail for another reason (s11). A
split or merged lifetime is refused, with "not general enough" rather than a
plain mismatch (s07, s08).

**t: a tuple with early-bound lifetimes**, `<((P, ..), R) as Native>::Out`,
generic over its lifetimes in the order they first appear in the signature,
the `#` member passing its own lifetimes in the same order. It works: every
probe as expected.

| Probe | What it checks | Expected |
|---|---|---|
| `t01_generic_pair.rs` | `push<T>` and `push_f64` at `T = f64` | compiles |
| `t02_generic_pair_wrong_param.rs` | a parameter that is not the uniform member's at the choice | E0277 |
| `t03_generic_pair_wrong_choice.rs` | a choice other than what the `#` member writes | E0277 |
| `t04_concrete_uniform_pair.rs` | `sum(&Stack<Uniform<f64>>)` and `sum_native(&Stack<f64>)` | compiles |
| `t05_concrete_uniform_pair_wrong.rs` | another element type | E0277 |
| `t06_two_lifetimes.rs` | two lifetimes, each kept | compiles |
| `t07_lifetimes_split.rs` | one lifetime split in two | E0107 |
| `t08_lifetimes_merged.rs` | two lifetimes merged into one | E0107 |
| `t09_lifetimes_rearranged.rs` | as many lifetimes, in another pattern | lifetime may not live long enough |
| `t10_result_borrows.rs` | a result borrowing the parameter | compiles |
| `t11_reference_into_native_refused.rs` | a uniform member that lends an element of the stack | E0277, `Ref: Native` |
| `t12_alias_in_native.rs` | the `#` member writing its stack through an alias | compiles |

Run `docs/rfc_extern/experiments/specialized-pair/run_probes.sh`. Build outputs
go to `target/` here, which git ignores.

Result, 2026-10-04, with stable rustc 1.97.1: the t probes all as expected; of
the s probes, s07, s08, s09 and s11 as described above. Each refusal's cause
was read from its diagnostic: a `Same` mismatch for t02, t03 and t05, `Ref:
Native` unsatisfied for t11, a family's lifetime count for t07 and t08.

What it does not show: both macros name a signature's lifetimes in the order
they first appear, an elided one by Rust's elision rules, which collecting a
signature's lifetimes needs as well (declarations §3). The registry's joining
of the two into one declaration, and the runtime's dispatch of each member,
are not exercised.
