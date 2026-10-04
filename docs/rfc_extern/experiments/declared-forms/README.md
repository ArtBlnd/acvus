# Probes: a written type's form, through traits

The evidence for types §3, the two forms an extern-defined type is written
in: whether a type's form can be decided through the types' own impls, so
that no syntax, alias or nesting decides it. `forms.rs` holds the instrument,
a trait whose constants state what a type holds, and stand-ins for the
extern crate's and an author's types; the trait is the probe's means, not a
decided primitive. An extern-defined type's impl is the one rule a derive
would emit for every such type. A runtime parameter is left out, since no form
depends on it, and a type variable is written as its stand-in,
`Uniform<Var<N>>`, as a declaration writes it.

Each probe names a type in a non-generic `const` item, so its check runs where
the type is written.

| Probe | What it checks | Expected |
|---|---|---|
| `f01_uniform_form.rs` | every argument a `Uniform` at its root: uniform, unmarked | compiles |
| `f02_native_form.rs` | no `Uniform` anywhere: whole native, marked | compiles |
| `f03_type_variable.rs` | a container of a type variable's stand-in: uniform | compiles |
| `f04_no_argument.rs` | a type with no argument: unmarked | compiles |
| `f05_mixed_tuple.rs` | an argument rooted at a tuple that holds a `Uniform` | E0080 |
| `f06_mixed_option.rs` | an argument rooted at an `Option` that holds a `Uniform` | E0080 |
| `f07_mixed_arguments.rs` | one argument uniform, the other native | E0080 |
| `f08_alias_uniform.rs` | an alias resolving to a `Uniform` | compiles |
| `f09_alias_hides_mixed.rs` | an alias hiding a mixed argument | E0080 |
| `f10_alias_hides_native.rs` | an alias hiding a whole native type, still marked | compiles |
| `f11_nested_native.rs` | a native type whose argument is native: one whole, marked | compiles |
| `f12_uniform_inside_native_argument.rs` | an argument rooted at an extern-defined type that holds a `Uniform` | E0080 |
| `f13_native_held_as_argument.rs` | a whole native type held uniformly as an argument | E0080 |
| `f14_signature_positions.rs` | a signature's positions: a marked type behind a reference, a uniform one in an `Option`, a scalar | compiles |
| `f15_mixed_behind_reference.rs` | a mixed type behind a reference | E0080 |

Run `docs/rfc_extern/experiments/declared-forms/run_probes.sh`. Build outputs
go to `target/` here, which git ignores.

Result, 2026-10-04, with stable rustc 1.97.1: all fifteen as expected. Each
E0080 is the check's own panic, "an extern-defined type's arguments are
neither all uniform at their root nor all native" for f05, f06, f07, f09, f12
and f15, and "a whole native extern-defined type stands as an argument of a
uniform one" for f13.

What it does not show: a generic item's check would run only where the item
is instantiated, so a macro writes a declaration's types with its type
variables put as their stand-ins, which it already does to declare them.
