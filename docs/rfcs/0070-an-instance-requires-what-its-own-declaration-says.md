# RFC-0070: An instance requires what its own declaration says

Status: Draft
Extends: RFC-0067, RFC-0068 (D7)

## Problem

`core::clone<T>(a: &T) -> T`, `core::eq<T>(a: &T, b: &T) -> bool` and
`core::hash<T>(a: &T) -> i64` are declared (`acvus-extern/src/core.rs`)
and have no instances. `dedup` and `hash_map` need `eq` and `hash` at the
element or key type; the two `dedup` tests and the seven `fixtures.rs`
tests fail on that resolution today.

An instance of `clone` at `Vec<T>` cannot be written. Its Rust body needs
`clone` at `T`, and the only way a handler receives an instance today is a
zero-width `Instance` parameter filled from the call site's table
(`Required::site`, `handler.rs`). `#[extern_fn]` refuses an instance
declaration that takes one (`acvus-extern-macro/src/lib.rs:476`): RFC-0067
D1 answered that a requiring instance stores what it requires beside the
value it serves. That answer holds for a stage (`Map<I, F>` is built by a
call that receives `next` at `I`) and fails for `Vec<T>`: a vector is not
built with `clone` at `T` beside it, and must not be — the requirement is
the instance's, not the value's.

## Decision

### D1. A zero-width parameter is the language's `where` clause

The signature is fixed: `clone` takes one argument. An instance declaration
may take, beside the signature's parameters, parameters of zero width in
the argument run (`Arg::Form = Nothing`, `ARGUMENTS = 0`, exactly what an
`Instance` parameter is today), which name what the instance requires:

```rust
#[extern_fn(instance = core::clone)]
fn clone_vec<T>(a: &Vec<T>, elem: InstanceOf<T, core::clone>) -> Vec<T>
```

This is `impl<T: Clone> Clone for Vec<T>`: the bound is a parameter the
script never writes and the checker fills. The macro's refusal at `:476`
is lifted for parameters of this kind; the site-table reading stays for
ordinary handlers.

### D2. The instance word names an entry, and an entry carries its requirements

RFC-0067 D1 made an instance one word beside the value: the address of a
mono glue. The word stays one word; what it addresses is an entry the
`Prepared` owns:

```rust
struct InstanceEntry {
    run: InstanceRun,                  // the glue and its task
    requires: Box<[&'static InstanceEntry]>,   // one per zero-width parameter, in declaration order
}
```

The glue's `Now`/`Later` type takes the entry (`fn(&InstanceEntry, ctx,
args)`); a zero-width parameter of the instance reads `entry.requires[i]`
where an ordinary handler's reads `site.requires[i]`. An instance that
requires nothing has an empty `requires` and costs one load more than
today, beside the indirect call it already pays.

The checker already resolves recursively: `Decision::Instance { required,
.. }` names, for `clone` at `Vec<i64>`, the instance `clone_vec` and its
requirement `clone` at `i64`. `prepare` writes that tree as entries, one
per (signature, type) pair, shared.

### D3. `InstanceOf` is the bundle of a type's entries

As RFC-0068 D7: a handler names the signatures it needs at a type as
bounds, the macro writes one struct with one word per bound, and each word
is an entry address from D2. `InstanceOf<T, core::clone>` in D1 is that
struct with one bound. `hash_map` takes `InstanceOf<K, (core::hash,
core::eq)>`.

### D4. No instance is derived

A type has an instance of `clone`, `eq` or `hash` only because an extern
registry declared one for it. `i64`, `f64`, `bool`, `String`, `Vec<T>`
(requiring `T`), `Option<T>` (requiring `T`) are declared in `acvus-ext`.
A type without one is refused by the checker where the requirement is
made (`InstanceWanted::Requirement`, `error.rs:26`), with the signature
and the type named. Structural equality of `Object` and `Enum` is a
language decision (whether total equality is equality) and is not made
here.

### D5. `hash_map` is two constructors

`hash_map()` requires `InstanceOf<K, (core::hash, core::eq)>` and fills
`Keying` from it; `hash_map_by(hash, eq)` takes the two closures as today
(`map.rs:450`). `Keying` is unchanged and the table code sees one call
either way. `Object`/`Enum` keys reach the map through the second
constructor until structural equality is decided.

### D6. `Instance` is an `InstanceOf` whose signature has a receiver

`next` at `Refs<Vec<T>>` is one function for every value of that type: it
is the type's, not the value's. RFC-0067 D1 placed its word beside the
value because no bundle existed; RFC-0068 D7 then separated `Instance`
(beside a receiver) from `InstanceOf` (a type's functions) on how the call
is made, which is not where the word belongs. With D1's zero-width
parameters the seam closes: a handler states `next` as a bound on the
stage's type and takes no `next` parameter,

```rust
fn map<I, T, U, E, Rt>(it: I, f: Closure<(T,), U, E, Rt>) -> Map<I, T, U, E, Rt>
where
    I: InstanceOf<sig::next<I, T, E, Rt>>,
```

and the macro synthesizes the zero-width parameter from the bound, filled
from the site table or the entry as D2 says. `Instance::call(ctx, &mut
recv, rest)` keeps its convention (`Ctx::name_receiver`), so `Instance<S,
I, Rt, T>` remains as the calling form of an `InstanceOf<S>` whose `S` has
a receiver, and stops being a second way an instance reaches a handler.
`iterator.rs`, `map.rs` and the twins lose their `next:` parameters in the
same change as D1.

## What it costs

- One more load per instance call (entry → glue).
- The macro grows: zero-width parameters on instance declarations, the
  bundle struct, entry-based reading.
- `InstanceTable::requiring` (`registry.rs:580`) learns transitive
  requirements: an instance's own zero-width parameters are met with the
  signature's instances the way a handler's are.

## Rejected

- **A uniform walker as the fallback instance** (`eq`/`clone` over `Value`
  by vtable): an implicit path where none was registered, and it decides
  structural equality by accident.
- **`T::clone` as a static call in the handler**: a uniform handler is one
  Rust body for every `T`; only a `#T` specialized member has a Rust `T`,
  and there Rust's own `Clone` already serves.
- **Storing the requirement beside the value** (RFC-0067 D1's answer):
  right for a stage, wrong for a container.

## Work, in order

1. `acvus-extern-macro`: lift the refusal at `lib.rs:476` for zero-width
   parameters on `instance =` declarations; emit the entry-taking glue
   type; emit the `InstanceOf` bundle struct per RFC-0068 D7.
2. `acvus-extern`: `InstanceEntry`; `InstanceRun`'s `Now`/`Later` take
   the entry; `Required::site` reads from site or entry by where the
   parameter stands; `InstanceTable` meets an instance's own requirements.
3. `acvus-mir`: `Decision::Instance` already carries `required`; confirm
   the tree is built for a requirement's requirement and that a missing
   one is `InstanceWanted::Requirement` naming the inner type.
4. `acvus-interpreter/prepare.rs`: `ChosenExtern { handler, requires }`
   (`:195`) becomes entries the `Prepared` owns; site tables and bundles
   point at them.
5. `acvus-ext`: the `next:` parameters become `I: InstanceOf<sig::next>`
   bounds (D6); instances of `clone`/`eq`/`hash` for the scalar types,
   `String`, `Vec<T>`, `Option<T>`; `dedup` over `InstanceOf<T,
   core::eq>`; `hash_map()`.
6. Tests: the two `dedup` and seven `fixtures.rs` tests pass; `clone` of
   `Vec<Vec<i64>>` (a requirement's requirement); a script requiring `eq`
   at a type without one is refused with the type named.

## What waits

- Structural `eq`/`hash` for `Object` and `Enum`.
- `InstanceOf` bounds on `#T` specialized members, where Rust's traits may
  stand in.
