# RFC-0070: An instance requires what its own declaration says

Status: Accepted
Extends: RFC-0067, RFC-0068 (D1, D5)

## Problem

`core::clone<T>(a: &T) -> T`, `core::eq<T>(a: &T, b: &T) -> bool` and
`core::hash<T>(a: &T) -> i64` are declared (`acvus-extern/src/core.rs`)
and have no instances.

An instance of `clone` at `Vec<T>` cannot be written. Its Rust body needs
`clone` at `T`, and the only way a handler receives an instance is a
zero-width `Required` parameter filled from the call site's table
(`Required::site`, `handler.rs`). A mono glue is one shape,
`fn(ctx, rest)`: it has no site table, so it has nowhere to read a
requirement from, and `#[extern_fn]` refuses an instance declaration that
takes one (`acvus-extern-macro/src/lib.rs:476`). RFC-0067 D1 answered that
a requiring instance stores what it requires beside the value it serves.
That answer holds for a stage (`Map<I, F>` is built by a call that receives
`next` at `I`) and fails for `Vec<T>`: a vector is not built with `clone`
at `T` beside it, and must not be — the requirement is the instance's, not
the value's.

Two further facts, corrected from the earlier draft of this RFC:

- The checker decides one level. `Scheme::requires` holds the requirements
  of the *declaration* called, and a candidate `InstanceSig` carries no
  requirements of its own (`ty.rs:357`), so `Decision::Instance` settles on
  `clone_vec` for `clone` at `Vec<i64>` and nothing then decides `clone`
  at `i64`.
- The nine tests the draft attributed to this gap (`dedup` ×2,
  `fixtures.rs` ×7) fail for other reasons: a `Dedup<'0, #i64, '0>`
  instance pattern against a `Dedup<Items<i64>, i64, Pure>` call type, and
  scripts piping a `Vec<T>` into `map` with no `next` at `Vec<T>`. Neither
  is this RFC's.

## The invariant this RFC keeps

**`T` is `T` on both sides of the boundary, and the boundary is the
macro.** An extension body sees its own Rust types and the `Instance`
values it was declared with; `Rt::Value` enters a body only as the fill of
a `Var<kind::Type>` parameter, and no body writes `erase`, `materialize`,
`Owned::from_value`, or a `Lent`. Every `T → Value` and `Value → T` in this
RFC is in code `#[extern_fn]` or `extern_signature!` writes. The crossing
primitives — `Cross`, `OneValue`, `Form`, `Loan`, `Owned`, `Ref`,
`Restore*`, `Lent` — are unchanged.

## Decision

### D1. An instance declaration takes what it requires

The signature is fixed: `clone` takes one argument. An instance declaration
may take, beside the signature's parameters, `Required` parameters —
zero-width in the argument run, `Form = Nothing`, `ARGUMENTS = 0`, exactly
what an ordinary handler's `Instance` parameter already is — naming what
the instance requires:

```rust
#[extern_fn(instance_of = core::clone, effect = pure)]
fn clone_vec<T, Rt>(ctx: &mut Ctx<'_, Rt>, a: &Vec<T>, elem: Instance<core::clone<T, Rt>, T, Rt>) -> Vec<T>
where
    T: Var<kind::Type> + Deref<Target = Rt::Value> + ..,
```

This is `impl<T: Clone> Clone for Vec<T>`: the bound is a parameter the
script never writes and the checker fills. The refusal at `lib.rs:476` is
lifted, and so is the macro's check that some parameter stands exactly at
the requirement's variable: that check served `instance_at`, the
site-table lookup by a parameter's settled type, which D2 removes. The
checker now settles a requirement's variable by unifying the pattern
beside the declaration's type, so `T` settles from inside `&Vec<T>`, and
`hash_map()`'s `K` settles from its return type at the first `insert`.
The site-table reading (`Required::site`) stays for ordinary handlers.

### D2. The instance word addresses an entry, and an entry carries its requirements

RFC-0067 D1 made an instance one word beside the value. The word stays one
word and `Instance` stays one `Rt::Value` (`ONE_VALUE`); what the word
addresses is an entry the `Prepared` owns:

```rust
pub struct InstanceEntry<Rt: Runtime> {
    pub run: InstanceRun,            // the glue and its task, as the registry holds it
    pub requires: Box<[Rt::Value]>,  // one instance word per `Required` of the instance's
                                     // declaration, in declaration order; each addresses an entry
}
```

`InstanceRun` keeps its meaning — the address of a mono glue and the task
it runs at — and the registry's `InstanceTable` keeps holding one per
instance. `Runtime::instance_value(&InstanceEntry<Self>) -> Value` and
`unsafe fn instance_entry(&Value) -> &InstanceEntry<Self>` replace the pair
over `InstanceRun`; the interpreter's `Kind::Instance` / `InstanceAwait`
tags the word by `entry.run.task()` as before.

The glue's `Now`/`Later` type takes the entry:

```rust
pub type Now<Rt> = for<'a> unsafe fn(&InstanceEntry<Rt>, &mut Ctx<'_, Rt>, Rest<'a, Rt>) -> Ret<Rt>;
```

and inside the glue a `Required` parameter is bound from
`entry.requires[i]` where an ordinary handler's is bound from
`site.requires[i]`. Which of the two the macro writes is decided by the
declaration kind, at expansion. An instance that requires nothing has an
empty `requires` and costs one load more than before, beside the indirect
call it already pays.

`prepare` builds entries: one per distinct (signature, chosen tree), shared
across the sites that chose it, address-stable (`Box`), owned by
`Prepared` for as long as its operations live. A site table's `requires`
holds the words of the entries the site's own requirements chose, so
`Required::site` is `Instance::at(site.requires[NTH])` and reads no
registry. `InstanceEntries::glue(signature, RequiredInstance)` is what
`prepare` asks the registry when it builds an entry, and is the only
question left on that trait; `InstanceEntries::overload` and
`InstanceTable::requiring` go.

### D3. The checker decides the tree, and the IR carries it

A candidate carries the requirements of its own declaration:

```rust
pub struct InstanceSig {
    pub ty: PolyTy,
    pub admits: Task,
    pub task: Task,
    /// Written at this instance's own variables, as `ty` is.
    pub requires: Vec<RequirementSig>,
}
```

When `Decision::Instance` settles on a candidate with requirements, the
solver instantiates the candidate's type and its requirement patterns with
one variable map, opens one `Decision::Instance` per requirement — `call`
the instantiated pattern at the requirement's task, candidates the required
signature's instances, `required: Some(signature)` — and records the
children on the parent in declaration order. The solver reaches a
signature's instances through a map lent at construction, beside the
`TypeRegistry`, because a nested requirement's candidates cannot be
embedded in the candidate (the tree is cyclic: `clone` at `Vec<T>`
requires `clone`, whose instances include `clone` at `Vec<T>`). A child
that reaches no instance fails as `Unsettled::NoInstance` with
`required: Some(signature)`, which `InstanceWanted::Requirement` already
reports with the signature and the type named.

The IR names the tree structurally:

```rust
pub struct Chosen {
    pub signature: QualifiedRef,
    pub instance: usize,        // the index among that signature's instances
    pub required: Vec<Chosen>,  // one per requirement of that instance's declaration
}

Callee::Extern { id: QualifiedRef, instance: usize, required: Vec<Chosen> }
```

`OverloadSpace` and its position integer go. The draft kept a single
position by mixed radix over a flat product; a tree over a cyclic
requirement graph has no finite radix, so the encoding cannot be kept.
`Chosen::signature` makes the node self-describing, so `prepare` asks the
registry only `glue(signature, instance)` per node.

### D4. `Instance::call` takes the receiver the signature declared

`Signature::Recv<'a>` is `&'a This`, `&'a mut This`, or `This` by the
signature's first parameter (`instance.rs:207`), and its doc already says
the handler's `Instance::call` is where a wrong mode is refused.
`Instance::call(ctx, recv: S::Recv<'r>, rest)` now takes it; the
`&'r mut I` it took before was one mode for all three, and a handler
holding `&T` — every element of a `&Vec<T>` — could not call `clone` at
`T`.

`Instance::call` names the receiver in `ctx` through `Receiver<Rt>`, a
trait on the `Recv` type with two impls: `&I` where
`I: Deref<Target = Rt::Value>` and `&mut I` where `I: DerefMut<..>`. The
bound sits on the method, not on the `Signature` impl, so a payload that
stores an `Instance<S, I, Rt>` states nothing about `I`; the mode is
checked where the call is written. A signature taking its receiver by
value has no impl, and the missing impl is the refusal: no customer
requires one. `Signature::call_now` / `call_later` keep their shape and
read the receiver the call named.

`Ctx::recv` is a `*const Rt::Value` and `Ctx::receiver` hands the glue a
`&Rt::Value`: every glue arm reads the word and opens what it names at the
loan the signature's first parameter declared, so no `&mut` to the
receiver word itself is ever made.

### D5. The core signatures

`core` holds the signatures the compiler names, and nothing else:

| signature | shape | the compiler names it at |
|---|---|---|
| `eq` | `<T>(a: &T, b: &T) -> Bool` | `==`, `!=` on an extension type (RFC-0020) |
| `cmp` | `<T>(a: &T, b: &T) -> Int`, `-1`/`0`/`1` | `<`, `<=`, `>`, `>=` on an extension type (RFC-0020 amended) |
| `clone` | `<T>(a: &T) -> T` | the explicit copy (RFC-0018); `String`'s is an instruction |
| `hash` | `<T>(a: &T) -> U64` | a map's keying, once a map literal exists |
| `to_string` | `<T>(a: &T) -> String` | interpolation, once it lowers to a call |

`cmp` answers an integer because the language has no `Ordering` and
declaring one would be a core type beside the signature; `-1`/`0`/`1` is
what `string::cmp`, `num::total_cmp` and the `sort_by` comparator already
answer. `hash` answers `U64` because a hash is a bit pattern, not a
number: `Hasher::finish` is `u64`, and `hash_float` folds `to_bits` into
an `i64` today only because the signature said so. `to_string` moves its
declaration from `acvus-ext/src/conversion.rs` to `acvus-extern/src/
core.rs`; its instances stay where they are.

An instance of `hash` at `T` is admitted only where an instance of `eq`
at `T` is declared: `Externs::combine` refuses the pair's absence
(`CombineError::HashWithoutEq`). That two `eq`-equal values hash equal is
the half the registry cannot hold; a test per type holds it.

The standard registry declares `eq`, `clone`, `cmp` and `hash` at `Int`,
`Float`, `Bool`, `Byte` and `String`, for the requirement sites that reach
those types; the operator table keeps its instructions for them (RFC-0020
amended), and a test per type pins the instance to the instruction's
meaning. `Vec<T>` has `eq`, `clone`, `cmp` (lexicographic) and `hash`,
each requiring the same signature at `T`. `Decimal` gains `cmp`.

`hash_map()` takes no argument and requires `hash` and `eq` at `K`;
`hash_map_by(hash, eq)` is the closure form, which is where an object key
goes. `dedup` requires `eq` at `T` and is one uniform handler.

`Object` and `Enum` have no instance of any of the five, and none is
declared by this RFC (RFC-0019 "Open questions").

## What it costs

- One more load per instance call (entry → glue).
- The macro grows: `Required` parameters on instance declarations bound
  from the entry, the entry-taking glue type, the receiver named by mode.
- `Callee::Extern` widens by a `Vec<Chosen>`, empty for every call of a
  declaration without requirements.
- `Runtime`'s two instance methods change their argument type; the four
  test runtimes in `acvus-extern/tests` and `acvus-ext/tests` follow.

## Rejected

- **A uniform walker as the fallback instance** (`eq`/`clone` over `Value`
  by vtable): an implicit path where none was registered, and it decides
  structural equality by accident.
- **`T::clone` as a static call in the handler**: a uniform handler is one
  Rust body for every `T`; only a `#T` specialized member has a Rust `T`,
  and there Rust's own `Clone` already serves.
- **Storing the requirement beside the value** (RFC-0067 D1's answer):
  right for a stage, wrong for a container.
- **`prepare` re-resolving inner requirements from settled types**: the
  machine checks nothing the compiler proved (RFC-0068 D5).
- **A single overload position over the tree**: no finite radix.
- **A bundle struct per bounded variable (`InstanceOf<T, (A, B)>`)** and
  **folding `Instance` into a bound** (the draft's D3, D5, D6): not needed
  by this RFC's customer, and RFC-0067's "one value per signature" holds.

## Work, in order

1. `acvus-mir`: `InstanceSig::requires`; `Chosen` and the widened
   `Callee::Extern`; `OverloadSpace` removed; the solver's nested
   decisions and the signature-instances map; `callee_of` building the
   tree; `graph/infer.rs` filling `InstanceSig::requires` from the
   declared instances.
2. `acvus-extern`: `InstanceEntry`; `Runtime::instance_value` /
   `instance_entry`; `CallSite::requires: &[Rt::Value]` and
   `Required::site` reading it; `ArgAt` without `instances`;
   `InstanceEntries` reduced to `glue`; `DeclaredInstance::requires` and
   `Instances::signatures` carrying it; `add_instance` reading
   `decl.requires`, meeting the bound and requiring the glue;
   `Signature::call_now` / `call_later` taking `Recv`.
3. `acvus-extern-macro`: refusal lifted; glue shape; `Required` bound
   from the entry inside a glue; receiver named by mode in the
   `Signature` impl; `instance_requiring_an_instance` compile-fail test
   becomes a passing declaration in `acvus-extern/tests/decl.rs`.
4. `acvus-interpreter`: `prepare` builds and interns entries per
   `Chosen` tree, owned by `Prepared`; site tables filled with words;
   `Runtime` impl; `Value::instance` over the entry.
5. `acvus-extern/src/core.rs`: `cmp` and `to_string` declared; `hash`
   answers `U64`; `combine` refuses `hash` without `eq`. `acvus-mir`:
   the comparison operators on an extension type call `core::cmp`.
6. `acvus-ext`: the instances D5 lists; `hash_map()` and `hash_map_by`;
   `dedup` over required `eq`.
7. Tests: `clone` of `Vec<Vec<i64>>` (a requirement's requirement); a
   script requiring `clone` at a type without one is refused with the
   type named; a handler calling `eq` over a `&T` receiver; each word
   instance against its instruction; `<` on `Decimal`.

## What waits

- Structural `eq`/`hash` for `Object` and `Enum` (RFC-0019).
- `Option<T>` instances.
- A by-value receiver requirement with a customer.
- `#T → T` demotion, so that a `Monomorphize` instance pattern such as
  `Dedup<'0, #i64, '0>` stands at a uniform call type; and `next` at
  `Vec<T>` for a script that pipes a vector into a stage. Neither is this
  RFC's.
