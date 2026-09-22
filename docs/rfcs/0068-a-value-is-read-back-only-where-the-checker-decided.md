# RFC-0068: A value is read back only where the checker decided

Status: Draft (2026-09-22) — the design before any code run; the order of
work below is the order of the runs.

## The principle

**Forgetting is a function; remembering is a choice.** A Rust type `X` maps
to an unconstrained `T` by one function, so `X → T` is sound. `T → X` has as
many candidates as there are `X`, so it is unsound unless something outside
Rust chose which. In this framework two things stand as `X`:

- a handler's own type variable `T: Var<kind::Type>`, which the glue fills
  with `Owned<Rt>` — the runtime's value — and which stays `T` in the
  handler's Rust;
- `#T`, the specialized representation, which is a real Rust instance
  (`Vec<u64>`), and which exists only as an optimization a declaring crate
  opts into.

While a value is held at its type variable, Rust's checker carries the
acvus checker's decision for free: two positions typed at the same `T` hold
the same acvus type because the acvus checker filled `T` once. The moment
`T` is erased to `Rt::Value` inside a handler, that decision is gone from
the Rust type, and reading the value back — `Instance<T>` beside
`Instance<U>` with `T ≠ U` — is a bypass of the acvus checker with nothing
left to refuse it.

The one place where `Rt::Value → T` has a ground is the crossing from the
machine into a handler: the site's ground type was settled by the acvus
checker, and `#[extern_fn]` wrote the glue from that type and from no other.
Every other `Value → T` is cut.

## Problem

`acvus-extern` exposes the reverse direction as public `unsafe fn`s and
handlers call them: 301 `unsafe` tokens in the materialize direction across
`acvus-extern`, its macro and `acvus-ext`, 71 of them in extension handlers
(`iter`, `iterator`, `map`, `vec`, `array`, `deque`). Each carries a
`// SAFETY:` sentence that restates a fact the acvus checker knows and Rust
does not. Two consequences:

- A handler author writes `unsafe` for ordinary work, and the sentence they
  write is not checkable by anything.
- The `unsafe` has no defined behavior of its own: Rust's type system is
  used as a checker of the handler's own code while the invariant the
  `unsafe` relies on lives in another compiler. What the block may do is
  whatever compiles.

`Instance` (RFC-0067) is the sharpest case. `Instance::at(value)` is
`pub unsafe` and turns any word into `Instance<S, I, Rt>`, so `call` must be
`unsafe` too; and the `ExternType` derive refuses a payload that names a
type variable, so every adaptor erases its `I` to `Owned<Rt>` and its `E` to
`Opaque`, stores `Instance<sig::next<Owned<Rt>, ..>, Owned<Rt>, Rt>`, and
retypes through `Instance::at` at construction — the exact bypass above,
forced by a guard whose stated reason is the carrier design RFC-0067
removed.

## Decision

### D1. Three rules, checked at review, not at run time

1. Only `T → Value` exists as a function. `Value → T` is not a public
   operation of `acvus-extern`.
2. A handler's type variable is stored and passed at the variable. A
   payload that holds a value at `I` holds it as `I`, and what stands beside
   it (an `Instance`, a `Closure`) is typed at the same `I` and `E`. The
   glue fills every such `I` with `Owned<Rt>` and every `E` with `()`
   (`generics.rs::runtime_stand_in`), so one Rust type and one `TypeId`
   result per declaration regardless of what the payload names; nothing is
   paid for the type.
3. `#T → T` always exists (`family_casts::erase`). An instance is resolved
   at the ground type including its representation: `matches_pattern_with`
   already reads a pattern's representation variable as `Uniform` and
   refuses `Specialized` (`ty.rs:696-704`), so a requirement at a
   variable is met only by the uniform instance.

Every check inside the crossing is a `debug_assert!`. The acvus checker is
the ground; the machine restates none of its proofs in release.

### D2. `Instance` is closed the way `Erased` is

`Erased<R, T>` is the model already in the crate: one `Owned<R>`, a `T` in
the type, one constructor (`new(rt, value: T)`), and `unsafe` only inside
methods whose invariant the constructor established.

- `Instance::at` becomes crate-private. The one constructor is
  `Required::site`, which reads `InstanceTable::instance_at` at the site's
  settled type — the crossing the checker decided.
- `Instance::call` and `call_await` become safe. Their receiver is `&mut I`
  with `I: DerefMut<Target = Rt::Value>`; the far side's mono glue reads
  that value as the instance's literal receiver type, which is the same
  crossing as any other parameter's.
- `Signature::call_now`'s `transmute` of the word to `S::Now` stays, under
  `Instance`'s invariant. `InstanceRun::at` stops being a public `usize`:
  the macro constructs it from the typed `fn` and nothing else can.
- `into_value` stays: it is the forgetting direction.

### D3. A payload names its variables

The two `ExternType` guards at `acvus-extern-macro/src/lib.rs:1274-1291`
(payload is the variable) and `:1322-1330` (payload mentions a variable)
are cut, with the `payload_per_instantiation` attribute that exists only to
opt out of the second. An adaptor is then:

```rust
pub struct NMapBody<I, E, Rt> where .. {
    inner: I,
    next: Instance<sig::next<I, i64, E, Rt>, I, Rt>,
    f: Closure<(i64,), i64, E, Rt>,
}
```

and `next_nmap` calls `it.0.next.call(ctx, &mut it.0.inner, ())` with no
`unsafe` and no retype.

What the guards protected is kept by the type instead. A payload is the Rust
type its arguments spell: `Items<String, ..>` built by a handler holds
`IntoIter<String>`, and the generic `next_items<T>`, whose one glue fills `T`
with the runtime's value, reads `IntoIter<Value>`. Those are two types, and
the declaration says so: a derived extension type's argument that names no
variable is the slot's specialized representation, `Items<#String>`
(`TyArg::held`, and the type's slots are `specializable`); an argument
naming a variable, or an `Erased<Rt, _>`, is uniform. `X<#T>` and `X<T>` do
not join, so a generic declaration over `X<T>` refuses an `X<#T>` at compile
time, where before the cut guard refused the payload and after it nothing
did.

### D4. The crossing is the only reader

`Runtime::materialize`, `value_as_ref/mut`, `inline_ref/mut`, `deref/mut`,
`OneValue::materialize`, `FromValue::from_value`, `Loan::borrow`,
`Ctx::receiver` and the `Restore*` traits remain `unsafe fn` and remain
callable from generated code, which lives in the declaring crate; they are
`#[doc(hidden)]` and no handler names them. A handler that needs a value at
a type has it at that type from its parameters, its payload, or an
instance's typed return.

### D5. A requirement is a call the checker decides

A handler's `Instance<sig::next<I, T, E, Rt>, I, Rt>` parameter is recorded
as `Requirement { signature, pattern, calls }`, where `pattern` is the
signature's own type with its variables replaced by what the marker names
(`RequirementOf::pattern`, written by `extern_signature!`). The compiler
carries it on `FnKind::Extern::requires`; `declared_scheme` pairs it with
the signature's instances; and `instantiate_scheme_with` opens one
`Decision::Instance` per requirement, its call type being the pattern at
the same fresh variables as the handler's type. The instance decision that
already serves a direct call of the signature then does everything: it
narrows by shape and by task, binds the signature's other variables from
the chosen instance (`T := i64` from `next` at `NRange`), and refuses with
the sentence that names the signature, the call it could not place, and the
instances it could have reached.

Cut with this: `TyVarBound::OneOf::required`, `Required`, `InnerBound`,
`InstanceShape`, `InstanceSets` and its `admits` recursion, `both_require`,
the registry's `meet` over instance sets, `BoundAt`, `path_to_var`, and the
two combine refusals that guarded them. A candidate carries its body's task
(`InstanceSig::task`), and a requirement admits only instances at or below
its `calls`.

An instance's own requirement — the `Doubled<T>` whose `advance` needs
`advance` at `T` — is not decided under a requirement. `#[extern_fn]`
refuses an instance that takes an `Instance` parameter (its requirement is
a field of its payload), so no declaration produces one; the bound machinery
that decided it recursively is gone with it.

### D6. A result at a signature variable crosses as the runtime's value

`next<I, T, E>` returns `Option<T>`. The instance at `NRange` returns
`Option<i64>`; the instance at `NMap<I, T, U, E>` returns `Option<U>`,
whose glue is compiled with `U` filled by `Owned<Rt>`; the requirer
`nsum<I, E>` reads `Option<i64>`. These are three Rust types, and one `fn`
pointer type per signature has to serve them. So a signature whose result
mentions one of its type variables crosses that result as `Rt::Value`: the
signature's module writes `Ret<Rt>` and a `Returned<Rt>` trait whose
`cross` erases what the instance's handler returned and whose `restore`
materializes at the requirer's own `Ret`, which the checker unified with
the instance's result (D5). A signature whose result is concrete crosses it
as it is typed, and `Returned` is the identity. `Signature::call_later`
returns `impl Future` so the restore adds no box.

### D7. `Instance` is a method of a value; `InstanceOf` is the functions of a type

Two requirements look alike and are not.

`Instance<S, I, Rt>` is one value's `S`: a receiver `I` and the glue that
runs `S` on it, called as `next.call(ctx, &mut it, rest)`. A handler that
holds one stores the receiver beside it (`NMapBody { inner: I, next }`).
Each `Instance` stands with its own receiver, so a handler takes as many as
it has receivers: `chain` holds two pipelines and the `next` of each. The
iterator is this shape.

`InstanceOf<S>` is a type's `S`: `clone` at `T`, `eq` at `T`, `hash` at
`T`, functions that take their arguments through the ordinary crossing and
belong to no value. With no receiver to stand beside, they arrive as one
bundle per handler, which names several signatures and stores nothing:

```rust
fn insert<K, V, I, Rt>(m: &mut HashMap<K, V>, key: K, value: V, at: &I)
where
    I: InstanceOf<core::hash<K, Rt>> + InstanceOf<core::eq<K, K, Rt>>,
```

`I` is the handler's own bundle: a struct `#[extern_fn]` writes with one
word per bound, each filled by `Required::site` from the call site's table
the way an `Instance` word is, and each `InstanceOf<S> for I` impl calls
its own field's glue at `S`'s `Now` type. Which pointer a call takes is
decided by the type, not looked up: the bundle is a vtable whose entries
Rust resolves statically. It is not RFC-0067's rejected bundle, which was
a runtime-typed list a value carried across the machine; this one lives in
one handler's frame, and a Rust helper generic over `J: InstanceOf<A>`
takes `&I` by monomorphization, while another handler called from a
script fills its own bundle from its own site.

`T` stays `T`. What a handler may do with a `T` is what its bounds enable;
nothing is stored at the type and nothing is asserted about a value.

### D8. `X<#T>` reaches `X<T>` through the family's declared cast

`#T` is a hint: the value kept as the Rust type it is. At a bare slot `#T`
and `T` are one type in two representations and the conversion between them
is the crossing's own. Under a constructor they are not: `X<#T>` and `X<T>`
are invariant in the slot, and only a function written for `X` can turn one
into the other. RFC-0041 already names it: a family declares its cast
`F<#m> -> F<m>`, a pure ExternFn marked `#[extern_cast]`, and a flow whose
only disagreement is the slot's representation is a conversion decision
answered by that cast. The cast is Rust over the whole payload, one call
whatever the element count.

1. With no declared cast, `X<#T>` where `X<T>` is asked is a type mismatch.
2. With one, the checker inserts it at the conversion site.
3. A value whose path would go `#T -> T -> #T` or `T -> #T -> T` is kept
   uniform from where it is made: the cast runs once, at the producer.
4. `X<#T> -> X<T>` is the direction that is declared. A value used only at
   `#T` never converts, which is the best case; nothing asks for the
   unboxing direction of an extension type, and a value that would need it
   is uniform from the start by rule 3.

RFC-0041's conversion sites are a call argument, a store, a return, a
pattern's source and an `else` branch, each a place where a slot's
representation meets a fixed one. A requirement adds one. A consumer is
`fn collect<I>(it: I, next: Instance<next<I, T>, ..>)`: its parameter is a
bare variable, which takes `Items<#String>` directly (RFC-0043 rule 1), and
the representation is first asked for inside the requirement, where no
instance of `next` stands at `Items<#String>` and one stands at
`Items<T>`. That is the conversion site: the argument standing at the
required variable. Where the requirement finds no instance at the settled
type and the family's cast reaches a type that has one, the conversion is
decided at that argument, consumes it (RFC-0041), and the variable is the
cast's target.

A source that wants the best case declares its own `next` at `X<#T>` and
never converts.

### No row gets slower

The machine is measured (`docs/performance.md`, `benches/README.md`) and
the measured state is the contract. Every run below is accepted only
against the base it started from, under the README's protocol: `setarch -R
taskset -c <core>`, alternating arms, one discarded warm-up, `uptime` before
each rep. A row is a regression when the two arms' ranges are disjoint with
the tree arm worse; a difference under the ~7 % inter-build floor is not a
result either way. `cargo bench --bench asm_probe` must read the same tail
jump count and no body newly holding a stack address across a listed
callee. A change that trades a row for convenience is not accepted; the
crossings, the site table and the mono glue already give every step a shape
that costs nothing, and the work is to use them.

### Order of work

1. Close `Instance` (D2) and cut the payload guards (D3); rewrite the
   `nit::` spike and `decl.rs` on it so no `unsafe` remains in a handler
   there. Acceptance: `iter_next.rs`, `accum.rs` and the two compile-fail
   goldens pass; `grep unsafe` over those handlers is empty.
2. Move `iter::` (`acvus-ext/src/iter.rs`, `iterator.rs`) onto `Instance`:
   the `Box<dyn>` chain, `drain`, and `Iter<_, _, _, Rt>` go. The 30
   materialize sites there are the chain's untyped yields and have no
   replacement to write. Acceptance: the `iter::` differential corpus and
   benches unmoved or faster.
3. `map`, `vec`, `array`, `deque`: element access through
   `Runtime::value_as_ref::<T>` on a `Vec<Owned<Rt>>` becomes a typed
   crossing at the boundary. Each is its own run.
4. Hide the remaining `unsafe` surface (D4) and take the `#[doc(hidden)]`
   inventory to zero public `unsafe` in the crate's documented API.

## What it costs

- The `#T` path for a requirement's variable: a `Vec<#T>` flowing into a
  bare `I` binds `I := Vec<#T>` and no instance matches, so the refusal is
  the `OneOf` sentence rather than an inserted erase. Whether the solver
  inserts the erase at a bare variable is examined at step 1 with a script;
  if it does not, that is a separate decision and this RFC records it as
  open.
- `Ctx::receiver` stays a raw pointer read by generated code. It is the
  receiver's crossing and is hidden, not removed.
- The `TypesOnly` runtime's stubs stay `unreachable!`; they are not a
  crossing.

## Rejected

- **A checked `materialize` (`TypeId` at run time, `Any`-style).** A safe
  reverse direction would make the bypass safe to write, and the check
  restates a proof the checker already holds. The framework's rule is that
  the machine checks nothing the compiler proved.
- **Keeping `Instance::at` public with a longer safety contract.** The
  contract cannot be checked by its reader; the constructor is the only
  place the fact is known.
- **`payload_per_instantiation` as the default rather than a cut.** The
  attribute exists only to opt out of the guard; with the guard gone it
  names nothing.
- **A `Value`-typed payload with a runtime tag.** A stamp in the value is
  RFC-0067's rejected list, for the same reason: it moves the checker's
  decision into the machine.

## What waits

- Whether the consumer's receiver bound (`I: Var<kind::Type> +
  DerefMut<Target = Rt::Value>`) moves onto `Instance` or `Signature`
  (RFC-0067, open) — unchanged by this RFC.
- The (b) class — lifetime extension in `Slice::at`, `StrView::as_str`,
  `lend_run`, and the two live `&mut` in `vec.rs`'s `ptr::swap` — is a
  different invariant and a different RFC.
