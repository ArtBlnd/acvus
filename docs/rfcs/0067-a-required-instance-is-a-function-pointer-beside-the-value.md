# RFC-0067: A required instance is a function pointer beside the value

Status: Accepted (2026-09-20) — design settled; the iterator is the first
customer, generic container handlers (`eq`, `ord`, `hash` of an element)
the second.

## Principle

**The machine holds no generics.** Every type at a call is ground; an
instance of a shared signature is a value — the entry pointer of one
concrete handler; no function type crosses the extern boundary as a
type-level list. There is no position in the machine where a type is
computed, and that absence is the proof that the framework needs no
higher-kinded types: whatever a type variable ranges over is settled by
the checker and arrives at the machine as a word.

The runtime shape this leaves is C's: a `void*` with a function pointer
beside it. The difference from C is the compiler standing next to it — the
checker picks the pointer, the loans analysis governs the borrow, the
effect system says whether the call may suspend — so the shape carries a
proof C never had.

## Problem

A handler could require nothing of a type variable but its kind. RFC-0019
gave the language shared signatures — `core::clone<T>(&T) -> T`, one
instance per concrete type, resolved at a call by the call's ground type —
and closed with "no generic functions in the language, and no `dyn`;
nothing new is inferred or dispatched". A Rust handler generic over `T`
could not call `clone` on its `T`: the marker bound `T: HasInstance<clone>`
narrowed admission and gave the body nothing to call, and the type-helper
pass deleted it for that reason.

The iterator paid for the gap. Without a way to say "`I` is a type with a
`next`" and call it, every design put the whole pipeline inside one
extension type: a `Box<dyn>` per stage (master), or one Rust type per
pipeline length with a bound of eight and 292 registry names (RFC-0065,
built and rejected). Both carried in a value what the checker already knew
at every site, and RFC-0065 in particular carried a type-level list of
element types across the boundary and unwound it with a recursive trait —
generics inside the machine, which the principle above forbids.

## What exists

- A shared signature: a name with one polymorphic type and no body
  (`SignatureDecl`); an instance: an `ExternFn` declared `instance_of =
  sig` at a concrete type; `Externs::combine` refuses two instances whose
  types unify (`DuplicateInstance`) — RFC-0019.
- A member instance stands at a pattern: the declaration with one variable
  replaced by a member type its `OneOf` bound names (`at_a_member`).
- A type variable's kind is `Var<K>`; what fills `kind::Type` is every type
  that names an acvus type (`TyArg: Var<kind::Type>`) and the runtime's
  carrier `Owned<Rt>`; a `Monomorphize<(..)>` variable is filled by each
  member in turn and the handler compiled once per member.
- A closure crosses as `Closure<A, R, E, Rt>`: a value the glue fills at
  the site and the body calls at its declared types (RFC-0050 rule 6).
- The machine's `FnValue` is `Arc<dyn Callable>` with captures; no `Kind`
  holds a bare function pointer.

## Decision

1. **A requirement is a Rust trait bound naming the signature at the
   handler's own variables.** `fn sum<I, T, E, Rt>(it: I) -> T where I:
   Instance<next<I, T, E, Rt>>`. What the instance's other variables are
   — the element `T` of an `I` — is read off the instance's type by
   unification, as a call reads it today. No associated type and no
   marker: the signature's own variables carry the relation, and the body
   calls `I::call(&it, rt, frame, args)`.

2. **An instance may stand at a pattern.** `fn next_map<I, T, U, E, Rt>(it:
   &mut Map<I, T, U, E, Rt>) -> Option<U> where I: Instance<next<I, T, E,
   Rt>>` is the instance of `next` at the pattern `Map<I, T, U, E>`. Two
   instances whose patterns unify are refused at combine, as two concrete
   instances at one type are today. A bound is therefore a predicate — an
   instance whose pattern unifies with the ground type exists and its own
   bounds hold — decided on ground types by structural recursion, which
   ends because types are finite trees. This lives in the checker only.

3. **The compiler passes the pointer.** At a call whose declaration bounds
   a variable by `Instance<S<…>>`, lowering resolves `S` at the ground type
   and passes the glue entry of that instance as one more argument: a
   word holding a function pointer of the framework's uniform value ABI.
   Nothing is looked up at run time; the pointer is right for the same
   reason a `Monomorphize` member is right — the checker chose it.

4. **The carrier of a bounded variable is the value with its pointers
   beside it.** In the uniform instantiation a type variable is filled by
   the runtime's carrier; a variable with `n` `Instance` bounds is filled by
   that carrier and `n` pointers, assembled by the glue from the arguments
   of Decision 3. `Instance<S>::call` reads the pointer for `S` out of
   `&self` and calls it. Because the pointers ride with the value, a
   handler that stores `it: I` in a value it builds (`Map(it, f)`) stores
   them too, and the instance of `next` at `Map` reaches its inner `next`
   through the field — no handler copies a pointer by hand. The machine
   sees a value word and `n` pointer words; the pairing of pointer to
   signature is Rust's, at compile time.

5. **The one specialization is `Monomorphize`.** A variable declared over a
   finite member set is filled by each member, the handler compiled once
   per member, and in that body `T` is the Rust type itself — `acc + x`
   over `T: Add`, no materialization — and `T::call` for an `Instance`
   bound on a member is a direct call to that member's handler. To the
   machine these are more instances, not generics.

### The iterator on this

```rust
extern_signature! { ns: "iter",
    fn next<I, T, E, Rt>(it: &mut I) -> Option<T>
    where I: Var<kind::Type>, T: Var<kind::Type>, E: Var<kind::Effect>, Rt: Runtime; }

#[extern_fn(instance_of = sig::next, effect = pure)]            // next @ Range
fn next_range(it: &mut Range) -> Option<i64> { … }

#[derive(ExternType)] #[extern_type(name = "Map")]
pub struct Map<I, T, U, E, Rt>(I, Closure<(T,), U, E, Rt>)
where I: Var<kind::Type> + Instance<sig::next<I, T, E, Rt>>, …;

#[extern_fn(effect = pure)]
fn map<I, T, U, E, Rt>(it: I, f: Closure<(T,), U, E, Rt>) -> Map<I, T, U, E, Rt>
where I: Instance<sig::next<I, T, E, Rt>> { Map(it, f) }

#[extern_fn(instance_of = sig::next, effect = E)]               // next @ Map<I, T, U, E>
fn next_map<I, T, U, E, Rt>(it: &mut Map<I, T, U, E, Rt>) -> Option<U>
where I: Instance<sig::next<I, T, E, Rt>> {
    let Map(inner, f) = it;
    let x = I::call(inner, rt, frame, ())?;                     // the pointer beside the value
    Some(f.call_now(rt, frame, (x,)))
}

#[extern_fn(effect = E)]
fn sum<I, T, E, Rt>(it: I) -> T
where I: Instance<sig::next<I, T, E, Rt>>, T: Monomorphize<(i64, f64)> + Add<Output = T> + Default {
    let mut it = it; let mut acc = T::default();
    while let Some(x) = I::call(&mut it, rt, frame, ()) { acc = acc + x; }
    acc
}
```

`range(0, n) | map(|x| x * 2) | filter(|x| x > 3) | sum()`: the checker
resolves `next@Range` at `map` (so the lambda is `i64 -> i64`),
`next@Map<Range, …>` at `filter`, `next@Filter<…>` at `sum` and the member
`sum@i64`; lowering passes one pointer at each of the three calls; per
element the machine runs three pointer calls and two closure calls, no
vtable, no box, no type. One `next` signature, a struct and an instance per
adaptor, an instance per source, a handler per consumer — about fifty-five
declarations, no length bound, and no `Iter` type: an iterator is a type
with an instance of `next`.

## What it costs

- A word form for a function pointer in the machine's value (`FnValue` is
  a heap `Arc<dyn Callable>` today; an instance is a bare pointer).
- The `Instance<S>` trait, the carrier that bundles a value with its
  pointers, and the macro output: an impl per declared instance, the
  bundle impls that select a slot by signature, the hidden arguments per
  bound.
- Pattern instances in `Externs::combine` (a generalization of
  `at_a_member`) and a deferred constraint in the solver that fires when
  the bounded variable becomes ground and unifies the signature's other
  variables — the lambda in `map(range(0, n), |x| x + 1)` is typed only
  after `next<Range, T>` yields `T`.
- A struct that stores a bounded `I` grows by one word per bound.
- The effect of a pattern instance (`next_map`'s `E`) is the join of the
  inner instance's and the closure's; the solver's effect rule decides how
  that join is written (open, below).

## Rejected

- **A marker bound with no handle** (`T: HasInstance<sig>`): a requirement
  the body cannot use. Deleted.
- **A dictionary the call site owns**: the consumer's site resolving the
  whole instance tree and handing the body a tree to walk. It reconstructs
  in the machine what the values already carry, and it leaves type
  structure on the machine's side of the boundary.
- **A run-time table** (`TypeId → handler` per signature, read by the
  runtime's value): a lookup where the compiler already knows the answer;
  the pointer is passed, not found.
- **A type-level list in the payload** (RFC-0065's `Iter<(T, (U, ()))>` and
  `Len`): generics inside the machine.
- **A `Box<dyn>` per stage** (master's iterator): the pointer sits in a
  vtable behind a box, and the checker's knowledge is thrown away at the
  boundary.
- **The requirement as an explicit parameter** (`next: Instance<…>` in the
  signature): the trait form with one bound, less the ability to require
  several instances of one variable without a parameter each, and a
  pointer field every adaptor copies by hand.

## Where this is hard

- Inference order: a requirement fires only when its variable is ground;
  a variable that never grounds leaves the requirement unfired, and the
  check must report which one.
- Bundles: the pairing of slot to signature is done by Rust's trait
  selection over the bundle's tuple; two bounds naming the same signature
  at different variables (`Instance<eq<A>> + Instance<eq<B>>`) must select
  different slots — the signature's full type is the key, not its name.
- `Map<Map<Range>>` resolves by structural recursion; an instance whose
  bound unifies with its own pattern is a cycle and is refused at combine.
- The closure inside `Map` is a value filled at `map`'s site and called
  from `next_map` at the carrier; the instance pointer is a fact of the
  types filled at the same site. Both ride in the value; only one of them
  is a closure.

## Order of work

1. Type-helper pass batches A–C (done: A `69cbd6e7`, B `3c2b4dec`; C in
   review) — `Closure`, `Loan`, the kinds, one crossing trait.
2. The function-pointer word in the machine's value; `Instance<S>`, the
   bundle carrier, the macro's impls and hidden arguments; `clone<T>` as
   the first bound — a handler that calls what it requires, at `T =
   Owned<Rt>` and at a `Monomorphize` member.
3. Pattern instances in combine and the solver's deferred requirement;
   `Map<I, …>`'s `next` as the first pattern instance.
4. The iterator on `next`: sources, adaptors, consumers; the dyn chain
   removed; numbers against master under `benches/README.md`'s protocol.
5. Container handlers requiring `eq`/`ord`/`hash` of their element.
