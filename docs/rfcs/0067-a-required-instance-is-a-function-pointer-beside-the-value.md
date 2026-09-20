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
   InstanceOf<next<I, T, E, Rt>>`. What the instance's other variables are
   — the element `T` of an `I` — is read off the instance's type by
   unification, as a call reads it today. No associated type and no
   marker: the signature's own variables carry the relation, and the body
   calls `I::call(&it, rt, frame, args)`.

2. **An instance may stand at a pattern.** `fn next_map<I, T, U, E, Rt>(it:
   &mut Map<I, T, U, E, Rt>) -> Option<U> where I: InstanceOf<next<I, T, E,
   Rt>>` is the instance of `next` at the pattern `Map<I, T, U, E>`. Two
   instances whose patterns unify are refused at combine, as two concrete
   instances at one type are today. A bound is therefore a predicate — an
   instance whose pattern unifies with the ground type exists and its own
   bounds hold — decided on ground types by structural recursion, which
   ends because types are finite trees. This lives in the checker only.

3. **The compiler passes the pointer.** At a call whose declaration bounds
   a variable by `InstanceOf<S<…>>`, lowering resolves `S` at the ground type
   and passes the glue entry of that instance as one more argument: a
   word holding a function pointer of the framework's uniform value ABI.
   Nothing is looked up at run time; the pointer is right for the same
   reason a `Monomorphize` member is right — the checker chose it.

4. **The carrier of a bounded variable is the value with its pointers
   beside it.** In the uniform instantiation a type variable is filled by
   the runtime's carrier; a variable with `n` `InstanceOf` bounds is filled by
   that carrier and `n` pointers, assembled by the glue from the arguments
   of Decision 3. `InstanceOf<S>::call` reads the pointer for `S` out of
   `&self` and calls it. Because the pointers ride with the value, a
   handler that stores `it: I` in a value it builds (`Map(it, f)`) stores
   them too, and the instance of `next` at `Map` reaches its inner `next`
   through the field — no handler copies a pointer by hand. The machine
   sees a value word and `n` pointer words; the pairing of pointer to
   signature is Rust's, at compile time.

5. **The one specialization is `Monomorphize`.** A variable declared over a
   finite member set is filled by each member, the handler compiled once
   per member, and in that body `T` is the Rust type itself — `acc + x`
   over `T: Add`, no materialization — and `T::call` for an `InstanceOf`
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
where I: Var<kind::Type> + InstanceOf<sig::next<I, T, E, Rt>>, …;

#[extern_fn(effect = pure)]
fn map<I, T, U, E, Rt>(it: I, f: Closure<(T,), U, E, Rt>) -> Map<I, T, U, E, Rt>
where I: InstanceOf<sig::next<I, T, E, Rt>> { Map(it, f) }

#[extern_fn(instance_of = sig::next, effect = E)]               // next @ Map<I, T, U, E>
fn next_map<I, T, U, E, Rt>(it: &mut Map<I, T, U, E, Rt>) -> Option<U>
where I: InstanceOf<sig::next<I, T, E, Rt>> {
    let Map(inner, f) = it;
    let x = I::call(inner, rt, frame, ())?;                     // the pointer beside the value
    Some(f.call_now(rt, frame, (x,)))
}

#[extern_fn(effect = E)]
fn sum<I, T, E, Rt>(it: I) -> T
where I: InstanceOf<sig::next<I, T, E, Rt>>, T: Monomorphize<(i64, f64)> + Add<Output = T> + Default {
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
- The `InstanceOf<S>` trait, the carrier that bundles a value with its
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
  at different variables (`InstanceOf<eq<A>> + InstanceOf<eq<B>>`) must select
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
2. The function-pointer word in the machine's value; `InstanceOf<S>`, the
   bundle carrier, the macro's impls and hidden arguments; `clone<T>` as
   the first bound — a handler that calls what it requires, at `T =
   Owned<Rt>` and at a `Monomorphize` member.
3. Pattern instances in combine and the solver's deferred requirement;
   `Map<I, …>`'s `next` as the first pattern instance.
4. The iterator on `next`: sources, adaptors, consumers; the dyn chain
   removed; numbers against master under `benches/README.md`'s protocol.
5. Container handlers requiring `eq`/`ord`/`hash` of their element.

## Consequences

Step 2 is built: an instance has an entry, a declaration records what it
requires, the checker's bound is met with the instance types, and the
pointer is placed in the call site's table. Nothing was added to the IR, to
typeck's call resolution, or to lowering.

### The requirement is a bound of the declaration

A handler writes the requirement at its own variables:

```rust
#[extern_fn(effect = pure)]
fn same<T, Rt>(rt: &Rt, frame: &mut Rt::Frame<'_>, a: T, b: T) -> bool
where
    T: Var<kind::Type> + Carrier<Rt> + InstanceOf<sig::eq<T, Rt>, Rt>,
    Rt: Runtime,
{
    T::call(&a, rt, frame, (&b,))
}
```

`extern_signature!` gives each marker the signature's own declared
variables, with the runtime appended as `__Rt` where the signature declares
none, and every parameter defaulted so that `instance_of = sig::eq` still
resolves. `#[extern_fn]` reads each variable's `InstanceOf<…>` bounds in
order and records them on `FnDecl::requires` as `Requirement { var,
signature }`; two bounds on one variable naming one signature are refused
at the macro, since a type has at most one instance of a signature
(RFC-0019) and the second could only select the first's entry.

`Carrier<Rt>` stands in the bound beside `InstanceOf` and is not redundant.
What a call takes and gives is the signature's, read through `<sig::eq<T,
Rt> as Signature<Rt>>`, and that projection normalizes only where the
impl applies — which is where `T` is a carrier. Without it the handler
would have to restate `Rest` and `Ret`, which is the restatement the whole
decision exists to avoid.

### The check is the `OneOf` meet RFC-0019 defined

`Externs::combine` meets each required variable's bound with
`OneOf(every type with an instance of the signature)`. Typeck's existing
`OneOf` check refuses a call outside it with the sentence it already has:
no new error kind, no new decision, no new pass. A requirement naming a
signature no registry declares is `CombineError::RequiredSignatureUnknown`.

### The pointer lives in the site table

`ArgAt` gains one field, `instances: &dyn InstanceEntries<Rt>`, and so
gains the runtime as a type parameter. `Externs::combine` builds the
`InstanceTable` behind it — per signature, the ground type each instance
stands at and that instance's entry — and `PrepareCtx` carries a reference
to it into `arg_sites`. A parameter whose type is a bounded variable is
`ByBound<C>`, whose `Sited::Site` is `[Entry<Rt>; n]`, one per bound in
bound order; a `()`-sited parameter ignores the new field entirely. The
MIR does not change, the argument run does not widen, and no value flows:
the fact is static at the site, which is why it is neither a MIR value nor
an argument word.

### The carrier is a struct the macro writes

Per bounded variable, `#[extern_fn]` writes a struct holding `Owned<Rt>`
and one `Entry<Rt>` field per bound, with `Var<kind::Type>`, `TyArg` (the
variable's own acvus type), `Carrier<Rt>`, and one `impl InstanceOf<S, Rt>`
per field. `Arg::take` builds it from the argument's value and the site's
entries. There is no `frunk` index and no bundle: the fields are named
where they are written, and each impl names its own.

**A bounded variable stands only where a parameter takes one whole value.**
A carrier is the value plus `n` pointers, and the machine's storage holds
the value alone, so there is nothing behind `&T` shaped like a carrier and
no room for one in a `Vec<T>`'s buffer — `Vec<T>` crosses as
`Vec<Owned<Rt>>`, and reading it back as a `Vec<carrier>` would reinterpret
a one-word element as an `n + 1`-word one. `#[extern_fn]` refuses every
other position with that sentence. The consequence is that a container
handler cannot require a signature of its element while the element is read
through the container: `index_of(xs: &Vec<T>, x: &T)` is not writable, and
nor is any by-value form of it. Requiring an instance of a container's
element waits for a crossing that carries the site, which is a decision for
step 5 and not this one.

### One frame convention

`Entry<Rt>`, `AtEntry` and the generated entry `fn`s take
`frame: &mut Rt::Frame<'_>`; a handler already names the frame that way,
so an entry passes its own straight through. `Handler`'s closure keeps the
window by value, and the one owned window → `&mut` in the crate is the glue
closure `#[extern_fn]` writes, once per declaration.

Taking the window by borrow in that closure as well was built first and
withdrawn on the measurement. `Handler::call` then has to give the moved
window a stack slot to lend it, the address escapes into the closure, and
`benches/asm_probe.rs` counted sixteen `Op::run` bodies that ended in a
cleanup landing pad instead of the tail jump — six `CallExtern1` and two
`CallWindow` over `flatten`, four more of the same pair over `vec_deque`,
two `CallExtern2` over `skip`, and four `CallExtern2` over `filter` and
`map` that kept the jump and gained a `ret` beside it. With the convention
above the figure is unmoved: 2741 tail jumps, 46 chain ends, 28 listed
stack addresses. No `Runtime::reborrow` exists and none is added; one
method on every host buys one word.

### The customer

`acvus-extern/tests/decl.rs` declares `same<T, Rt>` above and runs it
through `Tiny` at `i64`, both instances of `t::eq` being in the combined
registry, and asserts that `same`'s variable is bounded by
`OneOf([i64, Point])` — so a call at `String` is refused by the checker's
own sentence.

It does not run at `Point`, and the reason is `Point`'s and not the
requirement's: an object converts at the boundary, so a `&Point` argument
names no storage shaped like a `Point`, and `t::eq @ Point` is unreachable
through the values ABI whether a call site or a requirement reaches for it.
`an_instance_whose_parameter_converts_is_unreachable_through_the_values_abi`
pins that on the entry itself. An instance of a signature whose parameters
are `&T` is therefore usable as a requirement only at types that keep
storage of their own — the scalars, `String`, and the extension types.

### What waits

- **A requirement of a container's element**, per the paragraph above.
- **A `Monomorphize` member's impls**, refused as of step 3's second half
  and recorded under "What waits" there with the measurement.

### Step 3: a signature is a shape

`Signature` is the shape of a call and carries no receiver concept. Its
`Recv<'a>` is `&'a This`, `&'a mut This` or `This`, one per mode the
declaration wrote, and `InstanceOf::call` takes the receiver that way; a
handler that lends `&it` where the signature takes `&mut I` is refused at
its own `I::call`, which
`acvus-extern-macro/tests/compile_fail/instance_wrong_mode.rs` executes.
`extern_signature!` writes the impl for every signature whose first
parameter stands at its first type variable in any of the three modes and
whose later parameters and result are each one whole value — a parameter or
result mentioning a variable is fine, so `next<I, T, E, Rt>(it: &mut I) ->
Option<T>` has one. A later parameter that is a `str` view, a projection,
or a borrow of anything but that first variable gets none: a requiring
handler holds a carrier and its entries, and there is no third thing it
could lend.

One trait with a receiver projection was chosen over three traits — a
`Signature`, a `SignatureMut` and a `SignatureOwned`. With three, a
requiring handler's bound has to say which one it means, and that is the
signature's own shape restated at the requirement, which is what Decision 1
exists to remove.

`acvus-extern/tests/decl.rs` declares `t::step<I>(it: &mut I) -> i64` with
an instance at `i64` and a handler `drive<I, Rt>` requiring it;
`a_required_instance_whose_first_parameter_is_mut_runs_through_the_entry`
runs the entry and reads the bumped place back.

### Step 3, second half: an entry is a tree

A pattern instance stands, and what a site resolves for a requirement is a
tree rather than a pointer.

**A value carries no pointer.** An `ExternType` payload holding a bounded
variable's value holds `Held<Rt>` — one of the runtime's values and nothing
else. The acvus type parameter stays (`Doubled<I, Rt>` is still
`Doubled<Counter>` at the checker) but is phantom in the Rust payload, so
the handler that writes the payload and the instance that reads it back
name one Rust type. `#[derive(ExternType)]` refuses a payload naming a type
variable and says which held form to write;
`acvus-extern-macro/tests/compile_fail/carrier_in_a_payload.rs` executes
that refusal.

**An entry is the instance, with the instances its own bounds required
under it.** `EntryNode<Rt>` is `{ run: EntryFn<Rt>, children: Box<[Entry]> }`,
and `Entry<Rt>` is one word: the address of such a node. The nodes live in a
`NodeArena<Rt>` owned by the registry's `InstanceTable` and kept alive by
every site table that read one, so a node outlives every call the glue
holding its address can make. `Kind::Entry` carries the word, and
`Runtime::entry_value`/`entry_of` cross it.

`Externs::combine` records, per instance, where in that instance's pattern
each of its own bounds stands — a path of type-argument positions
(`BoundAt`). A bound standing somewhere no ground type can be walked to is
refused there (`CombineError::RequirementOffThePattern`). `InstanceTable::
entry_at` then walks a ground type by those paths, recursing once per
bound, and builds the node. The recursion is structural on the declaration
and the filling comes from the ground type, so no new solver task carries
it; what admits the call is still the `OneOf` meet of Decision 1.

**A carrier is rebuilt at each call, never stored.** In a requiring handler
`I` is (value, entries): `Carrier::of(value, bounds)`. `Signature::
call_entry` builds the run as receiver + rest + **the entry's own word**,
and the instance's glue reads its children off that trailing word. That is
one of the two sources of a pattern instance's inner entries; the other is
the site table, when the instance is called from a script site
(`advance(&mut d)` on a `Doubled<Counter>`). They are two mechanisms, one
per trait — `Sited` reads the site, `SitedAtEntry` reads the run — and the
glue that a declaration gets is chosen by which of the two it was built
for, so neither path pays for the other. The macro writes one body for both.

**The entries are the `where` clause's, not a parameter's.** A handler
whose parameter *holds* a bounded variable rather than taking it writes
only its acvus parameters; `#[extern_fn]` appends the binding carrying the
entries, named by the variable in lower case followed by `_bound`. Writing
that parameter by hand is refused: the bound already says the entries are
there, and restating it as an argument blurs the split between the bound
form here and step 5's parameter form (`Instance<S>`, a pointer passed as
its own argument for a container's element).

**Two traits, one direction.** `InstanceOf<S, Rt>` is the sync call,
`InstanceOfAsync<S, Rt>` the async one, and every sync instance is an async
instance through one blanket impl over a ready future. There is no impl the
other way: an async instance suspends and a sync caller has nowhere to
suspend to. A requiring handler's async body bounds its variable by
`InstanceOfAsync` and its `_now` twin by `InstanceOf`; either spelling
states the same requirement, so a site resolves the same entries for both.
`acvus-extern/tests/decl.rs`'s `drive_await` reaches `t::step`'s sync
instance at `i64` through the blanket impl.

An instance whose first parameter does not stand at the signature's first
variable, or whose later parameters are not each one whole value, gets no
`Signature` impl, and the missing impl is the refusal: `std::vec` and
`vec::filled` cannot be required as bounds.

#### Rejected

- **(a) Key the carrier by the signature, not by the declaration.**
  `Signature` grows a `Head`, and `acvus-extern` holds one `Bound<Rt,
  Heads>` whose instance impls are written once. Two declarations bounding
  a variable by one signature then name one type. Rejected because
  selecting an entry among several bounds becomes a position in `Heads` —
  the type-level index the per-declaration carrier removed.
- **(b) Resolve the inner entry at the site and carry nothing.** Rejected
  because it fails one level down: an instance reached through an entry
  from another instance's body has no site of its own to read. (c) is (b)
  with that level supplied: the site resolves the whole tree, and the run
  carries the branch each level needs.

#### What waits

**A variable that is both `Monomorphize<(A, B)>` and bounded** is refused.
The two fill it with different Rust types in one monomorphization:
`Monomorphize` compiles the body once per member with the variable standing
for that member, while a requirement fills it with the carrier, which has
none of the member's methods. Measured — `ByBound<i64>` demands `i64:
Carrier<Rt>` — and pinned by
`acvus-extern-macro/tests/compile_fail/monomorphized_and_bounded.rs`. A
carrier that derefs to its member would answer it, and that is step 5's
parameter form rather than this step's.

**A refusal does not name the signature it required.** `combine` meets a
requirement into `TyVarBound::OneOf`, which carries the patterns an
instance stands at but not the name of the signature, so the checker's
`TypeOutOfBound` says *"type i64 is outside the declared bound one of
Counter, Doubled<'0>"* — the ground type and every pattern, without the
signature. Carrying the name would widen `TyVarBound` or add `requires` to
`FnKind::Extern`, both shared across `acvus-mir`.

**The inner bound is checked at the constructor, not at the pattern.** A
value of `Doubled<X>` can only come from a handler that itself required `X:
InstanceOf<advance>`, so `entry_at` is total on what the checker admits.
That is a property of the declarations, not a guarantee: a registry that
built such a value without the bound would reach `entry_at`'s panic instead
of a refusal.

### The entry, unchanged from the first half

```rust
pub type Entry<Rt> = for<'a, 'w> unsafe fn(
    &'a Rt,
    &'a mut <Rt as Runtime>::Frame<'w>,
    &'a [<Rt as Runtime>::Value],
    &'a mut [<Rt as Runtime>::Value],
);
```

`#[extern_fn]` writes one `fn` item per instance beside the Rust body and
names it through a zero-sized type implementing `AtEntry`, which the glue
carries as a type parameter rather than a field.

An entry is written for an instance of a shared signature and for nothing
else. An instance still has none if it holds a `#[state]` value, takes a
projection parameter, or is reached only by the glue that suspends the
caller (`heavy`, and an `async fn` without a `sync =` companion). Of the
standard registries' synchronous instances, **none** is in that set:
`acvus-interpreter-test/tests/instance_entry.rs` asserts it over the
fifteen declared signatures, so the refusal list is empty and the refusal
is a path no registered instance reaches today. An intrinsic such as
`StringClone` is not a registry instance, so the `OneOf` meet excludes it
before the entry is ever asked for.

`acvus-interpreter`'s `Kind::Entry` and the `Runtime` contract's
`entry_value`/`entry_of` carry an entry as one of the runtime's values.
Nothing requires them yet: a requirement resolves at the site table, in
Rust, and the entry never becomes a machine value on that path.

### What it cost

An entry is a second caller of the Rust body it names, so that body is no
longer inlined into the operation holding it, and the operation ends in
`call`/`ret` instead of the tail jump `benches/asm_probe.rs` asserts.
Measured on the release machine: written for every declaration, 89
operations lose the tail jump — 20 `CallExtern2`, 10 `CallWindow`, 3
`CallExtern1`, 2 `CallExtern3` among them. Written for instances alone, the
figure is unchanged. That is why the entry is per instance and not per
declaration, and it is the ceiling on how far entries may spread: a body
that both an operation and an entry reach is a body the operation calls.

The same second caller doubles an author's parameter refusal, because the
entry `fn` restates the declaration's parameter markers and the trait
obligation then fails at two spans. Per instance, no `compile_fail`
expectation moves.
