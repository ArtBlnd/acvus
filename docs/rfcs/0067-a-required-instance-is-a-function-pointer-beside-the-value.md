# RFC-0067: A required instance is a function pointer beside the value

Status: Accepted — built for the iterator spike (`acvus-interpreter-test/
tests/iter_next.rs`); generic container handlers (`eq`, `ord`, `hash` of an
element) are the next customer.

## The principle

**The machine holds no generics.** Every type at a call is ground; an
instance of a shared signature is a value — the address of one concrete
handler's mono glue — and no function type crosses the extern boundary as a
type-level list. There is no position in the machine where a type is
computed, and that absence is why the framework needs no higher-kinded
types: whatever a type variable ranges over is settled by the checker and
arrives at the machine as a word.

The runtime shape this leaves is C's: a value with a function pointer beside
it. The difference from C is the compiler standing next to it — the checker
picks the pointer, the loans analysis governs the borrow, the effect system
says whether the call may suspend.

## Problem

A handler generic over a type variable needs the instance of a shared
signature at the type the site resolved. RFC-0019 gave the language shared
signatures — one instance per concrete type, resolved at a call by the
call's ground type — but a Rust handler generic over `I` had no way to call
`next` on its `I`: a marker bound narrowed admission and gave the body
nothing to call.

The iterator paid for the gap. Without a way to say "`I` is a type with a
`next`" and call it, every design put the whole pipeline inside one
extension type: a `Box<dyn>` per stage, or one Rust type per pipeline
length with a bound of eight and 292 registry names (RFC-0065, built and
rejected).

The first answer to that was a tree of entries carried in the value (the
first entry of "Rejected"). It reconstructed in Rust structs what the frame
and the call site already held, and it paid for it: on the spike at
`n = 100 000` it read 200.4 / 612.7 / 892.6 / 2019.3 µs on `range | sum`,
`map id | sum`, `map add | sum` and `map add fil | sum`, against the `dyn`
chain's 164.3 / 340.1 / 577.4 / 1387.1 µs on the same rows — 1.2× to 1.8×
the design it was meant to replace.

## Decision

### An instance is one value

```rust
pub struct Instance<S, I, Rt, T = Now>
where
    S: Signature<Rt>,
    Rt: Runtime,
{
    value: Rt::Value,
    at: PhantomData<fn() -> (S, I, T)>,
}
```

One of the runtime's values and nothing else — `Instance::at` reads a
`const ONE_VALUE` assert that says so. Its word is the address of the mono
glue `#[extern_fn]` wrote beside the handler of the instance of `S` at the
type the site resolved; `Kind::Instance` and `Kind::InstanceAwait` carry it
in the interpreter's value, and `Runtime::instance_value` /
`instance_run` cross it. `I` is the requiring handler's own type variable,
so `call` compiles only with the receiver the instance was declared for, and
`T` (`Now` or `Later`) is the task the requirement calls at.

**One value per signature, not a bundle.** A handler requiring two
signatures takes two `Instance` parameters and two words. A bundle — one
carrier holding a value and `n` pointers — puts a subtyping wall in the way:
a handler requiring `A + B` that calls one requiring only `A` would have to
coerce a two-slot bundle to a one-slot one, which is a position in a
type-level list and the generics the principle forbids. With one value per
signature, `A + B` → `A` passes the word and needs nothing.

**Not a stamp in the value.** An instance is not a vtable pointer in the
`Large` header, for three reasons. Scalars have no header, so an `i64`
receiver has nowhere to put one. An instance is chosen by the *acvus* type,
and several acvus types share one Rust payload, so the Rust value cannot
name which. And a `&'static` table fixed in a type cannot take another
crate's registration, while registries combine at run time.

### The requirement is declared by taking the parameter

```rust
#[extern_fn(effect = E)]
fn nsum<I, E, Rt>(ctx: &mut Ctx<'_, Rt>, it: I, next: Instance<sig::next<I, i64, E, Rt>, I, Rt>) -> i64
where
    I: Var<kind::Type> + DerefMut<Target = Rt::Value>, …
```

`#[extern_fn]` reads the parameter (`required_of`), checks that the variable
it stands at is one of the declaration's own `Var<kind::Type>` parameters,
and records `Requirement { var, signature, calls }` on `FnDecl::requires`,
where `calls` is `T`'s task met with the body's. `Externs::combine` meets
that variable's bound with the `OneOf` of every type holding an instance of
the signature at or below that task, and typeck's existing `OneOf` check
refuses a call outside it with the sentence it already has — no new error
kind and no new pass.

### The word lives in the site table

`prepare` resolves the instance once, per call site, through
`InstanceEntries::instance_at(signature, ty)`: a flat lookup in the
registry's `InstanceTable` over the settled type of the argument the
requirement's variable stands at. The parameter's crossing is
`Required<S, I, T, AT>`, whose `Sited::Site` is the `Instance` itself and
whose `ARGUMENTS` is `0`; `Arg::take` copies it out of the site table.

Not a window slot: a window is planned from the MIR call's own arguments
(`prepare::window`, `window_args`), and a constant exists only for a MIR
`InstKind::Const` the body already holds (`hoist_konsts`) — an instance word
has no `ValueId`, so putting it in the window would mean a new kind of MIR
value carried through lowering. The site table already exists and the
requirement costs zero words of ABI, no `Mov` before the call, and nothing
rebuilt per call.

### The call

```rust
pub unsafe fn call<'r>(self, ctx: &mut Ctx<'_, Rt>, recv: &mut I, rest: S::Rest<'r>) -> S::Ret
where
    I: DerefMut<Target = Rt::Value>,
{
    ctx.name_receiver(&mut *recv);
    unsafe { S::call_now(self.value, ctx, rest) }
}
```

`ctx` and the arguments cross as they are typed — a concrete parameter or
result is itself, `Option<i64>` in registers, not an erased run. The
receiver does not cross: `ctx` carries a `recv: *mut Rt::Value`,
`name_receiver` writes it, and the mono glue on the far side takes it out of
`ctx` and materializes the handler's literal `&mut NRange`, then reads its
fields natively. The receiver word is a raw pointer and is never tested
outside the `debug_assert!` in `Ctx::receiver`, which names the defect
("a mono glue read a receiver that no `Instance::call` named") rather than
branching on it in release.

The indirect call is the last expression of `call`. Nothing follows it: no
write-back, no drop, no result marshalling, so the caller's tail position is
the glue's.

### A payload lays the value and its instance side by side

An adaptor stores the inner iterator and the instance of `next` at it as two
fields, laid once at construction:

```rust
pub struct NMapBody<Rt> where Rt: Runtime {
    inner: Owned<Rt>,
    next: InnerNext<Rt>,
    f: Closure<(i64,), i64, Opaque, Rt>,
}
```

`next_nmap` calls `it.0.next.call(ctx, &mut it.0.inner, ())`. Nothing is
walked, nothing is rebuilt, and what an instance itself requires is a field
of its own payload — which is why `instance_at` is a lookup and not a
recursion. A payload names no type or effect parameter, so it holds the
instance at the erased ones (`Owned<Rt>`, `Opaque`), the same way it already
holds a `Closure`.

### `into_async` and the `Later` form

`Instance::into_async` retypes `Now` to `Later`, one way. The task branch
lives in the `Later` form alone: `call_await` is a method of
`Instance<S, I, Rt, Later>` and returns `BoxFuture<'a, S::Ret>` through
`Signature::call_later`. There is nothing the other way — an async instance
suspends and a sync caller has nowhere to suspend to.

### `Ctx` is owned by the machine

A handler is called with `ctx: &mut Ctx<'_, Rt>` where it was called with
`rt: &Rt` and `frame: &mut Rt::Frame<'_>`. `Ctx { rt, frame, recv }` is
owned where the frame state already lived: `Machine { ctx, .. }` lends
`&mut self.ctx` beside `self.regs`, exactly as it lent `above`, and the
rooted store an async glue holds owns one (`Runtime::ctx_of` in place of
`frame_of`, `Rooted<'a>` a GAT). The interpreter's `Frame<'a>` is the state
itself, so `&mut &mut FrameState` is gone.

Built in the call op's frame instead, the pair takes a stack slot and its
address escapes into the handler: the op's frame can no longer be elided and
the op returns where it should tail-jump. Measured — 151 extern-call op
bodies lost their tail jump (`CallWindow` 70, `CallExtern1` 45,
`CallExtern2` 34, `CallExtern3` 2). Owned by the machine, the pointer an op
passes already exists, and `asm_probe` reads 4695 / 49 / 19 / 16 against
4686 / 49 / 28 / 16: nine bodies stopped holding a stack address across a
listed callee, none lost a tail jump.

A handler still writing `rt: &Rt` was parsed as a value parameter of the
runtime's type; the macro refuses it by name (`runtime_as_a_parameter`).

### The iterator on this

```rust
extern_signature! { ns: "nit", effect = E,
    fn next<I, T, E, Rt>(it: &mut I) -> Option<T>
    where I: Var<kind::Type>, T: Var<kind::Type>, E: Var<kind::Effect>, Rt: Runtime; }

#[extern_fn(instance_of = sig::next, effect = pure)]
fn next_nrange(it: &mut NRange) -> Option<i64> { … }

#[extern_fn(effect = pure)]
fn nmap<I, E, Rt>(ctx: &mut Ctx<'_, Rt>, it: I, f: Closure<(i64,), i64, E, Rt>,
                  next: Instance<sig::next<I, i64, E, Rt>, I, Rt>) -> NMap<I, E, Rt>
where I: Var<kind::Type> + Into<Owned<Rt>>, … { … }

#[extern_fn(instance_of = sig::next, effect = E)]
fn next_nmap<I, E, Rt>(ctx: &mut Ctx<'_, Rt>, it: &mut NMap<I, E, Rt>) -> Option<i64>
where … {
    let x = unsafe { it.0.next.call(ctx, &mut it.0.inner, ()) }?;
    Some(it.0.f.call_now(ctx, (x,)))
}
```

One `next` signature, a struct and an instance per adaptor, an instance per
source, a handler per consumer. No `Iter` type, no length bound, no `dyn`:
an iterator is a type with an instance of `next`, and per element the
machine runs one plain function pointer per stage plus the stage's closure.

## What it costs

**Per call.** One store (the receiver word into `ctx`), one load of the
instance word — from the payload field for an adaptor, from the site table
for a consumer — one indirect call, and one load on the far side to take the
receiver back out of `ctx`. Nothing is allocated, nothing is written back.

**The consumer's receiver bound.** A handler that takes its iterator by
value and drives it writes `I: Var<kind::Type> + DerefMut<Target =
Rt::Value>`, and `+ Into<Owned<Rt>>` where it stores it. The bound is the
price of `call` naming the receiver through `&mut I` rather than restating
the signature's mode.

**A `&str` receiver has no mono glue.** `#[extern_fn]` writes the glue only
where the receiver is one value; a `&str` parameter is the two-word pair a
view occupies, and `Ctx`'s receiver is one `*mut Rt::Value`. An instance
holding a `#[state]` value, taking a projection parameter, declared `heavy`,
or itself requiring an instance likewise has none, and cannot be required.

**The measured state.** On the spike, `nit::` (this design) against `iter::`
(the `Box<dyn>` chain), one binary, five reps:

| row | n | `iter::` med | `nit::` med | med % |
|---|---:|---:|---:|---:|
| `range \| sum` | 100 000 | 164.3 | 110.1 | −33.0 % |
| `range \| sum` | 1 000 000 | 1636.3 | 1090.1 | −33.4 % |
| `map id \| sum` | 100 000 | 340.1 | 274.5 | −19.3 % |
| `map id \| sum` | 1 000 000 | 3399.2 | 2752.6 | −19.0 % |
| `map add \| sum` | 100 000 | 577.4 | 569.6 | −1.4 % |
| `map add \| sum` | 1 000 000 | 5767.4 | 5714.2 | −0.9 % |
| `map add fil \| sum` | 100 000 | 1387.1 | 1379.8 | ranges overlap |
| `map add fil \| sum` | 1 000 000 | 13788.2 | 13992.2 | ranges overlap |

The last two rows are equal because the closure call, identical on both
arms, is what they spend their time in. Against the old entry design's own
rows at `n = 100 000`: 200.4 → 110.1, 612.7 → 274.5, 892.6 → 569.6,
2019.3 → 1379.8 µs.

`asm_probe` reads 4691 / 49 / 19 / 16 against the base's
4686 / 49 / 28 / 16: figures two to four unmoved, the first down by four
because four op bodies no longer exist, and no body lost a tail jump. Four
exclusion-list entries the probe carried are no longer needed — every
instance of `CallExtern1<__extern_fn_next>`, `CallWindow<__extern_fn_next>`,
`CallExtern2<__extern_fn_max_by_key>` and `CallExtern2<__extern_fn_min_by_key>`
now tail-jumps.

The registered surface is unmoved. The `Ctx` change measured alone against
the base: no row slower beyond `benches/README.md`'s ~7 % inter-build floor;
22 rows faster with disjoint ranges (`while let vec` −9.4 %, `for slice add`
−9.2 %, `sync ext` −8.1 %, `while let map` −6.3 %), seven slower with
disjoint ranges and all under the floor (largest `cut owned` at 1 000 000
+3.1 % and `bf table` at 5 000 000 +3.1 %). Suite 2679 → 2666 with the cut,
2670 with the spike's own tests; `prepare_contract` 468 admitted / 468
prepared / 0 holes throughout; the differential corpus 1097 → 1093 → 1096
scripts with every counter but the removed and added files' own identical.

## Rejected

- **The entry tree and the carrier the macro wrote.** An `EntryNode` with
  children in a registry-owned `NodeArena`, a `Held`/`HeldMut` payload form,
  a `Bound`/`Bounds` slice, and a carrier struct per bounded variable. It
  rebuilt per element what the frame and the site already held, it took a
  `mem::take` and a write-back per element per stage on a value that is
  `Copy`, and the `I` it made was 24 bytes and growing with every further
  bound. Its rows are in "Problem" above.
- **A stamp in the value's spare bytes, or a bound bundle.** Both make a
  requirement a position in a type-level list, so `A + B` → `A` needs a
  coercion; and the stamp has nowhere to live on a scalar.
- **Arbitrary interfaces in the `Large` header's vtable.** Scalars have no
  header; an instance is chosen by the acvus type and several share one Rust
  payload; a `&'static` table cannot take another crate's registration.
- **`Ctx` built at the op.** Measured: 151 extern-call op bodies lose the
  tail jump, because the pair takes a stack slot in the op's frame and its
  address escapes into the handler.
- **The instance word as an ordinary argument of the call's window.** There
  is no `ValueId` for it; the site table holds it for zero ABI words.
- **A run-time table** (`TypeId → handler` per signature): a lookup where
  the compiler already knows the answer.
- **A `Box<dyn>` per stage**: the pointer sits in a vtable behind a box, and
  the checker's knowledge is thrown away at the boundary.
- **A type-level list in the payload** (RFC-0065's `Iter<(T, (U, ()))>` and
  `Len`): generics inside the machine.
- **A marker bound with no handle** (`T: HasInstance<sig>`): a requirement
  the body cannot use.
- **`Demoted` and a flat stage list** are not RFC-0067's question and are
  decided elsewhere.

## What waits

- **The async instance's customers.** `into_async` and `call_await` compile
  and are wired; no test drives an async instance through them yet.
- **`eq` at `T` and `step` at `&mut I`.** The shapes a by-value receiver and
  a `&mut` receiver take, beyond the spike's one shape.
- **The pairing golden.** A `compile_fail` case pinning that an `Instance`
  whose `I` is not the receiver's variable is refused at `call`.
- **The refusal for a declaration that is both an instance and requires
  one.** It gets no mono glue today, and the absence surfaces as a `prepare`
  panic ("has no mono glue") rather than a refusal at the macro or at
  combine. Two further `prepare` panics in `instance_at` re-check what
  `combine` proved and should be assertions of that proof, not run-time
  tests.
- **A two-word receiver.** An instance whose receiver is a `&str` view, and
  with it a `Ctx` receiver that is a run rather than one value.
- **A requirement of a container's element.** `index_of(xs: &Vec<T>, x: &T)`
  requiring `eq` at `T` needs a crossing that carries the site down to an
  element.
- **`LargeRef`, the allocator and the result channel.** Each is open and
  each is its own decision; none is settled by this one.
