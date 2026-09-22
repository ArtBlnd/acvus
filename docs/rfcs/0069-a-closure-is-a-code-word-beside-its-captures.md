Ru# RFC-0069: A closure is a code word beside its captures

Status: Draft
Extends: RFC-0018, RFC-0044, RFC-0050, RFC-0052, RFC-0067

## Problem

A closure value is `FnValue`:

```rust
pub struct FnValue {
    pub shared: Arc<InterpreterContext>,
    pub page: Arc<dyn RuntimeContext>,
    pub entry: Arc<dyn Callable>,
    pub captures: Arc<[Owned<AcvusRuntime>]>,
}
```

Two of the four fields are not the closure's. `shared` and `page` are the
run the closure is called in, which every caller already holds: a `Machine`
has both, and a handler has them through `Ctx::rt`. The third, `entry`, is a
prepared `Body` or `Expr` the module owns for as long as it runs, reached
through a vtable. The fourth is the only data of the closure, and it is a
second allocation.

`MakeClosure::run` therefore costs two allocations (the capture slice, the
`Large` box of the `FnValue`) and three atomic increments, for a closure of
no captures as for one of ten, and releasing it costs four atomic
decrements. A call (`Closure::call_now` → `Runtime::call_now` →
`Callable::call_in_window`) reads the record out of the box, calls through
the vtable, binds a window, writes one fresh `Value::reference` per capture
into the callee's registers, builds a `Machine`, enters the dispatch loop,
and sweeps the frame. Only a body that `prepare` folded to one chain
(`Code::Expr`, RFC-0044 stage 4) skips the frame and the machine.

Measured on this tree (`benches/accum.rs`, ns per element, n = 1 000 000):
`range | sum` 1.1, `map id | sum` 3.0, `map add | sum` 5.5, `map add fil |
sum` 14.4. A stage costs one to two nanoseconds; the rest of each row is
closure entry and exit. `benches/logs.rs` at n = 10 000: the same matcher
reached through a closure is 2033 ns per line, through a synchronous extern
62, in Rust 51. No change to the iterator surface reaches this cost.

## Decision

### D1. The run is the caller's

A closure value holds neither the interpreter nor the page. A call takes
them from where it is made: the `Machine` that runs `CallIndirect`, or the
`Ctx` a handler was handed. A closure made in one run and called in another
is not a case: a closure value does not outlive the run that made it
(RFC-0014), and a closure a `Space` holds is re-entered through that space's
own runtime.

### D2. The code is a word

What a closure runs is a `Code` the module's `Prepared` owns. A closure
names it by address, one word, fixed by `prepare` where `MakeClosure` is
built; nothing counts references to it. The head of a `Code` is the
function that enters it, chosen once where the `Code` is made (a framed
body, a chain), so a call is two loads and an indirect call: no `dyn`, and
no match on the shape at the call. This is RFC-0067's instance word, for a
body the interpreter prepared rather than one Rust compiled.

### D3. A closure of no captures is its code word

Its `Value` is inline: the kind says closure, the word is the code. Making
it allocates nothing, copying it is a register copy, and it is not among the
frame's `Large` registers, so no sweep releases it.

### D4. Captures are one block, read in place

A closure that captures holds one allocation: the code word, the capture
count, and the captures behind them. The body reads a capture through a
reference, as RFC-0018 says, and that reference is to the capture where it
lies in the block: a call puts the block's address in the callee's frame
once, and a capture read is a load at an offset `prepare` fixed. No
`Value::reference` is written per capture per call.

### D5. One entry, generic in how arguments arrive

`Callable`'s three entries (`call_in_window`, `call_in`, `start`) are one
body of work reached three ways, as `Handler`'s `call0..4`, `call_pair1..4`
and `call_out0..4` are. They become one entry whose argument source and
result form are type parameters the caller names, which is the shape the
Runtime contract takes for extern handlers; a closure call and an extern
call are then the same kind of operation to `prepare`, and the loop
operation that inlines a stage's `next` can hold a closure's entry the same
way.

## What it costs

- `MakeClosure` needs the capture block's layout written once, by hand or by
  a small typed wrapper: one `unsafe` site in the interpreter, none in
  `acvus-extern` or the extensions.
- A `Code` must not move or die while a closure word names it. The module's
  `Prepared` already outlives every frame of its run; the obligation is
  stated where the word is minted and checked in debug builds.
- The capture-read path changes in `prepare` (an offset from a block
  address rather than a register holding a reference). Loans analysis reads
  captures as references today and keeps doing so; only the lowering of the
  read changes.

## Rejected

- **Keeping the four `Arc`s and making the call cheaper.** The per-call work
  (`bind_captures`, `Machine::new`, the vtable) is separable from the value's
  shape, but the value's shape is what makes a capture-free lambda cost two
  allocations, and it is what every copy and release of a closure pays.
- **One box for every closure.** A closure of no captures has nothing to
  box. The word form is what lets `map(|x| -> x + 1)` hold an instance word
  and a closure word side by side with no allocation between them.

## Order of work

The closure comes before the Runtime contract. The contract's closure half
(`call_now`, `call_0`, `call_1`, `call_n`, `call_is_sync`, `CallFuture`) is
written against what a closure value is; reshaping the contract first would
write that half once for `FnValue` and once more for the word.

1. D1 and D2: the run from the caller, the code as a word. `Callable`'s
   three entries stay three for now, reached from the code word instead of
   through an `Arc<dyn>`.
2. D3, then D4.
3. The Runtime contract: one generic entry for extern handlers
   (`Handler::call` with `Arity` and `RetForm`), and D5 with it, so a closure
   call and an extern call are one kind of operation.
4. The loop operation and the micro-operation machine build on that entry.

Each step is measured alone: `accum`'s closure rows, `logs`'s `closure`
case, `asm_probe`.

## What waits

- How a closure word is printed and compared (`Debug`, equality of
  closures) once there is no `Arc` address to show.
- Whether a suspending closure (`Callable::start`, a `Store` per call) takes
  the same entry or keeps a second one for the future it returns.
