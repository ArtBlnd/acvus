# Context

A context `@x` is the state a host keeps between runs of a program. This
document decides what a run is, what a context may hold, how a body and its
callees reach a context during a run, and how a space stores one.

## RFC-0014: The run is the unit; a context is a static variable that outlives it

Status: Accepted

1. A program always terminates. One execution is a run, and the run is the
   unit of everything the host sees: its context writes leave together when it
   ends, and a run that does not end leaves nothing. No state between a run's
   start and its end can be held, saved or resumed outside the run.
2. A context `@x` is a static variable: a place whose lifetime is not bound to
   the run, whose value the host keeps from one run to the next. How a body
   reads and writes it is RFC-0025.
3. A context holds only what the host can keep: data — a scalar, a container
   of data, or an extension type whose type arguments are data. The checker
   rejects a function, a handle, an order, a reference, an identity-carrying
   value, and any type holding one. Whether an extension type in a context can
   be written down is the host's declaration (a codec), not the checker's.
4. A context carries no policy: neither volatile nor read-only. A value a
   script only reads is a function argument.
5. An ExternFn cannot reach a context; a script reads one and passes the
   value.
6. A context is never a function namespace: `@f(x)` calls a value a context
   cannot hold, and an extern function is called by its name.
7. Progress that must survive a run is written to a context by the author,
   and the host runs the program again. Resumption is re-execution from the
   state the last completed run left.

Whether the host may re-run a failed run on its own judgment is decided by the
program's effect (RFC-0013).

**Why.** Dumping a run mid-way needs a serialized form of everything it holds
— Rust payloads, closures, calls in flight — and a version for all of it; the
run as the unit raises none of those questions. What the host keeps between
runs is data it already keeps, and a crash costs at most one run.

**Rejected.**
- Dump and resume of a run's state — needs serialized closures, in-flight
  calls and versions.
- A yield statement, a suspension point, or a host-input call that suspends —
  a call that waits on the host waits; a program that must hand control back
  ends and is run again.
- A volatile context — a second door to the host beside ExternFn.
- A context read/write declaration on ExternFns — the ordering it existed for
  is carried by RFC-0007 and the summary of RFC-0025.
- An identity-carrying value in a context — its source would outlive the
  program that made it.

## RFC-0025: A context is a variable of the body that touches it, and a call is bracketed by its callee's summary

Status: Accepted

1. **The variable.** A body that names `@x` owns `x` as a local variable for
   its run. Inside the body, `@x`, `@x.f`, `&@x`, `&mut @x`, `@x = v` and
   `let a = @x` are the local-variable rules of RFC-0018 on `x`, with nothing
   added: a read of a non-primitive moves out, a borrow is a `Ref` to the
   variable's storage, an assign re-initializes, and a live borrow excludes
   every other touch. There is no reference to a context in the IR; a `Ref`
   targets a variable, a parameter, or the storage a reference names.
2. **The page.** The page — the host's store of contexts — is touched at four
   places and nowhere else:
   - **Entry.** Every context the body names is fetched into its variable,
     `Fetch { dst, context }`, unless every path assigns it whole before
     touching it. A call whose summary names it touches it, and a callee's
     assignment is not the body's. Such a variable starts unset, as a `let`
     with no value does, and RFC-0018's rules hold it until the assignment.
   - **Exit.** On every return, each such variable is committed back:
     `Commit { context, value, wrote }`, where `value` is a `Take` of the
     variable. `wrote` holds when the body or a call since the fetch may
     have written it (rule 4's write sets); a commit that did not write
     hands the value back to the page and stores nothing.
   - **Around a call.** A call whose summary touches `@x` is bracketed:
     `Commit` of the variable before it, `Fetch` into the variable after it.
     A call whose summary does not touch `@x` touches nothing.
   - **Around a spawn.** The `Commit` goes before the `Spawn` and the `Fetch`
     after the `Eval` of its handle; between them the variable is moved out.

   A context the body does not name is not fetched; a context touched only by
   callees is fetched and committed by the callees. `Fetch` and `Commit` carry
   no path: the page stores whole values, and a field of a context is a field
   of the variable. The page is read at a `Fetch` and written at a `Commit`
   that wrote, at the point each runs; a `Fetch` of an absent context is the
   host's to fill or refuse there (RFC-0090 rule 1).
3. **Moves.** A context left moved out at an exit is reported at the move the
   source wrote, once per move, as `context @x is moved out here and not
   assigned again before the run ends`. A context touched while a spawn that
   touches it is in flight is a use-after-move.
4. **The summary.** A function's type carries a context summary: the contexts
   its body may read and the contexts it may write. Reading `@x` or lending it
   with `&` puts it in the read set; assigning it or lending it with `&mut`
   puts it in the write set.
5. A call's summary is its callee's joined with the summaries of the function
   values passed to it. A function's summary is the union of its calls' and
   its own accesses, inferred bottom-up and closed over recursion in the pass
   that closes its effect.
6. The summary is a term of the function type. Where two function values join,
   at a branch or a phi, the joined summary is the union; a variable left open
   closes to the empty set.
7. An ExternFn's own summary is empty and cannot be declared. An ExternFn that
   returns a function value relays a summary variable from a parameter; one
   that does not is rejected at registration.
8. A call whose summary is unknown reads and writes every context.
9. The summary names whole contexts, not field paths.
10. **Ordering.** Page ops do not ride the `Order` chain: a context read or
    write is `Pure` on the reissue chain (RFC-0013). A pass that moves a call,
    or lets two calls share an `Order`, may not move a call past a store to a
    context in its read or write set, nor past a load of a context in its
    write set, and keeps a page op in its order against the calls and page ops
    of the same context. The page-op gate on a commutative run is stated once
    (RFC-0013). Dead-store elimination and sinking keep the sound
    default: a call reads every context.

**Why.** A context that is one variable is checked by the rules that already
exist for variables: a lend is a borrow of `x`, a closure writing `@x` is a
touch of `x`, and a context not assigned back is use-after-move. The bracket
around a call makes the page the medium between bodies: a callee's `Fetch`
finds the value because its caller committed it. A callee's access to a
context is invisible in the caller's body, and the summary is the fact the
caller asks for; it rides in the type because a closure is a value and only
the type follows it.

**Rejected.**
- A take and assign of the context at every use, with a lend through a
  temporary — the borrow checker saw no borrow of `x`, a value was forwarded
  stale across a call whose closure wrote it, and a second rule was needed
  for "assigned back".
- Field paths on page ops — the page stores whole values.
- Threading contexts through calls as hidden parameters and results — it does
  not scale to depth, recursion, or a closure called by an ExternFn.
- The summary in a side table — a closure is a value, and only its type
  follows it.
- A declared summary on an ExternFn — it would be a second door to a context.

## RFC-0033: A space holds a context as its type lays it out and its ops change it

Status: Accepted

1. A context lives in a space: content-addressed and append-only, holding
   nodes by the hash of their bytes and one head per identity, moved only by
   compare-and-exchange.
2. A node carries no tag; a value's bytes are laid out by its type. A value
   of a language shape is one state node: an integer in its width's bytes,
   `f64` in eight, a string as its length and bytes, an array as its count and
   elements, an object as its fields in name order, an option as one byte and
   the payload, an enum as its variant's index and payload.
3. A value of an extension type is a chain: a state node, then op nodes, each
   the parent's value changed by one recorded op. The type declares this once,
   as `Journaled`: its state layout, its ops, how an op replays, where its
   nested extension values are, and which node it was loaded from. The space
   interprets the declaration; the type never sees the space.
4. Loading follows the head back to the nearest state node and replays the ops
   forward. Committing appends the recorded ops onto the head the value was
   loaded at and moves the head; a head that moved since the load refuses the
   commit and reports where it is. A state node is written when enough ops
   have accrued, so a load is bounded.
5. A nested extension value has its own chain wherever it sits. Its parent's
   bytes name it by its head; committing the parent commits each nested value
   first, and a nested head that moved is a new parent state.
6. A space in `Plain` mode keeps no ops: every commit is a state node.
7. A page may sit over a space: a context loads from the space where a run
   or the host loads it, at the type its head records, and every context a
   run or the host stored is committed when the host asks, so a run that
   only reads moves no head. A head records its type by name and not by an
   interner's ids, so a program other than the one that wrote it reads it.
   `acvus run <script> --space <space>` runs over the location `acvus ctl`
   maps the space to, and `acvus ctl space ls <space>` lists it
   (RFC-0031).
8. `Deque` pops are tombstones: the ops at its two ends are counters, and a
   pop that would cross the other end's cursor is a conflict the replay
   detects. A value pushed and popped within one run leaves no op.
9. A type is a context a space holds when its `ExternTypeDecl::space` gives
   hooks, `SpaceHooks::of::<J>()`. `J` is the type as the runtime holds it:
   each type parameter at `Owned<Rt>`, each identity and effect parameter at
   `()`, each lifetime at `'static`. `#[derive(ExternType)]` writes these
   hooks under `#[extern_type(space)]`, and the author writes `Journaled` for
   `J`; without that impl the derive's use does not compile. The derive
   refuses `space` on a type with a `Chosen` type parameter.
10. A loaded value is written through `J`'s own crossing, so its box is the
    one the type's `Borrowable` reads: a derived type's box is keyed by its
    payload (RFC-0076), `Deque`'s by its canonical form.

**Why.** Committing a whole value every run makes an append-only log O(n)
where the change is O(1). The type is the schema, so no tag or JSON is stored.
Identity (RFC-0012) makes one head per context the unit, with no aliasing to
reconcile inside a run. The runtime carries no identity and no effect, and a
type variable is the runtime's value at run time, so every value of a type is
boxed as `J` is, up to the canonical form, and one set of hooks reads them
all. A `Chosen` parameter keys each instance's box at its own Rust type, which
no one `J` reads. A type's `Journaled` bytes are its stored format, so they
are the type's to write.

**Rejected.**
- Committing whole values each run — O(n) for an O(1) change.
- Tagged or JSON node layout — the type is already the schema.
- A generated `Journaled` — its bytes are the stored format and belong to
  the type.
- Writing a loaded value at `J` itself — a derived type's box is its
  payload's, so the read panics in a debug build and reads another type's box
  in a release build.
- Keying a derived type's box at the struct so that `J` itself is the key —
  it reverses RFC-0076's payload key for every derived type.
