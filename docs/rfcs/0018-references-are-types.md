# RFC-0018: A reference is a type, and only a primitive copies

Status: Accepted
Date: 2026-09-15
Supersedes: RFC-0015

## Ruling

A value is copied only when its type is a primitive — `Int`, `Float`,
`Bool`, `Byte`, `Unit`, `Order`. Every other value moves: a binding used
once is gone, and a second use is a type error. To use a value twice, a
program takes a reference to it or calls an extern `clone`. Identity is a
separate fact — a value that is a source of its own — and says nothing
about copying.

`&T` and `&mut T` are types. A value of `&T` is a second name for a
storage that holds a `T`: it is made by `&place` or `&mut place`, held in
a local binding, passed to a call, received by a parameter, and read
through `*r`, which yields the `T` when `T` is a primitive and is a type
error otherwise.

An expression that names no place is lent as well. The value it produces
is bound to a temporary storage, and the reference names that temporary:
`(a + b).to_string()`, `vec([1, 2]).len()`, `("a".to_string() + "b").len()`
and `f(&(a + b))` each bind one, where the receiver or the argument meets
a reference parameter. The temporary is an ordinary storage — it is the
place the loan names, the exclusion rule reaches it as it reaches a local,
and it is released where it dies, which is where a `let`'s storage is
released. A reference into a temporary is therefore refused from a
return, from data, and from a capture by the rules below, exactly as a
reference to a local is. A `&T` is never accepted where a `T` is expected, and a
`T` never where a `&T` is: types unify exactly. Assigning through a
`&mut T` stores into the storage it names.

A `*` is written at every read through a reference but two. A captured
name whose type is a word is seen by the lambda's body as that word and
copied at each use, wherever the use is; a captured name of any other
type is lent, and the body writes `*` to read it and `&` to pass it. And
an operator reads a `&T` operand the program wrote through the reference
where `T` is a word: `r + 1` with `r: &Int` is the word copy `*r` and
then the operation, and it reaches arithmetic, comparison and bit
operators; a `&T` whose `T` is not a word is not an operand of one. `==`,
`!=` and `+` lend their operands instead, at any size (RFC-0020).

A reference is not data. It is never stored in a container, an object, a
context, or a closure's captures; it is never returned from a function;
and it lives no longer than the function body that made it. Within that
body the checker holds one rule: while a `&mut` to a storage is live, no
other name of that storage — the place itself, a `&`, or another `&mut` —
is read or written; while a `&` is live, the storage is not assigned or
moved. Liveness is the last use the IR already computes.

That last use is transitive. A value that holds a reference keeps the
storage it names alive for as long as the value itself is used, and a
reference to that value is such a use: `let it = as_iter(&v)` lends `v`
to `it`, so `next(&mut it)` is a use of `v`, and `v` is dropped after the
last use of `it`, never at the `&v` that built it. A storage reached only
through a chain of such holders is alive along the whole chain.

A parameter's mode is its type. A function type is `Fn(T, &U, &mut V) ->
R`; there is no mode beside the type. A lambda writes `|a, b|`, and the
type of each parameter, reference or value, is the type the function that
receives the lambda declares — unification, from the extern signature
that names `Fn(&T) -> Bool`, fixes it.

A lambda's captures are the names the checker read from outside it while
checking its body, frozen into the resolution for the lowering to take as
written; the lowering does not decide captures from the syntax, so a name
that resolved to a function rather than to the binding it is spelled like
is not among them.

A closure owns what it captures, and a call borrows the closure: inside
the lambda a captured name of word type has that word's type and a
captured name of any other type has type `&T`, and `f(x)` on a local `f`
lends `f` for the call, so `f` may be called again. Passing `f` to a
function by value moves it. Both rules reach a captured name: a lambda
that captures a name the enclosing lambda captured captures the owned
`T`, and a captured `f` is called as a lent `f`.

A name that itself holds a reference is refused, and how a capture is
read is a decision of the solver, so a name whose type is still a
variable is read as the word or refused as a reference when that type
resolves, and not before.

Taking that owned `T` is a move out of a value the enclosing closure owns,
and the enclosing closure is called again, so the move is admitted only
where `T` is a word: there it is a copy. Every other type is refused at the
inner lambda, naming the name and the owned type; the program acts through
the reference the name already reads as, or writes `clone(&w)` where it
needs a value. This is the rule of Rust's `Fn` closures.

A context place (`@x`, `@x.field`) is a storage like a local: reading it
takes its value, and the place is uninitialized until it is assigned
again. `let a = @b` moves the value out; `@b = new` puts one back. A
context differs from a local in outliving the run, so a run that takes a
context must assign it on every path before it ends; the checker that
tracks an uninitialized local tracks this. Lending a context place
(`f(&@x)`, `push(&mut @x, v)`) is the same sequence written by the
compiler: take into a temporary, give the call a reference to it, store
the temporary back when the call returns. The journal is a version store
and is never aliased.

## Rationale

The runtime contract lends closure arguments (`call_1(f, &a)`), and a
handler's value is a name for storage the host owns. Without a reference
type on the language side a lambda's parameter is always a value, so a host
copies every lent argument into the callee's frame — a copy the IR never
asked for — and an extern iterating a `&mut` collection cannot hand the loop
body a reference to an element. What that needs is the type, not a new
mechanism: the IR already carries `Ref`, and the checker already knows each
value's last use.

RFC-0015 kept references out of the type system: no two names denoted one
storage, so nothing needed checking. Its cost was a copy at every binding of
a non-identity value, made by the runtime with no instruction naming it, and
a `Clone` required of every extension type. The rule here is Rust's: a copy
exists only where the type is a word, and every other duplication is a call
the program wrote. No two names denote one storage except through a
reference, and a reference is the one thing checked. That check is one
exclusion over a single function body with known liveness — a local dataflow
check, not a lifetime system. Keeping references out of data, returns, and
captures is what keeps it local.

Exact unification is what lets the solver pin a lambda's parameter type from
the extern signature alone; a rule that read `&T` as `T` or `T` as `&T`
would leave two candidates at every such parameter. Spelling `&` at the call
site and `*` at the read keeps every conversion in the program text.

## Not built

- No reference in data, in a return type, or in a capture. Each would
  make a reference outlive the body that made it, and checking that is a
  lifetime system. A closure that needs a captured place changed is
  written as a lambda that takes it as `&mut` — the extern that runs the
  lambda passes it.
- No `&&T`. Borrowing a reference is the reference.
- No coercion between `T` and `&T` in either direction. `f(x)` with
  `f: Fn(&T)` is a type error, and so is `g(r)` with `r: &T` and
  `g: Fn(T)`; the program writes `f(&x)` and `g(*r)` or `g(clone(r))`.
- No implicit copy of anything but a primitive. `*r` on a non-primitive
  is a type error; the copy is the extern `clone(r)`.
- No copy instruction in the IR, and no clone in a runtime's vtable. A
  runtime copies a word and nothing else.
- No aliasing of a context. A context place lent to a call is taken into
  a temporary and stored back.

## Consequences

- `ParamMode` is removed from function types; a parameter of reference
  type carries the mode. The argument-mode check becomes ordinary type
  unification of `&place : &T`.
- Move checking treats every type that is not a word as move-only, and an
  option is a word exactly when its payload is, since it has no
  representation of its own (RFC-0039); identity no longer decides it. A second use of a moved binding is an error at the
  use. A storage is read — by a take of a place in it, by a reference to
  one — only while the part read is alive, where two places overlap when
  one is a prefix of the other: a take of a place moves that place, a
  store into a place gives it and everything under it a value again, and
  a store cannot reach through a place that is gone. A reference to a
  moved place, and a reference to the whole of a storage some place
  inside which has moved, are both uses after move.
- The type checker reads a reference operand of an operator at what it
  names, and lowering emits `Take { Through }` before the `BinOp` where
  the operand register holds a `&word`.
- `Ref<T>` and `RefMut<T>` are types the checker admits as the type of a
  local binding and of a parameter, and rejects inside any data type, any
  return type, and any capture. `*r` is typed only for a primitive `T`.
- A lambda's parameter types are inference variables unified with the
  receiving function type; a parameter may resolve to a reference type.
- The move checker gains the exclusion rule: a live `&mut` forbids every
  other use of its storage until its last use; a live `&` forbids
  assignment to and moving of its storage.
- The IR's `Ref` names a local storage or a context path; a reference
  value is that name at runtime. A host represents a reference to a local
  as an alias to the register, and the runtime glue reads an argument of
  type `&T` by peeking the storage rather than taking it.
- A host's `call_0/1/n` take their arguments by value and move each into
  the callee's parameter. An extern whose closure parameter is `&T` passes
  `reference(&value)`: a reference word the host makes, used no longer
  than the call. A host reads a register by taking it; the only copy a
  host makes is of a word.
- The host binds each capture register to a reference into the closure
  value. The checker decides, once per capture, whether the body reads
  that register as the word it names or as the reference itself, and the
  lowering emits the reading it decided: a `Take { Through }` at the
  body's entry for a word, so that however often the body reads the name,
  the closure is read once; the register as it is for every other type,
  whose body-side type is the `&T` the checker gave it.
- Sharing in the language is an extern: `clone(&x)` for a type whose
  author implemented it. A type without one cannot be duplicated.
- A context read hands the value out of the journal; the journal never
  copies a value on its own. A run that takes a context and ends on any
  path without assigning it is rejected.
- A context lent to a call, `&@x` as much as `&mut @x`, lowers to a take
  before the call and an assign after it. The function's context summary
  (RFC-0017) still says read for `&@x` and write for `&mut @x`; an IR
  pass that orders by `Assign` sees the store-back of a shared lend as a
  write and orders conservatively. That costs parallelism, never
  correctness.
- An ExternFn's Rust `&T` and `&mut T` parameters declare `&T` and
  `&mut T` acvus types; `Fn1<&T, R>` in a signature declares a lambda that
  takes a reference. The iterator functions declare `Fn(&T) -> Bool` for
  predicates and `Fn(T) -> U` for maps.

## Open questions

none
