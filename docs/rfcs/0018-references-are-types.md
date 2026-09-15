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
error otherwise. A `&T` is never accepted where a `T` is expected, and a
`T` never where a `&T` is: types unify exactly. Assigning through a
`&mut T` stores into the storage it names.

A reference is not data. It is never stored in a container, an object, a
context, or a closure's captures; it is never returned from a function;
and it lives no longer than the function body that made it. Within that
body the checker holds one rule: while a `&mut` to a storage is live, no
other name of that storage — the place itself, a `&`, or another `&mut` —
is read or written; while a `&` is live, the storage is not assigned or
moved. Liveness is the last use the IR already computes.

A parameter's mode is its type. A function type is `Fn(T, &U, &mut V) ->
R`; there is no mode beside the type. A lambda writes `|a, b|`, and the
type of each parameter, reference or value, is the type the function that
receives the lambda declares — unification, from the extern signature
that names `Fn(&T) -> Bool`, fixes it.

A context place (`@x`, `@x.field`) has no take: nothing moves out of a
context. A program does one of three things with it — lends it
(`f(&@x)`, `push(&mut @x, v)`), assigns it (`@x = v`), or duplicates it
(`clone(&@x)`). Reading a non-primitive `@x` as a value is a type error;
reading a primitive one is the word copy every primitive gets. A lent
context place is loaded, the call receives a reference to that temporary,
and the temporary is stored back when the call returns: the journal is a
version store and is never aliased.

## Rationale

The runtime contract lends closure arguments (`call_1(f, &a)`), and a
handler's value is a name for storage the host owns. On the language side
the same fact had no type: a lambda's parameter was always a value, so a
host had to copy every lent argument into the callee's frame — a copy the
IR never asked for — and an extern iterating a `&mut` collection could
not hand the loop body a reference to an element. The gap was one missing
type, not a missing mechanism — the IR already carries `Ref`, `Load`, and
`Store`, and the checker already knows each value's last use.

RFC-0015 kept references out of the type system because no two names ever
denoted one storage, so nothing needed checking. It paid for that with a
copy at every binding of a non-identity value, made by the runtime with no
instruction naming it, and with a `Clone` the runtime needed from every
extension type. The rule here is Rust's: a copy exists only where the
type is a word, and every other duplication is a call the program wrote.
No two names denote one storage except through a reference, and a
reference is the one thing checked. That check is one exclusion over a
single function body with known liveness — a local dataflow check, not a
lifetime system. Keeping references out of data, returns, and captures is
what keeps it local.

Exact unification is what keeps the solver honest: a rule that reads `&T`
as `T` or `T` as `&T` would make the lambda parameter a type the solver
can no longer pin from the extern signature alone. Spelling `&` at the
call site and `*` at the read keeps every conversion in the program.

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
- No aliasing of a context. A context place lent to a call is copy-in,
  copy-out.

## Consequences

- `ParamMode` is removed from function types; a parameter of reference
  type carries the mode. The argument-mode check becomes ordinary type
  unification of `&place : &T`.
- Move checking treats every non-primitive type as move-only; identity no
  longer decides it. A second use of a moved binding is an error at the
  use.
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
- A host that receives a lent closure argument gives the callee's
  reference parameter an alias to the lent value, not a copy. The
  argument's borrow is exactly the call future's lifetime. A host reads a
  register by taking it; the only copy a host makes is of a word.
- Sharing in the language is an extern: `clone(&x)` for a type whose
  author implemented it. A type without one cannot be duplicated.
- A context read of a non-primitive type is rejected at the read; the
  journal is read only through a lent place, and never copies a value
  out on its own.
- An ExternFn's Rust `&T` and `&mut T` parameters declare `&T` and
  `&mut T` acvus types; `Fn1<&T, R>` in a signature declares a lambda that
  takes a reference. The iterator functions declare `Fn(&T) -> Bool` for
  predicates and `Fn(T) -> U` for maps.

## Open questions

none
