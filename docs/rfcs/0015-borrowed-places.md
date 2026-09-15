# RFC-0015: A place may be lent to a call

Status: Superseded by RFC-0018
Date: 2026-09-10
Supersedes: none

## Ruling

A place is a local binding, a context, or a field path of either. Every
place can be read, which yields its value; assigned, which stores a value
into it; and lent to a call, written `&place` or `&mut place` in the
argument position. A lent place is given to the call and is back when the
call returns: unchanged after `&`, holding whatever the callee left after
`&mut`. Nothing runs between the call taking the place and giving it
back.

`&` and `&mut` are modes of a parameter, not types. A function's type
says for each parameter whether it takes a value, borrows, or borrows
mutably; no value of a reference type exists, so a reference is never
stored, returned, captured, or compared. Lending does not move: after the
call the binding is still bound and initialized. Lending is the only way a
move-only value crosses a call without being consumed.

An argument in a borrowing position must be a place; a temporary cannot
be lent. One call may name a place at most once among its arguments,
whether lent or read by value.

A run's view is copy-in, copy-out: the place is loaded, the callee
receives the value, and the value the callee leaves is stored back. A
call therefore yields one result per place it borrowed mutably, next to
its return value.

## Rationale

Values are copied when bound and identities are moved, closures capture
by value, contexts are threaded through SSA, and there is no second
thread of control. No two names ever denote one storage, so a mutable
reference here needs none of what makes it hard elsewhere: no lifetime,
no rule against a second reader, no race. The one thing left to check is
that a call does not name the same place twice, because argument
evaluation order would otherwise decide the meaning.

The previous protocol returned the changed value through the return
type, so an iterator's `next` returned `Option<(T, Iter)>` and the loop
reassigned by hand. The change to a place is a fact about the parameter,
and putting it in the return type made every such function carry a
convention that a caller had to know. With the mode on the parameter the
return type is free, and the call site shows which place changes.

Reading a place, assigning it, and lending it are the same three
operations for a local and for a context. A context differs only in
living past the run (RFC-0014).

## Not built

- No reference values. A reference exists only between a call's
  argument list and its return.
- No method-call syntax. `place.f(args)` would take the mode from `f`'s
  signature and hide at the call site that `place` changes; the call is
  written `f(&mut place, args)`.
- No borrowing parameters on local functions or lambdas in this ruling.
  A local function calls with `&mut` freely; declaring one is a later
  ruling, and a lambda changing a captured place is a different question,
  since captures are values.
- No compiler proof that a callee given `&place` leaves it unchanged.
  The extension author writes the body; that it does not keep or alter
  the value is the author's fact, like commutativity.

## Consequences

- A parameter of a function type carries its mode; the checker rejects an
  argument whose mode does not match, a non-place in a borrowing
  position, and a place named twice in one call.
- An ExternFn declared from a Rust signature takes its modes from `&T` and
  `&mut T` parameters; the runtime glue lends the value to the body and
  takes it back.
- The move checker treats a lent place as neither moved nor copied.
- A call instruction yields the new value of each mutably borrowed place
  as a result; the lowering stores each back into its place, and the SSA
  pass sees an ordinary store.
- The iterator protocol becomes `next(&mut it) -> Option<T>`; collections
  gain `&mut` operations such as `push`.

## Open questions

- Whether `&` should be spelled at all, or a move-only value passed by
  value to a function whose parameter is `&T` is lent implicitly. The
  ruling spells it, on the side of the call site saying what happens.
