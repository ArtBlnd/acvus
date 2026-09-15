# RFC-0026: A String is an immutable value, copied by `StringClone`

Status: Accepted
Date: 2026-09-15
Supersedes: none

## Ruling

`String` is a language-owned, immutable value (RFC-0020). A use of a
`String` value that is not its last use copies it: the compiler inserts
`StringClone` before that use, so the use consumes the copy and the
original lives on to its last use, which moves it. The name `clone(&s)`
is the same instruction written by hand. No `&mut String` exists.

`StringClone`'s contract is one `String` with the same bytes. A host may
implement it as a copy, as a shared buffer, or as copy-on-write; with no
mutation in the language the choice is unobservable.

## Rationale

A string is the value a template language handles most, and a program
that binds one and names it twice is the ordinary program, not the
exceptional one. With the move rule alone (RFC-0018) every such program
is a use-after-move and must write `clone(&x)` at each reuse; with an
implicit copy the reuse is free to write, and its cost is one named
instruction the IR shows. Making the string immutable is what lets the
host choose a representation in which that instruction is a refcount
rather than an allocation; nothing in the language mutates a string today,
so the immutability costs nothing.

## Not built

- No copy of any other heap type. `Array`, `Object`, `Tuple`, `Option`,
  and every extension type move; a reuse is `clone(&x)` where an
  instance exists.
- No representation rule for the host. The interpreter copies today.
- No `&mut String`, and no in-place string operation.

## Consequences

- `StringClone { dst, src }` takes `src` as a `String` (not consumed) or a
  `&String`.
- The `string_copy` pass runs after SSA on every body: for each use of a
  `String`-typed value that is live after the instruction (by liveness),
  a `StringClone` is inserted before it and the use is redirected to the
  copy; a terminator argument is treated the same way.
- The move checker sees the copy as any other value; a `String` reaches
  its last use exactly once.
