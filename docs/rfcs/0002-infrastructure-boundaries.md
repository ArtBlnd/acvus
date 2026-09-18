# RFC-0002: Infrastructure boundaries

Status: Accepted
Date: 2026-03-30
Supersedes: none

## Ruling

acvus is compiler infrastructure for DSLs, not a scripting language for one
use case. The MIR is parametric over the front end (the AST) and the back end
(the runtime context); replacing those two hosts another DSL.

Three things are renounced: acvus is not a general-purpose language, it has no
C FFI, and it has no JIT. ExternFn is the only door to the outside world, and
everything inside that door is required to be sound.

## Rationale

Each layer stands on the layer below it and solves directly only what the
layer below does not. User scripts stand on ExternFn, ExternFn on the Rust
ecosystem, the acvus compiler on the interpreter, the interpreter on LLVM
through Rust, and all of it on Rust's memory safety. Renouncing a C FFI is
what keeps the bottom of that stack Rust rather than C, so acvus adds ordering
on top of a memory-safe base rather than establishing memory safety itself.

ExternFn separates the people who write extensions from the people who write
scripts, so the declarations can be placed on one side: the ExternFn author
writes them, and the script author writes none.

## Not built

- No general-purpose language features. The composition language stays small
  so every extension goes through ExternFn.
- No C FFI. Native calls would move scheduling out of the interpreter's hands
  and put a non-Rust layer below acvus.
- No JIT. Executable memory is never allocated, so generated code cannot
  become code injection, and the interpreter keeps control of every execution.
  This is a decision not to build, not a property the IR enforces: the MIR is
  typed SSA over words and pointers, and a fork that wants a native backend
  finds nothing in the way.

## Consequences

- Every extension enters through ExternFn and is a first-class citizen of the
  SSA once inside; nothing enters halfway.
- A new feature is judged by whether it is generic at the MIR level or bound
  to a particular DSL; DSL-bound behavior belongs in the front end or the
  runtime context, not in the MIR.
- Complexity may be placed on the ExternFn author; there is no average user on
  that side of the door.

## Open questions

none
