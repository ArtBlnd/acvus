# RFC-0008: Static context and resumable execution

Status: Accepted
Date: 2026-09-07
Supersedes: none

## Ruling

### Accepted

A context reference is a static variable. Reading and writing it are plain
loads and stores; the compiler may still thread those through SSA to expose
order, but nothing else distinguishes a context from a global.

The interpreter can dump its complete state together with a version, and a
later process can resume from that dump. Every program is therefore
resumable.

An ExternFn has no declaration of which contexts it reads or writes. A script
reads a context and passes the value as an argument.

A context is internal state and nothing else. Nothing observes it between
two instructions: there is no second thread, an ExternFn cannot reach it,
and a dump is taken to resume from, not to look at. What the host is meant
to see goes out through an ExternFn call. The compiler therefore treats
every context load and store as ordinary, with calls as the only barrier,
and a context carries no policy: not volatile, not read-only. A value a
script only reads is a function argument, not a context.

### Proposed

An ExternFn author declares an effect level, and the levels form a chain:

    Pure < Idempotent < Opaque

Pure calls are free to reorder and may be suspended before. Idempotent calls
keep their order against other effects but may be suspended before; on resume
the call is issued again, and issuing it twice is the same as once. Opaque
calls keep their order and may not be suspended. The default for an
undeclared ExternFn is Opaque, the top of the chain.

A dump point is any instruction boundary in sequential code at which no
Opaque call is in flight. Inside an order-irrelevant block the interpreter
suspends only when every in-flight call is Idempotent or Pure.

The name of the term in code is `Effect`. The removed effect system carried
context read and write sets; this term carries only the chain.

## Rationale

The language has no concurrency primitive. With a single thread of control,
a static variable and an SSA-threaded context are the same thing observed
from two sides, and the simpler statement is the static one.

An interpreter that mediates every execution (RFC-0003) holds the whole
program state at each instruction boundary; the only state it cannot hold is
the inside of an ExternFn call that has started. Whether a call may be
repeated is a fact its author knows and the script author does not, so it is
declared where purity is declared, and by the same person.

Two independent bits, pure and suspendable, would admit "pure but not
suspendable", which is not a real thing. A chain makes that combination
unwritable. The middle level is named for its cause, idempotence, not for its
consequence: any asynchronous call can yield, but only one whose repetition is
harmless can be resumed by re-issuing it.

## Not built

- No yield statement in the language. Suspension is a property of the call,
  not a construct the script author writes.
- No context policy. A volatile context would be a second door to the host
  beside ExternFn; a read-only context is an argument.
- No context read/write declaration on ExternFns. The reason it existed,
  ordering of context effects, is carried by RFC-0007.
- No suspension inside an order-irrelevant block while an Opaque call is in
  flight, and initially no suspension inside such a block at all.
- No suspension while a call is in flight. That would need cancel safety, a
  property stronger than idempotence; if it is ever wanted it enters as a
  refinement below Idempotent, not as a new level.

## Consequences

- The Fn type and UserDefined type arguments carry the effect as a term with
  variables in the inference phases; an unresolved variable resolves to
  Opaque.
- The runtime value that wraps an extern Rust value is not called opaque; the
  word belongs to the effect level.
- Deserialized state must carry the version it was produced by, and resume
  refuses a mismatched version.
- Passes that reorder calls treat Idempotent exactly as Opaque.

## Open questions

- Whether resumption inside an order-irrelevant block is ever opened, and
  with what join semantics.
