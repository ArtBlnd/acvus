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

### Proposed

An ExternFn author declares one of three levels, and the levels form a chain:

    Pure < Yieldable < Effectful

Pure calls are free to reorder and may be suspended before. Yieldable calls
keep their order against other effects but may be suspended before; on resume
the call is issued again, so a yieldable call is one whose repetition is
harmless. Effectful calls keep their order and may not be suspended. The
default for an undeclared ExternFn is Effectful.

A dump point is any instruction boundary in sequential code at which no
Effectful call is in flight. Inside an order-irrelevant block the interpreter
suspends only when every in-flight call is Yieldable or Pure.

## Rationale

The language has no concurrency primitive. With a single thread of control,
a static variable and an SSA-threaded context are the same thing observed
from two sides, and the simpler statement is the static one.

An interpreter that mediates every execution (RFC-0003) holds the whole
program state at each instruction boundary; the only state it cannot hold is
the inside of an ExternFn call that has started. Whether a call may be
repeated is a fact its author knows and the script author does not, so it is
declared where purity is declared, and by the same person.

Two independent bits, pure and yieldable, would admit "pure but not
yieldable", which is not a real thing. A chain makes that combination
unwritable.

## Not built

- No yield statement in the language. Suspension is a property of the call,
  not a construct the script author writes.
- No context read/write declaration on ExternFns. The reason it existed,
  ordering of context effects, is carried by RFC-0007.
- No suspension inside an order-irrelevant block while an Effectful call is
  in flight, and initially no suspension inside such a block at all.

## Consequences

- The Fn type carries the three-level declaration as a term with variables in
  the inference phases; an unresolved variable resolves to Effectful.
- Deserialized state must carry the version it was produced by, and resume
  refuses a mismatched version.
- Passes that reorder calls treat Yieldable exactly as Effectful.

## Open questions

- The exact name of the three-level term in code.
- Whether resumption inside an order-irrelevant block is ever opened, and
  with what join semantics.
