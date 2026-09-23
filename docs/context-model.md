# The context model

A context lets a program name a value that lives in a space — memory, a GPU,
a disk, a database — and use it as an ordinary value, without ever seeing
the space. `@x` is the observable face of that value; its type and its
operations are all the program sees, never the bytes or the device behind
it.

Two absences follow. The memory model goes, because the program has no
pointer to follow. Concurrency primitives go too, but by a separate
decision — the language schedules parallelism itself rather than exposing
primitives — and the two absences together cost the language its standard
library, which is why it is a guest that lives inside a host.

## The essence

The program deals with a space and never sees its substance. `@0x04ff` may
be a real memory address; `@weights` may be a GPU buffer; `@ledger` may be a
database row. The program writes the same code against all of them: a value
of a known type, reached through its operations. What the space is, and what
an operation does to it, is the host's business.

## The cores

**Identity.** Every `@` carries an identity (RFC-0012) distinct from every
other, minted rather than observed at runtime, and never issued twice. Three
facts follow from it: two contexts never alias, two regions are disjoint,
two IR values are different.

**A space, abstracted.** `@` names a value in a backing space. The space's
identity — which memory, which device — travels with the value; its
substance does not surface. A space is content-addressed and append-only,
and a type declares once how its value is laid out and which ops it records
(RFC-0033).

**A place, not a value** (RFC-0018, RFC-0025). A context is reached through
`&` and `&mut`; it is never moved out, and usually never copied. The value
is pinned to its space, as a memory-mapped register is pinned to its
address.

**Operations only.** A context's whole interface is its operator functions.
The type knows its own shape and its own persistence — a `Regex` dumps its
pattern and recompiles on restore — and the program reaches the value only
through those operations, never through a generic move, copy, or serialize.

**Effect and commutativity** (RFC-0013, RFC-0025 rule 4). Which contexts an
operation reads and writes, and whether two operations commute, is the
dependency and ordering model. It is what the optimizer reorders against,
and it is the schedule a transpiler would emit.

## What the cores remove

**The memory model.** The program reaches contexts only as places, distinct
identities never alias, and the space never surfaces. There is no pointer to
follow, no provenance to track, and no question of which write a read
observes: the identity settles aliasing exactly, and the effect model
settles ordering.

**Concurrency primitives.** The language schedules parallelism itself
instead of exposing them; the current runtime uses that schedule to overlap
IO. Context supplies what the schedule consumes — distinct identities for
aliasing, the effect model for dependencies — so a lock or an atomic is
never written, because the schedule is derived rather than requested.

## The cost, and the shape it forces

A standard library is built on a memory model and on concurrency
primitives, so a language with neither cannot carry one. There is no
standard library — an absence by decision, not omission.

Without one, the language cannot stand alone, so it lives as a guest inside
a host and borrows the host's world through the extern boundary (RFC-0023,
RFC-0039). The host supplies the operator functions and the concrete values
they act on. The current host is Rust: a `Regex` is Rust's `regex::Regex`,
and a context's persistence is Rust code.

The host is not fixed. A value's representation is the host's own type
(`Runtime::Value`), so a different host binds a different value world while
identity, effect and place stay unchanged. The "backend" an instance below
names is exactly this host: one host runs the program, another emits it.

The language promises no stability, across versions or across a decade. A
program that must keep working pins the version it was written against and
treats the language as a meta-language: a tool used to generate or drive an
artifact, where the artifact is what is kept. This follows from having no
standard library — with no stable surface of its own to promise, the
language leaves stability to the pin and to the host. A version makes no
promise to the next, and is still held to soundness within itself.

## Instances

Each instance is a projection of the cores, differing in what the host is
and in which of move, copy and consume it permits.

**Persistence and append-only storage — built** (RFC-0033). A context is a
place, so a value in it cannot be consumed. An operation that would change
its type — a `Deque` tainting into a `Vec` on a middle insert — is a
consuming operation, and consuming is unavailable, so only in-place
operations remain. Append-only is then a fact of the type, and the space is
an incremental log: content-addressed nodes, one head per identity, moved by
compare-and-exchange. Reads and writes stay typed and in place, so
persistence is a boundary event rather than a cost on every access.

**Allocation — not built.** `malloc` would split a space's identity into two
disjoint typed views, the writable child and the remainder, sharing one
provenance. The child would be move-only, so it could not be duplicated, and
`dealloc` would recognize it by its identity and merge it back — a static
fact rather than a runtime search.

**GPU transpilation — not built.** Where `@` is a pure tag — not moveable,
not copyable, reached only through operator functions — it is an SSA value
in a dataflow IR, and distinct identities give total noalias. Lowering each
`@` to a Triton IR value would then lose nothing, and the interpreter would
be a transpiler that knows aliasing exactly.
