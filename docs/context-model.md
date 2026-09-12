# The context model

A context lets a program name a value that lives in a space — memory, a GPU, a
disk, a database — and use it as an ordinary value, without ever seeing the
space. `@x` is the observable face of that value; its type and its operations
are all the program sees, never the bytes or the device behind it.

From that one move the memory model dissolves. Concurrency primitives fall away
too, but by a separate mechanism — the language schedules parallelism itself
rather than exposing them — and the two absences together cost the language its
standard library, which is why it is a guest that lives inside a host.

## The essence

The program deals with a space and never sees its substance. `@0x04ff` may be a
real memory address; `@weights` may be a GPU buffer; `@ledger` may be a database
row. The program writes the same code against all of them: a value of a known
type, reached through its operations. What the space is, and what an operation
does to it, is the host's business, not the program's.

## The cores

Everything below rests on five pieces.

**Identity.** Every `@` carries an identity (RFC-0012) that is provably distinct
from every other and can never be issued twice. Distinctness is minted, not
observed at runtime. From it come three facts a compiler usually pays dearly
for: two contexts never alias, two regions are disjoint, two IR values are
different.

**A space, abstracted.** `@` names a value in a backing space. The space's
identity — which memory, which device — travels with the value; its substance
does not surface. What the space is and what an operation means are chosen by
the host.

**A place, not a value** (RFC-0015). A context is reached through `&` and
`&mut`; it is never moved out, and usually never copied. The value is pinned to
its space, as a memory-mapped register is pinned to its address.

**Operations only.** A context's whole interface is its operator functions. The
type knows its own shape and its own persistence — a `Regex` dumps its pattern
and recompiles on restore — and the program reaches the value only through those
operations, never through a generic move, copy, or serialize.

**Effect and commutativity** (RFC-0013, RFC-0017). Which contexts an operation
reads and writes, and whether two operations commute, is the dependency and
ordering model. It is what an interpreter's optimizer reorders against, and it
is the schedule a transpiler emits.

## What dissolves

**The memory model.** The program reaches contexts only as places, distinct
identities never alias, and the space never surfaces. There is no pointer to
follow, no provenance to track, no question of which write a read observes: the
identity settles aliasing exactly, and the effect model settles ordering.

**Concurrency primitives** — set aside not by context but by a separate choice.
The language schedules parallelism itself instead of exposing primitives, a
runtime strategy the current runtime uses to overlap IO. Context only supplies
what that schedule consumes: distinct identities give aliasing, the effect model
gives dependencies. A lock or an atomic is never written because the schedule is
derived, not requested.

## The cost, and the shape it forces

A language with no memory model and no concurrency primitives cannot carry a
conventional standard library, because a standard library is built on both. So
there is no standard library — an absence by decision, not omission.

Without one, the language cannot stand alone, so it lives as a guest inside a
host and borrows the host's world through the extern boundary (RFC-0009,
RFC-0010). The host supplies the operator functions and the concrete values they
act on. The current host is Rust: a `Regex` is Rust's `regex::Regex`, and a
context's persistence is Rust code.

The host is not fixed. A value's representation is the host's own type
(`Runtime::Value`), so a different host binds a different value world while the
cores above — identity, effect, place — stay unchanged. The "backend" that an
instance below names is exactly this host: one host runs the program, another
emits it.

The language promises no stability — not across a decade, not across versions.
A program that must keep working pins the version it was written against and
treats the language as a meta-language: a tool used to generate or drive an
artifact, where the artifact, not the source's forward compatibility, is what is
kept. This too follows from having no standard library — with no stable surface
of its own to promise, the language leaves stability to the pin and to the host.
A version makes no promise to the next, yet is still held to soundness within
itself. In this, as in most things, it runs opposite to the direction languages
have taken toward compatibility guarantees.

## Instances

Each instance is a projection of the cores, differing only in what the host is
and which of move, copy, and consume it permits.

**Persistence and memory-mapped I/O.** The host runs the program against a live
space and serializes only at the run boundary. Reads and writes stay typed and
in place — a nested field is written without touching the rest — so persistence
is a boundary event, not a cost on every access.

**Allocation.** `malloc` splits a space's identity into two disjoint typed views
— the writable child and the remainder — sharing one provenance. The child is
move-only, so it cannot be duplicated; `dealloc` recognizes it by its identity
and merges it back, a static fact rather than a runtime search.

**Append-only storage.** A context is a place, so a value in it cannot be
consumed. An operation that would change its type — a `Deque` tainting into a
`List` on a middle insert — is a consuming operation, and consuming is
unavailable, so only in-place operations remain. Append-only is then a fact of
the type, and storage can be an incremental log.

**GPU transpilation.** When `@` is a pure tag — not moveable, not copyable,
reached only through operator functions — it is an SSA value in a dataflow IR.
Distinct identities give total noalias, the analysis a GPU compiler most needs
and can rarely recover, so lowering each `@` to a Triton IR value loses nothing.
The interpreter is then a transpiler that knows aliasing exactly.
