# RFC-0033: A space holds a context as its type lays it out and its ops change it

Status: Accepted (first instance)
Date: 2026-09-16
Extends: RFC-0012, RFC-0022, RFC-0025

## Ruling

A context lives in a space. A space is content-addressed and append-only:
it holds nodes by the hash of their bytes, and one head per identity, moved
only by compare-and-exchange. Nothing in a node carries a tag: a value's
bytes are laid out by its type, and the type is the schema.

A value of a language shape is one state node, laid out from its type —
integers and floats as eight bytes, a string as its length and bytes, an
array as its count and elements, an object as its fields in name order, an
option as one byte and the payload, an enum as its variant's index and
payload. A value of an extension type is a chain: a state node, then op
nodes, each the parent's value changed by one op the type recorded. The
type declares all of this once, as `Journaled`: how its state is laid out,
which ops it records, how an op is applied on replay, where its nested
extension values are, and which node it was loaded from. A space
interprets the declaration; the type never sees the space.

Loading follows the head back to the nearest state node and replays the
ops forward. Committing takes the ops the value recorded, appends them
onto the head the value was loaded at, and moves the identity's head to
the result; a head that moved since the load refuses the commit and
reports where it is. A state node is written when enough ops have
accrued since the last one, so a load is bounded.

A nested extension value has its own chain, wherever it sits: as an
element of an extension type, or inside an object, array, tuple, option,
or enum that is. Its parent's bytes name it by its head; committing the
parent walks the language shapes by type down to each nested value and
commits it first, and a nested head that moved is a change of the parent,
recorded as a new parent state. A log is therefore structural: `Deque<Deque<Int>>` is an
outer chain whose nodes name inner chains, and a push into an inner deque
is one op on the inner chain and one state on the outer.

A space in `Plain` mode keeps no ops: every commit is a state node, so a
host that wants only commit and restore has them from the same
declaration.

## Rationale

The page could hand out final values but not say what they were, and a
deque's whole value every run would make an append-only log O(n) in what
should be O(change). Typing the layout removes the tag and the JSON; the
op chain is what `Deque::record` already computed, given a place to land.

Content addressing and compare-and-exchange are the store the owner ran
before, and Irmin and Noms have shown that typed values under a git-like
store carry the whole design; identity (RFC-0012) is what makes a head
per context the natural unit, with no aliasing to reconcile inside a run.

`Deque`'s pops are tombstones on the log, not deletions of elements: the
ops at the two ends are counters, and a pop that would cross the other
end's cursor is a conflict the replay detects rather than a value lost.

A run's page may sit over a space: a context is loaded from the space at
the run's first fetch and every context the run held is committed when
the host asks. `acvus run --space <dir>` runs over a directory store —
`nodes/<hex>` for nodes, `heads/<id>.json` for a head and its type — and
`acvus space <dir>` lists what the directory holds.

## Not built

- No store shared between processes: the directory store moves heads
  under one process-wide lock; two processes on one directory are not
  coordinated. A store over a network implements the same five methods.
- No merge of two chains from one head: a moved head refuses the commit;
  branching on refusal, and merging by the type's commutativity
  (RFC-0013), come later.
- No hooks for `List`, `Regex`, or the LLM types: the first instance
  declares `Deque`; a language-shape context (`String`, `Int`, objects)
  needs no hook.
- No garbage collection of nodes no head reaches.
- The deque's log is the net change of a run: a value pushed and popped
  within one run leaves no op, because a popped value is the caller's and
  cannot be copied into the log without a clone the element type may not
  have. A per-event log needs that clone.

## Consequences

- `acvus-extern`: `Journaled<Rt>` and `SpaceHooks<Rt>`;
  `ExternTypeDecl::space`; `Externs.space` by type name.
- `acvus-ext`: `Deque<Rt::Value>` implements `Journaled`, with its head.
- `acvus-interpreter`: `layout` (a value as canonical bytes by its type),
  `space` (`Store` with `MemoryStore` and `DirStore`, `Space`,
  `Mode::{Plain, Log}`, nodes, heads, `cmpxchg`, `load`, `commit`,
  `SpacePage`); `InterpreterContext::with_space`; the interpreter's page
  is a `RuntimeContext` trait object.
- `acvus`: `--space <dir>`, `acvus space <dir>`.
