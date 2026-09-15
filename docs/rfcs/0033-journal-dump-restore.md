# RFC-0033: A context is dumped by its type, journaled by its change, restored by both

Status: Proposed
Date: 2026-09-16
Extends: RFC-0022, RFC-0025

## Ruling

Persistence is typed. A value is dumped as JSON by its declared type and
restored from JSON by the same type; no value carries a tag. The
language's own shapes — `Int`, `Float`, `Bool`, `Byte`, `Unit`, `String`,
`Array`, `Tuple`, `Object`, `Option`, a structural enum — dump
structurally, recursing by the type's arguments. An extension type dumps
through a `Persist` hook the type declares beside its type declaration;
the hook is attached to the type's vtable when the registries are combined
(RFC-0022 left the slot for it) and receives the type's arguments and the
dumper for a value, so `Deque<T>` dumps its elements as `T`. A type that
declares no hook cannot be a persisted context: dumping it is an error at
the run boundary, not a placeholder.

A run's changes to its contexts are a journal: an append-only sequence of
entries, one per context the run changed. A context of a type with a
`Delta` hook — `Deque` — journals the delta the type reports: what was
dropped at each end and what was pushed at each end since it was last
settled; the type is then settled. Any other changed context journals its
whole dump. A context the run only read has no entry: for a delta type the
delta is empty, for the rest the dump equals the loaded one.

A dump is a snapshot: every context of the page, keyed by name. A restore
takes a dump and a journal and replays the journal onto the dump —
`set` replaces a value, a deque delta pops the dropped ends and pushes
the pushed ends — so a page is the last dump plus every entry since.

## Rationale

The page could hand out final values but not say what they were: a
`Small` word is an `Int` or a `Float` only by its type, and an extension
value is opaque. Typing the dump is the only honest dump; the CLI already
did it for the language's shapes and printed `<Deque<Int>>` for the rest.

A deque is the append-only case the context model names
(docs/context-model.md): a run pushes and pops at the ends and the whole
value is large. Journaling its delta is what `Deque::record` was built
for; journaling its whole dump would make every run O(n) in a log that
should be O(change).

## Not built

- No journal for closures or handles: a run that leaves one in a context
  is an error at the boundary.
- No compaction: a restore replays every entry since the dump; when to
  take a new dump is the host's choice.
- No delta for `List`: a list changes by replacement.

## Consequences

- `acvus-extern`: `Persist` and `Delta` traits; `ExternTypeDecl`
  reports its hooks; `Externs` carries them by `TypeId`.
- `acvus-interpreter`: typed `dump` / `restore` over `Value` and `Ty`;
  the vtable's persistence slot; `Journal` entries and `replay`; the page
  reports a run's entries instead of bare final values.
- `acvus-ext`: `Persist` for `List`, `Regex` (pattern), and `Deque`, with
  `Delta` for `Deque`.
- `acvus`: `--journal <file.jsonl>` appends the run's entries; `--commit`
  writes the dump; `acvus restore <dump.json> <journal.jsonl>` prints
  the replayed page.
