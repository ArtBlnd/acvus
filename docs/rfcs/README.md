# acvus RFCs

An RFC records one design ruling that code cannot carry: a decision not to
build something, a direction the code is moving toward, or a boundary the
code must respect. Anything the code can carry lives in the code. An RFC is
therefore never a status report.

## Rules

- One ruling per RFC. A second ruling is a second RFC.
- An RFC contains no file paths, line numbers, test counts, crate names as
  implementation locations, or statements of implementation progress.
- Status is exactly one of `Accepted`, `Proposed`, or `Superseded by RFC-NNNN`.
  An RFC may hold an `Accepted` ruling and a `Proposed` mechanism only when the
  two are separated under their own headings inside `## Ruling`.
- A superseded RFC is not edited beyond its `Status:` line. The successor
  names it under `Supersedes:`.
- Numbers are allocated sequentially and never reused.

## Template

```markdown
# RFC-NNNN: <title>

Status: Accepted | Proposed | Superseded by RFC-NNNN
Date: YYYY-MM-DD
Supersedes: none | RFC-NNNN

## Ruling
The decision, stated positively, in a few sentences.

## Rationale
Why this ruling and not its alternatives.

## Not built
Decisions not to build, each with its reason. `none` if empty.

## Consequences
Interface-level obligations the code must satisfy. Never locations.

## Open questions
What the ruling leaves undecided. `none` if empty.
```

## Index

| RFC | Ruling | Status |
|---|---|---|
| [RFC-0001](0001-enrichment-pipeline.md) | Enrichment pipeline | Accepted |
| [RFC-0002](0002-infrastructure-boundaries.md) | Infrastructure boundaries | Accepted |
| [RFC-0003](0003-interpreter-direction.md) | Interpreter direction | Accepted |
| [RFC-0004](0004-no-first-class-hashmap.md) | No first-class HashMap | Accepted |
| [RFC-0005](0005-extern-fn-fusion.md) | ExternFn fusion | Proposed |
| [RFC-0006](0006-orchestration-lowering.md) | Orchestration lowering | Accepted |
| [RFC-0007](0007-io-ordering.md) | IO ordering | Accepted |
| [RFC-0008](0008-static-context-and-resume.md) | Static context and resumable execution | Superseded by RFC-0014 |
| [RFC-0009](0009-extern-declaration.md) | ExternFn declaration from a Rust function | Superseded by RFC-0023 |
| [RFC-0010](0010-runtime-contract.md) | The runtime contract | Superseded by RFC-0022 |
| [RFC-0011](0011-declared-type-bounds.md) | Declared bounds on type variables | Accepted |
| [RFC-0012](0012-identity-parameters.md) | Identity as a parameter of a user-defined type | Accepted |
| [RFC-0013](0013-commutative-effects.md) | Commutative effects | Accepted |
| [RFC-0014](0014-run-and-static-variables.md) | The run is the unit; a static variable outlives it | Accepted |
| [RFC-0015](0015-borrowed-places.md) | A place may be lent to a call | Superseded by RFC-0018 |
| [RFC-0016](0016-extension-types-as-views.md) | An extension type is a view over the runtime's value | Superseded by RFC-0022 |
| [RFC-0017](0017-context-access-summary.md) | A function's type says which contexts it reads and writes | Accepted |
| [RFC-0018](0018-references-are-types.md) | A reference is a type, and only a primitive copies | Accepted |
| [RFC-0019](0019-shared-signatures.md) | A shared signature and its instances | Accepted |
| [RFC-0020](0020-operators-borrow.md) | Operators on language-owned types are instructions; on extension types, a shared signature | Accepted |
| [RFC-0021](0021-registry.md) | A registry is a manifest and a handler table, combined once | Accepted |
| [RFC-0022](0022-thin-runtime-contract.md) | The runtime contract is erase, materialize, reference, and call | Accepted |
| [RFC-0023](0023-extern-declaration.md) | Declaring an ExternFn | Accepted |
| [RFC-0024](0024-patterns-through-a-reference.md) | A pattern matched against a reference binds references | Accepted |
| [RFC-0025](0025-context-is-a-variable.md) | A context is a variable of the body that touches it | Accepted |
| [RFC-0026](0026-string-copies.md) | A String is an immutable value, copied by `StringClone` | Accepted |
| [RFC-0027](0027-polymorphic-instances.md) | A polymorphic instance of a shared signature | Accepted |
