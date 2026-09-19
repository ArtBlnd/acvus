# acvus RFCs

An RFC records one design ruling that code cannot carry: a decision not to
build something, a direction the code is moving toward, or a boundary the
code must respect. Anything the code can carry lives in the code. An RFC is
therefore never a status report.

## Rules

- One ruling per RFC. A second ruling is a second RFC.
- An RFC cites the tree where a sentence is checkable there — a file and
  line at the commit it names — and carries no statements of
  implementation progress.
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
| [RFC-0022](0022-thin-runtime-contract.md) | The runtime contract is erase, materialize, reference, and call | Superseded by RFC-0039 |
| [RFC-0023](0023-extern-declaration.md) | Declaring an ExternFn | Accepted |
| [RFC-0024](0024-patterns-through-a-reference.md) | A pattern matched against a reference binds references | Accepted |
| [RFC-0025](0025-context-is-a-variable.md) | A context is a variable of the body that touches it | Accepted |
| [RFC-0026](0026-string-copies.md) | A String is an immutable value, copied by `StringClone` | Accepted |
| [RFC-0027](0027-polymorphic-instances.md) | A polymorphic instance of a shared signature | Accepted |
| [RFC-0028](0028-container-signatures.md) | A container is read through shared signatures; a reference is one carrier | Accepted |
| [RFC-0029](0029-exclusion-as-written.md) | Exclusion is checked as the source wrote it, over every holder | Accepted |
| [RFC-0030](0030-paths-and-method-calls.md) | A qualified call names a namespace; a method call is a call on its receiver | Accepted |
| [RFC-0031](0031-script-runner.md) | `acvus`, the script runner | Accepted |
| [RFC-0032](0032-objects-across-the-boundary.md) | An object crosses the boundary as its fields | Accepted |
| [RFC-0033](0033-space.md) | A space holds a context as its type lays it out and its ops change it | Accepted (first instance) |
| [RFC-0034](0034-vec.md) | The dynamic-length sequence is `Vec<T>` on both sides | Accepted |
| [RFC-0035](0035-runtime-parameter-by-use.md) | An extern fn takes its runtime only when it uses it | Accepted |
| [RFC-0036](0036-enums-across-the-boundary.md) | A Rust enum crosses the boundary as the language's enum | Accepted |
| [RFC-0037](0037-integer-widths.md) | Integers have a width, and a literal takes the width its use demands | Accepted |
| [RFC-0038](0038-result-and-trap.md) | `Result<T, E>` is a primitive, `?` widens the error, and a trap is not an error | Accepted (Result, `!`, `?` built; trap to follow) |
| [RFC-0039](0039-one-crossing.md) | One crossing at the boundary | Accepted |
| [RFC-0040](0040-instance-chosen-by-the-compiler.md) | The compiler chooses an ExternFn's instance and the runtime indexes it | Accepted |
| [RFC-0041](0041-representation-of-a-slot.md) | `#τ` is the representation of a slot, and an extension holds a uniform value through `Erased` | Accepted |
| [RFC-0042](0042-equality-and-decision.md) | The solver separates equality from decision: one join, one settlement | Accepted |
| [RFC-0043](0043-a-name-is-a-set-of-signatures.md) | A bare name is a set of signatures, decided as an instance is | Accepted |
| [RFC-0044](0044-a-body-is-prepared-once.md) | A body is prepared once, and a failure at run time is a panic | Accepted |
| [RFC-0045](0045-let-binds-and-assignment-assigns.md) | `let` binds, `x = e;` assigns, in one statement grammar | Accepted |
| [RFC-0046](0046-a-calls-task-is-an-effect.md) | A call's task is an effect: `Task::{Sync, Async, Heavy}` | Accepted |
| [RFC-0047](0047-a-slice-is-the-one-thing-the-machine-indexes.md) | A slice is the one thing the machine indexes, and a bound is proved by an interval | Accepted |
| [RFC-0048](0048-ownership-is-the-machines.md) | Ownership is the machine's: a value copies, a register is written once | Accepted |
| [RFC-0049](0049-a-cast-is-a-leaf.md) | `expr as T` is Rust's `as`, and inside a chain it is a leaf | Accepted |
| [RFC-0050](0050-an-aggregate-is-its-components-until-it-escapes.md) | An aggregate is its components until it escapes, and the heap is the spill | Accepted |
| [RFC-0051](0051-a-match-is-one-dispatch-and-is-exhaustive.md) | A `match` is one dispatch, and it is exhaustive where the variant set is known | Accepted |
| [RFC-0052](0052-an-operation-is-a-struct-the-machine-calls-once.md) | An operation is a struct the machine calls once, and it holds its successor | Accepted |
| [RFC-0053](0053-an-aggregate-that-does-not-escape-never-exists.md) | An aggregate that does not escape never exists | Accepted |
| [RFC-0054](0054-the-host-declares-what-main-returns.md) | The host declares what `main` returns, and the compilation holds the body to it | Accepted |
| [RFC-0055](0055-a-constant-expression-folds.md) | A constant expression folds | Accepted |
| [RFC-0056](0056-a-loop-multiplies-once.md) | A loop multiplies once | Accepted |
| [RFC-0057](0057-a-for-loop-is-a-terminator.md) | A `for` loop is a terminator | Accepted |
| [RFC-0058](0058-a-literal-says-its-type.md) | A literal says its type — `10u64`, `'c'`, `b"…"`, and `char` | Accepted |
| [RFC-0060](0060-a-small-pure-closure-called-where-it-was-made-is-its-body.md) | A small pure closure called where it was made is its body | Accepted |
| [RFC-0061](0061-a-store-nothing-reads-is-dead.md) | A store nothing reads is dead | Accepted |
| [RFC-0062](0062-a-string-slice-is-a-register-pair.md) | A string slice is a register pair | Accepted |
