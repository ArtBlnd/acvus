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
