# RFC-0005: ExternFn fusion

Status: Proposed
Date: 2026-03-31
Supersedes: none

## Ruling

A run of consecutive ExternFn calls on an SSA chain may be replaced by a single
fused ExternFn when every intermediate value is used only by the next call in
the run. Fusion is a pure optimization: a program that is not fused is still
valid. A fused ExternFn calls the original functions in order by default; its
implementer may replace that with a single loop.

## Rationale

Once every builtin is an ExternFn, fusion is the remaining optimization the
compiler can offer on its own. Restricting it to chains whose intermediates
have one use guarantees that fusion never increases computation.

## Not built

- No fusion across an intermediate value with more than one use, because the
  fused form would have to recompute or expose it.

## Consequences

- Fusion rules are expressed against ExternFn identities that are stable across
  compilation units, not against per-graph identifiers.
- A fused ExternFn has the same interface obligations as the ExternFns it
  replaces.

## Open questions

- The rule language for describing a fusable pattern and its captured
  parameters.
