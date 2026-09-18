# RFC-0001: Enrichment pipeline

Status: Accepted
Date: 2026-04-04
Supersedes: none

## Ruling

acvus is an enrichment pipeline, not a lowering compiler. Parsing discards
surface syntax and nothing else. Every pass after it records in the MIR a
property that was implicit in the source, and no pass removes a property.
Information is lost at one place, the execution boundary, and validation
stands immediately before it.

## Rationale

A lowering chain loses information at every stage, so a wrong result cannot be
attributed to a stage without reconstructing what each stage discarded. With
one discrete cliff, a wrong result is either a wrong MIR or a wrong
materialization of the MIR.

The pipeline separates two languages. Rust is the implementation language and
acvus is the composition language. The composition language describes the
implementation language's semantics through signatures, so a composition is
optimized without reading a Rust body. The same split holds between the
template surface and the expression surface inside acvus.

## Not built

- No lowering dialects between the MIR and execution. A lower dialect would
  reintroduce loss at each stage, and a wrong result would again name no
  stage.
- No runtime type tags. The MIR accumulates the type information that makes
  erasure at the execution boundary sound.

## Consequences

- A pass may add facts to the MIR and may not remove facts another pass could
  still use.
- Validation runs on the MIR as it stands at the execution boundary.
- A new feature is admitted only in the enrichment direction: it answers
  "does this lose information?" with no.

## Open questions

- The subsumption direction: whether a language T whose type system subsumes a
  language X can host X at no cost, so that one middle end serves every easier
  language. A direction; nothing here establishes it.
