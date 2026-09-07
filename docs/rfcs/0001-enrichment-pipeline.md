# RFC-0001: Enrichment pipeline

Status: Accepted
Date: 2026-04-04
Supersedes: none

## Ruling

acvus is an enrichment pipeline, not a lowering compiler. Parsing discards
surface syntax only. Every pass after it discovers a property that was implicit
in the source and records it in the MIR. Information moves in a U shape: one
drop at parsing, then accumulation until execution.

Information is lost at exactly one place, the execution boundary. Everything is
visible up to that boundary and validation stands immediately before it.

## Rationale

A lowering chain loses information at every stage, so a wrong result cannot be
blamed on a stage without reconstructing what each stage threw away. With one
discrete cliff, blame is always decidable at once: either the MIR is wrong or
the materialization of the MIR is wrong.

Decomposition in this pipeline is a homomorphism. The meaning of the pieces,
recombined, is exactly the meaning of the whole. That is what allows pieces to
be placed independently, including across execution topologies.

The pipeline separates two languages. Rust is the implementation language and
acvus is the composition language. Because the composition language describes
the implementation language's semantics through signatures, composition is
optimized without loss. The same pattern holds between the template surface and
the expression surface inside acvus itself.

## Not built

- No lowering dialects between the MIR and execution. A lower dialect would
  reintroduce gradual loss and undecidable blame.
- No runtime type tags. The MIR accumulates the type information that makes
  erasure at the execution boundary sound.

## Consequences

- A pass may add facts to the MIR and may not remove facts another pass could
  still use.
- Validation runs on the MIR as it stands at the execution boundary, on the
  most informed form of the program.
- Any new feature must answer "does this lose information?" and is admitted
  only in the enrichment direction.

## Open questions

- The subsumption direction: if a stricter language T's type system subsumes a
  language X, X can be implemented inside T at no cost, and one middle end
  serves every easier language. This is a direction, not a proven claim.
