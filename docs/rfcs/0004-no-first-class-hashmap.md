# RFC-0004: No first-class HashMap

Status: Accepted
Date: 2026-03-31
Supersedes: none

## Ruling

acvus has no first-class HashMap type. Statically shaped JSON is received into
an Object through a deserializing ExternFn whose result type is inferred from
the expected type. Dynamically keyed data is a list of key-value pairs. An
order-preserving map is that list with key uniqueness enforced at insertion.

## Rationale

A map is an Object that has lost two distinctions: how many fields it has and
what they are named. A list of pairs represents dynamically keyed data without
losing either, keeps insertion order, and needs no new syntax; iteration and
pattern matching already apply to it.

## Not built

- No HashMap type in the type system.
- No index or field access syntax on a map. Lookup goes through the same
  functions as any list.

## Consequences

- Deserialization of structured input is an ExternFn whose return type is
  determined by the expected type at the call site.
- Key uniqueness is enforced by the insertion function, not by a type.
- Lookup cost is linear in the number of keys; this is accepted for the
  intended DSL workloads.

## Open questions

- The exact inconvenience level of insertion, chosen so that Object remains
  the default choice.
