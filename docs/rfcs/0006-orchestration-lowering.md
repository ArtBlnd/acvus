# RFC-0006: Orchestration lowering

Status: Accepted
Date: 2026-03-26
Supersedes: none

## Ruling

A declarative orchestration spec is lowered by generating acvus source. Spans
of user-written content are mapped back to the spec field they came from, so
an error in user content is reported on that field. An error inside generated
glue is a lowerer bug and is a panic.

The language is frozen for orchestration: the type system, the syntax, and the
IR do not change to serve it. Every orchestration capability is an ExternFn.

A bot and its prompt are baked into one unit.

## Rationale

Type checking and ordering relations already verify generated source, so
generating source is safe and every static analysis applies to it unchanged.
Freezing the language keeps that guarantee: an orchestration feature that needed
new syntax would be a feature the analyses do not know.

Baking removes three problems at once. There is no memory ownership question,
no scoping question, and no display boundary question, because there is no
seam between bot and prompt for them to live on.

Axioms the design rests on: history is a sequence of messages with every
snapshot kept; display is a pure function from history to a view; the pipeline
is iterator composition; inside a map the author is free, while the topology
is composed only.

## Not built

- No prompt swapping across bots. A swap protocol would reopen the memory,
  scoping, and boundary questions that baking closed. It is added only if
  demand is confirmed, and then as a minimal protocol.
- No language extension for orchestration.

## Consequences

- LLM calls, input, and configuration all arrive as ExternFns or injected
  context, never as language constructs.
- Generated glue is held to a stricter standard than user content: it must be
  correct by construction.
- Display composition permits map and filter but not collection, so a display
  cannot break the safe-composition boundary.

## Open questions

- Whether prompt swapping has real demand.
- How scoping of journal state is experienced by the user.
- The iterator structure of a streaming tool-call loop.
