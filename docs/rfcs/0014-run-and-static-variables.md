# RFC-0014: The run is the unit; a static variable outlives it

Status: Accepted
Date: 2026-09-10
Supersedes: RFC-0008

## Ruling

A program always terminates. One execution of it is a run, and the run is
the unit of everything the host sees: its context writes leave together
when it ends, and a run that does not end leaves nothing. There is no
state of a run between its start and its end that anything outside the
run can hold, save, or resume.

A context reference `@x` is a static variable: a place whose lifetime is
not bound to the run. Reading it is a load and writing it is a store; the
compiler may thread those through SSA to expose order, and nothing else
distinguishes a context from a local binding except that the host keeps
its value from one run to the next. A context therefore holds only what
the host can keep: a data type, or an extension type that declares how it
is written down. A value that carries an identity lives in a local
binding and ends with the run. A context carries no policy: not volatile,
not read-only. A value a script only reads is a function argument, not a
context. An ExternFn cannot reach a context; a script reads one and passes
the value.

Progress that must survive a run is written to a static variable by the
author, and the host runs the program again. Resumption is re-execution
from the state the last completed run left.

An ExternFn author declares an effect level, and the levels form a chain:

    Pure < Idempotent < Opaque

Pure calls have no effect and stand nowhere in the order of a run.
Idempotent calls keep their order against other effects, and issuing one
twice is the same as issuing it once. Opaque calls keep their order and
must not be issued twice. The default for an undeclared ExternFn is
Opaque, the top of the chain. A function's effect is the join of the
effects of its calls. The name of the term in code is `Effect`; RFC-0013
adds its second axis.

What the chain decides is whether a run may be started again after it
failed to end. A run whose effect is Idempotent may be re-run by the host
without asking: every effect it issued before failing will be issued
again, and that is the same as once. A run whose effect is Opaque may not
be re-run on the host's own judgment; an effect may have reached the
world without its record reaching a static variable.

## Rationale

Dumping a run in the middle needs a serialized form of everything the run
holds: values that wrap Rust payloads, closures, calls in flight, and a
version for all of it. Each of those is a decision with no good answer,
and every one disappears when the run is the unit. What the host keeps
between runs is what it already keeps: the values of static variables,
which are data. The author decides how fine the units are by how much one
program does; a crash costs at most one run.

A static variable that outlives the run is what a context already was in
the interpreter, which applied a run's writes at its end and never in the
middle. The ruling names that behaviour as the meaning.

The language has no concurrency primitive. With a single thread of
control, a static variable and an SSA-threaded context are the same thing
observed from two sides, and the simpler statement is the static one.

Two independent bits, pure and re-issuable, would admit "pure but not
re-issuable", which is not a real thing. A chain makes that combination
unwritable. The middle level is named for its cause, idempotence, not for
its consequence.

## Not built

- No dump of a run's state, no resume from a dump, and no version for
  either. The run is the unit.
- No yield statement, no suspension point, and no host-input call that is
  its own suspension. A call that waits on the host waits; a program that
  must hand control back ends and is run again.
- No context policy. A volatile context would be a second door to the host
  beside ExternFn; a read-only context is an argument.
- No context read/write declaration on ExternFns. The ordering they
  existed for is carried by RFC-0007.
- No identity-carrying value in a context. Its meaning across runs would
  be a source that outlives the program that made it.

## Consequences

- A run's context writes are applied by the host when the run ends, never
  before; the interpreter yields them as one result.
- The type of a context is a data type or an extension type with a
  declared codec; the checker rejects any other.
- The Fn type and UserDefined type arguments carry the effect as a term
  with variables in the inference phases; a variable inference leaves open
  resolves to the join of what is constrained below it, Pure when nothing
  is.
- A host that re-runs a failed run does so only when the program's effect
  is Idempotent or Pure; the program's type says which.
- Passes that reorder calls treat Idempotent exactly as Opaque.

## Open questions

- Which extension types declare a codec, and in what form.
