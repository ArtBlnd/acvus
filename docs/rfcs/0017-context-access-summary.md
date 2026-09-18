# RFC-0017: A function's type says which contexts it reads and writes

Status: Accepted
Date: 2026-09-10
Supersedes: none

## Ruling

A function's type carries a context summary: the set of contexts its
body may read and the set it may write. Reading `@x`, or lending it with
`&`, puts `@x` in the read set; assigning `@x`, or lending it with
`&mut`, puts it in the write set. The summary of a call is the summary of
its callee; a function's summary is the union of the summaries of its
calls and of its own accesses, closed over recursion in the same pass
that closes its effect. An ExternFn's own summary is empty (RFC-0014);
a call that passes a function value contributes that value's summary,
so a closure that reads a context is a read at every call that may run
it.

A pass that moves a call, or lets two calls share an `Order`, asks the
summary: a call may not pass a store to a context in its read or write
set, nor a load of a context in its write set. A call whose summary is
unknown is taken to read and write every context. Commutativity
(RFC-0013) stays a statement about calls against calls; the summary is
the statement about calls against the run's own loads and stores.

## Rationale

A context is a static variable, and a callee's access to one is a fact
about the callee that the caller cannot see in its body. Inlining makes
the access visible by copying the body, and SSA then orders it against
the caller's stores by value flow; a call that is not inlined leaves
the caller blind. The default rule, that every call reads and writes
every context, is sound and is what dead-store elimination and sinking
use; the commutative run of RFC-0013 takes the opposite, that a
commutative call touches nothing, which holds for an ExternFn and not
for a local function that reads a context. One summary the caller can
ask answers both.

A compiler that keeps globals in memory answers the same question with a
per-function mod/ref summary computed bottom-up over the call graph and
consulted by every pass that reorders, rather than by threading the global
through every call as a value, which grows with call depth and has no form
at recursion or at an indirect call. The effect term is computed in that
shape here, and the summary is computed with it.

The summary lives in the type, not in a side table, because a closure is
a value: which contexts a call may touch depends on which closure it was
given, and only the type follows the value.

## Not built

- No threading of contexts through calls as hidden parameters and
  results. That is what inlining does by copying, and it does not scale
  to depth, recursion, or a closure called by an ExternFn.
- No declaration of a summary on an ExternFn. Its summary is empty by
  RFC-0014, and a summary it could declare would be a second door to a
  context.
- No refinement of dead-store elimination or sinking by the summary in
  this ruling. They keep the sound default; the ruling obliges only the
  passes that today assume the opposite.

## Consequences

- The Fn type carries read and write sets of contexts in its effect term;
  the polymorphic phases carry them as terms with variables. A summary
  term unifies where and with the polarity the effect term does: where
  two function values join, at a branch or a phi, the joined type's
  summary is the union of both, and a variable left open closes to the
  empty set. Every function value originates in a closure whose summary
  the checker computed, or is relayed by an ExternFn; an ExternFn that
  returns a function value relays a summary variable from a parameter,
  and one that does not is rejected at registration.
- A commutative run (RFC-0013) admits a call only when no load or store
  between it and its neighbour on the chain names a context in the
  call's summary as the ruling states.
- A function's frozen type shows its summary; a host can read from a
  program's type which contexts it touches and how.

## Open questions

- Whether a field path (`@a.x`) is summarized as the whole context or as
  the path. The ruling says the context.
- Whether the sound default for dead-store elimination and sinking should
  later become the summary, once local functions that are not inlined
  exist in programs.
