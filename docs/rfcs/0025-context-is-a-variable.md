# RFC-0025: A context is a variable of the body that touches it

Status: Accepted
Date: 2026-09-15
Supersedes: the context clauses of RFC-0018 (`Take`/`Assign` on a context
at every use; the lend of a context through a temporary) and the
`ContextLeftTaken` rule of the borrow checker.

## Ruling

A body that names `@x` owns `x` as a local variable for the length of its
run. The page is touched at four places and nowhere else:

- **Entry.** Every context the body names is fetched from the page into
  its variable: `Fetch { dst, context }`.
- **Exit.** On every return, each such variable is committed back:
  `Commit { context, value }`, where `value` is a `Take` of the variable.
- **Around a call.** A call whose summary (RFC-0017) touches `@x` — the
  callee's own summary joined with the summaries of the function values it
  is passed — is bracketed: `Commit` of the variable before it, `Fetch`
  into the variable after it. A call whose summary does not touch `@x`
  touches nothing.
- **Around a spawn.** The `Commit` goes before the `Spawn`, the `Fetch`
  after the `Eval` of its handle. Between them the variable is moved out.

Inside the body, `@x`, `@x.f`, `&@x`, `&mut @x`, `@x = v`, and `let a =
@x` are the variable rules of RFC-0018 on the local `x`, with nothing
added: a read of a non-primitive moves out, a borrow is a `Ref` to the
variable's storage, an assign re-initializes, and a borrow that is live
excludes every other touch. A context left moved out at an exit is a
use-after-move at the exit's `Take`; a context touched while a spawn that
touches it is in flight is the same error.

`Fetch` and `Commit` carry no path: a context is fetched and committed
whole. The page stores whole values; a field of a context is a field of
the variable.

## Rationale

The three defects found on 2026-09-15 have one cause: the context was not
one variable. A lend of `@x` borrowed a temporary, so the borrow checker
saw no borrow of `x` and an assign to `@x` during the lend was silently
overwritten by the write-back. A call whose closure wrote `@x` was not a
touch of `x`, so the SSA pass forwarded a value across it and the caller
read a stale constant. And the rule that a taken context be assigned back
was a second rule for what is, on a variable, use-after-move.

With `x` a variable, every one of those is caught by a rule that already
exists for variables, and the page ops are confined to the four places
where ownership actually changes hands. The bracket around a call is what
makes the page the medium between bodies: a callee's `Fetch` finds the
value because its caller committed it first. The summary that decides
which calls are bracketed is inferred bottom-up over the call graph and
rides in the function type (RFC-0017); no declaration is added.

## Not built

- No path on a page op. A body that needs one field of `@user` fetches
  `@user` whole; the journal's field take and field set are removed.
- No page op on the `Order` chain. A context read and write are
  `Reissue::Pure` (RFC-0017), so a body that only touches contexts has no
  chain to ride. Passes that move instructions keep a page op in its
  original order against the calls and page ops of the same context, by
  the summary; that is one rule in the reorder pass and the existing rule
  of commute and dead-store elimination, re-keyed on `Fetch`/`Commit`.
- No fetch of a context the body does not name. A context touched only by
  callees is committed and fetched by the callees.
- No reference to a context in the IR. `RefTarget::Context` is gone; a
  `Ref` targets a variable or a parameter.

## Consequences

- `InstKind::Fetch { dst, context }` and `InstKind::Commit { context,
  value }` replace `Take`/`Assign` on `RefTarget::Context`; `RefTarget` is
  `Var | Param`.
- Lowering pre-scans the body for the contexts it names (not those of
  nested lambdas, which are their own bodies), fetches them at entry into
  variable slots, commits them at every return, and brackets each call
  by the summary of its callee type joined with its function-typed
  arguments.
- The SSA pass treats a context variable as any local; its context
  machinery (entry take, write-back after merge, forwarding across the
  page) is removed.
- The borrow checker's `ContextLeftTaken` is removed; the exit `Take` is
  checked by the move checker.
- The interpreter's `Fetch` is the page's `take` (absent is a panic) and
  `Commit` its `set`; the journal keeps `take`, `set`, `take_writes`.
- The reorder pass keeps `Fetch`/`Commit` of a context in order against
  each other and against every `FunctionCall`/`Spawn`/`Eval` whose summary
  touches that context.
