# RFC-0040: The compiler chooses an ExternFn's instance

Status: Accepted
Date: 2026-09-16
Extends: RFC-0011, RFC-0019, RFC-0027

## Ruling

An ExternFn has a list of instances, numbered. The concrete instances
come first, in declaration order; the generic instance, whose type is the
function's own, comes last when the function has one. A `Monomorphize`
member, an instance of a shared signature, and the plain body of a
function with no `Monomorphize` are all entries of that one list: the
registry hands the compiler the instance types and the runtime the
handlers as the same list in the same order.

The checker settles every call on one number. A function with no concrete
instance is settled at its instantiation. A function with concrete
instances opens a choice per call, settled by exclusion as RFC-0027 says,
and settled on the generic instance when the call's type has excluded
every concrete one. A choice still pending once the body is checked is
tried again with every integer literal at its default width. A call whose
type stays too open to choose has no callee in the resolution, and the
lowering treats it as it treats any call it cannot resolve.

The lowering writes the number into the IR: `Callee::Extern { id,
instance }` is a call of an ExternFn, `Callee::Direct(id)` a call of a
function with a body, and a cast carries its instance the same way. The
runtime indexes the handler list with the number and runs; it matches no
types.

Two concrete instances of one function whose types unify are refused when
the registries combine, so a frozen call type matches at most one.

## Rationale

Until now the checker settled a choice and forgot which instance it had
settled on; the interpreter recovered it at every call by matching the
call's type against every instance's type, and a `Monomorphize` function
never opened a choice at all — its members were matched only at run time.
The fact was known at compile time and was recomputed, by a slower
mechanism, at run time. Recording it is the whole change.

A number, not a name. The instance is identified by its position in a
list both sides hold; there is no mangled name because nothing links by
name — no ABI, no linker, one process that built both lists.

The generic instance is a fallback, not a candidate. RFC-0027 settled a
choice by exclusion among instances that never overlap; a generic instance
overlaps every concrete one, so it is not in the candidate set. It runs
when the candidates are exhausted, which is still exclusion, never search.

## Not built

- The printer shows a call's function and not its instance number, so a
  MIR snapshot does not record which instance a call settled on; the
  interpreter tests carry that fact.
- No instance chosen by more than one variable, no specificity between
  concrete instances: RFC-0027's limits stand.
- Instances discovered from the types in a signature — every `X<I>`
  named by a registry getting its own instance of every function over
  `X<T>` — are the next design; this RFC gives them a list to enter.

## Consequences

- `acvus-extern`: `ExternEntry`, `MonoHandler`, `MonoInstance`, and
  `select` are gone. `Instance`, `Instances { concrete, generic }`,
  `ExternFn { decl, instances }`; `Handlers` maps a name to
  `Vec<ExternHandler>`; `Contribution::instances`. `Externs::combine`
  refuses two concrete instances of one function whose types unify.
- `acvus-extern-macro`: `#[extern_fn]` emits `Instances`; a plain function
  is `Instances::generic`.
- `acvus-mir`: `ty::Instances` on `FnKind::Extern` and `Scheme`;
  `InstanceChoice`, `ChoiceState::Settled(index)`,
  `Solver::settle_pending_choices`, `Solver::settled_instance`;
  `Callee::Extern`, `CastKind::Extern::instance`; the type checker's
  `direct_calls` and `operator_calls` carry a `Callee`.
- `acvus-interpreter`: `Executable::Extern(Vec<ExternHandler>)`; a call
  indexes it.
