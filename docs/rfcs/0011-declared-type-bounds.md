# RFC-0011: Declared bounds on type variables

Status: Accepted
Date: 2026-09-10
Supersedes: none

## Ruling

A declaration says, for each of its type variables, what the variable may
become:

    TyVarBound = Any | OneOf([Ty, ...])

The bound is declared next to the polymorphic type, by position, and it is
the only channel by which a declaration constrains a variable. An ExternFn
declares bounds for its type; an extension type declares bounds for its
type parameters. Inference itself declares nothing: every other constraint
on a variable arises from unification.

The solver carries the declared bound on the variable through resolution.
Unifying two variables meets their bounds and fails when nothing satisfies
both. Binding a variable keeps its bound. Freezing a variable verifies the
bound against the resolved type, and the checker reports a violation at the
use that instantiated the declaration. Nothing is checked earlier, and
nothing is defaulted.

An ExternFn whose Rust body is generic over a type in a finite set is
declared with `Monomorphize<(T0, T1, ..)>` on that parameter. The
declaration is one function whose variable is bounded `OneOf`; the handler
is compiled once per member, each carrying the function's type with the
variable set to that member, and the runtime runs the member whose type
matches the resolved call type. A script sees one name.

## Rationale

The solver already carries a finite-domain bound on a variable and meets
bounds when it unifies. Declaring bounds makes that mechanism serve declared
polymorphism too, instead of a second mechanism at the call site. An overload
set resolved by trial at each call has to decide before an argument whose
type is not yet known resolves; a bound on a variable is checked when the
variable freezes.

Rust has no overloading, so a body over several concrete types is written
once against a trait, and the declaration lists the types. Instantiating
the body per member is what a Rust caller would do; here the acvus type
system does the choosing.

## Not built

- No capability bounds (`Cloneable` and kin). Nothing declares them and
  nothing checks them.
- No bound on effect or length variables. Their solvers carry their own
  ranges (RFC-0014, array lengths); nothing declares a finite set of either.

## Consequences

- A polymorphic function type reaches the checker as a scheme: the type and
  its bounds. A local function's scheme has no bounds.
- A freeze can fail because a bound is violated, distinctly from failing
  because a variable is unresolved.
- A registered ExternFn carries one handler per member, each with its
  instantiated signature; a call's resolved function type selects the one
  it matches, wherever the variable occurs.
- A declaration's generic parameter carries the kind of the variable it is
  — `Var<kind::Type>`, `Var<kind::Effect>`, `Var<kind::Length>`,
  `Var<kind::Identity>` — and `Monomorphize<(..)>` is the type kind with a
  finite set. The kind is the bound's argument, not the bound's name.

## Open questions

- Whether a bound should admit open shapes (`List<T>` for any `T`) rather
  than only concrete types.
