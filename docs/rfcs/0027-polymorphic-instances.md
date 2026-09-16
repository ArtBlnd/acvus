# RFC-0027: A polymorphic instance of a shared signature

Status: Accepted
Date: 2026-09-15
Extends: RFC-0019
Extended by: RFC-0040

## Ruling

An instance of a shared signature may be declared at a type scheme, not
only at a concrete type: `into_iter_list<T>(List<T>) -> Iter<T>` is the
instance of `iter::into_iter` for every `List<T>`. Two instances of one
signature whose types unify are refused, so the head constructor of the
instance variable's type picks at most one instance.

A signature's other variables are determined by the instance. The
signature `iter::into_iter<C, T>(C) -> Iter<T>` says nothing about how `T`
follows from `C`; the instance for `List<T>` does. A call of the signature
is checked as the signature's type under a choice among the instances:
when the arguments checked so far leave one instance that can still match
the call's type, the call's type is unified with that instance's, and the
variables the signature left open take the instance's values. A call that
no instance can match is a type error at the call; a call that several
still match is a call whose types are not resolved, reported as any other.

A declared bound (RFC-0011) is a set of type schemes. It admits a type that
has the shape of one of them; the bound two variables share when they are
unified is the pairwise unifier of their schemes.

An instance declared as a cast (RFC-0023) registers a coercion from its
parameter scheme to its return scheme, resolved through the signature: the
coercion is a call of the signature, settled on an instance as any call
of the signature is.

The instance a call settles on is written into the call (RFC-0040); the
runtime runs it without looking at a type.

## Rationale

RFC-0019 fixed an instance at one concrete type because `clone` and `eq`
needed no more; iteration does. `List`, `Array`, and `Deque` each know how
to yield their elements, and a script wants one name — `into_iter(xs)`,
`as_iter(&xs)` — whatever the container, with the element type following
from the container. Writing `iter_array`, `into_iter_array`, and a cast per
container is the same fact written once per container per constructor.

The choice is resolved by exclusion, not by search: an instance is dropped
only when the call's type already contradicts it, and the last one left is
taken. This never guesses; a script that leaves the container open gets an
unresolved-type error, as it would for any unresolved variable. Checking
arguments left to right against the signature's type, and settling the
choice after each, is what lets a lambda later in the argument list see the
element type the container fixed.

## Not built

- No instance chosen by more than one variable. The instance variable is
  the signature's first type variable, and the instance is keyed by the
  type at that variable alone.
- No overlapping instances and no specificity ordering. `List<Int>` and
  `List<T>` are two instances of one signature only if the registries are
  wrong.
- No generic functions in the language over `C: into_iter`; a script calls
  the signature at a container it names.
- No search: a choice is settled by exclusion only, never by trying an
  instance and backtracking.

## Consequences

- `TyVarBound::OneOf` holds type schemes; a bound admits a type by shape,
  and the meet of two bounds is the set of pairwise unifiers.
- An Extern function carries the schemes of its instances; the solver holds
  a choice per instantiation of such a function and settles it whenever an
  argument has been checked and before bounds are verified.
- `Externs::combine` accepts an instance whose type has the signature's
  shape, refuses one that unifies with an instance already collected,
  registers a cast instance's coercion against the signature's name, and
  collects every registry's signatures before any registry's instances.
- The iterator registry declares `iter::into_iter` and `iter::as_iter` with
  instances for `List` and `Array`; the deque registry adds instances for
  `Deque`. `iter`, `iter_array`, and `into_iter_array` are gone. `std::list`
  becomes a signature with cast instances for `Array` and `Deque`: a deque
  demotes to a list, never the reverse.
