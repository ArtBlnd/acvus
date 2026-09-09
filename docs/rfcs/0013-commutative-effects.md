# RFC-0013: Commutative effects

Status: Accepted
Date: 2026-09-10
Supersedes: none

## Ruling

An ExternFn author may declare that the function's effect commutes: two
calls of commutative functions, in either order, are the same program.
Commutativity is a second axis of `Effect`, independent of the chain
`Pure < Idempotent < Opaque` of RFC-0008. `Pure` commutes by definition;
every other level may be declared commutative or not. The default is not.

    Effect = { reissue: Pure | Idempotent | Opaque, commutes: bool }

The join of two effects is the join on each axis: the higher level, and
commutative only if both are. A function's effect is the join of its calls,
as before.

Commutativity is used by the lowering of RFC-0007 alone. A run is a maximal
sequence of commutative calls that are neighbours on the `Order` chain. A
run is lowered as an `anyorder` block would be: each call takes the run's
entry `Order`, and one `merge` at the run's end yields the order that
follows. Only a call on the chain can end a run; a Pure call has no
`Order` and stands nowhere on the chain, so it neither joins a run nor
breaks one.
Nothing else reads the axis. A commutative call still keeps its place
against every non-commutative call, and a commutative call is not thereby
re-issuable: whether it may be suspended and issued again is the reissue
axis, as before.

The declaration is the author's fact about the outside world, like purity.
The compiler does not verify it, and a wrong declaration is a wrong
library. An executor may look for a counterexample the way RFC-0007
allows for a block: run a run's calls in more than one order, issue an
Idempotent call twice, and compare what the program observes. Two
outcomes are compared up to a renaming of fresh sources, which identity
already marks, so two fresh values in swapped positions are not a
difference. What lies outside the program is compared only where the
test supplies a world it can reset.

## Rationale

Order matters for an IO call only when someone can observe it. Whole
classes of real calls have no observer of order: a call that yields a
fresh value each time (an identifier, a random draw, a sampled model
completion), a read of immutable content, a write into a store built to
merge in any order. Their authors know this, and the script author does
not need to repeat it at every use with an `anyorder` block.

The chain of RFC-0008 was chosen because pure-but-not-suspendable is not a
real combination. Commutativity is a real second axis: a counter increment
commutes and is not re-issuable; a put to one key is re-issuable and does
not commute; a fresh identifier is both. Each of the four combinations
exists, so the two facts are declared separately.

The library is curated. The macro of RFC-0009 exists so that an ExternFn
is written once, by someone who knows the function, and that person
declares what the function is. The cost of a wrong declaration lands where
the knowledge was, which is the only place it can be checked.

## Not built

- No inference of commutativity from types, not even from a fresh identity
  in the return. A fresh identity makes the result unobservable in order,
  but the declaration stays explicit; a derived fact would be one more
  thing a reader has to know to read a signature.
- No per-pair or per-argument commutativity (two puts to different keys).
  That is a fact about arguments, not about a function, and it is what
  `anyorder` is for.
- No reordering of a commutative call across a non-commutative one.

## Consequences

- `Effect` carries two axes; its solver bounds, joins, and display carry
  both. The type of a function shows `commutative` when it holds.
- The ExternFn declaration takes `commutative` next to its level:
  `#[extern_fn(effect = idempotent, commutative)]`.
- The lowering of a run of commutative calls is the lowering of an
  `anyorder` block around that run.

## Open questions

none
