# Terms of the type checker model

What the design discussion has agreed, kept here so that it is not derived again. This file is
temporary: once the model implements these, it is deleted and its content goes into the docs.

## Join is not subtyping

These are two relations, and neither is explained by the other.

| | Join | Subtyping |
|---|---|---|
| Where | where values meet in one class: what enters a class | where a value is passed: the `x` of `f(x)` |
| What it gives | the head of the class, from every head that enters it | whether the argument's type may be used as the parameter's |
| Objects | `{x}` and `{y}` meet as `{x, y}` | toward fewer fields: `{x, y}` goes to `{x}` |

What a join merges is bounds, and not members. Whether the outcome looks wider or narrower
follows from what a bound means for the constructor, and is no property of the join.

| Constructor | Its bound | Merging two |
|---|---|---|
| Object | a field is there | widens: more fields are there |
| Function value | a closure's bounds: those of its parameters | narrows: the bounds of every closure must hold |

A closure value can only be instantiated. Using it checks two things, which are refused at
different places: that the arguments satisfy the parameters' bounds, and that the bounds
inside the closure hold once it is instantiated with what is outside.

That `{x}` and `{y}` meet as `{x, y}` is intended. It is not to be derived from subtyping, and
it is no error that the two go opposite ways for objects.

Subtyping comes from two sources, and what tells them apart is who can give it:

| Source | What it is |
|---|---|
| The language | A rule over structure. It is a family no list of functions covers: `{x, y}` to `{x}` is one function for each set of fields. |
| Outside | A declared function `P -> R`: finitely many, each a signature with holes. |

## Subtyping the language gives

| Type | Goes to |
|---|---|
| `!` | any type |
| a reference | `&mut T` to `&T` |
| an object | one with fewer fields, where no reference is above it |
| an enum, an `Error` | one with more variants |
| an array behind a reference | `&[T; n]` to `&[T]` |
| references to a reference | `&ⁿ U` to `&U`, for each count `n` of references |

Each of these loosens one index of the head and leaves what is below as it is: a mutability,
a set of fields, a set of variants, a length, a count of references. None is one function: each
is a family, one for each value of its index, which is why only the language can give it. Where
a value is passed they are looked at together, in one step, so `&&[T; n]` goes to `&[T]`.
An index is loosened only where the types do not match as they are written.

A function value does not go to one with more closures where it is passed. Its closures carry
the bounds of their parameters, so loosening there would have to follow variance, and it is
invariant instead. Where function values meet in a class, their bounds are merged, as a join does.

Subtyping the language gives is looked at first. A declared function is looked at only where
it does not go.

A value passed through declared functions passes through a chain of them, and no other
declared function is taken on the way into one: the chain is every function called there.

`a = f(x)` has no subtyping at `a <- f(x)`. The result enters `a` as a lower bound, and `a`
takes the join of what enters it.

## The three primitives

| Primitive | What it is | Gives a head while it waits | Examples |
|---|---|---|---|
| Bound | a relation of two classes that leaves them two | yes | enter, change, equal |
| Suspended constraint | waits for one class's head, then branches and adds constraints | no | a projection, a receiver's adjustment, `?`, `as _` |
| Adjustment | a record of what solving came to | it is no constraint | a deref, a receiver's `&`, a reborrow, any subtyping the language gives, a declared function's call |

An adjustment is an output of solving and never an input to it: no constraint reads one, and
it decides no head. A later stage reads it to write out what the text leaves unwritten. What
makes something an adjustment is that it leaves a trace in the code that runs: a `*`, a `&`, a
cast, a call. A domain, a fallback, a join and the connecting of parts leave none, and are no
adjustments. So subtyping the language gives is recorded though the two types it goes between
would tell it again, and a declared function is recorded as the call it is.

## Declared functions

A function declared from outside is a row of a table: a name, and a signature `P -> R`. The
language states one rule of them, and a row adds none.

> An argument goes through a row `P -> R` by being passed to `P`, and what the row gives being
> passed to the type given.

So passing through a row is being passed, twice, and what holds of being passed holds of each:

| What follows | From |
|---|---|
| A value goes through a chain of rows. | What a row gives is passed to the type given, by the language or through a row again. |
| No row is taken on the way into a row. | The argument is passed to `P` by what the language gives alone. |
| What the language gives is taken at each row, and not once before them all. | Each row is a passing. |
| A row is looked for only where the language gives nothing between the argument and the type given. | Subtyping the language gives is looked at first. |
| A row's signature is made anew wherever it is taken. | It is a signature. |

Exactly one way through rows must go, or the program is refused.

A row is read as any type given is: as far as it is written. The argument is read that far, and
its parts are the types the row writes for them, at every depth the row writes. So a row that
writes only a constructor, as `&Vec<T> -> &[T]`, is told as soon as the argument's constructor
is read, and a row that writes deeper, as `(u8, T) -> Vec<T>`, is told later. No row is taken
before every row is told of, as one not told of yet may be a second way.

There are no kinds of rows. A row once called a view is one whose parameter writes little, and
that is all that made it chosen early.

A row gives type inference nothing: the head it holds a part of the argument against is no
head asked of that part. The rung that gives a class a head from above looks at no row.

## Other rulings

- `take(Some(return))` with `take(Option<u8>)` is refused: `!` stands at a part, and no change
  is made at a part.
- `&{x, y}` does not go to `&{x}`. A reference points at the value as it lies, so no field is
  dropped behind one.
- Where two heads both go to every upper bound of a class, the class is refused, and neither
  is preferred.
- The fallback of a literal's domain and the fallback to `!` are two rungs, the domain's first.
  `!` says nothing at all is known, and what a literal falls back to may yet give a class a head.
  The head a class's parts make comes before both: it is what a written head would be, and a
  written head is there before any fallback.
- A head is decided at a fixpoint: a class waits until nothing more can enter it.

## Where a declared type is met

The text writes no type: not on a `let`, and not on a parameter. A declared type comes only
from a signature given from outside, so a value meets one in two ways:

| Way | What is recorded |
|---|---|
| It is passed as an argument. | A change to the parameter's type, which is the upper end. |
| It is the result of a call. | The result's type enters as a lower bound. |

No value enters a declared type as its upper bound. `e as T` is the passing of `e` to a
function `T -> T`, and adds no third way. The one `T` the text can write so far is `_`.

## `e as _` decides alone

`e as T` is the passing of `e` to a function `T -> T`. In `e as _`, the `_` writes the type `e`
comes to by itself: every class `e` makes is decided with nothing outside reaching in, fallback
included, and that finished type is the `T`. So what is outside is given a declared type, as a
lower bound, and compares with it.

The `_` is no hole. A hole is one class with what is passed to it, and `e as _` would then do
nothing.

A type decided at every depth is compared with whatever it meets, as a declared type is, and
is changed by nothing. So no declared type is written anew for the `_`: the type of `e`
enters as it is, once nothing in it is undecided.

| Program | Outcome |
|---|---|
| `let x = 0 as _; f(x);` with `f(u8)` | Refused: `0` falls back to `i32`, and `i32` does not go to `u8`. |
| `None as _` | `Option<!>`. Its payload is compared with what it meets outside, and not merged. |
| `0 as u8` | `u8`: the type is written, so the literal is given its head from above. |

What decides alone is what `e` makes, and not what `e` names. A class made outside `e` is
decided as the rules outside decide it, and `e as _` waits for it:

```text
let x = 0;
let y = x as _;
f(x);            // f(u8)
```

`x` is `u8`, by the passing to `f`, and `y` is given `u8` once `x` is decided.

`as _` follows any expression: an `if`, a `match`, a call. A block `{ e }` is a scope and an
order of statements, and does nothing to types. This replaces the earlier rule that a
standalone block decides alone.

## A hole is where types meet

An argument's part and the hole it is passed to are one class. A class's head is the join of
everything that enters the class once every connection that may merge it has been made.
A part of a declared type is compared with, and not merged. This
holds of a hole as of any part: a hole that corresponds to a declared part is compared with it,
and does not become it.

## Reading and deciding

A fact that nothing still to come can change may be read from a class whose list of what
enters it is not complete. Deciding a class happens once, after everything has been gathered,
and is never revised. A decision made early would report whatever comes later against it,
where a decision made once names everything that met.

A fact read carries what settled it, so that a refusal that follows names that too.

Refusals are gathered. One cause may give several refusals, and each is kept where it is true.

A refusal names what made the classes it is of meet: every connection that made each one
class, and every connection that made one class of what holds it as a part. It names all of
them, and not the fewest that would do: which of several connections merged two classes
depends on the order they were made in, and that all were made does not. The connections are
kept as they were made, each with the two classes it connected, so a later stage may name
fewer.

What a refusal names is an output, as an adjustment is. No rule reads it, and it decides no
head. A rule says what it refuses and of which classes, and what settled those classes is
gathered once, after everything has been connected and decided.

## References

`&&T` is a type, and a type other than `&T`. A reference may be the target of a reference.

A lend `&p` records `?0 = &?1` at once, with `?1` being `p` itself. It reads nothing of `p` and
waits for nothing. `?0` has no lower bound, and its upper bound is what it is passed to.
So a lend is no bound: it writes a reference whose part is `p`, as `Some(p)` writes an option
whose part is `p`. The bounds are three: enter, change, and equal.

Reading through a reference, `&&U` to `&U`, is subtyping the language gives, so it happens
only where an argument is passed, and never in a join. It copies the inner reference out, so
the inner reference must be one that can be copied: a shared one. The outer one may be
exclusive: `&mut &U` goes to `&U`.

A projection `p.f` reads through every reference above the part, however many, and derefs once
for each. It asks whether `p` is a reference, and what `p` refers to is asked the same in turn,
so no count is stated anywhere.

What follows from types being matched as they are written:

| Program | Outcome |
|---|---|
| `f(&r)` with `f(&T)` and `r: &u8` | `T` is `&u8`. Nothing is read through, as `&&u8` matches `&T` as it is. |
| `f(&r)` with `f(&u8)` and `r: &u8` | Accepted: `&&u8` goes to `&u8`. |
| `f(r)` with `f(U) -> &U` and `r: &T` | The result is `&&T`. |
| `if c { &r } else { &x }` with `r: &u8`, `x: u8` | Refused: `&&u8` and `&u8` do not join. |
| `g(&r, &x)` with `g(T, T)` | Refused, for the same reason. |

Folding `&&T` into `&T` once checking is done is a matter of how a checked type is represented,
and no rule of the type checker.

This replaces the earlier rule that no reference is the target of a reference, under which a
lend had to wait to learn whether its place was a reference.

## A head with parts is made from its parts

A head that has parts is made from those parts, and a head with none is made from nothing.
So a class nothing has given a part to has a head with no parts, and a class a part is recorded
on has a head with a place for it.

A class nothing gives a head takes the head its parts make. Where nothing enters a class, or
the join of what enters it is `!`, and no head is asked of it from above, it is:

| Parts named of it | Its head |
|---|---|
| fields `x`, `y` | `{x, y}` |
| payloads of variants of the enum `A` | `A`, with those variants |
| the payload of `Some`; of `Ok`, `Err`; of variants of an `Error` | `Option`; `Result`; that `Error` |
| a target, a component, an element | none: the name does not say which reference, how many components, or what holds the element |
| names of more than one of these | none |

So `let a; a.x = 1u8;` gives `a` the type `a = {x: 1u8}` gives it, and `let A::V(x) = a` makes
`a` an `A`. A part only read counts as one written does: `let a; a.x` is `{x: !}`. Where the
parts make no head the class is `!`, and the part is refused as having no place.

A part named of a class is one named on it, or on a class with no head that it enters, in
turn: the head it takes enters that class, so the part must be there. So a `let` between
changes nothing: `let a; let b = a; b.x = 1u8;` makes both `{x: u8}`.

A head that entered is not widened by a part named: `let a = {y: 1u8}; a.x` is refused.

The name of a payload says which type its variant is of, as the pattern that names it does. So
`let Some(x) = a` is held only by an `Option`, and not by an enum that has a variant `Some`.

Whether a class is a reference is whether anything has given it a target. This is a question
about parts, which a class holds before it is decided, and not one about its decided head.
A projection `p.f` asks it of `p`, to know whether `f` is a part of `p` or of what `p` refers to.

## Not settled yet

### Calls and function values

Left to the end, and to be explained before anything is derived. The model is complete without
them, and they are added without changing what is here. What is known to need ruling:

- Recursion: a lambda that takes itself as a parameter.
- Brand types: which brand a returned value has, and whether a brand may be a hole.
- Where a call makes types equal and where one only enters the other (the transparency rule).

What depends on calls waits with them: a receiver's adjustment, and a use with several
candidates.
