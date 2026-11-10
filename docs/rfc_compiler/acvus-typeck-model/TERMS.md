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
| Suspended constraint | waits for one class's head, then branches and adds constraints | no | a projection, a receiver's adjustment, `?` |
| Adjustment | a record of what solving came to | it is no constraint | a deref, a receiver's `&`, a reborrow, any subtyping the language gives, a declared function's call |

An adjustment is an output of solving and never an input to it: no constraint reads one, and
it decides no head. A later stage reads it to write out what the text leaves unwritten. What
makes something an adjustment is that it leaves a trace in the code that runs: a `*`, a `&`, a
cast, a call. A domain, a fallback, a join and the connecting of parts leave none, and are no
adjustments. So subtyping the language gives is recorded though the two types it goes between
would tell it again, and a declared function is recorded as the call it is.

## Conversions

A declared conversion is of one of two kinds, and they are chosen differently.

| Kind | What it is | Chosen by | When |
|---|---|---|---|
| View | Both ends are references, and the parameter writes its class node with a distinct hole at each part below, as `&Vec<T> -> &[T]`. | class nodes | before the argument is decided below its class node |
| Value conversion | any other | the argument's whole type | once the argument is decided at every depth |

A view takes any value of its class node, so choosing it early is never contradicted, and what
a part takes through it comes from the text. A value conversion runs a function that makes a
new value, so none is chosen while a type is undecided, and it gives type inference nothing:
it decides none of the argument's types.

Rung 2, which gives a class the head its upper bounds give, looks at no conversion.

## Other rulings

- `take(Some(return))` with `take(Option<u8>)` is refused: `!` stands at a part, and no change
  is made at a part.
- `&{x, y}` does not go to `&{x}`. A reference points at the value as it lies, so no field is
  dropped behind one.
- Where two heads both go to every upper bound of a class, the class is refused, and neither
  is preferred.
- The fallback of a literal's domain and the fallback to `!` are two rungs, the domain's first.
  `!` says nothing at all is known, and what a literal falls back to may yet give a class a head.
- A head is decided at a fixpoint: a class waits until nothing more can enter it.

## Where a declared type is met

The text writes no type: not on a `let`, and not on a parameter. A declared type comes only
from a signature given from outside, so a value meets one in two ways:

| Way | What is recorded |
|---|---|
| It is passed as an argument. | A change to the parameter's type, which is the upper end. |
| It is the result of a call. | The result's type enters as a lower bound. |

No value enters a declared type as its upper bound. `e as T`, once the language has a way to
write `T`, is the passing of `e` to a function `T -> T`, and adds no third way.

## A block decides alone

A standalone block `{ e }` decides everything inside it by itself, fallback included, and
gives what is outside only the finished type, as a lower bound. Nothing outside reaches in.
It is a call of a function whose result type is what its body comes to.

`let x = { 0 }; f(x);` with `f(u8)` is refused: `0` falls back to `i32` inside the block, and
no change goes from `i32` to `u8`. `{ None }` is `Option<!>`, and its payload is compared with
what it meets outside, and not merged.

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

Whether a class is a reference is whether anything has given it a target. This is a question
about parts, which a class holds before it is decided, and not one about its decided head.
A projection `p.f` asks it of `p`, to know whether `f` is a part of `p` or of what `p` refers to.

## Not settled yet

- Whether subtyping the language gives is taken once, before the chain of declared functions,
  as the model does, or also where a value is passed to each function of the chain.

- How a connection knows that nothing more will be given to a class, where a part may still
  come through a declared function not yet chosen.
- The head of a class that only parts are recorded on, as in `let a; a.x`.
- A refusal naming the connection that merged two classes.
- How many references a projection reads through, now that a target may be a reference.
