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

A row gives no candidate to type inference: the head it holds a part of the argument against
is no head asked of that part. The rung that gives a class a head from above gathers the heads
asked of the class, and no row asks one. Whether a candidate would be taken is judged by
being passed as it is, rows included: there is one rule of being passed, and a class supposed
to have a head is passed as a class that has it.

| Program, with the row `up: A -> B` | `x` |
|---|---|
| `let x; take_b(x);` | `B`: the one head asked. `A` would go too, through `up`, and is not asked |
| `let x; take_b(x); take_a(x);` | `A`: of the two asked, the one that goes to both |
| the same with no row | refused: neither goes to both |

A projection goes through no row. It reads a place the head of its class has, and subtyping is
only where a value is passed. So `v[0]` with `v: Vec<u8>` is refused, though `&v` is passed
as `&[u8]`: indexing a type from outside is a call, and a call passes its arguments. A part
read through a reference is no exception, as that is the language's rule of a projection and
no row.

## Other rulings

- `take(Some(return))` with `take(Option<u8>)` is refused: `!` stands at a part, and no change
  is made at a part.
- `&{x, y}` does not go to `&{x}`. A reference points at the value as it lies, so no field is
  dropped behind one.
- Where two heads both go to every upper bound of a class, the class is refused, and neither
  is preferred.
- The fallback of a literal's domain and the fallback to `!` are two rungs, the domain's first.
  `!` says nothing at all is known, and what a literal falls back to may yet give a class a head.
  The head the parts read on a class make comes before both: it is what a written head would
  be, and a written head is there before any fallback.
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

## `e?` exits with an error

`e?` does two things, and calls nothing. A body has a result class, where its tail, its
`return`s and the exits of its `?`s meet.

| | What | What it is |
|---|---|---|
| 1 | The payload of `Ok` of `e` is the value of `e?`, and the payload of `Err` is what it exits with, `E`. | Two parts are named on `e`, as a projection names one. |
| 2 | A `Result` whose error is a class of its own, `W`, enters the body's result. | A written head enters, at once. |
| 3 | What enters `W` is told by the head of `E`. | A suspended constraint: it waits for that head. |

| The head of `E` | What enters `W` |
|---|---|
| a type of a name `N`: an enum, a struct, an extern-defined type | `Error{N(E)}`, with that one variant |
| an `Error` with variants | that `Error`, and the join unites its variants with the others |
| `!` | `!`: nothing exits |
| any other, as `i32` | `Error<E>` |

So `parse(s)?` and `read(p)?` in one body give it the error `Error{ParseError(..), IoError(..)}`.

`Error<E>` and an `Error` with variants are two constructors and do not join: a body that exits
with `i32` in one place and with `ParseError` in another is refused.

`?` names `Ok` and `Err`, which an `Option` has no place for, so `?` of an `Option` is refused.

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

A write `a.x = v` writes a head, as a lend does: an object with the field `x`, which enters
`a`. So the head of a class is the join of what enters it and of the object of the fields
written into it, and an object that entered has those fields besides its own.

| Program | `a` |
|---|---|
| `let a; a.x = 1u8;` | `{x: u8}` |
| `let a = {y: 1u8}; a.x = 2u8;` | `{x: u8, y: u8}` |
| `let a = @cfg; a.extra = 1;`, with `@cfg: {name: String}` | `{extra: i32, name: String}` |
| `let a = {y: 1u8}; a.x` | refused: a field read is no field written |
| `let a = &mut o; a.x = 2u8;`, with `o: {y: u8}` | refused: `o` has no `x` |

What decides this is whether what is written can then be used: `let a = @cfg; a.extra = 1;
let b = a.extra;` reads what it wrote.

Only a field writes a head. A field says all an object's head says, that it is there, and
such bounds are gathered by the join. A component, an element and a target say no whole head:
not how many components, nor what holds the element, nor which reference. What is written to
one goes into its place, and the class that holds the place has its head from elsewhere:
`let x = [return; 3]; x[0] = 1u8;` is `[u8; 3]`, and `let t; t.0 = 1u8;` is refused.

A field is written into the class it is written on, and no other. `let a; a.x = 1u8; let b =
a; b.y = 0;` makes `b` `{x: u8, y: i32}` and leaves `a` `{x: u8}`. A field written through a
reference is written into nothing: a reference points at the value as it lies, so no field is
added behind one, as none is dropped. What it refers to must have the field already. A
declared class is not written into either, as it never changes.

A class nothing gives a head, by entering or by writing, takes the head the parts named on it
make, read or written. That is where nothing enters it, or the join of what enters it is `!`,
and no head is asked of it from above:

| Parts named on it | Its head |
|---|---|
| fields `x`, `y` | `{x, y}` |
| payloads of variants of the enum `A` | `A`, with those variants |
| the payload of `Some`; of `Ok`, `Err`; of variants of an `Error` | `Option`; `Result`; that `Error` |
| a target, a component, an element | none: the name does not say which reference, how many components, or what holds the element |
| names of more than one of these | none |

So `let a; a.x` is `{x: !}`, and `let A::V(x) = a` makes `a` an `A`. Where the parts make no
head the class is `!`, and the part is refused as having no place.

The name of a payload says which type its variant is of, as the pattern that names it does. So
`let Some(x) = a` is held only by an `Option`, and not by an enum that has a variant `Some`.

Whether a class is a reference is whether anything has given it a target. This is a question
about parts, which a class holds before it is decided, and not one about its decided head.
A projection `p.f` asks it of `p`, to know whether `f` is a part of `p` or of what `p` refers to.

## Not settled yet

### Programs of the earlier model that differ, not looked into

`programs/` holds the cases of the earlier model that the text can write. Of the 26 that do
not come to what their case expected, these have not been opened, to tell a change of the law
from a mistake of translation or of the model:

- refused as expected, by another cause: `rules/classes/a-an-integer-literal-into-a-float`,
  `rules/place/a-every-arm-applies-to-the-scrutinee`,
  `rules/solving/a-a-class-passing-into-a-type-that-holds-it-is-not-filled`,
  `rules/solving/a-component-names-no-member`, `spec/two-language-variants-of-two-constructors`;
- cause unknown: `completeness/a-script-with-a-try-and-a-text-tail`,
  `spec/an-empty-part-with-two-upper-bounds`.

### What the text cannot write

- The value of a literal: that `256` is not held by `u8`. `{integer}` writes every integer type.
- A variant with no payload, named by a pattern.

### Calls and function values

Left to the end, and to be explained before anything is derived. The model is complete without
them, and they are added without changing what is here. What is known to need ruling:

- Recursion: a lambda that takes itself as a parameter.
- Brand types: which brand a returned value has, and whether a brand may be a hole.
- Where a call makes types equal and where one only enters the other (the transparency rule).

A call by name, `x.m(a)`, is ruled below with the rest.

What has been said so far.

- **A lambda has two phases, its body and its instantiation.** From outside its body a lambda
  value shows its signature and nothing else; inside, the body stands where it is written.
- **A capture is transparent.** A captured name is the one class inside the body and outside
  it, and it is decided with what surrounds the lambda, before and apart from any
  instantiation. `let a = None; let f = || a = Some(0u8);` gives `a: Option<u8>`. However many
  times the lambda is instantiated, a captured class stays one.
- **What is not captured is opaque.** The body makes a template: a parameter is a type variable
  with the bounds the body puts on it, and each call checks its arguments against those bounds
  anew. Two calls may give a parameter two types.
- **Where two lambdas meet, their parameters keep the bounds of both and their results are
  equal.** An argument has to keep every bound of every lambda. A result is not joined: it
  comes already decided, so nothing is adjusted there, and `!` is no exception. A result that
  is not decided depends on a parameter, and its being equal is then a bound on that parameter,
  checked at each call. `|x| 0u8` and `|x| "a"` do not meet, called or not;
  `|x| x.to_string()` and `|x| x.to_int()` are refused where they are called. The condition is
  of the meeting and not of either lambda: the closures of one function value are copied
  together, each takes the arguments, and their results are equal, at every use.
- **A module is a template as a lambda is.** A struct and its `impl` make one template, whose
  variables are the type parameters of `Self`, one for each field. Inside a method they are
  what a capture is inside a lambda: classes of what surrounds it, one class in every method.
  The rule is one, that inside a template a name of the surrounding scope is that scope's
  class; a capture is decided once because what surrounds a lambda is decided once, and a
  parameter of `Self` is a variable because what surrounds a method is a template. So a bound
  a method puts on a parameter of `Self` is a bound of `Self`, and every method keeps it: with
  `fn add(&self, other) { .. self.x + other.x .. }` in the `impl`, `new` makes no `Self` whose
  `x` has no `+`, whether `add` is called or not. Two uses may give `Self` two types.
- A bound is a constraint of the template, and a bound checked is that constraint finished or
  refused in a copy.
- **What the last rung decided is finished, and a call that comes to be known after it is held
  against that.** A call is known when what it calls is decided, which the earlier rungs do,
  so the copies of a call are there before `!` is given. Where what is called is itself
  decided only by the last rung, as in `let g = (f, None) as _; g.0(0u8);`, the classes that
  call would have entered are decided already.
- **A closure is a template and the classes that stand where it captures, and a function value
  is a set of closures by name.** Nothing is compared inside a function value: two that meet
  are joined, the set of both, and there is no other way between them, no parameter widened
  and no result narrowed. A capture is therefore of the closure and not a part of the type.
  Two closures of one template are two closures: in
  `let f = |x| { let g = |y| x + y; g }; let p = f(0u8); let q = f(1u16);` each copy of `f`
  defines `g` anew, over its own `x`. A module copied makes the closures of its methods anew
  in the same way, over that copy's parameters of `Self`.
- **A call is a function value passed to a function type.** `f(a, b)` passes `f` to
  `Fn(a, b) -> r`, a type the call writes, and what the language gives between the two is
  that each closure is used: its template is copied, each class that stands for a capture is
  one type with what the closure captures, each argument is passed to its parameter, and the result
  is one type with `r`. A call of another count of arguments is refused.
- **An argument is passed to its parameter, whatever is called.** A declared function, a
  function type and a lambda take an argument the same way: it is passed, by what the language
  gives, to the type the parameter writes. A parameter that writes none is a hole, which takes
  what comes as it comes, so the argument and the parameter are one type. Nothing enters in a
  call; a value enters where it flows into a variable.
- **A function is used as a function type the same, a value or of a type that is written.**
  A function passed to a function type is a call. Of a function value each closure is copied;
  of a function type `Fn(P) -> R` passed to `Fn(A) -> S`, `A` is passed to `P`, by what the
  language gives, and `R` is one type with `S`. So `Fn(&u8) -> u8` is passed to
  `Fn(&mut u8) -> u8` and not the other way, and what a function gives is changed in neither.
  Nothing of this is behind a reference or at a part of another type.
- **A class with no head is a hole only where nothing says what its head is.** Nothing is
  merged with a declared class, so what is one type with it is bound to it and takes its head
  where classes are next decided. An argument passed to such a class before then is passed to
  the declared class instead. So a variable a declared function type enters is called as
  the declared one is: `g <- d; g(a)` with `d: Fn(&u8) -> u8` and `a: &mut u8` reborrows, and
  a parameter that writes a function type is called in the body.
- **A function goes to a function type wherever a function is chosen by what goes to it.**
  Whether a closure can be used as a function type is told by its copy and not by reading. So
  a function is a receiver of every function of the name whose first parameter writes a
  function type, and goes through every row whose parameter does, and is passed to that type
  once the signature is made: what the copy refuses is refused, and is no reason to choose
  another. Two such functions both take it, whatever each type takes.
- **A function of a module that takes a receiver is found as a declared one is.** It is one
  whose first parameter writes a type. `p.m(a)` finds it by the receiver going to that type,
  among the declared functions of the name and by the one rule, and then writes what
  `M::m(p, a)` writes, the receiver lent where it is lent.
- **A parameter may write a type, and a struct's field is its type parameter.** What is passed
  to a parameter that writes a type is passed as to a declared type, with what the language
  gives: `&&u8` is passed to `x: &u8`, read through, and `u8` is not, a lend being a
  receiver's alone. In the body the parameter is a variable that the declared value enters.
  The type a parameter writes names classes of the lambda, a capture or another parameter. A
  struct declares its fields in an order, and its type has one parameter for each: a field
  named on a value of the struct is the parameter at the field's place, found through
  references as any part is.
- **A call by name calls the one function of that name the receiver goes to.** The functions
  of a name are rows of the one table of declared functions, which have one parameter or
  several. In `x.m(a)` the receiver `x` goes to the first parameter of a row as any argument is
  passed, by what the language gives alone and read as far as the row writes, at every depth:
  `&mut T` to `&self` by a reborrow, `&&T` to `&self` read through, and `&T` to neither `self`
  nor `&mut self`. One thing more is given a receiver and no other argument: one that is no
  reference is lent where the parameter is a reference, so `v.len()` is `(&v).len()`. Exactly
  one row must take the receiver: none goes nowhere, and of two neither is chosen. The
  arguments after the receiver are passed to their parameters, and the result is one type
  with the result the call writes. A row of one parameter is also what a value is passed
  through; a row of another count is not. The receiver is decided by what enters it, and
  the functions of the name decide nothing of it: `1.plus(2u8)` with `plus(u8, u8)` alone
  is refused, the receiver being `i32`, though `x.plus(1)` with `x: u8` is not.
- **A template says which lambdas its body defines, and a copy names a closure of each
  anew**, over the copies of the classes written where it captures. Copying itself makes no
  closure and knows of none, and two copies share a class only where a closure captures it.
- **A refusal made in several copies is one refusal where it names the same.** A copy records
  each constraint by the item its template wrote it by, and a refusal is its cause, its items
  and its connections. What a body refuses apart from its parameters names the same in every
  copy and is one refusal; what a call's argument is refused for names that argument and is
  one for each call. Where a capture is in it, the refusal names every copy that connected to
  the capture, as it names every connection.
- **A module is a lambda that takes nothing and gives its functions.** Its fields are classes
  of the body, which each function captures with the other functions, and using the module
  is calling that lambda. So what is said of a lambda is said of a module and nothing more
  is: each use copies it whole, a function calls another of the same copy by its name, and a
  function that uses the module by the module's name makes a copy anew, as a call of it from
  anywhere does, the copy of a function lying under where it is called and not under the
  module's.
- **A brand is a generic parameter.** Which brand a returned value has is what the type
  variable of the signature became in that call. There is no brand that is itself a variable
  with a bound.
- **A function makes the brands its signature writes and no parameter writes.** A brand a
  parameter writes is the brand of what is passed, as any hole is what is passed: with
  `index_of<B>(&F<B>) -> Idx<B>` the index has the brand of the argument. Nothing passed can
  be a brand that no parameter writes, so the function makes it, and each use of the
  function makes one of its own, decided where the signature is made: with `open<B>() -> F<B>`,
  `let a = open(); let b = open();` are of two brands, and `if c { a } else { b }` is refused.
  A brand is a head unlike any other, so two do not join, and were it decided later the two
  classes would have been merged before. A lambda that calls `open` makes a brand at each
  call of it, its body being copied.
- **A type has one brand, and a brand is no type.** A type of a name that has a brand has it
  at one place. What stands there is a brand a call made, and never a type that is written. A
  type that has a brand stands wherever a type does, as an argument of another:
  `X<X<usize, #0>, #1>`.
- **Recursion: a call of a closure within a copy of it is a call of that copy.** No copy is
  made, and nothing else differs from any call: each argument is passed to the copy's parameter,
  and the result is one type with the copy's result. Which copy that is, is read from the scope
  the call lies in and not from when it is solved. So `let f = |g, n| if .. { 0u8 } else
  { g(g, n) }; f(f, 1u8)` gives `u8`, a lambda that captures the class it is assigned to calls
  itself the same way, and two lambdas call one another. The arguments of the call and of the
  call within it are passed to the one parameter, so a recursion that changes its type or
  grows it is refused, and `|g| g(g)`, which no branch gives a value, gives `!`.

Marked, not ruled:

- A recursion through a class that two lambdas entered.
- Two lambdas that meet and are not called: no copy is made where they meet, so results that
  do not depend on a parameter and differ are refused only where the two are called.
- Whether solving ends: each copy of a definition names a closure anew, which the rule of one
  instance of a closure does not hold back. A closure made in a copy is of a template written
  one level inside the one copied, which is not yet shown to be enough where a closure leaves
  through a capture and is called again.
