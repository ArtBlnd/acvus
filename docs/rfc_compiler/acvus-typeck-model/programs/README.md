# Programs translated from the cases of the earlier model

`cases/` holds the cases of a model two rebuilds back, in a language of its own
(`cases/README.md`), which nothing reads any more. Each file here is one of those cases
written as the text of the present model (`src/text`), with what the case expected written
as assertions. They are translated as they stand, and not made to hold: a program that
does not come to what its case expected shows where the present model differs.

`cargo run --bin check -- programs` runs them and says of each whether it holds.

## Where a program lives

`cases/<dir>/<name>.case` is `programs/<dir>/<name>.prog`. A case that is not translated is
listed, with why, in a file `SKIPPED*.md` of its directory.

A program begins with comments: the case it is from, the case's own leading comments as
they are, and then whatever the translation could not carry.

```text
// from cases/spec/a-value-meets-a-declared-parameter.case
// fn f(x: u8)   let a = 1u8; f(a)
// not asserted here: assert check `A ~> f.Param(0)` == identity;
```

## The text

`src/text/mod.rs` states the text, and `src/text/tests.rs` shows every form. In short:

| Statement | What is recorded |
|---|---|
| `a <- b;` | `b` enters `a` |
| `a <- Option<b>;` `a <- &mut b;` `a <- {x: b};` `a <- u8;` `a <- !;` | a class with that head enters `a`. A part is a class name or a type again |
| `a <- {integer};` `a <- {float}` | a literal with no suffix enters `a` |
| `a.x <- b;` `b <- a.x;` | the part `x` of `a` is written into, or read |
| `pass a to p;` | `a` is passed to `p` |
| `exit e to w;` | the exit of a `?` |
| `declare p = Option<t>;` | `p` is a class a declaration writes. A small-letter name in it is a class, and so a hole |
| `row up: A -> B;` `row view<T>: &Vec<T> -> &[T];` | a declared function a value may be passed through |
| `first: pass a to p;` | a label |

A path names parts: `a.x` a field, `a.0` a component, `a.*` what a reference refers to,
`a.[]` an element, `a.Some` `a.Ok` `a.Err` `a.E::V` `a.Error::V` a payload.

A word that begins with a small letter is a class, a field or a label. One that begins with
a capital is a type or a variant. These small-letter words are taken and name no class:
`pass` `to` `exit` `declare` `row` `as` `mut` `assert` `accepted` `refused` `names` `takes`
`at` `str`, and the scalars `i8` to `u64`, `usize`, `f32`, `f64`, `bool`, `char`.

| Assertion | What solving is to come to |
|---|---|
| `assert a: Option<u8>;` | the type of `a`, written whole |
| `assert accepted;` | no refusal |
| `assert refused Mismatch, UndecidedClass;` | refused by exactly these causes |
| `assert refused Mismatch, ..;` | refused by these at least |
| `assert refused ..;` | refused by some cause |

The causes are `Mismatch`, `NoChange`, `EmptyDomain`, `Occurs`, `NotHeld`, `UndecidedClass`.

## How a case is translated

### Names

A class of the case is a class here, by a small-letter name made from its own: `A` is `a`,
`` `Some(a)` `` is `some_a`, `` `f result` `` is `f_result`. Two classes of a case keep two
names. A `let` that only gives a type variable (`let A;`) writes nothing: a name is a class
from where it is first written.

| The case's name | The name here |
|---|---|
| a taken word, as `u8`, `str`, `to` | the word and `_`: `u8_`, `str_`, `to_` |
| `&x`, `&mut x`, `*x` | `ref_x`, `ref_mut_x`, `deref_x` |
| `f(..)`, `mk()` | `f_call`, `mk_call` |
| one that begins with a digit, as `0`, `1u8`, `256` | the number spelled out: `zero`, `one_u8`, `two_hundred_fifty_six` |
| `?k`, `@cfg` | the sign dropped: `k`, `cfg` |
| two that differ by capitals alone, `s` and `S` | the small one as it is, the other `cap_s` |
| a second class of a name already taken here | the name and `_2` |
| a declared function `core::as_slice` | its last word: `as_slice` |

Comments of the case are copied in ASCII: a sign that is not ASCII is written by the ASCII
that reads nearest, as `~>` for the arrow of a coercion and `<=` for the order.

### Classes and statements

| The case writes | The program writes |
|---|---|
| `let 0: lit 0;` | `zero <- {integer};` |
| `let x: lit 1.5;` | `x <- {float};` |
| `` let `Some(a)`: Option<S>; `` | `some_a <- Option<s>;` |
| `` let `@cfg`: host {name: N}; `` | `declare cfg = {name: n};` |
| `let D: out Result<U8, E>;` | `declare d = Result<u8_, e>;` |
| `S = A;` | `s <- a;` |
| `A == B;` | `a <- b; b <- a;` (classes that enter one another are one class) |
| `R = &P @ n;` `R = &mut P @ n;` | `n: r <- &p;` `n: r <- &mut p;` |
| `R = *P @ n;`, read | `n: r <- p.*;` |
| `R = P.x @ n;` `R = P.0 @ n;` `R = P[] @ n;`, read | `n: r <- p.x;` `n: r <- p.0;` `n: r <- p.[];` |
| `R = P as Some @ n;` `R = P as E::V @ n;` | `n: r <- p.Some;` `n: r <- p.E::V;` |
| a projection written into (below) | `n: p.x <- v;` |
| `#[cast] extern fn up(A) -> B;` | `row up: A -> B;` |
| `#[conversion] extern fn narrow(u16) -> u8;` | `row narrow: u16 -> u8;` |
| `extern type Vec<T>;` | nothing: `Vec<..>` is written where it is used |

`let R = ...;` is the statement alone.

### A projection is read or written

The earlier language names a part by a projection, `R = P.x @ n;`, and does not say whether
it is read or written. The text says it by where the path stands, so the translation tells
them apart by what the case does with the class `R` the projection gives:

| In the case, `R` | The program writes |
|---|---|
| passes into something, or is an operand: `S = R;`, `f(R)`, `&R`, `R.y` | a read: `n: r <- p.x;`, and `r` wherever the case writes `R` |
| has something pass into it: `R = V;` | a write: `n: p.x <- v;`. The name `r` is not written |
| both | the write `n: p.x <- v;`, and a read of its own for the rest: `r <- p.x;` |
| neither | a read: `n: r <- p.x;` |

Where several values pass into `R`, each is a write, and the first has the label. This holds
of every projection that names a place: a field, a component, an element, a deref. A
payload named by a pattern is only read.

A `bind` whose path passes no reference reads its position: `label: b <- position;`.

The views of an array the earlier language declared by itself (`core::array_view`,
`core::array_view_mut`) are not written: an array behind a reference goes to a slice by the
present language itself.

### A use of a declared function

`R = f(X, Y) @ n;`, where `f` is one `extern fn` and no other declaration has its name, is
the signature written once for this use, each argument passed to its parameter, and the
result entering:

```text
// extern fn push<T>(&mut Vec<T>, T) -> ();     `r` = push(V, X) @ n;
declare n_p0 = &mut Vec<n_t>;
declare n_p1 = n_t;
declare n_result = ();
n_0: pass v to n_p0;
n_1: pass x to n_p1;
r <- n_result;
```

A type parameter of the function is a class of this use alone (`n_t`), so two uses of one
function share nothing.

### `?`

`R = ?(X) into Q @ n;` is the value read, a `Result` whose error is a class of this `?`
entering what the exit passes into, and the exit:

```text
n: r <- x.Ok;
q <- Result<n_ok, n_error>;
exit x.Err to n_error;
```

### What the case expects

| The case writes | The program writes |
|---|---|
| `must pass;` | `assert accepted;` |
| `must fail;` with no kind of refusal that has a row here | `assert refused ..;` |
| `must fail;` with a kind that has a row here | the row of the kind alone |
| `assert refused no_join ...;` | `assert refused Mismatch, ..;` |
| `assert refused infinite ...;` | `assert refused Occurs, ..;` |
| `assert refused no_value ...;` | `assert refused UndecidedClass, ..;` |
| `assert refused conflict ...;` on a literal and a type that does not hold it | `assert refused EmptyDomain, ..;` |
| `assert refused conflict ...;` on anything else | no row: the line is kept as a comment |
| `assert X: T;` | `assert x: T;` |

Several kinds a case expects are written as one assertion: `assert refused Mismatch, Occurs, ..;`.

Any other expectation looks into the earlier model, at a trace, a certificate, a decision, a
partition, a count of outcomes, I\*, a check's coercion, a class with no value, and has no
form here. It is kept as a comment at the head of the program, the line as the case writes
it after `// not asserted here: `. So are the directives (`//@ orders`, `//@ order`): a
program is solved in the orders the present model takes.

## What is not translated

A case is translated whole or not at all: a program that leaves a statement out is another
program, and what it comes to says nothing of the case. A case is skipped where it writes any
of these, which have no form in the text yet:

- a `group`, a group's `fn` and a use of it, `#[by_name]`, `#[through_ref]`;
- a `lambda`, or a call of a function value `(F)(X)`;
- a use of a name that several declarations give (`f#0`, `f#1`);
- a `struct` whose field is read or written;
- a function type `fn(P) -> R`, a function value `Fn/n`;
- an extern-defined type with an identity (`#`);
- a `universal`;
- a `bind` whose path passes a reference, so that it lends;
- a projection of a variant with no payload, `P as None`;
- a function at a type of parts, where the earlier model takes a candidate at each part (`==`
  of two tuples);
- a case that tests a primitive on declarations alone (`assert join(..)`, `assert le(..)`,
  `assert coerce(..)`, `assert along_le(..)`) and records no program.

Most of these wait for calls and instances.

`?` is translated whatever its operand is, an `Option` included: it is one construct of the
language, and a program that the present model refuses is what the translation is for.

## The programs that differ, and why

25 of the 87 do not come to what their case expected. Each is here once, by what makes the
difference. Each was read and run in parts to find the cause.

| Why | Programs |
|---|---|
| The case was wrong: `?` exits into the class the tail enters, so a body with a `?` and a tail that is no `Result` is refused (`Ok(show(a))` is what is meant) | `completeness/a-script-with-a-try-and-a-text-tail` |
| The present law: `&&T` is a type | `spec/a-lend-that-reborrows-a-reference`, `rules/types/a-a-reference-to-a-reference` |
| The present law: a part read or written gives its class a head | `spec/a-field-written-by-the-text`, `spec/a-language-variant-names-its-constructor`, `spec/a-language-variant-reads-back-its-payload`, `probe/two-references-of-two-mutabilities-enter-one-class` |
| The present law: a part named on a class `!` entered | `holes/a-pattern-on-a-scrutinee-of-type-never` |
| The present law: a join makes the parts one type, so two types that meet at a part are a `Mismatch` and leave the classes above with no type | `spec/an-empty-part-with-two-upper-bounds` |
| The present law: a part nothing enters, with two upper bounds, is not filled; both refuse, for another cause | `rules/place/a-every-arm-applies-to-the-scrutinee` |
| The present law: a part named on a class that is `!`, or on a head that has no place for it, is `NotHeld`; both refuse, for another cause | `rules/solving/a-component-names-no-member`, `spec/two-language-variants-of-two-constructors` |
| The present model says more: a class passed into a type that holds it is `Occurs` | `rules/solving/a-a-class-passing-into-a-type-that-holds-it-is-not-filled` |
| The present law: a value is changed only where it is passed, to a parameter or by `as`. What enters a class is joined, and a join goes through no row, so a `&Vec<u8>` and a `&[u8; 2]` that enter a class written `&[u8]` are a `Mismatch`. Passed to it, each goes | `completeness/two-references-into-a-written-slice-reference`, `completeness/two-references-of-two-nodes-enter-one-class`, `phases/decomposition-through-a-cast` |
| The text cannot write the value of a literal, so `256` is not told from `1` | `rules/classes/a-a-literal-the-type-cannot-hold`, `rules/solving/a-litvar-default-that-cannot-hold-it`, `rules/solving/a-litvar-into-a-type-that-cannot-hold-it` |
| The present law: a projection goes through no row, and indexing a type from outside is a call. The case reads `v[0]` as a part of a `Vec` | `spec/an-index-through-a-view`, `phases/unify-a-view-links-an-index`, `spec/a-for-over-a-vec-by-value`, `completeness/an-index-of-an-unsuffixed-literal` |
| The present law: of two heads asked of a class, the one that goes to both is taken, so `x` passed to `A` and to `B` with a row `A -> B` is `A` | `rules/solving/a-a-class-with-two-contexts-is-refused` |
| `?` of an `Option`, which the present law refuses | `spec/a-try-in-a-match-arm` |

The law answers every one. The three of the literal's value wait for the text to write one;
each of the others differs as the present law means it to, or by a case that was wrong.
