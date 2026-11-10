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
`` `Some(a)` `` is `some_a`, `` `f result` `` is `f_result`, `0` is `zero`. Two classes of a
case keep two names, and a name that is a taken word is given a `_` after it. A `let` that
only gives a type variable (`let A;`) writes nothing: a name is a class from where it is
first written.

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
| `R = *P @ n;` | `n: r <- p.*;` |
| `R = P.x @ n;` `R = P.0 @ n;` `R = P[] @ n;` | `n: r <- p.x;` `n: r <- p.0;` `n: r <- p.[];` |
| `R = P as Some @ n;` `R = P as E::V @ n;` | `n: r <- p.Some;` `n: r <- p.E::V;` |
| `#[cast] extern fn up(A) -> B;` | `row up: A -> B;` |
| `#[conversion] extern fn narrow(u16) -> u8;` | `row narrow: u16 -> u8;` |
| `extern type Vec<T>;` | nothing: `Vec<..>` is written where it is used |

`let R = ...;` is the statement alone.

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
| `must fail;` with no kind of refusal | `assert refused ..;` |
| `assert refused no_join ...;` | `assert refused Mismatch, ..;` |
| `assert refused infinite ...;` | `assert refused Occurs, ..;` |
| `assert refused no_value ...;` | `assert refused UndecidedClass, ..;` |
| `assert refused conflict ...;` on a literal and a type that does not hold it | `assert refused EmptyDomain, ..;` |
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
- a `bind`, where it is more than a path of projections from its position;
- a case that tests a primitive on declarations alone (`assert join(..)`, `assert le(..)`,
  `assert coerce(..)`, `assert along_le(..)`) and records no program.

All of these but the last wait for calls and instances.
