# What the compiler says

A refusal names the fix. Every row below is a wrong program somebody
writes and the whole sentence the compiler answers it with; the programs
live in `acvus-cli/tests/refusals/` and their rendered diagnostics are
pinned there byte for byte, so this catalogue is checked rather than
claimed.

Three devices carry the fix:

- **the sentence** says what is wrong and, where one spelling settles it,
  what to write: `; did you mean `push`?`, `; write `.to_string()` for the
  owned text`, `; write `&mut` where the `&` is`;
- **a help note** (`= help:`) carries a fix that is longer than a clause:
  the call rewritten, the element type that moves;
- **labels** name the other places the story needs — where a value was
  moved, where a reference was taken (RFC-0064).

A rule is broken once. A consequence is a refusal on an operand the first
refusal already poisoned, and it is not raised at all rather than filtered
afterwards: `Ty::Error` unifies with anything, so what it reaches asks
nothing (`ty.rs`). Two sites hold that today, each because a corpus program
showed a second sentence without it:

- an object's or a tuple's refused component is poison, so the body's
  result is not refused again for a reference the aggregate does not hold
  (`37-a-reference-stored-in-an-object.acvus`);
- a `match` arm whose tag the enum does not carry is read against poison,
  so its pattern is not refused again and the name it binds is still bound
  (`12-match-missing-an-arm.acvus`, `21-enum-tag-misspelled.acvus`).

Poison replaces a component only where the aggregate's type is built out of
it. A list's element and a variant's payload unify into a variable the
aggregate's type already holds, and the solver never binds a variable to
poison (`solver.rs::take_other`), so poisoning there would leave the
variable open — and an open variable refuses in its own words, which is a
worse sentence, not none.

A span carries nothing about any of this. Two refusals one inside the other
are usually two facts: `total = i64::from_str(text)` with `total` an `i64`
and `text` a `String` is refused twice, at the argument and at the result,
and both are printed (`41-two-independent-refusals.acvus`).

One consequence is left, because no type carries it: `undefined variable x`
after `cannot assign to x` is the same fact under two kinds, and only the
first is printed.

Refusals are printed in source order, not in the order the checker raised
them — a deferred decision settles after the text after it was read.

## Names

| program | what it says |
| --- | --- |
| `let v = vec([1, 2]); v.pushh(2); v.len()` | undefined function `pushh`; did you mean `push`? |
| `let s = "a".to_string(); s.uper()` | undefined function `uper`; did you mean `upper`? |
| `v.into_iter() \| map(\|x\| -> x + 1) \| colect` | undefined function `colect`; did you mean `collect`? |
| `frobnicate(1)` | undefined function `frobnicate` |
| `let total = 1; totl + 1` | undefined variable `totl`; did you mean `total`? |
| `let p = { name: "x".to_string(), age: 3, }; p.nmae` | `p` has no `nmae` stored on every path that reaches here; did you mean `name`? |
| `match shape { Shape::Circl(r) => r, _ => 0.0, }` | unreachable pattern: `Shape::Circl(_)` is not a variant of `Shape{Circle(Float)}`; did you mean `Shape::Circle`? |
| `total = 1; total` | cannot assign to `total`: no binding named `total` is in scope; `let total = ...;` binds it |
| `@missing + 1` | `@missing` is not a declared context |

A candidate is within an edit distance of two of what was written, or has
it as a prefix; the three nearest are offered, and a name with nothing near
it keeps the sentence it had.

## Borrows

| program | what it says |
| --- | --- |
| `let v = vec([1, 2]); len(v)` | no `len` takes a call of type Fn(\<error>) -> _<br>`= help:` the parameter is a `&`; write `len(&v)` |
| `let s = "abc".to_string(); len(s)` | no `len` takes a call of type Fn(String) -> _<br>`= help:` the parameter is a `&`; write `len(&s)` |
| `let r = &v; r.push(2)` | `r` is a shared reference and cannot be borrowed mutably; bind it with `&mut` |
| `push(&v, 3)` | type mismatch: expected &mut Vec\<i64>, got &Vec\<i64>; write `&mut` where the `&` is |
| `let n = 1; let r = &n; *r = 2; n` | cannot store through `r`, of type &_: not a `&mut`; bind it with `&mut` |
| `let r = &v; let m = &mut v;` | `v` is written here while a reference to it is live<br>labels: `&v` borrowed here · `r` the reference is used here |
| `let a = vec([1, 2]); let b = a; a.len()` | `a` is used here after it was moved<br>label: `a` moved here |
| `for x in v` over a `Vec` | a container is not consumed by a loop; write `&v` or `&mut v` |
| `let s = v[0u64]` over a `Vec<String>` | cannot move out of index of `Vec<String>`<br>`= help:` the element is String, which moves; take a reference with `&a[i]` |

## Text

| program | what it says |
| --- | --- |
| `let v = vec(["a".to_string()]); v.push("b")` | no instance of the signature has the call type Fn(Array\<String, 1>) -> Vec\<&str>; write `.to_string()` for the owned text |
| `let s = "  abc  ".to_string(); s.trim()` as the body's value | a body does not return a reference; write `.to_string()` for the owned text |
| `let v = s.trim(); let f = \|k\| -> len(v); f(1)` | a lambda cannot capture a string or slice view; write `.to_string()` for the owned text |
| `let r = &v; { inner: r, }` | a reference cannot be stored in a list, object, or tuple |

## Types

| program | what it says |
| --- | --- |
| `let x = v.first(); x + 1` | type mismatch in `+`: Option\<&_> vs i64; an Option is not its payload -- write `.unwrap()`, `?` or match it |
| `let n = i64::from_str("7".to_string()); n + 1` | type mismatch in `+`: Result\<i64, ParseIntError{...}> vs i64; a Result is not its payload -- write `?`, `.unwrap()` or match it |
| `let s = "a".to_string(); s + 1` | type mismatch in `+`: String vs i64 |
| `1 == "a"` | type mismatch in `==`: i64 vs str |
| `let n = 1; while n { … }` | type mismatch: expected Bool, got i64 |
| `[1, "a"]` | heterogeneous list: expected i64, got &str |
| `let v = vec([1, 2]); v.push("b".to_string())` | no instance of the signature has the call type Fn(Array\<i64, 2>) -> Vec\<String> |
| `let x = 1; x.len()` | no `len` takes a call of type Fn(_) -> u64 |
| `1 as Integer` | `as` converts to i8, i16, i32, i64, u8, u16, u32, u64, f64 or char, not to `Integer` |
| `let x = 1; x[0u64]` | no `as_slice` takes a call of type Fn(&i64) -> &[_] |

## Arity and shape

| program | what it says |
| --- | --- |
| `concat("a")` | function `concat` expects 2 arguments, got 1 |
| `let f = \|a, b\| -> a + b; f(1)` | this closure expects 2 arguments, got 1 |
| `let s = "7".to_string(); s.pad_start("0", 2)` | type mismatch: expected i64, got &str · type mismatch: expected &str, got i64 |
| `let text = "42".to_string(); let total = 0; total = i64::from_str(text)` | type mismatch: expected i64, got Result<i64, ParseIntError{…}> · type mismatch: expected &str, got String |
| `match shape { Shape::Circle(r) => r, Shape::Square(s) => s, }` | unreachable pattern: `Shape::Square(_)` is not a variant of `Shape{Circle(Float)}` |
| `let n = 1; break; n` | `break` is only inside a loop |
| `let f = \|k\| -> { x = vec([3]); 0 }` | cannot assign to `x`: it is captured by the lambda, not bound in it |

## Four that do not yet name the fix

These are in the corpus and pinned, so what they say cannot drift, but the
sentence does not carry the fix and the reason is not the wording.

- `let v = vec([1, 2]); len(v)` prints the argument as `<error>`. The
  argument's type is a signature decision (`vec`) the solve has not settled
  when the call refuses, and it resolves to the error type. The datum would
  arrive only if the refusal were held until the solve finishes.
- `let x = 1; x.len()` prints the receiver as `_` for the same reason: a
  method call's receiver is admitted against a fresh parameter variable,
  and the refusal is raised before that variable is bound.
- `strng::len("a")` is not refused as a misspelled namespace at all.
  `ns::tag(payload)` where `ns` names no function is a structural enum
  variant (RFC-0030), so the program builds `strng::len("a")` as a value
  and is refused later for holding a view in data. Refusing it as a
  misspelling would change what the language admits.
- `let x = 1; *x` is refused by the IR validator in the machine's words
  (`Take takes Ref and got i64`). Typeck's deref arm unifies an open
  variable with a reference, which succeeds for `1`, so the script-facing
  `DerefOfNonReference` never fires. Naming the fix here is a typeck rule
  change, not a message.

`s.pad_start("0", 2)` gets two refusals, and they are not one consequence
of the other: each argument independently fails to meet its parameter, so
both stay.
