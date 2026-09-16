# RFC-0038: `Result<T, E>` is a primitive, `?` widens the error, and a trap is not an error

Status: Accepted (Result, `!`, and `?` built; trap to follow)
Date: 2026-09-16
Extends: RFC-0022, RFC-0023, RFC-0036

## Ruling

`Result<T, E>` is a language type beside `Option<T>`: `Ok(x)` and
`Err(e)` build one, `Ok(v) = r` and `Err(e) = r` match one, and at run
time it is Rust's `Result` at `T = E = Value`, so it crosses the extern
boundary as itself (RFC-0022). Its two sides are typed apart; a script
that builds a `Result` must let both be known, as Rust does.

`!` is the type with no value. A type variable that nothing constrained
by the time checking is complete is `!`: `let r = Ok(3)` gives `r` the
type `Result<i64, !>`, since nothing fails into it. `!` is below every
type, so a `Result<T, !>` goes wherever a `Result<T, E>` is expected.
The compile-time freeze that asks whether a declared type is fully known
keeps refusing an open variable; only the final freeze closes one to `!`.

`x?` on `x: Result<T, E1>` inside a function returning `Result<U, E2>`
yields the `T` and unifies `Result<_, E1>` with the return type. When
`E1` and `E2` are structural enums (RFC-0036), that unification is the
solver's enum merge: the function's error enum grows the variants of
every `?` in its body, and nothing declares that growth. `x?` on an
`Option` in a function returning an `Option` propagates `None`. A
function is a script or a lambda; each has a return type the checker
tracks, and the tail expression flows into it as `?` does. In a
function whose return is neither, `?` is a type error, and in a
template `?` is refused: a template's result is text and has no side to
carry an error. An operand whose type nothing has fixed when `?` is
applied — a lambda parameter, checked before any call — is taken to be a
`Result`; `?` on an `Option` needs its type known there.

With `!` in the language, a trapping `unwrap` for templates and a
`panic` the script calls become ordinary functions returning `!`; the
analyses then have to treat a `!`-typed expression as one that does not
continue. That is the trap step's work.

A failure the program can act on is a value: an extern fn returns
`Result<T, E>` with `E` a type it declares — usually a `#[derive(TyArg)]`
enum of that function's own failure modes — and the script matches or
`?`s it. A failure the program cannot act on is a trap: a broken
contract, a value the checker admitted that the runtime cannot honor.
An extern fn traps by returning `Result<T, Trap>`; the macro knows
`Trap` by name as it knows `Result`, so `Result<T, Trap>` stops the run
and any other `Result<T, E>` is the language's. A function that can do
both returns `Result<Result<T, E>, Trap>`. `ExternError` is gone;
`Runtime::Error: From<Trap>`.

## Rationale

A single error type for every failure is the generalization RFC-0036
refused: a rate limit with a retry-after, a context overflow, a refused
tool schema are different values with different next steps, and one
`String` flattens them into a message the program can only print. Each
extern fn declaring its own error enum keeps the failure space honest,
and `?` with structural merge keeps that honesty cheap: the caller's
error type is the union of what it called, written by the compiler.

A trap is not returned as a value because it has no reader: the script
cannot proceed from a contract the runtime broke. It is not a Rust panic
because the runtime also targets `wasm32`, where nothing unwinds; so the
trap is the runtime's error type, carried on the same path `ExternError`
used, under a name that says what it is.

## Not built

- `Trap` is not yet built; `Trap` stands until it is.
- No `Result` from an extern fn as a language value yet: the macro still
  reads a returned `Result<T, E>` as the abort path (RFC-0023). The trap
  step flips that.
- No `?` in a template, and no `!`-returning functions yet; the analyses
  do not know a diverging expression.
- A lambda body is one expression: `if` and `let` are statements of a
  script and do not yet appear inside a lambda or a parenthesis, so a
  lambda that branches is written around an `if` bound outside it. This
  is the grammar as it was, not this RFC's ruling.

## Consequences

- `acvus-ast`: `Ok` and `Err` tokens and variant expressions; `?` and
  `Expr::Try`.
- `acvus-mir`: `TyTerm::Never`; `Solver::close_ty` beside `freeze_ty`;
  the checker's `return_ty`, `check_try`, and `try_returns` handed to the
  lowering, which tests the tag, unwraps the payload, and returns the
  failure rebuilt at the function's return type.
- `acvus-mir`: `TyTerm::Result`, `TyHead::Result`, `SerTy::Result`;
  `resolve_builtin_variant` knows both builtin enums; lowering picks a
  variant's payload type by its tag.
- `acvus-interpreter`: `ResultValue`, `Composite::Result`, `MakeVariant`,
  `TestVariant`, `UnwrapVariant`, and path projection on a `Result`;
  `layout` writes a one-byte side then the payload.
- `acvus-extern`: `TyArg` and `Cross` for `Result<T, E>`.
- `acvus`: JSON writes `{"Ok": v}` or `{"Err": e}`.
