# RFC-0054: the host declares what `main` returns

Status: Accepted — 2026-09-19
Extends: RFC-0038 (`!` is below every type), RFC-0046 (a call's task is an
effect), RFC-0047 (the slice retry, whose T2 found this)

## Problem

A script's `main` leaves with a value, and that value goes to a **host** —
the test harness, the CLI, an embedder. Until this RFC nothing checked its
type against what the host reads.

`validate` said so itself, at `acvus-mir/src/validate/type_check.rs:105-113`
on master `92fcc405`:

```rust
// `main` is the body of a graph `Function`, and its declared return type
// is that function's `PolyTy::Fn { ret }` in the `CompilationGraph`, which
// `validate` does not receive: its `Return` is checked for everything but
// the declared type.
let mut ctx = CheckCtx::new("main".to_string(), None);
```

A closure got `declared.get(label)` — the `ret` of the `Ty::Fn` its
`MakeClosure` gave it — and its `Return` was held to it. `main` got `None`,
and `Return`'s arm in `check_inst` skips the check when the declaration is
absent. RFC-0047's T2 sabotage exercised exactly that: a `main` body edited
to return a wrong-typed value passed `validate` with no error.

The hole was upstream too. Every host built the entry's graph `Function` with

```rust
ret: Box::new(pb.fresh_ty_var()),
```

so the solver had nothing to join the body's returns against: it *inferred*
`main`'s return type from the body, and then the only thing that could have
checked the body was that same inference.

## Decision

**The host declares what `main` returns, at compile time, and the
compilation holds the body to it.**

### 1. A compilation declares `main`'s return type

The declaration is a `Ty`, and it enters through the slot the graph already
has for one: the entry `Function`'s `PolyTy::Fn { ret }`, lifted with
`lift_declaration` — the same call the hosts already make for context types,
two lines above. A declared entry is not a special case: it is an ordinary
declared function.

### 2. The solver joins the body against it on the shared path

`graph::infer::infer_scc` already freezes a concrete `ret` into
`expected_tail_ty` and passes it to `TypeChecker::check_script`. What
`check_script` did with it was a second, separate `convert_at` on the tail
only. It now sets the body's own return variable from it:

```rust
let return_ty = match expected_tail {
    Some(declared) => lift_ty(declared),
    None => self.solver.fresh_ty_var(),
};
```

so every return of the body joins the declaration through one path: the
tail's `flow`, and each `?`'s early return in `check_try`, which already
unified against `self.return_ty`. A body with no tail joins `Unit` there.
There is no `main`-only branch in the solver.

### 3. `validate` receives the declaration by structure

`MirModule` carries `ret: Ty` — the `ret` of the graph `Function` whose body
the module is. The lowerer takes it and `build_module` writes it; the inliner
carries it through. `check_types` builds `main`'s `CheckCtx` with
`Some(module.ret.clone())`, and the unchecked `CheckCtx::new("main", None)`
form is gone.

The field is a `Ty`, not an `Option<Ty>`: a module cannot be constructed
without saying what it returns, so there is no unchecked `main` to reach.

`MirModule` over `validate(&module, ret)`: the module is the thing that
travels — through `inline`, `dedup`, both optimization passes, `prepare` —
and `validate`'s three call sites and `optimize`'s eight stay as they were.
Nine struct literals changed, seven of them test fixtures.

### 4. The declaration is the contract, and its test is at the contract

`acvus-interpreter-test/tests/main_return_declaration.rs` is RFC-0047 T2's
sabotage, made permanent:

```
compile failed:
  [main] type mismatch: expected i64, got String
```

and the `?` case names the second return:

```
  [main] `?` leaves with Result<!, String> but the function returns i64
```

`acvus-mir-test/tests/return_type.rs` carries the same sabotage at
`validate`'s contract: a declared module whose `main` is edited to `Return` a
reference draws one refusal, where before this RFC it drew none — that file's
`a_main_is_not_checked_against_a_return_type_the_module_does_not_hold` was
the defect written down as a passing test, and it is now
`a_main_is_checked_against_what_the_host_declared`.

**The order between the two.** A source-level mismatch never reaches
`validate`: the checker refuses it, the function is `Incomplete`, and
`graph::lower` produces no module. `validate` holds the *pipeline* — the
declaration against what the optimization passes left — not the source.

### 5. The `!` declaration: a host that cannot name a type says so

A host that prints whatever the file returns has no type to state. It states
*that*: `!` — the type with no value (RFC-0038) — is the declaration "I state
no return type". `validate` accepts every `Return` under it, the checker holds
the tail to nothing, and the host reads the value it gets back by kind, which
the runtime carries with the value: `Value::kind()` for a word, the vtable's
`Composite` for an allocation. It is spelled by the host, never defaulted.

The rule was already in the code, unnamed. `types_match`'s `(Ty::Never, _)`
arm fires on the *expected* side, so a declared `!` already accepted any
actual type; its parameters are now `expected` and `actual`, and the arm says
which side it reads. The value-side fact — a `!` *value* satisfies any slot —
is a second site, `InstKind::Return`'s `!matches!(value_ty, Ty::Never)` guard.
Two facts, two places; neither arm carries both.

`acvus-cli` declares `Ty::Never` (`acvus-cli/src/compile.rs`; the
`fresh_ty_var()` is gone). Under that declaration `main`'s inferred `ret` is
`!` and no longer describes the value, so the CLI stops printing through a
`Ty`: `Compiled::ret_ty` is gone and `json::by_kind` reads the value the way
the declaration says the host does. Its output is unchanged — the ten
`acvus-cli/tests/cli.rs` cases, which assert exact stdout, are untouched.

Two tests carry the rule at the two contracts:
`a_main_declared_never_holds_its_return_to_nothing`
(`acvus-mir-test/tests/return_type.rs`) is
`a_main_is_checked_against_what_the_host_declared` one variable apart — the
same reference sabotage, declared `!`, drawing no refusal; and
`a_host_declaring_never_gets_the_value_and_reads_it_by_kind`
(`acvus-interpreter-test/tests/main_return_declaration.rs`) runs a `String`
body under a `!` declaration and reads the result through its vtable. The ten
`#[should_panic]` sites that declare `Ty::Never` are now correct by this rule
rather than by accident.

## What it costs

Every host names a type. In `acvus-interpreter-test` the declaration is a
`ret: Ty` parameter on `run_script`, `run_script_mode`, the `*_with_externs`
and `run_parsed*` forms, `compile_script_mode`,
`compile_source_with_externs*`, `prepared_script*` and `script_listing*` —
about 200 call sites across ~55 test files, each naming the type that test
then inspects. Two of those declarations were not what a reader would guess,
and the compiler said so: a container's `len()` is `u64`, and `collect`
yields a nominal `Vec<T>`, not an `Array`.

A template needs no argument from its caller: `check_template` fixes the tail
at `Ty::String`, so the template entry declares `Ty::String` itself.

## Rejected

**Inferring `main`'s return type and checking the body against it.** That is
what master did, and it is circular: the inference is a summary of the
returns, so every return agrees with it by construction. It refuses nothing.

**A run-time kind check on the host side.** `Value::kind()` is available, and
the CLI already switches on the entry's type to print. Checking there is a
convention every host must remember, it fires after the run, and it says
nothing at the contract the script was compiled against. The declaration is
the guarantee; the host reads the value with the witness it declared.

**An explicit-any `Ty` besides `!`.** A new variant — or `Error(ErrorToken)`,
which unifies with everything — would give the same acceptance under a second
name. The type system already has the word for "I cannot state a type": a host
that cannot return the type says so in `!`, and `types_match` already read it
that way on the expected side. A second spelling would have to be kept in step
with the first at every site that reads a declaration.

**A `--returns` flag at the CLI.** It moves the declaration from the host to
the person running the file, who has no more to say about it: the CLI prints
whatever comes back either way. It would also make the ten `#[should_panic]`
sites that declare `Ty::Never` wrong, and buy nothing they need — they assert
on the panic, never on the value.

**A `main`-only branch in the checker.** `check_script` already had the
declaration; the fix was to make it the body's return type rather than an
extra check on the tail. A branch for the entry would have left `?` unheld.

## Consequences

- A body with no tail now states `Unit` to the checker, so a host declaring
  a non-unit return for a script that ends in a statement is refused where
  before nothing was said.
- The `acvus-interpreter-test` entry is named `main` rather than `test`, so
  the error names what the rule names.
- `acvus-mir-test`'s IR-inspection helpers declare `!` by rule 5: they read
  the IR, not a value. `declared_script_module` is there for the ones that do
  declare a type.
- `acvus-cli` reads its result by kind, so a value whose type the printer
  could not name — a closure, an extern handle — prints as its Rust type name
  rather than as its `Ty`.
