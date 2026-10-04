# Experiment: an asynchronous call through a scope

The evidence for functions §3: a scope takes the callable and the arguments by
value, lends them to an asynchronous call, and gives them back with its
result, so work that outlives the future of the call that started it borrows
only what the runtime holds. `model.rs` is a safe runtime with no `unsafe`
(`forbid(unsafe_code)`), so memory safety holds by Rust alone; a runtime that
frees a value a call still uses, or lets its owner read it while a call holds
it exclusively, panics. One runtime per thread, so no `R` parameter. An acvus
function value is a `Program`, what the compiler would have compiled: it
waits a number of times, adds one to its exclusive argument, and returns the
length of its shared one. A call starts when it is made, and the runtime
takes it on every path (rfc_compiler effects §6).

| Part | What it checks | Run |
|---|---|---|
| `behavior.rs` | a scope that ends, one cancelled while its call runs, one forgotten while its call runs | `cargo test --manifest-path docs/rfc_extern/experiments/async-scope/Cargo.toml` |
| `probes/` | what rustc admits and refuses, each probe's first line stating the expected outcome | `docs/rfc_extern/experiments/async-scope/run_probes.sh` |

| Probe | What it checks | Expected |
|---|---|---|
| `a01_scope.rs` | a scope lending the function and the arguments to a call, giving them back | compiles |
| `a02_loan_escapes.rs` | the closure giving its loan out as its result | lifetime may not live long enough |
| `a03_call_escapes.rs` | the closure giving the started call out, unawaited | lifetime may not live long enough |
| `a04_arguments_used_during_the_call.rs` | the caller using an argument it gave the scope | E0382 |
| `a05_outer_loan_into_inner_call.rs` | an inner scope's call given the outer scope's loan | lifetime may not live long enough |
| `a06_caller_loan_covariant.rs` | the control: the caller's own value lent outside the scope through a covariant loan | E0597 |
| `a07_generic_arguments.rs` | a scope over arguments of a type variable of the caller | compiles |

Every build output goes to `target/` here, which git ignores.

Result, 2026-10-04, with stable rustc 1.97.1: all seven probes as expected, and
the three behaviour tests pass. A scope cancelled while its call runs leaves
its three values to the runtime, which frees them once the call ends; one
forgotten leaks them, and the call ends on values still alive.

The scope's closure is quantified over every loan the scope may make, so a
capture must outlive them all: a loan from outside the scope, the caller's own
or another scope's, fits none of its calls, whatever the loan type's variance
(a05, a06). The control's covariant loan was meant to show a caller's value
read while a cancelled call still held it; rustc refuses it before it runs.

What it does not show: the model's tasks hold `Value`s and need no lifetime,
so a runtime contract's spawn of a future that is not `'static` is not
modelled; arguments holding a loan of their own are not modelled; and an
optimized runtime, where the same misuse would be undefined behaviour, is not
built.
