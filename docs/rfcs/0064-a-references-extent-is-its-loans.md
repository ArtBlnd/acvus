# RFC-0064: A reference's extent is its loans

Status: Accepted (2026-09-20)

## Problem

A reference may not leave the scope that made it: it cannot be stored in
a container, an object or a tuple (RFC-0062 Decision 5), cannot be a
body's result (RFC-0062, decided 2026-09-20), and a lambda cannot capture
one (RFC-0018; `ReferenceCaptured`, raised at the lambda when a captured
name resolves to a reference). Each refusal is right today because the
language has no way to say how long a reference lives once it crosses a
boundary — a signature has no lifetime. The cost is ergonomic and real:

```
let r = &v;
let f = |k| -> len(r) + k;      // refused: a lambda cannot capture a reference
fn first(xs: &Vec<T>) -> &T      // unwritable: a body does not return a reference
```

and `substring(&s, 0, 3)` returns a `String` because `-> &str` cannot state
which argument the view borrows (RFC-0062, "the view return waits").

Rust states this with lifetime parameters written by the author. This
language's stance is that the checker states what the program means and
the author writes none of it; the question is what the checker needs.

## What exists

The borrow checker (RFC-0018, RFC-0029) is a dataflow over **regions**:
`Region { loans: Vec<Loan { storage, mutability }>, via }` per value, a
join-semilattice (`analysis/loans.rs`: join is union), liveness giving
every holder's extent (`analysis/liveness.rs`), and the rule that a
parameter or capture of reference type holds a loan on the storage it
names outside the body — inside the body **that storage is the parameter
itself**. Two arms of a `match` that hold different loans meet at the
join block by union. This is already location-sensitive, per instruction,
so the difficulty Rust's NLL solved (lexical regions too coarse for a
`match` arm) does not arise: the region of a value at an instruction is
the set of loans it holds there.

What is missing is one thing: a loan whose storage is **outside the body**
has no identity the caller can read. `Param(i)` is that identity.

## Decision

1. **A loan's storage is either a local or a parameter.** Inside a body, a
   reference parameter `p_i` introduces a loan whose storage is the
   parameter itself, as RFC-0029 says; this RFC names it `Loan::Param(i)`
   so it can be told from a local's. Parameters are unique ids in the
   IR already (a body's `params: Vec<ValueId>`); nothing is invented.

2. **A body has a summary.** After the borrow check, the region of the
   body's result, expressed over `Param` loans only (a local loan in the
   result is the existing refusal: a reference to a local cannot leave),
   is the body's **summary**: which parameters' loans the result may
   hold, with mutability. A body whose result is not a reference has the
   empty summary. A lambda's summary has two parts: the loans it
   **captures** (its region as a value — a lambda that captures a
   reference is a holder of that loan, exactly as a value assigned from
   the reference is) and its **call summary** (result loans over its own
   parameters). The summary is inferred; no syntax.

3. **A call substitutes.** At `f(a, b)`, the result's region is the
   callee's summary with `Param(i)` replaced by the argument's region;
   a lambda call adds the lambda's captured loans. This is the one new
   rule in the dataflow, and it is the same join: union of the
   substituted regions. Ambiguity (a result that may hold either
   argument's loan) is the union — conservative, never permissive.

4. **Recursion is a fixpoint.** Summaries are computed in dependency
   order; a cycle starts every summary at the bottom (no loans) and
   iterates to a fixpoint, which exists because regions are finite sets
   under union.

5. **The refusals become one rule.** A reference (or a holder of loans —
   a capturing lambda) may not be stored where it outlives every storage
   it names: a container element, an object field, a context, a spawn.
   A body's result may hold `Param` loans (that is what the summary
   says) and no local's. A lambda may capture a reference; the lambda is
   then a holder and the existing rules apply to it — it cannot be stored
   in a container, and using the storage it borrows while the lambda is
   live is a conflict, with the labels the checker already prints. The
   three refusals in the Problem stay for the cases that are still
   unsound and lift for the ones that were only unstated.

6. **Externs.** A handler declared `-> &str` or `-> Slice<T>` states its
   summary in its signature (RFC-0047 §3: a projection of one `&`
   parameter); the macro emits it as the declaration's summary, and rule
   3 applies at the call. This is what lets `substring`/`trim` return a
   view (RFC-0062's waiting item) once the pair-wide `CallShape` lands.

## What it costs

- `Loan` gains the `Param(i)` form and `Region` a substitution; typeck
  computes and stores a summary per body and per lambda type; the loans
  pass reads summaries at calls. Bodies are checked in dependency order
  with a fixpoint for cycles.
- A lambda's type carries its captured region (it is part of what the
  value is), so two lambdas capturing different loans are values with
  different regions — the same as two references to different storages.
- The diagnostics gain one case worth its own words: a lambda stored in
  a slot and called after the storage it borrows was written — two
  labels (`captured here`, `written here while the lambda is live`),
  which the labeled `Report` already carries.

## Rejected

- **Written lifetime parameters** (`'a`). The checker has the
  information; asking the author to restate it is the thing this
  language does not do. Where the inference is ambiguous the answer is
  the union, and a refusal names both origins (RFC-0063's identity
  labels are the model).
- **Capturing lambdas as holders without summaries** (the "stage one"
  the owner declined). It is the subset of this RFC that handles a
  lambda that never leaves; a lambda returned from a body needs the
  summary, and building the subset first means rebuilding it.
- **Loan regions on types only** (a lifetime as a type parameter). The
  extent is per instruction, not per type; putting it on the type is
  the lexical-region mistake.

## Where this is hard

- A lambda stored in a slot and called later: the holder is the slot,
  liveness covers it, the conflict is found; the diagnostic must point
  at the store and the later call, not only at the capture.
- A summary over a body whose result is a `match` of two references to
  two different parameters: the union `{Param(0), Param(1)}` — correct,
  and the caller sees both; a refusal downstream names both origins.
- Mutability: `&mut` loans through a summary keep their exclusivity at
  the call site; a result holding `Param(i)` mutably makes the argument
  unusable while the result lives, as a local `&mut` does.

## Order of work

1. `Loan::Param(i)` and the summary of a body's result; the body-result
   refusal relaxed to "no local loans"; tests for `fn first(&v) -> &T`.
2. Lambda: captured region, call summary, capture admitted; the store-
   then-call diagnostic.
3. Fixpoint over recursive bodies.
4. Extern summaries from signatures (with the pair-wide `CallShape`).

## Consequences

Step 1 landed. `Loan::storage` is `LoanStorage::Local(ValueId)` or
`LoanStorage::Param { index, value }`, both halves derived at one
constructor from the body's `params`, so no site can disagree about which
parameter a loan names. A body's summary is the `Param` loans of its result
with their mutability.

What a body may now return is a bare reference — `Ty::Ref` over anything but
`Slice` or `Str`. That bound is the machine's, not the checker's: a bare
reference is the one `Kind::Ref` word `control::Return` writes and a direct
call's destination receives, while a view is the register pair only an
extern call's `CallShape::Pair*` opens. `-> &str` and `-> Slice<T>` from a
body stay refused by `MirErrorKind::ReferenceReturnedFromBody` until the
machine has a pair destination for a body's result; `substring` and `trim`
still wait on that, as RFC-0062 said.

The refusal splits across two phases because the two facts are known in
different ones. Whether the result's type can leave at all is a type fact,
and typeck keeps it. Which storage the result names is a region fact that
exists only over the MIR, so `validate::borrow_check` raises
`ReferenceToLocalLeavesBody` where the local was borrowed.

A call's result substitutes the callee's summary where the summary is known
and otherwise takes the union of every argument's region. The union is a
superset of the substitution, so the reading without summaries refuses more
and admits nothing extra; every optimization pass keeps it, and only the
borrow check, which runs in dependency order, pays for the table. That order
is Tarjan's components over the call graph in pass 0 of `graph::optimize`. A
cyclic component is not the fixpoint of Decision 4: a body in one whose
result holds a reference is refused by name, and step 3 lifts that.
