# Ownership and references

Which values copy and which move, what a reference is, how long it may live,
and what may be done to a storage while it is lent. The checker states every
extent; the author writes no lifetime. The machine's ownership of values and
frames is RFC-0048.

## RFC-0018: A reference is a type, and only a word or a String copies

Status: Accepted

1. **The copy set.** A value of a word type copies. The words are the eight
   integer widths (RFC-0037), `Float`, `Char` (RFC-0058), `Bool`, `Unit`,
   `Never`, `Order`, a bare reference `&T`/`&mut T`, and an `Option` of a
   word, since an `Option` has no representation of its own (RFC-0039);
   `String` copies by rule 2 and is not a word, and neither is an `Option`
   of one. A view (`&str`, `&[T]`) moves. Every other value moves: a
   binding used after it moved is a type error at the use.
2. **String.** `String` is language-owned and immutable: there is no
   in-place string operation, and a `&mut String` replaces the value whole.
   A use of a `String` value that is not its last use copies it: the
   compiler inserts `StringClone` before that use, and the last use moves
   the original. `clone(&s)` is the
   same instruction written by hand. Its contract is one `String` with the
   same bytes; whether a host copies, shares or copies on write is
   unobservable. `StringClone` is the only copy the IR names; no other heap
   type copies implicitly.
3. **Duplication.** Any other value is duplicated by the extern `clone(&x)`,
   where an instance exists. A type without one cannot be duplicated.
   Identity (RFC-0012) says nothing about copying.
4. **References are types.** `&T` and `&mut T` are made by `&place` and
   `&mut place`, held in bindings, passed to calls and received by
   parameters. `*r` yields the `T` when `T` is a word and is a type error
   otherwise. Assigning through a `&mut T` stores into the storage it names.
5. **Temporaries.** An expression that names no place is lent through a
   temporary storage bound to its value: `(a + b).to_string()`,
   `f(&(a + b))`. The temporary is a storage like a local — the loan names
   it, exclusion reaches it, and it is released where a `let`'s storage is.
6. **Exact unification.** A `&T` is never accepted where a `T` is expected,
   nor a `T` where a `&T` is. The program writes `f(&x)`, `g(*r)` or
   `g(clone(r))`. The only reads through a reference without `*` are an
   operator's `&word` operand (RFC-0020) and a captured word (rule 10).
7. **Parts.** A take of a place moves that place; a store into a place gives
   it and everything under it a value again; a store cannot reach through a
   place that is gone. Two places overlap when one is a prefix of the other.
   A reference to a moved place, or to a storage some place inside which has
   moved, is a use after move.
8. **Exclusion.** While a `&mut` to a storage is live, no other name of that
   storage — the place, a `&`, another `&mut` — is read or written. While a
   `&` is live, the storage is not assigned or moved.
9. **Holders.** A loan is held by whatever names the reference: a value, a
   storage the reference was assigned into, an object or closure built from
   it, a spawn's handle. A holder is live from its definition to its last
   read, counted as liveness counts uses, and a use of a holder is a use of
   every storage it lends: `let it = as_iter(&v)` makes `next(&mut it)` a use
   of `v`, and `v` is dropped after the last use of `it`. Where a reference
   may be stored and returned is RFC-0064.
10. **Parameters and captures.** A parameter's mode is its type:
    `Fn(T, &U, &mut V) -> R`, with no mode beside it. A lambda's parameter
    types come from unification with the type of the function that receives
    it. A lambda's captures are the names the checker resolved from outside
    its body, frozen for the lowering. A closure owns its captures: inside
    the body a captured word, or a captured reference, is seen at its own
    type and copied at each use; a captured name of any other type is seen
    as `&T`. How a capture is read is a solver decision, settled when the
    name's type resolves.
11. **Calling a closure lends it.** `f(x)` on a local `f` lends `f`, so `f`
    may be called again; passing `f` by value moves it. An inner lambda that
    re-captures a name the enclosing lambda captured takes the owned `T`,
    which is admitted only where `T` is a word; any other type is refused at
    the inner lambda, naming the name and the type — the rule of Rust's `Fn`
    closures.

**Why.** A copy exists only where the type is a word, and every other
duplication is visible in the program; no two names denote one storage except
through a reference, and the reference is the one thing checked, as a
dataflow over one body's liveness. Exact unification is what lets the solver
pin a lambda's parameter type from the extern signature alone. `String`
reuse reads as reuse at the cost of one named instruction, and immutability
lets a host make that instruction a refcount.
**Rejected.**
- References outside the type system — a hidden runtime copy at
  every binding, and `Clone` required of every extension type.
- Implicit `T`↔`&T` coercion — two candidates at every lambda parameter; the
  solver cannot pin the type.
- A pure move rule for `String` — every reuse of a name would be `clone(&x)`.

## RFC-0024: A pattern matched against a reference binds references

Status: Accepted

1. A pattern's dimension is its source's, never the pattern's.
2. Against a value, each part reads as its type reads — a word or a `String`
   copies (RFC-0018), anything else moves out. A value with a part moved out
   is partly moved: its other parts may be read, a use of the whole is a use
   after move.
3. Against a `&T`, the pattern is matched against `T` and every name it binds
   is a `&part`: `[a, b, ..] = &xs` binds `a: &Int`. The source is lent for
   the match and nothing moves; a literal is compared through the reference.
4. A `Some` whose payload is a `None` has no storage to lend: the payload is
   the depth word (RFC-0039), so `Some(v) = &opt` at `Some(None)` binds `v`
   to that `None` by value. A reference to a `None` is that `None`.
5. A source whose type is still a variable has no dimension yet: the pattern
   is checked against a referent of its own, joined when the head resolves.
   A head that stays open reads the value — the least element (RFC-0042).
6. `&` is written on the source only: no `&` in a pattern, and no context
   bind through a reference, since a context holds no reference (RFC-0014).
   A `&mut` source, written or held, is matched as a `&` source is: each
   name binds a shared reborrow of its part (RFC-0029 rule 3), and the
   names read side by side.
7. A reference names a field, an index, or a variant's payload of its
   storage.

**Why.** A pattern is several reads of one expression at once, so it takes
that expression's dimension. Binding `&Int` rather than copying the word is
the price of one dimension; `*a` names the copy.
**Rejected.**
- A per-binding dimension (`{ &name }`) or a lifting pattern (`&[a, b]`) —
  two dimensions in one expression.
- Copying words through a reference and borrowing the rest — a third rule
  keyed on type.

## RFC-0029: Exclusion is checked as the source wrote it; a reference to a reference is a reborrow

Status: Accepted

1. The exclusion rule (RFC-0018) is checked on the body as lowering wrote
   it, before any pass. A re-check after a pass is an assertion on the pass;
   a conflict found only there is a defect in the pass, not a refusal shown
   to the reader.
2. The move rule is checked once, on the source shape, and never after
   optimization: optimization erases the moves it reads, so a second run can
   only repeat the first under other names.
3. `&r` where `r: &T` is `&T` naming what `r` names; `&mut r` where
   `r: &mut T` is `&mut T`; `&r` where `r: &mut T` is `&T`; `&mut r` where
   `r: &T` is a type error. No type `&&T` exists.
4. Where the place's type is still a variable, `&place` is a lend decision
   settled to a reborrow or a plain reference when the type resolves; it
   still carries a parameter's type back to the place. A lend nothing
   resolves closes on its least element, a plain reference.
5. A `&mut T` reaches a position of type `&T` — an argument, a flow, a view
   (`&[T]`, `&str`) of what it names — as the shared reborrow of rule 3, and
   the value it came from keeps `&mut T`. A candidate taking the `&T` takes
   such an argument directly (RFC-0043 rule 1). Where references meet — the
   branches of an `if`, the arms of a `match` — a `&T` and a `&mut T` meet
   at a `&T`, which each `&mut` reaches by that reborrow. A `&T` never
   reaches a `&mut T`.

**Why.** A check that needs an optimization to see a conflict is not a check
of the source. Typing `&r` as `&&T` while lowering it as a reborrow left the
validator to reject what the checker accepted, and every use of `&r` wants
what `r` names.
**Rejected.**
- Two-phase borrows — `f(&mut x, x.len)` is a conflict.
- Reads through a `&mut` while a shared reborrow of it is live — Rust admits
  some; this rule does not.

## RFC-0064: A reference's extent is its loans

Status: Accepted

1. **A loan's storage is a local or a parameter.** A parameter of reference
   type holds a loan on the storage it names outside the body; inside the
   body that storage is the parameter itself, `Loan::Param(i)`. A reference
   derived from it holds that loan, and reading or writing through the
   parameter while such a reference is live is a conflict, as for a local.
   A parameter whose type holds a loan (a capturing lambda) starts with a
   loan on itself the same way.
2. **A body has a summary**: the `Param` loans, with mutability, its result
   may hold. A lambda's summary also carries the loans it captures. It is
   inferred; there is no syntax.
3. **A call substitutes.** A call's result region is the callee's summary
   with each `Param(i)` replaced by the argument's region, joined by union; a
   lambda call adds the lambda's captured loans. A callee without a summary
   gives the union of every argument's region.
4. **Recursion is a fixpoint.** Summaries are computed in call-graph
   component order; a cycle starts every member at the empty summary and
   climbs to the least fixpoint. Recursive bodies may return references.
5. **One refusal rule.** A reference, or a holder of loans, may not be stored
   where it outlives what it names: a container element, an object or tuple
   field, a context, a spawn. A body's result may hold `Param` loans and no
   local's (`ReferenceToLocalLeavesBody`). A lambda may capture a reference
   and is then a holder under RFC-0018 rule 9; a view may not be captured
   (`ViewCaptured`), since a capture is one word. A reference to a closure's
   capture of an owned value names the closure's own storage and cannot
   leave the closure body. Whether a result's type may leave at all is
   decided by the type checker; which storage it names, by the region check
   over the MIR.
6. **Externs declare no summary.** An extern call takes the union of its
   arguments' regions (rule 3); an argument by value carries no region
   unless its type holds a reference.

**Why.** The borrow check is already a per-instruction dataflow over
regions whose join is union; what it lacked was an identity, readable by the
caller, for a loan whose storage is outside the body. `Param(i)` is that
identity, and with it the checker states what a signature would have.
**Rejected.**
- Written lifetime parameters — the checker has the information; where the
  inference is ambiguous the answer is the union, and a refusal names both
  origins.
- Lifetimes as type parameters — extent is per instruction; on types it is
  the lexical-region mistake.
- A declared summary on an extern — for every handler that lends one
  parameter it equals the union, and it would rest on a rule the macro
  states rather than one Rust checks.
