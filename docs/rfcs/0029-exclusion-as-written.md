# RFC-0029: Exclusion is checked as the source wrote it, over every holder

Status: Accepted
Date: 2026-09-16
Extends: RFC-0018

## Ruling

The exclusion rule of RFC-0018 is checked on the body as lowering wrote
it, before any pass has promoted, moved, or removed an instruction, as
the move rule already is. A pass that changes the body is checked again
after it runs, and a conflict found only there is a defect in the pass.

The move rule is not checked again: optimization erases the moves it
reads — a binding whose only use is dead is gone, and with it the second
use that was the error — so a second run cannot confirm the first and can
only repeat it, under the register names the optimizer left. The move
check runs once, on the shape the source wrote.

A loan is held by whatever names the reference: a value, a storage the
reference was assigned into, an object or closure built from it, a
spawn's handle. A holder is live from the instruction that defines it to
its last read, and a read of a storage is a read of the holder it is; the
check counts the same uses liveness does.

A parameter or capture of reference type holds a loan on the storage it
names outside the body. Inside the body that storage is the parameter
itself: a reference derived from the parameter holds the parameter's
loan, and reading or writing through the parameter while such a reference
is live is a conflict, as it is for a local.

A reference to a place that is itself a reference is a reborrow: `&r`
where `r: &T` is `&T` naming what `r` names, `&mut r` where `r: &mut T`
is `&mut T`, and `&r` where `r: &mut T` is `&T`; `&mut r` where `r: &T` is
a type error. No type `&&T` exists, in the checker or in the body.

Which of the two `&place` is depends on the place's type, so where that
type is still a variable the rule is deferred, never skipped: the lend is
a decision, and it settles to a reborrow or to a plain reference the
moment the place's type resolves. `&place` is a reference of the written
mutability either way — only what it names is open — so a lend still
carries a parameter's type back to the place. A lend nothing ever resolves
closes on its least element, a plain reference, since a place a program
never made a reference is not one. The same decision refuses a lambda's
capture of a name that resolves to a reference (RFC-0018).

## Rationale

The check ran after SSA promotion and missed nothing in practice only
because promotion turned a slot that held a reference into the reference
value; on the body as written, `r = &x; x = 2; *r` passed, since the walk
that decides which holders are live counted value uses and not the read of
`r`. A check that depends on an optimization to see a conflict is not a
check of the source.

A parameter of reference type had no loan, so nothing derived from it
could conflict with anything: a lambda writing through its `&mut`
parameter while a reborrow of it was live was accepted.

Typing `&r` as `&&T` while lowering it as a reborrow left the validator
to reject the body the checker accepted. A reference is never data
(RFC-0018); a reference to one would make the inner reference data, and
nothing needs it: every use of `&r` wants what `r` names.

## Not built

- No lifetimes and no variance; a reborrow's region is the region of
  what it reborrows, joined with the reference it went through.
- No two-phase borrows: `f(&mut x, x.len)` is a conflict.
- No reads through a `&mut` while a shared reborrow of it is live; Rust
  admits some of these, this rule does not.

## Consequences

- `optimize` runs `check_borrows` in its first pass, beside
  `check_moves`; `validate` runs `check_borrows` after optimization and
  does not run `check_moves`.
- `borrow_check` counts holder uses with `Loans::uses_with_storage`.
- `Loans::build` starts every reference-typed parameter and capture with
  a loan on itself.
- The checker types `&place` of a reference-typed place as the reborrow
  and rejects `&mut` of a shared reference.
- A place whose type is still a variable opens `Decision::Lend`, which
  `Solver::lend` answers from the resolved head and `solve` closes on
  `Lend::Reference` when nothing resolves it.
