# Design of the type checker model

How the model is built, as it stands. What the language rules is in `TERMS.md`, and this says
how those rulings are held in code. This file is temporary, as `TERMS.md` is: once the model
is complete, its content goes into the docs.

## 1. What there is

| Thing | What it is | Its life |
|---|---|---|
| a class | an equivalence class of types | undecided, holding the names of its parts, then decided, holding its head and its own parts. It changes once. |
| a constraint | a bound, which is a relation of two classes: entering, being equal, being passed; or a suspended constraint, which waits for a head and then adds constraints: `as _`, the exit of `?` | recorded, looked at, perhaps many times, then finished. The store holds it throughout. |
| what a class holds of constraints | guards, constraints that hold it apart, waiters | set and lifted as constraints are looked at and finished |
| the connections of a class | each connection made of it: the two ids connected and the item it went by | added to where the class is connected, merged where two classes are, never taken out. A decision adds none. |

## 2. The three operations

Each is done in one place, and there is no other way to do what it does.

| Operation | Contract | Place |
|---|---|---|
| connecting | Makes two classes one type. Neither decided: one class. One decided by solving or the text: the other is that class. One a declaration writes: nothing is merged, and the other is compared with it. Both decided: one head, parts connected in turn. | `operation::connect::connect` |
| looking at a constraint | Asks the law what the constraint comes to, and does what follows: finishes it, connects its pairs and records the constraints that take its place, or finishes it and refuses, or states what it waits on and holds apart. | `operation::judge::look` |
| deciding | Gathers what a rung offers, and gives those heads together through `Classes::deciding`. | `operation::decide::decide` |

`operation::settle::settle` takes them in turn: every constraint awake is looked at until none is,
then every class that may be decided is, and back to looking while anything was decided.

## 3. The law

The three operations know no rule of the language. Each asks a `Law`
(`operation::law`), which reads the store and answers. A law merges nothing, decides nothing
and finishes nothing.

| The law is asked | It answers |
|---|---|
| what a constraint comes to | `Looked`: it holds, with these pairs and adjustments, and these constraints in its place; it is refused; or it waits on these classes, holds these apart, and these pairs are one type already |
| which classes are one type by the constraints as a whole | pairs |
| where a part named on a class lies, once the class's head has no place of that name | `Place`: within another part, with an adjustment, or nowhere |
| whether a literal's class may take a head | yes or no |
| what head a rung offers a class | `Offer`: nothing, a head, or a refusal |
| what is refused of a class at the end | what is refused, and of which classes |

What a law refuses, it refuses of classes. What decided those classes, and every connection
that made them, is gathered in one place, `Outcome::gather`, once settling has run to its end.
That is the only place the connections of a class are read. A law is given the whole store to
read, so no type keeps a rule from reading them: what keeps it is that nothing a law answers
has a place to carry them, other than a refusal's own items.

A rule cannot leave a constraint half done: a guard, a wait and a finished constraint all follow from
what the rule answers, and none is the rule's to do. A law of no language, one that keeps the
contracts and states rules that are wrong for this language, runs through the same three
operations, which `operation::tests` does.

The language's law is `operation::language`, one rule in one place:

| Rule | Where |
|---|---|
| entering: the parts of one name are one type, and the head that enters is one the join took in | `enter` |
| being equal: the parts of one name are one type, and the two heads are of one layer | `equal` |
| being passed: the type given is read as far as it is written, and each index of the argument's head is loosened or left; else through one way of declared functions | `passing` |
| `e as _`: it asks `e` no head, waits until the type of `e` has a head at every depth, and then that type enters | `alone` |
| the exit of `e?`: it waits for the head of what `e` exits with, and then an `Error` made by that head enters the error of the body's result | `exit` |
| the functions declared: a table, and no rule | `functions` |
| classes that enter one another are one class | `enter::entering_one_another` |
| a part named on a reference is read through it, and is read where it lies, whatever it was where it was named | `Language::place`, `connect::take_in` |
| the rungs: the join of what enters and of what is written, the one head from above, the head the parts named make, a literal's fallback, `!` | `rungs` |
| at the end: a class that is its own part, a class still undecided | `rungs::end` |

A rule reads through a `View`, which is the store as it is, or with heads supposed of some
undecided classes. The rung that gives a head from above uses that: a class takes the one head,
of those the constraints waiting on it say they will hold it against, that every such one would
have it take, and "would" is the same rule looked at again with the head supposed.

## 4. What makes deciding safe

A class's head is the join of what enters it, so it is decided once, and only when nothing can
still be merged into it.

| What may still merge into a class | How the store sees it |
|---|---|
| a constraint not finished that says it may | The constraint holds classes apart, and every class below one of those is guarded by it. |
| the class is named as a part on an undecided class | The names the store keeps. A part is decided no earlier than the class it is named on, and where the part holds that class in turn, the two are decided together. |
| looking has not run to its end | Deciding is taken only once no constraint is awake. |

`Classes::deciding` gives a `Deciding`, which finds what is held once and borrows the store for
as long as it lives. Nothing is recorded and nothing is merged between that finding and the
decisions made through it. A head decided has the class's own parts: the part already named on
it, or a class made anew. So what gives a head gives its constructor and members alone, and a
bound connects its parts.

A constraint is awake when it is recorded and when a class it waits on is merged or decided, so a
constraint nothing has happened to is not looked at again.

## 5. What is not tested, and what is not there

- **Guards going down as structure grows, in a program.** The ledger gives a part that comes
  to be named, or merged in, the guards of the class above it, and its own tests hold it to
  that. No program run through the law shows it: a guard is read only when a `Deciding` is
  made, after looking has run to its end, and by then every constraint that waits has stated
  its guards anew.
- **The order of the last two rungs.** A literal's fallback and `!` offer heads to no class in
  common, one taking a class with a domain and the other a class with none, so which stands
  first shows nowhere.
- **Deciding from above, through what a class enters.** A head is taken from above only where
  every class the class enters would take it too, joined with what has entered that class
  already. Leaving out either, the asking of the class above or the join, fails no test.
- **What names the parts a head is made of.** A head joined with what the parts say is decided
  by the items that named those parts too, for a refusal to name. Leaving them out fails no
  test.
- **Where the rung of parts stands.** Putting it after the literal's fallback, or before the
  head from above, fails no test. It gives a head only where none is asked from above, so its
  order with that rung shows nowhere, and no program has been found where a head made from
  parts gives a literal its head.
- **When a refusal is gathered.** Gathering each refusal as it is made, and not at the end,
  fails no test: no program has been found where a class a refusal is of is connected again
  after the refusal. The orders the tests run differ in which class is asked first, and not
  in which constraint is looked at first.
- **A function value passed to a function type a row writes**, left as a `todo`.
- **That an argument is passed to a parameter of a lambda and does not enter it.** A parameter
  of a lambda writes no type, and a hole takes what comes as it comes; two arguments of one
  parameter, as a recursion gives, are then one class that joins what enters it, as it would
  were they to enter. A declared function's parameter writes a type, and there the two are
  told apart and tested.
- **That solving ends.** A call within a copy of what it calls makes no copy, and making one
  there does not end; but each copy names the closures its body defines anew, and that no
  chain of those goes on without end is argued and not tested.

## 6. The text

`src/text` writes a program as the operations that record it, one to a statement, and reads
it into a store with no judgment of its own (`src/text/mod.rs` states the text). It is made
for a generator: a name is a class from where it is first written, so a text that is read is
a program, but for the few `record::record` names. Assertions say what solving is to come to
and are no part of what is recorded.

`record::run` solves a text in each order of asking, verifies the store, holds the text to its
assertions and compares the refusals of the orders. `cargo run --bin check -- <paths>` runs
files of programs. `programs/` holds the cases of the earlier model written so, by the rules
in `programs/README.md`.

What is not there yet:

- **The order of statements is not shuffled.** `run` changes which class is asked first, and
  not which constraint is recorded or looked at first. Shuffling the statements would, and is
  what may show the two things section 5 says no test observes.
- **No generator.** Programs are written by hand or translated.
- **The tests of the law are not in the text.** `operation::tests` records each program by
  hand through `operation::testing::World`, and `operation::calls` writes each template by
  hand. The text writes what both write, a lambda included.
- **No assertion of a function value's type, nor of a class of a body.** What a lambda gives
  is asserted of the result of a call.

## 7. How other systems hold a constraint that waits

Gathered by a research agent from each project's sources on 2026-11-10, keeping only what it
marked as read. Not read by the author of this file.

| System | Who holds a waiting constraint | What keeps a variable from being fixed | Why the outcome does not depend on order |
|---|---|---|---|
| GHC | the solver: an inert set and a work list | touchability, by level; defaulting only after solving returns | no general claim: the split of phases |
| rustc | the fulfillment context, each obligation with the variables it is stalled on | nothing for each variable: fallback runs after one selection to a fixpoint | none claimed |
| Java, JLS §18.4 and §18.5.2.2 | the constraints of arguments not pertinent to applicability, each with input and output variables | a variable is resolved only in a smallest set closed under its dependencies; an output variable of a pending constraint stays unresolved | the smallest-set rule |
| Kotlin | the completer: postponed atoms | a readiness computed anew each turn | none: an order of priorities |
| Swift | the constraint system and its graph | a set of the constraints that delay a variable, used as a penalty and not as a block | none: search and ranking |
| OmniML, arXiv 2511.10343 | a wait list on each variable | a set of guards on each variable, one for each suspended constraint that captures it | proved |
| LVish, POPL 2014 | pools of handlers | a pool is quiesced before a value is frozen, and a write after the freeze is a fault | the same answer or an error |

The guards and waiters of §4 are OmniML's. The order of names, with a cycle decided together,
is JLS §18.4's smallest closed set. None of these systems holds the rule by types: each uses an
order of phases, a predicate computed again, or an assertion while running.
