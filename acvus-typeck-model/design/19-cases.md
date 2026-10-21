# 19 Cases and their runner

Rests on the crate's I2, I3 and I4, and on notes 13–15 (outcome and exploration,
the certificate, I\*).

## Meaning

A case is a file that states a program the model takes, at the level the model
takes it (classes, the values passing into them, uses, lends, projections,
bindings, groups), and what solving it must give. A case is written without Rust:
one runner, written once, reads every case, solves it, checks it, and reports. The
form follows Rust's look and rustc's compiletest's conventions (`//@` directives,
must-pass and must-fail), and is no complete grammar: it says everything the model
takes and everything a test of it asserts.

Three scopes are tested, each locally:

- **A program**, solved in every order (I2), with the checks of notes 14 and 15 on
  every final state.
- **A phase**, along one order the case writes, by what each phase reports it did
  (`src/trace.rs`, below).
- **A primitive** the spec defines on types alone: the join and `≤` (l.80–97), and
  a coercion (l.99–116), on the case's own declarations.

## Not owned here

- What the model decides: every expectation is checked against the model, never
  the model against the case's own reading of the spec.
- Generating cases (note 16).
- The primitives' tests that need Rust's values: `src/{ty,sig,decl,class,join,coerce}/tests.rs`
  stay as they are, over `fixture.rs`.

## Form

A case is `cases/**/*.case`. `//` begins a comment, `//@` a directive. A name is a
word of letters, digits, `_` and `?`, or words joined by `::`; any other name is
written in backquotes: `` `fill_all(..)` ``. Class names and node names are apart.
Classes, items and declarations are recorded in the order written, as
`ProgramBuilder` records them, so that a case and the Rust test it replaced reach
the same states.

```
//@ spec: Example 2, l.377–390
extern type Vec<T>;
extern fn new<T>() -> Vec<T>;
extern fn fill_all(&mut Vec<u8>);

let A;
let `new()`;
let `&mut a`;
let `fill_all(..)`;
`new()` = new() @ new;              // a use, the node `new`
A = `new()`;                        // `new()` passes into A
`&mut a` = &mut A @ lend;           // a lend, the node `lend`
`fill_all(..)` = fill_all(`&mut a`) @ fill_all;

must pass;
assert A: Vec<u8>;
assert lend == lend &mut place;
```

**Types**, as a declaration, an expectation or the text writes them: `&T`,
`&mut T`, `()`, `(T,)`, `(T, U)`, `[T]`, `[T; n]`, `!`, `{f: T}`, `E{V(T)}`,
`Fn/n` (a function value), `fn(P) -> R`, the scalars, `String`, `str`,
`Option<T>`, `Result<T, E>`, a declared struct `Pair<A, B>` (one argument for each
field, in declared order), and a declared extern-defined type `Vec<T>` (its
identity last where it has one). In a declaration a type parameter's name is that
variable, and `#` the identity. A type the text writes is its outermost
constructor, each part a class it names: `` let `Some(1u8)`: Option<S>; ``.

**Declarations.**

| Written | Declares |
|---|---|
| `extern type Vec<T>;`, `extern type F<T, #>;` | an extern-defined type, its arity, `#` its identity |
| `struct Pair { l, r }` | a module's struct, its fields in order |
| `extern fn f<T>(&mut Vec<T>, T) -> R;` | a function (`Role::Fn`); no `->` for `()` |
| `#[cast] extern fn …;`, `#[conversion] extern fn …;` | a cast, a conversion |

A name declared several times has its candidates in the order declared, written
`f#0`, `f#1` as `DeclId` numbers them.

**Groups and classes.** `group g { … }` holds a binding group's text, and
`group λ in run { … }` a lambda's, lying in `run` (`ProgramBuilder::group_in`); a
group is reopened by writing it again, and text outside any group is the entry's.
In a group, `fn g(A, B) -> R;` names its function by its parameters' and result's
classes, and `universal A;` a universal.

| Written | Class |
|---|---|
| `let A;` | a type variable (`Origin::Var`) |
| `let 0: lit 0;`, `let x: lit 1.5;` | a literal's LitVar |
| `` let `Some(a)`: Option<S>; `` | a type the text writes |
| `` let `@cfg`: host {name: N}; `` | a type the host gives |
| `` let F: lambda λ; `` | the function value of the lambda `λ` |

**Statements.**

| Written | Item |
|---|---|
| `S = A;` | `A` passes into `S` (`Place::Slot`); `#[by_name]`, `#[through_ref]` before it for the other places |
| `A == B;` | an equation |
| `R = f(X, Y) @ n;` | a use of `f`, the node `n` |
| `R = (F)(X) @ n;` | a call of the function value of `F` |
| `R = &P @ n;`, `R = &mut P @ n;` | a lend |
| `R = P.x @ n;`, `P.0`, `*P`, `P[]`, `P as Some`, `P as E::V` | a projection |
| `B = bind P via [p, q] @ n;` | a binding at `P` along the projections `p`, `q` |

`let R = …;` gives the class `R` and then the item.

**Directives.**

| Directive | Means | Where none is written |
|---|---|---|
| `//@ spec: …`, `//@ hole: H7` | what the case witnesses | — |
| `//@ reading: deferral = reading, captures = environment` | a run under these readings, the others the default; each line a run, `default` the defaults | one run, `Readings::default()` |
| `//@ orders: all` | every order, which must end within 100 000 states | `all` |
| `//@ orders: sampled 200` | 200 sampled orders (note 13, reading 2) | — |
| `//@ order: solve A, decide f, fallback` | one order: the entry's steps, then its first step until none is left, falling back where none is | — |
| `//@ order g: solve A` | the steps of the group `g` in that order; every other group takes its first step | — |

A step names a class by any of its members' names, as the arena names them
(`A`, `A.Payload(Some)`), and a node by its name; it must be a step the state may
take then, and `fallback` only where no step is left in the entry (l.297, l.299).

**Expectations.** Each holds of every outcome, and an I\* expectation of every
final state, of every run.

| Written | Holds where |
|---|---|
| `must pass;`, `must fail;` | the outcome is accepted, refused |
| `assert A: Vec<u8>;`, `assert S: none;` | the class has the value, or none |
| `assert f == f#1;` | the node's decision, as below |
| `assert refused infinite X;`, `no_join`, `no_value`, `conflict at "…"`, `undeclared` | some refusal is that, of that class where one is written |
| `` assert failing [`g@2/A ⇝ g@2/B`]; `` | the failing checks' relations are exactly these, in the outcome's order |
| `` assert check `1u16 ≤ push.Var(Type(0))` == conversion narrow#0; `` | a check of the relation has that coercion |
| `assert istar completeness probe;` | an I\* violation; the final states' violations are exactly those written, none where none is |
| `assert outcomes 2;` | the number of outcomes; one where none is written and no block is |
| `assert partition {A, B};` | an equivalence class of exactly these |
| `assert certificate set X: B fails ["…"];`, `assert certificate set probe == probe#1 fails ["…"];`, `in g` after `certificate` | the certificate of every final state, so changed, fails, the `what`s over them all exactly these |
| `assert trace "join A took …";` | with an order written: a report, or a refusal a step made, is that line; a refusal whose text holds `"`, a conflict's, is asserted by `assert refused` |
| `` assert split records `S ≤ A.Payload(Some)`; ``, `` assert no node records `…`; `` | with an order written: a report of the phase (`record`, `unify`, `join`, `split`, `node`, `fallback`) records the relation |
| `assert join(A, C) == B;`, `== none`; `assert le(A, B);`, `assert not le(B, A);` | the primitive, on the declarations |
| `assert coerce(&mut A, &B) == reborrow + cast cast2#0;`, `assert along_le(…) == …;` | the coercion, on the declarations |

A decision is written `f#0`, `unreached [f#0, f#1]` (`unreached` alone for any),
`never`, `lend &mut place`, `lend & through &mut` (`through _` where the value read
decides no mutability), `applied`, `applied through &`, `read`,
`bind lend & place`, `no result`, `calls {λ}`, `undecided`. A coercion is written
`identity`, `undecided`, `never`, `conversion f#0`, steps joined by `+` (`reborrow`,
`cast f#0`, `shape`, `wrap V(…)`), or `none`, `compositions […]`, `conversions […]`
for none. Both are compared with whitespace left out. A relation, a class and a
node are named as the model names them, a copy's as its instance does
(`nest@23/n == 0`, `put(..)#λ/V ≤ X`).

**Outcome blocks.** `outcome { … }` holds what one outcome and the final states
reaching it meet. Where a case writes blocks, its runs have exactly as many
outcomes, and each block meets a distinct one; what lies outside the blocks holds
of every outcome, and its I\* expectations join each block's.

**A hole.** A case under `//@ hole:` asserts what the model gives now. Where it
also writes `wanted { … }`, what the spec's text, its examples or the RFCs it cites
want, as `holes.md` states it, it is an expected failure: `XFAIL` while the model
does not meet the block, `XPASS`, a failure, once it does, so that `holes.md` is
updated with it.

**Golden outcomes.** Next to each case, `*.outcome` holds every run's outcomes in
full, the I\* violations of the final states reaching each, and a written order's
trace. A case whose golden differs, or has none, fails; `BLESS=1` writes them anew.
This catches a change no expectation names.

## The runner

`src/cases/` (test only, the standard library alone, I4): `parse.rs` reads a case,
an error naming its line; `build.rs` makes the `Decls` and the `Program`;
`run.rs` explores, samples or follows the written order, handing each final state
to the certificate's and I\*'s checks and to the changed certificates; `expect.rs`
compares; `render.rs` writes what the model gives as a case writes it. One test,
`cases::cases`, runs every case, each in a thread named after it, prints a table of
`PASS`, `FAIL`, `XFAIL` and `XPASS`, and fails where any is `FAIL` or `XPASS`.
`CASE=` runs the cases whose path contains it.

## The phases' reports

Each phase function is `pub(crate)` and returns what it did. A relation is named
by its place in the log, a class by its `ClassId`, named in the state the trace
ends in. `State::start`, `step`, `settle` and `fall_back` return a `Trace` of
`Report`s, in the order the phases ran; `of_group` and the solver's callers drop
it. `settle`'s trace holds what changed the state.

| Phase | Function | Returns |
|---|---|---|
| recording | `record` | `Entry`: its place, and what unification refused at it |
| | `record_origins`, `record_equations`, `record_items`, `call`, `record_call` | the relations recorded |
| unification | `hold` | the refusal it made |
| | `link_views`, `link_view` | `Linked`: the container, the element, the relations recorded, the refusal where no view is |
| the join | `refuse_infinite` | the refusal |
| | `join_lit_lowers`, `merge_cycles`, `join_lit_at` | the equations recorded |
| | `join` | `Took`: the kind it held, the head it joined, nothing, or no join |
| splits | `split_at` | `Split`: one class, the parts' relations, unsolved, or no parts |
| | `unify_lit_boundaries`, `split_parts` | the relations recorded |
| nodes | `decide`, `decide_at_once` | `Decided`: the node, its decision, the relations recorded |
| | `record_decision`, `record_at_once` | the relations recorded |
| fallback | `take_defaults` | `FellBack`: each class, its default, the refusal where it could not take it |

## Run

- The 38 tests of `examples.rs` are 51 cases under `cases/spec`, `cases/holes`,
  `cases/environment`, `cases/istar` and `cases/certificate`, a test's programs
  and readings with other expectations each a case of its own. Over the 38, the
  distinct dump lines of every exploration (`Exploration` with its final states'
  fingerprints) and of every sampled order (fingerprint and outcome) are the same
  set from the cases as from the tests at 68ea8114, and the same from the tests
  with the phases' reports as without them; `examples.rs` is removed. Each case
  asserts what its test asserted: a pattern the test matched (`Unreached(_)`, a
  refusal's kind) as the same pattern, and a text it matched in part
  (`starts_with`, `contains`) whole, as the model gives it. Example 14's copied
  decisions, which its test found by the end of their names, are named whole.
- 13 cases under `cases/phases` test each phase along a written order, by the line
  of the spec it rests on: recording (l.226, l.234–236), the join (l.80–90, l.176,
  l.275, l.280), splits (l.284–291), unification (l.161, l.165–170), nodes (l.252,
  l.267, l.331–344) and fallback (l.297–299). 2 cases under `cases/primitives` test
  the join, `≤` and coercion on the declarations of `fixture::decls()` without
  `Box`, coercion with a conversion `u16 → u8` besides.
- H6's case writes the narrative's `q` as `wanted`, and is `XFAIL`.
- Every case passes, and every golden is the model's.
