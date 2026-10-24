# 19 Cases and their runner

## Meaning

A case is a file that states a program the model takes, at the level the model
takes it (classes, the values passing into them, uses, lends, projections,
bindings, groups), and what solving it must give. A case is written without Rust:
one runner, written once, reads every case, solves it, checks it, and reports. The
form follows Rust's look and rustc's compiletest's conventions (`//@` directives,
must-pass and must-fail), and is no complete grammar: it says everything the model
takes and everything a test of it asserts.

Three scopes are tested, each locally:

- **A program**, solved in every order.
- **A phase**, along one order the case writes, by what each phase reports it did
  (`src/trace.rs`, below).
- **A primitive** on types alone: the join and `≤`, and a coercion, on the case's own
  declarations.

## Not owned here

- What the model decides: every expectation is checked against the model.
- Generating cases.
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
| `let D: out Result<U8, E>;` | a type the embedding declares for the run's result, which may hold an `Error` |
| `` let F: lambda λ; `` | the function value of the lambda `λ` |

**Statements.**

| Written | Item |
|---|---|
| `S = A;` | `A` passes into `S` (`Place::Slot`); `#[by_name]`, `#[through_ref]` before it for the other places |
| `A == B;` | an equation |
| `R = f(X, Y) @ n;` | a use of `f`, the node `n` |
| `R = (F)(X) @ n;` | a call of the function value of `F` |
| `R = ?(X) into Q @ n;` | `e?` of the operand `X`, its exit's value of the class `Q` |
| `R = &P @ n;`, `R = &mut P @ n;` | a lend |
| `R = P.x @ n;`, `P.0`, `*P`, `P[]`, `P as Some`, `P as E::V` | a projection |
| `B = bind P via [p, q] @ n;` | a binding at `P` along the projections `p`, `q` |

`let R = …;` gives the class `R` and then the item.

**Directives.**

| Directive | Means | Where none is written |
|---|---|---|
| `//@ orders: all` | every order, which must end within 100 000 states | `all` |
| `//@ orders: sampled 200` | 200 sampled orders | — |
| `//@ order: solve A, fill B, decide f, fallback` | one order: the entry's steps, then its first step until none is left, falling back where none is | — |
| `//@ order g: solve A` | the steps of the group `g` in that order; every other group takes its first step | — |

A step names a class by any of its members' names, as the arena names them
(`A`, `A.Payload(Some)`), and a node by its name; it must be a step the state may
take then, and `fallback` only where no step is left in the entry.

**Expectations.** Each holds of every outcome, and an I\* expectation of every
final state, of every run.

| Written | Holds where |
|---|---|
| `must pass;`, `must fail;` | the outcome is accepted, refused |
| `assert A: Vec<u8>;`, `assert S: none;` | the class has the value, or none |
| `assert f == f#1;` | the node's decision, as below |
| `assert refused infinite X;`, `no_join`, `no_fill`, `no_value`, `conflict at "…"`, `undeclared` | some refusal is that, of that class where one is written |
| `` assert failing [`g@2/A ~> g@2/B`]; `` | the failing checks' relations are exactly these, in the outcome's order |
| `` assert check `X ~> probe.Param(0)` == conversion down#0; `` | a check of the relation has that coercion |
| `assert istar completeness probe;` | an I\* violation; the final states' violations are exactly those written, none where none is |
| `assert outcomes 2;` | the number of outcomes; one where none is written and no block is |
| `assert partition {A, B};` | an equivalence class of exactly these |
| `assert certificate set X: B fails ["…"];`, `assert certificate set probe == probe#1 fails ["…"];`, `in g` after `certificate` | the certificate of every final state, so changed, fails, the `what`s over them all exactly these; the class is named as the outcome writes it, a class an instance or a node made among them, and a function value in the value writes its Values after it, `Fn/1{λ}`, `Fn/1{}` naming none |
| `assert trace "join A: nothing";` | with an order written: a line of its trace other than a `record` is that line (The trace); a `refuse` line whose text holds `"`, a conflict's, is asserted by `assert refused` |
| `` assert split records `S <= A.Payload(Some)`; ``, `` assert no node records `...`; `` | with an order written: a `record` of the relation lies under a line of the phase (`record`, `unify`, `join`, `split`, `node`, `fallback`; The trace) |
| `assert join(A, C) == B;`, `== none`; `assert le(A, B);`, `assert not le(B, A);` | the primitive, on the declarations |
| `assert coerce(&mut A, &B) == reborrow + cast cast2#0;`, `assert along_le(…) == …;` | the coercion, on the declarations |

A decision is written `f#0`, `never`, `lend &mut place`, `lend & through &mut`
(`through _` where the value read decides no mutability), `applied`,
`applied through &`, `read`, `bind lend & place`, `calls {λ}`, `undecided`. A
coercion is written `identity`, `undecided`, `never`, `conversion f#0`, steps
joined by `+` (`reborrow`, `cast f#0`, `shape`), or `none`,
`compositions […]`, `conversions […]` for none. Both are compared with whitespace left out. A relation, a class and a
node are named as the model names them, a copy's as its instance does
(`nest@23/n == 0`, `put(..)#λ/V <= X`).

**Notation.** What the model writes is ASCII, the names a case gives kept as it
writes them. A relation is `A <= B` (a lower bound), `A <=p B` (a part lower
bound), `A ~> B` (a boundary), `A = B`, `A: <kind>`, `A: <kind known in part>`,
`A: <LitVar>`, `A: reference`, `A: no reference`, or `A: Values contains λ`. A kind and a type are written as a
case writes a type, `Fn(P -> R)` for a function's; a kind known in part is
`ref T`, `field x: T`, `component i: T`, `index T`, `variant V: T` or
`variant E::V: T`; a LitVar `?{1, 2}: int`. What no type holds is `<a> vs <b>`: two
constructors; a LitVar and a constructor, or `<LitVar> vs another family`; a
constructor and a kind known in part; `reference` or `no reference` and a
constructor. A run of places in the log is `#a-#b`. A certificate's and an
expectation's messages write `<=`, `~>`, `!=` and `join(A, B)`.

**Outcome blocks.** `outcome { … }` holds what one outcome and the final states
reaching it meet. Where a case writes blocks, its runs have exactly as many
outcomes, and each block meets a distinct one; what lies outside the blocks holds
of every outcome, and its I\* expectations join each block's.

**Golden outcomes.** Next to each case, `*.outcome` holds every run's outcomes in
full, the I\* violations of the final states reaching each, and a written order's
trace. A check whose conversion brings a lambda the Values where it reaches do not
name writes them after it, `unnamed: {μ}`. A case whose golden differs, or has none,
fails; `BLESS=1` writes them anew.
This catches a change no expectation names.

## The runner

`src/cases/` (test only, the standard library alone): `parse.rs` reads a case,
an error naming its line; `build.rs` makes the `Decls` and the `Program`;
`run.rs` explores, samples or follows the written order, handing each final state
to the certificate's and I\*'s checks and to the changed certificates; `expect.rs`
compares; `render.rs` writes what the model gives as a case writes it. One test,
`cases::cases`, runs every case, each in a thread named after it, prints a table of
`PASS` and `FAIL`, and fails where any is `FAIL`.
`CASE=` runs the cases whose path contains it.

## The phases' reports

Each phase function is `pub(crate)` and returns what it did. A relation is named
by its place in the log, a class by its `ClassId`, named in the state the trace
ends in. `State::start`, `step`, `settle` and `fall_back` return a `Trace` of
`Report`s, in the order the phases ran; `of_group` and the solver's callers drop
it. `settle`'s trace holds what changed the state.

| Phase | Function | Returns |
|---|---|---|
| recording | `record` | `Entry`: its place, and the `Unification` the classes took it with |
| | `record_origins`, `record_equations`, `record_items`, `call`, `record_call` | `Recording`: an `Entry` for each relation recorded, and each instance made (`Instantiated`: the use, its recorder, the group whose C it copies, the places of what it records) |
| unification | `hold` | `Unification`: what unifying did, and the refusal it made |
| | `Classes::unifying` | `Unifying`, whose `give_kind`, `give_fact`, `give_lit`, `give_reference` and `unify` return `Unified`: each `Union` in order (its `Cause`, the members of the two classes before it, the kind it made), and the `Conflicted` that stopped it (the two classes left apart with their kinds, or the class that holds what no type holds) |
| | `link_views`, `link_view` | `Linked`: the container, the element, the relations recorded, the refusal where no view is |
| the join | `refuse_infinite` | the refusal |
| | `join_lit_lowers`, `merge_cycles`, `join_lit_at` | the relations recorded |
| | `join` | `Joined`: its `Sources` (its equations and kinds, the solved lower bounds and part lower bounds it counts, the kinds known in part, those left out) and `Took`: the kind it held, the head it joined with what unification did, nothing, or no join |
| | `fill` | `Filled`: each place the class passes into with its value, and `Fill`: the one type they have, with what unification did as the class took it, and the refusal where it could not take it, or the refusal where they have two |
| splits | `split_at` | `Split`: one class, the parts' relations, unsolved, or no parts |
| | `unify_lit_boundaries`, `split_parts` | the relations recorded |
| nodes | `decide`, `decide_at_once` | `Decided`: the node, its decision, the relations recorded, and for `decide` `DecidedBy`: the nodes that depend on it, the dependencies it broke, the undecided nodes that broke theirs on it, and its `ValueRead` (the nodes and relations it leaves out, and its `Reading`: a use's `Narrowing`, what a class reads as for a lend, a projection or a binding, or a function value's Values) |
| | `record_decision`, `record_at_once` | the relations recorded |
| instances | `Instance::record_rest` | the relations it records |
| fallback | `take_defaults` | `FellBack`: each class, its default, what unification did, the refusal where it could not take it |

## The trace

What solving did along an order is written as facts, alike in a golden's written
order and under `TRACE=1`. A line is one fact, `keyword subject: value`, either part
left out where the fact has none, followed by its attributes, each `key: value` after
two spaces; the facts a fact is made of are the lines indented under it. An attribute
or a line not written is none: a union whose class holds no kind has no `kind:`. A
phase that did nothing writes no line. This table is where each keyword is defined. A
class is named as the state the step left names its equivalence class, except under
`union` and `conflict`, where each class is named by its own name and each side by
the least of its members' names before the union, so that a later union of the same
step does not rename what an earlier one shows.

| Line | Means |
|---|---|
| `start g` | the binding group `g`'s state as solving starts it |
| `step k g: <step>` | the `k`th step of the order, `solve X`, `fill X` or `decide n`, taken in `g`; under `TRACE=1`, `steps: <step>, ...`, the steps the state could take |
| `fallback k g` | the `k`th step of the order is the entry's fallback; with no `default` under it, no class falls back |
| `default X: T` | under `fallback`: `X` takes its default `T` |
| `refuse: <refusal>` | under `start`, `step` and `fallback`: a refusal the step made, as an outcome writes it |
| `record #k <relation>` | the relation recorded at `#k` in the log |
| `link X: E` | the view of `X`'s extern-defined container links its element `E` |
| `common` | the common part of all the nodes not yet decided may become |
| `merge cycles` | the cycles of lower bounds alone merged |
| `lit lowers X` | `X`'s LitVar lower bounds unified with it |
| `lit join #k <relation>` | the LitVar lower bound at `#k` unified with the class it bounds |
| `join X: <took>` | `X` solved, having taken `held` (its kind), a constructor and its members, `nothing`, `no join`, or `infinite` (it reaches a part of its own) |
| `fill X: <took>` | `X`, which nothing enters, filled: the one type the places it passes into have, or `differ` where they have two |
| `lit boundaries X` | `X` unified with each type it passes into |
| `split #k <relation>: <what>` | the relation at `#k`: `one class`, `parts`, `unsolved`, `no parts` |
| `decide n: <decision>` | the node `n` decided, its decision written as an expectation writes one |

Under `TRACE=1` alone:

| Line | Means |
|---|---|
| `fixpoint g` | `g` has no step left |
| `by: <recorder>` | of `record` and `relation`: what recorded it: `origin`, `text i`, `node n`, `split X`, `cycle`, `function type X ~> Y`, `check`, `instance m of <recorder>` (what recorded its original in the group `m` instantiates) |
| `instance m: C of g` | the instance `m` made, of the C of `g`; `by: <recorder>`, `records: #k, ...` |
| `union A + B: {M, ...}` | `A` and `B` unified into the class of these members |
| `by: #k`, `by: congruence <part> of A + B`, `by: kind X at <part>`, `by: fact X: <kind known in part>` | of `union` and `conflict`: the equation at `#k`; the parts `<part>` of the kinds the union of `A` and `B` conjoined; the parts `<part>` of the kind `X` held and the kind it was given; a kind known in part of `X` and what its kind, or another kind known in part, names there |
| `kind: <kind>` | of `union`: the kind of the class it makes |
| `conflict A + B: <conflict>` | `A` and `B` left apart, holding what no type holds; `by:`, `kinds: <kind> vs <kind>` |
| `conflict {M, ...}: <conflict>` | the class of these members holds what no type holds |
| `equations: #k, ...` | under `join`: the relations that give `X` its kinds |
| `lower L: <constructor>` | under `join`: a solved lower bound, and the constructor and members it gives |
| `part lower L: <constructor>` | under `join`: a solved part lower bound, and the constructor and members it gives |
| `fact: <constructor>` | under `join`: the constructor and members a kind known in part gives |
| `left out: L, ...` | under `join`: the lower bounds and part lower bounds not solved |
| `place P: <type>` | under `fill`: a place `X` passes into, and its value |
| `exclude: n, ...` | under `decide`: the nodes whose relations its read leaves out |
| `broke: n, ...` | under `decide`: the decided nodes it broke its dependency on |
| `broken by: n, ...` | under `decide`: the undecided nodes that broke theirs on it |
| `left out: #k, ...` | under `decide`: the relations its read leaves out |
| `read <at> X: <value>` | under `decide`: the value read at `X`, at `argN`, `result`, `place`, `operand`, `position` or `callee`, `_` where it decides nothing |
| `as: <reference>` | of the `read` of a lend, a projection or a binding: `&`, `&mut`, `reference` (recorded, its mutability not decided), `no reference`, `none` |
| `passed: <mutability>` | of the `decide` of a binding: the least of the references its path passes, `_` where undecided |
| `values: λ, ...` | of `read callee`: the lambdas its Values names |
| `candidates: f#0, ...` | under the `decide` of a use |
| `narrow <at> <value>: f#0 <rank>, ... -> f#0, ...` | under the `decide` of a use: each candidate left at that position with the rank of its coercion there (`empty`, a rank, `none` where no coercion has one), then those that remain |
| `retry` | under the `decide` of a use: the narrowing again, each LitVar taken as its family's default |
| `unsolved X` | `X` is not solved; `universal: yes` for a universal, and `no value: yes`, `enclosing: yes` (of an enclosing group) or `wait:` |
| `undecided n` | `n` is not decided; `wait:` |
| `wait: <dependency>, ...` | of `unsolved` and `undecided`: the unbroken dependencies not yet met: `lower L`, `taking T`, `part of O`, `records n`, `reads n`, `reads outside`, `path n`, `chosen`; `none` where none is |
| `C g` | what `g` leaves in C at its fixpoint: under it `classes: ...`, `nodes: ...` and each `relation #k <relation>` with its `by:` |
| `written order`, `first order` | the order traced |
| `limit: 10000` | the first order did not end within 10 000 steps |
| `end` | where the first order ends; under it its outcome, as a golden writes one |
| `explored: N states, complete`, `incomplete` | the exploration |
| `sampled: N orders` | the sampling |
| `witness k` | the first order reaching the `k`th outcome: under it the outcome, its `//@ order` lines, `round-trip: yes` or `round-trip: no: <where>`, the replay's trace, and `replay: outcome k`, `replay: another outcome` or `replay: fails: <why>` |

A golden writes a written order's trace without the lines of `TRACE=1` alone.
`assert <phase> records` finds a `record` under a line of its phase: directly
under `start` or `step` (`record`), under `link` (`unify`), `merge`, `lit lowers`,
`lit join`, `join` or `fill` (`join`), `lit boundaries` or `split` (`split`),
`common` or `decide` (`node`), `default` (`fallback`).

### `TRACE=1`

`TRACE`, read like `BLESS` and `CASE`, is unset or `1`, and any other value
panics. Under it each selected case's runs print their trace to stdout, every
line prefixed with the case's path and flushed as soon as it is known. A step's
line is printed before the step is taken, so that a step that does not end is
named by the last line printed, and the rest once it returns. What is printed is
read from the reports the phase functions return and from the state between two
steps; `golden()` and every expectation read what they read without it.

- A written order: its full trace, the steps the case writes and then the first
  step until none is left.
- `orders: all` and `orders: sampled N`: first one order traced, its first step
  at every state as sampled order 0 takes, stopped at 10 000 steps over the whole
  order, a fallback counted as a step as a written order writes it; then the
  exploration or the sampling; then each distinct outcome's witness, which
  `explore` hands with each final state (`Order`: each group's steps, the entry's
  steps to each fixpoint, a fallback that fell a class back between them),
  replayed from its `Step` values.
- Under `start` and `fixpoint`, each `unsolved` and `undecided`; under the
  `fixpoint` of a group other than the entry, its `C`. The entry's `fixpoint`
  comes before each of its `fallback`s.

## Run

- The 38 tests of `examples.rs` are 51 cases under `cases/spec`, `cases/holes`,
  `cases/environment`, `cases/istar` and `cases/certificate`, a test's programs and
  readings with other expectations each a case of its own. Over the 38, the distinct
  dump lines of every exploration (`Exploration` with its final states' fingerprints)
  and of every sampled order (fingerprint and outcome) are the same set from the
  cases as from the tests, and the same from the tests with the phases' reports as
  without them; `examples.rs` is removed. Each case asserts what its test asserted: a
  pattern the test matched (`Unreached(_)`, a refusal's kind) as the same pattern,
  and a text it matched in part (`starts_with`, `contains`) whole, as the model gives
  it.
- 9 cases under `cases/spec`, `block-*`, test a block solved as a lambda called where
  it lies,
  each derived before it was run. `rules/classes/a-a-capture-shared-by-two-instances`
  and `rules/solving/a-two-instances-meet-a-capture-written-by-name`, become one
  program with one derivation with
  `spec/a-capture-written-by-name-called-with-two-types`, are merged into it.
- 4 cases under `cases/certificate`, `a-lambda-*` but the two of a conversion
  below, test the checker carrying
  Values as the split does: a lambda in a payload into a function type,
  through a cast, through a view, and into a function type both candidates write
  whole. The first three panicked in the checker before, where the solver had
  accepted them; each was derived before it was run, in every order. The first
  also sets the part's instance's copy of x to `u16`, and the checker fails at
  the type's parameter `u8` into it and at its pass into λ's result, as derived.
  The second and the third give the class λ reaches through the cast or the view,
  one class with the use's `T`, `Fn/1{}`, and the checker fails there and at that
  `T`, as derived.
- Since a use of one candidate has no node, 86 goldens lost that use's decision line
  and nothing else, and `phases/unify-a-view-links-an-index` records `new`'s
  relations with the text; 5 cases dropped an assertion of such a decision.
- `cases/certificate/a-function-value-passed-through-a-use` tests the certificate's
  Values: a lambda reaching a call only through a use's instance, and the call given
  no lambda to call. Where Values meet what no rule decides yet, the checker panics.
  `istar/a-call-left-out-leaves-out-its-instance-relations` declared `choose` to give
  `Fn/1`, a function value no declaration makes, and gives back the one its `T` takes
  in.
- 2 cases under `cases/rules/classes` test a conflict, which leaves the classes it
  names with no value, and 2 under `cases/rules/place` a part solved before its
  parent, which then gave two and three outcomes and gives one now.
- 3 cases under `cases/spec` test a closure a function returns: one instance's, two
  instances' apart, and a call of both, refused; each was derived before it was run,
  over 200 sampled orders, every order of one passing 100 000 states.
- 2 cases under `cases/spec` test where an `Error` is written: the run's result into
  the type the embedding declares (`out T`), and a declaration's parameter, each
  derived before it was run; that the text, a context and a declaration's result
  write none is `src/program/tests.rs` and `src/sig/tests.rs`.
- 1 case under `cases/spec`, `a-try-in-a-match-arm`, tests that a match
  arm's braces are no block expression: its `?` exits the
  script, derived before it was run.
- 7 cases under `cases/spec` test the common part: an overload chosen by its result
  through `?`, with the script's result typed, with no tail, and refused where the
  tail's result is untyped and a fill has a place with no value; a counter compared
  with a typed value; a reference cast at one argument of an overload set; and two
  operands of one unknown meeting, accepted where their bindings join and refused
  where one is written. Each was derived before it was run, two derivations failing
  first and redone: the operands' bindings join by the cast, and the untyped tail
  leaves a fill a place with no value.
- 7 cases under `cases/spec` test the environment cut by constraints: a lambda
  writing its parameter into a capture by an overload that writes one closed type
  there, accepted; pushing it into a capture by `push<T>`, refused at the parameter;
  calling a function that ties the capture and the parameter to nothing together,
  accepted, the function's instance in two parts; passing a capture and its parameter
  apart, accepted; never called, passing a literal capture to a function, which types
  it `u8`; writing two types a cast joins, each call deciding its own candidate; and
  calling a function that returns a closure, which ties the capture to the result
  through the closure's environment. Each was derived before it was run, in every
  order. Four cases of the environment and the phase case of the common part changed
  with it, their values and decisions kept.
- 11 cases under `cases/spec` test the constructs the recording had no rule for: a
  branch that returns gives no value; a tuple, an array and an object literal, and an
  array of two types, refused; a cast by its operand's type and the operators that
  are no lang items, the language's candidates written as declarations; a `for` over
  a reference, a mutable reference, an array, a `Vec` by value and a range; and a
  match on an overload chosen by its result. Each was derived before it was run, in
  every order.
- 3 cases under `cases/spec` test a sum of two literals whose result is unused,
  narrowed again at `i32`; `1 ^ 2`, whose two literals are one class along some
  orders and apart along others, one typing and so one outcome since outcomes compare
  by their typing; and a literal binding into `u8` and a `u16` meeting, refused at
  the meeting in every order. Each was derived before it was run.
- 4 cases under `cases/spec` test two uses that fill one literal, a literal into a
  use then another, a sum into a use that only reads it, and a literal through a
  generic use then two uses. Each was derived before it was run; the two refused ones
  report I\* completeness at both uses, as before.
- 6 cases under `cases/spec`, `try-*`, and `block-collects-try` test `e?`: two errors
  of a name
  meet as one `Error`, an `Error` is not wrapped, an `Option`, two errors of no name
  meet as one `Error⟨..⟩`, one of no name and one of a name have no join, an error
  part nothing enters wraps into `!`, and a block collects its `?`. Each was derived
  before it was run. Those with two `?` are run over 200 sampled orders, every order
  passing 100 000 states.
- 3 cases under `cases/spec` test a function value passing into a function type:
  refused where the lambda's result reaches the type's by no coercion, accepted where
  it does, and by a declared conversion at that boundary.
- 2 cases under `cases/spec` test two functions that call each other, `even(10u8)`,
  with and without an `i32` candidate of each: two outcomes and five, by whether a
  broken dependency's relations had waited for a node that only they decide and a
  LitVar's default had been taken at once, now one, accepted, over 200 sampled
  orders. Each was derived before it was run. `two-uses-that-fill-one-literal` and
  `a-literal-through-a-generic-use-then-two-uses`, refused with I\* completeness
  violations, are accepted with `x: u16`, `n` decided by the `u16` that `m`'s
  boundary unifies `x` with;
  `a-lambda-calls-a-function-with-a-capture-and-its-parameter` keeps its outcome over
  200 sampled orders, every order now passing 100 000 states.
- 2 cases under `cases/spec` test a lend of a use's result where the candidates'
  results are type parameters alone, which the use does not read: accepted with the
  lend through the reference the result is, where it had taken the place before the
  use decided, and with its place where the result is no reference. Each was derived
  before it was run.
- 2 cases under `cases/spec` test a function value a declared conversion carries:
  refused where the Values where it reaches do not name its lambda, alone and where a
  call of the value the conversion made would make no instance of it, and 2 under
  `cases/certificate` accepted where they name it, at one place and at the two places
  of one variable. Each was derived before it was run; the solver had accepted the
  first two, and the checker had found each.
- 2 cases under `cases/spec` test an array lent into a slice parameter, shared and
  exclusive: accepted by the array's view, where no coercion had reached the
  parameter. 1 case tests a call through a reference to a function value: accepted,
  calling the lambda the target names, where the call had never been decided. Each
  was derived before it was run.
- 13 cases under `cases/phases` test each phase along a written order: recording, the
  join, splits, unification, nodes and fallback. 2 cases under `cases/primitives`
  test the join, `≤` and coercion on the declarations of `fixture::decls()` without
  `Box`, coercion with a conversion `u16 → u8` besides.
- Every case passes, and every golden is the model's.
