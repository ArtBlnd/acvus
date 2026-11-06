# Cases

A case is a file that states a program at the level the model takes it, equivalence
classes, the values passing into them, uses, lends, projections, bindings and binding
groups, and what solving it must give. One runner reads every case, solves it, checks
it, and reports. A case tests one of three scopes:

- a program, solved in every order or in sampled orders;
- a phase, along one order the case writes, by the trace of what each judgment and
  fallback did;
- a primitive on the case's declarations alone: the join, `≤`, and coercion.

Every expectation is checked against the model; the case language decides nothing. Tests
that need Rust values are `src/*/tests.rs`, over `src/fixture.rs`, outside this language.
The `//@` directives follow rustc's compiletest.

## Where a case lives

A case is `cases/**/*.case`, and its golden is the file of the same name with the
extension `.outcome` next to it. The runner reads every `.case` file under `cases/`, at
any depth, in the order of their paths; the directories group cases and mean nothing to
the runner.

The runner is the one test `cases::cases` (`src/cases/`, compiled under `cfg(test)`,
depending on the standard library alone): `parse.rs` reads a case, an error naming its
line; `build.rs` makes the declarations and the `Program`; `run.rs` explores, samples or
follows the written order and hands each final state to the certificate's checker, to
I\* and to the changed certificates; `expect.rs` compares; `render.rs` writes what the
model gives as a case writes it. Each case runs in its own thread, named
`cases::<path>`, over as many workers as `std::thread::available_parallelism` gives; a
case that panics fails with the panic's message. The test prints one line for each case,
`PASS <path>` or `FAIL <path>` with each failure under it, and fails where any case
fails.

## Form

`//` begins a comment, which runs to the end of its line. A line that begins, after
white space, with `//@` is a directive (Directives); the parser reads the directives
first, and reads them as comments after.

A name is a word of letters, digits, `_` and `?`, or such words joined by `::`
(`core::array_view`, `E::V`). Any other name is written in backquotes,
`` `fill_all(..)` ``, which hold any text but a backquote and a line break; a name in
backquotes is never a keyword. A string is `"…"` on one line, with no `"` inside it.

Declarations, classes, items and expectations are recorded in the order written
(`ProgramBuilder`). Class names and disjunctive constraint names are apart:
`` `f(x)` = (F)(X) @ `f(x)`; `` gives a class and a call one name.

### Types

A declaration, a type the text writes, and an expectation write a type so:

| Written | Type |
|---|---|
| `&T`, `&mut T` | a shared, an exclusive reference |
| `()`, `(T,)`, `(T, U)` | the unit, a tuple; a tuple of one component takes its comma |
| `[T; n]`, `[T]` | an array of length `n`, a slice |
| `!` | never |
| `{f: T, g: U}` | an object |
| `E{V(T), W(U)}` | an enum of these variants |
| `Fn/n` | a function value of `n` parameters |
| `Fn/n{λ, μ}` | a function value whose Values name `λ` and `μ`, `Fn/n{}` naming none: only in a value a certificate change plants |
| `fn(P, Q) -> R` | a function type |
| `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `bool`, `char` | the scalars |
| `String`, `str` | |
| `Option<T>`, `Result<T, E>`, `Error<T>` | |
| `Pair<A, B>` | a declared struct, one argument for each field, in declared order |
| `Vec<T>`, `F<T, I>` | a declared extern-defined type, its identity last where it has one |
| `#` | a signature's identity, in a declaration alone |

In a declaration, a name of its type parameters, written with no arguments, is that
variable. A name that no declaration and the language give is an error.

## Declarations

| Written | Declares |
|---|---|
| `extern type Vec<T>;`, `extern type F<T, #>;` | an extern-defined type, its arity, `#` its identity |
| `struct Pair { l, r }` | a module's struct, its fields in order; no `;` follows |
| `extern fn f<T>(&mut Vec<T>, T) -> R;` | a function; with no `->`, its result is `()` |
| `#[cast] extern fn up(A) -> B;` | a cast |
| `#[conversion] extern fn narrow(u16) -> u8;` | a declared conversion |

The types are declared before the functions, wherever they are written. A name declared
several times has its candidates in the order declared, `f#0`, `f#1` (`DeclId`). The
language declares `?` first: `?#0` is `?(Option<T>) -> T`, and `?#1` is
`?(Result<T, E>) -> T`. For each array length a function's signature, or a class's
written, host or outward type, writes, the language's views of that array,
`core::array_view` (`&[T; n] -> &[T]`) and `core::array_view_mut`, are casts. The casts
form a tree (`DeclsBuilder::finish`); a case whose casts do not is an error.

## Groups and classes

Text outside any group is the entry's, the group `run`. `group g { … }` holds the text of
the binding group `g`, and `group λ in h { … }` that of a group lying in `h`, as a lambda
lies in its enclosing body (`ProgramBuilder::group_in`). Writing a group again reopens
it, and the group it lies in is the one its first block names. A group holds statements
alone, and groups do not nest in the text: a group lying in another is written at the
top level, with `in`.

| Written | Gives |
|---|---|
| `let A;` | a type variable (`Origin::Var`) |
| `let 0: lit 0;`, `let x: lit 1.5;` | a literal's LitVar, of an integer or a float |
| `` let `Some(a)`: Option<S>; `` | a type the text writes |
| `` let `@cfg`: host {name: N}; `` | a type the host gives |
| `let D: out Result<U8, E>;` | the type the embedding declares for the run's result, which may hold an `Error` |
| `let F: lambda λ;` | the function value of the lambda whose function is `λ` |
| `fn g(A, B) -> R;` | the function of the group it is written in, by its parameters' and its result's classes |
| `universal A;` | `A` is a universal (`ProgramBuilder::universal`) |

A written, host or outward type is its outermost constructor, each part a class it
names: `Option<S>` names the class `S`. A literal is decimal digits, with a fraction
where it is a float, and has no sign and no suffix. Every class an item, a function, a
universal or a written type names is given by `let`, once; every lambda a `lambda`
names is a group's `fn`.

## Statements

| Written | Item |
|---|---|
| `S = A;` | the value of `A` passes into `S` (`Place::Slot`) |
| `#[by_name] S = A;`, `#[through_ref] S = A;` | the same, into a binding of an enclosing group reached by name (`Place::ByName`), or through a reference (`Place::ThroughRef`) |
| `A == B;` | an equation |
| `R = f(X, Y) @ n;` | a use of `f`, a declared function or a group's `fn` |
| `R = ?(X) into Q @ n;` | `e?` of the operand `X`, the value of its exit passing into `Q` |
| `R = (F)(X, Y) @ n;` | a call of the function value of `F` |
| `R = &P @ n;`, `R = &mut P @ n;` | a lend |
| `R = *P @ n;`, `R = P.x @ n;`, `R = P.0 @ n;`, `R = P[] @ n;`, `R = P as Some @ n;`, `R = P as E::V @ n;` | a projection: a Deref, a field, a component, an element, a variant's payload |
| `B = bind P via [p, q] @ n;` | a binding of `B` at the position `P`, along the projections `p`, `q` |

`let R = …;` and `let R: <origin> = …;` give the class `R`, then the item. Each use,
call, lend, projection and binding is written with its disjunctive constraint's or its
use's name after `@`; a use of a group's `fn` is written with one too. `into` follows
`?` alone, which takes one operand.

## Directives

A directive is `//@ key: value`, one to a line.

| Directive | Means |
|---|---|
| `//@ orders: all` | every order (`explore_with`), which must be explored within 100 000 states; where no directive is written |
| `//@ orders: sampled 200` | 200 sampled orders (`sample_with`) |
| `//@ order: join A, decide f, fallback` | one order: the entry's items in turn |
| `//@ order g: join A` | the items of the group `g` in turn |

`orders` and `order` exclude each other; `orders` is written once, and the order of each
group once. An item is `join X`, `decide n` or `fallback`, items separated by commas; a
name may be written in backquotes, and a name holding a comma cannot be written.

Along a written order, each group in `Program::group_order`, then the entry, takes its
written items and then its first judgment until none is left; the entry falls back
wherever none is left, until nothing falls back. Each item is a judgment the state
allows when it is taken: `join X` names an equivalence class by any member's name as the
arena names it (`A`, `A.Payload(Some)`), naming one root; `decide n` names a disjunctive
constraint. `fallback` is written in the entry alone, where no judgment is left. An
order of a group that does not exist is an error.

## Expectations

Expectations are written at the top level, among the declarations and the text, never
in a group. An expectation of the outcome holds of every outcome of the run, and an I\*
expectation of every final state.

| Written | Holds where | Example |
|---|---|---|
| `must pass;`, `must fail;` | the outcome is accepted, refused | `must fail;` |
| `assert X: T;`, `assert X: none;` | the text class `X` has the value `T`, or none | `assert A: Vec<u8>;` |
| `assert n == <decision>;` | the disjunctive constraint `n` has that decision | `assert lend == lend &mut place;` |
| `assert refused <kind> [X];` | a refusal is of that kind, of a class with the member `X` where one is written; the kinds `infinite`, `no_join`, `contexts`, `no_value`, `no_candidate`, `ambiguous` | `assert refused no_join A;` |
| `assert refused conflict [at "…"];` | a refusal is a conflict, where a text is written of a class with a member of that name or with an item of that text | `assert refused conflict at "1.5 = X";` |
| `assert refused undeclared;` | a refusal is `undeclared` | `assert refused undeclared;` |
| `assert failing [R, …];` | the relations of the failing checks are exactly these, in the outcome's order | `` assert failing [`A ~> take.Param(0)`]; `` |
| `assert check R == <coercion>;` | a check of the relation `R` has that coercion | `` assert check `X ~> take.Param(0)` == identity; `` |
| `assert istar <kind> n;` | an I\* violation of the kind `completeness` or `soundness` at `n`; each final state's violations are exactly those written, none where none is | `assert istar completeness probe;` |
| `assert outcomes N;` | the run has `N` outcomes | `assert outcomes 2;` |
| `assert partition {A, …};` | an equivalence class of the outcome holds exactly these text classes | `assert partition {A, B};` |
| `assert certificate [in g] set X: T fails ["…", …];` | in every final state, the certificate whose sheet of `g` gives the class `X` the value `T` fails, and the violations' `what`s over all final states are exactly these | `assert certificate set X: B fails ["A <= B fails"];` |
| `assert certificate [in g] set n == <decision> fails ["…", …];` | the same, the sheet giving the disjunctive constraint `n` that decision | `assert certificate set probe == probe#1 fails ["argument 0: A into B fails"];` |
| `assert trace "…";` | along a written order, a step's line, a report's line or a `refuse:` line of the trace is this | `assert trace "decide f: f#0";` |
| `assert <phase> records R;`, `assert no <phase> records R;` | along a written order, a `record` of `R` lies under a report of the phase, or none does | `` assert disjunction records `x ~> f.Param(0)`; `` |
| `assert join(A, B) == T;`, `assert join(A, B) == none;` | the join of the two types, on the declarations | `assert join(u8, u16) == none;` |
| `assert le(A, B);`, `assert not le(A, B);` | `A ≤ B` holds, or does not | `assert not le(u8, u16);` |
| `assert coerce(A, B) == <coercion>;` | `A ⇝ B` by that coercion, on the declarations | `assert coerce(&mut A, &A) == reborrow;` |
| `assert along_le(A, B) == <coercion>;` | `A` changes so along `A ≤ B` | `assert along_le(A, B) == cast cast0#0;` |

The phases of `records` are those of the reports: `record`, what a group's start recorded
from the text; `unify`, `merge cycles`; `decomposition`; `join`; `disjunction`,
`decide`; `choice`, `narrow`; `fallback`, `default` (The trace).

A decision, a coercion, a trace line and the relation of `records` are compared with
every white-space character left out; a class's name, the relations of `failing` and
`check`, a partition's names and I\* violations are compared exactly. The type in
`assert X: T` and in the primitives is a closed type, written whole. A class or a
disjunctive constraint named like a keyword of these forms is written in backquotes.

**Decisions** are written as the model writes them (`render::decision`): `lend & place`,
`lend &mut place`, `lend & through`, `lend &mut through`; `applied`, `applied through`;
`read`; `bind lend & place`, `bind lend &mut through` and the others of that form;
`calls {λ, μ}`; what a use became, read off its choice (`render::chosen`), `f#0` or
`never`; and, in a decision expectation, `undecided`, which a use is where it became
nothing and a disjunctive constraint is where it took no alternative, and `refused`,
which a disjunctive constraint is where it took none and a class it reads has no value
(`render::decision_at_end`). A certificate change takes each of these but `undecided`
and `refused`.

**Coercions** are written as the model writes them (`render::coercion`): `identity`,
`undecided`, `never`, `conversion f#0`, or the steps of a composition joined by `+`, of
`reborrow`, `cast f#0` and `shape`; where none reaches, `none`; where more than one
composition reaches, `compositions [<coercion>, …]`; where no composition reaches and
more than one conversion does, `conversions [f#0, …]`.

**A certificate change** names the class as the outcome writes it, so that a class an
instance or a disjunctive constraint made is named as a text class is; `g` is the entry
where no `in g` is written. A value it plants writes each function value's Values,
`Fn/1{λ}`, `Fn/1{}`.

**Outcome blocks.** `outcome { … }` holds what one outcome and the final states reaching
it satisfy: `must`, a value, a decision, `refused`, `failing`, `check`, `partition` and
`istar`. The blocks are satisfied by distinct outcomes. Where a case writes blocks and
no `assert outcomes`, its run has exactly as many outcomes as blocks; where it writes
neither, one. What lies outside the blocks holds of every outcome, and its I\*
expectations join each block's.

**What else fails a case:** a parse or build error, naming its line; an exploration past
the budget; a final state whose certificate, unchanged, has a violation; a panic; and a
golden that differs or is missing.

### What the model writes

What the model writes is ASCII, the names a case gives kept as it writes them, and in
backquotes where a case must write them so (`render::name`).

- A class: its text name; a part, `A.Payload(Some)`, `A.Param(0)`, `A.Return`,
  `A.Arg(0)`, by its part's key; a class a disjunctive constraint made, `f.Param(0)`,
  `f.Result`, `f.Var(Type(0))`, `f.Lent`, `f.Target`, `f.Exit`, `f.Generalized(0)`,
  `f.GeneralizedParam(0)`, `f.GeneralizedResult`, `f.Chosen(0)`; an instance's copy,
  `m/A`, the instance's name and the class it copies; a class of a cast's or a view's
  signature linking two classes, `link(A -> B)[0].Param`, `.Result`, `.Var(Type(0))`.
- An instance: `f@i`, of a use of the function `f` at the program's item `i`; `n#λ`, of
  the call `n` of the lambda `λ`. The copy of a disjunctive constraint `n` in the
  instance `m` is `m/n`.
- A type: as a case writes it, but a function type, `Fn(P, Q -> R)`.
- A relation: `A <= B`, a lower bound; `A <=p B`, a part lower bound; `A ~> B`, a
  coercion constraint; `A = B`, an equation; `A: <structure>`; `A: <structure
  constraint>`; `A: <domain>`; `A: Values contains λ`; `wrap A into B`, `e?`'s error
  wrapped; `A: Text`, `A: Universal`, `A: Shared { unknown: true }`, its standing.
- A structure constraint: `ref T`, `field x: T`, `component i: T`, `index T`,
  `variant V: T`, `variant E::V: T`.
- A domain: `{u8, i32}`, the constructors it admits; `not {&, &mut}`, those it does not.
- What no type holds: `<constructor> vs <constructor>`, `<domain> vs <constructor>`,
  `<domain> vs another domain`, `<constructor> vs <structure constraint>`, a constructor
  in its `Debug` form (`Scalar(Int(U8))`, `Option`).
- A refusal: `<cause> {<members>} by {<items>}`, the cause `conflict: <what no type
  holds>`, `no_join`, `contexts`, `infinite`, `undeclared: <error>`, `no_value`,
  `no_candidate` or `ambiguous`, and each item on the class a relation, `disjunctive
  constraint n`, `instance m`, `lambda n` or `choice n`; a use to which several
  candidates remain at the end is refused `ambiguous`, its members those of the
  variables of its generalization, its items `choice n` and those variables.
- An item of the record: a relation, `disjunctive constraint n`, `instance m: C of g`,
  `lambda m of λ`, or `choice n`, the use `n` with its choice of a candidate.
- A use's generalized domain: `X: choice f[u8, _]`, the use whose choice it reads and
  the constructor each candidate writes there, `_` where it writes a variable of its own.

## The trace

The trace of an order is read from the reports (`Report`, `src/trace.rs`) that
`State::start`, `State::step` and `State::fall_back` return. Each line names a class as
the state the step left names its equivalence class, except under `union` and
`conflict`. A golden writes a written order's trace; `TRACE=1` prints, besides, the
lines of the last table.

**The steps.**

| Line | Means |
|---|---|
| `start g` | the binding group `g`'s state as solving starts it |
| `step k g: join X`, `step k g: decide n` | the `k`th item of the order, a judgment taken in `g`; `k` counts the items of the whole order, fallbacks among them |
| `fallback k run` | the `k`th item is the entry's fallback |

**The reports**, one line under the step for each:

| Line | Report |
|---|---|
| `merge cycles` | cycles of lower bounds and part lower bounds alone merged into one class each (`Report::Merged`) |
| `decomposition #k <item>: one class`, `parts`, `no parts` | the item at `#k` decomposed (`Report::Decomposed`): a lower bound, a part lower bound or a coercion constraint whose two sides are one class; the relations recorded between the parts of its solved sides, or by a wrap or a structure constraint once its class is solved; nothing left to relate. An item not yet decomposable is not reported |
| `join X: <constructor>` | `X` judged (`Report::Joined`, `Took`): the constructor its lower bounds join at; `contextual <constructor>`, the one constructor of what it passes into, where nothing enters it; `bounded <domain>`, the domain it takes where its sources give no constructor; `no join`; `contexts`; `infinite`, it reaches a part of its own |
| `decide n: <decision>` | the disjunctive constraint `n` decided (`Report::Decided`) |
| `narrow n: <became>` | the use `n`'s choice narrowed (`Report::Narrowed`): what the use became, `f#0` or `never`, or the candidates that remain, `{f#0, f#1}`, `{}` where none |
| `default X: <structure>` | fallback gave `X` that default, its parts by their classes' names (`Report::FellBack`); `fallback X: conflict`, `fallback X: no_value`, the refusal it gave `X` |

A `join` line writes the constructor in its `Debug` form: `join A: Scalar(Int(U8))`,
`join V: Extern(Vec)`, `join r: Ref(Shared)`, `join A: contextual Option`.

Under a report's line, and directly under the step for what a group's start recorded
from the text (`Report::Recorded`), which has no line of its own:

| Line | Means |
|---|---|
| `record #k <item>` | the item recorded at `#k` in the record |

After the step's reports:

| Line | Means |
|---|---|
| `refuse: <refusal>` | each refusal the step made; under `start`, each refusal of the state started |

**Under `TRACE=1` alone.** Every line printed is prefixed by the case's path, `<path>: `,
and indented by two spaces for each level.

| Line | Means |
|---|---|
| `run: every order`, `run: N sampled orders`, `run: the written order` | the run |
| `written order`, `first order` | the order traced: the written one; for `orders: all` and `orders: sampled N`, first the order sampled order 0 takes, the first judgment at every state |
| `step k g: <judgment>  judgments: <judgment>, …` | the step's line, with the judgments the state allowed, printed before the step is taken |
| `instance m: C of g  by: <recorder>  records: #a-#b, #c` | under a report: an instance made, what made it, and the places of what it recorded, a run of consecutive places written by its ends |
| `record #k <item>  by: <recorder>` | a record line, with its recorder |
| `union A + B: {M, …}  by: <cause>  structure: <structure>` | under a `record`: `A` and `B` unified into the class of these members, why, and the structure of the class made, where it holds one |
| `conflict A + B: <what no type holds>  by: <cause>  structures: <s> vs <s>` | under a `record`: `A` and `B` left apart, with their structures where either holds one |
| `conflict {M, …}: <what no type holds>` | under a `record`: the class of these members holds what no type holds |
| `unsolved X  universal: yes  no value: yes  enclosing: yes` | under `start` and `fixpoint`: a class with no structure, each attribute written where it holds: a universal; a refused class; a class of an enclosing group, written only where `no value` is not |
| `undecided n` | under `start` and `fixpoint`: a disjunctive constraint not decided |
| `fixpoint g` | `g` has no judgment left; the entry's comes before each of its fallbacks |
| `C g` | under the `fixpoint` of a group other than the entry: what it leaves in C, `classes: …`, `disjunctive constraints: …` and each `relation #k <item>  by: <recorder>` under it |
| `limit: 10000` | the first order took 10 000 items and was stopped |
| `end` | the first order ended; under it, its outcome as a golden writes one |
| `explored: N states, complete`, `explored: N states, incomplete` | the exploration |
| `sampled: N orders` | the sampling |
| `witness k` | the first order that reached the `k`th outcome; under it the outcome, its `//@ order` lines, `round-trip: yes` or `round-trip: no: <where>`, the trace of its replay from the model's own judgments, and `replay: outcome k`, `replay: another outcome` or `replay: fails: <why>` |

A recorder, the `by:` of a `record`, an `instance` or a `relation`: `origin`; `text i`,
the program's item `i`; `disjunctive constraint n`; `generalization n`; `join X`;
`decomposition X`; `cycle`; `function type X ~> Y`; `check`; `fallback`;
`instance m of <recorder>`, what recorded the original in the group the instance `m`
copies.

A cause, the `by:` of a `union` or a `conflict`: `#k`, the relation at `#k`;
`congruence <part> of A + B`, the parts `<part>` of the two structures the union of `A`
and `B` conjoined; `structure X at <part>`, the parts `<part>` of the structure `X` held
and of the structure it was given; `structure constraint X: <structure constraint>`, a
structure constraint of `X` and what its structure, or another structure constraint of
it, names there. Under `union` and `conflict`, each class is named by its own name and
each side by the least of its members' names before the union, so that a later union of
the same step does not rename what an earlier one shows.

Each line is flushed as soon as it is known, and a step's line is printed before the
step is taken, so that a step that does not end is named by the last line printed. What
`TRACE=1` prints is read from the reports and from the state between two steps; the
golden and every expectation read the same reports without it.

## Environment variables

| Variable | Values | Means |
|---|---|---|
| `CASE` | unset, or any text | the cases whose path under `cases/` contains the text; where none does, a panic |
| `BLESS` | unset, or `1` | `1`: each case run writes its golden anew instead of comparing it |
| `TRACE` | unset, or `1` | `1`: each case run prints its trace to stdout |

Any other value of `BLESS` or `TRACE`, and a value of any of the three that is not
Unicode, is a panic.

## Goldens

`<case>.outcome` holds what the model gives for the case, written by the runner:

```
// What the model gives, written by `BLESS=1`.
run: every order                          or `N sampled orders`, `the written order`
outcome 1
  accepted                                or `refused`
  value X: <type>                         each text class, by name; `none` where it has none
  decision n: <decision>                  each disjunctive constraint, by name; `undecided`,
                                          or `refused` where a class it reads has no value
  partition {A, B}                        each equivalence class of two text classes or more
  check <relation>: <coercion>            each check, in the outcome's order
  refusal <refusal>                       each refusal
  I* of a final state: [<kind> n, …]      each distinct set of I* violations among the
                                          final states reaching the outcome
outcome 2
…
<step line>                               a written order's trace
  <report line>
    record #k <item>
  record #k <item>                        under a report with no line
  refuse: <refusal>
```

A check's coercion is the Rust `Debug` form of its result (`Ok(Identity)`,
`Ok(Composition { … })`, `Err(None)`); a check whose declared conversion carries a
function value bringing lambdas the Values where it reaches do not name writes them
after it, `  unnamed: {μ}`. The trace's `TRACE=1`-only lines are not written.

Without `BLESS`, the golden is compared character by character with the text the run
gives; where they differ, the case fails, naming the first differing line, and a case
with no golden fails. Under `BLESS=1`, the golden is written whatever the expectations
give, and the expectations alone decide whether the case passes; a case that does not
parse, build or run writes none.
