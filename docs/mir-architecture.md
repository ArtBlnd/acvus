# The MIR

`acvus-mir` is the compiler between the parser and a runtime: it infers
every type, records every effect, lowers to an SSA form, optimizes it, and
validates the result. A template's `{{ }}`, a script and a single `-e`
expression reach it as the same typed AST and leave it as the same
`MirModule`. `acvus-mir` names no runtime; `acvus-interpreter` and
`kovac-interpreter` depend on it, never the reverse.

The reader's order here is the pipeline's. What each pass does at the
instruction level, and what breaks when it is changed, is
[the compiler engineer's guide](mir-architecture-for-compiler-engineer.md).

## What the compiler holds to

**One IR, several surfaces.** Template, script and expression share the
type system, the pipeline and the IR.

**Explicit over implicit.** A context is named, not discovered; an effect
is in the type, not in a side table; a cast is a call, not an instruction.

**A type is inferred from use, not declared.** `@data | map(f) | collect`
says `@data` is iterable, `f` returns something, and the result is a
sequence of that something. The host supplies concrete types for contexts;
the compiler checks that they satisfy what the code imposed.

**Only a word copies.** The integers, `Float`, `Bool`, `Unit` and a
reference are words. Every other value moves: a binding used twice is a
type error, and duplication is the `clone` extern (RFC-0018). `&T` and
`&mut T` are types; a reference is never data, never returned, never
captured.

**A name, not an opaque id.** A context is the qualified name `@ns:name`,
so the IR is readable and comparable without a symbol table.

## The pipeline

```
source -> extract -> infer -> lower -> optimize -> MirModule
```

**Extract** (`graph/extract.rs`) parses and caches the AST of every local
function.

**Infer** (`graph/infer.rs`) runs constraint-based inference over the call
graph in Tarjan SCC order, leaves first, so mutually recursive functions
are solved together. Within an SCC the functions see each other's tentative
types; after it, every variable is frozen concrete. Results are cached per
SCC by `IncrementalGraph`, with early cutoff: an SCC whose types did not
change does not re-infer its callers. The LSP is a wrapper over that same
graph — diagnostics and build errors come from one pipeline, not two.
`IncrementalGraph` also lowers and optimizes, so the inputs it reports are
the ones the fold leaves and its diagnostics reach the validator; inference
stays per SCC, while optimization runs over the whole graph on each change.

**Lower** (`graph/lower.rs`, `lower.rs`) turns the typed AST into flat MIR.
A read of a variable is a `Take`, an assignment an `Assign`, `&place` a
`Ref`, `a[i]` an `AsSlice` plus an `Index` (RFC-0047). A context is a
variable of the body that names it (RFC-0025): `Fetch` at entry, `Commit`
at every return, and a `Commit`/`Fetch` pair around each call whose summary
(RFC-0025 rule 4) touches it. Every call whose effect is not Pure takes the
current `Order` and yields a new one. A `match` lowers to one
`Switch` (RFC-0051).

**Optimize** (`graph/optimize.rs`) runs three passes over each module.

- **Pass 0**, on the shape the source wrote: move check, borrow check, and
  `match` exhaustiveness (RFC-0029, RFC-0051).
- **Pass 1**, per body: `switch_expand`, `ssa_pass`, `string_copy`, `dse`,
  `dce`; then `inliner::inline` across modules, which needs every module
  because it resolves cross-function calls and devirtualizes closures.
  `switch_expand` rewrites each `Switch` into the `TestVariant` + `JumpIf`
  chain the machine runs today; it is deleted when the machine gains a
  `switch` operation (RFC-0051).
- **Pass 2**, per body: `commute`, `spawn_split`, `sroa`, `ssa_pass`,
  `string_copy`, `fold`, `dse`, `dce`, `code_motion`, `lsr`, `reorder`, a
  debug-only structural check, `drop_insertion`. Then `validate` per
  module.

Validation runs on the final IR rather than after lowering, so that it
covers what the optimizer produced as well as what the source wrote.

## Effects

A call's effect (`ty.rs`, `Effect`) has five parts.

**Reissue** (RFC-0013 rule 1) — `Pure < Idempotent < Opaque`. A Pure call stands
nowhere in the order of a run. An Idempotent call keeps its order and may
be issued twice. An Opaque call keeps its order and must not be.

**Task** (RFC-0046) — `Sync < Async < Heavy`. Whether the call suspends,
and whether it is offloaded to a blocking pool. Task and purity are
independent: a `heavy` pure extern is `Pure` with `task = Heavy`.

**Commutes** (RFC-0013) — whether two calls of the function are the same
program in either order. A Pure call that writes nothing commutes; a call
that writes a context never does.

**Reads and writes** (RFC-0025 rule 4) — the contexts the call may touch. A
function's summary is the union over its calls and its own accesses, closed
over recursion within its SCC. An ExternFn's own summary is empty: a script
reads a context and passes the value.

The join of two effects is the join on each axis. What the compiler does
not verify is the declaration itself: an ExternFn that declares `pure` and
does IO is a wrong library. The default for an undeclared one is Opaque.

### Order is a value

Ordering between effectful calls is a value of type `Order` that no script
can name (RFC-0007). A call whose effect is not Pure takes an `Order` and
yields one, so sequential code is a chain. `Merge` joins orders the way a
phi joins values — associative, commutative, a value instruction rather
than control flow — and an `anyorder { … }` block is a fan-out from the
block's entry order with one `Merge` at its exit. Because order is a data
dependency, a pass that respects use-def respects order.

Whether two IO calls may be reordered is the author's intent, declared with
`anyorder` or by a `commutative` callee; it is never inferred from types.

## ExternFns in the IR

An ExternFn is declared once, as a Rust function under `#[extern_fn]`; its
type and its handler both come from that signature (RFC-0023). In the SSA
it is a node like any other: arguments are uses, results are defs, and
there is no marshalling step between. The runtime contract (RFC-0039) is
`erase` / `materialize` by Rust type, `deref` / `deref_mut` / `reference`
for references, and `call_0/1/n` for closures.

Because of that, the standard passes reach it: `dce` removes a Pure call
that writes no context and whose result is unused; `spawn_split` splits an
effectful direct call into `Spawn` + `Eval`; the inliner devirtualizes a
closure whose callee traces to a single `MakeClosure`; a registered cast
rule lowers to an ordinary pure `FunctionCall` of the cast ExternFn, so
there is no cast instruction.

Registration is all-or-nothing and happens once, through `Externs::combine`
(RFC-0021), which rejects a name declared twice. A name two namespaces
declare is a candidate set, decided at the call (RFC-0043).

## Concurrency between calls

The workload the language is embedded in waits on IO. Four passes turn
independent IO into overlapping IO without the script saying so.

1. **`commute`** — a maximal run of neighbouring commutative calls on the
   `Order` chain is rewritten as an `anyorder` block would be: every call
   takes the run's entry order, and one `Merge` stands where the last
   call's order stood. A call moves across blocks only where its block
   post-dominates the block of the call it follows.
2. **`spawn_split`** — every direct call that `Effect::runs_apart`
   (RFC-0046) becomes `Spawn` (issues the call, takes the order before it)
   and `Eval` (forces the `Handle`, yields the order after it).
3. **`code_motion`** — pure instructions are hoisted to the highest
   dominator where their operands are available; `Eval` and whole-context
   `Take`/`Assign` are sunk toward their first use. A `Spawn` is never
   hoisted: issuing it on a path that would not have reached it speculates
   an effect (RFC-0007).
4. **`reorder`** — within each block, a topological sort by use-def with
   priorities: `Spawn` earliest, `Eval` just before the first use of its
   result, everything else in its original order.

Not done: speculative issue (IO inside a branch is issued in that branch)
and parallelizing pure computation (only calls that `runs_apart` are
split). The scheduler is greedy, not optimal.

## SSA

The SSA pass builds a Cranelift-style `SSABuilder`: blocks take parameters
and jumps pass arguments — `Jump { label: L0, args: [v1, v2] }` reaching
`BlockLabel { label: L0, params: [v3, v4] }` — rather than LLVM PHI nodes
at block entry. Block params are defs and jump args are uses, so inlining
remaps arguments instead of performing PHI surgery, and construction needs
no predecessor set.

A storage — a local or a parameter — is promoted only when every access to
it is a whole `Take` or `Assign`. One `Ref`, or one access by field path,
pins it in memory. A context's variable is a local like any other, except
that a write inside a branch is written back at the merge, because a
context is external state.

An aggregate that does not escape its body is replaced before SSA by one
register per field, or by a `(tag, payload)` pair (`sroa`, RFC-0050 rule 11), over
the same phi builder under a wider key.

## Inlining

The inliner devirtualizes as part of its pass: where an `Indirect` callee
traces back to exactly one `MakeClosure`, and not through a block
parameter, the closure body is inlined with captures prepended to the
arguments. A closure reaching the call site through a block parameter could
be one of several at run time, so it stays indirect. Recursive calls are
detected by SCC and not inlined.

## Registers

MIR values stay as the optimizer left them: one `ValueId` per definition.
A runtime that needs physical registers assigns them itself — the
interpreter's `prepare` does, at module load (RFC-0044) — so the compiler
holds no coloring pass and a checker never sees one value defined twice.

## Types

**Structural, not nominal.** `{ name: String }` is a type, and any value
with that field has it, whatever host it came from. Two objects join to the
union of their fields, two enums to the union of their variants (RFC-0042).

**Open enums.** Writing `Shape::Circle(r)` introduces the variant; no
declaration site exists. Exhaustiveness is therefore decided in `validate`,
and only where the variant set is known (RFC-0051).

**Identity is a type parameter** (RFC-0012). A user-defined type declares
at most one identity parameter; two identities unify only when they are the
same, so a value from one source never mixes with a value from another. An
identity variable that nothing tied to a source becomes a fresh source when
the solver closes (`Solver::solve`). No structural type carries an
identity, and the runtime sees none of it.

**Declared bounds, not traits.** A type variable of an ExternFn carries its
bound: `Any`, `OneOf(types)` (RFC-0011), or `Integer` for a literal's width
(RFC-0037). A shared signature (RFC-0019) lowers `T: sig` to `OneOf(the
types with an instance)`. The cost is the message: "this type is not one
of …" names shapes where a trait system would name a trait.

**Representation is a fact about a slot** (RFC-0041). `#τ` is the
specialized, native Rust representation of a type argument or a reference
target; everything else is uniform, the runtime's `Value`.

## Validation

`validate(&MirModule)` runs on the final IR.

- **Type check** (`validate/type_check.rs`) — every instruction's operands
  match `val_types`: arity, constructor shape, and the `Order` edge (an
  effectful call carries one, a Pure call carries none). `main`'s `Return`
  is held to `MirModule::ret`, the type the host declared (RFC-0054).
- **Move check** (`validate/move_check.rs`) — every value that is not a
  word is move-only; an option is a word exactly when its payload is, since
  it has no representation of its own (RFC-0039). A `Take` of a storage
  whose value was taken is a use after move; an `Assign` revives it; at a
  merge, `Alive` joined with `Moved` is `Moved`.
- **Borrow check** (`validate/borrow_check.rs`) and **exhaustiveness**
  (`validate/exhaustive.rs`) run in pass 0, on the shape the source wrote.
- **Definite assignment** (`validate/init_check.rs`) — which fields of each
  storage are initialized at each point, and whether a call's arguments
  carry every field the parameter type requires. It runs on the pre-SSA
  `CfgBody`.

Not checked: integer overflow, an out-of-range index that no interval
proves, and division by zero are run-time panics (RFC-0044); termination is
not decided; a plugin's handler is trusted to match its declaration.

## Invariants

1. **Types are complete.** No type variable survives inference: lowered MIR
   is `Ty = TyTerm<Concrete>`, whose `Var` is uninhabited.
2. **Moves are checked.** Only a word or a reference copies; every other
   value is consumed exactly once.
3. **Output is deterministic.** The same source produces the same MIR,
   whatever the hash order.
4. **One source of truth.** A value's type is in `val_types`; a context's
   identity is its qualified name.
5. **Order is a dependency.** An effectful call takes an `Order` and yields
   one, and nothing hoists a `Spawn` above a branch.
6. **Validation is final.** It runs after every optimization, so it covers
   the optimizer as well as the source.
