# The MIR — the compiler engineer's guide

For changing `acvus-mir`. Part I is the map: the crates, the data
structures and the pipeline. Part II is what each invariant carries, and
what breaks when it is changed. The reader's overview — what the compiler
is for, and which RFC settles which rule — is
[mir-architecture.md](mir-architecture.md); this file does not repeat it.

---

# Part I — The territory

## Crates

```
acvus-utils          Interner, Astr, Freeze, LocalId, QualifiedRef
  |
acvus-ast            parser: template / script / expression -> AST
  |
acvus-mir            type system, IR, analysis, optimization, validation
  |
  +-- acvus-extern         #[extern_fn], Registry, Externs::combine, Runtime
  |     |
  |     +-- acvus-ext            the standard library of ExternFns
  |     +-- acvus-ext-net        HTTP
  |     +-- acvus-interpreter    the register machine
  |           |
  |           +-- pomollu-core         TOML specs into the same graph
  |
  +-- kovac-interpreter    a second runtime over the same MIR
  |
acvus-lsp            language server, over the same pipeline
```

`acvus-mir` names no runtime. Everything that runs the IR depends on it,
never the reverse.

## The two IR forms

```
MirBody                          CfgBody
+-- insts: Vec<Inst>             +-- blocks: Vec<Block>
+-- val_types: Map<ValueId,Ty>       +-- label: Label
+-- params: Vec<(Astr,ValueId)>      +-- params: Vec<ValueId>
+-- captures: Vec<(Astr,ValueId)>    +-- insts: Vec<Inst>   (no control flow)
+-- order_param: Option<ValueId>     +-- terminator: Terminator
+-- task: Task
+-- debug: DebugInfo             +-- label_to_block: Map<Label,BlockIdx>
+-- val_factory: LocalFactory    +-- val_types, params, captures,
+-- label_count: u32                 order_param, task, debug (shared)

MirModule                        Terminator
+-- main: MirBody                +-- Jump { label, args }
+-- closures: Map<Label,MirBody> +-- JumpIf { cond, then_*, else_* }
+-- ret: Ty                      +-- Switch { tag, arms, default }
                                 +-- Return { value, order }
                                 +-- Diverge
                                 +-- Fallthrough
```

`MirBody` is the flat form — a linear stream including the control-flow
instructions. It is what lowering produces and what the interpreter
consumes. `CfgBody` is the structured form, which every analysis and
optimization pass takes. The lifecycle is `promote(MirBody) -> CfgBody`,
run the passes, `demote(CfgBody) -> MirBody` (`cfg.rs`).

`order_param` is the `Order` a body takes first when its effect is not Pure
(RFC-0007); `Return { order }` is the one it yields last. `task` is the
join of the tasks of everything the body does (RFC-0046); the interpreter's
`Code::may_suspend` is `task > Sync`. `MirModule::ret` is the declared
return type of the graph `Function` the module is the body of — for the
entry, what the host declared (RFC-0054).

## `InstKind`

Forty-three variants (`ir.rs`).

| Category | Instructions | Notes |
|---|---|---|
| Constants | `Const`, `Undef`, `Poison` | `Undef` is the SSA placeholder (valid to move, UB to read); `Poison` marks a type error the checker already reported |
| Storage | `Ref`, `Take`, `Assign`, `Fetch`, `Commit` | `RefTarget` is `Var(slot)`, `Param(slot)` or `Through(reference)`; `path` names a field chain, empty being the storage itself |
| Slices | `AsSlice`, `Index`, `IndexSet` | RFC-0047; `Index`'s index is a `u64` and its one check is `index < len` |
| Scalar field | `FieldGet`, `FieldSet` | on a value, not a storage; `FieldSet` produces a new value |
| Arithmetic | `BinOp`, `UnaryOp` | words only |
| String | `StringConcat`, `StringEq`, `StringClone` | the language-owned string operators (RFC-0020, RFC-0018 rule 2); a template body is one `StringConcat` |
| Functions | `LoadFunction`, `FunctionCall` | `FunctionCall { callee: Direct \| Indirect, callee_ty, args, order: Option<OrderEdge> }` |
| Async | `Spawn`, `Eval`, `Merge` | `Spawn` takes the order before, `Eval` yields the order after, `Merge` joins orders |
| Construction | `MakeArray`, `MakeObject`, `MakeTuple`, `MakeVariant`, `MakeClosure` | all pure |
| Access | `TupleIndex`, `ArrayIndex`, `ObjectGet`, `UnwrapVariant` | `ArrayIndex` is a constant position in an owned scrutinee — pattern destructuring, not indexing a container |
| Tests | `TestLiteral`, `TestObjectKey`, `TestVariant` | `Bool`, read by `JumpIf` |
| Control flow | `BlockLabel`, `Jump`, `JumpIf`, `Switch`, `Return`, `Diverge` | become blocks and terminators in `CfgBody` |
| Resources | `Drop` | placed by `drop_insertion` at a value's last use; consumes, never defines |
| Utility | `Nop` | what an instruction removal leaves |

The five storage instructions (RFC-0018):

- `Ref { dst, target, path, mutability }` — `dst` is a `&T` or `&mut T`
  naming the value at `path` under `target`.
- `Take { dst, target, path }` — move the value out of a storage into
  `dst`; a word is copied and the storage keeps it. A read of a variable.
- `Assign { target, path, value }` — move `value` into a storage; the old
  value is dropped. An assignment to a variable.
- `Fetch { dst, context }` / `Commit { context, value }` — a context's whole
  value out of and into the run's page (RFC-0025).
- `RefTarget::Through(r)` — the storage a reference names: `*r` is
  `Take { Through(r), [] }`, `r.f` is `Take { Through(r), [f] }`, `&r.f` is
  `Ref { Through(r), [f] }`, and `*r = v` is an `Assign` through `r`, which
  must be a `&mut`.

A context is a variable of the body that names it (RFC-0025): the lowering
fetches every named context into a slot at entry, commits each slot at
every return, and brackets each call whose summary (RFC-0025 rule 5, joined with
the summaries of its function-typed arguments) touches the context with a
`Commit` before and a `Fetch` after (`lower.rs`). `@x`, `&@x` and `@x = v`
are then the variable rules on that slot, and a context left moved out at a
return is a use-after-move at the exit `Take`.

## Types

```
Ty = TyTerm<Concrete>;  PolyTy = TyTerm<Poly>;  InferTy = TyTerm<Infer>

TyTerm::Int(IntTy) | Float | String | Bool | Unit
   | Never                          -- below every type (RFC-0038)
   | Order                          -- IR-only (RFC-0007)
   | Array(Box<Ty>, LenTerm)        -- length is a term: known or a variable
   | Object(Map<Astr, Ty>)          -- structural record
   | Tuple(Vec<Ty>)
   | Option(Box<Ty>) | Result(Box<Ty>, Box<Ty>)
   | Fn { params: Vec<ParamTerm>, ret, captures, effect: EffectTerm }
   | UserDefined { id, type_args: Vec<TypeArg>, effect_args, identity_args }
   | Enum { name, variants: Map<Astr, Option<Box<Ty>>> }
   | Slice(Box<Ty>)                 -- unsized; only under a Ref (RFC-0047)
   | Handle(Box<Ty>)                -- from Spawn
   | Ref(Mutability, Box<TypeArg>)  -- &T / &mut T (RFC-0018); a word
   | Error(ErrorToken)
   | Var(Phase::TyVar)              -- uninhabited when Concrete
```

The phase parameter decides which variables exist: `Concrete` has none,
`Poly` has a declaration's positional placeholders, `Infer` has the
solver's. A `TypeArg { repr, ty }` carries the slot's representation
(RFC-0041): `Repr::{Uniform, Specialized, Var}`. A parameter that borrows
has a reference type — there is no mode beside the type.

```
Effect {
    reissue:  Reissue,   -- Pure < Idempotent < Opaque (RFC-0013 rule 1)
    task:     Task,      -- Sync < Async < Heavy      (RFC-0046)
    commutes: bool,      -- same program in either order (RFC-0013)
    reads:    Contexts,  -- what the call may read  (RFC-0025 rule 4)
    writes:   Contexts,  -- what the call may write (RFC-0025 rule 4)
}

EffectTerm<V> = Known(Effect) | Var(V::EffectVar)
```

Every constructor keeps two invariants: a Pure call that writes nothing
commutes, and a call that writes a context never commutes. The join is the
join on each axis. `Order` is how the effect reaches the IR: a call whose
effect is not Pure carries `OrderEdge { before, after }` and a Pure call
carries none — `validate` rejects the other two combinations.

A type variable carries its bound with it (`TyVarBound`): `Any`,
`OneOf(Vec<PolyTy>)` (RFC-0011 rule 2), or `Integer { signed, among }`
for a literal's width (RFC-0037). Identity is a parameter of a user-defined
type, at most one per declaration (RFC-0012); two identities unify only
when they are the same.

## The solver

One structure answers every question the checker asks after a join
(RFC-0042).

- **`Terms`** is the union-find over type, effect, length, identity and
  representation variables, and the join on it:
  `join(a, b, Position, JoinKind, registry)`. `Position` is `Value`, where
  `Never` is the bottom, or `Argument`, where every type constructor's
  argument is invariant. `JoinKind` is `Flow`, `Pattern` or `Decision`, and
  it decides what a join may bind: a pattern may name fewer members than
  its source, a signature's representation variable is bound only by a
  decision's join.
- **A `Decision`** is a position with more than one admissible answer, and
  it only shrinks: the integer width, the effect interval, the instance
  (RFC-0040), the representation, a conversion (RFC-0041), a signature
  (RFC-0043). It settles when one answer remains.
- **`settle`** is one fixpoint, idempotent, run per body. **`solve`** closes
  what remains by the least element — `Never`, the lower effect, `Uniform`,
  the generic instance, identity, `i64` for a literal — and only then
  renders messages as `Unsettled`. An identity variable nothing bound
  becomes a fresh source there (`solver.rs`, `Solver::solve`).
- **Freezing** has three forms: `freeze_ty` refuses a variable a bound left
  open and is what a resolution carries into lowering; `close_ty` is the
  resolution's; `written_ty` closes an unresolved variable to `Never` and is
  what a report shows (RFC-0043).

There is no polarity, no `lub`, no `coerce` and no snapshot/rollback pair
outside a decision's own step: a conversion is a decision answered from the
registry once both sides resolve, and the coercion is an ordinary pure
`FunctionCall` of the cast ExternFn — there is no cast instruction.

## The compilation graph

```
CompilationGraph {
    functions: Freeze<Vec<Function>>,   -- local + extern
    contexts:  Freeze<Vec<Context>>,    -- @user, @items, ...
}

Function { qref: QualifiedRef, kind: FnKind, ty: PolyTy }
FnKind::Local(ParsedAst, Inputs) | FnKind::Extern { bounds: Vec<TyVarBound> }
Inputs::Declared | Inputs::FromReads
Context { qref: QualifiedRef, ty: PolyTy }
```

The entry is an ordinary declared function: its `PolyTy::Fn { ret }` is
what the host declared `main` returns (RFC-0054), lifted through the same
`lift_declaration` the hosts call for context types. Its `PolyTy::Fn { params }`
are the `$` inputs the host declared, and `Inputs::Declared` refuses a `$`
outside them and the bindings (RFC-0054 rule 6). `Inputs::FromReads` makes each
further `$` the body reads one more parameter, for an analysis that reports
what a body requires.

## The pipeline

```
[Parse]              acvus_ast::parse / parse_script
[Graph]              Externs::combine + local functions + contexts
[0 Extract]          graph/extract.rs   AST per local function
[1 Infer]            graph/infer.rs     SCC-ordered inference
[2 Lower]            graph/lower.rs -> lower.rs
[3 Optimize]         graph/optimize.rs
[Runtime]            acvus-interpreter
```

`graph/optimize.rs` runs three passes.

**Pass 0**, per module, on the shape the source wrote:
`move_check::check_moves`, `borrow_check::check_borrows`,
`exhaustive::check_exhaustive`.

**Pass 1**, per body, then cross-module:

```
promote -> switch_expand -> ssa_pass -> string_copy -> dse -> dce -> demote
then inliner::inline over every module
```

**Pass 2**, per body, then `validate` per module:

```
promote
  commute        commutative runs share one Order
  spawn_split    a call that runs_apart -> Spawn + Eval
  sroa           a non-escaping aggregate becomes registers (RFC-0050 rule 11)
  ssa_pass       promote whole Take/Assign
  string_copy    a String read before its last use is copied (RFC-0018 rule 2)
  fold           a constant expression folds (RFC-0055)
  dse            dead context Commits
  dce            dead pure instructions, dead handles
  code_motion    move between control-equivalent blocks; sink Eval
  lsr            a loop multiplies once (RFC-0056)
  reorder        Spawn early, Eval late, within a block
  (debug)        use-def, dominance and type coverage
  drop_insertion Drop at the last use of a move-only value
demote
```

Each pass's position is a dependency, not a preference: `sroa` runs before
`ssa_pass`, whose builder places the phis its parts need, and before `dce`,
which sweeps the constructor left with no reader; `fold` runs after
`ssa_pass`, which brings a constant and its reader into one body, and
before `dse`/`dce`; `lsr` runs after the hoist, which leaves the preheader
a block of its own, and before `reorder`, which schedules within a block.

## Pass detail

Each pass takes `&mut CfgBody`. A call's effect is read from its own
`callee_ty`; no function-type table is threaded through.

**`commute`** — a run is a maximal sequence of neighbouring calls whose
effect commutes on the `Order` chain, read within one block after SSA (a
phi is not a neighbour, so a run never crosses a branch). Every call in the
run takes the run's entry `Order`, and one `Merge` stands where the last
call's order stood. Before runs are read, a commutative call whose block
post-dominates the block of the call it follows is moved there. Nothing
else reads the commutes axis.

**`spawn_split`** — every `Callee::Direct` call whose effect `runs_apart`
(the task is above `Sync`, or the effect is not pure) becomes:

```
BEFORE:  r1 = call fetch(r0)  order(o0 -> o1)
AFTER:   h  = spawn fetch(r0) order(o0)
         r1 = eval h          order(o1)
```

The handle type `Ty::Handle(ret)` comes from the instruction's own
`callee_ty`. Indirect calls pass through unchanged.

**`ssa_pass`** — a storage is promotable only when every access to it is a
whole `Take` or `Assign`; one `Ref`, or one access by field path, pins it.
A context's slot is a local like any other: `Fetch` defines its `dst` and
`Commit` uses its `value`, and neither is promoted, removed, duplicated or
moved. Entry definitions are spliced at the start of block 0 after phi
insertion. Trivial phis are eliminated, and the two substitution maps are
chained — the builder's, then the forwarding one.

**`sroa`** (RFC-0050 rule 11) — a storage slot of object or enum type that
`analysis::escape` says nothing outside the body reaches, and whose every
use one arm of the pass's table covers, is replaced by one register per
field or by a `(tag, payload)` pair. The phis are `SSABuilder`'s, under a
key widened from `SsaVar::Local` to `SsaVar::Part`. One escaping use
anywhere sinks the whole replacement for that slot: there is no partial
escape. An enum keeps its aggregate where a variant carries nothing, or
where two variants disagree on the payload type.

**`fold`** (RFC-0055) — a binary operation on two constants becomes the
constant it would have computed at the operand's width. Where the machine
would panic instead of producing a value — a zero divisor, a quotient that
leaves the width — nothing folds. Two constants one associative and
commutative operator apart join when the intermediate has one use. The pass
introduces no operator the body did not already hold, because
`prepare::arith_of` claims five operators and an operator outside them
splits the chain that reads the site.

**`dse`** — backward context liveness per block. A read is a `Fetch`, any
call, or a `Return`; a write is a `Commit`. A `Commit` overwritten on every
path before a read is removed.

**`dce`** — mark-sweep. Roots are the instructions with side effects:
`Fetch`, `Commit`, `Assign`, `Take` (a take empties its storage, which is
observable), `Eval`, and a `FunctionCall` whose effect is not Pure-and-
writes-nothing. `Spawn` is not a root — a handle no `Eval` consumes is dead
work. Terminator operands are roots.

**`code_motion`** — an instruction moves only between **control-equivalent**
blocks: the destination dominates the source and the source post-dominates
the destination, so the two execute under exactly the same condition. Every
target is also held to the source's loop depth (`analysis::loops`), because
post-dominance alone would let an instruction written after a loop land in
its header and run once per iteration. The one exception is a **shared
borrow** — `Ref` of a `Var`/`Param` under no path, and a shared `AsSlice`,
which is the same borrow one level down (RFC-0047 rule 3) — which moves under a
borrow condition instead, and is what lifts a slice out of a loop that does
not write its container. `Spawn` never moves: the work starts there, and
issuing it on a path that would not have reached it speculates an effect
(RFC-0007). `Eval` and whole-context `Take`/`Assign` sink within their block
until the first use of a value they define or the first call.

The criterion was purity until `bb8207f`, and purity is not infallibility:
`/` and `%` panic at a zero divisor and at `MIN / -1` (RFC-0037), so a
hoist under purity moved a panic off the program's path.

**`lsr`** (RFC-0056) — an induction variable is a header block parameter
`i` whose latch argument is `i + c` with `c` loop-invariant. `i * k + x`
with `k` and `x` invariant is then itself one: it gets its own header
parameter, started in the preheader and advanced in the latch. The rule is
the operation count — `i * k + x` sheds two and gains one, a bare `i * k`
sheds one and gains one and so is not reduced. Only integer multiplication
is reduced; the float form changes the result and is refused.

**`reorder`** — a topological sort within each block, over two dependency
chains: SSA use-def (an `Order` operand is an ordinary operand, so the
effect chain is preserved by that rule alone) and page-op ordering (a
`Fetch`/`Commit` of a context keeps its order against the page ops and
summary-touching calls of that context, `optimize::context_ops`). Priority:
`Spawn` earliest, `Eval` just before the first use of its result,
everything else in its original order.

**`drop_insertion`** — `Drop { src }` after the last use of every value for
which `is_move_only` holds and that no instruction or terminator consumes.
Phase 1 works within a block, at a value's last use where it is not
live-out; phase 2 works on edges, where a value live-out of A is neither
forwarded to B nor live-in to B. A `Take` that reaches a payload through
nothing but options takes the storage's whole value — `Some(v)` *is* `v`
(RFC-0039) — so neither phase drops that storage afterwards.

**`switch_expand`** — replaces every `Terminator::Switch` with the
`TestVariant` + `JumpIf` chain the machine runs today, first in pass 1, so
nothing downstream sees a `Switch`. Deleting it and adding the `switch`
handler in `acvus-interpreter/src/prepare.rs` is RFC-0051's second half;
the two artifacts move together.

## Analysis

All of it takes `&CfgBody` (`analysis/`).

- **`dataflow`** / **`domain`** — the generic forward/backward engine and
  its fixpoint algebra. An analysis supplies a key, a semilattice domain, a
  per-instruction transfer, the terminator's reads, and the two propagate
  directions.
- **`domtree`** — Cooper–Harvey–Kennedy. `DomTree::build`,
  `PostDomTree::build`, `idom`, `dominates`, `post_dominates`. An
  unreachable block has `idom = UNREACHABLE`, and every query handles it
  without panicking.
- **`loops`** — the only definition of a loop: a back edge whose head
  dominates its tail, with the body the head plus every block reaching the
  tail without passing the head. It answers the header, the latches, the
  body, a block's loop depth, and what loop-invariant means. `code_motion`
  and `lsr` both read it, so the two cannot drift apart.
- **`liveness`** — backward: `is_live_in` / `is_live_out` per block.
  `Return { value, order }` marks both live; `JumpIf` marks its condition.
- **`loans`** — the storage a value may name, as a region: a set of loans,
  joined by union. A `Ref` starts one at its slot; a reference-typed
  parameter or capture starts one at itself; a value whose type contains a
  reference joins the regions it was built from. Every pass that orders,
  moves, removes or allocates around storage asks here rather than reading
  the instruction itself.
- **`escape`** — does a storage slot reach anything outside this body?
  `validate::exhaustive` and `optimize::sroa` ask the same function, and a
  missed escape costs them opposite amounts — a refused exhaustiveness
  claim on one side, a miscompile on the other — so the arms are enumerated
  over the whole instruction set. A new instruction breaks the build rather
  than reading as non-escaping.
- **`inst_info`** — `defs(&InstKind)`, `uses(&InstKind)`. Used by
  everything.

## Validation

`validate(&MirModule)` runs on the final `MirBody`, after every
optimization and after `demote`, so it covers the optimizer as well as the
source. `borrow_check` and `exhaustive` run earlier, in pass 0, because
they read the shape the source wrote.

- **`type_check`** — every instruction's operands against `val_types`:
  arity, constructor shape, `Ref`/`Take`/`Assign` against
  `Ref(Mutability, T)`, `Merge` over `Order`, and the `OrderEdge` rule.
  `main`'s `Return` is held to `MirModule::ret`. `Ty::Error` matches
  anything.
- **`move_check`** — forward dataflow over a promoted clone. A `Take` of a
  storage whose value was taken is `UseAfterMove`; `Assign` consumes its
  value and revives the storage; calls, constructors, `Return`, `Drop` and
  `UnwrapVariant` consume their operands, while `Ref`, field access and
  tests do not; the join at a merge is `Alive ⊔ Moved = Moved`. A `Fn` with
  move-only captures is `FnOnce`, transitively.
- **`borrow_check`** — RFC-0018's exclusion over the CFG: while a `&mut` to
  a storage is live, no other name of that storage is read or written;
  while a `&` is live, the storage is not assigned, moved out of, or
  mutably referenced. A loan's holder is live from its definition to its
  last use (RFC-0029).
- **`exhaustive`** — reads `InstKind::Switch`, the one shape that names a
  `match`, and asks `known_variants` whether the variant set is closed
  (RFC-0051 rules 3 and 4). An `if let` is two arms and never wears a `Switch`.
- **`init_check`** — definite assignment, field-level, on the pre-SSA
  `CfgBody`: which fields of each storage are initialized at each point,
  and whether a call's arguments carry every field the parameter type
  requires. Required fields come from the instruction's `callee_ty`, not
  from `val_types`, which unification may have widened.

---

# Part II — What breaks if you change it

## Block parameters, not PHI nodes

```
Jump { label: L0, args: [v1, v2] }
BlockLabel { label: L0, params: [v3, v4] }
```

Params are defs, args are uses. Three consequences:

- Inlining remaps arguments; there is no PHI surgery.
- **Terminator args are uses.** They live inside terminators, which are not
  in the instruction array, so anything that indexes instructions flatly
  misses them — liveness, dataflow and every pass that counts uses.
- **`seal_block()` has an ordering requirement.** It may be called only
  after every predecessor of that block has been added. Earlier, it
  resolves phis against an incomplete predecessor set and silently loses
  values.

`ENTRY_LABEL = Label(u32::MAX)` identifies the implicit entry block, which
has no `BlockLabel` instruction. Labels are allocated from `label_count`,
so using `ENTRY_LABEL` as a real label makes `promote` mistake a regular
block for the entry.

In `use_var_sealed`, the phi value is inserted into `current_defs` before
`resolve_phi` runs; without that pre-definition a loop back edge recurses
forever. Trivial-phi elimination excludes the phi itself from "all incoming
values identical" — without the exclusion a back edge referencing the phi
counts as identical and the phi is wrongly removed.

## Contexts write back; locals do not

Whole `Take`/`Assign` of a promotable local disappear after SSA; values
flow through block params and nothing else. A context is external state, so
a store inside a branch must persist past the merge: the pass removes the
branch-internal whole `Assign` and splices one write-back of the phi value
at the merge block.

Promotion is all-or-nothing per storage. `collect_ssa_info` first scans for
any `Ref`, or any `Take`/`Assign` with a non-empty path, and pins that
storage; a pinned storage keeps every one of its instructions. **A new
instruction kind that touches a storage and is not added to that scan makes
the pass promote a storage still read through memory.**

Entry definitions are spliced at position 0 of block 0 after phi insertion.
If that splice moves, a written context's entry value is no longer defined
before its first use.

## The inliner binds parameters by slot

A callee's `params` and `captures` are `(name, ValueId)` pairs. The inliner
maps each capture and param register directly to the caller's argument, and
the callee's `order_param` to the call's `OrderEdge::before`. A callee with
an order param at a call with no edge, or the reverse, is a panic: the two
are set by the same type.

**Arguments must be remapped before inlining**, through the current
`val_remap`; afterwards new `ValueId`s are mixed in and the map is
unreliable. In `double(inc(3))`, if `inc`'s result was remapped r5→r10,
`double`'s argument must use r10.

Callee labels are offset by `label_offset = current.label_count`, then
`current.label_count += callee.label_count`. This relies on labels being
allocated linearly and never reused.

Devirtualization requires the `Indirect(v)` callee's definition to be
exactly one `MakeClosure`, traced through the def map, and not through a
block parameter. Devirtualizing through a phi is unsound: at run time the
closure could be a different one.

## Forward and backward dataflow are not inverses

`propagate_forward` maps jump args from the source block's exit state onto
the target's params, and joins the rest of the exit state into the target's
entry. `propagate_backward` marks a terminator's args live where the
successor's params are live at its entry, **and** joins the successor's
non-param entry values into this block's exit. That second half is the
asymmetry: in forward analysis values that bypass params propagate through
the join on their own; in backward analysis the flow-through must be
explicit, or a value live across a block boundary disappears from the exit
state.

`terminator_uses` is where a terminator's reads enter the state: `Return`
marks its value and its order, `JumpIf` its condition. Forward analysis
does not need it — it tracks what is true at a point, not what is used —
and a terminator never defines a value.

## Block boundaries

`promote()` builds the CFG from the flat stream: a `BlockLabel` starts a
block, carrying its `params`; a `Jump`/`JumpIf`/`Switch`/
`Return`/`Diverge` ends one as its terminator, and a block that ends
without one gets `Fallthrough`.

**A `Fallthrough`'s successor is `BlockIdx(current + 1)`**, which assumes
the block array is sequential. Reordering the block array breaks it.

**The entry block is always `BlockIdx(0)`**, with `ENTRY_LABEL`;
instructions before the first `BlockLabel` form it.

## Code motion is sound by construction

Control equivalence is "executes iff", not "executes as often", which is
why the loop-depth condition stands beside it. The set of instructions that
may move is decided by `hoistable`, an enumeration over `InstKind` whose
default is `No`: a new instruction kind never becomes movable silently.
`Spawn`'s `No` is RFC-0007; `MakeObject`/`MakeArray`/`MakeVariant`/
`MakeClosure` and a heap `Const` are built where they are used;
`UnwrapVariant` and a checked `Index` each assume a test that a block
before them established.

## Typeck

`param_types: SmallVec<[(Astr, InferTy); 4]>` preserves insertion order.
The declared parameter list is built by iterating it, and the caller places
arguments in that order; a hash map here makes argument binding
non-deterministic.

`check_lambda` swaps `body_effect` for a fresh effect variable while
checking the lambda body and restores the outer one afterwards; the
lambda's effect goes into its `Fn` type. A lambda produces effects at its
call site, not at its definition site, so propagating them upward would
mark the enclosing function effectful for a definition alone.

The same function rejects a captured `Ref` type (`ReferenceCaptured`) and a
returned one (`ReferenceReturned`), which is what keeps the reference check
local to one body (RFC-0018).

## Change this, break that

| Change | Consequence |
|---|---|
| Make `param_types` a hash map | argument-to-parameter binding becomes non-deterministic |
| Touch a storage from a new instruction without pinning it in `collect_ssa_info` | SSA promotes a storage still read through memory → stale value |
| Remove the pre-definition in `SSABuilder` | infinite recursion resolving a loop back edge |
| Remove the self-reference filter from trivial-phi elimination | a back edge counts as "all incoming identical" → the phi is wrongly removed |
| Drop `order` from `Terminator::Return`'s liveness | the body's final `Order` is not live at the exit |
| Remove the `Assign` revive in `move_check` | reassignment after a move is reported as use-after-move |
| Reorder the block array | `Fallthrough`'s `idx + 1` successor is wrong |
| Use `ENTRY_LABEL` as a real label | `promote` mistakes a block for the entry |
| Make an instruction movable without control equivalence | a panic or an effect runs on a path that did not reach it (RFC-0007, RFC-0037) |
| Allow a target deeper in the loop nest than the source | a hoist puts work inside a loop |
| Make `Spawn` a DCE root | a handle no `Eval` consumes stays, and dead work is issued |
| Make `Take` not a DCE root | a take that empties a storage is removed, and a later read sees a value that should be gone |
| Give a `UserDefined` type argument a non-invariant join position | widening inside a container → run-time type confusion |
| Remove the identity arm from the join | values from unrelated sources merge silently |
| Remove the occurs check | `T = Array<T>` is an infinite type and the resolver loops |
| Let a conversion decision accept several rules | ambiguous coercion resolved non-deterministically |
| Settle a decision inside an SCC before every function is checked | mutual recursion freezes types before later constraints arrive |
| Emit a Pure call with an `OrderEdge`, or an effectful one without | `type_check` rejects it — the invariant every pass relies on |
| Count fewer escape kinds in `analysis::escape` | `sroa` scalarizes an aggregate something outside the body reads |
| Fold an operation the machine would panic on | the program's panic moves or disappears (RFC-0055) |
| Introduce an operator `prepare::arith_of` does not claim | the chain the recognizer would have fused splits (RFC-0055) |

## What is not guaranteed

- **No speculative issue.** IO inside a branch is issued in that branch;
  only `commute`'s post-dominance rule moves a call across blocks.
- **Plugin effects are not verified.** A handler that declares `pure` and
  does IO is a wrong library, and nothing catches it.
- **Integer overflow, an out-of-range index no interval proves, and
  division by zero** are run-time panics (RFC-0044).
- **Termination.** Recursion is detected by SCC, not prevented.
- **`Undef` is not eliminated.** It is a placeholder: valid to move or
  copy, undefined to read as a concrete value.
