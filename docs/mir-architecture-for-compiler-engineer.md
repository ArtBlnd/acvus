# acvus MIR — Compiler Engineer's Guide

This document is for compiler engineers modifying the `acvus-mir` codebase. Part I maps the territory — crate hierarchy, data structures, and the full compilation pipeline. Part II explains why things are the way they are, and what breaks if you change them.

---

# Part I — The Territory

## Crate Hierarchy

```
acvus-utils          Foundation: Interner, Astr, Freeze, LocalId, QualifiedRef
  |
acvus-ast            Parser: Template/Script → AST (Expr, Stmt, Pattern, Pipe)
  |
acvus-mir            Compiler core: type system, IR, analysis, optimization
  |
  +-- acvus-extern         Declaring the world outside the language: #[extern_fn]
  |     |                  (acvus-extern-macro), Registry, Externs::combine, Runtime trait
  |     |
  |     +-- acvus-ext            Standard library of ExternFns
  |     +-- acvus-ext-llm        LLM ExternFns (depends on acvus-ext)
  |     +-- acvus-interpreter    Runtime: Executable, Executor (Sequential, Tokio)
  |           |
  |           +-- acvus-orchestration  Multi-source compilation, incremental rebuild
  |
  +-- kovac-interpreter    Second runtime over the same MIR (untyped scalar coloring)
  |
acvus-lsp            Language server (shares exact same pipeline as compiler)
```

**Key rule:** `acvus-mir` knows nothing about any runtime. The IR is designed for *any* backend. `acvus-extern`, `acvus-interpreter`, and `kovac-interpreter` depend on `acvus-mir`, never the reverse.

---

## Core Data Structures

### IR Representations

```
MirBody                          CfgBody
+-- insts: Vec<Inst>             +-- blocks: Vec<Block>
+-- val_types: Map<ValueId,Ty>       +-- label: Label
+-- params: Vec<(Astr,ValueId)>      +-- params: Vec<ValueId>
+-- captures: Vec<(Astr,ValueId)>    +-- insts: Vec<Inst>      (no control flow)
+-- order_param: Option<ValueId>     +-- terminator: Terminator
+-- debug: DebugInfo                 +-- merge_of: Option<Label>
+-- val_factory: LocalFactory    +-- label_to_block: Map<Label,BlockIdx>
+-- label_count: u32             +-- val_types, params, captures, order_param, ... (shared)

MirModule                        Terminator
+-- main: MirBody                +-- Jump { label, args }
+-- closures: Map<Label,MirBody> +-- JumpIf { cond, then_label/args, else_label/args }
                                 +-- Return { value, order }
                                 +-- Fallthrough
```

**MirBody** is the flat representation — a linear stream of instructions including control flow (`BlockLabel`, `Jump`, `JumpIf`, `Return`). This is what lowering produces and what the interpreter consumes.

**CfgBody** is the structured representation — basic blocks own their instructions, control flow is in terminators. This is what all analysis and optimization passes operate on.

**Lifecycle:** `promote(MirBody) → CfgBody` → run passes → `demote(CfgBody) → MirBody`.

`order_param` is the `Order` a body takes first when its effect is not Pure (RFC-0007); `Return { order }` is the one it yields last.

### InstKind — The 36 Instructions

| Category | Instructions | Notes |
|----------|-------------|-------|
| Constants | `Const`, `Undef`, `Poison` | Undef = SSA placeholder (valid to move, UB to read), Poison = type error marker |
| Storage | `Ref`, `Take`, `Assign`, `Load`, `Store` | `RefTarget` is `Var(slot)`, `Param(slot)`, or `Context(qref)`; `path` names a field chain, empty = the storage itself |
| Scalar field | `FieldGet`, `FieldSet` | On a value, not a storage; `FieldSet` produces a new value |
| Arithmetic | `BinOp`, `UnaryOp` | `UnaryOp::Deref` is lowered to `Load`, not `UnaryOp` |
| Functions | `LoadFunction`, `FunctionCall` | `FunctionCall { callee: Direct \| Indirect, callee_ty, args, order: Option<OrderEdge> }` |
| Async | `Spawn`, `Eval`, `Merge` | Spawn takes the order before, Eval yields the order after; Merge joins orders |
| Construction | `MakeArray`, `MakeObject`, `MakeTuple`, `MakeVariant`, `MakeClosure` | All pure |
| Access | `TupleIndex`, `ArrayIndex`, `ArrayGet`, `ObjectGet`, `UnwrapVariant` | `ArrayIndex` = constant index, `ArrayGet` = value index |
| Test predicates | `TestLiteral`, `TestObjectKey`, `TestVariant` | Return Bool, used by JumpIf |
| Control flow | `BlockLabel`, `Jump`, `JumpIf`, `Return` | Become blocks and terminators in CfgBody |
| Resources | `Drop` | Inserted by drop insertion at a value's last use; consumes, never defines |
| Utility | `Nop` | Placeholder after instruction removal |

The five storage instructions (RFC-0018):

- `Ref { dst, target, path, mutability }` — `dst` is a `&T` or `&mut T` naming the value at `path` under `target`.
- `Take { dst, target, path }` — move the value out of a storage into `dst`; a primitive is copied and the storage keeps it. A read of a variable.
- `Assign { target, path, value }` — move `value` into a storage; the old value is dropped. An assignment to a variable.
- `Fetch { dst, context }` — move a context's whole value out of the page into `dst` (RFC-0025).
- `Commit { context, value }` — move `value` into the page as the context's whole value (RFC-0025).
- `Load { dst, src }` — `*r`: read through a `&T`; `T` must be a primitive.
- `Store { dst, value }` — write through a `&mut T`.

A context is a variable of the body that names it (RFC-0025): the lowering fetches every named context into a slot at entry, commits each slot at every return, and brackets each call whose summary (RFC-0017, joined with the summaries of its function-typed arguments) touches the context with a `Commit` before and a `Fetch` after (`lower.rs`, `enter_contexts`, `emit_call`, `emit_return`). `@x`, `&@x`, and `@x = v` are then the variable rules on that slot; a context left moved out at a return is a use-after-move at the exit `Take`. The reorder pass keeps a `Fetch`/`Commit` in order against the page ops and the summary-touching calls of the same context.

### Type System

```
Ty = TyTerm<Concrete>;  PolyTy = TyTerm<Poly>;  InferTy = TyTerm<Infer>

TyTerm::Int | Float | String | Bool | Unit | Byte
   | Order                          -- dependency between effectful calls; IR-only (RFC-0007)
   | Array(Box<Ty>, LenTerm)        -- length is a term: known or a variable
   | Object(Map<Astr,Ty>)           -- structural typed record
   | Tuple(Vec<Ty>)
   | Option(Box<Ty>)
   | Fn { params: Vec<ParamTerm>, ret, captures, effect: EffectTerm }
   | UserDefined { id, type_args, effect_args, identity_args }
   | Enum { name, variants: Map<Astr, Option<Box<Ty>>> }
   | Handle(Box<Ty>)                -- async handle from Spawn
   | Ref(Mutability, Box<Ty>)       -- &T / &mut T (RFC-0018); a word, never data
   | Error(ErrorToken)              -- type error sentinel
   | Var(Phase::TyVar)              -- inference variable; uninhabited when Concrete
```

The phase parameter decides which variables exist: `Concrete` has none, `Poly` has positional placeholders for a declaration, `Infer` has solver variables. `ParamTerm { name, ty }` is a named parameter; a parameter that borrows has a reference type — there is no mode beside the type.

### Effect System

```
Effect {
    reissue:  Reissue,                -- Pure < Idempotent < Opaque (RFC-0014)
    commutes: bool,                   -- two calls are the same program in either order (RFC-0013)
    reads:    BTreeSet<QualifiedRef>, -- contexts the call may read  (RFC-0017)
    writes:   BTreeSet<QualifiedRef>, -- contexts the call may write (RFC-0017)
}

EffectTerm<V> = Known(Effect) | Var(V::EffectVar)
```

Every constructor keeps two invariants: a Pure call that writes nothing commutes, and a call that writes a context never commutes. The join of two effects is the join on each axis: the higher reissue level, commutative only if both are, union of reads and of writes.

`Order` is how the effect reaches the IR. A call whose effect is not Pure carries `OrderEdge { before, after }`; a Pure call carries none. The type checker of the final IR rejects the other two combinations.

### Identity (RFC-0012)

```
IdentityTerm<V> = Known(IdentityId) | Var(V::IdentityVar)
UserDefinedDecl { qref, type_params: Vec<TyVarBound>, effect_params: usize, identity_params: usize }
```

Identity is a parameter of a user-defined type, at most one per declaration. Two identities unify only when they are the same. An identity variable tied to no source is a new source when it freezes (`Solver::settle_identities`): an ExternFn that returns a type with an identity variable found nowhere in its parameters returns a fresh source at every call. A user-defined value with an identity parameter is one source and moves. No structural type carries an identity, and the runtime sees none of it.

### Type Variables and Bounds

```
TyVarBound = Any | OneOf(Vec<Ty>)     -- declared with the variable (RFC-0011)
Scheme { ty: PolyTy, bounds: Vec<TyVarBound> }
```

Solver variables (`TypeBoundId`) start unbound and get bound through `unify_ty`. `TyVarBound::meet` intersects two `OneOf` bounds; an empty intersection is `None` — an immediate type error. The occurs check (`occurs_in`) prevents a variable from binding to a type containing itself.

### Polarity (Variance)

```
Polarity::Covariant       -- a ≤ b (subtype allowed)
Polarity::Contravariant   -- b ≤ a (reversed)
Polarity::Invariant       -- a = b (exact match)
```

| Position | Polarity | Site in `unify_ty` |
|----------|----------|--------------------|
| Function params | `pol.flip()` | `Fn` vs `Fn` |
| Function return | `pol` | `Fn` vs `Fn` |
| Function effect | `pol` | `unify_effect` |
| `UserDefined` type args | `Invariant` | same-id case |
| `UserDefined` effect args | `pol` | same-id case |
| Identity args | exact | `unify_identity` |

### Unification (`Solver::unify_ty`)

```
unify_ty(a, b, pol, registry) → Result<Option<QualifiedRef>, (InferTy, InferTy)>
```

1. **Shallow resolve** both sides (follow variable bindings without recursing into structure).
2. **Trivial cases**: `Error` unifies with anything (poison). Identical primitives succeed.
3. **Same-head structural recursion** as in the polarity table. A `UserDefined` mismatch inside a snapshot is rolled back and handed to `lub_or_err_infer`.
4. **Different heads in a non-invariant context** → `try_coerce_infer(sub, sup)`. `Ok(Some(fn_ref))` means a cast ExternFn was chosen; the lowerer then emits a pure `FunctionCall` of it (`Lowerer::maybe_cast`).

### Coercion Rules (`CastRule`, `TypeRegistry`)

```
CastRule { from: PolyTy, to: PolyTy, fn_ref: QualifiedRef }
TypeRegistry { decls, from_rules: Map<QualifiedRef, Vec<CastRule>>, to_rules: Map<QualifiedRef, Vec<CastRule>> }
```

Cast rules are derived when registries are combined (RFC-0021). `from` must be a `UserDefined`; `from_rules` is keyed by its id, `to_rules` by the target's id when the target is `UserDefined`. Resolution probes each candidate under `snapshot` / `rollback`; exactly one match is applied. There is no cast instruction: the coercion is the call.

### Least Upper Bound (`try_lub_infer`)

When same-head unification fails in a non-invariant context:

| Mismatch | LUB |
|----------|-----|
| `Fn` vs `Fn` (params and return unify invariantly) | `Fn` with `lub_effect` of the two effects |
| `UserDefined(A)` vs `UserDefined(A)` (args differ) | the same id with `lub_effect` on effect args, invariant type args |
| `UserDefined(A)` vs `UserDefined(B)` | a target both can coerce to, via cast rules |

### Effect Unification (`Solver::unify_effect`)

- Two known effects: in `Invariant` position the context sets must be equal and `join ≤ meet`; in `Covariant` position `a` must be at most `b`; in `Contravariant`, the reverse.
- A variable against a known effect: `Invariant` binds it; `Covariant` / `Contravariant` narrow its range (`lower_upper` / `raise_lower`).
- Two variables are forwarded to one root.

`lub_effect` joins two known effects, or raises a variable's lower bound.

### Type Inference Pipeline (SCC-based)

```
[Build call graph]  extract_call_edges: function → callees
        |
[Tarjan's SCC]  tarjan_scc: reverse topological order (leaves first)
        |
[Per-SCC inference]  infer_scc
   For each function in SCC:
     1. Instantiate its return type; allocate a fresh effect variable
     2. Build a tentative Fn type with those variables
     3. Type-check the AST body (fills the solver via unify)
     4. Freeze all variables → concrete Fn type
   Available to subsequent SCCs as concrete types.
        |
InferResult { outcomes, ... }
```

**Key property:** Within an SCC, functions see each other's **tentative** types (with unbound vars). After the SCC finishes, all vars are resolved. The next SCC sees only concrete types. This allows polymorphic recursion within an SCC while maintaining concrete types across SCC boundaries.

### Compilation Graph

```
CompilationGraph {
    functions: Freeze<Vec<Function>>,   -- all callable entities (local + extern)
    contexts:  Freeze<Vec<Context>>,    -- all named external values (@user, @items, ...)
}

Function { qref: QualifiedRef, kind: FnKind, ty: PolyTy }
FnKind::Local(ParsedAst)                        -- user-written template/script
FnKind::Extern { bounds: Vec<TyVarBound> }      -- plugin-provided handler

Context { qref: QualifiedRef, ty: PolyTy }      -- Var = to be inferred
```

---

## The Full Compilation Pipeline

```
Source text
    |
    v
[Parse]  acvus_ast::parse / parse_script
    |
    v
ParsedAst (Template | Script)
    |
    v
[Build CompilationGraph]  externs (Externs::combine) + local functions + contexts
    |
    v
CompilationGraph { functions, contexts }
    |
    v
[Phase 0: Extract]  graph/extract.rs
    |    Parse and cache the AST of every local function.
    |    Output: ExtractResult { parsed: Map<QualifiedRef, ParsedSource> }
    v
[Phase 1: Infer]  graph/infer.rs
    |    Constraint-based type inference across all functions.
    |    SCC analysis for mutually recursive functions.
    |    Resolves: function signatures, context types, effects.
    |    Output: InferResult { outcomes: Map<QualifiedRef, FnInferOutcome> }
    v
[Phase 2: Lower]  graph/lower.rs → lower.rs
    |    Typed AST → flat MIR instructions (pre-SSA).
    |    Variable read → Take; assignment → Assign; &place → Ref; *r → Load.
    |    Context: Fetch at entry, Commit at return, Commit/Fetch around a call by summary.
    |    Effectful call → FunctionCall with an OrderEdge on the body's order slot.
    |    Pattern matching → TestXxx + JumpIf chains.
    |    Output: Map<QualifiedRef, MirModule>
    v
[Phase 3: Optimize]  graph/optimize.rs
    |
    |  ┌─── Pass 1 ──────────────────────────────────────────────┐
    |  │  For each body (main + closures):                        │
    |  │    promote(MirBody) → CfgBody                           │
    |  │    SSA pass  →  DSE  →  DCE                              │
    |  │    demote(CfgBody) → MirBody                            │
    |  │  Then:                                                   │
    |  │    Inline (cross-module, devirtualization)               │
    |  └──────────────────────────────────────────────────────────┘
    |
    |  ┌─── Pass 2 (per body) ───────────────────────────────────┐
    |  │    promote(MirBody) → CfgBody                           │
    |  │    Commute        Commutative runs share one Order       │
    |  │    SpawnSplit     Effectful FunctionCall → Spawn + Eval  │
    |  │    SSA pass       Re-normalize after inlining/splitting  │
    |  │    DSE            Dead context stores                    │
    |  │    DCE            Dead pure instructions, dead handles   │
    |  │    CodeMotion     Hoist pure; sink Eval and context ops  │
    |  │    Reorder        Spawn early, Eval late within blocks   │
    |  │    (debug build)  use-def / dominance / type coverage    │
    |  │    DropInsertion  Drop at last use of a move-only value  │
    |  │    RegColor       SSA-aware greedy register coloring     │
    |  │    demote(CfgBody) → MirBody                            │
    |  │  Then, per module:                                       │
    |  │    Validate (type check + move check on final MirBody)  │
    |  └──────────────────────────────────────────────────────────┘
    |
    v
Map<QualifiedRef, MirModule>  (optimized, validated)
    |
    v
[Runtime]  acvus-interpreter
    Executable::Module(MirModule) | Extern(ExternEntry)
    Executor: SequentialExecutor | TokioExecutor
```

---

## Pass 2 Pipeline — Detail

Each pass operates on `&mut CfgBody`. A call's effect is read from its own `callee_ty`; there is no function-type table threaded through the passes.

### 1. Commute (`optimize/commute.rs`)

**Input:** CfgBody after SSA construction, with `FunctionCall` order edges.
**Output:** Each maximal run of neighbouring commutative calls takes the run's entry `Order`; one `Merge` of everything they yielded stands where the last call's order stood.

- A run is read on `FunctionCall` edges within one block; a phi is not a neighbour, so a run never crosses a branch.
- Before runs are read, a commutative call whose block post-dominates the block of the call it follows is moved there (`PostDomTree`) — every path through that block reaches it, so nothing is speculated.
- A call joins a run only if no store between the run's first call and itself names a context in the call's read or write set, and no load names one in its write set (RFC-0017). A call that touches any context is never moved across blocks.

### 2. SpawnSplit (`optimize/spawn_split.rs`)

**Input:** CfgBody with `FunctionCall` instructions.
**Output:** Every `Callee::Direct` call whose effect is not Pure is replaced with `Spawn` + `Eval`.

```
BEFORE:  r1 = call fetch(r0)  order(o0 -> o1)
AFTER:   h  = spawn fetch(r0)  order(o0)
         r1 = eval h           order(o1)
```

- `is_io_call` is `callee_ty.effect()` not Pure: a call is split exactly when it stands on the `Order` chain.
- The handle type `Ty::Handle(ret)` is taken from the instruction's own `callee_ty`.
- Pure calls and indirect calls pass through unchanged.

### 3. SSA Pass, DSE, DCE

Re-run after splitting: see *Pass 1* below. DSE removes whole context `Assign`s overwritten on every path before a read; DCE removes unused pure instructions and unused `Spawn` handles.

### 4. CodeMotion (`optimize/code_motion.rs`)

**Input:** CfgBody with Spawn/Eval and pure instructions.
**Output:** Pure instructions hoisted to dominator ancestors; `Eval` and whole context `Take`/`Assign` sunk toward their first use.

Hoist algorithm:
1. Build the dominator tree.
2. For each hoistable instruction, walk UP the dominator chain to find the **highest ancestor** where all operands are available.
3. `def_block` is updated after each decision → operand chains resolved in one pass.
4. Fixpoint loop (typically 1 iteration).

**Hoistability (allowlist, default deny):**
- Hoistable: `BinOp`, `UnaryOp`, `Const`, `MakeArray`, `MakeObject`, `MakeTuple`, `MakeVariant`, `MakeClosure`, `Ref`, `FieldGet`, `FieldSet`, `ObjectGet`, `ArrayIndex`, `TupleIndex`, `TestLiteral`, `TestVariant`, `TestObjectKey`, `LoadFunction`.
- NOT hoistable: `Spawn` (RFC-0007: issuing it on a path that would not have reached it speculates an effect), `Eval`, any call, `Take`/`Assign`/`Load`/`Store`, `UnwrapVariant` and `ArrayGet` (each assumes a check its test block established).

Sink pass: one instruction at a time, `Eval` and whole context `Take`/`Assign` move down within their block until the first use of a value they define or the first call.

### 5. Reorder (`optimize/reorder.rs`)

**Input:** CfgBody with instructions in each block.
**Output:** Instructions reordered within each block by dependency + priority.

Dependency chains:
1. **SSA use-def** — use must follow def. An `Order` operand is an ordinary operand, so effectful calls stay in chain order.
2. **Context store ordering** — `Store` instructions whose `dst` is a `Ref` to the same context preserve original order.

Priority (topological sort with `BinaryHeap`):
- `Spawn` → earliest.
- `Eval` → `Scheduled(first_use, 0)`: just before the first use of its result.
- Everything else → `Scheduled(original_index, 1)`.

### 6. DropInsertion (`optimize/drop_insertion.rs`)

**Input:** CfgBody after all reordering.
**Output:** `Drop { src }` after the last use of every value for which `is_move_only` is true and that no instruction or terminator consumes.

- Phase 1, within a block: at a value's last use, if it is not live-out and the instruction does not consume it, a `Drop` follows the instruction. An unused definition is dropped right after it.
- Phase 2, on edges: a value live-out of A that is neither forwarded to B nor live-in to B is dropped at the start of B.

### 7. RegColor (`optimize/reg_color.rs`)

**Input:** CfgBody in SSA form.
**Output:** ValueIds compacted — non-overlapping lifetimes share slots.

- SSA-aware set-based greedy coloring (not interval-based linear scan).
- Backward dataflow liveness over CFG.
- Kill order within an instruction: color defs → kill dying uses → kill dead defs.
- Entry params/captures colored with a shared `entry_live` set.
- `color_body` reuses a slot only across the same type; `color_body_untyped` (kovac) colours scalars regardless of type.

---

## Pass 1 — SSA, DSE, DCE

### SSA Pass (`optimize/ssa_pass.rs`)

**Input:** CfgBody with `Take` / `Assign` on storages.
**Output:** Promotable storages replaced by SSA values and block parameters.

- A storage is promotable only when every access to it is a whole `Take` or `Assign` (empty path). One `Ref`, or one access by field path, pins the storage in memory.
- Whole `Take`/`Assign` of a promotable local or parameter → block parameters; the instructions are removed.
- Whole `Assign` of a promotable context inside a branch → removed; a write-back `Assign` of the phi value is spliced at the start of the merge block.
- Entry definitions spliced at the start of block 0: `Undef` for every local read before it is written, and a whole `Take` of every written context (its value on entry).
- Dominator-tree-scoped store-load forwarding for contexts (`forward_context_values`): a whole `Take` after a whole `Assign` in a dominating block is replaced by the assigned value; at a merge point, written contexts are cleared from the forwarding state.
- Trivial PHI elimination (all incoming edges provide the same value, excluding the phi itself).
- Chained substitution: `var_subst ∘ fwd_subst`.

### DSE (`optimize/dse.rs`)

Backward context liveness per block. A read is a whole context `Take`, any call (`FunctionCall`, `Spawn`, `Eval` — taken to read every context, the sound default of RFC-0017), or a `Return` (contexts are observable after the run). A write is a whole context `Assign`. A write that is dead on every path is removed.

### DCE (`optimize/dce.rs`)

Mark-sweep. Roots: `Store`, `Assign`, `Take` (a take leaves its storage empty, so it is observable), `Eval`, and a `FunctionCall` whose effect is not (Pure and writes nothing). `Spawn` is not a root — a handle no `Eval` consumes is dead. Terminator operands are roots.

---

## Analysis Infrastructure

All analyses operate on `&CfgBody`.

### Dataflow Framework (`analysis/dataflow.rs`)

Generic forward/backward dataflow engine.

```rust
trait DataflowAnalysis {
    type Key;                        // What we track (ValueId, ...)
    type Domain: SemiLattice;        // Lattice element (Liveness, AbstractValue, ...)

    fn transfer_inst(&self, inst, state);        // Per-instruction transfer
    fn terminator_uses(&self, term, state);      // Terminator read effects
    fn eval_branch_cond(&self, exit, cond);      // (forward only) prune a known branch
    fn propagate_forward(&self, ...);            // Source exit → target entry
    fn propagate_backward(&self, ...);           // Successor entry → block exit
}
```

Output: `DataflowResult { block_entry, block_exit }` per block.

### Dominator Tree (`analysis/domtree.rs`)

Cooper-Harvey-Kennedy algorithm. `DomTree::build(&CfgBody)`; `PostDomTree::build(&CfgBody)`.

- `idom(block) → Option<BlockIdx>` — immediate dominator.
- `dominates(a, b) → bool`; `post_dominates(a, b) → bool`.
- Unreachable blocks have `idom = UNREACHABLE`; every query handles it without panicking.
- Used by: CodeMotion (hoist target), SSA pass (forwarding scope), Commute (post-dominance), the debug validator.

### Liveness (`analysis/liveness.rs`)

Backward dataflow: which ValueIds are live at each block entry/exit.

- `analyze(&CfgBody) → LivenessResult`.
- `is_live_in(block, val)`, `is_live_out(block, val)`.
- `Return { value, order }` marks both live; `JumpIf` marks its condition live.
- Used by: RegColor, DropInsertion.

### Reachable Context (`analysis/reachable_context.rs`)

Classifies context keys as eager/lazy/pruned by analyzing which branches are reachable given known context values.

- Two-pass: `ValueDomainTransfer` (forward abstract values, `analysis/domain.rs`) → reach BFS over `cfg.successors()`.
- `partition_context_keys(&MirModule, &known) → ContextKeyPartition`.

### Val Def (`analysis/val_def.rs`)

Maps each ValueId to the instruction index that defines it. `build(&MirModule) → ValDefMap`.

### Inst Info (`analysis/inst_info.rs`)

Pure utility: `defs(&InstKind) → SmallVec<ValueId>`, `uses(&InstKind) → SmallVec<ValueId>`, `is_control_flow(&InstKind)`.

Used by everything — dataflow, liveness, reorder, code_motion, reg_color, drop_insertion, the inliner.

---

## Validation (`validate/`)

`validate(&MirModule)` runs on the **final MirBody after all optimizations and demote**. Catches optimizer bugs.

### Type Check (`validate/type_check.rs`)

Every instruction's operands match the types recorded in `val_types`: arity, constructor shape, `Ref`/`Take`/`Assign`/`Load`/`Store` against `Ref(Mutability, T)`, `Merge` over `Order`, and the `OrderEdge` rule — an effectful call carries an order, a Pure call carries none. `Ty::Error` matches anything.

### Move Check (`validate/move_check.rs`)

Every non-primitive, non-reference type is move-only (`is_move_only`). Forward dataflow over a promoted clone of the body:

- `Take` of a storage whose value was already taken → `UseAfterMove`. The take marks the storage `Moved`.
- `Assign` consumes its value and revives the storage (`Alive`).
- Calls, constructors, `Return`, `Drop`, `Store`, `UnwrapVariant` consume their operands; `Ref`, `Load`, field access and tests do not.
- Join at a merge: `Alive ⊔ Moved = Moved`.

### Definite Assignment (`validate/init_check.rs`)

`check_init(&CfgBody, external_contexts)` on the pre-SSA CfgBody: which fields of each storage (`Var`, `Param`, `Context`) are definitely initialized at each point. A storage is written by `Assign` and read by `Take` and `Ref`. At a call, the argument must carry every field the callee's parameter type (from the instruction's `callee_ty`) requires. It is not part of `validate()`; the test harness runs it.

---

# Part II — What Breaks If You Change It

## SSA: Block Parameters vs PHI Nodes

We use Cranelift-style block parameters, not LLVM-style PHI nodes.

```
Jump { label: L0, args: [v1, v2] }
BlockLabel { label: L0, params: [v3, v4] }
```

Block params are definitions (defs), jump args are uses. Consequences of this choice:

- **Inlining is simple.** Remap args and you're done. No PHI surgery required.
- **Terminator args are uses.** This affects liveness, dataflow, and reg_color. Args live inside terminators that are not in the instruction array, so flat instruction indexing misses them.
- **SSABuilder has a seal ordering requirement.** `seal_block()` may only be called after all predecessors of that block have been added. Calling it earlier resolves PHIs against an incomplete predecessor set, silently losing values.

### ENTRY_LABEL Sentinel

`ENTRY_LABEL = Label(u32::MAX)` identifies the implicit entry block, which has no `BlockLabel` instruction. Labels are allocated from `label_count`, so a collision is unrealistic in practice, but not formally guaranteed.

### Loop Back-Edge Cycle Breaking

In `use_var_sealed()`, the PHI value is inserted into `current_defs` before `resolve_phi` runs. Without this, a loop back-edge causes: PHI resolve → same PHI use → same PHI resolve → infinite recursion. The pre-definition breaks the cycle.

### Trivial PHI Elimination

If all incoming edges provide the same value (excluding the PHI itself), the PHI is replaced with that value. The self-exclusion filter is critical — a loop back-edge referencing the PHI itself does not count as "all incoming values are identical". A pending phi is resolved after its block was processed; the block's current definition is replaced only if it is still this phi.

---

## SSA Pass: Context vs Local Variable Asymmetry

The most important design decision: **contexts require write-back, local variables do not.**

Whole `Take`/`Assign` of a promotable local disappear entirely after SSA. Values flow through block params — nothing else.

Contexts are external state, so a store inside a branch must persist after the merge. The SSA pass removes branch-internal whole `Assign`s to a promotable context and splices one write-back `Assign` of the phi value at the merge block.

### Promotion Is All-or-Nothing per Storage

`collect_ssa_info` first scans for any `Ref`, or any `Take`/`Assign` with a non-empty path, and pins that storage. A pinned storage keeps every one of its instructions, including whole takes and assigns. If a new instruction kind touches a storage and is not added to that scan, the SSA pass will promote a storage that is still read through memory.

### Entry Definitions

Entry definitions are spliced at position 0 of block 0 after phi insertion: `Undef` for locals, a whole `Take` for each written context. The SSA builder's `ENTRY_BLOCK` definitions refer to these values; if the splice moves, the entry value of a written context is no longer defined before its first use.

### Chained Substitution

The SSA pass chains two substitution maps: `var_subst` (from SSABuilder) + `fwd_subst` (from store-load forwarding). If `var_subst` maps r10→r5 and `fwd_subst` maps r5→r3, the final result is r10→r3. Reversing the application order produces incomplete forwarding.

---

## Inliner: Parameters Are Bound by Slot

A callee's `params` and `captures` are `(name, ValueId)` pairs. The inliner maps each capture register and each param register directly to the caller's argument (`callee_remap`), and the callee's `order_param` to the call's `OrderEdge::before`. A callee with an order param at a call with no edge, or the reverse, is a panic: the two are set by the same type.

### val_remap Chain

When sequentially inlining multiple calls, each inline's result dst is added to `val_remap`. The next inline's args are remapped through the current `val_remap`. In `double(inc(3))`, if inc's result is remapped r5→r10, then double's arg must use r10 instead of r5.

**Ordering matters:** Args must be remapped before inlining. After inlining, new ValueIds are mixed in and remap becomes unreliable.

### Label Offset Accumulation

When inlining, all callee labels are offset by `label_offset = current.label_count`, then `current.label_count += callee.label_count`. This relies on labels being allocated linearly and never reused.

### Devirtualization Conditions

To devirtualize an `Indirect(v)` callee (`try_devirt`):
1. `v`'s definition must be exactly one `MakeClosure` (traced through def_map)
2. It must not pass through a block param (which would mean multiple possible definitions)

Devirtualizing through a phi is unsound — at runtime, the closure could be a different one.

---

## Dataflow: Forward and Backward Are Not Simple Inverses

### Forward: propagate_forward

Maps jump args from the source block's exit state to the target block's params. Additionally joins the entire exit state into the target entry (non-param values flow through).

### Backward: propagate_backward

If successor params are live at the successor's entry, marks the corresponding terminator args as live at this block's exit. **Additionally** joins non-param values from the successor's entry into this block's exit.

**Why the asymmetry:** In backward analysis, values live at a successor's entry that bypass params (= values live across block boundaries without going through block params) must also be live at this block's exit. In forward analysis, such values propagate naturally through the join. In backward, this flow-through must be explicit.

### Terminator Uses

`terminator_uses` is where a terminator's reads enter the state: `Return { value, order }` marks both, `JumpIf` marks its condition. Forward analysis does not need this — forward tracks "what is true at this point", not "what is used". A terminator never defines a value.

---

## CFG: Block Boundary Rules

The CFG is built from a flat instruction stream by `promote()`. Rules:

1. `BlockLabel` → starts a new block (flushes the previous one), carrying its `params` and `merge_of`
2. `Jump`/`JumpIf`/`Return` → ends a block (becomes its terminator); a block ending without one gets `Fallthrough`

**Fallthrough successor is `BlockIdx(current + 1)`.** This assumes the block array is sequential. Reordering blocks breaks fallthrough.

**Entry block is always `BlockIdx(0)`** with `ENTRY_LABEL`. Instructions before the first `BlockLabel` form the implicit entry block.

---

## Reorder: Two Dependency Chains

The reorder pass builds two dependency chains within each basic block:

1. **SSA use-def** — A use must come after the instruction that defines its ValueId. The `Order` operands of effectful calls are ordinary operands, so the effect chain is preserved by this rule alone.
2. **Context store ordering** — `Store` instructions whose `dst` traces (through the block's def_map) to a `Ref` of the same context preserve original order. **The `Ref` must be in the same block** — if it's in a different block, the def_map lookup fails and the ordering edge is not added.

If the chains conflict, a cycle can occur. The current implementation catches cycles with an assert but has no graceful recovery.

### Priority-Based Topological Sort

Uses a `BinaryHeap` (min-heap via `Reverse`). `Spawn` = earliest. `Eval` = `Scheduled(first_use, 0)` (just before first use). Normal = `Scheduled(index, 1)`.

---

## Spawn Split: What It Reads

Spawn split reads only the instruction: `callee_ty.effect()` decides whether to split, `callee_ty`'s return type gives the `Handle` type, and the `OrderEdge` is divided — `before` to the `Spawn`, `after` to the `Eval`. Nothing is looked up in a side table, so the pass has no ordering precondition on the SSA pass; the pipeline runs SSA after it to fold the values it introduced.

### Only Direct Calls Are Split

An `Indirect` callee passes through unchanged, whatever its effect.

---

## Code Motion: Soundness by Construction

The hoistability check uses an **allowlist** (default deny). Only provably pure instructions are hoisted. Unknown/new instruction kinds are automatically blocked. This means adding a new InstKind never silently becomes hoistable — you must explicitly opt in.

### Spawn Is Never Hoisted

The work starts at the `Spawn`. Moving it to a dominator would issue it on a path that never reaches it, and `Order` says "after", not "only if" (RFC-0007). The one cross-block motion of a call is Commute's post-dominance rule, which speculates nothing.

### Sink Barriers

`Eval` and whole context `Take`/`Assign` sink within their block until the first instruction that uses a value they define, or the first call (`FunctionCall`, `Spawn`, `Eval`). A call is a barrier for a context op because a call is taken to read and write every context (RFC-0017's sound default).

---

## Typeck: param_types Ordering Matters

`param_types: SmallVec<[(Astr, InferTy); 4]>` preserves insertion order. The declared parameter list is built by iterating it in order, and the caller places args in this order. A hash map here would make arg-param binding non-deterministic.

### Lambda Effects Do Not Propagate Upward

`check_lambda` swaps `body_effect` for a fresh effect variable while checking the lambda body and restores the outer one afterwards; the lambda's effect goes into its `Fn` type. This is deliberate: a lambda produces effects at call site, not at definition site. The enclosing function only needs to know "calling this lambda has effects".

### References Never Escape

The same function rejects a captured `Ref` type (`ReferenceCaptured`) and a returned one (`ReferenceReturned`). This is what keeps the reference check local to one body (RFC-0018).

---

## Move Check: Assign Revives Storage

After a move-only value is taken out of a storage, assigning a new value to the same storage brings it back to `Alive`.

```
a = move_only_value;  // a: Alive
consume(a);           // a: Moved  (Take marks the storage)
a = new_value;        // a: Alive again (Assign)
use(a);               // OK
```

**`Take` does not revive.** A take is a read, and for a move-only type it is the consumption.

### Conservative Join at Branch Merge

If one branch has `Alive` and the other has `Moved(at: 3)`, the merge result is `Moved(at: 3)`. This is conservative — if any path could have moved the value, it's treated as moved.

---

## Reachable Context: Branch Pruning with Known Values

`partition_context_keys` classifies context keys as eager/lazy/pruned. The key mechanism: known context values (provided by the caller) are used to evaluate branch conditions and prune dead branches.

**Pruned keys are type-inject only.** Code in a dead branch has already passed type checking and still carries the context's type; the caller injects the type and does not fetch the value.

---

## Register Coloring: SSA-Aware Greedy

The implementation uses **set-based greedy coloring** over CFG-aware liveness, not interval-based linear scan.

- `Coloring` struct: `assign()`, `color_of()`, `is_colored()`, `is_improvement()`.
- `LiveColors` struct: tracks which colors are currently live at a program point.
- `LastUseMap` struct: precomputes where each value's last use is within a block.
- Kill order within an instruction: color defs → kill dying uses → kill dead defs.
- Entry params/captures: colored with a shared `entry_live` set to prevent conflicts.

**Why not interval-based linear scan?** SSA produces a chordal interference graph, where greedy coloring on a perfect elimination ordering is optimal. The set-based approach naturally handles cross-block liveness without constructing intervals.

---

## "Change This, Break That"

| Change | Consequence |
|--------|-------------|
| Make `param_types` a HashMap | Extern function parameter order becomes non-deterministic → arg-param binding errors |
| Add a storage-touching instruction without pinning its storage in `collect_ssa_info` | SSA promotes a storage still read through memory → stale value |
| Remove pre-definition in SSABuilder | Infinite recursion on loop back-edge PHI resolution |
| Remove `order` from `Terminator::Return` liveness | The body's final `Order` is not marked live → reg_color reuses its slot |
| Remove Assign revive logic | Reassignment after move is falsely flagged as use-after-move |
| Allow cross-block `Ref` in reorder's store back-trace | def_map miss → ordering lost → stores to same context reordered → unsound |
| Propagate lambda effects to enclosing scope | Lambda definition alone marks outer function as effectful → unnecessary inline restrictions |
| Remove non-param flow-through in backward propagation | Cross-block live values disappear from exit state → reg_color reuses their slots → value corruption |
| Reorder block array | Fallthrough successor idx+1 calculation breaks → wrong successor visited |
| Use `ENTRY_LABEL` as a real label | `promote` confuses the entry block with a regular block |
| Remove self-reference filter from trivial PHI elimination | Loop back-edge self-reference triggers "all incoming identical" → PHI incorrectly eliminated |
| Add `Spawn` to the code-motion allowlist | A call is issued on a path that never reaches it → speculated effect (RFC-0007) |
| Add `UnwrapVariant` or `ArrayGet` to the allowlist | Runs before the test that guards it → wrong tag / out-of-range index |
| Make `Spawn` a DCE root | A handle no `Eval` consumes stays → dead work is issued |
| Make `Take` not a DCE root | A take that empties a storage is removed → later `Assign`/`Take` see a value that should be gone |
| Give `UserDefined` type args a non-invariant polarity | Widening inside a container → runtime type confusion |
| Remove identity from `unify_ty` | Values from unrelated sources silently merge → data corruption at API boundaries |
| Remove `TyVarBound::meet` | Two bounded variables unify without checking compatibility → bind to a type neither admits |
| Remove occurs check from variable binding | `T = Array<T>` creates an infinite type → resolver loops forever |
| Let `try_coerce_infer` accept multiple matches | Ambiguous coercion silently picks one → non-deterministic type resolution |
| Resolve types within SCC before all functions type-checked | Mutual recursion: early resolution freezes types before constraints from later functions arrive |
| Emit a Pure call with an `OrderEdge` (or an effectful one without) | Rejected by type check (`OrderEdge` error) — the invariant the passes rely on |

---

## What We Don't Guarantee

- **No speculative issue.** IO inside a branch is issued in that branch; code motion never hoists `Spawn`, and only Commute's post-dominance rule moves a call across blocks.
- **Plugin effects are not verified.** If a plugin declares pure but performs IO, the system cannot catch this.
- **Integer overflow, out-of-bounds access, and division by zero are not caught at compile time.**
- **Termination is not guaranteed.** Recursive functions and infinite loops are permitted.
- **`Undef` instructions from SSA are not eliminated.** They are placeholders: valid to move or copy, undefined to read as a concrete value.
