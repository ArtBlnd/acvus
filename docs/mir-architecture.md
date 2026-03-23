# acvus-mir Architecture

## Overview

MIR (Mid-level Intermediate Representation) is the type system and compilation backbone.
It takes AST input and produces typed, optimized IR modules ready for interpretation.

```
AST (acvus-ast)
  |
  v
TypeChecker (typeck.rs)  ──→  TypeResolution<Unchecked>
  |                                    |
  |                          check_completeness()
  |                                    |
  v                                    v
Lowerer (lower.rs)  ←──  TypeResolution<Checked>
  |
  v
MirModule (ir.rs)
  |
  ├── Optimizer (optimize/)
  ├── Validator (validate/)
  └── Analysis (analysis/)
```

## Modules

### Core Pipeline

#### `ty.rs` — Type System
- `Ty` enum: Int, Float, String, Bool, List, Deque, Sequence, Iterator, Object, Fn, ...
- `TySubst`: unification engine (substitution-based)
  - `unify(a, b, Polarity)`: Invariant | Covariant | Contravariant
  - `try_coerce(sub, sup)`: subtype coercion (Deque <= List <= Iterator, Deque <= Sequence <= Iterator)
  - `resolve(ty)`: walk binding chain to concrete type
- `Effect`: Pure | Effectful | Var
- `Origin`: value provenance tracking

#### `typeck.rs` — Type Checker

Input: AST + ContextTypeRegistry + optional hint + TySubst.
Output: `TypeResolution<Unchecked>` → `check_completeness()` → `TypeResolution<Checked>`.

**Phantom state markers:**
- `TypeResolution<Unchecked>`: vars may be unresolved (SCC not yet complete)
- `TypeResolution<Checked>`: all vars resolved, safe for lowering

**Resolution contents:**
- `type_map: TypeMap` — Span → Ty for every expression
- `builtin_map: BuiltinMap` — Span → BuiltinId for resolved builtins
- `coercion_map: CoercionMap` — Span → CastKind for implicit coercions
- `tail_ty: Ty` — the result type of the entire script/template

**`check_completeness(unchecked, subst) → Checked`:**
- Re-resolves entire type_map with final subst state
- Rejects any remaining `Ty::Var` (AmbiguousType error)
- Required because SCC peer units may resolve vars AFTER this unit's typecheck

**Shared TySubst:** multiple units typecheck in the same subst (for SCC).
Type information flows between units via shared vars.

**`unify_covariant(value_ty, expected_ty, span)`:**
- Single entry point for covariant unification in typechecker
- Calls `subst.unify(val, exp, Covariant)`
- On success: checks if coercion needed → records in coercion_map
- `CastKind::between(resolved_val, resolved_exp)` determines if IR cast instruction needed

**Analysis mode:** `analysis_mode = true` → unknown `@context` refs produce fresh vars
instead of errors. Allows partial type inference for LSP/incremental.

#### `ty.rs` — Type System + Unification

**Polarity-based subtyping (non-standard HM extension):**

Standard Hindley-Milner has no subtyping. We extend it with polarity:

```
Polarity::Invariant    — a = b (standard HM unification)
Polarity::Covariant    — a ≤ b (a is subtype, coercion allowed)
Polarity::Contravariant — b ≤ a (reversed, for function params)
```

**Where each polarity is used:**

| Context | Polarity | Reason |
|---------|----------|--------|
| Generic inner types (List\<T\>, Deque\<T\>) | Invariant | `List<Int> ≠ List<Float>` — no variance |
| Function arguments at call site | Covariant | `f(deque)` where `f: (List) → ...` — deque ≤ list |
| Function param types in signature | Contravariant | `(List) → T` ≤ `(Deque) → T` — contravariant input |
| Function return types | Covariant | `... → Deque` ≤ `... → List` — covariant output |
| Hint unification (expected_tail) | Covariant | actual ≤ expected |

**Coercion lattice (try_coerce):**

```
         Iterator<T, E>
        /              \
  List<T>          Sequence<T, O, E>
        \              /
         Deque<T, O>
```

- `Deque ≤ List` (drop origin)
- `Deque ≤ Sequence` (origin preserved, effect = Pure)
- `Deque ≤ Iterator` (drop origin, effect = Pure)
- `List ≤ Iterator` (effect = Pure)
- `Sequence ≤ Iterator` (drop origin, effect preserved)

**All coercion rules:** inner types unified **invariant**. Only the outer
constructor changes. This prevents `List<Int>` from coercing to `Iterator<String>`.

**Effect lattice:** `Pure ≤ Effectful`. Effect variance follows polarity.

**Origin:** value provenance. `Origin::Var` = unresolved, `Origin::Concrete(n)` = resolved.
Used to track which Deque/Sequence came from where (for storage tracking).

**`lub_or_err` (least upper bound):**
When same-constructor types have mismatched effects/origins:
- Invariant polarity → type error (no silent coercion)
- Non-invariant → compute LUB, rebind leaf var to LUB

**Key design decision: generics are invariant, only outer constructors coerce.**
This is a deliberate trade-off:
- Simpler than full covariant generics (no variance annotations)
- Sound: `List<Int>` doesn't implicitly become `Iterator<String>`
- Coercion only at constructor boundary (List→Iterator), not at element level

#### `lower.rs` — MIR Lowerer
- Input: AST + TypeResolution<Checked> + name_to_id mapping
- Output: MirModule (instructions + closures + metadata)
- Converts AST nodes → IR instructions using type info from resolution
- `ContextLoad { id: Id }`: emits Id-based context references (not name-based)
- `ExternCall { id: Id }`: emits Id-based extern function calls

#### `ir.rs` — IR Definition
- `MirModule`: main body + closures + debug info
- `MirBody`: instructions + value types + labels
- `InstKind`: Yield, ContextLoad, ExternCall, BuiltinCall, MakeDeque, MakeObject, ...
- All context references use `Id` (not Astr)

### Graph Engine (`graph/`)

Domain-free compilation graph. Knows nothing about orchestration concepts
(body, bind, assert, strategy, persistency, etc.).

#### `graph/types.rs` — Graph Types
- `Id`: unified identifier (replaces ContextId + UnitId)
- `Unit`: unified compilation unit
  - `body: Option<UnitBody>` — Some = compilable, None = extern (runtime provides)
  - `inputs: Vec<(Id, Ty)>` — type constraints on consumed units
  - `output_ty: Option<Ty>` — declared output (None = inferred)
  - `output_binding: Option<Id>` — ScopeLocal binding
- `Scope`: groups units, declares bindings (ScopeLocal, Derived)
- `CompilationGraph`: units + scopes + externals + id_table

#### `graph/resolve.rs` — Graph Resolution Engine
- `CompilationGraph::resolve()` → `ResolvedGraph` (Phase 0 → Phase 1)
- `CompilationGraph::compile()` → `CompiledGraph` (Phase 0 → Phase 2)
- Internally:
  1. Build dependency graph from scope bindings (Derived = DAG edge, ScopeLocal = SCC)
  2. Tarjan SCC detection
  3. Topo-order processing: per-SCC typecheck with shared TySubst
  4. check_completeness per unit
  5. Output: per-unit TypeResolution + resolved types + unit outputs

Key invariants:
- SCC membership: only units that participate in ScopeLocal (reference or output_binding)
- ScopeLocal allocation: only when participant units are in current SCC
- unit_output after SCC: uses tail_ty (not pre-registered var) to prevent coercion leak
- ExternDecl output: uses declared output_ty (not var)

### Analysis (`analysis/`)

All analyses run on compiled MirModule. Pure functions, no mutation.

#### `analysis/domain.rs` — Abstract Value Domain

Semilattice-based abstract domain for dataflow analysis.

**`SemiLattice` trait:**
- `bottom()` → initial state (no information)
- `join_mut(&mut self, other)` → least upper bound. Returns true if changed.

**`AbstractValue` — 3-level lattice:**
```
Top (unknown — all values possible)
 |
Finite(FiniteSet) — known set of possible values
 |
Bottom (unreachable — no values)
```

**`FiniteSet` variants:**
- `Intervals(SmallVec<[Interval; 4]>)` — integer ranges, e.g., {[1,3], [7,7]}
- `Bools(SmallVec<[bool; 2]>)` — {true}, {false}, {true, false}
- `Strings(SmallVec<[Astr; 4]>)` — known string values
- `Variants(SmallVec<[(Astr, Box<AbstractValue>); 4]>)` — enum variant + payload
- `Literals(SmallVec<[Literal; 4]>)` — mixed literal set
- `Tuple(Vec<AbstractValue>)` — per-element abstract value

**Widening:** set exceeds `MAX_SET_SIZE` (16) → graduate to Top.
Integer intervals: merge closest pairs until within limit (graduated widening).

**`BooleanDomain`:** `as_definite_bool() -> Option<bool>` — used for branch pruning.
If abstract value is exactly {true} or {false}, the branch is determined.

#### `analysis/cfg.rs` — Control Flow Graph
- `Cfg::build(body)` → basic blocks + edges
- `BasicBlock`: instruction range, terminator (Jump/Branch/Return/Unreachable)
- Predecessors/successors computed at build time
- Foundation for all dataflow analyses

#### `analysis/dataflow.rs` — Generic Dataflow Framework
- `forward_analysis<D: SemiLattice>(cfg, transfer_fn) → Vec<D>` — per-block state
- Worklist algorithm with semilattice join at merge points
- Converges when no block state changes (monotonicity guaranteed by semilattice)
- `BooleanDomain` trait for branch pruning integration

#### `analysis/value_transfer.rs` — Value Transfer Functions
- Per-instruction abstract interpretation: how does each instruction transform abstract state?
- `ContextLoad { id }` → abstract value for context variable (from known_context map)
- `Literal(v)` → `AbstractValue::from_literal(v)` (point value in domain)
- `BinOp/UnaryOp` → abstract arithmetic (interval arithmetic for Int, etc.)
- `VarLoad/VarStore` → variable tracking in abstract state

#### `analysis/reachable_context.rs` — Context Key Reachability
- **Purpose:** given known values, which context keys are actually needed at runtime?
- **Dead branch pruning:** if a branch condition is determined (AbstractValue = definite Bool),
  context keys in the dead branch are excluded from the "needed" set.
- `reachable_context_keys(module, known, val_def) → FxHashSet<Id>` — all needed keys
- `partition_context_keys(module, known, val_def) → { eager, lazy }` — eagerly vs conditionally needed
- Used by Resolver for dependency scheduling: eager deps are spawned immediately,
  lazy deps are spawned only when actually requested via NeedContext.

#### `analysis/val_def.rs` — Value Definition Map
- `ValDefMapAnalysis.run(module) → ValDefMap`
- Maps each ValueId → the instruction index that defined it
- Used by reachable_context for tracking which values are context-dependent

#### `analysis/var_dirty.rs` — Variable Dirtiness
- Tracks which `$variables` are modified on which control flow paths
- Forward dataflow: `VarStore` marks dirty, merge = union
- Used for detecting side effects in match arms, iterations

### Validation (`validate/`)

Post-lowering verification. Catches lowerer bugs and ensures IR invariants.
Entry point: `validate(module) -> Vec<ValidationError>`.

#### `validate/type_check.rs` — Type Consistency Verification

Walks every instruction, checks val_types consistency.

**Invariants checked:**
- Every ValueId has a type entry in val_types
- Operand types match instruction expectations (BinOp, UnaryOp, etc.)
- Constructor types match (MakeDeque elements, MakeObject fields, etc.)
- Function call arity matches signature
- Cast is the ONLY instruction allowed to change a value's type

**Type matching rules (invariant variance):**
- `Ty::Error` / `Ty::Var` / `Ty::Infer` → match anything (poison/unresolved escape)
- Primitives: exact match
- Containers: recursive invariant match on inner types
- Origins: `Origin::Var` matches anything, concrete must be equal
- Effects: `Effect::Var` matches anything, Pure ≤ Effectful (subtype)
- Enums: same name is sufficient (variants unified elsewhere)

#### `validate/move_check.rs` — Move Semantics (Linear Ownership)

Forward dataflow analysis over CFG. Detects use-after-move.

**Move-only types:**
- `Iterator<T, Effectful>` — effectful iterator is consumed on use
- `Sequence<T, O, Effectful>` — effectful sequence
- `Opaque` — unknown internals, always move-only
- Containers with move-only elements — transitive
- `Fn` with move-only captures → FnOnce (transitive)

**Copyable types:**
- All primitives (Int, Float, String, Bool, Unit, Range, Byte)
- `Iterator<T, Pure>`, `Sequence<T, O, Pure>`
- `List<T>`, `Deque<T, O>`, `Option<T>` (if inner is copyable)

**Dataflow:**
- Domain: per-ValueId state = Alive | Moved(inst_index)
- Join: `Alive ⊔ Moved = Moved` (conservative — if moved on any path, treat as moved)
- `VarStore` revives a variable (re-assignable)
- `VarLoad` of move-only type consumes it
- `Ty::Error` / `Ty::Var` / `Effect::Var` → skip (analysis mode)

**Key design: `is_move_only(ty) -> Option<bool>`**
- `Some(true)` → move-only, track usage
- `Some(false)` → copyable, no tracking needed
- `None` → unresolved (analysis mode), skip entirely

### Other

#### `builtins.rs` — Builtin Function Registry
- Type signatures for all builtins (map, filter, chain, append, etc.)
- Coercion lattice: Deque <= Sequence <= Iterator

#### `context_registry.rs` — Context Type Registry
- `ContextTypeRegistry`: maps context names → types (for TypeChecker)
- `PartialContextTypeRegistry`: pre-compilation registry (system, user, extern tiers)
- Being simplified: graph engine builds registries internally

#### `pass.rs` — Analysis Pass Trait
- `AnalysisPass::run(module, input) → output`
- Uniform interface for all analyses

#### `hints.rs` — Hint Table
- Stores type hints for LSP (hover, completion)

#### `printer.rs` — IR Pretty Printer
- Debug output for MirModule

#### `ser_ty.rs` — Type Serialization
- Ty → serde-compatible format (for WASM/frontend)

## Dependencies

```
acvus-ast (parsing)
  ↓
acvus-mir
  ├── ty.rs (no deps)
  ├── typeck.rs (depends: ty, context_registry, error)
  ├── lower.rs (depends: ty, typeck, ir, graph::Id)
  ├── ir.rs (depends: graph::Id)
  ├── graph/ (depends: ty, typeck, lower, context_registry)
  ├── analysis/ (depends: ir, ty)
  ├── validate/ (depends: ir, ty)
  ├── builtins.rs (depends: ty)
  └── context_registry.rs (depends: ty)
```

## Key Design Decisions

1. **Unified Id**: single ID space for all entities (units, context vars, scope locals).
   No separate ContextId/UnitId. Eliminates Derived bindings complexity.

2. **Unified Unit**: CompilationUnit and ExternDecl merged into `Unit`.
   `body: Some` = compilable, `body: None` = extern. Same type constraints system.

3. **Domain-free graph engine**: knows nothing about orchestration concepts.
   SCC/DAG structure derived purely from Scope bindings and Id references.

4. **Coercion safety**: unit outputs use `checked.tail_ty` (not pre-registered var)
   to prevent coercion leak across SCC units.

5. **ScopeLocal allocation**: only when participant units are in the current SCC.
   Prevents premature allocation in unrelated SCCs sharing the same scope.
