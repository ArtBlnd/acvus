# acvus MIR — Compiler Engineer's Guide

This document is for compiler engineers modifying the `acvus-mir` codebase. It does not describe what exists — it describes why things are the way they are, and what breaks if you change them.

---

## SSA: Block Parameters vs PHI Nodes

We use Cranelift-style block parameters, not LLVM-style PHI nodes.

```
Jump { label: L0, args: [v1, v2] }
BlockLabel { label: L0, params: [v3, v4] }
```

Block params are definitions (defs), jump args are uses. Consequences of this choice:

- **Inlining is simple.** Remap args and you're done. No PHI surgery required.
- **Terminator args are uses.** This affects liveness, dataflow, and reg_color. Args live inside terminators that are not in the instruction array, so flat instruction indexing misses them. This is why liveness analysis treats terminator position as `block_end + 1`.
- **SSABuilder has a seal ordering requirement.** `seal_block()` may only be called after all predecessors of that block have been added. Calling it earlier resolves PHIs against an incomplete predecessor set, silently losing values.

### ENTRY_BLOCK Sentinel

`ENTRY_BLOCK = Label(u32::MAX)` is a sentinel identifying the implicit entry block. Using `u32::MAX` as an actual label will collide. Labels are allocated via `label_count`, so collision is unrealistic in practice, but not formally guaranteed.

### Loop Back-Edge Cycle Breaking

In `use_var_sealed()`, the PHI value is pre-defined in `current_defs` before resolving it. Without this, a loop back-edge causes: PHI resolve → same PHI use → same PHI resolve → infinite recursion. The pre-definition breaks the cycle.

### Trivial PHI Elimination

If all incoming edges provide the same value (excluding the PHI itself), the PHI is replaced with that value. The self-exclusion filter is critical — a loop back-edge referencing the PHI itself does not count as "all incoming values are identical".

---

## SSA Pass: Context vs Local Variable Asymmetry

The most important design decision: **contexts require write-back, local variables do not.**

Local variables (`VarLoad`/`VarStore`) disappear entirely after SSA. Values flow through block params — nothing else.

Contexts (`ContextProject`/`ContextLoad`/`ContextStore`) are different. Contexts are external state, so stores inside branches must persist after merge. The SSA pass removes branch-internal `ContextStore`s and inserts a single write-back `ContextStore` after each merge block.

### ContextProject/ContextLoad Always Come in Pairs

The lowerer always emits `ContextProject` immediately followed by `ContextLoad`. The SSA pass's dead load elimination removes these pairs using **index-1 arithmetic** — if a `ContextLoad` is dead, the immediately preceding `ContextProject` is removed with it.

**When this breaks:** If someone inserts an instruction between `ContextProject` and `ContextLoad`, the index-1 removal deletes the wrong instruction. The lowerer currently guarantees adjacency, but this invariant is not enforced in code.

### Undef Initialization for Write-Only Contexts

When a context has `ContextStore` but no preceding `ContextLoad` (write-only), the SSA pass splices an `Undef` instruction at position 0. This splice happens after CFG construction but before other optimizations. If the splice timing changes, instruction indices drift and the def_map breaks.

### Chained Substitution

The SSA pass chains two substitution maps: `var_subst` (from SSABuilder) + `fwd_subst` (from forward context values). If `var_subst` maps r10→r5 and `fwd_subst` maps r5→r3, the final result is r10→r3. Reversing the application order produces incomplete forwarding.

---

## Inliner: Name-Based Parameter Binding

The inliner binds parameters **by name**, not by position.

`$x + $x` produces two `ParamLoad { name: "x" }` instructions. The inliner registers the first occurrence in `param_name_to_arg` and reuses the cached arg for subsequent occurrences of the same name.

**Why positional binding breaks:** If the same parameter is used twice, the second occurrence consumes the next arg instead of reusing the first.

### val_remap Chain

When sequentially inlining multiple calls, each inline's result dst is added to `val_remap`. The next inline's args are remapped through the current `val_remap`. In `double(inc(3))`, if inc's result is remapped r5→r10, then double's arg must use r10 instead of r5.

**Ordering matters:** Args must be remapped before inlining. After inlining, new ValueIds are mixed in and remap becomes unreliable.

### Label Offset Accumulation

When inlining, all callee labels are offset by `label_offset = current.label_count`, then `current.label_count += callee.label_count`. This relies on labels being allocated linearly and never reused.

### Devirtualization Conditions

To devirtualize an `Indirect(v)` callee:
1. `v`'s definition must be exactly one `MakeClosure` (traced through def_map)
2. It must not pass through a PHI/block param (which would mean multiple possible definitions)

Devirtualizing through a PHI is unsound — at runtime, the closure could be a different one.

---

## Dataflow: Forward and Backward Are Not Simple Inverses

### Forward: propagate_to_successor

Maps jump args from source block's exit state to target block's params. Additionally joins the entire exit state into the target entry (non-param values flow through).

### Backward: propagate_from_successor

If successor params are live at successor's entry, marks the corresponding terminator args as live at this block's exit. **Additionally** joins non-param values from successor's entry into this block's exit.

**Why the asymmetry:** In backward analysis, values live at a successor's entry that bypass params (= values live across block boundaries without going through block params) must also be live at this block's exit. In forward analysis, such values propagate naturally through the join. In backward, this flow-through must be explicit.

### JumpIf Cond Is Marked Live in Backward

The JumpIf terminator itself uses the cond value. In backward analysis, cond is set to `D::top()` in the block's exit state. Forward analysis doesn't need this — forward tracks "what is true at this point", not "what is used".

### Return(ValueId)

`Terminator::Return(val)` sets val to `D::top()`. Previously, `Terminator::Return` did not carry a ValueId, so the return value was not marked live in backward analysis — this was a bug and has been fixed.

### ListStep's Dual Role

`ListStep` appears in the instruction array AND becomes a terminator. All other terminators (Jump, JumpIf, Return) appear in the instruction array but are excluded from `inst_indices`. Only ListStep is **included** in `inst_indices`.

This asymmetry exists because ListStep defines dst/index_dst — regular terminators don't define values.

---

## CFG: Block Boundary Rules

The CFG is built from a flat instruction stream. Rules:

1. `BlockLabel` → starts a new block (flushes the previous one)
2. `Jump`/`JumpIf`/`Return` → ends a block (sets terminator, instruction is in inst_indices)
3. `ListStep` → ends a block + included in inst_indices + two successors (fallthrough + done)

**Fallthrough successor is computed as `idx.0 + 1`.** This assumes the block array is sequential. Reordering blocks breaks fallthrough.

**Entry block is always BlockIdx(0).** Instructions before the first `BlockLabel` form the implicit entry block.

---

## Reorder: Three Independent Dependency Chains

The reorder pass builds three dependency chains simultaneously within each basic block:

1. **SSA use-def** — A use must come after the instruction that defines its ValueId
2. **Token ordering** — Instructions sharing the same TokenId must preserve original order
3. **ContextStore ordering** — Stores to the same context must preserve original order

If the three chains conflict, a cycle can occur. The current implementation catches cycles with an assert but has no graceful recovery.

### ContextStore → ContextProject Back-Trace

A ContextStore's dst is the result ValueId of a ContextProject. The reorder pass traces this ValueId through def_map to determine which context the store targets. **The ContextProject must be in the same block** — if it's in a different block, def_map lookup fails and ordering is lost.

### Priority-Based Topological Sort

Uses a `BinaryHeap` (min-heap via Reverse). Priority: Spawn < Normal(original_index) < Eval. Spawns are scheduled earliest, Evals latest. Normal preserves original order.

**Ordering between instructions of the same priority depends on heap implementation.** In most cases, original_index acts as a tiebreaker, but among Spawns or among Evals there is no tiebreak. Currently this has no semantic impact, but could matter for future optimizations.

---

## Spawn Split: Preconditions

Spawn split must run after the SSA pass. Reason: `FunctionCall`'s `context_uses`/`context_defs` are populated by the SSA pass. Splitting before SSA produces Spawn/Eval with empty context fields, severing context flow.

### Handle Type Registration

On split, a Handle type is registered in `val_types`. The callee's return type and effect are extracted from `fn_types` to construct `Ty::Handle(ret, effect)`. **If the callee is missing from fn_types, the Handle type is not registered**, and downstream type checking fails.

### Only IO Is Split

`is_io_call()` checks only the `io` flag in the effect. Token-only effects are not split. This is a deliberate decision: Token functions must execute sequentially, so there's no benefit in splitting them for reorder.

---

## Typeck: param_types Ordering Matters

`param_types: SmallVec<[(Astr, Ty); 4]>` preserves insertion order. It was previously an `FxHashMap`, but HashMap's non-deterministic iteration order caused extern function parameter ordering to break.

**Why this matters:** `extern_params` iterates `param_types` in order to construct `Signature.params`. The caller places args in this order. If the order drifts, arg-param binding becomes incorrect.

### Lambda Effects Do Not Propagate Upward

When an effectful function is called inside a lambda, the effect is recorded in the lambda's effect — not the enclosing function's effect. The lambda scope freezes the effect on pop. This is deliberate: a lambda produces effects at call site, not at definition site. The enclosing function only needs to know "calling this lambda has effects".

---

## Move Check: VarStore Revives Values

After a move-only value is moved, storing a new value to the same variable (`VarStore`) brings it back to `Alive`.

```
a = move_only_value;  // a: Alive
consume(a);           // a: Moved
a = new_value;        // a: Alive again
use(a);               // OK
```

**`VarLoad`/`ParamLoad` do not revive.** A load is a read (and consumption for move-only types), not a new definition.

### Conservative Join at Branch Merge

If one branch has `Alive` and the other has `Moved(at: 3)`, the merge result is `Moved(at: 3)`. This is conservative — if any path could have moved the value, it's treated as moved.

---

## Reachable Context: Branch Pruning with Known Values

`partition_context_keys` classifies context loads as eager/lazy/pruned. The key mechanism: known context values (provided by the caller) are used to evaluate `TestLiteral` conditions and prune dead branches.

**Pruned contexts still require type injection.** Code in dead branches has already passed type checking, so missing type information causes the type checker to panic. The runtime doesn't need to resolve these values, but types must still be injected.

---

## Register Coloring: Why Flat Scan Breaks

The previous implementation scanned instructions linearly to compute def_pos/last_use. This breaks in multi-block programs:

1. **Cross-block liveness** — Value defined in Block A, used in Block B. The flat scan doesn't know about the control flow path between them. If it decides the value dies at the end of Block A, another value may take the slot before Block B uses it.
2. **Loop back-edges** — A value live across the entire loop, but the flat scan only sees the last use within the loop body. It misses liveness across loop iterations.
3. **Terminator args** — Jump/JumpIf args are in the instruction array but separately handled in `inst_indices`. The flat scan can miss these uses.

The current implementation uses backward dataflow liveness analysis, which correctly handles all three cases.

---

## "Change This, Break That"

| Change | Consequence |
|--------|-------------|
| Revert `param_types` to HashMap | Extern function parameter order becomes non-deterministic → arg-param binding errors |
| Insert instruction between `ContextProject`/`ContextLoad` | SSA pass's index-1 dead load elimination removes the wrong instruction |
| Remove pre-definition in SSABuilder | Infinite recursion on loop back-edge PHI resolution |
| Run spawn split before SSA | context_uses/context_defs are empty → context flow severed |
| Remove ValueId from Terminator::Return | Return value not marked live in backward analysis → reg_color reuses the return slot |
| Remove `SemiLattice::top()` | Cannot mark JumpIf cond and Return val as live in backward analysis |
| Remove VarStore revive logic | Reassignment after move is falsely flagged as use-after-move |
| Allow cross-block ContextProject in reorder back-trace | def_map miss → ordering lost → stores to same context reordered → unsound |
| Propagate lambda effects to enclosing scope | Lambda definition alone marks outer function as effectful → unnecessary inline restrictions |
| Remove ListStep from inst_indices | ListStep's dst/index_dst definitions missing from liveness/defs |
| Remove non-param flow-through in backward propagation | Cross-block live values disappear from exit state → reg_color reuses their slots → value corruption |
| Reorder block array | Fallthrough successor idx+1 calculation breaks → wrong successor visited |
| Use `ENTRY_BLOCK` sentinel as a real label | SSA builder confuses entry block with regular block |
| Remove self-reference filter from trivial PHI elimination | Loop back-edge self-reference triggers "all incoming identical" → PHI incorrectly eliminated |

---

## What We Don't Guarantee

- **ListStep only works on lists.** It is not a general-purpose iterator step. UserDefined iterators require a separate mechanism.
- **No cross-block reordering.** Reordering happens within basic blocks only.
- **Spawn split does not handle conditional IO.** IO inside branches is not split (no speculative execution).
- **Register coloring is linear scan, not graph coloring.** May not be optimal, but with virtual registers there's no spilling, so it's sufficient.
- **Plugin effects are not verified.** If a plugin declares pure but performs IO, the system cannot catch this.
- **Integer overflow, out-of-bounds access, and division by zero are not caught at compile time.**
- **Termination is not guaranteed.** Recursive functions and infinite loops are permitted.
