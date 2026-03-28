# acvus MIR Architecture

## What is this?

acvus is not a general-purpose programming language. It is a **script and template language** — the kind you embed in a larger system to let users write business logic, data transformations, and LLM orchestration without touching the host codebase.

The difference is that acvus takes this role seriously. Instead of a loosely-typed interpreter with string-based extension points (like most embedded scripting), it has a full compiler backend with type inference, effect tracking, and SSA-based IR. The goal is simple: **scripts and templates should be safe, fast, and pleasant to write** — with the same rigor that systems languages apply to systems code.

A single expression like `@users | filter(active) | map(name) | join(", ")` compiles to the same typed, optimized IR whether it appears inside `{{ }}` template interpolation or as a standalone script. The MIR knows not just *what* your code computes, but *which external state it reads and writes*, enabling automatic parallelization, incremental recompilation, and portable execution.

---

## Design Principles

**One IR, multiple surfaces.** Template and script share the same type system, the same compiler pipeline, the same IR. No special-casing.

**Effects are first-class.** Every function's effect footprint (which contexts it reads, writes, whether it does IO) is tracked at the type level. This isn't an annotation — it's inferred from code and propagated transitively through call chains.

**No opaque IDs where names suffice.** Contexts are identified by qualified names (`@namespace:name`), not opaque integer IDs. This makes the IR deterministic, serializable, and human-readable without a symbol table.

**Plugin-extensible type system.** External functions register their types alongside their handlers. Adding an ext function expands the set of valid programs — the type system grows with the ecosystem.

**Use first, define later.** Most things in acvus are inferred from usage, not declared upfront. Write `@data | map(f) | collect` and the compiler works *backwards* — `@data` must be iterable, `f` must return something, the result is a list of that something. Context types, function parameter types, effect footprints, even generic constraints are all discovered by analyzing how values are used, then propagated outward to the environment. The host system provides concrete types for contexts; the compiler checks that they satisfy the constraints the code imposed. This inverts the traditional "define type, then use" flow — users write code freely, and the system figures out what the environment must provide.

---

## The Pipeline and Why It's Split This Way

```
source -> extract -> infer -> lower -> optimize -> MirModule
```

The pipeline is split into four phases not because it's architecturally elegant, but because **the LSP needs to cut into the middle**. When a user edits a script, the system re-runs only the phases invalidated — not the entire compilation.

**Extract** parses source and discovers context references. This is cached by source hash — same source, skip re-parsing.

**Infer** resolves all types via constraint propagation. Inter-function inference uses Tarjan's SCC algorithm so mutually recursive functions are solved simultaneously. Infer results are cached per-SCC with **early cutoff** — if a function's type didn't change after re-inference, its callers don't need re-inference. Most edits touch one SCC, so recompilation is O(changed) not O(total).

**Lower** translates typed AST to MIR instructions. Context mutations (`@x = ...`) become `ContextProject`/`ContextStore` pairs — not yet SSA. This is deliberate: lowering produces a "pre-SSA" IR that's easy to generate from AST, and the SSA pass in optimize handles the hard part (PHI insertion, dead store elimination).

**Optimize** is the final phase. Two passes:
- **Pass 1** (cross-module): SSA promotion per module, then inlining across modules. Inline must see all modules because it resolves cross-function calls and devirtualizes closures.
- **Pass 2** (per-module): SpawnSplit → Reorder → SSA → RegColor → Validate. This is a `PassManager` pipeline with typed dependencies — each pass declares what it needs, and topological sort ensures correct ordering.

### Why not separate "resolve" and "lower"?

Early designs had a separate resolve phase that finalized all types. This was removed — infer now produces fully resolved types directly. The separate phase added complexity without buying anything: there was no meaningful cache boundary between "almost resolved" and "fully resolved". If inference succeeds, types are complete.

### Why validate at the end, not after lower?

Validation runs after all optimizations, not after lowering. The reasoning: optimizations (inline, reorder, SSA) transform the IR in ways that could theoretically break invariants. Validating the *final* IR catches bugs in the optimizer itself. If we validated early and skipped post-optimization validation, an optimizer bug could produce unsound IR silently.

---

## Effect System: Why Two Kinds of Effect Target

Every function tracks what it reads, writes, whether it does IO, and whether it's self-modifying. The key design decision is **two kinds of effect targets**:

**Context (`@name`)** — SSA-compatible. The compiler converts context reads/writes into SSA value flow (`context_uses`/`context_defs` on function calls). After SSA, there's no mutable state — just values flowing through block parameters. This means context ordering is handled entirely by SSA data dependencies. No special ordering logic needed.

**Token (`TokenId`)** — NOT SSA-compatible. Tokens represent external shared state (database connections, file handles) that can't be converted to value flow. Functions sharing the same Token must execute sequentially. The reorder pass preserves Token ordering explicitly.

### Why this split matters

Without the Context/Token distinction, we'd have two bad options:
1. Treat everything as Token → no automatic parallelization of context-using functions
2. Treat everything as SSA-compatible → unsound for external shared state

The split gives us the best of both: contexts (the common case in templates/scripts) get full SSA treatment and automatic parallelization, while tokens (rare, for external IO) get correct sequential ordering.

### What we guarantee

- If a function is marked pure, it truly has no side effects.
- Effects propagate transitively — if `f` calls `g` which reads `@users`, then `f` reads `@users`.
- Functions sharing a Token are never reordered relative to each other.

### What we don't guarantee

- **Optimal parallelism.** The reorder pass uses a greedy topological sort with priority heuristics (spawn early, eval late). It doesn't solve for globally optimal scheduling — that would require solving an NP-hard problem for marginal gain in typical template/script workloads.
- **Effect inference across plugin boundaries.** If a plugin function lies about its effects (claims pure but does IO), the system has no way to catch this. Plugin authors must be honest. This is a conscious trade-off: verifying plugin effects would require sandboxing or formal verification, neither of which is practical for an embedded scripting language.

---

## Automatic IO Parallelization

This is the centerpiece optimization. The insight: in template/script workloads, the bottleneck is almost always IO (API calls, database queries), not computation. If we can automatically parallelize independent IO operations, users get massive speedups without changing their code.

### How it works

1. **Spawn split** — IO function calls are split into `Spawn` (pure, creates a `Handle`) + `Eval` (effectful, forces the Handle). This is the key insight: scheduling work is pure, only forcing the result is effectful.

2. **Reorder** — Within each basic block, instructions are topologically sorted by dependencies (SSA use-def + Token ordering). Priority: Spawn = schedule earliest, Eval = schedule latest, everything else = original order. This naturally clusters Spawns at the top and defers Evals.

3. **SSA re-run** — After reorder, SSA eliminates dead code introduced by splitting and reordering.

### Spawn/Eval effect rules

| Situation | Spawn | Eval |
|---|---|---|
| Token present | Token-effectful | Token-effectful |
| No token (Context only) | Pure | Context-effectful |
| IO only | Pure | IO-effectful |

### Why split into two passes (spawn_split + reorder)?

We considered a single pass that does both. Splitting is better because:
- **spawn_split is trivial** — scan for IO calls, replace in-place. No analysis needed.
- **reorder is complex** — builds dependency graph, topological sort with priorities. Keeping it separate makes it testable independently.
- **Composability** — reorder works on any instruction sequence, not just spawn/eval pairs. Future optimizations (computation reordering, loop-invariant code motion) can reuse it.

### What we don't do

- **Cross-block reordering.** Reorder works within basic blocks only. Moving instructions across branches requires more sophisticated analysis (code motion) that isn't justified for the common case.
- **Speculative execution.** We don't spawn IO that might not be needed (e.g., inside an if-branch). Only unconditionally-reached IO calls are split. This is conservative but safe.
- **CPU-heavy parallelization.** The `cpu_heavy` effect is planned but not implemented. IO parallelism covers 95%+ of the value for template/script workloads.

---

## SSA: Why Cranelift-Style, Not LLVM-Style

The SSA pass uses a Cranelift-style `SSABuilder` rather than LLVM's dominance-frontier-based approach. The reason: **block parameters instead of PHI nodes**.

In LLVM's model, PHI nodes are special instructions at block entry that select values based on which predecessor you came from. This requires knowing predecessors at PHI creation time and makes instruction rewriting fragile.

In Cranelift's model, blocks take parameters (like function parameters), and jumps pass arguments. `Jump { label: L0, args: [v1, v2] }` → `BlockLabel { label: L0, params: [v3, v4] }`. This is:
- **Simpler to construct** — no need to track predecessors during construction
- **Simpler to transform** — inlining just remaps args, no PHI surgery
- **Naturally SSA** — block params are definitions, jump args are uses, standard use-def chain

### Write-back model for contexts

Context mutations inside branches pose a problem: after a branch merges, which value does `@x` have? The SSA pass handles this with a write-back model:
- Branch-internal `ContextStore`s are removed (the stores are converted to SSA value flow)
- A single write-back `ContextStore` is inserted after the merge block
- Local variables (`VarLoad`/`VarStore`) don't need write-back — they exist only in SSA form

### What this means for the optimizer

After SSA, the IR is genuinely functional — all "mutation" is expressed as new values flowing through block parameters. This makes analysis and transformation straightforward: liveness, register allocation, reordering all operate on standard SSA use-def chains with no aliasing concerns.

---

## Inlining: Why Devirtualize Before Inline

The inliner runs devirtualization as part of its pass: if an `Indirect` callee traces back to a single `MakeClosure` (not through a PHI/block parameter), the closure body is inlined directly with captures prepended to args.

### Why not a separate devirt pass?

Devirtualization and inlining share the same machinery — both need to trace value definitions, remap ValueIds, and splice instruction sequences. A separate pass would duplicate this work. More importantly, devirt decisions depend on the inline context: a closure might be worth devirtualizing only if the call site is also being inlined.

### What we don't devirtualize

- **PHI-sourced closures.** If a closure value comes from a block parameter (meaning it could be one of multiple closures depending on which branch was taken), we leave it as an indirect call. Sound devirt would require splitting the call site, which isn't justified for the common case.
- **Recursive closures.** Detected via SCC analysis. Inlining recursive calls would diverge.

---

## Register Coloring: Why Liveness-Based

Register coloring compacts ValueId allocation by reusing slots for values with non-overlapping lifetimes. The key decision: **CFG-aware liveness analysis** rather than flat linear scan.

### Why CFG-aware?

The original implementation used flat instruction-order liveness (scan instructions linearly, track first def and last use). This broke for multi-block programs: a value defined in block A and used in block B had its interval underestimated because the flat scan didn't account for the control flow path between blocks.

The fix: backward dataflow liveness analysis over the CFG. This correctly handles:
- **Cross-block liveness** — values live across block boundaries
- **Loop back-edges** — loop-carried values stay live through the entire loop
- **Terminator uses** — values used in `Jump`/`JumpIf` args (which aren't in the instruction array)

### Type-compatible slot reuse

Slots are only reused if the types match. This isn't strictly necessary for correctness (the interpreter could use untyped slots), but it preserves type information through `val_types` and makes the output IR easier to validate and debug.

### What we don't do

- **Graph coloring.** Linear scan is O(n log n) and produces results within 5-10% of optimal for the straight-line code typical in templates/scripts. Full graph coloring (Chaitin's algorithm) would add complexity for marginal benefit.
- **Spilling.** We always have enough "registers" (ValueIds are virtual). There's no physical register limit to spill for.

---

## Type System Decisions

### Why structural typing, not nominal

Templates and scripts are glue code — they receive data from external systems and transform it. Requiring users to declare types would defeat the purpose. Structural typing means `{ name: String }` works regardless of where the object came from. The compiler infers the minimum structural requirements from usage.

### Why open enums

In an embedded scripting context, the set of valid enum variants isn't always known at compile time (plugins can extend it). Open enums — where writing `Color::Red` automatically declares the variant — eliminate the declaration burden while still providing tag-based pattern matching.

### Why Identity tracking for deques

Deques (mutable lists) carry an origin tag to prevent accidentally mixing data from different sources. Without this, `extend(@history, @other_history)` would silently succeed even if the two histories have incompatible semantics. The origin system makes this a compile-time error unless explicitly coerced.

### Constraint-based generics: what we gain and lose

**Gain:** No trait declarations, no type class instances, no `where` clauses. Users write code and constraints are inferred. This is ideal for a scripting language where users shouldn't think about type theory.

**Lose:** Error messages for constraint violations can be confusing — "this value doesn't support operation X" is less clear than "type T doesn't implement trait Y". We accept this trade-off because the target audience writes short scripts, not library code.

---

## The Pass System: Why Typed Dependencies

The `PassManager` uses Rust's type system to enforce pass ordering:

```rust
impl TransformPass for ReorderPass {
    type Required<'a> = (&'a FnTypes, &'a TransformMarker<SpawnSplitPass>);
}
```

This means ReorderPass can only run after SpawnSplitPass. The dependency is enforced at compile time — you literally can't construct a PassManager with a missing dependency. Kahn's algorithm topologically sorts at construction time; cycles are detected and panic.

### Why not just a fixed list?

A fixed list works today but doesn't compose. When adding a new pass, you'd need to find the right insertion point in the list and hope you got the ordering right. With typed dependencies, adding a new pass is:
1. Declare what it needs
2. Add it to the Chain
3. If dependencies are wrong, it won't compile

### External dependency injection

`FnTypes` (the function type table) is injected into `PassContext` before the pipeline runs. This avoids threading function types through every pass signature and makes the pipeline reusable across different compilation contexts.

---

## Incremental Compilation: Why SCC-Based

The `IncrementalGraph` caches inference results per Strongly Connected Component, not per function. The reason: mutually recursive functions must be inferred together (their types depend on each other). Caching per-function would either miss cross-function constraints or require re-inferring the entire group anyway.

**Early cutoff** is the key optimization: if re-inferring an SCC produces the same types as before, none of its callers need re-inference. In practice, most edits (changing a string literal, adding a field access) don't change the function's type signature, so re-inference stops at the edited SCC.

### LSP integration

The LSP is a thin wrapper over `IncrementalGraph`. **LSP diagnostics and build errors are identical because they run the exact same pipeline, the exact same code.** There is no separate "LSP mode" that could diverge from the real compiler. This was a deliberate choice — maintaining two analysis paths is a guaranteed source of bugs.

---

## Validation: Soundness and Completeness

### What we check

**Type checking** — Every instruction's operands match expected types. Arity, constructor shape, materiality (functions can't be stored to context — they're ephemeral).

**Move checking** — Move-only values (`Handle<T, E>`, self-modifying iterators) are consumed exactly once. Use-after-move and unused handles are compile errors. Propagates transitively: a closure capturing a move-only value is itself move-only.

### What we guarantee

- If validation passes, the IR is well-typed and move-safe.
- Validation runs after all optimizations — it catches optimizer bugs, not just user bugs.

### What we don't guarantee

- **Runtime safety.** Integer overflow, out-of-bounds access, and division by zero are not caught at compile time. These would require full dependent types or abstract interpretation on every code path, which is overkill for a scripting language.
- **Termination.** Recursive functions and infinite loops are allowed. We detect recursion (SCC analysis) but don't prevent it.
- **Plugin correctness.** If a plugin's handler doesn't match its declared type/effect, runtime errors will occur. The compiler trusts plugin declarations.

---

## Key Invariants

1. **Types are complete.** No unresolved type variables survive past inference. Every value in lowered MIR has a concrete type.
2. **Effects are sound.** If a function is marked pure, it truly has no side effects. The system never over-promises.
3. **Moves are checked.** Move-only values are consumed exactly once. Use-after-move is a compile error.
4. **Output is deterministic.** Same source always produces the same MIR, regardless of hash map ordering or global state.
5. **Single source of truth.** Value types live in `val_types`, not duplicated in instruction fields. Context identity is the qualified name, not an opaque ID.
6. **Token ordering is preserved.** Functions sharing the same Token are never reordered relative to each other. SSA never lifts Tokens into value flow.
7. **Validation is final.** Type check and move check verify the IR after all optimizations. If they pass, the output is sound.
