# acvus MIR Architecture

## What is this?

acvus is not a general-purpose programming language. It is a **script and template language** — the kind you embed in a larger system to let users write business logic, data transformations, and LLM orchestration without touching the host codebase.

The difference is that acvus takes this role seriously. Instead of a loosely-typed interpreter with string-based extension points (like most embedded scripting), it has a full compiler backend with type inference, effect tracking, and SSA-based IR. The goal is simple: **scripts and templates should be safe, fast, and pleasant to write** — with the same rigor that systems languages apply to systems code.

A single expression like `@users | filter(active) | map(name) | join(", ")` compiles to the same typed, optimized IR whether it appears inside `{{ }}` template interpolation or as a standalone script. The MIR knows not just *what* your code computes, but *which external state it reads and writes*, enabling automatic parallelization, incremental recompilation, and portable execution.

---

## Design Principles

**One IR, multiple surfaces.** Template and script share the same type system, the same compiler pipeline, the same IR. No special-casing.

**Effects are first-class.** Every function's effect — how it may be reissued, whether it commutes, which contexts it reads and writes — is part of its type. This isn't an annotation — it's inferred from code and propagated through call chains by unification.

**No opaque IDs where names suffice.** Contexts are identified by qualified names (`@namespace:name`), not opaque integer IDs. This makes the IR deterministic, serializable, and human-readable without a symbol table.

**Plugin-extensible type system.** A registry contributes a manifest (type declarations, shared signatures, function declarations) and a handler table; all registries are combined once, before anything is checked (RFC-0021). Adding an ExternFn expands the set of valid programs — the type system grows with the ecosystem.

**Use first, define later.** Most things in acvus are inferred from usage, not declared upfront. Write `@data | map(f) | collect` and the compiler works *backwards* — `@data` must be iterable, `f` must return something, the result is a list of that something. Context types, function parameter types, effect footprints, even generic constraints are all discovered by analyzing how values are used, then propagated outward to the environment. The host system provides concrete types for contexts; the compiler checks that they satisfy the constraints the code imposed. This inverts the traditional "define type, then use" flow — users write code freely, and the system figures out what the environment must provide.

**Only a primitive copies.** the integers (`i8`…`u64`), `Float`, `Bool`, `Unit`, `Order`, and a reference are words and copy. Every other value moves: a binding used twice is a type error, and duplication is an explicit `clone` extern (RFC-0018). `&T` and `&mut T` are types; a reference is never data, never returned, never captured.

---

## ExternFn: First-Class SSA Citizens

External functions in acvus are not black boxes. They are **first-class citizens of the SSA graph**, indistinguishable from built-in operations at the IR level.

### What "first-class" means concretely

An ExternFn is declared once, as a Rust function under `#[extern_fn]`; its acvus type and its handler both come from that signature (RFC-0023). The declaration carries the function's type and effect (`effect = pure | idempotent | opaque`, and `commutative`). From that point on, the SSA sees it as just another node in the dataflow graph:

- **Uses** — SSA values flow *into* the ExternFn as arguments.
- **Defs** — SSA values flow *out* as results.

There is no marshalling and no opaque boundary. The runtime contract (RFC-0022) is `erase` / `materialize` by Rust type, `deref` / `deref_mut` / `reference` for reference values, and `call_0/1/n` for closure values; a container of a type variable crosses whole as the Rust type it is at the call's resolved type.

### What this enables

Because ExternFns participate fully in the SSA dataflow, the standard passes apply to them:

- **Dead code elimination** — A call whose effect is Pure and writes no context is dead if its result is unused (`optimize/dce.rs`).
- **Spawn/Eval splitting and scheduling** — An effectful direct call is split into `Spawn` + `Eval`, and the two are scheduled apart to hide latency (see *Automatic IO Parallelization*).
- **Devirtualization** — A closure passed to an ExternFn is a value; where an indirect call's callee traces to a single `MakeClosure`, the inliner inlines it.
- **Coercion as a call** — A registered cast rule lowers to an ordinary pure `FunctionCall` of the cast ExternFn. There is no cast instruction.

### Why this is possible

The all-or-nothing boundary. An ExternFn either hasn't entered the system (just a Rust function), or it has fully entered — with type and effect known to the compiler, and its name namespaced by the registry that declares it. There is no intermediate state where a function is "partially known." Registration happens once, through `Externs::combine`, which rejects a name declared twice.

### Safety guarantees

ExternFn authors cannot violate SSA invariants. The Uses/Defs interface ensures:

- No hidden reads (all inputs come through Uses)
- No hidden writes (all outputs go through Defs)
- No access to a context (an ExternFn's context summary is empty; a script reads a context and passes the value — RFC-0014, RFC-0017)

What the compiler does *not* verify is the effect declaration itself. An ExternFn author who declares `pure` and does IO has written a wrong library; the default for an undeclared ExternFn is Opaque, the sound direction.

---

## The Pipeline and Why It's Split This Way

```
source -> extract -> infer -> lower -> optimize -> MirModule
```

The pipeline is split into four phases not because it's architecturally elegant, but because **the LSP needs to cut into the middle**. When a user edits a script, the system re-runs only the phases invalidated — not the entire compilation.

**Extract** parses source and caches the AST per function. Context dependency tracking is not done here; infer does it.

**Infer** resolves all types via constraint propagation. Inter-function inference uses Tarjan's SCC algorithm so mutually recursive functions are solved simultaneously. Infer results are cached per-SCC with **early cutoff** — if a function's type didn't change after re-inference, its callers don't need re-inference. Most edits touch one SCC, so recompilation is O(changed) not O(total).

**Lower** translates typed AST to MIR instructions. A context is a variable of the body that names it (RFC-0025): every named context is fetched from the page at entry (`Fetch`), committed back at every return (`Commit`), and committed before and fetched after each call whose summary (RFC-0017) touches it. Inside the body, a read of a variable is a `Take`; an assignment is an `Assign`; `&place` / `&mut place` is a `Ref`; `*r` is a `Load`. Every effectful call takes the current `Order` and yields a new one. This is a "pre-SSA" IR that's easy to generate from AST; the SSA pass in optimize handles the hard part (phi insertion at merge points).

**Optimize** is the final phase. Two passes (`graph/optimize.rs`):
- **Pass 1** (per body, then cross-module): SSA → DSE → DCE on every body, then inlining across modules. Inline must see all modules because it resolves cross-function calls and devirtualizes closures.
- **Pass 2** (per body): Commute → SpawnSplit → SSA → DSE → DCE → CodeMotion → Reorder → DropInsertion. Then, per module, **Validate** (type check + move check) on the demoted `MirBody`.

### Why validate at the end, not after lower?

Validation runs after all optimizations, not after lowering. The reasoning: optimizations (inline, reorder, SSA) transform the IR in ways that could theoretically break invariants. Validating the *final* IR catches bugs in the optimizer itself. If we validated early and skipped post-optimization validation, an optimizer bug could produce unsound IR silently.

---

## Effect System: Reissue, Commutativity, and Context Summary

A call's effect (`ty.rs`, `Effect`) has three parts:

**Reissue** (RFC-0014) — the chain `Pure < Idempotent < Opaque`. A Pure call has no effect and stands nowhere in the order of a run. An Idempotent call keeps its order and may be issued twice. An Opaque call keeps its order and must not be issued twice.

**Commutes** (RFC-0013) — whether two calls of the function are the same program in either order. Pure commutes by definition; a call that writes a context never commutes.

**Context summary** (RFC-0017) — the set of contexts the call may read and the set it may write. A function's summary is the union over its calls and its own accesses, closed over recursion in the same SCC that closes its effect. An ExternFn's own summary is empty.

### Order is a value

Ordering between effectful calls is not a side table: it is a value of type `Order` that no script can name (RFC-0007). A call whose effect is not Pure takes an `Order` and yields one; sequential code is a chain where each call takes what the previous one yielded. `Merge` joins orders the way a phi joins values — associative, commutative, a value instruction rather than control flow. An `anyorder { ... }` block lowers to a fan-out from the block's entry order and one `Merge` at its exit.

Because order is a data dependency, every pass that respects use-def already respects order. Contexts are the other kind of state: a read is a `Take`, a write is an `Assign`, and the SSA pass threads them through block parameters where the storage is read and written whole.

### What we guarantee

- If a call's effect is Pure and its write set is empty, removing it when its result is unused changes nothing (DCE relies on this).
- Effects propagate through call chains — a function's effect is unified with the effects of its calls in the same SCC.
- Two calls on the `Order` chain are never issued out of order unless the author declared `anyorder` or both callees declare `commutative`.

### What we don't guarantee

- **Optimal parallelism.** The reorder pass uses a greedy topological sort with priorities (Spawn earliest, Eval just before first use). It doesn't solve for globally optimal scheduling.
- **Effect inference across plugin boundaries.** If a plugin function lies about its effects (claims pure but does IO), the system has no way to catch this. Plugin authors must be honest. This is a conscious trade-off: verifying plugin effects would require sandboxing or formal verification, neither of which is practical for an embedded scripting language.
- **Inference of IO ordering.** Whether two IO calls may be reordered is the script author's intent, declared with `anyorder`; it is never inferred from types (RFC-0007).

---

## Automatic IO Parallelization

This is the centerpiece optimization. The insight: in template/script workloads, the bottleneck is almost always IO (API calls, database queries), not computation. If we can automatically parallelize independent IO operations, users get massive speedups without changing their code.

### How it works

1. **Commute** — Maximal runs of neighbouring commutative calls on the `Order` chain are rewritten as an `anyorder` block would be: every call takes the run's entry order, one `Merge` stands where the last call's order stood.

2. **Spawn split** — Every effectful direct `FunctionCall` is split into `Spawn` (issues the call, takes the order before it) + `Eval` (forces the `Handle`, yields the order after it). Scheduling work and waiting for it become two instructions.

3. **Code motion** — Pure instructions are hoisted to the highest dominator where their operands are available; `Eval` and whole context `Take`/`Assign` are sunk toward their first use. A `Spawn` is never hoisted across a branch: issuing it on a path that would not have reached it speculates an effect (RFC-0007).

4. **Reorder** — Within each basic block, instructions are topologically sorted by use-def dependencies. Priority: Spawn = schedule earliest, Eval = just before the first use of its result, everything else = original order.

5. **SSA / DSE / DCE re-run** after spawn split, so that dead handles and dead stores introduced by splitting are removed before code motion.

### What we don't do

- **Speculative issue.** IO inside a branch is issued where the branch issues it. The one exception is a commutative call whose block post-dominates the block of the call it follows — every path through that block reaches it, so issuing it there speculates nothing.
- **CPU-heavy parallelization.** Only calls on the `Order` chain are split. Pure computation is never spawned.

---

## SSA: Why Cranelift-Style, Not LLVM-Style

The SSA pass uses a Cranelift-style `SSABuilder` rather than LLVM's dominance-frontier-based approach. The reason: **block parameters instead of PHI nodes**.

In LLVM's model, PHI nodes are special instructions at block entry that select values based on which predecessor you came from. This requires knowing predecessors at PHI creation time and makes instruction rewriting fragile.

In Cranelift's model, blocks take parameters (like function parameters), and jumps pass arguments. `Jump { label: L0, args: [v1, v2] }` → `BlockLabel { label: L0, params: [v3, v4] }`. This is:
- **Simpler to construct** — no need to track predecessors during construction
- **Simpler to transform** — inlining just remaps args, no PHI surgery
- **Naturally SSA** — block params are definitions, jump args are uses, standard use-def chain

### What is promoted

A storage — a local or a parameter — is promoted only when every access to it is a whole `Take` or `Assign`. A storage that is referenced (`Ref`) or accessed by field path stays in memory; the SSA pass leaves its instructions alone. A context's variable is a local like any other; `Fetch` defines a value and `Commit` uses one, and neither is promoted, removed, or moved.

### What this means for the optimizer

After SSA, the IR is functional where promotion applied — mutation is expressed as new values flowing through block parameters. Liveness, register coloring, and reordering all operate on standard SSA use-def chains.

---

## Inlining: Why Devirtualize Before Inline

The inliner runs devirtualization as part of its pass: if an `Indirect` callee traces back to a single `MakeClosure` (not through a block parameter), the closure body is inlined directly with captures prepended to args.

### Why not a separate devirt pass?

Devirtualization and inlining share the same machinery — both need to trace value definitions, remap ValueIds, and splice instruction sequences. A separate pass would duplicate this work. More importantly, devirt decisions depend on the inline context: a closure might be worth devirtualizing only if the call site is also being inlined.

### What we don't devirtualize

- **PHI-sourced closures.** If a closure value comes from a block parameter (meaning it could be one of multiple closures depending on which branch was taken), we leave it as an indirect call. Sound devirt would require splitting the call site, which isn't justified for the common case.
- **Recursive closures.** Detected via SCC analysis. Inlining recursive calls would diverge.

---

## Register allocation lives in the runtime

MIR values stay as the optimizer left them: one `ValueId` per definition. A runtime that needs physical registers allocates them itself (kovac plans a DAG-based selector); the compiler holds no coloring pass, so a checker never sees one value defined twice.

---

## Type System Decisions

### Why structural typing, not nominal

Templates and scripts are glue code — they receive data from external systems and transform it. Requiring users to declare types would defeat the purpose. Structural typing means `{ name: String }` works regardless of where the object came from. The compiler infers the minimum structural requirements from usage.

### Why open enums

In an embedded scripting context, the set of valid enum variants isn't always known at compile time (plugins can extend it). Open enums — where using a variant declares it — eliminate the declaration burden while still providing tag-based pattern matching.

### Why identity is a type parameter

Identity is a kind of type parameter, like effect (RFC-0012). A user-defined type declares whether it has an identity parameter (at most one); a value of such a type is one source and moves. Two identities unify only when they are the same, so a value from one source never mixes with a value from another. There is no identity type and no structural type is tagged; the runtime sees none of it.

### Declared bounds, not traits

A type variable of an ExternFn carries a bound declared with it: `Any`, or `OneOf(types)` (RFC-0011). Shared signatures (RFC-0019) lower a requirement `T: sig` to `OneOf(the types with an instance)`. Users write code and the bounds are checked when a variable freezes.

**Gain:** No trait declarations, no type class instances. This is ideal for a scripting language where users shouldn't think about type theory.

**Lose:** Error messages for bound violations can be confusing — "this type is not one of …" is less clear than "type T doesn't implement trait Y". We accept this trade-off because the target audience writes short scripts, not library code.

---

## Incremental Compilation: Why SCC-Based

The `IncrementalGraph` caches inference results per Strongly Connected Component, not per function. The reason: mutually recursive functions must be inferred together (their types depend on each other). Caching per-function would either miss cross-function constraints or require re-inferring the entire group anyway.

**Early cutoff** is the key optimization: if re-inferring an SCC produces the same types as before, none of its callers need re-inference. In practice, most edits (changing a string literal, adding a field access) don't change the function's type signature, so re-inference stops at the edited SCC.

### LSP integration

The LSP is a thin wrapper over `IncrementalGraph`. **LSP diagnostics and build errors are identical because they run the exact same pipeline, the exact same code.** There is no separate "LSP mode" that could diverge from the real compiler. This was a deliberate choice — maintaining two analysis paths is a guaranteed source of bugs.

---

## Validation: Soundness and Completeness

### What we check

**Type checking** (`validate/type_check.rs`) — Every instruction's operands match the types recorded in `val_types`. Arity, constructor shape, and the `Order` edge of a call (an effectful call carries one; a Pure call carries none).

**Move checking** (`validate/move_check.rs`) — Every value that is not a word is move-only (`is_move_only`); an option is a word exactly when its payload is, since it has no representation of its own (RFC-0039). A `Take` of a storage after its value was taken is a use after move; an `Assign` revives the storage. At a merge, `Alive` joined with `Moved` is `Moved`.

**Definite assignment** (`validate/init_check.rs`) — Field-level: which fields of each storage are definitely initialized at each point, and whether a call's arguments carry every field the callee's parameter type requires. Runs on the pre-SSA `CfgBody`.

### What we guarantee

- If validation passes, the IR is well-typed and move-safe.
- Validation runs after all optimizations — it catches optimizer bugs, not just user bugs.

### What we don't guarantee

- **Runtime safety.** Integer overflow, out-of-bounds access, and division by zero are not caught at compile time. These would require full dependent types or abstract interpretation on every code path, which is overkill for a scripting language.
- **Termination.** Recursive functions and infinite loops are allowed. We detect recursion (SCC analysis) but don't prevent it.
- **Plugin correctness.** If a plugin's handler doesn't match its declared type/effect, runtime errors will occur. The compiler trusts plugin declarations.

---

## Key Invariants

1. **Types are complete.** No unresolved type variables survive past inference. Every value in lowered MIR has a concrete type (`Ty = TyTerm<Concrete>`, whose `Var` is uninhabited).
2. **Effects are sound.** If a function is marked Pure, it truly has no side effects. The system never over-promises.
3. **Moves are checked.** Only a primitive or a reference copies; every other value is consumed exactly once. Use-after-move is a compile error.
4. **Output is deterministic.** Same source always produces the same MIR, regardless of hash map ordering or global state.
5. **Single source of truth.** Value types live in `val_types`, not duplicated in instruction fields. Context identity is the qualified name, not an opaque ID.
6. **Order is a dependency.** An effectful call takes an `Order` and yields one; a pass that respects use-def respects order. Nothing hoists a `Spawn` above a branch.
7. **Validation is final.** Type check and move check verify the IR after all optimizations. If they pass, the output is sound.
