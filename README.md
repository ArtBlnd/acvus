# Acvus

A compiled template and script language for LLM orchestration. Three surface syntaxes — template, expression, script — share a single compiler pipeline: parsing, type inference with variance, SSA-based MIR, optimization passes, and validation. No async runtime dependency. Runs natively in WASM.

## Pipeline

```
Source → Parse → Extract → Infer (SCC) → Lower → Optimize → Validate → Execute
```

**Parse** (lalrpop + logos): Three modes — template (`{{ }}`), expression, and script (`let`/`if`/`for`/`while`). All produce AST.

**Extract**: Discovers context references and traces projection chains to determine read vs. write.

**Infer**: Constraint-based type inference. Tarjan's SCC for mutually recursive functions. Resolves function signatures, context types, and effect footprints. Results cached per-SCC with early cutoff — if a function's type didn't change, its callers skip re-inference.

**Lower**: Typed AST to flat MIR instructions (pre-SSA). Context mutations become `ContextProject`/`ContextStore` pairs. Pattern matching becomes `TestXxx` + `JumpIf` chains.

**Optimize**: Two passes.
- Pass 1 (cross-module): SROA → SSA → DCE → inline (cross-function, closure devirtualization).
- Pass 2 (per-module): SpawnSplit → CodeMotion → Reorder → SSA → RegColor.

**Validate**: Type check and move check on final MIR, after all optimizations. Catches optimizer bugs, not just user bugs.

## Language

Three surface syntaxes, single MIR.

### Template mode

```
{{-- Comment --}}
Hello, {{ @name }}!

{{ "korean" = @language }}
한국어로 답변합니다.
{{ "english" = }}
Responding in English.
{{ _ }}
Default case.
{{ / }}

{{ item in @items }}
- {{ item.name }}: {{ item.value | to_string }}
{{ / }}

{{ @messages | filter(|m| -> m.role == "user") | len | to_string }}
```

### Scripts

```
let count = 0;
let it = as_iter(&@items);
while let Some(item) = next(&mut it) {
    if item.active {
        count = count + 1;
    };
}
let label = if count > 10 { "many" } else { "few" };
label
```

`let x = expr;` is a new binding and shadows an outer `x`; it ends with the
block that introduced it. `x = expr;` assigns the `x` in scope, and never
introduces one — assigning a name with no binding in scope is a compile
error, as is assigning a name a lambda captured. `if`/`if let`/`while`/
`while let`/`anyorder` and the tag form `pattern = source { body };` are
statements of the same rule in every block: the script's top level, a
lambda's block body, and any of those bodies.

### Expressions

```
@items | filter(|x| -> x.active) | map(|x| -> x.name) | join(", ")
```

The same expression language used inside `{{ }}` in templates: pipes and
pattern matching, with no statement around them.

### References

- `@name` — Context. Externally injected. Type inferred from usage or declared by host.
- `$name` — Extern parameter. Immutable, injected at call site.
- `name` — Local value. Immutable, SSA-resolved.

### Pipes

`a | f(b)` desugars to `f(a, b)`. Chains left-to-right: `a | b | c` is `c(b(a))`.

### Format strings

`"hello {{ name }}, count {{ count | to_string }}"` — full expression context inside `{{ }}`.

### Patterns

Irrefutable: `x`, `(a, b)`, `{ name, age, }`.
Refutable: literals, ranges (`0..10`), list destructuring (`[a, b, ..rest]`), variants (`Some(x)`, `None`), wildcard (`_`).

## Type system

Static types with variance, identity tracking, and effect inference. The type checker runs before execution.

**Types**: `Int`, `Float`, `String`, `Bool`, `Unit`, `Byte`, `Range`, `List<T>`, `Deque<T, I>`, `Object`, `Tuple`, `Option<T>`, `Enum`, `Fn`, `Handle<T, E>`, `UserDefined`, `Identity`.

### Inference

Types are inferred from usage, not declared. Write `@data | map(f) | collect` and the compiler works backwards — `@data` must be iterable, `f` must return something, the result is a list of that something. The host provides concrete types for contexts; the compiler checks they satisfy the inferred constraints.

### Variance

Covariant, contravariant, and invariant positions tracked during unification.

- Function parameters are contravariant. Return types are covariant.
- Collection elements are invariant.
- `Deque<T, I>` coerces to `List<T>` (identity erased). Explicit `Cast` instruction in MIR.
- `Range` coerces to `List<Int>`.
- Plugin-registered coercions via `ExternCast` rules.

### Identity

Collection literals receive a unique compile-time identity. Two deques from different `[]` literals have different identities. Mixing them in invariant position is a type error — prevents silently merging data from unrelated sources.

```
a = [1, 2, 3]       // Deque<Int, Identity(1)>
b = [4, 5, 6]       // Deque<Int, Identity(2)>
a | extend(b)        // Type error: different identities
```

Identity mismatch at branch merge resolves via LUB — `Deque<T, I1>` and `Deque<T, I2>` become `List<T>` (identity erased). Invariant position remains an error.

### Effects

Every function tracks what it reads, writes, whether it does IO, and whether it's self-modifying.

Two kinds of effect targets:
- **Context** (`@name`) — SSA-compatible. Converted to value flow after SSA. Enables automatic parallelization.
- **Token** (`TokenId`) — Not SSA-compatible. Represents external shared state. Functions sharing a Token execute sequentially.

Effect subtyping: `Pure ≤ Effectful` in covariant position.

### Move semantics

Values with `self_modifying` or `io` effects are move-only. The compiler's move check performs forward dataflow over the CFG:

- Use after move is a compile-time error.
- Reassignment revives a moved variable.
- Conservative join at branch merge: if any branch moves a value, it's moved after merge.

### Open enums

Writing `Color::Red` auto-declares the variant. No enum declaration needed. Tag-based pattern matching works normally.

### Structural typing

`{ name: String, age: Int }` is a type, not a declaration. Any value with those fields has that type. Identity provides opt-in nominal-like separation when needed.

## ExternFn

External functions are first-class SSA citizens. When registered, they declare their type signature and effect. From that point, SSA treats them as dataflow nodes — values flow in through Uses, out through Defs.

This enables:
- **Dead code elimination** — unused call results are removed.
- **Common subexpression elimination** — duplicate pure calls are deduplicated.
- **Constant folding** — pure ExternFn with constant inputs can be evaluated at compile time.
- **Fusion** — chains like `filter | map | collect` can be fused into a single call.

No sandbox needed. Rust's `Send + Sync` and the Uses/Defs contract prevent hidden side effects.

## Optimization

### SROA (Scalar Replacement of Aggregates)

`Ref<T>` decomposed into individual SSA values. Removes ref/load/store patterns, letting SSA track all values directly. 62–85% IR reduction in benchmarks.

### SSA

Cranelift-style block parameters, not LLVM-style PHI nodes. `VarLoad`/`VarStore` disappear — values flow through block params. Context mutations inside branches get write-back stores at merge points. Trivial PHI elimination.

### IO parallelization

Independent IO calls are automatically parallelized:

1. **SpawnSplit** — IO `FunctionCall` split into `Spawn` (pure, returns `Handle`) + `Eval` (effectful, forces handle).
2. **CodeMotion** — `Spawn` hoisted above branches via dominator tree.
3. **Reorder** — Within each block, Spawn scheduled earliest, Eval deferred to just before first use.

```
// Source: four IO calls, a→b dependency, c and d independent
// Optimized MIR:
r0 = spawn fetch_a()       // 3-way parallel start
r1 = spawn fetch_c()
r2 = spawn fetch_d()
r3 = eval r0               // a ready
r0 = spawn fetch_b(r3)     // dependent IO starts
r3 = eval r1               // c ready
r4 = eval r2               // d ready
r5 = eval r0               // b ready
```

### Other passes

- **DCE** — Dead code elimination.
- **DSE** — Dead store elimination.
- **Const dedup** — Constant deduplication.
- **RegColor** — SSA-aware greedy register coloring via backward dataflow liveness.
- **Inline** — Cross-module inlining with closure devirtualization.

## Standard library

All builtins are ExternFn. Each registry declares one namespace; a bare
name several namespaces declare is settled by the call's evidence
(RFC-0043), and `ns::name` picks one.

- **`iter`**: `into_iter`, `as_iter`, `filter`, `map`, `find`, `reduce`, `fold`, `any`, `all`, `flatten`, `flat_map`, `join`, `contains`, `last`, `collect`, `take`, `skip`, `chain`, `rev_iter`, `next`, `min`, `max`, `sum`, `count`.
- **`vec`**: `len`, `is_empty`, `get`, `get_mut`, `first`, `last`, `reverse`.
- **`array`**: `len`, `is_empty`, `get`, `get_mut`, `first`, `last`.
- **`deque`**: `deque`, `push_front`, `push_back`, `pop_front`, `pop_back`, `len`, `is_empty`, `get`, `get_mut`, `first`, `last`.
- **`string`**: `len`, `is_empty`, `contains`, `find`, `rfind`, `substring`, `trim`, `trim_start`, `trim_end`, `upper`, `lower`, `replace_str`, `split_str`, `starts_with_str`, `ends_with_str`, `repeat_str`, `char_at`, `chars`, `lines`.
- **`core`**: `clone`, `eq`, `hash`, `to_string`, `to_int` — one shared signature each, an instance per type.
- **Option**: `unwrap`, `unwrap_or`, `map`, `get_or_else`.
- **Conversion**: `to_string`, `to_float`, `to_int`, `char_to_int`, `int_to_char`.
- **Encoding**: Base64, URL encoding.
- **DateTime**: Date/time operations.
- **Regex**: Pattern matching.

## Orchestration

Orchestration compiles spec files (TOML) into the same compilation graph as user code. Specs are declarative — they describe what to run, not how.

### Spec items

- **Block** — Template rendering.
- **Llm** — LLM API call with compiled message templates.
- **Display** — Output formatting.

### Providers

Three LLM provider implementations, each registered as ExternFn:

- **OpenAI** (OpenAI-compatible APIs)
- **Anthropic** (Claude)
- **Google** (Gemini)

### Incremental compilation

`Session` manages namespaces via `IncrementalGraph`. Spec changes trigger incremental recompilation — only affected SCCs are re-inferred. Field-level error tracking maps type errors back to specific spec fields via `SpanMap`.

## Error handling

Errors are structured enums at every layer. No string formatting for error construction.

- **Parse errors**: Span-annotated.
- **Type errors**: `TypeMismatch`, `UnificationFailure`, `IdentityMismatch`, `UndefinedContext`, etc.
- **Validation errors**: `UseAfterMove`, `TypeMismatch` at instruction level. Post-optimization.
- **Runtime errors**: Propagated as `Result`. No panics. WASM-safe with `panic=abort`.

All error display requires an explicit interner reference — no hidden thread-local state.

## Architecture

```
acvus-utils             Interner (Astr), Freeze, LocalId, QualifiedRef
acvus-ast               Parser (lalrpop + logos) — Template, Script, Expr
acvus-mir               Type system, inference, MIR lowering, analysis, optimization, validation
acvus-interpreter       Register-based VM, ExternFn registry, sync/async handlers
acvus-ext               Standard library (10 ExternFn registries)
acvus-ext-llm           LLM providers (OpenAI, Anthropic, Google)
acvus-ext-net           Network IO (`http::fetch_get`)
acvus-orchestration     Spec → CompilationGraph, incremental Session
acvus-lsp               Language server — diagnostics, completions, context discovery
acvus-cli               `acvus run|check|mir`, the script runner (RFC-0031)
acvus-mir-test          MIR snapshot tests (insta)
acvus-interpreter-test  Interpreter e2e tests
pomollu-engine          WASM bindings (wasm-bindgen + tsify)
pomollu-frontend        Web UI (SvelteKit)
```

`acvus-mir` knows nothing about the interpreter. The IR is designed for any backend. `acvus-interpreter` depends on `acvus-mir`, never the reverse.

## WASM

The core pipeline has no tokio, no reqwest, no OS dependencies. `acvus-ast`, `acvus-mir`, and `acvus-interpreter` compile to `wasm32-unknown-unknown` directly. Runtime errors use `Result`, not panics, so `panic=abort` is safe.

## Examples

The `examples/` directory contains runnable projects:

- **chat** — Basic multi-turn conversation.
- **tool-chat** — LLM with tool calls.
- **budget-chat** — Token budget constraints on message history.
- **diamond-chat** — Diamond DAG: input fans out to multiple nodes, then merges.
- **format-chat** — Pattern matching on context for conditional system prompts.
- **inline-chat** — Inline templates (no `.acvus` files).
- **multi-budget-chat** — Multiple iterators with different budget priorities.
- **translate-chat** — Translation pipeline.

## License

Acvus License — free to use, copy, modify, and distribute. Cannot be sold as a product or service. See [LICENSE](LICENSE) for details.
