# Acvus

A compiled template engine for LLM orchestration. Templates go through a full compiler pipeline — parsing, type checking with variance, MIR lowering, static analysis — before executing as a dependency-resolved DAG. No async runtime dependency. Runs natively in WASM.

## How it works

Templates and scripts are compiled, not interpreted:

```
Source → AST → Type Check (with variance) → MIR (SSA) → Analysis Passes → DAG → Execution
```

Each stage catches a different class of errors before anything runs:

- **Parser** (lalrpop + logos): Syntax errors, malformed patterns, unclosed blocks.
- **Type checker**: Mismatched operations (`Int + String`), undefined context references, origin conflicts between deques, variance violations in coercions.
- **MIR lowering**: SSA intermediate representation with structured control flow, closures, iterators, and tagged variants.
- **Analysis passes**: Dead branch pruning, reachable context partitioning (eager vs. lazy), variable mutation tracking. A dependency-aware `PassManager` topologically orders passes by their type-level requirements.
- **DAG builder**: Infers node dependencies from context references, topologically sorts, detects cycles.
- **Resolver**: Coroutine-based event loop. Prefetches eager dependencies, defers lazy ones until actually needed. Serializes concurrent writes to the same node state.

## Template language

Three reference kinds:

- `@name` — Context. Externally injected, read-only. Types declared at compile time.
- `$name` — Variable. Local mutable storage. Type inferred at first assignment.
- `name` — Value. Local immutable, SSA-resolved at lowering.

Pattern matching is the only control flow. No if/for/while. All values are immutable (variables are rebindable, not mutable).

```
{{-- Match block: test a value against patterns --}}
{{ "korean" = @language }}
한국어로 답변합니다.
{{ "english" = }}
Responding in English.
{{ _ }}
Default case.
{{ / }}

{{-- Iteration: loop over a collection --}}
{{ item in @items }}
- {{ item.name }}: {{ item.value | to_string }}
{{ / }}

{{-- Inline expression --}}
{{ @input | trim }}

{{-- Lambda + pipe --}}
{{ @messages | filter(m -> m.role == "user") | len | to_string }}
```

String is the only emit type. Explicit `to_string` required for non-string values. `Int + Float` arithmetic is a type error — explicit conversion needed.

Pipe chains are first-class function application: `a | f(b)` desugars to `f(a, b)`.

Format strings interpolate expressions inline: `"hello {{ name }}, age {{ age | to_string }}"`. The `{{ }}` delimiters open a full expression context — any valid expression works inside, including pipes.

### Builtins

List: `filter`, `map`, `pmap`, `find`, `reduce`, `fold`, `any`, `all`, `len`, `reverse`, `flatten`, `join`, `contains`.

Type conversion: `to_string`, `to_float`, `to_int`, `char_to_int`, `int_to_char`.

String: `contains_str`, `substring`, `len_str`, `trim`, `trim_start`, `trim_end`, `upper`, `lower`, `replace_str`, `split_str`, `starts_with_str`, `ends_with_str`, `repeat_str`.

Byte: `to_bytes`, `to_utf8`, `to_utf8_lossy`.

Other: `unwrap`.

Extension modules (e.g. `acvus-ext`) register additional extern functions at compile time.

## Type system

Static types with variance. The type checker runs before execution and catches errors that other template engines only surface at runtime.

**Types**: `Int`, `Float`, `String`, `Bool`, `Unit`, `Byte`, `Range`, `List<T>`, `Deque<T>`, `Iterator<T>`, `Option<T>`, `Tuple`, `Object`, `Enum`, `Fn`.

**Variance**: Covariant, contravariant, and invariant positions are tracked during unification.

- `Deque<T>` unifies to `List<T>` in covariant position (safe widening).
- `List<T>` unifies to `Iterator<T>` in covariant position.
- Function parameters are contravariant — polarity flips.
- Invariant positions require exact match.

**Origin tracking**: Each `Deque` carries an origin identity. Two deques from different `[]` literals have different origins. Mixing them in invariant position is a compile-time error — prevents silent data corruption from accidentally extending the wrong collection.

**Analysis mode**: Unknown context references get fresh type variables instead of errors, enabling incremental type discovery. The frontend uses this to discover what parameters a template needs and what types they should have, before the user has filled them in.

## Orchestration

A project is a set of TOML node definitions and `.acvus` templates:

```
my-project/
  project.toml        # providers, context types, node list
  chat.toml            # node: LLM call with messages
  summarizer.toml      # node: another LLM call
  system.acvus         # template for system message
  user.acvus           # template for user message
```

### Nodes

Each node has a kind, a strategy, and typed context:

```toml
name = "chat"
kind = "llm"
provider = "gemini"
model = "gemini-2.5-flash"

[execution]
mode = "once-per-turn"

[persistency]
kind = "deque"
inline_bind = "@self | extend(@raw)"

[[messages]]
role = "system"
template = "system.acvus"

[[messages]]
iterator = "@self"      # iterate stored history

[[messages]]
role = "user"
template = "user.acvus"
```

**Node kinds**:
- `llm` — API call to a language model. Messages are compiled templates.
- `plain` — Template rendering. No API call.
- `expr` — Script evaluation. Returns a computed value.

**Execution strategies**:
- `always` — Runs every invocation.
- `once-per-turn` — Runs once per turn, result persisted.
- `if-modified:<key>` — Runs only when the referenced key changes.

**Persistency modes**:
- `ephemeral` — Not stored.
- `snapshot` — Full overwrite each turn.
- `deque` — Append-only with bind script. Tracks diffs.
- `diff` — Object field-level patches with bind script.

**Retry and assert**: Nodes can specify `retry` count and an `assert` script (must return `Bool`). If the assert fails, the node retries up to the limit.

```toml
retry = 3
assert = "@self | len_str > 0"
```

### Tool calls

Nodes can expose tools to LLM nodes. The LLM decides when to call them:

```toml
[[tools]]
name = "get_weather"
description = "Get current weather for a city"
node = "get_weather"
params = { city = "string" }
```

Tool parameters are type-checked at compile time. When the LLM emits a tool call, the resolver executes the target node with the parameters injected as typed context.

### DAG resolution

The orchestrator builds a dependency graph from context references: if node A's template references `@chat`, and a node named `chat` exists, A depends on `chat`. Dependencies are topologically sorted. Cycles are detected and reported as structured errors.

Context keys are partitioned by static analysis:

- **Eager**: Unconditionally needed — prefetched before the node runs.
- **Lazy**: Behind unknown branches — resolved on demand via coroutine suspension.
- **Pruned**: In dead branches — type information preserved, but no runtime fetch.

### Providers

Multiple providers supported per project. Each node picks its provider independently.

- OpenAI-compatible APIs
- Anthropic
- Google (Gemini), including context caching

The `Fetch` trait abstracts HTTP transport. The caller provides the implementation — `reqwest` on servers, browser `fetch()` in WASM.

### Token budgets

Message iterators support token budget constraints:

```toml
[[messages]]
iterator = "@self"
token_budget = { priority = 0, max = 12000 }
```

Multiple iterators with different priorities fill the budget in priority order.

## Context type registry

Context variables are organized into tiers with conflict detection:

- **extern_fns**: Builtin functions (regex, etc.)
- **system**: Orchestration-provided (`@turn_index`, node outputs)
- **scoped**: Node-local (`@self`, `@raw`, function params)
- **user**: Frontend-injected (`@input`, custom params)

The same key cannot appear in multiple tiers. Construction fails with `RegistryConflictError` on violations.

## History system

Conversation history uses a tree-structured journal backed by content-addressed blob storage.

**Blob store**: Immutable blobs identified by blake3 hash. Named refs with atomic compare-and-swap for concurrent access.

**Tree journal**: Each turn appends a node with `turn_diff` (changes this turn). Full snapshots are taken at intervals. Reconstruction walks up to the nearest snapshot and replays diffs forward.

- **Branch**: Fork from any point to explore alternative paths.
- **Undo**: Navigate to any previous node in the tree.
- **Prune**: Remove leaves or subtrees.
- **Merge**: CAS-based with automatic union on conflict (CRDT-style).
- **GC**: Mark live blobs from tree metadata, batch-remove garbage.

State updates use copy-on-write: the parent's accumulated state is shared via `Arc`, and each child only records its own diff.

## Error handling

Errors are structured enums at every layer. No string formatting for error construction.

- **Parse errors**: Span-annotated syntax errors.
- **Type errors**: `TypeMismatchBinOp`, `UnificationFailure`, `OriginMismatch`, `UndefinedContext`, etc. Each variant carries the relevant types for display.
- **Orchestration errors**: `CycleDetected`, `RegistryConflict`, `ToolParamType`, etc.
- **Runtime errors**: Propagated as `Result` through the coroutine protocol. No panics, no unwind. WASM-safe with `panic=abort`.

All error display requires an explicit interner reference — no hidden thread-local state.

## Architecture

```
acvus-ast               Parser (lalrpop + logos)
acvus-mir               Type checker + MIR lowering (variance, origin tracking)
acvus-mir-pass          Analysis passes (reachable context, val def map, var dirty)
acvus-mir-cli           CLI for MIR inspection
acvus-mir-test          MIR snapshot tests (insta)
acvus-interpreter       Runtime values, sync execution, RuntimeError
acvus-interpreter-test  Interpreter e2e tests
acvus-utils             Astr (interned strings), TrackedDeque, coroutine primitives
acvus-ext               Extension modules (regex, builtins)
acvus-orchestration     Node compilation, DAG, resolver, blob store, storage traits
acvus-chat              Chat engine — multi-turn orchestration with tree history
acvus-chat-cli          CLI — TOML projects, multi-provider HTTP
pomollu-engine          WASM bindings (wasm-bindgen + tsify)
pomollu-frontend        Web UI (SvelteKit + Tailwind) — block editor, typed params, grid layout
```

## WASM

The core pipeline has no tokio, no reqwest, no OS dependencies. `acvus-ast`, `acvus-mir`, `acvus-mir-pass`, and `acvus-interpreter` compile to `wasm32-unknown-unknown` directly. `acvus-orchestration` produces `BoxFuture` values — the caller supplies the executor and fetch implementation. Runtime errors use `Result`, not panics, so `panic=abort` is safe.

`pomollu-engine` exposes WASM bindings for the browser:

- `analyze()` — discover context keys and infer types.
- `typecheck()` — full type checking with span-annotated errors.
- `typecheckNodes()` — whole-project orchestration type check.
- `evaluate()` — execute a template with given context.
- `ChatSession` — multi-turn execution with tree history, undo, fork, and branch navigation.

## Examples

The `examples/` directory contains runnable projects:

- **chat** — Basic multi-turn conversation with history.
- **tool-chat** — LLM with tool calls (weather lookup).
- **budget-chat** — Token budget constraints on message history.
- **diamond-chat** — Diamond DAG: input fans out to translate + sentiment + chat, then merges in output.
- **format-chat** — Pattern matching on context (`@language`) for conditional system prompts.
- **inline-chat** — Inline templates (no `.acvus` files).
- **multi-budget-chat** — Multiple iterators with different budget priorities.
- **translate-chat** — Translation pipeline.

## License

Acvus License — free to use, copy, modify, and distribute. Cannot be sold as a product or service. See [LICENSE](LICENSE) for details.
