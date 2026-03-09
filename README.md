# acvus

A template engine for multi modal orchestration. Templates compile to typed intermediate representation, execute as a DAG, and run anywhere — no runtime dependency required.

## What it does

- Pattern matching as the only control flow. No if/for/while. All variables immutable.
- Static type system catches errors at compile time. No silent failures at runtime.
- Templates compile through a full pipeline: `Template → AST → Type Check → MIR (SSA) → Execution`.
- MIR enables static analysis passes — dead branch pruning, reachable context analysis, dependency-aware scheduling.
- DAG executor pre-resolves eager dependencies before spawning nodes. Static analysis partitions context keys into *eager* (definitely needed) and *lazy* (behind unknown branches), so only confirmed dependencies are prefetched. As each dependency resolves, the known-value set grows and may reveal new eager dependencies through branch pruning.
- The orchestration layer defines scheduling and the `Fetch` / `Storage` traits but owns no async runtime. The caller provides the runtime — tokio on servers, browser event loop in WASM, anything else elsewhere.
- The core crates (parser, type checker, MIR, interpreter) are pure computation with zero async dependencies. They compile to WASM as-is.
- Runtime errors propagate as `Result` through the coroutine protocol (`Stepped::Error`), not panics. WASM-safe — no unwind required.

## Architecture

```
acvus-ast               Parser (lalrpop + logos)
acvus-mir               Type checker + MIR lowering
acvus-mir-pass          Analysis passes (ValDefMap, reachable context)
acvus-mir-cli           CLI tool for MIR inspection
acvus-mir-test          MIR snapshot tests (insta)
acvus-interpreter       Runtime values, sync execution, RuntimeError
acvus-interpreter-test  Interpreter e2e tests
acvus-utils         Async coroutine primitive (Stepped pattern, error propagation)
acvus-ext               Extension modules (regex, etc.)
acvus-orchestration     Compilation, DAG builder, provider abstraction, Fetch / Storage traits
acvus-chat              Chat engine — multi-turn LLM orchestration, tool calls
acvus-chat-cli          Chat CLI — TOML project, multi-provider HTTP
acvus-playground        Web playground (axum)
```

## Template language

Three reference kinds:

- `@name` — Context. Externally injected, read-only. Types declared at compile time.
- `$name` — Variable. Local mutable. Type inferred at first assignment.
- `name` — Value. Local immutable, SSA-resolved at lowering.

String is the only emit type. Explicit `to_string` required for non-string values. Int + Float arithmetic is a type error — explicit conversion needed.

Pattern matching supports list, object, tuple, and range destructuring. `{{_}}` catch-all is required on all match blocks. Tuples are fixed-length and heterogeneous. Lists are homogeneous. Object matching is open (subset).

Format strings interpolate expressions inline: `"hello {{ name }}, you are {{ age | to_string }} years old"`. The `{{ }}` delimiters open an expression context — any valid expression works inside, including pipes. The parser desugars this to a `+` (string concat) chain.

Pipe chains work as first-class function application: `a | f(b)` desugars to `f(a, b)`.

### Builtin functions

List operations: `filter`, `map`, `pmap`, `find`, `reduce`, `fold`, `any`, `all`, `len`, `reverse`, `flatten`, `join`, `contains`.

Type conversions: `to_string`, `to_float`, `to_int`, `char_to_int`, `int_to_char`.

String operations: `contains_str`, `substring`, `len_str`, `trim`, `trim_start`, `trim_end`, `upper`, `lower`, `replace_str`, `split_str`, `starts_with_str`, `ends_with_str`, `repeat_str`.

Byte operations: `to_bytes`, `to_utf8`, `to_utf8_lossy`.

Other: `unwrap`.

Extension modules (e.g. `acvus-ext`) can register additional extern functions at compile time.

## Orchestration

A project is a directory of TOML files and `.acvus` templates:

```
my-project/
  project.toml        # providers, context, node file list
  summarizer.toml     # node definition
  reviewer.toml       # node definition
  system.acvus        # template
  user.acvus
```

Nodes declare a provider, model, and message templates. The orchestrator compiles all templates, builds a dependency DAG from context references, and executes with fuel-limited API calls.

Multiple providers (OpenAI, Anthropic, Google) are supported per project. Each node picks its provider independently.

### Error handling & retry

Runtime errors (LLM fetch failures, parse errors, builtin errors) propagate as `RuntimeError` through the coroutine protocol — no panics, no unwind. Each node can specify `retry` (max retry count) and `assert` (a Bool script evaluated after execution). If the assert fails, the node is retried:

```toml
[node]
name = "chat"
retry = 3
assert = '@self | len_str > 0'
```

### Storage

The `Storage` trait is generic — the caller injects the storage backend. `HashMapStorage` (in-memory) is provided out of the box. Custom backends (Redis, database, etc.) implement the trait and are passed to `ChatEngine::new`.

## WASM

The core pipeline has no tokio, no reqwest, no OS dependencies. `acvus-ast`, `acvus-mir`, `acvus-mir-pass`, and `acvus-interpreter` compile to `wasm32-unknown-unknown` directly. `acvus-orchestration` produces `BoxFuture` values — the caller supplies the executor and fetch implementation for their platform. Runtime errors use `Result`, not panics, so `panic=abort` (WASM default) is safe.
