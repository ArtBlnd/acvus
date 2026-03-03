# acvus

A template engine for multi modal orchestration. Templates compile to typed intermediate representation, execute as a DAG, and run anywhere — no runtime dependency required.

## What it does

- Pattern matching as the only control flow. No if/for/while. All variables immutable.
- Static type system catches errors at compile time. No silent failures at runtime.
- Templates compile through a full pipeline: `Template → AST → Type Check → MIR (SSA) → Execution`.
- MIR enables static analysis passes — dead branch pruning, reachable context analysis, dependency-aware scheduling.
- DAG executor runs independent nodes in parallel. Static analysis determines which dependencies are actually needed on live code paths, so nodes can launch early when dead branches hide their dependencies.
- The orchestration layer defines scheduling and the `Fetch` trait but owns no async runtime. The caller provides the runtime — tokio on servers, browser event loop in WASM, anything else elsewhere.
- The core crates (parser, type checker, MIR, interpreter) are pure computation with zero async dependencies. They compile to WASM as-is.

## Architecture

```
acvus-ast               Parser (lalrpop + logos)
acvus-mir               Type checker + MIR lowering
acvus-mir-pass          Analysis passes (ValDefMap, reachable context)
acvus-mir-cli           CLI tool for MIR inspection
acvus-mir-test          MIR snapshot tests (insta)
acvus-interpreter       Runtime values, sync execution
acvus-interpreter-test  Interpreter e2e tests
acvus-orchestration     Compilation, DAG builder, provider abstraction, Fetch trait
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

Pipe chains work as first-class function application: `a | f(b)` desugars to `f(a, b)`.

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

## WASM

The core pipeline has no tokio, no reqwest, no OS dependencies. `acvus-ast`, `acvus-mir`, `acvus-mir-pass`, and `acvus-interpreter` compile to `wasm32-unknown-unknown` directly. `acvus-orchestration` produces `BoxFuture` values — the caller supplies the executor and fetch implementation for their platform.
