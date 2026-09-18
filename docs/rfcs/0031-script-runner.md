# RFC-0031: `acvus`, the script runner

Status: Accepted
Date: 2026-09-16
Supersedes: acvus-mir-cli

## Ruling

`acvus` runs one file as one function. `acvus run <file>` compiles and
executes it; `acvus check <file>` compiles it and reports; `acvus mir
<file>` compiles it and prints the MIR. The file's extension names its
mode: `.acvus` is script mode, `.acvt` is a template; `-e <expr>` runs an
expression.

`--context <file.json>` is data: each top-level key is a context, and its
type is the value's type. A value whose type the data does not fix — an
empty array, `null` — is an error, not a guess. After a run, every write
to a context is reported on stderr as `write @name = <json>`; with
`--commit` the context file is rewritten with the new values.

stdout carries the result and nothing else: a template's text as it is,
a script's value as JSON with a bare string printed without quotes.
Every diagnostic goes to stderr in one shape:

```
error: no instance of the signature has the call type Fn(Deque<Int>) -> Int
  --> classify.acvus:4:12
   |
 4 | let n = len(d);
   |         ^^^^^^
```

Parse, inference, lowering, and validation errors are all reported, all
at once, in that shape. A run-time failure is not a diagnostic: it is a
panic, caught at the top of the process and printed as `error:
<message>` with no span (RFC-0038). Exit status: 0 on success, 1 when
compilation fails, 2 when the run fails, 64 for a usage error.

The standard registries and `acvus-ext-net` are always registered; the
LLM registries join with `--llm`, reading their keys from the
environment. Execution is sequential unless `--parallel`.

## Rationale

The pieces existed apart: the LSP turned errors into spans without lines,
the MIR CLI took typed context declarations it had to invent a notation
for, the test crate inferred context types from JSON and guessed where the
data was silent, and a runtime error said what failed but not where. One
runner with one diagnostic shape is what a script author reads first, and
the same rendering serves the LSP.

The context's type is the data's type: what the script reads is what the
file holds. The two places JSON cannot say what a value is are errors
because a guessed type is a type the script did not ask for.

## Not built

- No spec or namespace of many functions; that is the orchestration
  session's unit, reachable from a later subcommand.
- No declared context types: an extension type (`List`, `Option`, `Deque`)
  cannot be a context from JSON yet.
- No result rendering for extension types: a value of one prints as its
  type name.

## Consequences

- `acvus-ast::report`: a line index over a source and the rendering of a
  diagnostic at a span, shared by the CLI and available to the LSP.
- The CLI wraps its `block_on` in `catch_unwind` and prints a run-time
  panic as `error: <message>`.
- `acvus-cli` is the crate; `acvus-mir-cli` is removed.
