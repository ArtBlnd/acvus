# Tooling

The command-line runner is a host like any other. This document decides its
contract: what it runs, where a context's type comes from, and what goes to
stdout, stderr and the exit status.

## RFC-0031: `acvus` runs one file as one function, the result alone on stdout

Status: Accepted

1. `acvus run <file>` compiles and executes; `acvus check <file>` compiles and
   reports; `acvus mir <file>` prints the MIR; `acvus ops <file>` prints the
   prepared operations. The extension names the mode — `.acvus` a script,
   `.acvt` a template (RFC-0071) — and `-e <expr>` runs an expression. The mode
   chooses the pipeline, not the grammar (RFC-0045).
2. `--context <file.json>` is data: each top-level key is a context, and its
   type is the value's type. A value whose type the data does not fix — an
   empty array, `null` — is an error, not a guess.
3. After a run, every context write is reported on stderr as
   `write @name = <json>`; `--commit` rewrites the context file with the new
   values.
4. `--bind name=<json scalar>` binds the input `$name` to a constant
   (RFC-0071 rule 5).
5. stdout carries the result and nothing else: a template's text as it is, a
   script's value as JSON, a bare string without quotes.
6. Every diagnostic — parse, inference, lowering, validation — goes to stderr
   in one span-rendered shape, all at once. A run-time failure is a panic,
   caught at the top of the process and printed as `error: <message>` with no
   span (RFC-0044).
7. Exit status: 0 on success, 1 when compilation fails, 2 when the run fails,
   64 for a usage error.
8. Which registries the runner registers is this host's choice, as it is any
   host's.

**Why.** One runner with one diagnostic shape is what a script author reads
first, and the same rendering serves the LSP. The context's type is the
data's type: what the script reads is what the file holds.

**Rejected.**
- Guessing a type where JSON is silent — a guessed type is one the script did
  not ask for.
