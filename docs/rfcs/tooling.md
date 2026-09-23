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

## RFC-0078: A parse recovers past an error, and a tree that holds one cannot be lowered

Status: Accepted

1. A parse does not stop at its first error. A statement, an expression or a
   pattern that does not parse becomes an error node over its span, parsing
   resumes after it, and every error is reported, all at once (RFC-0031
   rule 6).
2. A character the lexer cannot read is an error token the grammar recovers
   over, not the end of the parse.
3. In a template (RFC-0071), a `%` line that does not parse is one error
   statement; a tag that does not parse is one error expression, and a tag
   that does not close runs to the end of its line and is reported; a block
   no `% end` closes is closed at the end of the source and reported.
4. The tree is parameterized by what an error node holds. A parse without
   errors yields the tree whose error slot is an uninhabited type; a parse
   with errors yields the tree whose error nodes hold their id and span,
   together with the errors.
5. The checker takes either tree. An error node's type is poison, so what
   parsed around it is checked without refusals that only repeat the parse
   error; a body that holds an error node is refused by its parse errors.
6. Lowering takes only the tree whose error slot is uninhabited, so a tree
   that holds an error node has no path to the machine. A match in lowering
   answers the error case by eliminating that slot, which the compiler
   checks; none answers it with a panic.
7. A batch path and an editor parse the same way, so `acvus check` and the
   editor report the same errors for the same source.
8. An editor keeps a document that does not parse in its graph: hover,
   definition and completion answer from the recovered tree, and the
   diagnostics are the parse errors and the refusals of what parsed.

**Why.** An editor sees a source that does not parse most of the time: a
statement being written has no `;` yet, and any broken line elsewhere made
the whole document unreadable, so hover, definition and completion answered
nothing exactly when they were needed. A tree that keeps what parsed answers
them. Carrying the error slot in the type makes "an erroneous tree never
reaches the machine" a fact the compiler checks rather than a convention every
match in lowering has to keep.

**Cost.** Every tree type that holds an expression, a statement or a pattern
takes the parameter, and the checker is instantiated for both trees. The
grammar carries recovery points, and each has to leave a tree whose spans
still cover the source they claim.

**Rejected.**
- An error variant without a parameter, with lowering fed a wrapper that
  promises no error node: every match in lowering still names the error case
  and answers it with `unreachable!`, a convention in place of a type.
- Converting a recovered tree into a clean one by a fold: a second walk of the
  whole tree that exists only because the parser did not say which tree it
  made.
- Repairing the probe source for completion by inserting the closing tokens
  the parser expected: it answers completion near the cursor only, leaves a
  broken line elsewhere fatal, and leaves hover and definition without a tree.
