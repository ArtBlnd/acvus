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
   chooses the pipeline, not the grammar (RFC-0045). `acvus lsp` serves an
   editor over stdio (RFC-0086) and takes no file.
2. `--context <file.json>` is data: each top-level key is a context, and its
   type is the value's type. A value whose type the data does not fix — an
   empty array, `null` — is an error, not a guess.
3. After a run, every context write is reported on stderr as
   `write @name = <json>`; `--commit` rewrites the context file with the new
   values.
4. `--bind name=<literal>` binds the input `$name` to the value a literal
   writes (RFC-0087 rule 1), in the script's own syntax, and a `-` before a
   float literal is its sign. Anything else — a name, a call, an operator —
   is a usage error.
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
- JSON for `--bind` — it has no enum, tuple, `char` or `Option`, and a
  variant written as an object could not be told from an object.

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
still cover the source they claim. An LR grammar recovers by dropping what it
cannot reduce, not by supplying a missing closer, so a construct the source
ends inside, such as a call whose `(` is never closed, becomes one error node,
and nothing inside it answers an editor.

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

## RFC-0084: A workspace lists its compilations again exactly when a file its listing read changes

Status: Proposed

1. A host lists its compilations through a reader that records what the
   listing read: each file it read, whether the read found it or not, and
   each directory it listed, with the kind — file, directory or absent — of
   every entry the listing saw.
2. A change to a path lists the compilations again when the listing read that
   path, or when it changes an entry of a directory the listing listed: an
   entry that appears, vanishes or changes kind. A listing that failed is
   changed by its directory and by any path under it.
3. Otherwise a change rereads and rechecks the compilations that hold the path
   as a document, and a path neither read nor held changes nothing.
4. A document the listing also read, such as a script whose value types
   another compilation's environment, lists again on its change, and is a
   document with hover, definition and completion like any other.
5. The editor's buffers overlay the disk for every read and every listing: a
   buffer is a file where the disk has none.

**Why.** What an environment depends on is what building it read. Recording
the reads leaves no dependency for a host to forget: a host that reads more
than it needs lists again more often, and never answers from a stale
environment.

**Cost.** A host reads and lists only through the reader; a file it reads
around the reader is a dependency the workspace does not see. A document the
listing read lists every compilation again on each edit.

**Rejected.**
- A flag by which a host declares that a document feeds environments: a
  declaration a host can omit, and the omission answers from a stale
  environment with no sign of it.
- Listing again on a change to any path no compilation holds: every unrelated
  save rebuilds every environment, and a document the listing read is still
  rechecked in place against the old one.

## RFC-0085: In the editor a host answers as its batch path does

Status: Proposed

1. The workspace reads a document through the host, which refuses a document
   it cannot read in the words its batch path uses.
2. A host's rules read, for each document of an accepted compilation, the type
   the checker solved the body's value to, beside the function type it
   inferred. Under a declared `!` (RFC-0054 rule 5) the function type states
   no return, and this is where a host reads the value's type.

**Why.** An editor and a batch path that word one failure two ways are two
contracts for one source (RFC-0078 rule 7). A host that needs the type a body
returns, while stating none, would otherwise declare a fresh variable: a second
spelling of "no stated return" that RFC-0054 rejects.

**Cost.** Every host implements the read, even one whose batch path reads a
file as the workspace would.

**Rejected.**
- A fixed wording hosts are told to match: a convention no check keeps.
- A hook that words only the error: the read and its refusal are one step of
  the batch path, and a host that reads a document otherwise than from the
  bytes of its file would still differ.

## RFC-0086: `acvus_lsp::serve` speaks the Language Server Protocol for any host

Status: Proposed

1. `serve` answers the protocol over one connection for the workspaces of
   any host, one workspace per folder the client serves, each with the host
   made for its root. `acvus lsp` serves it over stdio with the command-line
   host.
2. The workspace speaks byte offsets. Positions are converted at the protocol
   boundary, in UTF-8 where the client offers it and in UTF-16 otherwise. A
   document's positions are converted over the text its compilations parsed,
   read once per change for every compilation that holds it; any other path's
   over the text the workspace reads for it.
3. Diagnostics are published per path. A path published with diagnostics that
   the next answer does not hold is published empty, so the workspace answers
   only the paths that have diagnostics.
4. A refusal and a parse error carry the span the checker and the parser gave
   them, and a span at offset 0 is a place. A host diagnostic without a span
   is placed at the start of its file.
5. A completion replaces the identifier the cursor is in, over the range the
   workspace answers with it.
6. A refused rename is an error in the refusal's words, answered already when
   the client prepares the rename.
7. A request is answered by the workspace whose root is the longest prefix of
   its path, and a path under no root is answered with nothing. A buffer and a
   file change reach every workspace whose root holds the path. The published
   diagnostics are those of every workspace together, so a folder the client
   removes has its paths published empty unless another folder still reports
   them.
8. A host's own files navigate by the links its listing gives, each from a
   span of a path that is no document to a place. Definition in such a path
   answers the link whose span holds the offset, the narrowest where spans
   nest, else one ending at the offset, as a name at the cursor is chosen. A
   link from a document is refused at the listing, since a document's names
   are the checker's. The references of a context or an input hold the links
   that lead to its site.

**Why.** Every host with an editor needs the same boundary: positions,
publishing and the mapping of answers. One implementation keeps the
workspace's contract the only thing a host writes.

Links are data the listing carries, so they are as current as the listing
the file they are read from belongs to (RFC-0084).

**Cost.** acvus-lsp depends on a protocol crate and its types, and a host
cannot shape a protocol answer beyond what the workspace gives: navigation in
its own files is only the links it lists. Each folder builds its own
environments.

**Rejected.**
- A protocol layer per host: each repeats the position conversion and the
  clearing of published paths, and each can get them wrong on its own.
- One host given every root: a folder the client adds needs a host method to
  change roots, and one folder's broken listing refuses the others.
- A navigation callback on the host: it answers from whatever view the host
  keeps, which the workspace cannot keep current.
- UTF-16 offsets inside the workspace: every query pays for an encoding only
  the boundary needs.
- An empty entry for every document in the workspace's answer, so a client can
  clear it: a path that is no document, such as a host's manifest, or a
  document no compilation lists any longer, is still left uncleared.
