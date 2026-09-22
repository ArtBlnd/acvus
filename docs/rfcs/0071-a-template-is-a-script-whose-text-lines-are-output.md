# RFC-0071: a template is a script whose text lines are output

Status: Proposed
Date: 2026-09-22
Supersedes: none

## Ruling

A template file (`.acvt`) is a script with one extra rule: a line that
does not begin with `%` is text, appended to the result as written. A
line whose first non-blank character is `%` is one statement of the
script grammar, without its trailing `;` and without the braces that
would open or close a block. A block opened on a `%` line is closed by
`% end`. The template's value is the accumulated text, a `String`.

    You are a careful assistant for {{ $name }}.
    Answer in {{ if $lang == "ko" { "Korean" } else { "English" } }}.

    % if $mode == "review"
    Review the code below and list defects.
    % else if $mode == "explain"
    Explain the code below.
    % else
    Help with the code below.
    % end

    Recent turns:
    % for m in &$journal
    - {{ &m.role }}: {{ &m.content }}
    % end

    % match $mode
    % "review" =>
    Review the code.
    % _ =>
    Help.
    % end

    {{ rules() }}

1. **A `%` line is a statement and leaves nothing in the output**, its
   newline included. The statements are those of the script grammar:
   `let`, assignment and stores, `if` / `else if` / `else`, `if let`,
   `match` with one `% pattern =>` line per arm, `for` with the four
   heads of RFC-0057, `while`, `while let`, `anyorder`, `break`,
   `continue`, and an expression statement. `% //` is a comment, as `//`
   is in a script. The `%` may be indented.

2. **A text line is appended as written**, with its newline. A text line
   that ends in `\` is appended without that newline. A text line that
   begins with `%%` is appended with one `%` in its place. There is no
   other whitespace rule.

3. **`{{ expr }}` in a text line is the format string of RFC-0062.** The
   expression is a `String` or a `&str`; nothing is converted to text
   implicitly. The tag's content is tokenized as an expression, so a
   string literal inside it may hold `{{` or `}}`: `{{ "{{" }}` writes a
   literal `{{`.

4. **`$name` is an input the host injects, shared by the whole graph.**
   It is not a parameter of the template: a call passes nothing,
   `{{ rules() }}`, and `$lang` inside `rules` is the same injected value
   as `$lang` in its caller. `@name` is a context, as in a script. There
   is no include and no parameter declaration: the inputs a template
   requires are the `$` names its reachable code reads, and the analysis
   shows them.

5. **A bound `$` is a constant, and what it makes unreachable is `!`.**
   Binding `$mode = "review"` folds `% if $mode == "review"`; the arms
   it leaves behind are typed `!`, a `$` read only there constrains
   nothing and closes to `!` at the freeze (RFC-0038), and a `$` typed
   `!` is not required. Binding one input therefore narrows the set of
   inputs still required, and never widens it. A template whose value is
   `!` — every text line unreachable, or a `% return` reached — appends
   nothing and is absent from what its caller assembles.

The template forms `{{ pattern = expr }}`, `{{ pattern = }}`, `{{_}}`,
`{{/}}`, `{{/+N}}`, `{{/-N}}`, `{-{ }-}` and `{{-- --}}` are removed.

## Rationale

A template is string composition, and string composition is what the
script already lowers, analyzes and optimizes. Giving the template its
own statement forms gave it a second grammar to learn and a second
lowering to keep in step; RFC-0057 restored `for` to the script and the
template's iteration did not follow, which is the kind of drift a second
grammar invites. With a text line lowered as an append, every analysis
the script has — effects, reachability, what is constant across turns
— applies to a template unchanged, and orchestration reads those results
instead of asking the author.

Control flow on its own line is what removes the whitespace question.
An inline control tag has to decide who owns the spaces and newlines
around it, and every template language with inline control has grown
trim markers and global trim options to answer that. A `%` line owns
its whole line and nothing else, so the only case left is a newline the
author does not want, and `\` at the end of the line is that case.

A `match` arm is a line for the same reason it is ugly inline: a match
is a table, and a text line is already a row.

Inline branching stays available because `if` and `match` are operands
of the expression grammar: `{{ if c { "a" } else { "b" } }}` is a script
expression, not a template form.

`{{ }}` is kept for the inline expression, rather than `${ }`, because it
is the format string a script already writes: a text line is a format
string, and a template author who moves to a script carries the same
tag.

## Not built

- No inline block bodies (`@if[c]{text}{text}` in the Scribble style).
  It would make `{ }` mean text in one position and a block or object
  literal in another, and `@` is the context sigil.
- No trim markers and no trim options. A `%` line and `\` cover what the
  design needs; a second mechanism would reopen the question.
- No include or inheritance. A template is a function; composition is a
  call.
- No per-block closer (`endif`, `endfor`). A script closes every block
  with `}`; a template closes every block with `% end`.

## Consequences

- The lexer classifies lines, not tags: text, `%` statement, or a text
  line under `\` and `%%`. The parser feeds a `%` line to the script
  statement grammar with the block structure supplied by `% end`.
- A text line lowers as an append to the template's result; there is no
  template-specific node beyond that append.
- The tag scanner of the format string tokenizes its content, so a
  string literal containing braces is not a tag boundary. This applies
  to the script's format strings as well.
- `FunctionMeta.params` and `IncrementalGraph::context_info` report the
  `$` names a function's reachable code reads, at their types; a name
  whose type closed to `!` is not reported. Binding a `$` is a graph
  update, and only the functions that read it are re-inferred.
- The LSP shows the required `$` set of a document from `context_info`
  and narrows it as inputs are bound.
- `Grammar.md` is rewritten for the template section.

## Open questions

- Where the fold of a bound `$` runs: before typeck on the AST, or on
  the MIR with the required set read back from what survived.
