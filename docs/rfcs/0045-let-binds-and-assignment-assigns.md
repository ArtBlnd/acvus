# RFC-0045: `let` binds, `x = e;` assigns

Status: Accepted
Date: 2026-09-18
Extends: RFC-0018, RFC-0024, RFC-0031

## Ruling

There is one statement grammar, and it is the grammar of every block: a
script's top level, a lambda's block body, a `while`/`while let`/`anyorder`
body, a tag-form match-bind body, an `if`/`else` block.

- `let x = e;` introduces a binding. It shadows any outer `x`, and it ends
  with the block that introduced it.
- `let x;` introduces an uninitialized binding; the init check owns it.
- `x = e;` assigns the `x` in scope. It never introduces one. The assigned
  expression's type unifies with the binding's; a mismatch is the existing
  type error.
- An assignment to a name with no binding in scope is a compile error:

  ```
  cannot assign to `x`: no binding named `x` is in scope; `let x = ...;` binds it
  ```

- An assignment inside a lambda to a name bound outside it is a compile
  error:

  ```
  cannot assign to `x`: it is captured by the lambda, not bound in it
  ```

- A name assigned inside a `while`, `anyorder`, `if` or match-bind body is
  the binding the enclosing block introduced, so the new value is live
  after the body: the join carries it as a block parameter, the same path
  the `if`/`else` expression's arms already take.

`@ctx = e;`, `a.b = e;` and `*r = e;` are unchanged, and are statements of
the same one rule.

## Rationale

Before this, the language had two statement rules. `ScriptStmt` — the rule
of `parse_script`, of a lambda's block body, and of a tag-form body — read
a bare `x = e;` as `Stmt::Bind`, a fresh shadowing binding. `ScriptModeStmt`
— the rule of `parse_script_mode`, of `if`/`else` blocks and of `while`
bodies — read the same line as `Stmt::Assign`, a store into the binding in
scope. The same text meant a binding in one position and an assignment in
another, and which one it meant depended on which entry point had parsed
the file and how deep in the file the line sat.

The visible cost was that no tag-form body could report anything to its
enclosing block except through a context: `out = v;` in the body bound the
body's own `out`, which ended with the body, so `let out = 0.0; Some(v) =
Some(1.5) { out = v; }; out` was `0.0`. It is `1.5` now.

One rule removes the position from the question. `let` is the only way to
introduce a name, so the reader of a body knows every name it introduces by
reading the `let`s, and the writer of `x = e;` knows the store lands where
the `x` they can see was bound.

The two entry points differed only in their statement rule, so there is now
one grammar. `parse_script_mode` is the same parser as `parse_script`. The
CLI's `Mode` still names what the *file* is — `.acvus` a script, `.acvt` a
template, `-e` an expression — which is a choice of pipeline (script versus
template lowering, and what the tail means), not a choice of statement
grammar; the flag stays.

### The capture assumption

Refusing an assignment to a captured name rests on captures being by value
(RFC-0018): the closure owns the captured value, so a store inside the
lambda would write that copy and never reach the binding the writer named.
Rust refuses the same assignment without `mut` and `FnMut`. This RFC does
not build capture-by-reference; if the owner wants an assignment inside a
lambda to reach the outer binding, that is a capture-mode decision, and
this refusal is where it will surface.

## Rejected

- **Bare `x = e;` binds where no `x` is in scope, and assigns where one
  is.** This keeps every existing script compiling and needs no `let`. It
  is rejected because it is implicit: whether a line introduces a name
  depends on everything above it, a typo in a name becomes a new binding
  instead of an error, and a `let` moved or removed silently changes a
  store into a shadow. The owner's rule is explicit.
- **Keeping the two rules and making only the tag-form body an assignment
  position.** A patch at the symptom: the two rules would still disagree
  about every other block.

## Consequences

- `Stmt::Bind` is removed from the AST. Its consumers each became the
  `LetBind` or `Assign` case, or were removed.
- The grammar has one `Stmt` rule and one `pub Script` entry.
- `MirErrorKind::AssignToUnbound` and `MirErrorKind::AssignToCapture` carry
  the two texts above.
- The type checker resolves an assignment's target itself rather than
  through `lookup_var`: `lookup_var` records a capture, which is exactly
  what an assignment must refuse.
- Every script written against the old `ScriptStmt` rule needs `let` on the
  first store of each name. A template binding (`{{ x = expr }}`) is a
  `MatchBlock` with a `Binding` pattern, not a statement, and is untouched.
- A scrutinee that has no storage of its own is matched as a value, in the
  register that produced it. It is given a slot only where the pattern's own
  test reads a part out of it — `Some(Some(v))`, a tuple, an object, a list —
  because the interpreter's `read_slot` and `unwrap_*` move a `Large` out of
  the slot they read, so a part read to test and read again to bind would be
  one value taken twice. For `Some(v)` the test reads the register and the
  bind unwraps it once: `if let Some(v) = f()` loses its `assign $source` and
  `ref &$source`, and `take $source.payload` becomes `unwrap`. Measured at
  n = 1e6, medians of three alternating repetitions: `option while`
  14.8 → 11.8 ns/iteration, `while let vec` 18.0 → 14.0, `while let map`
  18.2 → 15.5; two dispatches per iteration gone in each.
