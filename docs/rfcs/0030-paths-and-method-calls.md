# RFC-0030: A qualified call names a namespace; a method call is a call on its receiver

Status: Accepted
Date: 2026-09-16
Extends: RFC-0021, RFC-0027

## Ruling

`a::b(args)` is a call of the function `b` in the namespace `a` when the
registries declare one (RFC-0021); otherwise it is the structural variant
`b` of the enum `a` with `args` as its one payload, as it was. `a::b`
without arguments stays a variant. A bare name `b(args)` resolves as
before: the script's own function, else the one function of that name
under any namespace.

`recv.f(args)` is the call `f(recv', args)`: `f` resolves as the bare
name `f` would, and `recv'` follows the type of `f`'s first parameter —
`&recv` for `&T`, `&mut recv` for `&mut T`, in both cases `recv` must be a
place; `recv` itself for `T`. Nothing else changes: the call is checked,
lowered, and borrow-checked as the call it stands for, and a chain
`xs.as_iter().map(f).collect()` is the pipeline written with references
where the callee asks for them.

`recv.f` without parentheses is a field, as before; a closure held in a
field is called as `(recv.f)(args)`.

## Rationale

Nothing in the grammar named a namespace: `std::len(&xs)` parsed as the
variant `len` of an enum called `std`, and the checker accepted the enum
because a structural enum needs no declaration. Namespaces are closed when
the registries are combined, so the checker can tell a namespace from an
enum name by looking, and the same expression means one thing in the
checker and in the lowering.

`len(&xs)`, `get(&d, 0).x`, `as_iter(&xs) | map(f) | collect` are the
calls the method form writes as `xs.len()`, `d.get(0).x`,
`xs.as_iter().map(f).collect()`; what the method form adds is the
reference the callee's own signature already asks for. Choosing the
receiver mode from the first parameter, as Rust does, keeps one rule for
signatures with instances (RFC-0027), whose first parameter is `&C` or
`C` before any argument is seen.

## Not built

- No method resolution by receiver type: `f` is found by name, and the
  receiver picks the instance only through the call's type as any
  argument would.
- No auto-deref chains: `r.f()` where `r: &T` and `f` takes `T` is a call
  with a reference where a value is needed, reported as such.
- No calling a closure held in a field without parentheses around the
  field.

## Consequences

- Grammar: `ident::ident(args)` is a call of a qualified identifier with
  any number of arguments; `postfix.ident(args)` is `Expr::MethodCall`.
- The checker resolves a qualified callee as a function first and a
  variant second, and records the callee it chose for the lowering.
- The lowering borrows a method receiver when the callee's first
  parameter is a reference, exactly as it lowers `&place` as an argument.
