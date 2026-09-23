# Identity

Identity says which values come from one source, so that parts of one source
are never mixed with parts of another. This document decides where identity
sits in the type language, when two identities agree, and where a new source
begins.

## RFC-0012: Identity is a parameter of a user-defined type

Status: Accepted

1. Identity is a kind of type parameter, like effect. A user-defined type
   declares whether it has one, and has at most one: a value is one source.
   A type expression fills it with an identity argument. There is no identity
   type, and no structural type carries an identity. Two values may each carry
   their own source, as a closure's captures do; one value never carries two.
2. An identity argument names a source and nothing more: two parts of one
   source share its identity, and the identity does not say which part is
   which.
3. Two identities unify only when they are the same. No join erases identity.
   A derived value keeps the source it came from or is a new source; no
   operation takes a value back to the sources it was made from.
4. An identity variable that nothing ties to a source is a source of its own
   when it freezes. An ExternFn whose return type has an identity variable
   found in none of its parameters returns a new source at every call; one
   that carries the variable from a parameter returns the source it was given.
   A local function that returns a source returns the same source at every
   call: its returned identities are not generalized.
5. Identity lives in the compiler; the runtime sees none of it. A rejoin of
   parts is a rewrite the compiler applies at the operation that split them,
   where their order is visible; a shared identity is a precondition of that
   rewrite and never its justification.
6. A source number names one source for a whole compilation: every solver of
   a compilation mints from one supply, so a frozen type passed between
   solvers still names the source it was frozen with.
7. A declaration names no source. A host declaring a context or parameter from
   a concrete type lifts it, turning every identity argument into a variable
   the compilation mints a source for.

**Why.** As a parameter kind, identity sits where effect already sits and is
solved in the shape type, effect and length variables are. A source number
carried in from outside would collide with the compilation's own.

**Rejected.**
- Identity as a type of its own — it could sit in any position, including a
  struct field, and would need a runtime notion of provenance to mean
  anything there.
- Two identity parameters on one type — remembering two sources only serves
  splitting the value back into them, which is the operation that can put
  parts back in the wrong order.
- Erasing identity at a join (a least upper bound) — a script that wants two
  sources in one place joins them through a function that declares a new
  source.
- Runtime provenance — rejoin is code the compiler emits.
