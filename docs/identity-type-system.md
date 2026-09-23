# Identity

Identity says which source a value came from. It is a **parameter of a
user-defined type** — a fourth kind of parameter beside type, effect and
length — and nothing else in the type language carries one. The ruling is
[RFC-0012](rfcs/identity.md); this document is how it works
and why it exists.

## Why the type system needs it

The language is structurally typed. `{ role: String, content: String }` is
not a type declared somewhere: it is the structure, and any value with those
fields has that type, wherever it came from. Two objects that meet join to
the union of their fields (RFC-0042).

That leaves no way to tell apart two structurally identical values from
different sources — an LLM response and a user input with the same two
fields are the same type. Identity is what tells them apart, without adding
a declaration site: the distinction rides on the type, and the compiler
propagates it through unification.

## What an identity is

A user-defined type declares whether it has an identity parameter, at most
one (`UserDefinedDecl::identity_params`). A type expression fills it with an
identity argument (`TyTerm::UserDefined { identity_args }`). There is no
identity type, no structural type is tagged, and the runtime sees none of
it: what the compiler proved about sources is spent before the code runs.

An identity argument names a source and says nothing more. Two parts of one
source share its identity, and the identity does not say which part is
which. A derived value keeps the source it came from or is a new source; no
operation takes a value back to the sources it was made from.

A source number names one source for a whole compilation. Every solver a
compilation runs mints from one `Sources`, so a frozen type that passes from
one solver to another — an earlier SCC's result, a cached signature — still
names the source it was frozen with, and no later solver mints that number
again. A host's declaration names no source: `lift_declaration` turns every
identity argument into a variable, and the compilation mints for it.

## The two rules

**Unification is invariant.** `Solver`'s `unify_identity` succeeds between
two known identities only when they are equal, and binds a variable to
whatever it meets. There is no least upper bound that forgets the
difference: two values with different sources do not unify, and a script
that wants both in one place joins them through a function that declares a
new source.

**An identity variable nothing tied to a source becomes one.** At the end of
`Solver::solve`, every identity variable still unbound is bound to a fresh
source. So an ExternFn whose return type carries an identity variable found
nowhere in its parameters returns a new source at every call, and one that
carries the variable from a parameter to its return returns the source it
was given:

```
split(s: &String) -> (StringRef<I>, StringRef<I>)
```

Both results carry the caller's `I` — one source, two parts.

If unification were covariant instead, two different sources would merge
silently and the provenance would be lost with no site to point at.
Invariance forces the merge to be written.

## What follows from it

These are consequences of unification and the move rule, not separate
features.

**Provenance at the type level.** Where every IO boundary returns a type
with a fresh identity, a value's origin is in its type, and values from
different sources cannot be passed for one another. The site where a script
joins them through a declaring function is the place that says "I accept
data from mixed sources".

**A rejoin the compiler can prove.** Two parts of one source share its
identity, which is what makes a rejoin of them a rewrite rather than a copy.
The rewrite is applied at the operation that split the parts, where their
order is visible; the shared identity is its precondition and never its
justification. Nothing is decided at run time.

**A declared source for the move rule.** Identity is what RFC-0012 gives the
move rule to stand on. In the tree today the rule is wider than that: every
`UserDefined` type is move-only (`validate/move_check.rs`, `is_move_only`),
identity parameter or not.

## What is not built

- **No type with two identity parameters.** Remembering two sources in one
  value would serve only taking the value apart into them again, which is
  the operation that can put parts back in the wrong order.
- **No erasure at a join.** No least upper bound forgets an identity.
- **No runtime provenance.** No tag, no check, no branch.
- **No identity on a structural type.** An object or an enum carries none.
- **No generalization of a local function's returned identities.** A local
  function that returns a source returns the same source at every call.
