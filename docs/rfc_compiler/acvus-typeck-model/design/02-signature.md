# 02 Types in declarations

Rests on the spec's l.54–55 (structs and extern-defined types), l.64 (what is
reached through a reference), l.322 (an extern declaration's type parameters) and
l.349 (an existential identity). The spec takes a declaration's form from the RFCs:
extern fn §1 (a signature: parameters and a result, over type parameters) and §2
(a declaration names one identity among its variables). The code is `src/sig.rs`,
and `src/sig/tests.rs` exercises its contract.

## Meaning

A declaration writes types over its own variables: its type parameters and, where
it names one, its identity. A signature is the types of a declaration's parameters
and of its result. Substituting closed types for a signature's variables gives
closed types, and matching a closed type against a signature's type finds them.
Two kinds of type are declared: an extern-defined type, by its number of arguments
and whether it carries an identity (l.55, Example 11), and a module's struct, by its
fields in declared order (l.54).

## Not owned here

- Which declarations a program has, their names, casts and conversions, and the cast
  tree, are note 03's.
- Matching against a type that is not closed is note 06's and after. A use's type
  parameters are unknowns made anew at each use (l.322), so a signature meets
  equivalence classes there.
- Effects, flows and declared facts have no representation, since no decision the
  spec defines reads them.
- A type parameter has no bound. Extern fn §1 states one, and the spec states none;
  the model follows the spec (`holes.md`, T1).

## Readings

1. **A reference substituted under a reference.** A signature may write `&T` or
   `&mut T`, and a use may make `T` a reference. No type has a reference directly as
   its target, and what is reached through a reference is reached by one reference,
   exclusive only where both are (l.64); a lend of a reference does the same (l.298).
   So such a substitution gives that one reference: `&T` and `&mut T` at `T = &U` are
   both `&U`, and `&mut T` at `T = &mut U` is `&mut U`. Matching binds a variable
   directly under a reference to the reference's target, which is never a reference,
   so substituting what matching found gives back the type matched.
2. **No function value.** A declaration makes no function value: it returns one
   only as a type variable takes it in, as `T -> T` does (l.186). So a signature
   writes none at any depth, and `Signature::new` refuses one
   (`SigError::FunctionValue`); the text and a context write none either
   (`ProgramBuilder::written`, `host`).

## Left open for change

- `DeclaredTypes::check` takes a `Term<P>` and looks at no part, so closed types,
  signatures and kinds share one check.
- `SigArgs` binds some of the variables, so matching can bind them one parameter at
  a time.
- Substitution and matching take any `HasTerm`, so a signature also meets a value
  a use reads (note 05). A term the signature writes does not match a position the
  value leaves undecided.
