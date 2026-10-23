# 03 Declarations and the cast tree

Rests on the spec's l.76–78 (the cast tree), l.105 (a declared conversion) and
l.320 (the declarations a name reaches), with extern fn §5 (several declarations
share a name), §6 (an impl is a declaration of its generic function's name) and §7
(a cast or a conversion goes from its parameter's type to its result's type), and
types §7 (a conversion is decided by the slot's type). The code is `src/decl.rs`,
and `src/decl/tests.rs` exercises its contract.

## Meaning

A program's declarations are its extern fns and impls, each with a signature, each
reached by its name, an impl by its generic function's name. A declaration may be a
cast, a step up the cast tree, or a conversion, which is no part of the tree. A
node of the cast tree is a constructor, or for a reference its mutability and its
target's constructor, and the declared casts form a tree over the nodes (l.78).
Every declaration is also a function called by its name, a cast and a conversion
included.

## Not owned here

- Which declaration a use reaches, narrowing included (l.322–334), is note 08's.
  A declaration's overload index names it and orders nothing a decision reads.
- A generic function's own signature, and the requirements a declaration states
  (extern fn §6), have no representation, since no decision the spec defines reads
  them. A generic function called at a structural type's parts (l.320) is note
  08's.
- Join, and which casts a join takes, is note 04's; which casts and conversion a
  coercion takes is note 05's.

## Readings

1. **A cast's result has a node.** The nodes are constructors and a cast goes up
   from a node (l.78), so a cast's result is a node: its outermost term is not a
   variable, and a reference's target there is not one. A cast written otherwise is
   refused (`TreeError::NoTargetNode`).
2. **The cast that the tree commutes with exists.** Where a cast goes from
   `&mut X` to `&mut Y`, a cast going up from `&X` exists and goes to `&Y` (l.78).
   Read as holding vacuously where `&X` has no cast, the join stops being
   associative: `(&mut X ⊔ &mut Y) ⊔ &Y = &mut Y ⊔ &Y = &Y`, while
   `&mut X ⊔ (&mut Y ⊔ &Y) = &mut X ⊔ &Y = &X ⊔ &Y` (l.88), and `&X` and `&Y` have no
   common ancestor. A class's value joins its lower bounds in whatever order they
   come, so the vacuous reading makes it depend on that order. Coercion says the
   same: `reborrow? ; cast*` (l.109) reaches `&Y` from `&mut X` only through `&X`'s
   cast.
3. **A cast goes up from every type of its node.** Its parameter writes the node
   with distinct type parameters as its parts, through a reference's target where
   it is one, and its result writes those type parameters alone (l.78, since the
   text's change of 2026-10-22): `Vec<u8> → Bytes` is refused
   (`TreeError::NarrowerThanNode`), as is `A → Box<T>` (`ResultUnbound`), so a join
   the tree gives always raises (`holes.md`, H1).
3. **A cast steps up only from the types its parameter matches.** A cast's node is
   its parameter's constructor, and its parameter may still be narrower than the
   node (`Vec<u8>` at the node `Vec`), or its result may write a variable its
   parameter does not. At a type of that node which its parameter does not match,
   or whose step up is not determined, no cast steps up. The spec gives no result
   there (`holes.md`, H1).
4. **A conversion is decided by both of its types.** A conversion applies once the
   slot's type decides it (types §7), so a conversion goes from `T` to `U` where its
   parameter matches `T` and its result matches `U` by one binding of its
   variables.
5. **A cast and a conversion take one parameter.** Each goes from its parameter's
   type to its result's type (extern fn §7).
6. **An array's view is a cast of the tree** (l.78, since the review of
   2026-10-23, as 준량 decided). No declaration carries an array, and a sequence
   arrives at one as a slice (extern fn §1), so the language gives, at each length
   `n`, `&[T; n] → &[T]` and `&mut [T; n] → &mut [T]` (`Role::ArrayView`,
   `decl::ARRAY_VIEW`, `ARRAY_VIEW_MUT`), each going up from the node a reference
   to an array of that length is (l.66: an array's length is in its constructor).
   They meet the tree's conditions as a declared cast does, but go from no
   extern-defined type. The model declares them at the lengths a case's types
   write (`DeclsBuilder::array_views`, `cases::build::array_lengths`), an array's
   length being what is written. Before, an array reached no declaration:
   `sum(&a)` with `a: [u8; 2]` and `sum(&[u8])` was refused (`cases/spec/l078-*`).
   Where references meet, their targets are equated all the same (note 11,
   reading 1; `holes.md`, H3), so the view is taken at a boundary alone.

## Left open for change

- `DeclsBuilder` takes declarations one at a time and checks the tree once, at
  `finish`, so the order in which a program declares them is no input to the tree.
- `Decls::raise` returns the casts it takes, so that a join can record them, and a
  coercion can name the functions it calls.
