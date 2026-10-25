# 06 Equivalence classes

The code is `src/class.rs`, and `src/class/tests.rs` exercises its contract; what a
recorded relation gives the classes is `State::hold` in `src/unify.rs`.

## Meaning

An equivalence class is kept by union-find over the classes of one solving. It
holds its members, at most one structure with where that structure came from, the structure
constraints (`StructureConstraint`) that projections give it, at most one LitVar, and whether it is
solved. Unifying two merges what they hold, and unifies their parts of one name in
turn.

## Readings

1. **Structure constraints beside the structure.** A structure constraint is held beside the structure. Two structure constraints
   of one part name one class, and a structure constraint meets the structure where the structure has that
   part: a struct's field is the part at its place in the declared order. A structure constraint the
   structure does not meet is a conflict: no type holds both. An extern-defined
   container's element is not its part, so "indexes `τ`" meets its structure in the state,
   along its view. A class whose only sources are structure constraints takes the constructor of each
   member they name, visible or not, `Some` and `None` naming `Option` and `Ok` and
   `Err` naming `Result` (`StructureConstraint::named_member`); visibility governs only a structure constraint
   adding a member to a type another source gives. A variant of no enum's name that
   is none of those four names no constructor: no witness shows one.
2. **A conflict.** Where two classes' structures have different constructors or members,
   or a LitVar meets a scalar that does not admit its values, the two are left apart
   and the relation that would make them one is refused. Unified, their structure would be
   a conjunction no type meets, so each class the conflict names has no value, as a
   class whose sources have no join (`State::conflicted_have_no_value`).
3. **"With an equation".** A class's value is the type every equation gives where
   it holds a structure the text or a declaration writes. That holds through
   unification, whatever else the class is unified with (`StructureSource::Written`).
4. **A written type is solved.** A type the text or a declaration writes is decided
   when solving starts, and nothing records into it, so it is solved as it
   is recorded. A class unified with a solved one is solved: its value is decided
   once, and its lower bounds, part lower bounds and boundaries are
   decomposed as they come.
5. **Names.** A text class keeps its name. A part made when a class is solved is
   named after the least of that class's members' names at that moment, a class a
   use or a lend makes after it and its place in the signature, and a class of a
   cast's signature after the relation it links. So the names of classes made while
   solving may depend on the order; the outcome names only the text's classes.
