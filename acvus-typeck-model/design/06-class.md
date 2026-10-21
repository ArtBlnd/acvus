# 06 Equivalence classes

Rests on the spec's l.118–176 (equivalence classes, type variables, kinds,
projections' kinds, unification and the join of a LitVar), with l.279 (a class
"with an equation") and l.282 (a value meets the kind its class holds). The code is
`src/class.rs`, and `src/class/tests.rs` exercises its contract; what a recorded
relation gives the classes is `State::hold` in `src/unify.rs`.

## Meaning

An equivalence class is kept by union-find over the classes of one solving. It
holds its members, at most one kind with where that kind came from, the kinds known
in part (`Fact`) that projections give it, at most one LitVar, and whether it is
solved. Unifying two merges what they hold, and unifies their parts of one name in
turn.

## Not owned here

- When a class takes a constructor, and what its value is, are note 11's.
- What records into a class, and in what order, are notes 07, 08 and 12's.

## Readings

1. **Facts beside the kind.** A kind known in part is held beside the kind
   (l.138–147). Two facts of one part name one class, and a fact meets the kind
   where the kind has that part (l.165): a struct's field is the part at its place
   in the declared order (l.325). A fact the kind does not meet is a conflict: no
   type holds both (l.282). An extern-defined container's element is not its part,
   so "indexes `τ`" meets its kind in the state, along its view (note 11, reading
   7).
2. **A conflict.** Where two classes' kinds have different constructors or members,
   or a LitVar meets a scalar that does not admit its values, the two are left apart
   and the relation that would make them one is refused.
3. **"With an equation".** A class's value is the type every equation gives where
   it holds a kind the text or a declaration writes (l.279). That holds through
   unification, whatever else the class is unified with (`KindSource::Written`).
4. **A written type is solved.** A type the text or a declaration writes is decided
   when solving starts (l.218), and nothing records into it, so it is solved as it
   is recorded. A class unified with a solved one is solved: its value is decided
   once (types §6), and its lower bounds and boundaries are split as they come.
5. **Names.** A text class keeps its name. A part made when a class is solved is
   named after the least of that class's members' names at that moment, a class a
   use or a lend makes after it and its place in the signature, and a class of a
   cast's signature after the relation it links. So the names of classes made while
   solving may depend on the order; the outcome names only the text's classes.
