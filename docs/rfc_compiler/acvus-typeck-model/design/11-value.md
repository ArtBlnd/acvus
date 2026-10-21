# 11 Values, splits and fills

Rests on the spec's l.263 (the value a node reads) and l.269–284 (values, splits,
the checks of entering values, and fills). The code is `src/value.rs`, the splits
in `src/split.rs`, a kind "indexes `τ`" linked along a view in `src/unify.rs`, the
state a value read is taken from in `src/solve.rs`, and the checks in
`src/check.rs`.

## Meaning

A class takes its value once its unbroken dependencies are solved or decided: the
type its equation gives, or the join of the constructors and members its sources
give, each part made anew, a part lower bound's constructor reaching it with no
cast; a LitVar source is unified with it. A lower bound, a part lower bound or a
boundary whose two sides are solved relates their parts by one rule (l.280): each
part of the value passes into the other side's part, a meeting where that part is
an unknown of the group (l.221), recording a part lower bound between a
structural type's parts and an equation between other parts, and a boundary
elsewhere. Where a lower bound's or a boundary's two constructors differ, the
same rule links the parts through the casts' signatures, or the targets of a
reborrow. A class nothing enters takes the one type the places it passes into
have, once each has one, and has no value where they have two (l.284). A value
read is decided on a separate state that replays the relations a node does not
leave out.

## Readings

1. **Two references join whatever their targets.** A reference's constructor is its
   mutability alone (l.66), so two references meet at the lesser mutability, and
   the split relates their targets by the rule of l.280: toward an unknown, an
   equation. Two lower bounds `&A ≤ R` and `&C ≤ R` then equate `A` and `C`,
   where the join of l.86 would raise them to a common ancestor (`holes.md`, H3).
   Toward a type a declaration writes the target is a boundary, so a view's cast
   is taken where the whole value reaches it (l.78, l.282).
2. **The targets of a reborrow.** Where an exclusive reference passes into a
   shared one by a lower bound or a boundary, the split links their targets by
   the rule of l.280, the target being the part a reborrow keeps.
3. **Linking through casts.** At each cast on the path between the two nodes, the
   current type's parts pass into the cast's parameter's by the rule of l.280, a
   variable of the cast's signature being an unknown and a type it writes not;
   the last cast's result passes on into the other side. A part lower bound's two
   sides of two constructors relate nothing: a part takes no cast (l.74, l.89),
   and its sources then have no join.
4. **Sources that give no constructor.** A class whose sources are only LitVars is
   unified with them and stays unsolved, holding a LitVar until fallback. Where
   they are of two families the unification conflicts (l.165), so its sources
   have no join: it has no value and is not solved again, refused by the
   conflict (l.171, l.292), as where the types its sources give have no join.
   A class that holds a kind takes its value from it, and a LitVar lower bound
   that conflicts with it is refused there alone. `?N ⇝ U` unifies a class with
   the types it passes into by a boundary only where LitVars alone enter it, no
   solved lower bound, part lower bound or kind known in part giving it a
   constructor (l.280, l.284): a class that also takes another source's type is
   that type, which its boundaries check.
5. **When a relation splits.** Once both of its sides are solved (`holes.md`, T2).
   A type the text or a declaration writes is solved as it is recorded (note 06,
   reading 4), so a boundary toward one splits as soon as the value's class is
   solved, and the places a class's part passes into are recorded with the part.
6. **A value read.** The state replays every relation but those the left-out nodes
   recorded and those solving derives (note 07), solves and fills each class in
   one fixed order, decides no node and falls nothing back. A class with no
   relation left gives nothing, and one holding only a LitVar gives that LitVar.
   A class no step solves or fills there gives nothing, and is not filled after:
   what it gave other classes is not changed. That a class gives nothing in a
   read (l.263) is not that nothing enters it (l.284): the first ends a read,
   whose dependencies are not waited for, and the second is of the state.
7. **An element along a view.** Where an extern-defined container's class holds
   "indexes `τ`", its parts are related to those of the view's parameter's target,
   and `τ` to the element of its result's slice, by equations, the view being the
   cast going up from a shared reference to the container (l.78, l.156); a cast up
   from `&mut N` has one from `&N`. A container whose cast goes elsewhere, or that
   has none, has no view, and the kind is refused.
8. **Visibility.** A kind gives its member where it names a field, or a variant of
   an enum the text names (l.278). Whether a lower bound the host gives makes it
   give none is `holes.md`, T5, and the model takes it as a reading
   (`Visibility`).
9. **`!` and a kind known in part.** `!` has no part, so it meets no kind known in
   part (`holes.md`, T6).
10. **A class nothing enters** (l.284). An unsolved class holding no kind, no
    LitVar and no kind known in part, each of whose lower bounds and part lower
    bounds is one nothing enters, and each of whose unbroken dependencies is met,
    a dependency on a lower bound or a part lower bound nothing enters being met
    (l.240). The unbroken dependencies have no cycle (l.267), so such classes are
    found from those with no lower bound up. A kind known in part is a source
    (l.276), even one that gives no member (l.278), so a class holding one is not
    one nothing enters.
11. **A fill** is a step, taken where the class's places, the `α` of each lower
    bound and part lower bound whose `τ` it is and the `U` of each boundary whose
    `T` it is, are at least one and each has a closed value (l.284). Where they are
    one type, the class takes it, each part a class made anew holding its part,
    every one solved; its relations then split as any solved class's do, and each
    entering value reaches its place with no coercion. Where they are two, the
    class has no value and is refused there (`Refusal::NoFill`). A class that
    passes into no place is not filled, and falls back (l.290).

