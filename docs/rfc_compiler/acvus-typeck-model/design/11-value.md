# 11 Values, splits and fills

Rests on the spec's l.261 (the value a node reads) and l.267–282 (values, splits,
the checks of entering values, and fills). The code is `src/value.rs`, the splits
in `src/split.rs`, a kind "indexes `τ`" linked along a view in `src/unify.rs`, the
state a value read is taken from in `src/solve.rs`, and the checks in
`src/check.rs`.

## Meaning

A class takes its value once its unbroken dependencies are solved or decided: the
type its equation gives, or the join of the constructors and members its sources
give, each part made anew, a part lower bound's constructor reaching it with no
cast; a LitVar source is unified with it. A lower bound, a part lower bound or a
boundary whose two sides are solved relates their parts by one rule (l.278): each
part of the value passes into the other side's part, a meeting where that part is
an unknown of the group (l.219), recording a part lower bound between a
structural type's parts and an equation between other parts, and a boundary
elsewhere. Where a lower bound's or a boundary's two constructors differ, the
same rule links the parts through the casts' signatures, or the targets of a
reborrow; where a function value passes into a function type, an instance of each
lambda its Values names links them (l.197). A class nothing enters takes the one type the places it passes into
have, once each has one or holds only a LitVar nothing more decides, which takes
any type of its family that holds it, and has no value where they have two
(l.282). A value
read is decided on a separate state that replays the relations a node does not
leave out.

## Readings

1. **Two references join whatever their targets.** A reference's constructor is its
   mutability alone (l.66), so two references meet at the lesser mutability, and
   the split relates their targets by the rule of l.278: toward an unknown, an
   equation. Two lower bounds `&A ≤ R` and `&C ≤ R` then equate `A` and `C`,
   where the join of l.86 would raise them to a common ancestor (`holes.md`, H3).
   Toward a type a declaration writes the target is a boundary, so a view's cast
   is taken where the whole value reaches it (l.78, l.280).
2. **The targets of a reborrow.** Where an exclusive reference passes into a
   shared one by a lower bound or a boundary, the split links their targets by
   the rule of l.278, the target being the part a reborrow keeps.
3. **Linking through casts.** At each cast on the path between the two nodes, the
   current type's parts pass into the cast's parameter's by the rule of l.278, a
   variable of the cast's signature being an unknown and a type it writes not;
   the last cast's result passes on into the other side. A part lower bound's two
   sides of two constructors relate nothing: a part takes no cast (l.74, l.89),
   and its sources then have no join.
4. **A function value into a function type.** A function value and a function type
   are of one constructor, the value carrying no parts (l.186), so no part of the
   value passes into the type's. Where the split of a boundary finds the two, it
   makes an instance of each lambda the value's Values names then, as though the
   type called it (l.197): each parameter of the type passes into the copy of the
   lambda's parameter, or into the environment's, and the copy of the lambda's
   result type, or the environment's, into the type's result, each a meeting or a
   boundary by l.219 (`State::instantiate_lambda`, shared with a call of a
   function value). What the instance records is recorded by
   `Recorder::FunctionType`, which a value read derives again, as it does a
   split's, and checks, as it does a node's. The whole boundary's own check is
   `Undecided`, a function value and a function type agreeing at every position
   both decide; what the type requires is checked in the instance.
4. **Sources that give no constructor.** A class whose sources are only LitVars is
   unified with them and stays unsolved, holding a LitVar until fallback. Where
   they are of two families the unification conflicts (l.163), so its sources
   have no join: it has no value and is not solved again, refused by the
   conflict (l.169, l.290), as where the types its sources give have no join.
   A class that holds a kind takes its value from it, and a LitVar lower bound
   that conflicts with it is refused there alone. `?N ⇝ U` unifies a class with
   the types it passes into by a boundary only where LitVars alone enter it, no
   solved lower bound, part lower bound or kind known in part giving it a
   constructor (l.278, l.282): a class that also takes another source's type is
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
   read (l.261) is not that nothing enters it (l.282): the first ends a read,
   whose dependencies are not waited for, and the second is of the state.
7. **An element along a view.** Where an extern-defined container's class holds
   "indexes `τ`", its parts are related to those of the view's parameter's target,
   and `τ` to the element of its result's slice, by equations, the view being the
   cast going up from a shared reference to the container (l.78, l.154); a cast up
   from `&mut N` has one from `&N`. A container whose cast goes elsewhere, or that
   has none, has no view, and the kind is refused.
8. **Visibility.** A kind adds a member to a type another source gives where it
   names a field, or a variant of an enum the text names (l.276): a field the text
   writes grows a class a type the host gives is a lower bound of (`holes.md`, T5).
   The reading by the sources is removed.
9. **`!` and a kind known in part.** `!` has no part, so it meets no kind known in
   part (`holes.md`, T6).
10. **A class nothing enters** (l.282). An unsolved class holding no kind, no
    LitVar and no kind known in part, each of whose lower bounds and part lower
    bounds is one nothing enters, and each of whose unbroken dependencies is met,
    a dependency on a lower bound or a part lower bound nothing enters being met
    (l.238). The unbroken dependencies have no cycle (l.265), so such classes are
    found from those with no lower bound up. A kind known in part is a source
    (l.274), even one that gives no member (l.276), so a class holding one is not
    one nothing enters.
11. **A fill** is a step, taken where the class's places, the `α` of each lower
    bound and part lower bound whose `τ` it is and the `U` of each boundary whose
    `T` it is, each have a closed value or hold only a LitVar nothing more decides,
    settled and with nothing left to take (`State::fills_now`), and one has a
    closed value (l.282, since 13289a2f). Where the closed values are one type and
    each such LitVar holds it (`PlaceValue`), as a LitVar decides nothing within
    its family at a narrowing (l.333), the class takes it, each part a class made
    anew holding its part, every one solved; its relations then split as any
    solved class's do, and each entering value reaches its place with no
    coercion. Otherwise the class has no value and is refused there
    (`Refusal::NoFill`). A class that passes into no place, or whose places hold
    only LitVars, is not filled, and falls back (l.288).
12. **`e?`'s wrapping** (l.278, l.337). The class `X` the wrapping passes into
    waits for the error part `E` to take a constructor (l.238 (a); `Edge::Taking`).
    Once it does, the split records, by `Recorder::Split(E)`: where `E` is an
    `Error` or `!`, `E ≤ X`, no wrapping; where it is a type of a name `N`, the
    kind `Error{N(P)}` of `X` and `E ≤ P`; and otherwise the kind `Error⟨P⟩` and
    `E ≤ P`, `P` made anew (`State::split_wrap`). An `E` nothing enters waits for
    nothing once `?` is decided, and falls back to `!` where its group may
    (l.288).
13. **Two identities** (l.351). A class whose solved lower bounds carry two
    different identities takes the nearest node above their join that carries
    none, along the cast tree (`join::without_identity`), each value cast up to
    it as any lower bound's is; where no such node is, or a part lower bound
    carries one of them, it has no join (`State::without_two_identities`). The
    closed join does the same (`join::join`). A deeper level's identity, refused
    since 345c838c, has no witness here: the model has no loop and no level.
14. **`!` and kinds** (l.276). `!`, which has no value, meets every kind
    (`Classes::normalize`), and once a class takes `!`, each kind known in part
    on it splits by recording `!` as the part it names (`State::split_fact`), so
    that what is written into such a part is refused where it enters it.

