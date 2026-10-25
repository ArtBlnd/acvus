# 11 Values, decompositions and fills

The code is `src/value.rs`, the decompositions
in `src/decomposition.rs`, a structure constraint "indexes `τ`" linked along a view in `src/unify.rs`, the
state a value read is taken from in `src/solve.rs`, and the checks in
`src/check.rs`.

## Meaning

A class takes its value once its unbroken dependencies are solved or decided: the
type its equation gives, or the join of the constructors and members its sources
give, each part made anew, a part lower bound's constructor reaching it with no
cast; a LitVar source is unified with it. A lower bound, a part lower bound or a
coercion constraint whose two sides are solved relates their parts by one rule: each
part of the value passes into the other side's part, a lower bound where that part is
an unknown of the group, recording a part lower bound between a
structural type's parts and an equation between other parts, and a coercion constraint
elsewhere. Where a lower bound's or a coercion constraint's two constructors differ, the
same rule links the parts through the casts' signatures, or the targets of a
reborrow, and through no conversion, whose check asks for the Values it carries;
where a function value passes into a function type, an instance of each
lambda its Values names links them. A class nothing enters takes the one type its
upper bounds
have, once each has one or holds only a LitVar nothing more decides, which takes
any type of its family that holds it, and has no value where they have two.
A value
read is decided on a separate state that replays the relations a disjunctive constraint does not
leave out.

## Readings

1. **Two references join whatever their targets.** A reference's constructor is its
   mutability alone, so two references join at the lesser mutability, and
   the decomposition relates their targets: toward an unknown, an
   equation. Two lower bounds `&A ≤ R` and `&C ≤ R` then equate `A` and `C`,
   where the join would raise them to a common ancestor.
   Toward a type a declaration writes the target is a coercion constraint, so a view's cast
   is taken where the whole value reaches it.
2. **The targets of a reborrow.** Where an exclusive reference passes into a
   shared one by a lower bound or a coercion constraint, the decomposition links their targets, the
   target being the part a reborrow keeps.
3. **Linking through casts.** At each cast on the path between the two nodes, the
   current type's parts pass into the cast's parameter's, a
   variable of the cast's signature being an unknown and a type it writes not;
   the last cast's result passes on into the other side. A part lower bound's two
   sides of two constructors relate nothing: a part takes no cast,
   and its sources then have no join.
4. **A function value into a function type.** A function value and a function type
   are of one constructor, the value carrying no parts, so no part of the
   value passes into the type's. Where the decomposition of a coercion constraint finds the two, it
   makes an instance of each lambda the value's Values names then, as though the
   type called it: each parameter of the type passes into the copy of the
   lambda's parameter, or into the environment's, and the copy of the lambda's
   result type, or the environment's, into the type's result, each a lower bound or a
   coercion constraint (`State::instantiate_lambda`, shared with a call of a
   function value). What the instance records is recorded by
   `Recorder::FunctionType`, which a value read derives again, as it does a
   decomposition's, and checks, as it does a disjunctive constraint's. The whole coercion constraint's own check is
   `Undecided`, a function value and a function type agreeing at every position
   both decide; what the type requires is checked in the instance.
4. **Sources that give no constructor.** A class whose sources are only LitVars is
   unified with them and stays unsolved, holding a LitVar until fallback. Where
   they are of two families the unification conflicts, so its sources
   have no join: it has no value and is not solved again, refused by the
   conflict, as where the types its sources give have no join.
   A class that holds a structure takes its value from it, and a LitVar lower bound
   that conflicts with it is refused there alone. `?N ⇝ U` unifies a class with
   the types it passes into by a coercion constraint only where LitVars alone enter it, no
   solved lower bound, part lower bound or structure constraint giving it a
   constructor: a class that also takes another source's type is
   that type, which its coercion constraints check.
5. **When a relation is decomposed.** Once both of its sides are solved.
   A type the text or a declaration writes is solved as it is recorded, so a coercion constraint
   toward one is decomposed as soon as the value's class is
   solved, and the upper bounds of a class's part are recorded with the part.
6. **A value read.** The state replays every relation but those the left-out disjunctive constraints
   recorded and those solving derives, solves and fills each class in
   one fixed order, decides no disjunctive constraint and falls nothing back. A class with no
   relation left gives nothing, and one holding only a LitVar gives that LitVar.
   A class no step solves or fills there gives nothing, and is not filled after:
   what it gave other classes is not changed. That a class gives nothing in a
   read is not that nothing enters it: the first ends a read,
   whose dependencies are not waited for, and the second is of the state.
7. **An element along a view.** Where an extern-defined container's class holds
   "indexes `τ`", its parts are related to those of the view's parameter's target,
   and `τ` to the element of its result's slice, by equations, the view being the
   cast going up from a shared reference to the container; a cast up
   from `&mut N` has one from `&N`. A container whose cast goes elsewhere, or that
   has none, has no view, and the structure constraint is refused.
8. **Visibility.** A structure constraint adds a member to a type another source gives where it
   names a field, or a variant of an enum the text names: a field the text
   writes grows a class a type the host gives is a lower bound of.
   The reading by the sources is removed.
9. **`!` and a structure constraint.** `!` has no part, so it satisfies no structure
   constraint.
10. **A class nothing enters**. An unsolved class holding no structure, no
    LitVar and no structure constraint, each of whose lower bounds and part lower
    bounds is one nothing enters, and each of whose unbroken dependencies is satisfied,
    a dependency on a lower bound or a part lower bound nothing enters being satisfied.
    The unbroken dependencies have no cycle, so such classes are
    found from those with no lower bound up. A structure constraint is a source,
    even one that gives no member, so a class holding one is not
    one nothing enters.
11. **A fill** is a step, taken where the class's upper bounds, the `α` of each lower
    bound and part lower bound whose `τ` it is and the `U` of each coercion constraint whose
    `T` it is, each have a closed value or hold only a LitVar nothing more decides,
    settled and with nothing left to take (`State::fills_now`), and one has a
    closed value. Where the closed values are one type and
    each such LitVar holds it (`PlaceValue`), as a LitVar decides nothing within
    its family at a narrowing, the class takes it, each part a class made
    anew holding its part, every one solved; its relations are then decomposed as any
    solved class's are, and each entering value reaches its upper bound with no
    coercion. Otherwise the class has no value and is refused there
    (`Refusal::NoFill`). A class that passes into no upper bound, or whose upper bounds hold
    only LitVars, is not filled, and falls back.
12. **`e?`'s wrapping**. The class `X` the wrapping passes into
    waits for the error part `E` to take a constructor (`Edge::Taking`).
    Once it does, the decomposition records, by `Recorder::Decomposition(E)`: where `E` is an
    `Error` or `!`, `E ≤ X`, no wrapping; where it is a type of a name `N`, the
    structure `Error{N(P)}` of `X` and `E ≤ P`; and otherwise the structure `Error⟨P⟩` and
    `E ≤ P`, `P` made anew (`State::decompose_wrap`). An `E` nothing enters waits for
    nothing once `?` is decided, and falls back to `!` where its group may.
13. **Two identities**. A class whose solved lower bounds carry two
    different identities takes the nearest node above their join that carries
    none, along the cast tree (`join::without_identity`), each value cast up to
    it as any lower bound's is; where no such node is, or a part lower bound
    carries one of them, it has no join (`State::without_two_identities`). The
    closed join does the same (`join::join`). A deeper level's identity, refused,
    has no witness here: the model has no loop and no level.
14. **`!` and structure constraints**. `!`, which has no value, satisfies every structure constraint
    (`Classes::normalize`), and once a class takes `!`, each structure constraint
    on it is decomposed by recording `!` as the part it names (`State::decompose_structure_constraint`), so
    that what is written into such a part is refused where it enters it.
15. **Values along a conversion**. The decomposition links nothing through
    a declared conversion, which the closed types decide once both sides are
    solved, so no relation brings the lambdas a function value it
    carries to where it reaches. The coercion constraint's check asks for them there: at
    each place the conversion's parameter writes a type parameter, each lambda
    of a function value in it is among the Values of the function value at the
    same place of each position its result writes that type parameter
    (`State::unnamed`); the check of a conversion that brings one they do not
    name fails, naming it (`Check::unnamed`, `unnamed: {μ}` in a golden).

