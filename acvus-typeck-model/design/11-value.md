# 11 Values and splits

Rests on the spec's l.267 (the value a node reads) and l.273–293 (values, splits
and the checks of entering values). The code is `src/value.rs` and the splits in
`src/solve.rs`.

## Meaning

A class takes its value once its unbroken dependencies are solved or decided: the
type its equation gives, or the join of the constructors and members its sources
give, each part made anew; a LitVar source is unified with it. A lower bound or a
boundary whose two sides are solved relates their parts by the table of l.286,
through the casts' signatures where the constructors differ. A value read is
decided on a separate state that replays the relations a node does not leave out.

## Readings

1. **Two references join whatever their targets.** A reference's constructor is its
   mutability alone (l.66), so two references meet at the lesser mutability, and
   their targets, an invariant part, are related by an equation (l.289). A view's
   cast, a step between two references' nodes (l.78), is then never taken at a
   class (`holes.md`, H3).
2. **A reborrow at a split.** Where an exclusive reference meets a shared one, the
   constructors differ and no cast links them (l.291). The model relates their
   targets by an equation, as the row `&mut T ⇝ &T` keeps `T` (`holes.md`, H4).
3. **Linking through casts.** At each cast on the path between the two nodes, the
   current type's parts and the cast's parameter's are related by equations, a
   cast's parameter being an extern-defined type or a reference, whose parts are
   invariant; the last cast's result is then related to the other side by l.286.
4. **Sources that give no constructor.** A class whose sources are only LitVars is
   unified with them and stays unsolved, holding a LitVar until fallback.
5. **When a relation splits.** Once both of its sides are solved (`holes.md`, T2).
6. **A value read.** The state replays every relation but those the left-out nodes
   recorded and those solving derives (note 07), solves each class from its sources
   in one fixed order, decides no node and falls nothing back. A class with no
   relation left gives nothing, and one holding only a LitVar gives that LitVar.
7. **An element along a view.** Where an extern-defined container's class holds
   "indexes `τ`", its parts are related to those of the view's parameter's target,
   and `τ` to the element of its result's slice, by equations, the view being the
   cast going up from a shared reference to the container (l.78, l.161); a cast up
   from `&mut N` has one from `&N`. A container whose cast goes elsewhere, or that
   has none, has no view, and the kind is refused.
8. **Visibility.** A kind gives its member where it names a field, or a variant of
   an enum the text names (l.282). Whether a lower bound the host gives makes it
   give none is `holes.md`, T5, and the model takes it as a reading
   (`Visibility`).
9. **`!` and a kind known in part.** `!` has no part, so it meets no kind known in
   part (`holes.md`, T6).
