# Transparency-constraint type system

## Compilation graph

A program is one compilation graph. Its nodes are units, extern fns and contexts, and its edges are uses: a body, the text of a free fn, a method or a lambda, uses what it names.

| Node | Sort | Use |
|---|---|---|
| Unit | module, script, template | calls its free fns and methods |
| Extern fn | a declaration of an external function | calls it |
| Context | a declared `@x` | assigns a value to it and takes its value out |

### Unit

| Unit | Free fns | Struct |
|---|---|---|
| Module | zero or more | at most one, the module itself, whose impl blocks hold its methods |
| Script | one, whose body is the script's text | none |
| Template | one, whose body is the template's text | none |

A template is a script whose body is the `String` formatted from its text, and whose result type is fixed to `String`. `{{ e }}` is a use of `core::display`. A script's or a template's input `$x` is a parameter of its function (programs §4).

A lambda lies inside another body, its enclosing body. A call made by a lambda that lies in the text of a free fn or a method is a call of that free fn or method. Free fns and methods that reach one another by calls form one binding group, and a lambda is a binding group of its own, lying inside the binding group of its enclosing body.

## Transparency

A body sees the target of a use at one of three levels.

| Target of the use | What the body sees | Level |
|---|---|---|
| itself | all of its own text | Transparent |
| another body: a free fn, a method or a lambda | its text, separately at each use | Semi-transparent |
| an extern fn, a context | its declaration | Opaque |

A module's struct has its fields named, and its values made, only inside that module's impl blocks.

A context's type is given by the host. A value in a context outlives the run, so its type has no positions (context §3, ownership §5), and a context holds neither a reference nor a function.

## Types

A type is a finite term. A term consists of one constructor, its members and its parts, and each part is again a type.

| Constructor | Notation | Parts | Members |
|---|---|---|---|
| Reference | `&T`, `&mut T` | the target | |
| Tuple | `(T₁, …, Tₙ)` | each component | |
| Array | `[T; n]` | the element | |
| Slice | `[T]` | the element | |
| Option | `Option⟨T⟩` | the payload | `Some`, `None` |
| Result | `Result⟨T, E⟩` | the ok, the error | `Ok`, `Err` |
| Function | `Fn(P̄ → R)` | each parameter, the result | |
| Object | `{f̄: T̄}` | each field's type | its fields |
| Enum | `E{V̄(T̄)}` | each payload | its variants |
| Struct | `Self⟨T̄⟩` | one type parameter for each field, in the order the struct declares them | |
| Extern-defined type | `N⟨T̄, ι⟩` | each argument, and the identity | |
| Scalar | integer, float, `bool`, `char` (Rust's scalar types) | none | |
| String | `String` | none | |
| str | `str` | none | |
| Unit type | `()` | none | |
| Bottom | `!` | none | |

A reference has a mutability: `&` is shared, and `&mut` is exclusive.

No type has a reference directly as its target. A reference's lifetime holds several loans; it is not written in the text, and is decided at each use as the least one. This is because another body is Semi-transparent, so what that body does with a reference is seen at each use. So the lifetimes of two references need not be told apart, and what is reached through a reference is reached by one reference: its lifetime holds the loans of both references, and it is exclusive only where both are (ownership §5). The parts inside a target are not subject to this rule, the argument of `&N⟨&U⟩` for example.

Two types are equal when their constructors, members and parts are all equal. A constructor is what remains of the notation without its members and parts: a reference's mutability, an enum's name, a struct's module, an extern-defined type's name, and a tuple's or an array's length are in it.

An object, enum, tuple, array, option or result type is a structural type, and is known by the names it carries.

The specialization mark of an extern-defined type (types §8) is not handled by this type system. A type carries no mark, and turning an unmarked value into a marked one is the optimizer's.

### Variance

Every part of a structural type is covariant, and every part of any other type is invariant. No part is contravariant. The relation between invariant parts is an equation.

### Cast tree

A declared cast goes from an extern-defined type or a reference to one, and changes neither whether it is a reference nor its mutability. An extern-defined container's view impl, an impl of `core::as_slice`, `core::as_slice_mut` or `core::as_str`, is a declared cast from a reference to that container to a reference to a slice or a `str` (operators §2, §5). The nodes of the cast tree are constructors, and a reference's node is its mutability and its target's constructor. The declared casts form a tree: at most one cast goes up from a node, they form no cycle, and where a cast goes from `&mut X` to `&mut Y`, the cast going up from `&X` goes to `&Y`.

### Join

The join of two types, and the order `≤` the join defines (`S ≤ T` where `S ⊔ T = T`):

| Definition | Description |
|---|---|
| `S ⊔ T = nca(S, T)` | Where `S` and `T` have a common ancestor node, each is raised by casts to their nearest common ancestor node, and left as it is where they are of one node; then their members are united and their covariant parts joined. |
| `&mut S ⊔ &T = &S ⊔ &T` | `&mut S` is taken as `&S`, and joined. |
| `! ⊔ T = T` | Of `!` and `T`, `T`. |

At a `Result`'s error part, the error join `⊔ᴱ` is taken in place of `⊔`.

| Definition | Description |
|---|---|
| `⟦!⟧ = ∅`, `⟦Error{N̄(Ē)}⟧ = {Ē}`, any other `⟦E⟧ = {E}` | The error part's type is taken as a set of error types. |
| `E₁ ⊔ᴱ E₂ = ⟦E₁⟧ ∪ ⟦E₂⟧` | Elements of one node are first joined by `⊔`. Then the join is `!` where no element remains, that type where one does, and where two or more do and all are types of distinct names, enums, structs or extern-defined types, `Error` with a variant for each element, named after it. |

`Error` is the language's enum, and its variants are made only by this join.

### Coercion

A coercion changes a value into a value of another type, and the function that changes it is called.

| Definition | Description | Function |
|---|---|---|
| `S ⇝ T` for each declared cast `S → T` | One step up the cast tree. | the declaration's function |
| `&mut T ⇝ &T` | `&mut T` to `&T`. | decided by the language |
| `{f̄, ḡ} ⇝ {f̄}` | An object to one with only some of its fields. | decided by the language |
| `E{Ā} ⇝ E{Ā, B̄}` | An enum to one with more variants. | decided by the language |
| `E ⇝ Error{N̄(Ē)}` where `E` is one of `Ē` | Wraps it in the variant of its name. | decided by the language |
| `T ⇝ U` where their constructors and members are equal and `Tᵢ ⇝ Uᵢ` at each covariant part | At each covariant part, a coercion of that part. | the parts' coercions |
| `! ⇝ U` | `!` to any type. | decided by the language |
| `T ⇝ U` by a declared conversion `T → U` | Where no composition of the rows above reaches the type. | the declaration's function |

### Rank

A coercion is a composition of the Coercion table's rows in the order `reborrow? ; cast* ; structural?`: reborrow is the `&mut T ⇝ &T` row, cast a declared cast's row, and structural the object, enum, `Error` and covariant-part rows. `! ⇝ U` and a declared conversion are no composition, and have no rank. Toward a given type, the composition is unique. Where no coercion is needed, the rank is `empty`. The stages are ordered `empty ≺ reborrow ≺ cast ≺ structural`, and the ranks of coercions are compared by their highest stage, then by their number of casts, then by the lower stages they take. The stages and the number of casts include those of the parts' coercions.

## Equivalence classes

A type variable is one type, represented by an equivalence class kept by union-find. An equivalence class and a kind are different things.

| | What it is | What it holds |
|---|---|---|
| Equivalence class | a collection of equal types | type variables, and at most one kind |
| Kind | a structure | a constructor, members, and each part's equivalence class |

Every type is split into these. For example, `Vec⟨T⟩` is one equivalence class holding the kind `Vec(·)`, whose part is `T`'s equivalence class.

### Type variables

| Sort | Notation | Description |
|---|---|---|
| `Var` | `?0` | A type not yet known. |
| `LitVar` | `?N: int`, `?N: float` | A literal's type: `N` the literal values it holds, a literal `n` being `?{n}`. It decides nothing. |

### Kind

A kind may be known only in part, without a constructor. Below, `τ` is that type's equivalence class.

| Kind | Meaning |
|---|---|
| a constructor, members, parts | One constructor of the type table, its members, and each part's equivalence class. |
| a reference to `τ` | Its target is `τ`. |
| has field `f: τ` | |
| has component `i: τ` | |
| indexes `τ` | Its element is `τ`. |
| has variant `V` with payload `τ` | |

### Projection

A projection names a part of a type not yet known: it gives the operand's equivalence class a kind known in part, and names that part's equivalence class `τ`.

| Primitive | Notation | Example | Kind the operand's equivalence class takes |
|---|---|---|---|
| `Deref` | `*?0` | `*r` | a reference to `τ` |
| `Field` | `?0.f` | `p.f` | has field `f: τ` |
| `Component` | `?0.i` | `t.0` | has component `i: τ` |
| `Elem` | `?0[]` | `a[i]` | indexes `τ` |
| `Payload` | `?0 as V` | `x` in `if let Some(x) = o` | has variant `V` with payload `τ` |

The kind "indexes `τ`" is unified with the element part at an array or a slice, and is linked to the slice's element along the signature of its view at an extern-defined container (operators §3). A type with no view cannot have this kind.

### Unify

Unifying two equivalence classes merges them into one, whose kind is the conjunction of the two. Parts of one name, that is the same field, the component at the same position, the target, the element and the payload of the same variant, have their equivalence classes unified in turn (congruence).

| Held in one equivalence class | Rule |
|---|---|
| A LitVar `?N` and a constructor | The constructor is one of `?N`'s family, integer or float, that holds every value in `N`. |
| LitVars `?N` and `?M` | Two of one family become one `?(N ∪ M)`. |

### Join of a LitVar

| Definition | Description |
|---|---|
| `?N ⊔ τ = τ` | `?N` and `τ` are unified. |

## Facts

A fact is an attribute a function carries, neither a part nor a member of a type. A fact is a join-semilattice with a top, nothing known; where a join is not defined, or no source decides it, a fact is the top. The facts of the function values that meet in one equivalence class are joined. A fact is decided once, from its sources, and inference chooses by no fact that is decided after it.

| Fact | Carried by | Order | Top | Sources |
|---|---|---|---|---|
| Effect | a function; an extern-defined type | pure ⊑ opaque | opaque | a function's body; for an extern-defined type, the functions its values hold |
| Flows | a function | listed flows ⊑ every flow | every flow | the function's body |
| Calls | a function | many ⊑ once | once | after inference, how its captures are used |
| Holds | a function | nothing ⊑ shared ⊑ exclusive | exclusive | after inference, its captures |
| Values | a function | sets of lambdas ⊑ any | any | the lambdas written |
| Declared facts | a declaration | what is stated ⊑ nothing stated | nothing stated | the declaration |

A function that becomes a value carries its facts in its function type, and a call performs the callee's effect in the body it is in. Since an extern fn gives no function value and a context holds no function, a function value's Values at an instance is always a set of lambdas.

A lambda's parameter and result types lie in that lambda's binding group; an equivalence class that holds a function value does not carry them as parts but names the lambda by Values. At each use of the value, that lambda's C is made anew as an instance (Instance).

## Instance

An instance is another body's types made anew at one use, and is made at that use's place, its call site (let polymorphism).

| Use | Instance |
|---|---|
| calls a free fn or a method of the binding group the use lies in, or of one enclosing it | none is made: the use takes that body's own types |
| calls any other free fn or method | one is made |
| calls a function value | it reads that function value (Dependency), and one is made for each lambda its Values names |
| passes a function value across a boundary into a given function type | one is made for each lambda the value's Values names, whose parameters and result are that type's |

A run, which starts at a function the embedding chooses, makes one instance of that function: the types the embedding declares for its inputs are put in the universals' places, and its result passes into the type the embedding declares for its result (programs).

## Solving

### Binding group

A binding group is solved as one constraint graph.

| | Definition |
|---|---|
| Constraints | The constraints the binding group's text records, without the text of the binding groups lying inside it, and the C each use in it records. |
| Universal | The type of a parameter. The use chooses it. A capture's type is not universal but an equivalence class of the enclosing binding group. |
| Unknown | An equivalence class in those constraints that is not universal, is not decided when solving starts, and is not of an enclosing binding group. |
| C | The equivalence classes left unsolved and the uses, lends, projections and bindings left undecided when the binding group reaches its fixpoint, and the constraints on them. |
| Order | A binding group is solved before the binding group in which a use records its C. |

When a use makes an instance, it makes anew the equivalence classes and the uses, lends, projections and bindings of C that are that binding group's own, puts its type arguments in the universals' places, and records them in its own binding group. Those copies are decided in the binding group they are recorded in. The type arguments are unknowns of the binding group the use lies in. The equivalence classes of an enclosing binding group are not made anew but shared.

### Meeting and boundary

Where a value passes into a type, it is a meeting when that type is an unknown of the binding group the place lies in, and a boundary otherwise. Where the type is that of a binding reached by name, its fields and parts included, and an unknown of a binding group enclosing the one the place lies in, it is a meeting too. A place that reaches an unknown of an enclosing binding group through a reference is a boundary. A body's result type is an unknown of that body, and its tail, its `return`s and its `?` exits are values passing into it. A template's result type is fixed to `String` and is not an unknown.

### Recording

The constraints a body's text gives are recorded in equivalence classes as soon as they are given, on no condition of what has been solved.

| Text | Recorded |
|---|---|
| a meeting | a lower bound `τ ≤ α` |
| a boundary | `T ⇝ U` |
| an invariant place | an equation |
| a projection | a kind |
| a use | The constraint that its types equal one of its candidates' signatures (Overload resolution), and where it makes an instance, that C. |

### Dependency

| What | What it depends on |
|---|---|
| an equivalence class | its unsolved lower bounds, and the relations not yet recorded into it by: (a) where it is a part of `τ` in a lower bound `τ ≤ α` or of `U` in a boundary `α ⇝ U`, `α` taking a constructor; (b) a use, lend, projection or binding at whose position it lies, or within the depth its candidates or pattern reach. |
| a part's equivalence class | every equivalence class it is a part of taking a constructor |
| a universal | the use that chooses it, that is, a use that makes an instance of that body; that use lies outside the binding group |
| a use | the uses, lends, projections and bindings that record relations into the equivalence classes at the positions it reads |
| a lend, a projection, a binding | the uses, lends, projections and bindings that record relations into the equivalence class of its place or operand, other than projections of a field, a component, an index or a variant, and uses that have already recorded whether that equivalence class is a reference |

The uses, lends, projections and bindings that record relations into an equivalence class are the uses, lends, projections and bindings that class, and the equivalence classes it reaches along dependencies between equivalence classes, depend on. A use, a lend, a projection or a binding depends on no equivalence class.

A use records whether an equivalence class is a reference even before it is decided. Where the types of the lower bounds or equations its candidates would record into one equivalence class are references in every candidate, it records at once that the class is a reference; where they are a reference in no candidate, it records at once that the class is not one. Where the candidates differ in this, it is recorded when the use is decided.

For example, under the two declarations `pick({x: i32}) -> {x: i32}` and `pick(u8) -> u8`:

```
let mut a;
a.x = 0;
a = pick(a);
```

- `pick`'s result `R` is a lower bound of `A`. The lower bounds the two candidates would record into `R`, `{x: i32}` and `u8`, are neither of them a reference, so that `R` is not a reference is recorded at once.
- The projection `a.x` reads only whether `A` is a reference, so it does not depend on `pick`. `A` is not a reference, and the kind "has field `x: τ`" applies to `A`.
- `pick` depends on that kind, so it reads `A` as `{x: τ}`, and `pick({x: i32})` remains.
- Were the second declaration `pick(u8) -> &u8`, the two candidates would differ in this, so `a.x` would depend on `pick` and `pick` on `a.x`. Each transitively depends on the other, so each breaks its dependency on the other, and the `A` that `pick` reads decides nothing, so the call is refused.

What a use, a lend, a projection or a binding becomes is decided by the value it reads. The value it reads is its equivalence class's value decided from the remaining relations alone, without the relations recorded by itself and by the uses, lends, projections and bindings that transitively depend on it. Here each equivalence class's value is decided by the table of values and by splits, whether or not what it depends on is solved, and an equivalence class with no relation remaining gives nothing. A use that calls a declaration reads the positions where its candidates differ, and a lend, a projection or a binding reads whether its equivalence class is a reference. In reading whether it is a reference, the relations recorded by projections of a field, a component, an index or a variant are not taken as recorded either (Place). A position where the value read decides nothing decides nothing.

The dependency graph has as its nodes the equivalence classes of the moment and the uses, lends, projections and bindings, and as its edges the dependencies the rows of the table above give. That `Y` transitively depends on `X` means that `Y` reaches `X` along the dependency graph. Each node breaks its dependency on itself and on every node that transitively depends on it. The relations recorded by a node that a broken dependency pointed to are recorded once the node that broke it is decided, and checked (Value).

The unbroken dependencies have no cycle. Were there one, for any node on it, the next node would transitively depend on it along that cycle, so that dependency would have been broken.

### Value

An equivalence class takes its constructor and members once everything its unbroken dependencies point to is solved or decided, and is then said to be solved; each part is solved on its own. A cycle of lower bounds alone is first merged into one equivalence class. Where an equivalence class reaches a part of its own along recorded lower bounds and parts, no finite type holds it, so it has no value and is split no further.

| | Value |
|---|---|
| with an equation | The type every equation gives. |
| without one | The join of the types its sources give, except a lower bound that a broken dependency points to: a lower bound gives its own type, a use the result type of the signature it chose, and a kind the member it names. |

A kind gives a member only where that member is visible to this body. An object's fields and an enum's variants are visible, since the body's text writes them; the members of `Option`, `Result` and `Error`, and of a type the host gives, are written outside and are not. An equivalence class whose only sources are kinds takes the constructor of the members they name. An equivalence class's value is a type that meets the kind it holds.

When an equivalence class takes a constructor, the relations between the parts of its lower bounds and boundaries are recorded in the parts' equivalence classes.

| Part | lower bound `τ ≤ α` | boundary `α ⇝ U`, `U`'s part an unknown | boundary `α ⇝ U`, otherwise |
|---|---|---|---|
| covariant | a lower bound | a lower bound | that part's `⇝` |
| invariant | an equation | an equation | an equation |

Where the constructors of the two sides of a lower bound or a boundary differ, the parts are linked through the signatures of the declared casts. `?N ⇝ U` unifies `?N` with `U`. A covariant part's `⇝` is a check made once everything is solved.

Once an equivalence class is solved, a value entering by a lower bound or a use's result reaches its value by the join's `≤`, and is changed at each step of that `≤` by a row of the Coercion table; but the fields an object gains are not changed, and are written before the value is used whole (ownership §4). A lower bound that does not reach by `≤` the value of an equivalence class whose value an equation decides, and a value entering by a boundary, reach that value by a coercion. Either way, its functions are called where the value enters.

### Fixpoint and fallback

A binding group is solved until no more equivalence classes are solved: its fixpoint. Inside a binding group, there is no fallback.

Once every binding group has reached its fixpoint, fallback is applied. An equivalence class that has no unbroken dependency and is unsolved takes its family's default if it is a LitVar, `i32` for an integer and `f64` for a float, and `!` otherwise. Fallback is not applied to the equivalence classes left in C, only to the copies a use makes anew. Solving then resumes to the fixpoint, and fallback and the fixpoint are repeated until no equivalence class is left to fall back.

A program is accepted where its declared casts follow the cast tree, every equivalence class that is not an original left in C has a value, every use, lend, projection and binding that is not an original left in C is decided, and every entering value reaches the value it enters. Any other program is refused at the place where a definition gives no result.

## Place

A place is a root, a binding, followed by projections, and its type is the equivalence class its last projection names.

A lend `&ₘ p` gives a reference. Where, in the value the lend reads, the place is `&ₙ U`, its target is `U` and its mutability the lesser of `m` and `n`; otherwise its target is the place's equivalence class. The relation to its target is an equation.

A projection other than a Deref whose operand is a reference applies to its target (auto-deref). Whether an equivalence class is a reference is decided by its sources other than the kinds of a field, a component, an index or a variant: its lower bounds, equations, uses and the kind of a Deref.

A pattern is projections of its scrutinee's place, and at each of its positions records:

| Position | Recorded |
|---|---|
| `(π̄)`, `{f̄: π̄}`, `&π`, `V(π)` | the projection at that position: Component, Field, Deref, Payload |
| a literal | a value passing into the type of that position |
| a binding | a read of that position where the path passes no reference; where it passes one, a lend of that position, at the least mutability of the references passed and of the position's own |
| `V` without a payload | the same as `V(_)` |
| `_` | nothing |

A `match` applies each arm's pattern to its scrutinee.

## Struct

`Self { f̄: ē }` is a value whose constructor is this module's struct and whose parts are equivalence classes `T̄` made anew at that place, and each `eᵢ` is a value passing into `Tᵢ`. A struct's fields are not members but its parts in the order it declares them, so a field's kind is unified with that part and adds nothing. The equivalence class of a receiver `self`, `&self` or `&mut self` carries the kind `Self⟨T̄⟩`, `&Self⟨T̄⟩` or `&mut Self⟨T̄⟩` as the text writes it, and `T̄` is universal.

`V(e)` is a value whose payload of `V` is an equivalence class `P` made anew at that place, and `e` is a value passing into `P`. Where `V` is a member of `Option` or `Result`, its constructor is `Option` or `Result` and its other part an equivalence class made anew; where it is `E::V(e)`, it is an enum named `E` whose only member is `V`. In a `V` written without a payload, `P` has no relation.

## Overload resolution

The candidates of a use that calls a declaration are the declarations its name reaches. An operator, `{{ e }}` and `e?` are uses the language sets, whose candidates are a lang item's (operators §2), `core::display`'s and the two the language gives. At a structural type, the candidate of a generic function is that function called at each part (types §1). The type parameters of a candidate that is an extern declaration are unknowns made anew at each use, of the binding group the use lies in.

A method call's receiver is its first argument, adjusted as written, then as `&`, then as `&mut` where the receiver is a place that may be written, and the first adjustment that reaches a candidate is taken (types §6).

A use with several candidates is recorded as the constraint "its types equal one of its candidates' signatures", and what all candidates share holds from this constraint at once.

The candidates are narrowed in the following order, and at each step only the candidates whose passing goes by a coercion of the lowest rank remain. Each position's type is the value the use reads (Dependency).

| Step | What passes | The type it passes into |
|---|---|---|
| each argument, in order | the argument's type | the candidate's parameter |
| the result | the candidate's result type | the type the result passes into |

A position that decides nothing reaches every candidate: a position the value the use reads does not decide, `!`, and a LitVar within its family. Narrowing takes no conversion (types §7). Where several candidates remain only because of a LitVar's position, that LitVar is taken as its family's default and the candidates are narrowed again. Where exactly one candidate remains, the use is decided as that candidate. A use whose argument is `!` is never carried out, so whatever candidates remain, it is decided with the result `!` (types §6).

## `?`

`e?` is a use with two candidates the language gives, at `Option⟨T⟩` and at `Result⟨T, E⟩`, and its result is `T`. Its exit is a value of `Option⟨!⟩` or `Result⟨!, E⟩`, and passes into the result of the innermost body, as a `return` does.

## Identity

An identity is a part of an extern-defined type, the name of the storage its values reach (ownership §6). A struct or a structural type carries no identity; only its parts may each carry one. A type carries at most one identity.

An identity in a declaration's result that no parameter's type carries is existential, and a call makes it anew where it lies. An identity a body makes stays in C even where its binding group solves it, and is made anew at each instance. Two different identities are never made one.

An execution is one call of a body or a lambda, or one iteration of a loop, and a level is its depth among nested executions. An equivalence class has the level of the place it arises at, and two merged into one have the shallower level. An identity a call makes anew has the level of the execution the call lies in.

An equivalence class holds neither two different identities nor an identity of a level deeper than its own. This is a constraint on its value, so where the sources of an equivalence class with no equation carry such identities, its value is the least type that meets the constraint, their nearest common ancestor that carries no identity, and each value is coerced to it.

## Examples

Below, `c` is a `bool`, and a function whose signature is in a comment is an extern declaration.

### 1. An if with information on one side

```
let a = if c { Some(1u8) } else { None };
```

Outcome: `a` is `Option⟨u8⟩`.

1. The values of both blocks meet `a`'s equivalence class `A`, recording `A ≥ Option⟨S⟩` and `A ≥ Option⟨?k⟩`. `S` is the payload `Some` makes anew, with `S ≥ u8`, and `?k` is the payload `None` makes anew, with no relation.
2. Both lower bounds are solved with the constructor `Option`, so `A` takes `Option`, and its payload `P` records `P ≥ S` and `P ≥ ?k`.
3. `P` depends on its unsolved lower bound `?k`, and `?k` depends on nothing, so fallback gives `!` to `?k` alone.
4. `S = u8`, so `P = u8 ⊔ ! = u8`.

### 2. A boundary's invariant part

```
let a = Vec::new();      // new() -> Vec⟨T⟩
fill_all(&mut a);        // fill_all(&mut Vec⟨u8⟩)
```

Outcome: `a` is `Vec⟨u8⟩`.

1. `new`'s `T` is an unknown `?t` made anew at this use, and `A ≥ Vec⟨?t⟩` is recorded.
2. `&mut a` gives a `&mut` whose target is `A`. Its value passes into `&mut Vec⟨u8⟩`, which the declaration gives, so it is a boundary.
3. A reference's target is invariant, so the boundary splits into the equation `A = Vec⟨u8⟩`.
4. `A` is the equation's type `Vec⟨u8⟩`, and against its lower bound `Vec⟨?t⟩` the invariant part gives the equation `?t = u8`.

### 3. A conversion where an equation wins

```
let mut buf = Vec::new();
push(&mut buf, 1u16);    // push(&mut Vec⟨T⟩, T)
consume(buf);            // consume(Vec⟨u8⟩)
```

Outcome: `buf` is `Vec⟨u8⟩`, and the declared conversion `u16 → u8` is called at `push`'s second argument. Without that conversion, it is refused.

1. As in 2, `B ≥ Vec⟨?t⟩`.
2. `push`'s `T` is a new unknown `?T`. The first argument is `&mut B ⇝ &mut Vec⟨?T⟩`, and the target is invariant, so `B = Vec⟨?T⟩`, that is, `?t = ?T`. The second argument passes into the unknown `?T`, so it is a meeting: `?T ≥ u16`.
3. `consume(buf)` is `B ⇝ Vec⟨u8⟩`, and `Vec`'s argument is invariant, so it gives the equation `?t = u8`.
4. The value of `?t`, which is `?T`, is the equation's `u8`. The lower bound `u16` does not reach `u8` by `≤`, so it must reach it by a coercion, and since no cast reaches it, the declared conversion is called at `push`'s second argument, where that value enters.

### 4. A covariant boundary is a check

```
let x = None;
let y = if c { x } else { Some(1u16) };
take(x);                 // take(Option⟨u8⟩)
```

Outcome: `x` is `Option⟨!⟩` and `y` is `Option⟨u16⟩`.

1. `X ≥ Option⟨?k⟩`, `Y ≥ X`, `Y ≥ Option⟨S⟩` and `S ≥ u16`.
2. When `X` takes `Option`, its payload `P` records `P ≥ ?k`. `take(x)` is the boundary `X ⇝ Option⟨u8⟩`; the payload is covariant and `u8` is not an unknown, so `P ⇝ u8` is recorded, a check made once everything is solved.
3. `?k` has no source, so fallback gives it `!`, and `P = !`.
4. `Y`'s payload `Q` has `Q ≥ P` and `Q ≥ S`, and `S = u16`, so `Q = u16`.
5. The check `! ⇝ u8` holds. Had the boundary's covariant part been recorded as an equation, `Q` would be `u8 ⊔ u16`, and it would be refused.

### 5. A lower bound reaching a part of its own

```
let mut x = None;
x = Some(x);
```

Outcome: refused.

1. `X ≥ Option⟨?k⟩`, and `Some(x)` is an `Option⟨S⟩` with a new payload `S` and `S ≥ X`, so `X ≥ Option⟨S⟩`.
2. When `X` takes `Option`, its payload `P` records `P ≥ ?k` and `P ≥ S`.
3. `X` reaches `P` along a part, `P` reaches `S` along a lower bound, and `S` reaches `X` again along a lower bound. `X` appears again inside a part of its own, so no finite type holds it, and it is refused.

### 6. A capture written by name

```
let mut last = None;
let remember = |x| last = Some(x);
remember(1u8);
```

Outcome: `last` is `Option⟨u8⟩`.

1. `L ≥ Option⟨?k⟩`.
2. The lambda is a binding group of its own, and `x`'s type `X` is universal. `last` is reached by name and its type `L` is an unknown of the enclosing binding group, so the write is a meeting: `L ≥ Option⟨S⟩` with `S ≥ X`. This lower bound rests on `X` through `S` and stays in the lambda's C.
3. `remember(1u8)` calls a function value whose Values is this lambda alone, so it makes an instance. `X` becomes a new unknown `X'` of the enclosing binding group, which the argument meets: `X' ≥ u8`. The copy of C is `L ≥ Option⟨S'⟩` with `S' ≥ X'`, `L` shared, not made anew.
4. `L`'s payload `P` has `P ≥ ?k` and `P ≥ S'`. `?k` is `!` by fallback, and `S'` has `X' = u8` as its lower bound and is `u8`, so `P = u8`.

### 7. A write through a reference

```
let mut x = None;
let r = &mut x;
*r = Some(1u8);
```

Outcome: `x` is `Option⟨u8⟩`.

1. `X ≥ Option⟨?k⟩`. `X` is not a reference, so `r`'s type is a `&mut` whose target is `X`.
2. `*r`'s type is `r`'s target, `X`. The write lies in the same binding group and `X` is its unknown, so it is a meeting: `X ≥ Option⟨S⟩` with `S ≥ u8`.
3. `X`'s payload has `?k` and `S` as lower bounds, and with `?k` `!` and `S = u8`, it is `u8`.

The same write made inside a lambda through a captured `r` is a boundary: although `X` is an unknown of the enclosing binding group, the place is reached through a reference.

### 8. Kinds and visibility

```
let a;
a.x = 0;
```

Outcome: `a` is `{x: i32}`.

1. The projection `a.x` gives `A` the kind "has field `x: τ`", and the write is `τ ≥ ?{0}`.
2. `A`'s only source is the kind. The field is visible, since this body's text writes it, so `A` takes the object, the constructor of that member, and is `{x: τ}`.
3. `τ` has only a LitVar, so fallback makes it `i32`.

```
@cfg.extra = 1;          // @cfg is { name }, given by the host
```

Outcome: refused. `@cfg`'s members are written by the host and are not visible, so the kind "has field `extra`" adds no member. No type has the conjunction of `{name}` and that kind.

```
if let Ok(v) = Some(1u8) { }
```

Outcome: refused. `Option`'s members are written by the language and are not visible, so the kind "has variant `Ok` with payload `τ`" adds no member.

### 9. A lend that reborrows a reference

```
let mut v = Vec::new();
let r = &mut v;
let s = &r;
```

Outcome: `s` is a `&` whose target is `v`'s equivalence class.

1. `r`'s type is a `&mut` whose target is `V`.
2. `&r` lends the place `r`. Its equivalence class is the reference `&mut V`, so the target is `V`, and the mutability the lesser of shared and exclusive, shared.

### 10. Recursion by name

```
fn g(a, b) { g(b, a) }
```

```
g(1u8, "s")
```

Outcome: refused at the recursive call inside the instance of `g(1u8, "s")`.

1. `g`'s recursive call is a call within its binding group, so it makes no instance and takes `g`'s own types. The parameters' types `A` and `B` are universal, not unknowns, so where the arguments pass into them are boundaries: `B ⇝ A` and `A ⇝ B`. Both rest on universals and stay in `g`'s C.
2. `g(1u8, "s")` is a call from another binding group, so it makes an instance. `A'` and `B'` are new unknowns that the arguments meet: `A' = u8` and `B' = &str`.
3. The copied boundary `&str ⇝ u8` has neither a composition nor a declared conversion that reaches it, and it is refused.

Recursion within one binding group makes no new instance, so a recursion that swaps its parameters' types is not admitted.

### 11. Identity

```
let a = open();                     // open() -> F⟨new⟩; a cast from F to G, which carries no identity, is declared
at(&a, index_of(&a));               // index_of(&F⟨p⟩) -> Idx⟨p⟩, at(&F⟨p⟩, Idx⟨p⟩)
let b = if c { open() } else { a };
let mut v = Vec::new();
while c { push(&mut v, open()); }   // push(&mut Vec⟨T⟩, T)
```

Outcome: the second line is admitted with one identity, `b` is `G` and `v` is `Vec⟨G⟩`. Without the cast to `G`, `let b` and the `push` are refused.

1. The identity `ι₀` the first `open()` makes belongs to this body's execution, and `a: F⟨ι₀⟩` is at the same level.
2. `index_of`'s `p` is a new unknown of this use. At the boundary where `&F⟨ι₀⟩` passes into `&F⟨p⟩`, the target and the identity are invariant, so `p` is `ι₀` and the result is `Idx⟨ι₀⟩`. `at`'s `p` fits as the one `ι₀` too.
3. The `if`'s result `B` records `F⟨ι₁⟩` and `F⟨ι₀⟩` as lower bounds. `B` holds no two different identities, so its value is their nearest common ancestor that carries no identity, `G`, and each value is cast to `G`.
4. The `ι₂` the `open()` in the loop makes belongs to its iteration. `push`'s `T` arises at this use, but `&mut Vec⟨T⟩`'s target is invariant, so it merges with `v`'s element and has the shallower, outer level. That equivalence class holds no deeper `ι₂`, so its value is `G`, and each value put in is cast to `G`.

### 12. Accumulation that starts from a literal

```
let mut x = 0;
let mut y = 1u16;
while c {
    x = x + y;
    y = y + 1;
}
```

Outcome: `x`, `y` and both literals are `u16`.

1. `X ≥ ?a`, `X ≥ R₁`, `Y ≥ u16` and `Y ≥ R₂` are recorded. `?a` and `?b` are the LitVars of `0` and `1`, and `R₁` and `R₂` the results of `x + y` and `y + 1`.
2. `x + y` is a use of `core::add(&x, &y)`, and its candidates, the impls of `add<T, O>(&T, &T) -> O`, differ at `T`. So does `y + 1`.
3. Both uses record relations into `Y`, the target of `&y`, and read `Y`, so each transitively depends on the other. So the value each use reads has the relations of neither use. In that value `Y` is the `u16` the text gives, and `R₁` and `R₂` have no relation. Each lend reads only whether its place is a reference, and since none of `add`'s candidates records a reference into `X`, `Y`, `R₁` or `R₂`, it does not depend on the two uses, and decides that its place is not a reference.
4. In `y + 1`, `Y`'s position is `u16` and `?b`'s decides nothing, so `u16`'s `add` remains. In `x + y`, `X`'s position is the LitVar `?a` alone and decides nothing within the integer family, and `Y`'s is `u16`, so `u16`'s `add` remains.
5. By their relations `X` and `Y` get the equation with `u16`, and the lower bounds `?a` and `?b` are unified with `u16`. `R₁` and `R₂` are `u16`, and as lower bounds of `X` and `Y` they reach those values as they are.

### 13. An overload chosen by its result

```
fill_all(&mut parse(s));     // two declarations, parse(&str) -> Vec⟨u8⟩ and parse(&str) -> Vec⟨u16⟩; fill_all(&mut Vec⟨u8⟩)
let _ = &mut parse(s);
```

Outcome: the first line's `parse` is `→ Vec⟨u8⟩`, and the second line is refused.

1. `Vec⟨u8⟩` and `Vec⟨u16⟩`, which `parse`'s two candidates would record into its result `R`, are neither of them a reference, so that `R` is not a reference is recorded at once. So the first line's lend does not depend on `parse`, and takes `R` as its target. That reference passes into `&mut Vec⟨u8⟩`, which the declaration gives, so it is a boundary; the target is invariant, so it gives the equation `R = Vec⟨u8⟩`.
2. The two candidates do not differ by their arguments, so they are narrowed by the result. The value of `R` that `parse` reads is the equation's `Vec⟨u8⟩`, and only `→ Vec⟨u8⟩` reaches it.
3. In the second line, the value of `R` that `parse` reads decides nothing. Both candidates remain, and the call is refused.

### 14. A struct, and recursion that grows

```
struct Pair { l, r }
impl Pair {
    fn new(l, r) { Self { l, r } }
    fn swap(self) { Self { l: self.r, r: self.l } }
    fn nest(self, n) { if n == 0 { 0u8 } else { Self { l: self, r: 0u8 }.nest(n - 1) } }
}
```

```
let p = Pair::new(1u8, "s");
let q = p.swap();
q.nest(3)
```

Below, `Pair⟨A, B⟩` is the `Self` whose fields `l` and `r` are of types `A` and `B`.

Outcome: `p` is `Pair⟨u8, &str⟩` and `q` is `Pair⟨&str, u8⟩`, and `q.nest(3)` is refused at the recursive call inside its instance.

1. In `new`, `Self { l, r }` makes a `Pair⟨P, Q⟩` whose parts are new equivalence classes `P` and `Q`, with `P ≥ L` and `Q ≥ R`. `L` and `R` are parameters' types, universal, so these stay in C. In the instance of `Pair::new(1u8, "s")` their copies meet the arguments, so `p` is `Pair⟨u8, &str⟩`.
2. `swap`'s receiver carries the kind `Pair⟨A, B⟩`, and `self.r` and `self.l` are the parts `B` and `A`. So its result is a `Pair⟨P, Q⟩` with `P ≥ B` and `Q ≥ A`. At the boundary where `p.swap()` passes `p` into the receiver, a struct's parts are invariant, so the copies of `A` and `B` are `u8` and `&str`, and `q` is `Pair⟨&str, u8⟩`.
3. `nest`'s recursive call is a call within its binding group, so it makes no instance. Its receiver `Self { l: self, r: 0u8 }` is a `Pair⟨P, Q⟩` with `P ≥ Pair⟨A, B⟩` and `Q ≥ u8`. Where this value passes into the receiver `Pair⟨A, B⟩` is a boundary, and a struct's parts are invariant, so it gives the equations `P = A` and `Q = B`. Then the lower bound `Pair⟨A, B⟩` must reach `A` and `u8` must reach `B`, and since both rest on universals, they stay in C.
4. In the instance of `q.nest(3)`, the copies of `A` and `B` are `&str` and `u8`. `n == 0` and `n - 1` keep several candidates only because of LitVars' positions, so the LitVars are taken as the default `i32` and both narrow to `i32`.
5. In the copy of C, the lower bound `Pair⟨&str, u8⟩` must reach `&str`, but neither a composition nor a declared conversion reaches it. So it is refused at that recursive call.
