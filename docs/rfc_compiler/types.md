# types

## 1. Transparency

Every type is written somewhere: in a body of the program, in a declaration, or by the host. Every value is held by one side at a time, and that side sees every use of the value while it holds it: the value is transparent there.

A type is widened, joined or unioned only where it is written, and only where the program writes it, since only there is every value of it made and every use of it seen. A type a declaration or the host writes never grows: it equals only itself and crosses exactly as declared. A type parameter of an extern fn is parametric, so a program's type passes through it transparently. Where values of one type the program writes meet in one slot, the type grows to hold each of them, and a path on which a value lacks what the grown type holds is refused where that value is used whole (ownership §4).

A value entering a slot is decided by who wrote the slot's type. Where the body the value is in wrote it, the type grows to hold the value, as above. Where anything else wrote it, the value is coerced to it (§3) by that body, which holds the value and sees every use of it. So no side grows a type another wrote:

| A value | enters a type written by | and so |
|---|---|---|
| an argument, into a function's parameter | the function | is coerced by the caller |
| a result, into its function's result | the function | grows it |
| a function's result, where its caller takes it | the function | is taken as it is: no caller widens it |
| an argument, into an extern fn's parameter | the declaration | is coerced by the caller |
| an extern fn's result, where its caller takes it | the declaration | is taken as declared |
| a value stored into a context | the host | is coerced by the body |
| a context's or an input's value, where a body reads it | the host | is taken as it is |

For example, a module's `fn` reads two fields of its parameter, and a script passes it an object of three:

```acvum
fn sum(p) { p.a + p.b }
```

```acvus
let x = { a: 1, b: 2, c: 3 };
sum(x)
```

`sum`'s scheme is settled in its module alone (programs §3), so its parameter is `{ a, b }`, and inside `sum` nothing else exists. Object types are invariant (§4): `{ a, b, c }` is not a subtype of `{ a, b }`, so no value reaches `sum` holding a field its type does not name. Were one to, what `sum` decides at `{ a, b }` would not hold of it: `p.clone()` clones `a` and `b`, and a hidden `c` may have no clone. The parameter's type is `sum`'s, so the script, which holds `x`, narrows it to `{ a, b }` at the call (§3). Neither side reaches into the other: the script does not add `c` to `sum`'s parameter, and `sum` does not narrow a value whose uses it cannot see.

The transparent structured types are six: the object, the enum, the tuple, the array, `Option` and `Result`. No other type is one. Each is transparent recursively: the side that holds a value of one sees each of its parts, and each part that is itself one of the six, at any depth. Only the object and the enum grow; the tuple, the array, `Option` and `Result` have one shape wherever they are written, so where they are written makes no difference to them.

The side that holds such a value sees all of it, and its shape is the language's own, so its meaning is its parts'. It has an impl of a generic function (extern fn §6) where the function's meaning at the type maps one to one onto its parts, every part has an impl of it, and that meaning relies on no law the parts' impls state. `clone` is such a function. Where the function's meaning at the type instead combines its parts' answers, by a combination the language defines, it has an impl only where every part's impl states a law (extern fn §3) that the combination keeps, and that impl states the law in turn. `eq` is such a function: a structure's equality is its parts' taken together, and equivalences taken together are an equivalence, so it has an impl of `eq` where every part's `eq` is declared an equivalence. Without the law it has none, since whether its equality would be partial or total cannot be known.

## 2. Join

Two object types join at the union of their fields. Two enums of one name join at the union of their variants; enums of two names do not join. `Option` and `Result` are primitive types, closed like an extern enum; a variant is written with its enum's name, except their `Some`, `None`, `Ok` and `Err`. Two function types join at the join of their effects and of their flows, which form lattices, are once where either is (functions §3), and may be split only where both sides may, at one cost (effects §6), so a joined function is never assumed to do less than either side; declared facts (extern fn §3) do not join, and a joined function carries none.

## 3. Subtyping

A declared extern cast from `S` to `T` makes `S` a subtype of `T`. The declared casts and the views (operators §5) form a tree: every type has at most one next step up, so every order they give is total. Where values of different types meet in one slot, the slot takes their nearest common ancestor, and each value is coerced to it. Values of one identity keep it; values of two identities meet at their nearest common ancestor that carries no identity, and where there is none, they do not meet. No identity is made to join two others.

Type constructors stay invariant: subtyping applies to a value entering a slot, not to a constructor's argument.

A value of a transparent structured type coerced to a slot of the same kind (§1) coerces by its kind's shape, and part by part down to the parts that are not one. Each of those parts, a concrete type, coerces as any value of its type is coerced to a slot: by the tree above and by §7. The shape coerces by its kind:

- An object narrows to the fields the slot names. The value holds each of them; the others end where it is narrowed (ownership §6).
- An enum widens to the variants the slot names. The slot names every variant the value may hold.
- A tuple keeps its arity, an array its length, and `Option` and `Result` their variants.

The coercion makes a value of the slot's type. No type becomes a subtype of another, and the constructors stay invariant.

## 4. Bottom and invariance

`!` is the bottom type, and every type constructor is invariant. No type is `!` by holding a part of type `!`. A reference's target is never a reference.

## 5. Defaults

Defaults apply only where nothing more can be decided. A type variable nothing constrains is `!`, an effect nothing constrains is opaque, a function type's flows nothing decides state every flow (as a declaration's default does, extern fn §4), a function value's loans nothing decides are none and how often it may be called is many (functions §3), and an argument whose `#` nothing decides is not specialized. A variable a bound constrains and no use answers is refused. An unsuffixed integer literal is `i64` where no use decides its width.

## 6. Deciding by type

What a `for` iterates, what `?` unwraps, and which declaration a name or a method call reaches are decided by a type, wherever in the program that type becomes known. A decision reads a type only once nothing can change it, and what the decision leads back into that type must fit it unchanged. The outcome does not depend on the order the program is checked in. A construct whose type never becomes known is refused.

A call reaches by its arguments in order, and a tuple by its parts in order: at each one, stepping up from its exact type, the declarations that match at the first step where any match are kept. Exactly one must be left; otherwise the call is refused.

A method call's receiver is its first argument, adjusted as Rust adjusts it: taken as written, then as `&`, then as `&mut` where the receiver is a place that may be written. The first adjustment that reaches a declaration is taken.

## 7. Conversions

Where a value's type is not the type its slot asks for and no subtyping relates them, the value converts only by a declared conversion, and exactly one must apply. Otherwise the program is refused, and the refusal names the conversions that left it undecided. A call reaches its declaration by subtyping alone (§6); a conversion applies once the slot's type decides it. A qualified name reaches one declaration, so a conversion can always be decided by writing one.

## 8. Specialization

A type argument may carry `#`, the mark that it can always be specialized; an argument without it is never specialized. `#τ` and `τ` are different types. `#` stands only at the root of a type argument: `T<#(X, Y)>` is a type, `T<(X, #Y)>` is not.
