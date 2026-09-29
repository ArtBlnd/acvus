# types

## 1. Transparency

Every type is written somewhere: in a body of the program, in a declaration, or by the host. Every value is held by one side at a time, and that side sees every use of the value while it holds it: the value is transparent there.

A type is widened, joined or unioned only where it is transparent: in the body that writes it, and there only at a use reached by name, a binding or a field or part reached from one by name, since only there is every value of it made and every use of it seen. A use reached through a reference reaches some place of the reference's target type, not which one, so it does not grow that type. A type a declaration or the host writes never grows: it equals only itself and crosses exactly as declared. A module's `Self` is transparent in its module, which sees every use of it; outside the module its fields are not seen at all (programs §3). A type parameter of an extern fn is parametric, so a program's type passes through it transparently. Where values of one type the program writes meet in one slot, the type grows to hold each of them, and a path on which a value lacks what the grown type holds is refused where that value is used whole (ownership §4).

A type is never path dependent. Every value, binding and slot has one type, the same at every point and on every path of the program. Values that meet in a slot where it is transparent, directly or through a chain of slots, are of one type, fixed over the whole program. That a type grows means only that this one type is decided from every value that meets in it, wherever in the program they meet: it never changes along a path, and no point sees a type other than another point sees. What differs along a path is only which parts of a value have been written (ownership §4). Where a value is coerced instead (§3), the coercion makes a value of the slot's type, and the value coerced keeps its own.

A value entering a slot is decided by whether the slot's type is transparent there. Where it is, the type grows to hold the value, as above. Where it is not, the value is coerced to it (§3) by the body the value is in, which holds the value and sees every use of it. So no side grows a type another wrote, and nothing grows a type through a reference:

| A value | enters a type written by | and so |
|---|---|---|
| an argument, into a function's parameter | the function | is coerced by the caller |
| a result, into its function's result | the function | grows it |
| a function's result, where its caller takes it | the function | is taken as it is: no caller widens it |
| an argument, into an extern fn's parameter | the declaration | is coerced by the caller |
| an extern fn's result, where its caller takes it | the declaration | is taken as declared |
| a value stored into a context | the host | is coerced by the body |
| a context's or an input's value, where a body reads it | the host | is taken as it is |
| a value stored through a reference | whoever wrote the storage's type | is coerced by the body: the reference does not name the storage |

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

The side that holds such a value sees all of it, and its shape is the language's own, so its meaning is its parts'. It has an impl of a generic function (extern fn §6) where the function's meaning at the type maps one to one onto its parts, every part has an impl of it, and that meaning relies on no law the parts' impls state. An enum, `Option` and `Result` hold one variant, so there the result's variant must follow from the value's: the function takes the type in exactly one parameter, and its result is that argument's variant. `clone` is such a function at every one of the six. Where the function's meaning at the type instead combines its parts' answers, by a combination the language defines, it has an impl only where every part's impl states a law (extern fn §3) that the combination keeps, and that impl states the law in turn. `eq` is such a function: a structure's equality is its parts' taken together, and equivalences taken together are an equivalence, so it has an impl of `eq` where every part's `eq` is declared an equivalence. Without the law it has none, since whether its equality would be partial or total cannot be known.

## 2. Join

Two object types join at the union of their fields. Two enums of one name join at the union of their variants; enums of two names do not join. `Option` and `Result` are primitive types, closed like an extern enum; a variant is written with its enum's name, except their `Some`, `None`, `Ok` and `Err`. Two function types join at the join of their effects and of their flows, which form lattices, are once where either is (functions §3), and may be split only where both sides may, at one cost (effects §6), so a joined function is never assumed to do less than either side. A function type also states which function values it may be: each lambda is of its own, and a join is of either side's; declared facts (extern fn §3) do not join, and a joined function carries none.

## 3. Subtyping

A declared extern cast from `S` to `T` makes `S` a subtype of `T`. The declared casts and the views (operators §5) form an order with no cycle: a type may have several next steps up, and no type is above itself. Where values of different types meet in one slot, the slot takes their nearest common ancestor, and each value is coerced to it. Where two nearest common ancestors fit, the meeting is ambiguous and refused.

An identity is a binding a value's type carries: it never changes, two identities are known to be one or not, and two different identities are never made one. A type carries at most one identity.

Only the compiler makes an identity. A call whose result's type carries an identity its parameters do not give makes a fresh one, one per call site in each instantiation of its body. A declaration may give its result the identity of a parameter, to one value or to several; it never puts two identities into one type. A value carrying an identity made inside a loop, live across that loop's next iteration, keeps it no longer: it takes its nearest ancestor that carries no identity, where there is one, and is refused where there is none.

Values of one identity keep it where they meet; values of two identities meet at their nearest common ancestor that carries no identity, and where there is none, they do not meet.

Type constructors stay invariant: subtyping applies to a value entering a slot, not to a constructor's argument.

A value of a transparent structured type coerced to a slot of the same kind (§1) coerces by its kind's shape, and part by part down to the parts that are not one. Each of those parts, a concrete type, coerces as any value of its type is coerced to a slot: by the order above and by §7. The shape coerces by its kind:

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

A type becomes known in two steps. Its constructor is known first: that it is an object, an enum of a name, a tuple of an arity, an array, a reference, a number of a width, and so on, once no value meeting in it can still move it up the order of §3. Its members, the fields of an object and the variants of an enum, are known after, once no value can still meet in it anywhere in the program (§1). A decision waits only for what it reads. Which declaration a name or a method call reaches, what a `for` iterates, what `?` unwraps, and whether a value entering a slot is folded, coerced or converted read constructors, and are decided once those are known. How a value is coerced by its kind's shape (§3), a comparison part by part (operators §4), and a generic function answered at a structure's parts (§1) read members, and are decided once those are known. No decision is made on what may still change, so none is made again, and a type is decided once, the same wherever it is read (§1).

A call reaches by its arguments in order, and a tuple by its parts in order: at each one, stepping up from its exact type, the declarations that match at the first step where any match are kept. Where more than one is left after them, the type the call's result is taken at decides last: those whose result matches it exactly are kept. Exactly one must be left; otherwise the call is refused.

A method call's receiver is its first argument, adjusted as Rust adjusts it: taken as written, then as `&`, then as `&mut` where the receiver is a place that may be written. The first adjustment that reaches a declaration is taken.

## 7. Conversions

Where a value's type is not the type its slot asks for and no subtyping relates them, the value converts only by a declared conversion, and exactly one must apply. Otherwise the program is refused, and the refusal names the conversions that left it undecided. A call reaches its declaration by subtyping alone (§6); a conversion applies once the slot's type decides it. A qualified name reaches one declaration, so a conversion can always be decided by writing one.

## 8. Specialization

A type argument may carry `#`, the mark that it can always be specialized; an argument without it is never specialized. `#τ` and `τ` are different types, and `#τ` is below `τ`: a type whose argument carries `#` steps up in the order of §3 to the same type with that argument unmarked, so a value of `F<#τ>` is one of `F<τ>` by that step, and never the reverse. `#` stands only at the root of a type argument: `T<#(X, Y)>` is a type, `T<(X, #Y)>` is not.
