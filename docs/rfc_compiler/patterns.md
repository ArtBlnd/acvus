# patterns

## 1. Patterns

Patterns mean what they mean in Rust.

## 2. Variants

A pattern is no position of the class of the value it matches: a class's positions are the bindings and expressions of the text whose value has its type, and its head is decided from what enters it ([the type system](../transparency-constraint-type-system.md), Equivalence classes; O3). So a pattern gives that class no head and adds no variant to the enum it matches. A pattern that binds a variant's payload, as `x` in `if let Some(x) = o`, records on the scrutinee's class the structural constraint "has variant `V` with payload `τ`" (the type system, Structural constraints), which gives the payload a class, decides no head, and holds of the head decided, which has every part it names (O7); such a pattern naming a variant the head decided does not have is refused (O22).

## 3. Exhaustiveness

Exhaustiveness is read off the scrutinee's type as the type system decides it: the head decided for an enum's class holds its variants as its members (the type system, Types and Equivalence classes). Where every variant the scrutinee may hold is visible to the compiler, a `match` covers them all. Where they are not, the `match` needs a `_` arm.

## 4. A scrutinee of type `!`

A scrutinee of type `!` reaches no arm. No slot of its pattern holds a value, so each decides nothing of its type (types §4), and no exhaustiveness question arises.
