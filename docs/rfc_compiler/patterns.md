# patterns

## 1. Patterns

Patterns mean what they mean in Rust.

## 2. Growth

A pattern that names a variant grows the enum it matches, under the transparency rule: only where every use of the enum is visible. An enum that is not visible there, an extern-defined one included, does not grow, and a pattern naming a variant it does not have is refused.

## 3. Exhaustiveness

Exhaustiveness follows the transparency rule too. Where every variant the scrutinee may hold is visible to the compiler, a `match` covers them all. Where they are not, as when the scrutinee comes from a function's argument, the `match` needs a `_` arm.

## 4. A scrutinee of type `!`

A scrutinee of type `!` reaches no arm. Each slot of its pattern is `!`, and no exhaustiveness question arises.
