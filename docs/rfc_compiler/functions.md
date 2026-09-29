# functions

## 1. Declarations

How a function is declared depends on the kind of source (programs §1-3).

## 2. Lambda

`move |…| -> e` moves each captured name into the closure when the closure is made. A lambda without `move` captures each name by reference for as long as the closure lives.

A function value bound by `let` is polymorphic, as a module's `fn` is. Its expression is evaluated once, and its environment captured once, where it is written. How each name is captured is decided there, by how the body uses it, and never by a use of the value. What its type leaves unconstrained is generalized, except what the enclosing scope or the captured environment holds: those belong to where they were written, by the transparency rule (types §1). A decision its body cannot yet make becomes a requirement of its scheme, carrying where it arose. Each use instantiates the scheme, makes each requirement there, as a declaration's requirements are reached, and runs code specialized to that instance over the one environment. Two function values joined into one slot require what either requires.

## 3. Once and many rules

A closure whose body moves a captured value out can be called once; every other closure can be called many times. An extern fn's closure parameter is always many.

## 4. Recursion

Only a `fn` may recurse. A lambda may not.
