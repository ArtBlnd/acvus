# functions

## 1. Declarations

How a function is declared depends on the kind of source (programs §1-3).

## 2. Lambda

`move |…| -> e` moves each captured name into the closure when the closure is made. A lambda without `move` captures each name by reference for as long as the closure lives.

## 3. Once and many rules

A closure whose body moves a captured value out can be called once; every other closure can be called many times. An extern fn's closure parameter is always many.

## 4. Recursion

Only a `fn` may recurse. A lambda may not.
