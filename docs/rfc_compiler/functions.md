# functions

## 1. Declarations

How a function is declared depends on the kind of source (programs §1-3).

## 2. Lambda

`move |…| -> e` moves each captured name into the closure when the closure is made. A lambda without `move` captures each name by reference for as long as the closure lives.

A lambda expression is a position whose head the text writes: a function value whose one member is the closure of that lambda ([the type system](../transparency-constraint-type-system.md), O29). A lambda expression is evaluated once, and its environment captured once, where it is written. How each name is captured is decided there, by how the body uses it, and never by a use of the value. A use of a function value is a Call constraint on a class, or an external function type a class passes into; for each closure that enters the class and each use on it, an instance of the lambda's text is recorded, whose captures are the classes the closure holds, and whose parameter and result classes are unified with the Call's arguments and result, or with the function type's parameters and result (O30). In each instance the classes of the lambda's own text are made anew, and every other class its text names is that same class (O26): a lambda's captures are the enclosing body's classes, and a capture written by a lambda is decided with every instance's writes (the type system, Consequences). The lambda's text is also recorded once more, with every class it names made anew, and settled to the fixpoint without occurrences and without fallback (O31): its refusals that depend on no use and no capture are found once, at their positions, and its parameters and captures are left to its instances (the type system, Consequences). Each use runs code specialized to its instance over the one environment. The head decided for a class has, at each use on it, an instance of each of its members (O30), so a closure that only a decision lets enter a class is refused at its uses (the type system, Consequences).

## 3. Once and many rules

A closure whose body moves a captured value out can be called once; every other closure can be called many times. An extern fn's closure parameter is called once or many times, as its declared function type states.

## 4. Recursion

Only a `fn` may recurse, and only a module declares one (programs §3). The calls among a graph's units, and within them, are seen whole: a cycle of calls through anything but `fn`s is refused, a lambda's or a template's included. The types of `fn`s that reach one another by calls are decided as one binding group (programs §3; O27).
