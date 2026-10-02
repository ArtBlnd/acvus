# limitations

## 1. Keeping

A handler keeps nothing it was lent past its call; what it keeps, it takes by value. Every value a call lends a handler carries the call's lifetime in its Rust type, so Rust refuses keeping one. No runtime type identity is read at the boundary.

An extern-defined result may keep arguments taken by value, with the loans they hold, a function value's captures included. The declaration's flows map each such loan to a lifetime of the result, and the checker holds it while the result is live. A kept loan is a reference value, never a Rust reference, and is opened only within a call (crossing §1).
