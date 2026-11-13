# Programs the model does not come to yet

Each file here states a program and what the design gives for it, where the model gives
something else. `cargo run --bin check -- wanted` says of each whether it holds yet.

What these programs rest on:

A decision closes what it decides. Nothing is closed while a constraint that may still say
something of it is open, and where nothing more closes, at the fixpoint, fallback closes
what is left.

The type of a lambda has no brand: it is seen by its signature alone. It is closed once
what its body says without a parameter is decided and every bound on a parameter is
against a type that is decided, so a lambda waits for each class its body names outside
it, and is used only once its type is closed. A use instantiates that type, and what the
use is passed to says nothing to the body.

Where lambdas meet, their type is the intersection. Results that are decided are one type
where they meet, each parameter keeps the bounds of every lambda, and an argument that
does not keep them is refused where it is passed.

A program that comes to hold is moved into `src/text/tests.rs`, and this directory goes when
it is empty.
