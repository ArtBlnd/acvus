# control flow

Control flow means what it means in Rust: blocks, `if`, `match`, `while`, `for`, `break`, `continue` and `return`, with a block that ends in a diverging statement typed `!`.

Where Rust decides a construct by a trait, the language decides it by a type, as types §6 states:

- `for` iterates by the kind its source's type gives: a reference to an array or a `Vec` iterates by reference (`&mut` for a mutable one), an array or a range by value.
- `?` applies to a `Result` or an `Option`. Its operand's type may become known after the `?`; a `?` whose operand never becomes one of the two is refused.
