# 12 Transcript by `concat`: expected structure

The owner's chat-app loop: `text = text.concat(…)` over the history. It is the same monoid as `text = text + …` (R08), spelled through the extern `string::concat`, which declares no law: it is `concat(a: &str, b: &str) -> String`, and RFC-0082 rule 2 states a law only over `f(a: T, b: T) -> T`.

| line | loop | expected | why |
|---|---|---|---|
| 16 | `for m in &history` (build the transcript) | S1 free {`m.role != "system"`}; S2 Storage(`text`) InOrder, law Concat through `string::concat`, and Storage(`turns`) AnyOrder `+`, with the label, which the arm both cycles lie under computes | `concat(a, b)` is associative with identity `""` and not commutative: chunks concatenate their turns and join in order. The skip arm leaves `text` as it was, so the identity stands in. `string::concat` states `law(associative, identity = "")` over the `str` views of a `String` (RFC-0082 rule 2), and `analysis::loop_deps` reads a call on the view `text` lends, through the temporaries each `.concat` stores, as that law. The label is computed in the arm, so it lies in the cycles' stage, which runs apart by its laws |
