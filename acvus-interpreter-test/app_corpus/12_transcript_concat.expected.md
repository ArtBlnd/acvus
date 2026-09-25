# 12 Transcript by `concat`: expected structure

The owner's chat-app loop: `text = text.concat(…)` over the history. It is the same monoid as `text = text + …` (R08), spelled through the extern `string::concat`, which declares no law today.

| line | loop | expected | why |
|---|---|---|---|
| 16 | `for m in &history` (build the transcript) | S1 free {`m.role != "system"`, the label}; S2 Storage(`text`) InOrder, law Concat through `string::concat`; S3 Storage(`turns`) AnyOrder `+` | `concat(a, b)` is associative with identity `""` and not commutative: chunks concatenate their turns and join in order. The skip arm leaves `text` as it was, so the identity stands in. Reached once `string::concat` declares `law(associative, identity = "")`. |
