# 11 Strip `<Log>` segments: expected structure

The owner's chat-app loop: a `find`-driven `while let` that cuts segments out of one message. Each step searches the rest the previous step left, so it is sequential. Over many messages the whole cleaner is the free work of each message: the grain that runs apart is the whole inner loop, reached through a call of the local function `strip`.

| line | loop | expected | why |
|---|---|---|---|
| 17 | `while let Some(start) = rest.find("<Log>")` (cut segments) | not a `For`; runs in place | `rest` is what the previous step left after the closing tag; the number of steps is data |
| 35 | `for m in &messages` (clean each message) | S1 free {the call of `strip` with its loop whole}; S2 Storage(`removed`) AnyOrder `+`; S3 Storage(`shown`) InOrder, law Fold(push) | `strip` reads only its argument; control is upfront (no exit), so a local call that holds a `while` is ordinary free work (RFC-0089 rule 5 restricts only work run ahead of an exit) |
| 40 | `shown.into_iter() \| join(…)` | a Stream with Concat InOrder | not a `For` (iterator pipeline) |
