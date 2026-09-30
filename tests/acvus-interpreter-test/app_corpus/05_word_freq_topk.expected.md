# 05 Word frequency and top-k: expected structure

Tokenizing is free per token; counting is keyed by the word; the ranking is `sort_by`, an extern that runs as one call; the top-k render is ordered text.

| line | loop | expected | why |
|---|---|---|---|
| 9 | `text.split_whitespace() \| collect` | extern `split_whitespace` walks the text; `collect` of its items: a Stream map with Fold(push) | not a `For` (iterator pipeline, D10 and the T2 Stream round) |
| 13 | `for t in &tokens` (normalize) | S1 free {`lower`, the four `trim_end_matches`, `to_string`, `is_empty`}; S2 Storage(`words`) InOrder, law Fold(push) | per-token pure work; the push keeps token order through its fold law |
| 21 | `for w in &words` (count) | S1 free {`w.to_string()`}; S2 Storage(`counts`) InOrder, law Fold (partial maps merged in chunk order, `+` per key) | `*or_insert(&mut counts, w, 0) += 1` is a keyed `+`, but the map iterates in insertion order and `keys` on line 27 reads it, so which key comes first is the order of first occurrences: partials must merge in chunk order. D8 decided that the `Equiv` instance of `or_insert` declares this fold law; it is not declared yet. Not AnyOrder. |
| 27 | `keys(&counts) \| map(…) \| collect` | a Stream map, free `to_string`, Fold(push) | not a `For` (iterator pipeline) |
| 29 | `for word in &distinct` (rows) | S1 free {`to_string`, `get`, `unwrap`, the object}; S2 Storage(`rows`) InOrder, law Fold(push) | `counts` is only read |
| 43 | `for i in 0u64..top` (top-k render) | S1 free {`rows[i]`, the texts}; S2 Storage(`shown`) InOrder, law Concat; S3 Storage(`covered`) AnyOrder `+` | a separator under `if i > 0` then concats: two assignments of one storage (as `01` line 102), and a sum |
