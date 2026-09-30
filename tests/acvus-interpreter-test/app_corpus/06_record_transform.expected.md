# 06 Record transform: expected structure

Parsing is per line and free; filtering keeps order through push's fold law; grouping is keyed by department; the mean is an iterator pipeline; the render is ordered text.

| line | loop | expected | why |
|---|---|---|---|
| 19 | `data.lines() \| collect` | a Stream copy with Fold(push) | not a `For` (iterator pipeline) |
| 25 | `for line in &lines` (parse) | S1 free {`starts_with`, `split \| collect`, `i64::parse`, the fields, the record}; S2 Storage(`parsed`) InOrder, law Fold(push) | nothing but the push touches a token; with the fold law each chunk parses and pushes into its own partial |
| 27 | `line.split(",") \| collect` | a Stream copy with Fold(push) | not a `For` (iterator pipeline), inside line 25's free work |
| 55 | `for r in &parsed` (filter) | S1 free {`!r.ok`, `r.active`}; S2 Carried(`malformed`) AnyOrder `+`; S3 Storage(`records`) InOrder, law Fold(push) | a conditional count and a conditional push (the skip arm is the fold's identity) |
| 68 | `for r in &records` (group) | S1 free {the three key copies}; S2 Storage(`heads`), Storage(`payroll`), Storage(`top`) InOrder, each law Fold (partial maps merged in chunk order: `+`, `+`, `max` per key) | as `05` line 21: keyed combines over maps whose insertion order `keys` on line 82 reads; D8's `or_insert` fold law, not declared yet |
| 80 | `records.as_iter().map(…) \| sum` | a Stream map, free `r.salary`, AnyOrder `+` | not a `For` (iterator pipeline); `sum` over `i64` is exact |
| 82 | `keys(&g.heads) \| map(…) \| collect` | a Stream map, free `to_string`, Fold(push) | not a `For` (iterator pipeline) |
| 85 | `for d in &depts` (render) | S1 free {the three `get`s and `to_string`s}; S2 Storage(`shown`) InOrder, law Concat | the maps are only read |
