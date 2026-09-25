# 08 Prefix sums and a histogram: expected structure

The prefix sum is the canonical scan, written into storage at the next slot; the range queries are a free map over it; the histogram is a keyed `+`; the median walk is a scan read by a first-hit search.

| line | loop | expected | why |
|---|---|---|---|
| 6 | `split_whitespace() \| map(parse) \| collect` | a Stream map, free `i64::parse`, Fold(push) | not a `For` (iterator pipeline) |
| 11 | `for i in 0u64..n` (prefix sums) | S1 free {`lat[i]`}; S2 Storage(`pre`) scan by `+`: chunks reduce, each chunk rescans from its offset | `pre[i + 1] = pre[i] + lat[i]` reads the slot the previous iteration wrote, so it is not `Disjoint` (the bases differ by `a`); the recurrence is `+`, associative, and `pre[i]` is the previous partial, so a scan law (RFC-0093 rule 8) runs it apart |
| 20 | `for q in 0u64..lo.len()` (range totals) | S1 free {`pre[hi[q]] − pre[lo[q]]`}; S2 Storage(`answer`) `Disjoint` | `pre` is only read; the store is at the counter |
| 31 | `for x in &lat` (histogram) | S1 free {`min(x / 25, buckets − 1)`}; S2 Storage(`hist`) split by key `b`, each slot AnyOrder `+` | data index, commutative `+` |
| 42 | `for b in 0u64..buckets` (median bucket) | S1 free {`hist[b]`}; S2 Carried(`seen`) scan by `+`; S3 Storage(`median`) AnyOrder, law `min` with identity `buckets` | `median` reads each partial of `seen`; `if median == buckets && … { median = b }` keeps the least `b` that satisfies the test |
