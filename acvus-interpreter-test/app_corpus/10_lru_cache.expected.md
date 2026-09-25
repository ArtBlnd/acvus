# 10 LRU cache simulation: expected structure

Negative app. Whether a request hits depends on every earlier request: the recency list is read, reordered and evicted from by each iteration, so the trace loop runs in order. The counters of hits and misses are laws inside that order, and the lookup inside one request is a search whose test runs ahead.

| line | loop | expected | why |
|---|---|---|---|
| 7 | `split_whitespace() \| collect` | a Stream copy with Fold(push) | not a `For` (iterator pipeline) |
| 14 | `for key in &trace` (requests) | one stage, InOrder, over Storage(`cache`) with the lookup loop whole; in it Carried/Storage(`hits`) and (`misses`) AnyOrder `+`, Storage(`evicted`) InOrder Fold(push) | `cache.remove(at)`, `cache.push(k)` and `cache.remove(0)` make the next request's hit or miss; no law describes a recency list. The counters are `+` of values the cache cycle decides, and the eviction log is a push. No free work but the element. |
| 17 | `for i in 0u64..cache.len()` (lookup) | S1 free {`&cache[i] == key`}; S2 Control InOrder {exit branch} | a search with `break`: the pure test runs ahead and is discarded past the exit (RFC-0089 rule 5); `at` is handed out by the exit |
| 35 | `evicted.into_iter() \| join(…)`, `cache.into_iter() \| join(…)` | a Stream map with Concat InOrder | not a `For` (iterator pipeline) |
