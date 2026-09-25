# 04 One k-means iteration: expected structure

The assignment step is the classic embarrassingly parallel map: every point's distances are free and the nearest-centroid search is a whole inner loop in the outer free stage. The sums are keyed by cluster. The centroid update is `Disjoint` per cluster, and the inertia is a sum.

| line | loop | expected | why |
|---|---|---|---|
| 16 | `for p in 0u64..n` (assign) | S1 free {the centroid loop whole, reading `xs[p]`, `ys[p]`, `cx`, `cy`}; S2 Storage(`assign`) `Disjoint` | `assign[p]` at the counter; `cx`, `cy` are not written in the loop, so the inner loop is pure work of the element (coarse grain) |
| 19 | `for c in 0u64..k` (nearest centroid) | S1 free {`dx`, `dy`, `d`}; S2 Carried(`best_d`, `best`) AnyOrder, law argmin on `(d, c)` | `if d < best_d { best_d = d; best = c }` keeps the lexicographically least `(d, c)`: strict `<` over increasing `c` keeps the first of equal distances, and the least pair is associative and commutative with identity `(MAX, 0)` |
| 35 | `for p in 0u64..n` (cluster sums) | S1 free {`assign[p]`, `xs[p]`, `ys[p]`}; S2 Storage(`sx`), Storage(`sy`), Storage(`size`) split by key `a`, each slot AnyOrder `+` | data index, commutative `+` (as `01` line 14) |
| 43 | `for c in 0u64..k` (move centroids) | S1 free {`size[c] > 0`, both divisions}; S2 Storage(`cx`), Storage(`cy`) `Disjoint` | read and written at the counter only |
| 51 | `for p in 0u64..n` (inertia) | S1 free {both squared differences}; S2 Carried(`inertia`) AnyOrder `+` | `inertia + dx·dx + dy·dy` is two `+` of the token (R09) |
| 60 | `for c in 0u64..cx.len()` (render centroids) | S1 free {the two `to_string`s}; S2 Storage(`shown`) InOrder, law Concat | concatenation, associative, not commutative |
| 64 | `for c in 0u64..size.len()` (render sizes) | as `01` line 102: S2 Storage(`shown`) InOrder, law Concat | a separator under `if c > 0` and a concat after it: two assignments of one storage in the iteration |
