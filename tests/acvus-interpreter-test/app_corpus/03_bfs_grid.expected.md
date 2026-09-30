# 03 BFS on a grid: expected structure

The search itself is a FIFO queue the body feeds, so it runs in place as written; its per-cell neighbour loop has free arithmetic and an ordered test-and-set. Parsing is a nested affine map; the summary is three reductions.

| line | loop | expected | why |
|---|---|---|---|
| 21 | `for r in 0u64..h` (parse rows) | S1 free {`chars | collect`, the wall loop whole}; S2 Storage(`wall`) `Disjoint`; S3 Carried(`start`), Carried(`goal`) InOrder, law last-set | row `r` writes `wall[r·w + c]` for `c < w` only, so rows never meet (two-level affine: `w` is invariant and the inner range is `0..w`); `start`/`goal` keep the last matching cell, which is associative (a later set wins) and not commutative, so the join is in order. |
| 23 | `for c in 0u64..w` (walls of one row) | S1 free {`line[c] == '#'`, `r·w + c`}; S2 Storage(`wall`) `Disjoint` | the store is at `r·w + c`, `a = 1` in the counter, base invariant (RFC-0089 rule 4) |
| 26 | `for c in 0u64..w` (find `S` and `G` in one row) | S1 free {both tests, both cell numbers}; S2 Carried(`start`) and Carried(`goal`) InOrder, law last-set | as line 21: a guarded overwrite keeps the latest hit |
| 46 | `while let Some(cell) = queue.pop_front()` (BFS) | not a `For`; runs in place | the body pushes onto the queue it pops: no source, and the pop order is the algorithm |
| 49 | `for d in 0u64..4u64` (neighbours of one cell) | S1 free {`r + dr[d]`, `c + dc[d]`, bounds, cell number, `wall[next]`}; S2 Storage(`dist`) InOrder, no law, and Storage(`queue`) InOrder, law Fold(`append`) | `dist[next]` is tested and set at a data index, so the four iterations join in order; `push_back` states a fold law over `append`, which keeps the parts in order, so the later pop order is the program's |
| 69 | `for i in 0u64..(h * w)` (summary) | S1 free {`dist[i]`, `d >= 0`}; S2 Carried(`reachable`) AnyOrder `+`, Carried(`total`) AnyOrder `+`, Carried(`farthest`) AnyOrder `max` | conditional count and sum (R01, R12) and a declared `max` law |
