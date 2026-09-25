# App corpus

Ten mid-size programs of the kind people write, plus two loops from a chat app, each with a hand-written expected structure per loop, compared with what `acvus mir <file>` prints at the default (`--opt full`) pipeline. The par corpus (`../par_corpus`) holds one idea per case; this corpus holds whole programs, so a loop's facts are seen next to the loops that feed it.

Every app is `<nn>_<name>.acvus`. Its header states `app`, `desc` and `expect` (the run's stdout, `\n` between lines). `<nn>_<name>.expected.md` states, per loop by source line, which grain runs apart and why. `facts/<nn>_<name>.facts` holds what `acvus mir` prints of each `For` (its stages line and the facts under it), and `acvus-cli`'s `app_corpus` test runs every app at `--opt full` and `--opt none` against its `expect` and compares its printed facts with that file.

A row's class is `matches` (the facts state the expected grain, or the expected sequential structure), `under-claim` (the facts state less parallelism than the expected structure; the missing rule, declaration or decision is named), or `OVER-CLAIM` (the facts state more than is sound). A stage the facts print as `disjoint` or with a law runs apart (RFC-0092 rules 1 and 2), whatever else its region holds; a `While` or an iterator pipeline prints no stages and runs in place.

The facts were read at beb6edba (master), with the binary built from this tree, and read again on the branch that adds RFC-0089 rule 4's readings through chains of assignments, over `Bool`, over several tokens, lifted over `Option`, `last` and the left-biased extremum, and the invariant stride; the rows that moved say so. They were read again with `first`, the first-iteration reset, `copies` and `total_order`, and no row's facts moved; and again with a token compared with a constant no write sends as its own `first` guard, `string::concat`'s law over the views of a `String`, `deque::push_back`'s fold law and `total_order` on `f64`'s `cmp`, where 03:49 (`queue` gains `Fold`) and 12:16 moved.

## Summary

- Apps: 12; all run to their expected output at `--opt full` and `--opt none` (24 of 24 runs).
- Loops classified: 75 (64 `for`/`while` loops and 11 iterator pipelines). `matches` 47, `under-claim` 28, `OVER-CLAIM` 0.
- No over-claim was found. Every `disjoint` stage was checked against every place its storage is read or written in the loop, every `any_order` law against whether the combined value's order is observed, and every free stage against the tokens it reads (the reading is in each row).

| missing rule, declaration or decision | under-claims |
|---|---|
| iterator pipeline is not a `For` (the T2 Stream round; D10 for pull loops) | 11 |
| cycle split by key over a vec's slots (RFC-0089 Open) | 6 |
| scan law (RFC-0089 Open), alone or per key | 6 |
| affine analysis one loop deep (RFC-0066 rule 4): a row loop over an inner column range | 2 |
| `or_insert`'s fold law on an `Equiv` map (D8, decided, not declared) | 2 |
| first-hit select `if x == none && p { x = k }` read as `first` (RFC-0093 rule 7): its guarded test can trap where the program skips it, and no rule discards a trap a chunk raises behind a guard an earlier chunk set | 1 |

## Loops

`‖` separates the `For`s of one row. The quoted facts drop each operation list after the stage's token and order.

| app | line | loop | expected | `acvus mir` now | class |
|---|---|---|---|---|---|
| [01](01_tarjan_scc.acvus) | 14 | csr: out-degrees `degree[*e] += 1` | S2 `degree` split by key, AnyOrder `+` | `stages [L1, L9]`; L1 free; L9 Storage in_order no law | under-claim: key split |
| 01 | 19 | csr: offsets, `run += degree[v]; offs.push(run)` | scan of `run`, Fold(push) InOrder | `stages [L4, L10, L11]`; L4 free; L10 Carried in_order no law {+}; L11 Storage in_order Fold | under-claim: scan law |
| 01 | 25 | csr: placement `targets[offs[v] + fill[v]] = …` | keyed exclusive scan of `fill`; `targets` InOrder | `stages [L7, L12, L13, L14]`; L12 Storage(fill) in_order; L13 free; L14 Storage(targets) in_order | under-claim: key split with a scan per key |
| 01 | 46 | tarjan: roots `for s` | one InOrder stage over `next_index`, `comps`, `index`, `low`, `on_stack`, `stack`, `call_node`, `call_edge`, `comp` | `stages [L1, L25]`; L1 free {const}; L25 cycle Carried+Carried+Storage×6 in_order; L25 Storage in_order; `cost in place: W=0` | matches: the tokens named are the DFS state |
| 01 | 55 | tarjan: DFS `while call_node.len() > 0` | runs in place | no `For` | matches |
| 01 | 82 | tarjan: pop a component `while open` | runs in place | no `For` | matches |
| 01 | 102 | render `if v > 0 { out += "," }; out += comp[v]` | Storage(`out`) InOrder, law Concat | `stages [L1, L5]`; L5 Storage in_order Op(Concat) | matches: the chain of both assignments (RFC-0089 rule 4) |
| [02](02_fwbw_scc.acvus) | 16 | csr: out-degrees | as 01:14 | as 01:14 | under-claim: key split |
| 02 | 21 | csr: offsets | as 01:19 | as 01:19 | under-claim: scan law |
| 02 | 27 | csr: placement | as 01:25 | as 01:25 | under-claim: key split with a scan per key |
| 02 | 41 | reach: levels `while frontier.len() > 0` | runs in place | no `For` | matches |
| 02 | 43 | reach: expand a level `for u in &frontier` | free row bounds; `seen` test-and-set split by key, in order within a key; `next` Fold(push) InOrder | `stages [L4, L15]`; L4 free; L15 Storage(seen) in_order no law; L15 Storage(next) in_order Fold | under-claim: key split |
| 02 | 44 | reach: neighbours `for k in offs[u]..offs[u + 1]` | as line 43, one level in | `stages [L7, L14]`; L7 free; L14 Storage(seen) in_order; L14 Storage(next) in_order Fold | under-claim: key split |
| 02 | 60 | trimmed: `for v`, both counting loops inside `if part[v] == p` | whole inner loops run apart; `dead[v]` Disjoint | `stages [L1, L20]`; L1 free {index, ==}; L20 Storage disjoint; the inner loops lie in L20 under the branch | matches: the inner loops run in the disjoint stage |
| 02 | 63 | trimmed: out-edges count | Carried AnyOrder `+` | `stages [L6, L18]`; L18 Carried any_order Op(Add) | matches |
| 02 | 69 | trimmed: in-edges count | Carried AnyOrder `+` | `stages [L11, L19]`; L19 Carried any_order Op(Add) | matches |
| 02 | 86 | worklist `while let Some(p) = work.pop()` | runs in place; partitions not claimed apart (label counter and worklist are tokens) | no `For` | matches |
| 02 | 88 | apply the trim | `label`, `part` Disjoint | `stages [L4, L45]`; L45 Storage disjoint ×2 | matches |
| 02 | 98 | pivot: `if pivot == n && part[v] == p { pivot = v }` | Carried InOrder, law `first`, `pivot` its own guard unset at `n` | `stages [L9, L48]`; L48 Carried in_order no law | under-claim: the guarded test `part[v]` can trap where the program skips it (the interval domain does not put `v` below `len(part)`), and no rule discards that trap behind a guard an earlier chunk set |
| 02 | 106 | relabel around the pivot | `label`, `part` Disjoint | `stages [L19, L49]`; L49 Storage disjoint ×2 | matches |
| 02 | 128 | count components | Carried AnyOrder `+` | `stages [L36, L46]`; L46 Storage any_order Op(Add) | matches |
| 02 | 134 | render | as 01:102 | `stages [L41, L47]`; L47 Storage in_order Op(Concat) | matches |
| [03](03_bfs_grid.acvus) | 21 | parse rows `for r` | `wall` Disjoint over rows; `start`, `goal` InOrder with a last-set law | `stages [L1, L17, L18]`; L1 free; L17 Storage(wall) in_order; L18 Carried+Carried in_order Product(Last, Last) | under-claim: affine one loop deep (the last-set law is read through the nested loop) |
| 03 | 23 | walls of a row `wall[r·w + c] = …` | Disjoint | `stages [L4, L13]`; L13 Storage disjoint | matches |
| 03 | 26 | find `S`, `G` in a row | Carried InOrder, last-set law | `stages [L7, L14, L15, L16]`; L14 Carried in_order Last; L16 Carried in_order Last | matches |
| 03 | 46 | BFS `while let Some(cell) = queue.pop_front()` | runs in place | no `For` | matches |
| 03 | 49 | neighbours `for d in 0..4` | free coordinates and bounds; `dist` InOrder; `queue` InOrder Fold(`append`) | `stages [L4, L22]`; L4 free; L22 Storage(dist) in_order; L22 Storage(queue) in_order Fold | matches |
| 03 | 69 | summary | `+`, `+`, `max` AnyOrder | `stages [L1, L5, L6]`; L5 Storage any_order Op(Add) ×2; L6 Storage any_order Call(max) | matches |
| [04](04_kmeans_step.acvus) | 16 | assign `for p`, nearest-centroid loop inside | inner loop whole in the free stage; `assign[p]` Disjoint | `stages [L1, L27]`; L1 free (holds the inner loop); L27 Storage disjoint | matches |
| 04 | 19 | nearest centroid `if d < best_d { best_d = d; best = c }` | Carried pair InOrder, left-biased argmin | `stages [L4, L25]`; L4 free; L25 Carried+Carried in_order Extremum(Min, best_d, carrying best) | matches (expected corrected from AnyOrder, as par-corpus R14) |
| 04 | 35 | cluster sums `sx[a] += …` | three slots split by key, AnyOrder `+` | `stages [L9, L19, L20, L21, L22, L23]`; L19, L21, L23 Storage in_order no law | under-claim: key split |
| 04 | 43 | move centroids | `cx`, `cy` Disjoint | `stages [L12, L26]`; L26 Storage disjoint ×2 | matches |
| 04 | 51 | inertia | Carried AnyOrder `+` | `stages [L17, L24]`; L24 Carried any_order Op(Add) | matches |
| 04 | 60 | render centroids | Concat InOrder | `stages [L1, L8]`; L8 Storage in_order Op(Concat) | matches |
| 04 | 64 | render sizes (separator under `if`) | Concat InOrder | `stages [L4, L9]`; L9 Storage in_order Op(Concat) | matches |
| [05](05_word_freq_topk.acvus) | 9 | `split_whitespace() \| collect` | Stream copy, Fold(push) | no `For` | under-claim: pipeline |
| 05 | 13 | normalize tokens | free text work; Fold(push) InOrder | `stages [L1, L18]`; L1 free {lower, trim_end_matches ×4, …}; L18 Storage in_order Fold | matches |
| 05 | 21 | count `*or_insert(&mut counts, w, 0) += 1` | Storage InOrder, fold law (partial maps merged in chunk order) | `stages [L6, L16]`; L16 Storage in_order no law | under-claim: D8 `or_insert` fold law |
| 05 | 27 | `keys \| map \| collect` | Stream map, Fold(push) | no `For` | under-claim: pipeline |
| 05 | 29 | rows | free `get`, object; Fold(push) | `stages [L9, L17]`; L9 free; L17 Storage in_order Fold | matches |
| 05 | 43 | top-k render and `covered` | Concat InOrder; `+` AnyOrder | `stages [L12, L19, L20, L21]`; L19 Storage in_order Op(Concat); L21 Storage any_order Op(Add) | matches |
| [06](06_record_transform.acvus) | 19 | `lines() \| collect` | Stream copy, Fold(push) | no `For` | under-claim: pipeline |
| 06 | 25 | parse lines | parsing apart; Fold(push) InOrder | `stages [L1, L10]`; L1 free; L10 Storage in_order Fold; the parse lies in L10 under `if !starts_with` | matches: the fold law runs the stage apart |
| 06 | 27 | `split(",") \| collect` | Stream copy, Fold(push) | no `For` | under-claim: pipeline |
| 06 | 55 | filter | `+` AnyOrder; Fold(push) InOrder | `stages [L1, L12]`; L12 Storage any_order Op(Add); L12 Storage in_order Fold | matches |
| 06 | 68 | group by department (three `or_insert`s) | three maps InOrder with fold laws | `stages [L1, L3, L4, L5, L6, L7]`; L3, L5, L7 Storage in_order no law | under-claim: D8 `or_insert` fold law |
| 06 | 80 | `as_iter() \| map \| sum` | Stream map, AnyOrder `+` | no `For` | under-claim: pipeline |
| 06 | 82 | `keys \| map \| collect` | Stream map, Fold(push) | no `For` | under-claim: pipeline |
| 06 | 85 | render | Concat InOrder | `stages [L9, L11]`; L11 Storage in_order Op(Concat) | matches |
| [07](07_stencil_double_buffer.acvus) | 11 | left edge `cur[r·w] = 100` | Disjoint (`a = w = 7`) | `stages [L1, L24]`; L24 Storage disjoint | matches: the interval domain proves `w` the constant 7 |
| 07 | 16 | left edge of `next` | as line 11 | `stages [L4, L25]`; L25 Storage disjoint | matches |
| 07 | 20 | sweeps | one InOrder stage over `cur`, `next` | `stages [L7, L33]`; L33 Storage+Storage in_order; `cost in place: W=0` | matches |
| 07 | 21 | rows of a sweep | `next` Disjoint over rows | `stages [L10, L31]`; L31 Storage in_order no law | under-claim: affine one loop deep |
| 07 | 22 | cells of a row `next[r·w + c] = …` | free reads of `cur`; `next` Disjoint | `stages [L13, L26]`; L13 free; L26 Storage disjoint | matches |
| 07 | 34 | checksum rows | `+`, `max` AnyOrder through the nested loop | `stages [L16, L32]`; L32 Storage any_order Op(Add); L32 Storage any_order Call(max) | matches |
| 07 | 35 | checksum cells | `+`, `max` AnyOrder | `stages [L19, L27, L28, L29]`; L27 any_order Op(Add); L29 any_order Call(max) | matches |
| 07 | 41 | render the centre row | Concat InOrder | `stages [L22, L30]`; L30 Storage in_order Op(Concat) | matches |
| [08](08_prefix_histogram.acvus) | 6 | `split_whitespace() \| map(parse) \| collect` | Stream map, Fold(push) | no `For` | under-claim: pipeline |
| 08 | 11 | prefix sums `pre[i + 1] = pre[i] + lat[i]` | scan by `+` | `stages [L1, L11]`; L11 Storage in_order no law | under-claim: scan law |
| 08 | 20 | range totals | `answer[q]` Disjoint | `stages [L1, L3]`; L3 Storage disjoint | matches |
| 08 | 31 | histogram `hist[b] += 1` | split by key, AnyOrder `+` | `stages [L1, L3]`; L3 Storage in_order no law | under-claim: key split |
| 08 | 42 | median bucket | scan of `seen`; `median` first-hit `min` | `stages [L4, L12, L13]`; L12 Carried in_order no law; L13 Storage in_order no law | under-claim: scan law (and first-hit select) |
| [09](09_brainfuck.acvus) | 9 | `chars() \| collect` | Stream copy, Fold(push) | no `For` | under-claim: pipeline |
| 09 | 11 | keep commands | free compares; Fold(push) InOrder | `stages [L1, L34]`; L34 Storage in_order Fold | matches |
| 09 | 21 | match brackets | one InOrder cycle over `open`, `jump` | `stages [L27, L33]`; L33 Storage(open) in_order; L33 Storage(jump) in_order | matches |
| 09 | 37 | execute `while pc < … && steps < budget` | runs in place | no `For` | matches |
| 09 | 74 | cells used | `+` AnyOrder | `stages [L20, L24]`; L24 Storage any_order Op(Add) | matches |
| [10](10_lru_cache.acvus) | 7 | `split_whitespace() \| collect` | Stream copy, Fold(push) | no `For` | under-claim: pipeline |
| 10 | 14 | requests | one InOrder stage over `cache` with the lookup whole; `hits`, `misses` `+`; `evicted` Fold | `stages [L1, L15]`; L15 Storage(cache) in_order; L15 Storage any_order Op(Add) ×2; L15 Storage in_order Fold; `cost in place: W=0` | matches |
| 10 | 17 | lookup with `break` | free test; Control InOrder | `stages [L4, L14]`; L4 free {…, string_eq}; L14 Control in_order; `n ≤ …` | matches |
| 10 | 35 | two `into_iter() \| join` | Stream, Concat InOrder | no `For` | under-claim: pipeline |
| [11](11_strip_log_segments.acvus) | 17 | cut segments `while let Some(start) = rest.find("<Log>")` | runs in place | no `For` | matches |
| 11 | 35 | clean each message: `strip(m)` | the call (with its `while`) in the free stage; `+` AnyOrder; Fold(push) | `stages [L1, L3, L4, L5]`; L1 free {call indirect, …}; L3 Storage any_order Op(Add); L4 free; L5 Storage in_order Fold | matches |
| 11 | 40 | `into_iter() \| join` | Stream, Concat InOrder | no `For` | under-claim: pipeline |
| [12](12_transcript_concat.acvus) | 16 | `text = text.concat(label).concat(…)` | Concat InOrder through `string::concat`; `+` AnyOrder | `stages [L1, L8]`; L8 Storage(text) in_order Call(concat); L8 Storage any_order Op(Add) | matches: `string::concat` states its law over the `str` views of a `String` (RFC-0082 rule 2); the label lies in the stage its arm shares with both cycles |

## Found while building

These are not facts of `acvus mir`; they are what writing the apps met.

1. **A `break` out of a `for` nested in a `while` loses the exit value, at both levels.** `02`'s pivot search is written without `break` for this reason. The smallest program found prints `0` where it means `1`, at `--opt full` and `--opt none`:

   ```
   let n = 4u64;
   let count = 0;
   let k = 0;
   while k < 1 {
       k = k + 1;
       let pivot = n;
       for v in 0u64..n {
           if v == 2u64 {
               pivot = v;
               break;
           };
       }
       if pivot < n {
           count = count + 1;
       };
   }
   count
   ```

   The MIR at `--opt none` is right: the `break` edge sends `v` to the exit block and the `if` reads it. Without the `break` the program prints `1`; the same search at the top level, or nested in a `for` (`10`'s lookup), prints the right value. `acvus ops` lowers the inner loop as `For<Range<u64>, Escapes>` followed by a `Mov` in the enclosing `Loop`'s body; whether that move overwrites the escaped value is not verified.

2. **`--opt full` colours a body into more than 64 scalar registers where `--opt none` runs it.** `04`, `06` and `08` as first written failed at `--opt full` only (73, 71 and 71 registers). Each literal element of a `vec([...])` is a register live until the array is built, so a 16-element literal costs 16. The apps put each step in a local function and read long data from text, as a program of that size would; the limit stays a limit a mid-size script meets.
