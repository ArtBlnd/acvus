# 02 Forward-backward SCC: expected structure

The worklist hands out partitions whose node sets are disjoint. Inside one partition, trimming is per node, the relabel is per node, and each BFS level expands its frontier with a test-and-set per discovered node. The coarse grain the program offers is: whole inner loops of the trim inside its per-node free stage, and per-node `Disjoint` writes of `part` and `label`.

Running two partitions apart is not claimed. The worklist and the label counter `next_part` are tokens every item reads and writes, and the labels an item allocates depend on the order items run; admitting it would need a proof that those label values never reach the output (they do not here, because the output is the least node of each component), which no rule states.

The pivot search is written without `break`: with `break` inside the worklist loop the program computes the wrong pivot at both levels (INDEX.md, "Found while building"). A search with `break` is expected as S1 free {test}, S2 Control InOrder (as `10_lru_cache`'s line 17).

| line | loop | expected | why |
|---|---|---|---|
| 16 | `for e in &from` (csr: degrees) | as `01` line 14: S2 Storage(`degree`) split by key, AnyOrder `+` | data index, commutative `+` |
| 21 | `for v in 0u64..n` (csr: offsets) | as `01` line 19: scan of `run`, Fold(push) InOrder | every partial is pushed |
| 27 | `for e in 0u64..from.len()` (csr: placement) | as `01` line 25: keyed exclusive scan of `fill`, `targets` InOrder | per-key counter read before its increment |
| 41 | `while frontier.len() > 0u64` (reach: levels) | not a `For`; runs in place | level `k + 1` is the frontier level `k` builds |
| 43 | `for u in &frontier` (reach: expand a level) | S1 free {`offs[*u]`, `offs[*u + 1]`}; S2 Storage(`seen`) split by key `w`, InOrder within a key, with the inner loop; S3 Storage(`next`) InOrder, law Fold(push) | the test-and-set `if !seen[w] { seen[w] = true; next.push(w) }` decides which iteration discovers `w`, and the push order of `next` is the program's; per key the first discoverer must win, so each key is in order. Without a key split: one InOrder stage holding `seen` and `next`. |
| 44 | `for k in offs[*u]..offs[*u + 1u64]` (reach: neighbours) | S1 free {`targets[k]`, `part[w] == p`}; S2 Storage(`seen`) test-and-set by key and Storage(`next`) Fold(push) InOrder | as line 43, one level in |
| 60 | `for v in 0u64..n` (trimmed) | S1 free {`part[v] == p`, both inner counting loops whole}; S2 Storage(`dead`) `Disjoint` | `dead[v]` at the counter (RFC-0089 rule 4, `a = 1`); the inner loops read `part`, `offs`, `targets`, none of which the loop writes, so they run whole in the free stage (coarse grain, D12's principle) |
| 63 | `for k in offs[v]..offs[v + 1u64]` (trimmed: out-edges) | S1 free {`part[targets[k]] == p`}; S2 Carried(`outs`) AnyOrder, law `+` | a conditional count (R01's reading) |
| 69 | `for k in roffs[v]..roffs[v + 1u64]` (trimmed: in-edges) | as line 63, over `ins` | |
| 86 | `while let Some(p) = work.pop()` (worklist) | not a `For`; runs in place | the body pushes the next partitions; see above for why partitions are not claimed apart |
| 88 | `for v in 0u64..n` (apply the trim) | S1 free {`dead[v]`}; S2 Storage(`label`) and Storage(`part`) `Disjoint` | both written at the counter only |
| 98 | `for v in 0u64..n` (pivot: first node of `p`) | S1 free {`part[v] == p`}; S2 Carried(`pivot`) AnyOrder, law `min` with identity `n` | `if pivot == n && part[v] == p { pivot = v }` keeps the least `v` that satisfies the test: over increasing `v` that is `min` over the satisfying ones, identity `n`. RFC-0089 rule 4 reads no such law: the branch reads the token (`pivot == n`), so it is no `last`; `first` is `last` guarded by a `\|\|` token the arm sets, and here no such token exists, the guard being a compare of the token with the constant `n`. Reading that compare as the guard needs that no hit sends `n` itself, a fact about the values sent that the rule does not read, and `min` needs besides that the hits are sent in increasing order |
| 106 | `for v in 0u64..n` (relabel around the pivot) | S1 free {`part[v] == p`, `f[v]`, `b[v]`}; S2 Storage(`label`), Storage(`part`) `Disjoint` | reads `f`, `b`, `part` at `v` and writes `label`, `part` at `v` only |
| 128 | `for v in 0u64..n` (count components) | S1 free {`label[v] == v`}; S2 Carried(`comps`) AnyOrder `+` | conditional count |
| 134 | `for v in 0u64..n` (render) | as `01` line 102: S2 Storage(`out`) InOrder, law Concat | two assignments of one storage in the iteration |
