# 01 Tarjan SCC: expected structure

Negative app. Tarjan's algorithm numbers nodes in discovery order and pops components off one stack, so every iteration of its outer loop reads what the previous one left: the index counter, the stack, the call stack and `low`. Nothing in `tarjan` may run apart; the facts must name those tokens.

The compressed-row builder `csr` is ordinary preprocessing and is parallel in parts.

| line | loop | expected | why |
|---|---|---|---|
| 14 | `for e in &edges_from` (csr: out-degrees) | S1 free {read `*e`}; S2 Storage(`degree`) split by key `*e`, each slot AnyOrder by `+` | `degree[*e] = degree[*e] + 1` writes at a data index, so two iterations may meet in one slot (no rule 4 `Disjoint`); `+` commutes, so a cycle split by key (RFC-0089 Open) runs slots apart. Without that split: InOrder. |
| 19 | `for v in 0u64..n` (csr: offsets) | S1 free {`degree[v] as u64`}; S2 Carried(`run`) scan by `+`; S3 Storage(`offs`) InOrder, law Fold(push) | `offs` receives every partial of `run`: an inclusive scan (RFC-0089 Open: scan law), the push joined in order by its fold law. |
| 25 | `for e in 0u64..edges_from.len()` (csr: placement) | S1 free {`edges_from[e]`, `offs[v]`, `edges_to[e]`}; S2 Storage(`fill`) split by key `v`, a scan per key; S3 Storage(`targets`) InOrder | `fill[v]` is a per-key counter whose value before the increment is read (`at`), so each key is an exclusive scan; the store `targets[at]` lands at distinct places only by that scan's reasoning, which no rule states, so it stays InOrder. |
| 46 | `for s in 0u64..n` (tarjan) | one stage, InOrder, no law, over Carried(`next_index`), Carried(`comps`), Storage(`index`, `low`, `on_stack`, `stack`, `call_node`, `call_edge`, `comp`) | the DFS started at `s` reads `index` to skip visited nodes and numbers new ones from `next_index`, both written by earlier roots; the stack order is the algorithm. |
| 55 | `while call_node.len() > 0u64` (tarjan: DFS) | not a `For`; runs in place | the explicit call stack is pushed and popped by the body: no count, no source. |
| 82 | `while open` (tarjan: pop a component) | not a `For`; runs in place | pops `stack` until the root: its trip count is data. |
| 102 | `for v in 0u64..n` (render) | S1 free {`comp[v].to_string()`}; S2 Storage(`out`) InOrder, law Concat | `out = out + ","` under `if v > 0` and then `out = out + …` is `out ⊕ (sep ⊕ text)`, two assignments of one storage in the iteration: the branch's other arm leaves `out` as it was, so `""` stands in (RFC-0089 rule 4), and the two combines associate into one; concatenation is not commutative, so the join is in order. The facts read `Op(Concat)` for one assignment under a branch and none for two assignments with or without the branch. |
