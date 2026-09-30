# 07 Stencil with a double buffer: expected structure

Each sweep reads `cur` and writes `next`, two storages, so within a sweep every cell is independent: the row loop and the column loop are both `Disjoint`. The sweeps are ordered: sweep `s + 1` reads what sweep `s` wrote, and the buffers swap between them. The checksums are reductions.

| line | loop | expected | why |
|---|---|---|---|
| 11 | `for r in 0u64..h` (left edge of `cur`) | S1 free {`r·w`}; S2 Storage(`cur`) `Disjoint` | the store is at `r·w`, `a = w = 7`, a word constant bound above the loop, nonzero |
| 16 | `for r in 0u64..h` (left edge of `next`) | as line 11, into `next` | |
| 20 | `for sweep in 0u64..3u64` (sweeps) | one stage, InOrder, no law, over Storage(`cur`) and Storage(`next`) | sweep `s + 1` reads the cells sweep `s` wrote (through the swap); a linear map applied three times has no law the IR states |
| 21 | `for r in 1u64..(h - 1u64)` (rows of one sweep) | S1 free {the column loop's arithmetic}; S2 Storage(`next`) `Disjoint` | row `r` writes `next[r·w + c]` for `1 ≤ c < w − 1` only: two-level affine with the inner range inside one row; `cur` is only read |
| 22 | `for c in 1u64..(w - 1u64)` (cells of one row) | S1 free {the five reads of `cur`, the sum, the division}; S2 Storage(`next`) `Disjoint` | `next[r·w + c]`, `a = 1`, base invariant; `cur` is another storage and is not written |
| 34 | `for r in 1u64..(h - 1u64)` (checksum rows) | S2 Storage(`sum`) AnyOrder `+`, Storage(`hottest`) AnyOrder `max`, through the nested loop | the inner loop's laws on the tokens it hands back (RFC-0089 rule 4, nested reading) |
| 35 | `for c in 1u64..(w - 1u64)` (checksum cells) | S1 free {the reads of `cur`}; S2 `sum` AnyOrder `+`; S3 `hottest` AnyOrder `max` | a sum and a declared `max` |
| 41 | `for c in 0u64..w` (render the centre row) | S1 free {`to_string`}; S2 Storage(`row`) InOrder, law Concat | concatenation |
