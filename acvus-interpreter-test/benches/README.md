# How these benches are run

A row in any of these tables is a measurement only under the protocol below.
Run outside it and the numbers move further than most of the differences the
benches are asked about: one binary, unchanged, spreads `accum`'s `for range`
row by +46.9 % across 30 back-to-back executions.

## The command

```
setarch -R taskset -c <core> <bench>
```

`setarch -R` disables address-space randomization for the execution, which
fixes the load base the kernel gives the position-independent binary. That load
base is the variable behind the spread above: the same `accum` binary reads
916.8 to 1347.1 µs on `for range` over 30 executions with randomization on, and
916.9 to 941.9 µs (+2.7 %) over 20 executions under `setarch -R`. Nothing in
the binary changes between those two rows.

`taskset -c <core>` pins the single thread the sequential cases run on. The
cases that need a pool — `logs`'s `heavy pure` and `heavy opq`, `spawn`'s heavy
rows — put their workers where the scheduler chooses and are not pinnable; they
are measured without pinning or not measured at all.

## One sweep

- Base and tree binaries **alternate within each rep**, one case at a time.
- **One warm-up execution per binary is discarded** before the first rep. The
  first execution of a bench binary of about 21 MB pays its page faults, and
  without the warm-up that cost lands on rep 1 of whichever arm runs first.
- **Three reps at least, five for `accum` and `shapes`**, whose rows carry the
  residual bimodality named below.
- `uptime` is recorded before every rep, and a rep does not begin while the
  1-minute load average is 16 or above. `bc` is not installed on this machine,
  so the comparison is `awk`:

  ```sh
  below16() { awk -v l="$1" 'BEGIN{exit !(l < 16)}'; }
  ```

## Reading the result

**A row is a change only when the two arms' ranges are disjoint** — the base
arm's worst rep better than the tree arm's best. Two medians that differ while
the ranges overlap are two draws from one distribution, and which arm drew the
high mode is what the sweep measured.

**Between two builds, a difference under about 7 % is not certifiable by any
number of reps.** A fresh build of one commit measured 7.5 % away from another
build of the same commit on `logs`'s `sync ext` row, and a commit whose entire
diff is one RFC document moved `accum` rows by up to 6 %. A comparison that has
to resolve less than that builds both arms in one target directory with one
toolchain, and 7 % remains the floor of what the difference can be said to
mean.

## The residual bimodality

Under `setarch -R`, on a pinned free core, some rows still read as two modes
rather than one. `accum`'s `range | sum` alternates between about 1655 and 1845
µs inside a single sweep arm, and `shapes`'s struct rows — `field read`,
`field write`, `construct` — alternate between about 2404 and 2760 µs. Which
mode a rep draws is not known to follow anything the protocol controls. This is
what the five-rep count on those two benches is for, and why one rep of either
is not evidence.

## Diffing `Op::run` bodies across builds

A tool that hashes the disassembly lines between one `Op::run` symbol and the
next counts the inter-function alignment padding as part of the body, so two
builds whose bodies are instruction-identical can read as differing — 740 of
2834 identical in one such run, where dropping the padding gives 2834 of 2834.
The mnemonic set to drop is the one `asm_probe.rs` already excludes:

```rust
const PADDING: &[&str] = &["nop", "nopw", "nopl", "int3", "xchg", "cs"];
```
