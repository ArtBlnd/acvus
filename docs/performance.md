# acvus against Rust, CPython, Node, Lua 5.4 and LuaJIT

Twenty-three rows of `acvus-interpreter-test/benches` measured at the working
tree after master `40c0429a2ba90ef6821281b52bfe75bd43c229bd` in six runtimes,
one sweep, 2026-09-22. The tree carried uncommitted changes to the
interpreter's operation dispatch when the bench binaries were built, so the
acvus and Rust columns are that tree and not the commit alone. The machine is
an AMD Ryzen 9 9950X (16 cores, 32 threads, `/proc/cpuinfo`); every execution
ran under `setarch -R taskset -c 15`, one core. The runtimes are
`rustc 1.97.1 (8bab26f4f 2026-07-14)` for both the bench binaries and the Rust
twins, CPython 3.14.7, Node v26.7.0 (the same binary with and without
`--jitless`), Lua 5.4.8 (`lua5.4`) and LuaJIT 2.1.1787165859. The protocol —
`setarch -R`, one warm-up execution per runtime discarded, `uptime` before
every rep and the rep held while the 1-minute load average is 16 or above — is
`acvus-interpreter-test/benches/README.md`'s; three reps per row, the reps
alternating across the six runtimes one row at a time. The 1-minute load read
2.35 to 4.82 across all 69 `uptime` lines, so no rep waited.

The **acvus** and **Rust** columns are one execution of the bench binary: its
`execute/us` and its `rust/us`, drawn in the same rep as the four ports beside
them. The Rust twin is the same computation written in Rust inside the bench
crate, and it is what the bench asserts the acvus script's value against. The
other four columns are the ports under `benches/ports/`, one file per bench per
language, the row selected by argument. Each port prints its result value, and
the sweep fails a row whose port prints anything but the Rust twin's value; no
row failed.

To reproduce:

```
CARGO_TARGET_DIR=<empty dir> \
  cargo build --profile bench --benches -p acvus-interpreter-test -p acvus-cli
BENCH_BINS=<that dir>/release/deps CORE=15 REPS=3 \
  benches/ports/sweep.sh <output dir>
```

`sweep.sh` reaches each bench binary by its plain name, so `BENCH_BINS` must
offer `accum`, `shapes`, `programs`, `mandelbrot` and `attention` without
cargo's hash suffix.

`benches/ports/rows.txt` is the row list with each Rust twin's value;
`benches/ports/results/2026-09-22/` holds the raw output of every rep of this
sweep, one plain-text file per row, with its `uptime` lines.
`benches/ports/results/2026-09-20/` holds the previous sweep, at master
`006619039ad7b3dc6418944408697e8ca2f87849`.

## Medians, microseconds

`accum` and `shapes` at `n = 1 000 000`, `programs` at `@1000000`.

| row | acvus | Rust | CPython | Node | Node `--jitless` | Lua 5.4 | LuaJIT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `accum` `int while` | 2953 | 245.6 | 27393 | 2919 | 7077 | 3312 | 517.0 |
| `accum` `float while` | 2790 | 439.4 | 32856 | 2910 | 6468 | 5434 | 631.0 |
| `accum` `range \| sum` | 1113 | 184.8 | 7536 | 11195 | 32754 | 2059 | 490.0 |
| `accum` `map id \| sum` | 2759 | 185.9 | 22708 | 19020 | 43225 | 7775 | 492.0 |
| `accum` `map add \| sum` | 5090 | 372.7 | 30422 | 18771 | 43849 | 9988 | 505.0 |
| `accum` `branch while` | 3168 | 266.9 | 41350 | 2332 | 14614 | 12586 | 1391 |
| `accum` `call while` | 2045 | 186.6 | 22597 | 185.9 | 10167 | 8866 | 390.0 |
| `accum` `collatz while` | 6439 | 321.0 | 50764 | 2944 | 11875 | 8390 | 1352 |
| `accum` `grade while` | 10736 | 545.1 | 39605 | 892.7 | 12169 | 7980 | 1691 |
| `accum` `while let vec` | 4316 | 276.1 | 41494 | 10897 | 30808 | 15354 | 1779 |
| `accum` `for range` | 1112 | 184.7 | 21037 | 2908 | 6278 | 2066 | 498.0 |
| `accum` `for slice add` | 4606 | 447.8 | 39378 | 11486 | 34251 | 8705 | 2120 |
| `shapes` `field read` | 2958 | 369.0 | 32033 | 371.1 | 9534 | 7234 | 779.0 |
| `shapes` `field write` | 2959 | 184.3 | 28788 | 4645 | 10485 | 6161 | 522.0 |
| `shapes` `construct` | 2960 | 211.1 | 76736 | 591.7 | 16350 | 54230 | 482.0 |
| `shapes` `enum match` | 6071 | 371.4 | 60668 | 1960 | 22892 | 69735 | 1352 |
| `shapes` `option match` | 4474 | 247.8 | 42735 | 1346 | 14247 | 14826 | 1383 |
| `shapes` `vec of objects` | 4544 | 272.2 | 25062 | 768.3 | 9793 | 6238 | 1163 |
| `programs` `bf table` | 7905 | 748.1 | 49465 | 5031 | 22921 | 15456 | 2632 |
| `programs` `bf scan` | 14862 | 1075 | 79094 | 4546 | 41419 | 25387 | 3068 |
| `programs` `bf call` | 7873 | 1278 | 53812 | 5205 | 24882 | 18637 | 2635 |
| `mandelbrot` `200x100x200` | 8829 | 1556 | 72371 | 1567 | 29553 | 14019 | 1607 |
| `attention` `256x128` | 444.5 | 136.5 | 1902 | 80.4 | 1459 | 570.0 | 45.0 |

## The same rows divided by Rust

| row | acvus | Rust | CPython | Node | Node `--jitless` | Lua 5.4 | LuaJIT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `accum` `int while` | 12.0 | 1.0 | 111.6 | 11.9 | 28.8 | 13.5 | 2.1 |
| `accum` `float while` | 6.3 | 1.0 | 74.8 | 6.6 | 14.7 | 12.4 | 1.4 |
| `accum` `range \| sum` | 6.0 | 1.0 | 40.8 | 60.6 | 177.3 | 11.1 | 2.7 |
| `accum` `map id \| sum` | 14.8 | 1.0 | 122.2 | 102.3 | 232.5 | 41.8 | 2.6 |
| `accum` `map add \| sum` | 13.7 | 1.0 | 81.6 | 50.4 | 117.7 | 26.8 | 1.4 |
| `accum` `branch while` | 11.9 | 1.0 | 154.9 | 8.7 | 54.8 | 47.2 | 5.2 |
| `accum` `call while` | 11.0 | 1.0 | 121.1 | 1.0 | 54.5 | 47.5 | 2.1 |
| `accum` `collatz while` | 20.1 | 1.0 | 158.1 | 9.2 | 37.0 | 26.1 | 4.2 |
| `accum` `grade while` | 19.7 | 1.0 | 72.7 | 1.6 | 22.3 | 14.6 | 3.1 |
| `accum` `while let vec` | 15.6 | 1.0 | 150.3 | 39.5 | 111.6 | 55.6 | 6.4 |
| `accum` `for range` | 6.0 | 1.0 | 113.9 | 15.7 | 34.0 | 11.2 | 2.7 |
| `accum` `for slice add` | 10.3 | 1.0 | 87.9 | 25.6 | 76.5 | 19.4 | 4.7 |
| `shapes` `field read` | 8.0 | 1.0 | 86.8 | 1.0 | 25.8 | 19.6 | 2.1 |
| `shapes` `field write` | 16.1 | 1.0 | 156.2 | 25.2 | 56.9 | 33.4 | 2.8 |
| `shapes` `construct` | 14.0 | 1.0 | 363.6 | 2.8 | 77.5 | 256.9 | 2.3 |
| `shapes` `enum match` | 16.3 | 1.0 | 163.3 | 5.3 | 61.6 | 187.7 | 3.6 |
| `shapes` `option match` | 18.1 | 1.0 | 172.5 | 5.4 | 57.5 | 59.8 | 5.6 |
| `shapes` `vec of objects` | 16.7 | 1.0 | 92.1 | 2.8 | 36.0 | 22.9 | 4.3 |
| `programs` `bf table` | 10.6 | 1.0 | 66.1 | 6.7 | 30.6 | 20.7 | 3.5 |
| `programs` `bf scan` | 13.8 | 1.0 | 73.6 | 4.2 | 38.5 | 23.6 | 2.9 |
| `programs` `bf call` | 6.2 | 1.0 | 42.1 | 4.1 | 19.5 | 14.6 | 2.1 |
| `mandelbrot` `200x100x200` | 5.7 | 1.0 | 46.5 | 1.0 | 19.0 | 9.0 | 1.0 |
| `attention` `256x128` | 3.3 | 1.0 | 13.9 | 0.6 | 10.7 | 4.2 | 0.3 |

## Since 2026-09-20

Eight rows' acvus cells moved beyond the ±7 % that
`acvus-interpreter-test/benches/README.md` sets as the floor of what a
difference between two builds can be said to mean. Four moved down.
`while let vec` reads 4316 µs against 8112: the loop's `next` is now a `Source`
of the `For` operation rather than a call the body makes, the form
`acvus-interpreter/src/ops/control.rs` carries as `Call<H, LARGE, WORD>`. `map id | sum` reads 2759 against 3482 and
`map add | sum` 5090 against 6021: a closure is now a code word beside its
captures in one block, RFC-0069, which `FnValue` in
`acvus-interpreter/src/value.rs` carries. `range | sum` reads 1113 against
1659, and it no longer draws the two modes the README records for it; it now
reads within one microsecond of `for range`'s 1112, which the two rows did not
do before. The mechanism behind that convergence is unattributed here.

Four moved up, all of them unattributed: `for range` 1112 against 936.9,
`for slice add` 4606 against 4103, `call while` 2045 against 1845, and
`vec of objects` 4544 against 4191. The bench binaries were built from a tree
with uncommitted interpreter changes, which is a difference from the 2026-09-20
commit that this sweep does not separate from the commits between them.

The Rust column moved too, and on four rows by more than the same ±7 %:
`int while` 245.6 µs against 184.8, `bf scan` 1075 against 816.5,
`collatz while` 321.0 against 370.8 and `vec of objects` 272.2 against 377.0.
Those are two builds of an unchanged twin, so the ratio table's cells on those
rows moved for a reason on the Rust side. `vec of objects` is the clearest: its
acvus cell moved 8 % and its ratio moved from 11.1 to 16.7.

## Notes per row

### What holds for every row

- **Integers.** The largest value any row carries is `collatz while`'s
  875 000 250 000; 2^53 is 9 007 199 254 740 992, and every intermediate is a
  partial sum of the same monotone series. A JS double and a LuaJIT double
  (LuaJIT is Lua 5.1 and has no integer subtype) therefore hold every value of
  every row exactly, and no row wraps a 64-bit integer. No row is blank in any
  language.
- **`black_box`.** The Rust twin brakes each loop with
  `std::hint::black_box`; no port has that spelling, so where a runtime reads
  at or below Rust the row below says what was checked.
- **Timers.** CPython `time.perf_counter_ns` and Node `process.hrtime.bigint`
  are monotonic wall clocks at nanosecond resolution. Lua and LuaJIT have
  `os.clock`, which is process CPU time at `CLOCKS_PER_SEC = 1 000 000`, so
  every Lua and LuaJIT cell is a whole microsecond.
- **Reps.** Each port's cell is the median of the same number of timed samples
  the bench binary takes for that row — four for `accum` and `shapes`, three
  for `programs`, four for `mandelbrot` and `attention` — after one discarded
  warm-up, and each table cell is the median of three such executions.
- **Timed region.** Each port times what its Rust twin's timed function
  times: `while let vec`, `for slice add`, `vec of objects` and the three `bf`
  rows build their containers inside the timed region, and `attention` builds
  its inputs outside it.
- **Records.** The twins' `struct Point` and `struct Row3` are named-field
  records. Python has a class with `__slots__`, JS an object literal, Lua a
  table with the same field names; Lua has one aggregate and no other spelling.
- **Tagged values.** No port language has a tagged union. The twins' `enum E`
  is a `(tag, payload)` tuple in Python, `{ t, v }` in JS and `{ t =, v = }`
  in Lua, the tag an integer standing for the discriminant.
- **Absence.** `Option` needs no encoding: `some_of` returns `None`, `null`
  and `nil` respectively.

### Rows whose shape had to change in a language

- **`range | sum`, `map id | sum`, `map add | sum`.** Lua has no lazy
  pipeline, so `range | sum` is written as the same numeric `for` that
  `for range` is — the two Lua cells are the same program — and the two `map`
  rows keep one `local function` call per element inside that `for`. JS keeps
  the pipeline lazy with a generator plus the iterator helpers
  `Iterator.prototype.map`/`reduce`, which is the nearest JS form of
  `range(0, n) | map(f) | sum`; CPython uses `sum(range(n))` and
  `sum(map(f, range(n)))`, and `sum(range(n))` runs entirely in C, which is
  why CPython's `range | sum` reads 40.8 against Rust where its
  `map id | sum`, one interpreted closure per element, reads 122.2.
- **`collatz while`.** LuaJIT rejects `//` at parse time, so both Lua ports
  halve with `/`, taken only where the numerator is even and therefore exact
  in both. Under Lua 5.4 that makes the row's accumulator a float; every
  partial sum is below 2^53 and the printed value is the twin's.
- **`while let vec` and `for slice add`.** The twin's explicit `it.next()` is
  an iterator object in Python (`next(it, sentinel)`) and in JS
  (`it.next().done`), and `ipairs` in Lua, which is Lua's stateful traversal.
  `for slice add` is an index traversal in Lua and a `for…of` in JS.
- **`branch while`, `call while`, `option match`, `bf call`, the two `map`
  rows.** The twin calls a named function per iteration or per element and so
  does every port. Rust may inline that call and no port can.
- **`bf table`, `bf scan`, `bf call`.** The twin decodes to an eight-variant
  `enum` and dispatches through `match`. JS uses a dense `switch`; Python and
  Lua use an `if`/`elif` chain in the twin's arm order. Lua tables index from
  one, so `prog`, `jumps` and `tape` are 1-based and `pc` and `ptr` start at
  one: every index shifts by one and no step the machine takes changes.
- **`attention` `vec 256x128`.** The `vec`/`deque` axis is internal to acvus;
  the acvus column is the `vec` container's `in-language.execute`, and each
  port uses its one list type. The value is `out[0]`, and every port printed
  the twin's `0.49719065560641074` bit for bit, so no libm difference in
  `sin`, `cos` or `exp` showed at this size.

### Rows where a runtime read at or below Rust

Node reads 1.0 on `call while`, `field read` and `mandelbrot` and 0.6 on
`attention`; LuaJIT reads 1.0 on `mandelbrot`, 0.3 on `attention` and 1.4 on
`float while` and `map add | sum`. It is the same set of cells the 2026-09-20
sweep drew.

Whether that is an elided loop was checked at that sweep by scaling `n` at
fixed `setarch -R taskset -c 15`, over the same port files and the same
runtime versions. Node's `field read` was linear: 733.0 µs at 2 000 000,
1465.8 µs at 4 000 000 and 2932.3 µs at 8 000 000, which is 0.366 ns per
iteration throughout; `construct` (733.3 → 2914.8 µs from 1 000 000 to
4 000 000) and `call while` (184.2 → 732.5 µs) were linear too, and so were
LuaJIT's `field read` (969 → 3877 µs) and `construct` (541 → 2009 µs). No loop
is elided, and these cells stand as measured.

Node's `field read` at `n = 1 000 000` was bimodal across executions at that
sweep — 400.4, 458.1, 460.7, 971.8 and 1473.7 µs over five — which is the
tiering of a loop short enough to finish before the top tier in some
executions. This sweep's three reps read 370.9, 372.9 and 371.1 µs, all in the
low mode and below the low mode's earlier draws.

### Rows that drew a known mode

`acvus-interpreter-test/benches/README.md` records `shapes`'s three struct rows
as alternating
between about 2404 and 2760 µs. This sweep drew neither mode: `field read` read
2961.2, 2956.2 and 2958.1 µs, `field write` 2959.7, 2959.1 and 2956.5, and
`construct` 2961.2, 2959.5 and 2958.3 — nine reps inside 5 µs, above both
recorded modes. `accum`'s `range | sum`, recorded as alternating between about
1655 and 1845 µs, read 1110.2, 1112.9 and 1112.7. Neither the struct rows'
single level nor `range | sum`'s is evidence that the bimodality is gone; it is
what these nine and three reps drew.

The two rows whose outlying rep the 2026-09-20 sweep recorded drew none here:
`shapes` `field write`'s Rust reps read 184.22, 184.29 and 186.93 µs, and
`accum` `call while`'s Node reps 185.9, 185.0 and 186.6. `shapes`
`vec of objects`'s Node reps spread instead — 846.0, 768.3 and 512.9 µs. Every
rep of every row is in `benches/ports/results/2026-09-22/`.

### The pin and the fixed load base

`taskset -c 15` reached all five runtimes: `Cpus_allowed_list: 15` in
`/proc/self/status` under CPython, Node, Node `--jitless`, `lua5.4` and
`luajit`. Under `setarch -R`, each runtime's own ELF load base repeated across
three executions — `/usr/bin/node` at `555555400000-555555559000` with and
without `--jitless`, and `python3`, `lua5.4` and `luajit` at `555555554000`.
What `setarch -R` does not fix is Node's *first* mapping, which is V8's own
region and which V8 places itself: it read `16871bc0000`, `af198240000` and
`1b75880000` in three executions under `-R`, and three further addresses under
`--jitless`. Node is the one runtime here whose address space is not fully
pinned by the kernel setting.
