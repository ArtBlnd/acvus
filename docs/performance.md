# acvus against Rust, CPython, Node, Lua 5.4 and LuaJIT

Twenty-three rows of `acvus-interpreter-test/benches` measured at master
`006619039ad7b3dc6418944408697e8ca2f87849` in six runtimes, one sweep,
2026-09-20. The machine is an AMD Ryzen 9 9950X (16 cores, 32 threads,
`/proc/cpuinfo`); every execution ran under `setarch -R taskset -c 15`, one
core. The runtimes are `rustc 1.97.1 (8bab26f4f 2026-07-14)` for both the
bench binaries and the Rust twins, CPython 3.14.7, Node v26.7.0 (the same
binary with and without `--jitless`), Lua 5.4.8 (`lua5.4`) and LuaJIT
2.1.1787165859. The protocol — `setarch -R`, one warm-up execution per
runtime discarded, `uptime` before every rep and the rep held while the
1-minute load average is 16 or above — is `benches/README.md`'s; three reps
per row, the reps alternating across the six runtimes one row at a time. The
1-minute load read 1.98 to 2.52 across all 69 `uptime` lines, so no rep
waited.

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

`benches/ports/rows.txt` is the row list with each Rust twin's value;
`benches/ports/results/2026-09-20/` holds the raw output of every rep of this
sweep, one plain-text file per row, with its `uptime` lines.

## Medians, microseconds

`accum` and `shapes` at `n = 1 000 000`, `programs` at `@1000000`.

| row | acvus | Rust | CPython | Node | Node `--jitless` | Lua 5.4 | LuaJIT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `accum` `int while` | 2772 | 184.8 | 27150 | 2899 | 6364 | 3304 | 523.0 |
| `accum` `float while` | 2738 | 445.1 | 34632 | 2893 | 6153 | 4509 | 628.0 |
| `accum` `range \| sum` | 1659 | 183.2 | 7557 | 11240 | 35890 | 2054 | 492.0 |
| `accum` `map id \| sum` | 3482 | 185.3 | 22258 | 19253 | 47306 | 7765 | 497.0 |
| `accum` `map add \| sum` | 6021 | 370.2 | 30159 | 19237 | 47525 | 9939 | 504.0 |
| `accum` `branch while` | 3180 | 265.8 | 39293 | 2291 | 14294 | 12524 | 1370 |
| `accum` `call while` | 1845 | 183.3 | 22423 | 185.6 | 10999 | 8838 | 390.0 |
| `accum` `collatz while` | 6356 | 370.8 | 50274 | 2920 | 10264 | 8359 | 1351 |
| `accum` `grade while` | 10674 | 536.8 | 39114 | 900.0 | 9748 | 7967 | 1696 |
| `accum` `while let vec` | 8112 | 277.2 | 41392 | 10516 | 30458 | 15233 | 2050 |
| `accum` `for range` | 936.9 | 186.1 | 20788 | 2889 | 7008 | 2057 | 491.0 |
| `accum` `for slice add` | 4103 | 447.4 | 39324 | 11658 | 33581 | 8681 | 2147 |
| `shapes` `field read` | 2767 | 368.7 | 31936 | 370.9 | 9690 | 7197 | 779.0 |
| `shapes` `field write` | 2792 | 188.1 | 29281 | 4648 | 10861 | 6173 | 538.0 |
| `shapes` `construct` | 2769 | 211.0 | 76970 | 587.9 | 16319 | 54304 | 481.0 |
| `shapes` `enum match` | 6121 | 370.2 | 60874 | 1964 | 22893 | 70010 | 1329 |
| `shapes` `option match` | 4374 | 247.7 | 41449 | 1341 | 15008 | 14779 | 1385 |
| `shapes` `vec of objects` | 4191 | 377.0 | 26646 | 954.3 | 9104 | 6224 | 1172 |
| `programs` `bf table` | 7913 | 712.2 | 47948 | 5045 | 24019 | 15422 | 2630 |
| `programs` `bf scan` | 14381 | 816.5 | 76317 | 4735 | 40731 | 25557 | 3082 |
| `programs` `bf call` | 7783 | 1168 | 52638 | 5227 | 27794 | 18644 | 2633 |
| `mandelbrot` `200x100x200` | 8392 | 1557 | 72526 | 1566 | 30968 | 13944 | 1607 |
| `attention` `256x128` | 434.1 | 130.7 | 1803 | 77.3 | 1391 | 574.0 | 47.0 |

## The same rows divided by Rust

| row | acvus | Rust | CPython | Node | Node `--jitless` | Lua 5.4 | LuaJIT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `accum` `int while` | 15.0 | 1.0 | 146.9 | 15.7 | 34.4 | 17.9 | 2.8 |
| `accum` `float while` | 6.2 | 1.0 | 77.8 | 6.5 | 13.8 | 10.1 | 1.4 |
| `accum` `range \| sum` | 9.1 | 1.0 | 41.3 | 61.4 | 195.9 | 11.2 | 2.7 |
| `accum` `map id \| sum` | 18.8 | 1.0 | 120.1 | 103.9 | 255.3 | 41.9 | 2.7 |
| `accum` `map add \| sum` | 16.3 | 1.0 | 81.5 | 52.0 | 128.4 | 26.8 | 1.4 |
| `accum` `branch while` | 12.0 | 1.0 | 147.8 | 8.6 | 53.8 | 47.1 | 5.2 |
| `accum` `call while` | 10.1 | 1.0 | 122.3 | 1.0 | 60.0 | 48.2 | 2.1 |
| `accum` `collatz while` | 17.1 | 1.0 | 135.6 | 7.9 | 27.7 | 22.5 | 3.6 |
| `accum` `grade while` | 19.9 | 1.0 | 72.9 | 1.7 | 18.2 | 14.8 | 3.2 |
| `accum` `while let vec` | 29.3 | 1.0 | 149.3 | 37.9 | 109.9 | 55.0 | 7.4 |
| `accum` `for range` | 5.0 | 1.0 | 111.7 | 15.5 | 37.7 | 11.1 | 2.6 |
| `accum` `for slice add` | 9.2 | 1.0 | 87.9 | 26.1 | 75.1 | 19.4 | 4.8 |
| `shapes` `field read` | 7.5 | 1.0 | 86.6 | 1.0 | 26.3 | 19.5 | 2.1 |
| `shapes` `field write` | 14.8 | 1.0 | 155.6 | 24.7 | 57.7 | 32.8 | 2.9 |
| `shapes` `construct` | 13.1 | 1.0 | 364.8 | 2.8 | 77.4 | 257.4 | 2.3 |
| `shapes` `enum match` | 16.5 | 1.0 | 164.4 | 5.3 | 61.8 | 189.1 | 3.6 |
| `shapes` `option match` | 17.7 | 1.0 | 167.3 | 5.4 | 60.6 | 59.7 | 5.6 |
| `shapes` `vec of objects` | 11.1 | 1.0 | 70.7 | 2.5 | 24.2 | 16.5 | 3.1 |
| `programs` `bf table` | 11.1 | 1.0 | 67.3 | 7.1 | 33.7 | 21.7 | 3.7 |
| `programs` `bf scan` | 17.6 | 1.0 | 93.5 | 5.8 | 49.9 | 31.3 | 3.8 |
| `programs` `bf call` | 6.7 | 1.0 | 45.0 | 4.5 | 23.8 | 16.0 | 2.3 |
| `mandelbrot` `200x100x200` | 5.4 | 1.0 | 46.6 | 1.0 | 19.9 | 9.0 | 1.0 |
| `attention` `256x128` | 3.3 | 1.0 | 13.8 | 0.6 | 10.6 | 4.4 | 0.4 |

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
  why CPython's `range | sum` reads 41.3 against Rust where its
  `map id | sum`, one interpreted closure per element, reads 120.1.
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
`attention`; LuaJIT reads 1.0 on `mandelbrot`, 0.4 on `attention` and 1.4 on
`float while` and `map add | sum`.

Whether that is an elided loop was checked by scaling `n` at fixed
`setarch -R taskset -c 15`. Node's `field read` is linear: 733.0 µs at
2 000 000, 1465.8 µs at 4 000 000 and 2932.3 µs at 8 000 000, which is
0.366 ns per iteration throughout; `construct` (733.3 → 2914.8 µs from
1 000 000 to 4 000 000) and `call while` (184.2 → 732.5 µs) are linear too,
and so are LuaJIT's `field read` (969 → 3877 µs) and `construct`
(541 → 2009 µs). No loop is elided, and these cells stand as measured.

Node's `field read` at `n = 1 000 000` is bimodal across executions — 400.4,
458.1, 460.7, 971.8 and 1473.7 µs over five — which is the tiering of a loop
short enough to finish before the top tier in some executions. The three reps
in the table all drew the low mode.

### Rows that drew a known mode

`shapes` `construct`'s acvus reps read 2769.1, 2417.2 and 2770.3 µs: this is
one of the three rows `benches/README.md` records as bimodal, with modes near
2404 and 2760 µs. `shapes` `field write`'s Rust reps read 186.9, 188.1 and
309.3 µs. `accum` `call while`'s Node reps read 184.4, 185.6 and
1069.8 µs. Every rep of every row is in
`benches/ports/results/2026-09-20/`.

### The pin and the fixed load base

`taskset -c 15` reached all five runtimes: `Cpus_allowed_list: 15` in
`/proc/self/status` under CPython, Node, Node `--jitless`, `lua5.4` and
`luajit`. Under `setarch -R`, each runtime's own ELF load base repeated across
three executions — `/usr/bin/node` at `555555400000-555555559000` with and
without `--jitless`, and `python3`, `lua5.4` and `luajit` at `555555554000`.
What `setarch -R` does not fix is Node's *first* mapping, which is V8's own
region and which V8 places itself: it read `ca95b80000` and `a2845c0000` in
two executions under `-R`. Node is the one runtime here whose address space is
not fully pinned by the kernel setting.
