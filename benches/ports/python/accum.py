"""accum rows at n = 1 000 000, ported from acvus-interpreter-test/benches/accum.rs.

Obligation across artifacts: each row here is the semantics of that file's
`rust_*` twin, which is what the bench's `rust/us` column times and what its
assertion holds the acvus script to. `black_box` has no Python spelling, so
what the twin is stopped from folding, CPython cannot fold anyway.

Usage: accum.py <row> [n] [reps]
"""

import sys
import time

REPS = 4
N = 1_000_000


def id_of(i):
    return i


def even_of(i):
    return i % 2 == 0


def step(x):
    return x + 1


def int_while(n):
    acc = 0
    i = 0
    while i < n:
        acc = acc + i
        i = i + 1
    return acc


def float_while(n):
    acc = 0.0
    i = 0
    while i < n:
        acc = acc + float(i)
        i = i + 1
    return acc


def range_sum(n):
    return sum(range(n))


def map_id_sum(n):
    return sum(map(lambda x: x, range(n)))


def map_add_sum(n):
    return sum(map(lambda x: x + 1, range(n)))


def branch_while(n):
    acc = 0
    i = 0
    while i < n:
        if even_of(i):
            acc = acc + i
        i = i + 1
    return acc


def call_while(n):
    i = 0
    while i < n:
        i = step(i)
    return i


def collatz_while(n):
    acc = 0
    i = 0
    while i < n:
        d = i // 2 if i % 2 == 0 else i * 3 + 1
        acc = acc + d
        i = i + 1
    return acc


def grade_while(n):
    a = 0
    b = 0
    i = 0
    while i < n:
        if i % 3 == 0:
            a = a + 1
        elif i % 3 == 1:
            b = b + 1
        else:
            a = a + 2
        i = i + 1
    return a + b


_EXHAUSTED = object()


def while_let_vec(n):
    v = list(range(n))
    it = iter(v)
    acc = 0
    while True:
        x = next(it, _EXHAUSTED)
        if x is _EXHAUSTED:
            break
        acc = acc + x
    return acc


def for_range(n):
    acc = 0
    for i in range(n):
        acc = acc + i
    return acc


def for_slice_add(n):
    v = list(range(n))
    acc = 0
    for x in v:
        acc = acc + x + 1
    return acc


ROWS = {
    "int while": int_while,
    "float while": float_while,
    "range | sum": range_sum,
    "map id | sum": map_id_sum,
    "map add | sum": map_add_sum,
    "branch while": branch_while,
    "call while": call_while,
    "collatz while": collatz_while,
    "grade while": grade_while,
    "while let vec": while_let_vec,
    "for range": for_range,
    "for slice add": for_slice_add,
}


def main():
    row = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else N
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else REPS
    body = ROWS[row]

    body(n)
    samples = []
    value = None
    for _ in range(reps):
        start = time.perf_counter_ns()
        value = body(n)
        samples.append(time.perf_counter_ns() - start)
    samples.sort()
    median = samples[len(samples) // 2]
    shown = repr(value) if isinstance(value, float) else str(value)
    print(f"{row}\t{median / 1000.0:.1f}\t{shown}")


main()
