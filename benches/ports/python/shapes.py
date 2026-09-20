"""shapes rows at n = 1 000 000, ported from acvus-interpreter-test/benches/shapes.rs.

Obligation across artifacts: each row is the semantics of that file's `rust_*`
twin. The twin's record is a `struct` with named fields, which in Python is a
class with `__slots__`; the twin's `enum E` is a tagged value, which is a
`(tag, payload)` tuple with an integer tag standing for the discriminant.

Usage: shapes.py <row> [n] [reps]
"""

import sys
import time

REPS = 4
N = 1_000_000

A = 0
B = 1


class Point:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


def some_of(i):
    return i if i % 2 == 0 else None


def field_read(n):
    p = Point(1, 2)
    acc = 0
    i = 0
    while i < n:
        acc = acc + p.x + p.y
        i = i + 1
    return acc


def field_write(n):
    p = Point(0, 0)
    i = 0
    while i < n:
        p.x = p.x + i
        i = i + 1
    return p.x


def construct(n):
    acc = 0
    i = 0
    while i < n:
        q = Point(i, i + 1)
        acc = acc + q.x
        i = i + 1
    return acc


def enum_match(n):
    acc = 0
    i = 0
    while i < n:
        e = (A, i) if i % 2 == 0 else (B, i + 1)
        if e[0] == A:
            acc = acc + e[1]
        else:
            acc = acc + e[1]
        i = i + 1
    return acc


def option_match(n):
    acc = 0
    i = 0
    while i < n:
        v = some_of(i)
        if v is not None:
            acc = acc + v
        i = i + 1
    return acc


def vec_of_objects(n):
    v = [Point(k, k + 1) for k in range(1000)]
    m = len(v)
    acc = 0
    r = 0
    while r < n // 1000:
        i = 0
        while i < m:
            acc = acc + v[i].x
            i = i + 1
        r = r + 1
    return acc


ROWS = {
    "field read": field_read,
    "field write": field_write,
    "construct": construct,
    "enum match": enum_match,
    "option match": option_match,
    "vec of objects": vec_of_objects,
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
    print(f"{row}\t{median / 1000.0:.1f}\t{value}")


main()
