"""attention 256x128, ported from acvus-interpreter-test/benches/attention.rs.

Obligation across artifacts: `inputs` and the five loops are that file's
`inputs` and `rust_attention`, in the same order, and the timed region
excludes building the inputs exactly as the bench's does. `sin`, `cos` and
`exp` are the platform libm's and are not bit-reproducible across runtimes;
the bench's own criterion for this row is an absolute difference below 1e-9.

Usage: attention.py <row> [reps]
"""

import math
import sys
import time

REPS = 4


def inputs(n, d):
    query = [math.sin(float(i)) for i in range(d)]
    keys = [[math.cos(float(t + i)) for i in range(d)] for t in range(n)]
    values = [[(t * d + i) / (n * d) for i in range(d)] for t in range(n)]
    return query, keys, values


def attention(query, keys, values):
    d = len(query)
    n = len(keys)
    scale = 1.0 / math.sqrt(float(d))

    scores = []
    for t in range(n):
        s = 0.0
        for i in range(d):
            s = s + query[i] * keys[t][i]
        scores.append(s * scale)

    m = -math.inf
    for s in scores:
        if s > m:
            m = s

    weights = []
    for s in scores:
        weights.append(math.exp(s - m))

    z = 0.0
    for w in weights:
        z = z + w

    out = []
    for j in range(d):
        acc = 0.0
        for t in range(n):
            acc = acc + weights[t] / z * values[t][j]
        out.append(acc)
    return out[0]


def main():
    row = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else REPS
    n, d = (int(part) for part in row.split("x"))
    query, keys, values = inputs(n, d)

    attention(query, keys, values)
    samples = []
    value = None
    for _ in range(reps):
        start = time.perf_counter_ns()
        value = attention(query, keys, values)
        samples.append(time.perf_counter_ns() - start)
    samples.sort()
    median = samples[len(samples) // 2]
    print(f"{row}\t{median / 1000.0:.1f}\t{value!r}")


main()
