"""mandelbrot 200x100x200, ported from acvus-interpreter-test/benches/mandelbrot.rs.

Obligation across artifacts: the three nested loops and the compound loop
condition are that file's `rust_mandelbrot`, which is what the bench's
`rust/us` column times.

Usage: mandelbrot.py <row> [reps]
"""

import sys
import time

REPS = 4


def mandelbrot(w, h, max_i):
    total = 0
    py = 0
    while py < h:
        px = 0
        while px < w:
            cx = -2.0 + 3.0 * px / w
            cy = -1.2 + 2.4 * py / h
            x = 0.0
            y = 0.0
            i = 0
            while i < max_i and x * x + y * y < 4.0:
                xt = x * x - y * y + cx
                y = 2.0 * x * y + cy
                x = xt
                i = i + 1
            total = total + i
            px = px + 1
        py = py + 1
    return total


def main():
    row = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else REPS
    w, h, max_i = (int(part) for part in row.split("x"))

    mandelbrot(w, h, max_i)
    samples = []
    value = None
    for _ in range(reps):
        start = time.perf_counter_ns()
        value = mandelbrot(w, h, max_i)
        samples.append(time.perf_counter_ns() - start)
    samples.sort()
    median = samples[len(samples) // 2]
    print(f"{row}\t{median / 1000.0:.1f}\t{value}")


main()
