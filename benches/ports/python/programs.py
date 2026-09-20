"""programs rows at n = 1 000 000, ported from acvus-interpreter-test/benches/programs.rs.

Obligation across artifacts: `B`, `STEPS_PER_OUTER`, `TAPE`, the opcode
numbers, `program`, `bracket_table` and the three machine bodies are that
file's `rust_table`, `rust_scan` and `rust_call`, whose timed region includes
building the program, decoding it and building the bracket table. The twin
dispatches an eight-variant `enum` through `match`; Python has no jump table
for that, so the port dispatches an integer opcode through an `if`/`elif`
chain ordered as the twin's arms are.

Usage: programs.py <row> [n] [reps]
"""

import sys
import time

REPS = 3
N = 1_000_000

B = 32
STEPS_PER_OUTER = 6 * B + 6
MIN_OUTER = 1
TAPE = 30000

INC = 0
DEC = 1
LEFT = 2
RIGHT = 3
OPEN = 4
CLOSE = 5
OUT = 6
IN = 7


def outer_count(n):
    return max(n // STEPS_PER_OUTER, MIN_OUTER)


def program(n):
    code = [INC] * outer_count(n)
    code.append(OPEN)
    code.append(RIGHT)
    code.extend([INC] * B)
    code.extend([OPEN, DEC, RIGHT, INC, LEFT, CLOSE])
    code.extend([LEFT, DEC, CLOSE])
    code.extend([RIGHT, RIGHT, OUT])
    return code


def decode(code):
    return [c if c <= OUT else IN for c in code]


def bracket_table(code):
    jumps = [0] * len(code)
    stack = []
    for i, c in enumerate(code):
        if c == OPEN:
            stack.append(i)
        if c == CLOSE and stack:
            o = stack.pop()
            jumps[o] = i
            jumps[i] = o
    return jumps


def bf_table(n):
    code = program(n)
    prog = decode(code)
    jumps = bracket_table(code)
    tape = [0] * TAPE
    pc = 0
    ptr = 0
    steps = 0
    out = 0
    plen = len(prog)
    while pc < plen:
        op = prog[pc]
        if op == INC:
            tape[ptr] += 1
        elif op == DEC:
            tape[ptr] -= 1
        elif op == LEFT:
            ptr -= 1
        elif op == RIGHT:
            ptr += 1
        elif op == OPEN:
            if tape[ptr] == 0:
                pc = jumps[pc]
        elif op == CLOSE:
            if tape[ptr] != 0:
                pc = jumps[pc]
        elif op == OUT:
            out += tape[ptr]
        pc += 1
        steps += 1
    return steps + out


def bf_scan(n):
    code = program(n)
    prog = decode(code)
    tape = [0] * TAPE
    pc = 0
    ptr = 0
    steps = 0
    out = 0
    plen = len(prog)
    while pc < plen:
        op = prog[pc]
        if op == INC:
            tape[ptr] += 1
        elif op == DEC:
            tape[ptr] -= 1
        elif op == LEFT:
            ptr -= 1
        elif op == RIGHT:
            ptr += 1
        elif op == OPEN:
            if tape[ptr] == 0:
                d = 1
                while d > 0:
                    pc += 1
                    inner = prog[pc]
                    if inner == OPEN:
                        d += 1
                    elif inner == CLOSE:
                        d -= 1
        elif op == CLOSE:
            if tape[ptr] != 0:
                d = 1
                while d > 0:
                    pc -= 1
                    inner = prog[pc]
                    if inner == OPEN:
                        d -= 1
                    elif inner == CLOSE:
                        d += 1
        elif op == OUT:
            out += tape[ptr]
        pc += 1
        steps += 1
    return steps + out


def bump(x, d):
    return x + d


def bf_call(n):
    code = program(n)
    prog = decode(code)
    jumps = bracket_table(code)
    tape = [0] * TAPE
    pc = 0
    ptr = 0
    steps = 0
    out = 0
    plen = len(prog)
    while pc < plen:
        op = prog[pc]
        if op == INC:
            tape[ptr] = bump(tape[ptr], 1)
        elif op == DEC:
            tape[ptr] = bump(tape[ptr], -1)
        elif op == LEFT:
            ptr -= 1
        elif op == RIGHT:
            ptr += 1
        elif op == OPEN:
            if tape[ptr] == 0:
                pc = jumps[pc]
        elif op == CLOSE:
            if tape[ptr] != 0:
                pc = jumps[pc]
        elif op == OUT:
            out += tape[ptr]
        pc += 1
        steps += 1
    return steps + out


ROWS = {"bf table": bf_table, "bf scan": bf_scan, "bf call": bf_call}


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
