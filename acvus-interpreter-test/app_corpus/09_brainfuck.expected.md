# 09 Brainfuck interpreter: expected structure

A state machine: the fetch-execute loop reads the program counter, the tape pointer and the tape the previous step left, so it runs in place. The preprocessing splits: filtering commands is a map with an ordered push; bracket matching is a stack and stays in order.

| line | loop | expected | why |
|---|---|---|---|
| 9 | `source.chars() \| collect` | a Stream copy with Fold(push) | not a `For` (iterator pipeline) |
| 11 | `for c in &chars` (keep commands) | S1 free {the eight compares}; S2 Storage(`code`) InOrder, law Fold(push) | a conditional push: the skip arm is the fold's identity |
| 21 | `for i in 0u64..len` (match brackets) | one cycle InOrder over Storage(`open`) and Storage(`jump`), no law | `open.pop()` answers what an earlier iteration pushed; the pair it matches decides where `jump` is written |
| 37 | `while pc < code.len() && steps < budget` (execute) | not a `For`; runs in place | `pc` moves by the jump table and the tape; the step count is data |
| 74 | `for x in &tape` (cells used) | S1 free {`*x != 0`}; S2 Storage(`used`) AnyOrder `+` | conditional count |
