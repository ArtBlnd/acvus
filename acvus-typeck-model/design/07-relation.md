# 07 Relations and what records them

Rests on the spec's l.228–238 (recording), with l.267 (a value read leaves out the
relations some nodes record). The code is `src/relation.rs`.

## Meaning

Every relation is recorded in a log with what recorded it: the type the text gives
a class, an item of the text, a use or a lend once decided, a class taking a
constructor, or a cycle of lower bounds merged. The equivalence classes are what
the log makes, so a value read is decided by replaying the relations that remain.

## Readings

1. **What a value read leaves out.** A split and a merged cycle are derived from the
   other relations (l.275, l.284), so a value read derives them again from the
   relations that remain, and replays only the others.
2. **A use's result.** A use's result type enters its result as a lower bound the
   use records: the value table gives a use as a source (l.280), and a use's result
   reaches its value by `≤` as a lower bound does (l.293).
