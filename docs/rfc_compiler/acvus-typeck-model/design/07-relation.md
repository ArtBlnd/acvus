# 07 Relations and what records them

Rests on the spec's l.221–231 (recording), with l.260 (a value read leaves out the
relations some nodes record). The code is `src/relation.rs`, and what the text
records, `src/record.rs`.

## Meaning

Every relation is recorded in a log with what recorded it: the type the text gives
a class, an item of the text, a use or a lend once decided, a class taking a
constructor, or a cycle of lower bounds merged. The equivalence classes are what
the log makes, so a value read is decided by replaying the relations that remain.

## Readings

1. **What a value read leaves out.** A split and a merged cycle are derived from the
   other relations (l.268, l.277), so a value read derives them again from the
   relations that remain, and replays only the others.
2. **A use's result.** A use's result type enters its result as a lower bound the
   use records: the value table gives a use as a source (l.273), and a use's result
   reaches its value by `≤` as a lower bound does (l.279).
3. **What a node recorded.** A node's relations are those it records once decided
   or before (l.245, l.321), and those of every instance it makes (l.196): an
   instance's relations name what made it, so a value read leaving a node out
   leaves out its instances' relations too (l.260). A use of a function makes its
   instance as the text is recorded, and is no node.
