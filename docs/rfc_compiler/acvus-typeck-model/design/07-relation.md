# 07 Relations and what records them

The code is `src/relation.rs`, and what the text records, `src/record.rs`.

## Meaning

Every relation is recorded in a log with what recorded it: the type the text gives
a class, an item of the text, a use or a lend once decided, a class taking a
constructor, a cycle of lower bounds merged, or a function value passing into a
function type, whose split makes its lambdas' instances. The equivalence classes are what
the log makes, so a value read is decided by replaying the relations that remain.

## Readings

1. **What a value read leaves out.** A split and a merged cycle are derived from the
   other relations, so a value read derives them again from the
   relations that remain, and replays only the others. What an instance copied of
   a split's relations is the instance's, and is replayed with the rest, a copy of
   a relation split in C staying split.
2. **A use's result.** A use's result type enters its result as a lower bound the
   use records: the value table gives a use as a source, and a use's result
   reaches its value by `≤` as a lower bound does.
3. **What a node recorded.** A node's relations are those it records once decided,
   its common part before (`Recorder::Common`, which no read leaves out), and those
   of every instance it makes: an instance's relations name what made it, so a value
   read leaving a node out leaves out its instances' relations too. A use of a
   function makes its instance as the text is recorded, and is no node.
