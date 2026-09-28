# effects

Treating any effect as opaque is always sound. A declared effect only relaxes what an opaque effect keeps; it never adds an obligation.

## 1. Effect of a call

A call's effect is its callee's declared effect. A pure call has no effect to order. Calls declared commutative take effect in either order.

## 2. Order

Effects take place in the order of sequential evaluation, except where a declaration relaxes it. An effect keeps its order against every point at which the run may not end.

## 3. anyorder, merge instruction, and order token

Every opaque effect is already ordered. Order is not a relation a program adds by annotating calls; it is there from the start, and `anyorder` cuts it.

Order is a value. A body that issues effects holds one order token at a time. An effectful call consumes the current token and yields the next, so every effectful call is chained to the one before it, and the thread of tokens is the sequential order. A pure call takes no token.

`anyorder { … }` cuts that chain inside the block. It takes the current token once, at its entry, and every effectful call inside consumes that same entry token, so no call inside is chained to another. The block yields a merge of every token its calls yielded, and the chain resumes after the block from that merge.

A merge follows every token it joins. It is associative, commutative and idempotent: it orders what comes after it behind all of its inputs, taken as a set, and it orders its inputs against nothing. A merge of one token is that token.

The tokens are the only record of order. Nothing else states which effect comes before which, so a transformation keeps the order of effects exactly by keeping every token defined before each of its uses, as any value is. A transformation that satisfies dominance over tokens and merges needs no other rule for order.

Because `anyorder` cuts by a boundary, it is well defined over any span. Whatever enters the boundary is cut, including calls a transformation later splits, duplicates or moves within it: each still consumes the entry token. An attribute on a call would not survive this. Once a transformation splits the call into several, which of them carry the attribute is not defined.

`anyorder` is lexical. A closure written inside the block belongs to it: its body consumes its own entry token in the same way, wherever it is called from. An `anyorder` inside an open one changes nothing.

The calls inside may therefore run one after another in any order, or at once. The program declares that every such order is correct; a wrong `anyorder` is the program's undefined behaviour.

## 4. Outcome of a run

A run's outcome is the effects it issued, followed by a value, a trap, or no end. Whether it traps and whether it ends are part of the outcome. Which trap, and which effects precede it, are not.

## 5. Panic

A panic is death. Nothing runs after it, nothing is cleaned up, and nothing is owed.

## 6. Split

A call is its start and its taking. The start stands where every argument and its order token are ready; the taking is where its effect and its result enter, and the call means what it means there. Between the two the call holds its arguments' loans (ownership §5). A start is taken exactly once, on every path from it. Only the compiler splits a call, and only where its declaration states that it may be split. The declaration also states a cost, which is the backend's: the compiler carries it and never reads it.
