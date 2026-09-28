# compilation

## 1. Meaning

Every transformation keeps the program's sequential meaning: the order of its effects, whether the run ends, and whether it traps. Within that, a transformation goes as far as the declarations allow. Which trap a run raises, and which effects precede it, are not part of the meaning, so a transformation is free to change them.

## 2. Cost

Compile cost is polynomial in the size of the program. A type the program builds counts toward that size, so types are stored hash-consed: a type written in k steps costs O(k), not O(2^k). Every pass states its cost and every fixpoint its bound.

## 3. Types

A pass keeps types, the identity of an extern-defined type included. A pass that rebuilds a value of such a type as a structural one erases what a backend needs, and it is a defect.

## 4. Loops

A loop is its head, its body and its join. The head yields the iterations. The join is where the iterations' values meet: a place that more than one iteration updates (extern fn §3, access), and a value one iteration hands to the next. The body is the rest. No two iterations meet in the body, so its iterations run in any order, and at once.

The join keeps the order of the sequential run, except where the declarations relax it (extern fn §3, action).
- If an update is the action of an associative combine, the join combines neighbouring iterations, or neighbouring runs of them, as soon as both are ready, keeping their positions.
- If the combine is also commutative, the join combines any two, in any order.
- Where the body reads what the join holds, it reads the combination of the iterations before it.

The head decides where the iterations may be split: a stream splittable at any position anywhere, and a stream that yields step by step only as it yields (extern fn §3, source). An iteration the sequential run would not reach, one past an exit, runs ahead only where its body neither traps nor has an effect.

A transformation reshapes a loop so that its join holds as little as it can. For example, a value handed from one iteration to the next that the iteration's position decides is computed from the position in the body instead.

Across loops, work forms one graph. Its nodes are the bodies, the joins, and the starts and takings of split calls (effects §6). Its edges are values, the orders the joins keep, and the order of effects (effects §2). Any work whose inputs are ready runs, in any order, and a run of iterations splits at any point where its join can combine.

Whether to split is the backend's decision. The compiler states each loop's cost as a formula over quantities the backend supplies: the work of an iteration, the cost of a split, and the count. It never evaluates that formula (effects §6).
