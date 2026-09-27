# compilation

Every transformation keeps the program's sequential meaning: the order of its effects, whether the run ends, and whether it traps. Within that, a transformation goes as far as the declarations allow. Which trap a run raises, and which effects precede it, are not part of the meaning, so a transformation is free to change them.

Compile cost is polynomial in the size of the program. A type the program builds counts toward that size, so types are stored hash-consed: a type written in k steps costs O(k), not O(2^k). Every pass states its cost and every fixpoint its bound.

A pass keeps types, the identity of an extern-defined type included. A pass that rebuilds a value of such a type as a structural one erases what a backend needs, and it is a defect.
