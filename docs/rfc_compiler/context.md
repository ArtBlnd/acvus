# context

A context is a value, or an object, of a fixed type that is always filled whenever the program runs.

## 1. What "filled" means

A program may take a context's value and put a value back.

"Filled" is an invariant at the run's boundary. Within a run a context may be empty after a take. On every path on which the run ends, it is filled again. What a run that traps or does not end leaves in a context is not stated.

## 2. Invariance

A context's type is invariant by the transparency rule: its uses beyond the program are not visible, so it is never widened or joined, and it equals only itself.

## 3. Load, store, and commit

What a load, a store or a commit does beyond the program is unknown to the language, as a syscall's is. A context is close to a typed vIOMMU.

The load and the commit are the effects. An assignment to a context inside the run is an assignment to a plain value and carries no effect of its own. Only promoting the context to a plain value removes a load or a commit; nothing else does.

A value stored into a context outlives the run, so it holds no reference into the run. It may carry an identity.
