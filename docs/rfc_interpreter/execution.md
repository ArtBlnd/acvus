# execution

## 1. Prepare and ops

Prepare lowers MIR to the interpreter's ops. The ops that compute (arithmetic, the closed loop forms, branches, script calls) stand on their own. The ops that cross into extern fns are built against the runtime contract.

## 2. Depth

Call depth is bounded by the stack budget the embedder gives the host. The bound is the implementation's goal, not a language promise: a run past it traps, and the trap is the implementation's.

## 3. Memory

What a dropped run leaves in its frames is the implementation's; a leak is safe. The contract obliges only what soundness needs: release at most once.
