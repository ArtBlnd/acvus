# soundness

## 1. The boundary

`unsafe` lives only in the boundary modules. Outside them there is no `unsafe` block, `unsafe fn`, `unsafe trait` or `unsafe impl`, so soundness is read from the boundary alone. A trait implementation that is genuinely the user's choice is the one exception, and each is named with its reason.

## 2. One fact, one place

Each fact `unsafe` relies on is established at exactly one place in the boundary, and everything else consumes it through a type. Changing the assumption there changes, or breaks at compile time, every dependent. No copy of an assumption lives in a comment elsewhere.

## 3. The checker is right

The interpreter's unchecked typed reads trust the call-site shape built from the checker's decision (extern: runtime contract §3). Debug assertions cross-check it, as aids to debugging, not as the guarantee.

## 4. The aliasing model

The interpreter adopts Tree Borrows as its aliasing model: soundness is judged under Tree Borrows, and Miri under Tree Borrows is the gate. A writable reference is made only from a `&mut`, and a pointer into an owned allocation is taken after the allocation's last move. The gate reads a word back with the tag of the pointer it was made from, which is stricter than the exposed provenance a native build uses, so a clean run under the gate means the native program is defined. What only Stacked Borrows reports is recorded as a model difference, not fixed.

## 5. Release once

Releasing a value is a permission, not a convention. The permission is not `Copy`, it is minted only where a fresh allocation is made, and it costs nothing at run time. A value is therefore released at most once, by a type fact.
