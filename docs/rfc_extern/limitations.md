# limitations

What an extern fn cannot do, and what it cannot rely on.

## 0. No value from a Rust value

An extern fn cannot make a runtime value from a Rust value. No such path exists anywhere (runtime contract §1).

## 1. Types

- No declaration carries an array type, or an array length. A sequence enters as a slice and leaves as a `Vec`. Registration refuses a declaration with an array type, whoever built it.
- An extern-defined type is opaque to the program: it equals only itself, is never widened, and crosses exactly as declared, containers included. A program's value meets it only through a declared conversion.
- An extern enum does not grow. A pattern naming a variant it lacks is refused.
- An extern's type parameter is parametric: the extern cannot inspect the type that passes through it.

## 2. Calls

- A callback is callable any number of times. An extern cannot ask for a callback that is called once.
- A callback's result borrows only what was lent to it in place, and no longer than that loan.
- Every `&mut` an awaited callback receives crosses by move. Writable storage lent to an awaited callback hands over its access, which comes back only with the call's result.
- An extern that wants threads brings its own pool and joins it before it returns. The contract spawns nothing.

## 3. Keeping

A handler keeps nothing it was lent past its call. What it wants to keep, it takes by value.

## 4. Trust

Every declared fact is trusted and unchecked in a release build. A false one is the author's undefined behaviour (compiler: extern fn).

## 5. unsafe

An extern author writes no `unsafe`. The only facts an author signs are the structural facts about a type the derive reads (safety §2), and a hand-written implementation for a foreign type signs the same facts.
