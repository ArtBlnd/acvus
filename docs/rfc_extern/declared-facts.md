# declared facts

What the attribute states beyond the signature: the effect, laws, `ensures`, `means`, and flows. Their meaning is the compiler's (compiler: extern fn §2-4); the compiler trusts each absolutely.

## 1. Checks by build level

A declared fact is checked by build level, and nothing else decides it:

1. release: never checked;
2. checked (`release_checked`): checked, opt-in;
3. debug: always checked.

A failed check panics where the fact is evaluated. The switch is read where the extern crate sees it, not in the author's crate.
