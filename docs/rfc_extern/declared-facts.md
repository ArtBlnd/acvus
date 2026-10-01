# declared facts

What the attribute states beyond the signature: the effect, the declared facts (access, action, source, relation), and flows. Their meaning is the compiler's (compiler: extern fn §2-4); the compiler trusts each absolutely.

## 1. Checks by build level

The attribute's duty is to state each fact to the compiler. Checking a fact is additional: where a fact can be evaluated at a call, it may be checked, by build level, and nothing else decides whether:

1. release: never checked;
2. checked (`release_checked`): checked where possible, opt-in;
3. debug: checked where possible.

No fact is required to be checked, and a fact left unchecked is trusted as every fact is. A failed check panics where the fact is evaluated. The switch is read where the extern crate sees it, not in the author's crate.
