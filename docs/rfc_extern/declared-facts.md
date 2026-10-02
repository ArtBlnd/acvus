# declared facts

The attribute states a declaration's effect, declared facts and flows (compiler: extern fn §2-§4), and the compiler trusts each.

## 1. Checks by build level

A fact that can be evaluated at a call may be checked there, by build level: never in release, where possible in a checked build, and where possible in debug. An unchecked fact is trusted as every fact is. A failed check panics where the fact is evaluated. The build level is the extern crate's.
