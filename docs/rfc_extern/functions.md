# functions

An extern receives an acvus function lent for the call, as a closure parameter, or kept past it, as a function value, and writes either in Rust's own closure signatures.

## 1. Closure parameters

An author writes a closure parameter as Rust writes one: a `Fn`, `FnMut` or `FnOnce` bound, its arguments' lifetimes stated by elision or by a binder. Its function type and flows are those its signature writes: a result that borrows an argument borrows that argument (compiler: extern fn §4), and the closure's effect joins the call's, since the call runs it (compiler: extern fn §2). The body receives a Rust closure that calls the acvus function value through the runtime; an argument `&T` or `&mut T` crosses as a `Ref` or a `Mut`, and a result returns as an owned value, a `Ref` or a `Mut`. A `Fn` bound is no excluded trick: an acvus closure is an ordinary lambda, whose body acvus checks and whose handling of a reference its flows state. A closure so lent is the call's alone, and keeping it past the call is refused.

## 2. A function kept past the call

An author names a function type it keeps, once, where it writes it, as an alias of Rust's own closure type marked `#[fn_type]`: `type Pred<'f, T> = dyn Fn(&T) -> bool + 'f`. The function crosses as a function value, kept as `Uniform<R, Pred<'f, T>>`, and is called through a view of it as a Rust closure for a scope, which borrows the runtime for that scope alone. `'f`, the loans the function holds, is named in its marker, so a type that keeps the function declares that it holds them. The runtime contract gives no `'static` handle for it.

## 3. Asynchronous calls

Work that outlives the future of the call that started it borrows only what it owns, since Rust guarantees no destructor runs. A scope takes the callable and its arguments by value, lends them to the call, and gives them back with its result, so the call's flows are plain moves and what the work keeps shows in its declaration. The runtime owns spawning: it keeps what a scope holds and the storage the call borrows until the call's frame ends, and it outlives its tasks. A function called asynchronously crosses as a function value; a closure lent for the call cannot outlive a cancellation. A started call is taken on every path from it (compiler: effects §6), so a call is cancelled only by its run's end.

*Rejected.* A per-call wrapper that gives a closure a fresh lifetime at each call through a family of marker types: it gives every reference argument of a call one lifetime, and its higher-ranked opener fails for a generic type. A second owner for asynchronous work: what the work uses it owns.
