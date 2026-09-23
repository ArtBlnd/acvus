//! The Rust type a box is keyed by: its payload's canonical form (RFC-0076).

use crate::ty_arg::{Kind, Var};

/// # Safety
/// `Canon` is `Self` with the `X` of every `Erased<Rt, X>` at a uniform part
/// replaced by `Never`, so that it is `Owned<Rt>` there, and with every
/// lifetime at `'static`, so that a box key is `'static` (RFC-0079 rule 7).
/// A `Chosen` part and any part that is not an `Erased` keep their own Rust
/// type. Neither type's layout reaches such an `X` through a trait.
///
/// A box is read at a type other than the one it was erased at, a `Self`
/// named over a `Canon`'s bytes or the reverse, and that read rests on three
/// layers:
/// - Release. `Drop` cannot be implemented with bounds narrower than the
///   type's (E0367), so `Erased<Rt, X>` has one `drop` for every `X`, and so
///   does every type built over it: what a box releases never depends on `X`.
/// - Size and alignment. Each read carries an inline `const` assert that the
///   two `Layout`s agree in both, so an instantiation where they differ does
///   not compile.
/// - Field order within one size and alignment. This is argued, not checked.
///   It rests on:
///   - acvus-extern has no trait impl on `Erased<Rt, X>` whose existence or
///     items depend on `X` at a runtime that makes values; the ones that
///     read `X` for the checker hold only at `TypesOnly`;
///   - a derived payload is proved `UniformPayload`: its layout reaches a
///     uniform parameter only by holding it, and the parameter's own
///     `Canonical` covers what it holds. `unsafe(uniform_payload)` is the
///     author's assertion of the same where a field type has no impl, and a
///     visible projection through a uniform parameter is refused;
///   - `Erased` is `repr(transparent)` over the runtime value;
///   - `PhantomData<fn() -> X>` has size 0 and alignment 1, and its auto
///     traits do not follow `X`;
///   - `Never` cannot be constructed.
///
/// The argument does not rest on a written promise that two instantiations
/// of one `repr(Rust)` type share a layout: Rust makes none.
/// `std::mem::TransmuteFrom` (unstable, `transmutability`, #99571) replaces
/// the third layer with a bound once it is stable.
pub unsafe trait Canonical<K>
where
    K: Kind,
{
    /// Its own canonical form, so a type named at its canonical form keys
    /// the same box, and a predicate restated at the canonical form holds
    /// there again.
    type Canon: Var<K> + Canonical<K, Canon = Self::Canon>;
}

macro_rules! same_layout {
    ($a:ty, $b:ty) => {
        const {
            ::core::assert!(
                ::core::alloc::Layout::new::<$a>().size() == ::core::alloc::Layout::new::<$b>().size()
                    && ::core::alloc::Layout::new::<$a>().align()
                        == ::core::alloc::Layout::new::<$b>().align(),
                "a box is read only between two types of one size and alignment"
            )
        }
    };
}

pub(crate) use same_layout;
