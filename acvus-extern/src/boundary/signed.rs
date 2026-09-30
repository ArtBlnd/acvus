//! The facts no type can carry (runtime contract §4, safety §2).

use crate::boundary::entry::Settled;
use crate::boundary::uniform::Uniform;
use crate::runtime::Runtime;
use crate::shape::{LentParts, LentPartsMut, Projected};


/// S1: storage a shared guard lends keeps its address, and is written only
/// under an exclusive loan, until the checker's loan ends, after the guard
/// drops as well.
///
/// # Safety
/// The implementing runtime keeps that storage so.
pub unsafe trait SignedS1 {}

/// S2: a value made as a reference to a target names that target, or its
/// written-back copy, until the checker's loan ends.
///
/// # Safety
/// The implementing runtime keeps every reference value so.
pub unsafe trait SignedS2 {}

/// S3: storage lent in place to another run, and a kept value, are released
/// only after the last work that may read them has ended or been dropped.
///
/// # Safety
/// The implementing runtime releases no such storage earlier.
pub unsafe trait SignedS3 {}

/// The layout identity a type has with a related Rust type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutFact {
    /// The type is its own canonical form.
    OwnCanonical,
    /// The type's layout is its canonical form's.
    CanonicalOf,
}

impl LayoutFact {
    /// The fact of a constructor whose canonical form is the same constructor
    /// over its parts' canonical forms: it is its own canonical form exactly
    /// when every part is.
    const fn of_parts(parts: &[LayoutFact]) -> LayoutFact {
        let mut i = 0;
        while i < parts.len() {
            if let LayoutFact::CanonicalOf = parts[i] {
                return LayoutFact::CanonicalOf;
            }
            i += 1;
        }
        LayoutFact::OwnCanonical
    }
}

/// A generic parameter the checker's flows track, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenericParam {
    Lifetime,
    Type,
}

/// A type's structural facts (safety §2): its layout identity (L), that its
/// layout and carriers reach its type parameters only by holding them (P),
/// that no part a lifetime or type parameter reaches has interior mutability
/// (M), and its generic parameters (G).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuralFacts {
    pub layout: LayoutFact,
    pub generics: &'static [GenericParam],
}

/// A type's structural facts, signed at the type by the derive or by this
/// crate for the types it defines, and how a value of it crosses.
///
/// A type crosses as one of two forms. An opaque type, which is every type
/// the derive signs and every scalar, `String` and `Vec`, is given and taken
/// whole and lent as a guard. A transparent structured type, a tuple, an
/// `Option` or a `Result`, is the language's value (crossing §2, compiler:
/// types §1-§2): it is composed from its parts' values, taken apart into its
/// owned view, and lent as its projection. The form is selected here, at the
/// type, because every crossing type reaches the crossing through this trait
/// alone: `Crosses` has one implementation for every `Structural` type.
///
/// # Safety
/// `Canon` is `Self` with every uniform index fixed and every lifetime
/// `'static`, and has `Self`'s size, alignment, field offsets and release;
/// `Self` reaches its type parameters only by holding them, with no interior
/// mutability over a lifetime or type parameter; `FACTS` states it. The
/// crossing items give, take and lend a value at `Self`, and a part at the
/// part's own type.
pub unsafe trait Structural {
    type Canon: Structural<Canon = Self::Canon> + 'static;
    const FACTS: StructuralFacts;

    /// A part of this type in a shared projection: a guard, or the part's
    /// own projection.
    #[doc(hidden)]
    type Lent<'c, R: Runtime>
    where
        Self: 'c;
    /// A part of this type in a mutable projection.
    #[doc(hidden)]
    type LentMut<'c, R: Runtime>
    where
        Self: 'c;

    /// The runtime value of `self`, at its settled type.
    #[doc(hidden)]
    fn give<R: Runtime>(self, rt: &R, at: Settled<'_>) -> R::Value
    where
        Self: Sized;
    /// The value `v` handed over, at its settled type.
    #[doc(hidden)]
    fn take<R: Runtime>(v: R::Value, rt: &R, at: Settled<'_>) -> Self
    where
        Self: Sized;
    /// The part at `place`, lent shared.
    #[doc(hidden)]
    fn lent<'c, R: Runtime>(rt: &'c R, place: &'c R::Value, at: Settled<'c>) -> Self::Lent<'c, R>
    where
        Self: 'c;
    /// The part at `place`, lent exclusively.
    #[doc(hidden)]
    fn lent_mut<'c, R: Runtime>(rt: &'c R, place: &'c mut R::Value, at: Settled<'c>) -> Self::LentMut<'c, R>
    where
        Self: 'c;
}

/// The crossing items of an opaque type, which is `Send + Sync`.
macro_rules! opaque_crossing {
    () => {
        type Lent<'c, Rt: $crate::runtime::Runtime>
            = $crate::boundary::seal::opaque::Lent<'c, Rt, Self>
        where
            Self: 'c;
        type LentMut<'c, Rt: $crate::runtime::Runtime>
            = $crate::boundary::seal::opaque::LentMut<'c, Rt, Self>
        where
            Self: 'c;

        fn give<Rt: $crate::runtime::Runtime>(self, rt: &Rt, at: $crate::boundary::entry::Settled<'_>) -> Rt::Value {
            $crate::boundary::seal::opaque::give(self, rt, at)
        }

        fn take<Rt: $crate::runtime::Runtime>(v: Rt::Value, rt: &Rt, at: $crate::boundary::entry::Settled<'_>) -> Self {
            $crate::boundary::seal::opaque::take(v, rt, at)
        }

        fn lent<'c, Rt: $crate::runtime::Runtime>(rt: &'c Rt, place: &'c Rt::Value, at: $crate::boundary::entry::Settled<'c>) -> Self::Lent<'c, Rt>
        where
            Self: 'c,
        {
            $crate::boundary::seal::opaque::lent(rt, place, at)
        }

        fn lent_mut<'c, Rt: $crate::runtime::Runtime>(rt: &'c Rt, place: &'c mut Rt::Value, at: $crate::boundary::entry::Settled<'c>) -> Self::LentMut<'c, Rt>
        where
            Self: 'c,
        {
            $crate::boundary::seal::opaque::lent_mut(rt, place, at)
        }
    };
}

pub(crate) use opaque_crossing;

/// The lending items of a transparent structured type: lent as its
/// projection. Its `give` composes it from its parts, and its `take` takes it
/// apart into its owned view; both are written at each constructor.
macro_rules! transparent_crossing {
    () => {
        type Lent<'c, R: Runtime>
            = <Self as Projected<R>>::Ref<'c>
        where
            Self: 'c;
        type LentMut<'c, R: Runtime>
            = <Self as Projected<R>>::Mut<'c>
        where
            Self: 'c;

        fn lent<'c, R: Runtime>(rt: &'c R, place: &'c R::Value, at: Settled<'c>) -> Self::Lent<'c, R>
        where
            Self: 'c,
        {
            <Self as Projected<R>>::project(LentParts::of_the_entry(rt, place, at))
        }

        fn lent_mut<'c, R: Runtime>(rt: &'c R, place: &'c mut R::Value, at: Settled<'c>) -> Self::LentMut<'c, R>
        where
            Self: 'c,
        {
            <Self as Projected<R>>::project_mut(LentPartsMut::of_the_entry(
                rt,
                place,
                <Self as Projected<R>>::AGGREGATE,
                at,
            ))
        }
    };
}

/// The types the crossing names as their own canonical form, signed here.
macro_rules! own_canonical {
    ($($t:ty),+ $(,)?) => {$(
        // SAFETY: `Canon` is `Self`, so size, alignment, field offsets and
        // release are one. The type has no lifetime or type parameter, so it
        // reaches none and has no interior mutability over one, as `FACTS`
        // states with an empty generic list. It crosses opaque, at itself.
        unsafe impl Structural for $t {
            type Canon = $t;

            const FACTS: StructuralFacts = StructuralFacts {
                layout: LayoutFact::OwnCanonical,
                generics: &[],
            };

            opaque_crossing!();
        }

    )+};
}

own_canonical!(
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    f64,
    bool,
    char,
    String,
    ()
);

/// A generic list entry per type parameter of a constructor.
macro_rules! type_param {
    ($p:ident) => {
        GenericParam::Type
    };
}

/// The standard constructors the crossing names, each its own canonical form
/// over its parts' canonical forms, crossing as `$form`.
macro_rules! canonical_over_parts {
    (<$($p:ident),+> $ty:ty => $canon:ty, $form:ident where [$($bound:tt)+] { $($items:tt)* }) => {
        // SAFETY: `Canon` is the same constructor applied to each part's
        // `Canon`. Each part has its canonical form's size, alignment, field
        // offsets and release by the part's own `Structural` signature, so
        // the constructor, whose layout and drop glue are computed from its
        // parts', has its canonical form's; Rust promises no layout shared by
        // two instantiations of a `repr(Rust)` type, so field order at equal
        // size and alignment is argued from this, and the size and alignment
        // are asserted per instantiation by `LayoutWitness::canonical`. The
        // constructor holds each type parameter as a part, projects through
        // none, and adds no interior mutability; `FACTS` lists one type
        // parameter per part and derives the layout fact from the parts'.
        // It crosses opaque at itself, or as the language's value from its
        // parts, each at its own type.
        unsafe impl<$($p),+> Structural for $ty
        where
            $($bound)+
        {
            type Canon = $canon;

            const FACTS: StructuralFacts = StructuralFacts {
                layout: LayoutFact::of_parts(&[$($p::FACTS.layout),+]),
                generics: &[$(type_param!($p)),+],
            };

            $form!();

            $($items)*
        }
    };
}

canonical_over_parts!(<T> Option<T> => Option<T::Canon>, transparent_crossing where [T: Structural] {
    fn give<R: Runtime>(self, rt: &R, at: Settled<'_>) -> R::Value {
        match self {
            None => rt.make_variant(tags::NONE, None, at),
            Some(value) => {
                let payload = value.give(rt, at);
                rt.make_variant(tags::SOME, Some(payload), at)
            }
        }
    }

    fn take<R: Runtime>(v: R::Value, rt: &R, at: Settled<'_>) -> Self {
        Uniform::<R, Self>::of_the_crossing(v).view(rt, at).map(|value| value.take(rt, at))
    }
});
canonical_over_parts!(<T, E> Result<T, E> => Result<T::Canon, E::Canon>, transparent_crossing where [T: Structural, E: Structural] {
    fn give<R: Runtime>(self, rt: &R, at: Settled<'_>) -> R::Value {
        let (tag, payload) = match self {
            Ok(value) => (tags::OK, value.give(rt, at)),
            Err(error) => (tags::ERR, error.give(rt, at)),
        };
        rt.make_variant(tag, Some(payload), at)
    }

    fn take<R: Runtime>(v: R::Value, rt: &R, at: Settled<'_>) -> Self {
        match Uniform::<R, Self>::of_the_crossing(v).view(rt, at) {
            Ok(value) => Ok(value.take(rt, at)),
            Err(error) => Err(error.take(rt, at)),
        }
    }
});
// A sequence leaves as a `Vec` (limitations §1): a runtime gives it at the
// settled array type.
canonical_over_parts!(<T> Vec<T> => Vec<T::Canon>, opaque_crossing where [T: Structural + Send + Sync, T::Canon: Send + Sync] {});

/// `Option`'s and `Result`'s tags: their variants' places in the order of
/// their names' text (compiler: types §2).
pub(crate) mod tags {
    pub(crate) const NONE: u32 = 0;
    pub(crate) const SOME: u32 = 1;
    pub(crate) const ERR: u32 = 0;
    pub(crate) const OK: u32 = 1;
}

/// Tuples up to twelve parts, the arity the standard library's own tuple
/// impls stop at.
macro_rules! tuple_canonical {
    ($(($($p:ident $v:ident),+))+) => {$(
        canonical_over_parts!(<$($p),+> ($($p,)+) => ($($p::Canon,)+), transparent_crossing where [$($p: Structural),+] {
            fn give<R: Runtime>(self, rt: &R, at: Settled<'_>) -> R::Value {
                let ($($v,)+) = self;
                let parts = [$($v.give(rt, at)),+];
                rt.make_tuple(parts.into_iter(), at)
            }

            fn take<R: Runtime>(v: R::Value, rt: &R, at: Settled<'_>) -> Self {
                let ($($v,)+) = Uniform::<R, Self>::of_the_crossing(v).view(rt, at);
                ($($v.take(rt, at),)+)
            }
        });
    )+};
}

tuple_canonical! {
    (A a)
    (A a, B b)
    (A a, B b, C c)
    (A a, B b, C c, D d)
    (A a, B b, C c, D d, E e)
    (A a, B b, C c, D d, E e, F f)
    (A a, B b, C c, D d, E e, F f, G g)
    (A a, B b, C c, D d, E e, F f, G g, H h)
    (A a, B b, C c, D d, E e, F f, G g, H h, I i)
    (A a, B b, C c, D d, E e, F f, G g, H h, I i, J j)
    (A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k)
    (A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k, L l)
}
