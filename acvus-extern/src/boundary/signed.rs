//! The facts no type can carry (runtime contract §4, safety §2).


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
    pub const fn of_parts(parts: &[LayoutFact]) -> LayoutFact {
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
/// crate for the types it defines.
///
/// # Safety
/// `Canon` is `Self` with every uniform index fixed and every lifetime
/// `'static`, and has `Self`'s size, alignment, field offsets and release;
/// `Self` reaches its type parameters only by holding them, with no interior
/// mutability over a lifetime or type parameter; `FACTS` states it.
pub unsafe trait Structural {
    type Canon: Structural<Canon = Self::Canon> + 'static;
    const FACTS: StructuralFacts;
}

/// The types the crossing names as their own canonical form, signed here.
macro_rules! own_canonical {
    ($($t:ty),+ $(,)?) => {$(
        // SAFETY: `Canon` is `Self`, so size, alignment, field offsets and
        // release are one. The type has no lifetime or type parameter, so it
        // reaches none and has no interior mutability over one, as `FACTS`
        // states with an empty generic list.
        unsafe impl Structural for $t {
            type Canon = $t;

            const FACTS: StructuralFacts = StructuralFacts {
                layout: LayoutFact::OwnCanonical,
                generics: &[],
            };
        }

    )+};
}

macro_rules! own_canonical_primitives {
    ($($t:ty => $decl:expr),+ $(,)?) => {
        own_canonical!($($t),+);
    };
}

primitives!(own_canonical_primitives);
own_canonical!(String);

/// A generic list entry per type parameter of a constructor.
macro_rules! type_param {
    ($p:ident) => {
        GenericParam::Type
    };
}

/// The standard constructors the crossing names, each its own canonical form
/// over its parts' canonical forms.
macro_rules! canonical_over_parts {
    (<$($p:ident),+> $ty:ty => $canon:ty) => {
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
        unsafe impl<$($p),+> Structural for $ty
        where
            $($p: Structural,)+
        {
            type Canon = $canon;

            const FACTS: StructuralFacts = StructuralFacts {
                layout: LayoutFact::of_parts(&[$($p::FACTS.layout),+]),
                generics: &[$(type_param!($p)),+],
            };
        }
    };
}

canonical_over_parts!(<T> Option<T> => Option<T::Canon>);
canonical_over_parts!(<T, E> Result<T, E> => Result<T::Canon, E::Canon>);
canonical_over_parts!(<T> Box<T> => Box<T::Canon>);

// SAFETY: `PhantomData<T>` is of size 0 and alignment 1 at every `T`, has no
// field and no release, and holds `T` only as a marker, through which nothing
// is reached or written. `FACTS` lists its one type parameter, and the layout
// fact is the part's, as for any constructor over a part.
unsafe impl<T: Structural> Structural for std::marker::PhantomData<T> {
    type Canon = std::marker::PhantomData<T::Canon>;

    const FACTS: StructuralFacts = StructuralFacts {
        layout: LayoutFact::of_parts(&[T::FACTS.layout]),
        generics: &[GenericParam::Type],
    };
}

/// Each tuple, its own canonical form over its parts'.
macro_rules! tuple_canonical {
    ($($n:literal ($($p:ident $v:ident $i:tt),+))+) => {$(
        canonical_over_parts!(<$($p),+> ($($p,)+) => ($($p::Canon,)+));
    )+};
}

tuples!(tuple_canonical);

// SAFETY: an effect's or an identity's stand-in is a unit struct with no
// field: `Canon` is `Self`, so size, alignment, field offsets and release are
// one, and it reaches no lifetime or type parameter. It is no type parameter
// the checker's flows read, as `FACTS` states with an empty generic list.
unsafe impl<const N: u32> Structural for crate::decl::EffectNth<N> {
    type Canon = Self;

    const FACTS: StructuralFacts = StructuralFacts { layout: LayoutFact::OwnCanonical, generics: &[] };
}

// SAFETY: as `EffectNth`'s.
unsafe impl<const N: u32> Structural for crate::decl::IdentityNth<N> {
    type Canon = Self;

    const FACTS: StructuralFacts = StructuralFacts { layout: LayoutFact::OwnCanonical, generics: &[] };
}
