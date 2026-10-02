//! A Rust type's acvus type, as a declaration states it: the Rust type as
//! written, `Uniform<R, X>` standing for `X` held uniformly, and a whole
//! concrete extern-defined type written natively marked `#` (rfc_compiler
//! types §8).

use acvus_mir::decl::{TyVarBound, UserTypeDecl};
use acvus_mir::ty::{EffectTerm, IdentityTerm, IntTy, Mark, MarkTerm, Mutability, Poly, TyId, TyKind, Types};
use acvus_utils::{Astr, Interner, QualifiedRef};

use crate::marker::{EffectVar, IdentityVar};
use crate::runtime::Runtime;
use crate::uniform::Uniform;

/// A declared type, in the declarations' one table.
pub type Ty = TyId<Poly>;

/// An extern-defined type as `#[derive(Opaque)]` states it.
pub struct OpaqueDecl<'a> {
    pub name: &'a str,
    pub type_args: &'a [Ty],
    pub lifetimes: usize,
    pub effect: Option<EffectTerm<Poly>>,
    pub identity: Option<IdentityTerm<Poly>>,
    /// Whether a value of it holds a uniform value anywhere.
    pub uniform: bool,
}

/// What declaring types builds: the type table, the names, and every
/// extern-defined type met on the way, each once.
pub struct Declaring<'d> {
    types: &'d mut Types<Poly>,
    names: &'d Interner,
    user_types: &'d mut Vec<UserTypeDecl>,
}

impl<'d> Declaring<'d> {
    pub fn new(types: &'d mut Types<Poly>, names: &'d Interner, user_types: &'d mut Vec<UserTypeDecl>) -> Self {
        Declaring { types, names, user_types }
    }

    pub fn intern(&mut self, kind: TyKind<Poly>) -> TyId<Poly> {
        self.types.intern(kind)
    }

    pub fn name(&self, name: &str) -> Astr {
        self.names.intern(name)
    }

    pub fn root(&self, name: &str) -> QualifiedRef {
        QualifiedRef::root(self.name(name))
    }

    /// Records an extern-defined type the first time it is met.
    fn user_type(&mut self, declared: UserTypeDecl) {
        if self.user_types.iter().all(|known| known.id != declared.id) {
            self.user_types.push(declared);
        }
    }

    /// An object of `fields`, given in their names' text order.
    pub fn object(&mut self, fields: &[(&str, Ty)]) -> Ty {
        let fields = fields.iter().map(|(name, ty)| (self.name(name), *ty)).collect();
        self.intern(TyKind::Object { fields, label: () })
    }

    /// The enum `name` of `variants`, given in their names' text order.
    pub fn enumeration(&mut self, name: &str, variants: &[(&str, Option<Ty>)]) -> Ty {
        let name = self.root(name);
        let variants = variants.iter().map(|(tag, payload)| (self.name(tag), *payload)).collect();
        self.intern(TyKind::Enum { name, variants, label: () })
    }

    pub fn tuple(&mut self, parts: &[Ty]) -> Ty {
        self.intern(TyKind::Tuple(parts.into(), ()))
    }

    /// An extern-defined type, recorded once. It is marked `#` where it is
    /// whole and written natively: where it has a type argument, and holds
    /// no uniform value. A type with no argument has no other form to be
    /// told apart from, so it is unmarked.
    pub fn opaque(&mut self, declared: OpaqueDecl<'_>) -> Ty {
        let id = self.root(declared.name);
        self.user_type(UserTypeDecl {
            id,
            type_params: vec![TyVarBound::Any; declared.type_args.len()].into(),
            carries_effect: declared.effect.is_some(),
            carries_identity: declared.identity.is_some(),
            lifetimes: declared.lifetimes,
        });
        let marked = if !declared.type_args.is_empty() && !declared.uniform { Mark::Marked } else { Mark::Unmarked };
        self.intern(TyKind::UserDefined {
            id,
            type_args: declared.type_args.into(),
            effect: declared.effect,
            identity: declared.identity,
            marked: MarkTerm::Known(marked),
        })
    }
}

/// A Rust type's acvus type.
pub trait Declared {
    /// Whether a value of the type holds a uniform value anywhere. A whole
    /// extern-defined type that holds none is written natively, so it is
    /// marked `#`.
    const UNIFORM: bool;

    fn declare(cx: &mut Declaring<'_>) -> TyId<Poly>;
}

/// The stand-in a declaration writes for its `N`-th type variable, inside a
/// `Uniform`: `Uniform<R, Var<N>>` declares `TyKind::Var(N)`. It is used only
/// to declare; at run time a type variable is `Uniform<R, Never>`.
pub struct Var<const N: u32>;

/// The stand-in for a declaration's `N`-th effect variable.
pub struct EffectAt<const N: u32>;

/// The stand-in for a declaration's `N`-th identity variable.
pub struct IdentityAt<const N: u32>;

/// An effect variable's stand-in, as an extern-defined type or a function
/// type carries it.
pub trait DeclaredEffect {
    fn effect() -> EffectTerm<Poly>;
}

/// An identity variable's stand-in, as an extern-defined type carries it.
pub trait DeclaredIdentity {
    fn identity() -> IdentityTerm<Poly>;
}

impl<const N: u32> Declared for Var<N> {
    const UNIFORM: bool = true;

    fn declare(cx: &mut Declaring<'_>) -> TyId<Poly> {
        cx.intern(TyKind::Var(N))
    }
}

impl<const N: u32> EffectVar for EffectAt<N> {}

impl<const N: u32> IdentityVar for IdentityAt<N> {}

impl<const N: u32> DeclaredEffect for EffectAt<N> {
    fn effect() -> EffectTerm<Poly> {
        EffectTerm::Var(N)
    }
}

impl<const N: u32> DeclaredIdentity for IdentityAt<N> {
    fn identity() -> IdentityTerm<Poly> {
        IdentityTerm::Var(N)
    }
}

impl<R, X> Declared for Uniform<R, X>
where
    R: Runtime,
    X: Declared,
{
    const UNIFORM: bool = true;

    fn declare(cx: &mut Declaring<'_>) -> TyId<Poly> {
        X::declare(cx)
    }
}

macro_rules! leaves {
    ($($leaf:ty => $kind:expr),* $(,)?) => {$(
        impl Declared for $leaf {
            const UNIFORM: bool = false;

            fn declare(cx: &mut Declaring<'_>) -> TyId<Poly> {
                cx.intern($kind)
            }
        }
    )*};
}

leaves! {
    i8 => TyKind::Int(IntTy::I8),
    i16 => TyKind::Int(IntTy::I16),
    i32 => TyKind::Int(IntTy::I32),
    i64 => TyKind::Int(IntTy::I64),
    u8 => TyKind::Int(IntTy::U8),
    u16 => TyKind::Int(IntTy::U16),
    u32 => TyKind::Int(IntTy::U32),
    u64 => TyKind::Int(IntTy::U64),
    f64 => TyKind::Float,
    bool => TyKind::Bool,
    char => TyKind::Char,
    () => TyKind::Unit,
    String => TyKind::String,
    str => TyKind::Str,
}

impl<X> Declared for [X]
where
    X: Declared,
{
    const UNIFORM: bool = X::UNIFORM;

    fn declare(cx: &mut Declaring<'_>) -> TyId<Poly> {
        let element = X::declare(cx);
        cx.intern(TyKind::Slice(element))
    }
}

impl<X> Declared for &X
where
    X: Declared + ?Sized,
{
    const UNIFORM: bool = X::UNIFORM;

    fn declare(cx: &mut Declaring<'_>) -> TyId<Poly> {
        let lent = X::declare(cx);
        cx.intern(TyKind::Ref(Mutability::Shared, lent))
    }
}

impl<X> Declared for &mut X
where
    X: Declared + ?Sized,
{
    const UNIFORM: bool = X::UNIFORM;

    fn declare(cx: &mut Declaring<'_>) -> TyId<Poly> {
        let lent = X::declare(cx);
        cx.intern(TyKind::Ref(Mutability::Mut, lent))
    }
}

impl<X> Declared for Option<X>
where
    X: Declared,
{
    const UNIFORM: bool = X::UNIFORM;

    fn declare(cx: &mut Declaring<'_>) -> TyId<Poly> {
        let payload = X::declare(cx);
        cx.intern(TyKind::Option(payload, ()))
    }
}

impl<X, E> Declared for Result<X, E>
where
    X: Declared,
    E: Declared,
{
    const UNIFORM: bool = X::UNIFORM || E::UNIFORM;

    fn declare(cx: &mut Declaring<'_>) -> TyId<Poly> {
        let (ok, err) = (X::declare(cx), E::declare(cx));
        cx.intern(TyKind::Result(ok, err, ()))
    }
}

/// Tuples declare at every arity their `Crossing` reaches.
macro_rules! tuples {
    ($(($($part:ident),+))+) => {$(
        impl<$($part),+> Declared for ($($part,)+)
        where
            $($part: Declared,)+
        {
            const UNIFORM: bool = $($part::UNIFORM)||+;

            fn declare(cx: &mut Declaring<'_>) -> TyId<Poly> {
                let parts = [$($part::declare(cx)),+];
                cx.tuple(&parts)
            }
        }
    )+};
}

tuples! {
    (A)
    (A, B)
    (A, B, C)
    (A, B, C, D)
    (A, B, C, D, E)
    (A, B, C, D, E, F)
    (A, B, C, D, E, F, G)
    (A, B, C, D, E, F, G, H)
    (A, B, C, D, E, F, G, H, I)
    (A, B, C, D, E, F, G, H, I, J)
    (A, B, C, D, E, F, G, H, I, J, K)
    (A, B, C, D, E, F, G, H, I, J, K, L)
}
