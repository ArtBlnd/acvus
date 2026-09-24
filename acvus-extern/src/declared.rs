use acvus_mir::ty::{LenTerm, PolyTy, TypeArg};
use acvus_utils::Interner;

use crate::registry::ExternTypeDecl;
use crate::ty_arg::{PolyVars, TyArg};

/// An extension type named at `()` for an identity parameter is declared at
/// that parameter's own variable, because `acvus-mir`'s `lift_declaration`
/// makes every identity a host declares a variable the compilation mints a
/// source for. `acvus-extern/tests/decl.rs`
/// (`an_identity_parameter_is_declared_at_a_variable`) holds the two in step.
///
/// The language's `!` (`Bottom`) has no impl, and that is a decision: a host
/// that states no type is the runtime's own tooling (RFC-0090 rule 6).
pub trait Declared: 'static {
    fn declared(interner: &Interner) -> PolyTy;
}

macro_rules! declared_as_ty_arg {
    ($($t:ty),* $(,)?) => {
        $(
            impl Declared for $t {
                fn declared(interner: &Interner) -> PolyTy {
                    <$t as TyArg>::poly_ty(interner, &PolyVars::empty())
                }
            }
        )*
    };
}

declared_as_ty_arg!(i8, i16, i32, i64, u8, u16, u32, u64, f64, char, String, bool, ());

impl<T> Declared for Option<T>
where
    T: Declared,
{
    fn declared(interner: &Interner) -> PolyTy {
        PolyTy::Option(Box::new(T::declared(interner)))
    }
}

impl<T, E> Declared for Result<T, E>
where
    T: Declared,
    E: Declared,
{
    fn declared(interner: &Interner) -> PolyTy {
        PolyTy::Result(
            Box::new(T::declared(interner)),
            Box::new(E::declared(interner)),
        )
    }
}

macro_rules! declared_tuple {
    ($($T:ident),+) => {
        impl<$($T),+> Declared for ($($T,)+)
        where
            $($T: Declared,)+
        {
            fn declared(interner: &Interner) -> PolyTy {
                PolyTy::Tuple(vec![$($T::declared(interner)),+])
            }
        }
    };
}

declared_tuple!(A);
declared_tuple!(A, B);
declared_tuple!(A, B, C);
declared_tuple!(A, B, C, D);
declared_tuple!(A, B, C, D, E);
declared_tuple!(A, B, C, D, E, F);
declared_tuple!(A, B, C, D, E, F, G);
declared_tuple!(A, B, C, D, E, F, G, H);

impl<T, const N: usize> Declared for [T; N]
where
    T: Declared,
{
    fn declared(interner: &Interner) -> PolyTy {
        PolyTy::Array(Box::new(T::declared(interner)), LenTerm::Known(N))
    }
}

/// Obligation across artifacts: the checker settles a `Vec` a script builds
/// with a uniform argument, since its elements are the runtime's values, and
/// this declaration must name the same representation for a page to read it.
impl<T> Declared for Vec<T>
where
    T: Declared + Send + Sync,
{
    fn declared(interner: &Interner) -> PolyTy {
        PolyTy::UserDefined {
            id: PolyVars::empty().extension::<Vec<T>>(interner),
            type_args: vec![TypeArg::uniform(T::declared(interner))],
            effect_args: vec![],
            identity_args: vec![],
            region_params: <Vec<T> as ExternTypeDecl>::REGION_PARAMS,
        }
    }
}
