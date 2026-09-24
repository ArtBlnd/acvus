use acvus_mir::ty::PolyTy;
use acvus_utils::Interner;

use crate::ty_arg::{PolyVars, TyArg};

/// An extension type named at `()` for an identity parameter is declared at
/// that parameter's own variable, because `acvus-mir`'s `lift_declaration`
/// makes every identity a host declares a variable the compilation mints a
/// source for. The two must stay in step.
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
