//! A generic type in an extern's signature states whether `#[extern_fn]`
//! reads it position by position (RFC-0096 rule 2); the macro never decides
//! it from the type's written name. The macro writes the flows of both
//! readings, and the declaration keeps the one `TyArg::LAYOUT` of the named
//! type states when it is built.

use std::marker::PhantomData;

use acvus_mir::ty::Laid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    Region,
    Type,
    Other,
}

/// # Safety
///
/// `PARAMS` has one entry per generic parameter of `Self`, in declaration
/// order, and `Self`'s `TyArg::poly_ty` is a `UserDefined` whose
/// `region_params` counts its `Region` parameters and whose `type_args` are
/// its `Type` parameters, in that order. A call reads a labelled flow's
/// positions off that type (`acvus_mir::analysis::loans::segments`), and a
/// region parameter is where the type holds what its lifetime borrows.
pub unsafe trait LaidOut {
    const PARAMS: &'static [ParamKind];
}

/// `T` brands the statement, so one type cannot state another's layout.
pub struct Layout<T>
where
    T: ?Sized,
{
    params: Option<&'static [ParamKind]>,
    of: PhantomData<fn() -> *const T>,
}

impl<T> Clone for Layout<T>
where
    T: ?Sized,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Layout<T> where T: ?Sized {}

impl<T> Layout<T>
where
    T: ?Sized,
{
    pub const UNREAD: Self = Layout {
        params: None,
        of: PhantomData,
    };

    pub const fn laid_out() -> Self
    where
        T: LaidOut,
    {
        Layout {
            params: Some(T::PARAMS),
            of: PhantomData,
        }
    }

    pub const fn is_laid_out(self) -> bool {
        self.params.is_some()
    }

    /// `#[extern_fn]` asserts this at compile time for every reading that
    /// lays the type out, so the flows it wrote for that reading are the
    /// type's.
    pub const fn admits(self, written: &[WrittenArg]) -> bool {
        let Some(params) = self.params else {
            return true;
        };
        if params.len() != written.len() {
            return false;
        }
        let mut at = 0;
        while at < params.len() {
            let fits = match (written[at], params[at]) {
                (WrittenArg::Lifetime, ParamKind::Region)
                | (WrittenArg::Positioned, ParamKind::Type)
                | (WrittenArg::NonType, ParamKind::Other)
                | (WrittenArg::Positionless, ParamKind::Type | ParamKind::Other) => true,
                (WrittenArg::Lifetime, _)
                | (WrittenArg::Positioned, _)
                | (WrittenArg::NonType, _)
                | (WrittenArg::Positionless, ParamKind::Region) => false,
            };
            if !fits {
                return false;
            }
            at += 1;
        }
        true
    }

    pub fn type_args(self, read: Vec<ReadArg>) -> Vec<Laid> {
        let params = self.params.unwrap_or(&[]);
        read.into_iter()
            .filter(|arg| params.get(arg.written_at) == Some(&ParamKind::Type))
            .map(|arg| arg.laid)
            .collect()
    }
}

/// How the macro read one written argument of a generic type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrittenArg {
    Lifetime,
    /// A type variable, or a type naming a lifetime, a type variable, a
    /// reference or a generic type.
    Positioned,
    /// An effect, length or identity variable, or the runtime.
    NonType,
    /// A type that names none of what `Positioned` names: it has no
    /// position, and stands at a type parameter or a parameter of another
    /// kind alike.
    Positionless,
}

/// One written argument the macro read as a type, and its shape.
pub struct ReadArg {
    pub written_at: usize,
    pub laid: Laid,
}
