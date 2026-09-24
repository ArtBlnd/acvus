//! `Vec<T>`: the dynamic-length sequence is the language's `Vec<T>` and
//! Rust's `Vec<T>` alike. In a uniform slot it crosses the boundary as the
//! runtime's `Vec<Owned<Rt>>` (RFC-0039 rule 5): as the whole buffer when the element
//! is a value in place, element by element when the element converts. In a
//! `#` slot it crosses as one box holding the Rust `Vec<T>` itself.

use std::mem::ManuallyDrop;

use acvus_mir::ty::{Poly, Ty, TypeArg};

use crate::canonical::Canonical;
use crate::obj::{InPlaceElement, OneValue, stored_as_container_of};
use crate::owned::Owned;
use crate::registry::ExternTypeDecl;
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, SlotRepr, TyArg, Var, kind};
use crate::{Interner, PolyTy, QualifiedRef, TyVarBound, UserDefinedDecl};

/// # Safety
/// `stored_as_container_of::<T, Rt>()`.
unsafe fn into_values<T, Rt>(items: Vec<T>) -> Vec<Owned<Rt>>
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    let mut items = ManuallyDrop::new(items);
    // SAFETY: `T` is `Value` or `repr(transparent)` over it, and `Owned<Rt>`
    // is `repr(transparent)` over `Value`, so all three element types share
    // size and alignment and the buffer came from one allocator.
    unsafe { Vec::from_raw_parts(items.as_mut_ptr().cast(), items.len(), items.capacity()) }
}

/// # Safety
/// As `into_values`.
unsafe fn from_values<T, Rt>(items: Vec<Owned<Rt>>) -> Vec<T>
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    let mut items = ManuallyDrop::new(items);
    // SAFETY: as in `into_values`.
    unsafe { Vec::from_raw_parts(items.as_mut_ptr().cast(), items.len(), items.capacity()) }
}

// SAFETY: the element is its own `At<'a>`.
unsafe impl<T> crate::Branded for Vec<T>
where
    T: crate::Branded,
{
    type At<'a> = Vec<T::At<'a>>;
}

crate::cross_one_value!(Vec<T>, T: crate::OneValue<__Rt>);

// SAFETY: each element crosses by `T`'s own crossing, or in place where `T` is
// stored as the runtime's value, inside the runtime's `Vec<Owned<Rt>>`; nothing
// else crosses, and the capability is not kept.
unsafe impl<T, Rt> OneValue<Rt> for Vec<T>
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    fn erase(self, rt: crate::Crossing<'_, Rt>) -> Rt::Value {
        let items: Vec<Owned<Rt>> = if stored_as_container_of::<T, Rt>() {
            // SAFETY: the branch condition is `into_values`'s contract.
            unsafe { into_values::<T, Rt>(self) }
        } else {
            self.into_iter().map(|v| Owned::erased(rt, v)).collect()
        };
        // SAFETY: `Vec<T>` is stored as `Vec<Owned<Rt>>` (RFC-0039 rule 5,
        // RFC-0048 rule 7).
        unsafe { rt.erase::<Vec<Owned<Rt>>>(items) }
    }

    unsafe fn materialize(rt: crate::Crossing<'_, Rt>, value: Rt::Value) -> Self {
        // SAFETY: the caller's contract, and `erase` boxes a `Vec<Owned<Rt>>`.
        let items = unsafe { rt.materialize::<Vec<Owned<Rt>>>(value) };
        if stored_as_container_of::<T, Rt>() {
            // SAFETY: the branch condition is `from_values`'s contract.
            unsafe { from_values::<T, Rt>(items) }
        } else {
            // SAFETY: the caller's contract, forwarded: `erase` erased every
            // element from a `T`.
            items
                .into_iter()
                .map(|v| unsafe { T::materialize(rt, v.into_value(rt.holding())) })
                .collect()
        }
    }
}

impl<T, Rt> crate::Borrowable<Rt> for Vec<T>
where
    T: InPlaceElement<Rt>,
    Rt: Runtime,
{
    unsafe fn deref<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a Self {
        // SAFETY: the caller's contract, and `erase` boxes a `Vec<Owned<Rt>>`.
        // SAFETY: the caller's contract names the storage a crossing wrote
        // at this type.
        T::in_place(unsafe { crate::Holding::new() }, unsafe {
            rt.deref::<Vec<Owned<Rt>>>(reference)
        })
    }

    unsafe fn deref_mut<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut Self {
        // SAFETY: the caller's contract, exclusively, and `erase` boxes a
        // `Vec<Owned<Rt>>`.
        // SAFETY: as `deref`'s.
        T::in_place_mut(unsafe { crate::Holding::new() }, unsafe {
            rt.deref_mut::<Vec<Owned<Rt>>>(reference)
        })
    }
}

crate::cross_whole!(crate::Specialized, Vec<T>, T: Var<kind::Type> + crate::Branded);

impl<T, Rt> crate::BorrowableSpecialized<Rt> for Vec<T>
where
    T: Var<kind::Type> + crate::Branded,
    Rt: Runtime,
{
    crate::whole_box_in_place!(Vec<T>, Rt);
}

impl<T> Var<kind::Type> for Vec<T> where T: Var<kind::Type> {}

// SAFETY: the element is its own canonical form's.
unsafe impl<T> Canonical<kind::Type> for Vec<T>
where
    T: Var<kind::Type>,
{
    type Canon = Vec<T::Canon>;
}

impl<T> TyArg for Vec<T>
where
    T: TyArg + Send + Sync + 'static,
{
    const SLOT: SlotRepr = T::SLOT;

    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::UserDefined {
            id: vars.extension::<Self>(i),
            type_args: vec![T::slot(i, vars)],
            effect_args: vec![],
            identity_args: vec![],
            region_params: <Self as ExternTypeDecl>::REGION_PARAMS,
        }
    }

    /// A `Vec` held in another type's box is the Rust `Vec` of what its
    /// element is held as: no `ρ` enters a held tree.
    fn held(i: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
        TypeArg::specialized(PolyTy::UserDefined {
            id: vars.extension::<Self>(i),
            type_args: vec![T::held(i, vars)],
            effect_args: vec![],
            identity_args: vec![],
            region_params: <Self as ExternTypeDecl>::REGION_PARAMS,
        })
    }
}

impl<T> ExternTypeDecl for Vec<T>
where
    T: Send + Sync + 'static,
{
    type DeclarationForm = Vec<()>;

    const REGION_PARAMS: usize = 0;

    fn type_decl(i: &Interner) -> UserDefinedDecl {
        UserDefinedDecl {
            qref: QualifiedRef::root(i.intern("Vec")),
            type_params: vec![TyVarBound::Any],
            effect_params: 0,
            identity_params: 0,
            region_params: Self::REGION_PARAMS,
            specializable: vec![true],
            vars: vec![crate::DeclaredVar {
                name: i.intern("T"),
                // SAFETY: a `Vec` runs no code of its own on its elements
                // but moving and releasing them, and every std handler over
                // it keeps no element past its call (RFC-0079 rule 8).
                lending: crate::Lending::Lent(unsafe { crate::NotKept::asserted() }),
            }],
        }
    }
}

/// `Vec<elem>` as a concrete type, for contexts declared outside a script.
pub fn vec_ty(interner: &Interner, elem: Ty) -> Ty {
    Ty::UserDefined {
        id: QualifiedRef::root(interner.intern("Vec")),
        type_args: vec![TypeArg::uniform(elem)],
        effect_args: vec![],
        identity_args: vec![],
        region_params: <Vec<()> as ExternTypeDecl>::REGION_PARAMS,
    }
}
