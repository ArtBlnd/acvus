use std::marker::PhantomData;

use acvus_mir::ty::IntTy;
use acvus_utils::Interner;

use super::{DeclType, EffectNth, Nth, ParamMode};
use crate::boundary::entry::mode::{ByCallback, ByMut, ByRef, BySlice, BySliceMut, ByStr, ByUniform, ByValue};
use crate::boundary::stub::Stub;
use crate::boundary::uniform::Uniform;
use crate::callback::{Callback, Lent};
use crate::decl::{DeclEffect, DeclTy};

fn decl<T: DeclType + ?Sized>() -> DeclTy {
    let names = Interner::new();
    T::decl_ty(&names)
}

fn param<T: ParamMode>() -> DeclTy {
    let names = Interner::new();
    T::decl(&names)
}

/// Compiles only where `M` is `P`'s mode.
fn mode_of<P: ParamMode, M>(_: PhantomData<M>)
where
    P::Mode: Is<M>,
{
}

trait Is<T> {}
impl<T> Is<T> for T {}

fn shared(target: DeclTy) -> DeclTy {
    DeclTy::Ref { mutable: false, target: Box::new(target) }
}

fn exclusive(target: DeclTy) -> DeclTy {
    DeclTy::Ref { mutable: true, target: Box::new(target) }
}

#[test]
fn the_languages_types_state_their_declared_types() {
    assert_eq!(decl::<i64>(), DeclTy::Int(IntTy::I64));
    assert_eq!(decl::<u8>(), DeclTy::Int(IntTy::U8));
    assert_eq!(decl::<String>(), DeclTy::String);
    assert_eq!(decl::<Option<bool>>(), DeclTy::Option(Box::new(DeclTy::Bool)));
    assert_eq!(
        decl::<Result<(), char>>(),
        DeclTy::Result(Box::new(DeclTy::Unit), Box::new(DeclTy::Char))
    );
    assert_eq!(decl::<(f64, String)>(), DeclTy::Tuple(vec![DeclTy::F64, DeclTy::String]));
}

#[test]
fn a_type_variables_value_is_its_parameter() {
    assert_eq!(decl::<Uniform<Stub, Nth<1>>>(), DeclTy::Param(1));
    assert_eq!(decl::<Option<Uniform<Stub, Nth<0>>>>(), DeclTy::Option(Box::new(DeclTy::Param(0))));
}

#[test]
fn a_reference_is_a_ref_to_its_target() {
    assert_eq!(decl::<&str>(), shared(DeclTy::Str));
    assert_eq!(decl::<&mut String>(), exclusive(DeclTy::String));
    assert_eq!(decl::<&[Uniform<Stub, Nth<0>>]>(), shared(DeclTy::Slice(Box::new(DeclTy::Param(0)))));
    assert_eq!(
        decl::<&mut [Uniform<Stub, i64>]>(),
        exclusive(DeclTy::Slice(Box::new(DeclTy::Int(IntTy::I64))))
    );
}

#[test]
fn a_callback_is_a_function_type_at_its_effect_variable() {
    assert_eq!(
        decl::<Callback<'_, (Lent<str>, i64), bool, EffectNth<2>, Stub>>(),
        DeclTy::Fn {
            params: vec![shared(DeclTy::Str), DeclTy::Int(IntTy::I64)],
            result: Box::new(DeclTy::Bool),
            effect: DeclEffect::Var(2),
        }
    );
}

#[test]
fn a_parameters_type_picks_its_mode() {
    mode_of::<i64, ByValue<i64>>(PhantomData);
    mode_of::<Uniform<Stub, Nth<0>>, ByUniform<Nth<0>>>(PhantomData);
    mode_of::<&String, ByRef<String>>(PhantomData);
    mode_of::<&mut String, ByMut<String>>(PhantomData);
    mode_of::<&[Uniform<Stub, Nth<0>>], BySlice<Nth<0>>>(PhantomData);
    mode_of::<&mut [Uniform<Stub, u8>], BySliceMut<u8>>(PhantomData);
    mode_of::<&str, ByStr>(PhantomData);
    mode_of::<Callback<'_, (i64,), i64, EffectNth<0>, Stub>, ByCallback<(i64,), i64, EffectNth<0>>>(PhantomData);
    assert_eq!(param::<&str>(), shared(DeclTy::Str));
}
