use std::marker::PhantomData;

use acvus_mir::ty::IntTy;
use acvus_utils::Interner;

use super::{DeclType, EffectNth, Nth};
use crate::boundary::stub::Stub;
use crate::boundary::uniform::Uniform;
use crate::callback::Callable;
use crate::decl::{DeclEffect, DeclTy, decl_in};
use crate::runtime::{MutKey, Open, Place, RefKey};

fn decl<T: DeclType + ?Sized>() -> DeclTy {
    let names = Interner::new();
    T::decl_ty(&names)
}

fn param<T>() -> DeclTy
where
    T: Open<'static, Stub>,
{
    let names = Interner::new();
    decl_in::<Stub, T>(&names)
}

/// Compiles only where `K` is `P`'s key.
fn key_of<P, K>(_: PhantomData<K>)
where
    P: Place<'static, Stub>,
    P::Key: Is<K>,
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
    assert_eq!(param::<&str>(), shared(DeclTy::Str));
    assert_eq!(param::<&mut String>(), exclusive(DeclTy::String));
    assert_eq!(param::<&[Uniform<Stub, Nth<0>>]>(), shared(DeclTy::Slice(Box::new(DeclTy::Param(0)))));
    assert_eq!(
        param::<&mut [Uniform<Stub, i64>]>(),
        exclusive(DeclTy::Slice(Box::new(DeclTy::Int(IntTy::I64))))
    );
}

#[test]
fn a_function_value_is_a_function_type_at_its_effect_variable() {
    assert_eq!(
        decl::<Uniform<Stub, Callable<'_, (RefKey<str>, i64), bool, EffectNth<2>>>>(),
        DeclTy::Fn {
            params: vec![shared(DeclTy::Str), DeclTy::Int(IntTy::I64)],
            result: Box::new(DeclTy::Bool),
            effect: DeclEffect::Var(2),
        }
    );
}

#[test]
fn a_parameters_type_picks_its_key() {
    key_of::<i64, i64>(PhantomData);
    key_of::<Uniform<Stub, Nth<0>>, Uniform<Stub, Nth<0>>>(PhantomData);
    key_of::<&String, RefKey<String>>(PhantomData);
    key_of::<&mut String, MutKey<String>>(PhantomData);
    key_of::<&[Uniform<Stub, Nth<0>>], RefKey<[Uniform<Stub, Nth<0>>]>>(PhantomData);
    key_of::<&mut [Uniform<Stub, u8>], MutKey<[u8]>>(PhantomData);
    key_of::<&[u8], RefKey<[u8]>>(PhantomData);
    key_of::<&[Uniform<Stub, u8>], RefKey<[u8]>>(PhantomData);
    key_of::<Uniform<Stub, u8>, u8>(PhantomData);
    key_of::<&str, RefKey<str>>(PhantomData);
    key_of::<Uniform<Stub, Callable<'_, (i64,), i64, EffectNth<0>>>, Uniform<Stub, Callable<'_, (i64,), i64, EffectNth<0>>>>(
        PhantomData,
    );
    key_of::<&Uniform<Stub, Callable<'_, (i64,), i64, EffectNth<0>>>, RefKey<Uniform<Stub, Callable<'_, (i64,), i64, EffectNth<0>>>>>(
        PhantomData,
    );
    assert_eq!(param::<&str>(), shared(DeclTy::Str));
}
