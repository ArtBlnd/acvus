//! A generic function's signature and its impls (rfc_extern impls §1-§2,
//! compiler: extern fn §6): the generic function declares a signature and no
//! body, and an impl states which generic function it is one of, at the
//! choice of its variables it writes.

use acvus_extern::decl::{DeclTy, ImplKind, Interner, Role, TypeVar};
use acvus_extern::register::{Declared, GenericDeclared};
use acvus_extern::executor::Executor;
use acvus_extern::runtime::Runtime;
use acvus_extern::{Uniform, extern_fn, generic_fn};
use acvus_extern_test::WordRuntime;

#[generic_fn]
fn pick<R: Runtime, T: TypeVar<R>>(xs: &[T]) -> &T;

#[extern_fn(effect = pure, impl_of = pick<Uniform<R, i64>>)]
fn pick_i64<'a, R: Runtime>(#[executor] cx: &Executor<'_, R>, xs: &'a [Uniform<R, i64>]) -> &'a Uniform<R, i64> {
    xs.iter().max_by_key(|x| x.get(cx)).unwrap_or(&xs[0])
}

#[extern_fn(effect = pure, impl_of = pick<T>, generic)]
fn pick_any<R: Runtime, T: TypeVar<R>>(xs: &[T]) -> &T {
    &xs[0]
}

fn shared(target: DeclTy) -> DeclTy {
    DeclTy::Ref { mutable: false, target: Box::new(target) }
}

#[test]
fn a_generic_function_declares_its_signature() {
    let names = Interner::new();
    let pick = <pick as GenericDeclared<WordRuntime>>::declaration(&names);

    assert_eq!(pick.binders.types.len(), 1);
    assert_eq!(pick.params, [(names.intern("xs"), shared(DeclTy::Slice(Box::new(DeclTy::Param(0)))))]);
    assert_eq!(pick.result, shared(DeclTy::Param(0)));
}

#[test]
fn an_impl_states_the_generic_function_it_is_one_of() {
    let names = Interner::new();
    let generic_function = <pick as GenericDeclared<WordRuntime>>::path(&names);

    let concrete = <pick_i64 as Declared<WordRuntime>>::declaration(&pick_i64::new(), &names);
    assert_eq!(concrete.role, Role::ImplOf { generic_function, kind: ImplKind::Concrete });
    assert_eq!(concrete.params, [(names.intern("xs"), shared(DeclTy::Slice(Box::new(DeclTy::Int(acvus_mir::ty::IntTy::I64)))))]);

    let generic = <pick_any as Declared<WordRuntime>>::declaration(&pick_any::new(), &names);
    assert_eq!(generic.role, Role::ImplOf { generic_function, kind: ImplKind::Generic });
    assert_eq!(generic.params, <pick as GenericDeclared<WordRuntime>>::declaration(&names).params);
}

/// A signature holding a reference or a lifetime inside a part has no call
/// forms, and still declares: its impls are declared and checked as any.
#[generic_fn]
fn find<R: Runtime, T: TypeVar<R>>(xs: &T) -> Option<&T>;

#[generic_fn]
fn apply<'f, R: Runtime, E: acvus_extern::decl::EffectVar>(
    f: &Uniform<R, acvus_extern::callback::Callable<'f, (i64,), i64, E>>,
    n: i64,
) -> i64;

#[test]
fn a_signature_without_call_forms_declares() {
    let names = Interner::new();
    let find = <find as GenericDeclared<WordRuntime>>::declaration(&names);
    let apply = <apply as GenericDeclared<WordRuntime>>::declaration(&names);

    assert_eq!(find.result, DeclTy::Option(Box::new(shared(DeclTy::Param(0)))));
    assert_eq!(apply.params.len(), 2);
}
