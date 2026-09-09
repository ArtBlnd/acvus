//! An ExternFn joined to its handler, and the registry that hands a set of
//! them, with their types, to the compiler and the interpreter.

use std::future::Future;

use acvus_interpreter::{
    Executable, ExternHandler, FromValue, IntoValue, RuntimeError, into_async_extern_handler,
    into_sync_extern_handler,
};
use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{
    CastRule, Effect, EffectTerm, ParamTerm, Poly, PolyTy, TypeRegistry, UserDefinedDecl,
};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::ty_arg::{PolyVars, TyArg};

/// One external function: its acvus type and its runtime handler.
pub struct ExternFn {
    pub qref: QualifiedRef,
    pub ty: PolyTy,
    pub handler: ExternHandler,
    /// A cast is registered as a coercion rule from its parameter type to
    /// its return type as well as a function.
    pub cast: bool,
}

/// What a declared item contributes to a registry.
pub trait ExternFnDecl {
    fn decl(interner: &Interner) -> ExternFn;
}

pub trait ExternTypeDecl {
    fn type_decl(interner: &Interner) -> UserDefinedDecl;
}

/// Everything one registry contributes.
pub struct ExternItems {
    pub types: Vec<UserDefinedDecl>,
    pub fns: Vec<ExternFn>,
}

pub struct ExternRegistry {
    factory: Box<dyn FnOnce(&Interner) -> ExternItems>,
}

/// The two halves of a registered registry: functions for the graph and
/// executables for the interpreter.
pub struct Registered {
    pub functions: Vec<Function>,
    pub executables: FxHashMap<QualifiedRef, Executable>,
}

impl ExternRegistry {
    pub fn new(factory: impl FnOnce(&Interner) -> ExternItems + 'static) -> Self {
        Self {
            factory: Box::new(factory),
        }
    }

    pub fn register(self, interner: &Interner, type_registry: &mut TypeRegistry) -> Registered {
        let items = (self.factory)(interner);
        for decl in items.types {
            type_registry.register(decl);
        }
        let mut functions = Vec::with_capacity(items.fns.len());
        let mut executables = FxHashMap::default();
        for f in items.fns {
            if f.cast {
                type_registry.register_cast(cast_rule(&f));
            }
            functions.push(Function {
                qref: f.qref,
                kind: FnKind::Extern,
                ty: f.ty,
            });
            executables.insert(f.qref, Executable::Extern(f.handler));
        }
        Registered {
            functions,
            executables,
        }
    }
}

fn cast_rule(f: &ExternFn) -> CastRule {
    let PolyTy::Fn {
        params,
        ret,
        effect,
        ..
    } = &f.ty
    else {
        panic!("cast {:?}: type is not a function", f.qref);
    };
    let [param] = params.as_slice() else {
        panic!(
            "cast {:?}: expected exactly one parameter, got {}",
            f.qref,
            params.len()
        );
    };
    assert_eq!(
        *effect,
        EffectTerm::Known(Effect::Pure),
        "cast {:?}: a cast is pure",
        f.qref
    );
    CastRule {
        from: param.ty.clone(),
        to: (**ret).clone(),
        fn_ref: f.qref,
    }
}

// -- Declaring a concrete ExternFn from a closure ---------------------

/// A closure whose parameter types name the acvus parameter types.
pub trait SyncHandler<Args> {
    type Ret: TyArg + IntoValue + 'static;
    fn signature(interner: &Interner) -> PolyTy;
    fn into_handler(self) -> ExternHandler;
}

pub trait AsyncHandler<Args> {
    type Ret: TyArg + IntoValue + 'static;
    fn signature(interner: &Interner) -> PolyTy;
    fn into_handler(self) -> ExternHandler;
}

fn signature_of(interner: &Interner, params: Vec<PolyTy>, ret: PolyTy, effect: Effect) -> PolyTy {
    PolyTy::Fn {
        params: params
            .into_iter()
            .enumerate()
            .map(|(i, ty)| ParamTerm::<Poly>::new(interner.intern(&format!("_{i}")), ty))
            .collect(),
        ret: Box::new(ret),
        captures: vec![],
        effect: EffectTerm::Known(effect),
    }
}

macro_rules! impl_handlers {
    ($($A:ident : $a:ident),*) => {
        impl<F, $($A,)* R> SyncHandler<($($A,)*)> for F
        where
            F: Fn(&Interner, $($A),*) -> Result<R, RuntimeError> + Send + Sync + 'static,
            $($A: TyArg + FromValue + 'static,)*
            R: TyArg + IntoValue + 'static,
        {
            type Ret = R;
            fn signature(interner: &Interner) -> PolyTy {
                let vars = PolyVars::empty();
                signature_of(
                    interner,
                    vec![$($A::poly_ty(interner, &vars)),*],
                    R::poly_ty(interner, &vars),
                    Effect::Opaque,
                )
            }
            fn into_handler(self) -> ExternHandler {
                into_sync_extern_handler(move |i: &Interner, ($($a,)*): ($($A,)*)| self(i, $($a),*))
            }
        }

        impl<F, Fut, $($A,)* R> AsyncHandler<($($A,)*)> for F
        where
            F: Fn(Interner, $($A),*) -> Fut + Send + Sync + 'static,
            Fut: Future<Output = Result<R, RuntimeError>> + Send + 'static,
            $($A: TyArg + FromValue + 'static,)*
            R: TyArg + IntoValue + 'static,
        {
            type Ret = R;
            fn signature(interner: &Interner) -> PolyTy {
                let vars = PolyVars::empty();
                signature_of(
                    interner,
                    vec![$($A::poly_ty(interner, &vars)),*],
                    R::poly_ty(interner, &vars),
                    Effect::Opaque,
                )
            }
            fn into_handler(self) -> ExternHandler {
                into_async_extern_handler(move |i: Interner, ($($a,)*): ($($A,)*)| self(i, $($a),*))
            }
        }
    };
}

impl_handlers!();
impl_handlers!(A: a);
impl_handlers!(A: a, B: b);
impl_handlers!(A: a, B: b, C: c);
impl_handlers!(A: a, B: b, C: c, D: d);

impl ExternFn {
    /// A concrete, stateful ExternFn from a synchronous closure. The effect
    /// is set afterwards with `with_effect`; undeclared is Opaque.
    pub fn sync<Args, F>(interner: &Interner, name: &str, f: F) -> Self
    where
        F: SyncHandler<Args>,
    {
        Self {
            qref: QualifiedRef::root(interner.intern(name)),
            ty: F::signature(interner),
            handler: f.into_handler(),
            cast: false,
        }
    }

    pub fn r#async<Args, F: AsyncHandler<Args>>(interner: &Interner, name: &str, f: F) -> Self {
        Self {
            qref: QualifiedRef::root(interner.intern(name)),
            ty: F::signature(interner),
            handler: f.into_handler(),
            cast: false,
        }
    }

    pub fn with_effect(mut self, effect: Effect) -> Self {
        let PolyTy::Fn { effect: slot, .. } = &mut self.ty else {
            panic!("ExternFn {:?}: type is not a function", self.qref);
        };
        *slot = EffectTerm::Known(effect);
        self
    }
}
