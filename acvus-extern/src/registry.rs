//! An ExternFn joined to its handler, and the registry that hands a set of
//! them, with their types, to the compiler and a runtime.

use std::future::Future;
use std::sync::Arc;

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{
    CastRule, Effect, EffectTerm, ParamTerm, Poly, PolyTy, TyVarBound, TypeRegistry,
    UserDefinedDecl,
};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::handler::{ExternEntry, ExternHandler};
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, TyArg, TyVar};

/// One external function: its acvus type and its runtime handler.
pub struct ExternFn<R: Runtime> {
    pub qref: QualifiedRef,
    pub ty: PolyTy,
    /// The declared bound of each type variable of `ty`, by position.
    pub bounds: Vec<TyVarBound>,
    pub handler: ExternEntry<R>,
    /// A cast is registered as a coercion rule from its parameter type to
    /// its return type as well as a function.
    pub cast: bool,
}

pub trait ExternFnDecl<R: Runtime> {
    fn decl(interner: &Interner) -> ExternFn<R>;
}

pub trait ExternTypeDecl {
    fn type_decl(interner: &Interner) -> UserDefinedDecl;
}

/// Everything one registry contributes.
pub struct ExternItems<R: Runtime> {
    pub types: Vec<UserDefinedDecl>,
    pub fns: Vec<ExternFn<R>>,
}

pub struct ExternRegistry<R: Runtime> {
    factory: Box<dyn FnOnce(&Interner) -> ExternItems<R>>,
}

/// The parts of a registered registry: functions for the graph, and handlers
/// for the runtime.
pub struct Registered<R: Runtime> {
    pub functions: Vec<Function>,
    pub handlers: FxHashMap<QualifiedRef, ExternEntry<R>>,
}

impl<R: Runtime> ExternRegistry<R> {
    pub fn new(factory: impl FnOnce(&Interner) -> ExternItems<R> + 'static) -> Self {
        Self {
            factory: Box::new(factory),
        }
    }

    pub fn register(self, interner: &Interner, type_registry: &mut TypeRegistry) -> Registered<R> {
        let items = (self.factory)(interner);
        for decl in items.types {
            type_registry.register(decl);
        }
        let mut functions = Vec::with_capacity(items.fns.len());
        let mut handlers = FxHashMap::default();
        for f in items.fns {
            if f.cast {
                type_registry.register_cast(cast_rule(&f));
            }
            functions.push(Function {
                qref: f.qref,
                kind: FnKind::Extern { bounds: f.bounds },
                ty: f.ty,
            });
            handlers.insert(f.qref, f.handler);
        }
        Registered {
            functions,
            handlers,
        }
    }
}

fn cast_rule<R: Runtime>(f: &ExternFn<R>) -> CastRule {
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
        EffectTerm::Known(Effect::PURE),
        "cast {:?}: a cast is pure",
        f.qref
    );
    CastRule {
        from: param.ty.clone(),
        to: (**ret).clone(),
        fn_ref: f.qref,
    }
}

// Declaring a concrete ExternFn from a closure.

/// A closure whose parameter types name the acvus parameter types.
pub trait SyncHandler<R: Runtime, Args> {
    fn signature(interner: &Interner) -> PolyTy;
    fn into_handler(self) -> ExternHandler<R>;
}

pub trait AsyncHandler<R: Runtime, Args> {
    fn signature(interner: &Interner) -> PolyTy;
    fn into_handler(self) -> ExternHandler<R>;
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
        impl<R, F, $($A,)* Ret> SyncHandler<R, ($($A,)*)> for F
        where
            R: Runtime,
            F: Fn(&R, $($A),*) -> Result<Ret, R::Error> + Send + Sync + 'static,
            $($A: TyArg + TyVar,)*
            Ret: TyArg + TyVar,
        {
            fn signature(interner: &Interner) -> PolyTy {
                let vars = PolyVars::empty();
                signature_of(
                    interner,
                    vec![$($A::poly_ty(interner, &vars)),*],
                    Ret::poly_ty(interner, &vars),
                    Effect::OPAQUE,
                )
            }
            fn into_handler(self) -> ExternHandler<R> {
                ExternHandler::Sync(Arc::new(move |rt: &R, args: Vec<R::Value>| {
                    let mut args = args.into_iter();
                    $(let $a = unsafe {
                        rt.materialize::<$A>(args.next().expect("arity checked by typeck"))
                    };)*
                    debug_assert!(args.next().is_none(), "arity checked by typeck");
                    let ret = self(rt, $($a),*)?;
                    Ok(unsafe { rt.erase::<Ret>(ret) })
                }))
            }
        }

        impl<R, F, Fut, $($A,)* Ret> AsyncHandler<R, ($($A,)*)> for F
        where
            R: Runtime + Clone,
            F: Fn(R, $($A),*) -> Fut + Send + Sync + 'static,
            Fut: Future<Output = Result<Ret, R::Error>> + Send + 'static,
            $($A: TyArg + TyVar,)*
            Ret: TyArg + TyVar,
        {
            fn signature(interner: &Interner) -> PolyTy {
                let vars = PolyVars::empty();
                signature_of(
                    interner,
                    vec![$($A::poly_ty(interner, &vars)),*],
                    Ret::poly_ty(interner, &vars),
                    Effect::OPAQUE,
                )
            }
            fn into_handler(self) -> ExternHandler<R> {
                ExternHandler::Async(Arc::new(move |rt: R, args: Vec<R::Value>| {
                    let mut args = args.into_iter();
                    $(let $a = unsafe {
                        rt.materialize::<$A>(args.next().expect("arity checked by typeck"))
                    };)*
                    debug_assert!(args.next().is_none(), "arity checked by typeck");
                    let fut = self(rt.clone(), $($a),*);
                    Box::pin(async move {
                        let ret = fut.await?;
                        Ok(unsafe { rt.erase::<Ret>(ret) })
                    })
                }))
            }
        }
    };
}

impl_handlers!();
impl_handlers!(A: a);
impl_handlers!(A: a, B: b);
impl_handlers!(A: a, B: b, C: c);
impl_handlers!(A: a, B: b, C: c, D: d);

impl<R: Runtime> ExternFn<R> {
    /// A concrete, stateful ExternFn from a synchronous closure. The effect
    /// is set afterwards with `with_effect`; undeclared is Opaque.
    pub fn sync<Args, F>(interner: &Interner, name: &str, f: F) -> Self
    where
        F: SyncHandler<R, Args>,
    {
        Self {
            qref: QualifiedRef::root(interner.intern(name)),
            ty: F::signature(interner),
            bounds: vec![],
            handler: ExternEntry::Single(f.into_handler()),
            cast: false,
        }
    }

    pub fn r#async<Args, F>(interner: &Interner, name: &str, f: F) -> Self
    where
        F: AsyncHandler<R, Args>,
    {
        Self {
            qref: QualifiedRef::root(interner.intern(name)),
            ty: F::signature(interner),
            bounds: vec![],
            handler: ExternEntry::Single(f.into_handler()),
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
