//! A registry is a manifest and a handler table; all registries are
//! combined once into the compiler's and the runtime's inputs (RFC-0021).

use std::fmt;

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{
    CastRule, Effect, EffectTerm, PolyTy, Ty, TyVarBound, TypeRegistry, UserDefinedDecl,
    matches_poly, try_freeze_poly,
};
use acvus_utils::Interner;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::handler::{ExternEntry, MonoHandler, MonoInstance};
use crate::runtime::Runtime;

// -- Declarations ------------------------------------------------------

/// A function as the compiler sees it.
pub struct FnDecl {
    pub qref: QualifiedRef,
    pub ty: PolyTy,
    /// The declared bound of each type variable of `ty`, by position.
    pub bounds: Vec<TyVarBound>,
    /// A cast is registered as a coercion rule from its parameter type to
    /// its return type as well as a function.
    pub cast: bool,
    /// The shared signature this function is an instance of (RFC-0019).
    pub instance_of: Option<QualifiedRef>,
    /// The shared signature each type variable requires, by position.
    pub requires: Vec<Option<QualifiedRef>>,
}

/// A shared signature: a name with a polymorphic type and no body.
pub struct SignatureDecl {
    pub qref: QualifiedRef,
    pub ty: PolyTy,
    pub bounds: Vec<TyVarBound>,
}

/// The declarations one registry contributes; nothing here names a runtime.
pub struct Manifest {
    pub types: Vec<UserDefinedDecl>,
    pub signatures: Vec<SignatureDecl>,
    pub fns: Vec<FnDecl>,
}

/// One declared function with its handler, as `#[extern_fn]` produces it.
pub struct ExternFn<R: Runtime> {
    pub decl: FnDecl,
    pub handler: ExternEntry<R>,
}

pub trait ExternTypeDecl {
    fn type_decl(interner: &Interner) -> UserDefinedDecl;
}

/// The marker type `extern_signature!` declares.
pub trait SharedSignature {
    fn qref(interner: &Interner) -> QualifiedRef;
    fn signature_decl(interner: &Interner) -> SignatureDecl;
}

/// `T: HasInstance<sig>`: a type parameter that requires an instance of the
/// signature `sig`. The macro reads it; every type satisfies the Rust bound.
pub trait HasInstance<Sig> {}
impl<T, Sig> HasInstance<Sig> for T {}

pub type Handlers<R> = FxHashMap<QualifiedRef, ExternEntry<R>>;

/// What one registry contributes.
pub struct Contribution<R: Runtime> {
    pub manifest: Manifest,
    pub handlers: Handlers<R>,
}

pub struct Registry<R: Runtime> {
    factory: Box<dyn FnOnce(&Interner) -> Contribution<R>>,
}

impl<R: Runtime> Registry<R> {
    pub fn new(factory: impl FnOnce(&Interner) -> Contribution<R> + 'static) -> Self {
        Self {
            factory: Box::new(factory),
        }
    }
}

// -- Combining ---------------------------------------------------------

#[derive(Debug)]
pub enum CombineError {
    DuplicateName(QualifiedRef),
    UnknownSignature {
        instance: QualifiedRef,
        signature: QualifiedRef,
    },
    InstanceMismatch {
        instance: QualifiedRef,
        signature: QualifiedRef,
    },
    DuplicateInstance {
        signature: QualifiedRef,
        ty: Ty,
    },
    RequiredSignatureUnknown {
        function: QualifiedRef,
        signature: QualifiedRef,
    },
    CastShape {
        function: QualifiedRef,
        reason: &'static str,
    },
}

impl fmt::Display for CombineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateName(q) => write!(f, "name {q:?} is declared twice"),
            Self::UnknownSignature {
                instance,
                signature,
            } => write!(f, "{instance:?} instantiates unknown signature {signature:?}"),
            Self::InstanceMismatch {
                instance,
                signature,
            } => write!(f, "{instance:?} does not have the type of {signature:?}"),
            Self::DuplicateInstance { signature, ty } => {
                write!(f, "{signature:?} has two instances for {ty:?}")
            }
            Self::RequiredSignatureUnknown {
                function,
                signature,
            } => write!(f, "{function:?} requires unknown signature {signature:?}"),
            Self::CastShape { function, reason } => write!(f, "cast {function:?}: {reason}"),
        }
    }
}

impl std::error::Error for CombineError {}

/// Every registry combined: the compiler's functions and types, and the
/// runtime's handlers.
pub struct Externs<R: Runtime> {
    pub functions: Vec<Function>,
    pub types: TypeRegistry,
    pub handlers: Handlers<R>,
}

/// The instances collected for one signature.
struct Collected<R: Runtime> {
    decl: SignatureDecl,
    instance_types: Vec<Ty>,
    instances: Vec<MonoInstance<R>>,
}

impl<R: Runtime> Externs<R> {
    pub fn combine(registries: Vec<Registry<R>>, interner: &Interner) -> Result<Self, CombineError> {
        let mut types = TypeRegistry::new();
        let mut names: FxHashSet<QualifiedRef> = FxHashSet::default();
        let mut signatures: FxHashMap<QualifiedRef, Collected<R>> = FxHashMap::default();
        let mut plain: Vec<ExternFn<R>> = Vec::new();

        let contributions: Vec<Contribution<R>> = std::iter::once(crate::core::core_registry())
            .chain(registries)
            .map(|r| (r.factory)(interner))
            .collect();

        for c in &contributions {
            for sig in &c.manifest.signatures {
                if !names.insert(sig.qref) {
                    return Err(CombineError::DuplicateName(sig.qref));
                }
            }
        }
        for c in contributions {
            let Contribution {
                manifest,
                mut handlers,
            } = c;
            for decl in manifest.types {
                types.register(decl);
            }
            for sig in manifest.signatures {
                signatures.insert(
                    sig.qref,
                    Collected {
                        decl: sig,
                        instance_types: Vec::new(),
                        instances: Vec::new(),
                    },
                );
            }
            for decl in manifest.fns {
                let handler = handlers
                    .remove(&decl.qref)
                    .unwrap_or_else(|| panic!("no handler for declared {:?}", decl.qref));
                match decl.instance_of {
                    Some(sig) => add_instance(&mut signatures, decl, handler, sig)?,
                    None => {
                        if !names.insert(decl.qref) {
                            return Err(CombineError::DuplicateName(decl.qref));
                        }
                        plain.push(ExternFn { decl, handler });
                    }
                }
            }
        }

        let mut functions = Vec::new();
        let mut handlers: Handlers<R> = FxHashMap::default();
        for ExternFn { mut decl, handler } in plain {
            for (i, required) in decl.requires.iter().enumerate() {
                let Some(sig) = required else {
                    continue;
                };
                let collected =
                    signatures
                        .get(sig)
                        .ok_or_else(|| CombineError::RequiredSignatureUnknown {
                            function: decl.qref,
                            signature: *sig,
                        })?;
                decl.bounds[i] = meet(&decl.bounds[i], &collected.instance_types);
            }
            if decl.cast {
                types.register_cast(cast_rule(&decl)?);
            }
            functions.push(Function {
                qref: decl.qref,
                kind: FnKind::Extern {
                    bounds: decl.bounds,
                },
                ty: decl.ty,
            });
            handlers.insert(decl.qref, handler);
        }
        let mut collected: Vec<Collected<R>> = signatures.into_values().collect();
        collected.sort_by_key(|c| c.decl.qref);
        for c in collected {
            let mut bounds = c.decl.bounds;
            if let Some(first) = bounds.first_mut() {
                *first = meet(first, &c.instance_types);
            }
            functions.push(Function {
                qref: c.decl.qref,
                kind: FnKind::Extern { bounds },
                ty: c.decl.ty,
            });
            handlers.insert(
                c.decl.qref,
                ExternEntry::Mono(MonoHandler {
                    instances: c.instances,
                }),
            );
        }
        Ok(Externs {
            functions,
            types,
            handlers,
        })
    }
}

/// The bound a variable keeps when it must also lie in `allowed`.
fn meet(bound: &TyVarBound, allowed: &[Ty]) -> TyVarBound {
    match bound {
        TyVarBound::Any => TyVarBound::OneOf(allowed.to_vec()),
        TyVarBound::OneOf(tys) => {
            TyVarBound::OneOf(tys.iter().filter(|t| allowed.contains(t)).cloned().collect())
        }
    }
}

fn add_instance<R: Runtime>(
    signatures: &mut FxHashMap<QualifiedRef, Collected<R>>,
    decl: FnDecl,
    handler: ExternEntry<R>,
    sig: QualifiedRef,
) -> Result<(), CombineError> {
    let collected = signatures
        .get_mut(&sig)
        .ok_or(CombineError::UnknownSignature {
            instance: decl.qref,
            signature: sig,
        })?;
    let concrete = try_freeze_poly(&decl.ty).ok_or(CombineError::InstanceMismatch {
        instance: decl.qref,
        signature: sig,
    })?;
    if !matches_poly(&concrete, &collected.decl.ty) {
        return Err(CombineError::InstanceMismatch {
            instance: decl.qref,
            signature: sig,
        });
    }
    let ty = instance_type(&collected.decl.ty, &concrete).ok_or(CombineError::InstanceMismatch {
        instance: decl.qref,
        signature: sig,
    })?;
    if collected.instance_types.contains(&ty) {
        return Err(CombineError::DuplicateInstance { signature: sig, ty });
    }
    let ExternEntry::Single(handler) = handler else {
        return Err(CombineError::InstanceMismatch {
            instance: decl.qref,
            signature: sig,
        });
    };
    collected.instance_types.push(ty);
    collected.instances.push(MonoInstance {
        signature: decl.ty,
        handler,
    });
    Ok(())
}

/// The type the signature's first variable takes in `concrete`: read off
/// the first parameter whose declared type is that variable, bare or
/// behind a reference.
fn instance_type(signature: &PolyTy, concrete: &Ty) -> Option<Ty> {
    let (PolyTy::Fn { params: sp, .. }, Ty::Fn { params: cp, .. }) = (signature, concrete) else {
        return None;
    };
    sp.iter().zip(cp).find_map(|(s, c)| match (&s.ty, &c.ty) {
        (PolyTy::Var(0), t) => Some(t.clone()),
        (PolyTy::Ref(_, inner), Ty::Ref(_, t)) if matches!(**inner, PolyTy::Var(0)) => {
            Some((**t).clone())
        }
        _ => None,
    })
}

fn cast_rule(decl: &FnDecl) -> Result<CastRule, CombineError> {
    let PolyTy::Fn {
        params,
        ret,
        effect,
        ..
    } = &decl.ty
    else {
        return Err(CombineError::CastShape {
            function: decl.qref,
            reason: "type is not a function",
        });
    };
    let [param] = params.as_slice() else {
        return Err(CombineError::CastShape {
            function: decl.qref,
            reason: "a cast takes exactly one parameter",
        });
    };
    if *effect != EffectTerm::Known(Effect::PURE) {
        return Err(CombineError::CastShape {
            function: decl.qref,
            reason: "a cast is pure",
        });
    }
    Ok(CastRule {
        from: param.ty.clone(),
        to: (**ret).clone(),
        fn_ref: decl.qref,
    })
}
