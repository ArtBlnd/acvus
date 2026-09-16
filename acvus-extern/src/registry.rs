//! A registry is a manifest and a handler table; all registries are
//! combined once into the compiler's and the runtime's inputs (RFC-0021).

use std::fmt;

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{
    CastRule, Effect, EffectTerm, PolyTy, TyVarBound, TypeRegistry, UserDefinedDecl,
    matches_pattern, unify_patterns,
};
use acvus_utils::Interner;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::handler::{ExternHandler, Instance, Instances};
use crate::runtime::Runtime;
use crate::space::SpaceHooks;

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

/// One declared function with its instances, as `#[extern_fn]` produces it.
pub struct ExternFn<R: Runtime> {
    pub decl: FnDecl,
    pub instances: Instances<R>,
}

pub trait ExternTypeDecl {
    fn type_decl(interner: &Interner) -> UserDefinedDecl;
    /// The type's space hooks (RFC-0033); a type without them cannot be a
    /// context a space holds.
    fn space<R>() -> Option<SpaceHooks<R>>
    where
        R: Runtime,
    {
        None
    }
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

pub type Handlers<R> = FxHashMap<QualifiedRef, Vec<ExternHandler<R>>>;

/// What one registry contributes.
pub struct Contribution<R: Runtime> {
    pub manifest: Manifest,
    pub instances: FxHashMap<QualifiedRef, Instances<R>>,
    pub space: FxHashMap<QualifiedRef, SpaceHooks<R>>,
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
    /// Two instances of one signature whose types unify (RFC-0027).
    DuplicateInstance {
        signature: QualifiedRef,
        ty: PolyTy,
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
            } => write!(
                f,
                "{instance:?} instantiates unknown signature {signature:?}"
            ),
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
    pub space: FxHashMap<QualifiedRef, SpaceHooks<R>>,
}

/// The instances collected for one signature.
struct Collected<R: Runtime> {
    decl: SignatureDecl,
    instance_types: Vec<PolyTy>,
    instances: Vec<Instance<R>>,
    casts: Vec<CastRule>,
}

impl<R: Runtime> Externs<R> {
    pub fn combine(
        registries: Vec<Registry<R>>,
        interner: &Interner,
    ) -> Result<Self, CombineError> {
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
        let mut plain_manifests = Vec::new();
        let mut space: FxHashMap<QualifiedRef, SpaceHooks<R>> = FxHashMap::default();
        for c in contributions {
            let Contribution {
                manifest,
                instances,
                space: hooks,
            } = c;
            for decl in manifest.types {
                types.register(decl);
            }
            space.extend(hooks);
            for sig in manifest.signatures {
                signatures.insert(
                    sig.qref,
                    Collected {
                        decl: sig,
                        instance_types: Vec::new(),
                        instances: Vec::new(),
                        casts: Vec::new(),
                    },
                );
            }
            plain_manifests.push((manifest.fns, instances));
        }
        for (fns, mut instances) in plain_manifests {
            for decl in fns {
                let instances = instances
                    .remove(&decl.qref)
                    .unwrap_or_else(|| panic!("no handler for declared {:?}", decl.qref));
                match decl.instance_of {
                    Some(sig) => add_instance(&mut signatures, decl, instances, sig)?,
                    None => {
                        if !names.insert(decl.qref) {
                            return Err(CombineError::DuplicateName(decl.qref));
                        }
                        plain.push(ExternFn { decl, instances });
                    }
                }
            }
        }

        let mut functions = Vec::new();
        let mut handlers: Handlers<R> = FxHashMap::default();
        for ExternFn {
            mut decl,
            instances,
        } in plain
        {
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
                    instances: instances.signatures(),
                },
                ty: decl.ty,
            });
            handlers.insert(decl.qref, instances.into_handlers());
        }
        let mut collected: Vec<Collected<R>> = signatures.into_values().collect();
        collected.sort_by_key(|c| c.decl.qref);
        for c in collected {
            let mut bounds = c.decl.bounds;
            if let Some(first) = bounds.first_mut() {
                *first = meet(first, &c.instance_types);
            }
            for cast in c.casts {
                types.register_cast(cast);
            }
            let instances = Instances {
                concrete: c.instances,
                generic: None,
            };
            functions.push(Function {
                qref: c.decl.qref,
                kind: FnKind::Extern {
                    bounds,
                    instances: instances.signatures(),
                },
                ty: c.decl.ty,
            });
            handlers.insert(c.decl.qref, instances.into_handlers());
        }
        Ok(Externs {
            functions,
            types,
            handlers,
            space,
        })
    }
}

/// The bound a variable keeps when it must also have one of the shapes in
/// `allowed`.
fn meet(bound: &TyVarBound, allowed: &[PolyTy]) -> TyVarBound {
    bound
        .meet(&TyVarBound::OneOf(allowed.to_vec()))
        .unwrap_or(TyVarBound::OneOf(Vec::new()))
}

fn add_instance<R: Runtime>(
    signatures: &mut FxHashMap<QualifiedRef, Collected<R>>,
    decl: FnDecl,
    instances: Instances<R>,
    sig: QualifiedRef,
) -> Result<(), CombineError> {
    let collected = signatures
        .get_mut(&sig)
        .ok_or(CombineError::UnknownSignature {
            instance: decl.qref,
            signature: sig,
        })?;
    let mismatch = || CombineError::InstanceMismatch {
        instance: decl.qref,
        signature: sig,
    };
    if !matches_pattern(&decl.ty, &collected.decl.ty) {
        return Err(mismatch());
    }
    let ty = instance_type(&collected.decl.ty, &decl.ty).ok_or_else(mismatch)?;
    if collected
        .instance_types
        .iter()
        .any(|existing| unify_patterns(existing, &ty).is_some())
    {
        return Err(CombineError::DuplicateInstance { signature: sig, ty });
    }
    let Instances {
        concrete,
        generic: Some(handler),
    } = instances
    else {
        return Err(mismatch());
    };
    if !concrete.is_empty() {
        return Err(mismatch());
    }
    if decl.cast {
        let mut rule = cast_rule(&decl)?;
        rule.fn_ref = sig;
        collected.casts.push(rule);
    }
    collected.instance_types.push(ty);
    collected.instances.push(Instance {
        signature: decl.ty,
        handler,
    });
    Ok(())
}

/// The type the signature's first variable takes in `instance`: read off
/// the first parameter whose declared type is that variable, bare or
/// behind a reference.
fn instance_type(signature: &PolyTy, instance: &PolyTy) -> Option<PolyTy> {
    let (PolyTy::Fn { params: sp, .. }, PolyTy::Fn { params: ip, .. }) = (signature, instance)
    else {
        return None;
    };
    sp.iter().zip(ip).find_map(|(s, i)| match (&s.ty, &i.ty) {
        (PolyTy::Var(0), t) => Some(t.clone()),
        (PolyTy::Ref(_, inner), PolyTy::Ref(_, t)) if matches!(inner.ty, PolyTy::Var(0)) => {
            Some(t.ty.clone())
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
