//! A registry is a manifest and a handler table; all registries are
//! combined once into the compiler's and the runtime's inputs (RFC-0021).

use std::fmt;

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{
    CastRule, Effect, EffectTerm, IdentityTerm, ParamTerm, Poly, PolyBuilder, PolyTy, Repr, Task,
    TyTerm, TyVarBound, TypeArg, TypeRegistry, UserDefinedDecl, matches_pattern, unify_patterns,
};
use acvus_utils::Interner;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::handler::{DeclaredInstance, ExternHandler, Instances};
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

/// Whether two declarations are one family cast: one name, one type, both
/// casts with instances only. Every `Monomorphize` member signature naming
/// a family declares that family's casts, so the declarations meet and
/// their instances are one list.
fn same_family_cast<R>(
    a: &FnDecl,
    a_instances: &Instances<R>,
    b: &FnDecl,
    b_instances: &Instances<R>,
) -> bool
where
    R: Runtime,
{
    a.qref == b.qref
        && a.cast
        && b.cast
        && a.ty == b.ty
        && a_instances.generic.is_none()
        && b_instances.generic.is_none()
}

/// A type of a `Monomorphize` member signature at its two representations,
/// with the handlers converting between them.
pub struct MemberType<R>
where
    R: Runtime,
{
    pub specialized: PolyTy,
    pub uniform: PolyTy,
    /// From `specialized` to `uniform`.
    pub erase: ExternHandler<R>,
    /// From `uniform` to `specialized`.
    pub materialize: ExternHandler<R>,
}

/// The two casts a family type in a `Monomorphize` member signature
/// declares, each with one instance at this member: `F::erase` from
/// `F<#T>` to `F<T>` and `F::materialize` back. A composite whose family
/// sits below the top, such as `Option<Vec<#T>>`, is not a family type and
/// declares nothing: the solver's cast rules are keyed by a user-defined
/// head, so a value of that type crosses only at its own representation.
pub fn family_casts<R>(i: &Interner, member: MemberType<R>) -> Vec<ExternFn<R>>
where
    R: Runtime,
{
    let MemberType {
        specialized,
        uniform,
        erase,
        materialize,
    } = member;
    if specialized == uniform {
        return Vec::new();
    }
    let TyTerm::UserDefined {
        id,
        type_args,
        effect_args,
        identity_args,
    } = &specialized
    else {
        return Vec::new();
    };
    let mut b = PolyBuilder::new();
    let ty_vars: Vec<PolyTy> = type_args.iter().map(|_| b.fresh_ty_var()).collect();
    let effect_vars: Vec<EffectTerm<Poly>> =
        effect_args.iter().map(|_| b.fresh_effect_var()).collect();
    let identity_vars: Vec<IdentityTerm<Poly>> = identity_args
        .iter()
        .map(|_| b.fresh_identity_var())
        .collect();
    let pattern = |repr_of: fn(&TypeArg<Poly>) -> Repr<Poly>| PolyTy::UserDefined {
        id: *id,
        type_args: type_args
            .iter()
            .zip(&ty_vars)
            .map(|(arg, var)| TypeArg::new(repr_of(arg), var.clone()))
            .collect(),
        effect_args: effect_vars.clone(),
        identity_args: identity_vars.clone(),
    };
    let specialized_pattern = pattern(|arg| arg.repr);
    let uniform_pattern = pattern(|_| Repr::Uniform);
    let family = written(i, *id);
    let fn_ty = |from: PolyTy, to: PolyTy| PolyTy::Fn {
        params: vec![ParamTerm::<Poly>::new(i.intern("value"), from)],
        ret: Box::new(to),
        captures: vec![],
        effect: EffectTerm::Known(Effect::PURE),
    };
    let cast = |name: &str, generic: PolyTy, instance: PolyTy, handler| ExternFn {
        decl: FnDecl {
            qref: QualifiedRef::qualified(i.intern(&family), i.intern(name)),
            ty: generic,
            bounds: vec![TyVarBound::Any; ty_vars.len()],
            cast: true,
            instance_of: None,
        },
        instances: Instances {
            concrete: vec![DeclaredInstance {
                signature: instance,
                handler,
                admits: Task::Heavy,
            }],
            generic: None,
        },
    };
    vec![
        cast(
            "erase",
            fn_ty(specialized_pattern.clone(), uniform_pattern.clone()),
            fn_ty(specialized.clone(), uniform.clone()),
            erase,
        ),
        cast(
            "materialize",
            fn_ty(uniform_pattern, specialized_pattern),
            fn_ty(uniform, specialized),
            materialize,
        ),
    ]
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

pub type Handlers<R> = FxHashMap<QualifiedRef, Vec<ExternHandler<R>>>;

/// What one registry contributes.
pub struct Contribution<R: Runtime> {
    pub manifest: Manifest,
    pub instances: FxHashMap<QualifiedRef, Instances<R>>,
    pub space: FxHashMap<QualifiedRef, SpaceHooks<R>>,
}

impl<R: Runtime> Contribution<R> {
    /// A contribution with the declarations and nothing running yet: what
    /// `extern_registry!` opens with, so the map types stay here.
    pub fn of(manifest: Manifest) -> Self {
        Self {
            manifest,
            instances: FxHashMap::default(),
            space: FxHashMap::default(),
        }
    }

    /// The space hooks of one declared type (RFC-0033).
    pub fn register_space(&mut self, qref: QualifiedRef, hooks: SpaceHooks<R>) {
        self.space.insert(qref, hooks);
    }

    /// Adds a declared function; a second declaration of one family cast
    /// adds its instances to the first.
    pub fn declare(&mut self, f: ExternFn<R>) {
        let declared = self.manifest.fns.iter().find(|d| d.qref == f.decl.qref);
        if let Some(decl) = declared
            && let Some(instances) = self.instances.get_mut(&decl.qref)
            && same_family_cast(decl, instances, &f.decl, &f.instances)
        {
            instances.add_concrete(f.instances.concrete);
            return;
        }
        self.instances.insert(f.decl.qref, f.instances);
        self.manifest.fns.push(f.decl);
    }
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
    CastShape {
        function: QualifiedRef,
        reason: &'static str,
    },
    /// A handler that runs above the task its declaration names (RFC-0046).
    HandlerTask {
        function: String,
        declared: Task,
        handler: Task,
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
            Self::CastShape { function, reason } => write!(f, "cast {function:?}: {reason}"),
            Self::HandlerTask {
                function,
                declared,
                handler,
            } => write!(
                f,
                "{function} is declared at Task::{declared}, so its handler \
                 may not run at Task::{handler}"
            ),
        }
    }
}

/// A qualified name as a script writes it.
fn written(i: &Interner, q: QualifiedRef) -> String {
    match q.namespace {
        Some(ns) => format!("{}::{}", i.resolve(ns), i.resolve(q.name)),
        None => i.resolve(q.name).to_string(),
    }
}

/// The task a declared type claims. A declaration generic in its effect
/// makes no claim here — its task is whatever the caller substitutes — so
/// the claim it could contradict does not exist.
fn declared_task(ty: &PolyTy) -> Option<Task> {
    match ty {
        PolyTy::Fn {
            effect: EffectTerm::Known(effect),
            ..
        } => Some(effect.task),
        _ => None,
    }
}

/// A declaration's task is the ceiling of its handler's: the glue that
/// runs a handler at `Async` or `Heavy` suspends the caller, and a
/// declaration claiming a lower task promises a caller it will not.
fn ceiling_admits<R>(
    i: &Interner,
    function: QualifiedRef,
    declared: &PolyTy,
    handler: &ExternHandler<R>,
) -> Result<(), CombineError>
where
    R: Runtime,
{
    let Some(declared) = declared_task(declared) else {
        return Ok(());
    };
    let handler = handler.task();
    (declared.join(handler) == declared)
        .then_some(())
        .ok_or_else(|| CombineError::HandlerTask {
            function: written(i, function),
            declared,
            handler,
        })
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
    instances: Vec<DeclaredInstance<R>>,
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
                    Some(sig) => add_instance(interner, &mut signatures, decl, instances, sig)?,
                    None => {
                        let declared = plain
                            .iter_mut()
                            .find(|f| same_family_cast(&f.decl, &f.instances, &decl, &instances));
                        match declared {
                            Some(existing) => existing.instances.add_concrete(instances.concrete),
                            None => {
                                if !names.insert(decl.qref) {
                                    return Err(CombineError::DuplicateName(decl.qref));
                                }
                                plain.push(ExternFn { decl, instances });
                            }
                        }
                    }
                }
            }
        }

        let mut functions = Vec::new();
        let mut handlers: Handlers<R> = FxHashMap::default();
        for ExternFn { decl, instances } in plain {
            if decl.cast {
                types.register_cast(cast_rule(&decl)?);
            }
            for instance in &instances.concrete {
                ceiling_admits(interner, decl.qref, &instance.signature, &instance.handler)?;
            }
            if let Some(generic) = &instances.generic {
                ceiling_admits(interner, decl.qref, &decl.ty, generic)?;
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
    i: &Interner,
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
    let admitted = at_declared_type(&decl.ty, instances).ok_or_else(mismatch)?;
    for instance in &admitted {
        ceiling_admits(i, decl.qref, &instance.signature, &instance.handler)?;
    }
    if decl.cast {
        let mut rule = cast_rule(&decl)?;
        rule.fn_ref = sig;
        collected.casts.push(rule);
    }
    collected.instance_types.push(ty);
    collected.instances.extend(admitted);
    Ok(())
}

fn at_declared_type<R>(
    declared: &PolyTy,
    instances: Instances<R>,
) -> Option<Vec<DeclaredInstance<R>>>
where
    R: Runtime,
{
    match instances {
        Instances {
            concrete,
            generic: Some(handler),
        } if concrete.is_empty() => Some(vec![DeclaredInstance {
            signature: declared.clone(),
            handler,
            admits: Task::Heavy,
        }]),
        Instances {
            concrete,
            generic: None,
        } if !concrete.is_empty() && concrete.iter().all(|i| i.signature == *declared) => {
            Some(concrete)
        }
        _ => None,
    }
}

fn instance_type(signature: &PolyTy, instance: &PolyTy) -> Option<PolyTy> {
    let (PolyTy::Fn { params: sp, .. }, PolyTy::Fn { params: ip, .. }) = (signature, instance)
    else {
        return None;
    };
    sp.iter()
        .zip(ip)
        .find_map(|(s, i)| instance_at_first_var(&s.ty, &i.ty))
}

/// The heads this walk does not descend — a function type, an array, a
/// slice, an option, a result, a handle, an object, an enum — are a
/// decision rather than an omission. A signature's first variable is the
/// one the per-length instance match reads, and reaching it through a
/// container would make it that container's element instead; a signature
/// written that way keeps the `InstanceMismatch` it was refused with
/// before this walk replaced the bare `Var(0)` and `&Var(0)` cases.
fn instance_at_first_var(signature: &PolyTy, instance: &PolyTy) -> Option<PolyTy> {
    match (signature, instance) {
        (PolyTy::Var(0), t) => Some(t.clone()),
        (PolyTy::Ref(_, s), PolyTy::Ref(_, i)) => instance_at_first_var(&s.ty, &i.ty),
        (PolyTy::UserDefined { type_args: sa, .. }, PolyTy::UserDefined { type_args: ia, .. }) => {
            sa.iter()
                .zip(ia)
                .find_map(|(s, i)| instance_at_first_var(&s.ty, &i.ty))
        }
        (PolyTy::Tuple(se), PolyTy::Tuple(ie)) => se
            .iter()
            .zip(ie)
            .find_map(|(s, i)| instance_at_first_var(s, i)),
        _ => None,
    }
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
