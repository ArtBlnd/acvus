//! A registry is a manifest and a handler table; all registries are
//! combined once into the compiler's and the runtime's inputs (RFC-0021).

use std::fmt;
use std::sync::Arc;

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{
    CastRule, Effect, EffectTerm, IdentityTerm, InstanceSets, InstanceShape, ParamTerm, Poly,
    PolyBuilder, PolyTy, Repr, Task, TyTerm, TyVarBound, TypeArg, TypeRegistry, UserDefinedDecl,
    Viewed, matches_pattern, unify_patterns,
};
use acvus_utils::Interner;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::handler::{DeclaredInstance, ExternHandler, Instances};
use crate::instance::InstanceRun;
use crate::runtime::Runtime;
use crate::space::SpaceHooks;

// -- Declarations ------------------------------------------------------

/// A function as the compiler sees it.
pub struct FnDecl {
    pub qref: QualifiedRef,
    pub ty: PolyTy,
    /// The declared bound of each type variable of `ty`, by position.
    pub bounds: Vec<TyVarBound>,
    pub coercion: Option<Coercion>,
    /// The shared signature this function is an instance of (RFC-0019).
    pub instance_of: Option<QualifiedRef>,
    /// What each type variable is required to have an instance of
    /// (RFC-0067 Decision 1), in the order `#[extern_fn]` read the bounds.
    pub requires: Vec<Requirement>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coercion {
    /// A rule from the parameter's type to the return type (RFC-0023),
    /// registered alongside the function.
    Cast,
    /// The machine's reading of the storage behind a reference (RFC-0047
    /// §5, RFC-0062 Decision 3): `&Vec<T>` as `&[T]`, `&String` as `&str`.
    ///
    /// This is not written as a `Cast` because a `CastRule` is indexed by
    /// a user-defined head and both sides of a view are references, so
    /// `TypeRegistry::register_cast` would reject it; and because a cast
    /// is a name a script resolves, which a view must not be.
    View,
}

/// One `InstanceOf<sig::S<..>>` bound of a declaration: which of the
/// declaration's type variables carries it, and which signature it names.
/// The order of these on a `FnDecl` is the order of the entries in the
/// carrier `#[extern_fn]` writes for that variable.
pub struct Requirement {
    /// The variable's position among the declaration's type variables,
    /// which is its position in `FnDecl::bounds`.
    pub var: usize,
    pub signature: QualifiedRef,
    /// The highest task an instance this bound reaches may run at
    /// (RFC-0046): what the bound's own spelling drives, met with what the
    /// requiring Rust body can. `InstanceOf` is a call that returns, so it
    /// is `Task::Sync` however the body was declared; `InstanceOfAsync` in
    /// an `async fn` body is `Task::Async`, and in a plain `fn` body —
    /// which can only drain a ready future — `Task::Sync` again.
    pub calls: Task,
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
        && a.coercion == Some(Coercion::Cast)
        && b.coercion == Some(Coercion::Cast)
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
            coercion: Some(Coercion::Cast),
            instance_of: None,
            requires: Vec::new(),
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
    /// A declaration marked the machine's view whose type is not one.
    ViewShape {
        function: QualifiedRef,
    },
    /// A declaration requiring a signature no registry declares
    /// (RFC-0067 Decision 1).
    RequiredSignatureUnknown {
        function: QualifiedRef,
        signature: QualifiedRef,
    },
    /// A requirement on a signature one of whose instances has no mono
    /// glue: the requirement resolves to a plain `fn` at whichever type the
    /// call settles on, and for that instance there is none (RFC-0067
    /// Decision 1).
    RequiredInstanceWithoutGlue {
        signature: String,
        instance: String,
        ty: PolyTy,
    },
    /// An instance whose own bound stands at a variable its pattern does
    /// not hold as a type argument, so no ground type the instance is
    /// chosen at says what fills it (RFC-0067, step 3 second half).
    /// An instance whose pattern is the bare variable it requires an
    /// instance of: every type with an instance has one, so the set is its
    /// own definition and no ground type ever decides it.
    RequirementIsThePattern {
        instance: QualifiedRef,
        signature: QualifiedRef,
    },
    RequirementOffThePattern {
        instance: QualifiedRef,
        signature: QualifiedRef,
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
            Self::ViewShape { function } => write!(
                f,
                "view {function:?}: a view lends the run behind its one reference parameter at the same mutability"
            ),
            Self::RequiredSignatureUnknown {
                function,
                signature,
            } => write!(
                f,
                "{function:?} requires an instance of {signature:?}, which no registry declares"
            ),
            Self::RequiredInstanceWithoutGlue {
                signature,
                instance,
                ty,
            } => write!(
                f,
                "{signature} is required, so every instance of it is called through a mono \
                 glue — a plain `fn` of the call's own arguments — and {instance}, its \
                 instance at {ty:?}, has none. A declaration written `heavy`, one holding a \
                 `#[state]` value, and one taking a `&str` or a projection parameter are the \
                 shapes that have no mono glue. Give that instance one of the other shapes, \
                 or drop the requirement."
            ),
            Self::RequirementIsThePattern {
                instance,
                signature,
            } => write!(
                f,
                "{instance:?} is an instance of {signature:?} at a bare type variable and \
                 requires an instance of {signature:?} of that same variable: the set of types \
                 it admits is its own definition, and no ground type decides it. Stand the \
                 instance at a pattern, or drop the requirement."
            ),
            Self::RequirementOffThePattern {
                instance,
                signature,
            } => write!(
                f,
                "{instance:?} requires an instance of {signature:?} of a variable its own \
                 pattern does not hold as a type argument, so the ground type it is chosen at \
                 does not say what fills that variable"
            ),
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
    pub instances: InstanceTable,
}

/// Every shared signature's instances that have a mono glue, in the order
/// `Externs::combine` collected them; a call site's table is filled from it
/// once, at prepare.
///
/// An instance without a glue is not a row here, and it cannot be missed:
/// `Externs::combine` refuses a requirement on a signature that has one
/// (`CombineError::RequiredInstanceWithoutGlue`), so no requirement ever
/// resolves at a type only such an instance stands at.
#[derive(Default)]
pub struct InstanceTable {
    by_signature: FxHashMap<QualifiedRef, Vec<GlueAt>>,
}

/// One instance as `prepare` reads it: the pattern it stands at, and the
/// mono glue a requirement resolves to.
struct GlueAt {
    ty: PolyTy,
    run: InstanceRun,
}

impl<R> crate::handler::InstanceEntries<R> for InstanceTable
where
    R: Runtime,
{
    fn instance_at(&self, signature: QualifiedRef, ty: &acvus_mir::ty::Ty) -> InstanceRun {
        let ty = match ty {
            TyTerm::Ref(_, target) => &target.ty,
            at => at,
        };
        let Some(instances) = self.by_signature.get(&signature) else {
            panic!("{signature:?} is required but no registry declares it")
        };
        let Some(found) = instances.iter().find(|at| matches_pattern(ty, &at.ty)) else {
            panic!(
                "{signature:?} has no instance at {ty:?}, so the OneOf bound the requirement \
                 was met with admitted a type it does not hold (RFC-0067 Decision 1)"
            )
        };
        found.run
    }
}

/// One instance of one shared signature as the checker's `InstanceSets`
/// needs it: the pattern it stands at, what its own bounds require, and the
/// task its body runs at.
pub struct InstanceAt {
    /// The declaration this instance came from, which is what a refusal
    /// naming it has to print.
    pub qref: QualifiedRef,
    pub ty: PolyTy,
    /// Obligation across artifacts: `#[extern_fn]` decides which
    /// declarations get a mono glue, and this is `None` for the rest.
    pub run: Option<InstanceRun>,
    pub requires: Vec<BoundAt>,
    /// The task this instance's own body runs at: the tightest of the
    /// handlers the declaration contributed, which for a declaration with a
    /// `sync =` twin is the twin's `Task::Sync`.
    pub task: Task,
}

/// One bound of an instance's own declaration, as both the checker and
/// `prepare` read it.
pub type BoundAt = acvus_mir::ty::InnerBound;

/// Where a declaration's type variable stands inside the pattern its
/// instance is matched by. `None` where the variable stands somewhere a
/// ground type cannot be walked to by type-argument position.
fn path_to_var(pattern: &PolyTy, var: u32, at: &mut Vec<usize>) -> bool {
    match pattern {
        TyTerm::Var(v) => *v == var,
        TyTerm::UserDefined { type_args, .. } => type_args.iter().enumerate().any(|(n, arg)| {
            at.push(n);
            path_to_var(&arg.ty, var, at) || {
                at.pop();
                false
            }
        }),
        _ => false,
    }
}

/// The instances collected for one signature.
struct Collected<R: Runtime> {
    decl: SignatureDecl,
    instance_types: Vec<PolyTy>,
    instances: Vec<DeclaredInstance<R>>,
    /// One per declared instance, which `instances` is not: a declaration
    /// with a `sync =` companion contributes two handlers and one type.
    entries: Vec<InstanceAt>,
    casts: Vec<CastRule>,
}

impl<R: Runtime> Collected<R> {
    /// The patterns of the instances a body at `task` can reach.
    fn types_at(&self, task: Task) -> Vec<PolyTy> {
        self.entries
            .iter()
            .filter(|at| at.task <= task)
            .map(|at| at.ty.clone())
            .collect()
    }
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
                        entries: Vec::new(),
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

        let sets = Arc::new(InstanceSets::new(
            signatures
                .iter()
                .map(|(sig, c)| {
                    let shapes = c
                        .entries
                        .iter()
                        .map(|at| InstanceShape {
                            ty: at.ty.clone(),
                            requires: at.requires.clone(),
                        })
                        .collect();
                    (*sig, shapes)
                })
                .collect(),
        ));
        let mut functions = Vec::new();
        let mut handlers: Handlers<R> = FxHashMap::default();
        let mut required_signatures: FxHashSet<QualifiedRef> = FxHashSet::default();
        for ExternFn {
            mut decl,
            instances,
        } in plain
        {
            for required in &decl.requires {
                required_signatures.insert(required.signature);
                let collected = signatures.get(&required.signature).ok_or(
                    CombineError::RequiredSignatureUnknown {
                        function: decl.qref,
                        signature: required.signature,
                    },
                )?;
                decl.bounds[required.var] = meet(
                    &decl.bounds[required.var],
                    required.signature,
                    &collected.types_at(required.calls),
                    &sets,
                );
            }
            match decl.coercion {
                Some(Coercion::Cast) => types.register_cast(cast_rule(&decl)?),
                Some(Coercion::View) => types.register_machine_view(decl.qref, view_of(&decl)?),
                None => {}
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
        required_signatures.extend(
            collected
                .iter()
                .flat_map(|c| &c.entries)
                .flat_map(|at| &at.requires)
                .map(|bound| bound.signature),
        );
        let mut instance_table = InstanceTable::default();
        for mut c in collected {
            let required = required_signatures.contains(&c.decl.qref);
            let glues: Vec<GlueAt> = std::mem::take(&mut c.entries)
                .into_iter()
                .filter_map(|at| match at.run {
                    Some(run) => Some(Ok(GlueAt { ty: at.ty, run })),
                    None if required => Some(Err(CombineError::RequiredInstanceWithoutGlue {
                        signature: written(interner, c.decl.qref),
                        instance: written(interner, at.qref),
                        ty: at.ty,
                    })),
                    None => None,
                })
                .collect::<Result<_, CombineError>>()?;
            instance_table.by_signature.insert(c.decl.qref, glues);
            let mut bounds = c.decl.bounds;
            if let Some(first) = bounds.first_mut() {
                *first = meet(first, c.decl.qref, &c.instance_types, &sets);
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
            instances: instance_table,
        })
    }
}

/// The bound a variable keeps when it must also have an instance of
/// `signature`, which stands at the shapes in `allowed`.
fn meet(
    bound: &TyVarBound,
    signature: QualifiedRef,
    allowed: &[PolyTy],
    sets: &Arc<InstanceSets>,
) -> TyVarBound {
    let required = TyVarBound::instances_of(signature, allowed.to_vec(), Arc::clone(sets));
    bound.meet(&required).unwrap_or(TyVarBound::instances_of(
        signature,
        Vec::new(),
        Arc::clone(sets),
    ))
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
    if decl.coercion == Some(Coercion::Cast) {
        let mut rule = cast_rule(&decl)?;
        rule.fn_ref = sig;
        collected.casts.push(rule);
    }
    let requires = decl
        .requires
        .iter()
        .map(|required| {
            let mut at = Vec::new();
            let var = u32::try_from(required.var)
                .expect("a declaration's type variables are numbered by PolyVars");
            if !path_to_var(&ty, var, &mut at) {
                return Err(CombineError::RequirementOffThePattern {
                    instance: decl.qref,
                    signature: required.signature,
                });
            }
            if at.is_empty() {
                return Err(CombineError::RequirementIsThePattern {
                    instance: decl.qref,
                    signature: required.signature,
                });
            }
            Ok(BoundAt {
                signature: required.signature,
                at,
            })
        })
        .collect::<Result<Vec<BoundAt>, CombineError>>()?;
    collected.entries.push(InstanceAt {
        qref: decl.qref,
        ty: ty.clone(),
        run: admitted.iter().find_map(|i| i.handler.instance()),
        requires,
        task: admitted
            .iter()
            .map(|i| i.handler.task())
            .fold(Task::Heavy, Task::meet),
    });
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

fn view_of(decl: &FnDecl) -> Result<Viewed, CombineError> {
    Viewed::of_declaration(&decl.ty).ok_or(CombineError::ViewShape {
        function: decl.qref,
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
