//! A registry is a manifest and a handler table; all registries are
//! combined once into the compiler's and the runtime's inputs (RFC-0021).

use std::fmt;

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::laws::{
    BinaryLaws, Identity, LawRole, Laws, Postcondition, Reaches, Returns, Unresolved,
};
use acvus_mir::ty::{
    CastRule, DuplicateType, Effect, EffectArg, EffectTerm, EffectVarBound, IdentityTerm,
    ParamTerm, Poly, PolyBuilder, PolyTy, RequirementSig, Task, TyTerm, TyVarBound, TypeArg,
    TypeRegistry, UserDefinedDecl, Viewed, bind_chosen, matches_pattern, unify_patterns,
};
use acvus_utils::Interner;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::handler::RequiredInstance;

use crate::handler::{DeclaredInstance, ExternHandler, Instances};
use crate::instance::InstanceRun;
use crate::name::{DeclarationForm, DeclaredType, NameKind, Named};
use crate::runtime::Runtime;
use crate::space::SpaceHooks;

// -- Declarations ------------------------------------------------------

/// A function as the compiler sees it.
pub struct FnDecl {
    pub qref: QualifiedRef,
    pub ty: PolyTy,
    /// The declared bound of each type variable of `ty`, by position.
    pub bounds: Vec<TyVarBound>,
    /// The declared bound of each effect variable of `ty`, by position.
    pub effect_bounds: Vec<EffectVarBound>,
    pub coercion: Option<Coercion>,
    /// The shared signature this function is an instance of (RFC-0019).
    pub instance_of: Option<QualifiedRef>,
    /// The instances this declaration requires (RFC-0067 rule 1), in the order
    /// `#[extern_fn]` read its `Instance` parameters.
    pub requires: Vec<Requirement>,
    pub names: Vec<Named>,
    pub laws: Laws,
    /// What the declaration promises of its result (RFC-0082 rule 4).
    pub ensures: Vec<Postcondition>,
    /// The places a call reaches through its reference arguments (RFC-0082
    /// rule 7).
    pub reaches: Reaches,
    pub returns: Returns,
    /// The weight in ticks of one call, when the declaration states it
    /// (`cost = N`, RFC-0066 rule 8); a declaration that states none weighs
    /// its family's row of the backend's table.
    pub cost: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coercion {
    /// A rule from the parameter's type to the return type (RFC-0023),
    /// registered alongside the function.
    Cast,
    /// The machine's reading of the storage behind a reference (RFC-0047 rule 3,
    /// RFC-0062 rule 3): `&Vec<T>` as `&[T]`, `&String` as `&str`.
    ///
    /// This is not written as a `Cast` because a `CastRule` is indexed by
    /// a user-defined head and both sides of a view are references, so
    /// `TypeRegistry::register_cast` would reject it; and because a cast
    /// is a name a script resolves, which a view must not be.
    View,
}

/// One `Instance<S<..>, I, Rt>` parameter of a declaration: the signature
/// it names, that signature's type at the declaration's own variables, and
/// the highest task an instance it reaches may run at (RFC-0046): what the
/// parameter's `Now`/`Later` spelling drives, met with what the requiring
/// Rust body can — an `async fn` body with a `sync =` twin requires what
/// the twin does.
pub struct Requirement {
    pub signature: QualifiedRef,
    pub pattern: PolyTy,
    pub calls: Task,
}

impl Requirement {
    pub fn signature_of(&self) -> RequirementSig {
        RequirementSig {
            signature: self.signature,
            pattern: self.pattern.clone(),
            calls: self.calls,
        }
    }
}

/// A shared signature: a name with a polymorphic type and no body.
pub struct SignatureDecl {
    pub qref: QualifiedRef,
    pub ty: PolyTy,
    pub bounds: Vec<TyVarBound>,
    /// The type variables of `ty` each instance chooses (RFC-0041): a
    /// variable bounded by `Chosen`, whose slots are `#` at the instance's
    /// own tree.
    pub chosen: Vec<u32>,
    pub names: Vec<Named>,
}

/// The declarations one registry contributes; nothing here names a runtime.
pub struct Manifest {
    pub types: Vec<DeclaredType>,
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
    let Some(FamilyPatterns {
        family,
        specialized: specialized_pattern,
        uniform: uniform_pattern,
        ty_vars,
    }) = FamilyPatterns::of(i, &specialized)
    else {
        return Vec::new();
    };
    let fn_ty = |from: PolyTy, to: PolyTy| PolyTy::Fn {
        params: vec![ParamTerm::<Poly>::new(i.intern("value"), from)],
        ret: Box::new(to),
        captures: vec![],
        effect: EffectTerm::Known(Effect::PURE),
        flows: acvus_mir::ty::Flows::Every.into(),
    };
    let cast = |name: &str, generic: PolyTy, instance: PolyTy, handler| ExternFn {
        decl: FnDecl {
            qref: QualifiedRef::qualified(i.intern(&family), i.intern(name)),
            ty: generic,
            bounds: vec![TyVarBound::Any; ty_vars],
            effect_bounds: vec![],
            coercion: Some(Coercion::Cast),
            instance_of: None,
            requires: Vec::new(),
            names: Vec::new(),
            laws: Laws::None,
            ensures: Vec::new(),
            reaches: Reaches::Lent,
            returns: Returns::Unstated,
            cost: None,
        },
        instances: Instances {
            concrete: vec![DeclaredInstance {
                signature: instance,
                handler,
                admits: Task::Heavy,
                requires: Vec::new(),
                effect_bounds: Vec::new(),
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

/// A family type as the two patterns its casts are declared between: every
/// argument a variable, a type argument at the representation the member
/// gave it and at the uniform one, and an effect argument at its own in
/// both, since a cast converts the member and leaves the effect's Rust type
/// as it was.
struct FamilyPatterns {
    family: String,
    specialized: PolyTy,
    uniform: PolyTy,
    ty_vars: usize,
}

impl FamilyPatterns {
    fn of(i: &Interner, specialized: &PolyTy) -> Option<Self> {
        let TyTerm::UserDefined {
            id,
            type_args,
            effect_args,
            identity_args,
            region_params,
        } = specialized
        else {
            return None;
        };
        let mut b = PolyBuilder::new();
        let ty_vars: Vec<PolyTy> = type_args.iter().map(|_| b.fresh_ty_var()).collect();
        let effect_vars: Vec<EffectArg<Poly>> = effect_args
            .iter()
            .map(|arg| arg.with_effect(b.fresh_effect_var()))
            .collect();
        let identity_vars: Vec<IdentityTerm<Poly>> = identity_args
            .iter()
            .map(|_| b.fresh_identity_var())
            .collect();
        let pattern = |at: fn(&TypeArg<Poly>, PolyTy) -> TypeArg<Poly>| PolyTy::UserDefined {
            id: *id,
            type_args: type_args
                .iter()
                .zip(&ty_vars)
                .map(|(arg, var)| at(arg, var.clone()))
                .collect(),
            effect_args: effect_vars.clone(),
            identity_args: identity_vars.clone(),
            region_params: *region_params,
        };
        Some(FamilyPatterns {
            family: written(i, *id),
            specialized: pattern(|arg, var| match arg {
                // `#T` held whole: the member fills `T` with a Rust type
                // written out, `#` at every part (`family_member_written`).
                TypeArg::Specialized(_) => TypeArg::specialized(var),
                TypeArg::Uniform(_) => TypeArg::uniform(var),
                TypeArg::Open(repr, _) => TypeArg::Open(*repr, var),
            }),
            uniform: pattern(|_, var| TypeArg::uniform(var)),
            ty_vars: ty_vars.len(),
        })
    }
}

/// A cast a declaration wrote from `X<#T>` to `X<T>` (RFC-0068 rule 8) is one
/// instance of the family's `X::erase`, the form `family_casts` declares, so
/// that every element type's cast is one rule and one name.
fn as_family_erase<R>(i: &Interner, declared: ExternFn<R>) -> ExternFn<R>
where
    R: Runtime,
{
    let ExternFn { decl, instances } = declared;
    let Some(written) = WrittenErase::of(i, &decl) else {
        return ExternFn { decl, instances };
    };
    let handler = match instances {
        Instances {
            concrete,
            generic: Some(handler),
        } if concrete.is_empty() => handler,
        instances => return ExternFn { decl, instances },
    };
    ExternFn {
        decl: FnDecl {
            qref: QualifiedRef::qualified(i.intern(&written.patterns.family), i.intern("erase")),
            ty: PolyTy::Fn {
                params: vec![ParamTerm::<Poly>::new(
                    written.value,
                    written.patterns.specialized,
                )],
                ret: Box::new(written.patterns.uniform),
                captures: vec![],
                effect: written.effect,
                flows: acvus_mir::ty::Flows::Every.into(),
            },
            bounds: vec![TyVarBound::Any; written.patterns.ty_vars],
            effect_bounds: vec![],
            coercion: Some(Coercion::Cast),
            instance_of: None,
            requires: Vec::new(),
            names: decl.names,
            laws: decl.laws,
            ensures: decl.ensures,
            reaches: decl.reaches,
            returns: decl.returns,
            cost: decl.cost,
        },
        instances: Instances {
            concrete: vec![DeclaredInstance {
                signature: decl.ty,
                handler,
                admits: Task::Heavy,
                requires: Vec::new(),
                effect_bounds: Vec::new(),
            }],
            generic: None,
        },
    }
}

/// What a cast declaration wrote, where its two sides are one family at two
/// representations.
struct WrittenErase {
    value: acvus_utils::Astr,
    effect: EffectTerm<Poly>,
    patterns: FamilyPatterns,
}

impl WrittenErase {
    fn of(i: &Interner, decl: &FnDecl) -> Option<Self> {
        let (
            Some(Coercion::Cast),
            PolyTy::Fn {
                params,
                ret,
                effect,
                ..
            },
        ) = (&decl.coercion, &decl.ty)
        else {
            return None;
        };
        let [value] = params.as_slice() else {
            return None;
        };
        let (TyTerm::UserDefined { id: from, .. }, TyTerm::UserDefined { id: to, .. }) =
            (&value.ty, &**ret)
        else {
            return None;
        };
        if from != to || value.ty == **ret {
            return None;
        }
        Some(WrittenErase {
            value: value.name,
            effect: effect.clone(),
            patterns: FamilyPatterns::of(i, &value.ty)?,
        })
    }
}

pub trait ExternTypeDecl {
    /// The type at `()` for each of its variables and `TypesOnly` for its
    /// runtime, whose `TypeId` is the identity behind its name.
    type DeclarationForm: 'static;

    /// The type's lifetime parameters: `type_decl`'s `region_params`, and
    /// the count every `TyTerm::UserDefined` of the type carries.
    const REGION_PARAMS: usize;

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
    /// Two functions, two signatures, or two Rust types under one name.
    DuplicateName {
        name: String,
    },
    /// One Rust type under two names.
    TwoNames {
        rust: &'static str,
        a: String,
        b: String,
    },
    /// A declaration naming an extension type no registry declares.
    UndeclaredType {
        declaration: String,
        ty: String,
    },
    UnknownSignature {
        instance: QualifiedRef,
        signature: QualifiedRef,
    },
    InstanceMismatch {
        instance: QualifiedRef,
        signature: QualifiedRef,
    },
    /// Two instances of one signature whose types unify (RFC-0019).
    DuplicateInstance {
        signature: QualifiedRef,
        ty: PolyTy,
    },
    CastShape {
        function: QualifiedRef,
        reason: &'static str,
    },
    /// A family cast at a member whose `#` argument has a part a run-time
    /// instantiation fills (RFC-0041): the family's pattern `F<#T>` holds
    /// `T` whole, `#` at every part, and a member such as `Vec<#(#f64, U)>`
    /// has the uniform part `U`, which that pattern would call `#`.
    FamilyMemberNotWritten {
        function: String,
        member: PolyTy,
    },
    /// A declaration marked the machine's view whose type is not one.
    ViewShape {
        function: QualifiedRef,
    },
    /// A declaration requiring a signature no registry declares
    /// (RFC-0067 rule 1).
    RequiredSignatureUnknown {
        function: QualifiedRef,
        signature: QualifiedRef,
    },
    /// A requirement on a signature one of whose instances has no mono
    /// glue: the requirement resolves to a plain `fn` at whichever type the
    /// call settles on, and for that instance there is none (RFC-0067
    /// rule 8).
    RequiredInstanceWithoutGlue {
        signature: String,
        instance: String,
        ty: PolyTy,
    },
    HashWithoutEq {
        instance: String,
        ty: PolyTy,
    },
    LawNamesUnfitExtern {
        function: String,
        law: &'static str,
        named: String,
        expected: &'static str,
    },
    LawOnUnfitSignature {
        function: String,
        law: &'static str,
    },
    IdentityNotOfResultType {
        function: String,
        constant: acvus_ast::Literal,
        ty: PolyTy,
    },
    /// A handler that runs above the task its declaration names (RFC-0046).
    HandlerTask {
        function: String,
        declared: Task,
        handler: Task,
    },
    /// `total` on an instance with a parameter holding a function value,
    /// whose calls the declaration cannot promise (RFC-0082 rule 9).
    TotalOverFunctionArgument {
        function: String,
        ty: PolyTy,
    },
    /// `total` on an instance that requires an instance of a signature,
    /// whichever instance a call resolves it to (RFC-0082 rule 9).
    TotalOverRequiredInstance {
        function: String,
        ty: PolyTy,
        signature: String,
    },
}

impl fmt::Display for CombineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateName { name } => write!(f, "name `{name}` is declared twice"),
            Self::TwoNames { rust, a, b } => {
                write!(
                    f,
                    "the Rust type {rust} is declared as both `{a}` and `{b}`"
                )
            }
            Self::UndeclaredType { declaration, ty } => write!(
                f,
                "{declaration} names the type `{ty}`, which no registry declares: list it in \
                 a registry's `types`"
            ),
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
            Self::FamilyMemberNotWritten { function, member } => write!(
                f,
                "{function} is a family cast, and its instance {member:?} is not of the \
                 family's pattern: a member is a Rust type written out, with no part a \
                 type variable or an erased value fills"
            ),
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
            Self::HashWithoutEq { instance, ty } => write!(
                f,
                "{instance} declares core::hash at {ty:?}, and no registry declares core::eq \
                 there: a hash is bound to the equality it agrees with, so declare core::eq at \
                 {ty:?} beside it"
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
            Self::LawNamesUnfitExtern {
                function,
                law,
                named,
                expected,
            } => write!(
                f,
                "{function} declares `{law}` naming `{named}`, which must be {expected}"
            ),
            Self::LawOnUnfitSignature { function, law } => write!(
                f,
                "{function} declares `{law}`, which its type is not of the shape to state"
            ),
            Self::IdentityNotOfResultType {
                function,
                constant,
                ty,
            } => write!(
                f,
                "{function} declares the identity {constant:?}, which is not a value of its \
                 result type {ty:?}"
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
            Self::TotalOverFunctionArgument { function, ty } => write!(
                f,
                "{function} declares `total` at {ty:?}, which takes a function value: a call \
                 of it runs that function too, and no declaration of {function} can promise \
                 that function never traps. Declare `returns` instead, or nothing."
            ),
            Self::TotalOverRequiredInstance {
                function,
                ty,
                signature,
            } => write!(
                f,
                "{function} declares `total` at {ty:?}, which requires an instance of \
                 {signature}: a call of it runs whichever instance the call resolves, and no \
                 declaration of {function} can promise that instance never traps. Declare \
                 `returns` instead, or nothing."
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

fn duplicate(i: &Interner, q: QualifiedRef) -> CombineError {
    CombineError::DuplicateName {
        name: written(i, q),
    }
}

/// Every type name the registries give, held to one Rust type each way.
#[derive(Default)]
struct TypeNames {
    by_name: FxHashMap<QualifiedRef, Named>,
    by_rust: FxHashMap<DeclarationForm, QualifiedRef>,
    declared: FxHashSet<QualifiedRef>,
}

impl TypeNames {
    /// Binds `named`, answering whether the name is new.
    fn bind(&mut self, i: &Interner, named: Named) -> Result<bool, CombineError> {
        if let Some(bound) = self.by_name.get(&named.qref) {
            return match bound.rust == named.rust {
                true => Ok(false),
                false => Err(duplicate(i, named.qref)),
            };
        }
        if let Some(other) = self.by_rust.get(&named.rust) {
            return Err(CombineError::TwoNames {
                rust: named.rust_path,
                a: written(i, *other),
                b: written(i, named.qref),
            });
        }
        self.by_name.insert(named.qref, named);
        self.by_rust.insert(named.rust, named.qref);
        Ok(true)
    }

    /// Binds a type a registry's `types` lists, answering whether its
    /// declaration is the first under its name.
    fn declare(&mut self, i: &Interner, named: Named) -> Result<bool, CombineError> {
        let fresh = self.bind(i, named)?;
        self.declared.insert(named.qref);
        Ok(fresh)
    }

    /// Binds the names `declaration`'s types reach; an extension type among
    /// them is one a registry declares.
    fn reach(
        &mut self,
        i: &Interner,
        declaration: QualifiedRef,
        reached: &[Named],
    ) -> Result<(), CombineError> {
        for named in reached {
            self.bind(i, *named)?;
            if named.kind == NameKind::Extension && !self.declared.contains(&named.qref) {
                return Err(CombineError::UndeclaredType {
                    declaration: written(i, declaration),
                    ty: written(i, named.qref),
                });
            }
        }
        Ok(())
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

/// The mono glue of every instance of every required signature, numbered
/// as the compiler numbers that signature's instances (`Instances`): this
/// is what `prepare` asks to build an entry for a settled requirement
/// (RFC-0070 rule 2).
///
/// A row is dense: a signature one of whose instances has no glue is not a
/// row, and `Externs::combine` refuses a requirement on such a signature
/// (`CombineError::RequiredInstanceWithoutGlue`).
#[derive(Default)]
pub struct InstanceTable {
    by_signature: FxHashMap<QualifiedRef, Vec<InstanceRun>>,
}

impl<R> crate::handler::InstanceEntries<R> for InstanceTable
where
    R: Runtime,
{
    fn glue(&self, signature: QualifiedRef, instance: RequiredInstance) -> InstanceRun {
        self.by_signature[&signature][instance.0]
    }
}

/// One instance of one shared signature: the pattern it stands at, its
/// mono glue, and the task its body runs at.
pub struct InstanceAt {
    /// The declaration this instance came from, which is what a refusal
    /// naming it has to print.
    pub qref: QualifiedRef,
    pub ty: PolyTy,
    /// Obligation across artifacts: `#[extern_fn]` decides which
    /// declarations get a mono glue, and this is `None` for the rest.
    pub run: Option<InstanceRun>,
    /// The task this instance's own body runs at: the tightest of the
    /// handlers the declaration contributed, which for a declaration with a
    /// `sync =` twin is the twin's `Task::Sync`.
    pub task: Task,
}

/// The instances collected for one signature.
struct Collected<R: Runtime> {
    decl: SignatureDecl,
    instance_types: Vec<PolyTy>,
    instances: Vec<LawfulInstance<R>>,
    /// One per declared instance, which `instances` is not: a declaration
    /// with a `sync =` companion contributes two handlers and one type.
    entries: Vec<InstanceAt>,
    casts: Vec<CastRule>,
}

struct LawfulInstance<R: Runtime> {
    instance: DeclaredInstance<R>,
    laws: Laws,
    ensures: Vec<Postcondition>,
    reaches: Reaches,
    returns: Returns,
    cost: Option<u64>,
}

impl<R: Runtime> Externs<R> {
    pub fn combine(
        registries: Vec<Registry<R>>,
        interner: &Interner,
    ) -> Result<Self, CombineError> {
        let mut types = TypeRegistry::new();
        let mut type_names = TypeNames::default();
        let mut names: FxHashSet<QualifiedRef> = FxHashSet::default();
        let mut signatures: FxHashMap<QualifiedRef, Collected<R>> = FxHashMap::default();
        let mut plain: Vec<ExternFn<R>> = Vec::new();

        let contributions: Vec<Contribution<R>> = std::iter::once(crate::core::core_registry())
            .chain(registries)
            .map(|r| (r.factory)(interner))
            .collect();

        for c in &contributions {
            for declared in &c.manifest.types {
                if type_names.declare(interner, declared.named)? {
                    types
                        .register(declared.decl.clone())
                        .map_err(|DuplicateType(qref)| duplicate(interner, qref))?;
                }
            }
        }
        for c in &contributions {
            for sig in &c.manifest.signatures {
                if !names.insert(sig.qref) {
                    return Err(duplicate(interner, sig.qref));
                }
                type_names.reach(interner, sig.qref, &sig.names)?;
            }
            let mut in_contribution: FxHashSet<QualifiedRef> = FxHashSet::default();
            for decl in &c.manifest.fns {
                if !in_contribution.insert(decl.qref) {
                    return Err(duplicate(interner, decl.qref));
                }
                type_names.reach(interner, decl.qref, &decl.names)?;
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
        let mut required_signatures: FxHashSet<QualifiedRef> = FxHashSet::default();
        for (fns, mut instances) in plain_manifests {
            for decl in fns {
                let instances = instances
                    .remove(&decl.qref)
                    .unwrap_or_else(|| panic!("no handler for declared {:?}", decl.qref));
                match decl.instance_of {
                    Some(sig) => add_instance(
                        interner,
                        &mut signatures,
                        &mut required_signatures,
                        decl,
                        instances,
                        sig,
                    )?,
                    None => {
                        let ExternFn { decl, instances } =
                            as_family_erase(interner, ExternFn { decl, instances });
                        let declared = plain
                            .iter_mut()
                            .find(|f| same_family_cast(&f.decl, &f.instances, &decl, &instances));
                        match declared {
                            Some(existing) => existing.instances.add_concrete(instances.concrete),
                            None => {
                                if !names.insert(decl.qref) {
                                    return Err(duplicate(interner, decl.qref));
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
            let mut decl = decl;
            for required in &decl.requires {
                required_signatures.insert(required.signature);
                let Some(signature) = signatures.get(&required.signature) else {
                    return Err(CombineError::RequiredSignatureUnknown {
                        function: decl.qref,
                        signature: required.signature,
                    });
                };
                if let Some(PolyTy::Var(standing)) =
                    instance_type(&signature.decl.ty, &required.pattern)
                {
                    let bound = &mut decl.bounds[standing as usize];
                    *bound = meet(bound, &signature.instance_types);
                }
            }
            match decl.coercion {
                Some(Coercion::Cast) => {
                    family_member_written(interner, &decl, &instances)?;
                    types.register_cast(cast_rule(&decl)?)
                }
                Some(Coercion::View) => {
                    let viewed = view_of(&decl)?;
                    let referent_taken = match &decl.ty {
                        PolyTy::Fn { params, .. } => match params.first().map(|p| &p.ty) {
                            Some(TyTerm::Ref(_, referent)) => referent.ty(),
                            _ => std::borrow::Cow::Borrowed(&decl.ty),
                        },
                        other => std::borrow::Cow::Borrowed(other),
                    };
                    types.register_machine_view(decl.qref, viewed, &referent_taken);
                }
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
                    effect_bounds: decl.effect_bounds,
                    instances: instances.signatures(
                        &decl.laws,
                        &decl.ensures,
                        &decl.reaches,
                        decl.returns,
                        decl.cost,
                    ),
                    requires: decl
                        .requires
                        .iter()
                        .map(Requirement::signature_of)
                        .collect(),
                },
                ty: decl.ty,
            });
            handlers.insert(decl.qref, instances.into_handlers());
        }
        let mut collected: Vec<Collected<R>> = signatures.into_values().collect();
        collected.sort_by_key(|c| c.decl.qref);
        hash_beside_eq(interner, &collected)?;
        let mut instance_table = InstanceTable {
            by_signature: FxHashMap::default(),
        };
        for c in collected {
            if required_signatures.contains(&c.decl.qref)
                && let Some(bare) = c.entries.iter().find(|at| at.run.is_none())
            {
                return Err(CombineError::RequiredInstanceWithoutGlue {
                    signature: written(interner, c.decl.qref),
                    instance: written(interner, bare.qref),
                    ty: bare.ty.clone(),
                });
            }
            let row: Option<Vec<InstanceRun>> = c
                .instances
                .iter()
                .map(|arm| arm.instance.handler.instance())
                .collect();
            if let Some(row) = row {
                instance_table.by_signature.insert(c.decl.qref, row);
            }
            let mut bounds = c.decl.bounds;
            if let Some(first) = bounds.first_mut() {
                *first = meet(first, &c.instance_types);
            }
            for cast in c.casts {
                types.register_cast(cast);
            }
            let signatures = acvus_mir::ty::Instances {
                concrete: c
                    .instances
                    .iter()
                    .map(|arm| {
                        arm.instance
                            .signature_under(
                                arm.laws.clone(),
                                arm.ensures.clone(),
                                arm.reaches.clone(),
                                arm.returns,
                                arm.cost,
                            )
                    })
                    .collect(),
                generic: None,
            };
            let instances = Instances {
                concrete: c.instances.into_iter().map(|arm| arm.instance).collect(),
                generic: None,
            };
            functions.push(Function {
                qref: c.decl.qref,
                kind: FnKind::Extern {
                    bounds,
                    effect_bounds: Vec::new(),
                    instances: signatures,
                    requires: Vec::new(),
                },
                ty: c.decl.ty,
            });
            handlers.insert(c.decl.qref, instances.into_handlers());
        }
        let declared: FxHashMap<QualifiedRef, &Function> =
            functions.iter().map(|f| (f.qref, f)).collect();
        for function in &functions {
            let FnKind::Extern {
                instances,
                requires,
                ..
            } = &function.kind
            else {
                continue;
            };
            let concrete = instances
                .concrete
                .iter()
                .map(|at| (&at.ty, &at.laws, at.returns, at.requires.first()));
            let generic = instances
                .generic
                .as_ref()
                .map(|at| (&function.ty, &at.laws, at.returns, None));
            for (ty, laws, returns, own_requirement) in concrete.chain(generic) {
                if returns == Returns::Total && takes_a_function_value(ty) {
                    return Err(CombineError::TotalOverFunctionArgument {
                        function: written(interner, function.qref),
                        ty: ty.clone(),
                    });
                }
                if returns == Returns::Total
                    && let Some(required) = own_requirement.or(requires.first())
                {
                    return Err(CombineError::TotalOverRequiredInstance {
                        function: written(interner, function.qref),
                        ty: ty.clone(),
                        signature: written(interner, required.signature),
                    });
                }
                LawSite {
                    interner,
                    declared: &declared,
                    function: function.qref,
                    ty,
                }
                .resolve(laws)?;
            }
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

fn takes_a_function_value(ty: &PolyTy) -> bool {
    fn holds_one(ty: &PolyTy) -> bool {
        matches!(ty, TyTerm::Fn { .. }) || ty.children().iter().any(|child| holds_one(child))
    }
    let PolyTy::Fn { params, .. } = ty else {
        return false;
    };
    params.iter().any(|param| holds_one(&param.ty))
}

struct LawSite<'a> {
    interner: &'a Interner,
    declared: &'a FxHashMap<QualifiedRef, &'a Function>,
    function: QualifiedRef,
    ty: &'a PolyTy,
}

impl LawSite<'_> {
    fn resolve(&self, laws: &Laws) -> Result<(), CombineError> {
        let law = match laws {
            Laws::None => return Ok(()),
            Laws::Binary(_) => "law",
            Laws::Fold(_) => "fold",
        };
        let PolyTy::Fn { ret, .. } = self.ty else {
            return Err(self.unshaped(law));
        };
        if let Laws::Binary(BinaryLaws {
            identity: Some(Identity::Const(constant)),
            ..
        }) = laws
            && !is_value_of(constant, ret)
        {
            return Err(CombineError::IdentityNotOfResultType {
                function: written(self.interner, self.function),
                constant: constant.clone(),
                ty: (**ret).clone(),
            });
        }
        let resolved =
            acvus_mir::laws::resolve(laws, self.ty, |named| self.declared.get(&named).copied());
        match resolved {
            Ok(_) => Ok(()),
            Err(Unresolved::UnfitDeclaration) => Err(self.unshaped(law)),
            Err(Unresolved::NoFittingInstance { role, named }) => {
                let (law, expected) = match role {
                    LawRole::Identity => (
                        "identity",
                        "a registered extern with one instance of no argument returning the \
                         result type",
                    ),
                    LawRole::FoldCombine => (
                        "fold combine",
                        "a registered extern with one instance `g(s: &mut S, part: S)` over the \
                         folded state `S`",
                    ),
                    LawRole::FoldIdentity => (
                        "fold identity",
                        "a registered extern with one instance of no argument returning the \
                         folded state",
                    ),
                };
                Err(CombineError::LawNamesUnfitExtern {
                    function: written(self.interner, self.function),
                    law,
                    named: written(self.interner, named),
                    expected,
                })
            }
        }
    }

    fn unshaped(&self, law: &'static str) -> CombineError {
        CombineError::LawOnUnfitSignature {
            function: written(self.interner, self.function),
            law,
        }
    }
}

fn is_value_of(constant: &acvus_ast::Literal, ty: &PolyTy) -> bool {
    use acvus_ast::Literal;
    match (constant, ty) {
        (Literal::Int(value), PolyTy::Int(int)) => int.holds(*value),
        (Literal::Float(_), PolyTy::Float) | (Literal::Bool(_), PolyTy::Bool) => true,
        (Literal::String(_), PolyTy::String) => true,
        _ => false,
    }
}

fn hash_beside_eq<R>(interner: &Interner, collected: &[Collected<R>]) -> Result<(), CombineError>
where
    R: Runtime,
{
    let of = |qref: QualifiedRef| collected.iter().find(|c| c.decl.qref == qref);
    let Some(hash) = of(<crate::core::hash as SharedSignature>::qref(interner)) else {
        return Ok(());
    };
    let eq_types: &[PolyTy] = of(<crate::core::eq as SharedSignature>::qref(interner))
        .map_or(&[], |eq| eq.instance_types.as_slice());
    for at in &hash.entries {
        if !eq_types
            .iter()
            .any(|eq| unify_patterns(eq, &at.ty).is_some())
        {
            return Err(CombineError::HashWithoutEq {
                instance: written(interner, at.qref),
                ty: at.ty.clone(),
            });
        }
    }
    Ok(())
}

/// The bound a signature's receiver keeps: its declared bound met with the
/// patterns its instances stand at.
fn meet(bound: &TyVarBound, allowed: &[PolyTy]) -> TyVarBound {
    bound
        .meet(&TyVarBound::one_of(allowed.to_vec()))
        .unwrap_or(TyVarBound::one_of(Vec::new()))
}

fn add_instance<R: Runtime>(
    i: &Interner,
    signatures: &mut FxHashMap<QualifiedRef, Collected<R>>,
    required_signatures: &mut FxHashSet<QualifiedRef>,
    decl: FnDecl,
    instances: Instances<R>,
    sig: QualifiedRef,
) -> Result<(), CombineError> {
    let requires: Vec<RequirementSig> = decl
        .requires
        .iter()
        .map(Requirement::signature_of)
        .collect();
    for required in &decl.requires {
        if !signatures.contains_key(&required.signature) {
            return Err(CombineError::RequiredSignatureUnknown {
                function: decl.qref,
                signature: required.signature,
            });
        }
        required_signatures.insert(required.signature);
    }
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
    let signature = bind_chosen(&collected.decl.ty, &decl.ty, &collected.decl.chosen);
    if !matches_pattern(&decl.ty, &signature) {
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
    let admitted = at_declared_type(&decl.ty, instances, &requires, &decl.effect_bounds)
        .ok_or_else(mismatch)?;
    for instance in &admitted {
        ceiling_admits(i, decl.qref, &instance.signature, &instance.handler)?;
    }
    if decl.coercion == Some(Coercion::Cast) {
        let mut rule = cast_rule(&decl)?;
        rule.fn_ref = sig;
        collected.casts.push(rule);
    }
    collected.entries.push(InstanceAt {
        qref: decl.qref,
        ty: ty.clone(),
        run: admitted.iter().find_map(|i| i.handler.instance()),
        task: admitted
            .iter()
            .map(|i| i.handler.task())
            .fold(Task::Heavy, Task::meet),
    });
    collected.instance_types.push(ty);
    collected
        .instances
        .extend(admitted.into_iter().map(|instance| LawfulInstance {
            instance,
            laws: decl.laws.clone(),
            ensures: decl.ensures.clone(),
            reaches: decl.reaches.clone(),
            returns: decl.returns,
            cost: decl.cost,
        }));
    Ok(())
}

/// The instances one declaration contributes to the signature it names.
/// A declaration whose handler is generic has none of its own, and the one
/// built here is where its requirements and effect bounds are written;
/// `#[extern_fn]` writes them on the instances it builds itself.
fn at_declared_type<R>(
    declared: &PolyTy,
    instances: Instances<R>,
    requires: &[RequirementSig],
    effect_bounds: &[EffectVarBound],
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
            requires: requires.to_vec(),
            effect_bounds: effect_bounds.to_vec(),
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

/// The instance's type facing the signature's first `Var(0)`, and `None`
/// where the signature names that variable only under a head with no arm
/// here — a function type, an option, a result, a handle, an object, an
/// enum. Descending those is not built, and `add_instance`
/// refuses such a signature with `InstanceMismatch`.
fn instance_at_first_var(signature: &PolyTy, instance: &PolyTy) -> Option<PolyTy> {
    match (signature, instance) {
        (PolyTy::Var(0), t) => Some(t.clone()),
        (PolyTy::Ref(_, s), PolyTy::Ref(_, i)) => instance_at_first_var(&s.ty(), &i.ty()),
        (PolyTy::Array(s, _), PolyTy::Array(i, _)) => instance_at_first_var(s, i),
        (PolyTy::Slice(s), PolyTy::Slice(i)) => instance_at_first_var(s, i),
        (PolyTy::UserDefined { type_args: sa, .. }, PolyTy::UserDefined { type_args: ia, .. }) => {
            sa.iter()
                .zip(ia)
                .find_map(|(s, i)| instance_at_first_var(&s.ty(), &i.ty()))
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

/// Every instance of a cast has the shape of the cast's declared type. For a
/// family cast, whose declared type holds each `#` argument whole as
/// `F<#T>`, this is the invariant `Held` rests on: a member fills `T` with a
/// Rust type written out, and one with a part an `Owned`, `Erased` or `Nth`
/// fills is refused here, where the registry is built. It is not a bound
/// on the member list: the members (`mono_member!`) are written scalars
/// already, and the part a variable fills comes from the family type the
/// declaration writes around its member, as `Vec<(T, U)>` would.
fn family_member_written<R>(
    i: &Interner,
    decl: &FnDecl,
    instances: &Instances<R>,
) -> Result<(), CombineError>
where
    R: Runtime,
{
    match instances
        .concrete
        .iter()
        .find(|instance| !matches_pattern(&instance.signature, &decl.ty))
    {
        Some(instance) => Err(CombineError::FamilyMemberNotWritten {
            function: written(i, decl.qref),
            member: instance.signature.clone(),
        }),
        None => Ok(()),
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
