use std::hash::Hash;

use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::{FxHashMap, FxHashSet};

use super::casts::{Base, CastTree, Head, VerifiedExternCast};
use super::conversions::{ConversionRule, VerifiedConversion};
use super::labels::{LabelledSignature, Lifetime, SegmentLabel};
use super::instance_fit::is_instance_of;
use super::{CastRule, Core, Declarations, RawDeclarations, Requirement, Scheme, TyVarBound, UserTypeDecl};
use crate::flows::{Segment, segments};
use crate::ty::{
    Alignment, Declared, Flow, FlowEnd, FlowTerm, Flows, Holds, Labelled, LifetimesOf, Mutability, ObjectTy,
    Poly, TyId, TyKind, Types,
};

/// Why a set of declarations is not taken in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refused {
    /// A declaration carries an array type (extern-fn §1).
    ArrayType { declaration: QualifiedRef },
    /// Two declared steps up from one head (types §3).
    TwoStepsUp { head: Head, via: [QualifiedRef; 2] },
    /// The steps up from `head` return to it (types §3).
    CastCycle { head: Head },
    /// A required instance's types are not smaller than the declaration's
    /// own (extern-fn §6).
    RequirementNotSmaller { declaration: QualifiedRef, signature: QualifiedRef },
    /// An end's labels are not one per segment of its declared type, or
    /// name a lifetime the signature does not have.
    LabelsDoNotFit { declaration: QualifiedRef },
    /// A declared cast, a view instance or a declared conversion whose
    /// `from` or `to` has no head: a type variable, `!`, the order token, a
    /// started call, or a reference to one of those. A tree step joins two
    /// heads (types §3), and so does a conversion (types §7).
    CastWithoutHead { via: QualifiedRef },
    /// An instance whose type is not its shared signature's type at some
    /// choice of the signature's variables (extern-fn §6).
    InstanceNotOfSignature { signature: QualifiedRef, instance: usize },
    /// A signature whose parameter names are not one per parameter of its
    /// function type (extern-fn §1).
    ParamNamesDoNotFit { function: QualifiedRef },
    /// Two declarations under one qualified name: reach could not tell
    /// them apart (types §6).
    DuplicateFunction { id: QualifiedRef },
    /// Two declarations of one context, which is one value of one fixed
    /// type (context).
    DuplicateContext { id: QualifiedRef },
    /// Two declarations of one input (programs §4).
    DuplicateInput { name: acvus_utils::Astr },
    /// A type the declarations write names an extern-defined type no
    /// declaration gives (extern-fn §8).
    UndeclaredType { written: Written, id: QualifiedRef },
    /// A declared type holds `Handle` or `Order`, which only the compiler's
    /// IR writes (effects §3, §6).
    InternalType { written: Written },
    /// A declared type holds a reference whose target is a reference
    /// (types §4).
    ReferenceToReference { written: Written },
    /// A scheme whose bounds do not give each of its type variables exactly
    /// one bound (extern-fn §1: a type parameter carries one bound).
    BoundsDoNotFit { function: QualifiedRef },
    /// Two types under one name.
    DuplicateType { id: QualifiedRef },
    /// A type with more than one identity parameter. Not stated in
    /// rfc_compiler; carried from the registry's rule until it is.
    TwoIdentities { id: QualifiedRef },
}

/// Where a declaration writes a type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Written {
    Function(QualifiedRef),
    Context(QualifiedRef),
    Input(acvus_utils::Astr),
    Cast(QualifiedRef),
    Type(QualifiedRef),
}

/// Build the verified set, or every refusal the set earns.
pub(super) fn verify(raw: RawDeclarations) -> Result<Declarations, Vec<Refused>> {
    let RawDeclarations {
        types,
        functions,
        user_types,
        casts,
        conversions,
        contexts,
        inputs,
        loop_markers,
        core,
    } = raw;
    let mut refused = Vec::new();

    let user_types = extern_types(user_types, &mut refused);
    let unreadable = refuse_unwritable_types(
        &types,
        DeclaredTypes {
            functions: &functions,
            user_types: &user_types,
            casts: &casts,
            conversions: &conversions,
            contexts: &contexts,
            inputs: &inputs,
        },
        &mut refused,
    );
    refuse_array_types(&types, &functions, &mut refused);
    refuse_unending_requirements(&types, &functions, &mut refused);
    refuse_unfit_bounds(&types, &functions, &mut refused);
    refuse_unfit_param_names(&types, &functions, &mut refused);
    let declared_order = first_of_each(functions.iter().map(|(id, _)| *id));
    let overloads = overloads(&declared_order);
    let functions = first_stands(functions, &mut refused, |id| Refused::DuplicateFunction { id });
    let contexts = first_stands(contexts, &mut refused, |id| Refused::DuplicateContext { id });
    let inputs = first_stands(inputs, &mut refused, |name| Refused::DuplicateInput { name });
    let misfits = refuse_misfit_instances(&types, &declared_order, &functions, &mut refused);
    let casts = cast_tree(&types, &casts, &functions, &misfits, &core, &mut refused);
    let conversions = conversions_by_head(&types, conversions, &mut refused);

    let mut decls = Declarations {
        types,
        functions,
        overloads,
        user_types,
        casts,
        conversions,
        contexts,
        inputs,
        loop_markers,
        core,
    };
    let lifetimes_readable: Vec<QualifiedRef> = declared_order
        .into_iter()
        .filter(|id| !unreadable.contains(id))
        .collect();
    take_in_lifetimes(&mut decls, &lifetimes_readable, &mut refused);

    match refused.is_empty() {
        true => Ok(decls),
        false => Err(refused),
    }
}

/// Each entry by its key. The first entry of a key stands; every later
/// one is refused by `duplicate`.
fn first_stands<K, V>(
    entries: Vec<(K, V)>,
    refused: &mut Vec<Refused>,
    duplicate: impl Fn(K) -> Refused,
) -> FxHashMap<K, V>
where
    K: Copy + Eq + Hash,
{
    let mut by_key = FxHashMap::default();
    for (key, value) in entries {
        match by_key.contains_key(&key) {
            true => refused.push(duplicate(key)),
            false => {
                by_key.insert(key, value);
            }
        }
    }
    by_key
}

/// `ids` in order, each once.
fn first_of_each(ids: impl Iterator<Item = QualifiedRef>) -> Vec<QualifiedRef> {
    let mut seen = FxHashSet::default();
    ids.filter(|id| seen.insert(*id)).collect()
}

/// The extern types by id (extern-fn §8). The first declaration of an id
/// stands; every later one is refused.
fn extern_types(user_types: Vec<UserTypeDecl>, refused: &mut Vec<Refused>) -> FxHashMap<QualifiedRef, UserTypeDecl> {
    let mut by_id: FxHashMap<QualifiedRef, UserTypeDecl> = FxHashMap::default();
    for decl in user_types {
        if decl.identity_params > 1 {
            refused.push(Refused::TwoIdentities { id: decl.id });
        }
        match by_id.contains_key(&decl.id) {
            true => refused.push(Refused::DuplicateType { id: decl.id }),
            false => {
                by_id.insert(decl.id, decl);
            }
        }
    }
    by_id
}

/// Every bare name with every declaration of that name, in any namespace,
/// in declaration order (extern-fn §5, types §6).
fn overloads(declared_order: &[QualifiedRef]) -> FxHashMap<Astr, Box<[QualifiedRef]>> {
    let mut by_name: FxHashMap<Astr, Vec<QualifiedRef>> = FxHashMap::default();
    for id in declared_order {
        by_name.entry(id.name).or_default().push(*id);
    }
    by_name
        .into_iter()
        .map(|(name, ids)| (name, ids.into_boxed_slice()))
        .collect()
}

/// Every type a declaration's signature is written at: the scheme's own,
/// then each concrete instance's.
fn signature_types(scheme: &Scheme) -> impl Iterator<Item = TyId<Poly>> + '_ {
    std::iter::once(scheme.ty).chain(
        scheme
            .instances
            .iter()
            .flat_map(|instances| instances.concrete.iter().map(|instance| instance.ty)),
    )
}

/// extern-fn §1: no declaration carries an array type.
fn refuse_array_types(types: &Types<Poly>, functions: &[(QualifiedRef, Scheme)], refused: &mut Vec<Refused>) {
    let mut seen = FxHashSet::default();
    for (id, scheme) in functions {
        if signature_types(scheme).any(|ty| holds_array(types, ty, &mut seen)) {
            refused.push(Refused::ArrayType { declaration: *id });
        }
    }
}

/// Whether `ty` holds an array anywhere. `clear` holds the types already
/// found to hold none, so a shared part is read once.
fn holds_array(types: &Types<Poly>, ty: TyId<Poly>, clear: &mut FxHashSet<TyId<Poly>>) -> bool {
    acvus_utils::grow(|| {
        if clear.contains(&ty) {
            return false;
        }
        let kind = types.kind(ty);
        let mut found = matches!(kind, TyKind::Array(..));
        kind.for_each_child(|child| found = found || holds_array(types, child, clear));
        if !found {
            clear.insert(ty);
        }
        found
    })
}

/// extern-fn §1: a type parameter carries one bound. A scheme's type
/// variables are `0..n` for its `n` bounds, each one mentioned by its type.
fn refuse_unfit_bounds(types: &Types<Poly>, functions: &[(QualifiedRef, Scheme)], refused: &mut Vec<Refused>) {
    for (function, scheme) in functions {
        let mut variables = FxHashSet::default();
        type_variables(types, scheme.ty, &mut FxHashSet::default(), &mut variables);
        let bounded = scheme.bounds.len();
        let one_per_variable = variables.len() == bounded
            && variables
                .iter()
                .all(|var| usize::try_from(*var).is_ok_and(|at| at < bounded));
        if !one_per_variable {
            refused.push(Refused::BoundsDoNotFit { function: *function });
        }
    }
}

/// extern-fn §1: a declaration names its parameters, one name per
/// parameter of each signature's function type. A signature that is no
/// function type has no parameter to name, so it fits no names.
fn refuse_unfit_param_names(types: &Types<Poly>, functions: &[(QualifiedRef, Scheme)], refused: &mut Vec<Refused>) {
    let fits = |ty: TyId<Poly>, names: &[Astr]| match types.kind(ty) {
        TyKind::Fn { params, .. } => params.len() == names.len(),
        _ => false,
    };
    for (function, scheme) in functions {
        let concrete = scheme.instances.iter().flat_map(|instances| instances.concrete.iter());
        let every_fits = fits(scheme.ty, &scheme.param_names)
            && concrete.into_iter().all(|instance| fits(instance.ty, &instance.param_names));
        if !every_fits {
            refused.push(Refused::ParamNamesDoNotFit { function: *function });
        }
    }
}

/// Every type variable `ty` mentions, into `variables`.
fn type_variables(
    types: &Types<Poly>,
    ty: TyId<Poly>,
    seen: &mut FxHashSet<TyId<Poly>>,
    variables: &mut FxHashSet<u32>,
) {
    acvus_utils::grow(|| {
        if !seen.insert(ty) {
            return;
        }
        let kind = types.kind(ty);
        if let TyKind::Var(var) = kind {
            variables.insert(*var);
        }
        kind.for_each_child(|child| type_variables(types, child, seen, variables));
    })
}

/// extern-fn §6: a required instance's types are smaller than the
/// declaration's own, so requirements end.
fn refuse_unending_requirements(
    types: &Types<Poly>,
    functions: &[(QualifiedRef, Scheme)],
    refused: &mut Vec<Refused>,
) {
    let mut sizes = TreeSizes {
        types,
        known: FxHashMap::default(),
    };
    for (id, scheme) in functions {
        sizes.refuse_not_smaller(*id, scheme.ty, &scheme.requires, refused);
        for instance in scheme.instances.iter().flat_map(|instances| instances.concrete.iter()) {
            sizes.refuse_not_smaller(*id, instance.ty, &instance.requires, refused);
        }
    }
}

/// The number of `TyKind` nodes of a type read as a tree, saturating.
struct TreeSizes<'t> {
    types: &'t Types<Poly>,
    known: FxHashMap<TyId<Poly>, usize>,
}

impl TreeSizes<'_> {
    /// Refuse each of `requires`, stated by `declaration` at `ty`, whose
    /// pattern is not smaller than `ty`.
    fn refuse_not_smaller(
        &mut self,
        declaration: QualifiedRef,
        ty: TyId<Poly>,
        requires: &[Requirement],
        refused: &mut Vec<Refused>,
    ) {
        let bound = self.of(ty);
        for requirement in requires {
            if self.of(requirement.pattern) >= bound {
                refused.push(Refused::RequirementNotSmaller {
                    declaration,
                    signature: requirement.signature,
                });
            }
        }
    }

    fn of(&mut self, ty: TyId<Poly>) -> usize {
        acvus_utils::grow(|| {
            if let Some(size) = self.known.get(&ty) {
                return *size;
            }
            let types = self.types;
            let mut size: usize = 1;
            types
                .kind(ty)
                .for_each_child(|child| size = size.saturating_add(self.of(child)));
            self.known.insert(ty, size);
            size
        })
    }
}

/// Every type the declarations write, by what writes it.
struct DeclaredTypes<'d> {
    functions: &'d [(QualifiedRef, Scheme)],
    user_types: &'d FxHashMap<QualifiedRef, UserTypeDecl>,
    casts: &'d [CastRule],
    conversions: &'d [ConversionRule],
    contexts: &'d [(QualifiedRef, TyId<Poly>)],
    inputs: &'d [(Astr, TyId<Poly>)],
}

/// The types a bound names.
fn bound_types(bound: &TyVarBound) -> &[TyId<Poly>] {
    match bound {
        TyVarBound::OneOf(shapes) => shapes,
        TyVarBound::Any | TyVarBound::Integer { .. } | TyVarBound::Settled => &[],
    }
}

/// Every type a scheme writes: its signature, its instances', every
/// requirement's pattern and every bound's shapes.
fn scheme_types(scheme: &Scheme) -> impl Iterator<Item = TyId<Poly>> + '_ {
    let concrete = scheme.instances.iter().flat_map(|instances| instances.concrete.iter());
    signature_types(scheme)
        .chain(scheme.requires.iter().map(|requirement| requirement.pattern))
        .chain(
            concrete
                .flat_map(|instance| instance.requires.iter())
                .map(|requirement| requirement.pattern),
        )
        .chain(scheme.bounds.iter().flat_map(bound_types).copied())
}

/// Every type the declarations write, each with where it is written.
fn written_types<'d>(declared: &DeclaredTypes<'d>) -> impl Iterator<Item = (Written, TyId<Poly>)> + 'd {
    let functions = declared
        .functions
        .iter()
        .flat_map(|(id, scheme)| scheme_types(scheme).map(|ty| (Written::Function(*id), ty)));
    let user_types = declared.user_types.values().flat_map(|decl| {
        decl.type_params
            .iter()
            .flat_map(bound_types)
            .map(|ty| (Written::Type(decl.id), *ty))
    });
    let casts = declared
        .casts
        .iter()
        .flat_map(|rule| [(Written::Cast(rule.via), rule.from), (Written::Cast(rule.via), rule.to)]);
    let conversions = declared
        .conversions
        .iter()
        .flat_map(|rule| [(Written::Cast(rule.via), rule.from), (Written::Cast(rule.via), rule.to)]);
    let contexts = declared.contexts.iter().map(|(id, ty)| (Written::Context(*id), *ty));
    let inputs = declared.inputs.iter().map(|(name, ty)| (Written::Input(*name), *ty));
    functions
        .chain(user_types)
        .chain(casts)
        .chain(conversions)
        .chain(contexts)
        .chain(inputs)
}

/// extern-fn §8: a type term names only declared types; effects §3, §6:
/// `Order` and `Handle` are written by the compiler's IR alone; types §4: a
/// reference's target is never a reference. Every type the declarations
/// write is held to all three. Returns the functions that break any, whose
/// segments are not read.
fn refuse_unwritable_types(
    types: &Types<Poly>,
    declared: DeclaredTypes<'_>,
    refused: &mut Vec<Refused>,
) -> FxHashSet<QualifiedRef> {
    let mut unreadable = FxHashSet::default();
    let mut reported: Vec<Refused> = Vec::new();
    for (written, ty) in written_types(&declared) {
        let mut terms = TypeTerms::default();
        terms.read(types, ty);
        let internal = terms.internal.then_some(Refused::InternalType { written });
        let reference_to_reference = terms
            .reference_to_reference
            .then_some(Refused::ReferenceToReference { written });
        let undeclared = terms
            .named
            .into_iter()
            .filter(|id| !declared.user_types.contains_key(id))
            .map(|id| Refused::UndeclaredType { written, id });
        for refusal in internal.into_iter().chain(reference_to_reference).chain(undeclared) {
            if let Written::Function(function) = written {
                unreadable.insert(function);
            }
            if !reported.contains(&refusal) {
                reported.push(refusal.clone());
                refused.push(refusal);
            }
        }
    }
    unreadable
}

/// What a type term names: every extern-defined type, each once, whether
/// it holds a type only the IR writes, and whether it holds a reference to
/// a reference.
#[derive(Default)]
struct TypeTerms {
    seen: FxHashSet<TyId<Poly>>,
    named: Vec<QualifiedRef>,
    internal: bool,
    reference_to_reference: bool,
}

impl TypeTerms {
    fn read(&mut self, types: &Types<Poly>, ty: TyId<Poly>) {
        acvus_utils::grow(|| {
            if !self.seen.insert(ty) {
                return;
            }
            let kind = types.kind(ty);
            match kind {
                TyKind::UserDefined { id, .. } if !self.named.contains(id) => self.named.push(*id),
                TyKind::Order | TyKind::Handle(_) => self.internal = true,
                TyKind::Ref(_, target) if matches!(types.kind(target.ty()), TyKind::Ref(..)) => {
                    self.reference_to_reference = true
                }
                _ => {}
            }
            kind.for_each_child(|child| self.read(types, child));
        })
    }
}

/// extern-fn §6: every concrete instance of a shared signature is the
/// signature's type at some choice of its variables. Returns the misfits,
/// by function and instance.
fn refuse_misfit_instances(
    types: &Types<Poly>,
    declared_order: &[QualifiedRef],
    functions: &FxHashMap<QualifiedRef, Scheme>,
    refused: &mut Vec<Refused>,
) -> FxHashSet<(QualifiedRef, usize)> {
    let mut misfits = FxHashSet::default();
    for signature in declared_order {
        let Some(scheme) = functions.get(signature) else {
            continue;
        };
        for (instance, sig) in scheme.instances.iter().flat_map(|instances| instances.concrete.iter().enumerate()) {
            if !is_instance_of(types, scheme.ty, sig.ty) {
                misfits.insert((*signature, instance));
                refused.push(Refused::InstanceNotOfSignature {
                    signature: *signature,
                    instance,
                });
            }
        }
    }
    misfits
}

/// The head a tree step joins at `ty` (types §3), or `None` where there
/// is none: a type variable, `!`, the order token or a started call, or a
/// reference to one of those.
fn head(types: &Types<Poly>, ty: TyId<Poly>) -> Option<Head> {
    match types.kind(ty) {
        TyKind::Ref(mutability, target) => Some(Head::Ref(*mutability, base(types, target.ty())?)),
        _ => Some(Head::Base(base(types, ty)?)),
    }
}

/// # Panics
/// `ty` is a reference: `head` reads a reference as its own head, and
/// `cast_tree` leaves out every rule whose end is a reference to a
/// reference, which `refuse_unwritable_types` refuses.
fn base(types: &Types<Poly>, ty: TyId<Poly>) -> Option<Base> {
    Some(match types.kind(ty) {
        TyKind::Int(width) => Base::Int(*width),
        TyKind::Float => Base::Float,
        TyKind::Char => Base::Char,
        TyKind::String => Base::String,
        TyKind::Bool => Base::Bool,
        TyKind::Unit => Base::Unit,
        TyKind::Str => Base::Str,
        TyKind::Array(..) => Base::Array,
        TyKind::Slice(_) => Base::Slice,
        TyKind::Tuple(items) => Base::Tuple(items.len()),
        TyKind::Option(_) => Base::Option,
        TyKind::Result(..) => Base::Result,
        TyKind::Object(ObjectTy::Structural(..)) => Base::Object,
        TyKind::Object(ObjectTy::Declared(id, _)) => Base::Named(*id),
        TyKind::Enum { name, .. } => Base::Enum(*name),
        TyKind::Fn { .. } => Base::Fn,
        TyKind::UserDefined { id, .. } => Base::Named(*id),
        TyKind::Ref(..) => panic!(
            "a reference's target is never a reference (types §4): a rule with such an end is refused as \
             ReferenceToReference and left out of the tree"
        ),
        TyKind::Var(_) | TyKind::Never | TyKind::Order | TyKind::Handle(_) => return None,
    })
}

/// A step of the tree: a cast from one head up to another.
struct Step {
    from: Head,
    to: Head,
    rule: CastRule,
}

impl Step {
    /// `rule` as a step, or `CastWithoutHead` where an end has no head.
    fn of(types: &Types<Poly>, rule: CastRule) -> Result<Step, Refused> {
        match (head(types, rule.from), head(types, rule.to)) {
            (Some(from), Some(to)) => Ok(Step { from, to, rule }),
            (None, _) | (_, None) => Err(Refused::CastWithoutHead { via: rule.via }),
        }
    }
}

/// The declared casts and the views as one tree (types §3): every head
/// steps up at most once, and no path of steps returns to its start.
fn cast_tree(
    types: &Types<Poly>,
    casts: &[CastRule],
    functions: &FxHashMap<QualifiedRef, Scheme>,
    misfits: &FxHashSet<(QualifiedRef, usize)>,
    core: &Core,
    refused: &mut Vec<Refused>,
) -> CastTree {
    let rules = casts
        .iter()
        .cloned()
        .map(Ok)
        .chain(view_rules(types, functions, misfits, core))
        .filter(|rule| {
            !rule
                .as_ref()
                .is_ok_and(|rule| ends_in_reference_to_reference(types, [rule.from, rule.to]))
        });
    let mut steps = Vec::new();
    for rule in rules {
        match rule.and_then(|rule| Step::of(types, rule)) {
            Ok(step) => steps.push(step),
            Err(refusal) => refused.push(refusal),
        }
    }

    let mut tree = CastTree::default();
    let mut above: FxHashMap<Head, Head> = FxHashMap::default();
    let mut from_order = Vec::new();
    for Step { from, to, rule } in steps {
        match tree.up.get(&from) {
            Some(first) => refused.push(Refused::TwoStepsUp {
                head: from,
                via: [first.rule().via, rule.via],
            }),
            None => {
                tree.up.insert(from, VerifiedExternCast(rule));
                above.insert(from, to);
                from_order.push(from);
            }
        }
    }
    refuse_cycles(&above, &from_order, refused);
    tree
}

/// Whether a cast's or a conversion's `from` or `to` is a reference to a
/// reference, which is refused as `ReferenceToReference` and has no head
/// (types §4).
fn ends_in_reference_to_reference(types: &Types<Poly>, ends: [TyId<Poly>; 2]) -> bool {
    ends.into_iter().any(|end| match types.kind(end) {
        TyKind::Ref(_, target) => matches!(types.kind(target.ty()), TyKind::Ref(..)),
        _ => false,
    })
}

/// The declared conversions by the head they convert from (extern-fn §7).
/// Each joins two heads, as a tree step does, but none is a step: a head
/// may have several, and they form no tree (types §7).
fn conversions_by_head(
    types: &Types<Poly>,
    conversions: Vec<ConversionRule>,
    refused: &mut Vec<Refused>,
) -> FxHashMap<Head, Box<[VerifiedConversion]>> {
    let mut by_head: FxHashMap<Head, Vec<VerifiedConversion>> = FxHashMap::default();
    for rule in conversions {
        if ends_in_reference_to_reference(types, [rule.from, rule.to]) {
            continue;
        }
        match (head(types, rule.from), head(types, rule.to)) {
            (Some(from), Some(_)) => by_head.entry(from).or_default().push(VerifiedConversion(rule)),
            (None, _) | (_, None) => refused.push(Refused::CastWithoutHead { via: rule.via }),
        }
    }
    by_head
        .into_iter()
        .map(|(head, conversions)| (head, conversions.into_boxed_slice()))
        .collect()
}

/// Each instance of `core::as_slice`, `core::as_slice_mut` and
/// `core::as_str` as a cast from its parameter to its result (operators
/// §3, §5). A generic instance's container is a variable, which has no
/// head. An instance refused as not of its signature is left out.
///
/// # Panics
/// A fitting instance is not a function of one parameter. Every fitting
/// instance has its view signature's shape, and core declares each view
/// as a function of one parameter.
fn view_rules(
    types: &Types<Poly>,
    functions: &FxHashMap<QualifiedRef, Scheme>,
    misfits: &FxHashSet<(QualifiedRef, usize)>,
    core: &Core,
) -> Vec<Result<CastRule, Refused>> {
    let mut rules = Vec::new();
    for via in [core.as_slice, core.as_slice_mut, core.as_str] {
        let Some(instances) = functions.get(&via).and_then(|scheme| scheme.instances.as_ref()) else {
            continue;
        };
        let fitting = instances
            .concrete
            .iter()
            .enumerate()
            .filter(|(at, _)| !misfits.contains(&(via, *at)));
        for (_, instance) in fitting {
            let TyKind::Fn { params, ret, .. } = types.kind(instance.ty) else {
                panic!("a view instance is a function type, as core declares its view")
            };
            let [param] = &params[..] else {
                panic!("a view instance takes one parameter, as core declares its view")
            };
            rules.push(Ok(CastRule {
                from: *param,
                to: *ret,
                via,
            }));
        }
        if instances.generic.is_some() {
            rules.push(Err(Refused::CastWithoutHead { via }));
        }
    }
    rules
}

/// One `CastCycle` per cycle of `above`, at the first head of the cycle a
/// walk from `from_order` meets.
fn refuse_cycles(above: &FxHashMap<Head, Head>, from_order: &[Head], refused: &mut Vec<Refused>) {
    let mut finished: FxHashSet<Head> = FxHashSet::default();
    for start in from_order {
        let mut walk: Vec<Head> = Vec::new();
        let mut at = *start;
        loop {
            if finished.contains(&at) {
                break;
            }
            if walk.contains(&at) {
                refused.push(Refused::CastCycle { head: at });
                break;
            }
            walk.push(at);
            let Some(next) = above.get(&at) else {
                break;
            };
            at = *next;
        }
        finished.extend(walk);
    }
}

/// Every declared signature's lifetimes become its function type's flows
/// (ownership §5, extern-fn §4), and nothing keeps a lifetime after.
fn take_in_lifetimes(decls: &mut Declarations, declared_order: &[QualifiedRef], refused: &mut Vec<Refused>) {
    let mut rewrites: Vec<FlowsOf> = Vec::new();
    for function in declared_order {
        let Some(scheme) = decls.functions.get(function) else {
            continue;
        };
        for signature in declared_signatures(*function, scheme) {
            let TyKind::Fn { params, ret, .. } = decls.types.kind(signature.ty) else {
                if signature.lifetimes.is_some() {
                    refused.push(Refused::LabelsDoNotFit { declaration: *function });
                }
                continue;
            };
            let flows = match signature.lifetimes {
                None => Flows::Every,
                Some(lifetimes) => match labelled_flows(decls, signature.declared, params, *ret, lifetimes) {
                    Ok(flows) => flows,
                    Err(LabelsDoNotFit) => {
                        refused.push(Refused::LabelsDoNotFit { declaration: *function });
                        continue;
                    }
                },
            };
            rewrites.push(FlowsOf {
                declared: signature.declared,
                flows,
            });
        }
    }

    for FlowsOf { declared, flows } in rewrites {
        let scheme = decls
            .functions
            .get_mut(&declared.function)
            .expect("a rewrite names a function read above");
        let slot = match declared.instance {
            None => &mut scheme.ty,
            Some(at) => {
                &mut scheme
                    .instances
                    .as_mut()
                    .expect("a rewrite of an instance names a declaration with instances")
                    .concrete[at]
                    .ty
            }
        };
        *slot = with_flows(&mut decls.types, *slot, flows);
    }
    for scheme in decls.functions.values_mut() {
        let Some(instances) = &mut scheme.instances else {
            continue;
        };
        for instance in instances.concrete.iter_mut() {
            instance.facts.lifetimes = None;
        }
        if let Some(generic) = &mut instances.generic {
            generic.facts.lifetimes = None;
        }
    }
}

/// The flows a declared signature's function type takes.
struct FlowsOf {
    declared: Declared,
    flows: Flows,
}

/// One signature a declaration states, with the lifetimes stated for it.
struct DeclaredSignature<'s> {
    declared: Declared,
    ty: TyId<Poly>,
    lifetimes: Option<&'s LabelledSignature>,
}

/// The scheme's own signature, whose lifetimes are its generic instance's,
/// then each concrete instance's.
fn declared_signatures(function: QualifiedRef, scheme: &Scheme) -> impl Iterator<Item = DeclaredSignature<'_>> {
    let generic = scheme
        .instances
        .as_ref()
        .and_then(|instances| instances.generic.as_ref())
        .and_then(|generic| generic.facts.lifetimes.as_ref());
    let own = DeclaredSignature {
        declared: Declared {
            function,
            instance: None,
        },
        ty: scheme.ty,
        lifetimes: generic,
    };
    let concrete = scheme.instances.iter().flat_map(move |instances| {
        instances
            .concrete
            .iter()
            .enumerate()
            .map(move |(at, instance)| DeclaredSignature {
                declared: Declared {
                    function,
                    instance: Some(at),
                },
                ty: instance.ty,
                lifetimes: instance.facts.lifetimes.as_ref(),
            })
    });
    std::iter::once(own).chain(concrete)
}

/// `ty`, a function type, with its flows replaced by `flows`.
fn with_flows(types: &mut Types<Poly>, ty: TyId<Poly>, flows: Flows) -> TyId<Poly> {
    let mut kind = types.kind(ty).clone();
    let TyKind::Fn { flows: slot, .. } = &mut kind else {
        panic!("only a function type's flows are rewritten")
    };
    *slot = FlowTerm::Known(flows);
    types.intern(kind)
}

struct LabelsDoNotFit;

/// The flows a labelled signature states: each output segment takes every
/// input segment of one lifetime with it, of a lifetime that outlives it,
/// or of the same type variable.
fn labelled_flows(
    decls: &Declarations,
    declared: Declared,
    params: &[TyId<Poly>],
    ret: TyId<Poly>,
    signature: &LabelledSignature,
) -> Result<Flows, LabelsDoNotFit> {
    let ends = fitted_ends(decls, params, ret, signature)?;
    let outlives = outlives_closure(signature, &ends)?;
    let reaches = |from: SegmentLabel, to: SegmentLabel| match (from, to) {
        (SegmentLabel::Lifetime(longer), SegmentLabel::Lifetime(shorter)) => {
            longer == shorter || outlives.contains(&(longer, shorter))
        }
        (SegmentLabel::TypeVar(from), SegmentLabel::TypeVar(to)) => from == to,
        (SegmentLabel::Lifetime(_), SegmentLabel::TypeVar(_))
        | (SegmentLabel::TypeVar(_), SegmentLabel::Lifetime(_)) => false,
    };

    let mut flows = Vec::new();
    for to in &ends {
        for from in ends.iter().filter(|end| matches!(end.end, FlowEnd::Param(_))) {
            let map: Box<[Box<[usize]>]> = to
                .labels
                .iter()
                .zip(&to.written)
                .map(|(&out, &written)| match written {
                    false => Box::default(),
                    true => from
                        .labels
                        .iter()
                        .enumerate()
                        .filter(|(_, input)| reaches(**input, out))
                        .map(|(at, _)| at)
                        .collect(),
                })
                .collect();
            if map.iter().any(|takes| !takes.is_empty()) {
                flows.push(Flow {
                    to: to.end,
                    from: from.end,
                    alignment: Alignment::Labelled(Box::new(Labelled { declared, map })),
                });
            }
        }
    }
    Ok(Flows::of(flows))
}

/// One end of a signature with its labels, one per segment, and for each
/// segment whether the call may write into it as an output.
struct FittedEnd<'s> {
    end: FlowEnd,
    labels: &'s [SegmentLabel],
    written: Vec<bool>,
}

/// The result and each parameter, with the labels the signature gives
/// them, when each end has exactly one label per segment and the
/// signature lists each end at most once. The callee's captures have no
/// segments, since a named function captures nothing, so they fit only
/// unlabelled and take part in no flow.
fn fitted_ends<'s>(
    decls: &Declarations,
    params: &[TyId<Poly>],
    ret: TyId<Poly>,
    signature: &'s LabelledSignature,
) -> Result<Vec<FittedEnd<'s>>, LabelsDoNotFit> {
    let mut given: FxHashMap<FlowEnd, &'s [SegmentLabel]> = FxHashMap::default();
    for end in signature.ends.iter() {
        if given.insert(end.end, &end.labels).is_some() {
            return Err(LabelsDoNotFit);
        }
    }
    let captures_labelled = given.remove(&FlowEnd::Captures).is_some_and(|labels| !labels.is_empty());
    if captures_labelled {
        return Err(LabelsDoNotFit);
    }

    let result = FittedEnd::fit(decls, FlowEnd::Result, ret, given.remove(&FlowEnd::Result))?;
    let mut fitted = vec![result];
    for (at, param) in params.iter().enumerate() {
        let end = FlowEnd::Param(at);
        fitted.push(FittedEnd::fit(decls, end, *param, given.remove(&end))?);
    }
    match given.is_empty() {
        true => Ok(fitted),
        false => Err(LabelsDoNotFit),
    }
}

impl<'s> FittedEnd<'s> {
    /// `end` of declared type `ty` with the labels the signature gives it.
    /// An end the signature lists no labels for is given none, so it fits
    /// only where its type has no segment.
    fn fit(
        decls: &Declarations,
        end: FlowEnd,
        ty: TyId<Poly>,
        given: Option<&'s [SegmentLabel]>,
    ) -> Result<FittedEnd<'s>, LabelsDoNotFit> {
        let labels = match given {
            Some(labels) => labels,
            None => &[],
        };
        let segments = segments(&decls.types, decls, ty);
        let one_per_segment = labels.len() == segments.len();
        if !one_per_segment || !labels.iter().zip(&segments).all(|(label, segment)| label.fits(*segment)) {
            return Err(LabelsDoNotFit);
        }
        let written = match end {
            FlowEnd::Result => vec![true; labels.len()],
            FlowEnd::Param(_) | FlowEnd::Captures => written_segments(&decls.types, decls, ty),
        };
        assert_eq!(written.len(), labels.len(), "one writability per segment, in segment order");
        Ok(FittedEnd { end, labels, written })
    }
}

impl SegmentLabel {
    /// A lifetime labels a reference, a declared lifetime or a function
    /// value; a type variable's label, that variable's part alone.
    fn fits(self, segment: Segment) -> bool {
        match (self, segment) {
            (SegmentLabel::Lifetime(_), Segment::Reference | Segment::Lifetime | Segment::FunctionValue) => true,
            (SegmentLabel::TypeVar(label), Segment::TypeVar(var)) => label == var,
            (SegmentLabel::Lifetime(_), Segment::TypeVar(_))
            | (SegmentLabel::TypeVar(_), Segment::Reference | Segment::Lifetime | Segment::FunctionValue) => false,
        }
    }
}

/// Every `(longer, shorter)` the signature's `Outlives` give, closed under
/// transitivity, when each names a lifetime some label uses.
fn outlives_closure(
    signature: &LabelledSignature,
    ends: &[FittedEnd<'_>],
) -> Result<FxHashSet<(Lifetime, Lifetime)>, LabelsDoNotFit> {
    let used: FxHashSet<Lifetime> = ends
        .iter()
        .flat_map(|end| end.labels.iter())
        .filter_map(|label| match label {
            SegmentLabel::Lifetime(lifetime) => Some(*lifetime),
            SegmentLabel::TypeVar(_) => None,
        })
        .collect();
    if signature
        .outlives
        .iter()
        .any(|order| !used.contains(&order.longer) || !used.contains(&order.shorter))
    {
        return Err(LabelsDoNotFit);
    }

    let mut closure: FxHashSet<(Lifetime, Lifetime)> = signature
        .outlives
        .iter()
        .map(|order| (order.longer, order.shorter))
        .collect();
    loop {
        let implied: Vec<(Lifetime, Lifetime)> = closure
            .iter()
            .flat_map(|&(a, b)| {
                closure
                    .iter()
                    .filter(move |&&(c, _)| c == b)
                    .map(move |&(_, d)| (a, d))
            })
            .filter(|pair| !closure.contains(pair))
            .collect();
        if implied.is_empty() {
            return Ok(closure);
        }
        closure.extend(implied);
    }
}

/// For each segment of a parameter's declared type, in `flows::segments`
/// order, whether a call may write into it: it lies in what a `&mut`
/// names, or it is a function value that captures mutably (`FlowEnd`).
fn written_segments(types: &Types<Poly>, lifetimes: &dyn LifetimesOf, ty: TyId<Poly>) -> Vec<bool> {
    let mut written = Vec::new();
    push_written(types, lifetimes, ty, false, &mut written);
    written
}

fn push_written(types: &Types<Poly>, lifetimes: &dyn LifetimesOf, ty: TyId<Poly>, under_mut: bool, out: &mut Vec<bool>) {
    acvus_utils::grow(|| match types.kind(ty) {
        TyKind::Int(_)
        | TyKind::Float
        | TyKind::Char
        | TyKind::String
        | TyKind::Bool
        | TyKind::Unit
        | TyKind::Never
        | TyKind::Str => {}
        TyKind::Order | TyKind::Handle(_) => panic!("a declared type holds no Handle and no Order"),
        TyKind::Fn { holds, .. } => out.push(under_mut || *holds == Holds::Mut),
        TyKind::Var(_) => out.push(under_mut),
        TyKind::Ref(mutability, target) => {
            out.push(under_mut);
            push_written(types, lifetimes, target.ty(), under_mut || *mutability == Mutability::Mut, out);
        }
        TyKind::UserDefined { id, type_args, .. } => {
            out.extend(std::iter::repeat_n(under_mut, lifetimes.lifetimes_of(*id)));
            for arg in type_args {
                push_written(types, lifetimes, arg.ty(), under_mut, out);
            }
        }
        TyKind::Array(element, _) | TyKind::Slice(element) | TyKind::Option(element) => {
            push_written(types, lifetimes, *element, under_mut, out)
        }
        TyKind::Result(ok, err) => {
            push_written(types, lifetimes, *ok, under_mut, out);
            push_written(types, lifetimes, *err, under_mut, out);
        }
        TyKind::Tuple(items) => {
            for item in items {
                push_written(types, lifetimes, *item, under_mut, out);
            }
        }
        TyKind::Object(object) => {
            for (_, field) in object.fields() {
                push_written(types, lifetimes, *field, under_mut, out);
            }
        }
        TyKind::Enum { variants, .. } => {
            for payload in variants.iter().filter_map(|(_, payload)| *payload) {
                push_written(types, lifetimes, payload, under_mut, out);
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use acvus_utils::Interner;

    use super::*;
    use crate::decl::{DeclaredFacts, EndLabels, GenericSig, InstanceSig, Instances, LoopMarkers, Outlives};
    use crate::ty::{Calls, Effect, EffectTerm, IntTy, LenTerm, TypeArg};

    struct Fixture {
        interner: Interner,
        raw: RawDeclarations,
    }

    impl Fixture {
        fn new() -> Fixture {
            let interner = Interner::new();
            let core = Core::of(&interner);
            Fixture {
                interner,
                raw: RawDeclarations {
                    types: Types::new(),
                    functions: Vec::new(),
                    user_types: Vec::new(),
                    casts: Vec::new(),
                    conversions: Vec::new(),
                    contexts: Vec::new(),
                    inputs: Vec::new(),
                    loop_markers: LoopMarkers::default(),
                    core,
                },
            }
        }

        fn name(&self, namespace: &str, name: &str) -> QualifiedRef {
            QualifiedRef::qualified(self.interner.intern(namespace), self.interner.intern(name))
        }

        fn ty(&mut self, kind: TyKind<Poly>) -> TyId<Poly> {
            self.raw.types.intern(kind)
        }

        fn reference(&mut self, mutability: Mutability, target: TyId<Poly>) -> TyId<Poly> {
            self.ty(TyKind::Ref(mutability, TypeArg::Uniform(target)))
        }

        fn shared_str(&mut self) -> TyId<Poly> {
            let str_ty = self.ty(TyKind::Str);
            self.reference(Mutability::Shared, str_ty)
        }

        fn user_type(&mut self, id: QualifiedRef, lifetimes: usize) -> TyId<Poly> {
            self.raw.user_types.push(UserTypeDecl {
                id,
                type_params: Box::new([]),
                specializes: Box::new([]),
                effect_params: 0,
                identity_params: 0,
                lifetimes,
            });
            self.ty(TyKind::UserDefined {
                id,
                type_args: Box::new([]),
                effect_args: Box::new([]),
                identity_args: Box::new([]),
            })
        }

        fn function(&mut self, params: &[TyId<Poly>], ret: TyId<Poly>) -> TyId<Poly> {
            self.ty(TyKind::Fn {
                params: params.into(),
                ret,
                effect: EffectTerm::Known(Effect::pure()),
                flows: FlowTerm::Known(Flows::none()),
                calls: Calls::Many,
                holds: Holds::Nothing,
            })
        }

        /// `p0`, `p1`, ..., one per parameter of the function type `ty`;
        /// none where `ty` is no function type.
        fn param_names(&self, ty: TyId<Poly>) -> Box<[Astr]> {
            let arity = match self.raw.types.kind(ty) {
                TyKind::Fn { params, .. } => params.len(),
                _ => 0,
            };
            (0..arity).map(|at| self.interner.intern(&format!("p{at}"))).collect()
        }

        /// A scheme at `ty`, whose type variables are `0..variables`, each
        /// bounded by any type.
        fn plain(&self, ty: TyId<Poly>, variables: usize) -> Scheme {
            Scheme {
                ty,
                param_names: self.param_names(ty),
                bounds: vec![TyVarBound::Any; variables].into(),
                instances: None,
                requires: Box::new([]),
            }
        }

        fn labelled(&self, ty: TyId<Poly>, variables: usize, lifetimes: LabelledSignature) -> Scheme {
            Scheme {
                instances: Some(Instances {
                    concrete: Box::new([]),
                    generic: Some(GenericSig {
                        facts: DeclaredFacts {
                            lifetimes: Some(lifetimes),
                            ..DeclaredFacts::default()
                        },
                    }),
                }),
                ..self.plain(ty, variables)
            }
        }

        /// A scheme whose concrete instances are `tys`.
        fn with_instances(&self, own_ty: TyId<Poly>, variables: usize, tys: &[TyId<Poly>]) -> Scheme {
            Scheme {
                instances: Some(Instances {
                    concrete: tys.iter().map(|ty| self.instance(*ty, Box::new([]))).collect(),
                    generic: None,
                }),
                ..self.plain(own_ty, variables)
            }
        }

        /// A concrete instance at `ty` stating no fact.
        fn instance(&self, ty: TyId<Poly>, requires: Box<[Requirement]>) -> InstanceSig {
            InstanceSig {
                ty,
                param_names: self.param_names(ty),
                requires,
                facts: DeclaredFacts::default(),
            }
        }

        fn declare(&mut self, id: QualifiedRef, scheme: Scheme) {
            self.raw.functions.push((id, scheme));
        }

        fn verify(self) -> Result<Declarations, Vec<Refused>> {
            Declarations::verify(self.raw)
        }
    }

    fn lt(at: u32) -> SegmentLabel {
        SegmentLabel::Lifetime(Lifetime(at))
    }

    fn end(end: FlowEnd, labels: &[SegmentLabel]) -> EndLabels {
        EndLabels {
            end,
            labels: labels.into(),
        }
    }

    fn signature(ends: Vec<EndLabels>, outlives: &[(u32, u32)]) -> LabelledSignature {
        LabelledSignature {
            ends: ends.into(),
            outlives: outlives
                .iter()
                .map(|&(longer, shorter)| Outlives {
                    longer: Lifetime(longer),
                    shorter: Lifetime(shorter),
                })
                .collect(),
        }
    }

    fn flow(to: FlowEnd, from: FlowEnd, declared: Declared, map: &[&[usize]]) -> Flow {
        Flow {
            to,
            from,
            alignment: Alignment::Labelled(Box::new(Labelled {
                declared,
                map: map.iter().map(|takes| Box::<[usize]>::from(*takes)).collect(),
            })),
        }
    }

    fn own(function: QualifiedRef) -> Declared {
        Declared {
            function,
            instance: None,
        }
    }

    fn flows_at(decls: &Declarations, ty: TyId<Poly>) -> Flows {
        let TyKind::Fn { flows, .. } = decls.types().kind(ty) else {
            panic!("a declared signature is a function type")
        };
        flows.get().clone()
    }

    fn own_flows(decls: &Declarations, function: QualifiedRef) -> Flows {
        flows_at(decls, decls.scheme(function).expect("declared").ty)
    }

    fn refusals(fixture: Fixture) -> Vec<Refused> {
        match fixture.verify() {
            Ok(_) => panic!("the set is expected to be refused"),
            Err(refused) => refused,
        }
    }

    /// A set with no declaration breaks no rule, so it is taken in.
    #[test]
    fn an_empty_set_is_taken_in() {
        assert!(Fixture::new().verify().is_ok());
    }

    /// extern-fn §1: no declaration carries an array type.
    #[test]
    fn a_signature_holding_an_array_is_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let array = fixture.ty(TyKind::Array(int, LenTerm::Known(3)));
        let unit = fixture.ty(TyKind::Unit);
        let ty = fixture.function(&[array], unit);
        fixture.declare(f, fixture.plain(ty, 0));
        assert_eq!(refusals(fixture), vec![Refused::ArrayType { declaration: f }]);
    }

    /// extern-fn §1: no declaration carries an array type, an instance's
    /// type included.
    #[test]
    fn an_instance_holding_an_array_is_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let var = fixture.ty(TyKind::Var(0));
        let unit = fixture.ty(TyKind::Unit);
        let own_ty = fixture.function(&[var], unit);
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let array = fixture.ty(TyKind::Array(int, LenTerm::Var(0)));
        let instance_ty = fixture.function(&[array], unit);
        fixture.declare(
            f,
            Scheme {
                instances: Some(Instances {
                    concrete: Box::new([fixture.instance(instance_ty, Box::new([]))]),
                    generic: None,
                }),
                ..fixture.plain(own_ty, 1)
            },
        );
        assert_eq!(refusals(fixture), vec![Refused::ArrayType { declaration: f }]);
    }

    /// An extern type is the single source of its parameters (extern-fn §8),
    /// so two declarations under one id are refused.
    #[test]
    fn two_types_under_one_id_are_refused() {
        let mut fixture = Fixture::new();
        let id = fixture.name("m", "T");
        fixture.user_type(id, 0);
        fixture.user_type(id, 1);
        assert_eq!(refusals(fixture), vec![Refused::DuplicateType { id }]);
    }

    /// The registry's rule `Refused::TwoIdentities` carries: a type has at
    /// most one identity parameter.
    #[test]
    fn a_type_with_two_identities_is_refused() {
        let mut fixture = Fixture::new();
        let id = fixture.name("m", "T");
        fixture.raw.user_types.push(UserTypeDecl {
            id,
            type_params: Box::new([]),
            specializes: Box::new([]),
            effect_params: 0,
            identity_params: 2,
            lifetimes: 0,
        });
        assert_eq!(refusals(fixture), vec![Refused::TwoIdentities { id }]);
    }

    /// types §3: a declared cast from `S` to `T` is `S`'s one step up, and
    /// `T` steps nowhere unless a cast says so.
    #[test]
    fn a_declared_cast_is_its_heads_one_step_up() {
        let mut fixture = Fixture::new();
        let (a, b, via) = (fixture.name("m", "A"), fixture.name("m", "B"), fixture.name("m", "a_to_b"));
        let from = fixture.user_type(a, 0);
        let to = fixture.user_type(b, 0);
        fixture.raw.casts.push(CastRule { from, to, via });
        let decls = fixture.verify().expect("one step up is a tree");
        let step = decls.casts().step_up(Head::Base(Base::Named(a))).expect("A steps up");
        assert_eq!(step.rule().via, via);
        assert!(decls.casts().step_up(Head::Base(Base::Named(b))).is_none());
    }

    /// types §3: every type has at most one next step up.
    #[test]
    fn two_steps_up_from_one_head_are_refused() {
        let mut fixture = Fixture::new();
        let (a, b, c) = (fixture.name("m", "A"), fixture.name("m", "B"), fixture.name("m", "C"));
        let (to_b, to_c) = (fixture.name("m", "to_b"), fixture.name("m", "to_c"));
        let a_ty = fixture.user_type(a, 0);
        let b_ty = fixture.user_type(b, 0);
        let c_ty = fixture.user_type(c, 0);
        fixture.raw.casts.push(CastRule {
            from: a_ty,
            to: b_ty,
            via: to_b,
        });
        fixture.raw.casts.push(CastRule {
            from: a_ty,
            to: c_ty,
            via: to_c,
        });
        assert_eq!(
            refusals(fixture),
            vec![Refused::TwoStepsUp {
                head: Head::Base(Base::Named(a)),
                via: [to_b, to_c]
            }]
        );
    }

    /// types §3: the steps form a tree, so no path of steps returns to its
    /// start. One cycle is one refusal.
    #[test]
    fn steps_that_return_to_their_start_are_refused() {
        let mut fixture = Fixture::new();
        let (a, b) = (fixture.name("m", "A"), fixture.name("m", "B"));
        let (to_b, to_a) = (fixture.name("m", "to_b"), fixture.name("m", "to_a"));
        let a_ty = fixture.user_type(a, 0);
        let b_ty = fixture.user_type(b, 0);
        fixture.raw.casts.push(CastRule {
            from: a_ty,
            to: b_ty,
            via: to_b,
        });
        fixture.raw.casts.push(CastRule {
            from: b_ty,
            to: a_ty,
            via: to_a,
        });
        assert_eq!(refusals(fixture), vec![Refused::CastCycle { head: Head::Base(Base::Named(a)) }]);
    }

    /// types §3: a cast between two types of one head is a step from that
    /// head to itself, a path back to its start.
    #[test]
    fn a_cast_within_one_head_is_refused() {
        let mut fixture = Fixture::new();
        let via = fixture.name("m", "widen");
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let string = fixture.ty(TyKind::String);
        let from = fixture.ty(TyKind::Option(int));
        let to = fixture.ty(TyKind::Option(string));
        fixture.raw.casts.push(CastRule { from, to, via });
        assert_eq!(refusals(fixture), vec![Refused::CastCycle { head: Head::Base(Base::Option) }]);
    }

    fn requiring(signature: QualifiedRef, pattern: TyId<Poly>) -> Box<[Requirement]> {
        Box::new([Requirement {
            signature,
            pattern,
            law: None,
        }])
    }

    /// extern-fn §6: a required instance's types are smaller than the
    /// declaration's own. `fn(T) -> ()` has three nodes, and so has the
    /// pattern `(T, T)`.
    #[test]
    fn a_requirement_as_large_as_its_declaration_is_refused() {
        let mut fixture = Fixture::new();
        let (f, shared) = (fixture.name("m", "f"), fixture.name("m", "hash"));
        let var = fixture.ty(TyKind::Var(0));
        let unit = fixture.ty(TyKind::Unit);
        let ty = fixture.function(&[var], unit);
        let pattern = fixture.ty(TyKind::Tuple(Box::new([var, var])));
        fixture.declare(
            f,
            Scheme {
                requires: requiring(shared, pattern),
                ..fixture.plain(ty, 1)
            },
        );
        assert_eq!(
            refusals(fixture),
            vec![Refused::RequirementNotSmaller {
                declaration: f,
                signature: shared
            }]
        );
    }

    /// extern-fn §6: the pattern `T`, one node, is smaller than `fn(T) -> ()`,
    /// three.
    #[test]
    fn a_smaller_requirement_is_taken_in() {
        let mut fixture = Fixture::new();
        let (f, shared) = (fixture.name("m", "f"), fixture.name("m", "hash"));
        let var = fixture.ty(TyKind::Var(0));
        let unit = fixture.ty(TyKind::Unit);
        let ty = fixture.function(&[var], unit);
        fixture.declare(
            f,
            Scheme {
                requires: requiring(shared, var),
                ..fixture.plain(ty, 1)
            },
        );
        assert!(fixture.verify().is_ok());
    }

    /// extern-fn §6: an instance's requirement is measured against the
    /// instance's own type. `(i64, i64, i64)` has four nodes: fewer than the
    /// instance `fn(Option<Option<i64>>) -> ()`, five, though not fewer than
    /// the scheme's `fn(T) -> ()`, three.
    #[test]
    fn an_instance_requirement_is_measured_against_the_instance() {
        let mut fixture = Fixture::new();
        let (f, shared) = (fixture.name("m", "f"), fixture.name("m", "hash"));
        let var = fixture.ty(TyKind::Var(0));
        let unit = fixture.ty(TyKind::Unit);
        let own_ty = fixture.function(&[var], unit);
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let inner = fixture.ty(TyKind::Option(int));
        let outer = fixture.ty(TyKind::Option(inner));
        let instance_ty = fixture.function(&[outer], unit);
        let pattern = fixture.ty(TyKind::Tuple(Box::new([int, int, int])));
        fixture.declare(
            f,
            Scheme {
                instances: Some(Instances {
                    concrete: Box::new([fixture.instance(instance_ty, requiring(shared, pattern))]),
                    generic: None,
                }),
                ..fixture.plain(own_ty, 1)
            },
        );
        assert!(fixture.verify().is_ok());
    }

    /// ownership §5, extern-fn §4: `fn f<'a, 'b>(x: &'a str, y: &'b str) ->
    /// &'a str`. The result's one segment shares `'a` with `x`'s alone.
    #[test]
    fn a_result_takes_from_the_input_of_its_lifetime() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let s = fixture.shared_str();
        let ty = fixture.function(&[s, s], s);
        let lifetimes = signature(
            vec![
                end(FlowEnd::Result, &[lt(0)]),
                end(FlowEnd::Param(0), &[lt(0)]),
                end(FlowEnd::Param(1), &[lt(1)]),
            ],
            &[],
        );
        fixture.declare(f, fixture.labelled(ty, 0, lifetimes));
        let decls = fixture.verify().expect("the labels fit");
        assert_eq!(
            own_flows(&decls, f),
            Flows::of([flow(FlowEnd::Result, FlowEnd::Param(0), own(f), &[&[0]])])
        );
    }

    /// extern-fn §4: `fn f<'a: 'b, 'b: 'c, 'c>(x: &'a str, y: &'b str, z:
    /// &'c str) -> &'c str`. `'a` outlives `'c` through `'b`, so the result
    /// takes from every input.
    #[test]
    fn a_result_takes_from_every_input_that_outlives_it() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let s = fixture.shared_str();
        let ty = fixture.function(&[s, s, s], s);
        let lifetimes = signature(
            vec![
                end(FlowEnd::Result, &[lt(2)]),
                end(FlowEnd::Param(0), &[lt(0)]),
                end(FlowEnd::Param(1), &[lt(1)]),
                end(FlowEnd::Param(2), &[lt(2)]),
            ],
            &[(0, 1), (1, 2)],
        );
        fixture.declare(f, fixture.labelled(ty, 0, lifetimes));
        let decls = fixture.verify().expect("the labels fit");
        assert_eq!(
            own_flows(&decls, f),
            Flows::of([
                flow(FlowEnd::Result, FlowEnd::Param(0), own(f), &[&[0]]),
                flow(FlowEnd::Result, FlowEnd::Param(1), own(f), &[&[0]]),
                flow(FlowEnd::Result, FlowEnd::Param(2), own(f), &[&[0]]),
            ])
        );
    }

    /// extern-fn §4: `fn f<'a: 'b, 'b>(x: &'a str, y: &'b str) -> &'a str`.
    /// `'b` does not outlive `'a`, so the result takes nothing from `y`.
    #[test]
    fn a_result_takes_nothing_from_an_input_it_outlives() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let s = fixture.shared_str();
        let ty = fixture.function(&[s, s], s);
        let lifetimes = signature(
            vec![
                end(FlowEnd::Result, &[lt(0)]),
                end(FlowEnd::Param(0), &[lt(0)]),
                end(FlowEnd::Param(1), &[lt(1)]),
            ],
            &[(0, 1)],
        );
        fixture.declare(f, fixture.labelled(ty, 0, lifetimes));
        let decls = fixture.verify().expect("the labels fit");
        assert_eq!(
            own_flows(&decls, f),
            Flows::of([flow(FlowEnd::Result, FlowEnd::Param(0), own(f), &[&[0]])])
        );
    }

    /// ownership §5, `FlowEnd`: `fn push<'a, T>(v: &'a mut [T], x: T)`. A
    /// parameter's output is what its `&mut` names, here `v`'s `T` part, and
    /// one type variable's parts meet k-th to k-th: `v` takes from `x` and
    /// from itself, into its `T` segment only.
    #[test]
    fn a_mut_parameter_takes_from_the_inputs_of_its_type_variable() {
        let mut fixture = Fixture::new();
        let push = fixture.name("m", "push");
        let var = fixture.ty(TyKind::Var(0));
        let slice = fixture.ty(TyKind::Slice(var));
        let v = fixture.reference(Mutability::Mut, slice);
        let unit = fixture.ty(TyKind::Unit);
        let ty = fixture.function(&[v, var], unit);
        let lifetimes = signature(
            vec![
                end(FlowEnd::Param(0), &[lt(0), SegmentLabel::TypeVar(0)]),
                end(FlowEnd::Param(1), &[SegmentLabel::TypeVar(0)]),
            ],
            &[],
        );
        fixture.declare(push, fixture.labelled(ty, 1, lifetimes));
        let decls = fixture.verify().expect("the labels fit");
        assert_eq!(
            segments(decls.types(), &decls, v),
            vec![Segment::Reference, Segment::TypeVar(0)]
        );
        assert_eq!(
            own_flows(&decls, push),
            Flows::of([
                flow(FlowEnd::Param(0), FlowEnd::Param(0), own(push), &[&[], &[1]]),
                flow(FlowEnd::Param(0), FlowEnd::Param(1), own(push), &[&[], &[0]]),
            ])
        );
    }

    /// ownership §5: each lifetime an extern-defined type declares is a
    /// position of its values. `fn iter<'a>(s: &'a String) -> Iter<'a>`.
    #[test]
    fn an_extern_types_lifetime_is_a_segment() {
        let mut fixture = Fixture::new();
        let (iter, iter_ty) = (fixture.name("m", "iter"), fixture.name("m", "Iter"));
        let result = fixture.user_type(iter_ty, 1);
        let string = fixture.ty(TyKind::String);
        let s = fixture.reference(Mutability::Shared, string);
        let ty = fixture.function(&[s], result);
        let lifetimes = signature(
            vec![end(FlowEnd::Result, &[lt(0)]), end(FlowEnd::Param(0), &[lt(0)])],
            &[],
        );
        fixture.declare(iter, fixture.labelled(ty, 0, lifetimes));
        let decls = fixture.verify().expect("the labels fit");
        assert_eq!(
            own_flows(&decls, iter),
            Flows::of([flow(FlowEnd::Result, FlowEnd::Param(0), own(iter), &[&[0]])])
        );
    }

    /// extern-fn §4: an end's labels are one per segment of its declared
    /// type. `&str` has one segment, labelled here twice.
    #[test]
    fn labels_not_one_per_segment_are_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let s = fixture.shared_str();
        let ty = fixture.function(&[s], s);
        let lifetimes = signature(
            vec![end(FlowEnd::Result, &[lt(0), lt(0)]), end(FlowEnd::Param(0), &[lt(0)])],
            &[],
        );
        fixture.declare(f, fixture.labelled(ty, 0, lifetimes));
        assert_eq!(refusals(fixture), vec![Refused::LabelsDoNotFit { declaration: f }]);
    }

    /// labels: every segment arrives labelled, so an end with segments and
    /// no labels does not fit.
    #[test]
    fn an_unlabelled_end_with_segments_is_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let s = fixture.shared_str();
        let ty = fixture.function(&[s], s);
        let lifetimes = signature(vec![end(FlowEnd::Result, &[lt(0)])], &[]);
        fixture.declare(f, fixture.labelled(ty, 0, lifetimes));
        assert_eq!(refusals(fixture), vec![Refused::LabelsDoNotFit { declaration: f }]);
    }

    /// extern-fn §4: every `Outlives` names lifetimes the labels use; `'1`
    /// labels no segment.
    #[test]
    fn an_order_on_an_unused_lifetime_is_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let s = fixture.shared_str();
        let ty = fixture.function(&[s], s);
        let lifetimes = signature(
            vec![end(FlowEnd::Result, &[lt(0)]), end(FlowEnd::Param(0), &[lt(0)])],
            &[(1, 0)],
        );
        fixture.declare(f, fixture.labelled(ty, 0, lifetimes));
        assert_eq!(refusals(fixture), vec![Refused::LabelsDoNotFit { declaration: f }]);
    }

    /// extern-fn §4: a declaration that states no lifetimes has every input
    /// flowing to every output, whatever flows its type was handed with.
    #[test]
    fn a_declaration_without_lifetimes_has_every_flow() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let s = fixture.shared_str();
        let ty = fixture.function(&[s], s);
        fixture.declare(f, fixture.plain(ty, 0));
        let decls = fixture.verify().expect("nothing to refuse");
        assert_eq!(own_flows(&decls, f), Flows::Every);
    }

    /// extern-fn §4: an instance's lifetimes become the instance's own flows,
    /// naming the instance; the scheme's type, with no generic instance to
    /// state lifetimes, has every flow. No lifetime remains after intake.
    #[test]
    fn an_instances_flows_name_the_instance_and_no_lifetime_remains() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let var = fixture.ty(TyKind::Var(0));
        let own_ty = fixture.function(&[var], var);
        let s = fixture.shared_str();
        let instance_ty = fixture.function(&[s], s);
        let lifetimes = signature(
            vec![end(FlowEnd::Result, &[lt(0)]), end(FlowEnd::Param(0), &[lt(0)])],
            &[],
        );
        fixture.declare(
            f,
            Scheme {
                instances: Some(Instances {
                    concrete: Box::new([InstanceSig {
                        ty: instance_ty,
                        param_names: fixture.param_names(instance_ty),
                        requires: Box::new([]),
                        facts: DeclaredFacts {
                            lifetimes: Some(lifetimes),
                            ..DeclaredFacts::default()
                        },
                    }]),
                    generic: None,
                }),
                ..fixture.plain(own_ty, 1)
            },
        );
        let decls = fixture.verify().expect("the labels fit");
        let scheme = decls.scheme(f).expect("declared");
        let instance = &scheme.instances.as_ref().expect("instances").concrete[0];
        let at_instance = Declared {
            function: f,
            instance: Some(0),
        };
        assert_eq!(own_flows(&decls, f), Flows::Every);
        assert_eq!(
            flows_at(&decls, instance.ty),
            Flows::of([flow(FlowEnd::Result, FlowEnd::Param(0), at_instance, &[&[0]])])
        );
        assert_eq!(instance.facts.lifetimes, None);
    }

    /// extern-fn §5, types §6: a bare name reaches every declaration of that
    /// name, in any namespace.
    #[test]
    fn a_bare_name_reaches_every_namespace() {
        let mut fixture = Fixture::new();
        let (a_len, b_len) = (fixture.name("a", "len"), fixture.name("b", "len"));
        let unit = fixture.ty(TyKind::Unit);
        let ty = fixture.function(&[], unit);
        fixture.declare(a_len, fixture.plain(ty, 0));
        fixture.declare(b_len, fixture.plain(ty, 0));
        let len = fixture.interner.intern("len");
        let other = fixture.interner.intern("other");
        let decls = fixture.verify().expect("nothing to refuse");
        assert_eq!(decls.overloads(len), &[a_len, b_len]);
        assert_eq!(decls.overloads(other), &[] as &[QualifiedRef]);
    }

    /// core's view signature: `fn(&C) -> &[T]` under `mutability`, or
    /// `fn(&C) -> &str` where `slice` is false.
    fn view_signature(fixture: &mut Fixture, mutability: Mutability, slice: bool) -> TyId<Poly> {
        let container = fixture.ty(TyKind::Var(0));
        let container_ref = fixture.reference(mutability, container);
        let view = match slice {
            true => {
                let element = fixture.ty(TyKind::Var(1));
                fixture.ty(TyKind::Slice(element))
            }
            false => fixture.ty(TyKind::Str),
        };
        let view_ref = fixture.reference(mutability, view);
        fixture.function(&[container_ref], view_ref)
    }

    /// operators §5, types §3: `as_slice` at `&Vec` is a step from
    /// `Ref(Shared, Named(Vec))` to `Ref(Shared, Slice)`; `as_str` at
    /// `&String` from `Ref(Shared, String)` to `Ref(Shared, Str)`. Two views
    /// leave two heads, so they form a tree.
    #[test]
    fn a_view_is_a_step_from_a_containers_reference_to_its_view() {
        let mut fixture = Fixture::new();
        let core = fixture.raw.core;
        let vec = fixture.name("m", "Vec");
        let vec_ty = fixture.user_type(vec, 0);
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let slice = fixture.ty(TyKind::Slice(int));
        let vec_ref = fixture.reference(Mutability::Shared, vec_ty);
        let slice_ref = fixture.reference(Mutability::Shared, slice);
        let as_slice_at_vec = fixture.function(&[vec_ref], slice_ref);
        let string = fixture.ty(TyKind::String);
        let string_ref = fixture.reference(Mutability::Shared, string);
        let str_ref = fixture.shared_str();
        let as_str_at_string = fixture.function(&[string_ref], str_ref);
        let (as_slice_ty, as_str_ty) = (view_signature(&mut fixture, Mutability::Shared, true), view_signature(&mut fixture, Mutability::Shared, false));
        fixture.declare(core.as_slice, fixture.with_instances(as_slice_ty, 2, &[as_slice_at_vec]));
        fixture.declare(core.as_str, fixture.with_instances(as_str_ty, 1, &[as_str_at_string]));
        let decls = fixture.verify().expect("two views from two heads are a tree");
        let from_vec = decls
            .casts()
            .step_up(Head::Ref(Mutability::Shared, Base::Named(vec)))
            .expect("&Vec steps up");
        assert_eq!(from_vec.rule().via, core.as_slice);
        let from_string = decls
            .casts()
            .step_up(Head::Ref(Mutability::Shared, Base::String))
            .expect("&String steps up");
        assert_eq!(from_string.rule().via, core.as_str);
        assert!(decls.casts().step_up(Head::Ref(Mutability::Shared, Base::Slice)).is_none());
    }

    /// types §3: the one-step-up rule holds for views as for declared casts.
    /// A declared cast from `&Vec` and `as_slice` at `&Vec` are two steps up
    /// from one head.
    #[test]
    fn a_view_and_a_declared_cast_from_one_head_are_refused() {
        let mut fixture = Fixture::new();
        let core = fixture.raw.core;
        let (vec, other, to_other) = (fixture.name("m", "Vec"), fixture.name("m", "Other"), fixture.name("m", "to_other"));
        let vec_ty = fixture.user_type(vec, 0);
        let other_ty = fixture.user_type(other, 0);
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let slice = fixture.ty(TyKind::Slice(int));
        let vec_ref = fixture.reference(Mutability::Shared, vec_ty);
        let other_ref = fixture.reference(Mutability::Shared, other_ty);
        let slice_ref = fixture.reference(Mutability::Shared, slice);
        let as_slice_at_vec = fixture.function(&[vec_ref], slice_ref);
        fixture.declare(core.as_slice, fixture.with_instances(as_slice_at_vec, 0, &[as_slice_at_vec]));
        fixture.raw.casts.push(CastRule {
            from: vec_ref,
            to: other_ref,
            via: to_other,
        });
        assert_eq!(
            refusals(fixture),
            vec![Refused::TwoStepsUp {
                head: Head::Ref(Mutability::Shared, Base::Named(vec)),
                via: [to_other, core.as_slice]
            }]
        );
    }

    /// types §3: a tree step joins two heads; a type variable has none.
    #[test]
    fn a_cast_without_a_head_is_refused() {
        let mut fixture = Fixture::new();
        let via = fixture.name("m", "from_any");
        let from = fixture.ty(TyKind::Var(0));
        let to = fixture.ty(TyKind::String);
        fixture.raw.casts.push(CastRule { from, to, via });
        assert_eq!(refusals(fixture), vec![Refused::CastWithoutHead { via }]);
    }

    /// types §7: a conversion joins two heads, as a tree step does; a type
    /// variable has none.
    #[test]
    fn a_conversion_without_a_head_is_refused() {
        let mut fixture = Fixture::new();
        let via = fixture.name("m", "from_any");
        let from = fixture.ty(TyKind::Var(0));
        let to = fixture.ty(TyKind::String);
        fixture.raw.conversions.push(ConversionRule { from, to, via });
        assert_eq!(refusals(fixture), vec![Refused::CastWithoutHead { via }]);
    }

    /// extern-fn §7, types §7: a conversion is no tree step, so a head may
    /// have several, even to one head, and a head with a cast up may have
    /// conversions too.
    #[test]
    fn several_conversions_from_one_head_are_taken_in() {
        let mut fixture = Fixture::new();
        let (parse, parse_again, widen) = (
            fixture.name("m", "parse"),
            fixture.name("m", "parse_again"),
            fixture.name("m", "widen"),
        );
        let (string, i32_, i64_) = (
            fixture.ty(TyKind::String),
            fixture.ty(TyKind::Int(IntTy::I32)),
            fixture.ty(TyKind::Int(IntTy::I64)),
        );
        fixture.raw.casts.push(CastRule {
            from: i32_,
            to: i64_,
            via: widen,
        });
        for (from, via) in [(string, parse), (string, parse_again), (i32_, parse)] {
            fixture.raw.conversions.push(ConversionRule { from, to: i64_, via });
        }
        let decls = fixture.verify().expect("conversions form no tree");
        let vias = |head| {
            decls
                .conversions_from(head)
                .iter()
                .map(|conversion| conversion.rule().via)
                .collect::<Vec<_>>()
        };
        assert_eq!(vias(Head::Base(Base::String)), vec![parse, parse_again]);
        assert_eq!(vias(Head::Base(Base::Int(IntTy::I32))), vec![parse]);
        assert_eq!(vias(Head::Base(Base::Int(IntTy::I64))), Vec::new());
    }

    /// types §4: a conversion's types are declared types, and a reference's
    /// target is never a reference; it is refused as that alone.
    #[test]
    fn a_conversion_from_a_reference_to_a_reference_is_refused() {
        let mut fixture = Fixture::new();
        let via = fixture.name("m", "deref");
        let string = fixture.ty(TyKind::String);
        let inner = fixture.reference(Mutability::Shared, string);
        let from = fixture.reference(Mutability::Shared, inner);
        fixture.raw.conversions.push(ConversionRule { from, to: string, via });
        assert_eq!(
            refusals(fixture),
            vec![Refused::ReferenceToReference {
                written: Written::Cast(via)
            }]
        );
    }

    /// extern-fn §8: a type a conversion names is declared.
    #[test]
    fn a_conversion_naming_an_undeclared_type_is_refused() {
        let mut fixture = Fixture::new();
        let (via, ghost) = (fixture.name("m", "to_ghost"), fixture.name("m", "Ghost"));
        let from = fixture.ty(TyKind::String);
        let to = undeclared(&mut fixture, ghost);
        fixture.raw.conversions.push(ConversionRule { from, to, via });
        assert_eq!(
            refusals(fixture),
            vec![Refused::UndeclaredType {
                written: Written::Cast(via),
                id: ghost
            }]
        );
    }

    /// types §6: two declarations under one qualified name cannot be told
    /// apart, so the set is refused.
    #[test]
    fn two_functions_under_one_name_are_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let unit = fixture.ty(TyKind::Unit);
        let ty = fixture.function(&[], unit);
        fixture.declare(f, fixture.plain(ty, 0));
        fixture.declare(f, fixture.plain(ty, 0));
        assert_eq!(refusals(fixture), vec![Refused::DuplicateFunction { id: f }]);
    }

    /// context: a context is one value of one fixed type, so two
    /// declarations of it are refused.
    #[test]
    fn two_contexts_under_one_name_are_refused() {
        let mut fixture = Fixture::new();
        let id = fixture.name("m", "user");
        let (int, string) = (fixture.ty(TyKind::Int(IntTy::I64)), fixture.ty(TyKind::String));
        fixture.raw.contexts.push((id, int));
        fixture.raw.contexts.push((id, string));
        assert_eq!(refusals(fixture), vec![Refused::DuplicateContext { id }]);
    }

    /// programs §4: an input `$x` has one type, so two declarations of it
    /// are refused.
    #[test]
    fn two_inputs_under_one_name_are_refused() {
        let mut fixture = Fixture::new();
        let name = fixture.interner.intern("x");
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        fixture.raw.inputs.push((name, int));
        fixture.raw.inputs.push((name, int));
        assert_eq!(refusals(fixture), vec![Refused::DuplicateInput { name }]);
    }

    fn undeclared(fixture: &mut Fixture, id: QualifiedRef) -> TyId<Poly> {
        fixture.ty(TyKind::UserDefined {
            id,
            type_args: Box::new([]),
            effect_args: Box::new([]),
            identity_args: Box::new([]),
        })
    }

    /// extern-fn §8: a type a signature names is declared. A labelled
    /// signature naming an undeclared type is refused for the type alone:
    /// its segments are not read.
    #[test]
    fn a_signature_naming_an_undeclared_type_is_refused() {
        let mut fixture = Fixture::new();
        let (f, ghost) = (fixture.name("m", "f"), fixture.name("m", "Ghost"));
        let ghost_ty = undeclared(&mut fixture, ghost);
        let s = fixture.shared_str();
        let ty = fixture.function(&[s], ghost_ty);
        let lifetimes = signature(
            vec![end(FlowEnd::Result, &[lt(0)]), end(FlowEnd::Param(0), &[lt(0)])],
            &[],
        );
        fixture.declare(f, fixture.labelled(ty, 0, lifetimes));
        assert_eq!(
            refusals(fixture),
            vec![Refused::UndeclaredType {
                written: Written::Function(f),
                id: ghost
            }]
        );
    }

    /// extern-fn §8: a type a cast names is declared.
    #[test]
    fn a_cast_naming_an_undeclared_type_is_refused() {
        let mut fixture = Fixture::new();
        let (via, ghost) = (fixture.name("m", "to_string"), fixture.name("m", "Ghost"));
        let from = undeclared(&mut fixture, ghost);
        let to = fixture.ty(TyKind::String);
        fixture.raw.casts.push(CastRule { from, to, via });
        assert_eq!(
            refusals(fixture),
            vec![Refused::UndeclaredType {
                written: Written::Cast(via),
                id: ghost
            }]
        );
    }

    /// Label kinds: a type variable's label fits only that variable's
    /// part, so `T` labelled with a lifetime does not fit.
    #[test]
    fn a_lifetime_on_a_type_variable_is_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let var = fixture.ty(TyKind::Var(0));
        let ty = fixture.function(&[var], var);
        let lifetimes = signature(
            vec![end(FlowEnd::Result, &[lt(0)]), end(FlowEnd::Param(0), &[lt(0)])],
            &[],
        );
        fixture.declare(f, fixture.labelled(ty, 1, lifetimes));
        assert_eq!(refusals(fixture), vec![Refused::LabelsDoNotFit { declaration: f }]);
    }

    /// Label kinds: `TypeVar(v)` fits only the part of `v`, not a reference
    /// and not another variable's part.
    #[test]
    fn a_type_variable_label_off_its_variable_is_refused() {
        let mut fixture = Fixture::new();
        let (f, g) = (fixture.name("m", "f"), fixture.name("m", "g"));
        let (t, u) = (fixture.ty(TyKind::Var(0)), fixture.ty(TyKind::Var(1)));
        let s = fixture.shared_str();
        let f_ty = fixture.function(&[t], u);
        let g_ty = fixture.function(&[s], s);
        let off_variable = signature(
            vec![
                end(FlowEnd::Result, &[SegmentLabel::TypeVar(0)]),
                end(FlowEnd::Param(0), &[SegmentLabel::TypeVar(0)]),
            ],
            &[],
        );
        let on_reference = signature(
            vec![
                end(FlowEnd::Result, &[SegmentLabel::TypeVar(0)]),
                end(FlowEnd::Param(0), &[SegmentLabel::TypeVar(0)]),
            ],
            &[],
        );
        fixture.declare(f, fixture.labelled(f_ty, 2, off_variable));
        fixture.declare(g, fixture.labelled(g_ty, 0, on_reference));
        assert_eq!(
            refusals(fixture),
            vec![
                Refused::LabelsDoNotFit { declaration: f },
                Refused::LabelsDoNotFit { declaration: g }
            ]
        );
    }

    /// extern-fn §6: an instance is its signature's type at some choice of
    /// the signature's variables. `fn(i64) -> String` is no choice of `T` in
    /// `fn(T) -> T`; `fn(i64) -> i64` is.
    #[test]
    fn an_instance_not_of_its_signature_is_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let var = fixture.ty(TyKind::Var(0));
        let own_ty = fixture.function(&[var], var);
        let (int, string) = (fixture.ty(TyKind::Int(IntTy::I64)), fixture.ty(TyKind::String));
        let fits = fixture.function(&[int], int);
        let misfit = fixture.function(&[int], string);
        fixture.declare(f, fixture.with_instances(own_ty, 1, &[fits, misfit]));
        assert_eq!(
            refusals(fixture),
            vec![Refused::InstanceNotOfSignature {
                signature: f,
                instance: 1
            }]
        );
    }

    /// extern-fn §6: a choice fixes each variable once, so `fn(&T) -> &T`
    /// has no instance `fn(&i64) -> &mut i64`, nor one with a different
    /// constructor.
    #[test]
    fn an_instance_is_matched_constructor_by_constructor() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let var = fixture.ty(TyKind::Var(0));
        let var_ref = fixture.reference(Mutability::Shared, var);
        let own_ty = fixture.function(&[var_ref], var_ref);
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let int_ref = fixture.reference(Mutability::Shared, int);
        let int_mut = fixture.reference(Mutability::Mut, int);
        let option = fixture.ty(TyKind::Option(int));
        let fits = fixture.function(&[int_ref], int_ref);
        let wrong_mutability = fixture.function(&[int_ref], int_mut);
        let wrong_constructor = fixture.function(&[option], int_ref);
        fixture.declare(f, fixture.with_instances(own_ty, 1, &[fits, wrong_mutability, wrong_constructor]));
        assert_eq!(
            refusals(fixture),
            vec![
                Refused::InstanceNotOfSignature {
                    signature: f,
                    instance: 1
                },
                Refused::InstanceNotOfSignature {
                    signature: f,
                    instance: 2
                }
            ]
        );
    }

    /// types §4: a reference's target is never a reference, so a cast from
    /// `&&String` is refused as that, and as nothing else.
    #[test]
    fn a_cast_from_a_reference_to_a_reference_is_refused() {
        let mut fixture = Fixture::new();
        let via = fixture.name("m", "deref");
        let string = fixture.ty(TyKind::String);
        let inner = fixture.reference(Mutability::Shared, string);
        let from = fixture.reference(Mutability::Shared, inner);
        fixture.raw.casts.push(CastRule { from, to: string, via });
        assert_eq!(
            refusals(fixture),
            vec![Refused::ReferenceToReference {
                written: Written::Cast(via)
            }]
        );
    }

    /// types §4: a reference's target is never a reference, wherever a
    /// declared type holds one: `fn(Option<&&i64>) -> ()`.
    #[test]
    fn a_signature_holding_a_reference_to_a_reference_is_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let (int, unit) = (fixture.ty(TyKind::Int(IntTy::I64)), fixture.ty(TyKind::Unit));
        let inner = fixture.reference(Mutability::Shared, int);
        let outer = fixture.reference(Mutability::Mut, inner);
        let option = fixture.ty(TyKind::Option(outer));
        let ty = fixture.function(&[option], unit);
        fixture.declare(f, fixture.plain(ty, 0));
        assert_eq!(
            refusals(fixture),
            vec![Refused::ReferenceToReference {
                written: Written::Function(f)
            }]
        );
    }

    /// types §3: a view signature's generic instance has a variable for its
    /// container, which has no head.
    #[test]
    fn a_generic_view_is_refused() {
        let mut fixture = Fixture::new();
        let core = fixture.raw.core;
        let ty = view_signature(&mut fixture, Mutability::Shared, true);
        fixture.declare(
            core.as_slice,
            Scheme {
                instances: Some(Instances {
                    concrete: Box::new([]),
                    generic: Some(GenericSig::default()),
                }),
                ..fixture.plain(ty, 2)
            },
        );
        assert_eq!(refusals(fixture), vec![Refused::CastWithoutHead { via: core.as_slice }]);
    }

    /// A view signature core declares with no instance adds no step.
    #[test]
    fn a_view_signature_without_instances_adds_no_step() {
        let mut fixture = Fixture::new();
        let core = fixture.raw.core;
        let ty = view_signature(&mut fixture, Mutability::Shared, true);
        fixture.declare(core.as_slice, fixture.plain(ty, 2));
        let decls = fixture.verify().expect("nothing to refuse");
        assert!(decls.casts().step_up(Head::Ref(Mutability::Shared, Base::String)).is_none());
    }

    /// types §3: a view instance's heads are what it states, and the tree
    /// judges them. `as_slice` at `&String` and `as_str` at `&String` are
    /// two steps up from `Ref(Shared, String)`.
    #[test]
    fn a_view_at_any_container_is_judged_by_the_tree() {
        let mut fixture = Fixture::new();
        let core = fixture.raw.core;
        let as_slice_ty = view_signature(&mut fixture, Mutability::Shared, true);
        let as_str_ty = view_signature(&mut fixture, Mutability::Shared, false);
        let string = fixture.ty(TyKind::String);
        let string_ref = fixture.reference(Mutability::Shared, string);
        let byte = fixture.ty(TyKind::Int(IntTy::U8));
        let bytes = fixture.ty(TyKind::Slice(byte));
        let bytes_ref = fixture.reference(Mutability::Shared, bytes);
        let str_ref = fixture.shared_str();
        let as_slice_at_string = fixture.function(&[string_ref], bytes_ref);
        let as_str_at_string = fixture.function(&[string_ref], str_ref);
        fixture.declare(core.as_slice, fixture.with_instances(as_slice_ty, 2, &[as_slice_at_string]));
        fixture.declare(core.as_str, fixture.with_instances(as_str_ty, 1, &[as_str_at_string]));
        assert_eq!(
            refusals(fixture),
            vec![Refused::TwoStepsUp {
                head: Head::Ref(Mutability::Shared, Base::String),
                via: [core.as_slice, core.as_str]
            }]
        );
    }

    /// extern-fn §8: a type term names only declared types; a context's
    /// type is one.
    #[test]
    fn a_context_naming_an_undeclared_type_is_refused() {
        let mut fixture = Fixture::new();
        let (context, ghost) = (fixture.name("m", "user"), fixture.name("m", "Ghost"));
        let ghost_ty = undeclared(&mut fixture, ghost);
        fixture.raw.contexts.push((context, ghost_ty));
        assert_eq!(
            refusals(fixture),
            vec![Refused::UndeclaredType {
                written: Written::Context(context),
                id: ghost
            }]
        );
    }

    /// extern-fn §8: a requirement's pattern is a type term too.
    #[test]
    fn a_requirement_naming_an_undeclared_type_is_refused() {
        let mut fixture = Fixture::new();
        let (f, shared, ghost) = (fixture.name("m", "f"), fixture.name("m", "hash"), fixture.name("m", "Ghost"));
        let ghost_ty = undeclared(&mut fixture, ghost);
        let var = fixture.ty(TyKind::Var(0));
        let pair = fixture.ty(TyKind::Tuple(Box::new([var, var])));
        let unit = fixture.ty(TyKind::Unit);
        let ty = fixture.function(&[pair], unit);
        fixture.declare(
            f,
            Scheme {
                requires: requiring(shared, ghost_ty),
                ..fixture.plain(ty, 1)
            },
        );
        assert_eq!(
            refusals(fixture),
            vec![Refused::UndeclaredType {
                written: Written::Function(f),
                id: ghost
            }]
        );
    }

    /// extern-fn §8: an input's type is a type term too.
    #[test]
    fn an_input_naming_an_undeclared_type_is_refused() {
        let mut fixture = Fixture::new();
        let ghost = fixture.name("m", "Ghost");
        let name = fixture.interner.intern("x");
        let ghost_ty = undeclared(&mut fixture, ghost);
        fixture.raw.inputs.push((name, ghost_ty));
        assert_eq!(
            refusals(fixture),
            vec![Refused::UndeclaredType {
                written: Written::Input(name),
                id: ghost
            }]
        );
    }

    /// effects §3, §6: only the compiler's IR writes `Handle` and `Order`.
    /// A labelled signature holding one is refused for it alone: its
    /// segments are not read.
    #[test]
    fn a_signature_holding_a_handle_is_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let s = fixture.shared_str();
        let handle = fixture.ty(TyKind::Handle(s));
        let ty = fixture.function(&[s], handle);
        let lifetimes = signature(
            vec![end(FlowEnd::Result, &[lt(0)]), end(FlowEnd::Param(0), &[lt(0)])],
            &[],
        );
        fixture.declare(f, fixture.labelled(ty, 0, lifetimes));
        assert_eq!(
            refusals(fixture),
            vec![Refused::InternalType {
                written: Written::Function(f)
            }]
        );
    }

    /// effects §3: a context's type holding the order token is refused.
    #[test]
    fn a_context_holding_the_order_token_is_refused() {
        let mut fixture = Fixture::new();
        let context = fixture.name("m", "token");
        let order = fixture.ty(TyKind::Order);
        let option = fixture.ty(TyKind::Option(order));
        fixture.raw.contexts.push((context, option));
        assert_eq!(
            refusals(fixture),
            vec![Refused::InternalType {
                written: Written::Context(context)
            }]
        );
    }

    /// extern-fn §1: a declaration names its parameters, one name per
    /// parameter. `fn(i64, i64) -> ()` named `p0` alone is refused.
    #[test]
    fn parameter_names_not_one_per_parameter_are_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let (int, unit) = (fixture.ty(TyKind::Int(IntTy::I64)), fixture.ty(TyKind::Unit));
        let ty = fixture.function(&[int, int], unit);
        let unary = fixture.function(&[int], unit);
        let one_name = fixture.param_names(unary);
        fixture.declare(
            f,
            Scheme {
                param_names: one_name,
                ..fixture.plain(ty, 0)
            },
        );
        assert_eq!(refusals(fixture), vec![Refused::ParamNamesDoNotFit { function: f }]);
    }

    /// extern-fn §1: an instance names its own parameters too. Its
    /// signature's names fit, and its own, one short, are refused.
    #[test]
    fn an_instances_parameter_names_not_one_per_parameter_are_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let (var, int, unit) = (
            fixture.ty(TyKind::Var(0)),
            fixture.ty(TyKind::Int(IntTy::I64)),
            fixture.ty(TyKind::Unit),
        );
        let own_ty = fixture.function(&[var], unit);
        let instance_ty = fixture.function(&[int], unit);
        fixture.declare(
            f,
            Scheme {
                instances: Some(Instances {
                    concrete: Box::new([InstanceSig {
                        param_names: Box::new([]),
                        ..fixture.instance(instance_ty, Box::new([]))
                    }]),
                    generic: None,
                }),
                ..fixture.plain(own_ty, 1)
            },
        );
        assert_eq!(refusals(fixture), vec![Refused::ParamNamesDoNotFit { function: f }]);
    }

    /// extern-fn §1: a type parameter carries one bound. `fn(T, U) -> T`
    /// has two type variables, and one bound is too few.
    #[test]
    fn too_few_bounds_are_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let (t, u) = (fixture.ty(TyKind::Var(0)), fixture.ty(TyKind::Var(1)));
        let ty = fixture.function(&[t, u], t);
        fixture.declare(f, fixture.plain(ty, 1));
        assert_eq!(refusals(fixture), vec![Refused::BoundsDoNotFit { function: f }]);
    }

    /// extern-fn §1: a type parameter carries one bound. `fn(T) -> T` has
    /// one type variable, and a second bound bounds none.
    #[test]
    fn too_many_bounds_are_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let t = fixture.ty(TyKind::Var(0));
        let ty = fixture.function(&[t], t);
        fixture.declare(f, fixture.plain(ty, 2));
        assert_eq!(refusals(fixture), vec![Refused::BoundsDoNotFit { function: f }]);
    }

    /// extern-fn §1: bounds are by position, one per type variable, so the
    /// variables are `0..n`. `fn(T0, T2) -> ()` with two bounds leaves the
    /// second bound on `T1`, which does not exist, and `T2` unbounded.
    #[test]
    fn bounds_over_sparse_variables_are_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let (first, third) = (fixture.ty(TyKind::Var(0)), fixture.ty(TyKind::Var(2)));
        let unit = fixture.ty(TyKind::Unit);
        let ty = fixture.function(&[first, third], unit);
        fixture.declare(f, fixture.plain(ty, 2));
        assert_eq!(refusals(fixture), vec![Refused::BoundsDoNotFit { function: f }]);
    }

    /// extern-fn §1: one bound per variable, by position, is taken in.
    #[test]
    fn one_bound_per_variable_is_taken_in() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let (t, u) = (fixture.ty(TyKind::Var(0)), fixture.ty(TyKind::Var(1)));
        let ty = fixture.function(&[t, u], t);
        fixture.declare(f, fixture.plain(ty, 2));
        assert!(fixture.verify().is_ok());
    }

    /// `Refused`: every refusal a set earns is reported, not the first.
    #[test]
    fn every_refusal_is_reported() {
        let mut fixture = Fixture::new();
        let (f, t) = (fixture.name("m", "f"), fixture.name("m", "T"));
        fixture.user_type(t, 0);
        fixture.user_type(t, 0);
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let array = fixture.ty(TyKind::Array(int, LenTerm::Known(1)));
        let ty = fixture.function(&[], array);
        fixture.declare(f, fixture.plain(ty, 0));
        let refused = refusals(fixture);
        assert!(refused.contains(&Refused::DuplicateType { id: t }));
        assert!(refused.contains(&Refused::ArrayType { declaration: f }));
        assert_eq!(refused.len(), 2);
    }
}
