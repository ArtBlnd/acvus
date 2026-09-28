use std::hash::Hash;

use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::{FxHashMap, FxHashSet};

use super::casts::{CastTree, Head, VerifiedExternCast};
use super::labels::{LabelledSignature, Lifetime, SegmentLabel};
use super::{CastRule, Core, Declarations, RawDeclarations, Requirement, Scheme, UserTypeDecl};
use crate::flows::{Alignment, Declared, Flow, FlowEnd, FlowTerm, Flows, Labelled, segments};
use crate::ty::{Holds, LifetimesOf, Mutability, ObjectTy, ParamTerm, Poly, TyId, TyKind, Types};

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
    /// Two types under one name.
    DuplicateType { id: QualifiedRef },
    /// A type with more than one identity parameter. Not stated in
    /// rfc_compiler; carried from the registry's rule until it is.
    TwoIdentities { id: QualifiedRef },
}

/// Build the verified set, or every refusal the set earns.
pub(super) fn verify(raw: RawDeclarations) -> Result<Declarations, Vec<Refused>> {
    let RawDeclarations {
        types,
        functions,
        user_types,
        casts,
        contexts,
        inputs,
        loop_markers,
        core,
    } = raw;
    let mut refused = Vec::new();

    let user_types = extern_types(user_types, &mut refused);
    let declared_order: Vec<QualifiedRef> = functions.iter().map(|(id, _)| *id).collect();
    refuse_array_types(&types, &functions, &mut refused);
    refuse_unending_requirements(&types, &functions, &mut refused);
    let overloads = overloads(&declared_order);
    let functions = unique(functions, "function");
    let casts = cast_tree(&types, &casts, &functions, &core, &mut refused);

    let mut decls = Declarations {
        types,
        functions,
        overloads,
        user_types,
        casts,
        contexts: unique(contexts, "context"),
        inputs: unique(inputs, "input"),
        loop_markers,
        core,
    };
    take_in_lifetimes(&mut decls, &declared_order, &mut refused);

    match refused.is_empty() {
        true => Ok(decls),
        false => Err(refused),
    }
}

/// Each declared entry by its key.
///
/// # Panics
/// Two entries share a key. rfc_compiler states no rule for two functions,
/// two contexts or two inputs under one name, and `Refused` has no refusal
/// for it: which one stands, or whether the set is refused, is not decided
/// here.
fn unique<K, V>(entries: Vec<(K, V)>, what: &str) -> FxHashMap<K, V>
where
    K: Eq + Hash + std::fmt::Debug,
{
    let mut by_key = FxHashMap::default();
    for (key, value) in entries {
        if by_key.contains_key(&key) {
            todo!("decl::verify: two declarations of the {what} {key:?}; no rule settles it yet")
        }
        by_key.insert(key, value);
    }
    by_key
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
        let named = by_name.entry(id.name).or_default();
        if !named.contains(id) {
            named.push(*id);
        }
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

/// The head a cast steps from or to (types §3).
///
/// # Panics
/// `ty` has no head: a type variable, `!`, the order token or a started
/// call. rfc_compiler does not say what a cast between such types steps
/// between, and `Refused` has no refusal for it.
fn head(types: &Types<Poly>, ty: TyId<Poly>) -> Head {
    match types.kind(ty) {
        TyKind::Int(width) => Head::Int(*width),
        TyKind::Float => Head::Float,
        TyKind::Char => Head::Char,
        TyKind::String => Head::String,
        TyKind::Bool => Head::Bool,
        TyKind::Unit => Head::Unit,
        TyKind::Str => Head::Str,
        TyKind::Array(..) => Head::Array,
        TyKind::Slice(_) => Head::Slice,
        TyKind::Tuple(items) => Head::Tuple(items.len()),
        TyKind::Option(_) => Head::Option,
        TyKind::Result(..) => Head::Result,
        TyKind::Object(ObjectTy::Structural(..)) => Head::Object,
        TyKind::Object(ObjectTy::Declared(id, _)) => Head::Named(*id),
        TyKind::Enum { name, .. } => Head::Enum(*name),
        TyKind::Fn { .. } => Head::Fn,
        TyKind::UserDefined { id, .. } => Head::Named(*id),
        TyKind::Ref(mutability, _) => Head::Ref(*mutability),
        TyKind::Var(_) | TyKind::Never | TyKind::Order | TyKind::Handle(_) => {
            todo!("decl::verify: a cast at {ty:?}, which has no head; no rule settles it yet")
        }
    }
}

/// The declared casts and the views as one tree (types §3): every head
/// steps up at most once, and no path of steps returns to its start.
fn cast_tree(
    types: &Types<Poly>,
    casts: &[CastRule],
    functions: &FxHashMap<QualifiedRef, Scheme>,
    core: &Core,
    refused: &mut Vec<Refused>,
) -> CastTree {
    refuse_views_until_settled(functions, core);

    let mut tree = CastTree::default();
    let mut above: FxHashMap<Head, Head> = FxHashMap::default();
    let mut from_order = Vec::new();
    for rule in casts {
        let (from, to) = (head(types, rule.from), head(types, rule.to));
        match tree.up.get(&from) {
            Some(first) => refused.push(Refused::TwoStepsUp {
                head: from,
                via: [first.rule().via, rule.via],
            }),
            None => {
                tree.up.insert(from, VerifiedExternCast(rule.clone()));
                above.insert(from, to);
                from_order.push(from);
            }
        }
    }
    refuse_cycles(&above, &from_order, refused);
    tree
}

/// The views of operators §5 are steps from `&C` to `&[T]` or `&str`, but
/// both ends have the head `Ref(Shared)`, or `Ref(Mut)` for
/// `as_slice_mut`: as heads, every view is a step from a head to itself,
/// and two views are two steps up from one head. Which heads a view steps
/// between is not decided here.
///
/// # Panics
/// A `core::as_slice`, `core::as_slice_mut` or `core::as_str` is declared.
fn refuse_views_until_settled(functions: &FxHashMap<QualifiedRef, Scheme>, core: &Core) {
    for view in [core.as_slice, core.as_slice_mut, core.as_str] {
        if functions.contains_key(&view) {
            todo!("decl::verify: the heads a view of {view:?} steps between are not settled")
        }
    }
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
    params: &[ParamTerm<Poly>],
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
    params: &[ParamTerm<Poly>],
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
        fitted.push(FittedEnd::fit(decls, end, param.ty, given.remove(&end))?);
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
        if labels.len() != segments(&decls.types, decls, ty).len() {
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
    use crate::flows::Segment;
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
            let params = params
                .iter()
                .enumerate()
                .map(|(at, ty)| ParamTerm {
                    name: self.interner.intern(&format!("p{at}")),
                    ty: *ty,
                })
                .collect();
            self.ty(TyKind::Fn {
                params,
                ret,
                effect: EffectTerm::Known(Effect::pure()),
                flows: FlowTerm::Known(Flows::none()),
                calls: Calls::Many,
                holds: Holds::Nothing,
            })
        }

        fn declare(&mut self, id: QualifiedRef, scheme: Scheme) {
            self.raw.functions.push((id, scheme));
        }

        fn verify(self) -> Result<Declarations, Vec<Refused>> {
            Declarations::verify(self.raw)
        }
    }

    fn plain(ty: TyId<Poly>) -> Scheme {
        Scheme {
            ty,
            bounds: Box::new([]),
            instances: None,
            requires: Box::new([]),
        }
    }

    fn labelled(ty: TyId<Poly>, lifetimes: LabelledSignature) -> Scheme {
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
            ..plain(ty)
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
        fixture.declare(f, plain(ty));
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
                    concrete: Box::new([InstanceSig {
                        ty: instance_ty,
                        requires: Box::new([]),
                        facts: DeclaredFacts::default(),
                    }]),
                    generic: None,
                }),
                ..plain(own_ty)
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
        let step = decls.casts().step_up(Head::Named(a)).expect("A steps up");
        assert_eq!(step.rule().via, via);
        assert!(decls.casts().step_up(Head::Named(b)).is_none());
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
                head: Head::Named(a),
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
        assert_eq!(refusals(fixture), vec![Refused::CastCycle { head: Head::Named(a) }]);
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
        assert_eq!(refusals(fixture), vec![Refused::CastCycle { head: Head::Option }]);
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
                ..plain(ty)
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
                ..plain(ty)
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
                    concrete: Box::new([InstanceSig {
                        ty: instance_ty,
                        requires: requiring(shared, pattern),
                        facts: DeclaredFacts::default(),
                    }]),
                    generic: None,
                }),
                ..plain(own_ty)
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
        fixture.declare(f, labelled(ty, lifetimes));
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
        fixture.declare(f, labelled(ty, lifetimes));
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
        fixture.declare(f, labelled(ty, lifetimes));
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
        fixture.declare(push, labelled(ty, lifetimes));
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
        fixture.declare(iter, labelled(ty, lifetimes));
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
        fixture.declare(f, labelled(ty, lifetimes));
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
        fixture.declare(f, labelled(ty, lifetimes));
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
        fixture.declare(f, labelled(ty, lifetimes));
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
        fixture.declare(f, plain(ty));
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
                        requires: Box::new([]),
                        facts: DeclaredFacts {
                            lifetimes: Some(lifetimes),
                            ..DeclaredFacts::default()
                        },
                    }]),
                    generic: None,
                }),
                ..plain(own_ty)
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
        fixture.declare(a_len, plain(ty));
        fixture.declare(b_len, plain(ty));
        let len = fixture.interner.intern("len");
        let other = fixture.interner.intern("other");
        let decls = fixture.verify().expect("nothing to refuse");
        assert_eq!(decls.overloads(len), &[a_len, b_len]);
        assert_eq!(decls.overloads(other), &[] as &[QualifiedRef]);
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
        fixture.declare(f, plain(ty));
        let refused = refusals(fixture);
        assert!(refused.contains(&Refused::DuplicateType { id: t }));
        assert!(refused.contains(&Refused::ArrayType { declaration: f }));
        assert_eq!(refused.len(), 2);
    }
}
