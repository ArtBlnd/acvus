use std::collections::hash_map::Entry;
use std::hash::Hash;

use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::{FxHashMap, FxHashSet};

use super::casts::{Base, CastTree, Head, VerifiedExternCast};
use super::conversions::{ConversionRule, VerifiedConversion};
use super::labels::{LabelledSignature, Lifetime, SegmentLabel};
use super::impl_fit::is_impl_of;
use super::{
    CastRule, ContextDecl, Core, Declarations, EntryDecl, GenericFunction, ImplKind, ImplOf, Impls, RawDeclarations, Requirement,
    Scheme, TyVarBound, UserTypeDecl, effect_variables,
};
use super::declared::{self, Atom, Subject};
use super::segments::{Segment, segments};
use crate::ty::{
    Declared, EffectTerm, FlowEnd, Holds, LenTerm, LifetimesOf, Mark, MarkTerm, Mutability,
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
    /// An impl requires an impl at types not smaller than its own
    /// (extern-fn §6).
    RequirementNotSmaller { declaration: QualifiedRef, generic_function: QualifiedRef },
    /// An end's labels are not one per segment of its declared type, or
    /// name a lifetime the signature does not have.
    LabelsDoNotFit { declaration: QualifiedRef },
    /// A declared cast, a view impl or a declared conversion whose
    /// `from` or `to` has no head: a type variable, `!`, the order token, a
    /// started call, or a reference to one of those. A tree step joins two
    /// heads (types §3), and so does a conversion (types §7).
    CastWithoutHead { via: QualifiedRef },
    /// An impl whose type is not its generic function's type at some
    /// choice of the generic function's variables (extern-fn §6).
    ImplNotOfGenericFunction { generic_function: QualifiedRef, r#impl: usize },
    /// A signature whose parameter names are not one per parameter of its
    /// function type (extern-fn §1).
    ParamNamesDoNotFit { function: QualifiedRef },
    /// Two declarations under one qualified name: reach could not tell
    /// them apart (types §6).
    DuplicateFunction { id: QualifiedRef },
    /// Two declarations of one context, which is one value of one fixed
    /// type (context).
    DuplicateContext { id: QualifiedRef },
    /// A type the declarations write names an extern-defined type no
    /// declaration gives (extern-fn §8).
    UndeclaredType { written: Written, id: QualifiedRef },
    /// A declared type holds `Handle` or `Order`, which only the compiler's
    /// IR writes (effects §3, §6).
    InternalType { written: Written },
    /// A declared type holds a reference whose target is a reference
    /// (types §4).
    ReferenceToReference { written: Written },
    /// A declared type holds `!` other than as a function signature's whole
    /// result, a call that gives no value: not supported yet.
    NeverOutsideResult { written: Written },
    /// A declared type holds a marked extern-defined type with a variable
    /// in it but its identity: only a whole concrete type carries `#`
    /// (types §8).
    MarkedOverTypeVariable { written: Written },
    /// A declared type holds a marked type as an extern-defined type's
    /// argument, which a marked type never is (types §8).
    MarkedPart { written: Written },
    /// A declared cast or a view from a marked type, or a reference to one:
    /// a cast is taken from the unmarked type, and a marked value taking it
    /// is raised (types §8).
    CastFromMarked { via: QualifiedRef },
    /// A declared cast from a type no extern declaration defines, or from a
    /// reference to one: only an extern-defined type steps up the tree
    /// (types §3).
    CastFromNonExtern { via: QualifiedRef },
    /// A declared cast or a view whose ends are of two modes: a step takes
    /// an owned value to an owned one, a `&` to a `&`, and a `&mut` to a
    /// `&mut` (types §3).
    CastChangesMode { via: QualifiedRef },
    /// A context's or an entry's declared type holds a type variable. A
    /// context is one value of one fixed type (context §2), and an entry is
    /// instantiated at the types the host declares for it (programs), so no
    /// use chooses the variable.
    TypeVariableInValueType { written: Written },
    /// A declared cast or conversion whose `via` is no declared function
    /// of type `fn(from) -> to` at the rule's own ends, or one with impls:
    /// the rule is performed by its `via`, exactly one function
    /// (types §3, §7).
    ViaNotItsFunction { written: Written },
    /// A scheme, or one of its concrete impls, whose bounds do not give each
    /// of its type variables exactly one bound (extern-fn §1: a type
    /// parameter carries one bound).
    BoundsDoNotFit { function: QualifiedRef },
    /// An effect variable a parameter gives that the call neither runs nor
    /// gives in its result: the work it names would run where no effect
    /// records it (extern-fn §2).
    EffectUnrecorded { declared: Declared, var: u32 },
    /// A `runs` entry that is no effect variable of its signature
    /// (extern-fn §2).
    RunsNoEffectVariable { declared: Declared, var: u32 },
    /// A requirement whose pattern names a type variable that is not one of
    /// its declaration's own: a declaration requires impls at its own type
    /// variables (extern-fn §6).
    RequirementForeignVariable { declaration: QualifiedRef, generic_function: QualifiedRef },
    /// A scheme or an impl whose type is not a function type: an extern fn
    /// is a function (extern-fn §1).
    NotAFunction { function: QualifiedRef },
    /// Two types under one name.
    DuplicateType { id: QualifiedRef },
    /// A generic function declared under a lang item's name: the compiler
    /// builds every lang item's generic function itself (operators §2).
    CoreGenericFunction { id: QualifiedRef },
    /// An impl of a lang item whose meaning is the language's own, which
    /// takes none (operators §2).
    TakesNoImpls { id: QualifiedRef },
    /// A declaration stating it is an impl of what is no generic function
    /// (extern-fn §6).
    ImplOfNoGenericFunction { function: QualifiedRef, generic_function: QualifiedRef },
    /// A second generic impl of one generic function: reach takes its
    /// generic impl only where no concrete one answers, so it has at most
    /// one.
    TwoGenericImpls { generic_function: QualifiedRef },
    /// A declared fact names a parameter its signature does not have: a
    /// parameter is numbered as a call's arguments are (extern-fn §3).
    FactNamesNoParameter { declared: Declared, param: usize },
    /// A place whose parameter is no reference: a place is storage a
    /// reference argument lends (extern-fn §3).
    PlaceNotLent { declared: Declared, param: usize },
    /// A place a call may write, by a `&mut` view, an open, a store or an
    /// update, lent by a shared reference, through which nothing is
    /// written (ownership §5).
    PlaceNotLentMutably { declared: Declared, param: usize },
    /// An element named by a parameter that is no `u64`: an index is a
    /// `u64` (operators §3).
    IndexNotAnIndex { declared: Declared, param: usize },
    /// An open of a place no key names: an open makes an absent entry a
    /// stated default, and an entry is the element a key names
    /// (extern-fn §3).
    OpenOfNoEntry { declared: Declared, param: usize },
    /// Two statements of the places a call touches: an access states the
    /// places a call touches, and it touches no other (extern-fn §3).
    TwoAccesses { declared: Declared },
    /// An equivalence stated on a declaration that is not an impl of
    /// `eq`: keys name one place as the key type's `eq` impl, declared an
    /// equivalence, calls them equal (extern-fn §3).
    EquivalenceNotOnEq { declared: Declared },
    /// A result stated to be what a place held, of another type than the
    /// place holds (extern-fn §3).
    ResultNotWhatThePlaceHeld { declared: Declared, param: usize },
    /// A result stated to be a declared function of what a place held,
    /// whose function is no declared `fn(&H) -> R`, `H` what the place
    /// holds and `R` the call's result (extern-fn §3).
    GivesThroughDoesNotFit { declared: Declared, through: QualifiedRef },
    /// A result stated to be a declared function of what a place held,
    /// whose function is not declared pure (extern-fn §3).
    GivesThroughNotPure { declared: Declared, through: QualifiedRef },
    /// Two statements of what a call's result is: a call has one result
    /// (extern-fn §3).
    TwoResultsGiven { declared: Declared },
    /// A total order stated on a declaration not of type `fn(&T, &T) ->
    /// int`, whose sign compares its two arguments (extern-fn §3).
    TotalOrderDoesNotFit { declared: Declared },
    /// An update whose combine is no declaration stated a combine: an
    /// update of a place is the action of a combine (extern-fn §3).
    UpdateByNoCombine { declared: Declared, combine: QualifiedRef },
    /// A combine that is no `fn(&mut S, S)` or `fn(S, S) -> S`, `S` the type
    /// the updated place holds: an update is the action of a combine on
    /// the place (extern-fn §3).
    CombineDoesNotFit { declared: Declared, combine: QualifiedRef },
    /// A combine stated on a declaration that is no associative operation
    /// on one type `S`, `fn(S, S) -> S` or `fn(&mut S, S)` (extern-fn §3).
    CombineNotAnOperation { declared: Declared },
    /// An identity that is no declared `fn() -> S` at the combine's state
    /// (extern-fn §3).
    IdentityDoesNotFit { declared: Declared, identity: QualifiedRef },
    /// An absorbing element that is no declared `fn() -> S` at the
    /// combine's state (extern-fn §3).
    AbsorbingDoesNotFit { declared: Declared, absorbing: QualifiedRef },
    /// An update declared the inverse of a function that states no update
    /// of the same place (extern-fn §3).
    InverseNotAnUpdateAtThePlace { declared: Declared, inverse: QualifiedRef },
    /// A stream declared both splittable and step by step, which are the
    /// two ways a stream yields (extern-fn §3).
    SplittableAndStepwise { declared: Declared },
    /// A closure argument that is no function of one element, or a
    /// predicate whose answer is no `bool` (extern-fn §3).
    ClosureDoesNotFit { declared: Declared, param: usize },
    /// A stream read from a form that makes no stream from another: the
    /// call's own stream's form, or a consumption, which gives a result
    /// (extern-fn §3).
    StreamOfNoStreamMadeFromAnother { declared: Declared },
    /// A consumption's initial value that is no declared `fn() -> S` and no
    /// parameter of type `S`, `S` its state (extern-fn §3).
    ConsumeInitDoesNotFit { declared: Declared, init: declared::Step },
    /// A consumption's update that is no declared function and no closure
    /// argument `fn(S, E) -> S` or `fn(&mut S, E)` over its state `S`
    /// (extern-fn §3).
    ConsumeUpdateDoesNotFit { declared: Declared, update: declared::Step },
    /// A consumption's finish that is no declared `fn(S) -> R`, `R` the
    /// call's result, or, where none is stated, a state `S` that is not
    /// the call's result (extern-fn §3).
    ConsumeFinishDoesNotFit { declared: Declared, finish: Option<QualifiedRef> },
    /// A relation over an atom that is no integer: a relation's terms are
    /// integers (extern-fn §3).
    RelationOverNonInteger { declared: Declared, atom: Atom },
    /// The length of a value whose type has no element count.
    LengthOfNoSequence { declared: Declared, subject: Subject },
    /// An atom as it stood when the call began, read through a parameter
    /// that is no `&mut`: only a `&mut` argument's value changes during a
    /// call (extern-fn §3).
    OldOfNoMutableReference { declared: Declared, param: usize },
}

/// Where a declaration writes a type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Written {
    Function(QualifiedRef),
    Context(QualifiedRef),
    /// An entry's input or result, by the function it starts at.
    Entry(QualifiedRef),
    Cast(QualifiedRef),
    Conversion(QualifiedRef),
}

/// Build the verified set, or every refusal the set earns.
pub(super) fn verify(raw: RawDeclarations) -> Result<Declarations, Vec<Refused>> {
    let RawDeclarations {
        mut types,
        functions,
        generic_functions,
        user_types,
        casts,
        conversions,
        contexts,
        entries,
        loop_markers,
        core,
    } = raw;
    let mut refused = Vec::new();

    let (functions, generic_functions) = with_lang_items(&mut types, &core, functions, generic_functions, &mut refused);
    let user_types = extern_types(user_types, &mut refused);
    let unreadable = refuse_unwritable_types(
        &types,
        DeclaredTypes {
            functions: &functions,
            generic_functions: &generic_functions,
            user_types: &user_types,
            casts: &casts,
            conversions: &conversions,
            contexts: &contexts,
            entries: &entries,
        },
        &mut refused,
    );
    refuse_type_variables_in_value_types(&types, &contexts, &entries, &mut refused);
    refuse_array_types(&types, &functions, &generic_functions, &mut refused);
    refuse_unending_requirements(&types, &functions, &mut refused);
    refuse_unfit_bounds(&types, &functions, &generic_functions, &mut refused);
    refuse_unrecorded_effects(&types, &functions, &mut refused);
    refuse_foreign_requirements(&types, &functions, &mut refused);
    refuse_unfit_signatures(&types, &functions, &generic_functions, &mut refused);
    let generic_order = first_of_each(generic_functions.iter().map(|(id, _)| *id));
    let mut generic_functions = first_stands(generic_functions, &mut refused, |id| Refused::DuplicateFunction { id });
    let function_order = first_of_each(functions.iter().map(|(id, _)| *id));
    let mut functions = first_stands(functions, &mut refused, |id| Refused::DuplicateFunction { id });
    // A function and a generic function under one name: neither stands, as
    // nothing orders one before the other.
    for id in &function_order {
        if generic_functions.contains_key(id) {
            functions.remove(id);
            generic_functions.remove(id);
            refused.push(Refused::DuplicateFunction { id: *id });
        }
    }
    let function_order: Vec<QualifiedRef> = function_order.into_iter().filter(|id| functions.contains_key(id)).collect();
    let generic_order: Vec<QualifiedRef> = generic_order.into_iter().filter(|id| generic_functions.contains_key(id)).collect();
    // A bare name reaches a generic function or a declaration that is no
    // impl: an impl is reached only through its generic function
    // (extern-fn §6).
    let reachable = generic_order
        .iter()
        .copied()
        .chain(function_order.iter().copied().filter(|id| functions.get(id).is_some_and(|scheme| scheme.impl_of.is_none())));
    let overloads = overloads(&first_of_each(reachable));
    let contexts = first_stands(contexts, &mut refused, |id| Refused::DuplicateContext { id });
    let impls = impls_of(&function_order, &functions, &generic_functions, &core, &mut refused);
    assert_generic_impls_are_their_signatures(&types, &generic_functions, &impls, &functions);
    let misfits = refuse_misfit_impls(&types, &generic_order, &generic_functions, &impls, &functions, &mut refused);
    let known = declared::Functions::new(&functions, &generic_functions, &impls);
    declared::refuse_ill_formed(&mut types, &core, &known, &function_order, &mut refused);
    refuse_vias_not_their_functions(&types, &functions, &casts, &conversions, &mut refused);
    let casts = cast_tree(&types, &casts, &functions, &impls, &misfits, &core, &mut refused);
    let conversions = conversions_by_head(&types, conversions, &mut refused);
    let lifts = generic_functions
        .iter()
        .filter_map(|(id, generic)| super::lift::lift_of(&types, generic).map(|lift| (*id, lift)))
        .collect();

    let context_identities = contexts
        .iter()
        .flat_map(|(id, declared)| super::identity_variables(&types, [declared.ty]).into_iter().map(|placeholder| ((*id, placeholder), crate::ty::IdentityId::alloc())))
        .collect();

    let mut impls = impls;
    let generic_functions = generic_functions
        .into_iter()
        .map(|(id, generic)| (id, (generic, impls.remove(&id).unwrap_or_default())))
        .collect();
    let decls = Declarations {
        types,
        functions,
        generic_functions,
        overloads,
        user_types,
        casts,
        conversions,
        lifts,
        contexts,
        context_identities,
        entries,
        loop_markers,
        core,
    };
    let lifetimes_readable: Vec<QualifiedRef> = function_order
        .into_iter()
        .filter(|id| !unreadable.contains(id))
        .collect();
    refuse_unfit_lifetimes(&decls, &lifetimes_readable, &mut refused);

    match refused.is_empty() {
        true => Ok(decls),
        false => Err(refused),
    }
}

/// Every lang item's generic function, built by the compiler from
/// operators §2's table, in front of the declared generic functions. A
/// function or a generic function declared under a lang item's name is
/// refused and left out.
fn with_lang_items(
    types: &mut Types<Poly>,
    core: &Core,
    functions: Vec<(QualifiedRef, Scheme)>,
    generic_functions: Vec<(QualifiedRef, GenericFunction)>,
    refused: &mut Vec<Refused>,
) -> (Vec<(QualifiedRef, Scheme)>, Vec<(QualifiedRef, GenericFunction)>) {
    let functions = not_under_core_names(core, functions, refused);
    let declared = not_under_core_names(core, generic_functions, refused);
    let all = core.generic_functions(types).into_iter().chain(declared).collect();
    (functions, all)
}

/// `declared` without what is declared under a lang item's name, which is
/// refused.
fn not_under_core_names<D>(core: &Core, declared: Vec<(QualifiedRef, D)>, refused: &mut Vec<Refused>) -> Vec<(QualifiedRef, D)> {
    let mut kept = Vec::new();
    for (id, declaration) in declared {
        match core.names(id) {
            true => refused.push(Refused::CoreGenericFunction { id }),
            false => kept.push((id, declaration)),
        }
    }
    kept
}

/// Each generic function's impls: every declaration that states it is one,
/// the concrete ones in declaration order. An impl of what is no generic
/// function is refused, as is an impl of a lang item that takes none, and
/// a second generic impl of one generic function.
fn impls_of(
    function_order: &[QualifiedRef],
    functions: &FxHashMap<QualifiedRef, Scheme>,
    generic_functions: &FxHashMap<QualifiedRef, GenericFunction>,
    core: &Core,
    refused: &mut Vec<Refused>,
) -> FxHashMap<QualifiedRef, Impls> {
    let mut concrete: FxHashMap<QualifiedRef, Vec<QualifiedRef>> = FxHashMap::default();
    let mut generic: FxHashMap<QualifiedRef, QualifiedRef> = FxHashMap::default();
    for function in function_order {
        let Some(ImplOf { generic_function, kind }) = functions.get(function).and_then(|scheme| scheme.impl_of) else {
            continue;
        };
        if !generic_functions.contains_key(&generic_function) {
            refused.push(Refused::ImplOfNoGenericFunction { function: *function, generic_function });
            continue;
        }
        if core.takes_no_impls(generic_function) {
            refused.push(Refused::TakesNoImpls { id: generic_function });
            continue;
        }
        match kind {
            ImplKind::Concrete => concrete.entry(generic_function).or_default().push(*function),
            ImplKind::Generic => match generic.entry(generic_function) {
                Entry::Occupied(_) => refused.push(Refused::TwoGenericImpls { generic_function }),
                Entry::Vacant(vacant) => {
                    vacant.insert(*function);
                }
            },
        }
    }
    let mut impls: FxHashMap<QualifiedRef, Impls> = concrete
        .into_iter()
        .map(|(id, concrete)| (id, Impls { concrete: concrete.into(), generic: None }))
        .collect();
    for (id, function) in generic {
        impls.entry(id).or_default().generic = Some(function);
    }
    impls
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

/// extern-fn §1: no declaration carries an array type, and no generic
/// function's signature does.
fn refuse_array_types(
    types: &Types<Poly>,
    functions: &[(QualifiedRef, Scheme)],
    generic_functions: &[(QualifiedRef, GenericFunction)],
    refused: &mut Vec<Refused>,
) {
    let mut seen = FxHashSet::default();
    let signatures = functions.iter().map(|(id, scheme)| (id, scheme.ty)).chain(generic_functions.iter().map(|(id, generic)| (id, generic.ty)));
    for (id, ty) in signatures {
        if holds_array(types, ty, &mut seen) {
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

/// extern-fn §1: a type parameter carries one bound. A signature's type
/// variables are `0..n` for its `n` bounds, each one mentioned by its type.
fn refuse_unfit_bounds(
    types: &Types<Poly>,
    functions: &[(QualifiedRef, Scheme)],
    generic_functions: &[(QualifiedRef, GenericFunction)],
    refused: &mut Vec<Refused>,
) {
    let signatures = functions
        .iter()
        .map(|(id, scheme)| (id, scheme.ty, &scheme.bounds[..]))
        .chain(generic_functions.iter().map(|(id, generic)| (id, generic.ty, &generic.bounds[..])));
    for (function, ty, bounds) in signatures {
        if !one_bound_per_variable(types, ty, bounds) {
            refused.push(Refused::BoundsDoNotFit { function: *function });
        }
    }
}

fn one_bound_per_variable(types: &Types<Poly>, ty: TyId<Poly>, bounds: &[TyVarBound]) -> bool {
    let mut variables = FxHashSet::default();
    type_variables(types, ty, &mut FxHashSet::default(), &mut variables);
    let bounded = bounds.len();
    variables.len() == bounded
        && variables
            .iter()
            .all(|var| usize::try_from(*var).is_ok_and(|at| at < bounded))
}

/// extern-fn §2: a call's effect is its own joined with the effect
/// variables of the work it runs, `runs`, each an effect variable of its
/// signature; an effect variable a parameter gives is run, or given in the
/// result, or both. A signature that is no function type is refused as
/// `NotAFunction` alone.
fn refuse_unrecorded_effects(types: &Types<Poly>, functions: &[(QualifiedRef, Scheme)], refused: &mut Vec<Refused>) {
    for (function, scheme) in functions {
        let declared = Declared { function: *function };
        let TyKind::Fn { params, ret, .. } = types.kind(scheme.ty) else {
            continue;
        };
        let of_signature = effect_variables(types, [scheme.ty]);
        for var in scheme.runs.iter().filter(|var| !of_signature.contains(var)) {
            refused.push(Refused::RunsNoEffectVariable { declared, var: *var });
        }
        let returned = effect_variables(types, [*ret]);
        let mut unrecorded: Vec<u32> = effect_variables(types, params.iter().copied())
            .into_iter()
            .filter(|var| !scheme.runs.contains(var) && !returned.contains(var))
            .collect();
        unrecorded.sort_unstable();
        for var in unrecorded {
            refused.push(Refused::EffectUnrecorded { declared, var });
        }
    }
}

/// extern-fn §6: a declaration requires impls at its own type variables.
fn refuse_foreign_requirements(types: &Types<Poly>, functions: &[(QualifiedRef, Scheme)], refused: &mut Vec<Refused>) {
    for (declaration, scheme) in functions {
        let mut own = FxHashSet::default();
        type_variables(types, scheme.ty, &mut FxHashSet::default(), &mut own);
        for requirement in &scheme.requires {
            let mut named = FxHashSet::default();
            type_variables(types, requirement.pattern, &mut FxHashSet::default(), &mut named);
            if !named.is_subset(&own) {
                refused.push(Refused::RequirementForeignVariable {
                    declaration: *declaration,
                    generic_function: requirement.generic_function,
                });
            }
        }
    }
}

/// extern-fn §1: an extern fn is a function, so its signature, and a
/// generic function's, is a function type; and it names its parameters,
/// one name per parameter of that function type. A signature of another
/// type is refused as `NotAFunction` alone, since it has no parameters to
/// name.
fn refuse_unfit_signatures(
    types: &Types<Poly>,
    functions: &[(QualifiedRef, Scheme)],
    generic_functions: &[(QualifiedRef, GenericFunction)],
    refused: &mut Vec<Refused>,
) {
    let signatures = functions
        .iter()
        .map(|(id, scheme)| (id, scheme.ty, &scheme.param_names[..]))
        .chain(generic_functions.iter().map(|(id, generic)| (id, generic.ty, &generic.param_names[..])));
    for (function, ty, names) in signatures {
        match types.kind(ty) {
            TyKind::Fn { params, .. } if params.len() == names.len() => {}
            TyKind::Fn { .. } => refused.push(Refused::ParamNamesDoNotFit { function: *function }),
            _ => refused.push(Refused::NotAFunction { function: *function }),
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

/// extern-fn §6: a requirement is answered by an impl, so only an impl's
/// requirements lead on to further ones, and an impl requires impls only
/// at types smaller than its own, so the requirements a call reaches end.
/// A declaration that is no impl is reached by its calls alone: its
/// requirements start a chain and are no link of one, so they are not
/// measured.
fn refuse_unending_requirements(
    types: &Types<Poly>,
    functions: &[(QualifiedRef, Scheme)],
    refused: &mut Vec<Refused>,
) {
    let mut sizes = TreeSizes {
        types,
        known: FxHashMap::default(),
    };
    for (id, scheme) in functions.iter().filter(|(_, scheme)| scheme.impl_of.is_some()) {
        sizes.refuse_not_smaller(*id, scheme.ty, &scheme.requires, refused);
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
                    generic_function: requirement.generic_function,
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
    generic_functions: &'d [(QualifiedRef, GenericFunction)],
    user_types: &'d FxHashMap<QualifiedRef, UserTypeDecl>,
    casts: &'d [CastRule],
    conversions: &'d [ConversionRule],
    contexts: &'d [(QualifiedRef, ContextDecl)],
    entries: &'d [EntryDecl],
}

/// Every type a scheme writes: its signature and every requirement's
/// pattern.
fn scheme_types(scheme: &Scheme) -> impl Iterator<Item = TyId<Poly>> + '_ {
    std::iter::once(scheme.ty).chain(scheme.requires.iter().map(|requirement| requirement.pattern))
}

/// Every type the declarations write, each with where it is written.
fn written_types<'d>(declared: &DeclaredTypes<'d>) -> impl Iterator<Item = (Written, TyId<Poly>)> + 'd {
    let functions = declared
        .functions
        .iter()
        .flat_map(|(id, scheme)| scheme_types(scheme).map(|ty| (Written::Function(*id), ty)))
        .chain(declared.generic_functions.iter().map(|(id, generic)| (Written::Function(*id), generic.ty)));
    let casts = declared
        .casts
        .iter()
        .flat_map(|rule| [(Written::Cast(rule.via), rule.from), (Written::Cast(rule.via), rule.to)]);
    let conversions = declared.conversions.iter().flat_map(|rule| {
        [
            (Written::Conversion(rule.via), rule.from),
            (Written::Conversion(rule.via), rule.to),
        ]
    });
    let contexts = declared.contexts.iter().map(|(id, declared)| (Written::Context(*id), declared.ty));
    functions
        .chain(casts)
        .chain(conversions)
        .chain(contexts)
        .chain(declared.entries.iter().flat_map(entry_types))
}

/// extern-fn §8: a type term names only declared types; effects §3, §6:
/// `Order` and `Handle` are written by the compiler's IR alone; types §4: a
/// reference's target is never a reference. Every type the declarations
/// write is held to all three. Returns the functions that break any, whose
/// segments are not read. A `!` written anywhere but as a function
/// signature's whole result is refused as not supported yet.
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
        let marked_over_variable = terms.marked_over_variable.then_some(Refused::MarkedOverTypeVariable { written });
        let marked_part = terms.marked_part.then_some(Refused::MarkedPart { written });
        let never = never_outside_result(types, written, ty).then_some(Refused::NeverOutsideResult { written });
        for refusal in marked_over_variable.into_iter().chain(marked_part).chain(never) {
            if !reported.contains(&refusal) {
                reported.push(refusal.clone());
                refused.push(refusal);
            }
        }
    }
    unreadable
}

/// Whether `ty`, written at `written`, holds a `!` other than a function
/// signature's whole result, the one place a declaration writes it.
fn never_outside_result(types: &Types<Poly>, written: Written, ty: TyId<Poly>) -> bool {
    match (written, types.kind(ty)) {
        (Written::Function(_), TyKind::Fn { params, ret, .. }) => {
            params.iter().any(|param| holds_never(types, *param)) || (!matches!(types.kind(*ret), TyKind::Never) && holds_never(types, *ret))
        }
        _ => holds_never(types, ty),
    }
}

/// Whether `ty` holds `!` anywhere.
fn holds_never(types: &Types<Poly>, ty: TyId<Poly>) -> bool {
    let mut seen = FxHashSet::default();
    let mut stack = vec![ty];
    while let Some(at) = stack.pop() {
        if !seen.insert(at) {
            continue;
        }
        match types.kind(at) {
            TyKind::Never => return true,
            kind => kind.for_each_child(|child| stack.push(child)),
        }
    }
    false
}

/// Every type an entry writes: its inputs' and its result's.
fn entry_types(entry: &EntryDecl) -> impl Iterator<Item = (Written, TyId<Poly>)> + '_ {
    let written = Written::Entry(entry.function);
    entry.inputs.iter().map(|(_, ty)| *ty).chain(entry.result).map(move |ty| (written, ty))
}

/// context §2: a context's declared type is one fixed type; programs: an
/// entry is instantiated at the types the host declares. So neither holds
/// a type variable.
fn refuse_type_variables_in_value_types(
    types: &Types<Poly>,
    contexts: &[(QualifiedRef, ContextDecl)],
    entries: &[EntryDecl],
    refused: &mut Vec<Refused>,
) {
    let contexts = contexts.iter().map(|(id, declared)| (Written::Context(*id), declared.ty));
    for (written, ty) in contexts.chain(entries.iter().flat_map(entry_types)) {
        if holds_type_variable(types, ty) {
            refused.push(Refused::TypeVariableInValueType { written });
        }
    }
}

fn is_marked(types: &Types<Poly>, ty: TyId<Poly>) -> bool {
    matches!(types.kind(ty), TyKind::UserDefined { marked: MarkTerm::Known(Mark::Marked), .. })
}

/// Whether `ty` is a marked type or a reference lending one.
fn is_or_lends_marked(types: &Types<Poly>, ty: TyId<Poly>) -> bool {
    match types.kind(ty) {
        TyKind::Ref(_, target) => is_marked(types, *target),
        _ => is_marked(types, ty),
    }
}

fn holds_type_variable(types: &Types<Poly>, ty: TyId<Poly>) -> bool {
    let mut seen = FxHashSet::default();
    let mut stack = vec![ty];
    while let Some(at) = stack.pop() {
        if !seen.insert(at) {
            continue;
        }
        match types.kind(at) {
            TyKind::Var(_) => return true,
            kind => kind.for_each_child(|child| stack.push(child)),
        }
    }
    false
}

/// Whether `ty` holds a variable of any kind but an identity, which is a
/// binding each call makes, not a variable a use settles (types §3, §8).
fn holds_variable(types: &Types<Poly>, ty: TyId<Poly>) -> bool {
    let mut seen = FxHashSet::default();
    let mut stack = vec![ty];
    while let Some(at) = stack.pop() {
        if !seen.insert(at) {
            continue;
        }
        let kind = types.kind(at);
        let variable = match kind {
            TyKind::Var(_) => true,
            TyKind::Array(_, len) => matches!(len, LenTerm::Var(_)),
            TyKind::Fn { effect, .. } => matches!(effect, EffectTerm::Var(_)),
            TyKind::UserDefined { effect, .. } => matches!(effect, Some(EffectTerm::Var(_))),
            _ => false,
        };
        if variable {
            return true;
        }
        kind.for_each_child(|child| stack.push(child));
    }
    false
}

/// What a type term names: every extern-defined type, each once, whether
/// it holds a type only the IR writes, whether it holds a reference to a
/// reference, and whether it holds a marked type over a type variable or
/// as a part no marked type may be (types §8).
#[derive(Default)]
struct TypeTerms {
    seen: FxHashSet<TyId<Poly>>,
    named: Vec<QualifiedRef>,
    internal: bool,
    reference_to_reference: bool,
    marked_over_variable: bool,
    marked_part: bool,
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
                TyKind::Order | TyKind::Handle(_) | TyKind::Closures(_) => self.internal = true,
                TyKind::Ref(_, target) if matches!(types.kind(*target), TyKind::Ref(..)) => {
                    self.reference_to_reference = true
                }
                _ => {}
            }
            if is_marked(types, ty) && holds_variable(types, ty) {
                self.marked_over_variable = true;
            }
            // Only an extern-defined type's argument is never a marked type;
            // a reference, a function type and a transparent structured
            // type hold one as any part (types §8).
            let argument = matches!(kind, TyKind::UserDefined { .. });
            kind.for_each_child(|child| {
                self.marked_part |= argument && is_marked(types, child);
                self.read(types, child)
            });
        })
    }
}

/// extern-fn §6: every concrete impl of a generic function is the
/// generic function's type at some choice of its variables. Returns the misfits,
/// by function and impl.
fn refuse_misfit_impls(
    types: &Types<Poly>,
    generic_order: &[QualifiedRef],
    generic_functions: &FxHashMap<QualifiedRef, GenericFunction>,
    impls: &FxHashMap<QualifiedRef, Impls>,
    functions: &FxHashMap<QualifiedRef, Scheme>,
    refused: &mut Vec<Refused>,
) -> FxHashSet<(QualifiedRef, usize)> {
    let mut misfits = FxHashSet::default();
    for generic_function in generic_order {
        let (Some(generic), Some(impls)) = (generic_functions.get(generic_function), impls.get(generic_function)) else {
            continue;
        };
        for (r#impl, sig) in impls.concrete.iter().enumerate().filter_map(|(at, id)| functions.get(id).map(|sig| (at, sig))) {
            if !is_impl_of(types, generic.ty, sig.ty) {
                misfits.insert((*generic_function, r#impl));
                refused.push(Refused::ImplNotOfGenericFunction {
                    generic_function: *generic_function,
                    r#impl,
                });
            }
        }
    }
    misfits
}

/// A generic impl's signature is its generic function's, the choice of no
/// variable (extern-fn §6): its function type is the same but for its
/// effect, which is the impl's own, and so are its bounds and parameter
/// names.
///
/// # Panics
/// A generic impl's signature is another: whoever built the declarations,
/// the compiler for a lang item or an extern fn's attribute for its own,
/// built a generic impl that is none.
fn assert_generic_impls_are_their_signatures(
    types: &Types<Poly>,
    generic_functions: &FxHashMap<QualifiedRef, GenericFunction>,
    impls: &FxHashMap<QualifiedRef, Impls>,
    functions: &FxHashMap<QualifiedRef, Scheme>,
) {
    let but_effect = |ty: TyId<Poly>| match types.kind(ty) {
        TyKind::Fn { params, ret, flows, calls, holds, .. } => Some((params, ret, flows, calls, holds)),
        _ => None,
    };
    for (id, impls) in impls {
        let (Some(r#impl), Some(generic)) = (impls.generic, generic_functions.get(id)) else {
            continue;
        };
        let Some(scheme) = functions.get(&r#impl) else {
            continue;
        };
        assert!(
            but_effect(scheme.ty) == but_effect(generic.ty)
                && scheme.bounds == generic.bounds
                && scheme.param_names == generic.param_names,
            "the generic impl {impl:?} of {id:?} is not its generic function's signature: the declarations were built wrong"
        );
    }
}

/// The head a tree step joins at `ty` (types §3), or `None` where there
/// is none: a type variable, `!`, the order token or a started call, or a
/// reference to one of those.
fn head(types: &Types<Poly>, ty: TyId<Poly>) -> Option<Head> {
    match types.kind(ty) {
        TyKind::Ref(mutability, target) => Some(Head::Ref(*mutability, base(types, *target)?)),
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
        TyKind::Option(..) => Base::Option,
        TyKind::Result(..) => Base::Result,
        TyKind::Object { .. } => Base::Object,
        TyKind::Enum { name, .. } => Base::Enum(*name),
        TyKind::Fn { .. } => Base::Fn,
        TyKind::UserDefined { id, .. } => Base::Named(*id),
        TyKind::Struct { .. } => panic!("a declared type holds no struct: a struct is its module's own"),
        TyKind::Ref(..) => panic!(
            "a reference's target is never a reference (types §4): a rule with such an end is refused as \
             ReferenceToReference and left out of the tree"
        ),
        TyKind::Var(_) | TyKind::Never | TyKind::Order | TyKind::Handle(_) | TyKind::Closures(_) => return None,
    })
}

/// Whether a value of `head` is an extern-defined type or a reference to
/// one, the only heads a step goes up from.
fn is_extern_head(head: Head) -> bool {
    matches!(head, Head::Base(Base::Named(_)) | Head::Ref(_, Base::Named(_)))
}

/// The mode a value of `head` is held in: owned, or lent at a mutability.
fn mode_of(head: Head) -> Option<Mutability> {
    match head {
        Head::Base(_) => None,
        Head::Ref(mutability, _) => Some(mutability),
    }
}

/// A rule the tree is built from: a declared cast, or a view at the
/// concrete impl it is (operators §5).
struct TreeRule {
    rule: CastRule,
    view_impl: Option<usize>,
}

/// A step of the tree: a cast from one head up to another.
struct Step {
    from: Head,
    to: Head,
    rule: TreeRule,
}

impl Step {
    /// `rule` as a step, or `CastWithoutHead` where an end has no head.
    fn of(types: &Types<Poly>, rule: TreeRule) -> Result<Step, Refused> {
        match (head(types, rule.rule.from), head(types, rule.rule.to)) {
            (Some(from), Some(to)) => Ok(Step { from, to, rule }),
            (None, _) | (_, None) => Err(Refused::CastWithoutHead { via: rule.rule.via }),
        }
    }
}

/// The declared casts and the views as one tree (types §3): every head
/// steps up at most once, and no path of steps returns to its start. A
/// step goes up from an extern-defined type, or a reference to one, and
/// keeps its mode. A view from any other container is no step: its impl is
/// a function the program calls by name (operators §5).
fn cast_tree(
    types: &Types<Poly>,
    casts: &[CastRule],
    functions: &FxHashMap<QualifiedRef, Scheme>,
    impls: &FxHashMap<QualifiedRef, Impls>,
    misfits: &FxHashSet<(QualifiedRef, usize)>,
    core: &Core,
    refused: &mut Vec<Refused>,
) -> CastTree {
    let declared = casts.iter().cloned().map(|rule| Ok(TreeRule { rule, view_impl: None }));
    let rules = declared
        .chain(view_rules(types, functions, impls, misfits, core))
        .filter(|rule| {
            !rule
                .as_ref()
                .is_ok_and(|tree_rule| ends_in_reference_to_reference(types, [tree_rule.rule.from, tree_rule.rule.to]))
        });
    let mut steps = Vec::new();
    for rule in rules {
        match rule.and_then(|rule| Step::of(types, rule)) {
            Ok(step) if is_or_lends_marked(types, step.rule.rule.from) => refused.push(Refused::CastFromMarked { via: step.rule.rule.via }),
            Ok(step) if !is_extern_head(step.from) => {
                if step.rule.view_impl.is_none() {
                    refused.push(Refused::CastFromNonExtern { via: step.rule.rule.via });
                }
            }
            Ok(step) if mode_of(step.from) != mode_of(step.to) => refused.push(Refused::CastChangesMode { via: step.rule.rule.via }),
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
                via: [first.rule().via, rule.rule.via],
            }),
            None => {
                let TreeRule { rule, view_impl } = rule;
                tree.up.insert(from, VerifiedExternCast { rule, view_impl });
                above.insert(from, to);
                from_order.push(from);
            }
        }
    }
    refuse_cycles(&above, &from_order, refused);
    tree
}

/// types §3, §7: a declared cast or conversion is performed by its `via`,
/// so the `via` is a declared function of type `fn(from) -> to` at the
/// rule's own ends. A rule is declared for one pair of types, so its `via`
/// is exactly one function: no generic function, and no impl, which is
/// reached only through its generic function.
fn refuse_vias_not_their_functions(
    types: &Types<Poly>,
    functions: &FxHashMap<QualifiedRef, Scheme>,
    casts: &[CastRule],
    conversions: &[ConversionRule],
    refused: &mut Vec<Refused>,
) {
    let casts = casts.iter().map(|rule| (Written::Cast(rule.via), rule.via, rule.from, rule.to));
    let conversions = conversions
        .iter()
        .map(|rule| (Written::Conversion(rule.via), rule.via, rule.from, rule.to));
    for (written, via, from, to) in casts.chain(conversions) {
        let performs = functions.get(&via).is_some_and(|scheme| match types.kind(scheme.ty) {
            TyKind::Fn { params, ret, .. } => scheme.impl_of.is_none() && params[..] == [from] && *ret == to,
            _ => false,
        });
        if !performs {
            refused.push(Refused::ViaNotItsFunction { written });
        }
    }
}

/// Whether a cast's or a conversion's `from` or `to` is a reference to a
/// reference, which is refused as `ReferenceToReference` and has no head
/// (types §4).
fn ends_in_reference_to_reference(types: &Types<Poly>, ends: [TyId<Poly>; 2]) -> bool {
    ends.into_iter().any(|end| match types.kind(end) {
        TyKind::Ref(_, target) => matches!(types.kind(*target), TyKind::Ref(..)),
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

/// Each impl of `core::as_slice`, `core::as_slice_mut` and
/// `core::as_str` as a cast from its parameter to its result, at that impl
/// (operators §3, §5); `cast_tree` keeps only those from an extern-defined
/// container. A generic impl's container is a variable, which has no head.
/// An impl refused as not of its generic function is left out.
///
/// # Panics
/// A fitting impl is not a function of one parameter. A fitting impl is
/// its view's type at some choice of the view's variables, and the
/// compiler builds each view's type from operators §2 as a function of one
/// parameter (`Core::generic_functions`); an impl of any other shape is
/// refused as `ImplNotOfGenericFunction` and left out above.
fn view_rules(
    types: &Types<Poly>,
    functions: &FxHashMap<QualifiedRef, Scheme>,
    impls: &FxHashMap<QualifiedRef, Impls>,
    misfits: &FxHashSet<(QualifiedRef, usize)>,
    core: &Core,
) -> Vec<Result<TreeRule, Refused>> {
    let mut rules = Vec::new();
    for via in [core.as_slice, core.as_slice_mut, core.as_str] {
        let Some(impls) = impls.get(&via) else {
            continue;
        };
        let fitting = impls
            .concrete
            .iter()
            .enumerate()
            .filter(|(at, _)| !misfits.contains(&(via, *at)))
            .filter_map(|(at, id)| functions.get(id).map(|r#impl| (at, r#impl)));
        for (at, r#impl) in fitting {
            let TyKind::Fn { params, ret, .. } = types.kind(r#impl.ty) else {
                panic!("a fitting view impl is a function type, as the compiler builds each view (operators §2)")
            };
            let [param] = &params[..] else {
                panic!("a fitting view impl takes one parameter, as the compiler builds each view (operators §2)")
            };
            rules.push(Ok(TreeRule {
                rule: CastRule {
                    from: *param,
                    to: *ret,
                    via,
                },
                view_impl: Some(at),
            }));
        }
        if impls.generic.is_some() {
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

/// extern-fn §4: a declaration's lifetimes fit its signature, one label per
/// segment of each end and each order on lifetimes the labels use. They
/// stay on the declaration, and each call of it reads them there
/// (`call_flows`).
fn refuse_unfit_lifetimes(decls: &Declarations, declared_order: &[QualifiedRef], refused: &mut Vec<Refused>) {
    for function in declared_order {
        let Some(scheme) = decls.functions.get(function) else {
            continue;
        };
        let Some(lifetimes) = &scheme.facts.lifetimes else {
            continue;
        };
        let fits = match decls.types.kind(scheme.ty) {
            TyKind::Fn { params, ret, .. } => fitted_ends(decls, params, *ret, lifetimes)
                .and_then(|ends| outlives_closure(lifetimes, &ends))
                .is_ok(),
            _ => false,
        };
        if !fits {
            refused.push(Refused::LabelsDoNotFit { declaration: *function });
        }
    }
}

pub(super) struct LabelsDoNotFit;

/// One end of a signature with its labels, one per segment, and for each
/// segment whether the call may write into it as an output.
pub(super) struct FittedEnd<'s> {
    pub(super) end: FlowEnd,
    pub(super) labels: &'s [SegmentLabel],
    pub(super) written: Vec<bool>,
}

/// The result and each parameter, with the labels the signature gives
/// them, when each end has exactly one label per segment and the
/// signature lists each end at most once. The callee's captures have no
/// segments, since a named function captures nothing, so they fit only
/// unlabelled and take part in no flow.
pub(super) fn fitted_ends<'s>(
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
pub(super) fn outlives_closure(
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

/// For each segment of a parameter's declared type, in `segments`
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
        TyKind::Order | TyKind::Handle(_) | TyKind::Closures(_) | TyKind::Struct { .. } => {
            panic!("a declared type holds no Handle, Order, Closures or struct: a struct is its module's own")
        }
        TyKind::Fn { holds, .. } => out.push(under_mut || *holds == Holds::Mut),
        TyKind::Var(_) => out.push(under_mut),
        TyKind::Ref(mutability, target) => {
            out.push(under_mut);
            push_written(types, lifetimes, *target, under_mut || *mutability == Mutability::Mut, out);
        }
        TyKind::UserDefined { id, type_args, .. } => {
            out.extend(std::iter::repeat_n(under_mut, lifetimes.lifetimes_of(*id)));
            for arg in type_args {
                push_written(types, lifetimes, *arg, under_mut, out);
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
        TyKind::Object { fields, .. } => {
            for (_, field) in fields {
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
    use crate::decl::{
        CallFlows, DeclaredFacts, EndLabels, GenericFunction, ImplKind, ImplOf, LoopMarkers, Outlives, TyVarBound, call_flows,
        declared_fn_flows,
    };
    use crate::ty::{Calls, Concrete, Effect, EffectTerm, FlowTerm, IntTy, LenTerm, Ty};

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
                    generic_functions: Vec::new(),
                    types: Types::new(),
                    functions: Vec::new(),
                    user_types: Vec::new(),
                    casts: Vec::new(),
                    conversions: Vec::new(),
                    contexts: Vec::new(),
                    entries: Vec::new(),
                    loop_markers: LoopMarkers::default(),
                    core,
                },
            }
        }

        fn name(&self, namespace: &str, name: &str) -> QualifiedRef {
            QualifiedRef::qualified(self.interner.namespace(&[self.interner.intern(namespace)]), self.interner.intern(name))
        }

        fn ty(&mut self, kind: TyKind<Poly>) -> TyId<Poly> {
            self.raw.types.intern(kind)
        }

        fn reference(&mut self, mutability: Mutability, target: TyId<Poly>) -> TyId<Poly> {
            self.ty(TyKind::Ref(mutability, target))
        }

        fn shared_str(&mut self) -> TyId<Poly> {
            let str_ty = self.ty(TyKind::Str);
            self.reference(Mutability::Shared, str_ty)
        }

        fn user_type(&mut self, id: QualifiedRef, lifetimes: usize) -> TyId<Poly> {
            self.raw.user_types.push(UserTypeDecl {
                id,
                type_params: Box::new([]),
                carries_effect: false,
                carries_identity: false,
                lifetimes,
            });
            self.ty(TyKind::UserDefined {
                id,
                type_args: Box::new([]),
                effect: None,
                identity: None,
                marked: MarkTerm::Known(Mark::Unmarked),
            })
        }

        fn function(&mut self, params: &[TyId<Poly>], ret: TyId<Poly>) -> TyId<Poly> {
            self.ty(TyKind::Fn {
                params: params.into(),
                ret,
                effect: EffectTerm::Known(Effect::pure()),
                flows: FlowTerm::Known(crate::decl::declared_fn_flows()),
                calls: Calls::Many,
                holds: Holds::Nothing,
                values: (),
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
                impl_of: None,
                requires: Box::new([]),
                facts: DeclaredFacts::default(),
                runs: Box::new([]),
                awaits: false,
                specialized: Box::new([]),
            }
        }

        /// A scheme without impls at `ty`, stating `lifetimes`.
        fn labelled(&self, ty: TyId<Poly>, variables: usize, lifetimes: LabelledSignature) -> Scheme {
            Scheme {
                facts: DeclaredFacts {
                    lifetimes: Some(lifetimes),
                    ..DeclaredFacts::default()
                },
                ..self.plain(ty, variables)
            }
        }

        /// The generic function `id` at `ty` whose one impl, the generic
        /// one, states `lifetimes`.
        fn labelled_generic(&mut self, id: QualifiedRef, ty: TyId<Poly>, variables: usize, lifetimes: LabelledSignature) {
            let generic_impl = Scheme {
                impl_of: Some(ImplOf { generic_function: id, kind: ImplKind::Generic }),
                facts: DeclaredFacts {
                    lifetimes: Some(lifetimes),
                    ..DeclaredFacts::default()
                },
                ..self.plain(ty, variables)
            };
            self.declare_impls(id, ty, variables, vec![generic_impl]);
        }

        /// The generic function `id` whose concrete impls are `tys`.
        fn with_impls(&mut self, id: QualifiedRef, own_ty: TyId<Poly>, variables: usize, tys: &[TyId<Poly>]) {
            let impls = tys.iter().map(|ty| self.r#impl(id, *ty, Box::new([]))).collect();
            self.declare_impls(id, own_ty, variables, impls);
        }

        /// The impl at `at` of `generic`, a declaration of its own named
        /// `generic#at`.
        fn impl_id(&self, generic: QualifiedRef, at: usize) -> QualifiedRef {
            QualifiedRef::qualified(generic.namespace, self.interner.intern(&format!("{}#{at}", self.interner.resolve(generic.name))))
        }

        /// A concrete impl of `generic_function` at `ty` stating no fact.
        fn r#impl(&self, generic_function: QualifiedRef, ty: TyId<Poly>, requires: Box<[Requirement]>) -> Scheme {
            let mut variables = FxHashSet::default();
            super::type_variables(&self.raw.types, ty, &mut FxHashSet::default(), &mut variables);
            Scheme {
                ty,
                bounds: vec![TyVarBound::Any; variables.len()].into(),
                param_names: self.param_names(ty),
                impl_of: Some(ImplOf { generic_function, kind: ImplKind::Concrete }),
                requires,
                facts: DeclaredFacts::default(),
                runs: Box::new([]),
                awaits: false,
                specialized: Box::new([]),
            }
        }

        /// The generic function `id` at `own_ty`, whose type variables are
        /// `0..variables`, each bounded by any type, and its `impls`, each
        /// a declaration of its own at the impl id of its position.
        fn declare_impls(&mut self, id: QualifiedRef, own_ty: TyId<Poly>, variables: usize, impls: Vec<Scheme>) {
            let generic = GenericFunction {
                ty: own_ty,
                param_names: self.param_names(own_ty),
                bounds: vec![TyVarBound::Any; variables].into(),
                specialized: Box::new([]),
            };
            self.raw.generic_functions.push((id, generic));
            for (at, scheme) in impls.into_iter().enumerate() {
                let impl_id = self.impl_id(id, at);
                self.raw.functions.push((impl_id, scheme));
            }
        }

        fn declare(&mut self, id: QualifiedRef, scheme: Scheme) {
            self.raw.functions.push((id, scheme));
        }

        /// Gives the lang item `item` the concrete impls `tys`, stating no
        /// fact.
        fn give(&mut self, item: QualifiedRef, tys: &[TyId<Poly>]) {
            for (at, ty) in tys.iter().enumerate() {
                let impl_id = self.impl_id(item, at);
                let scheme = self.r#impl(item, *ty, Box::new([]));
                self.raw.functions.push((impl_id, scheme));
            }
        }

        fn verify(self) -> Result<Declarations, Vec<Refused>> {
            Declarations::verify(self.raw)
        }

        /// `via`, declared as the function `fn(from) -> to` that performs a
        /// cast or a conversion between those ends (types §3, §7).
        fn performing(&mut self, via: QualifiedRef, from: TyId<Poly>, to: TyId<Poly>) {
            let ty = self.function(&[from], to);
            let variables = [from, to]
                .into_iter()
                .filter_map(|end| match self.raw.types.kind(end) {
                    TyKind::Var(at) => Some(*at as usize + 1),
                    _ => None,
                })
                .max()
                .unwrap_or(0);
            self.declare(via, self.plain(ty, variables));
        }

        /// A declared cast from `from` to `to`, performed by `via`.
        fn cast(&mut self, from: TyId<Poly>, to: TyId<Poly>, via: QualifiedRef) {
            self.performing(via, from, to);
            self.raw.casts.push(CastRule { from, to, via });
        }

        /// A declared conversion from `from` to `to`, performed by `via`.
        fn conversion(&mut self, from: TyId<Poly>, to: TyId<Poly>, via: QualifiedRef) {
            self.performing(via, from, to);
            self.raw.conversions.push(ConversionRule { from, to, via });
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

    fn own(function: QualifiedRef) -> Declared {
        Declared { function }
    }

    /// Flows into `to`, one list of `(parameter, position)` per position of
    /// `to`.
    fn into(to: FlowEnd, positions: &[&[(usize, usize)]]) -> (FlowEnd, Box<[Box<[(usize, usize)]>]>) {
        (to, positions.iter().map(|takes| Box::<[(usize, usize)]>::from(*takes)).collect())
    }

    /// The concrete `&str`.
    fn concrete_shared_str(types: &mut Types<Concrete>) -> Ty {
        let str_ty = types.intern(TyKind::Str);
        types.intern(TyKind::Ref(Mutability::Shared, str_ty))
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

    /// context §2: a context's type is one fixed type; programs: an entry
    /// is instantiated at the types the host declares. So a type variable
    /// in either is refused, in an entry's input or its result.
    #[test]
    fn a_context_or_an_entry_holding_a_type_variable_is_refused() {
        let mut fixture = Fixture::new();
        let context = fixture.name("m", "c");
        let (by_input, by_result) = (fixture.name("m", "f"), fixture.name("m", "g"));
        let input = fixture.interner.intern("i");
        let var = fixture.ty(TyKind::Var(0));
        let option = fixture.ty(TyKind::Option(var));
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        fixture.raw.contexts.push((context, ContextDecl { ty: option, volatile: false }));
        fixture.raw.entries.push(EntryDecl { function: by_input, inputs: vec![(input, var)], result: Some(int) });
        fixture.raw.entries.push(EntryDecl { function: by_result, inputs: vec![(input, int)], result: Some(option) });
        assert_eq!(
            refusals(fixture),
            vec![
                Refused::TypeVariableInValueType {
                    written: Written::Context(context)
                },
                Refused::TypeVariableInValueType {
                    written: Written::Entry(by_input)
                },
                Refused::TypeVariableInValueType {
                    written: Written::Entry(by_result)
                },
            ]
        );
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

    /// extern-fn §1: no declaration carries an array type, an impl's
    /// type included.
    #[test]
    fn an_impl_holding_an_array_is_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let var = fixture.ty(TyKind::Var(0));
        let unit = fixture.ty(TyKind::Unit);
        let own_ty = fixture.function(&[var], unit);
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let array = fixture.ty(TyKind::Array(int, LenTerm::Var(0)));
        let impl_ty = fixture.function(&[array], unit);
        let at_f = fixture.r#impl(f, impl_ty, Box::new([]));
        fixture.declare_impls(f, own_ty, 1, vec![at_f]);
        let declaration = fixture.impl_id(f, 0);
        assert_eq!(refusals(fixture), vec![Refused::ArrayType { declaration }]);
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

    /// types §3: a declared cast from `S` to `T` is `S`'s one step up, and
    /// `T` steps nowhere unless a cast says so.
    #[test]
    fn a_declared_cast_is_its_heads_one_step_up() {
        let mut fixture = Fixture::new();
        let (a, b, via) = (fixture.name("m", "A"), fixture.name("m", "B"), fixture.name("m", "a_to_b"));
        let from = fixture.user_type(a, 0);
        let to = fixture.user_type(b, 0);
        fixture.cast(from, to, via);
        let decls = fixture.verify().expect("one step up is a tree");
        let step = decls.casts().step_up(Head::Base(Base::Named(a))).expect("A steps up");
        assert_eq!(step.rule().via, via);
        assert!(decls.casts().step_up(Head::Base(Base::Named(b))).is_none());
    }

    /// `#A`, the declared extern type `id` of no parameter, marked.
    fn marked(fixture: &mut Fixture, id: QualifiedRef) -> TyId<Poly> {
        let marked = MarkTerm::Known(Mark::Marked);
        fixture.ty(TyKind::UserDefined { id, type_args: Box::new([]), effect: None, identity: None, marked })
    }

    /// The extern type `id` of one parameter, `V<T>`, declared.
    fn declare_one_parameter(fixture: &mut Fixture, id: QualifiedRef) {
        fixture.raw.user_types.push(UserTypeDecl {
            id,
            type_params: Box::new([TyVarBound::Any]),
            carries_effect: false,
            carries_identity: false,
            lifetimes: 0,
        });
    }

    fn one_parameter(fixture: &mut Fixture, id: QualifiedRef, arg: TyId<Poly>, mark: Mark) -> TyId<Poly> {
        let marked = MarkTerm::Known(mark);
        fixture.ty(TyKind::UserDefined { id, type_args: Box::new([arg]), effect: None, identity: None, marked })
    }

    /// types §8: `#A` stands as a binding's type, a parameter's or a
    /// result's, a function type's included, as what a reference lends, and
    /// as a part of a transparent structured type.
    #[test]
    fn a_marked_type_stands_as_a_binding_a_parameter_a_result_a_lent_type_or_a_part() {
        let mut fixture = Fixture::new();
        let a = fixture.name("m", "A");
        fixture.user_type(a, 0);
        let marked_a = marked(&mut fixture, a);
        let lent = fixture.reference(Mutability::Shared, marked_a);
        let inner = fixture.function(&[marked_a], marked_a);
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let tuple = fixture.ty(TyKind::Tuple(Box::new([marked_a, int])));
        let option = fixture.ty(TyKind::Option(marked_a));
        let ty = fixture.function(&[marked_a, lent, inner, option], tuple);
        let f = fixture.name("m", "f");
        fixture.declare(f, fixture.plain(ty, 0));
        fixture.raw.contexts.push((fixture.name("m", "c"), ContextDecl { ty: marked_a, volatile: false }));
        assert!(fixture.verify().is_ok());
    }

    /// types §8: only a whole concrete type carries `#`.
    #[test]
    fn a_marked_type_over_a_type_variable_is_refused() {
        let mut fixture = Fixture::new();
        let v = fixture.name("m", "V");
        declare_one_parameter(&mut fixture, v);
        let t = fixture.ty(TyKind::Var(0));
        let marked_v = one_parameter(&mut fixture, v, t, Mark::Marked);
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let ty = fixture.function(&[marked_v], int);
        let f = fixture.name("m", "f");
        fixture.declare(f, fixture.plain(ty, 1));
        assert_eq!(refusals(fixture), vec![Refused::MarkedOverTypeVariable { written: Written::Function(f) }]);
    }

    /// types §8: a marked type is never an extern-defined type's argument.
    #[test]
    fn a_marked_type_argument_is_refused() {
        let mut fixture = Fixture::new();
        let (a, v) = (fixture.name("m", "A"), fixture.name("m", "V"));
        fixture.user_type(a, 0);
        declare_one_parameter(&mut fixture, v);
        let marked_a = marked(&mut fixture, a);
        let argument = one_parameter(&mut fixture, v, marked_a, Mark::Unmarked);
        let c = fixture.name("m", "c");
        fixture.raw.contexts.push((c, ContextDecl { ty: argument, volatile: false }));
        assert_eq!(refusals(fixture), vec![Refused::MarkedPart { written: Written::Context(c) }]);
    }

    /// types §8: a cast is taken from the unmarked type, which a marked
    /// value taking it is raised to, so none is declared from a marked one.
    #[test]
    fn a_cast_from_a_marked_type_is_refused() {
        let mut fixture = Fixture::new();
        let (a, b, via) = (fixture.name("m", "A"), fixture.name("m", "B"), fixture.name("m", "a_to_b"));
        fixture.user_type(a, 0);
        let to = fixture.user_type(b, 0);
        let from = marked(&mut fixture, a);
        fixture.cast(from, to, via);
        assert_eq!(refusals(fixture), vec![Refused::CastFromMarked { via }]);
    }

    /// Only an extern-defined type, or a reference to one, steps up the
    /// tree: a declared cast from a number, from a reference to a built-in
    /// `String` or from an `Option` is refused.
    #[test]
    fn a_cast_from_a_type_no_declaration_defines_is_refused() {
        let mut fixture = Fixture::new();
        let a = fixture.name("m", "A");
        let a_ty = fixture.user_type(a, 0);
        let (i32_, i64_) = (fixture.ty(TyKind::Int(IntTy::I32)), fixture.ty(TyKind::Int(IntTy::I64)));
        let string = fixture.ty(TyKind::String);
        let string_ref = fixture.reference(Mutability::Shared, string);
        let str_ref = fixture.shared_str();
        let option = fixture.ty(TyKind::Option(i64_));
        let (widen, as_str, unwrap_a) = (fixture.name("m", "widen"), fixture.name("m", "as_str"), fixture.name("m", "unwrap_a"));
        fixture.cast(i32_, i64_, widen);
        fixture.cast(string_ref, str_ref, as_str);
        fixture.cast(option, a_ty, unwrap_a);
        assert_eq!(
            refusals(fixture),
            vec![
                Refused::CastFromNonExtern { via: widen },
                Refused::CastFromNonExtern { via: as_str },
                Refused::CastFromNonExtern { via: unwrap_a },
            ]
        );
    }

    /// A step keeps its value owned, `&` or `&mut` as it was: a cast from an
    /// owned value to a reference, from a reference to an owned value, or
    /// between `&` and `&mut`, is refused.
    #[test]
    fn a_cast_that_changes_its_mode_is_refused() {
        let mut fixture = Fixture::new();
        let names = ["A", "B", "C", "D", "E", "F", "G", "H"].map(|name| fixture.name("m", name));
        let types = names.map(|name| fixture.user_type(name, 0));
        let [a, b, c, d, e, f, g, h] = types;
        let b_ref = fixture.reference(Mutability::Shared, b);
        let c_ref = fixture.reference(Mutability::Shared, c);
        let e_ref = fixture.reference(Mutability::Shared, e);
        let f_mut = fixture.reference(Mutability::Mut, f);
        let g_mut = fixture.reference(Mutability::Mut, g);
        let h_ref = fixture.reference(Mutability::Shared, h);
        let vias = ["lend", "own", "exclusive", "share"].map(|via| fixture.name("m", via));
        fixture.cast(a, b_ref, vias[0]);
        fixture.cast(c_ref, d, vias[1]);
        fixture.cast(e_ref, f_mut, vias[2]);
        fixture.cast(g_mut, h_ref, vias[3]);
        assert_eq!(refusals(fixture), vias.map(|via| Refused::CastChangesMode { via }).to_vec());
    }

    /// A step goes up from an extern-defined type, or a reference to one, to
    /// a type of any kind in the same mode: `A` to `String`, `&A` to `&str`
    /// and `&mut V` to `&mut [i64]`. `A` and `&A` are two heads, each with
    /// its one step up.
    #[test]
    fn a_cast_from_an_extern_type_keeps_its_mode_and_ends_at_any_type() {
        let mut fixture = Fixture::new();
        let (a, v) = (fixture.name("m", "A"), fixture.name("m", "V"));
        let (a_ty, v_ty) = (fixture.user_type(a, 0), fixture.user_type(v, 0));
        let string = fixture.ty(TyKind::String);
        let a_ref = fixture.reference(Mutability::Shared, a_ty);
        let str_ref = fixture.shared_str();
        let v_mut = fixture.reference(Mutability::Mut, v_ty);
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let slice = fixture.ty(TyKind::Slice(int));
        let slice_mut = fixture.reference(Mutability::Mut, slice);
        let (to_string, view, view_mut) = (fixture.name("m", "to_string"), fixture.name("m", "view"), fixture.name("m", "view_mut"));
        fixture.cast(a_ty, string, to_string);
        fixture.cast(a_ref, str_ref, view);
        fixture.cast(v_mut, slice_mut, view_mut);
        let decls = fixture.verify().expect("each step is from an extern type and keeps its mode");
        let via_from = |head| decls.casts().step_up(head).map(|step| step.rule().via);
        assert_eq!(via_from(Head::Base(Base::Named(a))), Some(to_string));
        assert_eq!(via_from(Head::Ref(Mutability::Shared, Base::Named(a))), Some(view));
        assert_eq!(via_from(Head::Ref(Mutability::Mut, Base::Named(v))), Some(view_mut));
    }

    /// types §3, §7: a cast or a conversion is performed by its `via`, so
    /// a `via` that is not declared, or not of type `fn(from) -> to`, is
    /// refused.
    #[test]
    fn a_via_that_is_not_its_function_is_refused() {
        let mut fixture = Fixture::new();
        let (i32_, i64_, string) = (
            fixture.ty(TyKind::Int(IntTy::I32)),
            fixture.ty(TyKind::Int(IntTy::I64)),
            fixture.ty(TyKind::String),
        );
        let (undeclared_via, other_type) = (fixture.name("m", "widen"), fixture.name("m", "parse"));
        let (a, b) = (fixture.name("m", "A"), fixture.name("m", "B"));
        let (a_ty, b_ty) = (fixture.user_type(a, 0), fixture.user_type(b, 0));
        fixture.raw.casts.push(CastRule {
            from: a_ty,
            to: b_ty,
            via: undeclared_via,
        });
        fixture.performing(other_type, string, i32_);
        fixture.raw.conversions.push(ConversionRule {
            from: string,
            to: i64_,
            via: other_type,
        });
        assert_eq!(
            refusals(fixture),
            vec![
                Refused::ViaNotItsFunction {
                    written: Written::Cast(undeclared_via)
                },
                Refused::ViaNotItsFunction {
                    written: Written::Conversion(other_type)
                },
            ]
        );
    }

    /// types §3, §7: a cast's `via` is exactly one function, so a generic
    /// function with impls is refused as its `via`, even at its type.
    #[test]
    fn a_via_with_impls_is_refused() {
        let mut fixture = Fixture::new();
        let (a, b) = (fixture.name("m", "A"), fixture.name("m", "B"));
        let (a_ty, b_ty) = (fixture.user_type(a, 0), fixture.user_type(b, 0));
        let widen = fixture.name("m", "widen");
        let ty = fixture.function(&[a_ty], b_ty);
        fixture.with_impls(widen, ty, 0, &[ty]);
        fixture.raw.casts.push(CastRule {
            from: a_ty,
            to: b_ty,
            via: widen,
        });
        assert_eq!(
            refusals(fixture),
            vec![Refused::ViaNotItsFunction {
                written: Written::Cast(widen)
            }]
        );
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
        fixture.cast(a_ty, b_ty, to_b);
        fixture.cast(a_ty, c_ty, to_c);
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
        fixture.cast(a_ty, b_ty, to_b);
        fixture.cast(b_ty, a_ty, to_a);
        assert_eq!(refusals(fixture), vec![Refused::CastCycle { head: Head::Base(Base::Named(a)) }]);
    }

    /// types §3: a cast between two types of one head is a step from that
    /// head to itself, a path back to its start: `Box<i64>` to
    /// `Box<String>`.
    #[test]
    fn a_cast_within_one_head_is_refused() {
        let mut fixture = Fixture::new();
        let (boxed, via) = (fixture.name("m", "Box"), fixture.name("m", "widen"));
        fixture.raw.user_types.push(UserTypeDecl {
            id: boxed,
            type_params: Box::new([TyVarBound::Any]),
            carries_effect: false,
            carries_identity: false,
            lifetimes: 0,
        });
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let string = fixture.ty(TyKind::String);
        let mut box_of = |argument| {
            let unmarked = MarkTerm::Known(Mark::Unmarked);
            fixture.ty(TyKind::UserDefined { id: boxed, type_args: Box::new([argument]), effect: None, identity: None, marked: unmarked })
        };
        let (from, to) = (box_of(int), box_of(string));
        fixture.cast(from, to, via);
        assert_eq!(refusals(fixture), vec![Refused::CastCycle { head: Head::Base(Base::Named(boxed)) }]);
    }

    fn requiring(generic_function: QualifiedRef, pattern: TyId<Poly>) -> Box<[Requirement]> {
        Box::new([Requirement {
            generic_function,
            pattern,
            fact: None,
        }])
    }

    /// extern-fn §6: an impl requires impls only at types smaller than its
    /// own. The impl `fn(Option<i64>) -> ()` has four nodes, and so has the
    /// pattern `(i64, i64, i64)`.
    #[test]
    fn an_impl_requirement_as_large_as_the_impl_is_refused() {
        let mut fixture = Fixture::new();
        let (f, shared) = (fixture.name("m", "f"), fixture.name("m", "hash"));
        let var = fixture.ty(TyKind::Var(0));
        let unit = fixture.ty(TyKind::Unit);
        let own_ty = fixture.function(&[var], unit);
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let option = fixture.ty(TyKind::Option(int));
        let impl_ty = fixture.function(&[option], unit);
        let pattern = fixture.ty(TyKind::Tuple(Box::new([int, int, int])));
        let at_option = fixture.r#impl(f, impl_ty, requiring(shared, pattern));
        fixture.declare_impls(f, own_ty, 1, vec![at_option]);
        let declaration = fixture.impl_id(f, 0);
        assert_eq!(
            refusals(fixture),
            vec![Refused::RequirementNotSmaller {
                declaration,
                generic_function: shared
            }]
        );
    }

    /// extern-fn §6: a declaration that is no impl is reached by its calls
    /// alone, so its requirement, though no smaller than its own
    /// `fn(T) -> ()`, is taken in: it starts a chain of requirements and is
    /// no link of one.
    #[test]
    fn a_requirement_of_a_declaration_that_is_no_impl_is_not_measured() {
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
        assert!(fixture.verify().is_ok());
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

    /// extern-fn §6: an impl's requirement is measured against the
    /// impl's own type. `(i64, i64, i64)` has four nodes: fewer than the
    /// impl `fn(Option<Option<i64>>) -> ()`, five, though not fewer than
    /// the scheme's `fn(T) -> ()`, three.
    #[test]
    fn an_impl_requirement_is_measured_against_the_impl() {
        let mut fixture = Fixture::new();
        let (f, shared) = (fixture.name("m", "f"), fixture.name("m", "hash"));
        let var = fixture.ty(TyKind::Var(0));
        let unit = fixture.ty(TyKind::Unit);
        let own_ty = fixture.function(&[var], unit);
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let inner = fixture.ty(TyKind::Option(int));
        let outer = fixture.ty(TyKind::Option(inner));
        let impl_ty = fixture.function(&[outer], unit);
        let pattern = fixture.ty(TyKind::Tuple(Box::new([int, int, int])));
        let at_f = fixture.r#impl(f, impl_ty, requiring(shared, pattern));
        fixture.declare_impls(f, own_ty, 1, vec![at_f]);
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
        let mut types = Types::new();
        let s = concrete_shared_str(&mut types);
        assert_eq!(
            call_flows(&decls, own(f), &types, &[s, s], s),
            Some(CallFlows { into: vec![into(FlowEnd::Result, &[&[(0, 0)]])] })
        );
    }

    /// extern-fn §4: a generic impl's lifetimes are its function's own
    /// signature's, as a declaration's without impls are, and they stay on
    /// the declaration after intake.
    #[test]
    fn a_generic_impls_lifetimes_are_read_at_its_functions_own_signature() {
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
        fixture.labelled_generic(f, ty, 0, lifetimes.clone());
        let generic_impl = fixture.impl_id(f, 0);
        let decls = fixture.verify().expect("the labels fit");
        let mut types = Types::new();
        let s = concrete_shared_str(&mut types);
        assert_eq!(
            call_flows(&decls, own(generic_impl), &types, &[s, s], s),
            Some(CallFlows { into: vec![into(FlowEnd::Result, &[&[(0, 0)]])] })
        );
        let generic = decls.scheme(generic_impl).expect("a generic impl");
        assert_eq!(generic.facts.lifetimes, Some(lifetimes));
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
        let mut types = Types::new();
        let s = concrete_shared_str(&mut types);
        assert_eq!(
            call_flows(&decls, own(f), &types, &[s, s, s], s),
            Some(CallFlows { into: vec![into(FlowEnd::Result, &[&[(0, 0), (1, 0), (2, 0)]])] })
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
        let mut types = Types::new();
        let s = concrete_shared_str(&mut types);
        assert_eq!(
            call_flows(&decls, own(f), &types, &[s, s], s),
            Some(CallFlows { into: vec![into(FlowEnd::Result, &[&[(0, 0)]])] })
        );
    }

    /// ownership §5, `FlowEnd`: `fn push<'a, T>(v: &'a mut [T], x: T)`,
    /// called at `T = (&str, &str)`. A parameter's output is what its `&mut`
    /// names, here `v`'s `T` part, and one type variable's parts meet k-th
    /// to k-th: each position of `v`'s `T` part takes the same position of
    /// `x`'s and of its own, and `v`'s reference takes nothing.
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
        let mut types = Types::new();
        let s = concrete_shared_str(&mut types);
        let pair = types.intern(TyKind::Tuple(Box::new([s, s])));
        let pairs = types.intern(TyKind::Slice(pair));
        let v = types.intern(TyKind::Ref(Mutability::Mut, pairs));
        let unit = types.intern(TyKind::Unit);
        assert_eq!(
            call_flows(&decls, own(push), &types, &[v, pair], unit),
            Some(CallFlows {
                into: vec![into(FlowEnd::Param(0), &[&[], &[(0, 1), (1, 0)], &[(0, 2), (1, 1)]])]
            })
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
        let mut types = Types::new();
        let string = types.intern(TyKind::String);
        let s = types.intern(TyKind::Ref(Mutability::Shared, string));
        let result = types.intern(TyKind::UserDefined {
            id: iter_ty,
            type_args: Box::new([]),
            effect: None,
            identity: None,
            marked: MarkTerm::Known(Mark::Unmarked),
        });
        assert_eq!(
            call_flows(&decls, own(iter), &types, &[s], result),
            Some(CallFlows { into: vec![into(FlowEnd::Result, &[&[(0, 0)]])] })
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
    /// flowing to every output: a call of it reads no flows off it.
    #[test]
    fn a_declaration_without_lifetimes_has_every_flow() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let s = fixture.shared_str();
        let ty = fixture.function(&[s], s);
        fixture.declare(f, fixture.plain(ty, 0));
        let decls = fixture.verify().expect("nothing to refuse");
        let mut types = Types::new();
        let s = concrete_shared_str(&mut types);
        assert_eq!(call_flows(&decls, own(f), &types, &[s], s), None);
    }

    /// extern-fn §4: an impl's lifetimes are read at a call of that impl,
    /// and stay on it after intake.
    #[test]
    fn an_impls_lifetimes_are_read_at_a_call_of_the_impl() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let var = fixture.ty(TyKind::Var(0));
        let own_ty = fixture.function(&[var], var);
        let s = fixture.shared_str();
        let impl_ty = fixture.function(&[s], s);
        let lifetimes = signature(
            vec![end(FlowEnd::Result, &[lt(0)]), end(FlowEnd::Param(0), &[lt(0)])],
            &[],
        );
        let at_f = Scheme {
            ty: impl_ty,
            bounds: Box::new([]),
            param_names: fixture.param_names(impl_ty),
            impl_of: Some(ImplOf { generic_function: f, kind: ImplKind::Concrete }),
            requires: Box::new([]),
            facts: DeclaredFacts {
                lifetimes: Some(lifetimes.clone()),
                ..DeclaredFacts::default()
            },
            runs: Box::new([]),
            awaits: false,
            specialized: Box::new([]),
        };
        fixture.declare_impls(f, own_ty, 1, vec![at_f]);
        let impl_at = fixture.impl_id(f, 0);
        let decls = fixture.verify().expect("the labels fit");
        let r#impl = decls.scheme(impl_at).expect("declared");
        let at_impl = Declared { function: impl_at };
        let mut types = Types::new();
        let s = concrete_shared_str(&mut types);
        assert_eq!(
            call_flows(&decls, at_impl, &types, &[s], s),
            Some(CallFlows { into: vec![into(FlowEnd::Result, &[&[(0, 0)]])] })
        );
        assert_eq!(r#impl.facts.lifetimes, Some(lifetimes));
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
        let core_len = fixture.raw.core.len;
        let decls = fixture.verify().expect("nothing to refuse");
        assert_eq!(decls.overloads(len), &[core_len, a_len, b_len]);
        assert_eq!(decls.overloads(other), &[] as &[QualifiedRef]);
    }

    /// operators §5, types §3: `as_slice` at `&Vec`, an extern-defined
    /// container, is a step from `Ref(Shared, Named(Vec))` to
    /// `Ref(Shared, Slice)`. `as_str` at `&String`, a built-in one, is no
    /// step: the program calls it by name.
    #[test]
    fn a_view_is_a_step_from_an_extern_containers_reference_to_its_view() {
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
        fixture.give(core.as_slice, &[as_slice_at_vec]);
        fixture.give(core.as_str, &[as_str_at_string]);
        let decls = fixture.verify().expect("a view from a built-in container is no step and refused nothing");
        let from_vec = decls
            .casts()
            .step_up(Head::Ref(Mutability::Shared, Base::Named(vec)))
            .expect("&Vec steps up");
        assert_eq!(from_vec.rule().via, core.as_slice);
        assert!(decls.casts().step_up(Head::Ref(Mutability::Shared, Base::String)).is_none());
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
        fixture.give(core.as_slice, &[as_slice_at_vec]);
        fixture.cast(vec_ref, other_ref, to_other);
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
        fixture.cast(from, to, via);
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
        fixture.conversion(from, to, via);
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
        let a = fixture.name("m", "A");
        let (string, a_ty, i64_) = (fixture.ty(TyKind::String), fixture.user_type(a, 0), fixture.ty(TyKind::Int(IntTy::I64)));
        fixture.cast(a_ty, i64_, widen);
        let parse_a = fixture.name("m", "parse_a");
        for (from, via) in [(string, parse), (string, parse_again), (a_ty, parse_a)] {
            fixture.conversion(from, i64_, via);
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
        assert_eq!(vias(Head::Base(Base::Named(a))), vec![parse_a]);
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
            vec![
                Refused::ReferenceToReference {
                    written: Written::Conversion(via)
                },
                Refused::ViaNotItsFunction {
                    written: Written::Conversion(via)
                },
            ]
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
            vec![
                Refused::UndeclaredType {
                    written: Written::Conversion(via),
                    id: ghost
                },
                Refused::ViaNotItsFunction {
                    written: Written::Conversion(via)
                },
            ]
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
        fixture.raw.contexts.push((id, ContextDecl { ty: int, volatile: false }));
        fixture.raw.contexts.push((id, ContextDecl { ty: string, volatile: true }));
        assert_eq!(refusals(fixture), vec![Refused::DuplicateContext { id }]);
    }

    /// context §3: whether a context is volatile is its declaration's fact.
    #[test]
    fn a_context_declares_whether_it_is_volatile() {
        let mut fixture = Fixture::new();
        let (held, volatile, absent) = (fixture.name("m", "held"), fixture.name("m", "clock"), fixture.name("m", "none"));
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        fixture.raw.contexts.push((held, ContextDecl { ty: int, volatile: false }));
        fixture.raw.contexts.push((volatile, ContextDecl { ty: int, volatile: true }));
        let decls = Declarations::verify(fixture.raw).expect("two contexts under two names verify");
        assert_eq!(decls.context_is_volatile(held), Some(false));
        assert_eq!(decls.context_is_volatile(volatile), Some(true));
        assert_eq!(decls.context_is_volatile(absent), None);
        assert_eq!(decls.context(volatile), Some(int));
    }

    fn undeclared(fixture: &mut Fixture, id: QualifiedRef) -> TyId<Poly> {
        fixture.ty(TyKind::UserDefined {
            id,
            type_args: Box::new([]),
            effect: None,
            identity: None,
            marked: MarkTerm::Known(Mark::Unmarked),
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
            vec![
                Refused::UndeclaredType {
                    written: Written::Cast(via),
                    id: ghost
                },
                Refused::ViaNotItsFunction {
                    written: Written::Cast(via)
                },
            ]
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

    /// extern-fn §6: an impl is its generic function's type at some choice
    /// of the generic function's variables. `fn(i64) -> String` is no choice of `T` in
    /// `fn(T) -> T`; `fn(i64) -> i64` is.
    #[test]
    fn an_impl_not_of_its_generic_function_is_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let var = fixture.ty(TyKind::Var(0));
        let own_ty = fixture.function(&[var], var);
        let (int, string) = (fixture.ty(TyKind::Int(IntTy::I64)), fixture.ty(TyKind::String));
        let fits = fixture.function(&[int], int);
        let misfit = fixture.function(&[int], string);
        fixture.with_impls(f, own_ty, 1, &[fits, misfit]);
        assert_eq!(
            refusals(fixture),
            vec![Refused::ImplNotOfGenericFunction {
                generic_function: f,
                r#impl: 1
            }]
        );
    }

    /// extern-fn §6: a choice fixes each variable once, so `fn(&T) -> &T`
    /// has no impl `fn(&i64) -> &mut i64`, nor one with a different
    /// constructor.
    #[test]
    fn an_impl_is_matched_constructor_by_constructor() {
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
        fixture.with_impls(f, own_ty, 1, &[fits, wrong_mutability, wrong_constructor]);
        assert_eq!(
            refusals(fixture),
            vec![
                Refused::ImplNotOfGenericFunction {
                    generic_function: f,
                    r#impl: 1
                },
                Refused::ImplNotOfGenericFunction {
                    generic_function: f,
                    r#impl: 2
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
            vec![
                Refused::ReferenceToReference {
                    written: Written::Cast(via)
                },
                Refused::ViaNotItsFunction {
                    written: Written::Cast(via)
                },
            ]
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

    /// types §3: a view lang item's generic impl has a variable for its
    /// container, which has no head.
    #[test]
    fn a_generic_view_is_refused() {
        let mut fixture = Fixture::new();
        let core = fixture.raw.core;
        let (_, generic) = core
            .generic_functions(&mut fixture.raw.types)
            .into_iter()
            .find(|(id, _)| *id == core.as_slice)
            .expect("the compiler builds `as_slice`");
        let generic_impl = Scheme {
            ty: generic.ty,
            param_names: generic.param_names,
            bounds: generic.bounds,
            impl_of: Some(ImplOf { generic_function: core.as_slice, kind: ImplKind::Generic }),
            requires: Box::new([]),
            facts: DeclaredFacts::default(),
            runs: Box::new([]),
            awaits: false,
            specialized: Box::new([]),
        };
        let at = fixture.impl_id(core.as_slice, 0);
        fixture.raw.functions.push((at, generic_impl));
        assert_eq!(refusals(fixture), vec![Refused::CastWithoutHead { via: core.as_slice }]);
    }

    /// A view lang item given no impl adds no step.
    #[test]
    fn a_view_lang_item_without_impls_adds_no_step() {
        let fixture = Fixture::new();
        let decls = fixture.verify().expect("nothing to refuse");
        assert!(decls.casts().step_up(Head::Ref(Mutability::Shared, Base::String)).is_none());
    }

    /// types §3: a view from a built-in container is no step, so two views
    /// from one such container leave no head with two steps up. `as_slice`
    /// at `&String` and `as_str` at `&String` refuse nothing and step from
    /// nowhere.
    #[test]
    fn views_from_a_built_in_container_are_no_steps() {
        let mut fixture = Fixture::new();
        let core = fixture.raw.core;
        let string = fixture.ty(TyKind::String);
        let string_ref = fixture.reference(Mutability::Shared, string);
        let byte = fixture.ty(TyKind::Int(IntTy::U8));
        let bytes = fixture.ty(TyKind::Slice(byte));
        let bytes_ref = fixture.reference(Mutability::Shared, bytes);
        let str_ref = fixture.shared_str();
        let as_slice_at_string = fixture.function(&[string_ref], bytes_ref);
        let as_str_at_string = fixture.function(&[string_ref], str_ref);
        fixture.give(core.as_slice, &[as_slice_at_string]);
        fixture.give(core.as_str, &[as_str_at_string]);
        let decls = fixture.verify().expect("views from a built-in container refuse nothing");
        assert!(decls.casts().step_up(Head::Ref(Mutability::Shared, Base::String)).is_none());
    }

    /// extern-fn §8: a type term names only declared types; a context's
    /// type is one.
    #[test]
    fn a_context_naming_an_undeclared_type_is_refused() {
        let mut fixture = Fixture::new();
        let (context, ghost) = (fixture.name("m", "user"), fixture.name("m", "Ghost"));
        let ghost_ty = undeclared(&mut fixture, ghost);
        fixture.raw.contexts.push((context, ContextDecl { ty: ghost_ty, volatile: false }));
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

    /// extern-fn §8: an entry's types are type terms too.
    #[test]
    fn an_entry_naming_an_undeclared_type_is_refused() {
        let mut fixture = Fixture::new();
        let ghost = fixture.name("m", "Ghost");
        let function = fixture.name("m", "f");
        let name = fixture.interner.intern("x");
        let ghost_ty = undeclared(&mut fixture, ghost);
        fixture.raw.entries.push(EntryDecl { function, inputs: vec![(name, ghost_ty)], result: None });
        assert_eq!(
            refusals(fixture),
            vec![Refused::UndeclaredType {
                written: Written::Entry(function),
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

    /// functions §2: a closure of signatures is lower's alone, so a
    /// context's type holding one is refused.
    #[test]
    fn a_context_holding_a_closure_of_signatures_is_refused() {
        let mut fixture = Fixture::new();
        let context = fixture.name("m", "closures");
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let signature = fixture.function(&[int], int);
        let closures = fixture.ty(TyKind::Closures(Box::new([signature])));
        fixture.raw.contexts.push((context, ContextDecl { ty: closures, volatile: false }));
        assert_eq!(
            refusals(fixture),
            vec![Refused::InternalType {
                written: Written::Context(context)
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
        fixture.raw.contexts.push((context, ContextDecl { ty: option, volatile: false }));
        assert_eq!(
            refusals(fixture),
            vec![Refused::InternalType {
                written: Written::Context(context)
            }]
        );
    }

    /// types §4: a call of a function whose signature's whole result is `!`
    /// gives no value; such a declaration is taken in.
    #[test]
    fn a_signature_whose_whole_result_is_never_is_taken_in() {
        let mut fixture = Fixture::new();
        let abort = fixture.name("m", "abort");
        let never = fixture.ty(TyKind::Never);
        let ty = fixture.function(&[], never);
        fixture.declare(abort, fixture.plain(ty, 0));
        assert!(fixture.verify().is_ok());
    }

    /// A `!` a declaration writes anywhere but as a function signature's
    /// whole result is not supported yet: a parameter, a part of a result, a
    /// result of a function type a signature holds, a context's type, and
    /// an entry's input and result are each refused.
    #[test]
    fn a_never_written_outside_a_signatures_whole_result_is_refused() {
        let mut fixture = Fixture::new();
        let (by_param, by_part, by_callback) = (fixture.name("m", "f"), fixture.name("m", "g"), fixture.name("m", "h"));
        let (context, by_input, by_result) = (fixture.name("m", "c"), fixture.name("m", "e"), fixture.name("m", "r"));
        let input = fixture.interner.intern("i");
        let (never, int) = (fixture.ty(TyKind::Never), fixture.ty(TyKind::Int(IntTy::I64)));
        let takes_never = fixture.function(&[never], int);
        fixture.declare(by_param, fixture.plain(takes_never, 0));
        let option = fixture.ty(TyKind::Option(never));
        let gives_part = fixture.function(&[], option);
        fixture.declare(by_part, fixture.plain(gives_part, 0));
        let aborts = fixture.function(&[], never);
        let takes_callback = fixture.function(&[aborts], int);
        fixture.declare(by_callback, fixture.plain(takes_callback, 0));
        fixture.raw.contexts.push((context, ContextDecl { ty: never, volatile: false }));
        fixture.raw.entries.push(EntryDecl { function: by_input, inputs: vec![(input, never)], result: Some(int) });
        fixture.raw.entries.push(EntryDecl { function: by_result, inputs: vec![(input, int)], result: Some(never) });
        let refused = |written| Refused::NeverOutsideResult { written };
        assert_eq!(
            refusals(fixture),
            vec![
                refused(Written::Function(by_param)),
                refused(Written::Function(by_part)),
                refused(Written::Function(by_callback)),
                refused(Written::Context(context)),
                refused(Written::Entry(by_input)),
                refused(Written::Entry(by_result)),
            ]
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

    /// extern-fn §1: an extern fn is a function. A declaration whose
    /// signature is `i64` is refused as that, and not for its parameter
    /// names, since it has no parameters to name.
    #[test]
    fn a_signature_that_is_no_function_type_is_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        fixture.declare(f, fixture.plain(int, 0));
        assert_eq!(refusals(fixture), vec![Refused::NotAFunction { function: f }]);
    }

    /// extern-fn §1: an impl of an extern fn is a function too.
    #[test]
    fn an_impl_that_is_no_function_type_is_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let (var, int, unit) = (
            fixture.ty(TyKind::Var(0)),
            fixture.ty(TyKind::Int(IntTy::I64)),
            fixture.ty(TyKind::Unit),
        );
        let own_ty = fixture.function(&[var], unit);
        fixture.with_impls(f, own_ty, 1, &[int]);
        let function = fixture.impl_id(f, 0);
        assert_eq!(
            refusals(fixture),
            vec![
                Refused::NotAFunction { function },
                Refused::ImplNotOfGenericFunction {
                    generic_function: f,
                    r#impl: 0
                },
            ]
        );
    }

    /// extern-fn §1: an impl names its own parameters too. Its generic
    /// function's names fit, and its own, one short, are refused.
    #[test]
    fn an_impls_parameter_names_not_one_per_parameter_are_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let (var, int, unit) = (
            fixture.ty(TyKind::Var(0)),
            fixture.ty(TyKind::Int(IntTy::I64)),
            fixture.ty(TyKind::Unit),
        );
        let own_ty = fixture.function(&[var], unit);
        let impl_ty = fixture.function(&[int], unit);
        let at_f = Scheme {
            param_names: Box::new([]),
            ..fixture.r#impl(f, impl_ty, Box::new([]))
        };
        fixture.declare_impls(f, own_ty, 1, vec![at_f]);
        let function = fixture.impl_id(f, 0);
        assert_eq!(refusals(fixture), vec![Refused::ParamNamesDoNotFit { function }]);
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

    /// `fn() -> i64` whose effect is the effect variable `var`.
    fn worker(fixture: &mut Fixture, var: u32) -> TyId<Poly> {
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        fixture.ty(TyKind::Fn {
            params: Box::new([]),
            ret: int,
            effect: EffectTerm::Var(var),
            flows: FlowTerm::Known(declared_fn_flows()),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        })
    }

    /// extern-fn §2: an effect variable a parameter gives is run, given in
    /// the result, or both. `fn(fn() with E0) -> i64` running nothing is
    /// refused; running `E0` it is taken in, and so is
    /// `fn(fn() with E0) -> fn() with E0`, which gives it in its result.
    #[test]
    fn an_effect_variable_a_parameter_gives_is_run_or_returned() {
        for (returned, runs, refused) in [(false, vec![], true), (false, vec![0], false), (true, vec![], false)] {
            let mut fixture = Fixture::new();
            let f = fixture.name("m", "f");
            let given = worker(&mut fixture, 0);
            let ret = match returned {
                true => given,
                false => fixture.ty(TyKind::Int(IntTy::I64)),
            };
            let ty = fixture.function(&[given], ret);
            let scheme = Scheme { runs: runs.into(), ..fixture.plain(ty, 0) };
            fixture.declare(f, scheme);
            match refused {
                true => assert_eq!(
                    refusals(fixture),
                    vec![Refused::EffectUnrecorded { declared: Declared { function: f }, var: 0 }]
                ),
                false => assert!(fixture.verify().is_ok()),
            }
        }
    }

    /// extern-fn §2: a `runs` entry is an effect variable of the
    /// declaration's own signature, an impl's alike.
    #[test]
    fn a_runs_entry_that_is_no_effect_variable_is_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let t = fixture.ty(TyKind::Var(0));
        let own_ty = fixture.function(&[t], t);
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let impl_ty = fixture.function(&[int], int);
        let at_f = Scheme {
            runs: Box::new([2]),
            ..fixture.r#impl(f, impl_ty, Box::new([]))
        };
        fixture.declare_impls(f, own_ty, 1, vec![at_f]);
        let function = fixture.impl_id(f, 0);
        assert_eq!(
            refusals(fixture),
            vec![Refused::RunsNoEffectVariable { declared: Declared { function }, var: 2 }]
        );
    }

    /// extern-fn §6: a declaration requires impls at its own type
    /// variables. `fn(T0) -> T0` requiring an impl at `T1` is refused.
    #[test]
    fn a_requirement_at_a_foreign_type_variable_is_refused() {
        let mut fixture = Fixture::new();
        let (f, g) = (fixture.name("m", "f"), fixture.name("m", "g"));
        let (t, u) = (fixture.ty(TyKind::Var(0)), fixture.ty(TyKind::Var(1)));
        let own_ty = fixture.function(&[t], t);
        let requires = Box::new([Requirement { generic_function: g, pattern: u, fact: None }]);
        fixture.declare(f, Scheme { requires, ..fixture.plain(own_ty, 1) });
        assert_eq!(
            refusals(fixture),
            vec![Refused::RequirementForeignVariable { declaration: f, generic_function: g }]
        );
    }

    /// extern-fn §1, §6: a concrete impl bounds its own type variables as a
    /// scheme does, one bound per variable by position. An impl `fn(U) -> U`
    /// with no bound is refused; with one it is taken in, and the bound is
    /// the impl's own.
    #[test]
    fn an_impl_bounds_its_own_variables_one_each() {
        for (bounds, fit) in [(0, false), (1, true), (2, false)] {
            let mut fixture = Fixture::new();
            let f = fixture.name("m", "f");
            let t = fixture.ty(TyKind::Var(0));
            let ty = fixture.function(&[t], t);
            let at_f = Scheme {
                bounds: vec![TyVarBound::Settled; bounds].into(),
                ..fixture.r#impl(f, ty, Box::new([]))
            };
            fixture.declare_impls(f, ty, 1, vec![at_f]);
            let impl_at = fixture.impl_id(f, 0);
            match fit {
                true => {
                    let decls = fixture.verify().expect("an impl with one bound per variable verifies");
                    let r#impl = decls.scheme(impl_at).expect("the impl is declared");
                    assert_eq!(&r#impl.bounds[..], &[TyVarBound::Settled][..]);
                }
                false => assert_eq!(refusals(fixture), vec![Refused::BoundsDoNotFit { function: impl_at }]),
            }
        }
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

    /// operators §2: the compiler builds every lang item's generic function,
    /// so one declared under a lang item's name is refused.
    #[test]
    fn a_generic_function_under_a_lang_items_name_is_refused() {
        let mut fixture = Fixture::new();
        let core = fixture.raw.core;
        let t = fixture.ty(TyKind::Var(0));
        let ty = fixture.function(&[t], t);
        fixture.declare(core.clone, fixture.plain(ty, 1));
        assert_eq!(refusals(fixture), vec![Refused::CoreGenericFunction { id: core.clone }]);
    }

    /// extern-fn §6: an impl is of a generic function.
    #[test]
    fn an_impl_of_what_is_no_generic_function_is_refused() {
        let mut fixture = Fixture::new();
        let id = fixture.name("m", "f");
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let ty = fixture.function(&[int], int);
        fixture.give(id, &[ty]);
        let function = fixture.impl_id(id, 0);
        assert_eq!(refusals(fixture), vec![Refused::ImplOfNoGenericFunction { function, generic_function: id }]);
    }

    /// operators §2: `as_ref` and `as_mut` mean what the language says, and
    /// no declaration gives them impls, nor declares a function under
    /// their names.
    #[test]
    fn impls_of_as_ref_or_as_mut_are_refused() {
        let mut fixture = Fixture::new();
        let core = fixture.raw.core;
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let shared = fixture.ty(TyKind::Ref(Mutability::Shared, int));
        let ty = fixture.function(&[shared], shared);
        fixture.give(core.as_ref, &[ty]);
        assert_eq!(refusals(fixture), vec![Refused::TakesNoImpls { id: core.as_ref }]);
        let mut fixture = Fixture::new();
        let core = fixture.raw.core;
        let t = fixture.ty(TyKind::Var(0));
        let ty = fixture.function(&[t], t);
        fixture.declare(core.as_mut, fixture.plain(ty, 1));
        assert_eq!(refusals(fixture), vec![Refused::CoreGenericFunction { id: core.as_mut }]);
    }

    /// operators §2, extern-fn §6: an impl of a view that is not a function
    /// is refused as a signature and as no impl of the view's type, and the
    /// view rule never reads it.
    #[test]
    fn a_view_impl_that_is_no_function_is_refused_before_the_view_rule() {
        let mut fixture = Fixture::new();
        let core = fixture.raw.core;
        let not_a_function = fixture.ty(TyKind::Int(IntTy::I64));
        fixture.give(core.as_slice, &[not_a_function]);
        let function = fixture.impl_id(core.as_slice, 0);
        assert_eq!(
            refusals(fixture),
            vec![
                Refused::NotAFunction { function },
                Refused::ImplNotOfGenericFunction {
                    generic_function: core.as_slice,
                    r#impl: 0
                },
            ]
        );
    }

    /// extern-fn §6: reach takes a generic impl only where no concrete one
    /// answers, so a generic function has at most one.
    #[test]
    fn two_generic_impls_of_one_generic_function_are_refused() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let t = fixture.ty(TyKind::Var(0));
        let ty = fixture.function(&[t], t);
        let generic_impl = |fixture: &Fixture| Scheme {
            impl_of: Some(ImplOf { generic_function: f, kind: ImplKind::Generic }),
            ..fixture.plain(ty, 1)
        };
        let impls = vec![generic_impl(&fixture), generic_impl(&fixture)];
        fixture.declare_impls(f, ty, 1, impls);
        let refused = refusals(fixture);
        assert!(refused.contains(&Refused::TwoGenericImpls { generic_function: f }));
        assert_eq!(
            refused.iter().filter(|refusal| matches!(refusal, Refused::TwoGenericImpls { .. })).count(),
            1
        );
    }

    /// types §6: a function and a generic function under one name cannot
    /// be told apart, and nothing orders one before the other, so neither
    /// stands.
    #[test]
    fn a_function_and_a_generic_function_under_one_path_are_both_refused() {
        let mut fixture = Fixture::new();
        let id = fixture.name("m", "f");
        let t = fixture.ty(TyKind::Var(0));
        let ty = fixture.function(&[t], t);
        fixture.declare(id, fixture.plain(ty, 1));
        fixture.declare_impls(id, ty, 1, vec![]);
        assert_eq!(refusals(fixture), vec![Refused::DuplicateFunction { id }]);
    }

    /// extern-fn §6: a generic impl's signature is its generic function's;
    /// another one is a bug of whoever built the declarations, not a
    /// refusal.
    #[test]
    #[should_panic(expected = "is not its generic function's signature")]
    fn a_generic_impl_of_another_signature_panics() {
        let mut fixture = Fixture::new();
        let f = fixture.name("m", "f");
        let t = fixture.ty(TyKind::Var(0));
        let own_ty = fixture.function(&[t], t);
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let impl_ty = fixture.function(&[int], int);
        let generic_impl = Scheme {
            impl_of: Some(ImplOf { generic_function: f, kind: ImplKind::Generic }),
            ..fixture.plain(impl_ty, 0)
        };
        fixture.declare_impls(f, own_ty, 1, vec![generic_impl]);
        let _ = fixture.verify();
    }

    /// operators §2: an impl of a lang item is held to the compiler's type
    /// of it by the impl-fit rule.
    #[test]
    fn an_impl_is_held_to_the_compilers_lang_item() {
        let mut fixture = Fixture::new();
        let core = fixture.raw.core;
        let int = fixture.ty(TyKind::Int(IntTy::I64));
        let shared = fixture.reference(Mutability::Shared, int);
        let fits = fixture.function(&[shared, shared], int);
        let misfit = fixture.function(&[int, int], int);
        fixture.give(core.add, &[fits, misfit]);
        assert_eq!(
            refusals(fixture),
            vec![Refused::ImplNotOfGenericFunction {
                generic_function: core.add,
                r#impl: 1
            }]
        );
    }
}
