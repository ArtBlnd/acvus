use acvus_utils::{Interner, QualifiedRef};

use super::*;
use crate::decl::{
    Core, Declarations, DeclaredFacts, Ends, GenericImplSig, ImplSig, Impls, LoopMarkers, RawDeclarations, Scheme,
};
use crate::ty::{Mark, MarkTerm};
use crate::ty::{Calls, Effect, EffectTerm, FlowTerm, Holds, IntTy, Mutability, TyKind, Types};

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
                lang_impls: Vec::new(),
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

    fn ty(&mut self, kind: TyKind<Poly>) -> TyId<Poly> {
        self.raw.types.intern(kind)
    }

    fn int(&mut self) -> TyId<Poly> {
        self.ty(TyKind::Int(IntTy::I64))
    }

    fn index(&mut self) -> TyId<Poly> {
        self.ty(TyKind::Int(IntTy::U64))
    }

    fn reference(&mut self, mutability: Mutability, lent: TyId<Poly>) -> TyId<Poly> {
        self.ty(TyKind::Ref(mutability, lent))
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

    fn param_names(&self, arity: usize) -> Box<[acvus_utils::Astr]> {
        (0..arity).map(|at| self.interner.intern(&format!("p{at}"))).collect()
    }

    /// `name`, of type `fn(params) -> ret`, without impls, stating `facts`.
    fn declare(&mut self, name: &str, params: &[TyId<Poly>], ret: TyId<Poly>, facts: DeclaredFacts) -> QualifiedRef {
        let scheme = self.scheme(params, ret, None, facts);
        self.push(name, scheme)
    }

    /// `name`, of type `fn(params) -> ret`, whose one impl, the generic
    /// one, states `facts`.
    fn declare_generic(&mut self, name: &str, params: &[TyId<Poly>], ret: TyId<Poly>, facts: DeclaredFacts) -> QualifiedRef {
        let impls = Impls { concrete: Box::new([]), generic: Some(GenericImplSig { facts, awaits: false }) };
        let scheme = self.scheme(params, ret, Some(impls), DeclaredFacts::default());
        self.push(name, scheme)
    }

    fn scheme(&mut self, params: &[TyId<Poly>], ret: TyId<Poly>, impls: Option<Impls>, facts: DeclaredFacts) -> Scheme {
        Scheme {
            ty: self.function(params, ret),
            param_names: self.param_names(params.len()),
            bounds: Box::new([]),
            impls,
            requires: Box::new([]),
            facts,
            runs: Box::new([]),
            awaits: false,
            specialized: Box::new([]),
        }
    }

    fn push(&mut self, name: &str, scheme: Scheme) -> QualifiedRef {
        let id = QualifiedRef::root(self.interner.intern(name));
        self.raw.functions.push((id, scheme));
        id
    }

    /// Gives `eq` the impl `fn(&i64, &i64) -> bool` stating `facts`.
    fn give_eq(&mut self, facts: DeclaredFacts) {
        let int = self.int();
        let shared = self.reference(Mutability::Shared, int);
        let bool_ty = self.ty(TyKind::Bool);
        let ty = self.function(&[shared, shared], bool_ty);
        let r#impl = ImplSig { ty, bounds: Box::new([]), param_names: self.param_names(2), requires: Box::new([]), facts, runs: Box::new([]), specialized: Box::new([]), awaits: false };
        let eq = self.raw.core.eq;
        self.raw.lang_impls.push((eq, Impls { concrete: Box::new([r#impl]), generic: None }));
    }

    fn refusals(self) -> Vec<Refused> {
        match Declarations::verify(self.raw) {
            Ok(_) => Vec::new(),
            Err(refused) => refused,
        }
    }

    fn verified(self) -> Declarations {
        Declarations::verify(self.raw).unwrap_or_else(|refused| panic!("the declarations verify: {refused:?}"))
    }
}

fn touches(touches: &[Touch]) -> DeclaredFacts {
    DeclaredFacts { access: Box::new([Access::Touches(touches.into())]), ..DeclaredFacts::default() }
}

fn whole(param: usize) -> Place {
    Place { param, element: None }
}

fn update(place: Place, combine: QualifiedRef) -> Update {
    Update { place, combine }
}

fn combine(identity: Option<QualifiedRef>, commutative: bool, absorbing: Option<QualifiedRef>) -> DeclaredFacts {
    actions(vec![Action::Combine(Combine { identity, commutative, absorbing })])
}

fn actions(actions: Vec<Action>) -> DeclaredFacts {
    DeclaredFacts { action: actions.into(), ..DeclaredFacts::default() }
}

fn sources(sources: Vec<Source>) -> DeclaredFacts {
    DeclaredFacts { source: sources.into(), ..DeclaredFacts::default() }
}

fn relates(left: Atom, right: Term) -> DeclaredFacts {
    let relation = Relation { left: Term::Atom(left), comparison: Comparison::Le, right };
    DeclaredFacts { relations: Box::new([relation]), ..DeclaredFacts::default() }
}

fn ending(ends: Ends) -> DeclaredFacts {
    DeclaredFacts { ends, ..DeclaredFacts::default() }
}

fn own(f: QualifiedRef) -> Declared {
    Declared { function: f, r#impl: None }
}

/// One form of each kind, well formed: a store into a map's entry and a
/// read of a slice's element, an update by an associative combine with its
/// identity, an inverse of it, a total order, a map of a stream, a relation over
/// integers and lengths, and an `eq` impl declared an equivalence.
#[test]
fn well_formed_facts_of_every_kind_verify() {
    let mut f = Fixture::new();
    let int = f.int();
    let index = f.index();
    let unit = f.ty(TyKind::Unit);
    let mut_int = f.reference(Mutability::Mut, int);
    let shared_int = f.reference(Mutability::Shared, int);
    let slice = f.ty(TyKind::Slice(int));
    let shared_slice = f.reference(Mutability::Shared, slice);
    let map = f.ty(TyKind::UserDefined {
        id: QualifiedRef::root(f.interner.intern("Map")),
        type_args: Box::new([]),
        effect: None,
        identity: None,
        marked: MarkTerm::Known(Mark::Unmarked),
    });
    f.raw.user_types.push(crate::decl::UserTypeDecl {
        id: QualifiedRef::root(f.interner.intern("Map")),
        type_params: Box::new([]),
        carries_effect: false,
        carries_identity: false,
        lifetimes: 0,
    });
    let mut_map = f.reference(Mutability::Mut, map);

    let stored = Touch { place: Place { param: 0, element: Some(Element::Key(1)) }, mode: Mode::Store(Stored::Value(2)) };
    let read = Touch { place: Place { param: 3, element: Some(Element::Index(4)) }, mode: Mode::Read };
    f.declare("insert", &[mut_map, int, int, shared_slice, index], unit, touches(&[stored, read]));

    let zero = f.declare("zero", &[], int, DeclaredFacts::default());
    let add_into = f.declare("add_into", &[mut_int, int], unit, combine(Some(zero), true, None));
    let add = f.declare("add", &[mut_int, int], unit, actions(vec![Action::Update(update(whole(0), add_into))]));
    let decrement = Action::InverseOf { place: whole(0), other: add };
    f.declare("sub", &[mut_int, int], unit, actions(vec![decrement]));
    f.declare("compare", &[shared_int, shared_int], int, actions(vec![Action::TotalOrder]));

    let through = f.function(&[int], int);
    let map_stream = Source::Map { from: Stream::Param(0), through: 1 };
    let count = Source::Count { comparison: Comparison::Le, bound: Term::Atom(Atom::Len(Subject::Arg(2))) };
    f.declare("map", &[int, through, shared_slice], int, sources(vec![Source::Stepwise, map_stream, count]));

    let len_or_old = Term::Add(Box::new(Term::Atom(Atom::Len(Subject::Arg(1)))), Box::new(Term::Atom(Atom::Old(Before::Lent(0)))));
    f.declare("bounded", &[mut_int, shared_slice], int, relates(Atom::Result, len_or_old));

    f.give_eq(DeclaredFacts { access: Box::new([Access::Equivalence]), ..DeclaredFacts::default() });

    assert_eq!(f.refusals(), Vec::new());
}

#[test]
fn a_fact_naming_no_parameter_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let shared_int = f.reference(Mutability::Shared, int);
    let g = f.declare("g", &[shared_int], int, touches(&[Touch { place: whole(5), mode: Mode::Read }]));
    assert_eq!(f.refusals(), vec![Refused::FactNamesNoParameter { declared: own(g), param: 5 }]);
}

#[test]
fn an_access_naming_a_parameter_that_is_no_reference_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let g = f.declare("g", &[int], int, touches(&[Touch { place: whole(0), mode: Mode::Read }]));
    assert_eq!(f.refusals(), vec![Refused::PlaceNotLent { declared: own(g), param: 0 }]);
}

#[test]
fn a_store_through_a_shared_reference_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let shared_int = f.reference(Mutability::Shared, int);
    let emptied = Touch { place: whole(0), mode: Mode::Store(Stored::Emptied) };
    let g = f.declare("g", &[shared_int], int, touches(&[emptied]));
    assert_eq!(f.refusals(), vec![Refused::PlaceNotLentMutably { declared: own(g), param: 0 }]);
}

#[test]
fn an_index_that_is_not_an_index_parameter_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let slice = f.ty(TyKind::Slice(int));
    let shared_slice = f.reference(Mutability::Shared, slice);
    let read = Touch { place: Place { param: 0, element: Some(Element::Index(1)) }, mode: Mode::Read };
    let g = f.declare("g", &[shared_slice, int], int, touches(&[read]));
    assert_eq!(f.refusals(), vec![Refused::IndexNotAnIndex { declared: own(g), param: 1 }]);
}

#[test]
fn an_open_of_a_place_no_key_names_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let mut_int = f.reference(Mutability::Mut, int);
    let open = Touch { place: whole(0), mode: Mode::Open { default: 1 } };
    let g = f.declare("g", &[mut_int, int], int, touches(&[open]));
    assert_eq!(f.refusals(), vec![Refused::OpenOfNoEntry { declared: own(g), param: 0 }]);
}

#[test]
fn two_accesses_on_one_impl_are_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let shared_int = f.reference(Mutability::Shared, int);
    let read = Access::Touches(Box::new([Touch { place: whole(0), mode: Mode::Read }]));
    let facts = DeclaredFacts { access: Box::new([read.clone(), read]), ..DeclaredFacts::default() };
    let g = f.declare("g", &[shared_int], int, facts);
    assert_eq!(f.refusals(), vec![Refused::TwoAccesses { declared: own(g) }]);
}

#[test]
fn an_equivalence_on_a_declaration_other_than_an_eq_impl_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let shared_int = f.reference(Mutability::Shared, int);
    let bool_ty = f.ty(TyKind::Bool);
    let facts = DeclaredFacts { access: Box::new([Access::Equivalence]), ..DeclaredFacts::default() };
    let g = f.declare("same", &[shared_int, shared_int], bool_ty, facts);
    assert_eq!(f.refusals(), vec![Refused::EquivalenceNotOnEq { declared: own(g) }]);
}

#[test]
fn a_total_order_on_a_function_that_compares_no_two_values_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let shared_int = f.reference(Mutability::Shared, int);
    let g = f.declare("g", &[shared_int], int, actions(vec![Action::TotalOrder]));
    assert_eq!(f.refusals(), vec![Refused::TotalOrderDoesNotFit { declared: own(g) }]);
}

/// `append(&mut String, String)` is a combine, but of a `String`, and the
/// place holds an `i64`.
#[test]
fn a_combine_whose_type_does_not_fit_the_update_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let unit = f.ty(TyKind::Unit);
    let string = f.ty(TyKind::String);
    let mut_string = f.reference(Mutability::Mut, string);
    let mut_int = f.reference(Mutability::Mut, int);
    let append = f.declare("append", &[mut_string, string], unit, combine(None, false, None));
    let g = f.declare("g", &[mut_int, int], unit, actions(vec![Action::Update(update(whole(0), append))]));
    assert_eq!(f.refusals(), vec![Refused::CombineDoesNotFit { declared: own(g), combine: append }]);
}

#[test]
fn an_identity_of_another_type_than_the_state_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let unit = f.ty(TyKind::Unit);
    let string = f.ty(TyKind::String);
    let mut_int = f.reference(Mutability::Mut, int);
    let empty = f.declare("empty", &[], string, DeclaredFacts::default());
    let add_into = f.declare("add_into", &[mut_int, int], unit, combine(Some(empty), false, None));
    f.declare("g", &[mut_int, int], unit, actions(vec![Action::Update(update(whole(0), add_into))]));
    assert_eq!(f.refusals(), vec![Refused::IdentityDoesNotFit { declared: own(add_into), identity: empty }]);
}

#[test]
fn an_inverse_of_a_function_that_updates_no_such_place_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let unit = f.ty(TyKind::Unit);
    let mut_int = f.reference(Mutability::Mut, int);
    let add_into = f.declare("add_into", &[mut_int, int], unit, DeclaredFacts::default());
    let inverse = Action::InverseOf { place: whole(0), other: add_into };
    let g = f.declare("g", &[mut_int, int], unit, actions(vec![inverse]));
    assert_eq!(
        f.refusals(),
        vec![Refused::InverseNotAnUpdateAtThePlace { declared: own(g), inverse: add_into }]
    );
}

/// extern-fn §3: an update may be declared the inverse of another at the
/// same place, with no combine of its own. `pop` is declared the inverse
/// of `push`, whose update of the stack is the action of `append`.
#[test]
fn an_inverse_needs_no_combine_of_its_own() {
    let mut f = Fixture::new();
    let unit = f.ty(TyKind::Unit);
    let stack = f.ty(TyKind::String);
    let mut_stack = f.reference(Mutability::Mut, stack);
    let append = f.declare("append", &[mut_stack, stack], unit, combine(None, false, None));
    let push = f.declare("push", &[mut_stack, stack], unit, actions(vec![Action::Update(update(whole(0), append))]));
    f.declare("pop", &[mut_stack], stack, actions(vec![Action::InverseOf { place: whole(0), other: push }]));
    assert_eq!(f.refusals(), Vec::new());
}

/// An inverse updates its place, so the place is lent mutably.
#[test]
fn an_inverse_through_a_shared_reference_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let unit = f.ty(TyKind::Unit);
    let mut_int = f.reference(Mutability::Mut, int);
    let shared_int = f.reference(Mutability::Shared, int);
    let add_into = f.declare("add_into", &[mut_int, int], unit, combine(None, false, None));
    let add = f.declare("add", &[mut_int, int], unit, actions(vec![Action::Update(update(whole(0), add_into))]));
    let g = f.declare("g", &[shared_int], unit, actions(vec![Action::InverseOf { place: whole(0), other: add }]));
    assert_eq!(f.refusals(), vec![Refused::PlaceNotLentMutably { declared: own(g), param: 0 }]);
}

#[test]
fn a_stream_both_splittable_and_stepwise_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let g = f.declare("g", &[int], int, sources(vec![Source::Splittable, Source::Stepwise]));
    assert_eq!(f.refusals(), vec![Refused::SplittableAndStepwise { declared: own(g) }]);
}

#[test]
fn a_predicate_that_answers_no_bool_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let predicate = f.function(&[int], int);
    let g = f.declare("g", &[int, predicate], int, sources(vec![Source::Filter { from: Stream::Param(0), predicate: 1 }]));
    assert_eq!(f.refusals(), vec![Refused::ClosureDoesNotFit { declared: own(g), param: 1 }]);
}

#[test]
fn a_relation_over_an_atom_that_is_no_integer_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let string = f.ty(TyKind::String);
    let g = f.declare("g", &[string], int, relates(Atom::Result, Term::Atom(Atom::Arg(0))));
    assert_eq!(f.refusals(), vec![Refused::RelationOverNonInteger { declared: own(g), atom: Atom::Arg(0) }]);
}

/// extern-fn §1, §3: a concrete impl's variables carry the impl's own
/// bounds, as a scheme's carry its: a relation over an argument whose type
/// is the impl's variable stands where the impl bounds it to integers, and
/// is refused where the impl lets it be any type.
#[test]
fn a_relation_reads_a_concrete_impls_own_bounds() {
    let integer = TyVarBound::Integer { signed: true, among: Box::new([IntTy::I64]) };
    for (bound, refused) in [(integer, false), (TyVarBound::Any, true)] {
        let mut f = Fixture::new();
        let (var, int) = (f.ty(TyKind::Var(0)), f.int());
        let ty = f.function(&[var], int);
        let r#impl = ImplSig {
            ty,
            bounds: Box::new([bound]),
            param_names: f.param_names(1),
            requires: Box::new([]),
            facts: relates(Atom::Result, Term::Atom(Atom::Arg(0))),
            runs: Box::new([]),
            awaits: false,
            specialized: Box::new([]),
        };
        let scheme = Scheme {
            ty,
            param_names: f.param_names(1),
            bounds: Box::new([TyVarBound::Any]),
            impls: Some(Impls { concrete: Box::new([r#impl]), generic: None }),
            requires: Box::new([]),
            facts: DeclaredFacts::default(),
            runs: Box::new([]),
            awaits: false,
            specialized: Box::new([]),
        };
        let g = f.push("g", scheme);
        let at_impl = Declared { function: g, r#impl: Some(0) };
        let expected = match refused {
            true => vec![Refused::RelationOverNonInteger { declared: at_impl, atom: Atom::Arg(0) }],
            false => Vec::new(),
        };
        assert_eq!(f.refusals(), expected);
    }
}

#[test]
fn a_length_of_a_value_with_no_element_count_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let g = f.declare("g", &[int], int, relates(Atom::Result, Term::Atom(Atom::Len(Subject::Arg(0)))));
    assert_eq!(
        f.refusals(),
        vec![Refused::LengthOfNoSequence { declared: own(g), subject: Subject::Arg(0) }]
    );
}

#[test]
fn an_old_value_through_a_shared_reference_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let shared_int = f.reference(Mutability::Shared, int);
    let g = f.declare("g", &[shared_int], int, relates(Atom::Result, Term::Atom(Atom::Old(Before::Lent(0)))));
    assert_eq!(f.refusals(), vec![Refused::OldOfNoMutableReference { declared: own(g), param: 0 }]);
}

/// A declaration without impls states its facts itself, and a call of it
/// reaches them as impl 0.
#[test]
fn a_declaration_without_impls_states_its_own_facts() {
    let mut f = Fixture::new();
    let int = f.int();
    let shared_int = f.reference(Mutability::Shared, int);
    let read = touches(&[Touch { place: whole(0), mode: Mode::Read }]);
    let g = f.declare("g", &[shared_int], int, read.clone());
    let decls = f.verified();
    assert_eq!(decls.call_facts(g, None), &read);
}

/// `verify` checks the facts an impl states, the generic impl's at its
/// function's own signature.
#[test]
fn an_ill_formed_fact_on_a_generic_impl_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let g = f.declare_generic("g", &[int], int, touches(&[Touch { place: whole(0), mode: Mode::Read }]));
    assert_eq!(f.refusals(), vec![Refused::PlaceNotLent { declared: own(g), param: 0 }]);
}

/// A call of a declaration with impls reaches one of its impls, so facts
/// stated on the declaration itself would be read by no call.
#[test]
fn facts_on_a_declaration_with_impls_are_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let shared_int = f.reference(Mutability::Shared, int);
    let read = touches(&[Touch { place: whole(0), mode: Mode::Read }]);
    let impls = Impls { concrete: Box::new([]), generic: Some(GenericImplSig::default()) };
    let scheme = f.scheme(&[shared_int], int, Some(impls), read);
    let g = f.push("g", scheme);
    assert_eq!(f.refusals(), vec![Refused::FactsBesideImpls { function: g }]);
}

/// extern-fn §3: a call that stores into a place may give back what it
/// held, whole or through a declared `fn(&H) -> R`.
#[test]
fn a_result_given_from_what_a_place_held_verifies() {
    let mut f = Fixture::new();
    let string = f.ty(TyKind::String);
    let index = f.index();
    let mut_string = f.reference(Mutability::Mut, string);
    let shared_string = f.reference(Mutability::Shared, string);
    let replaced = Touch { place: whole(0), mode: Mode::Store(Stored::Value(1)) };
    let gives = |through| DeclaredFacts {
        access: Box::new([Access::Touches(Box::new([replaced])), Access::Gives { place: whole(0), through }]),
        ..DeclaredFacts::default()
    };
    let len = f.declare("len", &[shared_string], index, DeclaredFacts::default());
    f.declare("replace", &[mut_string, string], string, gives(None));
    f.declare("replace_len", &[mut_string, string], index, gives(Some(len)));
    assert_eq!(f.refusals(), Vec::new());
}

/// A result stated to be what the place held is of the type the place
/// holds, and a function of it takes a `&` to what the place holds and
/// answers the call's result.
#[test]
fn a_result_given_of_another_type_than_the_place_holds_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let string = f.ty(TyKind::String);
    let mut_string = f.reference(Mutability::Mut, string);
    let shared_int = f.reference(Mutability::Shared, int);
    let gives = |through| DeclaredFacts {
        access: Box::new([Access::Gives { place: whole(0), through }]),
        ..DeclaredFacts::default()
    };
    let of_int = f.declare("of_int", &[shared_int], int, DeclaredFacts::default());
    let g = f.declare("g", &[mut_string], int, gives(None));
    let h = f.declare("h", &[mut_string], int, gives(Some(of_int)));
    assert_eq!(
        f.refusals(),
        vec![
            Refused::ResultNotWhatThePlaceHeld { declared: own(g), param: 0 },
            Refused::GivesThroughDoesNotFit { declared: own(h), through: of_int },
        ]
    );
}

/// A combine is value shaped or update shaped, with an identity and an
/// absorbing element of its state.
#[test]
fn a_combine_of_either_shape_verifies() {
    let mut f = Fixture::new();
    let int = f.int();
    let unit = f.ty(TyKind::Unit);
    let mut_int = f.reference(Mutability::Mut, int);
    let least = f.declare("least", &[], int, DeclaredFacts::default());
    let greatest = f.declare("greatest", &[], int, DeclaredFacts::default());
    f.declare("max", &[int, int], int, combine(Some(least), true, Some(greatest)));
    f.declare("max_into", &[mut_int, int], unit, combine(Some(least), true, Some(greatest)));
    assert_eq!(f.refusals(), Vec::new());
}

#[test]
fn a_combine_on_no_operation_of_one_type_or_absorbing_at_another_type_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let string = f.ty(TyKind::String);
    let empty = f.declare("empty", &[], string, DeclaredFacts::default());
    let g = f.declare("g", &[int, string], int, combine(None, false, None));
    let max = f.declare("max", &[int, int], int, combine(None, true, Some(empty)));
    assert_eq!(
        f.refusals(),
        vec![
            Refused::CombineNotAnOperation { declared: own(g) },
            Refused::AbsorbingDoesNotFit { declared: own(max), absorbing: empty },
        ]
    );
}

#[test]
fn a_take_while_verifies_as_a_filter_does() {
    let mut f = Fixture::new();
    let int = f.int();
    let bool_ty = f.ty(TyKind::Bool);
    let predicate = f.function(&[int], bool_ty);
    let not_predicate = f.function(&[int], int);
    let take_while = sources(vec![Source::TakeWhile { from: Stream::Param(0), predicate: 1 }]);
    f.declare("take_while", &[int, predicate], int, take_while.clone());
    let g = f.declare("g", &[int, not_predicate], int, take_while);
    assert_eq!(f.refusals(), vec![Refused::ClosureDoesNotFit { declared: own(g), param: 1 }]);
}

/// The forms compose: `flat_map` is `Flatten` over `Map`, and the inner
/// form is held to what the outer one would be.
#[test]
fn a_stream_of_another_form_verifies_that_form() {
    let mut f = Fixture::new();
    let int = f.int();
    let through = f.function(&[int], int);
    let flat_map = |from| sources(vec![Source::Flatten { from: Stream::Of(Box::new(Source::Map { from, through: 1 })) }]);
    f.declare("flat_map", &[int, through], int, flat_map(Stream::Param(0)));
    let g = f.declare("g", &[int, through], int, flat_map(Stream::Param(5)));
    assert_eq!(f.refusals(), vec![Refused::FactNamesNoParameter { declared: own(g), param: 5 }]);
}

/// A consumption by declared steps, by closure arguments, and through a
/// finish.
#[test]
fn a_consumption_whose_steps_agree_on_one_state_verifies() {
    let mut f = Fixture::new();
    let int = f.int();
    let unit = f.ty(TyKind::Unit);
    let string = f.ty(TyKind::String);
    let mut_int = f.reference(Mutability::Mut, int);
    let slice = f.ty(TyKind::Slice(int));
    let shared_slice = f.reference(Mutability::Shared, slice);
    let zero = f.declare("zero", &[], int, DeclaredFacts::default());
    let add_into = f.declare("add_into", &[mut_int, int], unit, combine(Some(zero), true, None));
    let show = f.declare("show", &[int], string, DeclaredFacts::default());
    let step = f.function(&[int, int], int);
    let consume = |init, update, finish| sources(vec![Source::Consume { from: Stream::Param(0), init, update, finish }]);
    f.declare("sum", &[shared_slice], int, consume(Step::Declared(zero), Step::Declared(add_into), None));
    f.declare("fold", &[shared_slice, int, step], int, consume(Step::Param(1), Step::Param(2), None));
    f.declare("show_sum", &[shared_slice], string, consume(Step::Declared(zero), Step::Declared(add_into), Some(show)));
    assert_eq!(f.refusals(), Vec::new());
}

/// The state is the result where no finish is stated, and every step
/// agrees on it.
#[test]
fn a_consumption_whose_steps_disagree_on_the_state_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let string = f.ty(TyKind::String);
    let slice = f.ty(TyKind::Slice(int));
    let shared_slice = f.reference(Mutability::Shared, slice);
    let empty = f.declare("empty", &[], string, DeclaredFacts::default());
    let zero = f.declare("zero", &[], int, DeclaredFacts::default());
    let show = f.declare("show", &[int], string, DeclaredFacts::default());
    let not_a_step = f.function(&[int], int);
    let step = f.function(&[int, int], int);
    let consume = |init, update, finish| sources(vec![Source::Consume { from: Stream::Param(0), init, update, finish }]);
    let g = f.declare("g", &[shared_slice, step], int, consume(Step::Declared(empty), Step::Param(1), None));
    let h = f.declare("h", &[shared_slice, not_a_step], int, consume(Step::Declared(zero), Step::Param(1), None));
    let k = f.declare("k", &[shared_slice, step], int, consume(Step::Declared(zero), Step::Param(1), Some(show)));
    assert_eq!(
        f.refusals(),
        vec![
            Refused::ConsumeInitDoesNotFit { declared: own(g), init: Step::Declared(empty) },
            Refused::ConsumeUpdateDoesNotFit { declared: own(h), update: Step::Param(1) },
            Refused::ConsumeFinishDoesNotFit { declared: own(k), finish: Some(show) },
        ]
    );
}

#[test]
fn a_combine_is_the_named_fact_an_impl_stating_one_resolves() {
    let stated = combine(None, false, None);
    assert!(NamedFact::Combine.stated_by(&stated));
    assert!(!NamedFact::Combine.stated_by(&actions(vec![Action::TotalOrder])));
    assert!(!NamedFact::TotalOrder.stated_by(&stated));
}

/// extern-fn §3: an update of a place is the action of a combine, so the
/// function it names is declared one, and its laws are read there.
#[test]
fn an_update_by_a_function_not_declared_a_combine_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let unit = f.ty(TyKind::Unit);
    let mut_int = f.reference(Mutability::Mut, int);
    let add_into = f.declare("add_into", &[mut_int, int], unit, DeclaredFacts::default());
    let g = f.declare("g", &[mut_int, int], unit, actions(vec![Action::Update(update(whole(0), add_into))]));
    assert_eq!(f.refusals(), vec![Refused::UpdateByNoCombine { declared: own(g), combine: add_into }]);
}

/// A combine of either shape updates a place that holds its state.
#[test]
fn an_update_by_a_value_shaped_combine_verifies() {
    let mut f = Fixture::new();
    let int = f.int();
    let unit = f.ty(TyKind::Unit);
    let mut_int = f.reference(Mutability::Mut, int);
    let max = f.declare("max", &[int, int], int, combine(None, true, None));
    f.declare("raise", &[mut_int, int], unit, actions(vec![Action::Update(update(whole(0), max))]));
    assert_eq!(f.refusals(), Vec::new());
}

/// extern-fn §3: `through` is a declared pure function of what the place
/// held.
#[test]
fn a_result_given_through_a_function_not_declared_pure_is_refused() {
    let mut f = Fixture::new();
    let string = f.ty(TyKind::String);
    let index = f.index();
    let mut_string = f.reference(Mutability::Mut, string);
    let shared_string = f.reference(Mutability::Shared, string);
    let mut scheme = f.scheme(&[shared_string], index, None, DeclaredFacts::default());
    scheme.ty = f.ty(TyKind::Fn {
        params: Box::new([shared_string]),
        ret: index,
        effect: EffectTerm::Known(Effect::opaque()),
        flows: FlowTerm::Known(crate::decl::declared_fn_flows()),
        calls: Calls::Many,
        holds: Holds::Nothing,
        values: (),
    });
    let len = f.push("len", scheme);
    let gives = Access::Gives { place: whole(0), through: Some(len) };
    let facts = DeclaredFacts { access: Box::new([gives]), ..DeclaredFacts::default() };
    let g = f.declare("g", &[mut_string], index, facts);
    assert_eq!(f.refusals(), vec![Refused::GivesThroughNotPure { declared: own(g), through: len }]);
}

/// extern-fn §3: a call has one result.
#[test]
fn two_results_given_on_one_declaration_are_refused() {
    let mut f = Fixture::new();
    let string = f.ty(TyKind::String);
    let mut_string = f.reference(Mutability::Mut, string);
    let gives = |param| Access::Gives { place: whole(param), through: None };
    let facts = DeclaredFacts { access: Box::new([gives(0), gives(1)]), ..DeclaredFacts::default() };
    let g = f.declare("g", &[mut_string, mut_string], string, facts);
    assert_eq!(f.refusals(), vec![Refused::TwoResultsGiven { declared: own(g) }]);
}

/// extern-fn §3: a stream made from another is made by a form that makes
/// one; the call's own stream's forms and a consumption make none.
#[test]
fn a_stream_of_a_form_that_makes_no_stream_from_another_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let zero = f.declare("zero", &[], int, DeclaredFacts::default());
    let flatten = |inner| sources(vec![Source::Flatten { from: Stream::Of(Box::new(inner)) }]);
    let consume =
        Source::Consume { from: Stream::Param(0), init: Step::Declared(zero), update: Step::Param(0), finish: None };
    let g = f.declare("g", &[int], int, flatten(Source::Stepwise));
    let h = f.declare("h", &[int], int, flatten(consume));
    assert_eq!(
        f.refusals(),
        vec![
            Refused::StreamOfNoStreamMadeFromAnother { declared: own(g) },
            Refused::StreamOfNoStreamMadeFromAnother { declared: own(h) },
        ]
    );
}

/// extern-fn §4: a declaration that states nothing of how a call ends
/// leaves it unstated.
#[test]
fn how_a_call_ends_is_unstated_by_default() {
    assert_eq!(DeclaredFacts::default().ends, Ends::Unstated);
    let mut f = Fixture::new();
    let int = f.int();
    let g = f.declare("g", &[int], int, DeclaredFacts::default());
    assert_eq!(f.verified().call_facts(g, None).ends, Ends::Unstated);
}

/// extern-fn §4: how a call's own code ends, stated on a declaration or
/// on its impl, is kept through `verify`, on a call given a closure
/// argument as on any other.
#[test]
fn a_stated_ending_is_kept_through_verify() {
    let mut f = Fixture::new();
    let int = f.int();
    let returns = f.declare("returns", &[int], int, ending(Ends::Returns));
    let total = f.declare("total", &[int, int], int, ending(Ends::Total));
    let by_impl = f.declare_generic("by_impl", &[int], int, ending(Ends::Total));
    let callback = f.function(&[int], int);
    let calls_back = f.declare("calls_back", &[callback], int, ending(Ends::Total));
    let decls = f.verified();
    assert_eq!(decls.call_facts(returns, None).ends, Ends::Returns);
    assert_eq!(decls.call_facts(total, None).ends, Ends::Total);
    assert_eq!(decls.call_facts(by_impl, Some(0)).ends, Ends::Total);
    assert_eq!(decls.call_facts(calls_back, None).ends, Ends::Total);
}
