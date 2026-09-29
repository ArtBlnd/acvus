use acvus_utils::{Interner, QualifiedRef};

use super::*;
use crate::decl::{
    Core, Declarations, DeclaredFacts, GenericImplSig, ImplSig, Impls, LoopMarkers, RawDeclarations, Scheme,
};
use crate::ty::{Calls, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Mutability, TyKind, TypeArg, Types};

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
                inputs: Vec::new(),
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
        self.ty(TyKind::Ref(mutability, TypeArg::Uniform(lent)))
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
        let impls = Impls { concrete: Box::new([]), generic: Some(GenericImplSig { facts }) };
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
        let r#impl = ImplSig { ty, param_names: self.param_names(2), requires: Box::new([]), facts };
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
    Update { place, combine, identity: None, commutative: false, inverse_of: None }
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

fn own(f: QualifiedRef) -> Declared {
    Declared { function: f, r#impl: None }
}

/// One form of each kind, well formed: a store into a map's entry and a
/// read of a slice's element, an update by an associative combine with its
/// identity and inverse, a total order, a map of a stream, a relation over
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
        effect_args: Box::new([]),
        identity_args: Box::new([]),
    });
    f.raw.user_types.push(crate::decl::UserTypeDecl {
        id: QualifiedRef::root(f.interner.intern("Map")),
        type_params: Box::new([]),
        specializes: Box::new([]),
        effect_params: 0,
        identity_params: 0,
        lifetimes: 0,
    });
    let mut_map = f.reference(Mutability::Mut, map);

    let stored = Touch { place: Place { param: 0, element: Some(Element::Key(1)) }, mode: Mode::Store(Stored::Value(2)) };
    let read = Touch { place: Place { param: 3, element: Some(Element::Index(4)) }, mode: Mode::Read };
    f.declare("insert", &[mut_map, int, int, shared_slice, index], unit, touches(&[stored, read]));

    let add_into = f.declare("add_into", &[mut_int, int], unit, DeclaredFacts::default());
    let zero = f.declare("zero", &[], int, DeclaredFacts::default());
    let increment = Update { identity: Some(zero), commutative: true, ..update(whole(0), add_into) };
    let add = f.declare("add", &[mut_int, int], unit, actions(vec![Action::Update(increment)]));
    let decrement = Update { inverse_of: Some(add), ..update(whole(0), add_into) };
    f.declare("sub", &[mut_int, int], unit, actions(vec![Action::Update(decrement)]));
    f.declare("compare", &[shared_int, shared_int], int, actions(vec![Action::TotalOrder]));

    let through = f.function(&[int], int);
    let map_stream = Source::Map { from: 0, through: 1 };
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
    let append = f.declare("append", &[mut_string, string], unit, DeclaredFacts::default());
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
    let add_into = f.declare("add_into", &[mut_int, int], unit, DeclaredFacts::default());
    let empty = f.declare("empty", &[], string, DeclaredFacts::default());
    let with_identity = Update { identity: Some(empty), ..update(whole(0), add_into) };
    let g = f.declare("g", &[mut_int, int], unit, actions(vec![Action::Update(with_identity)]));
    assert_eq!(f.refusals(), vec![Refused::IdentityDoesNotFit { declared: own(g), identity: empty }]);
}

#[test]
fn an_inverse_of_a_function_that_updates_no_such_place_is_refused() {
    let mut f = Fixture::new();
    let int = f.int();
    let unit = f.ty(TyKind::Unit);
    let mut_int = f.reference(Mutability::Mut, int);
    let add_into = f.declare("add_into", &[mut_int, int], unit, DeclaredFacts::default());
    let inverse = Update { inverse_of: Some(add_into), ..update(whole(0), add_into) };
    let g = f.declare("g", &[mut_int, int], unit, actions(vec![Action::Update(inverse)]));
    assert_eq!(
        f.refusals(),
        vec![Refused::InverseNotAnUpdateAtThePlace { declared: own(g), inverse: add_into }]
    );
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
    let g = f.declare("g", &[int, predicate], int, sources(vec![Source::Filter { from: 0, predicate: 1 }]));
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
    assert_eq!(decls.call_facts(g, 0), &read);
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
