use acvus_utils::{Astr, Interner, QualifiedRef};

use super::*;
use crate::decl::{
    CastRule, CastTree, Core, Declarations, Head, LoopMarkers, RawDeclarations, Requirement, Scheme, TyVarBound,
};
use crate::flows::{Alignment, Flow, FlowEnd, FlowTerm, Flows};
use crate::ty::{
    Calls, Concrete, Contexts, Cost, Effect, EffectTerm, Holds, IntTy, Label, Mutability, ObjectTy, Poly, Reissue,
    SplitCost, TyKind, TypeArg, Types,
};

struct Fixture {
    infer: Infer,
    names: Interner,
}

impl Fixture {
    fn new() -> Self {
        Self {
            infer: Infer::new(),
            names: Interner::new(),
        }
    }

    fn name(&self, name: &str) -> Astr {
        self.names.intern(name)
    }

    fn kind(&mut self, kind: TyKind<crate::ty::Infer>) -> Term {
        self.infer.intern(kind)
    }

    fn int(&mut self, width: IntTy) -> Term {
        self.kind(TyKind::Int(width))
    }

    fn object(&mut self, fields: &[(&str, Term)], label: Label) -> Term {
        let mut fields: Vec<(Astr, Term)> = fields.iter().map(|(name, ty)| (self.name(name), *ty)).collect();
        fields.sort_by_key(|(name, _)| *name);
        self.kind(TyKind::Object(ObjectTy::Structural(fields.into(), label)))
    }

    fn r#enum(&mut self, name: &str, tags: &[&str], label: Label) -> Term {
        let name = QualifiedRef::root(self.name(name));
        let mut variants: Vec<(Astr, Option<Term>)> = tags.iter().map(|tag| (self.name(tag), None)).collect();
        variants.sort_by_key(|(tag, _)| *tag);
        self.kind(TyKind::Enum {
            name,
            variants: variants.into(),
            label,
        })
    }

    fn function(&mut self, effect: EffectTerm<crate::ty::Infer>, flows: Flows, calls: Calls, holds: Holds) -> Term {
        let ret = self.kind(TyKind::Unit);
        self.kind(TyKind::Fn {
            params: Box::new([]),
            ret,
            effect,
            flows: FlowTerm::Known(flows),
            calls,
            holds,
        })
    }

    /// `term` behind a fresh variable, as the program's terms are.
    fn held(&mut self, term: Term) -> Term {
        let var = self.infer.fresh(Bound::Any);
        self.infer.eq(var, term).expect("a fresh variable binds");
        var
    }

    fn fields_of(&self, term: Term) -> (Vec<Astr>, Label) {
        match self.infer.types().kind(self.infer.shallow(term)) {
            TyKind::Object(ObjectTy::Structural(fields, label)) => (fields.iter().map(|(name, _)| *name).collect(), *label),
            other => panic!("not a structural object: {other:?}"),
        }
    }

    fn freeze(&self, term: Term) -> (Types<Concrete>, crate::ty::Ty) {
        let mut into = Types::new();
        let ty = self
            .infer
            .freeze(&mut into, &mut FreezeMemo::default(), term)
            .expect("settled");
        (into, ty)
    }
}

struct Open(bool);

impl OpenSources for Open {
    fn may_change(&self, _term: Term) -> bool {
        self.0
    }
}

fn effect(reissue: Reissue, reads: &[QualifiedRef], split: SplitCost) -> Effect {
    Effect::new(
        reissue,
        false,
        reads.iter().copied().collect(),
        Contexts::new(),
        split,
    )
}

// eq (types §4, §8, §1)

/// types §4: `eq` binds an unbound variable to the other term.
#[test]
fn eq_binds_an_unbound_variable() {
    let mut fx = Fixture::new();
    let var = fx.infer.fresh(Bound::Any);
    let int = fx.int(IntTy::I64);
    fx.infer.eq(var, int).unwrap();
    assert_eq!(fx.infer.shallow(var), int);
}

/// types §4: unification with the occurs check; a variable never contains
/// itself.
#[test]
fn eq_refuses_a_variable_that_would_contain_itself() {
    let mut fx = Fixture::new();
    let var = fx.infer.fresh(Bound::Any);
    let option = fx.kind(TyKind::Option(var));
    let TyKind::Var(id) = *fx.infer.types().kind(var) else { unreachable!() };
    assert_eq!(fx.infer.eq(var, option), Err(Mismatch::Occurs { var: id, in_term: option }));
}

/// types §4: every constructor is invariant, so two arguments must be one.
#[test]
fn every_constructor_is_invariant() {
    let mut fx = Fixture::new();
    let (i64_, i32_) = (fx.int(IntTy::I64), fx.int(IntTy::I32));
    let (a, b) = (fx.kind(TyKind::Option(i64_)), fx.kind(TyKind::Option(i32_)));
    assert_eq!(fx.infer.eq(a, b), Err(Mismatch::Heads { a: i64_, b: i32_ }));
    let result = fx.kind(TyKind::Result(i64_, i64_));
    assert_eq!(fx.infer.eq(a, result), Err(Mismatch::Heads { a, b: result }));
}

/// types §8: `#τ` and `τ` are different types.
#[test]
fn a_mark_on_one_side_only_is_a_mismatch() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let plain = fx.kind(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(int)));
    let marked = fx.kind(TyKind::Ref(Mutability::Shared, TypeArg::Specialized(int)));
    assert_eq!(fx.infer.eq(plain, marked), Err(Mismatch::Mark { a: plain, b: marked }));
}

/// types §8: an open `#` is bound by `eq`.
#[test]
fn eq_binds_an_open_mark() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let mark = fx.infer.fresh_repr();
    let open = fx.kind(TyKind::Ref(Mutability::Shared, TypeArg::Open(mark, int)));
    let marked = fx.kind(TyKind::Ref(Mutability::Shared, TypeArg::Specialized(int)));
    fx.infer.eq(open, marked).unwrap();
    let (types, frozen) = fx.freeze(open);
    assert!(matches!(
        types.kind(frozen),
        TyKind::Ref(Mutability::Shared, TypeArg::Specialized(_))
    ));
}

/// types §1: sameness is decided by shape; labels never make two equal
/// shapes unequal.
#[test]
fn labels_never_make_two_equal_shapes_unequal() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let transparent = fx.object(&[("x", int)], Label::Transparent);
    let opaque = fx.object(&[("x", int)], Label::Opaque);
    assert_eq!(fx.infer.eq(transparent, opaque), Ok(()));
}

/// types §1, §6: a shape made one with an opaque shape is opaque, since
/// nothing may grow what one side cannot grow.
#[test]
fn a_shape_made_one_with_an_opaque_one_becomes_opaque() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let transparent = fx.object(&[("x", int)], Label::Transparent);
    let program = fx.held(transparent);
    let opaque = fx.object(&[("x", int)], Label::Opaque);
    fx.infer.eq(program, opaque).unwrap();
    let wider = fx.object(&[("x", int), ("y", int)], Label::Transparent);
    let wider = fx.held(wider);
    let y = fx.name("y");
    assert_eq!(
        fx.infer.join(program, wider),
        Err(Mismatch::OpaqueLacksField { object: opaque, field: y })
    );
}

/// types §4: two terms made one stay one: what one grows by later, the
/// other grows by too.
#[test]
fn two_variables_made_one_stay_one_as_either_grows() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let x = fx.object(&[("x", int)], Label::Transparent);
    let (a, b) = (fx.held(x), fx.held(x));
    fx.infer.eq(a, b).unwrap();
    let y = fx.object(&[("y", int)], Label::Transparent);
    let c = fx.held(y);
    fx.infer.join(a, c).unwrap();
    assert_eq!(fx.fields_of(b), (vec![fx.name("x"), fx.name("y")], Label::Transparent));
}

// Bounds (extern-fn §1, types §5)

/// extern-fn §1: an integer bound admits an integer type among its widths,
/// signed only where the bound is.
#[test]
fn an_integer_bound_admits_only_its_widths() {
    let mut fx = Fixture::new();
    let among: Box<[IntTy]> = Box::new([IntTy::U8, IntTy::I8]);
    let bounded = |fx: &mut Fixture| {
        fx.infer.fresh(Bound::Integer {
            signed: false,
            among: among.clone(),
        })
    };
    let (u8_, i8_, u16_, string) = (
        fx.int(IntTy::U8),
        fx.int(IntTy::I8),
        fx.int(IntTy::U16),
        fx.kind(TyKind::String),
    );
    let var = bounded(&mut fx);
    assert_eq!(fx.infer.eq(var, u8_), Ok(()));
    for refused in [i8_, u16_, string] {
        let var = bounded(&mut fx);
        let TyKind::Var(id) = *fx.infer.types().kind(var) else { unreachable!() };
        assert_eq!(fx.infer.eq(var, refused), Err(Mismatch::Bound { var: id, term: refused }));
    }
}

/// extern-fn §1, types §5: a settled bound is never taken as `!`; a
/// literal's width is an integer's.
#[test]
fn a_settled_bound_refuses_never_and_a_literal_admits_integers() {
    let mut fx = Fixture::new();
    let never = fx.kind(TyKind::Never);
    let settled = fx.infer.fresh(Bound::Settled);
    assert!(matches!(fx.infer.eq(settled, never), Err(Mismatch::Bound { .. })));
    let literal = fx.infer.fresh(Bound::Literal);
    let float = fx.kind(TyKind::Float);
    assert!(matches!(fx.infer.eq(literal, float), Err(Mismatch::Bound { .. })));
    let u32_ = fx.int(IntTy::U32);
    assert_eq!(fx.infer.eq(literal, u32_), Ok(()));
}

/// extern-fn §1: two bounded variables made one admit what both bounds
/// admit, and none where no type is admitted by both.
#[test]
fn two_bounded_variables_made_one_admit_what_both_admit() {
    let mut fx = Fixture::new();
    let a = fx.infer.fresh(Bound::Integer {
        signed: true,
        among: Box::new([IntTy::I32, IntTy::I64]),
    });
    let b = fx.infer.fresh(Bound::Integer {
        signed: true,
        among: Box::new([IntTy::I64, IntTy::U8]),
    });
    fx.infer.eq(a, b).unwrap();
    let i32_ = fx.int(IntTy::I32);
    assert!(matches!(fx.infer.eq(a, i32_), Err(Mismatch::Bound { .. })));
    let i64_ = fx.int(IntTy::I64);
    assert_eq!(fx.infer.eq(a, i64_), Ok(()));

    let c = fx.infer.fresh(Bound::Integer {
        signed: false,
        among: Box::new([IntTy::U8]),
    });
    let d = fx.infer.fresh(Bound::Integer {
        signed: true,
        among: Box::new([IntTy::I8]),
    });
    assert!(matches!(fx.infer.eq(c, d), Err(Mismatch::Bound { .. })));

    let settled = fx.infer.fresh(Bound::Settled);
    let any = fx.infer.fresh(Bound::Any);
    fx.infer.eq(settled, any).unwrap();
    let never = fx.kind(TyKind::Never);
    assert!(matches!(fx.infer.eq(any, never), Err(Mismatch::Bound { .. })));
}

// join (types §2)

/// types §2: two object types join at the union of their fields.
#[test]
fn objects_join_at_the_union_of_their_fields() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let x = fx.object(&[("x", int)], Label::Transparent);
    let y = fx.object(&[("y", int)], Label::Transparent);
    let (a, b) = (fx.held(x), fx.held(y));
    fx.infer.join(a, b).unwrap();
    let union = (vec![fx.name("x"), fx.name("y")], Label::Transparent);
    assert_eq!(fx.fields_of(a), union);
    assert_eq!(fx.fields_of(b), union);
    assert_eq!(fx.infer.eq(a, b), Ok(()));
}

/// types §1, §2: an opaque side never grows, so a join past what it has
/// is a mismatch.
#[test]
fn an_opaque_side_never_grows() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let wide = fx.object(&[("x", int), ("y", int)], Label::Transparent);
    let narrow = fx.object(&[("x", int)], Label::Opaque);
    let (a, b) = (fx.held(wide), fx.held(narrow));
    let y = fx.name("y");
    assert_eq!(
        fx.infer.join(a, b),
        Err(Mismatch::OpaqueLacksField { object: narrow, field: y })
    );
}

/// types §1, §2: a transparent side grows to an opaque one, and is
/// opaque after.
#[test]
fn a_transparent_side_grows_to_an_opaque_one() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let narrow = fx.object(&[("x", int)], Label::Transparent);
    let wide = fx.object(&[("x", int), ("y", int)], Label::Opaque);
    let (a, b) = (fx.held(narrow), fx.held(wide));
    fx.infer.join(a, b).unwrap();
    assert_eq!(fx.infer.shallow(a), wide);
}

/// types §2: a transparent node inside a term grows through the variable
/// that holds the term.
#[test]
fn a_nested_transparent_node_grows_through_the_variable_above_it() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let x = fx.object(&[("x", int)], Label::Transparent);
    let y = fx.object(&[("y", int)], Label::Transparent);
    let (some_x, some_y) = (fx.kind(TyKind::Option(x)), fx.kind(TyKind::Option(y)));
    let (a, b) = (fx.held(some_x), fx.held(some_y));
    fx.infer.join(a, b).unwrap();
    let TyKind::Option(inner) = *fx.infer.types().kind(fx.infer.shallow(a)) else {
        panic!("an option")
    };
    assert_eq!(fx.fields_of(inner), (vec![fx.name("x"), fx.name("y")], Label::Transparent));
}

/// types §2: two enums of one name join at the union of their variants.
#[test]
fn enums_of_one_name_join_at_the_union_of_their_variants() {
    let mut fx = Fixture::new();
    let a = fx.r#enum("E", &["A"], Label::Transparent);
    let b = fx.r#enum("E", &["B"], Label::Transparent);
    let union = fx.r#enum("E", &["A", "B"], Label::Transparent);
    let (a, b) = (fx.held(a), fx.held(b));
    fx.infer.join(a, b).unwrap();
    assert_eq!(fx.infer.shallow(a), union);
    assert_eq!(fx.infer.shallow(b), union);
}

/// types §2: enums of two names do not join.
#[test]
fn enums_of_two_names_do_not_join() {
    let mut fx = Fixture::new();
    let a = fx.r#enum("E", &["A"], Label::Transparent);
    let b = fx.r#enum("F", &["A"], Label::Transparent);
    let (a, b) = (fx.held(a), fx.held(b));
    let (e, f) = (QualifiedRef::root(fx.name("E")), QualifiedRef::root(fx.name("F")));
    assert_eq!(fx.infer.join(a, b), Err(Mismatch::TwoEnums { a: e, b: f }));
}

/// types §1, §2: a closed enum never grows by a variant it lacks.
#[test]
fn a_closed_enum_does_not_grow_by_a_variant() {
    let mut fx = Fixture::new();
    let open = fx.r#enum("E", &["A", "B"], Label::Transparent);
    let closed = fx.r#enum("E", &["A"], Label::Opaque);
    let (a, b) = (fx.held(open), fx.held(closed));
    let tag = fx.name("B");
    assert_eq!(
        fx.infer.join(a, b),
        Err(Mismatch::OpaqueLacksVariant { r#enum: closed, tag })
    );
}

/// types §2, effects §6: function types join at the join of their effects
/// and of their flows, are once where either is, hold the stronger loan,
/// and split only where both sides do, at one cost.
#[test]
fn function_types_join_effects_flows_calls_holds_and_split() {
    let mut fx = Fixture::new();
    let (r, s) = (QualifiedRef::root(fx.name("r")), QualifiedRef::root(fx.name("s")));
    let cost = SplitCost::WithCost(Cost::stated(3));
    let result_from = |from| Flow {
        to: FlowEnd::Result,
        from,
        alignment: Alignment::Aligned,
    };
    let f = fx.function(
        EffectTerm::Known(effect(Reissue::Pure, &[r], cost)),
        Flows::of([result_from(FlowEnd::Param(0))]),
        Calls::Many,
        Holds::Nothing,
    );
    let g = fx.function(
        EffectTerm::Known(effect(Reissue::Idempotent, &[s], cost)),
        Flows::of([result_from(FlowEnd::Captures)]),
        Calls::Once,
        Holds::Shared,
    );
    let (a, b) = (fx.held(f), fx.held(g));
    fx.infer.join(a, b).unwrap();
    let (types, frozen) = fx.freeze(a);
    let TyKind::Fn {
        effect: EffectTerm::Known(joined),
        flows,
        calls,
        holds,
        ..
    } = types.kind(frozen)
    else {
        panic!("a function type")
    };
    assert_eq!(joined, &effect(Reissue::Idempotent, &[r, s], cost));
    assert_eq!(
        flows.get(),
        &Flows::of([result_from(FlowEnd::Param(0)), result_from(FlowEnd::Captures)])
    );
    assert_eq!((*calls, *holds), (Calls::Once, Holds::Shared));

    let other_cost = SplitCost::WithCost(Cost::stated(4));
    let h = fx.function(
        EffectTerm::Known(effect(Reissue::Pure, &[], other_cost)),
        Flows::none(),
        Calls::Many,
        Holds::Nothing,
    );
    let c = fx.held(h);
    fx.infer.join(a, c).unwrap();
    let (types, frozen) = fx.freeze(c);
    let TyKind::Fn {
        effect: EffectTerm::Known(joined),
        ..
    } = types.kind(frozen)
    else {
        panic!("a function type")
    };
    assert_eq!(joined.split(), SplitCost::Unavailable);
}

// Effects (types §2)

/// types §2: the effects form a lattice: their join is commutative,
/// associative and idempotent.
#[test]
fn the_effect_join_is_a_lattice_join() {
    let mut fx = Fixture::new();
    let r = QualifiedRef::root(fx.name("r"));
    let terms = [
        EffectTerm::Known(Effect::pure()),
        EffectTerm::Known(effect(Reissue::Idempotent, &[r], SplitCost::Unavailable)),
        EffectTerm::Known(Effect::opaque()),
    ];
    for a in &terms {
        assert_eq!(&fx.infer.join_effects(a, a), a);
        for b in &terms {
            assert_eq!(fx.infer.join_effects(a, b), fx.infer.join_effects(b, a));
            for c in &terms {
                let left = fx.infer.join_effects(a, b);
                let right = fx.infer.join_effects(b, c);
                assert_eq!(fx.infer.join_effects(&left, c), fx.infer.join_effects(a, &right));
            }
        }
    }
}

/// types §2: an effect variable's value is the join of what was joined
/// into it.
#[test]
fn an_effect_variable_is_the_join_of_what_was_joined_into_it() {
    let mut fx = Fixture::new();
    let r = QualifiedRef::root(fx.name("r"));
    let var = EffectTerm::Var(fx.infer.fresh_effect());
    let reads = EffectTerm::Known(effect(Reissue::Pure, &[r], SplitCost::Unavailable));
    let idempotent = EffectTerm::Known(effect(Reissue::Idempotent, &[], SplitCost::Unavailable));
    assert_eq!(fx.infer.join_effects(&var, &reads), var);
    assert_eq!(fx.infer.join_effects(&idempotent, &var), var);
    let f = fx.function(var, Flows::none(), Calls::Many, Holds::Nothing);
    let (types, frozen) = fx.freeze(f);
    let TyKind::Fn {
        effect: EffectTerm::Known(settled),
        ..
    } = types.kind(frozen)
    else {
        panic!("a function type")
    };
    assert_eq!(settled, &effect(Reissue::Idempotent, &[r], SplitCost::Unavailable));
}

/// types §4: `eq` of two effect terms binds, and two effect variables made
/// one stay one.
#[test]
fn eq_of_two_effect_variables_binds() {
    let mut fx = Fixture::new();
    let (v, w) = (EffectTerm::Var(fx.infer.fresh_effect()), EffectTerm::Var(fx.infer.fresh_effect()));
    let f = fx.function(v, Flows::none(), Calls::Many, Holds::Nothing);
    let g = fx.function(w.clone(), Flows::none(), Calls::Many, Holds::Nothing);
    fx.infer.eq(f, g).unwrap();
    fx.infer.join_effects(&w, &EffectTerm::Known(Effect::opaque()));
    let (types, frozen) = fx.freeze(f);
    let TyKind::Fn {
        effect: EffectTerm::Known(settled),
        ..
    } = types.kind(frozen)
    else {
        panic!("a function type")
    };
    assert_eq!(settled, &Effect::opaque());

    let pure = fx.function(EffectTerm::Known(Effect::pure()), Flows::none(), Calls::Many, Holds::Nothing);
    assert_eq!(fx.infer.eq(f, pure), Err(Mismatch::Heads { a: f, b: pure }));
}

// closed (types §6)

/// types §6: an unbound variable's head is not known, so it is not closed.
#[test]
fn an_unbound_variable_is_not_closed() {
    let mut fx = Fixture::new();
    let var = fx.infer.fresh(Bound::Any);
    assert_eq!(fx.infer.closed(&CastTree::default(), &Open(false), var), None);
}

/// types §6: a head at the top of its cast tree is closed, whatever an open
/// source may still do.
#[test]
fn the_top_of_the_cast_tree_is_closed() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let held = fx.held(int);
    assert_eq!(
        fx.infer.closed(&CastTree::default(), &Open(true), held),
        Some(Closed {
            head: Head::Int(IntTy::I64),
            term: int
        })
    );
}

// freeze (compilation)

/// compilation: an unbound variable at freeze is `Unsettled`; none is
/// picked.
#[test]
fn an_unbound_variable_at_freeze_is_unsettled() {
    let mut fx = Fixture::new();
    let var = fx.infer.fresh(Bound::Literal);
    let TyKind::Var(id) = *fx.infer.types().kind(var) else { unreachable!() };
    let option = fx.kind(TyKind::Option(var));
    let mut into = Types::new();
    assert_eq!(
        fx.infer.freeze(&mut into, &mut FreezeMemo::default(), option),
        Err(Unsettled { var: id })
    );
}

/// compilation: freezing follows bound variables, and a term already
/// frozen is answered from the memo.
#[test]
fn freeze_follows_variables_and_answers_a_frozen_term_from_the_memo() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let object = fx.object(&[("x", int)], Label::Opaque);
    let held = fx.held(object);
    let pair = fx.kind(TyKind::Tuple(Box::new([held, object])));
    let mut into = Types::new();
    let mut memo = FreezeMemo::default();
    let frozen = fx.infer.freeze(&mut into, &mut memo, pair).unwrap();
    let TyKind::Tuple(items) = into.kind(frozen).clone() else {
        panic!("a tuple")
    };
    assert_eq!(items[0], items[1]);
    let x = fx.name("x");
    let expected_int = into.intern(TyKind::Int(IntTy::I64));
    assert_eq!(
        into.kind(items[0]),
        &TyKind::Object(ObjectTy::Structural(Box::new([(x, expected_int)]), ()))
    );
    assert_eq!(fx.infer.freeze(&mut into, &mut memo, pair), Ok(frozen));
}

// enter (types §3, §7)

/// types §6, §7: a value whose head is not known yet does not enter.
#[test]
fn a_value_of_no_known_head_does_not_enter_yet() {
    let mut fx = Fixture::new();
    let value = fx.infer.fresh(Bound::Any);
    let slot = fx.infer.fresh(Bound::Any);
    assert_eq!(fx.infer.enter(&CastTree::default(), value, slot), Ok(Entered::NotYet));
}

// Tests that build a `Declarations`: they wait on `Declarations::verify`.

fn pure_fn(types: &mut Types<Poly>, params: Vec<crate::ty::ParamTerm<Poly>>, ret: crate::ty::TyId<Poly>) -> crate::ty::TyId<Poly> {
    types.intern(TyKind::Fn {
        params: params.into(),
        ret,
        effect: EffectTerm::Known(Effect::pure()),
        flows: FlowTerm::Known(Flows::Every),
        calls: Calls::Many,
        holds: Holds::Nothing,
    })
}

fn raw(names: &Interner, types: Types<Poly>, functions: Vec<(QualifiedRef, Scheme)>, casts: Vec<CastRule>) -> RawDeclarations {
    RawDeclarations {
        types,
        functions,
        user_types: Vec::new(),
        casts,
        contexts: Vec::new(),
        inputs: Vec::new(),
        loop_markers: LoopMarkers::default(),
        core: Core::of(names),
    }
}

/// extern-fn §6, types §1: a scheme at one use has a fresh variable per
/// type parameter, with its declared bound, standing as is wherever the
/// parameter does; every structural node is opaque; a requirement comes
/// back at the use's own variables.
#[test]
fn instantiate_builds_opaque_nodes_at_fresh_variables() {
    let mut fx = Fixture::new();
    let (f, shared, x, a, b) = (
        QualifiedRef::root(fx.name("f")),
        QualifiedRef::root(fx.name("shared")),
        fx.name("x"),
        fx.name("a"),
        fx.name("b"),
    );
    let mut types = Types::<Poly>::new();
    let t = types.intern(TyKind::Var(0));
    let object = types.intern(TyKind::Object(ObjectTy::Structural(Box::new([(x, t)]), ())));
    let option = types.intern(TyKind::Option(t));
    let signature = pure_fn(
        &mut types,
        vec![
            crate::ty::ParamTerm { name: a, ty: object },
            crate::ty::ParamTerm { name: b, ty: t },
        ],
        option,
    );
    let integer = TyVarBound::Integer {
        signed: true,
        among: Box::new([IntTy::I32, IntTy::I64]),
    };
    let scheme = Scheme {
        ty: signature,
        bounds: Box::new([integer]),
        instances: None,
        requires: Box::new([Requirement {
            signature: shared,
            pattern: t,
            law: None,
        }]),
    };
    let decls = Declarations::verify(raw(&fx.names, types, vec![(f, scheme)], Vec::new())).expect("verified");
    let scheme = decls.scheme(f).expect("declared");

    let used = fx.infer.instantiate(&decls, scheme);

    let TyKind::Fn { params, ret, .. } = fx.infer.types().kind(used.ty).clone() else {
        panic!("a function type")
    };
    let var = params[1].ty;
    let TyKind::Var(id) = *fx.infer.types().kind(var) else {
        panic!("a type parameter's place is its variable")
    };
    assert_eq!(
        fx.infer.types().kind(params[0].ty),
        &TyKind::Object(ObjectTy::Structural(Box::new([(x, var)]), Label::Opaque))
    );
    assert_eq!(fx.infer.types().kind(ret), &TyKind::Option(var));
    assert_eq!(
        fx.infer.unbound().collect::<Vec<_>>(),
        vec![(
            id,
            &Bound::Integer {
                signed: true,
                among: Box::new([IntTy::I32, IntTy::I64])
            }
        )]
    );
    assert_eq!(
        used.requirements,
        vec![RequiredAt {
            signature: shared,
            pattern: var,
            law: None
        }]
    );
    let again = fx.infer.instantiate(&decls, scheme);
    assert_ne!(again.ty, used.ty, "each use has its own variables");
}

/// types §6: a head with a step up is closed only once no open source may
/// change it.
#[test]
fn a_head_with_a_step_up_is_closed_only_when_nothing_may_change_it() {
    let mut fx = Fixture::new();
    let (widen, v) = (QualifiedRef::root(fx.name("widen")), fx.name("v"));
    let mut types = Types::<Poly>::new();
    let (i32_, i64_) = (types.intern(TyKind::Int(IntTy::I32)), types.intern(TyKind::Int(IntTy::I64)));
    let signature = pure_fn(&mut types, vec![crate::ty::ParamTerm { name: v, ty: i32_ }], i64_);
    let scheme = Scheme {
        ty: signature,
        bounds: Box::new([]),
        instances: None,
        requires: Box::new([]),
    };
    let cast = CastRule {
        from: i32_,
        to: i64_,
        via: widen,
    };
    let decls = Declarations::verify(raw(&fx.names, types, vec![(widen, scheme)], vec![cast])).expect("verified");

    let int = fx.int(IntTy::I32);
    let held = fx.held(int);
    assert_eq!(fx.infer.closed(decls.casts(), &Open(true), held), None);
    assert_eq!(
        fx.infer.closed(decls.casts(), &Open(false), held),
        Some(Closed {
            head: Head::Int(IntTy::I32),
            term: int
        })
    );
}
