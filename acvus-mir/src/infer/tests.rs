use acvus_utils::{Astr, Interner, LocalIdOps, QualifiedRef};

use super::*;
use crate::decl::{
    CastRule, CastTree, ConversionRule, Core, Declarations, DeclaredFacts, Head, LoopMarkers, RawDeclarations, Requirement, Scheme, TyVarBound,
};
use crate::ty::{
    Alignment, Calls, CallsTerm, Concrete, Contexts, Cost, Effect, EffectTerm, Flow, FlowEnd, FlowTerm, Flows, Holds, HoldsTerm, IdentityId,
    IdentityTerm, IntTy, Label, Mutability, ObjectTy, Poly, Reach, Reissue, SplitCost, TyKind, TypeArg, Types,
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
        self.function_of(effect, flows, CallsTerm::Known(calls), HoldsTerm::Known(holds))
    }

    fn function_of(
        &mut self,
        effect: EffectTerm<crate::ty::Infer>,
        flows: Flows,
        calls: CallsTerm,
        holds: HoldsTerm,
    ) -> Term {
        let ret = self.kind(TyKind::Unit);
        let values = self.infer.fresh_values();
        self.kind(TyKind::Fn {
            params: Box::new([]),
            ret,
            effect,
            flows: FlowTerm::Known(flows),
            calls,
            holds,
            values,
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

    fn may_grow(&self, _term: Term) -> bool {
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
    let option = fx.kind(TyKind::Option(var, Label::Here));
    let TyKind::Var(id) = *fx.infer.types().kind(var) else { unreachable!() };
    assert_eq!(fx.infer.eq(var, option), Err(Mismatch::Occurs { var: id, in_term: option }));
}

/// types §4: a reference's target is never a reference, so a variable
/// under `&` is not bound to one.
#[test]
fn a_variable_under_a_reference_is_not_bound_to_a_reference() {
    let mut fx = Fixture::new();
    let var = fx.infer.fresh(Bound::Any);
    fx.kind(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(var)));
    let int = fx.int(IntTy::I64);
    let reference = fx.kind(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(int)));
    assert_eq!(
        fx.infer.eq(var, reference),
        Err(Mismatch::ReferenceToReference { target: reference })
    );
}

/// types §4: two variables made one keep the mark of a reference's target
/// if either has it, so the other is not bound to a reference either.
#[test]
fn a_variable_linked_to_a_references_target_is_not_bound_to_a_reference() {
    let mut fx = Fixture::new();
    let (target, other) = (fx.infer.fresh(Bound::Any), fx.infer.fresh(Bound::Any));
    fx.kind(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(target)));
    fx.infer.eq(target, other).expect("two unbound variables link");
    let int = fx.int(IntTy::I64);
    let reference = fx.kind(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(int)));
    assert_eq!(
        fx.infer.eq(other, reference),
        Err(Mismatch::ReferenceToReference { target: reference })
    );
    assert_eq!(fx.infer.eq(other, int), Ok(()));
}

/// types §4: every constructor is invariant, so two arguments must be one.
#[test]
fn every_constructor_is_invariant() {
    let mut fx = Fixture::new();
    let (i64_, i32_) = (fx.int(IntTy::I64), fx.int(IntTy::I32));
    let (a, b) = (fx.kind(TyKind::Option(i64_, Label::Here)), fx.kind(TyKind::Option(i32_, Label::Here)));
    assert_eq!(fx.infer.eq(a, b), Err(Mismatch::Heads { a: i64_, b: i32_ }));
    let result = fx.kind(TyKind::Result(i64_, i64_, Label::Here));
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
    let transparent = fx.object(&[("x", int)], Label::Here);
    let opaque = fx.object(&[("x", int)], Label::Elsewhere);
    assert_eq!(fx.infer.eq(transparent, opaque), Ok(()));
}

/// types §1, §6: a shape made one with an opaque shape is opaque, since
/// nothing may grow what one side cannot grow.
#[test]
fn a_shape_made_one_with_an_opaque_one_becomes_opaque() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let transparent = fx.object(&[("x", int)], Label::Here);
    let program = fx.held(transparent);
    let opaque = fx.object(&[("x", int)], Label::Elsewhere);
    fx.infer.eq(program, opaque).unwrap();
    let wider = fx.object(&[("x", int), ("y", int)], Label::Here);
    let wider = fx.held(wider);
    let y = fx.name("y");
    assert_eq!(
        fx.infer.join(program, wider, Reach::Named),
        Err(Mismatch::OpaqueLacksField { object: opaque, field: y })
    );
}

/// types §4: two terms made one stay one: what one grows by later, the
/// other grows by too.
#[test]
fn two_variables_made_one_stay_one_as_either_grows() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let x = fx.object(&[("x", int)], Label::Here);
    let (a, b) = (fx.held(x), fx.held(x));
    fx.infer.eq(a, b).unwrap();
    let y = fx.object(&[("y", int)], Label::Here);
    let c = fx.held(y);
    fx.infer.join(a, c, Reach::Named).unwrap();
    assert_eq!(fx.fields_of(b), (vec![fx.name("x"), fx.name("y")], Label::Here));
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
            literal: false,
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

fn literal() -> Bound {
    Bound::Integer {
        signed: true,
        among: IntTy::ALL.into(),
        literal: true,
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
    let literal = fx.infer.fresh(literal());
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
        literal: false,
    });
    let b = fx.infer.fresh(Bound::Integer {
        signed: true,
        among: Box::new([IntTy::I64, IntTy::U8]),
        literal: false,
    });
    fx.infer.eq(a, b).unwrap();
    let i32_ = fx.int(IntTy::I32);
    assert!(matches!(fx.infer.eq(a, i32_), Err(Mismatch::Bound { .. })));
    let i64_ = fx.int(IntTy::I64);
    assert_eq!(fx.infer.eq(a, i64_), Ok(()));

    let c = fx.infer.fresh(Bound::Integer {
        signed: false,
        among: Box::new([IntTy::U8]),
        literal: false,
    });
    let d = fx.infer.fresh(Bound::Integer {
        signed: true,
        among: Box::new([IntTy::I8]),
        literal: false,
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
    let x = fx.object(&[("x", int)], Label::Here);
    let y = fx.object(&[("y", int)], Label::Here);
    let (a, b) = (fx.held(x), fx.held(y));
    fx.infer.join(a, b, Reach::Named).unwrap();
    let union = (vec![fx.name("x"), fx.name("y")], Label::Here);
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
    let wide = fx.object(&[("x", int), ("y", int)], Label::Here);
    let narrow = fx.object(&[("x", int)], Label::Elsewhere);
    let (a, b) = (fx.held(wide), fx.held(narrow));
    let y = fx.name("y");
    assert_eq!(
        fx.infer.join(a, b, Reach::Named),
        Err(Mismatch::OpaqueLacksField { object: narrow, field: y })
    );
}

/// types §1, §2: a transparent side grows to an opaque one, and is
/// opaque after.
#[test]
fn a_transparent_side_grows_to_an_opaque_one() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let narrow = fx.object(&[("x", int)], Label::Here);
    let wide = fx.object(&[("x", int), ("y", int)], Label::Elsewhere);
    let (a, b) = (fx.held(narrow), fx.held(wide));
    fx.infer.join(a, b, Reach::Named).unwrap();
    assert_eq!(fx.infer.shallow(a), wide);
}

/// types §2, §4: `join` grows only a node's own members; a constructor's
/// arguments are related by `eq`, since every constructor is invariant.
#[test]
fn join_relates_a_constructors_arguments_by_eq() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let x = fx.object(&[("x", int)], Label::Here);
    let y = fx.object(&[("y", int)], Label::Here);
    let (some_x, some_y) = (fx.kind(TyKind::Option(x, Label::Here)), fx.kind(TyKind::Option(y, Label::Here)));
    let (a, b) = (fx.held(some_x), fx.held(some_y));
    assert_eq!(fx.infer.join(a, b, Reach::Named), Err(Mismatch::Heads { a: x, b: y }));

    let (inner_x, inner_y) = (fx.held(x), fx.held(y));
    let (some_x, some_y) = (fx.kind(TyKind::Option(inner_x, Label::Here)), fx.kind(TyKind::Option(inner_y, Label::Here)));
    fx.infer.join(inner_x, inner_y, Reach::Named).unwrap();
    let (a, b) = (fx.held(some_x), fx.held(some_y));
    fx.infer.join(a, b, Reach::Named).unwrap();
    assert_eq!(fx.fields_of(inner_y), (vec![fx.name("x"), fx.name("y")], Label::Here));
}

/// types §2, §4: an object's fields are its members, and their types are
/// related by `eq`, not grown.
#[test]
fn a_shared_fields_types_are_related_by_eq() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let x = fx.object(&[("x", int)], Label::Here);
    let y = fx.object(&[("y", int)], Label::Here);
    let (px, py) = (fx.object(&[("p", x)], Label::Here), fx.object(&[("p", y)], Label::Here));
    let (a, b) = (fx.held(px), fx.held(py));
    assert_eq!(fx.infer.join(a, b, Reach::Named), Err(Mismatch::Heads { a: x, b: y }));
}

/// types §2: two enums of one name join at the union of their variants.
#[test]
fn enums_of_one_name_join_at_the_union_of_their_variants() {
    let mut fx = Fixture::new();
    let a = fx.r#enum("E", &["A"], Label::Here);
    let b = fx.r#enum("E", &["B"], Label::Here);
    let union = fx.r#enum("E", &["A", "B"], Label::Here);
    let (a, b) = (fx.held(a), fx.held(b));
    fx.infer.join(a, b, Reach::Named).unwrap();
    assert_eq!(fx.infer.shallow(a), union);
    assert_eq!(fx.infer.shallow(b), union);
}

/// types §2: enums of two names do not join.
#[test]
fn enums_of_two_names_do_not_join() {
    let mut fx = Fixture::new();
    let a = fx.r#enum("E", &["A"], Label::Here);
    let b = fx.r#enum("F", &["A"], Label::Here);
    let (a, b) = (fx.held(a), fx.held(b));
    let (e, f) = (QualifiedRef::root(fx.name("E")), QualifiedRef::root(fx.name("F")));
    assert_eq!(fx.infer.join(a, b, Reach::Named), Err(Mismatch::TwoEnums { a: e, b: f }));
}

/// types §1, §2: a closed enum never grows by a variant it lacks.
#[test]
fn a_closed_enum_does_not_grow_by_a_variant() {
    let mut fx = Fixture::new();
    let open = fx.r#enum("E", &["A", "B"], Label::Here);
    let closed = fx.r#enum("E", &["A"], Label::Elsewhere);
    let (a, b) = (fx.held(open), fx.held(closed));
    let tag = fx.name("B");
    assert_eq!(
        fx.infer.join(a, b, Reach::Named),
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
    fx.infer.join(a, b, Reach::Named).unwrap();
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
    fx.infer.join(a, c, Reach::Named).unwrap();
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
    let mut join = |a: &EffectTerm<crate::ty::Infer>, b: &EffectTerm<crate::ty::Infer>| {
        fx.infer.join_effects(a, b).expect("two known effects join")
    };
    for a in &terms {
        assert_eq!(&join(a, a), a);
        for b in &terms {
            assert_eq!(join(a, b), join(b, a));
            for c in &terms {
                let left = join(a, b);
                let right = join(b, c);
                assert_eq!(join(&left, c), join(a, &right));
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
    assert_eq!(fx.infer.join_effects(&var, &reads), Ok(var.clone()));
    assert_eq!(fx.infer.join_effects(&idempotent, &var), Ok(var.clone()));
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
    let f = fx.function(v.clone(), Flows::none(), Calls::Many, Holds::Nothing);
    let g = fx.function(w.clone(), Flows::none(), Calls::Many, Holds::Nothing);
    fx.infer.eq(f, g).unwrap();
    fx.infer.join_effects(&w, &EffectTerm::Known(Effect::opaque())).expect("an unbound variable grows");
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
    assert_eq!(
        fx.infer.eq(f, pure),
        Err(Mismatch::Effects {
            a: v,
            b: EffectTerm::Known(Effect::pure()),
            a_is: Effect::opaque(),
            b_is: Effect::pure()
        })
    );
}

/// types §4, D93: `eq` makes two function types' `holds` and `calls`
/// variables one, so what one learns later the other learns too.
#[test]
fn eq_links_holds_and_calls_variables() {
    let mut fx = Fixture::new();
    let (holds, calls) = (fx.infer.fresh_holds(), fx.infer.fresh_calls());
    let (other_holds, other_calls) = (fx.infer.fresh_holds(), fx.infer.fresh_calls());
    let pure = || EffectTerm::Known(Effect::pure());
    let f = fx.function_of(pure(), Flows::none(), CallsTerm::Var(calls), HoldsTerm::Var(holds));
    let g = fx.function_of(pure(), Flows::none(), CallsTerm::Var(other_calls), HoldsTerm::Var(other_holds));
    let (a, b) = (fx.held(f), fx.held(g));
    fx.infer.eq(a, b).unwrap();
    fx.infer.join_holds(holds, Holds::Mut).expect("linked, not fixed");
    fx.infer.join_calls(calls, Calls::Once).expect("linked, not fixed");
    let (types, frozen) = fx.freeze(b);
    let TyKind::Fn { calls, holds, .. } = types.kind(frozen) else {
        panic!("a function type")
    };
    assert_eq!((*calls, *holds), (Calls::Once, Holds::Mut));
}

/// types §2, D93: `join` joins `holds` (none, shared, mut) and `calls`
/// (once where either is); a variable joined with a known value keeps
/// joining what is learned later.
#[test]
fn join_joins_holds_and_calls_and_a_variable_keeps_learning() {
    let mut fx = Fixture::new();
    let (holds, calls) = (fx.infer.fresh_holds(), fx.infer.fresh_calls());
    let pure = || EffectTerm::Known(Effect::pure());
    let lambda = fx.function_of(pure(), Flows::none(), CallsTerm::Var(calls), HoldsTerm::Var(holds));
    let slot = fx.function(pure(), Flows::none(), Calls::Many, Holds::Shared);
    let (a, b) = (fx.held(lambda), fx.held(slot));
    fx.infer.join(b, a, Reach::Named).unwrap();
    assert_eq!(fx.infer.holds(HoldsTerm::Var(holds)), Some(Holds::Shared));
    assert_eq!(fx.infer.calls(CallsTerm::Var(calls)), Some(Calls::Many));
    fx.infer.join_holds(holds, Holds::Nothing).expect("joined, not fixed");
    fx.infer.join_calls(calls, Calls::Once).expect("joined, not fixed");
    for side in [a, b] {
        let (types, frozen) = fx.freeze(side);
        let TyKind::Fn { calls, holds, .. } = types.kind(frozen) else {
            panic!("a function type")
        };
        assert_eq!((*calls, *holds), (Calls::Once, Holds::Shared));
    }
    let mutable = fx.function(pure(), Flows::none(), Calls::Many, Holds::Mut);
    let mutable = fx.held(mutable);
    fx.infer.join(a, mutable, Reach::Named).unwrap();
    assert_eq!(fx.infer.holds(HoldsTerm::Var(holds)), Some(Holds::Mut));
}

/// types §5, D93: at a stall, a `holds` nothing decides is none and a
/// `calls` many; a lambda's own variables are left to settling.
#[test]
fn unsettled_holds_and_calls_take_the_defaults_but_a_lambdas() {
    let mut fx = Fixture::new();
    let (holds, calls) = (fx.infer.fresh_holds(), fx.infer.fresh_calls());
    let (lambda_holds, lambda_calls) = (fx.infer.fresh_holds(), fx.infer.fresh_calls());
    let pure = || EffectTerm::Known(Effect::pure());
    let unclaimed = fx.function_of(pure(), Flows::none(), CallsTerm::Var(calls), HoldsTerm::Var(holds));
    let lambda = fx.function_of(pure(), Flows::none(), CallsTerm::Var(lambda_calls), HoldsTerm::Var(lambda_holds));
    assert!(fx.infer.default_unsettled_holds_and_calls(&[(lambda_holds, lambda_calls)]));
    let (types, frozen) = fx.freeze(unclaimed);
    let TyKind::Fn { calls, holds, .. } = types.kind(frozen) else {
        panic!("a function type")
    };
    assert_eq!((*calls, *holds), (Calls::Many, Holds::Nothing));
    assert_eq!(fx.infer.holds(HoldsTerm::Var(lambda_holds)), None);
    assert_eq!(fx.infer.calls(CallsTerm::Var(lambda_calls)), None);
    assert!(fx.infer.freeze(&mut Types::new(), &mut FreezeMemo::default(), lambda).is_err());
    assert!(!fx.infer.default_unsettled_holds_and_calls(&[(lambda_holds, lambda_calls)]));
}

/// effects §1, D95: `flows_into` is one way. What flows into a variable
/// raises it, then and whenever the source rises later; nothing flows back.
/// A variable `eq` fixed that a flow would grow is refused, and keeps its
/// value.
#[test]
fn an_effect_flows_into_another_one_way() {
    let mut fx = Fixture::new();
    let (callee, body) = (EffectTerm::Var(fx.infer.fresh_effect()), EffectTerm::Var(fx.infer.fresh_effect()));
    fx.infer.join_effects(&body, &EffectTerm::Known(Effect::pure())).unwrap();
    fx.infer.flows_into(&callee, &body);
    fx.infer.join_effects(&callee, &EffectTerm::Known(Effect::opaque())).unwrap();
    let settled = |fx: &mut Fixture, effect: EffectTerm<crate::ty::Infer>| {
        let f = fx.function(effect, Flows::none(), Calls::Many, Holds::Nothing);
        let (types, frozen) = fx.freeze(f);
        match types.kind(frozen) {
            TyKind::Fn { effect: EffectTerm::Known(effect), .. } => effect.clone(),
            other => panic!("a function type, not {other:?}"),
        }
    };
    assert_eq!(settled(&mut fx, body.clone()), Effect::opaque(), "the body rose with its callee");

    let (pure_callee, other) = (EffectTerm::Var(fx.infer.fresh_effect()), EffectTerm::Var(fx.infer.fresh_effect()));
    fx.infer.join_effects(&pure_callee, &EffectTerm::Known(Effect::pure())).unwrap();
    fx.infer.flows_into(&pure_callee, &other);
    fx.infer.join_effects(&other, &EffectTerm::Known(Effect::opaque())).unwrap();
    assert_eq!(settled(&mut fx, pure_callee), Effect::pure(), "nothing flows back into the callee");

    let (source, fixed) = (EffectTerm::Var(fx.infer.fresh_effect()), EffectTerm::Var(fx.infer.fresh_effect()));
    let slot = fx.function(EffectTerm::Known(Effect::pure()), Flows::none(), Calls::Many, Holds::Nothing);
    let lambda = fx.function(fixed.clone(), Flows::none(), Calls::Many, Holds::Nothing);
    fx.infer.eq(slot, lambda).unwrap();
    let flow = fx.infer.flows_into(&source, &fixed);
    assert!(fx.infer.take_effects_refused().is_empty());
    fx.infer.join_effects(&source, &EffectTerm::Known(Effect::opaque())).unwrap();
    assert_eq!(
        fx.infer.take_effects_refused(),
        vec![EffectRefused { flow, into: fixed.clone(), exceeds: Exceeds { fixed: Effect::pure(), needs: Effect::opaque() } }]
    );
    assert_eq!(settled(&mut fx, fixed), Effect::pure(), "a fixed effect keeps its value");
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
            head: Head::Base(crate::decl::Base::Int(IntTy::I64)),
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
    let var = fx.infer.fresh(literal());
    let TyKind::Var(id) = *fx.infer.types().kind(var) else { unreachable!() };
    let option = fx.kind(TyKind::Option(var, Label::Here));
    let mut into = Types::new();
    assert_eq!(
        fx.infer.freeze(&mut into, &mut FreezeMemo::default(), option),
        Err(Unsettled::Ty(id))
    );
    let effect = fx.infer.fresh_effect();
    let f = fx.function(EffectTerm::Var(effect), Flows::none(), Calls::Many, Holds::Nothing);
    assert_eq!(
        fx.infer.freeze(&mut into, &mut FreezeMemo::default(), f),
        Err(Unsettled::Effect(effect))
    );
}

/// compilation: freezing follows bound variables, and a term already
/// frozen is answered from the memo.
#[test]
fn freeze_follows_variables_and_answers_a_frozen_term_from_the_memo() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    let object = fx.object(&[("x", int)], Label::Elsewhere);
    let held = fx.held(object);
    let pair = fx.kind(TyKind::Tuple(Box::new([held, object]), Label::Here));
    let mut into = Types::new();
    let mut memo = FreezeMemo::default();
    let frozen = fx.infer.freeze(&mut into, &mut memo, pair).unwrap();
    let TyKind::Tuple(items, _) = into.kind(frozen).clone() else {
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

// Tests that build a `Declarations`: they wait on `Declarations::verify`.

fn pure_fn(types: &mut Types<Poly>, params: Vec<crate::ty::TyId<Poly>>, ret: crate::ty::TyId<Poly>) -> crate::ty::TyId<Poly> {
    types.intern(TyKind::Fn {
        params: params.into(),
        ret,
        effect: EffectTerm::Known(Effect::pure()),
        flows: FlowTerm::Known(Flows::Every),
        calls: Calls::Many,
        holds: Holds::Nothing,
        values: (),
    })
}

fn raw(names: &Interner, types: Types<Poly>, functions: Vec<(QualifiedRef, Scheme)>, casts: Vec<CastRule>) -> RawDeclarations {
    RawDeclarations {
        lang_impls: Vec::new(),
        types,
        functions,
        user_types: Vec::new(),
        casts,
        conversions: Vec::new(),
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
    let option = types.intern(TyKind::Option(t, ()));
    let signature = pure_fn(&mut types, vec![object, t], option);
    let integer = TyVarBound::Integer {
        signed: true,
        among: Box::new([IntTy::I32, IntTy::I64]),
    };
    let scheme = Scheme {
        ty: signature,
        param_names: Box::new([a, b]),
        bounds: Box::new([integer]),
        impls: None,
        requires: Box::new([Requirement {
            generic_function: shared,
            pattern: t,
            fact: None,
        }]),
        facts: DeclaredFacts::default(),
    };
    let decls = Declarations::verify(raw(&fx.names, types, vec![(f, scheme)], Vec::new())).expect("verified");
    let scheme = decls.scheme(f).expect("declared");

    let used = fx.infer.instantiate(&decls, scheme);

    let TyKind::Fn { params, ret, .. } = fx.infer.types().kind(used.ty).clone() else {
        panic!("a function type")
    };
    let var = params[1];
    let TyKind::Var(id) = *fx.infer.types().kind(var) else {
        panic!("a type parameter's place is its variable")
    };
    assert_eq!(
        fx.infer.types().kind(params[0]),
        &TyKind::Object(ObjectTy::Structural(Box::new([(x, var)]), Label::Elsewhere))
    );
    assert_eq!(fx.infer.types().kind(ret), &TyKind::Option(var, Label::Elsewhere));
    assert_eq!(
        fx.infer.unbound().collect::<Vec<_>>(),
        vec![(
            id,
            &Bound::Integer {
                signed: true,
                among: Box::new([IntTy::I32, IntTy::I64]),
                literal: false,
            }
        )]
    );
    assert_eq!(
        used.requirements,
        vec![RequiredAt {
            generic_function: shared,
            pattern: var,
            fact: None
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
    let signature = pure_fn(&mut types, vec![i32_], i64_);
    let scheme = Scheme {
        ty: signature,
        param_names: Box::new([v]),
        bounds: Box::new([]),
        impls: None,
        requires: Box::new([]),
        facts: DeclaredFacts::default(),
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
            head: Head::Base(crate::decl::Base::Int(IntTy::I32)),
            term: int
        })
    );
}

/// types §5: a literal's variable made one with an integer bound takes the
/// widths both admit, and stays a literal only where both are.
#[test]
fn a_literal_made_one_with_an_integer_bound_narrows_and_is_no_longer_a_literal() {
    let mut fx = Fixture::new();
    let a = fx.infer.fresh(literal());
    let b = fx.infer.fresh(Bound::Integer {
        signed: false,
        among: Box::new([IntTy::U8, IntTy::U16]),
        literal: false,
    });
    fx.infer.eq(a, b).unwrap();
    assert_eq!(
        fx.infer.unbound().map(|(_, bound)| bound.clone()).collect::<Vec<_>>(),
        vec![Bound::Integer {
            signed: true,
            among: Box::new([IntTy::U8, IntTy::U16]),
            literal: false,
        }]
    );
    let i8_ = fx.int(IntTy::I8);
    assert!(matches!(fx.infer.eq(a, i8_), Err(Mismatch::Bound { .. })));
}

// The bottom type (types §4)

/// types §4: no type is `!` by holding a part of type `!`. A tuple and an
/// object with a `!` part stay what they are, interned and frozen.
#[test]
fn a_product_with_a_part_of_type_never_is_not_never() {
    let mut fx = Fixture::new();
    let (never, int) = (fx.kind(TyKind::Never), fx.int(IntTy::I64));
    let tuple = fx.kind(TyKind::Tuple(Box::new([int, never]), Label::Here));
    assert_eq!(fx.infer.types().kind(tuple), &TyKind::Tuple(Box::new([int, never]), Label::Here));
    let object = fx.object(&[("x", int), ("y", never)], Label::Here);
    assert!(matches!(fx.infer.types().kind(object), TyKind::Object(_)));

    let var = fx.infer.fresh(Bound::Any);
    let pair = fx.kind(TyKind::Tuple(Box::new([int, var]), Label::Here));
    fx.infer.eq(var, never).unwrap();
    let (types, frozen) = fx.freeze(pair);
    let TyKind::Tuple(items, _) = types.kind(frozen) else {
        panic!("a tuple with a `!` part freezes to a tuple, not {:?}", types.kind(frozen));
    };
    assert_eq!(types.kind(items[1]), &TyKind::Never);
}

/// types §2, §4: `!` is the bottom type: its join with `T` is `T`, and it
/// equals only itself.
#[test]
fn never_joins_to_the_other_side_and_equals_only_itself() {
    let mut fx = Fixture::new();
    let (never, int) = (fx.kind(TyKind::Never), fx.int(IntTy::I64));
    let (a, b) = (fx.held(never), fx.held(int));
    fx.infer.join(a, b, Reach::Named).unwrap();
    assert_eq!(fx.infer.shallow(a), int);
    assert_eq!(fx.infer.join(never, b, Reach::Named), Ok(()));
    assert_eq!(fx.infer.eq(never, int), Err(Mismatch::Heads { a: never, b: int }));
    assert_eq!(fx.infer.eq(never, never), Ok(()));
}

/// types §4-6: a variable joined with `!` is left unbound, on either side,
/// so a later use fixes it as an earlier one would have: the outcome does
/// not depend on the order.
#[test]
fn never_joined_with_a_variable_leaves_it_to_its_other_uses() {
    let mut fx = Fixture::new();
    let (never, int) = (fx.kind(TyKind::Never), fx.int(IntTy::I64));
    for never_first in [true, false] {
        let var = fx.infer.fresh(Bound::Any);
        let (a, b) = match never_first {
            true => (never, var),
            false => (var, never),
        };
        fx.infer.join(a, b, Reach::Named).unwrap();
        assert!(matches!(fx.infer.types().kind(fx.infer.shallow(var)), TyKind::Var(_)), "left unbound");
        fx.infer.eq(var, int).unwrap();
        assert_eq!(fx.infer.shallow(var), int);
    }
    let var = fx.infer.fresh(Bound::Any);
    fx.infer.eq(var, int).unwrap();
    fx.infer.join(var, never, Reach::Named).unwrap();
    assert_eq!(fx.infer.shallow(var), int, "the use first, then `!`, the same");
}

/// types §4, §6: `!` has a head, and no step leaves it, so it is closed.
#[test]
fn never_is_closed() {
    let mut fx = Fixture::new();
    let never = fx.kind(TyKind::Never);
    let held = fx.held(never);
    assert_eq!(
        fx.infer.closed(&CastTree::default(), &Open(true), held),
        Some(Closed {
            head: Head::Base(crate::decl::Base::Never),
            term: never
        })
    );
}

// enter (types §3, §7)

/// Declarations whose casts are `from -> to` through a one-parameter
/// function named `via`, each of its own types.
fn with_casts(
    names: &Interner,
    types: Types<Poly>,
    user_types: Vec<crate::decl::UserTypeDecl>,
    casts: &[(&str, crate::ty::TyId<Poly>, crate::ty::TyId<Poly>)],
) -> Declarations {
    with_casts_and_conversions(names, types, user_types, casts, &[])
}

/// Declarations whose casts and conversions are `from -> to` through a
/// one-parameter function named `via`, each of its own types.
fn with_casts_and_conversions(
    names: &Interner,
    types: Types<Poly>,
    user_types: Vec<crate::decl::UserTypeDecl>,
    casts: &[(&str, crate::ty::TyId<Poly>, crate::ty::TyId<Poly>)],
    conversions: &[(&str, crate::ty::TyId<Poly>, crate::ty::TyId<Poly>)],
) -> Declarations {
    let mut types = types;
    let v = names.intern("v");
    let mut functions = Vec::new();
    let mut declare = |types: &mut Types<Poly>, via: &str, from, to| {
        let via = QualifiedRef::root(names.intern(via));
        let ty = pure_fn(types, vec![from], to);
        functions.push((
            via,
            Scheme {
                ty,
                param_names: Box::new([v]),
                bounds: Box::new([]),
                impls: None,
                requires: Box::new([]),
                facts: DeclaredFacts::default(),
            },
        ));
        via
    };
    let mut cast_rules = Vec::new();
    for (via, from, to) in casts {
        let via = declare(&mut types, via, *from, *to);
        cast_rules.push(CastRule {
            from: *from,
            to: *to,
            via,
        });
    }
    let mut conversion_rules = Vec::new();
    for (via, from, to) in conversions {
        let via = declare(&mut types, via, *from, *to);
        conversion_rules.push(ConversionRule {
            from: *from,
            to: *to,
            via,
        });
    }
    let mut raw = raw(names, types, functions, cast_rules);
    raw.user_types = user_types;
    raw.conversions = conversion_rules;
    Declarations::verify(raw).expect("verified")
}

/// `i8 -> i64` and `i16 -> i64`.
fn widening(names: &Interner) -> Declarations {
    let mut types = Types::<Poly>::new();
    let (i8_, i16_, i64_) = (
        types.intern(TyKind::Int(IntTy::I8)),
        types.intern(TyKind::Int(IntTy::I16)),
        types.intern(TyKind::Int(IntTy::I64)),
    );
    with_casts(names, types, Vec::new(), &[("widen8", i8_, i64_), ("widen16", i16_, i64_)])
}

impl Fixture {
    fn closed_of(&self, decls: &Declarations, term: Term) -> Closed {
        self.infer.closed(decls.casts(), &Open(false), term).expect("closed")
    }

    fn enter(&mut self, decls: &Declarations, value: Term, slot: Term) -> Result<Entered, Mismatch> {
        let closed = self.closed_of(decls, value);
        self.infer.enter(decls, value, closed, slot, Reach::Named)
    }
}

/// types §1: an unbound slot takes the value's type by being linked to the
/// value: the two are one type, so what the slot learns later the value
/// learns too.
#[test]
fn an_unbound_slot_is_linked_to_the_value_it_takes() {
    let mut fx = Fixture::new();
    let decls = widening(&fx.names);
    let int = fx.int(IntTy::I64);
    let x = fx.object(&[("x", int)], Label::Here);
    let value = fx.held(x);
    let slot = fx.infer.fresh(Bound::Any);
    assert_eq!(fx.enter(&decls, value, slot), Ok(Entered::Folded));
    assert_eq!(fx.infer.shallow(slot), x);
    let y = fx.object(&[("y", int)], Label::Here);
    let more = fx.held(y);
    fx.infer.join(slot, more, Reach::Named).unwrap();
    assert_eq!(fx.fields_of(value), (vec![fx.name("x"), fx.name("y")], Label::Here));
}

/// types §1-2: a value and a transparent slot of one head but two shapes
/// meet at their join, and the value is linked to the slot: both are the
/// join, one type.
#[test]
fn a_value_of_the_slots_head_joins_into_it_as_one_type() {
    let mut fx = Fixture::new();
    let decls = widening(&fx.names);
    let int = fx.int(IntTy::I64);
    let (x, y) = (
        fx.object(&[("x", int)], Label::Here),
        fx.object(&[("y", int)], Label::Here),
    );
    let (value, slot) = (fx.held(y), fx.held(x));
    assert_eq!(fx.enter(&decls, value, slot), Ok(Entered::Folded));
    assert_eq!(fx.fields_of(slot), (vec![fx.name("x"), fx.name("y")], Label::Here));
    assert_eq!(fx.fields_of(value), (vec![fx.name("x"), fx.name("y")], Label::Here));
}

/// types §3: a slot folds its values by ⊔, the nearest common ancestor,
/// which is commutative and idempotent, and a value below the slot's head
/// leaves the slot as it is.
#[test]
fn a_slot_folds_its_values_to_their_nearest_common_ancestor() {
    let mut fx = Fixture::new();
    let decls = widening(&fx.names);
    let (i8_, i16_, i64_) = (fx.int(IntTy::I8), fx.int(IntTy::I16), fx.int(IntTy::I64));
    for (first, second) in [(i8_, i16_), (i16_, i8_)] {
        let slot = fx.held(first);
        let value = fx.held(second);
        assert_eq!(fx.enter(&decls, value, slot), Ok(Entered::Folded));
        assert_eq!(fx.infer.shallow(slot), i64_);
        assert_eq!(fx.infer.shallow(value), second, "the value keeps its type");
    }
    let slot = fx.held(i8_);
    let value = fx.held(i8_);
    assert_eq!(fx.enter(&decls, value, slot), Ok(Entered::Folded));
    assert_eq!(fx.infer.shallow(slot), i8_);
    let slot = fx.held(i64_);
    assert_eq!(fx.enter(&decls, value, slot), Ok(Entered::Folded));
    assert_eq!(fx.infer.shallow(slot), i64_);
}

/// types §3, §7: where no subtyping relates a value and its slot, and no
/// declared conversion takes one to the other, they do not meet, and the
/// refusal names no conversion.
#[test]
fn a_value_and_a_slot_of_no_common_ancestor_do_not_meet() {
    let mut fx = Fixture::new();
    let decls = widening(&fx.names);
    let (i8_, string) = (fx.int(IntTy::I8), fx.kind(TyKind::String));
    let (value, slot) = (fx.held(string), fx.held(i8_));
    assert_eq!(
        fx.enter(&decls, value, slot),
        Err(Mismatch::Conversions {
            candidates: Box::new([])
        })
    );
}

/// `i8 -> i64` and `i16 -> i64` as casts, and as conversions `String ->
/// i64` through `parse`, `String -> i16` through `parse16`, and, where
/// `twice`, `String -> i64` through `parse_again`.
fn parsing(names: &Interner, twice: bool) -> Declarations {
    let mut types = Types::<Poly>::new();
    let (i8_, i16_, i64_, string) = (
        types.intern(TyKind::Int(IntTy::I8)),
        types.intern(TyKind::Int(IntTy::I16)),
        types.intern(TyKind::Int(IntTy::I64)),
        types.intern(TyKind::String),
    );
    let mut conversions = vec![("parse", string, i64_), ("parse16", string, i16_)];
    if twice {
        conversions.push(("parse_again", string, i64_));
    }
    with_casts_and_conversions(
        names,
        types,
        Vec::new(),
        &[("widen8", i8_, i64_), ("widen16", i16_, i64_)],
        &conversions,
    )
}

/// types §7: where no subtyping relates a value and its slot, the one
/// declared conversion from the value's head to the slot's applies, and
/// the slot takes the type it converts to.
#[test]
fn a_value_converts_by_the_one_conversion_to_its_slots_head() {
    let mut fx = Fixture::new();
    let decls = parsing(&fx.names, false);
    let (i64_, string) = (fx.int(IntTy::I64), fx.kind(TyKind::String));
    let (value, slot) = (fx.held(string), fx.held(i64_));
    let parse = QualifiedRef::root(fx.name("parse"));
    let Ok(Entered::Converted(conversion)) = fx.enter(&decls, value, slot) else {
        panic!("String enters an i64 slot by parse")
    };
    assert_eq!(conversion.rule().via, parse);
    assert_eq!(fx.infer.shallow(slot), i64_);
    assert_eq!(fx.infer.shallow(value), string, "the value keeps its type");
}

/// types §7: where several declared conversions apply, the value does not
/// enter, and the refusal names every one that left it undecided.
#[test]
fn several_conversions_to_the_slots_head_are_refused_by_name() {
    let mut fx = Fixture::new();
    let decls = parsing(&fx.names, true);
    let (i64_, string) = (fx.int(IntTy::I64), fx.kind(TyKind::String));
    let (value, slot) = (fx.held(string), fx.held(i64_));
    let (parse, parse_again) = (
        QualifiedRef::root(fx.name("parse")),
        QualifiedRef::root(fx.name("parse_again")),
    );
    assert_eq!(
        fx.enter(&decls, value, slot),
        Err(Mismatch::Conversions {
            candidates: Box::new([parse, parse_again])
        })
    );
}

/// types §3: a slot no variable holds cannot move up the tree.
#[test]
fn a_slot_no_variable_holds_does_not_move() {
    let mut fx = Fixture::new();
    let decls = widening(&fx.names);
    let (i8_, i16_, i64_) = (fx.int(IntTy::I8), fx.int(IntTy::I16), fx.int(IntTy::I64));
    let value = fx.held(i16_);
    assert!(fx.enter(&decls, value, i8_).is_err());
    assert_eq!(fx.enter(&decls, value, i64_), Ok(Entered::Folded));
}

/// types §3, §4: `!` enters any slot and leaves it as it is, an unbound
/// one unbound; a slot of type `!` takes the value's type.
#[test]
fn never_enters_any_slot() {
    let mut fx = Fixture::new();
    let decls = widening(&fx.names);
    let (never, i8_) = (fx.kind(TyKind::Never), fx.int(IntTy::I8));
    let (value, slot) = (fx.held(never), fx.held(i8_));
    assert_eq!(fx.enter(&decls, value, slot), Ok(Entered::Folded));
    assert_eq!(fx.infer.shallow(slot), i8_);
    let (value, slot) = (fx.held(never), fx.infer.fresh(Bound::Any));
    assert_eq!(fx.enter(&decls, value, slot), Ok(Entered::Folded));
    assert!(matches!(fx.infer.types().kind(fx.infer.shallow(slot)), TyKind::Var(_)));
    let (value, slot) = (fx.held(i8_), fx.held(never));
    assert_eq!(fx.enter(&decls, value, slot), Ok(Entered::Folded));
    assert_eq!(fx.infer.shallow(slot), i8_);
}

/// Extern types `Token` and `Span`, each with one identity parameter, and
/// the casts `Token<'i> -> Span<'i>` and, where `to_string`, `Span<'i> ->
/// String`.
fn identities(names: &Interner, to_string: bool) -> Declarations {
    let mut types = Types::<Poly>::new();
    let (token, span) = (
        QualifiedRef::root(names.intern("Token")),
        QualifiedRef::root(names.intern("Span")),
    );
    let with_identity = |types: &mut Types<Poly>, id| {
        types.intern(TyKind::UserDefined {
            id,
            type_args: Box::new([]),
            effect_args: Box::new([]),
            identity_args: Box::new([IdentityTerm::Var(0)]),
        })
    };
    let (token_ty, span_ty) = (with_identity(&mut types, token), with_identity(&mut types, span));
    let string = types.intern(TyKind::String);
    let declared = |id| crate::decl::UserTypeDecl {
        id,
        type_params: Box::new([]),
        specializes: Box::new([]),
        effect_params: 0,
        identity_params: 1,
        lifetimes: 0,
    };
    let mut casts = vec![("token_span", token_ty, span_ty)];
    if to_string {
        casts.push(("span_text", span_ty, string));
    }
    with_casts(names, types, vec![declared(token), declared(span)], &casts)
}

impl Fixture {
    fn token(&mut self, identity: usize) -> Term {
        let token = QualifiedRef::root(self.name("Token"));
        self.kind(TyKind::UserDefined {
            id: token,
            type_args: Box::new([]),
            effect_args: Box::new([]),
            identity_args: Box::new([IdentityTerm::Known(IdentityId::from_raw(identity))]),
        })
    }
}

/// types §3: values of one identity keep it.
#[test]
fn values_of_one_identity_keep_it() {
    let mut fx = Fixture::new();
    let decls = identities(&fx.names, true);
    let token = fx.token(0);
    let (value, slot) = (fx.held(token), fx.held(token));
    assert_eq!(fx.enter(&decls, value, slot), Ok(Entered::Folded));
    assert_eq!(fx.infer.shallow(slot), token);
}

/// types §3: values of two identities meet at the nearest common ancestor
/// that carries no identity; an identity-carrying ancestor is passed.
#[test]
fn two_identities_meet_past_every_identity_carrying_ancestor() {
    let mut fx = Fixture::new();
    let decls = identities(&fx.names, true);
    let (first, second) = (fx.token(0), fx.token(1));
    let (value, slot) = (fx.held(second), fx.held(first));
    assert_eq!(fx.enter(&decls, value, slot), Ok(Entered::Folded));
    assert_eq!(fx.infer.types().kind(fx.infer.shallow(slot)), &TyKind::String);
}

/// types §3, §7: two identities with no identity-free common ancestor do
/// not meet, and no conversion takes one to the other; no identity is made
/// to join two others.
#[test]
fn two_identities_with_no_identity_free_ancestor_do_not_meet() {
    let mut fx = Fixture::new();
    let decls = identities(&fx.names, false);
    let (first, second) = (fx.token(0), fx.token(1));
    let (value, slot) = (fx.held(second), fx.held(first));
    assert_eq!(
        fx.enter(&decls, value, slot),
        Err(Mismatch::Conversions {
            candidates: Box::new([])
        })
    );
}

/// types §1: a declared type that is not a function's, such as a
/// context's, is built opaque at its use.
#[test]
fn a_declared_type_is_instantiated_opaque() {
    let mut fx = Fixture::new();
    let mut types = Types::new();
    let field = fx.name("age");
    let int = types.intern(TyKind::Int(IntTy::I64));
    let object = types.intern(TyKind::Object(ObjectTy::Structural(Box::new([(field, int)]), ())));
    let decls = Declarations::verify(raw(&fx.names, types, Vec::new(), Vec::new())).expect("verified");
    let term = fx.infer.instantiate_type(&decls, object);
    assert_eq!(fx.fields_of(term), (vec![field], Label::Elsewhere));
}

/// types §5: an effect nothing constrains is opaque, and an argument whose
/// `#` nothing decides is not specialized.
#[test]
fn open_effects_and_marks_take_their_defaults() {
    let mut fx = Fixture::new();
    let effect = EffectTerm::Var(fx.infer.fresh_effect());
    let function = fx.function(effect, Flows::none(), Calls::Many, Holds::Nothing);
    let mark = fx.infer.fresh_repr();
    let unit = fx.kind(TyKind::Unit);
    let open = fx.kind(TyKind::Ref(Mutability::Shared, TypeArg::Open(mark, unit)));
    assert!(fx.infer.default_effects_and_marks());
    assert!(!fx.infer.default_effects_and_marks(), "each default applies once");
    let (types, frozen) = fx.freeze(function);
    let TyKind::Fn {
        effect: EffectTerm::Known(settled),
        ..
    } = types.kind(frozen)
    else {
        panic!("a function of a known effect");
    };
    assert_eq!(settled, &Effect::opaque());
    let (types, frozen) = fx.freeze(open);
    assert!(matches!(types.kind(frozen), TyKind::Ref(_, TypeArg::Uniform(_))));
}
