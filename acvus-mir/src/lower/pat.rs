//! Patterns (patterns §1). A `match` whose arms all test one enum's tags or
//! one kind of literal is one `Switch` dispatch, with a default where it has
//! a catch-all, and its join held by the `Switch`; any other `match` tests
//! its arms one at a time, in order, within the same region and to the same
//! join (D57). Exhaustiveness is `check`'s.
//!
//! A pattern is tested, then bound. A value scrutinee that is not a
//! reference is put in a slot first, as Rust's MIR matches on a place
//! (D89): the test reads its parts through references and moves nothing,
//! built so that a part is read only where the tests before it hold, and
//! binding takes each bound part once, where the whole test held. Through a
//! reference, as Rust's default binding modes do, the parts are places
//! under it and a name binds a reference to its part (patterns §1).

use acvus_ast::{AstId, Expr, Literal, MatchExprArm, Pattern, Slot, Span, TuplePatternElem};
use acvus_utils::Astr;

use super::call::convert;
use super::context;
use super::control::diamond;
use super::ctxt::{Home, Lcx};
use super::expr::lower_expr;
use super::place::{Operand, Place, assign, lend, read};
use super::stmt::lower_body;
use crate::ir::cfg::{BlockId, Op, Terminator};
use crate::ir::{BinOp, IndexBound, IndexMode, PathSeg, RefTarget, SwitchKey, ValueId, known_len};
use crate::ty::{Mutability, Ty, TyKind};

/// What a pattern is matched against: a reference value; a place the match
/// holds, its scrutinee's slot or a part of it, with the slot's type; or,
/// past a reference, a place under it, lent at `lent`.
#[derive(Debug, Clone)]
pub(super) enum Matched {
    Value { value: ValueId, ty: Ty },
    Held { place: Place, slot_ty: Ty },
    Place { place: Place, lent: Mutability },
}

/// A part a pattern names of what it matches.
#[derive(Debug, Clone, Copy)]
enum Part {
    Tuple(usize),
    Field(Astr),
    Payload(Astr),
    /// An array's element, at an index its type fixes.
    Element(usize),
}

/// Where an element a list pattern names stands.
#[derive(Debug, Clone, Copy)]
enum ListAt {
    Array(usize),
    /// A slice's element, counted from its front.
    SliceHead(usize),
    /// A slice's element, `back` from its end.
    SliceTail { back: usize },
}

impl Matched {
    /// The scrutinee `value`: a reference as it is, since tests read
    /// through it and never move it; any other value in a slot of its own,
    /// assigned at `span`, so that no test moves a part of it (D89).
    pub fn scrutinee(cx: &mut Lcx<'_, '_>, value: ValueId, span: Span) -> Matched {
        let ty = cx.value_ty(value);
        if matches!(cx.kind(ty), TyKind::Ref(..)) {
            return Matched::Value { value, ty };
        }
        let slot = cx.builder.slot(ty, span);
        let place = Place { target: RefTarget::Var(slot), path: Vec::new(), ty };
        assign(cx, place.clone(), value, span);
        Matched::Held { place, slot_ty: ty }
    }

    fn ty(&self) -> Ty {
        match self {
            Matched::Value { ty, .. } => *ty,
            Matched::Held { place, .. } | Matched::Place { place, .. } => place.ty,
        }
    }
}

/// `matched` past a reference, where it is one: a pattern that is not a
/// name destructures what the reference names (patterns §1). A shared
/// reference on the way keeps every name below it shared.
fn through(cx: &mut Lcx<'_, '_>, matched: &Matched, span: Span) -> Matched {
    let TyKind::Ref(held, target) = cx.kind(matched.ty()) else {
        return matched.clone();
    };
    let (held, target) = (*held, *target);
    let (reference, lent) = match matched {
        Matched::Value { value, .. } => (*value, held),
        Matched::Held { place, slot_ty } => (held_reference(cx, place, *slot_ty, held, span), held),
        Matched::Place { place, lent } => {
            let reference = read(cx, Operand::Place(place.clone()), span);
            let lent = lent.least(held);
            (super::place::reborrow(cx, reference, held, target, lent, span), lent)
        }
    };
    Matched::Place { place: Place { target: RefTarget::Through(reference), path: Vec::new(), ty: target }, lent }
}

/// The reference a held part is, read without moving it out of the slot. A
/// `&` copies, so it is read as it is. A `&mut` does not copy, and each use
/// reborrows it (ownership §5): it is read through a `&mut` to the slot, so
/// the slot keeps it and the drop plan sees no part of the slot move.
fn held_reference(cx: &mut Lcx<'_, '_>, place: &Place, slot_ty: Ty, held: Mutability, span: Span) -> ValueId {
    match held {
        Mutability::Shared => read(cx, Operand::Place(place.clone()), span),
        Mutability::Mut => {
            let slot = Place { target: place.target, path: Vec::new(), ty: slot_ty };
            let lent = lend(cx, Operand::Place(slot), Mutability::Mut, span);
            let op = Op::Take { target: RefTarget::Through(lent), path: place.path.clone(), taken_out: false };
            cx.builder.push(op, place.ty, span)
        }
    }
}

fn part(cx: &mut Lcx<'_, '_>, matched: &Matched, part: Part, span: Span) -> Matched {
    let ty = part_ty(cx, matched.ty(), part);
    match matched {
        Matched::Value { value, .. } => {
            let op = match part {
                Part::Tuple(index) => Op::TupleIndex { tuple: *value, index },
                Part::Field(field) => Op::FieldGet { object: *value, field, rest: Vec::new() },
                Part::Payload(_) => Op::UnwrapVariant { src: *value },
                Part::Element(index) => Op::ArrayIndex { array: *value, index },
            };
            Matched::Value { value: cx.builder.push(op, ty, span), ty }
        }
        Matched::Held { place, slot_ty } => Matched::Held { place: place.with(segment(part), ty), slot_ty: *slot_ty },
        Matched::Place { place, lent } => Matched::Place { place: place.with(segment(part), ty), lent: *lent },
    }
}

fn segment(part: Part) -> PathSeg {
    match part {
        Part::Tuple(index) | Part::Element(index) => PathSeg::Index(index),
        Part::Field(field) => PathSeg::Field(field),
        Part::Payload(_) => PathSeg::Payload,
    }
}

/// The element a list pattern names at `at`: an array's is a part; a
/// slice's is reached by index, past its length test.
fn element(cx: &mut Lcx<'_, '_>, matched: &Matched, at: ListAt, span: Span) -> Matched {
    let (place, lent) = match (at, matched) {
        (ListAt::Array(index), _) => return part(cx, matched, Part::Element(index), span),
        (ListAt::SliceHead(_) | ListAt::SliceTail { .. }, Matched::Place { place, lent }) => (place, *lent),
        (ListAt::SliceHead(_) | ListAt::SliceTail { .. }, Matched::Value { .. } | Matched::Held { .. }) => {
            panic!("a slice is matched only past the reference that lends it")
        }
    };
    let TyKind::Slice(ty) = cx.kind(place.ty) else {
        panic!("a slice's element is named of a slice")
    };
    let ty = *ty;
    let slice = slice_ref(cx, place, lent, span);
    let u64_ty = cx.u64_ty();
    let index = match at {
        ListAt::SliceTail { back } => {
            let len = cx.builder.push(Op::SliceLen { src: slice }, u64_ty, span);
            let back = cx.constant(Literal::Int(back as i128), u64_ty, span);
            let op = Op::BinOp { op: BinOp::Sub, left: len, right: back };
            cx.builder.push(op, u64_ty, span)
        }
        ListAt::SliceHead(front) | ListAt::Array(front) => cx.constant(Literal::Int(front as i128), u64_ty, span),
    };
    let element_ref = cx.ref_ty(lent, ty);
    let op = Op::Index { slice, index, mode: IndexMode::Ref, bound: IndexBound::Checked };
    let reference = cx.builder.push(op, element_ref, span);
    Matched::Place { place: Place { target: RefTarget::Through(reference), path: Vec::new(), ty }, lent }
}

/// A reference, at `lent`, to the slice `place` holds.
fn slice_ref(cx: &mut Lcx<'_, '_>, place: &Place, lent: Mutability, span: Span) -> ValueId {
    let ty = cx.ref_ty(lent, place.ty);
    let op = Op::Ref { target: place.target, path: place.path.clone(), mutability: lent };
    cx.builder.push_place(op, ty, span)
}

/// The type of `part` of a `whole`.
///
/// # Panics
/// `whole` has no such part: typeck admits a pattern only on a type that
/// has what it names.
fn part_ty(cx: &Lcx<'_, '_>, whole: Ty, part: Part) -> Ty {
    let tags = cx.decls.core().tags;
    let found = match (cx.kind(whole), part) {
        (TyKind::Tuple(items, _), Part::Tuple(index)) => items.get(index).copied(),
        (TyKind::Object { fields, .. }, Part::Field(field)) => {
            fields.iter().find(|(name, _)| *name == field).map(|(_, ty)| *ty)
        }
        (TyKind::Option(payload, _), Part::Payload(_)) => Some(*payload),
        (TyKind::Result(ok, err, _), Part::Payload(tag)) => Some(if tag == tags.ok { *ok } else { *err }),
        (TyKind::Enum { variants, .. }, Part::Payload(tag)) => {
            variants.iter().find(|(name, _)| *name == tag).and_then(|(_, payload)| *payload)
        }
        (TyKind::Array(element, _, _), Part::Element(_)) => Some(*element),
        _ => None,
    };
    found.unwrap_or_else(|| panic!("typeck admits {part:?} only of a type that has it, not {:?}", cx.kind(whole)))
}

/// Whether a pattern may fail on a value of its type. A list's length is
/// tested by `test` itself, so a list counts as refutable here.
fn refutable<S: Slot>(pattern: &Pattern<S>) -> bool {
    match pattern {
        Pattern::Binding { .. } | Pattern::Wildcard { .. } | Pattern::ContextBind { .. } => false,
        Pattern::Tuple { elements, .. } => elements.iter().any(|element| match element {
            TuplePatternElem::Pattern(part) => refutable(part),
            TuplePatternElem::Wildcard(_) => false,
        }),
        Pattern::Object { fields, .. } => fields.iter().any(|field| refutable(&field.pattern)),
        Pattern::Literal { .. } | Pattern::Variant { .. } | Pattern::List { .. } | Pattern::Error(_) => true,
    }
}

/// `held && next`, where `next` is taken only where `held` holds.
fn and(
    cx: &mut Lcx<'_, '_>,
    held: Option<ValueId>,
    span: Span,
    next: impl FnOnce(&mut Lcx<'_, '_>) -> Option<ValueId>,
) -> Option<ValueId> {
    let Some(held) = held else {
        return next(cx);
    };
    let bool_ty = cx.bool_ty();
    let joined = diamond(
        cx,
        held,
        bool_ty,
        span,
        |cx| match next(cx) {
            Some(holds) => holds,
            None => cx.bool_const(true, span),
        },
        |cx| cx.bool_const(false, span),
    );
    Some(joined)
}

/// Whether `pattern` matches: `None` where it cannot fail.
pub(super) fn test<S: Slot>(cx: &mut Lcx<'_, '_>, pattern: &Pattern<S>, matched: &Matched) -> Option<ValueId> {
    acvus_utils::grow(|| test_level(cx, pattern, matched))
}

fn test_level<S: Slot>(cx: &mut Lcx<'_, '_>, pattern: &Pattern<S>, matched: &Matched) -> Option<ValueId> {
    let span = pattern.span();
    match pattern {
        Pattern::Binding { .. } | Pattern::Wildcard { .. } | Pattern::ContextBind { .. } => None,
        Pattern::Literal { value, .. } => {
            let matched = through(cx, matched, span);
            let src = tag_source(cx, &matched, span);
            let bool_ty = cx.bool_ty();
            Some(cx.builder.push(Op::TestLiteral { src, value: value.clone() }, bool_ty, span))
        }
        Pattern::Variant { tag, payload, .. } => {
            let matched = through(cx, matched, span);
            let src = tag_source(cx, &matched, span);
            let bool_ty = cx.bool_ty();
            let tagged = cx.builder.push(Op::TestVariant { src, tag: *tag }, bool_ty, span);
            match payload {
                Some(payload) if refutable(payload) => and(cx, Some(tagged), span, |cx| {
                    let inner = part(cx, &matched, Part::Payload(*tag), span);
                    test(cx, payload, &inner)
                }),
                _ => Some(tagged),
            }
        }
        Pattern::Tuple { elements, .. } => {
            let matched = through(cx, matched, span);
            let mut held = None;
            for (index, element) in elements.iter().enumerate() {
                let TuplePatternElem::Pattern(element) = element else {
                    continue;
                };
                if !refutable(element) {
                    continue;
                }
                held = and(cx, held, span, |cx| {
                    let inner = part(cx, &matched, Part::Tuple(index), span);
                    test(cx, element, &inner)
                });
            }
            held
        }
        Pattern::Object { fields, .. } => {
            let matched = through(cx, matched, span);
            let mut held = None;
            for field in fields.iter().filter(|field| refutable(&field.pattern)) {
                held = and(cx, held, span, |cx| {
                    let inner = part(cx, &matched, Part::Field(field.key), span);
                    test(cx, &field.pattern, &inner)
                });
            }
            held
        }
        Pattern::List { head, rest, tail, .. } => {
            let matched = through(cx, matched, span);
            let mut held = slice_length_test(cx, &matched, head.len() + tail.len(), rest.is_some(), span);
            for (at, part_pattern) in list_elements(cx, &matched, head, tail) {
                if !refutable(part_pattern) {
                    continue;
                }
                held = and(cx, held, span, |cx| {
                    let inner = element(cx, &matched, at, span);
                    test(cx, part_pattern, &inner)
                });
            }
            held
        }
        Pattern::Error(_) => panic!("a pattern the parse left in error is refused, and never lowered"),
    }
}

/// A list pattern's length test: none on an array, whose type fixes its
/// length; on a slice, its length equal to the named elements', or at
/// least that where the list has a `..`.
fn slice_length_test(cx: &mut Lcx<'_, '_>, matched: &Matched, named: usize, rest: bool, span: Span) -> Option<ValueId> {
    let Matched::Place { place, lent } = matched else {
        return None;
    };
    if !matches!(cx.kind(place.ty), TyKind::Slice(_)) {
        return None;
    }
    let slice = slice_ref(cx, place, *lent, span);
    let u64_ty = cx.u64_ty();
    let len = cx.builder.push(Op::SliceLen { src: slice }, u64_ty, span);
    let named = cx.constant(Literal::Int(named as i128), u64_ty, span);
    let op = match rest {
        true => BinOp::Gte,
        false => BinOp::Eq,
    };
    let bool_ty = cx.bool_ty();
    Some(cx.builder.push(Op::BinOp { op, left: len, right: named }, bool_ty, span))
}

/// Each element a list pattern names, with where it stands.
fn list_elements<'p, S: Slot>(
    cx: &Lcx<'_, '_>,
    matched: &Matched,
    head: &'p [Pattern<S>],
    tail: &'p [Pattern<S>],
) -> Vec<(ListAt, &'p Pattern<S>)> {
    let heads = head.iter().enumerate();
    match cx.kind(matched.ty()) {
        TyKind::Array(_, len, _) => {
            let from_end = known_len(len) - tail.len();
            let tails = tail.iter().enumerate().map(|(at, part)| (ListAt::Array(from_end + at), part));
            heads.map(|(at, part)| (ListAt::Array(at), part)).chain(tails).collect()
        }
        TyKind::Slice(_) => {
            let tails = tail.iter().enumerate().map(|(at, part)| (ListAt::SliceTail { back: tail.len() - at }, part));
            heads.map(|(at, part)| (ListAt::SliceHead(at), part)).chain(tails).collect()
        }
        other => panic!("typeck admits a list pattern only on an array or a slice, not {other:?}"),
    }
}

/// What a test reads: the value, or a shared reference to the place, read
/// through as a `Switch` reads its tag through one; a test takes nothing
/// out (D89).
pub(super) fn tag_source(cx: &mut Lcx<'_, '_>, matched: &Matched, span: Span) -> ValueId {
    match matched {
        Matched::Value { value, .. } => *value,
        Matched::Held { place, .. } | Matched::Place { place, .. } => {
            let ty = cx.ref_ty(Mutability::Shared, place.ty);
            let op = Op::Ref { target: place.target, path: place.path.clone(), mutability: Mutability::Shared };
            cx.builder.push_place(op, ty, span)
        }
    }
}

/// Binds `pattern`'s names against `matched`, where its test held.
pub(super) fn bind<S: Slot>(cx: &mut Lcx<'_, '_>, pattern: &Pattern<S>, matched: &Matched) {
    acvus_utils::grow(|| bind_level(cx, pattern, matched))
}

fn bind_level<S: Slot>(cx: &mut Lcx<'_, '_>, pattern: &Pattern<S>, matched: &Matched) {
    let span = pattern.span();
    match pattern {
        Pattern::Binding { id, .. } => {
            let ty = cx.ty(*id);
            let value = bound(cx, matched, ty, span);
            let slot = cx.builder.slot(ty, span);
            assign(cx, Place { target: RefTarget::Var(slot), path: Vec::new(), ty }, value, span);
            cx.bind(*id, Home::Slot(slot), span);
        }
        Pattern::ContextBind { id, name, .. } => {
            let value = match matched {
                Matched::Value { value, .. } => *value,
                Matched::Held { place, .. } => read(cx, Operand::Place(place.clone()), span),
                Matched::Place { .. } => {
                    panic!("a name bound past a reference is a reference, which typeck refuses to store into a context")
                }
            };
            let value = convert(cx, *id, value, span);
            context::store(cx, *name, value, span);
        }
        Pattern::Wildcard { .. } | Pattern::Literal { .. } => {}
        Pattern::Variant { tag, payload, .. } => {
            if let Some(payload) = payload {
                let matched = through(cx, matched, span);
                let inner = part(cx, &matched, Part::Payload(*tag), span);
                bind(cx, payload, &inner);
            }
        }
        Pattern::Tuple { elements, .. } => {
            let matched = through(cx, matched, span);
            for (index, element) in elements.iter().enumerate() {
                if let TuplePatternElem::Pattern(element) = element {
                    let inner = part(cx, &matched, Part::Tuple(index), span);
                    bind(cx, element, &inner);
                }
            }
        }
        Pattern::Object { fields, .. } => {
            let matched = through(cx, matched, span);
            for field in fields {
                let inner = part(cx, &matched, Part::Field(field.key), span);
                bind(cx, &field.pattern, &inner);
            }
        }
        Pattern::List { head, tail, .. } => {
            let matched = through(cx, matched, span);
            for (at, part_pattern) in list_elements(cx, &matched, head, tail) {
                let inner = element(cx, &matched, at, span);
                bind(cx, part_pattern, &inner);
            }
        }
        Pattern::Error(_) => panic!("a pattern the parse left in error is refused, and never lowered"),
    }
}

/// What a name binds: the part itself, or past a reference, a reference to
/// it at the binder's mutability (patterns §1).
fn bound(cx: &mut Lcx<'_, '_>, matched: &Matched, ty: Ty, span: Span) -> ValueId {
    match matched {
        Matched::Value { value, ty: held } => {
            assert_eq!(*held, ty, "a name bound to a value has the value's type");
            *value
        }
        Matched::Held { place, .. } => {
            assert_eq!(place.ty, ty, "a name bound to a held part has the part's type");
            read(cx, Operand::Place(place.clone()), span)
        }
        Matched::Place { place, .. } => {
            let TyKind::Ref(mutability, _) = cx.kind(ty) else {
                panic!("a name bound past a reference binds a reference (patterns §1)")
            };
            let mutability = *mutability;
            lend(cx, Operand::Place(place.clone()), mutability, span)
        }
    }
}

/// A `Switch`'s plan: each keyed arm, by index, and the catch-all's.
struct SwitchPlan {
    keyed: Vec<(SwitchKey, usize)>,
    default: Option<usize>,
}

/// The one-dispatch plan of a `match` whose arms each test one tag of one
/// enum, or one literal of one kind, the first catch-all being the
/// default; `None` for any other.
fn switch_plan<S: Slot>(cx: &Lcx<'_, '_>, scrutinee: Ty, arms: &[MatchExprArm<S>]) -> Option<SwitchPlan> {
    let tested = match cx.kind(scrutinee) {
        TyKind::Ref(_, target) => *target,
        _ => scrutinee,
    };
    let tagged = matches!(cx.kind(tested), TyKind::Option(.., _) | TyKind::Result(..) | TyKind::Enum { .. });
    let mut keyed: Vec<(SwitchKey, usize)> = Vec::new();
    let mut default = None;
    for (at, arm) in arms.iter().enumerate() {
        let key = match &arm.pattern {
            Pattern::Wildcard { .. } | Pattern::Binding { .. } => {
                default = Some(at);
                break;
            }
            Pattern::Variant { tag, payload, .. } if tagged && payload.as_deref().is_none_or(|payload| !refutable(payload)) => {
                SwitchKey::Tag(*tag)
            }
            Pattern::Literal { value, .. } => literal_key(value)?,
            _ => return None,
        };
        let repeated = keyed.iter().any(|(earlier, _)| *earlier == key);
        let other_kind = keyed.first().is_some_and(|(first, _)| !first.same_kind(key));
        if repeated || other_kind {
            return None;
        }
        keyed.push((key, at));
    }
    match keyed.is_empty() {
        true => None,
        false => Some(SwitchPlan { keyed, default }),
    }
}

/// The key of a literal a `Switch` can dispatch on. A string's key would
/// name interned text, and lowering holds no interner (D60), so a string
/// literal is tested arm by arm.
fn literal_key(literal: &Literal) -> Option<SwitchKey> {
    match literal {
        Literal::Int(value) => Some(SwitchKey::Int(*value)),
        Literal::IntOf(suffixed) => Some(SwitchKey::Int(suffixed.value)),
        Literal::Bool(value) => Some(SwitchKey::Bool(*value)),
        Literal::Char(value) => Some(SwitchKey::Char(*value)),
        Literal::String(_) | Literal::Float(_) | Literal::Bytes(_) | Literal::List(_) | Literal::Unit => None,
    }
}

pub(super) fn lower_match<S: Slot>(
    cx: &mut Lcx<'_, '_>,
    site: AstId,
    scrutinee: &Expr<S>,
    arms: &[MatchExprArm<S>],
    span: Span,
) -> ValueId {
    let value = lower_expr(cx, scrutinee);
    let matched = Matched::scrutinee(cx, value, scrutinee.span());
    let ty = cx.ty(site);
    let live = cx.reachable;
    let join = cx.builder.new_block(&[ty]);
    let mut joined = false;
    match switch_plan(cx, matched.ty(), arms) {
        Some(plan) => {
            let keyed: Vec<(SwitchKey, BlockId, usize)> =
                plan.keyed.iter().map(|(key, at)| (*key, cx.builder.new_block(&[]), *at)).collect();
            let default = plan.default.map(|at| (cx.builder.new_block(&[]), at));
            // No catch-all: a value no key names goes to the block where no
            // arm answers, which matchck judges (patterns §3).
            let unmatched = default.is_none().then(|| cx.builder.new_block(&[]));
            let tag = tag_source(cx, &matched, span);
            let terminator = Terminator::Switch {
                tag,
                arms: keyed.iter().map(|(key, block, _)| (*key, *block, Vec::new())).collect(),
                default: default.map(|(block, _)| block).or(unmatched).map(|block| (block, Vec::new())),
                join,
            };
            cx.builder.terminate(terminator, span);
            if let Some(unmatched) = unmatched {
                cx.builder.switch_to(unmatched);
                cx.builder.terminate(Terminator::NoArmMatched { span }, span);
            }
            let entered = keyed.iter().map(|(_, block, at)| (*block, *at));
            for (block, at) in entered.chain(default) {
                cx.builder.switch_to(block);
                cx.reachable = live;
                joined |= arm_body(cx, &arms[at], &matched, join);
            }
        }
        None => {
            let mut caught = false;
            for arm in arms {
                match test(cx, &arm.pattern, &matched) {
                    None => {
                        joined |= arm_body(cx, arm, &matched, join);
                        caught = true;
                        break;
                    }
                    Some(holds) => {
                        let then_to = cx.builder.new_block(&[]);
                        let else_to = cx.builder.new_block(&[]);
                        let terminator = Terminator::Diamond {
                            cond: holds,
                            then_to,
                            then_args: Vec::new(),
                            else_to,
                            else_args: Vec::new(),
                            join,
                        };
                        cx.builder.terminate(terminator, arm.span);
                        cx.builder.switch_to(then_to);
                        cx.reachable = live;
                        joined |= arm_body(cx, arm, &matched, join);
                        cx.builder.switch_to(else_to);
                        cx.reachable = live;
                    }
                }
            }
            if !caught {
                cx.builder.terminate(Terminator::NoArmMatched { span }, span);
            }
        }
    }
    cx.builder.switch_to(join);
    cx.reachable = joined;
    cx.builder.block_params(join)[0]
}

/// One arm: its names bound, its body lowered, its value entering the
/// match's one slot at its tail's site or the arm's; whether it reaches
/// the join.
fn arm_body<S: Slot>(cx: &mut Lcx<'_, '_>, arm: &MatchExprArm<S>, matched: &Matched, join: BlockId) -> bool {
    bind(cx, &arm.pattern, matched);
    let value = lower_body(cx, &arm.body, arm.tail.as_deref(), arm.span);
    let at = arm.tail.as_deref().map_or(arm.id, Expr::id);
    let value = convert(cx, at, value, arm.span);
    let reached = cx.reachable;
    cx.jump(join, vec![value], arm.span);
    reached
}
