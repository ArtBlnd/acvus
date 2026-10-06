//! Patterns (patterns §1). A `match` whose arms all test one enum's tags or
//! one kind of literal is one `Switch` dispatch, with a default where it has
//! a catch-all, and its join held by the `Switch`; any other `match` tests
//! its arms one at a time, in order, within the same region and to the same
//! join. Exhaustiveness is `check`'s.
//!
//! A pattern is tested, then bound. A value scrutinee that is not a
//! reference is put in a slot first, as Rust's MIR matches on a place:
//! the test reads its parts through references and moves nothing,
//! built so that a part is read only where the tests before it hold, and
//! binding takes each bound part once, where the whole test held. Through a
//! reference, as Rust's default binding modes do, the parts are places
//! under it and a name binds a reference to its part (patterns §1).

use acvus_ast::{AstId, Expr, Literal, MatchExprArm, Pattern, Slot, Span, TuplePatternElem};
use acvus_utils::Astr;

use super::call::convert;
use super::context;
use super::control::diamond;
use super::ctxt::{Home, Lcx, Lowered, Unreached};
use super::expr::lower_expr;
use super::place::{Operand, Place, assign, lend, read};
use super::stmt::lower_body;
use crate::ir::cfg::{BlockId, Op, Terminator};
use crate::ir::{BinOp, IndexBound, IndexMode, PathSeg, RefTarget, SwitchKey, ValueId, known_len, tested_ty};
use crate::ty::structural::{PartKey, VariantKey, structure};
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
    Payload(VariantKey),
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
    /// assigned at `span`, so that no test moves a part of it.
    pub fn scrutinee(cx: &mut Lcx<'_, '_>, value: ValueId, span: Span) -> Matched {
        let ty = cx.value_ty(value);
        if matches!(cx.kind(ty), TyKind::Ref(..)) {
            return Matched::Value { value, ty };
        }
        let slot = cx.slot(ty, span);
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
fn through(cx: &mut Lcx<'_, '_>, matched: &Matched, span: Span) -> Lowered<Matched> {
    let TyKind::Ref(held, target) = cx.kind(matched.ty()) else {
        return Ok(matched.clone());
    };
    let (held, target) = (*held, *target);
    let (reference, lent) = match matched {
        Matched::Value { value, .. } => (*value, held),
        Matched::Held { place, slot_ty } => (held_reference(cx, place, *slot_ty, held, span)?, held),
        Matched::Place { place, lent } => {
            let reference = read(cx, Operand::Place(place.clone()), span)?;
            let lent = lent.least(held);
            (super::place::reborrow(cx, reference, held, target, lent, span), lent)
        }
    };
    Ok(Matched::Place { place: Place { target: RefTarget::Through(reference), path: Vec::new(), ty: target }, lent })
}

/// The reference a held part is, read without moving it out of the slot. A
/// `&` copies, so it is read as it is. A `&mut` does not copy, and each use
/// reborrows it (ownership §5): it is read through a `&mut` to the slot, so
/// the slot keeps it and the drop plan sees no part of the slot move.
fn held_reference(cx: &mut Lcx<'_, '_>, place: &Place, slot_ty: Ty, held: Mutability, span: Span) -> Lowered<ValueId> {
    match held {
        Mutability::Shared => read(cx, Operand::Place(place.clone()), span),
        Mutability::Mut => {
            let slot = Place { target: place.target, path: Vec::new(), ty: slot_ty };
            let lent = lend(cx, Operand::Place(slot), Mutability::Mut, span)?;
            let op = Op::Take { target: RefTarget::Through(lent), path: place.path.clone(), taken_out: false };
            cx.emit(op, place.ty, span)
        }
    }
}

/// `part` of what `matched` is: a value's taken out of it (`Lcx::emit`); a
/// place's, the place under it.
fn part(cx: &mut Lcx<'_, '_>, matched: &Matched, part: Part, span: Span) -> Lowered<Matched> {
    let ty = part_ty(cx, matched.ty(), part);
    Ok(match matched {
        Matched::Value { value, .. } => {
            let op = match part {
                Part::Tuple(index) => Op::TupleIndex { tuple: *value, index },
                Part::Field(field) => Op::FieldGet { object: *value, field, rest: Vec::new() },
                Part::Payload(variant) => Op::UnwrapVariant { src: *value, variant },
                Part::Element(index) => Op::ArrayIndex { array: *value, index },
            };
            Matched::Value { value: cx.emit(op, ty, span)?, ty }
        }
        Matched::Held { place, slot_ty } => Matched::Held { place: place.with(segment(part), ty), slot_ty: *slot_ty },
        Matched::Place { place, lent } => Matched::Place { place: place.with(segment(part), ty), lent: *lent },
    })
}

fn segment(part: Part) -> PathSeg {
    match part {
        Part::Tuple(index) | Part::Element(index) => PathSeg::Index(index),
        Part::Field(field) => PathSeg::Field(field),
        Part::Payload(variant) => PathSeg::Payload(variant),
    }
}

/// The element a list pattern names at `at`: an array's is a part; a
/// slice's is reached by index, past its length test (`Lcx::emit`).
fn element(cx: &mut Lcx<'_, '_>, matched: &Matched, at: ListAt, span: Span) -> Lowered<Matched> {
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
    let slice = slice_ref(cx, place, lent, span)?;
    let u64_ty = cx.u64_ty();
    let index = match at {
        ListAt::SliceTail { back } => {
            let len = cx.made(Op::SliceLen { src: slice }, u64_ty, span);
            let back = cx.constant(Literal::Int(back as i128), u64_ty, span);
            let op = Op::BinOp { op: BinOp::Sub, left: len, right: back };
            cx.made(op, u64_ty, span)
        }
        ListAt::SliceHead(front) | ListAt::Array(front) => cx.constant(Literal::Int(front as i128), u64_ty, span),
    };
    let element_ref = cx.ref_ty(lent, ty);
    let op = Op::Index { slice, index, mode: IndexMode::Ref, bound: IndexBound::Checked };
    let reference = cx.emit(op, element_ref, span)?;
    Ok(Matched::Place { place: Place { target: RefTarget::Through(reference), path: Vec::new(), ty }, lent })
}

/// A reference, at `lent`, to the slice `place` holds.
fn slice_ref(cx: &mut Lcx<'_, '_>, place: &Place, lent: Mutability, span: Span) -> Lowered<ValueId> {
    let ty = cx.ref_ty(lent, place.ty);
    let op = Op::Ref { target: place.target, path: place.path.clone(), mutability: lent };
    cx.emit(op, ty, span)
}

/// The type of `part` of a `whole`.
///
/// # Panics
/// `whole` has no such part: typeck admits a pattern only on a type that
/// has what it names.
fn part_ty(cx: &Lcx<'_, '_>, whole: Ty, part: Part) -> Ty {
    let found = match (cx.kind(whole), part) {
        (TyKind::Tuple(items, _), Part::Tuple(index)) => items.get(index).copied(),
        (TyKind::Object { fields, .. }, Part::Field(field)) => {
            fields.iter().find(|(name, _)| *name == field).map(|(_, ty)| *ty)
        }
        (_, Part::Payload(variant)) => structure(cx.types(), whole).and_then(|structure| structure.part(PartKey::Payload(variant))),
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
    next: impl FnOnce(&mut Lcx<'_, '_>) -> Lowered<Option<ValueId>>,
) -> Lowered<Option<ValueId>> {
    let Some(held) = held else {
        return next(cx);
    };
    let bool_ty = cx.bool_ty();
    let joined = diamond(
        cx,
        held,
        bool_ty,
        span,
        |cx| match next(cx)? {
            Some(holds) => Ok(holds),
            None => Ok(cx.bool_const(true, span)),
        },
        |cx| Ok(cx.bool_const(false, span)),
    )?;
    Ok(Some(joined))
}

/// Whether `pattern` matches: `None` where it cannot fail.
pub(super) fn test<S: Slot>(cx: &mut Lcx<'_, '_>, pattern: &Pattern<S>, matched: &Matched) -> Lowered<Option<ValueId>> {
    acvus_utils::grow(|| test_level(cx, pattern, matched))
}

fn test_level<S: Slot>(cx: &mut Lcx<'_, '_>, pattern: &Pattern<S>, matched: &Matched) -> Lowered<Option<ValueId>> {
    let span = pattern.span();
    Ok(match pattern {
        Pattern::Binding { .. } | Pattern::Wildcard { .. } | Pattern::ContextBind { .. } => None,
        Pattern::Literal { value, .. } => {
            let matched = through(cx, matched, span)?;
            let src = tag_source(cx, &matched, span)?;
            let bool_ty = cx.bool_ty();
            Some(cx.made(Op::TestLiteral { src, value: value.clone() }, bool_ty, span))
        }
        Pattern::Variant { tag, payload, .. } => {
            let matched = through(cx, matched, span)?;
            let src = tag_source(cx, &matched, span)?;
            let bool_ty = cx.bool_ty();
            let variant = cx.variant(matched.ty(), *tag);
            let tagged = cx.made(Op::TestVariant { src, variant }, bool_ty, span);
            match payload {
                Some(payload) if refutable(payload) => and(cx, Some(tagged), span, |cx| {
                    let inner = part(cx, &matched, Part::Payload(variant), span)?;
                    test(cx, payload, &inner)
                })?,
                _ => Some(tagged),
            }
        }
        Pattern::Tuple { elements, .. } => {
            let matched = through(cx, matched, span)?;
            let mut held = None;
            for (index, element) in elements.iter().enumerate() {
                let TuplePatternElem::Pattern(element) = element else {
                    continue;
                };
                if !refutable(element) {
                    continue;
                }
                held = and(cx, held, span, |cx| {
                    let inner = part(cx, &matched, Part::Tuple(index), span)?;
                    test(cx, element, &inner)
                })?;
            }
            held
        }
        Pattern::Object { fields, .. } => {
            let matched = through(cx, matched, span)?;
            let mut held = None;
            for field in fields.iter().filter(|field| refutable(&field.pattern)) {
                held = and(cx, held, span, |cx| {
                    let inner = part(cx, &matched, Part::Field(field.key), span)?;
                    test(cx, &field.pattern, &inner)
                })?;
            }
            held
        }
        Pattern::List { head, rest, tail, .. } => {
            let matched = through(cx, matched, span)?;
            let mut held = slice_length_test(cx, &matched, head.len() + tail.len(), rest.is_some(), span)?;
            for (at, part_pattern) in list_elements(cx, &matched, head, tail) {
                if !refutable(part_pattern) {
                    continue;
                }
                held = and(cx, held, span, |cx| {
                    let inner = element(cx, &matched, at, span)?;
                    test(cx, part_pattern, &inner)
                })?;
            }
            held
        }
        Pattern::Error(_) => panic!("a pattern the parse left in error is refused, and never lowered"),
    })
}

/// A list pattern's length test: none on an array, whose type fixes its
/// length; on a slice, its length equal to the named elements', or at
/// least that where the list has a `..`.
fn slice_length_test(cx: &mut Lcx<'_, '_>, matched: &Matched, named: usize, rest: bool, span: Span) -> Lowered<Option<ValueId>> {
    let Matched::Place { place, lent } = matched else {
        return Ok(None);
    };
    if !matches!(cx.kind(place.ty), TyKind::Slice(_)) {
        return Ok(None);
    }
    let slice = slice_ref(cx, place, *lent, span)?;
    let u64_ty = cx.u64_ty();
    let len = cx.made(Op::SliceLen { src: slice }, u64_ty, span);
    let named = cx.constant(Literal::Int(named as i128), u64_ty, span);
    let op = match rest {
        true => BinOp::Gte,
        false => BinOp::Eq,
    };
    let bool_ty = cx.bool_ty();
    Ok(Some(cx.made(Op::BinOp { op, left: len, right: named }, bool_ty, span)))
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
/// out (`Lcx::emit`).
pub(super) fn tag_source(cx: &mut Lcx<'_, '_>, matched: &Matched, span: Span) -> Lowered<ValueId> {
    match matched {
        Matched::Value { value, .. } => Ok(*value),
        Matched::Held { place, .. } | Matched::Place { place, .. } => {
            let ty = cx.ref_ty(Mutability::Shared, place.ty);
            let op = Op::Ref { target: place.target, path: place.path.clone(), mutability: Mutability::Shared };
            cx.emit(op, ty, span)
        }
    }
}

/// Binds `pattern`'s names against `matched`, where its test held.
pub(super) fn bind<S: Slot>(cx: &mut Lcx<'_, '_>, pattern: &Pattern<S>, matched: &Matched) -> Lowered<()> {
    acvus_utils::grow(|| bind_level(cx, pattern, matched))
}

fn bind_level<S: Slot>(cx: &mut Lcx<'_, '_>, pattern: &Pattern<S>, matched: &Matched) -> Lowered<()> {
    let span = pattern.span();
    match pattern {
        Pattern::Binding { id, .. } => {
            let ty = cx.ty(*id);
            let value = bound(cx, matched, ty, span)?;
            let slot = cx.slot(ty, span);
            assign(cx, Place { target: RefTarget::Var(slot), path: Vec::new(), ty }, value, span);
            cx.bind(*id, Home::Slot(slot), span);
        }
        Pattern::ContextBind { id, name, .. } => {
            let value = match matched {
                Matched::Value { value, .. } => *value,
                Matched::Held { place, .. } => read(cx, Operand::Place(place.clone()), span)?,
                Matched::Place { .. } => {
                    panic!("a name bound past a reference is a reference, which typeck refuses to store into a context")
                }
            };
            let value = convert(cx, *id, value, span)?;
            context::store(cx, *name, value, span);
        }
        Pattern::Wildcard { .. } | Pattern::Literal { .. } => {}
        Pattern::Variant { tag, payload, .. } => {
            if let Some(payload) = payload {
                let matched = through(cx, matched, span)?;
                let variant = cx.variant(matched.ty(), *tag);
                let inner = part(cx, &matched, Part::Payload(variant), span)?;
                bind(cx, payload, &inner)?;
            }
        }
        Pattern::Tuple { elements, .. } => {
            let matched = through(cx, matched, span)?;
            for (index, element) in elements.iter().enumerate() {
                if let TuplePatternElem::Pattern(element) = element {
                    let inner = part(cx, &matched, Part::Tuple(index), span)?;
                    bind(cx, element, &inner)?;
                }
            }
        }
        Pattern::Object { fields, .. } => {
            let matched = through(cx, matched, span)?;
            for field in fields {
                let inner = part(cx, &matched, Part::Field(field.key), span)?;
                bind(cx, &field.pattern, &inner)?;
            }
        }
        Pattern::List { head, tail, .. } => {
            let matched = through(cx, matched, span)?;
            for (at, part_pattern) in list_elements(cx, &matched, head, tail) {
                let inner = element(cx, &matched, at, span)?;
                bind(cx, part_pattern, &inner)?;
            }
        }
        Pattern::Error(_) => panic!("a pattern the parse left in error is refused, and never lowered"),
    }
    Ok(())
}

/// What a name binds: the part itself, or past a reference, a reference to
/// it at the binder's mutability (patterns §1).
fn bound(cx: &mut Lcx<'_, '_>, matched: &Matched, ty: Ty, span: Span) -> Lowered<ValueId> {
    match matched {
        Matched::Value { value, ty: held } => {
            assert_eq!(*held, ty, "a name bound to a value has the value's type");
            Ok(*value)
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

/// The one-dispatch plan of a `match` whose arms each test one variant of
/// one variant type, or one literal of one kind, the first catch-all being
/// the default; `None` for any other. The `Switch` reads the scrutinee's
/// type as `ir::tested_ty` states.
fn switch_plan<S: Slot>(cx: &Lcx<'_, '_>, scrutinee: Ty, arms: &[MatchExprArm<S>]) -> Option<SwitchPlan> {
    let tested = tested_ty(cx.types(), scrutinee);
    let of_variants = structure(cx.types(), tested).is_some_and(|structure| structure.variants().is_some());
    let mut keyed: Vec<(SwitchKey, usize)> = Vec::new();
    let mut default = None;
    for (at, arm) in arms.iter().enumerate() {
        let key = match &arm.pattern {
            Pattern::Wildcard { .. } | Pattern::Binding { .. } => {
                default = Some(at);
                break;
            }
            Pattern::Variant { tag, payload, .. } if of_variants && payload.as_deref().is_none_or(|payload| !refutable(payload)) => {
                SwitchKey::Variant(cx.variant(tested, *tag))
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
/// name interned text, and lowering holds no interner, so a string
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
) -> Lowered<ValueId> {
    let value = lower_expr(cx, scrutinee)?;
    let matched = Matched::scrutinee(cx, value, scrutinee.span());
    let ty = cx.ty(site);
    // The tag a `Switch` reads, before any block of the match is made.
    let dispatch = match switch_plan(cx, matched.ty(), arms) {
        Some(plan) => Some((plan, tag_source(cx, &matched, span)?)),
        None => None,
    };
    let join = cx.new_block(&[ty]);
    let mut joined = false;
    match dispatch {
        Some((plan, tag)) => {
            let keyed: Vec<(SwitchKey, BlockId, usize)> =
                plan.keyed.iter().map(|(key, at)| (*key, cx.new_block(&[]), *at)).collect();
            let default = plan.default.map(|at| (cx.new_block(&[]), at));
            // No catch-all: a value no key names goes to the block where no
            // arm answers, which matchck judges (patterns §3).
            let unmatched = default.is_none().then(|| cx.new_block(&[]));
            let terminator = Terminator::Switch {
                tag,
                arms: keyed.iter().map(|(key, block, _)| (*key, *block, Vec::new())).collect(),
                default: default.map(|(block, _)| block).or(unmatched).map(|block| (block, Vec::new())),
                join,
            };
            cx.terminate(terminator, span);
            if let Some(unmatched) = unmatched
                && cx.enter_block(unmatched, span).is_ok()
            {
                cx.end_block(Terminator::NoArmMatched { span }, span);
            }
            let entered = keyed.iter().map(|(_, block, at)| (*block, *at));
            for (block, at) in entered.chain(default) {
                if cx.enter_block(block, arms[at].span).is_ok() {
                    joined |= arm_body(cx, &arms[at], &matched, join);
                }
            }
        }
        None => {
            // The block a value no arm's test holds reaches: open unless an
            // arm catches every value or no run gets past a test.
            let mut open = true;
            for arm in arms {
                let holds = match test(cx, &arm.pattern, &matched) {
                    Ok(Some(holds)) => holds,
                    Ok(None) => {
                        joined |= arm_body(cx, arm, &matched, join);
                        open = false;
                        break;
                    }
                    Err(Unreached) => {
                        open = false;
                        break;
                    }
                };
                let then_to = cx.new_block(&[]);
                let else_to = cx.new_block(&[]);
                let terminator = Terminator::Diamond {
                    cond: holds,
                    then_to,
                    then_args: Vec::new(),
                    else_to,
                    else_args: Vec::new(),
                    join,
                };
                cx.terminate(terminator, arm.span);
                if cx.enter_block(then_to, arm.span).is_ok() {
                    joined |= arm_body(cx, arm, &matched, join);
                }
                if cx.enter_block(else_to, arm.span).is_err() {
                    open = false;
                    break;
                }
            }
            if open {
                cx.end_block(Terminator::NoArmMatched { span }, span);
            }
        }
    }
    // The join is a block of the match all the same: where no arm
    // reaches it, it ends in `Diverge`.
    let params = cx.enter_block(join, span)?;
    match joined {
        true => Ok(params[0]),
        false => Err(cx.end_block(Terminator::Diverge, span)),
    }
}

/// One arm: its names bound, its body lowered, its value entering the
/// match's one slot at its tail's site or the arm's; whether a run gets
/// past it to the join.
fn arm_body<S: Slot>(cx: &mut Lcx<'_, '_>, arm: &MatchExprArm<S>, matched: &Matched, join: BlockId) -> bool {
    let at = arm.tail.as_deref().map_or(arm.id, Expr::id);
    let lowered = bind(cx, &arm.pattern, matched)
        .and_then(|()| lower_body(cx, &arm.body, arm.tail.as_deref(), arm.span))
        .and_then(|value| convert(cx, at, value, arm.span));
    let Ok(value) = lowered else {
        return false;
    };
    cx.jump(join, vec![value], arm.span);
    true
}
