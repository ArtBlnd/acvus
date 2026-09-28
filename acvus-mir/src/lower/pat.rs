//! Patterns (patterns §1). A `match` whose arms all test one enum's tags or
//! one kind of literal is one `Switch` dispatch, with a default where it has
//! a catch-all, and its join held by the `Switch`; any other `match` tests
//! its arms one at a time, in order, within the same region and to the same
//! join (D57). Exhaustiveness is `check`'s.
//!
//! A pattern is tested, then bound. The test is a `bool` built so that a
//! part is taken only where the tests before it hold; binding takes the
//! parts again where the whole test held. Through a reference, as Rust's
//! default binding modes do, the parts are places under it and a name binds
//! a reference to its part (patterns §1).

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
use crate::ir::{PathSeg, RefTarget, SwitchKey, ValueId, known_len};
use crate::ty::{Mutability, Ty, TyKind};

/// What a pattern is matched against: a value, or, past a reference, a
/// place under it, lent at `lent`.
#[derive(Debug, Clone)]
pub(super) enum Matched {
    Value { value: ValueId, ty: Ty },
    Place { place: Place, lent: Mutability },
}

/// A part a pattern names of what it matches.
#[derive(Debug, Clone, Copy)]
enum Part {
    Tuple(usize),
    Field(Astr),
    Payload(Astr),
    Element(usize),
}

impl Matched {
    pub fn value(cx: &Lcx<'_, '_>, value: ValueId) -> Matched {
        Matched::Value { value, ty: cx.value_ty(value) }
    }

    fn ty(&self) -> Ty {
        match self {
            Matched::Value { ty, .. } => *ty,
            Matched::Place { place, .. } => place.ty,
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
    let (held, target) = (*held, target.ty());
    let (reference, lent) = match matched {
        Matched::Value { value, .. } => (*value, held),
        Matched::Place { place, lent } => {
            let reference = read(cx, Operand::Place(place.clone()), span);
            let lent = match (lent, held) {
                (Mutability::Mut, Mutability::Mut) => Mutability::Mut,
                _ => Mutability::Shared,
            };
            (reference, lent)
        }
    };
    Matched::Place { place: Place { target: RefTarget::Through(reference), path: Vec::new(), ty: target }, lent }
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
        Matched::Place { place, lent } => {
            let seg = match part {
                Part::Tuple(index) | Part::Element(index) => PathSeg::Index(index),
                Part::Field(field) => PathSeg::Field(field),
                Part::Payload(_) => PathSeg::Payload,
            };
            Matched::Place { place: place.with(seg, ty), lent: *lent }
        }
    }
}

/// The type of `part` of a `whole`.
///
/// # Panics
/// `whole` has no such part: typeck admits a pattern only on a type that
/// has what it names.
fn part_ty(cx: &Lcx<'_, '_>, whole: Ty, part: Part) -> Ty {
    let tags = cx.decls.core().tags;
    let found = match (cx.kind(whole), part) {
        (TyKind::Tuple(items), Part::Tuple(index)) => items.get(index).copied(),
        (TyKind::Object(object), Part::Field(field)) => {
            object.fields().iter().find(|(name, _)| *name == field).map(|(_, ty)| *ty)
        }
        (TyKind::Option(payload), Part::Payload(_)) => Some(*payload),
        (TyKind::Result(ok, err), Part::Payload(tag)) => Some(if tag == tags.ok { *ok } else { *err }),
        (TyKind::Enum { variants, .. }, Part::Payload(tag)) => {
            variants.iter().find(|(name, _)| *name == tag).and_then(|(_, payload)| *payload)
        }
        (TyKind::Array(element, _), Part::Element(_)) => Some(*element),
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
            let src = match matched {
                Matched::Value { value, .. } => value,
                Matched::Place { place, .. } => read(cx, Operand::Place(place), span),
            };
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
        Pattern::List { head, tail, .. } => {
            let matched = through(cx, matched, span);
            let mut held = None;
            for (index, element) in list_elements(cx, &matched, head, tail) {
                if !refutable(element) {
                    continue;
                }
                held = and(cx, held, span, |cx| {
                    let inner = part(cx, &matched, Part::Element(index), span);
                    test(cx, element, &inner)
                });
            }
            held
        }
        Pattern::Error(_) => panic!("a pattern the parse left in error is refused, and never lowered"),
    }
}

/// Each element a list pattern names, with its index in the array.
///
/// # Panics
/// The list is matched against a slice: its length is known only when
/// the program runs, and the IR has no operation that reads a slice's
/// length, so no rule lowers the length test yet.
fn list_elements<'p, S: Slot>(
    cx: &Lcx<'_, '_>,
    matched: &Matched,
    head: &'p [Pattern<S>],
    tail: &'p [Pattern<S>],
) -> Vec<(usize, &'p Pattern<S>)> {
    let len = match cx.kind(matched.ty()) {
        TyKind::Array(_, len) => known_len(len),
        TyKind::Slice(_) => todo!(
            "a list pattern on a slice: its length is tested when the program runs, and the IR has no operation \
             that reads a slice's length"
        ),
        other => panic!("typeck admits a list pattern only on an array or a slice, not {other:?}"),
    };
    let from_end = len - tail.len();
    head.iter().enumerate().chain(tail.iter().enumerate().map(|(at, part)| (from_end + at, part))).collect()
}

/// What a tag test reads: the value, or a shared reference to the place,
/// read through as a `Switch` reads its tag through one.
fn tag_source(cx: &mut Lcx<'_, '_>, matched: &Matched, span: Span) -> ValueId {
    match matched {
        Matched::Value { value, .. } => *value,
        Matched::Place { place, .. } => {
            let ty = cx.ref_ty(Mutability::Shared, place.ty);
            let op = Op::Ref { target: place.target, path: place.path.clone(), mutability: Mutability::Shared };
            cx.builder.push(op, ty, span)
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
            for (index, element) in list_elements(cx, &matched, head, tail) {
                let inner = part(cx, &matched, Part::Element(index), span);
                bind(cx, element, &inner);
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
        TyKind::Ref(_, target) => target.ty(),
        _ => scrutinee,
    };
    let tagged = matches!(cx.kind(tested), TyKind::Option(_) | TyKind::Result(..) | TyKind::Enum { .. });
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
    let matched = Matched::value(cx, value);
    let ty = cx.ty(site);
    let live = cx.reachable;
    let join = cx.builder.new_block(&[ty]);
    let mut joined = false;
    match switch_plan(cx, matched.ty(), arms) {
        Some(plan) => {
            let keyed: Vec<(SwitchKey, BlockId, usize)> =
                plan.keyed.iter().map(|(key, at)| (*key, cx.builder.new_block(&[]), *at)).collect();
            let default = plan.default.map(|at| (cx.builder.new_block(&[]), at));
            let terminator = Terminator::Switch {
                tag: value,
                arms: keyed.iter().map(|(key, block, _)| (*key, *block, Vec::new())).collect(),
                default: default.map(|(block, _)| (block, Vec::new())),
                join,
            };
            cx.builder.terminate(terminator, span);
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
                cx.builder.terminate(Terminator::Diverge, span);
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
