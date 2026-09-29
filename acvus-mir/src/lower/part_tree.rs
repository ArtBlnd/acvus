//! Equality and ordering, and the impls a transparent structured type has
//! by a lift, by the structure tree typeck recorded (operators §4, types §1,
//! D87). A tuple, an object, an `Option` and a `Result` are expanded part by
//! part in `ty::structural`'s order. An array is one `For` over its
//! elements, never unrolled: a comparison leaves it at the first element
//! whose answer settles the whole. A leaf is a call of the impl reached.
//!
//! A node's operand is a shared reference to its value, or, where the node
//! is itself a reference, that reference, since a reference's target is
//! never a reference (types §4).

use acvus_ast::{AstId, BinOp, Expr, Literal, Slot, Span};
use acvus_utils::Astr;

use super::call::{convert, issue, lent_operand};
use super::control::diamond;
use super::ctxt::Lcx;
use super::place::view_array;
use crate::ir::cfg::{BlockId, Op, Stages, Terminator};
use crate::ir::{ExitTrip, ForSource, IndexBound, IndexMode, PathSeg, RefTarget, UnaryOp, ValueId, known_len};
use crate::ty::structural::{Structure, structure};
use crate::ty::{IntTy, Mutability, Ty, TyKind};
use crate::typeck::PartTree;

/// What a comparison asks of its operands (operators §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Asked {
    /// `==`: a `bool`.
    Equal,
    /// An ordering: `cmp`'s `i64`, `-1`, `0` or `1` (operators §2).
    Order,
}

/// A node's type as its tree takes it apart: a structural type by
/// `ty::structural`, a reference by what it names, `!` by no part.
enum Shape {
    Tuple(Vec<Ty>),
    Object(Vec<(Astr, Ty)>),
    Array(ArrayOf),
    Option(Ty),
    Result { ok: Ty, err: Ty },
    Ref(Ty),
    Never,
}

/// An array's element type and length.
#[derive(Clone, Copy)]
struct ArrayOf {
    element: Ty,
    len: usize,
}

/// # Panics
/// `ty` is neither structural, a reference nor `!`: typeck records parts
/// only at those.
fn shape(cx: &Lcx<'_, '_>, ty: Ty) -> Shape {
    match structure(cx.builder.types(), ty) {
        Some(Structure::Tuple(items)) => Shape::Tuple(items.to_vec()),
        Some(Structure::Object(fields)) => Shape::Object(fields.to_vec()),
        Some(Structure::Array { element, len }) => Shape::Array(ArrayOf { element, len: known_len(&len) }),
        Some(Structure::Option(payload)) => Shape::Option(payload),
        Some(Structure::Result { ok, err }) => Shape::Result { ok, err },
        None => match cx.kind(ty) {
            TyKind::Ref(_, target) => Shape::Ref(target.ty()),
            TyKind::Never => Shape::Never,
            other => panic!("typeck records parts only at a structural type, a reference or `!`, not {other:?}"),
        },
    }
}

/// One part of a node: the step to it, its type, and its tree.
#[derive(Clone, Copy)]
struct Part<'t> {
    seg: PathSeg,
    ty: Ty,
    tree: &'t PartTree,
}

/// A tuple's parts, each with its tree.
fn tuple_parts<'t>(items: &[Ty], trees: &'t [PartTree]) -> Vec<Part<'t>> {
    items.iter().zip(trees).enumerate().map(|(at, (ty, tree))| Part { seg: PathSeg::Index(at), ty: *ty, tree }).collect()
}

/// An object's parts, each with its tree.
fn field_parts<'t>(fields: &[(Astr, Ty)], trees: &'t [PartTree]) -> Vec<Part<'t>> {
    fields.iter().zip(trees).map(|((name, ty), tree)| Part { seg: PathSeg::Field(*name), ty: *ty, tree }).collect()
}

/// The only part tree of a node of one part.
fn only(trees: &[PartTree]) -> &PartTree {
    let [tree] = trees else {
        panic!("typeck records one part at a node of one part type, not {}", trees.len())
    };
    tree
}

/// The operand of `part` of the node whose operand is `base`.
fn part_operand(cx: &mut Lcx<'_, '_>, base: ValueId, part: Part<'_>, span: Span) -> ValueId {
    let target = RefTarget::Through(base);
    let path = vec![part.seg];
    match cx.kind(part.ty) {
        TyKind::Ref(..) => cx.builder.push(Op::Take { target, path, taken_out: false }, part.ty, span),
        _ => {
            let ty = cx.ref_ty(Mutability::Shared, part.ty);
            cx.builder.push(Op::Ref { target, path, mutability: Mutability::Shared }, ty, span)
        }
    }
}

/// The operand of element `at` of the slice `elements` of `element`s.
fn element_operand(cx: &mut Lcx<'_, '_>, elements: ValueId, at: ValueId, element: Ty, span: Span) -> ValueId {
    let (mode, ty) = match cx.kind(element) {
        TyKind::Ref(..) => (IndexMode::Copy, element),
        _ => (IndexMode::Ref, cx.ref_ty(Mutability::Shared, element)),
    };
    cx.builder.push(Op::Index { slice: elements, index: at, mode, bound: IndexBound::Checked }, ty, span)
}

// -- comparison --------------------------------------------------------------

/// `==` or an ordering by the tree typeck recorded at `site` (operators
/// §4); an ordering is `cmp`'s answer compared with `0` (operators §2).
pub(super) fn compare<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, op: BinOp, left: &Expr<S>, right: &Expr<S>, span: Span) -> ValueId {
    let tree = cx
        .typeck
        .part_trees
        .get(&site)
        .unwrap_or_else(|| panic!("typeck records how every comparison compares, and not at {site:?}"));
    let asked = match op {
        BinOp::Eq | BinOp::Neq => Asked::Equal,
        _ => Asked::Order,
    };
    let compared_ty = cx.ty(left.id());
    let lhs = lent_operand(cx, left);
    let rhs = lent_operand(cx, right);
    let lhs = convert(cx, site, lhs, span);
    let rhs = convert(cx, site, rhs, span);
    let answer = compared(cx, tree, asked, compared_ty, lhs, rhs, span);
    let bool_ty = cx.ty(site);
    match op {
        BinOp::Eq => answer,
        BinOp::Neq => cx.builder.push(Op::UnaryOp { op: UnaryOp::Not, operand: answer }, bool_ty, span),
        BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
            let answer_ty = cx.value_ty(answer);
            let zero = cx.constant(Literal::Int(0), answer_ty, span);
            cx.builder.push(Op::BinOp { op: op.into(), left: answer, right: zero }, bool_ty, span)
        }
        other => panic!("{other:?} is not a comparison"),
    }
}

/// `a` against `b`, the operands of a node of type `ty`.
fn compared(cx: &mut Lcx<'_, '_>, tree: &PartTree, asked: Asked, ty: Ty, a: ValueId, b: ValueId, span: Span) -> ValueId {
    let trees = match tree {
        PartTree::Reached { chosen, callee } => {
            let called = cx.callee(chosen);
            return issue(cx, called, *callee, vec![a, b], span);
        }
        PartTree::Parts(trees) => trees,
    };
    match shape(cx, ty) {
        Shape::Ref(target) => compared(cx, only(trees), asked, target, a, b, span),
        Shape::Tuple(items) => in_order(cx, asked, &tuple_parts(&items, trees), a, b, span),
        Shape::Object(fields) => in_order(cx, asked, &field_parts(&fields, trees), a, b, span),
        Shape::Array(array) => array_compared(cx, asked, array, only(trees), a, b, span),
        Shape::Option(payload) => {
            let tags = cx.decls.core().tags;
            let payload = Part { seg: PathSeg::Payload, ty: payload, tree: only(trees) };
            tagged(cx, asked, tags.some, Tagged::Option(payload), a, b, span)
        }
        Shape::Result { ok, err } => {
            let tags = cx.decls.core().tags;
            let [ok_tree, err_tree] = &trees[..] else { panic!("a `Result` compares by its two payloads") };
            let ok = Part { seg: PathSeg::Payload, ty: ok, tree: ok_tree };
            let err = Part { seg: PathSeg::Payload, ty: err, tree: err_tree };
            tagged(cx, asked, tags.ok, Tagged::Result { ok, err }, a, b, span)
        }
        Shape::Never => answered(cx, asked, span),
    }
}

/// A variant type's payloads: `Option`'s one, or `Result`'s two.
enum Tagged<'t> {
    Option(Part<'t>),
    Result { ok: Part<'t>, err: Part<'t> },
}

/// What a comparison of no part, or of equal parts only, answers: equal.
fn answered(cx: &mut Lcx<'_, '_>, asked: Asked, span: Span) -> ValueId {
    match asked {
        Asked::Equal => cx.bool_const(true, span),
        Asked::Order => order_const(cx, 0, span),
    }
}

fn answer_ty(cx: &mut Lcx<'_, '_>, asked: Asked) -> Ty {
    match asked {
        Asked::Equal => cx.bool_ty(),
        Asked::Order => cx.builder.intern(TyKind::Int(IntTy::I64)),
    }
}

fn order_const(cx: &mut Lcx<'_, '_>, value: i128, span: Span) -> ValueId {
    let i64_ty = cx.builder.intern(TyKind::Int(IntTy::I64));
    cx.constant(Literal::Int(value), i64_ty, span)
}

/// Whether the answer so far leaves the comparison open: equal so far.
fn still_open(cx: &mut Lcx<'_, '_>, asked: Asked, answer: ValueId, span: Span) -> ValueId {
    match asked {
        Asked::Equal => answer,
        Asked::Order => {
            let zero = order_const(cx, 0, span);
            let bool_ty = cx.bool_ty();
            cx.builder.push(Op::BinOp { op: crate::ir::BinOp::Eq, left: answer, right: zero }, bool_ty, span)
        }
    }
}

fn part_compared(cx: &mut Lcx<'_, '_>, asked: Asked, part: Part<'_>, a: ValueId, b: ValueId, span: Span) -> ValueId {
    let a = part_operand(cx, a, part, span);
    let b = part_operand(cx, b, part, span);
    compared(cx, part.tree, asked, part.ty, a, b, span)
}

/// Parts compared in order, as Rust's derived comparisons do: equal while
/// every part is, and ordered by the first part that is not equal. A part
/// is compared only where the parts before it left the answer open.
fn in_order(cx: &mut Lcx<'_, '_>, asked: Asked, parts: &[Part<'_>], a: ValueId, b: ValueId, span: Span) -> ValueId {
    let mut answer: Option<ValueId> = None;
    for part in parts {
        let next = |cx: &mut Lcx<'_, '_>| part_compared(cx, asked, *part, a, b, span);
        answer = Some(match answer {
            None => next(cx),
            Some(before) => {
                let open = still_open(cx, asked, before, span);
                let answer_ty = cx.value_ty(before);
                diamond(cx, open, answer_ty, span, next, |_| before)
            }
        });
    }
    match answer {
        Some(answer) => answer,
        None => answered(cx, asked, span),
    }
}

/// Two arrays compared element by element in one `For` over their
/// indices, as Rust compares arrays: the loop leaves at the first element
/// whose answer is not equal, with that answer, and answers equal where it
/// runs to its end.
fn array_compared(cx: &mut Lcx<'_, '_>, asked: Asked, array: ArrayOf, tree: &PartTree, a: ValueId, b: ValueId, span: Span) -> ValueId {
    let ArrayOf { element, len } = array;
    let a_elements = view_array(cx, a, Mutability::Shared, element, span);
    let b_elements = view_array(cx, b, Mutability::Shared, element, span);
    let all_equal = answered(cx, asked, span);
    let answer_ty = answer_ty(cx, asked);
    indices(cx, len, answer_ty, all_equal, span, |cx, at, header, exit| {
        let x = element_operand(cx, a_elements, at, element, span);
        let y = element_operand(cx, b_elements, at, element, span);
        let answer = compared(cx, tree, asked, element, x, y, span);
        let open = still_open(cx, asked, answer, span);
        Terminator::JumpIf { cond: open, then_to: header, then_args: Vec::new(), else_to: exit, else_args: vec![answer] }
    })
}

/// A `For` over the indices `0..len`, a `u64` each (operators §3). `body`
/// lowers one iteration from its index and ends it with the terminator it
/// returns, given the loop's header and exit; the exit takes one value of
/// `exit_ty`, `at_end` where the loop runs to its end. The value the exit
/// takes is the result.
fn indices(
    cx: &mut Lcx<'_, '_>,
    len: usize,
    exit_ty: Ty,
    at_end: ValueId,
    span: Span,
    body: impl FnOnce(&mut Lcx<'_, '_>, ValueId, BlockId, BlockId) -> Terminator,
) -> ValueId {
    let u64_ty = cx.u64_ty();
    let len = u64::try_from(len).expect("an array's length fits its `u64` index (operators §3)");
    let from = cx.constant(Literal::Int(0), u64_ty, span);
    let to = cx.constant(Literal::Int(i128::from(len)), u64_ty, span);
    let live = cx.reachable;
    let header = cx.builder.new_block(&[]);
    cx.jump(header, Vec::new(), span);
    cx.builder.switch_to(header);
    cx.reachable = live;
    let body_block = cx.builder.new_block(&[u64_ty]);
    let exit = cx.builder.new_block(&[exit_ty]);
    let terminator = Terminator::For {
        source: ForSource::Range { at: from, hi: to },
        stages: Stages::lowered(body_block),
        exit,
        exit_trip: ExitTrip::Absent,
        exit_args: vec![at_end],
    };
    cx.builder.terminate(terminator, span);
    cx.builder.switch_to(body_block);
    cx.reachable = live;
    let at = cx.builder.block_params(body_block)[0];
    let end = body(cx, at, header, exit);
    let end = match cx.reachable {
        true => end,
        false => Terminator::Diverge,
    };
    cx.builder.terminate(end, span);
    cx.builder.switch_to(exit);
    cx.reachable = live;
    cx.builder.block_params(exit)[0]
}

/// An `Option` or a `Result` compared: by tag first, `None` before `Some`
/// and `Ok` before `Err` as Rust derives them, then by the payload the two
/// share.
fn tagged(cx: &mut Lcx<'_, '_>, asked: Asked, first: Astr, payloads: Tagged<'_>, a: ValueId, b: ValueId, span: Span) -> ValueId {
    let bool_ty = cx.bool_ty();
    let a_first = cx.builder.push(Op::TestVariant { src: a, tag: first }, bool_ty, span);
    let b_first = cx.builder.push(Op::TestVariant { src: b, tag: first }, bool_ty, span);
    let same = cx.builder.push(Op::BinOp { op: crate::ir::BinOp::Eq, left: a_first, right: b_first }, bool_ty, span);
    let answer_ty = answer_ty(cx, asked);
    let payloads_compared = |cx: &mut Lcx<'_, '_>| match payloads {
        Tagged::Option(payload) => {
            diamond(cx, a_first, answer_ty, span, |cx| part_compared(cx, asked, payload, a, b, span), |cx| answered(cx, asked, span))
        }
        Tagged::Result { ok, err } => diamond(
            cx,
            a_first,
            answer_ty,
            span,
            |cx| part_compared(cx, asked, ok, a, b, span),
            |cx| part_compared(cx, asked, err, a, b, span),
        ),
    };
    let tags_differ = |cx: &mut Lcx<'_, '_>| match asked {
        Asked::Equal => cx.bool_const(false, span),
        Asked::Order => {
            let (when_first, otherwise) = match payloads {
                Tagged::Option(_) => (1, -1),
                Tagged::Result { .. } => (-1, 1),
            };
            diamond(cx, a_first, answer_ty, span, |cx| order_const(cx, when_first, span), |cx| order_const(cx, otherwise, span))
        }
    };
    diamond(cx, same, answer_ty, span, payloads_compared, tags_differ)
}

// -- lifted impl -------------------------------------------------------------

/// A call typeck answered by a lifted impl at `site` (types §1,
/// `Impl::Structural`), of `args`: by the lift's shape, each part's
/// arguments are the arguments' parts, each part's impl is called at its
/// node, and the results rebuild the structure (`Lift::Rebuild`) or are
/// combined as the lift's law defines (`Lift::Combine`). An array's parts
/// are one `For` over its elements, never unrolled.
pub(super) fn lifted(cx: &mut Lcx<'_, '_>, site: AstId, args: Vec<ValueId>, span: Span) -> ValueId {
    todo!("the structure tree at `site`, expanded by the reached generic function's lift")
}
