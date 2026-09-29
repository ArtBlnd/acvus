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
use super::coerce::{Variant, by_variant};
use super::control::diamond;
use super::ctxt::Lcx;
use super::place::view_array;
use crate::decl::{Lift, LiftShape, Taking};
use crate::ir::cfg::{BlockId, Op, Stages, Terminator};
use crate::ir::{ExitTrip, ForSource, IndexBound, IndexMode, PathSeg, RefTarget, UnaryOp, ValueId, known_len};
use crate::ty::structural::{PartKey, Structure, structure};
use crate::ty::{IntTy, Mutability, Ty, TyKind};
use crate::typeck::StructureTree;

/// What a comparison asks of its operands (operators §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Asked {
    /// `==`: a `bool`.
    Equal,
    /// An ordering: `cmp`'s `i64`, `-1`, `0` or `1` (operators §2).
    Order,
}

/// A node's type as its tree takes it apart: a transparent structured type
/// by `ty::structural`, a reference by what it names, `!` by no part.
enum Shape {
    Tuple(Vec<Ty>),
    Object(Vec<(Astr, Ty)>),
    Array(ArrayOf),
    Option(Ty),
    Result { ok: Ty, err: Ty },
    Enum(Vec<(Astr, Option<Ty>)>),
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
/// `ty` is neither a transparent structured type, a reference nor `!`:
/// typeck records parts only at those.
fn shape(cx: &Lcx<'_, '_>, ty: Ty) -> Shape {
    match structure(cx.builder.types(), ty) {
        Some(Structure::Tuple(items, _)) => Shape::Tuple(items.to_vec()),
        Some(Structure::Object(fields, _)) => Shape::Object(fields.to_vec()),
        Some(Structure::Enum(variants, _)) => Shape::Enum(variants.to_vec()),
        Some(Structure::Array { element, len, .. }) => Shape::Array(ArrayOf { element, len: known_len(&len) }),
        Some(Structure::Option(payload, _)) => Shape::Option(payload),
        Some(Structure::Result { ok, err, .. }) => Shape::Result { ok, err },
        None => match cx.kind(ty) {
            TyKind::Ref(_, target) => Shape::Ref(target.ty()),
            TyKind::Never => Shape::Never,
            other => panic!("typeck records parts only at a structured type, a reference or `!`, not {other:?}"),
        },
    }
}

/// The variants of a node of `shape`, where it is a variant type.
fn variants(cx: &Lcx<'_, '_>, shape: &Shape) -> Option<Vec<Variant>> {
    match shape {
        Shape::Option(payload) => Some(Variant::of_option(cx, *payload).into()),
        Shape::Result { ok, err } => Some(Variant::of_result(cx, *ok, *err).into()),
        Shape::Enum(variants) => Some(Variant::of_enum(variants)),
        Shape::Tuple(_) | Shape::Object(_) | Shape::Array(_) | Shape::Ref(_) | Shape::Never => None,
    }
}

/// One part of a node: the step to it, its type, and its tree.
#[derive(Clone, Copy)]
struct Part<'t> {
    seg: PathSeg,
    ty: Ty,
    tree: &'t StructureTree,
}

/// The tree of the part at `key` among a node's `trees`.
///
/// # Panics
/// typeck recorded no part at `key`: it records one per part of the
/// node's type.
fn tree_at(trees: &[(PartKey, StructureTree)], key: PartKey) -> &StructureTree {
    trees
        .iter()
        .find(|(at, _)| *at == key)
        .map(|(_, tree)| tree)
        .unwrap_or_else(|| panic!("typeck records a tree at every part of a node, and not at {key:?}"))
}

/// A tuple's parts, each with its tree.
fn tuple_parts<'t>(items: &[Ty], trees: &'t [(PartKey, StructureTree)]) -> Vec<Part<'t>> {
    items
        .iter()
        .enumerate()
        .map(|(at, ty)| Part { seg: PathSeg::Index(at), ty: *ty, tree: tree_at(trees, PartKey::Index(at)) })
        .collect()
}

/// An object's parts, each with its tree.
fn field_parts<'t>(fields: &[(Astr, Ty)], trees: &'t [(PartKey, StructureTree)]) -> Vec<Part<'t>> {
    fields
        .iter()
        .map(|(name, ty)| Part { seg: PathSeg::Field(*name), ty: *ty, tree: tree_at(trees, PartKey::Field(*name)) })
        .collect()
}

/// The payload part of `variant`, where it has one.
fn payload_part<'t>(variant: &Variant, trees: &'t [(PartKey, StructureTree)]) -> Option<Part<'t>> {
    variant.payload.map(|(key, ty)| Part { seg: PathSeg::Payload, ty, tree: tree_at(trees, key) })
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
        .structure_trees
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
fn compared(cx: &mut Lcx<'_, '_>, tree: &StructureTree, asked: Asked, ty: Ty, a: ValueId, b: ValueId, span: Span) -> ValueId {
    let trees = match tree {
        StructureTree::Reached { chosen, callee } => {
            let called = cx.callee(chosen);
            return issue(cx, called, *callee, vec![a, b], span);
        }
        StructureTree::Parts(trees) => trees,
    };
    match shape(cx, ty) {
        // typeck compares a reference at its own node by what it names.
        Shape::Ref(target) => compared(cx, tree, asked, target, a, b, span),
        Shape::Tuple(items) => in_order(cx, asked, &tuple_parts(&items, trees), a, b, span),
        Shape::Object(fields) => in_order(cx, asked, &field_parts(&fields, trees), a, b, span),
        Shape::Array(array) => array_compared(cx, asked, array, tree_at(trees, PartKey::Element), a, b, span),
        Shape::Option(payload) => {
            let tags = cx.decls.core().tags;
            let payload = Part { seg: PathSeg::Payload, ty: payload, tree: tree_at(trees, PartKey::Some) };
            tagged(cx, asked, tags.some, Tagged::Option(payload), a, b, span)
        }
        Shape::Result { ok, err } => {
            let tags = cx.decls.core().tags;
            let ok = Part { seg: PathSeg::Payload, ty: ok, tree: tree_at(trees, PartKey::Ok) };
            let err = Part { seg: PathSeg::Payload, ty: err, tree: tree_at(trees, PartKey::Err) };
            tagged(cx, asked, tags.ok, Tagged::Result { ok, err }, a, b, span)
        }
        Shape::Enum(variants) => match asked {
            Asked::Equal => enum_equal(cx, &Variant::of_enum(&variants), trees, a, b, span),
            // NOTE: an open question, returned to the coordinator. A program's
            // enum has no declared order of its variants: its type keeps them
            // sorted by interned tag, which orders nothing the program wrote,
            // so no ledger sentence says which of two variants is less.
            Asked::Order => todo!("the order of two variants of a program's enum is not determined"),
        },
        Shape::Never => answered(cx, asked, span),
    }
}

/// Two values of an enum equal: of one variant, and of equal payloads
/// where it has one.
fn enum_equal(
    cx: &mut Lcx<'_, '_>,
    variants: &[Variant],
    trees: &[(PartKey, StructureTree)],
    a: ValueId,
    b: ValueId,
    span: Span,
) -> ValueId {
    let bool_ty = cx.bool_ty();
    if variants.is_empty() {
        return answered(cx, Asked::Equal, span);
    }
    by_variant(cx, variants, bool_ty, a, span, &|cx, variant| {
        let same = cx.builder.push(Op::TestVariant { src: b, tag: variant.tag }, bool_ty, span);
        let payloads = |cx: &mut Lcx<'_, '_>| match payload_part(variant, trees) {
            Some(payload) => part_compared(cx, Asked::Equal, payload, a, b, span),
            None => answered(cx, Asked::Equal, span),
        };
        diamond(cx, same, bool_ty, span, payloads, |cx| cx.bool_const(false, span))
    })
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
fn array_compared(cx: &mut Lcx<'_, '_>, asked: Asked, array: ArrayOf, tree: &StructureTree, a: ValueId, b: ValueId, span: Span) -> ValueId {
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
    let tree = cx
        .typeck
        .structure_trees
        .get(&site)
        .unwrap_or_else(|| panic!("typeck records the tree of every lifted impl it reached, and not at {site:?}"));
    let declaration = cx.reached(site).declaration;
    let lift = cx
        .decls
        .lift(declaration)
        .unwrap_or_else(|| panic!("a call reached at a lifted impl names a function with a lift, not {declaration:?}"))
        .clone();
    let callee_ty = *cx
        .typeck
        .callees
        .get(&site)
        .unwrap_or_else(|| panic!("typeck records the callee type at every reached call site, and not at {site:?}"));
    let TyKind::Fn { params, ret, .. } = cx.kind(callee_ty).clone() else {
        panic!("a callee's type is a function type");
    };
    let structured = match lift.shape().params.first() {
        None => ret,
        Some(Taking::Value) => params[0],
        Some(Taking::Ref(_)) => match cx.kind(params[0]) {
            TyKind::Ref(_, target) => target.ty(),
            other => panic!("a parameter that takes the lift's variable by reference is a reference, not {other:?}"),
        },
    };
    match &lift {
        // `lift_of` makes a `Combine` only of `(&T, &T)`, whose answer is
        // the parts' equality taken together (types §1).
        Lift::Combine { .. } => {
            let [a, b] = args[..] else {
                panic!("a combined impl takes its two operands")
            };
            compared(cx, tree, Asked::Equal, structured, a, b, span)
        }
        Lift::Rebuild(shape) => rebuilt(cx, tree, shape, structured, &args, span),
    }
}

/// The value of `ty` a `Lift::Rebuild` impl answers at a node of tree
/// `tree`, taking `args` as `lift` takes them.
fn rebuilt(cx: &mut Lcx<'_, '_>, tree: &StructureTree, lift: &LiftShape, ty: Ty, args: &[ValueId], span: Span) -> ValueId {
    let trees = match tree {
        StructureTree::Reached { chosen, callee } => {
            let called = cx.callee(chosen);
            return issue(cx, called, *callee, args.to_vec(), span);
        }
        StructureTree::Parts(trees) => trees,
    };
    let shape = shape(cx, ty);
    if let Some(variants) = variants(cx, &shape) {
        // NOTE: an open question, returned to the coordinator. With one
        // argument its variant is the variant rebuilt; with none, or with
        // several that may hold different variants, no ledger sentence says
        // which variant the rebuilt value is.
        let [decides] = args[..] else {
            todo!("a variant type rebuilt from {} arguments: which variant it is is not determined", args.len())
        };
        return by_variant(cx, &variants, ty, decides, span, &|cx, variant| {
            let payload = payload_part(variant, trees).map(|part| {
                let part_args = part_args(cx, args, lift, part, span);
                rebuilt(cx, part.tree, lift, part.ty, &part_args, span)
            });
            cx.builder.push(Op::MakeVariant { tag: variant.tag, payload }, ty, span)
        });
    }
    match shape {
        Shape::Tuple(items) => {
            let elements = tuple_parts(&items, trees)
                .into_iter()
                .map(|part| {
                    let part_args = part_args(cx, args, lift, part, span);
                    rebuilt(cx, part.tree, lift, part.ty, &part_args, span)
                })
                .collect();
            cx.builder.push(Op::MakeTuple { elements }, ty, span)
        }
        Shape::Object(fields) => {
            let fields = field_parts(&fields, trees)
                .into_iter()
                .map(|part| {
                    let PathSeg::Field(name) = part.seg else {
                        unreachable!("an object's part is a field")
                    };
                    let part_args = part_args(cx, args, lift, part, span);
                    (name, rebuilt(cx, part.tree, lift, part.ty, &part_args, span))
                })
                .collect();
            cx.builder.push(Op::MakeObject { fields }, ty, span)
        }
        // NOTE: an open question, returned to the coordinator, as for an
        // array's coercion (`coerce::structured`): no IR form builds a new
        // array in one `For`, and the rule excludes unrolling.
        Shape::Array(_) => todo!("an array rebuilt element by element in one `For`: no IR form builds a new array in a loop"),
        Shape::Option(_) | Shape::Result { .. } | Shape::Enum(_) => unreachable!("a variant type is rebuilt by its variant above"),
        Shape::Ref(_) | Shape::Never => panic!("a lift answers only at a transparent structured type, not {ty:?}"),
    }
}

/// `part`'s arguments, from the node's `args` as `lift` takes each: a
/// reference to the part through a reference, or the part moved out of a
/// value.
fn part_args(cx: &mut Lcx<'_, '_>, args: &[ValueId], lift: &LiftShape, part: Part<'_>, span: Span) -> Vec<ValueId> {
    args.iter()
        .zip(lift.params.iter())
        .map(|(arg, taking)| match taking {
            Taking::Ref(mutability) => {
                let ty = cx.ref_ty(*mutability, part.ty);
                let op = Op::Ref { target: RefTarget::Through(*arg), path: vec![part.seg], mutability: *mutability };
                cx.builder.push(op, ty, span)
            }
            Taking::Value => {
                let op = match part.seg {
                    PathSeg::Index(index) => Op::TupleIndex { tuple: *arg, index },
                    PathSeg::Field(field) => Op::FieldGet { object: *arg, field, rest: Vec::new() },
                    PathSeg::Payload => Op::UnwrapVariant { src: *arg },
                };
                cx.builder.push(op, part.ty, span)
            }
        })
        .collect()
}
