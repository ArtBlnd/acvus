//! Equality and ordering, and the impls a transparent structured type has
//! by a lift, by the structure tree typeck recorded (operators §4, types §1).
//! A tuple and an object are expanded part by part in `ty::structural`'s
//! order, and an enum, an `Option` and a `Result` variant by variant
//! (`Structure::variants`). An array is one `For` over its
//! elements, never unrolled: a comparison leaves it at the first element
//! whose answer settles the whole. A leaf is a call of the impl reached.
//!
//! A node's operand is a shared reference to its value, or, where the node
//! is itself a reference, that reference, since a reference's target is
//! never a reference (types §4).

use acvus_ast::{AstId, BinOp, Expr, Literal, Slot, Span};
use smallvec::SmallVec;

use super::call::{convert, issue, lent_operand};
use super::coerce::{ArrayMap, array_map, by_variant};
use super::control::diamond;
use super::ctxt::{Lcx, Lowered};
use super::place::{extract, view_array};
use crate::decl::{Lift, LiftShape, PairMember, Taking};
use crate::ir::cfg::{BlockId, Op, Terminator};
use crate::ir::{ForSource, IndexBound, IndexMode, PathSeg, RefTarget, UnaryOp, ValueId, fn_effect, known_len};
use crate::ir::parts::step;
use crate::ty::structural::{self, PartKey, Structure, Variant, VariantKey, structure};
use crate::ty::{Concrete, Effect, IntTy, Mutability, Ty, TyKind};
use crate::typeck::{Chosen, Own, StructureTree};

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
    /// A tuple or an object, by its fields (`Structure::fields`); the two
    /// are made by different operations.
    Tuple(Fields),
    Object(Fields),
    Array(ArrayOf),
    /// An enum, an `Option` or a `Result`, by its variants
    /// (`Structure::variants`).
    Variants(SmallVec<[Variant<Concrete>; 4]>),
    Ref(Ty),
    Never,
}

/// A product's fields, keyed.
type Fields = SmallVec<[structural::Part<Concrete>; 4]>;

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
    match structure(cx.types(), ty) {
        Some(structured @ Structure::Tuple(..)) => Shape::Tuple(structured.fields().expect("a tuple is a product")),
        Some(structured @ Structure::Object(..)) => Shape::Object(structured.fields().expect("an object is a product")),
        Some(Structure::Array { element, len, .. }) => Shape::Array(ArrayOf { element, len: known_len(&len) }),
        Some(structured @ (Structure::Enum(..) | Structure::Option(..) | Structure::Result { .. })) => {
            Shape::Variants(structured.variants().expect("an enum, an `Option` and a `Result` are variant types"))
        }
        None => match cx.kind(ty) {
            TyKind::Ref(_, target) => Shape::Ref(*target),
            TyKind::Never => Shape::Never,
            other => panic!("typeck records parts only at a structured type, a reference or `!`, not {other:?}"),
        },
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

/// The part of a node at `key`, of type `ty`, with its tree.
///
/// # Panics
/// `key` names an array's elements, which no one step names: an array's
/// part is reached by its elements' map or fold, never by a step.
fn part_at<'t>(key: PartKey, ty: Ty, trees: &'t [(PartKey, StructureTree)]) -> Part<'t> {
    let seg = step(key).unwrap_or_else(|| panic!("an array's elements are reached by a map or a fold, not by a step ({key:?})"));
    Part { seg, ty, tree: tree_at(trees, key) }
}

/// A product's parts, each with its tree.
fn field_parts<'t>(fields: &Fields, trees: &'t [(PartKey, StructureTree)]) -> Vec<Part<'t>> {
    fields.iter().map(|field| part_at(field.key, field.ty, trees)).collect()
}

/// The payload part of `variant`, where it has one.
fn payload_part<'t>(variant: &Variant<Concrete>, trees: &'t [(PartKey, StructureTree)]) -> Option<Part<'t>> {
    variant.payload.map(|ty| part_at(PartKey::Payload(variant.key), ty, trees))
}

/// The operand of `part` of the node whose operand is `base`, a shared
/// reference (operators §4): a reference part is read through it as a
/// shared reborrow, the least on its path (ownership §5).
fn part_operand(cx: &mut Lcx<'_, '_>, base: ValueId, part: Part<'_>, span: Span) -> Lowered<ValueId> {
    let target = RefTarget::Through(base);
    let path = vec![part.seg];
    match cx.kind(part.ty) {
        TyKind::Ref(made, referent) => {
            let (made, referent) = (*made, *referent);
            let taken = cx.emit(Op::Take { target, path, taken_out: false }, part.ty, span)?;
            Ok(super::place::reborrow(cx, taken, made, referent, Mutability::Shared, span))
        }
        _ => {
            let ty = cx.ref_ty(Mutability::Shared, part.ty);
            cx.emit(Op::Ref { target, path, mutability: Mutability::Shared }, ty, span)
        }
    }
}

/// The operand of element `at` of the shared slice `elements` of
/// `element`s: a reference element is read as a shared reborrow, the least
/// on its path (ownership §5).
fn element_operand(cx: &mut Lcx<'_, '_>, elements: ValueId, at: ValueId, element: Ty, span: Span) -> Lowered<ValueId> {
    match cx.kind(element) {
        TyKind::Ref(made, referent) => {
            let (made, referent) = (*made, *referent);
            let op = Op::Index { slice: elements, index: at, mode: IndexMode::Copy, bound: IndexBound::Checked };
            let taken = cx.emit(op, element, span)?;
            Ok(super::place::reborrow(cx, taken, made, referent, Mutability::Shared, span))
        }
        _ => {
            let ty = cx.ref_ty(Mutability::Shared, element);
            let op = Op::Index { slice: elements, index: at, mode: IndexMode::Ref, bound: IndexBound::Checked };
            cx.emit(op, ty, span)
        }
    }
}

// -- comparison --------------------------------------------------------------

/// `==` or an ordering by the tree typeck recorded at `site` (operators
/// §4); an ordering is `cmp`'s answer compared with `0` (operators §2).
pub(super) fn compare<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, op: BinOp, left: &Expr<S>, right: &Expr<S>, span: Span) -> Lowered<ValueId> {
    let tree = cx
        .decided
        .structure_tree(site)
        .unwrap_or_else(|| panic!("typeck records how every comparison compares, and not at {site:?}"));
    let asked = match op {
        BinOp::Eq | BinOp::Neq => Asked::Equal,
        _ => Asked::Order,
    };
    let compared_ty = cx.ty(left.id());
    let lhs = lent_operand(cx, left)?;
    let rhs = lent_operand(cx, right)?;
    let lhs = convert(cx, site, lhs, span)?;
    let rhs = convert(cx, site, rhs, span)?;
    let answer = compared(cx, tree, asked, compared_ty, lhs, rhs, span)?;
    let bool_ty = cx.ty(site);
    Ok(match op {
        BinOp::Eq => answer,
        BinOp::Neq => cx.made(Op::UnaryOp { op: UnaryOp::Not, operand: answer }, bool_ty, span),
        BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
            let answer_ty = cx.value_ty(answer);
            let zero = cx.constant(Literal::Int(0), answer_ty, span);
            cx.made(Op::BinOp { op: op.into(), left: answer, right: zero }, bool_ty, span)
        }
        other => panic!("{other:?} is not a comparison"),
    })
}

/// `a` against `b`, the operands of a node of type `ty`.
fn compared(cx: &mut Lcx<'_, '_>, tree: &StructureTree, asked: Asked, ty: Ty, a: ValueId, b: ValueId, span: Span) -> Lowered<ValueId> {
    let trees = match tree {
        StructureTree::Reached { chosen, callee, unmark, member } => {
            return reached_part(cx, chosen, *member, (*callee, *unmark), vec![a, b], span);
        }
        StructureTree::Parts(trees) => trees,
        StructureTree::Own(own) => unreachable!("the language answers only `clone` and `default` itself, no comparison ({own:?})"),
    };
    match shape(cx, ty) {
        // typeck compares a reference at its own node by what it names.
        Shape::Ref(target) => compared(cx, tree, asked, target, a, b, span),
        Shape::Tuple(fields) | Shape::Object(fields) => in_order(cx, asked, &field_parts(&fields, trees), a, b, span),
        Shape::Array(array) => array_compared(cx, asked, array, tree_at(trees, PartKey::Element), a, b, span),
        Shape::Variants(variants) => match asked {
            Asked::Equal => variant_equal(cx, &variants, trees, a, b, span),
            // typeck refuses ordering an enum, an `Option` or a `Result`,
            // which have no order (operators §4, `RefusalKind::Unordered`).
            Asked::Order => unreachable!("typeck refuses ordering a variant type (operators §4)"),
        },
        Shape::Never => unreachable!("the operands of a node of type `!` lend places no value is in, and no run gets past those lends"),
    }
}

/// Two values of a variant type equal: of one variant, and of equal
/// payloads where it has one (operators §4).
fn variant_equal(
    cx: &mut Lcx<'_, '_>,
    variants: &[Variant<Concrete>],
    trees: &[(PartKey, StructureTree)],
    a: ValueId,
    b: ValueId,
    span: Span,
) -> Lowered<ValueId> {
    let bool_ty = cx.bool_ty();
    by_variant(cx, variants, bool_ty, a, span, &|cx, variant| {
        let same = cx.made(Op::TestVariant { src: b, variant: variant.key }, bool_ty, span);
        let payloads = |cx: &mut Lcx<'_, '_>| match payload_part(variant, trees) {
            Some(payload) => part_compared(cx, Asked::Equal, payload, a, b, span),
            None => Ok(answered(cx, Asked::Equal, span)),
        };
        diamond(cx, same, bool_ty, span, payloads, |cx| Ok(cx.bool_const(false, span)))
    })
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
        Asked::Order => cx.intern(TyKind::Int(IntTy::I64)),
    }
}

fn order_const(cx: &mut Lcx<'_, '_>, value: i128, span: Span) -> ValueId {
    let i64_ty = cx.intern(TyKind::Int(IntTy::I64));
    cx.constant(Literal::Int(value), i64_ty, span)
}

/// Whether the answer so far leaves the comparison open: equal so far.
fn still_open(cx: &mut Lcx<'_, '_>, asked: Asked, answer: ValueId, span: Span) -> ValueId {
    match asked {
        Asked::Equal => answer,
        Asked::Order => {
            let zero = order_const(cx, 0, span);
            let bool_ty = cx.bool_ty();
            cx.made(Op::BinOp { op: crate::ir::BinOp::Eq, left: answer, right: zero }, bool_ty, span)
        }
    }
}

fn part_compared(cx: &mut Lcx<'_, '_>, asked: Asked, part: Part<'_>, a: ValueId, b: ValueId, span: Span) -> Lowered<ValueId> {
    let a = part_operand(cx, a, part, span)?;
    let b = part_operand(cx, b, part, span)?;
    compared(cx, part.tree, asked, part.ty, a, b, span)
}

/// Parts compared in order, as Rust's derived comparisons do: equal while
/// every part is, and ordered by the first part that is not equal. A part
/// is compared only where the parts before it left the answer open.
fn in_order(cx: &mut Lcx<'_, '_>, asked: Asked, parts: &[Part<'_>], a: ValueId, b: ValueId, span: Span) -> Lowered<ValueId> {
    let mut answer: Option<ValueId> = None;
    for part in parts {
        let next = |cx: &mut Lcx<'_, '_>| part_compared(cx, asked, *part, a, b, span);
        answer = Some(match answer {
            None => next(cx)?,
            Some(before) => {
                let open = still_open(cx, asked, before, span);
                let answer_ty = cx.value_ty(before);
                diamond(cx, open, answer_ty, span, next, |_| Ok(before))?
            }
        });
    }
    Ok(match answer {
        Some(answer) => answer,
        None => answered(cx, asked, span),
    })
}

/// Two arrays compared element by element in one `For` over their
/// indices, as Rust compares arrays: the loop leaves at the first element
/// whose answer is not equal, with that answer, and answers equal where it
/// runs to its end.
fn array_compared(cx: &mut Lcx<'_, '_>, asked: Asked, array: ArrayOf, tree: &StructureTree, a: ValueId, b: ValueId, span: Span) -> Lowered<ValueId> {
    let ArrayOf { element, len } = array;
    let a_elements = view_array(cx, a, Mutability::Shared, element, span);
    let b_elements = view_array(cx, b, Mutability::Shared, element, span);
    let all_equal = answered(cx, asked, span);
    let answer_ty = answer_ty(cx, asked);
    indices(cx, len, answer_ty, all_equal, span, |cx, at, header, exit| {
        let x = element_operand(cx, a_elements, at, element, span)?;
        let y = element_operand(cx, b_elements, at, element, span)?;
        let answer = compared(cx, tree, asked, element, x, y, span)?;
        let open = still_open(cx, asked, answer, span);
        Ok(Terminator::JumpIf { cond: open, then_to: header, then_args: Vec::new(), else_to: exit, else_args: vec![answer] })
    })
}

/// A `For` over the indices `0..len`, a `u64` each (operators §3). `body`
/// lowers one iteration from its index and ends it with the terminator it
/// yields, given the loop's header and exit, where a run gets past it; the
/// exit takes one value of `exit_ty`, `at_end` where the loop runs to its
/// end. The value the exit takes is the result.
fn indices(
    cx: &mut Lcx<'_, '_>,
    len: usize,
    exit_ty: Ty,
    at_end: ValueId,
    span: Span,
    body: impl FnOnce(&mut Lcx<'_, '_>, ValueId, BlockId, BlockId) -> Lowered<Terminator>,
) -> Lowered<ValueId> {
    let u64_ty = cx.u64_ty();
    let len = u64::try_from(len).expect("an array's length fits its `u64` index (operators §3)");
    let from = cx.constant(Literal::Int(0), u64_ty, span);
    let to = cx.constant(Literal::Int(i128::from(len)), u64_ty, span);
    let header = cx.new_block(&[]);
    cx.jump(header, Vec::new(), span);
    cx.enter_block(header, span)?;
    let body_block = cx.new_block(&[u64_ty]);
    let exit = cx.new_block(&[exit_ty]);
    let terminator = Terminator::For {
        source: ForSource::Range { at: from, hi: to },
        body: body_block,
        body_args: Vec::new(),
        exit,
        exit_args: vec![at_end],
    };
    cx.terminate(terminator, span);
    if let Ok(end) = cx.enter_block(body_block, span).and_then(|params| body(cx, params[0], header, exit)) {
        cx.terminate(end, span);
    }
    Ok(cx.enter_block(exit, span)?[0])
}

// -- lifted impl -------------------------------------------------------------

/// A call typeck answered by a lifted impl at `site` (types §1,
/// `Impl::Structural`), of `args`: by the lift's shape, each part's
/// arguments are the arguments' parts, each part's impl is called at its
/// node, and the results rebuild the structure (`Lift::Rebuild`) or are
/// combined as the lift's law defines (`Lift::Combine`). An array's parts
/// are one `For` over its elements, never unrolled.
pub(super) fn lifted(cx: &mut Lcx<'_, '_>, site: AstId, args: Vec<ValueId>, span: Span) -> Lowered<ValueId> {
    let tree = cx
        .decided
        .structure_tree(site)
        .unwrap_or_else(|| panic!("typeck records the tree of every lifted impl it reached, and not at {site:?}"));
    if let StructureTree::Own(Own::Lend(mutability)) = tree {
        let [lent] = args[..] else {
            panic!("`as_ref` and `as_mut` take one reference");
        };
        let made = match cx.decided.callee(site).map(|callee| cx.kind(callee)) {
            Some(TyKind::Fn { ret, .. }) => *ret,
            other => panic!("typeck records a lend's function type at its site, not {other:?}"),
        };
        return super::coerce::lend(cx, *mutability, lent, made, span);
    }
    let declaration = cx.reached(site).declaration;
    let lift = cx
        .decls
        .lift(declaration)
        .unwrap_or_else(|| panic!("a call reached at a lifted impl names a function with a lift, not {declaration:?}"))
        .clone();
    let callee_ty = cx
        .decided
        .callee(site)
        .unwrap_or_else(|| panic!("typeck records the callee type at every reached call site, and not at {site:?}"));
    let TyKind::Fn { params, ret, .. } = cx.kind(callee_ty).clone() else {
        panic!("a callee's type is a function type");
    };
    let structured = match lift.shape().params.first() {
        None => ret,
        Some(Taking::Value) => params[0],
        Some(Taking::Ref(_)) => match cx.kind(params[0]) {
            TyKind::Ref(_, target) => *target,
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
fn rebuilt(cx: &mut Lcx<'_, '_>, tree: &StructureTree, lift: &LiftShape, ty: Ty, args: &[ValueId], span: Span) -> Lowered<ValueId> {
    let trees = match tree {
        StructureTree::Reached { chosen, callee, unmark, member } => {
            return reached_part(cx, chosen, *member, (*callee, *unmark), args.to_vec(), span);
        }
        StructureTree::Parts(trees) => trees,
        StructureTree::Own(Own::Copy) => {
            return match lift.params.first() {
                Some(Taking::Ref(_)) => {
                    let op = Op::Take { target: RefTarget::Through(args[0]), path: Vec::new(), taken_out: false };
                    cx.emit(op, ty, span)
                }
                Some(Taking::Value) => Ok(args[0]),
                None => unreachable!("`clone` takes the value it copies"),
            };
        }
        StructureTree::Own(Own::Lend(_)) => unreachable!("typeck answers a lend only at its own call, never at a part"),
        StructureTree::Own(Own::OptionNone) => {
            return Ok(cx.made(Op::MakeVariant { variant: VariantKey::None, payload: None }, ty, span));
        }
    };
    match shape(cx, ty) {
        // typeck lifts a rebuild at a variant type only where the function
        // takes it in exactly one parameter, whose variant is the result's
        // (types §1, `structure::lifts_at`).
        Shape::Variants(variants) => {
            let [decides] = args[..] else {
                unreachable!("typeck rebuilds a variant type only from one argument, not {}", args.len())
            };
            by_variant(cx, &variants, ty, decides, span, &|cx, variant| {
                let payload = payload_part(variant, trees)
                    .map(|part| {
                        let part_args = part_args(cx, args, lift, part, span)?;
                        rebuilt(cx, part.tree, lift, part.ty, &part_args, span)
                    })
                    .transpose()?;
                Ok(cx.made(Op::MakeVariant { variant: variant.key, payload }, ty, span))
            })
        }
        Shape::Tuple(fields) => {
            let elements = field_parts(&fields, trees)
                .into_iter()
                .map(|part| {
                    let part_args = part_args(cx, args, lift, part, span)?;
                    rebuilt(cx, part.tree, lift, part.ty, &part_args, span)
                })
                .collect::<Lowered<_>>()?;
            Ok(cx.made(Op::MakeTuple { elements }, ty, span))
        }
        Shape::Object(fields) => {
            let fields = field_parts(&fields, trees)
                .into_iter()
                .map(|part| {
                    let PathSeg::Field(name) = part.seg else {
                        unreachable!("an object's part is a field")
                    };
                    let part_args = part_args(cx, args, lift, part, span)?;
                    Ok((name, rebuilt(cx, part.tree, lift, part.ty, &part_args, span)?))
                })
                .collect::<Lowered<_>>()?;
            Ok(cx.made(Op::MakeObject { fields }, ty, span))
        }
        // Each argument's elements are mapped together, one index at a
        // time; with no argument the length is the array type's.
        Shape::Array(ArrayOf { element, len }) => {
            let tree = tree_at(trees, PartKey::Element);
            let params = lift
                .params
                .iter()
                .map(|taking| match taking {
                    Taking::Value => element,
                    Taking::Ref(mutability) => super::coerce::part_reference(cx, *mutability, element),
                })
                .collect();
            let effect = tree_effect(cx, tree);
            let map = ArrayMap { arrays: args.to_vec(), params, mapped: element, len, effect, span };
            array_map(cx, map, &|cx, at| rebuilt(cx, tree, lift, element, at, span))
        }
        Shape::Ref(_) | Shape::Never => panic!("a lift answers only at a transparent structured type, not {ty:?}"),
    }
}

/// The call of the impl a part node reached, as `member` of its `#` pair,
/// its value taken at `unmark` where its uses settled it so (types §8).
fn reached_part(
    cx: &mut Lcx<'_, '_>,
    chosen: &Chosen,
    member: PairMember,
    (callee, unmark): (Ty, Option<Ty>),
    args: Vec<ValueId>,
    span: Span,
) -> Lowered<ValueId> {
    let called = cx.callee(chosen, member);
    let given = issue(cx, called, callee, args, span)?;
    match unmark {
        Some(taken) => cx.emit(Op::Unmark { src: given }, taken, span),
        None => Ok(given),
    }
}

/// The effect of the calls `tree` makes: every impl's it reaches, joined.
fn tree_effect(cx: &Lcx<'_, '_>, tree: &StructureTree) -> Effect {
    match tree {
        StructureTree::Reached { callee, .. } => {
            fn_effect(cx.types(), *callee).expect("a reached impl's type has an effect").clone()
        }
        StructureTree::Parts(parts) => parts.iter().fold(Effect::pure(), |joined, (_, part)| joined.join(&tree_effect(cx, part))),
        StructureTree::Own(_) => Effect::pure(),
    }
}

/// `part`'s arguments, from the node's `args` as `lift` takes each: a
/// reference to the part through a reference, or the part moved out of a
/// value (`place::extract`).
fn part_args(cx: &mut Lcx<'_, '_>, args: &[ValueId], lift: &LiftShape, part: Part<'_>, span: Span) -> Lowered<Vec<ValueId>> {
    args.iter()
        .zip(lift.params.iter())
        .map(|(arg, taking)| match taking {
            Taking::Ref(mutability) => {
                let ty = cx.ref_ty(*mutability, part.ty);
                let op = Op::Ref { target: RefTarget::Through(*arg), path: vec![part.seg], mutability: *mutability };
                cx.emit(op, ty, span)
            }
            Taking::Value => extract(cx, *arg, part.seg, span),
        })
        .collect()
}
