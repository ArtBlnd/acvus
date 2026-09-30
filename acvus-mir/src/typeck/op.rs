//! Operators: each a call of its lang item through `call`
//! (operators §2); `&&`, `||`, `!`, `^`, `&`, `<<` and `>>` are the
//! language's own (operators §2).

use acvus_ast::{AstId, BinOp, Expr, Slot, Span, UnaryOp};

use super::call::{reach_call, reach_in};
use super::ctxt::Ctxt;
use super::decision::{Callee, CoreSig, Decision, Domain};
use super::output::StructureNode;
use super::expr::check_expr;
use super::place::{self, Demand};
use super::refusal::{self, RefusalKind};
use crate::decl::{Base, Head};
use crate::infer::{Closed, Term};
use crate::ty::structural::{PartKey, Structure, structure};
use crate::ty::{EffectTerm, Infer as InferPhase, Mutability, TyKind};

/// `==` and the orderings compare part by part where the operands' type is
/// structural, and through `core::eq` or `core::cmp` otherwise
/// (operators §4): a `Compare` decision on the operands' one type.
pub(super) fn check_binary<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, left: &Expr<S>, op: BinOp, right: &Expr<S>, span: Span) -> Term {
    // A lang item takes its operands by `&` (operators §2), so they are
    // lent, not used whole.
    let demand = match op {
        BinOp::Add
        | BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::Mod
        | BinOp::Eq
        | BinOp::Neq
        | BinOp::Lt
        | BinOp::Gt
        | BinOp::Lte
        | BinOp::Gte => Demand::Lend(Mutability::Shared),
        BinOp::And | BinOp::Or | BinOp::Xor | BinOp::BitAnd | BinOp::Shl | BinOp::Shr => Demand::Read,
    };
    let lhs = check_expr(cx, left, demand);
    let rhs = check_expr(cx, right, demand);
    let operands = Operands { left, lhs, right, rhs };
    match op {
        BinOp::Add => lang_item(cx, site, CoreSig::Add, operands, span),
        BinOp::Sub => lang_item(cx, site, CoreSig::Sub, operands, span),
        BinOp::Mul => lang_item(cx, site, CoreSig::Mul, operands, span),
        BinOp::Div => lang_item(cx, site, CoreSig::Div, operands, span),
        BinOp::Mod => lang_item(cx, site, CoreSig::Rem, operands, span),
        BinOp::Eq | BinOp::Neq => compare(cx, site, CoreSig::Eq, operands, span),
        // An ordering is `core::cmp`'s answer compared with `0`: the
        // operator is a `bool` (operators §2).
        BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => compare(cx, site, CoreSig::Cmp, operands, span),
        BinOp::And | BinOp::Or => {
            let bool_ty = cx.bool();
            cx.eq(lhs, bool_ty, left.span());
            cx.eq(rhs, bool_ty, right.span());
            bool_ty
        }
        BinOp::Xor => {
            cx.eq(lhs, rhs, span);
            operand(cx, lhs, Domain::BoolOrInteger, span);
            lhs
        }
        BinOp::BitAnd => {
            cx.eq(lhs, rhs, span);
            operand(cx, lhs, Domain::Integer, span);
            lhs
        }
        BinOp::Shl | BinOp::Shr => {
            operand(cx, lhs, Domain::Integer, left.span());
            operand(cx, rhs, Domain::Integer, right.span());
            lhs
        }
    }
}

/// An operand of the language's own operator, decided in `domain` once
/// its type is known (operators §2).
fn operand(cx: &mut Ctxt<'_>, operand: Term, domain: Domain, span: Span) {
    super::decision::decide(cx, Decision::Operand { span, operand, domain });
}

struct Operands<'e, S> {
    left: &'e Expr<S>,
    lhs: Term,
    right: &'e Expr<S>,
    rhs: Term,
}

/// A binary operator as a call of its lang item on shared references to
/// its operands (operators §2).
fn lang_item<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, sig: CoreSig, operands: Operands<'_, S>, span: Span) -> Term {
    let args = vec![
        (operands.left.id(), cx.lend(operands.lhs, Mutability::Shared, operands.left.span())),
        (operands.right.id(), cx.lend(operands.rhs, Mutability::Shared, operands.right.span())),
    ];
    reach_call(cx, site, Callee::Core(sig), args, span)
}

/// A comparison: the lang items take both operands at one type
/// (operators §2), and how that type compares is decided once it is
/// known.
fn compare<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, sig: CoreSig, operands: Operands<'_, S>, span: Span) -> Term {
    cx.eq(operands.lhs, operands.rhs, operands.right.span());
    let body = super::effect::body_effect(cx);
    let compare = Decision::Compare {
        node: site,
        span,
        sig,
        operand: operands.lhs,
        body,
    };
    super::decision::decide(cx, compare);
    cx.bool()
}

/// A comparison fired at `node` on its operands' type (operators §4): a
/// transparent structured type compares its parts, each a node of its own;
/// a reference compares what it names at the same node, since a lang item's
/// `&T` is never a reference to a reference (types §4); any other type
/// reaches `core::eq` or `core::cmp` at `node`.
pub(super) fn fire_compare(
    cx: &mut Ctxt<'_>,
    node: AstId,
    span: Span,
    sig: CoreSig,
    operand: Closed,
    body: &EffectTerm<InferPhase>,
) {
    let types = cx.infer.types();
    let sum = matches!(structure(types, operand.term), Some(Structure::Enum(..) | Structure::Option(..) | Structure::Result { .. }));
    if sig == CoreSig::Cmp && sum {
        refusal::push(cx, RefusalKind::Unordered, span);
        return;
    }
    let parts: Option<Vec<(PartKey, Term)>> = match structure(types, operand.term) {
        Some(structure) => Some(structure.parts().iter().map(|part| (part.key, part.ty)).collect()),
        None => match types.kind(operand.term) {
            TyKind::Ref(_, target) => {
                let compare = Decision::Compare { node, span, sig, operand: *target, body: body.clone() };
                super::decision::decide(cx, compare);
                return;
            }
            TyKind::Never => Some(Vec::new()),
            _ => None,
        },
    };
    let Some(parts) = parts else {
        cx.out.structures.insert(node, StructureNode::Reached);
        let lent = cx.lend(operand.term, Mutability::Shared, span);
        reach_in(cx, body, node, Callee::Core(sig), vec![(node, lent), (node, lent)], span);
        return;
    };
    let nodes: Vec<(PartKey, AstId)> = parts.iter().map(|(key, _)| (*key, AstId::alloc())).collect();
    cx.out.structures.insert(node, StructureNode::Parts(nodes.clone().into()));
    for ((_, part), (_, part_node)) in parts.into_iter().zip(nodes) {
        let compare = Decision::Compare {
            node: part_node,
            span,
            sig,
            operand: part,
            body: body.clone(),
        };
        super::decision::decide(cx, compare);
    }
}

/// An operand of the language's own operator fired on its type
/// (operators §2): refused outside `domain`. A `!` operand is never read.
pub(super) fn fire_operand(cx: &mut Ctxt<'_>, span: Span, operand: Closed, domain: Domain) {
    let admitted = match operand.head {
        Head::Base(Base::Never) => true,
        Head::Base(Base::Bool) => domain != Domain::Integer,
        Head::Base(Base::Int(_)) => domain != Domain::Bool,
        Head::Base(_) | Head::Ref(..) => false,
    };
    if !admitted {
        let kind = match domain {
            Domain::Bool => RefusalKind::NotBool,
            Domain::Integer => RefusalKind::NotInteger,
            Domain::BoolOrInteger => RefusalKind::NotBoolOrInteger,
        };
        refusal::push(cx, kind, span);
    }
}

/// `-x`, `!x`, which is on `bool` alone (operators §2). `*r` is
/// `place::check_deref`.
pub(super) fn check_unary<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, op: UnaryOp, operand: &Expr<S>, span: Span) -> Term {
    match op {
        UnaryOp::Neg => {
            let value = check_expr(cx, operand, Demand::Lend(Mutability::Shared));
            let arg = cx.lend(value, Mutability::Shared, operand.span());
            reach_call(cx, site, Callee::Core(CoreSig::Neg), vec![(operand.id(), arg)], span)
        }
        UnaryOp::Not => {
            let value = check_expr(cx, operand, Demand::Read);
            self::operand(cx, value, Domain::Bool, operand.span());
            value
        }
        UnaryOp::Deref => place::check_deref(cx, site, operand, span),
    }
}
