//! `.clone()` (ownership §2, operators §7): the lang item `core::clone`,
//! which takes its receiver by `&` (operators §2). A structural type is
//! cloned part by part, a type that copies is copied (ownership §1), and
//! any other type is cloned through its `core::clone` impl. Each node of
//! the tree is decided once its type's head is known.

use acvus_ast::{AstId, Span};

use super::call::reach_in;
use super::ctxt::Ctxt;
use super::decision::{Callee, CoreSig, Decision};
use super::output::{Adjustment, CloneNode, PartNode};
use crate::decl::Head;
use crate::infer::{Closed, Term};
use crate::ty::structural::structure;
use crate::ty::{EffectTerm, Infer as InferPhase, Mutability, TyKind, copies_alone};

/// The receiver's adjustment by Rust's method probe (D85) against
/// `core::clone`'s receiver `&T`: a shared reference is taken as written,
/// and anything else is lent shared, a `&mut` by a shared reborrow.
pub(super) fn receiver_adjustment(head: Head) -> Adjustment {
    match head {
        Head::Ref(Mutability::Shared, _) => Adjustment::AsWritten,
        Head::Ref(Mutability::Mut, _) | Head::Base(_) => Adjustment::Ref(Mutability::Shared),
    }
}

/// What the adjusted receiver of head `head` and term `receiver` lends: the
/// type the clone is of.
pub(super) fn cloned_type(cx: &Ctxt<'_>, head: Head, receiver: Term) -> Term {
    match head {
        Head::Ref(..) => cx.referent(receiver),
        Head::Base(_) => receiver,
    }
}

/// The clone of a value of type `operand`, decided at `node`.
pub(super) fn decide(cx: &mut Ctxt<'_>, node: AstId, span: Span, operand: Term, body: EffectTerm<InferPhase>) {
    super::decision::decide(cx, Decision::Clone { node, span, operand, body });
}

/// A clone fired at `node` on its type: a structural type clones its parts,
/// each a node of its own, and `!` has none; a primitive or a shared
/// reference copies (ownership §1); any other type reaches `core::clone` at
/// `node` on a shared reference to it, and the impl's result is of the type
/// cloned.
pub(super) fn fire_clone(cx: &mut Ctxt<'_>, node: AstId, span: Span, operand: Closed, body: &EffectTerm<InferPhase>) {
    let types = cx.infer.types();
    let parts: Option<Vec<Term>> = match structure(types, operand.term) {
        Some(structure) => Some(structure.parts().to_vec()),
        None => match types.kind(operand.term) {
            TyKind::Never => Some(Vec::new()),
            _ => None,
        },
    };
    if let Some(parts) = parts {
        let nodes: Vec<AstId> = parts.iter().map(|_| AstId::alloc()).collect();
        let cloned = CloneNode { node: PartNode::Parts(nodes.clone().into()), ty: operand.term };
        cx.out.clones.insert(node, cloned);
        for (part, part_node) in parts.into_iter().zip(nodes) {
            decide(cx, part_node, span, part, body.clone());
        }
        return;
    }
    if copies_alone(types.kind(operand.term)) {
        cx.out.clones.insert(node, CloneNode { node: PartNode::Copied, ty: operand.term });
        return;
    }
    cx.out.clones.insert(node, CloneNode { node: PartNode::Reached, ty: operand.term });
    let lent = cx.lend(operand.term, Mutability::Shared, span);
    let cloned = reach_in(cx, body, node, Callee::Core(CoreSig::Clone), vec![(node, lent)], span);
    cx.eq(cloned, operand.term, span);
}
