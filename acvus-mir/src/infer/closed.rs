//! Whether a term can still change (types §6): a decision reads only a
//! closed term.

use super::{End, Infer, Term};
use crate::decl::{CastTree, Head};
use crate::ty::{Infer as InferPhase, ObjectTy, TyKind};

/// What the decisions still pending may do: typeck answers from its
/// decision graph; B only asks.
pub trait OpenSources {
    /// Whether some pending decision outside the cycles through `term` may
    /// still enter a value into it or bind it.
    fn may_change(&self, term: Term) -> bool;
}

/// A closed term's head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Closed {
    pub head: Head,
    pub term: Term,
}

impl Infer {
    /// `term`'s head, when nothing can change it: its head is the top of its
    /// cast tree, or no open source may change it.
    pub fn closed(&self, casts: &CastTree, open: &impl OpenSources, term: Term) -> Option<Closed> {
        let End::Node { node, .. } = self.end(term) else {
            return None;
        };
        let head = head_of(self.types.kind(node));
        (casts.step_up(head).is_none() || !open.may_change(term)).then_some(Closed { head, term: node })
    }
}

/// The head a cast step is taken from: the constructor, a named type by
/// its name.
fn head_of(kind: &TyKind<InferPhase>) -> Head {
    match kind {
        TyKind::Int(width) => Head::Int(*width),
        TyKind::Float => Head::Float,
        TyKind::Char => Head::Char,
        TyKind::String => Head::String,
        TyKind::Bool => Head::Bool,
        TyKind::Unit => Head::Unit,
        TyKind::Str => Head::Str,
        TyKind::Array(..) => Head::Array,
        TyKind::Slice(_) => Head::Slice,
        TyKind::Tuple(items) => Head::Tuple(items.len()),
        TyKind::Option(_) => Head::Option,
        TyKind::Result(..) => Head::Result,
        TyKind::Object(ObjectTy::Structural(..)) => Head::Object,
        TyKind::Object(ObjectTy::Declared(id, _)) | TyKind::UserDefined { id, .. } => Head::Named(*id),
        TyKind::Enum { name, .. } => Head::Enum(*name),
        TyKind::Fn { .. } => Head::Fn,
        TyKind::Ref(mutability, _) => Head::Ref(*mutability),
        TyKind::Never | TyKind::Order | TyKind::Handle(_) => todo!(
            "stop: Head has no form for `!`, the order token or a started call, so whether their head \
             is closed cannot be answered"
        ),
        TyKind::Var(_) => unreachable!("head_of: a node is never a variable"),
    }
}
