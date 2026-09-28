//! Whether a term can still change (types §6): a decision reads only a
//! closed term.

use super::{End, Infer, Term};
use crate::decl::{Base, CastTree, Head};
use crate::ty::{ObjectTy, Phase, Poly, TyId, TyKind, Types};

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
        let head = self.head_of(node)?;
        (casts.step_up(head).is_none() || !open.may_change(term)).then_some(Closed { head, term: node })
    }

    /// The head a cast step is taken from at `node` (types §3): its
    /// constructor, and for a reference its target's too. `None` while a
    /// reference's target is an unbound variable.
    ///
    /// # Panics
    /// `node` is the order token or a started call, or a reference to one:
    /// only the compiler's IR writes those (effects §3, §6), so no term of
    /// typeck's is one.
    pub(super) fn head_of(&self, node: Term) -> Option<Head> {
        match self.types.kind(node) {
            TyKind::Ref(mutability, target) => {
                let End::Node { node: target, .. } = self.end(target.ty()) else {
                    return None;
                };
                Some(Head::Ref(*mutability, base_of(self.types.kind(target))))
            }
            kind => Some(Head::Base(base_of(kind))),
        }
    }
}

/// The head of a declared cast's end (types §3).
///
/// # Panics
/// `ty` has no head: verify admits no such cast (`CastWithoutHead`,
/// `ReferenceToReference`).
pub(super) fn poly_head(types: &Types<Poly>, ty: TyId<Poly>) -> Head {
    match types.kind(ty) {
        TyKind::Ref(mutability, target) => Head::Ref(*mutability, base_of(types.kind(target.ty()))),
        kind => Head::Base(base_of(kind)),
    }
}

fn base_of<V: Phase>(kind: &TyKind<V>) -> Base {
    match kind {
        TyKind::Never => Base::Never,
        TyKind::Int(width) => Base::Int(*width),
        TyKind::Float => Base::Float,
        TyKind::Char => Base::Char,
        TyKind::String => Base::String,
        TyKind::Bool => Base::Bool,
        TyKind::Unit => Base::Unit,
        TyKind::Str => Base::Str,
        TyKind::Array(..) => Base::Array,
        TyKind::Slice(_) => Base::Slice,
        TyKind::Tuple(items) => Base::Tuple(items.len()),
        TyKind::Option(_) => Base::Option,
        TyKind::Result(..) => Base::Result,
        TyKind::Object(ObjectTy::Structural(..)) => Base::Object,
        TyKind::Object(ObjectTy::Declared(id, _)) | TyKind::UserDefined { id, .. } => Base::Named(*id),
        TyKind::Enum { name, .. } => Base::Enum(*name),
        TyKind::Fn { .. } => Base::Fn,
        TyKind::Ref(..) => panic!(
            "a reference's target is never a reference (types §4): verify refuses a declared one, \
             Infer::intern refuses to build one, and a variable a reference targets is never bound \
             to one (Mismatch::ReferenceToReference)"
        ),
        TyKind::Order | TyKind::Handle(_) => panic!(
            "only the compiler's IR writes the order token and a started call (effects §3, §6); \
             verify refuses a declared one (InternalType)"
        ),
        TyKind::Var(_) => panic!(
            "a head is read off a node: typeck's variables are followed first, and verify admits no \
             cast whose end is a variable (CastWithoutHead)"
        ),
    }
}
