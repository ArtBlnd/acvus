//! Whether a term can still change (types §6): a decision reads only a
//! closed term.

use super::{End, Infer, Term};
use crate::decl::{Base, CastTree, Head};
use super::cells::End as CellEnd;
use crate::ty::{Mark, MarkTerm, ObjectTy, Phase, Poly, TyId, TyKind, TyVarId, Types};

/// What the decisions still pending may do: typeck answers from its
/// decision graph; B only asks.
pub trait OpenSources {
    /// Whether some pending decision outside the cycles through `term` may
    /// still enter a value into it or bind it.
    fn may_change(&self, term: Term) -> bool;

    /// Whether some pending decision may still rebind a variable `term`'s
    /// members hang on: one that binds or enters into it, or a value
    /// entering a slot that may take it in, which links the value to the
    /// slot (types §1).
    fn may_grow(&self, term: Term) -> bool;
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

    /// Whether `term`'s members can no longer change: its chain ends at a
    /// node and no pending decision may still rebind a variable on it
    /// (types §1, §6). Closedness is of the head alone: a head at the top
    /// of its cast tree still gains fields or variants.
    pub fn fixed(&self, open: &impl OpenSources, term: Term) -> bool {
        matches!(self.end(term), End::Node { .. }) && !open.may_grow(term)
    }

    /// Whether `term` holds a mark cell not yet unmarked, as itself or a
    /// part but behind no reference and in no function type: a use may
    /// still raise it, and a mark is a member (types §6, §8).
    pub fn holds_raisable_mark(&self, term: Term) -> bool {
        acvus_utils::grow(|| {
            let End::Node { node, .. } = self.end(term) else {
                return false;
            };
            let kind = self.types.kind(node);
            let own = match kind {
                TyKind::Ref(..) | TyKind::Fn { .. } => return false,
                TyKind::UserDefined { marked: marked @ MarkTerm::Var(_), .. } => {
                    !matches!(self.mark_vars.end(marked), CellEnd::Var { value: Some(Mark::Unmarked), .. })
                }
                _ => false,
            };
            let mut parts = Vec::new();
            kind.for_each_child(|part| parts.push(part));
            own || parts.into_iter().any(|part| self.holds_raisable_mark(part))
        })
    }

    /// The variable a relation rebinds to change `term`'s head: the last
    /// on its chain. `None` for a term no variable holds, whose head no
    /// relation changes. Two terms linked through variables answer the
    /// same.
    pub(crate) fn last_variable(&self, term: Term) -> Option<TyVarId> {
        match self.end(term) {
            End::Unbound(var, _) | End::Node { holder: Some((var, _)), .. } => Some(var),
            End::Node { holder: None, .. } => None,
        }
    }

    /// Every variable whose rebinding changes `term`'s head: its own last
    /// variable and, for a reference, its target's, since a reference's
    /// head carries its target's constructor.
    pub(crate) fn head_variables(&self, term: Term) -> impl Iterator<Item = TyVarId> {
        let target = match self.end(term) {
            End::Node { node, .. } => match self.types.kind(node) {
                TyKind::Ref(_, target) => self.last_variable(*target),
                _ => None,
            },
            End::Unbound(..) => None,
        };
        self.last_variable(term).into_iter().chain(target)
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
                let End::Node { node: target, .. } = self.end(*target) else {
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
        TyKind::Ref(mutability, target) => Head::Ref(*mutability, base_of(types.kind(*target))),
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
        TyKind::Tuple(items, _) => Base::Tuple(items.len()),
        TyKind::Option(.., _) => Base::Option,
        TyKind::Result(..) => Base::Result,
        TyKind::Object(ObjectTy::Structural(..)) => Base::Object,
        TyKind::UserDefined { id, .. } => Base::Named(*id),
        TyKind::Enum { name, .. } => Base::Enum(*name),
        TyKind::Fn { .. } => Base::Fn,
        TyKind::Ref(..) => panic!(
            "a reference's target is never a reference (types §4): verify refuses a declared one, \
             Infer::intern refuses to build one, and a variable a reference targets is never bound \
             to one (Mismatch::ReferenceToReference)"
        ),
        TyKind::Order | TyKind::Handle(_) | TyKind::Closures(_) => panic!(
            "only the compiler's IR writes the order token and a started call (effects §3, §6); \
             verify refuses a declared one (InternalType)"
        ),
        TyKind::Var(_) => panic!(
            "a head is read off a node: typeck's variables are followed first, and verify admits no \
             cast whose end is a variable (CastWithoutHead)"
        ),
    }
}
