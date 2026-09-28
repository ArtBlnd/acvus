//! Bindings, and what a name reaches (types §6, programs §4).

use acvus_ast::{AstId, RefKind, Span};
use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::FxHashMap;

use super::ctxt::Ctxt;
use crate::infer::Term;

/// One binding: its binder's site and its term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Binding {
    pub binder: AstId,
    pub term: Term,
}

/// One frame of bindings; a lambda's body opens one of its own.
struct Frame {
    names: FxHashMap<Astr, Binding>,
    lambda: Option<AstId>,
}

/// A binding found, and the lambdas the lookup crossed on its way out: the
/// name is a capture of each of them (functions §2).
pub(super) struct Found {
    pub binding: Binding,
    pub crossed: Vec<AstId>,
}

/// The bindings in scope, innermost last.
pub(super) struct Scope {
    frames: Vec<Frame>,
}

impl Scope {
    pub fn new() -> Self {
        Self { frames: vec![Frame { names: FxHashMap::default(), lambda: None }] }
    }

    /// A block's frame.
    pub fn push(&mut self) {
        self.frames.push(Frame { names: FxHashMap::default(), lambda: None });
    }

    /// A lambda body's frame.
    pub fn push_lambda(&mut self, lambda: AstId) {
        self.frames.push(Frame { names: FxHashMap::default(), lambda: Some(lambda) });
    }

    /// # Panics
    /// Only the frames pushed are popped: the body's frame never is.
    pub fn pop(&mut self) {
        assert!(self.frames.len() > 1, "Scope::pop: the body's frame is never popped");
        self.frames.pop();
    }

    pub fn define(&mut self, name: Astr, binding: Binding) {
        self.frames
            .last_mut()
            .expect("Scope::define: the body's frame is never popped")
            .names
            .insert(name, binding);
    }

    pub fn lookup(&self, name: Astr) -> Option<Found> {
        let mut crossed = Vec::new();
        for frame in self.frames.iter().rev() {
            if let Some(binding) = frame.names.get(&name) {
                return Some(Found { binding: *binding, crossed });
            }
            if let Some(lambda) = frame.lambda {
                crossed.push(lambda);
            }
        }
        None
    }
}

/// `name`, `$name`: a binding, a declaration or an input.
pub(super) fn check_ident(
    cx: &mut Ctxt<'_>,
    site: AstId,
    name: QualifiedRef,
    ref_kind: RefKind,
    span: Span,
) -> Term {
    todo!("typeck::scope::check_ident")
}

/// `x = v`: a binding in scope; an input is never assigned (programs §4).
pub(super) fn check_assign(cx: &mut Ctxt<'_>, site: AstId, name: Astr, value: Term, span: Span) {
    todo!("typeck::scope::check_assign")
}
