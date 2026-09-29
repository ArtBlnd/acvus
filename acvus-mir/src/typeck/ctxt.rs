use acvus_ast::{AstId, Span};

use super::decision::{Decision, Decisions, Pending};
use super::output::Output;
use super::refusal;
use super::scope::{Run, Scope};
use crate::decl::Declarations;
use crate::infer::{Bound, Closed, Infer, Term};
use crate::ty::{EffectTerm, Infer as InferPhase, IntTy, Mutability, Poly, TyId, TyKind};

/// The shared state of one checked body (S1): the declarations it reads,
/// and four things it builds. No rule holds state of its own.
pub(super) struct Ctxt<'d> {
    pub decls: &'d Declarations,
    pub infer: Infer,
    pub scope: Scope,
    pub decisions: Decisions,
    pub out: Output,
}

impl<'d> Ctxt<'d> {
    pub fn new(decls: &'d Declarations, run: Run) -> Self {
        Self {
            decls,
            infer: Infer::new(),
            scope: Scope::new(run),
            decisions: Decisions::default(),
            out: Output::default(),
        }
    }

    pub fn fresh(&mut self) -> Term {
        self.infer.fresh(Bound::Any)
    }

    pub fn node(&mut self, kind: TyKind<InferPhase>) -> Term {
        self.infer.intern(kind)
    }

    /// A node the program writes, behind a fresh variable, so that the
    /// variable is where it grows (D20).
    pub fn held(&mut self, kind: TyKind<InferPhase>) -> Term {
        let node = self.infer.intern(kind);
        let var = self.fresh();
        self.infer
            .eq(var, node)
            .expect("a fresh unbounded variable that no reference targets binds to any node");
        var
    }

    pub fn unit(&mut self) -> Term {
        self.node(TyKind::Unit)
    }

    pub fn never(&mut self) -> Term {
        self.node(TyKind::Never)
    }

    pub fn bool(&mut self) -> Term {
        self.node(TyKind::Bool)
    }

    pub fn record(&mut self, site: AstId, term: Term) {
        self.out.types.insert(site, term);
        if let Some(owner) = self.decisions.owner {
            self.out.site_owners.insert(site, owner);
        }
    }

    pub fn eq(&mut self, a: Term, b: Term, span: Span) {
        if let Err(mismatch) = self.infer.eq(a, b) {
            refusal::mismatch(self, mismatch, span);
        }
    }

    /// `a`, reached by `reach`, joined with `b`, the use's own form
    /// (types §1-2).
    pub fn join(&mut self, a: Term, b: Term, reach: crate::ty::Reach, span: Span) {
        if let Err(mismatch) = self.infer.join(a, b, reach) {
            refusal::mismatch(self, mismatch, span);
        }
    }

    /// `value` enters `slot`, reached by `reach`, once `value` is closed
    /// (types §1, §3, §7), in the body being checked. A value already
    /// closed at the top of its cast tree can never change, so it enters at
    /// once.
    pub fn enter(&mut self, site: AstId, value: Term, slot: Term, reach: crate::ty::Reach, span: Span) {
        let body = super::effect::body_effect(self);
        self.enter_in(body, site, value, slot, reach, span);
    }

    /// `value` enters `slot` in the body whose effect is `body`, which the
    /// coercion's function joins (effects §1): a decision firing after the
    /// walk names the body it was written in.
    pub fn enter_in(&mut self, body: EffectTerm<InferPhase>, site: AstId, value: Term, slot: Term, reach: crate::ty::Reach, span: Span) {
        let enter = Decision::Enter {
            site,
            span,
            value,
            slot,
            body,
            reach,
        };
        super::decision::decide(self, enter);
    }

    /// A range's end, an integer (control-flow). Its width is a use's to
    /// decide, and `i64` where none does, as a literal's (types §5).
    pub fn integer(&mut self, span: Span) -> Term {
        let term = self.infer.fresh(Bound::Integer {
            signed: true,
            among: IntTy::ALL.into(),
            literal: true,
        });
        self.decisions.push(Decision::Settle { span, term });
        term
    }

    /// B's closed-head query, against the decisions still pending.
    pub fn closed(&self, term: Term) -> Option<Closed> {
        let pending = Pending {
            decisions: &self.decisions,
            infer: &self.infer,
        };
        self.infer.closed(self.decls.casts(), &pending, term)
    }

    /// A declared type that is not a function's (a context's, an input's)
    /// at this body: every node `Opaque` (types §1).
    pub fn declared(&mut self, ty: TyId<Poly>) -> Term {
        self.infer.instantiate_type(self.decls, ty)
    }

    /// A `&` or `&mut` of a value of term `target`, decided once its head
    /// is known: a value already a reference is reborrowed, since a
    /// reference's target is never a reference (types §4).
    pub fn lend(&mut self, target: Term, mutability: Mutability, span: Span) -> Term {
        let result = self.fresh();
        let lend = Decision::Lend {
            span,
            value: target,
            mutability,
            result,
        };
        super::decision::decide(self, lend);
        result
    }

    /// The target of a term closed as a reference.
    ///
    /// # Panics
    /// `reference` is not a reference: callers ask `closed` first.
    pub fn referent(&self, reference: Term) -> Term {
        match self.infer.types().kind(self.infer.shallow(reference)) {
            TyKind::Ref(_, target) => target.ty(),
            other => panic!("referent: a term closed as a reference is a reference, not {other:?}"),
        }
    }
}
