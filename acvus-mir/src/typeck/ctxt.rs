use acvus_ast::{AstId, Span};
use acvus_utils::{Interner, QualifiedRef};

use super::decision::{Decision, Decisions, Pending};
use super::interface::Interfaces;
use super::output::Output;
use super::refusal;
use super::scope::{Run, Scope};
use crate::decl::Declarations;
use crate::infer::{Bound, Closed, Infer, Level, Term};
use crate::ty::{EffectTerm, Infer as InferPhase, IntTy, Mutability, Poly, TyId, TyKind};

/// The shared state of one checked body (S1): the declarations and the
/// units' interfaces it reads, and four things it builds. No rule holds
/// state of its own.
pub(super) struct Ctxt<'d> {
    pub interner: &'d Interner,
    pub decls: &'d Declarations,
    pub interfaces: &'d Interfaces,
    pub infer: Infer,
    pub scope: Scope,
    pub decisions: Decisions,
    pub out: Output,
}

impl<'d> Ctxt<'d> {
    pub fn new(interner: &'d Interner, decls: &'d Declarations, interfaces: &'d Interfaces, run: Run) -> Self {
        Self {
            interner,
            decls,
            interfaces,
            infer: Infer::new(),
            scope: Scope::new(run),
            decisions: Decisions::default(),
            out: Output::default(),
        }
    }

    /// `check` run as one instantiation of a body, a lambda's or a loop's
    /// iteration, one level inside the body around it (types §3).
    pub fn in_body<R>(&mut self, check: impl FnOnce(&mut Ctxt<'d>) -> R) -> R {
        let inside = Level(self.infer.level().0 + 1);
        self.at_level(inside, check)
    }

    /// `check` run at `level`: what it makes is made there (types §3).
    pub fn at_level<R>(&mut self, level: Level, check: impl FnOnce(&mut Ctxt<'d>) -> R) -> R {
        let around = self.infer.set_level(level);
        let checked = check(self);
        self.infer.set_level(around);
        checked
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
        self.decisions.push(Decision::Settle { span, term }, self.infer.level());
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

    /// B's fixed-members query, against the decisions still pending.
    pub fn fixed(&self, term: Term) -> bool {
        let pending = Pending {
            decisions: &self.decisions,
            infer: &self.infer,
        };
        self.infer.fixed(&pending, term)
    }

    /// `context`'s declared type `ty` at this body: every node `Opaque`
    /// (types §1), and each identity it holds the context's own
    /// (context §3).
    pub fn context_type(&mut self, context: QualifiedRef, ty: TyId<Poly>) -> Term {
        self.infer.instantiate_context(self.decls, context, ty)
    }

    /// A `&` or `&mut` of a value of term `target`, decided once its head
    /// is known: a value already a reference is reborrowed, since a
    /// reference's target is never a reference (types §4).
    pub fn lend(&mut self, target: Term, mutability: Mutability, span: Span) -> Term {
        self.lend_as(target, super::decision::Lent::Borrow(mutability), span)
    }

    /// `target` borrowed, bound or read as `lent` says, decided once its
    /// head is known.
    pub fn lend_as(&mut self, target: Term, lent: super::decision::Lent, span: Span) -> Term {
        let result = self.fresh();
        let lend = Decision::Lend {
            span,
            value: target,
            lent,
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
            TyKind::Ref(_, target) => *target,
            other => panic!("referent: a term closed as a reference is a reference, not {other:?}"),
        }
    }
}
