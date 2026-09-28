//! Bindings, and what a name reaches (types §6, programs §4).

use acvus_ast::{AstId, RefKind, Span};
use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::FxHashMap;

use super::ctxt::Ctxt;
use super::decision::Decision;
use super::output::{Capture, Captured, Name};
use super::place::Demand;
use super::refusal::{self, RefusalKind};
use crate::infer::Term;
use crate::ty::{EffectTerm, Infer as InferPhase, Mutability};

/// One binding: its binder's site and its term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Binding {
    pub binder: AstId,
    pub term: Term,
}

/// What a body's `return` and `?` leave through, and what its calls do:
/// a lambda's own, or the run's.
struct Body {
    lambda: AstId,
    captures_by_move: bool,
    result: Term,
    effect: Option<EffectTerm<InferPhase>>,
}

/// One frame of bindings; a lambda's body opens one of its own, and so
/// does a loop's.
struct Frame {
    names: FxHashMap<Astr, Binding>,
    body: Option<Body>,
    is_loop: bool,
}

/// A binding found, and the lambdas the lookup crossed on its way out: the
/// name is a capture of each of them (functions §2).
pub(super) struct Found {
    pub binding: Binding,
    pub crossed: Vec<AstId>,
}

/// The bindings in scope, innermost last, and the run's own result slot
/// once a `return`, a `?` or the tail names it.
pub(super) struct Scope {
    frames: Vec<Frame>,
    run_result: Option<Term>,
}

impl Scope {
    pub fn new() -> Self {
        Self {
            frames: vec![Frame {
                names: FxHashMap::default(),
                body: None,
                is_loop: false,
            }],
            run_result: None,
        }
    }

    /// A block's frame.
    pub fn push(&mut self) {
        self.frames.push(Frame {
            names: FxHashMap::default(),
            body: None,
            is_loop: false,
        });
    }

    /// A loop body's frame, which a `break` or a `continue` leaves.
    pub fn push_loop(&mut self) {
        self.frames.push(Frame {
            names: FxHashMap::default(),
            body: None,
            is_loop: true,
        });
    }

    /// A lambda body's frame, leaving through `result`.
    pub fn push_lambda(&mut self, lambda: AstId, captures_by_move: bool, result: Term) {
        self.frames.push(Frame {
            names: FxHashMap::default(),
            body: Some(Body {
                lambda,
                captures_by_move,
                result,
                effect: None,
            }),
            is_loop: false,
        });
    }

    /// Whether a `break` or a `continue` here leaves a loop: a loop's
    /// frame is found before a lambda's boundary, since a lambda's `break`
    /// never leaves a loop outside it.
    pub fn in_loop(&self) -> bool {
        for frame in self.frames.iter().rev() {
            if frame.is_loop {
                return true;
            }
            if frame.body.is_some() {
                return false;
            }
        }
        false
    }

    /// Whether the lambda `lambda`, whose body is in scope, is written with
    /// `move` (functions §2).
    ///
    /// # Panics
    /// `lambda`'s body is not in scope: a lookup names only the lambdas it
    /// crosses.
    pub fn captures_by_move(&self, lambda: AstId) -> bool {
        self.frames
            .iter()
            .filter_map(|frame| frame.body.as_ref())
            .find(|body| body.lambda == lambda)
            .map(|body| body.captures_by_move)
            .expect("a crossed lambda's body is in scope")
    }

    /// # Panics
    /// Only the frames pushed are popped: the body's frame never is.
    pub fn pop(&mut self) {
        assert!(self.frames.len() > 1, "Scope::pop: the body's frame is never popped");
        self.frames.pop();
    }

    /// Pops a lambda body's frame, handing back the join of its calls'
    /// effects, `None` where it made none.
    ///
    /// # Panics
    /// The innermost frame is not a lambda body's: lambdas pop their own.
    pub fn pop_lambda(&mut self) -> Option<EffectTerm<InferPhase>> {
        assert!(self.frames.len() > 1, "Scope::pop_lambda: the body's frame is never popped");
        let frame = self.frames.pop().expect("Scope::pop_lambda: a frame is pushed");
        frame
            .body
            .expect("Scope::pop_lambda: the innermost frame is a lambda body's")
            .effect
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
                return Some(Found {
                    binding: *binding,
                    crossed,
                });
            }
            if let Some(body) = &frame.body {
                crossed.push(body.lambda);
            }
        }
        None
    }

    /// The slot the innermost body leaves through: its lambda's result, or
    /// the run's once something named it.
    pub fn body_result(&self) -> Option<Term> {
        match self.frames.iter().rev().find_map(|frame| frame.body.as_ref()) {
            Some(body) => Some(body.result),
            None => self.run_result,
        }
    }

    /// Names the run's result slot.
    ///
    /// # Panics
    /// The run's result is named already, or a lambda's body is innermost.
    pub fn set_run_result(&mut self, result: Term) {
        assert!(self.body_result().is_none(), "Scope::set_run_result: the run's result is named once, outside every lambda");
        self.run_result = Some(result);
    }

    /// The run's result slot, once something named it.
    pub fn run_result(&self) -> Option<Term> {
        self.run_result
    }

    /// Where the innermost lambda's calls' effects join; `None` outside
    /// every lambda, where they join the run's (`Output::body_effect`).
    pub fn lambda_effect(&mut self) -> Option<&mut Option<EffectTerm<InferPhase>>> {
        self.frames
            .iter_mut()
            .rev()
            .find_map(|frame| frame.body.as_mut())
            .map(|body| &mut body.effect)
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
    if ref_kind == RefKind::ExternParam {
        return check_input(cx, site, name.name, span);
    }
    if name.namespace.is_none()
        && let Some(found) = cx.scope.lookup(name.name)
    {
        cx.out.names.insert(site, Name::Binding(found.binding.binder));
        note_captures(cx, name.name, &found, Demand::Read, span);
        return found.binding.term;
    }
    let declarations: Box<[QualifiedRef]> = match name.namespace {
        Some(_) => cx.decls.scheme(name).map(|_| name).into_iter().collect(),
        None => cx.decls.overloads(name.name).into(),
    };
    match &declarations[..] {
        [declaration] => super::call::declaration_value(cx, site, *declaration, span),
        [] => {
            refusal::push(cx, RefusalKind::UnknownName { name: name.name }, span);
            cx.fresh()
        }
        // A name used as a value has no argument to decide among its
        // declarations by (types §6).
        _ => {
            let kind = RefusalKind::SeveralDeclarations {
                name: name.name,
                left: declarations,
            };
            refusal::push(cx, kind, span);
            cx.fresh()
        }
    }
}

/// `$name`: the value the host passes, at its declared type (programs §4).
fn check_input(cx: &mut Ctxt<'_>, site: AstId, name: Astr, span: Span) -> Term {
    match cx.decls.input(name) {
        Some(ty) => {
            cx.out.names.insert(site, Name::Input(name));
            cx.declared(ty)
        }
        None => {
            refusal::push(cx, RefusalKind::UnknownName { name }, span);
            cx.fresh()
        }
    }
}

/// `name`, found crossing lambdas, is a capture of each of them
/// (functions §2): by move in a lambda written with `move`, whose holds
/// then wait for the captured value's type; otherwise by reference, and
/// mutably where it is written or lent mutably.
pub(super) fn note_captures(cx: &mut Ctxt<'_>, name: Astr, found: &Found, demand: Demand, span: Span) {
    let wanted = match demand {
        Demand::Read | Demand::Lend(Mutability::Shared) => Mutability::Shared,
        Demand::Write | Demand::Lend(Mutability::Mut) => Mutability::Mut,
    };
    for lambda in &found.crossed {
        let by_move = cx.scope.captures_by_move(*lambda);
        let mut captures = cx
            .out
            .closures
            .get(lambda)
            .expect("a lambda records its closure before its body is checked")
            .captures
            .to_vec();
        let binder = found.binding.binder;
        match captures.iter_mut().find(|captured| captured.binder == binder) {
            Some(Captured {
                capture: Capture::Ref(mutability),
                ..
            }) => {
                if wanted == Mutability::Mut {
                    *mutability = Mutability::Mut;
                }
            }
            Some(Captured {
                capture: Capture::Move, ..
            }) => {}
            None if by_move => {
                captures.push(Captured {
                    name,
                    binder,
                    capture: Capture::Move,
                });
                cx.decisions.push(Decision::MovedCapture {
                    lambda: *lambda,
                    span,
                    term: found.binding.term,
                });
            }
            None => captures.push(Captured {
                name,
                binder,
                capture: Capture::Ref(wanted),
            }),
        }
        let closure = cx
            .out
            .closures
            .get_mut(lambda)
            .expect("a lambda records its closure before its body is checked");
        closure.captures = captures.into();
    }
}

/// `x = v`: a binding in scope; an input is never assigned (programs §4).
pub(super) fn check_assign(cx: &mut Ctxt<'_>, site: AstId, name: Astr, value: Term, span: Span) {
    let Some(found) = cx.scope.lookup(name) else {
        let kind = match cx.decls.input(name) {
            Some(_) => RefusalKind::InputAssigned { input: name },
            None => RefusalKind::UnknownName { name },
        };
        refusal::push(cx, kind, span);
        return;
    };
    cx.out.names.insert(site, Name::Binding(found.binding.binder));
    note_captures(cx, name, &found, Demand::Write, span);
    super::closure::refuse_recursion(cx, value, name, span);
    cx.enter(site, value, found.binding.term, span);
}
