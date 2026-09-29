//! Bindings, and what a name reaches (types §6, programs §4).

use acvus_ast::{AstId, RefKind, Span};
use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::FxHashMap;

use super::ctxt::Ctxt;
use super::output::{CaptureOf, CaptureUse, Name};
use super::place::Demand;
use super::refusal::{self, RefusalKind};
use crate::infer::Term;
use crate::ty::{EffectTerm, Infer as InferPhase, Mutability};

/// One binding: its binder's site and its term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Binding {
    pub binder: AstId,
    pub term: Term,
    /// Where the binding is generalized (functions §2), its scheme: each
    /// use instantiates it.
    pub scheme: Option<super::generalize::SchemeId>,
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

/// What kind of source the run is (programs §1-2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Run {
    Script,
    Template,
}

/// The run's inputs, its parameters (programs §4), each by name with its
/// term.
enum Inputs {
    /// Each `$name` read is a parameter: a fresh term, made at its first
    /// read, that its uses decide.
    Open(FxHashMap<Astr, Term>),
    /// The inputs given, each at its type; no other `$name` is one.
    Fixed(FxHashMap<Astr, Term>),
}

/// What `$name` is among the run's inputs.
pub(super) enum InputRead {
    Entered(Term),
    /// An open input read for the first time: its term is to be made.
    Unread,
    Unknown,
}

/// The bindings in scope, innermost last, the kind of run, its inputs, and
/// the run's own result slot once a `return`, a `?` or the tail names it.
pub(super) struct Scope {
    frames: Vec<Frame>,
    run: Run,
    inputs: Inputs,
    run_result: Option<Term>,
}

impl Scope {
    /// A run whose inputs are open, until `fix_inputs`.
    pub fn new(run: Run) -> Self {
        Self {
            run,
            inputs: Inputs::Open(FxHashMap::default()),
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

    /// Every lambda whose body is in scope, innermost first.
    pub fn lambdas(&self) -> Vec<AstId> {
        self.frames
            .iter()
            .rev()
            .filter_map(|frame| frame.body.as_ref())
            .map(|body| body.lambda)
            .collect()
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

    pub fn run(&self) -> Run {
        self.run
    }

    /// The run's inputs are `given`, each at its term, and no other.
    pub fn fix_inputs(&mut self, given: FxHashMap<Astr, Term>) {
        self.inputs = Inputs::Fixed(given);
    }

    pub fn input(&self, name: Astr) -> InputRead {
        let (Inputs::Open(entered) | Inputs::Fixed(entered)) = &self.inputs;
        match (entered.get(&name), &self.inputs) {
            (Some(term), _) => InputRead::Entered(*term),
            (None, Inputs::Open(_)) => InputRead::Unread,
            (None, Inputs::Fixed(_)) => InputRead::Unknown,
        }
    }

    /// Every input's term entered so far: the run's parameters, which the
    /// run's scope holds wherever each was first read.
    pub fn input_terms(&self) -> impl Iterator<Item = Term> + '_ {
        let (Inputs::Open(entered) | Inputs::Fixed(entered)) = &self.inputs;
        entered.values().copied()
    }

    /// An open input's term, made at its first read.
    ///
    /// # Panics
    /// The inputs are fixed, or `name` is already read: `input` answers
    /// `Unread` only for an open input not yet read.
    pub fn enter_input(&mut self, name: Astr, term: Term) {
        let Inputs::Open(read) = &mut self.inputs else {
            panic!("Scope::enter_input: only an open input is made at its read");
        };
        assert!(read.insert(name, term).is_none(), "Scope::enter_input: an input is made once, at its first read");
    }

    /// Whether `name` is a fixed input. An open run knows its inputs only as
    /// `$name` reads them.
    pub fn is_fixed_input(&self, name: Astr) -> bool {
        matches!(&self.inputs, Inputs::Fixed(given) if given.contains_key(&name))
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
    if name.namespace.is_root()
        && let Some(found) = cx.scope.lookup(name.name)
    {
        cx.out.names.insert(site, Name::Binding(found.binding.binder));
        note_captures(cx, capturing_binding(name.name, &found), Demand::Read, span);
        return match found.binding.scheme {
            Some(scheme) => super::generalize::instantiate(cx, scheme, site, span),
            None => found.binding.term,
        };
    }
    let declarations: Box<[QualifiedRef]> = if name.namespace.is_root() {
        cx.decls.overloads(name.name).into()
    } else {
        cx.decls.scheme(name).map(|_| name).into_iter().collect()
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

/// `$name`: a parameter of the run (programs §4), at its term. Read inside a
/// lambda, it is a capture of each lambda around it, as a binding is
/// (functions §2).
fn check_input(cx: &mut Ctxt<'_>, site: AstId, name: Astr, span: Span) -> Term {
    let entered = match cx.scope.input(name) {
        InputRead::Entered(term) => Some(term),
        InputRead::Unread => {
            let term = cx.fresh();
            cx.scope.enter_input(name, term);
            Some(term)
        }
        InputRead::Unknown => None,
    };
    match entered {
        Some(term) => {
            cx.out.names.insert(site, Name::Input(name));
            let crossed = cx.scope.lambdas();
            let capturing = Capturing {
                name,
                of: CaptureOf::Input(name),
                crossed: &crossed,
                term,
            };
            note_captures(cx, capturing, Demand::Read, span);
            term
        }
        None => {
            refusal::push(cx, RefusalKind::UnknownName { name }, span);
            cx.fresh()
        }
    }
}

/// A name read inside the lambdas `crossed`, innermost first, is a
/// capture of each of them (functions §2): by move in a lambda written
/// with `move`, whose holds then wait for the captured value's type, of
/// term `term`; otherwise by reference, and mutably where it is written or
/// lent mutably.
pub(super) struct Capturing<'a> {
    pub name: Astr,
    pub of: CaptureOf,
    pub crossed: &'a [AstId],
    pub term: Term,
}

pub(super) fn note_captures(cx: &mut Ctxt<'_>, capturing: Capturing<'_>, demand: Demand, _span: Span) {
    let Capturing { name, of, crossed, term } = capturing;
    if crossed.is_empty() {
        return;
    }
    let mutability = match demand {
        Demand::Read | Demand::Lend(Mutability::Shared) => Mutability::Shared,
        Demand::Write | Demand::Lend(Mutability::Mut) => Mutability::Mut,
    };
    cx.out.capture_uses.push(CaptureUse { name, of, crossed: crossed.into(), mutability, term });
}

/// `found`'s binding as a capture.
pub(super) fn capturing_binding(name: Astr, found: &Found) -> Capturing<'_> {
    Capturing {
        name,
        of: CaptureOf::Binding(found.binding.binder),
        crossed: &found.crossed,
        term: found.binding.term,
    }
}

/// `x = v`: a binding in scope; an input is never assigned (programs §4).
pub(super) fn check_assign(cx: &mut Ctxt<'_>, site: AstId, name: Astr, value: Term, span: Span) {
    let Some(found) = cx.scope.lookup(name) else {
        let kind = match cx.scope.is_fixed_input(name) {
            true => RefusalKind::InputAssigned { input: name },
            false => RefusalKind::UnknownName { name },
        };
        refusal::push(cx, kind, span);
        return;
    };
    cx.out.names.insert(site, Name::Binding(found.binding.binder));
    note_captures(cx, capturing_binding(name, &found), Demand::Write, span);
    super::closure::refuse_recursion(cx, value, name, span);
    cx.enter(site, value, found.binding.term, crate::ty::Reach::Named, span);
}
