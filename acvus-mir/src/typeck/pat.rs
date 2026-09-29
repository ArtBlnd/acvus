//! Patterns (patterns §1): a pattern naming a variant grows the enum where
//! transparent, and is refused where it is not (patterns §2); a `!`
//! scrutinee reaches no arm (patterns §4). Exhaustiveness is validate's.
//!
//! A pattern's names are bound when it is checked, each to a fresh term. How
//! it destructures its scrutinee depends on the scrutinee's head: through a
//! reference, Rust's default binding modes lend what it binds (patterns §1).
//! So a pattern is kept as a `Shape` and destructures in a decision fired
//! once that head is known (types §6). A pattern relates to its scrutinee by
//! `join`: a transparent scrutinee grows by what the pattern names, and an
//! opaque one must already have it.
//!
//! A name binds a read of a part of the scrutinee (D108 rule 1): past the
//! references the match passed, a reference part is a reborrow capped by
//! the path's mutability, and any other part is lent at it.

use acvus_ast::{AstId, ObjectPatternField, Pattern, RefKind, Slot, Span, TuplePatternElem};
use acvus_utils::{Astr, QualifiedRef};

use super::ctxt::Ctxt;
use super::decision::Decision;
use super::output::Name;
use super::place::{BindingMode, Through};
use super::refusal::{self, RefusalKind};
use super::scope::Binding;
use crate::decl::{Base, Head};
use crate::infer::Term;
use crate::ty::{EffectTerm, Infer as InferPhase, Label, LenTerm, Mutability, ObjectTy, TyKind};

/// A checked pattern, waiting for its scrutinee's head.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Shape {
    site: AstId,
    span: Span,
    form: Form,
}

#[derive(Debug, Clone, PartialEq)]
enum Form {
    /// A name, bound to `term`.
    Bind { term: Term },
    /// `@x`: the matched value enters the context. `body` is the effect of
    /// the body the pattern is written in, which a coercion into the
    /// context joins (effects §1).
    Context { context: Term, body: EffectTerm<InferPhase> },
    Ignore,
    /// A literal the scrutinee is compared with.
    Literal { literal: Term },
    List { head: Vec<Shape>, rest: bool, tail: Vec<Shape> },
    Object { fields: Vec<(Astr, Shape)> },
    Tuple { parts: Vec<Option<Shape>> },
    Variant { enum_name: Option<Astr>, tag: Astr, payload: Option<Box<Shape>> },
}

impl Shape {
    pub(super) fn span(&self) -> Span {
        self.span
    }

    /// The terms of every name this shape binds.
    pub(super) fn bindings(&self) -> Vec<Term> {
        let mut out = Vec::new();
        self.each_binding(&mut |term| out.push(term));
        out
    }

    fn each_binding(&self, on: &mut dyn FnMut(Term)) {
        match &self.form {
            Form::Bind { term } => on(*term),
            Form::Context { .. } | Form::Ignore | Form::Literal { .. } => {}
            Form::List { head, tail, .. } => head.iter().chain(tail).for_each(|part| part.each_binding(on)),
            Form::Object { fields } => fields.iter().for_each(|(_, part)| part.each_binding(on)),
            Form::Tuple { parts } => parts.iter().flatten().for_each(|part| part.each_binding(on)),
            Form::Variant { payload, .. } => {
                if let Some(payload) = payload {
                    payload.each_binding(on);
                }
            }
        }
    }
}

/// Binds `pattern`'s names against a scrutinee of term `scrutinee`.
pub(super) fn check_pattern<S: Slot>(cx: &mut Ctxt<'_>, pattern: &Pattern<S>, scrutinee: Term) {
    let shape = shape(cx, pattern);
    destructure(cx, shape, scrutinee, Vec::new());
}

fn shape<S: Slot>(cx: &mut Ctxt<'_>, pattern: &Pattern<S>) -> Shape {
    acvus_utils::grow(|| shape_level(cx, pattern))
}

fn shape_level<S: Slot>(cx: &mut Ctxt<'_>, pattern: &Pattern<S>) -> Shape {
    let (site, span) = (pattern.id(), pattern.span());
    let form = match pattern {
        Pattern::Binding {
            id,
            name,
            ref_kind,
            span,
        } => match ref_kind {
            RefKind::ExternParam => {
                refusal::push(cx, RefusalKind::InputAssigned { input: *name }, *span);
                Form::Ignore
            }
            RefKind::Value => {
                let term = cx.fresh();
                cx.record(*id, term);
                cx.scope.define(*name, Binding { binder: *id, term });
                Form::Bind { term }
            }
        },
        Pattern::ContextBind { id, name, span } => match super::context::context_term(cx, *id, *name, *span) {
            Some(context) => {
                super::context::refuse_held_loans(cx, context, *span);
                cx.out.names.insert(*id, Name::Context(*name));
                super::context::note_access(cx, *name, super::context::Access::Write, *span);
                let body = super::effect::body_effect(cx);
                Form::Context { context, body }
            }
            None => Form::Ignore,
        },
        Pattern::Literal { value, span, .. } => Form::Literal {
            literal: super::expr::check_literal(cx, value, *span),
        },
        Pattern::List { head, rest, tail, .. } => Form::List {
            head: head.iter().map(|part| shape(cx, part)).collect(),
            rest: rest.is_some(),
            tail: tail.iter().map(|part| shape(cx, part)).collect(),
        },
        Pattern::Object { fields, .. } => Form::Object {
            fields: fields.iter().map(|field| field_shape(cx, field)).collect(),
        },
        Pattern::Tuple { elements, .. } => Form::Tuple {
            parts: elements
                .iter()
                .map(|element| match element {
                    TuplePatternElem::Pattern(part) => Some(shape(cx, part)),
                    TuplePatternElem::Wildcard(_) => None,
                })
                .collect(),
        },
        Pattern::Wildcard { .. } | Pattern::Error(_) => Form::Ignore,
        Pattern::Variant {
            enum_name,
            tag,
            payload,
            ..
        } => Form::Variant {
            enum_name: *enum_name,
            tag: *tag,
            payload: payload.as_deref().map(|payload| Box::new(shape(cx, payload))),
        },
    };
    Shape { site, span, form }
}

fn field_shape<S: Slot>(cx: &mut Ctxt<'_>, field: &ObjectPatternField<S>) -> (Astr, Shape) {
    (field.key, shape(cx, &field.pattern))
}

/// `shape` against `scrutinee`, past the references in `through` the
/// match has passed. A name or `_` takes the value as it is; any other
/// shape waits for the scrutinee's head.
fn destructure(cx: &mut Ctxt<'_>, shape: Shape, scrutinee: Term, through: Vec<Through>) {
    match shape.form {
        Form::Bind { .. } | Form::Context { .. } | Form::Ignore => apply(cx, shape, scrutinee, None, through),
        _ => super::decision::decide(
            cx,
            Decision::Destructure {
                shape: Box::new(shape),
                scrutinee,
                through,
            },
        ),
    }
}

/// A destructuring fired on the scrutinee's head: through a reference, on
/// its target with names lent (patterns §1); on `!`, every name is `!`
/// (patterns §4); otherwise by the shape. `head` is `None` where the
/// pattern's use decides an open scrutinee at a stall: it is then the
/// shape, never a reference.
pub(super) fn fire_destructure(cx: &mut Ctxt<'_>, shape: Shape, scrutinee: Term, head: Option<Head>, mut through: Vec<Through>) {
    match head {
        Some(Head::Ref(..)) => {
            let target = cx.referent(scrutinee);
            through.push(Through::Object(scrutinee));
            destructure(cx, shape, target, through);
        }
        Some(Head::Base(Base::Never)) => {
            record_matched(cx, &shape, scrutinee);
            for term in shape.bindings() {
                let never = cx.never();
                cx.eq(term, never, shape.span);
            }
        }
        Some(Head::Base(_)) | None => apply(cx, shape, scrutinee, head, through),
    }
}

/// `shape` applied to `scrutinee`, whose head is `known` where it is known.
fn apply(cx: &mut Ctxt<'_>, shape: Shape, scrutinee: Term, known: Option<Head>, through: Vec<Through>) {
    record_matched(cx, &shape, scrutinee);
    let Shape { site, span, form } = shape;
    match form {
        Form::Bind { term } => bind(cx, span, term, scrutinee, through),
        Form::Context { context, body } => {
            let value = match lent(cx, &through) {
                Some(mutability) => cx.lend(scrutinee, mutability, span),
                None => scrutinee,
            };
            cx.enter_in(body, site, value, context, span);
        }
        Form::Ignore => {}
        Form::Literal { literal } => cx.join(scrutinee, literal, span),
        Form::List { head, rest, tail } => {
            let element = cx.fresh();
            let listed = match known {
                Some(Head::Base(Base::Slice)) => TyKind::Slice(element),
                _ => {
                    let len = match rest {
                        true => LenTerm::Var(cx.infer.fresh_len()),
                        false => LenTerm::Known(head.len() + tail.len()),
                    };
                    TyKind::Array(element, len)
                }
            };
            let listed = cx.held(listed);
            cx.join(scrutinee, listed, span);
            for part in head.into_iter().chain(tail) {
                destructure(cx, part, element, through.clone());
            }
        }
        Form::Object { fields } => object(cx, fields, scrutinee, through, span),
        Form::Tuple { parts } => {
            let terms: Vec<Term> = parts.iter().map(|_| cx.fresh()).collect();
            let tuple = cx.held(TyKind::Tuple(terms.clone().into()));
            cx.join(scrutinee, tuple, span);
            for (part, term) in parts.into_iter().zip(terms) {
                if let Some(part) = part {
                    destructure(cx, part, term, through.clone());
                }
            }
        }
        Form::Variant {
            enum_name,
            tag,
            payload,
        } => variant(cx, enum_name, tag, payload.map(|payload| *payload), scrutinee, through, span),
    }
}

/// The type a pattern's site has: what it matched, except at a name,
/// whose site is its binder and has the binding's type, lent where the
/// match passed through a reference (patterns §1).
fn record_matched(cx: &mut Ctxt<'_>, shape: &Shape, scrutinee: Term) {
    if !matches!(shape.form, Form::Bind { .. }) {
        cx.record(shape.site, scrutinee);
    }
}

/// A name `term` bound to `part`, a read of it (D108 rule 1). Past no
/// reference the path is the scrutinee's own `&mut` place, where a read of
/// any part is the part itself; past one it is the one part-read decision,
/// binding by reference (patterns §1).
fn bind(cx: &mut Ctxt<'_>, span: Span, term: Term, part: Term, through: Vec<Through>) {
    if through.is_empty() {
        cx.eq(term, part, span);
        return;
    }
    let read = Decision::PartRead {
        span,
        part,
        through: through.into(),
        mode: BindingMode::Ref,
        result: term,
    };
    super::decision::decide(cx, read);
}

/// The mutability a match lends at past the references in `through`, where
/// it passed any; each is closed, since the match passed it.
fn lent(cx: &Ctxt<'_>, through: &[Through]) -> Option<Mutability> {
    if through.is_empty() {
        return None;
    }
    let path = super::decision::path_through(cx, through, super::decision::Reading::CycleBroken, super::decision::Depth::Head);
    Some(path.expect("a reference a match passed is closed"))
}

/// `{ f: p, .. }`: a declared struct must have each field; any other
/// object grows by them where transparent (types §1-2). The scrutinee's
/// head is known here, so whether it is declared is read off it.
fn object(cx: &mut Ctxt<'_>, fields: Vec<(Astr, Shape)>, scrutinee: Term, through: Vec<Through>, span: Span) {
    let declared = match cx.infer.types().kind(cx.infer.shallow(scrutinee)) {
        TyKind::Object(ObjectTy::Declared(_, declared)) => Some(declared.clone()),
        _ => None,
    };
    if let Some(declared) = declared {
        for (key, part) in fields {
            match declared.iter().find(|(name, _)| *name == key) {
                Some((_, term)) => destructure(cx, part, *term, through.clone()),
                None => refusal::push(cx, RefusalKind::GrowsOpaque, part.span),
            }
        }
        return;
    }
    let terms: Vec<(Astr, Term)> = fields.iter().map(|(key, _)| (*key, cx.fresh())).collect();
    let mut sorted = terms.clone();
    sorted.sort_by_key(|(name, _)| *name);
    let object = cx.held(TyKind::Object(ObjectTy::Structural(sorted.into(), Label::Transparent)));
    cx.join(scrutinee, object, span);
    for ((_, part), (_, term)) in fields.into_iter().zip(terms) {
        destructure(cx, part, term, through.clone());
    }
}

/// A variant pattern: a named enum's variant grows the enum where
/// transparent (patterns §2); without a name, only `Some`, `None`, `Ok` and
/// `Err` of the primitive `Option` and `Result` (types §2).
fn variant(
    cx: &mut Ctxt<'_>,
    enum_name: Option<Astr>,
    tag: Astr,
    payload: Option<Shape>,
    scrutinee: Term,
    through: Vec<Through>,
    span: Span,
) {
    let tags = cx.decls.core().tags;
    let (kind, part) = match enum_name {
        Some(name) => {
            let part = payload.as_ref().map(|_| cx.fresh());
            let kind = TyKind::Enum {
                name: QualifiedRef::root(name),
                variants: Box::new([(tag, part)]),
                label: Label::Transparent,
            };
            (kind, part)
        }
        None if tag == tags.some || tag == tags.none => {
            let inner = cx.fresh();
            (TyKind::Option(inner), (tag == tags.some).then_some(inner))
        }
        None if tag == tags.ok || tag == tags.err => {
            let (ok, err) = (cx.fresh(), cx.fresh());
            (TyKind::Result(ok, err), Some(if tag == tags.ok { ok } else { err }))
        }
        None => {
            refusal::push(cx, RefusalKind::UnnamedVariant { tag }, span);
            return;
        }
    };
    let variant = cx.held(kind);
    cx.join(scrutinee, variant, span);
    match (payload, part) {
        (Some(payload), Some(part)) => destructure(cx, payload, part, through),
        (None, None) => {}
        (Some(_), None) | (None, Some(_)) => refusal::push(cx, RefusalKind::VariantArity { tag }, span),
    }
}
