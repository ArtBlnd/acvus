//! Patterns (patterns §1): a pattern naming a variant grows the enum where
//! transparent, and is refused where it is not (patterns §2); a `!`
//! scrutinee reaches no arm (patterns §4). Exhaustiveness is check's.
//!
//! A pattern's names are bound when it is checked, each to a fresh term. How
//! it destructures its scrutinee depends on the scrutinee's head: through a
//! reference, Rust's default binding modes lend what it binds (patterns §1).
//! So a pattern is kept as a `Shape` and destructures in a decision fired
//! once that head is known (types §6). A pattern relates to its scrutinee by
//! `join`: a transparent scrutinee grows by what the pattern names, and an
//! opaque one must already have it.

use acvus_ast::{AstId, ObjectPatternField, Pattern, RefKind, Slot, Span, TuplePatternElem};
use acvus_utils::{Astr, QualifiedRef};

use super::ctxt::Ctxt;
use super::decision::Decision;
use super::output::Name;
use super::refusal::{self, RefusalKind};
use super::scope::Binding;
use crate::decl::{Base, Head};
use crate::infer::Term;
use crate::ty::{EffectTerm, Infer as InferPhase, Label, LenTerm, Mutability, ObjectTy, Reach, TyKind};

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

    /// This shape with each term and effect at an instance (G).
    pub(super) fn map_terms(&self, f: &mut dyn super::decision::MapTerms) -> Shape {
        let form = match &self.form {
            Form::Bind { term } => Form::Bind { term: f.ty(*term) },
            Form::Context { context, body } => Form::Context { context: f.ty(*context), body: f.effect(body) },
            Form::Ignore => Form::Ignore,
            Form::Literal { literal } => Form::Literal { literal: f.ty(*literal) },
            Form::List { head, rest, tail } => Form::List {
                head: head.iter().map(|part| part.map_terms(f)).collect(),
                rest: *rest,
                tail: tail.iter().map(|part| part.map_terms(f)).collect(),
            },
            Form::Object { fields } => Form::Object { fields: fields.iter().map(|(name, part)| (*name, part.map_terms(f))).collect() },
            Form::Tuple { parts } => Form::Tuple { parts: parts.iter().map(|part| part.as_ref().map(|part| part.map_terms(f))).collect() },
            Form::Variant { enum_name, tag, payload } => Form::Variant {
                enum_name: *enum_name,
                tag: *tag,
                payload: payload.as_ref().map(|payload| Box::new(payload.map_terms(f))),
            },
        };
        Shape { site: f.site(self.site), span: f.span(self.span), form }
    }

    /// The effects this shape's `@x` parts join a coercion into.
    pub(super) fn effects(&self) -> Vec<EffectTerm<InferPhase>> {
        let mut out = Vec::new();
        self.each_effect(&mut |effect| out.push(effect.clone()));
        out
    }

    fn each_effect(&self, on: &mut dyn FnMut(&EffectTerm<InferPhase>)) {
        match &self.form {
            Form::Context { body, .. } => on(body),
            Form::Bind { .. } | Form::Ignore | Form::Literal { .. } => {}
            Form::List { head, tail, .. } => head.iter().chain(tail).for_each(|part| part.each_effect(on)),
            Form::Object { fields } => fields.iter().for_each(|(_, part)| part.each_effect(on)),
            Form::Tuple { parts } => parts.iter().flatten().for_each(|part| part.each_effect(on)),
            Form::Variant { payload, .. } => {
                if let Some(payload) = payload {
                    payload.each_effect(on);
                }
            }
        }
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
    destructure(cx, shape, scrutinee, None);
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
                cx.scope.define(*name, Binding { binder: *id, term, scheme: None });
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

/// `shape` against `scrutinee`, its names lent by `lent` where the match
/// passed through a reference. A name or `_` takes the value as it is; any
/// other shape waits for the scrutinee's head.
fn destructure(cx: &mut Ctxt<'_>, shape: Shape, scrutinee: Term, lent: Option<Mutability>) {
    match shape.form {
        Form::Bind { .. } | Form::Context { .. } | Form::Ignore => apply(cx, shape, (scrutinee, Reach::Named), None, lent),
        _ => super::decision::decide(
            cx,
            Decision::Destructure {
                shape: Box::new(shape),
                scrutinee,
                lent,
            },
        ),
    }
}

/// A destructuring fired on the scrutinee's head: through a reference, on
/// its target with names lent (patterns §1); on `!`, every name is `!`
/// (patterns §4); otherwise by the shape. `head` is `None` where the
/// pattern's use decides an open scrutinee at a stall: it is then the
/// shape, never a reference. `reach` is how the scrutinee was reached.
pub(super) fn fire_destructure(cx: &mut Ctxt<'_>, shape: Shape, (scrutinee, reach): (Term, Reach), head: Option<Head>, lent: Option<Mutability>) {
    match head {
        Some(Head::Ref(mutability, _)) => {
            let through = match lent {
                Some(Mutability::Shared) => Mutability::Shared,
                Some(Mutability::Mut) | None => mutability,
            };
            let target = cx.referent(scrutinee);
            destructure(cx, shape, target, Some(through));
        }
        Some(Head::Base(Base::Never)) => {
            record_matched(cx, &shape, scrutinee);
            for term in shape.bindings() {
                let never = cx.never();
                cx.eq(term, never, shape.span);
            }
        }
        Some(Head::Base(_)) | None => apply(cx, shape, (scrutinee, reach), head, lent),
    }
}

/// `shape` applied to `scrutinee`, whose head is `known` where it is known,
/// reached by `reach` or through a reference where the match passed one.
/// Each part it takes apart is reached as the scrutinee was, which is
/// recorded where that is through a reference (types §1).
fn apply(cx: &mut Ctxt<'_>, shape: Shape, (scrutinee, reach): (Term, Reach), known: Option<Head>, lent: Option<Mutability>) {
    record_matched(cx, &shape, scrutinee);
    let Shape { site, span, form } = shape;
    let least = lent.or_else(|| super::place::path_of(cx, scrutinee, reach));
    let reach = match lent {
        Some(_) => Reach::Through,
        None => reach,
    };
    match form {
        Form::Bind { term } => {
            let value = bound_value(cx, scrutinee, lent, span);
            cx.eq(term, value, span);
        }
        Form::Context { context, body } => {
            let value = bound_value(cx, scrutinee, lent, span);
            cx.enter_in(body, site, value, context, Reach::Named, span);
        }
        Form::Ignore => {}
        Form::Literal { literal } => cx.join(scrutinee, literal, reach, span),
        Form::List { head, rest, tail } => {
            let element = reached_part(cx, least);
            let listed = match known {
                Some(Head::Base(Base::Slice)) => TyKind::Slice(element),
                _ => {
                    let len = match rest {
                        true => LenTerm::Var(cx.infer.fresh_len()),
                        false => LenTerm::Known(head.len() + tail.len()),
                    };
                    TyKind::Array(element, len, Label::Here)
                }
            };
            let listed = cx.held(listed);
            cx.join(scrutinee, listed, reach, span);
            for part in head.into_iter().chain(tail) {
                destructure(cx, part, element, lent);
            }
        }
        Form::Object { fields } => object(cx, fields, (scrutinee, reach), lent, span),
        Form::Tuple { parts } => {
            let terms: Vec<Term> = parts.iter().map(|_| reached_part(cx, least)).collect();
            let tuple = cx.held(TyKind::Tuple(terms.clone().into(), Label::Here));
            cx.join(scrutinee, tuple, reach, span);
            for (part, term) in parts.into_iter().zip(terms) {
                if let Some(part) = part {
                    destructure(cx, part, term, lent);
                }
            }
        }
        Form::Variant {
            enum_name,
            tag,
            payload,
        } => variant(cx, enum_name, tag, payload.map(|payload| *payload), (scrutinee, reach), lent, span),
    }
}

/// A fresh term for a part a pattern takes apart, reached by `reach`:
/// recorded where that is through a reference, so that every step after it
/// is too (types §1).
fn reached_part(cx: &mut Ctxt<'_>, least: Option<Mutability>) -> Term {
    let part = cx.fresh();
    if let Some(least) = least {
        cx.out.through.insert(part, least);
    }
    part
}

/// The type a pattern's site has: what it matched, except at a name,
/// whose site is its binder and has the binding's type, lent where the
/// match passed through a reference (patterns §1).
fn record_matched(cx: &mut Ctxt<'_>, shape: &Shape, scrutinee: Term) {
    if !matches!(shape.form, Form::Bind { .. }) {
        cx.record(shape.site, scrutinee);
    }
}

/// What a name binds: the part itself, or where the match passed through
/// a reference, a reference to it.
fn bound_value(cx: &mut Ctxt<'_>, part: Term, lent: Option<Mutability>, span: Span) -> Term {
    match lent {
        Some(mutability) => cx.lend_as(part, super::decision::Lent::Bind(mutability), span),
        None => part,
    }
}

/// `{ f: p, .. }`: a declared struct must have each field; any other
/// object grows by them where transparent (types §1-2). The scrutinee's
/// head is known here, so whether it is declared is read off it.
fn object(cx: &mut Ctxt<'_>, fields: Vec<(Astr, Shape)>, (scrutinee, reach): (Term, Reach), lent: Option<Mutability>, span: Span) {
    let declared = match cx.infer.types().kind(cx.infer.shallow(scrutinee)) {
        TyKind::Object(ObjectTy::Declared(_, declared)) => Some(declared.clone()),
        _ => None,
    };
    if let Some(declared) = declared {
        for (key, part) in fields {
            match declared.iter().find(|(name, _)| *name == key) {
                Some((_, term)) => destructure(cx, part, *term, lent),
                None => refusal::push(cx, RefusalKind::GrowsOpaque, part.span),
            }
        }
        return;
    }
    let least = lent.or_else(|| super::place::path_of(cx, scrutinee, reach));
    let terms: Vec<(Astr, Term)> = fields.iter().map(|(key, _)| (*key, reached_part(cx, least))).collect();
    let mut sorted = terms.clone();
    sorted.sort_by_key(|(name, _)| *name);
    let object = cx.held(TyKind::Object(ObjectTy::Structural(sorted.into(), Label::Here)));
    cx.join(scrutinee, object, reach, span);
    for ((_, part), (_, term)) in fields.into_iter().zip(terms) {
        destructure(cx, part, term, lent);
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
    (scrutinee, reach): (Term, Reach),
    lent: Option<Mutability>,
    span: Span,
) {
    let tags = cx.decls.core().tags;
    let least = lent.or_else(|| super::place::path_of(cx, scrutinee, reach));
    let (kind, part) = match enum_name {
        Some(name) => {
            let part = payload.as_ref().map(|_| reached_part(cx, least));
            let kind = TyKind::Enum {
                name: QualifiedRef::root(name),
                variants: Box::new([(tag, part)]),
                label: Label::Here,
            };
            (kind, part)
        }
        None if tag == tags.some || tag == tags.none => {
            let inner = reached_part(cx, least);
            (TyKind::Option(inner, Label::Here), (tag == tags.some).then_some(inner))
        }
        None if tag == tags.ok || tag == tags.err => {
            let (ok, err) = (reached_part(cx, least), reached_part(cx, least));
            (TyKind::Result(ok, err, Label::Here), Some(if tag == tags.ok { ok } else { err }))
        }
        None => {
            refusal::push(cx, RefusalKind::UnnamedVariant { tag }, span);
            return;
        }
    };
    let variant = cx.held(kind);
    cx.join(scrutinee, variant, reach, span);
    match (payload, part) {
        (Some(payload), Some(part)) => destructure(cx, payload, part, lent),
        (None, None) => {}
        (Some(_), None) | (None, Some(_)) => refusal::push(cx, RefusalKind::VariantArity { tag }, span),
    }
}
