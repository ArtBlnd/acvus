use acvus_ast::{AstId, Expr, RefKind, Span};
use acvus_utils::{Astr, Interner};

use crate::graph::QualifiedRef;
use crate::ir::IndexAccess;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    Local(Astr),
    Input(Astr),
    Context(QualifiedRef),
}

impl Storage {
    fn of(expr: &Expr) -> Option<Self> {
        match expr {
            Expr::Ident {
                name,
                ref_kind: RefKind::Value,
                ..
            } => Some(Self::Local(name.name)),
            Expr::Ident {
                name,
                ref_kind: RefKind::ExternParam,
                ..
            } => Some(Self::Input(name.name)),
            Expr::ContextRef { name, .. } => Some(Self::Context(*name)),
            _ => None,
        }
    }

    fn of_store_root(root: acvus_ast::Root) -> Self {
        match root {
            acvus_ast::Root::Local(name) => Self::Local(name),
            acvus_ast::Root::ExternParam(name) => Self::Input(name),
            acvus_ast::Root::Context(name) => Self::Context(QualifiedRef::root(name)),
        }
    }
}

pub struct Projected<'e> {
    pub base: &'e Expr,
    pub fields: Vec<Astr>,
}

pub fn projected(expr: &Expr) -> Projected<'_> {
    let mut fields = Vec::new();
    let mut base = expr;
    loop {
        match base {
            Expr::FieldAccess { object, field, .. } => {
                fields.push(*field);
                base = object;
            }
            Expr::Paren { inner, .. } => base = inner,
            _ => break,
        }
    }
    fields.reverse();
    Projected { base, fields }
}

pub fn projected_store(place: &acvus_ast::Place) -> (&acvus_ast::PlaceBase, Vec<Astr>) {
    let mut fields = Vec::new();
    let mut node = place;
    let base = loop {
        match node {
            acvus_ast::Place::Field { object, field, .. } => {
                fields.push(*field);
                node = object;
            }
            acvus_ast::Place::Base(base) => break base,
        }
    };
    fields.reverse();
    (base, fields)
}

#[derive(Clone, Copy)]
pub struct Element<'e> {
    pub id: AstId,
    pub callee_id: AstId,
    pub container: &'e Expr,
    pub index: &'e Expr,
    pub span: Span,
}

impl<'e> Element<'e> {
    pub fn of(expr: &'e Expr) -> Option<Self> {
        let Expr::Index {
            id,
            callee_id,
            object,
            index,
            span,
        } = expr
        else {
            return None;
        };
        names_a_place(object).then_some(Self {
            id: *id,
            callee_id: *callee_id,
            container: object,
            index,
            span: *span,
        })
    }

    pub fn of_store(base: &'e acvus_ast::PlaceBase) -> Option<Self> {
        let acvus_ast::PlaceBase::Element {
            id,
            callee_id,
            container,
            index,
            span,
        } = base
        else {
            return None;
        };
        Some(Self {
            id: *id,
            callee_id: *callee_id,
            container: container.expr(),
            index,
            span: *span,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrittenBase {
    Storage(Storage),
    Element,
    Value,
}

impl WrittenBase {
    pub fn of(base: &Expr) -> Self {
        if let Some(storage) = Storage::of(base) {
            return Self::Storage(storage);
        }
        match Element::of(base) {
            Some(_) => Self::Element,
            None => Self::Value,
        }
    }

    pub fn of_store(base: &acvus_ast::PlaceBase) -> Self {
        match base {
            acvus_ast::PlaceBase::Root { root, .. } => Self::Storage(Storage::of_store_root(*root)),
            acvus_ast::PlaceBase::Element { .. } => Self::Element,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceBase {
    Storage(Storage),
    ThroughReferenceIn(Storage),
    ThroughReference,
    Element(IndexAccess),
    Temporary,
}

impl PlaceBase {
    /// Whether a place on this base is reached through a reference, so a
    /// read of it by value copies out of what the reference names.
    pub fn is_through_a_reference(self) -> bool {
        match self {
            Self::ThroughReferenceIn(_) | Self::ThroughReference | Self::Element(_) => true,
            Self::Storage(_) | Self::Temporary => false,
        }
    }
}

/// An index is left off: two elements of one container are one loan, the
/// one the container's slice holds (RFC-0047 rule 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loan {
    pub root: Storage,
    pub fields: Vec<Astr>,
}

impl Loan {
    pub fn of(expr: &Expr) -> Option<Self> {
        let Projected { base, fields } = projected(expr);
        let mut loan = match Storage::of(base) {
            Some(root) => Self {
                root,
                fields: Vec::new(),
            },
            None => Self::of(Element::of(base)?.container)?,
        };
        loan.fields.extend(fields);
        Some(loan)
    }

    pub fn display(&self, interner: &Interner) -> String {
        let mut out = match self.root {
            Storage::Local(name) | Storage::Input(name) => interner.resolve(name).to_string(),
            Storage::Context(qref) => format!("@{}", interner.resolve(qref.name)),
        };
        for field in &self.fields {
            out.push('.');
            out.push_str(interner.resolve(*field));
        }
        out
    }
}

pub fn names_a_place(expr: &Expr) -> bool {
    Loan::of(expr).is_some()
}
