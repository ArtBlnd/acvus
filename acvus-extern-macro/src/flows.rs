//! An extern's flows, read off its Rust signature (RFC-0079 rule 6).
//!
//! Each lifetime and type variable a signature names is a label. An output
//! end flows from an input end where a label of the input reaches a label
//! of the output: the same label; a lifetime it outlives, written
//! (`'a: 'b`) or implied (`&'b T` for every lifetime in `T`); a closure's
//! result, from what the closure captured and is handed; and, inside a type
//! the macro does not read or an `InstanceOf`, every label beside it. An
//! `Instance` parameter is read as the receiver it owns and as the
//! requirement at its variable.
//!
//! An end is *laid out* where the macro reads its type position by
//! position: a reference, an option, a result, a tuple, an array, a type
//! variable, and an extension type, read as its region parameters, then its
//! type arguments (RFC-0096 rule 2), a type argument it does not read being
//! one segment of every label it names. Between two laid-out ends a flow
//! maps each output segment to the input segments whose labels reach its
//! own (RFC-0096 rule 1): `Aligned` where that map is one to one, `Any`
//! where it is every to every, `Labelled` otherwise.
//!
//! Whether a generic type is an extension type is the type's own statement,
//! `TyArg::LAYOUT`, which the macro cannot see. The macro reads the
//! signature once per assignment of a reading to each generic type an end
//! names, and the declaration keeps the flows of the assignment the named
//! types state (`Derived::tokens`).

use std::collections::{BTreeMap, BTreeSet};

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::visit::Visit;
use syn::{FnArg, GenericArgument, GenericParam, Lifetime, PathArguments, Type, WherePredicate};

use crate::generics::{VarKind, Vars};

/// `'static` names no loan: it reaches nothing and nothing reaches it.
const STATIC: &str = "'static";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Label {
    /// Written lifetimes by name; an elided one by the place it is elided.
    Life(String),
    /// A type variable of kind `Type`.
    Var(String),
    /// A reference the handler lends a closure that the acvus parameter at
    /// this index holds: it may name anything the handler was handed.
    LentTo(usize),
}

impl Label {
    fn is_static(&self) -> bool {
        matches!(self, Label::Life(name) if name == STATIC)
    }
}

/// What a Rust parameter is to the checker.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Its lifetimes count for elision, and carry none a result may name.
    Ctx,
    /// The acvus parameter at this index: an end.
    Acvus(usize),
    /// A `#[state]` or an `InstanceOf`: no end.
    Other,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    Owned,
    Mut,
    Shared,
}

enum Alignment {
    Aligned,
    Labelled(Labelled),
    Any,
}

/// The macro's reading of `acvus_extern::Laid`.
#[derive(Clone, PartialEq, Eq)]
enum Laid {
    NoPosition,
    Ref(Box<Laid>),
    Option(Box<Laid>),
    Result(Box<Laid>, Box<Laid>),
    Tuple(Vec<Laid>),
    Array(Box<Laid>),
    User {
        candidate: usize,
        regions: usize,
        args: Vec<ReadArg>,
    },
    Var,
    Unread,
}

#[derive(Clone, PartialEq, Eq)]
struct ReadArg {
    written_at: usize,
    laid: Laid,
}

/// What one segment of a laid-out end is.
#[derive(Clone, PartialEq, Eq)]
enum Segment {
    /// One position: a reference, or a region parameter, at this lifetime.
    Life(Label),
    /// Every position of a type variable's value.
    Var(Label),
    /// Every position of a type argument the macro does not read, which
    /// may hold any of these labels.
    Unread(BTreeSet<Label>),
}

impl Segment {
    fn labels(&self) -> BTreeSet<Label> {
        match self {
            Segment::Life(label) | Segment::Var(label) => BTreeSet::from([label.clone()]),
            Segment::Unread(labels) => labels.clone(),
        }
    }
}

/// An end read position by position: its shape, and its segments in the
/// order `Laid` numbers them.
#[derive(Clone)]
struct Layout {
    laid: Laid,
    segments: Vec<Segment>,
}

impl Layout {
    fn nothing() -> Self {
        Layout {
            laid: Laid::NoPosition,
            segments: Vec::new(),
        }
    }
}

/// Where one output segment takes from: the input segment, and whether
/// position by position.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Take {
    segment: usize,
    aligned: bool,
}

struct Labelled {
    to: Laid,
    from: Laid,
    takes: Vec<Vec<Take>>,
}

/// Mirrors `acvus_extern::laid::WrittenArg`.
#[derive(Clone, Copy)]
enum WrittenArg {
    Lifetime,
    Positioned,
    NonType,
    Positionless,
}

/// A generic type an end names, which states its own layout.
#[derive(Clone)]
struct Candidate {
    /// Where the signature writes it, as the key a reading is looked up by.
    at: *const Type,
    ty: Type,
    written: Vec<WrittenArg>,
}

/// Which generic types a reading of the signature lays out: in discovery
/// every one, each recorded as it is met; otherwise one flag per candidate
/// discovery recorded.
enum Reading {
    Discovery,
    Assigned { laid: Vec<bool> },
}

/// The flows of one reading, and which candidates it lays out.
struct ReadFlows {
    laid: Vec<bool>,
    flows: Vec<Flow>,
}

/// What one reading of the signature finds.
struct Read {
    flows: Vec<Flow>,
    candidates: Vec<Candidate>,
}

/// One arm of the dispatch on the statements: the readings whose flows are
/// one token stream.
struct Arm {
    flows: TokenStream,
    rendered: String,
    patterns: Vec<TokenStream>,
}

#[derive(Clone, Copy)]
enum End {
    Result,
    Param(usize),
}

struct Edge {
    from: Label,
    to: Label,
}

struct Flow {
    to: End,
    from: usize,
    alignment: Alignment,
}

/// A lifetime the result names, and where.
struct ResultLifetime {
    name: String,
    span: Span,
}

/// Where a type is read.
#[derive(Clone)]
struct Cx {
    /// The acvus parameter being read; `None` for the result and for a
    /// parameter that is no end.
    end: Option<usize>,
    /// The brands of the carriers around it: a carrier reads what it names
    /// at its brand.
    brands: Vec<Label>,
    in_closure_args: bool,
    access: Access,
}

impl Cx {
    fn top(end: Option<usize>) -> Self {
        Cx {
            end,
            brands: Vec::new(),
            in_closure_args: false,
            access: Access::Owned,
        }
    }
}

/// A lifetime read in a `Cx`.
struct ReadLifetime {
    own: Label,
    labels: Vec<Label>,
}

#[derive(Default)]
struct Shape {
    labels: BTreeSet<Label>,
    /// The labels of what a callee handed this value may write into.
    written: BTreeSet<Label>,
    /// The positions read one by one; `None` where they are not the
    /// macro's to read.
    layout: Option<Layout>,
}

impl Shape {
    fn positionless() -> Self {
        Shape {
            layout: Some(Layout::nothing()),
            ..Shape::default()
        }
    }

    /// The parts side by side, laid out as `laid` makes of theirs where
    /// every part is laid out.
    fn concat(parts: Vec<Shape>, laid: impl FnOnce(Vec<Laid>) -> Laid) -> Self {
        let mut shape = Shape::default();
        let mut layouts = Some(Vec::new());
        for part in parts {
            shape.labels.extend(part.labels);
            shape.written.extend(part.written);
            layouts = match (layouts, part.layout) {
                (Some(mut all), Some(one)) => {
                    all.push(one);
                    Some(all)
                }
                (None, _) | (_, None) => None,
            };
        }
        shape.layout = layouts.map(|layouts| {
            let segments = layouts.iter().flat_map(|l| l.segments.iter().cloned()).collect();
            Layout {
                laid: laid(layouts.into_iter().map(|l| l.laid).collect()),
                segments,
            }
        });
        shape
    }

    fn wrapped(mut self, laid: impl FnOnce(Box<Laid>) -> Laid) -> Self {
        self.layout = self.layout.map(|layout| Layout {
            laid: laid(Box::new(layout.laid)),
            segments: layout.segments,
        });
        self
    }
}

enum Side {
    Params,
    /// An elided lifetime in the result takes the parameters' one lifetime,
    /// by Rust's elision rules; `None` where they name more or none.
    Result(Option<String>),
}

struct Walker<'v> {
    vars: &'v Vars,
    edges: Vec<Edge>,
    elided_count: usize,
    side: Side,
    named_by_params: BTreeSet<String>,
    named_by_ctx: BTreeSet<String>,
    carried: BTreeSet<String>,
    reading_ctx: bool,
    named_by_result: Vec<ResultLifetime>,
    /// Whether the type being read is an end's or the result's; a `ctx`, a
    /// `#[state]` or a requirement lays out no generic type.
    laying_out: bool,
    reading: &'v Reading,
    candidates: Vec<Candidate>,
}

impl Walker<'_> {
    fn lifetime(&mut self, written: Option<&Lifetime>, span: Span, cx: &Cx) -> syn::Result<ReadLifetime> {
        let name = match (written, &self.side) {
            (Some(lt), _) if lt.ident == "static" => STATIC.to_string(),
            (Some(lt), _) if lt.ident != "_" => format!("'{}", lt.ident),
            (_, Side::Params) => {
                self.elided_count += 1;
                format!("'_{}", self.elided_count)
            }
            (_, Side::Result(Some(taken))) => taken.clone(),
            (_, Side::Result(None)) => {
                return Err(syn::Error::new(
                    span,
                    "the result elides a lifetime, and the parameters do not name exactly one \
                     for it to take (Rust's elision rules): name the lifetime of the parameter \
                     the result borrows (RFC-0079 rule 6)",
                ));
            }
        };
        match self.side {
            Side::Params => {
                self.named_by_params.insert(name.clone());
                match self.reading_ctx {
                    true => self.named_by_ctx.insert(name.clone()),
                    false => self.carried.insert(name.clone()),
                };
            }
            Side::Result(_) => self.named_by_result.push(ResultLifetime {
                name: name.clone(),
                span,
            }),
        }
        let own = Label::Life(name);
        let mut labels: Vec<Label> = cx.brands.clone();
        match cx.in_closure_args {
            true => labels.extend(cx.end.map(Label::LentTo)),
            false if !own.is_static() => labels.push(own.clone()),
            false => {}
        }
        Ok(ReadLifetime { own, labels })
    }

    fn walk(&mut self, ty: &Type, cx: &Cx) -> syn::Result<Shape> {
        match ty {
            Type::Reference(r) => self.reference(r, cx),
            Type::Paren(p) => self.walk(&p.elem, cx),
            Type::Group(g) => self.walk(&g.elem, cx),
            Type::Array(a) => Ok(self.walk(&a.elem, cx)?.wrapped(Laid::Array)),
            Type::Slice(s) => Ok(self.walk(&s.elem, cx)?.wrapped(Laid::Array)),
            Type::Tuple(t) => {
                let parts = t
                    .elems
                    .iter()
                    .map(|elem| self.walk(elem, cx))
                    .collect::<syn::Result<_>>()?;
                Ok(Shape::concat(parts, Laid::Tuple))
            }
            Type::Never(_) => Ok(Shape::positionless()),
            Type::Path(p) if p.qself.is_none() => self.path(ty, &p.path, cx),
            _ => self.unread(ty, cx),
        }
    }

    fn reference(&mut self, r: &syn::TypeReference, cx: &Cx) -> syn::Result<Shape> {
        let read = self.lifetime(r.lifetime.as_ref(), r.and_token.span, cx)?;
        let access = match (cx.access, r.mutability) {
            (Access::Shared, _) | (_, None) => Access::Shared,
            (Access::Owned | Access::Mut, Some(_)) => Access::Mut,
        };
        let pointee_cx = Cx { access, ..cx.clone() };
        let pointee = match r.elem.as_ref() {
            Type::Path(p) if p.qself.is_none() && p.path.is_ident("str") => Shape::positionless(),
            elem => self.walk(elem, &pointee_cx)?,
        };
        self.outlive(&pointee.labels, &read.labels);
        let mut shape = Shape {
            layout: pointee.layout.map(|pointee| Layout {
                laid: Laid::Ref(Box::new(pointee.laid)),
                segments: std::iter::once(Segment::Life(read.own))
                    .chain(pointee.segments)
                    .collect(),
            }),
            ..Shape::default()
        };
        if access == Access::Mut {
            shape.written.extend(pointee.labels.iter().cloned());
        }
        shape.written.extend(pointee.written);
        shape.labels.extend(read.labels);
        shape.labels.extend(pointee.labels);
        Ok(shape)
    }

    fn path(&mut self, ty: &Type, path: &syn::Path, cx: &Cx) -> syn::Result<Shape> {
        if let Some(ident) = path.get_ident() {
            return Ok(match self.vars.lookup(ident) {
                Some((VarKind::Ty, _)) => self.type_var(ident, cx),
                // An effect, a length, an identity or the runtime has no
                // position, and a type written with no arguments names no
                // lifetime: `elided_lifetimes_in_paths` refuses a hidden one.
                Some(_) | None => Shape::positionless(),
            });
        }
        let Some(last) = path.segments.last() else {
            return self.unread(ty, cx);
        };
        let PathArguments::AngleBracketed(args) = &last.arguments else {
            return match path.segments.iter().all(|s| s.arguments.is_none()) {
                true => Ok(Shape::positionless()),
                false => self.unread(ty, cx),
            };
        };
        let lifetimes: Vec<&Lifetime> = args
            .args
            .iter()
            .filter_map(|a| match a {
                GenericArgument::Lifetime(lt) => Some(lt),
                _ => None,
            })
            .collect();
        let types: Vec<&Type> = args
            .args
            .iter()
            .filter_map(|a| match a {
                GenericArgument::Type(t) => Some(t),
                _ => None,
            })
            .collect();
        let only_types = lifetimes.is_empty() && types.len() == args.args.len();
        let only_written = lifetimes.len() + types.len() == args.args.len();
        match (last.ident.to_string().as_str(), lifetimes.as_slice(), types.as_slice()) {
            ("Option", _, [some]) if only_types => Ok(self.walk(some, cx)?.wrapped(Laid::Option)),
            ("Result", _, [ok, err]) if only_types => {
                let parts = vec![self.walk(ok, cx)?, self.walk(err, cx)?];
                Ok(Shape::concat(parts, |mut laid| {
                    let err = laid.pop().expect("a result has two parts");
                    let ok = laid.pop().expect("a result has two parts");
                    Laid::Result(Box::new(ok), Box::new(err))
                }))
            }
            ("Closure", [captures], [handed, returned, rest @ ..]) => {
                self.closure(captures, handed, returned, rest, cx)
            }
            ("Ref" | "Slice", [brand], [pointee, loan, rest @ ..]) => {
                self.carrier(brand, pointee, loan, rest, cx)
            }
            (_, lifetimes, types) if self.laying_out && only_written => {
                match self.laid_out(ty, lifetimes, types)? {
                    Some(candidate) => self.extension(candidate, lifetimes, types, cx),
                    None => self.unread(ty, cx),
                }
            }
            _ => self.unread(ty, cx),
        }
    }

    /// Whether this reading lays out the generic type `ty`, and which
    /// candidate it is; discovery records it the first time it is met.
    fn laid_out(&mut self, ty: &Type, lifetimes: &[&Lifetime], types: &[&Type]) -> syn::Result<Option<usize>> {
        let at: *const Type = ty;
        let known = self.candidates.iter().position(|candidate| candidate.at == at);
        match (self.reading, known) {
            (Reading::Discovery, Some(index)) => Ok(Some(index)),
            (Reading::Discovery, None) => {
                let written = lifetimes
                    .iter()
                    .map(|_| WrittenArg::Lifetime)
                    .chain(types.iter().map(|arg| self.written_arg(arg)))
                    .collect();
                self.candidates.push(Candidate {
                    at,
                    ty: ty.clone(),
                    written,
                });
                Ok(Some(self.candidates.len() - 1))
            }
            (Reading::Assigned { laid }, Some(index)) => Ok(laid[index].then_some(index)),
            (Reading::Assigned { .. }, None) => Err(syn::Error::new_spanned(
                ty,
                "#[extern_fn] met a generic type its discovery reading did not: a reading that \
                 lays out every generic type meets each one another reading meets",
            )),
        }
    }

    /// A lifetime or a type variable, a reference or a generic type written
    /// anywhere in `arg` gives it positions, and the type it stands at must
    /// be a type parameter.
    fn written_arg(&self, arg: &Type) -> WrittenArg {
        if let Type::Path(p) = arg
            && p.qself.is_none()
            && let Some((kind, _)) = p.path.get_ident().and_then(|ident| self.vars.lookup(ident))
        {
            return match kind {
                VarKind::Ty => WrittenArg::Positioned,
                VarKind::Effect | VarKind::Len | VarKind::Identity | VarKind::Runtime => WrittenArg::NonType,
            };
        }
        let mut found = Positions {
            vars: self.vars,
            any: false,
        };
        found.visit_type(arg);
        match found.any {
            true => WrittenArg::Positioned,
            false => WrittenArg::Positionless,
        }
    }

    /// An extension type, read as its region parameters, then its type
    /// arguments, in declaration order (RFC-0096 rule 2): a lifetime is one
    /// segment, a type argument laid out is its segments, and one that is
    /// not is one segment of every label it names. An argument that is an
    /// effect, a length, an identity or the runtime has no position. What
    /// the callee may write through it is what it could through an unread
    /// type: every label it names, and what the call lends a closure it
    /// may hold, so writes stay the union (RFC-0096 rule 3).
    fn extension(&mut self, candidate: usize, lifetimes: &[&Lifetime], types: &[&Type], cx: &Cx) -> syn::Result<Shape> {
        let mut shape = Shape::default();
        let mut segments = Vec::new();
        let mut args = Vec::new();
        for lt in lifetimes {
            let read = self.lifetime(Some(lt), lt.span(), cx)?;
            segments.push(Segment::Life(read.own));
            shape.labels.extend(read.labels);
        }
        for (index, arg) in types.iter().enumerate() {
            let written_at = lifetimes.len() + index;
            if let WrittenArg::NonType = self.candidates[candidate].written[written_at] {
                continue;
            }
            let part = self.walk(arg, cx)?;
            let laid = match part.layout {
                Some(layout) => {
                    segments.extend(layout.segments);
                    layout.laid
                }
                None => {
                    segments.push(Segment::Unread(part.labels.clone()));
                    Laid::Unread
                }
            };
            args.push(ReadArg { written_at, laid });
            shape.labels.extend(part.labels);
            shape.written.extend(part.written);
        }
        shape.written.extend(shape.labels.iter().cloned());
        shape.written.extend(cx.end.map(Label::LentTo));
        shape.layout = Some(Layout {
            laid: Laid::User {
                candidate,
                regions: lifetimes.len(),
                args,
            },
            segments,
        });
        Ok(shape)
    }

    fn type_var(&mut self, ident: &syn::Ident, cx: &Cx) -> Shape {
        let var = Label::Var(ident.to_string());
        let mut shape = Shape {
            layout: Some(Layout {
                laid: Laid::Var,
                segments: vec![Segment::Var(var.clone())],
            }),
            ..Shape::default()
        };
        shape.labels.insert(var.clone());
        if cx.access == Access::Mut {
            shape.written.insert(var);
        }
        shape
    }

    /// `Closure<'k, A, R, ..>`: one position, its captures at `'k`. What it
    /// is handed, `A` and its captures, is written into by the call; what
    /// it returns takes from both.
    fn closure(
        &mut self,
        captures: &Lifetime,
        handed: &Type,
        returned: &Type,
        rest: &[&Type],
        cx: &Cx,
    ) -> syn::Result<Shape> {
        let captures = self.lifetime(Some(captures), captures.span(), cx)?.labels;
        let handed = self.walk(
            handed,
            &Cx {
                brands: Vec::new(),
                in_closure_args: true,
                ..Cx::top(cx.end)
            },
        )?;
        let returned = self.walk(
            returned,
            &Cx {
                brands: captures.clone(),
                ..Cx::top(cx.end)
            },
        )?;
        let rest = self.labels_of(rest, cx)?;
        for from in handed.labels.iter().chain(&captures).chain(&rest) {
            for to in &returned.labels {
                self.edge(from, to);
            }
        }
        let mut shape = Shape::default();
        shape.labels.extend(captures.iter().cloned());
        shape.labels.extend(rest.iter().cloned());
        shape.written.extend(handed.labels);
        shape.written.extend(handed.written);
        shape.written.extend(captures);
        shape.written.extend(rest);
        Ok(shape)
    }

    /// `Ref<'r, T, M, ..>` or `Slice<'r, T, M, ..>`: a loan at `'r`, which
    /// the callee writes through unless `M` is `Shared`.
    fn carrier(
        &mut self,
        brand: &Lifetime,
        pointee: &Type,
        loan: &Type,
        rest: &[&Type],
        cx: &Cx,
    ) -> syn::Result<Shape> {
        let brand = self.lifetime(Some(brand), brand.span(), cx)?.labels;
        let shared = matches!(loan, Type::Path(p)
            if p.qself.is_none() && p.path.segments.last().is_some_and(|s| s.ident == "Shared"));
        let access = match (cx.access, shared) {
            (Access::Shared, _) | (_, true) => Access::Shared,
            (Access::Owned | Access::Mut, false) => Access::Mut,
        };
        let mut pointee_cx = Cx { access, ..cx.clone() };
        pointee_cx.brands.extend(brand.iter().cloned());
        let pointee = self.walk(pointee, &pointee_cx)?;
        let rest = self.labels_of(rest, cx)?;
        self.outlive(&pointee.labels, &brand);
        let mut shape = Shape::default();
        if access == Access::Mut {
            shape.written.extend(pointee.labels.iter().cloned());
        }
        shape.written.extend(pointee.written);
        shape.labels.extend(brand);
        shape.labels.extend(pointee.labels);
        shape.labels.extend(rest);
        Ok(shape)
    }

    fn labels_of(&mut self, tys: &[&Type], cx: &Cx) -> syn::Result<BTreeSet<Label>> {
        let mut labels = BTreeSet::new();
        for ty in tys {
            labels.extend(self.unread(ty, cx)?.labels);
        }
        Ok(labels)
    }

    /// A type whose positions the macro does not read: it may hold a
    /// closure joining any of its labels, and lent to by the call.
    fn unread(&mut self, ty: &Type, cx: &Cx) -> syn::Result<Shape> {
        let mut named = Named {
            vars: self.vars,
            lifetimes: Vec::new(),
            elided: Vec::new(),
            type_vars: BTreeSet::new(),
        };
        named.visit_type(ty);
        let mut labels = BTreeSet::new();
        for lt in &named.lifetimes {
            labels.extend(self.lifetime(Some(lt), lt.span(), cx)?.labels);
        }
        for span in named.elided {
            labels.extend(self.lifetime(None, span, cx)?.labels);
        }
        labels.extend(named.type_vars.into_iter().map(Label::Var));
        for from in &labels {
            for to in &labels {
                self.edge(from, to);
            }
        }
        let mut shape = Shape::default();
        shape.written.extend(labels.iter().cloned());
        shape.written.extend(cx.end.map(Label::LentTo));
        shape.labels = labels;
        Ok(shape)
    }

    /// Every lifetime in `pointee` outlives each of `outer`.
    fn outlive(&mut self, pointee: &BTreeSet<Label>, outer: &[Label]) {
        for from in pointee.iter().filter(|l| matches!(l, Label::Life(_))) {
            for to in outer {
                self.edge(from, to);
            }
        }
    }

    fn edge(&mut self, from: &Label, to: &Label) {
        if from != to {
            self.edges.push(Edge {
                from: from.clone(),
                to: to.clone(),
            });
        }
    }
}

struct Named<'v> {
    vars: &'v Vars,
    lifetimes: Vec<Lifetime>,
    elided: Vec<Span>,
    type_vars: BTreeSet<String>,
}

impl<'ast> Visit<'ast> for Named<'_> {
    fn visit_lifetime(&mut self, lt: &'ast Lifetime) {
        self.lifetimes.push(lt.clone());
    }

    fn visit_type_reference(&mut self, r: &'ast syn::TypeReference) {
        if r.lifetime.is_none() {
            self.elided.push(r.and_token.span);
        }
        syn::visit::visit_type_reference(self, r);
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        if let Some(ident) = path.get_ident()
            && let Some((VarKind::Ty, _)) = self.vars.lookup(ident)
        {
            self.type_vars.insert(ident.to_string());
        }
        syn::visit::visit_path(self, path);
    }
}

/// Whether a written type names a lifetime, a type variable, a reference
/// or a generic type.
struct Positions<'v> {
    vars: &'v Vars,
    any: bool,
}

impl<'ast> Visit<'ast> for Positions<'_> {
    fn visit_lifetime(&mut self, _: &'ast Lifetime) {
        self.any = true;
    }

    fn visit_type_reference(&mut self, _: &'ast syn::TypeReference) {
        self.any = true;
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        let generic = path.segments.iter().any(|segment| !segment.arguments.is_none());
        let variable = path
            .get_ident()
            .is_some_and(|ident| self.vars.lookup(ident).is_some());
        self.any |= generic || variable;
        syn::visit::visit_path(self, path);
    }

    fn visit_type(&mut self, ty: &'ast Type) {
        match ty {
            Type::Path(_) | Type::Reference(_) | Type::Tuple(_) | Type::Array(_) | Type::Slice(_)
            | Type::Paren(_) | Type::Group(_) | Type::Never(_) => syn::visit::visit_type(self, ty),
            _ => self.any = true,
        }
    }
}

/// A reading is run per assignment of laid out or unread to every
/// candidate, so a signature naming more generic types than this is refused
/// rather than read in exponentially many passes.
const MOST_CANDIDATES: usize = 12;

/// What `derive` reads off a signature: the generic types its ends name,
/// and the flows under each assignment of a reading to them.
pub struct Derived {
    candidates: Vec<Candidate>,
    readings: Vec<ReadFlows>,
}

/// The flows of the declaration whose Rust signature is `sig`, the role of
/// each of its inputs in `roles`.
pub fn derive(sig: &syn::Signature, roles: &[Role], vars: &Vars, ret: &Type) -> syn::Result<Derived> {
    let Read { candidates, .. } = read(sig, roles, vars, ret, &Reading::Discovery, Vec::new())?;
    if candidates.len() > MOST_CANDIDATES {
        return Err(syn::Error::new(
            sig.ident.span(),
            format!(
                "the parameters and the result name {} generic types, and #[extern_fn] reads at \
                 most {MOST_CANDIDATES}: each states whether it is laid out (RFC-0096 rule 2), and \
                 the macro reads the signature once per assignment",
                candidates.len()
            ),
        ));
    }
    let mut readings = Vec::new();
    for bits in 0..1usize << candidates.len() {
        let laid: Vec<bool> = (0..candidates.len()).map(|k| bits & (1 << k) != 0).collect();
        let reading = Reading::Assigned { laid: laid.clone() };
        let Read { flows, .. } = read(sig, roles, vars, ret, &reading, candidates.clone())?;
        readings.push(ReadFlows { laid, flows });
    }
    Ok(Derived { candidates, readings })
}

fn read(
    sig: &syn::Signature,
    roles: &[Role],
    vars: &Vars,
    ret: &Type,
    reading: &Reading,
    candidates: Vec<Candidate>,
) -> syn::Result<Read> {
    let mut walker = Walker {
        vars,
        edges: declared_outlives(&sig.generics),
        elided_count: 0,
        side: Side::Params,
        named_by_params: BTreeSet::new(),
        named_by_ctx: BTreeSet::new(),
        carried: BTreeSet::new(),
        reading_ctx: false,
        named_by_result: Vec::new(),
        laying_out: false,
        reading,
        candidates,
    };
    let mut ends: BTreeMap<usize, Shape> = BTreeMap::new();
    for (arg, role) in sig.inputs.iter().zip(roles) {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(arg, "an extern_fn has no self parameter"));
        };
        let end = match role {
            Role::Acvus(index) => Some(*index),
            Role::Ctx | Role::Other => None,
        };
        walker.reading_ctx = *role == Role::Ctx;
        walker.laying_out = end.is_some();
        let shape = walker.walk(&pat_type.ty, &Cx::top(end))?;
        if let Some(index) = end {
            ends.insert(index, shape);
        }
    }
    let taken = match walker.named_by_params.len() {
        1 => walker.named_by_params.first().cloned(),
        _ => None,
    };
    walker.side = Side::Result(taken);
    walker.laying_out = true;
    let result = walker.walk(ret, &Cx::top(None))?;
    let handed: BTreeSet<Label> = ends.values().flat_map(|end| end.labels.iter().cloned()).collect();
    for index in ends.keys() {
        for from in &handed {
            walker.edge(from, &Label::LentTo(*index));
        }
    }
    let reach = Reach::of(&walker.edges);
    let ctx_only: Vec<Label> = walker
        .named_by_ctx
        .difference(&walker.carried)
        .map(|name| Label::Life(name.clone()))
        .collect();
    if let Some(lent_by_ctx) = walker.named_by_result.iter().find(|lt| {
        let named = Label::Life(lt.name.clone());
        ctx_only.iter().any(|c| reach.reaches(c, &named))
    }) {
        let name = &lent_by_ctx.name;
        return Err(syn::Error::new(
            lent_by_ctx.span,
            format!(
                "the result names the lifetime `{name}`, which only `ctx` lends: a result \
                 borrows what an acvus parameter lends it, and `ctx` is none, so no flow can \
                 say what `{name}` holds (RFC-0079 rule 6). Name `{name}` on the parameter \
                 the result borrows, or return an owned value."
            ),
        ));
    }

    let mut flows = Vec::new();
    for (from, input) in &ends {
        if reach.any(&input.labels, &result.labels) {
            flows.push(Flow {
                to: End::Result,
                from: *from,
                alignment: alignment(&reach, input.layout.as_ref(), result.layout.as_ref()),
            });
        }
    }
    for (to, output) in &ends {
        for (from, input) in &ends {
            if reach.any(&input.labels, &output.written) {
                flows.push(Flow {
                    to: End::Param(*to),
                    from: *from,
                    alignment: Alignment::Any,
                });
            }
        }
    }
    Ok(Read {
        flows,
        candidates: walker.candidates,
    })
}

impl Derived {
    /// The flows as a `FlowTerm<Poly>` expression, each generic type named
    /// at `comp`'s form of it: the flows of the reading its statements
    /// assign, beside a compile-time check that each statement admits the
    /// arguments the laid-out reading read.
    pub fn tokens(&self, comp: impl Fn(&Type) -> Type) -> TokenStream {
        if let [ReadFlows { flows, .. }] = self.readings.as_slice()
            && self.candidates.is_empty()
        {
            return tokens(flows, &[]);
        }
        let statements: Vec<TokenStream> = self
            .candidates
            .iter()
            .map(|candidate| {
                let ty = comp(&candidate.ty);
                quote! { <#ty as ::acvus_extern::TyArg>::LAYOUT }
            })
            .collect();
        let admitted = self.candidates.iter().zip(&statements).map(|(candidate, statement)| {
            let written = candidate.written.iter().map(|arg| match arg {
                WrittenArg::Lifetime => quote! { ::acvus_extern::laid::WrittenArg::Lifetime },
                WrittenArg::Positioned => quote! { ::acvus_extern::laid::WrittenArg::Positioned },
                WrittenArg::NonType => quote! { ::acvus_extern::laid::WrittenArg::NonType },
                WrittenArg::Positionless => quote! { ::acvus_extern::laid::WrittenArg::Positionless },
            });
            let ty = &candidate.ty;
            let refused = format!(
                "`{}` states a layout whose parameters are not the arguments written here (RFC-0096 rule 2)",
                quote! { #ty }
            );
            quote! {
                const { ::core::assert!(#statement.admits(&[#(#written),*]), #refused) };
            }
        });
        let mut arms: Vec<Arm> = Vec::new();
        for ReadFlows { laid, flows } in &self.readings {
            let flows = tokens(flows, &statements);
            let pattern = quote! { [#(#laid),*] };
            let rendered = flows.to_string();
            match arms.iter_mut().find(|arm| arm.rendered == rendered) {
                Some(arm) => arm.patterns.push(pattern),
                None => arms.push(Arm {
                    flows,
                    rendered,
                    patterns: vec![pattern],
                }),
            }
        }
        let arms = arms.into_iter().map(|Arm { flows, patterns, .. }| quote! { #(#patterns)|* => #flows, });
        quote! {{
            #(#admitted)*
            match [#(#statements.is_laid_out()),*] {
                #(#arms)*
            }
        }}
    }
}

fn declared_outlives(generics: &syn::Generics) -> Vec<Edge> {
    let life = |lt: &Lifetime| Label::Life(format!("'{}", lt.ident));
    let mut edges = Vec::new();
    for param in &generics.params {
        if let GenericParam::Lifetime(def) = param {
            for bound in &def.bounds {
                edges.push(Edge {
                    from: life(&def.lifetime),
                    to: life(bound),
                });
            }
        }
    }
    for predicate in generics.where_clause.iter().flat_map(|w| &w.predicates) {
        if let WherePredicate::Lifetime(p) = predicate {
            for bound in &p.bounds {
                edges.push(Edge {
                    from: life(&p.lifetime),
                    to: life(bound),
                });
            }
        }
    }
    edges
}

struct Reach {
    beyond: BTreeMap<Label, BTreeSet<Label>>,
}

impl Reach {
    fn of(edges: &[Edge]) -> Self {
        let mut next: BTreeMap<&Label, Vec<&Label>> = BTreeMap::new();
        for edge in edges {
            next.entry(&edge.from).or_default().push(&edge.to);
        }
        let mut beyond = BTreeMap::new();
        for start in next.keys() {
            let mut seen: BTreeSet<Label> = BTreeSet::new();
            let mut stack: Vec<&Label> = vec![start];
            while let Some(at) = stack.pop() {
                for to in next.get(at).into_iter().flatten() {
                    if seen.insert((*to).clone()) {
                        stack.push(to);
                    }
                }
            }
            beyond.insert((*start).clone(), seen);
        }
        Reach { beyond }
    }

    fn reaches(&self, from: &Label, to: &Label) -> bool {
        if from.is_static() || to.is_static() {
            return false;
        }
        from == to || self.beyond.get(from).is_some_and(|seen| seen.contains(to))
    }

    fn any(&self, from: &BTreeSet<Label>, to: &BTreeSet<Label>) -> bool {
        from.iter().any(|f| to.iter().any(|t| self.reaches(f, t)))
    }
}

/// RFC-0096 rule 1, where both ends are laid out: each output segment takes
/// the input segments whose labels reach its own, position by position
/// where both are one type variable's (the label `(T, k)`). `Aligned` is
/// the map that takes segment `k` from segment `k` alone, each a lifetime
/// or one variable; `Any` the one that takes every segment into every
/// segment, none position by position; `Labelled` any other. `Any` where
/// an end is not laid out.
fn alignment(reach: &Reach, input: Option<&Layout>, output: Option<&Layout>) -> Alignment {
    let (Some(input), Some(output)) = (input, output) else {
        return Alignment::Any;
    };
    let takes: Vec<Vec<Take>> = output
        .segments
        .iter()
        .map(|to| {
            let to_labels = to.labels();
            input
                .segments
                .iter()
                .enumerate()
                .filter(|(_, from)| reach.any(&from.labels(), &to_labels))
                .map(|(segment, from)| Take {
                    segment,
                    aligned: matches!((from, to), (Segment::Var(a), Segment::Var(b)) if a == b),
                })
                .collect()
        })
        .collect();
    let one_to_one = input.segments.len() == output.segments.len()
        && takes.iter().zip(&output.segments).enumerate().all(|(k, (takes, to))| {
            let from = &input.segments[k];
            takes.as_slice() == [Take { segment: k, aligned: matches!(from, Segment::Var(_)) }]
                && match (from, to) {
                    (Segment::Life(_), Segment::Life(_)) => true,
                    (Segment::Var(a), Segment::Var(b)) => a == b,
                    _ => false,
                }
        });
    let every_to_every = takes.iter().all(|takes| {
        takes.len() == input.segments.len() && takes.iter().all(|take| !take.aligned)
    });
    match (one_to_one, every_to_every) {
        (true, _) => Alignment::Aligned,
        (false, true) => Alignment::Any,
        (false, false) => Alignment::Labelled(Labelled {
            to: output.laid.clone(),
            from: input.laid.clone(),
            takes,
        }),
    }
}

fn laid_tokens(laid: &Laid, statements: &[TokenStream]) -> TokenStream {
    let boxed = |inner: &Laid| {
        let inner = laid_tokens(inner, statements);
        quote! { ::std::boxed::Box::new(#inner) }
    };
    match laid {
        Laid::NoPosition => quote! { ::acvus_extern::Laid::NoPosition },
        Laid::Ref(inner) => {
            let inner = boxed(inner);
            quote! { ::acvus_extern::Laid::Ref(#inner) }
        }
        Laid::Option(inner) => {
            let inner = boxed(inner);
            quote! { ::acvus_extern::Laid::Option(#inner) }
        }
        Laid::Result(ok, err) => {
            let ok = boxed(ok);
            let err = boxed(err);
            quote! { ::acvus_extern::Laid::Result(#ok, #err) }
        }
        Laid::Tuple(parts) => {
            let parts = parts.iter().map(|part| laid_tokens(part, statements));
            quote! { ::acvus_extern::Laid::Tuple(::std::vec![#(#parts),*]) }
        }
        Laid::Array(inner) => {
            let inner = boxed(inner);
            quote! { ::acvus_extern::Laid::Array(#inner) }
        }
        Laid::User {
            candidate,
            regions,
            args,
        } => {
            let statement = &statements[*candidate];
            let args = args.iter().map(|ReadArg { written_at, laid }| {
                let laid = laid_tokens(laid, statements);
                quote! { ::acvus_extern::laid::ReadArg { written_at: #written_at, laid: #laid } }
            });
            quote! {
                ::acvus_extern::Laid::User {
                    regions: #regions,
                    args: #statement.type_args(::std::vec![#(#args),*]),
                }
            }
        }
        Laid::Var => quote! { ::acvus_extern::Laid::Var },
        Laid::Unread => quote! { ::acvus_extern::Laid::Unread },
    }
}

fn alignment_tokens(alignment: &Alignment, statements: &[TokenStream]) -> TokenStream {
    match alignment {
        Alignment::Aligned => quote! { ::acvus_extern::Alignment::Aligned },
        Alignment::Any => quote! { ::acvus_extern::Alignment::Any },
        Alignment::Labelled(labelled) => {
            let to = laid_tokens(&labelled.to, statements);
            let from = laid_tokens(&labelled.from, statements);
            let takes = labelled.takes.iter().map(|takes| {
                let takes = takes.iter().map(|Take { segment, aligned }| {
                    quote! { ::acvus_extern::Take { segment: #segment, aligned: #aligned } }
                });
                quote! { ::std::vec![#(#takes),*] }
            });
            quote! {
                ::acvus_extern::Alignment::Labelled(::std::boxed::Box::new(::acvus_extern::Labelled {
                    to: #to,
                    from: #from,
                    takes: ::std::vec![#(#takes),*],
                }))
            }
        }
    }
}

fn tokens(flows: &[Flow], statements: &[TokenStream]) -> TokenStream {
    if flows.is_empty() {
        return quote! { ::acvus_extern::Flows::none().into() };
    }
    let flows = flows.iter().map(|flow| {
        let to = match flow.to {
            End::Result => quote! { ::acvus_extern::FlowEnd::Result },
            End::Param(index) => quote! { ::acvus_extern::FlowEnd::Param(#index) },
        };
        let from = flow.from;
        let alignment = alignment_tokens(&flow.alignment, statements);
        quote! {
            ::acvus_extern::Flow {
                to: #to,
                from: ::acvus_extern::FlowEnd::Param(#from),
                alignment: #alignment,
            }
        }
    });
    quote! { ::acvus_extern::Flows::of([#(#flows),*]).into() }
}

/// The signature again under `deny(elided_lifetimes_in_paths)`, so every
/// lifetime `derive` reads is written: `Ref<'_, T, ..>`, not `Ref<T, ..>`.
pub fn lifetimes_written(sig: &syn::Signature) -> TokenStream {
    let mut probe = sig.clone();
    probe.ident = format_ident!("__lifetimes_written_{}", sig.ident);
    probe.asyncness = None;
    for arg in probe.inputs.iter_mut() {
        if let FnArg::Typed(pat_type) = arg {
            *pat_type.pat = syn::parse_quote! { _ };
            pat_type.attrs.clear();
        }
    }
    quote! {
        #[doc(hidden)]
        #[deny(elided_lifetimes_in_paths)]
        #[allow(dead_code)]
        #probe {
            ::core::unreachable!()
        }
    }
}
