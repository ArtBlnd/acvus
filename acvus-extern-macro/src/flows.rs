//! An extern's flows, read off its Rust signature (RFC-0079 rule 6).
//!
//! Each lifetime and type variable a signature names is a label. An output
//! end flows from an input end where a label of the input reaches a label
//! of the output: the same label; a lifetime it outlives, written
//! (`'a: 'b`) or implied (`&'b T` for every lifetime in `T`); a closure's
//! result, from what the closure captured and is handed; and, inside a type
//! the macro does not read or an `Instance`, every label beside it.

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
    /// A `#[state]` or an `Instance`: no end.
    Other,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    Owned,
    Mut,
    Shared,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Alignment {
    Aligned,
    Any,
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
    /// A label per position, a type variable's standing for all of its;
    /// `None` where the positions are not the macro's to read.
    layout: Option<Vec<Label>>,
}

impl Shape {
    fn positionless() -> Self {
        Shape {
            layout: Some(Vec::new()),
            ..Shape::default()
        }
    }

    fn concat(parts: Vec<Shape>) -> Self {
        let mut shape = Shape::positionless();
        for part in parts {
            shape.labels.extend(part.labels);
            shape.written.extend(part.written);
            shape.layout = match (shape.layout, part.layout) {
                (Some(mut a), Some(b)) => {
                    a.extend(b);
                    Some(a)
                }
                (None, _) | (_, None) => None,
            };
        }
        shape
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
            Type::Array(a) => self.walk(&a.elem, cx),
            Type::Slice(s) => self.walk(&s.elem, cx),
            Type::Tuple(t) => {
                let parts = t
                    .elems
                    .iter()
                    .map(|elem| self.walk(elem, cx))
                    .collect::<syn::Result<_>>()?;
                Ok(Shape::concat(parts))
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
            layout: pointee
                .layout
                .map(|pointee| std::iter::once(read.own).chain(pointee).collect()),
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
        match (last.ident.to_string().as_str(), lifetimes.as_slice(), types.as_slice()) {
            ("Option", _, [some]) if only_types => self.walk(some, cx),
            ("Result", _, [ok, err]) if only_types => {
                let parts = vec![self.walk(ok, cx)?, self.walk(err, cx)?];
                Ok(Shape::concat(parts))
            }
            ("Closure", [captures], [handed, returned, rest @ ..]) => {
                self.closure(captures, handed, returned, rest, cx)
            }
            ("Ref" | "Slice", [brand], [pointee, loan, rest @ ..]) => {
                self.carrier(brand, pointee, loan, rest, cx)
            }
            _ => self.unread(ty, cx),
        }
    }

    fn type_var(&mut self, ident: &syn::Ident, cx: &Cx) -> Shape {
        let var = Label::Var(ident.to_string());
        let mut shape = Shape {
            layout: Some(vec![var.clone()]),
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

/// The flows of the declaration whose Rust signature is `sig`, the role of
/// each of its inputs in `roles`, as a `FlowTerm<Poly>` expression.
pub fn derive(sig: &syn::Signature, roles: &[Role], vars: &Vars, ret: &Type) -> syn::Result<TokenStream> {
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
                alignment: alignment(&reach, input.layout.as_deref(), result.layout.as_deref()),
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
    Ok(tokens(&flows))
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

/// `Aligned` where both ends are laid out, as many positions long, and a
/// label reaches another only at its own position and of its own width (a
/// lifetime one position, a type variable its own); `Any` otherwise.
fn alignment(reach: &Reach, input: Option<&[Label]>, output: Option<&[Label]>) -> Alignment {
    let (Some(input), Some(output)) = (input, output) else {
        return Alignment::Any;
    };
    let position_to_position = input.len() == output.len()
        && input.iter().enumerate().all(|(k, from)| {
            output.iter().enumerate().all(|(j, to)| {
                !reach.reaches(from, to)
                    || (k == j
                        && match (from, to) {
                            (Label::Life(_), Label::Life(_)) => true,
                            (Label::Var(a), Label::Var(b)) => a == b,
                            _ => false,
                        })
            })
        });
    match position_to_position {
        true => Alignment::Aligned,
        false => Alignment::Any,
    }
}

fn tokens(flows: &[Flow]) -> TokenStream {
    if flows.is_empty() {
        return quote! { ::acvus_extern::Flows::none().into() };
    }
    let flows = flows.iter().map(|flow| {
        let to = match flow.to {
            End::Result => quote! { ::acvus_extern::FlowEnd::Result },
            End::Param(index) => quote! { ::acvus_extern::FlowEnd::Param(#index) },
        };
        let from = flow.from;
        let alignment = match flow.alignment {
            Alignment::Aligned => quote! { ::acvus_extern::Alignment::Aligned },
            Alignment::Any => quote! { ::acvus_extern::Alignment::Any },
        };
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
