//! `#[extern_fn(means(..))]`: RFC-0104's term, parsed into
//! `acvus_extern::means`.

use std::cell::Cell;

use proc_macro2::TokenStream;
use quote::quote;
use syn::parse::ParseStream;
use syn::{Ident, Lit, Path, Token, Type};

use crate::reaches::ReachesAttr;
use crate::{ExternParam, Mode};

pub(crate) struct MeansAttr {
    keyword: Ident,
    stmts: Vec<Stmt>,
    result: Expr,
}

enum Stmt {
    Let { name: Ident, value: Expr },
    Store { entry: EntryName, value: Expr },
}

struct EntryName {
    table: Ident,
    key: Ident,
}

enum Expr {
    Name(Ident),
    Lent(Ident),
    Entry(EntryName),
    LendEntry { mutable: bool, entry: EntryName },
    Some(Box<Expr>),
    None,
    Lit { negative: bool, lit: Lit },
    Not(Box<Expr>),
    Neg(Box<Expr>),
    Binary {
        op: Op,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Array(Vec<Expr>),
    Call { callee: Path, args: Vec<Expr> },
    Match {
        scrutinee: Box<Expr>,
        some_binder: Option<Ident>,
        then: Box<Expr>,
        none: Box<Expr>,
    },
}

const TERMS: &str = "a term is a parameter taken by value, `*x`, the entry `x[k]`, `&x[k]` or \
                     `&mut x[k]`, `Some(t)`, `None`, a constant, `!t`, `-t`, `t op t` for one \
                     of the language's operators, `[t, ..]`, a call of a registered extern, or \
                     `match t { Some(b) => t, None => t }` (RFC-0104 rule 1)";

/// The language's binary operators a term writes, by the precedence the
/// language parses them at (RFC-0104 rule 1).
#[derive(Clone, Copy)]
enum Op {
    Or,
    And,
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

impl Op {
    fn precedence(self) -> u8 {
        match self {
            Op::Or => 1,
            Op::And => 2,
            Op::Eq | Op::Neq | Op::Lt | Op::Lte | Op::Gt | Op::Gte => 3,
            Op::Add | Op::Sub => 4,
            Op::Mul | Op::Div | Op::Mod => 5,
        }
    }

    /// The operator at the head of `input`, not consumed.
    fn peek(input: ParseStream) -> Option<Op> {
        Some(if input.peek(Token![||]) {
            Op::Or
        } else if input.peek(Token![&&]) {
            Op::And
        } else if input.peek(Token![==]) {
            Op::Eq
        } else if input.peek(Token![!=]) {
            Op::Neq
        } else if input.peek(Token![<=]) {
            Op::Lte
        } else if input.peek(Token![>=]) {
            Op::Gte
        } else if input.peek(Token![<]) {
            Op::Lt
        } else if input.peek(Token![>]) {
            Op::Gt
        } else if input.peek(Token![+]) {
            Op::Add
        } else if input.peek(Token![-]) {
            Op::Sub
        } else if input.peek(Token![*]) {
            Op::Mul
        } else if input.peek(Token![/]) {
            Op::Div
        } else if input.peek(Token![%]) {
            Op::Mod
        } else {
            return None;
        })
    }

    fn consume(self, input: ParseStream) -> syn::Result<()> {
        match self {
            Op::Or => input.parse::<Token![||]>().map(drop),
            Op::And => input.parse::<Token![&&]>().map(drop),
            Op::Eq => input.parse::<Token![==]>().map(drop),
            Op::Neq => input.parse::<Token![!=]>().map(drop),
            Op::Lte => input.parse::<Token![<=]>().map(drop),
            Op::Gte => input.parse::<Token![>=]>().map(drop),
            Op::Lt => input.parse::<Token![<]>().map(drop),
            Op::Gt => input.parse::<Token![>]>().map(drop),
            Op::Add => input.parse::<Token![+]>().map(drop),
            Op::Sub => input.parse::<Token![-]>().map(drop),
            Op::Mul => input.parse::<Token![*]>().map(drop),
            Op::Div => input.parse::<Token![/]>().map(drop),
            Op::Mod => input.parse::<Token![%]>().map(drop),
        }
    }

    fn tokens(self) -> TokenStream {
        let op = match self {
            Op::Or => quote! { Or },
            Op::And => quote! { And },
            Op::Eq => quote! { Eq },
            Op::Neq => quote! { Neq },
            Op::Lt => quote! { Lt },
            Op::Lte => quote! { Lte },
            Op::Gt => quote! { Gt },
            Op::Gte => quote! { Gte },
            Op::Add => quote! { Add },
            Op::Sub => quote! { Sub },
            Op::Mul => quote! { Mul },
            Op::Div => quote! { Div },
            Op::Mod => quote! { Mod },
        };
        quote! { ::acvus_extern::means::BinOp::#op }
    }
}

impl MeansAttr {
    pub(crate) fn parse_after(keyword: Ident, input: ParseStream) -> syn::Result<Self> {
        let content;
        syn::parenthesized!(content in input);
        let mut stmts = Vec::new();
        loop {
            if content.peek(Token![let]) {
                content.parse::<Token![let]>()?;
                let name: Ident = content.parse()?;
                content.parse::<Token![=]>()?;
                let value = parse_expr(&content)?;
                content.parse::<Token![;]>()?;
                stmts.push(Stmt::Let { name, value });
                continue;
            }
            if content.peek(Ident)
                && content.peek2(syn::token::Bracket)
                && stores_after_entry(&content)
            {
                let entry = parse_entry(&content)?;
                content.parse::<Token![=]>()?;
                let value = parse_expr(&content)?;
                content.parse::<Token![;]>()?;
                stmts.push(Stmt::Store { entry, value });
                continue;
            }
            break;
        }
        let result = parse_expr(&content)?;
        if !content.is_empty() {
            return Err(syn::Error::new(
                content.span(),
                "a term's block is `let` bindings and stores `x[k] = t;`, each ending in `;`, \
                 then the term the call returns (RFC-0104 rule 1)",
            ));
        }
        Ok(MeansAttr {
            keyword,
            stmts,
            result,
        })
    }
}

fn stores_after_entry(input: ParseStream) -> bool {
    let fork = input.fork();
    parse_entry(&fork).is_ok() && fork.peek(Token![=]) && !fork.peek(Token![==])
}

fn parse_entry(input: ParseStream) -> syn::Result<EntryName> {
    let table: Ident = input.parse()?;
    let key;
    syn::bracketed!(key in input);
    let named: Ident = key.parse()?;
    if !key.is_empty() {
        return Err(syn::Error::new(key.span(), "an entry is `x[k]`, `k` a parameter"));
    }
    Ok(EntryName { table, key: named })
}

fn parse_expr(input: ParseStream) -> syn::Result<Expr> {
    parse_binary(input, 0)
}

/// Operators of precedence above `floor`, left-associative.
fn parse_binary(input: ParseStream, floor: u8) -> syn::Result<Expr> {
    let mut left = parse_unary(input)?;
    while let Some(op) = Op::peek(input)
        && op.precedence() > floor
    {
        op.consume(input)?;
        let right = parse_binary(input, op.precedence())?;
        left = Expr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        };
    }
    Ok(left)
}

fn parse_unary(input: ParseStream) -> syn::Result<Expr> {
    if input.peek(Token![!]) {
        input.parse::<Token![!]>()?;
        return Ok(Expr::Not(Box::new(parse_unary(input)?)));
    }
    if input.peek(Token![-]) && !input.peek2(Lit) {
        input.parse::<Token![-]>()?;
        return Ok(Expr::Neg(Box::new(parse_unary(input)?)));
    }
    if input.peek(Token![*]) {
        input.parse::<Token![*]>()?;
        return Ok(Expr::Lent(input.parse()?));
    }
    if input.peek(Token![&]) {
        input.parse::<Token![&]>()?;
        let mutable = input.peek(Token![mut]);
        if mutable {
            input.parse::<Token![mut]>()?;
        }
        return Ok(Expr::LendEntry {
            mutable,
            entry: parse_entry(input).map_err(|_| {
                syn::Error::new(input.span(), "`&` lends the entry `x[k]` alone in a term")
            })?,
        });
    }
    if input.peek(syn::token::Paren) {
        let inner;
        syn::parenthesized!(inner in input);
        return parse_expr(&inner);
    }
    if input.peek(syn::token::Bracket) {
        let items;
        syn::bracketed!(items in input);
        return Ok(Expr::Array(parse_args(&items)?));
    }
    if input.peek(Token![match]) {
        return parse_match(input);
    }
    let negative = input.peek(Token![-]);
    if negative {
        input.parse::<Token![-]>()?;
    }
    if negative || input.peek(Lit) {
        return Ok(Expr::Lit {
            negative,
            lit: input.parse()?,
        });
    }
    if input.peek(Ident) && input.peek2(syn::token::Bracket) {
        return Ok(Expr::Entry(parse_entry(input)?));
    }
    // A path without generic arguments, so `a < b` is a comparison.
    let callee = Path::parse_mod_style(input).map_err(|_| syn::Error::new(input.span(), TERMS))?;
    if input.peek(syn::token::Paren) {
        let args;
        syn::parenthesized!(args in input);
        let args = parse_args(&args)?;
        if callee.is_ident("Some") {
            let [arg] = <[Expr; 1]>::try_from(args)
                .map_err(|_| syn::Error::new_spanned(&callee, "`Some` holds one term"))?;
            return Ok(Expr::Some(Box::new(arg)));
        }
        return Ok(Expr::Call { callee, args });
    }
    if callee.is_ident("None") {
        return Ok(Expr::None);
    }
    match callee.get_ident() {
        Some(name) => Ok(Expr::Name(name.clone())),
        None => Err(syn::Error::new_spanned(
            callee,
            "a path names an extern, which a term calls: `ns::g(..)`",
        )),
    }
}

fn parse_match(input: ParseStream) -> syn::Result<Expr> {
    input.parse::<Token![match]>()?;
    // No term begins with a brace, so the scrutinee ends where the arms do.
    let scrutinee = Box::new(parse_expr(input)?);
    let arms;
    syn::braced!(arms in input);
    let mut some: Option<SomeArm> = None;
    let mut none: Option<Expr> = None;
    while !arms.is_empty() {
        let variant: Ident = arms.parse()?;
        let binder = match variant.to_string().as_str() {
            "Some" => {
                let inner;
                syn::parenthesized!(inner in arms);
                let binder = match inner.peek(Token![_]) {
                    true => {
                        inner.parse::<Token![_]>()?;
                        None
                    }
                    false => Some(inner.parse::<Ident>()?),
                };
                Some(binder)
            }
            "None" => None,
            _ => {
                return Err(syn::Error::new(
                    variant.span(),
                    "a term matches an `Option`: its arms are `Some(b)` and `None`",
                ));
            }
        };
        arms.parse::<Token![=>]>()?;
        let value = parse_expr(&arms)?;
        match binder {
            Some(binder) if some.is_none() => some = Some(SomeArm { binder, value }),
            None if none.is_none() => none = Some(value),
            _ => {
                return Err(syn::Error::new(
                    variant.span(),
                    format!("the arm `{variant}` is written twice"),
                ));
            }
        }
        if !arms.is_empty() {
            arms.parse::<Token![,]>()?;
        }
    }
    let missing = || {
        syn::Error::new(
            input.span(),
            "a term's `match` has a `Some(b)` arm and a `None` arm",
        )
    };
    let some = some.ok_or_else(missing)?;
    let none = none.ok_or_else(missing)?;
    Ok(Expr::Match {
        scrutinee,
        some_binder: some.binder,
        then: Box::new(some.value),
        none: Box::new(none),
    })
}

struct SomeArm {
    binder: Option<Ident>,
    value: Expr,
}

fn parse_args(input: ParseStream) -> syn::Result<Vec<Expr>> {
    let mut args = Vec::new();
    while !input.is_empty() {
        args.push(parse_expr(input)?);
        if !input.is_empty() {
            input.parse::<Token![,]>()?;
        }
    }
    Ok(args)
}

// -- Checking against the declaration ---------------------------------------

pub(crate) struct Declaration<'a> {
    pub(crate) fn_ident: &'a Ident,
    pub(crate) params: &'a [&'a ExternParam],
    pub(crate) reaches: Option<&'a ReachesAttr>,
    pub(crate) is_type_var: &'a dyn Fn(&Type) -> bool,
}

struct Scope<'a> {
    declaration: &'a Declaration<'a>,
    locals: Vec<Binding>,
    next_local: Cell<usize>,
    entry: Cell<Option<EntryAt>>,
}

struct Binding {
    name: Ident,
    local: usize,
}

/// The parameters an entry `x[k]` is at, numbered as the declaration's
/// acvus parameters are.
#[derive(Clone, Copy, PartialEq, Eq)]
struct EntryAt {
    table: usize,
    key: usize,
}

impl MeansAttr {
    pub(crate) fn checked(&self, declaration: &Declaration<'_>) -> syn::Result<TokenStream> {
        let mut scope = Scope {
            declaration,
            locals: Vec::new(),
            next_local: Cell::new(0),
            entry: Cell::new(None),
        };
        let mut stmts = Vec::new();
        for stmt in &self.stmts {
            stmts.push(match stmt {
                Stmt::Let { name, value } => {
                    let value = scope.term(value)?;
                    let local = scope.bind(name);
                    quote! { ::acvus_extern::means::Stmt::Let { local: #local, value: #value } }
                }
                Stmt::Store { entry, value } => {
                    scope.entry(entry, true)?;
                    let value = scope.term(value)?;
                    quote! { ::acvus_extern::means::Stmt::Store(#value) }
                }
            });
        }
        let result = scope.term(&self.result)?;
        let entry = match scope.entry.get() {
            Some(EntryAt { table, key }) => quote! {
                ::core::option::Option::Some(::acvus_extern::means::EntryParams {
                    table: #table,
                    key: #key,
                })
            },
            None => quote! { ::core::option::Option::None },
        };
        Ok(quote! {
            ::core::option::Option::Some(::acvus_extern::means::Means {
                entry: #entry,
                stmts: vec![#(#stmts),*],
                result: #result,
            })
        })
    }

    pub(crate) fn keyword(&self) -> &Ident {
        &self.keyword
    }

    /// Whether the term names `param`, as a parameter or as a part of its
    /// entry.
    pub(crate) fn names(&self, param: &Ident) -> bool {
        let mut named = self.stmts.iter().any(|stmt| match stmt {
            Stmt::Store { entry, .. } => entry.names(param),
            Stmt::Let { .. } => false,
        });
        let mut visit = |expr: &Expr| {
            named |= match expr {
                Expr::Name(name) | Expr::Lent(name) => name == param,
                Expr::Entry(entry) | Expr::LendEntry { entry, .. } => entry.names(param),
                _ => false,
            };
        };
        for stmt in &self.stmts {
            match stmt {
                Stmt::Let { value, .. } | Stmt::Store { value, .. } => value.visit(&mut visit),
            }
        }
        self.result.visit(&mut visit);
        named
    }
}

impl EntryName {
    fn names(&self, param: &Ident) -> bool {
        self.table == *param || self.key == *param
    }
}

impl Expr {
    fn visit(&self, on: &mut impl FnMut(&Expr)) {
        on(self);
        match self {
            Expr::Some(inner) | Expr::Not(inner) | Expr::Neg(inner) => inner.visit(on),
            Expr::Binary { left, right, .. } => {
                left.visit(on);
                right.visit(on);
            }
            Expr::Array(items) | Expr::Call { args: items, .. } => {
                for item in items {
                    item.visit(on);
                }
            }
            Expr::Match {
                scrutinee,
                then,
                none,
                ..
            } => {
                scrutinee.visit(on);
                then.visit(on);
                none.visit(on);
            }
            Expr::Name(_)
            | Expr::Lent(_)
            | Expr::Entry(_)
            | Expr::LendEntry { .. }
            | Expr::None
            | Expr::Lit { .. } => {}
        }
    }
}

impl Scope<'_> {
    fn bind(&mut self, name: &Ident) -> usize {
        let local = self.next_local.get();
        self.next_local.set(local + 1);
        self.locals.push(Binding {
            name: name.clone(),
            local,
        });
        local
    }

    fn param(&self, name: &Ident) -> Option<(usize, &ExternParam)> {
        let params = self.declaration.params;
        let at = params.iter().position(|param| *name == param.name)?;
        Some((at, params[at]))
    }

    fn no_param(&self, name: &Ident) -> syn::Error {
        syn::Error::new(
            name.span(),
            format!(
                "`{name}` names no parameter of `{}` and no `let` or `match` binding \
                 (RFC-0104 rule 1)",
                self.declaration.fn_ident
            ),
        )
    }

    /// `x[k]`: the entry the declaration's `reaches` names, and the one
    /// entry the term names.
    fn entry(&self, entry: &EntryName, writes: bool) -> syn::Result<()> {
        let (table, taken) = self.param(&entry.table).ok_or_else(|| self.no_param(&entry.table))?;
        let (key, _) = self.param(&entry.key).ok_or_else(|| self.no_param(&entry.key))?;
        let reached = self.declaration.reaches.is_some_and(|reaches| {
            reaches.names_entry(
                &entry.table,
                &entry.key,
                self.declaration.params,
                self.declaration.is_type_var,
            )
        });
        if !reached {
            return Err(syn::Error::new(
                entry.table.span(),
                format!(
                    "the entry `{}[{}]` is no entry the declaration's `reaches` names: a term \
                     reads and writes only an entry `reaches` states (RFC-0104 rule 1)",
                    entry.table, entry.key
                ),
            ));
        }
        if writes && taken.mode != Mode::BorrowMut {
            return Err(syn::Error::new(
                entry.table.span(),
                format!(
                    "`{}` is lent `&`, and a term writes an entry only through `&mut`",
                    entry.table
                ),
            ));
        }
        let at = EntryAt { table, key };
        match self.entry.get() {
            None => self.entry.set(Some(at)),
            Some(named) if named == at => {}
            Some(_) => {
                return Err(syn::Error::new(
                    entry.table.span(),
                    "a term names one entry `x[k]` (RFC-0104 rule 1)",
                ));
            }
        }
        Ok(())
    }

    fn term(&mut self, expr: &Expr) -> syn::Result<TokenStream> {
        let term = quote! { ::acvus_extern::means::Term };
        Ok(match expr {
            Expr::Name(name) => {
                if let Some(Binding { local, .. }) =
                    self.locals.iter().rev().find(|bound| bound.name == *name)
                {
                    return Ok(quote! { #term::Local(#local) });
                }
                let (at, taken) = self.param(name).ok_or_else(|| self.no_param(name))?;
                if taken.mode != Mode::Value || closure_arity(&taken.ty) {
                    return Err(syn::Error::new(
                        name.span(),
                        format!(
                            "`{name}` is not taken by value: a term reads a reference \
                             parameter as `*{name}`, and calls no closure (RFC-0104 rule 1)"
                        ),
                    ));
                }
                quote! { #term::Param(#at) }
            }
            Expr::Lent(name) => {
                let (at, taken) = self.param(name).ok_or_else(|| self.no_param(name))?;
                if !matches!(taken.mode, Mode::Borrow | Mode::BorrowMut | Mode::Str) {
                    return Err(syn::Error::new(
                        name.span(),
                        format!(
                            "`*{name}` reads what a reference parameter lends, and `{name}` is \
                             not one (RFC-0104 rule 1)"
                        ),
                    ));
                }
                quote! { #term::Lent(#at) }
            }
            Expr::Entry(entry) => {
                self.entry(entry, false)?;
                quote! { #term::Entry }
            }
            Expr::LendEntry { mutable, entry } => {
                self.entry(entry, *mutable)?;
                match mutable {
                    true => quote! { #term::LendEntry(::acvus_extern::Mutability::Mut) },
                    false => quote! { #term::LendEntry(::acvus_extern::Mutability::Shared) },
                }
            }
            Expr::Some(inner) => {
                let inner = self.term(inner)?;
                quote! { #term::Some(::std::boxed::Box::new(#inner)) }
            }
            Expr::None => quote! { #term::None },
            Expr::Lit { negative, lit } => {
                let literal = crate::step::constant(*negative, lit)?;
                quote! { #term::Const(#literal) }
            }
            Expr::Not(inner) => {
                let inner = self.term(inner)?;
                quote! { #term::Not(::std::boxed::Box::new(#inner)) }
            }
            Expr::Neg(inner) => {
                let inner = self.term(inner)?;
                quote! { #term::Neg(::std::boxed::Box::new(#inner)) }
            }
            Expr::Binary { op, left, right } => {
                let left = self.term(left)?;
                let right = self.term(right)?;
                let op = op.tokens();
                quote! {
                    #term::Binary {
                        op: #op,
                        left: ::std::boxed::Box::new(#left),
                        right: ::std::boxed::Box::new(#right),
                    }
                }
            }
            Expr::Array(items) => {
                let items = items
                    .iter()
                    .map(|item| self.term(item))
                    .collect::<syn::Result<Vec<_>>>()?;
                quote! { #term::Array(vec![#(#items),*]) }
            }
            Expr::Call { callee, args } => {
                if let Some(name) = callee.get_ident()
                    && (self.param(name).is_some()
                        || self.locals.iter().any(|bound| bound.name == *name))
                {
                    return Err(syn::Error::new(
                        name.span(),
                        format!("`{name}` is no extern, and a term calls only registered externs"),
                    ));
                }
                let args = args
                    .iter()
                    .map(|arg| self.term(arg))
                    .collect::<syn::Result<Vec<_>>>()?;
                let name = crate::law::qref_of(callee)?;
                quote! { #term::Call { name: #name, args: vec![#(#args),*] } }
            }
            Expr::Match {
                scrutinee,
                some_binder,
                then,
                none,
            } => {
                let scrutinee = self.term(scrutinee)?;
                let none = self.term(none)?;
                let bound = some_binder.as_ref().map(|binder| self.bind(binder));
                let then = self.term(then)?;
                if bound.is_some() {
                    self.locals.pop();
                }
                let some = match bound {
                    Some(local) => quote! { ::core::option::Option::Some(#local) },
                    None => quote! { ::core::option::Option::None },
                };
                quote! {
                    #term::Match {
                        scrutinee: ::std::boxed::Box::new(#scrutinee),
                        some: #some,
                        then: ::std::boxed::Box::new(#then),
                        none: ::std::boxed::Box::new(#none),
                    }
                }
            }
        })
    }
}

fn closure_arity(ty: &Type) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    path.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "Closure")
}
