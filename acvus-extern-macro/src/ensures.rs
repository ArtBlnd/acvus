//! `#[extern_fn(ensures(..))]`: the postconditions a declaration states
//! (RFC-0082 rule 4), and the debug evaluation of each at the function's
//! return (rule 5).
//!
//! A relation is `t1 = t2`, `t1 <= t2` or `t1 < t2`. A term is a constant,
//! a parameter, `ret`, `len(x)` of a parameter or of `ret`, and `+`, `-`,
//! `*` and `max(a, b)` of terms. The RFC writes `≤`, `−` and `×`; Rust's
//! lexer refuses those characters before a macro reads the attribute, so
//! the declaration spells them `<=`, `-` and `*`. Nothing else is read: no
//! quantifier, no condition and no function of the author's.

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::parse::ParseStream;
use syn::{Ident, LitInt, Token, Type};

use crate::ExternParam;

pub(crate) struct EnsuresAttr {
    relations: Vec<Relation>,
}

struct Relation {
    left: Term,
    relation: Rel,
    right: Term,
}

/// What `#[extern_fn]` emits for a declaration's postconditions.
pub(crate) struct Stated {
    /// The registry's `Vec<Postcondition>`.
    pub(crate) declared: TokenStream,
    /// The statements that check each relation against `__acvus_ret`.
    pub(crate) evaluated: TokenStream,
}

#[derive(Clone, Copy)]
enum Rel {
    Eq,
    Le,
    Lt,
}

enum Term {
    Const(i128),
    /// A bare word: a parameter's name, or `ret`.
    Named(Ident),
    Len(Ident),
    Add(Box<Term>, Box<Term>),
    Sub(Box<Term>, Box<Term>),
    Mul(Box<Term>, Box<Term>),
    Max(Box<Term>, Box<Term>),
}

const TERMS: &str = "a term is a constant, a parameter, `ret`, `len(x)` of a parameter or \
                     of `ret`, or `+`, `-`, `*` or `max(a, b)` of terms (RFC-0082 rule 4)";

impl EnsuresAttr {
    pub(crate) fn parse_after(keyword: &Ident, input: ParseStream) -> syn::Result<Self> {
        let content;
        syn::parenthesized!(content in input);
        let mut relations = Vec::new();
        while !content.is_empty() {
            relations.push(Relation::parse(&content)?);
            if !content.is_empty() {
                content.parse::<Token![,]>().map_err(|_| {
                    syn::Error::new(
                        content.span(),
                        "a relation is `t1 = t2`, `t1 <= t2` or `t1 < t2` and nothing after \
                         it: there is no condition and no connective (RFC-0082 rule 4)",
                    )
                })?;
            }
        }
        if relations.is_empty() {
            return Err(syn::Error::new(
                keyword.span(),
                "`ensures()` states no postcondition",
            ));
        }
        Ok(EnsuresAttr { relations })
    }

    /// A name that is no acvus parameter of the declaration is refused
    /// here. A `len` of something that is neither a slice nor a container,
    /// and a number read off something that is not an integer, are refused
    /// by the trait bounds the evaluation names, which only the type
    /// checker can decide.
    pub(crate) fn stated(&self, fn_ident: &Ident, params: &[&ExternParam]) -> syn::Result<Stated> {
        let mut declared = Vec::new();
        let mut evaluated = Vec::new();
        let function = fn_ident.to_string();
        for relation in &self.relations {
            let left = relation.left.declared(fn_ident, params)?;
            let right = relation.right.declared(fn_ident, params)?;
            let rel = relation.relation.declared();
            declared.push(quote! {
                ::acvus_extern::Postcondition { left: #left, relation: #rel, right: #right }
            });
            let left = relation.left.evaluated();
            let right = relation.right.evaluated();
            let written = relation.to_string();
            evaluated.push(quote! {
                ::acvus_extern::ensures::assert_postcondition(
                    #function,
                    #written,
                    ::acvus_extern::ensures::Evaluated { left: #left, relation: #rel, right: #right },
                );
            });
        }
        Ok(Stated {
            declared: quote! { vec![#(#declared),*] },
            evaluated: quote! { #(#evaluated)* },
        })
    }
}

impl Relation {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let left = Term::parse_sum(input)?;
        let relation = if input.peek(Token![<=]) {
            input.parse::<Token![<=]>()?;
            Rel::Le
        } else if input.peek(Token![==]) {
            let eq: Token![==] = input.parse()?;
            return Err(syn::Error::new_spanned(
                eq,
                "the relation is written `=` (RFC-0082 rule 4)",
            ));
        } else if input.peek(Token![=]) {
            input.parse::<Token![=]>()?;
            Rel::Eq
        } else if input.peek(Token![<]) {
            input.parse::<Token![<]>()?;
            Rel::Lt
        } else if input.peek(Token![>=]) || input.peek(Token![>]) {
            return Err(syn::Error::new(
                input.span(),
                "a relation is `=`, `<=` or `<`: write the terms the other way round \
                 (RFC-0082 rule 4)",
            ));
        } else {
            return Err(syn::Error::new(
                input.span(),
                "a postcondition relates two terms by `=`, `<=` or `<` (RFC-0082 rule 4)",
            ));
        };
        let right = Term::parse_sum(input)?;
        Ok(Relation {
            left,
            relation,
            right,
        })
    }
}

impl std::fmt::Display for Relation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rel = match self.relation {
            Rel::Eq => "=",
            Rel::Le => "<=",
            Rel::Lt => "<",
        };
        write!(f, "{} {rel} {}", self.left, self.right)
    }
}

impl std::fmt::Display for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::Const(c) => write!(f, "{c}"),
            Term::Named(word) => write!(f, "{word}"),
            Term::Len(word) => write!(f, "len({word})"),
            Term::Add(a, b) => write!(f, "({a} + {b})"),
            Term::Sub(a, b) => write!(f, "({a} - {b})"),
            Term::Mul(a, b) => write!(f, "({a} * {b})"),
            Term::Max(a, b) => write!(f, "max({a}, {b})"),
        }
    }
}

/// `forall i` or `i: ...`: a word that a name, a type or a body follows.
fn binds_a_name(input: ParseStream) -> bool {
    input.peek(Ident) || input.peek(Token![:]) || input.peek(Token![.])
}

impl Term {
    fn parse_sum(input: ParseStream) -> syn::Result<Self> {
        let mut term = Self::parse_product(input)?;
        loop {
            if input.peek(Token![+]) {
                input.parse::<Token![+]>()?;
                term = Term::Add(Box::new(term), Box::new(Self::parse_product(input)?));
            } else if input.peek(Token![-]) {
                input.parse::<Token![-]>()?;
                term = Term::Sub(Box::new(term), Box::new(Self::parse_product(input)?));
            } else {
                return Ok(term);
            }
        }
    }

    fn parse_product(input: ParseStream) -> syn::Result<Self> {
        let mut term = Self::parse_atom(input)?;
        while input.peek(Token![*]) {
            input.parse::<Token![*]>()?;
            term = Term::Mul(Box::new(term), Box::new(Self::parse_atom(input)?));
        }
        Ok(term)
    }

    fn parse_atom(input: ParseStream) -> syn::Result<Self> {
        if input.peek(syn::token::Paren) {
            let inner;
            syn::parenthesized!(inner in input);
            let term = Self::parse_sum(&inner)?;
            if !inner.is_empty() {
                return Err(syn::Error::new(inner.span(), TERMS));
            }
            return Ok(term);
        }
        if input.peek(LitInt) {
            let lit: LitInt = input.parse()?;
            if !lit.suffix().is_empty() {
                return Err(syn::Error::new_spanned(
                    lit,
                    "a constant term is an integer without a suffix",
                ));
            }
            return Ok(Term::Const(lit.base10_parse()?));
        }
        if !input.peek(Ident) {
            return Err(syn::Error::new(input.span(), TERMS));
        }
        let word: Ident = input.parse()?;
        if input.peek(syn::token::Paren) {
            let args;
            syn::parenthesized!(args in input);
            return match word.to_string().as_str() {
                "len" => {
                    let subject: Ident = args.parse().map_err(|_| {
                        syn::Error::new(
                            args.span(),
                            "`len` reads a parameter or `ret` (RFC-0082 rule 4)",
                        )
                    })?;
                    if !args.is_empty() {
                        return Err(syn::Error::new(
                            args.span(),
                            "`len` reads a parameter or `ret` (RFC-0082 rule 4)",
                        ));
                    }
                    Ok(Term::Len(subject))
                }
                "max" => {
                    let a = Self::parse_sum(&args)?;
                    args.parse::<Token![,]>()?;
                    let b = Self::parse_sum(&args)?;
                    if !args.is_empty() {
                        return Err(syn::Error::new(args.span(), "`max` takes two terms"));
                    }
                    Ok(Term::Max(Box::new(a), Box::new(b)))
                }
                other => Err(syn::Error::new(
                    word.span(),
                    format!(
                        "`{other}` is not in the vocabulary: the functions of a term are \
                         `len` and `max`, and there is no function of the author's \
                         (RFC-0082 rule 4)"
                    ),
                )),
            };
        }
        if binds_a_name(input) {
            return Err(syn::Error::new(
                word.span(),
                format!(
                    "`{word}` binds a name, and a postcondition has no quantifier \
                     (RFC-0082 rule 4)"
                ),
            ));
        }
        Ok(Term::Named(word))
    }

    fn declared(&self, fn_ident: &Ident, params: &[&ExternParam]) -> syn::Result<TokenStream> {
        let binary = |ctor: TokenStream, a: &Term, b: &Term| -> syn::Result<TokenStream> {
            let a = a.declared(fn_ident, params)?;
            let b = b.declared(fn_ident, params)?;
            Ok(quote! {
                ::acvus_extern::PostTerm::#ctor(::std::boxed::Box::new(#a), ::std::boxed::Box::new(#b))
            })
        };
        match self {
            Term::Const(c) => Ok(quote! { ::acvus_extern::PostTerm::Const(#c) }),
            Term::Named(word) if word == "ret" => Ok(quote! { ::acvus_extern::PostTerm::Ret }),
            Term::Named(word) => {
                let at = param_at(word, fn_ident, params)?;
                Ok(quote! { ::acvus_extern::PostTerm::Param(#at) })
            }
            Term::Len(word) if word == "ret" => Ok(quote! {
                ::acvus_extern::PostTerm::Len(::acvus_extern::Subject::Ret)
            }),
            Term::Len(word) => {
                let at = param_at(word, fn_ident, params)?;
                Ok(quote! {
                    ::acvus_extern::PostTerm::Len(::acvus_extern::Subject::Param(#at))
                })
            }
            Term::Add(a, b) => binary(quote! { Add }, a, b),
            Term::Sub(a, b) => binary(quote! { Sub }, a, b),
            Term::Mul(a, b) => binary(quote! { Mul }, a, b),
            Term::Max(a, b) => binary(quote! { Max }, a, b),
        }
    }

    fn evaluated(&self) -> TokenStream {
        let binary = |op: &str, a: &Term, b: &Term| {
            let op = Ident::new(op, proc_macro2::Span::call_site());
            let a = a.evaluated();
            let b = b.evaluated();
            quote! { ::acvus_extern::ensures::#op(#a, #b) }
        };
        match self {
            Term::Const(c) => quote! { ::core::option::Option::Some(#c) },
            Term::Named(word) => {
                let read = bound_to(word);
                quote! { ::acvus_extern::ensures::Integer::term(&#read) }
            }
            Term::Len(word) => {
                let read = bound_to(word);
                quote! { ::acvus_extern::ensures::len_of(&#read) }
            }
            Term::Add(a, b) => binary("add", a, b),
            Term::Sub(a, b) => binary("sub", a, b),
            Term::Mul(a, b) => binary("mul", a, b),
            Term::Max(a, b) => binary("max", a, b),
        }
    }
}

impl Rel {
    fn declared(self) -> TokenStream {
        match self {
            Rel::Eq => quote! { ::acvus_extern::Relation::Eq },
            Rel::Le => quote! { ::acvus_extern::Relation::Le },
            Rel::Lt => quote! { ::acvus_extern::Relation::Lt },
        }
    }
}

fn bound_to(word: &Ident) -> Ident {
    match word == "ret" {
        true => Ident::new("__acvus_ret", word.span()),
        false => word.clone(),
    }
}

fn param_at(word: &Ident, fn_ident: &Ident, params: &[&ExternParam]) -> syn::Result<usize> {
    params
        .iter()
        .position(|p| *word == p.name)
        .ok_or_else(|| {
            syn::Error::new(
                word.span(),
                format!(
                    "`{word}` names no parameter of `{fn_ident}`: a term reads the \
                     declaration's acvus parameters and `ret` (RFC-0082 rule 4)"
                ),
            )
        })
}

/// The body `block` of `fn_ident`, returning `ret`, with each postcondition
/// evaluated at its return in a debug build. The body runs as a closure,
/// or an `async` block for an `async fn`, so a `return` inside it returns
/// to the evaluation and not past it.
pub(crate) fn wrap_body(
    block: &syn::Block,
    ret: &Type,
    is_async: bool,
    evaluated: &TokenStream,
) -> syn::Block {
    let run = match is_async {
        true => quote! { async #block.await },
        false if names_a_borrow(ret) => quote! { (|| #block)() },
        false => quote! { (|| -> #ret #block)() },
    };
    syn::parse_quote! {{
        #[allow(clippy::redundant_closure_call)]
        let __acvus_ret = #run;
        if ::core::cfg!(debug_assertions) {
            #evaluated
        }
        __acvus_ret
    }}
}

/// A closure's return type cannot name an elided lifetime, so a result
/// that borrows is left for the closure to infer.
fn names_a_borrow(ty: &Type) -> bool {
    let written = ty.to_token_stream().to_string();
    written.contains('&') || written.contains('\'')
}
