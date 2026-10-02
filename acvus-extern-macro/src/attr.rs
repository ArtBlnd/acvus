//! The attribute's own arguments: `#[extern_fn(name = "..", effect = ..,
//! ends = .., sync = .., impl_of = g<..>, generic, relation(..),
//! combine(..))]`. Every key is known; any other is refused.

use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Expr, GenericArgument, Ident, LitStr, Path, PathArguments, Token, Type, parenthesized, token};

/// A call's effect as stated (compiler: extern fn §2): a level, or one of
/// the function's effect variables, the effect of the work it runs.
#[derive(Clone)]
pub enum Effect {
    Pure,
    Idempotent,
    Opaque,
    Var(Ident),
}

#[derive(Default)]
pub struct ExternFnAttr {
    /// The declaration's name where it overloads another in one module; the
    /// function's own otherwise.
    pub name: Option<LitStr>,
    /// Unstated, the effect is opaque, which is always sound.
    pub effect: Option<Effect>,
    /// How a call's own code ends (compiler: extern fn §4); unstated where
    /// none is written.
    pub ends: Option<Ends>,
    /// The relations stated, each as written (compiler: extern fn §3).
    pub relations: Vec<Expr>,
    /// That the function is a combine, with its laws (compiler: extern fn
    /// §3, Action): an associative operation, `combine` stating that much.
    pub combine: Option<Combine>,
    /// An awaited body's returning body, a function of its signature
    /// (macros §1).
    pub sync: Option<Path>,
    /// The generic function this is an impl of, at the choice of its
    /// variables written (extern fn §6).
    pub impl_of: Option<ImplOf>,
    /// That the impl is its generic function's generic one.
    pub generic: Option<Ident>,
}

/// `impl_of = g<A, ..>`: the generic function's family marker, and the
/// choice of each of its variables.
pub struct ImplOf {
    pub generic_function: Path,
    pub choice: Vec<Type>,
}

impl Parse for ImplOf {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut generic_function: Path = input.parse()?;
        let last = generic_function.segments.last_mut().expect("a path has a segment");
        let choice = match std::mem::replace(&mut last.arguments, PathArguments::None) {
            PathArguments::None => Vec::new(),
            PathArguments::AngleBracketed(args) => args
                .args
                .into_iter()
                .map(|arg| match arg {
                    GenericArgument::Type(ty) => Ok(ty),
                    other => Err(syn::Error::new_spanned(other, "a generic function's variable is chosen as a type")),
                })
                .collect::<syn::Result<_>>()?,
            other @ PathArguments::Parenthesized(_) => {
                return Err(syn::Error::new_spanned(other, "`impl_of` names a generic function as `g<A, ..>`"));
            }
        };
        Ok(ImplOf { generic_function, choice })
    }
}

/// A combine's laws as written: its identity and its absorbing element, each
/// another declaration, and whether it commutes.
#[derive(Default)]
pub struct Combine {
    pub identity: Option<Path>,
    pub commutative: bool,
    pub absorbing: Option<Path>,
}

/// How a call's own code ends.
#[derive(Clone, Copy)]
pub enum Ends {
    Returns,
    Total,
}

impl Parse for ExternFnAttr {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut attr = ExternFnAttr::default();
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            let twice = || syn::Error::new(key.span(), format!("`{key}` is stated twice"));
            match key.to_string().as_str() {
                "relation" => {
                    let content;
                    parenthesized!(content in input);
                    let relations: Punctuated<Expr, Token![,]> = content.parse_terminated(Expr::parse, Token![,])?;
                    attr.relations.extend(relations);
                }
                "combine" => {
                    let mut combine = Combine::default();
                    if input.peek(token::Paren) {
                        let content;
                        parenthesized!(content in input);
                        while !content.is_empty() {
                            let law: Ident = content.parse()?;
                            match law.to_string().as_str() {
                                "identity" => {
                                    content.parse::<Token![=]>()?;
                                    combine.identity = Some(content.parse()?);
                                }
                                "absorbing" => {
                                    content.parse::<Token![=]>()?;
                                    combine.absorbing = Some(content.parse()?);
                                }
                                "commutative" => combine.commutative = true,
                                _ => {
                                    return Err(syn::Error::new(
                                        law.span(),
                                        "a combine states `identity = f`, `commutative` and `absorbing = f`",
                                    ));
                                }
                            }
                            if !content.is_empty() {
                                content.parse::<Token![,]>()?;
                            }
                        }
                    }
                    if attr.combine.replace(combine).is_some() {
                        return Err(twice());
                    }
                }
                "name" if input.peek(Token![=]) => {
                    input.parse::<Token![=]>()?;
                    let name: LitStr = input.parse()?;
                    if attr.name.replace(name).is_some() {
                        return Err(twice());
                    }
                }
                "effect" if input.peek(Token![=]) => {
                    input.parse::<Token![=]>()?;
                    let value: Ident = input.parse()?;
                    let effect = match value.to_string().as_str() {
                        "pure" => Effect::Pure,
                        "idempotent" => Effect::Idempotent,
                        "opaque" => Effect::Opaque,
                        _ => Effect::Var(value),
                    };
                    if attr.effect.replace(effect).is_some() {
                        return Err(twice());
                    }
                }
                "impl_of" if input.peek(Token![=]) => {
                    input.parse::<Token![=]>()?;
                    let impl_of: ImplOf = input.parse()?;
                    if attr.impl_of.replace(impl_of).is_some() {
                        return Err(twice());
                    }
                }
                "generic" => {
                    if attr.generic.replace(key.clone()).is_some() {
                        return Err(twice());
                    }
                }
                "sync" if input.peek(Token![=]) => {
                    input.parse::<Token![=]>()?;
                    let body: Path = input.parse()?;
                    if attr.sync.replace(body).is_some() {
                        return Err(twice());
                    }
                }
                "ends" if input.peek(Token![=]) => {
                    input.parse::<Token![=]>()?;
                    let value: Ident = input.parse()?;
                    let ends = match value.to_string().as_str() {
                        "returns" => Ends::Returns,
                        "total" => Ends::Total,
                        _ => return Err(syn::Error::new(value.span(), "a call ends `returns` or `total`")),
                    };
                    if attr.ends.replace(ends).is_some() {
                        return Err(twice());
                    }
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        "`#[extern_fn]` knows `name = \"..\"`, `effect = ..`, `ends = ..`, `sync = ..`, `impl_of = ..`, `generic`, `relation(..)` and `combine(..)`, and nothing else",
                    ));
                }
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(attr)
    }
}
