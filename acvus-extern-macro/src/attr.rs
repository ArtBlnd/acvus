//! The attribute's own arguments: `#[extern_fn(name = "..", effect = ..,
//! ends = .., relation(..))]`. Every key is known; any other is refused.

use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Expr, Ident, LitStr, Token, parenthesized};

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
                        "`#[extern_fn]` knows `name = \"..\"`, `effect = ..`, `ends = ..` and `relation(..)`, and nothing else",
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
