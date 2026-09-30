//! The attribute's own arguments: `#[extern_fn(name = "..", effect = ..)]`.
//! Every key is known; any other is refused.

use syn::parse::{Parse, ParseStream};
use syn::{Ident, LitStr, Token};

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
}

impl Parse for ExternFnAttr {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut attr = ExternFnAttr::default();
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            let known = matches!(key.to_string().as_str(), "name" | "effect");
            if !known || !input.peek(Token![=]) {
                return Err(syn::Error::new(
                    key.span(),
                    "`#[extern_fn]` knows `name = \"..\"` and `effect = ..`, and nothing else",
                ));
            }
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "name" => {
                    let name: LitStr = input.parse()?;
                    if attr.name.replace(name).is_some() {
                        return Err(syn::Error::new(key.span(), "`name` is stated twice"));
                    }
                }
                "effect" => {
                    let value: Ident = input.parse()?;
                    let effect = match value.to_string().as_str() {
                        "pure" => Effect::Pure,
                        "idempotent" => Effect::Idempotent,
                        "opaque" => Effect::Opaque,
                        _ => Effect::Var(value),
                    };
                    if attr.effect.replace(effect).is_some() {
                        return Err(syn::Error::new(key.span(), "`effect` is stated twice"));
                    }
                }
                _ => unreachable!("the key is known"),
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(attr)
    }
}
