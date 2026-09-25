//! `#[extern_fn(reaches(..))]` (RFC-0082 rule 7).

use proc_macro2::TokenStream;
use quote::quote;
use syn::parse::ParseStream;
use syn::{Ident, Token, Type};

use crate::{ExternParam, Mode};

pub(crate) struct ReachesAttr {
    places: Vec<Place>,
}

struct Place {
    of: Ident,
    element: Option<Ident>,
}

const PLACES: &str = "a place is `x`, a reference parameter, or `x[i]`, its element at the \
                      `u64` parameter `i` (RFC-0082 rule 7)";

impl ReachesAttr {
    /// The name in a stated place that is `param`, as written.
    pub(crate) fn naming(&self, param: &Ident) -> Option<&Ident> {
        self.places
            .iter()
            .flat_map(|place| std::iter::once(&place.of).chain(&place.element))
            .find(|named| *named == param)
    }

    pub(crate) fn parse_after(keyword: &Ident, input: ParseStream) -> syn::Result<Self> {
        let content;
        syn::parenthesized!(content in input);
        let mut places = Vec::new();
        while !content.is_empty() {
            let of: Ident = content
                .parse()
                .map_err(|_| syn::Error::new(content.span(), PLACES))?;
            let element = match content.peek(syn::token::Bracket) {
                true => {
                    let index;
                    syn::bracketed!(index in content);
                    let at: Ident = index
                        .parse()
                        .map_err(|_| syn::Error::new(index.span(), PLACES))?;
                    if !index.is_empty() {
                        return Err(syn::Error::new(index.span(), PLACES));
                    }
                    Some(at)
                }
                false => None,
            };
            places.push(Place { of, element });
            if !content.is_empty() {
                content
                    .parse::<Token![,]>()
                    .map_err(|_| syn::Error::new(content.span(), PLACES))?;
            }
        }
        if places.is_empty() {
            return Err(syn::Error::new(
                keyword.span(),
                "`reaches()` states no place: a declaration that reaches nothing through \
                 a reference takes none",
            ));
        }
        Ok(ReachesAttr { places })
    }

    pub(crate) fn declared(&self, fn_ident: &Ident, params: &[&ExternParam]) -> syn::Result<TokenStream> {
        let mut declared = Vec::new();
        for place in &self.places {
            let (param, taken) = param_at(&place.of, fn_ident, params)?;
            if taken.mode == Mode::Value {
                return Err(syn::Error::new(
                    place.of.span(),
                    format!(
                        "`{}` is not taken by reference, and a call reaches a place only \
                         through a reference argument (RFC-0082 rule 7)",
                        place.of
                    ),
                ));
            }
            let element = match &place.element {
                Some(index) => {
                    let (at, taken) = param_at(index, fn_ident, params)?;
                    let is_u64 = matches!(&taken.ty, Type::Path(path) if path.path.is_ident("u64"));
                    if taken.mode != Mode::Value || !is_u64 {
                        return Err(syn::Error::new(
                            index.span(),
                            format!(
                                "`{index}` is not a `u64` taken by value, and an element is \
                                 named by its index (RFC-0082 rule 7)"
                            ),
                        ));
                    }
                    quote! { ::core::option::Option::Some(#at) }
                }
                None => quote! { ::core::option::Option::None },
            };
            declared.push(quote! {
                ::acvus_extern::ReachedPlace { param: #param, element: #element }
            });
        }
        let unnamed = params.iter().find(|param| {
            param.mode != Mode::Value && !self.places.iter().any(|place| place.of == param.name)
        });
        if let Some(unnamed) = unnamed {
            return Err(syn::Error::new(
                fn_ident.span(),
                format!(
                    "`reaches` names no place of the reference parameter `{}`: it states \
                     every place the call reaches through each (RFC-0082 rule 7)",
                    unnamed.name
                ),
            ));
        }
        Ok(quote! { ::acvus_extern::Reaches::Places(vec![#(#declared),*]) })
    }
}

fn param_at<'a>(
    word: &Ident,
    fn_ident: &Ident,
    params: &[&'a ExternParam],
) -> syn::Result<(usize, &'a ExternParam)> {
    params
        .iter()
        .position(|p| *word == p.name)
        .map(|at| (at, params[at]))
        .ok_or_else(|| {
            syn::Error::new(
                word.span(),
                format!(
                    "`{word}` names no parameter of `{fn_ident}`: a place reads the \
                     declaration's acvus parameters (RFC-0082 rule 7)"
                ),
            )
        })
}
