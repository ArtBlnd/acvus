//! `#[extern_fn(law(..))]`: the algebraic laws a declaration states
//! (RFC-0082 rules 2, 3 and 10).

use quote::{ToTokens, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Ident, Lit, Path, Token, Type};

use crate::{ExternParam, Mode, Returning};

/// `law(associative, commutative, identity = e)`,
/// `law(fold(combine = g, identity = e), commutative)` or
/// `law(total_order)`.
pub(crate) struct LawAttr {
    first_word: Ident,
    associative: Option<Ident>,
    commutative: Option<Ident>,
    identity: Option<IdentityAttr>,
    fold: Option<FoldAttr>,
    total_order: Option<Ident>,
}

enum IdentityAttr {
    Const(ConstIdentity),
    Extern(Path),
}

struct ConstIdentity {
    negative: bool,
    lit: Lit,
}

struct FoldAttr {
    keyword: Ident,
    combine: Path,
    identity: Path,
}

impl LawAttr {
    pub(crate) fn parse_after(keyword: Ident, input: ParseStream) -> syn::Result<Self> {
        let content;
        syn::parenthesized!(content in input);
        let mut first_word: Option<Ident> = None;
        let mut associative = None;
        let mut commutative = None;
        let mut identity = None;
        let mut fold = None;
        let mut total_order = None;
        while !content.is_empty() {
            let word: Ident = content.parse()?;
            let stated_twice = match word.to_string().as_str() {
                "associative" => associative.replace(word.clone()).is_some(),
                "commutative" => commutative.replace(word.clone()).is_some(),
                "identity" => {
                    content.parse::<Token![=]>()?;
                    let value = content.parse::<IdentityAttr>()?;
                    identity.replace(value).is_some()
                }
                "fold" => {
                    let stated = FoldAttr::parse_after(word.clone(), &content)?;
                    fold.replace(stated).is_some()
                }
                "total_order" => total_order.replace(word.clone()).is_some(),
                other => {
                    return Err(syn::Error::new(
                        word.span(),
                        format!(
                            "unknown law `{other}`: a law is `associative`, `commutative`, \
                             `identity = e`, `fold(combine = g, identity = e)`, or \
                             `total_order` (RFC-0082)"
                        ),
                    ));
                }
            };
            if stated_twice {
                return Err(syn::Error::new(
                    word.span(),
                    format!("the law `{word}` is stated twice"),
                ));
            }
            first_word.get_or_insert(word);
            if !content.is_empty() {
                content.parse::<Token![,]>()?;
            }
        }
        let Some(first_word) = first_word else {
            return Err(syn::Error::new(keyword.span(), "`law()` states no law"));
        };
        if let Some(fold) = &fold
            && (associative.is_some() || identity.is_some())
        {
            return Err(syn::Error::new(
                fold.keyword.span(),
                "the law `fold` is a storage write's, and `associative` and `identity` are a \
                 binary function's: a declaration states one form (RFC-0082)",
            ));
        }
        if let Some(order) = &total_order
            && (associative.is_some() || commutative.is_some() || identity.is_some() || fold.is_some())
        {
            return Err(syn::Error::new(
                order.span(),
                "the law `total_order` is a comparison's, and `associative`, `commutative`, \
                 `identity` and `fold` are a combining function's: a declaration states one \
                 form (RFC-0082)",
            ));
        }
        Ok(LawAttr {
            first_word,
            associative,
            commutative,
            identity,
            fold,
            total_order,
        })
    }

    pub(crate) fn checked_laws(
        &self,
        fn_ident: &Ident,
        params: &[&ExternParam],
        ret: &Type,
        returning: &Returning,
    ) -> syn::Result<proc_macro2::TokenStream> {
        let commutative = self.commutative.is_some();
        if let Some(order) = &self.total_order {
            let compares = match params {
                [a, b] => {
                    matches!(
                        (a.mode, b.mode),
                        (Mode::Borrow, Mode::Borrow) | (Mode::Str, Mode::Str)
                    ) && same_type(&a.ty, &b.ty)
                }
                _ => false,
            };
            let signed_word = matches!(returning, Returning::Value)
                && matches!(ret, Type::Path(path) if path.path.is_ident("i64"));
            if !compares || !signed_word {
                return Err(syn::Error::new(
                    order.span(),
                    format!(
                        "the law `total_order` is stated over `f(a: &T, b: &T) -> i64`, and \
                         `{fn_ident}` is not of that shape"
                    ),
                ));
            }
            return Ok(quote! { ::acvus_extern::Laws::TotalOrder });
        }
        if let Some(fold) = &self.fold {
            let over_storage = match params {
                [state, _] => state.mode == Mode::BorrowMut && is_unit(ret),
                _ => false,
            };
            if !over_storage {
                return Err(syn::Error::new(
                    fold.keyword.span(),
                    format!(
                        "the law `fold` is stated over `f(s: &mut S, x: X)` returning \
                         nothing, and `{fn_ident}` is not of that shape"
                    ),
                ));
            }
            let combine = qref_of(&fold.combine)?;
            let identity = qref_of(&fold.identity)?;
            return Ok(quote! {
                ::acvus_extern::Laws::Fold(::acvus_extern::FoldLaw {
                    combine: #combine,
                    identity: #identity,
                    commutative: #commutative,
                })
            });
        }
        let binary = match params {
            [a, b] if matches!(returning, Returning::Value) => match (a.mode, b.mode) {
                (Mode::Value, Mode::Value) => same_type(&a.ty, &b.ty) && same_type(&a.ty, ret),
                (Mode::Str, Mode::Str) => {
                    matches!(ret, Type::Path(path) if path.path.is_ident("String"))
                }
                _ => false,
            },
            _ => false,
        };
        if !binary {
            let word = &self.first_word;
            return Err(syn::Error::new(
                word.span(),
                format!(
                    "the law `{word}` is stated over `f(a: T, b: T) -> T` or \
                     `f(a: &str, b: &str) -> String`, and `{fn_ident}` is not of that shape"
                ),
            ));
        }
        let associative = self.associative.is_some();
        let identity = match &self.identity {
            None => quote! { ::core::option::Option::None },
            Some(IdentityAttr::Extern(path)) => {
                let qref = qref_of(path)?;
                quote! { ::core::option::Option::Some(::acvus_extern::Identity::Extern(#qref)) }
            }
            Some(IdentityAttr::Const(constant)) => {
                let literal = constant.literal()?;
                quote! { ::core::option::Option::Some(::acvus_extern::Identity::Const(#literal)) }
            }
        };
        Ok(quote! {
            ::acvus_extern::Laws::Binary(::acvus_extern::BinaryLaws {
                associative: #associative,
                commutative: #commutative,
                identity: #identity,
            })
        })
    }
}

impl Parse for IdentityAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let negative = input.peek(Token![-]);
        if negative {
            input.parse::<Token![-]>()?;
        }
        if negative || input.peek(Lit) {
            return Ok(IdentityAttr::Const(ConstIdentity {
                negative,
                lit: input.parse()?,
            }));
        }
        Ok(IdentityAttr::Extern(input.parse()?))
    }
}

impl ConstIdentity {
    fn literal(&self) -> syn::Result<proc_macro2::TokenStream> {
        let refused = || {
            syn::Error::new_spanned(
                &self.lit,
                "a constant identity is an integer, a float, a bool, or a string",
            )
        };
        match &self.lit {
            Lit::Int(int) => {
                let magnitude: i128 = int.base10_parse()?;
                let value = match self.negative {
                    true => -magnitude,
                    false => magnitude,
                };
                Ok(quote! { ::acvus_extern::Literal::Int(#value) })
            }
            Lit::Float(float) => {
                let magnitude: f64 = float.base10_parse()?;
                let value = match self.negative {
                    true => -magnitude,
                    false => magnitude,
                };
                Ok(quote! { ::acvus_extern::Literal::Float(#value) })
            }
            Lit::Bool(b) if !self.negative => Ok(quote! { ::acvus_extern::Literal::Bool(#b) }),
            Lit::Str(s) if !self.negative => {
                Ok(quote! { ::acvus_extern::Literal::String(#s.to_string()) })
            }
            _ => Err(refused()),
        }
    }
}

impl FoldAttr {
    fn parse_after(keyword: Ident, input: ParseStream) -> syn::Result<Self> {
        let content;
        syn::parenthesized!(content in input);
        let mut combine = None;
        let mut identity = None;
        while !content.is_empty() {
            let key: Ident = content.parse()?;
            content.parse::<Token![=]>()?;
            let path: Path = content.parse()?;
            let slot = match key.to_string().as_str() {
                "combine" => &mut combine,
                "identity" => &mut identity,
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown `fold` argument `{other}`: the law is \
                             `fold(combine = g, identity = e)`"
                        ),
                    ));
                }
            };
            if slot.replace(path).is_some() {
                return Err(syn::Error::new(
                    key.span(),
                    format!("`fold` states `{key}` twice"),
                ));
            }
            if !content.is_empty() {
                content.parse::<Token![,]>()?;
            }
        }
        let Some(combine) = combine else {
            return Err(syn::Error::new(
                keyword.span(),
                "the law `fold` names `combine = g`",
            ));
        };
        let Some(identity) = identity else {
            return Err(syn::Error::new(
                keyword.span(),
                "the law `fold` names `identity = e`",
            ));
        };
        Ok(FoldAttr {
            keyword,
            combine,
            identity,
        })
    }
}

/// `g` is the extern `g` in the namespace the registry declares this
/// function under, and `ns::g` the extern `g` in `ns`.
fn qref_of(path: &Path) -> syn::Result<proc_macro2::TokenStream> {
    let refused = || {
        syn::Error::new_spanned(
            path,
            "a law names an extern as `name` or `namespace::name`",
        )
    };
    if path.leading_colon.is_some() || path.segments.iter().any(|s| !s.arguments.is_none()) {
        return Err(refused());
    }
    let names: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
    match names.as_slice() {
        [name] => Ok(quote! {
            ::acvus_extern::QualifiedRef {
                namespace: __ns.map(|__n| __i.intern(__n)),
                name: __i.intern(#name),
            }
        }),
        [ns, name] => Ok(quote! {
            ::acvus_extern::QualifiedRef::qualified(__i.intern(#ns), __i.intern(#name))
        }),
        _ => Err(refused()),
    }
}

fn same_type(a: &Type, b: &Type) -> bool {
    a.to_token_stream().to_string() == b.to_token_stream().to_string()
}

fn is_unit(ty: &Type) -> bool {
    matches!(ty, Type::Tuple(t) if t.elems.is_empty())
}
