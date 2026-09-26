//! `#[extern_fn(law(..))]`: the algebraic laws a declaration states
//! (RFC-0082 rules 2, 3 and 10), and `#[extern_fn(payload(o))]` (rule 3).

use quote::{ToTokens, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Ident, Lit, Path, Token, Type};

use crate::{ExternParam, Mode, Returning};

/// `law(associative, commutative, identity = e)`,
/// `law(fold(combine = g, identity = e), commutative)`,
/// `law(total_order)`, `law(inverse = g)`, `law(equivalence)` or
/// `law(absent = v)`.
pub(crate) struct LawAttr {
    first_word: Ident,
    associative: Option<Ident>,
    commutative: Option<Ident>,
    identity: Option<IdentityAttr>,
    fold: Option<FoldAttr>,
    total_order: Option<Ident>,
    inverse: Option<InverseAttr>,
    equivalence: Option<Ident>,
    absent: Option<AbsentAttr>,
}

struct AbsentAttr {
    keyword: Ident,
    value: Ident,
}

struct InverseAttr {
    keyword: Ident,
    restore: Path,
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
    /// The first law the attribute states.
    pub(crate) fn first_word(&self) -> &Ident {
        &self.first_word
    }

    pub(crate) fn parse_after(keyword: Ident, input: ParseStream) -> syn::Result<Self> {
        let content;
        syn::parenthesized!(content in input);
        let mut first_word: Option<Ident> = None;
        let mut associative = None;
        let mut commutative = None;
        let mut identity = None;
        let mut fold = None;
        let mut total_order = None;
        let mut inverse = None;
        let mut equivalence = None;
        let mut absent = None;
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
                "inverse" => {
                    content.parse::<Token![=]>()?;
                    let restore: Path = content.parse()?;
                    let stated = InverseAttr {
                        keyword: word.clone(),
                        restore,
                    };
                    inverse.replace(stated).is_some()
                }
                "equivalence" => equivalence.replace(word.clone()).is_some(),
                "absent" => {
                    content.parse::<Token![=]>()?;
                    let value: Ident = content.parse()?;
                    let stated = AbsentAttr {
                        keyword: word.clone(),
                        value,
                    };
                    absent.replace(stated).is_some()
                }
                other => {
                    return Err(syn::Error::new(
                        word.span(),
                        format!(
                            "unknown law `{other}`: a law is `associative`, `commutative`, \
                             `identity = e`, `fold(combine = g, identity = e)`, \
                             `total_order`, `inverse = g`, `equivalence`, or `absent = v` \
                             (RFC-0082)"
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
        if let Some(stated) = &inverse
            && (associative.is_some()
                || commutative.is_some()
                || identity.is_some()
                || fold.is_some()
                || total_order.is_some())
        {
            return Err(syn::Error::new(
                stated.keyword.span(),
                "the law `inverse` is a storage read's, and `associative`, `commutative`, \
                 `identity`, `fold` and `total_order` are another form's: a declaration states \
                 one form (RFC-0082)",
            ));
        }
        let alone = [
            equivalence.as_ref(),
            absent.as_ref().map(|stated| &stated.keyword),
        ];
        for stated in alone.into_iter().flatten() {
            let others = [
                associative.is_some(),
                commutative.is_some(),
                identity.is_some(),
                fold.is_some(),
                total_order.is_some(),
                inverse.is_some(),
                equivalence.is_some(),
                absent.is_some(),
            ];
            if others.into_iter().filter(|stated| *stated).count() > 1 {
                return Err(syn::Error::new(
                    stated.span(),
                    format!(
                        "the law `{stated}` is stated alone: a declaration states one form \
                         (RFC-0082)"
                    ),
                ));
            }
        }
        Ok(LawAttr {
            first_word,
            associative,
            commutative,
            identity,
            fold,
            total_order,
            inverse,
            equivalence,
            absent,
        })
    }

    pub(crate) fn checked_laws(
        &self,
        fn_ident: &Ident,
        params: &[&ExternParam],
        ret: &Type,
        returning: &Returning,
        instance_of: Option<&Path>,
        keyed_by: Option<&Ident>,
    ) -> syn::Result<proc_macro2::TokenStream> {
        if let Some(stated) = &self.equivalence {
            return checked_equivalence(stated, fn_ident, params, ret, returning, instance_of);
        }
        if let Some(stated) = &self.absent {
            return checked_absent(stated, fn_ident, params, ret, keyed_by);
        }
        let commutative = self.commutative.is_some();
        if let Some(inverse) = &self.inverse {
            let reads_storage = match params {
                [state] => state.mode == Mode::BorrowMut,
                _ => false,
            };
            let gives_an_option =
                matches!(returning, Returning::Value) && option_payload(ret).is_some();
            if !reads_storage || !gives_an_option {
                return Err(syn::Error::new(
                    inverse.keyword.span(),
                    format!(
                        "the law `inverse` is stated over `f(s: &mut S) -> Option<X>`, and \
                         `{fn_ident}` is not of that shape (RFC-0082 rule 3)"
                    ),
                ));
            }
            let restore = qref_of(&inverse.restore)?;
            return Ok(quote! { ::acvus_extern::Laws::Inverse(#restore) });
        }
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
pub(crate) fn qref_of(path: &Path) -> syn::Result<proc_macro2::TokenStream> {
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
                scope: None,
            }
        }),
        [ns, name] => Ok(quote! {
            ::acvus_extern::QualifiedRef::qualified(__i.intern(#ns), __i.intern(#name))
        }),
        _ => Err(refused()),
    }
}

/// `payload(o)` on `f(o: Option<T>) -> T` (RFC-0082 rule 3): the law
/// `f` states, or the refusal naming the shape.
pub(crate) fn checked_payload(
    named: &Ident,
    fn_ident: &Ident,
    params: &[&ExternParam],
    ret: &Type,
    returning: &Returning,
) -> syn::Result<proc_macro2::TokenStream> {
    let fits = match params {
        [option] => {
            *named == option.name
                && option.mode == Mode::Value
                && matches!(returning, Returning::Value)
                && option_payload(&option.ty).is_some_and(|payload| same_type(payload, ret))
        }
        _ => false,
    };
    if !fits {
        return Err(syn::Error::new(
            named.span(),
            format!(
                "`payload({named})` is stated over `f({named}: Option<T>) -> T`, and \
                 `{fn_ident}` is not of that shape (RFC-0082 rule 3)"
            ),
        ));
    }
    Ok(quote! { ::acvus_extern::Laws::Payload })
}

/// `law(equivalence)` on a `core::eq` instance `f(a: &T, b: &T) -> bool`
/// (RFC-0082 rule 11). That the signature is `core::eq` is Rust's to check:
/// the declaration names `law::stated_over` at its `instance_of`, whose
/// bound only `core::eq` meets.
fn checked_equivalence(
    stated: &Ident,
    fn_ident: &Ident,
    params: &[&ExternParam],
    ret: &Type,
    returning: &Returning,
    instance_of: Option<&Path>,
) -> syn::Result<proc_macro2::TokenStream> {
    let Some(signature) = instance_of else {
        return Err(syn::Error::new(
            stated.span(),
            format!(
                "`law(equivalence)` is stated on a `core::eq` instance, and `{fn_ident}` is no \
                 signature's instance (RFC-0082 rule 11)"
            ),
        ));
    };
    let compares = match params {
        [a, b] => a.mode == Mode::Borrow && b.mode == Mode::Borrow && same_type(&a.ty, &b.ty),
        _ => false,
    };
    let answers = matches!(returning, Returning::Value)
        && matches!(ret, Type::Path(path) if path.path.is_ident("bool"));
    if !compares || !answers {
        return Err(syn::Error::new(
            stated.span(),
            format!(
                "the law `equivalence` is stated over `f(a: &T, b: &T) -> bool`, and \
                 `{fn_ident}` is not of that shape (RFC-0082 rule 11)"
            ),
        ));
    }
    Ok(quote! {
        {
            const _: () = ::acvus_extern::law::stated_over::<#signature>();
            ::acvus_extern::Laws::Equivalence
        }
    })
}

/// `law(absent = v)` on `f(x: &mut M, .., v: V) -> &mut V` whose `reaches`
/// names `x[k]` for its first parameter `x`.
fn checked_absent(
    stated: &AbsentAttr,
    fn_ident: &Ident,
    params: &[&ExternParam],
    ret: &Type,
    keyed_by: Option<&Ident>,
) -> syn::Result<proc_macro2::TokenStream> {
    let shape = || {
        syn::Error::new(
            stated.keyword.span(),
            format!(
                "the law `absent` is stated over `f(x: &mut M, k: K, v: V) -> &mut V` whose \
                 `reaches` names `x[k]`, and `{fn_ident}` is not of that shape"
            ),
        )
    };
    let Some(at) = params.iter().position(|param| stated.value == param.name) else {
        return Err(syn::Error::new(
            stated.value.span(),
            format!("`{}` names no parameter of `{fn_ident}`", stated.value),
        ));
    };
    let default = params[at];
    let lends_the_entry = match (params.first(), ret) {
        (Some(map), Type::Reference(entry)) => {
            map.mode == Mode::BorrowMut
                && at > 0
                && default.mode == Mode::Value
                && entry.mutability.is_some()
                && same_type(&entry.elem, &default.ty)
                && keyed_by.is_some_and(|keyed| *keyed == map.name)
        }
        _ => false,
    };
    if !lends_the_entry {
        return Err(shape());
    }
    Ok(quote! { ::acvus_extern::Laws::Absent { value: #at } })
}

/// `X` of a type written `Option<X>`.
fn option_payload(ty: &Type) -> Option<&Type> {
    let Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return None;
    }
    let segment = &path.path.segments[0];
    if segment.ident != "Option" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    match args.args.iter().collect::<Vec<_>>()[..] {
        [syn::GenericArgument::Type(payload)] => Some(payload),
        _ => None,
    }
}

fn same_type(a: &Type, b: &Type) -> bool {
    a.to_token_stream().to_string() == b.to_token_stream().to_string()
}

fn is_unit(ty: &Type) -> bool {
    matches!(ty, Type::Tuple(t) if t.elems.is_empty())
}
