//! RFC-0076 rule 1 and its one exception: at a runtime that makes values,
//! `Branded` is the only trait whose impl on `Erased<R, T>` depends on `T`.
//! An impl of a trait that requires `Branded` states `Self: Branded` or
//! `Self: Unbranded`, and so exists where `Branded`'s does, and bounds `T`
//! by nothing else.
//!
//! Rust has no bound that forbids a bound, so this reads every trait impl on
//! `Erased` or `Owned` the workspace writes: each `impl` item, and each
//! invocation of a macro whose first argument is one of the two. An `impl`
//! written inside a macro body at `Erased` is refused, since its bounds
//! cannot be read there. A refused program under `compile_fail` is not read.

use std::path::{Path, PathBuf};

use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens;
use syn::parse::{ParseStream, Parser};
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::{
    GenericArgument, GenericParam, Ident, PathArguments, ReturnType, Token, Type, TypeParamBound,
    WherePredicate,
};

/// One trait impl on `Erased` or `Owned`, where it is written, and why it
/// is refused if it is.
struct Found {
    site: String,
    trait_name: String,
    refused: Option<String>,
}

/// A trait impl on `Erased` or `Owned` as the judge reads it.
struct Impl<'a> {
    trait_name: &'a str,
    trait_args: TokenStream,
    /// `Erased<R, T>` or `Owned<R>`, the self type's last segment.
    self_segment: &'a syn::PathSegment,
    params: &'a [GenericParam],
    predicates: &'a [WherePredicate],
    items: TokenStream,
}

/// `bounded: bound`, from a parameter list or a `where` clause.
struct Bound {
    bounded: Type,
    bound: TypeParamBound,
}

/// What a macro invocation is to the judge.
enum Invocation {
    /// Its first argument is not `Erased` or `Owned`.
    Elsewhere,
    /// An impl on `Erased` or `Owned`, refused with the reason if it is.
    OnErased { refused: Option<String> },
}

fn last_segment(ty: &Type) -> Option<&syn::PathSegment> {
    match ty {
        Type::Path(p) if p.qself.is_none() => p.path.segments.last(),
        _ => None,
    }
}

fn type_args(segment: &syn::PathSegment) -> Vec<&Type> {
    match &segment.arguments {
        PathArguments::AngleBracketed(args) => args
            .args
            .iter()
            .filter_map(|a| match a {
                GenericArgument::Type(t) => Some(t),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn single_ident(ty: &Type) -> Option<&Ident> {
    match ty {
        Type::Path(p) if p.qself.is_none() => p.path.get_ident(),
        _ => None,
    }
}

fn mentions(tokens: TokenStream, ident: &Ident) -> bool {
    tokens.into_iter().any(|tt| match tt {
        TokenTree::Ident(i) => i == *ident,
        TokenTree::Group(g) => mentions(g.stream(), ident),
        _ => false,
    })
}

fn bound_trait_name(bound: &TypeParamBound) -> Option<String> {
    match bound {
        TypeParamBound::Trait(t) => t.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    }
}

fn is_static(bound: &TypeParamBound) -> bool {
    matches!(bound, TypeParamBound::Lifetime(l) if l.ident == "static")
}

/// Whether `ty` is `fn() -> var`.
fn is_fn_to(ty: &Type, var: &Ident) -> bool {
    match ty {
        Type::BareFn(f) => {
            f.inputs.is_empty()
                && matches!(&f.output, ReturnType::Type(_, out) if single_ident(out) == Some(var))
        }
        _ => false,
    }
}

/// The reason `imp` is refused, or `None` where it is admitted.
fn judge(imp: Impl<'_>) -> Option<String> {
    let Impl {
        trait_name,
        trait_args,
        self_segment: segment,
        params,
        predicates,
        items,
    } = imp;
    let args = type_args(segment);
    let mut bounds: Vec<Bound> = Vec::new();
    for param in params {
        if let GenericParam::Type(t) = param {
            let ident = &t.ident;
            let bounded: Type = syn::parse_quote! { #ident };
            bounds.extend(t.bounds.iter().map(|b| Bound {
                bounded: bounded.clone(),
                bound: b.clone(),
            }));
        }
    }
    for predicate in predicates {
        if let WherePredicate::Type(p) = predicate {
            bounds.extend(p.bounds.iter().map(|b| Bound {
                bounded: p.bounded_ty.clone(),
                bound: b.clone(),
            }));
        }
    }
    // `TyArg` and the checker's other reads hold at `TypesOnly` alone, which
    // makes no values: rule 1 does not reach there.
    let runtime = args.first().copied().and_then(single_ident);
    let types_only = bounds.iter().any(|Bound { bounded, bound }| {
        runtime.is_some()
            && single_ident(bounded) == runtime
            && bound_trait_name(bound).as_deref() == Some("HoldsNoValues")
    });
    if types_only {
        return None;
    }
    if segment.ident == "Owned" {
        return Some("an impl on `Owned` exists at one `T`, `Never`".into());
    }
    let declared = |ident: &Ident| {
        params
            .iter()
            .any(|p| matches!(p, GenericParam::Type(t) if t.ident == *ident))
    };
    let Some(var) = args
        .get(1)
        .copied()
        .and_then(single_ident)
        .filter(|i| declared(i))
    else {
        return Some("an impl at one `T`, not every `T`".into());
    };
    if mentions(trait_args, var) {
        return Some(format!("the trait's arguments name `{var}`"));
    }
    if mentions(items, var) {
        return Some(format!("an item names `{var}`"));
    }
    for Bound { bounded, bound } in &bounds {
        let bound_name = bound_trait_name(bound);
        if single_ident(bounded) == Some(var) {
            if !is_static(bound) {
                return Some(format!(
                    "`{var}` is bounded by `{}`",
                    bound.to_token_stream()
                ));
            }
        } else if single_ident(bounded).is_some_and(|i| i == "Self")
            || last_segment(bounded).is_some_and(|s| s.ident == "Erased")
        {
            if trait_name == "Branded"
                || !matches!(bound_name.as_deref(), Some("Branded" | "Unbranded"))
            {
                return Some(format!(
                    "`Self` is bounded by `{}`",
                    bound.to_token_stream()
                ));
            }
        } else if is_fn_to(bounded, var) {
            if trait_name != "Branded" || bound_name.as_deref() != Some("FromUnbranded") {
                return Some(format!(
                    "`fn() -> {var}` is bounded by `{}`",
                    bound.to_token_stream()
                ));
            }
        } else if mentions(bounded.to_token_stream(), var) || mentions(bound.to_token_stream(), var)
        {
            return Some(format!(
                "`{}: {}` names `{var}`",
                bounded.to_token_stream(),
                bound.to_token_stream()
            ));
        }
    }
    None
}

fn is_erased(ty: &Type) -> bool {
    last_segment(ty).is_some_and(|s| s.ident == "Erased" || s.ident == "Owned")
}

/// Whether `tokens` hold `for …Erased<` or `for …Owned<`: an impl on
/// either written where its bounds cannot be read.
fn writes_an_impl(tokens: TokenStream) -> bool {
    let flat: Vec<TokenTree> = tokens.into_iter().collect();
    for (i, tt) in flat.iter().enumerate() {
        if let TokenTree::Group(g) = tt
            && writes_an_impl(g.stream())
        {
            return true;
        }
        let TokenTree::Ident(kw) = tt else { continue };
        if kw != "for" {
            continue;
        }
        // A path: `$crate :: Erased <`, `:: acvus_extern :: Erased <`, …
        for (j, next) in flat.iter().enumerate().skip(i + 1).take(8) {
            match next {
                TokenTree::Ident(name) if name == "Erased" || name == "Owned" => {
                    if matches!(flat.get(j + 1), Some(TokenTree::Punct(p)) if p.as_char() == '<') {
                        return true;
                    }
                }
                TokenTree::Ident(_) => {}
                TokenTree::Punct(p) if p.as_char() == ':' || p.as_char() == '$' => {}
                _ => break,
            }
        }
    }
    false
}

/// A macro invocation whose first argument is `Erased` or `Owned`, read as
/// `cross_one_value!` takes it: `Ty, at Rt`, `Ty, [params] where preds`,
/// or `Ty, params`.
fn judge_invocation(name: &str, mac: &syn::Macro) -> Invocation {
    let parser = |input: ParseStream| -> syn::Result<Invocation> {
        // A body that does not open with a type is not an impl's arguments.
        let opens_with_erased = input.fork().parse::<Type>().is_ok_and(|ty| is_erased(&ty));
        if !opens_with_erased {
            let _: TokenStream = input.parse()?;
            return Ok(Invocation::Elsewhere);
        }
        let ty: Type = input.parse()?;
        // A body that is the type alone names it (`parse_quote!`) and writes
        // no impl.
        if input.is_empty() {
            return Ok(Invocation::Elsewhere);
        }
        let mut params: Vec<GenericParam> = Vec::new();
        let mut predicates: Vec<WherePredicate> = Vec::new();
        if input.parse::<Option<Token![,]>>()?.is_some() && !input.is_empty() {
            if input.peek(syn::Ident) && input.fork().parse::<Ident>()? == "at" {
                let _: Ident = input.parse()?;
                let _: Type = input.parse()?;
            } else if input.peek(syn::token::Bracket) {
                let inner;
                syn::bracketed!(inner in input);
                params.extend(Punctuated::<GenericParam, Token![,]>::parse_terminated(
                    &inner,
                )?);
                let _: Token![where] = input.parse()?;
                predicates.extend(Punctuated::<WherePredicate, Token![,]>::parse_terminated(
                    input,
                )?);
            } else {
                params.extend(Punctuated::<GenericParam, Token![,]>::parse_terminated(
                    input,
                )?);
            }
        }
        if !input.is_empty() {
            return Ok(Invocation::OnErased {
                refused: Some("the invocation's arguments are not read".into()),
            });
        }
        // The macro adds `__Rt` itself.
        params.push(syn::parse_quote! { __Rt });
        let self_segment = last_segment(&ty).expect("`is_erased` read a path");
        Ok(Invocation::OnErased {
            refused: judge(Impl {
                trait_name: name,
                trait_args: TokenStream::new(),
                self_segment,
                params: &params,
                predicates: &predicates,
                items: TokenStream::new(),
            }),
        })
    };
    match parser.parse2(mac.tokens.clone()) {
        Ok(invocation) => invocation,
        Err(e) => Invocation::OnErased {
            refused: Some(format!("the invocation's arguments are not read: {e}")),
        },
    }
}

struct Impls<'f> {
    file: &'f Path,
    found: Vec<Found>,
}

impl<'ast> Visit<'ast> for Impls<'_> {
    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        if let Some((_, path, _)) = &item.trait_
            && let Some(self_segment) =
                last_segment(&item.self_ty).filter(|_| is_erased(&item.self_ty))
        {
            let segment = path.segments.last().expect("a trait path has a segment");
            let trait_args = segment.arguments.to_token_stream();
            let params: Vec<GenericParam> = item.generics.params.iter().cloned().collect();
            let predicates: Vec<WherePredicate> = item
                .generics
                .where_clause
                .iter()
                .flat_map(|w| w.predicates.iter().cloned())
                .collect();
            let mut items = TokenStream::new();
            for i in &item.items {
                i.to_tokens(&mut items);
            }
            self.found.push(Found {
                site: format!(
                    "{}: impl {} for {}",
                    self.file.display(),
                    segment.ident,
                    item.self_ty.to_token_stream()
                ),
                trait_name: segment.ident.to_string(),
                refused: judge(Impl {
                    trait_name: &segment.ident.to_string(),
                    trait_args,
                    self_segment,
                    params: &params,
                    predicates: &predicates,
                    items,
                }),
            });
        }
        syn::visit::visit_item_impl(self, item);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        let name = mac
            .path
            .segments
            .last()
            .expect("a macro path has a segment")
            .ident
            .to_string();
        match judge_invocation(&name, mac) {
            Invocation::OnErased { refused } => self.found.push(Found {
                site: format!("{}: {name}!({})", self.file.display(), mac.tokens),
                trait_name: name,
                refused,
            }),
            Invocation::Elsewhere if writes_an_impl(mac.tokens.clone()) => self.found.push(Found {
                site: format!("{}: {name}!{{…}}", self.file.display()),
                trait_name: name,
                refused: Some(
                    "an impl on `Erased` in a macro body, whose bounds are not read".into(),
                ),
            }),
            Invocation::Elsewhere => {}
        }
        syn::visit::visit_macro(self, mac);
    }
}

fn read(file: &Path, source: &str) -> Vec<Found> {
    let parsed = syn::parse_file(source)
        .unwrap_or_else(|e| panic!("{} does not parse: {e}", file.display()));
    let mut impls = Impls {
        file,
        found: Vec::new(),
    };
    impls.visit_file(&parsed);
    impls.found
}

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{} is not readable: {e}", dir.display()));
    for entry in entries {
        let entry = entry.expect("a directory entry");
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let kind = entry.file_type().expect("a file type");
        if kind.is_dir() {
            if name.starts_with('.') || name == "target" || name == "compile_fail" {
                continue;
            }
            sources(&path, out);
        } else if kind.is_file() && name.ends_with(".rs") {
            out.push(path);
        }
    }
}

fn refusals(found: &[Found]) -> Vec<String> {
    found
        .iter()
        .filter_map(|f| f.refused.as_ref().map(|why| format!("{}: {why}", f.site)))
        .collect()
}

#[test]
fn only_branded_on_erased_depends_on_t() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("acvus-extern is in the workspace");
    let mut files = Vec::new();
    sources(root, &mut files);
    let mut found = Vec::new();
    for file in &files {
        let source = std::fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("{} is not readable: {e}", file.display()));
        let relative = file
            .strip_prefix(root)
            .expect("`sources` walked under the root");
        found.extend(read(relative, &source));
    }
    let refused = refusals(&found);
    assert!(refused.is_empty(), "{}", refused.join("\n"));
    // The read reaches `erased.rs`: the exception itself, and an impl whose
    // trait requires `Branded`.
    for name in ["Branded", "OneValue", "Stored", "cross_one_value"] {
        assert!(
            found.iter().any(|f| f.trait_name == name),
            "no impl of `{name}` on `Erased` was read"
        );
    }
}

/// A source the judge reads, and the reason it must give.
struct Case {
    source: &'static str,
    refused_for: &'static str,
}

#[test]
fn a_bound_on_t_is_refused() {
    let cases = [
        Case {
            source: "impl<R, T> Clone for Erased<R, T> where R: Runtime, T: Clone { fn clone(&self) -> Self { todo!() } }",
            refused_for: "`T` is bounded by `Clone`",
        },
        Case {
            source: "impl<R, T: Unbranded> OneValue<R> for Erased<R, T> where R: Runtime {}",
            refused_for: "`T` is bounded by `Unbranded`",
        },
        Case {
            source: "impl<R, T> Stored<R> for Erased<R, T> where R: Runtime, Self: Stored<R> {}",
            refused_for: "`Self` is bounded by `Stored < R >`",
        },
        Case {
            source: "impl<R, T> Foo for Erased<R, T> where R: Runtime, Vec<T>: Clone {}",
            refused_for: "`Vec < T >: Clone` names `T`",
        },
        Case {
            source: "impl<R, T> From<T> for Erased<R, T> where R: Runtime {}",
            refused_for: "the trait's arguments name `T`",
        },
        Case {
            source: "impl<R, T> Foo for Erased<R, T> where R: Runtime, fn() -> T: FromUnbranded {}",
            refused_for: "`fn() -> T` is bounded by `FromUnbranded`",
        },
        Case {
            source: "impl<R, T> Foo for Erased<R, T> where R: Runtime, T: 'static { fn id() -> TypeId { TypeId::of::<T>() } }",
            refused_for: "an item names `T`",
        },
        Case {
            source: "impl<R> Foo for Owned<R> where R: Runtime {}",
            refused_for: "an impl on `Owned`",
        },
        Case {
            source: "impl<R> Foo for Erased<R, i64> where R: Runtime {}",
            refused_for: "an impl at one `T`",
        },
        Case {
            source: "cross_one_value!(Erased<__Rt, T>, T: 'static + Clone);",
            refused_for: "`T` is bounded by `Clone`",
        },
        Case {
            source: "macro_rules! m { () => { impl<R, T> Foo for $crate::Erased<R, T> {} } }",
            refused_for: "in a macro body",
        },
    ];
    for Case {
        source,
        refused_for,
    } in cases
    {
        let refused = refusals(&read(Path::new("case.rs"), source));
        assert!(
            refused.iter().any(|r| r.contains(refused_for)),
            "{source}\nwas not refused for {refused_for:?}: {refused:?}"
        );
    }
    let admitted = [
        "unsafe impl<R, T> Branded for Erased<R, T> where R: Runtime, T: 'static, fn() -> T: brand::FromUnbranded { type At<'a> = Self; }",
        "impl<R, T> Stored<R> for Erased<R, T> where R: Runtime, T: 'static, Self: crate::Unbranded {}",
        "cross_one_value!(Erased<__Rt, T>, [T: 'static] where Self: crate::Branded,);",
        "impl<R, T> TyArg for Erased<R, T> where R: HoldsNoValues, T: TyArg + Unbranded {}",
    ];
    for source in admitted {
        let refused = refusals(&read(Path::new("case.rs"), source));
        assert!(refused.is_empty(), "{source}\nwas refused: {refused:?}");
    }
}
