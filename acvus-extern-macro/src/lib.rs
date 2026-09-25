//! Proc macros for acvus-extern. See RFC-0023.
//!
//! - `#[extern_fn]`: a Rust function declares an ExternFn.
//! - `#[derive(ExternType)]`: a Rust struct declares an extension type.
//! - `#[derive(TyArg)]`: a Rust struct declares a structural object type; a
//!   Rust enum declares the language's enum of the same name.
//! - `extern_registry!`: the items one registry contributes.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    Attribute, DeriveInput, FnArg, GenericParam, Ident, ItemFn, LitInt, LitStr, Pat, Path, ReturnType,
    Token, Type, TypeParamBound, parse_macro_input,
};

mod ensures;
mod reaches;
mod flows;
mod generics;
mod law;
mod subst;

use generics::{VarKind, Vars, signature_path};

// -- #[extern_fn] ----------------------------------------------------

#[proc_macro_attribute]
pub fn extern_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as ExternFnAttr);
    let mut func = parse_macro_input!(item as ItemFn);
    match generate_extern_fn(attr, &mut func) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// `#[extern_fn(name = "...", ns = "...", effect = pure | idempotent | E)]`.
struct ExternFnAttr {
    name: Option<LitStr>,
    /// The shared signature this function is an instance of (RFC-0019).
    instance_of: Option<Path>,
    effect: Option<Ident>,
    /// `commutative`: two calls of this function in either order are the
    /// same program (RFC-0013).
    commutative: bool,
    /// `heavy`: the call is offloaded to a blocking pool and awaited
    /// (RFC-0046).
    heavy: bool,
    /// `sync = f`: the plain `fn` that runs this declaration when the
    /// call's task is `Sync`, declared beside the `async fn` at the same
    /// signature (RFC-0046).
    sync: Option<Ident>,
    law: Option<law::LawAttr>,
    ensures: Option<ensures::EnsuresAttr>,
    reaches: Option<reaches::ReachesAttr>,
    /// `returns` or `total` (RFC-0082 rules 8 and 9). Either is the
    /// author's promise, which nothing here checks.
    returns: Option<StatedReturn>,
    /// `copies(x)`: the result is a value equal to what reference parameter
    /// `x` lends (RFC-0082 rule 10), the author's promise.
    copies: Option<Ident>,
    /// `payload(o)`: on `f(o: Option<T>) -> T`, `f(Some(x))` is `x` and
    /// `f(None)` traps (RFC-0082 rule 3), the author's promise.
    payload: Option<Ident>,
    /// `cost = N`: one call weighs `N` ticks of the backend's table
    /// (RFC-0066 rule 8), in place of its family's row.
    cost: Option<LitInt>,
    /// `dynamic`: the result is typed by each call site (RFC-0097 rule 3).
    dynamic: bool,
}

#[derive(Clone, Copy)]
enum StatedReturn {
    ReturnsOrTraps,
    Total,
}

impl StatedReturn {
    fn word(self) -> &'static str {
        match self {
            StatedReturn::ReturnsOrTraps => "returns",
            StatedReturn::Total => "total",
        }
    }
}

impl Parse for ExternFnAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut out = ExternFnAttr {
            name: None,
            instance_of: None,
            effect: None,
            commutative: false,
            heavy: false,
            sync: None,
            law: None,
            ensures: None,
            reaches: None,
            returns: None,
            copies: None,
            payload: None,
            cost: None,
            dynamic: false,
        };
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            if key == "commutative" {
                out.commutative = true;
                if !input.is_empty() {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }
            let stated = match key.to_string().as_str() {
                "returns" => Some(StatedReturn::ReturnsOrTraps),
                "total" => Some(StatedReturn::Total),
                _ => None,
            };
            if let Some(stated) = stated {
                if let Some(earlier) = out.returns {
                    let message = match key == earlier.word() {
                        true => format!("`{key}` is stated twice"),
                        false => format!(
                            "`{key}` beside `{}`: `total` states `returns`, so write one",
                            earlier.word()
                        ),
                    };
                    return Err(syn::Error::new(key.span(), message));
                }
                out.returns = Some(stated);
                if !input.is_empty() {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }
            if key == "dynamic" {
                out.dynamic = true;
                if !input.is_empty() {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }
            if key == "heavy" {
                out.heavy = true;
                if !input.is_empty() {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }
            if key == "law" {
                if out.law.is_some() {
                    return Err(syn::Error::new(key.span(), "`law(..)` is stated twice"));
                }
                out.law = Some(law::LawAttr::parse_after(key, input)?);
                if !input.is_empty() {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }
            if key == "ensures" {
                if out.ensures.is_some() {
                    return Err(syn::Error::new(
                        key.span(),
                        "`ensures(..)` is stated twice: state every relation in one",
                    ));
                }
                out.ensures = Some(ensures::EnsuresAttr::parse_after(&key, input)?);
                if !input.is_empty() {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }
            if key == "copies" {
                if out.copies.is_some() {
                    return Err(syn::Error::new(key.span(), "`copies(..)` is stated twice"));
                }
                let content;
                syn::parenthesized!(content in input);
                out.copies = Some(content.parse()?);
                if !content.is_empty() {
                    return Err(syn::Error::new(
                        content.span(),
                        "`copies(x)` names one reference parameter",
                    ));
                }
                if !input.is_empty() {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }
            if key == "payload" {
                if out.payload.is_some() {
                    return Err(syn::Error::new(key.span(), "`payload(..)` is stated twice"));
                }
                let content;
                syn::parenthesized!(content in input);
                out.payload = Some(content.parse()?);
                if !content.is_empty() {
                    return Err(syn::Error::new(
                        content.span(),
                        "`payload(o)` names one parameter",
                    ));
                }
                if !input.is_empty() {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }
            if key == "reaches" {
                if out.reaches.is_some() {
                    return Err(syn::Error::new(
                        key.span(),
                        "`reaches(..)` is stated twice: state every place in one",
                    ));
                }
                out.reaches = Some(reaches::ReachesAttr::parse_after(&key, input)?);
                if !input.is_empty() {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }
            input.parse::<Token![=]>()?;
            if key == "name" {
                out.name = Some(input.parse()?);
            } else if key == "instance_of" {
                out.instance_of = Some(input.parse()?);
            } else if key == "effect" {
                out.effect = Some(input.parse()?);
            } else if key == "sync" {
                out.sync = Some(input.parse()?);
            } else if key == "cost" {
                if out.cost.is_some() {
                    return Err(syn::Error::new(key.span(), "`cost` is stated twice"));
                }
                let weight: LitInt = input.parse()?;
                weight.base10_parse::<u64>().map_err(|_| {
                    syn::Error::new(
                        weight.span(),
                        "`cost` is a whole number of ticks that fits a `u64`",
                    )
                })?;
                out.cost = Some(weight);
            } else {
                return Err(syn::Error::new(
                    key.span(),
                    "expected `name`, `instance_of`, `effect`, `commutative`, `heavy`, `sync`, `law`, \
                     `ensures`, `reaches`, `copies`, `payload`, `returns`, `total`, `cost` or \
                     `dynamic`",
                ));
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(out)
    }
}

/// An acvus parameter of the declared function.
struct ExternParam {
    name: String,
    /// The acvus type: for `&T` and `&mut T`, the `T`.
    ty: Type,
    mode: Mode,
}

/// How the Rust parameter takes its argument (RFC-0023 rule 5).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Value,
    Borrow,
    BorrowMut,
    Str,
    /// `&[T]` / `&mut [T]`: the language's slice, taken as Rust's
    /// (RFC-0047, RFC-0068 rule 4).
    Slice,
    SliceMut,
    /// `SRef<'_>` / `SMut<'_>`, which `#[derive(TyArg)]` emits beside the
    /// struct (RFC-0050 rule 6).
    Projection,
}

impl Mode {
    /// A parameter that occupies the register pair a view or a slice is,
    /// or a projection: not one value, so no mono glue and no `Ctx`
    /// receiver can carry it (RFC-0067 rule 8).
    fn is_two_words(self) -> bool {
        matches!(
            self,
            Mode::Str | Mode::Slice | Mode::SliceMut | Mode::Projection
        )
    }

    fn lends_its_storage(self) -> bool {
        match self {
            Mode::Borrow
            | Mode::BorrowMut
            | Mode::Str
            | Mode::Slice
            | Mode::SliceMut
            | Mode::Projection => true,
            Mode::Value => false,
        }
    }

    /// The acvus type of a parameter whose Rust type is `ty` under this
    /// mode, with `rt` as the runtime a reference carrier names.
    fn acvus_ty(self, ty: &Type, rt: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        match self {
            Mode::Value => quote! { #ty },
            Mode::Borrow => {
                quote! { ::acvus_extern::Ref<'static, #ty, ::acvus_extern::Shared, #rt> }
            }
            Mode::BorrowMut => {
                quote! { ::acvus_extern::Ref<'static, #ty, ::acvus_extern::Mut, #rt> }
            }
            Mode::Str => quote! { ::acvus_extern::StrView },
            Mode::Slice => {
                quote! { ::acvus_extern::Slice<'static, #ty, ::acvus_extern::Shared, #rt> }
            }
            Mode::SliceMut => {
                quote! { ::acvus_extern::Slice<'static, #ty, ::acvus_extern::Mut, #rt> }
            }
            Mode::Projection => {
                let at_static = at_static(ty);
                quote! { #at_static }
            }
        }
    }
}

fn is_str(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.qself.is_none() && p.path.is_ident("str"))
}

/// How the Rust result crosses back, which is the `Mode` of the return
/// position.
#[derive(Clone, PartialEq, Eq)]
enum Returning {
    Value,
    Str,
    /// A Rust borrow of a parameter, at the carrier the declaration names
    /// for it (RFC-0047 rule 3, RFC-0068 rule 4).
    Lent(LentShape),
    /// `Finished<'call, T, Rt>`, the script's `Option<T>` (RFC-0097 rule 3).
    Finished(Type),
}

/// A returned borrow: `&T`, `&mut T`, `&[T]`, `&mut [T]`, or an `Option`
/// of one of the first two.
#[derive(Clone, PartialEq, Eq)]
struct LentShape {
    elem: Type,
    mutable: bool,
    slice: bool,
    option: bool,
}

impl LentShape {
    fn of(ty: &Type) -> Option<Self> {
        if let Some(inner) = option_payload(ty) {
            let mut shape = Self::of(inner)?;
            if shape.slice || shape.option {
                return None;
            }
            shape.option = true;
            return Some(shape);
        }
        let Type::Reference(r) = ty else {
            return None;
        };
        if is_str(&r.elem) {
            return None;
        }
        let mutable = r.mutability.is_some();
        match &*r.elem {
            Type::Slice(s) => Some(Self {
                elem: (*s.elem).clone(),
                mutable,
                slice: true,
                option: false,
            }),
            elem => Some(Self {
                elem: elem.clone(),
                mutable,
                slice: false,
                option: false,
            }),
        }
    }

    /// The carrier at `elem` filled as `filled`, naming the runtime `rt`.
    fn carrier(
        &self,
        filled: &proc_macro2::TokenStream,
        rt: &proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        let one = self.one_carrier(filled, rt);
        match self.option {
            true => quote! { ::core::option::Option<#one> },
            false => one,
        }
    }

    /// The carrier of the borrow itself, under any `Option`.
    fn one_carrier(
        &self,
        filled: &proc_macro2::TokenStream,
        rt: &proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        let loan = match self.mutable {
            true => quote! { ::acvus_extern::Mut },
            false => quote! { ::acvus_extern::Shared },
        };
        match self.slice {
            true => quote! { ::acvus_extern::Slice<'static, #filled, #loan, #rt> },
            false => quote! { ::acvus_extern::Ref<'static, #filled, #loan, #rt> },
        }
    }
}

/// The payload type of `Option<..>`, spelled as a path ending in `Option`.
fn option_payload(ty: &Type) -> Option<&Type> {
    let Type::Path(p) = ty else {
        return None;
    };
    let seg = p.path.segments.last()?;
    if seg.ident != "Option" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    let mut args = args.args.iter();
    match (args.next(), args.next()) {
        (Some(syn::GenericArgument::Type(t)), None) => Some(t),
        _ => None,
    }
}

impl Returning {
    fn of(ty: &Type, finished: Option<&SettledWritten>) -> Self {
        if let Some(finished) = finished {
            return Returning::Finished(finished.settled.clone());
        }
        match ty {
            Type::Reference(r) if is_str(&r.elem) => Returning::Str,
            _ => match LentShape::of(ty) {
                Some(shape) => Returning::Lent(shape),
                None => Returning::Value,
            },
        }
    }

    /// Whether the result borrows a parameter's storage.
    fn lends(&self) -> bool {
        matches!(self, Returning::Str | Returning::Lent(_))
    }

    /// The type the declaration's fills are applied to: the element of a
    /// borrow, the whole type otherwise.
    fn filled(&self, ret: &Type) -> Type {
        match self {
            Returning::Lent(shape) => shape.elem.clone(),
            Returning::Finished(settled) => settled.clone(),
            Returning::Value | Returning::Str => ret.clone(),
        }
    }

    /// The acvus type of a result whose `filled` type crosses as `filled`.
    fn acvus_ty(
        &self,
        filled: &proc_macro2::TokenStream,
        rt: &proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        match self {
            Returning::Value => quote! { #filled },
            Returning::Str => quote! { ::acvus_extern::StrView },
            Returning::Lent(shape) => shape.carrier(filled, rt),
            Returning::Finished(_) => quote! { ::core::option::Option<#filled> },
        }
    }

    /// The `Ret` marker the glue is built with, where `val` is the `Val`
    /// a result that crosses as itself takes.
    fn marker(
        &self,
        val: &proc_macro2::TokenStream,
        filled: &proc_macro2::TokenStream,
        rt: &proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        match self {
            Returning::Value => quote! { #val },
            Returning::Str => quote! { ::acvus_extern::RetStr },
            Returning::Lent(shape) => {
                let carrier = shape.carrier(filled, rt);
                quote! { ::acvus_extern::RetLent<#carrier> }
            }
            Returning::Finished(_) => quote! { ::acvus_extern::RetFinished },
        }
    }
}

/// A projection is a borrow of one declared aggregate the caller lent: its
/// arguments are the borrow's lifetime and, for an enum's exclusive form, the
/// runtime, and it names no type it carries. A carrier or an extension type
/// with a region parameter is a value that names what it carries beside its
/// lifetime, `Ref<'a, T, M, Rt>` or `Refs<'a, C, I, Rt>`, and crosses by value
/// (RFC-0079 rule 6). The test reads the arguments and not the type's name,
/// because reading a crossing out of a name is what RFC-0059 withdrew
/// `returns_slice` for: an alias defeated it.
fn borrows_caller(ty: &Type, runtime: Option<&Ident>) -> bool {
    let Type::Path(p) = ty else {
        return false;
    };
    let Some(syn::PathArguments::AngleBracketed(args)) =
        p.path.segments.last().map(|segment| &segment.arguments)
    else {
        return false;
    };
    let lends = args
        .args
        .iter()
        .any(|arg| matches!(arg, syn::GenericArgument::Lifetime(_)));
    let carries = args.args.iter().any(|arg| match arg {
        syn::GenericArgument::Lifetime(_) => false,
        syn::GenericArgument::Type(Type::Path(t)) => {
            !runtime.is_some_and(|runtime| t.qself.is_none() && t.path.is_ident(runtime))
        }
        _ => true,
    });
    lends && !carries
}

use subst::at_static;

/// A parameter marked `#[state]`: supplied when the registry is built,
/// held by the handler, and absent from the acvus type (RFC-0021).
struct StateParam {
    ident: Ident,
    ty: Type,
}

/// A declaration states a requirement by taking it (RFC-0067 rule 1).
struct RequiredParam {
    signature: Type,
    var: Ident,
    task: Type,
}

/// One Rust parameter after the runtime, in declaration order.
enum RustParam {
    Acvus(ExternParam),
    State(StateParam),
    /// `InstanceOf<S, I, Rt, T>`: a requirement standing at its type, which
    /// is no acvus parameter.
    Required(RequiredParam),
    /// `Instance<S, I, Rt, T>`: the acvus parameter at `I`, owned by the
    /// requirement the checker chose for its type.
    Owning {
        param: ExternParam,
        required: RequiredParam,
    },
    /// `Args<'_, (A, ..), Rt>`: one acvus parameter per member, each taken
    /// by value, held together by one view (RFC-0097 rule 1).
    Args(ArgsParam),
    /// `Output<'call, T, Rt>`, a `dynamic` declaration's result as the call
    /// fills it (RFC-0097 rule 3).
    Output(OutputParam),
}

struct OutputParam {
    written: Type,
    settled: Type,
    runtime: Type,
}

struct ArgsParam {
    ident: Ident,
    written: Type,
    runtime: Type,
    members: Vec<ExternParam>,
}

/// An `Args<'_, (..), Rt>` type's members and runtime, as written.
struct ArgsWritten {
    members: Vec<Type>,
    runtime: Type,
}

impl RustParam {
    /// The acvus parameters this Rust parameter takes, in run order.
    fn acvus(&self) -> &[ExternParam] {
        match self {
            RustParam::Acvus(param) | RustParam::Owning { param, .. } => std::slice::from_ref(param),
            RustParam::Args(args) => &args.members,
            RustParam::State(_) | RustParam::Required(_) | RustParam::Output(_) => &[],
        }
    }
}

const ARGS_SHAPE: &str = "an `Args` parameter is written `Args<'_, (A, B, ..), Rt>`: the call's \
     lifetime, a tuple of this declaration's own `Var<kind::Type>` parameters, and its runtime \
     (RFC-0097 rule 1)";

/// What an `Args<'_, (..), Rt>` type names, or `None` for any other type.
fn args_written(ty: &Type) -> syn::Result<Option<ArgsWritten>> {
    let Type::Path(p) = ty else {
        return Ok(None);
    };
    let Some(seg) = p.path.segments.last() else {
        return Ok(None);
    };
    if seg.ident != "Args" {
        return Ok(None);
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return Err(syn::Error::new_spanned(seg, ARGS_SHAPE));
    };
    let written: Vec<&syn::GenericArgument> = args.args.iter().collect();
    let [
        syn::GenericArgument::Lifetime(_),
        syn::GenericArgument::Type(members),
        syn::GenericArgument::Type(runtime),
    ] = written.as_slice()
    else {
        return Err(syn::Error::new_spanned(seg, ARGS_SHAPE));
    };
    let members: Vec<Type> = match members {
        Type::Tuple(tuple) => tuple.elems.iter().cloned().collect(),
        other => return Err(syn::Error::new_spanned(other, ARGS_SHAPE)),
    };
    if members.is_empty() {
        return Err(syn::Error::new_spanned(
            ty,
            "an `Args` of no member views no argument: a declaration that takes none takes no `Args` \
             (RFC-0097 rule 1)",
        ));
    }
    Ok(Some(ArgsWritten {
        members,
        runtime: runtime.clone(),
    }))
}

const RUST_FN_SHAPE: &str = "a `RustFn` result is written `RustFn<(A, B, ..), R, Rt>`: a tuple of \
     one to eight parameter types, the result type, and this declaration's runtime (RFC-0097 rule 2)";

struct RustFnWritten {
    params: Vec<Type>,
}

fn rust_fn_written(ty: &Type) -> syn::Result<Option<RustFnWritten>> {
    let Type::Path(p) = ty else {
        return Ok(None);
    };
    let Some(seg) = p.path.segments.last() else {
        return Ok(None);
    };
    if seg.ident != "RustFn" {
        return Ok(None);
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return Err(syn::Error::new_spanned(seg, RUST_FN_SHAPE));
    };
    let written: Vec<&syn::GenericArgument> = args.args.iter().collect();
    let [
        syn::GenericArgument::Type(Type::Tuple(params)),
        syn::GenericArgument::Type(_),
        syn::GenericArgument::Type(_),
    ] = written.as_slice()
    else {
        return Err(syn::Error::new_spanned(seg, RUST_FN_SHAPE));
    };
    if params.elems.is_empty() || params.elems.len() > 8 {
        return Err(syn::Error::new_spanned(params, RUST_FN_SHAPE));
    }
    Ok(Some(RustFnWritten {
        params: params.elems.iter().cloned().collect(),
    }))
}

fn rust_fn_effect(
    attr: &ExternFnAttr,
    vars: &Vars,
    written: &RustFnWritten,
    ret: &Type,
    fn_ident: &Ident,
    is_async: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    if let Some(variable) = written.params.iter().find(|param| vars.mentions_ty_var(param)) {
        return Err(syn::Error::new_spanned(
            variable,
            "a `RustFn` parameter is a concrete type, not one of this declaration's variables: \
             `Args` lends each argument at the type its parameter names, and a call of a function \
             value settles none (RFC-0097 rule 2)",
        ));
    }
    if vars.mentions_mono(ret) {
        return Err(syn::Error::new_spanned(
            ret,
            "a `RustFn` result names no `Monomorphize` variable: its body is one Rust closure, \
             compiled once (RFC-0097 rule 2)",
        ));
    }
    let task = match (is_async, attr.heavy) {
        (true, _) => Some("an `async fn`"),
        (false, true) => Some("`heavy`"),
        (false, false) => None,
    };
    let suspends = |declared: &str| {
        syn::Error::new(
            fn_ident.span(),
            format!(
                "`{fn_ident}` returns a `RustFn`, whose function type carries the effect this \
                 declaration states, and {declared} states a task above `Sync`: the `RustFn`'s \
                 body would be typed as one that suspends, and a body that suspends is a later \
                 step (RFC-0097 rule 2). Declare a plain `fn` at `effect = pure`, `idempotent` \
                 or `opaque`."
            ),
        )
    };
    if let Some(declared) = task {
        return Err(suspends(declared));
    }
    let commutes = if attr.commutative {
        quote! { .commutative() }
    } else {
        quote! {}
    };
    match &attr.effect {
        Some(e) if e == "pure" => Ok(quote! { ::acvus_extern::Effect::PURE }),
        Some(e) if e == "idempotent" => Ok(quote! { ::acvus_extern::Effect::IDEMPOTENT #commutes }),
        None => Ok(quote! { ::acvus_extern::Effect::OPAQUE #commutes }),
        Some(e) if e == "opaque" => Ok(quote! { ::acvus_extern::Effect::OPAQUE #commutes }),
        Some(_) => Err(suspends("an effect variable, whose task is the one it is instantiated at,")),
    }
    .map(|effect| quote! { ::acvus_extern::EffectTerm::Known(#effect) })
}

const OUTPUT_SHAPE: &str = "an `Output` parameter is written `Output<'call, T, Rt>`: the call's \
     lifetime, the type variable its `Finished` result names, and this declaration's runtime \
     (RFC-0097 rule 3)";

const FINISHED_SHAPE: &str = "a `dynamic` declaration returns `Finished<'call, T, Rt>`: the \
     call's lifetime, one of its own `Var<kind::Type>` parameters, whose script type is \
     `Option<T>`, and its runtime (RFC-0097 rule 3)";

/// A `Finished<'call, T, Rt>` or `Output<'call, T, Rt>` type's `T` and `Rt`.
struct SettledWritten {
    settled: Type,
    runtime: Type,
}

fn settled_written(ty: &Type, name: &str, shape: &str) -> syn::Result<Option<SettledWritten>> {
    let Type::Path(p) = ty else {
        return Ok(None);
    };
    let Some(seg) = p.path.segments.last() else {
        return Ok(None);
    };
    if seg.ident != name {
        return Ok(None);
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return Err(syn::Error::new_spanned(seg, shape));
    };
    let written: Vec<&syn::GenericArgument> = args.args.iter().collect();
    let [
        syn::GenericArgument::Lifetime(_),
        syn::GenericArgument::Type(settled),
        syn::GenericArgument::Type(runtime),
    ] = written.as_slice()
    else {
        return Err(syn::Error::new_spanned(seg, shape));
    };
    Ok(Some(SettledWritten {
        settled: settled.clone(),
        runtime: runtime.clone(),
    }))
}

/// The position among the declaration's type variables of the one a
/// `dynamic` declaration's result names, or `None` for any other
/// declaration; what `dynamic`, `Output` and `Finished` stand beside is
/// refused (RFC-0097 rule 3).
fn dynamic_result(
    rust_params: &[RustParam],
    vars: &Vars,
    attr: &ExternFnAttr,
    ret: &Type,
    finished: Option<&SettledWritten>,
    fn_ident: &Ident,
    is_async: bool,
    is_coercion: bool,
) -> syn::Result<Option<usize>> {
    let mut outputs = rust_params.iter().filter_map(|p| match p {
        RustParam::Output(output) => Some(output),
        RustParam::Acvus(_)
        | RustParam::State(_)
        | RustParam::Required(_)
        | RustParam::Owning { .. }
        | RustParam::Args(_) => None,
    });
    let output = outputs.next();
    if let Some(second) = outputs.next() {
        return Err(syn::Error::new_spanned(
            &second.written,
            "a second `Output`: a call fills one result (RFC-0097 rule 3)",
        ));
    }
    let (finished, output) = match (attr.dynamic, finished, output) {
        (false, None, None) => return Ok(None),
        (false, Some(_), _) => {
            return Err(syn::Error::new_spanned(
                ret,
                "a `Finished` result is a `dynamic` declaration's, whose script type the call \
                 site settles: write `#[extern_fn(dynamic, ..)]` (RFC-0097 rule 3)",
            ));
        }
        (false, None, Some(output)) => {
            return Err(syn::Error::new_spanned(
                &output.written,
                "an `Output` fills a `dynamic` declaration's result: write \
                 `#[extern_fn(dynamic, ..)]` and return the `Finished` it builds (RFC-0097 rule 3)",
            ));
        }
        (true, None, _) => return Err(syn::Error::new_spanned(ret, FINISHED_SHAPE)),
        (true, Some(_), None) => {
            return Err(syn::Error::new(
                fn_ident.span(),
                format!(
                    "`{fn_ident}` is `dynamic` and takes no `Output`: its `Finished` comes only \
                     from the `Output<'call, T, Rt>` the call hands it (RFC-0097 rule 3)"
                ),
            ));
        }
        (true, Some(finished), Some(output)) => (finished, output),
    };
    let settled = &finished.settled;
    let variable = match settled {
        Type::Path(p) if p.qself.is_none() => p.path.get_ident().and_then(|id| vars.lookup(id)),
        _ => None,
    };
    let Some((VarKind::Ty, at)) = variable else {
        return Err(syn::Error::new_spanned(settled, FINISHED_SHAPE));
    };
    if quote! { #settled }.to_string() != {
        let at = &output.settled;
        quote! { #at }.to_string()
    } {
        return Err(syn::Error::new_spanned(&output.settled, OUTPUT_SHAPE));
    }
    let names_the_runtime = |runtime: &Type| {
        matches!(runtime, Type::Path(p)
            if p.qself.is_none() && vars.runtime_ident().is_some_and(|rt| p.path.is_ident(rt)))
    };
    if !names_the_runtime(&finished.runtime) {
        return Err(syn::Error::new_spanned(&finished.runtime, FINISHED_SHAPE));
    }
    if !names_the_runtime(&output.runtime) {
        return Err(syn::Error::new_spanned(&output.runtime, OUTPUT_SHAPE));
    }
    if vars.mono_var().is_some() {
        return Err(syn::Error::new(
            fn_ident.span(),
            "a `dynamic` declaration is one Rust body that fills whatever type its site \
             settles, and a `Monomorphize` variable asks for one body per member type \
             (RFC-0041, RFC-0097 rule 3)",
        ));
    }
    let task = match (is_async, attr.heavy, &attr.sync) {
        (true, _, _) => Some("an `async fn`"),
        (false, true, _) => Some("`heavy`"),
        (false, false, Some(_)) => Some("`sync =`"),
        (false, false, None) => None,
    };
    if let Some(task) = task {
        return Err(syn::Error::new(
            fn_ident.span(),
            format!(
                "`{fn_ident}` is `dynamic` and states {task}: its `Output` borrows the call's \
                 site table and runtime, and a body that suspends or is offloaded holding one \
                 is not built. Declare a plain `fn` (RFC-0097 rule 3)."
            ),
        ));
    }
    if let Some(signature) = &attr.instance_of {
        return Err(syn::Error::new_spanned(
            signature,
            "an instance of a shared signature returns at the signature's own type, and a \
             `dynamic` result has none until its site settles it (RFC-0097 rule 3)",
        ));
    }
    if is_coercion {
        return Err(syn::Error::new(
            fn_ident.span(),
            "an extern_cast or extern_view converts to the one type its result names, and a \
             `dynamic` result names none (RFC-0023 rule 8, RFC-0097 rule 3)",
        ));
    }
    let stated = [
        attr.law.as_ref().map(|law| (law.first_word().span(), "a law")),
        attr.ensures.as_ref().map(|_| (fn_ident.span(), "`ensures`")),
        attr.copies.as_ref().map(|named| (named.span(), "`copies`")),
        attr.payload.as_ref().map(|named| (named.span(), "`payload`")),
    ];
    if let Some((span, what)) = stated.into_iter().flatten().next() {
        return Err(syn::Error::new(
            span,
            format!(
                "{what} is stated over the value a declaration returns, and a `dynamic` \
                 declaration returns a `Finished`, whose value its body never names at a type \
                 (RFC-0082, RFC-0097 rule 3)"
            ),
        ));
    }
    Ok(Some(at))
}

/// What an `Instance` or `InstanceOf` parameter's type says.
enum Requirement {
    Of(RequiredParam),
    Owning {
        required: RequiredParam,
        held: Type,
        mode: Mode,
    },
}

const REQUIREMENT_SHAPE: &str = "a requirement names the signature it requires, the variable it stands at, \
     and the runtime: `it: Instance<sig::next<I, i64, E, Rt>, I, Rt>` owns the receiver its \
     call steps, `eq: InstanceOf<core::eq<T, Rt>, T, Rt>` stands at a type (RFC-0067 rule 1)";

fn requirement_param(ty: &Type) -> syn::Result<Option<Requirement>> {
    let Type::Path(p) = ty else {
        return Ok(None);
    };
    let Some(seg) = p.path.segments.last() else {
        return Ok(None);
    };
    let owns = match seg.ident.to_string().as_str() {
        "Instance" => true,
        "InstanceOf" => false,
        _ => return Ok(None),
    };
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return Err(syn::Error::new_spanned(seg, REQUIREMENT_SHAPE));
    };
    let tys: Vec<&Type> = args
        .args
        .iter()
        .filter_map(|a| match a {
            syn::GenericArgument::Type(t) => Some(t),
            _ => None,
        })
        .collect();
    let (Some(signature), Some(at)) = (tys.first(), tys.get(1)) else {
        return Err(syn::Error::new_spanned(seg, REQUIREMENT_SHAPE));
    };
    let task = match tys.get(3) {
        Some(t) => (*t).clone(),
        None => syn::parse_quote! { ::acvus_extern::Now },
    };
    let variable = |ty: &Type| -> Option<Ident> {
        let Type::Path(v) = ty else {
            return None;
        };
        v.path.get_ident().cloned()
    };
    if !owns {
        let Some(var) = variable(at) else {
            return Err(syn::Error::new_spanned(
                at,
                "an instance stands at one of this declaration's own type variables",
            ));
        };
        return Ok(Some(Requirement::Of(RequiredParam {
            signature: (*signature).clone(),
            var,
            task,
        })));
    }
    let (var, held, mode) = match at {
        Type::Reference(r) if r.mutability.is_some() => {
            (variable(&r.elem), (*r.elem).clone(), Mode::BorrowMut)
        }
        at => (variable(at), (*at).clone(), Mode::Value),
    };
    let Some(var) = var else {
        return Err(syn::Error::new_spanned(
            at,
            "an `Instance` owns its receiver at one of this declaration's own type variables: \
             the value `I`, or the `&mut I` its requirer was lent (RFC-0067 rule 1)",
        ));
    };
    Ok(Some(Requirement::Owning {
        required: RequiredParam {
            signature: (*signature).clone(),
            var,
            task,
        },
        held,
        mode,
    }))
}

/// What an `Args` parameter stands beside, refused (RFC-0097 rule 1).
fn refuse_args_beside(
    rust_params: &[RustParam],
    vars: &Vars,
    attr: &ExternFnAttr,
    is_coercion: bool,
) -> syn::Result<()> {
    let mut views = rust_params.iter().filter_map(|p| match p {
        RustParam::Args(args) => Some(args),
        RustParam::Acvus(_)
        | RustParam::State(_)
        | RustParam::Required(_)
        | RustParam::Owning { .. }
        | RustParam::Output(_) => None,
    });
    let Some(view) = views.next() else {
        return Ok(());
    };
    if let Some(second) = views.next() {
        return Err(syn::Error::new_spanned(
            &second.written,
            format!(
                "`{}` is a second `Args` beside `{}`: one view holds every position this \
                 declaration names only by a variable (RFC-0097 rule 1)",
                second.ident, view.ident
            ),
        ));
    }
    let runtime = &view.runtime;
    let names_the_runtime = matches!(runtime, Type::Path(p)
        if p.qself.is_none() && vars.runtime_ident().is_some_and(|rt| p.path.is_ident(rt)));
    if !names_the_runtime {
        return Err(syn::Error::new_spanned(runtime, ARGS_SHAPE));
    }
    for member in &view.members {
        let ty = &member.ty;
        let variable = matches!(ty, Type::Path(p)
            if p.qself.is_none()
                && p.path.get_ident().is_some_and(|id| matches!(vars.lookup(id), Some((VarKind::Ty, _)))));
        if !variable {
            return Err(syn::Error::new_spanned(
                ty,
                format!(
                    "`{}` is a concrete member of `Args`: a position whose type Rust names is an \
                     ordinary parameter, taken as `x: {}` (RFC-0097 rule 1)",
                    quote! { #ty },
                    quote! { #ty },
                ),
            ));
        }
        if vars.mentions_mono(ty) {
            return Err(syn::Error::new_spanned(
                ty,
                format!(
                    "`{}` is a `Monomorphize` variable, which a member instance crosses \
                     specialized, and `Args` holds each position uniform (RFC-0041, RFC-0097 rule 1)",
                    quote! { #ty },
                ),
            ));
        }
    }
    if let Some(signature) = &attr.instance_of {
        return Err(syn::Error::new_spanned(
            signature,
            "an instance of a shared signature is reached through its mono glue, which crosses \
             each position at the signature's own types and builds no `Args` (RFC-0067 rule 8, \
             RFC-0097 rule 1)",
        ));
    }
    if is_coercion {
        return Err(syn::Error::new_spanned(
            &view.written,
            "an extern_cast or extern_view converts a value of the one type its parameter \
             names, and an `Args` member names none (RFC-0023 rule 8, RFC-0097 rule 1)",
        ));
    }
    if let Some(named) = &attr.copies
        && *named == view.ident
    {
        return Err(syn::Error::new(
            named.span(),
            format!(
                "`copies({named})` names an `Args`, which lends no reference the result could \
                 copy: its members are taken by value (RFC-0082 rule 10, RFC-0097 rule 1)"
            ),
        ));
    }
    if let Some(named) = attr.reaches.as_ref().and_then(|reaches| reaches.naming(&view.ident)) {
        return Err(syn::Error::new(
            named.span(),
            format!(
                "`reaches` names `{named}`, an `Args`: a call reaches a place only through a \
                 reference parameter, and an `Args` takes its members by value (RFC-0082 rule 7, \
                 RFC-0097 rule 1)"
            ),
        ));
    }
    if let Some(law) = &attr.law {
        let word = law.first_word();
        return Err(syn::Error::new(
            word.span(),
            format!(
                "the law `{word}` is stated over a declaration's typed parameters, and `{}` is \
                 an `Args`, whose members the body never names at a type (RFC-0082, RFC-0097 rule 1)",
                view.ident
            ),
        ));
    }
    Ok(())
}

/// A Rust parameter that takes a run of the call's arguments, by the
/// acvus parameters it covers, as positions in the declaration's list.
enum Taker {
    One(usize),
    Args(std::ops::Range<usize>),
}

impl Taker {
    fn of(rust_params: &[RustParam]) -> Vec<Taker> {
        let mut at = 0;
        let mut takers = Vec::new();
        for p in rust_params {
            match p {
                RustParam::Acvus(_) | RustParam::Owning { .. } => {
                    takers.push(Taker::One(at));
                    at += 1;
                }
                RustParam::Args(args) => {
                    takers.push(Taker::Args(at..at + args.members.len()));
                    at += args.members.len();
                }
                RustParam::State(_) | RustParam::Required(_) | RustParam::Output(_) => {}
            }
        }
        takers
    }
}

/// The index in `FnDecl::requires` each requirement a declaration states
/// takes, read off its Rust parameters in order.
struct RequirementIndex {
    /// Per Rust parameter that takes a run of the call's arguments: the
    /// requirement whose `Instance` owns it.
    owning: Vec<Option<usize>>,
    /// Per `InstanceOf` parameter, in order.
    standing: Vec<usize>,
}

impl RequirementIndex {
    fn of(rust_params: &[RustParam]) -> Self {
        let mut index = RequirementIndex {
            owning: Vec::new(),
            standing: Vec::new(),
        };
        let mut nth = 0;
        for p in rust_params {
            match p {
                RustParam::Acvus(_) | RustParam::Args(_) => index.owning.push(None),
                RustParam::Owning { .. } => {
                    index.owning.push(Some(nth));
                    nth += 1;
                }
                RustParam::Required(_) => {
                    index.standing.push(nth);
                    nth += 1;
                }
                RustParam::State(_) | RustParam::Output(_) => {}
            }
        }
        index
    }
}

/// Two `Instance`s cannot own one receiver (RFC-0067 rule 1).
fn refuse_two_owners(rust_params: &[RustParam]) -> syn::Result<()> {
    let owners: Vec<&RustParam> = rust_params
        .iter()
        .filter(|p| matches!(p, RustParam::Owning { .. }))
        .collect();
    for (at, owner) in owners.iter().enumerate() {
        let RustParam::Owning { param, required } = owner else {
            continue;
        };
        let first = owners[..at].iter().find_map(|earlier| match earlier {
            RustParam::Owning {
                param: first,
                required: earlier,
            } if earlier.var == required.var => Some(first),
            _ => None,
        });
        if let Some(first) = first {
            return Err(syn::Error::new(
                required.var.span(),
                format!(
                    "`{param}` and `{first}` are two `Instance`s at `{var}`, and two `Instance`s \
                     cannot own one receiver: a declaration takes one receiver-owning requirement \
                     per variable (RFC-0067 rule 1)",
                    param = param.name,
                    first = first.name,
                    var = required.var,
                ),
            ));
        }
    }
    Ok(())
}

/// The Rust inputs whose flows a declaration states, with their roles.
struct FlowInputs {
    sig: syn::Signature,
    roles: Vec<flows::Role>,
}

/// Each `Instance` parameter read twice: as the receiver it owns, which is
/// the acvus parameter, and as the requirement at its variable, whose
/// signature ties that variable to the signature's other ones, as it did
/// when the requirement was a parameter of its own. Each `Args` parameter
/// read as its members, each an input taken by value. An `Output` is no
/// input: the result it fills holds only values its body wrote.
fn flow_inputs(sig: &syn::Signature, roles: &[flows::Role]) -> syn::Result<FlowInputs> {
    let mut inputs = FlowInputs {
        sig: sig.clone(),
        roles: Vec::new(),
    };
    inputs.sig.inputs.clear();
    for (arg, role) in sig.inputs.iter().zip(roles) {
        if let FnArg::Typed(pat_type) = arg
            && settled_written(&pat_type.ty, "Output", OUTPUT_SHAPE)?.is_some()
        {
            continue;
        }
        if let (FnArg::Typed(pat_type), flows::Role::Acvus(first)) = (arg, role)
            && let Some(ArgsWritten { members, .. }) = args_written(&pat_type.ty)?
        {
            for (at, member) in members.into_iter().enumerate() {
                inputs.sig.inputs.push(syn::parse_quote! { _: #member });
                inputs.roles.push(flows::Role::Acvus(first + at));
            }
            continue;
        }
        inputs.sig.inputs.push(arg.clone());
        inputs.roles.push(*role);
        let FnArg::Typed(pat_type) = arg else {
            continue;
        };
        let Some(owned) = owned_receiver(&pat_type.ty) else {
            continue;
        };
        let mut requirement = pat_type.clone();
        *requirement.ty = at_variable(&pat_type.ty, &owned);
        if let Some(FnArg::Typed(at)) = inputs.sig.inputs.last_mut() {
            *at.ty = owned;
        }
        inputs.sig.inputs.push(FnArg::Typed(requirement));
        inputs.roles.push(flows::Role::Other);
    }
    Ok(inputs)
}

/// `Instance<S, R, ..>` with `R` replaced by the variable it holds.
fn at_variable(instance: &Type, owned: &Type) -> Type {
    let held = match owned {
        Type::Reference(r) => (*r.elem).clone(),
        owned => owned.clone(),
    };
    let mut at = instance.clone();
    if let Type::Path(p) = &mut at
        && let Some(seg) = p.path.segments.last_mut()
        && let syn::PathArguments::AngleBracketed(args) = &mut seg.arguments
        && let Some(slot) = args
            .args
            .iter_mut()
            .filter_map(|a| match a {
                syn::GenericArgument::Type(t) => Some(t),
                _ => None,
            })
            .nth(1)
    {
        *slot = held;
    }
    at
}

/// The receiver an `Instance<S, R, ..>` type owns, `R` as written.
fn owned_receiver(ty: &Type) -> Option<Type> {
    let Type::Path(p) = ty else {
        return None;
    };
    let seg = p.path.segments.last()?;
    if seg.ident != "Instance" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    args.args
        .iter()
        .filter_map(|a| match a {
            syn::GenericArgument::Type(t) => Some(t.clone()),
            _ => None,
        })
        .nth(1)
}

fn generate_extern_fn(
    attr: ExternFnAttr,
    func: &mut ItemFn,
) -> syn::Result<proc_macro2::TokenStream> {
    let is_cast = take_marker_attr(&mut func.attrs, "extern_cast");
    let is_view = take_marker_attr(&mut func.attrs, "extern_view");
    let is_async = func.sig.asyncness.is_some();
    let vars = Vars::from_generics(&func.sig.generics)?;
    if let Some(chosen) = vars.chosen() {
        return Err(syn::Error::new(
            chosen.span(),
            "a declaration has no instance to choose its variables: Chosen is written on an \
             extern_signature! parameter",
        ));
    }
    let Signature {
        takes_ctx,
        params: rust_params,
    } = parse_params(&mut func.sig, vars.runtime_ident())?;
    let params: Vec<&ExternParam> = rust_params.iter().flat_map(RustParam::acvus).collect();
    refuse_args_beside(&rust_params, &vars, &attr, is_cast || is_view)?;
    let states: Vec<&StateParam> = rust_params
        .iter()
        .filter_map(|p| match p {
            RustParam::State(st) => Some(st),
            RustParam::Acvus(_)
            | RustParam::Required(_)
            | RustParam::Owning { .. }
            | RustParam::Args(_)
            | RustParam::Output(_) => None,
        })
        .collect();
    // Every requirement in the order of the Rust parameters that state it,
    // which is the order `FnDecl::requires` and the site table index.
    let required: Vec<&RequiredParam> = rust_params
        .iter()
        .filter_map(|p| match p {
            RustParam::Required(r) | RustParam::Owning { required: r, .. } => Some(r),
            RustParam::Acvus(_) | RustParam::State(_) | RustParam::Args(_) | RustParam::Output(_) => None,
        })
        .collect();
    let at_requirement = RequirementIndex::of(&rust_params);
    refuse_two_owners(&rust_params)?;
    let ret = parse_return(&func.sig.output);
    refuse_static_argument(&ret)?;
    let finished = settled_written(&ret, "Finished", FINISHED_SHAPE)?;
    let dynamic_at = dynamic_result(
        &rust_params,
        &vars,
        &attr,
        &ret,
        finished.as_ref(),
        &func.sig.ident,
        is_async,
        is_cast || is_view,
    )?;
    let returning = Returning::of(&ret, finished.as_ref());
    let rust_fn_effect = match rust_fn_written(&ret)? {
        Some(written) => Some(rust_fn_effect(&attr, &vars, &written, &ret, &func.sig.ident, is_async)?),
        None => None,
    };
    let mut roles: Vec<flows::Role> = takes_ctx.then_some(flows::Role::Ctx).into_iter().collect();
    let mut acvus_index = 0;
    for p in &rust_params {
        match p {
            RustParam::Acvus(_) | RustParam::Owning { .. } | RustParam::Args(_) => {
                roles.push(flows::Role::Acvus(acvus_index));
                acvus_index += p.acvus().len();
            }
            RustParam::State(_) | RustParam::Required(_) | RustParam::Output(_) => {
                roles.push(flows::Role::Other)
            }
        }
    }
    let FlowInputs {
        sig: flow_sig,
        roles: flow_roles,
    } = flow_inputs(&func.sig, &roles)?;
    // A `Finished` is the script's `Option<T>`, holding what the body wrote
    // through its `Output`, and its flows are that type's.
    let flow_ret: Type = match &returning {
        Returning::Finished(settled) => syn::parse_quote! { ::core::option::Option<#settled> },
        Returning::Value | Returning::Str | Returning::Lent(_) => ret.clone(),
    };
    let derived_flows = flows::derive(&flow_sig, &flow_roles, &vars, &flow_ret)?;
    if returning.lends() {
        if !params.iter().any(|p| p.mode.lends_its_storage()) {
            return Err(syn::Error::new_spanned(
                &ret,
                "a declaration returning a borrow has no parameter it can be a borrow of: \
                 every parameter is taken by value, so what the result names belongs to \
                 no storage the caller kept (RFC-0047 rule 3). Take one parameter by \
                 reference, or return an owned value.",
            ));
        }
        if is_async || attr.heavy {
            return Err(syn::Error::new(
                func.sig.ident.span(),
                "a declaration returning a borrow runs at `Task::Sync`: the borrow names \
                 the frame the call laid its arguments on, and that frame is gone by the \
                 time an awaited or offloaded call resumes (RFC-0023 rule 6). Return an owned \
                 value, or declare this at `Sync`.",
            ));
        }
    }

    let fn_ident = &func.sig.ident;
    let vis = &func.vis;
    let acvus_name = attr
        .name
        .as_ref()
        .map(LitStr::value)
        .unwrap_or_else(|| fn_ident.to_string());
    let qref = qref_expr(&acvus_name);
    let decl_ident = format_ident!("__extern_fn_{}", fn_ident);
    let instance_of = match &attr.instance_of {
        Some(sig) => quote! {
            ::core::option::Option::Some(<#sig as ::acvus_extern::SharedSignature>::qref(__i))
        },
        None => quote! { ::core::option::Option::None },
    };

    if attr.heavy && is_async {
        return Err(syn::Error::new(
            fn_ident.span(),
            "`heavy` and `async fn` are two tasks; declare one",
        ));
    }
    if attr.sync.is_some() && !is_async {
        return Err(syn::Error::new(
            fn_ident.span(),
            "`sync` names the plain `fn` that runs this declaration at `Task::Sync`; \
             a declaration that is not an `async fn` already is that instance",
        ));
    }
    let commutes = if attr.commutative {
        quote! { .commutative() }
    } else {
        quote! {}
    };
    let task = match (attr.heavy, is_async) {
        (true, _) => quote! { ::acvus_extern::Task::Heavy },
        (false, true) => quote! { ::acvus_extern::Task::Async },
        (false, false) => quote! { ::acvus_extern::Task::Sync },
    };
    let at_task = quote! { .at_task(#task) };
    // `optimize::spawn_split` rewrites every call whose effect is not Pure
    // into a `Spawn` and an `Eval`, and an `Eval` awaits. Such a call runs
    // at `Async` however synchronous its Rust body is, so a declaration
    // that is not pure says so (RFC-0046).
    let spawned = quote! { .at_task(#task.join(::acvus_extern::Task::Async)) };
    let effect = match &attr.effect {
        None => {
            quote! { ::acvus_extern::EffectTerm::Known(::acvus_extern::Effect::OPAQUE #commutes #spawned) }
        }
        Some(e) if e == "pure" => {
            if attr.commutative {
                return Err(syn::Error::new(
                    e.span(),
                    "a pure function commutes by definition; drop `commutative`",
                ));
            }
            quote! { ::acvus_extern::EffectTerm::Known(::acvus_extern::Effect::PURE #at_task) }
        }
        Some(e) if e == "idempotent" => {
            quote! { ::acvus_extern::EffectTerm::Known(::acvus_extern::Effect::IDEMPOTENT #commutes #spawned) }
        }
        Some(e) if e == "opaque" => {
            quote! { ::acvus_extern::EffectTerm::Known(::acvus_extern::Effect::OPAQUE #commutes #spawned) }
        }
        Some(e) => match vars.lookup(e) {
            Some((VarKind::Effect, k)) => {
                if attr.commutative {
                    return Err(syn::Error::new(
                        e.span(),
                        "`commutative` cannot be declared on an effect variable",
                    ));
                }
                if attr.heavy {
                    return Err(syn::Error::new(
                        e.span(),
                        "`heavy` cannot be declared on an effect variable: the task is the variable's",
                    ));
                }
                // A `Suspends` variable's task is never `Sync`, so no site
                // takes a twin (RFC-0046 rule 5, RFC-0011 rule 5).
                match (vars.suspends(e), &attr.sync) {
                    (false, None) if is_async => {
                        return Err(syn::Error::new(
                            e.span(),
                            "an `async fn` generic in its effect takes its task from the variable, \
                             so it needs the plain `fn` that runs it at `Task::Sync`: declare \
                             `sync = <fn>`, or bound the variable by `Suspends`. Without one the \
                             glue awaits for every effect the variable takes, and the declared \
                             task is a claim nothing keeps (RFC-0046).",
                        ));
                    }
                    (true, Some(sync)) => {
                        return Err(syn::Error::new(
                            sync.span(),
                            "a `Suspends` effect variable's task is never `Sync`, so the \
                             `sync =` twin could never run: drop it (RFC-0046 rule 5)",
                        ));
                    }
                    (false, _) | (true, None) => {}
                }
                quote! { __vars.effects[#k].clone() }
            }
            _ => {
                return Err(syn::Error::new(
                    e.span(),
                    "effect must be `pure`, `idempotent`, `opaque`, or a `Var<kind::Effect>` parameter",
                ));
            }
        },
    };

    let coercion = match (is_cast, is_view) {
        (true, true) => {
            return Err(syn::Error::new(
                fn_ident.span(),
                "a declaration is either an extern_cast or an extern_view, not both",
            ));
        }
        (true, false) => quote! { ::core::option::Option::Some(::acvus_extern::Coercion::Cast) },
        (false, true) => quote! { ::core::option::Option::Some(::acvus_extern::Coercion::View) },
        (false, false) => quote! { ::core::option::Option::None },
    };
    if is_cast || is_view {
        let kind = if is_cast {
            "extern_cast"
        } else {
            "extern_view"
        };
        if params.len() != 1 {
            return Err(syn::Error::new(
                fn_ident.span(),
                format!("an {kind} takes exactly one parameter"),
            ));
        }
        if !matches!(&attr.effect, Some(e) if e == "pure") {
            return Err(syn::Error::new(
                fn_ident.span(),
                format!("an {kind} declares `effect = pure`"),
            ));
        }
    }

    let signature = |member: Option<&Type>| -> proc_macro2::TokenStream {
        let param_terms = params.iter().map(|p| {
            let name = &p.name;
            let comp_ty = p.mode.acvus_ty(
                &vars.to_compile_time_instance(&p.ty, member),
                &quote! { ::acvus_extern::TypesOnly },
            );
            quote! {
                ::acvus_extern::ParamTerm::<::acvus_extern::Poly>::new(
                    __i.intern(#name),
                    <#comp_ty as ::acvus_extern::TyArg>::poly_ty(__i, &__vars),
                )
            }
        });
        let comp_ret = vars.to_compile_time_instance(&returning.filled(&ret), member);
        let comp_ret =
            returning.acvus_ty(&quote! { #comp_ret }, &quote! { ::acvus_extern::TypesOnly });
        let ret_ty = match &rust_fn_effect {
            Some(fn_effect) => quote! { <#comp_ret>::declared_fn_ty(__i, &__vars, #fn_effect) },
            None => quote! { <#comp_ret as ::acvus_extern::TyArg>::poly_ty(__i, &__vars) },
        };
        let declared_flows =
            derived_flows.tokens(|named| vars.to_compile_time_instance(named, member));
        quote! {
            ::acvus_extern::PolyTy::Fn {
                params: vec![#(#param_terms),*],
                ret: Box::new(#ret_ty),
                captures: vec![],
                effect: #effect,
                flows: #declared_flows,
            }
        }
    };

    let takers = Taker::of(&rust_params);
    let arg_idents: Vec<Ident> = (0..takers.len()).map(|i| format_ident!("__a{i}")).collect();
    let inst_idents: Vec<Ident> = (0..at_requirement.standing.len())
        .map(|i| format_ident!("__q{i}"))
        .collect();
    // The `Output` takes no argument, so it stands after every parameter that
    // does, beside the requirements.
    let takes_output = rust_params.iter().any(|p| matches!(p, RustParam::Output(_)));
    let out_idents: Vec<Ident> = takes_output.then(|| format_ident!("__o")).into_iter().collect();
    let out_markers: Vec<proc_macro2::TokenStream> = takes_output
        .then(|| quote! { ::acvus_extern::ByOutput })
        .into_iter()
        .collect();
    let crossing = |ty: &Type, member: Option<&Type>| -> proc_macro2::TokenStream {
        if member.is_some() && vars.mentions_mono(ty) {
            quote! { ::acvus_extern::Specialized }
        } else {
            quote! { ::acvus_extern::Uniform }
        }
    };
    let sync_variant = if attr.heavy {
        quote! { heavy }
    } else {
        quote! { sync }
    };
    let state_tys: Vec<&Type> = states.iter().map(|st| &st.ty).collect();
    // A mono glue is a plain `fn` of the call's own arguments and its
    // entry: it has no state to capture, no offloaded body, and no pair of
    // words at a parameter.
    let has_glue = attr.instance_of.is_some()
        && states.is_empty()
        && !attr.heavy
        && !params.is_empty()
        && !params.iter().any(|p| p.mode.is_two_words());
    let sig_mod = attr.instance_of.as_ref().map(signature_module_path);
    let entries = std::cell::RefCell::new(Vec::<proc_macro2::TokenStream>::new());
    let glue = |member: Option<&Type>, callee: &Ident, awaits: bool| -> proc_macro2::TokenStream {
        let rt_tys: Vec<Type> = params
            .iter()
            .map(|p| vars.to_runtime_instance(&p.ty, member))
            .collect();
        let rt_ret = vars.to_runtime_instance(&returning.filled(&ret), member);
        let ret_marker = {
            let c = crossing(&ret, member);
            returning.marker(
                &quote! { ::acvus_extern::Val<#rt_ret, #c> },
                &quote! { #rt_ret },
                &quote! { __R },
            )
        };
        let base_markers: Vec<proc_macro2::TokenStream> = takers
            .iter()
            .map(|taker| {
                let at = match taker {
                    Taker::One(at) => *at,
                    Taker::Args(members) => {
                        let tys = &rt_tys[members.clone()];
                        return quote! { ::acvus_extern::ByArgs<(#(#tys,)*)> };
                    }
                };
                let p = params[at];
                let ty = &rt_tys[at];
                let c = crossing(&p.ty, member);
                match p.mode {
                    Mode::Value => quote! { ::acvus_extern::ByValue<#ty, #c> },
                    Mode::Borrow => {
                        quote! { ::acvus_extern::ByRef<#ty, ::acvus_extern::Shared, #c> }
                    }
                    Mode::BorrowMut => {
                        quote! { ::acvus_extern::ByRef<#ty, ::acvus_extern::Mut, #c> }
                    }
                    Mode::Str => quote! { ::acvus_extern::ByStr },
                    Mode::Slice => quote! { ::acvus_extern::BySlice<#ty, ::acvus_extern::Shared> },
                    Mode::SliceMut => quote! { ::acvus_extern::BySlice<#ty, ::acvus_extern::Mut> },
                    Mode::Projection => {
                        let at_static = at_static(ty);
                        quote! { ::acvus_extern::ByProjection<#at_static> }
                    }
                }
            })
            .collect();
        // An `Instance` parameter takes its argument as the acvus parameter
        // it owns does, and its word from the site table.
        let arg_markers: Vec<proc_macro2::TokenStream> = base_markers
            .iter()
            .zip(&at_requirement.owning)
            .map(|(marker, owner)| match owner {
                None => marker.clone(),
                Some(nth) => {
                    let r = required[*nth];
                    let sig = vars.to_runtime_instance(&r.signature, member);
                    let task = &r.task;
                    quote! { ::acvus_extern::Owning<#marker, #sig, #task, #nth> }
                }
            })
            .collect();
        let inst_markers: Vec<proc_macro2::TokenStream> = at_requirement
            .standing
            .iter()
            .map(|&nth| {
                let r = required[nth];
                let sig = vars.to_runtime_instance(&r.signature, member);
                let var = &r.var;
                let var: Type = syn::parse_quote! { #var };
                let var = vars.to_runtime_instance(&var, member);
                let task = &r.task;
                quote! { ::acvus_extern::Required<#sig, #var, #task, #nth> }
            })
            .collect();
        let entry_param = match required.is_empty() {
            true => quote! { _ },
            false => quote! { __entry },
        };
        let standing_from_entry = at_requirement
            .standing
            .iter()
            .zip(&inst_idents)
            .map(|(&nth, ident)| {
                let r = required[nth];
                let sig = vars.to_runtime_instance(&r.signature, member);
                let var = &r.var;
                let var: Type = syn::parse_quote! { #var };
                let var = vars.to_runtime_instance(&var, member);
                let task = &r.task;
                let nth = proc_macro2::Literal::usize_unsuffixed(nth);
                quote! {
                    // SAFETY: the entry's `requires` holds, at this
                    // declaration's own order, the word of the entry
                    // `prepare` chose for this requirement, and that entry
                    // lives as long as the one this glue runs.
                    let #ident: ::acvus_extern::InstanceOf<'__w, #sig, #var, __R, #task> =
                        unsafe { __rt.instance(__entry.requires[#nth]) };
                }
            });
        let owning_from_entry = at_requirement
            .owning
            .iter()
            .zip(&arg_idents)
            .filter_map(|(owner, ident)| owner.map(|nth| (nth, ident)))
            .map(|(nth, ident)| {
                let r = required[nth];
                let sig = vars.to_runtime_instance(&r.signature, member);
                let task = &r.task;
                let nth = proc_macro2::Literal::usize_unsuffixed(nth);
                quote! {
                    // SAFETY: as an `InstanceOf`'s, and the word was chosen
                    // for the type of the argument this glue has just read.
                    let #ident = unsafe {
                        __rt.instance_owning::<#sig, _, #task>(#ident, __entry.requires[#nth])
                    };
                }
            });
        let inst_from_entry: Vec<proc_macro2::TokenStream> =
            standing_from_entry.chain(owning_from_entry).collect();
        // A receiver is read through the loan device, which is where a
        // type with no storage of its own is refused; nothing else reads
        // a value's payload in place.
        let recv_binding = params.first().map(|p| {
            let c = crossing(&p.ty, member);
            let at = &arg_idents[0];
            let loan = match p.mode {
                Mode::BorrowMut => quote! { ::acvus_extern::Mut },
                _ => quote! { ::acvus_extern::Shared },
            };
            match p.mode {
                // The receiver is read through a `Lending`, which ends its
                // loan when the glue's scope ends, after the body.
                Mode::BorrowMut | Mode::Borrow => quote! {
                    let __recv_at = unsafe {
                        ::acvus_extern::Lending::<#loan, __R, _>::of(
                            __rt.rt(),
                            <__R as ::acvus_extern::Runtime>::reference(__rt.rt(), &*__ctx.receiver()),
                        )
                    };
                    let #at = unsafe {
                        ::acvus_extern::receiver_borrowed::<_, #loan, #c, __R>(__rt.rt(), &__recv_at)
                    };
                },
                _ => quote! {
                    let __scope = ();
                    let #at = unsafe {
                        ::acvus_extern::receiver_by_value::<_, #c, __R>(
                            __rt, *__ctx.receiver(), &__scope,
                        )
                    };
                },
            }
        });
        // The run after the receiver is the signature's, not this
        // instance's: the types below only say which instance's glue this
        // is, and the signature's own module decides what crosses.
        let rest_markers: Vec<&proc_macro2::TokenStream> = base_markers.iter().skip(1).collect();
        // Nothing crosses after the receiver where the signature carries
        // nothing, and the glue of such an instance is the code it was
        // before a position at a variable had a crossing of its own.
        let rest_idents: &[Ident] = arg_idents.get(1..).unwrap_or(&[]);
        let rest_param = match rest_idents.is_empty() {
            true => quote! { _ },
            false => quote! { __rest },
        };
        let restoring = (has_glue && !rest_idents.is_empty()).then(|| {
            let sig_mod = sig_mod.as_ref().expect("has_glue names a signature");
            quote! {
                let mut __lent = ::core::default::Default::default();
                // SAFETY: as the receiver's: the run is this signature's
                // own, at the types this instance has.
                let (#(#rest_idents,)*) = unsafe {
                    #sig_mod::restore(
                        __rt,
                        __rest,
                        &mut __lent,
                        ::core::marker::PhantomData::<(#(#rest_markers,)*)>,
                    )
                };
            }
        });
        let turbofish = vars.runtime_turbofish_instance(member);
        let mut acvus_at = 0usize;
        let mut state_at = 0usize;
        let mut inst_at = 0usize;
        let passed: Vec<proc_macro2::TokenStream> = rust_params
            .iter()
            .map(|p| match p {
                RustParam::Acvus(_) | RustParam::Owning { .. } | RustParam::Args(_) => {
                    let at = &arg_idents[acvus_at];
                    acvus_at += 1;
                    quote! { #at }
                }
                RustParam::State(_) => {
                    let at = proc_macro2::Literal::usize_unsuffixed(state_at);
                    state_at += 1;
                    quote! { &__state.#at }
                }
                RustParam::Required(_) => {
                    let at = &inst_idents[inst_at];
                    inst_at += 1;
                    quote! { #at }
                }
                RustParam::Output(_) => quote! { __o },
            })
            .collect();
        let capture_state = (!states.is_empty()).then(|| {
            quote! { let __state = ::std::sync::Arc::clone(&__state); }
        });
        let ctx_arg = takes_ctx.then(|| quote! { __ctx, });
        let ctx_param = quote! {
            __ctx: &mut ::acvus_extern::Ctx<'_, __R>
        };
        let call = quote! { #callee #turbofish (#ctx_arg #(#passed),*) };
        let taken = quote! {
            #(let #arg_idents = #arg_idents.take();)*
            #(let #inst_idents = #inst_idents.take();)*
            #(let #out_idents = #out_idents.take();)*
        };
        // An instance whose result is a borrow crosses it through the
        // marker that borrow stands at; an owned result through the
        // signature's own `Returned` (RFC-0068 rule 6).
        let (lent_cross, lent_cross_await) = match &returning {
            Returning::Lent(shape) if !shape.slice => {
                let carrier = shape.one_carrier(&quote! { #rt_ret }, &quote! { __R });
                let cross = quote! { <#carrier as ::acvus_extern::Passed<'_, __R>>::cross };
                match shape.option {
                    true => (
                        quote! { (#call).map(|__x| #cross(__rt, __x)) },
                        quote! { (#call).await.map(|__x| #cross(__rt, __x)) },
                    ),
                    false => (
                        quote! { #cross(__rt, #call) },
                        quote! { #cross(__rt, (#call).await) },
                    ),
                }
            }
            _ => (
                quote! { <_ as #sig_mod::Returned<__R>>::cross(__rt, #call) },
                quote! { <_ as #sig_mod::Returned<__R>>::cross(__rt, (#call).await) },
            ),
        };
        if awaits && has_glue {
            let at = entries.borrow().len();
            let glue_ident = format_ident!("__instance_{}_{}", fn_ident, at);
            let entry_ty = format_ident!("__ExternInstance{}{}", fn_ident, at);
            let recv_binding = recv_binding.clone().expect("has_glue has a receiver");
            let sig_mod = sig_mod.as_ref().expect("has_glue names a signature");
            entries.borrow_mut().push(quote! {
                #[doc(hidden)]
                unsafe fn #glue_ident<'__a, '__w, __R>(
                    #entry_param: &'__a ::acvus_extern::InstanceEntry<__R>,
                    __ctx: &'__a mut ::acvus_extern::Ctx<'__w, __R>,
                    #rest_param: #sig_mod::Rest<'__a, __R>,
                ) -> ::acvus_extern::BoxFuture<'__a, #sig_mod::Ret<__R>>
                where
                    __R: ::acvus_extern::Runtime,
                {
                    // SAFETY: this is the glue `#[extern_fn]` wrote, and it crosses
                    // each value at the declaration's own types.
                    let __rt = unsafe { ::acvus_extern::Crossing::new(__ctx.rt) };
                    ::std::boxed::Box::pin(async move {
                        // SAFETY: an `Instance::call` named the receiver
                        // for this call, and this glue is the body of an
                        // instance standing at the type the value it named
                        // holds. The bindings are inside the future
                        // because what a position taken by reference
                        // borrows has to live as long as the body it is
                        // lent to.
                        #recv_binding
                        #restoring
                        #(#inst_from_entry)*
                        #lent_cross_await
                    })
                }

                #[doc(hidden)]
                #[allow(non_camel_case_types)]
                pub struct #entry_ty;

                impl<__R> ::acvus_extern::AtInstance<__R> for #entry_ty
                where
                    __R: ::acvus_extern::Runtime,
                {
                    fn run() -> ::core::option::Option<::acvus_extern::InstanceRun> {
                        let __at: #sig_mod::Later<__R> = #glue_ident::<__R>;
                        // SAFETY: `__at` is typed at the signature's `Later`.
                        ::core::option::Option::Some(unsafe {
                            ::acvus_extern::InstanceRun::from_glue(
                                ::acvus_extern::repr::fn_addr(__at),
                                ::acvus_extern::Task::Async,
                            )
                        })
                    }
                }
            });
            quote! {
                ::acvus_extern::ExternHandler::awaited(
                    ::acvus_extern::async_glue_at_instance::<
                        __R,
                        _,
                        (#(#arg_markers,)* #(#inst_markers,)* #(#out_markers,)*),
                        #entry_ty,
                    >(move |#ctx_param, (#(#arg_idents,)* #(#inst_idents,)* #(#out_idents,)*)| {
                        // SAFETY: this is the glue `#[extern_fn]` wrote, and it crosses
                    // each value at the declaration's own types.
                    let __rt = unsafe { ::acvus_extern::Crossing::new(__ctx.rt) };
                        ::std::boxed::Box::pin(async move {
                            #taken
                            let __r = (#call).await;
                            <_ as ::acvus_extern::OneValue<__R>>::erase(__r, __rt)
                        })
                    })
                )
            }
        } else if awaits {
            quote! {
                ::acvus_extern::ExternHandler::awaited({
                    #capture_state
                    ::acvus_extern::async_glue::<__R, _, (#(#arg_markers,)* #(#inst_markers,)* #(#out_markers,)*)>(
                        move |#ctx_param, (#(#arg_idents,)* #(#inst_idents,)* #(#out_idents,)*)| {
                            #capture_state
                            // SAFETY: this is the glue `#[extern_fn]` wrote, and it crosses
                    // each value at the declaration's own types.
                    let __rt = unsafe { ::acvus_extern::Crossing::new(__ctx.rt) };
                            ::std::boxed::Box::pin(async move {
                                #taken
                                let __r = (#call).await;
                                <_ as ::acvus_extern::OneValue<__R>>::erase(__r, __rt)
                            })
                        }
                    )
                })
            }
        } else if has_glue {
            let at = entries.borrow().len();
            let glue_ident = format_ident!("__instance_{}_{}", fn_ident, at);
            let entry_ty = format_ident!("__ExternInstance{}{}", fn_ident, at);
            let recv_binding = recv_binding.clone().expect("has_glue has a receiver");
            let sig_mod = sig_mod.as_ref().expect("has_glue names a signature");
            entries.borrow_mut().push(quote! {
                #[doc(hidden)]
                unsafe fn #glue_ident<'__w, __R>(
                    #entry_param: &::acvus_extern::InstanceEntry<__R>,
                    __ctx: &mut ::acvus_extern::Ctx<'__w, __R>,
                    #rest_param: #sig_mod::Rest<'_, __R>,
                ) -> #sig_mod::Ret<__R>
                where
                    __R: ::acvus_extern::Runtime,
                {
                    // SAFETY: this is the glue `#[extern_fn]` wrote, and it crosses
                    // each value at the declaration's own types.
                    let __rt = unsafe { ::acvus_extern::Crossing::new(__ctx.rt) };
                    // SAFETY: an `Instance::call` named the receiver for
                    // this call, and this glue is the body of an instance
                    // standing at the type the value it named holds.
                    #recv_binding
                    #restoring
                    #(#inst_from_entry)*
                    #lent_cross
                }

                #[doc(hidden)]
                #[allow(non_camel_case_types)]
                pub struct #entry_ty;

                impl<__R> ::acvus_extern::AtInstance<__R> for #entry_ty
                where
                    __R: ::acvus_extern::Runtime,
                {
                    fn run() -> ::core::option::Option<::acvus_extern::InstanceRun> {
                        let __at: #sig_mod::Now<__R> = #glue_ident::<__R>;
                        // SAFETY: `__at` is typed at the signature's `Now`.
                        ::core::option::Option::Some(unsafe {
                            ::acvus_extern::InstanceRun::from_glue(
                                ::acvus_extern::repr::fn_addr(__at),
                                ::acvus_extern::Task::Sync,
                            )
                        })
                    }
                }
            });
            quote! {
                ::acvus_extern::ExternHandler::#sync_variant(
                    ::acvus_extern::glue_at_instance::<
                        __R,
                        _,
                        (#(#arg_markers,)* #(#inst_markers,)* #(#out_markers,)*),
                        #ret_marker,
                        #entry_ty,
                    >(move |#ctx_param, (#(#arg_idents,)* #(#inst_idents,)* #(#out_idents,)*), __ret| {
                        #taken
                        __ret.put(#call)
                    })
                )
            }
        } else {
            quote! {
                ::acvus_extern::ExternHandler::#sync_variant({
                    #capture_state
                    ::acvus_extern::glue::<__R, _, (#(#arg_markers,)* #(#inst_markers,)* #(#out_markers,)*), #ret_marker>(
                        move |#ctx_param, (#(#arg_idents,)* #(#inst_idents,)* #(#out_idents,)*), __ret| {
                            #taken
                            __ret.put(#call)
                        }
                    )
                })
            }
        }
    };

    let family_casts = |member: &Type| -> Vec<proc_macro2::TokenStream> {
        let mut seen = std::collections::HashSet::new();
        params
            .iter()
            .map(|p| &p.ty)
            .chain(std::iter::once(&ret))
            .filter(|ty| vars.mentions_mono(ty))
            .filter(|ty| seen.insert(quote! { #ty }.to_string()))
            .map(|ty| {
                let specialized = vars.to_compile_time_instance(ty, Some(member));
                let uniform = vars.to_compile_time_uniform(ty, member);
                let rt_ty = vars.to_runtime_instance(ty, Some(member));
                let cast_glue = |from: proc_macro2::TokenStream,
                                 into: proc_macro2::TokenStream|
                 -> proc_macro2::TokenStream {
                    quote! {
                        ::acvus_extern::ExternHandler::sync(
                            ::acvus_extern::glue::<
                                __R,
                                _,
                                (::acvus_extern::ByValue<#rt_ty, #from>,),
                                ::acvus_extern::Val<#rt_ty, #into>,
                            >(|_, (__v,), __ret| __ret.put::<#rt_ty>(__v.take::<#rt_ty>()))
                        )
                    }
                };
                let erase = cast_glue(
                    quote! { ::acvus_extern::Specialized },
                    quote! { ::acvus_extern::Uniform },
                );
                let materialize = cast_glue(
                    quote! { ::acvus_extern::Uniform },
                    quote! { ::acvus_extern::Specialized },
                );
                quote! {
                    __casts.extend(::acvus_extern::family_casts::<__R>(
                        __i,
                        ::acvus_extern::MemberType {
                            specialized: <#specialized as ::acvus_extern::TyArg>::poly_ty(__i, &__vars),
                            uniform: <#uniform as ::acvus_extern::TyArg>::poly_ty(__i, &__vars),
                            erase: #erase,
                            materialize: #materialize,
                        },
                    ));
                }
            })
            .collect()
    };

    let instance_requires = match attr.instance_of {
        Some(_) => quote! { __instance_requires.clone() },
        None => quote! { ::std::vec::Vec::new() },
    };
    let instance_effect_bounds = match attr.instance_of {
        Some(_) => {
            let bounds = vars.effect_bound_exprs();
            quote! { vec![#(#bounds),*] }
        }
        None => quote! { ::std::vec::Vec::new() },
    };
    // The instances one member of the declaration contributes, in the
    // order `Instances::into_handlers` indexes them.
    let at_member = |member: Option<&Type>| -> Vec<proc_macro2::TokenStream> {
        let declared_sig = signature(member);
        let declared_handler = glue(member, fn_ident, is_async);
        let declared = quote! {
            ::acvus_extern::DeclaredInstance {
                signature: #declared_sig,
                handler: #declared_handler,
                admits: ::acvus_extern::Task::Heavy,
                requires: #instance_requires,
                effect_bounds: #instance_effect_bounds,
            }
        };
        let Some(sync_fn) = &attr.sync else {
            return vec![declared];
        };
        let sync_sig = signature(member);
        let sync_handler = glue(member, sync_fn, false);
        vec![
            quote! {
                ::acvus_extern::DeclaredInstance {
                    signature: #sync_sig,
                    handler: #sync_handler,
                    admits: ::acvus_extern::Task::Sync,
                    requires: #instance_requires,
                    effect_bounds: #instance_effect_bounds,
                }
            },
            declared,
        ]
    };

    let mut casts: Vec<proc_macro2::TokenStream> = Vec::new();
    let instances = match vars.mono_var() {
        None => match &attr.sync {
            None => {
                let generic = glue(None, fn_ident, is_async);
                quote! { ::acvus_extern::Instances::generic(#generic) }
            }
            Some(_) => {
                let concrete = at_member(None);
                quote! {
                    ::acvus_extern::Instances {
                        concrete: vec![#(#concrete),*],
                        generic: ::core::option::Option::None,
                    }
                }
            }
        },
        Some(mono) => {
            let members = mono.mono.as_ref().expect("mono_var has members");
            let concrete: Vec<proc_macro2::TokenStream> = members
                .iter()
                .flat_map(|member| at_member(Some(member)))
                .collect();
            let generic = if mono.mono_fallback {
                let handler = glue(None, fn_ident, is_async);
                quote! { ::core::option::Option::Some(#handler) }
            } else {
                quote! { ::core::option::Option::None }
            };
            casts = members.iter().flat_map(|m| family_casts(m)).collect();
            quote! {
                ::acvus_extern::Instances {
                    concrete: vec![#(#concrete),*],
                    generic: #generic,
                }
            }
        }
    };
    let mut bounds = vars.bound_exprs();
    if let Some(at) = dynamic_at {
        bounds[at] = quote! { ::acvus_extern::TyVarBound::Settled };
    }
    let effect_bounds = vars.effect_bound_exprs();
    let requires: Vec<proc_macro2::TokenStream> = required
        .iter()
        .map(|r| {
            let path = signature_path(&r.signature)?;
            if !matches!(vars.lookup(&r.var), Some((VarKind::Ty, _))) {
                return Err(syn::Error::new(
                    r.var.span(),
                    "an instance stands at one of this declaration's own `Var<kind::Type>` \
                     parameters",
                ));
            }
            let task = &r.task;
            // The marker alone says the task: `Later` on a plain `fn` is a
            // declaration that holds the instance for a stage to call, and
            // the stage's own effect variable is what narrows the site that
            // runs its `sync =` twin (RFC-0046, RFC-0068 rule 5).
            let calls = quote! { <#task as ::acvus_extern::CalledAt>::TASK };
            let marker = vars.to_compile_time_instance(&r.signature, None);
            Ok(quote! {
                ::acvus_extern::Requirement {
                    signature: <#path as ::acvus_extern::SharedSignature>::qref(__i),
                    pattern: <#marker as ::acvus_extern::RequirementOf>::pattern(__i, &__vars),
                    calls: #calls,
                }
            })
        })
        .collect::<syn::Result<_>>()?;
    let declared_ty = signature(None);
    let laws = match (&attr.law, &attr.payload) {
        (Some(law), Some(named)) => {
            return Err(syn::Error::new(
                named.span(),
                format!(
                    "`payload` beside the law `{}`: `payload` is stated over \
                     `f(o: Option<T>) -> T`, which states no other law (RFC-0082 rule 3)",
                    law.first_word()
                ),
            ));
        }
        (Some(law), None) => law.checked_laws(fn_ident, &params, &ret, &returning)?,
        (None, Some(named)) => law::checked_payload(named, fn_ident, &params, &ret, &returning)?,
        (None, None) => quote! { ::acvus_extern::Laws::None },
    };
    let mut emitted = func.clone();
    let ensures = match &attr.ensures {
        Some(stated) => {
            let stated = stated.stated(fn_ident, &params)?;
            *emitted.block = ensures::wrap_body(
                &func.block,
                &ret,
                is_async,
                &stated.before,
                &stated.evaluated,
            );
            stated.declared
        }
        None => quote! { ::std::vec::Vec::new() },
    };
    let reaches = match &attr.reaches {
        Some(stated) => stated.declared(fn_ident, &params)?,
        None => quote! { ::acvus_extern::Reaches::Lent },
    };
    let copies = match &attr.copies {
        Some(named) => {
            let Some(at) = params.iter().position(|param| *named == param.name) else {
                return Err(syn::Error::new(
                    named.span(),
                    format!("`{named}` names no parameter of `{fn_ident}` (RFC-0082 rule 10)"),
                ));
            };
            // The text a `&str` lends is a `String`'s value (RFC-0070 rule
            // 5), as `acvus_mir::laws::copies_fits` reads it.
            let lent = quote::ToTokens::to_token_stream(&params[at].ty).to_string();
            let returned = quote::ToTokens::to_token_stream(&ret).to_string();
            let lends_the_result = matches!(returning, Returning::Value)
                && match params[at].mode {
                    Mode::Borrow => lent == returned,
                    Mode::Str => returned == "String",
                    _ => false,
                };
            if !lends_the_result {
                return Err(syn::Error::new(
                    named.span(),
                    format!(
                        "`copies({named})` is stated over `f(.., {named}: &T, ..) -> T` or \
                         `f(.., {named}: &str, ..) -> String`, and `{fn_ident}` is not of that \
                         shape (RFC-0082 rule 10)"
                    ),
                ));
            }
            quote! {
                ::core::option::Option::Some(::acvus_extern::Copies { param: #at })
            }
        }
        None => quote! { ::core::option::Option::None },
    };
    let returns = match attr.returns {
        Some(StatedReturn::ReturnsOrTraps) => quote! { ::acvus_extern::Returns::Stated },
        Some(StatedReturn::Total) => quote! { ::acvus_extern::Returns::Total },
        None => quote! { ::acvus_extern::Returns::Unstated },
    };
    let cost = match &attr.cost {
        Some(weight) => {
            let weight: u64 = weight.base10_parse()?;
            quote! { ::core::option::Option::Some(#weight) }
        }
        None => quote! { ::core::option::Option::None },
    };

    let fresh_vars = vars.fresh_vars_expr();
    let rt_bounds = quote! { __R: ::acvus_extern::Runtime, };
    let state_idents: Vec<&Ident> = states.iter().map(|st| &st.ident).collect();
    let state_arc = (!states.is_empty()).then(|| {
        quote! {
            let __state: ::std::sync::Arc<(#(#state_tys,)*)> =
                ::std::sync::Arc::new((#(#state_idents,)*));
        }
    });
    let entries = entries.into_inner();
    let lifetimes_written = flows::lifetimes_written(&func.sig);
    Ok(quote! {
        #emitted

        #lifetimes_written

        #(#entries)*

        #[doc(hidden)]
        #vis fn #decl_ident<__R>(
            __i: &::acvus_extern::Interner,
            __ns: ::core::option::Option<&str>,
            #(#state_idents: #state_tys,)*
        ) -> ::std::vec::Vec<::acvus_extern::ExternFn<__R>>
        where
            #rt_bounds
            #(#state_tys: ::core::marker::Send + ::core::marker::Sync + 'static,)*
        {
            let __vars = #fresh_vars;
            let __requires: ::std::vec::Vec<::acvus_extern::Requirement> =
                vec![#(#requires),*];
            let __instance_requires: ::std::vec::Vec<::acvus_extern::RequirementSig> = __requires
                .iter()
                .map(::acvus_extern::Requirement::signature_of)
                .collect();
            #state_arc
            let mut __casts: ::std::vec::Vec<::acvus_extern::ExternFn<__R>> = ::std::vec::Vec::new();
            #(#casts)*
            let __ty = #declared_ty;
            let __bounds = vec![#(#bounds),*];
            let __instances = #instances;
            let __declared = ::acvus_extern::ExternFn {
                decl: ::acvus_extern::FnDecl {
                    qref: #qref,
                    ty: __ty,
                    bounds: __bounds,
                    effect_bounds: vec![#(#effect_bounds),*],
                    coercion: #coercion,
                    instance_of: #instance_of,
                    requires: __requires,
                    names: __vars.names(),
                    laws: #laws,
                    ensures: #ensures,
                    reaches: #reaches,
                    returns: #returns,
                    copies: #copies,
                    cost: #cost,
                },
                instances: __instances,
            };
            ::core::iter::once(__declared).chain(__casts).collect()
        }
    })
}

fn take_marker_attr(attrs: &mut Vec<Attribute>, name: &str) -> bool {
    let before = attrs.len();
    attrs.retain(|a| !a.path().is_ident(name));
    attrs.len() != before
}

/// What a declaration's Rust signature carries in front of its acvus
/// parameters: `&mut Ctx<'_, R>`. A function that uses neither the runtime
/// nor the window takes it not at all.
struct Signature {
    takes_ctx: bool,
    params: Vec<RustParam>,
}

fn parse_params(sig: &mut syn::Signature, runtime: Option<&Ident>) -> syn::Result<Signature> {
    let mut inputs = sig.inputs.iter_mut().peekable();
    let takes_ctx = inputs.peek().is_some_and(|next| is_ctx_param(next));
    if takes_ctx {
        inputs.next();
    }

    let mut params = Vec::new();
    for (i, arg) in inputs.enumerate() {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "an extern_fn has no self parameter",
            ));
        };
        let is_state = take_marker_attr(&mut pat_type.attrs, "state");
        let ident = match pat_type.pat.as_ref() {
            Pat::Ident(p) => p.ident.clone(),
            _ => format_ident!("_{i}"),
        };
        if is_state {
            let Type::Reference(r) = pat_type.ty.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &pat_type.ty,
                    "a `#[state]` parameter is taken by shared reference",
                ));
            };
            params.push(RustParam::State(StateParam {
                ident,
                ty: (*r.elem).clone(),
            }));
            continue;
        }
        refuse_static_argument(pat_type.ty.as_ref())?;
        if let Some(SettledWritten { settled, runtime }) =
            settled_written(pat_type.ty.as_ref(), "Output", OUTPUT_SHAPE)?
        {
            params.push(RustParam::Output(OutputParam {
                written: (*pat_type.ty).clone(),
                settled,
                runtime,
            }));
            continue;
        }
        if settled_written(pat_type.ty.as_ref(), "Finished", FINISHED_SHAPE)?.is_some() {
            return Err(syn::Error::new_spanned(
                &pat_type.ty,
                "a `Finished` is a `dynamic` declaration's result and no parameter: it comes \
                 only from the `Output` of the call that returns it (RFC-0097 rule 3)",
            ));
        }
        if let Some(ArgsWritten { members, runtime }) = args_written(pat_type.ty.as_ref())? {
            let members = members
                .into_iter()
                .enumerate()
                .map(|(at, ty)| ExternParam {
                    name: format!("{ident}.{at}"),
                    ty,
                    mode: Mode::Value,
                })
                .collect();
            params.push(RustParam::Args(ArgsParam {
                ident,
                written: (*pat_type.ty).clone(),
                runtime,
                members,
            }));
            continue;
        }
        match requirement_param(pat_type.ty.as_ref())? {
            Some(Requirement::Of(required)) => {
                params.push(RustParam::Required(required));
                continue;
            }
            Some(Requirement::Owning {
                required,
                held,
                mode,
            }) => {
                params.push(RustParam::Owning {
                    param: ExternParam {
                        name: ident.to_string(),
                        ty: held,
                        mode,
                    },
                    required,
                });
                continue;
            }
            None => {}
        }
        if let Some(runtime) = runtime
            && crosses_as_ctx(pat_type.ty.as_ref(), runtime)
        {
            return Err(syn::Error::new_spanned(
                &pat_type.ty,
                "the runtime and the window above the calling frame cross as one parameter, \
                 `ctx: &mut Ctx<'_, Rt>`, written first (RFC-0023 rule 2). Write that in place \
                 of `rt: &Rt` and `frame: &mut Rt::Frame<'_>`, and read `ctx.rt` in the body.",
            ));
        }
        let (ty, mode) = match pat_type.ty.as_ref() {
            Type::Reference(r) if r.mutability.is_some() && is_str(&r.elem) => {
                return Err(syn::Error::new_spanned(
                    &pat_type.ty,
                    "there is no `&mut str`: a write through one could leave the bytes \
                     invalid UTF-8. Take `&str` to read, or `String` to own (RFC-0062).",
                ));
            }
            Type::Reference(r) if r.mutability.is_some() => match r.elem.as_ref() {
                Type::Slice(s) => ((*s.elem).clone(), Mode::SliceMut),
                elem => (elem.clone(), Mode::BorrowMut),
            },
            Type::Reference(r) if is_str(&r.elem) => ((*r.elem).clone(), Mode::Str),
            Type::Reference(r) => match r.elem.as_ref() {
                Type::Slice(s) => ((*s.elem).clone(), Mode::Slice),
                elem => (elem.clone(), Mode::Borrow),
            },
            ty if borrows_caller(ty, runtime) => (ty.clone(), Mode::Projection),
            ty => (ty.clone(), Mode::Value),
        };
        params.push(RustParam::Acvus(ExternParam {
            name: ident.to_string(),
            ty,
            mode,
        }));
    }
    Ok(Signature { takes_ctx, params })
}

/// Refuses a type in an extern's signature that takes `'static` as a
/// lifetime argument, as `Ref<'static, …>` does. Rust would refuse it too,
/// where the glue requires the handler's parameter `Within` the call, but
/// with an error at the glue; one behind an alias or a container is still
/// refused there (RFC-0079 rule 6).
fn refuse_static_argument(ty: &Type) -> syn::Result<()> {
    struct Find(Option<(Ident, syn::Lifetime)>);

    impl<'ast> syn::visit::Visit<'ast> for Find {
        fn visit_path_segment(&mut self, segment: &'ast syn::PathSegment) {
            if self.0.is_none()
                && let syn::PathArguments::AngleBracketed(args) = &segment.arguments
                && let Some(lifetime) = args.args.iter().find_map(|arg| match arg {
                    syn::GenericArgument::Lifetime(lifetime) if lifetime.ident == "static" => {
                        Some(lifetime)
                    }
                    _ => None,
                })
            {
                self.0 = Some((segment.ident.clone(), lifetime.clone()));
            }
            syn::visit::visit_path_segment(self, segment);
        }
    }

    let mut find = Find(None);
    syn::visit::Visit::visit_type(&mut find, ty);
    match find.0 {
        None => Ok(()),
        Some((carrier, lifetime)) => Err(syn::Error::new(
            lifetime.span(),
            format!(
                "`{carrier}<'static, …>` in an extern's signature: what a call hands the \
                 handler is at that call and is not kept past it, so the signature names the \
                 call's lifetime, `'_` or a lifetime parameter (RFC-0079 rule 6)"
            ),
        )),
    }
}

fn crosses_as_ctx(ty: &Type, runtime: &Ident) -> bool {
    let Type::Reference(r) = ty else {
        return false;
    };
    match r.mutability {
        None => matches!(r.elem.as_ref(), Type::Path(p) if p.path.is_ident(runtime)),
        Some(_) => matches!(r.elem.as_ref(), Type::Path(p)
            if p.path.segments.last().is_some_and(|last| last.ident == "Frame")),
    }
}

/// Read by its type and not by its position: a first acvus parameter taken
/// `&mut T` sits in the same place.
fn is_ctx_param(arg: &FnArg) -> bool {
    let FnArg::Typed(pat_type) = arg else {
        return false;
    };
    let Type::Reference(r) = pat_type.ty.as_ref() else {
        return false;
    };
    let Type::Path(p) = r.elem.as_ref() else {
        return false;
    };
    let named_ctx = p
        .path
        .segments
        .last()
        .is_some_and(|last| last.ident == "Ctx");
    r.mutability.is_some() && named_ctx
}

fn parse_return(output: &ReturnType) -> Type {
    match output {
        ReturnType::Default => syn::parse_quote! { () },
        ReturnType::Type(_, ty) => (**ty).clone(),
    }
}

/// The name under the namespace the registry passes as `__ns`.
fn qref_expr(name: &str) -> proc_macro2::TokenStream {
    quote! {
        ::acvus_extern::QualifiedRef {
            namespace: __ns.map(|__n| __i.intern(__n)),
            name: __i.intern(#name),
            host: None,
        }
    }
}

/// The name under a namespace fixed at the declaration.
fn qref_expr_in(ns: Option<&str>, name: &str) -> proc_macro2::TokenStream {
    match ns {
        Some(ns) => quote! {
            ::acvus_extern::QualifiedRef::qualified(__i.intern(#ns), __i.intern(#name))
        },
        None => quote! { ::acvus_extern::QualifiedRef::root(__i.intern(#name)) },
    }
}

// -- #[derive(ExternType)] -------------------------------------------

struct ExternTypeAttr {
    name: Option<String>,
    ns: Option<String>,
    /// `unsafe(uniform_payload)`: the author asserts of the payload what
    /// `UniformPayload` states, for a payload whose field types the marker
    /// does not reach (RFC-0076).
    uniform_payload: bool,
    /// `space`: the type is a context a space holds, through the author's
    /// `Journaled` impl at the form the runtime holds (RFC-0033).
    space: bool,
}

/// Whether the struct carries `#[repr(transparent)]`.
fn is_repr_transparent(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| {
        a.path().is_ident("repr")
            && a.parse_args::<Ident>()
                .is_ok_and(|ident| ident == "transparent")
    })
}

fn parse_extern_type_attr(attrs: &[Attribute]) -> syn::Result<ExternTypeAttr> {
    let mut out = ExternTypeAttr {
        name: None,
        ns: None,
        uniform_payload: false,
        space: false,
    };
    for attr in attrs {
        if !attr.path().is_ident("extern_type") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                meta.input.parse::<Token![=]>()?;
                out.name = Some(meta.input.parse::<LitStr>()?.value());
            } else if meta.path.is_ident("ns") {
                meta.input.parse::<Token![=]>()?;
                out.ns = Some(meta.input.parse::<LitStr>()?.value());
            } else if meta.path.is_ident("space") {
                out.space = true;
            } else if meta.path.is_ident("unsafe") {
                meta.parse_nested_meta(|inner| {
                    if !inner.path.is_ident("uniform_payload") {
                        return Err(inner.error("expected `uniform_payload`"));
                    }
                    out.uniform_payload = true;
                    Ok(())
                })?;
            } else {
                return Err(meta.error(
                    "expected `name`, `ns`, `space`, or `unsafe(uniform_payload)`",
                ));
            }
            Ok(())
        })?;
    }
    Ok(out)
}

#[proc_macro_derive(ExternType, attributes(extern_type))]
pub fn derive_extern_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match generate_extern_type(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn generate_extern_type(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let ident = &input.ident;
    let attr = parse_extern_type_attr(&input.attrs)?;
    let name = attr.name.clone().unwrap_or_else(|| ident.to_string());
    let vars = Vars::from_generics(&input.generics)?;
    if let Some(suspending) = vars.suspending() {
        return Err(syn::Error::new(
            suspending.span(),
            "a type does not bound its effect variables: Suspends is written on an \
             #[extern_fn] declaration",
        ));
    }
    if vars.has_len_vars() {
        return Err(syn::Error::new(
            ident.span(),
            "an extension type has no length parameters",
        ));
    }

    let syn::Data::Struct(data) = &input.data else {
        return Err(syn::Error::new(
            ident.span(),
            "ExternType is derived on a struct",
        ));
    };
    let syn::Fields::Unnamed(fields) = &data.fields else {
        return Err(syn::Error::new(
            ident.span(),
            "ExternType is derived on a tuple struct: payload first, then PhantomData",
        ));
    };
    let mut field_iter = fields.unnamed.iter();
    let Some(payload) = field_iter.next() else {
        return Err(syn::Error::new(
            ident.span(),
            "an extension type has a payload field",
        ));
    };
    let payload_ty = &payload.ty;
    let uniform = vars.uniform_type_vars();
    let non_type = vars.non_type_vars();
    if let Some((through, projection)) = projection_through(payload_ty, &uniform) {
        return Err(syn::Error::new_spanned(
            projection,
            format!(
                "the payload projects through the uniform type parameter `{through}`, so its \
                 layout is whatever a trait impl for `{through}` chooses and no one box serves \
                 every `{through}`: bound `{through}` by `Chosen`, which keys each instance's \
                 box at its own Rust type (RFC-0076)"
            ),
        ));
    }
    if !is_repr_transparent(&input.attrs) {
        return Err(syn::Error::new(
            ident.span(),
            "an extension type is `#[repr(transparent)]`: it is stored as its payload and read back through it",
        ));
    }
    for extra in field_iter {
        let is_phantom = match &extra.ty {
            Type::Path(p) => p
                .path
                .segments
                .last()
                .is_some_and(|s| s.ident == "PhantomData"),
            _ => false,
        };
        if !is_phantom {
            return Err(syn::Error::new_spanned(
                &extra.ty,
                "every field after the payload is PhantomData",
            ));
        }
    }

    let (_, ty_generics, _) = input.generics.split_for_impl();
    let arg_impl_generics = vars.arg_impl_generics();
    let impl_params = {
        let params = &input.generics.params;
        if params.is_empty() {
            quote! {}
        } else {
            quote! { #params, }
        }
    };
    // Each lifetime parameter is a region parameter of the type (RFC-0079
    // rule 6). `TyArg` and `ExternTypeDecl` are the marker's, which names the
    // type with every lifetime at `'static`; every other impl holds at every
    // lifetime, since a handler is handed the type at its call's.
    let lifetimes: Vec<syn::Lifetime> = input
        .generics
        .lifetimes()
        .map(|lt| lt.lifetime.clone())
        .collect();
    let n_regions = lifetimes.len();
    let static_params = {
        let params = input
            .generics
            .params
            .iter()
            .filter(|param| !matches!(param, GenericParam::Lifetime(_)));
        quote! { #(#params,)* }
    };
    let self_at = |lifetime: proc_macro2::TokenStream| -> proc_macro2::TokenStream {
        let args = input.generics.params.iter().map(|param| match param {
            GenericParam::Lifetime(_) => lifetime.clone(),
            GenericParam::Type(tp) => {
                let ident = &tp.ident;
                quote! { #ident }
            }
            GenericParam::Const(c) => {
                let ident = &c.ident;
                quote! { #ident }
            }
        });
        match input.generics.params.is_empty() {
            true => quote! { #ident },
            false => quote! { #ident<#(#args),*> },
        }
    };
    let static_self = self_at(quote! { 'static });
    let within_self = self_at(quote! { '__s });
    // The key and the canonical form name each uniform parameter's
    // canonical form, so every impl that names either restates the struct's
    // predicates at it.
    let self_ident = Ident::new("Self", Span::call_site());
    let type_params: Vec<Ident> = input
        .generics
        .type_params()
        .map(|tp| tp.ident.clone())
        .chain([self_ident.clone()])
        .collect();
    let read_at_canon: Vec<Ident> = uniform.iter().cloned().chain([self_ident]).collect();
    let uniform_check = (!attr.uniform_payload && names_any(payload_ty, &read_at_canon))
        .then(|| uniform_check(&input.generics, payload_ty, &vars.type_var_idents()));
    let payload_bound = (!attr.uniform_payload && names_any(payload_ty, &type_params))
        .then(|| quote! { #payload_ty: ::acvus_extern::UniformPayload<__M>, });
    let struct_predicates = struct_predicates(&input.generics);
    let restated: Vec<syn::WherePredicate> = struct_predicates
        .iter()
        .filter_map(|predicate| {
            let mut at_canon = predicate.clone();
            syn::visit_mut::VisitMut::visit_where_predicate_mut(
                &mut Canonicalize {
                    uniform: &uniform,
                    non_type: &non_type,
                },
                &mut at_canon,
            );
            (quote! { #at_canon }.to_string() != quote! { #predicate }.to_string())
                .then_some(at_canon)
        })
        .collect();
    // A `Chosen` part is its own Rust type in the key (RFC-0076 rule 2), so
    // it is `'static` wherever the key is named; the glue fills it with one.
    let chosen = vars.chosen_idents();
    let where_predicates = quote! { #(#struct_predicates,)* #(#restated,)* #(#chosen: 'static,)* };
    let static_predicates = struct_predicates
        .iter()
        .chain(&restated)
        .map(subst::predicate_at_static);
    let static_where = quote! { #(#static_predicates,)* #(#chosen: 'static,)* };
    let at_s = AtLifetime {
        lifetimes: &lifetimes,
        at: syn::parse_quote! { '__s },
    };
    let payload_within = {
        let type_params: Vec<Ident> =
            input.generics.type_params().map(|tp| tp.ident.clone()).collect();
        (names_any(payload_ty, &type_params) || at_s.names_lifetime(payload_ty)).then(|| {
            let payload = at_s.apply(payload_ty);
            quote! { #payload: ::acvus_extern::Within<'__s>, }
        })
    };
    let within_predicates = struct_predicates
        .iter()
        .chain(&restated)
        .map(|predicate| at_s.apply_predicate(predicate));
    let key_ty = {
        let mut key = payload_ty.clone();
        syn::visit_mut::VisitMut::visit_type_mut(
            &mut Canonicalize {
                uniform: &uniform,
                non_type: &non_type,
            },
            &mut key,
        );
        at_static(&key)
    };
    let canon_args = input.generics.params.iter().map(|param| match param {
        GenericParam::Type(tp) if uniform.contains(&tp.ident) => {
            let ident = &tp.ident;
            quote! { <#ident as ::acvus_extern::Canonical<::acvus_extern::kind::Type>>::Canon }
        }
        GenericParam::Type(tp) => {
            let ident = &tp.ident;
            match vars.lookup(ident) {
                Some((kind @ (VarKind::Effect | VarKind::Len | VarKind::Identity), _)) => {
                    let marker = kind.marker();
                    quote! { <#ident as ::acvus_extern::Canonical<#marker>>::Canon }
                }
                _ => quote! { #ident },
            }
        }
        GenericParam::Lifetime(_) => quote! { 'static },
        GenericParam::Const(c) => {
            let ident = &c.ident;
            quote! { #ident }
        }
    });
    let laid_params = input.generics.params.iter().map(|param| match param {
        GenericParam::Lifetime(_) => quote! { ::acvus_extern::laid::ParamKind::Region },
        GenericParam::Type(tp) => match vars.lookup(&tp.ident) {
            Some((VarKind::Ty, _)) => quote! { ::acvus_extern::laid::ParamKind::Type },
            _ => quote! { ::acvus_extern::laid::ParamKind::Other },
        },
        GenericParam::Const(_) => quote! { ::acvus_extern::laid::ParamKind::Other },
    });
    let type_arg_exprs = vars.type_arg_exprs();
    let effect_arg_exprs = vars.effect_arg_exprs();
    let identity_arg_exprs = vars.identity_arg_exprs();
    let n_tys = vars.count(VarKind::Ty);
    let n_effects = vars.count(VarKind::Effect);
    let n_identities = vars.count(VarKind::Identity);
    if n_identities > 1 {
        return Err(syn::Error::new(
            ident.span(),
            "an extension type has at most one identity parameter; a value is one source",
        ));
    }

    let qref = qref_expr_in(attr.ns.as_deref(), &name);
    let declaration_form = {
        let args = input.generics.params.iter().map(|param| match param {
            GenericParam::Lifetime(_) => quote! { 'static },
            GenericParam::Type(tp) if vars.runtime_ident() == Some(&tp.ident) => {
                quote! { ::acvus_extern::TypesOnly }
            }
            GenericParam::Type(_) => quote! { () },
            GenericParam::Const(c) => {
                let ident = &c.ident;
                quote! { #ident }
            }
        });
        quote! { #ident<#(#args),*> }
    };
    let space_hooks = match attr.space {
        false => quote! {},
        true => {
            if let Some(chosen) = vars.chosen() {
                return Err(syn::Error::new(
                    chosen.span(),
                    format!(
                        "`{chosen}` is a `Chosen` type parameter, so each instance's box is keyed \
                         at its own Rust type and no one set of space hooks reads them all: a \
                         type a space holds has only uniform type parameters (RFC-0033)"
                    ),
                ));
            }
            let held_args = input.generics.params.iter().map(|param| match param {
                GenericParam::Lifetime(_) => quote! { 'static },
                GenericParam::Type(tp) => {
                    let ident = &tp.ident;
                    quote! { #ident }
                }
                GenericParam::Const(c) => {
                    let ident = &c.ident;
                    quote! { #ident }
                }
            });
            let held: Type = if input.generics.params.is_empty() {
                syn::parse_quote! { #ident }
            } else {
                syn::parse_quote! { #ident<#(#held_args),*> }
            };
            let held = vars.to_runtime_instance(&held, None);
            quote! {
                fn space<__R>() -> ::core::option::Option<::acvus_extern::SpaceHooks<__R>>
                where
                    __R: ::acvus_extern::Runtime,
                {
                    ::core::option::Option::Some(::acvus_extern::SpaceHooks::of::<#held>())
                }
            }
        }
    };
    let one_value_run = one_value_run(returned_as_one_value());
    let payload_crossing = quote! {
        fn erase(self, __rt: ::acvus_extern::Crossing<'_, __R>) -> <__R as ::acvus_extern::Runtime>::Value {
            ::acvus_extern::derive::transparent::erase::<Self, #key_ty, __R>(self, __rt)
        }

        unsafe fn materialize(__rt: ::acvus_extern::Crossing<'_, __R>, __value: <__R as ::acvus_extern::Runtime>::Value) -> Self {
            // SAFETY: the caller's contract, and `erase` is `transparent::erase`.
            unsafe {
                ::acvus_extern::derive::transparent::materialize::<Self, #key_ty, __R>(__rt, __value)
            }
        }
    };
    // The key is the type `erase` hands the runtime, and the one
    // `deref_mut` and `project_mut` lend.
    let payload_lends_a_word = quote! { ::acvus_extern::repr::may_lie_in_the_word::<#key_ty>() };
    let payload_in_place = quote! {
        const LENDS_A_WORD: bool = #payload_lends_a_word;

        unsafe fn deref<'__a>(
            __rt: &__R,
            __reference: &'__a <__R as ::acvus_extern::Runtime>::Value,
        ) -> &'__a Self {
            // SAFETY: the caller's contract: a live storage of the payload.
            unsafe { ::acvus_extern::derive::transparent::deref::<Self, #key_ty, __R>(__rt, __reference) }
        }

        unsafe fn deref_mut<'__a>(
            __rt: &__R,
            __reference: &'__a <__R as ::acvus_extern::Runtime>::Value,
        ) -> &'__a mut Self {
            // SAFETY: as in `deref`, exclusively.
            unsafe {
                ::acvus_extern::derive::transparent::deref_mut::<Self, #key_ty, __R>(__rt, __reference)
            }
        }
    };
    let stored = (n_regions == 0).then(|| quote! {
        // SAFETY: the payload is the struct's one `#[repr(transparent)]` field,
        // checked above, which `erase` hands the runtime; nothing else is read,
        // and the capability is not used.
        unsafe impl<#static_params __R> ::acvus_extern::Stored<__R> for #static_self
        where
            __R: ::acvus_extern::Runtime,
            #static_where
        {
            type Payload = #key_ty;

            fn from_payload<'__p>(
                _: ::acvus_extern::Holding<'_, __R>,
                __payload: &'__p #key_ty,
            ) -> &'__p Self {
                ::acvus_extern::derive::transparent::from_payload::<Self, #key_ty>(__payload)
            }

            fn from_payload_mut<'__p>(
                _: ::acvus_extern::Holding<'_, __R>,
                __payload: &'__p mut #key_ty,
            ) -> &'__p mut Self {
                ::acvus_extern::derive::transparent::from_payload_mut::<Self, #key_ty>(__payload)
            }
        }
    });
    let payload_names_a_parameter = {
        let type_params: Vec<Ident> =
            input.generics.type_params().map(|tp| tp.ident.clone()).collect();
        names_any(payload_ty, &type_params)
    };
    let projected = (n_regions == 0 && !payload_names_a_parameter).then(|| quote! {
        impl<#static_params> ::acvus_extern::Borrowed for #static_self
        where
            #static_where
        {
            type Ref<'__a> = &'__a #key_ty where Self: '__a;
            type Mut<'__a> = &'__a mut #key_ty where Self: '__a;
        }

        impl<#static_params> ::acvus_extern::OwnStorage for #static_self
        where
            #static_where
        {
        }

        impl<#static_params __R> ::acvus_extern::Project<__R> for #static_self
        where
            __R: ::acvus_extern::Runtime,
            #static_where
        {
            type Table = ();

            const LENDS_A_WORD: bool = #payload_lends_a_word;

            fn table(_: ::acvus_extern::ArgAt<'_>) {}

            unsafe fn project<'__a>(
                __rt: &'__a __R,
                __value: &'__a <__R as ::acvus_extern::Runtime>::Value,
                _: &(),
            ) -> &'__a #key_ty {
                // SAFETY: the caller's contract, and `erase` boxed the payload
                // at this key.
                unsafe { __rt.value_as_ref::<#key_ty>(__value) }
            }

            unsafe fn project_mut<'__a>(
                __rt: &'__a __R,
                __value: &'__a mut <__R as ::acvus_extern::Runtime>::Value,
                _: &(),
            ) -> &'__a mut #key_ty {
                // SAFETY: as `project`, with the caller's exclusive loan.
                unsafe { __rt.value_as_mut::<#key_ty>(__value) }
            }

            unsafe fn loan_ended(
                _: &__R,
                __value: &mut <__R as ::acvus_extern::Runtime>::Value,
                _: &(),
            ) {
                if <Self as ::acvus_extern::Project<__R>>::LENDS_A_WORD {
                    // `project_mut` lent the value itself.
                    <__R as ::acvus_extern::Runtime>::loan_ended(__value)
                }
            }
        }
    });
    let identity_indices: Option<Vec<usize>> = input
        .generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Type(tp) => match vars.lookup(&tp.ident) {
                Some((VarKind::Identity, index)) => Some(index),
                _ => None,
            },
            GenericParam::Lifetime(_) | GenericParam::Const(_) => None,
        })
        .collect();
    let declared = identity_indices.map(|indices| {
        let (held, declaring) = match indices.is_empty() {
            true => (quote! { #ident }, quote! { #ident }),
            false => {
                let held = indices.iter().map(|_| quote! { () });
                let declaring = indices.iter().map(
                    |k| quote! { ::acvus_extern::Nth<::acvus_extern::kind::Identity, #k> },
                );
                (quote! { #ident<#(#held),*> }, quote! { #ident<#(#declaring),*> })
            }
        };
        quote! {
            impl ::acvus_extern::Declared for #held {
                fn declared(__i: &::acvus_extern::Interner) -> ::acvus_extern::PolyTy {
                    <#declaring as ::acvus_extern::TyArg>::poly_ty(
                        __i,
                        &::acvus_extern::PolyVars::fresh(0, 0, 0, #n_identities),
                    )
                }
            }
        }
    });
    let passed = quote! {
        type As = Self;

        fn cross(__rt: ::acvus_extern::Crossing<'_, __R>, __passed: Self) -> <__R as ::acvus_extern::Runtime>::Value {
            <Self as ::acvus_extern::OneValue<__R>>::erase(__passed, __rt)
        }

        unsafe fn restore(
            __rt: ::acvus_extern::Crossing<'_, __R>,
            __word: <__R as ::acvus_extern::Runtime>::Value,
        ) -> Self {
            // SAFETY: the caller's contract, which is `materialize`'s.
            unsafe { <Self as ::acvus_extern::OneValue<__R>>::materialize(__rt, __word) }
        }
    };

    Ok(quote! {
        impl<#impl_params> ::acvus_extern::Var<::acvus_extern::kind::Type> for #ident #ty_generics
        where
            #where_predicates
        {
        }

        #uniform_check

        // SAFETY: the type's own lifetimes are at `'__s`, and the payload, the
        // one field that is not `PhantomData`, is bounded here to hold its
        // carriers at `'__s` where it names a parameter; a payload that names
        // none is one type at every lifetime.
        unsafe impl<'__s, #static_params> ::acvus_extern::Within<'__s> for #within_self
        where
            #(#within_predicates,)*
            #payload_within
        {
        }

        // SAFETY: the struct is `#[repr(transparent)]` over its payload, which
        // is bounded by `UniformPayload` here where it names a type
        // parameter, or asserted to be one by `unsafe(uniform_payload)`.
        unsafe impl<#impl_params __M> ::acvus_extern::UniformPayload<__M> for #ident #ty_generics
        where
            #where_predicates
            #payload_bound
        {
        }

        // SAFETY: the canonical form takes each uniform parameter and each
        // effect, length and identity parameter to its own, each lifetime to
        // `'static`, and keeps every other, and the payload
        // is `UniformPayload`, proved above with each type variable held as
        // itself, or asserted to be by `unsafe(uniform_payload)`.
        unsafe impl<#impl_params> ::acvus_extern::Canonical<::acvus_extern::kind::Type>
            for #ident #ty_generics
        where
            #where_predicates
        {
            type Canon = #ident<#(#canon_args),*>;
        }

        impl #arg_impl_generics ::acvus_extern::TyArg for #static_self
        where
            #static_where
        {
            const LAYOUT: ::acvus_extern::Layout<Self> = ::acvus_extern::Layout::laid_out();

            fn poly_ty(
                __i: &::acvus_extern::Interner,
                __vars: &::acvus_extern::PolyVars,
            ) -> ::acvus_extern::PolyTy {
                ::acvus_extern::PolyTy::UserDefined {
                    id: __vars.extension::<Self>(__i),
                    type_args: vec![#(#type_arg_exprs),*],
                    effect_args: vec![#(#effect_arg_exprs),*],
                    identity_args: vec![#(#identity_arg_exprs),*],
                    region_params: <Self as ::acvus_extern::ExternTypeDecl>::REGION_PARAMS,
                }
            }
        }

        // SAFETY: every method is the type's own `OneValue` at one word;
        // nothing else crosses, and the capability is not kept.
        unsafe impl<#impl_params __R> ::acvus_extern::Cross<__R> for #ident #ty_generics
        where
            __R: ::acvus_extern::Runtime,
            #where_predicates
        {
            #one_value_run
        }

        // SAFETY: `transparent::erase` and `transparent::materialize` box and
        // unbox the payload the struct is transparent over, at the type the
        // derive names; nothing else crosses, and the capability is not kept.
        unsafe impl<#impl_params __R> ::acvus_extern::OneValue<__R> for #ident #ty_generics
        where
            __R: ::acvus_extern::Runtime,
            #where_predicates
        {
            #payload_crossing
        }

        // SAFETY: as the uniform impl's.
        unsafe impl<#impl_params __R> ::acvus_extern::OneValue<__R, ::acvus_extern::Specialized>
            for #ident #ty_generics
        where
            __R: ::acvus_extern::Runtime,
            #where_predicates
        {
            #payload_crossing
        }

        // SAFETY: `cross` and `restore` are the type's own `erase` and
        // `materialize`; nothing else crosses, and the capability is not kept.
        unsafe impl<'__p, #impl_params __R> ::acvus_extern::Passed<'__p, __R> for #ident #ty_generics
        where
            __R: ::acvus_extern::Runtime,
            #where_predicates
        {
            #passed
        }

        impl<#impl_params __R> ::acvus_extern::BorrowableSpecialized<__R> for #ident #ty_generics
        where
            __R: ::acvus_extern::Runtime,
            #where_predicates
        {
            #payload_in_place
        }

        // SAFETY: the struct is `#[repr(transparent)]`, checked above, with the
        // payload as its one non-zero-sized field: every other field is
        // `PhantomData`, also checked above. The key is the payload with each
        // uniform parameter at its canonical form and each lifetime at
        // `'static`, as the canonical form of the struct is.
        unsafe impl<#impl_params> ::acvus_extern::Transparent<#key_ty> for #ident #ty_generics
        where
            #where_predicates
        {
        }

        impl<#impl_params __R> ::acvus_extern::Borrowable<__R> for #ident #ty_generics
        where
            __R: ::acvus_extern::Runtime,
            #where_predicates
        {
            #payload_in_place
        }

        #stored

        #projected

        #declared

        // SAFETY: `laid_params` names each generic parameter in declaration
        // order; `poly_ty` above counts the lifetimes as `region_params` and
        // lists the type variables as `type_args` in that order.
        unsafe impl #arg_impl_generics ::acvus_extern::LaidOut for #static_self
        where
            #static_where
        {
            const PARAMS: &'static [::acvus_extern::laid::ParamKind] = &[#(#laid_params),*];
        }

        impl<#static_params> ::acvus_extern::ExternTypeDecl for #static_self
        where
            #static_where
        {
            type DeclarationForm = #declaration_form;

            const REGION_PARAMS: usize = #n_regions;

            fn type_decl(__i: &::acvus_extern::Interner) -> ::acvus_extern::UserDefinedDecl {
                ::acvus_extern::UserDefinedDecl {
                    qref: #qref,
                    type_params: vec![::acvus_extern::TyVarBound::Any; #n_tys],
                    effect_params: #n_effects,
                    identity_params: #n_identities,
                    region_params: Self::REGION_PARAMS,
                    specializable: vec![true; #n_tys],
                    // Not read from the payload: no statement of the payload's
                    // own fields reaches the derive, and the payload's spelling
                    // is not a fact about what it holds (RFC-0095 rule 4).
                    may_hold_a_function: true,
                }
            }

            #space_hooks
        }
    })
}

/// Every predicate a struct's generics carry, written in its `where`
/// clause or inline on a parameter.
fn struct_predicates(generics: &syn::Generics) -> Vec<syn::WherePredicate> {
    let inline = generics.params.iter().filter_map(|param| match param {
        GenericParam::Type(tp) if !tp.bounds.is_empty() => {
            let ident = &tp.ident;
            let bounds = &tp.bounds;
            Some(syn::parse_quote! { #ident: #bounds })
        }
        _ => None,
    });
    let written = generics
        .where_clause
        .iter()
        .flat_map(|w| w.predicates.iter().cloned());
    inline.chain(written).collect()
}

/// The payload's obligation, proved at a marker `__M` that only this `fn`
/// names, with each type variable assumed `UniformPayload<__M>`: a bound
/// the struct writes, or an associated type a trait declares, names no such
/// marker, so the proof reaches a type variable only by holding it.
fn uniform_check(
    generics: &syn::Generics,
    payload_ty: &Type,
    type_vars: &[Ident],
) -> proc_macro2::TokenStream {
    let params = generics.params.iter().map(|param| match param {
        GenericParam::Type(tp) => {
            let ident = &tp.ident;
            quote! { #ident }
        }
        GenericParam::Lifetime(lt) => {
            let lifetime = &lt.lifetime;
            quote! { #lifetime }
        }
        GenericParam::Const(c) => {
            let ident = &c.ident;
            let ty = &c.ty;
            quote! { const #ident: #ty }
        }
    });
    let predicates = struct_predicates(generics);
    let obligation = quote::quote_spanned! {syn::spanned::Spanned::span(payload_ty)=>
        ::acvus_extern::derive::uniform_payload::<#payload_ty, __M>()
    };
    quote! {
        const _: () = {
            #[allow(dead_code)]
            fn __uniform_payload<#(#params,)* __M>()
            where
                #(#predicates,)*
                #(#type_vars: ::acvus_extern::UniformPayload<__M>,)*
            {
                #obligation;
            }
        };
    }
}

/// Whether `node` names one of `idents` anywhere in its tokens.
fn names_any<T>(node: &T, idents: &[Ident]) -> bool
where
    T: quote::ToTokens,
{
    fn walk(tokens: proc_macro2::TokenStream, idents: &[Ident]) -> bool {
        tokens.into_iter().any(|tree| match tree {
            proc_macro2::TokenTree::Ident(ident) => idents.contains(&ident),
            proc_macro2::TokenTree::Group(group) => walk(group.stream(), idents),
            proc_macro2::TokenTree::Punct(_) | proc_macro2::TokenTree::Literal(_) => false,
        })
    }
    walk(node.to_token_stream(), idents)
}

/// Takes each uniform type parameter to its canonical form.
struct Canonicalize<'a> {
    uniform: &'a [Ident],
    non_type: &'a [(Ident, Type)],
}

impl syn::visit_mut::VisitMut for Canonicalize<'_> {
    fn visit_type_mut(&mut self, ty: &mut Type) {
        if let Type::Path(path) = ty
            && path.qself.is_none()
            && let Some(ident) = path.path.get_ident()
        {
            if self.uniform.contains(ident) {
                let ident = ident.clone();
                *ty = syn::parse_quote! {
                    <#ident as ::acvus_extern::Canonical<::acvus_extern::kind::Type>>::Canon
                };
                return;
            }
            if let Some((ident, marker)) = self.non_type.iter().find(|(other, _)| other == ident) {
                *ty = syn::parse_quote! {
                    <#ident as ::acvus_extern::Canonical<#marker>>::Canon
                };
                return;
            }
        }
        syn::visit_mut::visit_type_mut(self, ty);
    }
}

/// The first projection in `ty` whose self type is one of `uniform`,
/// written `<T as Tr>::A`, `<T>::A` or `T::A`, with that parameter.
fn projection_through(ty: &Type, uniform: &[Ident]) -> Option<(Ident, syn::TypePath)> {
    struct Find<'a> {
        uniform: &'a [Ident],
        found: Option<(Ident, syn::TypePath)>,
    }

    impl syn::visit_mut::VisitMut for Find<'_> {
        fn visit_type_path_mut(&mut self, path: &mut syn::TypePath) {
            let through = match &path.qself {
                Some(qself) => match &*qself.ty {
                    Type::Path(inner) if inner.qself.is_none() => inner.path.get_ident().cloned(),
                    _ => None,
                },
                None if path.path.leading_colon.is_none() && path.path.segments.len() > 1 => {
                    Some(path.path.segments[0].ident.clone())
                }
                None => None,
            };
            if self.found.is_none()
                && let Some(through) = through
                && self.uniform.contains(&through)
            {
                self.found = Some((through, path.clone()));
            }
            syn::visit_mut::visit_type_path_mut(self, path);
        }
    }

    let mut find = Find {
        uniform,
        found: None,
    };
    syn::visit_mut::VisitMut::visit_type_mut(&mut find, &mut ty.clone());
    find.found
}

// -- #[derive(Payload)] ----------------------------------------------

/// A payload's two proofs, `UniformPayload` at every marker (RFC-0076 rule 4)
/// and `Within` at every lifetime (RFC-0079 rule 6), each from the fields.
#[proc_macro_derive(Payload)]
pub fn derive_payload(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derived_fields(&input, "Payload") {
        Ok(fields) => {
            let uniform = generate_uniform_payload(&input, &fields);
            let within = generate_within(&input, &fields);
            quote! { #uniform #within }.into()
        }
        Err(err) => err.to_compile_error().into(),
    }
}

fn derived_fields<'a>(input: &'a DeriveInput, derive: &str) -> syn::Result<Vec<&'a syn::Field>> {
    match &input.data {
        syn::Data::Struct(data) => Ok(data.fields.iter().collect()),
        syn::Data::Enum(data) => Ok(data.variants.iter().flat_map(|v| v.fields.iter()).collect()),
        syn::Data::Union(_) => Err(syn::Error::new(
            input.ident.span(),
            format!("{derive} is derived on a struct or an enum"),
        )),
    }
}

fn generate_uniform_payload(
    input: &DeriveInput,
    fields: &[&syn::Field],
) -> proc_macro2::TokenStream {
    let ident = &input.ident;
    let type_params: Vec<Ident> = input
        .generics
        .type_params()
        .map(|tp| tp.ident.clone())
        .chain([Ident::new("Self", Span::call_site())])
        .collect();
    let bounded = fields
        .iter()
        .map(|field| &field.ty)
        .filter(|ty| names_any(*ty, &type_params));
    let written = input
        .generics
        .where_clause
        .iter()
        .flat_map(|w| w.predicates.iter());
    let params = input.generics.params.iter();
    let (_, ty_generics, _) = input.generics.split_for_impl();
    quote! {
        // SAFETY: each field that names a type parameter is bounded by
        // `UniformPayload` here, and a field that names none has one layout
        // at every instantiation.
        unsafe impl<#(#params,)* __M> ::acvus_extern::UniformPayload<__M> for #ident #ty_generics
        where
            #(#written,)*
            #(#bounded: ::acvus_extern::UniformPayload<__M>,)*
        {
        }
    }
}

// -- #[derive(Within)] -----------------------------------------------

/// `Within` alone, for a part of a payload whose `UniformPayload` the
/// extension type asserts by `unsafe(uniform_payload)` (RFC-0076 rule 5).
#[proc_macro_derive(Within)]
pub fn derive_within(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derived_fields(&input, "Within") {
        Ok(fields) => generate_within(&input, &fields).into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn generate_within(input: &DeriveInput, fields: &[&syn::Field]) -> proc_macro2::TokenStream {
    let ident = &input.ident;
    let lifetimes: Vec<syn::Lifetime> = input
        .generics
        .lifetimes()
        .map(|lt| lt.lifetime.clone())
        .collect();
    let at_s = AtLifetime {
        lifetimes: &lifetimes,
        at: syn::parse_quote! { '__s },
    };
    let predicates = struct_predicates(&input.generics)
        .into_iter()
        .map(|predicate| at_s.apply_predicate(&predicate));
    let type_params: Vec<Ident> = input.generics.type_params().map(|tp| tp.ident.clone()).collect();
    let bounded = fields
        .iter()
        .filter(|field| names_any(&field.ty, &type_params) || at_s.names_lifetime(&field.ty))
        .map(|field| {
            let ty = at_s.apply(&field.ty);
            quote! { #ty: ::acvus_extern::Within<'__s> }
        });
    let self_at = {
        let args = input.generics.params.iter().map(|param| match param {
            GenericParam::Lifetime(_) => quote! { '__s },
            GenericParam::Type(tp) => {
                let ident = &tp.ident;
                quote! { #ident }
            }
            GenericParam::Const(c) => {
                let ident = &c.ident;
                quote! { #ident }
            }
        });
        match input.generics.params.is_empty() {
            true => quote! { #ident },
            false => quote! { #ident<#(#args),*> },
        }
    };
    let params = input
        .generics
        .params
        .iter()
        .filter(|param| !matches!(param, GenericParam::Lifetime(_)));
    quote! {
        // SAFETY: the type's own lifetimes are at `'__s`, and each field that
        // names a parameter is bounded here to hold its carriers at `'__s`. A
        // field that names none is one type at every lifetime.
        unsafe impl<'__s, #(#params,)*> ::acvus_extern::Within<'__s> for #self_at
        where
            #(#predicates,)*
            #(#bounded,)*
        {
        }
    }
}

struct AtLifetime<'a> {
    lifetimes: &'a [syn::Lifetime],
    at: syn::Lifetime,
}

impl AtLifetime<'_> {
    fn apply(&self, ty: &Type) -> Type {
        let mut ty = ty.clone();
        syn::visit_mut::VisitMut::visit_type_mut(&mut &*self, &mut ty);
        ty
    }

    fn names_lifetime(&self, ty: &Type) -> bool {
        struct Find<'a> {
            lifetimes: &'a [syn::Lifetime],
            found: bool,
        }

        impl<'ast> syn::visit::Visit<'ast> for Find<'_> {
            fn visit_lifetime(&mut self, lifetime: &'ast syn::Lifetime) {
                self.found |= self.lifetimes.contains(lifetime);
            }
        }

        let mut find = Find {
            lifetimes: self.lifetimes,
            found: false,
        };
        syn::visit::Visit::visit_type(&mut find, ty);
        find.found
    }

    fn apply_predicate(&self, predicate: &syn::WherePredicate) -> syn::WherePredicate {
        let mut predicate = predicate.clone();
        syn::visit_mut::VisitMut::visit_where_predicate_mut(&mut &*self, &mut predicate);
        predicate
    }
}

impl syn::visit_mut::VisitMut for &AtLifetime<'_> {
    fn visit_lifetime_mut(&mut self, lifetime: &mut syn::Lifetime) {
        if self.lifetimes.contains(lifetime) {
            *lifetime = self.at.clone();
        }
    }
}

// -- #[derive(TyArg)] ------------------------------------------------

/// `#[ty_arg(ns = "..")]`: the namespace the type's name is under, the root
/// when absent.
fn parse_ty_arg_ns(attrs: &[Attribute]) -> syn::Result<Option<String>> {
    let mut ns = None;
    for attr in attrs.iter().filter(|a| a.path().is_ident("ty_arg")) {
        attr.parse_nested_meta(|meta| {
            if !meta.path.is_ident("ns") {
                return Err(meta.error("expected `ns`"));
            }
            meta.input.parse::<Token![=]>()?;
            ns = Some(meta.input.parse::<LitStr>()?.value());
            Ok(())
        })?;
    }
    Ok(ns)
}

#[proc_macro_derive(TyArg, attributes(projection, ty_arg))]
pub fn derive_ty_arg(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match generate_ty_arg(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn generate_ty_arg(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let ident = &input.ident;
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new(
            ident.span(),
            "a structural type has no generic parameters",
        ));
    }
    let qref = qref_expr_in(
        parse_ty_arg_ns(&input.attrs)?.as_deref(),
        &ident.to_string(),
    );
    match &input.data {
        syn::Data::Struct(data) => {
            let syn::Fields::Named(fields) = &data.fields else {
                return Err(syn::Error::new(
                    ident.span(),
                    "TyArg is derived on a struct with named fields, one per object field",
                ));
            };
            let shape = ObjectShape::of(fields);
            let ty = shape.declared_poly_ty(ident, &qref);
            let erase = shape.erase(quote! { self });
            let materialize = shape.materialize(quote! { __value }, quote! { Self });
            let projected = input.attrs.iter().any(|a| a.path().is_ident("projection"));
            let borrowing = match projected {
                true => Borrowing::AsProjection,
                false => Borrowing::Whole,
            };
            let cross = cross_impl(
                ident,
                Crossing {
                    ty,
                    erase,
                    materialize,
                    borrowing,
                    returned: shape.returned(),
                },
            );
            let projection = projected.then(|| shape.projection(ident));
            Ok(quote! { #cross #projection })
        }
        syn::Data::Enum(data) => {
            let projected = input.attrs.iter().any(|a| a.path().is_ident("projection"));
            generate_enum_ty_arg(ident, &qref, data, projected)
        }
        syn::Data::Union(_) => Err(syn::Error::new(
            ident.span(),
            "TyArg is derived on a struct or an enum",
        )),
    }
}

#[derive(Clone, Copy)]
enum Borrowing {
    Whole,
    AsProjection,
}

impl Borrowing {
    /// The `Borrowable` impl of an aggregate. Its storage is an object, which
    /// holds no `Self`, so a type without a projection has none; a type with
    /// one has an impl under `BorrowedWhole`, which has no impl, so that
    /// `&S` is refused with the message naming the projection.
    fn borrowable(self, ident: &Ident) -> proc_macro2::TokenStream {
        match self {
            Borrowing::Whole => quote! {},
            Borrowing::AsProjection => quote! {
                impl<__R> ::acvus_extern::Borrowable<__R> for #ident
                where
                    __R: ::acvus_extern::Runtime,
                    Self: ::acvus_extern::BorrowedWhole<__R>,
                {
                    const LENDS_A_WORD: bool = false;

                    unsafe fn deref<'__a>(
                        __rt: &__R,
                        __reference: &'__a <__R as ::acvus_extern::Runtime>::Value,
                    ) -> &'__a Self {
                        // SAFETY: the caller's contract, forwarded.
                        unsafe {
                            <Self as ::acvus_extern::BorrowedWhole<__R>>::deref(__rt, __reference)
                        }
                    }

                    unsafe fn deref_mut<'__a>(
                        __rt: &__R,
                        __reference: &'__a <__R as ::acvus_extern::Runtime>::Value,
                    ) -> &'__a mut Self {
                        // SAFETY: the caller's contract, forwarded.
                        unsafe {
                            <Self as ::acvus_extern::BorrowedWhole<__R>>::deref_mut(__rt, __reference)
                        }
                    }
                }
            },
        }
    }
}

/// How a derived type crosses: the acvus type it names, and the two
/// directions of the crossing itself.
struct Crossing {
    ty: proc_macro2::TokenStream,
    erase: proc_macro2::TokenStream,
    materialize: proc_macro2::TokenStream,
    borrowing: Borrowing,
    returned: proc_macro2::TokenStream,
}

/// The `TyArg` and `Cross` impls of a derived type.
fn cross_impl(ident: &Ident, crossing: Crossing) -> proc_macro2::TokenStream {
    let Crossing {
        ty,
        erase,
        materialize,
        borrowing,
        returned,
    } = crossing;
    let one_value_run = one_value_run(returned);
    let borrowable = borrowing.borrowable(ident);
    quote! {
        impl ::acvus_extern::Var<::acvus_extern::kind::Type> for #ident {}

        ::acvus_extern::within_every!(#ident);

        // SAFETY: a type with no type parameter reaches none.
        unsafe impl<__M> ::acvus_extern::UniformPayload<__M> for #ident {}

        // SAFETY: a converted type is no box and names no parameter.
        unsafe impl ::acvus_extern::Canonical<::acvus_extern::kind::Type> for #ident {
            type Canon = Self;
        }

        impl ::acvus_extern::TyArg for #ident {
            fn poly_ty(
                __i: &::acvus_extern::Interner,
                __vars: &::acvus_extern::PolyVars,
            ) -> ::acvus_extern::PolyTy {
                #ty
            }
        }

        impl ::acvus_extern::Declared for #ident {
            fn declared(__i: &::acvus_extern::Interner) -> ::acvus_extern::PolyTy {
                <Self as ::acvus_extern::TyArg>::poly_ty(__i, &::acvus_extern::PolyVars::empty())
            }
        }

        #borrowable

        // SAFETY: every method is the type's own `OneValue` or, for a struct's
        // result, its fields' own crossings; nothing else crosses, and the
        // capability is not kept.
        unsafe impl<__R> ::acvus_extern::Cross<__R> for #ident
        where
            __R: ::acvus_extern::Runtime,
        {
            #one_value_run
        }

        // SAFETY: `erase` and `materialize` are the derive's, which cross each
        // field or variant payload by that part's own crossing at the type the
        // derive read off the declaration; nothing else crosses, and the
        // capability is not kept.
        unsafe impl<__R> ::acvus_extern::OneValue<__R> for #ident
        where
            __R: ::acvus_extern::Runtime,
        {
            fn erase(self, __rt: ::acvus_extern::Crossing<'_, __R>) -> <__R as ::acvus_extern::Runtime>::Value {
                #erase
            }

            unsafe fn materialize(__rt: ::acvus_extern::Crossing<'_, __R>, __value: <__R as ::acvus_extern::Runtime>::Value) -> Self {
                #materialize
            }
        }

        // SAFETY: `cross` and `restore` are the type's own `erase` and
        // `materialize`; nothing else crosses, and the capability is not kept.
        unsafe impl<'__p, __R> ::acvus_extern::Passed<'__p, __R> for #ident
        where
            __R: ::acvus_extern::Runtime,
        {
            type As = Self;

            fn cross(__rt: ::acvus_extern::Crossing<'_, __R>, __passed: Self) -> <__R as ::acvus_extern::Runtime>::Value {
                <Self as ::acvus_extern::OneValue<__R>>::erase(__passed, __rt)
            }

            unsafe fn restore(
                __rt: ::acvus_extern::Crossing<'_, __R>,
                __word: <__R as ::acvus_extern::Runtime>::Value,
            ) -> Self {
                // SAFETY: the caller's contract, which is `materialize`'s.
                unsafe { <Self as ::acvus_extern::OneValue<__R>>::materialize(__rt, __word) }
            }
        }
    }
}

/// The `Cross` of a one-value crossing, as the library writes it: the two
/// forms and the three forwards to `OneValue`, with `__R` the runtime
/// parameter both derives already name. `returns` is the return half, which
/// is this same one value for every derived type but a struct.
fn one_value_run(returns: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    quote! {
        type Form = ::acvus_extern::One;

        unsafe fn from_run(
            __rt: ::acvus_extern::Crossing<'_, __R>,
            __run: &[<__R as ::acvus_extern::Runtime>::Value],
        ) -> Self {
            // SAFETY: the caller's contract, at one value.
            unsafe { <Self as ::acvus_extern::OneValue<__R>>::from_run(__rt, __run) }
        }

        fn into_run(self, __rt: ::acvus_extern::Crossing<'_, __R>, __out: &mut [<__R as ::acvus_extern::Runtime>::Value]) {
            <Self as ::acvus_extern::OneValue<__R>>::into_run(self, __rt, __out)
        }

        #returns
    }
}

/// A result that crosses back as the one heap value `OneValue::erase`
/// builds: every derived type but a struct, whose components go to the
/// caller's destination run instead.
fn returned_as_one_value() -> proc_macro2::TokenStream {
    quote! {
        type ReturnForm = ::acvus_extern::One;

        fn into_return_run(
            self,
            __rt: ::acvus_extern::Crossing<'_, __R>,
            __out: &mut [<__R as ::acvus_extern::Runtime>::Value],
        ) {
            <Self as ::acvus_extern::OneValue<__R>>::into_run(self, __rt, __out)
        }
    }
}

/// Named fields as an object: the struct's, or a struct variant's.
struct ObjectShape<'a> {
    idents: Vec<&'a Ident>,
    names: Vec<String>,
    tys: Vec<&'a Type>,
}

impl<'a> ObjectShape<'a> {
    /// The fields in RFC-0050 rule 8's order — ascending by the field name —
    /// and not in declaration order. Rule 8's order is what a run's layout,
    /// a committed object's canonical bytes and this table all have to agree
    /// on, and it is the field names sorted as strings: `ObjectTy` carries no
    /// declaration order for a `Declared` struct to be laid by. The comparison
    /// here is on the same strings `acvus_extern::Shape::of` resolves and
    /// compares, and `acvus-extern/tests/owned_holders.rs::
    /// a_derived_structs_field_table_is_the_shape_order` pins the two equal.
    fn of(fields: &'a syn::FieldsNamed) -> Self {
        let mut sorted: Vec<(String, &'a Ident, &'a syn::Type)> = fields
            .named
            .iter()
            .map(|f| {
                let ident = f.ident.as_ref().expect("named");
                (ident.to_string(), ident, &f.ty)
            })
            .collect();
        sorted.sort_by(|(a, ..), (b, ..)| a.cmp(b));
        Self {
            idents: sorted.iter().map(|(_, ident, _)| *ident).collect(),
            names: sorted.iter().map(|(name, ..)| name.clone()).collect(),
            tys: sorted.iter().map(|(.., ty)| *ty).collect(),
        }
    }

    fn fields(&self) -> proc_macro2::TokenStream {
        let (names, tys) = (&self.names, &self.tys);
        quote! {
            [#((
                __i.intern(#names),
                <#tys as ::acvus_extern::TyArg>::poly_ty(__i, __vars),
            )),*]
            .into_iter()
            .collect()
        }
    }

    /// The type of the struct `name` declares: these fields and no others,
    /// so an object reaching it has all of them (RFC-0042).
    fn declared_poly_ty(
        &self,
        ident: &Ident,
        qref: &proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        let fields = self.fields();
        quote! {
            ::acvus_extern::PolyTy::Object(
                ::acvus_extern::ObjectTy::declared(__vars.derived::<#ident>(#qref), #fields),
            )
        }
    }

    /// The type of a struct variant's payload, which an enum declares as
    /// the object its fields spell.
    fn written_poly_ty(&self) -> proc_macro2::TokenStream {
        let fields = self.fields();
        quote! {
            ::acvus_extern::PolyTy::Object(::acvus_extern::ObjectTy::written(#fields))
        }
    }

    /// Erases the fields reached as `#owner.field`;
    /// `acvus_extern::derive::object` owns the layout they go into.
    fn erase(&self, owner: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        let (idents, names, tys) = (&self.idents, &self.names, &self.tys);
        let width = syn::Index::from(idents.len());
        quote! {
            ::acvus_extern::derive::object::object_in_order::<__R, #width>(
                __rt,
                [#(#names),*],
                [#(::acvus_extern::derive::erase_field::<#tys, __R>(__rt, #owner.#idents)),*],
            )
        }
    }

    /// The struct's components, written into the caller's destination run in
    /// the same order `erase` writes them, which is rule 8's.
    ///
    /// The width is a literal because the derive counts the fields here.
    /// `Run<W>` reached through a type parameter's associated constant would
    /// need `generic_const_exprs`, which is unstable on the pinned
    /// toolchain.
    fn returned(&self) -> proc_macro2::TokenStream {
        let (idents, tys) = (&self.idents, &self.tys);
        let width = syn::Index::from(idents.len());
        quote! {
            type ReturnForm = ::acvus_extern::Run<#width>;

            fn into_return_run(
                self,
                __rt: ::acvus_extern::Crossing<'_, __R>,
                __out: &mut [<__R as ::acvus_extern::Runtime>::Value],
            ) {
                ::acvus_extern::derive::object::fields_into_run::<__R, #width>(
                    __rt.holding(),
                    [#(::acvus_extern::derive::erase_field::<#tys, __R>(__rt, self.#idents)),*],
                    __out,
                )
            }
        }
    }

    /// Erases fields already bound to their own idents (a matched variant).
    fn erase_bound(&self) -> proc_macro2::TokenStream {
        let (idents, names, tys) = (&self.idents, &self.names, &self.tys);
        let width = syn::Index::from(idents.len());
        quote! {
            ::acvus_extern::derive::object::object_in_order::<__R, #width>(
                __rt,
                [#(#names),*],
                [#(::acvus_extern::derive::erase_field::<#tys, __R>(__rt, #idents)),*],
            )
        }
    }

    /// `SRef<'a>` and `SMut<'a>`: one field per declared field, each a
    /// borrow of the value in the object's own storage, and the impls that
    /// build them out of an object the caller lent (RFC-0050 rule 6).
    ///
    /// The acvus type of a projection is `&S` over an **at-least** field set,
    /// where the struct's own is `Declared`: a projection naming a subset of
    /// the object's fields borrows those alone, which `ObjectTy::meet`
    /// already admits and `acvus-mir-test/tests/projection_parameter.rs`
    /// pins.
    ///
    /// Rule 6 gives every derived struct a projection and `#[projection]`
    /// asks for one, which is less. A projection type carries a lifetime and
    /// no runtime, so each of its fields must name a borrow that does not
    /// mention the runtime either — `&'a String`, `&'a i64`, a nested
    /// projection. A container field refutes that: the language stores a
    /// `Vec<T>` as `Vec<Owned<Rt>>`, so the borrow of such a field is a
    /// borrow of the runtime's own values and only `SRef<'a, Rt>` could name
    /// it. Until the projection carries the runtime, a struct with a
    /// container field has no projection to emit.
    fn projection(&self, owner: &Ident) -> proc_macro2::TokenStream {
        let (idents, names, tys) = (&self.idents, &self.names, &self.tys);
        let shared = format_ident!("{owner}Ref");
        let exclusive = format_ident!("{owner}Mut");
        let fields = self.fields();
        let width = syn::Index::from(idents.len());
        let doc_shared = format!("A shared projection of [`{owner}`] (RFC-0050 rule 6).");
        let doc_exclusive = format!("An exclusive projection of [`{owner}`] (RFC-0050 rule 6).");
        let at_least = quote! {
            ::acvus_extern::PolyTy::Object(::acvus_extern::ObjectTy::at_least(#fields))
        };
        let ats: Vec<syn::Index> = (0..idents.len()).map(syn::Index::from).collect();
        let table_ty = quote! {
            ::acvus_extern::ObjectAt<
                #width,
                (#(<#tys as ::acvus_extern::Project<__R>>::Table,)*),
            >
        };
        let table_of = quote! {
            let [#(#idents),*] = ::acvus_extern::object_fields_at(__at, [#(#names),*]);
            ::acvus_extern::ObjectAt {
                at: [#(#idents.0),*],
                fields: (#(
                    <#tys as ::acvus_extern::Project<__R>>::table(#idents.1),
                )*),
            }
        };
        quote! {
            #[doc = #doc_shared]
            pub struct #shared<'__a> {
                #(pub #idents: <#tys as ::acvus_extern::Borrowed>::Ref<'__a>,)*
            }

            #[doc = #doc_exclusive]
            pub struct #exclusive<'__a> {
                #(pub #idents: <#tys as ::acvus_extern::Borrowed>::Mut<'__a>,)*
            }

            impl ::acvus_extern::Borrowed for #owner {
                type Ref<'__a> = #shared<'__a> where Self: '__a;
                type Mut<'__a> = #exclusive<'__a> where Self: '__a;
            }

            impl ::acvus_extern::OwnStorage for #owner {}

            impl<'__q> ::acvus_extern::Projects for #shared<'__q> {
                type Owner = #owner;
                type Loan = ::acvus_extern::Shared;
            }

            impl<'__q> ::acvus_extern::Projects for #exclusive<'__q> {
                type Owner = #owner;
                type Loan = ::acvus_extern::Mut;
            }

            impl<'__a> #shared<'__a> {
                /// # Safety
                /// `obj` holds what the owner's crossing wrote and is live
                /// for `'__a`.
                pub unsafe fn over<__R>(
                    __rt: &'__a __R,
                    __obj: &'__a ::acvus_extern::Obj<::acvus_extern::Owned<__R>>,
                    __table: &#table_ty,
                ) -> Self
                where
                    __R: ::acvus_extern::Runtime,
                {
                    // SAFETY: the caller's contract, and every field read
                    // goes to this projection's own crossings below.
                    let __fields = unsafe {
                        ::acvus_extern::Fields::<::acvus_extern::Shared, __R>::of(__obj)
                    };
                    let [#(#idents),*] = __table.at;
                    Self {
                        #(#idents: {
                            // SAFETY: the caller's contract, and the field
                            // at that position holds what this field's own
                            // crossing wrote.
                            unsafe {
                                <#tys as ::acvus_extern::Project<__R>>::project(
                                    __rt,
                                    __fields.field(#idents),
                                    &__table.fields.#ats,
                                )
                            }
                        },)*
                    }
                }
            }

            impl<'__a> #exclusive<'__a> {
                /// # Safety
                /// As the shared projection's `over`, and `obj` is
                /// exclusively named for `'__a`.
                pub unsafe fn over<__R>(
                    __rt: &'__a __R,
                    __obj: &'__a mut ::acvus_extern::Obj<::acvus_extern::Owned<__R>>,
                    __table: &#table_ty,
                ) -> Self
                where
                    __R: ::acvus_extern::Runtime,
                {
                    // SAFETY: as the shared projection's, exclusively.
                    let [#(#idents),*] = unsafe {
                        ::acvus_extern::Fields::<::acvus_extern::Mut, __R>::of(__obj)
                            .disjoint::<#width>(__table.at)
                    };
                    Self {
                        // SAFETY: as the shared projection's, exclusively.
                        #(#idents: unsafe {
                            <#tys as ::acvus_extern::Project<__R>>::project_mut(
                                __rt,
                                #idents,
                                &__table.fields.#ats,
                            )
                        },)*
                    }
                }

                /// The projection `over` built has ended: each field's
                /// storage, at the position the table names, goes to its
                /// own crossing's `Project::loan_ended`, which hands a
                /// storage lent in place to `Runtime::loan_ended`.
                ///
                /// # Safety
                /// As `over`'s, over the object `over` projected, and no
                /// borrow the projection handed out is live.
                pub unsafe fn ended<__R>(
                    __rt: &__R,
                    __obj: &mut ::acvus_extern::Obj<::acvus_extern::Owned<__R>>,
                    __table: &#table_ty,
                ) where
                    __R: ::acvus_extern::Runtime,
                {
                    // SAFETY: as `over`'s: the same positions of the same
                    // object.
                    let [#(#idents),*] = unsafe {
                        ::acvus_extern::Fields::<::acvus_extern::Mut, __R>::of(__obj)
                            .disjoint::<#width>(__table.at)
                    };
                    // SAFETY: the caller's contract, at each field `over`
                    // projected with this table.
                    #(unsafe {
                        <#tys as ::acvus_extern::Project<__R>>::loan_ended(
                            __rt,
                            #idents,
                            &__table.fields.#ats,
                        )
                    };)*
                }
            }

            impl<__R> ::acvus_extern::Project<__R> for #owner
            where
                __R: ::acvus_extern::Runtime,
            {
                type Table = #table_ty;

                const LENDS_A_WORD: bool =
                    false #(|| <#tys as ::acvus_extern::Project<__R>>::LENDS_A_WORD)*;

                fn table(__at: ::acvus_extern::ArgAt<'_>) -> Self::Table {
                    #table_of
                }

                unsafe fn project<'__a>(
                    __rt: &'__a __R,
                    __value: &'__a <__R as ::acvus_extern::Runtime>::Value,
                    __table: &Self::Table,
                ) -> #shared<'__a> {
                    // SAFETY: the caller's contract: a nested aggregate field
                    // holds its own object.
                    unsafe {
                        #shared::over(
                            __rt,
                            ::acvus_extern::object::<::acvus_extern::Nested, ::acvus_extern::Shared, __R>(__rt, __value),
                            __table,
                        )
                    }
                }

                unsafe fn project_mut<'__a>(
                    __rt: &'__a __R,
                    __value: &'__a mut <__R as ::acvus_extern::Runtime>::Value,
                    __table: &Self::Table,
                ) -> #exclusive<'__a> {
                    // SAFETY: as `project`, with the caller's exclusive loan.
                    unsafe {
                        #exclusive::over(
                            __rt,
                            ::acvus_extern::object::<::acvus_extern::Nested, ::acvus_extern::Mut, __R>(__rt, __value),
                            __table,
                        )
                    }
                }

                unsafe fn loan_ended(
                    __rt: &__R,
                    __value: &mut <__R as ::acvus_extern::Runtime>::Value,
                    __table: &Self::Table,
                ) {
                    if !<Self as ::acvus_extern::Project<__R>>::LENDS_A_WORD {
                        return;
                    }
                    // SAFETY: the caller's contract: the object
                    // `project_mut` projected.
                    unsafe {
                        #exclusive::ended(
                            __rt,
                            ::acvus_extern::object::<::acvus_extern::Nested, ::acvus_extern::Mut, __R>(__rt, __value),
                            __table,
                        )
                    }
                }
            }

            impl<'__q, __R> ::acvus_extern::Param<__R> for #shared<'__q>
            where
                __R: ::acvus_extern::Runtime,
            {
                type Marker = ::acvus_extern::ByProjection<#shared<'static>>;
                type At<'__a> = #shared<'__a>;
            }

            impl<'__q, __R> ::acvus_extern::Param<__R> for #exclusive<'__q>
            where
                __R: ::acvus_extern::Runtime,
            {
                type Marker = ::acvus_extern::ByProjection<#exclusive<'static>>;
                type At<'__a> = #exclusive<'__a>;
            }

            // SAFETY: a projection is at its own lifetime, and borrows only
            // parts of the object the caller lent at it.
            unsafe impl<'__s> ::acvus_extern::Within<'__s> for #shared<'__s> {}

            impl<'__a, __R> ::acvus_extern::Projected<'__a, __R> for #shared<'__a>
            where
                __R: ::acvus_extern::Runtime,
            {
                type Loan = ::acvus_extern::Shared;
                type Table = #table_ty;

                /// A shared projection lends nothing exclusively.
                const LENDS_A_WORD: bool = false;

                fn table(__at: ::acvus_extern::ArgAt<'_>) -> Self::Table {
                    #table_of
                }

                unsafe fn of(
                    __rt: &'__a __R,
                    __reference: &'__a <__R as ::acvus_extern::Runtime>::Value,
                    __table: &Self::Table,
                ) -> #shared<'__a> {
                    // SAFETY: the caller's contract: a live object storage.
                    unsafe {
                        #shared::over(
                            __rt,
                            ::acvus_extern::object::<::acvus_extern::Lent, ::acvus_extern::Shared, __R>(__rt, __reference),
                            __table,
                        )
                    }
                }

                unsafe fn loan_ended(
                    _: &__R,
                    _: &<__R as ::acvus_extern::Runtime>::Value,
                    _: &Self::Table,
                ) {
                }
            }

            // SAFETY: a projection is at its own lifetime, and borrows only
            // parts of the object the caller lent at it.
            unsafe impl<'__s> ::acvus_extern::Within<'__s> for #exclusive<'__s> {}

            impl<'__a, __R> ::acvus_extern::Projected<'__a, __R> for #exclusive<'__a>
            where
                __R: ::acvus_extern::Runtime,
            {
                type Loan = ::acvus_extern::Mut;
                type Table = #table_ty;

                const LENDS_A_WORD: bool = <#owner as ::acvus_extern::Project<__R>>::LENDS_A_WORD;

                fn table(__at: ::acvus_extern::ArgAt<'_>) -> Self::Table {
                    #table_of
                }

                unsafe fn of(
                    __rt: &'__a __R,
                    __reference: &'__a <__R as ::acvus_extern::Runtime>::Value,
                    __table: &Self::Table,
                ) -> #exclusive<'__a> {
                    // SAFETY: as the shared projection's, exclusively.
                    unsafe {
                        #exclusive::over(
                            __rt,
                            ::acvus_extern::object::<::acvus_extern::Lent, ::acvus_extern::Mut, __R>(__rt, __reference),
                            __table,
                        )
                    }
                }

                unsafe fn loan_ended(
                    __rt: &__R,
                    __reference: &<__R as ::acvus_extern::Runtime>::Value,
                    __table: &Self::Table,
                ) {
                    if !<Self as ::acvus_extern::Projected<'__a, __R>>::LENDS_A_WORD {
                        return;
                    }
                    // SAFETY: the caller's contract: the object storage `of`
                    // projected, which no borrow names any longer.
                    unsafe {
                        #exclusive::ended(
                            __rt,
                            ::acvus_extern::object::<::acvus_extern::Lent, ::acvus_extern::Mut, __R>(__rt, __reference),
                            __table,
                        )
                    }
                }
            }

            impl ::acvus_extern::Var<::acvus_extern::kind::Type> for #shared<'static> {}

            // SAFETY: a projection is no box and names no parameter.
            unsafe impl ::acvus_extern::Canonical<::acvus_extern::kind::Type> for #shared<'static> {
                type Canon = Self;
            }

            impl ::acvus_extern::TyArg for #shared<'static> {
                fn poly_ty(
                    __i: &::acvus_extern::Interner,
                    __vars: &::acvus_extern::PolyVars,
                ) -> ::acvus_extern::PolyTy {
                    ::acvus_extern::PolyTy::Ref(
                        ::acvus_extern::Mutability::Shared,
                        ::std::boxed::Box::new(::acvus_extern::TypeArg::uniform(#at_least)),
                    )
                }
            }

            impl ::acvus_extern::Var<::acvus_extern::kind::Type> for #exclusive<'static> {}

            // SAFETY: as the shared projection's.
            unsafe impl ::acvus_extern::Canonical<::acvus_extern::kind::Type> for #exclusive<'static> {
                type Canon = Self;
            }

            impl ::acvus_extern::TyArg for #exclusive<'static> {
                fn poly_ty(
                    __i: &::acvus_extern::Interner,
                    __vars: &::acvus_extern::PolyVars,
                ) -> ::acvus_extern::PolyTy {
                    ::acvus_extern::PolyTy::Ref(
                        ::acvus_extern::Mutability::Mut,
                        ::std::boxed::Box::new(::acvus_extern::TypeArg::uniform(#at_least)),
                    )
                }
            }
        }
    }

    /// Materializes `#value` into `#path { fields }`.
    fn materialize(
        &self,
        value: proc_macro2::TokenStream,
        path: proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        let (idents, tys) = (&self.idents, &self.tys);
        let width = syn::Index::from(idents.len());
        quote! {{
            // SAFETY: the caller's contract, and `erase` built this object, so
            // its width is this table's and the destructuring is exhaustive.
            let [#(#idents),*] = unsafe {
                ::acvus_extern::derive::object::open_in_order::<__R, #width>(__rt, #value)
            };
            #path {
                // SAFETY: the caller's contract, forwarded: `erase` erased each
                // field from its declared type, at this position.
                #(#idents: unsafe {
                    ::acvus_extern::derive::materialize_field::<#tys, __R>(__rt, #idents)
                },)*
            }
        }}
    }
}

/// A Rust enum is the language's enum of the same name: a unit variant has
/// no payload, a one-field tuple variant's payload is that field, and a
/// struct variant's payload is the object its fields spell.
fn generate_enum_ty_arg(
    ident: &Ident,
    qref: &proc_macro2::TokenStream,
    data: &syn::DataEnum,
    projected: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let name = ident.to_string();
    let Some(last) = data.variants.len().checked_sub(1) else {
        return Err(syn::Error::new(
            ident.span(),
            "an enum declares at least one variant: a type with no value is `Never`",
        ));
    };
    let mut tags = Vec::new();
    let mut variant_tys = Vec::new();
    let mut erase_arms = Vec::new();
    let mut materialize_arms = Vec::new();
    let mut borrowed = Vec::new();
    for (at, variant) in data.variants.iter().enumerate() {
        let v = &variant.ident;
        let tag = v.to_string();
        tags.push(tag.clone());
        // The last position absorbs the index, which `variant::opened`
        // already proved to be one of `tags`.
        let matched = match at == last {
            true => quote! { _ },
            false => {
                let at = proc_macro2::Literal::usize_unsuffixed(at);
                quote! { #at }
            }
        };
        match &variant.fields {
            syn::Fields::Unit => {
                variant_tys.push(quote! { (__i.intern(#tag), ::core::option::Option::None) });
                erase_arms.push(quote! { #ident::#v => (#tag, ::core::option::Option::None) });
                materialize_arms.push(quote! { #matched => Self::#v });
                borrowed.push(BorrowedVariant {
                    ident: v,
                    tag: tag.clone(),
                    payload: None,
                });
            }
            syn::Fields::Unnamed(fields) => {
                let mut tys = fields.unnamed.iter().map(|f| &f.ty);
                let (Some(ty), None) = (tys.next(), tys.next()) else {
                    return Err(syn::Error::new_spanned(
                        &variant.fields,
                        "a tuple variant has one field, the variant's payload",
                    ));
                };
                variant_tys.push(quote! {
                    (
                        __i.intern(#tag),
                        ::core::option::Option::Some(::std::boxed::Box::new(
                            <#ty as ::acvus_extern::TyArg>::poly_ty(__i, __vars),
                        )),
                    )
                });
                erase_arms.push(quote! {
                    #ident::#v(__payload) => (
                        #tag,
                        ::core::option::Option::Some(
                            ::acvus_extern::derive::erase_field::<#ty, __R>(__rt, __payload),
                        ),
                    )
                });
                borrowed.push(BorrowedVariant {
                    ident: v,
                    tag: tag.clone(),
                    payload: Some(ty),
                });
                materialize_arms.push(quote! {
                    #matched => {
                        // SAFETY: the caller's contract, forwarded: `erase` erased
                        // this variant's payload from its declared type.
                        Self::#v(unsafe { ::acvus_extern::derive::materialize_payload::<#ty, __R>(
                            __rt, __payload, #tag,
                        ) })
                    }
                });
            }
            syn::Fields::Named(fields) => {
                if projected {
                    return Err(syn::Error::new_spanned(
                        &variant.fields,
                        "a struct variant has no projection: its payload is an object the enum \
                         writes and no Rust type names, so there is nothing to borrow it as. \
                         Give the variant one payload type deriving `#[derive(TyArg)] \
                         #[projection]`",
                    ));
                }
                let shape = ObjectShape::of(fields);
                let idents = &shape.idents;
                let ty = shape.written_poly_ty();
                let erase = shape.erase_bound();
                let materialize = shape.materialize(
                    quote! {
                        ::acvus_extern::derive::take_payload(__payload, #tag).into_value(__rt.holding())
                    },
                    quote! { Self::#v },
                );
                variant_tys.push(quote! {
                    (__i.intern(#tag), ::core::option::Option::Some(::std::boxed::Box::new(#ty)))
                });
                erase_arms.push(quote! {
                    #ident::#v { #(#idents),* } => (
                        #tag,
                        ::core::option::Option::Some(
                            // SAFETY: the object word is built here from
                            // fields the arm moved out, so no other holder
                            // owns it.
                            unsafe { ::acvus_extern::Owned::from_value(__rt.holding(), #erase) },
                        ),
                    )
                });
                materialize_arms.push(quote! { #matched => #materialize });
            }
        }
    }

    let ty = quote! {
        ::acvus_extern::PolyTy::Enum {
            name: __vars.derived::<#ident>(#qref),
            variants: [#(#variant_tys),*].into_iter().collect(),
            home: ::acvus_extern::Home::NONE,
        }
    };
    let erase = quote! {{
        let (__tag, __payload) = match self { #(#erase_arms,)* };
        ::acvus_extern::derive::variant::erase(__rt, __tag, __payload)
    }};
    let materialize = quote! {{
        // SAFETY: the caller's contract, and `erase` wrote this variant.
        let ::acvus_extern::derive::variant::Opened { at: __at, payload: __payload } = unsafe {
            ::acvus_extern::derive::variant::opened::<__R>(__rt, __value, #name, &[#(#tags),*])
        };
        match __at { #(#materialize_arms,)* }
    }};
    let borrowing = match projected {
        true => Borrowing::AsProjection,
        false => Borrowing::Whole,
    };
    let cross = cross_impl(
        ident,
        Crossing {
            ty: ty.clone(),
            erase,
            materialize,
            borrowing,
            returned: returned_as_one_value(),
        },
    );
    let projection = projected.then(|| enum_projection(ident, &borrowed, &erase_arms, &ty));
    Ok(quote! { #cross #projection })
}

struct BorrowedVariant<'a> {
    ident: &'a Ident,
    tag: String,
    payload: Option<&'a Type>,
}

/// `ERef<'a>`, `EArms<'a>` and `EMut<'a, Rt>`: the enum half of RFC-0050
/// rule 6.
///
/// The shape of the exclusive side is a decision. A Rust enum whose arms hold
/// the payload's `&mut` cannot also hold the whole variant to write both of
/// its words, so the two are two types: `EArms<'a>` is the enum of payload
/// borrows, and `EMut<'a, Rt>` is a struct over the variant that hands one
/// out per exclusive borrow of itself and rewrites `[tag, payload]` through
/// `set`. `EMut` names the runtime because a write erases a Rust value into
/// the runtime's words; `ERef` and `EArms` name none, which is what lets
/// `Borrowed` — a trait with no runtime parameter — carry them and an enum
/// nest inside another projection.
///
/// `EMut` holds the runtime it was built from, and `arms` and `set` both
/// read it there: a handler that declares no `&Rt` parameter has none to
/// hand them, and a second runtime passed to `set` could only disagree
/// with the one the words belong to.
fn enum_projection(
    owner: &Ident,
    variants: &[BorrowedVariant<'_>],
    erase_arms: &[proc_macro2::TokenStream],
    enum_ty: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let name = owner.to_string();
    let shared = format_ident!("{owner}Ref");
    let arms_ty = format_ident!("{owner}Arms");
    let exclusive = format_ident!("{owner}Mut");
    let doc_shared = format!("A shared projection of [`{owner}`] (RFC-0050 rule 6).");
    let doc_arms = format!("The payload [`{owner}Mut`] lends, one arm per variant.");
    let doc_exclusive = format!("An exclusive projection of [`{owner}`] (RFC-0050 rule 6).");

    let width = syn::Index::from(variants.len());
    let last = variants.len() - 1;
    let tags: Vec<&String> = variants.iter().map(|v| &v.tag).collect();
    let holders: Vec<Ident> = (0..variants.len())
        .map(|at| format_ident!("__at{at}"))
        .collect();
    let payload_tys: Vec<&Type> = variants.iter().filter_map(|v| v.payload).collect();
    let payload_holders: Vec<&Ident> = variants
        .iter()
        .zip(&holders)
        .filter(|(v, _)| v.payload.is_some())
        .map(|(_, holder)| holder)
        .collect();
    let payload_tags: Vec<&String> = variants
        .iter()
        .filter(|v| v.payload.is_some())
        .map(|v| &v.tag)
        .collect();

    let table_ty = quote! {
        ::acvus_extern::VariantAt<
            #width,
            (#(<#payload_tys as ::acvus_extern::Project<__R>>::Table,)*),
        >
    };
    let table_of = quote! {
        let [#(#holders),*] = ::acvus_extern::variant_tags_at(__at, [#(#tags),*]);
        ::acvus_extern::VariantAt {
            tags: [#(#holders.0),*],
            payloads: (#(
                <#payload_tys as ::acvus_extern::Project<__R>>::table(
                    ::acvus_extern::payload_at(#payload_holders.1, #payload_tags),
                ),
            )*),
        }
    };

    // The last arm absorbs the index, which `variant::arm_of` already proved
    // to be one of the tags.
    let matched = |at: usize| match at == last {
        true => quote! { _ },
        false => {
            let at = proc_macro2::Literal::usize_unsuffixed(at);
            quote! { #at }
        }
    };
    let mut declared_ref = Vec::new();
    let mut declared_arms = Vec::new();
    let mut read_arms = Vec::new();
    let mut write_arms = Vec::new();
    let mut end_arms = Vec::new();
    let mut payload_at = 0usize;
    for (at, variant) in variants.iter().enumerate() {
        let v = variant.ident;
        let matched = matched(at);
        match variant.payload {
            None => {
                declared_ref.push(quote! { #v });
                declared_arms.push(quote! { #v });
                read_arms.push(quote! { #matched => Self::#v });
                write_arms.push(quote! { #matched => Self::#v });
                end_arms.push(quote! { #matched => {} });
            }
            Some(ty) => {
                let held = syn::Index::from(payload_at);
                payload_at += 1;
                declared_ref.push(quote! {
                    #v(<#ty as ::acvus_extern::Borrowed>::Ref<'__a>)
                });
                declared_arms.push(quote! {
                    #v(<#ty as ::acvus_extern::Borrowed>::Mut<'__a>)
                });
                read_arms.push(quote! {
                    #matched => Self::#v({
                        // SAFETY: the caller's contract, and this variant's
                        // payload register holds what its own crossing wrote.
                        unsafe {
                            <#ty as ::acvus_extern::Project<__R>>::project(
                                __rt,
                                __payload,
                                &__table.payloads.#held,
                            )
                        }
                    })
                });
                write_arms.push(quote! {
                    #matched => Self::#v({
                        // SAFETY: as the shared projection's, exclusively.
                        unsafe {
                            <#ty as ::acvus_extern::Project<__R>>::project_mut(
                                __rt,
                                __payload,
                                &__table.payloads.#held,
                            )
                        }
                    })
                });
                end_arms.push(quote! {
                    // SAFETY: the caller's contract: the payload `over`
                    // handed this arm's `project_mut`.
                    #matched => unsafe {
                        <#ty as ::acvus_extern::Project<__R>>::loan_ended(
                            __rt,
                            __payload,
                            &__table.payloads.#held,
                        )
                    }
                });
            }
        }
    }

    quote! {
        #[doc = #doc_shared]
        pub enum #shared<'__a> {
            #(#declared_ref,)*
        }

        #[doc = #doc_arms]
        pub enum #arms_ty<'__a> {
            #(#declared_arms,)*
        }

        #[doc = #doc_exclusive]
        pub struct #exclusive<'__a, __R>
        where
            __R: ::acvus_extern::Runtime,
        {
            rt: &'__a __R,
            variant: &'__a mut ::acvus_extern::Variant<::acvus_extern::Owned<__R>>,
            table: #table_ty,
        }

        impl ::acvus_extern::Borrowed for #owner {
            type Ref<'__a> = #shared<'__a> where Self: '__a;
            type Mut<'__a> = #arms_ty<'__a> where Self: '__a;
        }

        impl ::acvus_extern::OwnStorage for #owner {}

        impl<'__q> ::acvus_extern::Projects for #shared<'__q> {
            type Owner = #owner;
            type Loan = ::acvus_extern::Shared;
        }

        impl<'__q> ::acvus_extern::Projects for #arms_ty<'__q> {
            type Owner = #owner;
            type Loan = ::acvus_extern::Mut;
        }

        impl<'__a> #shared<'__a> {
            /// # Safety
            /// `variant` holds what the owner's crossing wrote and is live
            /// for `'__a`.
            pub unsafe fn over<__R>(
                __rt: &'__a __R,
                __variant: &'__a ::acvus_extern::Variant<::acvus_extern::Owned<__R>>,
                __table: &#table_ty,
            ) -> Self
            where
                __R: ::acvus_extern::Runtime,
            {
                // SAFETY: the caller's contract: the crossing wrote the tag
                // register.
                let __tag = unsafe {
                    <__R as ::acvus_extern::Runtime>::tag_symbol(__rt, __variant.tag())
                };
                let __payload = __variant.payload();
                match ::acvus_extern::derive::variant::arm_of(__tag, &__table.tags, #name) {
                    #(#read_arms,)*
                }
            }
        }

        impl<'__a> #arms_ty<'__a> {
            /// # Safety
            /// As the shared projection's `over`, and `variant` is
            /// exclusively named for `'__a`.
            pub unsafe fn over<__R>(
                __rt: &'__a __R,
                __variant: &'__a mut ::acvus_extern::Variant<::acvus_extern::Owned<__R>>,
                __table: &#table_ty,
            ) -> Self
            where
                __R: ::acvus_extern::Runtime,
            {
                // SAFETY: as the shared projection's.
                let __tag = unsafe {
                    <__R as ::acvus_extern::Runtime>::tag_symbol(__rt, __variant.tag())
                };
                let __at = ::acvus_extern::derive::variant::arm_of(__tag, &__table.tags, #name);
                // SAFETY: the payload word is only handed to its arm's
                // `project_mut`, whose contract is what is written through it.
                let __payload = unsafe {
                    __variant.payload_mut().value_mut(::acvus_extern::Holding::new())
                };
                match __at {
                    #(#write_arms,)*
                }
            }

            /// The arm `over` lent has ended: the payload of the variant
            /// the tag now names goes to its own crossing's
            /// `Project::loan_ended`, which hands a storage lent in place to
            /// `Runtime::loan_ended`. A variant `set` wrote since is read by
            /// its own tag, and its words are the crossing's own.
            ///
            /// # Safety
            /// As `over`'s, over the variant `over` projected, and no borrow
            /// an arm handed out is live.
            pub unsafe fn ended<__R>(
                __rt: &__R,
                __variant: &mut ::acvus_extern::Variant<::acvus_extern::Owned<__R>>,
                __table: &#table_ty,
            ) where
                __R: ::acvus_extern::Runtime,
            {
                // SAFETY: as `over`'s.
                let __tag = unsafe {
                    <__R as ::acvus_extern::Runtime>::tag_symbol(__rt, __variant.tag())
                };
                let __at = ::acvus_extern::derive::variant::arm_of(__tag, &__table.tags, #name);
                // SAFETY: as `over`'s: the payload word goes only to its
                // arm's `loan_ended`.
                let __payload = unsafe {
                    __variant.payload_mut().value_mut(::acvus_extern::Holding::new())
                };
                match __at {
                    #(#end_arms,)*
                }
            }
        }

        impl<'__a, __R> #exclusive<'__a, __R>
        where
            __R: ::acvus_extern::Runtime,
        {
            /// # Safety
            /// As the shared projection's `over`, and `variant` is
            /// exclusively named for `'__a`.
            pub unsafe fn over(
                __rt: &'__a __R,
                __variant: &'__a mut ::acvus_extern::Variant<::acvus_extern::Owned<__R>>,
                __table: &#table_ty,
            ) -> Self {
                Self { rt: __rt, variant: __variant, table: ::core::clone::Clone::clone(__table) }
            }

            pub fn arms(&mut self) -> #arms_ty<'_> {
                // SAFETY: `over`'s contract, reborrowed for this borrow of
                // `self` and no longer.
                unsafe { #arms_ty::over(self.rt, self.variant, &self.table) }
            }

            pub fn set(&mut self, __value: #owner) {
                // SAFETY: `over`'s contract: the variant holds what the
                // owner's crossing wrote, and the value is erased at the
                // owner's own type.
                let __rt = unsafe { ::acvus_extern::Crossing::new(self.rt) };
                let (__tag, __payload) = match __value { #(#erase_arms,)* };
                *self.variant = ::acvus_extern::derive::variant::words(__rt, __tag, __payload);
            }
        }

        impl<__R> ::acvus_extern::Project<__R> for #owner
        where
            __R: ::acvus_extern::Runtime,
        {
            type Table = #table_ty;

            const LENDS_A_WORD: bool =
                false #(|| <#payload_tys as ::acvus_extern::Project<__R>>::LENDS_A_WORD)*;

            fn table(__at: ::acvus_extern::ArgAt<'_>) -> Self::Table {
                #table_of
            }

            unsafe fn project<'__a>(
                __rt: &'__a __R,
                __value: &'__a <__R as ::acvus_extern::Runtime>::Value,
                __table: &Self::Table,
            ) -> #shared<'__a> {
                // SAFETY: the caller's contract: a nested enum payload holds
                // its own variant.
                unsafe {
                    #shared::over(__rt, ::acvus_extern::variant::<::acvus_extern::Nested, ::acvus_extern::Shared, __R>(__rt, __value), __table)
                }
            }

            unsafe fn project_mut<'__a>(
                __rt: &'__a __R,
                __value: &'__a mut <__R as ::acvus_extern::Runtime>::Value,
                __table: &Self::Table,
            ) -> #arms_ty<'__a> {
                // SAFETY: as `project`, with the caller's exclusive loan.
                unsafe {
                    #arms_ty::over(__rt, ::acvus_extern::variant::<::acvus_extern::Nested, ::acvus_extern::Mut, __R>(__rt, __value), __table)
                }
            }

            unsafe fn loan_ended(
                __rt: &__R,
                __value: &mut <__R as ::acvus_extern::Runtime>::Value,
                __table: &Self::Table,
            ) {
                if !<Self as ::acvus_extern::Project<__R>>::LENDS_A_WORD {
                    return;
                }
                // SAFETY: the caller's contract: the variant `project_mut`
                // projected.
                unsafe {
                    #arms_ty::ended(__rt, ::acvus_extern::variant::<::acvus_extern::Nested, ::acvus_extern::Mut, __R>(__rt, __value), __table)
                }
            }
        }

        impl<'__q, __R> ::acvus_extern::Param<__R> for #shared<'__q>
        where
            __R: ::acvus_extern::Runtime,
        {
            type Marker = ::acvus_extern::ByProjection<#shared<'static>>;
            type At<'__a> = #shared<'__a>;
        }

        impl<'__q, __R> ::acvus_extern::Param<__R> for #exclusive<'__q, __R>
        where
            __R: ::acvus_extern::Runtime,
        {
            type Marker = ::acvus_extern::ByProjection<#exclusive<'static, __R>>;
            type At<'__a> = #exclusive<'__a, __R>;
        }

        // SAFETY: a projection is at its own lifetime, and borrows only parts
        // of the variant the caller lent at it.
        unsafe impl<'__s> ::acvus_extern::Within<'__s> for #shared<'__s> {}

        impl<'__a, __R> ::acvus_extern::Projected<'__a, __R> for #shared<'__a>
        where
            __R: ::acvus_extern::Runtime,
        {
            type Loan = ::acvus_extern::Shared;
            type Table = #table_ty;

            /// A shared projection lends nothing exclusively.
            const LENDS_A_WORD: bool = false;

            fn table(__at: ::acvus_extern::ArgAt<'_>) -> Self::Table {
                #table_of
            }

            unsafe fn of(
                __rt: &'__a __R,
                __reference: &'__a <__R as ::acvus_extern::Runtime>::Value,
                __table: &Self::Table,
            ) -> #shared<'__a> {
                // SAFETY: the caller's contract: a live variant storage.
                unsafe {
                    #shared::over(__rt, ::acvus_extern::variant::<::acvus_extern::Lent, ::acvus_extern::Shared, __R>(__rt, __reference), __table)
                }
            }

            unsafe fn loan_ended(
                _: &__R,
                _: &<__R as ::acvus_extern::Runtime>::Value,
                _: &Self::Table,
            ) {
            }
        }

        // SAFETY: as the shared projection's.
        unsafe impl<'__s, __R> ::acvus_extern::Within<'__s> for #exclusive<'__s, __R>
        where
            __R: ::acvus_extern::Runtime,
        {
        }

        impl<'__a, __R> ::acvus_extern::Projected<'__a, __R> for #exclusive<'__a, __R>
        where
            __R: ::acvus_extern::Runtime,
        {
            type Loan = ::acvus_extern::Mut;
            type Table = #table_ty;

            const LENDS_A_WORD: bool = <#owner as ::acvus_extern::Project<__R>>::LENDS_A_WORD;

            fn table(__at: ::acvus_extern::ArgAt<'_>) -> Self::Table {
                #table_of
            }

            unsafe fn of(
                __rt: &'__a __R,
                __reference: &'__a <__R as ::acvus_extern::Runtime>::Value,
                __table: &Self::Table,
            ) -> #exclusive<'__a, __R> {
                // SAFETY: as the shared projection's, exclusively.
                unsafe {
                    #exclusive::over(
                        __rt,
                        ::acvus_extern::variant::<::acvus_extern::Lent, ::acvus_extern::Mut, __R>(__rt, __reference),
                        __table,
                    )
                }
            }

            unsafe fn loan_ended(
                __rt: &__R,
                __reference: &<__R as ::acvus_extern::Runtime>::Value,
                __table: &Self::Table,
            ) {
                if !<Self as ::acvus_extern::Projected<'__a, __R>>::LENDS_A_WORD {
                    return;
                }
                // SAFETY: the caller's contract: the variant storage `of`
                // projected, which neither the projection nor an arm it
                // handed out names any longer.
                unsafe {
                    #arms_ty::ended(
                        __rt,
                        ::acvus_extern::variant::<::acvus_extern::Lent, ::acvus_extern::Mut, __R>(__rt, __reference),
                        __table,
                    )
                }
            }
        }

        impl ::acvus_extern::Var<::acvus_extern::kind::Type> for #shared<'static> {}

        // SAFETY: a projection is no box and names no parameter.
        unsafe impl ::acvus_extern::Canonical<::acvus_extern::kind::Type> for #shared<'static> {
            type Canon = Self;
        }

        impl ::acvus_extern::TyArg for #shared<'static> {
            fn poly_ty(
                __i: &::acvus_extern::Interner,
                __vars: &::acvus_extern::PolyVars,
            ) -> ::acvus_extern::PolyTy {
                ::acvus_extern::PolyTy::Ref(
                    ::acvus_extern::Mutability::Shared,
                    ::std::boxed::Box::new(::acvus_extern::TypeArg::uniform(#enum_ty)),
                )
            }
        }

        impl<__R> ::acvus_extern::Var<::acvus_extern::kind::Type> for #exclusive<'static, __R> where
            __R: ::acvus_extern::Runtime
        {
        }

        // SAFETY: as the shared projection's.
        unsafe impl<__R> ::acvus_extern::Canonical<::acvus_extern::kind::Type>
            for #exclusive<'static, __R>
        where
            __R: ::acvus_extern::Runtime,
        {
            type Canon = Self;
        }

        impl<__R> ::acvus_extern::TyArg for #exclusive<'static, __R>
        where
            __R: ::acvus_extern::Runtime,
        {
            fn poly_ty(
                __i: &::acvus_extern::Interner,
                __vars: &::acvus_extern::PolyVars,
            ) -> ::acvus_extern::PolyTy {
                ::acvus_extern::PolyTy::Ref(
                    ::acvus_extern::Mutability::Mut,
                    ::std::boxed::Box::new(::acvus_extern::TypeArg::uniform(#enum_ty)),
                )
            }
        }
    }
}

// -- extern_registry! ------------------------------------------------

/// `extern_registry! { ns: "core", types: [Vec<_>], signatures: [eq],
/// fns: [len, chat(client)] }` (RFC-0021).
struct RegistryInput {
    ns: LitStr,
    types: Vec<Type>,
    signatures: Vec<Path>,
    fns: Vec<RegistryFn>,
}

/// One listed function and the state values its declaration takes.
struct RegistryFn {
    path: Path,
    state: Vec<syn::Expr>,
}

impl Parse for RegistryFn {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let path: Path = input.parse()?;
        let state = if input.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in input);
            content
                .parse_terminated(syn::Expr::parse, Token![,])?
                .into_iter()
                .collect()
        } else {
            Vec::new()
        };
        Ok(Self { path, state })
    }
}

impl Parse for RegistryInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut ns = None;
        let mut types = Vec::new();
        let mut signatures = Vec::new();
        let mut fns = Vec::new();
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            if key == "ns" {
                ns = Some(input.parse::<LitStr>()?);
            } else {
                let content;
                syn::bracketed!(content in input);
                if key == "types" {
                    let list: Punctuated<Type, Token![,]> =
                        content.parse_terminated(Type::parse, Token![,])?;
                    types.extend(list);
                } else if key == "signatures" {
                    let list: Punctuated<Path, Token![,]> =
                        content.parse_terminated(Path::parse, Token![,])?;
                    signatures.extend(list);
                } else if key == "fns" {
                    let list: Punctuated<RegistryFn, Token![,]> =
                        content.parse_terminated(RegistryFn::parse, Token![,])?;
                    fns.extend(list);
                } else {
                    return Err(syn::Error::new(
                        key.span(),
                        "expected `ns`, `types`, `signatures`, or `fns`",
                    ));
                }
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        let Some(ns) = ns else {
            return Err(syn::Error::new(
                Span::call_site(),
                "a registry declares its namespace: `ns: \"...\"`",
            ));
        };
        Ok(Self {
            ns,
            types,
            signatures,
            fns,
        })
    }
}

#[proc_macro]
pub fn extern_registry(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as RegistryInput);
    let ns = input.ns;
    let types: Vec<Type> = input.types.iter().map(subst::infer_to_unit).collect();
    let signatures = input.signatures;
    let fns = input.fns.into_iter().map(|f| {
        let mut path = f.path;
        let last = path.segments.last_mut().expect("path has a segment");
        last.ident = format_ident!("__extern_fn_{}", last.ident);
        let state = f.state;
        quote! { #path(__i, __ns, #(#state),*) }
    });
    quote! {
        ::acvus_extern::Registry::new(move |__i: &::acvus_extern::Interner| {
            let __ns: ::core::option::Option<&str> = ::core::option::Option::Some(#ns);
            let mut __contribution = ::acvus_extern::Contribution::of(::acvus_extern::Manifest {
                types: vec![#(::acvus_extern::DeclaredType::of::<#types>(__i)),*],
                signatures: vec![#(
                    <#signatures as ::acvus_extern::SharedSignature>::signature_decl(__i)
                ),*],
                fns: ::std::vec::Vec::new(),
            });
            #(
                if let ::core::option::Option::Some(__hooks) =
                    <#types as ::acvus_extern::ExternTypeDecl>::space()
                {
                    __contribution.register_space(
                        <#types as ::acvus_extern::ExternTypeDecl>::type_decl(__i).qref,
                        __hooks,
                    );
                }
            )*
            for __f in ::std::vec::Vec::<::std::vec::Vec<::acvus_extern::ExternFn<_>>>::from([#(#fns),*])
                .into_iter()
                .flatten()
            {
                __contribution.declare(__f);
            }
            __contribution
        })
    }
    .into()
}

// -- extern_signature! -----------------------------------------------

/// `extern_signature! { ns: "core", fn eq<T>(a: &T, b: &T) -> bool
/// where T: Var<kind::Type>; }` declares a shared signature (RFC-0019) and a
/// marker type named after it. `extern_signature! { ns: "q", effect = E,
/// fn drain<S, E>(it: S) -> i64 where S: Var<kind::Type>,
/// E: Var<kind::Effect>; }` declares one whose call effect is `E`.
struct SignatureInput {
    ns: LitStr,
    effect: Option<Ident>,
    sig: syn::Signature,
}

impl Parse for SignatureInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let key: Ident = input.parse()?;
        if key != "ns" {
            return Err(syn::Error::new(key.span(), "expected `ns`"));
        }
        input.parse::<Token![:]>()?;
        let ns: LitStr = input.parse()?;
        input.parse::<Token![,]>()?;
        let effect = if input.peek(Ident) && input.peek2(Token![=]) {
            let key: Ident = input.parse()?;
            if key != "effect" {
                return Err(syn::Error::new(key.span(), "expected `effect`"));
            }
            input.parse::<Token![=]>()?;
            let var: Ident = input.parse()?;
            input.parse::<Token![,]>()?;
            Some(var)
        } else {
            None
        };
        let sig: syn::Signature = input.parse()?;
        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
        }
        Ok(Self { ns, effect, sig })
    }
}

#[proc_macro]
pub fn extern_signature(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as SignatureInput);
    match generate_signature(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn generate_signature(input: SignatureInput) -> syn::Result<proc_macro2::TokenStream> {
    let mut sig = input.sig;
    let ident = sig.ident.clone();
    let vars = Vars::from_generics(&sig.generics)?;
    if let Some(suspending) = vars.suspending() {
        return Err(syn::Error::new(
            suspending.span(),
            "a signature does not bound its effect variables: Suspends is written on an \
             #[extern_fn] declaration",
        ));
    }
    let Signature {
        params: rust_params,
        ..
    } = parse_params(&mut sig, None)?;
    let mut params = Vec::new();
    for p in rust_params {
        match p {
            RustParam::Acvus(a) => params.push(a),
            RustParam::State(st) => {
                return Err(syn::Error::new(
                    st.ident.span(),
                    "a signature declares no state",
                ));
            }
            RustParam::Required(RequiredParam { signature, .. })
            | RustParam::Owning {
                required: RequiredParam { signature, .. },
                ..
            } => {
                return Err(syn::Error::new_spanned(
                    signature,
                    "a signature requires no instance of its own",
                ));
            }
            RustParam::Args(args) => {
                return Err(syn::Error::new_spanned(
                    &args.written,
                    "a signature declares no `Args`: its instances are called through their mono \
                     glue, which builds none (RFC-0067 rule 8, RFC-0097 rule 1)",
                ));
            }
            RustParam::Output(output) => {
                return Err(syn::Error::new_spanned(
                    &output.written,
                    "a signature declares no `Output`: its instances return at the signature's \
                     own result type, which no call site settles (RFC-0097 rule 3)",
                ));
            }
        }
    }
    let ret = parse_return(&sig.output);
    let name = ident.to_string();
    let qref = qref_expr_in(Some(&input.ns.value()), &name);
    let param_terms = params.iter().map(|p| {
        let pname = &p.name;
        let comp_ty = p.mode.acvus_ty(
            &vars.to_compile_time_instance(&p.ty, None),
            &quote! { ::acvus_extern::TypesOnly },
        );
        quote! {
            ::acvus_extern::ParamTerm::<::acvus_extern::Poly>::new(
                __i.intern(#pname),
                <#comp_ty as ::acvus_extern::TyArg>::poly_ty(__i, &__vars),
            )
        }
    });
    let comp_ret = vars.to_compile_time_instance(&ret, None);
    let bounds = vars.bound_exprs();
    let chosen = vars.chosen_numbers();
    let fresh_vars = vars.fresh_vars_expr();
    let effect = match &input.effect {
        None => quote! { ::acvus_extern::EffectTerm::Known(::acvus_extern::Effect::PURE) },
        Some(e) => match vars.lookup(e) {
            Some((VarKind::Effect, k)) => quote! { __vars.effects[#k].clone() },
            _ => {
                return Err(syn::Error::new(
                    e.span(),
                    "a signature's effect is a `Var<kind::Effect>` parameter of its own \
                     generics; a signature that declares no effect is pure",
                ));
            }
        },
    };
    // The marker names the signature's own variables, so that a handler
    // writes the requirement at its own: `InstanceOf<sig::eq<T, Rt>>`. Every
    // parameter is defaulted, so the bare `eq` an `instance_of` attribute
    // names still resolves.
    let declared: Vec<&Ident> = vars.idents();
    let runtime: Ident = match vars.runtime_ident() {
        Some(rt) => rt.clone(),
        None => format_ident!("__Rt"),
    };
    let marker_params: Vec<Ident> = match vars.runtime_ident() {
        Some(_) => declared.iter().map(|&i| i.clone()).collect(),
        None => declared
            .iter()
            .map(|&i| i.clone())
            .chain(std::iter::once(runtime.clone()))
            .collect(),
    };
    let requirement_impl = requirement_of(&ident, &vars, &marker_params);
    // A signature with a two-word parameter has no mono glue and no
    // `Signature` impl, so it writes no crossing module either: the module
    // names the signature's own types, and a position that is not one
    // value has no place in it.
    let crossing = match params.iter().any(|p| p.mode.is_two_words()) {
        true => proc_macro2::TokenStream::new(),
        false => signature_module(&ident, &vars, &runtime, &params, &ret),
    };
    let signature_impl = signature_call(
        &input.ns,
        &ident,
        &vars,
        &marker_params,
        &runtime,
        &params,
        &ret,
    );
    Ok(quote! {
        #[allow(non_camel_case_types)]
        pub struct #ident<#(#marker_params = (),)*>(
            ::core::marker::PhantomData<fn() -> (#(#marker_params,)*)>,
        );

        impl<#(#marker_params,)*> ::acvus_extern::SharedSignature for #ident<#(#marker_params,)*> {
            fn qref(__i: &::acvus_extern::Interner) -> ::acvus_extern::QualifiedRef {
                #qref
            }
            fn signature_decl(__i: &::acvus_extern::Interner) -> ::acvus_extern::SignatureDecl {
                let __vars = #fresh_vars;
                ::acvus_extern::SignatureDecl {
                    qref: #qref,
                    ty: ::acvus_extern::PolyTy::Fn {
                        params: vec![#(#param_terms),*],
                        ret: Box::new(<#comp_ret as ::acvus_extern::TyArg>::poly_ty(__i, &__vars)),
                        captures: vec![],
                        effect: #effect,
                        flows: ::acvus_extern::Flows::Every.into(),
                    },
                    bounds: vec![#(#bounds),*],
                    chosen: vec![#(#chosen),*],
                    names: __vars.names(),
                }
            }
        }

        #crossing

        #requirement_impl

        #signature_impl
    })
}

/// The `RequirementOf` impl of a signature marker: the signature's own type
/// with variable `k` of each kind replaced by what the marker's parameter
/// standing at that variable names. The parameters are the signature's
/// variables in declaration order, so each kind's list is indexed by the
/// variable's number within that kind.
fn requirement_of(ident: &Ident, vars: &Vars, marker_params: &[Ident]) -> proc_macro2::TokenStream {
    let mut tys = Vec::new();
    let mut effects = Vec::new();
    let mut lens = Vec::new();
    let mut identities = Vec::new();
    let mut bounds = Vec::new();
    for param in marker_params {
        let kind = vars.lookup(param).map(|(kind, _)| kind);
        match kind {
            Some(VarKind::Ty) => {
                bounds.push(quote! { #param: ::acvus_extern::TyArg });
                tys.push(quote! { <#param as ::acvus_extern::TyArg>::poly_ty(__i, __vars) });
            }
            Some(VarKind::Effect) => {
                bounds.push(quote! { #param: ::acvus_extern::Term<::acvus_extern::kind::Effect> });
                effects.push(quote! {
                    <#param as ::acvus_extern::Term<::acvus_extern::kind::Effect>>::poly(__vars)
                });
            }
            Some(VarKind::Len) => {
                bounds.push(quote! { #param: ::acvus_extern::Term<::acvus_extern::kind::Length> });
                lens.push(quote! {
                    <#param as ::acvus_extern::Term<::acvus_extern::kind::Length>>::poly(__vars)
                });
            }
            Some(VarKind::Identity) => {
                bounds
                    .push(quote! { #param: ::acvus_extern::Term<::acvus_extern::kind::Identity> });
                identities.push(quote! {
                    <#param as ::acvus_extern::Term<::acvus_extern::kind::Identity>>::poly(__vars)
                });
            }
            Some(VarKind::Runtime) | None => {
                bounds.push(quote! { #param: ::acvus_extern::Runtime })
            }
        }
    }
    quote! {
        impl<#(#marker_params,)*> ::acvus_extern::RequirementOf for #ident<#(#marker_params,)*>
        where
            #(#bounds,)*
        {
            fn pattern(
                __i: &::acvus_extern::Interner,
                __vars: &::acvus_extern::PolyVars,
            ) -> ::acvus_extern::PolyTy {
                let __tys: Vec<::acvus_extern::PolyTy> = vec![#(#tys),*];
                let __effects: Vec<::acvus_extern::EffectTerm<::acvus_extern::Poly>> =
                    vec![#(#effects),*];
                let __lens: Vec<::acvus_extern::LenTerm<::acvus_extern::Poly>> = vec![#(#lens),*];
                let __identities: Vec<::acvus_extern::IdentityTerm<::acvus_extern::Poly>> =
                    vec![#(#identities),*];
                // The signature's own `ρ`s are its slots'; beside the
                // requiring declaration's type, each is a slot of its own
                // there too, so it is renamed to one the declaration has
                // not given out.
                let mut __reprs: ::std::collections::HashMap<u32, u32> =
                    ::std::collections::HashMap::new();
                <Self as ::acvus_extern::SharedSignature>::signature_decl(__i).ty.map(
                    &mut |__k: u32| __tys[__k as usize].clone(),
                    &mut |__k: u32| __identities[__k as usize],
                    &mut |__k: u32| __effects[__k as usize].clone(),
                    &mut |__k: u32| __lens[__k as usize],
                    &mut |__r: u32| {
                        ::acvus_extern::Repr::Var(
                            *__reprs.entry(__r).or_insert_with(|| __vars.fresh_repr()),
                        )
                    },
                    &mut ::acvus_extern::no_flow_var,
                )
            }
        }
    }
}

// -- shared helpers --------------------------------------------------

fn bound_ident(bound: &TypeParamBound) -> Option<&Ident> {
    match bound {
        TypeParamBound::Trait(t) => t.path.segments.last().map(|s| &s.ident),
        _ => None,
    }
}

fn span_of(param: &GenericParam) -> Span {
    match param {
        GenericParam::Type(t) => t.ident.span(),
        GenericParam::Lifetime(l) => l.lifetime.span(),
        GenericParam::Const(c) => c.ident.span(),
    }
}

/// The first parameter of a shared signature, as the `Signature` impl
/// spells it (RFC-0070 rule 4).
/// How a signature's first parameter takes its receiver, as
/// `Signature::Mode` names it.
struct ReceiverMode(proc_macro2::TokenStream);

impl ReceiverMode {
    /// `None` where the mode has no receiver form: a `str` view and a
    /// projection name storage the caller lent, and a bounded variable is
    /// not that storage.
    fn of(mode: Mode) -> Option<Self> {
        match mode {
            Mode::Borrow => Some(Self(quote! { ::acvus_extern::Shared })),
            Mode::BorrowMut => Some(Self(quote! { ::acvus_extern::Mut })),
            Mode::Value => Some(Self(quote! { ::acvus_extern::Moved })),
            Mode::Str | Mode::Slice | Mode::SliceMut | Mode::Projection => None,
        }
    }
}

/// Whether `ty` mentions `ident`.
fn mentions(ty: &Type, ident: &Ident) -> bool {
    let found = std::cell::Cell::new(false);
    subst::substitute(ty, &|at| {
        if at == ident {
            found.set(true);
        }
        None
    });
    found.get()
}

/// One parameter of a shared signature after the first, as the crossing
/// sees it: a position standing at the signature's own variable crosses as
/// the caller's own value, one type per runtime, and the instance's own
/// handler type is restored on the far side; every other position crosses
/// as the signature wrote it (RFC-0067 rule 6).
#[derive(Clone, Copy, PartialEq, Eq)]
enum RestAt {
    VariableShared,
    VariableExclusive,
    VariableValue,
    Itself(Mode),
}

impl RestAt {
    fn of(p: &ExternParam, vars: &Vars) -> Self {
        let at_variable = vars.type_vars().any(|v| mentions(&p.ty, v));
        match (p.mode, at_variable) {
            (Mode::Borrow, true) => Self::VariableShared,
            (Mode::BorrowMut, true) => Self::VariableExclusive,
            (Mode::Value, true) => Self::VariableValue,
            (mode, _) => Self::Itself(mode),
        }
    }

    /// The type this position has in the signature's rest run, over the
    /// type `ty` the signature wrote there.
    fn crossed(self, ty: &Type, runtime: &Ident) -> proc_macro2::TokenStream {
        let value = quote! { <#runtime as ::acvus_extern::Runtime>::Value };
        match self {
            Self::VariableShared => quote! { &'__a #value },
            Self::VariableExclusive => quote! { &'__a mut #value },
            Self::VariableValue => quote! { ::acvus_extern::Owned<#runtime> },
            Self::Itself(Mode::Borrow) => quote! { &'__a #ty },
            Self::Itself(Mode::BorrowMut) => quote! { &'__a mut #ty },
            Self::Itself(Mode::Str) => quote! { &'__a str },
            Self::Itself(_) => quote! { #ty },
        }
    }

    /// The type the instance's own handler takes at this position: `handler`,
    /// which the handler's own parameter fixes, at a variable.
    fn restored(self, handler: &Ident, ty: &Type) -> proc_macro2::TokenStream {
        match self {
            Self::VariableShared | Self::VariableExclusive | Self::VariableValue => {
                quote! { #handler }
            }
            Self::Itself(Mode::Borrow) => quote! { &'__b #ty },
            Self::Itself(Mode::BorrowMut) => quote! { &'__b mut #ty },
            Self::Itself(Mode::Str) => quote! { &'__b str },
            Self::Itself(_) => quote! { #ty },
        }
    }

    fn at_variable(self) -> bool {
        !matches!(self, Self::Itself(_))
    }

    /// The type this position has in the rest a requirer passes: a position
    /// at a variable borrowed as the signature takes it, or passed as
    /// `Passed::As` names it, and every other position as `crossed` spells
    /// it. A lifetime the signature wrote is `'static` here, as in every
    /// bound the impl states.
    fn required(self, ty: &Type, runtime: &Ident) -> proc_macro2::TokenStream {
        let ty = at_static(ty);
        match self {
            Self::VariableShared => quote! { &'__a #ty },
            Self::VariableExclusive => quote! { &'__a mut #ty },
            Self::VariableValue => {
                quote! { <#ty as ::acvus_extern::Passed<'__a, #runtime>>::As }
            }
            Self::Itself(_) => self.crossed(&ty, runtime),
        }
    }

    /// What `required`'s type must be for the glue to cross it into
    /// `crossed`'s: a borrowed variable names storage holding one of the
    /// runtime's values, as a receiver does (RFC-0068 rule 2), and a
    /// position taken by value is one a closure could be passed.
    fn required_bound(self, ty: &Type, runtime: &Ident) -> Option<proc_macro2::TokenStream> {
        let ty = at_static(ty);
        let value = quote! { <#runtime as ::acvus_extern::Runtime>::Value };
        match self {
            Self::VariableShared => Some(quote! { #ty: ::core::ops::Deref<Target = #value> }),
            Self::VariableExclusive => {
                Some(quote! { #ty: ::core::ops::DerefMut<Target = #value> })
            }
            Self::VariableValue => {
                Some(quote! { #ty: for<'__p> ::acvus_extern::Passed<'__p, #runtime> })
            }
            Self::Itself(_) => None,
        }
    }

    /// `arg`, of `required`'s type, crossed into `crossed`'s by the glue.
    fn cross_required(self, ty: &Type, runtime: &Ident, arg: &Ident) -> proc_macro2::TokenStream {
        let ty = at_static(ty);
        match self {
            Self::VariableShared => quote! { &**#arg },
            Self::VariableExclusive => quote! { &mut **#arg },
            Self::VariableValue => quote! {
                // SAFETY: `Passed::cross` erased the value it consumed, so no
                // other holder owns the word.
                unsafe {
                    ::acvus_extern::Owned::from_value(
                        __rt.holding(),
                        <#ty as ::acvus_extern::Passed<'__a, #runtime>>::cross(__rt, #arg),
                    )
                }
            },
            Self::Itself(_) => quote! { #arg },
        }
    }

    /// Whether the position is one a `Signature` impl can name. A borrow at
    /// the signature's own variable crosses as the caller's value; a borrow
    /// of a concrete type crosses as the Rust reference the requirer holds,
    /// which is the type the instance's handler takes there (RFC-0067 rule
    /// 6: `core::display`'s `out: &mut String`). A pair of words or a
    /// projection has no one-word form a requiring handler could hold
    /// (RFC-0067 rule 8).
    fn is_signature_shaped(self) -> bool {
        match self {
            Self::Itself(Mode::Str | Mode::Slice | Mode::SliceMut | Mode::Projection) => false,
            Self::VariableShared
            | Self::VariableExclusive
            | Self::VariableValue
            | Self::Itself(_) => true,
        }
    }
}

/// The crossing of one shared signature, beside the signature itself: the
/// types `Signature` projects to and every mono glue is written at, so that
/// the two are one definition and `call_now`'s word cannot be a `fn` of
/// another shape.
fn signature_module(
    ident: &Ident,
    vars: &Vars,
    runtime: &Ident,
    params: &[ExternParam],
    ret: &Type,
) -> proc_macro2::TokenStream {
    let module = signature_module_ident(ident);
    // A result standing at one of the signature's type variables is the
    // instance's own Rust type on one side and the requirer's on the other,
    // so it crosses as the runtime's value: the glue erases what its
    // handler returned and the requirer materializes at its own `Ret`.
    let returned = match RetShape::of(ret, vars) {
        RetShape::Concrete => quote! {
            pub type Ret<#runtime> = <#ret as ::acvus_extern::RestRun<#runtime>>::Run;

            // SAFETY: a concrete result crosses as itself; the capability
            // crosses nothing and is not kept.
            unsafe impl<#runtime> Returned<#runtime> for #ret
            where
                #runtime: ::acvus_extern::Runtime,
            {
                #[inline(always)]
                fn cross(_: ::acvus_extern::Crossing<'_, #runtime>, __r: Self) -> Ret<#runtime> {
                    __r
                }
            }
        },
        RetShape::OptionOfVariable(_) => quote! {
            pub type Ret<#runtime> =
                ::core::option::Option<<#runtime as ::acvus_extern::Runtime>::Value>;

            // SAFETY: the word is `__T`'s own `erase` of the present value,
            // at the `__T` the checker settled; nothing else crosses and the
            // capability is not kept.
            unsafe impl<#runtime, __T> Returned<#runtime> for ::core::option::Option<__T>
            where
                #runtime: ::acvus_extern::Runtime,
                __T: ::acvus_extern::OneValue<#runtime>,
            {
                #[inline(always)]
                fn cross(__rt: ::acvus_extern::Crossing<'_, #runtime>, __r: Self) -> Ret<#runtime> {
                    __r.map(|__v| <__T as ::acvus_extern::OneValue<#runtime>>::erase(__v, __rt))
                }
            }
        },
        RetShape::Whole => quote! {
            pub type Ret<#runtime> = <#runtime as ::acvus_extern::Runtime>::Value;

            // SAFETY: the word is `__T`'s own `erase` of the result, at the
            // `__T` the checker settled; nothing else crosses and the
            // capability is not kept.
            unsafe impl<#runtime, __T> Returned<#runtime> for __T
            where
                #runtime: ::acvus_extern::Runtime,
                __T: ::acvus_extern::OneValue<#runtime>,
            {
                #[inline(always)]
                fn cross(__rt: ::acvus_extern::Crossing<'_, #runtime>, __r: Self) -> Ret<#runtime> {
                    <__T as ::acvus_extern::OneValue<#runtime>>::erase(__r, __rt)
                }
            }
        },
    };
    let tail: Vec<&ExternParam> = params.iter().skip(1).collect();
    let rest: Vec<RestAt> = tail.iter().map(|p| RestAt::of(p, vars)).collect();
    let tys: Vec<&Type> = tail.iter().map(|p| &p.ty).collect();
    let markers: Vec<Ident> = (0..rest.len()).map(|at| format_ident!("__M{at}")).collect();
    let handlers: Vec<Ident> = (0..rest.len()).map(|at| format_ident!("__D{at}")).collect();
    let variable_handlers: Vec<&Ident> = rest
        .iter()
        .zip(&handlers)
        .filter(|(at, _)| at.at_variable())
        .map(|(_, handler)| handler)
        .collect();
    let args: Vec<Ident> = (0..rest.len()).map(|at| format_ident!("__x{at}")).collect();
    let lent: Vec<Ident> = rest
        .iter()
        .enumerate()
        .filter(|(_, at)| matches!(at, RestAt::VariableShared | RestAt::VariableExclusive))
        .map(|(at, _)| format_ident!("__at{at}"))
        .collect();
    // Each lent position's slot: the `Lending` its borrow is read through,
    // at the loan the position takes, which the glue drops after the body
    // and so ends the loan. A signature that lends nothing has no slot and
    // still names the lifetime.
    let lent_slots: Vec<proc_macro2::TokenStream> = rest
        .iter()
        .zip(&markers)
        .zip(&handlers)
        .filter_map(|((at, marker), handler)| match at {
            RestAt::VariableShared => Some(quote! {
                ::core::option::Option<::acvus_extern::Lending<'__r, ::acvus_extern::Shared, #runtime>>
            }),
            RestAt::VariableExclusive => Some(quote! {
                ::core::option::Option<::acvus_extern::Lending<
                    '__r,
                    ::acvus_extern::Mut,
                    #runtime,
                    <#handler as ::acvus_extern::RestoreExclusive<'__b, #marker, #runtime>>::Ending,
                >>
            }),
            RestAt::VariableValue | RestAt::Itself(_) => None,
        })
        .collect();
    let (lent_ty, lent_pattern) = match lent_slots.is_empty() {
        true => (
            quote! { ::core::marker::PhantomData<&'__r #runtime> },
            quote! { _ },
        ),
        false => (quote! { (#(#lent_slots,)*) }, quote! { (#(#lent,)*) }),
    };
    let crossed = rest
        .iter()
        .zip(&tys)
        .map(|(at, ty)| at.crossed(ty, runtime));
    let restored = rest
        .iter()
        .zip(&handlers)
        .zip(&tys)
        .map(|((at, handler), ty)| at.restored(handler, ty));
    let mut lent_at = lent.iter();
    let takes: Vec<proc_macro2::TokenStream> = rest
        .iter()
        .zip(&markers)
        .zip(&handlers)
        .zip(&args)
        .map(|(((at, marker), handler), arg)| match at {
            RestAt::VariableShared => {
                let slot = lent_at.next().expect("a shared position lends a slot");
                quote! {
                    unsafe {
                        <#handler as ::acvus_extern::RestoreShared<'__b, #marker, #runtime>>::restore_shared(
                            __rt, #slot, #arg,
                        )
                    }
                }
            }
            RestAt::VariableExclusive => {
                let slot = lent_at.next().expect("an exclusive position lends a slot");
                quote! {
                    unsafe {
                        <#handler as ::acvus_extern::RestoreExclusive<'__b, #marker, #runtime>>::restore_exclusive(
                            __rt, #slot, #arg,
                        )
                    }
                }
            }
            RestAt::VariableValue => quote! {
                unsafe {
                    <#handler as ::acvus_extern::RestoreByValue<'__b, #marker, #runtime>>::restore_by_value(
                        __rt, #arg,
                    )
                }
            },
            RestAt::Itself(_) => quote! { #arg },
        })
        .collect();
    let bounds = rest
        .iter()
        .zip(&markers)
        .zip(&handlers)
        .filter_map(|((at, marker), handler)| match at {
            RestAt::VariableShared => Some(
                quote! { #handler: ::acvus_extern::RestoreShared<'__b, #marker, #runtime> },
            ),
            RestAt::VariableExclusive => Some(
                quote! { #handler: ::acvus_extern::RestoreExclusive<'__b, #marker, #runtime> },
            ),
            RestAt::VariableValue => Some(
                quote! { #handler: ::acvus_extern::RestoreByValue<'__b, #marker, #runtime> },
            ),
            RestAt::Itself(_) => None,
        });
    // A type alias names every type parameter it takes (E0091), and a run
    // of concrete positions names none of its own, so it names the runtime
    // through `RestRun`.
    let run = quote! {
        <(#(#crossed,)*) as ::acvus_extern::RestRun<#runtime>>::Run
    };
    quote! {
        #[doc(hidden)]
        #[allow(non_snake_case)]
        pub mod #module {
            pub type Rest<'__a, #runtime> = #run;

            /// A result as it crosses between an instance's glue and a
            /// requirer: `Ret` is one type for every instance of the
            /// signature, and `restore` reads it back at the requirer's
            /// own type, which the checker unified with the instance's.
            ///
            /// # Safety
            /// The word `cross` hands back is exactly the result at the
            /// type the checker settled for it; it crosses nothing else
            /// with the capability; and it keeps no capability past the
            /// call.
            pub unsafe trait Returned<#runtime>: Sized
            where
                #runtime: ::acvus_extern::Runtime,
            {
                fn cross(rt: ::acvus_extern::Crossing<'_, #runtime>, r: Self) -> Ret<#runtime>;
            }

            #returned

            pub type Now<#runtime> = for<'__a> unsafe fn(
                &::acvus_extern::InstanceEntry<#runtime>,
                &mut ::acvus_extern::Ctx<'_, #runtime>,
                Rest<'__a, #runtime>,
            ) -> Ret<#runtime>;

            pub type Later<#runtime> = for<'__a> unsafe fn(
                &'__a ::acvus_extern::InstanceEntry<#runtime>,
                &'__a mut ::acvus_extern::Ctx<'_, #runtime>,
                Rest<'__a, #runtime>,
            ) -> ::acvus_extern::BoxFuture<'__a, Ret<#runtime>>;

            /// # Safety
            /// `rest` is the run of a call of an instance of this
            /// signature whose parameters after the first are `__M0..`,
            /// and the storages it names are live for `'b`.
            #[allow(clippy::needless_lifetimes, unused_variables)]
            #[inline(always)]
            pub unsafe fn restore<'__a, '__b, '__r, #runtime #(, #markers)* #(, #variable_handlers)*>(
                __rt: ::acvus_extern::Crossing<'__r, #runtime>,
                __rest: Rest<'__a, #runtime>,
                __lent: &'__b mut #lent_ty,
                _: ::core::marker::PhantomData<(#(#markers,)*)>,
            ) -> (#(#restored,)*)
            where
                '__a: '__b,
                #runtime: ::acvus_extern::Runtime,
                #(#bounds,)*
            {
                let (#(#args,)*) = __rest;
                let #lent_pattern = __lent;
                (#(#takes,)*)
            }
        }
    }
}

/// How a signature's result crosses between an instance's glue and a
/// requirer (RFC-0068 rule 6): as it is typed where it names no type variable;
/// as `Option<Rt::Value>` where it is an `Option` of one of the signature's
/// variables, so the option stays in registers; as one `Rt::Value` where a
/// variable stands anywhere else in it.
enum RetShape {
    Concrete,
    OptionOfVariable(Ident),
    Whole,
}

impl RetShape {
    fn of(ret: &Type, vars: &Vars) -> Self {
        if !vars.mentions_ty_var(ret) {
            return Self::Concrete;
        }
        let Type::Path(p) = ret else {
            return Self::Whole;
        };
        let Some(seg) = p.path.segments.last() else {
            return Self::Whole;
        };
        if seg.ident != "Option" {
            return Self::Whole;
        }
        let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
            return Self::Whole;
        };
        let mut args = args.args.iter();
        let (Some(syn::GenericArgument::Type(payload)), None) = (args.next(), args.next()) else {
            return Self::Whole;
        };
        match vars.type_vars().find(|v| Vars::is_exactly(payload, v)) {
            Some(var) => Self::OptionOfVariable(var.clone()),
            None => Self::Whole,
        }
    }
}

/// The result as the requirer receives it (RFC-0068 rule 6): its type at the
/// requirer's own markers, the bound that read needs, and the read itself
/// from the word the glue wrote (`__r`, with `__rt` in scope).
struct Received {
    ty: proc_macro2::TokenStream,
    bound: proc_macro2::TokenStream,
    read: proc_macro2::TokenStream,
}

impl Received {
    fn of(shape: &RetShape, ret: &Type, runtime: &Ident) -> Self {
        match shape {
            RetShape::Concrete => Self {
                ty: quote! { #ret },
                bound: quote! { #ret: ::acvus_extern::Cross<#runtime, Form = ::acvus_extern::One> },
                read: quote! { __r },
            },
            RetShape::OptionOfVariable(var) => Self {
                ty: quote! {
                    ::core::option::Option<<#var as ::acvus_extern::Passed<'__r, #runtime>>::As>
                },
                bound: quote! { #var: for<'__p> ::acvus_extern::Passed<'__p, #runtime> },
                read: quote! {
                    // SAFETY: the caller's contract: the requirer's own type
                    // here is what the checker unified the instance's result
                    // with, and the storage a borrow names is the receiver's.
                    __r.map(|__w| unsafe {
                        <#var as ::acvus_extern::Passed<'__r, #runtime>>::restore(__rt, __w)
                    })
                },
            },
            RetShape::Whole => Self {
                ty: quote! { <#ret as ::acvus_extern::Passed<'__r, #runtime>>::As },
                bound: quote! { #ret: for<'__p> ::acvus_extern::Passed<'__p, #runtime> },
                read: quote! {
                    // SAFETY: as the option shape's.
                    unsafe { <#ret as ::acvus_extern::Passed<'__r, #runtime>>::restore(__rt, __r) }
                },
            },
        }
    }
}

/// The module `extern_signature!` writes beside a signature, as the
/// `instance_of` path of one of its instances names it.
fn signature_module_ident(ident: &Ident) -> Ident {
    format_ident!("__sig_{}", ident)
}

/// The same module, from the path an `#[extern_fn(instance_of = ..)]`
/// wrote: the signature's own last segment, with no arguments.
fn signature_module_path(path: &Path) -> Path {
    let mut path = path.clone();
    let last = path
        .segments
        .last_mut()
        .expect("a parsed path names at least one segment");
    last.ident = signature_module_ident(&last.ident);
    last.arguments = syn::PathArguments::None;
    path
}

/// The `Signature` impl of a shared signature whose first parameter stands
/// at its first type variable in any of the three modes, and whose later
/// parameters and result are each one whole value: `core::eq` and
/// `iter::next` alike.
///
/// A signature outside that shape gets no impl, and the missing impl is the
/// refusal: a handler that writes `InstanceOf<sig::vec<C, T, Rt>>` is told
/// that `vec` is not a signature a bound can name, because its arguments
/// are not a run of whole values (RFC-0067 rule 1).
fn signature_call(
    ns: &LitStr,
    ident: &Ident,
    vars: &Vars,
    marker_params: &[Ident],
    runtime: &Ident,
    params: &[ExternParam],
    ret: &Type,
) -> proc_macro2::TokenStream {
    let _ = ns;
    let Some(first) = vars.first_ty() else {
        return proc_macro2::TokenStream::new();
    };
    let Some((head, tail)) = params.split_first() else {
        return proc_macro2::TokenStream::new();
    };
    if !Vars::is_exactly(&head.ty, first) {
        return proc_macro2::TokenStream::new();
    }
    let Some(ReceiverMode(receiver_mode)) = ReceiverMode::of(head.mode) else {
        return proc_macro2::TokenStream::new();
    };
    let rest: Vec<RestAt> = tail.iter().map(|p| RestAt::of(p, vars)).collect();
    if !rest.iter().all(|at| at.is_signature_shaped()) {
        return proc_macro2::TokenStream::new();
    }
    // The crossing of a position that is not at a variable is the type the
    // signature wrote there, so the signature's own types fill the run.
    let rest_tys: Vec<&Type> = tail.iter().map(|p| &p.ty).collect();
    // A concrete borrow crosses as the Rust reference itself and asks no
    // crossing of the type it names.
    let rest_crossings = rest.iter().zip(&rest_tys).filter_map(|(at, ty)| {
        matches!(at, RestAt::Itself(Mode::Value)).then(|| {
            quote! { #ty: ::acvus_extern::Cross<#runtime, Form = ::acvus_extern::One> }
        })
    });
    let module = signature_module_ident(ident);
    let kinds = vars.kind_predicates();
    let required = rest
        .iter()
        .zip(&rest_tys)
        .map(|(at, ty)| at.required(ty, runtime));
    let required_bounds: Vec<proc_macro2::TokenStream> = rest
        .iter()
        .zip(&rest_tys)
        .filter_map(|(at, ty)| at.required_bound(ty, runtime))
        .collect();
    let rest_args: Vec<Ident> = (0..rest.len()).map(|at| format_ident!("__x{at}")).collect();
    let crossed_args = rest
        .iter()
        .zip(&rest_tys)
        .zip(&rest_args)
        .map(|((at, ty), arg)| at.cross_required(ty, runtime, arg));
    let kinds_again = vars.kind_predicates();
    let rest_crossings: Vec<proc_macro2::TokenStream> = rest_crossings.collect();
    let Received {
        ty: received,
        bound: restore_bound,
        read: restore,
    } = Received::of(&RetShape::of(ret, vars), ret, runtime);
    quote! {
        impl<#(#marker_params,)*> ::acvus_extern::Signature<#runtime>
            for #ident<#(#marker_params,)*>
        where
            #runtime: ::acvus_extern::Runtime,
            #restore_bound,
            #(#kinds,)*
            #(#rest_crossings,)*
        {
            type This = #first;
            type Mode = #receiver_mode;
            type Words<'__a>
                = #module::Rest<'__a, #runtime>
            where
                Self: '__a;
            type Ret<'__r>
                = #received
            where
                Self: '__r;
            type Now = #module::Now<#runtime>;
            type Later = #module::Later<#runtime>;

            unsafe fn call_now<'__r>(
                __value: <#runtime as ::acvus_extern::Runtime>::Value,
                __ctx: &mut ::acvus_extern::Ctx<'_, #runtime>,
                __rest: <Self as ::acvus_extern::Signature<#runtime>>::Words<'__r>,
            ) -> <Self as ::acvus_extern::Signature<#runtime>>::Ret<'__r>
            where
                Self: '__r,
            {
                // SAFETY: the caller's contract: the word addresses the
                // entry of an instance of this signature.
                let __entry = unsafe {
                    <#runtime as ::acvus_extern::Runtime>::instance_entry(&__value)
                };
                // SAFETY: the entry's glue is the address `#[extern_fn]`
                // took of a mono glue of this signature, which it took at
                // this type and at no other.
                let __f: <Self as ::acvus_extern::Signature<#runtime>>::Now =
                    unsafe { ::acvus_extern::repr::fn_of(__entry.run.at()) };
                // SAFETY: this is the requirer's half of the crossing
                // `extern_signature!` wrote, and it reads the result at the
                // requirer's type, which the checker unified with the
                // instance's (RFC-0068 rule 6).
                let __rt = unsafe { ::acvus_extern::Crossing::new(__ctx.rt) };
                // SAFETY: the caller's contract, which is the glue's own.
                let __r = unsafe { __f(__entry, __ctx, __rest) };
                #restore
            }

            unsafe fn call_later<'__r>(
                __value: <#runtime as ::acvus_extern::Runtime>::Value,
                __ctx: &'__r mut ::acvus_extern::Ctx<'_, #runtime>,
                __rest: <Self as ::acvus_extern::Signature<#runtime>>::Words<'__r>,
            ) -> impl ::core::future::Future<
                Output = <Self as ::acvus_extern::Signature<#runtime>>::Ret<'__r>,
            > + ::core::marker::Send + '__r
            where
                Self: '__r,
                <Self as ::acvus_extern::Signature<#runtime>>::Ret<'__r>: ::core::marker::Send,
            {
                // SAFETY: the caller's contract.
                let __task = unsafe {
                    <#runtime as ::acvus_extern::Runtime>::instance_entry(&__value)
                }
                .run
                .task();
                if __task == ::acvus_extern::Task::Sync {
                    // SAFETY: the value's own task says the glue returns.
                    let __r = unsafe { Self::call_now(__value, __ctx, __rest) };
                    return ::acvus_extern::Either::Left(::core::future::ready(__r));
                }
                ::acvus_extern::Either::Right(async move {
                    // SAFETY: as `call_now`'s. The entry is read inside the
                    // future because the glue holds it for as long as the
                    // body it runs.
                    let __entry = unsafe {
                        <#runtime as ::acvus_extern::Runtime>::instance_entry(&__value)
                    };
                    // SAFETY: as `call_now`'s, at the awaiting shape the
                    // task above named.
                    let __f: <Self as ::acvus_extern::Signature<#runtime>>::Later =
                        unsafe { ::acvus_extern::repr::fn_of(__entry.run.at()) };
                    // SAFETY: as `call_now`'s.
                    let __rt = unsafe { ::acvus_extern::Crossing::new(__ctx.rt) };
                    // SAFETY: as `call_now`'s.
                    let __r = unsafe { __f(__entry, __ctx, __rest) }.await;
                    #restore
                })
            }
        }

        // SAFETY: a variable's position taken by value crosses by its own
        // `Passed`, a borrowed variable's as the word its storage derefs to,
        // and any other position as itself; nothing else crosses, and the
        // capability is not kept.
        unsafe impl<#(#marker_params,)*> ::acvus_extern::CrossesRest<#runtime>
            for #ident<#(#marker_params,)*>
        where
            #runtime: ::acvus_extern::Runtime,
            #restore_bound,
            #(#kinds_again,)*
            #(#rest_crossings,)*
            #(#required_bounds,)*
        {
            type Rest<'__a>
                = (#(#required,)*)
            where
                Self: '__a;

            #[inline(always)]
            fn cross_rest<'__a>(
                __rt: ::acvus_extern::Crossing<'_, #runtime>,
                __rest: <Self as ::acvus_extern::CrossesRest<#runtime>>::Rest<'__a>,
            ) -> <Self as ::acvus_extern::Signature<#runtime>>::Words<'__a>
            where
                Self: '__a,
            {
                let (#(#rest_args,)*) = __rest;
                (#(#crossed_args,)*)
            }
        }
    }
}
