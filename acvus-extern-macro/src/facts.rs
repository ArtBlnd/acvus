//! The facts the attribute states as written (rfc_extern declared facts),
//! carried to the compiler as its own forms (compiler: extern fn §3-§4). The
//! attribute states them and nothing else: whether each is well formed at the
//! signature is the compiler's to decide.
//!
//! A relation is `left = right`, `left <= right` or `left < right` between
//! affine terms: an integer constant, an argument by its name, `result`,
//! `len(..)` of an argument or of `result`, `old(..)` of an argument or of
//! its `len(..)`, sums and differences, and a constant's multiples. An
//! argument is named as the function names it, and numbered as a call's
//! arguments are: a capture and the context are no argument.

use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::parse::ParseStream;
use syn::{Attribute, BinOp, Expr, ExprCall, Ident, Lit, Path, Token, UnOp, parenthesized};

use crate::attr::Ends;

/// How a call ends, as the compiler holds it.
pub fn ends(ends: Option<Ends>) -> TokenStream {
    match ends {
        None => quote!(::acvus_extern::decl::Ends::Unstated),
        Some(Ends::Returns) => quote!(::acvus_extern::decl::Ends::Returns),
        Some(Ends::Total) => quote!(::acvus_extern::decl::Ends::Total),
    }
}

/// Each relation as the compiler holds it. `arguments` are the function's
/// arguments by name, in order.
pub fn relations(relations: &[Expr], arguments: &[&Ident]) -> syn::Result<Vec<TokenStream>> {
    relations.iter().map(|relation| self::relation(relation, arguments)).collect()
}

fn relation(relation: &Expr, arguments: &[&Ident]) -> syn::Result<TokenStream> {
    let (left, comparison, right) = match relation {
        Expr::Assign(assign) => (&*assign.left, quote!(Eq), &*assign.right),
        Expr::Binary(binary) => {
            let comparison = match binary.op {
                BinOp::Le(_) => quote!(Le),
                BinOp::Lt(_) => quote!(Lt),
                _ => return Err(refused(relation, "a relation is `=`, `<=` or `<`")),
            };
            (&*binary.left, comparison, &*binary.right)
        }
        _ => return Err(refused(relation, "a relation is `=`, `<=` or `<`")),
    };
    let (left, right) = (term(left, arguments)?, term(right, arguments)?);
    Ok(quote! {
        ::acvus_extern::decl::Relation {
            left: #left,
            comparison: ::acvus_extern::decl::Comparison::#comparison,
            right: #right,
        }
    })
}

fn term(term: &Expr, arguments: &[&Ident]) -> syn::Result<TokenStream> {
    let boxed = |term: TokenStream| quote!(::std::boxed::Box::new(#term));
    match term {
        Expr::Paren(paren) => self::term(&paren.expr, arguments),
        Expr::Lit(_) | Expr::Unary(_) => {
            let constant = constant(term)?;
            Ok(quote!(::acvus_extern::decl::Term::Const(#constant)))
        }
        Expr::Binary(binary) => {
            let (left, right) = (&*binary.left, &*binary.right);
            match binary.op {
                BinOp::Add(_) | BinOp::Sub(_) => {
                    let (left, right) = (boxed(self::term(left, arguments)?), boxed(self::term(right, arguments)?));
                    let op = match binary.op {
                        BinOp::Add(_) => quote!(Add),
                        _ => quote!(Sub),
                    };
                    Ok(quote!(::acvus_extern::decl::Term::#op(#left, #right)))
                }
                BinOp::Mul(_) => {
                    let (factor, scaled) = match (constant(left), constant(right)) {
                        (Ok(factor), _) => (factor, right),
                        (_, Ok(factor)) => (factor, left),
                        (Err(_), Err(_)) => return Err(refused(term, "a term is affine: one side of `*` is a constant")),
                    };
                    let scaled = boxed(self::term(scaled, arguments)?);
                    Ok(quote!(::acvus_extern::decl::Term::Scale(#factor, #scaled)))
                }
                _ => Err(refused(term, "a term is a sum, a difference or a constant's multiple")),
            }
        }
        Expr::Path(_) | Expr::Call(_) => {
            let atom = atom(term, arguments)?;
            Ok(quote!(::acvus_extern::decl::Term::Atom(#atom)))
        }
        _ => Err(refused(term, "a term is a constant, an argument, `result`, `len(..)`, `old(..)` or their affine sum")),
    }
}

fn atom(atom: &Expr, arguments: &[&Ident]) -> syn::Result<TokenStream> {
    if let Some(subject) = subject(atom, arguments)? {
        return Ok(match subject {
            Subject::Result => quote!(::acvus_extern::decl::Atom::Result),
            Subject::Arg(at) => quote!(::acvus_extern::decl::Atom::Arg(#at)),
        });
    }
    let Expr::Call(call) = atom else {
        return Err(refused(atom, "a term is a constant, an argument, `result`, `len(..)` or `old(..)`"));
    };
    let [of] = call.args.iter().collect::<Vec<_>>()[..] else {
        return Err(refused(atom, "`len` and `old` are of one term"));
    };
    match function(call)?.to_string().as_str() {
        "len" => match subject(of, arguments)? {
            Some(Subject::Result) => Ok(quote!(::acvus_extern::decl::Atom::Len(::acvus_extern::decl::Subject::Result))),
            Some(Subject::Arg(at)) => Ok(quote!(::acvus_extern::decl::Atom::Len(::acvus_extern::decl::Subject::Arg(#at)))),
            None => Err(refused(of, "`len` is of an argument or of `result`")),
        },
        "old" => old(of, arguments),
        _ => Err(refused(atom, "a call in a relation is `len(..)` or `old(..)`")),
    }
}

/// `old(x)`, an argument `x` as it stood when the call began, or `old(len(x))`.
fn old(of: &Expr, arguments: &[&Ident]) -> syn::Result<TokenStream> {
    if let Some(Subject::Arg(at)) = subject(of, arguments)? {
        return Ok(quote!(::acvus_extern::decl::Atom::Old(::acvus_extern::decl::Before::Lent(#at))));
    }
    if let Expr::Call(call) = of
        && function(call)? == "len"
        && let [inner] = call.args.iter().collect::<Vec<_>>()[..]
        && let Some(Subject::Arg(at)) = subject(inner, arguments)?
    {
        return Ok(quote!(::acvus_extern::decl::Atom::Old(::acvus_extern::decl::Before::Len(#at))));
    }
    Err(refused(of, "`old` is of an argument or of its `len`"))
}

/// What an atom speaks of: the result, or an argument by its number.
enum Subject {
    Result,
    Arg(usize),
}

/// `term` as `result` or an argument's name; none where it is neither a
/// name; refused where it names nothing the call has.
fn subject(term: &Expr, arguments: &[&Ident]) -> syn::Result<Option<Subject>> {
    let Expr::Path(path) = term else {
        return Ok(None);
    };
    let Some(name) = path.path.get_ident() else {
        return Err(refused(term, "an argument is named as the function names it"));
    };
    let at = arguments.iter().position(|argument| *argument == name);
    match (name == "result", at) {
        (true, Some(_)) => Err(refused(term, "`result` names the result in a relation, and an argument of that name is ambiguous")),
        (true, None) => Ok(Some(Subject::Result)),
        (false, Some(at)) => Ok(Some(Subject::Arg(at))),
        (false, None) => Err(refused(term, "a relation speaks of the call's arguments; a capture and the context are none")),
    }
}

/// The function a call in a relation names, `len` or `old` by its own word.
fn function(call: &ExprCall) -> syn::Result<&Ident> {
    let not_named = || refused(&call.func, "a call in a relation is `len(..)` or `old(..)`");
    let Expr::Path(path) = &*call.func else {
        return Err(not_named());
    };
    path.path.get_ident().ok_or_else(not_named)
}

/// An integer constant, written as a literal or a negated one.
fn constant(term: &Expr) -> syn::Result<i128> {
    match term {
        Expr::Lit(lit) => match &lit.lit {
            Lit::Int(int) => int.base10_parse::<i128>(),
            _ => Err(refused(term, "a constant is an integer")),
        },
        Expr::Unary(unary) if matches!(unary.op, UnOp::Neg(_)) => Ok(-constant(&unary.expr)?),
        Expr::Paren(paren) => constant(&paren.expr),
        _ => Err(refused(term, "a constant is an integer")),
    }
}

fn refused(at: &impl Spanned, message: &str) -> syn::Error {
    syn::Error::new(at.span(), message)
}

/// The facts written on one argument about its place.
#[derive(Default)]
pub struct PlaceFacts {
    pub touches: Vec<Touch>,
    pub gives: Vec<Gives>,
}

/// `#[touch(mode, element)]`.
pub struct Touch {
    pub mode: Mode,
    pub element: Option<Element>,
}

/// How a call touches a place.
pub enum Mode {
    Read,
    View,
    ViewMut,
    /// An absent entry becomes the argument `default`, and a `&mut` to it
    /// is given.
    Open { default: Ident },
    /// The place holds the argument named.
    Store(Ident),
    /// The place is emptied.
    Emptied,
}

/// One element of a place: the entry at a key argument, or the element at
/// an index argument.
pub enum Element {
    Key(Ident),
    Index(Ident),
}

/// `#[gives(element, through = f)]`: the result is what the place held when
/// the call began, or the declared function `f` of it.
pub struct Gives {
    pub element: Option<Element>,
    pub through: Option<Path>,
}

impl PlaceFacts {
    /// The facts `attrs` write, each a `#[touch(..)]` or a `#[gives(..)]`.
    pub fn read(attrs: &[Attribute]) -> syn::Result<PlaceFacts> {
        let mut facts = PlaceFacts::default();
        for attr in attrs {
            if attr.path().is_ident("touch") {
                facts.touches.push(attr.parse_args_with(touch)?);
            } else if matches!(attr.meta, syn::Meta::Path(_)) {
                facts.gives.push(Gives { element: None, through: None });
            } else {
                facts.gives.push(attr.parse_args_with(gives)?);
            }
        }
        Ok(facts)
    }
}

fn touch(input: ParseStream<'_>) -> syn::Result<Touch> {
    let word: Ident = input.parse()?;
    let mode = match word.to_string().as_str() {
        "read" => Mode::Read,
        "view" => Mode::View,
        "view_mut" => Mode::ViewMut,
        "open" => {
            let content;
            parenthesized!(content in input);
            let key: Ident = content.parse()?;
            if key != "default" {
                return Err(refused(&key, "`open` is `open(default = d)`"));
            }
            content.parse::<Token![=]>()?;
            Mode::Open { default: content.parse()? }
        }
        "store" => {
            let content;
            parenthesized!(content in input);
            let stored: Ident = content.parse()?;
            match stored == "empty" {
                true => Mode::Emptied,
                false => Mode::Store(stored),
            }
        }
        _ => {
            return Err(refused(&word, "a touch is `read`, `view`, `view_mut`, `open(default = d)`, `store(v)` or `store(empty)`"));
        }
    };
    let element = match input.is_empty() {
        true => None,
        false => {
            input.parse::<Token![,]>()?;
            Some(element(input)?)
        }
    };
    Ok(Touch { mode, element })
}

fn gives(input: ParseStream<'_>) -> syn::Result<Gives> {
    let mut gives = Gives { element: None, through: None };
    while !input.is_empty() {
        if input.peek(Ident) && input.fork().parse::<Ident>().is_ok_and(|word| word == "through") {
            input.parse::<Ident>()?;
            input.parse::<Token![=]>()?;
            gives.through = Some(input.parse()?);
        } else {
            gives.element = Some(element(input)?);
        }
        if !input.is_empty() {
            input.parse::<Token![,]>()?;
        }
    }
    Ok(gives)
}

/// `key = k` or `index = i`.
fn element(input: ParseStream<'_>) -> syn::Result<Element> {
    let word: Ident = input.parse()?;
    input.parse::<Token![=]>()?;
    let named: Ident = input.parse()?;
    match word.to_string().as_str() {
        "key" => Ok(Element::Key(named)),
        "index" => Ok(Element::Index(named)),
        _ => Err(refused(&word, "an element is `key = k` or `index = i`")),
    }
}

/// The call's access as the compiler holds it: every touch written, as one
/// whole, where any is, and each place given. `places` are each argument's
/// facts, in order; `runtime` the runtime a path of another declaration is
/// read at.
pub fn access(places: &[&PlaceFacts], arguments: &[&Ident], runtime: &Ident) -> syn::Result<Vec<TokenStream>> {
    let number = |name: &Ident| -> syn::Result<usize> {
        arguments
            .iter()
            .position(|argument| *argument == name)
            .ok_or_else(|| refused(name, "a fact of a place names the call's arguments; a capture and the context are none"))
    };
    let place = |param: usize, element: &Option<Element>| -> syn::Result<TokenStream> {
        let element = match element {
            None => quote!(::core::option::Option::None),
            Some(Element::Key(key)) => {
                let at = number(key)?;
                quote!(::core::option::Option::Some(::acvus_extern::decl::Element::Key(#at)))
            }
            Some(Element::Index(index)) => {
                let at = number(index)?;
                quote!(::core::option::Option::Some(::acvus_extern::decl::Element::Index(#at)))
            }
        };
        Ok(quote!(::acvus_extern::decl::Place { param: #param, element: #element }))
    };

    let mut touches = Vec::new();
    let mut given = Vec::new();
    for (param, facts) in places.iter().enumerate() {
        for touch in &facts.touches {
            let at = place(param, &touch.element)?;
            let mode = match &touch.mode {
                Mode::Read => quote!(::acvus_extern::decl::Mode::Read),
                Mode::View => quote!(::acvus_extern::decl::Mode::View(::acvus_extern::decl::Mutability::Shared)),
                Mode::ViewMut => quote!(::acvus_extern::decl::Mode::View(::acvus_extern::decl::Mutability::Mut)),
                Mode::Open { default } => {
                    let default = number(default)?;
                    quote!(::acvus_extern::decl::Mode::Open { default: #default })
                }
                Mode::Store(value) => {
                    let value = number(value)?;
                    quote!(::acvus_extern::decl::Mode::Store(::acvus_extern::decl::Stored::Value(#value)))
                }
                Mode::Emptied => quote!(::acvus_extern::decl::Mode::Store(::acvus_extern::decl::Stored::Emptied)),
            };
            touches.push(quote!(::acvus_extern::decl::Touch { place: #at, mode: #mode }));
        }
        for gives in &facts.gives {
            let at = place(param, &gives.element)?;
            let through = match &gives.through {
                None => quote!(::core::option::Option::None),
                Some(path) => quote! {
                    ::core::option::Option::Some(<#path as ::acvus_extern::register::Declared<#runtime>>::path(names))
                },
            };
            given.push(quote!(::acvus_extern::decl::Access::Gives { place: #at, through: #through }));
        }
    }
    let mut access = Vec::new();
    if !touches.is_empty() {
        access.push(quote!(::acvus_extern::decl::Access::Touches(::std::boxed::Box::new([#(#touches),*]))));
    }
    access.extend(given);
    Ok(access)
}
