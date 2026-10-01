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
use syn::{BinOp, Expr, ExprCall, Ident, Lit, UnOp};

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
