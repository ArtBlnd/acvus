//! A concrete type in the language's own notation, for the IR printer:
//! `i64`, `&mut [i64]`, `{a: i64, b: i64}`, `Option<String>`,
//! `fn(i64) -> i64`, a declared type by its qualified name with its
//! arguments. Names come from the interner; nothing is written but the
//! notation.
//!
//! Deterministic: the same type prints the same text. An object's fields and
//! an enum's variants print in the order the type holds them.

use acvus_utils::{Interner, QualifiedRef};

use crate::ir::known_len;
use crate::ty::{Concrete, ObjectTy, Ty, TyKind, TypeArg, Types};

/// `ty` written in the language's notation.
pub fn ty(types: &Types<Concrete>, names: &Interner, ty: Ty) -> String {
    let mut out = String::new();
    write_ty(types, names, ty, &mut out);
    out
}

/// A name as a program writes it, `ns::name` or the bare name; a function a
/// script declares, by its script and its instance, `ns::script::name#0`.
pub fn qualified(names: &Interner, name: QualifiedRef) -> String {
    let mut out = String::new();
    if let Some(namespace) = name.namespace {
        out.push_str(names.resolve(namespace));
        out.push_str("::");
    }
    if let Some(scope) = name.scope {
        out.push_str(names.resolve(scope.script));
        out.push_str("::");
    }
    out.push_str(names.resolve(name.name));
    if let Some(scope) = name.scope {
        out.push_str(&format!("#{}", scope.instance));
    }
    out
}

fn write_ty(types: &Types<Concrete>, names: &Interner, ty: Ty, out: &mut String) {
    acvus_utils::grow(|| match types.kind(ty) {
        TyKind::Int(int) => out.push_str(int.name()),
        TyKind::Float => out.push_str("f64"),
        TyKind::Char => out.push_str("char"),
        TyKind::String => out.push_str("String"),
        TyKind::Bool => out.push_str("bool"),
        TyKind::Unit => out.push_str("()"),
        TyKind::Never => out.push('!'),
        TyKind::Order => out.push_str("Order"),
        TyKind::Str => out.push_str("str"),
        TyKind::Handle(inner) => {
            out.push_str("Handle<");
            write_ty(types, names, *inner, out);
            out.push('>');
        }
        TyKind::Array(element, len, ()) => {
            out.push('[');
            write_ty(types, names, *element, out);
            out.push_str(&format!("; {}]", known_len(len)));
        }
        TyKind::Slice(element) => {
            out.push('[');
            write_ty(types, names, *element, out);
            out.push(']');
        }
        TyKind::Tuple(items, ()) => {
            out.push('(');
            write_list(types, names, items, out);
            if items.len() == 1 {
                out.push(',');
            }
            out.push(')');
        }
        TyKind::Option(payload, ()) => {
            out.push_str("Option<");
            write_ty(types, names, *payload, out);
            out.push('>');
        }
        TyKind::Result(ok, err, ()) => {
            out.push_str("Result<");
            write_ty(types, names, *ok, out);
            out.push_str(", ");
            write_ty(types, names, *err, out);
            out.push('>');
        }
        TyKind::Object(ObjectTy::Structural(fields, ())) => write_fields(types, names, fields, out),
        TyKind::Object(ObjectTy::Declared(name, fields)) => {
            out.push_str(&qualified(names, *name));
            write_fields(types, names, fields, out);
        }
        TyKind::Enum { name, variants, label: () } => {
            out.push_str(&qualified(names, *name));
            out.push('{');
            for (index, (tag, payload)) in variants.iter().enumerate() {
                if index > 0 {
                    out.push_str(" | ");
                }
                out.push_str(names.resolve(*tag));
                if let Some(payload) = payload {
                    out.push('(');
                    write_ty(types, names, *payload, out);
                    out.push(')');
                }
            }
            out.push('}');
        }
        TyKind::Fn { params, ret, .. } => {
            out.push_str("fn(");
            write_list(types, names, params, out);
            out.push_str(") -> ");
            write_ty(types, names, *ret, out);
        }
        TyKind::UserDefined { id, type_args, .. } => {
            out.push_str(&qualified(names, *id));
            if !type_args.is_empty() {
                out.push('<');
                for (index, arg) in type_args.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    write_arg(types, names, arg, out);
                }
                out.push('>');
            }
        }
        TyKind::Ref(mutability, target) => {
            out.push('&');
            if *mutability == crate::ty::Mutability::Mut {
                out.push_str("mut ");
            }
            write_arg(types, names, target, out);
        }
        TyKind::Closures(signatures) => {
            out.push_str("Closures<");
            write_list(types, names, signatures, out);
            out.push('>');
        }
        TyKind::Var(var) => match *var {},
    })
}

fn write_list(types: &Types<Concrete>, names: &Interner, items: &[Ty], out: &mut String) {
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        write_ty(types, names, *item, out);
    }
}

fn write_fields(types: &Types<Concrete>, names: &Interner, fields: &[(acvus_utils::Astr, Ty)], out: &mut String) {
    out.push('{');
    for (index, (name, field)) in fields.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(names.resolve(*name));
        out.push_str(": ");
        write_ty(types, names, *field, out);
    }
    out.push('}');
}

/// A type argument, `#τ` where it is marked specializable (types §8).
fn write_arg(types: &Types<Concrete>, names: &Interner, arg: &TypeArg<Concrete>, out: &mut String) {
    match arg {
        TypeArg::Uniform(ty) => write_ty(types, names, *ty, out),
        TypeArg::Specialized(ty) => {
            out.push('#');
            write_ty(types, names, *ty, out);
        }
        TypeArg::Open(var, _) => match *var {},
    }
}
