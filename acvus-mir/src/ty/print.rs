//! A concrete type in the language's own notation, for the IR printer:
//! `i64`, `&mut [i64]`, `{a: i64, b: i64}`, `Option<String>`,
//! `fn(i64) -> i64`, a declared type by its qualified name with its
//! arguments. Names come from the interner; nothing is written but the
//! notation.
//!
//! What a function type or a declared type carries beyond that notation, its
//! effect, flows, calls and holds, and a declared type's effect and identity
//! arguments, prints in a bracket after it, each where it differs from its
//! default (types §5) and by the ledger's own term: `fn(i64) -> i64 [pure,
//! calls=once]`, `Vec<i64> [e0=pure, id0=3]`. A type equal to its defaults
//! prints no bracket.
//!
//! Deterministic: the same type prints the same text. An object's fields and
//! an enum's variants print in the order the type holds them.

use acvus_utils::{Interner, LocalIdOps, QualifiedRef};

use crate::ir::{known_effect, known_len};
use crate::ty::{
    Alignment, Calls, Concrete, Effect, EffectArg, FlowEnd, FlowTerm, Flows, Holds, IdentityTerm, ObjectTy, Reissue,
    SplitCost, Ty, TyKind, TypeArg, Types,
};

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
        TyKind::Fn { params, ret, effect, flows, calls, holds, values: () } => {
            out.push_str("fn(");
            write_list(types, names, params, out);
            out.push_str(") -> ");
            write_ty(types, names, *ret, out);
            let mut carried = effect_terms(names, known_effect(effect));
            carried.extend(flows_term(names, flows));
            if *calls == Calls::Once {
                carried.push("calls=once".to_string());
            }
            match holds {
                Holds::Nothing => {}
                Holds::Shared => carried.push("holds=shared".to_string()),
                Holds::Mut => carried.push("holds=mut".to_string()),
            }
            write_bracket(&carried, out);
        }
        TyKind::UserDefined { id, type_args, effect_args, identity_args } => {
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
            let mut carried: Vec<String> = Vec::new();
            for (index, arg) in effect_args.iter().enumerate() {
                if let Some(effect) = effect_arg(names, arg) {
                    carried.push(format!("e{index}={effect}"));
                }
            }
            for (index, identity) in identity_args.iter().enumerate() {
                let identity = match identity {
                    IdentityTerm::Known(identity) => identity.to_raw(),
                    IdentityTerm::Var(var) => match *var {},
                };
                carried.push(format!("id{index}={identity}"));
            }
            write_bracket(&carried, out);
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

/// ` [a, b]`, or nothing for no term.
fn write_bracket(terms: &[String], out: &mut String) {
    if !terms.is_empty() {
        out.push_str(&format!(" [{}]", terms.join(", ")));
    }
}

/// The terms by which `effect` differs from the opaque effect (types §5):
/// its reissue, `commutative` where a call that is not pure commutes (a
/// pure one commutes exactly where it writes nothing, which `writes`
/// already shows), the contexts it reads and writes, and its split cost.
fn effect_terms(names: &Interner, effect: &Effect) -> Vec<String> {
    let mut terms = Vec::new();
    match effect.reissue() {
        Reissue::Pure => terms.push("pure".to_string()),
        Reissue::Idempotent => terms.push("idempotent".to_string()),
        Reissue::Opaque => {}
    }
    if effect.commutes() && effect.reissue() != Reissue::Pure {
        terms.push("commutative".to_string());
    }
    let contexts = |set: &crate::ty::Contexts| {
        set.iter().map(|context| format!("@{}", qualified(names, *context))).collect::<Vec<_>>().join(", ")
    };
    if !effect.reads().is_empty() {
        terms.push(format!("reads=[{}]", contexts(effect.reads())));
    }
    if !effect.writes().is_empty() {
        terms.push(format!("writes=[{}]", contexts(effect.writes())));
    }
    match effect.split() {
        SplitCost::Unavailable => {}
        SplitCost::WithCost(cost) => terms.push(format!("split={}", cost.for_backend())),
    }
    terms
}

/// An effect argument where it differs from an unmarked opaque effect:
/// its terms, `#` before them where it is marked specializable (types §8),
/// several in parentheses.
fn effect_arg(names: &Interner, arg: &EffectArg<Concrete>) -> Option<String> {
    let (marked, effect) = match arg {
        EffectArg::Uniform(effect) => (false, known_effect(effect)),
        EffectArg::Specialized(effect) => (true, known_effect(effect)),
    };
    let terms = effect_terms(names, effect);
    let effect = match terms.len() {
        0 if !marked => return None,
        0 => "opaque".to_string(),
        1 => terms[0].clone(),
        _ => format!("({})", terms.join(", ")),
    };
    Some(if marked { format!("#{effect}") } else { effect })
}

/// `flows=[p0->ret, ..]` where the flows are listed, nothing for every flow
/// (types §5).
fn flows_term(names: &Interner, flows: &FlowTerm<Concrete>) -> Option<String> {
    let Flows::Listed(listed) = flows.get() else {
        return None;
    };
    let listed: Vec<String> = listed
        .iter()
        .map(|flow| {
            let alignment = match &flow.alignment {
                Alignment::Aligned => String::new(),
                Alignment::Any => ":any".to_string(),
                Alignment::Labelled(labelled) => {
                    let declared = qualified(names, labelled.declared.function);
                    let declared = match labelled.declared.r#impl {
                        Some(at) => format!("{declared}#{at}"),
                        None => declared,
                    };
                    let map: Vec<String> = labelled
                        .map
                        .iter()
                        .map(|segments| format!("[{}]", segments.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")))
                        .collect();
                    format!(":labelled({declared} {})", map.join(" "))
                }
            };
            format!("{}->{}{alignment}", flow_end(flow.from), flow_end(flow.to))
        })
        .collect();
    Some(format!("flows=[{}]", listed.join(", ")))
}

fn flow_end(end: FlowEnd) -> String {
    match end {
        FlowEnd::Result => "ret".to_string(),
        FlowEnd::Param(index) => format!("p{index}"),
        FlowEnd::Captures => "cap".to_string(),
    }
}
