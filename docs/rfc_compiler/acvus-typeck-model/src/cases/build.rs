//! A case's declarations and text made into the model's `Decls` and
//! `Program`, recorded in the order written, as `ProgramBuilder` records them
//! (`design/19-cases.md`, Form).

use std::collections::{BTreeMap, BTreeSet};

use crate::certificate::Value;
use crate::decl::{Decls, DeclsBuilder};
use crate::lit::LitVar;
use crate::program::{ENTRY, Origin, Program, ProgramBuilder, Projection};
use crate::sig::{DeclaredTypes, ExternDecl, SigTy, SigVar, Signature, StructDecl};
use crate::ty::{FloatTy, IntTy, Name, Scalar, Term, Ty};

use super::parse::{Case, DeclItem, Located, OriginText, ProjectionText, TextItem, TypeExpr};

type Built<T> = Result<T, String>;

/// The scalar a name writes.
fn scalar(name: &str) -> Option<Scalar> {
    let int = |int| Some(Scalar::Int(int));
    match name {
        "i8" => int(IntTy::I8),
        "i16" => int(IntTy::I16),
        "i32" => int(IntTy::I32),
        "i64" => int(IntTy::I64),
        "u8" => int(IntTy::U8),
        "u16" => int(IntTy::U16),
        "u32" => int(IntTy::U32),
        "u64" => int(IntTy::U64),
        "f32" => Some(Scalar::Float(FloatTy::F32)),
        "f64" => Some(Scalar::Float(FloatTy::F64)),
        "bool" => Some(Scalar::Bool),
        "char" => Some(Scalar::Char),
        _ => None,
    }
}

/// The term `expr` writes at its outermost constructor, each part made by
/// `part`. A name the declarations and the language do not give is an error.
fn term<P, F>(types: &DeclaredTypes, expr: &TypeExpr, part: &mut F) -> Built<Term<P>>
where
    F: FnMut(&TypeExpr) -> Built<P>,
{
    let parts = |exprs: &[TypeExpr], part: &mut F| -> Built<Vec<P>> {
        exprs.iter().map(&mut *part).collect()
    };
    let named = |pairs: &[(String, TypeExpr)], part: &mut F| -> Built<BTreeMap<Name, P>> {
        let mut named = BTreeMap::new();
        for (name, expr) in pairs {
            if named.insert(Name::new(name), part(expr)?).is_some() {
                return Err(format!("{name} is written twice"));
            }
        }
        Ok(named)
    };
    Ok(match expr {
        TypeExpr::Ref(mutability, target) => Term::Ref(*mutability, part(target)?),
        TypeExpr::Tuple(components) if components.is_empty() => Term::Unit,
        TypeExpr::Tuple(components) => Term::Tuple(parts(components, part)?),
        TypeExpr::Array(len, element) => Term::Array(*len, part(element)?),
        TypeExpr::Slice(element) => Term::Slice(part(element)?),
        TypeExpr::Never => Term::Never,
        TypeExpr::Identity => return Err("`#` is a signature's identity".to_string()),
        TypeExpr::Object(fields) => Term::Object(named(fields, part)?),
        TypeExpr::Enum(name, variants) => Term::Enum(Name::new(name), named(variants, part)?),
        TypeExpr::FunctionValue(arity, None) => Term::FunctionValue { arity: *arity },
        TypeExpr::FunctionValue(arity, Some(_)) => {
            return Err(format!(
                "`Fn/{arity}{{..}}` names Values, which only a value a certificate change \
                 plants writes"
            ));
        }
        TypeExpr::Function(params, ret) => Term::Function {
            params: parts(params, part)?,
            ret: part(ret)?,
        },
        TypeExpr::Named(name, args) => {
            let arity = |n: usize| -> Built<()> {
                if args.len() == n {
                    Ok(())
                } else {
                    Err(format!(
                        "{name} takes {n} arguments, written {}",
                        args.len()
                    ))
                }
            };
            if let Some(scalar) = scalar(name) {
                arity(0)?;
                return Ok(Term::Scalar(scalar));
            }
            match name.as_str() {
                "String" => {
                    arity(0)?;
                    Term::String
                }
                "str" => {
                    arity(0)?;
                    Term::Str
                }
                "Option" => {
                    arity(1)?;
                    Term::Option(part(&args[0])?)
                }
                "Result" => {
                    arity(2)?;
                    Term::Result {
                        ok: part(&args[0])?,
                        err: part(&args[1])?,
                    }
                }
                "Error" => {
                    arity(1)?;
                    Term::ErrorOf(part(&args[0])?)
                }
                _ => {
                    let name_ = Name::new(name);
                    if let Some(decl) = types.struct_decl(&name_) {
                        arity(decl.fields.len())?;
                        Term::Struct(name_, parts(args, part)?)
                    } else if let Some(decl) = types.extern_decl(&name_) {
                        arity(decl.arity + usize::from(decl.identity))?;
                        let mut written = parts(args, part)?;
                        let identity = if decl.identity { written.pop() } else { None };
                        Term::Extern {
                            name: name_,
                            args: written,
                            identity,
                        }
                    } else {
                        return Err(format!("no type {name} is declared"));
                    }
                }
            }
        }
    })
}

/// The type a signature writes, its variables named by `generics`.
fn sig_ty(types: &DeclaredTypes, generics: &[String], expr: &TypeExpr) -> Built<SigTy> {
    match expr {
        TypeExpr::Identity => Ok(SigTy::Var(SigVar::Identity)),
        TypeExpr::Named(name, args) if args.is_empty() => {
            match generics.iter().position(|param| param == name) {
                Some(i) => Ok(SigTy::Var(SigVar::Type(i))),
                None => Ok(SigTy::term(term(types, expr, &mut |part| {
                    sig_ty(types, generics, part)
                })?)),
            }
        }
        _ => Ok(SigTy::term(term(types, expr, &mut |part| {
            sig_ty(types, generics, part)
        })?)),
    }
}

/// The closed type `expr` writes.
pub fn closed(types: &DeclaredTypes, expr: &TypeExpr) -> Built<Ty> {
    let term = term(types, expr, &mut |part| closed(types, part))?;
    Ty::new(term).map_err(|error| format!("{error:?}"))
}

/// The value a certificate change plants: the closed type `expr` writes, each
/// function value in it naming the lambdas written after it, its Values
/// (`design/14-certificate.md`).
pub fn planted(types: &DeclaredTypes, expr: &TypeExpr) -> Built<Value> {
    if let TypeExpr::FunctionValue(arity, values) = expr {
        let Some(values) = values else {
            return Err(format!(
                "a planted `Fn/{arity}` names its Values, `Fn/{arity}{{λ, ..}}`"
            ));
        };
        let mut lambdas = BTreeSet::new();
        for lambda in values {
            if !lambdas.insert(Name::new(lambda)) {
                return Err(format!("{lambda} is written twice"));
            }
        }
        return Value::planted(Term::FunctionValue { arity: *arity }, Some(lambdas));
    }
    let term = term(types, expr, &mut |part| planted(types, part))?;
    Value::planted(term, None)
}

/// A type the text writes: its outermost constructor, each part a class it
/// names.
fn written(types: &DeclaredTypes, expr: &TypeExpr) -> Built<Term<Name>> {
    term(types, expr, &mut |part| match part {
        TypeExpr::Named(name, args) if args.is_empty() => Ok(Name::new(name)),
        _ => Err(format!("a written type's parts are classes, not {part:?}")),
    })
}

fn at<T>(line: usize, built: Built<T>) -> Built<T> {
    built.map_err(|error| format!("line {line}: {error}"))
}

pub fn decls(case: &Case) -> Built<Decls> {
    let mut types = DeclaredTypes::default();
    for Located { line, item } in &case.decls {
        let declared = match item {
            DeclItem::Extern {
                name,
                arity,
                identity,
            } => types.declare_extern(
                Name::new(name),
                ExternDecl {
                    arity: *arity,
                    identity: *identity,
                },
            ),
            DeclItem::Struct { name, fields } => types.declare_struct(
                Name::new(name),
                StructDecl {
                    fields: fields.iter().map(|field| Name::new(field)).collect(),
                },
            ),
            DeclItem::Fn { .. } => continue,
        };
        at(*line, declared.map_err(|error| format!("{error:?}")))?;
    }
    let mut builder = DeclsBuilder::new(types);
    for Located { line, item } in &case.decls {
        let DeclItem::Fn {
            name,
            role,
            generics,
            identity,
            params,
            result,
        } = item
        else {
            continue;
        };
        let types = builder.types();
        let params = at(
            *line,
            params
                .iter()
                .map(|param| sig_ty(types, generics, param))
                .collect::<Built<Vec<_>>>(),
        )?;
        let result = at(*line, sig_ty(types, generics, result))?;
        let sig = Signature::new(
            generics.iter().map(|param| Name::new(param)).collect(),
            *identity,
            params,
            result,
            types,
        );
        let sig = at(*line, sig.map_err(|error| format!("{error:?}")))?;
        builder.declare(Name::new(name), sig, *role);
    }
    builder.finish().map_err(|error| format!("{error:?}"))
}

fn projection(text: &ProjectionText) -> Projection {
    match text {
        ProjectionText::Deref => Projection::Deref,
        ProjectionText::Field(field) => Projection::Field(Name::new(field)),
        ProjectionText::Component(index) => Projection::Component(*index),
        ProjectionText::Elem => Projection::Elem,
        ProjectionText::Payload { enum_name, variant } => Projection::Payload {
            enum_name: enum_name.as_deref().map(Name::new),
            variant: Name::new(variant),
        },
    }
}

/// The program the case's text writes.
pub fn program(case: &Case, decls: &Decls) -> Built<Program> {
    let types = decls.types();
    let mut p = ProgramBuilder::new(decls.clone());
    let name = |text: &String| Name::new(text);
    let names = |texts: &[String]| texts.iter().map(|text| Name::new(text)).collect::<Vec<_>>();
    for Located { line, item } in &case.text {
        match item {
            TextItem::Group { name, parent } => p.group_in(name, parent.as_deref()),
            TextItem::EndGroup => p.group(ENTRY),
            TextItem::Class { name, origin } => {
                let origin = match origin {
                    OriginText::Var => Origin::Var,
                    OriginText::Lit(literal) => Origin::Lit(LitVar::of(*literal)),
                    OriginText::Written(expr) => Origin::Written(at(*line, written(types, expr))?),
                    OriginText::Host(expr) => Origin::Host(at(*line, written(types, expr))?),
                    OriginText::Outward(expr) => Origin::Outward(at(*line, written(types, expr))?),
                    OriginText::Lambda(lambda) => Origin::Lambda(Name::new(lambda)),
                };
                p.class(name, origin);
            }
            TextItem::Universal(class) => p.universal(&name(class)),
            TextItem::Function {
                name: function,
                params,
                result,
            } => {
                let params = names(params);
                p.function(function, &params.iter().collect::<Vec<_>>(), &name(result));
            }
            TextItem::Pass { value, into, place } => p.pass_at(&name(value), &name(into), *place),
            TextItem::Equation(a, b) => p.equation(&name(a), &name(b)),
            TextItem::Use {
                node,
                callee,
                args,
                result,
                exit: None,
            } => {
                let args = names(args);
                p.call(
                    node,
                    callee,
                    &args.iter().collect::<Vec<_>>(),
                    &name(result),
                );
            }
            TextItem::Use {
                node,
                callee,
                args,
                result,
                exit: Some(exit),
            } => {
                let [operand] = args.as_slice() else {
                    return Err(format!(
                        "{node}: `?` takes one operand, written {}",
                        args.len()
                    ));
                };
                if callee != crate::decl::TRY {
                    return Err(format!("{node}: only `?` has an exit, written {callee}"));
                }
                p.try_(node, &name(operand), &name(result), &name(exit));
            }
            TextItem::CallValue {
                node,
                callee,
                args,
                result,
            } => {
                let args = names(args);
                p.call_value(
                    node,
                    &name(callee),
                    &args.iter().collect::<Vec<_>>(),
                    &name(result),
                );
            }
            TextItem::Lend {
                node,
                mutability,
                place,
                result,
            } => p.lend(node, *mutability, &name(place), &name(result)),
            TextItem::Project {
                node,
                projection: text,
                operand,
                result,
            } => p.project(node, projection(text), &name(operand), &name(result)),
            TextItem::Bind {
                node,
                path,
                position,
                binding,
            } => {
                let path: Vec<&str> = path.iter().map(String::as_str).collect();
                p.bind(node, &path, &name(position), &name(binding));
            }
        }
    }
    let program = p.build();
    check_names(&program)?;
    Ok(program)
}

/// Every class the program's items and functions name is one its text
/// gives, so that solving finds each.
fn check_names(program: &Program) -> Built<()> {
    let declared = |name: &Name| program.classes.iter().any(|class| class.name == *name);
    let named = program
        .items
        .iter()
        .flat_map(|(_, item)| item.classes())
        .chain(
            program
                .functions
                .iter()
                .flat_map(|function| function.params.iter().chain([&function.result])),
        )
        .chain(&program.universals)
        .chain(
            program
                .classes
                .iter()
                .flat_map(|class| match &class.origin {
                    Origin::Written(term) | Origin::Host(term) | Origin::Outward(term) => {
                        term.parts().into_iter().map(|(_, part)| part).collect()
                    }
                    Origin::Var | Origin::Lit(_) | Origin::Lambda(_) => Vec::new(),
                }),
        );
    for name in named {
        if !declared(name) {
            return Err(format!("the class {name} is named but not given by `let`"));
        }
    }
    for class in &program.classes {
        if let Origin::Lambda(lambda) = &class.origin
            && program.function(lambda).is_none()
        {
            return Err(format!(
                "the lambda {lambda} of {} is no group's fn",
                class.name
            ));
        }
        if program
            .classes
            .iter()
            .filter(|other| other.name == class.name)
            .count()
            > 1
        {
            return Err(format!("the class {} is given twice", class.name));
        }
    }
    Ok(())
}
