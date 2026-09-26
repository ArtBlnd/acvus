//! RFC-0104 rule 4's promise, checked declaration by declaration: every
//! call up to a bound computes what its declaration's term says.
//!
//! The ground is parametricity, as for the steps: a type variable stands at
//! `i64` over a small domain (a key over three values, any other variable
//! over two), a table holds at most three entries, and a concretely typed
//! position is sampled with its width's edges. A term that names no entry
//! is run as a program beside the call and the two programs' values are
//! compared; a term over an entry `x[k]` is evaluated here, over the table
//! as an ordered list of entries, and the call's program prints its result
//! and the table after it.
//!
//! The declarations are the registry's: every instance stating a term is
//! checked, and [`meaning_declarations`] fails on one whose types this
//! module cannot model rather than leaving it unchecked.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::means::{BinOp, Means, Stmt, Term};
use acvus_extern::{Externs, FnKind, Laws, Registry};
use acvus_interpreter::AcvusRuntime;
use acvus_mir::graph::QualifiedRef;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{IntTy, Mutability, PolyTy, TyTerm, TyVarBound};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use super::{ExternName, product, run_at, sequences};
use crate::corpus::Outcome;

/// A value one of the model's positions takes.
#[derive(Debug, Clone, PartialEq)]
enum Model {
    Int(i128),
    Bool(bool),
    Opt(Option<Box<Model>>),
    /// A reference to the entry's payload as the table holds it when read.
    EntryRef,
}

/// What a position of the declaration's type is, as far as the model
/// writes, reads and prints it.
#[derive(Debug, Clone, PartialEq)]
enum Shape {
    /// A type variable the key of an entry stands at.
    Key,
    /// Any other type variable.
    Parametric,
    Int(IntTy),
    Float,
    Bool,
    Char,
    Text,
    Decimal,
    Option(Box<Shape>),
    VecOf(Box<Shape>),
    Lent(Mutability, Box<Shape>),
    /// The `&str` a `String` lends.
    StrView,
    Table(TableKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TableKind {
    Map,
    Set,
}

#[derive(Debug, Clone)]
pub struct MeantDeclaration {
    pub extern_name: ExternName,
    means: Means,
    params: Vec<Shape>,
    ret: Shape,
    /// The externs the term calls, each with the term or law it is read by.
    named: FxHashMap<QualifiedRef, NamedExtern>,
}

#[derive(Debug, Clone)]
struct NamedExtern {
    written: ExternName,
    reading: Reading,
}

#[derive(Debug, Clone)]
enum Reading {
    Means(Means),
    Payload,
    Opaque,
}

impl MeantDeclaration {
    pub fn means(&self) -> &Means {
        &self.means
    }

    /// The same handler, held to another term: a sabotage a test states.
    pub fn held_to(self, means: Means) -> Self {
        MeantDeclaration { means, ..self }
    }
}

struct Named {
    vec: QualifiedRef,
    map: QualifiedRef,
    set: QualifiedRef,
    decimal: QualifiedRef,
}

/// # Panics
/// Where an instance stating a term has a type outside what [`Shape`]
/// models: a declaration added without a model fails here instead of going
/// unchecked.
pub fn meaning_declarations(
    interner: &Interner,
    registries: Vec<Registry<AcvusRuntime>>,
) -> Vec<MeantDeclaration> {
    let externs = Externs::combine(registries, interner).expect("the registries combine");
    let named = Named {
        vec: QualifiedRef::root(interner.intern("Vec")),
        map: QualifiedRef::root(interner.intern("HashMap")),
        set: QualifiedRef::root(interner.intern("HashSet")),
        decimal: QualifiedRef::root(interner.intern("Decimal")),
    };
    let mut found = Vec::new();
    for function in &externs.functions {
        let FnKind::Extern {
            bounds, instances, ..
        } = &function.kind
        else {
            continue;
        };
        // A concrete instance's type variables are its own, bounded by no
        // `OneOf` (its requirements are the only bounds it states); the
        // generic instance's are the function's scheme's.
        let concrete = instances
            .concrete
            .iter()
            .map(|instance| (&instance.ty, None, instance.means.as_ref()));
        let generic = instances
            .generic
            .as_ref()
            .map(|generic| (&function.ty, Some(bounds.as_slice()), generic.means.as_ref()));
        for (ty, bounds, means) in concrete.chain(generic) {
            let Some(means) = means else {
                continue;
            };
            let extern_name = ExternName::of(interner, function.qref);
            let PolyTy::Fn { params, ret, .. } = ty else {
                panic!("`{}` states a term and is no function", extern_name.name)
            };
            let key_var = means.entry.and_then(|entry| match &params[entry.key].ty {
                TyTerm::Var(v) => Some(*v),
                TyTerm::Ref(_, lent) => match &*lent.ty() {
                    TyTerm::Var(v) => Some(*v),
                    _ => None,
                },
                _ => None,
            });
            let shape = |ty: &PolyTy| {
                shape_of(ty, bounds, key_var, &named).unwrap_or_else(|why| {
                    panic!("the term of `{}` has no model: {why}", extern_name.name)
                })
            };
            let named = means
                .named_externs()
                .into_iter()
                .map(|qref| (qref, named_extern(interner, &externs, qref)))
                .collect();
            found.push(MeantDeclaration {
                params: params.iter().map(|param| shape(&param.ty)).collect(),
                ret: shape(ret),
                means: means.clone(),
                extern_name,
                named,
            });
        }
    }
    found.sort_by(|a, b| a.extern_name.cmp(&b.extern_name));
    found
}

struct InstanceStatement<'a> {
    means: Option<&'a Means>,
    laws: &'a Laws,
}

/// How the reader in `acvus_mir::laws` reads a call a term holds: by the
/// one term all its instances state, by the `payload` law all state, or as
/// an opaque call.
fn named_extern(interner: &Interner, externs: &Externs<AcvusRuntime>, qref: QualifiedRef) -> NamedExtern {
    let written = ExternName::of(interner, qref);
    let function = externs
        .functions
        .iter()
        .find(|function| function.qref == qref)
        .unwrap_or_else(|| panic!("a term calls `{}`, which combining admitted", written.name));
    let FnKind::Extern { instances, .. } = &function.kind else {
        panic!("a term calls `{}`, which is no extern", written.name)
    };
    let stated: Vec<InstanceStatement<'_>> = instances
        .concrete
        .iter()
        .map(|at| InstanceStatement {
            means: at.means.as_ref(),
            laws: &at.laws,
        })
        .chain(instances.generic.as_ref().map(|at| InstanceStatement {
            means: at.means.as_ref(),
            laws: &at.laws,
        }))
        .collect();
    let first = stated.first().and_then(|at| at.means);
    let reading = match first {
        Some(first) if stated.iter().all(|at| at.means == Some(first)) => {
            Reading::Means(first.clone())
        }
        _ if stated
            .iter()
            .all(|at| at.means.is_none() && *at.laws == Laws::Payload) =>
        {
            Reading::Payload
        }
        _ => Reading::Opaque,
    };
    NamedExtern { written, reading }
}

fn shape_of(
    ty: &PolyTy,
    bounds: Option<&[TyVarBound]>,
    key_var: Option<u32>,
    named: &Named,
) -> Result<Shape, String> {
    let shape = |ty: &PolyTy| shape_of(ty, bounds, key_var, named);
    Ok(match ty {
        TyTerm::Var(v) if Some(*v) == key_var => Shape::Key,
        TyTerm::Var(v) => match bounds.map(|bounds| bounds.get(*v as usize)) {
            None | Some(Some(TyVarBound::Any)) => Shape::Parametric,
            other => return Err(format!("a variable bounded by {other:?}")),
        },
        TyTerm::Int(width) => Shape::Int(*width),
        TyTerm::Float => Shape::Float,
        TyTerm::Bool => Shape::Bool,
        TyTerm::Char => Shape::Char,
        TyTerm::String => Shape::Text,
        TyTerm::Option(inner) => Shape::Option(Box::new(shape(inner)?)),
        TyTerm::Ref(mutability, lent) => match &*lent.ty() {
            TyTerm::Str => Shape::StrView,
            lent => Shape::Lent(*mutability, Box::new(shape(lent)?)),
        },
        TyTerm::UserDefined { id, type_args, .. } if *id == named.vec => match type_args.as_slice() {
            [element] => Shape::VecOf(Box::new(shape(&element.ty())?)),
            _ => return Err("a `Vec` of other than one argument".into()),
        },
        TyTerm::UserDefined { id, .. } if *id == named.map => Shape::Table(TableKind::Map),
        TyTerm::UserDefined { id, .. } if *id == named.set => Shape::Table(TableKind::Set),
        TyTerm::UserDefined { id, .. } if *id == named.decimal => Shape::Decimal,
        other => return Err(format!("a position of type {other:?}")),
    })
}

/// A key stands at one of three values, any other variable at one of two:
/// the fewest that tell a key met from a key missed, and one value from
/// another.
const KEYS: [i128; 3] = [0, 1, 2];
const VALUES: [i128; 2] = [0, 1];
/// Written through a `&mut` a call returns: no value the domain holds, so
/// the table printed after shows where it landed.
const WRITTEN_THROUGH: i128 = 7;
const MOST_ENTRIES: usize = 3;

/// The values a position is written as, in the script's source.
fn samples(shape: &Shape) -> Result<Vec<String>, String> {
    let written = |values: &[&str]| values.iter().map(|v| (*v).to_string()).collect();
    Ok(match shape {
        Shape::Key => KEYS.iter().map(i128::to_string).collect(),
        Shape::Parametric => VALUES.iter().map(i128::to_string).collect(),
        Shape::Int(IntTy::I64) => written(&["0", "1", "-1", "i64::MAX()", "i64::MIN()"]),
        Shape::Int(IntTy::U8) => written(&["0u8", "1u8", "255u8"]),
        Shape::Int(other) => return Err(format!("no sample of {other:?}")),
        Shape::Float => written(&["0.0", "-0.0", "1.5", "f64::MAX()", "f64::NAN()"]),
        Shape::Bool => written(&["true", "false"]),
        Shape::Char => written(&["'a'", "'\\u{10FFFF}'"]),
        Shape::Text | Shape::StrView => written(&["\"\" + \"\"", "\"\" + \"a\"", "\"\" + \",\""]),
        Shape::Decimal => ["0", "1.0", "1.00", "-2.5"]
            .iter()
            .map(|text| format!("result::unwrap(std::decimal(\"{text}\".to_string()))"))
            .collect(),
        Shape::Option(inner) => std::iter::once("None".to_string())
            .chain(samples(inner)?.into_iter().map(|v| format!("Some({v})")))
            .collect(),
        Shape::VecOf(inner) => sequences(&samples(inner)?, 2)
            .into_iter()
            .map(|items| match items.is_empty() {
                true => "vec::new()".to_string(),
                false => format!("vec([{}])", items.join(", ")),
            })
            .collect(),
        Shape::Lent(_, lent) => samples(lent)?,
        Shape::Table(_) => return Err("a table is built, not sampled".into()),
    })
}

/// A `String` expression printing a value of `shape`, read from `e`.
fn printed(shape: &Shape, e: &str) -> Result<String, String> {
    Ok(match shape {
        Shape::Key | Shape::Parametric => format!("({e} + 0).to_string()"),
        Shape::Int(_) | Shape::Float | Shape::Char | Shape::Decimal => format!("({e}).to_string()"),
        Shape::Bool => format!("(if {e} {{ \"t\" + \"\" }} else {{ \"f\" + \"\" }})"),
        Shape::Text => format!("(\"<\" + ({e}).to_string() + \">\")"),
        Shape::StrView => format!("(\"<\" + {e} + \">\")"),
        Shape::Option(inner) => format!(
            "(match {e} {{ Some(held) => \"S(\" + {} + \")\", None => \"N\" + \"\" }})",
            printed(inner, "held")?
        ),
        Shape::VecOf(inner) => format!(
            "(\"[\" + ({e}).as_iter().map(|item| -> {}).join(\",\".to_string()) + \"]\")",
            printed(inner, "(*item)")?
        ),
        Shape::Lent(_, lent) => printed(lent, &format!("(*{e})"))?,
        Shape::Table(_) => return Err("a table is printed by `printed_table`".into()),
    })
}

fn printed_table(kind: TableKind, e: &str) -> String {
    // `+ 0` settles an empty table's key and value types at `i64`, the
    // type every variable of the model stands at.
    let keys = format!("{e}.keys().map(|k| -> ((*k) + 0).to_string()).join(\",\".to_string())");
    match kind {
        TableKind::Map => format!(
            "(\"{{\" + {keys} + \"=\" + {e}.values().map(|v| -> ((*v) + 0).to_string()).join(\",\".to_string()) + \"}}\")"
        ),
        TableKind::Set => format!(
            "(\"{{\" + {e}.as_iter().map(|k| -> ((*k) + 0).to_string()).join(\",\".to_string()) + \"}}\")"
        ),
    }
}

// -- Cases ----------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Case {
    pub declaration: ExternName,
    pub call: String,
    pub expected: Expected,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expected {
    /// The value of a program running the term.
    Program(String),
    /// The text the call's program prints, computed here.
    Printed(String),
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub declaration: String,
    pub positions: Vec<String>,
    pub cases: Vec<Case>,
}

fn called(declaration: &MeantDeclaration, args: &[String]) -> String {
    let ExternName { namespace, name } = &declaration.extern_name;
    match namespace {
        Some(ns) => format!("{ns}::{name}({})", args.join(", ")),
        None => format!("{name}({})", args.join(", ")),
    }
}

pub fn plan(declaration: &MeantDeclaration) -> Result<Plan, String> {
    match declaration.means.entry {
        None => plan_valued(declaration),
        Some(_) => plan_entry(declaration),
    }
}

/// A term of no entry: the call's program and the term's program, each
/// printing its value.
fn plan_valued(declaration: &MeantDeclaration) -> Result<Plan, String> {
    let mut positions = Vec::new();
    let mut choices: Vec<Vec<String>> = Vec::new();
    for (at, shape) in declaration.params.iter().enumerate() {
        let values = samples(shape)?;
        positions.push(format!("parameter {at} {shape:?}: {} values", values.len()));
        choices.push(values);
    }
    let mut cases = Vec::new();
    for args in product(&choices) {
        let lets: String = args
            .iter()
            .enumerate()
            .map(|(at, value)| format!("let a{at} = {value};\n"))
            .collect();
        let passed: Vec<String> = declaration
            .params
            .iter()
            .enumerate()
            .map(|(at, shape)| match shape {
                Shape::Lent(Mutability::Mut, _) => format!("&mut a{at}"),
                Shape::Lent(Mutability::Shared, _) | Shape::StrView => format!("&a{at}"),
                _ => format!("a{at}"),
            })
            .collect();
        let call = printed(&declaration.ret, &called(declaration, &passed))?;
        let term = source_of(declaration)?;
        cases.push(Case {
            declaration: declaration.extern_name.clone(),
            call: format!("{lets}{call}\n"),
            expected: Expected::Program(format!(
                "{lets}{}\n",
                printed(&declaration.ret, &term)?
            )),
        });
    }
    Ok(Plan {
        declaration: declaration.extern_name.name.clone(),
        positions,
        cases,
    })
}

/// The term as script source, each parameter read from its `let`.
fn source_of(declaration: &MeantDeclaration) -> Result<String, String> {
    if !declaration.means.stmts.is_empty() {
        return Err("a term of no entry with statements".into());
    }
    term_source(&declaration.means.result, &declaration.named)
}

/// A script's `match` arm binds a name, and `Some(_)` is no pattern it
/// writes.
const UNUSED_PAYLOAD: &str = "unused_payload";

fn term_source(
    term: &Term,
    named: &FxHashMap<QualifiedRef, NamedExtern>,
) -> Result<String, String> {
    let each = |items: &[Term]| -> Result<Vec<String>, String> {
        items.iter().map(|item| term_source(item, named)).collect()
    };
    Ok(match term {
        Term::Param(at) | Term::Lent(at) => format!("a{at}"),
        Term::Local(at) => format!("b{at}"),
        Term::Some(inner) => format!("Some({})", term_source(inner, named)?),
        Term::None => "None".to_string(),
        Term::Const(literal) => literal_source(literal)?,
        Term::Not(inner) => format!("!({})", term_source(inner, named)?),
        Term::Neg(inner) => format!("(-({}))", term_source(inner, named)?),
        Term::Binary { op, left, right } => format!(
            "({} {} {})",
            term_source(left, named)?,
            op_source(*op)?,
            term_source(right, named)?
        ),
        Term::Array(items) => format!("[{}]", each(items)?.join(", ")),
        // `vec::vec(..)` in a script names the signature's own identity
        // type, which no `push` takes, and `vec(..)` is the call a script
        // writes; so a call is written by its bare name.
        Term::Call { name, args } => {
            let written = named
                .get(name)
                .ok_or("a call the declaration's term does not name")?;
            format!("{}({})", written.written.name, each(args)?.join(", "))
        }
        Term::Match {
            scrutinee,
            some,
            then,
            none,
        } => {
            let binder = some.map_or(UNUSED_PAYLOAD.to_string(), |at| format!("b{at}"));
            format!(
                "(match {} {{ Some({binder}) => {}, None => {} }})",
                term_source(scrutinee, named)?,
                term_source(then, named)?,
                term_source(none, named)?
            )
        }
        Term::Entry | Term::LendEntry(_) => return Err("an entry in a term of no entry".into()),
    })
}

fn op_source(op: BinOp) -> Result<&'static str, String> {
    Ok(match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
        BinOp::Eq => "==",
        BinOp::Neq => "!=",
        BinOp::Lt => "<",
        BinOp::Lte => "<=",
        BinOp::Gt => ">",
        BinOp::Gte => ">=",
        BinOp::And => "&&",
        BinOp::Or => "||",
        other => return Err(format!("no term writes the operator {other:?}")),
    })
}

/// A constant as script source: the literals a term's parser writes.
fn literal_source(literal: &acvus_ast::Literal) -> Result<String, String> {
    Ok(match literal {
        acvus_ast::Literal::Int(held) => format!("({held})"),
        acvus_ast::Literal::Bool(held) => held.to_string(),
        other => return Err(format!("no source of the constant {other:?}")),
    })
}

// -- A term over an entry ---------------------------------------------------

/// The table as the model holds it: entries in insertion order.
#[derive(Debug, Clone, PartialEq)]
struct Table {
    kind: TableKind,
    entries: Vec<TableEntry>,
}

#[derive(Debug, Clone, PartialEq)]
struct TableEntry {
    key: i128,
    value: i128,
}

impl Table {
    fn at(&self, key: i128) -> Option<&TableEntry> {
        self.entries.iter().find(|entry| entry.key == key)
    }

    /// The entry's value as a term reads it: a map's `Option<V>`, a set's
    /// `bool`.
    fn read(&self, key: i128) -> Model {
        let held = self.at(key);
        match self.kind {
            TableKind::Map => Model::Opt(held.map(|entry| Box::new(Model::Int(entry.value)))),
            TableKind::Set => Model::Bool(held.is_some()),
        }
    }

    /// A store keeps a present key where it stands, puts a new one last,
    /// and takes an absent one out keeping the others' order, as
    /// `IndexMap::shift_remove` does.
    fn store(&mut self, key: i128, value: &Model) -> Result<(), String> {
        let present = match (self.kind, value) {
            (TableKind::Map, Model::Opt(Some(held))) => match **held {
                Model::Int(held) => Some(held),
                _ => return Err(format!("a map entry stored {value:?}")),
            },
            (TableKind::Map, Model::Opt(None)) | (TableKind::Set, Model::Bool(false)) => None,
            (TableKind::Set, Model::Bool(true)) => Some(0),
            _ => return Err(format!("a {:?} entry stored {value:?}", self.kind)),
        };
        let at = self.entries.iter().position(|entry| entry.key == key);
        match (present, at) {
            (Some(value), Some(at)) => self.entries[at].value = value,
            (Some(value), None) => self.entries.push(TableEntry { key, value }),
            (None, Some(at)) => {
                self.entries.remove(at);
            }
            (None, None) => {}
        }
        Ok(())
    }

    fn printed(&self) -> String {
        let keys: Vec<String> = self.entries.iter().map(|entry| entry.key.to_string()).collect();
        match self.kind {
            TableKind::Map => {
                let values: Vec<String> =
                    self.entries.iter().map(|entry| entry.value.to_string()).collect();
                format!("{{{}={}}}", keys.join(","), values.join(","))
            }
            TableKind::Set => format!("{{{}}}", keys.join(",")),
        }
    }
}

/// Every table of at most [`MOST_ENTRIES`] entries over [`KEYS`], in every
/// insertion order, a map's entries at every value of [`VALUES`].
fn tables(kind: TableKind) -> Vec<Table> {
    let orders: Vec<Vec<i128>> = sequences(&KEYS, MOST_ENTRIES)
        .into_iter()
        .filter(|keys| {
            let mut sorted = keys.clone();
            sorted.sort_unstable();
            sorted.dedup();
            sorted.len() == keys.len()
        })
        .collect();
    let mut found = Vec::new();
    for keys in orders {
        let values: Vec<Vec<i128>> = match kind {
            TableKind::Map => product(&vec![VALUES.to_vec(); keys.len()]),
            TableKind::Set => vec![vec![0; keys.len()]],
        };
        for values in values {
            found.push(Table {
                kind,
                entries: keys
                    .iter()
                    .zip(values)
                    .map(|(key, value)| TableEntry { key: *key, value })
                    .collect(),
            });
        }
    }
    found
}

struct Eval<'d> {
    declaration: &'d MeantDeclaration,
    table: Table,
    key: i128,
    nested_depth: usize,
}

/// Terms that name each other in a cycle have no model; the bound ends the
/// walk rather than detecting the cycle.
const DEEPEST_NESTING: usize = 8;

#[derive(Debug, Clone, PartialEq)]
enum Ran {
    Value(Model),
    /// A call the term makes traps where the handler must trap too.
    Trapped,
}

impl Eval<'_> {
    fn block(&mut self, means: &Means, args: &[Option<Model>]) -> Result<Ran, String> {
        let mut locals: Vec<Option<Model>> = Vec::new();
        for stmt in &means.stmts {
            match stmt {
                Stmt::Let { local, value } => {
                    let Ran::Value(value) = self.term(value, args, &mut locals)? else {
                        return Ok(Ran::Trapped);
                    };
                    bind(&mut locals, *local, value);
                }
                Stmt::Store(value) => {
                    let Ran::Value(value) = self.term(value, args, &mut locals)? else {
                        return Ok(Ran::Trapped);
                    };
                    self.table.store(self.key, &value)?;
                }
            }
        }
        self.term(&means.result, args, &mut locals)
    }

    fn term(
        &mut self,
        term: &Term,
        args: &[Option<Model>],
        locals: &mut Vec<Option<Model>>,
    ) -> Result<Ran, String> {
        let value = |held: Option<&Model>, what: &str| {
            held.cloned()
                .map(Ran::Value)
                .ok_or_else(|| format!("the term reads {what}, which it does not hold"))
        };
        macro_rules! eval {
            ($term:expr) => {
                match self.term($term, args, locals)? {
                    Ran::Value(value) => value,
                    Ran::Trapped => return Ok(Ran::Trapped),
                }
            };
        }
        Ok(Ran::Value(match term {
            Term::Param(at) | Term::Lent(at) => {
                return value(args.get(*at).and_then(Option::as_ref), "a parameter");
            }
            Term::Local(at) => return value(locals.get(*at).and_then(Option::as_ref), "a local"),
            Term::Entry => self.table.read(self.key),
            Term::LendEntry(_) => match self.table.read(self.key) {
                Model::Opt(None) => Model::Opt(None),
                Model::Opt(Some(_)) => Model::Opt(Some(Box::new(Model::EntryRef))),
                other => return Err(format!("the term lends an entry of {other:?}")),
            },
            Term::Some(inner) => Model::Opt(Some(Box::new(eval!(inner)))),
            Term::None => Model::Opt(None),
            Term::Const(acvus_ast::Literal::Bool(held)) => Model::Bool(*held),
            Term::Const(acvus_ast::Literal::Int(held)) => Model::Int(*held),
            Term::Const(other) => return Err(format!("no model of the constant {other:?}")),
            Term::Not(inner) => match eval!(inner) {
                Model::Bool(held) => Model::Bool(!held),
                other => return Err(format!("`!` of {other:?}")),
            },
            Term::Neg(inner) => match eval!(inner) {
                Model::Int(held) => match in_i64(-held) {
                    Some(value) => Model::Int(value),
                    None => return Ok(Ran::Trapped),
                },
                other => return Err(format!("`-` of {other:?}")),
            },
            Term::Binary { op, left, right } => {
                let left = eval!(left);
                let right = eval!(right);
                match binary(*op, &left, &right)? {
                    Some(value) => value,
                    None => return Ok(Ran::Trapped),
                }
            }
            Term::Array(_) => return Err("no model of an array in a term over an entry".into()),
            Term::Call { name, args: passed } => {
                let mut values: Vec<Model> = Vec::new();
                for arg in passed {
                    values.push(eval!(arg));
                }
                let named = self
                    .declaration
                    .named
                    .get(name)
                    .ok_or("a call the declaration's term does not name")?;
                match (&named.reading, values.as_slice()) {
                    (Reading::Payload, [Model::Opt(Some(held))]) => (**held).clone(),
                    (Reading::Payload, [Model::Opt(None)]) => return Ok(Ran::Trapped),
                    (Reading::Means(means), _) if means.entry.is_none() => {
                        if self.nested_depth >= DEEPEST_NESTING {
                            return Err("terms nest deeper than the model reads".into());
                        }
                        let means = means.clone();
                        let nested: Vec<Option<Model>> = values.into_iter().map(Some).collect();
                        self.nested_depth += 1;
                        let ran = self.block(&means, &nested);
                        self.nested_depth -= 1;
                        return ran;
                    }
                    (reading, _) => {
                        return Err(format!(
                            "no model of `{}` read as {reading:?}",
                            named.written.name
                        ));
                    }
                }
            }
            Term::Match {
                scrutinee,
                some,
                then,
                none,
            } => match eval!(scrutinee) {
                Model::Opt(Some(held)) => {
                    if let Some(local) = some {
                        bind(locals, *local, *held);
                    }
                    eval!(then)
                }
                Model::Opt(None) => eval!(none),
                other => return Err(format!("`match` over {other:?}")),
            },
        }))
    }
}

/// A value the model's positions hold is an `i64` (every variable stands
/// at it); an operation leaving that width traps, as the language's does.
fn in_i64(value: i128) -> Option<i128> {
    (i128::from(i64::MIN)..=i128::from(i64::MAX))
        .contains(&value)
        .then_some(value)
}

/// `None` where the language's operation traps.
fn binary(op: BinOp, left: &Model, right: &Model) -> Result<Option<Model>, String> {
    Ok(Some(match (op, left, right) {
        (BinOp::Add, Model::Int(a), Model::Int(b)) => return Ok(in_i64(a + b).map(Model::Int)),
        (BinOp::Sub, Model::Int(a), Model::Int(b)) => return Ok(in_i64(a - b).map(Model::Int)),
        (BinOp::Mul, Model::Int(a), Model::Int(b)) => return Ok(in_i64(a * b).map(Model::Int)),
        (BinOp::Div | BinOp::Mod, Model::Int(_), Model::Int(0)) => return Ok(None),
        (BinOp::Div, Model::Int(a), Model::Int(b)) => return Ok(in_i64(a / b).map(Model::Int)),
        (BinOp::Mod, Model::Int(a), Model::Int(b)) => return Ok(in_i64(a % b).map(Model::Int)),
        (BinOp::Lt, Model::Int(a), Model::Int(b)) => Model::Bool(a < b),
        (BinOp::Lte, Model::Int(a), Model::Int(b)) => Model::Bool(a <= b),
        (BinOp::Gt, Model::Int(a), Model::Int(b)) => Model::Bool(a > b),
        (BinOp::Gte, Model::Int(a), Model::Int(b)) => Model::Bool(a >= b),
        (BinOp::And, Model::Bool(a), Model::Bool(b)) => Model::Bool(*a && *b),
        (BinOp::Or, Model::Bool(a), Model::Bool(b)) => Model::Bool(*a || *b),
        (BinOp::Eq, a, b) if !matches!(a, Model::EntryRef) && !matches!(b, Model::EntryRef) => {
            Model::Bool(a == b)
        }
        (BinOp::Neq, a, b) if !matches!(a, Model::EntryRef) && !matches!(b, Model::EntryRef) => {
            Model::Bool(a != b)
        }
        (op, a, b) => return Err(format!("no model of {a:?} {op:?} {b:?}")),
    }))
}

fn bind(locals: &mut Vec<Option<Model>>, local: usize, value: Model) {
    if locals.len() <= local {
        locals.resize(local + 1, None);
    }
    locals[local] = Some(value);
}

/// What the call's program prints of a value of `shape`, read after the
/// call from `table`.
fn model_printed(shape: &Shape, value: &Model, table: &Table, key: i128) -> Result<String, String> {
    Ok(match (shape, value) {
        (Shape::Key | Shape::Parametric | Shape::Int(_), Model::Int(held)) => held.to_string(),
        (Shape::Bool, Model::Bool(held)) => match held {
            true => "t".to_string(),
            false => "f".to_string(),
        },
        (Shape::Option(inner), Model::Opt(Some(held))) => {
            format!("S({})", model_printed(inner, held, table, key)?)
        }
        (Shape::Option(_), Model::Opt(None)) => "N".to_string(),
        (Shape::Lent(_, lent), Model::EntryRef) => {
            let entry = table.at(key).ok_or("a reference to an entry the table no longer holds")?;
            model_printed(lent, &Model::Int(entry.value), table, key)?
        }
        (Shape::Lent(_, lent), held) => model_printed(lent, held, table, key)?,
        (shape, value) => return Err(format!("{value:?} is no value of {shape:?}")),
    })
}

/// A term over an entry: every table, key and argument, the call's program
/// printing its result, then the table after it, and the model printing the
/// same from the term.
fn plan_entry(declaration: &MeantDeclaration) -> Result<Plan, String> {
    let entry = declaration.means.entry.ok_or("no entry")?;
    let Shape::Lent(table_mutability, table_shape) = &declaration.params[entry.table] else {
        return Err("the entry's table is no reference parameter".into());
    };
    let Shape::Table(kind) = **table_shape else {
        return Err(format!("the entry's table is a {table_shape:?}"));
    };
    let mut positions = vec![format!(
        "table {kind:?}: every order of at most {MOST_ENTRIES} of {} keys, values {VALUES:?}",
        KEYS.len()
    )];
    let mut choices: Vec<Vec<String>> = Vec::new();
    let mut free: Vec<usize> = Vec::new();
    for (at, shape) in declaration.params.iter().enumerate() {
        if at == entry.table {
            continue;
        }
        let values = samples(shape)?;
        positions.push(format!("parameter {at} {shape:?}: {} values", values.len()));
        choices.push(values);
        free.push(at);
    }
    let constructed = match kind {
        TableKind::Map => "map::hash_map()",
        TableKind::Set => "set::hash_set()",
    };
    let writes_through = match &declaration.ret {
        Shape::Lent(Mutability::Mut, _) => WriteThrough::Reference,
        Shape::Option(inner) if matches!(**inner, Shape::Lent(Mutability::Mut, _)) => {
            WriteThrough::Payload
        }
        _ => WriteThrough::Nothing,
    };
    let mut cases = Vec::new();
    for table in tables(kind) {
        for args in product(&choices) {
            let mut lets = format!("let m = {constructed};\n");
            for held in &table.entries {
                lets += &match kind {
                    TableKind::Map => format!("map::insert(&mut m, {}, {});\n", held.key, held.value),
                    TableKind::Set => format!("set::insert(&mut m, {});\n", held.key),
                };
            }
            let mut passed: Vec<String> = Vec::new();
            let mut modelled: Vec<Option<Model>> = Vec::new();
            let mut free_args = free.iter().zip(&args);
            for (at, shape) in declaration.params.iter().enumerate() {
                if at == entry.table {
                    passed.push(match table_mutability {
                        Mutability::Mut => "&mut m".to_string(),
                        Mutability::Shared => "&m".to_string(),
                    });
                    modelled.push(None);
                    continue;
                }
                let (_, written) = free_args.next().ok_or("an argument past the plan")?;
                lets += &format!("let a{at} = {written};\n");
                passed.push(match shape {
                    Shape::Lent(Mutability::Mut, _) => format!("&mut a{at}"),
                    Shape::Lent(Mutability::Shared, _) => format!("&a{at}"),
                    _ => format!("a{at}"),
                });
                let value: i128 = written
                    .parse()
                    .map_err(|_| format!("the model reads `{written}` as no integer"))?;
                modelled.push(Some(Model::Int(value)));
            }
            let Some(Model::Int(key)) = modelled[entry.key] else {
                return Err("the key is no integer".into());
            };
            let call = called(declaration, &passed);
            let observed = printed(&declaration.ret, "r")?;
            let write_through = match writes_through {
                WriteThrough::Reference => format!("*r = {WRITTEN_THROUGH};\n"),
                WriteThrough::Payload => format!(
                    "if std::is_some(r) {{ let p = std::unwrap(r); *p = {WRITTEN_THROUGH}; }};\n"
                ),
                WriteThrough::Nothing => String::new(),
            };
            let source = format!(
                "{lets}let r = {call};\nlet o = {observed};\n{write_through}o + \"|\" + {}\n",
                printed_table(kind, "m")
            );
            let mut eval = Eval {
                declaration,
                table: table.clone(),
                key,
                nested_depth: 0,
            };
            let expected = match eval.block(&declaration.means, &modelled)? {
                Ran::Trapped => Expected::Printed(TRAPPED.to_string()),
                Ran::Value(result) => {
                    let shown = model_printed(&declaration.ret, &result, &eval.table, key)?;
                    let refers = matches!(result, Model::EntryRef)
                        || result == Model::Opt(Some(Box::new(Model::EntryRef)));
                    if writes_through != WriteThrough::Nothing && refers {
                        let at = eval
                            .table
                            .entries
                            .iter()
                            .position(|held| held.key == key)
                            .ok_or("a reference to an entry the table does not hold")?;
                        eval.table.entries[at].value = WRITTEN_THROUGH;
                    }
                    Expected::Printed(format!("{shown}|{}", eval.table.printed()))
                }
            };
            cases.push(Case {
                declaration: declaration.extern_name.clone(),
                call: source,
                expected,
            });
        }
    }
    Ok(Plan {
        declaration: declaration.extern_name.name.clone(),
        positions,
        cases,
    })
}

/// A `&mut` the call returns is written through after its value is
/// printed, so the table printed next shows which entry it named.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteThrough {
    Reference,
    Payload,
    Nothing,
}

/// What a case expects where the term traps: the handler's run traps.
const TRAPPED: &str = "<trapped>";

// -- Running ----------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Broken {
    Differ {
        source: String,
        expected: String,
        ran: Outcome,
    },
    DidNotRun {
        source: String,
        outcome: Outcome,
    },
}

pub fn check(
    plans: &[Plan],
    registries: fn() -> Vec<Registry<AcvusRuntime>>,
    workers: usize,
) -> Vec<Broken> {
    let cases: Vec<&Case> = plans.iter().flat_map(|plan| &plan.cases).collect();
    let next = AtomicUsize::new(0);
    let broken: Mutex<Vec<Broken>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                while let Some(case) = cases.get(next.fetch_add(1, Ordering::Relaxed)) {
                    if let Some(why) = check_case(case, registries) {
                        broken.lock().expect("no worker panics holding it").push(why);
                    }
                }
            });
        }
    });
    broken.into_inner().expect("no worker panics holding it")
}

fn ran_to(source: &str, registries: fn() -> Vec<Registry<AcvusRuntime>>) -> Result<Outcome, Broken> {
    let outcome = run_at(source, Opt::None, registries).outcome;
    match outcome {
        Outcome::Value(_) | Outcome::RunPanicked(_) => Ok(outcome),
        other => Err(Broken::DidNotRun {
            source: source.to_string(),
            outcome: other,
        }),
    }
}

fn check_case(case: &Case, registries: fn() -> Vec<Registry<AcvusRuntime>>) -> Option<Broken> {
    let ran = match ran_to(&case.call, registries) {
        Ok(ran) => ran,
        Err(broken) => return Some(broken),
    };
    let expected = match &case.expected {
        Expected::Program(source) => match ran_to(source, registries) {
            Ok(Outcome::Value(value)) => value,
            Ok(Outcome::RunPanicked(_)) => TRAPPED.to_string(),
            Ok(other) | Err(Broken::DidNotRun { outcome: other, .. }) => {
                return Some(Broken::DidNotRun {
                    source: source.clone(),
                    outcome: other,
                });
            }
            Err(differ) => return Some(differ),
        },
        Expected::Printed(text) if text == TRAPPED => TRAPPED.to_string(),
        Expected::Printed(text) => serde_json::Value::from(text.clone()).to_string(),
    };
    let got = match &ran {
        Outcome::Value(value) => value.clone(),
        _ => TRAPPED.to_string(),
    };
    (got != expected).then(|| Broken::Differ {
        source: case.call.clone(),
        expected,
        ran,
    })
}
