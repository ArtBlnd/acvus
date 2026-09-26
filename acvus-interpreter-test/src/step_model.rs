//! RFC-0099 rule 1's promise, checked declaration by declaration: every
//! pipeline up to a bound runs to the same value, or traps with the same
//! message, whether the declaration's handler runs it (`Opt::None`) or the
//! loop `optimize::fusion` writes from its step (`Opt::Full`).
//!
//! The ground is parametricity. A declaration generic in a type variable
//! cannot inspect a value of it, so its behaviour at one small finite type
//! is its behaviour at every type: each such variable stands at `i64`
//! restricted to `0..size`, and every source up to the bound's length and
//! every closure table over that domain is run. A position of a concrete
//! type the declaration can read, the `i64` a `sum` adds, is sampled.
//!
//! The declarations are the registry's: every one stating a step is
//! checked, and `stepping_declarations` fails on one whose types this
//! module cannot model rather than leaving it unchecked.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::{Externs, FnKind, Laws, Registry};
use acvus_interpreter::{
    AcvusRuntime, Executable, Interpreter, InterpreterContext, PrepareCtx, SequentialExecutor,
    prepare_module,
};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{ParsedAst, QualifiedRef};
use acvus_mir::ir::{Callee, InstKind};
use acvus_mir::step::{AdaptorFlow, Step, Term};
use acvus_mir::ty::{IntTy, PolyTy, RequirementSig, Ty, TyTerm, TyVarBound};
use acvus_utils::Interner;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::corpus::{Outcome, render};

pub mod means;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Carrier {
    Parametric,
    Concrete(Concrete),
    Lent(Box<Carrier>),
    VecOf(Box<Carrier>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Concrete {
    I64,
    F64,
    Bool,
    Text,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamModel {
    Closure { args: Vec<Carrier>, ret: Carrier },
    Value(Carrier),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelledParam {
    pub at: usize,
    pub model: ParamModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instantiation {
    pub element: Carrier,
    pub params_but_the_stream: Vec<ModelledParam>,
}

/// An extern as a script names it: a `QualifiedRef` holds its interner's
/// symbols, and each run interns its own.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExternName {
    pub namespace: Option<String>,
    pub name: String,
}

impl ExternName {
    pub fn of(interner: &Interner, qref: QualifiedRef) -> Self {
        ExternName {
            namespace: qref.namespace.map(|ns| interner.resolve(ns).to_string()),
            name: interner.resolve(qref.name).to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Declaration {
    pub extern_name: ExternName,
    pub name: String,
    pub step: Step,
    pub instantiations: Vec<Instantiation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bound {
    pub parametric_domain_size: usize,
    pub source_len: usize,
    pub closure_vec_len: usize,
}

struct DeclaredAt<'f> {
    ty: &'f PolyTy,
    laws: &'f Laws,
    requires: &'f [RequirementSig],
}

/// # Panics
/// Where a stepping declaration's types are outside what [`Carrier`]
/// models: a declaration added without a model fails here instead of going
/// unchecked.
pub fn stepping_declarations(
    interner: &Interner,
    registries: Vec<Registry<AcvusRuntime>>,
) -> Vec<Declaration> {
    let externs = Externs::combine(registries, interner).expect("the registries combine");
    let vec_id = QualifiedRef::root(interner.intern("Vec"));
    let mut found = Vec::new();
    for function in &externs.functions {
        let FnKind::Extern {
            bounds,
            instances,
            requires,
            ..
        } = &function.kind
        else {
            continue;
        };
        // A declaration that is no signature's instance states its
        // requirements on its scheme, as it states its effect bounds there,
        // and its instances carry none.
        let concrete = instances.concrete.iter().map(|instance| DeclaredAt {
            ty: &instance.ty,
            laws: &instance.laws,
            requires: match instance.requires.is_empty() {
                true => requires.as_slice(),
                false => &instance.requires,
            },
        });
        let generic = instances.generic.as_ref().map(|generic| DeclaredAt {
            ty: &function.ty,
            laws: &generic.laws,
            requires,
        });
        let mut declaration: Option<Declaration> = None;
        for at in concrete.chain(generic) {
            let Laws::Step(step) = at.laws else {
                continue;
            };
            let name = interner.resolve(function.qref.name).to_string();
            let model = Model { bounds, vec_id };
            let instantiations = model
                .instantiations(step, at.ty, at.requires)
                .unwrap_or_else(|why| panic!("the step of `{name}` has no model: {why}"));
            let declaration = declaration.get_or_insert_with(|| Declaration {
                extern_name: ExternName::of(interner, function.qref),
                name,
                step: step.clone(),
                instantiations: Vec::new(),
            });
            for instantiation in instantiations {
                if !declaration.instantiations.contains(&instantiation) {
                    declaration.instantiations.push(instantiation);
                }
            }
        }
        found.extend(declaration);
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found
}

struct Model<'b> {
    bounds: &'b [TyVarBound],
    vec_id: QualifiedRef,
}

type ShapeOfBoundedVar = FxHashMap<u32, PolyTy>;

struct BoundedVar {
    var: u32,
    shapes: Vec<PolyTy>,
}

#[derive(Clone)]
struct ShapeChoice {
    var: u32,
    shape: PolyTy,
}

impl Model<'_> {
    /// One instantiation per choice of shape for each variable the
    /// declaration bounds to a set of them, the `T` of `sum` at `i64` and at
    /// `f64`: such a variable is a concrete type the declaration reads.
    fn instantiations(
        &self,
        step: &Step,
        ty: &PolyTy,
        requires: &[RequirementSig],
    ) -> Result<Vec<Instantiation>, String> {
        let mut shaped: Vec<BoundedVar> = Vec::new();
        let mut note = |v: u32| {
            if let Some(TyVarBound::OneOf { shapes }) = self.bounds.get(v as usize)
                && !shaped.iter().any(|seen| seen.var == v)
            {
                shaped.push(BoundedVar {
                    var: v,
                    shapes: shapes.clone(),
                });
            }
            TyTerm::Var(v)
        };
        let visited = std::iter::once(ty).chain(requires.iter().map(|required| &required.pattern));
        for visited in visited {
            visited.map::<acvus_mir::ty::Poly>(
                &mut note,
                &mut acvus_mir::ty::IdentityTerm::Var,
                &mut acvus_mir::ty::EffectTerm::Var,
                &mut acvus_mir::ty::LenTerm::Var,
                &mut acvus_mir::ty::Repr::Var,
                &mut acvus_extern::no_flow_var,
            );
        }
        let choices: Vec<Vec<ShapeChoice>> = shaped
            .into_iter()
            .map(|bounded| {
                bounded
                    .shapes
                    .into_iter()
                    .map(|shape| ShapeChoice {
                        var: bounded.var,
                        shape,
                    })
                    .collect()
            })
            .collect();
        product(&choices)
            .into_iter()
            .map(|choice| {
                let chosen: ShapeOfBoundedVar = choice
                    .into_iter()
                    .map(|choice| (choice.var, choice.shape))
                    .collect();
                self.instantiation(step, ty, requires, &chosen)
            })
            .collect()
    }

    fn instantiation(
        &self,
        step: &Step,
        ty: &PolyTy,
        requires: &[RequirementSig],
        chosen: &ShapeOfBoundedVar,
    ) -> Result<Instantiation, String> {
        let PolyTy::Fn { params, .. } = ty else {
            return Err(format!("its type {ty:?} is no function"));
        };
        let stream = step.stream();
        let required = requires
            .get(stream.requirement)
            .ok_or("its stream's requirement is missing")?;
        let PolyTy::Fn { ret, .. } = &required.pattern else {
            return Err("its stream's requirement is no function".into());
        };
        let PolyTy::Option(element) = &**ret else {
            return Err("its stream's `next` returns no option".into());
        };
        let mut modelled = Vec::new();
        for (at, param) in params.iter().enumerate() {
            if at == stream.param {
                continue;
            }
            let model = match &param.ty {
                PolyTy::Fn { params, ret, .. } => ParamModel::Closure {
                    args: params
                        .iter()
                        .map(|arg| self.carrier(&arg.ty, chosen))
                        .collect::<Result<_, _>>()?,
                    ret: self.carrier(ret, chosen)?,
                },
                other => ParamModel::Value(self.carrier(other, chosen)?),
            };
            modelled.push(ModelledParam { at, model });
        }
        Ok(Instantiation {
            element: self.carrier(element, chosen)?,
            params_but_the_stream: modelled,
        })
    }

    fn carrier(&self, ty: &PolyTy, chosen: &ShapeOfBoundedVar) -> Result<Carrier, String> {
        match ty {
            TyTerm::Var(v) if chosen.contains_key(v) => self.carrier(&chosen[v], chosen),
            TyTerm::Var(v) => match self.bounds.get(*v as usize) {
                Some(TyVarBound::Any) => Ok(Carrier::Parametric),
                Some(bound) => Err(format!("a variable bounded by {bound:?}")),
                None => Err(format!("the variable {v}, which declares no bound")),
            },
            TyTerm::Int(IntTy::I64) => Ok(Carrier::Concrete(Concrete::I64)),
            TyTerm::Float => Ok(Carrier::Concrete(Concrete::F64)),
            TyTerm::Bool => Ok(Carrier::Concrete(Concrete::Bool)),
            TyTerm::String => Ok(Carrier::Concrete(Concrete::Text)),
            TyTerm::Ref(_, lent) => Ok(Carrier::Lent(Box::new(self.carrier(&lent.ty(), chosen)?))),
            TyTerm::UserDefined { id, type_args, .. } if *id == self.vec_id => match type_args.as_slice() {
                [element] => Ok(Carrier::VecOf(Box::new(self.carrier(&element.ty(), chosen)?))),
                _ => Err("a `Vec` of other than one argument".into()),
            },
            other => Err(format!("a position of type {other:?}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Written(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coverage {
    Enumerated,
    Sampled,
}

#[derive(Debug, Clone)]
pub struct Domain {
    pub values: Vec<Written>,
    pub coverage: Coverage,
}

/// A concrete type is sampled at its width's edges, where a wrapping sum
/// wraps and a checked one traps, a value of each sign, and at `f64` the
/// special values IEEE addition treats apart.
fn sample(concrete: Concrete) -> Domain {
    let written = |values: &[&str], coverage: Coverage| Domain {
        values: values.iter().map(|w| Written((*w).to_string())).collect(),
        coverage,
    };
    match concrete {
        Concrete::I64 => written(
            &["0", "1", "-1", "i64::MAX()", "i64::MIN()"],
            Coverage::Sampled,
        ),
        Concrete::F64 => written(
            &["0.0", "-0.0", "1.5", "f64::MAX()", "f64::NAN()"],
            Coverage::Sampled,
        ),
        Concrete::Bool => written(&["true", "false"], Coverage::Enumerated),
        Concrete::Text => written(
            &["\"\".to_string()", "\"a\".to_string()", "\",\".to_string()"],
            Coverage::Sampled,
        ),
    }
}

pub fn domain(carrier: &Carrier, bound: &Bound) -> Domain {
    match carrier {
        Carrier::Parametric => Domain {
            values: (0..bound.parametric_domain_size)
                .map(|v| Written(v.to_string()))
                .collect(),
            coverage: Coverage::Enumerated,
        },
        Carrier::Concrete(concrete) => sample(*concrete),
        Carrier::Lent(lent) => domain(lent, bound),
        Carrier::VecOf(element) => {
            let elements = domain(element, bound);
            Domain {
                values: sequences(&elements.values, bound.closure_vec_len)
                    .into_iter()
                    .map(|items| match items.is_empty() {
                        true => Written("vec::new()".to_string()),
                        false => Written(format!(
                            "vec([{}])",
                            items.iter().map(|w| w.0.as_str()).collect::<Vec<_>>().join(", ")
                        )),
                    })
                    .collect(),
                coverage: elements.coverage,
            }
        }
    }
}

pub(crate) fn sequences<T: Clone>(items: &[T], max: usize) -> Vec<Vec<T>> {
    let mut all: Vec<Vec<T>> = vec![Vec::new()];
    let mut last: Vec<Vec<T>> = vec![Vec::new()];
    for _ in 0..max {
        last = product(&[last, items.iter().map(|item| vec![item.clone()]).collect()])
            .into_iter()
            .map(|parts| parts.concat())
            .collect();
        all.extend(last.iter().cloned());
    }
    all
}

pub(crate) fn product<T: Clone>(sets: &[Vec<T>]) -> Vec<Vec<T>> {
    let mut rows: Vec<Vec<T>> = vec![Vec::new()];
    for set in sets {
        rows = rows
            .iter()
            .flat_map(|row| {
                set.iter().map(move |item| {
                    let mut next = row.clone();
                    next.push(item.clone());
                    next
                })
            })
            .collect();
    }
    rows
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Answer {
    Value(Written),
    Trap,
}

fn closure_tables(args: &[Carrier], ret: &Carrier, bound: &Bound) -> Result<Vec<String>, String> {
    let mut arg_values: Vec<Vec<Written>> = Vec::new();
    for arg in args {
        let arg_domain = domain(arg, bound);
        if arg_domain.coverage != Coverage::Enumerated {
            return Err(format!(
                "a closure argument of {arg:?} has no finite domain to tabulate"
            ));
        }
        arg_values.push(arg_domain.values);
    }
    let cells = product(&arg_values);
    let answers: Vec<Answer> = domain(ret, bound)
        .values
        .into_iter()
        .map(Answer::Value)
        .chain([Answer::Trap])
        .collect();
    let read = |at: usize| match &args[at] {
        Carrier::Lent(_) => format!("(*a{at})"),
        _ => format!("a{at}"),
    };
    let named: Vec<String> = (0..args.len())
        .map(|at| format!("{}.to_string()", read(at)))
        .collect();
    let trap = format!(
        "std::panic(\"trap\".to_string() + \" \" + {})",
        named.join(" + \" \" + ")
    );
    let written = |answer: &Answer| match answer {
        Answer::Value(value) => value.0.clone(),
        Answer::Trap => trap.clone(),
    };
    let names: Vec<String> = (0..args.len()).map(|at| format!("a{at}")).collect();
    let tables = product(&vec![answers; cells.len()])
        .into_iter()
        .map(|row| {
            let mut body = written(&row[row.len() - 1]);
            for (cell, answer) in cells.iter().zip(&row).rev().skip(1) {
                let test: Vec<String> = cell
                    .iter()
                    .enumerate()
                    .map(|(at, value)| format!("{} == {}", read(at), value.0))
                    .collect();
                body = format!(
                    "if {} {{ {} }} else {{ {body} }}",
                    test.join(" && "),
                    written(answer)
                );
            }
            format!("|{}| -> {body}", names.join(", "))
        })
        .collect();
    Ok(tables)
}

#[derive(Debug, Clone)]
pub struct Case {
    pub declaration: ExternName,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub declaration: String,
    pub bound: Bound,
    pub positions: Vec<String>,
    pub cases: Vec<Case>,
}

pub fn plan(declaration: &Declaration, bound: Bound) -> Result<Plan, String> {
    let mut cases = Vec::new();
    let mut positions = Vec::new();
    for instantiation in &declaration.instantiations {
        let elements = domain(&instantiation.element, &bound);
        positions.push(format!(
            "element {:?}: {:?}, {} values, every source of length 0..={}",
            instantiation.element,
            elements.coverage,
            elements.values.len(),
            bound.source_len
        ));
        let mut arg_choices: Vec<Vec<String>> = Vec::new();
        for param in &instantiation.params_but_the_stream {
            match &param.model {
                ParamModel::Closure { args, ret } => {
                    let tables = closure_tables(args, ret, &bound)?;
                    positions.push(format!(
                        "closure {args:?} -> {ret:?}: every table, {} of them, each cell a \
                         value ({:?}) or a trap",
                        tables.len(),
                        domain(ret, &bound).coverage
                    ));
                    arg_choices.push(tables);
                }
                ParamModel::Value(carrier) => {
                    let values = domain(carrier, &bound);
                    positions.push(format!(
                        "value {carrier:?}: {:?}, {} values",
                        values.coverage,
                        values.values.len()
                    ));
                    arg_choices.push(values.values.into_iter().map(|w| w.0).collect());
                }
            }
        }
        let render = match &declaration.step {
            Step::Consumer { .. } => None,
            Step::Adaptor { flow, .. } => Some(match adaptor_output(flow, instantiation)? {
                Carrier::Parametric => "(*y + 0).to_string()",
                _ => "(*y).to_string()",
            }),
        };
        let first = elements.values.first().ok_or("an empty element domain")?;
        for args in product(&arg_choices) {
            for source in sequences(&elements.values, bound.source_len) {
                let setup = match source.is_empty() {
                    true => format!("let xs = vec([{}]);\nxs.clear();\n", first.0),
                    false => format!(
                        "let xs = vec([{}]);\n",
                        source.iter().map(|w| w.0.as_str()).collect::<Vec<_>>().join(", ")
                    ),
                };
                let call = format!("xs.into_iter().{}({})", declaration.name, args.join(", "));
                let source = match render {
                    None => format!("{setup}{call}\n"),
                    Some(rendered) => format!(
                        "{setup}let ys = {call}.collect();\nlet out = \"\".to_string();\n\
                         for y in &ys {{ out = out + {rendered} + \",\"; }}\nout\n"
                    ),
                };
                cases.push(Case {
                    declaration: declaration.extern_name.clone(),
                    source,
                });
            }
        }
    }
    Ok(Plan {
        declaration: declaration.name.clone(),
        bound,
        positions,
        cases,
    })
}

/// The largest bound whose plan stays within `budget` programs, from the
/// widest this module starts at: a shorter source first, then shorter
/// vectors a closure returns. The domain of a parametric variable stays at
/// two values, the fewest that tell two elements apart.
pub fn plan_within(declaration: &Declaration, budget: usize) -> Result<Plan, String> {
    let mut bound = Bound {
        parametric_domain_size: 2,
        source_len: 4,
        closure_vec_len: 2,
    };
    loop {
        let planned = plan(declaration, bound)?;
        if planned.cases.len() <= budget {
            return Ok(planned);
        }
        if bound.source_len > 1 {
            bound.source_len -= 1;
        } else if bound.closure_vec_len > 1 {
            bound.closure_vec_len -= 1;
        } else {
            return Err(format!(
                "even the least bound, {bound:?}, plans {} programs, over the budget of {budget}",
                planned.cases.len()
            ));
        }
    }
}

fn adaptor_output(flow: &AdaptorFlow, instantiation: &Instantiation) -> Result<Carrier, String> {
    let param = |at: usize| {
        instantiation
            .params_but_the_stream
            .iter()
            .find(|param| param.at == at)
            .map(|param| &param.model)
            .ok_or(format!("the step calls parameter {at}, which it does not take"))
    };
    let closure_ret = |at: usize| match param(at)? {
        ParamModel::Closure { ret, .. } => Ok(ret.clone()),
        ParamModel::Value(_) => Err(format!("the step calls parameter {at}, a value")),
    };
    match flow {
        AdaptorFlow::Yield(Term::Elem) => Ok(instantiation.element.clone()),
        AdaptorFlow::Yield(Term::CallClosure { param, .. }) => closure_ret(*param),
        AdaptorFlow::Nest(Term::CallClosure { param, .. }) => match closure_ret(*param)? {
            Carrier::VecOf(element) => Ok(*element),
            other => Err(format!("the step nests a {other:?}")),
        },
        AdaptorFlow::If {
            then, otherwise, ..
        } => adaptor_output(then, instantiation).or_else(|_| adaptor_output(otherwise, instantiation)),
        AdaptorFlow::Skip | AdaptorFlow::Done => Err("the flow passes nothing on".into()),
        AdaptorFlow::Yield(other) | AdaptorFlow::Nest(other) => {
            Err(format!("the step passes on {other:?}, which the model does not type"))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ran {
    pub outcome: Outcome,
    pub calls: Vec<ExternName>,
}

pub fn run_at(source: &str, opt: Opt, registries: fn() -> Vec<Registry<AcvusRuntime>>) -> Ran {
    let interner = Interner::new();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    let stopped = |outcome| Ran {
        outcome,
        calls: Vec::new(),
    };
    let parsed = match acvus_ast::parse_script(&interner, source) {
        Ok(ast) => ParsedAst::Script(ast),
        Err(error) => return stopped(Outcome::Refused(format!("parse: {error:?}"))),
    };
    let compiled = catch_unwind(AssertUnwindSafe(|| {
        crate::check_source(
            &interner,
            parsed,
            &Default::default(),
            registries(),
            Ty::Never,
            opt,
            |_| {},
        )
    }));
    let cr = match compiled {
        Ok(Ok(cr)) => cr,
        Ok(Err(refusal)) => return stopped(Outcome::Refused(refusal.messages.join("; "))),
        Err(panic) => return stopped(Outcome::CompilePanicked(panic_message(panic.as_ref()))),
    };
    let mut calls: FxHashSet<QualifiedRef> = FxHashSet::default();
    for module in cr.modules.values() {
        for body in std::iter::once(&module.main).chain(module.closures.values()) {
            for inst in &body.insts {
                if let InstKind::FunctionCall {
                    callee: Callee::Extern { id, .. },
                    ..
                } = &inst.kind
                {
                    calls.insert(*id);
                }
            }
        }
    }
    let mut calls: Vec<ExternName> = calls
        .into_iter()
        .map(|qref| ExternName::of(&interner, qref))
        .collect();
    calls.sort();
    let mut functions = cr.extern_executables;
    let prepared = {
        let ctx = PrepareCtx {
            interner: &interner,
            externs: &functions,
            context_names: &cr.context_names,
            instances: &cr.instances,
            access: acvus_mir::graph::Access::Sync,
            lowering: acvus_interpreter::Lowering::InPlace,
        };
        catch_unwind(AssertUnwindSafe(|| {
            cr.modules
                .iter()
                .map(|(qref, module)| {
                    prepare_module(module, &ctx)
                        .map(|prepared| (*qref, Executable::Module(Arc::new(prepared))))
                })
                .collect::<Result<Vec<_>, _>>()
        }))
    };
    let prepared = match prepared {
        Ok(Ok(prepared)) => prepared,
        Ok(Err(refusal)) => {
            return Ran {
                outcome: Outcome::Refused(format!("prepare: {refusal}")),
                calls,
            };
        }
        Err(panic) => {
            return Ran {
                outcome: Outcome::PreparePanicked(panic_message(panic.as_ref())),
                calls,
            };
        }
    };
    functions.extend(prepared);
    let shared = InterpreterContext::new(&interner, functions, Arc::new(SequentialExecutor))
        .with_fn_types(cr.fn_types)
        .with_context_names(cr.context_names);
    let no_contexts = Default::default();
    let mut interp = Interpreter::new(shared, cr.entry_qref, no_contexts);
    let outcome = match catch_unwind(AssertUnwindSafe(|| runtime.block_on(interp.execute()))) {
        Ok(Ok(value)) => Outcome::Value(render(&interner, &value).to_string()),
        Ok(Err(acvus_interpreter::HostError::Trapped { message })) => Outcome::RunPanicked(message),
        Ok(Err(other)) => Outcome::RunPanicked(format!("host error: {other:?}")),
        Err(panic) => Outcome::RunPanicked(panic_message(panic.as_ref())),
    };
    Ran { outcome, calls }
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    match (panic.downcast_ref::<&str>(), panic.downcast_ref::<String>()) {
        (Some(text), _) => (*text).to_string(),
        (None, Some(text)) => text.clone(),
        (None, None) => "a panic whose payload is no text".to_string(),
    }
}

#[derive(Debug, Clone)]
pub enum Broken {
    Differ {
        source: String,
        full: Outcome,
        none: Outcome,
    },
    NotFused { source: String },
    DidNotRun { source: String, outcome: Outcome },
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

fn check_case(case: &Case, registries: fn() -> Vec<Registry<AcvusRuntime>>) -> Option<Broken> {
    let full = run_at(&case.source, Opt::Full, registries);
    let none = run_at(&case.source, Opt::None, registries);
    for ran in [&full, &none] {
        if let Outcome::Refused(_)
        | Outcome::CompilePanicked(_)
        | Outcome::PreparePanicked(_)
        | Outcome::Prepared = &ran.outcome
        {
            return Some(Broken::DidNotRun {
                source: case.source.clone(),
                outcome: ran.outcome.clone(),
            });
        }
    }
    if full.calls.contains(&case.declaration) {
        return Some(Broken::NotFused {
            source: case.source.clone(),
        });
    }
    (full.outcome != none.outcome).then(|| Broken::Differ {
        source: case.source.clone(),
        full: full.outcome,
        none: none.outcome,
    })
}
