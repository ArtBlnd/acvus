//! Several hosts compiled into one graph, where a host calls another's entry
//! as a function along a DAG (RFC-0095).

use std::collections::{BTreeMap, VecDeque};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use acvus_ast::rename::{Renames, rename_script, rename_template};
use acvus_ast::{AstId, Expr, RefKind, Script, Span};
use acvus_extern::Registry;
use acvus_mir::graph::infer::InferResult;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{Bindings, ParsedAst, QualifiedRef, RecoveredAst};
use acvus_mir::ty::{ObjectTy, ParamTerm, PolyTy, Ty, TyTerm, TypeRegistry};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashSet;

use crate::executor::Executor;
use crate::host::{
    Access, AsyncAccess, Cause, EntryDecl, EntryDeclaration, Host, HostError, HostParts, Namespace,
    Origin, Program, Refusal, ResolvedShape, SyncAccess, compile,
};
use crate::init::{InitGiven, InitSource};
use crate::runtime::AcvusRuntime;

pub struct HostGraph<A = SyncAccess> {
    interner: Interner,
    registries: Vec<Registry<AcvusRuntime>>,
    hosts: Vec<GraphHost>,
    exposures: Vec<Exposure>,
    runs: Vec<Run>,
    access: PhantomData<fn() -> A>,
}

struct GraphHost {
    name: String,
    parts: HostParts,
}

struct Exposure {
    into: String,
    name: String,
    from: String,
    entry: String,
}

struct Run {
    host: String,
    entry: String,
}

pub(crate) struct Exposed {
    qref: QualifiedRef,
    name: String,
    inputs: Vec<ExposedInput>,
}

struct ExposedInput {
    in_graph: Astr,
    written: String,
}

impl HostGraph<SyncAccess> {
    pub fn new(registries: Vec<Registry<AcvusRuntime>>) -> Self {
        HostGraph {
            interner: Interner::new(),
            registries,
            hosts: Vec::new(),
            exposures: Vec::new(),
            runs: Vec::new(),
            access: PhantomData,
        }
    }

    pub fn async_access(self) -> HostGraph<AsyncAccess> {
        HostGraph {
            interner: self.interner,
            registries: self.registries,
            hosts: self.hosts,
            exposures: self.exposures,
            runs: self.runs,
            access: PhantomData,
        }
    }
}

impl<A> HostGraph<A>
where
    A: Access,
{
    /// `build` is handed a host that declares no registries: the graph's
    /// are every host's (RFC-0095 rule 1).
    pub fn host<F>(mut self, name: &str, build: F) -> Result<Self, HostError>
    where
        F: FnOnce(Host) -> Result<Host, HostError>,
    {
        let host = build(Host::in_graph(&self.interner))?;
        self.hosts.push(GraphHost {
            name: name.to_owned(),
            parts: host.parts,
        });
        Ok(self)
    }

    /// Host `into`'s scripts call `from_host`'s entry `from_entry` as
    /// `as_name({ field: value, … })`, one field per input of the entry.
    pub fn expose(mut self, into: &str, as_name: &str, from_host: &str, from_entry: &str) -> Self {
        self.exposures.push(Exposure {
            into: into.to_owned(),
            name: as_name.to_owned(),
            from: from_host.to_owned(),
            entry: from_entry.to_owned(),
        });
        self
    }

    /// The program runs `host`'s entry `entry`, which `Scope::entry` names
    /// as `host/entry`.
    pub fn entry(mut self, host: &str, entry: &str) -> Self {
        self.runs.push(Run {
            host: host.to_owned(),
            entry: entry.to_owned(),
        });
        self
    }

    pub fn compile<E>(self, executor: E) -> Result<Program<A>, HostError>
    where
        E: Executor + 'static,
    {
        let HostGraph {
            interner,
            registries,
            hosts,
            exposures,
            runs,
            access: _,
        } = self;
        let plan = match Plan::of(&hosts, &exposures, &runs) {
            Ok(plan) => plan,
            Err(mut structural) => {
                let mut refusals: Vec<Refusal> = hosts
                    .into_iter()
                    .flat_map(|host| under_host(&host.name, host.parts.parse_refusals))
                    .collect();
                refusals.append(&mut structural);
                return Err(HostError::Refused(refusals));
            }
        };
        let (merged, exposed) = merge(&interner, registries, hosts, &plan);
        let compiled =
            compile(merged, A::GRAPH, Arc::new(executor), &exposed).map_err(HostError::Refused)?;
        Ok(Program::of(compiled.running(&plan.runs)))
    }
}

struct Plan {
    exposed: Vec<PlannedExposure>,
    runs: FxHashSet<String>,
}

struct PlannedExposure {
    into: String,
    name: String,
    from: String,
    entry: String,
    inputs: ResolvedShape,
    declared: PolyTy,
}

impl Plan {
    fn of(hosts: &[GraphHost], exposures: &[Exposure], runs: &[Run]) -> Result<Plan, Vec<Refusal>> {
        let mut refusals = Vec::new();
        let mut by_name: BTreeMap<&str, &HostParts> = BTreeMap::new();
        for host in hosts {
            if by_name.insert(&host.name, &host.parts).is_some() {
                let message = format!("the graph is given the host `{}` twice", host.name);
                refusals.push(Refusal::of(None, message));
            }
            if host.parts.opt != Opt::Full {
                let message = format!(
                    "the host `{}` sets its own optimization, and a graph compiles its hosts as one",
                    host.name
                );
                refusals.push(Refusal::of(None, message));
            }
        }
        let missing_host = |host: &str| Refusal::of(None, format!("the graph has no host `{host}`"));

        let mut exposed = Vec::new();
        let mut exposed_as: FxHashSet<String> = FxHashSet::default();
        for exposure in exposures {
            let Exposure { into, name, from, entry } = exposure;
            let refused = |why: String| {
                Refusal::of(None, format!("`{from}/{entry}` is exposed to `{into}` as `{name}`, {why}"))
            };
            if !exposed_as.insert(Namespace::Host(into.clone()).key(name)) {
                refusals.push(refused(format!("and another entry is exposed to `{into}` under that name")));
            }
            let Some(into_parts) = by_name.get(into.as_str()) else {
                refusals.push(missing_host(into));
                continue;
            };
            if into_parts.entries.iter().any(|declared| declared.name == *name) {
                refusals.push(refused(format!("which is already the name of an entry of `{into}`")));
            }
            let Some(from_parts) = by_name.get(from.as_str()) else {
                refusals.push(missing_host(from));
                continue;
            };
            match from_parts.entries.iter().find(|declared| declared.name == *entry) {
                Some(EntryDecl {
                    declaration: EntryDeclaration::Typed { inputs, declared },
                    ..
                }) => exposed.push(PlannedExposure {
                    into: into.clone(),
                    name: name.clone(),
                    from: from.clone(),
                    entry: entry.clone(),
                    inputs: inputs.clone(),
                    declared: declared.clone(),
                }),
                Some(_) => refusals.push(refused(
                    "and it declares no inputs and result for a call to be checked against".to_owned(),
                )),
                None => refusals.push(refused(format!("and `{from}` has no entry `{entry}`"))),
            }
        }
        refusals.extend(cycles(exposures).into_iter().map(|hosts| {
            let mut written = hosts.clone();
            written.push(hosts[0].clone());
            let message = format!(
                "the exposures form the cycle {}, and the calls of a host graph form a DAG",
                written.join(" → ")
            );
            Refusal {
                cause: Some(Cause::Cycle { hosts }),
                ..Refusal::of(None, message)
            }
        }));

        let mut running = FxHashSet::default();
        for Run { host, entry } in runs {
            let Some(parts) = by_name.get(host.as_str()) else {
                refusals.push(missing_host(host));
                continue;
            };
            let refused = |why: String| Refusal::of(None, format!("the graph runs `{host}/{entry}`, {why}"));
            if !parts.entries.iter().any(|declared| declared.name == *entry) {
                refusals.push(refused(format!("and `{host}` has no entry `{entry}`")));
                continue;
            }
            if let Some(exposure) = exposures.iter().find(|x| x.from == *host && x.entry == *entry) {
                refusals.push(refused(format!(
                    "which is exposed to `{}` as `{}`, and only the hosts an entry is exposed to run it",
                    exposure.into, exposure.name
                )));
                continue;
            }
            running.insert(format!("{host}/{entry}"));
        }
        match refusals.is_empty() {
            true => Ok(Plan {
                exposed,
                runs: running,
            }),
            false => Err(refusals),
        }
    }
}

fn cycles(exposures: &[Exposure]) -> Vec<Vec<String>> {
    let mut calls: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for exposure in exposures {
        calls.entry(&exposure.into).or_default().push(&exposure.from);
        calls.entry(&exposure.from).or_default();
    }
    for callees in calls.values_mut() {
        callees.sort_unstable();
        callees.dedup();
    }
    let mut found = Vec::new();
    let mut reported: FxHashSet<&str> = FxHashSet::default();
    for &start in calls.keys() {
        if reported.contains(start) {
            continue;
        }
        let Some(path) = shortest_return(&calls, start) else {
            continue;
        };
        reported.extend(strongly_connected_with(&calls, start));
        found.push(path.into_iter().map(str::to_owned).collect());
    }
    found
}

fn shortest_return<'n>(calls: &BTreeMap<&'n str, Vec<&'n str>>, start: &'n str) -> Option<Vec<&'n str>> {
    let mut reached_from: BTreeMap<&str, &str> = BTreeMap::new();
    let mut frontier = VecDeque::from([start]);
    while let Some(at) = frontier.pop_front() {
        for &next in &calls[at] {
            if next == start {
                let mut path = vec![at];
                while let Some(&before) = reached_from.get(path[path.len() - 1]) {
                    path.push(before);
                }
                path.reverse();
                return Some(path);
            }
            if next != start && !reached_from.contains_key(next) {
                reached_from.insert(next, at);
                frontier.push_back(next);
            }
        }
    }
    None
}

fn strongly_connected_with<'n>(calls: &BTreeMap<&'n str, Vec<&'n str>>, start: &'n str) -> Vec<&'n str> {
    let reached = |from: &'n str| -> FxHashSet<&'n str> {
        let mut seen = FxHashSet::from_iter([from]);
        let mut stack = vec![from];
        while let Some(at) = stack.pop() {
            for &next in &calls[at] {
                if seen.insert(next) {
                    stack.push(next);
                }
            }
        }
        seen
    };
    let forward = reached(start);
    calls
        .keys()
        .copied()
        .filter(|host| forward.contains(host) && reached(host).contains(start))
        .collect()
}

fn under_host(host: &str, refusals: Vec<Refusal>) -> impl Iterator<Item = Refusal> + use<> {
    let under = Namespace::Host(host.to_owned());
    refusals.into_iter().map(move |refusal| Refusal {
        origin: refusal.origin.map(|origin| match origin {
            Origin::Entry(name) => Origin::Entry(under.key(&name)),
            Origin::Init(key) => Origin::Init(under.key(&key)),
            Origin::Binding(name) => Origin::Binding(under.key(&name)),
        }),
        ..refusal
    })
}

struct HostNames<'n> {
    interner: &'n Interner,
    host: &'n Namespace,
    functions: &'n FxHashSet<Astr>,
}

impl HostNames<'_> {
    fn under(&self, name: Astr) -> Astr {
        self.interner.intern(&self.host.key(self.interner.resolve(name)))
    }

    fn ast(&mut self, ast: &mut ParsedAst) {
        match ast {
            ParsedAst::Script(script) => rename_script(script, self),
            ParsedAst::Template(template) => rename_template(template, self),
            ParsedAst::Recovered(RecoveredAst::Script(script)) => rename_script(script, self),
            ParsedAst::Recovered(RecoveredAst::Template(template)) => rename_template(template, self),
        }
    }
}

impl Renames for HostNames<'_> {
    fn context(&mut self, name: Astr) -> Astr {
        self.under(name)
    }

    fn input(&mut self, name: Astr) -> Astr {
        self.under(name)
    }

    fn value(&mut self, name: Astr) -> Astr {
        match self.functions.contains(&name) {
            true => self.under(name),
            false => name,
        }
    }
}

fn merge(
    interner: &Interner,
    registries: Vec<Registry<AcvusRuntime>>,
    hosts: Vec<GraphHost>,
    plan: &Plan,
) -> (HostParts, Vec<Exposed>) {
    let mut bindings = Bindings::default();
    let mut entries: Vec<EntryDecl> = Vec::new();
    let mut inits: Vec<InitSource> = Vec::new();
    let mut parse_refusals: Vec<Refusal> = Vec::new();
    let mut refusals: Vec<Refusal> = Vec::new();
    let mut parse = Duration::ZERO;
    for GraphHost { name: host, parts } in hosts {
        let namespace = Namespace::Host(host.clone());
        let functions: FxHashSet<Astr> = parts
            .entries
            .iter()
            .map(|declared| declared.name.as_str())
            .chain(
                plan.exposed
                    .iter()
                    .filter(|exposure| exposure.into == host)
                    .map(|exposure| exposure.name.as_str()),
            )
            .map(|name| interner.intern(name))
            .collect();
        let mut names = HostNames {
            interner,
            host: &namespace,
            functions: &functions,
        };
        for (name, value) in parts.bindings.iter() {
            if let Err(refused) = bindings.bind(names.under(name), value.clone()) {
                let origin = Origin::Binding(namespace.key(interner.resolve(name)));
                refusals.push(Refusal::of(Some(origin), refused.to_string()));
            }
        }
        for mut declared in parts.entries {
            names.ast(&mut declared.ast);
            entries.push(EntryDecl {
                name: namespace.key(&declared.name),
                inputs_under: namespace.clone(),
                ..declared
            });
        }
        for InitSource { key, given } in parts.inits {
            let given = match given {
                InitGiven::Source(mut ast) => {
                    names.ast(&mut ast);
                    InitGiven::Source(ast)
                }
                InitGiven::Rust(rust) => InitGiven::Rust(rust),
            };
            inits.push(InitSource {
                key: namespace.key(&key),
                given,
            });
        }
        parse_refusals.extend(under_host(&host, parts.parse_refusals));
        refusals.extend(under_host(&host, parts.refusals));
        parse += parts.parse;
    }

    let mut exposed: Vec<Exposed> = Vec::new();
    for exposure in &plan.exposed {
        let from = Namespace::Host(exposure.from.clone());
        let entry = from.key(&exposure.entry);
        let positional = format!("{entry}/call");
        if !exposed.iter().any(|already| already.name == entry) {
            let inputs: Vec<ExposedInput> = exposure
                .inputs
                .fields
                .iter()
                .map(|(field, _)| {
                    let written = interner.resolve(*field).to_owned();
                    ExposedInput {
                        in_graph: interner.intern(&from.key(&written)),
                        written,
                    }
                })
                .collect();
            let params = inputs
                .iter()
                .zip(&exposure.inputs.fields)
                .map(|(input, (_, ty))| ParamTerm::new(input.in_graph, ty.clone()))
                .collect();
            entries.push(EntryDecl {
                name: positional.clone(),
                bare_name: None,
                inputs_under: Namespace::Root,
                ast: ParsedAst::Script(script(call_of(interner, &entry, Vec::new()))),
                declaration: EntryDeclaration::Positional {
                    params,
                    ret: exposure.declared.clone(),
                },
            });
            exposed.push(Exposed {
                qref: QualifiedRef::root(interner.intern(&entry)),
                name: entry,
                inputs,
            });
        }
        let argument = TyTerm::Object(ObjectTy::written(exposure.inputs.fields.iter().cloned().collect()));
        entries.push(EntryDecl {
            name: Namespace::Host(exposure.into.clone()).key(&exposure.name),
            bare_name: Some(exposure.name.clone()),
            inputs_under: Namespace::Root,
            ast: ParsedAst::Script(script(unpacked_call(interner, &exposure.inputs, &positional))),
            declaration: EntryDeclaration::Positional {
                params: vec![ParamTerm::new(interner.intern(EXPOSURE_ARGUMENT), argument)],
                ret: exposure.declared.clone(),
            },
        });
    }
    let mut named: FxHashSet<&str> = FxHashSet::default();
    let mut twice: Vec<String> = entries
        .iter()
        .filter(|declared| !named.insert(&declared.name))
        .map(|declared| declared.name.clone())
        .collect();
    twice.sort_unstable();
    twice.dedup();
    refusals.extend(
        twice
            .into_iter()
            .map(|name| Refusal::of(None, format!("two functions of the graph are named `{name}`"))),
    );

    let merged = HostParts {
        interner: interner.clone(),
        registries,
        bindings,
        entries,
        inits,
        parse_refusals,
        refusals,
        opt: Opt::Full,
        parse,
    };
    (merged, exposed)
}

const EXPOSURE_ARGUMENT: &str = "arguments";

fn script(tail: Expr) -> Script {
    Script {
        id: AstId::alloc(),
        stmts: Vec::new(),
        tail: Some(Box::new(tail)),
        span: Span::ZERO,
    }
}

fn ident(interner: &Interner, name: &str, ref_kind: RefKind) -> Expr {
    Expr::Ident {
        id: AstId::alloc(),
        name: QualifiedRef::root(interner.intern(name)),
        ref_kind,
        span: Span::ZERO,
    }
}

fn call_of(interner: &Interner, callee: &str, args: Vec<Expr>) -> Expr {
    Expr::FuncCall {
        id: AstId::alloc(),
        func: Box::new(ident(interner, callee, RefKind::Value)),
        args,
        span: Span::ZERO,
    }
}

fn unpacked_call(interner: &Interner, inputs: &ResolvedShape, positional: &str) -> Expr {
    let args = inputs
        .fields
        .iter()
        .map(|(field, _)| Expr::FieldAccess {
            id: AstId::alloc(),
            object: Box::new(ident(interner, EXPOSURE_ARGUMENT, RefKind::ExternParam)),
            field: *field,
            span: Span::ZERO,
        })
        .collect();
    call_of(interner, positional, args)
}

// -- Nothing callable crosses (RFC-0095 rule 4) ----------------------------

pub(crate) fn crossing_refusals(
    interner: &Interner,
    types: &TypeRegistry,
    inferred: &InferResult,
    exposed: &[Exposed],
) -> Vec<Refusal> {
    let reader = CallableReader { interner, types };
    let mut refusals = Vec::new();
    for entry in exposed {
        let Some(outcome) = inferred.outcomes.get(&entry.qref) else {
            continue;
        };
        let meta = outcome.meta();
        for input in &meta.inputs {
            let written = match entry.inputs.iter().find(|exposed| exposed.in_graph == input.name) {
                Some(exposed) => exposed.written.clone(),
                None => interner.resolve(input.name).to_owned(),
            };
            if let Some(held) = reader.first_in(&input.ty, format!("${written}")) {
                refusals.push(held.refusal(interner, &entry.name, "input"));
            }
        }
        if let Ty::Fn { ret, .. } = &meta.ty
            && let Some(held) = reader.first_in(ret, "result".to_owned())
        {
            refusals.push(held.refusal(interner, &entry.name, "result"));
        }
    }
    refusals
}

struct Callable {
    at: String,
    ty: Ty,
    kind: CallableKind,
}

enum CallableKind {
    Function,
    /// A task handle's value is the run of the code that spawned it.
    TaskHandle,
    UnstatedExtension,
}

impl Callable {
    fn refusal(&self, interner: &Interner, entry: &str, part: &str) -> Refusal {
        let ty = self.ty.display(interner);
        let what = match self.kind {
            CallableKind::Function => format!("the function type `{ty}`"),
            CallableKind::TaskHandle => format!("the task handle `{ty}`, which runs the code that made it"),
            CallableKind::UnstatedExtension => format!(
                "the extension type `{ty}`, whose declaration does not state that it holds no function"
            ),
        };
        let message = format!(
            "the exposed entry `{entry}` holds at `{}` of its {part} {what}, and nothing callable crosses \
             from one host to another",
            self.at
        );
        Refusal::of(Some(Origin::Entry(entry.to_owned())), message)
    }
}

struct CallableReader<'r> {
    interner: &'r Interner,
    types: &'r TypeRegistry,
}

impl CallableReader<'_> {
    fn first_in(&self, ty: &Ty, at: String) -> Option<Callable> {
        let found = |kind| {
            Some(Callable {
                at: at.clone(),
                ty: ty.clone(),
                kind,
            })
        };
        match ty {
            TyTerm::Int(_)
            | TyTerm::Float
            | TyTerm::Char
            | TyTerm::String
            | TyTerm::Bool
            | TyTerm::Unit
            | TyTerm::Never
            | TyTerm::Order
            | TyTerm::Str
            | TyTerm::Error(_) => None,
            TyTerm::Fn { .. } => found(CallableKind::Function),
            TyTerm::Handle(_) => found(CallableKind::TaskHandle),
            TyTerm::UserDefined { id, type_args, .. } => match self.types.get(*id).may_hold_a_function {
                true => found(CallableKind::UnstatedExtension),
                false => type_args
                    .iter()
                    .enumerate()
                    .find_map(|(n, arg)| self.first_in(&arg.ty(), format!("{at}<{n}>"))),
            },
            TyTerm::Array(element, _) | TyTerm::Slice(element) => self.first_in(element, format!("{at}[_]")),
            TyTerm::Option(inner) => self.first_in(inner, format!("{at}::Some")),
            TyTerm::Result(ok, err) => self
                .first_in(ok, format!("{at}::Ok"))
                .or_else(|| self.first_in(err, format!("{at}::Err"))),
            TyTerm::Tuple(elements) => elements
                .iter()
                .enumerate()
                .find_map(|(n, element)| self.first_in(element, format!("{at}.{n}"))),
            TyTerm::Object(object) => {
                let mut fields: Vec<(&str, &Ty)> = object
                    .iter()
                    .map(|(name, field)| (self.interner.resolve(*name), field))
                    .collect();
                fields.sort_unstable_by_key(|(name, _)| *name);
                fields
                    .into_iter()
                    .find_map(|(name, field)| self.first_in(field, format!("{at}.{name}")))
            }
            TyTerm::Enum { variants, .. } => {
                let mut payloads: Vec<(&str, &Ty)> = variants
                    .iter()
                    .filter_map(|(tag, payload)| Some((self.interner.resolve(*tag), payload.as_deref()?)))
                    .collect();
                payloads.sort_unstable_by_key(|(tag, _)| *tag);
                payloads
                    .into_iter()
                    .find_map(|(tag, payload)| self.first_in(payload, format!("{at}::{tag}")))
            }
            TyTerm::Ref(_, referent) => self.first_in(&referent.ty(), format!("*{at}")),
            TyTerm::Var(never) => match *never {},
        }
    }
}
