//! The graph: the units compiled together (programs), each a function at a
//! name the embedding gives, and the host's declarations beside them.
//!
//! Concepts:
//! - A `Unit` is a script or a template (a module is to-be): one function
//!   at a `QualifiedRef` the embedding gives. Its inputs `$name` are its
//!   parameters, passed by name (programs §1, §2, §4).
//! - Its `Interface` is its scheme, generalized exactly as a function value
//!   bound by `let` is (functions §2): a unit is a function whose
//!   environment is empty, so the two are one generalization. It is all
//!   another unit sees of it: its function type over what it generalizes,
//!   its parameters the inputs by name, and every decision its body could
//!   not make as a requirement made again at each call. Its types are
//!   inferred within the unit alone, and a call only instantiates them
//!   (transparency, types §1). It is kept in a store of its own, since a
//!   unit is checked apart from its callers.
//! - An `Instantiation` is a unit at types for its parameters and result:
//!   the scheme instantiated.
//! - An entry is a host-declared function a run may start at, with the
//!   types of its inputs and result (host; `decl::EntryDecl`).
//!
//! Relations:
//! - A unit reaches another only by its whole name, never by a bare one
//!   (programs); a bare name reaches only host declarations (types §6).
//!   The calls between units, read from each unit's syntax by those names,
//!   are the call graph. A cycle in it is refused: only a module's `fn`
//!   recurses (functions §4).
//! - A unit's interface is checked after the interfaces of the units it
//!   calls: `typeck` with its inputs open, generalized at the end
//!   (`typeck::Inputs::Open`). The host's declarations and the callees'
//!   interfaces are what its calls reach.
//! - An instantiation is checked again with its inputs fixed at its types
//!   (`typeck::Inputs::Fixed`), then lowered, taken through the required
//!   steps and the checks, and optimized: one body. The instantiations are
//!   gathered from the entries down: each body's calls of another unit ask
//!   for that unit's instantiation at the call's types.
//! - One source, one `Module`: every body of a unit's instantiations.
//! - Inline, per module, after every instantiation is compiled: a call of
//!   another unit's instantiation may be replaced by that body, its types
//!   carried into the calling module's table.
//!
//! Incremental:
//! - An interface is checked again only where its source, the host's
//!   declarations or an interface it read changed; where the result equals
//!   the one before, no unit that read it is checked again (early cutoff).
//! - An instantiation is compiled again only where its unit's interface
//!   inputs changed or its types did.
//! - Inline is done again for a module only where it or a module it inlines
//!   from changed.
//!
//! Deterministic: units, instantiations and bodies are taken in the order
//! of their names and types, never a hash map's.

use std::collections::{BTreeMap, BTreeSet};

use acvus_utils::{Astr, Interner, QualifiedRef};

use crate::decl::Declarations;
use crate::ir::cfg::{BodyId, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::pipeline::Level;
use crate::ty::{Concrete, Ty, TyKind, Types};
use crate::typeck::{Inputs, Typeck};

/// A unit's source.
pub enum Source {
    Script(acvus_ast::Script),
    Template(acvus_ast::Template),
}

pub struct Unit {
    pub name: QualifiedRef,
    pub source: Source,
}

pub use crate::typeck::{Interface, Interfaces};

#[cfg(test)]
mod tests;

/// The graph a host compiles, kept across compiles: the host's
/// declarations, verified once, and the units by name.
pub struct Graph {
    names: Interner,
    declarations: Declarations,
    units: BTreeMap<QualifiedRef, Unit>,
    compiled: BTreeMap<QualifiedRef, UnitModule>,
}

/// One source's module: one body per instantiation of its unit, each by
/// its parameters' types in the module's table, in the order the
/// instantiations were reached.
pub struct UnitModule {
    pub module: Module,
    pub instantiations: Vec<(Box<[Ty]>, BodyId)>,
}

/// A unit's instantiation's body in its module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    pub module: QualifiedRef,
    pub body: BodyId,
}

/// What a compile gives: each unit's module, and each host-declared entry's
/// body; or every refusal, by unit.
pub struct Compiled<'g> {
    pub modules: Vec<(QualifiedRef, &'g UnitModule)>,
    pub entries: Vec<(QualifiedRef, Entry)>,
}

#[derive(Debug)]
pub struct Refused {
    pub unit: QualifiedRef,
    pub refusals: Vec<Refusal>,
}

/// A refusal of a unit: its checker's (typeck's or check's), or the
/// graph's (a cycle of calls through no `fn`, an entry whose declared
/// types the unit's interface does not admit or which names no unit).
#[derive(Debug)]
pub enum Refusal {
    Typeck(crate::typeck::Refusal),
    Check(crate::check::Refusal),
    Cycle { through: Vec<QualifiedRef> },
    EntryDoesNotFit { entry: QualifiedRef },
}

impl Graph {
    pub fn new(names: Interner, declarations: Declarations) -> Graph {
        Graph { names, declarations, units: BTreeMap::new(), compiled: BTreeMap::new() }
    }

    /// Adds or replaces the unit at `unit.name`.
    pub fn set_unit(&mut self, unit: Unit) {
        self.units.insert(unit.name, unit);
    }

    pub fn remove_unit(&mut self, name: QualifiedRef) {
        self.units.remove(&name);
    }

    /// Compiles every unit at `level`: each interface after its callees',
    /// each instantiation the entries reach, each source's module through
    /// the required steps, the checks and the pipeline.
    pub fn compile(&mut self, level: Level) -> Result<Compiled<'_>, Vec<Refused>> {
        let order = self.order()?;
        let interfaces = self.interfaces(&order)?;
        let mut instantiating = Instantiating { graph: self, interfaces: &interfaces, modules: BTreeMap::new(), refused: Vec::new() };
        let entries = instantiating.entries();
        let Instantiating { mut modules, mut refused, .. } = instantiating;
        for (unit, compiled) in modules.iter_mut() {
            let mut am = AnalysisManager::new(&self.declarations);
            match crate::lower::required::run(&mut compiled.module, &mut am) {
                Ok(()) => crate::passes::pipeline::optimize(&mut compiled.module, &mut am, level),
                Err(refusals) => refused.push(Refused { unit: *unit, refusals: refusals.into_iter().map(Refusal::Check).collect() }),
            }
        }
        if !refused.is_empty() {
            return Err(refused);
        }
        self.compiled = modules;
        Ok(Compiled { modules: self.compiled.iter().map(|(unit, compiled)| (*unit, compiled)).collect(), entries })
    }

    /// Each unit's callees among the units: the qualified names its calls
    /// write that name a unit (programs: a unit reaches another only by its
    /// whole name).
    fn callees(&self, unit: &Unit) -> BTreeSet<QualifiedRef> {
        let nodes = match &unit.source {
            Source::Script(script) => acvus_ast::locate::Nodes::of_script(script),
            Source::Template(template) => acvus_ast::locate::Nodes::of_template(template),
        };
        nodes.callees().iter().copied().filter(|callee| self.units.contains_key(callee)).collect()
    }

    /// Every unit, each after the units it calls; a cycle of calls refused
    /// at each unit on it (functions §4).
    fn order(&self) -> Result<Vec<QualifiedRef>, Vec<Refused>> {
        let calls: BTreeMap<QualifiedRef, BTreeSet<QualifiedRef>> =
            self.units.iter().map(|(name, unit)| (*name, self.callees(unit))).collect();
        let mut order = Vec::new();
        let mut state: BTreeMap<QualifiedRef, Visit> = BTreeMap::new();
        let mut refused = Vec::new();
        for unit in calls.keys() {
            visit(*unit, &calls, &mut state, &mut Vec::new(), &mut order, &mut refused);
        }
        match refused.is_empty() {
            true => Ok(order),
            false => Err(refused),
        }
    }

    /// Each unit's interface, in `order`, each check reaching the
    /// interfaces before it.
    fn interfaces(&self, order: &[QualifiedRef]) -> Result<Interfaces, Vec<Refused>> {
        let mut interfaces = Interfaces::new();
        let mut refused = Vec::new();
        for name in order {
            let typeck = self.check(&self.units[name], &interfaces, Inputs::Open, &mut Types::new());
            match typeck.refusals.is_empty() {
                true => interfaces.set(*name, typeck.interface.expect("an open check gives its interface")),
                false => refused.push(Refused { unit: *name, refusals: typeck.refusals.into_iter().map(Refusal::Typeck).collect() }),
            }
        }
        match refused.is_empty() {
            true => Ok(interfaces),
            false => Err(refused),
        }
    }

    fn check(&self, unit: &Unit, interfaces: &Interfaces, inputs: Inputs<'_>, types: &mut Types<Concrete>) -> Typeck {
        match &unit.source {
            Source::Script(script) => crate::typeck::check_script(&self.names, &self.declarations, interfaces, inputs, types, script),
            Source::Template(template) => crate::typeck::check_template(&self.names, &self.declarations, interfaces, inputs, types, template),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Visit {
    Open,
    Done,
}

/// Depth first over `calls` from `unit`, each unit pushed onto `order`
/// after its callees; a call back to a unit still open is a cycle, refused
/// at each unit on it.
fn visit(
    unit: QualifiedRef,
    calls: &BTreeMap<QualifiedRef, BTreeSet<QualifiedRef>>,
    state: &mut BTreeMap<QualifiedRef, Visit>,
    path: &mut Vec<QualifiedRef>,
    order: &mut Vec<QualifiedRef>,
    refused: &mut Vec<Refused>,
) {
    match state.get(&unit) {
        Some(Visit::Done) => return,
        Some(Visit::Open) => {
            let from = path.iter().position(|on| *on == unit).expect("an open unit is on the path");
            let through: Vec<QualifiedRef> = path[from..].to_vec();
            for on in &through {
                refused.push(Refused { unit: *on, refusals: vec![Refusal::Cycle { through: through.clone() }] });
            }
            return;
        }
        None => {}
    }
    state.insert(unit, Visit::Open);
    path.push(unit);
    for callee in &calls[&unit] {
        visit(*callee, calls, state, path, order, refused);
    }
    path.pop();
    state.insert(unit, Visit::Done);
    order.push(unit);
}

/// The instantiations the entries reach, each checked with its inputs
/// fixed and lowered into its source's module.
struct Instantiating<'g> {
    graph: &'g Graph,
    interfaces: &'g Interfaces,
    modules: BTreeMap<QualifiedRef, UnitModule>,
    refused: Vec<Refused>,
}

impl Instantiating<'_> {
    /// Each host-declared entry, in the order of its name: the unit's
    /// instantiation at the declared types (programs, host).
    fn entries(&mut self) -> Vec<(QualifiedRef, Entry)> {
        let graph = self.graph;
        let mut declared: Vec<&crate::decl::EntryDecl> = graph.declarations.entries().iter().collect();
        declared.sort_by_key(|entry| entry.function);
        let mut entries = Vec::new();
        for entry in declared {
            let unit = entry.function;
            let Some(interface) = self.interfaces.get(unit) else {
                self.refuse(unit, Refusal::EntryDoesNotFit { entry: unit });
                continue;
            };
            let names: BTreeSet<Astr> = entry.inputs.iter().map(|(name, _)| *name).collect();
            if names != interface.params.iter().copied().collect() || names.len() != entry.inputs.len() {
                self.refuse(unit, Refusal::EntryDoesNotFit { entry: unit });
                continue;
            }
            let module = self.module(unit);
            let types: Box<[Ty]> = interface
                .params
                .iter()
                .map(|param| {
                    let (_, declared) = entry.inputs.iter().find(|(name, _)| name == param).expect("each parameter is declared");
                    crate::typeck::declared_concrete(&graph.declarations, module.module.types_mut(), *declared)
                })
                .collect();
            let Some(body) = self.instantiate(unit, types) else {
                continue;
            };
            if let Some(result) = entry.result {
                let module = self.module(unit);
                let result = crate::typeck::declared_concrete(&graph.declarations, module.module.types_mut(), result);
                if module.module.body(body).ret() != result {
                    self.refuse(unit, Refusal::EntryDoesNotFit { entry: unit });
                    continue;
                }
            }
            entries.push((unit, Entry { module: unit, body }));
        }
        entries
    }

    fn module(&mut self, unit: QualifiedRef) -> &mut UnitModule {
        self.modules.entry(unit).or_insert_with(|| UnitModule { module: Module::new(Types::new()), instantiations: Vec::new() })
    }

    fn refuse(&mut self, unit: QualifiedRef, refusal: Refusal) {
        self.refused.push(Refused { unit, refusals: vec![refusal] });
    }

    /// `unit` at `types`, its parameters' types in its module's table: its
    /// body, made once per types; then each unit it calls, at the call's
    /// types. `None` where its check refused it.
    fn instantiate(&mut self, unit: QualifiedRef, types: Box<[Ty]>) -> Option<BodyId> {
        let graph = self.graph;
        let interfaces = self.interfaces;
        let interface = interfaces.get(unit).expect("an instantiated unit has an interface");
        let module = self.module(unit);
        if let Some((_, body)) = module.instantiations.iter().find(|(at, _)| *at == types) {
            return Some(*body);
        }
        let fixed: Vec<(Astr, Ty)> = interface.params.iter().copied().zip(types.iter().copied()).collect();
        let source = &graph.units[&unit];
        let typeck = graph.check(source, interfaces, Inputs::Fixed(&fixed), module.module.types_mut());
        if !typeck.refusals.is_empty() {
            let refusals = typeck.refusals.into_iter().map(Refusal::Typeck).collect();
            self.refused.push(Refused { unit, refusals });
            return None;
        }
        let body = match &source.source {
            Source::Script(script) => crate::lower::lower_script_into(&graph.declarations, &typeck, &mut module.module, script),
            Source::Template(template) => crate::lower::lower_template_into(&graph.declarations, &typeck, &mut module.module, template),
        };
        module.instantiations.push((types, body));
        let mut calls: Vec<(acvus_ast::AstId, QualifiedRef)> = typeck.units.iter().map(|(site, call)| (*site, call.unit)).collect();
        calls.sort_by_key(|(site, _)| *site);
        for (site, callee) in calls {
            let callee_ty = typeck.callees[&site];
            let at = self.called_at(unit, callee, callee_ty);
            self.instantiate(callee, at);
        }
        Some(body)
    }

    /// The types a call in `caller`'s module, of function type `callee_ty`,
    /// instantiates `callee` at: its parameters, moved into the callee's
    /// module's table.
    fn called_at(&mut self, caller: QualifiedRef, callee: QualifiedRef, callee_ty: Ty) -> Box<[Ty]> {
        self.module(callee);
        let mut into = self.modules.remove(&callee).expect("made above");
        let from = self.modules[&caller].module.types();
        let TyKind::Fn { params, .. } = from.kind(callee_ty) else {
            panic!("a unit call's callee is a function type");
        };
        let at = params.iter().map(|param| into.module.types_mut().transplant(from, *param)).collect();
        self.modules.insert(callee, into);
        at
    }
}
