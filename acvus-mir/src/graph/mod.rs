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
//! - Each unit has a revision, raised by `set_unit` and `remove_unit`, and
//!   each interface a revision of its own.
//! - A unit's interface is checked again only where its revision or the
//!   revision of an interface it read changed since it was last checked; a
//!   unit refused is always checked again, since its refusals name sites.
//! - Where an interface checked again is the same as the one before
//!   (`Interface::same`: equal but for the names of its variables and the
//!   sites and spans its requirements carry, which only a refusal reads),
//!   its revision stays, so no unit that read it is checked again (early
//!   cutoff); the new interface is kept, for its sites.
//! - A unit's module is compiled again, whole, only where its revision, the
//!   revision of an interface it read, or the set of its instantiations the
//!   entries reach changed; otherwise it is the one before, with the
//!   instantiations its bodies asked of other units.
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

/// How far a compile takes each module: lowered only; through the
/// required steps and the checks; or on through the pipeline at a level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    Lowered,
    Required,
    Optimized(Level),
}

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
/// declarations, verified once, the units by name, and what the last
/// compile made of each (the module doc's incremental).
pub struct Graph {
    names: Interner,
    declarations: Declarations,
    units: BTreeMap<QualifiedRef, Unit>,
    /// Each unit's revision.
    revisions: BTreeMap<QualifiedRef, u64>,
    /// Each unit's last interface check, by the revisions it read.
    checked: BTreeMap<QualifiedRef, Checked>,
    /// Each unit's last module, by what it was compiled from.
    built: BTreeMap<QualifiedRef, Built>,
    next_revision: u64,
    work: Work,
}

/// What a compile did: the units whose interface it checked and those
/// whose module it built, each in the order it did so.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Work {
    pub checked: Vec<QualifiedRef>,
    pub built: Vec<QualifiedRef>,
}

/// A unit's last interface check: the unit's revision and each callee's
/// interface revision it read, its outcome, and its interface's revision.
struct Checked {
    read: (u64, Vec<(QualifiedRef, u64)>),
    outcome: Result<Interface, Vec<crate::typeck::Refusal>>,
    revision: u64,
}

/// A unit's last module: the revisions it was compiled from and how far,
/// the instantiations it holds, and each call of another unit its bodies
/// make, the callee's parameter types in this module's table.
struct Built {
    read: (u64, Vec<(QualifiedRef, u64)>),
    stop: Stop,
    module: UnitModule,
    asks: Vec<(QualifiedRef, Box<[Ty]>)>,
}

/// A unit's interface this compile, kept in `Interfaces` while the compile
/// runs: the revisions its check read, and its own revision.
struct Held {
    read: (u64, Vec<(QualifiedRef, u64)>),
    revision: u64,
}

/// One source's module: one body per instantiation of its unit, each by
/// its parameters' types in the module's table, in the order the
/// instantiations were reached.
pub struct UnitModule {
    pub module: Module,
    pub instantiations: Vec<(Box<[Ty]>, BodyId)>,
}

impl UnitModule {
    fn empty() -> UnitModule {
        UnitModule { module: Module::new(Types::new()), instantiations: Vec::new() }
    }
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
        Graph {
            names,
            declarations,
            units: BTreeMap::new(),
            revisions: BTreeMap::new(),
            checked: BTreeMap::new(),
            built: BTreeMap::new(),
            next_revision: 0,
            work: Work::default(),
        }
    }

    /// Adds or replaces the unit at `unit.name`.
    pub fn set_unit(&mut self, unit: Unit) {
        let revision = self.raise();
        self.revisions.insert(unit.name, revision);
        self.units.insert(unit.name, unit);
    }

    pub fn remove_unit(&mut self, name: QualifiedRef) {
        let revision = self.raise();
        self.revisions.insert(name, revision);
        self.units.remove(&name);
    }

    /// What the last compile checked and built.
    pub fn work(&self) -> &Work {
        &self.work
    }

    fn raise(&mut self) -> u64 {
        self.next_revision += 1;
        self.next_revision
    }

    /// Compiles every unit up to `stop`: each interface after its
    /// callees', each instantiation the entries reach, each source's module
    /// through the required steps, the checks and the pipeline; each only
    /// where what it was made from changed (the module doc's incremental).
    pub fn compile(&mut self, stop: Stop) -> Result<Compiled<'_>, Vec<Refused>> {
        self.work = Work::default();
        let units = &self.units;
        self.checked.retain(|name, _| units.contains_key(name));
        let order = self.order()?;
        let (interfaces, held) = self.interfaces(&order)?;
        let last = std::mem::take(&mut self.built);
        let mut building = Building {
            graph: self,
            interfaces: &interfaces,
            stop,
            last,
            asked: BTreeMap::new(),
            built: BTreeMap::new(),
            rebuilt: Vec::new(),
            refused: Vec::new(),
        };
        let entries = building.build(&order, &held);
        let Building { built, rebuilt, refused, .. } = building;
        self.built = built;
        self.work.built = rebuilt;
        self.keep(interfaces, held);
        if !refused.is_empty() {
            return Err(refused);
        }
        Ok(Compiled { modules: self.built.iter().map(|(unit, built)| (*unit, &built.module)).collect(), entries })
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
    /// interfaces before it: the last one where the unit's revision and
    /// the revisions of the interfaces it read are those it was checked at
    /// and it was not refused, and a new check otherwise. A new interface
    /// that is the same as the last keeps its revision (early cutoff).
    /// Refused, every interface is kept again before the refusals return.
    fn interfaces(&mut self, order: &[QualifiedRef]) -> Result<(Interfaces, BTreeMap<QualifiedRef, Held>), Vec<Refused>> {
        let mut interfaces = Interfaces::new();
        let mut held: BTreeMap<QualifiedRef, Held> = BTreeMap::new();
        let mut refused = Vec::new();
        for name in order {
            let read = self.read(*name, &held);
            match self.checked.remove(name) {
                Some(Checked { read: last_read, outcome: Ok(interface), revision }) if last_read == read => {
                    interfaces.set(*name, interface);
                    held.insert(*name, Held { read, revision });
                }
                last => {
                    self.work.checked.push(*name);
                    let typeck = self.check(&self.units[name], &interfaces, Inputs::Open, &mut Types::new());
                    if !typeck.refusals.is_empty() {
                        refused.push(Refused { unit: *name, refusals: typeck.refusals.iter().cloned().map(Refusal::Typeck).collect() });
                        let revision = self.raise();
                        self.checked.insert(*name, Checked { read, outcome: Err(typeck.refusals), revision });
                        continue;
                    }
                    let interface = typeck.interface.expect("an open check gives its interface");
                    let revision = match last {
                        Some(Checked { outcome: Ok(before), revision, .. }) if before.same(&interface) => revision,
                        _ => self.raise(),
                    };
                    interfaces.set(*name, interface);
                    held.insert(*name, Held { read, revision });
                }
            }
        }
        if !refused.is_empty() {
            self.keep(interfaces, held);
            return Err(refused);
        }
        Ok((interfaces, held))
    }

    /// What `unit`'s check reads: its revision, and the revision of each
    /// callee's interface, this compile's.
    fn read(&self, unit: QualifiedRef, held: &BTreeMap<QualifiedRef, Held>) -> (u64, Vec<(QualifiedRef, u64)>) {
        let callees = self
            .callees(&self.units[&unit])
            .into_iter()
            .map(|callee| {
                let revision = match held.get(&callee) {
                    Some(held) => held.revision,
                    None => self.checked[&callee].revision,
                };
                (callee, revision)
            })
            .collect();
        (self.revisions[&unit], callees)
    }

    /// Each interface held this compile back in its unit's last check.
    fn keep(&mut self, mut interfaces: Interfaces, held: BTreeMap<QualifiedRef, Held>) {
        for (name, Held { read, revision }) in held {
            let interface = interfaces.remove(name).expect("a held interface is in the compile's interfaces");
            self.checked.insert(name, Checked { read, outcome: Ok(interface), revision });
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

/// The modules of one compile: each unit, before the units it calls, asked
/// for its instantiations by the entries and by the modules of its callers;
/// its last module where what it was compiled from is unchanged, and a new
/// one otherwise.
struct Building<'g> {
    graph: &'g Graph,
    interfaces: &'g Interfaces,
    stop: Stop,
    last: BTreeMap<QualifiedRef, Built>,
    asked: BTreeMap<QualifiedRef, Asked>,
    built: BTreeMap<QualifiedRef, Built>,
    rebuilt: Vec<QualifiedRef>,
    refused: Vec<Refused>,
}

/// The instantiations asked of a unit this compile, each by its types in
/// the table of `into`.
struct Asked {
    into: Into,
    types: Vec<Box<[Ty]>>,
}

/// A host-declared entry that fits its unit, with its inputs' declared
/// types in the order of the unit's parameters.
struct Declared<'g> {
    entry: &'g crate::decl::EntryDecl,
    params: Vec<crate::ty::TyId<crate::ty::Poly>>,
}

/// Where a unit's asked types are kept: in its last module, whose
/// instantiations they are compared with, or in a new module where it has
/// none.
enum Into {
    Last(Built),
    New(UnitModule),
}

impl Into {
    fn types_mut(&mut self) -> &mut Types<Concrete> {
        match self {
            Into::Last(built) => built.module.module.types_mut(),
            Into::New(module) => module.module.types_mut(),
        }
    }
}

impl<'g> Building<'g> {
    /// Every module the entries reach, then each entry's body.
    fn build(&mut self, order: &[QualifiedRef], held: &BTreeMap<QualifiedRef, Held>) -> Vec<(QualifiedRef, Entry)> {
        let graph = self.graph;
        let declared = self.declared();
        for Declared { entry, params } in &declared {
            let asked = self.asked(entry.function);
            let types: Box<[Ty]> =
                params.iter().map(|declared| crate::typeck::declared_concrete(&graph.declarations, asked.into.types_mut(), *declared)).collect();
            if !asked.types.contains(&types) {
                asked.types.push(types);
            }
        }
        for unit in order.iter().rev() {
            let Some(asked) = self.asked.remove(unit) else {
                continue;
            };
            self.module(*unit, asked, &held[unit].read);
        }
        declared.iter().filter_map(|declared| self.entry(declared)).collect()
    }

    /// Each host-declared entry that fits its unit, in the order of its
    /// name (programs, host); each that does not, refused.
    fn declared(&mut self) -> Vec<Declared<'g>> {
        let graph = self.graph;
        let mut declared: Vec<&crate::decl::EntryDecl> = graph.declarations.entries().iter().collect();
        declared.sort_by_key(|entry| entry.function);
        let mut fits = Vec::new();
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
            let params = interface
                .params
                .iter()
                .map(|param| entry.inputs.iter().find(|(name, _)| name == param).expect("each parameter is declared").1)
                .collect();
            fits.push(Declared { entry, params });
        }
        fits
    }

    /// The entry's body in its unit's module, where the module was made
    /// and the body's result is the declared one.
    fn entry(&mut self, Declared { entry, params }: &Declared<'_>) -> Option<(QualifiedRef, Entry)> {
        let unit = entry.function;
        let declarations = &self.graph.declarations;
        let module = &mut self.built.get_mut(&unit)?.module;
        let types: Box<[Ty]> = params.iter().map(|declared| crate::typeck::declared_concrete(declarations, module.module.types_mut(), *declared)).collect();
        let body = module.instantiations.iter().find(|(at, _)| *at == types)?.1;
        let fits = match entry.result {
            Some(result) => module.module.body(body).ret() == crate::typeck::declared_concrete(declarations, module.module.types_mut(), result),
            None => true,
        };
        if !fits {
            self.refuse(unit, Refusal::EntryDoesNotFit { entry: unit });
            return None;
        }
        Some((unit, Entry { module: unit, body }))
    }

    /// What is asked of `unit` so far, its types kept in its last module
    /// where it has one.
    fn asked(&mut self, unit: QualifiedRef) -> &mut Asked {
        let last = &mut self.last;
        self.asked.entry(unit).or_insert_with(|| {
            let into = match last.remove(&unit) {
                Some(built) => Into::Last(built),
                None => Into::New(UnitModule::empty()),
            };
            Asked { into, types: Vec::new() }
        })
    }

    /// `unit` asked for its instantiation at `types`, of the table `from`.
    fn ask(&mut self, unit: QualifiedRef, from: &Types<Concrete>, types: &[Ty]) {
        let asked = self.asked(unit);
        let into = asked.into.types_mut();
        let types: Box<[Ty]> = types.iter().map(|ty| into.transplant(from, *ty)).collect();
        if !asked.types.contains(&types) {
            asked.types.push(types);
        }
    }

    /// `unit`'s module: its last, where it was compiled from `read` up to
    /// this compile's stop and holds exactly the instantiations asked; a
    /// new one otherwise. Then what its bodies ask of other units.
    fn module(&mut self, unit: QualifiedRef, asked: Asked, read: &(u64, Vec<(QualifiedRef, u64)>)) {
        let refused = self.refused.len();
        let built = match asked.into {
            Into::Last(last) if last.read == *read && last.stop == self.stop && holds_exactly(&last.module, &asked.types) => last,
            Into::Last(last) => {
                let mut module = UnitModule::empty();
                let from = last.module.module.types();
                let types = asked.types.iter().map(|types| types.iter().map(|ty| module.module.types_mut().transplant(from, *ty)).collect()).collect();
                self.rebuild(unit, module, types, read)
            }
            Into::New(module) => self.rebuild(unit, module, asked.types, read),
        };
        for (callee, types) in &built.asks {
            self.ask(*callee, built.module.module.types(), types);
        }
        if self.refused.len() == refused {
            self.built.insert(unit, built);
        }
    }

    /// `unit`'s module made anew: its body at each of `asked`, checked with
    /// its inputs fixed and lowered; then through the required steps, the
    /// checks and the pipeline, as far as this compile's stop.
    fn rebuild(&mut self, unit: QualifiedRef, mut module: UnitModule, asked: Vec<Box<[Ty]>>, read: &(u64, Vec<(QualifiedRef, u64)>)) -> Built {
        let graph = self.graph;
        let interfaces = self.interfaces;
        self.rebuilt.push(unit);
        let interface = interfaces.get(unit).expect("an instantiated unit has an interface");
        let source = &graph.units[&unit];
        let mut asks = Vec::new();
        for types in asked {
            let fixed: Vec<(Astr, Ty)> = interface.params.iter().copied().zip(types.iter().copied()).collect();
            let typeck = graph.check(source, interfaces, Inputs::Fixed(&fixed), module.module.types_mut());
            if !typeck.refusals.is_empty() {
                let refusals = typeck.refusals.into_iter().map(Refusal::Typeck).collect();
                self.refused.push(Refused { unit, refusals });
                continue;
            }
            let body = match &source.source {
                Source::Script(script) => crate::lower::lower_script_into(&graph.declarations, &typeck, &mut module.module, script),
                Source::Template(template) => crate::lower::lower_template_into(&graph.declarations, &typeck, &mut module.module, template),
            };
            module.instantiations.push((types, body));
            let mut calls: Vec<(acvus_ast::AstId, QualifiedRef)> = typeck.units.iter().map(|(site, call)| (*site, call.unit)).collect();
            calls.sort_by_key(|(site, _)| *site);
            for (site, callee) in calls {
                let TyKind::Fn { params, .. } = module.module.types().kind(typeck.callees[&site]) else {
                    panic!("a unit call's callee is a function type");
                };
                asks.push((callee, params.clone()));
            }
        }
        if self.stop != Stop::Lowered {
            let mut am = AnalysisManager::new(&graph.declarations);
            match crate::lower::required::run(&mut module.module, &mut am) {
                Ok(()) => {
                    if let Stop::Optimized(level) = self.stop {
                        crate::passes::pipeline::optimize(&mut module.module, &mut am, level);
                    }
                }
                Err(refusals) => self.refused.push(Refused { unit, refusals: refusals.into_iter().map(Refusal::Check).collect() }),
            }
        }
        Built { read: read.clone(), stop: self.stop, module, asks }
    }

    fn refuse(&mut self, unit: QualifiedRef, refusal: Refusal) {
        self.refused.push(Refused { unit, refusals: vec![refusal] });
    }
}

/// Whether `module` holds an instantiation at each of `types`, distinct,
/// and at no others.
fn holds_exactly(module: &UnitModule, types: &[Box<[Ty]>]) -> bool {
    module.instantiations.len() == types.len() && types.iter().all(|types| module.instantiations.iter().any(|(at, _)| at == types))
}
