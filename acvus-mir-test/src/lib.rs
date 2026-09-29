//! The test harness: one shape for every test. A `Case` is the units of a
//! graph, the host's declarations and a stop; it is compiled through the
//! graph as a host compiles, and gives one result, which a test asserts
//! admitted or refused and snapshots.
//!
//! - Admitted, the result is every module in the order of its unit's
//!   name, each printed as its control flow with types, renumbered
//!   canonically (`ir::cfg::print`, `View::CFG`), so two compiles that
//!   differ only in ids print the same.
//! - Refused, the result is every refusal, by unit.
//! - A case runs within one second; one that does not has hung, and fails.

use std::sync::mpsc;
use std::time::Duration;

use acvus_mir::decl::{Core, Declarations, EntryDecl, ImplSig, Impls, LoopMarkers, RawDeclarations, Scheme, DeclaredFacts, ContextDecl};
use acvus_mir::graph::{Graph, Refusal, Source, Stop, Unit};
use acvus_mir::ir::cfg::print::{self, View};
use acvus_mir::ty::{Calls, Effect, EffectTerm, FlowTerm, Flows, Holds, Poly, TyId, TyKind};
use acvus_utils::{Astr, Interner, QualifiedRef};

pub use acvus_mir::graph::Stop as At;
pub use acvus_mir::passes::pipeline::Level;

/// How long one case may run.
const BUDGET: Duration = Duration::from_secs(1);

/// The units, declarations and stop of one test.
pub struct Case {
    names: Interner,
    raw: RawDeclarations,
    units: Vec<(QualifiedRef, Source)>,
    stop: Stop,
}

impl Default for Case {
    fn default() -> Self {
        Self::new()
    }
}

impl Case {
    /// No unit and no declaration, compiled through the required steps.
    pub fn new() -> Case {
        let names = Interner::new();
        let core = Core::of(&names);
        Case {
            raw: RawDeclarations {
                lang_impls: Vec::new(),
                types: acvus_mir::ty::Types::new(),
                functions: Vec::new(),
                user_types: Vec::new(),
                casts: Vec::new(),
                conversions: Vec::new(),
                contexts: Vec::new(),
                entries: Vec::new(),
                loop_markers: LoopMarkers::default(),
                core,
            },
            names,
            units: Vec::new(),
            stop: Stop::Required,
        }
    }

    pub fn names(&self) -> &Interner {
        &self.names
    }

    /// `core`'s names, whose impls `lang_impls` gives.
    pub fn core(&self) -> &Core {
        &self.raw.core
    }

    /// A declared type.
    pub fn ty(&mut self, kind: TyKind<Poly>) -> TyId<Poly> {
        self.raw.types.intern(kind)
    }

    /// A function type of `params` to `ret` with `effect`, flowing nothing,
    /// called any number of times, capturing nothing.
    pub fn function(&mut self, params: &[TyId<Poly>], ret: TyId<Poly>, effect: Effect) -> TyId<Poly> {
        self.ty(TyKind::Fn {
            params: params.into(),
            ret,
            effect: EffectTerm::Known(effect),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        })
    }

    /// An extern fn at the root, its parameters named `p0`, `p1`, ...
    pub fn declare(&mut self, name: &str, ty: TyId<Poly>) -> QualifiedRef {
        let id = QualifiedRef::root(self.names.intern(name));
        let arity = match self.raw.types.kind(ty) {
            TyKind::Fn { params, .. } => params.len(),
            other => panic!("a declaration's type is a function type, not {other:?}"),
        };
        let param_names = (0..arity).map(|at| self.names.intern(&format!("p{at}"))).collect();
        let scheme = Scheme { ty, param_names, bounds: Box::new([]), impls: None, requires: Box::new([]), facts: DeclaredFacts::default() };
        self.raw.functions.push((id, scheme));
        id
    }

    /// Concrete impls of the lang item `item` at each of `impls`.
    pub fn lang_impls(&mut self, item: QualifiedRef, impls: &[TyId<Poly>]) {
        let concrete = impls
            .iter()
            .map(|ty| {
                let arity = match self.raw.types.kind(*ty) {
                    TyKind::Fn { params, .. } => params.len(),
                    other => panic!("an impl's type is a function type, not {other:?}"),
                };
                let param_names = (0..arity).map(|at| self.names.intern(&format!("p{at}"))).collect();
                ImplSig { ty: *ty, param_names, requires: Box::new([]), facts: DeclaredFacts::default() }
            })
            .collect();
        self.raw.lang_impls.push((item, Impls { concrete, generic: None }));
    }

    /// A context at the root, of type `ty`.
    pub fn context(&mut self, name: &str, ty: TyId<Poly>) {
        let id = QualifiedRef::root(self.names.intern(name));
        self.raw.contexts.push((id, ContextDecl { ty, volatile: false }));
    }

    fn name(&self, path: &[&str], name: &str) -> QualifiedRef {
        let path: Vec<Astr> = path.iter().map(|segment| self.names.intern(segment)).collect();
        QualifiedRef::qualified(self.names.namespace(&path), self.names.intern(name))
    }

    /// A script unit at `path::name`.
    ///
    /// # Panics
    /// `source` does not parse: a test's source is written to parse.
    pub fn script(&mut self, path: &[&str], name: &str, source: &str) -> QualifiedRef {
        let unit = self.name(path, name);
        let script = acvus_ast::parse_script(&self.names, source).unwrap_or_else(|error| panic!("{source:?} parses: {error:?}"));
        self.units.push((unit, Source::Script(script)));
        unit
    }

    /// A template unit at `path::name`.
    ///
    /// # Panics
    /// As `script`.
    pub fn template(&mut self, path: &[&str], name: &str, source: &str) -> QualifiedRef {
        let unit = self.name(path, name);
        let template = acvus_ast::parse_template(&self.names, source).unwrap_or_else(|error| panic!("{source:?} parses: {error:?}"));
        self.units.push((unit, Source::Template(template)));
        unit
    }

    /// An entry at `function`, its inputs by name at their types, its result
    /// declared where given (host).
    pub fn entry(&mut self, function: QualifiedRef, inputs: &[(&str, TyId<Poly>)], result: Option<TyId<Poly>>) {
        let inputs = inputs.iter().map(|(name, ty)| (self.names.intern(name), *ty)).collect();
        self.raw.entries.push(EntryDecl { function, inputs, result });
    }

    /// Compiled up to `stop` instead.
    pub fn at(mut self, stop: Stop) -> Case {
        self.stop = stop;
        self
    }

    /// The compiled modules, printed (the module doc).
    ///
    /// # Panics
    /// The case is refused, or runs past its budget.
    pub fn admitted(self) -> String {
        match self.run() {
            Ok(printed) => printed,
            Err(refused) => panic!("admitted, but refused:\n{refused}"),
        }
    }

    /// The refusals, printed by unit.
    ///
    /// # Panics
    /// The case is admitted, or runs past its budget.
    pub fn refused(self) -> String {
        match self.run() {
            Ok(printed) => panic!("refused, but admitted:\n{printed}"),
            Err(refused) => refused,
        }
    }

    /// The case compiled on a thread of its own, within the budget.
    fn run(self) -> Result<String, String> {
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = sender.send(self.compile());
        });
        match receiver.recv_timeout(BUDGET) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => panic!("the case ran past its budget of {BUDGET:?}"),
            Err(mpsc::RecvTimeoutError::Disconnected) => panic!("the case panicked while compiling"),
        }
    }

    fn compile(self) -> Result<String, String> {
        let declarations = Declarations::verify(self.raw).map_err(|refused| format!("the declarations are refused: {refused:?}"))?;
        let mut graph = Graph::new(self.names.clone(), declarations);
        for (name, source) in self.units {
            graph.set_unit(Unit { name, source });
        }
        let view = View { canonical: true, ..View::CFG };
        match graph.compile(self.stop) {
            Ok(compiled) => {
                let mut modules: Vec<(String, String)> = compiled
                    .modules
                    .iter()
                    .map(|(unit, compiled)| (written(&self.names, *unit), print::module(&compiled.module, &self.names, view)))
                    .collect();
                modules.sort();
                Ok(modules.into_iter().map(|(unit, printed)| format!("module {unit}\n{printed}")).collect::<Vec<_>>().join("\n"))
            }
            Err(refused) => {
                let mut lines: Vec<String> = refused
                    .iter()
                    .flat_map(|refused| refused.refusals.iter().map(|refusal| format!("{}: {}", written(&self.names, refused.unit), shown(&self.names, refusal))))
                    .collect();
                lines.sort();
                Err(lines.join("\n"))
            }
        }
    }
}

/// A refusal in the graph's terms: a unit by the name the program writes;
/// a checker's refusal as its kind and span.
fn shown(names: &Interner, refusal: &Refusal) -> String {
    match refusal {
        Refusal::Cycle { through } => format!("Cycle [{}]", through.iter().map(|unit| written(names, *unit)).collect::<Vec<_>>().join(", ")),
        Refusal::EntryDoesNotFit { entry } => format!("EntryDoesNotFit {}", written(names, *entry)),
        Refusal::Typeck(refusal) => format!("{:?} @{}..{}", refusal.kind, refusal.span.start, refusal.span.end),
        Refusal::Check(refusal) => format!("{:?} @{}..{}", refusal.kind, refusal.span.start, refusal.span.end),
    }
}

/// A qualified name as the program writes it.
fn written(names: &Interner, name: QualifiedRef) -> String {
    let mut segments: Vec<String> = names.path(name.namespace).into_iter().map(|segment| names.resolve(segment).to_owned()).collect();
    segments.push(names.resolve(name.name).to_owned());
    segments.join("::")
}
