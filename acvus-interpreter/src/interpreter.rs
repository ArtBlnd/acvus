//! What a run is made of: the functions a module can reach, the shared
//! state they read, and the entry point that drives one.

use std::sync::Arc;

use acvus_mir::graph::QualifiedRef;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;
#[cfg(feature = "tooling")]
use smallvec::SmallVec;

use crate::code::Prepared;
use crate::flight::{Flight, Flying, Tally};
#[cfg(feature = "tooling")]
use std::collections::HashMap;

use acvus_extern::{Holding, Owned};
use futures::FutureExt;

use crate::executor::AsyncJob;
use crate::host::HostError;
use crate::init::SolvedRustInit;
use crate::machine::{call_module, call_module_rooted};
#[cfg(feature = "tooling")]
use crate::port::ContextWrite;
use crate::port::{Held, Port, ended};
use crate::runtime::{AcvusRuntime, ExternHandler};
use crate::value::Value;

/// Call arguments. Stack-allocated for <=4 args.
#[cfg(feature = "tooling")]
pub type Args = SmallVec<[Value; 4]>;

/// A single executable unit - a prepared MIR module or an extern function's
/// instances.
pub enum Executable {
    Module(Arc<Prepared>),
    Extern(Vec<ExternHandler>),
}

impl Executable {
    fn variant_name(&self) -> &'static str {
        match self {
            Self::Module(_) => "Module",
            Self::Extern(_) => "Extern",
        }
    }
}

/// One compilation, told apart from every other in the process, so a holder
/// one compilation made is never read by another, whose interner names its
/// types and tags differently.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Compilation(u64);

impl Compilation {
    fn next() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Compilation(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

pub(crate) struct Init {
    body: InitBody,
    ty: Arc<Ty>,
}

enum InitBody {
    Script(QualifiedRef),
    Rust(SolvedRustInit),
}

impl Init {
    pub(crate) fn script(function: QualifiedRef, ty: Arc<Ty>) -> Self {
        Init {
            body: InitBody::Script(function),
            ty,
        }
    }

    pub(crate) fn rust(make: SolvedRustInit, ty: Arc<Ty>) -> Self {
        Init {
            body: InitBody::Rust(make),
            ty,
        }
    }

    /// Run from an operation that cannot wait; `Host::compile` admits a
    /// script init under synchronous access only where its body cannot
    /// suspend.
    pub(crate) fn run_now(&self, rt: &AcvusRuntime) -> Held {
        let value = match &self.body {
            InitBody::Script(function) => owned_result(call_module_rooted(rt, *function)),
            InitBody::Rust(make) => make.make(rt),
        };
        Held::new(value, Arc::clone(&self.ty), rt.shared.compilation)
    }

    pub(crate) async fn run_waited(&self, rt: &AcvusRuntime) -> Held {
        let value = match &self.body {
            InitBody::Script(function) => owned_result(call_module(rt.clone(), *function, Vec::new()).await),
            InitBody::Rust(make) => make.make(rt),
        };
        Held::new(value, Arc::clone(&self.ty), rt.shared.compilation)
    }
}

fn owned_result(value: Value) -> Owned<AcvusRuntime> {
    // SAFETY: the init's run moved its result out to this caller, and no
    // other holder owns it.
    unsafe { Owned::from_value(Holding::new(), value) }
}

/// Readonly shared state - clone is cheap (Freeze/Arc internally).
#[derive(Clone)]
pub struct InterpreterContext {
    pub(crate) compilation: Compilation,
    pub(crate) inits: Freeze<FxHashMap<Box<str>, Init>>,
    pub interner: Interner,
    pub functions: Freeze<FxHashMap<QualifiedRef, Executable>>,
    pub fn_types: Freeze<FxHashMap<QualifiedRef, Ty>>,
    pub context_names: Freeze<FxHashMap<QualifiedRef, Astr>>,
    pub executor: Arc<dyn crate::executor::Executor>,
    /// Space hooks by extension type (RFC-0033).
    pub space: Arc<crate::layout::Hooks>,
}

impl InterpreterContext {
    pub fn new(
        interner: &Interner,
        functions: FxHashMap<QualifiedRef, Executable>,
        executor: Arc<dyn crate::executor::Executor>,
    ) -> Self {
        Self {
            compilation: Compilation::next(),
            inits: Freeze::new(FxHashMap::default()),
            interner: interner.clone(),
            functions: Freeze::new(functions),
            fn_types: Freeze::new(FxHashMap::default()),
            context_names: Freeze::new(FxHashMap::default()),
            executor,
            space: Arc::new(crate::layout::Hooks::default()),
        }
    }

    pub fn with_fn_types(mut self, fn_types: FxHashMap<QualifiedRef, Ty>) -> Self {
        self.fn_types = Freeze::new(fn_types);
        self
    }

    pub fn with_context_names(mut self, context_names: FxHashMap<QualifiedRef, Astr>) -> Self {
        self.context_names = Freeze::new(context_names);
        self
    }

    pub fn with_space(mut self, hooks: crate::layout::Hooks) -> Self {
        self.space = Arc::new(hooks);
        self
    }

    pub(crate) fn with_inits(mut self, inits: FxHashMap<Box<str>, Init>) -> Self {
        self.inits = Freeze::new(inits);
        self
    }

    /// A runtime that reaches no storage: a run on it that touches a context
    /// ends with `HostError::Storage`.
    pub fn runtime_over_an_empty_page(&self) -> AcvusRuntime {
        self.runtime_over(Port::nothing())
    }

    pub(crate) fn runtime_over(&self, port: Arc<Port>) -> AcvusRuntime {
        AcvusRuntime::new(Arc::new(self.clone()), port, Flight::new(), Tally::outermost())
    }
}

fn lookup_function<'a>(shared: &'a InterpreterContext, id: &QualifiedRef) -> &'a Executable {
    shared.functions.get(id).unwrap_or_else(|| {
        let name = shared.interner.resolve(id.name);
        panic!("no function for {id:?} (name={name:?})")
    })
}

pub(crate) fn lookup_module<'a>(
    shared: &'a InterpreterContext,
    id: &QualifiedRef,
) -> &'a Arc<Prepared> {
    match lookup_function(shared, id) {
        Executable::Module(m) => m,
        other => panic!("expected Module for {id:?}, got {}", other.variant_name()),
    }
}


pub struct Interpreter {
    shared: Arc<InterpreterContext>,
    entry: QualifiedRef,
    port: Arc<Port>,
    flight: Arc<Flight>,
    tally: Arc<Tally>,
    args: Vec<Value>,
    _flying: Option<Flying>,
}

impl Interpreter {
    /// A run over contexts the tooling seeds, each at the type it states.
    #[cfg(feature = "tooling")]
    pub fn new(
        shared: InterpreterContext,
        entry: QualifiedRef,
        seeds: HashMap<String, (Ty, acvus_extern::Owned<AcvusRuntime>)>,
    ) -> Self {
        let compilation = shared.compilation;
        let holders = seeds
            .into_iter()
            .map(|(key, (ty, value))| (key, Held::new(value, Arc::new(ty), compilation)))
            .collect();
        Self::on_port(shared, entry, Port::seeded(holders), Vec::new())
    }

    pub(crate) fn on_port(
        shared: InterpreterContext,
        entry: QualifiedRef,
        port: Arc<Port>,
        args: Vec<Value>,
    ) -> Self {
        Self {
            shared: Arc::new(shared),
            entry,
            port,
            flight: Flight::new(),
            tally: Tally::outermost(),
            args,
            _flying: None,
        }
    }

    /// The deferred run a `Spawn` of a module issues: the spawning run's
    /// runtime, whose frame the new run's frames run within, and the
    /// arguments it passed.
    pub(crate) fn spawned(rt: &AcvusRuntime, entry: QualifiedRef, args: Vec<Value>) -> AsyncJob {
        let mut run = Self {
            shared: Arc::clone(&rt.shared),
            entry,
            port: Arc::clone(&rt.port),
            flight: Arc::clone(&rt.flight),
            tally: Arc::clone(&rt.tally),
            args,
            _flying: Some(rt.flight.start()),
        };
        AsyncJob::new(Box::pin(async move { run.run().await }))
    }

    fn runtime(&self) -> AcvusRuntime {
        AcvusRuntime::new(
            Arc::clone(&self.shared),
            Arc::clone(&self.port),
            Arc::clone(&self.flight),
            Arc::clone(&self.tally),
        )
    }

    /// Execute the entry module and return its value, or the error the run
    /// ended with where a storage access refused it (RFC-0090 rule 4).
    ///
    /// # Panics
    /// The entry's result is a view: a host reads one `Value` by kind
    /// (RFC-0054), which a register pair has none of. `CompilationGraph::
    /// entries` carries which bodies are entries and `typeck::ResultCrossing::
    /// OneValue` refuses it there, so reaching this assert means a program
    /// arrived without passing the checker.
    #[cfg(feature = "tooling")]
    pub async fn execute(&mut self) -> Result<Value, HostError> {
        self.ended_or_ran().await
    }

    /// The run to its result, or to the `HostError` it ended with; a run
    /// that fails otherwise goes on unwinding.
    pub(crate) async fn ended_or_ran(&mut self) -> Result<Value, HostError> {
        std::panic::AssertUnwindSafe(self.run())
            .catch_unwind()
            .await
            .map_err(ended)
    }

    async fn run(&mut self) -> Value {
        let entry = lookup_module(&self.shared, &self.entry).main.as_ref();
        assert!(
            !entry.returns_a_view,
            "the entry's result is a view, and a host reads one value by kind (RFC-0054); \
             typeck refuses this at `CompilationGraph::entries`, so the checker was bypassed"
        );
        let args = std::mem::take(&mut self.args);
        call_module(self.runtime(), self.entry, args).await
    }

    /// The final value of every context the run stored, drained from the
    /// seeded holders.
    #[cfg(feature = "tooling")]
    pub fn take_writes(&self) -> Vec<ContextWrite> {
        self.port.take_writes()
    }
}
