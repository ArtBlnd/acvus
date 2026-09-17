//! What a run is made of: the functions a module can reach, the shared
//! state they read, and the entry point that drives one.

use std::sync::Arc;

use acvus_mir::graph::QualifiedRef;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::code::Prepared;
use crate::journal::{ContextWrite, InMemoryContext, RuntimeContext};
use crate::machine::call_module;
use crate::runtime::{AcvusRuntime, ExternHandler};
use crate::value::Value;
use crate::vtable::VtableRegistry;

/// Call arguments. Stack-allocated for <=4 args.
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

/// Readonly shared state - clone is cheap (Freeze/Arc internally).
#[derive(Clone)]
pub struct InterpreterContext {
    pub interner: Interner,
    pub functions: Freeze<FxHashMap<QualifiedRef, Executable>>,
    pub fn_types: Freeze<FxHashMap<QualifiedRef, Ty>>,
    pub context_names: Freeze<FxHashMap<QualifiedRef, Astr>>,
    pub executor: Arc<dyn crate::executor::Executor>,
    pub vtables: Arc<VtableRegistry>,
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
            interner: interner.clone(),
            functions: Freeze::new(functions),
            fn_types: Freeze::new(FxHashMap::default()),
            context_names: Freeze::new(FxHashMap::default()),
            executor,
            vtables: Arc::new(VtableRegistry::default()),
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

    pub fn runtime(&self) -> AcvusRuntime {
        AcvusRuntime(Arc::new(self.clone()))
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
    page: Arc<dyn RuntimeContext>,
    spawn_args: Vec<Value>,
}

impl Interpreter {
    pub fn new(shared: InterpreterContext, entry: QualifiedRef, page: InMemoryContext) -> Self {
        Self::on_page(shared, entry, Arc::new(page))
    }

    /// An interpreter over a page the caller keeps a handle to: a space's
    /// page, committed after the run (RFC-0033).
    pub fn on_page(
        shared: InterpreterContext,
        entry: QualifiedRef,
        page: Arc<dyn RuntimeContext>,
    ) -> Self {
        Self {
            shared: Arc::new(shared),
            entry,
            page,
            spawn_args: Vec::new(),
        }
    }

    /// The deferred run a `Spawn` of a module issues: the spawning run's
    /// page, and the arguments it passed.
    pub(crate) fn spawned(
        shared: Arc<InterpreterContext>,
        entry: QualifiedRef,
        page: Arc<dyn RuntimeContext>,
        args: Vec<Value>,
    ) -> Self {
        Self {
            shared,
            entry,
            page,
            spawn_args: args,
        }
    }

    /// Execute the entry module and return its value. The page keeps every
    /// context the run assigned; `take_writes` hands them out.
    pub async fn execute(&mut self) -> Value {
        let args = std::mem::take(&mut self.spawn_args);
        call_module(
            Arc::clone(&self.shared),
            Arc::clone(&self.page),
            self.entry,
            args,
        )
        .await
    }

    pub fn take_writes(&self) -> Vec<ContextWrite> {
        self.page.take_writes()
    }
}
