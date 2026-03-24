//! ExternFn — unified declaration of external functions.
//!
//! Bundles type signature + runtime handler + effect in one place.
//! On registration, allocates a FunctionId and produces both the compile-time
//! `Function` (for the graph) and the runtime `Executable` (for the interpreter).
//!
//! ```ignore
//! let registry = ExternRegistry::new(vec![
//!     ExternFn::build("regex")
//!         .params(vec![Ty::String])
//!         .ret(opaque_ty())
//!         .pure()
//!         .sync_handler(|args| {
//!             let pattern = args[0].as_str();
//!             Ok(Value::opaque(compile_regex(pattern)))
//!         }),
//! ]);
//! let registered = registry.register(&interner);
//! // registered.functions  → Vec<Function>  (for CompilationGraph)
//! // registered.executables → FxHashMap<FunctionId, Executable>  (for Interpreter)
//! ```

use acvus_mir::graph::{Constraint, FnConstraint, FnKind, Function, FunctionId};
use acvus_mir::ty::{Effect, Param, Ty};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

use crate::interpreter::{AsyncBuiltinFn, BuiltinHandler, Executable, SyncBuiltinFn};

/// A fully-specified external function: signature + handler.
pub struct ExternFn {
    pub name: String,
    pub params: Vec<Ty>,
    pub ret: Ty,
    pub effect: Effect,
    pub handler: BuiltinHandler,
}

impl ExternFn {
    /// Start building an ExternFn.
    pub fn build(name: impl Into<String>) -> ExternFnBuilder {
        ExternFnBuilder::new(name)
    }
}

/// Builder for constructing an ExternFn.
pub struct ExternFnBuilder {
    name: String,
    params: Vec<Ty>,
    ret: Ty,
    effect: Effect,
}

impl ExternFnBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            ret: Ty::Unit,
            effect: Effect::pure(),
        }
    }

    pub fn params(mut self, params: Vec<Ty>) -> Self {
        self.params = params;
        self
    }

    pub fn ret(mut self, ty: Ty) -> Self {
        self.ret = ty;
        self
    }

    pub fn pure(mut self) -> Self {
        self.effect = Effect::pure();
        self
    }

    pub fn io(mut self) -> Self {
        self.effect = Effect::io();
        self
    }

    pub fn sync_handler(self, f: SyncBuiltinFn) -> ExternFn {
        ExternFn {
            name: self.name,
            params: self.params,
            ret: self.ret,
            effect: self.effect,
            handler: BuiltinHandler::Sync(f),
        }
    }

    pub fn async_handler(self, f: AsyncBuiltinFn) -> ExternFn {
        ExternFn {
            name: self.name,
            params: self.params,
            ret: self.ret,
            effect: self.effect,
            handler: BuiltinHandler::Async(f),
        }
    }
}

/// Result of registering an ExternRegistry — everything needed for both
/// compilation and execution.
pub struct Registered {
    /// Functions to add to CompilationGraph.
    pub functions: Vec<Function>,
    /// Runtime handlers keyed by the allocated FunctionId.
    pub executables: FxHashMap<FunctionId, Executable>,
}

/// A collection of ExternFns, created lazily with interner access.
pub struct ExternRegistry {
    factory: Box<dyn FnOnce(&Interner) -> Vec<ExternFn>>,
}

impl ExternRegistry {
    /// Create a registry from a factory that receives the interner.
    /// This allows ExternFn params/ret to use Astr-based types (Object, etc).
    pub fn new(factory: impl FnOnce(&Interner) -> Vec<ExternFn> + 'static) -> Self {
        Self {
            factory: Box::new(factory),
        }
    }

    /// Allocate FunctionIds and produce both graph Functions and runtime Executables.
    pub fn register(self, interner: &Interner) -> Registered {
        let fns = (self.factory)(interner);
        let mut functions = Vec::with_capacity(fns.len());
        let mut executables = FxHashMap::default();

        for f in fns {
            let id = FunctionId::alloc();
            let name = interner.intern(&f.name);

            // Build Ty::Fn for the graph.
            let params: Vec<Param> = f
                .params
                .iter()
                .enumerate()
                .map(|(i, ty)| Param::new(interner.intern(&format!("_{i}")), ty.clone()))
                .collect();
            let fn_ty = Ty::Fn {
                params,
                ret: Box::new(f.ret),
                captures: vec![],
                effect: f.effect,
            };

            functions.push(Function {
                id,
                name,
                kind: FnKind::Extern {
                    deps: Freeze::new(vec![]),
                },
                constraint: FnConstraint {
                    signature: None,
                    output: Constraint::Exact(fn_ty),
                },
            });

            executables.insert(id, Executable::Builtin(f.handler));
        }

        Registered {
            functions,
            executables,
        }
    }
}
