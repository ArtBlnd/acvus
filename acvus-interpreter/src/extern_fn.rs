//! ExternFn — unified declaration of external functions.
//!
//! Bundles type signature + runtime handler in one place.
//! On registration, allocates a FunctionId and produces both the compile-time
//! `Function` (for the graph) and the runtime `Executable` (for the interpreter).
//!
//! # Handler model
//!
//! Handlers receive **args** (function parameters) and **uses** (context reads),
//! and return **ret** (return value) and **defs** (context writes).
//!
//! ```ignore
//! ExternFnBuilder::new("llm_call", constraint)
//!     .handler(|interner, (prompt,): (String,)((history,)): Uses<(Vec<Value>,)>| {
//!         let new_history = /* ... */;
//!         Ok(("result".into(), Defs((new_history,))))
//!     });
//! ```
//!
//! - `Uses<T>` wraps context reads (immutable, captured at spawn).
//! - `Defs<T>` wraps context writes (must be returned — compiler enforces this).
//! - For pure functions with no context: `Uses(())` and `()`.

use std::pin::Pin;
use std::sync::Arc;

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::PolyTy;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::error::RuntimeError;
use crate::interpreter::{AsyncBuiltinFn, BuiltinHandler, Executable, SyncBuiltinFn};
use crate::value::{FromValues, IntoValue, Value};

// ── ExternHandler ───────────────────────────────────────────────────

/// Type-erased extern handler. Closure-based — can capture environment.
///
/// Two variants:
/// - `Sync`: blocking, may run on a blocking thread pool
/// - `Async`: non-blocking, runs on async runtime
///
/// Both receive the call arguments and return the result value.
/// Internally Arc-wrapped so it can be cheaply cloned into spawn closures.
#[derive(Clone)]
pub enum ExternHandler {
    Sync(Arc<dyn Fn(Vec<Value>, &Interner) -> Result<Value, RuntimeError> + Send + Sync>),
    /// Interner is owned (Arc clone) — no lifetime across await points.
    Async(
        Arc<
            dyn Fn(Vec<Value>, Interner) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + Send>>
                + Send
                + Sync,
        >,
    ),
}

impl ExternHandler {
    /// Whether this handler is sync (blocking).
    pub fn is_sync(&self) -> bool {
        matches!(self, Self::Sync(_))
    }
}

/// Convert a typed sync closure into a type-erased `ExternHandler::Sync`.
pub fn into_sync_extern_handler<A, R, F>(f: F) -> ExternHandler
where
    F: Fn(&Interner, A) -> Result<R, RuntimeError> + Send + Sync + 'static,
    A: FromValues + 'static,
    R: IntoValue + 'static,
{
    ExternHandler::Sync(Arc::new(move |args, interner| {
        let a = A::from_values(args)?;
        Ok(f(interner, a)?.into_value())
    }))
}

/// Convert a typed async closure into a type-erased `ExternHandler::Async`.
pub fn into_async_extern_handler<A, R, F, Fut>(f: F) -> ExternHandler
where
    F: Fn(Interner, A) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<R, RuntimeError>> + Send + 'static,
    A: FromValues + 'static,
    R: IntoValue + 'static,
{
    ExternHandler::Async(Arc::new(move |args, interner| {
        let a = match A::from_values(args) {
            Ok(v) => v,
            Err(e) => return Box::pin(std::future::ready(Err(e))),
        };
        let fut = f(interner, a);
        Box::pin(async move { Ok(fut.await?.into_value()) })
    }))
}

// ── Handler kind ────────────────────────────────────────────────────

/// Distinguishes legacy builtin handlers from new extern handlers.
enum HandlerKind {
    /// Legacy path: fn pointer, used by builtins and existing ExternFn registrations.
    Legacy(BuiltinHandler),
    /// New path: closure-based, uses/defs aware, SSA-sound.
    Extern(ExternHandler),
}

// ── ExternFn ────────────────────────────────────────────────────────

/// A fully-specified external function: type + handler.
pub struct ExternFn {
    pub name: String,
    pub ty: PolyTy,
    handler_kind: HandlerKind,
}

/// Builder for constructing an ExternFn.
pub struct ExternFnBuilder {
    name: String,
    ty: PolyTy,
}

impl ExternFnBuilder {
    pub fn new(name: impl Into<String>, ty: PolyTy) -> Self {
        Self {
            name: name.into(),
            ty,
        }
    }

    pub fn handler<A, R, F>(self, f: F) -> ExternFn
    where
        F: Fn(&Interner, A) -> Result<R, RuntimeError> + Send + Sync + 'static,
        A: FromValues + 'static,
        R: IntoValue + 'static,
    {
        ExternFn {
            name: self.name,
            ty: self.ty.clone(),
            handler_kind: HandlerKind::Extern(into_sync_extern_handler(f)),
        }
    }

    pub fn handler_async<A, R, F, Fut>(self, f: F) -> ExternFn
    where
        F: Fn(Interner, A) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<R, RuntimeError>> + Send + 'static,
        A: FromValues + 'static,
        R: IntoValue + 'static,
    {
        ExternFn {
            name: self.name,
            ty: self.ty.clone(),
            handler_kind: HandlerKind::Extern(into_async_extern_handler(f)),
        }
    }

    /// Legacy: register a sync handler (fn pointer, args only).
    pub fn sync_handler(self, f: SyncBuiltinFn) -> ExternFn {
        ExternFn {
            name: self.name,
            ty: self.ty.clone(),
            handler_kind: HandlerKind::Legacy(BuiltinHandler::Sync(f)),
        }
    }

    /// Legacy: register an async handler (fn pointer, receives &mut Interpreter).
    pub fn async_handler(self, f: AsyncBuiltinFn) -> ExternFn {
        ExternFn {
            name: self.name,
            ty: self.ty.clone(),
            handler_kind: HandlerKind::Legacy(BuiltinHandler::Async(f)),
        }
    }
}

// ── Registration ────────────────────────────────────────────────────

/// Result of registering an ExternRegistry — everything needed for both
/// compilation and execution.
pub struct Registered {
    /// Functions to add to CompilationGraph.
    pub functions: Vec<Function>,
    /// Runtime handlers keyed by QualifiedRef.
    pub executables: FxHashMap<QualifiedRef, Executable>,
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

    /// Construct QualifiedRefs and produce both graph Functions and runtime Executables.
    pub fn register(self, interner: &Interner) -> Registered {
        let fns = (self.factory)(interner);
        let mut functions = Vec::with_capacity(fns.len());
        let mut executables = FxHashMap::default();

        for f in fns {
            let name = interner.intern(&f.name);
            let qref = QualifiedRef::root(name);

            functions.push(Function {
                qref,
                kind: FnKind::Extern,
                ty: f.ty,
            });

            match f.handler_kind {
                HandlerKind::Legacy(h) => executables.insert(qref, Executable::Builtin(h)),
                HandlerKind::Extern(h) => executables.insert(qref, Executable::Extern(h)),
            };
        }

        Registered {
            functions,
            executables,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_mir::ty::{PolyParam, Ty, TyTerm, lift_to_poly};

    fn interner() -> Interner {
        Interner::new()
    }

    fn sig(interner: &Interner, params: Vec<Ty>, ret: Ty) -> PolyTy {
        let named: Vec<PolyParam> = params
            .iter()
            .enumerate()
            .map(|(i, ty)| PolyParam::new(interner.intern(&format!("_{i}")), lift_to_poly(ty)))
            .collect();
        TyTerm::Fn {
            params: named,
            ret: Box::new(lift_to_poly(&ret)),
            captures: vec![],
            hint: None,
        }
    }

    // ── Sync handler ──────────────────────────────────────────────

    #[test]
    fn sync_handler_pure_add() {
        let handler = into_sync_extern_handler(
            |_interner: &Interner, (a, b): (i64, i64)| Ok(a + b),
        );
        let interner = interner();
        let output = match &handler {
            ExternHandler::Sync(f) => {
                f(vec![Value::Int(10), Value::Int(32)], &interner).unwrap()
            }
            _ => panic!("expected sync"),
        };
        assert_eq!(output, Value::Int(42));
    }

    // ── Handler with uses and defs (context read + write) ─────────

    // ── Multiple defs ─────────────────────────────────────────────

    // ── Type mismatch error ───────────────────────────────────────

    #[test]
    fn from_value_type_mismatch() {
        let handler =
            into_sync_extern_handler(|_interner: &Interner, (x,): (i64,)| {
                Ok(x)
            });
        let interner = interner();
        // Pass String where i64 expected.
        let result = match &handler {
            ExternHandler::Sync(f) => f(vec![Value::string("not a number")], &interner),
            _ => panic!("expected sync"),
        };
        assert!(result.is_err());
    }

    // ── Environment capture ───────────────────────────────────────

    #[test]
    fn handler_captures_environment() {
        let multiplier = 7i64;
        let handler = into_sync_extern_handler(
            move |_interner: &Interner, (x,): (i64,)| {
                Ok(x * multiplier)
            },
        );
        let interner = interner();
        let output = match &handler {
            ExternHandler::Sync(f) => f(vec![Value::Int(6)], &interner).unwrap(),
            _ => panic!("expected sync"),
        };
        assert_eq!(output, Value::Int(42));
    }

    // ── Builder integration ───────────────────────────────────────

    #[test]
    fn builder_creates_extern_fn() {
        let i = interner();
        let ext = ExternFnBuilder::new("add", sig(&i, vec![Ty::Int, Ty::Int], Ty::Int)).handler(
            |_interner: &Interner, (a, b): (i64, i64)| Ok(a + b),
        );

        assert_eq!(ext.name, "add");
        assert!(matches!(ext.handler_kind, HandlerKind::Extern(_)));
    }

    // ── Registration produces Executable::Extern ──────────────────

    #[test]
    fn registry_produces_extern_executable() {
        let registry = ExternRegistry::new(|interner| {
            vec![
                ExternFnBuilder::new("add", sig(interner, vec![Ty::Int, Ty::Int], Ty::Int))
                    .handler(
                        |_interner: &Interner, (a, b): (i64, i64)| {
                            Ok(a + b)
                        },
                    ),
            ]
        });

        let interner = interner();
        let registered = registry.register(&interner);

        assert_eq!(registered.functions.len(), 1);
        assert_eq!(registered.executables.len(), 1);

        let (_, exec) = registered.executables.iter().next().unwrap();
        assert!(matches!(exec, Executable::Extern(_)));
    }
}
