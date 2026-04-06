//! ExternFn — unified declaration of external functions.
//!
//! Bundles type signature + runtime handler + effect in one place.
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
//!     .handler(|interner, (prompt,): (String,), Uses((history,)): Uses<(Vec<Value>,)>| {
//!         let new_history = /* ... */;
//!         Ok(("result".into(), Defs((new_history,))))
//!     });
//! ```
//!
//! - `Uses<T>` wraps context reads (immutable, captured at spawn).
//! - `Defs<T>` wraps context writes (must be returned — compiler enforces this).
//! - For pure functions with no context: `Uses(())` and `Defs(())`.

use std::pin::Pin;
use std::sync::Arc;

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{EffectConstraint, PolyTy};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::error::RuntimeError;
use crate::interpreter::{AsyncBuiltinFn, BuiltinHandler, Executable, SyncBuiltinFn};
use crate::value::{FromValues, IntoValue, IntoValues, Value};

// ── Newtypes for context boundary ───────────────────────────────────

/// Context reads — immutable values captured at spawn time.
/// Wraps a tuple of concrete types extracted via `FromValue`.
pub struct Uses<T>(pub T);

/// Context writes — new values that must be returned from the handler.
/// Wraps a tuple of concrete types converted via `IntoValue`.
/// Returning `Defs` is mandatory — the compiler enforces this.
pub struct Defs<T>(pub T);

// ── ExternHandler ───────────────────────────────────────────────────

/// Output of an extern handler call.
pub struct ExternOutput {
    pub rets: Vec<Value>,
    pub defs: Vec<Value>,
}

/// Type-erased extern handler. Closure-based — can capture environment.
///
/// Two variants:
/// - `Sync`: blocking, may run on a blocking thread pool
/// - `Async`: non-blocking, runs on async runtime
///
/// Both receive `(args, uses)` and return `(rets, defs)`.
/// Internally Arc-wrapped so it can be cheaply cloned into spawn closures.
#[derive(Clone)]
pub enum ExternHandler {
    /// Sync handler: `(args, uses, &Interner) -> Result<ExternOutput>`
    Sync(
        Arc<
            dyn Fn(Vec<Value>, Vec<Value>, &Interner) -> Result<ExternOutput, RuntimeError>
                + Send
                + Sync,
        >,
    ),
    /// Async handler: `(args, uses, Interner) -> Future<Result<ExternOutput>>`
    /// Interner is owned (Arc clone) — no lifetime across await points.
    Async(
        Arc<
            dyn Fn(
                    Vec<Value>,
                    Vec<Value>,
                    Interner,
                )
                    -> Pin<Box<dyn Future<Output = Result<ExternOutput, RuntimeError>> + Send>>
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
pub fn into_sync_extern_handler<A, U, R, D, F>(f: F) -> ExternHandler
where
    F: Fn(&Interner, A, Uses<U>) -> Result<(R, Defs<D>), RuntimeError> + Send + Sync + 'static,
    A: FromValues + 'static,
    U: FromValues + 'static,
    R: IntoValue + 'static,
    D: IntoValues + 'static,
{
    ExternHandler::Sync(Arc::new(move |args, uses, interner| {
        let a = A::from_values(args)?;
        let u = Uses(U::from_values(uses)?);
        let (ret, Defs(defs)) = f(interner, a, u)?;
        Ok(ExternOutput {
            rets: vec![ret.into_value()],
            defs: defs.into_values(),
        })
    }))
}

/// Convert a typed async closure into a type-erased `ExternHandler::Async`.
pub fn into_async_extern_handler<A, U, R, D, F, Fut>(f: F) -> ExternHandler
where
    F: Fn(Interner, A, Uses<U>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(R, Defs<D>), RuntimeError>> + Send + 'static,
    A: FromValues + 'static,
    U: FromValues + 'static,
    R: IntoValue + 'static,
    D: IntoValues + 'static,
{
    ExternHandler::Async(Arc::new(move |args, uses, interner| {
        let a = match A::from_values(args) {
            Ok(v) => v,
            Err(e) => return Box::pin(std::future::ready(Err(e))),
        };
        let u = match U::from_values(uses) {
            Ok(v) => Uses(v),
            Err(e) => return Box::pin(std::future::ready(Err(e))),
        };
        let fut = f(interner, a, u);
        Box::pin(async move {
            let (ret, Defs(defs)) = fut.await?;
            Ok(ExternOutput {
                rets: vec![ret.into_value()],
                defs: defs.into_values(),
            })
        })
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

/// A fully-specified external function: constraint + handler.
pub struct ExternFn {
    pub name: String,
    pub ty: PolyTy,
    pub effect_constraint: Option<EffectConstraint>,
    handler_kind: HandlerKind,
}

/// Builder for constructing an ExternFn.
pub struct ExternFnBuilder {
    name: String,
    ty: PolyTy,
    effect_constraint: Option<EffectConstraint>,
}

impl ExternFnBuilder {
    pub fn new(name: impl Into<String>, ty: PolyTy) -> Self {
        Self {
            name: name.into(),
            ty,
            effect_constraint: None,
        }
    }

    pub fn with_effect_constraint(mut self, constraint: EffectConstraint) -> Self {
        self.effect_constraint = Some(constraint);
        self
    }

    /// Register a sync type-safe handler with explicit `Uses` and `Defs`.
    pub fn handler<A, U, R, D, F>(self, f: F) -> ExternFn
    where
        F: Fn(&Interner, A, Uses<U>) -> Result<(R, Defs<D>), RuntimeError> + Send + Sync + 'static,
        A: FromValues + 'static,
        U: FromValues + 'static,
        R: IntoValue + 'static,
        D: IntoValues + 'static,
    {
        ExternFn {
            name: self.name,
            ty: self.ty.clone(),
            effect_constraint: self.effect_constraint.clone(),
            handler_kind: HandlerKind::Extern(into_sync_extern_handler(f)),
        }
    }

    /// Register an async type-safe handler with explicit `Uses` and `Defs`.
    pub fn handler_async<A, U, R, D, F, Fut>(self, f: F) -> ExternFn
    where
        F: Fn(Interner, A, Uses<U>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(R, Defs<D>), RuntimeError>> + Send + 'static,
        A: FromValues + 'static,
        U: FromValues + 'static,
        R: IntoValue + 'static,
        D: IntoValues + 'static,
    {
        ExternFn {
            name: self.name,
            ty: self.ty.clone(),
            effect_constraint: self.effect_constraint.clone(),
            handler_kind: HandlerKind::Extern(into_async_extern_handler(f)),
        }
    }

    /// Legacy: register a sync handler (fn pointer, args only).
    pub fn sync_handler(self, f: SyncBuiltinFn) -> ExternFn {
        ExternFn {
            name: self.name,
            ty: self.ty.clone(),
            effect_constraint: self.effect_constraint.clone(),
            handler_kind: HandlerKind::Legacy(BuiltinHandler::Sync(f)),
        }
    }

    /// Legacy: register an async handler (fn pointer, receives &mut Interpreter).
    pub fn async_handler(self, f: AsyncBuiltinFn) -> ExternFn {
        ExternFn {
            name: self.name,
            ty: self.ty.clone(),
            effect_constraint: self.effect_constraint.clone(),
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
                effect_constraint: f.effect_constraint,
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
    use acvus_mir::ty::{Effect, Param, Poly, PolyEffect, PolyParam, Ty, TyTerm, lift_to_poly, lift_effect};

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
            effect: lift_effect(&Effect::pure()),
            hint: None,
        }
    }

    // ── Pure handler, no context ──────────────────────────────────

    #[test]
    fn sync_handler_pure_add() {
        let handler = into_sync_extern_handler(
            |_interner: &Interner, (a, b): (i64, i64), Uses(()): Uses<()>| Ok((a + b, Defs(()))),
        );
        let interner = interner();
        let output = match &handler {
            ExternHandler::Sync(f) => {
                f(vec![Value::Int(10), Value::Int(32)], vec![], &interner).unwrap()
            }
            _ => panic!("expected sync"),
        };
        assert_eq!(output.rets.len(), 1);
        assert_eq!(output.rets[0], Value::Int(42));
        assert!(output.defs.is_empty());
    }

    // ── Handler with uses (context reads) ─────────────────────────

    #[test]
    fn sync_handler_with_uses() {
        // Handler reads a context value and adds it to the arg.
        let handler = into_sync_extern_handler(
            |_interner: &Interner, (x,): (i64,), Uses((offset,)): Uses<(i64,)>| {
                Ok((x + offset, Defs(())))
            },
        );
        let interner = interner();
        let output = match &handler {
            ExternHandler::Sync(f) => f(
                vec![Value::Int(10)],
                vec![Value::Int(100)], // uses: offset = 100
                &interner,
            )
            .unwrap(),
            _ => panic!("expected sync"),
        };
        assert_eq!(output.rets[0], Value::Int(110));
        assert!(output.defs.is_empty());
    }

    // ── Handler with uses and defs (context read + write) ─────────

    #[test]
    fn sync_handler_with_uses_and_defs() {
        // Handler reads history (uses), appends to it, returns new history (defs).
        let handler = into_sync_extern_handler(
            |_interner: &Interner, (msg,): (Value,), Uses((history,)): Uses<(Vec<Value>,)>| {
                let mut new_history = history;
                new_history.push(msg);
                let len = Value::Int(new_history.len() as i64);
                Ok((len, Defs((new_history,))))
            },
        );
        let interner = interner();

        // Initial history: [Int(1), Int(2)]
        let initial_history = Value::list(vec![Value::Int(1), Value::Int(2)]);
        let output = match &handler {
            ExternHandler::Sync(f) => f(
                vec![Value::Int(3)],   // args: msg = 3
                vec![initial_history], // uses: history = [1, 2]
                &interner,
            )
            .unwrap(),
            _ => panic!("expected sync"),
        };

        // ret = 3 (new length)
        assert_eq!(output.rets[0], Value::Int(3));
        // defs = [[1, 2, 3]] (updated history)
        assert_eq!(output.defs.len(), 1);
        match &output.defs[0] {
            Value::List(l) => {
                assert_eq!(l.len(), 3);
                assert_eq!(l[0], Value::Int(1));
                assert_eq!(l[1], Value::Int(2));
                assert_eq!(l[2], Value::Int(3));
            }
            other => panic!("expected List, got {other:?}"),
        }
    }

    // ── Multiple defs ─────────────────────────────────────────────

    #[test]
    fn sync_handler_multiple_defs() {
        // Handler writes two contexts.
        let handler =
            into_sync_extern_handler(|_interner: &Interner, (): (), Uses(()): Uses<()>| {
                Ok((Value::Unit, Defs((42i64, "hello".to_string()))))
            });
        let interner = interner();
        let output = match &handler {
            ExternHandler::Sync(f) => f(vec![], vec![], &interner).unwrap(),
            _ => panic!("expected sync"),
        };
        assert_eq!(output.defs.len(), 2);
        assert_eq!(output.defs[0], Value::Int(42));
        match &output.defs[1] {
            Value::String(s) => assert_eq!(&**s, "hello"),
            other => panic!("expected String, got {other:?}"),
        }
    }

    // ── Type mismatch error ───────────────────────────────────────

    #[test]
    fn from_value_type_mismatch() {
        let handler =
            into_sync_extern_handler(|_interner: &Interner, (x,): (i64,), Uses(()): Uses<()>| {
                Ok((x, Defs(())))
            });
        let interner = interner();
        // Pass String where i64 expected.
        let result = match &handler {
            ExternHandler::Sync(f) => f(vec![Value::string("not a number")], vec![], &interner),
            _ => panic!("expected sync"),
        };
        assert!(result.is_err());
    }

    // ── Environment capture ───────────────────────────────────────

    #[test]
    fn handler_captures_environment() {
        let multiplier = 7i64;
        let handler = into_sync_extern_handler(
            move |_interner: &Interner, (x,): (i64,), Uses(()): Uses<()>| {
                Ok((x * multiplier, Defs(())))
            },
        );
        let interner = interner();
        let output = match &handler {
            ExternHandler::Sync(f) => f(vec![Value::Int(6)], vec![], &interner).unwrap(),
            _ => panic!("expected sync"),
        };
        assert_eq!(output.rets[0], Value::Int(42));
    }

    // ── Builder integration ───────────────────────────────────────

    #[test]
    fn builder_creates_extern_fn() {
        let i = interner();
        let ext = ExternFnBuilder::new("add", sig(&i, vec![Ty::Int, Ty::Int], Ty::Int)).handler(
            |_interner: &Interner, (a, b): (i64, i64), Uses(()): Uses<()>| Ok((a + b, Defs(()))),
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
                        |_interner: &Interner, (a, b): (i64, i64), Uses(()): Uses<()>| {
                            Ok((a + b, Defs(())))
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
