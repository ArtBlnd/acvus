//! A host compiles scripts with `Host`, and inside `Program::scope` opens a
//! page over a storage and runs entries on it (RFC-0090).
//!
//! A page and an entry belong to the one scope that made them, so a page
//! opened by one program never serves another's entry:
//!
//! ```compile_fail
//! use acvus_interpreter::{MemoryStorage, Program};
//! async fn cross(a: &Program, b: &Program, storage: &mut MemoryStorage) {
//!     a.scope(async |a| {
//!         let mut page = a.open(storage);
//!         b.scope(async |b| {
//!             let entry = b.entry::<(), i64>("main").unwrap();
//!             let _ = entry.run(&mut page, ()).await;
//!         })
//!         .await;
//!     })
//!     .await;
//! }
//! ```
//!
//! ```
//! use acvus_interpreter::{HostError, MemoryStorage, Program};
//! async fn same(a: &Program, storage: &mut MemoryStorage) -> Result<i64, HostError> {
//!     a.scope(async |a| {
//!         let mut page = a.open(storage);
//!         let entry = a.entry::<(), i64>("main")?;
//!         entry.run(&mut page, ()).await?.with(|n: &i64| *n)
//!     })
//!     .await
//! }
//! ```
//!
//! and neither leaves the scope:
//!
//! ```compile_fail
//! use acvus_interpreter::{MemoryStorage, Program};
//! async fn kept(a: &Program, storage: &mut MemoryStorage) {
//!     let _page = a.scope(async |a| a.open(storage)).await;
//! }
//! ```
//!
//! A program compiled for synchronous access opens only a synchronous
//! storage, and one compiled with `Host::async_access` opens a storage whose
//! access waits (RFC-0090 rule 3):
//!
//! ```compile_fail
//! use acvus_interpreter::{AsyncStorage, Codec, Held, Program, StorageError};
//! struct Remote;
//! impl AsyncStorage for Remote {
//!     async fn load(&mut self, _: &str, _: &Codec<'_>) -> Result<Option<Held>, StorageError> {
//!         Ok(None)
//!     }
//!     async fn store(&mut self, _: &str, _: Held) -> Result<(), StorageError> {
//!         Ok(())
//!     }
//!     async fn restore(&mut self, _: &str, _: Held) -> Result<(), StorageError> {
//!         Ok(())
//!     }
//!     async fn commit(&mut self, _: &Codec<'_>) -> Result<(), StorageError> {
//!         Ok(())
//!     }
//! }
//! async fn opened(program: &Program, remote: &mut Remote) {
//!     program.scope(async |s| { let _ = s.open(remote); }).await;
//! }
//! ```
//!
//! ```
//! use acvus_interpreter::{AsyncAccess, AsyncStorage, Codec, Held, Program, StorageError};
//! struct Remote;
//! impl AsyncStorage for Remote {
//!     async fn load(&mut self, _: &str, _: &Codec<'_>) -> Result<Option<Held>, StorageError> {
//!         Ok(None)
//!     }
//!     async fn store(&mut self, _: &str, _: Held) -> Result<(), StorageError> {
//!         Ok(())
//!     }
//!     async fn restore(&mut self, _: &str, _: Held) -> Result<(), StorageError> {
//!         Ok(())
//!     }
//!     async fn commit(&mut self, _: &Codec<'_>) -> Result<(), StorageError> {
//!         Ok(())
//!     }
//! }
//! async fn opened(program: &Program<AsyncAccess>, remote: &mut Remote) {
//!     program.scope(async |s| { let _ = s.open(remote); }).await;
//! }
//! ```
//!
//! A host lends a result or a context to a closure, and nothing the closure
//! is lent outlives it (RFC-0090 rule 5):
//!
//! ```compile_fail
//! use acvus_interpreter::{HostError, Output};
//! fn kept<'o>(output: &'o Output<'_, String>) -> Result<&'o str, HostError> {
//!     output.with(|s: &str| s)
//! }
//! ```
//!
//! ```compile_fail
//! use acvus_interpreter::{MemoryStorage, Page};
//! async fn kept<'a>(page: &'a mut Page<'_, '_, MemoryStorage>) -> Vec<&'a str> {
//!     let mut kept = Vec::new();
//!     let _ = page.with("name", |s: &str| kept.push(s)).await;
//!     kept
//! }
//! ```
//!
//! A value the host keeps is a copy it makes in Rust, and `with` lends
//! shared alone, so a parameter that writes is lent by `with_mut`:
//!
//! ```
//! use acvus_interpreter::{HostError, MemoryStorage, Output, Page};
//! fn kept(output: &Output<'_, String>) -> Result<String, HostError> {
//!     output.with(|s: &str| s.to_owned())
//! }
//! async fn pushed(page: &mut Page<'_, '_, MemoryStorage>) -> Result<(), HostError> {
//!     page.with_mut("name", |s: &mut String| s.push('!')).await
//! }
//! ```
//!
//! ```compile_fail
//! use acvus_interpreter::{HostError, Output};
//! fn pushed(output: &Output<'_, String>) -> Result<(), HostError> {
//!     output.with(|s: &mut String| s.push('!'))
//! }
//! ```
//!
//! A storage moves holders it cannot open, so implementing one is safe and
//! names no runtime value:
//!
//! ```
//! use std::collections::HashMap;
//! use acvus_interpreter::{Codec, Held, Storage, StorageError};
//! #[derive(Default)]
//! struct Shelf(HashMap<String, Held>);
//! impl Storage for Shelf {
//!     fn load(&mut self, key: &str, _: &Codec<'_>) -> Result<Option<Held>, StorageError> {
//!         Ok(self.0.remove(key))
//!     }
//!     fn store(&mut self, key: &str, held: Held) -> Result<(), StorageError> {
//!         self.0.insert(key.to_owned(), held);
//!         Ok(())
//!     }
//!     fn restore(&mut self, key: &str, held: Held) -> Result<(), StorageError> {
//!         self.0.entry(key.to_owned()).or_insert(held);
//!         Ok(())
//!     }
//!     fn commit(&mut self, _: &Codec<'_>) -> Result<(), StorageError> {
//!         Ok(())
//!     }
//! }
//! ```

#![cfg_attr(
    not(feature = "tooling"),
    doc = r#"
A crate without the `tooling` feature cannot reach the runtime's value
word (RFC-0090 rule 6): not the machine's entry,

```compile_fail,E0603
use acvus_interpreter::machine::call_module;
```

not a space's raw load,

```compile_fail,E0624
fn load(space: &acvus_interpreter::Space, rt: &acvus_interpreter::AcvusRuntime, ty: &acvus_mir::ty::Ty) {
    let _ = space.load(rt, "x", ty);
}
```

not the word inside a holder a page moves,

```compile_fail,E0616
fn open(held: acvus_interpreter::Held) {
    let _ = held.value;
}
```

```compile_fail,E0624
fn into(held: acvus_interpreter::Held) {
    let _ = held.into_value();
}
```

not a raw read of an entry's result,

```compile_fail,E0599
fn raw(scope: acvus_interpreter::Scope<'_>) {
    let _ = scope.untyped_entry("main");
}
```

and not the value word's constructors or readers, through the runtime's
associated type,

```compile_fail,E0624
use acvus_extern::Runtime;
let _ = <acvus_interpreter::AcvusRuntime as Runtime>::Value::int(3);
```

```compile_fail,E0616
fn port(rt: &acvus_interpreter::AcvusRuntime) {
    let _ = &rt.port;
}
```
"#
)]

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acvus_ast::Span;
use acvus_ast::report::Label;
use acvus_extern::{
    Borrows, CombineError, Cross, Crossing, Declared, Externs, Form, FormKind, Gives, Handlers,
    Holding, InstanceTable, Lendable, OneValue, Owned, Registry, Returned, Shared, SpaceError,
    Uniform, Val, lend_run,
};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{
    Access as GraphAccess, Bindings, BoundValue, CompilationGraph, Context, ContextInfo, FnKind, Function,
    Inputs as GraphInputs, NotABoundValue, Parsed, ParsedAst, QualifiedRef, RecoveredAst, extract, infer, lower,
    optimize,
};
use acvus_mir::ir::{MirModule, ValueId};
use acvus_mir::ty::{
    Effect, Flows, ParamTerm, Poly, PolyBuilder, PolyTy, Ty, TyTerm, lift_declaration,
    try_freeze_poly,
};
use acvus_utils::{Astr, Freeze, Interner};
use futures::FutureExt;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::executor::Executor;
use crate::init::{DeclaredInits, GraphParts, InitGiven, InitKey, InitSource, RustInit};
use crate::interpreter::{Executable, Interpreter, InterpreterContext, lookup_module};
use crate::ops::storage::{fetch_now, fetch_waited};
use crate::port::{Gate, Held, Port, ended, serve};
use crate::prepare::{PrepareCtx, prepare_module};
use crate::runtime::AcvusRuntime;
use crate::value::Value;

type Brand<'p> = PhantomData<fn(&'p ()) -> &'p ()>;

// -- Sources and refusals ------------------------------------------------

pub enum Source<'a> {
    Script(&'a str),
    Template(&'a str),
    Expr(&'a str),
}

macro_rules! source_parse {
    ($v:vis) => {
        impl Source<'_> {
            $v fn parse(&self, interner: &Interner) -> Parsed {
                match self {
                    Source::Script(text) | Source::Expr(text) => {
                        Parsed::script(acvus_ast::parse_script(interner, text))
                    }
                    Source::Template(text) => Parsed::template(acvus_ast::parse(interner, text)),
                }
            }
        }
    };
}
tooling_vis!(source_parse);

/// The body or binding whose source a refusal's span points into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Origin {
    Entry(String),
    Init(String),
    Binding(String),
}

#[derive(Debug, Clone)]
pub struct Refusal {
    /// `None` for a refusal of the compilation as a whole.
    pub origin: Option<Origin>,
    pub message: String,
    pub span: Option<Span>,
    /// The words the primary marker carries; `None` repeats the message.
    pub primary: Option<String>,
    pub labels: Vec<Label>,
    /// What was refused, for a refusal a host resolves in its own terms
    /// (RFC-0031 rule 7).
    pub cause: Option<Cause>,
}

/// A refusal a host resolves in its own terms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cause {
    /// The entry of the refusal's origin has a name a script calls as a
    /// bare name, which reaches these extern functions, each written with
    /// its namespace (RFC-0043).
    Shadows { externs: Vec<String> },
    /// The exposures of a host graph hold this cycle of hosts, each calling
    /// the next and the last calling the first (RFC-0095 rule 3).
    Cycle { hosts: Vec<String> },
}

impl Refusal {
    pub(crate) fn of(origin: Option<Origin>, message: String) -> Self {
        Refusal {
            origin,
            message,
            span: None,
            primary: None,
            labels: Vec::new(),
            cause: None,
        }
    }
}

// -- Errors --------------------------------------------------------------

/// What a host asked for that the compilation does not have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Named {
    Entry(String),
    Context(String),
}

/// Where a declared type and an asked one differ.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Part {
    Context(String),
    /// The inputs of the named entry.
    Inputs(String),
    /// The result of the named entry.
    Result(String),
}

#[derive(Debug)]
pub enum HostError {
    Refused(Vec<Refusal>),
    NotInGraph { what: Named },
    Mismatched { what: Part, held: String, asked: String },
    Unfilled { key: String },
    Storage(StorageError),
    Trapped { message: String },
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostError::Refused(refusals) => {
                write!(f, "the compilation is refused")?;
                for refusal in refusals {
                    write!(f, "; {}", refusal.message)?;
                }
                Ok(())
            }
            HostError::NotInGraph {
                what: Named::Entry(name),
            } => write!(f, "the compilation has no entry `{name}`"),
            HostError::NotInGraph {
                what: Named::Context(key),
            } => write!(f, "the compilation has no `@{key}`"),
            HostError::Mismatched {
                what: Part::Context(key),
                held,
                asked,
            } => write!(f, "`@{key}` is held as {held}, and was asked for as {asked}"),
            HostError::Mismatched {
                what: Part::Inputs(name),
                held,
                asked,
            } => write!(
                f,
                "the entry `{name}` was declared to take {held}, and was asked for taking {asked}"
            ),
            HostError::Mismatched {
                what: Part::Result(name),
                held,
                asked,
            } => write!(
                f,
                "the entry `{name}` was declared to return {held}, and was asked for as {asked}"
            ),
            HostError::Unfilled { key } => {
                write!(f, "`@{key}` has neither a value in the storage nor an init")
            }
            HostError::Storage(error) => write!(f, "the storage refused: {error}"),
            HostError::Trapped { message } => write!(f, "the run trapped: {message}"),
        }
    }
}

impl std::error::Error for HostError {}

impl From<StorageError> for HostError {
    fn from(error: StorageError) -> Self {
        HostError::Storage(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageError {
    message: String,
}

impl StorageError {
    pub fn new(message: String) -> Self {
        StorageError { message }
    }
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for StorageError {}

impl From<SpaceError> for StorageError {
    fn from(error: SpaceError) -> Self {
        StorageError::new(error.to_string())
    }
}

// -- Storage -------------------------------------------------------------

/// The runtime a storage that encodes holders reaches the encoding through.
/// A storage that keeps holders whole ignores it.
pub struct Codec<'a> {
    rt: &'a AcvusRuntime,
}

impl<'a> Codec<'a> {
    pub(crate) fn of(rt: &'a AcvusRuntime) -> Self {
        Codec { rt }
    }

    pub(crate) fn rt(&self) -> &'a AcvusRuntime {
        self.rt
    }
}

/// Where a page's contexts live. A page holds no value: the storage is
/// read at a load and written at a store, where a run or the host makes one
/// (RFC-0090 rule 3). `load` moves a holder out, and `store` or `restore`
/// hands one back.
pub trait Storage: Send {
    fn load(&mut self, key: &str, codec: &Codec<'_>) -> Result<Option<Held>, StorageError>;
    /// Take a holder whose value may have changed since it was loaded, or
    /// one no load gave.
    fn store(&mut self, key: &str, held: Held) -> Result<(), StorageError>;
    /// Take back a holder `load` gave whose value nothing changed. A holder
    /// stored for `key` since that load is newer and stays.
    fn restore(&mut self, key: &str, held: Held) -> Result<(), StorageError>;
    /// Make what the storage holds durable.
    fn commit(&mut self, codec: &Codec<'_>) -> Result<(), StorageError>;
}

/// A storage whose access can wait. A program compiled with
/// `Host::async_access` opens one; every synchronous `Storage` is one.
pub trait AsyncStorage: Send {
    fn load(
        &mut self,
        key: &str,
        codec: &Codec<'_>,
    ) -> impl Future<Output = Result<Option<Held>, StorageError>> + Send;
    fn store(&mut self, key: &str, held: Held) -> impl Future<Output = Result<(), StorageError>> + Send;
    fn restore(&mut self, key: &str, held: Held) -> impl Future<Output = Result<(), StorageError>> + Send;
    fn commit(&mut self, codec: &Codec<'_>) -> impl Future<Output = Result<(), StorageError>> + Send;
}

impl<S> AsyncStorage for S
where
    S: Storage,
{
    fn load(
        &mut self,
        key: &str,
        codec: &Codec<'_>,
    ) -> impl Future<Output = Result<Option<Held>, StorageError>> + Send {
        std::future::ready(Storage::load(self, key, codec))
    }

    fn store(&mut self, key: &str, held: Held) -> impl Future<Output = Result<(), StorageError>> + Send {
        std::future::ready(Storage::store(self, key, held))
    }

    fn restore(&mut self, key: &str, held: Held) -> impl Future<Output = Result<(), StorageError>> + Send {
        std::future::ready(Storage::restore(self, key, held))
    }

    fn commit(&mut self, codec: &Codec<'_>) -> impl Future<Output = Result<(), StorageError>> + Send {
        std::future::ready(Storage::commit(self, codec))
    }
}

/// A storage whose holders are the data: it is not durable, and `commit`
/// changes nothing. A key keeps its slot while its holder is lent, so a load
/// and the restore after it allocate nothing.
#[derive(Default)]
pub struct MemoryStorage {
    holders: HashMap<String, Option<Held>>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        MemoryStorage::default()
    }
}

impl Storage for MemoryStorage {
    fn load(&mut self, key: &str, _: &Codec<'_>) -> Result<Option<Held>, StorageError> {
        Ok(self.holders.get_mut(key).and_then(Option::take))
    }

    fn store(&mut self, key: &str, held: Held) -> Result<(), StorageError> {
        match self.holders.get_mut(key) {
            Some(slot) => *slot = Some(held),
            None => {
                self.holders.insert(key.to_owned(), Some(held));
            }
        }
        Ok(())
    }

    fn restore(&mut self, key: &str, held: Held) -> Result<(), StorageError> {
        match self.holders.get_mut(key) {
            Some(slot) => {
                slot.get_or_insert(held);
            }
            None => {
                self.holders.insert(key.to_owned(), Some(held));
            }
        }
        Ok(())
    }

    fn commit(&mut self, _: &Codec<'_>) -> Result<(), StorageError> {
        Ok(())
    }
}

// -- Access --------------------------------------------------------------

/// A program's access to its storage, fixed when it compiles (RFC-0090
/// rule 3): `SyncAccess` opens a `Storage`, `AsyncAccess` an `AsyncStorage`.
pub trait Access: sealed::Sealed {
    #[doc(hidden)]
    const GRAPH: GraphAccess;
}

mod sealed {
    pub trait Sealed {}
}

pub struct SyncAccess;

pub struct AsyncAccess;

impl sealed::Sealed for SyncAccess {}

impl sealed::Sealed for AsyncAccess {}

impl Access for SyncAccess {
    const GRAPH: GraphAccess = GraphAccess::Sync;
}

impl Access for AsyncAccess {
    const GRAPH: GraphAccess = GraphAccess::Async;
}

// -- The graph environment ----------------------------------------------

struct Environment {
    functions: Vec<Function>,
    types: acvus_mir::ty::TypeRegistry,
    handlers: Handlers<AcvusRuntime>,
    space: crate::layout::Hooks,
    instances: InstanceTable,
    fn_types: FxHashMap<QualifiedRef, Ty>,
}

impl Environment {
    fn combine(interner: &Interner, registries: Vec<Registry<AcvusRuntime>>) -> Result<Self, CombineError> {
        let Externs {
            functions,
            types,
            handlers,
            space,
            instances,
        } = Externs::combine(registries, interner)?;
        let fn_types = functions
            .iter()
            .filter_map(|f| try_freeze_poly(&f.ty).map(|ty| (f.qref, ty)))
            .collect();
        Ok(Environment {
            functions,
            types,
            handlers,
            space,
            instances,
            fn_types,
        })
    }
}

/// Each named context at a type variable the whole graph solves (RFC-0090
/// rule 1), in name order.
fn open_contexts(open: &mut PolyBuilder, named: FxHashSet<QualifiedRef>) -> Vec<Context> {
    let mut contexts: Vec<Context> = named
        .into_iter()
        .map(|qref| Context {
            qref,
            ty: open.fresh_ty_var(),
            init: None,
        })
        .collect();
    contexts.sort_by_key(|context| context.qref.name.bits());
    contexts
}

macro_rules! tooling_graph {
    ($v:vis) => {
        /// The environment an editor checks a source in: the registries'
        /// functions and types and each named context, the graph `compile`
        /// builds before it adds the entries' functions.
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        $v fn environment(
            interner: &Interner,
            named: FxHashSet<QualifiedRef>,
            entries: Vec<QualifiedRef>,
            registries: Vec<Registry<AcvusRuntime>>,
        ) -> Result<CompilationGraph, CombineError> {
            let environment = Environment::combine(interner, registries)?;
            let contexts = open_contexts(&mut PolyBuilder::new(), named);
            Ok(CompilationGraph {
                functions: Freeze::new(environment.functions),
                contexts: Freeze::new(contexts),
                types: Freeze::new(environment.types),
                bindings: Bindings::default(),
                access: acvus_mir::graph::Access::Sync,
                entries,
            })
        }

        /// The type of an entry that declares `!` and reads its `$` inputs
        /// from its body (RFC-0054 rule 5, RFC-0090 rule 6).
        $v fn untyped_entry_ty() -> PolyTy {
            TyTerm::Fn {
                params: vec![],
                ret: Box::new(lift_declaration(&Ty::Never, &mut PolyBuilder::new())),
                captures: vec![],
                effect: Effect::OPAQUE.into(),
                flows: Flows::Every.into(),
            }
        }

        $v fn context_refs(ast: &ParsedAst) -> FxHashSet<QualifiedRef> {
            match ast {
                ParsedAst::Script(script) => acvus_ast::extract_script_context_refs(script),
                ParsedAst::Template(template) => acvus_ast::extract_template_context_refs(template),
                ParsedAst::Recovered(RecoveredAst::Script(script)) => {
                    acvus_ast::extract_script_context_refs(script)
                }
                ParsedAst::Recovered(RecoveredAst::Template(template)) => {
                    acvus_ast::extract_template_context_refs(template)
                }
            }
        }
    };
}
tooling_vis!(tooling_graph);

// -- Host ----------------------------------------------------------------

// -- Input shapes --------------------------------------------------------

/// An entry's `$` inputs built from data: named fields, each typed by a
/// Rust `T` (RFC-0090 rule 2).
#[derive(Default)]
pub struct InputShape {
    fields: Vec<ShapeField>,
}

struct ShapeField {
    name: String,
    declared: fn(&Interner) -> PolyTy,
}

impl InputShape {
    pub fn new() -> Self {
        InputShape::default()
    }

    pub fn field<T>(mut self, name: &str) -> Self
    where
        T: Declared + OneValue<AcvusRuntime>,
    {
        self.fields.push(ShapeField {
            name: name.to_owned(),
            declared: T::declared,
        });
        self
    }
}

impl fmt::Debug for InputShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.fields.iter().map(|field| &field.name)).finish()
    }
}

#[derive(Clone)]
pub(crate) struct ResolvedShape {
    pub(crate) fields: Vec<(Astr, PolyTy)>,
}

impl ResolvedShape {
    fn of_declared(interner: &Interner, ty: &PolyTy) -> Option<ResolvedShape> {
        match ty {
            TyTerm::Unit => Some(ResolvedShape { fields: Vec::new() }),
            TyTerm::Object(fields) => Some(ResolvedShape::sorted(
                interner,
                fields.iter().map(|(name, ty)| (*name, ty.clone())).collect(),
            )),
            _ => None,
        }
    }

    fn of_fields(interner: &Interner, shape: &InputShape) -> ResolvedShape {
        let fields = shape
            .fields
            .iter()
            .map(|field| (interner.intern(&field.name), (field.declared)(interner)))
            .collect();
        ResolvedShape::sorted(interner, fields)
    }

    /// A derived struct's crossing writes its fields in the order its derive
    /// sorted their names as strings (RFC-0050 rule 8), and a run pairs those
    /// values with these fields position by position, so this sort and the
    /// derive's in `acvus-extern-macro` must stay the same comparison.
    fn sorted(interner: &Interner, mut fields: Vec<(Astr, PolyTy)>) -> ResolvedShape {
        fields.sort_by(|(a, _), (b, _)| interner.resolve(*a).cmp(interner.resolve(*b)));
        ResolvedShape { fields }
    }

    fn len(&self) -> usize {
        self.fields.len()
    }

    fn position_of(&self, interner: &Interner, name: &str) -> Option<usize> {
        self.fields
            .binary_search_by(|(held, _)| interner.resolve(*held).cmp(name))
            .ok()
    }

    fn repeated_names(&self) -> Vec<Astr> {
        let mut repeated: Vec<Astr> = self
            .fields
            .windows(2)
            .filter(|pair| pair[0].0 == pair[1].0)
            .map(|pair| pair[0].0)
            .collect();
        repeated.dedup();
        repeated
    }

    fn same_fields(&self, other: &ResolvedShape) -> bool {
        self.fields.len() == other.fields.len()
            && self
                .fields
                .iter()
                .zip(&other.fields)
                .all(|((a, a_ty), (b, b_ty))| a == b && a_ty.same_erased(b_ty))
    }

    fn display(&self, interner: &Interner) -> String {
        shape_display(
            self.fields
                .iter()
                .map(|(name, ty)| (interner.resolve(*name), ty.display(interner).to_string())),
        )
    }
}

fn shape_display<'n>(fields: impl Iterator<Item = (&'n str, String)>) -> String {
    let written: Vec<String> = fields.map(|(name, ty)| format!("{name}: {ty}")).collect();
    match written.is_empty() {
        true => "()".to_owned(),
        false => format!("{{{}}}", written.join(", ")),
    }
}

/// The values of an entry's inputs built from data, one per field of its
/// shape (RFC-0090 rule 2).
#[derive(Default)]
pub struct Inputs {
    given: Vec<Given>,
}

struct Given {
    name: String,
    declared: fn(&Interner) -> PolyTy,
    cross: Box<dyn for<'c> FnOnce(Crossing<'c, AcvusRuntime>) -> Value + Send>,
}

impl Inputs {
    pub fn new() -> Self {
        Inputs::default()
    }

    pub fn set<T>(mut self, name: &str, value: T) -> Self
    where
        T: Declared + OneValue<AcvusRuntime> + Send,
    {
        self.given.push(Given {
            name: name.to_owned(),
            declared: T::declared,
            cross: Box::new(move |crossing| value.erase(crossing)),
        });
        self
    }
}

impl fmt::Debug for Inputs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.given.iter().map(|given| &given.name)).finish()
    }
}

pub(crate) enum EntryDeclaration {
    Typed { inputs: ResolvedShape, declared: PolyTy },
    Untyped,
    /// A body scripts call and no host runs: its return is what its body
    /// settles, and its `$` inputs are the ones it reads (RFC-0071 rule 4).
    Function,
    Positional { params: Vec<ParamTerm<Poly>>, ret: PolyTy },
}

#[derive(Clone)]
pub(crate) struct GraphName {
    pub(crate) host: Option<String>,
    pub(crate) namespace: Option<String>,
    pub(crate) name: String,
}

impl GraphName {
    pub(crate) fn qref(&self, interner: &Interner) -> QualifiedRef {
        QualifiedRef {
            namespace: self
                .namespace
                .as_deref()
                .map(|namespace| interner.intern(namespace)),
            name: interner.intern(&self.name),
            host: self.host.as_deref().map(|host| interner.intern(host)),
        }
    }
}

/// The key a program's caller names an entry or a context by, and a
/// refusal's origin shows: `host/name` for a host of a graph. Storage holds
/// a context under this key.
pub(crate) fn program_key(host: Option<&str>, name: &str) -> String {
    match host {
        Some(host) => format!("{host}/{name}"),
        None => name.to_owned(),
    }
}

fn program_key_of(interner: &Interner, qref: QualifiedRef) -> String {
    let name = match qref.namespace {
        Some(namespace) => format!(
            "{}::{}",
            interner.resolve(namespace),
            interner.resolve(qref.name)
        ),
        None => interner.resolve(qref.name).to_owned(),
    };
    program_key(qref.host.map(|host| interner.resolve(host)), &name)
}

pub(crate) struct EntryDecl {
    pub(crate) name: GraphName,
    pub(crate) ast: ParsedAst,
    pub(crate) declaration: EntryDeclaration,
}

/// The entries, inits and bindings of one compilation graph, so a context's
/// type is solved from every body that stores, reads or initializes it
/// (RFC-0090 rule 1).
pub struct Host<A = SyncAccess> {
    pub(crate) parts: HostParts,
    access: PhantomData<fn() -> A>,
}

pub(crate) struct HostParts {
    pub(crate) interner: Interner,
    pub(crate) registries: Vec<Registry<AcvusRuntime>>,
    pub(crate) bindings: Bindings,
    pub(crate) entries: Vec<EntryDecl>,
    pub(crate) inits: Vec<InitSource>,
    pub(crate) parse_refusals: Vec<Refusal>,
    pub(crate) refusals: Vec<Refusal>,
    pub(crate) opt: Opt,
    pub(crate) parse: Duration,
}

impl Host<SyncAccess> {
    pub fn new(registries: Vec<Registry<AcvusRuntime>>) -> Self {
        Host::over(Interner::new(), registries)
    }

    pub(crate) fn in_graph(interner: &Interner) -> Self {
        Host::over(interner.clone(), Vec::new())
    }

    fn over(interner: Interner, registries: Vec<Registry<AcvusRuntime>>) -> Self {
        Host {
            parts: HostParts {
                interner,
                registries,
                bindings: Bindings::default(),
                entries: Vec::new(),
                inits: Vec::new(),
                parse_refusals: Vec::new(),
                refusals: Vec::new(),
                opt: Opt::Full,
                parse: Duration::ZERO,
            },
            access: PhantomData,
        }
    }

    /// Compile for a storage whose access can wait: every fetch and commit
    /// of a context is `Async` (RFC-0090 rule 3).
    pub fn async_access(self) -> Host<AsyncAccess> {
        Host {
            parts: self.parts,
            access: PhantomData,
        }
    }
}

impl<A> Host<A>
where
    A: Access,
{
    fn parsed(&mut self, origin: Origin, source: Source<'_>) -> ParsedAst {
        let parts = &mut self.parts;
        let started = Instant::now();
        let Parsed { ast, errors } = source.parse(&parts.interner);
        parts.parse += started.elapsed();
        parts.parse_refusals.extend(errors.iter().map(|e| Refusal {
            span: span_of(e.span),
            ..Refusal::of(Some(origin.clone()), e.kind.to_string())
        }));
        ast
    }

    pub fn init(mut self, key: &str, source: Source<'_>) -> Self {
        let ast = self.parsed(Origin::Init(key.to_owned()), source);
        self.parts.inits.push(InitSource {
            key: InitKey {
                host: None,
                written: key.to_owned(),
            },
            given: InitGiven::Source(ast),
        });
        self
    }

    /// A key's first value made by `make`, which runs at every load that
    /// finds the storage lacking the key, on whichever thread the load runs.
    pub fn init_with<T, F>(mut self, key: &str, make: F) -> Self
    where
        T: Declared + OneValue<AcvusRuntime>,
        F: Fn() -> T + Send + Sync + 'static,
    {
        let declared = T::declared(&self.parts.interner);
        let make = Box::new(move |crossing: Crossing<'_, AcvusRuntime>| Owned::erased(crossing, make()));
        self.parts.inits.push(InitSource {
            key: InitKey {
                host: None,
                written: key.to_owned(),
            },
            given: InitGiven::Rust(RustInit::new(declared, make)),
        });
        self
    }

    pub fn entry<I, R>(mut self, name: &str, source: Source<'_>) -> Self
    where
        I: Declared,
        R: Declared,
    {
        let origin = Some(Origin::Entry(name.to_owned()));
        let interner = &self.parts.interner;
        let asked = I::declared(interner);
        let Some(inputs) = ResolvedShape::of_declared(interner, &asked) else {
            let message = format!(
                "the inputs of the entry `{name}` are declared as {}, which is neither `()` \
                 nor a struct of named fields",
                asked.display(interner)
            );
            self.parts.refusals.push(Refusal::of(origin, message));
            return self;
        };
        self.typed_entry::<R>(name, inputs, source)
    }

    pub fn entry_shaped<R>(self, name: &str, shape: InputShape, source: Source<'_>) -> Self
    where
        R: Declared,
    {
        let inputs = ResolvedShape::of_fields(&self.parts.interner, &shape);
        self.typed_entry::<R>(name, inputs, source)
    }

    fn typed_entry<R>(self, name: &str, inputs: ResolvedShape, source: Source<'_>) -> Self
    where
        R: Declared,
    {
        let declared = R::declared(&self.parts.interner);
        self.declare_entry(name, source, EntryDeclaration::Typed { inputs, declared })
    }

    fn declare_entry(
        mut self,
        name: &str,
        source: Source<'_>,
        declaration: EntryDeclaration,
    ) -> Self {
        let origin = Origin::Entry(name.to_owned());
        if self
            .parts
            .entries
            .iter()
            .any(|entry| entry.name.name == name)
        {
            let message = format!("the entry `{name}` is given twice");
            self.parts
                .refusals
                .push(Refusal::of(Some(origin.clone()), message));
        }
        let ast = self.parsed(origin, source);
        self.parts.entries.push(EntryDecl {
            name: GraphName {
                host: None,
                namespace: None,
                name: name.to_owned(),
            },
            ast,
            declaration,
        });
        self
    }

    /// Fix the input `$name` to the value `literal` writes, in the script's
    /// own syntax (RFC-0087 rule 1).
    pub fn bind(mut self, name: &str, literal: &str) -> Result<Self, HostError> {
        let refused = |message: String| {
            HostError::Refused(vec![Refusal::of(Some(Origin::Binding(name.to_owned())), message)])
        };
        let interner = &self.parts.interner;
        let expr = acvus_ast::parse_expr(interner, literal)
            .map_err(|e| refused(format!("${name}: `{literal}` does not parse: {e:?}")))?;
        let written = |span: Span| &literal[span.start..span.end];
        let value = BoundValue::from_expr(interner, &expr).map_err(|e| match e {
            NotABoundValue::NotALiteral { form } => refused(format!(
                "${name}: `{}` is not a value a literal writes",
                written(form)
            )),
            NotABoundValue::RepeatedField { key, second } => refused(format!(
                "${name}: `{}` in `{}` is a field an object holds once",
                interner.resolve(key),
                written(second)
            )),
        })?;
        self.parts
            .bindings
            .bind(QualifiedRef::root(interner.intern(name)), value)
            .map_err(|e| refused(format!("${name}: {e}")))?;
        Ok(self)
    }

    pub fn compile<E>(self, executor: E) -> Result<Program<A>, HostError>
    where
        E: Executor + 'static,
    {
        compile(self.parts, A::GRAPH, Arc::new(executor), &[])
            .map(Program::of)
            .map_err(HostError::Refused)
    }
}

macro_rules! host_tooling {
    ($v:vis) => {
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl<A> Host<A>
        where
            A: Access,
        {
            $v fn opt(mut self, opt: Opt) -> Self {
                self.parts.opt = opt;
                self
            }

            /// An entry that declares `!` and whose `$` inputs are the ones
            /// its body reads: the CLI's, which prints whatever comes back
            /// (RFC-0054 rule 5, RFC-0090 rule 6).
            $v fn untyped_entry(self, name: &str, source: Source<'_>) -> Self {
                self.declare_entry(name, source, EntryDeclaration::Untyped)
            }

            /// A body the entries call by name and no host runs: the CLI's
            /// scripts beside the one it runs (RFC-0054 rule 1).
            $v fn function(self, name: &str, source: Source<'_>) -> Self {
                self.declare_entry(name, source, EntryDeclaration::Function)
            }
        }
    };
}
tooling_vis!(host_tooling);

enum EntryShape {
    Typed { inputs: ResolvedShape, declared: PolyTy },
    Untyped,
}

enum CompiledShape {
    Typed { inputs: EntryInputs, declared: PolyTy },
    Untyped,
}

struct EntryInputs {
    shape: ResolvedShape,
    param_of: Vec<usize>,
    params: usize,
}

impl EntryInputs {
    fn of(interner: &Interner, shape: ResolvedShape, params: &[(Astr, ValueId)]) -> EntryInputs {
        let param_of: Vec<usize> = shape
            .fields
            .iter()
            .map(|(field, _)| {
                params.iter().position(|(param, _)| param == field).unwrap_or_else(|| {
                    panic!(
                        "the entry's declared input `${}` is no parameter of its module: a typed \
                         entry's parameters are its declared inputs",
                        interner.resolve(*field)
                    )
                })
            })
            .collect();
        if let Some((param, _)) = params
            .iter()
            .find(|(param, _)| shape.fields.iter().all(|(field, _)| field != param))
        {
            panic!(
                "the entry's module takes `${}`, which its declared inputs lack: a typed entry's \
                 parameters are its declared inputs",
                interner.resolve(*param)
            )
        }
        EntryInputs {
            shape,
            param_of,
            params: params.len(),
        }
    }

    fn checked(&self, interner: &Interner, entry: &str, inputs: Inputs) -> Result<Vec<Given>, HostError> {
        let shape = &self.shape;
        let at: Option<Vec<usize>> = inputs
            .given
            .iter()
            .map(|given| {
                let at = shape.position_of(interner, &given.name)?;
                (given.declared)(interner)
                    .same_erased(&shape.fields[at].1)
                    .then_some(at)
            })
            .collect();
        let mut taken = vec![false; shape.len()];
        let each_once = at.as_ref().is_some_and(|at| {
            at.iter().all(|at| !std::mem::replace(&mut taken[*at], true)) && taken.iter().all(|taken| *taken)
        });
        let (Some(at), true) = (at, each_once) else {
            let mut asked: Vec<(&str, String)> = inputs
                .given
                .iter()
                .map(|given| (given.name.as_str(), (given.declared)(interner).display(interner).to_string()))
                .collect();
            asked.sort();
            return Err(HostError::Mismatched {
                what: Part::Inputs(entry.to_owned()),
                held: shape.display(interner),
                asked: shape_display(asked.into_iter()),
            });
        };
        let mut given: Vec<(usize, Given)> = at.into_iter().zip(inputs.given).collect();
        given.sort_by_key(|(at, _)| *at);
        Ok(given.into_iter().map(|(_, given)| given).collect())
    }

    fn arguments<V>(&self, crossing: Crossing<'_, AcvusRuntime>, values: Vec<V>) -> Vec<Value>
    where
        V: FieldValue,
    {
        assert_eq!(
            values.len(),
            self.param_of.len(),
            "a run gives one value per field of the entry's shape"
        );
        let mut arguments: Vec<Value> = std::iter::repeat_with(Value::unit).take(self.params).collect();
        for (at, value) in self.param_of.iter().zip(values) {
            arguments[*at] = value.into_word(crossing);
        }
        arguments
    }
}

trait FieldValue {
    fn into_word(self, crossing: Crossing<'_, AcvusRuntime>) -> Value;
}

/// A derived struct's field, crossed into its own holder.
struct CrossedField(Owned<AcvusRuntime>);

impl FieldValue for CrossedField {
    fn into_word(self, crossing: Crossing<'_, AcvusRuntime>) -> Value {
        self.0.into_value(crossing.holding())
    }
}

impl FieldValue for Given {
    fn into_word(self, crossing: Crossing<'_, AcvusRuntime>) -> Value {
        (self.cross)(crossing)
    }
}

struct Declaration {
    name: String,
    shape: EntryShape,
}

struct LocalFunction {
    kind: FnKind,
    ty: PolyTy,
    /// `None` for a function no host runs.
    shape: Option<EntryShape>,
}

struct CompiledEntry {
    name: String,
    qref: QualifiedRef,
    shape: CompiledShape,
    ret: Ty,
    #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
    module: MirModule,
    #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
    required: Vec<ContextInfo>,
}

fn span_of(span: Span) -> Option<Span> {
    (span.start != 0 || span.end != 0).then_some(span)
}

/// How long each stage of a compilation took.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(feature = "tooling"), allow(dead_code))]
pub struct CompileTimes {
    pub opt: Opt,
    pub parse: Duration,
    /// `extract` and `infer`, which are not reported apart.
    pub typeck: Duration,
    pub lower: Duration,
    pub optimize: Duration,
    pub prepare: Duration,
}

#[cfg_attr(not(feature = "tooling"), allow(dead_code))]
impl CompileTimes {
    /// Every stage before `prepare`, which belongs to the interpreter.
    pub fn check(&self) -> Duration {
        self.parse + self.typeck + self.lower + self.optimize
    }
}

pub(crate) fn compile(
    host: HostParts,
    access: GraphAccess,
    executor: Arc<dyn Executor>,
    exposed: &[crate::host_graph::Exposed],
) -> Result<Compiled, Vec<Refusal>> {
    let HostParts {
        interner,
        registries,
        bindings,
        entries,
        inits,
        parse_refusals,
        refusals: structural,
        opt,
        parse,
    } = host;
    let interner = &interner;
    // A recovered tree is checked for what parsed and never lowered, so its
    // parse errors and those refusals are reported together (RFC-0078
    // rule 5); a structural refusal ends the compilation before typeck.
    let mut refusals = parse_refusals;
    let structural_at = refusals.len();
    refusals.extend(structural);

    let environment = match Environment::combine(interner, registries) {
        Ok(environment) => environment,
        Err(error) => {
            let message = format!("the registries do not combine: {error}");
            refusals.push(Refusal::of(None, message));
            return Err(refusals);
        }
    };
    let Environment {
        functions: extern_fns,
        types,
        handlers,
        space,
        instances,
        fn_types,
    } = environment;

    let mut open = PolyBuilder::new();
    let mut named: FxHashSet<QualifiedRef> = FxHashSet::default();
    let mut declared: FxHashMap<QualifiedRef, Declaration> = FxHashMap::default();
    let mut scripts: FxHashMap<QualifiedRef, String> = FxHashMap::default();
    let mut functions: Vec<Function> = Vec::with_capacity(entries.len() + extern_fns.len());
    for EntryDecl {
        name: written,
        ast,
        declaration,
    } in entries
    {
        let qref = written.qref(interner);
        let name = program_key_of(interner, qref);
        let origin = Some(Origin::Entry(name.clone()));
        let bare = written.namespace.is_none().then_some(written.name.as_str());
        let shadowed = match bare {
            Some(bare) => bare_callable(
                interner,
                &extern_fns,
                &types,
                QualifiedRef::root(interner.intern(bare)),
            ),
            None => Vec::new(),
        };
        if let (Some(bare), false) = (bare, shadowed.is_empty()) {
            let listed: Vec<String> = shadowed.iter().map(|name| format!("`{name}`")).collect();
            let what = match declaration {
                EntryDeclaration::Function | EntryDeclaration::Positional { .. } => "function",
                EntryDeclaration::Typed { .. } | EntryDeclaration::Untyped => "entry",
            };
            let message = format!(
                "the {what} `{name}` would shadow {}, which a script calls as `{bare}`",
                listed.join(", ")
            );
            refusals.push(Refusal {
                cause: Some(Cause::Shadows { externs: shadowed }),
                ..Refusal::of(origin.clone(), message)
            });
        }
        named.extend(
            context_refs(&ast)
                .into_iter()
                .map(|context| context.in_host(qref.host)),
        );
        let local = match declaration {
            EntryDeclaration::Typed { inputs, declared } => {
                refusals.extend(inputs.repeated_names().into_iter().map(|repeated| {
                    let message = format!(
                        "the input `${}` of the entry `{name}` is given twice",
                        interner.resolve(repeated)
                    );
                    Refusal::of(origin.clone(), message)
                }));
                refusals.extend(
                    inputs
                        .fields
                        .iter()
                        .filter(|(field, _)| bindings.get(QualifiedRef::root(*field).in_host(qref.host)).is_some())
                        .map(|(field, _)| {
                            let message = format!(
                                "the input `${}` of the entry `{name}` is already fixed by a binding",
                                interner.resolve(*field)
                            );
                            Refusal::of(origin.clone(), message)
                        }),
                );
                let params = inputs
                    .fields
                    .iter()
                    .map(|(field, ty)| ParamTerm::new(*field, ty.clone()))
                    .collect();
                let ty = TyTerm::Fn {
                    params,
                    ret: Box::new(declared.clone()),
                    captures: vec![],
                    effect: Effect::OPAQUE.into(),
                    flows: Flows::Every.into(),
                };
                let shape = EntryShape::Typed { inputs, declared };
                LocalFunction {
                    kind: FnKind::Local(ast, GraphInputs::Declared),
                    ty,
                    shape: Some(shape),
                }
            }
            EntryDeclaration::Untyped => LocalFunction {
                kind: FnKind::Local(ast, GraphInputs::FromReads),
                ty: untyped_entry_ty(),
                shape: Some(EntryShape::Untyped),
            },
            EntryDeclaration::Positional { params, ret } => LocalFunction {
                kind: FnKind::Local(ast, GraphInputs::Declared),
                ty: TyTerm::Fn {
                    params,
                    ret: Box::new(ret),
                    captures: vec![],
                    effect: Effect::OPAQUE.into(),
                    flows: Flows::Every.into(),
                },
                shape: None,
            },
            EntryDeclaration::Function => LocalFunction {
                kind: FnKind::Local(ast, GraphInputs::FromReads),
                ty: TyTerm::Fn {
                    params: vec![],
                    ret: Box::new(open.fresh_ty_var()),
                    captures: vec![],
                    effect: Effect::OPAQUE.into(),
                    flows: Flows::Every.into(),
                },
                shape: None,
            },
        };
        let LocalFunction { kind, ty, shape } = local;
        scripts.insert(qref, name.clone());
        if let Some(shape) = shape {
            declared.insert(qref, Declaration { name, shape });
        }
        functions.push(Function { qref, kind, ty });
    }
    let entry_refs: Vec<QualifiedRef> = functions
        .iter()
        .map(|f| f.qref)
        .filter(|qref| declared.contains_key(qref))
        .collect();
    functions.extend(extern_fns);

    let contexts = open_contexts(&mut open, named);
    let mut parts = GraphParts {
        open,
        contexts,
        functions,
        entries: entry_refs,
    };
    let declared_inits = match DeclaredInits::declare(interner, inits, &mut parts) {
        Ok(declared) => declared,
        Err(refused) => {
            refusals.extend(refused.into_iter().map(|refusal| {
                Refusal::of(
                    Some(Origin::Init(refusal.key().stored())),
                    refusal.to_string(),
                )
            }));
            return Err(refusals);
        }
    };
    if refusals.len() > structural_at {
        return Err(refusals);
    }
    let GraphParts {
        contexts,
        functions,
        entries: entry_refs,
        ..
    } = parts;
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        types: Freeze::new(types),
        bindings,
        access,
        entries: entry_refs,
    };
    let origin_of = |qref: &QualifiedRef| match scripts.get(qref) {
        Some(name) => Some(Origin::Entry(name.clone())),
        None => declared_inits
            .key_of(qref)
            .map(|key| Origin::Init(key.stored())),
    };

    let started = Instant::now();
    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext);
    let typeck = started.elapsed();
    refusals.extend(crate::host_graph::crossing_refusals(interner, &graph.types, &inf, exposed));
    refusals.extend(inf.errors().into_iter().flat_map(|(qref, errs)| {
        let origin = origin_of(&qref);
        errs.iter().map(move |e| Refusal {
            origin: origin.clone(),
            message: e.display(interner).to_string(),
            span: span_of(e.span),
            primary: e.primary(),
            labels: e.labels.clone(),
            cause: None,
        })
    }));

    let started = Instant::now();
    let lowered = lower::lower(interner, &graph, &ext.view(), &inf);
    let lower = started.elapsed();
    refusals.extend(lowered.errors.iter().flat_map(|le| {
        let origin = origin_of(&le.fn_id);
        le.errors.iter().map(move |e| Refusal {
            origin: origin.clone(),
            message: e.display(interner).to_string(),
            span: span_of(e.span),
            primary: e.primary(),
            labels: e.labels.clone(),
            cause: None,
        })
    }));
    if !refusals.is_empty() {
        return Err(refusals);
    }

    let started = Instant::now();
    let laws = acvus_mir::laws::LawTable::of(graph.functions.iter());
    let optimized = optimize::optimize(interner, &laws, lowered.modules, opt);
    let optimize = started.elapsed();
    refusals.extend(optimized.errors.into_iter().flat_map(|(qref, errs)| {
        let origin = origin_of(&qref);
        errs.into_iter().map(move |e| Refusal {
            origin: origin.clone(),
            message: e.display(interner).to_string(),
            span: span_of(e.span),
            primary: None,
            labels: e.labels().to_vec(),
            cause: None,
        })
    }));
    if !refusals.is_empty() {
        return Err(refusals);
    }

    let context_names: FxHashMap<QualifiedRef, Astr> = graph
        .contexts
        .iter()
        .map(|c| (c.qref, interner.intern(&program_key_of(interner, c.qref))))
        .collect();
    let mut executables: FxHashMap<QualifiedRef, Executable> = handlers
        .into_iter()
        .map(|(q, h)| (q, Executable::Extern(h)))
        .collect();
    let started = Instant::now();
    let mut prepared: Vec<(QualifiedRef, Executable)> = Vec::new();
    {
        let ctx = PrepareCtx {
            interner,
            externs: &executables,
            context_names: &context_names,
            instances: &instances,
            access,
        };
        for (q, m) in &optimized.modules {
            match prepare_module(m, &ctx) {
                Ok(module) => prepared.push((*q, Executable::Module(Arc::new(module)))),
                Err(refused) => refusals.push(Refusal {
                    origin: origin_of(q),
                    message: refused.to_string(),
                    span: span_of(refused.span),
                    primary: None,
                    labels: Vec::new(),
                    cause: None,
                }),
            }
        }
    }
    if !refusals.is_empty() {
        return Err(refusals);
    }
    let prepare = started.elapsed();
    executables.extend(prepared);
    let mut modules = optimized.modules;
    let mut required = optimized.inputs;
    let compiled_entries: HashMap<String, CompiledEntry> = declared
        .into_iter()
        .map(|(qref, Declaration { name, shape })| {
            let (Some(module), Some(required)) = (modules.remove(&qref), required.remove(&qref))
            else {
                panic!("`optimize` keeps a module and its inputs for an entry no stage refused")
            };
            let shape = match shape {
                EntryShape::Typed { inputs, declared } => CompiledShape::Typed {
                    inputs: EntryInputs::of(interner, inputs, &module.main.params),
                    declared,
                },
                EntryShape::Untyped => CompiledShape::Untyped,
            };
            let entry = CompiledEntry {
                name: name.clone(),
                qref,
                shape,
                ret: module.ret.clone(),
                module,
                required,
            };
            (name, entry)
        })
        .collect();
    let solved: BTreeMap<String, Arc<Ty>> = graph
        .contexts
        .iter()
        .map(|c| {
            let Some(ty) = inf.context_types.get(&c.qref) else {
                panic!("inference settles a type for every context of the graph")
            };
            (program_key_of(interner, c.qref), Arc::new(ty.clone()))
        })
        .collect();
    let init_functions: Vec<(InitKey, QualifiedRef)> = declared_inits
        .functions()
        .map(|(key, function)| (key.clone(), function))
        .collect();
    let inits = match declared_inits.solved(interner, &solved) {
        Ok(inits) => inits,
        Err(refused) => {
            refusals.extend(refused.into_iter().map(|refusal| {
                Refusal::of(
                    Some(Origin::Init(refusal.key().stored())),
                    refusal.to_string(),
                )
            }));
            return Err(refusals);
        }
    };
    let shared = InterpreterContext::new(interner, executables, executor)
        .with_fn_types(fn_types)
        .with_context_names(context_names)
        .with_space(space)
        .with_inits(inits);
    if access == GraphAccess::Sync {
        refusals.extend(
            init_functions
                .iter()
                .filter(|(_, function)| lookup_module(&shared, function).main.may_suspend)
                .map(|(key, _)| {
                    let message = format!(
                        "the init of `@{}` can wait, and a program compiled for synchronous access \
                         runs an init inside the fetch that finds its key absent, which cannot wait; \
                         compile with `Host::async_access`",
                        key.written
                    );
                    Refusal::of(Some(Origin::Init(key.stored())), message)
                }),
        );
        if !refusals.is_empty() {
            return Err(refusals);
        }
    }
    let rt = shared.runtime_over_an_empty_page();
    Ok(Compiled {
        solved,
        rt,
        shared,
        entries: compiled_entries,
        #[cfg(feature = "tooling")]
        listing_laws: laws,
        times: CompileTimes {
            opt,
            parse,
            typeck,
            lower,
            optimize,
            prepare,
        },
    })
}

/// The extern functions a script's bare `name` reaches (RFC-0021,
/// RFC-0043): every one of that name, in any namespace or at the root, that
/// the registries declared a function rather than a machine coercion; each
/// written with its namespace, in name order.
fn bare_callable(
    interner: &Interner,
    extern_fns: &[Function],
    types: &acvus_mir::ty::TypeRegistry,
    name: QualifiedRef,
) -> Vec<String> {
    let mut reached: Vec<String> = extern_fns
        .iter()
        .filter(|f| f.qref.name == name.name && types.machine_view(f.qref).is_none())
        .map(|f| match f.qref.namespace {
            Some(ns) => format!("{}::{}", interner.resolve(ns), interner.resolve(f.qref.name)),
            None => interner.resolve(f.qref.name).to_owned(),
        })
        .collect();
    reached.sort();
    reached
}

// -- Program -------------------------------------------------------------

/// What one compilation made, whatever access it was compiled for.
pub(crate) struct Compiled {
    shared: InterpreterContext,
    rt: AcvusRuntime,
    entries: HashMap<String, CompiledEntry>,
    solved: BTreeMap<String, Arc<Ty>>,
    #[cfg(feature = "tooling")]
    listing_laws: acvus_mir::laws::LawTable,
    #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
    times: CompileTimes,
}

impl Compiled {
    fn interner(&self) -> &Interner {
        &self.shared.interner
    }

    pub(crate) fn running(mut self, runs: &FxHashSet<String>) -> Self {
        self.entries.retain(|name, _| runs.contains(name));
        self
    }

    fn codec(&self) -> Codec<'_> {
        Codec::of(&self.rt)
    }

    fn solved_type(&self, key: &str) -> Result<&Arc<Ty>, HostError> {
        self.solved.get(key).ok_or_else(|| HostError::NotInGraph {
            what: Named::Context(key.to_owned()),
        })
    }

    fn mismatched(&self, what: Part, held: &Ty, asked: &PolyTy) -> HostError {
        HostError::Mismatched {
            what,
            held: held.display(self.interner()).to_string(),
            asked: asked.display(self.interner()).to_string(),
        }
    }

    fn compiled(&self, name: &str) -> Result<&CompiledEntry, HostError> {
        self.entries.get(name).ok_or_else(|| HostError::NotInGraph {
            what: Named::Entry(name.to_owned()),
        })
    }

    async fn run_over(&self, entry: QualifiedRef, port: Arc<Port>, args: Vec<Value>) -> Result<Value, HostError> {
        Interpreter::on_port(self.shared.clone(), entry, port, args)
            .ended_or_ran()
            .await
    }
}

pub struct Program<A = SyncAccess> {
    compiled: Compiled,
    access: PhantomData<fn() -> A>,
}

impl<A> Program<A>
where
    A: Access,
{
    pub(crate) fn of(compiled: Compiled) -> Self {
        Program {
            compiled,
            access: PhantomData,
        }
    }

    /// Every context of the compilation, in key order.
    pub fn contexts(&self) -> impl Iterator<Item = &str> {
        self.compiled.solved.keys().map(String::as_str)
    }

    /// The type the compilation solved for `key`, as the language writes it.
    pub fn context_type(&self, key: &str) -> Option<String> {
        let ty = self.compiled.solved.get(key)?;
        Some(ty.display(self.compiled.interner()).to_string())
    }

    /// Run `f` with this program's pages and entries, which are good for
    /// that call alone.
    pub async fn scope<F, T>(&self, f: F) -> T
    where
        F: for<'p> AsyncFnOnce(Scope<'p, A>) -> T,
    {
        f(Scope {
            program: &self.compiled,
            access: PhantomData,
            brand: PhantomData,
        })
        .await
    }
}

/// A listing of one entry for the runtime's tooling.
#[cfg(feature = "tooling")]
pub struct Listing<'p> {
    /// The `$` inputs the entry still requires, in name order; a binding's
    /// input is not among them (RFC-0071 rule 5).
    pub inputs: Vec<InputListing>,
    pub mir: String,
    pub prepared: &'p crate::code::Prepared,
}

#[cfg(feature = "tooling")]
pub struct InputListing {
    pub name: String,
    pub ty: String,
}

macro_rules! program_tooling {
    ($v:vis) => {
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl<A> Program<A>
        where
            A: Access,
        {
            $v fn interner(&self) -> &Interner {
                self.compiled.interner()
            }

            $v fn times(&self) -> &CompileTimes {
                &self.compiled.times
            }
        }
    };
}
tooling_vis!(program_tooling);

#[cfg(feature = "tooling")]
impl<A> Program<A>
where
    A: Access,
{
    pub fn listing(&self, name: &str) -> Result<Listing<'_>, HostError> {
        let program = &self.compiled;
        let compiled = program.compiled(name)?;
        let interner = program.interner();
        let mut required: Vec<&ContextInfo> = compiled.required.iter().collect();
        required.sort_by_key(|input| input.name.name.bits());
        let inputs = required
            .into_iter()
            .map(|input| InputListing {
                name: interner.resolve(input.name.name).to_owned(),
                ty: input.ty.display(interner).to_string(),
            })
            .collect();
        Ok(Listing {
            inputs,
            mir: acvus_mir::printer::dump_with_costs(
                interner,
                &compiled.module,
                &program.listing_laws,
                &crate::cost::INTERPRETER_COSTS,
            ),
            prepared: lookup_module(&program.shared, &compiled.qref),
        })
    }
}

// -- Scope ---------------------------------------------------------------

/// One `Program::scope` call's view of its program. The lifetime brands
/// every page and entry the scope makes.
pub struct Scope<'p, A = SyncAccess> {
    program: &'p Compiled,
    access: PhantomData<fn() -> A>,
    brand: Brand<'p>,
}

impl<A> Clone for Scope<'_, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A> Copy for Scope<'_, A> {}

impl<'p> Scope<'p, SyncAccess> {
    /// A page over `storage`, which it borrows. Opening reads nothing
    /// (RFC-0090 rule 3).
    pub fn open<'s, S>(self, storage: &'s mut S) -> Page<'p, 's, S, SyncAccess>
    where
        S: Storage,
    {
        Page::over(self.program, storage)
    }
}

impl<'p> Scope<'p, AsyncAccess> {
    /// A page over `storage`, which it borrows. Opening reads nothing
    /// (RFC-0090 rule 3).
    pub fn open<'s, S>(self, storage: &'s mut S) -> Page<'p, 's, S, AsyncAccess>
    where
        S: AsyncStorage,
    {
        Page::over(self.program, storage)
    }
}

impl<'p, A> Scope<'p, A>
where
    A: Access,
{
    pub fn entry<I, R>(self, name: &str) -> Result<Entry<'p, I, R, A>, HostError>
    where
        I: Declared + Cross<AcvusRuntime>,
        R: Declared,
    {
        let (compiled, inputs) = self.typed::<R>(name)?;
        let interner = self.program.interner();
        let asked = I::declared(interner);
        let refused = || HostError::Mismatched {
            what: Part::Inputs(name.to_owned()),
            held: inputs.shape.display(interner),
            asked: asked.display(interner).to_string(),
        };
        let Some(shape) = ResolvedShape::of_declared(interner, &asked) else {
            return Err(refused());
        };
        if !shape.same_fields(&inputs.shape) {
            return Err(refused());
        }
        let form = (<I::ReturnForm as Form>::KIND, <I::ReturnForm as Form>::WIDTH);
        let crosses_by_field = match form {
            (FormKind::Components, width) => width == shape.len(),
            _ => shape.len() == 0,
        };
        if !crosses_by_field {
            return Err(refused());
        }
        Ok(Entry::of(self.program, compiled, inputs))
    }

    pub fn entry_shaped<R>(self, name: &str) -> Result<Entry<'p, Inputs, R, A>, HostError>
    where
        R: Declared,
    {
        let (compiled, inputs) = self.typed::<R>(name)?;
        Ok(Entry::of(self.program, compiled, inputs))
    }

    fn typed<R>(self, name: &str) -> Result<(&'p CompiledEntry, &'p EntryInputs), HostError>
    where
        R: Declared,
    {
        let program = self.program;
        let compiled = program.compiled(name)?;
        let interner = program.interner();
        let asked = R::declared(interner);
        let CompiledShape::Typed { inputs, declared } = &compiled.shape else {
            return Err(program.mismatched(Part::Result(name.to_owned()), &Ty::Never, &asked));
        };
        if !asked.same_erased(declared) {
            return Err(HostError::Mismatched {
                what: Part::Result(name.to_owned()),
                held: declared.display(interner).to_string(),
                asked: asked.display(interner).to_string(),
            });
        }
        Ok((compiled, inputs))
    }
}

macro_rules! scope_tooling {
    ($v:vis) => {
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl<'p, A> Scope<'p, A>
        where
            A: Access,
        {
            /// An entry the tooling runs whatever it declares, whose result
            /// it reads by the settled type; one that still requires a `$`
            /// input is refused.
            $v fn untyped_entry(self, name: &str) -> Result<UntypedEntry<'p, A>, HostError> {
                let program = self.program;
                let compiled = program.compiled(name)?;
                if !compiled.required.is_empty() {
                    let interner = program.interner();
                    let held: Vec<String> = compiled
                        .required
                        .iter()
                        .map(|input| format!("${}", interner.resolve(input.name.name)))
                        .collect();
                    return Err(HostError::Mismatched {
                        what: Part::Inputs(name.to_owned()),
                        held: held.join(", "),
                        asked: "()".to_owned(),
                    });
                }
                Ok(UntypedEntry {
                    compiled,
                    access: PhantomData,
                    brand: PhantomData,
                })
            }
        }
    };
}
tooling_vis!(scope_tooling);

// -- Page ----------------------------------------------------------------

/// A program's view of one storage (RFC-0090 rule 3), made by
/// `Scope::open`. It holds no value: each method below and each run reads
/// and writes the storage where it loads and stores.
pub struct Page<'p, 's, S, A = SyncAccess> {
    program: &'p Compiled,
    storage: &'s mut S,
    #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
    filled: Vec<String>,
    access: PhantomData<fn() -> A>,
    brand: Brand<'p>,
}

impl<'p, 's, S, A> Page<'p, 's, S, A>
where
    S: AsyncStorage,
    A: Access,
{
    fn over(program: &'p Compiled, storage: &'s mut S) -> Self {
        Page {
            program,
            storage,
            filled: Vec::new(),
            access: PhantomData,
            brand: PhantomData,
        }
    }

    async fn lent<Q, F, O>(&mut self, key: &str, held: Held, f: F) -> Result<O, HostError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
        F::Marker: Lendable<AcvusRuntime, Loan = Shared>,
    {
        let program = self.program;
        self.lent_back(key, held, Back::Restore, |held| {
            held.lend(&program.rt, program.interner(), f)
        })
        .await
    }

    async fn lent_mut<Q, F, O>(&mut self, key: &str, held: Held, f: F) -> Result<O, HostError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
    {
        let program = self.program;
        self.lent_back(key, held, Back::Store, |held| {
            held.lend_mut(&program.rt, program.interner(), f)
        })
        .await
    }

    /// Lend `held` through `lend` and hand it back on every path (RFC-0090
    /// rule 5): by `lent` where the closure returned, by `restore` where the
    /// types did not match or the closure panicked, and then the panic
    /// resumes.
    async fn lent_back<O>(
        &mut self,
        key: &str,
        mut held: Held,
        lent: Back,
        lend: impl FnOnce(&mut Held) -> Result<O, PolyTy>,
    ) -> Result<O, HostError> {
        let program = self.program;
        // `AssertUnwindSafe`: the panic is resumed below, so the caller
        // observes it as if it had never been caught; the holder is the one
        // thing read after it, and it is handed back as the closure left it.
        let ended = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lend(&mut held)))
            .map(|ended| ended.map_err(|asked| program.mismatched(Part::Context(key.to_owned()), held.ty(), &asked)));
        let back = match ended {
            Ok(Ok(_)) => lent,
            Ok(Err(_)) | Err(_) => Back::Restore,
        };
        let handed = match back {
            Back::Store => AsyncStorage::store(&mut *self.storage, key, held).await,
            Back::Restore => AsyncStorage::restore(&mut *self.storage, key, held).await,
        };
        match ended {
            Ok(ended) => {
                handed?;
                ended
            }
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    pub async fn insert<T>(&mut self, key: &str, value: T) -> Result<(), HostError>
    where
        T: Declared + OneValue<AcvusRuntime>,
    {
        let program = self.program;
        let solved = program.solved_type(key)?;
        let asked = T::declared(program.interner());
        if !lift_declaration(solved, &mut PolyBuilder::new()).same_erased(&asked) {
            return Err(program.mismatched(Part::Context(key.to_owned()), solved, &asked));
        }
        // SAFETY: `value` crosses at `T`, the type the page holds `key` at.
        let value = Owned::erased(unsafe { Crossing::new(&program.rt) }, value);
        let held = Held::new(value, Arc::clone(solved), program.shared.compilation);
        AsyncStorage::store(&mut *self.storage, key, held).await?;
        Ok(())
    }

    /// Make what the storage holds durable (RFC-0090 rule 3).
    pub async fn commit(&mut self) -> Result<(), HostError> {
        let codec = self.program.codec();
        AsyncStorage::commit(&mut *self.storage, &codec).await?;
        Ok(())
    }

    pub fn storage(&self) -> &S {
        self.storage
    }
}

/// How a holder goes back to its storage after its closure returned.
enum Back {
    Store,
    Restore,
}

/// Closes a run's gate when the run returns or is dropped, before the
/// page's borrow of the storage ends.
struct Closing(Arc<Port>);

impl Drop for Closing {
    fn drop(&mut self) {
        self.0.close();
    }
}

impl<'p, 's, S> Page<'p, 's, S, SyncAccess>
where
    S: Storage,
{
    /// Lend `key`'s value to `f`, whose parameter crosses as a handler's
    /// shared one does (RFC-0090 rule 3), and hand the holder back.
    pub async fn with<Q, F, O>(&mut self, key: &str, f: F) -> Result<O, HostError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
        F::Marker: Lendable<AcvusRuntime, Loan = Shared>,
    {
        let held = self.loaded(key)?;
        self.lent(key, held, f).await
    }

    /// Lend `key`'s value to `f` exclusively, whose parameter crosses as a
    /// handler's does, shared or exclusive, and store what it left.
    pub async fn with_mut<Q, F, O>(&mut self, key: &str, f: F) -> Result<O, HostError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
    {
        let held = self.loaded(key)?;
        self.lent_mut(key, held, f).await
    }

    fn loaded(&mut self, key: &str) -> Result<Held, HostError> {
        let program = self.program;
        let settled = program.solved_type(key)?;
        let storage: &mut dyn Storage = &mut *self.storage;
        // SAFETY: `closing` closes the gate before this function returns,
        // while `self` still borrows the storage exclusively.
        let port = Port::gate(unsafe { Gate::open(storage) });
        let closing = Closing(Arc::clone(&port));
        let rt = program.shared.runtime_over(Arc::clone(&port));
        let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fetch_now(&rt, key, settled)))
            .map_err(ended);
        drop(closing);
        self.filled.extend(port.take_filled());
        loaded
    }

    async fn run(&mut self, entry: QualifiedRef, args: Vec<Value>) -> Result<Value, HostError> {
        let storage: &mut dyn Storage = &mut *self.storage;
        // SAFETY: `closing` closes the gate when this future returns or is
        // dropped, while `self` still borrows the storage exclusively.
        let port = Port::gate(unsafe { Gate::open(storage) });
        let closing = Closing(Arc::clone(&port));
        let ran = self.program.run_over(entry, Arc::clone(&port), args).await;
        drop(closing);
        self.filled.extend(port.take_filled());
        ran
    }
}

impl<'p, 's, S> Page<'p, 's, S, AsyncAccess>
where
    S: AsyncStorage,
{
    /// Lend `key`'s value to `f`, whose parameter crosses as a handler's
    /// shared one does (RFC-0090 rule 3), and hand the holder back.
    pub async fn with<Q, F, O>(&mut self, key: &str, f: F) -> Result<O, HostError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
        F::Marker: Lendable<AcvusRuntime, Loan = Shared>,
    {
        let held = self.loaded(key).await?;
        self.lent(key, held, f).await
    }

    /// Lend `key`'s value to `f` exclusively, whose parameter crosses as a
    /// handler's does, shared or exclusive, and store what it left.
    pub async fn with_mut<Q, F, O>(&mut self, key: &str, f: F) -> Result<O, HostError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
    {
        let held = self.loaded(key).await?;
        self.lent_mut(key, held, f).await
    }

    async fn loaded(&mut self, key: &str) -> Result<Held, HostError> {
        let program = self.program;
        let settled = program.solved_type(key)?;
        let (port, requests) = Port::queue();
        let rt = program.shared.runtime_over(Arc::clone(&port));
        let codec = program.codec();
        let fetched = std::panic::AssertUnwindSafe(fetch_waited(&rt, key, settled)).catch_unwind();
        let loaded = serve(&mut *self.storage, &codec, requests, fetched).await.map_err(ended);
        self.filled.extend(port.take_filled());
        loaded
    }

    async fn run(&mut self, entry: QualifiedRef, args: Vec<Value>) -> Result<Value, HostError> {
        let program = self.program;
        let (port, requests) = Port::queue();
        let codec = program.codec();
        let run = program.run_over(entry, Arc::clone(&port), args);
        let ran = serve(&mut *self.storage, &codec, requests, run).await;
        self.filled.extend(port.take_filled());
        ran
    }
}

macro_rules! page_tooling {
    ($v:vis) => {
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl<S, A> Page<'_, '_, S, A> {
            /// The keys whose init a run on this page ran, in the order it
            /// ran them.
            $v fn filled(&self) -> &[String] {
                &self.filled
            }
        }

        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl<S> Page<'_, '_, S, SyncAccess>
        where
            S: Storage,
        {
            /// Load every key that has an init, which runs where the storage
            /// lacks the key, and hand each back; the keys filled, in key
            /// order.
            $v async fn fill(&mut self) -> Result<Vec<String>, HostError> {
                let program = self.program;
                let mut keys: Vec<&str> = program.shared.inits.keys().map(|key| &**key).collect();
                keys.sort_unstable();
                let before = self.filled.len();
                for key in keys {
                    let held = self.loaded(key)?;
                    Storage::restore(&mut *self.storage, key, held)?;
                }
                Ok(self.filled[before..].to_vec())
            }
        }
    };
}
tooling_vis!(page_tooling);

// -- Entry and Output ----------------------------------------------------

pub struct Entry<'p, I, R, A = SyncAccess> {
    program: &'p Compiled,
    compiled: &'p CompiledEntry,
    inputs: &'p EntryInputs,
    declared: PhantomData<fn(I) -> R>,
    access: PhantomData<fn() -> A>,
    brand: Brand<'p>,
}

impl<'p, I, R, A> Entry<'p, I, R, A> {
    fn of(program: &'p Compiled, compiled: &'p CompiledEntry, inputs: &'p EntryInputs) -> Self {
        Entry {
            program,
            compiled,
            inputs,
            declared: PhantomData,
            access: PhantomData,
            brand: PhantomData,
        }
    }

    fn arguments(&self, inputs: I) -> Result<Vec<Value>, HostError>
    where
        I: RunInputs,
    {
        let program = self.program;
        inputs.arguments(sealed_inputs::ArgumentsOf {
            rt: &program.rt,
            interner: program.interner(),
            entry: &self.compiled.name,
            inputs: self.inputs,
        })
    }

    fn output(&self, value: Value) -> Output<'p, R> {
        Output {
            // SAFETY: the run moved its result out to this caller, and no
            // other holder owns it.
            value: unsafe { Owned::from_value(Holding::new(), value) },
            program: self.program,
            compiled: self.compiled,
            result: PhantomData,
            brand: PhantomData,
        }
    }
}

impl<'p, I, R> Entry<'p, I, R, SyncAccess>
where
    I: RunInputs,
{
    pub async fn run<S>(&self, page: &mut Page<'p, '_, S, SyncAccess>, inputs: I) -> Result<Output<'p, R>, HostError>
    where
        S: Storage,
    {
        let arguments = self.arguments(inputs)?;
        let value = page.run(self.compiled.qref, arguments).await?;
        Ok(self.output(value))
    }
}

impl<'p, I, R> Entry<'p, I, R, AsyncAccess>
where
    I: RunInputs,
{
    pub async fn run<S>(&self, page: &mut Page<'p, '_, S, AsyncAccess>, inputs: I) -> Result<Output<'p, R>, HostError>
    where
        S: AsyncStorage,
    {
        let arguments = self.arguments(inputs)?;
        let value = page.run(self.compiled.qref, arguments).await?;
        Ok(self.output(value))
    }
}

/// What a run takes as an entry's inputs: a derived struct or `()`, whose
/// shape `Scope::entry` compared with the entry's, or `Inputs`, which each
/// run checks field by field before any value crosses.
pub trait RunInputs: sealed_inputs::Given {}

impl<I> RunInputs for I where I: sealed_inputs::Given {}

mod sealed_inputs {
    use super::{AcvusRuntime, EntryInputs, HostError, Interner, Value};

    pub struct ArgumentsOf<'a> {
        pub(super) rt: &'a AcvusRuntime,
        pub(super) interner: &'a Interner,
        pub(super) entry: &'a str,
        pub(super) inputs: &'a EntryInputs,
    }

    pub trait Given {
        #[doc(hidden)]
        fn arguments(self, of: ArgumentsOf<'_>) -> Result<Vec<Value>, HostError>;
    }
}

impl<I> sealed_inputs::Given for I
where
    I: Cross<AcvusRuntime>,
    I::ReturnForm: Returned<Verdict = ()>,
{
    fn arguments(self, of: sealed_inputs::ArgumentsOf<'_>) -> Result<Vec<Value>, HostError> {
        // SAFETY: `Scope::entry` made this entry only after finding `I`'s
        // fields the entry's, each at its type, and `I` crossing as one
        // component per field in that same order.
        let crossing = unsafe { Crossing::new(of.rt) };
        let holding = crossing.holding();
        let width = of.inputs.shape.len();
        let mut fields: Vec<Owned<AcvusRuntime>> = std::iter::repeat_with(|| Owned::vacant(holding))
            .take(width)
            .collect();
        if width > 0 {
            // SAFETY: every slot of `fields` is vacant.
            let run = unsafe { lend_run(holding, &mut fields) };
            <I as Gives<Val<I, Uniform>, AcvusRuntime>>::give(self, crossing, run);
        }
        Ok(of.inputs.arguments(crossing, fields.into_iter().map(CrossedField).collect()))
    }
}

impl sealed_inputs::Given for Inputs {
    fn arguments(self, of: sealed_inputs::ArgumentsOf<'_>) -> Result<Vec<Value>, HostError> {
        let given = of.inputs.checked(of.interner, of.entry, self)?;
        // SAFETY: `checked` found each value's `T` declared at the type of
        // the field it fills.
        let crossing = unsafe { Crossing::new(of.rt) };
        Ok(of.inputs.arguments(crossing, given))
    }
}

/// An entry's result, owned until dropped, at the type the checker settled
/// for `R` (RFC-0090 rule 3).
pub struct Output<'p, R> {
    value: Owned<AcvusRuntime>,
    program: &'p Compiled,
    compiled: &'p CompiledEntry,
    result: PhantomData<fn() -> R>,
    brand: Brand<'p>,
}

impl<R> Output<'_, R> {
    /// Lend the result to `f`, whose parameter crosses as a handler's shared
    /// one does.
    pub fn with<Q, F, O>(&self, f: F) -> Result<O, HostError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
        F::Marker: Lendable<AcvusRuntime, Loan = Shared>,
    {
        let program = self.program;
        // SAFETY: the run crossed the result at `ret` and `self` owns it for
        // the borrow; a shared parameter writes nothing.
        unsafe { acvus_extern::lend(&program.rt, program.interner(), &self.value, &self.compiled.ret, f) }
            .map_err(|asked| self.mismatched(&asked))
    }

    /// Lend the result to `f` exclusively, whose parameter crosses as a
    /// handler's does, shared or exclusive.
    pub fn with_mut<Q, F, O>(&mut self, f: F) -> Result<O, HostError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
    {
        let program = self.program;
        // SAFETY: `&mut self` names the result exclusively, and a lent
        // parameter writes inside the storage the word names or the word in
        // place, never another holder's.
        let word = unsafe { self.value.value_mut(Holding::new()) };
        // SAFETY: the run crossed the result at `ret`, and `word` is the only
        // live name of it for the call.
        unsafe { acvus_extern::lend(&program.rt, program.interner(), word, &self.compiled.ret, f) }
            .map_err(|asked| self.mismatched(&asked))
    }

    fn mismatched(&self, asked: &PolyTy) -> HostError {
        let what = Part::Result(self.compiled.name.clone());
        self.program.mismatched(what, &self.compiled.ret, asked)
    }
}

/// An entry the runtime's tooling runs and reads by the settled type.
pub struct UntypedEntry<'p, A = SyncAccess> {
    compiled: &'p CompiledEntry,
    access: PhantomData<fn() -> A>,
    brand: Brand<'p>,
}

pub struct UntypedOutput<'p> {
    value: Owned<AcvusRuntime>,
    compiled: &'p CompiledEntry,
    brand: Brand<'p>,
}

macro_rules! untyped_run {
    ($v:vis) => {
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl<'p> UntypedEntry<'p, SyncAccess> {
            $v async fn run<S>(&self, page: &mut Page<'p, '_, S, SyncAccess>) -> Result<UntypedOutput<'p>, HostError>
            where
                S: Storage,
            {
                let value = page.run(self.compiled.qref, Vec::new()).await?;
                Ok(UntypedOutput {
                    // SAFETY: the run moved its result out to this caller,
                    // and no other holder owns it.
                    value: unsafe { Owned::from_value(Holding::new(), value) },
                    compiled: self.compiled,
                    brand: PhantomData,
                })
            }
        }

        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl UntypedOutput<'_> {
            $v fn with_value<O, F>(&self, f: F) -> O
            where
                F: FnOnce(&Value, &Ty) -> O,
            {
                f(&self.value, &self.compiled.ret)
            }
        }
    };
}
tooling_vis!(untyped_run);
