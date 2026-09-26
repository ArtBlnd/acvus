//! A straight body of many operations, run inside a `wasm32` module, which
//! marks the engine's call depth and the linear stack pointer where the body
//! starts and where it ends (RFC-0105 rule 3).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use acvus_extern::{Externs, Registry, extern_fn, extern_registry};
use acvus_interpreter::{
    AcvusRuntime, Executable, Interpreter, InterpreterContext, PrepareCtx, SequentialExecutor,
    prepare_module,
};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{
    Access, Bindings, CompilationGraph, FnKind, Function, Inputs, ParsedAst, QualifiedRef,
    extract, infer, lower, optimize,
};
use acvus_mir::ty::{Effect, Flows, PolyBuilder, Ty, TyTerm, lift_declaration};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Mark {
    pub value: i64,
    pub engine_frames: u32,
    pub linear_sp: u32,
}

static MARKS: Mutex<Vec<Mark>> = Mutex::new(Vec::new());
static PANIC: Mutex<String> = Mutex::new(String::new());

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "probe")]
unsafe extern "C" {
    /// The frames on the engine's stack at the call, counted by the embedder.
    fn engine_frames() -> u32;
}

#[cfg(target_arch = "wasm32")]
fn engine_depth() -> u32 {
    // SAFETY: the import takes nothing and returns a number.
    unsafe { engine_frames() }
}

#[cfg(target_arch = "wasm32")]
#[inline(never)]
fn linear_sp() -> u32 {
    let local = 0u8;
    let at = std::hint::black_box(std::ptr::addr_of!(local)).addr();
    u32::try_from(at).expect("a wasm32 address is 32 bits")
}

/// Natively this crate hands `wasm_probe` the body and the registry to
/// prepare, and runs neither: a mark reads an engine and a linear stack that
/// only the module has.
#[cfg(not(target_arch = "wasm32"))]
fn engine_depth() -> u32 {
    panic!("`probe::mark` ran natively; its engine depth is the wasm32 module's")
}

#[cfg(not(target_arch = "wasm32"))]
fn linear_sp() -> u32 {
    panic!("`probe::mark` ran natively; its linear stack pointer is the wasm32 module's")
}

#[extern_fn(effect = pure)]
fn mark(value: i64) -> i64 {
    let seen = Mark {
        value,
        engine_frames: engine_depth(),
        linear_sp: linear_sp(),
    };
    MARKS.lock().expect("one thread").push(seen);
    value
}

#[extern_fn(effect = pure)]
fn step(value: i64) -> i64 {
    value + 1
}

pub fn registry() -> Registry<AcvusRuntime> {
    extern_registry! { ns: "probe", fns: [mark, step] }
}

pub fn straight_body(steps: u32) -> String {
    let mut source = String::from("let a = mark(1);\nlet k = mark(7);\nlet p = mark(1000003);\n");
    for _ in 0..steps {
        source.push_str("a = step(a * k % p);\n");
    }
    source.push_str("mark(a)\n");
    source
}

/// A finished run, whose prepared body is dropped apart from the run so that
/// the embedder sees which of the two ran out of stack.
pub struct Finished {
    pub marks: Vec<Mark>,
    pub interpreter: Interpreter,
}

pub fn run_straight_body(steps: u32) -> Finished {
    MARKS.lock().expect("one thread").clear();
    let interner = Interner::new();
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(registry());
    let main = acvus_ast::parse_script(&interner, &straight_body(steps)).expect("the body parses");
    let mut pb = PolyBuilder::new();
    let entry = QualifiedRef::root(interner.intern("main"));
    let mut functions = vec![Function {
        qref: entry,
        kind: FnKind::Local(ParsedAst::Script(main), Inputs::FromReads),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(lift_declaration(&Ty::I64, &mut pb)),
            captures: vec![],
            effect: Effect::OPAQUE.into(),
            flows: Flows::Every.into(),
        },
    }];
    let Externs {
        functions: extern_fns,
        types,
        handlers,
        instances,
        ..
    } = Externs::combine(registries, &interner).expect("the registries combine");
    functions.extend(extern_fns);
    let mut executables: FxHashMap<QualifiedRef, Executable> = handlers
        .into_iter()
        .map(|(qref, handler)| (qref, Executable::Extern(handler)))
        .collect();
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(Vec::new()),
        types: Freeze::new(types),
        bindings: Bindings::default(),
        access: Access::Sync,
        entries: vec![entry],
    };
    let ext = extract::extract(&interner, &graph);
    let inf = infer::infer(&interner, &graph, &ext);
    let lowered = lower::lower(&interner, &graph, &ext.view(), &inf);
    assert!(lowered.errors.is_empty(), "the body lowers");
    let laws = acvus_mir::laws::LawTable::of(graph.functions.iter(), &graph.types);
    let optimized = optimize::optimize(&interner, &laws, lowered.modules.clone(), Opt::Full);
    assert!(optimized.errors.is_empty(), "the body optimizes");
    let context_names = FxHashMap::default();
    let ctx = PrepareCtx {
        interner: &interner,
        externs: &executables,
        context_names: &context_names,
        instances: &instances,
        access: Access::Sync,
        // The straight body has no loop to lower ahead, and `wasm_probe`
        // counts the operations of the same body prepared in place.
        lowering: acvus_interpreter::Lowering::InPlace,
    };
    let prepared: Vec<(QualifiedRef, Executable)> = optimized
        .modules
        .iter()
        .map(|(qref, module)| {
            let body = prepare_module(module, &ctx)
                .unwrap_or_else(|refused| panic!("the body is refused: {refused}"));
            (*qref, Executable::Module(Arc::new(body)))
        })
        .collect();
    executables.extend(prepared);
    let shared = InterpreterContext::new(&interner, executables, Arc::new(SequentialExecutor));
    let mut interpreter = Interpreter::new(shared, entry, HashMap::new());
    if let Err(error) = futures::executor::block_on(interpreter.execute()) {
        panic!("the run failed: {error}");
    }
    Finished {
        marks: MARKS.lock().expect("one thread").clone(),
        interpreter,
    }
}

thread_local! {
    static LAST: RefCell<Option<Finished>> = const { RefCell::new(None) };
}

/// A panic on `wasm32` is a trap that carries no message, so the embedder
/// reads it at `panic_message_ptr` after the trap.
#[unsafe(no_mangle)]
pub extern "C" fn run(steps: u32) -> u32 {
    std::panic::set_hook(Box::new(|info| {
        *PANIC.lock().expect("one thread") = info.to_string();
    }));
    let finished = run_straight_body(steps);
    let count = u32::try_from(finished.marks.len()).expect("a run makes four marks");
    LAST.set(Some(finished));
    count
}

#[unsafe(no_mangle)]
pub extern "C" fn marks() -> *const Mark {
    LAST.with_borrow(|last| {
        last.as_ref()
            .expect("`run` finished before `marks`")
            .marks
            .as_ptr()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn release() {
    LAST.set(None);
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_stack_pointer() -> u32 {
    linear_sp()
}

#[unsafe(no_mangle)]
pub extern "C" fn panic_message_ptr() -> *const u8 {
    PANIC.lock().expect("one thread").as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn panic_message_len() -> u32 {
    u32::try_from(PANIC.lock().expect("one thread").len()).expect("a message under 4 GiB")
}
