use std::collections::HashMap;
use std::sync::Arc;

use acvus_interpreter::{
    ContextOverlay, Executable, Interpreter, InterpreterContext, SequentialExecutor, Value,
};
use acvus_mir::graph::{FunctionId, QualifiedRef};
use acvus_mir::ir::*;
use acvus_mir::ty::Ty;
use acvus_utils::{Interner, LocalFactory};
use rustc_hash::FxHashMap;

// ── Helpers ─────────────────────────────────────────────────────────

/// Allocate N sequential ValueIds from a factory.
fn alloc_n(factory: &mut LocalFactory<ValueId>, n: usize) -> Vec<ValueId> {
    (0..n).map(|_| factory.next()).collect()
}

fn inst(kind: InstKind) -> Inst {
    Inst {
        span: acvus_ast::Span::ZERO,
        kind,
    }
}

fn empty_overlay() -> ContextOverlay {
    ContextOverlay::new(Arc::new(HashMap::<String, Value>::new()), Interner::new())
}

fn make_context(
    interner: &Interner,
    functions: FxHashMap<FunctionId, Executable>,
) -> InterpreterContext {
    let executor = Arc::new(SequentialExecutor);
    InterpreterContext::new(interner, functions, executor)
}

// ── Tests ───────────────────────────────────────────────────────────

/// Spawn a callee that returns arg + 1, eval it.
/// Entry:  Spawn(callee, [arg]) → Eval(handle) → Return(result)
/// Callee: param + 1 → Return
#[tokio::test]
async fn spawn_eval_basic() {
    let interner = Interner::new();

    let entry_id = FunctionId::alloc();
    let callee_id = FunctionId::alloc();

    // ── Callee module: receives one param, returns param + 1 ──
    let callee_module = {
        let mut f = LocalFactory::<ValueId>::new();
        let vids = alloc_n(&mut f, 3); // v0=param, v1=const(1), v2=result
        let insts = vec![
            inst(InstKind::Const {
                dst: vids[1],
                value: acvus_ast::Literal::Int(1),
            }),
            inst(InstKind::BinOp {
                dst: vids[2],
                op: acvus_ast::BinOp::Add,
                left: vids[0],
                right: vids[1],
            }),
            inst(InstKind::Return(vids[2])),
        ];
        MirModule {
            main: MirBody {
                insts,
                val_types: FxHashMap::default(),
                param_regs: vec![vids[0]],
                capture_regs: vec![],
                debug: DebugInfo::new(),
                val_factory: f,
                label_count: 0,
            },
            closures: FxHashMap::default(),
        }
    };

    // ── Entry module: spawn callee with arg=41, eval, return ──
    let entry_module = {
        let mut f = LocalFactory::<ValueId>::new();
        let vids = alloc_n(&mut f, 3); // v0=const(41), v1=handle, v2=result
        let insts = vec![
            inst(InstKind::Const {
                dst: vids[0],
                value: acvus_ast::Literal::Int(41),
            }),
            inst(InstKind::Spawn {
                dst: vids[1],
                callee: Callee::Direct(callee_id),
                args: vec![vids[0]],
                context_uses: vec![],
            }),
            inst(InstKind::Eval {
                dst: vids[2],
                src: vids[1],
                context_defs: vec![],
            }),
            inst(InstKind::Return(vids[2])),
        ];
        MirModule {
            main: MirBody {
                insts,
                val_types: FxHashMap::default(),
                param_regs: vec![],
                capture_regs: vec![],
                debug: DebugInfo::new(),
                val_factory: f,
                label_count: 0,
            },
            closures: FxHashMap::default(),
        }
    };

    let mut functions = FxHashMap::default();
    functions.insert(entry_id, Executable::Module(entry_module));
    functions.insert(callee_id, Executable::Module(callee_module));

    let shared = make_context(&interner, functions);
    let overlay = empty_overlay();
    let mut interp = Interpreter::new(shared, entry_id, overlay);
    let result = interp.execute().await.expect("execution failed");

    assert_eq!(result.value, Value::Int(42));
}

/// Spawn a callee that writes to a context, eval with context_defs,
/// then read the context to verify it was merged.
#[tokio::test]
async fn spawn_eval_context_defs() {
    let interner = Interner::new();

    let entry_id = FunctionId::alloc();
    let callee_id = FunctionId::alloc();
    let ctx_name = interner.intern("counter");
    let ctx_id = QualifiedRef::root(ctx_name);

    // ── Callee: writes 99 to context "counter", returns 0 ──
    let callee_module = {
        let mut f = LocalFactory::<ValueId>::new();
        let vids = alloc_n(&mut f, 3);
        // v0=project(ctx), v1=const(99), v2=const(0) for return
        let insts = vec![
            inst(InstKind::ContextProject {
                dst: vids[0],
                ctx: ctx_id,
            }),
            inst(InstKind::Const {
                dst: vids[1],
                value: acvus_ast::Literal::Int(99),
            }),
            inst(InstKind::ContextStore {
                dst: vids[0],
                value: vids[1],
            }),
            inst(InstKind::Const {
                dst: vids[2],
                value: acvus_ast::Literal::Int(0),
            }),
            inst(InstKind::Return(vids[2])),
        ];
        MirModule {
            main: MirBody {
                insts,
                val_types: FxHashMap::default(),
                param_regs: vec![],
                capture_regs: vec![],
                debug: DebugInfo::new(),
                val_factory: f,
                label_count: 0,
            },
            closures: FxHashMap::default(),
        }
    };

    // ── Entry: spawn callee → eval with context_defs → read context → return ──
    let entry_module = {
        let mut f = LocalFactory::<ValueId>::new();
        let vids = alloc_n(&mut f, 5);
        // v0=handle, v1=eval_result, v2=context_def(post-eval SSA for ctx),
        // v3=context_load, v4=unused
        let insts = vec![
            inst(InstKind::Spawn {
                dst: vids[0],
                callee: Callee::Direct(callee_id),
                args: vec![],
                context_uses: vec![],
            }),
            inst(InstKind::Eval {
                dst: vids[1],
                src: vids[0],
                context_defs: vec![(ctx_id, vids[2])],
            }),
            // Now vids[2] is the new SSA name for ctx_id.
            // ContextLoad reads from overlay using projection_map[vids[2]] → ctx_id.
            inst(InstKind::ContextLoad {
                dst: vids[3],
                src: vids[2],
            }),
            inst(InstKind::Return(vids[3])),
        ];
        MirModule {
            main: MirBody {
                insts,
                val_types: FxHashMap::default(),
                param_regs: vec![],
                capture_regs: vec![],
                debug: DebugInfo::new(),
                val_factory: f,
                label_count: 0,
            },
            closures: FxHashMap::default(),
        }
    };

    let mut functions = FxHashMap::default();
    functions.insert(entry_id, Executable::Module(entry_module));
    functions.insert(callee_id, Executable::Module(callee_module));

    let mut context_names = FxHashMap::default();
    context_names.insert(ctx_id, ctx_name);

    let shared = make_context(&interner, functions).with_context_names(context_names);
    let overlay = empty_overlay();
    let mut interp = Interpreter::new(shared, entry_id, overlay);
    let result = interp.execute().await.expect("execution failed");

    // Callee wrote 99 to "counter", eval merged it, ContextLoad should read 99.
    assert_eq!(result.value, Value::Int(99));
}

/// Spawn with multiple args — callee receives two params and returns their sum.
#[tokio::test]
async fn spawn_eval_multi_args() {
    let interner = Interner::new();

    let entry_id = FunctionId::alloc();
    let callee_id = FunctionId::alloc();

    // ── Callee: param0 + param1 ──
    let callee_module = {
        let mut f = LocalFactory::<ValueId>::new();
        let vids = alloc_n(&mut f, 3); // v0=param0, v1=param1, v2=result
        let insts = vec![
            inst(InstKind::BinOp {
                dst: vids[2],
                op: acvus_ast::BinOp::Add,
                left: vids[0],
                right: vids[1],
            }),
            inst(InstKind::Return(vids[2])),
        ];
        MirModule {
            main: MirBody {
                insts,
                val_types: FxHashMap::default(),
                param_regs: vec![vids[0], vids[1]],
                capture_regs: vec![],
                debug: DebugInfo::new(),
                val_factory: f,
                label_count: 0,
            },
            closures: FxHashMap::default(),
        }
    };

    // ── Entry: spawn(callee, [10, 32]) → eval → return ──
    let entry_module = {
        let mut f = LocalFactory::<ValueId>::new();
        let vids = alloc_n(&mut f, 4); // v0=10, v1=32, v2=handle, v3=result
        let insts = vec![
            inst(InstKind::Const {
                dst: vids[0],
                value: acvus_ast::Literal::Int(10),
            }),
            inst(InstKind::Const {
                dst: vids[1],
                value: acvus_ast::Literal::Int(32),
            }),
            inst(InstKind::Spawn {
                dst: vids[2],
                callee: Callee::Direct(callee_id),
                args: vec![vids[0], vids[1]],
                context_uses: vec![],
            }),
            inst(InstKind::Eval {
                dst: vids[3],
                src: vids[2],
                context_defs: vec![],
            }),
            inst(InstKind::Return(vids[3])),
        ];
        MirModule {
            main: MirBody {
                insts,
                val_types: FxHashMap::default(),
                param_regs: vec![],
                capture_regs: vec![],
                debug: DebugInfo::new(),
                val_factory: f,
                label_count: 0,
            },
            closures: FxHashMap::default(),
        }
    };

    let mut functions = FxHashMap::default();
    functions.insert(entry_id, Executable::Module(entry_module));
    functions.insert(callee_id, Executable::Module(callee_module));

    let shared = make_context(&interner, functions);
    let overlay = empty_overlay();
    let mut interp = Interpreter::new(shared, entry_id, overlay);
    let result = interp.execute().await.expect("execution failed");

    assert_eq!(result.value, Value::Int(42));
}
