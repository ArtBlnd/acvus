use std::collections::HashMap;
use std::sync::Arc;

use acvus_extern::Owned;
use acvus_interpreter::{AcvusRuntime, Executable, Interpreter, InterpreterContext, SequentialExecutor};
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ir::*;
use acvus_mir::ty::Task;
use acvus_mir::ty::Ty;
use acvus_utils::{Interner, LocalFactory};
use rustc_hash::FxHashMap;

// -- Helpers ---------------------------------------------------------

/// Allocate N sequential ValueIds from a factory.
fn alloc_n(factory: &mut LocalFactory<ValueId>, n: usize) -> Vec<ValueId> {
    (0..n).map(|_| factory.next()).collect()
}

fn types(entries: Vec<(ValueId, Ty)>) -> FxHashMap<ValueId, Ty> {
    entries.into_iter().collect()
}

fn inst(kind: InstKind) -> Inst {
    Inst {
        span: acvus_ast::Span::ZERO,
        kind,
    }
}

fn empty_page() -> HashMap<String, (Ty, Owned<AcvusRuntime>)> {
    HashMap::new()
}

fn make_context(
    interner: &Interner,
    functions: FxHashMap<QualifiedRef, Executable>,
) -> InterpreterContext {
    let executor = Arc::new(SequentialExecutor);
    InterpreterContext::new(interner, functions, executor)
}

// -- Tests -----------------------------------------------------------

/// Spawn a callee that returns arg + 1, eval it.
/// Entry:  Spawn(callee, [arg]) -> Eval(handle) -> Return(result)
/// Callee: param + 1 -> Return
#[tokio::test]
async fn spawn_eval_basic() {
    let interner = Interner::new();

    let entry_id = QualifiedRef::root(interner.intern("entry"));
    let callee_id = QualifiedRef::root(interner.intern("callee"));

    // -- Callee module: receives one param, returns param + 1 --
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
                op: acvus_mir::ir::BinOp::Add(acvus_mir::ir::Overflow::Trap),
                left: vids[0],
                right: vids[1],
            }),
            inst(InstKind::Return {
                value: vids[2],
                order: None,
            }),
        ];
        MirModule {
            declared_params: 1,
            main: MirBody {
                demoted_diamonds: Default::default(),
                task: Task::Sync,
                insts,
                val_types: types(vec![
                    (vids[0], Ty::I64),
                    (vids[1], Ty::I64),
                    (vids[2], Ty::I64),
                ]),
                params: vec![(interner.intern("p0"), vids[0])],
                captures: vec![],
                debug: DebugInfo::new(),
                val_factory: f,
                label_count: 0,
                order_param: None,
            },
            closures: FxHashMap::default(),
            ret: Ty::I64,
            flows: acvus_mir::ty::Flows::Every,
        }
    };

    // -- Entry module: spawn callee with arg=41, eval, return --
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
                callee_ty: acvus_mir::ty::Ty::error(),
                args: vec![vids[0]],
                order: None,
            }),
            inst(InstKind::Eval {
                dst: vids[2],
                src: vids[1],
                order: None,
            }),
            inst(InstKind::Return {
                value: vids[2],
                order: None,
            }),
        ];
        MirModule {
            declared_params: 0,
            main: MirBody {
                demoted_diamonds: Default::default(),
                // An `Eval` awaits (RFC-0046).
                task: Task::Async,
                insts,
                val_types: types(vec![
                    (vids[0], Ty::I64),
                    (vids[1], Ty::Handle(Box::new(Ty::I64))),
                    (vids[2], Ty::I64),
                ]),
                params: vec![],
                captures: vec![],
                debug: DebugInfo::new(),
                val_factory: f,
                label_count: 0,
                order_param: None,
            },
            closures: FxHashMap::default(),
            ret: Ty::I64,
            flows: acvus_mir::ty::Flows::Every,
        }
    };

    let mut functions = FxHashMap::default();
    let no_externs = FxHashMap::default();
    let no_contexts = FxHashMap::default();
    let prepare_ctx = acvus_interpreter::PrepareCtx {
        interner: &interner,
        externs: &no_externs,
        context_names: &no_contexts,
        instances: &acvus_extern::NoInstances,
        access: acvus_mir::graph::Access::Sync,
    };
    let prepare = |module| {
        Executable::Module(std::sync::Arc::new(acvus_interpreter::prepare_module(
            &module,
            &prepare_ctx,
        ).unwrap_or_else(|refused| panic!("the body is refused: {refused}"))))
    };
    functions.insert(entry_id, prepare(entry_module));
    functions.insert(callee_id, prepare(callee_module));

    let shared = make_context(&interner, functions);
    let page = empty_page();
    let mut interp = Interpreter::new(shared, entry_id, page);
    let result = interp
        .execute()
        .await
        .expect("the seeds hold every context the run fetches");

    assert_eq!(result.as_int(), 42);
}

/// Spawn with multiple args - callee receives two params and returns their sum.
#[tokio::test]
async fn spawn_eval_multi_args() {
    let interner = Interner::new();

    let entry_id = QualifiedRef::root(interner.intern("entry"));
    let callee_id = QualifiedRef::root(interner.intern("callee"));

    // -- Callee: param0 + param1 --
    let callee_module = {
        let mut f = LocalFactory::<ValueId>::new();
        let vids = alloc_n(&mut f, 3); // v0=param0, v1=param1, v2=result
        let insts = vec![
            inst(InstKind::BinOp {
                dst: vids[2],
                op: acvus_mir::ir::BinOp::Add(acvus_mir::ir::Overflow::Trap),
                left: vids[0],
                right: vids[1],
            }),
            inst(InstKind::Return {
                value: vids[2],
                order: None,
            }),
        ];
        MirModule {
            declared_params: 2,
            main: MirBody {
                demoted_diamonds: Default::default(),
                task: Task::Sync,
                insts,
                val_types: types(vec![
                    (vids[0], Ty::I64),
                    (vids[1], Ty::I64),
                    (vids[2], Ty::I64),
                ]),
                params: vec![
                    (interner.intern("p0"), vids[0]),
                    (interner.intern("p1"), vids[1]),
                ],
                captures: vec![],
                debug: DebugInfo::new(),
                val_factory: f,
                label_count: 0,
                order_param: None,
            },
            closures: FxHashMap::default(),
            ret: Ty::I64,
            flows: acvus_mir::ty::Flows::Every,
        }
    };

    // -- Entry: spawn(callee, [10, 32]) -> eval -> return --
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
                callee_ty: acvus_mir::ty::Ty::error(),
                args: vec![vids[0], vids[1]],
                order: None,
            }),
            inst(InstKind::Eval {
                dst: vids[3],
                src: vids[2],
                order: None,
            }),
            inst(InstKind::Return {
                value: vids[3],
                order: None,
            }),
        ];
        MirModule {
            declared_params: 0,
            main: MirBody {
                demoted_diamonds: Default::default(),
                // An `Eval` awaits (RFC-0046).
                task: Task::Async,
                insts,
                val_types: types(vec![
                    (vids[0], Ty::I64),
                    (vids[1], Ty::I64),
                    (vids[2], Ty::Handle(Box::new(Ty::I64))),
                    (vids[3], Ty::I64),
                ]),
                params: vec![],
                captures: vec![],
                debug: DebugInfo::new(),
                val_factory: f,
                label_count: 0,
                order_param: None,
            },
            closures: FxHashMap::default(),
            ret: Ty::I64,
            flows: acvus_mir::ty::Flows::Every,
        }
    };

    let mut functions = FxHashMap::default();
    let no_externs = FxHashMap::default();
    let no_contexts = FxHashMap::default();
    let prepare_ctx = acvus_interpreter::PrepareCtx {
        interner: &interner,
        externs: &no_externs,
        context_names: &no_contexts,
        instances: &acvus_extern::NoInstances,
        access: acvus_mir::graph::Access::Sync,
    };
    let prepare = |module| {
        Executable::Module(std::sync::Arc::new(acvus_interpreter::prepare_module(
            &module,
            &prepare_ctx,
        ).unwrap_or_else(|refused| panic!("the body is refused: {refused}"))))
    };
    functions.insert(entry_id, prepare(entry_module));
    functions.insert(callee_id, prepare(callee_module));

    let shared = make_context(&interner, functions);
    let page = empty_page();
    let mut interp = Interpreter::new(shared, entry_id, page);
    let result = interp
        .execute()
        .await
        .expect("the seeds hold every context the run fetches");

    assert_eq!(result.as_int(), 42);
}
