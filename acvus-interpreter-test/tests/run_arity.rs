//! `call_module` runs a module only with one argument per parameter. The
//! tooling route passes none, so a body that still reads an input no binding
//! fixed is stopped at the entry rather than run with a register nothing
//! filled.

use std::sync::Arc;

use acvus_interpreter_test::{check_graph, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

#[test]
fn a_module_run_one_argument_short_is_stopped_at_its_entry() {
    let i = Interner::new();
    let parsed = ParsedAst::Script(acvus_ast::parse_script(&i, "$n + 1").expect("main parses"));
    let compiled = check_graph(
        &i,
        parsed,
        &[],
        &FxHashMap::default(),
        acvus_ext::std_registries::<acvus_interpreter::AcvusRuntime>(),
        Ty::I64,
        Opt::Full,
        |_| {},
    )
    .unwrap_or_else(|refusal| panic!("`$n + 1` compiles: {:?}", refusal.messages));
    let (_, mut interp) = execute_compiled(
        &i,
        compiled,
        std::collections::HashMap::new(),
        Arc::new(acvus_interpreter::SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    let ran = runtime.block_on(interp.execute()).map(|_| ());
    let Err(acvus_interpreter::HostError::Trapped { message }) = ran else {
        panic!("a module run one argument short is stopped at its entry, and it gave {ran:?}")
    };
    assert!(message.contains("is run with 0 arguments, and it takes 1"), "{message}");
}
