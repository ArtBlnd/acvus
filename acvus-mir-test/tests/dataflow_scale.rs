use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use acvus_mir::ty::Ty;
use acvus_mir_test::compile_script_ir_with;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

const BRANCHES: usize = 64;

const LINEAR_WORK_LIMIT: Duration = Duration::from_secs(20);

#[test]
fn a_run_of_branches_is_validated_in_time_linear_in_its_length() {
    let source: String = std::iter::once("let x = 0;\n".to_string())
        .chain((0..BRANCHES).map(|i| format!("if @a == {i} {{ x = x + 1; }} else {{ x = x + 2; }};\n")))
        .chain(std::iter::once("x\n".to_string()))
        .collect();
    let (sent, received) = mpsc::channel();
    thread::spawn(move || {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("a"), Ty::I64)]);
        sent.send(compile_script_ir_with(&i, &source, &context, &[]))
            .expect("the test waits for the result");
    });
    let compiled = received.recv_timeout(LINEAR_WORK_LIMIT).expect(
        "a block is visited once per change to its entry, not once per path through 64 branches",
    );
    if let Err(err) = compiled {
        panic!("the program is admitted: {err}");
    }
}
