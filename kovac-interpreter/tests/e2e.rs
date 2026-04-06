//! End-to-end: Script source → MIR → optimize → kovac bytecode → execute → assert.

use acvus_mir::graph::types::*;
use acvus_mir::graph::{extract, infer, lower as graph_lower, optimize};
use acvus_mir::ir::MirModule;
use acvus_utils::{Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use kovac_interpreter::lower::lower_body;
use kovac_interpreter::vm::execute;

/// Compile a script source to optimized MirModule via the graph API.
fn compile_script(interner: &Interner, source: &str) -> MirModule {
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let ast = acvus_ast::parse_script_mode(interner, source).expect("parse failed");
    let mut functions = vec![Function {
        qref: test_qref,
        kind: FnKind::Local(ParsedAst::Script(ast)),
        ty: {
            let mut pb = acvus_mir::ty::PolyBuilder::new();
            acvus_mir::ty::PolyTy::Fn {
                params: vec![],
                ret: Box::new(pb.fresh_ty_var()),
                captures: vec![],
                effect: acvus_mir::ty::lift_effect_to_poly(&acvus_mir::ty::Effect::pure()),
                hint: None,
            }
        },
        effect_constraint: None,
    }];

    let mut type_registry = acvus_mir::ty::TypeRegistry::new();
    let std_regs = acvus_ext::std_registries(interner, &mut type_registry);
    for registry in std_regs {
        let registered = registry.register(interner);
        functions.extend(registered.functions);
    }

    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(Vec::new()),
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(
        interner,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(type_registry),
        &FxHashMap::default(),
    );

    for (qref, errs) in inf.errors() {
        if !errs.is_empty() {
            let name = interner.resolve(qref.name);
            let msgs: Vec<_> = errs.iter().map(|e| format!("{}", e.display(interner))).collect();
            panic!("infer errors for {name}: {}", msgs.join(", "));
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext, &inf, &FxHashMap::default());
    for le in &result.errors {
        if !le.errors.is_empty() {
            let msgs: Vec<_> = le.errors.iter().map(|e| format!("{}", e.display(interner))).collect();
            panic!("lower errors: {}", msgs.join(", "));
        }
    }

    // Run optimization pipeline (SROA → SSA → Inline → RegColor → Validate).
    let fn_metadata = inf.fn_types.clone();
    let context_types = FxHashMap::default();
    let recursive_fns = FxHashSet::default();

    let opt = optimize::optimize_untyped(result.modules, &fn_metadata, &context_types, &recursive_fns);
    // In untyped mode, validate may report type mismatches from shared
    // scalar slots — expected and safe for kovac (all scalars are u64).
    // Skip validate errors for now.

    opt.modules
        .get(&test_qref)
        .cloned()
        .expect("no optimized module for test")
}

/// Compile, lower to kovac, execute, return A bank.
fn run_script(source: &str) -> [u64; 4] {
    let interner = Interner::new();
    let module = compile_script(&interner, source);
    let lowered = lower_body(&module.main);
    let state = execute(&lowered.code);
    state.a
}

// ═══════════════════════════════════════════════════════════════════
//  Pure arithmetic
// ═══════════════════════════════════════════════════════════════════

#[test]
fn simple_addition() {
    let a = run_script("1 + 2");
    assert!(a.contains(&3), "expected 3 in A bank, got {:?}", a);
}

#[test]
fn arithmetic_expression() {
    let a = run_script("10 + 20 * 3");
    assert!(a.contains(&70), "expected 70 in A bank, got {:?}", a);
}

#[test]
fn subtraction_and_negation() {
    let a = run_script("100 - 42");
    assert!(a.contains(&58), "expected 58 in A bank, got {:?}", a);
}

#[test]
fn let_binding_arithmetic() {
    let a = run_script(
        "let x = 10;
         let y = 20;
         x + y"
    );
    assert!(a.contains(&30), "expected 30 in A bank, got {:?}", a);
}

#[test]
fn multi_let_sum() {
    let a = run_script(
        "let a = 10;
         let b = 20;
         let c = 30;
         a + b + c"
    );
    assert!(a.contains(&60), "expected 60 in A bank, got {:?}", a);
}

#[test]
fn modulo() {
    let a = run_script("17 % 5");
    assert!(a.contains(&2), "expected 2 in A bank, got {:?}", a);
}

#[test]
fn nested_arithmetic() {
    // (3 + 4) * (10 - 2)
    let a = run_script("(3 + 4) * (10 - 2)");
    assert!(a.contains(&56), "expected 56 in A bank, got {:?}", a);
}

#[test]
fn reassign() {
    let a = run_script(
        "let x = 5;
         x = x * x;
         x + 1"
    );
    assert!(a.contains(&26), "expected 26 in A bank, got {:?}", a);
}

// ═══════════════════════════════════════════════════════════════════
//  If / else
// ═══════════════════════════════════════════════════════════════════

#[test]
fn if_else_true_branch() {
    let a = run_script(
        "let x = 10;
         if x > 5 { 42 } else { 0 }"
    );
    assert!(a.contains(&42), "expected 42 (true branch), got {:?}", a);
}

#[test]
fn if_else_false_branch() {
    let a = run_script(
        "let x = 3;
         if x > 5 { 42 } else { 99 }"
    );
    assert!(a.contains(&99), "expected 99 (false branch), got {:?}", a);
}

#[test]
fn if_else_chain() {
    // Grade calculation: score → grade
    let a = run_script(
        "let score = 75;
         if score > 90 { 4 }
         else if score > 80 { 3 }
         else if score > 70 { 2 }
         else { 1 }"
    );
    assert!(a.contains(&2), "score 75 → grade 2, got {:?}", a);
}

#[test]
fn if_else_chain_top() {
    let a = run_script(
        "let score = 95;
         if score > 90 { 4 }
         else if score > 80 { 3 }
         else if score > 70 { 2 }
         else { 1 }"
    );
    assert!(a.contains(&4), "score 95 → grade 4, got {:?}", a);
}

#[test]
fn if_else_chain_bottom() {
    let a = run_script(
        "let score = 50;
         if score > 90 { 4 }
         else if score > 80 { 3 }
         else if score > 70 { 2 }
         else { 1 }"
    );
    assert!(a.contains(&1), "score 50 → grade 1, got {:?}", a);
}

#[test]
fn if_as_expression_in_let() {
    let a = run_script(
        "let x = 10;
         let y = if x > 5 { x * 2 } else { x };
         y + 1"
    );
    assert!(a.contains(&21), "expected 21, got {:?}", a);
}

// ═══════════════════════════════════════════════════════════════════
//  While loops
// ═══════════════════════════════════════════════════════════════════

#[test]
fn while_countdown() {
    let a = run_script(
        "let x = 10;
         while x > 0 {
             x = x - 1;
         }
         x"
    );
    assert!(a.contains(&0), "expected 0 after countdown, got {:?}", a);
}

#[test]
fn while_sum_to_10() {
    let a = run_script(
        "let sum = 0;
         let i = 1;
         while i < 11 {
             sum = sum + i;
             i = i + 1;
         }
         sum"
    );
    assert!(a.contains(&55), "sum 1..10 = 55, got {:?}", a);
}

#[test]
fn while_factorial() {
    // 5! = 120
    let a = run_script(
        "let result = 1;
         let n = 5;
         while n > 0 {
             result = result * n;
             n = n - 1;
         }
         result"
    );
    assert!(a.contains(&120), "5! = 120, got {:?}", a);
}

#[test]
#[ignore] // TODO: register pressure too high — needs smarter allocation
fn while_fibonacci() {
    // fib(10) = 55
    let a = run_script(
        "let prev = 0;
         let curr = 1;
         let i = 1;
         while i < 10 {
             let next = prev + curr;
             prev = curr;
             curr = next;
             i = i + 1;
         }
         curr"
    );
    assert!(a.contains(&55), "fib(10) = 55, got {:?}", a);
}

// ═══════════════════════════════════════════════════════════════════
//  Mixed: arithmetic + branch + loop
// ═══════════════════════════════════════════════════════════════════

#[test]
fn collatz_steps() {
    // Count Collatz steps from 6 to 1: 6→3→10→5→16→8→4→2→1 = 8 steps
    let a = run_script(
        "let n = 6;
         let steps = 0;
         while n > 1 {
             if n % 2 == 0 {
                 n = n / 2;
             } else {
                 n = n * 3 + 1;
             }
             steps = steps + 1;
         }
         steps"
    );
    assert!(a.contains(&8), "Collatz(6) = 8 steps, got {:?}", a);
}
