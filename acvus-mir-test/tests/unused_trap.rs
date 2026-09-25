//! An operation whose value nothing reads stays where it can trap and goes
//! where `analysis::raise` shows it cannot (RFC-0048 rule 8): a checked
//! index, a call of a local function, a call of an extern by whether its
//! instance is declared `total` (RFC-0082 rule 9).

use acvus_mir::graph::optimize::Opt;
use acvus_mir::ir::{BinOp, Callee, InstKind, MirModule};
use acvus_mir::printer::dump_with;
use acvus_mir::ty::{ParamTerm, Poly, PolyParam, Ty, TyTerm};
use acvus_mir_test::{compile_script_module_at, multi_fn_module_at};
use acvus_utils::Interner;

const LEVELS: [Opt; 2] = [Opt::None, Opt::Full];

fn script(i: &Interner, source: &str, opt: Opt) -> MirModule {
    let contexts = [(i.intern("x"), Ty::I64)].into_iter().collect();
    compile_script_module_at(i, source, &contexts, opt).unwrap_or_else(|e| panic!("{source}\n{e}"))
}

fn with_helper(i: &Interner, main: &str, helper: &str, opt: Opt) -> MirModule {
    let takes: Vec<PolyParam> = vec![ParamTerm::<Poly>::new(i.intern("n"), TyTerm::I64)];
    multi_fn_module_at(i, ("main", main), &[("h", helper, takes)], &[], &[], opt)
        .unwrap_or_else(|e| panic!("{main}\n{e}"))
        .module
}

fn count(module: &MirModule, held: impl Fn(&InstKind) -> bool) -> usize {
    module.main.insts.iter().filter(|inst| held(&inst.kind)).count()
}

fn is_index(kind: &InstKind) -> bool {
    matches!(kind, InstKind::Index { .. })
}

fn is_extern_call(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::FunctionCall {
            callee: Callee::Extern { .. },
            ..
        }
    )
}

fn is_local_call(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::FunctionCall {
            callee: Callee::Direct(_),
            ..
        } | InstKind::Spawn {
            callee: Callee::Direct(_),
            ..
        }
    )
}

fn is_call_or_inlined_division(kind: &InstKind) -> bool {
    is_local_call(kind)
        || matches!(
            kind,
            InstKind::BinOp {
                op: BinOp::Div,
                ..
            }
        )
}

#[test]
fn an_unused_index_past_the_length_stays() {
    let i = Interner::new();
    let source = "let a = [1, 2]; let k = 5; let e = a[k]; 3";
    for opt in LEVELS {
        let module = script(&i, source, opt);
        assert_eq!(count(&module, is_index), 1, "at {opt:?}\n{}", dump_with(&i, &module));
    }
}

#[test]
fn an_unused_index_the_intervals_prove_is_gone() {
    let i = Interner::new();
    let source = "let a = [1, 2]; let k = 1; let e = a[k]; 3";
    for opt in LEVELS {
        let module = script(&i, source, opt);
        assert_eq!(count(&module, is_index), 0, "at {opt:?}\n{}", dump_with(&i, &module));
    }
}

#[test]
fn an_unused_call_of_a_local_function_that_divides_by_zero_stays() {
    let i = Interner::new();
    for opt in LEVELS {
        let module = with_helper(&i, "let d = h(1); 5", "$n / 0", opt);
        assert_eq!(
            count(&module, is_call_or_inlined_division),
            1,
            "at {opt:?}\n{}",
            dump_with(&i, &module)
        );
    }
}

#[test]
fn an_unused_call_of_a_local_function_that_cannot_trap_is_gone() {
    let i = Interner::new();
    for opt in LEVELS {
        let module = with_helper(&i, "let d = h(1); 5", "$n / 2", opt);
        assert_eq!(
            count(&module, is_call_or_inlined_division),
            0,
            "at {opt:?}\n{}",
            dump_with(&i, &module)
        );
    }
}

#[test]
fn an_unused_call_of_a_recursive_local_function_stays() {
    let i = Interner::new();
    let helper = "if $n == 0 { 0 } else { h($n - 1) }";
    for opt in LEVELS {
        let module = with_helper(&i, "let d = h(3); 5", helper, opt);
        assert_eq!(count(&module, is_local_call), 1, "at {opt:?}\n{}", dump_with(&i, &module));
    }
}

#[test]
fn an_unused_call_of_an_extern_not_declared_total_stays() {
    let i = Interner::new();
    for opt in LEVELS {
        let module = script(&i, "let d = @x.wrapping_div(3); 5", opt);
        assert_eq!(count(&module, is_extern_call), 1, "at {opt:?}\n{}", dump_with(&i, &module));
    }
}

#[test]
fn an_unused_call_of_an_extern_declared_total_is_gone() {
    let i = Interner::new();
    for opt in LEVELS {
        let module = script(&i, "let d = @x.wrapping_add(3); 5", opt);
        assert_eq!(count(&module, is_extern_call), 0, "at {opt:?}\n{}", dump_with(&i, &module));
    }
}
