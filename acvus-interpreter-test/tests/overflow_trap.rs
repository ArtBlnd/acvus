//! RFC-0037 rule 3: the program's `+`, `-`, `*` and negation trap where the
//! exact result does not fit the width, and a shift where its amount is the
//! width or more, each with Rust's text; an operation a pass writes wraps.
//! Each source program runs at both optimization levels, with its operands
//! from the page so that nothing folds, and again with constant operands so
//! that the fold decides.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use acvus_extern::Owned;
use acvus_interpreter::{
    AcvusRuntime, Executable, HostError, Interpreter, InterpreterContext, SequentialExecutor,
};
use acvus_interpreter_test::corpus::{self, Init, Outcome, Stage};
use acvus_mir::graph::QualifiedRef;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ir::{
    BinOp, Checked, DebugInfo, Inst, InstKind, MirBody, MirModule, Overflow, UnaryOp, ValueId,
};
use acvus_mir::ty::{IntTy, Task, Ty};
use acvus_utils::{Interner, LocalFactory};
use rustc_hash::FxHashMap;

const LIMIT: Duration = Duration::from_secs(30);

const ADD: &str = "attempt to add with overflow";
const SUB: &str = "attempt to subtract with overflow";
const MUL: &str = "attempt to multiply with overflow";
const NEG: &str = "attempt to negate with overflow";
const SHL: &str = "attempt to shift left with overflow";
const SHR: &str = "attempt to shift right with overflow";

#[test]
fn corpus_child() {
    corpus::child();
}

/// The page seeds `@key` with the value the script `source` evaluates to.
fn bind(key: &str, source: &str) -> Init {
    Init {
        key: key.to_owned(),
        source: source.to_owned(),
    }
}

fn outcome(source: &str, inits: &[Init], opt: Opt) -> Outcome {
    acvus_interpreter_test::attempt_within!(source, inits = inits, opt, Stage::Run, LIMIT)
        .unwrap_or_else(|lapse| panic!("at {opt:?}, {lapse:?}: {source}"))
}

fn runs_to(source: &str, inits: &[Init], value: &str) {
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, inits, opt) {
            Outcome::Value(got) => assert_eq!(got, value, "at {opt:?}: {source} {inits:?}"),
            other => panic!("at {opt:?}, expected {value}, got {other:?}: {source} {inits:?}"),
        }
    }
}

fn traps_with(source: &str, inits: &[Init], text: &str) {
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, inits, opt) {
            Outcome::RunPanicked(why) => assert_eq!(why, text, "at {opt:?}: {source} {inits:?}"),
            other => panic!("at {opt:?}, expected `{text}`, got {other:?}: {source} {inits:?}"),
        }
    }
}

// -- Each operator, at a signed and an unsigned width -----------------

#[test]
fn an_addition_past_the_width_traps() {
    traps_with("@a + @b", &[bind("a", "200u8"), bind("b", "56u8")], ADD);
    runs_to("@a + @b", &[bind("a", "200u8"), bind("b", "55u8")], "255");
    traps_with(
        "@a + @b",
        &[bind("a", "9223372036854775807"), bind("b", "1")],
        ADD,
    );
    runs_to(
        "@a + @b",
        &[bind("a", "9223372036854775806"), bind("b", "1")],
        "9223372036854775807",
    );
    traps_with("@a + @b", &[bind("a", "-100i8"), bind("b", "-29i8")], ADD);
    runs_to(
        "@a + @b",
        &[bind("a", "-100i8"), bind("b", "-28i8")],
        "-128",
    );
}

#[test]
fn a_subtraction_past_the_width_traps() {
    traps_with("@a - @b", &[bind("a", "0u8"), bind("b", "1u8")], SUB);
    runs_to("@a - @b", &[bind("a", "1u8"), bind("b", "1u8")], "0");
    traps_with(
        "@a - @b",
        &[bind("a", "-2147483647i32"), bind("b", "2i32")],
        SUB,
    );
    runs_to(
        "@a - @b",
        &[bind("a", "-2147483647i32"), bind("b", "1i32")],
        "-2147483648",
    );
}

#[test]
fn a_multiplication_past_the_width_traps() {
    traps_with(
        "@a * @b",
        &[bind("a", "4294967296u64"), bind("b", "4294967296u64")],
        MUL,
    );
    runs_to(
        "@a * @b",
        &[bind("a", "4294967296u64"), bind("b", "2u64")],
        "8589934592",
    );
    traps_with("@a * @b", &[bind("a", "64i8"), bind("b", "2i8")], MUL);
    runs_to("@a * @b", &[bind("a", "-64i8"), bind("b", "2i8")], "-128");
}

/// Negation takes a signed width alone (RFC-0037 rule 2), so the unsigned
/// half of this operator is a type error, not a run.
#[test]
fn a_negation_of_the_minimum_traps() {
    traps_with("-@a", &[bind("a", "-127i8 - 1i8")], NEG);
    runs_to("-@a", &[bind("a", "-127i8")], "127");
    traps_with("-@a", &[bind("a", "-9223372036854775807 - 1")], NEG);
    runs_to(
        "-@a",
        &[bind("a", "-9223372036854775807")],
        "9223372036854775807",
    );
}

/// A chain of three operators is one prepared operation (RFC-0044 rule 5),
/// and its nodes trap as the operations it replaced do.
#[test]
fn an_operator_inside_a_chain_traps() {
    traps_with("@n * 2 + 1 - 3", &[bind("n", "100i8")], MUL);
    runs_to("@n * 2 + 1 - 3", &[bind("n", "50i8")], "98");
    traps_with(
        "@n * 2 + 1 - 3",
        &[bind("n", "9223372036854775808u64")],
        MUL,
    );
    traps_with("@n - 3 + 1", &[bind("n", "1u16")], SUB);
}

// -- A select and a removed loop ------------------------------------

/// A `Select` runs its arm's `+`, `-` or `*` whichever way the test goes, as
/// the overflowing form, and traps only where the computing side is taken
/// (RFC-0074 rule 2): the arm's overflow traps where the program takes the
/// arm, and not where it skips it.
#[test]
fn a_select_traps_only_on_the_side_it_takes() {
    for (op, text, start) in [
        ("+", ADD, "9223372036854775807"),
        ("-", SUB, "-9223372036854775807 - 1"),
        ("*", MUL, "9223372036854775807"),
    ] {
        let source = format!("let acc = @s; if @z != 0 {{ acc = acc {op} 2; }}; acc");
        traps_with(&source, &[bind("s", start), bind("z", "1")], text);
        let skipped = outcome(&source, &[bind("s", start), bind("z", "0")], Opt::None);
        let Outcome::Value(value) = skipped else {
            panic!("the skipped arm runs to a value: {skipped:?}")
        };
        runs_to(&source, &[bind("s", start), bind("z", "0")], &value);
    }
    let taken_else = "let acc = @s; if @z == 0 { } else { acc = acc + 1u8; }; acc";
    traps_with(taken_else, &[bind("s", "255u8"), bind("z", "1")], ADD);
    runs_to(taken_else, &[bind("s", "255u8"), bind("z", "0")], "255");
    runs_to(taken_else, &[bind("s", "254u8"), bind("z", "1")], "255");
}

/// A loop whose only work is a step that can overflow is removed, and its
/// trap is one exact check of the last step at the loop's place
/// (RFC-0088 rule 8): at both ends of the width, on both sides.
#[test]
fn a_removed_loop_keeps_its_step_trap() {
    let up = "let s = 0u8; for i in 0..@n { s = s + 3u8; } s";
    runs_to(up, &[bind("n", "85")], "255");
    traps_with(up, &[bind("n", "86")], ADD);
    runs_to(up, &[bind("n", "0")], "0");
    runs_to(up, &[bind("n", "-4")], "0");
    let down = "let s = -120i8; for i in 7..@n { s = s + @k; } s";
    runs_to(down, &[bind("n", "15"), bind("k", "-1i8")], "-128");
    traps_with(down, &[bind("n", "16"), bind("k", "-1i8")], ADD);
    runs_to(down, &[bind("n", "1000"), bind("k", "0i8")], "-120");
    runs_to(down, &[bind("n", "31"), bind("k", "10i8")], "120");
    traps_with(down, &[bind("n", "32"), bind("k", "10i8")], ADD);
    traps_with(
        "let s = 250u8; for x in [1, 2, 3] { s = s + 2u8; } s",
        &[],
        ADD,
    );
    runs_to(
        "let s = 249u8; for x in [1, 2, 3] { s = s + 2u8; } s",
        &[],
        "255",
    );
}

// -- The fold ---------------------------------------------------------

#[test]
fn an_overflowing_operation_on_constants_is_not_folded() {
    traps_with("let x = 200u8; x + 100", &[], ADD);
    traps_with("let x = 0u8; x - 1", &[], SUB);
    traps_with("let x = 64i8; x * 2", &[], MUL);
    runs_to("1u8 + 2", &[], "3");
    runs_to("let x = 200u8; x + 55", &[], "255");
}

/// At `i8`, `(x + 100) + 100` would join to `x + 200`, and `200` does not
/// fit the width, so the two stay two: `-100` runs through both, and `0`
/// traps at the second.
#[test]
fn two_constants_that_do_not_fit_together_do_not_join() {
    runs_to("(@x + 100) + 100", &[bind("x", "-100i8")], "100");
    traps_with("(@x + 100) + 100", &[bind("x", "0i8")], ADD);
}

// -- What a pass writes does not trap ---------------------------------

/// `code_motion.rs`'s example: `i + 1` hoisted into the loop head would run
/// `255 + 1` on the exit iteration.
#[test]
fn an_increment_in_a_loop_body_does_not_run_on_the_exit_iteration() {
    runs_to(
        "let i = 250; while i < @n { i = i + 1; } i",
        &[bind("n", "255u8")],
        "255",
    );
}

/// The counter of a `u8` range ends at 255, and IV canonicalization
/// computes `j` from it and after the loop from the trip count.
#[test]
fn a_counted_loop_whose_variable_ends_at_the_width_s_maximum_runs() {
    runs_to(
        "let j = 0u8; let s = 0u64; for i in 0u8..255u8 { j = j + 1u8; s = s + (j as u64); } \
         s + (j as u64)",
        &[],
        "32895",
    );
}

/// IV canonicalization writes `base + k·step` with `k·step` past `i64::MAX`
/// on the second and third iterations and after the loop, while `j` itself
/// never leaves the width.
#[test]
fn a_stepped_variable_near_the_width_s_ends_runs() {
    runs_to(
        "let j = @base; let s = 0; for i in 0..3 { s = s + j / 1099511627776; \
         j = j + 4611686018427387904; } s + j / 1099511627776",
        &[bind("base", "-9223372036854775807 - 1")],
        "-8388608",
    );
}

/// Strength reduction carries `i * k + x` and advances it after the last
/// iteration to `2 · 2^62 + 7`, a value the program never computes.
#[test]
fn a_reduced_counter_advanced_past_the_width_after_the_last_iteration_runs() {
    runs_to(
        "let k = @k; let x = @x; let acc = 0; \
         for i in 0..2 { acc = acc * acc % 1000003 + (i * k + x); } acc",
        &[bind("k", "4611686018427387904"), bind("x", "7")],
        "4611686018427387960",
    );
}

// -- The prepared operations, shifts included -------------------------

/// A trap's text, as the run reported it.
#[derive(Debug, PartialEq)]
struct Trapped(String);

fn trapped(text: &str) -> Result<i128, Trapped> {
    Err(Trapped(text.to_owned()))
}

/// A body of one operation over constants: its instructions, the values
/// they define, all at one integer width, and the factory that made them.
struct OneOperation {
    insts: Vec<Inst>,
    values: Vec<ValueId>,
    factory: LocalFactory<ValueId>,
}

fn at_zero(kind: InstKind) -> Inst {
    Inst {
        span: acvus_ast::Span::ZERO,
        kind,
    }
}

fn constant(dst: ValueId, value: i128) -> Inst {
    at_zero(InstKind::Const {
        dst,
        value: acvus_ast::Literal::Int(value),
    })
}

fn returned(value: ValueId) -> Inst {
    at_zero(InstKind::Return { value, order: None })
}

/// The surface writes no shift, so the shifts run from a body built here:
/// `left op right` on two constants, prepared and run without the
/// optimizer, whose fold would otherwise decide them.
async fn prepared(op: BinOp, ty: IntTy, left: i128, right: i128) -> Result<i128, Trapped> {
    let mut factory = LocalFactory::<ValueId>::new();
    let [l, r, dst] = [factory.next(), factory.next(), factory.next()];
    let body = OneOperation {
        insts: vec![
            constant(l, left),
            constant(r, right),
            at_zero(InstKind::BinOp {
                dst,
                op,
                left: l,
                right: r,
            }),
            returned(dst),
        ],
        values: vec![l, r, dst],
        factory,
    };
    run(body, ty).await
}

async fn prepared_negation(op: UnaryOp, ty: IntTy, operand: i128) -> Result<i128, Trapped> {
    let mut factory = LocalFactory::<ValueId>::new();
    let [src, dst] = [factory.next(), factory.next()];
    let body = OneOperation {
        insts: vec![
            constant(src, operand),
            at_zero(InstKind::UnaryOp {
                dst,
                op,
                operand: src,
            }),
            returned(dst),
        ],
        values: vec![src, dst],
        factory,
    };
    run(body, ty).await
}

async fn run(body: OneOperation, ty: IntTy) -> Result<i128, Trapped> {
    let interner = Interner::new();
    let entry = QualifiedRef::root(interner.intern("entry"));
    let OneOperation {
        insts,
        values,
        factory,
    } = body;
    let module = MirModule {
        declared_params: 0,
        main: MirBody {
            demoted_diamonds: Default::default(),
            task: Task::Sync,
            insts,
            val_types: values
                .into_iter()
                .map(|value| (value, Ty::Int(ty)))
                .collect(),
            params: vec![],
            captures: vec![],
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
            order_param: None,
        },
        closures: FxHashMap::default(),
        ret: Ty::Int(ty),
        flows: acvus_mir::ty::Flows::Every,
    };
    let no_externs = FxHashMap::default();
    let no_contexts = FxHashMap::default();
    let ctx = acvus_interpreter::PrepareCtx {
        interner: &interner,
        externs: &no_externs,
        context_names: &no_contexts,
        instances: &acvus_extern::NoInstances,
        access: acvus_mir::graph::Access::Sync,
    };
    let prepared = Executable::Module(Arc::new(acvus_interpreter::prepare_module(&module, &ctx)));
    let functions: FxHashMap<QualifiedRef, Executable> =
        std::iter::once((entry, prepared)).collect();
    let shared = InterpreterContext::new(&interner, functions, Arc::new(SequentialExecutor));
    let page: HashMap<String, (Ty, Owned<AcvusRuntime>)> = HashMap::new();
    let mut interp = Interpreter::new(shared, entry, page);
    match interp.execute().await {
        Ok(value) => Ok(ty.read(value.bits())),
        Err(HostError::Trapped { message }) => Err(Trapped(message)),
        Err(other) => panic!("a body of constants fetches nothing: {other:?}"),
    }
}

const TRAP: Overflow = Overflow::Trap;
const WRAP: Overflow = Overflow::Wrap;

#[tokio::test]
async fn a_shift_by_the_width_or_more_traps() {
    assert_eq!(
        prepared(BinOp::Shl(TRAP), IntTy::U8, 1, 8).await,
        trapped(SHL)
    );
    assert_eq!(prepared(BinOp::Shl(TRAP), IntTy::U8, 1, 7).await, Ok(128));
    assert_eq!(prepared(BinOp::Shl(TRAP), IntTy::U8, 200, 1).await, Ok(144));
    assert_eq!(
        prepared(BinOp::Shl(TRAP), IntTy::I64, 1, 64).await,
        trapped(SHL)
    );
    assert_eq!(
        prepared(BinOp::Shr(TRAP), IntTy::U64, 1, 64).await,
        trapped(SHR)
    );
    assert_eq!(prepared(BinOp::Shr(TRAP), IntTy::U64, 256, 8).await, Ok(1));
    assert_eq!(
        prepared(BinOp::Shr(TRAP), IntTy::I8, -8, -1).await,
        trapped(SHR)
    );
    assert_eq!(prepared(BinOp::Shr(TRAP), IntTy::I8, -8, 7).await, Ok(-1));
}

#[tokio::test]
async fn a_wrapping_shift_takes_its_amount_modulo_the_width() {
    assert_eq!(prepared(BinOp::Shl(WRAP), IntTy::U8, 1, 8).await, Ok(1));
    assert_eq!(prepared(BinOp::Shl(WRAP), IntTy::I64, 1, 65).await, Ok(2));
    assert_eq!(prepared(BinOp::Shr(WRAP), IntTy::I8, -8, -1).await, Ok(-1));
}

#[tokio::test]
async fn a_prepared_trapping_operation_traps_and_a_wrapping_one_wraps() {
    assert_eq!(
        prepared(BinOp::Add(TRAP), IntTy::U8, 255, 1).await,
        trapped(ADD)
    );
    assert_eq!(prepared(BinOp::Add(WRAP), IntTy::U8, 255, 1).await, Ok(0));
    assert_eq!(
        prepared(BinOp::Sub(TRAP), IntTy::I64, i128::from(i64::MIN), 1).await,
        trapped(SUB)
    );
    assert_eq!(
        prepared(BinOp::Sub(WRAP), IntTy::I64, i128::from(i64::MIN), 1).await,
        Ok(i128::from(i64::MAX))
    );
    assert_eq!(
        prepared(BinOp::Mul(TRAP), IntTy::U64, 1 << 32, 1 << 32).await,
        trapped(MUL)
    );
    assert_eq!(
        prepared(BinOp::Mul(WRAP), IntTy::U64, 1 << 32, 1 << 32).await,
        Ok(0)
    );
    assert_eq!(
        prepared_negation(UnaryOp::Neg(TRAP), IntTy::I8, -128).await,
        trapped(NEG)
    );
    assert_eq!(
        prepared_negation(UnaryOp::Neg(WRAP), IntTy::I8, -128).await,
        Ok(-128)
    );
}

/// `check left op right` over two constants, then the left operand
/// returned: the check traps where the operation would, and otherwise the
/// run goes on with nothing written.
async fn prepared_check(op: Checked, ty: IntTy, left: i128, right: i128) -> Result<i128, Trapped> {
    let mut factory = LocalFactory::<ValueId>::new();
    let [l, r] = [factory.next(), factory.next()];
    let body = OneOperation {
        insts: vec![
            constant(l, left),
            constant(r, right),
            at_zero(InstKind::Check {
                op,
                left: l,
                right: r,
            }),
            returned(l),
        ],
        values: vec![l, r],
        factory,
    };
    run(body, ty).await
}

#[tokio::test]
async fn a_check_traps_where_its_operation_would_and_otherwise_runs_on() {
    assert_eq!(
        prepared_check(Checked::Add, IntTy::U8, 255, 1).await,
        trapped(ADD)
    );
    assert_eq!(
        prepared_check(Checked::Add, IntTy::U8, 254, 1).await,
        Ok(254)
    );
    assert_eq!(
        prepared_check(Checked::Sub, IntTy::I64, i128::from(i64::MIN), 1).await,
        trapped(SUB)
    );
    assert_eq!(
        prepared_check(Checked::Sub, IntTy::I64, i128::from(i64::MIN), -1).await,
        Ok(i128::from(i64::MIN))
    );
    assert_eq!(
        prepared_check(Checked::Mul, IntTy::I8, -128, -1).await,
        trapped(MUL)
    );
    assert_eq!(
        prepared_check(Checked::Mul, IntTy::I8, -64, 2).await,
        Ok(-64)
    );
}
