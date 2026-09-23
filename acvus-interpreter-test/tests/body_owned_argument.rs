//! Inlining once substituted the argument for a parameter the body names as
//! storage, as `&$p` does. The storage target then named a register that an
//! instruction of the caller defines; `dce` swept that instruction as unread,
//! and at `Opt::Full` the machine read a string that was never made.

use acvus_interpreter::Value;
use acvus_interpreter_test::{Helper, Refusal, check_graph, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{
    LenTerm, Mutability, ObjectTy, ParamTerm, Poly, PolyParam, Ty, TypeArg, lift_to_poly,
};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use std::sync::Arc;

fn param(i: &Interner, ty: Ty) -> Vec<PolyParam> {
    vec![ParamTerm::<Poly>::new(i.intern("p"), lift_to_poly(&ty))]
}

fn holder(i: &Interner) -> Ty {
    Ty::Object(ObjectTy::written(FxHashMap::from_iter([(
        i.intern("s"),
        Ty::String,
    )])))
}

fn strings() -> Ty {
    Ty::Array(Box::new(Ty::String), LenTerm::Known(2))
}

fn compile_and_run(
    i: &Interner,
    helpers: &[Helper<'_>],
    main: &str,
    opt: Opt,
) -> Result<Value, Refusal> {
    let parsed = ParsedAst::Script(acvus_ast::parse_script(i, main).expect("main parses"));
    let cr = check_graph(
        i,
        parsed,
        helpers,
        &FxHashMap::default(),
        acvus_ext::std_registries::<acvus_interpreter::AcvusRuntime>(),
        Ty::I64,
        opt,
        |_| {},
    )?;
    let (_, mut interp) = execute_compiled(
        i,
        cr,
        std::collections::HashMap::new(),
        Arc::new(acvus_interpreter::SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    Ok(runtime.block_on(interp.execute()))
}

fn integer_at(i: &Interner, helpers: &[Helper<'_>], main: &str, opt: Opt) -> i64 {
    compile_and_run(i, helpers, main, opt)
        .unwrap_or_else(|r| panic!("{opt:?} refused:\n  {}", r.messages.join("\n  ")))
        .as_int()
}

fn one(i: &Interner, source: &'static str, ty: Ty) -> Vec<Helper<'static>> {
    vec![Helper {
        name: "h",
        source,
        params: param(i, ty),
    }]
}

#[test]
fn a_string_parameter_lent_to_len_at_full() {
    let i = Interner::new();
    let h = one(&i, "len(&$p) as i64\n", Ty::String);
    assert_eq!(integer_at(&i, &h, "h(\"ab\".to_string())\n", Opt::Full), 2);
}

#[test]
fn a_string_parameter_lent_to_len_at_none() {
    let i = Interner::new();
    let h = one(&i, "len(&$p) as i64\n", Ty::String);
    assert_eq!(integer_at(&i, &h, "h(\"ab\".to_string())\n", Opt::None), 2);
}

#[test]
fn a_string_parameter_compared_at_full() {
    let i = Interner::new();
    let h = one(&i, "if $p == \"ab\" { 1 } else { 0 }\n", Ty::String);
    assert_eq!(integer_at(&i, &h, "h(\"ab\".to_string())\n", Opt::Full), 1);
}

#[test]
fn a_string_parameter_compared_at_none() {
    let i = Interner::new();
    let h = one(&i, "if $p == \"ab\" { 1 } else { 0 }\n", Ty::String);
    assert_eq!(integer_at(&i, &h, "h(\"ab\".to_string())\n", Opt::None), 1);
}

#[test]
fn a_string_parameter_returned_whole_at_full() {
    let i = Interner::new();
    let h = one(&i, "$p\n", Ty::String);
    let main = "let r = h(\"ab\".to_string());\nlen(&r) as i64\n";
    assert_eq!(integer_at(&i, &h, main, Opt::Full), 2);
}

#[test]
fn a_string_parameter_returned_whole_at_none() {
    let i = Interner::new();
    let h = one(&i, "$p\n", Ty::String);
    let main = "let r = h(\"ab\".to_string());\nlen(&r) as i64\n";
    assert_eq!(integer_at(&i, &h, main, Opt::None), 2);
}

const READ_THEN_RETURNED_STRING: &str = "if len(&$p) == 2 { $p } else { \"\".to_string() }\n";

#[test]
fn a_string_parameter_read_then_returned_at_full() {
    let i = Interner::new();
    let h = one(&i, READ_THEN_RETURNED_STRING, Ty::String);
    let main = "let r = h(\"ab\".to_string());\nlen(&r) as i64\n";
    assert_eq!(integer_at(&i, &h, main, Opt::Full), 2);
}

#[test]
fn a_string_parameter_read_then_returned_at_none() {
    let i = Interner::new();
    let h = one(&i, READ_THEN_RETURNED_STRING, Ty::String);
    let main = "let r = h(\"ab\".to_string());\nlen(&r) as i64\n";
    assert_eq!(integer_at(&i, &h, main, Opt::None), 2);
}

/// The first call is not the last use of `s`, so RFC-0018 rule 2 passes it
/// a copy.
const REUSED_AFTER_THE_CALL: &str = "let s = \"ab\".to_string();\nh(s) * 10 + h(s)\n";

#[test]
fn a_reused_string_lent_to_len_at_full() {
    let i = Interner::new();
    let h = one(&i, "len(&$p) as i64\n", Ty::String);
    assert_eq!(integer_at(&i, &h, REUSED_AFTER_THE_CALL, Opt::Full), 22);
}

#[test]
fn a_reused_string_lent_to_len_at_none() {
    let i = Interner::new();
    let h = one(&i, "len(&$p) as i64\n", Ty::String);
    assert_eq!(integer_at(&i, &h, REUSED_AFTER_THE_CALL, Opt::None), 22);
}

#[test]
fn a_reused_string_compared_at_full() {
    let i = Interner::new();
    let h = one(&i, "if $p == \"ab\" { 1 } else { 0 }\n", Ty::String);
    assert_eq!(integer_at(&i, &h, REUSED_AFTER_THE_CALL, Opt::Full), 11);
}

#[test]
fn a_reused_string_compared_at_none() {
    let i = Interner::new();
    let h = one(&i, "if $p == \"ab\" { 1 } else { 0 }\n", Ty::String);
    assert_eq!(integer_at(&i, &h, REUSED_AFTER_THE_CALL, Opt::None), 11);
}

const HOLDER: &str = "h({ s: \"ab\".to_string(), })\n";

#[test]
fn an_object_parameter_read_by_field_at_full() {
    let i = Interner::new();
    let h = one(&i, "len(&$p.s) as i64\n", holder(&i));
    assert_eq!(integer_at(&i, &h, HOLDER, Opt::Full), 2);
}

#[test]
fn an_object_parameter_read_by_field_at_none() {
    let i = Interner::new();
    let h = one(&i, "len(&$p.s) as i64\n", holder(&i));
    assert_eq!(integer_at(&i, &h, HOLDER, Opt::None), 2);
}

const READ_THEN_RETURNED_HOLDER: &str =
    "if len(&$p.s) == 2 { $p } else { { s: \"\".to_string(), } }\n";

const HOLDER_RETURNED: &str = "let r = h({ s: \"ab\".to_string(), });\nlen(&r.s) as i64\n";

#[test]
fn an_object_parameter_read_then_returned_at_full() {
    let i = Interner::new();
    let h = one(&i, READ_THEN_RETURNED_HOLDER, holder(&i));
    assert_eq!(integer_at(&i, &h, HOLDER_RETURNED, Opt::Full), 2);
}

#[test]
fn an_object_parameter_read_then_returned_at_none() {
    let i = Interner::new();
    let h = one(&i, READ_THEN_RETURNED_HOLDER, holder(&i));
    assert_eq!(integer_at(&i, &h, HOLDER_RETURNED, Opt::None), 2);
}

const STRINGS: &str = "h([\"a\".to_string(), \"bcd\".to_string()])\n";

#[test]
fn an_array_parameter_read_by_index_at_full() {
    let i = Interner::new();
    let h = one(&i, "len(&$p[1]) as i64\n", strings());
    assert_eq!(integer_at(&i, &h, STRINGS, Opt::Full), 3);
}

#[test]
fn an_array_parameter_read_by_index_at_none() {
    let i = Interner::new();
    let h = one(&i, "len(&$p[1]) as i64\n", strings());
    assert_eq!(integer_at(&i, &h, STRINGS, Opt::None), 3);
}

const READ_THEN_RETURNED_STRINGS: &str =
    "if len(&$p[1]) == 3 { $p } else { [\"\".to_string(), \"\".to_string()] }\n";

const STRINGS_RETURNED: &str =
    "let r = h([\"a\".to_string(), \"bcd\".to_string()]);\nlen(&r[1]) as i64\n";

#[test]
fn an_array_parameter_read_then_returned_at_full() {
    let i = Interner::new();
    let h = one(&i, READ_THEN_RETURNED_STRINGS, strings());
    assert_eq!(integer_at(&i, &h, STRINGS_RETURNED, Opt::Full), 3);
}

#[test]
fn an_array_parameter_read_then_returned_at_none() {
    let i = Interner::new();
    let h = one(&i, READ_THEN_RETURNED_STRINGS, strings());
    assert_eq!(integer_at(&i, &h, STRINGS_RETURNED, Opt::None), 3);
}

const LENT: &str = "let s = \"ab\".to_string();\nh(&s)\n";

fn lent_string() -> Ty {
    Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::String)))
}

#[test]
fn control_a_lent_string_parameter_at_full() {
    let i = Interner::new();
    let h = one(&i, "len($p) as i64\n", lent_string());
    assert_eq!(integer_at(&i, &h, LENT, Opt::Full), 2);
}

#[test]
fn control_a_lent_string_parameter_at_none() {
    let i = Interner::new();
    let h = one(&i, "len($p) as i64\n", lent_string());
    assert_eq!(integer_at(&i, &h, LENT, Opt::None), 2);
}

const LAMBDA: &str = "let h = |p| -> len(&p) as i64;\nh(\"ab\".to_string())\n";

#[test]
fn a_lambda_lending_its_string_parameter_at_full() {
    let i = Interner::new();
    assert_eq!(integer_at(&i, &[], LAMBDA, Opt::Full), 2);
}

#[test]
fn a_lambda_lending_its_string_parameter_at_none() {
    let i = Interner::new();
    assert_eq!(integer_at(&i, &[], LAMBDA, Opt::None), 2);
}
