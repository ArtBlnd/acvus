//! Scratch tool: print the prepared block listing of a bench body.
//!
//! Every failure here is propagated as a panic: the tool either measures or
//! does not run. Nothing about the listing is inferred — a superinstruction's
//! parts are printed from `Op::owns`, not guessed.

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::code::{Body, Code, CodeBody, ExprBody, Named, Op};
use acvus_interpreter::{AcvusRuntime, PrepareCtx, Value, prepare_module};
use acvus_interpreter_test::listing::last_path_segment;
use acvus_interpreter_test::{Context, compile_source_with_externs, split_context, typed};
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[extern_fn(effect = pure)]
fn some_of(i: i64) -> Option<i64> {
    if i % 2 == 0 { Some(i) } else { None }
}

#[extern_fn(effect = pure)]
fn id_of(i: i64) -> i64 {
    i
}

#[extern_fn(effect = pure)]
fn even_of(i: i64) -> bool {
    i % 2 == 0
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "bench",
        fns: [some_of, id_of, even_of],
    });
    regs
}

/// One context name the bench sources read, with the type the compiler is
/// told and the value a run would see.
struct ContextName {
    name: &'static str,
    ty: Ty,
    value: Value,
}

/// The operation's type with the module paths dropped, its generic arguments
/// included, which is the instance `prepare` chose.
fn named(of: &dyn Named) -> String {
    last_path_segment(of)
}

fn dump_blocks(indent: &str, heads: &[Box<dyn Op>]) {
    for (at, head) in heads.iter().enumerate() {
        println!("{indent}block {at}:");
        dump_chain(indent, head.as_ref());
    }
}

/// The chain from `head`, printed one operation per line: the node with no
/// successor is what ends it.
fn dump_chain(indent: &str, head: &dyn Op) {
    let mut at = head;
    loop {
        match at.successor() {
            Some(next) => {
                println!("{indent}  {}", named(at));
                for owned in at.owns() {
                    println!("{indent}     {}:", owned.part);
                    dump_chain(&format!("{indent}     "), owned.head);
                }
                at = next;
            }
            None => {
                println!("{indent}  -> {}", named(at));
                return;
            }
        }
    }
}

fn dump_body(name: &str, code: &Body) {
    println!(
        "== {name}: blocks={} entry={} frame_len={} params={:?} captures={:?}",
        code.heads.len(),
        code.entry,
        code.frame_len,
        code.params,
        code.captures
    );
    println!(
        "   slot_kinds=[{}] entry_konsts=[{}]",
        code.slot_kinds
            .iter()
            .map(|k| format!("r{}={:?}", k.slot.index(), k.kind))
            .collect::<Vec<_>>()
            .join(", "),
        code.entry_konsts
            .iter()
            .map(|k| format!("r{}={:?}", k.slot.index(), k.value))
            .collect::<Vec<_>>()
            .join(", ")
    );
    dump_blocks("  ", &code.heads);
}

fn dump(name: &str, code: &Code) {
    match &code.body {
        CodeBody::Body(body) => dump_body(name, body),
        CodeBody::Expr(expr) => match &expr.body {
            ExprBody::Argument(at) => println!(
                "== {name}: CodeBody::Expr arity={} returns argument {at} (no frame, no Machine)",
                expr.arity
            ),
            ExprBody::Chain(body) => println!(
                "== {name}: CodeBody::Expr arity={} chain shape={:?} root={:?} kind={:?} konsts={:?} \
                 (no frame, no Machine)",
                expr.arity, body.plan.shape, body.plan.root, body.kind, body.konsts
            ),
        },
        CodeBody::Rust => println!("== {name}: CodeBody::Rust (a Rust closure, no frame, no Machine)"),
    }
}

/// The return type the host declares for the source's `main` (RFC-0054),
/// named by the second command-line argument; `i64` when it is absent.
fn declared_return(name: Option<&str>) -> Ty {
    match name.unwrap_or("i64") {
        "i64" => Ty::I64,
        "f64" => Ty::Float,
        "bool" => Ty::Bool,
        "string" => Ty::String,
        "unit" => Ty::Unit,
        other => panic!("unknown return type `{other}`: i64, f64, bool, string or unit"),
    }
}

fn main() {
    let source = std::env::args()
        .nth(1)
        .expect("a source argument on the command line");
    let ret = declared_return(std::env::args().nth(2).as_deref());
    let interner = Interner::new();
    let names = [
        ContextName {
            name: "n",
            ty: Ty::I64,
            value: Value::int(1000),
        },
        ContextName {
            name: "w",
            ty: Ty::I64,
            value: Value::int(64),
        },
        ContextName {
            name: "h",
            ty: Ty::I64,
            value: Value::int(64),
        },
        ContextName {
            name: "max",
            ty: Ty::I64,
            value: Value::int(50),
        },
    ];
    let mut context: Context = names
        .into_iter()
        .map(|ContextName { name, ty, value }| (interner.intern(name), typed(ty, value)))
        .collect();
    let rows = serde_json::json!({
        "query": [0.1, 0.2],
        "keys": [[0.1, 0.2], [0.3, 0.4]],
        "values": [[1.0, 2.0], [3.0, 4.0]],
    });
    for (name, value) in rows.as_object().expect("an object of contexts") {
        context.insert(
            interner.intern(name),
            acvus_interpreter_test::value_from_json(&interner, value),
        );
    }
    let (context_types, _snapshot) = split_context(&interner, context);
    let ast = ParsedAst::Script(acvus_ast::parse_script(&interner, &source).expect("parse error"));
    let cr = compile_source_with_externs(&interner, ast, &context_types, registries(), ret);
    let ctx = PrepareCtx {
        interner: &interner,
        externs: &cr.extern_executables,
        context_names: &cr.context_names,
        instances: &cr.instances,
        access: acvus_mir::graph::Access::Sync,
    };
    for (qref, module) in &cr.modules {
        let prepared = prepare_module(module, &ctx)
            .unwrap_or_else(|refused| panic!("the body is refused: {refused}"));
        dump_body(&format!("{qref:?} main"), &prepared.main);
        for (label, closure) in &prepared.closures {
            dump(&format!("{qref:?} closure {label:?}"), closure);
        }
    }
}
