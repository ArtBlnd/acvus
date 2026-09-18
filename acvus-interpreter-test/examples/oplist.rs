//! Scratch tool: print the prepared block listing of a bench body.
//!
//! Every failure here is propagated as a panic: the tool either measures or
//! does not run. Nothing about the listing is inferred — a superinstruction's
//! blocks are printed from `Terminator::owns`, not guessed.

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::code::{Block, Body, Code, ExprBody, Named, Op};
use acvus_interpreter::{AcvusRuntime, PrepareCtx, Value, prepare_module};
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

/// The short name of an operation or a terminator: the last path segment
/// of its Rust type, which is the instance `prepare` chose.
fn named(of: &dyn Named) -> &str {
    let full = of.name();
    match full.rsplit_once("::") {
        Some((_, last)) => last,
        None => full,
    }
}

fn dump_blocks(indent: &str, blocks: &[Block]) {
    for (at, block) in blocks.iter().enumerate() {
        println!("{indent}block {at}:");
        dump_ops(indent, &block.ops);
        println!("{indent}  -> {}", named(block.end.as_ref()));
    }
}

fn dump_ops(indent: &str, ops: &[Box<dyn Op>]) {
    for op in ops {
        println!("{indent}  {}", named(op.as_ref()));
        for owned in op.owns() {
            println!("{indent}     {}:", owned.part);
            dump_ops(&format!("{indent}       "), owned.ops);
        }
    }
}

fn dump_body(name: &str, code: &Body) {
    println!(
        "== {name}: blocks={} entry={} frame_len={} params={:?} captures={:?}",
        code.blocks.len(),
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
    dump_blocks("  ", &code.blocks);
}

fn dump(name: &str, code: &Code) {
    match code {
        Code::Body(body) => dump_body(name, body),
        Code::Expr(expr) => match &expr.body {
            ExprBody::Argument(at) => println!(
                "== {name}: Code::Expr arity={} returns argument {at} (no frame, no Machine)",
                expr.arity
            ),
            ExprBody::Chain(body) => println!(
                "== {name}: Code::Expr arity={} chain shape={:?} root={:?} kind={:?} konsts={:?} \
                 (no frame, no Machine)",
                expr.arity, body.plan.shape, body.plan.root, body.kind, body.konsts
            ),
        },
    }
}

fn main() {
    let source = std::env::args()
        .nth(1)
        .expect("a source argument on the command line");
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
    let cr = compile_source_with_externs(&interner, ast, &context_types, registries());
    let ctx = PrepareCtx {
        interner: &interner,
        externs: &cr.extern_executables,
        context_names: &cr.context_names,
    };
    for (qref, module) in &cr.modules {
        let prepared = prepare_module(module, &ctx);
        dump(&format!("{qref:?} main"), &prepared.main);
        for (label, closure) in &prepared.closures {
            dump(&format!("{qref:?} closure {label:?}"), closure);
        }
    }
}
