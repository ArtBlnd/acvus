//! Scratch tool: print the prepared op listing of a bench body, with the
//! address of each operation's function so `nm` can name it.
//!
//! Every failure here is propagated as a panic: the tool either measures or
//! does not run. Nothing about the listing is inferred — a loop's blocks are
//! printed from the payload table, not guessed from an operation's words.

use std::collections::HashMap;

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::code::{Body, Chain, Code, DiamondArm, ExprBody, Op, Payload};
use acvus_interpreter::{AcvusRuntime, PrepareCtx, SlotCount, Slots, Value, prepare_module};
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

/// The load address of the running executable's first mapping: `nm`
/// addresses are file-relative, so this is what turns a runtime function
/// address back into one.
fn load_base() -> usize {
    let exe = std::fs::read_link("/proc/self/exe").expect("a readable /proc/self/exe");
    let exe = exe.to_str().expect("a utf-8 executable path");
    let maps = std::fs::read_to_string("/proc/self/maps").expect("a readable /proc/self/maps");
    for line in maps.lines() {
        if line.ends_with(exe) {
            let start = line.split('-').next().expect("a mapping start");
            return usize::from_str_radix(start, 16).expect("a hex mapping start");
        }
    }
    panic!("the executable has no mapping of its own path")
}

fn word(w: u32) -> String {
    if w == u32::MAX {
        "-".to_string()
    } else {
        w.to_string()
    }
}

fn line(at: usize, op: &Op, base: usize) -> String {
    let f = (op.f as usize)
        .checked_sub(base)
        .expect("an operation's function lies above the load base");
    format!(
        "{at:>3}: f=0x{f:x} a={} b={} c={} d={} p={}",
        word(op.a),
        word(op.b),
        word(op.c),
        word(op.d),
        op.p
    )
}

fn dump_chain(indent: &str, chain: &Chain) -> SlotCount {
    let shape = chain.shape;
    let slots = Slots::of(chain);
    let offsets: Vec<String> = chain.leaf_offsets[..shape.leaves()]
        .iter()
        .map(|off| format!("r{}", (off - 8) / 16))
        .collect();
    println!(
        "{indent}chain shape={shape:?} nodes={} ops={:?} root={:?}",
        shape.interior() + 1,
        &chain.post_order_ops[..shape.interior()],
        chain.root
    );
    println!(
        "{indent}  leaves=[{}] slots={:?}+{:?}",
        offsets.join(", "),
        &slots.ops[..shape.interior()],
        slots.root
    );
    slots.count()
}

fn dump_body(name: &str, code: &Body, base: usize, tally: &mut SlotCount) {
    println!(
        "== {name}: frame_len={} params={:?} captures={:?} top-level ops={} entry_konsts={:?}",
        code.frame_len,
        code.params,
        code.captures,
        code.ops.len(),
        code.entry_konsts
            .iter()
            .map(|k| format!("r{}={:?}", k.slot, k.value))
            .collect::<Vec<_>>()
    );
    for (at, op) in code.ops.iter().enumerate() {
        println!("  {}", line(at, op, base));
    }
    for (index, payload) in code.payloads.iter().enumerate() {
        match payload {
            Payload::Chain(chain) => {
                println!("  payload {index}: Chain");
                let count = dump_chain("     ", chain);
                tally.concrete += count.concrete;
                tally.total += count.total;
            }
            Payload::Diamond(arms) => {
                println!("  payload {index}: Diamond");
                dump_arm("on_true", &arms.on_true, base);
                dump_arm("on_false", &arms.on_false, base);
            }
            Payload::Loop(body) => {
                println!("  payload {index}: Loop (cond_slot={})", body.cond_slot);
                println!(
                    "     moves: enter={} into_body={} back={} exit={}",
                    body.enter.len(),
                    body.into_body.len(),
                    body.back.len(),
                    body.exit.len()
                );
                println!("     head:");
                for (at, op) in body.head.iter().enumerate() {
                    println!("       {}", line(at, op, base));
                }
                println!("     body:");
                for (at, op) in body.body.iter().enumerate() {
                    println!("       {}", line(at, op, base));
                }
            }
            _ => {}
        }
    }
}

fn dump_arm(side: &str, arm: &DiamondArm, base: usize) {
    println!("     {side}: join moves={}", arm.join.len());
    for (at, op) in arm.block.iter().enumerate() {
        println!("       {}", line(at, op, base));
    }
}

fn dump(name: &str, code: &Code, base: usize, tally: &mut SlotCount) {
    match code {
        Code::Body(body) => dump_body(name, body, base, tally),
        Code::Expr(expr) => match &expr.body {
            ExprBody::Argument(at) => println!(
                "== {name}: Code::Expr arity={} returns argument {at} (no frame, no Machine)",
                expr.arity
            ),
            ExprBody::Chain(body) => {
                println!(
                    "== {name}: Code::Expr arity={} eval=0x{:x} konsts={:?} (no frame, no Machine)",
                    expr.arity,
                    (body.eval as usize)
                        .checked_sub(base)
                        .expect("an evaluator lies above the load base"),
                    body.konsts
                );
                let count = dump_chain("  ", &body.chain);
                tally.concrete += count.concrete;
                tally.total += count.total;
            }
        },
    }
}

fn main() {
    let base = load_base();
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
    let context: Context = names
        .into_iter()
        .map(|ContextName { name, ty, value }| (interner.intern(name), typed(ty, value)))
        .collect();
    let (context_types, _snapshot): (_, HashMap<String, Value>) = split_context(&interner, context);
    let ast = ParsedAst::Script(acvus_ast::parse_script(&interner, &source).expect("parse error"));
    let cr = compile_source_with_externs(&interner, ast, &context_types, registries());
    let ctx = PrepareCtx {
        interner: &interner,
        externs: &cr.extern_executables,
        context_names: &cr.context_names,
    };
    println!("load base 0x{base:x}");
    let mut tally = SlotCount {
        concrete: 0,
        total: 0,
    };
    for (qref, module) in &cr.modules {
        let prepared = prepare_module(module, &ctx);
        dump(&format!("{qref:?} main"), &prepared.main, base, &mut tally);
        for (label, closure) in &prepared.closures {
            dump(
                &format!("{qref:?} closure {label:?}"),
                closure,
                base,
                &mut tally,
            );
        }
    }
    println!("slot hit rate: {}/{} concrete", tally.concrete, tally.total);
}
