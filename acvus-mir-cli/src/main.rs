use std::io::Read;
use std::{env, fs, process};

use acvus_mir::graph::types::*;
use acvus_mir::graph::{extract, infer, lower as graph_lower};
use acvus_mir::printer::dump;
use acvus_mir::ty::{Effect, Param, Ty};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;
use serde::Deserialize;

#[derive(Deserialize, Default)]
struct Context {
    #[serde(default)]
    context: FxHashMap<String, TypeDef>,
    #[serde(default)]
    extern_fns: FxHashMap<String, FnDef>,
}

#[derive(Deserialize)]
struct FnDef {
    params: Vec<TypeDef>,
    ret: TypeDef,
    #[serde(default)]
    effectful: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum TypeDef {
    Int,
    Float,
    String,
    Bool,
    Unit,
    Range,
    List(Box<TypeDef>),
    Object(FxHashMap<String, TypeDef>),
}

impl TypeDef {
    fn to_ty(&self, interner: &Interner) -> Ty {
        match self {
            TypeDef::Int => Ty::Int,
            TypeDef::Float => Ty::Float,
            TypeDef::String => Ty::String,
            TypeDef::Bool => Ty::Bool,
            TypeDef::Unit => Ty::Unit,
            TypeDef::Range => Ty::Range,
            TypeDef::List(inner) => Ty::List(Box::new(inner.to_ty(interner))),
            TypeDef::Object(fields) => Ty::Object(
                fields
                    .iter()
                    .map(|(k, v)| (interner.intern(k), v.to_ty(interner)))
                    .collect(),
            ),
        }
    }
}

fn main() {
    let interner = Interner::new();
    let raw_args: Vec<String> = env::args().collect();

    let is_script = raw_args.iter().any(|a| a == "--script");
    let args: Vec<&String> = raw_args.iter().filter(|a| !a.starts_with("--")).collect();

    if args.len() < 2 || raw_args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("Usage: acvus-mir-cli [--script] <source-file> [context.json]");
        eprintln!();
        eprintln!("  --script        Treat input as script (default: template)");
        eprintln!("  source-file     Path to source file (or - for stdin)");
        eprintln!("  context.json    Optional JSON with context types and extern fns");
        eprintln!();
        eprintln!("Context JSON format:");
        eprintln!(r#"  {{"#);
        eprintln!(r#"    "context": {{ "name": "string", "count": "int" }},"#);
        eprintln!(
            r#"    "extern_fns": {{ "fetch": {{ "params": ["int"], "ret": "string", "effectful": true }} }}"#
        );
        eprintln!(r#"  }}"#);
        process::exit(1);
    }

    // Read source.
    let source = if *args[1] == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .unwrap_or_else(|e| {
                eprintln!("error: failed to read stdin: {e}");
                process::exit(1);
            });
        buf
    } else {
        fs::read_to_string(args[1]).unwrap_or_else(|e| {
            eprintln!("error: failed to read {}: {e}", args[1]);
            process::exit(1);
        })
    };

    // Read context JSON.
    let ctx: Context = if let Some(&ctx_path) = args.get(2) {
        let json = fs::read_to_string(ctx_path).unwrap_or_else(|e| {
            eprintln!("error: failed to read {ctx_path}: {e}");
            process::exit(1);
        });
        serde_json::from_str(&json).unwrap_or_else(|e| {
            eprintln!("error: failed to parse context JSON: {e}");
            process::exit(1);
        })
    } else {
        Context::default()
    };

    // Build context list.
    let mut contexts: Vec<acvus_mir::graph::types::Context> = ctx
        .context
        .iter()
        .map(|(name, def)| acvus_mir::graph::types::Context {
            qref: QualifiedRef::root(interner.intern(name)),
            constraint: Constraint::Exact(def.to_ty(&interner)),
        })
        .collect();

    // Extern fns are also contexts with Ty::Fn.
    for (name, def) in &ctx.extern_fns {
        let params: Vec<Param> = def
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| Param::new(interner.intern(&format!("_{i}")), p.to_ty(&interner)))
            .collect();
        let fn_ty = Ty::Fn {
            params,
            ret: Box::new(def.ret.to_ty(&interner)),
            captures: vec![],
            effect: if def.effectful {
                Effect::io()
            } else {
                Effect::pure()
            },
        };
        contexts.push(acvus_mir::graph::types::Context {
            qref: QualifiedRef::root(interner.intern(name)),
            constraint: Constraint::Exact(fn_ty),
        });
    }

    let parsed_ast = if is_script {
        ParsedAst::Script(
            acvus_ast::parse_script(&interner, &source)
                .unwrap_or_else(|e| {
                    eprintln!("error: parse failed: {e:?}");
                    process::exit(1);
                }),
        )
    } else {
        ParsedAst::Template(
            acvus_ast::parse(&interner, &source)
                .unwrap_or_else(|e| {
                    eprintln!("error: parse failed: {e:?}");
                    process::exit(1);
                }),
        )
    };

    let fn_qref = QualifiedRef::root(interner.intern("main"));
    let mut functions = acvus_mir::builtins::standard_builtins(&interner);
    let mut type_registry = acvus_mir::ty::TypeRegistry::new();
    let std_regs = acvus_ext::std_registries(&interner, &mut type_registry);
    for registry in std_regs {
        let registered = registry.register(&interner);
        functions.extend(registered.functions);
    }
    functions.push(Function {
        qref: fn_qref,
        kind: FnKind::Local(parsed_ast),
        constraint: FnConstraint {
            signature: None,
            output: Constraint::Inferred,
            effect: None,
        },
    });

    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
    };

    // Run pipeline: extract → infer → lower.
    let ext = extract::extract(&interner, &graph);
    let inf = infer::infer(&interner, &graph, &ext, &FxHashMap::default(), Freeze::new(type_registry));
    let result = graph_lower::lower(&interner, &graph, &ext, &inf);
    if result.has_errors() {
        for le in &result.errors {
            for e in &le.errors {
                eprintln!(
                    "error [{}..{}]: {}",
                    e.span.start,
                    e.span.end,
                    e.display(&interner)
                );
            }
        }
        process::exit(1);
    }

    match result.module(fn_qref) {
        Some(module) => {
            println!("{}", dump(&interner, module));
        }
        None => {
            eprintln!("error: no module produced for main function");
            process::exit(1);
        }
    }
}
