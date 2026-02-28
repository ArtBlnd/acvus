use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::{env, fs, process};

use acvus_mir::printer::dump;
use acvus_mir::ty::Ty;
use serde::Deserialize;

#[derive(Deserialize, Default)]
struct Context {
    #[serde(default)]
    storage: HashMap<String, TypeDef>,
    #[serde(default)]
    extern_fns: HashMap<String, FnDef>,
}

#[derive(Deserialize)]
struct FnDef {
    params: Vec<TypeDef>,
    ret: TypeDef,
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
    Object(BTreeMap<std::string::String, TypeDef>),
}

impl TypeDef {
    fn to_ty(&self) -> Ty {
        match self {
            TypeDef::Int => Ty::Int,
            TypeDef::Float => Ty::Float,
            TypeDef::String => Ty::String,
            TypeDef::Bool => Ty::Bool,
            TypeDef::Unit => Ty::Unit,
            TypeDef::Range => Ty::Range,
            TypeDef::List(inner) => Ty::List(Box::new(inner.to_ty())),
            TypeDef::Object(fields) => {
                Ty::Object(fields.iter().map(|(k, v)| (k.clone(), v.to_ty())).collect())
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("Usage: acvus-mir-cli <template-file> [context.json]");
        eprintln!();
        eprintln!("  template-file   Path to .acvus template (or - for stdin)");
        eprintln!("  context.json    Optional JSON with storage types and extern fns");
        eprintln!();
        eprintln!("Context JSON format:");
        eprintln!(r#"  {{"#);
        eprintln!(r#"    "storage": {{ "name": "string", "count": "int" }},"#);
        eprintln!(r#"    "extern_fns": {{ "fetch": {{ "params": ["int"], "ret": "string" }} }}"#);
        eprintln!(r#"  }}"#);
        process::exit(1);
    }

    // Read template.
    let source = if args[1] == "-" {
        let mut buf = std::string::String::new();
        std::io::stdin().read_to_string(&mut buf).unwrap_or_else(|e| {
            eprintln!("error: failed to read stdin: {e}");
            process::exit(1);
        });
        buf
    } else {
        fs::read_to_string(&args[1]).unwrap_or_else(|e| {
            eprintln!("error: failed to read {}: {e}", args[1]);
            process::exit(1);
        })
    };

    // Read context.
    let ctx: Context = if let Some(ctx_path) = args.get(2) {
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

    let storage_types: HashMap<String, Ty> = ctx
        .storage
        .iter()
        .map(|(k, v)| (k.clone(), v.to_ty()))
        .collect();

    let extern_fns: HashMap<String, (Vec<Ty>, Ty)> = ctx
        .extern_fns
        .iter()
        .map(|(k, v)| {
            let params: Vec<Ty> = v.params.iter().map(|p| p.to_ty()).collect();
            let ret = v.ret.to_ty();
            (k.clone(), (params, ret))
        })
        .collect();

    // Parse.
    let template = match acvus_ast::parse(&source) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("parse error: {e}");
            process::exit(1);
        }
    };

    // Compile.
    match acvus_mir::compile(&template, storage_types, extern_fns) {
        Ok((module, _hints)) => {
            println!("{}", dump(&module));
        }
        Err(errors) => {
            for e in &errors {
                eprintln!(
                    "error [{}..{}]: {}",
                    e.span.start, e.span.end, e
                );
            }
            process::exit(1);
        }
    }
}
