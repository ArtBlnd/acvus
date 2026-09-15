//! `acvus`: the script runner (RFC-0031).

mod compile;
mod context;
mod json;
mod llm;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use acvus_ast::report::{Report, Severity};
use acvus_interpreter::{
    ContextWrite, Executor, InMemoryContext, Interpreter, InterpreterContext, SequentialExecutor,
    TokioExecutor,
};
use acvus_utils::Interner;

use crate::compile::{Compiled, Diagnostic, Mode};

const EXIT_COMPILE: u8 = 1;
const EXIT_RUN: u8 = 2;
const EXIT_USAGE: u8 = 64;

const USAGE: &str = "\
usage: acvus run   <file.acvus|file.acvt> [--context ctx.json] [--commit] [--llm] [--parallel]
       acvus run   -e <expr>              [--context ctx.json] [--llm] [--parallel]
       acvus check <file>                 [--context ctx.json]
       acvus mir   <file>                 [--context ctx.json]

  .acvus is script mode, .acvt is a template; -e runs one expression.
  --context  a JSON object: each key is a context, its type the value's type
  --commit   write the contexts back to the context file after the run
  --llm      register the LLM providers (keys from the environment)
  --parallel run spawned calls on the tokio executor";

enum Command {
    Run,
    Check,
    Mir,
}

struct Args {
    command: Command,
    source: Source,
    context: Option<PathBuf>,
    commit: bool,
    llm: bool,
    parallel: bool,
}

enum Source {
    File(PathBuf),
    Expr(String),
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut it = argv.iter();
    let command = match it.next().map(String::as_str) {
        Some("run") => Command::Run,
        Some("check") => Command::Check,
        Some("mir") => Command::Mir,
        Some(other) => return Err(format!("unknown command `{other}`")),
        None => return Err("no command".to_string()),
    };
    let mut source = None;
    let mut context = None;
    let mut commit = false;
    let mut llm = false;
    let mut parallel = false;
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-e" => {
                let expr = it.next().ok_or("-e takes an expression")?;
                source = Some(Source::Expr(expr.clone()));
            }
            "--context" => {
                let path = it.next().ok_or("--context takes a file")?;
                context = Some(PathBuf::from(path));
            }
            "--commit" => commit = true,
            "--llm" => llm = true,
            "--parallel" => parallel = true,
            flag if flag.starts_with('-') => return Err(format!("unknown flag `{flag}`")),
            path => {
                if source.is_some() {
                    return Err("one source at a time".to_string());
                }
                source = Some(Source::File(PathBuf::from(path)));
            }
        }
    }
    let source = source.ok_or("no source")?;
    if commit && context.is_none() {
        return Err("--commit needs --context".to_string());
    }
    Ok(Args {
        command,
        source,
        context,
        commit,
        llm,
        parallel,
    })
}

fn mode_of(path: &Path) -> Result<Mode, String> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("acvus") => Ok(Mode::Script),
        Some("acvt") => Ok(Mode::Template),
        _ => Err(format!(
            "{}: a source is .acvus (script) or .acvt (template)",
            path.display()
        )),
    }
}

fn report(path: &str, source: &str, diagnostics: &[Diagnostic]) {
    for d in diagnostics {
        eprint!(
            "{}",
            Report {
                severity: Severity::Error,
                message: d.message.clone(),
                path,
                source,
                span: d.span,
            }
        );
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&argv) {
        Ok(args) => args,
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let (path, source, mode) = match &args.source {
        Source::File(p) => {
            let mode = match mode_of(p) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(EXIT_USAGE);
                }
            };
            match std::fs::read_to_string(p) {
                Ok(s) => (p.display().to_string(), s, mode),
                Err(e) => {
                    eprintln!("error: {}: {e}", p.display());
                    return ExitCode::from(EXIT_USAGE);
                }
            }
        }
        Source::Expr(e) => ("<expr>".to_string(), e.clone(), Mode::Expr),
    };
    let interner = Interner::new();
    let loaded = match &args.context {
        Some(p) => match context::load(&interner, p) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: {}: {e}", p.display());
                return ExitCode::from(EXIT_USAGE);
            }
        },
        None => context::Loaded::default(),
    };
    let registries = {
        let mut r = acvus_ext::std_registries();
        r.push(acvus_ext_net::http_registry());
        if args.llm {
            r.extend(llm::registries());
        }
        r
    };
    let compiled = match compile::compile(&interner, &source, mode, &loaded.types, registries) {
        Ok(c) => c,
        Err(diagnostics) => {
            report(&path, &source, &diagnostics);
            return ExitCode::from(EXIT_COMPILE);
        }
    };
    match args.command {
        Command::Check => ExitCode::SUCCESS,
        Command::Mir => {
            print!("{}", compiled.mir_dump(&interner));
            ExitCode::SUCCESS
        }
        Command::Run => run(&interner, &path, &source, compiled, loaded, &args).await,
    }
}

async fn run(
    interner: &Interner,
    path: &str,
    source: &str,
    compiled: Compiled,
    loaded: context::Loaded,
    args: &Args,
) -> ExitCode {
    let executor: Arc<dyn Executor> = if args.parallel {
        Arc::new(TokioExecutor)
    } else {
        Arc::new(SequentialExecutor)
    };
    let Compiled {
        entry,
        functions,
        fn_types,
        context_names,
        ret_ty,
        ..
    } = compiled;
    let shared = InterpreterContext::new(interner, functions, executor)
        .with_fn_types(fn_types)
        .with_context_names(context_names);
    let mut interp = Interpreter::new(shared, entry, InMemoryContext::new(loaded.snapshot));
    let value = match interp.execute().await {
        Ok(v) => v,
        Err(e) => {
            eprint!(
                "{}",
                Report {
                    severity: Severity::Error,
                    message: e.to_string(),
                    path,
                    source,
                    span: e.span,
                }
            );
            return ExitCode::from(EXIT_RUN);
        }
    };
    // The body commits every context it fetched (RFC-0025); a commit of
    // the value that was loaded is not a write.
    let mut writes: Vec<ContextWrite> = Vec::new();
    for w in interp.take_writes() {
        let Some(ty) = loaded.types.get(&interner.intern(&w.key)) else {
            eprintln!("error: @{}: written but not in the context file", w.key);
            return ExitCode::from(EXIT_RUN);
        };
        let json = json::of(interner, ty, &w.value);
        if loaded.raw.get(&w.key) == Some(&json) {
            continue;
        }
        eprintln!("write @{} = {json}", w.key);
        writes.push(w);
    }
    if args.commit
        && let Some(p) = &args.context
        && let Err(e) = context::commit(interner, p, &loaded.types, &writes)
    {
        eprintln!("error: {}: {e}", p.display());
        return ExitCode::from(EXIT_RUN);
    }
    match &ret_ty {
        acvus_mir::ty::Ty::String => {
            // SAFETY: the entry's return type is String.
            println!("{}", unsafe { value.as_str() });
        }
        acvus_mir::ty::Ty::Unit => {}
        ty => println!("{}", json::of(interner, ty, &value)),
    }
    ExitCode::SUCCESS
}
