//! `acvus`: the script runner (RFC-0031).

mod compile;
mod context;
mod json;
mod llm;
mod oplist;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use acvus_ast::Span;
use acvus_ast::report::{LineIndex, Report, Severity};
use acvus_extern::Registry;
use acvus_interpreter::{
    AcvusRuntime, Composite, ContextWrite, DirStore, Executor, InMemoryContext, Interpreter,
    InterpreterContext, Kind, Mode as SpaceMode, RuntimeContext, SequentialExecutor, Space,
    SpacePage, TokioExecutor, Value, hex,
};
use acvus_utils::Interner;

use acvus_mir::graph::optimize::Opt;

use crate::compile::{CompileTimes, Compiled, Diagnostic, Mode, Stopwatch, Timed};

const EXIT_COMPILE: u8 = 1;
const EXIT_RUN: u8 = 2;
const EXIT_USAGE: u8 = 64;

const USAGE: &str = "\
usage: acvus run   <file.acvus|file.acvt> [--context ctx.json] [--commit] [--llm] [--parallel] [--opt L] [--time]
       acvus run   -e <expr>              [--context ctx.json] [--llm] [--parallel] [--opt L] [--time]
       acvus check <file>                 [--context ctx.json] [--json] [--opt L] [--time]
       acvus mir   <file>                 [--context ctx.json] [--json] [--opt L] [--time]
       acvus ops   <file>                 [--context ctx.json] [--llm] [--json] [--opt L] [--time]
       acvus space <dir>

  .acvus is script mode, .acvt is a template; -e runs one expression.
  --context  a JSON object: each key is a context, its type the value's type
  --commit   write the contexts back to the context file after the run
  --space    a directory holding contexts (RFC-0033): the run fetches them
             from it and commits its changes to it; --context seeds it
  --json     stdout is JSON: the diagnostics as an array of
             {severity, message, path, line, col, span}, and, where `ops`
             has a listing to print, the listing
  --llm      register the LLM providers (keys from the environment)
  --parallel run spawned calls on the tokio executor
  --opt      full (the default) runs every optimization; none runs only what
             a program needs to reach the machine, and both refuse the same
             programs
  --time     after the output, one `time:` line on stderr per stage this
             command ran -- compile, with the level it compiled at and its
             parse, typeck, lower and optimize; prepare; run -- in
             milliseconds; under --json a trailing {\"time\": ...} object on
             stdout instead";

enum Command {
    Run,
    Check,
    Mir,
    Ops,
    Space,
}

struct Args {
    command: Command,
    source: Source,
    context: Option<PathBuf>,
    commit: bool,
    json: bool,
    llm: bool,
    parallel: bool,
    time: bool,
    opt: Opt,
    space: Option<PathBuf>,
}

/// The spelling `--opt` takes and the `time:` line reports.
fn level(opt: Opt) -> &'static str {
    match opt {
        Opt::None => "none",
        Opt::Full => "full",
    }
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
        Some("ops") => Command::Ops,
        Some("space") => Command::Space,
        Some(other) => return Err(format!("unknown command `{other}`")),
        None => return Err("no command".to_string()),
    };
    let mut source = None;
    let mut context = None;
    let mut commit = false;
    let mut json = false;
    let mut llm = false;
    let mut parallel = false;
    let mut time = false;
    let mut opt = Opt::Full;
    let mut space = None;
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
            "--json" => json = true,
            "--llm" => llm = true,
            "--parallel" => parallel = true,
            "--time" => time = true,
            "--opt" => {
                opt = match it.next().map(String::as_str) {
                    Some("none") => Opt::None,
                    Some("full") => Opt::Full,
                    Some(other) => return Err(format!("--opt takes none or full, not `{other}`")),
                    None => return Err("--opt takes none or full".to_string()),
                }
            }
            "--space" => {
                let dir = it.next().ok_or("--space takes a directory")?;
                space = Some(PathBuf::from(dir));
            }
            flag if flag.starts_with('-') => return Err(format!("unknown flag `{flag}`")),
            path => {
                if source.is_some() {
                    return Err("one source at a time".to_string());
                }
                source = Some(Source::File(PathBuf::from(path)));
            }
        }
    }
    if json && matches!(command, Command::Run | Command::Space) {
        return Err("--json is for check, mir and ops".to_string());
    }
    if time && matches!(command, Command::Space) {
        return Err("--time is for run, check, mir and ops".to_string());
    }
    if matches!(command, Command::Space) {
        let Some(Source::File(dir)) = source else {
            return Err("space takes a directory".to_string());
        };
        return Ok(Args {
            command,
            source: Source::Expr(String::new()),
            context: None,
            commit: false,
            json: false,
            llm: false,
            parallel: false,
            time: false,
            opt: Opt::Full,
            space: Some(dir),
        });
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
        json,
        llm,
        parallel,
        time,
        opt,
        space,
    })
}

fn open_space(interner: &Interner, dir: &Path) -> Result<Arc<Space>, String> {
    let store = DirStore::open(dir, interner).map_err(|e| e.to_string())?;
    Ok(Arc::new(Space::over(
        SpaceMode::Log {
            checkpoint_every: 64,
        },
        Box::new(store),
    )))
}

fn list_space(interner: &Interner, dir: &Path) -> ExitCode {
    let space = match open_space(interner, dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    match space.identities() {
        Ok(ids) => {
            for (id, ty) in ids {
                let head = hex(&space
                    .head(&id)
                    .expect("identities are listed by their heads"));
                println!("@{id}: {} = {head}", ty.display(interner));
            }
            println!("{} nodes", space.node_count());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(EXIT_RUN)
        }
    }
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

/// Where a diagnostic goes and in what shape: `error: …` with the span's
/// line, source line and caret on stderr, or, under `--json`, the whole set
/// as one array on stdout.
enum Rendering {
    Text,
    Json,
}

impl Rendering {
    fn of(json: bool) -> Self {
        match json {
            true => Rendering::Json,
            false => Rendering::Text,
        }
    }

    fn report(&self, path: &str, source: &str, diagnostics: &[Diagnostic]) {
        match self {
            Rendering::Text => {
                for d in diagnostics {
                    eprint!(
                        "{}",
                        Report {
                            severity: Severity::Error,
                            message: d.message.clone(),
                            primary: d.primary.clone(),
                            path,
                            source,
                            span: d.span,
                            labels: d.labels.clone(),
                        }
                    );
                }
            }
            Rendering::Json => {
                let index = LineIndex::new(source);
                let array: Vec<serde_json::Value> = diagnostics
                    .iter()
                    .map(|d| {
                        let at = |span: Span| index.line_col(span.start.min(source.len()));
                        let labels: Vec<serde_json::Value> = d
                            .labels
                            .iter()
                            .map(|l| {
                                serde_json::json!({
                                    "line": l.span.map(|s| at(s).line),
                                    "col": l.span.map(|s| at(s).col),
                                    "span": l.span.map(|s| [s.start, s.end]),
                                    "text": l.text,
                                })
                            })
                            .collect();
                        serde_json::json!({
                            "severity": Severity::Error.to_string(),
                            "message": d.message,
                            "primary": d.primary,
                            "path": path,
                            "line": d.span.map(|s| at(s).line),
                            "col": d.span.map(|s| at(s).col),
                            "span": d.span.map(|s| [s.start, s.end]),
                            "labels": labels,
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string(&array).expect("a diagnostic array serializes")
                );
            }
        }
    }
}

/// `run` holds the machine alone: a script's own `print` happens inside
/// that number, and reading or committing the context file happens outside
/// it.
///
/// A compile that failed reports no time at all. Its diagnostics are the
/// answer to what the command did, a list of the stages that got as far as
/// running would be a second answer to the same question, and every failing
/// command keeps the output it had.
struct Timings {
    compile: Duration,
    stages: CompileTimes,
    prepare: Option<Duration>,
    run: Option<Duration>,
}

impl Timings {
    fn of(compile: Option<Duration>, stages: Option<CompileTimes>) -> Option<Self> {
        Some(Timings {
            compile: compile?,
            stages: stages?,
            prepare: None,
            run: None,
        })
    }

    fn report(&self, rendering: &Rendering) {
        let CompileTimes {
            opt,
            parse,
            typeck,
            lower,
            optimize,
        } = self.stages;
        match rendering {
            Rendering::Text => {
                eprintln!(
                    "time: compile {:.3} ms at opt {} (parse {:.3}, typeck {:.3}, lower {:.3}, optimize {:.3})",
                    ms(self.compile),
                    level(opt),
                    ms(parse),
                    ms(typeck),
                    ms(lower),
                    ms(optimize)
                );
                if let Some(prepare) = self.prepare {
                    eprintln!("time: prepare {:.3} ms", ms(prepare));
                }
                if let Some(run) = self.run {
                    eprintln!("time: run     {:.3} ms", ms(run));
                }
            }
            Rendering::Json => {
                let mut time = serde_json::Map::new();
                time.insert(
                    "compile".to_string(),
                    serde_json::json!({
                        "opt": level(opt),
                        "total": ms(self.compile),
                        "parse": ms(parse),
                        "typeck": ms(typeck),
                        "lower": ms(lower),
                        "optimize": ms(optimize),
                    }),
                );
                if let Some(prepare) = self.prepare {
                    time.insert("prepare".to_string(), ms(prepare).into());
                }
                if let Some(run) = self.run {
                    time.insert("run".to_string(), ms(run).into());
                }
                println!("{}", serde_json::json!({ "time": time }));
            }
        }
    }
}

/// What a script run by this CLI can call without a flag. `std_registries`
/// is the language's own surface; the four beside it are the resources this
/// host decides to hand a script. The `--llm` providers are not here
/// because they are the one set a flag guards.
///
/// `acvus-cli` is a binary, so `tests/cli.rs` cannot call this and rebuilds
/// the same list to compare `ops` against the interpreter's own walk. The
/// two lists are one contract: a registry added here is added there.
fn cli_registries() -> Vec<Registry<AcvusRuntime>> {
    let mut registries = acvus_ext::std_registries();
    registries.push(acvus_ext::regex_registry());
    registries.push(acvus_ext::datetime_registry());
    registries.push(acvus_ext::io_registry());
    registries.push(acvus_ext_net::http_registry());
    registries
}

/// Milliseconds to the microsecond, so the text and the JSON carry the same
/// number.
fn ms(duration: Duration) -> f64 {
    duration.as_micros() as f64 / 1_000.0
}

/// A run-time failure is a panic, and this is where the process stops
/// being a Rust program and becomes a script runner: the operation's
/// message on one `error:` line, the same form every other failure here
/// prints, and `EXIT_RUN`. The default hook's `thread 'main' panicked at
/// acvus-interpreter/src/...` names this compiler's source, which is not
/// where a script author looks; `RUST_BACKTRACE` puts it back.
fn main() -> ExitCode {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::panic::set_hook(Box::new(|_| {}));
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("a multi-thread tokio runtime");
    match catch_unwind(AssertUnwindSafe(|| runtime.block_on(cli()))) {
        Ok(code) => code,
        Err(panic) => {
            eprintln!("error: {}", panic_message(panic.as_ref()));
            ExitCode::from(EXIT_RUN)
        }
    }
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> &str {
    if let Some(message) = panic.downcast_ref::<&'static str>() {
        return message;
    }
    match panic.downcast_ref::<String>() {
        Some(message) => message,
        None => "the run panicked with a payload that is not a message",
    }
}

async fn cli() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&argv) {
        Ok(args) => args,
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let interner = Interner::new();
    if let Command::Space = args.command {
        return list_space(
            &interner,
            args.space.as_deref().expect("space takes a directory"),
        );
    }
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
    let mut loaded = match &args.context {
        Some(p) => match context::load(&interner, p) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: {}: {e}", p.display());
                return ExitCode::from(EXIT_USAGE);
            }
        },
        None => context::Loaded::default(),
    };
    let space = match &args.space {
        Some(dir) => match open_space(&interner, dir) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(EXIT_USAGE);
            }
        },
        None => None,
    };
    if let Some(space) = &space {
        match space.identities() {
            Ok(ids) => {
                for (id, ty) in ids {
                    loaded.types.entry(interner.intern(&id)).or_insert(ty);
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(EXIT_USAGE);
            }
        }
    }
    let registries = {
        let mut r = cli_registries();
        if args.llm {
            r.extend(llm::registries());
        }
        r
    };
    let rendering = Rendering::of(args.json);
    let timed = Timed::of(args.time);
    let watch = Stopwatch::start(timed);
    let (checked, stages) = match compile::check(
        &interner,
        &source,
        mode,
        &loaded.types,
        registries,
        timed,
        args.opt,
    ) {
        Ok(c) => c,
        Err(diagnostics) => {
            rendering.report(&path, &source, &diagnostics);
            return ExitCode::from(EXIT_COMPILE);
        }
    };
    let mut timings = Timings::of(watch.stop(), stages);
    match args.command {
        Command::Check => {
            rendering.report(&path, &source, &[]);
            if let Some(timings) = &timings {
                timings.report(&rendering);
            }
            ExitCode::SUCCESS
        }
        Command::Mir => {
            match rendering {
                Rendering::Text => print!("{}", checked.mir_dump()),
                Rendering::Json => rendering.report(&path, &source, &[]),
            }
            if let Some(timings) = &timings {
                timings.report(&rendering);
            }
            ExitCode::SUCCESS
        }
        Command::Ops => {
            let form = match args.json {
                true => oplist::Form::Json,
                false => oplist::Form::Text,
            };
            let watch = Stopwatch::start(timed);
            let compiled = checked.prepare(&interner);
            if let Some(timings) = &mut timings {
                timings.prepare = watch.stop();
            }

            match oplist::dump(compiled.entry_prepared(), form) {
                Ok(text) => {
                    print!("{text}");
                    if let Some(timings) = &timings {
                        timings.report(&rendering);
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: the listing does not serialize: {e}");
                    ExitCode::from(EXIT_RUN)
                }
            }
        }
        Command::Run => {
            let watch = Stopwatch::start(timed);
            let compiled = checked.prepare(&interner);
            if let Some(timings) = &mut timings {
                timings.prepare = watch.stop();
            }

            run(
                &interner, compiled, loaded, space, &args, &rendering, timings,
            )
            .await
        }
        Command::Space => unreachable!("handled before compiling"),
    }
}

async fn run(
    interner: &Interner,
    compiled: Compiled,
    loaded: context::Loaded,
    space: Option<Arc<Space>>,
    args: &Args,
    rendering: &Rendering,
    mut timings: Option<Timings>,
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
        space: hooks,
        ..
    } = compiled;
    let shared = InterpreterContext::new(interner, functions, executor)
        .with_fn_types(fn_types)
        .with_context_names(context_names)
        .with_space(hooks);
    let snapshot = loaded.snapshot;
    let (page, mut interp): (Option<Arc<SpacePage>>, Interpreter) = match &space {
        Some(space) => {
            let seed = snapshot
                .into_iter()
                .map(|(k, v)| {
                    let ty = loaded.types[&interner.intern(&k)].clone();
                    (k, (ty, v))
                })
                .collect();
            let page = match SpacePage::new(Arc::clone(space), seed) {
                Ok(p) => Arc::new(p),
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(EXIT_RUN);
                }
            };
            let interp =
                Interpreter::on_page(shared, entry, Arc::clone(&page) as Arc<dyn RuntimeContext>);
            (Some(page), interp)
        }
        None => (
            None,
            Interpreter::new(shared, entry, InMemoryContext::new(snapshot)),
        ),
    };
    let watch = Stopwatch::start(Timed::of(args.time));
    let value = interp.execute().await;
    if let Some(timings) = &mut timings {
        timings.run = watch.stop();
    }

    if let Some(page) = &page {
        match page.commit(interp.runtime()) {
            Ok(heads) => {
                for (id, head) in heads {
                    eprintln!("commit @{id} = {}", hex(&head));
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(EXIT_RUN);
            }
        }
    }
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
    print_result(interner, &value);
    if let Some(timings) = &timings {
        timings.report(rendering);
    }
    ExitCode::SUCCESS
}

/// RFC-0054: this host declares `!`, so it has no type for what comes back
/// and reads the value by kind.
fn print_result(interner: &Interner, value: &Value) {
    match value.composite() {
        // SAFETY, both arms: the vtable is the runtime's witness of the type
        // behind the pointer.
        Some(Composite::String) => println!("{}", unsafe { value.as_str() }),
        Some(Composite::Tuple) if unsafe { value.as_tuple() }.is_empty() => {}
        _ if value.kind() == Kind::Unit => {}
        _ => println!("{}", json::by_kind(interner, value)),
    }
}
