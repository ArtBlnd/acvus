//! `acvus`: the script runner (RFC-0031).

mod compile;
mod ctl;
mod json;
mod location;
mod lsp_host;
mod oplist;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use acvus_ast::Span;
use acvus_ast::report::{LineIndex, Report, Severity};
use acvus_extern::Registry;
use acvus_interpreter::{
    AcvusRuntime, CompileTimes, Composite, Executor, HostError, InputListing, Kind, MemoryStorage,
    Page, Part, Program, SequentialExecutor, Space, SpaceStorage, Storage, TokioExecutor,
    UntypedEntry, Value, hex,
};
use acvus_utils::Interner;

use acvus_mir::graph::optimize::Opt;

use crate::compile::{Binding, Diagnostic, Mode, Refused, Role, Stopwatch, Timed, Unit};
use crate::ctl::{
    ConfigFile, CtlError, Defaults, Layers, OptLevel, Parallel, ResolvedSpace, SpaceChoice,
    SpaceSource, Timing,
};
use crate::location::{EXPR_ENTRY, ScriptKind};

const EXIT_COMPILE: u8 = 1;
const EXIT_RUN: u8 = 2;
const EXIT_USAGE: u8 = 64;
/// The Language Server Protocol has a server that exits without a prior
/// `shutdown` exit with 1; `acvus lsp` ends every other failure the same way.
const EXIT_LSP_FAILED: u8 = 1;

/// The entry a lone file or expression compiles to, with no space around it.
const LONE_ENTRY: &str = "main";

const USAGE: &str = "\
usage: acvus run   <file.acvus|file.acvt|script> [name=literal]... [--space S] [--parallel[=P]] [--opt L] [--time[=T]]
       acvus run   -e <expr>                     [name=literal]... [--space S] [--parallel[=P]] [--opt L] [--time[=T]]
       acvus check <file|script> | -e <expr>     [name=literal]... [--space S] [--json] [--opt L] [--time[=T]]
       acvus mir   <file|script> | -e <expr>     [name=literal]... [--space S] [--json] [--opt L] [--time[=T]]
       acvus ops   <file|script> | -e <expr>     [name=literal]... [--space S] [--json] [--opt L] [--time[=T]]
       acvus ctl   ...                           (`acvus ctl` lists its commands)
       acvus lsp

  .acvus is script mode, .acvt is a template; -e runs one expression.
  name=literal  binds the input `$name` to the value the literal writes, in
             the script's own syntax: 10, '\"jun\"', Some(1), [1, 2],
             { a: 1, }, a variant; the code it decides against is gone, and
             the inputs it alone read are no longer required
  --space    a space the active ctl context maps to a location: its scripts
             and inits compile as one graph, the positional names the script
             to run, -e compiles beside them, and the run commits its writes
             to it. A context the run fetches and the space lacks gets its
             first value from its init, which `acvus ctl space init` stores.
             Without --space the nearest .acvus/ above the working directory
             names the space; with neither, a source that names no context
             runs alone
  lsp        serve the Language Server Protocol on stdin and stdout, over the
             sources under each folder the client serves
  --json     stdout is JSON: the diagnostics as an array of
             {severity, message, path, line, col, span}, and, where `ops`
             has a listing to print, the listing
  --parallel spawned calls run on tokio; --parallel=sequential (the
             default) runs them in order
  --opt      full (the default) runs every optimization; none runs only what
             a program needs to reach the machine, and both refuse the same
             programs
  --time     after the output, one `time:` line on stderr per stage this
             command ran -- compile, with the level it compiled at and its
             parse, typeck, lower and optimize; prepare; run -- in
             milliseconds; under --json a trailing {\"time\": ...} object on
             stdout instead. --time=off (the default) prints none
  A flag overrides `acvus ctl set` on the space, which overrides the ctl
  context's, which overrides the default.";

enum Command {
    Run,
    Check,
    Mir,
    Ops,
}

struct Args {
    command: Command,
    source: Source,
    bindings: Vec<Binding>,
    json: bool,
    flags: Defaults,
    space: Option<String>,
}

fn level(opt: Opt) -> &'static str {
    match opt {
        Opt::None => "none",
        Opt::Full => "full",
    }
}

/// Without a space the positional is a file; with one it is the name of a
/// script the space holds.
enum Source {
    Positional(String),
    Expr(String),
}

enum Invocation {
    Lsp,
    Ctl(Vec<String>),
    Compile(Args),
}

fn parse_args(argv: &[String]) -> Result<Invocation, String> {
    let mut it = argv.iter();
    let command = match it.next().map(String::as_str) {
        Some("lsp") => {
            return match it.next() {
                Some(extra) => Err(format!("lsp takes no arguments, not `{extra}`")),
                None => Ok(Invocation::Lsp),
            };
        }
        Some("ctl") => return Ok(Invocation::Ctl(it.cloned().collect())),
        Some("run") => Command::Run,
        Some("check") => Command::Check,
        Some("mir") => Command::Mir,
        Some("ops") => Command::Ops,
        Some(other) => return Err(format!("unknown command `{other}`")),
        None => return Err("no command".to_string()),
    };
    let mut source = None;
    let mut bindings: Vec<Binding> = Vec::new();
    let mut json = false;
    let mut flags = Defaults::default();
    let mut space = None;
    while let Some(arg) = it.next() {
        if let Some(set) = ctl::run_flag(&mut flags, arg, &mut || it.next().cloned()) {
            set?;
            continue;
        }
        match arg.as_str() {
            "-e" => {
                let expr = it.next().ok_or("-e takes an expression")?;
                if source.is_some() {
                    return Err("one source at a time".to_string());
                }
                source = Some(Source::Expr(expr.clone()));
            }
            "--json" => json = true,
            "--space" => {
                let name = it.next().ok_or("--space takes a space name")?;
                space = Some(name.clone());
            }
            flag if flag.starts_with('-') => return Err(format!("unknown flag `{flag}`")),
            word => match binding(word) {
                Some(bound) => bindings.push(bound),
                None => {
                    if source.is_some() {
                        return Err("one source at a time".to_string());
                    }
                    source = Some(Source::Positional(word.to_string()));
                }
            },
        }
    }
    if json && matches!(command, Command::Run) {
        return Err("--json is for check, mir and ops".to_string());
    }
    let source = source.ok_or("no source")?;
    Ok(Invocation::Compile(Args {
        command,
        source,
        bindings,
        json,
        flags,
        space,
    }))
}

/// `name=<literal>`, where `name` is an input's name; any other word is a
/// source.
fn binding(word: &str) -> Option<Binding> {
    let (name, text) = word.split_once('=')?;
    let mut chars = name.chars();
    let first = chars.next()?;
    let is_name = (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    is_name.then(|| Binding {
        name: name.to_string(),
        text: text.to_string(),
    })
}

fn mode_of(path: &Path) -> Result<Mode, String> {
    match ScriptKind::of_path(path) {
        Some(kind) => Ok(kind.mode()),
        None => Err(format!(
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

    /// What `check` says about the `$` names the source still requires: one
    /// line each on stderr, or, under `--json`, one object after the
    /// diagnostics array.
    fn inputs(&self, required: &[InputListing]) {
        match self {
            Rendering::Text => {
                for input in required {
                    eprintln!("input ${}: {}", input.name, input.ty);
                }
            }
            Rendering::Json => {
                let listed: Vec<serde_json::Value> = required
                    .iter()
                    .map(|input| {
                        serde_json::json!({
                            "name": input.name,
                            "type": input.ty,
                        })
                    })
                    .collect();
                println!("{}", serde_json::json!({ "inputs": listed }));
            }
        }
    }

    fn report(&self, units: &[Unit], diagnostics: &[Diagnostic]) {
        match self {
            Rendering::Text => {
                for d in diagnostics {
                    let Some(unit) = d.unit.map(|at| &units[at]) else {
                        eprintln!("error: {}", d.message);
                        continue;
                    };
                    eprint!(
                        "{}",
                        Report {
                            severity: Severity::Error,
                            message: d.message.clone(),
                            primary: d.primary.clone(),
                            path: &unit.path,
                            source: &unit.text,
                            span: d.span,
                            labels: d.labels.clone(),
                        }
                    );
                }
            }
            Rendering::Json => {
                let array: Vec<serde_json::Value> = diagnostics
                    .iter()
                    .map(|d| {
                        let unit = d.unit.map(|at| &units[at]);
                        let lines = unit.map(|unit| Lines {
                            text: &unit.text,
                            index: LineIndex::new(&unit.text),
                        });
                        let at = |span: Span| {
                            let lines = lines.as_ref()?;
                            Some(lines.index.line_col(span.start.min(lines.text.len())))
                        };
                        let labels: Vec<serde_json::Value> = d
                            .labels
                            .iter()
                            .map(|l| {
                                serde_json::json!({
                                    "line": l.span.and_then(at).map(|p| p.line),
                                    "col": l.span.and_then(at).map(|p| p.col),
                                    "span": l.span.map(|s| [s.start, s.end]),
                                    "text": l.text,
                                })
                            })
                            .collect();
                        serde_json::json!({
                            "severity": Severity::Error.to_string(),
                            "message": d.message,
                            "primary": d.primary,
                            "path": unit.map(|unit| unit.path.as_str()),
                            "line": d.span.and_then(at).map(|p| p.line),
                            "col": d.span.and_then(at).map(|p| p.col),
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

struct Lines<'u> {
    text: &'u str,
    index: LineIndex,
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
    fn of(timed: Timed, stages: &CompileTimes) -> Option<Self> {
        match timed {
            Timed::On => Some(Timings {
                compile: stages.check(),
                stages: *stages,
                prepare: None,
                run: None,
            }),
            Timed::Off => None,
        }
    }

    fn report(&self, rendering: &Rendering) {
        let CompileTimes {
            opt,
            parse,
            typeck,
            lower,
            optimize,
            ..
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

/// What a script run by this CLI can call. `std_registries` and the four
/// beside it are the registries this runner host chooses to hand a script;
/// the language has none of its own (RFC-0031 rule 8).
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
        Ok(Invocation::Compile(args)) => args,
        Ok(Invocation::Lsp) => return lsp(),
        Ok(Invocation::Ctl(rest)) => return ctl(&rest).await,
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    match compile_command(args).await {
        Ok(code) => code,
        Err(stop) => stop.reported(),
    }
}

fn working_dir() -> Result<PathBuf, Stop> {
    std::env::current_dir().map_err(|e| Stop::usage(format!("the working directory: {e}")))
}

async fn ctl(rest: &[String]) -> ExitCode {
    if rest.is_empty() {
        eprintln!("{}", ctl::USAGE);
        return ExitCode::from(EXIT_USAGE);
    }
    let done = match rest {
        [space, fill, args @ ..] if space == "space" && fill == "fill" => fill_space(args).await,
        _ => working_dir().and_then(|cwd| ctl::ctl(rest, &cwd).map_err(Stop::from)),
    };
    match done {
        Ok(()) => ExitCode::SUCCESS,
        Err(stop) => stop.reported(),
    }
}

/// `acvus ctl space fill <space>`: every init whose context the space lacks
/// runs, and the space commits what they made.
async fn fill_space(args: &[String]) -> Result<(), Stop> {
    let [name] = args else {
        return Err(Stop::usage("usage: acvus ctl space fill <space>".to_string()));
    };
    let config = ConfigFile::read()?;
    let space = config.resolve(&SpaceChoice {
        name: name.clone(),
        source: SpaceSource::Flag,
    })?;
    let units = space_units(&space)?;
    let program = match compile::compile(&units, None, &[], cli_registries(), Opt::Full, SequentialExecutor) {
        Ok(program) => program,
        Err(Refused::Usage(message)) => return Err(Stop::usage(message)),
        Err(Refused::Diagnostics(diagnostics)) => {
            Rendering::Text.report(&units, &diagnostics);
            return Err(Stop {
                message: format!("space `{name}` does not compile"),
                exit: EXIT_COMPILE,
            });
        }
    };
    let contexts = space.location.open_contexts().map_err(Stop::run)?;
    program
        .scope(async |scope| -> Result<(), Stop> {
            let mut storage = SpaceStorage::new(&contexts);
            let mut page = scope.open(&mut storage);
            let filled = page.fill().await.map_err(|e| page_refusal(&space.name, e))?;
            for key in &filled {
                eprintln!("init @{key}");
            }
            commit(&mut page).await?;
            if filled.is_empty() {
                println!("space `{name}` holds every context it has an init of");
            }
            Ok(())
        })
        .await
}

/// A command that ends before its output, with the message and the exit
/// status RFC-0031 rule 7 gives it.
struct Stop {
    message: String,
    exit: u8,
}

impl Stop {
    fn usage(message: String) -> Self {
        Stop {
            message,
            exit: EXIT_USAGE,
        }
    }

    fn run(message: String) -> Self {
        Stop {
            message,
            exit: EXIT_RUN,
        }
    }

    fn reported(self) -> ExitCode {
        eprintln!("error: {}", self.message);
        ExitCode::from(self.exit)
    }
}

impl From<CtlError> for Stop {
    fn from(error: CtlError) -> Self {
        match error {
            CtlError::Refused(message) => Stop::usage(message),
            CtlError::Failed(message) => Stop::run(message),
        }
    }
}

/// The units a command compiles, and which one it is about.
struct Sources {
    units: Vec<Unit>,
    target: usize,
}

fn lone_file(path: &Path) -> Result<Sources, Stop> {
    let mode = mode_of(path).map_err(Stop::usage)?;
    let text = std::fs::read_to_string(path)
        .map_err(|e| Stop::usage(compile::unreadable_source(path, &e)))?;
    Ok(Sources {
        units: vec![Unit {
            role: Role::Entry(LONE_ENTRY.to_string()),
            space: None,
            path: path.display().to_string(),
            mode,
            text,
        }],
        target: 0,
    })
}

fn expr_unit(entry: &str, text: &str) -> Unit {
    Unit {
        role: Role::Entry(entry.to_string()),
        space: None,
        path: "<expr>".to_string(),
        mode: Mode::Expr,
        text: text.to_string(),
    }
}

/// Every script the space holds, one entry each, and every init it holds
/// (RFC-0031 rule 3).
fn space_units(space: &ResolvedSpace<'_>) -> Result<Vec<Unit>, Stop> {
    let scripts = space.location.scripts().map_err(Stop::run)?;
    let inits = space.location.inits().map_err(Stop::run)?;
    let scripts = scripts.into_iter().map(|script| Unit {
        path: format!("{}/{}.{}", space.name, script.name, script.kind.extension()),
        role: Role::Entry(script.name.as_str().to_string()),
        space: Some(space.name.clone()),
        mode: script.kind.mode(),
        text: script.text,
    });
    let inits = inits.into_iter().map(|init| Unit {
        path: format!("{}/inits/{}.{}", space.name, init.key, init.kind.extension()),
        role: Role::Init(init.key.as_str().to_string()),
        space: Some(space.name.clone()),
        mode: init.kind.mode(),
        text: init.text,
    });
    Ok(scripts.chain(inits).collect())
}

/// The space's units, and the script named or `-e`'s expression beside
/// them as the target.
fn space_sources(space: &ResolvedSpace<'_>, source: &Source) -> Result<Sources, Stop> {
    let mut units = space_units(space)?;
    let target = match source {
        Source::Positional(name) => match units
            .iter()
            .position(|unit| matches!(&unit.role, Role::Entry(held) if held == name))
        {
            Some(at) => at,
            None => {
                return Err(Stop::usage(format!(
                    "space `{}` holds no script `{name}`; `acvus ctl space add-script {} <file>` stores one",
                    space.name, space.name
                )));
            }
        },
        Source::Expr(text) => {
            units.push(expr_unit(EXPR_ENTRY, text));
            units.len() - 1
        }
    };
    Ok(Sources { units, target })
}

async fn compile_command(args: Args) -> Result<ExitCode, Stop> {
    let cwd = working_dir()?;
    let config = ConfigFile::read()?;
    let space = match ctl::choose_space(args.space.as_deref(), &cwd)? {
        Some(choice) => Some(config.resolve(&choice)?),
        None => None,
    };
    let settings = Layers {
        flag: &args.flags,
        space: space.as_ref().map(|space| space.defaults),
        context: config.context_defaults(),
    }
    .settle();
    let Sources { units, target } = match (&space, &args.source) {
        (Some(space), source) => space_sources(space, source)?,
        (None, Source::Positional(path)) => lone_file(Path::new(path))?,
        (None, Source::Expr(text)) => Sources {
            units: vec![expr_unit(LONE_ENTRY, text)],
            target: 0,
        },
    };

    let rendering = Rendering::of(args.json);
    let timed = match settings.time.value {
        Timing::On => Timed::On,
        Timing::Off => Timed::Off,
    };
    let opt = match settings.opt.value {
        OptLevel::None => Opt::None,
        OptLevel::Full => Opt::Full,
    };
    let executor: Box<dyn Executor> = match settings.parallel.value {
        Parallel::Tokio => Box::new(TokioExecutor),
        Parallel::Sequential => Box::new(SequentialExecutor),
    };
    let program = match compile::compile(&units, Some(target), &args.bindings, cli_registries(), opt, executor) {
        Ok(program) => program,
        Err(Refused::Usage(message)) => return Err(Stop::usage(message)),
        Err(Refused::Diagnostics(diagnostics)) => {
            rendering.report(&units, &diagnostics);
            return Ok(ExitCode::from(EXIT_COMPILE));
        }
    };
    let mut timings = Timings::of(timed, program.times());
    let Role::Entry(entry) = &units[target].role else {
        panic!("a command's target is a script or an expression, each an entry")
    };
    let about = program.listing(entry).map_err(|e| Stop::run(e.to_string()))?;
    match args.command {
        Command::Check => {
            rendering.report(&units, &[]);
            rendering.inputs(&about.inputs);
            if let Some(timings) = &timings {
                timings.report(&rendering);
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Mir => {
            match rendering {
                Rendering::Text => print!("{}", about.mir),
                Rendering::Json => rendering.report(&units, &[]),
            }
            if let Some(timings) = &timings {
                timings.report(&rendering);
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Ops => {
            let form = match args.json {
                true => oplist::Form::Json,
                false => oplist::Form::Text,
            };
            if let Some(timings) = &mut timings {
                timings.prepare = Some(program.times().prepare);
            }
            let text = oplist::dump(about.prepared, form)
                .map_err(|e| Stop::run(format!("the listing does not serialize: {e}")))?;
            print!("{text}");
            if let Some(timings) = &timings {
                timings.report(&rendering);
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Run => {
            if !about.inputs.is_empty() {
                for input in &about.inputs {
                    eprintln!(
                        "error: `${}` is required and not bound; `{}=<literal>` binds it",
                        input.name, input.name
                    );
                }
                return Ok(ExitCode::from(EXIT_COMPILE));
            }
            if let Some(timings) = &mut timings {
                timings.prepare = Some(program.times().prepare);
            }
            let run = Run {
                program: &program,
                entry,
                mode: units[target].mode,
                timed,
            };
            run.run(space.as_ref(), &mut timings).await?;
            if let Some(timings) = &timings {
                timings.report(&rendering);
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// A failed `serve` exits without joining the stdio threads: the reader
/// waits on a stdin the client may keep open, and joining it would hang.
fn lsp() -> ExitCode {
    let (connection, io_threads) = lsp_server::Connection::stdio();
    if let Err(error) = acvus_lsp::serve(connection, lsp_host::CliHost::new) {
        eprintln!("error: {error}");
        return ExitCode::from(EXIT_LSP_FAILED);
    }
    match io_threads.join() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: the stdio transport failed: {error}");
            ExitCode::from(EXIT_LSP_FAILED)
        }
    }
}

/// A context's refusal in the words of the command that resolves it
/// (RFC-0031 rule 7).
fn page_refusal(space: &str, error: HostError) -> Stop {
    match error {
        HostError::Unfilled { key } => Stop::run(format!(
            "`@{key}` is not in space `{space}` yet and has no init; `acvus ctl space init {space} {key} -e <expr>` stores one"
        )),
        HostError::Mismatched {
            what: Part::Context(key),
            held,
            asked,
        } => Stop::run(format!(
            "`@{key}` is held in space `{space}` as {held}, and the space's scripts and inits solve it to {asked}; `acvus ctl space add-script {space} <file>` changes the scripts back"
        )),
        other => run_failure(other),
    }
}

fn run_failure(error: HostError) -> Stop {
    match error {
        HostError::Trapped { message } => Stop::run(message),
        other => Stop::run(other.to_string()),
    }
}

/// Commit every context the runs stored, one `commit` line each on stderr.
async fn commit(page: &mut Page<'_, '_, SpaceStorage<'_>>) -> Result<(), Stop> {
    page.commit().await.map_err(|e| Stop::run(e.to_string()))?;
    for committed in page.storage().committed() {
        eprintln!("commit @{} = {}", committed.id, hex(&committed.head));
    }
    Ok(())
}

struct Run<'a> {
    program: &'a Program,
    entry: &'a str,
    mode: Mode,
    timed: Timed,
}

impl Run<'_> {
    /// A run over a space opens a page for the types the space's scripts and
    /// inits solve, runs the entry, and commits; a source that names no
    /// context runs alone.
    async fn run(self, space: Option<&ResolvedSpace<'_>>, timings: &mut Option<Timings>) -> Result<(), Stop> {
        let Some(space) = space else {
            let named: Vec<String> = self.program.contexts().map(|key| format!("`@{key}`")).collect();
            if !named.is_empty() {
                return Err(Stop::usage(format!(
                    "{} {} kept in a space, and no space is named here; `acvus ctl space add <space> dir:<path>` maps one, `acvus ctl space add-script <space> <file>` stores this script in it, and `acvus run <script> --space <space>` runs it",
                    named.join(", "),
                    match named.len() {
                        1 => "is",
                        _ => "are",
                    }
                )));
            }
            return self
                .program
                .scope(async |scope| -> Result<(), Stop> {
                    let entry = scope.untyped_entry(self.entry).map_err(|e| Stop::run(e.to_string()))?;
                    let mut storage = MemoryStorage::new();
                    let mut page = scope.open(&mut storage);
                    let printed = self
                        .run_on(&entry, &mut page, timings)
                        .await
                        .map_err(run_failure)?;
                    printed.print();
                    Ok(())
                })
                .await;
        };
        let contexts: Space = space.location.open_contexts().map_err(Stop::run)?;
        self.program
            .scope(async |scope| -> Result<(), Stop> {
                let entry = scope.untyped_entry(self.entry).map_err(|e| Stop::run(e.to_string()))?;
                let mut storage = SpaceStorage::new(&contexts);
                let mut page = scope.open(&mut storage);
                let ran = self.run_on(&entry, &mut page, timings).await;
                for key in page.filled() {
                    eprintln!("init @{key}");
                }
                let printed = ran.map_err(|e| page_refusal(&space.name, e))?;
                commit(&mut page).await?;
                printed.print();
                Ok(())
            })
            .await
    }

    async fn run_on<'p, S>(
        &self,
        entry: &UntypedEntry<'p>,
        page: &mut Page<'p, '_, S>,
        timings: &mut Option<Timings>,
    ) -> Result<Printed, HostError>
    where
        S: Storage,
    {
        let watch = Stopwatch::start(self.timed);
        let output = entry.run(page).await?;
        if let Some(timings) = timings {
            timings.run = watch.stop();
        }
        let interner = self.program.interner();
        Ok(output.with_value(|value, _| Printed::of(interner, value, self.mode)))
    }
}

/// RFC-0054: this host declares `!`, so it has no type for what comes back
/// and reads the value by kind. The text is made while the result is lent
/// and printed after the page commits.
enum Printed {
    Line(String),
    Text(String),
    Nothing,
}

impl Printed {
    fn of(interner: &Interner, value: &Value, mode: Mode) -> Self {
        match value.composite() {
            // SAFETY, both arms: the vtable is the runtime's witness of the
            // type behind the pointer.
            Some(Composite::String) => {
                let text = unsafe { value.as_str() }.to_owned();
                match mode {
                    Mode::Template => Printed::Text(text),
                    Mode::Script | Mode::Expr => Printed::Line(text),
                }
            }
            Some(Composite::Tuple) if unsafe { value.as_tuple() }.is_empty() => Printed::Nothing,
            _ if value.kind() == Kind::Unit => Printed::Nothing,
            _ => Printed::Line(json::by_kind(interner, value).to_string()),
        }
    }

    fn print(self) {
        match self {
            Printed::Line(text) => println!("{text}"),
            Printed::Text(text) => print!("{text}"),
            Printed::Nothing => {}
        }
    }
}
