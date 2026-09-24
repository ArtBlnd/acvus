//! The sources of one command as one `Host` compilation: each source is one
//! entry of the graph (RFC-0054 rule 1), or one context's init, and a
//! context's type is what the graph solves from all of them (RFC-0090
//! rule 1). A refusal comes back as a diagnostic on the unit it names.

use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

use acvus_ast::Span;
use acvus_ast::report::Label;
use acvus_extern::{CombineError, Registry};
use acvus_interpreter::{AcvusRuntime, Executor, Host, HostError, Origin, Program, Refusal, Source};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{Parsed, QualifiedRef};
use acvus_utils::Interner;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Script,
    Template,
    Expr,
}

/// One source of a compilation, and the path its diagnostics are rendered
/// at.
pub struct Unit {
    pub role: Role,
    pub path: String,
    pub mode: Mode,
    pub text: String,
}

pub enum Role {
    Entry(String),
    /// The init of the context of this name.
    Init(String),
}

pub struct Diagnostic {
    /// The unit whose text `span` points into; `None` for a refusal of the
    /// graph as a whole, which no source holds.
    pub unit: Option<usize>,
    pub message: String,
    /// The words the primary marker carries; `None` leaves it repeating
    /// the message where another span is marked.
    pub primary: Option<String>,
    pub span: Option<Span>,
    pub labels: Vec<Label>,
}

/// `name=<literal>`, binding the input `$name` (RFC-0031 rule 4).
pub struct Binding {
    pub name: String,
    pub text: String,
}

pub enum Refused {
    /// A binding outside the literal grammar, which is a usage error as a
    /// form outside that syntax is.
    Usage(String),
    Diagnostics(Vec<Diagnostic>),
}

#[derive(Clone, Copy)]
pub enum Timed {
    On,
    Off,
}

pub struct Stopwatch(Option<Instant>);

impl Stopwatch {
    pub fn start(timed: Timed) -> Self {
        Stopwatch(match timed {
            Timed::On => Some(Instant::now()),
            Timed::Off => None,
        })
    }

    pub fn stop(self) -> Option<Duration> {
        Some(self.0?.elapsed())
    }
}

/// The entry a lone source compiles to.
pub fn entry_ref(interner: &Interner) -> QualifiedRef {
    QualifiedRef::root(interner.intern("main"))
}

pub fn combine_refusal(error: &CombineError) -> String {
    format!("the registries do not combine: {error}")
}

pub fn unreadable_source(path: &Path, error: &io::Error) -> String {
    format!("{}: {error}", path.display())
}

pub fn source(mode: Mode, text: &str) -> Source<'_> {
    match mode {
        Mode::Script => Source::Script(text),
        Mode::Template => Source::Template(text),
        Mode::Expr => Source::Expr(text),
    }
}

pub fn parse(interner: &Interner, mode: Mode, text: &str) -> Parsed {
    source(mode, text).parse(interner)
}

/// Compile `units` as one graph, every entry declaring `!` and reading its
/// `$` inputs from its body.
pub fn compile<E>(
    units: &[Unit],
    bindings: &[Binding],
    registries: Vec<Registry<AcvusRuntime>>,
    opt: Opt,
    executor: E,
) -> Result<Program, Refused>
where
    E: Executor + 'static,
{
    let mut host = Host::new(registries).opt(opt);
    for Binding { name, text } in bindings {
        host = host.bind(name, text).map_err(|error| Refused::Usage(usage_of(error)))?;
    }
    for unit in units {
        let source = source(unit.mode, &unit.text);
        host = match &unit.role {
            Role::Entry(name) => host.untyped_entry(name, source),
            Role::Init(key) => host.init(key, source),
        };
    }
    host.compile(executor)
        .map_err(|error| Refused::Diagnostics(diagnostics(units, error)))
}

fn usage_of(error: HostError) -> String {
    match error {
        HostError::Refused(refusals) => {
            let messages: Vec<String> = refusals.into_iter().map(|refusal| refusal.message).collect();
            messages.join("; ")
        }
        other => other.to_string(),
    }
}

fn diagnostics(units: &[Unit], error: HostError) -> Vec<Diagnostic> {
    let HostError::Refused(refusals) = error else {
        return vec![Diagnostic {
            unit: None,
            message: error.to_string(),
            primary: None,
            span: None,
            labels: Vec::new(),
        }];
    };
    refusals
        .into_iter()
        .map(|refusal| {
            let Refusal {
                origin,
                message,
                span,
                primary,
                labels,
            } = refusal;
            Diagnostic {
                unit: origin.and_then(|origin| unit_of(units, &origin)),
                message,
                primary,
                span,
                labels,
            }
        })
        .collect()
}

fn unit_of(units: &[Unit], origin: &Origin) -> Option<usize> {
    units.iter().position(|unit| match (&unit.role, origin) {
        (Role::Entry(held), Origin::Entry(name)) => held == name,
        (Role::Init(held), Origin::Init(key)) => held == key,
        _ => false,
    })
}
