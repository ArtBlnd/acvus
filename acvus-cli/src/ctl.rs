//! `acvus ctl`: which location each space name means, and how runs run
//! (RFC-0031 rule 9). The config holds named contexts; one is active, each
//! maps space names to locations, and a context and a space each keep run
//! defaults.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};

use acvus_interpreter::hex;
use acvus_utils::Interner;
use serde::{Deserialize, Serialize};

use crate::location::{
    ContextKey, Location, Removal, ScriptKind, ScriptName, Stored, admitted_name,
};

const MARKER_DIR: &str = ".acvus";
const MARKER_FILE: &str = "space.toml";

pub enum CtlError {
    Refused(String),
    Failed(String),
}

type Ctl<T> = Result<T, CtlError>;

fn refused<T>(message: String) -> Ctl<T> {
    Err(CtlError::Refused(message))
}

fn failed(message: String) -> CtlError {
    CtlError::Failed(message)
}

// -- Run defaults ------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Parallel {
    Sequential,
    Tokio,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptLevel {
    None,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Timing {
    On,
    Off,
}

pub trait Spelled: Copy + Sized + 'static {
    const KEY: &'static str;
    const ALL: &'static [Self];
    const BUILT_IN: Self;

    fn word(self) -> &'static str;

    fn parse(word: &str) -> Result<Self, String> {
        match Self::ALL.iter().find(|value| value.word() == word) {
            Some(value) => Ok(*value),
            None => Err(format!(
                "{} takes {}, not `{word}`",
                Self::KEY,
                Self::choices()
            )),
        }
    }

    fn choices() -> String {
        let words: Vec<&str> = Self::ALL.iter().map(|value| value.word()).collect();
        words.join(" or ")
    }
}

impl Spelled for Parallel {
    const KEY: &'static str = "parallel";
    const ALL: &'static [Self] = &[Parallel::Sequential, Parallel::Tokio];
    const BUILT_IN: Self = Parallel::Sequential;

    fn word(self) -> &'static str {
        match self {
            Parallel::Sequential => "sequential",
            Parallel::Tokio => "tokio",
        }
    }
}

impl Spelled for OptLevel {
    const KEY: &'static str = "opt";
    const ALL: &'static [Self] = &[OptLevel::None, OptLevel::Full];
    const BUILT_IN: Self = OptLevel::Full;

    fn word(self) -> &'static str {
        match self {
            OptLevel::None => "none",
            OptLevel::Full => "full",
        }
    }
}

impl Spelled for Timing {
    const KEY: &'static str = "time";
    const ALL: &'static [Self] = &[Timing::On, Timing::Off];
    const BUILT_IN: Self = Timing::Off;

    fn word(self) -> &'static str {
        match self {
            Timing::On => "on",
            Timing::Off => "off",
        }
    }
}

/// A run flag, and the value it sets on `defaults`: a bare `--parallel` is
/// `tokio` and a bare `--time` is `on`, so turning either off takes
/// `--parallel=sequential` or `--time=off`; `--opt` always takes its level.
/// `None` for a word that is no run flag.
pub fn run_flag(
    defaults: &mut Defaults,
    word: &str,
    next: &mut dyn FnMut() -> Option<String>,
) -> Option<Result<(), String>> {
    let (flag, written) = match word.split_once('=') {
        Some((flag, value)) => (flag, Some(value.to_owned())),
        None => (word, None),
    };
    let set = match flag {
        "--parallel" => written
            .map_or(Ok(Parallel::Tokio), |value| Parallel::parse(&value))
            .map(|value| defaults.parallel = Some(value)),
        "--time" => written
            .map_or(Ok(Timing::On), |value| Timing::parse(&value))
            .map(|value| defaults.time = Some(value)),
        "--opt" => match written.or_else(next) {
            Some(value) => OptLevel::parse(&value).map(|value| defaults.opt = Some(value)),
            None => Err(format!("opt takes {}", OptLevel::choices())),
        },
        _ => return None,
    };
    Some(set.map_err(|message| format!("--{message}")))
}

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Defaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel: Option<Parallel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opt: Option<OptLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time: Option<Timing>,
}

impl Defaults {
    fn is_unset(&self) -> bool {
        self.parallel.is_none() && self.opt.is_none() && self.time.is_none()
    }
}

#[derive(Clone, Copy)]
enum Key {
    Parallel,
    Opt,
    Time,
}

impl Key {
    fn parse(word: &str) -> Ctl<Key> {
        match word {
            "parallel" => Ok(Key::Parallel),
            "opt" => Ok(Key::Opt),
            "time" => Ok(Key::Time),
            _ => refused(format!(
                "`{word}` is not a run default; `acvus ctl set` takes parallel, opt or time"
            )),
        }
    }

    fn set(self, defaults: &mut Defaults, word: &str) -> Result<(), String> {
        match self {
            Key::Parallel => defaults.parallel = Some(Parallel::parse(word)?),
            Key::Opt => defaults.opt = Some(OptLevel::parse(word)?),
            Key::Time => defaults.time = Some(Timing::parse(word)?),
        }
        Ok(())
    }

    fn unset(self, defaults: &mut Defaults) {
        match self {
            Key::Parallel => defaults.parallel = None,
            Key::Opt => defaults.opt = None,
            Key::Time => defaults.time = None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    Flag,
    Space,
    Context,
    BuiltIn,
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Origin::Flag => "flag",
            Origin::Space => "space",
            Origin::Context => "context",
            Origin::BuiltIn => "built-in",
        })
    }
}

pub struct Setting<T> {
    pub value: T,
    pub origin: Origin,
}

pub struct Settings {
    pub parallel: Setting<Parallel>,
    pub opt: Setting<OptLevel>,
    pub time: Setting<Timing>,
}

/// A flag overrides the space, the space the context, and the context the
/// built-in default (RFC-0031 rule 9).
pub struct Layers<'a> {
    pub flag: &'a Defaults,
    pub space: Option<&'a Defaults>,
    pub context: Option<&'a Defaults>,
}

impl Layers<'_> {
    fn pick<T>(&self, field: fn(&Defaults) -> Option<T>) -> Setting<T>
    where
        T: Spelled,
    {
        if let Some(value) = field(self.flag) {
            return Setting {
                value,
                origin: Origin::Flag,
            };
        }
        if let Some(value) = self.space.and_then(field) {
            return Setting {
                value,
                origin: Origin::Space,
            };
        }
        if let Some(value) = self.context.and_then(field) {
            return Setting {
                value,
                origin: Origin::Context,
            };
        }
        Setting {
            value: T::BUILT_IN,
            origin: Origin::BuiltIn,
        }
    }

    pub fn settle(&self) -> Settings {
        Settings {
            parallel: self.pick(|d| d.parallel),
            opt: self.pick(|d| d.opt),
            time: self.pick(|d| d.time),
        }
    }
}

// -- The config file ---------------------------------------------------

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    active: Option<String>,
    #[serde(default)]
    contexts: BTreeMap<String, CtlContext>,
}

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CtlContext {
    #[serde(default, skip_serializing_if = "Defaults::is_unset")]
    defaults: Defaults,
    #[serde(default)]
    spaces: BTreeMap<String, SpaceEntry>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpaceEntry {
    location: Location,
    #[serde(default, skip_serializing_if = "Defaults::is_unset")]
    defaults: Defaults,
}

pub struct ConfigFile {
    pub path: PathBuf,
    pub config: Option<Config>,
}

/// `$XDG_CONFIG_HOME/acvus/config.toml`, else `$HOME/.config/acvus/config.toml`.
/// The XDG Base Directory specification has a relative `XDG_CONFIG_HOME`
/// ignored as an empty one is.
fn config_path() -> Result<PathBuf, String> {
    let absolute = |dir: OsString| {
        let dir = PathBuf::from(dir);
        dir.is_absolute().then_some(dir)
    };
    if let Some(dir) = std::env::var_os("XDG_CONFIG_HOME").and_then(absolute) {
        return Ok(dir.join("acvus").join("config.toml"));
    }
    match std::env::var_os("HOME").and_then(absolute) {
        Some(home) => Ok(home.join(".config").join("acvus").join("config.toml")),
        None => Err(
            "neither XDG_CONFIG_HOME nor HOME names an absolute directory, so ctl has no config file; set XDG_CONFIG_HOME"
                .to_string(),
        ),
    }
}

impl ConfigFile {
    pub fn read() -> Ctl<Self> {
        let path = config_path().map_err(CtlError::Refused)?;
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ConfigFile { path, config: None });
            }
            Err(e) => return Err(failed(format!("{}: {e}", path.display()))),
        };
        let config: Config = toml::from_str(&text).map_err(|e| {
            CtlError::Refused(format!(
                "{}: {e}; fix the file, or remove it and start again with `acvus ctl use <context>`",
                path.display()
            ))
        })?;
        Ok(ConfigFile {
            path,
            config: Some(config),
        })
    }

    fn write(&self, config: &Config) -> Ctl<()> {
        let text = toml::to_string(config)
            .map_err(|e| failed(format!("{}: {e}", self.path.display())))?;
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| failed(format!("{}: {e}", dir.display())))?;
        }
        std::fs::write(&self.path, text).map_err(|e| failed(format!("{}: {e}", self.path.display())))
    }

    fn missing(&self) -> String {
        format!(
            "no ctl config at {}; `acvus ctl use <context>` makes one, and `acvus ctl space add <space> dir:<path>` maps a space in it",
            self.path.display()
        )
    }

    fn active(&self) -> Ctl<NamedContext<'_>> {
        let Some(config) = &self.config else {
            return refused(self.missing());
        };
        let Some(name) = &config.active else {
            return refused(format!(
                "no ctl context is active in {}; `acvus ctl use <context>` makes one active",
                self.path.display()
            ));
        };
        match config.contexts.get(name) {
            Some(context) => Ok(NamedContext { name, context }),
            None => refused(format!(
                "the active ctl context `{name}` is not in {}; `acvus ctl use <context>` makes one active",
                self.path.display()
            )),
        }
    }

    pub fn context_defaults(&self) -> Option<&Defaults> {
        let config = self.config.as_ref()?;
        let context = config.contexts.get(config.active.as_ref()?)?;
        Some(&context.defaults)
    }

    pub fn resolve(&self, choice: &SpaceChoice) -> Ctl<ResolvedSpace<'_>> {
        let NamedContext {
            name: context_name,
            context,
        } = self.active()?;
        match context.spaces.get(&choice.name) {
            Some(entry) => Ok(ResolvedSpace {
                name: choice.name.clone(),
                location: entry.location.clone(),
                defaults: &entry.defaults,
            }),
            None => refused(format!(
                "space `{}` is not in ctl context `{context_name}`; `acvus ctl space add {} dir:<path>` maps it",
                choice.name, choice.name
            )),
        }
    }
}

struct NamedContext<'c> {
    name: &'c str,
    context: &'c CtlContext,
}

struct NamedContextMut<'c> {
    name: String,
    context: &'c mut CtlContext,
}

pub struct ResolvedSpace<'c> {
    pub name: String,
    pub location: Location,
    pub defaults: &'c Defaults,
}

// -- Which space a command is about ------------------------------------

pub enum SpaceSource {
    Flag,
    Marker(PathBuf),
}

pub struct SpaceChoice {
    pub name: String,
    pub source: SpaceSource,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Marker {
    space: String,
}

/// `--space`, else the nearest `.acvus/` at or above `cwd` (RFC-0031
/// rule 9).
pub fn choose_space(flag: Option<&str>, cwd: &Path) -> Ctl<Option<SpaceChoice>> {
    if let Some(name) = flag {
        admitted_name("space", name).map_err(CtlError::Refused)?;
        return Ok(Some(SpaceChoice {
            name: name.to_owned(),
            source: SpaceSource::Flag,
        }));
    }
    let Some(root) = cwd.ancestors().find(|root| root.join(MARKER_DIR).exists()) else {
        return Ok(None);
    };
    let dir = root.join(MARKER_DIR);
    let file = dir.join(MARKER_FILE);
    let text = std::fs::read_to_string(&file).map_err(|e| {
        CtlError::Refused(format!(
            "{}: {e}; `acvus ctl space mark <space>` in {} names the space it stands for",
            file.display(),
            root.display()
        ))
    })?;
    let marker: Marker = toml::from_str(&text).map_err(|e| {
        CtlError::Refused(format!(
            "{}: {e}; remove {} and run `acvus ctl space mark <space>` again",
            file.display(),
            dir.display()
        ))
    })?;
    admitted_name("space", &marker.space)
        .map_err(|e| CtlError::Refused(format!("{}: {e}", file.display())))?;
    Ok(Some(SpaceChoice {
        name: marker.space,
        source: SpaceSource::Marker(dir),
    }))
}

// -- The commands ------------------------------------------------------

struct Options {
    positional: Vec<String>,
    within: Option<String>,
    space: Option<String>,
    init: Option<InitGiven>,
    flags: Defaults,
}

/// How `ctl space init` is given the init's source.
enum InitGiven {
    Expr(String),
    File(String),
    Remove,
}

enum Admits {
    In,
    Space,
    InitSource,
    RunFlags,
}

fn options(args: &[String], admits: &[Admits]) -> Ctl<Options> {
    let mut options = Options {
        positional: Vec::new(),
        within: None,
        space: None,
        init: None,
        flags: Defaults::default(),
    };
    let admitted = |wanted: fn(&Admits) -> bool| admits.iter().any(wanted);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut next = || it.next().cloned();
        if admitted(|a| matches!(a, Admits::RunFlags))
            && let Some(set) = run_flag(&mut options.flags, arg, &mut next)
        {
            set.map_err(CtlError::Refused)?;
            continue;
        }
        let mut value = |flag: &str| match next() {
            Some(value) => Ok(value),
            None => refused(format!("{flag} takes a value")),
        };
        let mut given = |init: InitGiven| match options.init.replace(init) {
            None => Ok(()),
            Some(_) => refused("-e, -f and --rm each give the init; give one".to_string()),
        };
        match arg.as_str() {
            "--in" if admitted(|a| matches!(a, Admits::In)) => options.within = Some(value("--in")?),
            "--space" if admitted(|a| matches!(a, Admits::Space)) => {
                options.space = Some(value("--space")?)
            }
            "-e" if admitted(|a| matches!(a, Admits::InitSource)) => {
                given(InitGiven::Expr(value("-e")?))?
            }
            "-f" if admitted(|a| matches!(a, Admits::InitSource)) => {
                given(InitGiven::File(value("-f")?))?
            }
            "--rm" if admitted(|a| matches!(a, Admits::InitSource)) => given(InitGiven::Remove)?,
            flag if flag.starts_with('-') => {
                return refused(format!("`{flag}` is not a flag of this command; `acvus ctl` lists the commands"));
            }
            _ => options.positional.push(arg.clone()),
        }
    }
    if options.within.is_some() && options.space.is_some() {
        return refused("--in and --space name two different places; give one".to_string());
    }
    Ok(options)
}

pub const USAGE: &str = "\
usage: acvus ctl use <context>
       acvus ctl show [--space <space>] [--parallel[=P]] [--opt L] [--time[=T]]
       acvus ctl set <key> <value> [--in <context> | --space <space>]
       acvus ctl unset <key> [--in <context> | --space <space>]
       acvus ctl space add <space> dir:<path> [--in <context>]
       acvus ctl space rm <space> [--in <context>]
       acvus ctl space ls [<space>]
       acvus ctl space mark <space>
       acvus ctl space add-script <space> <file>...
       acvus ctl space rm-script <space> <script>
       acvus ctl space init <space> <key> -e <expr> | -f <file>
       acvus ctl space init --rm <space> <key>
       acvus ctl space fill <space>

  a ctl context maps space names to locations; `use` makes one active,
  and makes it where the config has none by that name. A location is
  dir:<path>, a directory holding the space's scripts, inits and contexts.
  `mark` names the space for every command under the working directory.
  `init` stores the init of `@key`: the expression or file whose value is
  the context's first value, run when a run fetches `@key` and the space
  lacks it; `fill` runs every init whose context the space lacks.
  keys: parallel (sequential | tokio), opt (full | none), time (on | off);
  a run takes a flag first, then the space's, then the context's, then
  the built-in default.";

pub fn ctl(argv: &[String], cwd: &Path) -> Ctl<()> {
    let file = ConfigFile::read()?;
    let words: Vec<&str> = argv.iter().take(2).map(String::as_str).collect();
    match words.as_slice() {
        ["use", ..] => use_context(file, &options(&argv[1..], &[])?),
        ["show", ..] => show(&file, &options(&argv[1..], &[Admits::Space, Admits::RunFlags])?, cwd),
        ["set", ..] => set(file, &options(&argv[1..], &[Admits::In, Admits::Space])?, Change::Set),
        ["unset", ..] => set(file, &options(&argv[1..], &[Admits::In, Admits::Space])?, Change::Unset),
        ["space", "add"] => space_add(file, &options(&argv[2..], &[Admits::In])?, cwd),
        ["space", "rm"] => space_rm(file, &options(&argv[2..], &[Admits::In])?),
        ["space", "ls"] => space_ls(&file, &options(&argv[2..], &[])?),
        ["space", "mark"] => space_mark(&options(&argv[2..], &[])?, cwd),
        ["space", "init"] => space_init(&file, &options(&argv[2..], &[Admits::InitSource])?),
        ["space", "add-script"] => add_script(&file, &options(&argv[2..], &[])?),
        ["space", "rm-script"] => rm_script(&file, &options(&argv[2..], &[])?),
        _ => refused(format!("`acvus ctl {}` is not a command\n{USAGE}", argv.join(" "))),
    }
}

fn exactly<const N: usize>(options: &Options, shape: &str) -> Ctl<[String; N]> {
    match <[String; N]>::try_from(options.positional.clone()) {
        Ok(words) => Ok(words),
        Err(_) => refused(format!("usage: acvus ctl {shape}")),
    }
}

fn use_context(mut file: ConfigFile, options: &Options) -> Ctl<()> {
    let [name] = exactly(options, "use <context>")?;
    admitted_name("ctl context", &name).map_err(CtlError::Refused)?;
    let mut config = match file.config.take() {
        Some(config) => config,
        None => Config::default(),
    };
    if !config.contexts.contains_key(&name) {
        config.contexts.insert(name.clone(), CtlContext::default());
        println!("made ctl context `{name}`");
    }
    config.active = Some(name.clone());
    file.write(&config)?;
    println!("ctl context `{name}` is active");
    Ok(())
}

fn context_mut<'c>(
    file: &ConfigFile,
    config: &'c mut Config,
    within: Option<&str>,
) -> Ctl<NamedContextMut<'c>> {
    let name = match within {
        Some(name) => name.to_owned(),
        None => match &config.active {
            Some(name) => name.clone(),
            None => {
                return refused(format!(
                    "no ctl context is active in {}; `acvus ctl use <context>` makes one active, or `--in <context>` names one",
                    file.path.display()
                ));
            }
        },
    };
    match config.contexts.get_mut(&name) {
        Some(context) => Ok(NamedContextMut { name, context }),
        None => refused(format!(
            "no ctl context `{name}` in {}; `acvus ctl use {name}` makes it",
            file.path.display()
        )),
    }
}

fn space_add(mut file: ConfigFile, options: &Options, cwd: &Path) -> Ctl<()> {
    let [name, written] = exactly(options, "space add <space> dir:<path> [--in <context>]")?;
    admitted_name("space", &name).map_err(CtlError::Refused)?;
    let location = Location::parse(&written, cwd).map_err(CtlError::Refused)?;
    let Some(mut config) = file.config.take() else {
        return refused(file.missing());
    };
    let NamedContextMut {
        name: context_name,
        context,
    } = context_mut(&file, &mut config, options.within.as_deref())?;
    match context.spaces.get(&name) {
        Some(entry) if entry.location != location => {
            return refused(format!(
                "space `{name}` is {} in ctl context `{context_name}`; `acvus ctl space rm {name}` first",
                entry.location
            ));
        }
        Some(_) => {}
        None => {
            context.spaces.insert(
                name.clone(),
                SpaceEntry {
                    location: location.clone(),
                    defaults: Defaults::default(),
                },
            );
        }
    }
    location.create().map_err(failed)?;
    file.write(&config)?;
    println!("space `{name}` is {location} in ctl context `{context_name}`");
    Ok(())
}

fn space_rm(mut file: ConfigFile, options: &Options) -> Ctl<()> {
    let [name] = exactly(options, "space rm <space> [--in <context>]")?;
    let Some(mut config) = file.config.take() else {
        return refused(file.missing());
    };
    let NamedContextMut {
        name: context_name,
        context,
    } = context_mut(&file, &mut config, options.within.as_deref())?;
    let Some(entry) = context.spaces.remove(&name) else {
        return refused(format!(
            "space `{name}` is not in ctl context `{context_name}`; `acvus ctl space ls` lists its spaces"
        ));
    };
    file.write(&config)?;
    println!(
        "space `{name}` is no longer in ctl context `{context_name}`; its files stay at {}",
        entry.location
    );
    Ok(())
}

fn space_ls(file: &ConfigFile, options: &Options) -> Ctl<()> {
    let NamedContext {
        name: context_name,
        context,
    } = file.active()?;
    let name = match options.positional.as_slice() {
        [] => {
            println!("ctl context `{context_name}`");
            for (name, entry) in &context.spaces {
                println!("{name} = {}", entry.location);
            }
            return Ok(());
        }
        [name] => name,
        _ => return refused("usage: acvus ctl space ls [<space>]".to_string()),
    };
    let resolved = file.resolve(&SpaceChoice {
        name: name.clone(),
        source: SpaceSource::Flag,
    })?;
    let location = &resolved.location;
    println!("space `{name}` = {location}");
    for script in location.scripts().map_err(failed)? {
        println!("script {}.{}", script.name, script.kind.extension());
    }
    for init in location.inits().map_err(failed)? {
        println!("init @{} ({})", init.key, init.kind.extension());
    }
    let interner = Interner::new();
    let space = location.open_contexts(&interner).map_err(failed)?;
    for (id, ty) in space.identities().map_err(|e| failed(e.to_string()))? {
        let Some(head) = space.head(&id) else {
            return Err(failed(format!("@{id} is listed by its head and has none")));
        };
        let mut values = 0usize;
        let mut at = Some(head);
        while let Some(hash) = at {
            values += 1;
            at = space.get(hash).map_err(|e| failed(e.to_string()))?.parent();
        }
        println!(
            "@{id}: {} = {}, {values} values in its log",
            ty.display(&interner),
            hex(&head)
        );
    }
    println!("{} nodes", space.node_count());
    Ok(())
}

fn space_mark(options: &Options, cwd: &Path) -> Ctl<()> {
    let [name] = exactly(options, "space mark <space>")?;
    admitted_name("space", &name).map_err(CtlError::Refused)?;
    let dir = cwd.join(MARKER_DIR);
    let file = dir.join(MARKER_FILE);
    if file.exists() {
        return refused(format!(
            "{} already names a space; remove {} to name another",
            file.display(),
            dir.display()
        ));
    }
    std::fs::create_dir_all(&dir).map_err(|e| failed(format!("{}: {e}", dir.display())))?;
    let text = toml::to_string(&Marker { space: name.clone() })
        .map_err(|e| failed(format!("{}: {e}", file.display())))?;
    std::fs::write(&file, text).map_err(|e| failed(format!("{}: {e}", file.display())))?;
    println!(
        "{} names space `{name}` for every command under {}",
        dir.display(),
        cwd.display()
    );
    Ok(())
}

fn space_init(file: &ConfigFile, options: &Options) -> Ctl<()> {
    const SHAPE: &str = "space init <space> <key> -e <expr> | -f <file>, or space init --rm <space> <key>";
    let [space, key] = exactly(options, SHAPE)?;
    let key = ContextKey::new(&key).map_err(CtlError::Refused)?;
    let Some(given) = &options.init else {
        return refused(format!("usage: acvus ctl {SHAPE}"));
    };
    let resolved = file.resolve(&SpaceChoice {
        name: space.clone(),
        source: SpaceSource::Flag,
    })?;
    let stored = match given {
        InitGiven::Expr(text) => resolved.location.put_init(&key, ScriptKind::Script, text),
        InitGiven::File(source) => {
            let Some(kind) = ScriptKind::of_path(Path::new(source)) else {
                return refused(format!(
                    "{source}: an init is .acvus (script) or .acvt (template); rename it to one"
                ));
            };
            let text = std::fs::read_to_string(source).map_err(|e| failed(format!("{source}: {e}")))?;
            resolved.location.put_init(&key, kind, &text)
        }
        InitGiven::Remove => {
            return match resolved.location.remove_init(&key).map_err(failed)? {
                Removal::Removed => {
                    println!("space `{space}` no longer holds an init of `@{key}`");
                    Ok(())
                }
                Removal::Absent => refused(format!(
                    "space `{space}` holds no init of `@{key}`; `acvus ctl space ls {space}` lists its inits"
                )),
            };
        }
    };
    match stored.map_err(failed)? {
        Stored::Added => println!("space `{space}` holds the init of `@{key}`"),
        Stored::Replaced => println!("space `{space}` holds the init of `@{key}`, replacing the one it held"),
    }
    Ok(())
}

fn add_script(file: &ConfigFile, options: &Options) -> Ctl<()> {
    let Some((space, sources)) = options.positional.split_first() else {
        return refused("usage: acvus ctl space add-script <space> <file>...".to_string());
    };
    if sources.is_empty() {
        return refused("usage: acvus ctl space add-script <space> <file>...".to_string());
    }
    let resolved = file.resolve(&SpaceChoice {
        name: space.clone(),
        source: SpaceSource::Flag,
    })?;
    for source in sources {
        let path = Path::new(source);
        let Some(kind) = ScriptKind::of_path(path) else {
            return refused(format!(
                "{source}: a script is .acvus and a template .acvt; rename it to one"
            ));
        };
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            return refused(format!("{source}: a script's file name is UTF-8; rename it"));
        };
        let name = ScriptName::new(stem)
            .map_err(|e| CtlError::Refused(format!("{source}: {e}; rename the file")))?;
        let text = std::fs::read_to_string(path).map_err(|e| failed(format!("{source}: {e}")))?;
        match resolved.location.put_script(&name, kind, &text).map_err(failed)? {
            Stored::Added => println!("space `{space}` holds script `{name}`"),
            Stored::Replaced => println!("space `{space}` holds script `{name}`, replacing the one it held"),
        }
    }
    Ok(())
}

fn rm_script(file: &ConfigFile, options: &Options) -> Ctl<()> {
    let [space, name] = exactly(options, "space rm-script <space> <script>")?;
    let name = ScriptName::new(&name).map_err(CtlError::Refused)?;
    let resolved = file.resolve(&SpaceChoice {
        name: space.clone(),
        source: SpaceSource::Flag,
    })?;
    match resolved.location.remove_script(&name).map_err(failed)? {
        Removal::Removed => {
            println!("space `{space}` no longer holds script `{name}`");
            Ok(())
        }
        Removal::Absent => refused(format!(
            "space `{space}` holds no script `{name}`; `acvus ctl space ls {space}` lists its scripts"
        )),
    }
}

enum Change {
    Set,
    Unset,
}

struct DefaultsOf<'c> {
    defaults: &'c mut Defaults,
    place: String,
}

fn set(mut file: ConfigFile, options: &Options, change: Change) -> Ctl<()> {
    let Some(key) = options.positional.first() else {
        return refused("usage: acvus ctl set <key> <value> | unset <key> [--in <context> | --space <space>]".to_string());
    };
    let key = Key::parse(key)?;
    let Some(mut config) = file.config.take() else {
        return refused(file.missing());
    };
    let NamedContextMut {
        name: context_name,
        context,
    } = context_mut(&file, &mut config, options.within.as_deref())?;
    let DefaultsOf { defaults, place } = match &options.space {
        Some(space) => match context.spaces.get_mut(space) {
            Some(entry) => DefaultsOf {
                defaults: &mut entry.defaults,
                place: format!("space `{space}`"),
            },
            None => {
                return refused(format!(
                    "space `{space}` is not in ctl context `{context_name}`; `acvus ctl space add {space} dir:<path>` maps it"
                ));
            }
        },
        None => DefaultsOf {
            defaults: &mut context.defaults,
            place: format!("ctl context `{context_name}`"),
        },
    };
    match (change, &options.positional[1..]) {
        (Change::Set, [value]) => key.set(defaults, value).map_err(CtlError::Refused)?,
        (Change::Unset, []) => key.unset(defaults),
        (Change::Set, _) => {
            return refused("usage: acvus ctl set <key> <value> [--in <context> | --space <space>]".to_string());
        }
        (Change::Unset, _) => {
            return refused("usage: acvus ctl unset <key> [--in <context> | --space <space>]".to_string());
        }
    }
    file.write(&config)?;
    println!("{place} updated");
    Ok(())
}

fn show(file: &ConfigFile, options: &Options, cwd: &Path) -> Ctl<()> {
    if !options.positional.is_empty() {
        return refused("usage: acvus ctl show [--space <space>] [--parallel[=P]] [--opt L] [--time[=T]]".to_string());
    }
    println!("config: {}", file.path.display());
    match &file.config {
        None => println!("context: none; `acvus ctl use <context>` makes one"),
        Some(config) => match &config.active {
            None => println!("context: none active; `acvus ctl use <context>` makes one active"),
            Some(name) => println!("context: {name}"),
        },
    }
    let resolved = match choose_space(options.space.as_deref(), cwd)? {
        Some(choice) => {
            let resolved = file.resolve(&choice)?;
            let from = match &choice.source {
                SpaceSource::Flag => "--space".to_string(),
                SpaceSource::Marker(dir) => dir.display().to_string(),
            };
            println!("space: {} = {} (from {from})", resolved.name, resolved.location);
            Some(resolved)
        }
        None => {
            println!("space: none");
            None
        }
    };
    let settings = Layers {
        flag: &options.flags,
        space: resolved.as_ref().map(|r| r.defaults),
        context: file.context_defaults(),
    }
    .settle();
    print_setting(&settings.parallel);
    print_setting(&settings.opt);
    print_setting(&settings.time);
    Ok(())
}

fn print_setting<T>(setting: &Setting<T>)
where
    T: Spelled,
{
    println!("{} = {} ({})", T::KEY, setting.value.word(), setting.origin);
}
