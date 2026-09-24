//! Where a space keeps its scripts and its contexts (RFC-0031 rules 3
//! and 9). Every read and write of a space's files goes through
//! `Location`, so a new kind of location changes this file and no caller.

use std::fmt;
use std::path::{Path, PathBuf};

use acvus_interpreter::{DirStore, Log, Space};

use crate::compile::Mode;

/// The interval a directory store checkpointed at before spaces were named
/// by `acvus ctl`.
const CHECKPOINT_EVERY: usize = 64;

const SCRIPTS: &str = "scripts";
const INITS: &str = "inits";

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Location {
    Dir(PathBuf),
}

pub struct StoredScript {
    pub name: ScriptName,
    pub kind: ScriptKind,
    pub text: String,
}

pub struct StoredInit {
    pub key: ContextKey,
    pub kind: ScriptKind,
    pub text: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ScriptKind {
    Script,
    Template,
}

impl ScriptKind {
    pub fn of_path(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()? {
            "acvus" => Some(ScriptKind::Script),
            "acvt" => Some(ScriptKind::Template),
            _ => None,
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            ScriptKind::Script => "acvus",
            ScriptKind::Template => "acvt",
        }
    }

    pub fn mode(self) -> Mode {
        match self {
            ScriptKind::Script => Mode::Script,
            ScriptKind::Template => Mode::Template,
        }
    }
}

/// The entry an `-e` expression compiles to beside a space's scripts; no
/// `ScriptName` spells it.
pub const EXPR_ENTRY: &str = "-e";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ScriptName(String);

pub fn admitted_name(what: &str, name: &str) -> Result<(), String> {
    let admitted = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-';
    match name.chars().next() {
        Some(first) if first != '-' && name.chars().all(admitted) => Ok(()),
        _ => Err(format!(
            "`{name}` is not a {what} name; a {what} is named by letters, digits, `_` and `-`, not led by `-`"
        )),
    }
}

impl ScriptName {
    pub fn new(name: &str) -> Result<Self, String> {
        admitted_name("script", name)?;
        Ok(ScriptName(name.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ScriptName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ContextKey(String);

impl ContextKey {
    pub fn new(key: &str) -> Result<Self, String> {
        let mut chars = key.chars();
        let admitted = match chars.next() {
            Some(first) => {
                (first.is_ascii_alphabetic() || first == '_')
                    && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
            }
            None => false,
        };
        match admitted {
            true => Ok(ContextKey(key.to_owned())),
            false => Err(format!(
                "`{key}` is not a context's name; `@name` is a letter or `_`, then letters, digits and `_`"
            )),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ContextKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

pub enum Stored {
    Added,
    Replaced,
}

pub enum Removal {
    Removed,
    Absent,
}

impl Location {
    pub fn parse(text: &str, cwd: &Path) -> Result<Self, String> {
        match text.split_once(':') {
            Some(("dir", path)) if !path.is_empty() => {
                let joined = cwd.join(path);
                match joined.to_str() {
                    Some(_) => Ok(Location::Dir(joined)),
                    None => Err(format!(
                        "{} is not UTF-8, and a config file stores a location as text",
                        joined.display()
                    )),
                }
            }
            _ => Err(format!(
                "`{text}` is not a location; a location is `dir:<path>`"
            )),
        }
    }

    fn io(&self, error: std::io::Error) -> String {
        format!("{self}: {error}")
    }

    pub fn create(&self) -> Result<(), String> {
        match self {
            Location::Dir(dir) => {
                std::fs::create_dir_all(dir.join(SCRIPTS)).map_err(|e| self.io(e))?;
                std::fs::create_dir_all(dir.join(INITS)).map_err(|e| self.io(e))?;
                DirStore::open(dir).map_err(|e| e.to_string())?;
                Ok(())
            }
        }
    }

    pub fn open_contexts(&self) -> Result<Space, String> {
        match self {
            Location::Dir(dir) => {
                let store = DirStore::open(dir).map_err(|e| e.to_string())?;
                Ok(Space::over(
                    Log {
                        checkpoint_every: CHECKPOINT_EVERY,
                    },
                    Box::new(store),
                ))
            }
        }
    }

    pub fn scripts(&self) -> Result<Vec<StoredScript>, String> {
        let sources = self.sources(SCRIPTS, "script")?;
        let mut scripts = Vec::with_capacity(sources.len());
        for Source { path, stem, kind, text } in sources {
            let name = ScriptName::new(&stem).map_err(|e| format!("{}: {e}", path.display()))?;
            scripts.push(StoredScript { name, kind, text });
        }
        scripts.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(scripts)
    }

    /// A location made before inits were kept has no directory for them,
    /// and holds none.
    pub fn inits(&self) -> Result<Vec<StoredInit>, String> {
        let Location::Dir(dir) = self;
        if !dir.join(INITS).exists() {
            return Ok(Vec::new());
        }
        let sources = self.sources(INITS, "init")?;
        let mut inits = Vec::with_capacity(sources.len());
        for Source { path, stem, kind, text } in sources {
            let key = ContextKey::new(&stem).map_err(|e| format!("{}: {e}", path.display()))?;
            inits.push(StoredInit { key, kind, text });
        }
        inits.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(inits)
    }

    fn sources(&self, under: &str, what: &str) -> Result<Vec<Source>, String> {
        match self {
            Location::Dir(dir) => {
                let listed = std::fs::read_dir(dir.join(under)).map_err(|e| {
                    format!("{}; `acvus ctl space add` makes a location's directories", self.io(e))
                })?;
                let mut sources = Vec::new();
                for entry in listed {
                    let path = entry.map_err(|e| self.io(e))?.path();
                    let Some(kind) = ScriptKind::of_path(&path) else {
                        return Err(format!(
                            "{} is not a source; a space's {what}s are .acvus and .acvt, and `rm` removes it",
                            path.display()
                        ));
                    };
                    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                        return Err(format!("{}: a {what}'s file name is UTF-8", path.display()));
                    };
                    let stem = stem.to_owned();
                    let text = std::fs::read_to_string(&path).map_err(|e| self.io(e))?;
                    sources.push(Source { path, stem, kind, text });
                }
                Ok(sources)
            }
        }
    }

    fn source_path(dir: &Path, under: &str, stem: &str, kind: ScriptKind) -> PathBuf {
        dir.join(under).join(format!("{stem}.{}", kind.extension()))
    }

    pub fn put_script(
        &self,
        name: &ScriptName,
        kind: ScriptKind,
        text: &str,
    ) -> Result<Stored, String> {
        self.put(SCRIPTS, name.as_str(), kind, text)
    }

    pub fn remove_script(&self, name: &ScriptName) -> Result<Removal, String> {
        self.remove(SCRIPTS, name.as_str())
    }

    pub fn put_init(&self, key: &ContextKey, kind: ScriptKind, text: &str) -> Result<Stored, String> {
        self.put(INITS, key.as_str(), kind, text)
    }

    pub fn remove_init(&self, key: &ContextKey) -> Result<Removal, String> {
        self.remove(INITS, key.as_str())
    }

    fn put(&self, under: &str, stem: &str, kind: ScriptKind, text: &str) -> Result<Stored, String> {
        let replaced = self.remove(under, stem)?;
        match self {
            Location::Dir(dir) => {
                std::fs::create_dir_all(dir.join(under)).map_err(|e| self.io(e))?;
                std::fs::write(Self::source_path(dir, under, stem, kind), text)
                    .map_err(|e| self.io(e))?;
            }
        }
        Ok(match replaced {
            Removal::Removed => Stored::Replaced,
            Removal::Absent => Stored::Added,
        })
    }

    fn remove(&self, under: &str, stem: &str) -> Result<Removal, String> {
        match self {
            Location::Dir(dir) => {
                let mut removal = Removal::Absent;
                for kind in [ScriptKind::Script, ScriptKind::Template] {
                    match std::fs::remove_file(Self::source_path(dir, under, stem, kind)) {
                        Ok(()) => removal = Removal::Removed,
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => return Err(self.io(e)),
                    }
                }
                Ok(removal)
            }
        }
    }
}

/// A file a location holds under `scripts/` or `inits/`, before its stem is
/// read as a name.
struct Source {
    path: PathBuf,
    stem: String,
    kind: ScriptKind,
    text: String,
}

/// A location as a config file stores it: `dir:` and an absolute path, since
/// no working directory is at hand when the file is read.
impl TryFrom<String> for Location {
    type Error = String;

    fn try_from(text: String) -> Result<Self, String> {
        match text.split_once(':') {
            Some(("dir", path)) if Path::new(path).is_absolute() => {
                Ok(Location::Dir(PathBuf::from(path)))
            }
            _ => Err(format!(
                "`{text}` is not a stored location; a stored location is `dir:<absolute path>`"
            )),
        }
    }
}

impl From<Location> for String {
    fn from(location: Location) -> String {
        location.to_string()
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Location::Dir(path) => write!(f, "dir:{}", path.display()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_script_is_named_as_the_expression_entry() {
        assert!(ScriptName::new(EXPR_ENTRY).is_err());
    }
}
