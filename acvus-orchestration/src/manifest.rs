use std::collections::BTreeMap;
use std::collections::HashSet;
use std::collections::btree_map::Entry;
use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub model: Model,
    #[serde(default, rename = "segment")]
    pub segments: Vec<Segment>,
    #[serde(default, rename = "tool")]
    pub tools: Vec<Tool>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub provider: Provider,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Anthropic,
    OpenAI,
    Google,
}

/// One piece of the request body, in the order the model reads them.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Segment {
    pub kind: Kind,
    pub source: Source,
    /// A cache boundary the user places after this segment. Absent, the
    /// lowering places boundaries from what the compiler proved constant.
    #[serde(default)]
    pub cache: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Instruction,
    User,
    Model,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Source {
    Text(Text),
    Script(Script),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Text {
    pub text: String,
}

/// An acvus script run before the request is sent; its string is the
/// segment's content.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Script {
    pub script: PathBuf,
    pub order: Order,
}

/// Position of a script among the scripts. Lower runs first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(transparent)]
pub struct Order(pub u32);

/// An acvus script the model may call after the request is sent.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub script: PathBuf,
}

/// The scripts that share one `Order`.
#[derive(Debug)]
pub enum Level<'a> {
    /// Runs by itself; any effect is allowed.
    Alone(&'a Script),
    /// Run in no fixed order among themselves; each must be pure.
    Shared(Vec<&'a Script>),
}

#[derive(Debug)]
pub enum Error {
    Toml(toml::de::Error),
    DuplicateTool(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Toml(e) => write!(f, "{e}"),
            Self::DuplicateTool(name) => write!(f, "tool `{name}` is declared twice"),
        }
    }
}

impl std::error::Error for Error {}

impl From<toml::de::Error> for Error {
    fn from(e: toml::de::Error) -> Self {
        Self::Toml(e)
    }
}

impl Manifest {
    pub fn parse(text: &str) -> Result<Self, Error> {
        let manifest: Self = toml::from_str(text)?;
        let mut seen = HashSet::new();
        for tool in &manifest.tools {
            if !seen.insert(&tool.name) {
                return Err(Error::DuplicateTool(tool.name.clone()));
            }
        }
        Ok(manifest)
    }

    pub fn scripts(&self) -> impl Iterator<Item = &Script> {
        self.segments.iter().filter_map(|s| match &s.source {
            Source::Script(script) => Some(script),
            Source::Text(_) => None,
        })
    }

    /// Scripts grouped by `Order`, lowest first.
    pub fn levels(&self) -> Vec<Level<'_>> {
        let mut by_order: BTreeMap<Order, (&Script, Vec<&Script>)> = BTreeMap::new();
        for script in self.scripts() {
            match by_order.entry(script.order) {
                Entry::Vacant(vacant) => {
                    vacant.insert((script, Vec::new()));
                }
                Entry::Occupied(mut occupied) => occupied.get_mut().1.push(script),
            }
        }
        by_order
            .into_values()
            .map(|(head, rest)| {
                if rest.is_empty() {
                    Level::Alone(head)
                } else {
                    Level::Shared(std::iter::once(head).chain(rest).collect())
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = include_str!("../manifest.example.toml");

    #[test]
    fn example_parses() {
        let manifest = Manifest::parse(EXAMPLE).unwrap();
        assert_eq!(manifest.model.provider, Provider::Anthropic);
        assert_eq!(manifest.segments.len(), 4);
        assert_eq!(manifest.tools.len(), 1);
        assert!(matches!(manifest.segments[1].source, Source::Text(_)));
        assert!(manifest.segments[1].cache);
    }

    #[test]
    fn levels_group_scripts_by_order() {
        let manifest = Manifest::parse(EXAMPLE).unwrap();
        let levels = manifest.levels();
        assert_eq!(levels.len(), 2);
        assert!(matches!(levels[0], Level::Alone(s) if s.script.ends_with("persona.acv")));
        assert!(matches!(&levels[1], Level::Shared(ss) if ss.len() == 2));
    }

    #[test]
    fn duplicate_tool_is_refused() {
        let text = r#"
            [model]
            provider = "anthropic"
            name = "m"

            [[tool]]
            name = "a"
            description = "d"
            script = "a.acv"

            [[tool]]
            name = "a"
            description = "d"
            script = "b.acv"
        "#;
        assert!(matches!(Manifest::parse(text), Err(Error::DuplicateTool(n)) if n == "a"));
    }

    #[test]
    fn source_with_both_text_and_script_is_refused() {
        let text = r#"
            [model]
            provider = "anthropic"
            name = "m"

            [[segment]]
            kind = "user"
            source = { text = "hi", script = "a.acv", order = 1 }
        "#;
        assert!(matches!(Manifest::parse(text), Err(Error::Toml(_))));
    }

    #[test]
    fn unknown_field_is_refused() {
        let text = r#"
            [model]
            provider = "anthropic"
            name = "m"
            temperature = 0.5
        "#;
        assert!(matches!(Manifest::parse(text), Err(Error::Toml(_))));
    }
}
