use std::fmt;

use acvus_ast::Span;
use acvus_ast::report::Label;
use acvus_utils::Interner;

use crate::ty::{Concrete, Types};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShownValue {
    Named(String),
    Anonymous,
}

impl fmt::Display for ShownValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShownValue::Named(name) => write!(f, "`{name}`"),
            ShownValue::Anonymous => write!(f, "this value"),
        }
    }
}

/// The names a refusal offers in place of the one it did not find. A
/// candidate is within an edit distance of two of what was written, or has
/// it as a prefix; the nearest three are kept, and an empty list prints
/// nothing, so a refusal with no near name keeps the sentence it had.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DidYouMean(Vec<String>);

impl DidYouMean {
    /// At most this many names, so the sentence stays one line.
    pub const SHOWN: usize = 3;

    /// The names near `wanted` among `known`, nearest first and then
    /// alphabetically, with `wanted` itself never among them.
    pub fn of<I>(wanted: &str, known: I) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        let mut near: Vec<(usize, String)> = known
            .into_iter()
            .filter(|name| name != wanted)
            .filter_map(|name| near(wanted, &name).map(|d| (d, name)))
            .collect();
        near.sort();
        near.dedup_by(|a, b| a.1 == b.1);
        near.truncate(Self::SHOWN);
        Self(near.into_iter().map(|(_, name)| name).collect())
    }
}

impl fmt::Display for DidYouMean {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [first, rest @ ..] = self.0.as_slice() else {
            return Ok(());
        };
        write!(f, "; did you mean `{first}`")?;
        for (i, name) in rest.iter().enumerate() {
            match i + 2 == self.0.len() {
                true => write!(f, " or `{name}`")?,
                false => write!(f, ", `{name}`")?,
            }
        }
        write!(f, "?")
    }
}

/// How far `name` is from `wanted`, where near enough to offer: an edit
/// distance of at most two, or `wanted` written as a prefix of it or as
/// one of its `_`-separated words (`iter` names `as_iter` and `into_iter`,
/// not `filter`). A name of one or two characters is near everything of
/// its length, so only a prefix or a word counts there.
fn near(wanted: &str, name: &str) -> Option<usize> {
    if wanted.len() >= 3 && (name.starts_with(wanted) || name.split('_').any(|word| word == wanted))
    {
        return Some(0);
    }
    if wanted.len() < 3 {
        return None;
    }
    match edit_distance(wanted, name) {
        d @ 0..=2 => Some(d),
        _ => None,
    }
}

/// The Levenshtein distance, over characters.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let next = match ca == *cb {
                true => diagonal,
                false => 1 + diagonal.min(row[j]).min(row[j + 1]),
            };
            diagonal = row[j + 1];
            row[j + 1] = next;
        }
    }
    row[b.len()]
}

#[derive(Debug, Clone)]
pub struct MirError {
    pub kind: MirErrorKind,
    pub span: Span,
    pub labels: Vec<Label>,
}

#[derive(Debug, Clone)]
pub enum MirErrorKind {
    /// A field the body reads on a path that never stored it
    /// (`validate::init_check`, RFC-0042).
    FieldNotStored {
        subject: ShownValue,
        fields: Vec<String>,
        near: DidYouMean,
    },
}

/// Everything one compilation refused, whichever stage raised it: the stages
/// that read the source and its types speak `MirError`, and the validator
/// that reads a lowered body speaks `ValidationError`.
#[derive(Debug, Clone)]
pub enum Refusal {
    Mir(MirError),
    Invalid(crate::validate::ValidationError),
}

impl Refusal {
    pub fn span(&self) -> Span {
        match self {
            Refusal::Mir(error) => error.span,
            Refusal::Invalid(error) => error.span,
        }
    }

    pub fn labels(&self) -> &[Label] {
        match self {
            Refusal::Mir(error) => &error.labels,
            Refusal::Invalid(error) => error.labels(),
        }
    }

    /// The refusal in words; a validator's refusal reads its types in
    /// `types`, the table of the module it was raised in.
    pub fn display<'a>(&'a self, types: &'a Types<Concrete>, interner: &'a Interner) -> RefusalDisplay<'a> {
        RefusalDisplay {
            refusal: self,
            types,
            interner,
        }
    }
}

pub struct RefusalDisplay<'a> {
    refusal: &'a Refusal,
    types: &'a Types<Concrete>,
    interner: &'a Interner,
}

impl fmt::Display for RefusalDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.refusal {
            Refusal::Mir(error) => error.fmt(f),
            Refusal::Invalid(error) => error.display(self.types, self.interner).fmt(f),
        }
    }
}

impl fmt::Display for MirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            MirErrorKind::FieldNotStored {
                subject,
                fields,
                near,
            } => {
                let named: Vec<String> = fields.iter().map(|f| format!("`{f}`")).collect();
                write!(
                    f,
                    "{subject} has no {} stored on every path that reaches here{near}",
                    named.join(", ")
                )
            }
        }
    }
}
