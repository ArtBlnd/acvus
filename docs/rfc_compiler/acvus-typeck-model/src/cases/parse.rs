//! A case's text read into what it declares, what its program's text gives,
//! and what it expects (`design/19-cases.md`, Form). Nothing is resolved here:
//! a type is read as it is written, and the builder resolves it.

use std::collections::{BTreeMap, BTreeSet};

use crate::decl::Role;
use crate::istar::Kind;
use crate::lit::Literal;
use crate::program::{BindingPath, Captures, Deferral, Place, Readings, Visibility};
use crate::ty::Mutability;

/// Where in a case's file something is written.
pub type Line = usize;

/// What a case writes, with the line it is written on.
#[derive(Clone, Debug)]
pub struct Located<T> {
    pub line: Line,
    pub item: T,
}

/// A type as a case writes it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TypeExpr {
    Ref(Mutability, Box<TypeExpr>),
    /// `()`, `(T,)`, `(T, U)`.
    Tuple(Vec<TypeExpr>),
    Array(u64, Box<TypeExpr>),
    Slice(Box<TypeExpr>),
    Never,
    /// `#`, a signature's identity.
    Identity,
    Object(Vec<(String, TypeExpr)>),
    /// `E{V(T)}`.
    Enum(String, Vec<(String, TypeExpr)>),
    /// `Fn/n`.
    FunctionValue(usize),
    /// `fn(P̄) -> R`.
    Function(Vec<TypeExpr>, Box<TypeExpr>),
    /// A name, with the arguments written after it.
    Named(String, Vec<TypeExpr>),
}

#[derive(Clone, Debug)]
pub enum DeclItem {
    Extern {
        name: String,
        arity: usize,
        identity: bool,
    },
    Struct {
        name: String,
        fields: Vec<String>,
    },
    Fn {
        name: String,
        role: Role,
        generics: Vec<String>,
        identity: bool,
        params: Vec<TypeExpr>,
        result: TypeExpr,
    },
}

#[derive(Clone, Debug)]
pub enum OriginText {
    Var,
    Lit(Literal),
    Written(TypeExpr),
    Host(TypeExpr),
    Lambda(String),
}

#[derive(Clone, Debug)]
pub enum ProjectionText {
    Deref,
    Field(String),
    Component(usize),
    Elem,
    Payload {
        enum_name: Option<String>,
        variant: String,
    },
}

/// One thing the program's text gives, in the order written.
#[derive(Clone, Debug)]
pub enum TextItem {
    /// What follows lies in this group, until `EndGroup`.
    Group {
        name: String,
        parent: Option<String>,
    },
    EndGroup,
    Class {
        name: String,
        origin: OriginText,
    },
    Universal(String),
    Function {
        name: String,
        params: Vec<String>,
        result: String,
    },
    Pass {
        value: String,
        into: String,
        place: Place,
    },
    Equation(String, String),
    Use {
        node: String,
        callee: String,
        args: Vec<String>,
        result: String,
    },
    CallValue {
        node: String,
        callee: String,
        args: Vec<String>,
        result: String,
    },
    Lend {
        node: String,
        mutability: Mutability,
        place: String,
        result: String,
    },
    Project {
        node: String,
        projection: ProjectionText,
        operand: String,
        result: String,
    },
    Bind {
        node: String,
        path: Vec<String>,
        position: String,
        binding: String,
    },
}

/// A step of an order a case writes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum StepText {
    Solve(String),
    Decide(String),
    Fallback,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Orders {
    /// Every order, which must be explored within the budget.
    All,
    /// Orders sampled (`design/13-exploration.md`, reading 2).
    Sampled(u64),
    /// One order, the steps of each group listed by its name, each group
    /// otherwise taking its first step.
    Written(BTreeMap<String, Vec<StepText>>),
}

/// A refusal an expectation names: its kind, and where it names one, its
/// class's name or the place a conflict is refused at.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RefusalText {
    Infinite(Option<String>),
    NoJoin(Option<String>),
    NoValue(Option<String>),
    Conflict(Option<String>),
    Undeclared,
}

/// What a phase's report is, for `assert <phase> records`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Record,
    Unify,
    Join,
    Split,
    Node,
    Fallback,
}

/// A change to a certificate (`design/14-certificate.md`).
#[derive(Clone, Debug)]
pub enum Change {
    Value { class: String, ty: TypeExpr },
    Decision { node: String, decision: String },
}

#[derive(Clone, Debug)]
pub enum Expect {
    Accepted(bool),
    /// The class's value; none where it has none.
    Value {
        class: String,
        ty: Option<TypeExpr>,
    },
    /// The node's decision, as `render::decision` writes it.
    Decision {
        node: String,
        decision: String,
    },
    Refused(RefusalText),
    /// The relations of the failing checks, all of them, in the outcome's
    /// order.
    Failing(Vec<String>),
    /// A check of the relation, with its coercion as `render::coercion`
    /// writes it.
    Check {
        relation: String,
        coercion: String,
    },
    Istar {
        kind: Kind,
        node: String,
    },
    Outcomes(usize),
    Partition(BTreeSet<String>),
    Certificate {
        group: String,
        change: Change,
        fails: BTreeSet<String>,
    },
    /// A line of a written order's trace.
    Trace(String),
    Records {
        phase: Phase,
        relation: String,
        present: bool,
    },
    Join {
        a: TypeExpr,
        b: TypeExpr,
        joined: Option<TypeExpr>,
    },
    Le {
        a: TypeExpr,
        b: TypeExpr,
        holds: bool,
    },
    Coerce {
        along_le: bool,
        from: TypeExpr,
        to: TypeExpr,
        coercion: String,
    },
}

#[derive(Clone, Debug)]
pub struct Case {
    pub spec: Vec<String>,
    pub hole: Option<String>,
    /// One run for each; none written is one run of the default readings.
    pub readings: Vec<Readings>,
    pub orders: Orders,
    pub decls: Vec<Located<DeclItem>>,
    pub text: Vec<Located<TextItem>>,
    /// What every outcome and final state meets.
    pub expects: Vec<Located<Expect>>,
    /// Each `outcome { … }`, met by exactly one outcome.
    pub outcomes: Vec<Located<Vec<Located<Expect>>>>,
    /// A hole's `wanted { … }`: what the outcome the spec's text, its examples
    /// or the RFCs it cites want meets, as `holes.md` states it.
    pub wanted: Option<Vec<Located<Expect>>>,
}

#[derive(Clone, PartialEq, Debug)]
enum Tok {
    Word(String),
    /// A name in backquotes, never a keyword.
    Quoted(String),
    Str(String),
    Punct(&'static str),
}

#[derive(Clone, Debug)]
struct Token {
    tok: Tok,
    line: Line,
    start: usize,
    end: usize,
}

const PUNCTS: [&str; 22] = [
    "->", "==", "(", ")", "{", "}", "[", "]", "<", ">", ",", ";", ":", "=", "&", "*", ".", "#",
    "@", "!", "/", "+",
];

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '?'
}

/// The tokens of `source`, each comment left out; a directive is read before.
fn lex(source: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<(usize, char)> = source.char_indices().collect();
    let mut line = 1;
    let mut i = 0;
    let at = |i: usize| chars.get(i).map(|&(_, c)| c);
    let offset = |i: usize| chars.get(i).map_or(source.len(), |&(o, _)| o);
    while i < chars.len() {
        let c = chars[i].1;
        if c == '\n' {
            line += 1;
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '/' && at(i + 1) == Some('/') {
            while i < chars.len() && chars[i].1 != '\n' {
                i += 1;
            }
            continue;
        }
        let start = offset(i);
        if c == '`' || c == '"' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].1 != c {
                if chars[j].1 == '\n' {
                    return Err(format!("line {line}: an unclosed {c}"));
                }
                j += 1;
            }
            if j == chars.len() {
                return Err(format!("line {line}: an unclosed {c}"));
            }
            let text = source[offset(i + 1)..offset(j)].to_string();
            let tok = if c == '`' {
                Tok::Quoted(text)
            } else {
                Tok::Str(text)
            };
            tokens.push(Token {
                tok,
                line,
                start,
                end: offset(j + 1),
            });
            i = j + 1;
            continue;
        }
        if is_word(c) {
            let mut j = i;
            loop {
                while j < chars.len() && is_word(chars[j].1) {
                    j += 1;
                }
                if at(j) == Some(':') && at(j + 1) == Some(':') && at(j + 2).is_some_and(is_word) {
                    j += 2;
                    continue;
                }
                break;
            }
            tokens.push(Token {
                tok: Tok::Word(source[start..offset(j)].to_string()),
                line,
                start,
                end: offset(j),
            });
            i = j;
            continue;
        }
        let rest = &source[start..];
        let Some(punct) = PUNCTS.iter().find(|punct| rest.starts_with(**punct)) else {
            return Err(format!("line {line}: unexpected {c:?}"));
        };
        let len = punct.chars().count();
        tokens.push(Token {
            tok: Tok::Punct(punct),
            line,
            start,
            end: offset(i + len),
        });
        i += len;
    }
    Ok(tokens)
}

struct Parser<'a> {
    source: &'a str,
    tokens: Vec<Token>,
    at: usize,
    /// The source's last line, on which its end lies.
    last_line: Line,
}

/// `<T, U, #>`: a declaration's type parameters, and whether it names an
/// identity.
struct Generics {
    params: Vec<String>,
    identity: bool,
}

type Parsed<T> = Result<T, String>;

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.at).map(|token| &token.tok)
    }

    fn peek_at(&self, ahead: usize) -> Option<&Tok> {
        self.tokens.get(self.at + ahead).map(|token| &token.tok)
    }

    /// The line of the token at hand, or the last line where every token is
    /// taken.
    fn line(&self) -> Line {
        match self.tokens.get(self.at) {
            Some(token) => token.line,
            None => self.last_line,
        }
    }

    fn error<T>(&self, what: &str) -> Parsed<T> {
        let found = match self.peek() {
            Some(tok) => format!("{tok:?}"),
            None => "the end".to_string(),
        };
        Err(format!(
            "line {}: expected {what}, found {found}",
            self.line()
        ))
    }

    fn is_punct(&self, punct: &str) -> bool {
        matches!(self.peek(), Some(Tok::Punct(p)) if *p == punct)
    }

    fn is_word(&self, word: &str) -> bool {
        matches!(self.peek(), Some(Tok::Word(w)) if w == word)
    }

    fn eat_punct(&mut self, punct: &str) -> bool {
        let found = self.is_punct(punct);
        if found {
            self.at += 1;
        }
        found
    }

    fn eat_word(&mut self, word: &str) -> bool {
        let found = self.is_word(word);
        if found {
            self.at += 1;
        }
        found
    }

    fn punct(&mut self, punct: &str) -> Parsed<()> {
        if self.eat_punct(punct) {
            Ok(())
        } else {
            self.error(&format!("`{punct}`"))
        }
    }

    fn word(&mut self, word: &str) -> Parsed<()> {
        if self.eat_word(word) {
            Ok(())
        } else {
            self.error(&format!("`{word}`"))
        }
    }

    /// A name: a word, or text in backquotes.
    fn name(&mut self) -> Parsed<String> {
        match self.peek().cloned() {
            Some(Tok::Word(word)) | Some(Tok::Quoted(word)) => {
                self.at += 1;
                Ok(word)
            }
            _ => self.error("a name"),
        }
    }

    fn string(&mut self) -> Parsed<String> {
        match self.peek().cloned() {
            Some(Tok::Str(text)) => {
                self.at += 1;
                Ok(text)
            }
            _ => self.error("a string"),
        }
    }

    fn number<T>(&mut self) -> Parsed<T>
    where
        T: std::str::FromStr,
    {
        match self.peek().cloned() {
            Some(Tok::Word(word)) => match word.parse() {
                Ok(number) => {
                    self.at += 1;
                    Ok(number)
                }
                Err(_) => self.error("a number"),
            },
            _ => self.error("a number"),
        }
    }

    /// Names separated by commas, up to `close`, which is taken.
    fn names_until(&mut self, close: &str) -> Parsed<Vec<String>> {
        let mut names = Vec::new();
        while !self.eat_punct(close) {
            names.push(self.name()?);
            if !self.eat_punct(",") {
                self.punct(close)?;
                break;
            }
        }
        Ok(names)
    }

    fn strings_until(&mut self, close: &str) -> Parsed<Vec<String>> {
        let mut strings = Vec::new();
        while !self.eat_punct(close) {
            strings.push(self.string()?);
            if !self.eat_punct(",") {
                self.punct(close)?;
                break;
            }
        }
        Ok(strings)
    }

    /// The text written from here up to the word `stop`, or `;` where
    /// `stop` is none, which is taken.
    fn raw_until(&mut self, stop: Option<&str>) -> Parsed<String> {
        let first = self.at;
        loop {
            match (self.peek(), stop) {
                (None, _) => return self.error("`;`"),
                (Some(Tok::Punct(";")), None) => break,
                (Some(Tok::Word(word)), Some(stop)) if word == stop => break,
                _ => self.at += 1,
            }
        }
        if self.at == first {
            return self.error("text");
        }
        let (start, end) = (self.tokens[first].start, self.tokens[self.at - 1].end);
        self.at += 1;
        Ok(self.source[start..end].to_string())
    }

    fn ty(&mut self) -> Parsed<TypeExpr> {
        if self.eat_punct("&") {
            let mutability = if self.eat_word("mut") {
                Mutability::Exclusive
            } else {
                Mutability::Shared
            };
            return Ok(TypeExpr::Ref(mutability, Box::new(self.ty()?)));
        }
        if self.eat_punct("(") {
            let mut components = Vec::new();
            let mut trailing = false;
            while !self.eat_punct(")") {
                components.push(self.ty()?);
                trailing = self.eat_punct(",");
                if !trailing {
                    self.punct(")")?;
                    break;
                }
            }
            if components.len() == 1 && !trailing {
                return self.error("`,` after a tuple's one component");
            }
            return Ok(TypeExpr::Tuple(components));
        }
        if self.eat_punct("[") {
            let element = Box::new(self.ty()?);
            if self.eat_punct(";") {
                let len = self.number()?;
                self.punct("]")?;
                return Ok(TypeExpr::Array(len, element));
            }
            self.punct("]")?;
            return Ok(TypeExpr::Slice(element));
        }
        if self.eat_punct("!") {
            return Ok(TypeExpr::Never);
        }
        if self.eat_punct("#") {
            return Ok(TypeExpr::Identity);
        }
        if self.eat_punct("{") {
            let mut fields = Vec::new();
            while !self.eat_punct("}") {
                let field = self.name()?;
                self.punct(":")?;
                fields.push((field, self.ty()?));
                if !self.eat_punct(",") {
                    self.punct("}")?;
                    break;
                }
            }
            return Ok(TypeExpr::Object(fields));
        }
        if self.is_word("Fn") && matches!(self.peek_at(1), Some(Tok::Punct("/"))) {
            self.at += 2;
            return Ok(TypeExpr::FunctionValue(self.number()?));
        }
        if self.is_word("fn") && matches!(self.peek_at(1), Some(Tok::Punct("("))) {
            self.at += 2;
            let mut params = Vec::new();
            while !self.eat_punct(")") {
                params.push(self.ty()?);
                if !self.eat_punct(",") {
                    self.punct(")")?;
                    break;
                }
            }
            self.punct("->")?;
            return Ok(TypeExpr::Function(params, Box::new(self.ty()?)));
        }
        let name = self.name()?;
        if self.eat_punct("{") {
            let mut variants = Vec::new();
            while !self.eat_punct("}") {
                let variant = self.name()?;
                self.punct("(")?;
                let payload = self.ty()?;
                self.punct(")")?;
                variants.push((variant, payload));
                if !self.eat_punct(",") {
                    self.punct("}")?;
                    break;
                }
            }
            return Ok(TypeExpr::Enum(name, variants));
        }
        let mut args = Vec::new();
        if self.eat_punct("<") {
            while !self.eat_punct(">") {
                args.push(self.ty()?);
                if !self.eat_punct(",") {
                    self.punct(">")?;
                    break;
                }
            }
        }
        Ok(TypeExpr::Named(name, args))
    }

    fn generics(&mut self) -> Parsed<Generics> {
        let mut params = Vec::new();
        let mut identity = false;
        if self.eat_punct("<") {
            while !self.eat_punct(">") {
                if self.eat_punct("#") {
                    identity = true;
                } else {
                    params.push(self.name()?);
                }
                if !self.eat_punct(",") {
                    self.punct(">")?;
                    break;
                }
            }
        }
        Ok(Generics { params, identity })
    }

    /// `#[name]`.
    fn attribute(&mut self) -> Parsed<String> {
        self.punct("#")?;
        self.punct("[")?;
        let name = self.name()?;
        self.punct("]")?;
        Ok(name)
    }

    fn extern_fn(&mut self, role: Role) -> Parsed<DeclItem> {
        self.word("extern")?;
        self.word("fn")?;
        let name = self.name()?;
        let Generics {
            params: generics,
            identity,
        } = self.generics()?;
        self.punct("(")?;
        let mut params = Vec::new();
        while !self.eat_punct(")") {
            params.push(self.ty()?);
            if !self.eat_punct(",") {
                self.punct(")")?;
                break;
            }
        }
        let result = if self.eat_punct("->") {
            self.ty()?
        } else {
            TypeExpr::Tuple(Vec::new())
        };
        self.punct(";")?;
        Ok(DeclItem::Fn {
            name,
            role,
            generics,
            identity,
            params,
            result,
        })
    }

    /// What follows `=` in a statement, giving `result`.
    fn rhs(&mut self, result: String) -> Parsed<TextItem> {
        let node = |parser: &mut Self| -> Parsed<String> {
            parser.punct("@")?;
            parser.name()
        };
        if self.eat_punct("&") {
            let mutability = if self.eat_word("mut") {
                Mutability::Exclusive
            } else {
                Mutability::Shared
            };
            let place = self.name()?;
            return Ok(TextItem::Lend {
                node: node(self)?,
                mutability,
                place,
                result,
            });
        }
        if self.eat_punct("*") {
            let operand = self.name()?;
            return Ok(TextItem::Project {
                node: node(self)?,
                projection: ProjectionText::Deref,
                operand,
                result,
            });
        }
        if self.eat_punct("(") {
            let callee = self.name()?;
            self.punct(")")?;
            self.punct("(")?;
            let args = self.names_until(")")?;
            return Ok(TextItem::CallValue {
                node: node(self)?,
                callee,
                args,
                result,
            });
        }
        if self.is_word("bind")
            && self.tokens.get(self.at + 2).map(|t| &t.tok) == Some(&Tok::Word("via".to_string()))
        {
            self.at += 1;
            let position = self.name()?;
            self.word("via")?;
            self.punct("[")?;
            let path = self.names_until("]")?;
            return Ok(TextItem::Bind {
                node: node(self)?,
                path,
                position,
                binding: result,
            });
        }
        let name = self.name()?;
        if self.eat_punct("(") {
            let args = self.names_until(")")?;
            return Ok(TextItem::Use {
                node: node(self)?,
                callee: name,
                args,
                result,
            });
        }
        if self.eat_punct(".") {
            let part = self.name()?;
            let projection = match part.parse() {
                Ok(index) => ProjectionText::Component(index),
                Err(_) => ProjectionText::Field(part),
            };
            return Ok(TextItem::Project {
                node: node(self)?,
                projection,
                operand: name,
                result,
            });
        }
        if self.eat_punct("[") {
            self.punct("]")?;
            return Ok(TextItem::Project {
                node: node(self)?,
                projection: ProjectionText::Elem,
                operand: name,
                result,
            });
        }
        if self.eat_word("as") {
            let path = self.name()?;
            let projection = match path.rsplit_once("::") {
                Some((enum_name, variant)) => ProjectionText::Payload {
                    enum_name: Some(enum_name.to_string()),
                    variant: variant.to_string(),
                },
                None => ProjectionText::Payload {
                    enum_name: None,
                    variant: path,
                },
            };
            return Ok(TextItem::Project {
                node: node(self)?,
                projection,
                operand: name,
                result,
            });
        }
        Ok(TextItem::Pass {
            value: name,
            into: result,
            place: Place::Slot,
        })
    }

    fn origin(&mut self) -> Parsed<OriginText> {
        if self.eat_word("lit") {
            let whole: u128 = self.number()?;
            if self.eat_punct(".") {
                let fraction: String = self.name()?;
                let value: f64 = format!("{whole}.{fraction}")
                    .parse()
                    .map_err(|_| format!("line {}: a float literal", self.line()))?;
                return Ok(OriginText::Lit(Literal::float(value)));
            }
            return Ok(OriginText::Lit(Literal::Int(whole)));
        }
        if self.eat_word("host") {
            return Ok(OriginText::Host(self.ty()?));
        }
        if self.eat_word("lambda") {
            return Ok(OriginText::Lambda(self.name()?));
        }
        Ok(OriginText::Written(self.ty()?))
    }

    /// A statement of the program's text, pushed to `text`.
    fn statement(&mut self, text: &mut Vec<Located<TextItem>>) -> Parsed<()> {
        let line = self.line();
        if self.eat_word("let") {
            let name = self.name()?;
            let origin = if self.eat_punct(":") {
                self.origin()?
            } else {
                OriginText::Var
            };
            text.push(Located {
                line,
                item: TextItem::Class {
                    name: name.clone(),
                    origin,
                },
            });
            if self.eat_punct("=") {
                let item = self.rhs(name)?;
                text.push(Located { line, item });
            }
            return self.punct(";");
        }
        if self.eat_word("universal") {
            let item = TextItem::Universal(self.name()?);
            text.push(Located { line, item });
            return self.punct(";");
        }
        if self.eat_word("fn") {
            let name = self.name()?;
            self.punct("(")?;
            let params = self.names_until(")")?;
            self.punct("->")?;
            let result = self.name()?;
            text.push(Located {
                line,
                item: TextItem::Function {
                    name,
                    params,
                    result,
                },
            });
            return self.punct(";");
        }
        if self.is_punct("#") {
            let place = match self.attribute()?.as_str() {
                "by_name" => Place::ByName,
                "through_ref" => Place::ThroughRef,
                other => return Err(format!("line {line}: no attribute {other} on a pass")),
            };
            let into = self.name()?;
            self.punct("=")?;
            let value = self.name()?;
            text.push(Located {
                line,
                item: TextItem::Pass { value, into, place },
            });
            return self.punct(";");
        }
        let name = self.name()?;
        if self.eat_punct("==") {
            let item = TextItem::Equation(name, self.name()?);
            text.push(Located { line, item });
            return self.punct(";");
        }
        self.punct("=")?;
        let item = self.rhs(name)?;
        text.push(Located { line, item });
        self.punct(";")
    }

    fn refusal(&mut self) -> Parsed<RefusalText> {
        let named = |parser: &mut Self| -> Parsed<Option<String>> {
            if parser.is_punct(";") {
                Ok(None)
            } else {
                parser.name().map(Some)
            }
        };
        let kind = self.name()?;
        Ok(match kind.as_str() {
            "infinite" => RefusalText::Infinite(named(self)?),
            "no_join" => RefusalText::NoJoin(named(self)?),
            "no_value" => RefusalText::NoValue(named(self)?),
            "conflict" => {
                if self.eat_word("at") {
                    RefusalText::Conflict(Some(self.string()?))
                } else {
                    RefusalText::Conflict(None)
                }
            }
            "undeclared" => RefusalText::Undeclared,
            other => return Err(format!("line {}: no refusal {other}", self.line())),
        })
    }

    fn phase(word: &str) -> Option<Phase> {
        Some(match word {
            "record" => Phase::Record,
            "unify" => Phase::Unify,
            "join" => Phase::Join,
            "split" => Phase::Split,
            "node" => Phase::Node,
            "fallback" => Phase::Fallback,
            _ => return None,
        })
    }

    /// The two types of a primitive's arguments, `(T, U)`.
    fn pair(&mut self) -> Parsed<(TypeExpr, TypeExpr)> {
        self.punct("(")?;
        let a = self.ty()?;
        self.punct(",")?;
        let b = self.ty()?;
        self.punct(")")?;
        Ok((a, b))
    }

    /// What follows `assert`.
    fn assertion(&mut self) -> Parsed<Expect> {
        let keyword = match self.peek() {
            Some(Tok::Word(word)) => Some(word.clone()),
            _ => None,
        };
        let next = self.peek_at(1).cloned();
        let expect = match (keyword.as_deref(), &next) {
            (Some("refused"), _) => {
                self.at += 1;
                Expect::Refused(self.refusal()?)
            }
            (Some("failing"), Some(Tok::Punct("["))) => {
                self.at += 2;
                Expect::Failing(self.names_until("]")?)
            }
            (Some("check"), Some(Tok::Word(_) | Tok::Quoted(_))) => {
                self.at += 1;
                let relation = self.name()?;
                self.punct("==")?;
                return Ok(Expect::Check {
                    relation,
                    coercion: self.raw_until(None)?,
                });
            }
            (Some("istar"), _) => {
                self.at += 1;
                let kind = match self.name()?.as_str() {
                    "completeness" => Kind::Completeness,
                    "soundness" => Kind::Soundness,
                    other => return Err(format!("line {}: no I* kind {other}", self.line())),
                };
                Expect::Istar {
                    kind,
                    node: self.name()?,
                }
            }
            (Some("outcomes"), Some(Tok::Word(_))) => {
                self.at += 1;
                Expect::Outcomes(self.number()?)
            }
            (Some("partition"), Some(Tok::Punct("{"))) => {
                self.at += 2;
                Expect::Partition(self.names_until("}")?.into_iter().collect())
            }
            (Some("certificate"), _) => {
                self.at += 1;
                let group = if self.eat_word("in") {
                    self.name()?
                } else {
                    crate::program::ENTRY.to_string()
                };
                self.word("set")?;
                let target = self.name()?;
                let change = if self.eat_punct(":") {
                    let ty = self.ty()?;
                    self.word("fails")?;
                    Change::Value { class: target, ty }
                } else {
                    self.punct("==")?;
                    Change::Decision {
                        node: target,
                        decision: self.raw_until(Some("fails"))?,
                    }
                };
                self.punct("[")?;
                let fails = self.strings_until("]")?.into_iter().collect();
                Expect::Certificate {
                    group,
                    change,
                    fails,
                }
            }
            (Some("trace"), Some(Tok::Str(_))) => {
                self.at += 1;
                Expect::Trace(self.string()?)
            }
            (Some("no"), Some(Tok::Word(phase))) if Self::phase(phase).is_some() => {
                self.at += 2;
                let phase = Self::phase(phase).expect("the phase is named");
                self.word("records")?;
                Expect::Records {
                    phase,
                    relation: self.name()?,
                    present: false,
                }
            }
            (Some(phase), Some(Tok::Word(records)))
                if records == "records" && Self::phase(phase).is_some() =>
            {
                let phase = Self::phase(phase).expect("the phase is named");
                self.at += 2;
                Expect::Records {
                    phase,
                    relation: self.name()?,
                    present: true,
                }
            }
            (Some("join"), Some(Tok::Punct("("))) => {
                self.at += 1;
                let (a, b) = self.pair()?;
                self.punct("==")?;
                let joined = if self.eat_word("none") {
                    None
                } else {
                    Some(self.ty()?)
                };
                Expect::Join { a, b, joined }
            }
            (Some("le"), Some(Tok::Punct("("))) => {
                self.at += 1;
                let (a, b) = self.pair()?;
                Expect::Le { a, b, holds: true }
            }
            (Some("not"), Some(Tok::Word(le))) if le == "le" => {
                self.at += 2;
                let (a, b) = self.pair()?;
                Expect::Le { a, b, holds: false }
            }
            (Some(primitive @ ("coerce" | "along_le")), Some(Tok::Punct("("))) => {
                let along_le = primitive == "along_le";
                self.at += 1;
                let (from, to) = self.pair()?;
                self.punct("==")?;
                return Ok(Expect::Coerce {
                    along_le,
                    from,
                    to,
                    coercion: self.raw_until(None)?,
                });
            }
            _ => {
                let name = self.name()?;
                if self.eat_punct(":") {
                    let ty = if self.eat_word("none") {
                        None
                    } else {
                        Some(self.ty()?)
                    };
                    Expect::Value { class: name, ty }
                } else {
                    self.punct("==")?;
                    return Ok(Expect::Decision {
                        node: name,
                        decision: self.raw_until(None)?,
                    });
                }
            }
        };
        self.punct(";")?;
        Ok(expect)
    }

    fn expectation(&mut self) -> Parsed<Option<Expect>> {
        if self.eat_word("must") {
            let accepted = if self.eat_word("pass") {
                true
            } else {
                self.word("fail")?;
                false
            };
            self.punct(";")?;
            return Ok(Some(Expect::Accepted(accepted)));
        }
        if self.eat_word("assert") {
            return self.assertion().map(Some);
        }
        Ok(None)
    }

    /// `{ expectation* }`.
    fn expect_block(&mut self) -> Parsed<Vec<Located<Expect>>> {
        self.punct("{")?;
        let mut expects = Vec::new();
        while !self.eat_punct("}") {
            let line = self.line();
            match self.expectation()? {
                Some(item) => expects.push(Located { line, item }),
                None => return self.error("`must` or `assert`"),
            }
        }
        Ok(expects)
    }
}

/// The directives: `//@ key: value` lines. `orders` and `order` exclude each
/// other, and each is written once.
fn directives(source: &str, case: &mut Case) -> Parsed<()> {
    let mut orders: Option<Orders> = None;
    for (i, line) in source.lines().enumerate() {
        let Some(directive) = line.trim_start().strip_prefix("//@") else {
            continue;
        };
        let at = i + 1;
        let (key, value) = directive
            .split_once(':')
            .ok_or_else(|| format!("line {at}: a directive is `//@ key: value`"))?;
        let (key, value) = (key.trim(), value.trim());
        match key.split_whitespace().collect::<Vec<_>>().as_slice() {
            ["spec"] => case.spec.push(value.to_string()),
            ["hole"] => case.hole = Some(value.to_string()),
            ["reading"] => case
                .readings
                .push(readings(value).map_err(|e| format!("line {at}: {e}"))?),
            ["orders"] => {
                if orders.is_some() {
                    return Err(format!("line {at}: the orders are written twice"));
                }
                orders = Some(
                    match value.split_whitespace().collect::<Vec<_>>().as_slice() {
                        ["all"] => Orders::All,
                        ["sampled", n] => Orders::Sampled(
                            n.parse()
                                .map_err(|_| format!("line {at}: a number of orders"))?,
                        ),
                        _ => {
                            return Err(format!("line {at}: `orders: all` or `orders: sampled N`"));
                        }
                    },
                );
            }
            ["order", rest @ ..] => {
                let group = match rest {
                    [] => crate::program::ENTRY.to_string(),
                    [group] => group.to_string(),
                    _ => return Err(format!("line {at}: `order: ...` or `order <group>: ...`")),
                };
                let steps = value
                    .split(',')
                    .map(str::trim)
                    .filter(|step| !step.is_empty())
                    .map(|step| match step.split_once(' ') {
                        Some(("solve", class)) => Ok(StepText::Solve(unquote(class))),
                        Some(("decide", node)) => Ok(StepText::Decide(unquote(node))),
                        None if step == "fallback" => Ok(StepText::Fallback),
                        _ => Err(format!("line {at}: no step {step}")),
                    })
                    .collect::<Parsed<Vec<_>>>()?;
                let written = match orders.get_or_insert_with(|| Orders::Written(BTreeMap::new())) {
                    Orders::Written(written) => written,
                    Orders::All | Orders::Sampled(_) => {
                        return Err(format!("line {at}: a written order with `orders`"));
                    }
                };
                if written.insert(group.clone(), steps).is_some() {
                    return Err(format!("line {at}: the order of {group} is written twice"));
                }
            }
            _ => return Err(format!("line {at}: no directive {key}")),
        }
    }
    if let Some(orders) = orders {
        case.orders = orders;
    }
    Ok(())
}

/// A name a directive writes, in backquotes where it has other characters.
fn unquote(text: &str) -> String {
    let text = text.trim();
    text.strip_prefix('`')
        .and_then(|text| text.strip_suffix('`'))
        .unwrap_or(text)
        .to_string()
}

/// `default`, or `key = value` pairs, the rest as `Readings::default()`.
fn readings(text: &str) -> Result<Readings, String> {
    let mut readings = Readings::default();
    if text == "default" {
        return Ok(readings);
    }
    for pair in text.split(',') {
        let (key, value) = pair
            .split_once('=')
            .ok_or_else(|| format!("a reading is `key = value`: {pair}"))?;
        match (key.trim(), value.trim()) {
            ("visibility", "by_member") => readings.visibility = Visibility::ByMember,
            ("visibility", "by_sources") => readings.visibility = Visibility::BySources,
            ("binding_path", "as_stated") => readings.binding_path = BindingPath::AsStated,
            ("binding_path", "after_path") => readings.binding_path = BindingPath::AfterPath,
            ("deferral", "recording") => readings.deferral = Deferral::Recording,
            ("deferral", "reading") => readings.deferral = Deferral::Reading,
            ("captures", "as_stated") => readings.captures = Captures::AsStated,
            ("captures", "environment") => readings.captures = Captures::Environment,
            (key, value) => return Err(format!("no reading {key} = {value}")),
        }
    }
    Ok(readings)
}

pub fn parse(source: &str) -> Parsed<Case> {
    let mut case = Case {
        spec: Vec::new(),
        hole: None,
        readings: Vec::new(),
        orders: Orders::All,
        decls: Vec::new(),
        text: Vec::new(),
        expects: Vec::new(),
        outcomes: Vec::new(),
        wanted: None,
    };
    directives(source, &mut case)?;
    let mut parser = Parser {
        source,
        tokens: lex(source)?,
        at: 0,
        last_line: source.lines().count(),
    };
    while parser.peek().is_some() {
        let line = parser.line();
        if let Some(item) = parser.expectation()? {
            case.expects.push(Located { line, item });
            continue;
        }
        if parser.eat_word("outcome") {
            let item = parser.expect_block()?;
            case.outcomes.push(Located { line, item });
            continue;
        }
        if parser.eat_word("wanted") {
            if case.wanted.is_some() {
                return Err(format!("line {line}: a case has one `wanted` block"));
            }
            case.wanted = Some(parser.expect_block()?);
            continue;
        }
        if parser.is_word("extern")
            && matches!(parser.peek_at(1), Some(Tok::Word(w)) if w == "type")
        {
            parser.at += 2;
            let name = parser.name()?;
            let Generics { params, identity } = parser.generics()?;
            parser.punct(";")?;
            case.decls.push(Located {
                line,
                item: DeclItem::Extern {
                    name,
                    arity: params.len(),
                    identity,
                },
            });
            continue;
        }
        if parser.eat_word("struct") {
            let name = parser.name()?;
            parser.punct("{")?;
            let fields = parser.names_until("}")?;
            case.decls.push(Located {
                line,
                item: DeclItem::Struct { name, fields },
            });
            continue;
        }
        if parser.is_word("extern") {
            let item = parser.extern_fn(Role::Fn)?;
            case.decls.push(Located { line, item });
            continue;
        }
        if parser.is_punct("#")
            && matches!(parser.tokens.get(parser.at + 4).map(|t| &t.tok), Some(Tok::Word(w)) if w == "extern")
        {
            let role = match parser.attribute()?.as_str() {
                "cast" => Role::Cast,
                "conversion" => Role::Conversion,
                other => return Err(format!("line {line}: no attribute {other} on an extern fn")),
            };
            let item = parser.extern_fn(role)?;
            case.decls.push(Located { line, item });
            continue;
        }
        if parser.eat_word("group") {
            let name = parser.name()?;
            let parent = if parser.eat_word("in") {
                Some(parser.name()?)
            } else {
                None
            };
            case.text.push(Located {
                line,
                item: TextItem::Group { name, parent },
            });
            parser.punct("{")?;
            while !parser.eat_punct("}") {
                if parser.peek().is_none() {
                    return parser.error("`}`");
                }
                parser.statement(&mut case.text)?;
            }
            case.text.push(Located {
                line,
                item: TextItem::EndGroup,
            });
            continue;
        }
        parser.statement(&mut case.text)?;
    }
    Ok(case)
}
