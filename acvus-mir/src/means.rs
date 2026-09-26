//! What a call computes, stated by its declaration as a term (RFC-0104).
//! `#[extern_fn(means(..))]` writes it, a reader reads a call as its term,
//! and `acvus-interpreter-test`'s model checker holds the handler to it.
//!
//! A parameter is numbered as the declaration's acvus parameters are,
//! which is the order of a call's arguments. A term names at most one entry
//! `x[k]`, the one its declaration's `reaches` names (RFC-0082 rule 7).
//!
//! The readers here are symbolic: [`entry_access`] evaluates a term over
//! the two states an entry can be in, absent or holding a payload `b`, and
//! says what the call does to the entry in each; nothing reads a name.

pub use acvus_ast::BinOp;
use acvus_ast::Literal;
use acvus_utils::QualifiedRef;

use crate::ty::Mutability;

#[derive(Debug, Clone, PartialEq)]
pub struct Means {
    /// The entry `x[k]` the term reads or writes, where it names one.
    pub entry: Option<EntryParams>,
    pub stmts: Vec<Stmt>,
    /// The call's result.
    pub result: Term,
}

/// `x[k]`: `table` numbers the reference parameter `x`, `key` the key
/// parameter `k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntryParams {
    pub table: usize,
    pub key: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// `let name = t;`, bound as [`Term::Local`] `local`.
    Let { local: usize, value: Term },
    /// `x[k] = t;`: the entry holds `t` from here on. A map's entry is an
    /// `Option<V>`, `None` where the table holds no value at the key; a
    /// set's is a `bool`.
    Store(Term),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// A parameter taken by value.
    Param(usize),
    /// `*x`: the value reference parameter `x` lends.
    Lent(usize),
    /// A `let` binding, or the payload a `match` arm binds.
    Local(usize),
    /// `x[k]`: the entry's value.
    Entry,
    /// `&x[k]` or `&mut x[k]`: the entry lent as its borrowed view,
    /// `Option<&V>`.
    LendEntry(Mutability),
    Some(Box<Term>),
    None,
    Const(Literal),
    /// `!t` over a `bool`.
    Not(Box<Term>),
    /// `-t` over a number.
    Neg(Box<Term>),
    /// `a op b`, one of the language's binary operators, as the language
    /// computes it (an overflow traps, RFC-0037 rule 3).
    Binary {
        op: BinOp,
        left: Box<Term>,
        right: Box<Term>,
    },
    /// `[t, ..]`: an array.
    Array(Vec<Term>),
    /// A registered extern, named as RFC-0082 rule 2 names one.
    Call { name: QualifiedRef, args: Vec<Term> },
    /// `match t { Some(b) => a, None => c }`, `b` bound as `some` where the
    /// arm names it.
    Match {
        scrutinee: Box<Term>,
        some: Option<usize>,
        then: Box<Term>,
        none: Box<Term>,
    },
}

impl Means {
    pub fn named_externs(&self) -> Vec<QualifiedRef> {
        let mut named = Vec::new();
        let terms = self
            .stmts
            .iter()
            .map(|stmt| match stmt {
                Stmt::Let { value, .. } | Stmt::Store(value) => value,
            })
            .chain([&self.result]);
        for term in terms {
            term.visit(&mut |term| {
                if let Term::Call { name, .. } = term {
                    named.push(*name);
                }
            });
        }
        named
    }

    /// `x` of a term that is `*x` alone.
    pub fn lent_value(&self) -> Option<usize> {
        match (&self.entry, &self.stmts[..], &self.result) {
            (None, [], Term::Lent(param)) => Some(*param),
            _ => None,
        }
    }

    pub fn closed(&self) -> Option<&Term> {
        match (&self.entry, &self.stmts[..]) {
            (None, []) if !self.result.reads_params() => Some(&self.result),
            _ => None,
        }
    }
}

impl Term {
    pub fn visit<'t>(&'t self, on: &mut impl FnMut(&'t Term)) {
        on(self);
        match self {
            Term::Param(_)
            | Term::Lent(_)
            | Term::Local(_)
            | Term::Entry
            | Term::LendEntry(_)
            | Term::None
            | Term::Const(_) => {}
            Term::Some(inner) | Term::Not(inner) | Term::Neg(inner) => inner.visit(on),
            Term::Binary { left, right, .. } => {
                left.visit(on);
                right.visit(on);
            }
            Term::Array(items) | Term::Call { args: items, .. } => {
                for item in items {
                    item.visit(on);
                }
            }
            Term::Match {
                scrutinee,
                then,
                none,
                ..
            } => {
                scrutinee.visit(on);
                then.visit(on);
                none.visit(on);
            }
        }
    }

    fn reads_params(&self) -> bool {
        let mut reads = false;
        self.visit(&mut |term| {
            reads |= matches!(
                term,
                Term::Param(_) | Term::Lent(_) | Term::Entry | Term::LendEntry(_)
            );
        });
        reads
    }
}

// -- The symbolic reading --------------------------------------------------

/// A call a term names is read by its own term, or by the `payload` law
/// (RFC-0082 rule 3); any other stays an opaque call.
#[derive(Debug, Clone, Copy)]
pub enum Named<'m> {
    Means(&'m Means),
    Payload,
}

/// A set's `bool` is read as `Option<()>`, so a set's entry and a map's
/// meet one reading: `true` is `Opt(Some(Unit))`.
#[derive(Debug, Clone, PartialEq)]
enum Sym {
    Param(usize),
    Lent(usize),
    /// What a present entry held where the call began.
    Payload,
    Unit,
    Opt(Option<Box<Sym>>),
    Const(Literal),
    RefPayload(Mutability),
    Array(Vec<Sym>),
    Call { name: QualifiedRef, args: Vec<Sym> },
    /// An operator's value, read by its structure alone.
    Neg(Box<Sym>),
    Binary {
        op: BinOp,
        left: Box<Sym>,
        right: Box<Sym>,
    },
}

impl Sym {
    fn reads_payload(&self) -> bool {
        match self {
            Sym::Payload | Sym::RefPayload(_) => true,
            Sym::Param(_) | Sym::Lent(_) | Sym::Unit | Sym::Const(_) => false,
            Sym::Opt(inner) => inner.as_ref().is_some_and(|inner| inner.reads_payload()),
            Sym::Array(items) | Sym::Call { args: items, .. } => {
                items.iter().any(Sym::reads_payload)
            }
            Sym::Neg(inner) => inner.reads_payload(),
            Sym::Binary { left, right, .. } => left.reads_payload() || right.reads_payload(),
        }
    }
}

struct Frame {
    /// What a nested term's call passed for each parameter; `None` for the
    /// call's own term, whose parameters are themselves.
    passed: Option<Vec<Sym>>,
    locals: Vec<Option<Sym>>,
}

struct Eval<'r> {
    named: &'r dyn Fn(QualifiedRef) -> Option<Named<'r>>,
    entry: Sym,
    nested_depth: usize,
}

/// Terms that name each other in a cycle have no reading; the bound ends
/// the walk rather than detecting the cycle.
const DEEPEST_NESTING: usize = 8;

impl Eval<'_> {
    fn stmts(&mut self, means: &Means, frame: &mut Frame) -> Option<Sym> {
        for stmt in &means.stmts {
            match stmt {
                Stmt::Let { local, value } => {
                    let value = self.term(value, frame)?;
                    set_local(frame, *local, value);
                }
                Stmt::Store(value) => self.entry = self.term(value, frame)?,
            }
        }
        self.term(&means.result, frame)
    }

    fn term(&mut self, term: &Term, frame: &mut Frame) -> Option<Sym> {
        Some(match term {
            Term::Param(at) => match &frame.passed {
                None => Sym::Param(*at),
                Some(passed) => passed.get(*at)?.clone(),
            },
            Term::Lent(at) => match &frame.passed {
                None => Sym::Lent(*at),
                Some(passed) => match passed.get(*at)? {
                    Sym::RefPayload(_) => self.current_payload()?,
                    _ => return None,
                },
            },
            Term::Local(at) => frame.locals.get(*at)?.clone()?,
            Term::Entry => self.entry.clone(),
            Term::LendEntry(mutability) => match &self.entry {
                Sym::Opt(None) => Sym::Opt(None),
                Sym::Opt(Some(_)) => Sym::Opt(Some(Box::new(Sym::RefPayload(*mutability)))),
                _ => return None,
            },
            Term::Some(inner) => Sym::Opt(Some(Box::new(self.term(inner, frame)?))),
            Term::None => Sym::Opt(None),
            Term::Const(Literal::Bool(true)) => Sym::Opt(Some(Box::new(Sym::Unit))),
            Term::Const(Literal::Bool(false)) => Sym::Opt(None),
            Term::Const(literal) => Sym::Const(literal.clone()),
            Term::Not(inner) => match self.term(inner, frame)? {
                Sym::Opt(None) => Sym::Opt(Some(Box::new(Sym::Unit))),
                Sym::Opt(Some(_)) => Sym::Opt(None),
                _ => return None,
            },
            Term::Neg(inner) => Sym::Neg(Box::new(self.term(inner, frame)?)),
            Term::Binary { op, left, right } => Sym::Binary {
                op: *op,
                left: Box::new(self.term(left, frame)?),
                right: Box::new(self.term(right, frame)?),
            },
            Term::Array(items) => Sym::Array(
                items
                    .iter()
                    .map(|item| self.term(item, frame))
                    .collect::<Option<_>>()?,
            ),
            Term::Call { name, args } => {
                let args: Vec<Sym> = args
                    .iter()
                    .map(|arg| self.term(arg, frame))
                    .collect::<Option<_>>()?;
                match (self.named)(*name) {
                    None => Sym::Call { name: *name, args },
                    Some(Named::Payload) => match args.as_slice() {
                        [Sym::Opt(Some(payload))] => (**payload).clone(),
                        _ => return None,
                    },
                    Some(Named::Means(means)) => {
                        if means.entry.is_some() || self.nested_depth >= DEEPEST_NESTING {
                            return None;
                        }
                        let mut nested = Frame {
                            passed: Some(args),
                            locals: Vec::new(),
                        };
                        self.nested_depth += 1;
                        let value = self.stmts(means, &mut nested);
                        self.nested_depth -= 1;
                        value?
                    }
                }
            }
            Term::Match {
                scrutinee,
                some,
                then,
                none,
            } => match self.term(scrutinee, frame)? {
                Sym::Opt(Some(payload)) => {
                    if let Some(local) = some {
                        set_local(frame, *local, *payload);
                    }
                    self.term(then, frame)?
                }
                Sym::Opt(None) => self.term(none, frame)?,
                _ => return None,
            },
        })
    }

    fn current_payload(&self) -> Option<Sym> {
        match &self.entry {
            Sym::Opt(Some(payload)) => Some((**payload).clone()),
            _ => None,
        }
    }
}

fn set_local(frame: &mut Frame, local: usize, value: Sym) {
    if frame.locals.len() <= local {
        frame.locals.resize(local + 1, None);
    }
    frame.locals[local] = Some(value);
}

struct Outcome {
    entry: Sym,
    result: Sym,
}

struct Cases {
    began_absent: Outcome,
    began_present: Outcome,
}

fn cases<'r>(means: &Means, named: &'r dyn Fn(QualifiedRef) -> Option<Named<'r>>) -> Option<Cases> {
    means.entry?;
    let run = |entry: Sym| {
        let mut eval = Eval {
            named,
            entry,
            nested_depth: 0,
        };
        let mut frame = Frame {
            passed: None,
            locals: Vec::new(),
        };
        let result = eval.stmts(means, &mut frame)?;
        Some(Outcome {
            entry: eval.entry,
            result,
        })
    };
    Some(Cases {
        began_absent: run(Sym::Opt(None))?,
        began_present: run(Sym::Opt(Some(Box::new(Sym::Payload))))?,
    })
}

/// What one call does to the entry `x[k]` its term names (RFC-0098 rule 5),
/// read off the term over both states the entry can be in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryAccess {
    /// Leaves a present entry, makes an absent one `Some(default)`, and
    /// returns a `&mut` to its payload: a map's `or_insert`.
    Opens { default: usize },
    /// Changes nothing and returns the entry's borrowed view: a map's
    /// `get` (`Shared`) and `get_mut` (`Mut`).
    Views(Mutability),
    /// Leaves the entry holding `stored` whatever it held; the result reads
    /// what it held where `result_reads_entry`.
    Stores {
        stored: Stored,
        result_reads_entry: bool,
    },
    /// Changes nothing, and the result reads the entry.
    Reads,
}

/// A value an entry holds after a call whatever it held before, reading
/// none of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stored {
    /// `Some(v)`, `v` the parameter numbered so.
    Payload(usize),
    /// A set's `true`.
    Marked,
    /// A map's `None` or a set's `false`: the entry is taken out.
    Absent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntryReading {
    pub entry: EntryParams,
    pub access: EntryAccess,
}

/// `None` for a term that names no entry, or does to it what no
/// [`EntryAccess`] is.
pub fn entry_access<'r>(
    means: &Means,
    named: &'r dyn Fn(QualifiedRef) -> Option<Named<'r>>,
) -> Option<EntryReading> {
    let entry = means.entry?;
    let Cases {
        began_absent: absent,
        began_present: present,
    } = cases(means, named)?;
    let unchanged =
        absent.entry == Sym::Opt(None) && present.entry == Sym::Opt(Some(Box::new(Sym::Payload)));
    let result_reads_entry = absent.result != present.result || present.result.reads_payload();
    let access = if unchanged {
        let views = |mutability| {
            absent.result == Sym::Opt(None)
                && present.result == Sym::Opt(Some(Box::new(Sym::RefPayload(mutability))))
        };
        if views(Mutability::Shared) {
            EntryAccess::Views(Mutability::Shared)
        } else if views(Mutability::Mut) {
            EntryAccess::Views(Mutability::Mut)
        } else if result_reads_entry {
            EntryAccess::Reads
        } else {
            return None;
        }
    } else if let (Sym::Opt(Some(opened)), Sym::Opt(Some(kept))) = (&absent.entry, &present.entry)
        && let Sym::Param(default) = **opened
        && **kept == Sym::Payload
        && absent.result == Sym::RefPayload(Mutability::Mut)
        && present.result == Sym::RefPayload(Mutability::Mut)
    {
        EntryAccess::Opens { default }
    } else if absent.entry == present.entry && !absent.entry.reads_payload() {
        let stored = match &absent.entry {
            Sym::Opt(None) => Stored::Absent,
            Sym::Opt(Some(held)) => match **held {
                Sym::Param(at) => Stored::Payload(at),
                Sym::Unit => Stored::Marked,
                _ => return None,
            },
            _ => return None,
        };
        EntryAccess::Stores {
            stored,
            result_reads_entry,
        }
    } else {
        return None;
    };
    Some(EntryReading { entry, access })
}
