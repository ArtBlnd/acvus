use std::fmt;

use acvus_utils::{Astr, QualifiedRef};

pub use crate::literal::{IntWidth, LiteralErrorKind, SuffixedInt};
use crate::span::Span;

acvus_utils::declare_local_id!(pub AstId);

impl AstId {
    pub fn alloc() -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: id + 1 is always >= 1.
        Self(unsafe { std::num::NonZero::new_unchecked(id + 1) })
    }
}

/// What an error node of a tree holds (RFC-0078 rule 4): a parse without
/// errors yields the tree over `Clean`, a parse with errors the tree over
/// `ErrorNode`.
pub trait Slot: Clone + fmt::Debug + PartialEq {
    fn node(&self) -> ErrorNode;
}

/// `acvus-mir` lowers only the tree over this slot, so a tree that holds an
/// error node has no path to the machine (RFC-0078 rule 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Clean {}

impl Slot for Clean {
    fn node(&self) -> ErrorNode {
        match *self {}
    }
}

/// The source a statement, an expression or a pattern that did not parse
/// covered, which the parse replaced by this node and reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorNode {
    pub id: AstId,
    pub span: Span,
}

impl Slot for ErrorNode {
    fn node(&self) -> ErrorNode {
        *self
    }
}

/// A parsed script (standalone expressions with semicolons).
#[derive(Debug, Clone, PartialEq)]
pub struct Script<S = Clean> {
    pub id: AstId,
    pub stmts: Vec<Stmt<S>>,
    pub tail: Option<Box<Expr<S>>>,
    pub span: Span,
}

/// A statement in a script.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt<S = Clean> {
    /// Store into a place: `@a = 0;`, `a.x.y = 0;`, `a[i] = 0;`,
    /// `a.x[i].y = 0;`.
    Store {
        id: AstId,
        place: Place<S>,
        expr: Expr<S>,
        span: Span,
    },
    /// Store through a `&mut`: `*r = 0;`.
    DerefStore {
        id: AstId,
        target: Box<Expr<S>>,
        expr: Expr<S>,
        span: Span,
    },
    Expr(Expr<S>),

    // -- Script mode statements --------------------------------------
    /// `let x = expr;` - new binding (Script mode).
    LetBind {
        id: AstId,
        binder: Binder,
        expr: Expr<S>,
        span: Span,
    },
    /// `let x;` - uninitialized binding (Script mode).
    LetUninit {
        id: AstId,
        binder: Binder,
        span: Span,
    },
    /// `x = expr;` - reassignment to existing binding (Script mode).
    Assign {
        id: AstId,
        name: Astr,
        /// Where `name` is written.
        name_span: Span,
        expr: Expr<S>,
        span: Span,
    },
    /// `while cond { body }` - conditional loop (Script mode).
    While {
        id: AstId,
        cond: Expr<S>,
        body: Vec<Stmt<S>>,
        span: Span,
    },
    /// `for x in head { body }` - one traversal (RFC-0057 rule 1).
    For {
        id: AstId,
        /// Where the head's `as_slice` instance is recorded, as an index
        /// expression records its own (RFC-0047 rule 3).
        callee_id: AstId,
        binder: Binder,
        head: ForHead<S>,
        body: Vec<Stmt<S>>,
        span: Span,
    },
    /// `break;` - leave the innermost loop (RFC-0057 rule 4).
    Break {
        id: AstId,
        span: Span,
    },
    /// `continue;` - start the innermost loop's next iteration.
    Continue {
        id: AstId,
        span: Span,
    },
    /// `while let pattern = source { body }` - pattern loop (Script mode).
    WhileLet {
        id: AstId,
        pattern: Pattern<S>,
        source: Expr<S>,
        body: Vec<Stmt<S>>,
        span: Span,
    },
    /// `anyorder { body }` - the order of effects inside is irrelevant
    /// (RFC-0007). Script mode.
    Anyorder {
        id: AstId,
        body: Vec<Stmt<S>>,
        span: Span,
    },
    /// A template's text line or one of its `{{ }}` tags: the value is
    /// appended to the template's result (RFC-0071 rules 2 and 3).
    Append {
        id: AstId,
        expr: Expr<S>,
        span: Span,
    },
    Error(S),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Place<S = Clean> {
    Base(PlaceBase<S>),
    Field {
        id: AstId,
        object: Box<Place<S>>,
        field: Astr,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlaceBase<S = Clean> {
    Root {
        id: AstId,
        root: Root,
        span: Span,
    },
    /// `place[i]`. The container is not a `Place`, and that is a decision:
    /// an index step settles its `as_slice_mut` instance and lends its
    /// container through the machinery every `a[i]` goes through, a read's
    /// included, and that machinery reads an `Expr` (RFC-0047).
    Element {
        id: AstId,
        callee_id: AstId,
        container: PlaceExpr<S>,
        index: Box<Expr<S>>,
        span: Span,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Root {
    Local(Astr),
    ExternParam(Astr),
    Context(Astr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlaceExpr<S = Clean>(Expr<S>);

impl<S> PlaceExpr<S> {
    pub fn of(expr: Expr<S>) -> Option<Self> {
        match Self::names_a_place(&expr) {
            true => Some(Self(expr)),
            false => None,
        }
    }

    pub fn expr(&self) -> &Expr<S> {
        &self.0
    }

    fn names_a_place(expr: &Expr<S>) -> bool {
        match expr {
            Expr::Ident {
                ref_kind: RefKind::Value | RefKind::ExternParam,
                ..
            }
            | Expr::ContextRef { .. } => true,
            Expr::FieldAccess { object, .. } | Expr::Index { object, .. } => {
                Self::names_a_place(object)
            }
            Expr::Error(_) => false,
            _ => false,
        }
    }
}

impl<S> Place<S>
where
    S: Slot,
{
    pub fn of(expr: Expr<S>) -> Option<Self> {
        match expr {
            Expr::FieldAccess {
                id,
                object,
                field,
                span,
            } => Some(Self::Field {
                id,
                object: Box::new(Self::of(*object)?),
                field,
                span,
            }),
            base => PlaceBase::of(base).map(Self::Base),
        }
    }

    pub fn id(&self) -> AstId {
        match self {
            Self::Base(base) => base.id(),
            Self::Field { id, .. } => *id,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            Self::Base(base) => base.span(),
            Self::Field { span, .. } => *span,
        }
    }
}

impl<S> PlaceBase<S>
where
    S: Slot,
{
    fn of(expr: Expr<S>) -> Option<Self> {
        match expr {
            Expr::Ident {
                id,
                name,
                ref_kind,
                span,
            } => {
                let root = match ref_kind {
                    RefKind::Value => Root::Local(name.name),
                    RefKind::ExternParam => Root::ExternParam(name.name),
                };
                Some(Self::Root { id, root, span })
            }
            Expr::ContextRef { id, name, span } => Some(Self::Root {
                id,
                root: Root::Context(name.name),
                span,
            }),
            Expr::Index {
                id,
                callee_id,
                object,
                index,
                span,
            } => Some(Self::Element {
                id,
                callee_id,
                container: PlaceExpr::of(*object)?,
                index,
                span,
            }),
            Expr::Error(_) => None,
            _ => None,
        }
    }

    pub fn id(&self) -> AstId {
        match self {
            Self::Root { id, .. } | Self::Element { id, .. } => *id,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            Self::Root { span, .. } | Self::Element { span, .. } => *span,
        }
    }
}

/// What a `for` traverses, as the parser reads it. Which of the four heads
/// a `ForHead::Value` is -- `&v`, `&mut v` or an array by value -- is the
/// expression's type, which the checker settles (RFC-0057 rule 1).
#[derive(Debug, Clone, PartialEq)]
pub enum ForHead<S = Clean> {
    Value(Expr<S>),
    Range { lo: Expr<S>, hi: Expr<S> },
}

/// A parsed template: a script whose text lines are output (RFC-0071).
/// Its statements are the script's, and a text line or a `{{ }}` tag is
/// the one statement the script does not write, `Stmt::Append`.
#[derive(Debug, Clone, PartialEq)]
pub struct Template<S = Clean> {
    pub id: AstId,
    pub body: Vec<Stmt<S>>,
    pub span: Span,
}

/// An expression in the template language.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr<S = Clean> {
    /// A reference: `name`, `$name`, or `@name`.
    Ident {
        id: AstId,
        name: QualifiedRef,
        ref_kind: RefKind,
        span: Span,
    },
    /// A literal value.
    Literal {
        id: AstId,
        value: Literal,
        span: Span,
    },
    /// A binary operation: `a + b`.
    BinaryOp {
        id: AstId,
        left: Box<Expr<S>>,
        op: BinOp,
        right: Box<Expr<S>>,
        span: Span,
    },
    /// A unary operation: `-x`, `!x`.
    UnaryOp {
        id: AstId,
        op: UnaryOp,
        operand: Box<Expr<S>>,
        span: Span,
    },
    /// Field access: `a.b`.
    FieldAccess {
        id: AstId,
        object: Box<Expr<S>>,
        field: Astr,
        span: Span,
    },
    /// Index: `a[i]` (RFC-0047). A place, as `a.f` is. `callee_id` is the
    /// id the checker records the `as_slice` / `as_slice_mut` instance
    /// under, the same way a method call records its callee.
    Index {
        id: AstId,
        callee_id: AstId,
        object: Box<Expr<S>>,
        index: Box<Expr<S>>,
        span: Span,
    },
    /// Function call: `f(args)`.
    FuncCall {
        id: AstId,
        func: Box<Expr<S>>,
        args: Vec<Expr<S>>,
        span: Span,
    },
    /// `recv.f(args)`: the call `f(recv', args)` (RFC-0030). `callee_id`
    /// is the id the checker records the callee's type under.
    MethodCall {
        id: AstId,
        callee_id: AstId,
        receiver: Box<Expr<S>>,
        name: Astr,
        /// Where `name` is written, which is where the callee is.
        name_span: Span,
        args: Vec<Expr<S>>,
        span: Span,
    },
    /// Pipe: `expr | func`.
    Pipe {
        id: AstId,
        left: Box<Expr<S>>,
        right: Box<Expr<S>>,
        span: Span,
    },
    /// Lambda: `|x| -> expr` or `|x, y| -> expr`.
    Lambda {
        id: AstId,
        params: Vec<Binder>,
        body: Box<Expr<S>>,
        span: Span,
    },
    /// Parenthesized expression: `(expr)`.
    Paren {
        id: AstId,
        inner: Box<Expr<S>>,
        span: Span,
    },
    /// A reference to a place: `&place` or `&mut place` (RFC-0018).
    Borrow {
        id: AstId,
        mutable: bool,
        place: Box<Expr<S>>,
        span: Span,
    },
    /// A list: `[a, b, c]`, `[a, b, ..]`, `[.., a, b]`, `[a, .., b]`.
    /// `rest` is `Some` if `..` is present. `head` is before `..`, `tail` is after.
    /// If no `..`, all elements are in `head` and `tail` is empty.
    List {
        id: AstId,
        head: Vec<Expr<S>>,
        rest: Option<Span>,
        tail: Vec<Expr<S>>,
        span: Span,
    },
    /// A group used for lambda parameter lists: `(a, b)`.
    /// This is a temporary node that only appears as the LHS of `->`.
    Group {
        id: AstId,
        elements: Vec<Expr<S>>,
        span: Span,
    },
    /// An object literal: `{ field1, $field2, field3 }`.
    Object {
        id: AstId,
        fields: Vec<ObjectExprField<S>>,
        span: Span,
    },
    /// A tuple: `(a, b, c)` - 0 or 2+ elements.
    /// Elements can be expressions or wildcards `_`.
    Tuple {
        id: AstId,
        elements: Vec<TupleElem<S>>,
        span: Span,
    },
    /// A block expression: `{ stmt; stmt; expr }`.
    Block {
        id: AstId,
        stmts: Vec<Stmt<S>>,
        tail: Box<Expr<S>>,
        span: Span,
    },
    /// A context reference: `@name`.
    ContextRef {
        id: AstId,
        name: QualifiedRef,
        span: Span,
    },
    /// A variant constructor: `Some(expr)`, `None`, or `Color::Red`.
    Variant {
        id: AstId,
        enum_name: Option<Astr>,
        tag: Astr,
        payload: Option<Box<Expr<S>>>,
        span: Span,
    },
    /// `expr as T`, whose value is Rust's `as` at `target`'s type
    /// (RFC-0049).
    Cast {
        id: AstId,
        expr: Box<Expr<S>>,
        target: Astr,
        target_span: Span,
        span: Span,
    },
    /// `inner?`: the `Ok` or `Some` payload, or an early return of the
    /// `Err` or `None` (RFC-0038).
    Try {
        id: AstId,
        inner: Box<Expr<S>>,
        span: Span,
    },
    /// `return value`: leave the enclosing body -- a script or a lambda --
    /// with `value`, from any depth. Its own type is `!`.
    Return {
        id: AstId,
        value: Box<Expr<S>>,
        span: Span,
    },

    // -- Script mode expressions -------------------------------------
    /// `if cond { body; tail } else { ... }` - conditional expression (Script mode).
    If {
        id: AstId,
        cond: Box<Expr<S>>,
        then_body: Vec<Stmt<S>>,
        then_tail: Option<Box<Expr<S>>>,
        else_branch: Option<Box<ElseBranch<S>>>,
        span: Span,
    },
    /// `match scrutinee { P => e, .. }` - one dispatch over the scrutinee
    /// (RFC-0051). The scrutinee is evaluated once; every arm has the type
    /// of the whole.
    Match {
        id: AstId,
        scrutinee: Box<Expr<S>>,
        arms: Vec<MatchExprArm<S>>,
        span: Span,
    },
    /// `if let pattern = source { body; tail } else { ... }` - pattern match expression (Script mode).
    IfLet {
        id: AstId,
        pattern: Pattern<S>,
        source: Box<Expr<S>>,
        then_body: Vec<Stmt<S>>,
        then_tail: Option<Box<Expr<S>>>,
        else_branch: Option<Box<ElseBranch<S>>>,
        span: Span,
    },
    Error(S),
}

/// One arm of a `match` expression: `P => e` or `P => { stmts; }`. An arm
/// with no tail is typed `Unit`, as an `if` branch with no tail is.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchExprArm<S = Clean> {
    pub id: AstId,
    pub pattern: Pattern<S>,
    pub body: Vec<Stmt<S>>,
    pub tail: Option<Box<Expr<S>>>,
    pub span: Span,
}

/// An else branch in an `if` / `if let` expression.
#[derive(Debug, Clone, PartialEq)]
pub enum ElseBranch<S = Clean> {
    /// `else if ...` or `else if let ...` - chains to another conditional.
    ElseIf(Expr<S>),
    /// `else { body; tail }` - terminal else block.
    Else {
        body: Vec<Stmt<S>>,
        tail: Option<Box<Expr<S>>>,
        span: Span,
    },
}

/// An element in a tuple expression: either a real expression or a wildcard `_`.
#[derive(Debug, Clone, PartialEq)]
pub enum TupleElem<S = Clean> {
    Expr(Expr<S>),
    Wildcard(Span),
}

impl<S> Expr<S>
where
    S: Slot,
{
    pub fn id(&self) -> AstId {
        match self {
            Expr::Ident { id, .. }
            | Expr::Literal { id, .. }
            | Expr::BinaryOp { id, .. }
            | Expr::UnaryOp { id, .. }
            | Expr::FieldAccess { id, .. }
            | Expr::Index { id, .. }
            | Expr::FuncCall { id, .. }
            | Expr::MethodCall { id, .. }
            | Expr::Pipe { id, .. }
            | Expr::Lambda { id, .. }
            | Expr::Paren { id, .. }
            | Expr::Cast { id, .. }
            | Expr::Try { id, .. }
            | Expr::Return { id, .. }
            | Expr::Borrow { id, .. }
            | Expr::List { id, .. }
            | Expr::Group { id, .. }
            | Expr::Object { id, .. }
            | Expr::Tuple { id, .. }
            | Expr::ContextRef { id, .. }
            | Expr::Variant { id, .. }
            | Expr::Block { id, .. }
            | Expr::If { id, .. }
            | Expr::Match { id, .. }
            | Expr::IfLet { id, .. } => *id,
            Expr::Error(node) => node.node().id,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            Expr::Ident { span, .. }
            | Expr::Literal { span, .. }
            | Expr::BinaryOp { span, .. }
            | Expr::UnaryOp { span, .. }
            | Expr::FieldAccess { span, .. }
            | Expr::Index { span, .. }
            | Expr::FuncCall { span, .. }
            | Expr::MethodCall { span, .. }
            | Expr::Pipe { span, .. }
            | Expr::Lambda { span, .. }
            | Expr::Paren { span, .. }
            | Expr::Cast { span, .. }
            | Expr::Try { span, .. }
            | Expr::Return { span, .. }
            | Expr::Borrow { span, .. }
            | Expr::List { span, .. }
            | Expr::Group { span, .. }
            | Expr::Object { span, .. }
            | Expr::Tuple { span, .. }
            | Expr::ContextRef { span, .. }
            | Expr::Variant { span, .. }
            | Expr::Block { span, .. }
            | Expr::If { span, .. }
            | Expr::Match { span, .. }
            | Expr::IfLet { span, .. } => *span,
            Expr::Error(node) => node.node().span,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Binder {
    pub id: AstId,
    pub name: Astr,
    pub span: Span,
}

/// A field in an object expression.
/// Shorthand `{ name }` -> key="name", value=Ident("name", Value).
/// Shorthand `{ $name }` -> key="name", value=Ident("name", Variable).
/// Shorthand `{ @name }` -> key="name", value=Ident("name", Context).
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectExprField<S = Clean> {
    pub id: AstId,
    pub key: Astr,
    pub value: Expr<S>,
    pub span: Span,
}

/// A pattern used on the LHS of `=` in a match block.
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern<S = Clean> {
    /// A binding that captures a value: `item` or `$name`.
    Binding {
        id: AstId,
        name: Astr,
        ref_kind: RefKind,
        span: Span,
    },
    /// A context binding: `@name`.
    ContextBind {
        id: AstId,
        name: QualifiedRef,
        span: Span,
    },
    /// A literal pattern that filters: `true`, `"admin"`, `42`.
    Literal {
        id: AstId,
        value: Literal,
        span: Span,
    },
    /// A list pattern: `[a, b, c]`, `[a, b, ..]`, `[.., a, b]`, `[a, .., b]`.
    /// Same structure as `Expr::List`: `head` before `..`, `tail` after.
    List {
        id: AstId,
        head: Vec<Pattern<S>>,
        rest: Option<Span>,
        tail: Vec<Pattern<S>>,
        span: Span,
    },
    /// An object pattern: `{ name, $value, status: "active" }`.
    Object {
        id: AstId,
        fields: Vec<ObjectPatternField<S>>,
        span: Span,
    },
    /// A tuple pattern: `(a, b, c)`.
    Tuple {
        id: AstId,
        elements: Vec<TuplePatternElem<S>>,
        span: Span,
    },
    /// `_`: the position is not read and nothing binds. A `match` arm that
    /// is a bare `_` is the catch-all (RFC-0051).
    Wildcard {
        id: AstId,
        span: Span,
    },
    /// A variant pattern: `Some(inner)`, `None`, or `Color::Red`.
    Variant {
        id: AstId,
        enum_name: Option<Astr>,
        tag: Astr,
        payload: Option<Box<Pattern<S>>>,
        span: Span,
    },
    Error(S),
}

impl<S> Pattern<S>
where
    S: Slot,
{
    pub fn id(&self) -> AstId {
        match self {
            Pattern::Binding { id, .. }
            | Pattern::Wildcard { id, .. }
            | Pattern::ContextBind { id, .. }
            | Pattern::Literal { id, .. }
            | Pattern::List { id, .. }
            | Pattern::Object { id, .. }
            | Pattern::Tuple { id, .. }
            | Pattern::Variant { id, .. } => *id,
            Pattern::Error(node) => node.node().id,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            Pattern::Binding { span, .. }
            | Pattern::Wildcard { span, .. }
            | Pattern::ContextBind { span, .. }
            | Pattern::Literal { span, .. }
            | Pattern::List { span, .. }
            | Pattern::Object { span, .. }
            | Pattern::Tuple { span, .. }
            | Pattern::Variant { span, .. } => *span,
            Pattern::Error(node) => node.node().span,
        }
    }
}

/// An element in a tuple pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum TuplePatternElem<S = Clean> {
    /// A sub-pattern.
    Pattern(Pattern<S>),
    /// A wildcard `_` that ignores the element.
    Wildcard(Span),
}

/// A field in an object pattern: `{ key: pattern }` or shorthand `{ name }` / `{ $name }`.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectPatternField<S = Clean> {
    pub id: AstId,
    pub key: Astr,
    pub pattern: Pattern<S>,
    pub span: Span,
}

/// A binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    And,
    Or,
    Xor,
    BitAnd,
    BitOr,
    Shl,
    Shr,
    Mod,
}

/// A unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    /// `*r`: the primitive a reference names (RFC-0018).
    Deref,
}

/// The kind of reference for an identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefKind {
    /// A bare name: `x`.
    Value,
    /// An extern parameter: `$x` (immutable, externally injected).
    ExternParam,
}

/// A literal value.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    String(String),
    Int(i128),
    /// `10u64` (RFC-0058).
    IntOf(SuffixedInt),
    Float(f64),
    /// `'c'` (RFC-0058).
    Char(char),
    /// `b"…"` (RFC-0058).
    Bytes(Vec<u8>),
    Bool(bool),
    List(Vec<Literal>),
    Unit,
}

impl Literal {
    /// The same value in the literal forms that predate RFC-0058: a
    /// suffixed integer as its value, a byte string as the list of its
    /// bytes. A `Char` is already such a form and stays one.
    ///
    /// The width a suffix named and the `u8` a byte string's elements
    /// have are the literal's *type*, which RFC-0037 keeps beside the
    /// value rather than in it; a reader that has the type has no use for
    /// the suffix, and the ones that do not are below the checker.
    pub fn desugared(&self) -> Literal {
        match self {
            Literal::IntOf(suffixed) => Literal::Int(suffixed.value),
            Literal::Bytes(bytes) => {
                Literal::List(bytes.iter().map(|b| Literal::Int(i128::from(*b))).collect())
            }
            already => already.clone(),
        }
    }
}

// -- AST walk: context reference extraction --------------------------

/// Extract all `@name` context references from a Script AST.
pub fn extract_script_context_refs<S>(script: &Script<S>) -> rustc_hash::FxHashSet<QualifiedRef> {
    script_context_refs(script, true)
}

pub fn direct_script_context_refs<S>(script: &Script<S>) -> rustc_hash::FxHashSet<QualifiedRef> {
    script_context_refs(script, false)
}

pub fn direct_expr_context_refs<S>(expr: &Expr<S>) -> rustc_hash::FxHashSet<QualifiedRef> {
    let mut refs = ContextRefs::new(false);
    walk_expr(expr, &mut refs);
    refs.set
}

fn script_context_refs<S>(
    script: &Script<S>,
    into_lambdas: bool,
) -> rustc_hash::FxHashSet<QualifiedRef> {
    let mut refs = ContextRefs::new(into_lambdas);
    walk_stmts(&script.stmts, &mut refs);
    if let Some(tail) = &script.tail {
        walk_expr(tail, &mut refs);
    }
    refs.set
}

struct ContextRefs {
    set: rustc_hash::FxHashSet<QualifiedRef>,
    into_lambdas: bool,
}

impl ContextRefs {
    fn new(into_lambdas: bool) -> Self {
        Self {
            set: rustc_hash::FxHashSet::default(),
            into_lambdas,
        }
    }
}

fn walk_stmts<S>(stmts: &[Stmt<S>], refs: &mut ContextRefs) {
    for stmt in stmts {
        match stmt {
            Stmt::Store { place, expr, .. } => {
                walk_place(place, refs);
                walk_expr(expr, refs);
            }
            Stmt::DerefStore { target, expr, .. } => {
                walk_expr(target, refs);
                walk_expr(expr, refs);
            }
            Stmt::Expr(expr) => walk_expr(expr, refs),
            // Script mode statements
            Stmt::LetBind { expr, .. } | Stmt::Assign { expr, .. } => walk_expr(expr, refs),
            Stmt::LetUninit { .. } => {}
            Stmt::WhileLet {
                pattern,
                source,
                body,
                ..
            } => {
                walk_pattern(pattern, refs);
                walk_expr(source, refs);
                walk_stmts(body, refs);
            }
            Stmt::While { cond, body, .. } => {
                walk_expr(cond, refs);
                walk_stmts(body, refs);
            }
            Stmt::For { head, body, .. } => {
                match head {
                    ForHead::Value(e) => walk_expr(e, refs),
                    ForHead::Range { lo, hi } => {
                        walk_expr(lo, refs);
                        walk_expr(hi, refs);
                    }
                }
                walk_stmts(body, refs);
            }
            Stmt::Break { .. } | Stmt::Continue { .. } => {}
            Stmt::Anyorder { body, .. } => walk_stmts(body, refs),
            Stmt::Append { expr, .. } => walk_expr(expr, refs),
            Stmt::Error(_) => {}
        }
    }
}

/// Extract all `@name` context references from a Template AST.
pub fn extract_template_context_refs<S>(
    template: &Template<S>,
) -> rustc_hash::FxHashSet<QualifiedRef> {
    template_context_refs(template, true)
}

pub fn direct_template_context_refs<S>(
    template: &Template<S>,
) -> rustc_hash::FxHashSet<QualifiedRef> {
    template_context_refs(template, false)
}

fn template_context_refs<S>(
    template: &Template<S>,
    into_lambdas: bool,
) -> rustc_hash::FxHashSet<QualifiedRef> {
    let mut refs = ContextRefs::new(into_lambdas);
    walk_stmts(&template.body, &mut refs);
    refs.set
}

fn walk_pattern<S>(pattern: &Pattern<S>, refs: &mut ContextRefs) {
    match pattern {
        Pattern::ContextBind { name, .. } => {
            refs.set.insert(*name);
        }
        Pattern::Binding { .. }
        | Pattern::Wildcard { .. }
        | Pattern::Literal { .. }
        | Pattern::Error(_) => {}
        Pattern::List { head, tail, .. } => {
            for p in head {
                walk_pattern(p, refs);
            }
            for p in tail {
                walk_pattern(p, refs);
            }
        }
        Pattern::Object { fields, .. } => {
            for f in fields {
                walk_pattern(&f.pattern, refs);
            }
        }
        Pattern::Tuple { elements, .. } => {
            for e in elements {
                match e {
                    TuplePatternElem::Pattern(p) => walk_pattern(p, refs),
                    TuplePatternElem::Wildcard(_) => {}
                }
            }
        }
        Pattern::Variant { payload, .. } => {
            if let Some(p) = payload {
                walk_pattern(p, refs);
            }
        }
    }
}

fn walk_place<S>(place: &Place<S>, refs: &mut ContextRefs) {
    match place {
        Place::Field { object, .. } => walk_place(object, refs),
        Place::Base(PlaceBase::Root {
            root: Root::Context(name),
            ..
        }) => {
            refs.set.insert(QualifiedRef::root(*name));
        }
        Place::Base(PlaceBase::Root { .. }) => {}
        Place::Base(PlaceBase::Element {
            container, index, ..
        }) => {
            walk_expr(container.expr(), refs);
            walk_expr(index, refs);
        }
    }
}

fn walk_expr<S>(expr: &Expr<S>, refs: &mut ContextRefs) {
    match expr {
        Expr::ContextRef { name, .. } => {
            refs.set.insert(*name);
        }
        Expr::Ident { .. } | Expr::Literal { .. } | Expr::Error(_) => {}
        Expr::Variant { payload, .. } => {
            if let Some(payload) = payload {
                walk_expr(payload, refs);
            }
        }
        Expr::BinaryOp { left, right, .. } | Expr::Pipe { left, right, .. } => {
            walk_expr(left, refs);
            walk_expr(right, refs);
        }
        Expr::UnaryOp { operand, .. }
        | Expr::Paren { inner: operand, .. }
        | Expr::Cast { expr: operand, .. }
        | Expr::Try { inner: operand, .. }
        | Expr::Return { value: operand, .. }
        | Expr::Borrow { place: operand, .. } => {
            walk_expr(operand, refs);
        }
        Expr::FieldAccess { object, .. } => walk_expr(object, refs),
        Expr::Index { object, index, .. } => {
            walk_expr(object, refs);
            walk_expr(index, refs);
        }
        Expr::MethodCall { receiver, args, .. } => {
            walk_expr(receiver, refs);
            for a in args {
                walk_expr(a, refs);
            }
        }
        Expr::FuncCall { func, args, .. } => {
            walk_expr(func, refs);
            for arg in args {
                walk_expr(arg, refs);
            }
        }
        Expr::Lambda { body, .. } => {
            if refs.into_lambdas {
                walk_expr(body, refs);
            }
        }
        Expr::List { head, tail, .. } => {
            for e in head {
                walk_expr(e, refs);
            }
            for e in tail {
                walk_expr(e, refs);
            }
        }
        Expr::Group { elements, .. } => {
            for e in elements {
                walk_expr(e, refs);
            }
        }
        Expr::Object { fields, .. } => {
            for f in fields {
                walk_expr(&f.value, refs);
            }
        }
        Expr::Tuple { elements, .. } => {
            for e in elements {
                if let TupleElem::Expr(expr) = e {
                    walk_expr(expr, refs);
                }
            }
        }
        Expr::Block { stmts, tail, .. } => {
            walk_stmts(stmts, refs);
            walk_expr(tail, refs);
        }
        Expr::If {
            cond,
            then_body,
            then_tail,
            else_branch,
            ..
        } => {
            walk_expr(cond, refs);
            walk_stmts(then_body, refs);
            if let Some(tail) = then_tail {
                walk_expr(tail, refs);
            }
            if let Some(eb) = else_branch {
                walk_else_branch(eb, refs);
            }
        }
        Expr::Match {
            scrutinee, arms, ..
        } => {
            walk_expr(scrutinee, refs);
            for arm in arms {
                walk_pattern(&arm.pattern, refs);
                walk_stmts(&arm.body, refs);
                if let Some(tail) = &arm.tail {
                    walk_expr(tail, refs);
                }
            }
        }
        Expr::IfLet {
            pattern,
            source,
            then_body,
            then_tail,
            else_branch,
            ..
        } => {
            walk_pattern(pattern, refs);
            walk_expr(source, refs);
            walk_stmts(then_body, refs);
            if let Some(tail) = then_tail {
                walk_expr(tail, refs);
            }
            if let Some(eb) = else_branch {
                walk_else_branch(eb, refs);
            }
        }
    }
}

fn walk_else_branch<S>(eb: &ElseBranch<S>, refs: &mut ContextRefs) {
    match eb {
        ElseBranch::ElseIf(expr) => walk_expr(expr, refs),
        ElseBranch::Else { body, tail, .. } => {
            walk_stmts(body, refs);
            if let Some(tail) = tail {
                walk_expr(tail, refs);
            }
        }
    }
}
