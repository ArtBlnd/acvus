use acvus_ast::{Literal, Span};
use acvus_utils::LocalFactory;
use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::QualifiedRef;
use crate::ty::{CastTy, Mutability, Task, Ty};

/// A call that is an instruction of the language (RFC-0020): the
/// compiler's own instance of a shared signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Intrinsic {
    StringClone,
}

acvus_utils::declare_local_id!(pub ValueId);

/// What an integer `+`, `-`, `*`, `<<`, `>>` or negation does where its
/// exact result does not fit its width, or where a shift's amount is the
/// width or more (RFC-0037 rule 3).
///
/// It is a required field of each operation that has the question, and of
/// no other, so a match on one of them names which kind it handles, and a
/// pass reads the kind here rather than working out where the operation
/// came from. At `f64` the question does not arise: both kinds are the IEEE
/// operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Overflow {
    /// The source's operation. The program means nothing where it
    /// overflows, and a run that reaches such an overflow ends there with
    /// Rust's text. Only the lowering writes it; a pass that rewrites an
    /// operation into an equal one keeps it, and no pass writes it anew.
    Trap,
    /// An operation a pass writes, which the program never asked for: the
    /// result modulo `2^width`, and a shift's amount modulo the width, as
    /// Rust's `wrapping_*`.
    Wrap,
}

/// A binary operation of the MIR: the source's operators, and `Min` and
/// `Max`, which no source expression lowers to and only a pass writes.
///
/// It is its own enum, and not `acvus_ast::BinOp`, so that the parser
/// cannot write `Min` or `Max`: the parser's type has no such variant, and
/// the lowering reaches this enum through `From<acvus_ast::BinOp>`, which
/// yields neither. The same conversion is where a source operator becomes
/// an [`Overflow::Trap`] operation, the one kind the source writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add(Overflow),
    Sub(Overflow),
    Mul(Overflow),
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
    Shl(Overflow),
    Shr(Overflow),
    Mod,
    /// The lesser of two integers of one width, compared at that width's
    /// signedness. Total at every width: it neither wraps nor traps.
    Min,
    /// The greater of two integers of one width, as `Min` is the lesser.
    Max,
}

impl BinOp {
    /// Whether the operation on two integers can end the run: a trapping
    /// `+`, `-`, `*` or shift, and `/` and `%` (RFC-0037 rules 2 and 3).
    /// Such an operation moves only onto exactly the paths it ran on
    /// (RFC-0048 rule 8). On any other operand type no operation traps.
    pub fn can_trap_on_integers(self) -> bool {
        match self {
            BinOp::Add(overflow)
            | BinOp::Sub(overflow)
            | BinOp::Mul(overflow)
            | BinOp::Shl(overflow)
            | BinOp::Shr(overflow) => overflow == Overflow::Trap,
            BinOp::Div | BinOp::Mod => true,
            BinOp::Eq
            | BinOp::Neq
            | BinOp::Lt
            | BinOp::Gt
            | BinOp::Lte
            | BinOp::Gte
            | BinOp::And
            | BinOp::Or
            | BinOp::Xor
            | BinOp::BitAnd
            | BinOp::BitOr
            | BinOp::Min
            | BinOp::Max => false,
        }
    }
}

impl From<acvus_ast::BinOp> for BinOp {
    fn from(op: acvus_ast::BinOp) -> BinOp {
        match op {
            acvus_ast::BinOp::Add => BinOp::Add(Overflow::Trap),
            acvus_ast::BinOp::Sub => BinOp::Sub(Overflow::Trap),
            acvus_ast::BinOp::Mul => BinOp::Mul(Overflow::Trap),
            acvus_ast::BinOp::Div => BinOp::Div,
            acvus_ast::BinOp::Eq => BinOp::Eq,
            acvus_ast::BinOp::Neq => BinOp::Neq,
            acvus_ast::BinOp::Lt => BinOp::Lt,
            acvus_ast::BinOp::Gt => BinOp::Gt,
            acvus_ast::BinOp::Lte => BinOp::Lte,
            acvus_ast::BinOp::Gte => BinOp::Gte,
            acvus_ast::BinOp::And => BinOp::And,
            acvus_ast::BinOp::Or => BinOp::Or,
            acvus_ast::BinOp::Xor => BinOp::Xor,
            acvus_ast::BinOp::BitAnd => BinOp::BitAnd,
            acvus_ast::BinOp::BitOr => BinOp::BitOr,
            acvus_ast::BinOp::Shl => BinOp::Shl(Overflow::Trap),
            acvus_ast::BinOp::Shr => BinOp::Shr(Overflow::Trap),
            acvus_ast::BinOp::Mod => BinOp::Mod,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Checked {
    Add,
    Sub,
    Mul,
}

impl Checked {
    pub fn trapping_op(self) -> BinOp {
        match self {
            Checked::Add => BinOp::Add(Overflow::Trap),
            Checked::Sub => BinOp::Sub(Overflow::Trap),
            Checked::Mul => BinOp::Mul(Overflow::Trap),
        }
    }

    pub fn wrapping_op(self) -> BinOp {
        match self {
            Checked::Add => BinOp::Add(Overflow::Wrap),
            Checked::Sub => BinOp::Sub(Overflow::Wrap),
            Checked::Mul => BinOp::Mul(Overflow::Wrap),
        }
    }

    pub fn of_trapping(op: BinOp) -> Option<Checked> {
        match op {
            BinOp::Add(Overflow::Trap) => Some(Checked::Add),
            BinOp::Sub(Overflow::Trap) => Some(Checked::Sub),
            BinOp::Mul(Overflow::Trap) => Some(Checked::Mul),
            BinOp::Add(Overflow::Wrap)
            | BinOp::Sub(Overflow::Wrap)
            | BinOp::Mul(Overflow::Wrap)
            | BinOp::Div
            | BinOp::Eq
            | BinOp::Neq
            | BinOp::Lt
            | BinOp::Gt
            | BinOp::Lte
            | BinOp::Gte
            | BinOp::And
            | BinOp::Or
            | BinOp::Xor
            | BinOp::BitAnd
            | BinOp::BitOr
            | BinOp::Shl(_)
            | BinOp::Shr(_)
            | BinOp::Mod
            | BinOp::Min
            | BinOp::Max => None,
        }
    }
}

/// A one-operand operation of the MIR.
///
/// It is its own enum, and not `acvus_ast::UnaryOp`, for the reason
/// [`BinOp`] is: negation carries its [`Overflow`], which the parser does
/// not know, and a dereference is a `Take` in the MIR (RFC-0018), so the
/// enum has no variant for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// `-x`: a signed integer or `f64`. A trapping negation of the width's
    /// minimum ends the run.
    Neg(Overflow),
    /// `!b` on a `Bool`.
    Not,
}

/// Decision not to build, RFC-0051: no `Float` and no `Bytes` key. Equality
/// on a float is not a jump, and a byte string is a list. A `match` written
/// with such an arm keeps the chain `lower_match_expr` emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SwitchKey {
    Tag(Astr),
    Int(i128),
    Bool(bool),
    Char(char),
    Str(Astr),
}

impl SwitchKey {
    pub fn of_literal(literal: &Literal, interner: &Interner) -> Option<SwitchKey> {
        match literal.desugared() {
            Literal::Int(n) => Some(SwitchKey::Int(n)),
            Literal::Bool(b) => Some(SwitchKey::Bool(b)),
            Literal::Char(c) => Some(SwitchKey::Char(c)),
            Literal::String(s) => Some(SwitchKey::Str(interner.intern(&s))),
            Literal::Float(_)
            | Literal::Bytes(_)
            | Literal::IntOf(_)
            | Literal::List(_)
            | Literal::Unit => None,
        }
    }

    pub fn tag(self) -> Option<Astr> {
        match self {
            SwitchKey::Tag(tag) => Some(tag),
            _ => None,
        }
    }

    pub fn same_kind(self, other: SwitchKey) -> bool {
        std::mem::discriminant(&self) == std::mem::discriminant(&other)
    }

    pub fn shown(self, interner: &Interner) -> String {
        match self {
            SwitchKey::Tag(tag) => interner.resolve(tag).to_string(),
            SwitchKey::Int(n) => n.to_string(),
            SwitchKey::Bool(b) => b.to_string(),
            SwitchKey::Char(c) => format!("{c:?}"),
            SwitchKey::Str(text) => format!("{:?}", interner.resolve(text)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Label(pub u32);

#[derive(Debug, Clone)]
pub struct Inst {
    pub span: Span,
    pub kind: InstKind,
}

/// Type coercion kind - 1:1 with the subtyping rules in `try_coerce`.
#[derive(Debug, Clone, PartialEq)]
pub enum CastKind {
    /// Coercion performed by a registered pure ExternFn.
    Extern(ExternCast),
    /// At a call argument that borrows a place: `cast` runs on the place's
    /// value before the call, into a temporary the call borrows, and `back`
    /// on the temporary after it, stored back into the place (RFC-0041);
    /// the reference itself is not cast.
    ThroughRef {
        mutability: Mutability,
        cast: ExternCast,
        back: ExternCast,
    },
    /// At a call argument of `&C` whose parameter is `&[T]`: the container's
    /// own `as_slice` of the reference, which the lowering emits as an
    /// `AsSlice` and not a call (RFC-0047 rule 6).
    Slice {
        mutability: Mutability,
        as_slice: ExternCast,
    },
    /// At a call argument of `&String` whose parameter is `&str`: the
    /// `String`'s own `as_str` of the reference, which the lowering emits as
    /// an `AsSlice` and not a call (RFC-0062 rule 3).
    Str { as_str: ExternCast },
    /// A `&mut` where the `&` `shared` is taken: the shared reborrow `&r`
    /// gives (RFC-0029 rule 3).
    Reborrow { shared: Ty },
}

/// A cast function at the type of one call of it.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternCast {
    pub fn_ref: QualifiedRef,
    pub instance: usize,
    pub required: Vec<Chosen>,
    pub callee_ty: Ty,
}

/// The kind of named storage a Ref points to.
///
/// Var and Param are identified by a **storage ValueId** (like LLVM's alloca),
/// not by name. This ensures uniqueness after inlining - different functions'
/// local variables have different ValueIds even if they share the same name.
/// Names are stored in DebugInfo for human readability.
/// One step of a path under a storage (RFC-0024).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PathSeg {
    Field(Astr),
    Index(usize),
    Payload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefTarget {
    /// A local variable, identified by its storage slot ValueId.
    Var(ValueId),
    /// An extern parameter, identified by its param_reg ValueId.
    Param(ValueId),
    /// The storage a reference value names (RFC-0018).
    Through(ValueId),
}

/// Target of a function call.
#[derive(Debug, Clone)]
pub enum Callee {
    /// A function with a body in the graph. Enables pre-fetch and inlining.
    Direct(QualifiedRef),
    /// An ExternFn at the instance the checker settled on (RFC-0040), with
    /// what that call's requirements settled on (RFC-0070 rule 3).
    Extern {
        id: QualifiedRef,
        instance: usize,
        required: Vec<Chosen>,
    },
    /// Runtime-determined callable (closure, variable holding a function).
    Indirect(ValueId),
}

impl Callee {
    /// The named function of a `Direct` or `Extern` callee.
    pub fn id(&self) -> QualifiedRef {
        match self {
            Self::Direct(id) | Self::Extern { id, .. } => *id,
            Self::Indirect(_) => unreachable!("an indirect callee has no name"),
        }
    }
}

/// One node of the instance tree the checker settled (RFC-0070 rule 3).
/// `instance` numbers `signature`'s instances as the registry numbers its
/// handlers, and `required` stands one to one against the requirements the
/// chosen instance's declaration states, which is the order
/// `InstanceEntry::requires` is read in.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Chosen {
    pub signature: QualifiedRef,
    pub instance: usize,
    pub required: Vec<Chosen>,
}

/// An ExternFn at the instance the checker settled on (RFC-0040), for an
/// instruction that runs one without being a `FunctionCall`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternInstance {
    pub id: QualifiedRef,
    pub instance: usize,
}

impl From<ExternInstance> for Callee {
    fn from(ExternInstance { id, instance }: ExternInstance) -> Callee {
        Callee::Extern {
            id,
            instance,
            required: Vec::new(),
        }
    }
}

/// How an `Index` yields its element (RFC-0047 rule 4). The checker decides it
/// from the element type, so no run-time branch reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexMode {
    /// The element type is a word: `dst` is the element value itself.
    Copy,
    /// `dst` is a reference into the slice's storage, carrying its loan.
    Ref,
}

/// Whether an `Index` or `IndexSet` compares `index < len` when it runs
/// (RFC-0047 rule 7). The lowering writes `Checked` everywhere;
/// `optimize::bce` writes `Proven` where `analysis::interval` derives the
/// bound, and `validate::bounds` refuses a `Proven` it cannot derive again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexBound {
    Checked,
    Proven,
}

/// How one `a[i]` reaches its element: which slice the container gives up,
/// and how the element comes back (RFC-0047 rules 3 and 4). The checker settles
/// both; the lowering reads them and decides neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexAccess {
    pub mutability: crate::ty::Mutability,
    pub mode: IndexMode,
}

/// What a `for` traverses (RFC-0057 rule 1). Every source is settled
/// before the header runs: a slice pair, an array value, or two bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForSource {
    /// A `&[T]`: the element is a `&T`.
    Slice(ValueId),
    /// A `&mut [T]`: the element is a `&mut T`, and the container is held
    /// exclusively for the loop (RFC-0057 rule 5).
    SliceMut(ValueId),
    /// An `Array<T, N>` moved into the loop: the element is a `T` taken out
    /// of it, and the array is empty when the loop ends.
    Array(ValueId),
    /// `at..hi` of one integer width: the element is that width's value.
    Range { at: ValueId, hi: ValueId },
}

impl ForSource {
    pub fn uses(&self) -> smallvec::SmallVec<[ValueId; 2]> {
        match self {
            Self::Slice(v) | Self::SliceMut(v) | Self::Array(v) => smallvec::smallvec![*v],
            Self::Range { at, hi } => smallvec::smallvec![*at, *hi],
        }
    }

    pub fn uses_mut(&mut self) -> Vec<&mut ValueId> {
        match self {
            Self::Slice(v) | Self::SliceMut(v) | Self::Array(v) => vec![v],
            Self::Range { at, hi } => vec![at, hi],
        }
    }

    pub fn for_each_use(&mut self, mut f: impl FnMut(&mut ValueId)) {
        match self {
            Self::Slice(v) | Self::SliceMut(v) | Self::Array(v) => f(v),
            Self::Range { at, hi } => {
                f(at);
                f(hi);
            }
        }
    }

    /// How many of the body block's leading parameters the terminator fills
    /// itself; the carried values follow them.
    pub fn supplied_params(&self) -> usize {
        match self {
            Self::Range { .. } => 1,
            Self::Slice(_) | Self::SliceMut(_) | Self::Array(_) => 2,
        }
    }

    /// Which body parameter is the loop's counter, the induction variable
    /// the machine's `For` advances.
    pub fn counter_param(&self) -> usize {
        match self {
            Self::Range { .. } => 0,
            Self::Slice(_) | Self::SliceMut(_) | Self::Array(_) => 1,
        }
    }
}

/// Whether a `for`'s exit edge defines the loop's trip count (RFC-0057
/// rule 9). Where it does, the terminator fills the exit block's leading
/// parameter with it, as it fills the body block's leading parameters, and
/// `exit_args` are the carried values that follow. A pass sets it where it
/// reads the count, and nothing else does, so a loop no pass asked about
/// keeps the exit edge it was lowered with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitTrip {
    /// The exit block's parameters are `exit_args`, one for one.
    Absent,
    /// The exit block's first parameter is the number of times the body
    /// ran, a `u64`: `max(hi − at, 0)` for a range and the source's length
    /// for a slice or an array.
    Defined,
}

impl ExitTrip {
    /// How many of the exit block's leading parameters the terminator
    /// fills itself; the carried values follow them.
    pub fn supplied_params(self) -> usize {
        match self {
            Self::Absent => 0,
            Self::Defined => 1,
        }
    }

    /// The exit block's parameters that `exit_args` supplies.
    pub fn carried_params(self, exit_params: &[ValueId]) -> &[ValueId] {
        exit_params.get(self.supplied_params()..).unwrap_or(&[])
    }

    /// The exit block's parameter that holds the trip count, where the edge
    /// defines one.
    pub fn trip_param(self, exit_params: &[ValueId]) -> Option<ValueId> {
        match self {
            Self::Absent => None,
            Self::Defined => exit_params.first().copied(),
        }
    }
}

/// Where a `For`'s body is cut (RFC-0089 rule 1): the entry block of each
/// stage, in body order. The first is the body block, whose parameters are
/// the element and the counter. Each stage's last block jumps to the next
/// stage's entry and the last stage's jumps to the header. The chain names
/// no token: which stage serializes what, in which order and by which law,
/// is `analysis::loop_deps`'s to compute from the tokens and the
/// declarations, and `validate::stages` refuses a chain some dependence
/// cycle crosses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stages {
    body: Label,
    rest: Vec<Label>,
}

impl Stages {
    pub fn new(body: Label, rest: Vec<Label>) -> Self {
        Self { body, rest }
    }

    /// The chain a `for` is lowered with: one stage, the body.
    pub fn lowered(body: Label) -> Self {
        Self::new(body, Vec::new())
    }

    pub fn body(&self) -> Label {
        self.body
    }

    pub fn len(&self) -> usize {
        1 + self.rest.len()
    }

    pub fn entries(&self) -> impl Iterator<Item = Label> + '_ {
        std::iter::once(self.body).chain(self.rest.iter().copied())
    }

    pub fn entries_mut(&mut self) -> impl Iterator<Item = &mut Label> {
        std::iter::once(&mut self.body).chain(self.rest.iter_mut())
    }
}

/// Which of the four heads a `for` was written with (RFC-0057 rule 1).
/// The checker settles it from the head's type; the lowering reads it and
/// decides nothing, as it reads an [`IndexAccess`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForKind {
    Slice(Mutability),
    Array,
    Range,
}

/// The `Order` a call waits for and the `Order` it yields (RFC-0007).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrderEdge {
    pub before: ValueId,
    pub after: ValueId,
}

#[derive(Debug, Clone)]
pub enum InstKind {
    // Constants
    Const {
        dst: ValueId,
        value: Literal,
    },
    /// Obligation across artifacts: the bytes `dst` names are not in this
    /// instruction. `acvus-interpreter`'s preparation copies each distinct
    /// text of a module once and the module's prepared code owns that copy,
    /// so a `&str` constant is the pointer and length of a run whose owner
    /// is the code holding this operation (RFC-0062 rule 2).
    ConstStr {
        dst: ValueId,
        text: String,
    },
    /// A template's output: the parts joined into one `String`. A part is a
    /// `String`, moved in, or a `&String`, read through.
    StringConcat {
        dst: ValueId,
        parts: Vec<ValueId>,
    },
    /// A template's append: the bytes of `part` are written onto the end of
    /// the `String` that `target`, a `&mut String`, names. `part` is a
    /// `String`, moved in, or a `&str`/`&String`, read through, as a
    /// `StringConcat` part is. The accumulator is not copied, which is what
    /// keeps a loop body's appends linear in the text they write
    /// (RFC-0071).
    StringAppend {
        target: ValueId,
        part: ValueId,
    },
    /// Whether the two strings `a` and `b` name hold the same bytes.
    StringEq {
        dst: ValueId,
        a: ValueId,
        b: ValueId,
    },
    /// A new `String` with the bytes of the one `src` names.
    StringClone {
        dst: ValueId,
        src: ValueId,
    },
    /// RFC-0020. `leaves` is in `structural::structural_leaves` order over
    /// the type `a` lends, the order `acvus_interpreter::prepare` reads it in.
    StructuralEq {
        dst: ValueId,
        a: ValueId,
        b: ValueId,
        leaves: Vec<Chosen>,
    },
    /// RFC-0020. `leaves` is in `structural::structural_leaves` order over
    /// the type `src` lends, the order `acvus_interpreter::prepare` reads it
    /// in.
    StructuralClone {
        dst: ValueId,
        src: ValueId,
        leaves: Vec<Chosen>,
    },

    // -- Storage (RFC-0018) -----------------------------------------
    /// A reference to a storage: `dst` is a `&T` or `&mut T` naming the
    /// value at `path` under `target`. `path: vec![]` is the storage itself,
    /// `vec![a, b]` the field `a.b` of it.
    Ref {
        dst: ValueId,
        target: RefTarget,
        path: Vec<PathSeg>,
        mutability: Mutability,
    },
    /// Move the value out of a storage into `dst`; a primitive is copied
    /// and the storage keeps it. A read of a variable.
    Take {
        dst: ValueId,
        target: RefTarget,
        path: Vec<PathSeg>,
        /// The take-out of a place lent to a call through a conversion
        /// (RFC-0041): the place is taken out, a word's as any other's,
        /// until the `Assign` that restores it after the call.
        taken_out: bool,
    },
    /// Move `value` into a storage; the storage's old value is dropped. An
    /// assignment to a variable.
    Assign {
        target: RefTarget,
        path: Vec<PathSeg>,
        value: ValueId,
        /// The store that puts back a place a `Take` took out for a call.
        restores: bool,
    },
    // -- Slices (RFC-0047) ------------------------------------------
    /// Take the whole run of `container`'s elements: `dst` is a
    /// `Ref(mutability, Slice(T))` holding the loan `container` holds, and
    /// `instance` is the `core::as_slice` / `core::as_slice_mut` the
    /// container's own evidence settled on.
    ///
    /// It is not a `FunctionCall` because the kind itself states what the
    /// hoist must know: a pure, infallible borrow projection, so
    /// `code_motion` lifts a shared one out of every loop that does not
    /// write the container, which it may never do for a call.
    AsSlice {
        dst: ValueId,
        container: ValueId,
        mutability: Mutability,
        instance: ExternInstance,
    },
    /// Element `index` of `slice`. The index is `u64`; the one check is
    /// `index < len`, made unless `bound` is `Proven`.
    Index {
        dst: ValueId,
        slice: ValueId,
        index: ValueId,
        mode: IndexMode,
        bound: IndexBound,
    },
    /// Write `value` into element `index` of a `&mut [T]`, dropping the
    /// element that was there.
    IndexSet {
        slice: ValueId,
        index: ValueId,
        value: ValueId,
        bound: IndexBound,
    },

    /// Move a context's whole value out of the page into `dst` (RFC-0025).
    Fetch {
        dst: ValueId,
        context: QualifiedRef,
    },
    /// Move `value` into the page as the context's whole value (RFC-0025).
    /// `wrote` holds where the body may have written the variable since its
    /// fetch (RFC-0025 rules 2, 4); a commit that did not write hands the
    /// value back and stores nothing.
    Commit {
        context: QualifiedRef,
        value: ValueId,
        wrote: bool,
    },
    // -- Scalar field access --------------------------------------
    /// Extract a field from a scalar value. 1+ depth via `field` + `rest`.
    FieldGet {
        dst: ValueId,
        object: ValueId,
        field: Astr,
        rest: Vec<Astr>,
    },
    /// Replace a field in a scalar value, producing a new value. 1+ depth.
    FieldSet {
        dst: ValueId,
        object: ValueId,
        field: Astr,
        rest: Vec<Astr>,
        value: ValueId,
    },

    // Arithmetic / logic
    BinOp {
        dst: ValueId,
        op: BinOp,
        left: ValueId,
        right: ValueId,
    },
    UnaryOp {
        dst: ValueId,
        op: UnaryOp,
        operand: ValueId,
    },
    /// Ends the run with `op`'s trap where `left op right` does not fit the
    /// operands' width. A pass that rewrites a trapping operation's value
    /// into a computation that wraps writes one at the operation's place, so
    /// the operation's trap stays where it was (RFC-0037 rule 3).
    Check {
        op: Checked,
        left: ValueId,
        right: ValueId,
    },
    /// Ends the run with `+`'s trap where `from + count·step`, over the
    /// integers, does not fit the width `from` and `step` share; `count` is
    /// a `u64`. It is, once, the `Check { op: Add }` of a value that starts
    /// at `from` and advances by `step` on each of `count` iterations: that
    /// value is monotone, so every step fits exactly where the last one
    /// does. The removal of a loop whose only work was such checks writes
    /// it at the loop's place, so their trap stays on exactly the runs that
    /// had it (RFC-0088 rule 8).
    CheckSteps {
        from: ValueId,
        step: ValueId,
        count: ValueId,
    },
    /// `src as to` (RFC-0049), total, with Rust's `as` values.
    ///
    /// It carries no source type, and that is a decision: `src`'s own type
    /// is the one every reader already holds, and a second copy of it here
    /// could disagree with it.
    ///
    /// `CastKind` above is not reused because it is not this: it names a
    /// pure ExternFn performing a declared coercion between user-defined
    /// types (RFC-0023 rule 8), which is a call, not a width.
    Cast {
        dst: ValueId,
        src: ValueId,
        to: CastTy,
    },

    // Functions
    /// Load a graph-level function into a value (for passing as argument, storing, etc.)
    LoadFunction {
        dst: ValueId,
        id: QualifiedRef,
    },
    /// Unified function call. Callee can be a direct graph function or an indirect value.
    /// Semantically equivalent to Spawn + Eval (synchronous call = spawn then immediately eval).
    /// A call whose effect is not Pure carries an `OrderEdge`; a Pure call carries none.
    FunctionCall {
        dst: ValueId,
        callee: Callee,
        callee_ty: Ty,
        args: Vec<ValueId>,
        order: Option<OrderEdge>,
    },
    /// Issue a call and receive a Handle<T> for its result. The work starts
    /// here; `order` is the `Order` it waits for when the call is effectful.
    /// `dst` receives a Handle whose type carries the callee's return type.
    Spawn {
        dst: ValueId,
        callee: Callee,
        callee_ty: Ty,
        args: Vec<ValueId>,
        order: Option<ValueId>,
    },
    /// Evaluate (force) a Handle, consuming it.
    /// `src` must be a Handle<T>. `dst` receives T; `order` receives the
    /// `Order` that follows the call when the call is effectful.
    Eval {
        dst: ValueId,
        src: ValueId,
        order: Option<ValueId>,
    },
    /// Join orders: `dst` follows every order in `orders`. Associative and
    /// commutative; a value instruction, not control flow.
    Merge {
        dst: ValueId,
        orders: Vec<ValueId>,
    },

    // Composite constructors
    MakeArray {
        dst: ValueId,
        elements: Vec<ValueId>,
    },
    MakeObject {
        dst: ValueId,
        fields: Vec<(Astr, ValueId)>,
    },
    MakeTuple {
        dst: ValueId,
        elements: Vec<ValueId>,
    },
    TupleIndex {
        dst: ValueId,
        tuple: ValueId,
        index: usize,
    },

    // Pattern matching (decision tree)
    TestLiteral {
        dst: ValueId,
        src: ValueId,
        value: Literal,
    },
    TestObjectKey {
        dst: ValueId,
        src: ValueId,
        key: Astr,
    },
    ArrayIndex {
        dst: ValueId,
        array: ValueId,
        index: usize,
    },
    ObjectGet {
        dst: ValueId,
        object: ValueId,
        key: Astr,
    },

    // Closures
    MakeClosure {
        dst: ValueId,
        body: Label,
        captures: Vec<ValueId>,
    },

    // Variant (tagged union)
    MakeVariant {
        dst: ValueId,
        tag: Astr,
        payload: Option<ValueId>,
    },
    TestVariant {
        dst: ValueId,
        src: ValueId,
        tag: Astr,
    },
    UnwrapVariant {
        dst: ValueId,
        src: ValueId,
    },

    // Control flow
    BlockLabel {
        label: Label,
        params: Vec<ValueId>,
    },
    Jump {
        label: Label,
        args: Vec<ValueId>,
    },
    JumpIf {
        cond: ValueId,
        then_label: Label,
        then_args: Vec<ValueId>,
        else_label: Label,
        else_args: Vec<ValueId>,
    },
    /// `join` is the one label here that no edge carries, so a pass that
    /// rewrites the labels it reaches through edges misses it. Every such
    /// pass rewrites this one alongside — `graph::inliner::remap`,
    /// `cfg::promote` and `demote`, `optimize::forward::collapse`,
    /// `optimize::sroa::settle_joins` (RFC-0063 rule 4).
    Diamond {
        cond: ValueId,
        then_label: Label,
        then_args: Vec<ValueId>,
        else_label: Label,
        else_args: Vec<ValueId>,
        join: Label,
    },
    /// One dispatch (RFC-0051). `tag` is the value read -- once -- and
    /// `arms` the labels its keys take; `default` is the arm a key outside
    /// `arms` takes, and it is present exactly when the `match` had a
    /// catch-all.
    ///
    /// Every arm's key satisfies [`SwitchKey::same_kind`] with every other,
    /// because `tag`'s type decides which kind the source could write.
    Switch {
        tag: ValueId,
        arms: Vec<(SwitchKey, Label, Vec<ValueId>)>,
        default: Option<(Label, Vec<ValueId>)>,
    },
    /// One traversal (RFC-0057) whose body is cut into a chain of stages
    /// (RFC-0089 rule 1); `validate::stages` refuses a chain that is not
    /// rule 1's shape or that a dependence cycle crosses.
    For {
        source: ForSource,
        stages: Stages,
        exit: Label,
        exit_trip: ExitTrip,
        exit_args: Vec<ValueId>,
    },
    /// Leave the body with `value`; `order` is the `Order` the body yields
    /// last when its effect is not Pure.
    Return {
        value: ValueId,
        order: Option<ValueId>,
    },
    /// The body does not continue past here: the instruction before it
    /// produced a `!`, a call that traps instead of returning (RFC-0038).
    /// Never executed; a block it ends has no successor.
    Diverge,
    /// Undefined value - valid to move/copy, UB to read as a concrete value.
    /// Used as initial value for SSA variables that are defined inside loops
    /// (iteration bindings, write-only contexts).
    Undef {
        dst: ValueId,
    },
    Nop,

    /// Drop a value, releasing its resources.
    ///
    /// Inserted by the compiler at the end of a value's live range (last use
    /// or scope exit). No `dst` - Drop only consumes, never produces.
    ///
    /// At runtime, the interpreter calls the Owned vtable's drop function.
    /// Copy types (SBO) have a no-op drop. Boxed types invoke their destructor.
    Drop {
        src: ValueId,
    },

    /// Poison value: result of a compile-time error (e.g. undefined function).
    /// The typechecker already reported the error; this exists so the lowerer
    /// can continue without panicking. Must never be reached at runtime.
    Poison {
        dst: ValueId,
    },
}

pub struct TwoWay<'a> {
    pub cond: ValueId,
    pub then_label: Label,
    pub then_args: &'a [ValueId],
    pub else_label: Label,
    pub else_args: &'a [ValueId],
}

/// RFC-0063 rule 4 asks for one helper where a pass treats
/// [`InstKind::JumpIf`] and [`InstKind::Diamond`] alike. Most such passes do
/// not call it: the two variants' field names agree, so one or-pattern over
/// both is the whole arm, and that cannot drift from the enum the way a
/// helper's body can. What is left for this are the readers that destructure
/// the same instruction more than once, where the or-pattern would be
/// repeated.
pub fn two_way(kind: &InstKind) -> Option<TwoWay<'_>> {
    let (InstKind::JumpIf {
        cond,
        then_label,
        then_args,
        else_label,
        else_args,
    }
    | InstKind::Diamond {
        cond,
        then_label,
        then_args,
        else_label,
        else_args,
        ..
    }) = kind
    else {
        return None;
    };
    Some(TwoWay {
        cond: *cond,
        then_label: *then_label,
        then_args,
        else_label: *else_label,
        else_args,
    })
}

/// A block may end without a terminator and fall through to the block below
/// it — `cfg::Terminator::Fallthrough` is that block, and
/// `acvus_interpreter::prepare`'s `Edges::successors` reads the same IR the
/// same way.
fn successor_labels(insts: &[Inst], at: usize) -> Vec<Label> {
    for (offset, inst) in insts[at..].iter().enumerate() {
        match &inst.kind {
            InstKind::BlockLabel { label, .. } if offset > 0 => return vec![*label],
            InstKind::Jump { label, .. } => return vec![*label],
            InstKind::JumpIf {
                then_label,
                else_label,
                ..
            }
            | InstKind::Diamond {
                then_label,
                else_label,
                ..
            } => return vec![*then_label, *else_label],
            InstKind::Switch { arms, default, .. } => {
                return arms
                    .iter()
                    .map(|(_, label, _)| *label)
                    .chain(default.iter().map(|(label, _)| *label))
                    .collect();
            }
            InstKind::For { stages, exit, .. } => return vec![stages.body(), *exit],
            InstKind::Return { .. } | InstKind::Diverge => return Vec::new(),
            _ => {}
        }
    }
    Vec::new()
}

/// Whether control entering `from` reaches `join`, over the blocks `insts`
/// holds. `join` answers `true` without being one of them, which is what lets
/// `lower` ask this of an arm it has written before the join block exists.
pub fn reaches(insts: &[Inst], from: Label, join: Label) -> bool {
    let index: FxHashMap<Label, usize> = insts
        .iter()
        .enumerate()
        .filter_map(|(at, inst)| match &inst.kind {
            InstKind::BlockLabel { label, .. } => Some((*label, at)),
            _ => None,
        })
        .collect();
    let mut seen: FxHashSet<Label> = FxHashSet::default();
    let mut work = vec![from];
    while let Some(label) = work.pop() {
        if label == join {
            return true;
        }
        if !seen.insert(label) {
            continue;
        }
        if let Some(&at) = index.get(&label) {
            work.extend(successor_labels(insts, at));
        }
    }
    false
}

/// A two-way branch, and the block whose terminator it is.
pub struct Branch {
    pub at: usize,
    pub block: Label,
}

/// The branches of `insts` that `demoted` marks: the ones a pass wrote as a
/// `JumpIf` over a `Diamond` the lowering had written
/// (`cfg::demote_diamond`).
pub fn demoted_branches<'a>(
    insts: &'a [Inst],
    demoted: &'a FxHashSet<Label>,
) -> impl Iterator<Item = Branch> + 'a {
    let mut block = crate::cfg::ENTRY_LABEL;
    insts.iter().enumerate().filter_map(move |(at, inst)| {
        match &inst.kind {
            InstKind::BlockLabel { label, .. } => block = *label,
            InstKind::JumpIf { .. } if demoted.contains(&block) => {
                return Some(Branch { at, block });
            }
            _ => {}
        }
        None
    })
}

/// The first block after the branch at `at` that both its arms reach.
///
/// `lower`'s `close_diamond` decides whether to write a `Diamond` by asking
/// [`reaches`] of the arms it has just emitted, over the instructions between
/// the branch and the join. This asks that same question of a branch a pass
/// demoted; the two must stay one question, or a pass restores a terminator
/// the lowering would not have written.
pub fn meets_again(insts: &[Inst], at: usize) -> Option<Label> {
    let branch = two_way(&insts[at].kind)?;
    let (then_label, else_label) = (branch.then_label, branch.else_label);
    insts[at + 1..]
        .iter()
        .enumerate()
        .find_map(|(offset, inst)| {
            let InstKind::BlockLabel { label, .. } = &inst.kind else {
                return None;
            };
            let (label, arms) = (*label, &insts[at..=at + offset]);
            let meets = reaches(arms, then_label, label) && reaches(arms, else_label, label);
            meets.then_some(label)
        })
}

/// Debug info for a single Val: where it came from in source.
#[derive(Debug, Clone)]
pub enum ValOrigin {
    /// A named variable binding: `user`, `x`, `item`.
    Named(Astr),
    /// A context reference: `@name`.
    Context(Astr),
    /// An extern parameter: `$name`.
    ExternParam(Astr),
    /// A field access on a scalar value: `user.name` -- (object val, field name).
    Field(ValueId, Astr),
    /// A field projection on named storage: `@ctx.field`, `x.a.b`.
    RefField(RefTarget, Vec<PathSeg>),
    /// Result of a function call: `to_string(...)`, `fetch(...)`.
    Call(Astr),
    /// An intermediate/anonymous value (arithmetic, pattern test, etc.).
    Expr,
}

#[derive(Debug, Clone)]
pub struct DebugInfo {
    pub val_origins: FxHashMap<ValueId, ValOrigin>,
}

impl Default for DebugInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugInfo {
    pub fn new() -> Self {
        Self {
            val_origins: FxHashMap::default(),
        }
    }

    pub fn set(&mut self, val: ValueId, origin: ValOrigin) {
        self.val_origins.insert(val, origin);
    }

    pub fn get(&self, val: ValueId) -> Option<&ValOrigin> {
        self.val_origins.get(&val)
    }

    /// Human-readable label for a Val.
    pub fn label(&self, val: ValueId, interner: &Interner) -> String {
        match self.val_origins.get(&val) {
            Some(ValOrigin::Named(name)) => interner.resolve(*name).to_string(),
            Some(ValOrigin::Context(name)) => format!("@{}", interner.resolve(*name)),
            Some(ValOrigin::ExternParam(name)) => format!("${}", interner.resolve(*name)),
            Some(ValOrigin::Field(_, field)) => interner.resolve(*field).to_string(),
            Some(ValOrigin::RefField(target, path)) => {
                let base = match target {
                    RefTarget::Var(slot) => {
                        // Look up debug name from val_origins for the slot.
                        match self.val_origins.get(slot) {
                            Some(ValOrigin::Named(n)) => interner.resolve(*n).to_string(),
                            Some(ValOrigin::Context(n)) => format!("@{}", interner.resolve(*n)),
                            _ => format!("var_{}", slot.0),
                        }
                    }
                    RefTarget::Param(slot) => match self.val_origins.get(slot) {
                        Some(ValOrigin::ExternParam(n)) => format!("${}", interner.resolve(*n)),
                        _ => format!("$param_{}", slot.0),
                    },
                    RefTarget::Through(r) => format!("(*r{})", r.0),
                };
                let fields: Vec<String> = path
                    .iter()
                    .map(|seg| match seg {
                        PathSeg::Field(f) => interner.resolve(*f).to_string(),
                        PathSeg::Index(i) => format!("[{i}]"),
                        PathSeg::Payload => "payload".to_string(),
                    })
                    .collect();
                format!("{}.{}", base, fields.join("."))
            }
            Some(ValOrigin::Call(func)) => format!("{}(...)", interner.resolve(*func)),
            Some(ValOrigin::Expr) | None => format!("v{}", val.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MirBody {
    pub insts: Vec<Inst>,
    pub val_types: FxHashMap<ValueId, Ty>,
    /// Function parameters: (name, register). Register holds the initial SSA value.
    pub params: Vec<(Astr, ValueId)>,
    /// Captured variables: (name, register). Register holds the captured value.
    pub captures: Vec<(Astr, ValueId)>,
    /// The `Order` the body takes first when its effect is not Pure.
    pub order_param: Option<ValueId>,
    /// The join of the tasks of everything the body does (RFC-0046). The
    /// interpreter's `Code::may_suspend` is `task > Sync`.
    pub task: Task,
    pub debug: DebugInfo,
    pub val_factory: LocalFactory<ValueId>,
    pub label_count: u32,
    /// The blocks whose `Diamond` a pass demoted to a `JumpIf`
    /// (`cfg::demote_diamond`). `optimize::rejoin` restores the ones whose
    /// arms a later pass brought back together, and `validate` refuses one it
    /// left behind.
    pub demoted_diamonds: FxHashSet<Label>,
}

impl Default for MirBody {
    fn default() -> Self {
        Self::new()
    }
}

impl MirBody {
    pub fn new() -> Self {
        Self {
            insts: Vec::new(),
            val_types: FxHashMap::default(),
            params: Vec::new(),
            captures: Vec::new(),
            task: Task::Sync,
            order_param: None,
            debug: DebugInfo::new(),
            val_factory: LocalFactory::new(),
            label_count: 0,
            demoted_diamonds: FxHashSet::default(),
        }
    }

    /// Convenience: get param register ValueIds.
    pub fn param_regs(&self) -> Vec<ValueId> {
        self.params.iter().map(|(_, v)| *v).collect()
    }

    /// Convenience: get capture register ValueIds.
    pub fn capture_regs(&self) -> Vec<ValueId> {
        self.captures.iter().map(|(_, v)| *v).collect()
    }
}

#[derive(Debug, Clone)]
pub struct MirModule {
    /// How many of `main.params`, first, the function's declaration names:
    /// a call or a run passes them whether or not the body reads them. The
    /// rest are inputs the body reads (RFC-0071 rule 4), and one it no longer
    /// reads after the folds is not required and is no parameter
    /// (RFC-0071 rule 5).
    pub declared_params: usize,
    pub main: MirBody,
    pub closures: FxHashMap<Label, MirBody>,
    /// The `ret` of the graph `Function` this module is the body of; for the
    /// entry, what the host declared (RFC-0054).
    pub ret: Ty,
    /// What `main` joins into its outputs, as the checker inferred it from
    /// the body (RFC-0079 rule 5); a closure's are on the function type of
    /// the `MakeClosure` that makes it.
    pub flows: crate::ty::Flows,
}
