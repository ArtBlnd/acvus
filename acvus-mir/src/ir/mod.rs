use acvus_utils::Astr;
use rustc_hash::FxHashMap;

pub mod cfg;
pub(crate) mod parts;

use acvus_utils::QualifiedRef;
use crate::ty::{Concrete, Ty, Types};

/// An operation of the language no single op states, each read by
/// `Op::Intrinsic` over the arguments its doc lists. A backend that runs
/// one keeps its meaning as stated here, whatever layout it chooses
/// (undefined behaviour §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Intrinsic {
    StringClone,
    /// `[structure]`, a transparent structured value made of references to
    /// the parts of one value, each of one mutability: the reference to
    /// that value it is (ownership §5). Its loans are the part references'.
    Gather,
    /// `[a_1, …, a_k, f]`, `k` arrays of one length `n`, or references to
    /// them, and a function value of `k` parameters: the array of `n`
    /// elements, each `f` of the `k` elements at its index, in order. Where
    /// `k` is `0`, `n` is the length the result's type states. The elements
    /// are coerced or lifted one by one without unrolling the array
    /// (types §1, §3). An array is passed its element as it holds it; a
    /// reference to one, its element lent at the reference's mutability,
    /// or, where the element is itself a reference, that reference
    /// reborrowed at the weaker of the two, since a reference's target is
    /// never a reference (types §4, ownership §5).
    ArrayMap,
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
/// `BitOr` is written by no source either, since bitwise or has no
/// operator (operators §2); it stays because the interpreter consumes it
/// and the ledger may give it an operator later.
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
    /// `value << amount`. The amount is an integer of any width, its own;
    /// a trapping shift traps where the amount is negative or at least the
    /// value's width (Rust's `u8 << i64` included).
    Shl(Overflow),
    /// `value >> amount`, its amount and trap as `Shl`'s.
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
    pub fn same_kind(self, other: SwitchKey) -> bool {
        std::mem::discriminant(&self) == std::mem::discriminant(&other)
    }

}

/// The kind of named storage a Ref points to.
///
/// Var and Param are identified by a **storage ValueId** (like LLVM's alloca),
/// not by name. This ensures uniqueness after inlining - different functions'
/// local variables have different ValueIds even if they share the same name.
/// Names are stored in DebugInfo for human readability.
/// One step of a path under a storage (RFC-0024).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    /// A context, a place outside the body on the host's page, by its name
    /// (context §3). It aliases no other place. `Take` copies or moves it
    /// by its type, `Assign` stores into it as into any place, `Ref` lends
    /// it in place.
    Context(QualifiedRef),
}

/// Target of a function call.
#[derive(Debug, Clone)]
pub enum Callee {
    /// A unit's function in the graph, at its instantiation at the call's
    /// `callee_ty` (graph). Enables pre-fetch and inlining.
    Direct(QualifiedRef),
    /// An ExternFn at the impl the checker settled on (RFC-0040), none for
    /// a function that has no impls, with what that call's requirements
    /// settled on (RFC-0070 rule 3).
    Extern {
        id: QualifiedRef,
        r#impl: Option<usize>,
        required: Vec<Chosen>,
    },
    /// Runtime-determined callable (closure, variable holding a function).
    Indirect(ValueId),
}

impl Callee {
}

/// One node of the impl tree the checker settled (RFC-0070 rule 3).
/// `impl` numbers `generic_function`'s impls as the registry numbers its
/// handlers, and `required` stands one to one against the requirements the
/// chosen impl's declaration states, which is the order
/// `InstanceEntry::requires` is read in.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Chosen {
    pub generic_function: QualifiedRef,
    pub r#impl: usize,
    pub required: Vec<Chosen>,
}

/// An ExternFn at the impl the checker settled on (RFC-0040), none for a
/// function that has no impls, for an instruction that runs one without
/// being a `FunctionCall`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternImpl {
    pub id: QualifiedRef,
    pub r#impl: Option<usize>,
}

impl From<ExternImpl> for Callee {
    fn from(ExternImpl { id, r#impl }: ExternImpl) -> Callee {
        Callee::Extern {
            id,
            r#impl,
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
/// `passes::optimize::bce` makes one `Proven` through `Editor::discharge`,
/// on the fact that its check never fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexBound {
    Checked,
    Proven,
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

    /// The exit block's parameter that holds the trip count, where the edge
    /// defines one.
    pub fn trip_param(self, exit_params: &[ValueId]) -> Option<ValueId> {
        match self {
            Self::Absent => None,
            Self::Defined => exit_params.first().copied(),
        }
    }
}

/// The `Order` a call waits for and the `Order` it yields (RFC-0007).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrderEdge {
    pub before: ValueId,
    pub after: ValueId,
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

}

/// Whether a value of `ty` is a word (RFC-0018 rule 1): a type with no
/// parts held as one word, a reference, or an `Option` of a word.
///
/// The answer for an option is one half of a contract with the runtime's
/// representation of one, written in RFC-0039 and `docs/runtime-value.md`:
/// a host gives an option no storage of its own, so `Some(v)` is `v` and a
/// `None` is a word counting the `Some`s around it. A host that gave an
/// option a box would have to move this arm with it.
pub fn is_word(types: &Types<Concrete>, ty: Ty) -> bool {
    acvus_utils::grow(|| match types.kind(ty) {
        crate::ty::TyKind::Option(payload, _) => is_word(types, *payload),
        kind => is_scalar(kind) || matches!(kind, crate::ty::TyKind::Ref(..)),
    })
}

/// A type with no parts, held as one word.
pub fn is_scalar(kind: &crate::ty::TyKind<Concrete>) -> bool {
    use crate::ty::TyKind;
    match kind {
        TyKind::Int(_) | TyKind::Float | TyKind::Char | TyKind::Bool | TyKind::Unit | TyKind::Never | TyKind::Order => {
            true
        }
        TyKind::String
        | TyKind::Array(..)
        | TyKind::Object(_)
        | TyKind::Tuple(.., _)
        | TyKind::Option(.., _)
        | TyKind::Result(..)
        | TyKind::Fn { .. }
        | TyKind::UserDefined { .. }
        | TyKind::Enum { .. }
        | TyKind::Slice(_)
        | TyKind::Str
        | TyKind::Handle(_)
        | TyKind::Closures(_)
        | TyKind::Ref(..) => false,
        TyKind::Var(var) => match *var {},
    }
}

/// The effect of a concrete type's effect term, which holds no variable.
pub fn known_effect(effect: &crate::ty::EffectTerm<Concrete>) -> &crate::ty::Effect {
    match effect {
        crate::ty::EffectTerm::Known(effect) => effect,
        crate::ty::EffectTerm::Var(var) => match *var {},
    }
}

/// The length of a concrete type's length term, which holds no variable.
pub fn known_len(len: &crate::ty::LenTerm<Concrete>) -> usize {
    match len {
        crate::ty::LenTerm::Known(len) => *len,
        crate::ty::LenTerm::Var(var) => match *var {},
    }
}

/// The effect a value of `ty` has when called, where `ty` is a function
/// type.
pub fn fn_effect(types: &Types<Concrete>, ty: Ty) -> Option<&crate::ty::Effect> {
    match types.kind(ty) {
        crate::ty::TyKind::Fn { effect, .. } => Some(known_effect(effect)),
        _ => None,
    }
}
