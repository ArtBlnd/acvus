use std::sync::Arc;

use acvus_ast::{BinOp, Literal, RangeKind, Span, UnaryOp};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::builtins::BuiltinId;
use crate::ty::{Effect, Ty};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(pub u32);

#[derive(Debug, Clone)]
pub struct Inst {
    pub span: Span,
    pub kind: InstKind,
}

/// Type coercion kind — 1:1 with the subtyping rules in `try_coerce`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    /// `Deque<T, O> → List<T>` — origin erased, container preserved.
    DequeToList,
    /// `List<T> → Iterator<T, Pure>` — lazy iteration.
    ListToIterator,
    /// `Deque<T, O> → Iterator<T, Pure>` — direct consumption.
    DequeToIterator,
    /// `Deque<T, O> → Sequence<T, O, Pure>` — origin preserved.
    DequeToSequence,
    /// `Sequence<T, O, E> → Iterator<T, E>` — origin erased.
    SequenceToIterator,
    /// `Range → Iterator<Int, Pure>` — materialise range into lazy iterator.
    RangeToIterator,
}

impl CastKind {
    /// Determine the CastKind needed when a value of type `from` flows into
    /// a position expecting type `to`. Returns `None` if no cast is needed
    /// (types are compatible without coercion).
    pub fn between(from: &Ty, to: &Ty) -> Option<CastKind> {
        match (from, to) {
            (Ty::Deque(..), Ty::List(_)) => Some(CastKind::DequeToList),
            (Ty::List(_), Ty::Iterator(..)) => Some(CastKind::ListToIterator),
            (Ty::Deque(..), Ty::Iterator(..)) => Some(CastKind::DequeToIterator),
            (Ty::Deque(..), Ty::Sequence(..)) => Some(CastKind::DequeToSequence),
            (Ty::Sequence(..), Ty::Iterator(..)) => Some(CastKind::SequenceToIterator),
            (Ty::Range, Ty::Iterator(..)) => Some(CastKind::RangeToIterator),
            _ => None,
        }
    }

    /// Compute the result type of applying this cast to a value of the given
    /// source type. Panics if `src_ty` doesn't match the expected source
    /// constructor for this CastKind.
    pub fn result_ty(&self, src_ty: &Ty) -> Ty {
        match (self, src_ty) {
            (CastKind::DequeToList, Ty::Deque(elem, _)) => Ty::List(elem.clone()),
            (CastKind::ListToIterator, Ty::List(elem)) => {
                Ty::Iterator(elem.clone(), Effect::Pure)
            }
            (CastKind::DequeToIterator, Ty::Deque(elem, _)) => {
                Ty::Iterator(elem.clone(), Effect::Pure)
            }
            (CastKind::DequeToSequence, Ty::Deque(elem, o)) => {
                Ty::Sequence(elem.clone(), *o, Effect::Pure)
            }
            (CastKind::SequenceToIterator, Ty::Sequence(elem, _, e)) => {
                Ty::Iterator(elem.clone(), *e)
            }
            (CastKind::RangeToIterator, Ty::Range) => {
                Ty::Iterator(Box::new(Ty::Int), Effect::Pure)
            }
            _ => panic!("CastKind::result_ty: {self:?} incompatible with {src_ty:?}"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum InstKind {
    // Output
    Yield(ValueId),

    // Constants / variables
    Const {
        dst: ValueId,
        value: Literal,
    },
    ContextLoad {
        dst: ValueId,
        name: Astr,
    },
    VarLoad {
        dst: ValueId,
        name: Astr,
    },
    VarStore {
        name: Astr,
        src: ValueId,
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
    FieldGet {
        dst: ValueId,
        object: ValueId,
        field: Astr,
    },

    // Calls
    BuiltinCall {
        dst: ValueId,
        builtin: BuiltinId,
        args: Vec<ValueId>,
    },
    ExternCall {
        dst: ValueId,
        name: Astr,
        args: Vec<ValueId>,
    },

    // Composite constructors
    MakeDeque {
        dst: ValueId,
        elements: Vec<ValueId>,
    },
    MakeObject {
        dst: ValueId,
        fields: Vec<(Astr, ValueId)>,
    },
    MakeRange {
        dst: ValueId,
        start: ValueId,
        end: ValueId,
        kind: RangeKind,
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
    TestListLen {
        dst: ValueId,
        src: ValueId,
        min_len: usize,
        exact: bool,
    },
    TestObjectKey {
        dst: ValueId,
        src: ValueId,
        key: Astr,
    },
    TestRange {
        dst: ValueId,
        src: ValueId,
        start: i64,
        end: i64,
        kind: RangeKind,
    },
    ListIndex {
        dst: ValueId,
        list: ValueId,
        index: i32,
    },
    ListGet {
        dst: ValueId,
        list: ValueId,
        index: ValueId,
    },
    ListSlice {
        dst: ValueId,
        list: ValueId,
        skip_head: usize,
        skip_tail: usize,
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
    ClosureCall {
        dst: ValueId,
        closure: ValueId,
        args: Vec<ValueId>,
    },

    /// Pull one element from an Iterator.
    ///
    /// `src` must be an `Iterator<T, E>` value.
    /// `dst` receives `Option<(T, Iterator<T, E>)>`:
    /// - `Some((item, rest))` if an element is available
    /// - `None` if exhausted
    ///
    /// This is the language-level iteration primitive. The user-facing `next()`
    /// builtin delegates to `ExecCtx::exec_next`, which is the same underlying
    /// operation, but for-loops use this instruction directly — no builtin dependency.
    IterStep {
        dst: ValueId,
        src: ValueId,
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
        /// If set, this block is the merge point of a match expression.
        /// The label points to the first arm's test block, whose reachability
        /// the merge point should inherit (the match structure guarantees
        /// that exactly one arm executes and jumps here).
        merge_of: Option<Label>,
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
    Return(ValueId),
    Nop,

    /// Explicit type coercion — inserted by the lowerer when the type checker
    /// determines a subtype cast is needed (e.g. `Deque → List`).
    Cast {
        dst: ValueId,
        src: ValueId,
        kind: CastKind,
    },

    /// Poison value: result of a compile-time error (e.g. undefined function).
    /// The typechecker already reported the error; this exists so the lowerer
    /// can continue without panicking. Must never be reached at runtime.
    Poison {
        dst: ValueId,
    },
}

/// Debug info for a single Val: where it came from in source.
#[derive(Debug, Clone)]
pub enum ValOrigin {
    /// A named variable binding: `user`, `x`, `item`.
    Named(Astr),
    /// A context reference: `@name`.
    Context(Astr),
    /// A variable reference: `$name`.
    Variable(Astr),
    /// A field access: `user.name` -- (object val, field name).
    Field(ValueId, Astr),
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
            Some(ValOrigin::Variable(name)) => format!("${}", interner.resolve(*name)),
            Some(ValOrigin::Field(_, field)) => interner.resolve(*field).to_string(),
            Some(ValOrigin::Call(func)) => format!("{}(...)", interner.resolve(*func)),
            Some(ValOrigin::Expr) | None => format!("v{}", val.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MirBody {
    pub insts: Vec<Inst>,
    pub val_types: FxHashMap<ValueId, Ty>,
    pub debug: DebugInfo,
    pub val_count: u32,
    pub label_count: u32,
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
            debug: DebugInfo::new(),
            val_count: 0,
            label_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClosureBody {
    pub capture_names: Vec<Astr>,
    pub param_names: Vec<Astr>,
    pub body: MirBody,
}

#[derive(Debug, Clone, Default)]
pub struct MirModule {
    pub main: MirBody,
    pub closures: FxHashMap<Label, Arc<ClosureBody>>,
}
