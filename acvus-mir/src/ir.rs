use std::sync::Arc;

use acvus_ast::{BinOp, Literal, RangeKind, Span, UnaryOp};
use acvus_utils::LocalFactory;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

use crate::graph::{ContextId, FunctionId};
use crate::ty::{Effect, Ty};

acvus_utils::declare_local_id!(pub ValueId);

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
                Ty::Iterator(elem.clone(), Effect::pure())
            }
            (CastKind::DequeToIterator, Ty::Deque(elem, _)) => {
                Ty::Iterator(elem.clone(), Effect::pure())
            }
            (CastKind::DequeToSequence, Ty::Deque(elem, o)) => {
                Ty::Sequence(elem.clone(), *o, Effect::pure())
            }
            (CastKind::SequenceToIterator, Ty::Sequence(elem, _, e)) => {
                Ty::Iterator(elem.clone(), e.clone())
            }
            (CastKind::RangeToIterator, Ty::Range) => {
                Ty::Iterator(Box::new(Ty::Int), Effect::pure())
            }
            _ => panic!("CastKind::result_ty: {self:?} incompatible with {src_ty:?}"),
        }
    }
}

/// Target of a function call.
#[derive(Debug, Clone)]
pub enum Callee {
    /// Compile-time known function. Enables pre-fetch and inlining.
    Direct(FunctionId),
    /// Runtime-determined callable (closure, variable holding a function).
    Indirect(ValueId),
}

#[derive(Debug, Clone)]
pub enum InstKind {
    // Constants / variables
    Const {
        dst: ValueId,
        value: Literal,
    },
    /// Create a projection handle to a context. This is a Ref (path),
    /// not a value — use ContextLoad to materialize the value (copy).
    /// `ty` is the context's root type (like alloca's type in LLVM).
    ContextProject {
        dst: ValueId,
        id: ContextId,
        ty: Ty,
    },
    /// Materialize a projection into a value (copy). Severs the store connection.
    /// `src` must be a projection (from ContextProject or FieldAccess on a projection).
    ContextLoad {
        dst: ValueId,
        src: ValueId,
    },
    /// Write a value through a projection.
    /// `dst` must be a projection (from ContextProject or FieldAccess on a projection).
    ContextStore {
        dst: ValueId,
        value: ValueId,
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

    // Functions
    /// Load a graph-level function into a value (for passing as argument, storing, etc.)
    LoadFunction {
        dst: ValueId,
        id: FunctionId,
    },
    /// Unified function call. Callee can be a direct graph function or an indirect value.
    /// Semantically equivalent to Spawn + Eval (synchronous call = spawn then immediately eval).
    FunctionCall {
        dst: ValueId,
        callee: Callee,
        args: Vec<ValueId>,
    },
    /// Spawn a deferred computation. Creates a Handle<T, E> without executing.
    /// Pure instruction — no side effects. The actual execution happens at Eval.
    /// `dst` receives a Handle whose type carries the callee's return type and effect.
    /// `context_uses` binds context SSA values that the callee will read from.
    Spawn {
        dst: ValueId,
        callee: Callee,
        args: Vec<ValueId>,
        context_uses: Vec<(ContextId, ValueId)>,
    },
    /// Evaluate (force) a Handle, consuming it. This is where effects actually occur.
    /// `src` must be a Handle<T, E>. `dst` receives T. Effect E happens here.
    /// `context_defs` captures new SSA values for contexts the callee wrote.
    Eval {
        dst: ValueId,
        src: ValueId,
        context_defs: Vec<(ContextId, ValueId)>,
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

    /// Pull one element from an iterator (SSA-clean).
    ///
    /// Consumes `iter_src`, produces element in `dst` and rest iterator in `iter_dst`.
    /// If exhausted, jumps to `done` with `done_args`. Replaces the old
    /// IterStep + TestVariant + JumpIf + UnwrapVariant + TupleIndex chain.
    IterStep {
        dst: ValueId,
        iter_src: ValueId,
        iter_dst: ValueId,
        done: Label,
        done_args: Vec<ValueId>,
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
    pub param_regs: Vec<ValueId>,
    pub capture_regs: Vec<ValueId>,
    pub debug: DebugInfo,
    pub val_factory: LocalFactory<ValueId>,
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
            param_regs: Vec::new(),
            capture_regs: Vec::new(),
            debug: DebugInfo::new(),
            val_factory: LocalFactory::new(),
            label_count: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MirModule {
    pub main: MirBody,
    pub closures: FxHashMap<Label, Arc<MirBody>>,
}

impl MirModule {
    /// Extract all context keys (ContextProject ids) referenced by this module.
    pub fn extract_context_keys(&self) -> FxHashSet<ContextId> {
        let mut keys = FxHashSet::default();
        for inst in &self.main.insts {
            if let InstKind::ContextProject { id, .. } = &inst.kind {
                keys.insert(*id);
            }
        }
        for closure in self.closures.values() {
            for inst in &closure.insts {
                if let InstKind::ContextProject { id, .. } = &inst.kind {
                    keys.insert(*id);
                }
            }
        }
        keys
    }
}
