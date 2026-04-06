use acvus_ast::{BinOp, Literal, RangeKind, Span, UnaryOp};
use acvus_utils::LocalFactory;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

use crate::graph::QualifiedRef;
use crate::ty::Ty;

acvus_utils::declare_local_id!(pub ValueId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Label(pub u32);

#[derive(Debug, Clone)]
pub struct Inst {
    pub span: Span,
    pub kind: InstKind,
}

/// Type coercion kind — 1:1 with the subtyping rules in `try_coerce`.
#[derive(Debug, Clone, PartialEq)]
pub enum CastKind {
    /// `Deque<T, O> → List<T>` — origin erased, container preserved.
    DequeToList,
    /// `Range → List<Int>` — materialise range into list.
    RangeToList,
    /// ExternCast — coercion performed by a registered pure ExternFn.
    /// `callee_ty` is the full Fn type of the cast function at this call site.
    Extern { fn_ref: QualifiedRef, callee_ty: Ty },
}

impl CastKind {
    /// Determine the CastKind needed when a value of type `from` flows into
    /// a position expecting type `to`. Returns `None` if no cast is needed
    /// (types are compatible without coercion).
    pub fn between(from: &Ty, to: &Ty) -> Option<CastKind> {
        match (from, to) {
            (Ty::Deque(..), Ty::List(_)) => Some(CastKind::DequeToList),
            _ => None,
        }
    }

    /// Compute the result type of applying this cast to a value of the given
    /// source type. Panics if `src_ty` doesn't match the expected source
    /// constructor for this CastKind.
    pub fn result_ty(&self, src_ty: &Ty) -> Ty {
        match (self, src_ty) {
            (CastKind::DequeToList, Ty::Deque(elem, _)) => Ty::List(elem.clone()),
            (CastKind::RangeToList, Ty::Range) => Ty::List(Box::new(Ty::Int)),
            _ => panic!("CastKind::result_ty: {self:?} incompatible with {src_ty:?}"),
        }
    }
}

/// The kind of named storage a Ref points to.
///
/// Var and Param are identified by a **storage ValueId** (like LLVM's alloca),
/// not by name. This ensures uniqueness after inlining — different functions'
/// local variables have different ValueIds even if they share the same name.
/// Names are stored in DebugInfo for human readability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefTarget {
    /// A local variable, identified by its storage slot ValueId.
    Var(ValueId),
    /// An extern parameter, identified by its param_reg ValueId.
    Param(ValueId),
    /// A context: `@name`. Globally unique by QualifiedRef.
    Context(QualifiedRef),
}

/// Target of a function call.
#[derive(Debug, Clone)]
pub enum Callee {
    /// Compile-time known function. Enables pre-fetch and inlining.
    Direct(QualifiedRef),
    /// Runtime-determined callable (closure, variable holding a function).
    Indirect(ValueId),
}

#[derive(Debug, Clone)]
pub enum InstKind {
    // Constants
    Const {
        dst: ValueId,
        value: Literal,
    },

    // ── Projection (memory world) ────────────────────────────────
    /// Create a projection to named storage. No-op at runtime — produces a path.
    /// `path: vec![]` = identity (root of the storage).
    /// `path: vec![f]` = 1-depth field projection.
    /// `path: vec![a, b]` = multi-depth field projection (a.b).
    Ref {
        dst: ValueId,
        target: RefTarget,
        path: Vec<Astr>,
    },
    /// Materialize a projection into a value (copy). `src` must be `Ref<T>`.
    /// `volatile`: if true, SSA must not elide or forward this load.
    Load {
        dst: ValueId,
        src: ValueId,
        volatile: bool,
    },
    /// Write a value through a projection. `dst` must be `Ref<T>`.
    /// `volatile`: if true, SSA must not elide this store.
    Store {
        dst: ValueId,
        value: ValueId,
        volatile: bool,
    },

    // ── Scalar field access ──────────────────────────────────────
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

    // Functions
    /// Load a graph-level function into a value (for passing as argument, storing, etc.)
    LoadFunction {
        dst: ValueId,
        id: QualifiedRef,
    },
    /// Unified function call. Callee can be a direct graph function or an indirect value.
    /// Semantically equivalent to Spawn + Eval (synchronous call = spawn then immediately eval).
    /// `context_uses` binds context SSA values the callee reads.
    /// `context_defs` captures new SSA values for contexts the callee writes.
    FunctionCall {
        dst: ValueId,
        callee: Callee,
        callee_ty: Ty,
        args: Vec<ValueId>,
        context_uses: Vec<(QualifiedRef, ValueId)>,
        context_defs: Vec<(QualifiedRef, ValueId)>,
    },
    /// Spawn a deferred computation. Creates a Handle<T, E> without executing.
    /// Pure instruction — no side effects. The actual execution happens at Eval.
    /// `dst` receives a Handle whose type carries the callee's return type and effect.
    /// `context_uses` binds context SSA values that the callee will read from.
    Spawn {
        dst: ValueId,
        callee: Callee,
        callee_ty: Ty,
        args: Vec<ValueId>,
        context_uses: Vec<(QualifiedRef, ValueId)>,
    },
    /// Evaluate (force) a Handle, consuming it. This is where effects actually occur.
    /// `src` must be a Handle<T, E>. `dst` receives T. Effect E happens here.
    /// `context_defs` captures new SSA values for contexts the callee wrote.
    Eval {
        dst: ValueId,
        src: ValueId,
        context_defs: Vec<(QualifiedRef, ValueId)>,
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

    /// Index-based iteration over List/Deque.
    ///
    /// If `index_src >= len(list)`, jumps to `done` with `done_args`.
    /// Otherwise, `dst = list[index_src]`, `index_dst = index_src + 1`.
    /// List is borrowed (not consumed). Index is a plain Int.
    ListStep {
        dst: ValueId,
        list: ValueId,
        index_src: ValueId,
        index_dst: ValueId,
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
    /// Undefined value — valid to move/copy, UB to read as a concrete value.
    /// Used as initial value for SSA variables that are defined inside loops
    /// (iteration bindings, write-only contexts).
    Undef {
        dst: ValueId,
    },
    Nop,

    /// Explicit type coercion — inserted by the lowerer when the type checker
    /// determines a subtype cast is needed (e.g. `Deque → List`).
    Cast {
        dst: ValueId,
        src: ValueId,
        kind: CastKind,
    },

    /// Clone a value. `dst` receives an independent copy of `src`.
    ///
    /// Emitted for explicit `clone(x)` calls. The compiler statically verifies
    /// that `src`'s type is cloneable (Copy or has a registered clone function).
    /// MoveOnly types → compile error, never reaches this instruction.
    ///
    /// At runtime, the interpreter calls `TryClone::try_clone` on the value.
    /// For Copy types this is a bitwise copy; for UserDefined types it invokes
    /// the registered clone function via the Owned vtable.
    Clone {
        dst: ValueId,
        src: ValueId,
    },

    /// Drop a value, releasing its resources.
    ///
    /// Inserted by the compiler at the end of a value's live range (last use
    /// or scope exit). No `dst` — Drop only consumes, never produces.
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
    RefField(RefTarget, Vec<Astr>),
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
                            _ => format!("var_{}", slot.0),
                        }
                    }
                    RefTarget::Param(slot) => {
                        match self.val_origins.get(slot) {
                            Some(ValOrigin::ExternParam(n)) => format!("${}", interner.resolve(*n)),
                            _ => format!("$param_{}", slot.0),
                        }
                    }
                    RefTarget::Context(qref) => format!("@{}", interner.resolve(qref.name)),
                };
                let fields: Vec<_> = path.iter().map(|f| interner.resolve(*f).to_string()).collect();
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
            params: Vec::new(),
            captures: Vec::new(),
            debug: DebugInfo::new(),
            val_factory: LocalFactory::new(),
            label_count: 0,
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

#[derive(Debug, Clone, Default)]
pub struct MirModule {
    pub main: MirBody,
    pub closures: FxHashMap<Label, MirBody>,
}

impl MirModule {
    /// Extract all context keys (Ref(Context) targets) referenced by this module.
    pub fn extract_context_keys(&self) -> FxHashSet<QualifiedRef> {
        let mut keys = FxHashSet::default();
        for inst in &self.main.insts {
            if let InstKind::Ref {
                target: RefTarget::Context(ctx),
                ..
            } = &inst.kind
            {
                keys.insert(*ctx);
            }
        }
        for closure in self.closures.values() {
            for inst in &closure.insts {
                if let InstKind::Ref {
                    target: RefTarget::Context(ctx),
                    ..
                } = &inst.kind
                {
                    keys.insert(*ctx);
                }
            }
        }
        keys
    }
}

