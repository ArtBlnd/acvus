use acvus_ast::{BinOp, Literal, Span, UnaryOp};
use acvus_utils::LocalFactory;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::graph::QualifiedRef;
use crate::ty::{Mutability, Task, Ty};

/// A call that is an instruction of the language (RFC-0020): the
/// compiler's own instance of a shared signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Intrinsic {
    StringClone,
}

acvus_utils::declare_local_id!(pub ValueId);

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
    /// ExternCast - coercion performed by a registered pure ExternFn.
    /// `callee_ty` is the full Fn type of the cast function at this call site.
    Extern {
        fn_ref: QualifiedRef,
        instance: usize,
        callee_ty: Ty,
    },
    /// At a call argument that borrows a place: `cast` runs on the place's
    /// value before the call and `back` on it after, each stored back into
    /// the place; the reference itself is not cast.
    ThroughRef {
        mutability: Mutability,
        cast: ExternCast,
        back: ExternCast,
    },
}

/// A cast function at the type of one call of it.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternCast {
    pub fn_ref: QualifiedRef,
    pub instance: usize,
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
#[derive(Debug, Clone, Copy)]
pub enum Callee {
    /// A function with a body in the graph. Enables pre-fetch and inlining.
    Direct(QualifiedRef),
    /// An ExternFn at the instance the checker settled on (RFC-0040).
    Extern { id: QualifiedRef, instance: usize },
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

/// An ExternFn at the instance the checker settled on (RFC-0040), for an
/// instruction that runs one without being a `FunctionCall`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternInstance {
    pub id: QualifiedRef,
    pub instance: usize,
}

impl From<ExternInstance> for Callee {
    fn from(ExternInstance { id, instance }: ExternInstance) -> Callee {
        Callee::Extern { id, instance }
    }
}

/// How an `Index` yields its element (RFC-0047 §4). The checker decides it
/// from the element type, so no run-time branch reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexMode {
    /// The element type is a word: `dst` is the element value itself.
    Copy,
    /// `dst` is a reference into the slice's storage, carrying its loan.
    Ref,
}

/// How one `a[i]` reaches its element: which slice the container gives up,
/// and how the element comes back (RFC-0047 §3, §4). The checker settles
/// both; the lowering reads them and decides neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexAccess {
    pub mutability: crate::ty::Mutability,
    pub mode: IndexMode,
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
    /// A template's output: the parts joined into one `String`. A part is a
    /// `String`, moved in, or a `&String`, read through.
    StringConcat {
        dst: ValueId,
        parts: Vec<ValueId>,
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
    },
    /// Move `value` into a storage; the storage's old value is dropped. An
    /// assignment to a variable.
    Assign {
        target: RefTarget,
        path: Vec<PathSeg>,
        value: ValueId,
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
    /// `index < len`, and every `Index` the MIR holds is checked.
    Index {
        dst: ValueId,
        slice: ValueId,
        index: ValueId,
        mode: IndexMode,
    },
    /// Write `value` into element `index` of a `&mut [T]`, dropping the
    /// element that was there.
    IndexSet {
        slice: ValueId,
        index: ValueId,
        value: ValueId,
    },

    /// Move a context's whole value out of the page into `dst` (RFC-0025).
    Fetch {
        dst: ValueId,
        context: QualifiedRef,
    },
    /// Move `value` into the page as the context's whole value (RFC-0025).
    Commit {
        context: QualifiedRef,
        value: ValueId,
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
