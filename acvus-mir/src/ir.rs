use acvus_ast::{BinOp, Literal, RangeKind, Span, UnaryOp};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::builtins::BuiltinId;
use crate::extern_module::ExternFnId;
use crate::ty::Ty;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallTarget {
    Builtin(BuiltinId),
    Extern(ExternFnId),
}

#[derive(Debug, Clone)]
pub struct Inst {
    pub span: Span,
    pub kind: InstKind,
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
        bindings: Vec<(Astr, ValueId)>,
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
    Call {
        dst: ValueId,
        func: CallTarget,
        args: Vec<ValueId>,
    },
    AsyncCall {
        dst: ValueId,
        func: CallTarget,
        args: Vec<ValueId>,
    },
    Await {
        dst: ValueId,
        src: ValueId,
    },

    // Composite constructors
    MakeList {
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
    CallClosure {
        dst: ValueId,
        closure: ValueId,
        args: Vec<ValueId>,
    },

    // Iteration
    IterInit {
        dst: ValueId,
        src: ValueId,
    },
    IterNext {
        dst_value: ValueId,
        dst_done: ValueId,
        iter: ValueId,
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
    Return(ValueId),
    Nop,
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

#[derive(Debug, Clone)]
pub struct MirModule {
    pub main: MirBody,
    pub closures: FxHashMap<Label, ClosureBody>,
    /// Extern fn name table: `ExternFnId` -> name.
    pub extern_names: FxHashMap<ExternFnId, Astr>,
}
