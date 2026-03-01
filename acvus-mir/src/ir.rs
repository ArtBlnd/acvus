use std::collections::HashMap;

use acvus_ast::{BinOp, Literal, RangeKind, Span, UnaryOp};

use crate::ty::Ty;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(pub u32);

#[derive(Debug, Clone)]
pub struct Inst {
    pub span: Span,
    pub kind: InstKind,
}

#[derive(Debug, Clone)]
pub enum InstKind {
    // Output — index into MirModule::texts
    EmitText(usize),
    EmitValue(ValueId),

    // Constants / variables
    Const { dst: ValueId, value: Literal },
    StorageLoad { dst: ValueId, name: String },
    StorageStore { name: String, src: ValueId },

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
        field: String,
    },

    // Calls
    Call {
        dst: ValueId,
        func: String,
        args: Vec<ValueId>,
    },
    AsyncCall {
        dst: ValueId,
        func: String,
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
        fields: Vec<(String, ValueId)>,
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
        key: String,
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
    ListSlice {
        dst: ValueId,
        list: ValueId,
        skip_head: usize,
        skip_tail: usize,
    },
    ObjectGet {
        dst: ValueId,
        object: ValueId,
        key: String,
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
    Named(String),
    /// A storage reference: `$name`.
    Storage(String),
    /// A field access: `user.name` — (object val, field name).
    Field(ValueId, String),
    /// Result of a function call: `to_string(...)`, `fetch(...)`.
    Call(String),
    /// An intermediate/anonymous value (arithmetic, pattern test, etc.).
    Expr,
}

#[derive(Debug, Clone)]
pub struct DebugInfo {
    pub val_origins: HashMap<ValueId, ValOrigin>,
}

impl DebugInfo {
    pub fn new() -> Self {
        Self {
            val_origins: HashMap::new(),
        }
    }

    pub fn set(&mut self, val: ValueId, origin: ValOrigin) {
        self.val_origins.insert(val, origin);
    }

    pub fn get(&self, val: ValueId) -> Option<&ValOrigin> {
        self.val_origins.get(&val)
    }

    /// Human-readable label for a Val.
    pub fn label(&self, val: ValueId) -> String {
        match self.val_origins.get(&val) {
            Some(ValOrigin::Named(name)) => name.clone(),
            Some(ValOrigin::Storage(name)) => format!("${name}"),
            Some(ValOrigin::Field(_, field)) => field.clone(),
            Some(ValOrigin::Call(func)) => format!("{func}(...)"),
            Some(ValOrigin::Expr) | None => format!("v{}", val.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MirBody {
    pub insts: Vec<Inst>,
    pub val_types: HashMap<ValueId, Ty>,
    pub debug: DebugInfo,
    pub val_count: u32,
    pub label_count: u32,
}

impl MirBody {
    pub fn new() -> Self {
        Self {
            insts: Vec::new(),
            val_types: HashMap::new(),
            debug: DebugInfo::new(),
            val_count: 0,
            label_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClosureBody {
    pub capture_names: Vec<String>,
    pub param_names: Vec<String>,
    pub body: MirBody,
}

#[derive(Debug, Clone)]
pub struct MirModule {
    pub main: MirBody,
    pub closures: HashMap<Label, ClosureBody>,
    pub texts: Vec<String>,
}
