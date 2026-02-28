use std::collections::HashMap;

use acvus_ast::{BinOp, Literal, RangeKind, Span, UnaryOp};

use crate::ty::Ty;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Val(pub u32);

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
    EmitValue(Val),

    // Constants / variables
    Const { dst: Val, value: Literal },
    StorageLoad { dst: Val, name: String },
    StorageStore { name: String, src: Val },

    // Arithmetic / logic
    BinOp {
        dst: Val,
        op: BinOp,
        left: Val,
        right: Val,
    },
    UnaryOp {
        dst: Val,
        op: UnaryOp,
        operand: Val,
    },
    FieldGet {
        dst: Val,
        object: Val,
        field: String,
    },

    // Calls
    Call {
        dst: Val,
        func: String,
        args: Vec<Val>,
    },
    AsyncCall {
        dst: Val,
        func: String,
        args: Vec<Val>,
    },
    Await {
        dst: Val,
        src: Val,
    },

    // Composite constructors
    MakeList {
        dst: Val,
        elements: Vec<Val>,
    },
    MakeObject {
        dst: Val,
        fields: Vec<(String, Val)>,
    },
    MakeRange {
        dst: Val,
        start: Val,
        end: Val,
        kind: RangeKind,
    },

    // Pattern matching (decision tree)
    TestLiteral {
        dst: Val,
        src: Val,
        value: Literal,
    },
    TestListLen {
        dst: Val,
        src: Val,
        min_len: usize,
        exact: bool,
    },
    TestObjectKey {
        dst: Val,
        src: Val,
        key: String,
    },
    TestRange {
        dst: Val,
        src: Val,
        start: i64,
        end: i64,
        kind: RangeKind,
    },
    ListIndex {
        dst: Val,
        list: Val,
        index: i32,
    },
    ListSlice {
        dst: Val,
        list: Val,
        skip_head: usize,
        skip_tail: usize,
    },
    ObjectGet {
        dst: Val,
        object: Val,
        key: String,
    },

    // Closures
    MakeClosure {
        dst: Val,
        body: Label,
        captures: Vec<Val>,
    },
    CallClosure {
        dst: Val,
        closure: Val,
        args: Vec<Val>,
    },

    // Iteration
    IterInit {
        dst: Val,
        src: Val,
    },
    IterNext {
        dst_value: Val,
        dst_done: Val,
        iter: Val,
    },

    // Control flow
    BlockLabel {
        label: Label,
        params: Vec<Val>,
    },
    Jump {
        label: Label,
        args: Vec<Val>,
    },
    JumpIf {
        cond: Val,
        then_label: Label,
        then_args: Vec<Val>,
        else_label: Label,
        else_args: Vec<Val>,
    },
    Return(Val),
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
    Field(Val, String),
    /// Result of a function call: `to_string(...)`, `fetch(...)`.
    Call(String),
    /// An intermediate/anonymous value (arithmetic, pattern test, etc.).
    Expr,
}

#[derive(Debug, Clone)]
pub struct DebugInfo {
    pub val_origins: HashMap<Val, ValOrigin>,
}

impl DebugInfo {
    pub fn new() -> Self {
        Self {
            val_origins: HashMap::new(),
        }
    }

    pub fn set(&mut self, val: Val, origin: ValOrigin) {
        self.val_origins.insert(val, origin);
    }

    pub fn get(&self, val: Val) -> Option<&ValOrigin> {
        self.val_origins.get(&val)
    }

    /// Human-readable label for a Val.
    pub fn label(&self, val: Val) -> String {
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
    pub val_types: HashMap<Val, Ty>,
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
